from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hashmind.backends import SimulatedS9Backend
from hashmind.backends.s9_model import BM1387_MIN_DIFFICULTY_BITS, feature_cost
from hashmind.core import ChallengeConfig, HashMindLayer, HashMindNode, InputMapping
from hashmind.core.layer import supervised_thresholds
from hashmind.experiments.next_token import HiddenStates, HMConfig, NextTokenBench, run_hashmind, run_linear
from hashmind.gguf import read_gguf
from hashmind.llm.llama import LlamaReference, rope
from hashmind.llm.tokenizer import SPMTokenizer
from make_tiny_gguf import make_tiny_gguf


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_gguf(tmp_path_factory.mktemp("m") / "tiny.gguf")


# --- tokenizer ----------------------------------------------------------------

def _toy_tok() -> SPMTokenizer:
    toks = ["<unk>", "<s>", "</s>", "▁", "h", "e", "l", "o", "w", "r", "d",
            "▁h", "el", "lo", "▁hel", "▁hello", "▁w", "or", "▁wor", "ld", "▁world"]
    toks += [f"<0x{b:02X}>" for b in range(256)]
    scores = [0.0] * len(toks)
    for i, t in enumerate(toks):
        if len(t) > 1 and not t.startswith("<"):
            scores[i] = float(len(t))  # prefer longer merges
    return SPMTokenizer(toks, scores)


def test_spm_merges_and_roundtrip() -> None:
    tok = _toy_tok()
    ids = tok.encode("hello world")
    assert [tok.tokens[i] for i in ids] == ["<s>", "▁hello", "▁world"]
    assert tok.decode(ids) == "hello world"


def test_spm_byte_fallback() -> None:
    tok = _toy_tok()
    ids = tok.encode("hé", bos=False)
    assert tok.decode(ids) == "hé"
    assert any(tok.tokens[i].startswith("<0x") for i in ids)


# --- llama reference forward ---------------------------------------------------

def test_rope_preserves_norm_and_position0() -> None:
    x = np.random.default_rng(0).standard_normal((1, 2, 5, 8)).astype(np.float32)
    y = rope(x, 10000.0)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)
    np.testing.assert_allclose(y[:, :, 0], x[:, :, 0])


def test_forward_is_causal_and_shaped(tiny: Path) -> None:
    ref = LlamaReference(read_gguf(tiny))
    a = np.array([[1, 5, 9, 12, 40]])
    b = a.copy()
    b[0, -1] = 77
    ra, rb = ref.forward(a), ref.forward(b)
    assert ra.final_normed.shape == (1, 5, 64)
    assert set(ra.hidden) == set(range(ref.hp.n_layer + 1))
    np.testing.assert_allclose(ra.final_normed[0, :4], rb.final_normed[0, :4], atol=1e-5)
    assert not np.allclose(ra.final_normed[0, 4], rb.final_normed[0, 4])
    np.testing.assert_array_equal(ra.hidden[0][0], ref.embedding[a[0]])


# --- context + supervised quantizer -------------------------------------------------

def test_context_layer_requires_context_and_is_exact() -> None:
    X = np.zeros((6, 4), np.float32)
    ctx = np.array([[1, 9], [1, 9], [2, 9], [1, 8], [2, 8], [2, 8]])
    L = HashMindLayer(4, 64, seed=0, tuple_size=0, context_dim=2, context_per_node=2)
    L.fit(X)
    with pytest.raises(ValueError):
        L.transform(X)
    F = L.transform(X, ctx)
    assert np.array_equal(F[0], F[1]) and np.array_equal(F[4], F[5])
    assert not np.array_equal(F[0], F[2]) and not np.array_equal(F[0], F[3])


def test_context_free_layer_unchanged_by_phase3() -> None:
    """Context support uses a separate RNG stream: phase-2 wiring is untouched."""
    X = np.random.default_rng(0).standard_normal((20, 8)).astype(np.float32)
    L = HashMindLayer(8, 64, seed=7)
    assert L._indices == HashMindLayer(8, 64, seed=7, context_dim=3)._indices


def test_table_log2() -> None:
    L = HashMindLayer(32, 2048, seed=0, tuple_size=2, levels=4, nonces_per_node=16)
    assert L.table_log2() == pytest.approx(4 + np.log2(128 * 16))
    C = HashMindLayer(32, 2048, seed=0, tuple_size=0, context_dim=3, context_per_node=2)
    assert C.table_log2(32000) == pytest.approx(2 * np.log2(32000) + np.log2(128 * 16))


def test_supervised_thresholds_find_the_step() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2000, 2))
    y = (X[:, 0] > 0.3).astype(float)  # only dim 0 matters, step at 0.3
    t = supervised_thresholds(X, y, levels=2)
    assert abs(t[0, 0] - 0.3) < 0.05


# --- S9-realistic modes ---------------------------------------------------------------

@pytest.mark.parametrize("mode,kw", [
    ("threshold", {"difficulty_bits": 5, "window": 32}),
    ("nonce_bits", {"difficulty_bits": 5, "window": 32, "bits_per_hash": 4}),
])
def test_windowed_modes_match_backend(mode: str, kw: dict) -> None:
    m = InputMapping((0, 1), np.zeros((2, 1), np.float32))
    node = HashMindNode(99, m, ChallengeConfig(mode=mode, nonces=256, **kw))
    for c in ([0, 1], [1, 1], [1, 0]):
        codes = np.array(c, np.uint8)
        a = node.evaluate(codes)
        b = node.evaluate(codes, backend=SimulatedS9Backend())
        np.testing.assert_array_equal(a, b)
        assert a.shape == (node.n_features,)


def test_nonce_bits_layout() -> None:
    c = ChallengeConfig(mode="nonce_bits", nonces=16, window=8, bits_per_hash=3, difficulty_bits=3)
    p = np.zeros(16, bool)
    p[5] = p[7] = True  # window 0: first share at offset 5 = 0b101; window 1: none
    np.testing.assert_array_equal(c.aggregate(p), [1, 1, 0, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        ChallengeConfig(mode="nonce_bits", nonces=16, window=4, bits_per_hash=3)


def test_s9_cost_model() -> None:
    t = feature_cost("threshold")
    assert t.p_share_in_window == pytest.approx(1 - np.exp(-1))
    assert t.features_per_second == pytest.approx(13.5e12 / 2**32)
    n = feature_cost("nonce_bits", nonce_bits=16)
    assert n.features_per_second == pytest.approx(17 * t.features_per_second)
    with pytest.raises(ValueError):
        feature_cost("threshold", difficulty_bits=BM1387_MIN_DIFFICULTY_BITS - 1)


# --- next-token bench -------------------------------------------------------------------

def test_next_token_bench_on_tiny_model(tiny: Path) -> None:
    g = read_gguf(tiny)
    ref = LlamaReference(g)
    # strongly structured sequences so a readout has something to learn
    toks = np.stack([np.r_[1, (np.arange(23) * 7 + s) % 200 + 3] for s in range(16)])
    r = ref.forward(toks)
    head = ref.lm_head()
    hs = HiddenStates(toks, {L: v.astype(np.float16) for L, v in r.hidden.items()},
                      r.final_normed.astype(np.float16), (r.final_normed @ head.T).argmax(-1), 0.0)
    bench = NextTokenBench(hs, head, alphas=(1.0,))
    t = bench.teacher_row()
    assert t.teacher_agreement == 1.0
    lin = run_linear(bench, 2, 8)
    hm = run_hashmind(bench, 2, HMConfig("hm", output_dim=128), r=8, vocab=256)
    ctx = run_hashmind(bench, 0, HMConfig("ctx", tuple_size=0, context_cols=2, context_per_node=2,
                                          output_dim=128), r=8, vocab=256)
    for row in (lin, hm, ctx):
        assert 0.0 <= row.teacher_agreement <= 1.0
    assert hm.sha256d_logical > 0 and ctx.table_log2 > hm.table_log2
    assert len(bench.ngram_rows()) == 3
