from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hashmind.frozen.convert_ops import OpConv, parse_label, quantize_rows, quantize_weight, dequantize_weight
from hashmind.frozen.hashsrc import HashSource, fingerprint
from hashmind.frozen.hmfrozen import convert_frozen, file_sha256, load_frozen
from hashmind.frozen.metrics import generation_fidelity, logit_fidelity
from hashmind.frozen.runtime import FrozenRuntime, FrozenWeights
from hashmind.gguf import read_gguf
from hashmind.llm.llama import LlamaReference
from make_tiny_gguf import make_tiny_gguf


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_gguf(tmp_path_factory.mktemp("m5") / "tiny.gguf")


@pytest.fixture(scope="module")
def toks() -> np.ndarray:
    return np.stack([np.r_[1, (np.arange(11) * 7 + s) % 200 + 3] for s in range(3)])


def test_exact_runtime_equals_reference(tiny: Path, toks: np.ndarray) -> None:
    g = read_gguf(tiny)
    ref = LlamaReference(g)
    want = ref.logits(ref.forward(toks).final_normed)
    got = FrozenRuntime(FrozenWeights.from_gguf(g)).forward(toks)
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_left_padding_is_masked(tiny: Path, toks: np.ndarray) -> None:
    rt = FrozenRuntime(FrozenWeights.from_gguf(tiny))
    a = rt.forward(toks[:1, :6], last_only=True)
    b = rt.forward(np.r_[[0, 0], toks[0, :6]][None], pad=np.array([2]), last_only=True)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_hash_source_is_deterministic_and_content_keyed() -> None:
    x = np.random.default_rng(0).standard_normal((5, 40)).astype(np.float32)
    s = HashSource("sha256d")
    u1 = s.uniform(7, fingerprint(x), 16)
    u2 = HashSource("sha256d").uniform(7, fingerprint(x[::-1]), 16)[::-1]
    np.testing.assert_array_equal(u1, u2)  # row order / batch composition does not matter
    assert u1.min() >= 0 and u1.max() < 1 and s.evaluations == 80
    assert not np.array_equal(u1, HashSource("splitmix").uniform(7, fingerprint(x), 16))


def test_stochastic_rounding_is_unbiased() -> None:
    x = np.full((1, 20000), 0.3, np.float32)
    x[0, 0] = 1.0  # absmax -> scale 1/7 at 4 bits
    u = HashSource("splitmix").uniform(1, np.arange(1), 20000)
    q = quantize_rows(x, 4, u)
    assert abs(q[0, 1:].mean() - 0.3) < 0.005
    assert set(np.unique(q[0, 1:] * 7).round(4)) <= {2.0, 3.0}


def test_weight_quantization_roundtrip() -> None:
    W = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    c, s = quantize_weight(W, 8)
    assert np.abs(dequantize_weight(c, s) - W).max() < 0.03
    c2, s2 = quantize_weight(W, 2)
    assert set(np.unique(c2)) <= {-1, 0, 1}


@pytest.mark.parametrize("conv", [OpConv("quant", 8, "sha256d"), OpConv("sampled", samples=64),
                                  OpConv("topk", samples=64)])
def test_linear_conversions_approximate(tiny: Path, toks: np.ndarray, conv: OpConv) -> None:
    rt = FrozenRuntime(FrozenWeights.from_gguf(tiny), {"mlp_down": conv})
    rt.forward(toks, probe=True)
    assert rt.probe["mlp_down"].summary()["cosine"] > 0.6
    if conv.uses_hash:
        assert rt.hash_evaluations()["sha256d"] > 0


def test_sampled_matmul_unbiased() -> None:
    from hashmind.frozen.convert_ops import Converter

    rng = np.random.default_rng(1)
    W = rng.standard_normal((16, 128)).astype(np.float32)
    x = rng.standard_normal((400, 128)).astype(np.float32)
    cv = Converter()  # same input, 1600 different op keys -> independent hash-driven samples
    y = np.stack([cv.linear(x[:1], W, OpConv("sampled", samples=8, primitive="splitmix"), i, "t")[0]
                  for i in range(1600)])
    exact = x[0] @ W.T
    assert np.linalg.norm(y.mean(0) - exact) / np.linalg.norm(exact) < 0.15  # ~ per-sample error / sqrt(1600)


def test_frozen_conversion_reproducible_and_loadable(tiny: Path, toks: np.ndarray, tmp_path: Path) -> None:
    conv = {"mlp_in": OpConv("quant", 4, "sha256d"), "act": OpConv("lut", 8)}
    m1 = convert_frozen(tiny, tmp_path / "a.hmmodel", 4, "sha256d", conv)
    m2 = convert_frozen(tiny, tmp_path / "b.hmmodel", 4, "sha256d", conv)
    assert m1["hmmodel_sha256"] == m2["hmmodel_sha256"] == file_sha256(tmp_path / "b.hmmodel")
    assert m1["conversion"]["learned_parameters"] == 0 and m1["source"]["sha256"] == file_sha256(tiny)
    w, c, man = load_frozen(tmp_path / "a.hmmodel")
    assert c["mlp_in"] == conv["mlp_in"] and c["attn_qkv"].kind == "exact"
    assert np.isfinite(FrozenRuntime(w, c).forward(toks)).all()


def test_fidelity_metrics() -> None:
    a = np.random.default_rng(0).standard_normal((50, 100))
    f = logit_fidelity(a, a)
    assert f["top1_agreement"] == 1 and abs(f["kl_ref_to_conv"]) < 1e-9 and f["logit_spearman"] > 0.999
    g = logit_fidelity(a, -a)
    assert g["top1_agreement"] == 0 and g["logit_cosine"] < -0.99
    gf = generation_fidelity([[1, 2, 3], [4, 5]], [[1, 2, 9], [4, 5]])
    assert gf["exact_match_rate"] == 0.5 and gf["mean_matching_prefix"] == 2


def test_parse_label_roundtrip() -> None:
    for c in (OpConv(), OpConv("quant", 2, "splitmix"), OpConv("sampled", samples=9, primitive="sha256d"),
              OpConv("lut", 6, slots=32, primitive="sha256d"), OpConv("hashed", slots=10, primitive="pcg32")):
        assert parse_label(c.label()) == c
