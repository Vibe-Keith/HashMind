from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from hashcortex.architecture.config import HashCortexConfig
from hashcortex.architecture.model import HashCortexModel
from hashcortex.backends import HashJob, SimulatedS9Backend, meets_target, sha256d
from hashcortex.cli import main
from hashcortex.conversion.analysis import Disposition, analyze, classify_tensor
from hashcortex.conversion.convert import convert
from hashcortex.formats.hcmodel import HCModel, read_manifest
from hashcortex.gguf import inspect_gguf, read_gguf, write_gguf
from hashcortex.gguf.constants import GGMLType
from hashcortex.gguf.reader import dequantize
from make_tiny_gguf import make_tiny_gguf

SMALL = dict(proj_dim=16, reservoir_bits=16, n_tuples=32, tuple_bits=6, nonces_per_tuple=2)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_gguf(tmp_path_factory.mktemp("m") / "tiny.gguf")


# --- GGUF -----------------------------------------------------------------

def test_roundtrip_f32_f16_q8(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 64)).astype(np.float32)
    write_gguf(tmp_path / "x.gguf", {"general.architecture": "test", "k.arr": [1, 2, 3]},
               {"a": (a, GGMLType.F32), "b": (a, GGMLType.F16), "c": (a, GGMLType.Q8_0)})
    g = read_gguf(tmp_path / "x.gguf")
    assert g.metadata["k.arr"] == [1, 2, 3]
    assert g.tensors["a"].dims == (64, 4) and g.tensors["a"].shape == (4, 64)
    np.testing.assert_array_equal(g.tensor("a"), a)
    np.testing.assert_allclose(g.tensor("b"), a, atol=2e-3)
    np.testing.assert_allclose(g.tensor("c"), a, atol=np.abs(a).max() / 100)


def test_q4_0_dequant_known_block() -> None:
    d = np.float16(0.5).tobytes()
    qs = bytes([(15 << 4) | 0] * 16)  # low nibble 0 -> -8, high nibble 15 -> 7
    out = dequantize(np.frombuffer(d + qs, np.uint8), GGMLType.Q4_0, 32)
    assert np.all(out[:16] == -4.0) and np.all(out[16:] == 3.5)


def test_bad_magic(tmp_path: Path) -> None:
    (tmp_path / "bad.gguf").write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(Exception, match="magic"):
        read_gguf(tmp_path / "bad.gguf")


def test_inspector(tiny: Path) -> None:
    s = inspect_gguf(tiny)
    assert s.architecture == "llama"
    assert (s.embedding_dim, s.n_layers, s.ffn_dim, s.vocab_size) == (64, 2, 128, 256)
    assert (s.n_heads, s.head_dim) == (4, 16)
    assert s.dominant_quantization == "Q8_0"
    expected = 256 * 64 * 2 + 64 + 2 * (64 * 4 * 64 + 3 * 128 * 64 + 2 * 64)
    assert s.parameter_count == expected
    assert s.tokenizer["ggml.model"] == "llama"
    assert s.tokenizer["ggml.bos_token_id"] == 1


# --- analysis -------------------------------------------------------------

@pytest.mark.parametrize("name,disp", [
    ("token_embd.weight", Disposition.PRESERVED),
    ("output.weight", Disposition.PRESERVED),
    ("blk.3.attn_norm.weight", Disposition.HOST_ONLY),
    ("output_norm.weight", Disposition.HOST_ONLY),
    ("blk.0.attn_q.weight", Disposition.REPLACED),
    ("blk.0.attn_v.weight", Disposition.TRANSFORMED),
    ("blk.0.ffn_up.weight", Disposition.TRANSFORMED),
    ("blk.0.ffn_down.weight", Disposition.TRANSFORMED),
    ("something_weird", Disposition.REPLACED),
])
def test_classify(name: str, disp: Disposition) -> None:
    assert classify_tensor(name).disposition == disp


def test_report_totals(tiny: Path) -> None:
    s = inspect_gguf(tiny)
    tot = analyze(s).totals()
    assert sum(v["parameters"] for v in tot.values()) == s.parameter_count
    assert tot["PRESERVED"]["tensors"] == 2


# --- backend --------------------------------------------------------------

def test_meets_target() -> None:
    zero = bytes(32)
    assert meets_target(zero, 256)
    top_set = bytes(31) + b"\x80"  # little-endian: MSB of the 256-bit value
    assert not meets_target(top_set, 1) and meets_target(top_set, 0)


def test_simulator_matches_reference() -> None:
    prefix = bytes(range(76))
    be = SimulatedS9Backend()
    res = be.run_jobs([HashJob(7, prefix, 0, 64, 2)])[0]
    ref = [n for n in range(64)
           if int.from_bytes(hashlib.sha256(hashlib.sha256(prefix + struct.pack("<I", n))
                                            .digest()).digest(), "little") >> 254 == 0]
    assert res.job_id == 7 and res.nonces == ref
    assert be.stats.hashes == 64


def test_bitcoin_genesis_block() -> None:
    hdr = bytes.fromhex(
        "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b2"
        "7ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c")
    assert sha256d(hdr)[::-1].hex().startswith("000000000019d6689c085ae165831e93")
    nonce = struct.unpack("<I", hdr[76:])[0]
    res = SimulatedS9Backend().run_jobs([HashJob(0, hdr[:76], nonce - 3, 8, 32)])[0]
    assert res.nonces == [nonce]


def test_ticket_floor() -> None:
    with pytest.raises(ValueError):
        SimulatedS9Backend(min_difficulty_bits=8).submit_job(HashJob(0, bytes(76), 0, 1, 1))


# --- conversion / hcmodel / simulation -----------------------------------

def test_convert_and_roundtrip(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "tiny.hcmodel"
    m = convert(tiny, out, **SMALL)
    assert m.conversion["projection"]["method"] == "svd_gram"
    assert m.params.projection.shape == (64, 16)
    np.testing.assert_allclose(m.params.projection.T @ m.params.projection, np.eye(16), atol=1e-4)
    man = read_manifest(out)
    assert man["source"]["architecture"] == "llama"
    assert man["seeds"]["seed"] == m.config.seed
    assert man["tensors"]["preserved/token_embd"]["dtype"] == "float16"
    m2 = HCModel.load(out)
    assert m2.config == m.config
    np.testing.assert_array_equal(m2.wiring.tuple_index, m.wiring.tuple_index)
    np.testing.assert_allclose(m2.params.embedding, m.params.embedding, atol=1e-3)


def test_simulation_deterministic_and_dense(tiny: Path) -> None:
    hc = convert(tiny, **SMALL)
    toks = [5, 9, 5, 100, 7]
    f1 = HashCortexModel(hc.config, hc.params, hc.wiring, SimulatedS9Backend()).run_features(toks)
    f2 = HashCortexModel(hc.config, hc.params, hc.wiring, SimulatedS9Backend()).run_features(toks)
    np.testing.assert_array_equal(f1, f2)
    assert f1.shape == (5, hc.config.n_features)
    assert 0.35 < f1.mean() < 0.65  # difficulty_bits=1 -> p=0.5
    assert not np.array_equal(f1[0], f1[2])  # same token, different reservoir context


def test_readout_fit_reduces_error(tiny: Path) -> None:
    hc = convert(tiny, **SMALL)
    model = HashCortexModel(hc.config, hc.params, hc.wiring, SimulatedS9Backend())
    toks = np.random.default_rng(0).integers(0, 256, 40).tolist()
    feats = model.run_features(toks)
    targets = np.random.default_rng(1).standard_normal((40, 64)).astype(np.float32) * 0.1
    base = float(np.mean((model.hidden(toks, feats) - targets) ** 2))
    assert model.fit_readout(feats, toks, targets) < base
    assert model.logits(toks).shape == (40, 256)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        HashCortexConfig(hidden_dim=8, vocab_size=10, proj_dim=16).validate()


def test_cli_pipeline(tiny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["pipeline", str(tiny), "-o", str(tmp_path / "p.hcmodel"), "-n", "3"]
    args += [x for k, v in SMALL.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    assert main(args) == 0
    out = capsys.readouterr().out
    for s in ("[1] GGUF inspector", "[2] Conversion analysis", "[3] HashCortex", "[4] CPU S9"):
        assert s in out
