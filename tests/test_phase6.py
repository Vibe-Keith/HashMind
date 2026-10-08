from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hashmind.equihash.protocol import (
    ASICConfig,
    EquihashASICBackend,
    EquihashReferenceBackend,
    Work,
    compact_size,
    decode_solution,
    parse_submit,
    read_compact_size,
)
from hashmind.equihash.reference import (
    ZCASH,
    EquihashParams,
    indices_from_minimal,
    minimal_from_indices,
    nonce_bytes,
    solve,
    verify,
)
from hashmind.frozen.convert_ops import OpConv
from hashmind.frozen.runtime import FrozenRuntime, FrozenWeights
from hashmind.hetero.backends import (
    ARTIFACT_KINDS,
    CPUBackend,
    EquihashZ15Backend,
    Operation,
    SHA256S9Backend,
)
from hashmind.hetero.plan import ExecutionPlan, build_graph, plan
from hashmind.llm.llama import LlamaHParams
from make_tiny_gguf import make_tiny_gguf

VEC = json.loads((Path(__file__).parent / "data" / "zcash_equihash_96_5.json").read_text())
P96 = EquihashParams(96, 5)
TOY = EquihashParams(48, 5)


# --- software Equihash vs Zcash's own test vectors ------------------------------------

@pytest.mark.parametrize("v", VEC["solver"], ids=lambda v: f"{v['I'][:8]}-{v['nonce']}")
def test_solver_matches_zcash_vectors(v: dict) -> None:
    got = sorted(tuple(s) for s in solve(P96, v["I"].encode(), nonce_bytes(v["nonce"])).solutions)
    assert got == sorted(tuple(s) for s in v["solutions"])


@pytest.mark.parametrize("i", range(len(VEC["validator"])))
def test_validator_matches_zcash_vectors(i: int) -> None:
    v = VEC["validator"][i]
    assert verify(P96, v["I"].encode(), nonce_bytes(v["nonce"]), v["solution"]).valid == v["valid"]


def test_all_bits_matter() -> None:
    v = VEC["validator"][0]
    m = minimal_from_indices(v["solution"], P96.index_bits)
    for i in range(len(m) * 8):
        mm = bytearray(m)
        mm[i // 8] ^= 1 << (i % 8)
        assert not verify(P96, v["I"].encode(), nonce_bytes(1), indices_from_minimal(bytes(mm), P96.index_bits)).valid


def test_minimal_encoding_zcash_vector() -> None:
    # TestMinimalSolnRepr "Test 4" from zcash src/gtest/test_equihash.cpp (cBitLen 20 -> 21-bit indices)
    idx = [68, 41, 2097151, 1233, 665, 1023, 1, 1048575]
    enc = bytes.fromhex("000220000a7ffffe004d10014c800ffc00002fffff")
    assert minimal_from_indices(idx, 21) == enc and indices_from_minimal(enc, 21) == idx


def test_zcash_parameters() -> None:
    assert (ZCASH.list_size, ZCASH.solution_indices, ZCASH.index_bits, ZCASH.minimal_bytes, ZCASH.digest_bytes) == \
        (2**21, 512, 21, 1344, 50)


# --- controller protocol (ZIP 301) and the ASIC view -----------------------------------------

def test_compact_size_and_submit_roundtrip() -> None:
    assert compact_size(1344) == bytes.fromhex("fd4005") and read_compact_size(bytes.fromhex("fd4005")) == (1344, 3)
    w = Work.from_payload("job1", b"model data", target=(1 << 256) - 1)
    ref = EquihashReferenceBackend(TOY)
    asic = EquihashASICBackend(TOY)
    r = ref.compute(w, 0)
    a = asic.submit(w, nonces=1)
    assert len(a.submits) == len(r.solutions)
    for s, sol in zip(a.submits, r.solutions):
        p = parse_submit(s.line())
        assert p.job_id == "job1" and decode_solution(p.solution, TOY) == sol
    notify = json.loads(w.notify())
    assert notify["method"] == "mining.notify" and len(notify["params"]) == 8


def test_asic_view_hides_intermediates_and_filters() -> None:
    w = Work.from_payload("j", b"x")
    full = EquihashASICBackend(TOY).submit(w, nonces=8)
    assert not hasattr(full, "hashes") and all(not hasattr(s, "round_sizes") for s in full.submits)
    hard = EquihashASICBackend(TOY).submit(Work.from_payload("j", b"x", target=0), nonces=8)
    assert hard.submits == []  # nothing below the pool target is ever seen
    nonce_only = EquihashASICBackend(TOY, ASICConfig(exposure="nonce_only")).submit(w, nonces=8)
    assert len(nonce_only.submits) == len(full.submits) and all(s.solution == "" for s in nonce_only.submits)


def test_work_generation_is_deterministic() -> None:
    a = Work.from_payload("j", bytes(range(96)))
    assert a.header_prefix() == Work.from_payload("j", bytes(range(96))).header_prefix()
    assert len(a.header_prefix()) == 108
    with pytest.raises(ValueError):
        Work.from_payload("j", bytes(97))


# --- backends, planning, heterogeneous execution ---------------------------------------------

HP = LlamaHParams(2, 64, 4, 4, 16, 10000.0, 1e-5)
CPU = CPUBackend(flops_per_s=1e10, values_per_s=1e7)
BK = {"cpu": CPU, "sha256_s9": SHA256S9Backend(), "equihash_z15": EquihashZ15Backend()}


def test_asic_backends_only_support_artifacts() -> None:
    for kind in ("matmul", "attention", "norm", "lm_head"):
        op = Operation("x", kind, flops=1e6)
        assert CPU.supports(op) and not BK["sha256_s9"].supports(op) and not BK["equihash_z15"].supports(op)
    assert all(BK["sha256_s9"].supports(Operation("a", k)) for k in ARTIFACT_KINDS)


def test_min_cost_plan_and_forced_fallback() -> None:
    conv = {"attn_qkv": OpConv("quant", 8, "sha256d"), "lm_head": OpConv("sampled", samples=16)}
    ops = build_graph(HP, conv, 128, 256, context=8)
    assert any(o.kind == "hash_dither" for o in ops) and any(o.kind == "index_sampling" for o in ops)
    best = plan(ops, BK)
    assert set(best.assignment().values()) == {"cpu"}  # CPU is cheaper for every artifact at these rates
    forced = plan(ops, BK, forced={"hash_dither": "sha256_s9", "matmul": "sha256_s9"})
    a = forced.assignment()
    assert a["attn_qkv.hash_dither"] == "sha256_s9" and a["attn_qkv"] == "cpu"
    assert any("fell back to cpu" in n for n in forced.notes)
    assert ExecutionPlan.from_json(forced.to_json()).to_json() == forced.to_json()  # reproducible plan


def test_plan_drives_runtime_primitives(tmp_path: Path) -> None:
    w = FrozenWeights.from_gguf(make_tiny_gguf(tmp_path / "t.gguf"))
    conv = {"mlp_down": OpConv("quant", 8, "sha256d")}
    ops = build_graph(w.hp, conv, w.layers[0]["w_in"].shape[0] // 2, w.head.shape[0])
    pl = plan(ops, BK, forced={"hash_dither": "cpu"})
    rconv = pl.runtime_conv(conv, {k: b.primitive for k, b in BK.items()})
    assert rconv["mlp_down"].rounding == "splitmix"
    toks = np.array([[1, 5, 9, 12, 40, 7]])
    rt = FrozenRuntime(w, rconv)
    out = rt.forward(toks)
    assert np.isfinite(out).all() and set(rt.hash_evaluations()) == {"splitmix"}
    pl2 = plan(ops, BK, forced={"hash_dither": "sha256_s9"})
    assert pl2.runtime_conv(conv, {k: b.primitive for k, b in BK.items()})["mlp_down"].rounding == "sha256d"


def test_no_training_guarantee(tmp_path: Path) -> None:
    """The converted runtime has no fitted state: same GGUF + plan -> identical logits; weights untouched."""
    g = make_tiny_gguf(tmp_path / "t.gguf")
    w = FrozenWeights.from_gguf(g)
    before = [m.copy() for _, m in w.matrices()]
    conv = {"lm_head": OpConv("sampled", samples=16, primitive="equihash"), "attn_qkv": OpConv("quant", 8, "sha256d")}
    toks = np.array([[1, 5, 9, 12]])
    a = FrozenRuntime(w, conv).forward(toks)
    b = FrozenRuntime(FrozenWeights.from_gguf(g), conv).forward(toks)
    np.testing.assert_array_equal(a, b)
    assert all(np.array_equal(x, m) for x, (_, m) in zip(before, w.matrices()))
