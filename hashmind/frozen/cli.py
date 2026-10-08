"""Phase-5 CLI commands: convert-frozen, compare, benchmark-frozen, analyze-conversion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..gguf.reader import read_gguf
from ..llm.tokenizer import SPMTokenizer
from .convert_ops import OP_CLASSES, parse_label
from .hardware import end_to_end
from .hmfrozen import convert_frozen, load_frozen
from .metrics import generation_fidelity, logit_fidelity
from .runtime import FrozenRuntime, FrozenWeights


def _ops(specs: list[str] | None) -> dict:
    out = {}
    for s in specs or []:
        op, label = s.split("=", 1)
        if op not in OP_CLASSES:
            raise SystemExit(f"unknown op class {op}; choose from {OP_CLASSES}")
        out[op] = parse_label(label)
    return out


def _prompts(gguf: str, path: str | None) -> tuple[SPMTokenizer, list[list[int]]]:
    tok = SPMTokenizer.from_gguf_metadata(read_gguf(gguf).metadata)
    if path:
        items = json.loads(Path(path).read_text())
    else:
        from ..experiments.phase5 import PROMPTS_DEV
        items = PROMPTS_DEV
    return tok, [tok.encode(p) if isinstance(p, str) else list(p) for p in items]


BACKEND_POLICIES = {
    "cpu": {},
    "sha256": {"hash_dither": "sha256_s9", "table_address": "sha256_s9", "threshold_event": "sha256_s9"},
    "equihash": {"index_sampling": "equihash_z15", "table_address": "equihash_z15"},
    "heterogeneous": {"hash_dither": "sha256_s9", "index_sampling": "equihash_z15", "table_address": "equihash_z15"},
}


def make_plan(gguf: str, conv: dict, backend: str):
    from ..hetero.backends import default_backends
    from ..hetero.plan import build_graph, plan

    w = FrozenWeights.from_gguf(gguf)
    B = default_backends()
    ops = build_graph(w.hp, conv, w.layers[0]["w_in"].shape[0] // 2, w.head.shape[0])
    forced = BACKEND_POLICIES[backend]
    pl = plan(ops, B, "forced" if forced else "min_cost", forced or None)
    return pl, pl.runtime_conv(conv, {k: b.primitive for k, b in B.items()}) if conv else conv


def cmd_convert(a: argparse.Namespace) -> None:
    bits = None if a.weight_bits == "none" else int(a.weight_bits)
    conv = _ops(a.op)
    pl_json = None
    if a.backend:
        pl, conv = make_plan(a.gguf, conv, a.backend)
        pl_json = pl.to_json()
    m = convert_frozen(a.gguf, a.out, bits, a.weight_rounding, conv, log=print if a.verbose else None,
                       execution_plan=pl_json)
    print(json.dumps({"out": a.out, "hmmodel_sha256": m["hmmodel_sha256"], "source": m["source"],
                      "conversion": m["conversion"]}, indent=1, default=str))


def cmd_compare(a: argparse.Namespace) -> None:
    tok, prompts = _prompts(a.gguf, a.prompts)
    ref = FrozenRuntime(FrozenWeights.from_gguf(a.gguf))
    w, conv, man = load_frozen(a.hmmodel)
    hm = FrozenRuntime(w, conv)
    res = {"prompts": len(prompts), "positions": []}
    lr, lh, nt = [], [], 0
    t_ref = t_hm = 0.0
    for p in prompts:
        x = np.array([p])
        t0 = time.perf_counter(); lr.append(ref.forward(x)[0]); t_ref += time.perf_counter() - t0
        t0 = time.perf_counter(); lh.append(hm.forward(x)[0]); t_hm += time.perf_counter() - t0
        nt += len(p)
    res["fidelity"] = logit_fidelity(np.concatenate(lr), np.concatenate(lh))
    g_ref, g_hm = ref.generate(prompts, a.n_new), hm.generate(prompts, a.n_new)
    res["generation"] = generation_fidelity(g_ref, g_hm)
    res["generated"] = [{"original": tok.decode(r), "hashmind": tok.decode(h)} for r, h in zip(g_ref, g_hm)]
    res["latency_s_per_token"] = {"original": t_ref / nt, "hashmind_cpu": t_hm / nt}
    res["hash_evaluations"] = hm.hash_evaluations()
    res["host_ops"] = {op: {k: v for k, v in c.items()} for op, c in hm.cv.counts.items()}
    print(json.dumps(res, indent=1, default=str))


def cmd_benchmark(a: argparse.Namespace) -> None:
    from ..experiments.next_token import default_corpus

    tok = SPMTokenizer.from_gguf_metadata(read_gguf(a.gguf).metadata)
    x = np.array([[tok.bos_id] + tok.encode(default_corpus(), bos=False)[:a.tokens - 1]])
    ref = FrozenRuntime(FrozenWeights.from_gguf(a.gguf))
    w, conv, _ = load_frozen(a.hmmodel)
    hm = FrozenRuntime(w, conv)
    t0 = time.perf_counter(); lr = ref.forward(x); t_ref = (time.perf_counter() - t0) / x.size
    t0 = time.perf_counter(); lh = hm.forward(x); t_hm = (time.perf_counter() - t0) / x.size
    macs = sum(c.get("fp_macs", 0) for c in ref.cv.counts.values()) / x.size
    counts = {k: sum(c.get(k, 0) for c in hm.cv.counts.values()) / x.size
              for k in ("fp_macs", "int_macs", "adds", "lut_lookups")}
    hashes = sum(hm.hash_evaluations().values()) / x.size
    from ..core.primitives import get_primitive, measure_throughput
    e2e = end_to_end(counts, hashes, t_ref, t_hm, macs, measure_throughput(get_primitive("sha256d")),
                     sum(m.size for _, m in w.matrices()) * 0.5)
    print(json.dumps({"tokens": int(x.size), "fidelity": logit_fidelity(lr[0], lh[0]), "end_to_end": e2e},
                     indent=1, default=str))


def cmd_analyze(a: argparse.Namespace) -> None:
    from ..experiments.next_token import default_corpus

    tok = SPMTokenizer.from_gguf_metadata(read_gguf(a.gguf).metadata)
    x = np.array([[tok.bos_id] + tok.encode(default_corpus(), bos=False)[:a.tokens - 1]])
    src = FrozenWeights.from_gguf(a.gguf)
    w, conv, man = load_frozen(a.hmmodel)
    weight_err = {}
    for (n, A), (_, B) in zip(src.matrices(), w.matrices()):
        d = (B - A).astype(np.float64)
        weight_err[n] = {"relative_error": float(np.linalg.norm(d) / np.linalg.norm(A))}
    hm = FrozenRuntime(w, conv)
    lh = hm.forward(x, probe=True)
    lr = FrozenRuntime(src).forward(x)
    print(json.dumps({"manifest": {k: man[k] for k in ("source", "conversion")},
                      "weight_relative_error_mean": float(np.mean([v["relative_error"] for v in weight_err.values()])),
                      "op_tensor_error": {k: v.summary() for k, v in hm.probe.items()},
                      "fidelity": logit_fidelity(lr[0], lh[0]),
                      "hash_evaluations_per_token": {k: v / x.size for k, v in hm.hash_evaluations().items()}},
                     indent=1, default=str))


def cmd_hardware_info(a: argparse.Namespace) -> None:
    from ..hetero.backends import ASIC_FACTS, EquihashZ15Backend, SHA256S9Backend

    print(json.dumps({
        "sha256_s9": {"exposes": "nonces whose SHA-256d meets the ticket mask (floor 2^-32): one value per share",
                      "values_per_s_modelled": SHA256S9Backend().values_per_s, **ASIC_FACTS["s9"]},
        "equihash_z15": {"exposes": "mining.submit with the full 1344-byte (200,9) solution, for shares only "
                                    "(see docs/PHASE6_EQUIHASH_HARDWARE.md: PARTIAL_SOLUTION_ACCESS)",
                         **{v: EquihashZ15Backend(v).values_per_s for v in ("observed", "link_bound", "nominal")},
                         **ASIC_FACTS["z15pro"]},
        "cpu": {"exposes": "everything"}}, indent=1, default=str))


def cmd_analyze_backends(a: argparse.Namespace) -> None:
    pl, conv = make_plan(a.gguf, _ops(a.op), a.backend)
    print(pl.to_json())
    print(json.dumps(pl.summary(), indent=1))


def cmd_benchmark_backends(a: argparse.Namespace) -> None:
    from ..experiments.phase6 import backend_matrix
    from ..hetero.backends import CPUBackend

    print(json.dumps(backend_matrix(CPUBackend()), indent=1, default=str))


def add_parsers(sub: argparse._SubParsersAction) -> None:
    sp = sub.add_parser("convert-frozen", help="deterministic frozen conversion GGUF -> .hmmodel (no training)")
    sp.add_argument("gguf"); sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--weight-bits", default="8", help="2|4|8|16|none")
    sp.add_argument("--weight-rounding", default="rtn", help="rtn | sha256d | splitmix (hash dither)")
    sp.add_argument("--op", action="append", help="op_class=label, e.g. mlp_in=quant8-sha256d (repeatable)")
    sp.add_argument("--backend", choices=sorted(BACKEND_POLICIES),
                    help="record a heterogeneous execution plan (artifact ops -> S9 / Equihash / CPU)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(fn=cmd_convert)
    sp = sub.add_parser("hardware-info", help="what each ASIC backend exposes (facts, sources, modelled rates)")
    sp.set_defaults(fn=cmd_hardware_info)
    sp = sub.add_parser("analyze-backends", help="operation graph + backend assignment for a GGUF")
    sp.add_argument("gguf"); sp.add_argument("--backend", default="cpu", choices=sorted(BACKEND_POLICIES))
    sp.add_argument("--op", action="append", help="op_class=label conversions to plan for")
    sp.set_defaults(fn=cmd_analyze_backends)
    sp = sub.add_parser("benchmark-backends", help="measured CPU vs modelled S9 / Z15 primitive rates")
    sp.set_defaults(fn=cmd_benchmark_backends)
    sp = sub.add_parser("compare", help="original GGUF vs frozen .hmmodel on prompts")
    sp.add_argument("gguf"); sp.add_argument("hmmodel"); sp.add_argument("--prompts", help="JSON list of strings")
    sp.add_argument("--n-new", type=int, default=16); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("benchmark-frozen", help="latency + modelled S9 cost of a frozen .hmmodel")
    sp.add_argument("gguf"); sp.add_argument("hmmodel"); sp.add_argument("--tokens", type=int, default=64)
    sp.set_defaults(fn=cmd_benchmark)
    sp = sub.add_parser("analyze-conversion", help="weight and per-op tensor error of a frozen .hmmodel")
    sp.add_argument("gguf"); sp.add_argument("hmmodel"); sp.add_argument("--tokens", type=int, default=64)
    sp.set_defaults(fn=cmd_analyze)
