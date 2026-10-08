"""Phase 6: Equihash + SHA-256 heterogeneous frozen-model study.

    python -m hashmind experiment phase6 model.gguf -o docs/results/phase6 [--backend all|sha256|equihash|heterogeneous]

1. reference correctness   software Equihash vs Zcash's own (96,5) vectors; (200,9) solves verified
2. information channel     what a solution carries: count, distribution, entropy, uniqueness,
                           dependence on the input (locality), shaping cost, decode cost
3. backend benchmark       measured CPU rates vs modelled S9 / Z15 Pro rates, per useful value
4. frozen-model runs       CPU-only vs CPU+S9 vs CPU+Equihash vs CPU+S9+Equihash on TinyLlama:
                           logit fidelity vs the original model + modelled tokens/s from the plan
No training, fitting or calibration anywhere. Conversions are the fixed Phase-5 ones.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..equihash.protocol import ASICConfig, EquihashASICBackend, Work, decode_solution, parse_submit
from ..equihash.reference import ZCASH, EquihashParams, minimal_from_indices, nonce_bytes, solve, verify
from ..frozen.convert_ops import LINEAR_OPS, OpConv
from ..frozen.metrics import generation_fidelity, logit_fidelity
from ..frozen.runtime import FrozenRuntime, FrozenWeights
from ..gguf.reader import read_gguf
from ..hetero.backends import ASIC_FACTS, CPUBackend, EquihashZ15Backend, SHA256S9Backend, default_backends
from ..hetero.plan import build_graph, plan
from ..llm.tokenizer import SPMTokenizer
from .phase4_common import environment, write_json
from .phase5 import EvalData

TESTVEC = Path(__file__).resolve().parents[2] / "tests" / "data" / "zcash_equihash_96_5.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- 1. correctness --

def reference_correctness(n_200_9: int = 3) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if TESTVEC.exists():
        d = json.loads(TESTVEC.read_text())
        p = EquihashParams(d["n"], d["k"])
        solver_ok = []
        for v in d["solver"]:
            got = sorted(tuple(s) for s in solve(p, v["I"].encode(), nonce_bytes(v["nonce"])).solutions)
            solver_ok.append(got == sorted(tuple(s) for s in v["solutions"]))
        val_ok = [verify(p, v["I"].encode(), nonce_bytes(v["nonce"]), v["solution"]).valid == v["valid"]
                  for v in d["validator"]]
        out["zcash_96_5"] = {"solver_vectors": len(solver_ok), "solver_match": sum(solver_ok),
                             "validator_vectors": len(val_ok), "validator_match": sum(val_ok), "source": d["source"]}
    rows = []
    for v in range(n_200_9):
        I = b"HashMind phase 6 reference".ljust(108, b"\0")
        r = solve(ZCASH, I, nonce_bytes(v))
        rows.append({"nonce": v, "solutions": len(r.solutions), "seconds": r.seconds,
                     "all_verified": all(verify(ZCASH, I, nonce_bytes(v), s).valid for s in r.solutions),
                     "minimal_bytes": [len(minimal_from_indices(s, 21)) for s in r.solutions]})
    out["zcash_200_9_software"] = rows
    return out


# ------------------------------------------------------- 2. information channel --

def information_channel(p: EquihashParams = EquihashParams(96, 5), n_jobs: int = 200, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    N, K, B = p.list_size, p.solution_indices, p.index_bits
    sols_per_nonce, allsol, t_solve = [], [], 0.0
    for j in range(n_jobs):
        w = Work.from_payload(f"j{j}", rng.integers(0, 256, 96, dtype=np.uint8).tobytes())
        t0 = time.perf_counter()
        s = solve(p, w.header_prefix(), nonce_bytes(0)).solutions
        t_solve += time.perf_counter() - t0
        sols_per_nonce.append(len(s))
        allsol += s
    S = np.array(allsol, np.int64)  # (M, K)
    pos_mean = (S / N).mean(0)
    hist = np.bincount((S.ravel() * 64) // N, minlength=64)
    expct = S.size / 64
    chi2 = float(((hist - expct) ** 2 / expct).sum())
    # entropy per position (histogram over 64 bins -> lower bound on index entropy at 6-bit resolution)
    def ent(x: np.ndarray, bins: int = 64) -> float:
        h = np.bincount((x * bins) // N, minlength=bins) / len(x)
        h = h[h > 0]
        return float(-(h * np.log2(h)).sum())
    pos_entropy_6bit = [ent(S[:, i]) for i in range(K)]
    # canonical ordering removes one bit per internal tree node: K - 1 bits
    raw_bits = K * B
    # locality: similar inputs -> similar solutions?
    base = rng.standard_normal((30, 96))
    def payload(x: np.ndarray) -> bytes:
        return np.clip(np.round(x / np.abs(x).max() * 127), -127, 127).astype(np.int8).tobytes()
    def idxset(x: np.ndarray) -> set[int]:
        w = Work.from_payload("l", payload(x))
        out: set[int] = set()
        v = 0
        while not out:
            for s in solve(p, w.header_prefix(), nonce_bytes(v)).solutions:
                out |= set(s)
            v += 1
        return out
    def jac(a: set[int], b: set[int]) -> float:
        return len(a & b) / max(len(a | b), 1)
    near, far, same = [], [], []
    for x in base:
        y = x + 0.02 * rng.standard_normal(96)  # cosine ~0.9998, mostly identical int8 payload
        z = rng.standard_normal(96)
        a = idxset(x)
        near.append(jac(a, idxset(y)))
        far.append(jac(a, idxset(z)))
        same.append(jac(a, idxset(x)))
    chance = K / N
    # shaping: probability that some solution of a nonce contains an index from a target set of size m
    shaping = []
    for m in (1, 16, 256, 4096):
        target = set(rng.choice(N, m, replace=False).tolist())
        hit = np.mean([any(set(s) & target for s in solve(p, Work.from_payload("s", rng.integers(
            0, 256, 96, dtype=np.uint8).tobytes()).header_prefix(), nonce_bytes(0)).solutions) for _ in range(60)])
        lam = float(np.mean(sols_per_nonce))
        analytic = 1 - math.exp(-lam * (1 - (1 - m / N) ** K))
        shaping.append({"target_set": m, "measured_p_per_nonce": float(hit), "analytic_p_per_nonce": analytic})
    # shaping by rejection: the controller wants b chosen bits in a fixed, checkable place (top bits of the
    # index at solution position 1). Measured hit probability per solution vs 2**-b, from the collected solutions.
    chosen = []
    for bits in (1, 2, 4, 6, 8):
        t = rng.integers(0, 2**bits, 200)
        hits = [(S[:, 1] >> (B - bits) == tt).mean() for tt in t]
        chosen.append({"bits": bits, "measured_p_per_solution": float(np.mean(hits)), "uniform_p": 2.0**-bits})
    lam = float(np.mean(sols_per_nonce))
    z = ZCASH
    zf = ASIC_FACTS["z15pro"]
    extrap = {b: {"expected_solutions": 2.0**b, "expected_nonces": 2.0**b / lam,
                  "z15pro_seconds_nominal_rate": 2.0**b / zf["nominal_solutions_per_s"],
                  "z15pro_seconds_observed_share_rate": 2.0**b / zf["observed_shares_per_s"]}
              for b in (1, 8, 16, 32, 64)}
    # decode cost of one (200, 9) submit message
    I = b"decode".ljust(108, b"\0")
    r = solve(z, I, nonce_bytes(0))
    if not r.solutions:
        r = solve(z, I, nonce_bytes(2))
    asic = EquihashASICBackend(z)
    sub = Work.from_payload("d", b"decode")
    mini = minimal_from_indices(r.solutions[0], z.index_bits)
    from ..equihash.protocol import Submit, compact_size
    line = Submit("w", "d", "00000000", "00" * 28, (compact_size(len(mini)) + mini).hex()).line()
    t0 = time.perf_counter()
    for _ in range(200):
        idx = decode_solution(parse_submit(line).solution, z)
    dec_us = (time.perf_counter() - t0) / 200 * 1e6
    return {
        "params": asdict(p), "jobs": n_jobs, "solutions": len(allsol),
        "solutions_per_nonce_mean": float(np.mean(sols_per_nonce)), "solutions_per_nonce_hist":
            np.bincount(sols_per_nonce).tolist(), "software_solve_s_mean": t_solve / n_jobs,
        "raw_bits_per_solution": raw_bits, "ordering_constraint_bits": K - 1,
        "upper_bound_bits_per_solution": raw_bits - (K - 1),
        "position_mean_normalized": pos_mean.tolist(), "first_index_mean_normalized": float(pos_mean[0]),
        "global_uniformity_chi2_63dof": chi2, "position_entropy_6bit": pos_entropy_6bit,
        "distinct_indices_within_solution": bool(all(len(set(s)) == K for s in allsol)),
        "locality": {"jaccard_same_input": float(np.mean(same)), "jaccard_near_input": float(np.mean(near)),
                     "jaccard_random_input": float(np.mean(far)), "chance_per_index": chance},
        "shaping_target_set": shaping, "shaping_chosen_bits": chosen, "shaping_extrapolated_200_9": extrap,
        "decode_us_per_submit_200_9": dec_us, "decoded_indices": len(idx), "submit_line_bytes": len(line),
    }


# ---------------------------------------------------------- 3. backend matrix --

def backend_matrix(cpu: CPUBackend) -> dict[str, Any]:
    import hashlib
    from ..core.primitives import get_primitive, measure_throughput

    rows = []
    sha = measure_throughput(get_primitive("sha256d"), 2048, 16)
    t0 = time.perf_counter()
    for i in range(20000):
        hashlib.blake2b(i.to_bytes(4, "little"), digest_size=50).digest()
    blake = 20000 / (time.perf_counter() - t0)
    eq = {}
    for (n, k), reps in (((48, 5), 50), ((96, 5), 5), ((200, 9), 1)):
        p = EquihashParams(n, k)
        t0 = time.perf_counter()
        ns = sum(len(solve(p, b"bench".ljust(108, b"\0"), nonce_bytes(v)).solutions) for v in range(reps))
        dt = time.perf_counter() - t0
        eq[f"{n},{k}"] = {"solves_per_s": reps / dt, "solutions_per_s": ns / dt,
                          "values_per_s": ns / dt * p.solution_indices * p.index_bits / 32}
    rows.append({"backend": "CPU (this host, 1 core unless noted)", "measured": True, "primitive": "splitmix64",
                 "useful_values_per_s": cpu.values_per_s, "bytes_per_s": cpu.values_per_s * 4,
                 "latency_s": 0.0, "power_w": cpu.power_w, "j_per_value": cpu.power_w / cpu.values_per_s})
    rows.append({"backend": "CPU SHA-256d (python hashlib, 1 core)", "measured": True, "primitive": "sha256d",
                 "useful_values_per_s": sha, "bytes_per_s": sha * 4, "power_w": cpu.power_w,
                 "j_per_value": cpu.power_w / sha})
    rows.append({"backend": "CPU BLAKE2b-400 (python hashlib, 1 core)", "measured": True, "primitive": "blake2b",
                 "useful_values_per_s": blake * 50 / 4, "bytes_per_s": blake * 50})
    for k, v in eq.items():
        rows.append({"backend": f"CPU software Equihash ({k}), numpy reference", "measured": True,
                     "primitive": "equihash", **{kk: vv for kk, vv in v.items()}, "useful_values_per_s": v["values_per_s"]})
    s9 = SHA256S9Backend()
    rows.append({"backend": "S9 (BM1387), share floor", "measured": False, "primitive": "sha256d",
                 "hashes_per_s": s9.f["hashrate_hs"], "shares_per_s": s9.values_per_s,
                 "useful_values_per_s": s9.values_per_s,
                 "link_values_per_s": s9.f["link_bytes_per_s"] / (s9.f["job_bytes_out"] + s9.f["share_bytes_in"]),
                 "jobs_per_s": s9.values_per_s, "latency_s": s9.f["job_latency_s"], "power_w": s9.f["power_w"],
                 "j_per_value": s9.f["power_w"] / s9.values_per_s, "source": s9.f["source"]})
    for vis in ("observed", "link_bound", "nominal"):
        z = EquihashZ15Backend(vis)
        rows.append({"backend": f"Z15 Pro Equihash ({vis})", "measured": False, "primitive": "equihash(200,9)",
                     "solutions_per_s": z.solutions_per_s, "useful_values_per_s": z.values_per_s,
                     "bytes_per_s": z.solutions_per_s * z.f["submit_bytes"], "latency_s": z.f["job_latency_s"],
                     "power_w": z.f["power_w"], "j_per_value": z.f["power_w"] / z.values_per_s, "source": z.f["source"]})
    return {"rows": rows, "cpu_flops_per_s_measured": cpu.flops_per_s,
            "electricity_usd_per_kwh_assumed": 0.15}


# ----------------------------------------------------------- 4. frozen model --

PHASE5_SELECTED_LINEAR = OpConv("quant", 8, "sha256d")  # Phase-5 selection for every linear op class


def configurations(which: str) -> list[tuple[str, dict[str, OpConv], dict[str, str]]]:
    """(name, conversion, forced backend per artifact kind). Conversions fixed a priori:
    S9 work = Phase-5 selected hash-dithered 8-bit activations on the attention/MLP projections;
    Equihash work = index sampling for the LM head (1024 samples) + Equihash-addressed embedding table."""
    s9_conv = {op: PHASE5_SELECTED_LINEAR for op in ("attn_qkv", "attn_out", "mlp_in", "mlp_down")}
    eq_conv = {"lm_head": OpConv("sampled", samples=1024, primitive="equihash"),
               "embedding": OpConv("hashed", slots=512000, primitive="equihash")}
    cfg = [("CPU only, original operations", {}, {})]
    if which in ("all", "sha256", "heterogeneous"):
        cfg += [("CPU only, S9 conversions with CPU randomness", s9_conv, {"hash_dither": "cpu"}),
                ("CPU + S9", s9_conv, {"hash_dither": "sha256_s9"})]
    if which in ("all", "equihash", "heterogeneous"):
        cfg += [("CPU only, Equihash conversions with CPU randomness", eq_conv,
                 {"index_sampling": "cpu", "table_address": "cpu"}),
                ("CPU + Equihash", eq_conv, {"index_sampling": "equihash_z15", "table_address": "equihash_z15"})]
    if which in ("all", "heterogeneous"):
        cfg += [("CPU + S9 + Equihash", {**s9_conv, **eq_conv},
                 {"hash_dither": "sha256_s9", "index_sampling": "equihash_z15", "table_address": "equihash_z15"})]
    return cfg


def frozen_runs(gguf: str | Path, which: str, logf: Callable[[str], None], cpu: CPUBackend,
                quick: bool = False) -> dict[str, Any]:
    g = read_gguf(gguf)
    tok = SPMTokenizer.from_gguf_metadata(g.metadata)
    data = EvalData(tok, n_dev=1 if quick else 4, n_test=1, n_new=2 if quick else 8)
    w = FrozenWeights.from_gguf(g)
    ffn = w.layers[0]["w_in"].shape[0] // 2
    vocab = w.head.shape[0]
    B = default_backends(cpu)
    prim = {name: b.primitive for name, b in B.items()}
    x = data.dev
    ref_rt = FrozenRuntime(w)
    t0 = time.perf_counter()
    ref_logits = ref_rt.forward(x)[:, :-1]
    t_ref = time.perf_counter() - t0
    prompts = data.prompts_dev
    ref_gen = ref_rt.generate(prompts, data.n_new)
    rows = []
    for name, conv, forced in configurations(which):
        ops = build_graph(w.hp, conv, ffn, vocab, context=x.shape[1])
        pl = plan(ops, B, "forced" if forced else "min_cost", forced or None, context=x.shape[1])
        rconv = pl.runtime_conv(conv, prim) if conv else {}
        rt = FrozenRuntime(w, rconv)
        t0 = time.perf_counter()
        lg = rt.forward(x)[:, :-1]
        t_sim = time.perf_counter() - t0
        fid = logit_fidelity(ref_logits, lg, x[:, 1:])
        fid["perplexity_true"] = math.exp(fid["nll_true"])
        gen = rt.generate(prompts, data.n_new)
        row = {"name": name, "runtime_conversions": {k: v.label() for k, v in rconv.items()},
               "plan": json.loads(pl.to_json()), "plan_summary": pl.summary(), "fidelity": fid,
               "generation": generation_fidelity(ref_gen, gen),
               "generated_text": [tok.decode(s) for s in gen],
               "cpu_simulation_s_per_token_measured": t_sim / x.size,
               "hash_evaluations_per_token": {k: v / x.size for k, v in rt.hash_evaluations().items()}}
        rows.append(row)
        logf(f"6-frozen {name:52s} top1 {fid['top1_agreement']:.3f}  modelled {row['plan_summary']['tokens_per_s']:.4g} tok/s")
    # cost-optimal plan for the most converted configuration: does the scheduler pick any ASIC at all?
    conv_all = configurations("heterogeneous")[-1][1]
    best = plan(build_graph(w.hp, conv_all, ffn, vocab, context=x.shape[1]), B, "min_cost", context=x.shape[1])
    return {"reference": {"tokens": int(x.size), "s_per_token_measured": t_ref / x.size,
                          "generation": [tok.decode(s) for s in ref_gen]},
            "rows": rows, "min_cost_plan_for_all_conversions": {"assignment": best.assignment(),
                                                                "summary": best.summary()},
            "data": data.info}


# ---------------------------------------------------------------- driver ------

def run_phase6(gguf: str | Path, out: str | Path, which: str = "all", logf: Callable[[str], None] = log,
               quick: bool = False) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    logf("measuring CPU")
    cpu = CPUBackend()
    logf("1. reference correctness")
    corr = reference_correctness(1 if quick else 3)
    write_json(out / "equihash_reference.json", {"environment": env, **corr})
    logf("2. information channel")
    info = information_channel(n_jobs=20 if quick else 200)
    write_json(out / "information_channel.json", {"environment": env, **info})
    logf("3. backend matrix")
    bm = backend_matrix(cpu)
    write_json(out / "backend_benchmark.json", {"environment": env, **bm})
    logf("4. frozen-model heterogeneous runs")
    fr = frozen_runs(gguf, which, logf, cpu, quick)
    write_json(out / "heterogeneous.json", {"environment": env, "asic_facts": ASIC_FACTS, **fr})
    summary = {"environment": env, "correctness": corr.get("zcash_96_5"), "information_channel": {
        k: info[k] for k in ("solutions_per_nonce_mean", "upper_bound_bits_per_solution", "locality",
                             "first_index_mean_normalized", "decode_us_per_submit_200_9")},
        "heterogeneous": [{"name": r["name"], "top1": r["fidelity"]["top1_agreement"],
                           "kl": r["fidelity"]["kl_ref_to_conv"], "ppl": r["fidelity"]["perplexity_true"],
                           "gen_exact": r["generation"]["exact_match_rate"],
                           "modelled_tokens_per_s": r["plan_summary"]["tokens_per_s"],
                           "modelled_seconds_by_backend": r["plan_summary"]["seconds_by_backend"],
                           "energy_j_per_token": r["plan_summary"]["energy_j_per_token"]} for r in fr["rows"]],
        "min_cost_plan": fr["min_cost_plan_for_all_conversions"]}
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(report(corr, info, bm, fr))
    return summary


def report(corr, info, bm, fr) -> str:
    L = ["# Phase 6 generated report", "", "## Equihash reference correctness", "", "```",
         json.dumps(corr, indent=1)[:3000], "```", "", "## Information channel ((96,5) software, extrapolated to (200,9))", "",
         f"- solutions per nonce: {info['solutions_per_nonce_mean']:.2f} (hist {info['solutions_per_nonce_hist']})",
         f"- raw bits/solution {info['raw_bits_per_solution']}, minus ordering constraint -> at most "
         f"{info['upper_bound_bits_per_solution']} bits",
         f"- first index mean (normalized): {info['first_index_mean_normalized']:.3f} (uniform = 0.5)",
         f"- locality (Jaccard of solution index sets): same {info['locality']['jaccard_same_input']:.3f}, "
         f"near {info['locality']['jaccard_near_input']:.3f}, random {info['locality']['jaccard_random_input']:.3f}",
         f"- decode one (200,9) submit: {info['decode_us_per_submit_200_9']:.0f} µs", "",
         "| target set | measured P/nonce | analytic |", "|---:|---:|---:|"]
    L += [f"| {s['target_set']} | {s['measured_p_per_nonce']:.3f} | {s['analytic_p_per_nonce']:.3f} |" for s in info["shaping_target_set"]]
    L += ["", "| chosen bits | measured P/solution | 2^-b | Z15 Pro s (nominal) | Z15 Pro s (observed share rate) |",
          "|---:|---:|---:|---:|---:|"]
    ex = info["shaping_extrapolated_200_9"]
    for c in info["shaping_chosen_bits"]:
        e = ex.get(c["bits"]) or ex.get(str(c["bits"])) or {}
        L.append(f"| {c['bits']} | {c['measured_p_per_solution']:.4f} | {c['uniform_p']:.4f} | "
                 f"{e.get('z15pro_seconds_nominal_rate', float('nan')):.3g} | {e.get('z15pro_seconds_observed_share_rate', float('nan')):.3g} |")
    L += ["", "Extrapolated cost of forcing b chosen bits into a (200,9) solution: " +
          json.dumps({k: {kk: f"{vv:.3g}" for kk, vv in v.items()} for k, v in ex.items()})]
    L += ["", "## Backend benchmark (measured = this host; others modelled)", "",
          "| backend | measured | useful 32-bit values/s | power W | J/value |", "|---|:-:|---:|---:|---:|"]
    for r in bm["rows"]:
        L.append(f"| {r['backend']} | {'yes' if r['measured'] else 'model'} | {r['useful_values_per_s']:,.3g} | "
                 f"{r.get('power_w', '-')} | {r.get('j_per_value', float('nan')):.3g} |")
    L += ["", "## Frozen TinyLlama, heterogeneous configurations (dev set)", "",
          "| configuration | top-1 | top-5 | KL | ppl | gen exact | modelled tok/s | modelled J/token | ASIC values/token |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fr["rows"]:
        f, s = r["fidelity"], r["plan_summary"]
        L.append(f"| {r['name']} | {f['top1_agreement']:.1%} | {f['top5_agreement']:.1%} | {f['kl_ref_to_conv']:.3f} | "
                 f"{f['perplexity_true']:.2f} | {r['generation']['exact_match_rate']:.0%} | {s['tokens_per_s']:.4g} | "
                 f"{s['energy_j_per_token']:.3g} | {s['asic_values_per_token']:,.0f} |")
    L += ["", "Cost-optimal plan when every conversion is enabled: " +
          json.dumps(fr["min_cost_plan_for_all_conversions"]["summary"]["seconds_by_backend"]), ""]
    return "\n".join(L) + "\n"
