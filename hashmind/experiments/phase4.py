"""Phase-4 runner: four tracks on the standardized suite, machine-readable results,
pre-registered decision rules, REPORT.md and SVG plots.

    python -m hashmind experiment phase4-all model.gguf -o docs/results/phase4

Decision rules (fixed before any Phase-4 result was seen; see docs/PHASE4.md):

* Comparisons are *paired by seed* (same seed = same split, same wiring; only the
  compared factor differs). A difference is **significant** if
  |mean| > 2 * std / sqrt(n) over seeds AND |mean| >= 0.5 percentage points.
* Q1 SHA vs cheap: primary comparison SHA-256d vs splitmix (strong non-crypto
  mixer) on every Track-A/B/C/D row pair. yes = SHA significantly better in a
  majority of pairs and never significantly worse; no = SHA significantly better
  in < 25% of pairs; else inconclusive.
* Q2 multi-resolution: "B: PCA + sha256d multires bits" vs "PCA + current
  HashMind" per task. yes = significant gain on the context task and on at least
  one other task, no significant loss; no = no significant gain on any task.
* Q3 routing: each routed row vs its unrouted twin. yes = significant gain in at
  least half of the pairs; no = no significant gain anywhere; else inconclusive.
* Q4 events: "bucket B=16" (128 evals) vs dense bits (2048 evals), PCA + multires,
  SHA-256d. yes = not significantly worse on any task AND fewer modelled S9
  sweeps AND smaller per-example storage; no = significantly worse on a majority.
* Q5 hardware: yes only if Q1 = yes AND some SHA-256d architecture meets all six
  success criteria; not yet if it meets the predictive ones (1-4, 6) but not the
  economics or Q1; no otherwise, or whenever Q1 = no.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import hash_ablation, learned_routing, multiresolution, sparse_events
from .phase4_common import (
    DEFAULT_SEEDS,
    RowCache,
    Task,
    _THROUGHPUT,
    aggregate,
    environment,
    primitive_throughput,
    reference_rows,
    table,
    write_json,
)
from .svgplot import grouped_bars, lines, scatter

TRACKS = {
    "hash_ablation": hash_ablation,
    "multiresolution": multiresolution,
    "learned_routing": learned_routing,
    "sparse_events": sparse_events,
}
TRACK_TITLES = {
    "hash_ablation": "Track A: hash primitive ablation",
    "multiresolution": "Track B: multi-resolution hashing",
    "learned_routing": "Track C: learned sparse routing",
    "sparse_events": "Track D: sparse / event-based output",
}
CURRENT, PCA_CURRENT = "current HashMind (sha256d single t2l4)", "PCA + current HashMind (sha256d single t2l4)"
MIN_EFFECT = 0.005


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------- running --

def run_track(name: str, tasks: list[Task], seeds: tuple[int, ...], cache: RowCache, out: Path,
              meta: dict[str, Any]) -> dict[str, Any]:
    mod = TRACKS[name]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows += reference_rows(task)
        for rep in mod.reps(task):
            for s in seeds:
                rows.append(cache.get(task, rep, s))
    for p in ("sha256d", "fnv1a", "splitmix", "pcg32"):
        primitive_throughput(p)
    res = {"track": name, "title": TRACK_TITLES[name], "environment": environment(), "meta": meta,
           "seeds": list(seeds), "tasks": {t.name: t.info for t in tasks},
           "cpu_evals_per_s_single_core": dict(_THROUGHPUT), "aggregated": aggregate(rows), "rows": rows}
    write_json(out / f"{name}.json", res)
    (out / f"{name}.md").write_text(track_markdown(res, tasks))
    return res


def track_markdown(res: dict[str, Any], tasks: list[Task]) -> str:
    out = [f"# Phase 4 {res['title']}", "",
           f"Seeds {res['seeds']} (mean ± sample std over seeds). Raw rows: `{res['track']}.json`.", ""]
    for t in tasks:
        aggs = [a for a in res["aggregated"] if a["task"] == t.name]
        out += [f"## {t.name}", "", f"`{t.info}`", "", table(aggs, t.kind, [
            ("S9 ms/ex (model)", "", lambda a: f"{a['s9_feature_ms_per_example']:.1f}"
             if "s9_feature_ms_per_example" in a else "-"),
            ("CPU µs/ex (cacheless, 1 core)", "", lambda a: f"{a['cacheless_feature_us_per_example']:.0f}"
             if "cacheless_feature_us_per_example" in a else "-"),
        ]), ""]
    return "\n".join(out)


# ------------------------------------------------------------------ analysis --

def _by(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    d: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for r in rows:
        if r.get("reference"):
            continue
        d.setdefault((r["task"], r["rep"]), {})[r["seed"]] = r
    return d


def paired(idx: dict, task: str, a: str, b: str, key: str = "score") -> dict[str, Any] | None:
    """a - b, paired by seed."""
    ra, rb = idx.get((task, a)), idx.get((task, b))
    if not ra or not rb:
        return None
    seeds = sorted(set(ra) & set(rb))
    d = np.array([ra[s][key] - rb[s][key] for s in seeds])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    m = float(d.mean())
    sig = bool(abs(m) >= MIN_EFFECT and (n > 1 and abs(m) > 2 * sd / np.sqrt(n)))
    return {"task": task, "a": a, "b": b, "mean_diff": m, "std": sd, "n": n, "significant": sig,
            "a_mean": float(np.mean([ra[s][key] for s in seeds])),
            "b_mean": float(np.mean([rb[s][key] for s in seeds]))}


def analyse(results: dict[str, dict[str, Any]], tasks: list[Task]) -> dict[str, Any]:
    rows = [r for res in results.values() for r in res["rows"]]
    idx = _by(rows)
    aggs = {(a["task"], a["rep"]): a for res in results.values() for a in res["aggregated"]}
    tnames = [t.name for t in tasks]
    ctx_task = next((t.name for t in tasks if t.context_dim), None)

    # Q1 ---------------------------------------------------------------------------
    pairs = []
    for (task, rep) in idx:
        if "sha256d" in rep and not rep.startswith(("current", "PCA + current", "B-scale")):
            for cheap in ("splitmix", "fnv1a", "pcg32"):
                p = paired(idx, task, rep, rep.replace("sha256d", cheap))
                if p:
                    p["cheap"] = cheap
                    pairs.append(p)
    prim = [p for p in pairs if p["cheap"] == "splitmix"]
    better = sum(p["significant"] and p["mean_diff"] > 0 for p in prim)
    worse = sum(p["significant"] and p["mean_diff"] < 0 for p in prim)
    if prim and better > len(prim) / 2 and worse == 0:
        q1 = "yes"
    elif not prim or better < 0.25 * len(prim):
        q1 = "no"
    else:
        q1 = "inconclusive"
    q1_ev = {"pairs_vs_splitmix": len(prim), "sha_significantly_better": better, "sha_significantly_worse": worse,
             "mean_abs_diff_vs_splitmix": float(np.mean([abs(p["mean_diff"]) for p in prim])) if prim else None,
             "max_sha_advantage": max((p["mean_diff"] for p in prim), default=None),
             "secondary": {c: {"pairs": sum(p["cheap"] == c for p in pairs),
                               "sha_better": sum(p["cheap"] == c and p["significant"] and p["mean_diff"] > 0
                                                 for p in pairs),
                               "sha_worse": sum(p["cheap"] == c and p["significant"] and p["mean_diff"] < 0
                                                for p in pairs)} for c in ("fnv1a", "pcg32")},
             "pairs": pairs}

    # Q2 ---------------------------------------------------------------------------
    mr = [paired(idx, t, "B: PCA + sha256d multires bits", PCA_CURRENT) for t in tnames]
    mr = [p for p in mr if p]
    gain = {p["task"] for p in mr if p["significant"] and p["mean_diff"] > 0}
    loss = {p["task"] for p in mr if p["significant"] and p["mean_diff"] < 0}
    if ctx_task in gain and len(gain) >= 2 and not loss:
        q2 = "yes"
    elif not gain:
        q2 = "no"
    else:
        q2 = "inconclusive"
    cells = {t: {k: aggs.get((t, r), {}).get("repeated_cell_rate") for k, r in (
        ("current", CURRENT), ("multires", "B: sha256d multires bits"),
        ("single_t4l4", "A: sha256d single t4l4"), ("single_t8l2", "A: sha256d single t8l2"))} for t in tnames}
    q2_ev = {"pairs": mr, "repeated_cell_rate": cells}

    # Q3 ---------------------------------------------------------------------------
    rt = []
    for (task, rep) in idx:
        if rep.endswith(" + routing"):
            twin = rep[:-len(" + routing")].replace("C: ", "B: " if "multires" in rep else "A: ", 1)
            p = paired(idx, task, rep, twin)
            if p:
                rt.append(p)
    g3 = sum(p["significant"] and p["mean_diff"] > 0 for p in rt)
    q3 = "yes" if rt and g3 >= len(rt) / 2 else ("no" if g3 == 0 else "inconclusive")
    q3_ev = {"pairs": rt, "significant_gains": g3,
             "learned_state_bytes": sorted({a.get("learned_state_bytes") for a in aggs.values()
                                            if a.get("learned_state_bytes")})}

    # Q4 ---------------------------------------------------------------------------
    ev = [paired(idx, t, "D: PCA + sha256d multires bucket B=16", "B: PCA + sha256d multires bits") for t in tnames]
    ev = [p for p in ev if p]
    worse4 = [p for p in ev if p["significant"] and p["mean_diff"] < 0]
    a_b = aggs.get((tnames[0], "D: PCA + sha256d multires bucket B=16"), {})
    a_d = aggs.get((tnames[0], "B: PCA + sha256d multires bits"), {})
    sweeps_ok = a_b.get("s9_sweeps_per_example", 1e18) <= a_d.get("s9_sweeps_per_example", 0)
    store_ok = a_b.get("storage_bytes_per_example_sparse_ids", 1e18) < a_d.get(
        "storage_bytes_per_example_dense_f32", 0)
    if ev and not worse4 and sweeps_ok and store_ok:
        q4 = "yes"
    elif len(worse4) > len(ev) / 2:
        q4 = "no"
    else:
        q4 = "inconclusive"
    q4_ev = {"pairs": ev, "s9_sweeps_bucket16": a_b.get("s9_sweeps_per_example"),
             "s9_sweeps_bits": a_d.get("s9_sweeps_per_example"),
             "storage_bytes_bucket16_ids": a_b.get("storage_bytes_per_example_sparse_ids"),
             "storage_bytes_bits_f32": a_d.get("storage_bytes_per_example_dense_f32"),
             "sweeps_ok": sweeps_ok, "storage_ok": store_ok}

    # Q5: success criteria for every SHA-256d architecture -----------------------------
    archs = sorted({rep for (_, rep) in idx if "sha256d" in rep and not rep.startswith(
        ("current", "PCA + current", "B-scale"))})
    crit = {}
    for rep in archs:
        base = PCA_CURRENT if "PCA" in rep else CURRENT
        c: dict[str, Any] = {}
        per_task = {t: paired(idx, t, rep, base) for t in tnames}
        sig_gain = {t for t, p in per_task.items() if p and p["significant"] and p["mean_diff"] >= 0.01}
        hard = [t for t in tnames if t.startswith(("nexttoken", "context"))]
        c["1_better_than_current"] = bool(hard) and all(t in sig_gain for t in hard)
        c["2_better_on_context"] = ctx_task in sig_gain
        rc = aggs.get((ctx_task, rep), {})
        c["3_feature_reuse"] = bool(rc) and rc.get("repeated_cell_rate", 0) >= 0.5 and rc.get("test_coverage", 0) >= 0.5
        comp = True
        for t in tnames:
            a = aggs.get((t, rep))
            if not a:
                comp = False
                continue
            ctrl = [aggs.get((t, n), {}).get("score", 0) for n in (
                "PCA + random Fourier features", "PCA + random sign projections (LSH bits)")]
            twin = aggs.get((t, rep.replace("sha256d", "splitmix")), {}).get("score", 0)
            if a["score"] < max(ctrl + [twin]) - 0.01:
                comp = False
        c["4_competitive_with_cheap_controls"] = comp
        a0 = aggs.get((tnames[-1], rep), {})
        tw = aggs.get((tnames[-1], rep.replace("sha256d", "splitmix")), {})
        s9_ms = a0.get("s9_feature_ms_per_example")
        cpu_ms = tw.get("cacheless_feature_us_per_example", 0) / 1e3 if tw else None
        c["5_asic_path"] = bool(s9_ms is not None and cpu_ms is not None and s9_ms < cpu_ms)
        c["5_detail"] = {"s9_modelled_ms_per_example": s9_ms, "cpu_cheap_hash_ms_per_example_1core": cpu_ms}
        c["6_no_dense_hidden_network"] = True  # by construction: learned state = integer indices
        c["all"] = all(v for k, v in c.items() if k[0].isdigit() and not k.endswith("detail"))
        crit[rep] = c
    predictive = [r for r, c in crit.items() if all(c[k] for k in (
        "1_better_than_current", "2_better_on_context", "3_feature_reuse", "4_competitive_with_cheap_controls"))]
    full = [r for r, c in crit.items() if c["all"]]
    if q1 == "yes" and full:
        q5 = "yes"
    elif q1 != "no" and predictive:
        q5 = "not yet"
    else:
        q5 = "no"
    rec = "A" if q5 == "yes" else ("C" if q1 == "no" else "B")
    return {
        "Q1_sha_beats_cheap": {"answer": q1, "evidence": q1_ev},
        "Q2_multires_recovers_locality": {"answer": q2, "evidence": q2_ev},
        "Q3_routing_worth_it": {"answer": q3, "evidence": q3_ev},
        "Q4_events_better_interface": {"answer": q4, "evidence": q4_ev},
        "Q5_build_bm1387_backend": {"answer": q5, "evidence": {"success_criteria": crit,
                                                                "meets_predictive_criteria": predictive,
                                                                "meets_all_criteria": full}},
        "recommendation": rec,
    }


# --------------------------------------------------------------------- plots --

def make_plots(results: dict[str, dict[str, Any]], tasks: list[Task], out: Path) -> list[str]:
    aggs = {(a["task"], a["rep"]): a for res in results.values() for a in res["aggregated"]}
    names = [t.name for t in tasks]
    files = []

    def get(t: str, r: str, k: str = "score") -> float | None:
        return aggs.get((t, r), {}).get(k)

    # A: primitives on the phase-2/3 layout, per task
    prims = ("sha256d", "splitmix", "pcg32", "fnv1a")
    reps = {t.name: ("A: {} single t2l4" if not t.context_dim else "A: PCA + {} ctx-multires bits") for t in tasks}
    vals = [[get(t, reps[t].format(p)) for t in names] for p in prims]
    errs = [[get(t, reps[t].format(p), "score_std") or 0 for t in names] for p in prims]
    if any(v is not None for row in vals for v in row):
        (out / "plot_hash_ablation.svg").write_text(grouped_bars(
            "Track A: same layout, different hash primitive", names, list(prims), vals, errs,
            "score (accuracy / teacher agreement)"))
        files.append("plot_hash_ablation.svg")
    # B: scaling
    for t in tasks:
        lay = "context_multires" if t.context_dim else "multires"
        xs = list(multiresolution.SCALING_NODES)
        ser = {
            "single t2l4 (sha256d)": [get(t.name, f"B-scale: sha256d single t2l4, {n} nodes") for n in xs],
            f"{lay} (sha256d)": [get(t.name, f"B-scale: sha256d {lay}, {n} nodes") for n in xs],
            f"{lay} (splitmix)": [get(t.name, f"B-scale: splitmix {lay}, {n} nodes") for n in xs],
        }
        if all(v is not None for vs in ser.values() for v in vs):
            fn = f"plot_scaling_{t.name}.svg"
            (out / fn).write_text(lines(f"Track B scaling, {t.name}", [n * 16 for n in xs], ser,
                                        "hash features (= evaluations per example)", "score"))
            files.append(fn)
    # locality hypothesis: repeated-cell rate vs score, all hashed rows
    for t in tasks:
        sha, cheap = [], []
        for (tn, r), a in aggs.items():
            if tn == t.name and a.get("repeated_cell_rate") is not None and "score" in a:
                (sha if "sha256d" in r or r.startswith(("current", "PCA + current", "phase3")) else cheap).append(
                    (a["repeated_cell_rate"], a["score"], r))
        if sha:
            fn = f"plot_cells_{t.name}.svg"
            (out / fn).write_text(scatter(f"Cell reuse vs score, {t.name}", {"SHA-256d rows": sha,
                                                                              "cheap-hash rows": cheap},
                                          "repeated-cell rate (train)", "score"))
            files.append(fn)
    # D: S9 modelled cost vs score
    t = tasks[-1]
    pts = []
    for (tn, r), a in aggs.items():
        if tn == t.name and a.get("s9_feature_ms_per_example") and "score" in a:
            pts.append((a["s9_feature_ms_per_example"], a["score"], r))
    if pts:
        (out / "plot_s9_cost.svg").write_text(scatter(
            f"Modelled S9 feature time vs score, {t.name}", {"SHA-256d rows": pts},
            "modelled S9 ms per example (log)", "score", xfmt=".3g", logx=True))
        files.append("plot_s9_cost.svg")
    return files


# -------------------------------------------------------------------- report --

def report_markdown(results: dict[str, dict[str, Any]], tasks: list[Task], summary: dict[str, Any],
                    plots: list[str]) -> str:
    env = next(iter(results.values()))["environment"]
    meta = next(iter(results.values()))["meta"]
    out = ["# HashMind Phase 4: generated report", "",
           f"Generated {env['time_utc']}, commit `{env['git_commit'][:10]}`, numpy {env['numpy']}, "
           f"python {env['python']}. Model: `{meta.get('model')}` ({meta.get('model_name')}).", "",
           "Answers below are computed by the pre-registered rules in `hashmind/experiments/phase4.py`.", "",
           "| question | answer |", "|---|---|"]
    for k, v in summary["answers"].items():
        if isinstance(v, dict):
            out.append(f"| {k} | **{v['answer']}** |")
    out += [f"| recommendation | **{summary['answers']['recommendation']}** |", "",
            "CPU single-core evaluations/s (measured): " + ", ".join(
                f"{k} {v:,.0f}" for k, v in summary["cpu_evals_per_s_single_core"].items()), ""]
    for f in plots:
        out += [f"![{f}]({f})", ""]
    for name, res in results.items():
        out += [f"## {res['title']}", ""]
        for t in tasks:
            aggs = [a for a in res["aggregated"] if a["task"] == t.name]
            out += [f"### {t.name}", "", table(aggs, t.kind), ""]
    return "\n".join(out)


def run_all(tasks: list[Task], meta: dict[str, Any], out: str | Path, seeds: tuple[int, ...] = DEFAULT_SEEDS,
            tracks: tuple[str, ...] = tuple(TRACKS), logf: Callable[[str], None] = log) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = RowCache(logf, out / "rows_cache.jsonl")
    results = {}
    for name in tracks:
        logf(f"=== {TRACK_TITLES[name]} ===")
        results[name] = run_track(name, tasks, seeds, cache, out, meta)
    if set(tracks) != set(TRACKS):
        return {"results": results}
    answers = analyse(results, tasks)
    plots = make_plots(results, tasks, out)
    summary = {"environment": environment(), "meta": meta, "seeds": list(seeds),
               "tasks": {t.name: t.info for t in tasks}, "cpu_evals_per_s_single_core": dict(_THROUGHPUT),
               "answers": answers, "plots": plots}
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(report_markdown(results, tasks, summary, plots))
    return {"results": results, "summary": summary}
