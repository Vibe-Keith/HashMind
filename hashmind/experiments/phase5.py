"""Phase 5: frozen-model conversion study.

    python -m hashmind experiment phase5 model.gguf -o docs/results/phase5

The source GGUF is the reference implementation. Nothing is trained, fitted or
calibrated; every converted model is a deterministic function of the GGUF
weights and fixed settings.

Protocol
--------
* Evaluation text: CPython pydoc topics (ships with Python). Development set =
  the first sequences, held-out test set = disjoint sequences from the second
  half. Fixed greedy-generation prompts, split the same way.
* 5A operation study and 5D quantization study: development set only.
* Selection rule (fixed before running): for each operation class pick the
  SHA-256d conversion with the highest dev top-1 agreement; candidates within
  0.5 points count as tied and the one with less host work wins.
* 5C partial and full conversions use only the selected settings and are
  evaluated once on the test set.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.primitives import get_primitive, measure_throughput
from ..frozen.convert_ops import LINEAR_OPS, OP_CLASSES, OpConv
from ..frozen.hardware import ASSUMPTIONS, break_even, end_to_end
from ..frozen.hashsrc import HashSource
from ..frozen.hmfrozen import convert_frozen, file_sha256, load_frozen, quantize_frozen
from ..frozen.metrics import generation_fidelity, logit_fidelity
from ..frozen.runtime import FrozenRuntime, FrozenWeights
from ..gguf.reader import read_gguf
from ..llm.tokenizer import SPMTokenizer
from .next_token import default_corpus
from .phase4_common import environment, write_json

PROMPTS_DEV = ["The capital of France is", "def fibonacci(n):", "Once upon a time, there was a",
               "The three primary colors are"]
PROMPTS_TEST = ["The largest planet in the solar system is", "import numpy as np\n", "Water boils at a temperature of",
                "In Python, a list comprehension", "The quick brown fox", "The president of the United States",
                "To install a package with pip, run", "A triangle has three"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- data -------

class EvalData:
    def __init__(self, tok: SPMTokenizer, n_dev: int = 4, n_test: int = 16, seq_len: int = 64,
                 n_new: int = 16) -> None:
        ids = tok.encode(default_corpus(), bos=False)
        L = seq_len - 1
        half = len(ids) // 2

        def seqs(start: int, n: int) -> np.ndarray:
            body = np.asarray(ids[start:start + n * L]).reshape(n, L)
            return np.concatenate([np.full((n, 1), tok.bos_id), body], 1)

        self.dev, self.test = seqs(0, n_dev), seqs(half, n_test)
        self.prompts_dev = [tok.encode(p) for p in PROMPTS_DEV]
        self.prompts_test = [tok.encode(p) for p in PROMPTS_TEST]
        self.n_new = n_new
        self.tok = tok
        self.info = {"corpus": "pydoc_data.topics", "corpus_tokens": len(ids), "seq_len": seq_len,
                     "dev_sequences": n_dev, "test_sequences": n_test, "dev_offset": 0, "test_offset": half,
                     "prompts_dev": PROMPTS_DEV, "prompts_test": PROMPTS_TEST, "new_tokens": n_new}

    def seqs(self, split: str) -> np.ndarray:
        return self.dev if split == "dev" else self.test

    def prompts(self, split: str) -> list[list[int]]:
        return self.prompts_dev if split == "dev" else self.prompts_test


class Reference:
    """Original-model outputs: teacher-forced logits and greedy generations, plus timing."""

    def __init__(self, w: FrozenWeights, data: EvalData, splits: tuple[str, ...] = ("dev", "test")) -> None:
        rt = FrozenRuntime(w)
        self.logits, self.gen, self.time = {}, {}, {}
        for s in splits:
            x = data.seqs(s)
            t0 = time.perf_counter()
            self.logits[s] = rt.forward(x)[:, :-1].astype(np.float32)
            self.time[s + "_teacher_forced_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.gen[s] = rt.generate(data.prompts(s), data.n_new)
            self.time[s + "_generation_s"] = time.perf_counter() - t0
        rt.reset_counts()
        rt.forward(data.seqs("dev")[:1])
        self.macs_per_token = sum(v.get("fp_macs", 0) for v in rt.cv.counts.values()) / data.seqs("dev").shape[1]


# -------------------------------------------------------------- evaluation ----

def evaluate(w: FrozenWeights, conv: dict[str, OpConv], data: EvalData, ref: Reference, split: str,
             probe: bool = False, generate: bool = False, name: str = "") -> dict[str, Any]:
    rt = FrozenRuntime(w, conv)
    x = data.seqs(split)
    t0 = time.perf_counter()
    lg = rt.forward(x, probe=probe)[:, :-1]
    tf = time.perf_counter() - t0
    ntok = x.shape[0] * x.shape[1]
    counts = {op: {k: v / ntok for k, v in c.items()} for op, c in rt.cv.counts.items()}
    hashes = {k: v / ntok for k, v in rt.hash_evaluations().items()}
    row = {"name": name, "split": split, "conv": {k: v.label() for k, v in conv.items()},
           "fidelity": logit_fidelity(ref.logits[split], lg, x[:, 1:]),
           "teacher_forced_s": tf, "s_per_token_batched": tf / ntok,
           "ops_per_token": counts, "hash_evals_per_token": hashes,
           "hash_evals_per_token_total": float(sum(hashes.values()))}
    if probe and rt.probe:
        row["tensor_error"] = {k: v.summary() for k, v in rt.probe.items()}
    if generate:
        rt.reset_counts()
        t0 = time.perf_counter()
        g = rt.generate(data.prompts(split), data.n_new)
        row["generation_s"] = time.perf_counter() - t0
        row["generation"] = generation_fidelity(ref.gen[split], g)
        row["generated_text"] = [data.tok.decode(s) for s in g]
    return row


def host_work(row: dict[str, Any]) -> float:
    c = row["ops_per_token"]
    return sum(v.get("fp_macs", 0) + v.get("int_macs", 0) * 0.5 + v.get("adds", 0) * 0.25 + v.get("lut_lookups", 0)
               for v in c.values())


# ------------------------------------------------------------- candidates ----

def op_candidates(op: str) -> list[OpConv]:
    if op in LINEAR_OPS:
        c = [OpConv("quant", b, "rtn") for b in (2, 4, 8)] + [OpConv("quant", b, "sha256d") for b in (2, 4, 8)]
        c += [OpConv("quant", 4, "splitmix")]
        c += [OpConv("sampled", samples=s, primitive=p) for s in (256, 1024) for p in ("sha256d", "splitmix")]
        c += [OpConv("topk", samples=s) for s in (256, 1024)]
        return c
    if op in ("attn_scores", "attn_values"):
        return [OpConv("quant", b, "rtn") for b in (2, 4, 8)] + [OpConv("quant", b, "sha256d") for b in (2, 4, 8)] + \
            [OpConv("quant", 4, "splitmix")]
    if op in ("act", "softmax"):
        return [OpConv("lut", b) for b in (4, 6, 8)] + \
            [OpConv("lut", b, slots=2 ** (b - 1), primitive="sha256d") for b in (4, 6, 8)] + \
            [OpConv("lut", 6, slots=32, primitive="splitmix")]
    if op == "embedding":
        return [OpConv("hashed", slots=m, primitive="sha256d") for m in (32000, 128000, 512000)] + \
            [OpConv("hashed", slots=128000, primitive="splitmix")]
    raise ValueError(op)


def select(op_rows: list[dict[str, Any]], op: str) -> OpConv | None:
    sha = [r for r in op_rows if r["op"] == op and r["uses_sha"]]
    if not sha:
        return None
    best = max(r["fidelity"]["top1_agreement"] for r in sha)
    tied = [r for r in sha if r["fidelity"]["top1_agreement"] >= best - 0.005]
    return OpConv(**min(tied, key=host_work)["opconv"])


PARTIAL_GROUPS = {
    "embedding only": ("embedding",),
    "attention only": ("attn_qkv", "attn_scores", "softmax", "attn_values", "attn_out"),
    "attention projections only": ("attn_qkv", "attn_out"),
    "MLP only": ("mlp_in", "act", "mlp_down"),
    "MLP projections only": ("mlp_in", "mlp_down"),
    "nonlinearities only (SiLU + softmax exp)": ("act", "softmax"),
    "LM head only": ("lm_head",),
    "attention + MLP": ("attn_qkv", "attn_scores", "softmax", "attn_values", "attn_out", "mlp_in", "act", "mlp_down"),
    "all operation classes": OP_CLASSES,
}


def to_splitmix(c: OpConv) -> OpConv:
    d = asdict(c)
    if d["rounding"] == "sha256d":
        d["rounding"] = "splitmix"
    if d["primitive"] == "sha256d":
        d["primitive"] = "splitmix"
    return OpConv(**d)


# ------------------------------------------------------------------- run ------

def run_phase5(gguf: str | Path, out: str | Path, logf: Callable[[str], None] = log,
               n_test: int = 16, quick: bool = False) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g = read_gguf(gguf)
    tok = SPMTokenizer.from_gguf_metadata(g.metadata)
    data = EvalData(tok, n_dev=2 if quick else 4, n_test=4 if quick else n_test, n_new=4 if quick else 16)
    logf("dequantizing source GGUF")
    w = FrozenWeights.from_gguf(g)
    src = {"file": Path(gguf).name, "sha256": file_sha256(gguf), "name": g.metadata.get("general.name"),
           "architecture": g.metadata.get("general.architecture"), "file_type": g.metadata.get("general.file_type"),
           "hparams": asdict(w.hp), "tensor_types": sorted({t.type_name for t in g.tensors.values()})}
    meta = {"environment": environment(), "source": src, "data": data.info, "assumptions": ASSUMPTIONS}
    logf("reference outputs")
    ref = Reference(w, data)
    meta["reference_timing"] = ref.time
    meta["reference_macs_per_token"] = ref.macs_per_token
    t_ref = ref.time["test_teacher_forced_s"] / data.test.size
    t_ref_gen = ref.time["test_generation_s"] / (data.n_new * len(data.prompts_test))
    meta["cpu_hash_evals_per_s_single_core"] = {p: measure_throughput(get_primitive(p)) for p in ("sha256d", "splitmix")}

    # 5A ---------------------------------------------------------------------------
    op_rows = []
    for op in OP_CLASSES:
        for c in op_candidates(op):
            r = evaluate(w, {op: c}, data, ref, "dev", probe=True, name=f"{op}: {c.label()}")
            r.update({"op": op, "opconv": asdict(c), "uses_sha": c.hash_primitive == "sha256d",
                      "primitive": c.hash_primitive})
            op_rows.append(r)
            f = r["fidelity"]
            logf(f"5A {r['name']:40s} top1 {f['top1_agreement']:.3f} KL {f['kl_ref_to_conv']:.3f}")
    selected = {op: select(op_rows, op) for op in OP_CLASSES}
    selected = {k: v for k, v in selected.items() if v is not None}
    write_json(out / "operation_conversion.json", {**meta, "rows": op_rows,
                                                   "selection_rule": __doc__.split("Selection rule")[1].split("*")[0],
                                                   "selected": {k: v.label() for k, v in selected.items()}})

    # 5D ---------------------------------------------------------------------------
    q_rows = []
    wbits = (2, 4, 8, 16)
    for bits in wbits:
        for rnd in ("rtn", "splitmix", "sha256d"):
            if quick and rnd == "sha256d" and bits != 4:
                continue
            t0 = time.perf_counter()
            qw, _ = quantize_frozen(w, bits, rnd)
            conv_s = time.perf_counter() - t0
            r = evaluate(qw, {}, data, ref, "dev", name=f"weights {bits}-bit {rnd}")
            r.update({"weight_bits": bits, "weight_rounding": rnd, "act_bits": None, "conversion_s": conv_s,
                      "weight_hash_evals": int(sum(m.size for _, m in w.matrices())) if rnd != "rtn" else 0})
            q_rows.append(r)
            del qw
            logf(f"5D {r['name']:32s} top1 {r['fidelity']['top1_agreement']:.3f}")
    for bits in wbits:
        for rnd in ("rtn", "splitmix", "sha256d"):
            c = OpConv("quant", bits, rnd)
            r = evaluate(w, {op: c for op in LINEAR_OPS}, data, ref, "dev", name=f"activations {bits}-bit {rnd}")
            r.update({"weight_bits": None, "act_bits": bits, "act_rounding": rnd})
            q_rows.append(r)
            logf(f"5D {r['name']:32s} top1 {r['fidelity']['top1_agreement']:.3f}")
    for rnd in ("rtn", "sha256d"):
        qw, _ = quantize_frozen(w, 4, rnd)
        r = evaluate(qw, {op: OpConv("quant", 8, rnd) for op in LINEAR_OPS}, data, ref, "dev",
                     name=f"W4A8 {rnd}")
        r.update({"weight_bits": 4, "act_bits": 8, "weight_rounding": rnd, "act_rounding": rnd})
        q_rows.append(r)
        del qw
        logf(f"5D {r['name']:32s} top1 {r['fidelity']['top1_agreement']:.3f}")
    write_json(out / "quantization.json", {**meta, "rows": q_rows})

    # 5C (test) ---------------------------------------------------------------------
    p_rows = [evaluate(w, {}, data, ref, "test", generate=True, name="original (reference runtime)")]
    for gname, ops in PARTIAL_GROUPS.items():
        conv = {op: selected[op] for op in ops if op in selected}
        r = evaluate(w, conv, data, ref, "test", generate=True, name=f"SHA-256d: {gname}")
        r["group"] = gname
        p_rows.append(r)
        logf(f"5C {r['name']:44s} top1 {r['fidelity']['top1_agreement']:.3f}")
        if gname in ("MLP only", "all operation classes"):
            r2 = evaluate(w, {k: to_splitmix(v) for k, v in conv.items()}, data, ref, "test", generate=True,
                          name=f"splitmix control: {gname}")
            r2["group"] = gname
            p_rows.append(r2)
            logf(f"5C {r2['name']:44s} top1 {r2['fidelity']['top1_agreement']:.3f}")
    write_json(out / "partial_conversion.json", {**meta, "selected": {k: v.label() for k, v in selected.items()},
                                                 "rows": p_rows})

    # Full conversion through .hmmodel (test) -----------------------------------------
    f_rows = []
    hm_dir = out / "hmmodel_tmp"
    hm_dir.mkdir(exist_ok=True)
    full_specs = [("W8 rtn, no op conversion (quantization only)", 8, "rtn", {}),
                  ("W8 sha256d + all selected SHA conversions", 8, "sha256d", selected),
                  ("W4 sha256d + all selected SHA conversions", 4, "sha256d", selected)]
    for name, bits, rnd, conv in full_specs:
        path = hm_dir / f"w{bits}_{rnd}_{len(conv)}.hmmodel"
        t0 = time.perf_counter()
        man = convert_frozen(gguf, path, bits, rnd, conv, weights=w)
        conv_s = time.perf_counter() - t0
        reproduce = None
        if rnd == "rtn":
            p2 = hm_dir / "repeat.hmmodel"
            convert_frozen(gguf, p2, bits, rnd, conv, weights=w)
            reproduce = file_sha256(p2) == man["hmmodel_sha256"]
            p2.unlink()
        fw, fconv, _ = load_frozen(path)
        r = evaluate(fw, fconv, data, ref, "test", generate=True, name=name)
        r.update({"hmmodel_sha256": man["hmmodel_sha256"], "hmmodel_bytes": path.stat().st_size,
                  "conversion_s": conv_s, "byte_identical_on_reconversion": reproduce,
                  "source_sha256": man["source"]["sha256"], "conversion": man["conversion"]})
        f_rows.append(r)
        del fw
        path.unlink()
        logf(f"FULL {name:46s} top1 {r['fidelity']['top1_agreement']:.3f}")
    # trivial hash control: logits are SHA-256d output of the context fingerprint
    hs = HashSource("sha256d")
    x = data.test
    V = ref.logits["test"].shape[-1]
    pos = np.broadcast_to(np.arange(x.shape[1] - 1), (x.shape[0], x.shape[1] - 1))
    ctx = (x[:, :-1].astype(np.uint64) << np.uint64(16)) | pos.astype(np.uint64)  # (token id, position)
    rnd_logits = hs.uniform(0x7A11D, ctx.reshape(-1), V)
    f_rows.append({"name": "trivial control: SHA-256d random logits", "split": "test",
                   "fidelity": logit_fidelity(ref.logits["test"], rnd_logits.reshape(x.shape[0], -1, V), x[:, 1:])})
    hm_dir.rmdir()
    write_json(out / "full_conversion.json", {**meta, "rows": f_rows})

    # 5E/5F -------------------------------------------------------------------------
    hw = {"assumptions": ASSUMPTIONS, "measured": {
        "original_cpu_s_per_token_batched": t_ref, "original_cpu_s_per_token_generation": t_ref_gen,
        "reference_macs_per_token": ref.macs_per_token, "cpu_hash_evals_per_s_single_core":
            meta["cpu_hash_evals_per_s_single_core"]}}
    mac_rate = ref.macs_per_token / t_ref
    hw["break_even"] = break_even(mac_rate, meta["cpu_hash_evals_per_s_single_core"]["sha256d"])
    weight_bytes = {16: 2, 8: 1, 4: 0.5}
    nparams = sum(m.size for _, m in w.matrices())
    hw["per_configuration"] = []
    for r in p_rows[1:] + f_rows[1:3]:
        if "ops_per_token" not in r:
            continue
        counts = {k: sum(c.get(k, 0) for c in r["ops_per_token"].values())
                  for k in ("fp_macs", "int_macs", "adds", "lut_lookups")}
        e2e = end_to_end(counts, r["hash_evals_per_token_total"], t_ref, r["s_per_token_batched"],
                         ref.macs_per_token, meta["cpu_hash_evals_per_s_single_core"]["sha256d"],
                         nparams * weight_bytes[4])
        saved = ref.macs_per_token - (counts["fp_macs"] + counts["int_macs"])
        hw["per_configuration"].append({
            "name": r["name"], "top1_agreement": r["fidelity"]["top1_agreement"],
            "hash_evals_per_token": r["hash_evals_per_token_total"], "host_ops_per_token": counts,
            "fp_macs_removed_per_token": saved,
            "macs_removed_per_hash": saved / r["hash_evals_per_token_total"] if r["hash_evals_per_token_total"] else None,
            "end_to_end": e2e})
    write_json(out / "hardware_model.json", hw)

    summary = summarize(op_rows, q_rows, p_rows, f_rows, hw, selected, meta)
    write_json(out / "summary.json", summary)
    (out / "REPORT.md").write_text(report(op_rows, q_rows, p_rows, f_rows, hw, summary))
    return summary


# ---------------------------------------------------------------- reporting ----

def summarize(op_rows, q_rows, p_rows, f_rows, hw, selected, meta) -> dict[str, Any]:
    """Mechanical facts the PHASE5.md conclusion is drawn from (no verdict here)."""
    by_op = {}
    for op in OP_CLASSES:
        rows = [r for r in op_rows if r["op"] == op]
        sha = [r for r in rows if r["uses_sha"]]
        nonsha = [r for r in rows if not r["uses_sha"] and r["primitive"] != "splitmix"]
        sm = [r for r in rows if r["primitive"] == "splitmix"]
        by_op[op] = {
            "best_sha": max((r["fidelity"]["top1_agreement"] for r in sha), default=None),
            "best_non_hash": max((r["fidelity"]["top1_agreement"] for r in nonsha), default=None),
            "best_splitmix": max((r["fidelity"]["top1_agreement"] for r in sm), default=None),
            "selected": selected[op].label() if op in selected else None,
        }
    # paired SHA vs splitmix (same conversion, primitive swapped) across all studies
    pairs = []
    allrows = op_rows + q_rows + p_rows + f_rows
    names = {r["name"]: r for r in allrows}
    for r in allrows:
        n = r["name"]
        for a, b in (("sha256d", "splitmix"), ("SHA-256d: ", "splitmix control: ")):
            if a in n and n.replace(a, b) in names:
                pairs.append({"sha": n, "control": n.replace(a, b),
                              "top1_sha": r["fidelity"]["top1_agreement"],
                              "top1_control": names[n.replace(a, b)]["fidelity"]["top1_agreement"]})
    return {"meta": meta, "per_operation": by_op, "selected": {k: v.label() for k, v in selected.items()},
            "partial": {r["name"]: {"top1": r["fidelity"]["top1_agreement"], "kl": r["fidelity"]["kl_ref_to_conv"],
                                    "gen_exact": r.get("generation", {}).get("exact_match_rate")} for r in p_rows},
            "full": {r["name"]: {"top1": r["fidelity"]["top1_agreement"], "kl": r["fidelity"]["kl_ref_to_conv"],
                                 "gen_exact": r.get("generation", {}).get("exact_match_rate")} for r in f_rows},
            "sha_vs_splitmix_pairs": pairs, "break_even": hw["break_even"]}


def _fid_table(rows: list[dict[str, Any]], extra: bool = True) -> str:
    out = ["| configuration | top-1 agree | top-5 | top-10 | KL(ref‖conv) | logit cos | Spearman | "
           "hash evals/token | gen exact | gen prefix |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        f = r["fidelity"]
        g = r.get("generation", {})
        out.append(f"| {r['name']} | {f['top1_agreement']:.1%} | {f['top5_agreement']:.1%} | {f['top10_agreement']:.1%} | "
                   f"{f['kl_ref_to_conv']:.3f} | {f['logit_cosine']:.3f} | {f['logit_spearman']:.3f} | "
                   f"{r.get('hash_evals_per_token_total', 0):,.0f} | "
                   f"{g.get('exact_match_rate', float('nan')):.0%} | {g.get('mean_matching_prefix', float('nan')):.1f} |")
    return "\n".join(out)


def report(op_rows, q_rows, p_rows, f_rows, hw, summary) -> str:
    m = summary["meta"]
    out = ["# Phase 5 generated report", "",
           f"Source: `{m['source']['file']}` sha256 `{m['source']['sha256'][:16]}…`, {m['source']['name']}. "
           f"Commit `{m['environment']['git_commit'][:10]}`. All agreement numbers are against the original "
           "model's own logits.", "", "## 5A operation study (dev set, op class converted in all 22 layers)", ""]
    out.append("| op class | conversion | top-1 agree | KL | tensor rel. error | tensor cosine | hash evals/token |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for r in op_rows:
        te = r.get("tensor_error", {}).get(r["op"], {})
        out.append(f"| {r['op']} | {OpConv(**r['opconv']).label()} | {r['fidelity']['top1_agreement']:.1%} | "
                   f"{r['fidelity']['kl_ref_to_conv']:.3f} | {te.get('relative_error', float('nan')):.3f} | "
                   f"{te.get('cosine', float('nan')):.4f} | {r['hash_evals_per_token_total']:,.0f} |")
    out += ["", "Selected per op class (rule in phase5.py): " + ", ".join(f"{k}={v}" for k, v in summary["selected"].items()),
            "", "## 5D quantization (dev)", "", _fid_table(q_rows), "",
            "## 5C partial conversion (test)", "", _fid_table(p_rows), "", "## Full conversion via .hmmodel (test)", "",
            _fid_table(f_rows), "", "## Hardware model", "", "```", json.dumps(hw["break_even"], indent=1), "```", ""]
    out.append("| configuration | top-1 | hash evals/token | original CPU s/tok (meas.) | HashMind CPU s/tok (meas.) | "
               "S9 s/tok (model) | S9 J/tok (model) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in hw["per_configuration"]:
        e = c["end_to_end"]["per_token"]
        out.append(f"| {c['name']} | {c['top1_agreement']:.1%} | {c['hash_evals_per_token']:,.0f} | "
                   f"{e['original_cpu_measured']['s_per_token']:.3f} | {e['hashmind_cpu_sim_measured']['s_per_token']:.3f} | "
                   f"{e['hashmind_s9_modelled']['s_per_token']:,.1f} | {e['hashmind_s9_modelled']['J_per_token']:,.0f} |")
    return "\n".join(out) + "\n"
