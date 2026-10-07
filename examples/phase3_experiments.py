"""Run all phase-3 studies on a llama GGUF and write docs/results/phase3_*.{json,md}.

    python examples/phase3_experiments.py model.gguf --cache hidden.npz [--quick]

1. locality vs input-space size (cut layer 11 and embedding layer with context)
2. replace the top of the transformer: next-token prediction from hidden states
3. BM1387 difficulty floor: scaled-difficulty sweep + S9 cost model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hashmind.backends.s9_model import feature_cost
from hashmind.conversion.weights import pca_basis
from hashmind.experiments.next_token import (
    HMConfig,
    NextTokenBench,
    collect_hidden_states,
    format_rows,
    rows_to_dicts,
    run_hashmind,
    run_linear,
)
from hashmind.experiments.s9_difficulty import as_dicts, sweep
from hashmind.experiments.token_probe import vocab_tasks
from hashmind.gguf.reader import read_gguf
from hashmind.llm.llama import LlamaReference


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--cache", required=True, help="npz cache for hidden states")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--only", choices=["locality", "nexttoken", "difficulty"], action="append")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = set(args.only or ["locality", "nexttoken", "difficulty"])

    g = read_gguf(args.gguf)
    if todo & {"locality", "nexttoken"}:
        log("collecting hidden states (cached)")
        hs = collect_hidden_states(args.gguf, args.n_seq, args.seq_len, cache=args.cache)
        log(f"hidden states: tokens {hs.tokens.shape}, forward {hs.forward_seconds:.0f}s")
        bench = NextTokenBench(hs, LlamaReference(g).lm_head())
        ref_rows = [bench.teacher_row(), *bench.ngram_rows()]

    if "nexttoken" in todo:
        rows = list(ref_rows)
        for L in sorted(hs.hidden):
            log(f"next-token: cut {L}")
            rows.append(run_linear(bench, L, 64))
            rows.append(run_hashmind(bench, L, HMConfig(f"HashMind t=2 l=4 on PCA-64(h_{L})")))
            rows.append(run_hashmind(bench, L, HMConfig(f"PCA-64 + HashMind t=2 l=4 (h_{L})",
                                                        concat_linear=True)))
        (out / "phase3_nexttoken.json").write_text(json.dumps(rows_to_dicts(rows), indent=2))
        (out / "phase3_nexttoken.md").write_text(format_rows(rows) + "\n")
        log("\n" + format_rows(rows))

    if "locality" in todo:
        rows = list(ref_rows)
        L = 11
        rows.append(run_linear(bench, L, 64))
        rows.append(run_linear(bench, L, None))
        for cfg in [
            HMConfig("t=2 l=4 quantile"),
            HMConfig("t=2 l=4 supervised", quantizer="supervised"),
            HMConfig("t=4 l=4 quantile", tuple_size=4),
            HMConfig("t=4 l=4 supervised", tuple_size=4, quantizer="supervised"),
            HMConfig("t=8 l=2 quantile", tuple_size=8, levels=2),
            HMConfig("t=8 l=2 supervised", tuple_size=8, levels=2, quantizer="supervised"),
            HMConfig("t=12 l=2 supervised", tuple_size=12, levels=2, quantizer="supervised"),
        ]:
            log(f"locality: {cfg.name}")
            cfg.name = f"HashMind {cfg.name} (h_{L})"
            rows.append(run_hashmind(bench, L, cfg))
        L = 0
        rows.append(run_linear(bench, L, 64))
        for cfg in [
            HMConfig("t=2 l=4, no context"),
            HMConfig("t=1 l=4 + 1 ctx symbol of [tok_t, tok_t-1]", tuple_size=1, context_cols=2,
                     context_per_node=1),
            HMConfig("ctx only: 2 symbols of [tok_t, tok_t-1]", tuple_size=0, context_cols=2,
                     context_per_node=2),
            HMConfig("ctx only: 2 of [tok_t, tok_t-1, tok_t-2]", tuple_size=0, context_cols=3,
                     context_per_node=2),
            HMConfig("PCA-64 + ctx 2 of [tok_t, tok_t-1, tok_t-2]", tuple_size=0, context_cols=3,
                     context_per_node=2, concat_linear=True),
            HMConfig("PCA-64 + ctx 2 of 3, 8192 features", tuple_size=0, context_cols=3,
                     context_per_node=2, concat_linear=True, output_dim=8192),
        ]:
            log(f"locality: {cfg.name}")
            cfg.name = f"HashMind {cfg.name} (h_{L})"
            rows.append(run_hashmind(bench, L, cfg))
        (out / "phase3_locality.json").write_text(json.dumps(rows_to_dicts(rows), indent=2))
        (out / "phase3_locality.md").write_text(format_rows(rows) + "\n")
        log("\n" + format_rows(rows))

    if "difficulty" in todo:
        E = g.tensor("token_embd.weight")
        tasks = vocab_tasks(list(g.metadata["tokenizer.ggml.tokens"]))
        ids, y = tasks["word_start"]
        sel = np.sort(np.random.default_rng(0).choice(len(ids), 3000, replace=False))
        ids, y = ids[sel], y[sel]
        B, m, _ = pca_basis(E, 32)
        Z = ((E[ids] - m) @ B).astype(np.float32)
        settings = [{"mode": "hash_bits", "difficulty_bits": 1, "window": 1}]
        settings += [{"mode": "threshold", "difficulty_bits": d, "window": 2**d} for d in (1, 4, 8, 12)]
        settings += [{"mode": "threshold", "difficulty_bits": 8, "window": 16}]  # sparse: lambda = 1/16
        settings += [{"mode": "nonce_bits", "difficulty_bits": d, "window": 2**d, "bits_per_hash": b,
                      "features_per_node": 16} for d, b in ((4, 3), (8, 6), (12, 8))]
        log("difficulty sweep")
        runs = sweep(Z, y, settings)
        costs = [feature_cost("threshold").to_dict()] + [
            feature_cost("nonce_bits", nonce_bits=k).to_dict() for k in (8, 16, 24)]
        (out / "phase3_difficulty.json").write_text(json.dumps({"runs": as_dicts(runs), "s9_costs": costs},
                                                               indent=2))
        lines = ["| mode | difficulty bits | window | features | test acc | density | SHA-256d executed | s |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {r.mode} | {r.difficulty_bits} | {r.window} | {r.features} | {r.accuracy:.1%} | "
                  f"{r.density:.3f} | {r.sha256d_executed:,} | {r.seconds:.0f} |" for r in runs]
        lines += ["", "S9 cost model at the BM1387 floor (difficulty 1 = 32 zero bits), 2048 features/token:", "",
                  "| mode | features/window | p(share in window) | features/s | s/token | jobs/token |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {c['mode']} | {c['features_per_window']} | {c['p_share_in_window']:.3f} | "
                  f"{c['features_per_second']:,.0f} | {c['seconds_per_token']:.3f} | {c['jobs_per_token']:.0f} |"
                  for c in costs]
        (out / "phase3_difficulty.md").write_text("\n".join(lines) + "\n")
        log("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
