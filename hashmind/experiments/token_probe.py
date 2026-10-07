"""Phase-2 experiment: does information from a real GGUF survive HashMind?

    GGUF model
        -> extract representation (token embedding rows, dequantized)
        -> deterministic reduction (PCA from the weight plan)
        -> HashMind layer (SHA-256d features on the CPU S9 simulator)
        -> ridge readout
        -> prediction

Tasks are derived from the model's own tokenizer vocabulary, so no external
dataset is needed. Labels are computed from the token *strings*; the model only
sees the token *embedding*. A readout can therefore only succeed if the
property is encoded in the learned embedding and survives hashing.

Controls
--------
* majority class            accuracy of always predicting the most common label
* original embedding        ridge on the full d-dim embedding (reference ceiling)
* PCA-r (pre-hash)          ridge on the HashMind input, before SHA-256
* HashMind                  ridge on SHA-256d features
* HashMind / shuffled       same pipeline, but embedding rows permuted across tokens.
                            If this is near majority, HashMind's accuracy comes from the
                            GGUF weights, not from the hash or the label distribution.
"""

from __future__ import annotations

import resource
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.layer import HashMindLayer
from ..core.readout import RidgeReadout
from ..conversion.weights import pca_basis
from ..gguf.reader import read_gguf

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
WORD_PREFIXES = ("▁", "Ġ")  # sentencepiece '▁', GPT-2 byte-level 'Ġ'


@dataclass
class ProbeConfig:
    n_samples: int = 6000
    test_frac: float = 0.25
    reduced_dim: int = 32
    output_dim: int = 2048
    feature_mode: str = "hash_bits"
    tuple_size: int = 2
    levels: int = 4
    nonces: int = 16
    seed: int = 0
    mode_sweep: bool = True
    tuple_sweep: tuple[int, ...] = (1, 3, 4)  # extra runs of the main mode at other tuple sizes
    threshold_difficulty_bits: int = 4  # threshold mode: p(feature=1) = 1/16 (sparse shares)
    neighbor_queries: int = 200
    neighbor_k: int = 10


@dataclass
class RunMetrics:
    representation: str
    feature_dim: int
    accuracy: float
    alpha: float
    fit_seconds: float
    transform_seconds: float = 0.0
    transform_us_per_sample: float = 0.0
    readout_us_per_sample: float = 0.0
    peak_mem_mb: float = 0.0
    sha256d_logical: int = 0
    sha256d_executed: int = 0
    asic_native: bool | None = None


@dataclass
class TaskResult:
    task: str
    classes: dict[str, int]
    n_train: int
    n_test: int
    majority_accuracy: float
    runs: list[RunMetrics] = field(default_factory=list)


# ---------------------------------------------------------------- labels ----

def _is_special(tok: str) -> bool:
    return (tok.startswith("<") and tok.endswith(">")) or tok == ""


def vocab_tasks(tokens: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Label tasks computable from token strings. Returns {task: (token_ids, labels)}."""
    ids, word, cls = [], [], []
    for i, t in enumerate(tokens):
        if _is_special(t):
            continue
        body = t[1:] if t.startswith(WORD_PREFIXES) else t
        if not body:
            continue
        ids.append(i)
        word.append(int(t.startswith(WORD_PREFIXES)))
        if body.isdigit():
            cls.append("digit")
        elif body.isalpha() and body.islower():
            cls.append("lowercase")
        elif body.isalpha() and body[0].isupper():
            cls.append("capitalized")
        else:
            cls.append("punct/other")
    a = np.asarray(ids)
    return {"word_start": (a, np.asarray(word)), "char_class": (a, np.asarray(cls))}


# ---------------------------------------------------------------- helpers ---

def _split(n: int, test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = rng.permutation(n)
    k = int(round(n * test_frac))
    return p[k:], p[:k]


def _fit_best(F_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> tuple[RidgeReadout, float]:
    """Pick ridge alpha on an inner validation split of the training set, then refit."""
    rng = np.random.default_rng(seed)
    tr, va = _split(len(y_tr), 0.2, rng)
    best, best_acc = ALPHAS[0], -1.0
    for a in ALPHAS:
        r = RidgeReadout(a).fit_classes(F_tr[tr], y_tr[tr])
        acc = float((r.predict_classes(F_tr[va]) == y_tr[va]).mean())
        if acc > best_acc:
            best, best_acc = a, acc
    return RidgeReadout(best).fit_classes(F_tr, y_tr), best


def _evaluate(name: str, F: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
              seed: int, **extra: Any) -> RunMetrics:
    t0 = time.perf_counter()
    model, alpha = _fit_best(F[tr], y[tr], seed)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict_classes(F[te])
    ro_us = (time.perf_counter() - t0) / len(te) * 1e6
    return RunMetrics(name, F.shape[1], float((pred == y[te]).mean()), alpha, fit_s,
                      readout_us_per_sample=ro_us, **extra)


def _hashmind_features(Z: np.ndarray, cfg: ProbeConfig, mode: str,
                       tuple_size: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    layer = HashMindLayer(Z.shape[1], cfg.output_dim, seed=cfg.seed, feature_mode=mode,
                          tuple_size=tuple_size or cfg.tuple_size, levels=cfg.levels, nonces_per_node=cfg.nonces,
                          difficulty_bits=cfg.threshold_difficulty_bits if mode == "threshold" else 1)
    tracemalloc.start()
    t0 = time.perf_counter()
    F = layer.fit_transform(Z)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return F, {
        "transform_seconds": dt,
        "transform_us_per_sample": dt / len(Z) * 1e6,
        "peak_mem_mb": peak / 1e6,
        "sha256d_logical": layer.stats.sha256d_logical,
        "sha256d_executed": layer.stats.sha256d_executed,
        "asic_native": layer.asic_native,
    }


def neighbor_recall(ref: np.ndarray, test: np.ndarray, queries: np.ndarray, k: int) -> float:
    """Mean overlap of top-k cosine neighbours in ``ref`` vs ``test`` space."""

    def unit(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(0)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    a, b = unit(ref.astype(np.float64)), unit(test.astype(np.float64))
    tot = 0.0
    for q in queries:
        sa, sb = a @ a[q], b @ b[q]
        sa[q] = sb[q] = -np.inf
        na = set(np.argpartition(-sa, k)[:k].tolist())
        nb = set(np.argpartition(-sb, k)[:k].tolist())
        tot += len(na & nb) / k
    return tot / len(queries)


# ---------------------------------------------------------------- main ------

def run_token_probe(
    gguf_path: str | Path | None = None,
    cfg: ProbeConfig | None = None,
    embeddings: np.ndarray | None = None,
    tasks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run the probe. Pass ``gguf_path`` (real model) or ``embeddings`` + ``tasks``."""
    cfg = cfg or ProbeConfig()
    t_start = time.perf_counter()
    source: dict[str, Any] = {}
    if gguf_path is not None:
        g = read_gguf(gguf_path)
        info = g.tensors["token_embd.weight"]
        t0 = time.perf_counter()
        E = g.tensor("token_embd.weight")
        source = {"gguf": str(gguf_path), "architecture": g.metadata.get("general.architecture"),
                  "token_embd_type": info.type_name, "token_embd_shape": list(info.shape),
                  "dequant_seconds": time.perf_counter() - t0}
        if tasks is None:
            tasks = vocab_tasks(list(g.metadata["tokenizer.ggml.tokens"]))
    else:
        if embeddings is None or tasks is None:
            raise ValueError("need gguf_path, or embeddings and tasks")
        E = np.asarray(embeddings, np.float32)
    assert tasks is not None

    # Deterministic reduction fitted on the whole vocabulary (no labels used).
    t0 = time.perf_counter()
    B, mean, ratio = pca_basis(E, min(cfg.reduced_dim, E.shape[1]))
    pca_s = time.perf_counter() - t0

    rng = np.random.default_rng(cfg.seed)
    results: list[TaskResult] = []
    neighbor: dict[str, float] = {}
    for ti, (task, (ids, labels)) in enumerate(tasks.items()):
        if len(ids) > cfg.n_samples:
            sel = np.sort(rng.choice(len(ids), cfg.n_samples, replace=False))
            ids, labels = ids[sel], labels[sel]
        X = E[ids]
        Z = ((X - mean) @ B).astype(np.float32)
        tr, te = _split(len(ids), cfg.test_frac, rng)
        vals, counts = np.unique(labels, return_counts=True)
        maj = float((labels[te] == vals[counts.argmax()]).mean())
        tres = TaskResult(task, {str(v): int(c) for v, c in zip(vals, counts)}, len(tr), len(te), maj)

        tres.runs.append(_evaluate(f"original embedding ({X.shape[1]}d)", X, labels, tr, te, cfg.seed))
        tres.runs.append(_evaluate(f"PCA-{Z.shape[1]} (pre-hash)", Z, labels, tr, te, cfg.seed))
        modes = ["hash_bits", "hash_bytes", "hamming", "bucket", "threshold"] if cfg.mode_sweep \
            else [cfg.feature_mode]
        for mode in modes:
            F, ex = _hashmind_features(Z, cfg, mode)
            tres.runs.append(_evaluate(f"HashMind[{mode}, tuple={cfg.tuple_size}]", F, labels, tr, te,
                                       cfg.seed, **ex))
            if mode == cfg.feature_mode and ti == 0 and cfg.neighbor_queries:
                q = rng.choice(len(ids), min(cfg.neighbor_queries, len(ids)), replace=False)
                neighbor = {
                    "k": cfg.neighbor_k,
                    "chance": cfg.neighbor_k / (len(ids) - 1),
                    f"PCA-{Z.shape[1]}": neighbor_recall(X, Z, q, cfg.neighbor_k),
                    f"HashMind[{mode}]": neighbor_recall(X, F, q, cfg.neighbor_k),
                }
        for ts in cfg.tuple_sweep:
            if ts == cfg.tuple_size or ts > Z.shape[1]:
                continue
            F, ex = _hashmind_features(Z, cfg, cfg.feature_mode, ts)
            tres.runs.append(_evaluate(f"HashMind[{cfg.feature_mode}, tuple={ts}]", F, labels, tr, te,
                                       cfg.seed, **ex))
        # Control: break the token<->embedding pairing.
        perm = np.random.default_rng(cfg.seed + 1).permutation(len(ids))
        F, ex = _hashmind_features(Z[perm], cfg, cfg.feature_mode)
        tres.runs.append(_evaluate(f"HashMind[{cfg.feature_mode}] on SHUFFLED embeddings", F, labels,
                                   tr, te, cfg.seed, **ex))
        results.append(tres)

    return {
        "config": asdict(cfg),
        "source": source,
        "pca": {"seconds": pca_s, "explained_variance": float(ratio.sum())},
        "tasks": [asdict(r) for r in results],
        "neighbor_recall": neighbor,
        "total_seconds": time.perf_counter() - t_start,
        "lookup_table_note": "sha256d_executed counts distinct (node, payload, nonce) triples. When it is "
        "far below sha256d_logical the layer's input space is small enough to precompute as a table.",
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def format_results(res: dict[str, Any]) -> str:
    c = res["config"]
    out = ["# HashMind phase-2 probe", ""]
    if res["source"]:
        s = res["source"]
        out.append(f"Model: `{Path(s['gguf']).name}` ({s['architecture']}), token_embd "
                   f"{s['token_embd_shape']} {s['token_embd_type']}")
    out.append(f"PCA-{c['reduced_dim']} explained variance: {res['pca']['explained_variance']:.1%}. "
               f"HashMind: {c['output_dim']} features, tuple_size={c['tuple_size']}, "
               f"levels={c['levels']}, nonces/node={c['nonces']}.")
    for t in res["tasks"]:
        out += ["", f"## Task `{t['task']}`  (train {t['n_train']}, test {t['n_test']}, classes {t['classes']})",
                "", f"Majority-class accuracy: **{t['majority_accuracy']:.1%}**", "",
                "| representation | dim | test acc | SHA-256d (logical / executed) | ASIC-native | "
                "transform µs/sample | readout µs/sample | peak MB |",
                "|---|---:|---:|---:|:---:|---:|---:|---:|"]
        for r in t["runs"]:
            sha = f"{r['sha256d_logical']:,} / {r['sha256d_executed']:,}" if r["sha256d_logical"] else "-"
            nat = "" if r["asic_native"] is None else ("yes" if r["asic_native"] else "no (sim)")
            tr = f"{r['transform_us_per_sample']:.0f}" if r["transform_us_per_sample"] else "-"
            pk = f"{r['peak_mem_mb']:.0f}" if r["peak_mem_mb"] else "-"
            out.append(f"| {r['representation']} | {r['feature_dim']} | {r['accuracy']:.1%} | {sha} | "
                       f"{nat} | {tr} | {r['readout_us_per_sample']:.1f} | {pk} |")
    if res["neighbor_recall"]:
        n = res["neighbor_recall"]
        out += ["", f"## Neighbour preservation (top-{n['k']} cosine neighbours vs original embedding)", ""]
        out += [f"- {k}: {v:.1%}" for k, v in n.items() if k != "k"]
    out += ["", "SHA-256d *logical* = what a cache-less device computes; *executed* = distinct "
            "(node, payload, nonce) evaluations. Executed << logical means the layer's input space is "
            "small enough to precompute as a lookup table.",
            "", f"Total time {res['total_seconds']:.1f}s, process max RSS {res['process_max_rss_mb']:.0f} MB."]
    return "\n".join(out)
