"""Phase-4 standardized benchmark suite: tasks, representations, one evaluator.

Every Phase-4 track (hash ablation, multi-resolution, learned routing, sparse
events) evaluates its representations through :func:`run_rep` on the same
:class:`Task` objects, so rows are directly comparable:

Task 1  probe_word_start, probe_char_class   phase-2 TinyLlama embedding probe
        (PCA-32 of token_embd, 6000 tokens, 75/25 split, seed 0, same RNG
        sequence as ``run_token_probe`` so the splits are identical)
Task 2  nexttoken_L11                        phase-3 hidden-state task (cut at
        layer 11, PCA-64, 96/32 sequence split, ridge alpha in {1,10,100} on a
        sequence-held-out validation split, preserved LM head)
Task 3  context_L0                           phase-3 context/locality task
        (PCA-64 of h_0 plus exact token ids [tok_t, tok_t-1, tok_t-2])

Nothing here touches test labels before the final readout fit: alpha and the
learned routing are selected on validation rows carved out of the training set.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..backends.s9_model import BM1387_MIN_DIFFICULTY_BITS, S9_HASHRATE
from ..conversion.weights import pca_basis
from ..core.diagnostics import cell_stats, feature_stats
from ..core.layer import HashMindLayer
from ..core.multiresolution import (
    GroupSpec,
    MultiResolutionLayer,
    context_multires_groups,
    single_resolution,
    standard_multires_groups,
)
from ..core.primitives import PRIMITIVES, get_primitive, measure_throughput, pack_payloads
from ..core.multiresolution import cell_fingerprint
from ..core.readout import RidgeReadout
from ..core.routing import GreedyRouter, RoutingConfig
from ..core.sparse_features import OutputConfig, SparseEvents, SparseRidge
from .next_token import HiddenStates, NextTokenBench, NTResult, collect_hidden_states
from .token_probe import ALPHAS, ProbeConfig, _fit_best, _split, vocab_tasks

DEFAULT_SEEDS = (0, 1, 2)
HASH_FEATURES = 2048  # matched hash-feature budget (phase-2/3 default)


# =================================================================== tasks ===

@dataclass
class Task:
    name: str
    kind: str  # probe | nexttoken
    X_tr: np.ndarray
    X_te: np.ndarray
    C_tr: np.ndarray | None = None
    C_te: np.ndarray | None = None
    y_tr: np.ndarray | None = None  # probe labels
    y_te: np.ndarray | None = None
    bench: NextTokenBench | None = None
    vocab: int = 32000
    probe_seed: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def context_dim(self) -> int:
        return 0 if self.C_tr is None else self.C_tr.shape[1]

    # ---- final evaluation (alpha chosen on train-only validation rows) -----------
    def evaluate(self, F_tr: np.ndarray, F_te: np.ndarray) -> dict[str, Any]:
        if self.kind == "probe":
            t0 = time.perf_counter()
            model, alpha = _fit_best(F_tr, self.y_tr, self.probe_seed)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            pred = model.predict_classes(F_te)
            pr_s = time.perf_counter() - t0
            return {"accuracy": float((pred == self.y_te).mean()), "alpha": alpha,
                    "readout_fit_s": fit_s, "readout_predict_s": pr_s, "readout_outputs": len(model.classes),
                    "score": float((pred == self.y_te).mean())}
        b = self.bench
        t0 = time.perf_counter()
        r: NTResult = b.evaluate("", F_tr, F_te, None)
        tot = time.perf_counter() - t0
        # predict-only time on the test rows, measured separately
        rr = RidgeReadout(r.alpha).fit(F_tr[:64], b.y_tr[:64])
        t0 = time.perf_counter()
        rr.predict(F_te).astype(np.float32) @ b.head.T
        pr_s = time.perf_counter() - t0
        return {"teacher_agreement": r.teacher_agreement, "top1": r.top1, "top5": r.top5, "alpha": r.alpha,
                "readout_fit_s": tot, "readout_predict_s": pr_s, "readout_outputs": b.y_tr.shape[1],
                "score": r.teacher_agreement}

    # ---- routing hooks -----------------------------------------------------------------
    def routing_problem(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[np.ndarray], float],
                                       tuple[float, ...]]:
        """(fit_rows, val_rows, Y_fit, metric(pred_val) -> higher better, alphas) on train only."""
        if self.kind == "probe":
            fit, val = _split(len(self.y_tr), 0.2, np.random.default_rng(self.probe_seed))
            classes, idx = np.unique(self.y_tr, return_inverse=True)
            Y = -np.ones((len(self.y_tr), len(classes)))
            Y[np.arange(len(idx)), idx] = 1.0
            yv = idx[val]
            return fit, val, Y[fit], lambda P: float((P.argmax(1) == yv).mean()), ALPHAS
        b = self.bench
        tv = b.teacher_tr[b._val_rows]
        head = b.head
        return (b._fit_rows, b._val_rows, b.y_tr[b._fit_rows],
                lambda P: float(((P.astype(np.float32) @ head.T).argmax(1) == tv).mean()), b.alphas)

    def reference_rows(self) -> list[dict[str, Any]]:
        if self.kind == "probe":
            vals, counts = np.unique(np.r_[self.y_tr, self.y_te], return_counts=True)
            maj = float((self.y_te == vals[counts.argmax()]).mean())
            return [{"rep": "majority class", "accuracy": maj, "score": maj}]
        rows = [self.bench.teacher_row()] + (self.bench.ngram_rows() if self.context_dim else [])
        return [{"rep": r.name, "teacher_agreement": r.teacher_agreement, "top1": r.top1, "top5": r.top5,
                 "score": r.teacher_agreement} for r in rows]


def probe_tasks(E: np.ndarray, tokens: list[str], cfg: ProbeConfig | None = None) -> list[Task]:
    """Replays the phase-2 ``run_token_probe`` RNG sequence: identical samples and splits."""
    cfg = cfg or ProbeConfig()
    tasks = vocab_tasks(tokens)
    B, mean, ratio = pca_basis(E, min(cfg.reduced_dim, E.shape[1]))
    rng = np.random.default_rng(cfg.seed)
    out = []
    for ti, (task, (ids, labels)) in enumerate(tasks.items()):
        if len(ids) > cfg.n_samples:
            sel = np.sort(rng.choice(len(ids), cfg.n_samples, replace=False))
            ids, labels = ids[sel], labels[sel]
        Z = ((E[ids] - mean) @ B).astype(np.float32)
        tr, te = _split(len(ids), cfg.test_frac, rng)
        if ti == 0 and cfg.neighbor_queries:  # phase 2 drew neighbour queries here
            rng.choice(len(ids), min(cfg.neighbor_queries, len(ids)), replace=False)
        out.append(Task(f"probe_{task}", "probe", Z[tr], Z[te], y_tr=labels[tr], y_te=labels[te],
                        probe_seed=cfg.seed,
                        info={"pca_dim": Z.shape[1], "pca_explained_variance": float(ratio.sum()),
                              "n_train": len(tr), "n_test": len(te)}))
    return out


def nexttoken_tasks(hs: HiddenStates, head: np.ndarray, cut: int = 11, r: int = 64,
                    vocab: int = 32000) -> list[Task]:
    """Task 2 (cut layer ``cut``, PCA-r) and Task 3 (layer 0 + 3 context token ids)."""
    bench = NextTokenBench(hs, head)
    out = []
    for name, L, ctx in ((f"nexttoken_L{cut}", cut, 0), ("context_L0", 0, 3)):
        if L not in hs.hidden:
            continue
        X_tr, X_te = bench.x(L, "train"), bench.x(L, "test")
        B, mean, ratio = pca_basis(X_tr, r)
        Z_tr, Z_te = ((X_tr - mean) @ B).astype(np.float32), ((X_te - mean) @ B).astype(np.float32)
        C_tr = bench.context("train", ctx).astype(np.uint32) if ctx else None
        C_te = bench.context("test", ctx).astype(np.uint32) if ctx else None
        out.append(Task(name, "nexttoken", Z_tr, Z_te, C_tr, C_te, bench=bench, vocab=vocab,
                        info={"cut_layer": L, "pca_dim": r, "pca_explained_variance": float(ratio.sum()),
                              "context_cols": ["tok_t", "tok_t-1", "tok_t-2"][:ctx],
                              "n_train": len(Z_tr), "n_test": len(Z_te)}))
    return out


def load_tasks(gguf: str | Path, cache: str | Path | None = None, n_seq: int = 128, seq_len: int = 128,
               cut: int = 11, probe_samples: int = 6000) -> tuple[list[Task], dict[str, Any]]:
    from ..gguf.reader import read_gguf
    from ..llm.llama import LlamaReference

    g = read_gguf(gguf)
    E = g.tensor("token_embd.weight")
    tokens = list(g.metadata["tokenizer.ggml.tokens"])
    tasks = probe_tasks(E, tokens, ProbeConfig(n_samples=probe_samples))
    del E
    hs = collect_hidden_states(gguf, n_seq, seq_len, layers=(0, cut), cache=cache)
    tasks += nexttoken_tasks(hs, LlamaReference(g).lm_head(), cut=cut, vocab=len(tokens))
    meta = {"model": Path(gguf).name, "architecture": g.metadata.get("general.architecture"),
            "model_name": g.metadata.get("general.name"), "n_seq": n_seq, "seq_len": seq_len,
            "cut_layer": cut, "hidden_forward_seconds": hs.forward_seconds}
    return tasks, meta


# ========================================================= representations ===

@dataclass(frozen=True)
class Rep:
    """One representation. ``kind``: pca | rff | lsh | hash | phase3.

    hash: ``layout`` in single | multires | context_multires, plus primitive,
    output mode, node budget, evaluations per node and optional learned routing.
    phase3: the unchanged phase-3 ``HashMindLayer`` with ``phase3_kw`` (context rows).
    """

    name: str
    kind: str
    primitive: str = "sha256d"
    layout: str = "single"
    tuple_size: int = 2
    levels: int = 4
    n_nodes: int = 128
    evals: int = 16
    output: OutputConfig = OutputConfig()
    concat_pca: bool = False
    routing: bool = False
    n_random: int = HASH_FEATURES  # rff / lsh
    phase3_kw: tuple = ()

    def signature(self) -> tuple:
        d = asdict(self)
        d.pop("name")
        return tuple(sorted((k, str(v)) for k, v in d.items()))


def build_layer(rep: Rep, task: Task, seed: int) -> MultiResolutionLayer:
    d = task.X_tr.shape[1]
    if rep.layout == "single":
        return single_resolution(d, rep.n_nodes, rep.tuple_size, rep.levels, rep.evals, rep.primitive, seed,
                                 rep.output, wiring="legacy")
    if rep.layout == "multires":
        groups = standard_multires_groups(rep.n_nodes, rep.evals)
    elif rep.layout == "context_multires":
        groups = context_multires_groups(rep.n_nodes, rep.evals)
    else:
        raise ValueError(rep.layout)
    return MultiResolutionLayer(d, groups, rep.primitive, rep.output, seed, task.context_dim if any(
        g.ctx_cols for g in groups) else 0)


def s9_model(layer: MultiResolutionLayer | None, rep: Rep) -> dict[str, Any] | None:
    """Modelled (NOT measured) S9 feature-generation cost per example at the BM1387 floor.

    bits:        dense p=1/2 bits are not directly available at difficulty 32; use the phase-3
                 ``nonce_bits`` scheme (17 feature bits per full 2**32 sweep).
    hit (d):     one feature window of 2**(32-d) hashes per feature: D * 2**-d sweeps.
    bucket:      one sweep per event; the first share's position mod B (B <= 2**16).
    first_nonce: one sweep per node.
    """
    if rep.primitive != "sha256d" or layer is None:
        return None
    o = layer.output
    D = layer.n_features
    nodes = len(layer.nodes)
    if o.mode == "bits":
        sweeps = D / 17
    elif o.mode == "hit":
        sweeps = D * 2.0 ** -o.difficulty_bits
    elif o.mode == "bucket":
        sweeps = layer.evals_per_example
    else:
        sweeps = nodes
    sweep_s = 2.0 ** BM1387_MIN_DIFFICULTY_BITS / S9_HASHRATE
    return {"sweeps_per_example": sweeps, "hashes_per_example": sweeps * 2.0 ** 32,
            "feature_ms_per_example": sweeps * sweep_s * 1e3,
            "examples_per_second": 1.0 / (sweeps * sweep_s),
            "note": "modelled at 13.5 TH/s nominal, difficulty floor 2^-32; excludes dispatch, UART, host"}


def _random_features(rep: Rep, task: Task, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([seed, 0xFEA7])
    X_tr, X_te = task.X_tr, task.X_te
    W = rng.standard_normal((X_tr.shape[1], rep.n_random)).astype(np.float32)
    if rep.kind == "lsh":
        return (X_tr @ W > 0).astype(np.float32), (X_te @ W > 0).astype(np.float32)
    s = X_tr[rng.choice(len(X_tr), min(1000, len(X_tr)), replace=False)]
    dist = np.sqrt(((s[:, None, :] - s[None, :, :]) ** 2).sum(-1))
    sigma = float(np.median(dist[dist > 0]))  # median heuristic, fixed a priori
    b = rng.uniform(0, 2 * np.pi, rep.n_random).astype(np.float32)
    W /= sigma
    k = np.sqrt(2.0 / rep.n_random)
    return (k * np.cos(X_tr @ W + b)).astype(np.float32), (k * np.cos(X_te @ W + b)).astype(np.float32)


def _phase3_layer_cells(layer: HashMindLayer, X: np.ndarray, C: np.ndarray | None) -> list[np.ndarray]:
    cells = []
    for node in layer.nodes:
        codes = node.input_mapping.encode(X)
        cols = list(node.input_mapping.context_indices)
        ctx = C[:, cols].astype(np.uint32) if cols else None
        cells.append(cell_fingerprint(pack_payloads(codes, ctx)))
    return cells


_THROUGHPUT: dict[str, float] = {}


def primitive_throughput(name: str) -> float:
    if name not in _THROUGHPUT:
        _THROUGHPUT[name] = measure_throughput(get_primitive(name))
    return _THROUGHPUT[name]


def run_rep(task: Task, rep: Rep, seed: int, routing_cfg: RoutingConfig | None = None) -> dict[str, Any]:
    """Build, fit, transform, evaluate one representation; return a fully documented row."""
    row: dict[str, Any] = {"task": task.name, "rep": rep.name, "seed": seed, "kind": rep.kind,
                           "primitive": rep.primitive if rep.kind in ("hash", "phase3") else None,
                           "layout": rep.layout if rep.kind == "hash" else rep.kind,
                           "output_mode": rep.output.mode if rep.kind == "hash" else None,
                           "concat_pca": rep.concat_pca or rep.kind in ("rff", "lsh"),
                           "routing": rep.routing, "config": {k: str(v) for k, v in asdict(rep).items()}}
    X_tr, X_te = task.X_tr, task.X_te
    layer = None
    ev_tr = None
    t0 = time.perf_counter()
    if rep.kind == "pca":
        H_tr = H_te = None
    elif rep.kind in ("rff", "lsh"):
        H_tr, H_te = _random_features(rep, task, seed)
    elif rep.kind == "phase3":
        kw = dict(rep.phase3_kw)
        layer3 = HashMindLayer(X_tr.shape[1], HASH_FEATURES, seed=seed, tuple_size=kw.get("tuple_size", 2),
                               levels=4, nonces_per_node=16, context_dim=kw.get("context_cols", 0),
                               context_per_node=kw.get("context_per_node", 0))
        C_tr = task.C_tr[:, : layer3.context_dim] if layer3.context_dim else None
        C_te = task.C_te[:, : layer3.context_dim] if layer3.context_dim else None
        layer3.fit(X_tr)
        H_tr, H_te = layer3.transform(X_tr, C_tr), layer3.transform(X_te, C_te)
        row["cells"] = cell_stats(_phase3_layer_cells(layer3, X_tr, C_tr), _phase3_layer_cells(layer3, X_te, C_te))
        row["evals_per_example"] = layer3.n_nodes * 16
        row["table_log2"] = layer3.table_log2(task.vocab)
        row["evals_executed"] = layer3.stats.sha256d_executed
        row["s9"] = s9_model(None, rep)
    else:
        layer = build_layer(rep, task, seed).fit(X_tr)
        if rep.routing:
            fit, val, Y_fit, metric, alphas = task.routing_problem()
            extra = X_tr if rep.concat_pca else None
            # alpha for routing rounds: best on the validation rows for the initial wiring
            init = layer.transform_nodes(X_tr, task.C_tr).assemble()
            Fi = init.to_dense() if isinstance(init, SparseEvents) else init
            Fi = np.hstack([extra, Fi]) if extra is not None else Fi
            alpha = max(alphas, key=lambda a: metric(RidgeReadout(a).fit(Fi[fit], Y_fit).predict(Fi[val])))
            hist = GreedyRouter(layer, routing_cfg or RoutingConfig(seed=seed)).fit(
                X_tr, task.C_tr, fit, val, Y_fit, alpha, metric, extra)
            row["routing_history"] = hist.to_dict()
            row["routing_alpha"] = alpha
            row["learned_state_bytes"] = hist.learned_index_bytes
        layer.counters = type(layer.counters)()
        o_tr = layer.transform_nodes(X_tr, task.C_tr)
        o_te = layer.transform_nodes(X_te, task.C_te)
        feat_s = layer.counters.seconds
        H_tr, H_te = o_tr.assemble(), o_te.assemble()
        buckets = None
        if rep.output.mode == "bucket":
            buckets = [b[:, 0] for b in o_tr.blocks]
        row["cells"] = cell_stats(o_tr.cells, o_te.cells, buckets)
        row["evals_per_example"] = layer.evals_per_example
        row["evals_executed"] = layer.counters.evals_executed
        row["table_log2"] = layer.table_log2(task.vocab)
        row["layer"] = layer.describe()
        row["s9"] = s9_model(layer, rep)
        row["feature_seconds"] = feat_s
        if isinstance(H_tr, SparseEvents):
            ev_tr, ev_te = H_tr, H_te
            H_tr, H_te = ev_tr.to_dense(), ev_te.to_dense()
    gen_s = time.perf_counter() - t0
    row.setdefault("feature_seconds", gen_s)
    row["feature_and_setup_seconds"] = gen_s

    if H_tr is not None:
        row["features"] = feature_stats(ev_tr if ev_tr is not None else H_tr)
        row["features"]["storage_bytes_per_example_dense_f32"] = H_tr.shape[1] * 4
        if ev_tr is not None:
            row["features"]["storage_bytes_per_example_sparse_ids"] = ev_tr.nbytes_per_example()
    if rep.kind == "pca":
        F_tr, F_te = X_tr, X_te
    elif row["concat_pca"]:
        F_tr, F_te = np.hstack([X_tr, H_tr]), np.hstack([X_te, H_te])
    else:
        F_tr, F_te = H_tr, H_te
    row["dim"] = int(F_tr.shape[1])
    row["hash_features"] = 0 if H_tr is None else int(H_tr.shape[1])
    row.update(task.evaluate(F_tr, F_te))
    row["readout_state_bytes"] = row["dim"] * row["readout_outputs"] * 8  # float64 ridge weights

    # compute accounting -------------------------------------------------------------
    n_te = len(X_te)
    if rep.kind in ("hash", "phase3"):
        thr = primitive_throughput(rep.primitive)
        row["cpu_evals_per_s_single_core"] = thr
        row["cacheless_feature_us_per_example"] = row["evals_per_example"] / thr * 1e6
        row["inference_us_per_example"] = row["cacheless_feature_us_per_example"] + \
            row["readout_predict_s"] / n_te * 1e6
    else:
        row["inference_us_per_example"] = row["readout_predict_s"] / n_te * 1e6
    if ev_tr is not None and row["dim"] <= 9000:
        row["sparse_readout"] = _sparse_readout_check(task, ev_tr, ev_te, H_tr, H_te, X_tr, X_te,
                                                      row["alpha"], row["concat_pca"])
    return row


def _sparse_readout_check(task: Task, ev_tr: SparseEvents, ev_te: SparseEvents, H_tr: np.ndarray,
                          H_te: np.ndarray, X_tr: np.ndarray, X_te: np.ndarray, alpha: float,
                          concat: bool) -> dict[str, Any]:
    """Time the event-native readout against the dense one at the selected alpha (hash
    features only, no PCA concat, so the comparison is purely dense vs sparse)."""
    Y = task.bench.y_tr if task.kind == "nexttoken" else None
    if Y is None:
        classes, idx = np.unique(task.y_tr, return_inverse=True)
        Y = -np.ones((len(idx), len(classes)))
        Y[np.arange(len(idx)), idx] = 1.0
    t0 = time.perf_counter()
    d = RidgeReadout(alpha).fit(H_tr, Y)
    dense_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    pd = d.predict(H_te)
    dense_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = SparseRidge(alpha).fit(ev_tr, Y)
    sparse_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ps = s.predict(ev_te)
    sparse_pred = time.perf_counter() - t0
    return {"dense_fit_s": dense_fit, "dense_predict_s": dense_pred, "sparse_fit_s": sparse_fit,
            "sparse_predict_s": sparse_pred, "max_abs_prediction_diff": float(np.abs(pd - ps).max()),
            "alpha": alpha}


# ===================================================================== suite ===

def environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = ""
    from .. import __version__

    return {"hashmind_version": __version__, "git_commit": commit, "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform(), "time_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class RowCache:
    """Identical (task, representation, seed) triples are computed once across tracks."""

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.rows: dict[tuple, dict[str, Any]] = {}
        self.log = log or (lambda m: None)

    def get(self, task: Task, rep: Rep, seed: int) -> dict[str, Any]:
        key = (task.name, rep.signature(), seed)
        if key not in self.rows:
            t0 = time.perf_counter()
            self.rows[key] = run_rep(task, rep, seed)
            r = self.rows[key]
            self.log(f"{task.name:18s} {rep.name:44s} seed {seed}  score {r['score']:.4f}  "
                     f"({time.perf_counter() - t0:.0f}s)")
        row = dict(self.rows[key])
        row["rep"] = rep.name
        return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean / std / n over seeds for each (task, rep)."""
    keys = ("score", "accuracy", "teacher_agreement", "top1", "top5")
    cell_keys = ("unique_cell_fraction", "repeated_cell_rate", "singleton_cell_fraction", "test_coverage",
                 "mean_examples_per_cell", "median_examples_per_cell", "effective_cardinality",
                 "collision_rate")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["task"], r["rep"]), []).append(r)
    out = []
    for (task, rep), rs in groups.items():
        a: dict[str, Any] = {"task": task, "rep": rep, "n_seeds": len(rs), "seeds": [r.get("seed") for r in rs]}
        for k in keys:
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                a[k] = float(np.mean(v))
                a[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        for k in cell_keys:
            v = [r["cells"][k] for r in rs if "cells" in r and k in r["cells"]]
            if v:
                a[k] = float(np.mean(v))
        f = [r["features"] for r in rs if "features" in r]
        if f:
            for k in ("active_per_example", "sparsity", "feature_reuse_rate", "feature_entropy_normalized",
                      "storage_bytes_per_example_dense_f32", "storage_bytes_per_example_sparse_ids"):
                v = [x[k] for x in f if k in x]
                if v:
                    a[k] = float(np.mean(v))
        r0 = rs[0]
        for k in ("dim", "hash_features", "evals_per_example", "table_log2", "primitive", "layout",
                  "output_mode", "concat_pca", "routing", "cpu_evals_per_s_single_core",
                  "cacheless_feature_us_per_example", "inference_us_per_example", "readout_state_bytes",
                  "learned_state_bytes"):
            if k in r0:
                a[k] = r0[k]
        for k in ("feature_seconds", "readout_fit_s", "readout_predict_s", "evals_executed"):
            v = [r[k] for r in rs if k in r]
            if v:
                a[k] = float(np.mean(v))
        if r0.get("s9"):
            a["s9_feature_ms_per_example"] = r0["s9"]["feature_ms_per_example"]
            a["s9_sweeps_per_example"] = r0["s9"]["sweeps_per_example"]
        if "sparse_readout" in r0:
            a["sparse_readout"] = r0["sparse_readout"]
        if "routing_history" in r0:
            a["routing_val_gain"] = float(np.mean([r["routing_history"]["val_scores"][-1]
                                                   - r["routing_history"]["val_scores"][0] for r in rs]))
            a["routing_accept_rate"] = float(np.mean([np.mean(r["routing_history"]["accepted"] or [0])
                                                      for r in rs]))
        if "reference" in r0:
            a["reference"] = True
        out.append(a)
    return out


def reference_rows(task: Task) -> list[dict[str, Any]]:
    rows = task.reference_rows()
    for r in rows:
        r.update({"task": task.name, "reference": True, "seed": None})
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def fmt_pct(a: dict[str, Any], k: str) -> str:
    if k not in a:
        return "-"
    s = a.get(k + "_std")
    return f"{a[k]:.1%}" + (f" ± {s:.1%}" if s else "")


def fmt_num(v: Any, spec: str = ".3g") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return format(v, spec)


def table(aggs: list[dict[str, Any]], task_kind: str, extra_cols: list[tuple[str, str, Callable]] | None = None) -> str:
    """Markdown table of aggregated rows for one task."""
    if task_kind == "probe":
        cols = [("accuracy", lambda a: fmt_pct(a, "accuracy"))]
    else:
        cols = [("teacher agree", lambda a: fmt_pct(a, "teacher_agreement")),
                ("top-1", lambda a: fmt_pct(a, "top1")), ("top-5", lambda a: fmt_pct(a, "top5"))]
    base = [("dim", lambda a: str(a.get("dim", "-"))),
            ("evals/ex", lambda a: str(a.get("evals_per_example", "-"))),
            ("uniq-cell frac", lambda a: fmt_num(a.get("unique_cell_fraction"), ".3f")),
            ("repeated-cell", lambda a: fmt_num(a.get("repeated_cell_rate"), ".3f")),
            ("singleton", lambda a: fmt_num(a.get("singleton_cell_fraction"), ".3f")),
            ("test cover", lambda a: fmt_num(a.get("test_coverage"), ".3f")),
            ("active/ex", lambda a: fmt_num(a.get("active_per_example"), ".0f")),
            ("log2 table", lambda a: fmt_num(a.get("table_log2"), ".1f"))]
    cols = cols + base + [(h, f) for h, _, f in (extra_cols or [])]
    lines = ["| representation | " + " | ".join(h for h, _ in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for a in aggs:
        lines.append(f"| {a['rep']} | " + " | ".join(f(a) for _, f in cols) + " |")
    return "\n".join(lines)


def controls(task: Task) -> list[Rep]:
    """Rows present in every track: PCA-only, current HashMind, random-feature controls,
    and (context task) the phase-3 context configurations, unchanged."""
    out = [
        Rep("PCA only", "pca"),
        Rep("current HashMind (sha256d single t2l4)", "hash"),
        Rep("PCA + current HashMind (sha256d single t2l4)", "hash", concat_pca=True),
        Rep("PCA + random Fourier features", "rff"),
        Rep("PCA + random sign projections (LSH bits)", "lsh"),
    ]
    if task.context_dim:
        out += [
            Rep("phase3: t1 + 1 ctx of [tok_t, tok_t-1]", "phase3",
                phase3_kw=(("tuple_size", 1), ("context_cols", 2), ("context_per_node", 1))),
            Rep("phase3: ctx only, 2 of [tok_t..tok_t-2]", "phase3",
                phase3_kw=(("tuple_size", 0), ("context_cols", 3), ("context_per_node", 2))),
            Rep("phase3: PCA + ctx 2 of [tok_t..tok_t-2]", "phase3", concat_pca=True,
                phase3_kw=(("tuple_size", 0), ("context_cols", 3), ("context_per_node", 2))),
        ]
    return out
