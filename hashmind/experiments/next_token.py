"""Phase-3 experiment: replace the top of a real transformer with HashMind.

    real GGUF forward pass (host reference, numpy)
        -> residual stream h_L at cut layer L         (blocks 0..L-1 still run on the host)
        -> PCA-r of h_L
        -> HashMind layer (SHA-256d features)          replaces blocks L..n-1
        -> ridge readout  -> estimate of the final normed hidden state
        -> PRESERVED LM head (output.weight)           -> next-token logits

Metrics per configuration: top-1 agreement with the teacher (the original
model's own argmax), top-1 / top-5 accuracy against the true next token,
feature dimensionality, SHA-256d count, wall time, and log2 of the
lookup-table size needed to precompute the layer (is the ASIC needed at all?).

Count-based n-gram baselines (fit on the same training tokens) are reported so
that a context-hashing HashMind layer cannot be mistaken for something more
than a hashed n-gram model.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..conversion.weights import pca_basis
from ..core.layer import HashMindLayer
from ..core.readout import RidgeReadout
from ..gguf.reader import read_gguf
from ..llm.llama import LlamaReference
from ..llm.tokenizer import SPMTokenizer


# ------------------------------------------------------------------ corpus ---

def default_corpus() -> str:
    """English text that ships with CPython (the pydoc topic help), so no download."""
    import pydoc_data.topics as t

    return "\n\n".join(t.topics[k] for k in sorted(t.topics))


def make_sequences(tok: SPMTokenizer, text: str, n_seq: int, seq_len: int) -> np.ndarray:
    ids = tok.encode(text, bos=False)
    need = n_seq * (seq_len - 1)
    if len(ids) < need:
        raise ValueError(f"corpus has {len(ids)} tokens, need {need}")
    body = np.asarray(ids[:need]).reshape(n_seq, seq_len - 1)
    return np.concatenate([np.full((n_seq, 1), tok.bos_id), body], 1)


@dataclass
class HiddenStates:
    tokens: np.ndarray  # (S, T)
    hidden: dict[int, np.ndarray]  # L -> (S, T, d) float16
    final: np.ndarray  # (S, T, d) float16, input of the LM head
    teacher_top1: np.ndarray  # (S, T)
    forward_seconds: float

    def save(self, path: Path) -> None:
        np.savez(path, tokens=self.tokens, final=self.final, teacher_top1=self.teacher_top1,
                 forward_seconds=self.forward_seconds,
                 **{f"h{L}": v for L, v in self.hidden.items()})

    @classmethod
    def load(cls, path: Path) -> "HiddenStates":
        z = np.load(path)
        hid = {int(k[1:]): z[k] for k in z.files if k.startswith("h")}
        return cls(z["tokens"], hid, z["final"], z["teacher_top1"], float(z["forward_seconds"]))


def collect_hidden_states(gguf_path: str | Path, n_seq: int = 128, seq_len: int = 128,
                          layers: tuple[int, ...] = (0, 6, 11, 16, 22), cache: str | Path | None = None,
                          verbose: bool = True) -> HiddenStates:
    if cache is not None and Path(cache).exists():
        return HiddenStates.load(Path(cache))
    g = read_gguf(gguf_path)
    tok = SPMTokenizer.from_gguf_metadata(g.metadata)
    ref = LlamaReference(g)
    layers = tuple(L for L in layers if L <= ref.hp.n_layer)
    seqs = make_sequences(tok, default_corpus(), n_seq, seq_len)
    t0 = time.perf_counter()
    r = ref.forward(seqs, capture=set(layers), verbose=verbose)
    head = ref.lm_head()
    top1 = np.empty(seqs.shape, np.int64)
    for i in range(seqs.shape[0]):
        top1[i] = (r.final_normed[i] @ head.T).argmax(-1)
    hs = HiddenStates(seqs, {L: v.astype(np.float16) for L, v in r.hidden.items()},
                      r.final_normed.astype(np.float16), top1, time.perf_counter() - t0)
    if cache is not None:
        hs.save(Path(cache))
    return hs


# ------------------------------------------------------------------ metrics --

@dataclass
class NTResult:
    name: str
    cut_layer: int | None
    feature_dim: int
    teacher_agreement: float
    top1: float
    top5: float
    alpha: float | None = None
    seconds: float = 0.0
    sha256d_logical: int = 0
    sha256d_executed: int = 0
    table_log2: float | None = None
    notes: str = ""


def _topk_metrics(logits: np.ndarray, true_next: np.ndarray, teacher: np.ndarray) -> tuple[float, float, float]:
    top5 = np.argpartition(-logits, 5, axis=1)[:, :5]
    top1 = logits.argmax(1)
    return (float((top1 == teacher).mean()), float((top1 == true_next).mean()),
            float((top5 == true_next[:, None]).any(1).mean()))


class NextTokenBench:
    """Holds the split data and the preserved LM head; evaluates feature sets."""

    def __init__(self, hs: HiddenStates, head: np.ndarray, train_frac: float = 0.75, seed: int = 0,
                 alphas: tuple[float, ...] = (1.0, 10.0, 100.0)) -> None:
        S, T = hs.tokens.shape
        perm = np.random.default_rng(seed).permutation(S)
        ntr = int(S * train_frac)
        self.seq_tr, self.seq_te = np.sort(perm[:ntr]), np.sort(perm[ntr:])
        self.hs, self.head, self.alphas = hs, head.astype(np.float32), alphas
        self.T = T

        def flat(seqs: np.ndarray, arr: np.ndarray) -> np.ndarray:
            return arr[seqs, : T - 1].reshape(len(seqs) * (T - 1), *arr.shape[2:])

        self._flat = flat
        self.y_tr = flat(self.seq_tr, hs.final).astype(np.float32)
        self.next_tr = hs.tokens[self.seq_tr, 1:].reshape(-1)
        self.next_te = hs.tokens[self.seq_te, 1:].reshape(-1)
        self.teacher_te = flat(self.seq_te, hs.teacher_top1)
        self.cur_tr = flat(self.seq_tr, hs.tokens)
        self.cur_te = flat(self.seq_te, hs.tokens)
        # inner validation split (by sequence) for alpha selection
        nv = max(1, ntr // 5)
        self._val_rows = np.arange(len(self.y_tr))[-(nv * (T - 1)):]
        self._fit_rows = np.arange(len(self.y_tr))[: -(nv * (T - 1))]
        self.teacher_tr = flat(self.seq_tr, hs.teacher_top1)

    def x(self, L: int, part: str) -> np.ndarray:
        seqs = self.seq_tr if part == "train" else self.seq_te
        return self._flat(seqs, self.hs.hidden[L]).astype(np.float32)

    def context(self, part: str, n: int = 3) -> np.ndarray:
        """Discrete context columns: [token_t, token_{t-1}, ..., token_{t-n+1}] (0 = before start)."""
        seqs = self.seq_tr if part == "train" else self.seq_te
        toks = self.hs.tokens[seqs]
        cols = []
        for k in range(n):
            shifted = np.zeros_like(toks)
            shifted[:, k:] = toks[:, : toks.shape[1] - k]
            cols.append(shifted[:, : self.T - 1].reshape(-1))
        return np.stack(cols, 1)

    def evaluate(self, name: str, F_tr: np.ndarray, F_te: np.ndarray, cut: int | None, **extra: Any) -> NTResult:
        t0 = time.perf_counter()
        best, best_a = -1.0, self.alphas[0]
        for a in self.alphas:
            r = RidgeReadout(a).fit(F_tr[self._fit_rows], self.y_tr[self._fit_rows])
            lg = r.predict(F_tr[self._val_rows]).astype(np.float32) @ self.head.T
            agree = float((lg.argmax(1) == self.teacher_tr[self._val_rows]).mean())
            if agree > best:
                best, best_a = agree, a
        r = RidgeReadout(best_a).fit(F_tr, self.y_tr)
        lg = r.predict(F_te).astype(np.float32) @ self.head.T
        ag, t1, t5 = _topk_metrics(lg, self.next_te, self.teacher_te)
        return NTResult(name, cut, F_tr.shape[1], ag, t1, t5, best_a, time.perf_counter() - t0, **extra)

    # ---- reference rows -------------------------------------------------------

    def teacher_row(self) -> NTResult:
        fin = self._flat(self.seq_te, self.hs.final).astype(np.float32)
        ag, t1, t5 = _topk_metrics(fin @ self.head.T, self.next_te, self.teacher_te)
        return NTResult("teacher: original model, all blocks", None, 0, ag, t1, t5)

    def ngram_rows(self) -> list[NTResult]:
        tr_seq = self.hs.tokens[self.seq_tr]
        uni = Counter(tr_seq[:, 1:].ravel().tolist())
        top_uni = [w for w, _ in uni.most_common(5)]
        big: dict[int, Counter] = defaultdict(Counter)
        tri: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for s in tr_seq:
            for t in range(len(s) - 1):
                big[int(s[t])][int(s[t + 1])] += 1
                if t:
                    tri[(int(s[t - 1]), int(s[t]))][int(s[t + 1])] += 1
        ctx = self.context("test", 2)
        rows = []
        for name, fn in (
            ("unigram (most frequent token)", lambda c, p: top_uni),
            ("bigram counts (backoff unigram)", lambda c, p: [w for w, _ in big[c].most_common(5)] or top_uni),
            ("trigram counts (backoff bigram)", lambda c, p: [w for w, _ in tri[(p, c)].most_common(5)]
             or [w for w, _ in big[c].most_common(5)] or top_uni),
        ):
            preds = [fn(int(c), int(p)) for c, p in ctx]
            t1 = np.array([p[0] for p in preds])
            t5 = np.mean([n in p[:5] for n, p in zip(self.next_te, preds)])
            rows.append(NTResult(name, None, 0, float((t1 == self.teacher_te).mean()),
                                 float((t1 == self.next_te).mean()), float(t5)))
        return rows


@dataclass
class HMConfig:
    name: str
    tuple_size: int = 2
    levels: int = 4
    nonces: int = 16
    output_dim: int = 2048
    quantizer: str = "quantile"
    context_per_node: int = 0
    context_cols: int = 0
    concat_linear: bool = False
    seed: int = 0


def run_hashmind(bench: NextTokenBench, L: int, cfg: HMConfig, r: int = 64,
                 vocab: int = 32000) -> NTResult:
    X_tr, X_te = bench.x(L, "train"), bench.x(L, "test")
    B, mean, _ = pca_basis(X_tr, r)
    Z_tr, Z_te = (X_tr - mean) @ B, (X_te - mean) @ B
    ctx_tr = bench.context("train", cfg.context_cols) if cfg.context_cols else None
    ctx_te = bench.context("test", cfg.context_cols) if cfg.context_cols else None
    layer = HashMindLayer(r, cfg.output_dim, seed=cfg.seed, tuple_size=cfg.tuple_size, levels=cfg.levels,
                          nonces_per_node=cfg.nonces, context_dim=cfg.context_cols,
                          context_per_node=cfg.context_per_node)
    y_sup = None
    if cfg.quantizer == "supervised":
        Yc = bench.y_tr - bench.y_tr.mean(0)
        _, _, Vt = np.linalg.svd(Yc[::4], full_matrices=False)
        y_sup = Yc @ Vt[:8].T
    t0 = time.perf_counter()
    layer.fit(Z_tr, y_sup, cfg.quantizer)
    F_tr = layer.transform(Z_tr, ctx_tr)
    F_te = layer.transform(Z_te, ctx_te)
    secs = time.perf_counter() - t0
    if cfg.concat_linear:
        F_tr, F_te = np.hstack([Z_tr, F_tr]), np.hstack([Z_te, F_te])
    res = bench.evaluate(cfg.name, F_tr, F_te, L, sha256d_logical=layer.stats.sha256d_logical,
                         sha256d_executed=layer.stats.sha256d_executed,
                         table_log2=layer.table_log2(vocab))
    res.seconds += secs
    return res


def run_linear(bench: NextTokenBench, L: int, r: int | None) -> NTResult:
    X_tr, X_te = bench.x(L, "train"), bench.x(L, "test")
    if r is None:
        return bench.evaluate(f"linear ridge on full h_{L}", X_tr, X_te, L)
    B, mean, _ = pca_basis(X_tr, r)
    return bench.evaluate(f"linear ridge on PCA-{r}(h_{L})", (X_tr - mean) @ B, (X_te - mean) @ B, L)


def format_rows(rows: list[NTResult]) -> str:
    out = ["| configuration | cut L | dim | teacher agree | top-1 | top-5 | SHA-256d logical / executed | "
           "log2 table | time s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sha = f"{r.sha256d_logical:,} / {r.sha256d_executed:,}" if r.sha256d_logical else "-"
        tb = f"{r.table_log2:.1f}" if r.table_log2 is not None else "-"
        cut = "-" if r.cut_layer is None else str(r.cut_layer)
        out.append(f"| {r.name} | {cut} | {r.feature_dim or '-'} | {r.teacher_agreement:.1%} | {r.top1:.1%} | "
                   f"{r.top5:.1%} | {sha} | {tb} | {r.seconds:.0f} |")
    return "\n".join(out)


def rows_to_dicts(rows: list[NTResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
