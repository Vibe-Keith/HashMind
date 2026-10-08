"""Sparse / event-based HashMind outputs (Phase 4, Track D).

A BM1387 never returns a dense vector; it returns *events* (passing nonces).
This module turns per-evaluation output words (see :mod:`.primitives`) into
either a dense float32 block or sparse feature IDs, and provides a readout that
consumes the IDs directly:

    hash evaluations -> sparse feature events -> sparse (or dense) linear readout

Output modes (per node, ``e`` evaluations each):

``bits``         dense; feature k = 1 if evaluation k is a difficulty-1 share
                 (phase-2/3 ``hash_bits``). ``e`` features, ~e/2 active.
``hit``          sparse; same features at difficulty ``d`` (p = 2**-d). Only
                 hits are events. ``e`` features, ~e * 2**-d active.
``bucket``       sparse categorical; each evaluation emits one event: a bucket
                 ID in [0, B) from the digest's low word. ``e * B`` features,
                 exactly ``e`` active. Simulator view of a nonce-position event.
``first_nonce``  sparse categorical, ASIC-faithful: the ``e`` evaluations form
                 one window at difficulty ``d``; the event is the offset of the
                 first share mod B, or a "miss" category. ``B + 1`` features,
                 exactly 1 active. This is what a chip returning one nonce
                 actually tells the host.

On hardware at the BM1387 floor (d = 32) a whole 2**32 sweep produces one
first-share position whose low bits are uniform, so ``bucket`` with
``B <= 2**16`` is modelled as one sweep per event (see s9_model.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTPUT_MODES = ("bits", "hit", "bucket", "first_nonce")


@dataclass(frozen=True)
class OutputConfig:
    mode: str = "bits"
    buckets: int = 16  # bucket / first_nonce
    difficulty_bits: int = 4  # hit / first_nonce

    def __post_init__(self) -> None:
        if self.mode not in OUTPUT_MODES:
            raise ValueError(f"mode must be one of {OUTPUT_MODES}")
        if self.buckets < 2 or not 1 <= self.difficulty_bits <= 32:
            raise ValueError("buckets >= 2 and difficulty_bits in [1, 32]")

    @property
    def sparse(self) -> bool:
        return self.mode != "bits"

    def features_per_node(self, evals: int) -> int:
        if self.mode in ("bits", "hit"):
            return evals
        if self.mode == "bucket":
            return evals * self.buckets
        return self.buckets + 1

    def events_per_node(self, evals: int) -> int:
        """Max events per node per example (columns of the id array)."""
        return {"bits": evals, "hit": evals, "bucket": evals, "first_nonce": 1}[self.mode]


def node_output(words: np.ndarray, cfg: OutputConfig) -> np.ndarray:
    """(U, e, 2) uint32 words -> per-node block.

    bits:   (U, e) float32 dense.
    others: (U, k) int32 local feature ids in [0, features_per_node), -1 = no event.
    """
    w0, w1 = words[..., 0], words[..., 1]
    e = words.shape[1]
    if cfg.mode == "bits":
        return (w0 >> np.uint32(31) == 0).astype(np.float32)
    if cfg.mode == "hit":
        hit = (w0 >> np.uint32(32 - cfg.difficulty_bits)) == 0
        return np.where(hit, np.arange(e, dtype=np.int32)[None, :], -1).astype(np.int32)
    if cfg.mode == "bucket":
        return (np.arange(e, dtype=np.int64)[None, :] * cfg.buckets + (w1 % np.uint32(cfg.buckets))).astype(np.int32)
    hit = (w0 >> np.uint32(32 - cfg.difficulty_bits)) == 0
    first = hit.argmax(1)
    return np.where(hit.any(1), first % cfg.buckets, cfg.buckets).astype(np.int32)[:, None]


@dataclass
class SparseEvents:
    """Per-example feature-ID events. ``ids`` (N, K) int32, -1 = empty slot."""

    ids: np.ndarray
    n_features: int

    def __post_init__(self) -> None:
        if self.ids.ndim != 2:
            raise ValueError("ids must be 2-D")

    @property
    def n(self) -> int:
        return self.ids.shape[0]

    def active_per_example(self) -> float:
        return float((self.ids >= 0).sum(1).mean())

    def column_counts(self) -> np.ndarray:
        v = self.ids[self.ids >= 0]
        return np.bincount(v, minlength=self.n_features)

    def nbytes_per_example(self) -> float:
        """Storage if each event is a uint16/uint32 feature id (padding slots excluded)."""
        width = 2 if self.n_features <= 65536 else 4
        return self.active_per_example() * width

    def to_dense(self, dtype: type = np.float32) -> np.ndarray:
        F = np.zeros((self.n, self.n_features), dtype)
        r, c = np.nonzero(self.ids >= 0)
        np.add.at(F, (r, self.ids[r, c]), 1)
        return F

    def compact(self) -> "SparseEvents":
        """Drop padding: valid ids first in each row, width = max events in any row."""
        order = np.argsort(self.ids < 0, axis=1, kind="stable")
        ids = np.take_along_axis(self.ids, order, 1)
        k = int((self.ids >= 0).sum(1).max()) if self.n else 0
        return SparseEvents(np.ascontiguousarray(ids[:, :max(k, 1)]), self.n_features)

    def take(self, rows: np.ndarray) -> "SparseEvents":
        return SparseEvents(self.ids[rows], self.n_features)

    @staticmethod
    def hstack(parts: list["SparseEvents"]) -> "SparseEvents":
        off, cols = 0, []
        for p in parts:
            cols.append(np.where(p.ids >= 0, p.ids + off, -1))
            off += p.n_features
        return SparseEvents(np.concatenate(cols, 1).astype(np.int32), off)


class SparseRidge:
    """Ridge readout on sparse events. Same estimator as :class:`RidgeReadout`
    (centered primal solve), but the Gram matrix and predictions are built from
    feature IDs: training cost O(N K^2), inference = gather-sum of K weight rows.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.muW: np.ndarray | None = None

    @staticmethod
    def gram(ev: SparseEvents) -> np.ndarray:
        ev = ev.compact()
        D, K = ev.n_features, ev.ids.shape[1]
        chunk = max(1, 20_000_000 // (K * K))
        G = np.zeros(D * D, np.float64)
        for i in range(0, ev.n, chunk):
            ids = ev.ids[i:i + chunk].astype(np.int64)
            a = ids[:, :, None]
            b = ids[:, None, :]
            ok = (a >= 0) & (b >= 0)
            G += np.bincount((a * D + b)[ok], minlength=D * D)
        return G.reshape(D, D)

    def fit(self, ev: SparseEvents, Y: np.ndarray) -> "SparseRidge":
        Y = np.asarray(Y, np.float64)
        Y = Y[:, None] if Y.ndim == 1 else Y
        N, D = ev.n, ev.n_features
        cnt = ev.column_counts().astype(np.float64)
        self.mu = cnt / N
        ym = Y.mean(0)
        G = self.gram(ev) - N * np.outer(self.mu, self.mu)
        FtY = np.zeros((D, Y.shape[1]))
        r, c = np.nonzero(ev.ids >= 0)
        np.add.at(FtY, ev.ids[r, c], Y[r])
        FtY -= np.outer(cnt, ym)  # = Fc^T (Y - ym)
        self.W = np.linalg.solve(G + self.alpha * np.eye(D), FtY)
        self.b = ym
        self.muW = self.mu @ self.W
        return self

    def predict(self, ev: SparseEvents, chunk: int = 256) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("readout not fitted")
        ev = ev.compact()
        Wp = np.vstack([self.W, np.zeros((1, self.W.shape[1]))])  # id -1 -> zero row
        ids = np.where(ev.ids >= 0, ev.ids, self.W.shape[0])
        out = np.empty((ev.n, self.W.shape[1]))
        for i in range(0, ev.n, chunk):
            out[i:i + chunk] = Wp[ids[i:i + chunk]].sum(1)
        return out - self.muW + self.b
