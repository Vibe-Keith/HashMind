"""Cell and feature statistics that test the phase-3 failure hypothesis.

A hashed node can only help a linear readout if the cells it sees at test time
were seen (several times) during training. These statistics measure exactly
that, per node, then average over nodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .sparse_features import SparseEvents


def cell_stats(cells_tr: list[np.ndarray], cells_te: list[np.ndarray] | None = None,
               bucket_of: list[np.ndarray] | None = None) -> dict[str, Any]:
    """Per-node cell keys (N,) -> averaged statistics.

    total_examples          training examples
    unique_cells            mean distinct cells per node (train)
    unique_cell_fraction    unique_cells / total_examples
    mean/median examples per cell (train)
    singleton_cell_fraction fraction of a node's distinct cells seen exactly once
    repeated_cell_rate      fraction of training examples whose cell occurs >= 2 times
    test_coverage           fraction of test examples whose cell was seen in training
    effective_cardinality   total distinct (node, cell) pairs in training
    collision_rate          (bucket outputs) fraction of distinct cells sharing a bucket
                            with another distinct cell of the same node
    """
    n = len(cells_tr[0])
    uniq, means, medians, single, repeated, cover, coll = [], [], [], [], [], [], []
    for j, c in enumerate(cells_tr):
        u, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
        uniq.append(len(u))
        means.append(cnt.mean())
        medians.append(np.median(cnt))
        single.append((cnt == 1).mean())
        repeated.append((cnt[inv] >= 2).mean())
        if cells_te is not None:
            cover.append(np.isin(cells_te[j], u).mean())
        if bucket_of is not None:
            b = bucket_of[j][np.unique(inv, return_index=True)[1]]
            _, bc = np.unique(b, return_counts=True)
            coll.append(float((bc[bc > 1]).sum()) / len(u))
    out = {
        "total_examples": n,
        "unique_cells": float(np.mean(uniq)),
        "unique_cell_fraction": float(np.mean(uniq) / n),
        "mean_examples_per_cell": float(np.mean(means)),
        "median_examples_per_cell": float(np.mean(medians)),
        "singleton_cell_fraction": float(np.mean(single)),
        "repeated_cell_rate": float(np.mean(repeated)),
        "effective_cardinality": int(np.sum(uniq)),
    }
    if cells_te is not None:
        out["test_coverage"] = float(np.mean(cover))
    if bucket_of is not None:
        out["collision_rate"] = float(np.mean(coll))
    return out


def feature_stats(F: np.ndarray | SparseEvents) -> dict[str, Any]:
    """dimensionality, active/example, sparsity, reuse rate, normalized entropy."""
    if isinstance(F, SparseEvents):
        D, counts, active = F.n_features, F.column_counts().astype(np.float64), F.active_per_example()
        n = F.n
    else:
        nz = F != 0
        D, n = F.shape[1], F.shape[0]
        counts = nz.sum(0).astype(np.float64)
        active = float(nz.sum(1).mean())
    used = counts > 0
    p = counts[used] / counts.sum() if counts.sum() else np.zeros(0)
    ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    return {
        "dimensionality": int(D),
        "active_per_example": active,
        "sparsity": 1.0 - active / D if D else 0.0,
        # fraction of features that are ever active which are active in >= 2 examples
        "feature_reuse_rate": float((counts >= 2).sum() / max(used.sum(), 1)),
        "dead_feature_fraction": float(1 - used.mean()) if D else 0.0,
        "feature_entropy_bits": ent,
        "feature_entropy_normalized": ent / np.log2(D) if D > 1 else 0.0,
        "examples": int(n),
    }
