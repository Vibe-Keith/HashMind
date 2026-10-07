"""Weight-preservation plan: decide, per GGUF tensor, what is kept and how.

Nothing is silently dropped. Every tensor gets at least one :class:`WeightRecord`:

PRESERVED    stored verbatim (2-D matrices as float16, 1-D as float32)
TRANSFORMED  stored as a deterministic, lower-dimensional derivative
             (PCA basis / low-rank SVD factors), with the energy retained
DISCARDED    not stored, with the reason (e.g. no dequantizer available)

Transformations
---------------
token_embd   PCA to ``reduced_dim``: basis B (d x r), mean m, reduced table (vocab x r).
             The reduced table is the HashMind layer's input representation.
output       Projected into the same basis: W_out B (vocab x r), i.e. the LM head
             restricted to the subspace HashMind sees.
blk.* 2-D    Truncated SVD to ``lowrank_rank``: U*S (out x k) and V^T (k x in).
             Residual energy is recorded as discarded.

All decompositions are deterministic: fixed seeds for the randomized SVD and a
sign convention (largest-magnitude entry of each basis vector is positive).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..gguf.reader import GGUFFile, UnsupportedQuantizationError

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class WeightRecord:
    name: str
    gguf_type: str
    original_shape: tuple[int, ...]
    action: str  # PRESERVED | TRANSFORMED | DISCARDED
    method: str
    stored_as: list[str] = field(default_factory=list)
    stored_shapes: list[tuple[int, ...]] = field(default_factory=list)
    energy_retained: float | None = None  # ||approx||_F^2 / ||W||_F^2
    note: str = ""


@dataclass
class WeightPlan:
    tensors: dict[str, np.ndarray]
    records: list[WeightRecord]

    def summary(self) -> dict[str, Any]:
        by: dict[str, dict[str, int]] = {}
        for r in self.records:
            n = int(np.prod(r.original_shape))
            d = by.setdefault(r.action, {"records": 0, "source_parameters": 0})
            d["records"] += 1
            d["source_parameters"] += n
        stored = sum(v.size for v in self.tensors.values())
        nbytes = sum(v.nbytes for v in self.tensors.values())
        return {"by_action": by, "stored_values": stored, "stored_bytes": nbytes}

    def records_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.records]


def _fix_signs(V: np.ndarray) -> np.ndarray:
    """Make the largest-|.| entry of each column positive (deterministic sign)."""
    idx = np.abs(V).argmax(axis=0)
    s = np.sign(V[idx, np.arange(V.shape[1])])
    s[s == 0] = 1
    return V * s


def pca_basis(E: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-r principal directions of rows of E. Returns (basis d x r, mean d, explained var ratio r)."""
    mean = E.mean(0, dtype=np.float64)
    C = np.zeros((E.shape[1], E.shape[1]), np.float64)
    for i in range(0, E.shape[0], 8192):  # chunked to bound memory
        X = E[i:i + 8192].astype(np.float64) - mean
        C += X.T @ X
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    B = _fix_signs(evecs[:, order[:r]])
    ratio = evals[order[:r]] / max(evals.sum(), 1e-30)
    return B.astype(np.float32), mean.astype(np.float32), ratio


def randomized_svd(W: np.ndarray, k: int, seed: int, n_iter: int = 3,
                   oversample: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic randomized truncated SVD (Halko et al. 2011)."""
    m, n = W.shape
    k = min(k, m, n)
    rng = np.random.default_rng(seed)
    A = W.astype(np.float32)
    Q = A @ rng.standard_normal((n, min(k + oversample, n))).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Q)
    Ub, S, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    U = Q @ Ub[:, :k]
    Vt = Vt[:k]
    # sign convention on V rows
    sgn = np.sign(Vt[np.arange(k), np.abs(Vt).argmax(1)])
    sgn[sgn == 0] = 1
    return U * sgn, S[:k], Vt * sgn[:, None]


def build_weight_plan(
    g: GGUFFile,
    reduced_dim: int = 32,
    lowrank_rank: int = 32,
    lowrank_layers: int | None = None,
    seed: int = 0,
    max_verbatim_other: int = 1 << 20,
) -> WeightPlan:
    tensors: dict[str, np.ndarray] = {}
    records: list[WeightRecord] = []

    def rec(info_name: str, action: str, method: str, keys: list[str] | None = None,
            energy: float | None = None, note: str = "") -> None:
        t = g.tensors[info_name]
        keys = keys or []
        records.append(WeightRecord(info_name, t.type_name, t.shape, action, method, keys,
                                    [tuple(tensors[k].shape) for k in keys], energy, note))

    def load(name: str) -> np.ndarray | None:
        try:
            return g.tensor(name)
        except UnsupportedQuantizationError as e:
            rec(name, "DISCARDED", "none", note=f"cannot dequantize: {e}")
            return None

    # --- embeddings ----------------------------------------------------------
    if "token_embd.weight" not in g.tensors:
        raise ValueError("model has no token_embd.weight")
    E = load("token_embd.weight")
    if E is None:
        raise UnsupportedQuantizationError("token_embd.weight cannot be dequantized")
    r = min(reduced_dim, E.shape[1])
    B, mean, ratio = pca_basis(E, r)
    tensors["preserved/token_embd"] = E.astype(np.float16)
    rec("token_embd.weight", "PRESERVED", "verbatim (float16)", ["preserved/token_embd"])
    tensors["transformed/embd_basis"] = B
    tensors["transformed/embd_mean"] = mean
    tensors["transformed/token_embd_reduced"] = ((E - mean) @ B).astype(np.float32)
    rec("token_embd.weight", "TRANSFORMED", f"PCA to {r} dims (HashMind input space)",
        ["transformed/embd_basis", "transformed/embd_mean", "transformed/token_embd_reduced"],
        float(ratio.sum()), "energy = explained variance of centered embeddings")

    if "output.weight" in g.tensors:
        W = load("output.weight")
        if W is not None:
            tensors["preserved/output"] = W.astype(np.float16)
            rec("output.weight", "PRESERVED", "verbatim (float16)", ["preserved/output"])
            WB = W @ B
            tensors["transformed/output_reduced"] = WB.astype(np.float32)
            energy = float((WB.astype(np.float64) ** 2).sum() / max((W.astype(np.float64) ** 2).sum(), 1e-30))
            rec("output.weight", "TRANSFORMED", f"projected onto embedding PCA basis ({r} dims)",
                ["transformed/output_reduced"], energy)

    # --- everything else -------------------------------------------------------
    for name, t in g.tensors.items():
        if name in ("token_embd.weight", "output.weight"):
            continue
        m = _LAYER_RE.match(name)
        if len(t.shape) == 1:
            v = load(name)
            if v is not None:
                key = f"preserved/{name}"
                tensors[key] = v.astype(np.float32)
                rec(name, "PRESERVED", "verbatim (float32, host-only op)", [key])
            continue
        if len(t.shape) == 2 and m:
            if lowrank_layers is not None and int(m.group(1)) >= lowrank_layers:
                rec(name, "DISCARDED", "none", note=f"layer >= lowrank_layers={lowrank_layers}")
                continue
            W = load(name)
            if W is None:
                continue
            U, S, Vt = randomized_svd(W, lowrank_rank, seed=seed + len(records))
            ku, kv = f"lowrank/{name}/US", f"lowrank/{name}/Vt"
            tensors[ku] = (U * S).astype(np.float16)
            tensors[kv] = Vt.astype(np.float16)
            total = float((W.astype(np.float64) ** 2).sum())
            energy = float((S.astype(np.float64) ** 2).sum() / max(total, 1e-30))
            rec(name, "TRANSFORMED", f"truncated SVD rank {len(S)}", [ku, kv], energy,
                f"residual {1 - energy:.1%} of Frobenius energy discarded")
            continue
        if t.n_elements <= max_verbatim_other:
            v = load(name)
            if v is not None:
                key = f"preserved/{name}"
                tensors[key] = v.astype(np.float32)
                rec(name, "PRESERVED", "verbatim (float32)", [key])
        else:
            rec(name, "DISCARDED", "none", note="large tensor with no HashMind mapping yet")
    return WeightPlan(tensors, records)


def format_weight_plan(plan: WeightPlan, max_rows: int = 30) -> str:
    lines = [f"{'tensor':<30} {'type':<8} {'action':<12} {'energy':>7}  method"]
    for r in plan.records[:max_rows]:
        e = f"{r.energy_retained:.1%}" if r.energy_retained is not None else ""
        lines.append(f"{r.name:<30} {r.gguf_type:<8} {r.action:<12} {e:>7}  {r.method} {r.note}".rstrip())
    if len(plan.records) > max_rows:
        lines.append(f"... {len(plan.records) - max_rows} more records")
    s = plan.summary()
    for a, d in s["by_action"].items():
        lines.append(f"{a:<12} records={d['records']:<5} source params={d['source_parameters']:,}")
    lines.append(f"stored: {s['stored_values']:,} values, {s['stored_bytes'] / 1e6:.1f} MB")
    return "\n".join(lines)
