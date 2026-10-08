"""Fidelity of converted logits against the frozen original model's logits."""

from __future__ import annotations

from typing import Any

import numpy as np


def _logsoftmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def _ranks(a: np.ndarray) -> np.ndarray:
    r = np.empty_like(a, dtype=np.float64)
    o = np.argsort(a, axis=1)
    r[np.arange(len(a))[:, None], o] = np.arange(a.shape[1])[None, :]
    return r


def logit_fidelity(ref: np.ndarray, out: np.ndarray, true_next: np.ndarray | None = None,
                   rank_rows: int = 256) -> dict[str, Any]:
    """ref/out: (N, V) logits at the same positions. Rank correlation on <= rank_rows rows."""
    ref = ref.reshape(-1, ref.shape[-1])
    out = out.reshape(-1, out.shape[-1])
    r1 = ref.argmax(1)
    top = {k: np.argpartition(-out, k - 1, axis=1)[:, :k] for k in (5, 10)}
    rtop = {k: np.argpartition(-ref, k - 1, axis=1)[:, :k] for k in (5, 10)}
    lr, lo = _logsoftmax(ref), _logsoftmax(out)
    pr = np.exp(lr)
    kl = (pr * (lr - lo)).sum(1)
    ce = -(pr * lo).sum(1)
    cos = (ref * out).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(out, axis=1) + 1e-30)
    idx = np.linspace(0, len(ref) - 1, min(rank_rows, len(ref))).astype(int)
    ra, rb = _ranks(ref[idx]), _ranks(out[idx])
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    spear = (ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))
    res = {
        "positions": int(len(ref)),
        "top1_agreement": float((out.argmax(1) == r1).mean()),
        "top5_agreement": float((top[5] == r1[:, None]).any(1).mean()),  # reference top-1 within converted top-5
        "top10_agreement": float((top[10] == r1[:, None]).any(1).mean()),
        "top5_set_overlap": float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top[5], rtop[5])])),
        "kl_ref_to_conv": float(kl.mean()),
        "cross_entropy_vs_ref": float(ce.mean()),
        "ref_entropy": float(-(pr * lr).sum(1).mean()),
        "logit_cosine": float(cos.mean()),
        "logit_spearman": float(spear.mean()),
        "finite": bool(np.isfinite(out).all()),
    }
    if true_next is not None:
        t = true_next.reshape(-1)
        res["top1_accuracy_true"] = float((out.argmax(1) == t).mean())
        res["ref_top1_accuracy_true"] = float((r1 == t).mean())
        res["nll_true"] = float(-lo[np.arange(len(t)), t].mean())
    return res


def generation_fidelity(ref: list[list[int]], out: list[list[int]]) -> dict[str, Any]:
    exact = [a == b for a, b in zip(ref, out)]
    prefix = []
    for a, b in zip(ref, out):
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        prefix.append(n)
    pos = np.mean([np.mean([x == y for x, y in zip(a, b)]) for a, b in zip(ref, out)])
    return {"prompts": len(ref), "exact_match_rate": float(np.mean(exact)),
            "mean_matching_prefix": float(np.mean(prefix)), "new_tokens": len(ref[0]) if ref else 0,
            "positionwise_agreement": float(pos)}
