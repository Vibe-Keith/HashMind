"""Phase 4 Track C: can learned sparse geometry (which input dims each node reads)
compensate for the absence of locality in the hash?

Greedy rewiring on train-only validation rows (hashmind/core/routing.py); every
routed row has an unrouted twin with the same layout, primitive and seed.
"""

from __future__ import annotations

from .phase4_common import Rep, Task, controls


def reps(task: Task) -> list[Rep]:
    out = list(controls(task))
    lay = "context_multires" if task.context_dim else "multires"
    for p in ("sha256d", "splitmix"):
        out.append(Rep(f"B: PCA + {p} multires bits", "hash", p, lay, concat_pca=True))
        out.append(Rep(f"C: PCA + {p} multires bits + routing", "hash", p, lay, concat_pca=True, routing=True))
    if not task.context_dim:
        for p in ("sha256d", "splitmix"):
            out.append(Rep(f"A: {p} single t4l4", "hash", p, "single", 4, 4))
            out.append(Rep(f"C: {p} single t4l4 + routing", "hash", p, "single", 4, 4, routing=True))
    return out
