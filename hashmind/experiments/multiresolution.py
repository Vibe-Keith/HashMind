"""Phase 4 Track B: does multi-resolution structure (many small, partially overlapping
hash groups) fix the uniqueness/locality problem without learned routing?

Same evaluation budget as the current layer (128 nodes x 16 evaluations = 2048
features), plus a node-count scaling sweep (32 / 128 / 512 nodes).
"""

from __future__ import annotations

from .phase4_common import Rep, Task, controls

SCALING_NODES = (32, 128, 512)


def reps(task: Task) -> list[Rep]:
    out = list(controls(task))
    lay = "context_multires" if task.context_dim else "multires"
    for p in ("sha256d", "splitmix"):
        out.append(Rep(f"B: {p} multires bits", "hash", p, lay))
        out.append(Rep(f"B: PCA + {p} multires bits", "hash", p, lay, concat_pca=True))
    for n in SCALING_NODES:
        out.append(Rep(f"B-scale: sha256d single t2l4, {n} nodes", "hash", "sha256d", "single", n_nodes=n))
        out.append(Rep(f"B-scale: sha256d {lay}, {n} nodes", "hash", "sha256d", lay, n_nodes=n))
        out.append(Rep(f"B-scale: splitmix {lay}, {n} nodes", "hash", "splitmix", lay, n_nodes=n))
    return out
