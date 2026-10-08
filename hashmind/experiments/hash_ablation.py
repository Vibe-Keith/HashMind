"""Phase 4 Track A: is SHA-256d itself contributing, or is this randomized quantized
feature expansion that any hash would provide?

Identical quantization, node wiring, tuple size, node count, evaluations per node,
feature dimensionality, readout, split and seeds; only the primitive changes.

    python -m hashmind experiment phase4-hash-ablation model.gguf -o docs/results/phase4
"""

from __future__ import annotations

from .phase4_common import Rep, Task, controls

PRIMITIVES = ("sha256d", "fnv1a", "splitmix", "pcg32")
LAYOUTS = (("t2l4", 2, 4), ("t4l4", 4, 4), ("t8l2", 8, 2))  # phase-3 locality sweep points


def reps(task: Task) -> list[Rep]:
    out = list(controls(task))
    if task.context_dim:
        for p in PRIMITIVES:
            out.append(Rep(f"A: {p} ctx-multires bits", "hash", p, "context_multires"))
        for p in PRIMITIVES:
            out.append(Rep(f"A: PCA + {p} ctx-multires bits", "hash", p, "context_multires", concat_pca=True))
        return out
    for lname, t, lv in LAYOUTS:
        for p in PRIMITIVES:
            out.append(Rep(f"A: {p} single {lname}", "hash", p, "single", t, lv))
    for p in PRIMITIVES:
        out.append(Rep(f"A: PCA + {p} single t2l4", "hash", p, "single", 2, 4, concat_pca=True))
    return out
