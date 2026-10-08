"""Phase 4 Track D: sparse / event-based outputs vs the dense float32 feature matrix.

    hash evaluations -> sparse feature events -> sparse or dense linear readout

All rows use the standard multi-resolution layout with the PCA input concatenated.
"""

from __future__ import annotations

from ..core.sparse_features import OutputConfig
from .phase4_common import Rep, Task, controls

OUTPUTS = (
    ("bits (dense)", OutputConfig("bits"), 16),
    ("hit d=4", OutputConfig("hit", difficulty_bits=4), 16),
    ("bucket B=16", OutputConfig("bucket", buckets=16), 1),
    ("bucket B=64", OutputConfig("bucket", buckets=64), 1),
    ("first_nonce d=4 W=64 B=16", OutputConfig("first_nonce", buckets=16, difficulty_bits=4), 64),
)


def reps(task: Task) -> list[Rep]:
    out = list(controls(task))
    lay = "context_multires" if task.context_dim else "multires"
    for p in ("sha256d", "splitmix"):
        for name, oc, ev in OUTPUTS:
            nm = f"B: PCA + {p} multires bits" if oc.mode == "bits" else f"D: PCA + {p} multires {name}"
            out.append(Rep(nm, "hash", p, lay, evals=ev, output=oc, concat_pca=True))
    return out
