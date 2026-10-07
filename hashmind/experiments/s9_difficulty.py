"""Does the BM1387 difficulty floor break ASIC-native features?

The real chip only reports shares at difficulty >= 32 zero bits. We cannot
simulate 2**32 hashes per feature on a CPU, but SHA-256d outputs are
(to all known tests) uniform, so a window of W = c * 2**d nonces at difficulty d
has the same feature statistics for every d. We therefore measure accuracy at
scaled difficulties d = 1..12 with the window scaled alongside; if accuracy is
flat in d, the d = 32 hardware setting should behave the same. That extrapolation
is an assumption about SHA-256 uniformity, not a measurement on hardware.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..core.layer import HashMindLayer
from .token_probe import _evaluate, _split


@dataclass
class DifficultyRun:
    mode: str
    difficulty_bits: int
    window: int
    features: int
    accuracy: float
    density: float
    sha256d_executed: int
    seconds: float


def sweep(Z: np.ndarray, labels: np.ndarray, settings: list[dict[str, Any]], output_dim: int = 1024,
          tuple_size: int = 2, levels: int = 4, seed: int = 0) -> list[DifficultyRun]:
    rng = np.random.default_rng(seed)
    tr, te = _split(len(labels), 0.25, rng)
    out = []
    for st in settings:
        mode, d, W = st["mode"], st["difficulty_bits"], st["window"]
        per_window = 1 if mode in ("threshold", "hash_bits") else 1 + st.get("bits_per_hash", 1)
        windows = -(-st.get("features_per_node", 16) // per_window)
        kw = {} if mode == "hash_bits" else {"difficulty_bits": d, "window": W}
        if mode == "nonce_bits":
            kw["bits_per_hash"] = st["bits_per_hash"]
        layer = HashMindLayer(Z.shape[1], output_dim, seed=seed, feature_mode=mode, tuple_size=tuple_size,
                              levels=levels, nonces_per_node=windows * W, **kw)
        t0 = time.perf_counter()
        F = layer.fit_transform(Z)
        dt = time.perf_counter() - t0
        m = _evaluate(mode, F, labels, tr, te, seed)
        out.append(DifficultyRun(mode, d, W, F.shape[1], m.accuracy, float(F.mean()),
                                 layer.stats.sha256d_executed, dt))
    return out


def as_dicts(runs: list[DifficultyRun]) -> list[dict[str, Any]]:
    return [asdict(r) for r in runs]
