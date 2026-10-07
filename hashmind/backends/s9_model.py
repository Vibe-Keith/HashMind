"""Throughput model of a real Antminer S9 used as a HashMind feature generator.

Hardware facts used (sources in docs/PHASE3.md):

* 189 BM1387 chips (3 hash boards x 63), nominal ~13.5 TH/s.
* The chip reports a nonce only if the hash meets the *ticket mask*. The lowest
  setting is mask 0 = difficulty 1, i.e. the top 32 bits of the hash are zero,
  so p(share) = 2**-32 per hash. cgminer's gekko driver forces mask 0 for the
  BM1387. The share rate cannot be pushed above about hashrate / 2**32.

Consequence: a dense, ASIC-native binary feature needs a nonce *window* of
about 2**32 hashes, which is one full nonce sweep of one header. The S9 sweeps
about hashrate / 2**32 ≈ 3,100 headers per second, so it produces ~3,100 such
features per second. ``nonce_bits`` mode extracts several bits from the
*position* of the returned nonce, multiplying the yield per header.

These are first-order estimates; job dispatch latency, UART bandwidth and
duplicate/missed nonces are not modeled and must be measured in the hardware phase.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

S9_HASHRATE = 13.5e12  # H/s, nominal
S9_CHIPS = 189
BM1387_MIN_DIFFICULTY_BITS = 32  # ticket mask 0 == difficulty 1 == 2**-32 per hash


@dataclass
class FeatureCost:
    mode: str
    difficulty_bits: int
    window_log2: float  # hashes per feature window
    features_per_window: int
    p_share_in_window: float
    hashes_per_feature: float
    features_per_second: float
    seconds_per_token: float  # for ``features_per_token`` features
    jobs_per_token: float

    def to_dict(self) -> dict:
        return asdict(self)


def feature_cost(mode: str, features_per_token: int = 2048, difficulty_bits: int = BM1387_MIN_DIFFICULTY_BITS,
                 window_log2: float | None = None, nonce_bits: int = 16,
                 hashrate: float = S9_HASHRATE) -> FeatureCost:
    """Cost of producing ASIC-native features on an S9.

    mode="threshold": 1 feature per window (1 if any share).
    mode="nonce_bits": 1 + ``nonce_bits`` features per window.
    Default window = 2**difficulty_bits hashes (p(share in window) = 1 - 1/e).
    """
    if difficulty_bits < BM1387_MIN_DIFFICULTY_BITS:
        raise ValueError(f"BM1387 cannot report shares below {BM1387_MIN_DIFFICULTY_BITS} zero bits")
    wl = float(difficulty_bits if window_log2 is None else window_log2)
    per_window = 1 if mode == "threshold" else 1 + nonce_bits
    W = 2.0 ** wl
    lam = W * 2.0 ** -difficulty_bits
    p = 1 - math.exp(-lam)
    hpf = W / per_window
    fps = hashrate / hpf
    jobs = features_per_token / per_window * max(1.0, W / 2**32)
    return FeatureCost(mode, difficulty_bits, wl, per_window, p, hpf, fps, features_per_token / fps, jobs)
