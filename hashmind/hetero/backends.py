"""Hardware backends for heterogeneous frozen-model execution.

A backend says which *operations* it supports, what they cost, and executes them
(in software emulation here). The rule this module enforces: an ASIC backend only
ever executes its native primitive and returns the artifact that hardware would
return (hash-derived values, events, solution indices). Every multiply, add,
normalization or table read that turns those artifacts into tensors is a separate
CPU operation in the plan.

Operation kinds
---------------
matmul, attention, norm, elementwise, residual, embedding_lookup, lm_head   (CPU only)
hash_dither       uniforms for stochastic rounding (one 32-bit value per element)
index_sampling    uniforms / indices for Monte-Carlo matmul (one value per sample)
table_address     hash-addressed table slot per key (computed once, at conversion time)
threshold_event   Bernoulli events (p fixed by the device)

MEASURED vs MODELLED: CPU rates are measured on this machine. Every ASIC number
comes from ASIC_FACTS with its source; none of it was measured on hardware here.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..backends.s9_model import S9_HASHRATE

ARTIFACT_KINDS = ("hash_dither", "index_sampling", "table_address", "threshold_event")
CPU_KINDS = ("matmul", "attention", "norm", "elementwise", "residual", "embedding_lookup", "lm_head")

ASIC_FACTS: dict[str, dict[str, Any]] = {
    "s9": {
        "hashrate_hs": S9_HASHRATE, "power_w": 1323.0,
        "share_floor_hashes": 2.0**32,
        "values_per_share": 1,  # one hash-derived 32-bit value (nonce position) per share
        "job_bytes_out": 80, "share_bytes_in": 9,
        "link_bytes_per_s": 11_520,  # UART 115200 baud (S9 default chain baud; faster rates exist, unmeasured)
        "job_latency_s": 2e-3,  # assumed dispatch latency per job batch
        "source": "docs/PHASE3.md (BM1387 ticket-mask floor, cgminer driver-gekko.c); power: vendor nominal",
    },
    "z15pro": {
        "nominal_solutions_per_s": 840e3, "power_w": 2780.0,
        "observed_shares_per_s": 457.28 / 60.0,  # 'Work Utility (diff1 shares solved/min)' in a stock Z15 Pro log
        "indices_per_solution": 512, "index_bits": 21,
        "solution_bytes": 1347, "submit_bytes": 2 * 1347 + 120,  # hex JSON line
        "job_bytes_out": 400,  # mining.notify JSON
        "link_bytes_per_s": 100e6 / 8,  # RJ45 Ethernet, assume 100 Mbit/s effective
        "job_latency_s": 50e-3,  # assumed: stratum round trip + device pipeline
        "source": "nominal rate/power: vendor listings (asicminervalue / manual.plus Z15 Pro manual); observed share "
                  "rate: z15pro_miner.log in github.com/hashsource/hashsource_antminer_zx (third-party, unverified)",
    },
}


@dataclass
class Cost:
    backend: str
    seconds: float  # wall time attributable to this op (ASIC or CPU), per call
    device_s: float = 0.0
    transfer_s: float = 0.0
    latency_s: float = 0.0
    host_pre_s: float = 0.0
    host_post_s: float = 0.0
    energy_j: float = 0.0
    measured: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Operation:
    name: str
    kind: str
    layer: int = -1
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    values: float = 0.0  # artifact values needed per call (artifact kinds)
    flops: float = 0.0  # host arithmetic per call (CPU kinds)
    conversion: str = "exact"
    attrs: dict[str, Any] = field(default_factory=dict)


class HardwareBackend:
    name = "abstract"
    primitive = ""

    def supports(self, op: Operation) -> bool:
        raise NotImplementedError

    def estimate_cost(self, op: Operation) -> Cost:
        raise NotImplementedError

    def execute(self, op: Operation, keys: np.ndarray, n: int) -> np.ndarray:
        """Artifact ops only: (len(keys), n) uniforms in [0, 1) from this backend's primitive."""
        from ..frozen.hashsrc import HashSource, op_key

        if op.kind not in ARTIFACT_KINDS:
            raise ValueError(f"{self.name} executes artifact operations only")
        return HashSource(self.primitive).uniform(op_key(op.name, op.layer), keys, n)


def _measure(fn, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


class CPUBackend(HardwareBackend):
    """Host CPU: all tensor math, and artifacts from a fast non-cryptographic generator (splitmix)."""

    name = "cpu"
    primitive = "splitmix"

    def __init__(self, flops_per_s: float | None = None, values_per_s: float | None = None,
                 power_w: float = 65.0) -> None:
        self.flops_per_s = flops_per_s or self.measure_flops()
        self.values_per_s = values_per_s or self.measure_values()
        self.power_w = power_w

    @staticmethod
    def measure_flops() -> float:
        a = np.random.default_rng(0).standard_normal((512, 2048)).astype(np.float32)
        b = np.random.default_rng(1).standard_normal((2048, 2048)).astype(np.float32)
        return 2 * 512 * 2048 * 2048 / _measure(lambda: a @ b)

    @staticmethod
    def measure_values() -> float:
        from ..core.primitives import get_primitive, measure_throughput
        return measure_throughput(get_primitive("splitmix"), 8192, 64)

    def supports(self, op: Operation) -> bool:
        return True

    def estimate_cost(self, op: Operation) -> Cost:
        if op.kind in ARTIFACT_KINDS:
            s = op.values / self.values_per_s
        else:
            s = op.flops / self.flops_per_s
        return Cost(self.name, s, device_s=s, energy_j=s * self.power_w, measured=True,
                    note="measured CPU rate (energy uses assumed package power)")


class SHA256S9Backend(HardwareBackend):
    """Antminer S9 (BM1387): one hash-derived value per share, at the 2**-32 share floor."""

    name = "sha256_s9"
    primitive = "sha256d"

    def __init__(self, facts: dict[str, Any] | None = None) -> None:
        self.f = facts or ASIC_FACTS["s9"]

    @property
    def values_per_s(self) -> float:
        return self.f["hashrate_hs"] / self.f["share_floor_hashes"] * self.f["values_per_share"]

    def supports(self, op: Operation) -> bool:
        return op.kind in ARTIFACT_KINDS

    def estimate_cost(self, op: Operation) -> Cost:
        f = self.f
        dev = op.values / self.values_per_s
        xfer = op.values * (f["job_bytes_out"] + f["share_bytes_in"]) / f["link_bytes_per_s"]
        lat = f["job_latency_s"]
        s = max(dev, xfer) + lat
        return Cost(self.name, s, dev, xfer, lat, energy_j=dev * f["power_w"], measured=False,
                    note="modelled: share floor 2^-32, UART link, assumed latency")


class EquihashZ15Backend(HardwareBackend):
    """Antminer Z15 Pro through Stratum: full solutions for submitted shares only.

    ``visibility``: 'observed' = share rate seen in a stock-firmware log (default);
    'link_bound' = every solution reported, limited by Ethernet; 'nominal' = every
    solution at the rated 840 kSol/s with no I/O limit (an upper bound no interface provides)."""

    name = "equihash_z15"
    primitive = "equihash"

    def __init__(self, visibility: str = "observed", facts: dict[str, Any] | None = None) -> None:
        self.f = facts or ASIC_FACTS["z15pro"]
        self.visibility = visibility

    @property
    def solutions_per_s(self) -> float:
        f = self.f
        if self.visibility == "observed":
            return f["observed_shares_per_s"]
        if self.visibility == "link_bound":
            return min(f["nominal_solutions_per_s"], f["link_bytes_per_s"] / f["submit_bytes"])
        return f["nominal_solutions_per_s"]

    @property
    def values_per_solution(self) -> float:
        return self.f["indices_per_solution"] * self.f["index_bits"] / 32  # 32-bit values per solution

    @property
    def values_per_s(self) -> float:
        return self.solutions_per_s * self.values_per_solution

    def supports(self, op: Operation) -> bool:
        return op.kind in ("index_sampling", "table_address", "hash_dither")

    def estimate_cost(self, op: Operation) -> Cost:
        f = self.f
        sols = op.values / self.values_per_solution
        dev = sols / self.solutions_per_s
        xfer = sols * f["submit_bytes"] / f["link_bytes_per_s"] if self.visibility != "nominal" else 0.0
        lat = f["job_latency_s"]
        return Cost(self.name, max(dev, xfer) + lat, dev, xfer, lat, energy_j=dev * f["power_w"], measured=False,
                    note=f"modelled ({self.visibility} solution visibility)")


def default_backends(cpu: CPUBackend | None = None) -> dict[str, HardwareBackend]:
    cpu = cpu or CPUBackend()
    return {"cpu": cpu, "sha256_s9": SHA256S9Backend(), "equihash_z15": EquihashZ15Backend()}
