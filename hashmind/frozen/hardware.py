"""S9 / BM1387 mapping, end-to-end cost model and break-even analysis (Phase 5E/5F).

MEASURED inputs (passed in): CPU time per token of the original model and of the
CPU simulation, CPU hash rates, per-token hash and host-operation counts.
MODELLED: everything about the S9, GPU, energy and cost. Every modelled number
carries its assumption in the output.

What the BM1387 can return: for a job (80-byte header, nonce range) it reports
only nonces whose SHA-256d meets the ticket mask; the floor is difficulty 1 =
p 2**-32 per hash (docs/PHASE3.md). So:

* one hash-derived *uniform* (dither / sampling variable / table address) costs
  one share = on average 2**32 hashes; the chip produces
  13.5e12 / 2**32 ~ 3,143 such events per second, total.
* a *Bernoulli(p)* event is only available for p = 2**-32 (or coarser windows).
"""

from __future__ import annotations

from typing import Any

from ..backends.s9_model import BM1387_MIN_DIFFICULTY_BITS, S9_HASHRATE

ASSUMPTIONS = {
    "s9_hashrate_hs": S9_HASHRATE,
    "s9_power_w": 1323.0,  # nominal 13.5 TH/s at ~0.098 J/GH (vendor nominal, not measured here)
    "s9_share_difficulty_bits": BM1387_MIN_DIFFICULTY_BITS,
    "job_bytes_out": 80,  # one header per job (midstate variant is ~45 bytes); not measured
    "event_bytes_in": 9,  # nonce + chip/job framing per returned share; not measured
    "link_bytes_per_s": {"uart_115200": 11_520, "uart_3M": 300_000},
    "cpu_power_w": 65.0,  # assumed package power of the measuring CPU; not measured
    "gpu_mem_bandwidth_bytes_per_s": 900e9,  # hypothetical data-centre GPU, bandwidth-bound decode model
    "gpu_power_w": 300.0,
    "electricity_usd_per_kwh": 0.15,
    "hypothetical_asic_note": "a SHA ASIC that returned every hash (no difficulty floor) is limited by the link",
}


def s9_events_per_second(a: dict[str, Any] = ASSUMPTIONS) -> float:
    return a["s9_hashrate_hs"] / 2.0 ** a["s9_share_difficulty_bits"]


def end_to_end(counts_per_token: dict[str, float], hashes_per_token: float, t_ref_cpu: float, t_sim_cpu: float,
               ref_macs_per_token: float, cpu_hash_rate: float, weight_bytes: float,
               a: dict[str, Any] = ASSUMPTIONS) -> dict[str, Any]:
    """Per-token cost of: original on CPU (measured), HashMind CPU sim (measured),
    HashMind with hashes on a modelled S9 (host work modelled from the measured CPU MAC rate)."""
    mac_rate = ref_macs_per_token / t_ref_cpu  # measured effective host MAC/s
    host_ops = counts_per_token.get("fp_macs", 0) + counts_per_token.get("int_macs", 0) + \
        counts_per_token.get("adds", 0) + counts_per_token.get("lut_lookups", 0)
    host_s = host_ops / mac_rate
    ev = s9_events_per_second(a)
    s9_hash_s = hashes_per_token / ev
    link = {k: hashes_per_token * (a["job_bytes_out"] + a["event_bytes_in"]) / v
            for k, v in a["link_bytes_per_s"].items()}
    s9_total = host_s + max(s9_hash_s, min(link.values()))
    gpu_s = weight_bytes / a["gpu_mem_bandwidth_bytes_per_s"]
    kwh = lambda w, s: w * s / 3.6e6  # noqa: E731
    rows = {
        "original_cpu_measured": {"s_per_token": t_ref_cpu, "J_per_token": a["cpu_power_w"] * t_ref_cpu},
        "hashmind_cpu_sim_measured": {"s_per_token": t_sim_cpu, "J_per_token": a["cpu_power_w"] * t_sim_cpu},
        "hashmind_s9_modelled": {"s_per_token": s9_total, "host_s": host_s, "s9_hash_s": s9_hash_s,
                                 "link_s": link, "J_per_token": a["cpu_power_w"] * host_s + a["s9_power_w"] * s9_hash_s},
        "hashes_on_cpu_modelled": {"s_per_token": host_s + hashes_per_token / max(cpu_hash_rate, 1e-9)},
        "original_gpu_modelled": {"s_per_token": gpu_s, "J_per_token": a["gpu_power_w"] * gpu_s,
                                  "note": "bandwidth-bound decode upper bound, weights read once per token"},
    }
    for r in rows.values():
        r["tokens_per_s"] = 1.0 / r["s_per_token"] if r["s_per_token"] else float("inf")
        if "J_per_token" in r:
            r["usd_per_million_tokens"] = kwh(1.0, r["J_per_token"]) * 1e6 * a["electricity_usd_per_kwh"]
    return {"per_token": rows, "host_ops_per_token": host_ops, "hashes_per_token": hashes_per_token,
            "host_mac_rate_measured": mac_rate, "s9_events_per_s": ev}


def break_even(mac_rate: float, cpu_hash_rate: float, a: dict[str, Any] = ASSUMPTIONS) -> dict[str, Any]:
    """How much host work one S9-provided hash would have to remove to pay for itself."""
    ev = s9_events_per_second(a)
    return {
        "s9_hash_derived_values_per_s": ev,
        "cpu_hash_derived_values_per_s_measured": cpu_hash_rate,
        "cpu_over_s9_hash_rate": cpu_hash_rate / ev,
        "host_macs_one_s9_value_must_replace": mac_rate / ev,
        "s9_power_per_value_J": a["s9_power_w"] / ev,
        "note": "the S9 only helps if each hash it supplies removes more host work than the host does in "
                "1/(S9 events/s) seconds; a CPU computing the same hash itself is the other bound",
    }
