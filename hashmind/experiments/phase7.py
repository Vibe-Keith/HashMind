"""Phase 7: commercial mining-ASIC search for HashMind.

    python -m hashmind experiment phase7 [model.gguf] -o docs/results/phase7

1. candidate registry (evidence-graded facts about each ASIC family and its software interface)
2. information-channel + shapeability experiments on each family's *interface-visible* artifact,
   produced by faithful software models (hashmind/asic/*)
3. scoring with a fixed, pre-declared formula and ranking
4. frozen TinyLlama test of the top-ranked candidate's artifact stream on one operation
   (LM-head index sampling), with the end-to-end plan cost including the device rate

Hashrate, artifacts/s and useful model operations/s are kept separate throughout.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..asic.artifacts import GENERATORS, ASICArtifact
from ..core.primitives import get_primitive, measure_throughput
from .phase4_common import environment, write_json


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 1. registry --
# rates: nominal vendor figures (aggregator listings), share floors as stated; every
# "assumed" field is an assumption, not a measurement.

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "Antminer E9 Pro", "key": "ethash_e9", "algorithm": "Ethash/Etchash", "manufacturer": "Bitmain",
        "model": "E9 Pro", "commercial_availability": "yes (2023+, ETC/ETHW firmware)", "estimated_price": "~$2-4k (used/new listings vary)",
        "hardware_interface": "Ethernet controller board -> hash boards (closed)",
        "software_interface": "Stratum: ETH-proxy only (Bitmain E9 FAQ); web UI; cgminer-style API",
        "work_input": "32-byte header hash + seed hash (epoch) + share target; header hash fully chosen by the pool",
        "returned_result": "eth_submitWork/ETHPROXY: [nonce(8), headerHash(32), mixHash(32)] per share",
        "artifact_type": "MIXHASH", "intermediate_access": "PARTIAL_INTERMEDIATE_ACCESS (mixHash only, per share)",
        "native_operations": ["keccak", "random 128-byte DAG reads", "32-bit multiply (FNV)", "xor"],
        "memory_behavior": "64 random 128-byte reads into a multi-GB DAG generated on-device from the epoch seed (not loadable with our data)",
        "hashrate_hs": 3.68e9, "power_w": 2200.0,
        "artifacts_per_s": {"share_at_2^32_hashes": 3.68e9 / 2**32, "link_bound_100Mbit_300B": 100e6 / 8 / 300},
        "evidence": ["ethminer EthStratumClient.cpp: ETHPROXY/STRATUM submit carries mixHash; EthereumStratum/1.0.0 (NiceHash) carries nonce only",
                     "Bitmain E9 FAQ: ETH proxy stratum only (support.bitmain.com/hc/en-us/articles/12711073390617)",
                     "Whether mixHash comes from silicon or is recomputed by the controller: unknown (closed firmware)"],
        "confidence": "medium (protocol), low (device internals)",
    },
    {
        "name": "iPollo G1", "key": "cuckatoo_g1", "algorithm": "Cuckatoo32 (Grin)", "manufacturer": "iPollo/Nano Labs",
        "model": "G1 (36 graphs/s, 2.8 kW)", "commercial_availability": "yes (used market)", "estimated_price": "~$1-3k",
        "hardware_interface": "Ethernet controller (closed)", "software_interface": "Grin Stratum (pool or own node)",
        "work_input": "pre-PoW header (chosen by our stratum server) + nonce",
        "returned_result": "Grin stratum submit {height, job_id, nonce, edge_bits, pow: [42 x u64]} per share",
        "artifact_type": "GRAPH_WITNESS", "intermediate_access": "WITNESS_ACCESS (full 42-edge cycle; trimmed graph not exposed)",
        "native_operations": ["siphash-2-4 (add/rotate/xor)", "edge trimming over 2^32 edges", "cycle finding"],
        "memory_behavior": "multi-GB edge trimming on-chip/on-board",
        "hashrate_hs": 36.0, "power_w": 2800.0,
        "artifacts_per_s": {"witnesses_at_min_share_diff": 36.0 / 42.0},
        "evidence": ["grin servers/src/mining/stratumserver.rs: SubmitParams{height, job_id, nonce, edge_bits, pow: Vec<u64>}; configurable minimum_share_difficulty",
                     "Grin docs: proof = sorted 42 edge indices; ~1/42 cycles per graph",
                     "iPollo G1 rated 36 GPS / 2800 W (asicminervalue, kryptex)"],
        "confidence": "medium",
    },
    {
        "name": "Antminer Z15 Pro", "key": "equihash_z15", "algorithm": "Equihash (200,9)", "manufacturer": "Bitmain",
        "model": "Z15 Pro (840 kSol/s, ~2.78 kW)", "commercial_availability": "yes (used market)", "estimated_price": "~$1-3k",
        "hardware_interface": "controller + FPGA -> BM1746 chains (closed)", "software_interface": "Zcash Stratum ZIP 301",
        "work_input": "108-byte input, 96 bytes chosen by the pool",
        "returned_result": "mining.submit with full 1344-byte solution, shares only",
        "artifact_type": "SOLUTION_INDICES", "intermediate_access": "WITNESS_ACCESS (PARTIAL_SOLUTION_ACCESS, Phase 6)",
        "native_operations": ["blake2b", "sort/collision search", "xor"], "memory_behavior": "~144 MB lists per solve",
        "hashrate_hs": 840e3, "power_w": 2780.0,
        "artifacts_per_s": {"observed_diff1_shares": 457.28 / 60.0},
        "evidence": ["docs/PHASE6_EQUIHASH_HARDWARE.md"], "confidence": "medium",
    },
    {
        "name": "Antminer KA3 / Goldshell KD-Box", "key": "blake2s_kadena", "algorithm": "Blake2s (Kadena)",
        "manufacturer": "Bitmain / Goldshell", "model": "KA3 166 TH/s 3154 W; KD-Box Pro 2.6 TH/s 230 W",
        "commercial_availability": "yes", "estimated_price": "KD-Box ~$100-300; KA3 ~$1-3k",
        "hardware_interface": "Ethernet controller (closed)", "software_interface": "Kadena stratum (mining.notify/mining.submit)",
        "work_input": "header with chosen fields; share target", "returned_result": "nonce per share (no digest, no state)",
        "artifact_type": "NONCE", "intermediate_access": "NONCE_ONLY",
        "native_operations": ["32-bit add", "xor", "rotate", "10-round permutation"], "memory_behavior": "none",
        "hashrate_hs": 166e12, "power_w": 3154.0,
        "artifacts_per_s": {"shares_at_2^32_hashes": 166e12 / 2**32},
        "evidence": ["Kadena/Alephium/Decred stratum variants submit nonce (+extranonce) only; the pool recomputes the hash",
                     "KA3 rated 166 TH/s 3154 W (miningboard)", "device share floor: unknown, 2^32 assumed for parity with S9"],
        "confidence": "medium (protocol), low (floor)",
    },
    {
        "name": "Goldshell AL-BOX (Blake3, Alephium)", "key": "blake3_al", "algorithm": "Blake3", "manufacturer": "Goldshell",
        "model": "AL-BOX 360 GH/s 180 W (II/III up to 1.25 TH/s)", "commercial_availability": "yes (2024)", "estimated_price": "~$440-1100",
        "hardware_interface": "Ethernet", "software_interface": "Alephium stratum", "work_input": "header blob + target",
        "returned_result": "nonce per share", "artifact_type": "NONCE", "intermediate_access": "NONCE_ONLY",
        "native_operations": ["32-bit add", "xor", "rotate"], "memory_behavior": "none",
        "hashrate_hs": 360e9, "power_w": 180.0, "artifacts_per_s": {"shares_at_2^32_hashes": 360e9 / 2**32},
        "evidence": ["AL-BOX 360 GH/s 180 W (asicminervalue)", "Alephium stratum submits nonce only"],
        "confidence": "medium",
    },
    {
        "name": "Antminer S9", "key": "sha256_s9", "algorithm": "SHA-256d", "manufacturer": "Bitmain",
        "model": "S9 (BM1387) 13.5 TH/s", "commercial_availability": "yes (used, cheap)", "estimated_price": "~$50-150",
        "hardware_interface": "UART chains via controller", "software_interface": "Stratum v1; open cgminer drivers",
        "work_input": "76 bytes chosen (midstate interface)", "returned_result": "nonce per share (ticket-mask floor 2^-32)",
        "artifact_type": "NONCE", "intermediate_access": "NONCE_ONLY (midstate is host-supplied, not returned)",
        "native_operations": ["32-bit add", "xor", "rotate", "SHA-256 rounds"], "memory_behavior": "none",
        "hashrate_hs": 13.5e12, "power_w": 1323.0, "artifacts_per_s": {"shares_at_2^32_hashes": 13.5e12 / 2**32},
        "evidence": ["docs/PHASE3.md (ticket mask floor)"], "confidence": "high",
    },
]

OTHER_FAMILIES = [
    {"name": "Scrypt (Antminer L7/L9)", "artifact_type": "NONCE", "status": "C", "reason": "nonce-only share interface; memory-hard internals not exposed"},
    {"name": "X11 (Antminer D9)", "artifact_type": "NONCE", "status": "C", "reason": "nonce-only; 11 chained hashes, no intermediates"},
    {"name": "Eaglesong (Goldshell CK)", "artifact_type": "NONCE", "status": "C", "reason": "nonce-only"},
    {"name": "kHeavyHash (IceRiver KS, Antminer KS)", "artifact_type": "NONCE", "status": "E", "reason": "closed by prior reverse engineering (matrix product not software-visible)"},
    {"name": "Blake-256r14 (Antminer DR5)", "artifact_type": "NONCE", "status": "C", "reason": "nonce-only; Decred moved PoW to BLAKE3 (2023) so the device class is obsolete"},
    {"name": "Cryptonight / RandomX", "artifact_type": "-", "status": "E", "reason": "no current commercial ASIC (algorithm changes targeted ASICs)"},
    {"name": "Lyra2REv2", "artifact_type": "NONCE", "status": "E", "reason": "ASICs obsolete/unavailable"},
    {"name": "Autolykos2 (Ergo)", "artifact_type": "-", "status": "E", "reason": "no confirmed commercial ASIC found"},
]


# ------------------------------------------------- 2. channel & shapeability --

def _bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, np.uint8))


def _content(a: ASICArtifact) -> bytes:
    """The part of the artifact that carries computed information (for NONCE: the nonce itself)."""
    if a.artifact_type.value == "MIXHASH":
        return a.artifact_bytes[8:]
    return a.artifact_bytes


def channel_experiment(key: str, n: int, rng: np.random.Generator, gen_kw: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
    gen = GENERATORS[key]
    kw = gen_kw or {}
    arts, t0 = [], time.perf_counter()
    inputs = [rng.integers(0, 256, 32, dtype=np.uint8).tobytes() for _ in range(n)]
    for x in inputs:
        arts.append(gen(x, **kw))
    sw_s = (time.perf_counter() - t0) / n
    C = [_content(a) for a in arts]
    L = min(len(c) for c in C)
    B = np.stack([_bits(c[:L]) for c in C])
    p1 = B.mean(0)
    bit_entropy = float(np.mean([-(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0 for p in p1]))
    repeat = 1 - len(set(C)) / len(C)
    # locality: flip one input bit
    near, far = [], []
    for x in inputs[: max(4, n // 4)]:
        y = bytearray(x)
        y[0] ^= 1
        a, b = _content(gen(x, **kw))[:L], _content(gen(bytes(y), **kw))[:L]
        c = _content(gen(rng.integers(0, 256, 32, dtype=np.uint8).tobytes(), **kw))[:L]
        near.append(float((_bits(a) != _bits(b)).mean()))
        far.append(float((_bits(a) != _bits(c)).mean()))
    # input-bit / output-bit correlation (crude mutual-information proxy)
    X = np.stack([_bits(x) for x in inputs])
    corr = np.abs(np.corrcoef(np.hstack([X[:, :32], B[:, :32]]).T)[:32, 32:])
    corr = np.nan_to_num(corr)
    # shapeability: artifacts needed until the first b content bits equal a chosen target
    shaping = []
    for bits in (0, 1, 2, 4, 8):
        if bits == 0:
            shaping.append({"bits": 0, "artifacts_needed_mean": 1.0})
            continue
        tries = []
        for t in range(6 if key in ("equihash_z15", "cuckatoo_g1") and bits >= 4 else 12):
            target = int(rng.integers(0, 2**bits))
            k, start = 0, 0
            x = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
            while True:
                k += 1
                a = gen(x + k.to_bytes(4, "little"), **kw)
                if int(_bits(_content(a)[:4])[:bits].dot(1 << np.arange(bits)[::-1])) == target:
                    break
                if k > 4 * 2**bits + 64:
                    break
            tries.append(k)
        shaping.append({"bits": bits, "artifacts_needed_mean": float(np.mean(tries)), "expected_uniform": 2.0**bits})
        if key in ("equihash_z15", "cuckatoo_g1") and bits >= 4:
            break  # too slow in software; extrapolated below
    return {"key": key, "artifacts": n, "artifact_type": arts[0].artifact_type.value,
            "artifact_bytes": len(arts[0].artifact_bytes), "free_bits_per_artifact": arts[0].free_bits,
            "attempts_per_artifact_mean": float(np.mean([a.attempts for a in arts])),
            "software_seconds_per_artifact": sw_s, "content_bit_entropy": bit_entropy, "repeat_rate": repeat,
            "locality_bitflip_hamming": float(np.mean(near)), "random_pair_hamming": float(np.mean(far)),
            "max_abs_inputbit_outputbit_corr": float(corr.max()), "mean_abs_corr": float(corr.mean()),
            "shaping": shaping, "example_hex": arts[0].artifact_bytes[:48].hex()}


# ------------------------------------------------------------- 3. scoring -----

WEIGHTS = {"availability": 1, "software_access": 2, "computational_relevance": 1, "observable_information": 2,
           "controllability": 2, "information_efficiency": 3, "rejection_cost": 2, "host_reconstruction": 1,
           "end_to_end_potential": 3}


def score(c: dict[str, Any], ch: dict[str, Any], cpu_bits_per_s: float) -> dict[str, Any]:
    rate = max(c["artifacts_per_s"].values())
    useful_bits = rate * ch["free_bits_per_artifact"]
    s = {
        "availability": 5 if "yes" in c["commercial_availability"] else 1,
        "software_access": 4 if "Stratum" in c["software_interface"] or "stratum" in c["software_interface"] else 2,
        "computational_relevance": {"MIXHASH": 3, "GRAPH_WITNESS": 3, "SOLUTION_INDICES": 3}.get(c["artifact_type"], 2),
        "observable_information": {"NONCE": 1, "MIXHASH": 2, "GRAPH_WITNESS": 3, "SOLUTION_INDICES": 3}[c["artifact_type"]],
        "controllability": 3,  # every family: our data enters only through header bytes that are then hashed
        "information_efficiency": max(0, min(5, int(5 + math.log10(useful_bits / cpu_bits_per_s)))),
        "rejection_cost": 0,  # measured: chosen bits cost ~2^b artifacts for every family (exponential)
        "host_reconstruction": 4,  # decode is cheap; but a CPU could produce equivalent pseudo-random bits itself
        "end_to_end_potential": 0 if useful_bits < cpu_bits_per_s else 2,
    }
    total = sum(WEIGHTS[k] * v for k, v in s.items())
    return {"scores": s, "weighted_total": total, "max_total": 5 * sum(WEIGHTS.values()),
            "useful_bits_per_s": useful_bits, "artifacts_per_s_best_case": rate,
            "useful_bits_vs_one_cpu_core": useful_bits / cpu_bits_per_s}


def category(c: dict[str, Any], ch: dict[str, Any], sc: dict[str, Any]) -> str:
    if c["artifact_type"] == "NONCE":
        return "C"
    exp = all(s.get("artifacts_needed_mean", 1) >= 0.5 * s.get("expected_uniform", 1) for s in ch["shaping"] if s["bits"])
    if exp and sc["useful_bits_vs_one_cpu_core"] < 1:
        return "D"
    return "B"


# ------------------------------------------------------- 4. frozen-model test --

def frozen_test(gguf: str | Path, key: str, c: dict[str, Any], logf: Callable[[str], None]) -> dict[str, Any]:
    """LM-head index sampling (1024 samples) on TinyLlama dev tokens, randomness from the candidate's
    artifact family (software model) vs CPU splitmix vs exact; end-to-end plan cost at the device rate."""
    from ..frozen.convert_ops import OpConv
    from ..frozen.metrics import logit_fidelity
    from ..frozen.runtime import FrozenRuntime, FrozenWeights
    from ..gguf.reader import read_gguf
    from ..hetero.backends import CPUBackend, Cost, HardwareBackend, Operation
    from ..hetero.plan import build_graph, plan
    from ..llm.tokenizer import SPMTokenizer
    from .phase5 import EvalData

    g = read_gguf(gguf)
    w = FrozenWeights.from_gguf(g)
    data = EvalData(SPMTokenizer.from_gguf_metadata(g.metadata), n_dev=2, n_test=1, n_new=1)
    x = data.dev
    ref = FrozenRuntime(w).forward(x)[:, :-1]
    prim = {"blake2s_kadena": "blake2s", "sha256_s9": "sha256d", "equihash_z15": "equihash"}.get(key, "blake2s")
    rows = {}
    for name, p in (("cpu splitmix", "splitmix"), (f"{key} artifact stream", prim)):
        rt = FrozenRuntime(w, {"lm_head": OpConv("sampled", samples=1024, primitive=p)})
        t0 = time.perf_counter()
        lg = rt.forward(x)[:, :-1]
        rows[name] = {"fidelity": logit_fidelity(ref, lg, x[:, 1:]), "cpu_sim_s": time.perf_counter() - t0}
        logf(f"7-frozen {name:32s} top1 {rows[name]['fidelity']['top1_agreement']:.3f}")

    class Dev(HardwareBackend):
        name = key
        primitive = prim

        def supports(self, op: Operation) -> bool:
            return op.kind in ("index_sampling", "hash_dither", "table_address")

        def estimate_cost(self, op: Operation) -> Cost:
            vals_per_s = max(c["artifacts_per_s"].values()) * 1.0  # one 32-bit value per artifact (nonce-class)
            dev = op.values / vals_per_s
            return Cost(key, dev + 2e-3, dev, latency_s=2e-3, energy_j=dev * c["power_w"], note="modelled")

    cpu = CPUBackend()
    B = {"cpu": cpu, key: Dev()}
    conv = {"lm_head": OpConv("sampled", samples=1024, primitive=prim)}
    ops = build_graph(w.hp, conv, w.layers[0]["w_in"].shape[0] // 2, w.head.shape[0], context=x.shape[1])
    exact_ops = build_graph(w.hp, {}, w.layers[0]["w_in"].shape[0] // 2, w.head.shape[0], context=x.shape[1])
    return {"operation": "lm_head index sampling, 1024 samples", "rows": rows,
            "cpu_only_exact": plan(exact_ops, B).summary(),
            "cpu_with_conversion": plan(ops, B, forced={"index_sampling": "cpu"}).summary(),
            "cpu_plus_asic": plan(ops, B, forced={"index_sampling": key}).summary()}


# ---------------------------------------------------------------- driver ------

def run_phase7(gguf: str | Path | None, out: str | Path, logf: Callable[[str], None] = log,
               quick: bool = False) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    cpu_vals = measure_throughput(get_primitive("splitmix"), 8192, 64)
    cpu_bits = cpu_vals * 32
    sizes = {"sha256_s9": 60, "blake2s_kadena": 60, "ethash_e9": 30, "equihash_z15": 16, "cuckatoo_g1": 6}
    if quick:
        sizes = {k: max(4, v // 6) for k, v in sizes.items()}
    channels = {}
    for key, n in sizes.items():
        logf(f"channel experiment {key}")
        channels[key] = channel_experiment(key, n, rng)
    ranked = []
    for c in CANDIDATES:
        ch = channels.get(c["key"])
        if ch is None:  # Blake3 (no stdlib) -> same NONCE behaviour as Blake2s
            ch = dict(channels["blake2s_kadena"], key=c["key"], note="measured on the Blake2s model (nonce artifact)")
        sc = score(c, ch, cpu_bits)
        # extrapolated shaping cost on the real device for 16 / 32 chosen bits
        rate = max(c["artifacts_per_s"].values())
        shp = {b: {"artifacts": 2.0**b, "seconds": 2.0**b / rate, "joules": 2.0**b / rate * c["power_w"]}
               for b in (0, 1, 2, 4, 8, 16, 32)}
        entry = {**{k: v for k, v in c.items() if k != "key"}, "key": c["key"], "channel": ch, **sc,
                 "shapeability": {"measured": ch["shaping"], "device_extrapolation": shp},
                 "controllability": "header fields chosen by our pool; data is hashed before any visible effect",
                 "shapeability_summary": "~2^b artifacts for b chosen bits (exponential)",
                 "estimated_artifact_rate": rate, "estimated_useful_bits_per_second": sc["useful_bits_per_s"],
                 "host_cost": "decode bytes (us); any model use = host math as before",
                 "model_use_cases": "pseudo-random dither / index sampling / table addressing only (Phases 5-6)"}
        entry["status"] = category(c, ch, sc)
        ranked.append(entry)
    ranked.sort(key=lambda e: -e["weighted_total"])
    frozen = None
    if gguf is not None:
        best = ranked[0]
        logf(f"frozen-model test with top candidate {best['key']}")
        frozen = frozen_test(gguf, best["key"], best, logf)
    res = {"environment": environment(), "cpu_values_per_s_single_core": cpu_vals, "cpu_bits_per_s": cpu_bits,
           "weights": WEIGHTS, "candidates": ranked, "other_families": OTHER_FAMILIES, "frozen_test": frozen}
    write_json(out / "asic_candidates.json", res)
    return res
