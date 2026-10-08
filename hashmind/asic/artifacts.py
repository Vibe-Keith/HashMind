"""ASIC artifacts: what a commercial miner's normal software interface returns, modelled per family.

An ``ASICArtifact`` is the unit the HashMind compiler reasons about: not "a hash rate" but
"after this much ASIC work, the host receives these bytes, under this success condition".

Each generator emulates in software exactly the fields the real interface carries (per
docs/PHASE7_ASIC_SEARCH.md) and nothing more; ``internal`` keeps the reference algorithm's
intermediates separately, only for analysis (never available from hardware).
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


class ArtifactType(str, Enum):
    HASH = "HASH"
    NONCE = "NONCE"
    MIXHASH = "MIXHASH"
    SOLUTION_INDICES = "SOLUTION_INDICES"
    GRAPH_WITNESS = "GRAPH_WITNESS"
    MIDSTATE = "MIDSTATE"
    PARTIAL_STATE = "PARTIAL_STATE"
    OTHER = "OTHER"


@dataclass
class ASICArtifact:
    backend_name: str
    input_description: str
    artifact_type: ArtifactType
    artifact_bytes: bytes  # exactly what crosses the software interface (beyond what the host sent)
    success_condition: str
    attempts: int  # ASIC attempts (hashes / solver runs / graphs) spent to produce it
    elapsed_time: float  # software emulation time (NOT hardware time)
    free_bits: float = 0.0  # upper bound on information bits not fixed by format/ordering constraints
    internal: dict[str, Any] = field(default_factory=dict, repr=False)  # reference-only, never on hardware


def _share_nonce(hashfn: Callable[[bytes], bytes], header: bytes, start: int, difficulty_bits: int,
                 max_attempts: int) -> tuple[int | None, int]:
    """First nonce >= start whose hash has ``difficulty_bits`` leading zero bits (an emulated share)."""
    for a in range(max_attempts):
        n = start + a
        h = hashfn(header + struct.pack("<I", n & 0xFFFFFFFF))
        if int.from_bytes(h[:4], "big") >> (32 - difficulty_bits) == 0:
            return n, a + 1
    return None, max_attempts


def sha256d_share(header: bytes, start: int = 0, difficulty_bits: int = 8, backend: str = "sha256_s9") -> ASICArtifact:
    t0 = time.perf_counter()
    f = lambda b: hashlib.sha256(hashlib.sha256(b).digest()).digest()  # noqa: E731
    n, a = _share_nonce(f, header, start, difficulty_bits, 1 << (difficulty_bits + 8))
    return ASICArtifact(backend, "80-byte header (76 chosen) ", ArtifactType.NONCE, struct.pack("<I", n or 0),
                        f"SHA256d < 2^-{difficulty_bits} (hardware floor 2^-32)", a, time.perf_counter() - t0,
                        free_bits=float(np.log2(max(a, 1))) + 1)


def blake2_share(header: bytes, start: int = 0, difficulty_bits: int = 8, backend: str = "blake2s_kadena",
                 variant: str = "blake2s") -> ASICArtifact:
    t0 = time.perf_counter()
    f = (lambda b: hashlib.blake2s(b).digest()) if variant == "blake2s" else (lambda b: hashlib.blake2b(b).digest())
    n, a = _share_nonce(f, header, start, difficulty_bits, 1 << (difficulty_bits + 8))
    return ASICArtifact(backend, "header with chosen fields", ArtifactType.NONCE, struct.pack("<I", n or 0),
                        f"{variant} < 2^-{difficulty_bits} (pool share target)", a, time.perf_counter() - t0,
                        free_bits=float(np.log2(max(a, 1))) + 1)


_ETHASH = None


def ethash_share(header: bytes, start: int = 0, difficulty_bits: int = 4, backend: str = "ethash_e9") -> ASICArtifact:
    """eth_submitWork / ETHPROXY: [nonce(8), headerHash(32, ours), mixHash(32)] for a share."""
    global _ETHASH
    from .ethash import Ethash

    if _ETHASH is None:
        _ETHASH = Ethash()
    t0 = time.perf_counter()
    hh = hashlib.sha256(header).digest()  # the 32-byte header hash we hand out as work
    for a in range(1 << (difficulty_bits + 8)):
        r = _ETHASH.hashimoto(hh, start + a)
        if int.from_bytes(r.final[:4], "big") >> (32 - difficulty_bits) == 0:
            return ASICArtifact(backend, "32-byte header hash (fully chosen) + epoch seed", ArtifactType.MIXHASH,
                                struct.pack("<Q", start + a) + r.mix_hash,
                                f"keccak256(seed||mixHash) < 2^-{difficulty_bits}", a + 1, time.perf_counter() - t0,
                                free_bits=256.0, internal={"seed": r.seed, "dag_indices": r.dag_indices,
                                                           "final": r.final})
    raise RuntimeError("no share found")


def equihash_share(header: bytes, start: int = 0, params: Any = None, backend: str = "equihash_z15") -> ASICArtifact:
    from ..equihash.reference import EquihashParams, minimal_from_indices, nonce_bytes, solve

    p = params or EquihashParams(96, 5)
    t0 = time.perf_counter()
    I = header.ljust(108, b"\0")[:108]
    for a in range(64):
        s = solve(p, I, nonce_bytes(start + a)).solutions
        if s:
            return ASICArtifact(backend, "108-byte input (96 chosen)", ArtifactType.SOLUTION_INDICES,
                                minimal_from_indices(s[0], p.index_bits), "valid Equihash solution meeting share target",
                                a + 1, time.perf_counter() - t0,
                                free_bits=float(p.solution_indices * p.index_bits - (p.solution_indices - 1)),
                                internal={"indices": s[0]})
    raise RuntimeError("no solution")


def cuckoo_share(header: bytes, start: int = 0, edge_bits: int = 14, length: int = 42,
                 backend: str = "cuckatoo_g1") -> ASICArtifact:
    from math import lgamma, log

    from .cuckoo import find_cycles

    t0 = time.perf_counter()
    for a in range(2000):
        r = find_cycles(header, start + a, edge_bits, length)
        if r.cycles:
            w = r.cycles[0]
            free = length * edge_bits - lgamma(length + 1) / log(2)  # sorted list loses log2(L!) bits
            return ASICArtifact(backend, "pre-PoW header (chosen) + nonce", ArtifactType.GRAPH_WITNESS,
                                struct.pack(f"<{length}Q", *w), f"{length}-cycle in SipHash graph", a + 1,
                                time.perf_counter() - t0, free_bits=float(free), internal={"edges": w})
    raise RuntimeError("no cycle")


GENERATORS: dict[str, Callable[..., ASICArtifact]] = {
    "sha256_s9": sha256d_share,
    "blake2s_kadena": blake2_share,
    "ethash_e9": ethash_share,
    "equihash_z15": equihash_share,
    "cuckatoo_g1": cuckoo_share,
}
