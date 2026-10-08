"""What an Equihash miner actually exposes: Zcash Stratum (ZIP 301) and the block header.

We act as the pool/controller. The only software-visible result channel of a
commercial miner (Antminer Z15 / Z15 Pro, stock firmware) is ``mining.submit``:

    params = [WORKER_NAME, JOB_ID, TIME, NONCE_2, EQUIHASH_SOLUTION]

where EQUIHASH_SOLUTION is the full minimal-encoded solution with its
compactSize prefix (1347 bytes for (200, 9): fd 40 05 || 1344 bytes). A share is
only submitted if SHA256d(header || solution) <= the pool target, and the miner
firmware may filter further (on-chip ticket mask; see docs/PHASE6_EQUIHASH_HARDWARE.md).

Header (140 bytes + solution):
    version(4) | prevhash(32) | merkleroot(32) | reserved(32) | time(4) | bits(4) | nonce(32)
I = first 108 bytes (Equihash input), V = nonce = NONCE_1 || NONCE_2.

``EquihashASICBackend.submit`` emulates that interface on top of the software solver
and returns ONLY those submit messages: no hashes, no intermediate rounds, no
solutions that fail the target.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any

from .reference import ZCASH, EquihashParams, indices_from_minimal, minimal_from_indices, solve, verify


def compact_size(n: int) -> bytes:
    if n < 0xFD:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfd" + struct.pack("<H", n)
    return b"\xfe" + struct.pack("<I", n)


def read_compact_size(b: bytes) -> tuple[int, int]:
    if b[0] < 0xFD:
        return b[0], 1
    if b[0] == 0xFD:
        return struct.unpack("<H", b[1:3])[0], 3
    return struct.unpack("<I", b[1:5])[0], 5


@dataclass(frozen=True)
class Work:
    """One job as the controller (us) sends it. 96 bytes of prevhash/merkleroot/reserved are
    arbitrary from the miner's point of view: this is the input channel for model data."""

    job_id: str
    prevhash: bytes = bytes(32)
    merkleroot: bytes = bytes(32)
    reserved: bytes = bytes(32)
    version: int = 4
    time: int = 0x5F000000
    bits: int = 0x1F07FFFF
    nonce1: bytes = bytes(4)
    target: int = (1 << 256) - 1  # pool target; max = accept every solution

    def header_prefix(self) -> bytes:
        return (struct.pack("<I", self.version) + self.prevhash + self.merkleroot + self.reserved
                + struct.pack("<I", self.time) + struct.pack("<I", self.bits))

    def notify(self, clean: bool = True) -> str:
        """ZIP-301 mining.notify line (hex fields as in the header)."""
        return json.dumps({"id": None, "method": "mining.notify", "params": [
            self.job_id, struct.pack("<I", self.version).hex(), self.prevhash.hex(), self.merkleroot.hex(),
            self.reserved.hex(), struct.pack("<I", self.time).hex(), struct.pack("<I", self.bits).hex(), clean]})

    def set_target(self) -> str:
        return json.dumps({"id": None, "method": "mining.set_target", "params": [f"{self.target:064x}"]})

    @staticmethod
    def from_payload(job_id: str, payload: bytes, **kw: Any) -> "Work":
        """Pack up to 96 bytes of arbitrary data into prevhash | merkleroot | reserved."""
        if len(payload) > 96:
            raise ValueError("at most 96 bytes of payload per job")
        p = payload.ljust(96, b"\0")
        return Work(job_id, p[:32], p[32:64], p[64:], **kw)


@dataclass
class Submit:
    worker: str
    job_id: str
    time: str
    nonce2: str
    solution: str  # hex, compactSize-prefixed minimal encoding

    def line(self, msg_id: int = 4) -> str:
        return json.dumps({"id": msg_id, "method": "mining.submit",
                           "params": [self.worker, self.job_id, self.time, self.nonce2, self.solution]})


def parse_submit(line: str) -> Submit:
    m = json.loads(line)
    if m.get("method") != "mining.submit" or len(m.get("params", [])) != 5:
        raise ValueError("not a ZIP-301 mining.submit")
    return Submit(*m["params"])


def decode_solution(sol_hex: str, p: EquihashParams = ZCASH) -> list[int]:
    raw = bytes.fromhex(sol_hex)
    n, off = read_compact_size(raw)
    body = raw[off:off + n]
    if n != p.minimal_bytes or len(body) != n:
        raise ValueError(f"solution length {n}, expected {p.minimal_bytes}")
    return indices_from_minimal(body, p.index_bits)


def header_hash(prefix: bytes, nonce: bytes, solution_minimal: bytes) -> int:
    full = prefix + nonce + compact_size(len(solution_minimal)) + solution_minimal
    d = hashlib.sha256(hashlib.sha256(full).digest()).digest()
    return int.from_bytes(d[::-1], "big")  # compared as a 256-bit number (block hash display order)


# ----------------------------------------------------------------- backends --

@dataclass
class ReferenceResult:
    """Everything Equihash computes for one (work, nonce): the theoretical information content."""

    work: Work
    nonce: bytes
    input: bytes
    hashes: Any  # (N, n/8) uint8
    solutions: list[list[int]]
    verified: list[bool]
    pow_values: list[int]
    meets_target: list[bool]
    round_sizes: list[int]
    seconds: float


class EquihashReferenceBackend:
    """Software Equihash with every intermediate artifact exposed (Section 3, 'compute')."""

    name = "equihash_reference"

    def __init__(self, params: EquihashParams = ZCASH) -> None:
        self.p = params
        self.solves = 0

    def compute(self, work: Work, nonce2: int = 0) -> ReferenceResult:
        V = work.nonce1 + nonce2.to_bytes(32 - len(work.nonce1), "little")
        I = work.header_prefix()
        r = solve(self.p, I, V, keep_hashes=True)
        self.solves += 1
        mins = [minimal_from_indices(s, self.p.index_bits) for s in r.solutions]
        pv = [header_hash(I, V, m) for m in mins]
        return ReferenceResult(work, V, I, r.hashes, r.solutions, [verify(self.p, I, V, s).valid for s in r.solutions],
                               pv, [v <= work.target for v in pv], r.round_sizes, r.seconds)


@dataclass
class ASICConfig:
    """Behaviour of the emulated miner. ``exposure``:
    full_solution  ZIP-301 behaviour: every share carries its full solution (what the protocol requires)
    nonce_only     hypothetical firmware that reports only that a share was found (no indices)
    ``ticket_bits``: extra on-device filter, as a power-of-two fraction of solutions kept (0 = none)."""

    exposure: str = "full_solution"
    ticket_bits: int = 0
    worker: str = "hashmind.0"


@dataclass
class ASICReturn:
    submits: list[Submit] = field(default_factory=list)
    nonces_tried: int = 0

    def lines(self) -> list[str]:
        return [s.line() for s in self.submits]


class EquihashASICBackend:
    """Only what a stock miner returns through Stratum (Section 3, 'submit')."""

    name = "equihash_asic"

    def __init__(self, params: EquihashParams = ZCASH, cfg: ASICConfig | None = None) -> None:
        self.p = params
        self.cfg = cfg or ASICConfig()
        self._ref = EquihashReferenceBackend(params)

    def submit(self, work: Work, nonces: int = 1, start: int = 0) -> ASICReturn:
        out = ASICReturn()
        for v in range(start, start + nonces):
            r = self._ref.compute(work, v)
            out.nonces_tried += 1
            for s, pv, ok in zip(r.solutions, r.pow_values, r.meets_target):
                if not ok or (self.cfg.ticket_bits and pv >> (256 - self.cfg.ticket_bits)):
                    continue
                mini = minimal_from_indices(s, self.p.index_bits)
                sol = (compact_size(len(mini)) + mini).hex() if self.cfg.exposure == "full_solution" else ""
                out.submits.append(Submit(self.cfg.worker, work.job_id, struct.pack("<I", work.time).hex(),
                                          r.nonce[len(work.nonce1):].hex(), sol))
        return out
