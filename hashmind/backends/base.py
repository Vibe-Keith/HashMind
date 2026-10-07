"""ASIC backend interface.

The interface models what a BM1387 actually does, not what we wish it did:

* Input: an 80-byte Bitcoin-style block header, of which the host controls the
  first 76 bytes; the chip iterates the final 4-byte nonce over a range.
* Work: double SHA-256 of each header.
* Output: ONLY the nonces whose hash meets the share target. The chip never
  returns digests. (The host can recompute any returned digest with one hash.)

On real hardware the host sends a midstate (SHA-256 state after the first 64
bytes) plus the last 12 bytes. Midstate computation needs a raw SHA-256
compression function and is deferred to the phase-2 hardware driver; the job
definition here is already compatible with it.
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

HEADER_PREFIX_LEN = 76


@dataclass(frozen=True)
class HashJob:
    job_id: int
    header_prefix: bytes  # 76 bytes; nonce is appended little-endian
    nonce_start: int
    nonce_count: int
    difficulty_bits: int  # hash passes if its top `difficulty_bits` bits are zero

    def __post_init__(self) -> None:
        if len(self.header_prefix) != HEADER_PREFIX_LEN:
            raise ValueError("header_prefix must be 76 bytes")
        if self.nonce_start < 0 or self.nonce_count < 0 or self.nonce_start + self.nonce_count > 2**32:
            raise ValueError("nonce range outside uint32")
        if not 0 <= self.difficulty_bits <= 256:
            raise ValueError("difficulty_bits must be in [0, 256]")


@dataclass
class HashResult:
    job_id: int
    nonces: list[int] = field(default_factory=list)  # passing nonces, ascending
    hashes_computed: int = 0


def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def pow_value(digest: bytes) -> int:
    """Bitcoin convention: the digest is compared as a little-endian 256-bit int."""
    return int.from_bytes(digest, "little")


def meets_target(digest: bytes, difficulty_bits: int) -> bool:
    return pow_value(digest) >> (256 - difficulty_bits) == 0 if difficulty_bits else True


def header_with_nonce(prefix: bytes, nonce: int) -> bytes:
    return prefix + struct.pack("<I", nonce)


class ASICBackend(ABC):
    """Abstract SHA-256 ASIC. Asynchronous job submit / result poll."""

    @abstractmethod
    def submit_job(self, job: HashJob) -> int:
        """Queue a job; returns its job_id."""

    @abstractmethod
    def get_result(self, job_id: int, timeout: float | None = None) -> HashResult | None:
        """Return the result for ``job_id`` or None if not ready within ``timeout``."""

    def run_jobs(self, jobs: list[HashJob]) -> list[HashResult]:
        ids = [self.submit_job(j) for j in jobs]
        out: list[HashResult] = []
        for i in ids:
            r = self.get_result(i, timeout=None)
            if r is None:
                raise TimeoutError(f"job {i} did not complete")
            out.append(r)
        return out
