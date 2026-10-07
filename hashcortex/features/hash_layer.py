"""SHA-256 n-tuple feature layer: host-side job construction + ASIC execution."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from ..architecture.config import HashCortexConfig
from ..backends.base import ASICBackend, HashJob

HEADER_VERSION_TAG = b"HCX1"


@dataclass
class TupleWiring:
    """Seeded, fixed wiring of the hash layer (stored in the .hcmodel)."""

    tuple_index: np.ndarray  # (n_tuples, tuple_bits) int32 indices into u
    challenges: np.ndarray  # (n_tuples, 32) uint8 per-job "prev block hash"
    reservoir_source: np.ndarray  # (reservoir_bits,) int32 feature indices
    reservoir_keep_mask: np.ndarray  # (reservoir_bits,) bool

    @classmethod
    def from_config(cls, cfg: HashCortexConfig) -> "TupleWiring":
        rng = np.random.default_rng(cfg.seed)
        idx = np.stack(
            [rng.choice(cfg.input_bits, cfg.tuple_bits, replace=False) for _ in range(cfg.n_tuples)]
        ).astype(np.int32)
        ch = np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(f"{cfg.challenge_salt}:{cfg.seed}:{j}".encode()).digest(), np.uint8
                )
                for j in range(cfg.n_tuples)
            ]
        )
        src = rng.choice(cfg.n_features, cfg.reservoir_bits, replace=False).astype(np.int32)
        keep = rng.random(cfg.reservoir_bits) < cfg.reservoir_keep
        return cls(idx, ch, src, keep)


def pack_tuple(bits: np.ndarray) -> bytes:
    """Pack up to 256 bits into the 32-byte merkle-root field."""
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes().ljust(32, b"\0")


def build_header_prefix(challenge: bytes, payload: bytes, tuple_id: int, difficulty_bits: int) -> bytes:
    """76-byte header: version | prev_hash | merkle_root | time | nbits."""
    assert len(challenge) == 32 and len(payload) == 32
    return (
        HEADER_VERSION_TAG
        + challenge
        + payload
        + struct.pack("<I", tuple_id)
        + struct.pack("<I", difficulty_bits)
    )


class HashFeatureLayer:
    """Turns a bit vector into binary features using an :class:`ASICBackend`."""

    def __init__(self, cfg: HashCortexConfig, wiring: TupleWiring, backend: ASICBackend) -> None:
        cfg.validate()
        self.cfg = cfg
        self.wiring = wiring
        self.backend = backend
        self._next_job = 0

    def make_jobs(self, u: np.ndarray) -> list[HashJob]:
        cfg = self.cfg
        jobs = []
        for j in range(cfg.n_tuples):
            payload = pack_tuple(u[self.wiring.tuple_index[j]])
            prefix = build_header_prefix(
                self.wiring.challenges[j].tobytes(), payload, j, cfg.difficulty_bits
            )
            jobs.append(HashJob(self._next_job, prefix, 0, cfg.nonces_per_tuple, cfg.difficulty_bits))
            self._next_job += 1
        return jobs

    def features(self, u: np.ndarray) -> np.ndarray:
        """u: (input_bits,) {0,1} -> f: (n_features,) uint8 {0,1}."""
        if u.shape != (self.cfg.input_bits,):
            raise ValueError(f"expected {self.cfg.input_bits} input bits, got {u.shape}")
        R = self.cfg.nonces_per_tuple
        f = np.zeros(self.cfg.n_features, np.uint8)
        for j, res in enumerate(self.backend.run_jobs(self.make_jobs(u))):
            for n in res.nonces:
                f[j * R + n] = 1
        return f

    def update_reservoir(self, r: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Leaky binary reservoir: kept bits shift, the rest refresh from features."""
        if r.size == 0:
            return r
        shifted = np.roll(r, 1)
        fresh = f[self.wiring.reservoir_source] ^ shifted
        return np.where(self.wiring.reservoir_keep_mask, shifted, fresh).astype(np.uint8)
