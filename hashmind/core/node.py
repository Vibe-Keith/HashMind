"""HashMindNode: deterministic encoding -> SHA-256d challenge -> features.

A node owns a small, fixed subset of input dimensions (its *input mapping*).
For each input it

1. quantizes those dimensions to discrete levels (deterministic encoding),
2. packs the level codes into the 32-byte payload of an 80-byte header whose
   other fields come from the node's seed (SHA-256 challenge generation),
3. evaluates SHA-256d over ``nonces`` consecutive nonces (CPU simulator now,
   S9 later), and
4. extracts numerical features from the result.

Because the payload only contains a few quantized dimensions, a node is a
random lookup table over a small grid cell of input space (n-tuple / WiSARD
style). Nearby inputs land in the same cell more often than distant ones, which
is what lets a linear readout generalize. A node is NOT a neuron and computes
nothing like the original layer.

ASIC-native vs. simulator-only feature modes
--------------------------------------------
A BM1387 never returns digests, only nonces whose digest meets a target.

* ``threshold`` (difficulty_bits >= 1) and ``hash_bits`` with
  ``bits_per_hash == 1`` are ASIC-native: each feature is "did nonce n pass".
* ``hash_bytes``, ``hamming``, ``bucket`` and multi-bit ``hash_bits`` need the
  digest. On real hardware the host would have to recompute it, which defeats
  the point; they exist here to compare feature quality in simulation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..backends.base import ASICBackend, HashJob, header_with_nonce, sha256d

NODE_HEADER_TAG = b"HMN1"


class FeatureMode(str, Enum):
    HASH_BITS = "hash_bits"
    HASH_BYTES = "hash_bytes"
    HAMMING = "hamming"
    BUCKET = "bucket"
    THRESHOLD = "threshold"


@dataclass(frozen=True)
class InputMapping:
    """Which input dimensions a node reads and how they are quantized.

    ``thresholds`` has shape (len(indices), levels - 1), ascending per row.
    Level code of x_i = number of thresholds below x_i, in [0, levels).
    """

    indices: tuple[int, ...]
    thresholds: np.ndarray = field(compare=False)

    def __post_init__(self) -> None:
        if len(self.indices) == 0 or len(self.indices) > 32:
            raise ValueError("a node maps 1..32 input dimensions (32-byte payload)")
        if self.thresholds.shape[0] != len(self.indices):
            raise ValueError("thresholds must have one row per mapped index")

    @property
    def levels(self) -> int:
        return self.thresholds.shape[1] + 1

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X: (N, input_dim) float -> (N, len(indices)) uint8 level codes."""
        sub = X[:, list(self.indices)]
        return (sub[:, :, None] > self.thresholds[None, :, :]).sum(axis=2).astype(np.uint8)


@dataclass(frozen=True)
class ChallengeConfig:
    mode: FeatureMode = FeatureMode.HASH_BITS
    nonces: int = 8  # SHA-256d evaluations per node per input
    bits_per_hash: int = 1  # hash_bits
    bytes_per_hash: int = 4  # hash_bytes
    buckets: int = 8  # bucket
    difficulty_bits: int = 1  # threshold: p(feature=1) = 2**-difficulty_bits
    salt: str = "hashmind-node-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", FeatureMode(self.mode))
        if self.nonces < 1:
            raise ValueError("nonces must be >= 1")
        if not 1 <= self.bits_per_hash <= 256:
            raise ValueError("bits_per_hash in [1, 256]")
        if not 1 <= self.bytes_per_hash <= 32:
            raise ValueError("bytes_per_hash in [1, 32]")
        if self.buckets < 2:
            raise ValueError("buckets >= 2")
        if not 1 <= self.difficulty_bits <= 32:
            raise ValueError("difficulty_bits in [1, 32]")

    @property
    def features_per_hash(self) -> int:
        return {
            FeatureMode.HASH_BITS: self.bits_per_hash,
            FeatureMode.HASH_BYTES: self.bytes_per_hash,
            FeatureMode.HAMMING: 1,
            FeatureMode.BUCKET: self.buckets,
            FeatureMode.THRESHOLD: 1,
        }[self.mode]

    @property
    def features_per_node(self) -> int:
        return self.nonces * self.features_per_hash

    @property
    def asic_native(self) -> bool:
        return self.mode == FeatureMode.THRESHOLD or (
            self.mode == FeatureMode.HASH_BITS and self.bits_per_hash == 1
        )

    @property
    def job_difficulty(self) -> int:
        return self.difficulty_bits if self.mode == FeatureMode.THRESHOLD else 1


def digest_features(digest: bytes, cfg: ChallengeConfig) -> np.ndarray:
    """Features from one SHA-256d digest (bitcoin byte order: digest[31] is most significant)."""
    be = digest[::-1]  # most-significant byte first, as compared against the target
    m = cfg.mode
    if m == FeatureMode.HASH_BITS:
        bits = np.unpackbits(np.frombuffer(be, np.uint8))[: cfg.bits_per_hash]
        return 1.0 - bits.astype(np.float32)  # 1 == leading bit zero == "share" for 1 bit
    if m == FeatureMode.HASH_BYTES:
        return np.frombuffer(be[: cfg.bytes_per_hash], np.uint8).astype(np.float32) / 255.0
    if m == FeatureMode.HAMMING:
        return np.array([int.from_bytes(be, "big").bit_count() / 256.0], np.float32)
    if m == FeatureMode.BUCKET:
        one = np.zeros(cfg.buckets, np.float32)
        one[int.from_bytes(be, "big") % cfg.buckets] = 1.0
        return one
    v = int.from_bytes(be, "big") >> (256 - cfg.difficulty_bits)
    return np.array([1.0 if v == 0 else 0.0], np.float32)


class HashMindNode:
    """One SHA-256 feature node. See module docstring."""

    def __init__(self, seed: int, input_mapping: InputMapping, challenge_config: ChallengeConfig) -> None:
        self.seed = int(seed) & 0xFFFFFFFF
        self.input_mapping = input_mapping
        self.challenge_config = challenge_config
        self.challenge = hashlib.sha256(
            f"{challenge_config.salt}:{self.seed}".encode()
        ).digest()

    @property
    def n_features(self) -> int:
        return self.challenge_config.features_per_node

    def payload(self, codes: np.ndarray) -> bytes:
        """Level codes for one input -> 32-byte payload (merkle-root field)."""
        return codes.astype(np.uint8).tobytes().ljust(32, b"\0")

    def header_prefix(self, payload: bytes) -> bytes:
        """76 bytes: tag | challenge | payload | seed | difficulty. Nonce appended by hasher."""
        return (
            NODE_HEADER_TAG
            + self.challenge
            + payload
            + struct.pack("<I", self.seed)
            + struct.pack("<I", self.challenge_config.job_difficulty)
        )

    def job(self, job_id: int, payload: bytes) -> HashJob:
        c = self.challenge_config
        return HashJob(job_id, self.header_prefix(payload), 0, c.nonces, c.job_difficulty)

    def features_from_nonces(self, nonces: list[int]) -> np.ndarray:
        """ASIC-native path: features from the list of passing nonces."""
        if not self.challenge_config.asic_native:
            raise ValueError(f"mode {self.challenge_config.mode.value} needs digests, not nonces")
        f = np.zeros(self.challenge_config.nonces, np.float32)
        f[nonces] = 1.0
        return f

    def features_from_digests(self, prefix: bytes) -> np.ndarray:
        """Simulator path: compute every digest on the CPU and extract features."""
        c = self.challenge_config
        return np.concatenate(
            [digest_features(sha256d(header_with_nonce(prefix, n)), c) for n in range(c.nonces)]
        )

    def evaluate(self, codes: np.ndarray, backend: ASICBackend | None = None, job_id: int = 0) -> np.ndarray:
        """Features for one input's level codes. Uses ``backend`` when the mode is ASIC-native."""
        p = self.payload(codes)
        if backend is not None and self.challenge_config.asic_native:
            res = backend.run_jobs([self.job(job_id, p)])[0]
            return self.features_from_nonces(res.nonces)
        return self.features_from_digests(self.header_prefix(p))
