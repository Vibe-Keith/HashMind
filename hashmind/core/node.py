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
    NONCE_BITS = "nonce_bits"


@dataclass(frozen=True)
class InputMapping:
    """Which inputs a node reads and how they are encoded.

    ``indices``/``thresholds``: continuous dimensions, quantized to levels.
    ``thresholds`` has shape (len(indices), levels - 1), ascending per row.
    Level code of x_i = number of thresholds below x_i, in [0, levels).

    ``context_indices``: columns of a discrete context array (e.g. previous
    token ids) that are hashed *exactly*. Discrete symbols need no locality:
    the readout generalizes when the same symbol recurs, as with hashed n-gram
    features. They make a node's input space large without making it sparse.
    """

    indices: tuple[int, ...]
    thresholds: np.ndarray = field(compare=False)
    context_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.indices) > 16 and self.context_indices:
            raise ValueError("with context, a node maps at most 16 continuous dimensions")
        if len(self.indices) > 32:
            raise ValueError("a node maps at most 32 continuous dimensions (32-byte payload)")
        if len(self.context_indices) > 4:
            raise ValueError("a node hashes at most 4 context symbols")
        if not self.indices and not self.context_indices:
            raise ValueError("a node needs at least one input")
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
    difficulty_bits: int = 1  # threshold / nonce_bits: share test p = 2**-difficulty_bits
    salt: str = "hashmind-node-v1"
    window: int = 1  # threshold / nonce_bits: nonces aggregated into one feature group

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
        if self.window < 1 or self.nonces % self.window:
            raise ValueError("window must be >= 1 and divide nonces")
        if self.mode == FeatureMode.NONCE_BITS and 2 ** self.bits_per_hash > self.window:
            raise ValueError("nonce_bits: bits_per_hash must be <= log2(window)")

    @property
    def features_per_hash(self) -> int:
        """Digest modes only (threshold/nonce_bits are per window, see features_per_node)."""
        return {
            FeatureMode.HASH_BITS: self.bits_per_hash,
            FeatureMode.HASH_BYTES: self.bytes_per_hash,
            FeatureMode.HAMMING: 1,
            FeatureMode.BUCKET: self.buckets,
            FeatureMode.THRESHOLD: 1,
            FeatureMode.NONCE_BITS: 1,
        }[self.mode]

    @property
    def windows(self) -> int:
        return self.nonces // self.window

    @property
    def features_per_node(self) -> int:
        if self.mode == FeatureMode.THRESHOLD:
            return self.windows
        if self.mode == FeatureMode.NONCE_BITS:
            return self.windows * (1 + self.bits_per_hash)
        return self.nonces * self.features_per_hash

    @property
    def asic_native(self) -> bool:
        return self.mode in (FeatureMode.THRESHOLD, FeatureMode.NONCE_BITS) or (
            self.mode == FeatureMode.HASH_BITS and self.bits_per_hash == 1
        )

    @property
    def job_difficulty(self) -> int:
        if self.mode in (FeatureMode.THRESHOLD, FeatureMode.NONCE_BITS):
            return self.difficulty_bits
        return 1

    def aggregate(self, passes: np.ndarray) -> np.ndarray:
        """ASIC-native features from the per-nonce share outcomes (bool, length ``nonces``).

        hash_bits / threshold(window=1): the outcomes themselves.
        threshold(window=W): 1 if any share in each window of W nonces.
        nonce_bits: per window, [found, low ``bits_per_hash`` bits of the first
            passing nonce's offset] (all zero if no share). The returned nonce
            *value* is itself hash-derived randomness, so one share yields several bits.
        """
        p = passes.reshape(self.windows, self.window)
        if self.mode == FeatureMode.NONCE_BITS:
            found = p.any(1)
            first = np.where(found, p.argmax(1), 0)
            bits = (first[:, None] >> np.arange(self.bits_per_hash)) & 1
            return np.concatenate([found[:, None], bits * found[:, None]], 1).astype(np.float32).ravel()
        return p.any(1).astype(np.float32)


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

    def payload(self, codes: np.ndarray, context: np.ndarray | None = None) -> bytes:
        """One input -> 32-byte payload (merkle-root field).

        Without context: level codes, zero-padded (phase-2 layout, unchanged).
        With context: 16 bytes of level codes + up to 4 little-endian uint32 symbols.
        """
        c = codes.astype(np.uint8).tobytes()
        if context is None or len(context) == 0:
            return c.ljust(32, b"\0")
        return c.ljust(16, b"\0") + np.asarray(context, "<u4").tobytes().ljust(16, b"\0")

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
        p = np.zeros(self.challenge_config.nonces, bool)
        p[nonces] = True
        return self.challenge_config.aggregate(p)

    def features_from_digests(self, prefix: bytes) -> np.ndarray:
        """Simulator path: compute every digest on the CPU and extract features."""
        c = self.challenge_config
        if c.asic_native:  # fast path: one share test per nonce, SHA-256 midstate reused
            d = c.job_difficulty
            mid = hashlib.sha256(prefix[:64])
            tail = prefix[64:]
            out = np.empty(c.nonces, bool)
            shift = 256 - d
            for n in range(c.nonces):
                h = mid.copy()
                h.update(tail + struct.pack("<I", n))
                out[n] = int.from_bytes(hashlib.sha256(h.digest()).digest(), "little") >> shift == 0
            return c.aggregate(out)
        return np.concatenate(
            [digest_features(sha256d(header_with_nonce(prefix, n)), c) for n in range(c.nonces)]
        )

    def evaluate(self, codes: np.ndarray, backend: ASICBackend | None = None, job_id: int = 0,
                 context: np.ndarray | None = None) -> np.ndarray:
        """Features for one input's level codes. Uses ``backend`` when the mode is ASIC-native."""
        p = self.payload(codes, context)
        if backend is not None and self.challenge_config.asic_native:
            res = backend.run_jobs([self.job(job_id, p)])[0]
            return self.features_from_nonces(res.nonces)
        return self.features_from_digests(self.header_prefix(p))
