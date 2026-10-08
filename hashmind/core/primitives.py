"""Interchangeable hash primitives for Phase 4 ablations.

Every primitive implements the same contract:

    words(node_seed, payloads, n_evals) -> (U, n_evals, 2) uint32

``payloads`` is a (U, 32) uint8 array: the packed cell key of one node for U
distinct inputs (same 32-byte layout as the phase-2/3 SHA-256 header payload).
Evaluation ``n`` (the "nonce") of payload ``u`` yields two 32-bit words:

* word 0: the *most significant* 32 bits of the output. A share at difficulty d
  means ``word0 >> (32 - d) == 0``. ``hash_bits`` features use d = 1.
* word 1: 32 independent low-order bits (used for digest-derived bucket IDs).

Because every primitive sees the same payload bytes, node seed and evaluation
index, swapping primitives changes nothing but the mixing function. That is the
whole point of Track A.

Primitives
----------
sha256d   SHA-256d over the exact phase-2/3 80-byte node header. Bit-exact with
          :class:`~hashmind.core.node.HashMindNode` (tested). The only one an
          S9 can compute.
fnv1a     FNV-1a 64-bit over payload | seed | nonce bytes. Classic fast
          non-cryptographic hash, weak avalanche.
splitmix  Integer mixing: payload words folded through the splitmix64
          finalizer (strong avalanche, no cryptographic claims).
pcg32     PRNG-style mapping: the cell key seeds a PCG32 generator (stream =
          node seed); evaluation n is the n-th draw.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .node import NODE_HEADER_TAG

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_U32 = np.uint64(0xFFFFFFFF)


def pack_payloads(codes: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
    """(U, tc) level codes [+ (U, tx) uint32 symbols] -> (U, 32) uint8 payloads.

    Layout is identical to :meth:`HashMindNode.payload`: without context the
    codes are zero-padded to 32 bytes; with context, 16 bytes of codes followed
    by up to four little-endian uint32 symbols.
    """
    codes = np.asarray(codes)
    U = codes.shape[0]
    out = np.zeros((U, 32), np.uint8)
    tc = codes.shape[1] if codes.ndim == 2 else 0
    if context is None or context.shape[1] == 0:
        if tc > 32:
            raise ValueError("at most 32 level codes per payload")
        out[:, :tc] = codes
        return out
    if tc > 16 or context.shape[1] > 4:
        raise ValueError("with context: at most 16 codes and 4 symbols")
    out[:, :tc] = codes
    ctx = np.ascontiguousarray(np.asarray(context, "<u4"))
    out[:, 16:16 + 4 * ctx.shape[1]] = ctx.view(np.uint8).reshape(U, -1)
    return out


class FeaturePrimitive(ABC):
    name: str = ""
    asic_native: bool = False  # can a SHA-256 ASIC compute it?
    cryptographic: bool = False

    @abstractmethod
    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        """(U, 32) uint8 -> (U, n_evals, 2) uint32. See module docstring."""

    def describe(self) -> dict:
        return {"name": self.name, "asic_native": self.asic_native, "cryptographic": self.cryptographic}


# ----------------------------------------------------------------- SHA-256d ---

def _sha_words_one_node(args: tuple[int, bytes, int, str]) -> bytes:
    seed, flat, n_evals, salt = args
    challenge = hashlib.sha256(f"{salt}:{seed}".encode()).digest()
    tail_fix = struct.pack("<I", seed) + struct.pack("<I", 1)  # difficulty field = 1 (hash_bits)
    nonce_b = [struct.pack("<I", n) for n in range(n_evals)]
    out = bytearray()
    sha = hashlib.sha256
    for i in range(0, len(flat), 32):
        prefix = NODE_HEADER_TAG + challenge + flat[i:i + 32] + tail_fix
        mid = sha(prefix[:64])
        tail = prefix[64:]
        for nb in nonce_b:
            h = mid.copy()
            h.update(tail + nb)
            d = sha(h.digest()).digest()
            out += d[28:32] + d[0:4]  # word0 = top 32 bits (LE pow value), word1 = low 32 bits
    return bytes(out)


class Sha256dPrimitive(FeaturePrimitive):
    """SHA-256d of the phase-2/3 node header. Parallel over nodes when asked."""

    name = "sha256d"
    asic_native = True
    cryptographic = True

    def __init__(self, salt: str = "hashmind-node-v1") -> None:
        self.salt = salt

    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        raw = _sha_words_one_node((int(node_seed) & 0xFFFFFFFF, np.ascontiguousarray(payloads).tobytes(),
                                   n_evals, self.salt))
        return np.frombuffer(raw, "<u4").reshape(len(payloads), n_evals, 2).astype(np.uint32)

    def words_many(self, jobs: list[tuple[int, np.ndarray, int]], workers: int | None = None) -> list[np.ndarray]:
        """Evaluate several nodes; SHA-256 work is spread over processes (pure CPU speedup)."""
        workers = workers or min(4, os.cpu_count() or 1)
        args = [(int(s) & 0xFFFFFFFF, np.ascontiguousarray(p).tobytes(), e, self.salt) for s, p, e in jobs]
        total = sum(len(p) * e for _, p, e in jobs)
        if workers <= 1 or total < 200_000:
            raws = [_sha_words_one_node(a) for a in args]
        else:
            with ProcessPoolExecutor(workers) as ex:
                raws = list(ex.map(_sha_words_one_node, args, chunksize=max(1, len(args) // (4 * workers))))
        return [np.frombuffer(r, "<u4").reshape(len(p), e, 2).astype(np.uint32)
                for r, (_, p, e) in zip(raws, jobs)]


# ------------------------------------------------------- cheap alternatives ---

def _splitmix(z: np.ndarray) -> np.ndarray:
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _split_words(h: np.ndarray) -> np.ndarray:
    return np.stack([(h >> np.uint64(32)).astype(np.uint32), (h & _U32).astype(np.uint32)], -1)


class Fnv1aPrimitive(FeaturePrimitive):
    name = "fnv1a"
    OFFSET, PRIME = np.uint64(0xCBF29CE484222325), np.uint64(0x100000001B3)

    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        U = len(payloads)
        h = np.full(U, self.OFFSET, np.uint64)
        for b in range(32):
            h = (h ^ payloads[:, b].astype(np.uint64)) * self.PRIME
        for b in struct.pack("<I", int(node_seed) & 0xFFFFFFFF):
            h = (h ^ np.uint64(b)) * self.PRIME
        n = np.arange(n_evals, dtype=np.uint32)
        H = np.repeat(h[:, None], n_evals, 1)
        for k in range(4):
            H = (H ^ ((n >> np.uint32(8 * k)) & np.uint32(0xFF)).astype(np.uint64)[None, :]) * self.PRIME
        return _split_words(H)


class SplitMixPrimitive(FeaturePrimitive):
    name = "splitmix"

    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        w = np.ascontiguousarray(payloads).view("<u4").astype(np.uint64)  # (U, 8)
        h = _splitmix(np.full(len(payloads), np.uint64(int(node_seed) & 0xFFFFFFFF), np.uint64))
        for i in range(8):
            h = _splitmix(h ^ w[:, i])
        n = np.arange(n_evals, dtype=np.uint64) * np.uint64(0xD1B54A32D192ED03)
        return _split_words(_splitmix(h[:, None] ^ n[None, :]))


class Pcg32Primitive(FeaturePrimitive):
    """PCG32 (XSH-RR). The cell key is folded with a plain polynomial (no mixer)
    into the initial state; the node seed picks the stream. Draw 2n -> word 0,
    draw 2n+1 -> word 1."""

    name = "pcg32"
    MULT = np.uint64(6364136223846793005)

    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        w = np.ascontiguousarray(payloads).view("<u4").astype(np.uint64)
        init = np.zeros(len(payloads), np.uint64)
        for i in range(8):
            init = init * np.uint64(1000003) + w[:, i]
        inc = np.uint64(((int(node_seed) & 0xFFFFFFFF) << 1) | 1)
        state = np.zeros(len(payloads), np.uint64) * self.MULT + inc
        state = (state + init) * self.MULT + inc
        out = np.empty((len(payloads), 2 * n_evals), np.uint32)
        for k in range(2 * n_evals):
            old = state
            state = old * self.MULT + inc
            xs = (((old >> np.uint64(18)) ^ old) >> np.uint64(27)) & _U32
            rot = (old >> np.uint64(59)).astype(np.uint64)
            out[:, k] = ((xs >> rot) | (xs << ((np.uint64(32) - rot) & np.uint64(31)))) & _U32
        return out.reshape(len(payloads), n_evals, 2)


PRIMITIVES: dict[str, type[FeaturePrimitive]] = {
    "sha256d": Sha256dPrimitive,
    "fnv1a": Fnv1aPrimitive,
    "splitmix": SplitMixPrimitive,
    "pcg32": Pcg32Primitive,
}


def get_primitive(name: str | FeaturePrimitive) -> FeaturePrimitive:
    if isinstance(name, FeaturePrimitive):
        return name
    try:
        return PRIMITIVES[name]()
    except KeyError:
        raise ValueError(f"unknown primitive {name!r}; choose from {sorted(PRIMITIVES)}") from None


def words_for_nodes(prim: FeaturePrimitive, jobs: list[tuple[int, np.ndarray, int]]) -> list[np.ndarray]:
    if isinstance(prim, Sha256dPrimitive):
        return prim.words_many(jobs)
    return [prim.words(s, p, e) for s, p, e in jobs]


def measure_throughput(prim: FeaturePrimitive, n_payloads: int = 4096, n_evals: int = 16,
                       seed: int = 0) -> float:
    """Single-process CPU evaluations/second on random payloads (best of 3)."""
    p = np.random.default_rng(seed).integers(0, 256, (n_payloads, 32), dtype=np.uint8)
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        prim.words(12345, p, n_evals)
        best = max(best, n_payloads * n_evals / (time.perf_counter() - t0))
    return best
