"""Deterministic hash-derived randomness for frozen-model conversion.

Every way SHA-256d can enter a frozen network without training reduces to one
of three uses, all provided here:

* uniform(): a hash-derived uniform number in [0, 1) per (op, row, element):
  dither for stochastic rounding, or the sampling variable of a Monte-Carlo
  matmul. On a BM1387 this is the *position* of a share in a nonce sweep.
* slot(): a hash-derived table address (hash-indexed lookup tables).
* (Bernoulli events are uniform() < p; on the chip p is fixed at 2**-32.)

Keys are content-derived (activation fingerprint) so a token's computation
depends only on its own inputs, never on batch composition.
"""

from __future__ import annotations

import numpy as np

from ..core.primitives import FeaturePrimitive, _splitmix, get_primitive, words_for_nodes

_FP_R = _splitmix(np.arange(1, 65537, dtype=np.uint64))  # fixed fingerprint multipliers


def fingerprint(x: np.ndarray) -> np.ndarray:
    """(R, d) float -> (R,) uint64 content key of the int8-quantized row."""
    x = np.asarray(x, np.float32)
    s = np.abs(x).max(1, keepdims=True)
    q = np.round(x / np.where(s > 0, s, 1) * 127).astype(np.int64).astype(np.uint64)
    d = x.shape[1]
    R = _FP_R[np.arange(d) % len(_FP_R)] ^ np.uint64(d)
    h = np.zeros(len(x), np.uint64)
    for i in range(0, d, 256):  # chunked uint64 dot (wraps mod 2**64)
        h += (q[:, i:i + 256] * R[None, i:i + 256]).sum(1, dtype=np.uint64)
    return _splitmix(h)


class HashSource:
    def __init__(self, primitive: str | FeaturePrimitive = "sha256d", chunk_rows: int = 64) -> None:
        self.prim = get_primitive(primitive)
        self.chunk_rows = chunk_rows
        self.evaluations = 0

    @property
    def name(self) -> str:
        return self.prim.name

    def _payloads(self, op_key: int, row_keys: np.ndarray) -> np.ndarray:
        rk = np.asarray(row_keys, np.uint64)
        w = np.zeros((len(rk), 8), np.uint32)
        w[:, 0] = op_key & 0xFFFFFFFF
        w[:, 1] = (op_key >> 32) & 0xFFFFFFFF
        w[:, 2] = (rk & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        w[:, 3] = (rk >> np.uint64(32)).astype(np.uint32)
        w[:, 4] = 0x484D3546  # "HM5F" domain tag
        return w.view(np.uint8).reshape(len(rk), 32)

    def uniform(self, op_key: int, row_keys: np.ndarray, n: int) -> np.ndarray:
        """(R,) keys -> (R, n) float32 uniforms; evaluation index = element index (the nonce)."""
        p = self._payloads(op_key, row_keys)
        seed = int(op_key) & 0xFFFFFFFF
        jobs = [(seed, p[i:i + self.chunk_rows], n) for i in range(0, len(p), self.chunk_rows)]
        words = words_for_nodes(self.prim, jobs)
        self.evaluations += len(p) * n
        w0 = np.concatenate([w[:, :, 0] for w in words]) if words else np.zeros((0, n), np.uint32)
        return (w0.astype(np.float64) / 2.0**32).astype(np.float32)

    def slot(self, op_key: int, keys: np.ndarray, modulus: int) -> np.ndarray:
        """Hash-addressed table slot in [0, modulus) for each integer key."""
        p = self._payloads(op_key, np.asarray(keys, np.uint64))
        words = words_for_nodes(self.prim, [(int(op_key) & 0xFFFFFFFF, p, 1)])[0]
        self.evaluations += len(p)
        return (words[:, 0, 1] % np.uint32(modulus)).astype(np.int64)


def op_key(name: str, layer: int) -> int:
    """Stable 64-bit key for an (operation, layer) application."""
    h = np.uint64(layer + 1)
    for b in name.encode():
        h = _splitmix(np.array([h ^ np.uint64(b)], np.uint64))[0]
    return int(h)
