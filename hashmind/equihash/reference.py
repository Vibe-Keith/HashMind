"""Software Equihash reference (Zcash variant) exposing every intermediate artifact.

Follows the Zcash protocol specification, section "Equihash":

* parameters (n, k); collision length c = n / (k + 1); list size N = 2**(c + 1)
* BLAKE2b, personalization b"ZcashPoW" || le32(n) || le32(k),
  digest length (512 // n) * n / 8 bytes; the digest for counter g = i // (512 // n)
  covers indices g * (512 // n) ... ; X_i is its (i mod (512 // n))-th n-bit slice
* hash input: I (header without nonce, 108 bytes in a block) || V (32-byte nonce) || le32(g)
* a solution is 2**k distinct indices such that, at every level of the binary tree,
  sibling subtrees' XORs collide on the next c bits and the left subtree's first index
  is smaller than the right's; the XOR of all 2**k hashes is zero
* minimal encoding: each index as (c + 1) bits, big-endian, concatenated
  (1344 bytes for (200, 9))

The solver is Wagner's algorithm in numpy with back-pointers instead of index
lists, so (200, 9) fits in memory. It is a correctness reference, not a fast
miner. Tested against Zcash's own (96, 5) solver and validator test vectors.
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EquihashParams:
    n: int = 200
    k: int = 9

    def __post_init__(self) -> None:
        if self.n % 8 or self.n % (self.k + 1) or self.k < 1 or self.n >= 512:
            raise ValueError("need n % 8 == 0, (k + 1) | n, n < 512")

    @property
    def c(self) -> int:  # collision bit length
        return self.n // (self.k + 1)

    @property
    def list_size(self) -> int:
        return 2 ** (self.c + 1)

    @property
    def solution_indices(self) -> int:
        return 2**self.k

    @property
    def index_bits(self) -> int:
        return self.c + 1

    @property
    def per_hash(self) -> int:
        return 512 // self.n

    @property
    def digest_bytes(self) -> int:
        return self.per_hash * self.n // 8

    @property
    def minimal_bytes(self) -> int:
        return self.solution_indices * self.index_bits // 8

    def personalization(self) -> bytes:
        return b"ZcashPoW" + struct.pack("<II", self.n, self.k)


ZCASH = EquihashParams(200, 9)


# ---------------------------------------------------------------- hashing ---

def hash_state(p: EquihashParams, I: bytes, V: bytes) -> "hashlib._Hash":
    h = hashlib.blake2b(digest_size=p.digest_bytes, person=p.personalization())
    h.update(I)
    h.update(V)
    return h


def generate_hashes(p: EquihashParams, I: bytes, V: bytes, indices: np.ndarray | None = None) -> np.ndarray:
    """(N or len(indices), n/8) uint8: X_i for every index (or the given ones)."""
    base = hash_state(p, I, V)
    nb = p.n // 8
    if indices is None:
        G = -(-p.list_size // p.per_hash)
        out = np.empty((G, p.digest_bytes), np.uint8)
        for g in range(G):
            h = base.copy()
            h.update(struct.pack("<I", g))
            out[g] = np.frombuffer(h.digest(), np.uint8)
        return out.reshape(G * p.per_hash, nb)[: p.list_size]
    idx = np.asarray(indices, np.int64)
    out = np.empty((len(idx), nb), np.uint8)
    for j, i in enumerate(idx):
        h = base.copy()
        h.update(struct.pack("<I", int(i) // p.per_hash))
        d = h.digest()
        o = (int(i) % p.per_hash) * nb
        out[j] = np.frombuffer(d[o:o + nb], np.uint8)
    return out


def _bits_to_words(X: np.ndarray) -> np.ndarray:
    """(R, n/8) big-endian bytes -> (R, W) uint64 big-endian words (zero-padded on the right)."""
    R, nb = X.shape
    W = -(-nb // 8)
    pad = np.zeros((R, W * 8), np.uint8)
    pad[:, :nb] = X
    return pad.view(">u8").astype(np.uint64)


def _field(words: np.ndarray, start: int, length: int) -> np.ndarray:
    """Bits [start, start+length) (big-endian bit order) of each row, as uint64 (length <= 32)."""
    w0, off = divmod(start, 64)
    v = words[:, w0].copy()
    if off + length <= 64:
        return (v >> np.uint64(64 - off - length)) & np.uint64((1 << length) - 1)
    hi_len = 64 - off
    lo_len = length - hi_len
    hi = v & np.uint64((1 << hi_len) - 1)
    lo = words[:, w0 + 1] >> np.uint64(64 - lo_len)
    return (hi << np.uint64(lo_len)) | lo


# ---------------------------------------------------------------- encoding --

def minimal_from_indices(indices: list[int] | np.ndarray, bits: int) -> bytes:
    acc, nacc, out = 0, 0, bytearray()
    for i in indices:
        acc = (acc << bits) | int(i)
        nacc += bits
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def indices_from_minimal(data: bytes, bits: int) -> list[int]:
    acc, nacc, out = 0, 0, []
    for b in data:
        acc = (acc << 8) | b
        nacc += 8
        if nacc >= bits:
            nacc -= bits
            out.append((acc >> nacc) & ((1 << bits) - 1))
    return out


# ------------------------------------------------------------- verification --

@dataclass
class Verification:
    valid: bool
    reason: str = "ok"
    level_collisions: list[bool] = field(default_factory=list)


def verify(p: EquihashParams, I: bytes, V: bytes, indices: list[int]) -> Verification:
    if len(indices) != p.solution_indices:
        return Verification(False, "wrong number of indices")
    if len(set(indices)) != len(indices):
        return Verification(False, "duplicate indices")
    if max(indices) >= p.list_size:
        return Verification(False, "index out of range")
    X = [int.from_bytes(bytes(r), "big") for r in generate_hashes(p, I, V, np.array(indices))]
    nodes = [(x, [i]) for x, i in zip(X, indices)]
    levels = []
    for lvl in range(p.k):
        shift = p.n - (lvl + 1) * p.c
        nxt = []
        ok_level = True
        for a, b in zip(nodes[0::2], nodes[1::2]):
            if a[1][0] >= b[1][0]:
                return Verification(False, f"ordering violated at level {lvl}", levels)
            if (a[0] ^ b[0]) >> shift != 0:
                ok_level = False
            nxt.append((a[0] ^ b[0], a[1] + b[1]))
        levels.append(ok_level)
        if not ok_level:
            return Verification(False, f"no collision at level {lvl}", levels)
        nodes = nxt
    if nodes[0][0] != 0:
        return Verification(False, "final XOR is not zero", levels)
    return Verification(True, "ok", levels)


# ------------------------------------------------------------------ solver --

@dataclass
class SolveResult:
    params: EquihashParams
    I: bytes
    V: bytes
    solutions: list[list[int]]
    candidates: int  # final-round candidates before the distinct-index filter
    round_sizes: list[int]
    seconds: float
    hashes: np.ndarray | None = None  # X_i for i < N (only if requested)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.params.n, "k": self.params.k, "solutions": self.solutions, "candidates": self.candidates,
                "round_sizes": self.round_sizes, "seconds": self.seconds}


def solve(p: EquihashParams, I: bytes, V: bytes, keep_hashes: bool = False) -> SolveResult:
    """All solutions Wagner's algorithm finds (the set Zcash's basic solver returns)."""
    t0 = time.perf_counter()
    X = generate_hashes(p, I, V)
    words = _bits_to_words(X)
    N = p.list_size
    cur_words = words
    first = np.arange(N, dtype=np.int64)  # smallest index in each row's subtree (left-most leaf)
    parents: list[tuple[np.ndarray, np.ndarray]] = []  # per round: (left row, right row) into previous level
    sizes = [N]
    for r in range(p.k):
        last = r == p.k - 1
        if last:
            key = np.stack([_field(cur_words, r * p.c, p.c), _field(cur_words, (r + 1) * p.c, p.c)], 1)
            key = key[:, 0] << np.uint64(p.c) | key[:, 1]
        else:
            key = _field(cur_words, r * p.c, p.c)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        brk = np.flatnonzero(np.diff(ks)) + 1
        starts = np.r_[0, brk]
        ends = np.r_[brk, len(ks)]
        sz = ends - starts
        L, R = [], []
        for m in np.unique(sz[sz > 1]):
            g = starts[sz == m]
            ii, jj = np.triu_indices(int(m), 1)
            L.append(order[(g[:, None] + ii[None, :]).ravel()])
            R.append(order[(g[:, None] + jj[None, :]).ravel()])
        if not L:
            parents.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
            sizes.append(0)
            cur_words = cur_words[:0]
            first = first[:0]
            break
        a, b = np.concatenate(L), np.concatenate(R)
        swap = first[a] > first[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        new_words = cur_words[a] ^ cur_words[b]
        # drop pairs whose XOR is entirely zero before the last round (trivial / duplicate subtrees)
        if not last:
            nz = new_words.any(1)
            a, b, new_words = a[nz], b[nz], new_words[nz]
        parents.append((a, b))
        first = first[a]
        cur_words = new_words
        sizes.append(len(a))
    # expand final candidates into leaf index lists
    sols: list[list[int]] = []
    if parents and len(parents) == p.k and len(parents[-1][0]):
        rows = np.arange(len(parents[-1][0]))
        leaves = rows[:, None]
        for a, b in reversed(parents):
            leaves = np.stack([a[leaves], b[leaves]], 2).reshape(len(rows), -1)
        for row in leaves:
            if len(np.unique(row)) == len(row):
                sols.append([int(x) for x in row])
    sols = sorted({tuple(s) for s in sols})
    return SolveResult(p, I, V, [list(s) for s in sols], len(parents[-1][0]) if len(parents) == p.k else 0,
                       sizes, time.perf_counter() - t0, X if keep_hashes else None)


def nonce_bytes(v: int) -> bytes:
    """Zcash test convention: ArithToUint256(v) = 32-byte little-endian integer."""
    return int(v).to_bytes(32, "little")
