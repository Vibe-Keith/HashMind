"""Ethash (hashimoto) with the real algorithm structure and a toy-sized cache/DAG.

Same steps as the Ethash spec: Keccak-512 seeded cache with RandMemoHash rounds,
dataset items from 256 FNV-mixed cache parents, hashimoto with 64 dataset reads into
a 128-byte mix, FNV compression to the 32-byte ``mixHash``, final Keccak-256.
The real cache is ~16+ MB and the DAG ~1+ GB (epoch dependent); here they are a few KB
so it runs in Python. Statistical behaviour of mixHash (a pseudo-random function of
header and nonce) does not depend on the size; exact chain values do.

Pipeline exposed per evaluation: seed (keccak512(header||nonce)), the 64 DAG indices,
mix, mixHash, final hash. A commercial miner returns at most (nonce, header, mixHash)
per share (eth_submitWork / ETHPROXY), see docs/PHASE7_ASIC_SEARCH.md.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .keccak import keccak256, keccak512

FNV_PRIME = 0x01000193
WORD_BYTES, HASH_BYTES, MIX_BYTES = 4, 64, 128
DATASET_PARENTS, CACHE_ROUNDS, ACCESSES = 256, 3, 64


def fnv(a: int, b: int) -> int:
    return ((a * FNV_PRIME) ^ b) & 0xFFFFFFFF


def _w(b: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def _b(w: list[int]) -> bytes:
    return struct.pack(f"<{len(w)}I", *w)


@dataclass
class EthashResult:
    seed: bytes
    dag_indices: list[int]
    mix: list[int]
    mix_hash: bytes
    final: bytes


class Ethash:
    def __init__(self, seed: bytes = bytes(32), cache_items: int = 64, dataset_items: int = 2048) -> None:
        self.n_cache, self.n_full = cache_items, dataset_items
        o = [keccak512(seed)]
        for _ in range(cache_items - 1):
            o.append(keccak512(o[-1]))
        cache = [_w(x) for x in o]
        for _ in range(CACHE_ROUNDS):
            for i in range(cache_items):
                v = cache[i][0] % cache_items
                cache[i] = _w(keccak512(bytes(a ^ b for a, b in zip(_b(cache[(i - 1 + cache_items) % cache_items]),
                                                                     _b(cache[v])))))
        self.cache = cache
        self.dataset = [self._item(i) for i in range(dataset_items)]

    def _item(self, i: int) -> list[int]:
        r = HASH_BYTES // WORD_BYTES
        mix = list(self.cache[i % self.n_cache])
        mix[0] ^= i
        mix = _w(keccak512(_b(mix)))
        for j in range(DATASET_PARENTS):
            p = fnv(i ^ j, mix[j % r]) % self.n_cache
            mix = [fnv(a, b) for a, b in zip(mix, self.cache[p])]
        return _w(keccak512(_b(mix)))

    def hashimoto(self, header: bytes, nonce: int) -> EthashResult:
        n = self.n_full // (MIX_BYTES // HASH_BYTES)  # number of 128-byte pages
        w = MIX_BYTES // WORD_BYTES
        mixhashes = MIX_BYTES // HASH_BYTES
        s = keccak512(header + struct.pack("<Q", nonce))
        sw = _w(s)
        mix = sw * mixhashes
        idx = []
        for i in range(ACCESSES):
            p = fnv(i ^ sw[0], mix[i % w]) % n * mixhashes
            idx.append(p)
            newdata = []
            for j in range(mixhashes):
                newdata += self.dataset[p + j]
            mix = [fnv(a, b) for a, b in zip(mix, newdata)]
        cmix = [fnv(fnv(fnv(mix[i], mix[i + 1]), mix[i + 2]), mix[i + 3]) for i in range(0, len(mix), 4)]
        mh = _b(cmix)
        return EthashResult(s, idx, mix, mh, keccak256(s + mh))
