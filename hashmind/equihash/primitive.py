"""Equihash solutions as a hash primitive (the "Equihash-native" randomness/index stream).

For each payload row the controller builds deterministic work (the payload goes into
the header's free fields), the solver returns solutions, and the host concatenates the
solution indices (``index_bits`` each) into a bit stream that is cut into 32-bit words.
This is exactly what a host could do with ``mining.submit`` results. Default parameters
are a small toy instance (48, 5) so CPU emulation is fast; the statistics of the stream
(uniform pseudo-random indices) do not depend on (n, k).
"""

from __future__ import annotations

import numpy as np

from ..core.primitives import FeaturePrimitive
from .reference import EquihashParams, nonce_bytes, solve


class EquihashPrimitive(FeaturePrimitive):
    name = "equihash"
    asic_native = True  # on an Equihash ASIC
    cryptographic = True

    def __init__(self, params: EquihashParams = EquihashParams(48, 5)) -> None:
        self.p = params
        self.solves = 0
        self.solutions = 0

    def _stream(self, header: bytes, n_bits: int) -> int:
        acc, have, v = 0, 0, 0
        while have < n_bits:
            for s in solve(self.p, header, nonce_bytes(v)).solutions:
                for i in s:
                    acc = (acc << self.p.index_bits) | i
                    have += self.p.index_bits
                self.solutions += 1
            self.solves += 1
            v += 1
        return acc >> (have - n_bits)

    def words(self, node_seed: int, payloads: np.ndarray, n_evals: int) -> np.ndarray:
        out = np.empty((len(payloads), n_evals, 2), np.uint32)
        nb = 64 * n_evals
        for r, pl in enumerate(np.asarray(payloads, np.uint8)):
            header = (b"HMEQ" + int(node_seed & 0xFFFFFFFF).to_bytes(4, "little") + pl.tobytes()).ljust(108, b"\0")
            x = self._stream(header, nb)
            w = np.frombuffer(x.to_bytes(nb // 8, "big"), ">u4").astype(np.uint32)
            out[r] = w.reshape(n_evals, 2)
        return out
