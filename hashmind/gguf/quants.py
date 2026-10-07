"""Pure-numpy dequantization of GGML tensor formats.

Native: F32, F16, BF16, F64, I8/16/32, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. Layouts follow ggml-quants.c.

IQ* (importance-matrix codebook) types need large lookup grids; for those we
delegate to the optional ``gguf`` pip package if it is installed.
"""

from __future__ import annotations

import numpy as np

from .constants import GGMLType

QK_K = 256


class UnsupportedQuantizationError(Exception):
    """Tensor type cannot be dequantized in this environment."""


def _f16(b: np.ndarray) -> np.ndarray:
    return b.copy().view("<f2").astype(np.float32)


def _blocks(raw: np.ndarray, size: int) -> np.ndarray:
    return np.frombuffer(raw.tobytes(), np.uint8).reshape(-1, size)


def _q4_0(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 18)
    d, qs = _f16(blk[:, :2]), blk[:, 2:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return (q - 8.0) * d


def _q4_1(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 20)
    d, m, qs = _f16(blk[:, :2]), _f16(blk[:, 2:4]), blk[:, 4:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32)
    return q * d + m


def _q5_high(qh_bytes: np.ndarray) -> np.ndarray:
    qh = qh_bytes.copy().view("<u4")  # (nb, 1)
    bits = (qh >> np.arange(32, dtype=np.uint32)) & 1  # (nb, 32)
    return (bits << 4).astype(np.uint8)


def _q5_0(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 22)
    d, hi, qs = _f16(blk[:, :2]), _q5_high(blk[:, 2:6]), blk[:, 6:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1) | hi
    return (q.astype(np.float32) - 16.0) * d


def _q5_1(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 24)
    d, m, hi, qs = _f16(blk[:, :2]), _f16(blk[:, 2:4]), _q5_high(blk[:, 4:8]), blk[:, 8:]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1) | hi
    return q.astype(np.float32) * d + m


def _q8_0(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 34)
    return blk[:, 2:].copy().view(np.int8).astype(np.float32) * _f16(blk[:, :2])


def _q2_k(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 84)
    sc = blk[:, :16]
    qs = blk[:, 16:80].reshape(-1, 2, 1, 32)
    d, dmin = _f16(blk[:, 80:82]), _f16(blk[:, 82:84])
    shifts = np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
    q = ((qs >> shifts) & 3).astype(np.float32)  # (nb, 2, 4, 32)
    scale = np.repeat((sc & 0xF).reshape(-1, 2, 4, 2), 16, axis=3).astype(np.float32)
    mins = np.repeat((sc >> 4).reshape(-1, 2, 4, 2), 16, axis=3).astype(np.float32)
    out = d[:, :, None, None] * scale * q - dmin[:, :, None, None] * mins
    return out.reshape(-1, QK_K)


def _q3_k(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 110)
    hmask = blk[:, :32]
    qs = blk[:, 32:96].reshape(-1, 2, 1, 32)
    sraw = blk[:, 96:108].astype(np.uint32)
    d = _f16(blk[:, 108:110])
    # Unpack 16 x 6-bit scales (see dequantize_row_q3_K).
    lo = np.concatenate([sraw[:, 0:8] & 0xF, sraw[:, 0:8] >> 4], axis=1)  # 16 low nibbles
    hi_src = sraw[:, 8:12]  # 4 bytes, 2 bits per scale
    hi = np.concatenate([(hi_src >> s) & 3 for s in (0, 2, 4, 6)], axis=1)
    scales = (lo | (hi << 4)).astype(np.int32) - 32  # (nb, 16)
    shifts = np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
    q = ((qs >> shifts) & 3).astype(np.int32)  # (nb, 2, 4, 32)
    m = np.arange(8).reshape(2, 4)
    hbit = (hmask[:, None, None, :] >> m[None, :, :, None].astype(np.uint8)) & 1
    q = q - np.where(hbit == 0, 4, 0)
    s = np.repeat(scales.reshape(-1, 2, 4, 2), 16, axis=3).astype(np.float32)
    return (d[:, :, None, None] * s * q).reshape(-1, QK_K)


def _k_scale_min(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 8 x (6-bit scale, 6-bit min) from 12 bytes (get_scale_min_k4)."""
    s = s.astype(np.uint8)
    sc = np.empty((s.shape[0], 8), np.uint8)
    mn = np.empty((s.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q4_k(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 144)
    d, dmin = _f16(blk[:, 0:2]), _f16(blk[:, 2:4])
    sc, mn = _k_scale_min(blk[:, 4:16])
    qs = blk[:, 16:].reshape(-1, 4, 32)
    q = np.stack([qs & 0xF, qs >> 4], axis=2).astype(np.float32)  # (nb, 4, 2, 32)
    s = sc.reshape(-1, 4, 2, 1)
    m = mn.reshape(-1, 4, 2, 1)
    return (d[:, :, None, None] * s * q - dmin[:, :, None, None] * m).reshape(-1, QK_K)


def _q5_k(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 176)
    d, dmin = _f16(blk[:, 0:2]), _f16(blk[:, 2:4])
    sc, mn = _k_scale_min(blk[:, 4:16])
    qh = blk[:, 16:48]
    qs = blk[:, 48:].reshape(-1, 4, 32)
    q = np.stack([qs & 0xF, qs >> 4], axis=2).astype(np.int32)  # (nb, 4, 2, 32)
    bit = np.arange(8, dtype=np.uint8).reshape(1, 4, 2, 1)
    q = q + (((qh[:, None, None, :] >> bit) & 1).astype(np.int32) << 4)
    s = sc.reshape(-1, 4, 2, 1)
    m = mn.reshape(-1, 4, 2, 1)
    return (d[:, :, None, None] * s * q - dmin[:, :, None, None] * m).reshape(-1, QK_K)


def _q6_k(raw: np.ndarray) -> np.ndarray:
    blk = _blocks(raw, 210)
    ql = blk[:, :128].reshape(-1, 2, 64)
    qh = blk[:, 128:192].reshape(-1, 2, 32)
    sc = blk[:, 192:208].copy().view(np.int8).astype(np.float32).reshape(-1, 2, 8)
    d = _f16(blk[:, 208:210])
    q1 = (ql[:, :, :32] & 0xF) | (((qh >> 0) & 3) << 4)
    q2 = (ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
    q3 = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
    q = np.stack([q1, q2, q3, q4], axis=2).astype(np.float32) - 32.0  # (nb, 2, 4, 32)
    s = np.repeat(sc.reshape(-1, 2, 4, 2), 16, axis=3)
    return (d[:, :, None, None] * s * q).reshape(-1, QK_K)


_NATIVE = {
    GGMLType.Q4_0: _q4_0,
    GGMLType.Q4_1: _q4_1,
    GGMLType.Q5_0: _q5_0,
    GGMLType.Q5_1: _q5_1,
    GGMLType.Q8_0: _q8_0,
    GGMLType.Q2_K: _q2_k,
    GGMLType.Q3_K: _q3_k,
    GGMLType.Q4_K: _q4_k,
    GGMLType.Q5_K: _q5_k,
    GGMLType.Q6_K: _q6_k,
}

_PLAIN = {
    GGMLType.F32: "<f4",
    GGMLType.F16: "<f2",
    GGMLType.F64: "<f8",
    GGMLType.I8: "i1",
    GGMLType.I16: "<i2",
    GGMLType.I32: "<i4",
}

NATIVE_TYPES: frozenset[GGMLType] = frozenset(_NATIVE) | frozenset(_PLAIN) | {GGMLType.BF16}


def has_gguf_package() -> bool:
    try:
        import gguf.quants  # noqa: F401
    except ImportError:
        return False
    return True


def can_dequantize(t: GGMLType) -> bool:
    return t in NATIVE_TYPES or has_gguf_package()


def dequantize(raw: np.ndarray, t: GGMLType, n: int) -> np.ndarray:
    """Dequantize a raw byte buffer of GGML type ``t`` into ``n`` float32 values."""
    if t in _PLAIN:
        return np.frombuffer(raw.tobytes(), _PLAIN[t], n).astype(np.float32)
    if t == GGMLType.BF16:
        u = np.frombuffer(raw.tobytes(), "<u2", n).astype(np.uint32) << 16
        return u.view(np.float32).copy()
    if t in _NATIVE:
        return _NATIVE[t](raw).reshape(-1)[:n].astype(np.float32)
    if has_gguf_package():
        from gguf.quants import dequantize as gdq
        from gguf.constants import GGMLQuantizationType

        return np.asarray(gdq(np.asarray(raw, np.uint8), GGMLQuantizationType(int(t))),
                          np.float32).reshape(-1)[:n]
    raise UnsupportedQuantizationError(
        f"dequantization of {t.name} needs the optional 'gguf' package (pip install gguf)"
    )
