"""Minimal GGUF writer, used to build synthetic test models.

Supports F32, F16 and Q8_0 tensors and scalar/string/array metadata. This is
not a general-purpose llama.cpp exporter.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from .constants import GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGMLType, GGUFValueType


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _value(v: Any) -> tuple[GGUFValueType, bytes]:
    if isinstance(v, bool):
        return GGUFValueType.BOOL, struct.pack("<?", v)
    if isinstance(v, int):
        if v < 0:
            return GGUFValueType.INT64, struct.pack("<q", v)
        return GGUFValueType.UINT32 if v < 2**32 else GGUFValueType.UINT64, (
            struct.pack("<I", v) if v < 2**32 else struct.pack("<Q", v)
        )
    if isinstance(v, float):
        return GGUFValueType.FLOAT32, struct.pack("<f", v)
    if isinstance(v, str):
        return GGUFValueType.STRING, _s(v)
    if isinstance(v, (list, tuple)):
        if not v:
            return GGUFValueType.ARRAY, struct.pack("<IQ", GGUFValueType.INT32, 0)
        et, _ = _value(v[0])
        if et == GGUFValueType.UINT32 and any(isinstance(x, int) and x < 0 for x in v):
            et = GGUFValueType.INT32
        body = b"".join(
            struct.pack("<i", x) if et == GGUFValueType.INT32 else _value(x)[1] for x in v
        )
        return GGUFValueType.ARRAY, struct.pack("<IQ", et, len(v)) + body
    raise TypeError(f"unsupported metadata value: {type(v)}")


def quantize_q8_0(x: np.ndarray) -> bytes:
    flat = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(flat).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    df = d.astype(np.float32)
    q = np.where(df == 0, 0, np.round(flat / np.where(df == 0, 1, df))).astype(np.int8)
    out = np.empty((flat.shape[0], 34), np.uint8)
    out[:, :2] = d.view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def write_gguf(
    path: str | Path,
    metadata: dict[str, Any],
    tensors: dict[str, tuple[np.ndarray, GGMLType]],
    alignment: int = GGUF_DEFAULT_ALIGNMENT,
) -> None:
    """Write ``tensors`` ({name: (array, type)}) and ``metadata`` to ``path``."""
    blobs: list[bytes] = []
    infos = b""
    offset = 0
    for name, (arr, t) in tensors.items():
        if t == GGMLType.F32:
            blob = arr.astype("<f4").tobytes()
        elif t == GGMLType.F16:
            blob = arr.astype("<f2").tobytes()
        elif t == GGMLType.Q8_0:
            if arr.size % 32:
                raise ValueError(f"{name}: Q8_0 needs size divisible by 32")
            blob = quantize_q8_0(arr)
        else:
            raise ValueError(f"writer does not support {t.name}")
        dims = tuple(reversed(arr.shape))
        infos += _s(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", t, offset)
        pad = (-len(blob)) % alignment
        blobs.append(blob + b"\0" * pad)
        offset += len(blob) + pad

    kv = b""
    for k, v in metadata.items():
        vt, vb = _value(v)
        kv += _s(k) + struct.pack("<I", vt) + vb

    header = struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(metadata)) + kv + infos
    header += b"\0" * ((-len(header)) % alignment)
    with Path(path).open("wb") as f:
        f.write(header)
        for b in blobs:
            f.write(b)
