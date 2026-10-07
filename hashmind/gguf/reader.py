"""Minimal, dependency-free GGUF reader (numpy only).

Parses the header, metadata key/value store and tensor table, and memory-maps
tensor data lazily. Dequantization is provided for the types listed in
``hashmind.gguf.quants``; other types can still be inspected (name, shape, type,
byte size) but not materialized.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .quants import UnsupportedQuantizationError, can_dequantize, dequantize

from .constants import (
    GGML_BLOCK_INFO,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    SUPPORTED_GGUF_VERSIONS,
    GGMLType,
    GGUFValueType,
    ggml_type_name,
)


class GGUFError(Exception):
    """Malformed or unsupported GGUF file."""


_SCALAR_FMT: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.BOOL: "<?",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]  # GGUF order: dims[0] is the fastest-varying axis
    ggml_type: int
    offset: int  # relative to start of the data section

    @property
    def shape(self) -> tuple[int, ...]:
        """Numpy (row-major) shape, i.e. reversed GGUF dims."""
        return tuple(reversed(self.dims))

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return ggml_type_name(self.ggml_type)

    @property
    def n_bytes(self) -> int | None:
        try:
            block, size = GGML_BLOCK_INFO[GGMLType(self.ggml_type)]
        except (ValueError, KeyError):
            return None
        return self.n_elements // block * size


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    data_offset: int
    alignment: int
    _mmap: np.memmap | None = field(default=None, repr=False)

    def _data(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mmap

    def raw_bytes(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        nb = info.n_bytes
        if nb is None:
            raise UnsupportedQuantizationError(f"{name}: unknown size for type {info.type_name}")
        start = self.data_offset + info.offset
        return np.asarray(self._data()[start : start + nb])

    def can_dequantize(self, name: str) -> bool:
        try:
            return can_dequantize(GGMLType(self.tensors[name].ggml_type))
        except ValueError:
            return False

    def tensor(self, name: str) -> np.ndarray:
        """Return tensor ``name`` as float32 with numpy shape ``info.shape``."""
        info = self.tensors[name]
        if not self.can_dequantize(name):
            raise UnsupportedQuantizationError(
                f"{name}: dequantization of {info.type_name} not implemented"
            )
        flat = dequantize(self.raw_bytes(name), GGMLType(info.ggml_type), info.n_elements)
        return flat.reshape(info.shape)


class _Cursor:
    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        buf = self.f.read(size)
        if len(buf) != size:
            raise GGUFError("unexpected end of file")
        return struct.unpack(fmt, buf)[0]

    def string(self) -> str:
        n = self.read("<Q")
        buf = self.f.read(n)
        if len(buf) != n:
            raise GGUFError("unexpected end of file in string")
        return buf.decode("utf-8", errors="replace")

    def value(self, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.STRING:
            return self.string()
        if vtype == GGUFValueType.ARRAY:
            etype = GGUFValueType(self.read("<I"))
            count = self.read("<Q")
            if etype in _SCALAR_FMT and etype != GGUFValueType.BOOL:
                fmt = _SCALAR_FMT[etype]
                itemsize = struct.calcsize(fmt)
                buf = self.f.read(itemsize * count)
                return np.frombuffer(buf, np.dtype(fmt)).tolist()
            return [self.value(etype) for _ in range(count)]
        return self.read(_SCALAR_FMT[vtype])


def read_gguf(path: str | Path) -> GGUFFile:
    path = Path(path)
    with path.open("rb") as f:
        c = _Cursor(f)
        if c.read("<I") != GGUF_MAGIC:
            raise GGUFError(f"{path}: not a GGUF file (bad magic)")
        version = c.read("<I")
        if version not in SUPPORTED_GGUF_VERSIONS:
            raise GGUFError(f"{path}: unsupported GGUF version {version}")
        n_tensors = c.read("<Q")
        n_kv = c.read("<Q")

        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = c.string()
            vtype = GGUFValueType(c.read("<I"))
            metadata[key] = c.value(vtype)

        tensors: dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = c.string()
            n_dims = c.read("<I")
            dims = tuple(c.read("<Q") for _ in range(n_dims))
            ttype = c.read("<I")
            offset = c.read("<Q")
            tensors[name] = TensorInfo(name, dims, ttype, offset)

        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        pos = f.tell()
        data_offset = (pos + alignment - 1) // alignment * alignment

    return GGUFFile(path, version, metadata, tensors, data_offset, alignment)
