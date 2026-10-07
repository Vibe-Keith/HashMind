"""GGUF format constants.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

These constants describe the binary GGUF container used by llama.cpp. We
re-implement a minimal reader rather than depend on the ``gguf`` pip package so
the prototype has no external dependencies beyond numpy.
"""

from __future__ import annotations

from enum import IntEnum

# Magic: the four ASCII bytes "GGUF" interpreted as a little-endian uint32.
GGUF_MAGIC = 0x46554747  # b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
SUPPORTED_GGUF_VERSIONS = (2, 3)


class GGUFValueType(IntEnum):
    """Metadata value types in a GGUF key/value store."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """GGML tensor data types (includes quantization formats)."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


# Block size (number of weights per quantization block) and the size in bytes
# of one encoded block, for the subset of types we can dequantize natively.
# ``type_size`` is bytes-per-block; ``block_size`` is weights-per-block.
QK_K = 256  # super-block size used by the "K" quants

GGML_BLOCK_INFO: dict[GGMLType, tuple[int, int]] = {
    # type: (block_size, type_size_bytes)
    GGMLType.F32: (1, 4),
    GGMLType.F16: (1, 2),
    GGMLType.BF16: (1, 2),
    GGMLType.F64: (1, 8),
    GGMLType.I8: (1, 1),
    GGMLType.I16: (1, 2),
    GGMLType.I32: (1, 4),
    GGMLType.I64: (1, 8),
    GGMLType.Q4_0: (32, 18),
    GGMLType.Q4_1: (32, 20),
    GGMLType.Q5_0: (32, 22),
    GGMLType.Q5_1: (32, 24),
    GGMLType.Q8_0: (32, 34),
    GGMLType.Q8_1: (32, 36),
    GGMLType.Q2_K: (QK_K, 84),
    GGMLType.Q3_K: (QK_K, 110),
    GGMLType.Q4_K: (QK_K, 144),
    GGMLType.Q5_K: (QK_K, 176),
    GGMLType.Q6_K: (QK_K, 210),
    GGMLType.Q8_K: (QK_K, 292),
}

# Which GGML types this prototype can dequantize to float32 in pure numpy.
# Types outside this set are reported by the inspector but raise a clear
# UnsupportedQuantizationError if a caller tries to materialize their values.
DEQUANTIZABLE_TYPES: frozenset[GGMLType] = frozenset(
    {
        GGMLType.F32,
        GGMLType.F16,
        GGMLType.BF16,
        GGMLType.F64,
        GGMLType.I8,
        GGMLType.I16,
        GGMLType.I32,
        GGMLType.Q8_0,
        GGMLType.Q4_0,
        GGMLType.Q4_1,
    }
)


def ggml_type_name(t: int) -> str:
    """Return a human-readable name for a GGML type id, even if unknown."""
    try:
        return GGMLType(t).name
    except ValueError:
        return f"UNKNOWN({t})"
