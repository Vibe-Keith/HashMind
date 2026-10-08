"""Frozen-conversion .hmmodel container ("hmfrozen" format).

A zip (stored, fixed timestamps, sorted entries, so the same GGUF + settings give
a byte-identical file) with:

manifest.json
    format "hmfrozen", format_version, hashmind_version, conversion_version
    source:     GGUF file name, size, sha256, architecture metadata
    conversion: weight bits / group / rounding, per-op-class conversions, layer scope
    tensors:    {name: {shape, dtype, sha256}}
tensors/<name>.npy
    blk.L.{wqkv,wo,w_in,w_down}.codes (int8/int16) + .scales (f32, per 32-input group)
    output.codes/.scales, token_embd (f32, unchanged), *.norm (f32, unchanged)

Every value is traceable to the source GGUF weights and the recorded settings.
Nothing in the file is produced from a training or calibration dataset.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import HASHMIND_VERSION
from ..gguf.reader import read_gguf
from ..llm.llama import LlamaHParams
from .convert_ops import OP_CLASSES, OpConv, dequantize_weight, quantize_weight
from .hashsrc import HashSource, fingerprint, op_key
from .runtime import FrozenWeights

FROZEN_FORMAT_VERSION = 1
FROZEN_CONVERSION_VERSION = 1
_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def weight_dither(name: str, W: np.ndarray, rounding: str) -> np.ndarray | None:
    """Weight-derived hash dither: key = (tensor name, row fingerprint ^ row index), nonce = column."""
    if rounding == "rtn":
        return None
    src = HashSource(rounding, chunk_rows=256)
    rk = fingerprint(W) ^ np.arange(W.shape[0], dtype=np.uint64)
    return src.uniform(op_key("weight:" + name, 0), rk, W.shape[1])


def quantize_frozen(w: FrozenWeights, bits: int | None, rounding: str = "rtn", group: int = 32,
                    log: Callable[[str], None] | None = None, dequant: bool = True
                    ) -> tuple[FrozenWeights | None, dict[str, np.ndarray]]:
    """Deterministically re-quantize every matrix (not embeddings / norms). bits=None: unchanged.
    ``dequant=False`` returns only the codes/scales (saves memory when writing a file)."""
    tensors: dict[str, np.ndarray] = {}
    if bits is None:
        return w, tensors
    new_layers = []
    for L, lw in enumerate(w.layers):
        d = dict(lw)
        for k in ("wqkv", "wo", "w_in", "w_down"):
            name = f"blk.{L}.{k}"
            codes, scales = quantize_weight(lw[k], bits, group, weight_dither(name, lw[k], rounding))
            if dequant:
                d[k] = dequantize_weight(codes, scales)
            tensors[name + ".codes"] = codes.astype(np.int8) if bits <= 8 else codes
            tensors[name + ".scales"] = scales
        new_layers.append(d)
        if log:
            log(f"quantized block {L}")
    codes, scales = quantize_weight(w.head, bits, group, weight_dither("output", w.head, rounding))
    tensors["output.codes"] = codes.astype(np.int8) if bits <= 8 else codes
    tensors["output.scales"] = scales
    if not dequant:
        return None, tensors
    head = dequantize_weight(codes, scales)
    return FrozenWeights(w.hp, w.embedding, head, w.output_norm, new_layers, dict(w.source)), tensors


def _npy(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, a, allow_pickle=False)
    return b.getvalue()


def convert_frozen(gguf: str | Path, out: str | Path, weight_bits: int | None = 8, weight_rounding: str = "rtn",
                   conv: dict[str, OpConv] | None = None, group: int = 32,
                   log: Callable[[str], None] | None = None, weights: FrozenWeights | None = None) -> dict[str, Any]:
    """``weights``: already-dequantized FrozenWeights of this same GGUF (avoids a second copy in memory)."""
    g = read_gguf(gguf)
    w = weights if weights is not None else FrozenWeights.from_gguf(g)
    _, qt = quantize_frozen(w, weight_bits, weight_rounding, group, log, dequant=False)
    tensors: dict[str, np.ndarray] = {"token_embd": w.embedding, "output_norm": w.output_norm}
    for L, lw in enumerate(w.layers):
        tensors[f"blk.{L}.attn_norm"] = lw["attn_norm"]
        tensors[f"blk.{L}.ffn_norm"] = lw["ffn_norm"]
        if weight_bits is None:
            for k in ("wqkv", "wo", "w_in", "w_down"):
                tensors[f"blk.{L}.{k}"] = lw[k]
    if weight_bits is None:
        tensors["output"] = w.head
    tensors.update(qt)
    md = g.metadata
    conv = conv or {}
    manifest = {
        "format": "hmfrozen", "format_version": FROZEN_FORMAT_VERSION, "hashmind_version": HASHMIND_VERSION,
        "conversion_version": FROZEN_CONVERSION_VERSION,
        "source": {"file": Path(gguf).name, "bytes": Path(gguf).stat().st_size, "sha256": file_sha256(gguf),
                   "architecture": md.get("general.architecture"), "name": md.get("general.name"),
                   "file_type": md.get("general.file_type"), "hparams": asdict(w.hp),
                   "vocab_size": len(md.get("tokenizer.ggml.tokens", []))},
        "conversion": {"weight_bits": weight_bits, "weight_group": group, "weight_rounding": weight_rounding,
                       "ops": {k: asdict(conv.get(k, OpConv())) for k in OP_CLASSES},
                       "learned_parameters": 0, "calibration_data": None},
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                        "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()}
                    for k, v in sorted(tensors.items())},
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        z.writestr(zipfile.ZipInfo("manifest.json", _ZIP_TIME), json.dumps(manifest, indent=1, sort_keys=True))
        for k in sorted(tensors):
            z.writestr(zipfile.ZipInfo(f"tensors/{k}.npy", _ZIP_TIME), _npy(np.ascontiguousarray(tensors[k])))
    manifest["hmmodel_sha256"] = file_sha256(out)
    return manifest


def load_frozen(path: str | Path) -> tuple[FrozenWeights, dict[str, OpConv], dict[str, Any]]:
    with zipfile.ZipFile(path) as z:
        m = json.loads(z.read("manifest.json"))
        if m.get("format") != "hmfrozen":
            raise ValueError("not a frozen-conversion .hmmodel (format != hmfrozen)")
        t = {n[len("tensors/"):-4]: np.load(io.BytesIO(z.read(n))) for n in z.namelist() if n.startswith("tensors/")}
    hp = LlamaHParams(**m["source"]["hparams"])

    def mat(name: str) -> np.ndarray:
        if name in t:
            return t[name].astype(np.float32)
        return dequantize_weight(t[name + ".codes"], t[name + ".scales"])

    layers = [{"attn_norm": t[f"blk.{L}.attn_norm"], "ffn_norm": t[f"blk.{L}.ffn_norm"],
               **{k: mat(f"blk.{L}.{k}") for k in ("wqkv", "wo", "w_in", "w_down")}} for L in range(hp.n_layer)]
    w = FrozenWeights(hp, t["token_embd"], mat("output"), t["output_norm"], layers, m["source"])
    conv = {k: OpConv(**v) for k, v in m["conversion"]["ops"].items()}
    return w, conv, m
