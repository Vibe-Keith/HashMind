""".hmmodel container: a zip with ``manifest.json`` and ``tensors.npz``.

Phase-1 ``.hcmodel`` files (format "hcmodel", version 1) load unchanged.

manifest.json
    format_version, hashmind_version, conversion_version
    source:      original GGUF metadata summary (architecture, dims, tokenizer...)
    config:      HashMindConfig
    seeds:       seed + challenge salt (wiring is reproducible from these)
    conversion:  per-tensor dispositions, totals, warnings, projection provenance,
                 weights: per-tensor PRESERVED/TRANSFORMED/DISCARDED records (v2)
    hashmind_layer: HashMindLayerSpec of the feature layer (v2, optional)
    tensors:     {name: {group, shape, dtype}}
tensors.npz
    preserved/*, transformed/*, lowrank/*, readout/*, wiring/*, layer/*
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import CONVERSION_VERSION, HASHMIND_VERSION, HMMODEL_FORMAT_VERSION
from ..architecture.config import HashMindConfig
from ..architecture.model import HashMindParams
from ..features.hash_layer import TupleWiring


class HMModelError(Exception):
    pass


@dataclass
class HMModel:
    config: HashMindConfig
    params: HashMindParams
    wiring: TupleWiring
    source: dict[str, Any] = field(default_factory=dict)
    conversion: dict[str, Any] = field(default_factory=dict)
    extra_tensors: dict[str, np.ndarray] = field(default_factory=dict)  # weight plan, layer thresholds
    hashmind_layer: dict[str, Any] | None = None

    def tensor_dict(self) -> dict[str, np.ndarray]:
        p, w = self.params, self.wiring
        d: dict[str, np.ndarray | None] = {
            "preserved/token_embd": p.embedding,
            "preserved/output": p.lm_head,
            "preserved/output_norm": p.output_norm,
            "transformed/projection": p.projection,
            "transformed/projection_center": p.projection_center,
            "readout/weight": p.readout,
            "readout/bias": p.readout_bias,
            "wiring/tuple_index": w.tuple_index,
            "wiring/challenges": w.challenges,
            "wiring/reservoir_source": w.reservoir_source,
            "wiring/reservoir_keep_mask": w.reservoir_keep_mask,
        }
        out = dict(self.extra_tensors)
        out.update({k: v for k, v in d.items() if v is not None})
        return out

    def save(self, path: str | Path, embed_dtype: str = "float16") -> None:
        tensors = self.tensor_dict()
        stored = {
            k: (v.astype(embed_dtype) if k.startswith("preserved/") and v.ndim == 2 else v)
            for k, v in tensors.items()
        }
        manifest = {
            "format": "hmmodel",
            "format_version": HMMODEL_FORMAT_VERSION,
            "hashmind_version": HASHMIND_VERSION,
            "conversion_version": CONVERSION_VERSION,
            "source": self.source,
            "config": self.config.to_dict(),
            "seeds": {"seed": self.config.seed, "challenge_salt": self.config.challenge_salt},
            "conversion": self.conversion,
            "hashmind_layer": self.hashmind_layer,
            "tensors": {
                k: {"group": k.split("/")[0], "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in stored.items()
            },
            "disclaimer": "Experimental. NOT equivalent to the source model. Embeddings and LM "
            "head are preserved; transformer-block matrices are stored only as low-rank factors "
            "and are not executed; computation is replaced by a SHA-256 feature layer whose "
            "readout must be trained.",
        }
        buf = io.BytesIO()
        np.savez(buf, **stored)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2, default=_json_default),
                       compress_type=zipfile.ZIP_DEFLATED)
            z.writestr("tensors.npz", buf.getvalue(), compress_type=zipfile.ZIP_STORED)

    @classmethod
    def load(cls, path: str | Path) -> "HMModel":
        with zipfile.ZipFile(path) as z:
            m = json.loads(z.read("manifest.json"))
            if m.get("format") not in ("hmmodel", "hcmodel"):
                raise HMModelError("not an .hmmodel/.hcmodel file")
            if m["format_version"] > HMMODEL_FORMAT_VERSION:
                raise HMModelError(f"format_version {m['format_version']} is newer than supported")
            npz = np.load(io.BytesIO(z.read("tensors.npz")))
            t = {k: npz[k] for k in npz.files}

        def f32(k: str) -> np.ndarray | None:
            return t[k].astype(np.float32) if k in t else None

        params = HashMindParams(
            embedding=f32("preserved/token_embd"),  # type: ignore[arg-type]
            lm_head=f32("preserved/output"),
            output_norm=f32("preserved/output_norm"),
            projection=t["transformed/projection"],
            projection_center=t["transformed/projection_center"],
            readout=t["readout/weight"],
            readout_bias=t["readout/bias"],
        )
        wiring = TupleWiring(
            t["wiring/tuple_index"],
            t["wiring/challenges"],
            t["wiring/reservoir_source"],
            t["wiring/reservoir_keep_mask"],
        )
        extra = {k: v for k, v in t.items() if k not in _CORE_KEYS}
        return cls(HashMindConfig.from_dict(m["config"]), params, wiring, m["source"], m["conversion"],
                   extra, m.get("hashmind_layer"))


_CORE_KEYS = frozenset({
    "preserved/token_embd", "preserved/output", "preserved/output_norm",
    "transformed/projection", "transformed/projection_center", "readout/weight", "readout/bias",
    "wiring/tuple_index", "wiring/challenges", "wiring/reservoir_source", "wiring/reservoir_keep_mask",
})


def read_manifest(path: str | Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("manifest.json"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
