""".hcmodel container: a zip with ``manifest.json`` and ``tensors.npz``.

manifest.json
    format_version, hashcortex_version, conversion_version
    source:      original GGUF metadata summary (architecture, dims, tokenizer...)
    config:      HashCortexConfig
    seeds:       seed + challenge salt (wiring is reproducible from these)
    conversion:  per-tensor dispositions, totals, warnings, projection provenance
    tensors:     {name: {group, shape, dtype}}
tensors.npz
    preserved/*, transformed/*, readout/*, wiring/*
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import CONVERSION_VERSION, HASHCORTEX_VERSION, HCMODEL_FORMAT_VERSION
from ..architecture.config import HashCortexConfig
from ..architecture.model import HashCortexParams
from ..features.hash_layer import TupleWiring


class HCModelError(Exception):
    pass


@dataclass
class HCModel:
    config: HashCortexConfig
    params: HashCortexParams
    wiring: TupleWiring
    source: dict[str, Any] = field(default_factory=dict)
    conversion: dict[str, Any] = field(default_factory=dict)

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
        return {k: v for k, v in d.items() if v is not None}

    def save(self, path: str | Path, embed_dtype: str = "float16") -> None:
        tensors = self.tensor_dict()
        stored = {
            k: (v.astype(embed_dtype) if k.startswith("preserved/") and v.ndim == 2 else v)
            for k, v in tensors.items()
        }
        manifest = {
            "format": "hcmodel",
            "format_version": HCMODEL_FORMAT_VERSION,
            "hashcortex_version": HASHCORTEX_VERSION,
            "conversion_version": CONVERSION_VERSION,
            "source": self.source,
            "config": self.config.to_dict(),
            "seeds": {"seed": self.config.seed, "challenge_salt": self.config.challenge_salt},
            "conversion": self.conversion,
            "tensors": {
                k: {"group": k.split("/")[0], "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in stored.items()
            },
            "disclaimer": "Not equivalent to the source model. Embeddings and LM head are "
            "preserved; all transformer blocks are replaced by a SHA-256 feature layer "
            "whose readout must be trained.",
        }
        buf = io.BytesIO()
        np.savez(buf, **stored)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2, default=_json_default))
            z.writestr("tensors.npz", buf.getvalue())

    @classmethod
    def load(cls, path: str | Path) -> "HCModel":
        with zipfile.ZipFile(path) as z:
            m = json.loads(z.read("manifest.json"))
            if m.get("format") != "hcmodel":
                raise HCModelError("not an hcmodel file")
            if m["format_version"] > HCMODEL_FORMAT_VERSION:
                raise HCModelError(f"format_version {m['format_version']} is newer than supported")
            npz = np.load(io.BytesIO(z.read("tensors.npz")))
            t = {k: npz[k] for k in npz.files}

        def f32(k: str) -> np.ndarray | None:
            return t[k].astype(np.float32) if k in t else None

        params = HashCortexParams(
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
        return cls(HashCortexConfig.from_dict(m["config"]), params, wiring, m["source"], m["conversion"])


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
