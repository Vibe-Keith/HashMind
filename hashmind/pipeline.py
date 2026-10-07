"""GGUF representation -> HashMind features -> readout, as one object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends.base import ASICBackend
from .core.layer import HashMindLayer
from .core.readout import RidgeReadout
from .formats.hmmodel import HMModel


@dataclass
class HashMindPipeline:
    """x (original d-dim vector) -> (x - mean) B -> HashMindLayer -> [readout].

    ``basis``/``mean`` come from the weight plan (PCA of token embeddings).
    """

    basis: np.ndarray  # (d, r)
    mean: np.ndarray  # (d,)
    layer: HashMindLayer
    readout: RidgeReadout | None = None

    @classmethod
    def from_hmmodel(cls, m: HMModel, backend: ASICBackend | None = None) -> "HashMindPipeline":
        if m.hashmind_layer is None:
            raise ValueError("model has no HashMind layer (phase-1 .hcmodel?); re-convert it")
        layer = HashMindLayer.from_spec(m.hashmind_layer, m.extra_tensors["layer/thresholds"], backend)
        return cls(m.extra_tensors["transformed/embd_basis"], m.extra_tensors["transformed/embd_mean"], layer)

    def reduce(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, np.float32) - self.mean) @ self.basis).astype(np.float32)

    def features(self, X: np.ndarray) -> np.ndarray:
        return self.layer.transform(self.reduce(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("no readout attached")
        return self.readout.predict(self.features(X))
