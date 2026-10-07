"""HashCortex v1 model: host math around an ASIC feature layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backends.base import ASICBackend
from ..features.hash_layer import HashFeatureLayer, TupleWiring
from .config import HashCortexConfig


@dataclass
class HashCortexParams:
    embedding: np.ndarray  # (vocab, d)            PRESERVED
    lm_head: np.ndarray | None  # (vocab, d) or None = tied  PRESERVED
    output_norm: np.ndarray | None  # (d,)         HOST-ONLY (preserved)
    projection: np.ndarray  # (d, k)               TRANSFORMED
    projection_center: np.ndarray  # (k,)          TRANSFORMED
    readout: np.ndarray  # (n_features, d)         NEW (trained on host)
    readout_bias: np.ndarray  # (d,)               NEW


def rmsnorm(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * w if w is not None else y


class HashCortexModel:
    def __init__(
        self,
        cfg: HashCortexConfig,
        params: HashCortexParams,
        wiring: TupleWiring,
        backend: ASICBackend,
    ) -> None:
        self.cfg = cfg
        self.p = params
        self.layer = HashFeatureLayer(cfg, wiring, backend)

    def input_bits(self, token: int, reservoir: np.ndarray) -> np.ndarray:
        e = rmsnorm(self.p.embedding[token], None, self.cfg.norm_eps)
        z = e @ self.p.projection - self.p.projection_center
        return np.concatenate([(z > 0).astype(np.uint8), reservoir])

    def run_features(self, tokens: list[int]) -> np.ndarray:
        """(T, n_features) binary features for a token sequence."""
        r = np.zeros(self.cfg.reservoir_bits, np.uint8)
        out = np.zeros((len(tokens), self.cfg.n_features), np.uint8)
        for t, tok in enumerate(tokens):
            f = self.layer.features(self.input_bits(tok, r))
            out[t] = f
            r = self.layer.update_reservoir(r, f)
        return out

    def hidden(self, tokens: list[int], feats: np.ndarray | None = None) -> np.ndarray:
        f = self.run_features(tokens) if feats is None else feats
        e = self.p.embedding[np.asarray(tokens)]
        return e + (2.0 * f.astype(np.float32) - 1.0) @ self.p.readout + self.p.readout_bias

    def logits(self, tokens: list[int], feats: np.ndarray | None = None) -> np.ndarray:
        h = rmsnorm(self.hidden(tokens, feats), self.p.output_norm, self.cfg.norm_eps)
        head = self.p.lm_head if self.p.lm_head is not None else self.p.embedding
        return h @ head.T

    def fit_readout(self, feats: np.ndarray, tokens: list[int], targets: np.ndarray) -> float:
        """Ridge-fit W_r so that e + W_r^T(2f-1) + b ~= targets (T, d). Returns train MSE.

        ``targets`` would typically be hidden states of the original model
        (distillation); obtaining those is out of scope for phase 1.
        """
        X = 2.0 * feats.astype(np.float64) - 1.0
        Y = targets.astype(np.float64) - self.p.embedding[np.asarray(tokens)]
        xm, ym = X.mean(0), Y.mean(0)
        Xc, Yc = X - xm, Y - ym
        A = Xc.T @ Xc + self.cfg.ridge_lambda * np.eye(X.shape[1])
        W = np.linalg.solve(A, Xc.T @ Yc)
        self.p.readout = W.astype(np.float32)
        self.p.readout_bias = (ym - xm @ W).astype(np.float32)
        pred = self.hidden(tokens, feats)
        return float(np.mean((pred - targets) ** 2))
