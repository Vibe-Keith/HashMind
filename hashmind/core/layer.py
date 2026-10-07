"""HashMindLayer: a bank of HashMindNodes producing a dense numeric tensor."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..backends.base import ASICBackend
from ..backends.simulated import SimulatedS9Backend
from .node import ChallengeConfig, FeatureMode, HashMindNode, InputMapping


@dataclass
class LayerStats:
    calls: int = 0
    samples: int = 0
    sha256d_logical: int = 0  # what a cache-less device would compute
    sha256d_executed: int = 0  # actually computed (after per-call dedup of identical payloads)
    asic_jobs: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HashMindLayerSpec:
    """Everything needed to rebuild a layer bit-exactly (stored in .hmmodel)."""

    input_dim: int
    output_dim: int
    seed: int
    feature_mode: str
    tuple_size: int
    levels: int
    nonces_per_node: int
    challenge: dict[str, Any] = field(default_factory=dict)
    context_dim: int = 0
    context_per_node: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HashMindLayer:
    """Maps (N, input_dim) float -> (N, output_dim) float32 SHA-256 features.

    Parameters
    ----------
    input_dim, output_dim
        Input width and number of output features (truncated to exactly output_dim).
    seed
        Determines node wiring (which inputs each node reads) and challenges.
    feature_mode
        One of ``hash_bits``, ``hash_bytes``, ``hamming``, ``bucket``, ``threshold``.
    tuple_size
        Input dimensions per node. Small (3-8) keeps locality; large makes the
        layer a pure memorizer.
    levels
        Quantization levels per input dimension (2 = sign bit).
    nonces_per_node
        SHA-256d evaluations per node per input.
    backend
        ASIC backend for ASIC-native modes. Defaults to :class:`SimulatedS9Backend`.
        Digest-based modes always use the CPU (they are simulator-only).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed: int = 0,
        feature_mode: str | FeatureMode = "hash_bits",
        tuple_size: int = 2,
        levels: int = 4,
        nonces_per_node: int = 16,
        backend: ASICBackend | None = None,
        thresholds: np.ndarray | None = None,
        context_dim: int = 0,
        context_per_node: int = 0,
        **challenge_kwargs: Any,
    ) -> None:
        lo = 0 if context_per_node > 0 else 1
        if not lo <= tuple_size <= min(input_dim, 16 if context_per_node else 32):
            raise ValueError("tuple_size out of range for this input/context configuration")
        if not 0 <= context_per_node <= min(context_dim, 4):
            raise ValueError("context_per_node must be in [0, min(context_dim, 4)]")
        if not 2 <= levels <= 256:
            raise ValueError("levels must be in [2, 256]")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.tuple_size = tuple_size
        self.levels = levels
        self.challenge = ChallengeConfig(mode=FeatureMode(feature_mode), nonces=nonces_per_node,
                                         **challenge_kwargs)
        # With the default simulator, ASIC-native features are computed by the
        # node's CPU digest path, which is bit-identical (tested) and much faster
        # than building HashJob objects. An explicitly passed backend is always used.
        self._explicit_backend = backend is not None
        self.backend = backend if backend is not None else SimulatedS9Backend()
        self.stats = LayerStats()
        fpn = self.challenge.features_per_node
        self.n_nodes = -(-output_dim // fpn)
        rng = np.random.default_rng(seed)
        self._indices = [tuple(int(i) for i in rng.choice(input_dim, tuple_size, replace=False))
                         for _ in range(self.n_nodes)]
        self._node_seeds = rng.integers(0, 2**32, self.n_nodes, dtype=np.uint64)
        # Separate stream so context-free layers stay bit-identical to phase 2.
        self.context_dim, self.context_per_node = context_dim, context_per_node
        crng = np.random.default_rng([seed, 0xC0E7])
        self._ctx = [tuple(int(i) for i in sorted(crng.choice(context_dim, context_per_node, replace=False)))
                     if context_per_node else () for _ in range(self.n_nodes)]
        self.thresholds: np.ndarray | None = None
        self.nodes: list[HashMindNode] = []
        if thresholds is not None:
            self.set_thresholds(thresholds)

    # -- configuration --------------------------------------------------------

    @property
    def asic_native(self) -> bool:
        return self.challenge.asic_native

    @property
    def spec(self) -> HashMindLayerSpec:
        c = asdict(self.challenge)
        c["mode"] = self.challenge.mode.value
        return HashMindLayerSpec(self.input_dim, self.output_dim, self.seed, self.challenge.mode.value,
                                 self.tuple_size, self.levels, self.challenge.nonces, c,
                                 self.context_dim, self.context_per_node)

    @classmethod
    def from_spec(cls, spec: HashMindLayerSpec | dict[str, Any], thresholds: np.ndarray,
                  backend: ASICBackend | None = None) -> "HashMindLayer":
        s = spec if isinstance(spec, dict) else spec.to_dict()
        ch = {k: v for k, v in s["challenge"].items() if k not in ("mode", "nonces")}
        return cls(s["input_dim"], s["output_dim"], s["seed"], s["feature_mode"], s["tuple_size"],
                   s["levels"], s["nonces_per_node"], backend, thresholds,
                   s.get("context_dim", 0), s.get("context_per_node", 0), **ch)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        t = np.asarray(thresholds, np.float32)
        if t.shape != (self.input_dim, self.levels - 1):
            raise ValueError(f"thresholds must be ({self.input_dim}, {self.levels - 1})")
        self.thresholds = t
        self.nodes = [
            HashMindNode(int(s), InputMapping(idx, t[list(idx)], ctx), self.challenge)
            for idx, s, ctx in zip(self._indices, self._node_seeds, self._ctx)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, quantizer: str = "quantile") -> "HashMindLayer":
        """Fit per-dimension quantization thresholds. Deterministic; no gradients.

        quantizer="quantile"   equal-mass bins of X (unsupervised; phase-2 default)
        quantizer="supervised" greedy cut points per dimension that maximize the
                               reduction of squared error of ``y`` (N,) or (N, m)
                               (a 1-D regression-tree split search on each input)
        """
        X = np.asarray(X, np.float64)
        if quantizer == "quantile":
            q = np.arange(1, self.levels) / self.levels
            self.set_thresholds(np.quantile(X, q, axis=0).T)
        elif quantizer == "supervised":
            if y is None:
                raise ValueError("supervised quantizer needs y")
            self.set_thresholds(supervised_thresholds(X, y, self.levels))
        else:
            raise ValueError(f"unknown quantizer {quantizer!r}")
        return self

    def table_log2(self, context_vocab: int = 0) -> float:
        """log2 of SHA-256d evaluations needed to precompute this layer as a lookup table."""
        per_node = self.tuple_size * np.log2(self.levels)
        if self.context_per_node:
            per_node += self.context_per_node * np.log2(max(context_vocab, 2))
        return float(per_node + np.log2(self.n_nodes * self.challenge.nonces))

    # -- forward --------------------------------------------------------------

    def transform(self, X: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        if not self.nodes:
            raise RuntimeError("layer has no thresholds: call fit(X) or pass thresholds")
        X = np.asarray(X, np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"expected (N, {self.input_dim}) input, got {X.shape}")
        t0 = time.perf_counter()
        N = X.shape[0]
        fpn = self.challenge.features_per_node
        out = np.empty((N, self.n_nodes * fpn), np.float32)
        nonces = self.challenge.nonces
        job_id = self.stats.asic_jobs
        if self.context_per_node:
            if context is None or context.shape != (N, self.context_dim):
                raise ValueError(f"layer needs context of shape (N, {self.context_dim})")
            context = np.asarray(context, np.uint32)
        nt = self.tuple_size
        for j, node in enumerate(self.nodes):
            codes = node.input_mapping.encode(X).astype(np.uint32)
            ctx_cols = node.input_mapping.context_indices
            key = np.concatenate([codes, context[:, list(ctx_cols)]], 1) if ctx_cols else codes
            uniq, inverse = np.unique(key, axis=0, return_inverse=True)

            def pl(u: np.ndarray) -> bytes:
                return node.payload(u[:nt], u[nt:] if ctx_cols else None)

            if self.asic_native and self._explicit_backend:
                jobs = [node.job(job_id + k, pl(u)) for k, u in enumerate(uniq)]
                job_id += len(jobs)
                feats = np.stack([node.features_from_nonces(r.nonces)
                                  for r in self.backend.run_jobs(jobs)])
            else:
                feats = np.stack([node.features_from_digests(node.header_prefix(pl(u)))
                                  for u in uniq])
            out[:, j * fpn:(j + 1) * fpn] = feats[inverse.reshape(-1)]
            self.stats.sha256d_executed += len(uniq) * nonces
        self.stats.asic_jobs = job_id
        self.stats.calls += 1
        self.stats.samples += N
        self.stats.sha256d_logical += N * self.n_nodes * nonces
        self.stats.seconds += time.perf_counter() - t0
        return out[:, : self.output_dim]

    __call__ = transform

    def fit_transform(self, X: np.ndarray, context: np.ndarray | None = None, y: np.ndarray | None = None,
                      quantizer: str = "quantile") -> np.ndarray:
        return self.fit(X, y, quantizer).transform(X, context)


def supervised_thresholds(X: np.ndarray, y: np.ndarray, levels: int, n_candidates: int = 63) -> np.ndarray:
    """Per-dimension greedy split search: choose ``levels - 1`` cut points that most
    reduce the squared error of y when y is predicted by its bin mean."""
    Y = np.asarray(y, np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    Y = Y - Y.mean(0)
    cands = np.quantile(X, np.linspace(0, 1, n_candidates + 2)[1:-1], axis=0)  # (C, d)
    out = np.empty((X.shape[1], levels - 1))
    for i in range(X.shape[1]):
        x = X[:, i]
        order = np.argsort(x)
        xs, cs = x[order], np.cumsum(Y[order], 0)

        def sse_gain(cuts: list[float]) -> float:
            pos = np.searchsorted(xs, np.sort(cuts), side="right")
            edges = np.concatenate([[0], pos, [len(x)]])
            g = 0.0
            for a, b in zip(edges[:-1], edges[1:]):
                if b > a:
                    s = (cs[b - 1] - (cs[a - 1] if a else 0))
                    g += float((s * s).sum()) / (b - a)
            return g

        chosen: list[float] = []
        for _ in range(levels - 1):
            best, best_g = None, -1.0
            for c in np.unique(cands[:, i]):
                if c in chosen:
                    continue
                g = sse_gain(chosen + [c])
                if g > best_g:
                    best, best_g = c, g
            chosen.append(best if best is not None else (chosen[-1] if chosen else 0.0))
        out[i] = np.sort(chosen)
    return out
