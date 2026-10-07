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
        **challenge_kwargs: Any,
    ) -> None:
        if not 1 <= tuple_size <= min(input_dim, 32):
            raise ValueError("tuple_size must be in [1, min(input_dim, 32)]")
        if not 2 <= levels <= 256:
            raise ValueError("levels must be in [2, 256]")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        self.tuple_size = tuple_size
        self.levels = levels
        self.challenge = ChallengeConfig(mode=FeatureMode(feature_mode), nonces=nonces_per_node,
                                         **challenge_kwargs)
        self.backend = backend if backend is not None else SimulatedS9Backend()
        self.stats = LayerStats()
        fpn = self.challenge.features_per_node
        self.n_nodes = -(-output_dim // fpn)
        rng = np.random.default_rng(seed)
        self._indices = [tuple(int(i) for i in rng.choice(input_dim, tuple_size, replace=False))
                         for _ in range(self.n_nodes)]
        self._node_seeds = rng.integers(0, 2**32, self.n_nodes, dtype=np.uint64)
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
                                 self.tuple_size, self.levels, self.challenge.nonces, c)

    @classmethod
    def from_spec(cls, spec: HashMindLayerSpec | dict[str, Any], thresholds: np.ndarray,
                  backend: ASICBackend | None = None) -> "HashMindLayer":
        s = spec if isinstance(spec, dict) else spec.to_dict()
        ch = {k: v for k, v in s["challenge"].items() if k not in ("mode", "nonces")}
        return cls(s["input_dim"], s["output_dim"], s["seed"], s["feature_mode"], s["tuple_size"],
                   s["levels"], s["nonces_per_node"], backend, thresholds, **ch)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        t = np.asarray(thresholds, np.float32)
        if t.shape != (self.input_dim, self.levels - 1):
            raise ValueError(f"thresholds must be ({self.input_dim}, {self.levels - 1})")
        self.thresholds = t
        self.nodes = [
            HashMindNode(int(s), InputMapping(idx, t[list(idx)]), self.challenge)
            for idx, s in zip(self._indices, self._node_seeds)
        ]

    def fit(self, X: np.ndarray) -> "HashMindLayer":
        """Set per-dimension quantization thresholds to equal-mass quantiles of X.

        This is the only data-dependent step and it is deterministic; there is
        no gradient training inside the layer.
        """
        q = np.arange(1, self.levels) / self.levels
        self.set_thresholds(np.quantile(np.asarray(X, np.float64), q, axis=0).T)
        return self

    # -- forward --------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
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
        for j, node in enumerate(self.nodes):
            codes = node.input_mapping.encode(X)
            uniq, inverse = np.unique(codes, axis=0, return_inverse=True)
            if self.asic_native:
                jobs = [node.job(job_id + k, node.payload(u)) for k, u in enumerate(uniq)]
                job_id += len(jobs)
                feats = np.stack([node.features_from_nonces(r.nonces)
                                  for r in self.backend.run_jobs(jobs)])
            else:
                feats = np.stack([node.features_from_digests(node.header_prefix(node.payload(u)))
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

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
