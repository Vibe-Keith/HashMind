"""Multi-resolution hashed feature layer (Phase 4, Track B).

Instead of one big cryptographic lookup cell per node, the layer is a family of
*groups*; each group has its own projection, tuple size, quantizer and node
count:

    input -> projection A -> small tuple  -> primitive -> output block
          -> projection B -> small tuple  -> primitive -> output block
          -> projection C -> larger tuple -> primitive -> output block
          -> concatenate (dense) or merge events (sparse) -> linear readout

Each node is still a random function of a quantized grid cell. Coarse groups
(small tuples) repeat cells often and give the readout something to
generalize from; finer groups add resolution where data supports it. This is
the CMAC / tile-coding idea, with the tiles keyed through an interchangeable
hash primitive.

The primitive (Track A), the wiring (fixed random here; learned in
:mod:`.routing`) and the output mode (Track D) are independent arguments.

Wiring modes
------------
random   each node draws ``tuple_size`` distinct input dims
overlap  nodes take sliding windows of a random permutation with stride
         ``tuple_size // 2`` (neighbouring nodes share half their dims)
legacy   bit-exact phase-2/3 :class:`HashMindLayer` wiring (single group,
         no context); used to reproduce the current baseline exactly
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .primitives import FeaturePrimitive, _splitmix, get_primitive, pack_payloads, words_for_nodes
from .sparse_features import OutputConfig, SparseEvents, node_output


@dataclass(frozen=True)
class GroupSpec:
    name: str
    n_nodes: int
    tuple_size: int  # continuous dims per node (0 = context only)
    levels: int = 4
    evals: int = 16  # primitive evaluations per node per input
    projection: str = "axis"  # axis | rotation
    wiring: str = "random"  # random | overlap | legacy
    ctx_cols: tuple[int, ...] = ()  # discrete context columns hashed exactly by every node in the group

    def __post_init__(self) -> None:
        if self.projection not in ("axis", "rotation"):
            raise ValueError("projection must be axis or rotation")
        if self.wiring not in ("random", "overlap", "legacy"):
            raise ValueError("wiring must be random, overlap or legacy")
        if self.tuple_size == 0 and not self.ctx_cols:
            raise ValueError("a group needs continuous dims or context columns")
        if self.tuple_size > (16 if self.ctx_cols else 32) or len(self.ctx_cols) > 4:
            raise ValueError("payload overflow: <=32 dims, or <=16 dims + <=4 context symbols")
        if not 2 <= self.levels <= 256:
            raise ValueError("levels in [2, 256]")


@dataclass(frozen=True)
class NodeSpec:
    group: int
    dims: tuple[int, ...]
    seed: int


@dataclass
class LayerCounters:
    calls: int = 0
    samples: int = 0
    evals_logical: int = 0  # what a cache-less device computes
    evals_executed: int = 0  # after per-call dedup of identical cells
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeOutputs:
    """Per-node outputs of one transform call."""

    blocks: list[np.ndarray]  # dense (N, f) float32 or ids (N, k) int32
    cells: list[np.ndarray]  # (N,) uint64 cell keys (payload fingerprint), for diagnostics
    sparse: bool
    widths: list[int]  # features per node

    def assemble(self) -> np.ndarray | SparseEvents:
        if not self.sparse:
            return np.hstack(self.blocks).astype(np.float32)
        return SparseEvents.hstack([SparseEvents(b, w) for b, w in zip(self.blocks, self.widths)])


def cell_fingerprint(payloads: np.ndarray) -> np.ndarray:
    w = np.ascontiguousarray(payloads).view("<u4").astype(np.uint64)
    h = np.zeros(len(payloads), np.uint64)
    for i in range(w.shape[1]):
        h = _splitmix(h ^ w[:, i])
    return h


class MultiResolutionLayer:
    def __init__(self, input_dim: int, groups: list[GroupSpec], primitive: str | FeaturePrimitive = "sha256d",
                 output: OutputConfig | None = None, seed: int = 0, context_dim: int = 0) -> None:
        self.input_dim, self.groups, self.seed = input_dim, list(groups), seed
        self.context_dim = context_dim
        self.primitive = get_primitive(primitive)
        self.output = output or OutputConfig()
        self.counters = LayerCounters()
        self.projections: list[np.ndarray | None] = []
        self.thresholds: list[np.ndarray | None] = [None] * len(self.groups)
        self.nodes: list[NodeSpec] = []
        for gi, g in enumerate(self.groups):
            if g.tuple_size > input_dim:
                raise ValueError(f"group {g.name}: tuple_size > input_dim")
            if any(c >= context_dim for c in g.ctx_cols):
                raise ValueError(f"group {g.name}: context column out of range")
            if g.projection == "rotation":
                q, r = np.linalg.qr(np.random.default_rng([seed, gi, 0x524F]).standard_normal((input_dim, input_dim)))
                self.projections.append((q * np.sign(np.diag(r))).astype(np.float32))
            else:
                self.projections.append(None)
            self.nodes += self._wire(gi, g)

    # -- wiring ---------------------------------------------------------------

    def _wire(self, gi: int, g: GroupSpec) -> list[NodeSpec]:
        t, d = g.tuple_size, self.input_dim
        if g.wiring == "legacy":
            if len(self.groups) != 1 or g.ctx_cols:
                raise ValueError("legacy wiring reproduces a single context-free HashMindLayer only")
            rng = np.random.default_rng(self.seed)
            dims = [tuple(int(i) for i in rng.choice(d, t, replace=False)) for _ in range(g.n_nodes)]
            seeds = rng.integers(0, 2**32, g.n_nodes, dtype=np.uint64)
            return [NodeSpec(gi, dm, int(s)) for dm, s in zip(dims, seeds)]
        rng = np.random.default_rng([self.seed, gi, 0x4D52])
        if g.wiring == "overlap" and t:
            perm, stride = rng.permutation(d), max(1, t // 2)
            dims = [tuple(int(perm[(j * stride + k) % d]) for k in range(t)) for j in range(g.n_nodes)]
        else:
            dims = [tuple(int(i) for i in rng.choice(d, t, replace=False)) if t else () for _ in range(g.n_nodes)]
        seeds = rng.integers(0, 2**32, g.n_nodes, dtype=np.uint64)
        return [NodeSpec(gi, dm, int(s)) for dm, s in zip(dims, seeds)]

    def fresh_seed(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 2**32, dtype=np.uint64))

    # -- fit / encode ------------------------------------------------------------

    def project(self, X: np.ndarray, gi: int) -> np.ndarray:
        P = self.projections[gi]
        return X if P is None else X @ P

    def fit(self, X: np.ndarray) -> "MultiResolutionLayer":
        X = np.asarray(X, np.float32)
        for gi, g in enumerate(self.groups):
            if g.tuple_size:
                q = np.arange(1, g.levels) / g.levels
                self.thresholds[gi] = np.quantile(self.project(X, gi).astype(np.float64), q, axis=0).T.astype(np.float32)
        return self

    def payloads(self, Xg: dict[int, np.ndarray], C: np.ndarray | None, node: NodeSpec) -> np.ndarray:
        g = self.groups[node.group]
        if g.tuple_size:
            sub = Xg[node.group][:, list(node.dims)]
            thr = self.thresholds[node.group][list(node.dims)]
            codes = (sub[:, :, None] > thr[None]).sum(2).astype(np.uint8)
        else:
            codes = np.zeros((len(C), 0), np.uint8)
        ctx = None
        if g.ctx_cols:
            ctx = np.asarray(C[:, list(g.ctx_cols)], np.uint32)
        return pack_payloads(codes, ctx)

    def node_width(self, node: NodeSpec) -> int:
        return self.output.features_per_node(self.groups[node.group].evals)

    @property
    def n_features(self) -> int:
        return sum(self.node_width(n) for n in self.nodes)

    @property
    def evals_per_example(self) -> int:
        return sum(self.groups[n.group].evals for n in self.nodes)

    # -- forward ---------------------------------------------------------------------

    def transform_nodes(self, X: np.ndarray, C: np.ndarray | None = None,
                        nodes: list[NodeSpec] | None = None) -> NodeOutputs:
        if any(t is None for g, t in zip(self.groups, self.thresholds) if g.tuple_size):
            raise RuntimeError("call fit(X) first")
        X = np.asarray(X, np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"expected (N, {self.input_dim}) input")
        nodes = self.nodes if nodes is None else nodes
        if self.context_dim and (C is None or C.shape != (len(X), self.context_dim)):
            raise ValueError(f"layer needs context of shape (N, {self.context_dim})")
        t0 = time.perf_counter()
        Xg = {gi: self.project(X, gi) for gi in {n.group for n in nodes} if self.groups[gi].tuple_size}
        jobs, inverses, cells = [], [], []
        for node in nodes:
            p = self.payloads(Xg, C, node)
            uniq, inv = np.unique(p, axis=0, return_inverse=True)
            jobs.append((node.seed, uniq, self.groups[node.group].evals))
            inverses.append(inv.reshape(-1))
            cells.append(cell_fingerprint(uniq)[inv.reshape(-1)])
        words = words_for_nodes(self.primitive, jobs)
        blocks = [node_output(w, self.output)[inv] for w, inv in zip(words, inverses)]
        N = len(X)
        self.counters.calls += 1
        self.counters.samples += N
        self.counters.evals_logical += N * sum(e for _, _, e in jobs)
        self.counters.evals_executed += sum(len(u) * e for _, u, e in jobs)
        self.counters.seconds += time.perf_counter() - t0
        return NodeOutputs(blocks, cells, self.output.sparse, [self.node_width(n) for n in nodes])

    def transform(self, X: np.ndarray, C: np.ndarray | None = None) -> np.ndarray | SparseEvents:
        return self.transform_nodes(X, C).assemble()

    __call__ = transform

    def table_log2(self, context_vocab: int = 32000) -> float:
        """log2 of evaluations needed to precompute every node as a lookup table."""
        tot = 0.0
        for n in self.nodes:
            g = self.groups[n.group]
            bits = g.tuple_size * np.log2(g.levels) + len(g.ctx_cols) * np.log2(max(context_vocab, 2))
            tot += 2.0 ** bits * g.evals
        return float(np.log2(tot))

    def describe(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive.describe(),
            "output": asdict(self.output),
            "seed": self.seed,
            "groups": [asdict(g) for g in self.groups],
            "n_nodes": len(self.nodes),
            "n_features": self.n_features,
            "evals_per_example": self.evals_per_example,
        }


def single_resolution(input_dim: int, n_nodes: int = 128, tuple_size: int = 2, levels: int = 4,
                      evals: int = 16, primitive: str | FeaturePrimitive = "sha256d", seed: int = 0,
                      output: OutputConfig | None = None, wiring: str = "legacy") -> MultiResolutionLayer:
    """The phase-2/3 HashMind layer as a one-group MultiResolutionLayer.

    With ``wiring='legacy'``, ``primitive='sha256d'`` and ``bits`` output this is
    bit-identical to ``HashMindLayer(input_dim, n_nodes * evals, seed, 'hash_bits',
    tuple_size, levels, evals)``.
    """
    g = GroupSpec(f"t{tuple_size}l{levels}", n_nodes, tuple_size, levels, evals, wiring=wiring)
    return MultiResolutionLayer(input_dim, [g], primitive, output, seed)


def standard_multires_groups(budget_nodes: int = 128, evals: int = 16) -> list[GroupSpec]:
    """Phase-4 standard multi-resolution family for continuous inputs (fixed a priori).

    Node shares (of ``budget_nodes``): t=1 (3/16), t=2 axis (1/4), t=2 rotated (3/16),
    t=4 overlapping (1/4), t=8 levels=2 (1/8).
    """
    b = budget_nodes
    n = [3 * b // 16, b // 4, 3 * b // 16, b // 4]
    n.append(b - sum(n))
    return [
        GroupSpec("t1_l8", n[0], 1, 8, evals),
        GroupSpec("t2_l4_axis", n[1], 2, 4, evals),
        GroupSpec("t2_l4_rot", n[2], 2, 4, evals, projection="rotation"),
        GroupSpec("t4_l4_overlap", n[3], 4, 4, evals, wiring="overlap"),
        GroupSpec("t8_l2", n[4], 8, 2, evals),
    ]


def context_multires_groups(budget_nodes: int = 128, evals: int = 16) -> list[GroupSpec]:
    """Phase-4 standard family for the context task: continuous h_0 dims plus exact
    token-id context columns [tok_t, tok_t-1, tok_t-2] at 1-, 2- and 3-gram order."""
    b = budget_nodes
    n = [b // 4, b // 8, b // 8, b // 4, b // 8]
    n.append(b - sum(n))
    return [
        GroupSpec("cont_t2_l4", n[0], 2, 4, evals),
        GroupSpec("ctx_tok_t", n[1], 0, 4, evals, ctx_cols=(0,)),
        GroupSpec("ctx_tok_t-1", n[2], 0, 4, evals, ctx_cols=(1,)),
        GroupSpec("ctx_bigram", n[3], 0, 4, evals, ctx_cols=(0, 1)),
        GroupSpec("ctx_trigram", n[4], 0, 4, evals, ctx_cols=(0, 1, 2)),
        GroupSpec("ctx_tok_t-1+cont_t1", n[5], 1, 4, evals, ctx_cols=(1,)),
    ]
