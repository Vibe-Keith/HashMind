"""Learned sparse routing (Phase 4, Track C).

The hash primitive stays fixed. What is learned (on the CPU, from training data
only) is the *wiring*: which input dimensions each node reads. The procedure is
greedy, not differentiable:

1. start from the layer's random sparse wiring;
2. fit the ridge readout on the fit rows, score on held-out validation rows;
3. score each node by the energy of its contribution to the readout output,
   and each input dim by the (shrunk) mean score of the nodes reading it;
4. rewire the weakest ``replace_frac`` of each group's nodes: new dims are
   drawn in proportion to the dim scores, with a fresh node seed;
5. recompute only the rewired nodes, refit, rescore;
6. keep the round if the validation score improved, else revert;
7. repeat for a fixed number of rounds.

Constraints this respects: each node still reads exactly ``tuple_size`` dims
(sparse); the learned state is integer index tuples (a few bytes per node);
inference is the same deterministic hash computation as before. There is no
dense hidden layer: the only real-valued learned parameters are the linear
readout's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .multiresolution import MultiResolutionLayer, NodeSpec
from .readout import RidgeReadout
from .sparse_features import SparseEvents


@dataclass
class RoutingConfig:
    rounds: int = 4
    replace_frac: float = 0.25
    seed: int = 0


@dataclass
class RoutingHistory:
    val_scores: list[float] = field(default_factory=list)  # after each round (best so far)
    accepted: list[bool] = field(default_factory=list)
    rewired_per_round: list[int] = field(default_factory=list)
    learned_index_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _dense(block: np.ndarray, width: int, sparse: bool) -> np.ndarray:
    return SparseEvents(block, width).to_dense() if sparse else block


class GreedyRouter:
    def __init__(self, layer: MultiResolutionLayer, cfg: RoutingConfig | None = None) -> None:
        self.layer = layer
        self.cfg = cfg or RoutingConfig()
        self.history = RoutingHistory()

    def routable(self, node: NodeSpec) -> bool:
        g = self.layer.groups[node.group]
        return 0 < g.tuple_size < self.layer.input_dim

    def fit(self, X: np.ndarray, C: np.ndarray | None, fit_rows: np.ndarray, val_rows: np.ndarray,
            Y_fit: np.ndarray, alpha: float, metric: Callable[[np.ndarray], float],
            extra: np.ndarray | None = None) -> RoutingHistory:
        """Rewire ``self.layer`` in place. ``extra``: fixed features (e.g. PCA) concatenated
        before the hashed features, as in the final readout. ``metric(pred_val)``: higher is better."""
        L, cfg = self.layer, self.cfg
        rng = np.random.default_rng([cfg.seed, 0x5254])
        sparse = L.output.sparse
        out = L.transform_nodes(X, C)
        blocks = [_dense(b, w, sparse) for b, w in zip(out.blocks, out.widths)]

        def evaluate(bl: list[np.ndarray]) -> tuple[float, np.ndarray]:
            F = np.hstack(([extra] if extra is not None else []) + bl).astype(np.float32)
            r = RidgeReadout(alpha).fit(F[fit_rows], Y_fit)
            score = metric(r.predict(F[val_rows]))
            off = 0 if extra is None else extra.shape[1]
            Fc = F[fit_rows] - r.mu
            imp = np.empty(len(bl))
            for j, b in enumerate(bl):
                c = slice(off, off + b.shape[1])
                imp[j] = float(np.mean((Fc[:, c] @ r.W[c]) ** 2))
                off += b.shape[1]
            return score, imp

        best, imp = evaluate(blocks)
        self.history.val_scores.append(best)
        for _ in range(cfg.rounds):
            new_nodes = list(L.nodes)
            changed: list[int] = []
            for gi, g in enumerate(L.groups):
                members = [j for j, n in enumerate(L.nodes) if n.group == gi and self.routable(n)]
                if not members:
                    continue
                k = max(1, int(round(cfg.replace_frac * len(members))))
                d = L.input_dim
                s, cnt = np.zeros(d), np.zeros(d)
                for j in members:
                    for dim in L.nodes[j].dims:
                        s[dim] += imp[j]
                        cnt[dim] += 1
                prior = float(np.mean(imp[members]))
                score_d = (s + prior) / (cnt + 1)
                p = score_d / score_d.sum()
                for j in sorted(members, key=lambda j: imp[j])[:k]:
                    dims = tuple(int(i) for i in rng.choice(d, g.tuple_size, replace=False, p=p))
                    new_nodes[j] = NodeSpec(gi, dims, L.fresh_seed(rng))
                    changed.append(j)
            old_nodes = L.nodes
            L.nodes = new_nodes
            redo = L.transform_nodes(X, C, [new_nodes[j] for j in changed])
            trial = list(blocks)
            for j, b, w in zip(changed, redo.blocks, redo.widths):
                trial[j] = _dense(b, w, sparse)
            score, trial_imp = evaluate(trial)
            ok = score > best
            if ok:
                best, imp, blocks = score, trial_imp, trial
            else:
                L.nodes = old_nodes
            self.history.val_scores.append(best)
            self.history.accepted.append(ok)
            self.history.rewired_per_round.append(len(changed))
        self.history.learned_index_bytes = sum(len(n.dims) for n in L.nodes)  # one byte per dim index
        return self.history
