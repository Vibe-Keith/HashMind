from __future__ import annotations

import numpy as np
import pytest

from hashmind.core import HashMindLayer, HashMindNode, InputMapping, ChallengeConfig, RidgeReadout
from hashmind.core.diagnostics import cell_stats, feature_stats
from hashmind.core.multiresolution import (
    GroupSpec,
    MultiResolutionLayer,
    context_multires_groups,
    single_resolution,
    standard_multires_groups,
)
from hashmind.core.primitives import PRIMITIVES, Sha256dPrimitive, get_primitive, pack_payloads
from hashmind.core.routing import GreedyRouter, RoutingConfig
from hashmind.core.sparse_features import OutputConfig, SparseEvents, SparseRidge, node_output


def _X(n: int = 400, d: int = 12, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


# --- primitives ------------------------------------------------------------------

def test_pack_payloads_matches_node_layout() -> None:
    m = InputMapping((0, 1, 2), np.zeros((3, 3), np.float32))
    node = HashMindNode(5, m, ChallengeConfig())
    codes = np.array([[1, 2, 3], [0, 3, 1]], np.uint8)
    ctx = np.array([[7, 70000], [1, 2]], np.uint32)
    p = pack_payloads(codes)
    pc = pack_payloads(codes, ctx)
    for i in range(2):
        assert p[i].tobytes() == node.payload(codes[i])
        assert pc[i].tobytes() == node.payload(codes[i], ctx[i])


def test_single_resolution_sha_is_bit_exact_with_phase2_layer() -> None:
    X = _X()
    old = HashMindLayer(12, 32 * 16, seed=3, tuple_size=2, levels=4, nonces_per_node=16).fit(X)
    new = single_resolution(12, 32, 2, 4, 16, "sha256d", seed=3).fit(X)
    np.testing.assert_array_equal(old.transform(X), new.transform(X))


def test_sha_parallel_path_matches_serial() -> None:
    p = Sha256dPrimitive()
    pl = np.random.default_rng(0).integers(0, 256, (20000, 32), dtype=np.uint8)
    jobs = [(11, pl, 16), (12, pl[:5], 3)]
    par = p.words_many(jobs, workers=2)
    for (s, q, e), w in zip(jobs, par):
        np.testing.assert_array_equal(w, p.words(s, q, e))


@pytest.mark.parametrize("name", sorted(PRIMITIVES))
def test_primitives_deterministic_and_balanced(name: str) -> None:
    prim = get_primitive(name)
    pl = np.random.default_rng(1).integers(0, 256, (3000, 32), dtype=np.uint8)
    a = prim.words(9, pl, 8)
    assert a.shape == (3000, 8, 2) and a.dtype == np.uint32
    np.testing.assert_array_equal(a, prim.words(9, pl, 8))
    assert not np.array_equal(a, prim.words(10, pl, 8))  # seed matters
    assert abs((a[..., 0] >> 31).mean() - 0.5) < 0.03


# --- multi-resolution layer ---------------------------------------------------------

def test_multires_layout_and_determinism() -> None:
    X = _X()
    g = standard_multires_groups(64, 4)
    assert sum(x.n_nodes for x in g) == 64
    L1 = MultiResolutionLayer(12, g, "splitmix", seed=1).fit(X)
    L2 = MultiResolutionLayer(12, g, "splitmix", seed=1).fit(X)
    F = L1.transform(X)
    assert F.shape == (400, 64 * 4) and L1.evals_per_example == 256
    np.testing.assert_array_equal(F, L2.transform(X))
    ov = [n.dims for n in L1.nodes if L1.groups[n.group].wiring == "overlap"]
    assert len(set(ov[0]) & set(ov[1])) == 2  # stride t/2 -> neighbours share half their dims


def test_context_groups_need_and_hash_context_exactly() -> None:
    X = np.zeros((6, 4), np.float32)
    C = np.array([[1, 9, 0], [1, 9, 0], [2, 9, 0], [1, 8, 0], [2, 8, 0], [2, 8, 0]], np.uint32)
    L = MultiResolutionLayer(4, [GroupSpec("bi", 4, 0, ctx_cols=(0, 1), evals=8)], "sha256d", context_dim=3).fit(X)
    with pytest.raises(ValueError):
        L.transform(X)
    F = L.transform(X, C)
    assert np.array_equal(F[0], F[1]) and np.array_equal(F[4], F[5]) and not np.array_equal(F[0], F[2])
    assert len(context_multires_groups(128)) == 6


# --- sparse events --------------------------------------------------------------------

def test_output_modes_shapes() -> None:
    w = get_primitive("splitmix").words(1, np.zeros((50, 32), np.uint8) + np.arange(50, dtype=np.uint8)[:, None], 64)
    assert node_output(w[:, :16], OutputConfig("bits")).shape == (50, 16)
    b = node_output(w[:, :2], OutputConfig("bucket", buckets=8))
    assert b.shape == (50, 2) and b[:, 0].max() < 8 and b[:, 1].min() >= 8
    f = node_output(w, OutputConfig("first_nonce", buckets=16, difficulty_bits=4))
    assert f.shape == (50, 1) and 0 <= f.min() and f.max() <= 16
    h = node_output(w[:, :16], OutputConfig("hit", difficulty_bits=2))
    assert ((h >= 0).mean() - 0.25) < 0.1


def test_sparse_ridge_equals_dense_ridge() -> None:
    X = _X(300)
    L = MultiResolutionLayer(12, standard_multires_groups(32, 2), "splitmix",
                             OutputConfig("bucket", buckets=8)).fit(X)
    ev = L.transform(X)
    assert isinstance(ev, SparseEvents) and ev.active_per_example() == 64
    Y = np.random.default_rng(0).standard_normal((300, 3))
    d = RidgeReadout(3.0).fit(ev.to_dense(), Y).predict(ev.to_dense())
    s = SparseRidge(3.0).fit(ev, Y).predict(ev)
    np.testing.assert_allclose(d, s, atol=1e-8)
    hit = MultiResolutionLayer(12, standard_multires_groups(16, 8), "splitmix", OutputConfig("hit")).fit(X)
    eh = hit.transform(X)
    np.testing.assert_allclose(RidgeReadout(1.0).fit(eh.to_dense(), Y).predict(eh.to_dense()),
                               SparseRidge(1.0).fit(eh, Y).predict(eh), atol=1e-8)


# --- diagnostics -------------------------------------------------------------------------

def test_cell_and_feature_stats() -> None:
    tr = [np.array([1, 1, 2, 3], np.uint64)]
    te = [np.array([1, 4], np.uint64)]
    c = cell_stats(tr, te)
    assert c["unique_cells"] == 3 and c["singleton_cell_fraction"] == pytest.approx(2 / 3)
    assert c["repeated_cell_rate"] == 0.5 and c["test_coverage"] == 0.5
    f = feature_stats(np.array([[1, 0, 0], [1, 1, 0]], np.float32))
    assert f["active_per_example"] == 1.5 and f["dead_feature_fraction"] == pytest.approx(1 / 3)
    assert f["feature_reuse_rate"] == 0.5


# --- routing ------------------------------------------------------------------------------

def test_router_is_sparse_monotone_and_deterministic() -> None:
    X = _X(600, 12, 2)
    y = (X[:, 3] * X[:, 7] > 0).astype(float)
    Y = np.stack([y, 1 - y], 1) * 2 - 1
    fit, val = np.arange(450), np.arange(450, 600)
    metric = lambda P: float((P.argmax(1) == (1 - y[val]).astype(int)).mean())  # noqa: E731

    def run() -> tuple[MultiResolutionLayer, list]:
        L = single_resolution(12, 24, 2, 4, 8, "splitmix", seed=0).fit(X)
        h = GreedyRouter(L, RoutingConfig(rounds=3, seed=0)).fit(X, None, fit, val, Y[fit], 1.0, metric)
        return L, h

    L, h = run()
    L2, _ = run()
    assert [n.dims for n in L.nodes] == [n.dims for n in L2.nodes]
    assert all(len(n.dims) == 2 for n in L.nodes)
    assert all(b >= a for a, b in zip(h.val_scores, h.val_scores[1:]))
    assert h.learned_index_bytes == 48 and len(h.accepted) == 3


# --- benchmark plumbing -------------------------------------------------------------------

def test_probe_tasks_reproduce_phase2_splits() -> None:
    from hashmind.experiments.phase4_common import Rep, probe_tasks, run_rep
    from hashmind.experiments.token_probe import ProbeConfig, run_token_probe, vocab_tasks

    rng = np.random.default_rng(0)
    toks = [("▁" if i % 3 else "") + ("ab" if i % 2 else "Cd") for i in range(400)]
    E = rng.standard_normal((400, 24)).astype(np.float32)
    E[:, 0] += np.array([t.startswith("▁") for t in toks]) * 2
    cfg = ProbeConfig(n_samples=300, reduced_dim=8, output_dim=64, mode_sweep=False, tuple_sweep=())
    old = run_token_probe(embeddings=E, tasks=vocab_tasks(toks), cfg=cfg)
    new = probe_tasks(E, toks, cfg)
    for t_old, t_new in zip(old["tasks"], new):
        pca_old = next(r["accuracy"] for r in t_old["runs"] if r["representation"].startswith("PCA"))
        assert run_rep(t_new, Rep("PCA only", "pca"), 0)["accuracy"] == pytest.approx(pca_old)


def test_cli_routes_phase4_commands() -> None:
    from hashmind.cli import PHASE4_COMMANDS

    assert set(PHASE4_COMMANDS) == {"phase4-hash-ablation", "phase4-multires", "phase4-routing", "phase4-sparse",
                                    "phase4-all"}
