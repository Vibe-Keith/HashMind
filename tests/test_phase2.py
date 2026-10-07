from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from hashmind.backends import SimulatedS9Backend, meets_target, sha256d
from hashmind.backends.base import header_with_nonce
from hashmind.conversion.convert import convert
from hashmind.conversion.weights import build_weight_plan, pca_basis, randomized_svd
from hashmind.core import ChallengeConfig, FeatureMode, HashMindLayer, HashMindNode, InputMapping, RidgeReadout
from hashmind.experiments.token_probe import ProbeConfig, format_results, run_token_probe, vocab_tasks
from hashmind.formats.hmmodel import HMModel, read_manifest
from hashmind.gguf import read_gguf
from hashmind.gguf.constants import GGMLType
from hashmind.gguf.quants import dequantize, has_gguf_package
from hashmind.pipeline import HashMindPipeline
from make_tiny_gguf import make_tiny_gguf

MODES = [m.value for m in FeatureMode]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_tiny_gguf(tmp_path_factory.mktemp("m") / "tiny.gguf")


# --- rename / backwards compatibility --------------------------------------

def test_hashcortex_alias_is_same_module() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import hashcortex
        import hashcortex.core.layer as old_layer
        from hashcortex.formats.hcmodel import HCModel
    import hashmind
    import hashmind.core.layer as new_layer

    assert hashcortex is hashmind
    assert old_layer is new_layer
    assert HCModel is HMModel


def test_loads_phase1_hcmodel_format(tiny: Path, tmp_path: Path) -> None:
    import io
    import json
    import zipfile

    m = convert(tiny, tmp_path / "x.hmmodel", hm_output_dim=64)
    # Rewrite as a phase-1 style file: format "hcmodel", v1, no layer/plan.
    with zipfile.ZipFile(tmp_path / "x.hmmodel") as z:
        man = json.loads(z.read("manifest.json"))
        npz = np.load(io.BytesIO(z.read("tensors.npz")))
        keep = {k: npz[k] for k in npz.files if k.split("/")[0] in ("preserved", "readout", "wiring")
                or k.startswith("transformed/projection")}
    man.update(format="hcmodel", format_version=1)
    man.pop("hashmind_layer")
    buf = io.BytesIO()
    np.savez(buf, **keep)
    with zipfile.ZipFile(tmp_path / "old.hcmodel", "w") as z:
        z.writestr("manifest.json", json.dumps(man))
        z.writestr("tensors.npz", buf.getvalue())
    old = HMModel.load(tmp_path / "old.hcmodel")
    assert old.hashmind_layer is None and old.config == m.config


# --- dequantization ---------------------------------------------------------

@pytest.mark.skipif(not has_gguf_package(), reason="reference 'gguf' package not installed")
@pytest.mark.parametrize("name,size,dpos", [
    ("Q2_K", 84, (80, 82)), ("Q3_K", 110, (108,)), ("Q4_K", 144, (0, 2)),
    ("Q5_K", 176, (0, 2)), ("Q6_K", 210, (208,)), ("Q5_0", 22, (0,)), ("Q5_1", 24, (0, 2)),
])
def test_kquants_match_reference(name: str, size: int, dpos: tuple[int, ...]) -> None:
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize as ref

    rng = np.random.default_rng(0)
    b = rng.integers(0, 256, (16, size), dtype=np.uint8)
    for p in dpos:
        b[:, p:p + 2] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
    raw = b.reshape(-1)
    want = ref(raw, GGMLQuantizationType[name]).reshape(-1)
    got = dequantize(raw, GGMLType[name], want.size)
    np.testing.assert_array_equal(got, want)


# --- node --------------------------------------------------------------------

def _node(mode: str, **kw: int) -> HashMindNode:
    m = InputMapping((0, 2), np.array([[-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]], np.float32))
    return HashMindNode(1234, m, ChallengeConfig(mode=mode, nonces=8, **kw))


def test_input_mapping_encode() -> None:
    m = InputMapping((1,), np.array([[-1.0, 0.0, 1.0]], np.float32))
    X = np.array([[9, -2], [9, -0.5], [9, 0.5], [9, 2]], np.float32)
    assert m.encode(X).ravel().tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", MODES)
def test_node_modes_shape_and_determinism(mode: str) -> None:
    n = _node(mode)
    codes = np.array([1, 3], np.uint8)
    f1, f2 = n.evaluate(codes), n.evaluate(codes)
    assert f1.shape == (n.n_features,) and np.array_equal(f1, f2)
    assert np.all((f1 >= 0) & (f1 <= 1))


def test_node_asic_path_matches_digest_path() -> None:
    for mode, kw in (("hash_bits", {}), ("threshold", {"difficulty_bits": 2})):
        n = _node(mode, **kw)
        codes = np.array([2, 0], np.uint8)
        via_asic = n.evaluate(codes, backend=SimulatedS9Backend())
        via_cpu = n.features_from_digests(n.header_prefix(n.payload(codes)))
        np.testing.assert_array_equal(via_asic, via_cpu)
        assert n.challenge_config.asic_native


def test_hash_bits_definition() -> None:
    n = _node("hash_bits")
    prefix = n.header_prefix(n.payload(np.array([0, 0], np.uint8)))
    f = n.features_from_digests(prefix)
    for k in range(8):
        assert f[k] == float(meets_target(sha256d(header_with_nonce(prefix, k)), 1))


def test_digest_modes_flagged_not_native() -> None:
    for mode in ("hash_bytes", "hamming", "bucket"):
        assert not _node(mode).challenge_config.asic_native
        with pytest.raises(ValueError):
            _node(mode).features_from_nonces([0])


# --- layer -------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_layer_shape_and_dtype(mode: str) -> None:
    X = np.random.default_rng(0).standard_normal((50, 12)).astype(np.float32)
    L = HashMindLayer(12, 100, seed=3, feature_mode=mode)
    F = L.fit_transform(X)
    assert F.shape == (50, 100) and F.dtype == np.float32
    assert L.stats.sha256d_logical == 50 * L.n_nodes * 16
    assert 0 < L.stats.sha256d_executed <= L.stats.sha256d_logical


def test_layer_deterministic_and_seed_dependent() -> None:
    X = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
    a = HashMindLayer(8, 64, seed=1).fit_transform(X)
    b = HashMindLayer(8, 64, seed=1).fit_transform(X)
    c = HashMindLayer(8, 64, seed=2).fit_transform(X)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_layer_spec_roundtrip() -> None:
    X = np.random.default_rng(0).standard_normal((30, 8)).astype(np.float32)
    L = HashMindLayer(8, 64, seed=5, feature_mode="bucket", buckets=4).fit(X)
    L2 = HashMindLayer.from_spec(L.spec.to_dict(), L.thresholds)
    np.testing.assert_array_equal(L.transform(X), L2.transform(X))


def test_layer_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        HashMindLayer(4, 8).transform(np.zeros((1, 4), np.float32))


def test_locality_nearby_inputs_share_features() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 16)).astype(np.float32)
    L = HashMindLayer(16, 512, seed=0).fit(X)
    x = X[:1]
    near = L.transform(x + 0.01)
    far = L.transform(rng.standard_normal((1, 16)).astype(np.float32))
    base = L.transform(x)
    assert (near == base).mean() > 0.95
    assert (far == base).mean() < 0.8


# --- readout -------------------------------------------------------------------

def test_ridge_primal_and_dual_agree() -> None:
    rng = np.random.default_rng(0)
    F = rng.standard_normal((20, 30))
    y = rng.standard_normal(20)
    a = RidgeReadout(1.0).fit(F, y).predict(F)
    Fc, yc = F - F.mean(0), y - y.mean()
    W = np.linalg.solve(Fc.T @ Fc + np.eye(30), Fc.T @ yc)
    np.testing.assert_allclose(a.ravel(), Fc @ W + y.mean(), atol=1e-8)


def test_hashmind_readout_learns_clusters() -> None:
    """Labels decodable from SHA features of clustered inputs, not from shuffled ones."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 16)) * 2
    y = rng.integers(0, 3, 600)
    X = (centers[y] + rng.standard_normal((600, 16))).astype(np.float32)
    L = HashMindLayer(16, 512, seed=0)
    F = L.fit_transform(X)
    r = RidgeReadout(10.0).fit_classes(F[:450], y[:450])
    acc = (r.predict_classes(F[450:]) == y[450:]).mean()
    Fs = HashMindLayer(16, 512, seed=0).fit_transform(X[rng.permutation(600)])
    rs = RidgeReadout(10.0).fit_classes(Fs[:450], y[:450])
    acc_s = (rs.predict_classes(Fs[450:]) == y[450:]).mean()
    assert acc > 0.9 and acc_s < 0.55


# --- weights / conversion --------------------------------------------------------

def test_pca_and_svd_deterministic() -> None:
    E = np.random.default_rng(0).standard_normal((300, 20)).astype(np.float32)
    B1, m1, _ = pca_basis(E, 5)
    B2, m2, _ = pca_basis(E, 5)
    np.testing.assert_array_equal(B1, B2)
    np.testing.assert_allclose(B1.T @ B1, np.eye(5), atol=1e-5)
    W = np.random.default_rng(1).standard_normal((40, 30)).astype(np.float32)
    U, S, Vt = randomized_svd(W, 30, seed=0)
    np.testing.assert_allclose((U * S) @ Vt, W, atol=1e-3)


def test_weight_plan_accounts_for_every_tensor(tiny: Path) -> None:
    g = read_gguf(tiny)
    plan = build_weight_plan(g, reduced_dim=8, lowrank_rank=4)
    named = {r.name for r in plan.records}
    assert named == set(g.tensors)
    acts = {(r.name, r.action) for r in plan.records}
    assert ("token_embd.weight", "PRESERVED") in acts and ("token_embd.weight", "TRANSFORMED") in acts
    assert ("blk.0.attn_q.weight", "TRANSFORMED") in acts
    for r in plan.records:
        for k, shp in zip(r.stored_as, r.stored_shapes):
            assert plan.tensors[k].shape == shp
        if r.method.startswith("truncated SVD"):
            assert 0 < r.energy_retained <= 1.0 + 1e-6


def test_weight_plan_lowrank_layers_discards_with_reason(tiny: Path) -> None:
    plan = build_weight_plan(read_gguf(tiny), reduced_dim=8, lowrank_rank=4, lowrank_layers=1)
    d = [r for r in plan.records if r.action == "DISCARDED"]
    assert d and all(r.name.startswith("blk.1.") and "lowrank_layers" in r.note for r in d)


def test_convert_hmmodel_and_pipeline(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "t.hmmodel"
    convert(tiny, out, reduced_dim=8, lowrank_rank=4, hm_output_dim=128)
    man = read_manifest(out)
    assert man["format"] == "hmmodel" and man["format_version"] == 2
    assert man["hashmind_layer"]["output_dim"] == 128
    assert man["conversion"]["weights"]
    hm = HMModel.load(out)
    pipe = HashMindPipeline.from_hmmodel(hm)
    F = pipe.features(hm.params.embedding[:10])
    assert F.shape == (10, 128)
    np.testing.assert_allclose(pipe.reduce(hm.params.embedding[:3]),
                               hm.extra_tensors["transformed/token_embd_reduced"][:3], atol=2e-2)


# --- experiment ----------------------------------------------------------------

def test_vocab_tasks() -> None:
    toks = ["<s>", "▁the", "The", "▁The", "42", ",", "▁", "<0x0A>"]
    t = vocab_tasks(toks)
    ids, word = t["word_start"]
    assert ids.tolist() == [1, 2, 3, 4, 5]
    assert word.tolist() == [1, 0, 1, 0, 0]
    assert t["char_class"][1].tolist() == ["lowercase", "capitalized", "capitalized", "digit", "punct/other"]


def test_probe_end_to_end_synthetic() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((2, 64)) * 1.5
    y = rng.integers(0, 2, 800)
    E = (centers[y] + rng.standard_normal((800, 64))).astype(np.float32)
    cfg = ProbeConfig(n_samples=800, reduced_dim=8, output_dim=256, mode_sweep=False,
                      tuple_sweep=(), neighbor_queries=20)
    res = run_token_probe(embeddings=E, tasks={"cluster": (np.arange(800), y)}, cfg=cfg)
    runs = {r["representation"]: r for r in res["tasks"][0]["runs"]}
    hm = runs["HashMind[hash_bits, tuple=2]"]
    ctrl = next(v for k, v in runs.items() if "SHUFFLED" in k)
    assert hm["accuracy"] > 0.85 and ctrl["accuracy"] < 0.65
    assert hm["sha256d_logical"] > 0 and hm["feature_dim"] == 256
    assert "HashMind phase-2 probe" in format_results(res)
