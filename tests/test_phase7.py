from __future__ import annotations

import struct

import numpy as np

from hashmind.asic.artifacts import ArtifactType, GENERATORS, blake2_share, cuckoo_share, ethash_share, sha256d_share
from hashmind.asic.cuckoo import cuckoo_graph, find_cycles, siphash24
from hashmind.asic.ethash import Ethash
from hashmind.asic.keccak import keccak256, keccak512
from hashmind.core.primitives import get_primitive


def test_keccak_vectors() -> None:
    assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert keccak512(b"").hex().startswith("0eab42de4c3ceb9235fc91acffe746b29c29a8c366b7c60e4e67c466f36a4304")


def test_siphash_reference_vector() -> None:
    k0, k1 = struct.unpack("<QQ", bytes(range(16)))
    m = np.array([int.from_bytes(bytes(range(8)), "little")], np.uint64)
    assert int(siphash24(k0, k1, m)[0]) == 0x93F5F5799A932462


def test_ethash_deterministic_and_exposes_pipeline() -> None:
    E = Ethash(cache_items=16, dataset_items=256)
    a, b = E.hashimoto(b"h" * 32, 7), E.hashimoto(b"h" * 32, 7)
    assert a.mix_hash == b.mix_hash and len(a.mix_hash) == 32 and len(a.dag_indices) == 64
    assert a.final == keccak256(a.seed + a.mix_hash)
    assert E.hashimoto(b"h" * 32, 8).mix_hash != a.mix_hash


def test_cuckoo_cycle_is_a_real_cycle() -> None:
    for nonce in range(400):
        r = find_cycles(b"t", nonce, edge_bits=10, length=8)
        if r.cycles:
            break
    cyc = r.cycles[0]
    u, v = cuckoo_graph(b"t", nonce, 10)
    deg: dict[tuple[str, int], int] = {}
    for e in cyc:
        deg[("u", int(u[e]))] = deg.get(("u", int(u[e])), 0) + 1
        deg[("v", int(v[e]))] = deg.get(("v", int(v[e])), 0) + 1
    assert len(cyc) == 8 and all(d == 2 for d in deg.values()) and cyc == sorted(cyc)


def test_artifacts_expose_only_interface_fields() -> None:
    s = sha256d_share(b"x" * 76, difficulty_bits=6)
    assert s.artifact_type == ArtifactType.NONCE and len(s.artifact_bytes) == 4 and s.internal == {}
    b = blake2_share(b"x" * 76, difficulty_bits=6)
    assert b.artifact_type == ArtifactType.NONCE
    e = ethash_share(b"y", difficulty_bits=2)
    assert e.artifact_type == ArtifactType.MIXHASH and len(e.artifact_bytes) == 8 + 32
    assert "dag_indices" in e.internal  # reference-only; not part of artifact_bytes
    c = cuckoo_share(b"z", edge_bits=10, length=8)
    assert c.artifact_type == ArtifactType.GRAPH_WITNESS and len(c.artifact_bytes) == 8 * 8
    assert set(GENERATORS) >= {"sha256_s9", "blake2s_kadena", "ethash_e9", "equihash_z15", "cuckatoo_g1"}


def test_blake2s_primitive() -> None:
    p = get_primitive("blake2s")
    pl = np.random.default_rng(0).integers(0, 256, (500, 32), dtype=np.uint8)
    w = p.words(3, pl, 8)
    assert w.shape == (500, 8, 2) and abs((w[..., 0] >> 31).mean() - 0.5) < 0.05
    np.testing.assert_array_equal(w, p.words(3, pl, 8))
