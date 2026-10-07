"""GGUF -> .hmmodel conversion."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..architecture.config import HashMindConfig
from ..core.layer import HashMindLayer
from ..architecture.model import HashMindParams, rmsnorm
from ..features.hash_layer import TupleWiring
from ..formats.hmmodel import HMModel
from ..gguf.inspector import ModelSummary, inspect_gguf
from ..gguf.reader import GGUFFile, UnsupportedQuantizationError, read_gguf
from .analysis import analyze
from .weights import build_weight_plan

_PROJ_SOURCES = re.compile(r"^blk\.(\d+)\.(ffn_up|ffn_gate|attn_v)\.weight$")


def derive_projection(
    g: GGUFFile, d: int, k: int, n_layers: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Top-k right singular directions of FFN-up/gate and attn-V in the first layers.

    These are the input-space directions the original network is most
    sensitive to, so sign bits of x along them carry more of the learned
    structure than random hyperplanes would. Falls back to a seeded Gaussian
    projection if no source tensor can be dequantized.
    """
    gram = np.zeros((d, d), np.float64)
    used, skipped = [], []
    for name, info in g.tensors.items():
        m = _PROJ_SOURCES.match(name)
        if not m or int(m.group(1)) >= n_layers:
            continue
        if info.shape[-1] != d:
            skipped.append(f"{name}: input dim {info.shape[-1]} != {d}")
            continue
        try:
            W = g.tensor(name).astype(np.float64)
        except UnsupportedQuantizationError as e:
            skipped.append(str(e))
            continue
        gram += W.T @ W
        used.append(name)
    if not used:
        rng = np.random.default_rng(seed)
        P = rng.standard_normal((d, k)).astype(np.float32) / np.sqrt(d)
        return P, {"method": "random_gaussian", "sources": [], "skipped": skipped}
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1][:k]
    P = evecs[:, order].astype(np.float32)
    sv = np.sqrt(np.clip(evals[order], 0, None))
    return P, {
        "method": "svd_gram",
        "sources": used,
        "skipped": skipped,
        "singular_values": sv.tolist(),
        "energy_captured": float(evals[order].sum() / max(evals.sum(), 1e-30)),
    }


def convert(
    gguf_path: str | Path,
    out_path: str | Path | None = None,
    reduced_dim: int = 32,
    lowrank_rank: int = 32,
    lowrank_layers: int | None = None,
    hm_output_dim: int = 2048,
    hm_feature_mode: str = "hash_bits",
    hm_tuple_size: int = 2,
    hm_levels: int = 4,
    hm_nonces: int = 16,
    **config_overrides: Any,
) -> HMModel:
    """GGUF -> .hmmodel.

    ``reduced_dim`` .. ``hm_nonces`` control the phase-2 weight plan and the
    HashMind feature layer; ``config_overrides`` go to the phase-1
    :class:`HashMindConfig` (token-level reservoir model).
    """
    g = read_gguf(gguf_path)
    summary: ModelSummary = inspect_gguf(g)
    report = analyze(summary)
    if "token_embd.weight" not in g.tensors:
        raise ValueError("model has no token_embd.weight; cannot preserve embeddings")
    if not g.can_dequantize("token_embd.weight"):
        raise UnsupportedQuantizationError(
            f"token_embd.weight is {g.tensors['token_embd.weight'].type_name}; "
            "install the optional 'gguf' package for IQ* types"
        )
    emb = g.tensor("token_embd.weight")
    vocab, d = emb.shape
    cfg = HashMindConfig(hidden_dim=d, vocab_size=vocab, **config_overrides)
    cfg.validate()

    lm_head = None
    if "output.weight" in g.tensors:
        lm_head = g.tensor("output.weight") if g.can_dequantize("output.weight") else None
        if lm_head is None:
            report.warnings.append("output.weight not dequantizable; falling back to tied embeddings")
    out_norm = g.tensor("output_norm.weight") if "output_norm.weight" in g.tensors else None

    P, proj_info = derive_projection(g, d, cfg.proj_dim, cfg.projection_source_layers, cfg.seed)
    # Center so each sign bit splits the vocabulary roughly in half.
    z = rmsnorm(emb, None, cfg.norm_eps) @ P
    center = np.median(z, axis=0).astype(np.float32)

    params = HashMindParams(
        embedding=emb,
        lm_head=lm_head,
        output_norm=out_norm,
        projection=P,
        projection_center=center,
        readout=np.zeros((cfg.n_features, d), np.float32),
        readout_bias=np.zeros(d, np.float32),
    )
    md = g.metadata
    source = {k: v for k, v in asdict(summary).items() if k != "tensors"}
    source["tokenizer_tokens"] = md.get("tokenizer.ggml.tokens")
    source["general"] = {k: v for k, v in md.items() if k.startswith("general.")}
    conversion = report.to_dict()
    conversion["projection"] = proj_info
    conversion["readout_trained"] = False

    plan = build_weight_plan(g, reduced_dim, lowrank_rank, lowrank_layers, seed=cfg.seed)
    conversion["weights"] = plan.records_dicts()
    conversion["weights_summary"] = plan.summary()

    layer = HashMindLayer(plan.tensors["transformed/embd_basis"].shape[1], hm_output_dim,
                          seed=cfg.seed, feature_mode=hm_feature_mode, tuple_size=hm_tuple_size,
                          levels=hm_levels, nonces_per_node=hm_nonces)
    layer.fit(plan.tensors["transformed/token_embd_reduced"])
    extra = dict(plan.tensors)
    extra["layer/thresholds"] = layer.thresholds  # type: ignore[assignment]

    model = HMModel(cfg, params, TupleWiring.from_config(cfg), source, conversion,
                    extra, layer.spec.to_dict())
    if out_path is not None:
        model.save(out_path)
    return model
