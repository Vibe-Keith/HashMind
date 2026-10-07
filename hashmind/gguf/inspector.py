"""High-level model inspection on top of the GGUF reader."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .reader import GGUFFile, read_gguf

_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class TensorSummary:
    name: str
    shape: tuple[int, ...]
    type: str
    n_elements: int
    n_bytes: int | None
    layer: int | None


@dataclass
class ModelSummary:
    path: str
    gguf_version: int
    architecture: str
    name: str | None
    parameter_count: int
    total_tensor_bytes: int
    quantization: dict[str, int]  # type name -> tensor count
    dominant_quantization: str
    file_type: int | None
    embedding_dim: int | None
    n_layers: int | None
    context_length: int | None
    n_heads: int | None
    n_kv_heads: int | None
    head_dim: int | None
    ffn_dim: int | None
    vocab_size: int | None
    tokenizer: dict[str, Any]
    tensors: list[TensorSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _arch_key(md: dict[str, Any], arch: str, key: str) -> Any:
    return md.get(f"{arch}.{key}")


def inspect_gguf(source: str | Path | GGUFFile) -> ModelSummary:
    g = source if isinstance(source, GGUFFile) else read_gguf(source)
    md = g.metadata
    arch = str(md.get("general.architecture", "unknown"))

    tensors: list[TensorSummary] = []
    qcount: Counter[str] = Counter()
    qelems: Counter[str] = Counter()
    layers: set[int] = set()
    params = 0
    nbytes = 0
    for t in g.tensors.values():
        m = _LAYER_RE.match(t.name)
        layer = int(m.group(1)) if m else None
        if layer is not None:
            layers.add(layer)
        tensors.append(TensorSummary(t.name, t.shape, t.type_name, t.n_elements, t.n_bytes, layer))
        qcount[t.type_name] += 1
        qelems[t.type_name] += t.n_elements
        params += t.n_elements
        nbytes += t.n_bytes or 0

    def tshape(name: str) -> tuple[int, ...] | None:
        return g.tensors[name].shape if name in g.tensors else None

    emb = _arch_key(md, arch, "embedding_length")
    emb_shape = tshape("token_embd.weight")
    if emb is None and emb_shape:
        emb = emb_shape[-1]

    n_layers = _arch_key(md, arch, "block_count")
    if n_layers is None and layers:
        n_layers = max(layers) + 1

    n_heads = _arch_key(md, arch, "attention.head_count")
    n_kv = _arch_key(md, arch, "attention.head_count_kv") or n_heads
    head_dim = _arch_key(md, arch, "attention.key_length")
    if head_dim is None and n_heads and emb:
        head_dim = int(emb) // int(n_heads)

    ffn = _arch_key(md, arch, "feed_forward_length")
    if ffn is None and (s := tshape("blk.0.ffn_up.weight")):
        ffn = s[0]

    tokens = md.get("tokenizer.ggml.tokens")
    vocab = len(tokens) if tokens is not None else (emb_shape[0] if emb_shape else None)
    tok: dict[str, Any] = {
        k.removeprefix("tokenizer."): (f"<array len={len(v)}>" if isinstance(v, list) else v)
        for k, v in md.items()
        if k.startswith("tokenizer.") and not k.startswith("tokenizer.chat_template")
    }
    if "tokenizer.chat_template" in md:
        tok["chat_template"] = "<present>"
    if tokens:
        tok["sample_tokens"] = list(tokens[:8])

    return ModelSummary(
        path=str(g.path),
        gguf_version=g.version,
        architecture=arch,
        name=md.get("general.name"),
        parameter_count=params,
        total_tensor_bytes=nbytes,
        quantization=dict(qcount),
        dominant_quantization=qelems.most_common(1)[0][0] if qelems else "none",
        file_type=md.get("general.file_type"),
        embedding_dim=emb,
        n_layers=n_layers,
        context_length=_arch_key(md, arch, "context_length"),
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        ffn_dim=ffn,
        vocab_size=vocab,
        tokenizer=tok,
        tensors=tensors,
    )


def format_summary(s: ModelSummary, max_tensors: int = 40) -> str:
    lines = [
        f"Model:          {s.name or '?'}  ({s.path})",
        f"Architecture:   {s.architecture}   GGUF v{s.gguf_version}",
        f"Parameters:     {s.parameter_count:,}",
        f"Tensor bytes:   {s.total_tensor_bytes:,}",
        f"Quantization:   dominant={s.dominant_quantization}  {s.quantization}",
        f"Embedding dim:  {s.embedding_dim}",
        f"Layers:         {s.n_layers}",
        f"Attention:      heads={s.n_heads} kv_heads={s.n_kv_heads} head_dim={s.head_dim}",
        f"FFN dim:        {s.ffn_dim}",
        f"Context length: {s.context_length}",
        f"Vocab size:     {s.vocab_size}",
        "Tokenizer:",
    ]
    lines += [f"  {k}: {v}" for k, v in s.tokenizer.items()]
    lines.append(f"Tensors ({len(s.tensors)}):")
    for t in s.tensors[:max_tensors]:
        lines.append(f"  {t.name:<32} {str(t.shape):<20} {t.type}")
    if len(s.tensors) > max_tensors:
        lines.append(f"  ... {len(s.tensors) - max_tensors} more")
    return "\n".join(lines)
