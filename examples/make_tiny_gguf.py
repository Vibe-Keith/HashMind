"""Write a tiny random llama-shaped GGUF so the pipeline runs without a download.

    python examples/make_tiny_gguf.py tiny.gguf
    python -m hashcortex pipeline tiny.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hashcortex.gguf.constants import GGMLType
from hashcortex.gguf.writer import write_gguf


def make_tiny_gguf(path: str | Path, vocab: int = 256, d: int = 64, ffn: int = 128,
                   layers: int = 2, heads: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    r = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)  # noqa: E731
    t: dict[str, tuple[np.ndarray, GGMLType]] = {
        "token_embd.weight": (r(vocab, d), GGMLType.F16),
        "output_norm.weight": (np.ones(d, np.float32), GGMLType.F32),
        "output.weight": (r(vocab, d), GGMLType.Q8_0),
    }
    for i in range(layers):
        b = f"blk.{i}."
        t |= {
            b + "attn_norm.weight": (np.ones(d, np.float32), GGMLType.F32),
            b + "attn_q.weight": (r(d, d), GGMLType.Q8_0),
            b + "attn_k.weight": (r(d, d), GGMLType.Q8_0),
            b + "attn_v.weight": (r(d, d), GGMLType.Q8_0),
            b + "attn_output.weight": (r(d, d), GGMLType.Q8_0),
            b + "ffn_norm.weight": (np.ones(d, np.float32), GGMLType.F32),
            b + "ffn_gate.weight": (r(ffn, d), GGMLType.Q8_0),
            b + "ffn_up.weight": (r(ffn, d), GGMLType.Q8_0),
            b + "ffn_down.weight": (r(d, ffn), GGMLType.Q8_0),
        }
    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(vocab - 3)]
    md = {
        "general.architecture": "llama",
        "general.name": "tiny-random-llama",
        "general.file_type": 7,
        "llama.context_length": 128,
        "llama.embedding_length": d,
        "llama.block_count": layers,
        "llama.feed_forward_length": ffn,
        "llama.attention.head_count": heads,
        "llama.attention.head_count_kv": heads,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": [0.0] * vocab,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }
    write_gguf(path, md, t)
    return Path(path)


if __name__ == "__main__":
    print(make_tiny_gguf(sys.argv[1] if len(sys.argv) > 1 else "tiny.gguf"))
