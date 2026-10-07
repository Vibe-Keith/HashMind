"""Reference numpy forward pass for llama-architecture GGUF models (host-side).

This runs the ORIGINAL network so experiments have real hidden states and a
teacher distribution. It is not part of HashMind's computation path.

Processing is layer-major over a batch of equal-length sequences: each block's
weights are dequantized once, applied to every sequence, then freed, so peak
memory stays near (one layer of fp32 weights + activations).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..gguf.reader import GGUFFile


@dataclass
class LlamaHParams:
    n_layer: int
    d_model: int
    n_head: int
    n_head_kv: int
    head_dim: int
    rope_base: float
    rms_eps: float

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "LlamaHParams":
        md = g.metadata
        arch = md.get("general.architecture")
        if arch != "llama":
            raise ValueError(f"reference forward supports 'llama' only, got {arch!r}")
        d = int(md["llama.embedding_length"])
        nh = int(md["llama.attention.head_count"])
        return cls(int(md["llama.block_count"]), d, nh, int(md.get("llama.attention.head_count_kv", nh)),
                   int(md.get("llama.attention.key_length", d // nh)),
                   float(md.get("llama.rope.freq_base", 10000.0)),
                   float(md.get("llama.attention.layer_norm_rms_epsilon", 1e-5)))


@dataclass
class ForwardResult:
    hidden: dict[int, np.ndarray]  # layer index -> (B, T, d) residual stream *before* that block; n_layer = final (pre-norm)
    final_normed: np.ndarray  # (B, T, d) rmsnorm(h_final) * output_norm, input to the LM head
    seconds_per_layer: list[float] = field(default_factory=list)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def rope(x: np.ndarray, base: float) -> np.ndarray:
    """ggml NORM rope (adjacent pairs), as used for llama GGUF files. x: (B, H, T, hd)."""
    T, hd = x.shape[2], x.shape[3]
    inv = base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


class LlamaReference:
    def __init__(self, g: GGUFFile) -> None:
        self.g = g
        self.hp = LlamaHParams.from_gguf(g)
        self._embd: np.ndarray | None = None

    def W(self, name: str) -> np.ndarray:
        return np.ascontiguousarray(self.g.tensor(name), dtype=np.float32)

    @property
    def embedding(self) -> np.ndarray:
        if self._embd is None:
            self._embd = self.W("token_embd.weight")
        return self._embd

    def lm_head(self) -> np.ndarray:
        return self.W("output.weight") if "output.weight" in self.g.tensors else self.embedding

    def forward(self, tokens: np.ndarray, capture: set[int] | None = None, verbose: bool = False) -> ForwardResult:
        """tokens: (B, T) int. ``capture``: layer indices whose input residual to keep."""
        hp = self.hp
        B, T = tokens.shape
        capture = set(range(hp.n_layer + 1)) if capture is None else set(capture)
        x = self.embedding[tokens].astype(np.float32)  # (B, T, d)
        hidden: dict[int, np.ndarray] = {}
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
        group = hp.n_head // hp.n_head_kv
        secs = []
        for L in range(hp.n_layer):
            t0 = time.perf_counter()
            if L in capture:
                hidden[L] = x.copy()
            p = f"blk.{L}."
            h = rmsnorm(x, self.W(p + "attn_norm.weight"), hp.rms_eps).reshape(B * T, -1)
            q = (h @ self.W(p + "attn_q.weight").T).reshape(B, T, hp.n_head, hp.head_dim).transpose(0, 2, 1, 3)
            k = (h @ self.W(p + "attn_k.weight").T).reshape(B, T, hp.n_head_kv, hp.head_dim).transpose(0, 2, 1, 3)
            v = (h @ self.W(p + "attn_v.weight").T).reshape(B, T, hp.n_head_kv, hp.head_dim).transpose(0, 2, 1, 3)
            q, k = rope(q, hp.rope_base), rope(k, hp.rope_base)
            k = np.repeat(k, group, axis=1)
            v = np.repeat(v, group, axis=1)
            att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hp.head_dim) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ v).transpose(0, 2, 1, 3).reshape(B * T, -1)
            x = x + (o @ self.W(p + "attn_output.weight").T).reshape(B, T, -1)
            h = rmsnorm(x, self.W(p + "ffn_norm.weight"), hp.rms_eps).reshape(B * T, -1)
            gate = h @ self.W(p + "ffn_gate.weight").T
            up = h @ self.W(p + "ffn_up.weight").T
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + (act @ self.W(p + "ffn_down.weight").T).reshape(B, T, -1)
            secs.append(time.perf_counter() - t0)
            if verbose:
                print(f"  layer {L:2d}  {secs[-1]:.1f}s", flush=True)
        if hp.n_layer in capture:
            hidden[hp.n_layer] = x.copy()
        fin = rmsnorm(x, self.W("output_norm.weight"), hp.rms_eps)
        return ForwardResult(hidden, fin, secs)

    def logits(self, final_normed: np.ndarray, head: np.ndarray | None = None) -> np.ndarray:
        head = self.lm_head() if head is None else head
        return final_normed @ head.T
