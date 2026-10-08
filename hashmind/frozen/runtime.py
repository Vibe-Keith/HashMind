"""Frozen-model runtime: the original llama forward pass with per-operation-class
conversions plugged in. With every op ``exact`` it is the reference implementation
(numerically equal to :class:`hashmind.llm.llama.LlamaReference`, tested).

Operation classes (see convert_ops.OP_CLASSES):

    embedding     token -> row lookup
    attn_qkv      q/k/v projections          attn_scores  q.k^T
    softmax       exp inside the softmax     attn_values  softmax @ v
    attn_out      output projection          mlp_in       gate/up projections
    act           SiLU                       mlp_down     down projection
    lm_head       output projection to logits

Nothing here learns. Weights come from the GGUF (optionally deterministically
re-quantized); conversions are fixed functions of weights + activations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..gguf.reader import GGUFFile, read_gguf
from ..llm.llama import LlamaHParams, rmsnorm, rope
from .convert_ops import OP_CLASSES, Converter, OpConv, hashed_embedding
from .hashsrc import op_key

EXACT = OpConv()


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


@dataclass
class FrozenWeights:
    hp: LlamaHParams
    embedding: np.ndarray
    head: np.ndarray
    output_norm: np.ndarray
    layers: list[dict[str, np.ndarray]]  # attn_norm, wqkv, wo, ffn_norm, w_in (gate;up), w_down
    source: dict[str, Any] = field(default_factory=dict)

    def matrices(self) -> list[tuple[str, np.ndarray]]:
        out = []
        for L, d in enumerate(self.layers):
            out += [(f"blk.{L}.{k}", d[k]) for k in ("wqkv", "wo", "w_in", "w_down")]
        return out + [("output", self.head)]

    @classmethod
    def from_gguf(cls, g: GGUFFile | str | Path, verbose: bool = False) -> "FrozenWeights":
        g = read_gguf(g) if not isinstance(g, GGUFFile) else g
        hp = LlamaHParams.from_gguf(g)
        W = lambda n: np.ascontiguousarray(g.tensor(n), dtype=np.float32)  # noqa: E731
        layers = []
        for L in range(hp.n_layer):
            p = f"blk.{L}."
            layers.append({
                "attn_norm": W(p + "attn_norm.weight"),
                "wqkv": np.concatenate([W(p + "attn_q.weight"), W(p + "attn_k.weight"), W(p + "attn_v.weight")]),
                "wo": W(p + "attn_output.weight"),
                "ffn_norm": W(p + "ffn_norm.weight"),
                "w_in": np.concatenate([W(p + "ffn_gate.weight"), W(p + "ffn_up.weight")]),
                "w_down": W(p + "ffn_down.weight"),
            })
            if verbose:
                print(f"  dequantized block {L}", flush=True)
        E = W("token_embd.weight")
        head = W("output.weight") if "output.weight" in g.tensors else E
        return cls(hp, E, head, W("output_norm.weight"), layers)


@dataclass
class TensorErr:
    n: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    ref_sq: float = 0.0
    cos_sum: float = 0.0
    rows: int = 0

    def add(self, ref: np.ndarray, out: np.ndarray) -> None:
        r = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
        o = out.reshape(-1, out.shape[-1]).astype(np.float64)
        d = o - r
        ok = np.isfinite(r).all(1) & np.isfinite(o).all(1)
        r, o, d = r[ok], o[ok], d[ok]
        self.n += d.size
        self.abs_sum += float(np.abs(d).sum())
        self.sq_sum += float((d * d).sum())
        self.ref_sq += float((r * r).sum())
        nr, no = np.linalg.norm(r, axis=1), np.linalg.norm(o, axis=1)
        good = (nr > 0) & (no > 0)
        self.cos_sum += float(((r * o).sum(1)[good] / (nr[good] * no[good])).sum())
        self.rows += int(good.sum())

    def summary(self) -> dict[str, float]:
        return {"mae": self.abs_sum / max(self.n, 1), "rmse": float(np.sqrt(self.sq_sum / max(self.n, 1))),
                "relative_error": float(np.sqrt(self.sq_sum / max(self.ref_sq, 1e-30))),
                "cosine": self.cos_sum / max(self.rows, 1)}


class FrozenRuntime:
    def __init__(self, weights: FrozenWeights, conv: dict[str, OpConv] | None = None,
                 layers: set[int] | None = None) -> None:
        self.w = weights
        self.conv = {k: EXACT for k in OP_CLASSES}
        self.conv.update(conv or {})
        bad = set(self.conv) - set(OP_CLASSES)
        if bad:
            raise ValueError(f"unknown op classes {bad}")
        self.layers = layers  # None = conversions in every layer
        self.cv = Converter()
        self._E = None
        self.probe: dict[str, TensorErr] | None = None

    def c(self, op: str, L: int) -> OpConv:
        if self.layers is not None and L >= 0 and L not in self.layers:
            return EXACT
        return self.conv[op]

    def _apply(self, op: str, fn, conv: OpConv):
        out = fn(conv)
        if self.probe is not None and conv.kind != "exact":
            ref = fn(EXACT)
            self.probe.setdefault(op, TensorErr()).add(ref, out)
        return out

    @property
    def embedding(self) -> np.ndarray:
        if self._E is None:
            self._E = hashed_embedding(self.w.embedding, self.conv["embedding"], self.cv)
        return self._E

    def forward(self, tokens: np.ndarray, pad: np.ndarray | None = None, probe: bool = False,
                last_only: bool = False) -> np.ndarray:
        """tokens (B, T) -> logits (B, T, V) float32 (or (B, V) if ``last_only``).
        ``pad[b]`` = number of left-padding positions in row b (masked out as keys)."""
        hp, w = self.w.hp, self.w
        B, T = tokens.shape
        self.probe = {} if probe else None
        x = self.embedding[tokens].astype(np.float32)
        if probe and self.conv["embedding"].kind != "exact":
            self.probe.setdefault("embedding", TensorErr()).add(w.embedding[tokens], x)
        mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)[None, None]
        if pad is not None and np.any(pad):
            m = np.broadcast_to(mask, (B, 1, T, T)).copy()
            for b, n in enumerate(pad):
                m[b, 0, :, :n] = -np.inf
                m[b, 0, np.arange(n), np.arange(n)] = 0.0  # pad rows attend to themselves (avoid NaN)
            mask = m
        nh, nkv, hd = hp.n_head, hp.n_head_kv, hp.head_dim
        group = nh // nkv
        cnt = self.cv._count
        for L, lw in enumerate(w.layers):
            h = rmsnorm(x, lw["attn_norm"], hp.rms_eps).reshape(B * T, -1)
            qkv = self._apply("attn_qkv", lambda c: self.cv.linear(h, lw["wqkv"], c, op_key("attn_qkv", L),
                                                                   "attn_qkv"), self.c("attn_qkv", L))
            dq, dk = nh * hd, nkv * hd
            q = qkv[:, :dq].reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
            k = qkv[:, dq:dq + dk].reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
            v = qkv[:, dq + dk:].reshape(B, T, nkv, hd).transpose(0, 2, 1, 3)
            q, k = rope(q, hp.rope_base), rope(k, hp.rope_base)
            k, v = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)

            def scores(c: OpConv) -> np.ndarray:
                qq = self.cv.quant(q, c, op_key("attn_scores.q", L), "attn_scores")
                kk = self.cv.quant(k, c, op_key("attn_scores.k", L), "attn_scores")
                return (qq @ kk.transpose(0, 1, 3, 2)) / np.sqrt(hd)

            att = self._apply("attn_scores", scores, self.c("attn_scores", L)) + mask
            cnt("attn_scores", fp_macs=B * nh * T * T * hd)
            z = att - att.max(-1, keepdims=True)
            e = self._apply("softmax", lambda c: np.where(np.isfinite(z), self.cv.lut(np.where(
                np.isfinite(z), z, 0), np.exp, c, "softmax"), 0.0), self.c("softmax", L))
            att = e / e.sum(-1, keepdims=True)

            def values(c: OpConv) -> np.ndarray:
                aa = self.cv.quant(att, c, op_key("attn_values.a", L), "attn_values", signed=False)
                vv = self.cv.quant(v.transpose(0, 1, 3, 2), c, op_key("attn_values.v", L), "attn_values")
                return aa @ vv.transpose(0, 1, 3, 2)

            o = self._apply("attn_values", values, self.c("attn_values", L))
            cnt("attn_values", fp_macs=B * nh * T * T * hd)
            o = o.transpose(0, 2, 1, 3).reshape(B * T, -1)
            x = x + self._apply("attn_out", lambda c: self.cv.linear(o, lw["wo"], c, op_key("attn_out", L),
                                                                     "attn_out"), self.c("attn_out", L)).reshape(B, T, -1)
            h = rmsnorm(x, lw["ffn_norm"], hp.rms_eps).reshape(B * T, -1)
            gu = self._apply("mlp_in", lambda c: self.cv.linear(h, lw["w_in"], c, op_key("mlp_in", L), "mlp_in"),
                             self.c("mlp_in", L))
            ff = gu.shape[1] // 2
            gate, up = gu[:, :ff], gu[:, ff:]
            a = self._apply("act", lambda c: self.cv.lut(gate, silu, c, "act"), self.c("act", L)) * up
            x = x + self._apply("mlp_down", lambda c: self.cv.linear(a, lw["w_down"], c, op_key("mlp_down", L),
                                                                     "mlp_down"), self.c("mlp_down", L)).reshape(B, T, -1)
        fin = rmsnorm(x, w.output_norm, hp.rms_eps)
        if last_only:
            fin = fin[:, -1:]
        Tn = fin.shape[1]
        hf = fin.reshape(B * Tn, -1)
        logits = self._apply("lm_head", lambda c: self.cv.linear(hf, w.head, c, op_key("lm_head", -1), "lm_head"),
                             self.c("lm_head", -1))
        logits = logits.reshape(B, Tn, -1)
        return logits[:, 0] if last_only else logits

    def generate(self, prompts: list[list[int]], n_new: int) -> list[list[int]]:
        """Greedy decoding; prompts batched with left padding (pad keys masked)."""
        seqs = [list(p) for p in prompts]
        for _ in range(n_new):
            T = max(len(s) for s in seqs)
            pad = np.array([T - len(s) for s in seqs])
            toks = np.array([[0] * n + s for n, s in zip(pad, seqs)])
            nxt = self.forward(toks, pad, last_only=True).argmax(-1)
            for s, t in zip(seqs, nxt):
                s.append(int(t))
        return [s[len(p):] for s, p in zip(seqs, prompts)]

    def hash_evaluations(self) -> dict[str, int]:
        return {k: s.evaluations for k, s in self.cv.sources.items()}

    def reset_counts(self) -> None:
        self.cv.counts = {}
        for s in self.cv.sources.values():
            s.evaluations = 0
