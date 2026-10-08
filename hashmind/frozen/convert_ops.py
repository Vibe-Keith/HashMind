"""Deterministic, training-free conversions of individual transformer operations.

Every function here maps (frozen weights, current activation, fixed settings) to
an approximation of the original operation. Nothing is fitted to data.

Conversion kinds
----------------
exact      the original operation (reference).
quant      activation quantized to ``bits`` (per-row symmetric absmax) before the
           op; rounding is round-to-nearest ("rtn") or stochastic with a hash
           dither (primitive name, e.g. "sha256d" / "splitmix").
sampled    Monte-Carlo matmul: S input columns drawn with probability
           proportional to |x_i| * ||W[:, i]|| using hash-derived stratified
           uniforms; unbiased estimate of x W^T. Host work: S x out adds.
topk       deterministic non-hash control for ``sampled``: keep the S largest
           |x_i| * ||W[:, i]|| terms exactly.
lut        elementwise nonlinearity via a 2**bits-entry table over a fixed
           input range; ``slots`` < 2**bits makes it a hash-addressed table
           (code -> hash -> slot), slot value = mean of the codes it holds.
hashed     embedding rows stored in a hash-addressed table of M slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashsrc import HashSource, fingerprint, op_key

LINEAR_OPS = ("attn_qkv", "attn_out", "mlp_in", "mlp_down", "lm_head")
QUANT_OPS = ("attn_scores", "attn_values")
LUT_OPS = ("act", "softmax")
OP_CLASSES = ("embedding",) + LINEAR_OPS[:2] + QUANT_OPS + ("softmax", "mlp_in", "act", "mlp_down", "lm_head")

LUT_RANGES = {"act": (-12.0, 12.0), "softmax": (-30.0, 0.0)}  # fixed a priori, data-free


@dataclass(frozen=True)
class OpConv:
    kind: str = "exact"  # exact | quant | sampled | topk | lut | hashed
    bits: int = 8
    rounding: str = "rtn"  # rtn | <primitive name>
    samples: int = 256
    primitive: str = "sha256d"  # sampled / lut slots / hashed
    slots: int = 0  # lut: 0 = direct indexing; hashed: number of table slots

    def label(self) -> str:
        k = self.kind
        if k == "exact":
            return "exact"
        if k == "quant":
            return f"quant{self.bits}-{self.rounding}"
        if k in ("sampled", "topk"):
            return f"{k}{self.samples}" + (f"-{self.primitive}" if k == "sampled" else "")
        if k == "lut":
            return f"lut{self.bits}" + (f"-{self.primitive}{self.slots}" if self.slots else "-direct")
        return f"hashed{self.slots}-{self.primitive}"

    @property
    def uses_hash(self) -> bool:
        return (self.kind == "quant" and self.rounding != "rtn") or self.kind == "sampled" or \
            (self.kind == "lut" and self.slots > 0) or self.kind == "hashed"

    @property
    def hash_primitive(self) -> str | None:
        if not self.uses_hash:
            return None
        return self.rounding if self.kind == "quant" else self.primitive


# ----------------------------------------------------------------- quantization --

def quantize_rows(x: np.ndarray, bits: int, u: np.ndarray | None = None, signed: bool = True) -> np.ndarray:
    """Per-row absmax quantization, returns the dequantized array. ``u`` = dither in [0,1)
    for stochastic rounding (floor(v + u)); None = round to nearest."""
    qmax = (2 ** (bits - 1) - 1) if signed else (2**bits - 1)
    a = np.abs(x).max(-1, keepdims=True)
    s = np.where(a > 0, a / qmax, 1.0).astype(np.float32)
    v = x / s
    q = np.round(v) if u is None else np.floor(v + u)
    lo = -qmax if signed else 0
    return (np.clip(q, lo, qmax) * s).astype(np.float32)


def quantize_weight(W: np.ndarray, bits: int, group: int = 32, u: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Groupwise symmetric quantization along the input dim. Returns (codes int16, scales f32)."""
    out, d = W.shape
    g = group if d % group == 0 else d
    Wg = W.reshape(out, d // g, g)
    qmax = 2 ** (bits - 1) - 1
    a = np.abs(Wg).max(-1, keepdims=True)
    s = np.where(a > 0, a / qmax, 1.0).astype(np.float32)
    v = Wg / s
    q = np.round(v) if u is None else np.floor(v + u.reshape(Wg.shape))
    return np.clip(q, -qmax, qmax).astype(np.int16).reshape(out, d), s[..., 0]


def dequantize_weight(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    out, d = codes.shape
    g = d // scales.shape[1]
    return (codes.reshape(out, -1, g).astype(np.float32) * scales[..., None]).reshape(out, d)


# ---------------------------------------------------------------------- ops --

class Converter:
    """Applies OpConv settings; owns the hash sources and counts their evaluations."""

    def __init__(self) -> None:
        self.sources: dict[str, HashSource] = {}
        self.counts: dict[str, dict[str, float]] = {}
        self._colnorm: dict[int, np.ndarray] = {}
        self._lut: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def src(self, name: str) -> HashSource:
        if name not in self.sources:
            self.sources[name] = HashSource(name)
        return self.sources[name]

    def _count(self, op: str, **kv: float) -> None:
        c = self.counts.setdefault(op, {})
        for k, v in kv.items():
            c[k] = c.get(k, 0.0) + float(v)

    def dither(self, conv: OpConv, key: int, x: np.ndarray, op: str) -> np.ndarray | None:
        if conv.rounding == "rtn":
            return None
        R = x.reshape(-1, x.shape[-1])
        u = self.src(conv.rounding).uniform(key, fingerprint(R), R.shape[1])
        self._count(op, hashes=u.size)
        return u.reshape(x.shape)

    def linear(self, x: np.ndarray, W: np.ndarray, conv: OpConv, key: int, op: str) -> np.ndarray:
        """x (R, d) @ W(out, d).T under ``conv``."""
        R, d = x.shape
        out = W.shape[0]
        if conv.kind == "exact":
            self._count(op, fp_macs=R * d * out)
            return x @ W.T
        if conv.kind == "quant":
            xq = quantize_rows(x, conv.bits, self.dither(conv, key, x, op))
            self._count(op, int_macs=R * d * out)
            return xq @ W.T
        if conv.kind in ("sampled", "topk"):
            S = min(conv.samples, d)
            cid = id(W)
            if cid not in self._colnorm:
                self._colnorm[cid] = np.linalg.norm(W, axis=0).astype(np.float32)
            imp = np.abs(x) * self._colnorm[cid][None, :]
            xt = np.zeros_like(x)
            rows = np.arange(R)[:, None]
            if conv.kind == "topk":
                idx = np.argpartition(-imp, S - 1, axis=1)[:, :S]
                xt[rows, idx] = x[rows, idx]
            else:
                tot = imp.sum(1, keepdims=True)
                p = imp / np.where(tot > 0, tot, 1)
                cdf = np.cumsum(p, 1)
                u = self.src(conv.primitive).uniform(key, fingerprint(x), S)
                self._count(op, hashes=u.size)
                t = (np.arange(S)[None, :] + u) / S  # stratified
                idx = np.minimum(np.stack([np.searchsorted(cdf[r], t[r], side="right") for r in range(R)]), d - 1)
                w = x[rows, idx] / (S * np.maximum(p[rows, idx], 1e-30))
                np.add.at(xt, (np.repeat(np.arange(R), S), idx.ravel()), w.ravel())
            self._count(op, adds=R * S * out)
            return xt @ W.T
        raise ValueError(f"{conv.kind} is not a linear conversion")

    def quant(self, x: np.ndarray, conv: OpConv, key: int, op: str, signed: bool = True) -> np.ndarray:
        if conv.kind == "exact":
            return x
        if conv.kind != "quant":
            raise ValueError(f"{op}: only quant conversions")
        if signed:
            return quantize_rows(x, conv.bits, self.dither(conv, key, x, op))
        return quantize_rows(x, conv.bits, self.dither(conv, key, x, op), signed=False)

    def lut(self, x: np.ndarray, fn, conv: OpConv, op: str) -> np.ndarray:
        if conv.kind == "exact":
            return fn(x)
        lo, hi = LUT_RANGES[op]
        n = 2**conv.bits
        k = (op, conv.bits, conv.slots, conv.primitive)
        if k not in self._lut:
            levels = np.linspace(lo, hi, n).astype(np.float32)
            vals = fn(levels).astype(np.float32)
            if conv.slots:
                slot = self.src(conv.primitive).slot(op_key(op, -1), np.arange(n), conv.slots)
                table = np.zeros(conv.slots, np.float32)
                cnt = np.bincount(slot, minlength=conv.slots)
                np.add.at(table, slot, vals)
                table = table / np.maximum(cnt, 1)
                vals = table[slot]  # effective value per code after hash collisions
            self._lut[k] = (levels, vals)
        levels, vals = self._lut[k]
        code = np.clip(np.round((x - lo) / (hi - lo) * (n - 1)), 0, n - 1).astype(np.int64)
        self._count(op, lut_lookups=x.size)
        return vals[code]


def hashed_embedding(E: np.ndarray, conv: OpConv, conv_src: Converter) -> np.ndarray:
    """Effective embedding matrix when rows live in a hash-addressed table of ``slots`` slots
    (slot value = mean of the token rows hashed there; deterministic, weights only)."""
    if conv.kind == "exact":
        return E
    V = E.shape[0]
    slot = conv_src.src(conv.primitive).slot(0xE3BD, np.arange(V), conv.slots)
    table = np.zeros((conv.slots, E.shape[1]), np.float64)
    np.add.at(table, slot, E)
    cnt = np.bincount(slot, minlength=conv.slots)
    table /= np.maximum(cnt, 1)[:, None]
    return table[slot].astype(np.float32)


def parse_label(label: str) -> OpConv:
    """Inverse of OpConv.label(): exact | quant8-rtn | quant4-sha256d | sampled256-sha256d | topk256 |
    lut6-direct | lut6-sha256d32 | hashed128000-sha256d."""
    import re

    if label == "exact":
        return OpConv()
    m = re.fullmatch(r"quant(\d+)-(\w+)", label)
    if m:
        return OpConv("quant", int(m[1]), m[2])
    m = re.fullmatch(r"sampled(\d+)-(\w+)", label)
    if m:
        return OpConv("sampled", samples=int(m[1]), primitive=m[2])
    m = re.fullmatch(r"topk(\d+)", label)
    if m:
        return OpConv("topk", samples=int(m[1]))
    m = re.fullmatch(r"lut(\d+)-direct", label)
    if m:
        return OpConv("lut", int(m[1]))
    m = re.fullmatch(r"lut(\d+)-([a-z0-9]+?)(\d+)", label)
    if m:
        return OpConv("lut", int(m[1]), primitive=m[2], slots=int(m[3]))
    m = re.fullmatch(r"hashed(\d+)-(\w+)", label)
    if m:
        return OpConv("hashed", slots=int(m[1]), primitive=m[2])
    raise ValueError(f"cannot parse conversion label {label!r}")
