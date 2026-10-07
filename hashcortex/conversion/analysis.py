"""Classify each GGUF tensor by what HashCortex can do with it.

Categories
----------
PRESERVED    Used as-is (possibly re-encoded) on the host. Learned values kept.
TRANSFORMED  Not executed as-is, but its learned values are distilled into a
             HashCortex parameter (e.g. principal directions -> host projection).
REPLACED     Its computation is removed and substituted by the SHA-256 feature
             layer / reservoir. Learned values are discarded.
HOST-ONLY    Cheap elementwise ops (norms, biases) that must stay on the host;
             the BM1387 cannot do them, but they are not the bottleneck.

The BM1387 computes double SHA-256 over 80-byte block headers and reports
nonces whose hash meets a target. Nothing else. So every matmul, softmax,
activation function and normalization is either host-side or replaced.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..gguf.inspector import ModelSummary


class Disposition(str, Enum):
    PRESERVED = "PRESERVED"
    TRANSFORMED = "TRANSFORMED"
    REPLACED = "REPLACED"
    HOST_ONLY = "HOST-ONLY"


@dataclass(frozen=True)
class Rule:
    pattern: str
    disposition: Disposition
    role: str
    reason: str


# Patterns follow llama.cpp tensor naming. First match wins.
RULES: tuple[Rule, ...] = (
    Rule(r"^token_embd\.weight$", Disposition.PRESERVED, "token embedding",
         "Table lookup; host does it. Primary carrier of learned token semantics."),
    Rule(r"^output\.weight$", Disposition.PRESERVED, "LM head",
         "Host readout hidden->vocab logits. Kept as the final projection."),
    Rule(r"norm\.weight$|norm\.bias$|_norm\.", Disposition.HOST_ONLY, "normalization",
         "RMSNorm/LayerNorm gain; elementwise float math, host only."),
    Rule(r"^blk\.\d+\.attn_(q|k)\.(weight|bias)$", Disposition.REPLACED, "attention Q/K",
         "Softmax(QK^T) attention has no hash analogue; replaced by binary hash reservoir context."),
    Rule(r"^blk\.\d+\.attn_qkv\.weight$", Disposition.REPLACED, "fused attention QKV",
         "Fused attention projection; replaced by hash reservoir context."),
    Rule(r"^blk\.\d+\.attn_(v|output)\.(weight|bias)$", Disposition.TRANSFORMED, "attention V/O",
         "Value/output subspace kept as candidate directions for the host projection."),
    Rule(r"^blk\.\d+\.ffn_(up|gate)(_exps)?\.(weight|bias)$", Disposition.TRANSFORMED, "FFN up/gate",
         "Dominant input directions (SVD) seed the host projection that feeds the hash layer; "
         "the activation nonlinearity is REPLACED by SHA-256."),
    Rule(r"^blk\.\d+\.ffn_down(_exps)?\.(weight|bias)$", Disposition.TRANSFORMED, "FFN down",
         "Could initialize the readout prior; in v1 only its subspace statistics are recorded."),
    Rule(r"^blk\.\d+\.ffn_gate_inp\.weight$", Disposition.REPLACED, "MoE router",
         "Expert routing replaced by hash-tuple selection."),
    Rule(r"rope_freqs|rope_factors", Disposition.REPLACED, "rotary position",
         "Position encoded via reservoir recurrence / position salt in the hash header."),
)

# Operations (not tensors) and their fate.
OPERATIONS: dict[str, tuple[Disposition, str]] = {
    "embedding lookup": (Disposition.PRESERVED, "host table lookup"),
    "matmul (projections)": (Disposition.REPLACED, "BM1387 has no multiply; one low-rank host projection remains"),
    "self-attention / softmax": (Disposition.REPLACED, "binary hash reservoir over token history"),
    "FFN activation (SiLU/GELU)": (Disposition.REPLACED, "SHA-256d n-tuple feature bits"),
    "RMSNorm / LayerNorm": (Disposition.HOST_ONLY, "host float math"),
    "residual connections": (Disposition.HOST_ONLY, "host add"),
    "LM head + softmax sampling": (Disposition.HOST_ONLY, "host readout using preserved output.weight"),
    "feature readout (ridge)": (Disposition.HOST_ONLY, "new linear layer, trained on host"),
}


@dataclass
class TensorDisposition:
    name: str
    shape: tuple[int, ...]
    type: str
    n_elements: int
    disposition: Disposition
    role: str
    reason: str


@dataclass
class ConversionReport:
    model: str
    architecture: str
    parameter_count: int
    tensors: list[TensorDisposition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def totals(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for d in Disposition:
            ts = [t for t in self.tensors if t.disposition == d]
            n = sum(t.n_elements for t in ts)
            out[d.value] = {
                "tensors": len(ts),
                "parameters": n,
                "fraction": n / self.parameter_count if self.parameter_count else 0.0,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for t in d["tensors"]:
            t["disposition"] = t["disposition"].value
        d["totals"] = self.totals()
        d["operations"] = {k: {"disposition": v[0].value, "note": v[1]} for k, v in OPERATIONS.items()}
        return d


def classify_tensor(name: str) -> Rule:
    for r in RULES:
        if re.search(r.pattern, name):
            return r
    return Rule("", Disposition.REPLACED, "unrecognized",
                "No HashCortex mapping; dropped. Review manually.")


def analyze(summary: ModelSummary) -> ConversionReport:
    rep = ConversionReport(summary.name or summary.path, summary.architecture, summary.parameter_count)
    for t in summary.tensors:
        r = classify_tensor(t.name)
        rep.tensors.append(TensorDisposition(t.name, t.shape, t.type, t.n_elements,
                                             r.disposition, r.role, r.reason))
        if r.role == "unrecognized":
            rep.warnings.append(f"unrecognized tensor '{t.name}' marked REPLACED")
    if summary.architecture not in {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "phi3"}:
        rep.warnings.append(f"architecture '{summary.architecture}' not specifically tested; "
                            "rules assume llama.cpp tensor naming")
    if "output.weight" not in {t.name for t in summary.tensors}:
        rep.warnings.append("no output.weight: model uses tied embeddings; LM head = token_embd^T")
    rep.warnings.append("The HashCortex model is NOT functionally equivalent to the source model. "
                        "Only embeddings and the LM head are kept verbatim; everything between "
                        "them is replaced and must be re-fit (readout) before it is useful.")
    return rep


def format_report(rep: ConversionReport) -> str:
    lines = [f"Conversion report: {rep.model} ({rep.architecture}), {rep.parameter_count:,} params", ""]
    lines.append(f"{'Disposition':<12} {'tensors':>8} {'parameters':>16} {'share':>7}")
    for k, v in rep.totals().items():
        lines.append(f"{k:<12} {int(v['tensors']):>8} {int(v['parameters']):>16,} {v['fraction']:>6.1%}")
    lines += ["", "Per-role:"]
    seen: dict[tuple[str, Disposition], tuple[int, str]] = {}
    for t in rep.tensors:
        key = (t.role, t.disposition)
        c, reason = seen.get(key, (0, t.reason))
        seen[key] = (c + 1, reason)
    for (role, d), (c, reason) in seen.items():
        lines.append(f"  [{d.value:<11}] {role:<18} x{c:<4} {reason}")
    lines += ["", "Operations:"]
    for op, (d, note) in OPERATIONS.items():
        lines.append(f"  [{d.value:<11}] {op:<28} {note}")
    if rep.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in rep.warnings]
    return "\n".join(lines)
