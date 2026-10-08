"""Frozen transformer -> operation graph -> backend assignment -> execution plan.

The plan is deterministic and contains no learned parameters: it records, per
operation, which backend runs it, which conversion it uses, the artifact values
or FLOPs it needs per token, and the modelled/measured cost. It is stored in the
frozen .hmmodel manifest (``execution_plan``) and drives the runtime: an artifact
op assigned to a backend makes the runtime draw that op's randomness/indices from
the backend's primitive (CPU: splitmix, S9: SHA-256d, Equihash ASIC: Equihash).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..frozen.convert_ops import OpConv
from ..llm.llama import LlamaHParams
from .backends import ARTIFACT_KINDS, Cost, HardwareBackend, Operation

# op class -> where its artifact comes from in the runtime
_ARTIFACT_OF = {"quant": "hash_dither", "sampled": "index_sampling", "lut": "table_address", "hashed": "table_address"}


def _artifact_values(op: str, c: OpConv, hp: LlamaHParams, ffn: int, vocab: int, T: int) -> tuple[float, bool]:
    """(values per token, conversion_time_only) for the artifact op attached to a conversion."""
    d, nh, nkv, hd = hp.d_model, hp.n_head, hp.n_head_kv, hp.head_dim
    if c.kind == "quant" and c.rounding != "rtn":
        per = {"attn_qkv": d, "attn_out": nh * hd, "mlp_in": d, "mlp_down": ffn, "lm_head": d,
               "attn_scores": 2 * nh * hd, "attn_values": nh * T + nh * hd}[op]
        return per * (1 if op == "lm_head" else hp.n_layer), False
    if c.kind == "sampled":
        return c.samples * (1 if op == "lm_head" else hp.n_layer), False
    if c.kind == "lut" and c.slots:
        return 2.0**c.bits, True
    if c.kind == "hashed":
        return float(vocab), True
    return 0.0, False


def build_graph(hp: LlamaHParams, conv: dict[str, OpConv], ffn: int, vocab: int, context: int = 64) -> list[Operation]:
    """Per-token operations of one decode step at context length ``context`` (layers folded:
    per-layer ops carry n_layer x their per-layer cost)."""
    d, nh, nkv, hd, L = hp.d_model, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_layer
    T = context
    ops = [
        Operation("embedding", "embedding_lookup", outputs=("x",), flops=d),
        Operation("attn_norm", "norm", inputs=("x",), outputs=("h",), flops=4 * d * L),
        Operation("attn_qkv", "matmul", inputs=("h", "Wqkv"), outputs=("q", "k", "v"), flops=2 * d * (nh + 2 * nkv) * hd * L),
        Operation("rope", "elementwise", inputs=("q", "k"), outputs=("q", "k"), flops=6 * (nh + nkv) * hd * L),
        Operation("attn_scores", "attention", inputs=("q", "k"), outputs=("s",), flops=2 * nh * T * hd * L),
        Operation("softmax", "elementwise", inputs=("s",), outputs=("a",), flops=5 * nh * T * L),
        Operation("attn_values", "attention", inputs=("a", "v"), outputs=("o",), flops=2 * nh * T * hd * L),
        Operation("attn_out", "matmul", inputs=("o", "Wo"), outputs=("y",), flops=2 * nh * hd * d * L),
        Operation("residual_attn", "residual", inputs=("x", "y"), outputs=("x",), flops=d * L),
        Operation("ffn_norm", "norm", inputs=("x",), outputs=("h",), flops=4 * d * L),
        Operation("mlp_in", "matmul", inputs=("h", "Win"), outputs=("g", "u"), flops=2 * d * 2 * ffn * L),
        Operation("act", "elementwise", inputs=("g", "u"), outputs=("m",), flops=6 * ffn * L),
        Operation("mlp_down", "matmul", inputs=("m", "Wdown"), outputs=("y",), flops=2 * ffn * d * L),
        Operation("residual_mlp", "residual", inputs=("x", "y"), outputs=("x",), flops=d * L),
        Operation("output_norm", "norm", inputs=("x",), outputs=("h",), flops=4 * d),
        Operation("lm_head", "lm_head", inputs=("h", "Wout"), outputs=("logits",), flops=2 * d * vocab),
    ]
    out = []
    for o in ops:
        c = conv.get(o.name)
        if c is not None and c.kind != "exact":
            o = replace(o, conversion=c.label())
            if c.kind == "sampled":  # host work shrinks to S x out adds per row
                S = min(c.samples, d if o.name != "mlp_down" else ffn)
                width = {"attn_qkv": (nh + 2 * nkv) * hd, "attn_out": d, "mlp_in": 2 * ffn, "mlp_down": d,
                         "lm_head": vocab}[o.name]
                o = replace(o, flops=S * width * (1 if o.name == "lm_head" else L))
            kind = _ARTIFACT_OF.get(c.kind)
            vals, conv_time = _artifact_values(o.name, c, hp, ffn, vocab, T)
            if kind and vals:
                out.append(Operation(f"{o.name}.{kind}", kind, inputs=o.inputs[:1], outputs=(f"{o.name}.artifact",),
                                     values=vals, conversion=c.label(),
                                     attrs={"for": o.name, "conversion_time_only": conv_time}))
                o = replace(o, inputs=o.inputs + (f"{o.name}.artifact",))
        out.append(o)
    return out


@dataclass
class Step:
    backend: str
    operation: str
    kind: str
    inputs: list[str]
    outputs: list[str]
    conversion: str
    values_per_token: float
    flops_per_token: float
    cost_per_token: dict[str, Any]
    conversion_time_only: bool = False


@dataclass
class ExecutionPlan:
    steps: list[Step]
    context: int
    policy: str
    backends: list[str]
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ExecutionPlan":
        d = json.loads(s)
        return cls([Step(**x) for x in d["steps"]], d["context"], d["policy"], d["backends"], d.get("notes", []))

    def assignment(self) -> dict[str, str]:
        return {s.operation: s.backend for s in self.steps}

    def summary(self) -> dict[str, Any]:
        """Per-token time: CPU work is serial with ASIC waits that it depends on (no overlap
        assumed for the decode critical path); conversion-time ops excluded."""
        rt = [s for s in self.steps if not s.conversion_time_only]
        by: dict[str, float] = {}
        energy = 0.0
        for s in rt:
            by[s.backend] = by.get(s.backend, 0.0) + s.cost_per_token["seconds"]
            energy += s.cost_per_token["energy_j"]
        total = sum(by.values())
        return {"seconds_per_token": total, "tokens_per_s": 1 / total if total else float("inf"),
                "seconds_by_backend": by, "energy_j_per_token": energy,
                "asic_values_per_token": sum(s.values_per_token for s in rt if s.backend != "cpu"),
                "conversion_time_values": sum(s.values_per_token for s in self.steps if s.conversion_time_only)}

    def runtime_conv(self, conv: dict[str, OpConv], primitives: dict[str, str]) -> dict[str, OpConv]:
        """OpConv map for the runtime: each hashed conversion uses the primitive of the backend
        its artifact op was assigned to."""
        out = dict(conv)
        for s in self.steps:
            if s.kind in ARTIFACT_KINDS:
                op = s.operation.split(".")[0]
                c = conv[op]
                prim = primitives[s.backend]
                out[op] = replace(c, rounding=prim) if c.kind == "quant" else replace(c, primitive=prim)
        return out


def plan(ops: list[Operation], backends: dict[str, HardwareBackend], policy: str = "min_cost",
         forced: dict[str, str] | None = None, context: int = 64) -> ExecutionPlan:
    """policy: min_cost (cheapest supporting backend per op) or forced (``forced`` maps
    artifact op kind or op name -> backend; others fall back to CPU)."""
    forced = forced or {}
    steps, notes = [], []
    for o in ops:
        cand = [b for b in backends.values() if b.supports(o)]
        want = forced.get(o.name) or forced.get(o.kind)
        if want:
            b = backends[want]
            if not b.supports(o):
                notes.append(f"{o.name}: {want} does not support {o.kind}; fell back to cpu")
                b = backends["cpu"]
        elif policy == "min_cost" or o.kind not in ARTIFACT_KINDS:
            b = min(cand, key=lambda b: b.estimate_cost(o).seconds)
        else:
            b = backends["cpu"]
        c: Cost = b.estimate_cost(o)
        steps.append(Step(b.name, o.name, o.kind, list(o.inputs), list(o.outputs), o.conversion, o.values, o.flops,
                          c.to_dict(), bool(o.attrs.get("conversion_time_only"))))
    return ExecutionPlan(steps, context, policy if not forced else "forced", sorted(backends), notes)
