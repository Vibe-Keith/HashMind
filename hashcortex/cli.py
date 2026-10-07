"""Command line: ``python -m hashcortex <inspect|analyze|convert|simulate|pipeline> ...``"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .architecture.model import HashCortexModel
from .backends.simulated import SimulatedS9Backend
from .conversion.analysis import analyze, format_report
from .conversion.convert import convert
from .formats.hcmodel import HCModel
from .gguf.inspector import format_summary, inspect_gguf


def _tokens(model: HCModel, spec: str | None, n: int) -> list[int]:
    if spec:
        return [int(x) for x in spec.split(",")]
    return np.random.default_rng(0).integers(0, model.config.vocab_size, n).tolist()


def cmd_inspect(a: argparse.Namespace) -> None:
    s = inspect_gguf(a.gguf)
    print(json.dumps(s.to_dict(), indent=2, default=str) if a.json else format_summary(s))


def cmd_analyze(a: argparse.Namespace) -> None:
    rep = analyze(inspect_gguf(a.gguf))
    print(json.dumps(rep.to_dict(), indent=2, default=str) if a.json else format_report(rep))


def _overrides(a: argparse.Namespace) -> dict[str, int]:
    keys = ("proj_dim", "reservoir_bits", "n_tuples", "tuple_bits", "nonces_per_tuple", "difficulty_bits", "seed")
    return {k: getattr(a, k) for k in keys if getattr(a, k) is not None}


def cmd_convert(a: argparse.Namespace) -> None:
    m = convert(a.gguf, a.out, **_overrides(a))
    p = m.conversion["projection"]
    print(f"wrote {a.out}: features={m.config.n_features} hashes/token={m.config.hashes_per_token} "
          f"projection={p['method']} energy={p.get('energy_captured', 'n/a')}")


def cmd_simulate(a: argparse.Namespace) -> None:
    hc = HCModel.load(a.hcmodel)
    backend = SimulatedS9Backend()
    model = HashCortexModel(hc.config, hc.params, hc.wiring, backend)
    toks = _tokens(hc, a.tokens, a.n)
    t0 = time.perf_counter()
    feats = model.run_features(toks)
    dt = time.perf_counter() - t0
    logits = model.logits(toks, feats)
    print(f"tokens:           {toks}")
    print(f"features shape:   {feats.shape}  density={feats.mean():.3f}")
    print(f"hashes:           {backend.stats.hashes:,} in {dt:.3f}s "
          f"({backend.stats.hashes / max(dt, 1e-9):,.0f} H/s CPU; an S9 does ~1.4e13 H/s)")
    print(f"next-token argmax: {logits.argmax(-1).tolist()}")
    if not hc.conversion.get("readout_trained"):
        print("note: readout untrained (zeros); logits reflect preserved embedding+LM head only")


def cmd_pipeline(a: argparse.Namespace) -> None:
    out = a.out or str(Path(a.gguf).with_suffix(".hcmodel"))
    print("=" * 70, "\n[1] GGUF inspector\n" + "=" * 70)
    s = inspect_gguf(a.gguf)
    print(format_summary(s, max_tensors=12))
    print("=" * 70, "\n[2] Conversion analysis\n" + "=" * 70)
    print(format_report(analyze(s)))
    print("=" * 70, "\n[3] HashCortex representation\n" + "=" * 70)
    a.out = out
    cmd_convert(a)
    print("=" * 70, "\n[4] CPU S9 simulator\n" + "=" * 70)
    a.hcmodel = out
    cmd_simulate(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hashcortex", description="HashCortex-S9 phase 1 tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    def conv_opts(sp: argparse.ArgumentParser) -> None:
        for k in ("proj_dim", "reservoir_bits", "n_tuples", "tuple_bits", "nonces_per_tuple", "difficulty_bits", "seed"):
            sp.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)

    def sim_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--tokens", help="comma-separated token ids")
        sp.add_argument("-n", type=int, default=8, help="random tokens if --tokens not given")

    sp = sub.add_parser("inspect"); sp.add_argument("gguf"); sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_inspect)
    sp = sub.add_parser("analyze"); sp.add_argument("gguf"); sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_analyze)
    sp = sub.add_parser("convert"); sp.add_argument("gguf"); sp.add_argument("-o", "--out", required=True)
    conv_opts(sp); sp.set_defaults(fn=cmd_convert)
    sp = sub.add_parser("simulate"); sp.add_argument("hcmodel"); sim_opts(sp); sp.set_defaults(fn=cmd_simulate)
    sp = sub.add_parser("pipeline"); sp.add_argument("gguf"); sp.add_argument("-o", "--out")
    conv_opts(sp); sim_opts(sp); sp.set_defaults(fn=cmd_pipeline)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
