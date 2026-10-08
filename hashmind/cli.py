"""Command line: ``python -m hashmind <inspect|analyze|weights|convert|simulate|pipeline|experiment> ...``"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .architecture.model import HashMindModel
from .backends.simulated import SimulatedS9Backend
from .conversion.analysis import analyze, format_report
from .conversion.convert import convert
from .conversion.weights import build_weight_plan, format_weight_plan
from .experiments.token_probe import ProbeConfig, format_results, run_token_probe
from .gguf.reader import read_gguf
from .formats.hmmodel import HMModel
from .pipeline import HashMindPipeline
from .gguf.inspector import format_summary, inspect_gguf


def _tokens(model: HMModel, spec: str | None, n: int) -> list[int]:
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


def cmd_weights(a: argparse.Namespace) -> None:
    plan = build_weight_plan(read_gguf(a.gguf), a.reduced_dim, a.lowrank_rank, a.lowrank_layers)
    print(json.dumps(plan.records_dicts(), indent=2, default=str) if a.json
          else format_weight_plan(plan, max_rows=a.max_rows))


def cmd_convert(a: argparse.Namespace) -> None:
    m = convert(a.gguf, a.out, reduced_dim=a.reduced_dim, lowrank_rank=a.lowrank_rank,
                lowrank_layers=a.lowrank_layers, hm_output_dim=a.hm_output_dim,
                hm_feature_mode=a.feature_mode, **_overrides(a))
    p = m.conversion["projection"]
    ws = m.conversion["weights_summary"]
    print(f"wrote {a.out}")
    print(f"  weights: " + ", ".join(f"{k}={v['records']}" for k, v in ws["by_action"].items())
          + f"; stored {ws['stored_bytes'] / 1e6:.1f} MB")
    print(f"  HashMind layer: {m.hashmind_layer}")
    print(f"  phase-1 token model: features={m.config.n_features} projection={p['method']}")


def cmd_experiment(a: argparse.Namespace) -> None:
    cfg = ProbeConfig(n_samples=a.samples, reduced_dim=a.reduced_dim, output_dim=a.hm_output_dim,
                      feature_mode=a.feature_mode, tuple_size=a.tuple_size, levels=a.levels,
                      nonces=a.nonces, mode_sweep=not a.no_sweep,
                      tuple_sweep=() if a.no_sweep else (1, 3, 4))
    res = run_token_probe(a.gguf, cfg)
    text = format_results(res)
    print(text)
    if a.out:
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2, default=str))
        Path(a.out).with_suffix(".md").write_text(text + "\n")


def cmd_simulate(a: argparse.Namespace) -> None:
    hm = HMModel.load(a.hmmodel)
    toks = _tokens(hm, a.tokens, a.n)
    print(f"tokens: {toks}")
    if hm.hashmind_layer is not None:
        pipe = HashMindPipeline.from_hmmodel(hm, SimulatedS9Backend())
        t0 = time.perf_counter()
        F = pipe.features(hm.params.embedding[np.asarray(toks)])
        dt = time.perf_counter() - t0
        st = pipe.layer.stats
        print("HashMind layer (embedding -> PCA -> SHA-256d features):")
        print(f"  features shape {F.shape}, mean {F.mean():.3f}, ASIC-native={pipe.layer.asic_native}")
        print(f"  SHA-256d logical={st.sha256d_logical:,} executed={st.sha256d_executed:,} in {dt:.3f}s")
        print(f"  first token, first 32 features: {F[0, :32].astype(int).tolist()}")
    backend = SimulatedS9Backend()
    model = HashMindModel(hm.config, hm.params, hm.wiring, backend)
    t0 = time.perf_counter()
    feats = model.run_features(toks)
    dt = time.perf_counter() - t0
    logits = model.logits(toks, feats)
    print("Phase-1 token reservoir model:")
    print(f"  features shape {feats.shape}, density={feats.mean():.3f}")
    print(f"  hashes {backend.stats.hashes:,} in {dt:.3f}s "
          f"({backend.stats.hashes / max(dt, 1e-9):,.0f} H/s CPU; an S9 does ~1.4e13 H/s)")
    print(f"  next-token argmax: {logits.argmax(-1).tolist()}")
    if not hm.conversion.get("readout_trained"):
        print("note: readouts untrained; this run demonstrates the computation path only")


def cmd_pipeline(a: argparse.Namespace) -> None:
    out = a.out or str(Path(a.gguf).with_suffix(".hmmodel"))
    print("=" * 70, "\n[1] GGUF inspector\n" + "=" * 70)
    s = inspect_gguf(a.gguf)
    print(format_summary(s, max_tensors=12))
    print("=" * 70, "\n[2] Conversion analysis\n" + "=" * 70)
    print(format_report(analyze(s)))
    print("=" * 70, "\n[3] HashMind representation (.hmmodel)\n" + "=" * 70)
    a.out = out
    cmd_convert(a)
    print("=" * 70, "\n[4] CPU S9 simulator\n" + "=" * 70)
    a.hmmodel = out
    cmd_simulate(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hashmind", description="HashMind (HashMind-S9) research tools")
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
    def plan_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--reduced-dim", type=int, default=32)
        sp.add_argument("--lowrank-rank", type=int, default=32)
        sp.add_argument("--lowrank-layers", type=int, default=None, help="only first N blocks (default all)")

    def hm_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--hm-output-dim", type=int, default=2048)
        sp.add_argument("--feature-mode", default="hash_bits",
                        choices=["hash_bits", "hash_bytes", "hamming", "bucket", "threshold"])

    sp = sub.add_parser("weights", help="show the weight-preservation plan")
    sp.add_argument("gguf"); sp.add_argument("--json", action="store_true")
    sp.add_argument("--max-rows", type=int, default=40); plan_opts(sp); sp.set_defaults(fn=cmd_weights)
    sp = sub.add_parser("convert"); sp.add_argument("gguf"); sp.add_argument("-o", "--out", required=True)
    conv_opts(sp); plan_opts(sp); hm_opts(sp); sp.set_defaults(fn=cmd_convert)
    sp = sub.add_parser("experiment", help="GGUF -> HashMind -> readout probe")
    sp.add_argument("gguf"); sp.add_argument("-o", "--out", help="write <out>.json and <out>.md")
    sp.add_argument("--samples", type=int, default=6000)
    sp.add_argument("--reduced-dim", type=int, default=32)
    sp.add_argument("--tuple-size", type=int, default=2)
    sp.add_argument("--levels", type=int, default=4)
    sp.add_argument("--nonces", type=int, default=16)
    sp.add_argument("--no-sweep", action="store_true", help="skip feature-mode and tuple-size sweeps")
    hm_opts(sp); sp.set_defaults(fn=cmd_experiment)
    from .frozen.cli import add_parsers
    add_parsers(sub)
    sp = sub.add_parser("simulate"); sp.add_argument("hmmodel"); sim_opts(sp); sp.set_defaults(fn=cmd_simulate)
    sp = sub.add_parser("pipeline"); sp.add_argument("gguf"); sp.add_argument("-o", "--out")
    conv_opts(sp); plan_opts(sp); hm_opts(sp); sim_opts(sp); sp.set_defaults(fn=cmd_pipeline)
    return p


PHASE4_COMMANDS = {
    "phase4-hash-ablation": ("hash_ablation",),
    "phase4-multires": ("multiresolution",),
    "phase4-routing": ("learned_routing",),
    "phase4-sparse": ("sparse_events",),
    "phase4-all": ("hash_ablation", "multiresolution", "learned_routing", "sparse_events"),
}


def cmd_phase4(argv: list[str]) -> int:
    from .experiments.phase4 import run_all
    from .experiments.phase4_common import DEFAULT_SEEDS, load_tasks

    p = argparse.ArgumentParser(prog="hashmind experiment phase4-*")
    p.add_argument("command", choices=sorted(PHASE4_COMMANDS))
    p.add_argument("gguf")
    p.add_argument("-o", "--out", default="docs/results/phase4")
    p.add_argument("--cache", help="npz cache for hidden states (default <out>/hidden_states.npz, not committed)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--n-seq", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--tasks", nargs="+", help="subset of task names (default: all)")
    a = p.parse_args(argv)
    cache = a.cache or str(Path(a.out) / "hidden_states.npz")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks, meta = load_tasks(a.gguf, cache, a.n_seq, a.seq_len)
    if a.tasks:
        tasks = [t for t in tasks if t.name in a.tasks]
    run_all(tasks, meta, a.out, tuple(a.seeds), PHASE4_COMMANDS[a.command])
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 2 and argv[0] == "experiment" and argv[1] == "phase7":
        from .experiments.phase7 import run_phase7
        p = argparse.ArgumentParser(prog="hashmind experiment phase7")
        p.add_argument("gguf", nargs="?"); p.add_argument("-o", "--out", default="docs/results/phase7")
        p.add_argument("--quick", action="store_true")
        a = p.parse_args(argv[2:])
        run_phase7(a.gguf, a.out, quick=a.quick)
        return 0
    if len(argv) >= 2 and argv[0] == "experiment" and argv[1] == "phase6":
        from .experiments.phase6 import run_phase6
        p = argparse.ArgumentParser(prog="hashmind experiment phase6")
        p.add_argument("gguf"); p.add_argument("-o", "--out", default="docs/results/phase6")
        p.add_argument("--backend", default="all", choices=["all", "sha256", "equihash", "heterogeneous"])
        p.add_argument("--quick", action="store_true")
        a = p.parse_args(argv[2:])
        run_phase6(a.gguf, a.out, a.backend, quick=a.quick)
        return 0
    if len(argv) >= 2 and argv[0] == "experiment" and argv[1] == "phase5":
        from .experiments.phase5 import run_phase5
        p = argparse.ArgumentParser(prog="hashmind experiment phase5")
        p.add_argument("gguf"); p.add_argument("-o", "--out", default="docs/results/phase5")
        p.add_argument("--quick", action="store_true", help="tiny smoke run")
        p.add_argument("--fast", action="store_true", help="reduced run (see run_phase5 docstring)")
        a = p.parse_args(argv[2:])
        run_phase5(a.gguf, a.out, quick=a.quick, fast=a.fast)
        return 0
    if len(argv) >= 2 and argv[0] == "experiment" and argv[1] in PHASE4_COMMANDS:
        return cmd_phase4(argv[1:])
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
