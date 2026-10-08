# Heterogeneous backends (Phase 6 design)

HashMind runs a frozen GGUF model with three kinds of backend:

```
Frozen GGUF ─► HashMind compiler ─► operation graph ─► backend assignment ─► execution plan (.hmmodel)
                                                                          │
                       ┌──────────────────────────┬───────────────────────┼──────────────────────┐
                       ▼                          ▼                       ▼                      │
                 SHA-256 / S9               Equihash ASIC             Host CPU                   │
          hash-derived values/events     solution indices      all tensor math, glue,           │
                       └──────────────────────────┴──── artifacts ─► reconstruction ─► logits ◄─┘
```

## The rule

An ASIC backend only executes its native primitive and returns what that hardware actually returns. Everything that turns those artifacts into tensors appears in the plan as a CPU operation, with its own cost. Nothing hides host arithmetic behind a "hash" label.

| Backend | Native capability | Returns | Operations it supports |
|---|---|---|---|
| `cpu` | general math | full tensors | everything: matmul, attention, norm, residual, elementwise, lookups, and artifacts via splitmix |
| `sha256_s9` | SHA-256d at the 2^-32 share floor | one hash-derived value per share | `hash_dither`, `index_sampling`, `table_address`, `threshold_event` |
| `equihash_z15` | Equihash (200, 9) search | full solutions for shares only (512 × 21-bit indices) | `index_sampling`, `table_address`, `hash_dither` |

No ASIC backend supports `matmul`, `attention`, `norm`, `residual`, `elementwise` or `lm_head`. A forced assignment of those to an ASIC falls back to the CPU and is recorded in `plan.notes`.

## Artifact operations

These are created by a conversion and come from the Phase-5 conversion study (`hashmind/frozen/convert_ops.py`).

| Conversion | Artifact op | Values per token | Host work that remains |
|---|---|---|---|
| `quant{b}-<prim>` (hash-dithered activations) | `hash_dither` | one per quantized input element | the full matmul (integer), scaling, residual |
| `sampled{S}-<prim>` (Monte-Carlo matmul) | `index_sampling` | S per row | CDF build, sample → column, S × out adds |
| `lut{b}-<prim>{slots}` (hash-addressed nonlinearity table) | `table_address` | 2^b, once at conversion time | table read per element |
| `hashed{M}-<prim>` (hash-addressed embedding table) | `table_address` | vocab, once at conversion time | table read per token |

## Cost model (`hashmind/hetero/backends.py`)

- **CPU:** FLOP/s and values/s are **measured** on the host. Energy uses an assumed package power.
- **S9:** 13.5 TH/s and one value per 2^32 hashes give ≈ 3,143 values/s. The link is an 80-byte job plus a 9-byte share over a 115.2 kbaud UART, plus an assumed 2 ms latency. Nominal power is 1,323 W. **Modelled.**
- **Z15 Pro:** 840 kSol/s nominal; ≈ 7.6 visible shares/s per the stock-firmware log, used by default; 512 × 21 bits = 336 32-bit values per solution. The link is a ~2.8 kB submit line over 100 Mbit/s Ethernet, plus an assumed 50 ms latency. Power is 2,780 W. **Modelled.**
- **Per-token time:** the sum over operations on the decode critical path. No CPU/ASIC overlap is assumed, because every artifact is needed before the CPU op that consumes it. Conversion-time artifacts (tables) are reported separately and excluded.

## Execution plan

`hashmind/hetero/plan.py`:

- **`build_graph`** produces the per-token operation list, with one artifact op attached to every converted operation.
- **`plan(..., policy="min_cost")`** picks the cheapest supporting backend for each operation. `policy="forced"` takes an artifact-kind → backend map.
- **`ExecutionPlan.to_json()`** is stored in the frozen `.hmmodel` manifest under `execution_plan` (`convert-frozen --backend ...`). The plan holds no learned parameters, only assignments, conversion labels, counts and costs.
- **`runtime_conv`** makes the runtime draw each artifact from the assigned backend's primitive: CPU uses splitmix, S9 uses SHA-256d, Equihash uses the solution-index stream.

## CLI

```
hashmind hardware-info
hashmind analyze-backends model.gguf --backend heterogeneous --op mlp_in=quant8-sha256d
hashmind benchmark-backends
hashmind experiment phase6 model.gguf --backend all|sha256|equihash|heterogeneous
hashmind convert-frozen model.gguf -o m.hmmodel --op lm_head=sampled1024-equihash --backend equihash
hashmind compare model.gguf m.hmmodel
hashmind benchmark-frozen model.gguf m.hmmodel
```

Everything runs in software emulation. The Equihash primitive solves a small (48, 5) instance on the CPU to produce a real solution-index stream. The (200, 9) reference is used for correctness, information-channel and benchmark measurements.

Results and the conclusion are in [PHASE6.md](PHASE6.md).
