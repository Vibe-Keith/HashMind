# HashMind (HashMind-S9)

Research prototype. It re-expresses a GGUF language model in a new,
experimental architecture whose nonlinear feature layer is SHA-256d
computation, the only thing an Antminer S9 (BM1387) can do. This repo is
CPU-only so far; the real S9 backend is a later phase.

Formerly **HashCortex**. `import hashcortex` and `.hcmodel` files still work.

**HashMind is not equivalent to the source model.** Phase 2 shows that useful
information from a real GGUF survives the SHA-256 representation. It does not
show that the network was converted. See [docs/PHASE2.md](docs/PHASE2.md) and
[docs/PHASE3.md](docs/PHASE3.md) (real hidden states, next-token prediction,
locality study, BM1387 difficulty floor).

```
model.gguf
  -> GGUF inspector            hashmind/gguf/            (F16/BF16, Q4-Q8, K-quants; IQ* via optional `gguf`)
  -> conversion analysis       hashmind/conversion/analysis.py
  -> weight plan               hashmind/conversion/weights.py   PRESERVED / TRANSFORMED / DISCARDED
  -> .hmmodel                  hashmind/formats/hmmodel.py
  -> HashMind encoder + nodes  hashmind/core/node.py, layer.py
  -> CPU S9 simulator          hashmind/backends/simulated.py
  -> ridge readout             hashmind/core/readout.py
```

## Headline result (TinyLlama 1.1B, token-embedding probe)

| | word_start | char_class |
|---|---:|---:|
| majority | 50.0% | 66.5% |
| PCA-32 (HashMind input) | 98.4% | 95.0% |
| **HashMind, ASIC-native hash_bits** | **95.3%** | **93.5%** |
| HashMind on shuffled embeddings (control) | 49.9% | 64.1% |

Caveat: at these settings the layer has few enough distinct inputs to
precompute as a lookup table. See PHASE2.md, "Interpretation".

## Quick start

```bash
pip install -e .[dev]
python examples/make_tiny_gguf.py tiny.gguf
python -m hashmind pipeline tiny.gguf -n 6

python -m hashmind inspect    model.gguf [--json]
python -m hashmind analyze    model.gguf [--json]
python -m hashmind weights    model.gguf [--lowrank-rank 32 --lowrank-layers N]
python -m hashmind convert    model.gguf -o model.hmmodel [--feature-mode hash_bits]
python -m hashmind simulate   model.hmmodel --tokens 1,2,3
python -m hashmind experiment model.gguf -o docs/results/probe
pytest
```

## Core API

```python
from hashmind.core import HashMindLayer, RidgeReadout

layer = HashMindLayer(input_dim=32, output_dim=2048, seed=0, feature_mode="hash_bits")
F = layer.fit_transform(Z)                 # (N, 2048) float32; layer.stats counts SHA-256d ops
readout = RidgeReadout(10.0).fit_classes(F, labels)
pred = readout.predict_classes(layer.transform(Z_new))
```

`ASICBackend.submit_job / get_result` is unchanged from phase 1.
`SimulatedS9Backend` returns only passing nonces, as the real chip does.
