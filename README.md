# HashCortex-S9

Research prototype: convert a GGUF language model into an architecture whose
nonlinear feature layer is SHA-256d computation, as done by an Antminer S9
(BM1387). Phase 1 is CPU-only; no hardware I/O.

```
model.gguf
    -> GGUF inspector          hashcortex/gguf/
    -> conversion analysis     hashcortex/conversion/analysis.py
    -> HashCortex repr         hashcortex/conversion/convert.py, formats/hcmodel.py
    -> CPU S9 simulator        hashcortex/backends/simulated.py, features/hash_layer.py
```

The output is **not** equivalent to the source model. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
pip install -e .[dev]           # numpy only at runtime
python examples/make_tiny_gguf.py tiny.gguf
python -m hashcortex pipeline tiny.gguf -n 6

python -m hashcortex inspect  model.gguf [--json]
python -m hashcortex analyze  model.gguf [--json]
python -m hashcortex convert  model.gguf -o model.hcmodel [--n-tuples 256 --tuple-bits 8 ...]
python -m hashcortex simulate model.hcmodel --tokens 1,2,3
pytest
```

Dequantization: F32, F16, BF16, F64, I8/16/32, Q8_0, Q4_0, Q4_1. Other types
(K-quants, IQ) are inspected and classified, but conversion falls back to a
random projection if FFN weights cannot be read, and needs a supported type for
`token_embd`/`output`.

## Backend interface

```python
class ASICBackend(ABC):
    def submit_job(self, job: HashJob) -> int: ...
    def get_result(self, job_id: int, timeout: float | None = None) -> HashResult | None: ...
```

`HashJob` = 76-byte header prefix + nonce range + difficulty. `HashResult`
returns only passing nonces, matching real chip behaviour.
`SimulatedS9Backend` is bit-exact against Bitcoin's genesis block (see tests).
