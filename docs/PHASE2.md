# HashMind Phase 2: GGUF -> HashMind layer -> readout

Status: experimental. HashMind has **not** converted the original neural
network. Phase 2 tests one narrower claim: useful information from a real
GGUF model survives being re-expressed as SHA-256d features.

## Pipeline

```
GGUF model                         hashmind/gguf/        (reader, K-quant dequant)
    |  extract representation      token_embd rows (dequantized)
    v
deterministic reduction            hashmind/conversion/weights.py   PCA basis B (d x r), mean m
    |  z = (x - m) B
    v
HashMind encoder                   InputMapping: each node reads `tuple_size` dims,
    |                              quantized to `levels` equal-mass bins
    v
SHA-256 challenge generation       80-byte header: "HMN1" | sha256(salt:seed) | level codes | seed | difficulty
    v
CPU S9 simulator                   SimulatedS9Backend (ASIC-native modes) or CPU digests (sim-only modes)
    v
hash / nonce features              hashmind/core/node.py, layer.py
    v
HashMind feature vector  ->  RidgeReadout  ->  prediction       hashmind/core/readout.py
```

## Components

| Piece | Where | Notes |
|---|---|---|
| `HashMindNode(seed, input_mapping, challenge_config)` | `core/node.py` | encode -> SHA-256d over N nonces -> features |
| `HashMindLayer(input_dim, output_dim, seed, feature_mode)` | `core/layer.py` | bank of nodes, returns `(N, output_dim)` float32; counts SHA-256d ops |
| `RidgeReadout` | `core/readout.py` | closed-form ridge; regression or one-vs-rest classes |
| `build_weight_plan` | `conversion/weights.py` | PRESERVED / TRANSFORMED / DISCARDED record for every tensor |
| `HashMindPipeline` | `pipeline.py` | `.hmmodel` -> reduce -> layer -> readout |
| `run_token_probe` | `experiments/token_probe.py` | the experiment below |

### Feature modes

| mode | feature per nonce | ASIC-native? |
|---|---|---|
| `hash_bits` (1 bit) | 1 if digest < 2^255 (a difficulty-1 share) | yes |
| `threshold` | 1 if digest meets `difficulty_bits` (sparse, p = 2^-d) | yes |
| `hash_bits` (>1 bit), `hash_bytes`, `hamming`, `bucket` | derived from the digest | **no**: the BM1387 never returns digests. These modes exist only for comparison in simulation |

## Weight preservation (TinyLlama 1.1B, `python -m hashmind weights`)

| Tensor | Action | Method | Energy retained |
|---|---|---|---|
| `token_embd` | PRESERVED + TRANSFORMED | verbatim fp16; PCA to 32 dims | 6.1% variance in 32 dims |
| `output` (LM head) | PRESERVED + TRANSFORMED | verbatim fp16; projected onto the embedding basis | 1.7% |
| norms (45) | PRESERVED | verbatim fp32 (host-only ops) | 100% |
| 154 block matrices (q,k,v,o,gate,up,down) | TRANSFORMED | randomized SVD, rank 32, fixed seed | 6% (ffn_down) to 82% (attn_k) |
| none | DISCARDED | | |

Every tensor gets a record. `lowrank_layers=N` limits SVD to the first N
blocks and records the others as DISCARDED with that reason. The low-rank
block factors are **stored but not executed**: no part of HashMind runs
attention or an FFN.

## Experiment (`python -m hashmind experiment model.gguf`)

**Model:** TinyLlama-1.1B-Chat, IQ2_XXS mix (token_embd is Q2_K, 32000 x 2048).
Source: the npm data package `slaunt-model-tinyllama-1b-part{1,2}`, because
huggingface.co is blocked from the build container. That provenance is
third-party. The file parses as a valid GGUF v3, llama architecture, 22
layers, 32k SentencePiece vocabulary.

**Tasks:** labels come from token strings; the readout sees only the embedding.

- `word_start`: does the token begin with `▁`? (binary)
- `char_class`: lowercase, capitalized, digit, or punct/other (4 classes)

6000 tokens per task, 75/25 split, seed 0. Ridge alpha picked on an inner
validation split. HashMind settings: 2048 features, r=32, tuple_size=2,
levels=4, 16 nonces/node, so 128 nodes.

### Results (test accuracy)

| Representation | word_start | char_class |
|---|---:|---:|
| majority class | 50.0% | 66.5% |
| original embedding (2048-d) | 99.5% | 98.0% |
| PCA-32, before hashing | 98.4% | 95.0% |
| **HashMind hash_bits (ASIC-native)** | **95.3%** | **93.5%** |
| HashMind threshold, d=4 (ASIC-native, sparse) | 95.5% | 92.8% |
| HashMind hamming (sim only) | 95.6% | 93.0% |
| HashMind hash_bytes (sim only) | 94.5% | 91.5% |
| HashMind bucket (sim only) | 94.7% | 89.2% |
| HashMind, tuple_size=1 | 95.5% | 92.1% |
| HashMind, tuple_size=3 | 91.6% | 89.6% |
| HashMind, tuple_size=4 | 79.9% | 81.4% |
| **control: HashMind on shuffled embeddings** | **49.9%** | **64.1%** |

Neighbour preservation: overlap of the top-10 cosine neighbours with the
original embedding space was 20.8% for PCA-32, 6.1% for HashMind, and 0.2% for
chance.

### Cost (hash_bits, 6000 samples)

| Metric | Value |
|---|---|
| Feature dimensionality | 2048 |
| SHA-256d per sample (logical) | 2048 (128 nodes x 16 nonces) |
| SHA-256d total, logical / distinct | 12,288,000 / 32,768 |
| HashMind transform, CPU simulator | ~120 µs/sample (with per-call deduplication) |
| Readout inference | ~10 µs/sample |
| Peak memory, layer transform | ~50 MB (tracemalloc) |
| Process max RSS | ~1.6 GB (dominated by the dequantized 32000 x 2048 fp32 embedding) |
| Full experiment wall time | ~80 s on the build container |

Full tables: `docs/results/tinyllama_probe.md`, raw data in `.json`.

## Interpretation

**Facts**
- Label information in TinyLlama's embeddings survives the SHA-256d layer.
  HashMind loses 3 points on word_start and 1.5 on char_class compared with its
  own PCA-32 input. The shuffled control falls to the majority baseline, so the
  accuracy comes from the GGUF weights, not from the hash or the label
  distribution.
- The ASIC-native modes do as well as or better than the digest-based modes. So
  restricting the S9 to returning nonces costs nothing here.
- Tuple size is the main design trade-off. Larger tuples (more interaction per
  node) cut accuracy sharply: 3 gives 91.6% and 4 gives 79.9%. A cryptographic
  hash destroys locality inside a cell, so cells must stay small.
- Fine geometry is mostly lost: neighbour overlap is 6% against 21% for PCA.
  HashMind keeps coarse, linearly decodable properties, not the embedding's
  metric structure.

**Caveats and open questions**
1. **Lookup-table equivalence.** With tuple_size=2 and levels=4, each node has
   only 16 distinct inputs, so the whole layer is 32,768 distinct SHA-256d
   values. They could be precomputed once, and the S9 would then add nothing at
   inference. ASIC work only matters when the input space per node is large
   (bigger tuples, more levels, or recurrent/context bits). Those settings are
   exactly where accuracy currently drops. This is the central tension for
   phase 3.
2. tuple_size=1 matches tuple_size=2. These tasks may be close to linearly
   separable in PCA space, so they do not yet show that the hash
   nonlinearity adds anything beyond quantization.
3. The tasks are surface properties of tokens (spelling and segmentation), not
   semantics or next-token prediction.
4. The source is a 2-bit-quantized model, so these results are a lower bound on
   what the embedding contains.
5. The low-rank block factors are stored but unused; no transformer
   computation is reproduced.

## Reproduce

```bash
pip install -e .[dev]        # numpy; optional: pip install gguf (IQ* dequant + reference tests)
python -m hashmind weights    model.gguf
python -m hashmind convert    model.gguf -o model.hmmodel
python -m hashmind simulate   model.hmmodel --tokens 1,2,3
python -m hashmind experiment model.gguf -o docs/results/my_probe
```
