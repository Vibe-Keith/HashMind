# HashMind Phase 3: locality, real hidden states, S9 difficulty floor

Status: experimental. Three questions from phase 2, answered with
measurements on TinyLlama-1.1B-Chat (IQ2_XXS GGUF). Every result is
reproducible with:

```bash
python examples/phase3_experiments.py model.gguf --cache hidden.npz
```

Raw tables: `docs/results/phase3_{nexttoken,locality,difficulty}.{md,json}`.

## Setup

- **Reference forward pass:** `hashmind/llm/llama.py` (numpy) and
  `hashmind/llm/tokenizer.py` (SentencePiece-style tokenizer driven by the GGUF
  vocab). Validation:
  - Token IDs match published Llama IDs ("Hello" = 15043, "world" = 3186).
  - Perplexity on a held-out sentence is 12.4.
  - Greedy output is coherent: "The capital of France is the second largest city in the country."
  - The test suite checks causality and RoPE invariants.
- **Corpus:** CPython's built-in pydoc help text (no download needed). 128
  sequences x 128 tokens = 16,384 tokens. Split by sequence: 96 for training,
  32 for test. The forward pass took 660 s on 4 CPU cores.
- **Task:** cut the network at layer L. HashMind plus a ridge readout predicts
  the final normalized hidden state, which goes through the **preserved LM head**.
  Blocks 0..L-1 still run on the host.
- **Metrics:**
  - Top-1 agreement with the original model's argmax ("teacher agree").
  - Top-1 and top-5 accuracy against the true next token.
  - Ridge alpha chosen on a validation split held out from the training data.
- **Baselines:** count-based unigram, bigram and trigram models fit on the same
  training tokens.

## 1. Large node input space with locality: not found

`log2 table` = log2 of the SHA-256d evaluations needed to precompute the whole
layer as a lookup table. Around 30 or more would make precomputation
impractical, so the ASIC would actually be needed.

| Configuration (cut at layer 11 unless noted) | log2 table | teacher agree | top-1 |
|---|---:|---:|---:|
| linear ridge on PCA-64(h) | - | 25.1% | 16.8% |
| HashMind t=2 l=4 (phase-2 default) | 15.0 | 25.3% | 18.3% |
| t=2 l=4, supervised quantizer | 15.0 | 25.2% | 17.4% |
| t=4 l=4, quantile | 19.0 | 15.2% | 11.0% |
| t=4 l=4, supervised | 19.0 | 21.8% | 15.0% |
| t=8 l=2, quantile | 19.0 | 17.6% | 12.4% |
| t=8 l=2, supervised | 19.0 | 21.9% | 15.4% |
| t=12 l=2, supervised | 23.0 | 19.2% | 13.7% |
| *embedding layer (L=0):* | | | |
| t=2 l=4, no context | 15.0 | 24.4% | 17.6% |
| t=1 + 1 context token ID | 28.0 | 25.0% | 17.9% |
| context only: 2 of [tok_t, tok_t-1] | 40.9 | 14.6% | 10.9% |
| context only: 2 of [tok_t, tok_t-1, tok_t-2] | 40.9 | 13.9% | 10.4% |
| PCA-64 + context, 2048 features | 40.9 | 20.4% | 14.4% |
| PCA-64 + context, 8192 features | 42.9 | 17.5% | 12.8% |
| **trigram counts** | - | 23.5% | **25.5%** |

**Findings**
- A supervised quantizer halves the damage from large tuples:
  - t=4 goes from 15.2% to 21.8%.
  - t=8 goes from 17.6% to 21.9%.
  - Both still fall short of t=2.
- Accuracy falls as the table grows. The best non-trivial point is t=1 with
  one context token (table 2^28), which ties the t=2 baseline. It gains
  nothing.
- Hashed token-ID context (the hashed n-gram idea) gives an intractably large
  table (2^41) but **loses badly to a plain trigram count model**: 10.4% vs
  25.5% top-1. With 12k training tokens and a 32k vocabulary, a ridge readout
  over 2048 random hash features cannot memorize an n-gram table. Going to 8192
  features overfits and does worse.
- Conclusion: on this data, every config where the S9 is needed (table above
  2^28) is worse than one that could be precomputed or computed directly on
  the host. A cryptographic hash has no partial similarity between inputs, so a
  hashed cell helps a linear readout only if the same cell shows up again at
  test time. Large input spaces make that rare.

**Not tried:** multi-resolution tiling (CMAC), recurrent state bits, much
larger training sets. The last one is the most likely to help the context
variants, because n-gram coverage grows with data.

## 2. Real hidden states and next-token prediction

All rows use 2048 HashMind features (t=2, l=4, 16 nonces per node).

| Cut layer L | linear PCA-64 | HashMind | **PCA-64 + HashMind** | teacher (all 22 blocks) |
|---:|---:|---:|---:|---:|
| 0 (embedding only) | 16.1 / 12.5 | 24.4 / 17.6 | 24.4 / 17.6 | 100 / 43.1 |
| 6 | 22.3 / 15.0 | 24.4 / 17.3 | 27.4 / 18.6 | |
| 11 | 25.1 / 16.8 | 25.3 / 18.3 | 29.0 / 19.3 | |
| 16 | 30.5 / 19.8 | 29.3 / 19.0 | 33.1 / 21.6 | |
| 22 (after last block) | 43.4 / 26.2 | 40.0 / 24.3 | 45.4 / 26.8 | |

Cells are teacher-agreement % / top-1 accuracy %. Linear ridge on the **full**
2048-d h_11 gives 41.6% / 26.5%.

**Findings**
- HashMind features carry real information beyond the linear features: PCA-64
  + HashMind beats PCA-64 alone at every cut, by 2 to 8 points of teacher
  agreement.
- The input bottleneck dominates. HashMind reads only the top 64 PCA
  components, and a plain linear readout on the full 2048-d hidden state (41.6%
  at L=11) beats every HashMind variant at that cut.
- Replacing the upper half of the transformer (L=11) with HashMind keeps 29%
  agreement with the original model and 19.3% top-1, against the teacher's
  43.1%. That is well short of the original model, and also below a trigram
  count model on top-1.
- At L=22 (after the last block, before the final norm) the readout is close to
  an identity map; that row is a sanity check, not a result.

## 3. BM1387 difficulty floor

**Facts**
- The BM1387 reports a nonce only if the hash meets its ticket mask.
  cgminer's gekko driver forces the mask to 0 for the BM1387, and its ticket
  table labels mask 0x00 as difficulty 1, "all nonces". Difficulty 1 means 32
  leading zero bits, so p = 2^-32 per hash.
  ([driver-gekko.c](https://raw.githubusercontent.com/kanoi/cgminer/master/driver-gekko.c))
- A community driver states that difficulty = ticket_mask + 1, and that each
  reported nonce represents difficulty x 2^32 hashes
  ([Opticell/bm138x](https://github.com/Opticell/bm138x)).
- **The threshold mode cannot run at low difficulty on real hardware.** Bitmain's
  official datasheet could not be fetched from this environment, so this rests
  on driver source, not the datasheet.

**Simulation.** The CPU cannot do 2^32 hashes per feature. Instead I scaled
difficulty d and window W together. Task: word_start, 3000 tokens, 1024
features.

| Mode | d | window | test acc | density |
|---|---:|---:|---:|---:|
| hash_bits (phase 2) | 1 | 1 | 94.5% | 0.50 |
| threshold | 4 | 16 | 94.8% | 0.64 |
| threshold | 8 | 256 | 94.5% | 0.63 |
| threshold | 12 | 4096 | 94.5% | 0.63 |
| threshold, sparse (λ = 1/16) | 8 | 16 | 92.5% | 0.06 |
| nonce_bits | 4 | 16 | 95.1% | 0.39 |
| nonce_bits | 8 | 256 | 94.4% | 0.36 |
| nonce_bits | 12 | 4096 | 94.7% | 0.34 |

Accuracy is flat in d (94.5 ± 0.3%) when the window scales as 2^d. Assuming
SHA-256d outputs are uniform, the d = 32 hardware setting should behave the
same. Sparse features (window too small for the difficulty) cost about 2
points.

**S9 cost model** (`hashmind/backends/s9_model.py`; 13.5 TH/s, 189 chips, 2048 features per token):

| Mode at d = 32 | features per header sweep | features/s | time per token |
|---|---:|---:|---:|
| threshold (any share in 2^32 nonces) | 1 | 3,143 | 0.65 s |
| nonce_bits, 8 bits | 9 | 28,289 | 72 ms |
| nonce_bits, 16 bits | 17 | 53,435 | 38 ms |
| nonce_bits, 24 bits | 25 | 78,580 | 26 ms |

`nonce_bits` is a new ASIC-native mode. The position of the first returned
nonce in the sweep is itself a deterministic, hash-derived random number, so one
share gives many feature bits. Not modeled: job dispatch latency, UART
bandwidth, missed or duplicate nonces, and whether every chip covers the nonce
range deterministically. All of these must be measured in the hardware phase.

## Overall assessment

1. The S9's minimum difficulty is workable: with windowed or nonce-bit features,
   accuracy matches the idealized p = 0.5 features. The cost is about 26 to 650
   ms per token.
2. On the tasks tested, HashMind adds a modest nonlinear boost on top of a
   linear readout. It does not come close to the transformer blocks it
   replaces.
3. Configurations that need the S9 (lookup tables too big to precompute) are
   the ones that perform worst. Until a design breaks that trade-off, a host
   lookup table or a cheap non-cryptographic hash would do the same job faster.
   This is the main open problem.
