# HashMind phase-2 probe

Model: `tinyllama.gguf` (llama), token_embd [32000, 2048] Q2_K
PCA-32 explained variance: 6.1%. HashMind: 2048 features, tuple_size=2, levels=4, nonces/node=16.

## Task `word_start`  (train 4500, test 1500, classes {'0': 2909, '1': 3091})

Majority-class accuracy: **50.0%**

| representation | dim | test acc | SHA-256d (logical / executed) | ASIC-native | transform µs/sample | readout µs/sample | peak MB |
|---|---:|---:|---:|:---:|---:|---:|---:|
| original embedding (2048d) | 2048 | 99.5% | - |  | - | 11.1 | - |
| PCA-32 (pre-hash) | 32 | 98.4% | - |  | - | 0.2 | - |
| HashMind[hash_bits, tuple=2] | 2048 | 95.3% | 12,288,000 / 32,768 | yes | 120 | 11.0 | 51 |
| HashMind[hash_bytes, tuple=2] | 2048 | 94.5% | 3,072,000 / 8,192 | no (sim) | 43 | 9.4 | 51 |
| HashMind[hamming, tuple=2] | 2048 | 95.6% | 12,288,000 / 32,768 | no (sim) | 158 | 8.8 | 50 |
| HashMind[bucket, tuple=2] | 2048 | 94.7% | 1,536,000 / 4,096 | no (sim) | 18 | 9.4 | 52 |
| HashMind[threshold, tuple=2] | 2048 | 95.5% | 12,288,000 / 32,768 | yes | 115 | 11.0 | 50 |
| HashMind[hash_bits, tuple=1] | 2048 | 95.5% | 12,288,000 / 8,192 | yes | 63 | 12.7 | 50 |
| HashMind[hash_bits, tuple=3] | 2048 | 91.6% | 12,288,000 / 131,072 | yes | 257 | 8.6 | 50 |
| HashMind[hash_bits, tuple=4] | 2048 | 79.9% | 12,288,000 / 524,288 | yes | 633 | 8.7 | 50 |
| HashMind[hash_bits] on SHUFFLED embeddings | 2048 | 49.9% | 12,288,000 / 32,768 | yes | 117 | 10.1 | 50 |

## Task `char_class`  (train 4500, test 1500, classes {'capitalized': 1431, 'digit': 4, 'lowercase': 3979, 'punct/other': 586})

Majority-class accuracy: **66.5%**

| representation | dim | test acc | SHA-256d (logical / executed) | ASIC-native | transform µs/sample | readout µs/sample | peak MB |
|---|---:|---:|---:|:---:|---:|---:|---:|
| original embedding (2048d) | 2048 | 98.0% | - |  | - | 9.4 | - |
| PCA-32 (pre-hash) | 32 | 95.0% | - |  | - | 0.3 | - |
| HashMind[hash_bits, tuple=2] | 2048 | 93.5% | 12,288,000 / 32,768 | yes | 125 | 8.9 | 50 |
| HashMind[hash_bytes, tuple=2] | 2048 | 91.5% | 3,072,000 / 8,192 | no (sim) | 40 | 11.8 | 51 |
| HashMind[hamming, tuple=2] | 2048 | 93.0% | 12,288,000 / 32,768 | no (sim) | 166 | 10.1 | 50 |
| HashMind[bucket, tuple=2] | 2048 | 89.2% | 1,536,000 / 4,096 | no (sim) | 21 | 9.8 | 52 |
| HashMind[threshold, tuple=2] | 2048 | 92.8% | 12,288,000 / 32,768 | yes | 98 | 10.9 | 50 |
| HashMind[hash_bits, tuple=1] | 2048 | 92.1% | 12,288,000 / 8,192 | yes | 63 | 12.0 | 50 |
| HashMind[hash_bits, tuple=3] | 2048 | 89.6% | 12,288,000 / 131,072 | yes | 224 | 10.0 | 50 |
| HashMind[hash_bits, tuple=4] | 2048 | 81.4% | 12,288,000 / 524,288 | yes | 674 | 10.1 | 50 |
| HashMind[hash_bits] on SHUFFLED embeddings | 2048 | 64.1% | 12,288,000 / 32,768 | yes | 114 | 10.2 | 50 |

## Neighbour preservation (top-10 cosine neighbours vs original embedding)

- chance: 0.2%
- PCA-32: 20.8%
- HashMind[hash_bits]: 6.1%

SHA-256d *logical* = what a cache-less device computes; *executed* = distinct (node, payload, nonce) evaluations. Executed << logical means the layer's input space is small enough to precompute as a lookup table.

Total time 78.5s, process max RSS 1584 MB.
