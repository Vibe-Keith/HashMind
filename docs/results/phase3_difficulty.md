| mode | difficulty bits | window | features | test acc | density | SHA-256d executed | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| hash_bits | 1 | 1 | 1024 | 94.5% | 0.502 | 16,384 | 0 |
| threshold | 1 | 2 | 1024 | 93.6% | 0.756 | 32,768 | 0 |
| threshold | 4 | 16 | 1024 | 94.8% | 0.643 | 262,144 | 0 |
| threshold | 8 | 256 | 1024 | 94.5% | 0.630 | 4,194,304 | 4 |
| threshold | 12 | 4096 | 1024 | 94.5% | 0.627 | 67,108,864 | 65 |
| threshold | 8 | 16 | 1024 | 92.5% | 0.061 | 262,144 | 0 |
| nonce_bits | 4 | 16 | 1024 | 95.1% | 0.388 | 65,536 | 0 |
| nonce_bits | 8 | 256 | 1024 | 94.4% | 0.357 | 602,112 | 1 |
| nonce_bits | 12 | 4096 | 1024 | 94.7% | 0.341 | 7,471,104 | 7 |

S9 cost model at the BM1387 floor (difficulty 1 = 32 zero bits), 2048 features/token:

| mode | features/window | p(share in window) | features/s | s/token | jobs/token |
|---|---:|---:|---:|---:|---:|
| threshold | 1 | 0.632 | 3,143 | 0.652 | 2048 |
| nonce_bits | 9 | 0.632 | 28,289 | 0.072 | 228 |
| nonce_bits | 17 | 0.632 | 53,435 | 0.038 | 120 |
| nonce_bits | 25 | 0.632 | 78,580 | 0.026 | 82 |
