| configuration | cut L | dim | teacher agree | top-1 | top-5 | SHA-256d logical / executed | log2 table | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher: original model, all blocks | - | - | 100.0% | 43.1% | 67.0% | - | - | 0 |
| unigram (most frequent token) | - | - | 10.9% | 9.7% | 19.8% | - | - | 0 |
| bigram counts (backoff unigram) | - | - | 22.5% | 20.8% | 39.1% | - | - | 0 |
| trigram counts (backoff bigram) | - | - | 23.5% | 25.5% | 39.6% | - | - | 0 |
| linear ridge on PCA-64(h_11) | 11 | 64 | 25.1% | 16.8% | 35.4% | - | - | 10 |
| linear ridge on full h_11 | 11 | 2048 | 41.6% | 26.5% | 49.5% | - | - | 18 |
| HashMind t=2 l=4 quantile (h_11) | 11 | 2048 | 25.3% | 18.3% | 36.6% | 33,292,288 / 65,536 | 15.0 | 17 |
| HashMind t=2 l=4 supervised (h_11) | 11 | 2048 | 25.2% | 17.4% | 35.8% | 33,292,288 / 65,184 | 15.0 | 22 |
| HashMind t=4 l=4 quantile (h_11) | 11 | 2048 | 15.2% | 11.0% | 24.7% | 33,292,288 / 1,048,512 | 19.0 | 24 |
| HashMind t=4 l=4 supervised (h_11) | 11 | 2048 | 21.8% | 15.0% | 32.5% | 33,292,288 / 771,392 | 19.0 | 26 |
| HashMind t=8 l=2 quantile (h_11) | 11 | 2048 | 17.6% | 12.4% | 29.0% | 33,292,288 / 1,048,560 | 19.0 | 26 |
| HashMind t=8 l=2 supervised (h_11) | 11 | 2048 | 21.9% | 15.4% | 32.8% | 33,292,288 / 624,704 | 19.0 | 33 |
| HashMind t=12 l=2 supervised (h_11) | 11 | 2048 | 19.2% | 13.7% | 30.2% | 33,292,288 / 2,901,728 | 23.0 | 30 |
| linear ridge on PCA-64(h_0) | 0 | 64 | 16.1% | 12.5% | 27.8% | - | - | 11 |
| HashMind t=2 l=4, no context (h_0) | 0 | 2048 | 24.4% | 17.6% | 35.3% | 33,292,288 / 65,440 | 15.0 | 18 |
| HashMind t=1 l=4 + 1 ctx symbol of [tok_t, tok_t-1] (h_0) | 0 | 2048 | 25.0% | 17.9% | 35.9% | 33,292,288 / 7,124,960 | 28.0 | 40 |
| HashMind ctx only: 2 symbols of [tok_t, tok_t-1] (h_0) | 0 | 2048 | 14.6% | 10.9% | 24.6% | 33,292,288 / 18,028,544 | 40.9 | 47 |
| HashMind ctx only: 2 of [tok_t, tok_t-1, tok_t-2] (h_0) | 0 | 2048 | 13.9% | 10.4% | 22.9% | 33,292,288 / 19,331,952 | 40.9 | 50 |
| HashMind PCA-64 + ctx 2 of [tok_t, tok_t-1, tok_t-2] (h_0) | 0 | 2112 | 20.4% | 14.4% | 31.2% | 33,292,288 / 19,331,952 | 40.9 | 49 |
| HashMind PCA-64 + ctx 2 of 3, 8192 features (h_0) | 0 | 8256 | 17.5% | 12.8% | 24.9% | 133,169,152 / 76,837,424 | 42.9 | 192 |
