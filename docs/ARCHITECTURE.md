# HashCortex-S9 Architecture (v1, phase 1)

## Hard constraint

A BM1387 does one thing: double SHA-256 over an 80-byte block header, iterating
the 4-byte nonce, and reports nonces whose hash meets a share target. No
multiply, no float, no memory, no digests returned. Everything else runs on the
host.

## Data path per token

```
token id
  -> [host] e = E[token]                        PRESERVED   token_embd.weight
  -> [host] z = P^T rmsnorm(e) - mu             TRANSFORMED P = top-k SVD of FFN up/gate + attn V (first L layers)
  -> [host] s = (z > 0)                         k sign bits
  -> [host] u = s ++ r                          r = c-bit binary reservoir (replaces attention)
  -> [host] n_tuples random tuples of b bits from u
  -> [ASIC] per tuple: header = "HCX1" | challenge_j | tuple bits | j | difficulty
            nonces 0..R-1, report those with top `difficulty` bits zero
  -> [host] f in {0,1}^(n_tuples*R); r <- leaky_update(r, f)
  -> [host] h = e + W_r^T (2f-1) + b            NEW readout, ridge-fit on host
  -> [host] logits = W_out rmsnorm(h)           PRESERVED   output.weight (+ output_norm)
```

## Why n-tuples, not "hash the whole vector"

SHA-256 is an avalanche function: flipping one input bit decorrelates the
output. Hashing the full input would give a readout that can only memorize.
Hashing small b-bit tuples makes each feature a random boolean function of a
few bits (WiSARD / n-tuple RAM network, a binary random-feature / ELM layer).
Inputs that share most sign bits share most features, so the readout can
generalize. The reservoir gives an echo-state-like memory of past tokens.

## Ideas this draws on

- Extreme Learning Machines / random-feature networks: fixed random nonlinear
  layer, only the linear readout is trained.
- Reservoir computing: fixed recurrent state, trained readout.
- n-tuple / WiSARD weightless networks: random lookup on bit tuples.

## What is preserved (honestly)

| Part | Fate |
|---|---|
| Token embedding | Kept verbatim (stored fp16) |
| LM head, final norm | Kept verbatim |
| FFN/attn-V first layers | Only their top-k input subspace survives, as P |
| Attention Q/K, FFN down, deeper layers | Discarded |
| Readout W_r | New; zeros after conversion, must be trained (distillation) |

The converted model is **not** equivalent to the source. Before readout
training its output is a bigram-like function of embedding + LM head.

## Open questions for phase 2

1. BM1387 minimum reportable difficulty (ticket mask). `difficulty_bits=1`
   (p = 0.5 features) is idealized; if the floor is high, features become
   sparse and n-tuples per token must rise. `SimulatedS9Backend(min_difficulty_bits=...)`
   models this.
2. Job throughput is bounded by host -> chip job rate and nonce-report
   bandwidth, not by the ~14 TH/s hash rate. Needs measurement.
3. Midstate: the driver sends SHA-256 state after the first 64 header bytes;
   tuple bits currently sit inside those 64 bytes, so every job needs a fresh
   midstate on the host. Moving payload into the last 12 bytes would allow
   midstate reuse but limits payload to 8 bytes.
4. Readout training data: hidden states from the original model via llama.cpp.
