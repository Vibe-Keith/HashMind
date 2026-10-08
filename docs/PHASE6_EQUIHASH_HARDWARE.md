# Phase 6: what an Equihash ASIC actually exposes

**Conclusion: `PARTIAL_SOLUTION_ACCESS`**

- **What we get:** every share a stock Antminer Z15 / Z15 Pro submits carries the **full** (200, 9) solution: 512 indices × 21 bits = 1344 bytes.
- **The limit:** only *shares* are visible. These are solutions whose block-header hash meets a target, after firmware-side filtering. The great majority of the solutions the chips find never leave the machine.
- **Not verified:** no miner was available. Nothing here was measured on hardware.

Evidence is graded: **[spec]** protocol specification, **[log]** third-party stock-firmware logs, **[infer]** inference, **[model]** our assumption.

## 1. Algorithm and parameters

| | |
|---|---|
| Target miners | Bitmain Antminer Z15 (420 kSol/s) and Z15 Pro (840 kSol/s, ~2.6–2.8 kW; vendor listings disagree on power) |
| Chip | BM1746, 3 per Z15 hash board. A stock Z15 Pro log reports `ChipSetting_get_addr_ZEC_1746 detect 6 chips` **[log]** |
| Algorithm | Equihash as used by Zcash and Horizen. The logs say `bitmain_ZCASH_init` and `opt_algo 9, zec_1746` **[log]** |
| Parameters | (n, k) = (200, 9). Collision length 20 bits, list size 2^21, 512 indices of 21 bits. BLAKE2b personalization `ZcashPoW` ‖ le32(200) ‖ le32(9) **[spec]** |
| Software reference | `hashmind/equihash/reference.py`. It matches all 11 of Zcash's (96, 5) solver vectors and all 10 validator vectors, and rejects all 544 single-bit mutations (`tests/test_phase6.py`). (200, 9) solves verify and encode to 1344 bytes. |

## 2. Controller → miner (what we send)

Zcash Stratum, [ZIP 301](https://github.com/zcash/zips/blob/main/zips/zip-0301.rst) **[spec]**:

- **`mining.notify`** carries `JOB_ID, VERSION, PREVHASH, MERKLEROOT, RESERVED, TIME, BITS, CLEAN_JOBS`.
- **`mining.subscribe`** returns `NONCE_1`, the first part of the 32-byte nonce.
- **`mining.set_target`** sets the share target. The miner compares the block hash to it as a 256-bit integer.
- **The input channel.** The pool (us) chooses the whole 108-byte Equihash input *I* = version ‖ prevhash ‖ merkleroot ‖ reserved ‖ time ‖ bits. Of that, 96 bytes (prevhash, merkleroot, reserved) are not interpreted by the miner. They are the only way model data can enter the ASIC. The miner chooses `NONCE_2`.

## 3. Miner → controller (what comes back)

| Channel | What it carries | Grade |
|---|---|---|
| `mining.submit` | `WORKER, JOB_ID, TIME, NONCE_2, EQUIHASH_SOLUTION`. The solution is "encoded as in a block header (including the compactSize)": `fd4005` ‖ 1344 bytes. | **[spec]** |
| Share filter | A solution is submitted only if SHA256d(header ‖ solution) ≤ share target. | **[spec]** |
| On-device filter | The firmware sets a chip ticket mask (`set_ticket_mask_chain chainID0 ticket_mask = 0x00000010`, Z15). How it maps onto solution difficulty is not documented. | **[log]** |
| Controller re-checks results | The Z15 Pro log counts `Total Hardware errors 21455`. In cgminer, a hardware error is a returned result that fails the host's re-check, so the controller receives enough to verify a solution. | **[log] + [infer]** |
| Miner HTTP/RPC API | Ports 4028 (cgminer API) and 6060. The documented 6060 endpoints return only aggregate rates, status and serial number. No per-share data, nonces or solutions. | **[log]** (`z15_pro_6060_api_endpoints.md`) |
| Firmware logs | Startup, frequency tuning, error and summary counters. No solutions. | **[log]** |
| Controller ↔ hash-board protocol | Bitmain cgminer 4.9.0 fork (`Started cgminer 4.9.0`) with an FPGA (`FPGA version: B031`) between the controller and the chip chains. The driver is closed source; no open Z-series driver was found. | **[log]** |

Log sources: `z15_miner.log` and `z15pro_miner.log` in [hashsource/hashsource_antminer_zx](https://github.com/hashsource/hashsource_antminer_zx), a third-party firmware dump. It is unverified, and its provenance and uptime are unknown.

## 4. The ten questions

1. **Algorithm:** Equihash (Zcash variant, BLAKE2b). **[spec]/[log]**
2. **(n, k):** (200, 9). **[spec]/[log]**
3. **Miner receives:** the 108-byte input via `mining.notify` (96 bytes freely chosen by us), plus `NONCE_1` and a share target. **[spec]**
4. **Miner returns:** `mining.submit` with `NONCE_2` and the full solution, for shares only. **[spec]**
5. **Visibility:**
   - **Stratum:** full solutions, shares only.
   - **RPC/API:** aggregates only.
   - **Controller protocol:** closed, behind an FPGA.
   - **Logs:** counters only.
   - **Share submission:** = Stratum.
6. **Does the ASIC return the full solution?** Yes, for every submitted share. Consensus validation needs the full solution, so a pool could not accept a share without it. **[spec] + [infer]**
7. **All 512 indices?** Yes. 1344 bytes = 512 × 21 bits. **[spec]**
8. **Lower-level software-accessible fields?** None found. The FPGA/driver path is closed and the API is aggregate-only. **[log]**
9. **Nonce only, solution discarded internally?** No for shares. Yes, in effect, for every solution that fails the share target or the on-chip ticket filter. Those are discarded before the controller and are not software-visible.
10. **Can open-source software submit controlled jobs and collect results?** Yes, without modifying the miner: run our own Stratum server (ZIP 301), point the stock miner at it, choose `PREVHASH/MERKLEROOT/RESERVED` per job and parse `mining.submit`. `hashmind/equihash/protocol.py` implements and tests both sides. Replacing the miner software itself is not possible: no open Z15 driver exists, and modifying the hardware is out of scope.

## 5. How many solutions are visible?

| Scenario | Solutions/s | Grade |
|---|---:|---|
| Rated solver throughput, Z15 Pro | 840,000 | vendor nominal |
| Difficulty-1 shares in a stock Z15 Pro log (`Work Utility 457.28/min`) | ≈ 7.6 | **[log]** |
| Actual submissions in that log (26,220 over ≈ 1,200 min at the pool's difficulty) | ≈ 0.36 | **[log] + [infer]** |
| Hypothetically every solution reported, limited by 100 Mbit/s Ethernet (~2.8 kB per submit line) | ≈ 4,400 | **[model]** |

Whether stock firmware honours a pool target below difficulty 1 is **unknown**, and the ticket mask suggests an on-chip floor. The analysis uses ≈ 7.6 visible solutions per second as the default and reports the other cases. This is the same structure as the S9: the chip does an enormous amount of work, but the interface only shows rare, filtered events.

## 6. Can the returned solution be shaped?

- **Input dependence.** The solution indices are a pseudo-random function of (I, nonce) through BLAKE2b. Model data enters only through the 96 free header bytes.
- **No locality.** Measured on (96, 5): inputs differing only by small noise share no solution indices (Jaccard 0.000, chance ≈ 0.0002). The solution therefore carries no similarity structure of the input.
- **The collision search is not over our data.** The list the ASIC searches is BLAKE2b(I ‖ V ‖ i), generated inside the chip. Model vectors cannot be supplied as the list, so Equihash's collision search cannot be turned into a search over model data (for example, nearest-neighbour or LSH bucket matching).
- **Ordering skew.** Canonical ordering forces each subtree's first index to be its minimum. Index position 0 averages 0.03 of the range, not 0.5, so a raw index stream is non-uniform by position. At most 512 × 21 − 511 = 10,241 bits per solution are free.
- **Forcing chosen content.** Getting b chosen bits into a fixed place costs about 2^b solutions; measured on (96, 5), this matches 2^-b per solution.

| Chosen bits b | Z15 Pro, rated 840 kSol/s | Z15 Pro, ~7.6 visible solutions/s |
|---:|---:|---:|
| 16 | 0.08 s | ≈ 2.4 h |
| 32 | ≈ 1.4 h | ≈ 18 years |

Rejection shaping is exponential in the information wanted, so it is rejected as an encoding primitive.

## 7. Not done

- No Z15 or other Equihash miner was available. Nothing was run on hardware.
- The ticket-mask semantics and the minimum pool difficulty honoured by stock firmware are unknown. They would be the first thing to measure with a real unit and our own Stratum server.
- KS/kHeavyHash intermediate extraction is treated as closed, as instructed. Nothing here reopens it.
