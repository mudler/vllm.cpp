# Variadic load report over 4 legs

## Realised prompt-length histogram, read back from each server's own usage.prompt_tokens

| prompt tokens | OURS | THEIRS |
|---|---|---|
| 0-127 | 40 | 40 |
| 128-255 | 234 | 234 |
| 256-511 | 52 | 52 |

| band | n | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| `M` | 652 | 93 | 175 | 286 | 457 | 192 |

## Per leg: throughput and the spread between rounds

| arm | c | round | ok/n | out tok/s | decode-only tok/s | mean TTFT ms | p95 TTFT ms | wall s | counted by |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 163/163 | 39.19 | 53.70 | 900.9 | 1370.5 | 532.3 | usage |
| OURS | 1 | 2 | 163/163 | 39.06 | 53.59 | 907.1 | 1364.8 | 534.2 | usage |
| THEIRS | 1 | 1 | 163/163 | 32.95 | 44.70 | 979.4 | 2053.0 | 593.6 | usage |
| THEIRS | 1 | 2 | 163/163 | 34.07 | 45.04 | 880.2 | 1796.7 | 574.1 | usage |

## Round-to-round spread per cell

| arm | c | out tok/s per round | spread | p95 TTFT ms per round | spread |
|---|---|---|---|---|---|
| OURS | 1 | 39.19 / 39.06 | 0.3% | 1370.5 / 1364.8 | 0.4% |
| THEIRS | 1 | 32.95 / 34.07 | 3.4% | 2053.0 / 1796.7 | 14.3% |

## Percentiles, pooled over both rounds (warm only, warmup discarded)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 326 | ttft ms | 863.1 | 1219.0 | 1371.6 | 1535.8 | 1818.8 | 904.0 |
| OURS | 1 | 326 | tpot ms | 18.8 | 23.4 | 24.7 | 27.5 | 29.3 | 18.6 |
| OURS | 1 | 9381 | itl ms | 82.2 | 83.1 | 83.4 | 84.0 | 86.0 | 82.3 |
| OURS | 1 | 326 | e2el ms | 3275.6 | 4063.7 | 4170.1 | 4766.3 | 5044.7 | 3271.3 |
| THEIRS | 1 | 326 | ttft ms | 762.7 | 1739.2 | 1959.9 | 2850.8 | 3946.9 | 929.8 |
| THEIRS | 1 | 326 | tpot ms | 22.1 | 29.8 | 31.6 | 34.2 | 36.2 | 22.3 |
| THEIRS | 1 | 36004 | itl ms | 0.0 | 116.9 | 122.5 | 128.1 | 258.3 | 24.0 |
| THEIRS | 1 | 326 | e2el ms | 3418.7 | 4756.6 | 5280.7 | 6062.0 | 7270.5 | 3581.7 |

## Percentiles, pooled over both rounds (with warmup included)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 328 | ttft ms | 863.2 | 1224.9 | 1377.5 | 1697.1 | 27455.6 | 1059.0 |
| OURS | 1 | 328 | tpot ms | 18.8 | 23.4 | 24.7 | 27.5 | 29.3 | 18.6 |
| OURS | 1 | 9441 | itl ms | 82.2 | 83.1 | 83.4 | 84.0 | 101.2 | 82.3 |
| OURS | 1 | 328 | e2el ms | 3279.5 | 4065.9 | 4186.2 | 4942.3 | 29957.8 | 3427.1 |
| THEIRS | 1 | 328 | ttft ms | 763.5 | 1771.1 | 1964.2 | 2964.4 | 7919.7 | 954.2 |
| THEIRS | 1 | 328 | tpot ms | 22.2 | 29.7 | 31.6 | 34.2 | 36.2 | 22.3 |
| THEIRS | 1 | 36222 | itl ms | 0.0 | 116.9 | 122.5 | 128.2 | 832.0 | 24.0 |
| THEIRS | 1 | 328 | e2el ms | 3426.1 | 4778.2 | 5302.8 | 6251.5 | 11302.2 | 3609.4 |

## Streaming granularity, and what it does to TTFT

`first chunk tokens` is an ESTIMATE: streaming carries no per-chunk token count, so it is the first chunk's characters divided by that request's own characters-per-token.

| arm | c | tokens per chunk | chars per token | first chunk tokens (est) | p50 TTFT raw ms | p50 TTFT corrected ms (est) |
|---|---|---|---|---|---|---|
| OURS | 1 | 4.47 | - | - | 863.1 | - |
| THEIRS | 1 | 1.08 | - | - | 762.7 | - |

## Acceptance

| arm | c | round | source | value |
|---|---|---|---|---|
| OURS | 1 | 1 | absent | NOT EXPOSED BY THIS ENGINE |
| OURS | 1 | 2 | absent | NOT EXPOSED BY THIS ENGINE |
| THEIRS | 1 | 1 | usage | accepted 15148, 0.774 per output token |
| THEIRS | 1 | 2 | usage | accepted 15162, 0.775 per output token |

## Cold start: what the first request costs

| arm | c | round | warmup TTFT ms, first request | warm p50 TTFT ms | mean TTFT ms warm | mean TTFT ms with warmup |
|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 27455.6 | 863.1 | 900.9 | 1062.8 |
| OURS | 1 | 2 | 25178.6 | 863.0 | 907.1 | 1055.1 |
| THEIRS | 1 | 1 | 7919.7 | 765.4 | 979.4 | 1021.7 |
| THEIRS | 1 | 2 | 1934.7 | 759.1 | 880.2 | 886.6 |
