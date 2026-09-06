# Variadic load report over 9 legs

## Realised prompt-length histogram, read back from each server's own usage.prompt_tokens

| prompt tokens | OURS | THEIRS |
|---|---|---|
| 0-127 | 266 | 133 |
| 128-255 | 260 | 130 |
| 256-511 | 48 | 24 |
| 512-1023 | 96 | 48 |
| 1024-2047 | 26 | 13 |
| 2048-4095 | 72 | 36 |

| band | n | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| `S` | 381 | 78 | 109 | 139 | 160 | 111 |
| `M` | 480 | 93 | 170 | 276 | 457 | 185 |
| `L` | 183 | 739 | 930 | 1127 | 1139 | 933 |
| `XL` | 108 | 2288 | 2962 | 3127 | 3153 | 2811 |

## Per leg: throughput and the spread between rounds

| arm | c | round | ok/n | out tok/s | decode-only tok/s | mean TTFT ms | p95 TTFT ms | wall s | counted by | publishable |
|---|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 128/128 | 35.85 | 59.05 | 1975.1 | 9471.7 | 649.2 | usage | yes |
| OURS | 1 | 2 | 128/128 | 35.77 | 59.02 | 1984.7 | 9474.5 | 650.6 | usage | yes |
| OURS | 4 | 1 | 128/128 | 53.18 | 16.58 | 2701.4 | 10273.8 | 438.8 | usage | yes |
| OURS | 4 | 2 | 128/128 | 53.16 | 16.83 | 2742.8 | 10543.1 | 438.8 | usage | yes |
| OURS | 8 | 1 | 128/128 | 53.83 | 7.86 | 3797.2 | 10837.9 | 432.2 | usage | yes |
| OURS | 8 | 2 | 128/128 | 54.03 | 7.92 | 3910.4 | 11346.3 | 431.4 | usage | yes |
| THEIRS | 1 | 1 | 128/128 | 33.10 | 44.96 | 1359.7 | 4641.4 | 673.8 | usage | yes |
| THEIRS | 4 | 1 | 128/128 | 33.33 | 44.31 | 16681.3 | 21807.4 | 668.2 | usage | yes |
| THEIRS | 8 | 1 | 128/128 | 33.52 | 44.31 | 36600.6 | 45418.1 | 665.5 | usage | yes |

## Round-to-round spread per cell

| arm | c | out tok/s per round | spread | p95 TTFT ms per round | spread |
|---|---|---|---|---|---|
| OURS | 1 | 35.85 / 35.77 | 0.2% | 9471.7 / 9474.5 | 0.0% |
| OURS | 4 | 53.18 / 53.16 | 0.0% | 10273.8 / 10543.1 | 2.6% |
| OURS | 8 | 53.83 / 54.03 | 0.4% | 10837.9 / 11346.3 | 4.7% |
| THEIRS | 1 | 33.10 | - | 4641.4 | - |
| THEIRS | 4 | 33.33 | - | 21807.4 | - |
| THEIRS | 8 | 33.52 | - | 45418.1 | - |

## Percentiles, pooled over both rounds (warm only, warmup discarded)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 256 | ttft ms | 802.2 | 3937.7 | 9798.6 | 10530.2 | 10715.3 | 1979.9 |
| OURS | 1 | 256 | ttft_corrected (est) ms | 802.2 | 3937.7 | 9798.6 | 10530.2 | 10715.3 | 1979.9 |
| OURS | 1 | 256 | tpot ms | 16.1 | 21.8 | 24.3 | 28.3 | 29.5 | 16.9 |
| OURS | 1 | 9781 | itl ms | 80.0 | 86.2 | 87.2 | 88.3 | 119.0 | 81.0 |
| OURS | 1 | 256 | e2el ms | 3944.3 | 8784.7 | 13097.9 | 14608.5 | 15116.0 | 5077.0 |
| OURS | 4 | 256 | ttft ms | 1226.6 | 8980.2 | 10434.1 | 13325.7 | 13523.3 | 2722.1 |
| OURS | 4 | 256 | ttft_corrected (est) ms | 1226.6 | 8980.2 | 10434.1 | 13325.7 | 13523.3 | 2722.1 |
| OURS | 4 | 256 | tpot ms | 50.6 | 97.7 | 113.5 | 159.9 | 175.4 | 59.9 |
| OURS | 4 | 9714 | itl ms | 170.2 | 174.9 | 523.5 | 3364.8 | 10550.5 | 288.5 |
| OURS | 4 | 256 | e2el ms | 11696.2 | 22845.9 | 23954.9 | 31680.3 | 36557.5 | 13670.3 |
| OURS | 8 | 256 | ttft ms | 1921.7 | 9530.0 | 11179.7 | 13711.9 | 13816.3 | 3853.8 |
| OURS | 8 | 256 | ttft_corrected (est) ms | 1921.7 | 9530.0 | 11179.7 | 13711.9 | 13816.3 | 3853.8 |
| OURS | 8 | 256 | tpot ms | 123.2 | 193.5 | 205.3 | 226.1 | 278.2 | 126.7 |
| OURS | 8 | 9711 | itl ms | 328.7 | 731.4 | 1177.3 | 8957.6 | 13258.3 | 605.8 |
| OURS | 8 | 256 | e2el ms | 25732.3 | 40348.0 | 45486.2 | 49377.7 | 54770.1 | 26837.7 |
| THEIRS | 1 | 128 | ttft ms | 738.6 | 2900.9 | 4641.4 | 5449.9 | 5782.2 | 1359.7 |
| THEIRS | 1 | 128 | ttft_corrected (est) ms | 738.6 | 2900.9 | 4641.4 | 5449.9 | 5782.2 | 1356.5 |
| THEIRS | 1 | 128 | tpot ms | 21.3 | 29.2 | 32.2 | 35.1 | 36.2 | 22.2 |
| THEIRS | 1 | 20479 | itl ms | 0.0 | 117.9 | 123.1 | 129.1 | 257.7 | 24.4 |
| THEIRS | 1 | 128 | e2el ms | 4655.2 | 8606.8 | 9452.2 | 10343.2 | 11586.8 | 5264.2 |
| THEIRS | 4 | 128 | ttft ms | 16583.0 | 20341.0 | 21807.4 | 26089.1 | 28039.2 | 16681.3 |
| THEIRS | 4 | 128 | ttft_corrected (est) ms | 16574.2 | 20334.3 | 21807.4 | 26089.1 | 28039.2 | 16678.3 |
| THEIRS | 4 | 128 | tpot ms | 22.2 | 30.1 | 33.1 | 35.9 | 38.2 | 22.6 |
| THEIRS | 4 | 20507 | itl ms | 0.0 | 118.0 | 122.8 | 129.7 | 253.9 | 24.7 |
| THEIRS | 4 | 128 | e2el ms | 20465.9 | 24869.9 | 26583.8 | 30864.4 | 33092.2 | 20645.0 |
| THEIRS | 8 | 128 | ttft ms | 36498.1 | 44658.5 | 45418.1 | 48067.2 | 48160.9 | 36600.6 |
| THEIRS | 8 | 128 | ttft_corrected (est) ms | 36498.1 | 44658.5 | 45407.2 | 48054.8 | 48160.9 | 36597.4 |
| THEIRS | 8 | 128 | tpot ms | 22.0 | 30.3 | 32.0 | 33.1 | 37.2 | 22.6 |
| THEIRS | 8 | 20528 | itl ms | 0.0 | 117.6 | 123.6 | 129.7 | 268.3 | 24.7 |
| THEIRS | 8 | 128 | e2el ms | 40480.2 | 48687.1 | 50419.9 | 51358.9 | 52890.3 | 40560.1 |

## Percentiles, pooled over both rounds (with warmup included)

Pooled n per cell is both rounds together, so p99 and max are read off that pool and not off a single leg.

| arm | c | n | axis | p50 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| OURS | 1 | 264 | ttft ms | 801.7 | 3943.2 | 9803.2 | 10552.3 | 27106.0 | 2058.3 |
| OURS | 1 | 264 | ttft_corrected (est) ms | 801.7 | 3943.2 | 9803.2 | 10552.3 | 27106.0 | 2058.3 |
| OURS | 1 | 264 | tpot ms | 16.1 | 21.9 | 24.5 | 28.3 | 29.5 | 17.0 |
| OURS | 1 | 10109 | itl ms | 80.0 | 86.1 | 87.1 | 88.3 | 119.0 | 81.0 |
| OURS | 1 | 264 | e2el ms | 3944.3 | 8788.4 | 13153.5 | 14783.4 | 30434.5 | 5162.5 |
| OURS | 4 | 264 | ttft ms | 1303.9 | 8767.3 | 10416.7 | 13324.1 | 13523.3 | 2777.1 |
| OURS | 4 | 264 | ttft_corrected (est) ms | 1303.9 | 8767.3 | 10416.7 | 13324.1 | 13523.3 | 2777.1 |
| OURS | 4 | 264 | tpot ms | 49.9 | 97.4 | 113.1 | 157.8 | 175.4 | 59.1 |
| OURS | 4 | 10038 | itl ms | 170.2 | 174.8 | 521.5 | 3303.1 | 10550.5 | 284.1 |
| OURS | 4 | 264 | e2el ms | 11638.6 | 22821.4 | 23946.1 | 31661.1 | 36557.5 | 13582.1 |
| OURS | 8 | 272 | ttft ms | 2074.8 | 10783.1 | 11928.5 | 20681.2 | 20684.2 | 4446.2 |
| OURS | 8 | 272 | ttft_corrected (est) ms | 2074.8 | 10783.1 | 11928.5 | 20681.2 | 20684.2 | 4446.2 |
| OURS | 8 | 272 | tpot ms | 119.6 | 190.7 | 204.8 | 226.1 | 278.2 | 122.9 |
| OURS | 8 | 10340 | itl ms | 328.5 | 723.8 | 1074.5 | 8901.8 | 13258.3 | 586.2 |
| OURS | 8 | 272 | e2el ms | 25732.3 | 39650.7 | 45195.4 | 49349.9 | 54770.1 | 26735.2 |
| THEIRS | 1 | 132 | ttft ms | 740.7 | 3696.6 | 4874.2 | 5682.5 | 7839.8 | 1422.7 |
| THEIRS | 1 | 132 | ttft_corrected (est) ms | 740.6 | 3695.0 | 4874.2 | 5682.5 | 7839.8 | 1419.4 |
| THEIRS | 1 | 132 | tpot ms | 21.5 | 29.8 | 32.1 | 35.1 | 36.2 | 22.3 |
| THEIRS | 1 | 21124 | itl ms | 0.0 | 118.0 | 123.2 | 129.1 | 760.1 | 24.5 |
| THEIRS | 1 | 132 | e2el ms | 4688.3 | 8795.3 | 9615.4 | 11231.5 | 12918.3 | 5342.1 |
| THEIRS | 4 | 132 | ttft ms | 16490.7 | 20222.2 | 21737.5 | 26018.2 | 28039.2 | 16415.7 |
| THEIRS | 4 | 132 | ttft_corrected (est) ms | 16490.5 | 20213.6 | 21737.5 | 26018.2 | 28039.2 | 16412.5 |
| THEIRS | 4 | 132 | tpot ms | 22.3 | 30.0 | 32.8 | 35.8 | 38.2 | 22.6 |
| THEIRS | 4 | 21141 | itl ms | 0.0 | 118.0 | 122.8 | 129.6 | 253.9 | 24.7 |
| THEIRS | 4 | 132 | e2el ms | 20249.8 | 24817.5 | 26515.2 | 30851.5 | 33092.2 | 20378.6 |
| THEIRS | 8 | 136 | ttft ms | 36136.0 | 44525.4 | 45406.2 | 48059.8 | 48160.9 | 35387.9 |
| THEIRS | 8 | 136 | ttft_corrected (est) ms | 36136.0 | 44525.4 | 45392.2 | 48047.2 | 48160.9 | 35384.5 |
| THEIRS | 8 | 136 | tpot ms | 22.0 | 29.8 | 31.9 | 33.1 | 37.2 | 22.5 |
| THEIRS | 8 | 21802 | itl ms | 0.0 | 117.5 | 123.6 | 129.6 | 268.3 | 24.6 |
| THEIRS | 8 | 136 | e2el ms | 40059.5 | 48589.7 | 50375.0 | 51350.1 | 52890.3 | 39332.5 |

## Streaming granularity, and what it does to TTFT

`first chunk tokens` is an ESTIMATE: streaming carries no per-chunk token count, so it is the first chunk's characters divided by that request's own characters-per-token.

| arm | c | tokens per chunk | chars per token | first chunk tokens (est) | p50 TTFT raw ms | p50 TTFT corrected ms (est) | corrections refused |
|---|---|---|---|---|---|---|---|
| OURS | 1 | 4.87 | 3.60 | 0.58 | 802.2 | 802.2 | 0/256 |
| OURS | 4 | 4.89 | 3.59 | 0.58 | 1226.6 | 1226.6 | 0/256 |
| OURS | 8 | 4.88 | 3.61 | 0.58 | 1921.7 | 1921.7 | 0/256 |
| THEIRS | 1 | 1.09 | 3.56 | 1.02 | 738.6 | 738.6 | 0/128 |
| THEIRS | 4 | 1.09 | 3.58 | 1.02 | 16583.0 | 16574.2 | 0/128 |
| THEIRS | 8 | 1.09 | 3.59 | 1.01 | 36498.1 | 36498.1 | 0/128 |

## Acceptance

| arm | c | round | source | value |
|---|---|---|---|---|
| OURS | 1 | 1 | /metrics | accepted/proposed = 18522/34377 = 0.539, 0.796 per output token |
| OURS | 1 | 2 | /metrics | accepted/proposed = 18522/34377 = 0.539, 0.796 per output token |
| OURS | 4 | 1 | /metrics | accepted/proposed = 18628/34055 = 0.547, 0.798 per output token |
| OURS | 4 | 2 | /metrics | accepted/proposed = 18635/33964 = 0.549, 0.799 per output token |
| OURS | 8 | 1 | /metrics | accepted/proposed = 18562/33957 = 0.547, 0.798 per output token |
| OURS | 8 | 2 | /metrics | accepted/proposed = 18599/34048 = 0.546, 0.798 per output token |
| THEIRS | 1 | 1 | usage | accepted 17346, 0.778 per output token (this engine exports no proposed count) |
| THEIRS | 4 | 1 | usage | accepted 17250, 0.774 per output token (this engine exports no proposed count) |
| THEIRS | 8 | 1 | usage | accepted 17288, 0.775 per output token (this engine exports no proposed count) |

## Cold start: what the first request costs

| arm | c | round | warmup TTFT ms, first request | warm p50 TTFT ms | mean TTFT ms warm | mean TTFT ms with warmup |
|---|---|---|---|---|---|---|
| OURS | 1 | 1 | 27106.0 | 801.7 | 1975.1 | 2153.6 |
| OURS | 1 | 2 | 791.2 | 803.1 | 1984.7 | 1963.0 |
| OURS | 4 | 1 | 4572.3 | 1266.9 | 2701.4 | 2756.8 |
| OURS | 4 | 2 | 4421.9 | 1212.2 | 2742.8 | 2797.5 |
| OURS | 8 | 1 | 7252.7 | 1935.1 | 3797.2 | 3997.9 |
| OURS | 8 | 2 | 20684.2 | 1921.7 | 3910.4 | 4894.5 |
| THEIRS | 1 | 1 | 7839.8 | 738.6 | 1359.7 | 1422.7 |
| THEIRS | 4 | 1 | 973.1 | 16583.0 | 16681.3 | 16415.7 |
| THEIRS | 8 | 1 | 25726.4 | 36498.1 | 36600.6 | 35387.9 |
