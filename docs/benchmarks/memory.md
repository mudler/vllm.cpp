# Memory

Qwen3.6-27B NVFP4, GB10, whole serving window.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |
| Weight offload, resident device bytes (`ENG-WEIGHT-OFFLOAD` W6) | not measured | not measured | n/a | **BLOCKED**, not pending: unmeasurable on every host we own (GB10 shares one pool, so `cpu_offload_gb` frees nothing). Needs a discrete-GPU rig ([record](../../.agents/benchmark-record.md)) |
| Disk residency via `--offload-config` (`ENG-RESIDENCY-CONFIG`, [#1110](https://github.com/mudler/vllm.cpp/issues/1110)) | not measured | n/a (no disk tier upstream) | n/a | **PENDING** a GB10 run. The row changes no kernel, dtype or allocation, so it claims no throughput axis; the 370 GiB reproduction through the JSON form is owed ([spec](../../.agents/specs/weight-residency-config.md)) |
| Decode-graph executables and device bytes, `VT_CUDA_GRAPH_DEDUP` (`ENG-CUDAGRAPH-DEDUP`, [#1162](https://github.com/mudler/vllm.cpp/issues/1162)) | COARSE key 3/7 and 5/11 execs; 15.40 vs 29.24 MiB nominal at 7 buckets | n/a | 0.43x / 0.45x execs; bytes NOT ESTABLISHED | **NEGATIVE, decided.** The fold engages; the saving fails its null control -- 0.42% of process at 7 buckets, none at 11. Default stays OFF. No time figure, clocks unpinned ([record](../../.agents/benchmark-record.md)) |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.
