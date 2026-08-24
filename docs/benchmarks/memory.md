# Memory

Qwen3.6-27B NVFP4, GB10, whole serving window — that is the subject of the first
four rows only. Every row below them names its own model, host and window,
because they do not share one.

| Axis | vllm.cpp | vLLM | Ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24.88 GiB | 28.18 GiB | 1.133x | **PASS** |
| Peak RSS | 24.88 GiB | 28.56 GiB | 1.148x | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720x | **PASS** |
| Peak `MemAvailable` drop | 68.35 GiB | 80.66 GiB | 1.180x | **PASS** |
| Weight offload, resident device bytes (`ENG-WEIGHT-OFFLOAD` W6) | not measured | not measured | n/a | **BLOCKED**, not pending: unmeasurable on every host we own (GB10 shares one pool, so `cpu_offload_gb` frees nothing). Needs a discrete-GPU rig ([record](../../.agents/benchmark-record.md)) |
| Disk residency via `--offload-config` (`ENG-RESIDENCY-CONFIG`, [#1110](https://github.com/mudler/vllm.cpp/issues/1110)) | not measured | n/a (no disk tier upstream) | n/a | **PENDING** a GB10 run. The row changes no kernel, dtype or allocation, so it claims no throughput axis; the 370 GiB reproduction through the JSON form is owed ([spec](../../.agents/specs/weight-residency-config.md)) |
| Decode-graph executables and device bytes, `VT_CUDA_GRAPH_DEDUP` (`ENG-CUDAGRAPH-DEDUP`, [#1162](https://github.com/mudler/vllm.cpp/issues/1162)) | COARSE key 3/7 and 5/11 execs; 15.40 vs 29.24 MiB nominal at 7 buckets | n/a | 0.43x / 0.45x execs; bytes NOT ESTABLISHED | **NEGATIVE, decided.** The fold engages; the saving fails its null control -- 0.42% of process at 7 buckets, none at 11. Default stays OFF. No time figure, clocks unpinned ([record](../../.agents/benchmark-record.md)) |
| Vision-tower skip, `--language-model-only` peak HOST RSS at load. **Qwen3-VL-4B-Instruct only**, `--device cpu`, `thor:gpu0` under an `rc` lease, `41ab550b9` (`ENG-MM-INPUT-PIPELINE` L3, [#607](https://github.com/mudler/vllm.cpp/issues/607), [#1358](https://github.com/mudler/vllm.cpp/issues/1358)) | 10,209,501,184 B default vs 8,553,709,568 B with the flag = **1,655,791,616 B freed (1.542 GiB)**; swapped pair 1,655,992,320 B, spread 200,704 B against a 192,512 B leg-to-leg repeat | n/a: this is our own flag A/B, not an upstream comparison. vLLM has the mirrored predicate but no measurement was taken on it | 0.997x of the 1,661,390,848 B tower predicted from the checkpoint headers | **MET on both pairs, FIRST HALF ONLY.** Threshold 1,495,251,763 B, declared before any number existed. Three caveats: (1) about half the saving is [#1359](https://github.com/mudler/vllm.cpp/issues/1359), a bf16→host-f32 widening — the tower is 0.774 GiB on disk, so fixing #1359 should roughly HALVE this, correctly; (2) LOAD-TIME peak RSS, the arms stop at `/health`, not a served request and not VRAM; (3) half 2 — the default arm within 2% of pre-L3 `edbc47ce0` — is a separate run and is NOT asserted, so it stays owed. `muse-glimmer-30b` remains unmeasured against its own 90%-of-7.161-GiB threshold, blocked on ~56 G of worker-local disk. Evidence: [report](../bench-evidence/tower-skip-rss-qwen3vl-thor-20260824.log), [legs](../bench-evidence/tower-skip-rss-qwen3vl-thor-20260824.legs.log); [spec](../../.agents/specs/multimodal-track.md) §1.5 L3 |

35B steady-serving PSS is 3.53 GiB against vLLM's 13.3 GiB after the routed-expert
host mirror is freed once the device Marlin resident is built.
