# W4 spike: bounded safetensors source-page residency

**Row:** `ENG-HOST-WEIGHT-RESIDENCY`  
**Lifecycle:** `ACTIVE`
**Owner:** `CLAIM-HOST-WEIGHT-RESIDENCY-1`  
**Baseline:** project `6c30657`, evidence
`/tmp/qwen35-host-residency-6c30657`

## Problem and scope

The local Qwen3.5-4B launch peak is 15.55 GiB PSS versus 6.69 GiB for fresh
vLLM. `LoadShards` opens and mmaps every checkpoint shard before model loading,
and `LoadQwen3_5Dense` copies globals followed by every layer into owned host
vectors while those source mappings remain resident. The observed peak is
therefore the complete 8.585-GiB owned host model plus most touched source
pages. W1-W3 release the owned model only after device preparation and cannot
affect this earlier interval.

W4 first bounds source-page residency without invalidating the current random
tensor resolver. After globals and after each complete dense layer, the loader
will issue `madvise(MADV_DONTNEED)` for every read-only private shard mapping.
The virtual mappings and `StTensor` pointers remain valid; a later access can
fault the file page back in. `VT_SAFETENSORS_DISCARD_PAGES=0` restores the
current behavior from the same binary. An unavailable or failed advice call is
a fail-open optimization miss, not a load failure.

In scope:

- Linux read-only private safetensors mappings;
- the Qwen3.5 dense safetensors loader's global and per-layer boundaries;
- exact reload-after-discard behavior and launch PSS/load-time measurement;
- deciding from measurement whether direct-to-device loading is still needed.

Out of scope:

- unmapping shards or invalidating tensor pointers during loading;
- changing tensor order, dtype conversion, model math, CUDA kernels, or
  sampling;
- GGUF, quantized 27B/35B, MoE/expert streaming, CPU offload, or UMA policy;
- claiming Spark/GB10 results from the local RTX 5070 Ti.

## Whole execution chain

### vLLM oracle

- `.venv-vllm/lib/python3.12/site-packages/vllm/model_executor/model_loader/
  default_loader.py:266-298` selects the ordinary safetensors iterator and
  `:414-429` passes that stream directly to `model.load_weights`.
- `.venv-vllm/lib/python3.12/site-packages/vllm/model_executor/model_loader/
  weight_utils.py:820-954` sorts shards, opens one shard in a scoped
  `safe_open`, and yields its tensors before closing it. The default local-file
  path does not prefetch the whole checkpoint.
- This is a loading/lifetime difference, not a generated-kernel selection
  difference. Process PSS sampled over launch is the execution ground truth;
  an nsys kernel trace cannot establish host page residency.

### vllm.cpp

- `src/vllm/model_executor/model_loader/safetensors_reader.cpp:43-59` maps an
  entire shard `MAP_PRIVATE` and stores that mapping in `SafetensorsFile`.
- `src/vllm/entrypoints/model_loader.cpp:42-56` opens all shards before model
  construction and retains the vector through loading.
- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:394-430` builds an
  all-shard name index, copies globals, then appends every fully materialized
  layer. No consumed source pages are discarded.
- `src/vllm/entrypoints/model_loader.cpp:224-231` clears shard mappings only
  after the complete model has been constructed.

## Port and test map

| Work | Implementation anchor | Tests/evidence |
|---|---|---|
| Reader advice API | `include/vllm/model_executor/model_loader/safetensors_reader.h`; `src/vllm/model_executor/model_loader/safetensors_reader.cpp` | `tests/vllm/test_safetensors.cpp`: touch, discard, exact reread; empty moved-from object |
| Dense checkpoints and opt-out | `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp` | existing dense loader/forward/engine tests in both default and opt-out modes |
| Peak and regression gate | unchanged benchmark harness | three interleaved reps each for W4 ON, W4 OFF and fresh vLLM under one `/tmp/gpu` lock |

Upstream's iterator is the behavioral specification but has no direct
`MADV_DONTNEED` test to port. The local reader test pins the C++ deviation that
preserves pointer validity while approximating its bounded shard lifetime.

## Gates

1. CPU and native-sm_120 builds are warning-clean; full CPU CTest and focused
   safetensors/dense loader/forward/engine tests pass.
2. Default and `VT_SAFETENSORS_DISCARD_PAGES=0` produce byte-identical project
   outputs on the exact 128-request greedy benchmark and the natural parity
   corpus. Existing cross-engine parity debt may not worsen.
3. Three lock-held memory repetitions use the exact `6c30657` workload and a
   fresh vLLM denominator. Record launch peak PSS, stable PSS, process VRAM,
   load time and full process lifetime; monitored throughput remains void.
4. Unmonitored three-repetition same-binary A/B shows no throughput regression.
   Every accepted comparison records total/output throughput and vLLM ratio.
5. Stable PSS and VRAM may not regress. Peak PSS must reproducibly improve;
   `ENG-HOST-WEIGHT-RESIDENCY` stays `ACTIVE` while any memory axis exceeds
   vLLM. If the owned-host-model floor remains above vLLM, W4 proceeds to a
   separately recorded direct-to-device/bounded materialization phase.
6. All processes exit, GPU memory returns to idle, and the read-only kernel
   journal has no adjacent Xid, UVM/AER fault, reset, or lockup.

## Work breakdown

1. `W4.1`: add the fail-open reader advice API and exact reread tests.
2. `W4.2`: add default-on dense-loader checkpoints and the environment opt-out.
3. `W4.3`: run correctness, memory, load-time and throughput gates against a
   fresh vLLM denominator.
4. `W4.4`: if W4.3 remains above vLLM, spike and implement layer-bounded
   direct-device materialization; do not describe W4.1-W4.3 as peak parity.

Implementation checkpoint: W4.1-W4.2 are complete. CPU CTest passes 105/105;
native-sm_120 focused tests pass 5/5 with discard enabled and 2/2 with
`VT_SAFETENSORS_DISCARD_PAGES=0`. W4.3 memory/load-time/throughput measurement
is `PENDING`, so no improved number or peak-parity claim is accepted.

## Risks and mitigations

- Advice can increase file refaults and load time. Checkpoints are layer-sized,
  not tensor-sized, and load time is measured explicitly.
- Kernel readahead may keep nearby pages resident. PSS measurement decides the
  result; no reduction is inferred from the API call alone.
- Advice failure or a platform without `MADV_DONTNEED` preserves prior
  correctness and residency behavior.
- Mapped pointers must remain valid. W4 does not unmap, resize, write, or close
  the backing mapping before the existing loader lifetime ends.
- Re-reading a tensor after a checkpoint remains correct by file-backed page
  fault. The unit test and model opt-out A/B pin this property.
