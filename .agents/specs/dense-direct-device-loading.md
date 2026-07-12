# W4.4 spike: layer-bounded direct-device dense loading

**Row:** `ENG-HOST-WEIGHT-RESIDENCY`  
**Lifecycle:** `ACTIVE`
**Owner:** `CLAIM-HOST-WEIGHT-RESIDENCY-1`  
**Dependency:** W4.3 consumed-range checkpoint `a077d72`

## Problem and scope

Consumed-range advice reduces local Qwen3.5-4B launch peak PSS from 15.562 to
8.168 GiB, but fresh vLLM averages 7.695 GiB. The remaining project floor is the
complete owned host model: device upload and host release currently occur only
after `ModelRegistry::Load` returns every layer. vLLM instead constructs
parameters on the target device and streams safetensors into them.

W4.4 preselects the same dense-runner queue before safetensors construction,
passes a non-owning pointer through `ModelSource`, and after each completed
plain-BF16 layer reuses `Qwen3_5DenseModel::PrepareBf16Resident`, synchronizes,
and calls `ReleaseResidentQwen3_5DenseHostWeights`. The first checkpoint occurs
after layer 0 so eligibility is proved from both config and loaded tensors.
Already resident earlier tensors are no-ops on later passes. The queue is handed
to the runner rather than creating a second queue.

In scope:

- owned Qwen3.5 dense safetensors with no top-level or nested
  `quantization_config`;
- `VT_RELEASE_HOST_WEIGHTS` enabled, CUDA queue, non-UMA backend;
- one queue selected before load and reused by the runner;
- synchronized per-layer device residency and host release;
- local peak PSS, VRAM, startup, correctness and throughput gates.

Out of scope:

- CPU fallback, GB10/UMA, GGUF, quantized W4A4 27B, MoE, borrowed weights,
  offload, expert streaming, or changing queue ownership generally;
- direct parsing into final packed device layouts or asynchronous double
  buffering;
- model math, dtype, kernel dispatch, scheduler, sampling, or support claims;
- Spark/GB10 execution.

## Whole execution chain

### vLLM oracle

- `vllm/model_executor/model_loader/base_loader.py:43-82` enters the target
  device context before model construction and weight loading.
- `vllm/model_executor/model_loader/default_loader.py:266-298,414-429` selects
  the safetensors stream and passes it directly to `model.load_weights`.
- `vllm/model_executor/model_loader/weight_utils.py:820-954` scopes one
  `safe_open` shard and yields tensors rather than materializing a second full
  host model.
- `vllm/model_executor/models/qwen3_5.py:279-294,483-492` loads stacked/tied
  parameters into their canonical module-owned target-device storage.

### vllm.cpp

- `src/vllm/entrypoints/model_loader.cpp:18-37,224-233` currently selects the
  queue only inside `LoadedEngine` construction, after complete weight loading.
- `src/vllm/model_executor/models/model_registry.cpp:209-220,413-419` routes
  dense safetensors but carries no load-device context in `ModelSource`.
- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:394-447` appends all
  layers before returning the owned model.
- `src/vllm/model_executor/models/model_registry.cpp:228-241` and
  `src/vllm/model_executor/models/qwen3_5.cpp:3358-3402` already implement the
  exact eager representations, synchronization policy and final release used
  after loading. W4.4 moves those existing operations to bounded checkpoints;
  it does not add a second representation.

This changes host/device lifetime and launch ordering, not a GPU compute kernel.
Process PSS/VRAM over complete launch is the execution trace that decides it;
nsys kernel names are not an appropriate residency proof.

## Dispatch and lifetime contract

1. `FromModelDir` resolves the architecture before weights. Only the dense
   safetensors path preselects a queue and passes it to both loader and runner.
2. Early staging requires all of: queue present, CUDA, backend non-UMA,
   `VT_RELEASE_HOST_WEIGHTS!=0`, no top-level/nested quantization config, and
   the currently loaded weights passing `IsPlainBf16Qwen3_5Dense`.
3. After layer 0 and each later layer, prepare every currently present tensor;
   resident tensors short-circuit through their existing `d_dev`/`d_dev_f32`.
   Synchronize before releasing any host bytes.
4. `OwnedTensor` logical-presence metadata remains unchanged after release, so
   dispatch and canonical row-sliced fallbacks remain stable.
5. `VT_RELEASE_HOST_WEIGHTS=0` disables both early staging and later release.
6. If queue selection falls back to CPU, the existing host-only load and runner
   path is preserved. No temporary second CUDA queue is created.

## Port and test map

| Work | Implementation anchor | Tests/evidence |
|---|---|---|
| Carry selected load queue | `ModelSource`; `LoadedEngine::FromModelDir` and private construction seam | `test_model_registry`; `test_loaded_engine_dense` |
| Eligibility and per-layer stage/release | dense loader plus existing prepare/release helpers | `test_qwen27_dense_forward`; focused CUDA default and `VT_RELEASE_HOST_WEIGHTS=0` |
| Real-model acceptance | unchanged local memory driver | three interleaved direct ON, release-OFF and fresh-vLLM reps; exact token A/B |

There is no upstream unit test for this C++ ownership seam. Existing upstream
loader tests specify streaming target-device behavior; local tests pin the
non-owning queue propagation and exclusions. No upstream test is dropped.

## Gates

1. CPU and native-sm_120 builds pass; full CPU CTest and focused loader,
   registry, dense forward and loaded-engine tests are green.
2. CPU, CUDA release-OFF, UMA policy, quantized and borrowed behavior remain
   unchanged. No early host release occurs without a synchronized resident
   representation.
3. Default versus `VT_RELEASE_HOST_WEIGHTS=0` is byte-identical on all 128
   requests and default matches the accepted W4.3 project output 128/128.
4. Three valid CUDA-flake memory reps under one lock record peak/stable PSS,
   process VRAM, full wall and startup proxy against fresh vLLM.
5. Peak PSS must be no higher than fresh vLLM without regressing stable PSS or
   VRAM. Otherwise the row remains `ACTIVE` and the remaining lifetime is
   identified rather than averaged away.
6. Three unmonitored reps show no throughput/latency regression versus
   same-binary OFF and record every available axis versus fresh vLLM.
7. Every process exits; GPU returns idle; kernel journal is free of adjacent
   Xid, UVM/AER fault, reset or lockup.

## Work breakdown

1. `W4.4a`: carry a non-owning preselected dense load queue through
   `ModelSource` and reuse it in the runner.
2. `W4.4b`: add strict plain-BF16 discrete-CUDA eligibility and per-layer
   prepare/synchronize/release checkpoints.
3. `W4.4c`: build/test both enabled and release-OFF modes.
4. `W4.4d`: run the exact W4.3 memory/performance campaign and either close the
   peak axis or identify the next concrete lifetime.

W4.4a-W4.4d are complete. `ModelSource` carries the
non-owning queue, dense `FromModelDir` preselects and reuses one queue only when
direct loading is requested, and the loader stages/synchronizes/releases after
each eligible layer. `VT_DIRECT_DEVICE_LOAD=0` restores W4.3 from the same
binary. CPU focused tests pass 4/4; native-sm_120 default passes 5/5 and direct
OFF passes 3/3; full CPU CTest passes 105/105.

The immutable `5508f71` campaign closes the local launch-PSS axis. Three-run
mean peak PSS is **1.855 GiB** ON (1.577-2.411), **8.168 GiB** OFF and
**7.640 GiB** fresh vLLM (7.048-8.058), so ON is 0.2428x vLLM and its worst run
is below vLLM's best. Stable PSS is 0.757/0.754/4.109 GiB and peak process VRAM
is 11,688/11,682/12,929 MiB. The startup proxy improves 6.171 to 6.074 s.
Unmonitored throughput is **6556.86** ON, **6556.90** OFF and **6717.48** vLLM:
the path is neutral versus OFF but remains 0.9761x vLLM. ON/OFF output is exact
128/128 in all repetitions; existing project/vLLM corpus exactness remains
79/128. Raw evidence is `/tmp/qwen35-direct-w4-5508f71`; driver SHA-256 is
`7d90bfb16804bff3db38097f29cd2e971dc0b7e0e7ad14eb0e2c22eeb9fbf800`.
The W4.4 launch-memory work is accepted, while the owning row remains `ACTIVE`
for cross-engine correctness and performance parity.

Post-rebase `80f370f` checkpoint: upstream `e10786e` composes its current
prefix-caching resolution with the preselected direct-load queue. The exact
three-arm campaign reproduces low peak PSS (1.847 GiB ON, 8.168 OFF, 6.883
vLLM) and reaches 6607.68/6605.42/6717.32 tok/s. Closed-loop mean TTFT is
658.12/658.65/904.32 ms, while P99 is 3537.55/3538.98/3101.13 ms.

Disposition is **FAILED correctness / diagnostic metrics**. ON/OFF exactness is
123/128, 122/128 and 121/128; ON-ON and OFF-OFF repetitions vary similarly,
while vLLM is stable 128/128. This refutes a direct-load-specific corruption
inference but fails the same-binary precondition. The pre-rebase accepted result
is not relabeled onto rebased code. Raw evidence is
`/tmp/qwen35-direct-postrebase-80f370f`; driver SHA-256 is
`7d90bfb16804bff3db38097f29cd2e971dc0b7e0e7ad14eb0e2c22eeb9fbf800`.

## Risks and mitigations

- Early asynchronous free is a use-after-free. Every layer checkpoint
  synchronizes the selected queue before `ReleaseHost`.
- A second queue could reorder copies or leak resources. W4.4 selects one queue
  and passes the same handle into the eventual runner.
- Partial-vector moves could invalidate address-keyed state. Dense layers are
  reserved before append and residency lives in each `OwnedTensor` shared owner,
  not an external address map.
- Quantized and UMA models have different ownership economics. Config and
  backend guards exclude them before the first release.
- Rewalking prior layers could add startup cost. Existing resident lookups are
  no-ops; full startup and throughput A/B decide acceptance.
- Load exceptions after queue creation must destroy the preselected queue before
  rethrow unless ownership has transferred to the runner.
