# Qwen3.5 plain-BF16 dense loading and bounded device residency

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`,
`LOAD-SAFETENSORS-DIRECT-DENSE`
**Lifecycle:** `GATING`
**Owner:** unassigned

## Scope

Transplant the still-relevant part of `local-blackwell-environment` onto current
`main`: load ordinary, unquantized Qwen3.5 dense checkpoints such as
`Qwen/Qwen3.5-4B`, preserve their stacked/tied parameter topology, and bound
host residency by staging completed layers onto the selected discrete-CUDA
queue. This is an additive storage mode for the existing text-only
`Qwen3_5ForConditionalGeneration` implementation.

In scope:

- select ordinary `.weight` BF16/F32 tensors when compressed-tensors NVFP4
  companions are absent;
- keep Q/K/V/O, dense MLP, GDN QKVZ/BA and output projections in vLLM's raw
  torch-Linear orientation and use the existing `MatmulBT` path;
- represent tied `lm_head` as the embedding owner rather than copying it;
- reuse main's merged GDN projection and fused-attention policies unchanged;
- mirror vLLM's quantization-independent CUDA-graph selection for pure decode by
  reusing the existing Qwen3.5 dense graph driver for plain BF16;
- admit the exact 4B `Hq/Hkv=16/4`, head-dim 256 topology to the existing
  generic FA2 group-swap split-KV decode launcher, behind
  `VT_FA2_DECODE_4B=0` rollback;
- preselect one engine queue, pass it through `ModelSource`, stage each completed
  plain-BF16 layer, synchronize, release only host bytes with an authoritative
  device resident, then give that same queue to the runner;
- preserve `VT_DIRECT_DEVICE_LOAD=0` and the platform residency policy as
  same-binary controls;
- retain the NixOS development shell and local diagnostic helpers needed to
  reproduce the RTX 5070 Ti path.

Out of scope: replacing `VT_LOAD_WINDOWED_RELEASE`, the old tensor-range
`madvise` implementation, the old decomposed h32 GDN Triton kernel, the global
random-sampler scratch implementation, the unrestricted irregular-BF16 GEMM
fallback, changing the 27B/35B fusion defaults, GGUF, MoE, CPU offload, or a
support/performance claim before current-oracle gates pass.

User-directed local gate (2026-07-21): this 16-GiB machine cannot execute the
27B/35B checkpoints, so the active benchmark is the available plain-BF16 4B on
the preserved exact workload. Compare the current direct ON/OFF binary against
the prior 4B branch evidence and the locally available vLLM arm. The unavailable
27B/35B regressions remain explicit external-hardware follow-ups; they do not
prevent publishing an honest 4B diagnostic/result, but no 4B result is promoted
into a 27B/35B support claim.

## Upstream chain

| Upstream/dependency anchor | Contract to mirror |
|---|---|
| pinned vLLM `vllm/model_executor/model_loader/base_loader.py:43-82` | Construct parameters on the selected target device, load there, and finish post-load processing before returning the model. |
| pinned vLLM `vllm/model_executor/model_loader/weight_utils.py:820-954` | Yield safetensors incrementally rather than retaining a second complete host model. |
| pinned vLLM `vllm/model_executor/models/qwen3_5.py:276-303` | Stack QKV, gate/up, GDN BA, and GDN QKVZ physical parameters in the declared shard order. |
| pinned vLLM `vllm/model_executor/models/qwen3_5.py:483-492` | Tied output embeddings share the embedding parameter rather than copying bytes. |
| pinned vLLM `tests/models/test_initialization.py` | Model initialization and tied-parameter lifetime remain valid after loading. |
| installed vLLM 0.25 `vllm/config/compilation.py:590-619,1316-1460`; `vllm/compilation/cuda_graph.py:145-164` | CUDA graph mode is selected by runtime batch/model capability, not weight quantization. Full decode capture caches a graph per batch descriptor and replays it on later calls. |

The required `/home/mudler/_git/vllm` checkout is absent on this host. The
available `/home/rich/python/vllm` checkout confirms the same target-device and
stacked-parameter contracts, but it is not substituted as parity evidence. The
implementation therefore uses the already-recorded pinned anchors above and
requires a current pinned/v0.25 oracle run before any correctness or benchmark
claim. This is a loader/lifetime change; process PSS/RSS and per-process VRAM,
not nsys kernel timing, decide its memory gate. Any performance comparison still
requires matching nsys traces under the project parity protocol.

## Our baseline

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:116-166` assumes
  NVFP4 companions for attention, GDN output, and dense MLP projections.
- `src/vllm/model_executor/models/qwen3_5_dense.cpp:60-80` loads after the queue
  context is lost and has an empty dense prepare hook.
- `src/vllm/entrypoints/model_loader.cpp:355-374` owns safetensors correctly for
  deferred MoE streaming, but constructs the runner queue only after model load.
- `include/vllm/model_executor/models/qwen3_5_weights.h:39-75` has resident
  device handles and `ReleaseHost`, but `Empty()` still equates released host
  bytes with an absent logical weight.
- Main's `LOAD-SAFETENSORS` progressive interior-page release and merged
  GDN QKVZ/BA owners are binding and must remain the baseline.
- Historical branch evidence showed the local path was feasible and memory
  positive, but its vLLM 0.24 numbers and 121-125/128 correctness are diagnostic
  only after this transplant.

## Port map

| Work | Current destination | Required adaptation |
|---|---|---|
| plain BF16/F32 recognition and raw-NK load | `qwen3_5_dense_weights.cpp`, `qwen3_5_dense.h` | Compose with `dense_weight_loaders.h` and main's merged QKVZ/BA owners; do not restore split owners. |
| plain projection execution | `qwen3_5.cpp` | Add raw-NK/plain branches without changing NVFP4/FP8/GGUF behavior or `FuseAttnPreambleOn`. |
| plain BF16 decode graph | `qwen3_5_dense.cpp`, `qwen3_5.cpp` | Remove the registry's FP4-only eligibility restriction. Reuse the existing dense graph driver, capture sizes, persistent metadata, eager fallback and `VLLM_CPP_DENSE_DECODE_GRAPH=0` rollback unchanged. |
| 4B full-attention decode | `qwen3_5.cpp`, `cuda_paged_attn.cu`, `cuda_flash_attn_fa2.cu` | Mirror the observed vLLM FA2 split-KV dispatch for Hq/Hkv=16/4, D256. Extend only the generic launcher's bounded admission and keep a topology-specific rollback. |
| tied logits | dense weight metadata plus dense forward | Select `embed_tokens` as the logits operand; never copy it. |
| logical host release | `OwnedTensor`, dense traversal | Preserve logical presence after synchronized release; reject host `View()` once released. |
| load queue propagation | `ModelSource`, `qwen3_5_dense.cpp`, `LoadedEngine` | Add a non-owning queue pointer while retaining `safetensors_owned`; transfer the same queue to the runner. |
| local environment | `flake.nix`, `flake.lock`, `.environment.md`, `.gitignore`, focused tools | Current snapshot only; old run chronology stays in Git/ledger rather than README. |

## Tests to port

| Upstream/branch contract | Local tier |
|---|---|
| Qwen3.5 stacked mapping in `qwen3_5.py:276-303` | `tests/vllm/models/test_qwen35_plain_weights.cpp` loads the real 4B checkpoint and checks ordinary BF16 attention/GDN/MLP owners, merged order and F32 state-parameter conversion. |
| tied parameter initialization | The real-checkpoint gate asserts one logical embedding/head owner; the CUDA full-engine ON/OFF case exercises that owner through repeated logits. |
| target-device construction/lifetime | `tests/vllm/models/test_model_registry.cpp` covers queue propagation; the real 4B CUDA case compares retained-host and direct-device full-engine execution in one binary. |
| released-host safety | Unit-test logical presence, byte count, invalid host access, and resident-only forward behavior. |
| full model | `tests/vllm/models/test_qwen35_plain_weights.cpp:162-196` skips without CUDA/the cached 4B model; when available it compares prompt/output token IDs for both direct-load arms. Current-vLLM oracle comparison remains a separate W4 gate. |
| graph selection and replay | The same real-checkpoint test runs enough tokens to reach cold, capture and replay states, then compares graph ON against the eager rollback. Matched `nsys --cuda-graph-trace=node` evidence must contain project graph-node rows before speed credit. |
| 4B FA2 decode | `test_ops_paged_attn` ports the existing ratio-6/ratio-8 composed-reference sweep to ratio 4 over batch 1/2/4/8/16, plus explicit toggle-off, finite-window and unsupported-geometry fallback cases. The real 4B gate and same-binary token A/B precede speed credit. |
| exact benchmark corpus and output capture | `tests/examples/test_bench.cpp` covers first-turn ShareGPT loading, exact request count, fixed output length and submission-order JSON token-ID serialization; the same header-only path is compiled into `vllm-bench`. |

No upstream test is dropped. GPU/model tests that cannot run in CI remain
checked in with an explicit model/backend skip.

## Gates

1. CPU configure/build and focused loader/registry/forward tests are warning
   clean; full CPU CTest is run or every unrelated failure is isolated.
2. CUDA compile and focused op/model tests pass on sm_120. The local benchmark
   target is Qwen3.5-4B; existing 27B 235/235 and 35B 315/315 gates remain
   unchanged external-hardware regressions, not local prerequisites.
3. Direct ON/OFF outputs are byte-identical for the same binary. Cross-engine
   correctness uses the current pinned/v0.25 oracle and cannot inherit v0.24
   branch numbers.
4. Three uncontended, interleaved ON/OFF/vLLM repetitions record peak/stable
   PSS/RSS, VRAM, startup, total/output throughput, request rate, TTFT, TPOT and
   ITL. Any missing axis leaves the row `ACTIVE`/`GATING`.
5. Direct load must be no worse than OFF on correctness, timing and VRAM and no
   worse than vLLM on memory before memory closure is claimed.
6. CPU, UMA-policy, quantized 27B, MoE/35B, GGUF, borrowed weights and explicit
   opt-out preserve their current dispatch. Sanitizer/memcheck checks the
   synchronized release boundary.
7. `scripts/check-agent-record.py`, its mutation suite,
   `scripts/check-doc-checkpoint.py`, README, BENCHMARKS, matrices, roadmap,
   ledger and state agree at every checkpoint.

Current checkpoint: `GATING / speed-pending` after transplanting the relevant
H32 AOT repair onto current `main` and completing the plain-BF16 graph and
ratio-4 FA2 work. CUDA arch `120a`, generated Triton target `sm_120`,
FlashInfer CUTLASS, `VLLM_CPP_TRITON=ON` and regeneration ON are verified in
the build. The real cached `Qwen/Qwen3.5-4B` graph/direct/eager gate passes
**3/3 cases and 1,672/1,672 assertions**; direct ON/OFF outputs are identical.

Immutable root `/tmp/qwen35-main-final-fa2-20260725` completed all 18 guarded
legs at 128 requests, 131,784 actual input tokens, 128 output tokens/request
and concurrency 32. A clean immediate vLLM confirmation supplies the stable
denominator. Direct ON/OFF/vLLM-0.25 means are **5769.99/5660.70/5849.80 total
tok/s**, **638.03/625.94/646.85 output tok/s**, **4.9867/4.8900/5.0536
req/s**, **834.91/943.43/1047.13 ms mean TTFT**, and
**43.72/43.84/38.55 ms mean TPOT/ITL**. Peak PSS is
**2.406/8.592/7.662 GiB**, stable PSS **0.759/8.589/4.029 GiB**, and peak VRAM
**12850.7/12843.3/12942.7 MiB**. ON is **+1.93%** total over same-binary OFF
and cuts peak/stable PSS **72.0%/91.2%**; ON=OFF output IDs match 128/128 in
all three pairs. The +7.3 MiB ON-vs-OFF VRAM delta is within the +8 MiB gate.

The repaired components are individually positive: H32 AOT **+4.5906%**,
plain-BF16 graph **+0.3873%**, and ratio-4 FA2 **+1.6004%** total throughput
against their same-binary rollbacks. The final `--cuda-graph-trace=node`
capture contains 453 graph launches and 200,972 graph-child kernel events.
Local FA2 is **180.28 us/call** versus vLLM **178.40 us/call**, and local
graph-child GPU time is below the matched vLLM trace, but binding TPOT remains
43.72 vs 38.55 ms. CUDA API attribution identifies the open lever: 497
sampled-ID D2H operations are followed by immediate main-stream synchronization
on discrete CUDA, costing 20.975 s / 42.20 ms each in the profile, whereas vLLM
defers that wait with events. The required repair is a device-resident
sampled-token map that remains correct across request compaction. Full evidence
and hashes:
[2026-07-25 4B repair](../../docs/bench-evidence/qwen35-4b-main-repair-20260725.md).
Sanitizer availability and external 27B/35B regressions remain follow-ups.

Corrected AOT reproduction entry point:

```sh
flock /tmp/gpu nix develop .#cuda --command bash -c '
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
cmake -S . -B build-nix-cuda-transplant-triton -G Ninja \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=120a \
  -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
  -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER" \
  -DVLLM_CPP_CUTLASS_DIR="$PWD/.venv-vllm/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DVLLM_CPP_FLASH_ATTN=ON -DVLLM_CPP_TRITON=ON \
  -DVLLM_CPP_TRITON_REGEN=ON \
  -DVLLM_CPP_TRITON_PYTHON="$PWD/.venv-vllm/bin/python" \
  -DVLLM_CPP_TRITON_VENDORED_DIR=/tmp/vllm-cpp-triton-aot \
  -DVLLM_CPP_TRITON_VENDORED_ARCH=sm_120 -DCMAKE_BUILD_TYPE=RelWithDebInfo
'
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton -j4
flock /tmp/gpu env LD_LIBRARY_PATH=/run/opengl-driver/lib \
  HF_HOME=$PWD/.hf-cache \
  build-nix-cuda-transplant-triton/tests/test_qwen35_plain_weights --no-skip
flock /tmp/gpu env LD_LIBRARY_PATH=/run/opengl-driver/lib \
  ctest --test-dir build-nix-cuda-transplant-triton \
  -R 'test_(qwen36_weights|model_registry|qwen27_dense_forward|qwen35_plain_weights|loaded_engine_dense)' \
  --output-on-failure
REQUIRE_TRITON_AOT=1 \
  CPP_BENCH=$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench \
  CMAKE_CACHE=$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt \
  flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh \
  /tmp/qwen35-main-final-fa2-<commit>
```

## Dependencies

- Existing rows: `LOAD-SAFETENSORS`,
  `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`, `BACKEND-PLATFORM`,
  `KERNEL-GEMM-BF16`, and the current merged-GDN/fusion implementations.
- Toolchain: Nix CUDA shell, CUDA 12.9/native sm_120 locally; production gates
  still require the canonical CUDA/CUTLASS/Triton configurations.
- Data/hardware: local RTX 5070 Ti 16 GiB plus `Qwen/Qwen3.5-4B`; GB10 gate
  models remain separate external-hardware regressions.
- License: behavior-only port from vLLM; no generated dependency kernel is
  copied by this leaf.

## Work breakdown

1. `W0` environment: **COMPLETE** — transplant and compact the Nix/local reproduction files,
   the NVFP4 device-compile guard, and safe diagnostic scripts.
2. `W1` loader: **COMPLETE** — ordinary BF16/F32 detection, merged raw-NK physical owners,
   tied logits, and focused loader tests.
3. `W2` execution: **COMPLETE / CUDA-COMPILED** — plain raw-NK branches and packed MLP use composed with
   current preamble/GDN dispatch; regression tests.
4. `W3` residency: **COMPLETE / LOCAL-CUDA-GATED** — logical host-release state, dense prepare/release traversal,
   queue propagation/reuse, exclusions and retained-host/direct-device token equivalence; sanitizer remains W4.
5. `W4` gates: **GATING / LOCAL 4B COMPLETE** — 18/18 guarded legs and matching
   traces bind the current diagnostic. Direct loading is exact and
   memory-positive. Stable vLLM-0.25 comparison is 0.9864x total/output
   throughput and slower on TPOT/ITL, so speed remains open.
6. `W5` plain-BF16 decode graph: **COMPLETE** — exact graph/eager/direct test,
   128/128 benchmark identity and attribution-complete node trace; +0.3873%.
7. `W6` 4B FA2 decode: **COMPLETE** — exact ratio-4 gate, composed-reference
   ladder and node trace; +1.6004%. Remaining work is the discrete-CUDA
   sampled-token device map and event-overlapped output consumption outside
   graph-child kernels.

## Risks and decisions

- Releasing host bytes before an asynchronous copy completes is a use-after-free;
  every release boundary synchronizes the selected queue.
- `Empty()` is dispatch metadata. Host reclamation must preserve logical
  presence without allowing a later CPU `View()` of freed bytes.
- A second queue can reorder copies or strand resources. The selected load queue
  is transferred exactly once to the runner; exceptions destroy it locally.
- Main's FP8/NVFP4/GGUF branches are numerically gated. Plain behavior is keyed
  by actual tensor presence/orientation, never by `num_experts == 0` alone.
- Historical branch benchmarks are not current evidence. README and
  BENCHMARKS carry `PENDING` until rerun.
