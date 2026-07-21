# Qwen3.5 plain-BF16 dense loading and bounded device residency

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`,
`LOAD-SAFETENSORS-DIRECT-DENSE`
**Lifecycle:** `ACTIVE`
**Owner:** `CLAIM-LOCAL-BF16-TRANSPLANT`

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

## Upstream chain

| Upstream/dependency anchor | Contract to mirror |
|---|---|
| pinned vLLM `vllm/model_executor/model_loader/base_loader.py:43-82` | Construct parameters on the selected target device, load there, and finish post-load processing before returning the model. |
| pinned vLLM `vllm/model_executor/model_loader/weight_utils.py:820-954` | Yield safetensors incrementally rather than retaining a second complete host model. |
| pinned vLLM `vllm/model_executor/models/qwen3_5.py:276-303` | Stack QKV, gate/up, GDN BA, and GDN QKVZ physical parameters in the declared shard order. |
| pinned vLLM `vllm/model_executor/models/qwen3_5.py:483-492` | Tied output embeddings share the embedding parameter rather than copying bytes. |
| pinned vLLM `tests/models/test_initialization.py` | Model initialization and tied-parameter lifetime remain valid after loading. |

The required `/home/mudler/_git/vllm` checkout is absent on this host. The
available `/home/rich/python/vllm` checkout confirms the same target-device and
stacked-parameter contracts, but it is not substituted as parity evidence. The
implementation therefore uses the already-recorded pinned anchors above and
requires a current pinned/v0.25 oracle run before any correctness or benchmark
claim. This is a loader/lifetime change; process PSS/RSS and per-process VRAM,
not nsys kernel timing, decide its memory gate. Any performance comparison still
requires matching nsys traces under the project parity protocol.

## Our baseline

- `src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:115-165` assumes
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
| tied logits | dense weight metadata plus dense forward | Select `embed_tokens` as the logits operand; never copy it. |
| logical host release | `OwnedTensor`, dense traversal | Preserve logical presence after synchronized release; reject host `View()` once released. |
| load queue propagation | `ModelSource`, `qwen3_5_dense.cpp`, `LoadedEngine` | Add a non-owning queue pointer while retaining `safetensors_owned`; transfer the same queue to the runner. |
| local environment | `flake.nix`, `flake.lock`, `.environment.md`, `.gitignore`, focused tools | Current snapshot only; old run chronology stays in Git/ledger rather than README. |

## Tests to port

| Upstream/branch contract | Local tier |
|---|---|
| Qwen3.5 stacked mapping in `qwen3_5.py:276-303` | Extend `tests/vllm/models/test_qwen27_dense_forward.cpp` with ordinary BF16 attention/GDN/MLP fixtures, merged order and F32 state-parameter conversion. |
| tied parameter initialization | Assert one logical owner and identical logits operand; exercise repeated forward after host release. |
| target-device construction/lifetime | Extend `tests/vllm/models/test_model_registry.cpp` and `tests/vllm/entrypoints/test_loaded_engine_dense.cpp` for queue propagation, exclusions and queue reuse. |
| released-host safety | Unit-test logical presence, byte count, invalid host access, and resident-only forward behavior. |
| full model | Local RTX test remains SKIPPED without `Qwen/Qwen3.5-4B`; when present, compare current vLLM oracle tokens and both direct-load arms. |

No upstream test is dropped. GPU/model tests that cannot run in CI remain
checked in with an explicit model/backend skip.

## Gates

1. CPU configure/build and focused loader/registry/forward tests are warning
   clean; full CPU CTest is run or every unrelated failure is isolated.
2. CUDA compile and focused op/model tests pass on sm_120. Existing 27B
   235/235 and 35B 315/315 gates remain unchanged before local-model credit.
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

Reproduction entry point after implementation:

```sh
nix develop .#cuda --command cmake -S . -B build-nix-cuda -G Ninja \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=native \
  -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
  -DCMAKE_CUDA_HOST_COMPILER="$CMAKE_CUDA_HOST_COMPILER"
nix develop .#cuda --command cmake --build build-nix-cuda -j4
nix develop .#cuda --command ctest --test-dir build-nix-cuda \
  -R 'test_(model_registry|qwen27_dense_forward|loaded_engine_dense)' \
  --output-on-failure
```

## Dependencies

- Existing rows: `LOAD-SAFETENSORS`,
  `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`, `BACKEND-PLATFORM`,
  `KERNEL-GEMM-BF16`, and the current merged-GDN/fusion implementations.
- Toolchain: Nix CUDA shell, CUDA 12.9/native sm_120 locally; production gates
  still require the canonical CUDA/CUTLASS/Triton configurations.
- Data/hardware: local RTX 5070 Ti 16 GiB plus `Qwen/Qwen3.5-4B`; GB10 gate
  models remain separate mandatory regressions.
- License: behavior-only port from vLLM; no generated dependency kernel is
  copied by this leaf.

## Work breakdown

1. `W0` environment: transplant and compact the Nix/local reproduction files,
   the NVFP4 device-compile guard, and safe diagnostic scripts.
2. `W1` loader: ordinary BF16/F32 detection, merged raw-NK physical owners,
   tied logits, and focused loader tests.
3. `W2` execution: plain raw-NK branches and packed MLP use composed with
   current preamble/GDN dispatch; regression tests.
4. `W3` residency: logical host-release state, dense prepare/release traversal,
   queue propagation/reuse, exclusion tests and sanitizer checks.
5. `W4` gates: current-oracle local correctness, matching traces and the full
   memory/performance series; update lifecycle honestly.

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
