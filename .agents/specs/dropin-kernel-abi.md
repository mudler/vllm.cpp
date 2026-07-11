# Drop-in kernel ABI (`BACKEND-ABI-VT`) — implementation spike

Status: accepted implementation spike; W0 additive spine is `ACTIVE` for the
claimed `W0-GPU` cross-build/runtime closure. Pins: vLLM `e24d1b24fe96`,
CUTLASS `v4.4.2`, vLLM
FlashAttention `2c839c33`, and the dependency revisions recorded by the pinned
vLLM build. This is a design and execution plan, not an implementation or a
claim that another kernel/backend is supported.

### W0 implementation checkpoint (2026-07-10)

The narrow W0 claim has implemented the backend-neutral scalar-ID/layout/
tensor descriptor, monotonic queue identity, typed op registration, explicit
device-resource free functions, CUDA device/stream guard, queue-scoped
workspace roles and initialization policies, device scalars, capture-time
growth rejection, cleanup, and a raw-signature probe. The matching upstream
CUDA-context and graph-capture cases are ported in
`tests/vt/test_dropin_abi.cpp`; the clean CPU build and 94/94 CPU tests pass.
No production launcher uses the helper yet, so no kernel-family support or
throughput state changed and README remains unchanged.

The row remains `GATING`, not `DONE`, under these named W0 debts:

- `W0-GPU`: clean CUDA cross-builds for `80`, `90a`, and `121a`, then GB10
  probe/workspace/capture runtime and both gate-model correctness/trace/memory
  checks. The shared GB10 is held by `CLAIM-SERVE-GATE-1`; exact handoff below.
- `W0-SCALAR-FORWARDER`: W0 carries the exact upstream `ScalarType::id()` bit
  vocabulary in `include/vt/ops.h`, verified against the vendored Marlin
  header. Promoting that vendored class to one common forwarding header would
  touch an unclaimed production-kernel include, so it is deliberately deferred
  to the first family-migration claim; duplicate semantic IDs are forbidden.
- `W0-BACKEND-SHIM`: new adapters use `vt::{Alloc,Free,CreateQueue,DestroyQueue}`
  and CUDA registers device-index-aware callbacks in `cuda_dropin.cu`. Existing
  `Backend` virtual calls remain explicit index-0 migration shims because the
  production CPU/CUDA backend TUs were outside W0 ownership. Each family claim
  removes its direct shim use; the ABI row cannot close while adapter resources
  can bypass queue cleanup.

The objective is to reshape the `vt::` CUDA/ROCm **adapter boundary** around
the raw pointer, shape, stride, semantic dtype, workspace, device, and stream
contracts that vLLM and its kernel dependencies already expose. A source lift
then keeps the upstream raw dispatcher unchanged and replaces only the
`torch::stable::Tensor` launcher.

## Scope

The exact owning row is `BACKEND-ABI-VT`, the ABI leaf of `ROAD-V1-C1`.

In scope:

- A backend-neutral public op contract that remains
  `Queue&, Tensor..., XxxArgs`, plus a CUDA adapter helper that converts those
  values into the upstream raw launcher's pointer/shape/stride/stream form.
- An explicit semantic scalar-type and layout descriptor for packed FP4/FP8
  storage. `DType::kI8` remains the byte-storage type and is never guessed to
  mean both FP4 and FP8.
- Device-explicit allocation/queue decisions, typed registration, workspace
  roles and lifetimes, capture safety, error handling, and the three deferred
  M0.6 decisions from [backends.md](../backends.md).
- CUDA implementation and runtime gates first. ROCm reserves the identical
  boundary (`hipStream_t`, HIP errors, device+queue-scoped workspace), then
  receives a compile/runtime leaf under `BACKEND-ROCM` when hardware exists.
- A per-family incremental migration. Every family keeps its current behavior,
  selection rules, math, layout, and scale conventions; this ABI is glue only.

Out of scope:

- Implementing an absent kernel family, changing kernel math, or marking any
  kernel-matrix row `DONE` merely because its launcher uses the common helper.
- Treating Python orchestration, Triton/CuTe source, Inductor output, or a JIT
  package as an arbitrary C++ source drop-in. Those remain porting references
  or use the already bounded vendored-AOT route.
- A big-bang rewrite of every CUDA translation unit, or GPU execution during
  this docs-only spike.

Dispatch behavior does not change: the op table is selected by
`(OpId, DeviceType)` and the queue carries the device index and native stream.
Unsupported dtype/layout/shape/architecture combinations fail before launch;
there is no silent fallback or precision change.

## Upstream chain

### The actual stable-ABI seam

Pinned vLLM's wheel-facing sources under `csrc/libtorch_stable/` have two
layers:

1. A **Layer-A launcher** accepts `torch::stable::Tensor`, validates device,
   scalar type, sizes and strides, allocates temporary tensors, installs a
   device guard, gets the current stream, and unpacks pointers.
2. A **Layer-B dispatcher/kernel** consumes raw pointers, integer geometry,
   optional scalar-type IDs/parameter structs, workspace, device capability,
   and `cudaStream_t`/`hipStream_t`.

The shared stable helper makes the ambient stream explicit at
`csrc/libtorch_stable/torch_utils.h:76-82`. Representative Layer-A to Layer-B
boundaries, covering every kernel-matrix block, are:

| Kernel-matrix rows | Pinned upstream/dependency boundary | Signature convention to preserve |
|---|---|---|
| `KERNEL-CUDA-DISPATCH-AOT` | `csrc/libtorch_stable/torch_utils.h:76-82`; `CMakeLists.txt:375-406,536-538` | device guard + current native stream; per-source architecture dispatch |
| `KERNEL-GEMM-BF16`, `KERNEL-GEMM-FP8`, `KERNEL-GEMM-NVFP4-W4A4`, `KERNEL-GEMM-W4A8`, `KERNEL-GEMM-QUTLASS-MX`, `KERNEL-GEMM-DEEPGEMM` | FP4 Layer A and workspace at `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu:178-196,235-300`; external cuBLASLt/CUTLASS/QuTLASS/DeepGEMM dispatch remains dependency-owned | raw A/B/C/scale pointers, M/N/K and leading/stride layout, device scalar alpha, workspace pointer/bytes, stream |
| `KERNEL-GEMM-MARLIN-W4A16`, `KERNEL-GEMM-INT-LOWBIT` | raw `marlin_mm` at `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu:344-356`; Tensor launcher at `:543-560,621-715` | raw data/workspace pointers, exact `vllm::ScalarType` IDs, shapes/config, device index, stream |
| `KERNEL-EW-NORM-ACT`, `KERNEL-EW-NORM-QUANT` | activation unpack/launch at `csrc/libtorch_stable/activation_kernels.cu:235-288`; RMSNorm at `layernorm_kernels.cu:219-265`; FlashInfer fused norm+FP4 entry/stream shapes at `/home/mudler/_git/flashinfer-ref/cute_dsl/rmsnorm_fp4quant.py:695-743` and `add_rmsnorm_fp4quant.py:945-1006` | typed raw input/output/weight/scale pointers, token/hidden geometry, optional pointers, stream |
| `KERNEL-ROPE-QKNORM` | raw RoPE kernel at `csrc/libtorch_stable/pos_encoding_kernels.cu:77-99`; Tensor launcher at `:103-145` | raw positions/Q/K/cache, element strides, head counts/dims, mode scalars, stream |
| `KERNEL-KV-CACHE`, `KERNEL-KV-QUANT` | raw cache kernel at `csrc/libtorch_stable/cache_kernels.cu:255-325`; Tensor launcher/stream at `:701-724`; FlashInfer workspace is caller-reserved at `/home/mudler/_git/flashinfer-ref/decode.py:676-692,816-835` | raw cache/data/slot/scale pointers, block/page/head strides, layout/dtype enum, stream; some workspaces require first-use zeroing |
| `KERNEL-ATTN-PAGED`, `KERNEL-ATTN-FA2`, `KERNEL-ATTN-FA3-FA4`, `KERNEL-ATTN-FLASHINFER-TRTLLM`, `KERNEL-ATTN-TRITON-FLEX-HPC`, `KERNEL-ATTN-MLA-SPARSE` | dependency raw FA-2 dispatcher vendored from `2c839c33` at `src/vt/cuda/flash_attn/src/flash_fwd_launch_template.h:55-164`; vLLM attention-state launcher at `csrc/libtorch_stable/attention/merge_attn_states.cu:246-320`; dynamic backend priority at `vllm/platforms/cuda.py:89-160,360-493` | params struct or raw Q/K/V/cache/metadata pointers, explicit element/byte strides, workspace, stream; backend selection remains outside the ABI helper |
| `KERNEL-MOE-ROUTING`, `KERNEL-MOE-UNQUANTIZED`, `KERNEL-MOE-QUANTIZED`, `KERNEL-MOE-SPECIAL` | raw align/sort at `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:295-350`; raw Marlin boundary above; MoE source/build inventory at `CMakeLists.txt:1135-1299` | routing/output/workspace pointers, expert/token geometry, quant scalar IDs, stream |
| `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`, `KERNEL-SSM-MAMBA` | shared CUDA/ROCm `SSMParamsBase&` + stream at `csrc/libtorch_stable/mamba/selective_scan_fwd.cu:393-460`; FlashInfer GDN caches a device workspace at `/home/mudler/_git/flashinfer-ref/gdn_prefill.py:345-361` | stable POD parameter struct containing raw pointers/strides/shapes plus explicit stream and scratch lifetime |
| `KERNEL-SAMPLING` | Tensor launcher and raw pointers at `csrc/libtorch_stable/sampler.cu:618-668`; persistent top-k raw launch accepts stream at `persistent_topk.cuh:1294-1335` | logits/mask/output pointers, row/vocab strides, deterministic scalar config, stream |
| `KERNEL-COLLECTIVES` | custom all-reduce launcher at `csrc/libtorch_stable/custom_all_reduce.cu:70-116` | opaque communicator/registered-buffer pointers, byte count/dtype, rank topology, explicit device stream |
| `KERNEL-SPEC-DECODE` | generated setup/rejection kernels at `vllm/v1/spec_decode/utils.py:306-565` and tests below | not a direct csrc lift today; a future raw POD/stream launcher must preserve ordering, RNG and acceptance semantics |

The common pattern is not one universal kernel function. It is a stable
**adapter vocabulary**: pointer + semantic scalar type + layout + sizes/strides
+ device + workspace role + native stream, followed by the upstream family's
unchanged raw function.

### Runtime trace plan

The ABI helper has no kernel-selection policy of its own. For each migrated
family, capture the pre/post adapter on the same warmed workload with
`nsys profile --trace=cuda,nvtx` and compare both `cuda_api_sum` and
`cuda_gpu_kern_sum`. Kernel names, call counts, launch geometry, and selected
cuBLASLt/CUTLASS/FlashInfer tactic must remain identical. The CUDA API trace
must show no steady-state `cudaMalloc*`/`cudaFree*`; a workspace growth event is
allowed only during warmup. Generated/JIT families additionally archive the
generated source or manifest hash. This supplements, never replaces, each
family row's own selection trace.

## Our baseline

The public tensor/runtime pieces already exist:

- `Tensor` is a non-owning POD with data, storage `DType`, `Device`, rank,
  four sizes and element strides (`include/vt/tensor.h:12-37`).
- `Queue` already carries `Device` and an explicit native handle
  (`include/vt/device.h:14-27`).
- Function aliases and `*Args` PODs are centralized in
  `include/vt/ops.h:160-285`; `src/vt/ops.cpp:9-33` stores one implementation
  per `(OpId, DeviceType)`.
- The backend registry and allocation interface are still per device type
  (`include/vt/backend.h:11-57`, `src/vt/backend.cpp:16-32`).
- Output validation already widened the old f32-only rule to f32/bf16 for the
  applicable ops (`src/vt/ops.cpp:16-17,35-60`).

Three shipped lifts prove the raw-dispatcher approach while also exposing the
boilerplate to consolidate:

| Lift | Existing adapter evidence | Friction the ABI removes |
|---|---|---|
| CUTLASS NVFP4/FP8 | `src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:46-96,501-571`; sibling FP8 adapter `cuda_matmul_fp8_cutlass.cu:52-106,273-309` | duplicate stream/error helpers, per-stream workspace/output staging, device alpha pointer |
| Marlin NVFP4 W4A16 | `src/vt/cuda/cuda_moe_marlin.cu:39-100,118-153` calling the vendored raw dispatcher | duplicate stream/error helpers and `c_tmp` pool; semantic BF16/FP4/FP8 scalar IDs are hard-coded because storage `kI8` is ambiguous |
| FlashAttention-2 | `src/vt/cuda/cuda_flash_attn_fa2.cu:78-126,176-257` | duplicate error helper; `LsePool` at `:92-115` is one process-global pointer rather than device+queue scoped; params retain a compatibility `at::PhiloxCudaState` at `:226` |

At baseline, 12 CUDA TUs define `AsStream`, 13 directly cast `q.handle`, 12
allocate CUDA memory, and the CUTLASS, Marlin, and FA-2 adapters each implement
their own scratch lifetime. The registry/device/queue/capture slice works and
is tested (`tests/vt/test_backend.cpp:10-47`,
`tests/vt/test_cuda_backend.cpp:36-145`); the common drop-in adapter does not
exist. `BACKEND-ABI-VT` is therefore honestly partial before implementation.

### Adapter-boundary design

The public op alias remains backend-neutral. Packed representations add an
explicit field to their existing `*Args` POD; no public CUDA type leaks into
`include/vt/ops.h`. The CUDA helper performs the Layer-A conversion:

```cpp
struct KernelTensorDesc {
  void* data;
  DType storage_dtype;
  vllm::ScalarTypeId scalar_type;  // logical type, including sub-byte types
  Device device;
  int rank;
  int64_t shape[kMaxRank];
  int64_t stride[kMaxRank];        // elements, not bytes
  KernelLayout layout;             // strided, packed-2x4, blockscale, Marlin...
};

KernelTensorDesc Describe(const Tensor&, vllm::ScalarTypeId, KernelLayout);
cudaStream_t Stream(const Queue&);
WorkspaceLease AcquireWorkspace(const Queue&, OpId, WorkspaceSlot,
                                size_t bytes, size_t alignment,
                                WorkspaceInit init);
DeviceScalarLease StageScalar(const Queue&, OpId, WorkspaceSlot, float value);
```

Implementation rules:

- Promote the already vendored, torch-free upstream
  `src/vt/cuda/marlin/core/scalar_type.hpp:14-21,315-350` into a common
  include used by adapters. Normal F32/F16/BF16/I8/I32/I64 conversion may use
  `ToScalarType(DType)`. Packed data **must** provide its exact logical ID in
  the op args (`kFE2M1f`, `kFE4M3fn`, `kFE8M0fnu`, etc.); `kI8` is storage only.
- `KernelLayout` records the family-visible contract (ordinary strided,
  packed-two-FP4-per-byte, linear/swizzled block scale, Marlin interleave). It
  validates the adapter; Layer B still receives its original individual raw
  arguments, so copied kernel code does not learn this wrapper.
- Workspace identity is
  `(DeviceType, device_index, Queue::id, native_handle, OpId, WorkspaceSlot)`.
  Device+queue identity is mandatory because default stream handles may be
  null on multiple devices and CUDA may recycle a destroyed stream handle.
  Multiple named slots allow workspace, output staging, LSE, semaphore, and
  device scalars to coexist.
- Workspace entries have stable heap ownership behind a mutex. Capacity grows
  only outside capture; `cudaStreamIsCapturing` plus insufficient capacity is
  a pre-launch error. Warmup reserves the high-water mark. `WorkspaceInit`
  distinguishes uninitialized scratch, zero-on-first-use, and zero-each-use.
  `DestroyQueue` synchronizes and releases its entries; there is no process-exit
  leak and no stale entry after handle reuse.
- Allocation, scalar copies, and launches use the same queue stream and an RAII
  device guard. A shared CUDA/HIP error helper supplies operation context.
  The hot path performs map lookup only and never allocates.

### M0.6 decisions resolved

1. **Registry/resource granularity:** keep kernel and backend code registered
   per `DeviceType`; one compiled implementation serves every index. Make
   resource operations explicit per `Device`: add device-taking allocation,
   free, and queue-creation forms and carry a monotonic queue ID. Temporary
   index-0 overloads may ease migration, but new adapter code may not use them.
   This chooses the `Alloc(Device, ...)` branch of the old
   "per-Device registry or explicit Device" question; duplicating op tables per
   GPU is unnecessary.
2. **Function aliases:** `include/vt/ops.h` is the single cross-backend ABI.
   CUDA, ROCm, CPU, Metal, Vulkan, and XPU register against the same alias with
   a typed registrar/static cast so signature drift is a compile error. A new
   backend never forks an alias.
3. **Output widening:** no global `out=f32` rule. Each public op declares the
   exact upstream-supported output set. The current matmul family keeps
   `{f32,bf16}` (`src/vt/ops.cpp:16-17,35-60`); f16 is added only for a family
   whose upstream contract supports it. Unsupported output types throw before
   dispatch and are never silently narrowed.

### Migration decision

Use **per-family incremental migration**, not a big bang.

The ABI spine lands additively with a tiny raw-signature probe kernel and no
production-adapter change. Each existing family then migrates its own disjoint
launcher, keeps a temporary same-binary old/new selector, runs its existing
goldens, traces, and independently records the benchmark checkpoint required by
`workflow.md`. Preparation can occur in separate worktrees, but speed-sensitive
migrations merge **one at a time**; a second migration cannot stack before the
first has reproduced pre/post A/B evidence. This preserves bisection and exact
attribution while avoiding a repository-wide adapter rewrite.

## Port map

| Pinned upstream/dependency | Local implementation target | Exact rule/deviation |
|---|---|---|
| `csrc/libtorch_stable/torch_utils.h:20-82` device properties/current stream | new `src/vt/cuda/cuda_dropin.{h,cu}`; HIP twin later under `src/vt/rocm/` | queue supplies device+stream explicitly; no ambient PyTorch stream |
| `csrc/core/scalar_type.hpp:14-23,315-350` as vendored today at `src/vt/cuda/marlin/core/scalar_type.hpp:14-21,315-350` | common torch-free scalar-type header + adapter conversion | preserve upstream IDs so raw Marlin/Machete dispatchers remain unchanged; storage dtype remains separate |
| `torch::stable::Tensor` metadata access used by the Layer-A launchers above | `KernelTensorDesc` built from `vt::Tensor` + explicit semantic/layout args | element strides preserved; optional tensors become nullable pointers; no owning tensor wrapper |
| `torch::stable::empty/new_empty` caching allocator use, plus FlashInfer caller-owned buffers | device+queue+op+slot workspace manager | preserve each caller's size/alignment/init contract; prohibit capture-time growth |
| Layer-B stable csrc raw dispatchers (Marlin, selective scan, core kernels, collectives) | corresponding existing/future `src/vt/cuda/cuda_*.cu` launcher | copy dispatcher/kernel with upstream path+commit header; only Layer A is replaced; `STD_TORCH_CHECK` maps to `VT_CHECK` |
| FA/CUTLASS/FlashInfer/other dependency launchers | one focused dependency adapter TU per family | preserve params structs, tactic/architecture selection and stream; generated code uses the vendored-AOT policy, not runtime Python |
| `tests/cuda`, `tests/kernels`, `tests/v1/cudagraph` cases below | `tests/vt/test_dropin_abi.cpp` plus the existing family test | test name/header cites the upstream case; blocked hardware cases land named and skipped |

No porting-inventory deviation is added by this spike. The implementation
records the shared adapter helper and any unavoidable departure from a copied
Layer B under porting-inventory §9. Existing CUTLASS/Marlin/FA-2 deviations
remain unchanged.

## Tests to port

The spine and every migration carry their tests in the same implementation
change. This table is the complete test surface for the ABI migration; it does
not imply absent families are implemented.

| Migration block | Pinned upstream tests/cases | Local tier and initial policy |
|---|---|---|
| ABI spine/device/stream | `tests/cuda/test_cuda_context.py:54-83`; capture/replay cases `tests/v1/cudagraph/test_cudagraph_dispatch.py:271-354` | new T-unit/T-parity `tests/vt/test_dropin_abi.cpp`: explicit device+stream, two devices/default handles, queue-ID reuse, workspace roles/growth/init, device scalar, typed registration, capture-time growth rejection; CUDA cases skip only when CUDA is absent |
| EW/norm/quant | `tests/kernels/core/test_activation.py:53,130,213`; `test_layernorm.py:43,96,140,226`; `test_fused_quant_layernorm.py:170`; `tests/kernels/test_fused_quant_activation.py:44` | existing RMSNorm/glue/CUDA-op tests plus named ports; numeric and alias/in-place behavior unchanged |
| dense FP8/NVFP4/low-bit GEMM | `tests/kernels/quantization/test_cutlass_scaled_mm.py:161-359`; `test_nvfp4_scaled_mm.py:57`; `test_marlin_gemm.py:150-576`; tile/quant modules listed in `kernel-family-inventory.md` | existing `test_ops_fp8_cutlass`, `test_ops_nvfp4_fp4`, `test_ops_nvfp4_matmul`; old/new adapter output and selected tactic identical |
| RoPE/QK norm | `tests/kernels/core/test_fused_qk_norm_rope.py`; `test_mrope.py`; `tests/kernels/test_fused_qk_norm_rope_gate.py` | existing `test_ops_rope`/`test_ops_attn_preamble`, preserving strides, optional K and in-place semantics |
| KV/cache | `tests/kernels/attention/test_cache.py:58,180,436,479,852`; `tests/kernels/test_cache_kernels.py:17`; `tests/quantization/test_per_token_kv_cache.py:176-462` | existing reshape-cache/paged-attention tests; blocked quant formats are checked-in skips with the owning row ID |
| attention | `tests/kernels/attention/test_flash_attn.py:95-109`; `test_attention.py`; `test_prefix_prefill.py`; FlashInfer/TRT-LLM/MLA modules inventoried in the kernel spike | existing `test_ops_paged_attn`; ragged, paged, causal/window, output/LSE/workspace paths; unavailable SM/dependency cases stay named skips |
| MoE | `tests/kernels/moe/test_fused_topk.py:57-215`; `test_grouped_topk.py:46`; `test_nvfp4_moe.py:53,179`; unquantized/DeepGEMM modules from the kernel spike | existing `test_ops_moe`, `test_ops_moe_grouped`; empty/ragged experts, tie/NaN policy, permute round trip and quant layouts |
| GDN/SSM | `tests/kernels/mamba/test_mamba_ssm.py:200-861`; `test_gdn_forward_core_split.py:154`; `test_gdn_prefill_cutedsl.py` | existing `test_ops_gdn`; defer any `cuda_gdn.cu` edit until `CLAIM-PR3` releases it |
| sampling | `tests/v1/sample/test_sampler.py:206-416`; `test_topk_topp_sampler.py:62-329`; `test_rejection_sampler.py:133-314` | existing `test_ops_sample`; preserve tie order, RNG and rejection semantics |
| collectives/spec decode | `tests/distributed/test_custom_all_reduce.py:123`; `test_quick_all_reduce.py`; `tests/v1/spec_decode/test_eagle_step_kernel.py:51-133`; rejection tests above | checked skips until topology/implementation exists; ABI migration alone cannot make these rows supported |
| End to end | pinned-vLLM oracle plus local `tests/parity/test_qwen27_paged_engine.cpp:110` and `test_qwen36_paged_engine.cpp:78` | both gate models; adapter refactors must preserve the accepted oracle-deterministic token span and all existing assertions |

`tests/cuda/test_cuda_compatibility_path.py` remains owned by
`KERNEL-CUDA-DISPATCH-AOT`: it tests loader environment-path setup, not the
kernel adapter boundary. No upstream case is silently claimed by this leaf.

## Gates

1. **ABI gate:** the test probe and then each migrated production launcher bind
   a raw pointer/shape/stride/semantic-type/workspace/stream function without a
   change to Layer B. Packed type/layout mistakes and unsupported SMs fail
   before launch.
2. **Device/resource gate:** two explicit CUDA queues and, where hardware
   permits, two device indices receive isolated workspace/scalar slots. Queue
   destroy/recreate cannot recover a stale pointer. CPU unit tests cover key
   equality and lifecycle without GPU hardware.
3. **Capture gate:** all needed capacities are reserved in warmup; replay does
   not allocate, change addresses, or grow. Capture-time under-capacity throws.
4. **Correctness gate:** old/new adapter outputs are bit-identical for glue and
   identical to the existing family tolerance/golden for numerical kernels;
   both gate-model correctness tests pass before measurement.
5. **Build/backend gate:** clean CPU build; clean CUDA builds for `80`, `90a`
   and `121a`; CUDA runtime on GB10. ROCm compile/runtime remains an explicit
   checked skip until the `BACKEND-ROCM` hardware leaf exists, so the umbrella
   ABI row cannot overclaim full ROCm completion.
6. **Trace/performance/memory gate:** same-binary old/new A/B, fresh graphed
   vLLM denominator, both models, every applicable throughput/latency/memory
   axis, 2–3 reproducing uncontended runs. Kernel names/calls remain identical,
   no steady-state allocator calls appear, and workspace high-water/peak memory
   do not regress. Per the 2026-07-10 checkpoint rule, do not merge/stack the
   next migration before this evidence is in the ledger.

Exact local record/build commands for the implementation change:

```sh
python3 scripts/check-agent-record.py
python3 tests/scripts/test_agent_record.py
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build-cpu --clean-first -j"$(nproc)"
ctest --test-dir build-cpu --output-on-failure
```

Deferred W0 GPU handoff (run only after `CLAIM-SERVE-GATE-1` releases the
shared machine; compilation itself needs no lock, execution holds one lock for
the whole series):

```sh
ssh dgx.casa
export PATH=/usr/local/cuda/bin:$PATH
git -C ~/work/vllm.cpp-backend-abi-w0 fetch origin codex/backend-abi-vt-w0
git -C ~/work/vllm.cpp-backend-abi-w0 checkout --detach FETCH_HEAD

for arch in 80 90a 121a; do
  cmake -S ~/work/vllm.cpp-backend-abi-w0 \
    -B ~/work/vllm.cpp-backend-abi-w0/build-cuda-${arch} \
    -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
    -DVLLM_CPP_CUDA_ARCHITECTURES=${arch}
  cmake --build ~/work/vllm.cpp-backend-abi-w0/build-cuda-${arch} \
    --clean-first -j"$(nproc)"
done

flock /tmp/gpu -c '
  set -eu
  cd ~/work/vllm.cpp-backend-abi-w0
  ctest --test-dir build-cuda-121a \
    -R "test_dropin_abi|test_cuda_backend|test_qwen27_paged_engine|test_qwen36_paged_engine" \
    --output-on-failure
  compute-sanitizer --tool memcheck --error-exitcode=99 \
    ./build-cuda-121a/tests/test_dropin_abi
'
```

Then resolve and record the exact snapshot directories beneath the two model
roots in `.agents/environment.md`; compare the parent-main and W0 binaries with
the same `vllm-bench` commands at 27B `np=96,c=16` and `np=192,c=32`, and 35B
`np=200,c=64`, with 2–3 interleaved runs. One additional flock-held nsys pair
must show the production kernel lists/call counts unchanged and no W0 workspace
allocator calls outside `test_dropin_abi`; record total/output throughput,
req/s, TTFT, TPOT/ITL, peak memory, temperatures, commands, commits, and ratios.
Because W0 has no production-family call site, any non-noise model delta is a
regression, not a claimed ABI gain.

Exact GB10 build/test/profile skeleton (run in a dedicated synced clone; the
family migration substitutes its test and `VT_DROPIN_<FAMILY>` selector):

```sh
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
cmake --build build-cuda --clean-first -j"$(nproc)"
flock /tmp/gpu -c "ctest --test-dir build-cuda \
  -R 'test_dropin_abi|test_cuda_backend|test_ops_nvfp4_fp4|test_ops_nvfp4_matmul|test_ops_moe_grouped|test_ops_paged_attn|test_qwen27_paged_engine|test_qwen36_paged_engine' \
  --output-on-failure"
flock /tmp/gpu -c "VT_DROPIN_FAMILY=0 nsys profile --trace=cuda,nvtx \
  --force-overwrite=true -o /tmp/dropin-old ./build-cuda/vllm-bench \
  --model \"$MODEL\" --num-prompts 96 --input-len 1024 --output-len 128 \
  --concurrency 16 --seed 0 --temperature 0 --num-blocks 300 && \
  VT_DROPIN_FAMILY=1 nsys profile --trace=cuda,nvtx \
  --force-overwrite=true -o /tmp/dropin-new ./build-cuda/vllm-bench \
  --model \"$MODEL\" --num-prompts 96 --input-len 1024 --output-len 128 \
  --concurrency 16 --seed 0 --temperature 0 --num-blocks 300"
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum /tmp/dropin-old.nsys-rep
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum /tmp/dropin-new.nsys-rep
```

For the binding checkpoint, resolve `MODEL27`/`MODEL35` to the exact snapshot
directories, run ours at 27B `np=96,c=16` and `np=192,c=32`, and 35B
`np=200,c=64`; run 2–3 interleaved repetitions of each old/new arm. The fresh
vLLM arm uses the identical tokens/lengths and production graphs:

```sh
~/venvs/vllm-oracle/bin/vllm bench throughput --model "$MODEL" \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --random-range-ratio 0 --num-prompts "$NP" --max-num-seqs "$C" --seed 0
```

The implementation ledger records the full resolved paths, commit, commands,
temperatures/power, returned memory, repetitions, every axis, and ratios.

## Dependencies

- No lifecycle predecessor blocks the additive `BACKEND-ABI-VT` spine: the
  existing typed op/queue/backend baseline is sufficient. The adjacent
  `KERNEL-CUDA-DISPATCH-AOT` row remains independently incomplete and is not
  falsely promoted by this spike.
- A family migration starts only after the CUDA spine is merged and gated. It
  also obeys that family's model/quant/backend dependencies from
  [kernel-family-inventory.md](kernel-family-inventory.md).
- `CLAIM-PR3` exclusively owns `KERNEL-GDN-AOT-BF16`,
  `KERNEL-GDN-SCRATCH`, and `cuda_gdn.cu`; GDN migration waits for its release.
- Toolchain/hardware: C++17, CMake, CUDA 13/nvcc, nsys, GB10/sm_121a for the
  first runtime gate; cross-builds for sm_80/sm_90; ROCm compiler and supported
  AMD hardware for the later HIP gate; 2-GPU isolation evidence joins D2 when
  that hardware exists.
- Data: the existing Qwen3.6 27B/35B NVFP4 snapshots on `dgx.casa` and pinned
  vLLM oracle. No new training data or checkpoint license is introduced.
- Source/license: project/vLLM Apache-2.0; preserve every copied source's
  upstream copyright/SPDX notice. CUTLASS and vLLM-FlashAttention/Marlin retain
  their existing vendored provenance/license obligations. The helper itself
  introduces no new external dependency.

## Work breakdown

Each row below is a separate claim/checkpoint. Disjoint files permit advance
preparation, but performance-sensitive commits are gated and merged serially.

| Order | Claim block / stable rows | Narrow owned files | Exit/dependency |
|---:|---|---|---|
| 0 | ABI spine: `BACKEND-ABI-VT` | common scalar/layout descriptor; `include/vt/{device,backend,ops}.h`; `src/vt/{backend,ops}.cpp`; new `src/vt/cuda/cuda_dropin.{h,cu}`; new `tests/vt/test_dropin_abi.cpp`; narrow CMake lines | additive raw-signature probe, explicit-device resources, workspace/capture tests, CPU+GB10 gates; no production adapter migration |
| 1 | Proven GEMM lifts: one claim each for `KERNEL-GEMM-NVFP4-W4A4`, `KERNEL-GEMM-FP8`, `KERNEL-GEMM-MARLIN-W4A16` | respectively `cuda_matmul_nvfp4_cutlass.cu`, `cuda_matmul_fp8_cutlass.cu`, `cuda_moe_marlin.cu` | after order 0; each old/new golden+trace+A/B checkpoint completes before the next merges |
| 2 | Core GEMM and low-bit rows: `KERNEL-GEMM-BF16`, `KERNEL-GEMM-INT-LOWBIT`, `KERNEL-GEMM-W4A8`, `KERNEL-GEMM-QUTLASS-MX`, `KERNEL-GEMM-DEEPGEMM` | one family TU/adapter at a time (`cuda_matmul*.cu` or new dependency adapter) | migrate only existing launchers; absent families remain inventoried until their own implementation spike |
| 3 | EW/RoPE: `KERNEL-EW-NORM-ACT`, `KERNEL-EW-NORM-QUANT`, `KERNEL-ROPE-QKNORM` | `cuda_ops.cu`, then `cuda_glue.cu`, one claim per row | core upstream test ports and bit-identical old/new adapter behavior |
| 4 | KV/attention: `KERNEL-KV-CACHE`, `KERNEL-KV-QUANT`, `KERNEL-ATTN-PAGED`, `KERNEL-ATTN-FA2`, `KERNEL-ATTN-FA3-FA4`, `KERNEL-ATTN-FLASHINFER-TRTLLM`, `KERNEL-ATTN-TRITON-FLEX-HPC`, `KERNEL-ATTN-MLA-SPARSE` | `cuda_cache.cu`, `cuda_paged_attn.cu`, `cuda_flash_attn_fa2.cu`, then one new adapter per dependency family | existing rows only; FA-2 migration replaces the unsafe global LSE pool; absent backends remain skipped |
| 5 | MoE: `KERNEL-MOE-ROUTING`, `KERNEL-MOE-UNQUANTIZED`, `KERNEL-MOE-QUANTIZED`, `KERNEL-MOE-SPECIAL` | `cuda_moe.cu` and one quant/dependency TU per claim | routing/permute/GEMM/combine chain and leaf goldens; no umbrella pass hides formats |
| 6 | Recurrent: `KERNEL-GDN-AOT-BF16`, `KERNEL-GDN-SCRATCH`, `KERNEL-SSM-MAMBA` | `cuda_gdn.cu` only after PR3; future SSM adapter separately | PR3 closed first; scratch/capture and sequential-state gates |
| 7 | `KERNEL-SAMPLING` | `cuda_sample.cu` | deterministic tie/RNG/order tests and independent A/B checkpoint |
| 8 | `KERNEL-COLLECTIVES`, `KERNEL-SPEC-DECODE` | future `cuda_collectives*.cu` / `cuda_spec_decode*.cu` adapter only | waits for their implementation/topology specs; ABI shape is not support |
| 9 | ROCm mirror: `BACKEND-ROCM` plus the same applicable family rows | `src/vt/rocm/rocm_dropin.*` and one HIP family adapter | after D1 ROCm spike/toolchain/hardware; same scalar/layout/workspace contract and native-vLLM floor |

## Risks and decisions

- **Packed semantic types:** treating `kI8` as both FP4 and FP8 would reproduce
  today's hard-coded Marlin assumptions. The explicit scalar ID is binding; no
  inference fallback is permitted.
- **Multi-device/default-stream aliasing:** a stream handle alone is not a
  resource identity. Device index + queue ID + slot are binding, with cleanup
  on queue destruction.
- **Capture and concurrency:** buffer addresses cannot move during capture;
  map entries cannot be invalidated by rehash; multiple roles cannot alias.
  The warmup/high-water and stable-entry rules are correctness requirements.
- **Workspace semantics vary:** CUTLASS scratch is uninitialized, some
  FlashInfer semaphore/workspace paths require zeroing, and device scalars need
  stream-ordered copies. The caller selects an explicit init policy; a generic
  helper may not guess.
- **Not every kernel is a source drop-in:** cuBLASLt heuristics, Python
  orchestration, Triton/CuTe, Inductor, DeepGEMM/FlashInfer JIT, and collective
  topology still need their real dependency chain and runtime trace. The ABI
  standardizes their C++ launch edge; it does not erase those dependencies.
- **Incremental, independently gated migration is final:** a big bang is
  rejected because it destroys attribution/bisection and violates the
  independent performance-checkpoint rule. Preparation may overlap; merges may
  not stack before the prior row's reproduced checkpoint.
- **Behavior/product decisions:** none. vLLM's shapes, strides, dtype IDs,
  layouts, capability errors, selection rules, math and tests remain the floor.
  ROCm hardware availability is an execution blocker, not permission to claim
  support from a CUDA-only result.
- Matrix counts do not change: backend remains 51 rows and kernel remains 30.
  This spike moves `BACKEND-ABI-VT` from `SPIKE` to `READY`; implementation
  status remains future work.
