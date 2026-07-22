# Metal/MLX reuse study + the acceleration-provider seam (`BACKEND-ACCEL-PROVIDER`)

Status: **STUDY + MEASURED MLX BASELINE, 2026-07-22.** No source file changed.
Owner: `CLAIM-BACKEND-FANOUT-1`.
Rows: `BACKEND-METAL-MLX` (unchanged `ACTIVE`), `BACKEND-GATE-METAL-MLXLM`
(**stays `INVENTORIED`** — the MLX arm is now installed, pinned and measured, but
a competitor baseline is not a gate: there is no implementation code and no
ours-vs-MLX evidence to anchor, so promoting it would be exactly the ungrounded
transition `scripts/check-agent-record.py` rejects. It moves at work row `M3b`).

**Why a separate spec rather than an extension of the fan-out spike.**
[backend-fanout-metal-vulkan-xpu.md](backend-fanout-metal-vulkan-xpu.md) is a
*per-backend port map* — it answers "what files does Metal/Vulkan/XPU need".
This document answers a different and more general question the user posed
directly: **"we need to build it in a way that we can extend acceleration
easily to other platforms, and e.g. integrate with MLX."** That is a question
about the *seam*, not about Metal, and its conclusions bind CUDA and CPU as
much as Apple. Folding it into a 762-line per-backend port map would bury it.
The fan-out spike stays the owner of the M/V/X work rows; this spec owns the
provider seam and the MLX evidence, and the two cross-link.

Pins: MLX `v0.29.3` (`4bce5f9b2dc4743b9e0bdf101a0d593053f4f099`, 2025-10-17) —
the version actually installed on the M4; MLX-LM `0.29.1`
(clone `cf10f962b7a20e63a6df43dbf0faf06070153d40`). Local base `1cb5f64`.
Every MLX citation below is `file:line` in that pinned tree.

---

## 0. Executive answer

1. **The engine reuses almost completely.** Running Qwen3-dense on Apple GPU
   needs **10 `vt::` ops**, of which Metal already has 3 — so **7 new MSL
   kernels**. OPT needs **9 ops**, 3 already present — **6 new kernels**. Every
   other layer (engine, scheduler, KV cache, sampler source, weight loading, all
   argument validation, the entire 9-recipe fusion catalog, the quant block/geometry
   tier) is device-agnostic and runs unchanged. §2, §3.
2. **The blocker is small and now exactly quantified**: 3 strictly-unconditional
   `vt/cuda/` includes, 1 hardcoded `kCUDA` queue, and **1 real portability BUG**
   in the shared dense attention path. §3.3.
3. **Our current seam extends to a new PLATFORM cleanly (proven twice) but has
   NO seam for a second PROVIDER on an existing platform.** `RegisterOp` is a
   flat `[OpId][DeviceType]` table with one `void*` slot and last-writer-wins;
   two implementations of one op on one device cannot coexist. That, not MLX, is
   the structural gap. §4.
4. **MLX integration is VIABLE, and the lazy-eval objection is REFUTED by
   source.** MLX's lazy graph sits strictly *above* its kernel layer; the kernel
   layer is already an eager, per-op, encode-into-a-command-buffer API taking
   plain C++ arguments (`steel_matmul`, `matmul.h:105-142`). We can call it
   without ever constructing a graph. §5.
5. **Recommendation: build `vt::OpProvider` — generalize the arch-tactic
   registry into the op table.** ~50 lines, zero call-site edits, and it serves
   MLX-on-Metal, cuBLASLt/CUTLASS/flashinfer-on-CUDA, llama.cpp-on-CPU/Vulkan
   with one mechanism. MLX then becomes a build-time configuration choice
   (`VLLM_CPP_MLX`), exactly as `VLLM_CPP_TRITON` already is. §6.
6. **MLX baseline MEASURED** on Qwen3-1.7B-bf16: 27.6 tok/s decode @ b=1 rising
   to 213.4 tok/s @ b=16, prefill saturating ~1200 tok/s, 3.78–5.28 GB peak.
   Recorded as an **UNOPPOSED FLOOR** — we have no Metal model and claim no
   comparison. §7.

---

## 1. How MLX organizes its Metal backend — and how it compares to our W0

Read from the pinned MLX tree. Our side is the Metal W0 skeleton
(`src/vt/metal/*`, landed `13bb724`).

### 1.1 Kernel source layout

`mlx/backend/metal/` is **24 `.metal` TUs + 30 `.h`** kernel headers, plus 30
host `.cpp` dispatch TUs (13,746 lines). The split is deliberate and is the same
split we chose: `*.metal`/`*.h` = device code; `<op>.cpp` = host-side shape
analysis, kernel-name construction, and encode. Ours is `metal_msl.h` (382 lines,
device) + `metal_ops.mm` (343, host) — the same shape at 1/40th the scale.

The heavy kernels live in `kernels/steel/` — a templated MSL tile library
(`steel/gemm/mma.h` builds on `#include <metal_simdgroup_matrix>` at `:6` and a
`BaseMMAFrag<T,8,8>` fragment abstraction at `:33-46`). This is MLX's CUTLASS
analogue and is the single largest thing we would not want to rewrite.

### 1.2 MSL compilation — MLX has BOTH routes, and ships the AOT one

| Route | MLX | Us (W0) |
|---|---|---|
| **AOT metallib (default)** | `kernels/CMakeLists.txt:27-33` shells out to `xcrun -sdk macosx metal ... -o ${TARGET}.air`, linked into `mlx.metallib`; loaded at `device.cpp:45-201` (`load_library_from_path`, colocated-binary search, SwiftPM bundle fallback) | **NOT AVAILABLE — no full Xcode on the M4, so no `metal` compiler.** This is why W0 chose runtime compilation |
| **Runtime source compilation (`MLX_METAL_JIT`)** | `Device::build_library_`, `device.cpp:504-528` — `device_->newLibrary(ns_code, options, &error)` | `metal_context.mm` — `newLibraryWithSource:`. **Identical mechanism** |
| **Relaxed-math pin** | `options->setFastMathEnabled(false)` (`device.cpp:512`); AOT path passes `-fno-fast-math` (`kernels/CMakeLists.txt:12`) | `MTLMathModeSafe`. **Independently converged on the same decision** |
| **Library cache** | `library_map_` under `std::shared_lock`/`unique_lock` (`device.cpp:484-502`) | pipeline cache in `metal_context.mm` |

Two findings that matter:

**(a) W0's runtime-MSL choice is validated by MLX itself**, which maintains the
same code path as a first-class build mode. It is not a workaround.

**(b) MLX ships the compiled artifact in its wheel.** On the M4:
`~/mlx-venv/lib/python3.9/site-packages/mlx/lib/mlx.metallib` = **104,894,650
bytes**, alongside `libmlx.dylib`. So *consuming* MLX needs no Xcode — but it
means a ~105 MB precompiled metallib plus a C++ shared library enter our
dependency surface. That is the honest cost line in §6, and it is measured, not
estimated.

**(c) Specialization we do not have.** MLX specializes kernels via
`MTLFunctionConstantValues` (`device.cpp:540-580`) and builds kernel names as
strings (`quantized.cpp:19-37`, `scaled_dot_product_attention.cpp:60`). Our W0
compiles one fixed variant per op. This is a real MLX capability, and it is
*independent of adopting MLX* — the technique ports to our own MSL for free.

### 1.3 Dispatch and buffer model — the one place MLX is meaningfully ahead

| Aspect | MLX | Us (W0) |
|---|---|---|
| Command buffers | ONE per stream, many encoders; `maybeInsertBarrier()` inserts `memoryBarrier(BarrierScopeBuffers)` **only when a read aliases a previous write** (`device.cpp:264-273`, hazard set at `:231-236`) | **ONE command buffer PER OP, commit + wait.** Documented in the spike as "correct, not fast" |
| Storage mode | `ResourceStorageModeShared \| ResourceHazardTrackingModeUntracked` (`allocator.cpp:14`) — unified memory, manual hazards | shared storage; hazards implicit in commit+wait |
| Allocation | `BufferCache<MTL::Buffer>` pool + an `MTL::Heap` for <256 B allocations (`allocator.h:49-57`), `ResidencySet` wiring the working set at `0.95 * recommendedMaxWorkingSetSize` (`allocator.cpp:87,98`) | plain `newBuffer` per allocation; no pool, no residency set |
| **Interior pointer -> (buffer, offset)** | `base_offset = a.data<char>() - buffer->contents()` (`device.cpp:238-239`) — pointer arithmetic, because the `array` **carries its owning buffer handle** | an **allocation registry** — because `vt::Tensor::Slice`/`View` hand out a bare interior pointer with no owner |

The last row is the sharpest structural comparison available, and it cuts **in
MLX's favour on elegance but not on portability**: MLX can subtract because its
tensor type is fused to its allocator. Our `vt::Tensor` is deliberately
allocator-agnostic (it is the same type on CPU and CUDA), so a registry is the
correct price of that generality — and Vulkan V1 hit the identical problem and
solved it the identical way (`vulkan_buffers.h`). **The registry is not a Metal
wart; it is the portable answer, now confirmed by two independent backends.**

The **barrier model is a free, architecture-neutral speedup we should take
regardless of the MLX decision**: batching encoders into one command buffer with
hazard-tracked barriers is strictly our own kernels running faster, no dependency
added. Record it as a Metal work item.

---

## 2. The reuse map — ranked

Yardstick per the fan-out spike: the CPU backend proves **64 of 75** ops is a
sufficient target. Corrected counts, verified this session:
**CPU 64/75, CUDA 74/75** (the spike's 73 missed `kDropinProbe`, registered via
`RegisterTypedOp` at `cuda_dropin.cu:291-293`; CUDA's sole gap is the CPU-only
`kMatmulBTQuant`), **Metal 8/75, Vulkan 8/75.**

### Tier A — runs on Apple GPU UNCHANGED, zero new code (the bulk)

| What | Anchor | Why it transfers |
|---|---|---|
| The whole engine (~80 TUs) | `CMakeLists.txt:176-279` | zero device code |
| **All op argument validation** | `src/vt/ops.cpp`, 2,630 lines | every wrapper validates rank/shape/dtype/contiguity/device *before* `GetOp`; canonical `Matmul` `ops.cpp:125-137`. A backend writes **zero** validation |
| Layout/storage contract | `Describe()`, `ops.cpp:45-91` | enforces the layout<->dtype contract device-independently |
| **The entire fusion catalog** | `include/vt/recipes.h`, 9 recipes | Tier-0 composite is ONE device-agnostic walker (`ops.cpp:804-898`); a backend registers `kFusedChain` once (Metal already did) |
| Sampler source | `sampler.cpp`, `ops/penalties.cpp`, `ops/bad_words.cpp`, `logits_processor/builtin.cpp` | **zero CUDA references across all four TUs** |
| Weight loading | `dense_weight_loaders.h:53,62,79,97` | host-side memcpy/transpose. **Needs 0 OpIds** — only `Backend::{Alloc,Free,Copy}` |
| Quant block geometry | `src/vt/dtype.cpp:29-56` | pure `switch(DType)` -> `{block_elems, block_bytes, ggml_type}`, already in `namespace vt` |
| Quant block structs | `src/vt/cpu/cpu_quant_blocks.h:26-94` | pure byte layout + `static_assert`s. Only defect: it sits in `namespace vt::cpu` |
| Keep-quant routing + residency | `gguf_keep_quant.{h,cpp}` | names no device anywhere; `RouteGgufTensor` (`:76-93`) is a pure function |
| 20 of 26 `vt::Backend` virtuals | `src/vt/backend.cpp:19-32` | corrected: **6 pure, 18 defaulted** (24 virtuals + dtor). All 7 async-output primitives already degenerate correctly on unified memory |
| Attention selection | `include/vllm/v1/attention/registry.h:59-78` | selection is data; an unregistered name is skipped |
| f64 test oracle | `tests/vt/test_ops_quant_dot.cpp` | 78,052 assertions, already green on Apple arm64 |

### Tier B — mechanical transcription (math unchanged, loop form changes)

| What | Anchor | Note |
|---|---|---|
| The 7 `DequantQ*` bodies | `src/vt/cpu/cpu_quant_dequant.cpp:48,64,79,125,155,192,229` | scalar portable C++, no SIMD, no `#ifdef`; a `__device__`/MSL transcription is mechanical. `ReadF16`'s `memcpy` becomes a native half load |
| The ~20 elementwise/norm/rope/activation ops | fan-out spike M2 | W0 landed 8; the structural classes are proven |

### Tier C — must be written as Metal kernels (the real work)

Structural classes of the 67 ops Metal lacks: attention family 12, quant 11,
sampler 11, cast/layout 9, MoE 6, dense GEMM 3, GDN/conv 5.

| What | Why it cannot transfer |
|---|---|
| **GEMM** (`kMatmul`, `kMatmulBT`, `kBatchedMatmul`) | tile/MMA structure is hardware-specific. §5 |
| **Paged attention** (`kPagedAttention`, `kReshapeAndCache`) | ditto; and MLX has no paged-KV primitive (§5.3) |
| The 6 `vec_dot` bodies | `cpu_quant_dot.cpp:42-428`. Its own header (`:22-28`) says the `aux8`/`aux16`/`aux32`/`sums[8]` shape exists to coax **host** auto-vectorization — upstream measured a "more natural" rewrite 4x slower. **Actively wrong on a GPU** |
| Activation quantizers + GEMM driver | `cpu_quant_act.cpp` (per-row serial max-scan -> needs a threadgroup reduction); `cpu_quant_gemm.cpp:111-158` (CPU threadpool row chunking) |
| The 10 NVIDIA-specific quant ops | NVFP4/FP8/Marlin/cuBLASLt — no Apple analogue; not owed |

---

## 3. What a first model actually costs (the W0b row, quantified)

### 3.1 Qwen3-dense — **10 ops, 7 of them new**

Forward path calls exactly 9 OpIds: `kCastBf16` (`dense_attn_block.h:277`),
`kEmbedding` (`qwen3.cpp:181`), `kMatmulBT` (`dense_attn_block.h:340-342,466`;
`qwen3.cpp:108,113,225`), `kPagedAttention` (`:445`), `kReshapeAndCache` (`:440`),
`kRmsNorm` (`:397,398`), `kRopeCosSinCache` (`:270`), `kRopeFromCache` (`:402`),
`kSiluAndMul` (`qwen3.cpp:111`); `+kMatmul` (`qwen3.cpp:227`) if `lm_head` is
untied. Greedy sampling adds `kGreedyArgmax`. **Weight loading adds none.**

Metal has 3 of them (`kCastBf16`, `kRmsNorm`, `kSiluAndMul`) ->
**7 new kernels**: `kEmbedding`, `kMatmulBT`, `kPagedAttention`,
`kReshapeAndCache`, `kRopeCosSinCache`, `kRopeFromCache`, `kGreedyArgmax`.

CUDA content *in the Qwen3-dense TU itself*: **2 gates, both quarantined inside
`if (w.IsNvfp4())`** (`qwen3.cpp:84` `#ifdef VT_MARLIN_NVFP4`, `:85` `== kCUDA`).
**`qwen3.cpp:106-231` — the entire BF16 forward — is 100% CUDA-free**, and
`qwen3_dense.cpp` / `qwen3_weights.cpp` have zero CUDA references.

### 3.2 OPT — **9 ops, 6 of them new, and the TU is literally CUDA-free**

`kAdd`, `kEmbedding`, `kLayerNorm`, `kMatmulBT`, `kPagedAttention`, `kQkvSplit`,
`kRelu`, `kReshapeAndCache` (+`kMatmul` if untied, +`kGreedyArgmax`). Metal has
`kAdd`, `kLayerNorm`, `kRelu` -> **6 new kernels**.

**CUDA content across all four OPT TUs: ZERO** — verified case-insensitively; no
`vt/cuda` include, no `DeviceType::kCUDA`, no `#ifdef VT_*`, not even in a
comment. **OPT is the cheapest correct first non-CUDA model** and should be the
bring-up target; Qwen3-dense follows immediately because it is the one MLX also
runs, and the competitor floor needs the arms comparable
(`BACKEND-GATE-METAL-MLXLM`).

### 3.3 The seam fixes — M = 4, and one is a real BUG

| # | Site | Fix |
|---|---|---|
| 1 | `src/vllm/entrypoints/model_loader.cpp:37-39` | hardcoded `GetBackend(kCUDA).CreateQueue()` -> `CurrentPlatform().device_type()` |
| 2 | **`include/vllm/model_executor/models/dense_attn_block.h:140,157`** | **REAL PORTABILITY BUG.** `if (!GetPlatform(...).is_cuda())` selects host-pointer aliasing vs device upload in `ResidentWeight`/`ResidentWeightF32`. A Metal or Vulkan run takes the **host-alias** branch and hands a HOST pointer to a DEVICE kernel. Must become `is_cpu()`-shaped. Found by inspection, not by a test — no test can see it until a model runs |
| 3 | `src/vllm/v1/worker/gpu/runner.cpp:30` | unguarded `#include "vt/cuda/combine_tokens.h"` (+ `is_cuda()` at `:516,690,1068`). The runner is on **every** path |
| 4 | `src/vllm/platforms/metal.cpp:61-65` | `get_attn_backend_priority()` returns `{}`, so `SelectAttentionBackendName` throws. One string, once `kPagedAttention` exists |

**Corrections to the fan-out spike, from re-measurement:**
- CUDA-include leakage is **5 sites, not 4**, but only **3 are strictly
  unconditional** — the public `dense_nvfp4_gemm.h:66` is already inside
  `#ifdef VT_MARLIN_NVFP4` (`:65-67`), so it is *less* severe than the spike
  called it, not more.
- Raw `kCUDA` comparisons are **54 on 53 lines, not 43** — 37 in `qwen3_5.cpp`
  as stated, the rest legitimate registrar/priority keys. **Plus 13
  `platform.is_cuda()` call sites** the spike did not count: the same coupling
  wearing a nicer hat, and item 2 above is one of them.
- `#ifdef VT_*` gates in `qwen3_5.cpp` are **26, not ~25**.

### 3.4 Quant tier — verdict

The fan-out spike's claim **holds**, with one refinement. Block geometry, block
structs, dequant math, keep-quant routing and residency are device-agnostic; the
six `vec_dot` bodies are CPU-tier-only.

The refinement concerns W0b item 7 ("lift `QuantTypeTraits` out of `vt::cpu`").
**It cannot be lifted as-is, because two of its five fields would become lies.**
`QuantTypeTraits` (`include/vt/quant.h:51-60`, confirmed inside `namespace vt::cpu`
at `:36`) mixes encoding facts with *implementation* facts: `vec_dot_type`
(`:57`) and `nrows` (`:59`) describe what a particular kernel eats, not what the
encoding is — `nrows` is already ISA-conditional (`quant.h:25-28`,
`cpu_quant_traits.cpp:14-18`), and `vec_dot_type` is Q8_K only because that is
what the *CPU* kernel eats. MLX's own quantized path confirms this is a real
axis, not a hypothetical: it is parameterized on `(group_size, bits, mode)` with
`mode` a runtime string (`quantized.cpp:19-37`) selecting `affine` vs `mxfp4`
families, and `bits ∈ {2,3,4,5,6,8}` with its own `get_pack_factor()` /
`get_bytes_per_pack()` (`quantized.h:17-25`) — **a completely different block
convention from GGUF k-quant**. An MLX quant kernel could not consume our
Q4_K blocks without a repack.

**Correct split:** keep the device-neutral `vt::BlockGeometry` (already exists,
`dtype.cpp:29`); move `{from_float, to_float, vec_dot, vec_dot_type, nrows}` into
a **per-(DeviceType, provider) traits table keyed exactly like the arch-tactic
registry**, with `vt::cpu::QuantTypeTraits` as its CPU row, unchanged. Two call
sites make this cheap: `gguf_keep_quant.cpp:71`
(`vt::cpu::HasQuantDotKernel(dt)` -> `HasQuantDotKernel(dt, device)`) and the
`cpu_quant_gemm.cpp` path selection. `HasQuantDotKernel`
(`cpu_quant_traits.cpp:83-88`) is *already written as a two-sided capability
query* — precisely the `ArchTacticSupportsFn` predicate shape, one layer up.
**The quant tier and the tactic registry want the same seam.** That is the
strongest single piece of evidence for §6.

---

## 4. Judging our current structure against "extend acceleration easily"

### 4.1 Adding a PLATFORM: solved, and proven twice

| Enabler | Anchor |
|---|---|
| Open device enum | `include/vt/device.h:16-17` — `{kCPU,kCUDA,kMETAL,kVULKAN,kXPU}`, `kNumDeviceTypes=5`; untouched by both W0 commits |
| Priority already lists all five | `src/vllm/platforms/platform.cpp:38-40` |
| Every kernel takes `Queue&` first | `include/vt/ops.h:439-602` — no ambient stream anywhere |
| Partial backends are a supported state | `ops.cpp:104-111` throws; `OpRegistered` (`:119-123`) probes |
| Platform declines to register when absent | `metal.cpp:75-81` guards on `MetalDeviceAvailable()` so `CurrentPlatform()` falls through to CPU |

**Measured additivity.** Vulkan V1 added 11 new files + 8 shaders and edited
**exactly two pre-existing files** — `CMakeLists.txt` (+49) and
`tests/CMakeLists.txt` (+8). **Zero edits to any pre-existing `src/` or
`include/` source file.** Metal W0's extra edits were a shared seam publication
(`OpRegistered`) and 7 real `-Werror`/UB fixes surfaced by building on macOS —
incidental, not scatter. `tests/vt/test_backend_cross_device.cpp` was **not
modified** to pick Vulkan up; it already enumerates every registered non-CPU
backend. **The PR-#4 additivity property holds, demonstrated rather than
asserted.** Marginal cost of platform #6: one CMake block, one test row, and 2
lines in `device.h`.

### 4.2 Adding a PROVIDER within a platform: **NOT solved. This is the gap.**

```cpp
// src/vt/ops.cpp:10-15
using OpTable = std::array<std::array<void*, kNumDeviceTypes>, OpId::kCount>;
// :98-102
void RegisterOp(OpId op, DeviceType device, void* fn) { ... Table()[op][device] = fn; }
```

One `void*` per `(OpId, DeviceType)`. **Last writer wins, silently** — no check,
no warning. Static-init order across TUs is unspecified, so "register MLX's
matmul alongside ours" is a *nondeterministic build*, not a configuration. The
contract is stated outright at `include/vt/ops.h:47-55`: "Backends register
**one** kernel per (OpId, DeviceType)". `vt::Backend` and `Platform` have the
same shape (`backend.cpp:40-68`, `platform.cpp:25-28`).

There is no priority, no capability predicate, no shape-based decline, and no
provenance. Everything the seam needs is missing — **except that we already
built it once, one layer down.**

### 4.3 The three precedents, judged

**(a) `cuda_arch_tactics` — the right shape, in the wrong place.**
`src/vt/cuda/cuda_arch_tactics.{h,cu}`. Per-family fixed-capacity table
(`kMaxTacticsPerFamily = 8`, `.cu:69-86`), no dynamic allocation so
static-init order is safe (`.cu:64-68`); registration is table-fill only and
never throws (`:126-131`); selection is a linear scan in registration order,
**first `supports(caps)` wins, `nullptr` on no match so the caller falls back to
its portable path** (`:145-170`); doubly type-erased —
`ArchTacticSupportsFn = bool(*)(const DeviceCaps&)` (`.h:83`) and
`ArchTacticLaunchFn = bool(*)(const DeviceCaps&, void* args)` (`.h:84`), where
the `bool` return is a **second fallback axis: "I decline this shape"**;
instrumented with `selections`/`fallbacks`/`last_selected` and a
`VT_ARCH_TACTIC_STATS=1` line (`.h:108-120`).

**Its CUDA couplings are exactly 3 and all shallow**, as the fan-out spike
claimed — verified: (1) `.h:44` includes `cuda_device_caps.h`, the only real one
(`DeviceCaps` is `sm_major`/`sm_minor`/`multiprocessor_count`/... — CUDA-shaped);
(2) `.h:47` `namespace vt::cuda`, pure placement; (3) `.h:66`
`void* stream = nullptr; // cudaStream_t, erased to keep this header CUDA-free`
— already erased, and the comment admits the semantic coupling. The header
includes no CUDA header, names no `cuda*_t`, and has no `__CUDA_ARCH__`.
**~95% portable as-is.** Its only weakness is population: exactly one tactic is
registered (`.h:20-22` says so) — the shape is proven, the exercise is thin.

**(b) The fusion framework — the right *philosophy*, already generalized.**
`include/vt/recipes.h` declares 9 recipes; adding one touches `recipes.h` + a
test and **zero `src/vt/` files**. The `FusedRecipe` POD
(`fused_recipe.h:146-153`) has no `vt::` dependency at all. Dispatch is already a
**three-rung fallback ladder** (`ops.cpp:963-995`): bespoke fast op if
`OpRegistered(fast_op, device)` -> Tier-1 interpreter if `RecipeIsTier1Able` and
`VT_FUSED_TIER=1` -> Tier-0 composite, always available. That ladder is
*exactly* the provider-selection semantics we want, and it already exists —
just hardcoded for one axis instead of generalized.
*(Record correction: `test_fused_chain_additivity.cpp:420-439` gates only **7**
of the 9 — `kFusedAddRmsNormStd` and `kAttnQkNormRope` were added to `recipes.h`
without catalog rows, so the `CHECK(... == 7)` count guard has drifted. Repair
owed.)*

**(c) `DeviceResourceOps` — a C-ABI provider table that already exists.**
`backend.h:114-138`: a 4-function-pointer table (`alloc`/`free`/`create_queue`/
`destroy_queue`) registered per `DeviceType`, with the free functions
(`backend.cpp:71-114`) preferring the table and falling back to a `Backend`
shim. This is the closest existing thing to a pluggable external-library table —
and it proves the pattern is already accepted in-tree.

**(d) `dropin-kernel-abi.md` (`BACKEND-ABI-VT`) — a provider seam for ONE
provider shape.** It reshapes the `vt::` adapter boundary around raw
pointer/shape/stride/workspace/stream contracts so an upstream vLLM `csrc/`
Layer-B dispatcher binds unchanged. **It is genuinely an acceleration-provider
seam** — for libraries with a *raw C launcher* (CUTLASS, Marlin, cuBLASLt,
flashinfer, and by design ROCm). **It does not cover MLX**, whose entry points
take `const array&`, `metal::Device&` and `Stream` — a C++ *object model*, not
raw pointers. That is the precise gap: we have a provider ABI for C launchers
and none for object-model libraries.

### 4.4 Verdict

> **A new accelerated PLATFORM plugs in through what exists — that is measured,
> not projected. A new PROVIDER within a platform does not, and cannot without a
> new mechanism. We should build the explicit acceleration-provider seam.**

---

## 5. MLX as a compute path — the impedance question, answered from source

`backends.md:84-90` flags MLX's "lazy-evaluation graph semantics ... same
impedance class that disqualified ggml". **Read against the source, that framing
is wrong**, and the burden is now to show why.

### 5.1 The lazy graph sits strictly ABOVE an eager kernel layer

The graph machinery (`array::eval()` `array.h:137`, the DFS tape in
`transforms.cpp:87-160`, `async_eval` `:290`, sibling/donation tracking) all
terminates at one call:

```cpp
// mlx/backend/metal/eval.cpp:32-48
void eval(array& arr) {
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);
  ...
  arr.primitive().eval_gpu(arr.inputs(), outputs);
}
```

`Primitive::eval_gpu` is an **eager, per-op, encode-into-the-stream's-command-
buffer** call. Below that line there is no laziness, no tape, no autodiff, no
graph.

And the compute entry points are **free functions**, not methods on the graph:

```cpp
// mlx/backend/metal/matmul.h:105-142
inline void steel_matmul(const Stream& s, metal::Device& d,
    const array& a, const array& b, array& out,
    int M, int N, int K, int batch_size_out, int lda, int ldb,
    bool transpose_a, bool transpose_b, std::vector<array>& copies, ...);
```

`Matmul::eval_gpu` (`matmul.cpp:818`) is merely *one caller* of it. Others exist
for AddMM, GatherMM, SegmentedMM. **We can be another caller** — `steel_matmul`
never touches the tape.

The same holds for the kernels we would actually want:
`sdpa_full_self_attention_metal` and `sdpa_vector` /`sdpa_vector_2pass`
(`scaled_dot_product_attention.cpp:16,151,240`), and the quantized family
`qmv`/`qmm`/`qvm`/`gather_qmm` (`quantized.cpp:165,223,405,458,533`) — all free
functions taking `(Stream, Device&, arrays..., ints...)`.

### 5.2 The zero-copy bridge is real, and it is three facts wide

1. `allocator::Buffer` is a **bare `void*` wrapper** (`mlx/allocator.h:12-29`),
   and for Metal that `void*` **is** the `MTL::Buffer*` (`allocator.cpp:146-149`).
2. `array::set_data(allocator::Buffer, size_t data_size, Strides, Flags, Deleter)`
   (`array.h:417-422`) constructs an array over an **existing** buffer with a
   caller-supplied deleter — pass a no-op and MLX never owns our memory.
3. `set_input_array` computes the binding offset as
   `a.data<char>() - buffer->contents()` (`device.cpp:238-239`). Our Metal
   buffers are already `StorageModeShared`, so an interior `vt::Tensor::Slice`
   pointer lies inside `contents()` and **this arithmetic is correct on our
   tensors unmodified**.

So the boundary is: wrap our `MTLBuffer` -> `array::set_data` with a no-op
deleter -> call the free function -> `end_encoding` + commit + wait on our
`vt::Queue`. **An explicit eval/sync boundary per op — no graph, no copy, no
allocator surrender.**

### 5.3 What it costs, honestly

| Cost | Detail |
|---|---|
| **Dependency weight** | `libmlx.dylib` + a **104,894,650-byte `mlx.metallib`**. `discipline.md:86-92` prefers header-only deps. This is a real deviation and must be a **gated, optional** provider — precedent: `VLLM_CPP_TRITON` (`discipline.md:58`) is already accepted as exactly that |
| **Build** | building MLX from source needs `xcrun metal` (`kernels/CMakeLists.txt:28`) i.e. **full Xcode, which the M4 does not have** — so we would consume the prebuilt wheel artifacts, or set `MLX_METAL_JIT` |
| **Sync tax** | one command-buffer commit+wait per delegated op unless we share a command buffer with MLX's stream. Real, and **must be MEASURED, not assumed** — it is precisely the "eager-wrapper tax" `backends.md:97` names |
| **Numerics** | MLX pins `setFastMathEnabled(false)` (`device.cpp:512`) — same posture as our `MTLMathModeSafe`, so the NMSE <= 5e-4 bar is plausible. But it is a **different reduction order**, so bit-exactness across providers is not on offer (already our stated position, fan-out spike § Gates) |
| **Coverage gap** | **MLX has NO paged-KV attention.** `ScaledDotProductAttention::use_fallback` (`scaled_dot_product_attention.cpp:375-412`) gates on contiguous q/k/v and head-dim; there is no block-table concept anywhere in the file. `kPagedAttention` and `kReshapeAndCache` **must be ours regardless.** MLX wins on GEMM, not on our KV cache |
| **Quant mismatch** | MLX's `(group_size, bits, mode)` convention (`quantized.h:17-25`) is not GGUF k-quant. Delegating quant GEMM needs a repack |

### 5.4 Recommendation

> **MLX is VIABLE as a compute path for GEMM (and plausibly non-paged
> attention), and the lazy-eval objection does not survive contact with the
> source — the graph is above the kernel layer and we can enter below it. But it
> must be an OPTIONAL, GATED PROVIDER behind the seam, never the bring-up path
> and never a hard dependency.** Native MSL stays the default: it needs zero
> installs, W0 already works, and `kPagedAttention` has to be ours either way.
> MLX earns its place only where it MEASURES faster than our own kernel — which
> is exactly what a provider seam with a capability predicate and a decline
> return is for.

---

## 6. The proposed seam — `vt::OpProvider`

**Design: make `OpTable` a `FamilyTable`.** The arch-tactic registry already has
every property the op table lacks. Generalize it out of `vt::cuda`, key it on
`(OpId, DeviceType)`, and let `GetOp` select.

```
// sketch — the shape, not the final API
struct OpProvider {
  const char*      name;        // "vt-msl", "mlx", "cublaslt", "llamacpp"
  int              priority;    // ties broken by registration order
  bool (*supports)(const DeviceCaps&, const OpShape&);   // cheap, side-effect free
  void*            fn;          // the existing kernel pointer, unchanged
};
```

- **Capacity-bounded static storage**, no dynamic allocation, registration is
  table-fill only — copied verbatim from `cuda_arch_tactics.cu:64-86,126-131`,
  which is what makes static-init order safe.
- **Selection = the existing linear scan**, `cuda_arch_tactics.cu:145-170`:
  highest priority whose `supports()` is true; `nullptr` -> caller's portable
  path. The kernel may still **decline by return value** — the second fallback
  axis already in `ArchTacticLaunchFn`.
- **Instrumentation is not optional.** Port `selections`/`fallbacks`/
  `last_selected` + a `VT_OP_PROVIDER_STATS=1` dump (`.h:108-120`). Without it,
  "did MLX actually run?" is unanswerable — and the fan-out spike's Risk 4
  (a probe failing *silently* into the slow path) is exactly this failure mode.
- **`DeviceCaps` must be generalized** to a device-neutral capability struct
  (the one genuine CUDA coupling, `.h:44`). This IS fan-out spike W0b item 8
  (`include/vt/arch_tactics.h`) — **so W0b item 8 and this seam are the same
  work, and should be done once.**

**Why this is cheap:** all ~70 op entry points in `ops.cpp` are the same shape —
validate, then `reinterpret_cast<XFn>(GetOp(OpId::kX, q.device.type))(...)`
(`:136`, `:180`, `:784`, `:1103`, `:2223`). A seam inserted at `GetOp` is picked
up by **every op at once, with zero call-site edits.** Estimate: ~50 lines in
`ops.cpp` + the generalized capability header.

### 6.1 It generalizes — the test the user set

| Platform | Provider it would delegate to | Entry shape | Covered? |
|---|---|---|---|
| Metal | **MLX** `steel_matmul` / `sdpa_*` / `qmm` | C++ objects (`array`, `Device&`) | **yes — this is the new capability** |
| CUDA | cuBLASLt / CUTLASS / flashinfer / Marlin | raw C launcher | yes — and `dropin-kernel-abi.md` already specifies the *argument* half; this supplies the missing *selection* half |
| CUDA | the existing `sm_12x` fp4 tactic | raw | yes — it is the registry this is generalized FROM |
| Vulkan | llama.cpp-style coopmat/coopmat2 shaders | our own dispatch | yes — fan-out spike V3 explicitly wants "the coopmat tactic via the W0-generalized arch-tactic registry" |
| CPU | llama.cpp `vec_dot` / repack (G5/G6 tiers) | raw C | yes — and the quant traits split (§3.4) keys on the same predicate |

**Five platforms, four provider shapes, one mechanism.** It is not an MLX seam;
MLX is one row.

### 6.2 Ranked W-plan for the Metal model bring-up

| ID | Work | Depends | Why here |
|---|---|---|---|
| **W0b-1** | Seam fixes 1-4 of §3.3 — **incl. the `dense_attn_block.h:140,157` host-pointer BUG** | — | blocks ANY model on ANY non-CUDA backend. The bug is a correctness defect, not portability polish |
| **W0b-2** | `vt::OpProvider` + device-neutral `DeviceCaps` (= old W0b item 8) | W0b-1 | do it BEFORE three backends each invent one. Behavior-preserving: register every existing kernel as a single priority-0 provider whose `supports()` returns true |
| **W0b-3** | Split `QuantTypeTraits` per §3.4 (= old W0b item 7, corrected — split, do not lift) | W0b-2 | keys on the same predicate |
| **M2r** | M2 residue: the ~12 remaining elementwise/rope/gated ops | W0b-1 | |
| **M3a** | **OPT on Metal** — 6 new kernels, incl. `kMatmulBT` + `kPagedAttention` | M2r | the only CUDA-free model TU in the tree; cheapest correct first model. Gate: token-exact vs our own CPU backend on the same M4 |
| **M3b** | **Qwen3-dense on Metal** — +`kRopeCosSinCache`, `kRopeFromCache`, `kRmsNorm` path | M3a | the model MLX also runs -> unlocks `BACKEND-GATE-METAL-MLXLM` against §7 |
| **M3c** | Batched encoders + hazard barriers (§1.3), replacing one-command-buffer-per-op | M3a | pure win for OUR kernels, no dependency. Port `device.cpp:264-273` |
| **M5** | **MLX as a registered provider** (`VLLM_CPP_MLX`, default OFF) for `kMatmul`/`kMatmulBT`, measured A/B against M3's own MSL via `VT_OP_PROVIDER_STATS` | W0b-2, M3b | this is where the §5 bridge is built, and it is now a *configuration*, not a rewrite. Adopt only where it measures faster |
| **M4** | GGUF k-quant Metal shaders | M3b, W0b-3, `QUANT-GGUF-CIQ-GEMM` G4 | |

---

## 7. MLX baseline — **UNOPPOSED FLOOR, NOT a parity claim**

**We have no Metal model. There is no "ours" column and none is manufactured.
No Metal speed result is claimed anywhere in this document.** These are MLX's
own numbers, produced by MLX's own harness, recorded so the Metal bring-up has a
target to design against.

### 7.1 Exact repro recipe (reproduction is a gate)

| Item | Value |
|---|---|
| Box | Apple M4, 16 GiB unified, macOS 26.5.2 (25F84), `ssh 192.168.68.103` |
| GPU | `applegpu_g16g`, `max_recommended_working_set_size` 12,713,115,648 B (11.84 GiB), `max_buffer_length` 9,534,832,640 B |
| Install | `/usr/bin/python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm` — **venv route, brew deliberately NOT used** (it would put `python@3.14` first on the PATH our macOS builds use and change `find_package(Python3)`) |
| MLX | **`mlx` 0.29.3, `mlx-metal` 0.29.3, `mlx-lm` 0.29.1** (Python 3.9.6 CLT caps the resolve below brew's 0.32.0 — recorded, since an unpinned competitor arm is not a floor) |
| Model | `mlx-community/Qwen3-1.7B-bf16`, revision **`9cd6692855d3e06772228e9a962b2606359b2d24`** |
| Harness | **MLX-LM's own** `python -m mlx_lm.benchmark` (`mlx_lm/benchmark.py`) — their tool, their definitions |
| Command | `HF_HOME=$HOME/hf-cache ~/mlx-venv/bin/python -m mlx_lm.benchmark --model mlx-community/Qwen3-1.7B-bf16 -p 512 -g 128 -b <B> -n 3` |
| Workload | 512 random prompt tokens, 128 generated, EOS disabled (`benchmark.py:118`), seed `mx.random.seed(0)`, 1 warmup + 3 timed trials |
| Vllm.cpp commit | `1cb5f64` (no code change; the tree was not built for this) |

**Model-size choice.** Qwen3-**1.7B** bf16, not 4B/8B: the box shows ~13 GB of
17.2 GB already in use by the desktop session, and the fan-out spike's HONEST
LIMIT 7 stands — the M4 cannot hold the 27B/35B gate models and **no Apple
result may ever be extrapolated to them.** 1.7B leaves headroom for a b=16 arm
without paging, which a larger model would not.

### 7.2 Measured — MLX 0.29.3, Qwen3-1.7B-bf16, p=512 g=128

Direct outputs (mean of 3 trials; `prompt_tps` and `generation_tps` are
**aggregate over the batch**, `benchmark.py` -> `generate.py:1657,1660`):

| B | prompt_tps | generation_tps | peak_memory (GB) | trial spread |
|---:|---:|---:|---:|---|
| 1 | 1089.92 | **27.57** | 3.776 | gen 0.12%, prompt 0.20% |
| 2 | 1137.45 | 48.91 | 3.974 | gen 0.25% |
| 4 | 1178.67 | 90.15 | 4.184 | gen 0.63% |
| 8 | 1199.06 | 156.95 | 4.466 | gen 0.36% |
| 16 | 1194.64 | **213.39** | 5.279 | gen 0.14% |

Derived per-axis figures (the axes AGENTS.md § Acceptance rule binds on):

| B | TTFT = B·512/prompt_tps | ITL = B/generation_tps | out tok/s per req | out tok/s aggregate | peak mem |
|---:|---:|---:|---:|---:|---:|
| 1 | **470 ms** | **36.3 ms** | 27.57 | 27.57 | 3.78 GB |
| 2 | 900 ms | 40.9 ms | 24.45 | 48.91 | 3.97 GB |
| 4 | 1,738 ms | 44.4 ms | 22.54 | 90.15 | 4.18 GB |
| 8 | 3,416 ms | 51.0 ms | 19.62 | 156.95 | 4.47 GB |
| 16 | 6,857 ms | 75.0 ms | 13.34 | **213.39** | 5.28 GB |

**Reading.** Prefill saturates at **~1,200 tok/s** by B=8 and does not improve
(B=16 is marginally *lower*) — a compute roof. Decode scales **7.74x from B=1 to
B=16**, the signature of a bandwidth-bound decode amortizing weight traffic
across the batch. Peak memory grows only 1.40x over a 16x batch increase — the
1.7B bf16 weights (~3.4 GB) dominate and the KV cache is small at 512+128.

**Reproducibility.** Trial spread is **0.12%–0.63%** across all five arms. That
is well inside run-noise and is itself evidence that the box was not meaningfully
contended for this workload (see §8).

### 7.3 Status of these numbers

**`BLOCKED-ON-SUDO` — INDICATIVE, NOT BINDING.** The `com.localai.worker` root
LaunchDaemon could not be stopped (§8). Per the standing rule that a contended
run is void, **these figures may not be cited as a binding competitor floor.**
They are recorded as the best available *design target* and as the exact recipe
to re-run once the daemon is down. The re-run is a one-command repeat of §7.1
and should be done before `BACKEND-GATE-METAL-MLXLM` binds.

What they are sufficient for **today**: they answer §5's cost question. MLX
delivers 27.6 tok/s single-stream on a 1.7B bf16 model on this box. Whether
delegating GEMM to MLX is worth the 105 MB dependency is a question our own M3
MSL GEMM will answer by measurement against this line — which is exactly why the
provider seam (§6) must carry `VT_OP_PROVIDER_STATS` instrumentation.

---

## 8. LocalAI worker — disposition

**NOT STOPPED. `sudo` requires a password on the M4 and none is available to an
agent** (`sudo -n true` -> `sudo: a password is required`, measured 2026-07-22).
No number above is presented as binding, and none was fabricated.

**Measured disposition of the daemon** (`launchctl print system/com.localai.worker`,
`ps`, `ioreg`):

| Fact | Value |
|---|---|
| PID / state | 327, `running`, `keepalive`+`runatload`, `/Library/LaunchDaemons/com.localai.worker.plist` |
| CPU | **0.0%** |
| RSS | 51,824 KB (~51 MB, 0.3% of 16 GiB) |
| Uptime | 1d 08:36 |
| GPU | `IOAccelerator PerformanceStatistics`: **`Device Utilization % = 0`, `Renderer Utilization % = 0`, `Tiler Utilization % = 0`** — it holds no GPU work |
| Activity | `worker.log` shows only `NATS backend.list` events, ~1/6 h |

**It is genuinely idle and holds no GPU.** That is consistent with the 0.12%
trial spread. It is nonetheless *up*, so the standing rule applies and the
numbers stay non-binding.

**Note a second contender the brief did not anticipate**, found while
investigating load average 1.47: `WallpaperAerialsExtension` (PID 472, 8.2% CPU)
and `VTDecoderXPCService` (PID 518, 2.2%) — the desktop **aerial video
wallpaper**, which decodes video continuously and touches the GPU. For a binding
run this should be disabled too (System Settings -> Wallpaper, or log the console
user out). It was left untouched.

**Commands for the user (both required before a binding run):**

```sh
ssh 192.168.68.103
sudo launchctl bootout system/com.localai.worker            # stop
# ... run §7.1 ...
sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist   # RESTORE
```

The box was left exactly as found: nothing stopped, nothing killed, nothing
uninstalled. Added: `~/mlx-venv` (venv, off the build PATH) and `~/hf-cache`
(3.2 GB model cache) — both removable with `rm -rf`, neither on any PATH our
builds consult.

---

## 9. Corrections owed to other records

Landed with this spec:

1. Fan-out spike: CUDA op coverage **74/75, not 73** (`kDropinProbe` via
   `RegisterTypedOp`, `cuda_dropin.cu:291-293`); `vt::Backend` is **6 pure + 18
   defaulted (24 virtuals), not 6 + 20 of 26**; raw `kCUDA` comparisons **54, not
   43**, plus **13 uncounted `is_cuda()` call sites**; CUDA-include leakage **5
   sites of which 3 strictly unconditional**, and the public
   `dense_nvfp4_gemm.h:66` is already `#ifdef`-guarded (less severe than stated);
   `#ifdef VT_*` in `qwen3_5.cpp` **26**.
2. W0b item 7 restated: `QuantTypeTraits` must be **SPLIT, not lifted** (§3.4).
3. W0b item 8 and this seam are **the same work** (§6).
4. **Open defect, not fixed here:** `test_fused_chain_additivity.cpp:420-439`
   gates 7 of the 9 declared recipes; the `CHECK(... == 7)` count guard whose
   stated purpose is "tie catalog growth to test growth" has drifted. Repair owed
   under the fusion row.
5. **Open defect, not fixed here:** `dense_attn_block.h:140,157` hands a HOST
   pointer to a DEVICE kernel on any non-CUDA device backend (§3.3 item 2).
   Latent today because no model runs on one; a hard blocker for M3.
