# Metal/MLX reuse study + the acceleration-provider seam (`BACKEND-ACCEL-PROVIDER`)

Status: **STUDY + MLX BASELINE + TWO MODELS RUNNING (OPT M3a, Qwen3-dense M3b) + FIRST OURS-VS-MLX BENCHMARK, 2026-07-23.** §13 records `M3b` — Qwen3-dense on Apple GPU: the forward is ORACLE-CONFIRMED CORRECT (near-tie-robust — vLLM teacher-forced on the Metal prefix, all divergences within 0.5 nats, max 0.125), gated 16/16 against Metal's own oracle golden; NOT strict-token-exact (0.6B is a near-tie model, a strict gate needs Qwen3-4B — deferred). The corrected op count (3 new RoPE kernels, Metal 18/75) and the INDICATIVE ours-vs-MLX numbers stand. The CUDA gate is GREEN (the France/Italy tie is build-sensitive; production build → France 9625, portable-only → Italy — the "stale golden" claim is disproven).
Earlier status line: **STUDY + MEASURED MLX BASELINE + FIRST MODEL RUNNING, 2026-07-22.**
Work rows `W0b-2` (the `vt::OpProvider` seam), `W0b-1`, `M5` (MLX as a registered
provider for the dense GEMM) and — since 2026-07-22 — **`M3a`: OPT-125m running
END TO END ON APPLE GPU, STRICT token-exact 6/6 (96/96 tokens) against the
committed vLLM 0.25.0 goldens** are LANDED AND GATED, together with a native MSL
GEMM that `M5` needed in order to have anything to be an alternative *to*.
§10 records the seam work; **§12 records `M3a`** — what was implemented, which of
this document's predictions held, the one it got wrong, and the one seam bug it
did not anticipate. **No Metal SPEED number is claimed or owed** (§12.5).
Owners: `CLAIM-BACKEND-FANOUT-1` (the study) and
`CLAIM-BACKEND-ACCEL-PROVIDER-1` (the implementation).
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

> **CORRECTION FROM THE IMPLEMENTATION (2026-07-22).** The *input* half of this
> is exactly right and is what shipped: `array::set_data` over our own
> `MTL::Buffer*` with a no-op deleter works, and MLX never owns our memory. The
> *output* half does not survive contact with the shipped wheel. Two measured
> facts: (a) `nm -gU libmlx.dylib` shows **`steel_matmul_axpby<false>` is not an
> exported symbol** — only the `get_steel_*_kernel` helpers are — so the free
> function above is declared in the installed header but is **not linkable**
> against the prebuilt artifact; (b) even if it were, `Matmul::eval_gpu`
> re-`set_data`s its output from MLX's own allocator, so a pre-bound output
> buffer would be discarded. The provider therefore calls the exported
> `mlx::core::matmul` and `memcpy`s the result into our tensor — a host copy of
> M*N elements against an O(M*N*K) GEMM on unified memory. Real, cheap, and
> **not** to be described as zero-copy. One further practical detail the study
> could not have known: an `array` built this way starts `unscheduled`, so it
> must be marked `Status::available` or MLX's evaluator throws
> "Attempting to eval an array without a primitive".

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
| **W0b-1** (PARTLY LANDED) | Seam fixes 1-4 of §3.3 — **incl. the `dense_attn_block.h:140,157` host-pointer BUG** | — | blocks ANY model on ANY non-CUDA backend. The bug is a correctness defect, not portability polish |
| **W0b-2** (LANDED 2026-07-22) | `vt::OpProvider` + device-neutral `DeviceCaps` (= old W0b item 8) | W0b-1 | do it BEFORE three backends each invent one. Behavior-preserving: register every existing kernel as a single priority-0 provider whose `supports()` returns true |
| **W0b-3** | Split `QuantTypeTraits` per §3.4 (= old W0b item 7, corrected — split, do not lift) | W0b-2 | keys on the same predicate |
| **M2r** | M2 residue: the ~12 remaining elementwise/rope/gated ops | W0b-1 | |
| **M3a** (LANDED 2026-07-22) | **OPT on Metal** — 5 new kernels (`kMatmulBT` had already landed with the seam), plus `W0b-1` items 1/3/4 and one NEW seam bug the study did not predict | M2r | the only CUDA-free model TU in the tree; cheapest correct first model. **Gate MET, and at a STRONGER bar than this row asked for**: the row proposed token-exactness vs our own CPU backend on the M4; what shipped is token-exactness vs the COMMITTED dgx-captured vLLM 0.25.0 goldens — 6/6 prompts, 96/96 tokens — because those goldens are device-INDEPENDENT (they are vLLM's tokens, not ours), so Metal had to meet the bar CUDA already met rather than one re-derived on Metal. See §12 |
| **M3b** | **Qwen3-dense on Metal** — +`kRopeCosSinCache`, `kRopeFromCache`, `kRmsNorm` path | M3a | the model MLX also runs -> unlocks `BACKEND-GATE-METAL-MLXLM` against §7 |
| **M3c** | Batched encoders + hazard barriers (§1.3), replacing one-command-buffer-per-op | M3a | pure win for OUR kernels, no dependency. Port `device.cpp:264-273` |
| **M5** (LANDED 2026-07-22, gated `VLLM_CPP_MLX`) | **MLX as a registered provider** (`VLLM_CPP_MLX`, default OFF) for `kMatmul`/`kMatmulBT`, measured A/B against M3's own MSL via `VT_OP_PROVIDER_STATS` | W0b-2, M3b | this is where the §5 bridge is built, and it is now a *configuration*, not a rewrite. Adopt only where it measures faster |
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
4. ~~**Open defect, not fixed here:**~~ **FIXED 2026-07-22.**
   `test_fused_chain_additivity.cpp` gated 7 of the 9 declared recipes; the count
   guard whose stated purpose is "tie catalog growth to test growth" had drifted.
   `kFusedAddRmsNormStd` and `kAttnQkNormRope` now have byte-exact CPU drivers in
   the catalog and the guard reads 9.
5. ~~**Open defect, not fixed here:**~~ **FIXED 2026-07-22.**
   `dense_attn_block.h:140,157` handed a HOST pointer to a DEVICE kernel on any
   non-CUDA device backend (§3.3 item 2). Both `ResidentWeight` and
   `ResidentWeightF32` now test `is_cpu()` rather than `!is_cuda()` — host-pointer
   aliasing is a property of the host, not of "not NVIDIA". kCPU and kCUDA
   behaviour is bit-identical, evidenced by the unchanged dgx regression set.

---

## 10. What actually landed (2026-07-22, `CLAIM-BACKEND-ACCEL-PROVIDER-1`)

Rows this section is the spike for: **`BACKEND-ACCEL-PROVIDER`** (the seam as a
cross-backend platform concern) and **`KERNEL-ACCEL-PROVIDER-SELECT`** (the same
seam viewed as the selection layer sitting above every kernel family), plus the
already-owned `BACKEND-METAL-MLX`. The full spike contract for both — upstream
surface, dispatch rules, files to port, tests, hardware, gates, dependencies and
the row-sized work breakdown — is §4 (the structural gap), §5 (MLX as a compute
path and its costs), §6 (the design and its five-platform generalization test),
§6.2 (the ranked work breakdown) and the table below (what landed, what is owed).

**Landed and gated:**

| Row | What | Where |
|---|---|---|
| `W0b-2` | **`vt::OpProvider`** — the seam. Capacity-bounded static storage, device-neutral `ProviderCaps`, capability predicate, per-call decline-and-fall-back (`GetOpFallback`), selection stats, `VT_OP_PROVIDER_STATS` / `VT_OP_PROVIDER_DISABLE`. Selection is `(priority DESC, name ASC)` — **deterministic**, which was the actual defect. `RegisterOp`/`GetOp`/`OpRegistered` kept their exact signatures and merely changed TU, so **all ~70 op wrappers are byte-unchanged** | `include/vt/op_provider.h`, `src/vt/op_provider.cpp`, `tests/vt/test_op_provider.cpp` |
| `W0b-1` (part) | The `dense_attn_block.h` **host-pointer bug**. The other three seam fixes (model-loader queue, runner include, Metal attn priority) are still owed and are only reachable once a model runs | `include/vllm/model_executor/models/dense_attn_block.h` |
| new | **Native MSL dense GEMM** `kMatmul`/`kMatmulBT`. Not in the original plan; required, because a seam that selects between providers needs a second provider to select. Stays the DEFAULT | `src/vt/metal/metal_msl.h`, `src/vt/metal/metal_ops.mm` |
| `M5` | **MLX as a registered provider** for the dense GEMM, `VLLM_CPP_MLX` default OFF | `src/vt/metal/metal_mlx_provider.mm` |

**Still owed as of that change:** `W0b-1` items 1/3/4, `W0b-3` (the
`QuantTypeTraits` split, which keys on the same predicate the seam now provides),
`M2r`, `M3a` (OPT on Metal — still the right first model), `M3b`, `M3c`, `M4`.

> **SUPERSEDED IN PART, 2026-07-22.** `M3a` LANDED and with it `W0b-1` items 1
> and 4; item 3 was found to be a NON-ISSUE by measurement (the `vt/cuda/`
> include in the runner is declaration-only and compiles and links on a
> Metal-only macOS build — the study's "strictly unconditional" count was
> correct about the include and wrong about its consequence). **A model now RUNS
> on Metal.** Still no Metal SPEED number is claimed or owed, and
> `BACKEND-GATE-METAL-MLXLM` correctly stays `INVENTORIED` — it binds on
> Qwen3-dense (`M3b`), which is the model MLX also runs. See §12.

**§6.1's generalization claim, re-judged against the implementation.** Of the
five rows in that table, exactly one is now POPULATED (Metal/MLX) and the other
four are DESIGNED FOR but empty. What the implementation *does* settle is that
the object-model shape — the row `dropin-kernel-abi.md` could not express — fits
the same `OpProvider` struct a raw-C launcher uses, with the C++ objects confined
entirely to the provider's own TU. That was the open question; it is answered.
Whether cuBLASLt, llama.cpp and coopmat fit as cleanly is a claim this change
does not yet get to make.

---

## 11. Structured spike contract — `BACKEND-ACCEL-PROVIDER` / `KERNEL-ACCEL-PROVIDER-SELECT`

The narrative above is the study. This section is the same material in the
canonical spike shape, so the two rows this document now owns are grounded
field-by-field rather than by cross-reference.

### Scope

One mechanism by which **two or more implementations of a single `vt::` op on a
single `DeviceType` coexist**, with a deterministic, observable, per-call-
refusable selection between them. In scope: the registry, the selection rule,
the device-neutral capability record, the decline-and-fall-back path, the
instrumentation, and the first two providers (`vt-native`, `mlx`). Explicitly
OUT of scope: any change to op semantics, argument validation, or any op
wrapper body; any model; `kPagedAttention`/`kReshapeAndCache` delegation (MLX
has no paged-KV primitive at all, §5.3); and any speed claim.

### Upstream chain

vLLM has no single file to mirror here, because this is the shape its *whole
runtime chain* already uses instead of compile-time pinning, and which we are
adopting one layer up:

- **flashinfer** — per-arch tactic registry plus runtime selection,
  `flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h:187-220` (the 32 SM12
  tactics already mirrored in our `nvfp4_tactic_ids.h`).
- **cuBLASLt / CUTLASS** — `cublasLtMatmulAlgoGetHeuristic` resolves the kernel
  *per call* from device properties, never from `__CUDA_ARCH__`.
- **vLLM itself** — `vllm/platforms/cuda.py::CudaPlatform.get_device_capability`
  (`cuda.py:154-166`) gates which kernel family is reachable at all; ported in
  `src/vllm/platforms/cuda.cpp:64-71`.
- **MLX** (the first non-CUDA provider) — `mlx/backend/metal/eval.cpp:32-48`,
  `mlx/array.h:417-422`, `mlx/allocator.h:12-29`, pinned at `v0.29.3`.

### Our baseline

`src/vt/ops.cpp` held `std::array<std::array<void*, kNumDeviceTypes>, OpId::kCount>`
— **one** `void*` per (OpId, DeviceType) — and `RegisterOp` assigned into it with
no check and no warning (§4.2). The contract was stated outright at
`include/vt/ops.h:47-55`: "Backends register **one** kernel per (OpId,
DeviceType)". The right shape already existed one layer down and inside
`vt::cuda`: `src/vt/cuda/cuda_arch_tactics.{h,cu}` (§4.3(a)), whose CUDA
couplings are exactly three and all shallow. `DeviceResourceOps`
(`backend.h:114-138`) proves a C-ABI provider table is already accepted in-tree
(§4.3(c)); `dropin-kernel-abi.md` covers the ARGUMENT half for raw-C launchers
and cannot express an object-model library (§4.3(d)).

### Port map

| File | Action |
|---|---|
| `include/vt/op_provider.h` | NEW — `OpProvider`, `ProviderCaps`, the registry/lookup/stats API, and the declarations of `RegisterOp`/`GetOp`/`OpRegistered` moved here |
| `src/vt/op_provider.cpp` | NEW — registry, deterministic selection, `GetOpFallback`, caps publication, stats, `VT_OP_PROVIDER_STATS` / `VT_OP_PROVIDER_DISABLE`. Structure ported from `cuda_arch_tactics.cu:64-170` |
| `src/vt/ops.cpp` | Table + three symbols DELETED (they moved); every op wrapper untouched |
| `include/vt/ops.h` | Includes `vt/op_provider.h`; the three declarations replaced by a pointer to it |
| `src/vt/metal/metal_msl.h`, `metal_ops.mm` | NEW `vt_matmul` MSL kernel + `kMatmul`/`kMatmulBT` registration (the native default provider) |
| `src/vt/metal/metal_mlx_provider.mm` | NEW — the MLX provider, `VLLM_CPP_MLX` |
| `CMakeLists.txt` | `src/vt/op_provider.cpp`; the additive `VLLM_CPP_MLX` / `MLX_ROOT` block inside the existing Metal block |
| `include/vllm/model_executor/models/dense_attn_block.h` | `!is_cuda()` -> `is_cpu()` (§3.3 item 2) |

### Tests to port

vLLM has no analogue to port: it has no C++ op table, and its equivalent
selection lives in Python dispatch plus vendor heuristics. The tests are
therefore newly authored against the *properties* the upstream mechanisms
guarantee, and are named for them:

- `tests/vt/test_op_provider.cpp` — deterministic selection under REVERSED
  registration order; equal-priority name tie-break; duplicate-name rejection;
  capability-predicate skip; re-resolution after `SetDeviceProviderCaps`;
  decline-and-fall-back down a three-deep stack; the `declines` counter;
  per-call `selections`; the runtime disable lever; behaviour preservation of
  plain `RegisterOp`; unrealized (op, device) still probing false and throwing.
- `tests/vt/test_metal_backend.cpp` — the same properties on a real accelerator,
  plus MLX-vs-MSL-vs-CPU NMSE per op at real shapes, plus an end-to-end DECLINE.
- `tests/vt/test_fused_chain_additivity.cpp` — the drifted count guard repaired
  to the full 9-recipe catalog (§9 item 4).

### Gates

1. Clean `-Werror`, **0 warnings**, on AppleClang (CLT-only, Metal ON and
   Metal+MLX ON), GCC/Linux CPU, and nvcc 13.0 `sm_121a` on dgx.
2. Metal, Vulkan and CPU backend unit suites green; the seam's own tests green,
   including decline-and-fallback and deterministic selection.
3. **Correctness bar NMSE <= 5e-4** for the GEMM arms, against our CPU backend
   as the oracle, at real projection widths. Bit-exactness across providers is
   **not** promised and must not be claimed unmeasured.
4. Every arm must PROVE which provider executed (`last_selected` *and*
   `declines`), because both providers compute the same function.
5. dgx regressions, each STANDALONE, ALL UNCHANGED: 27B 235/235 · 35B 315/315 ·
   Qwen3-Coder 6/6 · Qwen3-dense 16/16 · OPT 6/6 · DeepSeek-V2 8/8. `ops.cpp` is
   the hottest shared file in the tree, so this is the real risk and is proven,
   not assumed.
6. Both record checkers.

### Dependencies

Depends on nothing. **Depended on by** work rows `W0b-3` (the `QuantTypeTraits`
split, §3.4, which keys on the same predicate), `M5` (satisfied here), the
Vulkan coopmat tactic (fan-out spike V3), and any future CUDA vendor-library
provider — whose argument half is `dropin-kernel-abi.md`. Optional build
dependency for the MLX provider only: `libmlx.dylib` + `mlx.metallib`
(~19 MB + ~105 MB), default OFF, precedent `VLLM_CPP_TRITON`.

### Work breakdown

See §6.2 for the full ranked plan and §10 for exactly which of its rows landed.
The rows owned here are `W0b-2` (LANDED), the `dense_attn_block.h` half of
`W0b-1` (LANDED), the native MSL GEMM (LANDED, unplanned but required), and `M5`
(LANDED, gated). `W0b-1` items 1/3/4, `W0b-3`, `M2r`, `M3a`, `M3b`, `M3c` and
`M4` are unchanged and still owed.

### Risks/decisions

| Risk | Disposition |
|---|---|
| **Hot-path regression in `ops.cpp`** — `GetOp` is called by every op in the tree, and `OpRegistered` per step by the fused-recipe ladder | Selection is RESOLVED ONCE and cached in a relaxed atomic; negative resolution is memoized so a missing op stays O(1); the disable-list lookup short-circuits lock-free. Proven by the unchanged dgx regression set, not by argument |
| **A provider silently declining into the slow path** (fan-out spike Risk 4) | `declines` is a first-class counter and every accelerator arm asserts it is zero. A green numeric test alone would NOT have caught this |
| **Ties broken by registration order** would reintroduce the very nondeterminism being fixed | Ties break on `name` by `strcmp`; duplicate names are rejected so the order stays total |
| **MLX becoming a hard dependency** (`discipline.md` prefers header-only) | Build-gated, default OFF, native MSL is the default and ships; `VT_OP_PROVIDER_DISABLE=mlx` switches it off in an existing binary |
| **Claiming bit-exactness across providers** | Refused. MLX-vs-MSL measured 0 on the tested shapes and is recorded explicitly as an observation, not a promise; the gate is NMSE |
| **Publishing a contended MLX timing** | Refused. The M4 could not be quieted; no MLX-vs-MSL speed number is published, and the commands the user must run first are in `docs/BENCHMARKS.md` |

---

## 12. `M3a` — OPT-125m on Apple GPU (LANDED 2026-07-22)

**The first model to run on a non-CUDA backend in this tree.** Row:
`BACKEND-METAL-MLX` (stays `ACTIVE` — correctness is met, speed is a separate
and unmet bar). Owner: `CLAIM-BACKEND-METAL-M3A-1`.

### 12.1 What was implemented

Five MSL kernels, which is exactly what §3.2 predicted MINUS `kMatmulBT` (that
landed early with the provider seam, because a seam selecting between providers
needed a second provider to select). Each transcribes the PER-ELEMENT MATH of our
own CPU reference — itself the vLLM-parity golden — and takes only its dispatch
shape from llama.cpp's Metal backend:

| Op | Ported from | Dispatch shape | Bar met |
|---|---|---|---|
| `kEmbedding` | `cpu_ops.cpp:531-543` | one thread per output element | **bit-exact** |
| `kQkvSplit` | `cpu_ops.cpp:1529-1543` | one thread per merged element | **bit-exact** |
| `kReshapeAndCache` | `cpu_cache.cpp:33-72` | one thread per (token, page element); RAW element copy, not via LoadF32/StoreF32, because upstream is a `memcpy` | **bit-exact**, incl. the `slot < 0` padded-token skip |
| `kPagedAttention` | `cpu_paged_attn.cpp:51-131` | one THREADGROUP per (query token, q-head), request resolved by an in-kernel scan of `query_start_loc` | **NMSE 4.99e-13** (bar 5e-4) |
| `kGreedyArgmax` | `cpu_sample.cpp:40-57` | one threadgroup per logits row | **bit-exact** |

**The one algorithmic deviation, stated plainly.** The CPU `PagedAttentionKernel`
is a three-pass softmax that MATERIALIZES the whole score row. On a GPU that row
is unbounded, so the Metal kernel is the algebraically identical ONLINE (flash)
form: keys in chunks of 256 with a running (max, denominator) and a rescaled
accumulator. Same f32 accumulation, same single rounding on store, DIFFERENT
reduction order — so the bar is NMSE and **no bit-exactness is claimed for it**.
The measured 4.99e-13 is eleven orders inside the bar, but that is an
observation, not a promise.

**Why four of the five ARE bit-exact and that is not a stronger claim than it
looks:** they perform no floating-point reduction, so a GPU implementation has no
reordering freedom at all. `kGreedyArgmax` does reduce, but over a max with an
explicit lowest-index tie-break, which is associative and commutative on the
(value, index) pair — genuinely order-independent, which is why the tree
reduction reproduces `torch.argmax`'s first-maximum rule exactly. The test pins
this with a deliberate two-position tie and an all-equal row.

### 12.2 Seam fixes — the study predicted 4, reality was 3 real + 1 NEW

| # | Study's claim | What was actually found |
|---|---|---|
| 1 | `model_loader.cpp` hardcodes `GetBackend(kCUDA)` | **CONFIRMED and fixed.** This was THE single line keeping every non-NVIDIA accelerator on the CPU reference regardless of how complete it was. Now asks `CurrentPlatform()` |
| 2 | `dense_attn_block.h` host-pointer bug | already fixed 2026-07-22 (§9 item 5) |
| 3 | `runner.cpp:30` unguarded `vt/cuda/` include is a blocker | **REFUTED BY MEASUREMENT.** The header is declaration-only and both COMPILES and LINKS in a Metal-only macOS build; the call sites are `is_cuda()`-guarded. The study was right that the include is unconditional and wrong that it blocks anything. No change was needed and none was made |
| 4 | Metal `get_attn_backend_priority()` returns `{}` | **CONFIRMED and fixed** — now `{"FLASH_ATTN"}`, with `FlashAttentionBackend` self-registering for `kMETAL` (a NAME registration only: its host metadata is device-agnostic and the Metal kernels read the same NHD layout). MLA stays unoffered so a `use_mla` request keeps failing loudly |
| **NEW** | *(not predicted)* | **`runner.cpp:516` gated KV-cache DEVICE RESIDENCY on `is_cuda()`.** On Metal the cache fell into a host `std::vector` and `vt::ReshapeAndCache` was handed a HOST pointer. This is the SAME defect class as item 2 and was invisible to inspection for the same reason: no test can see it until a model runs on a non-NVIDIA device. Fixed to `!is_cpu()` — device residency is a property of HAVING a device, not of the vendor |

**The study's tree-friction counts, re-judged.** The 54 raw `kCUDA` comparisons
and 13 `is_cuda()` sites are accurate as counts, but they are NOT uniform in
severity, and OPT-on-Metal needed only **2 of the 13** `is_cuda()` sites changed.
The rest are legitimate: registrar/priority keys, and CUDA-only fast paths that
correctly no-op elsewhere. The 5 CUDA includes needed **zero** changes. So the
honest headline is not "54+13 sites of coupling to unpick" but **"2 of 13
`is_cuda()` sites encoded 'not NVIDIA means no device memory', and both were real
bugs"**. No tree-wide unpicking campaign was started, and none is warranted on
this evidence.

### 12.3 A NEW seam the study did not anticipate: `Platform::supports_model_architecture`

Making `SelectQueue` platform-aware exposed a question that could not arise while
only CPU and CUDA existed: **"which device is this process running on" is not the
same question as "which device can run THIS model".** CUDA is 74/75 ops, so the
two questions had the same answer for its whole life. Metal is 15/75, and
pointing a Qwen3-dense engine at it produced a late, obscure failure inside a
kernel bind rather than a clean fall-back.

The fix is a `Platform` virtual defaulting to `true` — so CUDA and CPU are
byte-unchanged — which a PARTIAL backend overrides with the architectures whose
ops it has actually registered. Metal's list is exactly `{"OPTForCausalLM"}` and
shrinks to nothing as kernels land. This is the "a partial backend is a
supported, tested state" contract (`ops.cpp:104-111`) lifted one layer up, and it
is keyed on the architecture string rather than an OpId manifest deliberately: a
manifest would have to track every forward and would fail in exactly the silent
way this prevents.

**This was caught by the macOS regression suite, not by design** — three
previously-green tests went red the moment `SelectQueue` started selecting Metal.
Recorded as such rather than presented as foresight.

### 12.4 Evidence

* **Correctness:** `test_opt_paged_engine` on the M4 — **6/6 prompts token-exact,
  96/96 tokens**, against the committed dgx-captured vLLM 0.25.0 goldens
  (md5-verified identical before and after the run). The gate-selection evidence
  (vLLM self-determinism, 0 multi-valued cells over K=5) is re-asserted in the
  same test, so the STRICT bar remains the derived bar and not an assumption.
* **The Metal path PROVABLY executed.** A green token comparison does not say
  which device ran — every backend computes the same function. The test asserts
  `runner().device().type == kMETAL` and, for all NINE ops OPT dispatches,
  `selections > 0` AND `declines == 0` (`kPagedAttention` selections = **1152**).
  `last_selected` alone would NOT be sufficient: a provider can be selected and
  then decline INSIDE its kernel and forward down (fan-out spike Risk 4). The
  per-op unit tests additionally NaN-POISON every output buffer, so a kernel that
  never ran cannot pass a numeric check by accident.
* **Per-op vs the CPU oracle on the same M4:** the four layout/selection ops
  bit-exact; `kPagedAttention` NMSE 4.99e-13.
* **Suites:** macOS `-Werror` 0 warnings on a CLEAN full rebuild (AppleClang 21,
  CLT-only, no offline `metal`); `test_metal_backend` **12 cases / 18,535
  assertions**; full macOS ctest **154/156**, the two misses being the documented
  pre-existing platform gaps (`test_serve_low_tools`, `test_safetensors` —
  Linux-only `sched_getaffinity` and `/proc/self/smaps`). `test_capi` was
  separately observed failing STANDALONE on macOS and was proven PRE-EXISTING by
  building unmodified `origin/main` on the same box and reproducing the identical
  `CHECK(1 == 2)`; it passes under ctest.

### 12.5 What is NOT claimed

**No Metal speed number, and none is owed.** The M4 could not be quieted (the
root `com.localai.worker` LaunchDaemon needs interactive sudo; the desktop aerial
wallpaper is the larger contender at 8.2% CPU), so any timing would be void under
the standing contended-run rule. Correctness and speed are separate bars and only
correctness is met. `BACKEND-GATE-METAL-MLXLM` therefore stays `INVENTORIED`: it
binds on Qwen3-dense (`M3b`), the model MLX also runs, because the arms must be
comparable.

Dispatch is still one command buffer per op with commit+wait (`M3c` is the
batched-encoder work), the GEMM is a plain threadgroup-tiled loop with no
simdgroup matrix use, and nothing here should be read as a performance result.

---

## 13. `M3b` — Qwen3-dense on Apple GPU (forward oracle-confirmed correct, near-tie-robust) + the first ours-vs-MLX benchmark (LANDED 2026-07-23)

**The SECOND model on a non-CUDA backend, and the model MLX also runs — so the
native-competitor arms are finally comparable.** Row: `BACKEND-METAL-MLX` (stays
`ACTIVE`), `BACKEND-GATE-METAL-MLXLM` (`INVENTORIED` → `ACTIVE`). Owner:
`CLAIM-BACKEND-METAL-M3B-1`, base `origin/main` `4884d03`.

### 13.1 The ops — the study's count corrected against the CURRENT Metal op set

§3.1 predicted "10 ops, 7 new" and named `kRopeFromCache`. Measured against the
op set M3a actually left (15 of 75, not the study's baseline), Qwen3-dense needed
**3 new MSL kernels**, because the DEFAULT dense attention path is
`VT_QWEN3_ROPE_CACHE`-**ON** (`dense_attn_block.h`), which dispatches:

| Op | Role on the default path | Bar met |
|---|---|---|
| `kRopeCosSinCache` | build the per-step cos\|sin cache once per step (`RopeCosSinCache`, line ~275, called whenever `rot>0`) | NMSE 4.07e-15 (f32 `pow`/`precise::cos/sin`; Metal has no double) |
| `kRopeFromCache` | apply the bf16 cache to q/k (`RopeFromCache`, line ~411) — the op the gate actually exercises | **BIT-EXACT** (reads the shared c/s, only rotates; no reduction/transcendental) |
| `kRopeNeox` | the `VT_QWEN3_ROPE_CACHE=0` opt-out (in-place fp-transcendental rotation) | NMSE 0 (measured) |

So the study was right that Qwen3 needs `kRopeFromCache`, but omitted that the
default path ALSO builds the cache (`kRopeCosSinCache`) and that `kRopeNeox`
covers the opt-out. Metal is now **18 of 75**. All three transcribe our own CPU
reference math; `kRopeFromCache` moves the transcendental OUT of the kernel (it is
in the cache) so it stays bit-exact, which is why it is the one the gate leans on.

### 13.2 Correctness — forward ORACLE-CONFIRMED CORRECT (near-tie-robust), NOT strict-token-exact

**HONEST STATE (2026-07-23, supersedes an earlier "16/16 strict / 4 near-ties at
gap-0" claim that was UNSUBSTANTIATED — the branch test had no teeth; see the RCA +
oracle ledger rows).** Qwen3-0.6B is a NEAR-TIE MODEL: at ~4 first-divergence
positions its top-2 tokens are separated by thousandths of a nat, so a correct
backend can legitimately pick either. The Metal forward diverges from the CUDA
golden at p0 tok5 (15344 " Italy" vs 9625 " France"), p5 tok10, p10 tok10, p11 tok1.

**THE DECISIVE MEASUREMENT (the vLLM oracle teacher-forced on the METAL prefix).**
Captured the Metal token sequence on the M4, transferred to dgx, and teacher-forced
vLLM 0.25.0 (`scripts/qwen3-neartie-gap.py`, batch=1, `gpu_memory_utilization=0.40`,
`enforce_eager`) on the Metal prefix. RESULT: **every one of the 60 Metal-vs-CUDA
divergent positions is within 0.5 nats of vLLM's OWN argmax given the Metal prefix —
max gap 0.125 nats, none outside vLLM's top-20.** Per-position nats at the 4
first-divergences: **p0 tok5 gap 0.0000** (vLLM's teacher-forced argmax on the
identical prefix IS 15344 " Italy" — it CONTRADICTS its own CUDA-capture " France"
pick; the France/Italy top-2 margin is 0.003–0.007 nats), **p5 tok10 0.1250**,
**p10 tok10 0.0000**, **p11 tok1 metal=11 0.1250** (Metal here matches vLLM's greedy
where CUDA didn't). ⇒ the Metal forward is CORRECT by the ratified near-tie-robust
bar, oracle-backed.

**THE GATE (landed, honest).** Device-appropriate ORACLE goldens:
`goldens/qwen3_greedy_0_6b/our_ids_metal.npy` (the Metal forward's deterministic
greedy) + `neartie_gap_mnats_metal.npy` (vLLM teacher-forced on the Metal prefix).
The gate is IDENTICAL logic on every device — hard anchor `REQUIRE` + ≤0.5-nat
near-tie band — differing ONLY in which device's oracle golden it loads; NO
cross-device latitude (Metal gated against Metal's OWN golden). Metal PASSES
**16/16** (10 strict token-exact vs vLLM greedy + 6 near-tie-band, max gap 0.125, 0
forward-divergent). PROVEN to have TEETH: perturbing the committed Metal anchor →
hard anchor-drift FAILURE; perturbing a gap to 0.6 nats (>0.5 band) →
FORWARD-DIVERGENCE band FAILURE; both restore to PASS. Metal execution PROVEN:
`device == kMETAL` + `selections > 0 ∧ declines == 0` for all 9 ops
(`kRopeFromCache`/`kPagedAttention` 7168 each); per-op tests NaN-poison outputs.
Per-op correctness stays the NMSE ≤5e-4-vs-CPU proof (RCA, all 28 layers).

**STRICT token-exactness on Qwen3-0.6B is ILL-POSED (near-tie model); a strict Metal
gate needs a bigger DETERMINISTIC dense model (Qwen3-4B) — not present on the M4 —
DEFERRED.**

### 13.3 The France/Italy tie is BUILD-SENSITIVE on CUDA too (the "stale golden" claim DISPROVEN)

An earlier note claimed the dgx CUDA gate was RED with a stale golden. **DISPROVEN.**
The France/Italy tie is BUILD-SENSITIVE: the PRODUCTION dgx build (FA2+Marlin+Triton
+CUTLASS, `-DCMAKE_CUDA_ARCHITECTURES=121a`) resolves p0 tok5 → **France 9625** and
the CUDA gate passes **16/16** (Qwen3-0.6B: 10 strict + 6 near-tie, max gap 0;
Qwen3-4B: 16/16, 11 strict + 5 near-tie, max 0.25 nats), `-Werror` 0 warn. Only a
PORTABLE-KERNEL-ONLY CUDA build (FA2/Marlin/Triton OFF) resolves it → **Italy 15344**
(the same as Metal). So the committed golden is the production build's resolution of
a genuine numerical near-tie, NOT stale — and Metal picking Italy is the correct
near-tie resolution its own numerical path produces, oracle-confirmed in §13.2.
Regressions GREEN (OPT 6/6, Qwen3-Coder 6/6; 27B/35B/DeepSeek binaries
byte-identical — only the one parity test .cpp compiles differently on dgx-CUDA).
Goldens md5 `2965ef5772b556d3f3f86fedf4221b2f` UNCHANGED (the 2 Metal files are
additive).

### 13.4 The benchmark — INDICATIVE / BLOCKED-ON-SUDO, and it is a FLOOR

Both arms Qwen3-1.7B bf16, p=512 g=128, same box/session. Ours (Metal
`vllm-bench`, device=2 confirmed) decode **4.29 → 2.14 tok/s/stream**, TTFT
**4.9 → 47.2 s**, peak **7.36 → 8.78 GB** (b=1→16); MLX (`mlx_lm.benchmark`)
**27.77 → 211.55 tok/s**, TTFT **0.47 → 7.16 s**, peak **3.78 → 5.28 GB**. Ours
loses every axis — **~6–11× slower decode, ~7–10× slower TTFT, ~2× memory** — and
that is a knowingly-unoptimised FLOOR: one command buffer per op, a naive
threadgroup-tiled GEMM, no batched encoders (`M3c`), vs MLX's `steel` kernels. The
Mac could not be quieted (no passwordless sudo; root `com.localai.worker` + aerial
wallpaper up), so it is INDICATIVE, not binding. `BACKEND-GATE-METAL-MLXLM` is
`ACTIVE` (not `DONE`): the named levers to reach parity are `M3c` (batched
encoders) and a simdgroup-matrix GEMM, plus a quiet-box re-run.
