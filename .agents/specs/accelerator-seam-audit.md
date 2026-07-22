# Accelerator seam audit — do MLX/Vulkan ride vLLM's CUDA-path logic? (`BACKEND-SEAM-AUDIT`)

Status: **AUDIT + PLAN (2026-07-22) with work row `S1` LANDED (2026-07-22).**
`S1` — the DSR ratchet — is implemented and CI-gated; see **§8** for the as-built
metric, the re-derived baseline (**86**, not 94) and the re-verified class
composition. `S2`–`S8` remain plan only; no source file under `src/`/`include/`
has been changed by this row and no build or GPU run is owed.
Owners: `CLAIM-BACKEND-SEAM-AUDIT-1` (audit), `CLAIM-BACKEND-SEAM-S1-1` (`S1`).
Row: **`BACKEND-SEAM-AUDIT`** (`ACTIVE` — `S1` landed, `S2`–`S8` owed). No other
row moves.

Audit base: `72f5db2`; `S1` base: `18094ee`. Pins: vLLM
`/home/mudler/_git/vllm` @ `e24d1b24`; llama.cpp `237ad9b96`; MLX `v0.29.3`.
Every count in §§0–7 was measured at `72f5db2`; **§8 supersedes the numeric
claims** where the two differ.

**The user's question, verbatim:** *"In the architecture for MLX and Vulkan, are
we 'porting' the same strategy of the CUDA path of vLLM? Does it map correctly?"*
and *"I would like us to ride the same logical implementations so we do not have
to re-implement all, but maybe just the equivalent ops per accelerator."*

---

## 0. The straight verdict

> **PARTIALLY — and the part that fails is the part that matters most for the
> stated goal.**
>
> **YES** at the two seams we deliberately ported: `Platform`
> (`platforms/interface.py:134-229` → `include/vllm/platforms/interface.h:109`)
> and the attention-backend registry
> (`v1/attention/backends/registry.py` → `include/vllm/v1/attention/registry.h`).
> Both are faithful, and our attention registry is **structurally better than
> upstream's** (open self-registration vs a closed hard-coded enum). Vulkan and
> Metal genuinely plug into these with zero engine edits — that is measured, not
> claimed: Vulkan V1 added 11 files and edited **zero** pre-existing `src/` or
> `include/` source file.
>
> **NO** at the seam that carries the actual model math. vLLM's device branching
> lives in a **shared layer library** (`model_executor/layers/`: Linear +
> QuantMethod + CustomOp + FusedMoE), which every model and every platform reuse.
> **We did not port that library. We inlined it into the model files.** So our
> "shared logical layer" is thinner than vLLM's by exactly one library, and the
> device branching it was supposed to absorb landed in the models instead.
>
> **The single decisive number.** Upstream `qwen3_next.py` is **802 lines with
> ONE device predicate** (`:321`). Our counterpart `qwen3_5.cpp` is **6,389 lines
> with 67** (38 `kCUDA` + 3 `is_cuda()` + 26 `#ifdef VT_*`). Across the whole
> upstream tree, **9 of 287 model files** touch a device predicate at all
> (14 sites); across ours, one model file holds **71% of all device-specific
> references in the shared layer**.
>
> So: we ride vLLM's *seams*, but not vLLM's *shared implementations*. A new
> accelerator today rides the engine, the scheduler, the KV manager, the sampler,
> the validation tier and the fusion catalog for free — and then meets a model
> file that was written against CUDA. That is the distance to close.

**Second verdict, on the sub-question "does the ops-only story hold?":** yes, and
it is the healthy part. OPT needs **9 ops (6 new)**, Qwen3-dense **10 (7 new)**.
That is the right *shape*. The defect is not the op count — it is that those ops
are a **correctness** gate rather than a **performance** budget (§4.3), because
we have no portable fallback tier where vLLM has all of torch.

---

### Scope

**In.** A structural audit of our accelerator seam against vLLM's, at
`file:line` on both sides; a re-measured inventory of every device-specific
reference in our shared layer with a per-site classification; a proposed
CI-guarded metric; and a ranked work plan to reach "shared logic, swapped ops".

**Out.** Any code change (a concurrent claim owns `src/vllm/`); any build,
benchmark or GPU run; any per-backend port map (owned by
[backend-fanout-metal-vulkan-xpu.md](backend-fanout-metal-vulkan-xpu.md)); any
change to the `vt::OpProvider` mechanism (owned by
[metal-mlx-reuse-study.md](metal-mlx-reuse-study.md)); ROCm, ANE, TPU
implementation. This document does not move `BACKEND-METAL-MLX`,
`BACKEND-VULKAN`, `BACKEND-XPU`, `BACKEND-PLATFORM`, `BACKEND-ATTN-REGISTRY`,
`BACKEND-ACCEL-PROVIDER` or any gate row.

---

### Upstream chain

The audit's comparators, all at pin `e24d1b24`:

| Upstream surface | Why it is the honest comparator |
|---|---|
| `vllm/platforms/interface.py:134-229` (`class Platform`), `:363-372` (`get_attn_backend_cls`), `:409-479` (capability queries), `:926-984` (fp8/mx/allreduce capability), `:1051-1072` (hybrid-KV / static-graph / DeepGEMM support) | the platform seam we claim to mirror. **~85 methods**; ours has 8 |
| `vllm/platforms/rocm.py` (1,052 lines), `:531-600` `get_attn_backend_cls` | **the closest analogue to our situation**: same algorithms, different kernels, one platform file |
| `vllm/platforms/xpu.py` (487 lines), `:121-167` selector policy | a platform whose kernels are OUT OF TREE — the shape a from-scratch backend takes |
| `vllm/platforms/cpu.py:42-125` | the minimal platform |
| `vllm/v1/attention/backends/registry.py:34-120` (`AttentionBackendEnum`) | the attention seam; a **closed enum**, 20+ entries incl. 5 ROCm and 1 XPU |
| `vllm/v1/attention/backends/{rocm_attn,cpu_attn,triton_attn,flashinfer,flex_attention}.py` | per-platform backends behind ONE `AttentionBackend/Impl/MetadataBuilder` contract |
| `vllm/model_executor/custom_op.py:103-200` (`CustomOp`, `forward_native/cuda/hip/xpu/cpu/tpu/oot`), **42 `@CustomOp.register` sites** | **the op-granularity comparator.** Every one has a `forward_native` that runs anywhere torch runs |
| `vllm/model_executor/layers/quantization/base_config.py:20-229` (`QuantizeMethodBase`, `QuantizationConfig.get_quant_method`), `vllm/model_executor/layers/linear.py:141-230` (`LinearMethodBase`, `UnquantizedLinearMethod`) | **the coarse seam we lack**; 18 quant config modules |
| `vllm/model_executor/layers/` as a whole | where **199 of vLLM's 544** `current_platform.is_*` sites live |

**Where upstream has NO answer, recorded honestly.** vLLM has **no Vulkan and no
Metal platform** — `ls vllm/platforms/` is `{cpu, cuda, rocm, tpu, xpu, zen_cpu}`.
Those two are vllm.cpp extensions already recorded in
[porting-inventory.md](../porting-inventory.md) §9 and
[backends.md](../backends.md), with llama.cpp `ggml/src/ggml-vulkan/` and
`ggml/src/ggml-metal/` as the port-FROM sources. **This audit confirms that
disposition is correct and correctly recorded**, and adds one binding rule:
where upstream HAS an answer (Platform methods, attention contract, quant method
dispatch, CustomOp fallback), a new backend mirrors upstream even though the
backend itself is an extension. Metal/Vulkan are permitted to be new *platforms*;
they are not permitted to be a new *architecture*.

---

### Our baseline

#### 1. The side-by-side mapping

| # | vLLM construct (`file:line`) | Our construct (`file:line`) | Mirror / diverge / absent | Justified? |
|---|---|---|---|---|
| 1 | `platforms/interface.py:134` `class Platform` | `include/vllm/platforms/interface.h:109` `class Platform` | **MIRROR** (shape), **DIVERGE** (surface: ~85 methods → 8) | **Partly.** ~30 are torch/Python-only (`get_pass_manager_cls:240`, `get_compile_backend:248`, `inference_mode:509`, `get_punica_wrapper:879`, `stateless_init_device_torch_dist_pg:1030`) — correctly absent. But `supports_fp8:933`, `is_fp8_fnuz:940`, `fp8_dtype:954`, `supports_mx:926`, `opaque_attention_op:977`, `is_integrated_gpu:914`, `support_hybrid_kv_cache:1051`, `support_static_graph_mode:1058`, `num_compute_units:1133`, `is_arch_support_pdl:1153`, `get_device_total_memory:491` are portable, are things our code branches on TODAY via `is_cuda()`, and are **ACCIDENTALLY absent**. → work row `S3` |
| 2 | `interface.py:189-215` `is_cuda/is_rocm/is_xpu/is_cpu/is_tpu` | `interface.h:116-117` `is_cuda/is_cpu` only | **MIRROR** | Yes — we have two platforms in production. But `is_cuda()` is being used as a proxy for capabilities it does not mean (§3.2 class D), which is the same misuse upstream avoids by having `is_cuda_alike:220` and the capability predicates |
| 3 | `interface.py:410-479` `get/has/is_device_capability(_family)` | `interface.h:130-134` `get_device_capability` / `has_device_capability` | **MIRROR**, `is_device_capability_family` **ABSENT** | Accidental-minor. Upstream uses it **53×**; it is how ROCm gfx-family and CUDA sm-family gating is expressed. We will need it for the ROCm port |
| 4 | `interface.py:181-187` `supported_dtypes` | `interface.h:138` `supported_dtypes()` | **MIRROR** | Yes |
| 5 | `platforms/__init__.py` `current_platform` resolution | `src/vllm/platforms/platform.cpp:38-40` priority `{kCUDA,kXPU,kVULKAN,kMETAL,kCPU}` + `CurrentPlatform()` | **MIRROR** | Yes. Ours is a static priority walk vs upstream's import-probe — a C++ necessity, and it already lists all five devices |
| 6 | `platforms/cuda.py::CudaPlatform` (1,017 lines) | `src/vllm/platforms/cuda.cpp` (113 lines) | **MIRROR** in kind | Yes for scope; the 9× size gap is item 1's missing methods plus Python config plumbing we do not have |
| 7 | `platforms/cuda.py::_get_backend_priorities:84-176` | `include/vllm/platforms/cuda_attn_priority.h` (134 lines) + `cuda.cpp:69-77` | **MIRROR**, both branches | Yes — a faithful data-table port |
| 8 | `v1/attention/backends/registry.py:34` `AttentionBackendEnum` — a **CLOSED enum**, adding a backend edits a central file | `include/vllm/v1/attention/registry.h:44` `RegisterAttentionBackend(DeviceType, name, factory)` — **OPEN self-registration**, additive | **DIVERGE — in our favour** | Yes, and record it as a deliberate improvement. Upstream's enum is not additive; ours is. A future upstream PR adding an enum entry ports to one `AttentionBackendRegistrar` line — still mechanical |
| 9 | `platforms/*.py::get_attn_backend_cls` + `backend.py:307-360 validate_configuration` (returns *invalid reasons*) | `src/vllm/v1/attention/registry.cpp:71` `SelectAttentionBackendName` + `is_mla()`/`is_sparse()` filter | **MIRROR** (structure), **DIVERGE** (upstream returns a diagnosable reason list per candidate; we return first-registered) | Minor accidental. Upstream's `invalid_reasons` string is a real debuggability feature (`rocm.py:569-586`); worth porting when a platform first has >1 registered backend |
| 10 | `v1/attention/backends/{rocm_attn,cpu_attn,triton_attn,flashinfer,flex_attention,mla/*}.py` — per-platform Impl classes behind one contract | `src/vllm/v1/attention/backend.cpp` + `backends/gdn_attn.cpp`; concrete kernel via `vt::PagedAttention` → `GetOp` | **MIRROR** | Yes. Note our kernel lands one layer LOWER (op table) than upstream's (Impl class). That is the granularity divergence of item 14 |
| 11 | `model_executor/custom_op.py:103` `CustomOp` + 42 registrations, dispatching `forward_cuda/hip/xpu/cpu/tpu/oot` with **`forward_native` as universal fallback** | `include/vt/ops.h:88` `enum OpId` (**75 ops**) + `RegisterOp(OpId, DeviceType, fn)`; `GetOp` **throws** when absent | **MIRROR in granularity, DIVERGE in fallback** | **Granularity: yes, justified** (42 CustomOps ≈ 75 vt ops — the same class of thing, §4.1). **Fallback: NO, accidental and the most consequential divergence in this table.** Upstream's `forward_native` means a platform with zero kernels is CORRECT; our `GetOp` throw means a platform with zero kernels does not run. → work row `S5` |
| 12 | `quantization/base_config.py:87` `QuantizationConfig` + `:180` `get_quant_method(layer, prefix)`; `linear.py:141` `LinearMethodBase` (`create_weights`/`apply`/`process_weights_after_loading`); 18 config modules | **NOTHING.** `src/vllm/model_executor/layers/quantization/` contains only `compressed_tensors/`. Scheme selection is a per-model tensor-name probe: `include/vllm/model_executor/models/qwen3.h:60` `IsNvfp4() { return !qkv_proj_fp4.Empty(); }` | **ABSENT** | **NO — accidental, and this is the missing coarse seam** (§5). It is where upstream puts per-accelerator kernel choice, and quant kernel choice is per-accelerator by nature |
| 13 | `model_executor/layers/` as a shared library — Linear, layernorm, rotary, activation, fused_moe, mamba/gdn — **199 of 544** device predicates | `src/vllm/model_executor/layers/` = `{attention, quantization, rotary_embedding}` only; Linear/MoE/GDN/norm logic is **inlined into the model TUs** | **ABSENT** | **NO — accidental, and it is the root cause of the leakage distribution.** → work row `S7` |
| 14 | `vllm/platforms/rocm.py` — one file adds a platform whose kernels differ but whose layers/models are byte-identical | `src/vllm/platforms/{metal,vulkan}.cpp` (84/96 lines) + `src/vt/{metal,vulkan}/*` | **MIRROR — and it works.** Vulkan V1 edited **0** pre-existing `src/`/`include/` files | Yes. This is the audit's good news and it is measured |
| 15 | flashinfer per-arch tactic registry (`fp4_gemm_cutlass_template_sm120.h:187-220`); cuBLASLt `cublasLtMatmulAlgoGetHeuristic`; torch dispatch — **three unrelated mechanisms** | `include/vt/op_provider.h` `vt::OpProvider` — ONE mechanism, `(priority DESC, name ASC)`, capability predicate, decline-and-fallback, stats | **DIVERGE — a unification upstream does not have** | **Both an improvement and a bounded risk** (§6) |
| 16 | `platforms/interface.py:531-633,805` `check_and_update_config` / `update_block_size_for_backend` / `register_custom_kv_cache_specs` — the platform amends engine config | **ABSENT** | **ABSENT** | Justified today (no platform needs it), but it is upstream's declared seam for "this device needs a different block size / KV spec", and Metal and Vulkan will both want it. Record, do not build |
| 17 | `interface.py:824` `verify_quantization` / `:812` `verify_model_arch` — platform rejects an unsupported scheme with a message | **ABSENT** — an unsupported scheme reaches a `VT_CHECK` or a missing op | **ABSENT** | Accidental-minor; falls out of `S4` for free |

**Summary of the table.** 17 constructs: **8 mirror cleanly**, **1 diverges in
our favour** (#8), **1 diverges as a defensible unification** (#15), **3 diverge
by omission of portable surface** (#1, #3, #9), and **4 are absent where they
should not be** (#11 fallback, #12 quant method, #13 layer library, #16/#17
config/verify). The four absences are not independent: **#13 is the container,
#12 is its most important member, #11 is what makes a partial member safe.**

#### 2. Where the shared layer leaks — full inventory, re-measured at `72f5db2`

Scope of the count: `src/vllm/` + `include/vllm/` (the layer that is supposed to
be device-agnostic). `src/vt/cuda/`, `src/vt/metal/`, `src/vt/vulkan/` are
device legs by definition and are excluded.

| Bucket | Total lines | Platform-definition (legitimate, allowlisted) | **Shared-layer leakage** |
|---|---:|---:|---:|
| `DeviceType::kCUDA` textual references | 63 | 16 | **47** |
| `platform.is_cuda()` call sites (excl. the definition + 4 comment lines) | 16 | 1 (`interface.h:116`, the definition) | **11** |
| Unconditional `#include "vt/cuda/…"` / `<cuda_runtime.h>` | 6 | 1 (`platforms/cuda.cpp:6`) | **4** (+1 already `#ifdef`-guarded) |
| `#ifdef VT_*` kernel-feature gates | 32 | 0 | **32** |
| **Device-Specific References (DSR) total** | | | **94** |

> **Superseded by §8.1.** Re-derived by the as-built checker: **88 at this same
> `72f5db2`**, **86 at `18094ee`**. The `kCUDA` bucket here counts comment
> mentions (the `is_cuda()` bucket does not — an internal inconsistency), and the
> "unconditional CUDA includes" bucket is a measurement error: all 4 were already
> `#ifdef`-guarded.

Allowlist (these ARE the CUDA leg, and counting them would be wrong):
`platforms/cuda.cpp` (6), `platforms/platform.cpp:39` (the device priority
list), `platforms/vulkan.cpp:85` + `platforms/metal.cpp:74` (comments naming the
priority walk), `platforms/interface.h:116` (the `is_cuda()` definition),
`v1/attention/registry.h:43,90` (doc), `v1/attention/backend.cpp:179,196` +
`backends/gdn_attn.cpp:162,166` (the `(kCUDA, name)` registrar keys).

**Distribution — the finding.**

| File | kCUDA | is_cuda() | `#ifdef VT_*` | DSR |
|---|---:|---:|---:|---:|
| `model_executor/models/qwen3_5.cpp` (6,389 lines) | 38 | 3 | 26 | **67 (71%)** |
| `v1/worker/gpu/runner.cpp` | 0 | 3 | 0 | 3 (+1 include) |
| `entrypoints/model_loader.cpp` | 3 | 0 | 0 | 3 (+1 include) |
| `models/dense_nvfp4_gemm.h` | 2 | 0 | 3 | 5 |
| `models/{qwen3,qwen3_moe,qwen3_5_dense_weights,deepseek_v2}.cpp` | 4 | 1 | 1 | 6 |
| `models/{qwen3_5_dense,qwen3_5_moe,deepseek_v2_registry,qwen3_moe_registry}.cpp` | 0 | 4 | 0 | 4 |
| `models/qwen3_5_weights.cpp`, `models/device_pool.h` | 0 | 0 | 2 | 2 |

**The upstream comparator, and it is not flattering.** vLLM's whole tree carries
**544** `current_platform.is_*` sites — far more than our 94. But their
distribution is the opposite of ours:

| Where | vLLM | Ours |
|---|---:|---:|
| `model_executor/layers/` (the shared layer library) | **199** | — (library does not exist) |
| `model_executor/models/` | **14, in 9 of 287 files** | **72, in 8 of ~30 files** |
| `v1/worker/` | 17 | 4 |
| the single biggest model file | `deepseek_v2.py`: **5** | `qwen3_5.cpp`: **67** |
| our gate model's upstream twin | `qwen3_next.py`: **1** of 802 lines | `qwen3_5.cpp`: **67** of 6,389 |

vLLM tolerates a lot of device branching — but it puts **essentially none of it
in model definitions**, because the layer library absorbs it once for all 287
models. We put **71% of ours in one model definition**, because there is no
library to absorb it.

**The counts have already regrown, which is the argument for a metric.**
[metal-mlx-reuse-study.md](metal-mlx-reuse-study.md) §3.3 measured, days ago:
54 `kCUDA` / 13 `is_cuda()` / 5 CUDA includes / 37 in `qwen3_5.cpp`. Today:
**63 / 16 / 6 / 38**. Nothing regressed deliberately — DeepSeek-V2, Qwen3-Coder
and the registry work each added a couple of device tests in passing. **Leakage
grows silently under normal, well-executed work.** That is precisely the failure
mode a ratchet exists for.

#### 3. Per-site classification, and the fix per class

| Class | Sites | What it actually is | Fix |
|---|---:|---|---|
| **A — fast-path availability** | ~45 (34 in `qwen3_5.cpp`; `qwen3.cpp:85`, `qwen3_moe.cpp:384`, `deepseek_v2.cpp:885`, `dense_nvfp4_gemm.h:433,439`, `qwen3_5_dense.cpp:98`, `qwen3_5_moe.cpp:77`, the two `*_registry.cpp:104/105`) | Shape `env_on && device.type == kCUDA && <shape>` — i.e. "is the fused/quant fast path available here?". **Not a device question at all**; the device test is a stand-in for "is this kernel registered and does it accept this shape?" | **`OpRegistered(op, device) && provider_supports(shape)`.** The mechanism already exists — `vt::OpProvider` `supports(ProviderCaps)` + `GetOpFallback` decline, and the 3-rung fused-recipe ladder (`ops.cpp:963-995`). Work row `S6`; the quant half needs `S4` |
| **B — hard device assertions** | 5 (`qwen3_5.cpp:1144,1166,2780,3151,5298`) | `VT_CHECK(device.type == kCUDA, …)` guarding a CUDA-only kernel | Delete. `vt::GetOp` (`ops.cpp:104-111`) already throws a better message for an unregistered op. The assertion duplicates the op table and is wrong the moment a second device registers the op. `S6` |
| **C — platform-definition keys** | 16 (allowlisted above) | The `(kCUDA, …)` keys that DEFINE the CUDA leg | **None — correct as written.** Must be excluded from the metric or the metric punishes adding backends |
| **D — real platform capability, asked via the wrong predicate** | 6 (`runner.cpp:516,690,1068`; `model_loader.cpp:276`; `qwen3_5_dense_weights.cpp:115`; `deepseek_v2.cpp:311`) | Genuine device-dependent behaviour (device-side combine/scatter, graph-capture-safe paths, weight repack residency) expressed as `is_cuda()` | **New `Platform` capability fields**, mirroring upstream: `opaque_attention_op` (`interface.py:977`), `support_static_graph_mode` (`:1058`), `is_integrated_gpu` (`:914`), `supports_fp8` (`:933`). Mechanical, buys the ROCm port for free. Work row `S3` |
| **E — hard-coded device choice** | 1 (`model_loader.cpp:39` `GetBackend(kCUDA).CreateQueue()`) | A defect: the loader creates a CUDA queue regardless of platform | `CurrentPlatform().device_type()`. **Already owed** as reuse-study `W0b-1` item 1. Work row `S2` |
| **F — unconditional CUDA includes** | 4 (`model_loader.cpp:22`, `runner.cpp:30`, `qwen3_5.cpp:40,46`) | A non-CUDA build does not compile. `runner.cpp` is on **every** path | `#ifdef VLLM_CPP_CUDA` (16 such gates already exist in-tree). `S2`. Note `dense_nvfp4_gemm.h:66` is already guarded — less severe than the fan-out spike stated |
| **G — build-time kernel-feature gates** | 32 `#ifdef VT_*` | Whether a CUDA kernel family was compiled in | Mostly legitimate *today*, but they are a build-time axis where `OpProvider` is the runtime axis. Long-term they collapse into "is a provider registered". Tracked, not scheduled |

**Nothing in class A, B, E or F is genuinely device-dependent behaviour.** Of the
94 DSR, **6 (class D) are real device policy**, 32 (class G) are build config, and
**~56 are the shared layer asking a question the op/provider table already
answers.**

---

### Port map

This audit ports **no files**. The map below is the *target* structure — the
upstream file each future work row mirrors — so the plan is grounded rather than
invented. Implementation is owned by the work rows in § Work breakdown, each of
which needs its own claim.

| Upstream | Target local file | Work row | Note |
|---|---|---|---|
| `vllm/platforms/interface.py:914,926-984,1051-1072,1133,1153` | `include/vllm/platforms/interface.h` (extend), `src/vllm/platforms/{cuda,cpu,metal,vulkan}.cpp` | `S3` | additive virtuals with defaults ⇒ no existing platform edit is forced |
| `vllm/platforms/interface.py:466` `is_device_capability_family` | `include/vllm/platforms/interface.h` | `S3` | needed for ROCm gfx families and CUDA sm families |
| `vllm/model_executor/custom_op.py:138` `forward_native` | `src/vt/op_provider.cpp` + a `vt::ref` provider tier registering CPU kernels at negative priority on unified-memory devices | `S5` | no new mechanism — `OpProvider` already supports priority + `supports()` |
| `vllm/model_executor/layers/quantization/base_config.py:20-229`, `vllm/model_executor/layers/linear.py:141-230` | `include/vllm/model_executor/layers/quantization/config.h`, `.../linear_method.h`; schemes under `.../schemes/` keyed `(scheme, DeviceType)` | `S4` | replaces `qwen3.h:60` `IsNvfp4()`-style tensor-name probes |
| `vllm/model_executor/layers/{linear,layernorm,rotary_embedding,activation}.py`, `layers/fused_moe/`, `layers/mamba/gdn/` | `include/vllm/model_executor/layers/…` — extract from `qwen3_5.cpp` / `dense_attn_block.h` | `S7` | the structural end-state; strictly behaviour-preserving extraction |
| `vllm/platforms/rocm.py:531-600` `get_valid_backends` invalid-reason reporting | `src/vllm/v1/attention/registry.cpp:71` | `S6` (opportunistic) | debuggability parity |
| `vllm/platforms/xpu.py:121-167` | `src/vllm/platforms/xpu.cpp` (data only, no kernels) | `S8` | already specced as fan-out `X1`; listed here as the seam's third-platform proof |
| — (no upstream analogue; precedent `cmake/CudaArchFeaturesTest.cmake` + `scripts/check-agent-record.py`) | `scripts/check-device-leakage.py` + a CI job | `S1` | the ratchet, §"Gates" |

---

### Tests to port

Upstream-test provenance is genuine for the platform/capability half and absent
for the metric half; the inventory says which is which.

| Source | Ours | Tier | Note |
|---|---|---|---|
| `vllm/tests/test_zen_cpu_platform_detection.py:8-37` (platform detection shape) | `tests/vllm/platforms/test_platform.cpp` — add cases for every new capability field of `S3`, per registered platform | doctest | the existing CPU/CUDA cases are the pattern |
| `vllm/tests/kernels/quantization/**` (scheme-parameterized `create_weights`/`apply`) | new `tests/vllm/model_executor/layers/test_linear_method.cpp` — scheme×device selection table, incl. "scheme unsupported on this device declines cleanly" | doctest | `S4`; ported cases named after the upstream module |
| `vllm/tests/kernels/core/test_activation.py`, `test_layernorm.py` (`forward_native` vs `forward_cuda` equality) | `tests/vt/test_backend_cross_device.cpp` — **extend, do not duplicate**: assert the `vt::ref` fallback tier produces the registered kernel's answer within NMSE ≤ 5e-4 | doctest | `S5`. This IS upstream's native-vs-device test, ported |
| `vllm/tests/v1/attention/test_attention_selector.py` (invalid-reason reporting) | `tests/vllm/v1/attention/test_attn_backend_registry.cpp` — add reason-string cases | doctest | `S6` opportunistic |
| NEW — no upstream analogue (vLLM has no such ratchet) | `tests/tools/test_device_leakage_check.py` — a mutation suite for `scripts/check-device-leakage.py`: a planted `kCUDA` in a shared-layer file must FAIL; a planted one in an allowlisted platform leg must PASS; removing a site must lower the count | script test, mirroring the existing `check-agent-record.py` mutation suite | `S1`. Required — an unpoliced checker is worse than none |
| existing `tests/vt/test_op_provider.cpp`, `tests/vt/test_metal_backend.cpp`, `tests/vt/test_vulkan_backend.cpp` | **reuse unchanged** as the regression net for `S5`/`S6` | — | they already assert partial-backend behaviour as an executable fact |

---

### Gates

**This change (the audit) has exactly two gates, both non-computational:**
`scripts/check-agent-record.py` green and
`scripts/check-doc-checkpoint.py` green. No build, no test run, no GPU, no
benchmark — nothing was implemented. `docs/BENCHMARKS.md` records this as
`NOT APPLICABLE` with that reason stated.

**The gates the PLAN must meet**, per work row:

1. **The tracked metric — DSR (Device-Specific References in the shared layer).
   LANDED 2026-07-22 — see §8 for the as-built definition and the re-derived
   baseline.** `scripts/check-device-leakage.py` counts, over `src/vllm/` +
   `include/vllm/`, the four buckets of §"Our baseline" 2 minus the allowlist, and
   compares to `scripts/device-leakage-baseline.json`. **Baseline at `18094ee` =
   86** (44 + 10 + 0 + 32); the audit's `72f5db2` figure of 94 was 6 too high for
   composition reasons and 2 too high because main has since reduced real leakage
   (§8.1). CI **fails on any increase**, per bucket and in total; a decrease
   requires the baseline be lowered in the same commit (a ratchet, not a
   threshold). The allowlist is explicit, per-file, per-bucket, with an **exact
   expected count** and a stated reason, so "add a backend" never trips it and
   "branch a model on CUDA" always does.
   *Precedent:* `cmake/CudaArchFeaturesTest.cmake` driven by `cmake -P` in
   `.github/workflows/ci.yml:96` — a structural property enforced in CI with no
   GPU. *Escape hatch:* an explicit `// DSR-ALLOW(<row-id>): <reason>` comment,
   which the checker counts separately and reports on every run, so a justified
   exception is visible rather than invisible. **0 in force today.**
   Per-row targets, restated against the corrected baseline: `S2` → 86 (its
   include/queue half is already done, §8.2), `S3` → 83, `S6` → ~37,
   `S4`+`S7` → **< 10, all of it class C/D**.
2. **Behaviour preservation is absolute on every row.** `S3`–`S7` are all
   refactors of code on the 27B/35B hot path. Each requires, STANDALONE:
   dgx 27B 235/235 + 35B 315/315 token-exact, Qwen3-Coder 6/6, Qwen3-dense
   16/16, OPT 6/6, DeepSeek-V2 8/8, all UNCHANGED; clean `-Werror` 0 warnings on
   GCC/Linux-CPU, nvcc 13.0 `sm_121a`, and AppleClang (Metal ON).
3. **`S3`–`S7` must be throughput-neutral**, proven by a same-binary A/B on the
   serving grid, not asserted. `qwen3_5.cpp`'s device tests sit inside the decode
   loop; replacing a compile-time-predictable `== kCUDA` with a table lookup is
   exactly the class of change that costs 1–2% if done carelessly. The
   `OpProvider` precedent (resolve once, cache in a relaxed atomic) is the
   required pattern.
4. **`S5` acceptance is a model, not a unit test:** OPT runs token-exact on
   Metal against our own CPU backend on the same M4 **with zero Metal kernels
   registered beyond the existing 10**, via the reference tier. That is the
   executable proof that op count became a performance budget.
5. **No performance claim is owed by any row here**, and none may be made for
   Metal or Vulkan until a model runs on them.

---

### Dependencies

| Need | Status |
|---|---|
| `S1` (metric) | **none — self-contained.** Depends on no row and blocks nothing; do it first so every later row is measured |
| `S2` (seam-fix residue) | already specced as [metal-mlx-reuse-study.md](metal-mlx-reuse-study.md) `W0b-1` items 1/3/4. **Currently owed and unclaimed.** Blocks any model on any non-CUDA backend |
| `S3` (platform capability fields) | `BACKEND-PLATFORM` (landed). Additive virtuals with defaults |
| `S5` (reference tier) | **`BACKEND-ACCEL-PROVIDER` (landed)** — needs no new mechanism, only a negative-priority provider and a device-neutral memory access rule. Restricted to unified-memory backends first (`Backend::UnifiedMemory()`, `backend.h:45`) |
| `S4` (LinearMethod/QuantMethod) | `S3`; overlaps `QUANT-GGUF-CIQ-GEMM` and the `QuantTypeTraits` split (reuse study `W0b-3`) — **`W0b-3` should be absorbed into `S4`, not done separately**; they are the same seam seen from the CPU side |
| `S6` (fast-path gates → capability queries) | `S4` (quant half), `S5` (safety net) |
| `S7` (layer library extraction) | `S4`, `S6`. The largest row; must be split per layer family and must not be attempted while a serving-gate claim owns `qwen3_5.cpp` |
| `S8` (XPU data-only) | `S3`. No hardware needed — selection is pure data |
| Cross-claim hazard | `qwen3_5.cpp` is the single hottest contended file in the tree (`CLAIM-SERVE-GATE-*`, `CLAIM-EW-NORM-*`, `CLAIM-FP4-*`). **`S6`/`S7` must be scheduled in a window with no active `qwen3_5.cpp` claim**, or they will lose every merge |
| Hardware | `S1` none; `S2`/`S3`/`S4`/`S6`/`S7` need dgx for the regression gate; `S5` needs the M4; `S8` none |

---

### Work breakdown

Ranked by (leakage removed + additivity gained) ÷ effort. Each row is a separate
claim; none is claimed by this audit.

| Rank | ID | Work | DSR after | Depends | Why here |
|---:|---|---|---:|---|---|
| **1** | `S1` | **The DSR metric + CI ratchet — `DONE` 2026-07-22 (`CLAIM-BACKEND-SEAM-S1-1`).** [`scripts/check-device-leakage.py`](../../scripts/check-device-leakage.py) + [baseline](../../scripts/device-leakage-baseline.json) + CI job `device-leakage` + [24-case mutation suite](../../tests/scripts/test_device_leakage.py). Per-file/per-bucket allowlist with exact counts and reasons, `DSR-ALLOW` escape hatch, per-bucket AND total monotonic decrease | **86** (baseline, §8) | — | Cheapest row in the plan and it guards every row after it. **The counts already regrew 54→63 and 13→16 in days of well-executed work** — without this, every reduction below decays |
| **2** | `S2` | **Reuse-study `W0b-1` residue** — **the include + hard-coded-queue half is ALREADY DONE on main** (§8.2): `model_loader.cpp` now calls `SelectQueue()`, and all four "unconditional" CUDA includes were in fact already `#ifdef`-guarded (an audit measurement error). Remaining: Metal `get_attn_backend_priority()` | 86 | `S1` | Class E and class F are both empty at `18094ee`; what is left is the Metal priority list, which the metric does not count |
| **3** | `S3` | **Platform capability fields** mirroring `interface.py:914,933,977,1058,1133,1153` + `is_device_capability_family:466`; migrate the **3** remaining class-D sites off `is_cuda()`/`kCUDA` (`runner.cpp:702,1080`, `qwen3_5_dense_weights.cpp:115` — §8.3) | 83 | `S2` | Pure mirror of upstream, mechanical, and it is what makes ROCm a *config* rather than a port. Highest fidelity-per-line in the plan |
| **4** | `S5` | **Portable reference tier** — register CPU kernels as negative-priority `OpProvider` entries on unified-memory devices, so an unregistered op FALLS BACK instead of throwing. Mirrors `CustomOp.forward_native` | 84 | `S3` | **Changes the economics of every future backend**: op count stops being a correctness gate. Uses only the mechanism we already shipped. Gate = OPT on Metal token-exact with today's 10 kernels |
| **5** | `S4` | **`QuantizationConfig` + `LinearMethod`/`QuantMethod`** keyed `(scheme, DeviceType)`, mirroring `base_config.py:20-229` + `linear.py:141-230`. Absorbs the `QuantTypeTraits` split (`W0b-3`) and replaces `IsNvfp4()` tensor-name probes | ~55 | `S3` | **The missing coarse seam** (§5). Biggest single leakage reduction available, and the only one that makes a new accelerator's quant work O(1) per scheme instead of O(models) |
| **6** | `S6` | **Class A + B: fast-path gates become capability queries.** `env && device==kCUDA && shape` → `OpRegistered(op, device) && supports(shape)`; delete the 5 `VT_CHECK(device==kCUDA)` | ~37 | `S4`, `S5` | Removes the largest single class (**46 sites**, re-measured §8.3). Deferred to rank 6 only because it is hot-path-sensitive and needs `S5` as the safety net and `S4` for the quant half |
| **7** | `S7` | **Extract the layer library** — `layers/{linear,layernorm,rotary,activation,fused_moe,gdn}` mirroring `model_executor/layers/`, shrinking `qwen3_5.cpp` toward `qwen3_next.py`'s shape | **< 10** | `S4`, `S6` | The structural end-state and the literal answer to "ride the same logical implementations". Largest and riskiest; must be split per family and scheduled against `qwen3_5.cpp` claim windows |
| **8** | `S8` | **XPU platform, data only** (`xpu.py:121-167` selector policy, no kernels) | < 10 | `S3` | The cheap falsification test: if a THIRD upstream platform ports mechanically with no engine edit, the seam claim is proven rather than argued. Needs no hardware |

---

### Risks/decisions

| # | Risk / decision | Disposition |
|---|---|---|
| 1 | **DECISION — Metal and Vulkan are permitted new PLATFORMS, not a new ARCHITECTURE.** Upstream has neither, so there is nothing to mirror for the *platform file*; but everything above it (Platform contract, attention contract, quant method dispatch, native fallback) is mirrored even for them | Recorded here and consistent with [backends.md](../backends.md) + [porting-inventory.md](../porting-inventory.md) §9. This audit adds the binding half: absence of an upstream *platform* never licenses absence of an upstream *seam* |
| 2 | **RISK — the metric punishes the wrong thing.** A naive `grep kCUDA` count would penalize adding a backend (every backend adds registrar keys) and would be gamed by moving a test behind a helper | Allowlist is per-file with stated reasons and covers exactly the platform legs and registrars (16 sites today). A helper that hides a device test still trips it, because the helper itself lives in the shared layer and is counted. The mutation suite (`S1`) is what proves this, and it is mandatory |
| 3 | **RISK — a ratchet becomes a rubber stamp** via liberal `DSR-ALLOW` | Every `DSR-ALLOW` names a row ID and is counted and PRINTED separately by the checker, so the exception budget is visible in CI output. Same posture as the record checker: repair the record, do not weaken the checker |
| 4 | **RISK — `S6`/`S7` cost throughput.** Device tests sit inside the decode loop; a table lookup per step is a real cost | Gate 3: same-binary A/B on the serving grid, not an argument. Required pattern is `OpProvider`'s: resolve once, cache in a relaxed atomic, memoize negatives |
| 5 | **RISK — `S7` merge-loses against serving-gate claims** on `qwen3_5.cpp` | Scheduling constraint, stated in § Dependencies: `S6`/`S7` need a window with no active `qwen3_5.cpp` claim. Split per layer family so each piece is small |
| 6 | **`OpProvider` divergence** — a unification upstream does not have, in a project whose purpose is mechanical upstream porting (§6) | **Accepted, with one binding rule recorded:** `OpProvider` selects among implementations of an op that is ALREADY DEFINED. It must never become where a POLICY decision lives — which quant scheme, which attention backend, which dtype, which block size. Those map to `Platform` / the attention registry / `LinearMethod`, which ARE upstream's seams. Violating this is what would break mechanical porting; the mechanism itself does not |
| 7 | **RISK — `S5`'s reference tier hides a missing kernel** and a backend silently runs at CPU speed | `OpProvider` already carries `selections`/`declines`/`last_selected` + `VT_OP_PROVIDER_STATS`. `S5` must additionally make reference-tier selection **loud**: counted, dumpable, and asserted zero in any performance arm. This is fan-out spike Risk 4 in a new costume and gets the same treatment |
| 8 | **RISK — `S5` is unsound on discrete memory.** A CPU kernel reading a device pointer works on unified memory (GB10, Apple, integrated Vulkan) and is memory corruption on a discrete GPU | Scope `S5` to `Backend::UnifiedMemory() == true` in its first form; a staging-copy variant for discrete backends is a later, separately-gated row. Stated up front rather than discovered |
| 9 | **HONEST LIMIT — this audit did not build or run anything.** Every count is static analysis of the tree at `72f5db2`; every upstream claim is `file:line` at pin `e24d1b24` | Stated. The one behavioural claim reused (Vulkan V1 edited zero pre-existing source files) is carried from [backend-fanout-metal-vulkan-xpu.md](backend-fanout-metal-vulkan-xpu.md) § V1 landed, where it was measured |
| 10 | **NOT REOPENED:** MLX as an optional gated provider; native MSL as the Metal default; OPT as the first bring-up model; NMSE ≤ 5e-4 rather than bit-exactness across devices; Vulkan's committed-SPIR-V route | All are settled by the two companion specs and this audit endorses them unchanged |

---

## 4. Granularity — is a new accelerator implementing the right amount? (the deep question)

### 4.1 Our seam is NOT finer than vLLM's. That premise is wrong.

The brief supposed our ~75-entry `vt::` op table is "closer to ggml's
granularity" than vLLM's "torch ops / `CustomOp` / layer classes". Measured, that
is not so. vLLM has **42 `@CustomOp.register` sites** (`silu_and_mul`,
`rms_norm`, `rotary_embedding`, `quant_fp8`, `grouped_topk`,
`chunk_gated_delta_rule`, …) — the same *kind* of object as our `OpId`s, at the
same *level*, in the same 40–80 range. Where vLLM appears coarser is that
`nn.Linear`/`matmul`/`softmax` are torch primitives it never had to name, so they
do not appear in the count. Adding those back, the two seams are the same
granularity.

**So coarsening the op table is not the lever, and would be a mistake.** It is
the one place where a per-accelerator swap is genuinely what we want, it is
already device-additive, and `OpProvider` just gave it multi-implementation
selection.

### 4.2 But vLLM's per-accelerator FLOOR is zero ops, and ours is N.

`CustomOp.forward_native` (`custom_op.py:138`) is a pure-torch implementation of
every one of the 42. `dispatch_forward` (`:174`) picks `forward_cuda`/`_hip`/
`_xpu`/`_cpu`/`_tpu`/`_oot` **if the platform provides one, and `forward_native`
otherwise**. A brand-new vLLM platform that implements **zero** custom kernels
still produces correct output for every model, because torch is the universal
portable op layer beneath.

We have no such layer. `vt::GetOp` **throws** (`src/vt/ops.cpp:104-111`) — a
documented, tested, deliberate state, and the right call for a project with no
torch. But the consequence is the asymmetry:

| | vLLM | vllm.cpp today |
|---|---|---|
| ops to be **CORRECT** on a new accelerator | **0** | **6–7** for the first model, ~55 for the non-quant surface, 64 for CPU-equivalence |
| ops to be **FAST** | as many as you profile into | the same |
| a partially-implemented backend | runs, slowly | **does not run** |

**So the honest answer to "is a new accelerator implementing the right amount?"
is: the right amount for SPEED, the wrong amount for CORRECTNESS.** 6 kernels to
make OPT fast on Metal is an excellent number. 6 kernels before OPT produces a
single token on Metal is a needless cliff — it is why Metal and Vulkan have been
`ACTIVE` skeletons with **10 and 8** registered ops and **no model** for their
entire existence.

### 4.3 The fix is a fallback tier, not a coarser seam.

Register the **existing CPU kernels** as negative-priority `OpProvider` entries
on unified-memory devices. On GB10-Vulkan, Apple-Metal and CPU — every backend we
have or plan except a discrete GPU — a host pointer and a device pointer are the
same pointer (`Backend::UnifiedMemory()`, `include/vt/backend.h:45`; MLX's own
allocator makes the identical assumption, `allocator.cpp:14`
`ResourceStorageModeShared`). The mechanism needs **nothing new**: `OpProvider`
already has priority ordering, a capability predicate, decline-and-fallback, and
selection stats to prove which tier ran.

That is work row `S5`, and it is ranked 4th because it changes the economics of
every backend that comes after it.

---

## 5. Is the missing `LinearMethod` layer the coarse seam we need? — YES

**The evidence is distributional and it is decisive.**

- vLLM puts **199 of its 544** device predicates in `model_executor/layers/`,
  and **14** in `model_executor/models/` (9 of 287 files).
- We have no `layers/` library for Linear/MoE/GDN/norm, and **72 of our 94**
  device references are in model files, **67 in one**.
- Our gate model's upstream twin, `qwen3_next.py`, is **802 lines / 1 predicate**.
  Ours is **6,389 lines / 67**. The missing ~5,600 lines are not extra features —
  they are upstream's `layers/fused_moe/`, `layers/mamba/gdn/`, `layers/linear.py`,
  `layers/layernorm.py` and `layers/quantization/`, inlined.

**Quantization is the sharpest instance, and it is exactly the per-accelerator
axis.** Upstream: `QuantizationConfig.get_quant_method(layer, prefix)`
(`base_config.py:180`) returns a `LinearMethodBase` (`linear.py:141`) with
`create_weights` / `process_weights_after_loading` / `apply` — one object that
owns *how this scheme's weights are laid out and multiplied on this device*, with
18 config modules and per-device branches inside them
(`marlin_utils.py`, `fp8_utils.py`, `mxfp4_utils.py` all carry `is_rocm` arms).
Ours: `include/vllm/model_executor/models/qwen3.h:60`

```cpp
bool IsNvfp4() const { return !qkv_proj_fp4.Empty(); }
```

— scheme detection by **tensor-name presence, re-implemented per model**, with
the device branch bolted on at each call site
(`dense_attn_block.h:329,466`, `qwen3.cpp:78`, and the ~34 `qwen3_5.cpp` sites).

**Consequences, stated plainly:**
1. Adding a **scheme** costs an edit in every model that should support it.
2. Adding an **accelerator** costs an edit at every `device == kCUDA` quant gate
   in every model — which is precisely the work the user asked to not have to do.
3. There is no place to *put* a Metal or Vulkan quant kernel choice, so
   `W0b-3`'s `QuantTypeTraits` split has nowhere to land. That is why the reuse
   study found "the quant tier and the tactic registry want the same seam"
   (§3.4) — they want `LinearMethod`.

**So yes: `LinearMethod`/`QuantizationConfig` is the coarse seam §4 was looking
for.** It is not a *replacement* for the op table — the op table stays exactly as
it is. It is the **layer above** it, the one vLLM has and we skipped, where
"which kernel family does this (scheme, device) use" is answered once instead of
per model per call site. Work row `S4`; and `S4` + `S7` together are what take
DSR below 10.

---

## 6. Does `vt::OpProvider` correspond to anything upstream? — No, and that is both

**What it is.** `include/vt/op_provider.h` — our generalization of
`src/vt/cuda/cuda_arch_tactics.{h,cu}` out of `vt::cuda` and up to
`(OpId, DeviceType)`: capacity-bounded static storage, device-neutral
`ProviderCaps`, a `supports()` predicate, deterministic `(priority DESC,
name ASC)` selection, per-call decline-and-fall-back (`GetOpFallback`), and
selection stats.

**Upstream has no counterpart, because upstream solves this three separate ways:**
capability-gated kernel choice inside quantization methods
(`marlin_utils.py`, `fp8_utils.py`); the attention-backend registry
(`registry.py` + `get_attn_backend_cls`); and torch dispatch / vendor heuristics
(`cublasLtMatmulAlgoGetHeuristic`, flashinfer's per-arch tactic table at
`fp4_gemm_cutlass_template_sm120.h:187-220`). Our unification is genuinely a
thing upstream does not have.

**Assessment — it is an improvement, and the porting risk is real but bounded and
now fenced:**

*Improvement.* One mechanism serves five platforms and four provider shapes;
selection is deterministic where the old table was a nondeterministic
static-init race; and it is observable, which none of the three upstream
mechanisms is from vLLM's own side.

*Why the porting risk is smaller than it looks.* Mechanical portability is a
property of **where an upstream change lands**, not of how many mechanisms exist.
A vLLM PR adding a new fp4 tactic lands in a kernel file and a capability
predicate — and lands in ours the same way. `OpProvider` sits *under* the op
table with `RegisterOp`/`GetOp`/`OpRegistered` signature-identical and all ~70 op
wrappers byte-unchanged; nothing upstream ports *to* it, so nothing upstream can
mis-port *because of* it.

*Where the risk is real.* If `OpProvider` becomes the place we express things
upstream expresses in `Platform` methods or in `QuantizationConfig.get_quant_method`,
then upstream PRs stop mapping — because the corresponding upstream change would
have no local home. **Binding rule, recorded as Risks/decisions 6:** *`OpProvider`
selects among implementations of an op that is already defined. Policy decisions
— which quant scheme, which attention backend, which dtype, which block size —
live in `Platform`, the attention registry, and `LinearMethod`, which are
upstream's own seams.* With that rule, the divergence is an implementation
detail below the mirrored surface. Without it, it is a slow drift, and the fact
that `S4` does not exist yet makes the drift *likely* — quant selection has
nowhere else to go today, so `OpProvider` is exactly where it would leak. That is
a further reason `S4` outranks convenience.

---

## 7. What this audit does NOT claim

- It did not build, test, benchmark or run anything. No performance or
  correctness result is produced or implied.
- It does not claim Metal or Vulkan work. **No model runs on either**; they are
  gated skeletons with 10 and 8 registered ops, exactly as their specs state.
- It does not re-open MLX's disposition, the bring-up model choice, the numeric
  tolerance, or any per-backend port map.
- The DSR baseline of **94** is a static count over `src/vllm/` + `include/vllm/`
  at `72f5db2` under the stated allowlist. It is a metric definition, not a
  quality score, and it is meaningless without the allowlist that accompanies it.
  **Superseded by §8: the as-built metric measures 86, and 6 of the 8-point
  difference is a correction to this audit's own composition, not progress.**

---

## 8. `S1` LANDED — the as-built ratchet (2026-07-22, `CLAIM-BACKEND-SEAM-S1-1`)

Ships `scripts/check-device-leakage.py`, `scripts/device-leakage-baseline.json`,
CI job `device-leakage` (`.github/workflows/ci.yml`), and
`tests/scripts/test_device_leakage.py` (24 hard expectations). **No file under
`src/` or `include/` was touched**, so no behaviour changed and no build or GPU
run is owed. The checker runs standalone with no CUDA toolkit and no GPU, exactly
like `cmake/CudaArchFeaturesTest.cmake`.

### 8.1 The baseline moved 94 → 86, and most of that is a correction

The audit instructed that the count be re-derived rather than carried. It was.
Running the as-built checker at the audit's own base `72f5db2` yields **88**, and
at `18094ee` **86**. So the 8-point difference splits cleanly:

| Bucket | Audit @ `72f5db2` | As-built @ `72f5db2` | As-built @ `18094ee` | Why it moved |
|---|---:|---:|---:|---|
| `kCUDA` | 47 | 45 | **44** | −2 **composition**: the audit counted `kCUDA` appearing in COMMENTS while excluding comments from its `is_cuda()` bucket. −1 **real**: `model_loader.cpp`'s hard-coded `GetBackend(kCUDA)` became `SelectQueue()` |
| `is_cuda()` | 11 | 11 | **10** | −1 **real**: `runner.cpp:516` migrated off `is_cuda()` |
| unconditional CUDA includes | 4 | 0 | **0** | −4 **composition — an audit measurement ERROR.** All four cited sites (`model_loader.cpp:22`, `runner.cpp:30`, `qwen3_5.cpp:40,46`) were ALREADY inside `#ifdef VLLM_CPP_CUDA` / `#ifdef VT_*` guards at `72f5db2`. The audit grepped for the include without reading the surrounding preprocessor context |
| `#ifdef VT_*` | 32 | 32 | **32** | unchanged |
| **DSR** | **94** | **88** | **86** | **−6 composition, −2 real reduction on main** |

**Two deliberate corrections to the audit's composition**, both in the direction
of making the metric mean what it says:

1. **Comments and string literals are stripped before matching, uniformly.** A
   prose mention of a device name, or a `VT_CHECK` diagnostic that says "needs
   kCUDA", is not a branch. The audit applied this rule to `is_cuda()` and not to
   `kCUDA`; applying it to both is what makes the number a count of *behaviour*.
   A useful side effect: the allowlist shrank from **16 sites to 10**, because 6
   of the audit's allowlisted entries were comments that now need no exemption at
   all — and an exemption that exists only to excuse a comment is an exemption
   that could later shelter a real branch.
2. **An include inside a CUDA/`VT_*` preprocessor guard is not counted.** The
   bucket's stated meaning is "a non-CUDA build does not compile"; a guarded
   include compiles fine. The checker tracks `#if`/`#elif`/`#else`/`#endif` depth,
   and — asserted by mutant `M5` — a CUDA include in the `#else` (portable) arm
   of such a guard **is** counted, because that arm is the non-CUDA build.

### 8.2 The seven-class composition, re-verified against `18094ee`

The audit's classification was checked site-by-site, not assumed. It holds, with
three corrections — all of which *strengthen* its central argument:

| Class | Audit | Re-measured | Verdict |
|---|---:|---:|---|
| **A** — fast-path availability (`env && device==kCUDA && shape`) | ~45 | **46** | **Confirmed.** Still the largest class and still not a device question — it asks whether a fused/quant kernel is available, which the op/provider table already answers |
| **B** — `VT_CHECK(device == kCUDA)` duplicating `GetOp`'s own throw | 5 | **5** | **Confirmed exactly** — `qwen3_5.cpp:1144,1166,2780,3151` + `:5298` |
| **C** — platform-definition keys (allowlisted) | 16 | **10** | **Corrected down.** 6 of the audit's 16 were comments; the as-built metric ignores comments entirely, so they need no allowlist entry. The remaining 10 are real code: 4 in `platforms/cuda.cpp`, 1 priority walk, 2 for the `is_cuda()` definition, 3 registrar keys |
| **D** — real platform capability asked via the wrong predicate | 6 | **3** | **Corrected down.** `runner.cpp:516` was migrated off `is_cuda()` on main; and `deepseek_v2.cpp:311` (`GroupedMoeEligible`) reads as class **A** on inspection, not D — it gates the availability of `vt::MoeGroupedGemmBf16`, not a device policy. Genuine class D is `runner.cpp:702,1080` (device combine/scatter residency) and `qwen3_5_dense_weights.cpp:115` (direct-device-load residency) |
| **E** — hard-coded device choice | 1 | **0** | **Fixed on main** since the audit |
| **F** — unconditional CUDA includes | 4 | **0** | **Never existed** — audit measurement error (§8.1) |
| **G** — `#ifdef VT_*` build gates | 32 | **32** | **Confirmed exactly** |

**The audit's headline verdict survives its own corrections, and gets sharper.**
It said "only 6 of 94 are genuinely device-dependent behaviour". The re-measured
answer is **3 of 86** — 46 are class A, 5 are class B, 32 are class G. So
**96.5% of the shared layer's device references are not device policy**: they are
build config, or a duplicated throw, or the shared layer asking a question
`vt::GetOp`/`OpProvider` already answers. That is the argument for `S4`/`S6`/`S7`,
stated more strongly than the audit could state it.

### 8.3 How the ratchet enforces monotonic decrease

- **Per bucket AND in total.** Any bucket above its baseline fails. This is
  stricter than the audit specified, deliberately: a total-only check would let a
  removed `#ifdef VT_*` pay for a newly added `kCUDA`, which is precisely the
  silent regrowth the row exists to stop.
- **A reduction fails too, until the baseline is lowered in the same commit.**
  A bucket *below* its baseline is an error naming the exact command
  (`--write-baseline`). That is what makes it a ratchet rather than a threshold:
  the number cannot drift down informally and then drift back up unnoticed.
- **`--write-baseline` refuses to write a higher number** even when asked
  directly (mutant `M12`), so the one move the ratchet must never make is not
  available through the tool that maintains it.
- **The allowlist is itself a ratchet.** Entries carry an *exact* expected count,
  not a blanket file exemption. A third `kCUDA` in a two-registrar file fails
  (`M8`); so does removing one of the two (`M9`). Only `platforms/cuda.cpp` — the
  file that IS the CUDA platform — is exempt wholesale.
- **The escape hatch is loud.** `// DSR-ALLOW(<row-id>): <reason>` is printed on
  every run, pass or fail, with its count; a marker without a row id and a reason
  grants no exemption (`M15`). **0 are in force today.**

### 8.4 Mutation evidence — the checker demonstrably catches leaks

24/24 in `tests/scripts/test_device_leakage.py`, each planting one defect in a
synthetic shared-layer tree and requiring the specific error:

| Proves | Mutants |
|---|---|
| an added device reference FAILS | `M1` `kCUDA`, `M2` `is_cuda()`, `M3` unguarded CUDA include, `M6` `#ifdef VT_*`, `M18` the same test hidden behind a helper (audit Risk 2 — the metric is not gameable by indirection, because the helper is in the shared layer too) |
| adding a BACKEND never trips it | `M7` (new platform file + extra methods on the CUDA leg → still green) |
| the allowlist cannot rot into a blanket exemption | `M8` (extra ref in a budgeted file fails), `M9` (stale-wide allowlist fails) |
| monotonic decrease | `M10` (reduction fails until declared), `M11` (same-commit lowering passes), `M12` (upward write refused) |
| the escape hatch is real, bounded, loud | `M13`, `M14`, `M15` |
| composition is honest in both directions | `M4`/`M5` (guarded vs `#else`-arm include), `M16`/`M17` (comment and string mentions not counted), `M19` (`src/vt/cuda/` device legs not scanned at all) |
| the real tree, not just mutants | baseline matches exactly; every allowlisted path exists; every allowlist entry states a reason; **`qwen3_5.cpp` holds a strict majority of all DSR** — the audit's headline finding asserted as an executable fact, so if it ever stops being true the `S4`/`S7` plan is known to be mis-aimed |

### 8.5 Where the leakage sits at `18094ee` (`--report`)

`qwen3_5.cpp` = **66 of 86 (77%)** — up from the audit's 71% share, because the
two real reductions since `72f5db2` both landed *outside* it. Everything else:
`dense_nvfp4_gemm.h` 5, `deepseek_v2.cpp` 2, `qwen3.cpp` 2, `runner.cpp` 2, and
seven files with 1 each.
