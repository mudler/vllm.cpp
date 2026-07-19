# SPIKE: Portable Automatic Op-Fusion Framework (`KERNEL-FUSION-FRAMEWORK`)

*Lifecycle: `SPIKE`. Roadmap_v1 ORDER-1 cornerstone (extensibility architecture). READ-ONLY design — no code, no GPU in this spike.*
*Grounding pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; deps (flashinfer) `/home/mudler/_git/flashinfer-ref`; our tree `/home/mudler/_git/vllm.cpp`.*
*Supersedes-and-absorbs: `specs/fusion-architecture-2026-07-08.md` (the design-panel TDR), reconciled with the 2026-07-19 measured attribution (state/ledger tasks #61/#62).*

---

## 0. Thesis (the spine)

This framework is **not primarily a perf lever**. The measured 35B prefill gap is compute-bound with a fusion ceiling of **~3.5%/step @ c1** (35B c1 190 vs 183 ms; 27B c1 is already ours-faster) — stated honestly and up front (task #61). Prefill is **99.4% GPU-busy**; ~71% is real compute at parity (Marlin MoE / FLA GDN / fp8 dense / flash-attn), and the residual ~20% "glue" is only per-step-modest wall.

Its **primary value is EXTENSIBILITY + mechanical upstream-sync**, delivered by one structural move that mirrors vLLM's own `CustomOp` seam (`vllm/model_executor/custom_op.py:103`): **a fusion is DECLARED ONCE (backend-agnostic) and REALIZED PER-BACKEND** through the existing `vt::` op table. That makes three growth axes additive at once:

- **A new vLLM fusion PR ports as ONE catalog declaration** — not hand-wiring at call sites.
- **A new GPU/backend realizes the catalog** — an additive file; a backend with no fused kernel inherits the CPU-composite fallback (correct on day 1).
- **A new model declares its fusion patterns** — additive.

The design is judged by the **PR-#4 test**: does adding the next fusion / GPU / model touch **fewer shared files**? Today each new fusion is a bespoke kernel hand-called at model sites in `qwen3_5.cpp`; the framework retires that scatter into a declarative catalog + one interpreter per backend.

---

## 1. Upstream inventory — vLLM's complete fusion surface @ `e24d1b24`

vLLM's fusion is a **two-layer** pipeline (grounded): model code emits backend-agnostic `vllm_ir.*` ops + `CustomOp`s → **pattern-match-and-replace passes** rewrite op-chains into fused `torch.ops._C.*` / `torch.ops.vllm.*` ops → `VllmIRLoweringPass` lowers remaining IR → cleanup/functionalization. Separately, Inductor does **generic horizontal codegen** (combo_kernels) that is *not* a pattern pass.

### 1a. The pass manager (the assembly + order = our catalog order)

`PostGradPassManager(CustomGraphPass)` — `vllm/compilation/passes/pass_manager.py:86`; assembled in `configure()` (`:138-200`). Passes are appended in a fixed order, each gated by a `PassConfig` flag. **This ordered, config-gated list IS the port backlog shape** — our catalog is the C++ analog of `self.passes`:

| # | pass_manager line | Pass | Gate (`PassConfig`) |
|---|---|---|---|
| 1 | `:143` | `NoOpEliminationPass` | `eliminate_noops` |
| 2 | `:146` | `SequenceParallelismPass` | `enable_sp` |
| 3 | `:148` | `AsyncTPPass` | `enable_sp and fuse_gemm_comms` |
| 4 | `:151` | `RocmAiter…PadFusionPass` | ROCm+`fuse_act_padding` |
| 5 | `:156` | `AllReduceFusionPass` (ROCm variant else) | `fuse_allreduce_rms` |
| 6 | `:162` | **`RMSNormQuantFusionPass`** | `fuse_norm_quant` |
| 7 | `:169` | **`ActivationQuantFusionPass`** | `fuse_act_quant` |
| 8 | `:174` | `MLADualRMSNormFusionPass` | ROCm+`fuse_mla_dual_rms_norm` |
| 9 | `:181` | `RopeKVCacheFusionPass` (+split/scatter) | `fuse_rope_kvcache` |
| 10 | `:186` | `MLARoPEKVCacheCatFusionPass` | `fuse_rope_kvcache_cat_mla` |
| 11 | `:189` | `AttnQuantFusionPass`, `MLAAttnQuantFusionPass` | `fuse_attn_quant` |
| 12 | `:193` | **`QKNormRoPEFusionPass`** (+split-coalesce) | `enable_qk_norm_rope_fusion` |

Fixed tail (`:197-200`): `VllmIRLoweringPass`, `UnsafeCloneEliminationPass`, `PostCleanupPass`, `FixFunctionalizationPass` (runs last). `uuid()` (`:206-228`) hashes the pass set for Inductor's code cache.

### 1b. The enumerated fusion passes = the FINITE port backlog

For each: class def, the source subgraph it matches, the fused op it inserts, and the kernel that lands. **Bold** = the Class-A HBM-pass chains a portable framework can own (and which we already hand-mirror).

- **`RMSNormQuantFusionPass`** — `passes/fusion/rms_quant_fusion.py:618`. Matches `rms_norm`/`fused_add_rms_norm` + FP8/NVFP4 quant. Variants (each a registered pattern class): static-FP8 no-res (`:171`→`_C.rms_norm_static_fp8_quant`), **static-FP8 +residual** (`:226`→`_C.fused_add_rms_norm_static_fp8_quant`), dynamic-per-token-FP8 ±res (`:494`/`:553`→`_C.rms_norm_dynamic_per_token_quant`), FP8 group/block-128&64 ±res (`:400`/`:293`→`_C.rms_norm_per_block_quant`). Kernels: csrc `layernorm_quant_kernels`. → **we mirror as `vt::RmsNormQuantFp8` (see §2).**
- **`ActivationQuantFusionPass`** — `passes/fusion/act_quant_fusion.py:283`. Matches `_C.silu_and_mul` + quant. Variants: +static-FP8 (`:81`→`_C.silu_and_mul_quant`), **+NVFP4** (`:128`→`_C.silu_and_mul_nvfp4_quant`), +FP8 block (`:184`→`_C.silu_and_mul_per_block_quant`). → **we mirror as `vt::SiluMulFp4Quant`; the sigmoid-gate analog is `vt::SigmoidGateFp4Quant`.**
- **`QKNormRoPEFusionPass`** — `passes/fusion/qk_norm_rope_fusion.py:188` (pattern `:31`). Matches split-qkv → per-head reshape → `rms_norm(q)`,`rms_norm(k)` → reshape → `rotary_embedding` (`:118-144`). Inserts `_C.fused_qk_norm_rope` (`:26`). → **we mirror as `vt::AttnQkNormRopeGate`.**
- `AttnQuantFusionPass` — `passes/fusion/attn_quant_fusion.py:362`. Folds post-attention quant INTO `torch.ops.vllm.unified_attention_with_output`'s `output_scale`/`output_block_scale` (attn writes quantized output directly). Static-FP8 (`:38`), NVFP4 (`:172`). Gated on `layer.impl.fused_output_quant_supported(...)` (`:391`). → **epilogue-into-attention class; realized per attn backend, not a standalone chain.**
- `RopeKVCacheFusionPass` — `passes/fusion/rope_kvcache_fusion.py:412`. `rotary_embedding` + `unified_kv_cache_update` → `vllm.fused_rope_and_unified_kv_cache_update` (impl `:46-77`); variant also folds static-FP8 Q-quant (`:100`). Decode-only (`is_applicable_for_range` `:461-465`).
- `MLAAttnQuantFusionPass` (`mla_attn_quant_fusion.py:574`) and `MLARoPEKVCacheCatFusionPass` (`mla_rope_kvcache_cat_fusion.py:240`) — MLA-specific (DeepSeek); not on our Qwen gate hot path. Inventory-only.
- `SequenceParallelismPass` — `passes/fusion/sequence_parallelism.py:498`. Rewrites `all_reduce + rms_norm` → `reduce_scatter → rms_norm(shard) → all_gather` (`:161-183`); +FP8/NVFP4 variants. **Multi-GPU only — HW-blocked here (no multi-GPU), inventory-only.**
- `AsyncTPPass` — `passes/fusion/collective_fusion.py:900`. GEMM⊕collective into `symm_mem.fused_matmul_reduce_scatter` (`:359`) / `fused_all_gather_matmul` (`:396`). **Multi-GPU only — inventory-only.**
- `AllReduceFusionPass` — `passes/fusion/allreduce_rms_fusion.py:932`. `all_reduce(+add)+rms_norm(+quant)` → `vllm.flashinfer_trtllm_fused_allreduce_norm` (`:286`) via `flashinfer_comm.allreduce_fusion`. Disabled for `tp_size<=1` (`:935`). **Multi-GPU only — inventory-only.**
- `rocm_aiter_fusion.py` (`:546`, `:674`) — ROCm/AITER-only; imported only when `rocm_aiter_ops.is_enabled()` (`pass_manager.py:25`). **Backend-specific realization — inventory-only until ROCm.**

**Utility/IR passes** (graph hygiene, port only if/when we build a real graph layer): `NoOpEliminationPass` (`utility/noop_elimination.py:18`, removes reshape/slice — *prerequisite for RMS-quant matching*), `ScatterSplitReplacementPass` (`:35`), `SplitCoalescingPass` (`:31`), `PostCleanupPass` (`:8`, DCE), `FixFunctionalizationPass` (`:19`, de-functionalize copies, runs last); IR: `VllmIRLoweringPass` (`ir/lowering_pass.py:25`), `UnsafeCloneEliminationPass` (`ir/clone_elimination.py:72`), `VllmIRInplaceFunctionalizationPass` (`ir/inplace_functionalization.py:21`).

**Pattern-matching infra** (the "how a pass is a pass"): `VllmPatternMatcherPass`/`VllmFusionPatternMatcherPass` (`vllm_inductor_pass.py:95,296`) with `register()`→`pm.register_replacement` (`:308`); `MatcherCustomOp` (`matcher_utils.py:52`) re-implements a `CustomOp`'s `forward_custom`/`forward_native` split so a pattern matches whichever realization ran. `MatcherQuantFP8` (`:307`), `MatcherRotaryEmbedding` (`:86`), `MatcherSiluAndMul` (`:458`), `MatcherRMSNormGated` (`:165`).

### 1c. The catalog-SELECT vs CODEGEN boundary (scope fence)

- **Pattern-match-and-replace passes (§1b) are PORTABLE — this is what we mirror.** They are a finite, enumerated set of *named subgraph→fused-op* rules. Porting one = declaring one recipe.
- **Inductor generic codegen (combo_kernels) is OUT OF SCOPE.** `combo_kernels=True` is injected into `inductor_compile_config` at `vllm/config/compilation.py:967` (with `benchmark_combo_kernel` `:968`) — it is a flag to Inductor's *horizontal-fusion codegen*, **not** a vLLM pattern class (grep confirms no `combo_kernels` under `passes/`). This is the WHOLE-GRAPH horizontal fusion of the residual glue (the ~20% task #61 attributes the per-step gap to). Replicating it needs a runtime code generator — **future work, explicitly not in this framework** (a codegen backend could later realize catalog entries, but we SELECT pre-written portable fused kernels, we do not runtime-codegen novel kernels).

### 1d. The declare-once/realize-per-backend analog we mirror: `CustomOp`

`CustomOp(nn.Module)` — `vllm/model_executor/custom_op.py:103`. **This is the exact seam our framework mirrors:**
- Declare once: `@CustomOp.register(name)` (`:307`) → `op_registry`.
- Realize per backend: `forward_native` (composite PyTorch fallback, `:138`), `forward_cuda` (`:146`), `forward_hip`→cuda (`:149`), `forward_xpu/cpu/tpu/oot`→native (`:153-172`).
- Dispatch: `dispatch_forward()` (`:174-207`) caches `_forward_method`; `enabled()` (`:271`) reads `custom_ops` list; if **disabled**, returns `forward_native` so **Inductor fuses it** (`:191`); if **enabled**, dispatches to the platform method (`:196`). `default_on()` (`:291`) = ON unless `custom_ops` contains `"none"` (the Inductor default).

The key insight for our design: **vLLM's `forward_native` is simultaneously the composite fallback AND the thing Inductor fuses.** Our `FusedRecipe` composite tier is the direct C++ analog — one declaration serves as both the oracle-composite and the fusion source.

### 1e. The kernels that actually land (flashinfer, for grounding realizations)

flashinfer (`/home/mudler/_git/flashinfer-ref`): `fused_add_rmsnorm` (`norm/__init__.py:240`), `gemma_fused_add_rmsnorm` (`:437`), `rmsnorm_quant` (`:186`), `fused_add_rmsnorm_quant` (`:294`) [FP8 = precompiled; FP4 dispatches to CuTe-DSL `cute_dsl/rmsnorm_fp4quant.py:776` / `add_rmsnorm_fp4quant.py:1047`], `silu_and_mul_scaled_nvfp4_experts_quantize` (`activation.py:204`). Note: inside `vllm/compilation/passes/` vLLM fuses to its **own** `_C.*`/`vllm.*` ops, reaching flashinfer only via `flashinfer.comm.allreduce_fusion` and the `mm_fp4`/`nvfp4_quantize` shims (`vllm/utils/flashinfer.py`). We mirror the **patterns**, realize the **kernels ourselves** (the ground-every-impl-in-upstream rule: cite the CuTe-DSL/csrc source we transcribe from).

---

## 2. Our current state

### 2a. The `vt::` op dispatch table — the natural interception point

`src/vt/ops.cpp`: a `Table()` of `void*` indexed `[OpId::kCount][kNumDeviceTypes]` (`ops.cpp:10-15`), populated by `RegisterOp(OpId, DeviceType, void*)` and read by `GetOp` (declared `ops.h:466-467`). Every op is a thin **validate-then-dispatch wrapper**: `void FusedChain(...)` at `ops.cpp:639` validates shapes/dtypes/devices at the chokepoint, then `reinterpret_cast<FusedChainFn>(GetOp(OpId::kFusedChain, q.device.type))(...)` (`:664`). `OpId` enum in `include/vt/ops.h:88-155` (154 ops, `kCount` sentinel). **This is exactly the seam a fusion realizes through — one `OpId`, N backend registrations, zero new dispatch machinery.**

### 2b. The hand-mirrored fusions already in-tree (the catalog-to-be)

Each is a bespoke `OpId` + `Fn` alias + CUDA kernel + CPU composite oracle + hand call-site in `qwen3_5.cpp`:

| vt op | Mirrors vLLM pass | CUDA kernel | CPU oracle | Hand call-sites in `qwen3_5.cpp` |
|---|---|---|---|---|
| `kRmsNormQuantFp8` (`ops.h:656`) | `RMSNormQuantFusionPass` static-FP8+res (`rms_quant_fusion.py:124`) | `cuda_ops.cu:395` | `cpu_ops.cpp:280` | `:2536`, `:4536` |
| `kRmsNormGatedQuantFp8` (`ops.h:660`) | gated-RMSNorm epilogue + FP8 quant | `cuda_gdn.cu:1610` | `cpu_ops.cpp:836` | `:2966`, `:3459` |
| `kSiluMulFp4Quant` (`ops.h:~327`) | `ActivationQuantFusionPass` NVFP4 (`act_quant_fusion.py:128`) | `cuda_ops.cu` | `cpu_ops.cpp` | `:4690` |
| `kSigmoidGateFp4Quant` | (`ActivationQuant` analog — sigmoid gate; `glue-fusion-2026-07-19.md`) | `cuda_ops.cu` | `cpu_ops.cpp` | `:1964` |
| `kAttnQkNormRopeGate` | `QKNormRoPEFusionPass` (`qk_norm_rope_fusion.py`) | `cuda_ops.cu:390` (block-reduce skeleton) | `cpu_ops.cpp` | `:3521`, `:3661` |
| `kMoeCombineGate` | (MoE combine+gate glue) | `cuda_ops.cu` | `cpu_ops.cpp` | `:4028`, `:4395` |
| `kGdnPostConv`/`kGdnGBeta`/`kGdnConvSplit` | FLA GDN post-conv glue | `cuda_gdn.cu` | `cpu_ops.cpp` | `:2920`, `:3330` |

**The problem the framework solves:** each row above is a full bespoke kernel wired by hand at named call sites; the CPU oracle is a *hand-kept parallel* of the CUDA kernel (drift risk); a new backend must re-hand-write every row (the PR-#4 anti-pattern reproduced N-kernels × M-backends).

### 2c. The TDR Phase-0 skeleton — LANDED but UNADOPTED and UNGENERALIZED

The 2026-07-08 design panel's "Tiered Declarative Recipe" reached a **partial Phase-0 implementation** already in-tree, which this spike must build on honestly:

- **Declaration:** `include/vt/fused_recipe.h` — `FusedRecipe` POD (`:72`): a `FStep[8]` (`:56`) opcode list. Opcodes `FOp` = `{kAdd,kMul,kRmsNorm}` (`:28`); reduce kind `FReduce` (`:37`); operand **roles** `FOperand` = `{kIn,kResidual,kWeight,kOut}` (`:46`). `FusedTier()` env selector (`VT_FUSED_TIER`, `:83`).
- **Catalog:** `include/vt/recipes.h` — exactly **one** recipe, `kFusedAddRmsNorm` (`:35`), transcribing the add+RMSNorm chain with an upstream citation.
- **Realization:** one `OpId::kFusedChain` (`ops.h:142`) + `FusedChainFn` (`ops.h:458`) + dispatch wrapper (`ops.cpp:639`). **Tier-0 composite** walks steps through registered primitives — CUDA `FusedChainComposite` (`cuda_ops.cu:1128`), CPU `FusedChainCompositeKernel` (`cpu_ops.cpp:1474`). **Tier-1 interpreter** single-pass — CUDA `FusedChainInterpKernel` (`cuda_ops.cu:1062`), CPU (`cpu_ops.cpp:1427`). Registered CPU `cpu_ops.cpp:1615`, CUDA `cuda_ops.cu:1220`.
- **Test:** `tests/vt/test_ops_fused_chain.cpp` — asserts Tier-0 == Tier-1 == the `vt::RmsNorm(residual)` golden, bit-identical, CPU primary + CUDA-on-dgx.

**Honest gaps in the landed skeleton** (this is what W1–W2 build):
1. **Unadopted:** `grep vt::FusedChain( src/` is empty — no model call site uses it; the model still hand-calls the bespoke ops (§2b).
2. **Not general enough to hold the catalog:** only 3 opcodes (no quant terminal, no silu, no rope, no gate) and a **fixed 4-role operand model** (`kIn/kResidual/kWeight/kOut`). The real fusions are multi-input: silu_and_mul needs (gate, up); qk_norm_rope needs (q, k, cos, sin, gate); sigmoid-gate needs (attn, gate). The POD cannot express them yet.
3. **No pattern layer:** recipes are hand-picked constants, not matched against a captured forward graph.

### 2d. The extensibility seams the framework must COMPOSE with (all landed, ORDER-1)

The framework must reuse — not reinvent — the **same static-`Registrar` self-registration idiom** already used by every seam:
- **Platform** (`include/vllm/platforms/interface.h:82`): `residency_policy()` (`:114`), `get_attn_backend_priority()` (`:124`), self-registered per `DeviceType`.
- **Attn-backend registry** (`include/vllm/v1/attention/registry.h`): `RegisterAttentionBackend` (`:47`), `AttentionBackendRegistrar` (`:78`), `SelectAttentionBackendName` (`:67`) — "adding a backend's attention = 1 self-registering TU + 1 priority slot."
- **Model self-registration** (`include/vllm/model_executor/models/model_registry.h:153`): `REGISTER_VLLM_MODEL`/`ModelRegistrar` — "adding a model = 1 new TU + 1 REGISTER line."
- **Op table** (`RegisterOp`, §2a).

The fusion framework's catalog is a **fifth registry of the same shape**: a recipe self-registers into a catalog; a backend realization self-registers a fused-kernel `OpId`. This is the additivity mechanism.

---

## 3. Framework DESIGN

Recommendation: extend the landed **TDR** with a generalized recipe + a *model-build-time* match, staying at the `vt::` op-table seam. Rationale grounded below.

### 3a. Graph capture — evaluate the options

We have **no torch.fx**; we have the `vt::` op table + an explicit C++ forward in `qwen3_5.cpp`. Three options:

1. **A real captured IR** (record the forward into a DAG, match subgraphs, replace) — this is re-implementing Inductor's pattern matcher (`vllm_inductor_pass.py`). The design panel's `opgraph-ir-lowering` candidate; rejected as *premature high-effort generality* (from-scratch recorder + symbolic shape-carrying values + DAG matcher) whose payoff is entirely deferred to backends that don't exist yet, and which adds a silent-wrong-fusion attack surface to a byte-exact codebase.
2. **Op-sequence interception at the `vt::` dispatch layer** (a runtime trace of emitted ops, matched lazily) — the `declarative-dsl-lazy` candidate; rejected: per-step interpreter dispatch on the *prefill* hot path (varlen tokens forfeit shape-bucket replay), and the perf-critical GEMM-epilogue class can't be materialized from a runtime tree.
3. **A declarative recipe matched at MODEL-BUILD time** — the model declares, per fusable site, which `FusedRecipe` applies (a compile/build-time binding, not a per-forward match). This is what the landed Phase-0 already is (a `constexpr` recipe passed to `FusedChain`), generalized so the *binding* of recipe→site is declared once per model rather than hand-called.

**Recommend Option 3.** It is the only one that (a) matches our architecture (explicit C++ forward, no fx), (b) has **zero per-forward overhead** — the match/binding happens once at model build, so the framework overhead is ~0 (a gate requirement, §8), and (c) is already the shape of the landed skeleton. "Automatic" here means *the model declares its fusable sites against the catalog once*, and realization/fallback is automatic per backend — not a runtime graph compiler. The finite vLLM pattern set (§1b) makes a captured-IR matcher unnecessary: we know every pattern ahead of time.

### 3b. Pattern declaration (declare once, backend-agnostic)

A fusion is a `constexpr FusedRecipe` in `include/vt/recipes.h` (the WHAT), transcribing a named vLLM pass's subgraph→fused-op rule with a cited upstream `file:line` (as `kFusedAddRmsNorm` already does, `recipes.h:13-34`). **W1 generalizes the POD** beyond the 4-role/3-opcode Phase-0 limit to express the real catalog:
- Opcodes grow: `+kSilu, +kSigmoid, +kMul(gate·up), +kRope, +kQuantFp8, +kQuantFp4` (each `+1` switch case per backend realization — O(1), §1d's `forward_native` analog).
- Operands grow from 4 fixed roles to a **small indexed operand table** (bind K named tensors), so multi-input chains (silu_and_mul, qk_norm_rope_gate, sigmoid_gate) are expressible.
- The **quant terminal is backend-negotiated, not portable** (per `backends.md`): `QUANT_BLK_FP4/FP8` are NVIDIA-specific; Metal/Vulkan/XPU use GGUF k-quants. So the norm/act/rope/gate *prefix* ports across backends via the composite/interpreter tiers; the quant *tail* is a per-backend opcode (`+1` case). Recorded honestly, not hidden.

### 3c. Dispatch / realization (realize per backend, through the op table)

Realize AT the existing seam: one `OpId::kFusedChain` dispatched through the exact `RegisterOp`/`GetOp` table every op uses (§2a). Two tiers (both landed at Phase-0, generalized in W1):
- **Tier 0 — composite over existing `vt::` primitives (correctness, every backend, FREE).** Walk the recipe step-by-step through already-registered ops (`RmsNorm`, `SiluAndMul`, `RopeNeox`, …). This is the **default** and the **CPU golden oracle**; a backend with no fused kernel inherits it and is correct on day 1. It is the C++ analog of vLLM's `forward_native` (`custom_op.py:138`).
- **Tier 1 — one interpreter kernel per backend (portable speed).** Single-pass over each row (`FusedChainInterpKernel`); one kernel port per backend lights up *every* Class-A recipe (not one kernel per fusion). This is where Vulkan/Metal get every fusion cheaply.
- **(Tier 2 — build-time structural specialization, reserved.)** Only where Tier-1 measurably loses to a hand-fused kernel on an ALU-heavy chain; reuses the existing `AttnQkNormRopeGateKernel` block-reduce skeleton (`cuda_ops.cu:390`) as archetype. Not built speculatively.

**Graceful degradation is the default, not an error path.** `GetOp` throws on an unregistered `(op,backend)` (`ops.cpp`), so the fallback must be that *every backend registers `kFusedChain`* and Tier-0 composite handles any recipe — a backend that lacks a fast realization still runs correctly. (This fixes the design panel's noted `GetOp`-hard-crash flaw.)

### 3d. The catalog-SELECT vs CODEGEN boundary

We **SELECT** pre-written portable fused kernels from the catalog (tractable, finite, §1b). We do **NOT** runtime-codegen novel kernels — that is Inductor's combo_kernels job (§1c), explicitly future work. Cutlass EVT (C++ compile-time GEMM-epilogue fusion) is a *separate* per-backend track (Class B), not this framework — grounded: our GEMMs use only the auto-builder `LinearCombination`/`alpha_ptr` epilogue today (no EVT visitor tree in-tree), so generic-recipe→EVT lowering is net-new, CUDA-only, and out of scope here.

---

## 4. THE ADDITIVITY MECHANISM (primary deliverable)

Judged by the PR-#4 test. Concretely, contrast today's hand-wiring with the framework:

**Adding a new vLLM fusion pass** (e.g. a future `silu_and_mul_per_block_quant` variant):
- *Today:* new `OpId` + `Fn` alias in `ops.h` + CUDA kernel in `cuda_ops.cu` + CPU oracle in `cpu_ops.cpp` + hand call at each `qwen3_5.cpp` site = **~5 shared files edited**.
- *Framework:* **one `constexpr FusedRecipe` in `recipes.h`** (transcribe the pattern) + a one-line binding at the model's fusable-site declaration. Realization is automatic (Tier-0 composite works immediately; Tier-1 interpreter already handles any recipe). **1 additive declaration, 0 kernel files, 0 dispatch edits.**

**Adding a new GPU/backend** (the PR-#4 scenario):
- *Today:* re-hand-write all ~7 fused kernels (§2b) for the new backend = kernels scattered + re-tested everywhere.
- *Framework:* the backend **registers `kFusedChain`** (Tier-0 composite inherited free → correct day 1; Tier-1 interpreter = **one** kernel port → all recipes fast) + per-backend quant-terminal opcode cases. **One additive backend file + one registration**, mirroring the Platform/attn-registry/model-registry seams (§2d). Zero shared-code edits, zero per-recipe work.

**Adding a new model:**
- *Framework:* the model file **declares its fusable-site→recipe bindings** (additive, in the model TU that already self-registers via `REGISTER_VLLM_MODEL`). Recipes it shares with existing models are reused from the catalog; novel ones are one declaration each. **Additive model file, zero shared edits.**

**PR-#4 diff shape:** the catalog (`recipes.h`) and the interpreter grow; `qwen3_5.cpp` *shrinks* (bespoke hand-calls → declarative bindings); a new backend/model/fusion is a new file + a registration line. One component delivers perf-parity-application (safe-flip the fusions) + mechanical-sync (§1b patterns = declarations) + additive models/GPUs.

---

## 5. Numerics discipline

Fused reductions must be **bit-identical to the unfused sequence** or gated OFF — the RMSNorm-saga near-tie razor (the `a875397` revert). Grounded precedent: `vt::RmsNormQuantFp8` is documented BIT-IDENTICAL to `RmsNorm(bf16)→QuantFp8Static` **via the bf16-intermediate form the unfused path already rounds through** (`ops.h:649-651`) — the fused kernel squares the *rounded* residual in f32 exactly as `fused_add_rms_norm` does (`recipes.h:29-34`). This is the model:
- The `FusedRecipe` is the **single source of truth**; Tier-0 composite == Tier-1 interpreter == the pre-existing unfused golden, asserted **bit-for-bit** per pattern (`test_ops_fused_chain.cpp` already does this for `kFusedAddRmsNorm`).
- The **CPU composite is the oracle** (Tier-0), and it is now *literally the same declaration* as the fast path — this **structurally eliminates the CPU/CUDA drift** the hand-kept parallel oracles carry today (§2b).
- A recipe whose fused reduction cannot be made byte-exact (the Add+RMSNorm+Quant f32-reduction-order hazard) ships **gated OFF** until proven, exactly like the current fast-kernel discipline (parity-enablers-ship-as-defaults).
- Framework enforcement: every recipe carries a **per-pattern byte-exact `fused==composite==unfused` test** (§7); the migration of a hand-op to a recipe (W2) is behavior-identical by construction (the recipe reproduces the same op sequence), re-gated token-exact on both models.

Note vLLM's own tests use *near*-equality (`assert_close`, fp16 2e-3 / bf16 1e-2; `test_fusion.py:276-281`) because Inductor may reorder. **We hold the stricter bar** (byte-exact) because our razor demands it — a deliberate discipline delta recorded here.

---

## 6. Portability

CPU-ref + CUDA now; Metal/Vulkan/ROCm/XPU later via the `vt::`/`backends.md` seam. The **declaration layer is backend-agnostic** (POD `constexpr` above `vt::`); **realization is per-backend** (the op-table registration). A new backend: Tier-0 composite (inherited, correct) → Tier-1 interpreter (one kernel port → all recipes fast) → per-backend quant terminal (native format: NVFP4 on CUDA, GGUF k-quant on Metal/Vulkan/XPU). This mirrors exactly how the Platform/attn/model seams already make new backends additive (§2d). The norm/act/rope/gate prefix is portable; the quant tail is backend-negotiated (accepted per `backends.md`).

---

## 7. Tests to port

Re-express vLLM's `tests/compile/` (at this pin, under `tests/compile/passes/`) in our doctest/parity tiers, named traceably, upstream file cited in header (test-porting.md protocol):

- **The oracle-comparison discipline** — vLLM `passes/test_fusion.py:249-287` (`_run_fusion_test`): builds a small `nn.Module`, compiles it two ways via `TestBackend(…fusion_pass…)` vs `TestBackend(…no fusion…)`, asserts `assert_close` (`:281`) **plus** structural `matched_count==N` + `check_before/after_ops` (`:283-285`, oracle in `backend.py:106-122`). **Our mirror:** for each recipe, assert `FusedChain(Tier-1) == FusedChain(Tier-0 composite) == the pre-fusion op sequence`, **byte-exact** (stricter than upstream's `assert_close`), CPU-primary + CUDA-on-dgx — generalizing the existing `test_ops_fused_chain.cpp`.
- Per-pass unit tests to mirror as per-recipe tests: `test_fusion.py` (RMS+quant, our `kRmsNormQuantFp8` recipe), `test_silu_mul_quant_fusion.py:282` (our `kSiluMulFp4Quant`), `test_qk_norm_rope_fusion.py` (our `kAttnQkNormRopeGate`), `test_fusion_attn.py` (attn+quant epilogue — realized per attn backend).
- Pass-manager plumbing: `test_pass_manager.py` (UUID/config-toggle stability, `:63-83`) → our catalog-registration + build-time-binding stability test.
- Functionalization: `test_functionalization.py:276` (`assert_close(func, no_func)` `:323`) — relevant only if/when we add a real graph layer (W1 IR passes); check in SKIPPED with tracked reason otherwise.
- e2e fusion counting: `fusions_e2e/test_tp1_quant.py` + `conftest.py:183-304` (asserts per-pass match counts in full-model runs) → our "the catalog bound N recipes at model build" assertion, folded into the 27B/35B token-exact suite.
- **The mechanical-sync regression** (the point of the framework): a test asserting that *adding a pattern = adding a declaration + its byte-exact test*, and that a new backend inheriting the catalog stays correct (the W4 mock-backend test).
- Multi-GPU passes (`test_sequence_parallelism.py`, `test_fusion_all_reduce.py`, `test_async_tp.py`) — checked in SKIPPED (HW-blocked: no multi-GPU), tracked.

---

## 8. Gates

- **Token-exact:** 27B **235/235** + 35B **315/315** greedy token-for-token, both models, on every W that touches a call site (the standing MVP gate; the migration Ws are byte-identical by construction).
- **Per-pattern byte-exact:** every recipe's `fused == Tier-0 composite == unfused-golden`, bit-for-bit, CPU + CUDA (§5).
- **Perf NEUTRAL-OR-BETTER:** must not regress any axis vs the current hand-fused production defaults (benchmark-protocol every-axis rule, both models). The **framework overhead itself must be ~0** — the recipe→site binding is resolved at model-build, not per-forward (§3a); a measurable per-step interpreter-dispatch cost is a fail.
- **Per-backend CPU parity:** CPU CTest green (the composite oracle is the reference).
- **DGX confirmation:** clean CUDA `-Werror` build, memcheck 0 errors, on `sm_121a` (per the sibling ORDER-1 items' gate pattern).

---

## 9. HW needs + Dependencies

**HW:** dgx.casa GB10 (`sm_121`) for the CUDA token-exact + memcheck gates and any perf re-measure (flock the GPU per `sharing-a-gpu-with-flock`); dev-box for the CPU byte-exact + composite-oracle tier (primary correctness gate). No multi-GPU → sequence/allreduce/async-TP passes are inventory-only. Metal/Vulkan realization proof needs the M4 dev-box (deferred to backend #2).

**Dependencies:**
- The `vt::` op table (`ops.cpp`/`ops.h`) — the dispatch seam (§2a). No change to the table mechanism.
- The landed TDR Phase-0 skeleton (`fused_recipe.h`, `recipes.h`, `kFusedChain`, `test_ops_fused_chain.cpp`) — W0 is *already done as a proof*; W1 generalizes it.
- The existing hand-fused kernels (§2b) — become the first migrated recipes (W2); each is the byte-exact golden its recipe must match.
- The ORDER-1 seams it composes with: `BACKEND-PLATFORM`, `BACKEND-ATTN-REGISTRY`, `MODEL-FACTORY-registry` (§2d) — the framework's catalog is the same Registrar idiom; no new coordination.
- No dependency on multi-GPU, GGUF, or Inductor-codegen work.

---

## 10. Row-sized work breakdown (incremental, no big-bang)

Each W is independently claimable + gated. **W0 is substantially LANDED** (the Phase-0 skeleton) — the spike's first act is to record it as an `ANCHOR-BACKFILL` and route ONE existing hand-fusion through it.

- **W0 — declaration + registry + dispatch skeleton (LANDED, backfill + adopt one).** `fused_recipe.h`/`recipes.h`/`kFusedChain`/Tier-0+Tier-1/`test_ops_fused_chain.cpp` exist and pass. **Remaining:** wire `kFusedAddRmsNorm` through `FusedChain` at ONE real call site (`qwen3_5.cpp:2536`, behavior-identical, token-exact both models) to prove the seam end-to-end in production, and backfill the matrix anchors. *Gate: 235/235 + 315/315, byte-exact composite==interp==golden.*
- **W1 — generalize the recipe POD (the "pattern/graph" layer).** Extend `FOp` (add silu/sigmoid/mul/rope/quant opcodes) and the operand model (4 fixed roles → small indexed operand table) so multi-input chains are expressible; add the build-time recipe→site binding declaration surface. *Gate: the generalized POD re-expresses `kFusedAddRmsNorm` byte-exact (regression), CPU green.*
- **W2 — migrate the existing hand-fusions to declarations.** One at a time, each behavior-identical + token-exact: `kSiluMulFp4Quant`, `kRmsNormQuantFp8`, `kRmsNormGatedQuantFp8`, `kAttnQkNormRopeGate`, `kSigmoidGateFp4Quant`, `kMoeCombineGate`, GDN glue. Retire each bespoke `OpId` only once its recipe reaches byte-exact parity. `qwen3_5.cpp` shrinks. *Gate: per-op byte-exact + 235/235 + 315/315 after each migration.*
- **W3 — the mechanical-sync path.** Demonstrate a *new* vLLM pass ports as ONE declaration + its ported byte-exact test (pick an unported §1b variant, e.g. a per-block-quant activation variant), with the upstream `file:line` citation. This is the deliverable proof of the primary value. *Gate: the ported test passes; the sync touches only `recipes.h` + a test.*
- **W4 — backend-realization additivity proof.** A mock second backend (CPU-ref-shaped stand-in) registers `kFusedChain` and inherits the whole catalog at Tier-0 correctness with zero per-recipe work — the PR-#4 additivity test made executable. *Gate: mock backend runs every recipe correct via composite, one registration, zero shared-code edits.*
- **Wn — honest perf re-measure.** With the catalog adopted, re-measure 35B prefill vs the vLLM oracle on the binding grid (flock, `--cuda-graph-trace=node`, warmup-excluded). Record the *actual* per-step delta against the ~3.5%/step ceiling. *Gate: neutral-or-better on every axis; the number recorded in BENCHMARKS.md with the honest ceiling framing (do not oversell).*

---

## 11. HONEST payoff model

- **Perf: ceiling ~3.5%/step @ c1, compute-bound on 35B** (35B c1 190 vs 183 ms; 27B c1 already ours-faster). Prefill is 99.4% GPU-busy; ~71% real compute at parity; the ~20% glue is per-step-modest wall. The WHOLE-GRAPH horizontal (combo_kernels) residual is out of scope (§1c). Realistic 35B expectation: flip c1, lift means, partial tail relief — **not** a 20% close, **not** the failing-tail cure alone (task #61/#62; mixed-step piecewise cudagraph already REJECTED, async runner covers it). **Do not sell this as a perf lever.**
- **Primary value = extensibility + mechanical upstream-sync, scaling with model breadth:** (a) a new vLLM fusion PR ports as ONE declaration (§4, W3); (b) a new GPU inherits the catalog additively (§4, W4) — the direct PR-#4 remedy; (c) a new model declares its patterns additively; (d) it structurally **eliminates the CPU/CUDA oracle drift** (one declaration, both tiers) and **retires the growing bespoke-kernel surface** into a declarative catalog + one interpreter per backend. This is the roadmap_v1 extensibility cornerstone alongside `BACKEND-PLATFORM`/`BACKEND-ATTN-REGISTRY`/`MODEL-FACTORY-registry`.

---

## 12. Proposed stable-ID + matrix placement + roadmap row

**Stable-ID: `KERNEL-FUSION-FRAMEWORK`, primary home = `kernel-matrix.md`.** Argument: the deliverable is *realized fused kernels selected through the `vt::` op-dispatch table* — the exact subject of the kernel-matrix ("kernel-family and dispatch parity inventory across vLLM and its runtime dependency chain"). The declaration layer sits above `vt::` and touches the engine forward, but the row's evidence (per-backend kernels, dispatch, byte-exact gates) is kernel territory. Add a **cross-reference pointer** from `engine-matrix.md` (the forward-pass build-time binding surface) noting the row lives in kernel-matrix — do not duplicate the row. (Rejected `ENG-FUSION-FRAMEWORK` as primary: the engine matrix is scheduling/KV/serving; fusion realization is kernel dispatch.)

See `kernel-matrix.md` row `KERNEL-FUSION-FRAMEWORK` (added same-change) and the `roadmap_v1.md` ORDER-1 `ROAD-V1-C1` narrative for the portfolio row.
