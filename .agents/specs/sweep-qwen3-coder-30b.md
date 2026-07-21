# SPIKE: Breadth-Sweep Model Bring-Up — Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`)

*Breadth-sweep Tier-1 (`.agents/specs/breadth-sweep-plan.md` §B.3). Row `MODEL-TEXT-qwen3-moe-qwen3-moe-for-causal-lm` (model-matrix, `INVENTORIED` → `SPIKE`). READ-ONLY design — no code. Template: `.agents/specs/first-additive-model-qwen3-dense.md`.*
*Grounding pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; our tree. Standards: PR-#4 additivity ([[extensibility-first-additive-hw-models]]), near-tie-robust gate ([[near-tie-distributional-gate]]), mirror-vLLM.*

---

## 0. Purpose

Add `Qwen3MoeForCausalLM` — a **non-hybrid, full-attention, BF16 MoE** — as mostly-new files by COMBINING two already-done paths: the Qwen3-dense standard attention (`qwen3.cpp`, DONE) and the 35B MoE-expert machinery (`qwen3_5.cpp` `MoeBlock`, DONE). It is the flagship recent-MoE additive proof and the honest test of the key breadth question: *does "done dense attention" + "done MoE experts" compose additively for a full-attention MoE, or does the 35B MoE path assume the GDN-hybrid shape?*

**Headline: it composes — runner + attention are already shape-agnostic (ZERO runner change) — but there are FOUR model-layer seams**, one a real kernel gap: (#1/#2) the reusable pieces are file-static in two TUs and must be extracted/exposed; (#3) the 35B MoE `MoeBlock` assumes a shared expert (Coder has none); (#4) there is **no BF16 safetensors expert loader** (all our expert loading is NVFP4). Plus the SPEED gap: **no fast BF16 grouped-MoE GEMM** (only NVFP4-Marlin), so BF16 runs a slow per-expert reference loop.

---

## 1. TARGET + checkpoint reality (verified on dgx 2026-07-21)

Checkpoint present: `models--Qwen--Qwen3-Coder-30B-A3B-Instruct` (snapshot `b2cff646…`). `config.json`:

| field | value | consequence |
|---|---|---|
| `architectures` | **`Qwen3MoeForCausalLM`** | maps to `qwen3_moe` (registry.py:192) |
| `hidden_size` | 2048 | note head_dim*num_heads = 4096 ≠ hidden (explicit head_dim) |
| `num_hidden_layers` | 48 | |
| `num_attention_heads` / `num_key_value_heads` | 32 / 4 | GQA 32/4, head_dim 128 |
| `head_dim` | **128** | reuses the d128 FA2 kernels (proven on Qwen3-4B) |
| `num_experts` / `num_experts_per_tok` | **128 / 8** | top-8 of 128 |
| `moe_intermediate_size` | 768 | per-expert MLP width |
| `decoder_sparse_step` / `mlp_only_layers` | 1 / `[]` | every layer is MoE (uniform, simplest case) |
| shared expert | **absent** | NO shared expert — differs from the 35B path |
| `tie_word_embeddings` | **false** | separate `lm_head` (differs from Qwen3-0.6B) |
| `norm_topk_prob` | true | router renormalize |
| `rope_theta` | 10000000 | |
| `torch_dtype` / `quantization_config` | **bfloat16 / ABSENT** | UNQUANTIZED BF16 — new MoE compute regime |

Expert weights (safetensors index): `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight` (per-expert separate), router `model.layers.{L}.mlp.gate.weight`, standard attn/norm/embed/lm_head — plain `model.layers.` prefix (like Qwen3 dense), NOT the 35B multimodal wrapper.

**Oracle:** vLLM 0.25.0 contains `qwen3_moe.py`; runnable token-exact oracle today. A3B = 3B active (fast decode), 30B total (~60 GB bf16 — fits GB10 128 GB unified).

---

## 2. Upstream grounding — `vllm/model_executor/models/qwen3_moe.py @ e24d1b24`

- **`Qwen3MoeAttention` (:254-354):** qkv_proj split `[q,kv,kv]`; per-head `q_norm`/`k_norm` before RoPE (:333-334,:344-351); `get_rope(head_dim)` (:311); full causal Attention (:317). **Byte-for-byte the same shape as `qwen3.py::Qwen3Attention`** (the dense path we did). No attention gate. NO GDN.
- **`Qwen3MoeSparseMoeBlock` (:130-251):** router `gate = ReplicatedLinear(hidden, num_experts, bias=False)` (:172-178); `FusedMoE(top_k, renormalize=norm_topk_prob)` (:204-217); shared expert ONLY when `shared_expert_intermediate_size>0` (:180-202) — Coder has none → `shared_expert=None`. `scoring_func="softmax"`.
- **`Qwen3MoeDecoderLayer` (:357-429):** standard RMSNorm input/post (:404-407), fused residual-add (:418-427); all-MoE for Coder.
- **`Qwen3MoeForCausalLM` (:541-657):** `packed_modules_mapping = {qkv_proj:[q,k,v]}`; `lm_head = ParallelLMHead`, tied only if `tie_word_embeddings` (:578-585) — Coder untied.

**1:1 map:** Qwen3-Coder = `qwen3.py` dense attention/norm/embeddings (DONE in `qwen3.cpp`) with the dense MLP replaced by the `qwen3_5.cpp` MoE block, minus GDN, minus multimodal prefix, minus NVFP4, minus shared expert, plus untied lm_head + BF16 experts.

---

## 3. ADDITIVITY LEDGER (primary deliverable)

### (a) NEW FILES a clean bring-up ADDS
| New file | Purpose | Reuses |
|---|---|---|
| `include/vllm/model_executor/models/qwen3_moe.h` | `Qwen3MoeWeights` + Forward decls | mirrors `qwen3.h` + `MoeBlockWeights` (qwen3_5_weights.h:266) |
| `src/vllm/model_executor/models/qwen3_moe_weights.cpp` | BF16 loader: `model.layers.` map + NEW bf16 per-expert loader + router gate + untied lm_head | reuses `dense_weight_loaders.h` + qwen3_weights.cpp patterns |
| `src/vllm/model_executor/models/qwen3_moe.cpp` | forward: dense `AttnBlock` (reused) + `MoeBlock` (reused) per layer | §4 |
| `src/vllm/model_executor/models/qwen3_moe_registry.cpp` | `REGISTER_VLLM_MODEL("Qwen3MoeForCausalLM")` + factory + `MakeQwen3MoeKVCache` (full-attn-only) + `ParseQwen3MoeConfig` | mirrors `qwen3_dense.cpp` |
| `tests/vllm/models/test_qwen3_moe_forward.cpp` + `tests/parity/test_qwen3_moe_paged_engine.cpp` | forward doctest + near-tie token-exact vs oracle | mirrors qwen3/qwen27 tests |

### (b) SHARED FILES it must TOUCH — classified
| Shared touch | Classification | Notes |
|---|---|---|
| in-TU `REGISTER_VLLM_MODEL` | **DESIGNED SEAM** | zero shared-array edit (item 5) |
| `CMakeLists.txt` +~4 TUs | **UNAVOIDABLE build-glue** | no C++ module auto-discovery |
| qwen3.cpp dense `AttnBlock` file-static (anon ns :51; AttnBlock :362-533, RunLayer :555) | **★ SEAM GAP #1 (extract)** | extract the attention preamble to a shared header (like `dense_weight_loaders.h`); then any full-attn model reuses it. One-time. |
| qwen3_5.cpp `MoeBlock` file-static (anon ns :346; MoeBlock :4328) | **★ SEAM GAP #2 (expose)** | expose `MoeBlock` + bf16 `ExpertMlp` via a header; router/silu/combine already public vt:: ops |
| `MoeBlock` calls `SharedExpert` unconditionally (:4401); reads `shared_expert_intermediate_size` (:3796) | **★ SEAM GAP #3 (no-shared guard)** | Coder has none → guard shared on `>0` (pass nullptr to MoeCombine). Small, model-agnostic. |
| NO bf16 safetensors expert loader — `LoadMoeExpertsInto` (qwen3_5_weights.cpp:331) loads only NVFP4 | **★ SEAM GAP #4 (loader)** | NEW: load `mlp.experts.{e}.{gate,up,down}_proj.weight` as bf16; reuses `LoadBf16*` |
| `runner.cpp` KV-alloc + GDN step path | **NO CHANGE (already generalized)** | `has_mamba_group = gdn_group_id_>=0` (:464); GDN guarded (:667). Full-attn MoE → gdn_group_id_<0 → behaves like Qwen3 dense. `ENG-RUNNER-MODELSHAPE` already covers it. |
| `hf_config.cpp` | **likely NONE** | HfConfig carries num_experts/top_k/moe_intermediate/shared_expert (hf_config.h:86-89) |

### Honest count & verdict vs PR-#4
- **New files: ~6** (4 code + 2 tests).
- **Shared touches: 2 designed + 4 seam gaps** — #1/#2 are EXTRACTION/EXPOSE (the two done pieces live file-static in different TUs; the first model to combine them is the forcing function, like the dense bring-up forced the loader-helper extraction); #3 is a 1-predicate guard; #4 is a real new bf16-expert loader. None is per-model scatter — after them, the next full-attn MoE reuses them new-files-only.
- **KEY question answered:** combining "done dense attention" + "done 35B MoE experts" for a non-hybrid full-attention MoE **DOES compose** — the runner is already shape-agnostic and `MoeBlock` takes no `gdn_meta`. The 35B's hybrid assumptions live in `Qwen3_5Model::Forward` (which Coder does NOT call — it composes `AttnBlock`+`MoeBlock` in a new GDN-free layer loop). So **NO runner seam gap** (unlike the dense bring-up). All seams are model-layer.
- **PR-#4 test: PASSES decisively at the engine layer** (zero runner/sampler/KV edit); passes at the model layer modulo the one-time #1/#2 refactors + the #4 bf16-expert loader.

---

## 4. Forward wiring (compose done pieces)

`Qwen3MoeModel::Forward` = the qwen3.cpp `ForwardBody` with the per-layer MLP replaced by the MoE block:

| Step | Reused from | New? |
|---|---|---|
| embed → residual init | qwen3.cpp ForwardBody | reuse |
| input RMSNorm (std add) | `kFusedAddRmsNormStd` / vt::RmsNorm | reuse |
| attention (qkv, per-head q/k norm, RoPE, d128 FA2, o_proj) | qwen3.cpp `AttnBlock` (:362-533) | **reuse via extraction (#1)** — GQA 32/4 head_dim 128 proven on Qwen3-4B |
| post RMSNorm (std add) | qwen3.cpp | reuse |
| MoE block (router softmax+top8+renorm, expert gather, bf16 ExpertMlp, combine) | qwen3_5.cpp `MoeBlock` (:4328) bf16 branch (:4350-4408) | **reuse via expose (#2) + no-shared guard (#3)** |
| final RMSNorm → lm_head (untied) | qwen3.cpp ForwardBody; untied vt::Matmul (:657) | reuse |

Router numerics match: `vt::MoeRouterTopK(renormalize=true)` = softmax+top-k+renormalize (cuda_moe.cu, grounded in vLLM `topk_softmax_kernels.cu`); Qwen3Moe uses `scoring_func="softmax"`, `renormalize=norm_topk_prob=true`. Numeric contract mirrors qwen3.cpp bf16 discipline. Confirm at W3: non-gemma RMSNorm (have `kFusedAddRmsNormStd`), the bf16 `MoeBlock` reference vs vLLM fused_moe.

---

## 5. QUANT path

**No quant** (`quantization_config` absent, `torch_dtype: bfloat16`) — the headline shape difference vs the 35B (`W4A16_NVFP4` Marlin experts + fp8 dense).
- **Correctness: COVERED** by the existing bf16 `MoeBlock` reference path (`fp4=false`, qwen3_5.cpp:4350) — per-expert host-gather + bf16 `ExpertMlp` + `MoeCombine` (modulo #3).
- **Speed: GAP.** Our only fast MoE is NVFP4-Marlin (`MoeBlockFusedMarlinCuda` :4175) / NVFP4 grouped. **No BF16 grouped-MoE GEMM.** vLLM runs Coder's bf16 experts through its Triton fused_moe grouped GEMM. So a **fast BF16 grouped-MoE path** (port vLLM fused_moe, or a cuBLAS/cutlass grouped-GEMM) is required for the speed bar. Also the 35B decode CUDA-graph is fp4-gated (qwen3_5_moe.cpp:76-82) → a bf16 Coder gets no decode graph → eager-decode host tax risk.

So quant is not a coverage gap (bf16 is simpler than NVFP4) — it is a **performance-kernel gap** (the bf16 MoE fast path does not exist). This is the single largest work item and the crux of the SPEED deliverable.

---

## 6. Tests to port + gates

- Port: registry-resolution case (`Qwen3MoeForCausalLM` → our MoE factory, full-attn-only KV); forward doctest; near-tie token-exact paged-engine parity.
- **SACRED gate:** near-tie-robust token-exact vs vLLM 0.25.0 (our token within 0.5 nats of vLLM's teacher-forced argmax, strict where equal). A3B decode is fast → cheap capture. Regression: 27B 235/235 + 35B 315/315 unchanged (Coder touches none of `Qwen3_5Model::Forward`; holds by construction once #2/#3 preserve the fp4 branch).
- **SPEED gate:** match/beat graphed vLLM 0.25.0 every axis at the large-conc operating point — where the bf16-fast-MoE gap bites; correctness may land before speed as separate W-rows.
- `-Werror` 0-warn + memcheck 0.

---

## 7. W-plan

- **W0 — config + registry stub.** `qwen3_moe_registry.cpp`: REGISTER + stub factory + `MakeQwen3MoeKVCache` (clone the dense full-attn-only spec) + `ParseQwen3MoeConfig`; `is_dense_model=false`. Gate: CPU build + registry-resolution test. *No runner change.*
- **W1 — extraction/expose seams (#1/#2/#3).** Extract qwen3.cpp `AttnBlock`+glue to a shared header; expose qwen3_5.cpp `MoeBlock`+bf16 `ExpertMlp`; add the no-shared-expert guard. Gate: BEHAVIOR-PRESERVING — Qwen3-dense 0.6B/4B near-tie + 27B 235/235 + 35B 315/315 UNCHANGED.
- **W2 — BF16 loader (#4).** `qwen3_moe_weights.cpp`: attn/embed/norm/lm_head (untied) + NEW bf16 per-expert loader + router. Gate: load checkpoint, shape/count asserts, no unmapped, memcheck 0.
- **W3 — forward.** `qwen3_moe.cpp` composing reused `AttnBlock` + `MoeBlock`. Gate: forward doctest vs golden.
- **W4 — correctness gate (SACRED).** Near-tie token-exact vs vLLM 0.25.0, regression preserved. Row → correctness-complete.
- **W5 — SPEED.** BF16 fast grouped-MoE GEMM (the §5 kernel gap) + eager-decode host-tax review. Match/beat vLLM every axis. The largest item; gates DONE.

**HW/deps:** checkpoint present (bf16, ~60 GB, fits GB10); oracle 0.25.0; no download/multi-GPU. Depends on the delivered seams (model self-reg, Platform/residency, attn-registry, fusion catalog, `ENG-RUNNER-MODELSHAPE`). New work = model-layer seams #1-#4 + the W5 bf16-MoE kernel. Per [[parity-enablers-ship-as-defaults]], the W5 bf16-fast-MoE ships default-ON (gated) before any binding speed run.

---

## 8. Key findings

1. Checkpoint: `Qwen3MoeForCausalLM`, 48L, GQA 32/4 head_dim 128, 128 experts top-8, moe_intermediate 768, **BF16 unquantized**, **NO shared expert**, **untied lm_head**, **NO GDN**. All layers MoE.
2. **It composes** — runner already fully shape-agnostic (`gdn_group_id_>=0` gates all GDN) → **zero runner change** (unlike the dense bring-up). The 35B hybrid assumptions live in `Qwen3_5Model::Forward`, which Coder does not call.
3. **Four model-layer seams**, none per-model scatter: #1 extract dense `AttnBlock`; #2 expose bf16 `MoeBlock`; #3 no-shared-expert guard; #4 NEW bf16 expert loader. #1/#2 are the "first model to combine two done pieces" forcing function.
4. **Quant = the headline SPEED gap, not coverage.** BF16 correctness covered by the reference path; no fast BF16 grouped-MoE GEMM (only NVFP4-Marlin) + no bf16 decode graph. Porting a bf16 grouped-MoE (vLLM fused_moe) is the crux of the speed bar.
5. Router/attention numerics already match; d128 FA2 + non-square o_proj proven on Qwen3-4B.
6. Row `MODEL-TEXT-qwen3-moe-qwen3-moe-for-causal-lm` (`INVENTORIED` → `SPIKE`). Gate: near-tie-robust token-exact + every-axis vLLM speed; regression 27B 235/235 + 35B 315/315.

---

## 9. Structured spike contract (stable row `MODEL-TEXT-qwen3-moe-qwen3-moe-for-causal-lm`)

The prose above (§0–§8) is the full spike; this restates it in the record-checker's structured fields.

### Scope
Add `Qwen3MoeForCausalLM` (non-hybrid, full-attention, BF16 MoE: standard per-head q/k-norm+RoPE attention, 128 experts top-8, NO shared expert, untied lm_head, NO GDN) as MOSTLY NEW FILES by composing the done Qwen3-dense attention + the done 35B MoE `MoeBlock`. In scope: config hook, registry TU, full-attention-only KV spec, the extract/expose/guard refactors, bf16 expert loader, forward, near-tie token-exact gate, and the bf16-fast-MoE speed kernel. Out of scope: quant (Coder is bf16), MLA, sliding-window.

### Upstream chain
`vllm/model_executor/models/qwen3_moe.py::Qwen3MoeForCausalLM` @ `e24d1b24` (§2: `Qwen3MoeAttention` = same shape as `qwen3.py::Qwen3Attention`; `Qwen3MoeSparseMoeBlock` router+FusedMoE, shared expert only when `shared_expert_intermediate_size>0`; standard RMSNorm; untied lm_head); `registry.py:192`; config `Qwen3-Coder-30B-A3B-Instruct/config.json`.

### Our baseline
Templates: `qwen3.cpp`/`qwen3_dense.cpp` (the DONE Qwen3-dense standard-attention path), `qwen3_5.cpp` `MoeBlock` (the DONE 35B MoE experts), the model-shape-agnostic runner (`ENG-RUNNER-MODELSHAPE`), the fusion catalog, `dense_weight_loaders.h`. Coder = Qwen3-dense attention + 35B MoE experts, minus GDN/shared-expert/NVFP4/multimodal-prefix, plus untied lm_head + bf16 experts.

### Port map
New files (§3a): `qwen3_moe.h`, `qwen3_moe_registry.cpp`, `qwen3_moe_weights.cpp`, `qwen3_moe.cpp`, tests. Shared touches (§3b): in-TU REGISTER (designed seam), CMake TU glue, and the 4 model-layer seams — #1 extract file-static dense `AttnBlock` → `dense_attn_block.h`, #2 expose file-static bf16 `MoeBlock` → `qwen3_5_moe_block.h`, #3 no-shared-expert guard, #4 NEW bf16 safetensors expert loader. ZERO runner change (the shape-agnostic runner covers full-attn MoE).

### Tests to port
(a) registry resolution (`Qwen3MoeForCausalLM` → the MoE factory, full-attn-only KV spec); (b) forward doctest `test_qwen3_moe_forward.cpp`; (c) near-tie token-exact `test_qwen3_moe_paged_engine.cpp` vs the vLLM 0.25.0 oracle. Mirrors the qwen3/qwen27 test family.

### Gates
W0: CPU build + registry test. W1: dgx CUDA `-Werror` 0-warn; BEHAVIOUR-PRESERVING — Qwen3-dense 0.6B/4B near-tie 16/16 + 27B 235/235 + 35B 315/315 UNCHANGED (the extract/expose/guard are pure moves + an inert guard). W4 SACRED: near-tie-robust token-exact vs vLLM 0.25.0. W5: every-axis vLLM speed parity. memcheck 0 throughout.

### Dependencies
Builds on the delivered seams (model self-reg item 5, Platform/residency, attn-registry, fusion catalog, `ENG-RUNNER-MODELSHAPE`). New work = the 4 model-layer seams + the W5 bf16-fast-MoE kernel. Checkpoint present (bf16, ~60 GB, fits GB10 128 GB unified); oracle vLLM 0.25.0 present; no download/multi-GPU.

### Work breakdown
W0 config + registry stub (LANDED 2026-07-21: `qwen3_moe_registry.cpp` REGISTER + full-attn-only KV + stub factory + `ParseQwen3MoeConfig`; registry resolves 189/189; no runner change). W1 extract/expose/guard (LANDED 2026-07-21: dense `AttnBlock`→`dense_attn_block.h`, bf16 `MoeBlock`→`qwen3_5_moe_block.h` `RunMoeBlock`, no-shared-expert guard; BEHAVIOR-PRESERVING — Qwen3-dense 0.6B/4B 16/16 + 27B 235/235 + 35B 315/315 UNCHANGED, `-Werror` 0-warn, memcheck 0). W2 bf16 per-expert loader (LANDED 2026-07-21: `qwen3_moe_weights.cpp` `LoadQwen3MoeForCausalLMWeights` — merged qkv/o + per-head q/k norm via `dense_weight_loaders.h`, NEW bf16 per-expert loader `LoadQwen3MoeBlock` (router gate + 128 experts × gate/up/down transposed to Matmul-B, no shared expert), UNTIED lm_head loaded separately; load gate `test_qwen3_moe_load` all 18867 tensors mapped/shaped, NO leftover, 131746 assertions pass on dgx). W3 forward (LANDED 2026-07-21: `qwen3_moe.cpp` `Qwen3MoeModel::Forward/ForwardDevice` composing the shared dense `AttnBlock` + the exposed `RunMoeBlock` per layer, NO GDN/shared, untied lm_head; factory wired. Doctest `test_qwen3_moe_forward` 3/3 — CPU synthetic finite+deterministic, fusion-catalog ADOPT byte-identical to fallback, and real-checkpoint prefill argmax=**12095 (" Paris")** for "The capital of France is" (correct, matches the dense Qwen3 result — not a near-tie). Regression 27B 235/235 + 35B 315/315 UNCHANGED, `-Werror` 0-warn). **W4 correctness gate (SACRED) LANDED 2026-07-21:** paged-engine greedy vs vLLM 0.25.0 oracle. GATE SELECTION — vLLM's OWN greedy is DETERMINISTIC on Qwen3-Coder (K=5 per-prompt runs, `scripts/qwen3coder-oracle-capture.py`: 0 multi-valued (prompt,pos) cells), so the STRICT-where-well-posed near-tie-robust gate applies (identical ratified methodology to the dense `test_qwen3_paged_engine.cpp`). RESULT: **6/6 prompts PASS** (138 assertions) — STRICT token-exact 4/6; near-tie-band only 2/6 (prompt[3] "largest planet", prompt[4] "chemical symbol"); **max teacher-forced gap 0.125 nats** @ prompt[3] tok6 (≪ the 0.5-nat bar); **0 forward-divergent**. Diagnosis of the 2 near-tie prompts: teacher-forcing vLLM on OUR exact prefix (`scripts/qwen3coder-neartie-gap.py`) shows the divergent tokens ARE vLLM's OWN argmax (e.g. prompt[4] tok15 our=11483 == vLLM_argmax, gap 0.0000) — the free-running greedy chose a different token only via prefill-vs-decode / independent-bf16-decoder near-ties, NOT a forward bug. ORACLE NOTE: the auto-selected FlashInfer-CUTLASS unquantized-MoE backend OOM-kills the 57 GiB model on GB10's 119 GiB unified pool (57 GiB weights + 57 GiB checkpoint page-cache + workspace), so the oracle runs vLLM's first-class **TRITON** MoE backend (same bf16 grouped-MoE math) with gpu-mem-util 0.58 + a checkpoint page-cache evictor; deterministic K=5. New files: `tests/vllm/models/test_qwen3coder_paged_engine.cpp` + `scripts/qwen3coder-oracle-capture.py` + `scripts/qwen3coder-neartie-gap.py` + goldens `tests/parity/goldens/qwen3coder_greedy/{greedy_ids,greedy_dist,our_ids,neartie_gap_mnats}.npy` + `p{0..5}_prompt.i32`. GATES: dgx CUDA `-Werror` 0-warn; 6/6 gate; 27B 235/235 + 35B 315/315 UNCHANGED; memcheck 0 on the engine path. Row → correctness-DONE (only W5 speed remains). W5 bf16-fast-MoE grouped GEMM + every-axis speed (the largest item; gates DONE).

### Risks/decisions

**Decision — BF16 unquantized.** Coder's `quantization_config` is absent (`torch_dtype: bfloat16`), so no quant loader/kernel coverage is needed; correctness is covered by the existing bf16 `MoeBlock` reference path. Simpler than the 35B's `W4A16_NVFP4` Marlin experts.

**Decision — compose, don't rebuild.** The runner is already model-shape-agnostic (`ENG-RUNNER-MODELSHAPE`) and `MoeBlock` takes no GDN metadata, so Coder reuses the done dense attention + the done 35B MoE experts with ZERO runner change. The 35B hybrid assumptions live in `Qwen3_5Model::Forward`, which Coder does not call.

**Decision — near-tie-robust gate** ([[near-tie-distributional-gate]]): correctness = our token within 0.5 nats of vLLM's teacher-forced argmax, strict where equal (the Qwen3-established bar). A3B decode is fast → cheap capture.

**Risk (headline) — the SPEED gap is a real new kernel.** There is no fast BF16 grouped-MoE GEMM (only NVFP4-Marlin), and the 35B decode CUDA-graph is fp4-gated → a bf16 Coder gets no decode graph. W5 (port a bf16 grouped-MoE GEMM, e.g. vLLM fused_moe, + review the eager-decode host tax) is the largest item and the crux of the every-axis speed bar. Correctness can land (W4) before speed (W5) as separate rows.

**Risk — the W1 refactors must be byte-identical.** #1 extract (dense `AttnBlock` → shared header) must keep Qwen3-dense 0.6B/4B byte-identical; #2 expose (bf16 `MoeBlock`) is a pure add (35B never calls it); #3 the no-shared-expert guard must be inert for the 35B/27B (shared>0, fp4 path). Verified via unchanged token-exact gates on all existing models.

**Risk — bf16 MoE reference-path perf.** The bf16 correctness path is a per-expert host-gather loop; it is correct but slow — acceptable for W4 (correctness) but is exactly what W5 must replace.
