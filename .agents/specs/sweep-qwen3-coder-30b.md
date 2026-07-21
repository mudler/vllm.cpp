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
- **W5 — SPEED. LANDED 2026-07-21, gate MISSED (row stays `ACTIVE`).** `vt::MoeGroupedGemmBf16` + `MoeBlockBf16Cuda` (default ON) + the host-mirror release closed the §5 kernel gap and won 3.4× TTFT / 1.59× TPOT same-binary, but ours is 0.41–0.80× of GRAPHED vLLM on TTFT/TPOT and 0.49–0.75× on throughput. Attributed by `nsys` to the prefill WMMA tile (56% of GPU time, ~1.7% of bf16 peak) and the block-starved naive decode kernel + missing w13 fusion + missing bf16 decode graph. Full table, per-kernel attribution and the named remaining levers: §9 "Work breakdown".

**HW/deps:** checkpoint present (bf16, ~60 GB, fits GB10); oracle 0.25.0; no download/multi-GPU. Depends on the delivered seams (model self-reg, Platform/residency, attn-registry, fusion catalog, `ENG-RUNNER-MODELSHAPE`). New work = model-layer seams #1-#4 + the W5 bf16-MoE kernel. Per [[parity-enablers-ship-as-defaults]], the W5 bf16-fast-MoE ships default-ON (gated) before any binding speed run.

---

## 8. Key findings

1. Checkpoint: `Qwen3MoeForCausalLM`, 48L, GQA 32/4 head_dim 128, 128 experts top-8, moe_intermediate 768, **BF16 unquantized**, **NO shared expert**, **untied lm_head**, **NO GDN**. All layers MoE.
2. **It composes** — runner already fully shape-agnostic (`gdn_group_id_>=0` gates all GDN) → **zero runner change** (unlike the dense bring-up). The 35B hybrid assumptions live in `Qwen3_5Model::Forward`, which Coder does not call.
3. **Four model-layer seams**, none per-model scatter: #1 extract dense `AttnBlock`; #2 expose bf16 `MoeBlock`; #3 no-shared-expert guard; #4 NEW bf16 expert loader. #1/#2 are the "first model to combine two done pieces" forcing function.
4. **Quant = the headline SPEED gap, not coverage.** BF16 correctness covered by the reference path; the fast BF16 grouped-MoE GEMM did not exist (only NVFP4-Marlin). **W5 built it** (`vt::MoeGroupedGemmBf16`, default ON) — worth 3.4× TTFT / 1.59× TPOT same-binary — but it does NOT yet reach GRAPHED vLLM (0.41–0.80× latency, 0.49–0.75× throughput). The residual is a KERNEL-QUALITY gap, not a structural one: the WMMA tile is untuned (~1.7% of bf16 peak) and three vLLM mirrors are still missing (w13 gate+up fusion, deterministic split-K for the block-starved decode kernel, bf16 decode CUDA graph). See §9.
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
W0 config + registry stub (LANDED 2026-07-21: `qwen3_moe_registry.cpp` REGISTER + full-attn-only KV + stub factory + `ParseQwen3MoeConfig`; registry resolves 189/189; no runner change). W1 extract/expose/guard (LANDED 2026-07-21: dense `AttnBlock`→`dense_attn_block.h`, bf16 `MoeBlock`→`qwen3_5_moe_block.h` `RunMoeBlock`, no-shared-expert guard; BEHAVIOR-PRESERVING — Qwen3-dense 0.6B/4B 16/16 + 27B 235/235 + 35B 315/315 UNCHANGED, `-Werror` 0-warn, memcheck 0). W2 bf16 per-expert loader (LANDED 2026-07-21: `qwen3_moe_weights.cpp` `LoadQwen3MoeForCausalLMWeights` — merged qkv/o + per-head q/k norm via `dense_weight_loaders.h`, NEW bf16 per-expert loader `LoadQwen3MoeBlock` (router gate + 128 experts × gate/up/down transposed to Matmul-B, no shared expert), UNTIED lm_head loaded separately; load gate `test_qwen3_moe_load` all 18867 tensors mapped/shaped, NO leftover, 131746 assertions pass on dgx). W3 forward (LANDED 2026-07-21: `qwen3_moe.cpp` `Qwen3MoeModel::Forward/ForwardDevice` composing the shared dense `AttnBlock` + the exposed `RunMoeBlock` per layer, NO GDN/shared, untied lm_head; factory wired. Doctest `test_qwen3_moe_forward` 3/3 — CPU synthetic finite+deterministic, fusion-catalog ADOPT byte-identical to fallback, and real-checkpoint prefill argmax=**12095 (" Paris")** for "The capital of France is" (correct, matches the dense Qwen3 result — not a near-tie). Regression 27B 235/235 + 35B 315/315 UNCHANGED, `-Werror` 0-warn). **W4 correctness gate (SACRED) LANDED 2026-07-21:** paged-engine greedy vs vLLM 0.25.0 oracle. GATE SELECTION — vLLM's OWN greedy is DETERMINISTIC on Qwen3-Coder (K=5 per-prompt runs, `scripts/qwen3coder-oracle-capture.py`: 0 multi-valued (prompt,pos) cells), so the STRICT-where-well-posed near-tie-robust gate applies (identical ratified methodology to the dense `test_qwen3_paged_engine.cpp`). RESULT: **6/6 prompts PASS** (138 assertions) — STRICT token-exact 4/6; near-tie-band only 2/6 (prompt[3] "largest planet", prompt[4] "chemical symbol"); **max teacher-forced gap 0.125 nats** @ prompt[3] tok6 (≪ the 0.5-nat bar); **0 forward-divergent**. Diagnosis of the 2 near-tie prompts: teacher-forcing vLLM on OUR exact prefix (`scripts/qwen3coder-neartie-gap.py`) shows the divergent tokens ARE vLLM's OWN argmax (e.g. prompt[4] tok15 our=11483 == vLLM_argmax, gap 0.0000) — the free-running greedy chose a different token only via prefill-vs-decode / independent-bf16-decoder near-ties, NOT a forward bug. ORACLE NOTE: the auto-selected FlashInfer-CUTLASS unquantized-MoE backend OOM-kills the 57 GiB model on GB10's 119 GiB unified pool (57 GiB weights + 57 GiB checkpoint page-cache + workspace), so the oracle runs vLLM's first-class **TRITON** MoE backend (same bf16 grouped-MoE math) with gpu-mem-util 0.58 + a checkpoint page-cache evictor; deterministic K=5. New files: `tests/vllm/models/test_qwen3coder_paged_engine.cpp` + `scripts/qwen3coder-oracle-capture.py` + `scripts/qwen3coder-neartie-gap.py` + goldens `tests/parity/goldens/qwen3coder_greedy/{greedy_ids,greedy_dist,our_ids,neartie_gap_mnats}.npy` + `p{0..5}_prompt.i32`. GATES: dgx CUDA `-Werror` 0-warn; 6/6 gate; 27B 235/235 + 35B 315/315 UNCHANGED; memcheck 0 on the engine path. Row → correctness-DONE (only W5 speed remains).

**W5 bf16-fast-MoE grouped GEMM + every-axis speed — LANDED 2026-07-21 (row stays `ACTIVE`: correctness DONE, speed residual documented).**

*Delivered.* (a) NEW op `vt::MoeGroupedGemmBf16` (`include/vt/ops.h` `OpId::kMoeGroupedGemmBf16` + `src/vt/ops.cpp` validation + `src/vt/cuda/cuda_matmul_nvfp4.cu` CUDA impl) — the dtype-native analog of `MoeGroupedGemmNvfp4`, STRUCTURALLY REUSING the NVFP4 grouped-MoE scheduling verbatim (the same `MoeHistKernel`/`MoeOffsetsKernel`/`MoeScatterKernel`/`MoeTileMapKernel` expert counting-sort + ragged per-BM-tile expert map + `EnsureMoeScratch` graph-safe persistent scratch, `cuda_matmul_nvfp4.cu:617-661,766-788`), with the on-the-fly fp4 decode replaced by a direct bf16 `[K,N]` weight read. Three launch regimes mirroring the fp4 path: naive one-thread-per-output for `P < kTileMinRows(32)` (decode), BM=16 decode WMMA tile for `P <= kMoeDecodeMaxP(512)` (mirrors vLLM `fused_moe` `BLOCK_SIZE_M=16` for small M), BM=64 prefill WMMA tile above. f32 accumulation; f32 out (gate/up, matching the reference `MatmulF32`) or bf16 out (down). (b) `MoeBlockBf16Cuda` (`qwen3_5.cpp`) — the bf16 analog of `MoeBlockFusedCuda`: replaces the per-expert host-gather reference loop (download hidden → HOST router gather → up to E serialized cuBLASLt `ExpertMlp` launches) with ~3 grouped GEMM launches kept entirely on-device, per-layer RESIDENT expert device-pointer arrays + pair→token row map. **DEFAULT ON** (`VT_MOE_BF16_FAST=0` rolls back), per [[parity-enablers-ship-as-defaults]], gated BEFORE the binding run. (c) **LAYOUT GUARD** `MoeBf16FastLayoutOk` — the grouped kernel can only read the `nk == false` `[K,N]` Matmul-B orientation, so the fast path is taken ONLY when the router gate + every expert is in that layout; the `nk == true` producer (35B MTP `LoadBf16RawNK`/`CopyRawNK`, `qwen3_5_mtp.cpp:109-133`) and anything else fall through to the layout-agnostic reference loop instead of being silently transposed. (d) **HOST-MIRROR RELEASE** — the decisive memory lever, reusing the EXISTING `OwnedTensor::ReleaseHost()` + `platforms::ShouldReleaseHostWeights` residency mechanism the 35B Marlin path already uses (`qwen3_5.cpp:4185-4241`): the routed experts are ~57 GiB (~94% of the checkpoint) and host `.bytes` + device `d_dev` are DISTINCT allocations out of ONE 119 GiB GB10 unified pool. Released per layer at first touch (so peak never doubles). `VT_MOE_HOST_FREE=0` A/B.

*Correctness (SACRED) — RE-VERIFIED, and the gate got STRICTER.* Swapping a per-expert cuBLASLt loop for a grouped tensor-core GEMM changes the f32 ACCUMULATION ORDER, which re-resolves bf16 near-ties, so one token moved (prompt[3] tok2). Ratified procedure followed: re-dump our ids → re-teacher-force vLLM (`scripts/qwen3coder-neartie-gap.py`) → re-run the SAME 0.5-nat gate. RESULT **6/6 PASS** (138 assertions) with **STRICT token-exact 4/6 → 5/6** and **max teacher-forced gap 0.125 nats → 0.0000 nats** (the single remaining divergent token IS vLLM's own argmax on our prefix), **0 forward-divergent**. Expected: vLLM computes these experts with its own Triton GROUPED `fused_moe` GEMM, so a grouped GEMM lands CLOSER to vLLM than a per-expert loop. The gate was NOT loosened — the goldens are re-measured, the bar is unchanged. NEW kernel-level test `tests/vt/test_ops_moe_grouped_bf16.cpp` (mirrors `test_ops_moe_grouped.cpp`) proves the GEMM against the single-expert f32 reference across ALL THREE launch regimes + both out dtypes + identity/`row_map` routing: **4/4**.

*Same-binary A/B (W5 win).* `VT_MOE_BF16_FAST` 0 → 1, same binary, c1 input-len 1024: median TTFT **3092 → 904 ms (3.4×)**, median TPOT **64.4 → 40.4 ms (1.59×)**, per-stream decode **15.9 → 24.8 tok/s (1.56×)**. (Reference arm ran 4 prompts × 32 output tokens; fast arm 16 × 128 — per-request medians, so comparable.)

*BINDING BENCHMARK vs vLLM 0.25.0 — honest denominator.* vLLM = PRODUCTION GRAPHED (`enforce_eager=False`, `CUDAGraphMode.FULL_AND_PIECEWISE`, Inductor `VLLM_COMPILE`, `moe_backend='triton'` — the OOM-safe backend; FlashInfer-CUTLASS OOM-kills the 57 GiB model on GB10's 119 GiB unified pool), `vllm serve` + `vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 --ignore-eos --max-concurrency C`; ours = `examples/vllm-bench --input-len 1024 --output-len 128 --concurrency C` (production defaults). num-prompts 16/24/48/96 for c1/c2/c4/c8. Idle box, `flock /tmp/gpu`, checkpoint page-cache evicted between runs. Ratios are ours÷vLLM for throughput, vLLM÷ours for latency (>1 = we win):

| axis | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| median TTFT ms (ours / vLLM) | 904.5 / 370.0 → **0.41×** | 1684.6 / 248.9 → **0.15×** | 2489.6 / 458.4 → **0.18×** | 2501.4 / 600.4 → **0.24×** |
| median TPOT ms (ours / vLLM) | 40.40 / 32.22 → **0.80×** | 43.81 / 44.71 → **1.02×** | 98.98 / 74.09 → **0.75×** | 148.85 / 81.80 → **0.55×** |
| output throughput tok/s (ours / vLLM) | 18.55 / 28.76 → **0.65×** | 31.85 / 42.32 → **0.75×** | 32.20 / 51.24 → **0.63×** | 46.55 / 94.52 → **0.49×** |

**PARITY VERDICT: MISS — every-axis parity NOT met; row stays `ACTIVE`.** Only c2 TPOT (1.02×) reaches parity. NOT hand-waved — attributed below by kernel from an `nsys --cuda-graph-trace=node` capture of ours at c1 (`nsys stats --report cuda_gpu_kern_sum`, 2 × 1024-token prefill + 16 decode steps):

1. **TTFT (worst axis, 0.41× @ c1) = ENTIRELY the bf16 prefill MoE GEMM.** `MoeGroupedGemmBf16Wmma<float>` (gate+up) **35.9%** of all GPU time (192 inst, 5.71 ms avg) + `MoeGroupedGemmBf16Wmma<bf16>` (down) **20.1%** (96 inst, 6.39 ms avg) = **56.0%**. Per prefill: 48 layers × (2 × 5.71 + 6.39) = **855 ms ≈ our entire 904 ms TTFT**. Quantified: a 1024-token prefill is 3.71 TFLOP of MoE GEMM, so our WMMA kernel sustains **~4.3 TFLOP/s ≈ 1.7% of GB10 bf16 tensor-core peak**; vLLM's Triton `fused_moe` sustains **~10.0 TFLOP/s** on the same work (2.3× us — and itself untuned: vLLM logs `Using default MoE config. Performance might be sub-optimal! Config file not found at .../E=128,N=768,device_name=NVIDIA_GB10.json`). ROOT CAUSE: the WMMA tile is a first-cut naive GEMM, not a tuned one — `BK=32` (tiny K-tile ⇒ poor operand reuse), NO `cp.async` double-buffering (full `__syncthreads` stall per K-tile), `BN=64`, and no register blocking beyond the wmma fragments. LEVER: tile/pipeline it (larger BK/BN, async-copy double buffer, register blocking) — headroom is ~2.3× to reach vLLM and far more to reach the hardware.
2. **TPOT (0.80× @ c1 degrading to 0.55× @ c8) = the naive decode kernel + two missing structural mirrors.** `MoeGroupedGemmBf16Naive<float>` (gate/up) **15.7%** (166.7 µs avg) + `<bf16>` (down) **5.6%** (119.5 µs avg) = **21.3%**, i.e. ~453 µs/layer ⇒ **21.7 ms of our 40 ms TPOT is MoE**. Bandwidth: gate/up moves 25.2 MB in 166.7 µs = **151 GB/s (55% of GB10's ~273 GB/s)**; down moves the same 25.2 MB in 119.5 µs = **211 GB/s (77%)**. The gap between them is BLOCK-PARALLELISM STARVATION at small N: gate/up has `N=768` ⇒ grid `(⌈768/256⌉=3, P=8)` = **24 blocks**, down has `N=2048` ⇒ **64 blocks**. Three named levers, in gain order: (i) mirror vLLM `fused_moe`'s **w13 gate+up FUSION** — one grouped GEMM over a concatenated `[K, 2I]` operand (we already ship exactly this for Marlin as `VT_MOE_FUSED_W13`), doubling N to 1536 and halving the sort/launch count; (ii) deterministic **split-K** (two-pass partials + fixed-order reduce — NOT `atomicAdd`, which would break run-to-run token determinism) to lift gate/up off 55% bandwidth; (iii) **no bf16 decode CUDA graph** — the 35B decode graph is fp4-gated (`qwen3_5_moe.cpp:76-82`), so Coder pays the eager per-step host tax that vLLM's `FULL_AND_PIECEWISE` graph avoids (this is also why our TPOT degrades 3.4× c2→c8 while vLLM's degrades only 1.83×).
3. **Memory axis: closed by (d).** Before the host-mirror release the model cost ~114 GiB of the 119 GiB unified pool (61 GiB host mirror + 57 GiB device), leaving free=1 GiB — MEASURED collapse: c2 output throughput fell BELOW c1 and c4 never completed. After the release ours is ~62 GiB vs vLLM's 56.93 GiB weights + 11.38 GiB KV, and c2/c4/c8 scale normally. Every pre-fix c≥2 number is VOID.

*Gates.* dgx CUDA `-Werror` **0-warn** (clean full build); `test_qwen3coder_paged_engine` **6/6** with the fast GEMM DEFAULT-ON; `test_ops_moe_grouped_bf16` **4/4**; `test_qwen3_moe_forward` **3/3** (real-checkpoint prefill argmax **12095 " Paris"** UNCHANGED); regression **27B 235/235 + 35B 315/315 UNCHANGED**; memcheck **0 errors** on the MoE GEMM (`test_ops_moe_grouped_bf16`) and the engine path (`test_llm_engine` 5/5, `test_runner` 14/14) — the only retained allocation is the by-design graph-safe `EnsureMoeScratch` block (`RetireGraphScratch` never frees it; the pre-existing NVFP4 `test_ops_moe_grouped` shows the identical pattern).

*Remaining to close the row to `DONE`:* the three named decode levers (w13 fusion, deterministic split-K, bf16 decode graph) and the prefill WMMA tile/pipeline rework.

**W6 MoE-GEMM tile/pipeline rework + deterministic split-K — LANDED 2026-07-21 (row stays `ACTIVE`: 12 of 16 grid cells at/above vLLM, 4 residual cells attributed).**

*Delivered — two kernel levers, both in `src/vt/cuda/cuda_matmul_nvfp4.cu`.*

**(1) `MoeGroupedGemmBf16WmmaPipe` — the tuned prefill/decode tile (W5 lever iv).** The W5 tile's dominant defect was NOT the MMA: its weight stage read `w[gk * n_cols + gn]` with the fastest-varying loop index on `gk`, so consecutive lanes touched addresses `n_cols` elements apart — a fully UNCOALESCED read of the `[K,N]` expert weight (32 distinct 32-byte sectors per warp, 2 useful bytes each). A 1024-token prefill streams ~1.2 GB of expert weight per layer against GB10's ~273 GB/s at an arithmetic intensity of ~64 flop/byte (machine balance ~900), so this GEMM is WEIGHT-BANDWIDTH bound and the staging, not the tensor cores, set the ~4.3 TFLOP/s ceiling. The rewrite: (a) shared `Ws` transposed to `[BK][BN]` (k-major) and the fragment loaded as `wmma::matrix_b` in ROW_MAJOR — which for matrix_b means K-major — so the global read runs along N, contiguous in the weight; (b) 16-byte vectorized activation and weight stages; (c) a 3-deep `cp.async` MULTI-STAGE PIPELINE replacing the per-K-tile `__syncthreads` stall; (d) BN 64 -> 128; (e) +8-half shared-row padding (16-byte aligned) to cap wmma fragment loads at 2-way bank conflicts. Tile/pipeline SHAPE ported from vLLM's Triton grouped-MoE GEMM — `vllm/model_executor/layers/fused_moe/fused_moe.py:294` (`fused_moe_kernel`'s f32-accumulate `for k in range(cdiv(K, BLOCK_SIZE_K))` mainloop) with the constants from `fused_moe.py:1238` (`get_default_config` bf16 branch: `block_n = 64 if M <= 64 else 128`, `num_stages = 4 if M <= 32 else 3`, `num_warps`); the C++ form of what Triton's `num_stages` lowers to is CUTLASS `include/cutlass/gemm/threadblock/mma_multistage.h` (STAGES-1 prologue groups, `cp_async_wait<STAGES-2>` + issue-next-in-loop, tail iterations committed with predicates false). BK stays 32 (not vLLM's 64) because at BN=128/STAGES=3 that is what fits the 48 KB static shared budget while holding a 3-deep pipeline. Shape-guarded by `Bf16PipeShapeOk` (8-bf16 row pitches, so every `cp.async` granule is 16-byte aligned); ragged shapes fall through to the W5 tile, which stays live as the layout-agnostic reference. `VT_MOE_BF16_PIPE=0` rolls back.

**(2) `MoeGroupedGemmBf16NaiveSplitK` + `MoeGroupedGemmBf16SplitKReduce` — DETERMINISTIC split-K for small-P decode (W5 lever ii).** At c=1 decode the naive kernel ran on grid `(ceil(768/256), P=8)` = **24 thread blocks** of pure weight streaming — measured 151 GB/s = 55% of peak, i.e. block-parallelism starvation, not a memory-system limit. Split-K partitions the K reduction across `MoeSplitKCount` blocks (mirrors vLLM's `SPLIT_K` knob on the same kernel, `fused_moe.py:338` threaded through `get_default_config` at :1254/:1300/:1350). We deliberately do NOT mirror an atomicAdd accumulation — float atomics reduce in nondeterministic ORDER and would break greedy reproducibility and the SACRED gate; instead each split writes an f32 PARTIAL and a second pass sums them in FIXED ascending split order. The split count is a pure function of the (deterministic) launch shape. Partials live in `EnsureMoePartials`, the f32 analog of `EnsureMoeScratch` with the same graph-safe retire-don't-free contract. `VT_MOE_SPLIT_K=0` rolls back.

*Correctness (SACRED) — UNCHANGED, zero token movement.* The pipelined tile preserves the W5 tile's per-output K-reduction ORDER exactly (ascending k, one m16n16k16 accumulate per 16 k, accumulator carried across K-tiles), so it is BIT-IDENTICAL to W5 — only the staging changed. `test_qwen3coder_paged_engine` **6/6 PASS** (138 assertions), STRICT token-exact **5/6**, **max teacher-forced gap 0.0000 nats**, **0 forward-divergent** — identical to W5, so no golden re-dump was needed (goldens md5-verified identical on dgx before and after). `test_ops_moe_grouped_bf16` extended 4 -> **7 cases / 19 assertions**, adding an aligned-pitch pipelined PREFILL case (K=264 not a multiple of BK=32, N=200 not a multiple of BN=128 — exercising the `cp.async` zfill tails), an aligned-pitch pipelined DECODE case + the identity-row-map/bf16-out down-projection shape, and a split-K case that additionally asserts the reduction is RUN-TO-RUN BIT-REPRODUCIBLE over 3 reps. The pre-existing ragged-pitch cases now cover the retained W5 fallback tile. One real bug was caught by the new decode case and fixed before landing: the mainloop skipped `__pipeline_commit()` past the last k-tile, so `__pipeline_wait_prior` under-counted and the final tiles could be consumed before their copies landed.

*Kernel A/B (same binary, `VT_MOE_BF16_PIPE`/`VT_MOE_SPLIT_K` 0 -> 1; E=128, K=2048, top-8, 20-100 reps):*

| shape | W5 | W6 | speedup |
|---|---|---|---|
| prefill T=1024 (P=8192), N=768 | 5.724 ms / **4.50 TFLOP/s** | 2.121 ms / **12.15 TFLOP/s** | **2.70x** |
| prefill T=1024 (P=8192), N=2048 | 15.304 ms / **4.49 TFLOP/s** | 5.505 ms / **12.48 TFLOP/s** | **2.78x** |
| decode-WMMA T=8 (P=64), N=768 | 2.738 ms | 0.964 ms | **2.84x** |
| split-K T=1 (P=8), N=768 | 0.108 ms | 0.041 ms | **2.63x** |

vLLM's Triton `fused_moe` sustains **~10.0 TFLOP/s** on the same work (W5 measurement), so the bf16 MoE GEMM now runs at **~1.2x vLLM's rate** — the W5 headline kernel deficit (1.7% of bf16 peak) is CLOSED. In situ at c2 the pipelined tile reaches **19.9 TFLOP/s** on the 2048-token prefill step (two M-tiles per expert let the weight tile hit in L2).

*DENOMINATOR CORRECTION (applies retroactively to the W5 grid).* W5 drove all four concurrencies against ONE `vllm serve` process with `vllm bench serve --seed 0` each time. vLLM's `RandomDataset` builds prompts as the reproducible sequence `(offset + index + arange(input_len)) % vocab_size` (`vllm/benchmarks/datasets/datasets.py:557-566`), so runs 2-4 REPLAYED run 1's prompts into vLLM's prefix cache (`enable_prefix_caching=True` by default): the serve log shows **43.6%-63.5% prefix-cache hit rate**, and vLLM's c2 TTFT (248.9 ms) came out FASTER than its c1 (370.0 ms). Our arm is a fresh process per concurrency, so it never had that carry-over — the W5 c2/c4/c8 denominators were inflated and the W5 ratios there are PESSIMISTIC. W6 re-measures with a **FRESH `vllm serve` per concurrency**, confirmed **0.0% prefix-cache hit rate** in all four runs. (c1 was always clean: 324.04 ms then, 321.88 ms now — 0.7% apart, which also bounds run-to-run noise on this axis.) Our own `--enable-prefix-caching` mirrors vLLM and is likewise default-ON for Qwen3-Coder (`model_loader.cpp:121-131`, non-hybrid decoder-only), so the two arms are configured alike.

*BINDING BENCHMARK vs vLLM 0.25.0 — same production-graphed oracle, CLEAN denominator.* vLLM = `enforce_eager=False`, `CUDAGraphMode.FULL_AND_PIECEWISE`, Inductor `VLLM_COMPILE`, `moe_backend='triton'`, fresh server per C; ours = `examples/vllm-bench` production defaults. random 1024/128 `--ignore-eos`, num-prompts 16/24/48/96, idle box under one `flock /tmp/gpu`, checkpoint page-cache evicted. Ratios are vLLM÷ours for latency, ours÷vLLM for throughput (>1 = we win):

| axis | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| median TTFT ms (ours / vLLM) | 313.9 / 321.9 → **1.03x WIN** | 533.6 / 435.5 → **0.82x** | 754.1 / 789.6 → **1.05x WIN** | 762.3 / 904.7 → **1.19x WIN** |
| median TPOT ms (ours / vLLM) | 36.22 / 31.75 → **0.88x** | 44.12 / 44.67 → **1.01x WIN** | 60.74 / 74.26 → **1.22x WIN** | 81.41 / 85.29 → **1.05x WIN** |
| median ITL ms (ours / vLLM) | 36.14 / 31.79 → **0.88x** | 42.94 / 44.06 → **1.03x WIN** | 58.40 / 71.41 → **1.22x WIN** | 70.97 / 77.56 → **1.09x WIN** |
| output throughput tok/s (ours / vLLM) | 22.62 / 29.45 → **0.77x** | 37.24 / 41.44 → **0.90x** | 55.03 / 50.45 → **1.09x WIN** | 86.02 / 87.27 → **0.99x** |

Ours vs OUR OWN W5 numbers (identical methodology, so directly comparable): median TTFT **904.5 -> 313.9 / 1684.6 -> 533.6 / 2489.6 -> 754.1 / 2501.4 -> 762.3 ms** (2.9x-3.3x); median TPOT **40.40 -> 36.22 / 43.81 -> 44.12 / 98.98 -> 60.74 / 148.85 -> 81.41 ms**; output throughput **18.55 -> 22.62 / 31.85 -> 37.24 / 32.20 -> 55.03 / 46.55 -> 86.02 tok/s**. TPOT no longer degrades pathologically with concurrency (W5 3.4x c2->c8; now 1.8x, matching vLLM's 1.9x).

**PARITY VERDICT: STILL A MISS — every-axis parity NOT met; row stays `ACTIVE`.** **c4 passes every axis** (1.05x / 1.22x / 1.22x / 1.09x) and **c8 passes three of four** (TTFT 1.19x, TPOT 1.05x, ITL 1.09x) with output throughput 0.99x. The **4 residual cells** are c1 TPOT/ITL (0.88x) + c1 output throughput (0.77x), c2 TTFT (0.82x) + c2 output throughput (0.90x), and c8 output throughput (0.99x). Attributed from `nsys --cuda-graph-trace=node` + `cuda_gpu_kern_sum` captures of ours at c1 and c2:

1. **c1 decode residual = the MISSING bf16 DECODE CUDA GRAPH (host tax), not kernel speed.** Per decode step the summed GPU kernel time is **~31 ms against a measured 36.22 ms TPOT ⇒ ~86% GPU-busy, ~5 ms/step of host/launch gap** — and 4.5 ms/step is exactly the whole c1 TPOT deficit vs vLLM. The kernels themselves are now near the memory roof: the fused decode MoE GEMM (`MoeGroupedGemmBf16NaiveSplitK<float>`, one instantiation serving gate+up+down) is **37.9%** of GPU time at **116.4 us** avg (W5: 166.7 us for gate/up alone), and the dense qkv/o projections (cuBLAS `gemvx`) are **24.5%** at ~190-211 GB/s = **70-77% of GB10's ~273 GB/s**. vLLM runs this step under `FULL_AND_PIECEWISE` graphs; we run it eager. **This is W5's named lever (iii) and is now the single largest remaining item** — it also explains the c1 output-throughput cell (0.77x), which is TPOT-driven at concurrency 1.
2. **c2 TTFT residual is NO LONGER the MoE GEMM.** The prefill tile dropped from **56.0% to 16.6%** of GPU time at c1 and is **21.1%** at c2; per 2048-token prefill step it is 48 x (2 x 2.487 + 2.812) = **374 ms**, i.e. 7.42 TFLOP at **19.9 TFLOP/s**, against a 533.6 ms TTFT and vLLM's 435.5 ms. So the ~100 ms c2 deficit sits in the NON-MoE prefill path (attention + dense projections + per-step glue), not in the grouped GEMM. Next lever there is a prefill-step profile of the non-MoE glue, mirroring the Qwen3-dense TTFT campaign.
3. **Levers NOT taken (and why).** vLLM's **w13 gate+up fusion** (W5 lever i) was left unimplemented: its two benefits are doubling the N-grid (block starvation — already solved by split-K, which is strictly more general because it also helps the down projection) and halving the launch count (subsumed by the decode-graph lever). It remains available as a follow-on if the decode graph does not close c1. Building it as a genuine `[K,2I]` concatenated operand would also need the loader to materialise fused expert buffers, and the routed experts are ~57 GiB out of a 119 GiB unified pool, so a transitional double-allocation is not free — the cheap form is a dual-operand launch (one kernel, weight-pointer array and output pointer selected by N-tile half), which is where a follow-on should start.

*Gates.* dgx CUDA `-Werror` **0-warn** (clean full rebuild from scratch); `test_qwen3coder_paged_engine` **6/6** with both new kernels DEFAULT-ON; `test_ops_moe_grouped_bf16` **7/7 (19 assertions)**; `test_ops_moe_grouped` (NVFP4, unaffected) **9/9**; `test_qwen3_moe_forward` **3/3**; regression **27B 235/235 + 35B 315/315 UNCHANGED**; `compute-sanitizer memcheck` **0 memory errors** on the MoE GEMM and on the engine path (`test_llm_engine` 5/5, `test_runner` 14/14) — the only reported allocations are the by-design graph-safe persistent scratch blocks (`EnsureMoeScratch` + the new `EnsureMoePartials`, both `RetireGraphScratch`-retained; the pre-existing NVFP4 path shows the identical pattern).

*Remaining to close the row to `DONE`:* (i) the **bf16 decode CUDA graph** (largest, closes c1 TPOT/ITL/throughput), (ii) the **non-MoE prefill glue** at c2 (closes c2 TTFT/throughput), (iii) optionally the dual-operand w13 fusion.

**W7 BF16 decode CUDA graph — LANDED 2026-07-21 (W6 lever (i)).**

*Delivered — `Qwen3MoeDecodeGraph`, the BF16 full-attention-MoE decode CUDA-graph driver.* New class declared in `include/vllm/model_executor/models/qwen3_moe.h` and implemented in `src/vllm/model_executor/models/qwen3_moe.cpp`, dispatched from `src/vllm/model_executor/models/qwen3_moe_registry.cpp::ForwardQwen3MoeForCausalLM` on `input.pure_decode && platform.is_cuda()`. It is the third sibling of the SAME driver design already gated in-tree — `Qwen3_5DecodeGraph` (35B hybrid MoE, `qwen3_5.cpp:5902`) and `Qwen3_5DenseDecodeGraph` (27B dense, `qwen3_5.cpp:6104`) — reusing their cold -> warm -> capture -> replay state machine, their per-padded-size `SizeSlot` (own fixed-address host inputs + persistent embed target + persistent logits output + instantiated graph), the `cols_changed` graph invalidation, and the shared capture-size set `DecodeGraphSizes`/`PadToCaptureSize` (`include/vllm/model_executor/models/decode_graph_sizes.h`).

*Ported from upstream.* `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner` @ `e24d1b24` (the `_dummy_run` warm-up then `capture_model` capture, and the per-decode-step graph dispatch) + `vllm/compilation/cuda_graph.py::CUDAGraphWrapper.__call__` (pad the batch up to the nearest captured size, replay, else run eager). The capture-size set itself is the already-ported `_set_cudagraph_sizes` reduction (`vllm/config/vllm.py`, + `vllm/config/compilation.py:683-684,1438-1444`) recorded in `decode_graph_sizes.h`. In-repo template: `Qwen3_5DecodeGraph` (`qwen3_5.cpp:5902-6091`).

*Two structural differences from the siblings, both because Qwen3-Coder is pure full attention.* (a) **No GDN**: the padded-input builder is attention-only (`BuildPaddedDecodeAttn`, the GDN-free analog of `qwen3_5.cpp:5843` `BuildPaddedDecode`) and there is no `ValidateGdnDecodeGraphState` / `CanUseGdnDecodeGraphSize` gate, so EVERY captured size is usable. Padding rows are inert exactly as vLLM's: token/position 0, `slot_mapping = -1` (ReshapeAndCache skips the KV write, `cuda_cache.cu:50`), `seq_lens = 1` + block-table row 0 (a valid in-bounds read whose output row is discarded). The decode forward is row-independent — paged attention is per-request causal and the grouped-MoE GEMM's counting sort groups (token,expert) PAIRS with each output row reducing only over its own K — so padding cannot perturb the real rows. (b) **No defensive copy of the persistent hidden buffer**: unlike the 35B's `RunLayerPaged`, the Coder layer loop only READS `hidden_in` (the fused add+RMSNorm accumulates into `res` and `hidden` is then reassigned to the MoE output), so `ForwardLayers` consumes the slot's persistent embed buffer directly.

*Refactor.* `ForwardBody` was split into `EmbedInto` (embed only — kept OUTSIDE the graph because `vt::Embedding` allocates a device bounds-check flag and synchronizes the stream, `cuda_ops.cu:525,535`, both illegal inside a capture region, and it consumes the HOST token ids) + `ForwardLayers` (the capturable region: residual init, the 48 full-attention MoE layers, final RMSNorm, untied lm_head), with `ForwardBody` retained as the eager composition used by `Forward`/`ForwardDevice` and by the driver's eager fallback / cold-size pre-warm. VERIFIED BEHAVIOR-PRESERVING: with `VT_QWEN3MOE_CUDAGRAPH=0` the SACRED gate passes 6/6 on the refactored binary.

*ONE REAL BUG FOUND AND FIXED — a latent graph-safety defect in the shared dense attention glue.* The first capture produced a WRONG token (`prompt[0] tok=2`, engine 3555 vs committed 576) — a gross divergence, not a near-tie, and exactly the "if any token moves, investigate rather than re-baseline" signal. Root cause: `BuildStepInputs` (`dense_attn_block.h`) built the RoPE identity row-index in a **stack-local** `std::vector<int32_t>` and uploaded it with a host->device copy. Eagerly that is fine; under capture the copy becomes a graph memcpy NODE that bakes the host SOURCE ADDRESS, and that address is dead stack the instant `BuildStepInputs` returns — so every replay re-read freed stack memory, giving a garbage token-index lookup into the cos|sin cache and a wrong RoPE. Fixed by serving the identity index from a process-persistent per-T table (contents are a pure function of T, storage created once per distinct T and never resized or moved), the host-side analog of the `EnsureMoeScratch`/`EnsureMoePartials` retire-don't-free contract (`cuda_matmul_nvfp4.cu:767,986`). Contents are byte-identical to the stack-local form, so the eager dense path is unchanged (`test_qwen3_paged_engine` still passes). This was latent, not a regression: the 35B/27B drivers use a different builder (`BuildStepDevInputs`) and the Qwen3-dense path had no decode graph, so Qwen3-Coder is the first caller to capture this glue.

*Graph-safety audit of the bf16 decode path (recorded because capture requires stable pointers and no host sync / stream-ordered alloc inside the region).* Embedding stays outside. All device scratch comes from the shared `DevicePool`, whose blocks are recycled and never returned to the driver, and the cold pre-warm step at the exact padded size populates every size class the capture then reuses. The grouped-MoE index scratch (`EnsureMoeScratch`) and split-K partials (`EnsureMoePartials`) are graph-safe by design. The per-layer expert device-pointer arrays and the pair->token row map (`MoeBf16Resident`, `qwen3_5.cpp:4436-4505`) are uploaded ONCE at first touch, during the pre-warm. `ResidentWeight` uploads each weight once. The FA-2 varlen-decode launcher throws on a per-shape scratch MISS during capture (`cuda_flash_attn_fa2.cu:706`), and the pre-warm at the same size populates it; its host `max_seq_len` only sizes the split-KV grid while the per-request causal geometry and each split's KV range come from the DEVICE `seqused_k`, so a captured graph stays correct as sequences grow (the identical contract the already-gated 35B/27B graphs rely on). cuBLASLt's workspace is a one-time per-context allocation (`cuda_matmul.cu:101`).

*Correctness (SACRED) — ZERO TOKEN MOVEMENT, as a numerics-neutral graph must be.* A CUDA graph replays the same kernels in the same order over the same buffers, so the expectation was bit-identity, and after the row-index fix that is exactly what is measured: `test_qwen3coder_paged_engine` **6/6 PASS** (138 assertions) with the graph DEFAULT-ON, STRICT token-exact **5/6**, **max teacher-forced gap 0.0000 nats**, **0 forward-divergent** — identical to W5/W6, no golden re-dump needed (goldens md5-verified identical on dgx before and after the run). The gate was NOT loosened.

*Rollback.* The framework-wide `VLLM_CPP_CUDAGRAPH=0` disables it, as does the lever-local `VT_QWEN3MOE_CUDAGRAPH=0` (same-binary A/B). `VT_DECODE_GRAPH_STATS=1` prints each capture.

*MEASURED host-tax reduction at c1 (`nsys --cuda-graph-trace=node` + `cuda_gpu_kern_sum`, graph-node rows present so the attribution is complete).* Summed decode GPU kernel time is **~31.5 ms/step** over ~60.7 captured decode steps, against a measured median TPOT of **34.22 ms** => **~92% GPU-busy, ~2.7 ms/step of host/launch tax**, versus W6's **~86% GPU-busy, ~5 ms/step**. So the graph removed **~2.3 ms/step (~46%) of the host tax** and median TPOT went **36.22 -> 34.22 ms**. Per-step kernel shares are unchanged from W6 (the decode MoE GEMM `MoeGroupedGemmBf16NaiveSplitK` **41.7%** at 115.1 us avg = 16.6 ms/step; the dense qkv/o cuBLAS `gemvx` family **27.7%** = ~11.0 ms/step; FA-2 splitkv decode 3.6%), confirming this was a host-side, not kernel-side, lever. Launch-count corroboration from the same capture: **60 `cudaGraphLaunch`** (one per decode step — the graph really is serving every decode step) against only **3,569 `cudaLaunchKernel` in the entire window** (~59/step outside the graph: embed + sampler + the prefill steps), versus the ~1.4k eager launches/step the pre-graph path issued.

*BINDING BENCHMARK vs vLLM 0.25.0 — same production-graphed oracle, same methodology as W6.* vLLM = `enforce_eager=False`, `CUDAGraphMode.FULL_AND_PIECEWISE`, Inductor `VLLM_COMPILE`, `moe_backend='triton'`, `--gpu-memory-utilization 0.58 --max-model-len 2048 --max-num-seqs 32`, **FRESH `vllm serve` per concurrency** with **0.0% prefix-cache hit rate VERIFIED in all four serve logs**; ours = `examples/vllm-bench` production defaults, fresh process per concurrency, 2 reps each with the cold first leg discarded (rep-to-rep spread <= 2.6% on every cell). random 1024/128 `--ignore-eos`, num-prompts 16/24/48/96, idle box under one `flock /tmp/gpu`, checkpoint page-cache evicted between runs. Ratios are vLLM/ours for latency, ours/vLLM for throughput (>1 = we win):

| axis | c1 | c2 | c4 | c8 |
|---|---|---|---|---|
| median TTFT ms (ours / vLLM) | 315.3 / 320.4 -> **1.02x WIN** | 534.1 / 485.7 -> **0.91x** | 753.4 / 792.3 -> **1.05x WIN** | 766.3 / 900.3 -> **1.17x WIN** |
| median TPOT ms (ours / vLLM) | 34.13 / 31.75 -> **0.93x** | 39.68 / 44.56 -> **1.12x WIN** | 57.97 / 73.80 -> **1.27x WIN** | 78.22 / 84.59 -> **1.08x WIN** |
| median ITL ms (ours / vLLM) | 34.02 / 31.76 -> **0.93x** | 40.10 / 43.96 -> **1.10x WIN** | 55.54 / 71.21 -> **1.28x WIN** | 68.05 / 77.50 -> **1.14x WIN** |
| output throughput tok/s (ours / vLLM) | 24.60 / 29.46 -> **0.84x** | 39.85 / 41.64 -> **0.96x** | 56.34 / 50.63 -> **1.11x WIN** | 90.10 / 87.27 -> **1.03x WIN** |

*Denominator reproducibility.* Three of the four vLLM cells reproduce W6's numbers to within 0.5%: c1 TTFT 321.9 -> 320.4, c4 792.3 vs 789.6, c8 900.3 vs 904.7; c1 TPOT is IDENTICAL at 31.75 ms. The exception is **c2 TTFT, which swung 435.5 -> 485.7 ms (+11.5%) between W6 and W7** — so that single cell's denominator carries real run-to-run variance and its ratio should be read as a band (0.82x-0.91x), not a point. Our own c2 TTFT is unchanged (533.6 -> 534.1 ms), which is expected: a DECODE graph cannot move TTFT.

*Ours vs OUR OWN W6 (identical methodology, directly comparable).* median TPOT **36.22 -> 34.13 / 44.12 -> 39.68 / 60.74 -> 57.97 / 81.41 -> 78.22 ms**; median ITL **36.14 -> 34.02 / 42.94 -> 40.10 / 58.40 -> 55.54 / 70.97 -> 68.05 ms**; output throughput **22.62 -> 24.60 / 37.24 -> 39.85 / 55.03 -> 56.34 / 86.02 -> 90.10 tok/s**; median TTFT unchanged within noise (313.9 -> 315.3 / 533.6 -> 534.1 / 754.1 -> 753.4 / 762.3 -> 766.3 ms), exactly as a decode-only lever should behave. The win is a clean per-step WALL-TIME reduction, so it lifts BOTH latency and throughput (the closed-loop coupling check passes: no cell traded TPOT for throughput).

**PARITY VERDICT: STILL A MISS — every-axis parity NOT met; row stays `ACTIVE`.** **11 of 16 cells at/above vLLM. c4 passes EVERY axis (1.05x/1.27x/1.28x/1.11x) and c8 now ALSO passes EVERY axis (1.17x/1.08x/1.14x/1.03x)** — c8 output throughput crossed 0.99x -> 1.03x, so two of the four concurrencies are fully clean. The **5 residual cells are all at c1 and c2**: c1 TPOT (0.93x), c1 ITL (0.93x), c1 output throughput (0.84x), c2 TTFT (0.91x), c2 output throughput (0.96x). Every one improved over W6 (0.88/0.88/0.77/0.82/0.90).

*Attribution of the residual — and an explicitly REFUTED hypothesis.*

1. **c1 TPOT/ITL/throughput = the REMAINING ~2.7 ms/step host tax, still not kernel speed.** Our summed decode GPU kernel time is ~31.5 ms/step and vLLM's median TPOT is 31.75 ms — i.e. **vLLM's whole decode step costs about what our KERNELS ALONE cost**, so closing the remaining host tax would land c1 at parity and no kernel work is required to get there. c1 output throughput (0.84x) is TPOT-driven at concurrency 1 and follows.
2. **REFUTED: the residual is NOT CUDA-API overhead.** The obvious suspect was `vt::Embedding`, which is deliberately kept outside the graph and per call does `cudaMalloc` + kernel + `cudaMemcpyAsync` D2H + `cudaStreamSynchronize` + `cudaFree` (`cuda_ops.cu:525-538`) — a full pipeline drain plus two device-serializing allocator calls every decode step. `nsys --report cuda_api_sum` DISPROVES it: `cudaStreamSynchronize` has a **median of 4.5 us** (171 calls; the large mean is entirely model-load syncs), and the 19,184 `cudaMalloc` calls are dominated by the one-time `ResidentWeight` weight uploads, not the steady state. `cudaGraphLaunch` itself costs **438 us/step** of host time. So CUDA API time accounts for well under 1 ms of the 2.7 ms.
3. **NEXT LEVER (c1): engine-side per-step HOST bookkeeping, which needs a CPU-side profile, not a CUDA one.** With kernels at ~31.5 ms, CUDA API at <1 ms and the graph replay at 0.44 ms, the balance is host CPU work in the runner/scheduler/sampler/detokenizer path between steps. Async scheduling (`max_concurrent_batches=2`) is already on, so the question is why it is not fully overlapping — mirroring vLLM's `gpu_model_runner` step pipeline. This is the [[profile-full-step-not-just-kernels]] playbook applied to the HOST side.
4. **c2 TTFT (0.91x) and c2 output throughput (0.96x) are UNTOUCHED by this work and remain W6 lever (ii)** — the non-MoE prefill glue (attention + dense projections + per-step glue). A decode graph cannot help a prefill-dominated cell, and our c2 TTFT is unchanged to within 0.1%. Note the denominator caveat above.

*Gates.* dgx CUDA `-Werror` **0-warn** (clean full rebuild from scratch); `test_qwen3coder_paged_engine` **6/6 (138 assertions)** with the graph DEFAULT-ON, ZERO token movement; behavior-preservation check `VT_QWEN3MOE_CUDAGRAPH=0` also **6/6**; `test_ops_moe_grouped_bf16` **7/7 (19 assertions)**; NVFP4 `test_ops_moe_grouped` **9/9**; `test_qwen3_moe_forward` **3/3**; `test_decode_graph_sizes` green; Qwen3-dense `test_qwen3_paged_engine` green (the shared `dense_attn_block.h` row-index fix is contents-identical); regression **27B 235/235 + 35B 315/315 UNCHANGED**; `compute-sanitizer memcheck` **0 errors** on the runner decode path (`test_runner` 14/14), the engine path (`test_llm_engine` 5/5) AND the full Qwen3-Coder decode-graph engine gate (`test_qwen3coder_paged_engine` 6/6 under memcheck).

*Remaining to close the row to `DONE`:* (i) the **engine-side per-step host bookkeeping at c1** (CPU-side profile; the last ~2.7 ms/step, which is the whole c1 deficit), (ii) the **non-MoE prefill glue at c2** (TTFT + the throughput cell that follows it), (iii) optionally the dual-operand w13 fusion.

### Risks/decisions

**Decision — BF16 unquantized.** Coder's `quantization_config` is absent (`torch_dtype: bfloat16`), so no quant loader/kernel coverage is needed; correctness is covered by the existing bf16 `MoeBlock` reference path. Simpler than the 35B's `W4A16_NVFP4` Marlin experts.

**Decision — compose, don't rebuild.** The runner is already model-shape-agnostic (`ENG-RUNNER-MODELSHAPE`) and `MoeBlock` takes no GDN metadata, so Coder reuses the done dense attention + the done 35B MoE experts with ZERO runner change. The 35B hybrid assumptions live in `Qwen3_5Model::Forward`, which Coder does not call.

**Decision — near-tie-robust gate** ([[near-tie-distributional-gate]]): correctness = our token within 0.5 nats of vLLM's teacher-forced argmax, strict where equal (the Qwen3-established bar). A3B decode is fast → cheap capture.

**Risk (headline) — the SPEED gap is a real new kernel.** There is no fast BF16 grouped-MoE GEMM (only NVFP4-Marlin), and the 35B decode CUDA-graph is fp4-gated → a bf16 Coder gets no decode graph. W5 (port a bf16 grouped-MoE GEMM, e.g. vLLM fused_moe, + review the eager-decode host tax) is the largest item and the crux of the every-axis speed bar. Correctness can land (W4) before speed (W5) as separate rows.

**Risk — the W1 refactors must be byte-identical.** #1 extract (dense `AttnBlock` → shared header) must keep Qwen3-dense 0.6B/4B byte-identical; #2 expose (bf16 `MoeBlock`) is a pure add (35B never calls it); #3 the no-shared-expert guard must be inert for the 35B/27B (shared>0, fp4 path). Verified via unchanged token-exact gates on all existing models.

**Risk — bf16 MoE reference-path perf.** The bf16 correctness path is a per-expert host-gather loop; it is correct but slow — acceptable for W4 (correctness) but is exactly what W5 must replace.
