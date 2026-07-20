# SPIKE: First Additive-Model Bring-Up — Qwen3 dense (`Qwen3ForCausalLM`)

*Roadmap order-2 / `ROAD-V1-C2`. Row `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (model-matrix, `INVENTORIED` → `SPIKE`). READ-ONLY design — no code in this spike.*
*Grounding pins: vLLM `/home/mudler/_git/vllm` @ `e24d1b24`; our tree `/home/mudler/_git/vllm.cpp`. Standards: the PR-#4 additivity test ([[extensibility-first-additive-hw-models]]), mirror-vLLM.*

---

## 0. Purpose

The first additive-model bring-up: add a NEW architecture as MOSTLY NEW FILES, reusing every delivered seam (Platform, attn-registry, model self-registration, residency, the fusion catalog W0-W4), touching as FEW shared files as possible. It validates the extensibility foundation AND grows model breadth. Its second deliverable is the **honest additivity ledger** (§3) — every shared-file touch classified as the designed seam or a gap to close.

---

## 1. Target selection — honest tractability

dgx checkpoint inventory (`~/.cache/huggingface/hub`), standard-dense candidates:

| Checkpoint | Arch | Quant | Runnable oracle? | Verdict |
|---|---|---|---|---|
| `Qwen--Qwen3-0.6B` | `Qwen3ForCausalLM` | BF16 | YES (vllm 0.25.0 `qwen3.py`) | **PRIMARY first target** |
| `RedHatAI--Qwen3-32B-NVFP4` | Qwen3 dense | NVFP4 | YES | quant follow-on (W-next) |
| `RedHatAI--Qwen3-32B-NVFP4A16` | Qwen3 dense | NVFP4A16 | YES | quant follow-on |
| `facebook--opt-125m` | `OPTForCausalLM` | bf16 | YES | different arch, not chosen |
| **No Llama / no Mistral checkpoint** | — | — | — | roadmap-first Llama needs a **download** |

Oracle: `~/venvs/vllm-oracle` = vLLM 0.25.0 (live). Porting pin is `e24d1b24` (order-9 advance pending). `Qwen3ForCausalLM` is in both the pin (`qwen3.py`, 339 lines) and 0.25.0 — a runnable token-exact oracle exists today.

**Decision — Qwen3 dense (`Qwen3ForCausalLM`), gate on `Qwen3-0.6B` (BF16):**
- The roadmap lists Llama dense first, but **no Llama/Mistral checkpoint is on dgx** → Llama-first is not runnable today without a gated-repo download. Qwen3 dense is checkpoint-present + oracle-runnable now, is a pure standard dense transformer (NO GDN, NO MoE, `sliding_window: null`), and is the closest neighbour to our template — the cleanest, lowest-risk first additivity test.
- `Qwen3-0.6B`: hidden 1024, 28 layers, 16 q / 8 kv heads, head_dim 128, `tie_word_embeddings: true`, BF16 → fast CPU+GPU oracle gate, no quant complexity on the first pass.
- **Honest caveat:** because Qwen3 dense is so close to Qwen3.6 dense, it under-stresses weight-name/config divergence. Recommend **Llama-3.2-1B as the immediate W-next** (needs a download): its distinct traits — no qk-norm, optional attention_bias, `LlamaMLP` naming, non-gemma RMSNorm — make it the stronger cross-family additivity test. Sequence: Qwen3-dense (prove the seam, runnable now) → Llama dense (genuine cross-family) → Mistral.
- NVFP4 32B is the quant-path follow-on reusing our compressed-tensors NVFP4 loader (`qwen3_5_dense_weights.cpp:LoadCtNvfp4Raw`) — confirm its `architectures` field before claiming.

---

## 2. Upstream grounding — `vllm/model_executor/models/qwen3.py @ e24d1b24`

- **`Qwen3Attention` (`:65-168`):** `qkv_proj` (`QKVParallelLinear`, `bias=qkv_bias` default false) → split `[q_size,kv_size,kv_size]`; **per-head `q_norm`/`k_norm` = `RMSNorm(head_dim, eps=rms_norm_eps)`** applied before RoPE (`:148-164`); `get_rope(head_dim, rope_parameters)` (`:121`); full causal `Attention` (`:132`, DECODER); `o_proj` (`RowParallelLinear`, no bias). `scaling = head_dim**-0.5`.
- **`Qwen3MLP` = `Qwen2MLP` (`:58`):** SwiGLU — merged `gate_up_proj` → `SiluAndMul` → `down_proj`.
- **`Qwen3DecoderLayer` (`:171-242`):** `input_layernorm`/`post_attention_layernorm` = **standard `RMSNorm`** (HF `Qwen3RMSNorm.forward = weight * hidden`, i.e. **`gemma=false`**, weight `w` not `1+w`), applied as the fused residual-add form `layernorm(hidden, residual)` (`:233,240`). `set_default_rope_theta(default=1000000)` (`:181`).
- **`Qwen3Model = Qwen2Model` (`:260`)**, `Qwen3ForCausalLM` (`:267-339`): `embed_tokens` → N layers → `norm`; `lm_head` **tied to `embed_tokens` when `tie_word_embeddings`** (`:294-295`) else `ParallelLMHead`. `packed_modules_mapping`: `qkv_proj←[q,k,v]_proj`, `gate_up_proj←[gate,up]_proj` (`:271-274`).
- **Config (`Qwen3-0.6B/config.json`):** `hidden_size 1024, num_hidden_layers 28, num_attention_heads 16, num_key_value_heads 8, head_dim 128, intermediate_size 3072, rms_norm_eps 1e-6, rope_theta 1e6, vocab_size 151936, tie_word_embeddings true, attention_bias false, sliding_window null`. No `layer_types` field (not hybrid).

**1:1 map (template = `qwen3_5_dense`):** Qwen3 dense is a strict subset of our Qwen3.6-dense full-attention path minus GDN, minus the multimodal `model.language_model.` prefix wrapper, minus NVFP4 (0.6B is BF16), minus the attention-output gate, plus tied embeddings, using standard (non-gemma) RMSNorm.

---

## 3. The additivity ledger (primary deliverable)

### (a) NEW FILES a clean bring-up ADDS
| New file | Purpose | Reuses |
|---|---|---|
| `include/vllm/model_executor/models/qwen3.h` | `Qwen3DenseWeights` + `Qwen3Model::Forward/ForwardDevice` decls | mirrors `qwen3_5_dense.h` |
| `src/vllm/model_executor/models/qwen3_weights.cpp` | safetensors loader for the Qwen3 name map + tied `lm_head` | reuses `LoadBf16Direct`/`LoadBf16Transposed`/`LoadMergedBf16RawNK` (made shared) |
| `src/vllm/model_executor/models/qwen3.cpp` | dense forward, composed from `vt::` ops + fusion catalog; NO GDN, one KV group | vt ops + recipes (§4) |
| `src/vllm/model_executor/models/qwen3_dense.cpp` | registry TU: `Qwen3DenseLoadedModel` + factory + `MakeQwen3ForCausalLMKVCache` (full-attn only, no MambaSpec) + `REGISTER_VLLM_MODEL(qwen3, "Qwen3ForCausalLM", …)` | mirrors `qwen3_5_dense.cpp` |
| `tests/vllm/models/test_qwen3_forward.cpp` + `tests/parity/test_qwen3_paged_engine.cpp` | forward doctest + token-exact greedy vs oracle | mirrors the qwen27 tests |

### (b) SHARED FILES it must TOUCH — classified
| Shared touch | Classification | Notes |
|---|---|---|
| `REGISTER_VLLM_MODEL(...)` line | **DESIGNED SEAM** (item 5) | lives INSIDE the new TU → zero shared-array edit. The ~1 registration PR-#4 allows. |
| `CMakeLists.txt:170-178` — add 4 new `src/` TUs | **UNAVOIDABLE build-glue** | C++ has no module auto-discovery. (Future: a `models/*.cpp` glob removes even this.) |
| **`runner.cpp:458-460`** — `config_.layer_types[l] == "linear_attention"` in the KV-buffer loop | **★ GENUINE SEAM GAP #1** | `layer_types` is empty for pure-dense Qwen3 → out-of-bounds. The runner hardcodes the Qwen3.6 hybrid KV topology. Fix: treat "no mamba group / absent layer_types" as all-full-attention. One-time, model-agnostic. |
| **`runner.cpp:645-730`** — unconditional GDN metadata build / `remap_gdn_state_slots` / gdn block-table gather each step | **★ GENUINE SEAM GAP #2** | Mostly guarded (`remap` early-returns on `gdn_cols<=0`) but metadata build + `gdn_state_` wiring assume the hybrid shape. Same generalizing fix: gate the whole GDN path on "model has a linear-attention group." |
| loader helpers (`LoadBf16Direct`/`LoadBf16Transposed`/`LoadMergedBf16RawNK`) in an anon namespace in `qwen3_5_dense_weights.cpp` | **minor SEAM GAP #3** | Extract to a shared header (`dense_weight_loaders.h`) to reuse — additive-friendly. |
| `model_registry.h:228-235` synthetic `Make*/Borrow*` decls | **avoidable** | Only if the test constructs in-memory weights. If the parity test LOADS `Qwen3-0.6B` from disk (recommended), no header edit — resolution is `ModelRegistry::Load(config, source)`. |
| `hf_config.cpp` config parse | **likely NONE** | `HfConfig` already carries every consumed Qwen3 field. `ParseQwen3ForCausalLMConfig` is a no-op hook like `ParseQwen3_5Config`. |

### Honest additivity count
- **New files: 6** (5 code + 1-2 tests) — the bulk, as designed.
- **Shared touches: ~2 designed + ~3 gaps.** Designed/unavoidable: the in-TU `REGISTER` line, the `CMakeLists` build-list. Genuine seam gaps → follow-up extensibility fixes: **runner #1 (layer_types indexing), #2 (GDN metadata/step path), #3 (loader-helper extraction).**
- **Verdict vs PR-#4:** the model-registry/factory seam is CLEAN (new TU + 1 REGISTER, zero shared-array edit — item 5 delivered). The **runner is NOT yet model-shape-agnostic** — it has only ever executed the Qwen3.6 hybrid, so the first pure-dense model is the forcing function that exposes runner gaps #1/#2. These are small, one-time, generalizing edits (make the runner treat a full-attention-only KV config as such), NOT per-model scatter. After them, every subsequent dense arch (Llama, Mistral) adds with new-files-only, zero further runner edits. **This runner generalization IS the extensibility deliverable this bring-up forces.**

---

## 4. Fusion-catalog reuse

The `FusedRecipe` POD already carries a `gemma` flag (`fused_recipe.h:127`) and `kRmsNorm` is Tier-1-able, so Qwen3's standard (non-gemma) RMSNorm needs no new primitive — only new *declarations*:

| Qwen3 forward step | Existing recipe | Reuse / new |
|---|---|---|
| residual-add + input/post RMSNorm | `kFusedAddRmsNorm` (gemma=true) | **NEW 1-line recipe** `kFusedAddRmsNormStd` (gemma=false) — the W3 one-declaration pattern |
| qk-norm + RoPE preamble | `kAttnQkNormRopeGate` (gemma + gate) | **NEW variant** `kAttnQkNormRope` (gemma=false, NO gate — Qwen3 has no attention gate), or compose `RmsNorm(q,gemma=false)+RmsNorm(k)+RopeFromCache` from standalone ops |
| SwiGLU MLP (BF16) | `kSiluMulFp4Quant`/`kSiluMulQuantFp8` are quant-tailed | BF16 0.6B uses plain `vt::MoeSiluMul` directly; NVFP4 32B follow-on reuses `kSiluMulFp4Quant` as-is |
| final norm, matmuls, FA2 attention | `vt::RmsNorm`/`vt::MatmulBT`/`vt::MatmulNvfp4`/FA2 | direct reuse, no new decl |

Net new catalog work: **≈2 one-line `constexpr FusedRecipe` declarations**, each with its byte-exact composite test — textbook additive per W3/W4. **Numerics checkpoint:** confirm our Qwen3.6/`qwen3_5` norm gemma-ness; Qwen3 (original) is confirmed standard (HF `Qwen3RMSNorm = weight*hidden`).

---

## 5. Tests to port + the correctness GATE

- **Upstream (the spec):** vLLM has no dedicated `qwen3.py` unit test; coverage is via `tests/models/test_initialization.py`, `tests/models/registry.py` (`Qwen3ForCausalLM` HF_EXAMPLE), and the generic greedy-logprob harness. Port: (a) a registry/initialization case asserting `Qwen3ForCausalLM` resolves to our factory (extend `test_model_registry.cpp`); (b) a forward doctest `test_qwen3_forward.cpp` mirroring `test_qwen27_dense_forward.cpp`; (c) a greedy token-exact parity `test_qwen3_paged_engine.cpp` mirroring `test_qwen27_paged_engine.cpp`. Not-yet-passing cases checked in SKIPPED with a tracked reason (test-porting.md).
- **THE SACRED GATE:** token-exact greedy decode of `Qwen3-0.6B` (BF16), our engine vs `~/venvs/vllm-oracle` (vLLM 0.25.0), identical prompt set, 16/16 token-for-token (extend to the standard battery). Precondition for any perf claim. 0.6B fits CPU → a CPU-tier forward sanity check runs in CI ahead of the GPU token-exact gate.

---

## 6. HW / checkpoint needs, dependencies, work breakdown

**HW/checkpoint:** `Qwen3-0.6B` present on dgx (BF16, ~1.5 GB) — trivially fits GB10 and CPU. Oracle present (0.25.0). No download, no multi-GPU. NVFP4 32B present for the quant follow-on.

**Dependencies:** builds on the delivered seams (model self-registration item 5, Platform/residency items 1-2, attn-registry item 4, fusion catalog W0-W4). The only *new* dependency work is the runner generalization (gaps #1/#2).

**Work breakdown (incremental, each row byte/token-gated):**
- **W0 — config + registry stub.** Confirm `HfConfig` parses Qwen3 fields; add `qwen3_dense.cpp` with `REGISTER_VLLM_MODEL("Qwen3ForCausalLM")` + a stub factory + full-attention-only `MakeQwen3ForCausalLMKVCache` (one KV group, no MambaSpec); extend `test_model_registry` Resolve case. Gate: CPU build + registry test green.
- **W1 — runner generalization (SEAM-GAP FIX #1/#2).** Make `runner.cpp` KV-alloc loop (`:458`) and forward GDN path (`:645-730`) treat an absent GDN group / empty `layer_types` as all-full-attention. Gate: BEHAVIOUR-PRESERVING — Qwen3.6 27B 235/235 + 35B 315/315 token-exact UNCHANGED on dgx, plus a full-attention-only smoke. *(the extensibility deliverable this bring-up forces)*
- **W2 — weight loader.** Extract shared BF16 loader helpers to a header; `qwen3_weights.cpp` with the Qwen3 name map + tied lm_head + the separate-vs-merged qkv choice. Gate: load `Qwen3-0.6B`, shapes/counts assert, no unmapped/leftover tensors.
- **W3 — forward.** `qwen3.cpp` composing vt ops + the 2 new catalog recipes; wire the dense decode graph. Gate: `test_qwen3_forward` vs a captured reference-activation golden; composite==golden byte-exact for the new recipes.
- **W4 — correctness gate (SACRED).** Token-exact greedy `Qwen3-0.6B` vs vLLM 0.25.0 oracle, 16/16; SKIPPED→passing. Update model-matrix row → `DONE` with code+test+ledger anchors, README model table, roadmap `ROAD-V1-C2`.
- **W-next (separate rows):** NVFP4 `Qwen3-32B-NVFP4` (reuse `LoadCtNvfp4Raw`); **Llama-3.2-1B** (download) as the genuine cross-family additivity proof — should touch new files + 1 REGISTER + CMake only if W1 fully generalized the runner.

---

## 7. Model-matrix row + roadmap wiring

- **`model-matrix.md`** row `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (`Qwen3ForCausalLM`; `registry.py:191`; `qwen3.py::Qwen3ForCausalLM`): `INVENTORIED → SPIKE` on acceptance, owner in `coordination.md`, then `READY`/`ACTIVE` per the W-breakdown. Enabling dependency: `MODEL-FACTORY-registry` (item 5).
- **`roadmap_v1.md` order-2 `ROAD-V1-C2`:** satisfies the Qwen3 arm now (runnable today); reframe the sequence honestly as Qwen3-dense-first (checkpoint-present) → Llama-dense (download) → Mistral (dgx has no Llama/Mistral checkpoint). The runner-generalization (W1) is the real extensibility payload — track it as a first-class engine deliverable since it unblocks every future dense/non-hybrid arch.
- **`[[parity-enablers-ship-as-defaults]]`:** none of this measures perf → no default-flip; the gate is purely token-exact correctness.

---

## 8. Key findings

1. **Tractable first target = Qwen3 dense on `Qwen3-0.6B` (BF16)** — the only standard-dense arch with both a checkpoint on dgx and a runnable 0.25.0 oracle today. Llama needs a download → W-next.
2. **The model-registry/factory seam is CLEAN** (new TU + 1 in-TU REGISTER, zero shared-array edit — item 5 delivered).
3. **The honest additivity gaps are in the RUNNER, not the registry:** `runner.cpp:458-460` (layer_types indexing crashes on pure-dense) and `:645-730` (unconditional GDN metadata/step path) hardcode the Qwen3.6 hybrid topology. The first pure-dense bring-up is the forcing function that must generalize them — small, one-time, model-agnostic — after which Llama/Mistral add with new-files-only.
4. **Fusion catalog reuses everything**; net-new = ≈2 one-line recipe declarations (standard/non-gemma add-rmsnorm + non-gated qk-norm-rope); the `gemma` flag already exists.
5. **Sacred gate:** token-exact greedy `Qwen3-0.6B` vs vLLM 0.25.0 oracle 16/16; regression gate = 27B 235/235 + 35B 315/315 unchanged after the runner generalization.

## 9. Structured spike contract (stable rows `MODEL-TEXT-qwen3-qwen3-for-causal-lm` + `ENG-RUNNER-MODELSHAPE`)

The prose above (§0–§8) is the full spike; this section restates it in the record-checker's structured fields. Two stable rows share this spec: the model row `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (the arch bring-up) and the engine row `ENG-RUNNER-MODELSHAPE` (the runner generalization the bring-up forces — a first-class extensibility deliverable).

### Scope
Add `Qwen3ForCausalLM` (pure standard-dense transformer: no GDN, no MoE, standard non-gemma RMSNorm, per-head q/k norm, tied lm_head, `sliding_window: null`) as MOSTLY NEW FILES, and generalize the `GPUModelRunner` so a full-attention-only KV config is first-class. In scope: config hook, registry TU, full-attention-only KV spec, runner generalization, weight loader, dense forward, token-exact gate. Out of scope: vision tower, NVFP4 (32B follow-on), Llama/Mistral (W-next).

### Upstream chain
`vllm/model_executor/models/qwen3.py::Qwen3ForCausalLM` @ `e24d1b24` (§2: `Qwen3Attention` qkv+per-head q/k RMSNorm+RoPE+`o_proj`; `Qwen3MLP`=`Qwen2MLP` SwiGLU; standard `RMSNorm`; tied `lm_head`); `registry.py:191` (`REGISTER_MODEL`); config `Qwen3-0.6B/config.json`. Runner ground: `vllm/v1/worker/gpu/model_runner.py` `initialize_kv_cache` drives layer typing off `kv_cache_config.kv_cache_groups` (never a hardcoded hybrid) — the model-agnostic pattern `ENG-RUNNER-MODELSHAPE` restores.

### Our baseline
Template = the Qwen3.6 dense path: `qwen3_5_dense.cpp` (registry TU idiom), `qwen3_5_common.*` (config hook + KV spec), `model_registry.h` `REGISTER_VLLM_MODEL`. Runner = `src/vllm/v1/worker/gpu/runner.cpp` (the KV-buffer alloc loop + the forward-step GDN path). Qwen3 dense is a strict subset of our 27B dense full-attention path minus GDN, minus the multimodal prefix, minus NVFP4, minus the attention gate, plus tied embeddings and non-gemma RMSNorm.

### Port map
New files: `include/vllm/model_executor/models/qwen3.h`, `src/vllm/model_executor/models/qwen3_dense.cpp` (registry TU: `REGISTER_VLLM_MODEL(qwen3,"Qwen3ForCausalLM")` + `MakeQwen3ForCausalLMKVCache` full-attn-only + `ParseQwen3ForCausalLMConfig` + stub factory), then W2 `qwen3_weights.cpp` + W3 `qwen3.cpp`. Shared touches (classified §3b): `CMakeLists.txt` (TU add, unavoidable build-glue), the in-TU REGISTER line (designed seam), and THE ONE generalization `runner.cpp` (`ENG-RUNNER-MODELSHAPE`, seam gaps #1 alloc-loop `layer_types` index, #2 GDN metadata/step) driven off a model-agnostic `has_mamba_group`/`gdn_group_id_>=0` predicate.

### Tests to port
(a) registry resolution `tests/vllm/models/test_model_registry.cpp` (`Qwen3ForCausalLM` resolves to the dense factory; FA-only KV spec); (b) runner generalization `tests/vllm/v1/worker/test_runner.cpp` (full-attention-only KV alloc + step, RED→GREEN vs the pre-generalization crash); (c) W3 forward doctest `test_qwen3_forward.cpp`; (d) W4 SACRED token-exact `test_qwen3_paged_engine.cpp` vs the vLLM 0.25.0 oracle. Mirrors the qwen27 test family.

### Gates
W0: CPU build + registry test green. W1: dgx CUDA `-Werror` 0-warn; BEHAVIOUR-PRESERVING 27B `test_qwen27_paged_engine` 235/235 + 35B `test_qwen36_paged_engine` 315/315 token-exact UNCHANGED; new CPU runner tests RED→GREEN; memcheck 0 on the affected paths. W4 SACRED: token-exact greedy `Qwen3-0.6B` vs vLLM 0.25.0 oracle 16/16 (precondition for any perf claim).

### Dependencies
Builds on the delivered extensibility seams: model self-registration (item 5), Platform/residency (items 1-2), attn-registry (item 4), the fusion catalog (W0-W4). The only NEW dependency work is the runner generalization (`ENG-RUNNER-MODELSHAPE`, gaps #1/#2). Checkpoint `Qwen3-0.6B` (BF16) + oracle `~/venvs/vllm-oracle` (vLLM 0.25.0) present on dgx; no download, no multi-GPU.

### Work breakdown
W0 config + registry stub (LANDED 2026-07-20). W1 runner generalization / `ENG-RUNNER-MODELSHAPE` (LANDED 2026-07-20, behaviour-preserving). W2 weight loader (LANDED 2026-07-20: shared BF16 helpers extracted to `dense_weight_loaders.h` byte-identical; `qwen3_weights.cpp` Qwen3 name map + merged qkv/gate_up + tied lm_head skipping the checkpoint `lm_head.weight`; dgx load gate 1567/1567, 311 tensors, no leftover, memcheck 0; 27B 235/235 + 35B 315/315 unchanged). W3 forward (compose vt ops + ≈2 new one-line fusion recipes). W4 SACRED correctness gate + row → `DONE`. W-next: NVFP4 32B, then Llama-3.2-1B (download) as the genuine cross-family additivity proof.

### Risks/decisions
Decision: gate on `Qwen3-0.6B` because it is the only standard-dense arch with both a dgx checkpoint and a runnable 0.25.0 oracle today (Llama needs a download). Risk: Qwen3 dense's closeness to Qwen3.6 dense under-stresses weight-name/config divergence → Llama-3.2-1B recommended as the immediate W-next for a genuine cross-family test. Honest additional seam gap found in W1 (beyond #1/#2): the SHARED 27B `Qwen3_5DenseModel::Forward` asserts `gdn_meta.num_actual_tokens == T` (`qwen3_5.cpp:5463`) — a FORWARD-side hybrid assumption, resolved by Qwen3's own W3 forward (not a runner gap). Numerics decision: confirm non-gemma RMSNorm (`weight*hidden`) at W3.
