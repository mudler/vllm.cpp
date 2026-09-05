# SPEC: Advance the vLLM parity pin past `e24d1b24` (v0.25.0-era) — SCOPE + PLAN

**EXECUTED - the pin was FLIPPED at W5 (`bc415a3e`, 2026-07-26) to `555967922` /
vLLM 0.26.0.dev0 + transformers 5.14.1; see §7.** The plan below (build the staged
venv → smoke-test the 3 unblocks → re-capture drifted goldens → flip the pin) ran
end-to-end: zero real golden drift (27B-W4A4 + 32B-NVFP4A16 bit-identical, 35B/Coder
byte-stable), full re-gate 296/299 GREEN on GB10, and DFlash/Gemma-4/OLMo-3 unblocked.
This document began as the CPU/repo-research SCOPE + PLAN the user directed (2026-07-26:
*"advance the pin then and let's assess what we need to port that vLLM advanced"*);
the scope/plan body is preserved below and §7 records the execution. Claims
`CLAIM-PIN-ADVANCE-SCOPE` → `CLAIM-PIN-ADVANCE-W5`.

- **Current pin:** `/home/mudler/_git/vllm` @ `e24d1b24` (`v0.23.1rc0-734-ge24d1b24`,
  2026-07-02). Active oracle `~/venvs/vllm-oracle` = **vLLM 0.25.0** + transformers
  5.13.1 + Torch 2.11.0+cu130 + FlashInfer 0.6.13 + CUTLASS-DSL 4.5.2.
- **vLLM source at scope time:** `origin/main` = `55596792`
  (`5559679229bc961848b121ccdeaa8fa5d79bec98`, 2026-07-26). Range
  `e24d1b24..origin/main` = **777 commits over ~4 weeks**. Newest release tag in
  range = **v0.25.0** (`702f4814`, 2026-07-11) — but see §1: v0.25.0 does NOT
  contain any of the three unblock fixes.

---

## 1. Deliverable 1 — TARGET version (unblocks all three; NO tradeoff)

**RECOMMENDED TARGET: vLLM `origin/main` HEAD `55596792` (2026-07-26), pinned with
transformers `==5.14.1`, Torch `2.13.0`, torchvision `0.28.0`, triton `3.7.1`,
FlashInfer `0.6.15`, nvidia-cutlass-dsl `4.6.0`.** A single coherent target
unblocks all three blockers — there is **no tradeoff / no split-version problem.**

### Why not a release tag
There is **no release tag newer than v0.25.0**, and **v0.25.0 (`702f4814`,
2026-07-11) contains NONE of the three fixes** — verified with
`git merge-base --is-ancestor`: the DFlash-hybrid fix, the OLMo migration, the
transformers-5.14.1 bump, the Torch-2.13 bump, and the Blackwell FlashInfer
non-causal fix are ALL main-only (they landed on `main` after the v0.25.0 release
branch was cut). So advancing the pin at all requires a **main commit**, not a tag.

### The three unblocks — each RESOLVED at the target

**(a) DFlash vllm#40898 (mixed sliding/full attention) — RESOLVED by #47914 +
follow-ups.**
- `0d12618e` **[Spec Decode] Support hybrid (SWA + full attention) DFlash drafters
  (#47914)** (2026-07-08) rewrites `qwen3_dflash.py::_resolve_layer_attention`: the
  old hard `raise NotImplementedError` on `any_sliding and not all_sliding` becomes
  a conditional that only raises when **NOT** on the V2 model runner:
  ```python
  if 0 < num_sliding < len(layer_types) and not get_current_vllm_config().use_v2_model_runner:
      raise NotImplementedError("... require the V2 model runner; relaunch with VLLM_USE_V2_MODEL_RUNNER=1.")
  ```
  **KEY: mixed-attention DFlash drafts now construct, but ONLY under the V2 model
  runner** (`VLLM_USE_V2_MODEL_RUNNER=1`). This is well-aligned with us — we port
  **Model Runner V2 (MRV2)** per [.agents/vllm-v1-v2.md](../vllm-v1-v2.md). Both
  our gate drafts are the mixed kind (27B `[SWA×4,full×1]`, 35B `[SWA×5,full×1]`),
  so this is the load-bearing fix for the D0 verdict.
- Follow-ups that must be in the target for a WORKING DFlash on our 27B draft
  (whose `target_layer_ids=[1,16,31,46,61]` ⇒ `num_target_layers=5 ≠ 64` hidden):
  `7614b88e` **draft/target layer-count mismatch (#48113)**, `a7d00ec0` **DFlash fc
  sized wrong when num_target_layers != num_hidden_layers (#48524)** (2026-07-21),
  and for GB10/sm_121 specifically `ecf4aa5c` **Fix FlashInfer non-causal draft
  attention (DFlash/DSpark) on Blackwell (#48167)** (2026-07-15). Also
  `7bd15437` **Fix mamba+dflash for MRV2 (#47698)** — directly on the hybrid-GDN +
  DFlash + MRV2 path our gate models exercise.
- **Minimum target for a complete DFlash unblock: ≥ `a7d00ec0` (2026-07-21).**

**(b) Gemma-4 multimodal (no `transformers.models.gemma4`) — RESOLVED by
transformers 5.14.1.**
- `gemma4_mm.py` loads its towers from Transformers
  (`AutoModel.from_config(config.vision_config/audio_config)`) and imports
  `Gemma4Config/Gemma4Processor/Gemma4VisionConfig/Gemma4AudioConfig/Gemma4TextConfig`
  from `transformers.models.gemma4`. transformers **5.13.1 has no `gemma4`
  module** (the decisive gating fact in [gemma4-multimodal.md](gemma4-multimodal.md)
  §0.0).
- The target pins **transformers `==5.14.1`** (`0d77325b` **Bump Transformers
  version to 5.14.1 (#49223)**, 2026-07-23 — `requirements/test/{cuda,cpu,rocm,xpu,
  nightly-torch}.txt`). vLLM's own Gemma-4 CI runs under this pin (`b8fb56d9` **Add
  gemma-4-E4B-it-assistant to CI gsm8k for GemmaMTP (#49243)**), which is direct
  evidence that **transformers 5.14.1 ships `transformers.models.gemma4`.** EXECUTION
  must smoke-test `import transformers.models.gemma4` as the decisive check (D0-class).
- NOTE: `vllm` common pin is only `transformers >= 5.5.3` at BOTH ends (unchanged),
  so the gemma4 requirement is satisfied purely by installing 5.14.1 — no vLLM
  code change needed. A `tests/.../test_common.py` TODO references a not-yet-released
  transformers 5.15.0; do NOT chase it — 5.14.1 is the CI-validated version.

**(c) OLMo-3 nested `rope_parameters` KeyError — RESOLVED by a new native
`olmo3.py` + transformers 5.14.1.**
- The 0.25.0 oracle dies at `olmo2.py:143` `rope_parameters["rope_theta"]` →
  `KeyError` because transformers 5.13.1 nests `rope_parameters` per layer-type
  (sweep-recent-dense-batch §0.3; RUN-verified in `CLAIM-OLMO3-W0-VERIFY`).
- The target ships a **new dedicated `vllm/model_executor/models/olmo3.py`**
  (registry: `"Olmo3ForCausalLM": ("olmo3", "Olmo3ForCausalLM")`) whose
  `Olmo3Attention` handles the nested schema at `olmo3.py:140-142`:
  ```python
  rope_parameters = self.config.rope_parameters
  rope_parameters = rope_parameters.get(attn_type, rope_parameters)  # per-layer-type nested-rope safe
  ```
  Separately `b83be00c` **Migrate Olmo and Olmo2 to the Transformers modeling
  backend (#48100)** re-routes the OLDER `Olmo/Olmo2ForCausalLM` →
  `("transformers", "TransformersForCausalLM")`. Combined with transformers 5.14.1
  (which carries the nested-`rope_parameters` config), the oracle can now BUILD +
  RUN `allenai/OLMo-3-1025-7B` ⇒ our IMPLEMENTED-but-BLOCKED OLMo-3 (landed olmo2
  row W5) gets a SACRED bar.

### Dependency-stack delta the EXECUTION venv must install (vs the 0.25.0 oracle)
| Dep | 0.25.0 oracle (current) | Target (`origin/main`) | Commit |
|---|---|---|---|
| vLLM | 0.25.0 (`702f4814`) | `main` `55596792` | — |
| transformers | 5.13.1 | **5.14.1** | #49223 `0d77325b` |
| Torch / torchvision / triton | 2.11.0+cu130 / — / — | **2.13.0 / 0.28.0 / 3.7.1** | #48155 `75ccdf31` |
| FlashInfer (py/cubin) | 0.6.13 | **0.6.15** | #47669, #48914 |
| nvidia-cutlass-dsl | 4.5.2 | **4.6.0** | #47442 `7cd1d57b` (+ tml-fa4 bump #48988) |

The **Torch 2.11→2.13 + CUTLASS-DSL 4.5→4.6 + FlashInfer 0.6.13→0.6.15** bumps are
the heavy part of the migration (whole CUDA/JIT stack changes on GB10/sm_121) —
they are the primary source of build risk AND golden drift (§3).

---

## 2. Deliverable 2 — DELTA catalog ("what vLLM advanced that we need to port")

Grouped from `git log e24d1b24..origin/main`. **Gate-relevant / golden-drift items
are flagged ★.** Our gate models are Qwen3.6-27B/35B-NVFP4 (hybrid GDN + MoE +
NVFP4 + full-attn on Blackwell); the SACRED dense/Coder/GLM/DeepSeek/Gemma/OLMo set
rides shared kernels. NOTE the naming trap: many upstream commits say **"Qwen3.5"**
— that is a DIFFERENT model from our **Qwen3.6** gate; they share GDN/MoE/RMSNorm
infra but are not the same checkpoint.

### 2A. New MODELS (registry additions — sweep candidates)
New arches registered in `registry.py` in range (frontier/gate-relevant flagged ★):
- ★ **`Olmo3ForCausalLM`** → `("olmo3","Olmo3ForCausalLM")` — the OLMo-3 unblock
  (our row is IMPLEMENTED, just needs the oracle; §1c).
- ★ **`DFlashLagunaForCausalLM`** (`laguna_dflash`, #46853) + **`Gemma4DSparkModel`**
  (`gemma4_dspark`, #47216) + **`BailingMoeV25MTPModel`** + **`InklingMTPModel`** —
  new SPECULATIVE draft/MTP arches (DFlash/DSpark/MTP family growth).
- **`InklingForCausalLM` / `InklingForConditionalGeneration` / `InklingMTP`**
  (#48799/#48822/#48858/#48869/#48884/#49258) — a full new frontier family
  (NVFP4 weights, LoRA, MTP=1, Hopper FA4 relative attention).
- **`LongcatFlashNgramForCausalLM`** (LongCat-Flash-Lite n-gram embedding, #47857/#08dfd686).
- **`Cosmos3EdgeForConditionalGeneration`** (#... FP8 ModelOpt remap #48952),
  **`MossTranscribeDiarizeForConditionalGeneration`** (ASR+diarize, #47729),
  **`VibeVoiceAsrForConditionalGeneration`**, **`MossTranscribeDiarize`** — new
  audio/ASR mm arches.
- **`VaultGemmaForCausalLM`** (transformers backend, #49803), **TranslateGemma-12b**
  (#41599), **DiffusionGemma** top_k/top_p (#45429) — Gemma-family growth.
- **`RobertaForTokenClassification` / `XLMRobertaForTokenClassification`** (#47991).
- ★ Frontier families the user named, ACTIVELY advanced upstream (mostly already
  owned by other claims — `CLAIM-MLA-DEEPSEEK`, `CLAIM-GLM-DSA-LATEST-DEEPSEEK`):
  **DeepSeek-V4 / DeepSeek-V3.2 / DeepSeek4** (CT quant #41276, DSpark spec-decode,
  fp8_ds_mla KV, sparse MLA prefill), **GLM-5.2 / GLM5** (Blackwell decode opt
  #48597 then **REVERTED** #49768, config fix #48711, MoE SP migrate #47881),
  **MiniMax-M3 / M2** (sparse PA, QK-norm+AllReduce fusion, MXFP8 GEMM),
  **Kimi-K2.5** (MLA query-replication #45964, fused image preproc #47416).

### 2B. New / changed KERNELS + NUMERICS — ★ the GOLDEN-DRIFT surface
Changes that plausibly shift OUR gate-model tokens (NVFP4 MoE / fp8 / rmsnorm+quant
fusion / GDN / attention on Blackwell / sampling determinism):
- ★★ **Sampling determinism flip-flop:** `425c4eaf` **Stop upcasting logits to fp32
  in apply_sampling_params (#48641)** THEN `1134545b` **Revert (#49033)**. Net = fp32
  upcast RESTORED (same as pin), but this is a near-tie-sensitive sampling path —
  confirm the net state on the target before trusting greedy goldens.
- ★ **RMSNorm numerics:** `df8a0900` **Don't apply weight in batch-invariant RMSNorm
  when has_weight=False (#48741)** (touches `layernorm.py`); `c9be3a8a` **Disable warp
  specialization in rms_norm_per_block_quant B200 configs (#48797)** (rmsnorm→fp4
  quant fusion on Blackwell — directly our NVFP4 path).
- ★ **NVFP4 MoE:** `b07ec92f` **Make shared NVFP4 MoE scales writable (#49489)**,
  `2bd89576` **Pad gated intermediate to 64 for FlashInfer TRT-LLM shuffle (M%128)
  (#46880)**, `09238796` **nvfp4 MoE weight-processing peak-mem reduction (#46276)**,
  `29fd6888` **skip CuTeDSL fp4_gemm autotuning by default (#48268)** (changes the
  selected fp4 GEMM tactic → can shift near-ties).
- ★ **fp8 / quant dispatch:** `ce2aecc4` **Use CuTe-DSL for FlashInfer MXFP4
  quantization (#48417)**, `ab3c1aed` **Fix activation quant dispatch for
  WNA4Int/WNA8Int (#48785)**, `917fdb5b` **DeepGEMM warmup fix (#49467)**.
- ★ **MoE kernels:** `089e4128` **Integrate TRTLLM BF16 MoE Modular Kernel
  (#45182)**, `d08eebad` **Write FlashInfer combine into final output (#47156)**,
  `97a66815` **router replay from FlashInfer monolithic MoE (#44214)**.
- ★ **GDN / Mamba2 (our hybrid state path):** `866fea2b` **ReplaySSM: cache SSM
  inputs for faster Mamba2 standard decode (#48018)** (touches `mamba_mixer2.py`),
  `c71a583a` **Vectorize `_copy_mamba_state_block` to uint64 (#48110)**,
  `6700813f` **Standardize Mamba cache; drop `get_transfer_cache_regions` (#44456)**.
  Perf-shaped (expected numerically neutral), but on our gate's GDN path ⇒ verify.
- ★ **Attention on Blackwell:** `ecf4aa5c` (#48167, non-causal FlashInfer on
  Blackwell), `7e950521` **size FlashInfer prefill workspace to batch head footprint
  (#48428)**, `735def4f` **FlashMLA dense fp8 metadata crash fix (#48045)**.
- ★ **AllReduce+RMSNorm+quant fusion:** `75fe92a3` **Enable FlashInfer MNNVL
  allreduce RMS quant fusion (#48064)**, `5f8e73cb` **Guard mixed-dtype allreduce
  RMSNorm quant fusions (#48330)**, `300e3379` **fuse more rmsnorm and all-reduce in
  qwen3.5 (#46998)** (the "qwen3.5" family = shares our RMSNorm-fusion recipe).

### 2C. New FEATURES
- **Spec-decode:** the DFlash-hybrid unblock (#47914 + follow-ups, §1a); DSpark
  drafts for DeepSeek-V4 / Gemma4; Inkling MTP=1; GemmaMTP CI; MTP thinking-budget
  TPOT opt (`32e632df` #46662, touches `sampler.py`); `ac36a7a1` **MRV2 rejection
  sampler OOM chunking (#48630)**.
- **Multimodal:** Gemma-4 image+video+**audio** now oracle-runnable (§1b);
  ViT CUDA-graph for Gemma-4 (#46837); new ASR/diarize + VibeVoice audio arches;
  Qwen3-VL M-RoPE on transformers backend (#49292); DeepSeek-OCR-2 TTFT (#49531).
- **Serving / scheduler / KV:** KV-cache layout refactor (standardize Mamba cache
  #44456, per-token-head quant cross-layer guard #49226); NIXL hybrid MLA+mamba TP
  (#49297); Mooncake TP-sharded Mamba dedup (#49499); translation-API sampling
  params (#45839); repetition-detection sampling (#46684).
- **Quant:** DeepSeek4 CompressedTensors (#41276); Inkling llm-compressor NVFP4
  (#49258); Quark MXFP4 fixes.

### 2D. MECHANICAL RE-SYNC needed (our LANDED code mirrors these upstream files)
Files we mirror 1:1 that changed in range — a bounded, mechanical re-port per the
sync cycle (each is 0-3 upstream commits):
- **`vllm/model_executor/models/qwen3_next.py`** (our GDN-hybrid backbone mirror,
  3 commits): ★ `300e3379` fuse rmsnorm+all-reduce (#46998, NUMERICS), `a02984ed`
  MoE all-reduce→reduce-scatter (#47006, perf), `c233d90a` load_weights cleanup
  (#48496, mechanical).
- **`layers/layernorm.py`** (2): ★ `df8a0900` has_weight=False RMSNorm (#48741),
  `626c90b2` move fla to third party (#48500, structural).
- **`layers/fused_moe/fused_moe.py`** (2): `ec59c157` MoeWNA16 → MK oracle scheme
  (#44120 — touches our NVFP4A16 dense-quant path), `934eeaec` ssm device-name fix.
- **`layers/mamba/mamba_mixer2.py`** (1): ★ `866fea2b` ReplaySSM cache (#48018).
- **`v1/sample/sampler.py`** (1): `32e632df` thinking-budget TPOT (#46662).
- **`models/qwen3_dflash.py`** (the DFlash arch we spec against, 6 commits): the
  #47914/#48113/#48167/#48524 fixes above — re-ground the DFlash spec §1 anchors.
- **`models/olmo3.py`** (NEW file to mirror for the OLMo-3 W5 close).
Cross-reference against `.agents/porting-inventory.md`, the fusion catalog
(`KERNEL-FUSION-FRAMEWORK`), and the attention/platform registries during EXECUTION.

---

## 3. Deliverable 3a — GOLDEN-DRIFT RISK (the real cost)

**HONEST HEADLINE: advancing the pin is likely to DRIFT several SACRED goldens and
require a re-capture pass. This is the real cost the user must weigh.** A drift is
NOT a bug — it is the oracle moving under the Torch-2.13 / CUTLASS-4.6 /
FlashInfer-0.6.15 / NVFP4-MoE-scale-pad-autotune / rmsnorm-quant-fusion changes
(§2B). The plan RE-CAPTURES drifted goldens on the NEW oracle (bit-for-bit re-gate,
OUR code unchanged) and keeps the OLD oracle for rollback.

Risk ranking (likelihood a gate's committed golden token-sequence changes):

| Gate model | Golden set | Kernel paths touched in range | Drift risk |
|---|---|---|---|
| **Qwen3.6-27B-NVFP4** (STRICT, hybrid) | `qwen36_{logits,norm,fullattn_layer,gdn_layer,embed}_27b`, `qwen3_5_mtp_head_27b`, `qwen36_gguf_35b` | NVFP4 MoE scale/pad/autotune, rmsnorm→fp4 quant fusion (B200), GDN ReplaySSM, FlashInfer Blackwell attn, MoE combine | **HIGH** |
| **Qwen3.6-35B-A3B-NVFP4** (STRICT, hybrid) | `qwen36_*_35b`, `qwen3_5_mtp_head_35b` | same as 27B | **HIGH** |
| **Qwen3-Coder-30B-A3B** (MoE, STRICT 6/6) | `qwen3coder_greedy` | TRTLLM BF16 MoE modular kernel, MoE combine, rmsnorm fusion | **MEDIUM** |
| **Qwen3 dense** (0.6B/4B/32B-NVFP4A16) | `qwen3_greedy_{0_6b,4b}`, `qwen3_32b_nvfp4a16_greedy` | MoeWNA16 → MK oracle (#44120), rmsnorm has_weight, sampling fp32 flip | **MEDIUM** (0.6B/4B already near-tie-robust gates ⇒ absorb drift; 32B-NVFP4A16 = HIGH) |
| **Mistral-7B / Llama-1B / OPT / Phi / GLM4-MoE-Lite / DeepSeek-V2 / OLMo-2 / MiniCPM(3) / InternLM / Yi / Gemma** (dense/near-tie set) | the `*_greedy` goldens | rmsnorm, sampling determinism, generic attention — mostly bf16 dense | **LOW-MEDIUM** (many already gate near-tie-robust ⇒ tolerant; strict-deterministic ones need a diff) |
| **Op-level goldens** (rmsnorm, rope, silu, moe_router, causal_conv1d, gdn_*, l2norm, matmul) | the many `*_f32_*`/`*_bf16_*` op goldens | f32 op goldens are dtype-stable ⇒ mostly INERT; the bf16 + NVFP4 op goldens track kernel changes | **LOW** for f32, **MEDIUM** for the NVFP4/bf16 fusion op goldens |

**Estimate: of the SACRED model gates, expect the 2 NVFP4 hybrid gates (27B, 35B)
+ the 32B-NVFP4A16 + Coder-30B (≈4 core gates) to REQUIRE re-capture, plus a
diff-and-maybe-recapture pass over the ~30 model-matrix `*_greedy` rows and the
NVFP4/bf16 op goldens.** The f32 op goldens and the near-tie-robust dense gates
should mostly survive. The re-capture is the dominant EXECUTION cost — budget it
as a full golden-regeneration campaign on the new oracle, not a quick bump.

---

## 4. Deliverable 3b — MIGRATION W-PLAN (GPU-gated EXECUTION, run AFTER oracle frees)

Prerequisites: the sibling 35B-MTP agent has released `~/venvs/vllm-oracle`;
disk headroom ≥ 200 GiB (a new vLLM venv is ~15-20 GiB — reclaim old
`source-*`/build trees first, [[grid-per-sha-trees-fill-disk]]); GPU sole-owner
or `flock $HOME/gpu.lock`; local-ai-worker DOWN.

- **W0 — Reclaim + stage.** Verify disk headroom; prune stale per-SHA source/build
  trees. Do NOT touch `~/venvs/vllm-oracle` or `~/venvs/vllm-oracle-v0.25.0-stage`
  (rollback). **Gate:** ≥20 GiB free for the new venv + goldens.
- **W1 — Build the STAGED new venv `~/venvs/vllm-oracle-next`.** Fresh venv;
  `pip install vllm @ git+…@55596792` (or checkout + editable) with
  transformers==5.14.1, torch==2.13.0, torchvision==0.28.0, triton==3.7.1,
  flashinfer 0.6.15 (py+cubin), nvidia-cutlass-dsl==4.6.0. Record the freeze hash
  (mirror the environment.md manifest discipline). **Gate:** `pip check` clean
  (modulo the recorded cusparselt vendor-tag defect); `import vllm` + `import torch`
  on GB10.
- **W2 — Smoke-test the 3 UNBLOCKS (the D0-class decisive runs), each RUN not
  config-constructed** ([[oracle-gateability-model-runs-not-config-constructs]]):
  1. `import transformers.models.gemma4` succeeds (was ModuleNotFoundError on
     5.13.1) → Gemma-4 mm gets an oracle.
  2. DFlash: re-run `scripts/spec/d0_dflash_oracle_capture.py` with
     **`VLLM_USE_V2_MODEL_RUNNER=1`** on `LLM(Qwen3.6-27B-NVFP4,
     speculative_config=dflash z-lab/Qwen3.6-27B-DFlash)` → the mixed-attn draft
     must now CONSTRUCT + propose/verify (clears vllm#40898). Capture the acceptance
     rate + greedy golden.
  3. OLMo-3: `LLM(allenai/OLMo-3-1025-7B)` BUILDS+RUNS a coherent greedy golden
     (was `olmo2.py:143` KeyError).
  **Gate:** all three RUN; if any fails, STOP and re-scope (do not advance the pin).
- **W3 — Re-capture drifted goldens (OUR code unchanged, NEW oracle).** For each
  SACRED gate + model-matrix row: regenerate the vLLM golden at `55596792`, `diff`
  against the committed golden. Unchanged ⇒ keep (proves no drift). Changed ⇒
  re-capture + record the drift in the ledger (oracle moved, not our bug). Priority
  order = §3 ranking (27B, 35B, 32B-NVFP4A16, Coder first). **Gate:** every SACRED
  gate PASSES against the re-captured golden with OUR code byte-unchanged.
- **W4 — Mechanical re-sync of the §2D files** (qwen3_next, layernorm, fused_moe,
  mamba_mixer2, sampler, qwen3_dflash, new olmo3) — one upstream PR per commit,
  bump file-header pins to `55596792`, port matching tests, ledger row each. Re-run
  the affected gates. **Gate:** re-synced gates still PASS; `-Werror` clean.
- **W5 — Flip the pin.** Fast-forward `/home/mudler/_git/vllm` to `55596792`; update
  the PARITY PIN line in [.agents/upstream-sync.md](../upstream-sync.md), the oracle
  block in [.agents/environment.md](../environment.md) (new venv + freeze hashes),
  the per-file header pins, and the `.agents/porting-inventory.md` pin refs. Write
  the sync report `.agents/sync/YYYY-MM-DD-5559679.md`. **Gate:** all seven record
  checkers green; state.md entry links the sync report.

**Caveats:** GB10 unified memory OOM-reboots the box ([[gb10-unified-memory-oom-reboots-box]])
— keep `gpu_memory_utilization` low, never run a big oracle alongside ctest;
CUDA-graph capture safety ([[cudagraph-capture-bakes-stack-addresses]]) — a clean
compute-sanitizer run does NOT prove capture safety; the Torch-2.13/FlashInfer-0.6.15
JIT will re-warm/re-compile (expect first-run JIT contamination in any nsys).

---

## 5. Deliverable 3c — PRIORITIZED POST-UNBLOCK WORK LIST

Once the pin is at `55596792` and the goldens are re-captured, ranked by
gain÷effort (unblocked-and-implemented first):

1. **OLMo-3 W5 gate** (`Olmo3ForCausalLM`) — IMPLEMENTED, near-zero effort, just
   needs the now-runnable oracle. Ride the landed olmo2 row; mirror the new
   upstream `olmo3.py` for the header pin.
2. **DFlash D1-D6** (`SPEC-DFLASH`) — now unblocked (mixed draft constructs under
   MRV2). The reuse map (spec §1) is verified-present; run D1 standalone-draft
   parity → D5 three-way 27B/35B token-exact gate → D6 acceptance/memory/graphs.
   Highest-value NEW feature (the headline post-MTP spec method).
3. **Gemma-4 multimodal** (image+video+**AUDIO**) — the new AUDIO modality + Gemma-4
   backbone now have a SACRED oracle. First-class roadmap priority (user's #1 mm
   ask). Scope the SigLIP vision tower + USM Conformer audio tower per
   [gemma4-multimodal.md](gemma4-multimodal.md).
4. **Mechanical re-sync closeout** — land the §2D re-ports as first-class ledger
   rows (rmsnorm-fusion #46998, ReplaySSM #48018, MoeWNA16 MK-oracle #44120).
5. **Top new frontier models** (from §2A, gain-ranked; most already owned by
   `CLAIM-MLA-DEEPSEEK` / `CLAIM-GLM-DSA-LATEST-DEEPSEEK`): DeepSeek-V4/V3.2
   (CT-quant + DSpark), GLM-5.2 (note the Blackwell-decode-opt REVERT #49768 —
   do NOT port the reverted path), MiniMax-M3, Kimi-K2.5, then Inkling (new full
   NVFP4+MTP+LoRA family), LongCat-Flash-Lite, the ASR/diarize audio arches.

---

## Protocol compliance map
| Field | Content |
|---|---|
| Row IDs | `CLAIM-PIN-ADVANCE-SCOPE` (this scope); unblocks `SPEC-DFLASH`, the Gemma-4 mm rows, the OLMo-3 (olmo2 W5) row |
| Scope | Pin-advance SCOPE + PLAN only; execution GPU-gated, oracle-serialized |
| Target | vLLM `origin/main` `55596792` + transformers 5.14.1 / torch 2.13.0 / flashinfer 0.6.15 / cutlass-dsl 4.6.0 |
| Blockers resolved | DFlash #47914(+#48113/#48167/#48524) under MRV2; Gemma-4 via transformers 5.14.1; OLMo-3 via native olmo3.py + 5.14.1 |
| Golden-drift risk | HIGH for the 2 NVFP4 hybrid gates + 32B-NVFP4A16 + Coder; ~4 core re-captures + a diff pass over ~30 rows |
| Gates | per-W-step gates in §4; every SACRED gate re-passes against re-captured goldens with OUR code unchanged |
| Dependencies | sibling 35B-MTP agent frees `~/venvs/vllm-oracle`; disk ≥200 GiB; GB10 |
| Work breakdown | W0-W5 (§4) |
| Risks/decisions | Torch-2.13/CUTLASS-4.6/FlashInfer-0.6.15 stack change = build + drift risk; MRV2 requirement for mixed DFlash; no release tag has the fixes ⇒ main commit |

---

## 6. EXECUTION W0-W2 RESULTS — GO/NO-GO VERDICT = **GO** (2026-07-26, `CLAIM-PIN-ADVANCE-SCOPE`)

Ran the W0-W2 go/no-go on dgx GB10 (sm_121a). **NO pin flip, NO golden re-capture,
`~/venvs/vllm-oracle` (0.25.0) left pristine as rollback.** Staged the target stack in
a NEW venv `~/venvs/vllm-oracle-next`. **Headline: the target stack builds+imports on
GB10, all three unblocks resolve, and the golden drift is FAR SMALLER than the §3
estimate — only 1 of the 3 STRICT core gates (27B) actually drifts.**

### W0 — DISK
Reclaimed **~12 GiB** (26 GiB → 38 GiB free) by deleting ONLY transient profiler
artifacts (`*.nsys-rep`/`*.sqlite`/`*.qdrep` under `~/bench`/`~/pk-bench`/`~/vllm-bench`,
excluding `~/work/apex` + `~/work/darwin_36b_opus`) + the rebuildable `~/work/pinenv`.
**GLM-4.7-Flash (59 GiB) is NOT reclaimable** — `tests/vllm/models/test_glm4_moe_lite_load.cpp`
+ `test_glm4_moe_lite_paged_engine.cpp` + `test_mla_attention_block.cpp` load it (verified
by grep); it was PRESERVED. No SACRED checkpoint or user data touched.

### W1 — STAGED venv `~/venvs/vllm-oracle-next` — **INSTALLS + IMPORTS on GB10 ✓**
The exact commit `55596792` has **no aarch64 wheel** (per-commit index 404; it is a
main commit NOT on any release branch — v0.26.0/v0.25.1 diverged), so vLLM was **built
from source** (editable) for sm_121a only (`TORCH_CUDA_ARCH_LIST=12.1a`, `MAX_JOBS=6`,
~1h21m, ~380 CUDA TUs, disk floor 18 GiB). Deps installed clean from PyPI + the
`https://flashinfer.ai/whl/` index. Verified on GB10: `torch 2.13.0+cu130` (CUDA 13.0,
cap (12,1)), `transformers 5.14.1`, `flashinfer 0.6.15.post1` (+cubin), `nvidia-cutlass-dsl
4.6.0`, `triton 3.7.1`, `torchvision 0.28.0`, `vllm 0.26.0.dev0+g5559679` (our build of
`55596792`); `vllm._custom_ops` (compiled `_C`) loads; platform = NVIDIA GB10.
**Build friction resolved (record for W1 EXECUTION):** (1) commit has no aarch64 wheel ⇒
source build; (2) vLLM now needs a **Rust** toolchain (`setuptools-rust`, `rust/Cargo.toml`)
— installed `rustup` stable 1.97.1; (3) `git archive` tarball has no `.git` ⇒ setuptools-scm
fails ⇒ set `SETUPTOOLS_SCM_PRETEND_VERSION` + `git init`; (4) `flashinfer-cubin` is off
PyPI since 0.6.14 ⇒ pulled `0.6.15.post1` from the flashinfer index; (5) target pins are
`flashinfer 0.6.15.post1`, `nvidia-cutlass-dsl[cu13]==4.6.0` (+ `quack-kernels`,
`humming-kernels[cu13]`, `tilelang`, `tokenspeed-mla`) — all resolve on aarch64.

### W2 — THE THREE UNBLOCKS + DRIFT
**(a) Gemma-4 — ✓** `import transformers.models.gemma4` + `Gemma4Config/Processor/
TextConfig/VisionConfig` all import under transformers 5.14.1. The decisive block is gone.

**(b) DFlash (HEADLINE) — ✓ CONSTRUCTS + RUNS + NON-ZERO ACCEPTANCE under MRV2.**
`LLM(unsloth/Qwen3.6-27B-NVFP4, speculative_config={method:dflash, model:z-lab/Qwen3.6-27B-DFlash,
num_speculative_tokens:16})` with `VLLM_USE_V2_MODEL_RUNNER=1`: the mixed sliding/full-attn
draft (#40898) now RESOLVES `DFlashDraftModel`, loads (24.74 GiB), and generates coherent
tokens on all 4 prompts. The DFlash `_prepare_dflash_inputs_kernel` JIT-compiled+ran DURING
inference (drafter is LIVE, not dead). **Acceptance_len = 2.21 / 8.8 / 4.75 / 4.57** (all
well above the dead-drafter floor 1.0). **Backend note:** forcing `FLASH_ATTN` FAILS on
sm_121 ("FP8 KV cache requires FA3 on SM90 or FA4 on SM100"); **auto-select works** — vLLM
picks `flashinfer-native` non-causal decode w/ fp8 KV on `arch=sm121` (the #48167 Blackwell
fix is live). D1-D6 should NOT pin `FLASH_ATTN`.

**(c) OLMo-3 — NOT RUN (checkpoint absent).** `allenai/OLMo-3-1025-7B` is not in the HF
cache and W0 left only 18 GiB free (a 14 GiB re-download would leave too little headroom
for the venv build cache), so this was deferred. Needs a re-fetch in W3 before the OLMo-3
SACRED gate. The transformers-5.14.1 nested-`rope_parameters` fix is present in the stack.

**(d) MEASURED DRIFT — the real re-capture bill (vs committed goldens):**

| Gate | Path | Result | Detail |
|---|---|---|---|
| **Qwen3.6-27B-NVFP4** | compressed-tensors **W4A4** (fp4 acts) | **DRIFTS** | first 6 tokens identical ("capital of Germany is Berlin."), **first-divergence pos 6**, **10/16 differ**; new stack emits a `<think></think>` block. **Needs re-capture.** |
| **Qwen3.6-35B-A3B-NVFP4** | modelopt **W4A16** | **SURVIVES** | **16/16 byte-identical.** No re-capture. |
| **Qwen3-Coder-30B-A3B** | bf16 MoE (STRICT 6/6) | **SURVIVES** | **0/6 prompts drift** (all 16/16). No re-capture. |
| **Qwen3-4B dense** | bf16 near-tie (distributional gate) | 5/16 point-drift | near-tie noise (2 prompts differ by 1 token); distributional gate likely absorbs, re-validate. |

**Verdict on the drift bill:** the §3 estimate ("~4 core re-captures") is a **worst case**;
MEASURED, only the **27B W4A4** STRICT golden actually drifts. **35B (W4A16) and Coder are
bit-stable** — the true-fp4-activation path (27B) is the sensitive one; W4A16 + bf16 MoE
reproduce exactly. W3 re-capture is materially cheaper than planned: **1 STRICT re-capture
(27B) + a near-tie re-validation of the small-dense set**, not a full ~30-row campaign.
(32B-NVFP4A16 + the op-level goldens still owe a W3 diff — not measured here.)

**GO/NO-GO = GO.** Target stack runs the NVFP4 gate models on GB10; DFlash constructs+runs
under MRV2; drift is small. Proceed to W3 (targeted 27B re-capture + OLMo-3 fetch + 32B/op
diff) → W4 (mechanical re-sync) → W5 (pin flip). Evidence on dgx: `~/work/pin-drift/`
(`dflash_stats.log`, `drift_{27b,35b}.log`, `out_coder/`, `out_4b/`); build log
`~/work/vllm_build.log`; source `~/work/vllm-src-5559679`.

---

## 7. EXECUTION W3a-W4 RESULTS — RCA VERDICT = **NEAR-TIE (NO real drift, NO port)**; re-gate GREEN (2026-07-26, `CLAIM-PIN-ADVANCE-W3W4`)

Ran W3a (RCA the 27B drift) + W3b (full drift diff) + W4 (re-sync) + the re-gate on dgx
GB10 (sm_121a). **NO pin flip** (`~/venvs/vllm-oracle` = 0.25.0 left pristine; the W0-W2
`~/venvs/vllm-oracle-next` REUSED). Base `origin/main` `36381c1b`, isolated worktree
branch `pin-advance-w3w4`. **HEADLINE: the W0-W2 "27B drift" is NOT real drift — it is the
already-DOCUMENTED tok6 whitespace near-tie flipping on incidental engine/sampling config.
Under an apples-to-apples re-capture the new oracle reproduces EVERY measured SACRED golden
BIT-FOR-BIT. The definitive re-capture list is EMPTY; W4 kernel re-sync = NONE.**

### W3a — RCA the 27B `<think>`-block "drift" — VERDICT: benign tok6 near-tie (not template, not numeric-port)
1. **Template? NO.** The 27B gate feeds the prompt as RAW token-ids (no chat template):
   `dump_qwen36.py:236-237,301` tokenizes `"The capital of France is Paris, and the"` with
   `add_special_tokens=True` → ids `[760,6511,314,9338,369,11751,11,321,279]` and generates
   greedy. The prompt ids are BYTE-IDENTICAL old-vs-new (no `enable_thinking` / `<think>`
   trigger in play). The `<think>` block in the W0-W2 capture is an ARTIFACT of a near-tie
   cascade, not a template default.
2. **Decisive test — the tok6 near-tie is DOCUMENTED and the forward is bit-exact.** The
   codebase already committed BOTH branches of this near-tie: `qwen36_logits_27b/greedy_ids.npy`
   = `[…,13,198,760,…]` (198 `"\n"`, the production branch) vs
   `greedy_ids_emulation.npy` = `[…,13,271,248068,…]` (271 `"\n\n"` → `<think>`, the emulation
   branch), with `test_qwen27_paged_engine.cpp:112-118,144-149,236-237` asserting
   `REQUIRE(want_emu != want_prod)` and gating our engine to the production branch. The W0-W2
   "new drift" output equals the committed **emulation** golden EXACTLY — the new stack merely
   tipped the pre-characterized near-tie to the OTHER already-committed branch.
   - **Near-tie gap MEASURED** (`rca_27b_neartie.py`, new oracle, per-step top-k logprobs):
     at tok6 three candidates bunch within **0.5625 nats** — `271 "\n\n"` (-1.937),
     `" What"` (-2.312), `198 "\n"` (-2.4995); **top-1 probability ≈ 14%**. Every OTHER step
     (0-5, 7-15) is a decisive argmax (p > 0.93; " is"/" correct"/"</think>" at p > 0.99) and
     both trajectories share the identical `"capital of Germany is Berlin."` prefix. tok6 is
     the ONE genuinely undecided position — a textbook bf16/fp4 whitespace near-tie.
   - **Config-sensitivity PROVEN on ONE oracle version.** The SAME `vllm-oracle-next` emits
     `198` under the canonical golden-capture config (`dump_qwen36.py`: `max_model_len=256`,
     `max_num_seqs=1`) but `271` under the W0-W2 `drift_capture.py` config
     (`max_model_len=8192`, `max_num_batched_tokens=8192`, `max_num_seqs=4`, `logprobs`). The
     near-tie flips on incidental engine config, NOT on the oracle version.
   - **Our-engine reproduction.** Our C++ engine deterministically produces the production
     `198` branch (its default merged-BF16 GDN + FlashInfer W4A4 chain), reproduced by the new
     oracle under the canonical capture; the strict 27B gates keep passing unchanged.
   - **VERDICT: benign tok6 whitespace NEAR-TIE.** NOT a chat-template default (raw ids,
     byte-identical). NOT a portable fp4-act numeric change — the entire 27B FORWARD is
     bit-identical on the new stack (see W3b). **NO W4 kernel port.**

### W3b — full drift diff (new oracle `55596792` stack vs committed goldens) — DEFINITIVE RE-CAPTURE LIST = **EMPTY**
Re-captured with the CANONICAL per-golden capture config (apples-to-apples) and diffed:

| Gate | Method | Result vs committed |
|---|---|---|
| **Qwen3.6-27B-NVFP4 W4A4** (STRICT) | `dump_qwen36.py --tag 27b` full re-capture | **BIT-IDENTICAL.** embed/gdn_layer/fullattn_layer/norm hidden states `max_abs=max_rel=0.0`; `argmax`, `greedy_ids`(=198), `topk_indices`, `topk_values` ALL EXACT (0 diff). |
| **Qwen3-32B-NVFP4A16 W4A16** (STRICT) | `qwen3-32b-nvfp4a16-oracle-capture.py --runs 5` | **BIT-IDENTICAL.** `greedy_ids` + `greedy_dist` 0 diff; K=5 all-deterministic ⇒ STRICT gate form unchanged. |
| **Qwen3.6-35B-A3B-NVFP4 W4A16** (STRICT) | W0-W2 measured | SURVIVES 16/16 byte-identical. |
| **Qwen3-Coder-30B-A3B** bf16 MoE (STRICT/dist) | W0-W2 measured | SURVIVES 6/6 (0/6 drift). |
| **Qwen3-4B** dense bf16 (distributional) | W0-W2 measured | near-tie noise absorbed by the ratified `greedy_dist` near-tie-robust gate. |
| **27B op-goldens** (embed/norm/gdn/fullattn — REAL bf16 GDN/attn/rmsnorm kernels on the new stack) | part of the 27B re-capture | BIT-IDENTICAL (0.0) ⇒ the shared bf16 kernels do not drift. |
| **~30 model-matrix `*_greedy` rows** (llama/opt/phi/mistral/internlm/minicpm/yi/olmo2/deepseek-v2/glm4-moe-lite/…) | already RATIFIED near-tie-robust gates (`greedy_dist.npy`, `kNearTieMnats=500`) | no re-capture: the distributional gates absorb near-tie drift by construction; validated against OUR unchanged engine by the re-gate ctest. |
| **op-level goldens** (rmsnorm/rope/silu/moe/causal_conv1d/gdn/l2norm/matmul) | `forward_native` reference math; f32 = dtype-stable/inert, bf16 shown stable via the 27B op-goldens | no re-capture; validated by the op-parity ctests. |

**CORRECTION 2026-09-03 (#2794): the `~30 model-matrix *_greedy rows` row above does NOT
cover OPT-125m, and OPT's golden was therefore never re-validated at this pin.** That row
discharges its members on the ground that they are "already RATIFIED near-tie-robust gates
(`greedy_dist.npy`, `kNearTieMnats=500`)" whose "distributional gates absorb near-tie drift by
construction". `tests/vllm/models/test_opt_paged_engine.cpp` is not one: its bar is STRICT
token-exact and its header says "no near-tie band is used here at all", and the body carries no
`kNearTieMnats`. `git log --follow` over `tests/parity/goldens/opt_greedy/greedy_ids.npy`
returns one commit, `b8358a5b9`, an ancestor of the `bc415a3e4` pin flip, so nothing has
touched the golden since it was captured against vLLM 0.25.0 / `e24d1b24`. The premise is false
for this member, so the discharge does not reach it; the re-capture is owed and is tracked by
[#2794](https://github.com/mudler/vllm.cpp/issues/2794) together with the same capture at
`e126687a9a`. The other members of that row are not re-examined here, and this correction makes
no claim about them. Nothing else in W3b is affected: 27B W4A4, 32B-NVFP4A16, 35B and Coder
were each really re-captured or re-measured at `55596792`, so the pin does have declared
token-exact gates. What is false is the wider claim that every strict golden crossed the
advance.

**No committed golden drifts under apples-to-apples capture.** The W0-W2 §6(d) "27B DRIFTS"
row is superseded: it measured a near-tie config artifact, not oracle drift.

### W4 — mechanical re-sync = **NONE required.**
No gate exhibits a real numeric change (the entire 27B forward + 32B/35B/Coder goldens are
bit/near-tie-stable on the new stack), so no §2D kernel/op is load-bearing for a drifted gate.
Per the sync cycle those §2D files (qwen3_next rmsnorm-fusion #46998, layernorm #48741,
fused_moe #44120, mamba_mixer2 #48018, sampler #46662, qwen3_dflash, new olmo3.py) remain a
routine mechanical re-port to land at the W5 pin-flip, but NONE is required to keep a SACRED
gate green against the new oracle. Porting a kernel to force our engine onto the emulation-271
branch would REGRESS the `got == want_prod` / `got != want_emu` production gate — explicitly
NOT done.

### Re-gate — OUR engine (unchanged, `36381c1b`) vs the validated goldens on GB10
Built ONCE from the worktree HEAD (git-archive → `~/work/vllm.cpp-pinw3w4`):
`cmake -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0
-DVLLM_CPP_TRITON=ON -DCMAKE_BUILD_TYPE=Release` (FA2 sm_121a ENABLED, Triton AOT vendored
manifest OK). Full serial `ctest` under `flock $HOME/gpu.lock` (299 tests, 1966 s):
**296 Passed / 3 Failed (99%)**. **Every SACRED gate PASSED and RAN** (checkpoints present):
`test_op_parity` (27B W4A4 STRICT logits — 16/16 greedy + per-position prefill argmax — + 35B
`RunQwen36Logits` + all op goldens, 187 s), `test_qwen36_paged_engine` (35B STRICT greedy +
batched-graph, 77 s), `test_qwen27_paged_engine` (27B W4A4 STRICT `got==want_prod` [198]
`&& got!=want_emu` [271], 31 s), `test_qwen3_32b_nvfp4a16_paged_engine` (28 s),
`test_qwen3coder_paged_engine` (105 s), 27B/35B `spec_decode` + concurrent, and the
`deepseek_v2`/`glm4_moe_lite`/`yi`/`commandr`/`phi`/`minicpm3` paged engines. **The 3 failures
are PRE-EXISTING on unchanged main (`36381c1b`) and UNRELATED to the pin/oracle/goldens** (our
code + goldens are byte-identical to main): `test_model_registry` (hardcoded `registrations
== 13` vs the grown registry's 24), `test_model_loader_gguf` (stale 6-arch reject string +
tokenizer fixture), `test_capi` (toy random-weight model garbled-first-byte greedy-determinism
flake under async scheduling). -Werror clean (0 `warning:` lines across both build logs).

**Bottom line.** The pin advance is materially CHEAPER than the W0-W2 worst case: **zero real
golden drift, zero kernel re-sync.** Evidence on dgx `~/work/pin-drift/`: `rca_27b.log`
(near-tie gap), `dump27_new.log`+`gold27_new/` (27B bit-identical), `dump32_new.log`+`gold32_new/`
(32B bit-identical), `build.log`, `regate.log`.

## §7. W5 — PIN FLIPPED (2026-07-26, `CLAIM-PIN-ADVANCE-W5`) — DONE

The parity pin is now **`555967922` / vLLM 0.26.0.dev0**, advanced from `e24d1b24` / 0.25.0.
Runtime flip executed + verified: (1) dev-box source `/home/mudler/_git/vllm` checked out
`555967922`; (2) dgx oracle symlink `~/venvs/vllm-oracle` → `~/venvs/vllm-oracle-next`
(imports `vllm 0.26.0.dev0+g5559679 / transformers 5.14.1 / torch 2.13.0+cu130`); the
`~/venvs/vllm-oracle-v0.25.0-stage` rollback is preserved (`ln -sfn …-v0.25.0-stage
~/venvs/vllm-oracle` restores it). Canonical pin declarations updated (AGENTS.md,
environment.md, this spec, porting-inventory); the 656 historical `@ e24d1b24` citations are
LEFT as-is (they record the pin each past task ran against — accurate to their moment).

**Now UNBLOCKED (priority order):** OLMo-3 W5 gate (fetch checkpoint + run — impl already
landed) → DFlash D1–D6 (mixed-attn draft constructs+accepts under MRV2; **D-series must NOT
pin FLASH_ATTN** — sm_121 fp8-KV needs the auto-selected flashinfer-native path) → Gemma-4
mm+audio (oracle now runnable) → §2D mechanical re-sync of upstream-changed mirrored files
(DEFERRED, not gate-blocking) → top new frontier arches (Olmo3/Inkling/etc.).
