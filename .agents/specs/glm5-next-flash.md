# `Glm5NextForConditionalGeneration` (GLM-5.3-Flash)

**SCOPING ONLY. NO PRODUCT CODE LANDS UNDER THIS PULL REQUEST.** This document
and its records are the whole deliverable of the pull request that introduces
it. Implementation follows in separate `row/MODEL-MM-GLM53-FLASH-W<n>` branches
dispatched from the committed spec, per the split-pull-request case in AGENTS.md
§"Spec before code".

- **Matrix row:** `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`
  (`.agents/model-matrix.md`, §MODEL-MM).
- **Campaign row:** `MODEL-MM-GLM53-FLASH`.
- **Issue:** [#1998](https://github.com/mudler/vllm.cpp/issues/1998).
- **State on landing:** `READY` — the spec is committed and no product code has
  landed. It may not become `ACTIVE` until wave 1 is claimed.
- **Model card:** <https://huggingface.co/zai-org/GLM-5.3-Flash>, read live
  2026-08-26.

Every fact below carries the date it was read. Nothing here is measured on
hardware: no lease was taken, no GPU ran, and no weights were downloaded. Where
a number is arithmetic over measured file sizes it says so, and where a
quantity is unknown it says "not measured" rather than being estimated.

## Scope

Port `Glm5NextForConditionalGeneration` — GLM-5.3-Flash, 321.32B total
parameters, ~18B activated, natively multimodal (text + image + video) — so that
it loads and generates through `ModelRegistry::Forward` and the `include/vllm.h`
ABI, with the quantized arms AGENTS.md makes a standing requirement.

IN scope for the campaign this spec plans:

- The text backbone: a 45-layer hybrid of 34 KDA linear-attention layers and 11
  DeepSeek-Sparse-Attention (DSA) MLA layers, with a manifold hyper-connection
  (mHC) residual topology, a 288-expert + 1-shared fine-grained MoE, and a
  clamped SwiGLU.
- The vision tower (24-layer ViT, patch 14, spatial merge 2, temporal patch 2)
  and the image/video placeholder expansion.
- Config resolution, architecture registration, weight-name mapping, and a
  forward that refuses by name until each primitive exists.
- The GGUF k-quant arm, including the converter that has to be authored because
  no upstream tool can emit this architecture.

OUT of scope, explicitly:

- Advancing the vLLM parity pin. `555967922` does not reach this architecture
  and neither does vLLM `main`; see §Oracles. Nothing in this campaign may move
  `.agents/upstream-sync.md`.
- The MTP speculative head (`num_nextn_predict_layers = 1`). It is recorded
  under §Owed and gets its own row when the backbone runs.
- Any claim of speed parity. There is no denominator: no oracle runs this model
  on any device this project can reach.
- Reworking the shared MLA block for models other than this one. Where the
  shared seam cannot represent the geometry, §Design says which seam is
  extended and under which wave.

## Why this needs a spec before code

Three reasons, each of which has already cost this repository once.

**The name collides with a different, blocked model.**
`.agents/model-matrix.md` already carries a row labelled "GLM-5":
`MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm` (`GlmMoeDsaForCausalLM`),
state `BLOCKED`, owned by `CLAIM-GLM-DSA-LATEST-DEEPSEEK`. That row is
DeepSeek-V3.2 verbatim (`deepseek_v2.py:1917-1918` at the pin: `class
GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`) and is blocked at 753.9B /
1404 GiB. `Glm5NextForConditionalGeneration` is a **different architecture with
its own `model_type` (`glm5_next`)**, a different parameter count, a different
attention topology and a vision tower. It is a new row, not a revival of the
blocked one, and this paragraph exists so the next reader does not merge them.

**The obvious reuse is the wrong reuse in one specific place.** Three of the
four exotic families this model needs already exist in this tree — mHC, the DSA
lightning indexer, and KDA. An implementer who sees that will reach for them
directly. Two of the three have a documented delta that a token gate cannot
see, and §Port map names them: the KDA forget gate takes a **different branch**
from the one our Kimi reference ports, and the DSA indexer adds a **k-pool
compression stage** our DeepSeek-V4 indexer does not have.

**Nothing can gate it.** No oracle implements this architecture at a pinned
revision, and no artifact of it fits any device on this fleet. A campaign that
does not decide up front what "correct" means here will produce a model that
generates plausible text and is never checked. §Gates decides it.

## Oracles

Everything in this section was read live on **2026-08-26**.

### vLLM at the parity pin `555967922` — implements NOTHING

`git grep -n "Glm5\|glm5_next" 555967922 -- vllm/` returns zero hits. The
registry at the pin carries `GlmForCausalLM`, `Glm4ForCausalLM`,
`Glm4MoeForCausalLM`, `Glm4MoeLiteForCausalLM`, `GlmMoeDsaForCausalLM`,
`Glm4vForConditionalGeneration`, `Glm4vMoeForConditionalGeneration`,
`GlmOcrForConditionalGeneration`, `GlmAsrForConditionalGeneration` and three
GLM MTP heads — `vllm/model_executor/models/registry.py:113-117,409-413,640-642`
@ `555967922`. There is no `Glm5*` entry of any kind.

### vLLM `main` — implements NOTHING EITHER

Fetched to `origin/main` = `c71f6f8a81d3d3c49a045c8b88eed36366cc7d92`
(2026-08-26 08:42:52 -0700). `git grep -n "Glm5\|glm5_next" origin/main --
vllm/` returns zero hits. The `vllm/models/` package at `main` holds `common`,
`deepseek_v32`, `deepseek_v4`, `dots3_note`, `inkling`, `kimi_k3` and
`minimax_m3` — no `glm5next`. The GLM entries in
`registry.py:114-118,401-405,655-657` @ `c71f6f8a81` are the same nine plus
`GlmMoeDsaForCausalLM` re-homed to `vllm.models.deepseek_v32`.

**This is absence, not pin staleness.** Advancing the parity pin does not reach
this architecture, so this campaign carries no pin-advance dependency and must
not create one. What advancing the pin *would* cost, stated so the option is
priced rather than left vague: 348 commits between `555967922` and
`c71f6f8a81`, each of which AGENTS.md §"Pin vLLM" requires be reconciled
against every affected row and gate, and the last such cycle
(`.agents/specs/pin-advance.md`) re-validated the binding golden grids on GB10.
That cost buys nothing here, because the architecture is absent at both ends.

### vLLM PR #53906 — OPEN, and therefore INADMISSIBLE

[vllm#53906](https://github.com/vllm-project/vllm/pull/53906) "[Model] add
GLM-5.3-Flash support" was opened 2026-08-26T14:12:00Z, is **open, unmerged, and
`mergeable: false`**, one commit at head `933876c388fb129ad82590660e6506614559cb86`
over base `903a02192fca19e4c89705af7017c0f24971ea4f`, 85 files, +12,511/-540.
It would register three architectures:
`Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration` and
`Glm5NextMTPModel`, all against a new `vllm/models/glm5next` package
(`vllm/model_executor/models/registry.py` hunks at `+119`, `+425`, `+672` in the
PR diff).

An unmerged pull request is not a revision and cannot be pinned, so it is not an
oracle. This repository already recorded that rule on the `MODEL-MM-QWEN4-EXP`
row for SGLang #36497 and it applies here unchanged. **The PR is cited in this
spec only as evidence of upstream shape** — it tells us which components
upstream considers new enough to need their own file, and that is planning
information, not a port source. No `file:line` from it may appear in a port map
cell. When it merges, §Stop conditions says what happens.

What it tells us about shape, and nothing more: upstream needed a new
`sparse_attn_indexer_kpool.py` (+1009), a `glm5next/nvidia/ops/kpool_compress.py`
(+948), a `glm5next/nvidia/kda.py` (+628), a `glm5next/nvidia/attention.py`
(+599), a `fused_eh_norm.py` (+77), and edits to `layers/mhc.py`,
`layers/mla.py` and `third_party/flash_linear_attention/ops/kda.py` (+93/-55).
That is the same four-family split this spec independently derives from the
transformers reference, which raises confidence in the decomposition without
being usable as its source.

### transformers — the ONLY admissible reference, and it needs a lane pin

`glm5_next` exists in `huggingface/transformers`. The implementing commit is
`eb4d9e2a64` (2026-08-26T14:26:40Z, PR
[transformers#48342](https://github.com/huggingface/transformers/pull/48342)
"[Glm 5.3 Flash] GLM 5.3 Flash Support"). The package is
`src/transformers/models/glm5_next/`: `modular_glm5_next.py` (95,314 bytes, the
source of truth), `modeling_glm5_next.py` (102,403), `configuration_glm5_next.py`
(14,202), `processing_glm5_next.py` (9,598), `image_processing_glm5_next.py`,
`image_processing_pil_glm5_next.py`, `video_processing_glm5_next.py`.

**The first RELEASE carrying it is `v5.16.1`** (published 2026-08-26T14:50:01Z).
This was bounded rather than assumed, by fetching
`src/transformers/models/glm5_next/modeling_glm5_next.py` at each tag:

| tag | published | `modeling_glm5_next.py` |
|---|---|---|
| `v5.15.1` | 2026-08-19T10:50:47Z | HTTP **404** |
| `v5.16.0` | 2026-08-26T12:35:15Z | HTTP **404** |
| `v5.16.1` | 2026-08-26T14:50:01Z | HTTP **200** |

`.agents/oracles/transformers.md` pins transformers to **5.14.1**, deliberately
tied to what the pinned vLLM environment resolves so that environment cannot
hold two `transformers` at once. 5.14.1 does not contain `glm5_next`. This row
therefore needs a **lane-scoped second pin at `v5.16.1`**, on exactly the
argument the `MODEL-MM-QWEN4-EXP` row records and had accepted: the invariant
guards a *vLLM environment* against drifting from its transformers, and here
there is no vLLM implementation to drift from. It expires the moment vLLM
registers `glm5_next`.

**Note for whoever writes that oracle file: the `MODEL-MM-QWEN4-EXP` lane pins
`5.16.0` and this lane pins `5.16.1`.** Two lanes, two different releases, one
day apart, because `Qwen4Exp` merged before the 5.16.0 cut and `Glm5Next` after
it. That is not a contradiction to reconcile; it is why a lane pin is written as
a lane pin. Do not "tidy" one onto the other.

### llama.cpp — implements NOTHING

Code search `glm5_next repo:ggml-org/llama.cpp` returns **0**. PR
[ggml-org/llama.cpp#27752](https://github.com/ggml-org/llama.cpp/pull/27752)
"model : add GLM-5.3-Flash (glm5next)" is **open**. Our llama.cpp oracle is
pinned at tag `b10451` (`.agents/oracles/llama-cpp.md`), which predates the
model entirely. **Consequence: `convert_hf_to_gguf.py` cannot emit this
architecture, `llama-quantize` cannot quantize it, and the GGUF arms of this
model have no llama.cpp oracle and no llama.cpp floor.** §Hardware says what
that forces.

### SGLang — implements NOTHING

`glm5_next repo:sgl-project/sglang` code search returns **0**, and
`python/sglang/srt/models/` at `main` carries `glm4*`, `glm_ocr*`, `glm_image_vl`,
`glmasr`, `chatglm` and no `glm5`. PR
[sgl-project/sglang#36507](https://github.com/sgl-project/sglang/pull/36507)
"GLM-5.3-Flash support" is **open**. The GLM-5.3-Flash *cookbook* PRs #36440 and
#36513 **did** merge (2026-08-26T14:00:16Z, 14:39:29Z) — documentation landed
ahead of the implementation, which is a trap: the model card links a cookbook
that describes a code path SGLang `main` does not yet contain. SGLang is
inadmissible here.

### vLLM-Omni — implements NOTHING

`glm5 repo:vllm-project/vllm-omni` code search returns **0**. This is not an
omni-only architecture and there is no reason to reach for that repository.

### Gateability

`gateable = no` for every oracle, on **memory**, and this is the load-bearing
verdict of the whole scoping exercise. AGENTS.md admits `gateable = yes` only
after an oracle demonstrably builds and runs the model. The only admissible
oracle is transformers `v5.16.1`, and running it needs the FP8 checkpoint
(305.78 GiB) or the BF16 one (598.5 GiB) resident. The largest device this
project can reach is `dgx:gpu0` at ~119.63 GiB of unified memory. **There is no
device on this fleet, and no combination of them, on which the reference
implementation of this model can be executed.** No token-exact end-to-end gate
against an oracle is reachable, now or after any amount of implementation work.

That is a statement about the full model. It is NOT a statement that this
campaign is ungateable; §Gates constructs the gate that is actually reachable.

## Upstream chain

`transformers` `v5.16.1`, `src/transformers/models/glm5_next/`. The modular file
is the source of truth and every port-map cell below cites
`modular_glm5_next.py:<line>` at that tag. The expanded
`modeling_glm5_next.py` is generated from it and is the file to read when the
modular inheritance is ambiguous.

The inheritance graph, which is most of the port plan (`modular_glm5_next.py`
@ `v5.16.1`):

| class | line | inherits |
|---|---|---|
| `Glm5NextTextConfig` | `:92` | `GlmMoeDsaConfig` |
| `Glm5NextVisionConfig` | `:248` | `GlmOcrVisionConfig` |
| `Glm5NextConfig` | `:265` | `PreTrainedConfig` |
| `Glm5NextTextMLP` | `:321` | `Qwen2MoeMLP` (+ `swiglu_limit` clamp) |
| `Glm5NextTextExperts` | `:335` | `MiniMaxM3VLExperts` |
| `Glm5NextTextTopkRouter` | `:350` | `DeepseekV3TopkRouter` (`pass`) |
| `Glm5NextTextMoE` | `:354` | `DeepseekV3MoE` |
| `Glm5NextTextHyperConnection` | `:364` | `DeepseekV4HyperConnection` (`pass`) |
| `Glm5NextTextHyperHead` | `:368` | `nn.Module` — **unweighted mean** |
| `Glm5NextTextForgetGate` | `:375` | `nn.Module` — **NEW branch** |
| `Glm5NextTextRMSNormGated` | `:409` | `Qwen3_5RMSNormGated`, activation `sigmoid` |
| `l2norm` | `:429` | free function, `sqrt(sum + eps)` not `max(.., eps)` |
| `recurrent_kimi_delta_attention` | `:441` | free function (decode) |
| `chunk_kimi_delta_attention` | `:495` | free function (prefill) |
| `Glm5NextTextLinearAttention` | `:597` | `nn.Module` (KDA arm) |
| `Glm5NextTextIndexer` | `:749` | `GlmMoeDsaIndexer` (+ k-pool) |
| `Glm5NextTextAttention` | `:1025` | `GlmMoeDsaAttention` (MLA, NoPE) |
| `Glm5NextTextDecoderLayer` | `:1142` | `GlmMoeDsaDecoderLayer` |
| `Glm5NextTextModel` | `:1285` | `Glm5NextPreTrainedModel` |
| `Glm5NextVisionMLP` / `PatchMerger` / `Block` / `Model` | `:1373`/`:1387`/`:1403`/`:1410` | `GlmOcrVision*` |
| `Glm5NextModel` | `:1422` | `Exaone4_5_Model` |
| `Glm5NextForConditionalGeneration` | `:1563` | `Glm46VForConditionalGeneration` |
| `Glm5NextProcessor` | `:1704` | `Glm46VProcessor` |
| `Glm5NextImageProcessor` | `:1792` | `GlmgaImageProcessor` |
| `Glm5NextVideoProcessor` | `:2121` | `GlmgaVideoProcessor` |

Read that table as: **the model is assembled almost entirely from families this
tree has already engaged**, and the work is in the handful of classes that are
not a `pass`.

## Our baseline

What this tree already has, with the maturity stated rather than implied. Every
anchor below was read at `21fe11cf1`.

### DSA lightning indexer — host reference plus host-vector CUDA, no fused pipeline

- CPU reference: `src/vllm/model_executor/models/deepseek_v4_dsa.cpp:14`
  (`DsaIndexerWeightFold`), `:31` (`DsaIndexerLogits`), `:72` (`DsaTopkSelect`),
  `:116` (`SoftmaxWithSink`), `:136` (`GroupedOutputLora`).
- CUDA kernels: `src/vt/cuda/cuda_deepseek_v4.cu:592`, `:598`, `:624`, `:705`,
  `:722`; registered `:2094`.
- **Maturity caveat:** the device entry points take and return host
  `std::vector` (`include/vllm/model_executor/models/deepseek_v4_device.h:82-102`),
  so each call is upload → launch → download → sync. The one in-place device
  kernel is `decode_attn` (`deepseek_v4_device.h:112`), called from
  `src/vllm/model_executor/models/deepseek_v4.cpp:853`. The V4 forward is
  DERIVED and structurally gated at a synthetic shape, not real-checkpoint
  token-gated — its own header says so at
  `src/vllm/model_executor/models/deepseek_v4.cpp:1-17`.

### MLA — present, and it REFUSES this model's geometry

`MlaBlockDims` is `include/vllm/model_executor/models/mla_attention.h:108-166`;
`Validate()` is `src/vllm/model_executor/layers/attention/mla_attention.cpp:89`.

- `qk_rope_head_dim == 0` is **refused**: `mla_attention.cpp:90-93` requires
  every dimension `> 0`, and `:95-99` additionally requires it even. GLM-5.3-Flash
  sets `qk_rope_head_dim: 0`. This is the single hardest structural item in the
  port.
- Kimi-Linear is the closest analogue and is **not the same thing**: it sets
  `mla_use_nope = true` (`include/vllm/model_executor/models/kimi_linear.h:88`)
  while keeping `qk_rope_head_dim = 64` (`:86`) — the rope slice still exists in
  the cache row, only the rotation is skipped. Here the slice does not exist and
  `head_size()` (`mla_attention.h:162`) becomes `512`, not `576`.
- `v_head_dim <= qk_head_dim()` (`mla_attention.cpp:100-104`) passes at 256/256
  once the rope check is relaxed.
- `q_lora_rank = 1536` is the supported DeepSeek-V3 branch
  (`mla_attention.h:117`, `:164`); the loader must fuse `q_a_proj` +
  `kv_a_proj_with_mqa` (`mla_attention.h:238-244`).
- Prefill FA-2 compiles head dims `{128, 192, 256}`
  (`src/vt/cuda/cuda_mla_prefill.cu:194-208`), refusing `> 256` by name at
  `:208`; `qk_head_dim == 256` is already live (GLM-4.7-Flash). Decode reads
  `head_size` dynamically (`src/vt/cuda/cuda_mla_attn.cu:487`) and branches on
  `v_head_dim <= 512` at `:564`, with a shared-memory guard at `:545-546`; a
  512/256 pair should fit and is **untested**.

### mHC — host and CUDA, configurable, DeepSeek-V4-exclusive

`src/vllm/model_executor/models/deepseek_v4_mhc.cpp:23` (`MhcSinkhorn`), `:72`
(`MhcPre`), `:149` (`MhcPost`), `:168` (`HcHeadCollapse`); all take `int64_t hc`
as a runtime argument. CUDA at `src/vt/cuda/cuda_deepseek_v4.cu:143`, `:208`,
`:270`, `:285`, `:534`, `:552`; registered `:2092`. Reached only from
`src/vllm/model_executor/models/deepseek_v4.cpp:255,269,284,287,298,301,1815`.

### KDA — the most mature of the three, and default-OFF on device

Kimi-Linear-48B-A3B runs end to end with a real gate (`docs/FEATURES.md:151`:
engine==CLI 128/128 byte-identical, 122/128 vs golden, ~19.0 tok/s vs vLLM ~21).

- `vt::KdaGatedDeltaRule`: `include/vt/ops.h:105`, `:2867`; CUDA
  `src/vt/cuda/cuda_gdn.cu:3207`, registered `:6648`.
- `vt::KdaChunkPrefill`: `include/vt/ops.h:2871`, vendored FLA Triton-AOT cubins.
- **Both are opt-in.** `VT_KIMI_DEVICE_KDA` and `VT_KIMI_DEVICE_KDA_CHUNK` are
  default OFF (`docs/ENVIRONMENT.md:166`, `:171`); the default production path is
  the host reference compose (`src/vllm/model_executor/models/kimi_linear.cpp:13-21`).
- KDA-specific host references: `src/vllm/model_executor/models/kimi_kda.cpp:24`
  (`KdaLowRankDecay`), `:60` (`KdaDecayGate`).
- **AOT specialization limits:** `src/vt/cuda/cuda_gdn.cu:5217-5220` and
  `:5277-5278` require `dk == dv == 128`, `hk_n == 16`, `hv_n ∈ {48, 32}`.
  GLM-5.3-Flash asks for **64 heads**, which falls off every specialization onto
  the hand C++ kernel (`:5195-5199`). Not a refusal; a speed cliff.
- `gate_lower_bound` is **not implemented anywhere in the tree** — it appears
  only in `.agents/specs/kimi-k3.md:48,:123` and
  `.agents/specs/kda-kernel-delta.md:101`. GLM-5.3-Flash sets it to `-5.0` and
  §Port map shows it selects a *different formula*, not a clamp.

### MoE — the routing rule this model needs is already live

`vt::MoeRouterTopK` args at `include/vt/ops.h:1391-1410` carry `top_k`,
`renormalize`, `scoring_func`, `num_expert_group`, `topk_group`,
`routed_scaling_factor`. Scoring funcs are exactly `kSoftmax` and `kSigmoid`
(`include/vt/ops.h:1378-1381`). Grouped `noaux_tc` is a separate kernel
(`:1385-1390`) with `e_score_correction_bias` as the trailing bias
(`:3132`, `:3143`). CUDA `src/vt/cuda/cuda_moe.cu:64`, `:238`, grouped `:350`,
order `renormalize` **then** `routed_scaling_factor` at `:519-526`, registered
`:935`. `n_group = topk_group = 1` is live and gated on GLM-4.7-Flash.

Clamped SwiGLU already exists and is already fused into the keep-quant grouped
epilogue: semantics `include/vllm/model_executor/models/deepseek_v4_moe.h:148-166`,
host `src/vllm/model_executor/models/deepseek_v4_moe.cpp:105`, CUDA
`src/vt/cuda/cuda_deepseek_v4.cu:1053`, fused form `include/vt/ops.h:1846-1859`.
**Read `:588-592` before using it:** a zero limit means "clamp to zero" in one
form and "no clamp" in the other, a recorded semantic divergence.

Expert GEMM ops: `kMoeGroupedGemmBf16` (`include/vt/ops.h:202`),
`kMoeGroupedGemmBf16GateUpSilu` (`:335`), `kMoeGateUpSwiGLUGrouped` (`:305`),
`kMoeCombine` / `kMoeCombineGate` (`:108`, `:160`), `kSharedExpertGate` (`:159`).
No expert-count ceiling was found; 288 is a config value.

### Shared seams

| Seam | Definition | Example router |
|---|---|---|
| `vt::FusedChain` | `include/vt/ops.h:2517`, `:2531`, `:2544`, `:2551` | `src/vllm/model_executor/models/kimi_linear.cpp:31-33` |
| `layers::MlpGateUpMethodBase` | `include/vllm/model_executor/layers/linear.h:82` | `src/vllm/model_executor/models/gemma.cpp:131` |
| `vt::MergedGemmGroup` | `include/vt/merged_gemm.h:44`, `:64`, `:94` | `include/vllm/model_executor/models/dense_fp8_block_gemm.h:490` |
| `ModelRegistry::Forward` | `src/vllm/model_executor/models/model_registry.cpp:376` | `src/vllm/v1/worker/gpu/runner.cpp:1827` |
| `dense_attn::AttnBlock` | `include/vllm/model_executor/models/dense_attn_block.h:361` | `src/vllm/model_executor/models/qwen3.cpp:185` |
| on-device sampling | `include/vllm/v1/sample/sampler.h:72`, `:96` | `ForwardLogits::on_device()`, `include/vllm/model_executor/models/qwen3_5.h:117` |
| ABI | `include/vllm.h` (v23 at `:297`) | `mmproj_path` `:273`, mm limits `:170-185` |

Note before planning: most models deliberately do **not** route attention
through `dense_attn::AttnBlock` and each says why in a header comment — GLM-4
`src/vllm/model_executor/models/glm4.cpp:48` is the family precedent. A GLM-5.3
attention block will be its own function, and that is the established pattern
rather than a seam bypass.

### Vision — one production ViT, and it is the wrong patch size

`src/vllm/model_executor/models/qwen3_vl_vision.cpp` is the only production
ViT-style encoder for a text LLM (patch embed `:306`, `:385`; merger `:206`,
`:262`). Its config
(`include/vllm/model_executor/models/qwen3_vl_vision.h:35-49`) already matches
GLM-5.3-Flash on `hidden_size 1024`, `num_heads 16`, `depth 24`,
`temporal_patch_size 2`, `spatial_merge_size 2` — and differs on `patch_size`,
which defaults to **16** there and is **14** here
(`include/vllm/multimodal/qwen3vl_processor.h:31`). Both are fields, not
constants, so 14 is a config change; nothing in the tree exercises it. mm
preprocessing lives in `src/vllm/multimodal/qwen3vl_processor.cpp` (image
contract `qwen3vl_processor.h:82`, video `:93-103`, smart_resize `:58`).

### MTP — supported, and this model's head is out of scope

`docs/SPECULATIVE-DECODING.md:41` accepts `mtp`; implementation
`src/vllm/model_executor/models/qwen3_5_mtp.cpp`. The GLM-4.7-Flash precedent is
to **skip** the nextn tail rather than run it
(`src/vllm/model_executor/models/glm4_moe_lite_registry.cpp:21-26`,
`allow_mtp_tail = true`). This campaign does the same and records the head as
owed.

### GGUF — reader only, no converter anywhere in the tree

The ggml type table is `src/vllm/model_executor/model_loader/gguf_reader.cpp:200`
(`FindGgmlTraits`), one `case` per id. Ids with traits: 0 F32, 1 F16, 2 Q4_0,
8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 16 IQ2_XXS, 18 IQ3_XXS,
19 IQ1_S, 21 IQ1_XXXS, 22 IQ2_S, 23 IQ4_XS, 24-28 I8/I16/I32/I64/F64, 30 BF16,
39 MXFP4, 40 NVFP4, 4x Q1_0. **Traits are not dequant:** the dequant dispatch
(`gguf_dequant.cpp:89-160`) covers 0, 1, 2, 8, 10, 11, 12, 13, 14, 16, 18, 19,
22, 26, 30, 39, 40, 66 — so **IQ4_XS (23) and IQ1_XXXS (21) have traits and no
dequant path**, and a plan that assumes "it's in the table so it loads" is
wrong. Verify per type before choosing an arm.

`scripts/` contains no safetensors→GGUF converter: `gen-gguf-nvfp4-goldens.py`
only reads GGUF, and the `check-*-gguf-namemap.py` / `gen-*-gguf-manifest.py`
scripts consume existing files.

### Registration

`REGISTER_VLLM_MODEL` at `include/vllm/model_executor/models/model_registry.h:532`
(seam doc `:502-523`). The closest structural template is
`src/vllm/model_executor/models/glm4_moe_lite_registry.cpp`, which registers
GLM-4.7-Flash by composing DeepSeek-V2's forward and loader over the same
weights and changing three things (`:18-38`). The canonical refuse-by-name
forward is `src/vllm/model_executor/models/kimi_k3.cpp:44-51`, used at `:54`:
name the architecture, name each missing primitive, name the owning row, cite
the spec.

## Design

### What the model IS

Read from `config.json` at `zai-org/GLM-5.3-Flash`, live 2026-08-26.
`architectures: ["Glm5NextForConditionalGeneration"]`, `model_type: glm5_next`,
`transformers_version: 5.16.0`.

Text (`text_config`, `model_type: glm5_next_text`):

| key | value |
|---|---|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 45 |
| `intermediate_size` (dense MLP) | 12288 |
| `num_attention_heads` / `num_key_value_heads` | 64 / 64 |
| `vocab_size` | 154880 |
| `max_position_embeddings` | 1048576 |
| `rms_norm_eps` | 1e-05 |
| `hidden_act` | silu, with `swiglu_limit` 10.0 |
| `tie_word_embeddings` | false |
| `layer_types` | 45 entries: 34 `linear_attention`, 11 `deepseek_sparse_attention` at layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 |
| `mlp_layer_types` | 45 entries: `dense` for layers 0-2, `sparse` thereafter (`first_k_dense_replace: 3`) |
| **MLA** | `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256, `qk_rope_head_dim` **0**, `qk_head_dim` 256, `v_head_dim` 256, `mla_use_nope` true, `head_dim` 0 |
| **DSA indexer** | `index_head_dim` 128, `index_n_heads` 32, `index_topk` 2048, `index_kpool` 4, `index_kpool_compress` true, `index_kpool_always_select_tail` true, `indexer_rope_interleave` true, `index_share_for_mtp_iteration` true, `indexer_types` all `full` |
| **KDA** | `linear_attn_config`: `num_heads` 64, `head_dim` 128, `short_conv_kernel_size` 4, `gate_lower_bound` -5.0, plus explicit `kda_layers` (34) and `full_attn_layers` (11) index lists |
| **mHC** | `mhc` true, `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-06 |
| **MoE** | `n_routed_experts` 288, `n_shared_experts` 1, `num_experts_per_tok` 8, `moe_intermediate_size` 2048, `scoring_func` sigmoid, `topk_method` noaux_tc, `n_group` 1, `topk_group` 1, `routed_scaling_factor` 2.5, `norm_topk_prob` true, `moe_router_dtype` float32 |
| **MTP** | `num_nextn_predict_layers` 1 |

**The text stack has NO rotary embedding anywhere — not in the MLA block and
not in the indexer.** `text_config` carries no `rope_theta` and no
`rope_scaling`; `Glm5NextTextConfig` *deletes* the inherited `rope_parameters`
field outright (`modular_glm5_next.py:167-169` sets it to the `@strict`
`AttributeError()` sentinel), `Glm5NextTextModel.forward` passes
`position_embeddings=None` to every layer (`:1361-1362`), and `position_ids` is
threaded but never read by any sublayer. `Glm5NextTextAttention` has no rotary
call and no `q_pass`/`q_rot` split.

**`indexer_rope_interleave: true` in `config.json` is a VESTIGIAL FLAG.** The
parent `GlmMoeDsaIndexer` applies interleaved rotary
(`parent glm_moe_dsa modeling:232`), and `Glm5NextTextIndexer.forward` overrides
that method wholesale and applies no rotary at all. Implementing it because the
config asks for it is implementing a bug. Positional information in this model
comes exclusively from the 34 KDA layers' channel-wise decay and their short
convs. An implementer who wires a rotary into the MLA block because "MLA has
rope", or into the indexer because the config names one, produces a model that
decodes fluently and is wrong.

Vision (`vision_config`, `model_type: glm5_next_vision`): depth 24, hidden 1024,
`num_heads` 16, `intermediate_size` 4096, `out_hidden_size` 4096,
`projection_intermediate_size` 10240, `patch_size` 14, `image_size` 448,
`spatial_merge_size` 2, `temporal_patch_size` 2, `swiglu_limit` 10.0,
`rms_norm_eps` 1e-05, `attention_bias` true. Placeholder ids: image 154854,
video 154855, image start/end 154830/154831, video start/end 154832/154833.

Processor (`processor_config.json`): CLIP mean/std
`[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`,
`merge_size` 2, `patch_size` 14, `temporal_patch_size` 2, `patch_expand_factor`
1, image tokens 16..8000, video tokens 16..240000, `fps` 2. Tokenizer is a
`tokenizers` backend BPE, `model_max_length` 1048576, eos ids
`[154820, 154827, 154829]`, pad 154820; `generation_config.json` carries
`temperature 1.0`, `top_p 0.95`.

### The four families, and where each one lands

1. **KDA linear attention, 34 layers.** Reuse the Kimi-Linear/GDN machinery.
   Net-new: the forget-gate branch (below) and 64 heads falling off the AOT
   specializations.
2. **DSA MLA, 11 layers.** Reuse the DeepSeek-V4 indexer numerics. Net-new: the
   k-pool compression stage, and a NoPE MLA geometry the shared block currently
   refuses.
3. **mHC residual topology.** Reuse `deepseek_v4_mhc.*` essentially as-is;
   `Glm5NextTextHyperConnection` is `pass` over `DeepseekV4HyperConnection`
   (`modular_glm5_next.py:364-366`). One documented difference, and it is in the
   *head*: `Glm5NextTextHyperHead.forward` is
   `hidden_streams.mean(dim=2)` — an **unweighted mean**, whose own docstring
   says "Unlike DeepSeek-V4" (`modular_glm5_next.py:368-373`). Our
   `HcHeadCollapse` (`deepseek_v4_mhc.cpp:168`) implements the DeepSeek-V4
   weighted collapse (weight-free RMSNorm → `hc_head_fn` → sigmoid gate →
   weighted sum). Using it here is wrong and a token gate on a 4-stream mean
   versus a gated sum will not obviously look wrong; it will look slightly off.
4. **MoE.** Reuse `vt::MoeRouterTopK` grouped `noaux_tc` and the clamped-SwiGLU
   grouped epilogue. The router is `pass` over `DeepseekV3TopkRouter`
   (`modular_glm5_next.py:350-351`).

Plus the vision tower, which is a GLM-OCR ViT
(`Glm5NextVisionModel(GlmOcrVisionModel)`, `modular_glm5_next.py:1410`) and has
no counterpart in this tree — the nearest is Qwen3-VL's, which differs in patch
size and in whichever GLM-OCR specifics wave 6 has to establish.

## Port map

`OURS` is a target path in this tree; `<-` names the upstream anchor. Every
upstream anchor is `transformers v5.16.1`
`src/transformers/models/glm5_next/modular_glm5_next.py` unless stated. Anchors
were read at that tag on 2026-08-26; a wave that finds one stale re-reads it and
corrects this table in the same change.

| OURS (planned) | <- upstream | reuse or net-new |
|---|---|---|
| `glm5_next_config.{h,cpp}` | `:92-247` (`Glm5NextTextConfig`), `:248-264` (vision), `:265-316` (top) | net-new, mechanical |
| `glm5_next_registry.cpp` | registry contract is ours | pattern from `glm4_moe_lite_registry.cpp:18-38` |
| KDA arm | `:597-748` (`Glm5NextTextLinearAttention`), `:441-494` (recurrent), `:495-596` (chunk) | REUSE `vt::KdaGatedDeltaRule` / `KdaChunkPrefill` |
| KDA forget gate | `:375-408` (`Glm5NextTextForgetGate`) | **NET-NEW BRANCH — see below** |
| KDA output norm | `:409-428` (`Glm5NextTextRMSNormGated`) | REUSE, but strict-fp32 cast |
| `l2norm` | `:429-440` | REUSE; `sqrt(sum(x*x) + eps)`, eps 1e-6, NOT `max(..., eps)` |
| DSA indexer | `:749-1024` (`Glm5NextTextIndexer`) | REUSE `deepseek_v4_dsa.cpp:14,31,72` + **NET-NEW k-pool** |
| MLA block | `:1025-1141` (`Glm5NextTextAttention`) | EXTEND `mla_attention.*` for `qk_rope_head_dim == 0` |
| mHC pre/post/sinkhorn | `:364-367` -> `DeepseekV4HyperConnection` | REUSE `deepseek_v4_mhc.cpp:23,72,149` |
| mHC head collapse | `:368-374` (`Glm5NextTextHyperHead`) | **NET-NEW — unweighted mean, NOT `HcHeadCollapse`** |
| MoE router | `:350-353` | REUSE `vt::MoeRouterTopK` grouped `noaux_tc` |
| MoE experts | `:335-349` (`Glm5NextTextExperts`) | REUSE `kMoeGateUpSwiGLUGrouped` clamped epilogue |
| dense MLP | `:321-334` (`Glm5NextTextMLP`) | REUSE clamped SwiGLU; note the asymmetry below |
| decoder layer | `:1142-1208` | net-new control flow |
| text model | `:1285-1372` | net-new |
| vision tower | `:1373-1421` | net-new, nearest `qwen3_vl_vision.cpp` |
| mm wrapper | `:1563-1703` | net-new, nearest `qwen3_vl.cpp` |
| processor | `processing_glm5_next.py`, `image_processing_glm5_next.py`, `video_processing_glm5_next.py` | net-new, nearest `qwen3vl_processor.cpp` |

### The forget gate is a DIFFERENT FORMULA, not a clamp

This is the finding most likely to be got wrong, so it is written out.
`Glm5NextTextForgetGate.forward` (`modular_glm5_next.py:389-407`) computes the
low-rank projection exactly as Kimi KDA does — `f_b_proj(f_a_proj(x))`, add
`dt_bias`, reshape to `[..., H, D]`, `decay_rate = exp(A_log)` — and then
**branches**:

```python
if self.safe_gate_lower_bound is not None:
    return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
return -decay_rate * g_softplus
```

`safe_gate_lower_bound` is `config.linear_lower_bound`, and the published
checkpoint sets `linear_attn_config.gate_lower_bound = -5.0`, which is **not
None**. So GLM-5.3-Flash takes the FIRST branch:
`g_out = -5.0 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`.

Our `KdaDecayGate` (`src/vllm/model_executor/models/kimi_kda.cpp:60`, softplus
at `:15`) implements the SECOND branch — `-exp(A_log) * softplus(g + dt_bias)`,
which is Kimi-Linear's. **The two are different functions of the same inputs.**
Both are smooth, both are negative, both produce a plausible decay, and both
produce fluent text. A model built on the wrong one is wrong everywhere and
looks fine. Note also the sign: `decay_rate = +exp(A_log)` in the sigmoid branch,
where the softplus branch negates it. Wave 2's red-first test is this branch.

### The k-pool indexer is a compression stage DeepSeek-V4 does not have

`Glm5NextTextIndexer` inherits `GlmMoeDsaIndexer` — the DeepSeek-V3.2 lightning
indexer whose numerics `deepseek_v4_dsa.cpp` already ports — and adds
`index_kpool = 4`, `index_kpool_compress = true`,
`index_kpool_always_select_tail = true`. The candidate set the top-`2048`
selection runs over is therefore **pooled keys, not raw tokens**, with the tail
always kept. Two consequences the wave that implements it must design for, both
recorded here because they are the same shape as a trap already recorded on the
`MODEL-MM-QWEN4-EXP` row:

- Feeding raw-token candidates into the top-k, or pooled candidates into a
  consumer expecting raw ones, yields plausible tokens either way.
- **A short-prompt gate cannot catch it.** With `index_topk = 2048`, any context
  at or below 2048 candidate positions selects everything, so the selection is
  the identity and the pooling is unobservable. Any indexer gate must run past
  that threshold. §Gates makes this a requirement, not a note.

**The pooling operator is LEARNED, per channel, and it is not a mean.** Read at
`modular_glm5_next.py:897-970` (`get_pooled_states`):

```
P            = ceil(kv_len / kpool)                       # kpool = 4
first_key[b] = argmax(valid_keys[b])                      # index of the first NON-PAD token
member(b,p,j)= first_key[b] + p*kpool + j
logits[b,p,j,c] = fp32(gate_scores[member][c]) + fp32(ape[j][c])   # -inf on invalid members
prob            = nan_to_num(softmax(logits, dim=j))      # 128 INDEPENDENT 4-way softmaxes
pool_key[b,p,c] = sum_j prob[b,p,j,c] * key[member(b,p,j)][c]
```

`index_kpool_compress_ape` is a learned `[kpool, head_dim] = [4, 128]`
intra-pool absolute-position embedding; `index_kpool_compress_gate` is a
`[128, 4096]` weight applied as `F.linear(x, W)` and its output is **cached per
token** (see the packed cache below). `index_kpool_compress: true` selects this
path and the reference implements no mean-pool fallback, so a `false` value is
not expressible.

Four more facts that are the actual work, none of which DeepSeek-V4 has:

- **The pool grid is batch-dependent.** Pools start at the first *valid* token,
  not at slot 0 (`:938-945`), so a left-padded row groups differently from an
  unpadded one.
- **A pool must be complete to be a candidate** (`pool_valid[b,p]` = all four
  members valid), and a pool is *visible* iff its LAST member is visible to the
  query (`:830-842`). Invalid candidates are masked with `finfo(dtype).min`, not
  `-inf`.
- **The budget is `select_k = min(index_topk // index_kpool, P) = min(512, P)`**,
  and `validate_architecture` enforces `index_topk % index_kpool == 0`
  (`:234-235`). The selected pools are then expanded back to raw token indices —
  `[B,S,512,4]` flattened to `[B,S,2048]` (`:850-862`).
- **The ragged tail is appended raw and UNSCORED** when
  `index_kpool_always_select_tail` is set (`append_visible_tail`, `:972-1022`),
  widening the output to `index_topk + kpool - 1 = **2051**`, not 2048. `-1` is
  the invalid sentinel throughout, and duplicate indices are possible and are
  absorbed downstream by `scatter_add_` + `ne(0)`.

**The indexer cache is packed and wider than DeepSeek-V4's.** The parent caches
`k` only (128 floats/token/layer); this caches
`concat[k(128), gate_scores(128), valid(1)]` = **257 floats/token/layer**
(`:798-808`), because pools are re-formed over the whole history on every call.
The §Hardware KV arithmetic uses the compressed figure and W5 must re-derive it
against the real allocator.

**This round trip is the genuinely new idea in the model: score coarse, attend
fine, always keep the ragged tail.** DeepSeek-V4's compressed-sparse-attention
indexer selects compressed blocks and then *attends to the compressed entries*;
this selects pools and then attends to the **raw** tokens they cover. The
pooling *kernel* itself is not novel — `DeepseekV4HCACompressor` already does a
softmax-over-window weighted pool — so the port reuses that shape and must not
reuse its `compress_rate`, its post-pool RMSNorm, or its RoPE, none of which
appear here.

### The clamped SwiGLU is asymmetric, and we already have the asymmetry

`Glm5NextTextMLP.forward` (`:326-333`) clamps `gate` with `max` only and `up`
with both bounds, at `swiglu_limit = 10.0`; `Glm5NextTextExperts._apply_gate`
(`:341-348`) does the same and then plain `silu(gate) * up` with no alpha or
beta. That is exactly the semantic our
`include/vllm/model_executor/models/deepseek_v4_moe.h:148-166` documents with
`alpha = 1`, `beta = 0`. Reuse it, and read `include/vt/ops.h:588-592` first for
the zero-limit divergence between the two forms.

The same clamp appears in **five** places, and dropping it or symmetrising it in
any one of them is a silent defect: the dense MLP (layers 0-2, width 12288), the
shared expert (width 2048), the routed experts, the vision MLP, and the vision
patch merger.

### Constants and layouts a port gets silently wrong

Collected here because each one is a plausible default that is not this model's
value.

| item | value here | the wrong-but-plausible value |
|---|---|---|
| text RMSNorm eps | **1e-5** (`rms_norm_eps`) | 1e-6 — the `GlmMoeDsa` parent constructs its MLA LoRA norms with the default 1e-6; `Glm5NextTextAttention.__init__` passes `eps` explicitly (`modular_glm5_next.py:1029-1032`) |
| `hc_eps` | **1e-6** | 1e-5 — it is a *different* constant from `rms_norm_eps` and is added to every Sinkhorn denominator, not used as a floor |
| `l2norm` eps | **1e-6**, inside the sqrt | `F.normalize`'s `max(norm, eps)` — the reference deliberately matches FLA's Triton `sqrt(sum + eps)` (`:429-437`) |
| KDA output-norm activation | **sigmoid** | silu — both Qwen3.5-GDN and FLA's KDA use silu here (`:409-412`) |
| KDA output-norm eps | **1e-5** (`rms_norm_eps` passed in at `:635`) | 1e-6, the constructor default |
| indexer `k_norm` | **`nn.LayerNorm(128, eps=1e-6)` WITH bias** | an RMSNorm — the checkpoint carries `indexer.k_norm.bias`, which settles it |
| `first_k_dense_replace` | the literal **3**, hardcoded at `:176` | reading the config key — the runtime class declares no such field and `__post_init__` never names it; the inherited attribute is deleted in `modular_glm5_next.py:169`, so the checkpoint's `first_k_dense_replace: 3` is an inert extra kwarg that happens to agree |
| per-layer schedule | the top-level **`layer_types`** list | `linear_attn_config.kda_layers` / `full_attn_layers` — the reference **ignores** both lists |
| `index_kpool` | **4** (from the checkpoint) | 16, the config class default |
| vision `out_hidden_size` | **4096** | 1536, the config class default |
| vision `image_size` | **448** | 336, the config class default |
| vision merger context dim | **`projection_intermediate_size` = 10240** | `out_hidden_size * in_channels` = 12288, which is the `GlmOcr` parent's rule |
| router selection vs weighting | select on **biased** scores, weight with **unbiased** ones, normaliser `+ 1e-20` | using one set for both |
| group routing | `n_group = topk_group = 1` makes the group stage a **no-op**; skip it | implementing masked group selection that cannot change the result |

Checkpoint layout differs from the reference's packing in three places, and the
loader has to bridge each:

- **KDA convs.** The reference uses ONE grouped `nn.Conv1d` over the
  concatenated `[q; k; v]` channel axis (`in = out = groups = 3 * 8192 = 24576`,
  kernel 4, **bias-free**). The checkpoint stores **three separate depthwise
  convs**: `self_attn.{q,k,v}_conv1d.weight`. Concatenate in **q, k, v** order,
  or run three convs.
- **Experts.** The reference packs `gate_up_proj: [288, 2*2048, 4096]` with gate
  in the first half. The checkpoint stores per-expert
  `mlp.experts.{0..287}.{gate,up,down}_proj.weight`, FP8 block-quantised with
  `weight_scale_inv` companions.
- **Hyper-connections.** The reference's module paths are
  `attn_hc.{fn,base,scale}` / `ffn_hc.*`; the checkpoint stores them **flat on
  the layer** as `hc_attn_{fn,base,scale}` and `hc_ffn_{fn,base,scale}`, and
  `hyper_connection` is in the FP8 `modules_to_not_convert` list so they ship
  unquantised. **There are no `hc_head.*` tensors at all**, which independently
  confirms the unweighted-mean head — do not allocate them.

Two more, in the layer body:

- **`comb` is consumed TRANSPOSED.** The mix is
  `out[k] = sum_j comb[j, k] * residual[j]`, summing over the *first* hc axis
  (`:1194-1203`). The Sinkhorn result is doubly stochastic but **asymmetric**, so
  transposing it wrongly degrades quality silently instead of crashing.
- **`post` and `comb` are produced in fp32 and cast to the activation dtype
  BEFORE the mix**, at both the attention and the FFN site. Keeping them fp32
  through the mix does not reproduce the reference.

### MLA: cache the latent, do not cache what the reference caches

The reference caches the **expanded** `K [B, 64, S, 256]` and `V [B, 64, S, 256]`
— 32,768 values per token per layer. Across 11 DSA layers that is not a design
this project should copy. The standard MLA absorption applies here and is
**trivially valid precisely because the model is NoPE**: with `qk_rope_head_dim
== 0` there is no rope slice that has to stay outside the absorption, so
`kv_b_proj` folds into `q_b_proj` and the cache holds the 512-wide latent. The
§Hardware KV arithmetic assumes the absorbed form, and W3 owns proving the
equivalence rather than asserting it.

### The MTP block is in the checkpoint, and the reference throws it away

`num_hidden_layers` is 45, and the checkpoint carries **46** layer directories.
`model.language_model.layers.45.*` is a DeepSeek-V3-style MTP block: `enorm`,
`hnorm`, `eh_proj` (`[hidden, 2*hidden]`), a full MLA layer **with its own
indexer**, a full 288-expert sparse MoE, and `shared_head.norm` — it reuses
`lm_head` and has no head of its own. The transformers reference discards it:
`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]`
(`modular_glm5_next.py:1235`). `num_nextn_predict_layers` and
`index_share_for_mtp_iteration` are not config fields there; they arrive as
inert kwargs.

Two structural facts worth having on record before O2 is ever picked up: the MTP
block is **DSA/MLA, not KDA**, and it carries **no `hc_*` tensors**, so it runs
on a single residual stream rather than the four-stream manifold.

**Consequence for W7:** those weights are a large share of the checkpoint (a
46th layer whose MoE alone is 288 experts) and the converter should skip them,
following the `glm4_moe_lite_registry.cpp:21-26` precedent of requesting only
`[0, num_hidden_layers)`.

### Image and video share ONE token id

`image_token_id` is 154854 and `video_token_id` is 154855, and **the processor
emits 154854 for both**. Video is written frame by frame as
`<|begin_of_image|>{<|image|> * n}<|end_of_image|>{ts:.1f} seconds`, with the
whole run wrapped in `<|begin_of_video|>` / `<|end_of_video|>` by the chat
template. Disambiguation is by SPAN, not by id:
`in_video_span = cumsum(id == video_start) > cumsum(id == video_end)`, then
`image_mask = is_mm & ~in_video_span` (`modular_glm5_next.py:1477-1485`, which
calls this out as "the core difference to other VLMs"). A port that keys off the
token id gets every video frame classified as an image.

Also in the vision path, and both differ from Qwen3-VL: the rope is **2D (h, w)
only, with no temporal axis** — `get_video_features` rewrites `video_grid_thw`
from `[t, h, w]` into `t` rows of `[1, h, w]` so each frame is an independent
`t = 1` image — and it uses **half-split `rotate_half`**, not the interleaved
form the GLM-MoE-DSA text side uses. There is no mrope: `Glm5NextModel.__init__`
does `del self.rope_deltas` (`:1429`).

## Dependencies

| # | Dependency | State | Effect |
|---|---|---|---|
| D1 | MLA `qk_rope_head_dim == 0` | refused at `mla_attention.cpp:90-93` | blocks every DSA layer; wave 3 |
| D2 | KDA device path default-OFF | `docs/ENVIRONMENT.md:166`, `:171` | first arm runs the host compose; speed is not a wave-1..5 claim |
| D3 | KDA AOT specializations exclude 64 heads | `cuda_gdn.cu:5217-5220`, `:5277-5278` | hand-kernel fallback; a named speed residual, not a blocker |
| D4 | DSA device entry points are host-vector | `deepseek_v4_device.h:82-102` | upload/download per call; a named speed residual |
| D5 | No GGUF converter in tree | CLOSED by W7a: `scripts/convert-glm5-next-gguf.py` | — |
| D6 | llama.cpp has no `glm5_next` | holds at `master` `539f24529` and at the pin; but the KDA/MLA/indexer KEYS and NAMES it needs are all present at `b10451` | no llama.cpp quant oracle and no floor for the GGUF arms; the CONVENTION is available without a pin advance |
| D7 | transformers lane pin at `v5.16.1` | `.agents/oracles/transformers.md` pins 5.14.1 | wave 0 writes the lane pin; `gateable = no` |
| D8 | Live seam contention | PRs [#1971](https://github.com/mudler/vllm.cpp/pull/1971) (DSA geometry), [#1977](https://github.com/mudler/vllm.cpp/pull/1977) (DSv4 KV multicache) | waves 3 and 5 rebase onto whichever lands first; do NOT fork the seam |

D8 is scheduling, not a blocker, and it is why the wave order below puts config
and KDA before the DSA and KV work.

## Work breakdown

Eight waves. Each is a separate `row/MODEL-MM-GLM53-FLASH-W<n>` branch, a
separate pull request, a fresh implementer and a fresh reviewer. **W0-W2 and W4
are CPU-gateable and need no GPU at all. W3 and W5-W8 need a GPU.** Sizes are
the author's estimate of reviewable diff, not a budget.

### W0 — records and the lane oracle pin (CPU, small)

Write `.agents/oracles/transformers.md`'s lane-scoped `v5.16.1` pin for this row
with `gateable = no` and the issue that owes the measurement. **Measure what
`scripts/check-oracle-pins.py` reads of that block rather than assuming it reads
anything: W0 measured that it reads nothing.** The checker's `BLOCK` regex is
```` ^```oracle-pin\n ````, so an `oracle-pin-lane` fence never matches it and
the lane pin is unchecked prose (**O13**,
[#2099](https://github.com/mudler/vllm.cpp/issues/2099)). **Deliverable:** the
pin and nothing else. **Exclusion:** no model code, and no checker. **Gate:**
`agent-preflight.sh` green, which is the whole of W0's gate: the checker stays
at exit 0 whether the lane block is correct, corrupt or deleted outright.
**Stop:** if the checker refuses a second lane pin, return `NEEDS_DECISION`
rather than editing the checker.

### W1 — config, registration, refuse-by-name (CPU, medium) — [#2067](https://github.com/mudler/vllm.cpp/issues/2067)

Resolve `glm5_next`'s nested `text_config` / `vision_config` /
`quantization_config`; register `Glm5NextForConditionalGeneration`; enumerate
the 76,108-tensor weight name map structurally; make `Forward` refuse by name
naming every unimplemented primitive and this spec. **And wire the
`general.architecture` dispatch entry that discharges O9**, so the file W7a's
converter writes is opened by this tree rather than refused by name.
**Scope:** `glm5_next.{h,cpp}`, `glm5_next_registry.cpp`,
`glm5_next_weights.{h,cpp}`, one row in `kGgufArchArms`
(`entrypoints/model_loader.cpp`).
**Exclusions:** no forward math, no kernels, no loader materialization,
**and no relaxation of `MlaBlockDims::Validate`** — that stays W3's, and W1's
refusals name it so a later wave lands on the right file.
**Anchors:** `modular_glm5_next.py:92-316`; pattern
`glm4_moe_lite_registry.cpp:18-38`; refusal pattern `kimi_k3.cpp:44-51`;
GGUF-family-owns-its-builder pattern `qwen4_exp_gguf_weights.{h,cpp}`.
**Tests:** the C++ tensor name map equals the converter's table for table,
gated inside `test_convert_glm5_next_gguf.py` -- no upstream tool writes this
container, so nothing external pins the spellings and two hand-maintained copies
of one mapping is the shape that drifts silently; registry resolve; config
descent including the
`linear_attn_config` → `linear_*` key remap and the two per-layer index lists;
the 45-entry `layer_types` / `mlp_layer_types` split; refuse-by-name message
names each missing primitive; a `glm5next` GGUF reaches
`Glm5NextHfConfigFromGguf` through `LoadedEngine::FromModelDir`.
**Gate:** CPU build `-DVLLM_CPP_CUDA=OFF`, focused ctest, full preflight.
**Evidence:** the registry contract test's architecture count moves by exactly
one. **Reachability:** the registration is reached from
`ModelRegistry::Forward` and the GGUF arm from `LoadedEngine::FromModelDir`;
deleting the `REGISTER_VLLM_MODEL` line, or the `kGgufArchArms` row, must red
the focused gate.

**Two in-flow bugs, both filed and both fixed in this wave.**
[#2070](https://github.com/mudler/vllm.cpp/issues/2070): the shared config
reader synthesizes `layer_types` from `linear_attn_config.kda_layers` under
Kimi-Linear's ONE-INDEXED rule, and this model's list is ZERO-INDEXED, so a
`glm5_next` config without an explicit `layer_types` came out shifted by one
with a third of the stack on the wrong attention kind — from a list the
reference ignores entirely. Fixed by resolving the schedule from this model's
own `text_config`; the Kimi branch is untouched, because it is correct for the
family it was written for. And `index_topk_pattern`, upstream's FIRST fallback
for an absent `indexer_types` (`configuration_glm5_next.py:177-190`), was
neither mirrored nor refused, so a pattern config resolved to a different
indexer schedule silently; both upstream spellings are now implemented.

**One parser, two sources, and that is the reachability design rather than a
convenience.** `Glm5NextHfConfigFromGguf` does not build a second, parallel
notion of what a `glm5_next` config is. It reads the GGUF metadata the
converter writes and synthesizes an HF-shaped `text_config` /
`vision_config` under the *same key spellings* `config.json` uses, so the GGUF
path descends through `ParseGlm5NextParams` — the same function, the same
validation, the same refusals. A malformed GGUF is therefore refused by the
code that refuses a malformed `config.json`, and a divergence between the two
sources is a compile-time impossibility rather than a test's responsibility.

**Upstream REQUIRES the geometry our MLA block refuses, and the polarity is
worth writing down.** `Glm5NextTextConfig.validate_architecture`
(`configuration_glm5_next.py` @ transformers `v5.16.1`) raises
`"Expecting NoPE for the DSA attention layers, but got {qk_rope_head_dim} as
RoPE dim."` when `qk_rope_head_dim > 0`. Our `MlaBlockDims::Validate`
(`src/vllm/model_executor/layers/attention/mla_attention.cpp:90-93`) raises when
it is not `> 0`. The two validators are exact complements over this field, and
there is no value that satisfies both. W1 mirrors upstream and accepts `0`;
W3 owns making the MLA block agree.

### W2 — the KDA arm's numerics (CPU, medium) — [#2097](https://github.com/mudler/vllm.cpp/issues/2097)

Port `Glm5NextTextForgetGate`'s **sigmoid branch**, the strict-fp32
`RMSNormGated`, and `l2norm`, as portable host references with an independent
double-precision derivation, exactly as `kimi_kda.cpp` did for the softplus
branch. Wire the KDA layer's q/k/v short convs (`short_conv_kernel_size = 4`)
and the delta recurrence onto the existing `vt::KdaGatedDeltaRule` /
`KdaChunkPrefill` seams.
**Scope:** `glm5_next_kda.{h,cpp}` plus the seam call sites. Note three
layout facts the wave must handle: the checkpoint stores **three separate
depthwise convs** where the reference has one grouped conv over `[q; k; v]`
(concatenate in q, k, v order); `g`, `beta` and the output gate are all computed
from the **pre-conv** hidden states, so they must not be fused into the conv
path; and the KDA cache is a `[B, 24576, 4]` conv state plus a **fp32**
`[B, 64, 128, 128]` recurrent state, 4 MiB per layer per sequence, ~136 MiB
across the 34 layers.
**Exclusions:** do not touch `src/vt/cuda/cuda_gdn.cu`; do not touch
`kimi_kda.cpp` (Kimi-Linear's gate is the other branch and stays).
**Anchors:** `modular_glm5_next.py:375-408`, `:409-428`, `:429-440`, `:597-748`.
**Tests, RED FIRST:** a case whose expected value is computed from the sigmoid
branch and which FAILS against `kimi_kda.cpp:60`'s softplus branch. That failure
is the deliverable's proof.
**Gate:** CPU, focused ctest + full preflight. **CPU-gateable.**

### W3 — NoPE MLA and the DSA k-pool indexer (GPU, large)

Extend `MlaBlockDims::Validate` to accept `qk_rope_head_dim == 0` with
`head_size() == kv_lora_rank`, and thread it through the decode and prefill
paths; resolve and implement the k-pool compression and tail-keep in the
indexer.
**Scope:** `mla_attention.{h,cpp}`, `cuda_mla_attn.cu`, `cuda_mla_prefill.cu`,
`glm5_next_dsa.{h,cpp}`.
**Exclusions:** do not change any existing model's resolved geometry. SACRED
inertness against DeepSeek-V2/V3, Kimi-Linear and GLM-4.7-Flash goldens is a
gate, not a hope.
**Anchors:** `modular_glm5_next.py:749-1024`, `:1025-1141`.
**Tests:** NoPE geometry validation, both accept and refuse; a k-pool selection
case at a context **strictly greater than `index_topk`** so the selection is not
the identity; existing MLA goldens byte-identical.
**Needs GPU:** the decode kernel's 512/256 head pair is untested and the
shared-memory guard at `cuda_mla_attn.cu:545-546` can only be checked by
running. **Rebase note:** coordinate with PRs #1971 and #1977.

### W4 — mHC wiring and the unweighted head (CPU, small) — [#2098](https://github.com/mudler/vllm.cpp/issues/2098)

**LANDED 2026-08-27** (`CLAIM-GLM53-FLASH-W4`).

Reuse `MhcSinkhorn` / `MhcPre` / `MhcPost` at `hc_mult = 4`,
`hc_sinkhorn_iters = 20`, `hc_eps = 1e-06`; implement the **unweighted-mean**
head collapse as a GLM-5-specific function and do NOT reuse `HcHeadCollapse`.
**Anchors:** `modular_glm5_next.py:364-374`; ours `deepseek_v4_mhc.cpp:23,72,149,168`.
**Tests, RED FIRST:** a case that passes against the mean and fails against
`HcHeadCollapse`. **CPU-gateable.**

What landed, and the two anchors that needed correcting. `glm5_next_mhc.{h,cpp}`
holds three entry points: `glm5_next::MhcPre` and `glm5_next::MhcPost` wrap
`deepseek_v4_mhc.cpp:72` and `:149` unchanged and exist to bind this model's five
constants in ONE place — `rms_norm_eps` (1e-5, NOT `hc_eps`) for the folded
weight-free RMSNorm, `hc_eps` (1e-6) as BOTH the pre epsilon and the Sinkhorn
epsilon, `hc_post_alpha` 2.0, and `hc_sinkhorn_iters` 20 — and
`glm5_next::HcHeadCollapseMean` is the net-new mean. The name is deliberately
NOT `HcHeadCollapse`: two functions that differ this way should not share a short
name. This spec's `:364-374` is the modular block; the two classes inside it are
`:364-365` (`pass`) and `:368-372`, and the flattened bodies are
`modeling_glm5_next.py:267-295` and `:298-302`. All four of ours resolve as
written.

The discriminator is hand-derivable and needs no tuning: with `fn == 0` and
`base == 0`, V4's gate is `sigmoid(0) + hc_eps` on every stream, so
`HcHeadCollapse` returns `(2 + 4e-6)x` the mean at `hc_mult == 4`. The RED run
read that as `worst := 0` — the wrong reuse and the stub were the same function —
and the gate is 59 of 98 assertions failed. Goldens are the RUN output of
unmodified `Glm5NextTextHyperConnection.forward` and
`Glm5NextTextHyperHead.forward` at transformers `v5.16.1`, captured by
`tests/vllm/models/fixtures/gen_glm5_next_mhc_goldens.py`, which refuses to emit
under any other version. DeepSeek-V4's mHC is inert: its two files are
byte-identical to the base by sha256 and `test_deepseek_v4_mhc`'s 125 assertion
lines are byte-identical before and after. **Not reached from a production entry
point — see O16.**

### W5 — MoE, the decoder layer, and the assembled text forward (GPU, large)

Wire the 288+1 expert MoE through the existing grouped `noaux_tc` router and
clamped-SwiGLU epilogue; implement the per-layer control flow (KDA vs DSA,
dense vs sparse, mHC stream threading); assemble `Glm5NextTextModel::Forward`;
build the heterogeneous KV cache (11 MLA groups + 34 KDA state groups + the
indexer side cache) through `MakeKVCache`, following
`kimi_linear_registry.cpp:135-166`.
**Anchors:** `modular_glm5_next.py:321-363`, `:1142-1208`, `:1285-1372`.
**Needs GPU.** **Rebase note:** the KV grouping overlaps PR #1977 directly.

### W6 — vision tower, processor, mm placeholder expansion (GPU, large)

The 24-layer GLM-OCR-style ViT at patch 14, the patch merger at
`spatial_merge_size 2`, the projection at `projection_intermediate_size 10240`,
and the image/video preprocessing including `fps 2`, `temporal_patch_size 2`,
the 16..8000 image and 16..240000 video token bounds, and the six placeholder
ids. **Anchors:** `modular_glm5_next.py:1373-1421`, `:1563-1703`,
`processing_glm5_next.py`, `image_processing_glm5_next.py`,
`video_processing_glm5_next.py`. **Nearest ours:** `qwen3_vl_vision.cpp`,
`qwen3vl_processor.cpp`.

### W7 — the GGUF converter and the first fitting arm

Split into two, because the two halves have different blockers and only one of
them needs a box. The split is recorded here rather than derived again: W7a is
CPU-only and needs no checkpoint, W7b needs 300–600 GiB of staged weights and
explicit developer authority for a large-asset download.

#### W7a — the converter and its synthetic gate (CPU, large). LANDED

Issue: [#2011](https://github.com/mudler/vllm.cpp/issues/2011).

Author `scripts/convert-glm5-next-gguf.py` (no upstream tool can do it, D5/D6)
and gate it on synthetic tiny-shape fixtures. **Deliverable:** the converter, the
metadata schema, the tensor-name map, the FP8 e4m3 block dequant, the expert
stacking, the arm table with refusals, and
`tests/scripts/test_convert_glm5_next_gguf.py`. **Exclusions:** no artifact
(O7), no i-quant arm (R4), and no Q3_K/Q4_K/Q5_K encoder (O8).

Three findings this wave moved, each of which changes what a later wave should
believe:

**D6 was too strong, and the correction is in our favour.** llama.cpp has no
`glm5_next` — that part holds, verified at `origin/master` = `539f24529`
(fetched 2026-08-26) and at our pin: the arch enumerators are `LLM_ARCH_GLM4`,
`LLM_ARCH_GLM4_MOE` and `LLM_ARCH_GLM_DSA` (`src/llama-arch.h:86-88`), and
`src/models/glm-dsa.cpp` is GLM-5.2, citing `zai-org/GLM-5.2/blob/main/config.json`.
But **everything the converter needs by way of convention already exists AT THE
PIN `b10451`**, and no pin advance is required by any part of this wave:

- `gguf-py/gguf/constants.py:262-264` @ b10451 already carries `class KDA` with
  `{arch}.kda.head_dim` and **`{arch}.kda.gate_lower_bound`**. `KDA.SAFE_GATE`
  is the only member that is `master`-only, and GLM-5.3-Flash declares no
  `safe_gate` key, so nothing here reaches for it.
- The KDA tensor spellings are at `src/llama-arch.cpp:465-479` @ b10451 —
  `ssm_conv1d_q/k/v` (the three separate depthwise convs, exactly the packing
  this checkpoint uses), `ssm_f_a`, `ssm_f_b`, `ssm_g_a`, `ssm_g_b`, `ssm_beta`,
  `ssm_a`, `ssm_dt`, `ssm_norm`.
- `gguf-py/gguf/tensor_mapping.py:896-933` @ b10451 maps them from Kimi-Linear's
  HF module paths, which are **GLM-5.3-Flash's paths verbatim**.
- The indexer names, including `indexer_compressor_ape` and
  `indexer_compressor_gate` for the k-pool stage, are at
  `src/llama-arch.cpp:626-636` @ b10451.

**Upstream Python cannot quantize at all.** `gguf.quants.Q2_K` implements
`dequantize_blocks` and **no** `quantize_blocks`; the Q2_K, Q6_K and Q8_0
encoders exist only in `ggml/src/ggml-quants.c`. They are therefore ported, and
gated byte-for-byte against the pinned C reference over a frozen golden
(`tests/scripts/fixtures/glm5_next_kquant_golden_b10451.json`) rather than
against a tolerance. Two traps changed bytes during the port and are recorded in
the source so the next reader does not re-find them: `nearest_int` is the
`+12582912.0` add-and-mask trick at `:621` and rounds half to EVEN, not
`round`; and C `roundf` in `quantize_row_q8_0_ref` rounds half AWAY FROM ZERO
where `np.rint` rounds half to even, which mis-encodes every exact `.5`.

**The tensor inventory is exact, not inferred.** The real
`model.safetensors.index.json` (8.4 MB, 76,108 entries) and three shard headers
(shards 2, 32 and 62) were read by HTTP RANGE on 2026-08-26, payload never
fetched. They confirm four things §Port map asserted: the three separate
`{q,k,v}_conv1d` convs; `hc_{attn,ffn}_{fn,base,scale}` flat on the layer with
**no `hc_head.*` tensor at any layer**; `indexer.k_norm.bias` present, which
settles LayerNorm-with-bias over RMSNorm; and `index_kpool_compress_ape` /
`index_kpool_compress_gate` present on 12 layers — the 11 DSA layers plus the
MTP block.

#### W7b — the first fitting artifact (GPU + large asset). NOT STARTED

Produce the Q2_K arm and run it. **This wave needs explicit developer authority
for a large-asset download** and is the only wave that does. Owed as O7 with
what it needs named. Its gate is §Gates' W7 row unchanged: the arm loads, the
runner serves it through `include/vllm.h`, and it generates coherent text on a
prompt with an image — a RUN gate, not a token gate.

### W8 — speed, once and only once a correctness gate exists

Not scheduled by this spec. AGENTS.md forbids accepting a performance result
before the declared correctness gate stands, and §Gates explains why the
end-to-end one cannot stand here. W8 opens when W7 closes and its scope is set
then.

## Tests to port

`tests/models/` in transformers `v5.16.1` is the upstream suite. What is
portable and what is not, stated rather than assumed:

- **Config resolution and the layer-type expansions** port directly: the 45-entry
  `layer_types`, `mlp_layer_types`, `indexer_types` lists and the
  `linear_attn_config` → `linear_*` remap are pure functions of the config.
- **The processor tests port**: upstream's own
  `tests/transformers_utils/processors/test_glm5next.py` shape (a 389-line
  processor suite exists in the inadmissible vLLM PR, which is evidence that a
  processor suite is the right unit, not a source to copy). Ours goes against
  the transformers processor at the lane pin.
- **Component numerics do not port as tests; they port as GOLDENS.** transformers
  has no numeric fixtures for these layers. §Gates says how we make them.
- **No end-to-end generation test ports**, because no side can run the model.

Local red-first tests, one per wave, are named in each wave above. Every one of
them must fail for the intended reason before the implementation exists, and the
failure must be captured.

## Gates

### The end-to-end token gate is UNREACHABLE, and saying so is the gate decision

No oracle implements this architecture at a pinned revision except transformers
`v5.16.1`, and running it needs 305.78 GiB (FP8) or 598.5 GiB (BF16). The
largest reachable device is `dgx:gpu0` at ~119.63 GiB unified. **There is no
token-exact end-to-end gate against an oracle for this model on this fleet, and
no amount of implementation work creates one.** Any campaign report claiming one
is wrong. This is recorded as visible debt under §Owed, not waived.

### What IS reachable, and it is the campaign's spine

**A tiny-shape reference oracle, on CPU, from the pinned transformers.** A
randomly-initialised `Glm5NextConfig` at a small shape — the shape is wave 1's
to fix, on the order of `hidden_size 128`, 4 layers alternating KDA and DSA,
8 experts, `hc_mult 4`, a 32-token vocabulary — instantiates and runs on CPU in
seconds with no weights and no GPU. Dumping per-component activations from it
gives a REAL numerical oracle for every component in waves 1-6, at f32/f64
tolerance, hermetic and seed-reproducible.

Two things this is and is not, stated because the distinction has been got
wrong here before:

- It **is** an oracle for the *numerics*: the reference implementation computes
  the values, not a hand derivation, so a transcription bug in our port is
  caught.
- It is **not** oracle gateability for the *model*. AGENTS.md's bar is that the
  oracle demonstrably builds and runs THE MODEL; constructing a config proves
  nothing. `gateable` stays `no` for `transformers` on this row and the oracle
  file says so.

This is the same shape as the DeepSeek-V4 W3/W5 lane
(`deepseek_v4_dsa.cpp`, `deepseek_v4_mhc.cpp`) and it is strictly stronger,
because those gate against hand-derived cases and this gates against the
reference implementation's own output.

### Binding gates per wave

| wave | gate | CPU or GPU |
|---|---|---|
| W0 | preflight green, and that is the whole gate: NO checker parses an `oracle-pin-lane` block, so nothing validates the lane pin's fields (**O13**, [#2099](https://github.com/mudler/vllm.cpp/issues/2099)) | CPU |
| W1 | registry resolve, config descent, refuse-by-name; architecture count +1; a `glm5next` GGUF reaches its OWN builder through `LoadedEngine::FromModelDir`; preflight | CPU |
| W2 | tiny-shape forget-gate / gated-norm / l2norm goldens; RED-first against the softplus branch | CPU |
| W3 | NoPE MLA accept+refuse; k-pool selection at context **> `index_topk` = 2048**; SACRED inertness on DeepSeek-V2/V3, Kimi-Linear, GLM-4.7-Flash goldens byte-identical | GPU |
| W4 | mHC goldens at `hc_mult 4`; RED-first against `HcHeadCollapse` | CPU |
| W5 | per-layer control-flow goldens; assembled tiny-model forward vs the tiny reference | GPU |
| W6 | processor parity vs the transformers processor at the lane pin; placeholder-expansion goldens | GPU |
| W7 | **the arm below RUNS on `dgx:gpu0` and generates coherent text** | GPU |

### W7's verification target, set by the developer 2026-08-26

**A low GGUF quant that fits on `dgx:gpu0` (GB10, ~119.63 GiB usable).** The
candidates, with the arithmetic and its limits, are in §Hardware. The gate is:
the arm loads, the runner serves it through `include/vllm.h`, and it generates
coherent text on a prompt with an image. It is a **run gate, not a token gate**,
because §Gates' first paragraph holds — nothing can produce the reference
tokens. Do not report it as a correctness gate.

## Hardware

### The measured artifacts

Read live 2026-08-26 from the HuggingFace API, and for the primary repo by HTTP
RANGE over all 62 safetensors headers — payload was never fetched.

| repo | bytes | GiB | vs 119.63 GiB |
|---|---|---|---|
| `zai-org/GLM-5.3-Flash` (FP8 e4m3, block 128x128) | 328,326,771,576 | 305.78 | **2.56x over** |
| `unsloth/GLM-5.3-Flash-FP8` | 328,366,169,538 | 305.82 | 2.56x over |
| `zai-org/GLM-5.3-Flash-BF16` | 642,676,397,788 | 598.53 | 5.00x over |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | 194,692,687,003 | 181.32 | **1.52x over** |

**NOTHING PUBLISHED FITS.** The smallest published artifact is 1.52x the whole
GB10 pool.

**No GGUF exists.** Four repositories are named `*-GGUF` and every one of them
contains **zero `.gguf` files**: `unsloth/GLM-5.3-Flash-GGUF` (README +
`.gitattributes`), `AtomicChat/GLM-5.3-Flash-GGUF` (README + four PNGs),
`aj9o9/GLM-5.3-Flash-GGUF` (README), `vcruz305/GLM-5.3-Flash-GGUF` (README).
A repository name is not an artifact.

### The exact parameter split, from the shard headers

62 shards, 76,108 tensors. By dtype: `F8_E4M3` 314,396,639,232 elements,
`BF16` 6,926,096,640, `F32` 19,484,766 (the block scales — 314.4e9 / 128 / 128 =
19.19M, which is what that count is). **Total real parameters:
321,322,735,872 (321.32B)**, consistent with the card's "320B total".

By group: routed experts **311.65B (97.0%)** across 37,152 tensors; everything
else 6.76B; shared expert 1.08B; embedding 634M; `lm_head` 634M; vision 564M.
That 97% is the single most important number for quantization planning: the
mixed bits-per-weight of any GGUF arm is, to within a percent, the bits-per-weight
of whatever type the experts get.

**The MTP block is inside that total.** `model.language_model.layers.45.*` is
888 tensors and **7.43B parameters, 2.31% of the model** — a 46th layer whose
MoE alone carries 288 experts. Skipping it in the converter (O2, and the
`glm4_moe_lite_registry.cpp:21-26` precedent) removes ~2.3 GiB from a Q2_K arm.
Every arm figure below is stated INCLUDING layer 45, so skipping it is headroom
this table does not already spend.

### GGUF arms — arithmetic, NOT measurement

Bits per weight are from our own reader's block traits
(`gguf_reader.cpp:200`), so they are exact for the type: Q2_K 84B/256 = 2.625,
IQ2_S 82/256 = 2.5625, IQ2_XXS 66/256 = 2.0625, IQ3_XXS 98/256 = 3.0625,
IQ1_S 50/256 = 1.5625, Q4_K 144/256 = 4.5, Q6_K 210/256 = 6.5625, Q8_0
34/32 = 8.5. Mixed arms below put the experts at the named type and the
remaining 3% at Q6_K.

| arm (experts @) | mixed bpw | weights GiB | fits 119.63? | headroom |
|---|---|---|---|---|
| Q8_0 | 8.50 | 318.0 | no | — |
| Q6_K | 6.56 | 245.5 | no | — |
| Q4_K | 4.56 | 170.6 | no | — |
| IQ3_XXS | 3.16 | 118.1 | **no, not really** | ~1.5 GiB — not a margin |
| Q2_K | 2.74 | 102.6 | yes | ~17 GiB |
| IQ2_S | 2.68 | 100.3 | yes | ~19 GiB |
| IQ2_XXS | 2.20 | 82.3 | yes | ~37 GiB |
| IQ1_S | 1.71 | 64.0 | yes | ~56 GiB |

The KV side is small, which is the whole point of a linear-attention hybrid.
Per token: 11 MLA layers x `kv_lora_rank` 512 x 2 B = 11,264 B, plus the indexer
side cache at `index_head_dim` 128 x 2 B / `index_kpool` 4 = 64 B per layer x 11
= 704 B. **~11.7 KiB/token**, so 128K context is ~1.5 GiB and the full 1M
context is ~11.4 GiB. The 34 KDA layers hold a per-sequence recurrent state
rather than per-token KV: 64 heads x 128 x 128 x 4 B x 34 ~ 143 MiB per
sequence, plus conv states. **This arithmetic is unverified against our own
allocator** and #1963/#1966 record that the KV byte accounting has been wrong by
48x before, so W5 re-derives it from the runner rather than from this table.

**W7a superseded this table with the converter's own plan, and the numbers
moved in the right direction.** The table above is bits-per-weight times a
parameter count. The converter resolves a TYPE PER TENSOR, so its plan is the
arithmetic that will actually be written, and running its type resolver over the
real topology gives 1719 output tensors and **313,890,512,702 parameters
carried** — 321.32B minus the 7.43B MTP block, which is the independent check
that the skip is exactly the 2.31% §Port map measured:

| arm | weights | mixed bpw | breakdown |
|---|---|---|---|
| `q2_k` (experts Q2_K, everything else Q6_K) | **100.35 GiB** | 2.746 | Q2_K 93.02, Q6_K 7.17, F32 0.08, Q8_0 0.07 |
| `q6_k` | 239.89 GiB | 6.565 | Q6_K 239.73 |
| `q8_0` | 310.67 GiB | 8.502 | Q8_0 310.58 |
| `bf16` | 584.67 GiB | 16.000 | BF16 584.67 |

The Q2_K arm is **100.35 GiB, not 102.6**, because the table above stated every
figure including layer 45 and the converter drops it. The F32 and Q8_0 slivers
are the fallback ladder: a k-quant needs `ne0 % 256 == 0`, so a row that does not
divide steps down to Q8_0 (block 32) and only falls to F32 when even 32 does not
divide (the depthwise conv kernels at 4) or the tensor is 4-D or 5-D (the
patch-embed and downsample kernels). Stepping down rather than jumping to F32
matters for more than tidiness: with an F32 fallback a FINER arm can come out
LARGER than a coarser one, which is not a property a size table may have.

Against ~119.63 GiB at 128K context and one sequence: weights 100.35, KV **1.43
GiB** (11 MLA layers x `kv_lora_rank` 512 x 2 B, plus an indexer side cache of
`index_head_dim` 128 x 2 B / `index_kpool` 4 = 64 B per layer x 11, so 11,968
B/token), KDA recurrent state **0.14 GiB** (64 heads x 128 x 128 x 4 B x 34
layers) plus conv states. **~17.7 GiB is left** for activations, allocator
overhead and page cache. Still arithmetic and still not measurement: #1963 and
#1966 record this accounting being wrong by 48x, and W5 re-derives it from the
runner.

**Recommended first arm: experts at Q2_K, ~100.35 GiB, with ~17.7 GiB for KV,
activations and page cache at 128K context.** Not IQ2_S or below, and the reason
is not quality — it is that **i-quants need an importance matrix, and an
importance matrix needs a forward pass over the model, which needs 181 GiB
minimum.** The dependency is circular on this fleet. K-quants do not need one.
That makes Q2_K the only fitting arm that is *producible* here, and it should be
stated in W7's scope rather than discovered.

Second-choice, if Q2_K's ~17 GiB proves thin once W5 measures the real KV: drop
the experts to Q2_K and leave the 3% at Q5_K, or accept a shorter maximum
context. Do not reach for IQ2_XXS to buy headroom without first solving the
imatrix problem.

### Three things that make "fits in VRAM" the wrong question

- **GB10 is unified memory.** The 119.63 GiB is the whole pool, not a VRAM
  budget beside a separate host RAM.
- **`gpu_memory_utilization` does not bound host RAM on GB10.** A plan that sets
  it and considers the question closed has not bounded anything.
- **The oracle side has been measured at ~103 GB of host RAM at KV profiling**
  for a large model on an idle box (#1431). Here it is moot — the oracle cannot
  run at all — but the same shape applies to our own loader, and W7 measures
  peak RSS rather than assuming the on-disk size is the resident size.

### Fleet verdict

| device | can run the reference oracle | can run our Q2_K arm |
|---|---|---|
| `dgx:gpu0` (GB10, ~119.63 GiB) | **no** — 305.78 GiB needed | plausible on arithmetic; **not measured** |
| `thor:gpu0` | no | no |
| `orin:gpu0` | no | no |

No lease was taken for this spec and no GPU ran. Every entry above is arithmetic
over measured file sizes.

## Risks

**R1 — the forget-gate branch.** Highest-probability silent defect in the
campaign. Mitigated by W2's RED-first test against the softplus branch. If that
test is not red before the implementation, the wave has not proven anything.

**R2 — the mHC head collapse.** Reusing `HcHeadCollapse` produces a working
model with a wrong final projection. Mitigated by W4's RED-first test.

**R3 — the k-pool selection is unobservable below 2048 tokens.** Any gate at or
under `index_topk` selects everything and passes regardless. W3's gate requires
a context strictly greater than 2048. This is the single most likely way for the
campaign to ship a green gate over a broken indexer.

**R4 — the i-quant/imatrix circularity.** An importance matrix requires running
the model; running the model requires 181 GiB; therefore no i-quant arm is
producible on this fleet. Recorded as a boundary, not a task.

**R5 — the converter is ours to write and has no oracle.** llama.cpp cannot emit
this architecture, so there is no reference GGUF to diff against and no
llama.cpp floor for the arm. W7 must state which side ran what.

**R6 — seam contention.** PRs #1971 and #1977 are live on the DSA-geometry and
DSv4-KV seams. W3 and W5 rebase; they do not fork.

**R7 — vLLM #53906 may merge mid-campaign.** That is the good case and it
changes the oracle. §Stop conditions says what to do; the risk is that a wave
in flight silently keeps using transformers as the mirror source after vLLM
becomes authoritative.

**R8 — 64 KDA heads fall off every AOT specialization.** A correctness-complete
model at an indefensible decode speed. Named as a residual for W8, not
designed around now.

**R10 — image and video share token id 154854.** The processor emits the image
id for video frames and disambiguates by `<|begin_of_video|>` / `<|end_of_video|>`
span. A port that keys off `video_token_id` classifies every frame as an image
and still produces output. W6 gates on a placeholder-expansion golden that
contains both an image and a video in one prompt.

**R11 — the eps values are the parent classes' defaults in five places and this
model overrides them.** The table in §Port map lists them. Each one is a
plausible default that produces a slightly-off model rather than a failure.

**R9 — the campaign can complete without ever being token-gated.** This is not a
risk to mitigate; it is the recorded state of the world (§Gates). The risk is
that a later reader forgets it. Hence §Owed.

### Decisions taken in this spec

**D-a. One matrix row, not three.** vLLM #53906 would register
`Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration` and
`Glm5NextMTPModel`. None of the three is registered at any vLLM revision today,
and the only architecture any published artifact declares is
`Glm5NextForConditionalGeneration` (`config.json`, read 2026-08-26). One row is
what is checkable. If #53906 merges, the text-only and MTP arms get their own
rows in the change that reconciles the pin, and the `MODEL` ratchet moves then.

**D-b. transformers is the mirror source for this row, and only until vLLM
implements it.** Not a preference — it is the only admissible oracle. Every
port-map cell cites transformers. The moment vLLM registers `glm5_next`, this
row reconciles onto vLLM per AGENTS.md §"When vLLM has no implementation", the
lane pin expires, and the spec records the change.

**D-c. Pull-request shape: separate spec and implementation.** Not asked of the
developer. The AGENTS.md split case applies on its face — this pull request
deliberately adds a row, an issue and a spec with no product code, and the
campaign wants scope agreement before eight implementation waves start. Recorded
so it is not re-derived, and NOT written into
`.agents/developer-preferences.md`, because an inference must not be filed where
a developer answer is read.

## Evidence required

Per wave, before its pull request is opened: the RED capture of its named
red-first test, the focused ctest green after, a full `agent-preflight.sh` whose
**verdict line** is read (not its exit code), and for W3/W5-W7 the exact build
and run recipe, device, driver, and contention state. W7 additionally records
the artifact's sha256, the exact conversion recipe, and peak RSS.

## Stop conditions

- **If vLLM #53906 merges**, stop the wave in flight at its next commit
  boundary, reconcile this spec onto vLLM as the mirror source, and record the
  change. Do not advance the parity pin as part of that reconciliation without a
  separate row: 348 commits of reconciliation is its own unit of work.
- **If a wave finds an upstream anchor stale**, correct the port map in the same
  change. Do not port from memory.
- **If the tiny-shape reference cannot be constructed** (the config refuses a
  small shape, or a component hard-codes a dimension), return `NEEDS_DECISION`.
  That would remove the campaign's only numerical oracle and the plan must
  change rather than proceed hand-derived.
- **If W7's arithmetic does not survive contact** — the Q2_K arm does not fit
  once real KV and activations are measured — stop and re-plan the arm. Do not
  reach for a smaller i-quant, because R4 says it is not producible.
- **No wave takes a GPU lease without `rc`**, and no wave downloads a large
  asset without explicit developer authority.

## Owed

Debts this row carries, each visible rather than waived:

- **O1 — no end-to-end token gate exists or can exist on this fleet** for
  `Glm5NextForConditionalGeneration`. Owed against a device that can hold 306
  GiB, or against a multi-device execution path this project does not have.
  Tracked by [#1998](https://github.com/mudler/vllm.cpp/issues/1998).
- **O2 — the MTP head** (`num_nextn_predict_layers: 1`,
  `index_share_for_mtp_iteration: true`) is not implemented and the nextn tail is
  skipped, following `glm4_moe_lite_registry.cpp:21-26`.
- **O3 — the text-only arm** `Glm5NextForCausalLM` has no row and no
  implementation; it is not declared by any published artifact today.
- **O4 — no llama.cpp RELEASE defines this architecture, so the floor is still
  owed, but a scoped PR-pinned oracle now exists** (D6). Corrected on
  2026-08-28 by [#2178](https://github.com/mudler/vllm.cpp/issues/2178), which
  registered [`llama-cpp-glm5next`](../oracles/llama-cpp-glm5next.md) at
  `ggml-org/llama.cpp` PR #27752, object
  `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`. Re-measured in a fresh bare clone
  that day: `git grep -il 'glm5next\|glm5_next' b10451` is rc=1 tree-wide
  against a `glm4_moe` control at rc=0, and the same grep at `master`
  `50f068fff` is rc=1 too, so the release half of this entry HOLDS. What no
  longer holds is the second clause: there IS a llama.cpp that knows
  `glm5next`, it is unmerged, and it is pinned by object id. **Still owed:**
  (a) the FLOOR itself — no speed or memory number has been taken against that
  oracle, which is `gateable = no` because nothing has been built or run at the
  pin; and (b) a VISION denominator against the PUBLISHED `mmproj-BF16.gguf`,
  which neither head can open: that file declares
  `clip.projector_type = glm5next` and neither projector table defines that
  string. Do NOT read (b) as "#27773 has no vision path" — it has its own,
  measured at its head: `conversion/qwen3vl.py:254-260` registers
  `Glm5NextForConditionalGeneration` emitting
  `gguf.VisionProjectorType.GLM5V`, `constants.py:5723` defines
  `GLM5V = "glm5v"`, and `tools/mtmd/clip-impl.h:551` accepts it. A vision
  denominator is therefore obtainable by CONVERTING the checkpoint with that
  head, and unobtainable only by pointing it at the published mmproj.
- **O5 — no i-quant arm is producible on this fleet** (R4).
- **O6 — speed.** No number on any axis, and no denominator exists.
- **O7 — no artifact of this model exists.** W7a authored the converter and
  gated it on synthetic fixtures; it has never been run against the real
  checkpoint. Producing the Q2_K arm needs the 300–600 GiB checkpoint staged on
  local disk (not CIFS), explicit developer authority for the download, and a
  box with room for the source and the ~100.35 GiB output at once. Until then
  every GPU gate on this row — W3, W5, W6 and W7b — has nothing to load, and
  §Evidence's sha256, conversion recipe and peak RSS are unpaid.
  W7b/[#2011](https://github.com/mudler/vllm.cpp/issues/2011) owns it.
- **O8 — the Q3_K, Q4_K and Q5_K encoders are not ported** and the converter
  refuses those arms by name. Only Q2_K, Q6_K and Q8_0 are ported from the
  pinned llama.cpp reference and gated byte-for-byte against it. No arm this row
  needs uses them today; the §Hardware second-choice line that mentions Q5_K for
  the non-expert 3% would need this first.
  [#2011](https://github.com/mudler/vllm.cpp/issues/2011) records it.
- **O9 — DISCHARGED by W1 ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)).**
  `glm5next` now has its row in the `general.architecture` dispatch table
  (`kGgufArchArms`, `src/vllm/entrypoints/model_loader.cpp`) and its own config
  builder `Glm5NextHfConfigFromGguf`, so a file the W7a converter writes is
  opened, its metadata read, its topology cross-checked against its tensor
  inventory, and its config validated — by the same `ParseGlm5NextParams` a
  `config.json` descends through. It is still refused ABOVE that, by name, at
  weight materialization: what W1 buys is that the refusal now names the
  loader wave that owes the work instead of naming the file's architecture as
  unrecognized. That distinction is the whole of O9 and it is not more than
  that.
- **O10 — nothing above the config layer is implemented, and the loader, the
  forward and the KV-cache spec all refuse by name.** W1 makes
  `Glm5NextForConditionalGeneration` RESOLVE and makes its config PARSE and
  VALIDATE. It does not make the model load and it does not make it forward.
  The KDA sigmoid forget-gate branch is W2's, the NoPE MLA and the k-pool
  indexer W3's, the unweighted mHC head W4's, the assembled text forward W5's,
  the vision tower and processor W6's. Each refusal names its wave.
  [#2067](https://github.com/mudler/vllm.cpp/issues/2067) records it.
- **O11 — `MlaBlockDims::Validate` still refuses this model's geometry.** W1
  deliberately did NOT relax it. The config layer mirrors upstream, which
  *requires* `qk_rope_head_dim == 0`
  (`validate_architecture`: "Expecting NoPE for the DSA attention layers"), so
  the two validators are exact complements and no value satisfies both. W3 owns
  the relaxation; `test_glm5_next_scaffold.cpp` pins the refusal as a live fact
  so W3 cannot land the geometry without also moving the pin.
  [#2067](https://github.com/mudler/vllm.cpp/issues/2067) records it.
- **O12 — DISCHARGED by W0 ([#2096](https://github.com/mudler/vllm.cpp/issues/2096)).**
  `.agents/oracles/transformers.md` now carries a `glm5_next` lane block at
  `transformers` `5.16.1`, with `gateable = no`, the reason, `owner_row`, and
  the issue that owes the measurement. The registry pin stays at `5.14.1` and
  `.agents/upstream-sync.md` is untouched. Every wave that cites `v5.16.1` — W1
  included — now cites a revision the oracle registry records. **One clause of
  this entry was wrong and O13 replaces it:** the lane block is NOT "a record
  surface a checker reads". W0 measured that no checker parses an
  `oracle-pin-lane` fence.
- **O13 — a lane pin is unchecked prose, and W0 measured it rather than
  assuming it.** `scripts/check-oracle-pins.py` matches `^```oracle-pin\n`, so
  an `oracle-pin-lane` fence never matches and no checker in this tree parses
  either lane block in `.agents/oracles/transformers.md`. Corrupting this row's
  lane `pin`, `gateable` or `pinned_on`, and deleting the whole lane block,
  each leave the checker at exit 0; corrupting the registry `oracle-pin` block
  reds it. Read W0's gate in §Gates as "the checker stayed green", never as
  "the checker validated these fields". Not repaired in flow: W0's scope
  excludes every checker, and teaching one to parse a lane block is a semantic
  checker change that AGENTS.md requires to carry its own spec, a red-before
  mutation, and a decision about which keys a lane record requires.
  [#2099](https://github.com/mudler/vllm.cpp/issues/2099) owns it.
- **O14 — `vt::KdaChunkPrefill` cannot serve this model, so both KDA paths run
  the recurrence.** The chunked prefill op takes the RAW gate projection and
  FUSES the gate, `-exp(a_log)*softplus(g_raw + dt_bias)`, inside the vendored
  FLA Triton-AOT cubins (`include/vt/ops.h`) and inside its CPU reference
  (`src/vt/cpu/cpu_ops.cpp:1779-1786`). That is the SOFTPLUS branch.
  GLM-5.3-Flash needs the sigmoid branch, and no `(a_log, dt_bias, g_raw)`
  reproduces it: inverting the fused softplus needs
  `g_raw = log(exp(-target) - 1)`, which diverges to `-inf` as the gate
  approaches 0, which is where most channels of 34 layers sit. W2 therefore
  routes BOTH prefill and decode through `vt::KdaGatedDeltaRule`, which consumes
  an already-computed per-K-channel log-decay and is branch-agnostic. Closing
  this needs a chunk op that accepts a precomputed `g`, which is a change to a
  shared kernel family this row has no gate for. No correctness consequence; a
  named speed cliff on top of the one §Our baseline "KDA" already records for
  the 64-head geometry. [#2097](https://github.com/mudler/vllm.cpp/issues/2097)
  records it.
- **O15 — the KDA arm is NOT REACHED from a production entry point.** W2 lands
  `glm5_next_kda.{h,cpp}`, and `Glm5NextForConditionalGeneration::Forward`
  still refuses by name (O10), so the only call sites at that merge commit are
  the focused gate's. This is the staged-slice disclosure AGENTS.md "Nothing
  lands dead" requires and not an exception claimed by silence: the wiring
  belongs to **W5**, the assembled text forward, on row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, and W5 has no
  issue of its own yet, so [#1998](https://github.com/mudler/vllm.cpp/issues/1998)
  tracks it. What W2 buys is that when W5 wires the layer it wires a gated one.
- **O16 — W4's mHC bricks are not reached from a production entry point.**
  `src/vllm/model_executor/models/glm5_next_mhc.cpp` is a host reference and
  nothing in the shipped tree calls it: the loader and `Forward` still refuse by
  name (O10), so no `include/vllm.h` entry point, no registered server path and
  no command-line default can reach `MhcPre`, `MhcPost` or `HcHeadCollapseMean`.
  The gate enters through the test binary, which measures the functions and not
  a capability. This is the staged-slice exception in AGENTS.md §"Nothing lands
  dead", declared rather than silent. **W5 owns the wiring** — it assembles
  `Glm5NextTextModel::Forward` and the decoder layer's two mHC sites — on the row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, and
  [#2098](https://github.com/mudler/vllm.cpp/issues/2098) records it under the
  campaign issue [#1998](https://github.com/mudler/vllm.cpp/issues/1998).

## Now

`ACTIVE`, 2026-08-27. Advanced from `READY` by W7a
([#2011](https://github.com/mudler/vllm.cpp/issues/2011),
`CLAIM-GLM53-FLASH-W7A`), which landed the first product code on the row: the
safetensors→GGUF converter and its synthetic-fixture gate, with the Q2_K, Q6_K
and Q8_0 encoders byte-identical to the pinned llama.cpp `b10451` reference.

W1 ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)) then made the
architecture RESOLVE. `Glm5NextForConditionalGeneration` is registered from its
own translation unit, `glm5next` has its `general.architecture` dispatch row,
and both sources — a `config.json` and a converter-written GGUF — descend
through one `ParseGlm5NextParams` that mirrors upstream's `__post_init__` and
all five `validate_architecture` rejections. **O9 is discharged.**

**No artifact exists** (O7) and **nothing loads or forwards** (O10): the
loader, the forward and the KV-cache spec each refuse by name, and
`MlaBlockDims::Validate` still refuses this model's NoPE geometry (O11). No GPU
gate has moved and no correctness claim about the MODEL has been made.

W0 ([#2096](https://github.com/mudler/vllm.cpp/issues/2096)) then wrote the lane
oracle pin. `.agents/oracles/transformers.md` records `transformers` `5.16.1`
for `model_type: glm5_next` only, `gateable = no`, expiring when vLLM registers
the architecture; the registry pin stays at `5.14.1` and the vLLM parity pin is
untouched. **O12 is discharged.** O13 records what W0 measured on the way: no
checker in this tree parses an `oracle-pin-lane` block, so W0's §Gates line
means the checker stayed green and not that it validated the fields
([#2099](https://github.com/mudler/vllm.cpp/issues/2099)).

W1's file also broke the Windows build, repaired here as
[#2101](https://github.com/mudler/vllm.cpp/issues/2101): seven range-`for` loop
variables named `n` in `Glm5NextExpectedGgufTensors` hid the function-scope
`const size_t n` at `glm5_next_weights.cpp:252`, and under `/W4 /WX` MSVC's
C4456 became `error C2220`, so `windows-msvc-cpu` and `windows-msvc-vulkan`
failed the build on `main` and on every branch that merged it. The
function-scope local is now `layer_count` and the loop variables are `tn`. Two
facts are worth keeping: sibling scopes do not hide one another, so the shadowed
declaration was never another loop's variable, and CI reported four of the seven
sites because MSVC stops at the first `error C2220` — GCC's `-Wshadow` names all
seven and is the local instrument for this class.

W2 ([#2097](https://github.com/mudler/vllm.cpp/issues/2097),
`CLAIM-GLM53-FLASH-W2`) then landed the KDA arm's numerics — the forget gate's
SIGMOID branch, the strict-fp32 `RMSNormGated`, `l2norm`, the checkpoint's three
depthwise convs concatenated into the reference's one grouped conv, and the
assembled host layer on the `vt::KdaGatedDeltaRule` seam — gated RED-first
against `kimi_kda.cpp:60`'s softplus branch. The fresh review found two
defects and both are repaired on this branch: the conv-order case swapped two
weight TENSORS and so moved under any fixed concat order — it is now gated
against references built with the wrong PAIRING, and it reds under a q,v,k and
under a k,q,v mutation — and `dt_bias` was optional, which upstream has no
mode for (`:384` declares it unconditionally, `:393` always adds it), so an
absent or misshaped tensor is now refused by name. That code is **not reached** from
any production entry point (O15) and `vt::KdaChunkPrefill` cannot serve this
model (O14). W0 has since landed the lane pin on `main`.

W4 ([#2098](https://github.com/mudler/vllm.cpp/issues/2098),
`CLAIM-GLM53-FLASH-W4`) then landed the mHC arm. Three of the topology's four
pieces reuse DeepSeek-V4 unchanged, because `Glm5NextTextHyperConnection` is a
bare `pass` over `DeepseekV4HyperConnection`; the fourth does not.
`Glm5NextTextHyperHead.forward` is `hidden_streams.mean(dim=2)`, so
`glm5_next::HcHeadCollapseMean` is an UNWEIGHTED mean where V4's
`HcHeadCollapse` is a sigmoid-gated weighted sum, and the checkpoint carries no
`hc_head.*` tensor a gated collapse could read. `glm5_next::MhcPre` and
`MhcPost` add no numerics; they bind this model's five constants in one place.
Every golden is the RUN output of the unmodified reference modules at
transformers `v5.16.1`, not a transcription, and the gate was RED first against
the wrong reuse at 59 of 98 assertions failed. That code is **not reached** from
any production entry point (O16); W5 owns the wiring.

The next actions are W3 and W5, and, whenever the developer grants a
large-asset download, W7b.
