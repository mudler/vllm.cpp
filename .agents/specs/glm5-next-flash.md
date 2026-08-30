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

### W3 — NoPE MLA and the DSA k-pool indexer (GPU, large) — [#2213](https://github.com/mudler/vllm.cpp/issues/2213)

**LANDED 2026-08-28** (`CLAIM-GLM53-FLASH-W3`), CPU-gated; the CUDA arm is
committed and its lease is `PENDING` (below).

Extend `MlaBlockDims::Validate` to accept `qk_rope_head_dim == 0` with
`head_size() == kv_lora_rank`, and thread it through the decode and prefill
paths; resolve and implement the k-pool compression and tail-keep in the
indexer.
**Scope:** `mla_attention.{h,cpp}`, `vt/ops.cpp`, `glm5_next_dsa.{h,cpp}`.
**Exclusions:** do not change any existing model's resolved geometry. SACRED
inertness against DeepSeek-V2/V3, Kimi-Linear and GLM-4.7-Flash goldens is a
gate, not a hope.
**Anchors:** `modular_glm5_next.py:749-1024`, `:1025-1141`.
**Tests:** NoPE geometry validation, both accept and refuse; a k-pool selection
case at a context **strictly greater than `index_topk`** so the selection is not
the identity; existing MLA goldens byte-identical.
**Needs GPU:** the decode kernel's 512/256 head pair is untested and the
shared-memory guard at `cuda_mla_attn.cu:545-546` can only be checked by
running. **Rebase note:** coordinate with PRs #1971 and #1977 — both MERGED
before this wave started, so there was no live coordination left.

#### What W3 actually resolved

**The two validators were exact complements, and 0 is now the ABSENT rotary.**
`MlaBlockDims::Validate` required every dimension `> 0`; upstream's
`validate_architecture` requires `qk_rope_head_dim == 0`. The relaxation splits
that one clause out: `qk_rope_head_dim >= 0` with 0 meaning there is no
decoupled-rope slice at all, so `head_size()` is `kv_lora_rank` (512, not 576)
and `qk_head_dim()` is the nope part alone (256). Three refusals were ADDED
rather than removed, because accepting 0 must not become accepting anything: a
NEGATIVE width is refused by its own message; an ODD width is still refused, so
0 passes because it is even and non-negative and not because the check was
deleted; and `is_neox_style` / `indexer_rope_is_neox_style` are refused at 0,
because upstream builds no rotary for this model and a rotation STYLE on a layer
with no rotation is a caller that believes it is on a DeepSeek layer.
`v_head_dim <= qk_head_dim()` binds HARDER under NoPE — 320 fits a 256+64 query
and does not fit a 256 one — which is the clause a port silently violates by
copying a DeepSeek-shaped v width across.

**Threading was four NOT-TAKEN branches and two wrapper clauses, not new code.**
The block already guarded its rope with `R > 0` in the fused arm; W3 added the
guard on the two A-projection rope GEMMs (`Tensor::Slice` refuses an empty row
range, which is what the first NoPE run actually hit), and took the two rope
VIEW offsets to 0 at `R == 0` so no pointer is formed past a zero-width buffer.
On the op side `vt::ConcatMlaNopeRope` and `vt::ConcatAndCacheMla` each refused
a zero-width rope part in their WRAPPER while both their CPU and CUDA kernels
already handled it — their rope loop runs zero times — so the change is the
wrapper admitting a shape the implementations always served. `dn == 0` and
`kv_lora_rank == 0` stay refused.

**The indexer's pooling is the genuinely new work.** `glm5_next_dsa.{h,cpp}`
mirrors `Glm5NextTextIndexer` function for function: `PackIndexerStates`
(`:795-801`, the 257-float packed row), `GetVisibleTokens` (`:877-895`),
`GetPooledStates` (`:897-970`), `AppendVisibleTail` (`:972-1022`) and
`SelectIndexerTopk` (`:771-875`). `deepseek_v4_dsa.cpp` is NOT reused and the
header says why: it has no pooling stage, so it selects the wrong candidate set
and yields plausible indices either way.

**Gate.** `tests/vllm/models/test_glm5_next_dsa.cpp` against
`glm5_next_dsa_goldens.inc`, GENERATED by
`fixtures/gen_glm5_next_dsa_goldens.py` running the unmodified reference at
transformers `v5.16.1` (the installed `modeling_glm5_next.py` is sha256
`2092bbb4…` byte-identical to `raw.githubusercontent.com` at that tag). The
fixture is `seq_len` 21 against `index_topk` 8 — strictly past the threshold —
with row 1 left-padded by three so the two rows do not share a pool grid, and
`index_kpool` 4 so two of five pools are chosen. It asserts SET equality of the
selected indices and PRINTS the margin: 17 discriminating rows, smallest margin
**2.58e-3**, zero ties.

**The captured `index_scores` are ASSERTED, not only the argmax over them.**
The first review of this wave found that they were not, and that two live scale
defects therefore passed the whole file: dropping `n_heads**-0.5` from the
per-head mix (relative error 1.83) and building `softmax_scale` from the wrong
head dim (relative error 1.0, the trap the header warns about in prose). Both
are uniform positive rescalings, so they permute nothing and the discrete
top-k cannot see them; the printed margin could not either, because it is
computed from OUR OWN scores and so scales WITH the defect. The 210 oracle
values are now compared with an absolute tolerance of 2e-4, against a measured
worst difference of 7.63e-6 on a largest |golden| of 45.17 — 26x of headroom
against reduction-order drift, and still red for any uniform relative scale
error above 4.4e-6. Both defects were re-applied and each reds.

**An empty pool set is SERVED, not refused.** Below `index_kpool` valid tokens
no pool is complete, `keep = pool_valid.any(0)` is empty (`:967-970`), `P` is 0
and `select_k` is 0 — and upstream carries the empty candidate dimension
through, so `append_visible_tail` returns the raw visible tail on its own. We
refused it by name instead, which rejected four prompts the oracle answers: the
run at transformers v5.16.1 serves `seq_len` 1, 2 and 3, and a `seq_len` 3 row
left-padded by one, with a well-formed tail-only selection. Those four runs are
generated into the same fixture as `kShort*` and asserted positionwise; the
refusal is gone. `tests/vllm/model_executor/layers/attention/
test_mla_attention_block.cpp` adds the NoPE accept and refuse cases and runs the
NoPE geometry decode / prefill / chunked-context / MIXED against the SAME
double-precision `RefBlock` every DeepSeek case uses, plus the
decode-vs-prefill two-path agreement that proves the absorption identity under
NoPE rather than asserting it.

**SACRED inertness, measured.** The six-arm DeepSeek byte-identity probe was
run on the base SHA `150b37852` and on the head with only the product files
swapped: all six FNV-1a fingerprints identical (`a2f1e41a168210a8`,
`278156e492ef2281`, `232c61867237916e`, `1e0874090a29a4fa`, `85d76ad77adbbb47`,
`82d987ccac222326`). Twelve DeepSeek-V2/V4, Kimi-Linear and GLM-4.7-Flash test
binaries were built and run at both trees with identical results; two of them —
`test_deepseek_v2_paged_engine` and `test_glm4_moe_lite_paged_engine` — report
`assertions: 0`, which is a SKIP wearing a pass and is recorded as such rather
than counted as coverage.

**GPU gate: `PENDING`, and the reason is a lease, not a result.** `dgx:gpu0` was
held by another session's LTX-2.5 oracle render; `rc hold` queued at position 1
and was released rather than blocked on. `orin:gpu0` and `strix:gpu0` were free
and are the wrong devices for this measurement. What CAN be said statically, and
is not a substitute for running: the decode dispatch sizes its dynamic shared
memory as `(kBlockH + n_tile) * head_size * 4` with `kBlockH = 16`,
`kNTile = 8`, so the published `head_size` 512 asks **49,152** bytes against the
**55,296** that DeepSeek's 576 already gets on this fleet — strictly less than a
live configuration.

**It is NOT the same code path as DeepSeek's 576, and the earlier draft of this
paragraph said it was.** `cuda_mla_attn.cu:554` and `:565` gate the
`cudaFuncSetAttribute` opt-in on `smem > 48u * 1024u`, and 49,152 is exactly
`48 * 1024`, so at `head_size` 512 the opt-in is NOT taken; at 576 it is.
`:639` and `:644` use the same strict `>`, so `DynamicSmemFits` is never
consulted at 512 either — both expressions short-circuit before it. What the
argument therefore rests on is different and weaker: 48 KiB is the
architecture-guaranteed dynamic shared-memory limit a block gets WITHOUT any
opt-in, the request is exactly that limit and not one byte over, and
`cuda_mla_attn.cu` declares exactly one `__shared__` array (`:223`, the `extern`
dynamic one), so no static allocation is competing for the same budget. The
launch relies on the guaranteed floor rather than on the opt-in DeepSeek's 576
takes. That is still an argument and not a measurement, and it does not settle
occupancy, which is what a run would report. The CUDA case is committed and
guarded by `HasCuda()`, so it runs on the first lease.

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

### W5 — the MoE and the KV-cache spec (CPU, large). LANDED

Issue: [#2223](https://github.com/mudler/vllm.cpp/issues/2223).

**The wave SPLIT, and the split is the decision this section records.** #2223
named four deliverables — the MoE, the per-layer control flow, the assembled
`Glm5NextTextModel::Forward`, and the heterogeneous KV cache. Two of them
landed. The other two are **blocked on a block that does not exist**, and that
was not visible from the issue: W3 landed the DSA indexer's *selection*
(`SelectIndexerTopk`) and relaxed `MlaBlockDims::Validate` for the NoPE
geometry, but it landed no assembled `Glm5NextTextAttention` over them. The
decoder layer's DSA arm therefore has nothing to call. `q_a_proj` /
`q_a_layernorm` / `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `expand_kv`
(`modeling_glm5_next.py:1136-1153`), the attention itself and
`build_attention_mask_from_topk` (`:1218-1257`) are all unwritten. That is a
wave, not a paragraph, so it is **W5b** with its own issue rather than a silent
narrowing of this one.

**Landed here.**

1. **The MoE**, `src/vllm/model_executor/models/glm5_next_moe.{h,cpp}`. It
   BINDS rather than reimplements: the router is `vt::MoeRouterTopK`'s grouped
   `noaux_tc` arm and the epilogue is `deepseek_v4::ClampedSwiGLU` at
   `alpha = 1, beta = 0`. Anchors `modeling_glm5_next.py:158-183`
   (`Glm5NextTextTopkRouter.forward`), `:137-142` (`_apply_gate`), `:98-104`
   (`Glm5NextTextMLP.forward`), `:120-135` + `:200-207` (the composed block).
2. **The heterogeneous KV cache**, `MakeGlm5NextKVCache`, following
   `kimi_linear_registry.cpp:135-166` for the MLA + KDA pair and
   `qwen4_exp_registry.cpp` (W5c-1, #2206) for the third-group discipline.

**THREE published groups, and the three numbers that a plausible port gets
wrong.**

| # | layers | spec | geometry |
|---|---|---|---|
| 0 | 11 DSA | `MLAAttentionSpec` | head **512**, `num_kv_heads` 1 |
| 1 | 34 KDA | `MambaSpec`, 2 states | conv `[24576, 4]` bf16 + recurrent `[64, 128, 128]` **f32** |
| 2 | 11 DSA | `MLAAttentionSpec` | head **257**, `compress_ratio` **1** |

- **512, not 576.** Every DeepSeek variant and Kimi-Linear publish
  `kv_lora_rank + qk_rope_head_dim` = 512 + 64. This model's
  `qk_rope_head_dim` is ZERO and upstream REQUIRES it to be
  ("Expecting NoPE for the DSA attention layers"), so the latent row is 512.
  Reusing 576 over-allocates by 12.5% and nothing downstream reads the
  difference.
- **`conv_kernel_dim`, not `conv_kernel_dim - 1`.** The reference ALLOCATES the
  conv state at the full kernel width:
  `LinearAttentionLayer.lazy_initialization` builds
  `torch.zeros((*shape[:-1], conv_kernel_size))` (`transformers` v5.16.1
  `cache_utils.py:1015-1024`) and `Glm5NextTextLinearAttention.forward` passes
  `conv_kernel_size=self.conv_kernel_size` (`modeling_glm5_next.py:669-671`);
  `causal_conv1d_update` then reads `state_len = conv_state.shape[-1]` (`:382`)
  and writes back that many columns, so the slack column is part of the
  contract. `kimi_linear_registry.cpp:157` publishes `K - 1` for ITS model, and
  copying that across hands the runner a cache one column short of what the
  layer reads. `glm5_next_kda.h` recorded the same width for the host
  reference and this is the spec agreeing with it.
- **257, not 128, and `compress_ratio` 1, not `index_kpool`.**
  `PackIndexerStates` stores `concat[k(128), gate_scores(128), valid(1)]` PER
  TOKEN (`modeling_glm5_next.py:798-801`). Our DeepSeek-V4 parent stores the
  key alone, 128, and reading that across under-allocates by half. The k-pool
  compresses at READ time inside `GetPooledStates`, so nothing divides by
  `index_kpool` — the opposite of `MODEL-MM-QWEN4-EXP`'s QSA side cache, where
  the compression IS in the store and `compress_ratio` is 4.

**ONE conv state, not three.** The checkpoint stores `self_attn.{q,k,v}_conv1d`
separately and the reference declares ONE grouped depthwise conv over the
concatenated channel axis (`:620-628`), so the cache is one
`3 * num_heads * head_dim` channel state. The refusal this function replaced
said "three separate conv states", which would have tripled the group; that is
one of the three stale sentences [#2230](https://github.com/mudler/vllm.cpp/issues/2230)
repairs.

**Group 2 must be an `MLAAttentionSpec`, and a `FullAttentionSpec` fails in
silence.** It is not an MLA claim — it is the key-only page budget. A
`FullAttentionSpec` there is absorbed by the runner's leftover scan as the
single `fa_draft` draft-KV slot, `multi_cache_topology` stays false, and the
side cache is published and never allocated with nothing reported. Measured on
`MODEL-MM-QWEN4-EXP` W5c-1 (#2206) and the same arm is live here.

**REACHED, and this is the first piece of this row that is.** The cases enter
through `ModelRegistry::Resolve` and the `make_kv_cache` factory hook — the
same pair `LoadedEngine::FromModelDir` uses — and nothing constructs
`MakeGlm5NextKVCache` by name. Unwiring `.make_kv_cache` reds the gate at three
`REQUIRE` sites; DELETING the row does not compile at all, because
`-Werror=unused-function` fires on `MakeGlm5NextKVCache`. The toolchain
therefore proves the factory row is the ONLY reference to it, which is a
stronger statement than the red.

**The MoE is NOT reached** — nothing calls it until W5b assembles the layer —
and that is O23, declared rather than silent.

### W5 — what the staged artifact actually does, measured

> **SUPERSEDED as a STATE, retained as a MEASUREMENT.** Everything in this
> section was read on 2026-08-29 and was true that day. The loader no longer
> stops where this says it stops:
> [#2245](https://github.com/mudler/vllm.cpp/issues/2245) landed the IQ2_XS and
> IQ4_XS decoders, [#2247](https://github.com/mudler/vllm.cpp/issues/2247) made
> both keep their blocks, and W5c
> ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) resolves all 1383
> backbone tensors of this artifact. The census below is still the census; only
> the "decodable here" column has moved, and O5, O7 and O8 above carry the
> current reading.

`unsloth/GLM-5.3-Flash-GGUF` rev `d425e572fb9686125831f476129e51cea34bc5b4`,
arm `UD-Q2_K_XL`, staged read-only at
`/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` (four shards, 101.2535 GiB).
Run through the PRODUCTION entry point `LoadedEngine::FromModelDir`, 2026-08-29:

```
REFUSED: gguf: tensor "blk.3.ffn_gate_exps.weight" has unknown ggml type id 17
         in .../GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf
```

So it opens the file, resolves `general.architecture = glm5next` against our
own registration, walks the 4-way split into shard 2, and stops on a TENSOR
TYPE. Type 17 is `IQ2_XS`. A census of every tensor header across all four
shards:

| ggml id | type | tensors | decodable here |
|---:|---|---:|---|
| 0 | F32 | 638 | yes |
| 8 | Q8_0 | 346 | yes |
| 10 | Q2_K | 2 | yes |
| 11 | Q3_K | 1 | **no** (O8) |
| 12 | Q4_K | 1 | **no** (O8) |
| 13 | Q5_K | 181 | **no** (O8) |
| 14 | Q6_K | 117 | yes |
| 17 | IQ2_XS | 82 | **no** (O5) |
| 18 | IQ3_XXS | 41 | **no** (O5) |
| 23 | IQ4_XS | 3 | **no** (O5) |

**The "Q2_K" in the arm name is a floor, not a format.** Unsloth Dynamic mixes
eight encodings and only two of the 288-expert tensors are actually Q2_K; the
experts are IQ2_XS and IQ3_XXS. Six of the eight are undecodable in this tree,
and three of those six are i-quants that no wave has scoped. **This changes what
the row must believe about W7b**: the published artifact is not made loadable by
adding a weight tower, and O7's premise ("no artifact of this model exists") is
superseded by a harder problem than the one it named.

Three facts from the same read confirm ports made blind:
`glm5next.expert_shared_feed_forward_length` is **2048**, so the shared expert
is `moe_intermediate_size` and not `intermediate_size` (12288) — the §W5 trap
list said so from the reference and the artifact agrees. `ffn_gate_inp` is
**F32** on all 43 sparse layers, which is the fp32 router GEMM as an on-disk
fact. And `ssm_conv1d_{q,k,v}` appear as three separate tensors on each of the
34 KDA layers, which is the layout `glm5_next_kda.h` records and the reason the
cache is ONE grouped conv state.

`glm5next.layer_types` is **absent** and the schedule is
`glm5next.attention.head_count_kv`, exactly as
[#2177](https://github.com/mudler/vllm.cpp/issues/2177) measured. That question
is not yet REACHABLE on this file, because the type refusal above preempts the
config read — worth knowing before #2177 is gated against the real artifact.

**Anchors:** `modeling_glm5_next.py:98-104`, `:120-143`, `:158-183`,
`:196-207`, `:620-628`, `:669-671`, `:798-801`; `cache_utils.py:1015-1024`.
**Rebase note (RESOLVED):** the KV grouping overlapped PR #1977, which MERGED
on 2026-08-27, so this built on current `main`.

### W5b — the decoder layer, the DSA attention block, and the assembled forward — SPLIT, see the two sections below

This was W5's PLAN for W5b, kept so the two readings do not look like a
contradiction. It scoped the per-layer control flow, the attention block and
`Glm5NextTextModel::Forward` as one wave. **Anchors:**
`modeling_glm5_next.py:1064-1257` (`Glm5NextTextAttention`, `expand_kv`,
`build_attention_mask_from_topk`), `:1259-1329`
(`Glm5NextTextDecoderLayer`), `:1409-1494` (`Glm5NextTextModel.forward`). The
manifold is threaded from the embedding as
`inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)` (`:1477`) and collapsed
by the UNWEIGHTED `hc_head` before the final norm (`:1493`), so the whole stack
carries `[T, hc_mult, hidden]` and not `[T, hidden]`.

**It splits, and the split is not a size decision.** The attention block and the
`OwnedTensor` bridge answer to `transformers` v5.16.1 and to the llama.cpp
#27752 container, and both can be gated with no cache and no decoder layer over
them. The decoder layer, the mHC threading and the forward answer additionally
to `MakeGlm5NextKVCache` and to the `[T, hc_mult, hidden]` manifold, and they are
where reachability lands. Landing them together would produce one diff whose
correctness argument runs through two unrelated oracles at once.

### W5b-1 — the DSA attention block and the `OwnedTensor` bridge (CPU, large). LANDED — [#2324](https://github.com/mudler/vllm.cpp/issues/2324)

Two deliverables.

**(a) `Glm5NextTextAttention`** (`modeling_glm5_next.py:1064-1257`), as
`src/vllm/model_executor/models/glm5_next_attn.{h,cpp}` — a host f32 reference,
exactly as `glm5_next_dsa.cpp`, `glm5_next_mhc.cpp` and `glm5_next_moe.cpp` are.
`QResid` (`:1167`), `CompressKv` (`:1170-1172`), `ExpandKv` (`:1136-1153`) over
the checkpoint's SPLIT half-transposed `kv_b_proj` halves,
`BuildAttentionMaskFromTopk` (`:1218-1256`), `IndexerRoleFor` (`:1130-1134`) and
`Attention` (`:1155-1216`) with `eager_attention_forward` (`:1039-1061`)
inlined, which upstream says at `:1227-1228` is the only interface a 3-D
per-(query, key) mask can reach.

Three things it gets right that a fluent wrong port gets wrong, each with its own
discriminating case:

* **The `kv_b_proj` halves are SPLIT and only the K half is TRANSPOSED.** The
  file carries `attn_k_b` at `[H, kv_lora, qk_nope]` and `attn_v_b` at
  `[H, v_head, kv_lora]`, so K contracts over its FIRST inner axis and V over its
  SECOND. At the published geometry a swap is a shape error; the gate therefore
  ALSO carries a SQUARE case at `kv_lora == qk_nope == v_head`, where the
  untransposed reading is shape-valid and merely wrong, and asserts the
  separation (2.9469 over every one of 900 values, printed by the case).
* **Cross-layer top-k sharing.** `indexer_types[layer_idx] == "shared"` means the
  layer builds NO indexer and reuses the previous full layer's selection. The
  gate runs the shared layer against BOTH the correct output and the output a
  RECOMPUTING port produces from a decoy indexer — both captured from the same
  oracle run — and asserts ours is the first: 320 of 800 values differ, max
  separation 1.52, over 20 of 50 query rows.
* **The all-masked row is `finfo.min`, not `-inf`** (`:1253`). A left-padded
  query row has every key masked; with `finfo.min` its softmax is uniform and
  its output finite, and with `-inf` the NaN reaches `o_proj` and then the
  residual stream. The `-inf` mutation reds 49 of 160 assertions.

**No rope branch, and upstream is what says so.** `validate_architecture`
(`configuration_glm5_next.py:225-228`) RAISES for any positive
`qk_rope_head_dim`, measured by constructing one in the golden generator, so
`expand_kv`'s concat has a zero-width second half and `key_states` IS `k_nope`.
`MlaDims::Validate` mirrors the refusal in upstream's own words rather than
half-implementing a branch no released config can select.

**(b) The `OwnedTensor` -> host f32 bridge**, as
`glm5_next_bridge.{h,cpp}` — O22's open question, answered. See O25 below for
the decision and its arithmetic.

**Its FOUR advertised refusals are each a gate, which they were not when the
wave was first proposed for review.** `glm5_next_bridge.h` lists four cases
`DecodeOwnedTensorToF32` refuses by name, and the fresh review found that
deleting any of the block element-count check, either byte-span check or the
`default:` dtype arm left the suite fully green. Two of those were not cosmetic:
without the elementwise byte-span check `std::memcpy(out.data(), src, need)`
reads past a short `t.bytes` and serves the HEAP as weight values, and without
the `default:` arm an encoding the bridge cannot widen returns the ZERO-filled
buffer it allocated — the same failure the `host_released` refusal exists to
stop, reached by another door. Five cases now pin them, each proved by
disabling the refusal in a scratch copy with the mutant's BUILD rc recorded
beside its TEST rc:

| refusal | mutation | result |
|---|---|---|
| block element count is a whole number of blocks | `if (false)` | BUILD 0 / TEST 1, 1 assertion |
| block byte span equals `RowSizeBytes` | `if (false)` | BUILD 0 / TEST 1, 3 assertions |
| a block dtype has a `BlockToFloat` decoder | `if (false)` | BUILD 0 / TEST 0 — SURVIVES |
| elementwise byte span equals `numel * SizeOf` | `if (false)` | BUILD 0 / TEST 1, 3 assertions |
| `default:` refuses a non-float encoding | `return out;` | BUILD 0 / TEST 1, 4 assertions |

The survivor is disclosed rather than chased, and it is the "unselected branch"
shape: `vt::IsBlockQuant` is true for exactly the 18 dtypes `BlockToFloat`
answers for, so no input can reach that arm in this build. What the suite gates
instead is the PREMISE — every block dtype has a decoder — and that gate is
ARMED, measured by a second mutation on the other side. Rewriting
`BlockToFloat`'s `kQ8_0` case to `return nullptr` (BUILD rc=0) reds the premise
case AND makes the refusal fire by name in two more:
`` `moe.gate_exps` is q8_0, which this build has no `BlockToFloat` decoder for``.
So the branch is live under the only condition that can reach it, which is the
state IQ2_XS and IQ4_XS were in before #2245.

**NOT REACHED.** Nothing in either file is called from a production entry point
at this merge commit; the only call sites are the two focused gates'. W5b-2
owns the wiring. O25 carries the disclosure.

### W5b-2 — the decoder layer, the mHC threading and the assembled forward — SPLIT AGAIN, see the two sections below

This was W5b-1's PLAN for W5b-2, kept so the two readings do not look like a
contradiction. It scoped `Glm5NextTextDecoderLayer` (`:1259-1329`), the mHC
stream threading, `Glm5NextTextModel::Forward` (`:1409-1494`), the binding of the
attention block to `MakeGlm5NextKVCache`, AND the discharge of O15, O16, O17,
O23 and O25 at the moment the layer calls the five primitives.

**The first four landed together and the fifth did not, because the fifth is not
the same kind of work.** The layer, the threading, the forward and the cache
binding answer to `transformers` v5.16.1 and to `MakeGlm5NextKVCache`, and they
are gated by running the reference. Discharging the five reachability debts
needs something else entirely: a weight bridge for the FOUR arms W5b-1 did not
bridge, and an engine binding that turns a `ModelForwardInput` into per-request
sequences and carries per-layer state across steps. §W5b-2b states the
arithmetic that makes the first of those a design problem rather than four more
`BridgeDsaLayer`s.

### W5b-2a — the decoder layer, the mHC threading, the forward and the KV binding (CPU, large). LANDED — [#2241](https://github.com/mudler/vllm.cpp/issues/2241)

`src/vllm/model_executor/models/glm5_next_layer.{h,cpp}`:
`DecoderLayerForward` (`:1279-1329`) with all four control-flow arms
`:1261-1272` selects between, `TextModelForward` (`:1431-1494`), and
`ExpandToHiddenStreams` (`:1477`). Plus the cache binding, as an additive
`DsaCache*` on `Attention` (`glm5_next_attn.h`) and a
`SelectIndexerTopkFromPacked` lifted out of `SelectIndexerTopk`'s body
(`glm5_next_dsa.h`), both null-default and byte-identical on the uncached path
W5b-1 gated.

**THE MANIFOLD IS THE WHOLE POINT, and it is gated three ways that do not
overlap.** `:1477` expands the embedding to `[B, S, hc_mult, H]` and nothing
collapses it until `hc_head` at `:1493`; a port that threads `[B, S, H]` and
collapses early RUNS, is finite, and emits fluent text, and every sublayer gate
on this row stays green because the collapsed stream is exactly what the
sublayers consume. So: the per-layer `[B, S, 4, H]` streams are asserted
ELEMENTWISE; `kStreamSeparation` carries the oracle's own minimum pairwise
distance between the four streams (**6.4703**) so those assertions are shown to
be discriminating rather than four copies of one value; and `kEarlyCollapseFinal`
is a DECOY produced by the SAME oracle modules with the manifold collapsed to its
mean and re-broadcast after every layer, which the gate asserts we differ from by
the oracle's own measured **2.4032**.

**The KV binding stores the LATENT, not what the reference stores.** Upstream
caches the EXPANDED `key_states`/`value_states` at `:1175-1179` — 32,768 values
per token per layer — and `DsaCache` stores the 512-wide `k_pass` and the
257-wide packed indexer row instead, which is exactly what
`MakeGlm5NextKVCache`'s groups 0 and 2 publish. §"MLA: cache the latent" decided
that and said the equivalence was to be PROVED; the proof is that `ExpandKv` is
token-wise, which NoPE is what makes true, and it is a case asserting
`ExpandKv(a ++ b) == ExpandKv(a) ++ ExpandKv(b)` EXACTLY on real values. It is
NOT the absorption optimization, which is a speed change and W8's.

**A MUTATION FOUND THE INSTRUMENT, NOT THE PORT, AND IT IS RECORDED AS A
FINDING.** The mutation that truncates the attention's key range to the current
window under a filled cache — the "a cache nobody reads back" defect — SURVIVED
the whole suite at 1647 of 1647 assertions on the first pass. Its output is
all-NaN, `NaN - want` is NaN, and `NaN > x` is FALSE for every x, so the
running maximum in the test's `MaxGap` helper never moved off its initial zero
and an all-NaN forward read as a PERFECT match on every gap assertion in the
file. `MinStreamSeparation` and the cached-tail comparison were blind the same
way, because `std::max(m, NaN)` returns `m`. All three now treat a non-finite
value as an infinite gap and report the count separately, so a failure
distinguishes "wrong number" from "not a number"; the mutation then reds 3
assertions. This is the broken-instrument-fails-toward-a-code-verdict class, and
it was found only because the mutation battery was run at all.

**RED FIRST, and the red found a real defect in the oracle configuration
rather than in the port.** The first run read 4 of 10 cases and 7 of 1647
assertions failed, layer 0 (KDA) green and every DSA layer red by 2.7 to 8.1.
The cause was that `Glm5NextPreTrainedModel` sets `_supports_sdpa = True`, so the
DEFAULT `_attn_implementation` this config resolves is `sdpa`, and
`build_attention_mask_from_topk` returns a BOOLEAN mask on that arm
(`:1249-1250`) instead of the additive `finfo.min` one the eager arm builds
(`:1252-1256`). The two disagree on a LEFT-PADDED query row where every key is
masked — torch's SDPA emits 0.0 there and eager's uniform softmax emits the mean
of the values, measured as 0.0 against our 0.509 — so the generator now pins
`cfg._attn_implementation = "eager"`, which is the arm W5b-1 gated and the one
`:1227-1228` names as the only interface this model's 3-D per-(query, key) mask
can reach. Green is 10/10 cases and 1656/1656 assertions after the instrument repair (1647 before it).

**NOT REACHED, and O26 carries the disclosure.** `ForwardGlm5NextForConditionalGeneration`
still refuses by name. W5b-2b owns the wiring.

### W5b-2b — the weight bridge for the other four arms, and the engine binding (CPU, large). LANDED — [#2337](https://github.com/mudler/vllm.cpp/issues/2337)

What makes `ModelRegistry::Forward` stop refusing by name, and therefore what
discharges O15, O16, O17, O23, O25 and O10's remaining half. Two deliverables,
and the first is a design problem rather than more of W5b-1's bridge.

**(a) The bridge for the KDA, MoE, dense-MLP and mHC arms.**
`BridgeDsaLayer` covers `Glm5NextMlaWeights` and its nested indexer, and
`TextModelWeights` needs four more: `Glm5NextKdaWeights` (15 tensors),
`Glm5NextMlpWeights` (3), `Glm5NextMhcWeights` (3, twice per layer) and the
`Glm5NextWeights` head — `embed_tokens`, `norm`, `lm_head`. Three of those four
are mechanical. **The MoE is not**, and the arithmetic is this row's own:

| what, at the published geometry | f32 GiB |
|---|---:|
| one bridged DSA layer (W5b-1, measured) | 0.4654 |
| the 34 KDA layers' projections, all held | ~18.5 |
| ONE sparse layer's 288 routed experts (`gate_up` + `down`) | **~27** |
| the 42 sparse layers' routed experts, all held | **~1,150** |
| usable on `dgx:gpu0` | ~119.63 |

`kBridgeTensorF32ByteCeiling` is 1 GiB and the smallest expert bank is 9.0 GiB,
so `DecodeOwnedTensorToF32` REFUSES a bank by name today — which is O25's
`byte_ceiling` gate working exactly as designed, not an obstacle to route
around. Only 8 of 288 experts are selected per token, so the shape that fits is
an on-demand per-expert decode: `MoeForward` grows an optional expert source
consulted for each SELECTED expert when `expert_gate_up` is empty, which keeps
`vt::MoeRouterTopK` and `vt::MoeCombine` as the seams and adds no parallel path.
That is one seam change in W5's file with its own red-first gate, and it is why
this is a wave and not a paragraph.

**(b) The engine binding, which is the SMALLER half and has a house pattern.**
This was surveyed rather than guessed, because "where does per-request state
live across steps" looked like the hard part and is not. TWO registered models
already carry a host arm inside their `forward` hook that ignores the device
paged caches entirely: `NemotronHForCausalLM`
(`nemotron_h_registry.cpp:200-213`, whose comment says the host reference
"consumes three of `ModelForwardInput`'s fields — `token_ids`,
`logits_indices`, `queue`") and `KimiLinearForCausalLM`
(`kimi_linear_forward.cpp:462-476`, which `(void)`s `positions`, `attn_meta`,
`attn_kv` and `queue`). Both RE-RUN THE WHOLE PREFIX each step
(`nemotron_h.cpp:1047-1066`, `kimi_linear_forward.cpp:436-460`) rather than
owning state, and a survey of every `: public LoadedModel` in the tree found NO
model that keeps per-request KV or recurrent state on its `LoadedModel` — the
persistent members are weights, a CUDA-graph driver, a derived read-only cache,
or load-time bookkeeping. Kimi-Linear is the closest architecture there is to
this one (KDA plus MLA), so its shape is the precedent to follow.

The rest is house pattern too: `HostLogits(std::vector<float>&&, vocab)`
(`qwen3_5_common.cpp:24-30`) builds the carrier and derives `rows` from
`host.size() / vocab`, and the gather-then-lm_head — empty `logits_indices`
means EVERY row, the gather happens BEFORE lm_head so it never runs on the full
`T`, and the indices are bounds-checked — is `nemotron_h.cpp:997-1022` with an
identical copy at `kimi_linear_forward.cpp:409-433`. Ragged batching, if W5b-2b
wants it rather than the single-sequence shape both precedents use, is
`attn_meta.query_start_loc` sliced as at `kimi_linear_device.cpp:2120-2126`,
with the empty-`num_computed_tokens_cpu` fallback `dots3_note_device.cpp:336-346`
warns "falls OPEN".

**A full-prefix recompute makes `LayerCache` unreached on that path, and that is
a decision W5b-2b has to take rather than inherit.** W5b-2a's cache binding is
gated and correct; if the production hook follows Nemotron-H and Kimi-Linear it
will not call it, which would leave the binding in the same unreached state this
wave is disclosing. The alternative — carrying `std::vector<LayerCache>` on
`Glm5NextLoadedModel` — has no precedent in this tree, and inventing one on a
model that cannot be run end to end on this fleet is the wrong place to try.

**O19 / [#2260](https://github.com/mudler/vllm.cpp/issues/2260) is W5b-2b's to
answer, and W5b-2a cannot make it reachable.** `glm5_next_layer.cpp` calls
`MoeForward`, whose only `vt` ops remain `vt::MoeRouterTopK` and
`vt::MoeCombine` — every expert GEMM is still a host `std::vector<float>`
accumulation — so it reaches neither `vt::MergedGemmGroup` nor
`MoeGateUpSwiGLUGroupedCuda` and the `gate/up must be the SAME CUDA keep-quant
dtype` throw cannot fire from this row. The per-expert source in (a) keeps that
property by construction, because it hands the block host floats.

#### What W5b-2b actually landed

**`ModelRegistry::Forward` reaches this model.** Both deliverables landed as
scoped, and the plan above was followed rather than reinterpreted: the MoE half
is an on-demand per-expert source and the binding is the surveyed house pattern.

**(a) The bridge.** `BridgeKdaLayer` (15 tensors), `BridgeMlp` (3, used by the
dense layers and by every sparse layer's shared expert) and `BridgeMhcSite` (3,
twice per layer) are shape-checked decodes on the pattern `BridgeDsaLayer`
already set. `BridgeMoeLayer` bridges the ROUTER, the correction bias and the
shared expert and **leaves the three expert banks EMPTY by construction**.

The primitive underneath is new: `DecodeOwnedTensorRowsToF32` decodes a
contiguous LEADING-AXIS ROW RANGE out of a block-resident `OwnedTensor`, and
`HostF32RowBytes` is the shape-only budget for one row.
`kBridgeTensorF32ByteCeiling` is UNCHANGED at 1 GiB and the RANGE is checked
against that same ceiling, so asking for all 288 rows of a bank is refused by
exactly the arithmetic that refuses the whole tensor — the row API is not a way
around the gate. A block row that is not a whole number of blocks is refused by
name, because a slice starting mid-block does not fail: the decoder reads the
next block's scale and returns plausible values from the wrong quantization.

The arithmetic that forces the shape, all of it asserted rather than written:

| what, at the published geometry (288 experts, `moe_I` 2048, `H` 4096) | f32 bytes | GiB |
|---|---:|---:|
| one expert's `gate_up`, `[2 * 2048, 4096]` | 67,108,864 | 0.0625 |
| one expert's `down`, `[4096, 2048]` | 33,554,432 | 0.03125 |
| **one expert, both** | **100,663,296** | **0.09375** |
| ONE ROW of `ffn_up_exps` | 33,554,432 | 0.03125 |
| one bank (`ffn_up_exps`, `[288, 2048, 4096]`) | 9,663,676,416 | **9.0** |
| one sparse layer's THREE banks | 28,991,029,248 | **27.0** |
| the 42 sparse layers' banks, all held | 1,217,623,228,416 | **1,134.0** |
| usable on `dgx:gpu0` | | ~119.63 |

A resident float bank set is **9.5x over the box**. One row is **32x UNDER the
1 GiB ceiling** while the bank is 9x over, which is where the two populations
separate and why the ceiling did not move. `num_experts_per_tok` is 8 of 288, so
what a token needs is at most 8 experts and what a step needs is the DISTINCT
experts its tokens hit; `MoeForward` now groups the `[T, K]` routing slots by
expert and visits each hit expert ONCE, which is upstream's own order
(`Glm5NextTextExperts.forward` loops the hit experts, not the tokens) and which
is what bounds the peak at one expert. Every `[t, j]` slot is still computed
independently into its own row of the combine's `[T, K, H]` operand, so the
resident path is byte-identical to the token-major visit it replaced — asserted
EXACTLY, not to a tolerance, against a resident bank built from the same
tensors.

`MoeLayerWeights` grows a BORROWED `ExpertSource*`. Exactly one of the two
residencies is admissible and the other two states are refused by name: both
populated is ambiguous, and neither populated is a null dereference — **measured
at rc=139 (SIGSEGV), not assumed**, which is why the refusal's message names
that failure and not the softer "reads as zeros".

**(b) The engine binding.** `glm5_next_forward.{h,cpp}`.
`Glm5NextGgufLayerSource` is a `LayerWeightSource` holding ONE slot: it bridges
layer `i`, hands back a reference valid until the next call, and drops the
previous layer BEFORE bridging the next so the peak is never two. There is
deliberately no map keyed by layer index, for the reason `glm5_next_bridge.h`
gives for having no `BridgeTower`. `TextModelForward` grows an overload over
that source, and the resident overload now delegates to it through a trivial
`ResidentLayerSource`, so the manifold expansion, the `prev_topk` threading and
the `hc_head` collapse stay in ONE loop rather than being copied into a second
forward.

`Glm5NextHostForward` is the rest: a per-token embedding gather out of the
`[154880, 4096]` table (2.36 GiB in f32, which the ceiling refuses; a row is 16
KiB), the stack, the logits gather BEFORE `lm_head` with empty meaning every row
and every index bounds-checked, and `lm_head` streamed in `kLmHeadChunkBytes`
(64 MiB) chunks. The chunk size is a PARAMETER and that is not a convenience: at
any geometry small enough to run in CI the default gives ONE chunk, so a
chunk-offset defect would never be reached by a gate that could only take the
default.

The residency ladder the forward actually runs at:

| what | GiB |
|---|---:|
| the tower as loaded, block-resident | 101.14 |
| the tower expanded to f32 (what a `TextModelWeights` would be) | 426.72 |
| ONE bridged DSA layer | 0.4654 |
| ONE bridged KDA layer | 0.5449 |
| ONE sparse layer's router + shared expert | 0.0996 |
| ONE routed expert | 0.0938 |
| one `lm_head` chunk | 0.0625 |

The forward's f32 peak is one layer plus one expert plus one chunk — under 0.75
GiB, 0.6% of the box — while the tower stays exactly as the loader left it.

**The hook follows the house pattern, with ONE divergence stated rather than
inherited.** `ForwardGlm5NextForConditionalGeneration` opens the handle with
`ModelAs<Glm5NextLoadedModel>` (which W5c made possible and this wave gave
something to open it FOR), `(void)`s `positions`, `attn_meta`, `gdn_meta`,
`attn_kv` and `gdn_state`, and re-runs the whole prefix each step — the shape
`NemotronHForCausalLM` and `KimiLinearForCausalLM` both use, on a tree where no
`LoadedModel` keeps per-request state. **The divergence:** both precedents take
`token_ids` as one sequence whatever `num_reqs` says, which on a two-request
step attends across the request boundary. For this model that is fluent wrong
text no gate on this fleet could detect (§Gates), so a multi-request step is
REFUSED BY NAME and ragged batching is owed. A non-CPU queue is refused too,
because every primitive on this row is host f32 and `vt::MoeRouterTopK`
dispatches on the queue's device.

**RED FIRST, and the red came from an EXISTING gate rather than a new one.**
`test_glm5_next_scaffold.cpp`'s case pinning "the forward REFUSES BY NAME, and
names what is UNASSEMBLED" went red at 8 assertions in 1 case the moment the
hook ran. That pin MOVED with the change rather than being deleted by it — the
same shape W3 used when `MlaBlockDims::Validate` stopped refusing the NoPE
geometry — and now asserts the state that replaced the refusal: the blanket
message is GONE (asserted as an absence, with the exact sentence it carried, so
a revert reds it) and a foreign handle is refused by the DOWNCAST naming this
architecture.

**Thirteen negative mutations, and TWO of them survived the first suite.** Both
survivals are recorded below because each is a fact about the gate, not a
footnote. Measured on tree `91354df62`, each mutant proved APPLIED by a diff
hash, BUILT (rc=0 on all thirteen — a build failure reads as a passing test) and
restored byte-for-byte:

| # | mutation | test | rc | failed |
|---|---|---|---:|---:|
| M1 | **REACHABILITY**: delete the `Glm5NextHostForward` call in the registry hook | forward | 1 | 11 of 118 |
| M2 | the layer source ignores its index (always bridges layer 0) | forward | 1 | 32 of 81 |
| M3 | `DecodeOwnedTensorRowsToF32` drops the row offset | bridge / forward | 1 / 1 | 11520 of 32228 / 5 of 118 |
| M4 | `GgufExpertSource` always decodes expert 0 | bridge | 1 | 224 of 32228 |
| M5 | the NEITHER-residency refusal disabled | bridge | **139** | SIGSEGV |
| M6 | the mid-block row-alignment refusal disabled | bridge | 1 | 2 of 32228 |
| M7 | the `lm_head` chunk writes at `o` instead of `first + o` | forward | 1 | 2 of 118 |
| M8 | the non-CPU-queue refusal disabled | forward | 1 | 2 of 118 |
| M9 | the multi-request refusal disabled | forward | 1 | 1 of 118 |
| M10 | the row-range ceiling check disabled | bridge | 1 | 2 of 32228 |
| M11 | the gate and up halves swapped in the per-expert fusion | bridge / forward | 1 / 1 | 224 of 32228 / 3 of 118 |
| M12 | the two mHC sites swapped in the layer source | forward | 1 | 24 of 118 |
| M13 | the per-expert grouping removed from `MoeForward` | bridge | 1 | 3 of 32228 |

**M12 SURVIVED at 86 of 86 before the repair, and the cause is the FIXTURE.**
Exchanging `attn_hc` and `mlp_hc` inside `Glm5NextGgufLayerSource` left every
logit BIT-IDENTICAL. The miniature's mHC `fn` payloads are ramps in the hundreds
and thousands, so `F.linear(normed, fn) + base` saturates every sigmoid gate and
the Sinkhorn projection converges to the same matrix from either site; the swap
is arithmetically invisible downstream. A gate that could only see it through
the logits is a mute switch at that geometry, so the mapping is now asserted
STRUCTURALLY against the loader's own two tensors, with the two tensors asserted
to DIFFER so the equality is a fact and not a tautology. M12 then reds 24.

**M13 SURVIVED at 32030 of 32030 before the repair, and the cause is the
POPULATION.** The "each expert decoded exactly once" case ran ONE token, and at
one token every selected expert is hit once whatever the code does. The case now
also runs SIX tokens, where the router fills 12 slots from 2 distinct experts and
a per-slot visit decodes each of them six times — six 96 MiB decodes at the
published geometry. M13 then reds 3.

**M5 kills by CRASHING rather than by failing an assertion**, and that is
recorded as the result it is: without the refusal the loop dereferences a null
`expert_source`. The message was corrected in the same flow, because it had
claimed the experts "would read as a ZERO weight".

**Green:** `test_glm5_next_forward` 9 cases / 118 assertions, `test_glm5_next_bridge`
19 / 32228, `test_glm5_next_scaffold` 38 / 2652, and the six sibling suites
unchanged and green (layer 10/1656, moe 8/1614, gguf_load 16/8731, attn 14/160,
kda 28/342, dsa 10/1934, mhc 5/98).

**One record edit this wave owes and does not hide:** `glm5_next` is added to
`scripts/runner-routing-allowlist.txt`, because the decode returns HOST logits.
That is the honest state and the entry says why — there is nothing to be
device-native against yet, W3's CUDA arm being committed and unmeasured, and the
forward refuses a device queue rather than pretending otherwise. **The
checker's OTHER invariant misclassifies this model and that is disclosed rather
than fixed here:** `check-runner-routing-consistency.py` reports it in the "30
keep bf16-resident DBuf activations" bucket, which is false — every buffer on
this path is host f32. The reason is the cross-TU resolution hole the
`nemotron_h` allowlist entry already names (#1410): the invariant reads the
registry TU's decode file set, and this model's f32 residual declarations live
in `glm5_next_forward.cpp`. Teaching it to follow the hop is a semantic checker
change, which AGENTS.md routes to its own row, spec and red-before.

### W5c — the weight tower and `load_weights` — SUPERSEDED, see the LANDED section below

This was W5's PLAN for W5c, and it is kept only so the two readings do not look
like a contradiction. It said no `LoadedModel` of this architecture could exist,
so `ModelRegistry::Forward` was unreachable BY CONSTRUCTION, and that the
published artifact was additionally blocked behind O5/O8. All three stopped
being true while W5 was in review: [#2245](https://github.com/mudler/vllm.cpp/issues/2245)
landed the IQ2_XS and IQ4_XS decoders and W5c
([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) landed the tower. The
section immediately below is what actually happened.

### W5c — the weight tower and `load_weights` (CPU, large). LANDED — [#2242](https://github.com/mudler/vllm.cpp/issues/2242)

Split out of W5 because a load and a forward have different blockers and
different oracles: the load answers to the CONTAINER, the forward to the
ALGORITHM. This wave is the container half, and it is the one that ends the
"registered but not loadable" state the row has been in since W1.

**What landed.** `src/vllm/model_executor/models/glm5_next_loader.{h,cpp}` — the
loaded weight set and `LoadGlm5NextFromGguf`, plus `Glm5NextLoadedModel`, which
is the first `LoadedModel` of this architecture that has ever existed. The GGUF
arm of the registry's `load_weights` hook returns it; the safetensors arm still
refuses, and now says why (every published safetensors artifact exceeds every
device this project owns) rather than saying the loader is unported.

Field names and shapes mirror the host references W2, W3 and W4 landed —
`Glm5NextKdaLayerWeights`, `glm5_next_dsa::IndexerWeights`, `HcSite` — one for
one, so W5b's bridge from `OwnedTensor` to those f32 buffers is mechanical
rather than a second name map. **The tower is `OwnedTensor` and not host f32,
and that is a decision rather than a convenience:** the artifact fits at all
only because 736 of its tensors keep their ggml blocks, and a float tower would
be 4x the file. The bridge is W5b's.

**The two oracles, both read at source rather than relayed.** The ALGORITHM is
`transformers` v5.16.1, whose `modeling_glm5_next.py` sha256s to
`2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b` — asserted
against `raw.githubusercontent.com` at the tag, not assumed. The CONTAINER is
llama.cpp PR [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) at head
`8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`, the pin
[`oracles/llama-cpp-glm5next.md`](../oracles/llama-cpp-glm5next.md) records,
whose `conversion/glm5next.py` is 4714 bytes and sha256
`bfacba27746096e7bb3ca4a2549c9026d3475e226c7f3edf230c37ffadc7b6b3`.

**Three convert-time facts the row had wrong, and one it had right.** #2242's own
scope sentence said the tower is mapped from "the GGUF names W7a's converter and
the published artifact agree on". They did not agree. `.dt_bias` is renamed to
`.dt_proj.bias` and lands as `ssm_dt.bias`; `kv_b_proj` is SPLIT into
`attn_k_b` and `attn_v_b` with the k half transposed, so one HF name maps to two
GGUF tensors at different shapes; and `ssm_a` holds `-exp(A_log)` rather than
`A_log`. All three are fixed on both sides under
[#2291](https://github.com/mudler/vllm.cpp/issues/2291). The fourth candidate is
the one the row had right by not having it: **there is no `+1` norm fold in this
chain**, unlike the Qwen3-Next converter the sibling `qwen4exp` loader has to
undo, so a loader that copied that file would subtract 1.0 from every gamma in
the model.

**The dtype polarity, and its three annotated exceptions.** Everything inherits
the model dtype. `router` is f32 because UPSTREAM computes the router GEMM at
f32 (`F.linear(hidden.type(torch.float32), self.weight.type(torch.float32))`)
and the file already stores it that way, so it is a mirror and not a widening.
`e_score_correction_bias` is f32 because it selects experts discretely and a
rounding error there swaps an expert rather than scaling an output. The mHC
`base` and `scale` are f32 because every Sinkhorn denominator adds
`hc_eps = 1e-6` and the bf16 quantum near 1.0 is 3.9e-3, 3900x that eps —
4.86 kB for the whole model. `a_log` and `dt_bias` are f32 for the reason the
sibling row already records.

**Gates.** `tests/vllm/models/test_glm5_next_gguf_load.cpp` (16 cases, 8731
assertions) plus `tests/scripts/test_convert_glm5_next_gguf.py`, and the name
map is gated against the REAL artifact with no asset:
`tests/vllm/models/glm5_next_gguf_manifest.inc` freezes the 1412-tensor header
table of `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL at revision
`d425e572fb9686125831f476129e51cea34bc5b4`, generated by
`scripts/gen-glm5-next-gguf-manifest.py` from the shard headers alone.

**`blk.45` is NOT a decoder layer, and it is asserted three ways** because each
one alone is satisfiable by a wrong loader: a depth of 45 is equally true of a
stack built from blocks 1..45; "no `blk.45.*` name is enumerated" is equally
true of a file that never had an MTP block; so the loader also COUNTS the 29
tensors it skipped, and the synthetic fixture carries a real MTP block for it to
skip. `1383 + 29 = 1412` closes the arithmetic in both directions.

**The fixture's schedule is `[0, 0, 1, 0, 1]` and not `idx % 4 == 3`.** That is
[#2177](https://github.com/mudler/vllm.cpp/issues/2177) made expressible: the
published checkpoint's own schedule happens to BE the stride, so a fixture at the
stride cannot tell a reader that synthesizes it from one that reads it. This one
puts the single DSA layer where the stride would put a KDA layer.

**How far it got on the real artifact, and what was NOT materialized.** Driven at
`/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` through the same chain
`LoadedEngine::FromModelDir` uses for a GGUF — `GgufFile::Open` on shard 1,
`Glm5NextHfConfigFromGguf`, `ParseGlm5NextParams`,
`EnumerateGlm5NextGgufTensors` — HEADERS ONLY. All four shards open and merge to
1412 tensors; the config resolves to 45 layers, hidden 4096, vocab 154880, 288
experts, 34 KDA + 11 DSA, `hc_mult` 4, `index_kpool` 4, `swiglu_limit` 10.0 and
a fully NoPE MLA (`q_lora` 1536, `kv_lora` 512, `qk_nope` 256, `qk_rope` 0,
`v_head` 256); and every one of the 1383 names the tower reads RESOLVES — 0
missing, 0 unexplained — at **41 MB peak RSS**, with no weight byte read. The
residency the load would take, from `PeekRoute` over those same names under
`mmap_residency`: **736 tensors keep their blocks at 98.260 GiB** and **647
expand to bf16 at 0.446 GiB**, totalling 98.707 GiB against the file's 101.2535
— the difference is the 2.55 GiB MTP block this load drops.

**A materializing load was ATTEMPTED and STOPPED, deliberately.** It reached
8.09 GiB RSS in 2m02s in uninterruptible-sleep state, reading the artifact over
CIFS, and was killed. Nothing about a materialized load, a peak RSS at load, a
token or a speed is claimed by this wave. That measurement belongs on
`dgx:gpu0` under an `rc` lease with the artifact on local disk, and it is W7b's
([#2225](https://github.com/mudler/vllm.cpp/issues/2225)) to take.

**Reachability.** The production entry point is the loader:
`ModelRegistry::Load` -> the registration's `load_weights` hook -> the GGUF arm.
Every case in the suite enters there, through `Glm5NextHfConfigFromGguf` and
`ModelSource::FromGguf`, and none constructs a `Glm5NextWeights` by hand.
Deleting the `LoadGlm5NextFromGguf` call site — leaving the type and returning a
default-constructed `Glm5NextWeights{}` — reds 48 assertions.

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
| W5 | router goldens at the PUBLISHED 288/top-8 asserting SET equality with the margin printed; the clamped-SwiGLU epilogue on a row that ACTUALLY clamps in both halves; the composed routed+shared block; the three KV groups and their geometry reached through the `make_kv_cache` factory hook, with the hook unwired as the reachability mutation | CPU |
| W5b | per-layer control-flow goldens; assembled tiny-model forward vs the tiny reference | GPU |
| W5c | `load_weights` builds a `LoadedModel` from a synthetic tiny GGUF and `ModelRegistry::Forward` reaches it; deleting that call site reds | GPU |
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

### The measured residency, and the two `vec_dot` rows that produced it

[#2247](https://github.com/mudler/vllm.cpp/issues/2247), rows
`QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS`.

**Scope.** Two keep-quant `vec_dot` kernels in `src/vt/cpu/cpu_quant_dot.cpp`
beside the fifteen already there, their two block structs, and their two
`QuantTraits` rows. Nothing else: #2245 had already landed the row decoders, the
reader strides and the vt geometry, so this row is purely the dot-product side.

**Upstream anchors**, read out of the pinned object with `git cat-file` and
`git archive` from a fresh partial clone of `ggml-org/llama.cpp`, never a working
tree. `refs/tags/b10451` was confirmed to resolve to
`10bf611e533d81f739128304991c5e133c6aebd8` in that clone.

| ported | from |
|---|---|
| `VecDotIQ2_XSQ8_K` | `ggml/src/ggml-cpu/quants.c:948` `ggml_vec_dot_iq2_xs_q8_K_generic` |
| `VecDotIQ4_XSQ8_K` | `ggml/src/ggml-cpu/quants.c:1283` `ggml_vec_dot_iq4_xs_q8_K_generic` |
| `BlockIQ2_XS` (74 B) | `ggml/src/ggml-common.h:388-393` `block_iq2_xs` |
| `BlockIQ4_XS` (136 B) | `ggml/src/ggml-common.h:454-460` `block_iq4_xs` |
| IQ2_XS traits row | `ggml/src/ggml-cpu/ggml-cpu.c:342-347` |
| IQ4_XS traits row | `ggml/src/ggml-cpu/ggml-cpu.c:385-390` |

**The activation pairing was RESOLVED, not assumed.** IQ4_XS reuses IQ4_NL's
`kvalues_iq4nl` byte for byte, and IQ4_NL pairs with Q8_0, so the shape of the
question was real. `type_traits_cpu` answers it: `[GGML_TYPE_IQ4_XS]` carries
`.vec_dot = ggml_vec_dot_iq4_xs_q8_K` and `.vec_dot_type = GGML_TYPE_Q8_K`
(ggml-cpu.c:385-390), against `[GGML_TYPE_IQ4_NL]`'s `GGML_TYPE_Q8_0`
(:379-384), and the kernel's own name carries the same answer. The reason is
geometry rather than codebook: IQ4_NL's block is 32 elements and pairs with the
32-element activation encoding, IQ4_XS's is a 256-element super-block and pairs
with the 256-element one. Both IQ2_XS and IQ4_XS therefore dot against Q8_K.

**Design.** Both bodies are kept verbatim, including the accumulation order:
IQ2_XS folds `sumi` into `bsum` TWICE per 32-element sub-block because the two
halves take different scale nibbles, and IQ4_XS forms `d1`/`d2` as f32 before the
integer sums and accumulates into `sumf` eight times per super-block. That order
is what makes our GEMM bit-reproducible against upstream, and rewriting either
body "more naturally" would break the gate below rather than merely change a
rounding.

**Risk this row exists to manage: a `vec_dot` is a REDUCTION.** A wrong grid
entry, a swapped scale nibble or a mis-shifted `scales_h` bit pair does not
throw. It moves the sum a little, and every consistency check in the tree
(vec_dot against `BlockToFloat`, `MatmulBTQuant` against per-row vec_dot) reads
the same decode twice and agrees with the defect. IQ2_XXS / IQ2_XS / IQ2_S are
three same-shaped codebooks, so a kernel pointed at a sibling still indexes in
range and still returns a plausible magnitude.

**Tests.** Gated BIT FOR BIT against the ORACLE'S OWN KERNELS on REAL bytes of
the staged artifact — the same 4 IQ2_XS super-blocks of
`blk.3.ffn_gate_exps.weight` and 4 IQ4_XS super-blocks of
`blk.11.ffn_down_exps.weight` that #2245's decoder goldens use, re-verified
against the live file by `dd` on 2026-08-29. The activation side is the oracle's
own `quantize_row_q8_K_generic` over a deterministic integer-valued signal, and
the goldens carry the resulting Q8_K bytes so the test can also assert that OUR
`from_float` reproduces them. The comparison is against upstream's own f32
accumulation and not against a cleaner f64 reference, because a double
accumulator agrees with a reduction-order defect. Total and per-super-block
values are both pinned, so a defect that cancels across blocks is still caught.
Goldens and the full reproduction recipe: `tests/vt/iq2xs_iq4xs_dot_golden.h`.

Both types also join `kWeightCases` in `tests/vt/test_ops_quant_dot.cpp`, which
runs the whole existing battery over them (random-block decode against an
independent f64 reference, `MatmulBTQuant` against per-row vec_dot, the NMSE
ceiling, run-to-run bit-exactness).

**The codebook seal is now COUPLED to the kernel.** #2245 sealed `kIq2xsGrid`
with an FNV-1a digest and a lane histogram, which proves the TABLE holds the
pinned bytes and says nothing about which table the kernel reads. Swapping
`kIq2xsGrid` for `kIq2xxsGrid` inside `VecDotIQ2_XSQ8_K` leaves the seal green
and reds the oracle golden; that mutation is the proof, and the coupling case
states the two facts it depends on (the tables differ over their shared first 256
rows, and the blocks dotted use indices above 255).

**The measured residency.** `RouteGgufTensor` — the production decision — driven
over all 1412 tensors of the artifact's own headers, roles assigned by the
loader's convention (`token_embd.weight` a gather, 3-D `*_exps.weight` stacked
expert weights, other 2-D weights GEMM weights, 1-D vectors), costing a kept
tensor its file bytes and an expanded one `numel x 2`. Nothing is loaded: the
reader mmaps and only the tensor table is touched.

| type | n | disk GiB | resident GiB | resident before #2247 |
|---|---:|---:|---:|---:|
| F32 | 638 | 0.21 | 0.10 | 0.10 |
| IQ2_XS | 82 | 53.33 | **53.33** | **369.00** |
| IQ3_XXS | 41 | 35.31 | 35.31 | 35.31 |
| IQ4_XS | 3 | 3.59 | **3.59** | **13.50** |
| Q2_K | 2 | 1.48 | 1.48 | 1.48 |
| Q3_K | 1 | 0.97 | 0.97 | 0.97 |
| Q4_K | 1 | 0.33 | 0.33 | 0.33 |
| Q5_K | 181 | 3.03 | 3.03 | 3.03 |
| Q6_K | 117 | 2.20 | 2.20 | 2.20 |
| Q8_0 | 346 | 0.80 | 0.80 | 0.80 |
| **TOTAL** | **1412** | **101.24** | **101.14** | **426.72** |

774 of the 1412 tensors route to `kKeepQuant`. All-bf16 is 597.46 GiB. The
saving is **325.58 GiB**, and 101.14 GiB against ~119.63 GiB leaves 18.49 GiB.
The "before" column is the same loop with the two types forced to expand on the
GEMM roles, which is exactly the tree at `94de63ff5`.

**This is a RESIDENCY result and not a speed one, and on `dgx:gpu0` it is not
even a GPU one.** `RouteGgufTensor` is the production decision and the table
above is what it decides, so the artifact genuinely fits. What fits does not
follow the same path afterwards: the CUDA arm has NO keep-quant kernel for
either new type, so on that box the expert GEMM these 85 tensors feed takes the
CPU fallback behind a full `cudaStreamSynchronize`, and the fused
`vt::MergedGemmGroup` seam throws outright. That gap is
[#2260](https://github.com/mudler/vllm.cpp/issues/2260), recorded as **O19**
under `## Owed` with the mechanism, the reachability argument and the three
options. Nothing on this row reaches the fused seam today, so this row breaks
nothing; W5b and W5c make it live. Read every number in this section as "the
model is resident", never as "the model runs at this speed on GB10".

**IQ4_XS has a SECOND consumer, and the expert-tower lane is all-or-nothing.**
`GgufExpertTowersReachSlotLane` (`gguf_device_fit.cpp`) loops the matching
towers and returns false on the FIRST one that does not reach a keep residency,
so a handful of IQ4_XS towers drops a whole arm out of the streaming lane. On
the GLM-5.3 (non-Flash) `UD-IQ1_S` arm that is 4 IQ4_XS tensors beside 106
IQ1_S, 71 IQ3_XXS, 44 IQ2_XXS and 3 K-quant, every one of which already kept:
one expert tower then goes 6.375 GiB to 24.000 GiB of bf16 and the uniform slot
goes 6.375 MiB to 24.00 MiB, turning a 4096-slot cache from 25.5 GiB into 96
GiB. `tests/vllm/model_executor/test_gguf_device_fit.cpp` carries the assertion
directly, on a fixture whose towers are IQ2_XS and IQ4_XS in the
`kStackedExpertWeight` role both models store them in — the role `PeekRoute`
asks about, and the one this row is actually load-bearing for.

**Gates.** `tests/vt/test_ops_quant_dot.cpp`,
`tests/vt/test_ops_quant_traits.cpp`, `tests/vllm/test_gguf_keep_quant.cpp`,
`tests/vllm/model_executor/test_gguf_device_fit.cpp`, plus
`scripts/agent-preflight.sh --fail-on-skip`. The routing table in
`test_gguf_keep_quant.cpp` is restated rather than refitted: its GEMM term moves
20 -> 24 and its GATHER term stays 13, the mirror image of #2245's decode-only
move (gather 11 -> 13, GEMM 20).

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
- **O5 — no i-quant arm is producible on this fleet** (R4). **PRODUCIBLE, not
  readable — and the wording above misled a reader into concluding the whole
  i-quant lane was absent.** O5 is about the CONVERTER, the write side: this
  tree has no i-quant ENCODER and cannot emit one of these arms. The READ side
  is far better covered and always was: `gguf_dequant.cpp` decodes IQ1_S,
  IQ1_XXXS, IQ2_XXS, IQ2_S, IQ3_XXS and IQ4_NL, and
  [#2240](https://github.com/mudler/vllm.cpp/issues/2240) added IQ2_XS (17) and
  IQ4_XS (23), the last two the staged UD-Q2_K_XL arm needed. Every one of them
  is gated byte-for-byte against the pinned llama.cpp. The clarification is
  recorded here rather than in the report that noticed it, because the next
  reader will land on this line and not on that report.
- **O6 — speed.** No number on any axis, and no denominator exists.
- **O7 — NARROWED by W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)):
  an artifact EXISTS, and what is still owed is a conversion of OURS.** The
  original entry said "no artifact of this model exists", and that sentence was
  true when W7a wrote it and stopped being true when `unsloth/GLM-5.3-Flash-GGUF`
  published `UD-Q2_K_XL` — revision `d425e572fb9686125831f476129e51cea34bc5b4`,
  four shards, 1412 tensors, 101.2535 GiB, `general.architecture = glm5next`,
  now staged at `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` and read
  header-first by this row three times. **It was also still in PRODUCT OUTPUT**,
  as the second sentence of the loader's GGUF refusal, and W5c removed it there
  as well: a record correction that leaves the lie in the product is not a
  correction.

  W7b/[#2225](https://github.com/mudler/vllm.cpp/issues/2225) still owns what
  remains, and it is smaller than it was: W7a's converter has still never been
  run against the real checkpoint, so §Evidence's sha256 of OUR output, its
  conversion recipe and its peak RSS are unpaid, and producing it still needs
  the 300–600 GiB source staged on LOCAL disk (not CIFS), explicit developer
  authority for the download, and a box with room for source and output at once.
  What is no longer owed is a file to load: W5c resolves all 1383 of the
  published artifact's backbone tensor names through the production chain.
- **O8 — the Q3_K, Q4_K and Q5_K ENCODERS are not ported** and the converter
  refuses those arms by name. Write side, like O5: the matching DECODERS have
  been present and gated since the k-quant port, so this entry never said
  anything about loading a file that carries those encodings — and the staged
  UD-Q2_K_XL arm carries 181 Q5_K, 117 Q6_K, 1 Q4_K and 1 Q3_K tensors, all of
  which our reader sizes and our loader decodes today. Only Q2_K, Q6_K and Q8_0 are ported from the
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
- **O10 — NARROWED TWICE, to the FORWARD alone: by W5c
  ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)), which made the model
  LOAD, and by W5 ([#2223](https://github.com/mudler/vllm.cpp/issues/2223)),
  which made it PUBLISH a KV-cache spec.**
  W1 made `Glm5NextForConditionalGeneration` RESOLVE and made its config PARSE
  and VALIDATE; W2, W3 and W4 landed the KDA sigmoid forget gate, the NoPE MLA
  with the k-pool indexer, and the unweighted mHC head as host references; W5c
  landed the weight tower, so the GGUF arm of `load_weights` now returns a real
  `Glm5NextLoadedModel` and this architecture has a `LoadedModel` for the first
  time.

  **The KV-CACHE SPEC NO LONGER REFUSES, and W5 is where it stopped.**
  `MakeGlm5NextKVCache` is wired into `kGlm5NextFactory` as `.make_kv_cache` and
  returns THREE real groups on the published topology — the MLA latent over the
  11 DSA layers, one uniform recurrent group over the 34 KDA layers, and the
  257-wide DSA indexer side cache — entered through `ModelRegistry::Resolve` and
  the production `make_kv_cache` factory hook. An earlier revision of this
  paragraph assigned the KV-cache spec to W5b alongside the forward. That
  sentence was falsified by W5's own diff, and it is the failure
  [#2230](https://github.com/mudler/vllm.cpp/issues/2230) documents: a refusal
  that names a wave which already landed sends the next reader to redo finished
  work. It is corrected here rather than carried.

  What still refuses, and who owns each: the FORWARD is W5b's
  ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)); the VISION
  TOWER, processor and placeholder expansion are W6's; the MTP HEAD is O2's; the
  SAFETENSORS arm is deferred rather than unwritten, because every published
  safetensors artifact exceeds every device this project owns, and its refusal
  now says so. Each refusal names its wave.
  [#2067](https://github.com/mudler/vllm.cpp/issues/2067),
  [#2242](https://github.com/mudler/vllm.cpp/issues/2242) and
  [#2223](https://github.com/mudler/vllm.cpp/issues/2223) record it.
- **O11 — DISCHARGED by W3 ([#2213](https://github.com/mudler/vllm.cpp/issues/2213)).**
  `MlaBlockDims::Validate` accepts `qk_rope_head_dim == 0` as the ABSENT state
  of the decoupled rotary, so `head_size()` is `kv_lora_rank` (512) and the
  block's rope branches are NOT TAKEN. W1 deliberately did not relax it and
  `test_glm5_next_scaffold.cpp` pinned the refusal as a live fact; that pin
  MOVED with the change rather than being deleted by it, and now asserts the
  accept plus the 512 / 256 identities, with the geometry's own refuse cases
  living beside the relaxation in `test_mla_attention_block.cpp`. Accepting 0
  did not become accepting anything: negative, odd, and a rotation STYLE on a
  layer with no rotation are each refused by name.
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
  belongs to **W5b**, the assembled text forward (W5 landed the MoE and the
  KV-cache spec and does not call the KDA arm), on row
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
  dead", declared rather than silent. **W5b owns the wiring** — it assembles
  `Glm5NextTextModel::Forward` and the decoder layer's two mHC sites — on the row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, and
  [#2098](https://github.com/mudler/vllm.cpp/issues/2098) records it under the
  campaign issue [#1998](https://github.com/mudler/vllm.cpp/issues/1998). W5
  ([#2223](https://github.com/mudler/vllm.cpp/issues/2223)) landed the MoE block
  and the KV-cache spec and calls none of these three; the split and its reason
  are in `### W5` above.
- **O17 — W3's DSA indexer and the NoPE geometry are NOT REACHED from a
  production entry point.** `src/vllm/model_executor/models/glm5_next_dsa.cpp`
  is a host reference and nothing in the shipped tree calls it: the loader and
  `Glm5NextForConditionalGeneration::Forward` still refuse by name (O10), so no
  `include/vllm.h` entry point, no registered server path and no command-line
  default reaches `SelectIndexerTopk`. The gate enters through the test binary,
  which measures the functions and not a capability. The MLA half is different
  in kind and is stated separately rather than folded in: `MlaBlockDims` and
  `ForwardMlaAttentionBlock` ARE production code with four live callers, and W3
  changed them — what is unreached is the NoPE *configuration* of that seam,
  because no registered model resolves `qk_rope_head_dim == 0` yet. This is the
  staged-slice disclosure AGENTS.md "Nothing lands dead" requires, declared
  rather than claimed by silence. **W5b owns the wiring** — it assembles
  `Glm5NextTextModel::Forward`, builds the `MlaBlockDims` for the 11 DSA layers
  and calls the indexer from the decoder layer — on the row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, tracked by
  [#1998](https://github.com/mudler/vllm.cpp/issues/1998). W5
  ([#2223](https://github.com/mudler/vllm.cpp/issues/2223)) did NOT do it and
  says why: there is no assembled `Glm5NextTextAttention` for the DSA arm to
  call, so the layer would have one live branch and one that throws. **W5 DID
  publish the KV-cache group this geometry needs** — an `MLAAttentionSpec` at
  head 512 for the 11 DSA layers and a second one at 257 for the indexer side
  cache — and that group IS reached, through the production `make_kv_cache`
  hook, so the NoPE latent width is no longer only a test's opinion. What W3 buys is that when W5 wires the layer, the geometry it needs
  is representable and the candidate set it selects over is the pooled one.

  **The same entry carries W3's second debt, because it is the same wave's and
  splitting it would take an O-number a concurrent wave may already be using:
  the CUDA arm of W3 is committed and UNMEASURED.** The 512 / 256 head pair has
  a `HasCuda()`-guarded case in `test_mla_attention_block.cpp` and no run behind
  it: `dgx:gpu0` was leased by another session's LTX-2.5 oracle render for the
  whole of W3's window, `rc hold` queued at position 1, and the wave released
  the queue rather than blocking. The static argument — 49,152 bytes of dynamic
  shared memory, which is EXACTLY the 48 KiB every architecture guarantees a
  block without an opt-in, with the file's single `__shared__` declaration
  (`cuda_mla_attn.cu:223`) competing for none of it — narrows the risk and
  settles nothing. It is expressly NOT the path DeepSeek's 576-wide row takes:
  `:554`, `:565`, `:639` and `:644` all gate on `smem > 48u * 1024u`, so at 512
  the `cudaFuncSetAttribute` opt-in is skipped and `DynamicSmemFits` is never
  consulted, while at 576 both fire. Owed against the next
  `dgx:gpu0` lease on this row;
  [#2213](https://github.com/mudler/vllm.cpp/issues/2213) records it.

- **O18 — the three per-layer CONFIG ARRAYS are DISCHARGED, and the loader now
  stops one geometry key further on. The artifact now FITS, and the sentence
  that said otherwise was true only until
  [#2247](https://github.com/mudler/vllm.cpp/issues/2247) landed the two
  keep-quant kernels (see the residency paragraph at the foot of this entry).**
  With
  [#2240](https://github.com/mudler/vllm.cpp/issues/2240)'s IQ2_XS and IQ4_XS
  decoders in, `LoadedEngine::FromModelDir` opens all four shards of the staged
  `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` artifact, sizes all 1412
  tensors, and ran on into config resolution, where it stopped with
  `glm5_next gguf: key glm5next.attention.head_count_kv is not an integer`. The
  published artifact stores that key as a per-layer `array[i32]` of length 46,
  and `Glm5NextHfConfigFromGguf` read it as a scalar.
  `glm5next.swiglu_clamp_exp` and `glm5next.swiglu_clamp_shexp` are per-layer
  `array[f32]` of the same length, and there is no `glm5next.layer_types` key at
  all, so the same shape was waiting twice more and a `ReqStrArray` behind that.

  **Discharged by [#2243](https://github.com/mudler/vllm.cpp/issues/2243) and
  [#2177](https://github.com/mudler/vllm.cpp/issues/2177) together**, which are
  one defect seen from two sides: the builder now accepts llama.cpp's
  scalar-or-array spelling of `attention.head_count_kv`
  (`b10451:src/llama-model.cpp:1177` reads it through
  `get_key_or_arr(..., n_layer, false)`) and DERIVES the schedule from the values
  with llama.cpp's own predicate, `is_recr_impl[i] = hparams.n_head_kv(i) == 0`
  (`b10451:src/models/kimi-linear.cpp:18`, "KDA layers are recurrent"). When both
  spellings are present they are cross-checked on the layer KIND and a clash
  refuses by name; a per-layer array whose length is not `block_count` refuses by
  name with the shape found; a non-uniform clamp array refuses, because upstream
  has ONE `swiglu_limit`; and a file that states the schedule in neither spelling
  still refuses, naming both keys. The `idx % 4 != 3` fallback survives ONLY on
  the `config.json` path, where it is upstream's own default.

  **THE NEW STOPPING POINT, measured 2026-08-29 on one tree and one binary**,
  with the array fix reverted and restored so the before/after is not a
  cross-build comparison. Driven through `LoadedEngine::FromModelDir` on
  `device = kCPU`, headers only, no tensor materialised:

  ```text
  without the fix : glm5_next gguf: key glm5next.attention.head_count_kv is not an integer
  with    the fix : vt: glm5_next gguf: attention.key_length_mla - attention.key_length
                    is -256 but rope.dimension_count is 0; the file states this model's
                    rotary width twice and the two disagree
  ```

  The refusal is the rotary-width cross-check in `Glm5NextHfConfigFromGguf`,
  named here rather than by `file:line` because the anchor moved once inside the
  pull request that measured it. The append-only index row for
  [#2268](https://github.com/mudler/vllm.cpp/issues/2268) quotes the line number
  it had when the row was appended and cannot be edited; this entry is the
  corrected surface.

  **BLOCKS ARE NOT LAYERS, and the GGUF path used to conflate them.**
  `c.num_hidden_layers` was set straight from `block_count`, so the SAME model
  resolved to a 45-layer backbone from its `config.json` and a 46-layer one from
  its GGUF, and the extra entry was the MTP block. Nothing downstream would have
  refused it: `ParseGlm5NextParams` sizes all three schedules from
  `num_hidden_layers`, so W5b and W5c would have built a decoder layer out of
  the MTP block, and it would have run and produced plausible tokens. The
  contract is BACKBONE depth — `glm5_next.h:193` already annotates the field as
  `// 45` — and llama.cpp states the relationship in its own converters:
  `self.block_count = self.hparams["num_hidden_layers"] +
  self.hparams.get("num_nextn_predict_layers", 0)`
  (`b10451:conversion/exaone.py:134`, and the same `+=` at
  `b10451:conversion/deepseek.py:470` and `:545`). The builder therefore
  resolves `num_hidden_layers = block_count - nextn_predict_layers`, validates
  every per-block array against `block_count`, and truncates the three
  schedules to the backbone. A file claiming more MTP blocks than blocks is
  refused by name. Our own converter writes `block_count = n_layers` with
  `nextn_predict_layers = 0` (`scripts/convert-glm5-next-gguf.py:997`, `:1024`),
  so the formula leaves its output unchanged.

  **What W5b inherits from this.** The MTP block is read, counted and DROPPED:
  its entry in `attention.head_count_kv` — index 45, value `1`, MLA-shaped — is
  not carried into `layer_types`, and no field on `HfConfig` or
  `Glm5NextParams` holds `nextn_predict_layers` yet. So W5b
  ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) gets a stack sized
  to 45 and must not build a layer for block 45; if the MTP head needs that
  block's kind, W5b adds the field, because this change deliberately did not.
  O2 still owns the head itself.

  That is NOT a malformed file. `attention.key_length` is 512 and
  `attention.key_length_mla` is 256 in the published artifact because llama.cpp
  writes `key_length = kv_lora_rank + qk_rope_head_dim` and
  `key_length_mla = qk_nope_head_dim + qk_rope_head_dim`
  (`b10451:conversion/deepseek.py:345-348`), while
  `scripts/convert-glm5-next-gguf.py` writes `key_length = qk_nope_head_dim` — a
  different quantity under the same name. `glm5next.attention.linear_head_count`,
  a `ReqInt` here, appears in none of the file's 72 keys, and llama.cpp spells it
  nowhere. Both are one defect and both change the WRITE side, so they are filed
  as [#2268](https://github.com/mudler/vllm.cpp/issues/2268) rather than folded
  into a config-array fix, and this row owns them. **O20 DISCHARGES both**, and
  carries the stopping point that replaced this one.

  **The array is 34 zeros and 12 ones.** Parsed 2026-08-29 from shard 1's KV
  block. Key index 21, `glm5next.attention.head_count_kv: array[i32] len=46`:

  ```text
  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
  ```

  The ones sit at indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 **and 45**.
  An earlier version of this entry said `0` on 35 KDA layers and `1` on 11
  DSA/MLA layers. That count is wrong on both halves.

  **The length is 46 because `block_count` counts the MTP block.** The same KV
  block carries `glm5next.block_count = 46` and
  `glm5next.nextn_predict_layers = 1`, and `config.json` declares
  `num_hidden_layers = 45`. Entries 0 to 44 are the model's layers, and entry
  45 is the multi-token-prediction block that §"The MTP block is in the
  checkpoint" already records as **DSA/MLA, not KDA**. Over entries 0 to 44 the
  stride holds exactly: `idx % 4 == 3` selects the 11 DSA layers and the other
  34 are KDA. `test_glm5_next_scaffold.cpp` asserts that 34 / 11 split from
  `config.json` and is CORRECT. Entry 45 is a `1` for a different reason, and
  `45 % 4 == 1`.

  **The checkpoint therefore holds 12 MLA-shaped blocks, not 11.** A consumer
  that runs `idx % 4 == 3` over all 46 entries selects eleven, drops the MTP
  block, and reports no error. Whoever sizes the MLA set for
  [#2243](https://github.com/mudler/vllm.cpp/issues/2243) or
  [#2177](https://github.com/mudler/vllm.cpp/issues/2177) must READ the 46
  values and treat entry 45 as the MTP block. Do not re-derive them from a
  stride, and do not read `block_count` as a layer count. §W7a's tensor
  inventory says the same thing from the other side: `index_kpool_compress_ape`
  and `index_kpool_compress_gate` are present on 12 layers, the 11 DSA layers
  plus the MTP block, read by HTTP RANGE from the safetensors index on
  2026-08-26. Two independent sources, one count. The append-only index row for
  #2243 quotes the superseded 35 / 11 and cannot be edited; that row names this
  entry, so this entry is the corrected surface.

  **Reaching config resolution is not the same as the model fitting, and that
  half is now PAID.** Both types were DECODE-ONLY when this entry was written:
  neither had a keep-quant `vec_dot`, so `HasQuantDotKernel` was false and every
  GEMM weight of those two types expanded to bf16 at load. Measured from the
  staged artifact's own headers, all four shards and all 1412 tensors: the file
  is **101.24 GiB on disk and 597.46 GiB as bf16**, an expansion of 5.9x, and
  the resident cost was **426.72 GiB** against about 119.63 GiB on `dgx:gpu0`.
  [#2247](https://github.com/mudler/vllm.cpp/issues/2247) ported the two
  kernels, and the same measurement now reads **101.14 GiB**, which fits with
  18.49 GiB of headroom, for a saving of **325.58 GiB**. Both figures come from
  driving the production `RouteGgufTensor` over the artifact's real tensor list
  (§"The measured residency" above), not from arithmetic. Every other encoding
  in this file already kept its quantization, IQ3_XXS (`VecDotIQ3_XXSQ8_K`)
  included, so these two types were the whole gap. The `QUANT-GGUF-IQ2_XS` and
  `QUANT-GGUF-IQ4_XS` rows of
  [`quantization-matrix.md`](../quantization-matrix.md) now carry `C` = `Y`.
  **The remaining blockers on a real load are functional, not memory.** This
  sentence has now been rewritten twice as `origin/main` moved under this
  branch, so it names the whole chain rather than one milestone. It first named
  [#2243](https://github.com/mudler/vllm.cpp/issues/2243) /
  [#2177](https://github.com/mudler/vllm.cpp/issues/2177) (the per-layer
  `head_count_kv` array), which the first half of this very entry records as
  DISCHARGED by [#2269](https://github.com/mudler/vllm.cpp/pull/2269); then the
  rotary-width cross-check
  ([#2268](https://github.com/mudler/vllm.cpp/issues/2268)), which **O20**
  discharges by moving both sides onto llama.cpp's `attention.key_length`
  meaning; then the `glm4` pre-tokenizer
  ([#2277](https://github.com/mudler/vllm.cpp/issues/2277)), which **O21**
  discharges. None of the three was a memory blocker, and none is left. The
  stopping point today is the WEIGHT LOADER itself:
  `src/vllm/model_executor/models/glm5_next_registry.cpp:78` refuses by name,
  which is O10's refusal reached from the published artifact, and W5b and W5c
  own it. **And the residency figure is not a compute claim:** the CUDA arm has no keep-quant kernel for either type,
  so on `dgx:gpu0` the 101.14 GiB fits with its expert GEMM on the CPU fallback
  — O19 below.

  **O7 is stale beside it and is not corrected here.** "No artifact of this
  model exists" was true when it was written; the UD-Q2_K_XL arm is now staged,
  complete, and read end to end by our own reader. What remains true is the part
  O7 is actually about — our converter has never been run — so the correction
  belongs to W7b, which owns that sentence, rather than to a dequant change that
  merely walked past it.

- **O19 — the 101.14 GiB is a RESIDENCY result. On `dgx:gpu0` the expert GEMM
  for both new types runs on the CPU, and the fused seam THROWS.**
  [#2260](https://github.com/mudler/vllm.cpp/issues/2260).
  [#2247](https://github.com/mudler/vllm.cpp/issues/2247) landed the two CPU
  keep-quant `vec_dot` kernels, which is what flips the artifact's 82 IQ2_XS and
  3 IQ4_XS tensors from `kExpandBf16` to `kKeepQuant` and makes it fit. The CUDA
  arm has no kernel for either:
  `src/vt/cuda/cuda_quant_dot.cu::IsCudaKeepQuantSupported` admits ten
  Q8_K-family encodings — IQ2_XXS, IQ3_XXS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S,
  IQ1_S, IQ1_XXXS — and neither IQ2_XS nor IQ4_XS is among them, while
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp::DeviceKeepQuantSupported`
  returns `true` for CUDA on its `default:` arm regardless, on the recorded
  ground that "CUDA falls back to the CPU kernel for anything it lacks". So the
  residency measurement holds on a CUDA device and the SPEED does not:

  - `src/vt/cuda/cuda_quant_dot.cu::MatmulBTQuantGroupedKernelCuda` takes the
    CPU-fallback arm behind a full `cudaStreamSynchronize` ("keepquant-grouped
    CPU-fallback drain") on every grouped expert GEMM. Correct, and it
    round-trips the routed-expert weight bytes to the host cores per step.
  - `src/vt/cuda/cuda_quant_dot.cu::MoeGateUpSwiGLUGroupedCuda` THROWS
    `gate/up must be the SAME CUDA keep-quant dtype`, because
    `IsCudaKeepQuantSupported` fails for both operands and `MergedGemm` selects
    the fused op on device registration alone, with no dtype predicate.

  **Not reached today, which is why this is a disclosure and not a defect in
  [#2256](https://github.com/mudler/vllm.cpp/pull/2256).** `glm5_next_moe.cpp`
  is W5's host reference and does not use the fused seam; `laguna.cpp` is the
  only model reaching `MoeGateUpSwiGLUGrouped`. It becomes live the moment this
  row obeys AGENTS.md `## Shared seams`, which routes mergeable MLP projections
  through `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup` — that is
  exactly what W5b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) and
  W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) are for, and at
  that moment a 101 GiB-resident model throws at first forward.

  #2260 carries the analysis and three options — port the two CUDA kernels (the
  only one that yields a speed number worth quoting), keep EXPANDING these two
  on CUDA (honest, but then the artifact does not fit at 426.72 GiB), or refuse
  by name at load rather than throwing with the model resident. This row owns
  the consequence; #2260 owns the fix. Until one lands, **no speed or e2e number
  on this artifact may be quoted as a GPU result**, and the `QUANT-GGUF-IQ2_XS`
  and `QUANT-GGUF-IQ4_XS` rows of
  [`quantization-matrix.md`](../quantization-matrix.md) say so in place.
- **O20 — the MLA key CONVENTION and the KDA head count are DISCHARGED, and the
  loader now stops in the TOKENIZER.** [#2268](https://github.com/mudler/vllm.cpp/issues/2268).

  **The number is O20 and not O19 deliberately.** `origin/main` at
  `c3522bc7d` carried O1 to O18; [#2256](https://github.com/mudler/vllm.cpp/issues/2256)
  was adding an O19 on a branch that had not merged. Two branches that each
  append an `O19` produce a duplicate rather than a conflict, so this entry
  skipped the number rather than racing for it. **That reservation worked and
  the gap is now CLOSED:** #2256 merged `origin/main` into itself and its O19
  sits directly above this entry, so O18 to O21 run consecutively and no entry
  is missing.

  **The delta, and it is a delta in MEANING and not in spelling.**
  `%s.attention.key_length` is a name llama.cpp already owns, and for an MLA
  model it names the width of one CACHED K row — the latent plus the rope slice
  — because llama.cpp caches the latent. The per-head query geometry is spelled
  by the two `_mla` keys beside it. `b10451:conversion/deepseek.py`,
  `DeepseekModel.set_gguf_parameters`:

  ```python
  :345  self.gguf_writer.add_key_length(kv_lora_rank + hparams["qk_rope_head_dim"])
  :346  self.gguf_writer.add_value_length(kv_lora_rank)
  :347  self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
  :348  self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])
  :369  self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
  ```

  | key | llama.cpp's meaning | ours, BEFORE | ours, AFTER |
  |---|---|---|---|
  | `attention.key_length` | `kv_lora_rank + qk_rope_head_dim` = 512 | `qk_nope_head_dim` = 256 | llama.cpp's |
  | `attention.value_length` | `kv_lora_rank` = 512 | `v_head_dim` = 256 | llama.cpp's |
  | `attention.key_length_mla` | `qk_nope_head_dim + qk_rope_head_dim` = 256 | same | same |
  | `attention.value_length_mla` | `v_head_dim` = 256 | same | same |
  | `rope.dimension_count` | `qk_rope_head_dim` = 0 | same | same |
  | `attention.linear_head_count` | **spelled nowhere** | KDA `num_heads` | kept, and no longer required |

  The two `_mla` keys already agreed, which is why the defect presented as an
  arithmetic absurdity rather than as a missing key: the reader subtracted a
  cache width from a query width and got `256 - 512 = -256` for a rotary slice
  the same file states as `0`.

  **BOTH SIDES MOVED, and the reader could not move alone.** llama.cpp writes
  `rope.dimension_count` from `qk_rope_head_dim` (`deepseek.py:369`), so the
  cross-check that survives is `key_length - kv_lora_rank == rope.dimension_count`
  — and our former output fails it by construction (`256 - 512 != 0`). Putting
  the reader on llama.cpp's meaning therefore REQUIRED moving
  `scripts/convert-glm5-next-gguf.py` in the same change, which is exactly why
  #2268 was filed as a row-and-spec decision rather than folded into #2243.
  O7 records that this converter has never been run against the checkpoint, so
  no artifact of ours is invalidated by the move, and a file in the former
  private spelling is REFUSED by name — `key_length < kv_lora_rank` is
  impossible under llama.cpp's meaning — rather than read under either
  convention. **The check was not widened.** Four refusals stand where one did:
  a `key_length` below `kv_lora_rank`, a rotary width the file states twice and
  disagrees with itself about, a `key_length_mla` that leaves no room for a
  no-rope slice, and a `value_length` that is not the latent rank.

  **THE KDA HEAD COUNT IS DERIVED, and this port is deliberately STRICTER than
  the oracle.** `attention.linear_head_count` is ours: `git grep
  linear_head_count b10451` is rc=1 tree-wide. llama.cpp's `glm5next` branch
  writes the same number through its Kimi-Linear parent's `ssm.*` names —
  `add_ssm_inner_size(num_heads * head_dim)`, `add_ssm_state_size(head_dim)`,
  `add_ssm_group_count(num_heads)`, `conversion/glm5next.py:78-80` at
  `refs/pull/27752/head` `8a8d0bcc4` — and the published artifact carries NONE
  of those either. So the reader reads whichever of four places the file states
  it in: `attention.linear_head_count`, `ssm.group_count`, `ssm.inner_size /
  kda.head_dim`, and finally the `blk.<L>.ssm_a` tensor of the first
  `linear_attention` block, which is one entry per KDA head. On the published
  artifact that is `[64]`, beside `kda.head_dim = 128` and a
  `blk.0.attn_q.weight` of `[4096, 8192] = 64 * 128` — the file is
  self-consistent about a number it never names. Wherever `ssm_a` is present it
  cross-checks whatever rung answered, and a file that states the count nowhere
  and carries no `ssm_a` is refused by name, listing every place that would have
  answered.

  llama.cpp does not do this. It reads no head count for this architecture at
  all and sizes the recurrent state with `n_head() * n_embd_head_kda`, saying
  why in the file: *"note: n_embd_r()/n_embd_s() size the recurrent state with
  n_head()\*n_embd_head_kda, which works only because
  linear_attn_config.num_heads == num_attention_heads"*
  (`src/models/glm5next.cpp:121-122` at `8a8d0bcc4`). That invariant HOLDS on
  this checkpoint — `attention.head_count` is 64 and `ssm_a` is `[64]` — and it
  is a property of the checkpoint rather than of the architecture. It is the
  exact shape of the defect [#2177](https://github.com/mudler/vllm.cpp/issues/2177)
  already cost this row: a value that is right here and silently wrong on the
  next file, with no gate able to see it. The divergence is recorded here rather
  than left to be rediscovered.

  `kda.head_dim` gained llama.cpp's own fallback in the same pass:
  `src/models/glm5next.cpp:110-113` reads it optionally and falls back to
  `ssm.state_size`, and the pinned revision's converter writes only
  `ssm.state_size` (`conversion/glm5next.py:79`), so that arm is live for a file
  that branch produced rather than a legacy path.

  **THE STAGED ARTIFACT WAS NOT PRODUCED BY THE PINNED ORACLE REVISION.**
  Measured, not inferred: at `8a8d0bcc4` the `glm5next` converter writes
  `ssm.inner_size`, `ssm.state_size` and `ssm.group_count` and calls
  `add_kda_head_dim` NOWHERE (`git grep add_kda_head_dim` at that object returns
  only `conversion/bailingmoe3.py:60`, `conversion/kimi_k3.py:217` and
  `conversion/kimi_linear.py:103`, none of which is `Glm5NextModel`'s chain).
  The staged artifact carries the opposite set: `kda.head_dim = 128`,
  `ssm.conv_kernel = 4`, and none of the `ssm.inner_size` / `ssm.state_size` /
  `ssm.group_count` trio. So the two describe different revisions of the same
  pull request, and
  [`../oracles/llama-cpp-glm5next.md`](../oracles/llama-cpp-glm5next.md)'s pin
  does not describe the file this row gates against. That is why the reader
  accepts BOTH sets rather than the pinned one, and it is a caveat on any future
  llama.cpp denominator taken on this artifact at that pin.

  **THE NEW STOPPING POINT, measured 2026-08-29 on one tree and one build
  directory**, three legs of one probe object driven through
  `LoadedEngine::FromModelDir` on `device = kCPU` at
  `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf`,
  headers only, no tensor materialised:

  ```text
  baseline reader      : vt: glm5_next gguf: attention.key_length_mla - attention.key_length
                         is -256 but rope.dimension_count is 0
  MLA convention only  : vt: glm5_next gguf: missing metadata key glm5next.attention.linear_head_count
  both                 : tokenizer: unsupported tokenizer.ggml.pre "glm4"
  ```

  The middle leg is why both keys had to move together: fixing the convention
  alone moves the refusal exactly one key along, which is what `head_count_kv`
  did before `swiglu_clamp_exp`. The third leg is past config resolution
  entirely — `Glm5NextHfConfigFromGguf` returns, and the refusal comes from
  `src/vllm/tokenizer/tokenizer.cpp::FromGguf`, which maps seven pre names and
  not `glm4`. **That is the next milestone and it is
  [#2277](https://github.com/mudler/vllm.cpp/issues/2277)**, which also records
  the one thing the mapping is not free on: `b10451:src/llama-vocab.cpp:2259`
  sets `special_bos_id = LLAMA_TOKEN_NULL` for this pre-type while the artifact
  states `tokenizer.ggml.bos_token_id = 154822`, so a port that reads the id and
  prepends it emits a token no reference run emits. **That sentence is wrong on
  its first half and O21 corrects it:** `:2259` is a DEFAULT the file's own kv
  overwrites at `:2559-2578`, so llama.cpp keeps 154822 and merely declines to
  PREPEND it. The conclusion — do not prepend — survives; the mechanism does
  not, and the mechanism is what a port mirrors.

  **Still not loaded.** Reaching the tokenizer is not fitting: O10, the weight
  loader's refusal by name, stands unchanged. **The memory half no longer does,
  and this paragraph was corrected when
  [#2247](https://github.com/mudler/vllm.cpp/issues/2247) merged into the branch
  carrying it.** It said O18's 426.72 GiB resident cost and #2247's keep-quant
  `vec_dot` both stood; #2247 has since landed the two CPU kernels, and O18 and
  O19 now read 101.14 GiB. Reaching the tokenizer was never a fitting claim
  either way. No token was produced and none is claimed.

- **O21 — the `glm4` PRE-TOKENIZER is DISCHARGED, and the loader now stops in the
  WEIGHT LOADER.** [#2277](https://github.com/mudler/vllm.cpp/issues/2277).

  **The number is O21 and not O19.** `origin/main` at `785d4304f` carried O1 to
  O18 plus O20, and so did `a36add6a8`, the base this branch was cut from; [#2256](https://github.com/mudler/vllm.cpp/issues/2256) was adding an
  O19 on a branch that had not merged. Two branches that each append an `O19`
  produce a duplicate rather than a conflict, so this entry skipped the number
  for the same reason O20 did. **The gap is now CLOSED** — #2256 merged
  `origin/main` into itself, and O18 to O21 run consecutively above.

  **THE SPLITTING RULE IS EXACT, AND THE COMPARISON IS OVER BYTES.**
  `tok::Tokenizer::FromGguf` now maps `glm4` and `chatglm-bpe` — exactly the two
  names llama.cpp maps to `LLAMA_VOCAB_PRE_TYPE_CHATGLM4`
  (`b10451:src/llama-vocab.cpp:2256-2258`) — onto `SplitPattern::kLlama3`. That
  is not an approximation. Extracted from the pinned object rather than read off
  the page, on 2026-08-29, in a fresh bare clone fetched at depth 1:

  ```sh
  git cat-file -p 10bf611e533d81f739128304991c5e133c6aebd8:src/llama-vocab.cpp
  # sha256 3fea10f4481b504d5ca894b32fc177bf2eb83ffdf3f38f3f9c9175f62f62cd4b, 4427 lines
  sed -n '289p' llama-vocab.cpp   # LLAMA_VOCAB_PRE_TYPE_LLAMA3's one regex   (case at :283)
  sed -n '398p' llama-vocab.cpp   # LLAMA_VOCAB_PRE_TYPE_CHATGLM4's one regex (case at :396)
  ```

  Both RAW lines, indentation included, are md5
  `9000538f3f07df64ebcc73e41b916cab`; `diff` and `cmp` of the two are rc=0; the
  two string literals with leading whitespace stripped are sha256
  `4ec934e1de5157434e9663b9b7c8421e5396e50d5fc427a2bb9f99fca0f51a05`. Each arm
  is a ONE-element `regex_exprs` list, so there is no second stage on either
  side to differ in. `test_bpe.cpp` carries both literals transcribed and checks
  them equal, which makes the claim executable; the sha above is what makes it
  *evidence*, because a transcription cannot gate what it transcribes.

  **WHAT DELIBERATELY DID NOT COME WITH THE ALIAS.** llama.cpp's `llama-bpe`
  arm (`:2157-2159`) sets `ignore_merges = true` and `add_bos = true` beside its
  pre-type; the `glm4` arm sets NEITHER. Sharing one `SplitPattern` therefore
  had to carry the split rule and none of those flags, and it does:
  `ignore_merges_` stays false on every GGUF path, and no GGUF path prepends a
  BOS.

  **#2277's BOS PREMISE IS FALSE, AND THE TRUE STATEMENT IS NARROWER.** That
  issue and O20's forward pointer both say llama.cpp DISCARDS the artifact's
  `tokenizer.ggml.bos_token_id = 154822`. It does not. `:2259` sets
  `special_bos_id = LLAMA_TOKEN_NULL` on this arm, but that assignment is a
  DEFAULT and it is overwritten a few hundred lines later in the SAME function
  (`llama_vocab::impl::load`, `:1923`): the loop at `:2559-2578` walks
  `special_token_types` (`:2537-2538` binds `LLM_KV_TOKENIZER_BOS_ID` to
  `special_bos_id` BY REFERENCE) and assigns `id = new_id` whenever the file
  states the key and the value is in vocab range. It is straight-line code, so
  llama.cpp finishes this load with `special_bos_id = 154822`.

  What llama.cpp declines to do is PREPEND it. The prepend at `:3382-3384` tests
  `add_bos`, which defaults `false` (`:1815`), which this arm does not set, and
  which the staged file does not state — `tokenizer.ggml.add_bos_token` is not
  among its 72 KV entries (parsed 2026-08-29 from shard 1's own KV block).

  **So the mirror is: read the id, prepend nothing** — which is what this tree
  already did, and the change makes it a pinned fact rather than an accident.
  `BosId()` reports 154822 and `template_bos_` stays `-1`, so
  `EncodeWithSpecialTokens` reduces to `Encode`. On this checkpoint that is
  load-bearing rather than academic: id 154822 is `[gMASK]` (read out of the
  `tokenizer.ggml.tokens` array, token_type 3), and the file's own
  `tokenizer.chat_template` opens with the LITERAL text `[gMASK]<sop>`. A
  tokenizer that also prepended the id would double it on every request — one
  extra token per prompt, which a shape check, a load check and a "does it
  generate" check all pass. `test_bpe.cpp` fails if `template_bos_` is ever set
  from the GGUF BOS id; that mutation was run and it reds two assertions.

  **THE NEW STOPPING POINT, measured 2026-08-29 on one tree and one build
  directory**, with the pre-name arm reverted and restored so the before/after
  is not a cross-build comparison. Driven through `LoadedEngine::FromModelDir`
  on `device = kCPU` at
  `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf`,
  headers only, no tensor materialised:

  ```text
  without the arm : tokenizer: unsupported tokenizer.ggml.pre "glm4"
  with    the arm : Glm5NextForConditionalGeneration: the GGUF config is read and
                    validated, but the weight loader is not ported (W5 owes the KDA,
                    NoPE MLA, mHC and stacked-expert weight tower). Separately, NO
                    `.gguf` of this model exists anywhere: scripts/convert-glm5-next-gguf.py
                    can write one but has never been run against the 305.78 GiB
                    checkpoint (O7). See .agents/specs/glm5-next-flash.md and issue #1998.
  ```

  1.77 s wall, 95.9 MB peak RSS — which is the arithmetic proof that no tensor
  was materialised, on a file whose weights are 100 GiB. **This is O10's refusal,
  reached at last from the published artifact.** Every step above the weight
  tower now passes on a real file: four shards opened, 1412 tensors sized, the
  config resolved and validated, and the vocabulary — 154880 tokens, 321649
  merges — built. The next milestone is `load_weights` itself, which W5c
  ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) owns.

  **Still not loaded, and no token is claimed.** Reaching the weight loader is
  not fitting: O10 stands unchanged. **The memory half does not, and this
  paragraph was corrected when
  [#2247](https://github.com/mudler/vllm.cpp/issues/2247) merged into the branch
  carrying it.** It said O18's 426.72 GiB resident cost and #2247's keep-quant
  `vec_dot` both stood; #2247 has since landed the two CPU kernels, and O18 and
  O19 now read 101.14 GiB against ~119.63 GiB. The artifact FITS and still does
  not LOAD, which are two different sentences: O10 is a weight-tower gap, not a
  memory one.

  **STILL OWED, and filed rather than papered over:
  [#2279](https://github.com/mudler/vllm.cpp/issues/2279).** `FromGguf` never
  reads `tokenizer.ggml.add_bos_token` at all. llama.cpp does
  (`:2585-2586`), and that flag is the only thing that decides the prepend, so a
  GGUF declaring it `true` gets a BOS from llama.cpp and none from us. Nothing is
  red today because no artifact this row touches states the key, and the `glm4`
  arm is correct WITHOUT it — but the divergence is general to every GGUF this
  tree loads, and it is already live on the `llama-bpe` family in the other
  direction, masked only because that path has never been token-gated against
  llama.cpp with `add_special = true`. Out of #2277's scope, which is one pre
  name.

- **O22 — what W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242))
  did NOT do, named so the next wave does not have to infer it.**

  - **No materialized load, and therefore no peak RSS, no token and no speed.**
    **The first clause of this bullet stopped being true on 2026-08-30 and O28 is
    the corrected surface**; it is kept as the measurement W5c made rather than
    rewritten, the way O5 and O18 keep theirs. A materialized load of this
    artifact now exists on `dgx:gpu0`
    ([#2343](https://github.com/mudler/vllm.cpp/issues/2343)); peak RSS, a token
    and a speed number still do not.
    The load was driven at the staged artifact HEADERS ONLY: all four shards
    open, the config resolves, and all 1383 backbone tensor names resolve at 41
    MB peak RSS. A materializing load WAS attempted on this box and STOPPED at
    8.09 GiB RSS in 2m02s of uninterruptible-sleep I/O over CIFS. The real one
    belongs on `dgx:gpu0` under an `rc` lease with the artifact on local disk,
    and it is W7b's ([#2225](https://github.com/mudler/vllm.cpp/issues/2225)).
  - **The bridge from `OwnedTensor` to the host references is W5b's.** The tower
    mirrors `Glm5NextKdaLayerWeights`, `glm5_next_dsa::IndexerWeights` and
    `HcSite` field for field, but W2/W3/W4 consume `std::vector<float>` and this
    tower is block-resident by necessity — the artifact fits only because 736 of
    its tensors keep their ggml blocks, and a float tower would be 4x the file.
    Whoever writes the forward decides whether to decode per layer or to go
    device-native; nothing here forecloses either.
  - **The fused MoE seam is still NOT reached, and O19 stays live.** AGENTS.md
    `## Shared seams` routes mergeable MLP projections through
    `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`, and O19 records that
    the moment this row does so on CUDA, `MoeGateUpSwiGLUGroupedCuda` throws
    because neither IQ2_XS nor IQ4_XS is in `IsCudaKeepQuantSupported`. W5c is a
    LOAD and reaches no GEMM, so it does not make that live — but it is now the
    only thing standing between the artifact and that throw, and W5b must read
    O19 before it routes anything.
  - **The MTP block is read, counted and dropped**, which is what the reference
    does. `Glm5NextWeights::mtp_block_tensors_dropped` is 29 on the published
    artifact. Nothing consumes it and O2 still owns the head.
  - **The vision tower is refused up front rather than one tensor at a time.**
    The `glm5next` container is text-only — the published artifact ships its
    tower as a separate `mmproj-BF16.gguf` and llama.cpp #27752 drops the vision
    tensors at convert time — so a config declaring a `vision_config` alongside
    a text-only file is refused by name at the top of the load. W6 owns the arm,
    and O4 owns the fact that no llama.cpp revision can open that mmproj.
  - **The converter has never been RUN since #2291 moved it.** Its three
    corrections — `ssm_dt.bias`, the `kv_b_proj` split with the k half
    transposed, and `ssm_a = -exp(A_log)` — are gated on synthetic fixtures by
    `tests/scripts/test_convert_glm5_next_gguf.py` and by the C++/Python interop
    case, and on nothing else. O7 carries the run itself.
- **O23 — W5's MoE block is NOT REACHED from a production entry point, and the
  decoder layer that would reach it is a WAVE and not a paragraph.**
  `src/vllm/model_executor/models/glm5_next_moe.{h,cpp}` is a host reference and
  the only call sites at this merge commit are the focused gate's. This is the
  staged-slice disclosure AGENTS.md "Nothing lands dead" requires, declared
  rather than claimed by silence, and it is narrower than it looks: W5's OTHER
  deliverable, `MakeGlm5NextKVCache`, IS reached, through the production
  `make_kv_cache` factory hook, and deleting that row is a compile error. **The
  wiring belongs to W5b**, on row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, tracked by
  [#2241](https://github.com/mudler/vllm.cpp/issues/2241) under campaign issue
  [#1998](https://github.com/mudler/vllm.cpp/issues/1998).
  **Why W5 did not do it, in the specific:** the decoder layer's DSA arm has
  nothing to call. W3 landed `SelectIndexerTopk` — the indexer's SELECTION — and
  relaxed `MlaBlockDims::Validate`, and it landed no assembled
  `Glm5NextTextAttention` over either. `q_a_proj`/`q_a_layernorm`/`q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, `expand_kv`
  (`modeling_glm5_next.py:1136-1153`), the attention itself and
  `build_attention_mask_from_topk` (`:1218-1257`) are all unwritten. Landing a
  decoder layer whose sparse arm throws would be a control-flow shell with one
  live branch, which is worse than an honest split.

  **This entry is what makes O19's disclosure checkable, because it lands the
  file O19 names.** O19 states that "`glm5_next_moe.cpp` is W5's host reference
  and does not use the fused seam", and until this merge it said that about a
  file no tree contained. It is true as landed and was verified rather than
  assumed: the block's only `vt` ops are `vt::MoeRouterTopK` and
  `vt::MoeCombine`, and every expert GEMM is a host `std::vector<float>`
  accumulation, so it reaches neither `vt::MergedGemmGroup` nor
  `MoeGateUpSwiGLUGroupedCuda` and the `gate/up must be the SAME CUDA
  keep-quant dtype` throw of
  [#2260](https://github.com/mudler/vllm.cpp/issues/2260) cannot fire from this
  row today. It becomes reachable the moment W5b routes the experts through the
  shared seam, which is the wave O19 already names.
- **O24 — RETIRED 2026-08-29, premise falsified by W5c
  ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)).** As written by W5
  this entry said "`load_weights` refuses, so `ModelRegistry::Forward` is
  unreachable BY CONSTRUCTION and no wave before W5c can claim otherwise", and
  it bounded what O15, O16, O17 and O23 could be discharged by. W5c landed the
  weight tower: the GGUF arm returns a real `LoadedModel` and resolves all 1383
  backbone tensor names of the published artifact, so a handle reaching the
  `forward` hook is now real and `ForeignLoadedModel` is no longer the only way
  to reach it. The entry is kept rather than deleted because it is the reason
  the numbering skips: it was live when the tests below it were written. What it
  bounded is now bounded by the forward itself, which W5b owes.
- **O25 — W5b-1's two files are NOT REACHED from a production entry point, and
  the residency question O22 left open is now ANSWERED.** Two separate things,
  in one entry because one wave owns both.

  **The reachability disclosure.**
  `src/vllm/model_executor/models/glm5_next_attn.{h,cpp}` and
  `glm5_next_bridge.{h,cpp}` are host references, and the only call sites at
  this merge commit are `tests/vllm/models/test_glm5_next_attn.cpp` and
  `test_glm5_next_bridge.cpp`. `grep` over `src/`, `include/` and `examples/`
  for `glm5_next::Attention`, `BridgeDsaLayer`, `DecodeOwnedTensorToF32`,
  `IndexerRoleFor` and the two headers returns NOTHING outside those four files.
  This is the staged-slice disclosure AGENTS.md "Nothing lands dead" requires,
  declared rather than claimed by silence. There is no production call site to
  delete, so `.agents/reachability.md`'s reachability mutation has already been
  answered: the change has no entry-point chain, and saying so is the answer.
  **The wiring belongs to W5b-2**, on row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, tracked by
  [#2241](https://github.com/mudler/vllm.cpp/issues/2241) under campaign issue
  [#1998](https://github.com/mudler/vllm.cpp/issues/1998).

  **And one arm stays unreached even after the wiring, which is the shape that
  SURVIVES W5b-2.** `IndexerRoleFor`'s `shared` arm is CONFIG-KEYED, and the
  published `GLM-5.3-Flash` `config.json` selects it on ZERO of its 45 layers —
  the gate measures that and prints `published schedule: 0 shared layers of 45`
  rather than asserting it from prose. So the cross-layer sharing this wave
  gates is correct against `transformers` v5.16.1 on a schedule the released
  checkpoint does not contain, and the wave's own headline should be read with
  that clause attached. It is the same "unselected branch"
  (`.agents/reachability.md`) shape as the rope half, which this spec already
  discloses as "a branch no released config selects" — the difference being
  that the rope branch is REFUSED and this one is IMPLEMENTED and gated. Once
  O25's reachability half is discharged the two files become reached; the
  `shared` arm still is not, and closing that needs either a config that selects
  it or an accepted decision to leave it gated by fixture alone.

  **`byte_ceiling` is a DEFAULT ARGUMENT, so the ceiling arithmetic below binds
  the bridge and not the function.** `DecodeOwnedTensorToF32`'s third parameter
  defaults to `kBridgeTensorF32ByteCeiling` and any caller may pass a larger
  one. The STRUCTURAL claim — that `BridgeDsaLayer`, the only entry point, has
  no overload taking an expert bank — holds unconditionally. The NUMERIC claim
  holds for every call that takes the default, which is every call in this tree.
  A caller that raises the ceiling has opted out, and nothing here stops it.

  **`Numel` iterates `i < t.rank` against a fixed `shape[vt::kMaxRank]`**, so a
  hand-built `OwnedTensor` with `rank > kMaxRank` would read past the array. No
  loader-produced tensor can hold one, and the loader is the only producer this
  bridge is reachable from; recorded rather than guarded so the next reader does
  not have to re-derive that it is unreachable.

  **The residency decision: DECODE ONE LAYER AT A TIME, ON DEMAND, AND NEVER
  RETAIN THE TOWER IN FLOAT.** O22 wrote "Whoever writes the forward decides
  whether to decode per layer or to go device-native; nothing here forecloses
  either." The arithmetic forces the first, and every number is this row's own
  measurement rather than an estimate:

  | what | GiB |
  |---|---:|
  | the published `UD-Q2_K_XL` artifact, block-resident as loaded | **101.14** |
  | the same tower with every tensor expanded (`### The measured residency`) | **426.72** |
  | all-bf16 | 597.46 |
  | usable on `dgx:gpu0`, the largest device this project reaches | **~119.63** |
  | ONE bridged DSA layer, f32 | **0.4654** |
  | all ELEVEN DSA layers held at once | 5.12 |

  A decoded tower is 426.72 GiB against 119.63, which is 3.57x over — and that
  is the figure #2245 and #2247 spent six pull requests removing. A float tower
  is not expensive; it does not exist on any hardware this project can reach.
  One layer is 499,657,728 bytes, 0.39% of the box, and the caller's peak is one
  layer because the mirror is a value it can drop. There is deliberately no
  `BridgeTower`, no cache and no map keyed by layer index: each of those turns
  "one layer" into "every layer visited so far", which is the tower again with a
  slower ramp.

  **Device-native was NOT chosen, and the reason is not preference.** There is
  nothing to be device-native against. Every glm5_next primitive on this row is
  a host f32 reference and W3's CUDA arm is committed and UNMEASURED for want of
  a `dgx:gpu0` lease. A device bridge would land beside a device forward that
  does not exist, which is the "unpassed parameter" shape. W5b-2 revisits it.

  **O19 / [#2260](https://github.com/mudler/vllm.cpp/issues/2260) stays live and
  this bridge cannot make it reachable**, gated rather than argued. Structurally
  there is no overload taking `Glm5NextMoeWeights`, `Glm5NextMlpWeights` or any
  expert bank — the whole surface is `Glm5NextMlaWeights` and
  `Glm5NextIndexerWeights`, which carry no IQ2_XS or IQ4_XS tensor. Numerically
  `kBridgeTensorF32ByteCeiling` is 1 GiB, which is EXACTLY 4x above the largest
  legitimate tensor (`o_proj`, 0.25 GiB) and EXACTLY 9x below the smallest
  expert bank (`up_exps`, 9.0 GiB); both sides are asserted, because a ceiling
  above everything is a mute switch and one below the real population is a gate
  that fires on ordinary work. The check runs from the SHAPE, before any
  allocation, and the test proves it by handing the bridge a published-size bank
  carrying no bytes at all.

  **One mutation SURVIVED and is recorded as EQUIVALENT rather than chased.**
  Rewriting `BridgedDsaLayer::host_f32_bytes` to take
  `BridgedDsaLayerF32Bytes(d, id)` instead of summing the decoded buffers leaves
  every assertion green — and it must, because `DecodeShaped` refuses any tensor
  whose shape disagrees with the dims, so no reachable state separates the two
  computations. The test pins the sum against the buffers THEMSELVES rather than
  against the predictor, which is what makes the remaining agreement a fact and
  not a tautology; an earlier version asserted only
  `host_f32_bytes == BridgedDsaLayerF32Bytes(...)` and that mutation passed it.

- **O26 — `glm5_next_layer.{h,cpp}` is NOT REACHED from a production entry
  point, and O15, O16, O17, O23 and O25's reachability halves are therefore NOT
  discharged.** W5b-2a ([#2241](https://github.com/mudler/vllm.cpp/issues/2241))
  landed the decoder layer, the mHC stream threading, `TextModelForward` and the
  KV binding, and `ForwardGlm5NextForConditionalGeneration` still refuses by
  name, so the only call site at this merge commit is
  `tests/vllm/models/test_glm5_next_layer.cpp`'s. This is the staged-slice
  disclosure AGENTS.md "Nothing lands dead" requires, declared rather than
  claimed by silence, and it is stated in the STRONG form on purpose: W5b-2a's
  own scope said it would discharge those five, and it does not.
  `.agents/reachability.md` is explicit that "an intermediate hop that is itself
  unreached does not carry", so a KDA arm now called from `glm5_next_layer.cpp`
  is exactly as unreached as it was when only its own gate called it.
  **The wiring belongs to W5b-2b**, on row
  `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, tracked by
  [#2241](https://github.com/mudler/vllm.cpp/issues/2241) under campaign issue
  [#1998](https://github.com/mudler/vllm.cpp/issues/1998).

  **What DID change, stated precisely so the next reader does not re-derive
  it.** Before this merge the five primitives had five separate dead ends and no
  assembly point; after it they have ONE, gated against the reference at 10 cases
  and 1647 assertions, and the remaining gap to a production entry point is a
  single hop — the weight bridge for four arms plus the engine binding, §W5b-2b.
  That is a smaller and better-defined debt than five, and it is still a debt.

  **There is a production call site to delete after W5b-2b and there is none
  now**, so `.agents/reachability.md`'s reachability mutation is answered here
  the way it was for W5b-1: the change has no entry-point chain, and saying so is
  the answer. The mutation this change CAN run, and does, is the one for the KV
  binding's own seam — deleting the `DsaCache*` argument at
  `glm5_next_layer.cpp`'s `Attention` call reds the cached case while every
  uncached case stays green, which is what proves the cached path is entered
  through the layer rather than only by the cache test.

  **The `shared` indexer arm stays unreached even after W5b-2b**, unchanged from
  O25: the published `config.json` selects `shared` on ZERO of its 45 layers, so
  the cross-layer sharing this wave now threads through the layer loop is
  correct against `transformers` v5.16.1 on a schedule the released checkpoint
  does not contain.

  **One divergence inside the oracle itself is recorded here rather than only in
  the generator**, because it is a fact about `glm5_next` and not about this
  fixture. `Glm5NextPreTrainedModel` sets `_supports_sdpa = True`, so a default
  `Glm5NextTextConfig` resolves `_attn_implementation` to `sdpa`, and
  `build_attention_mask_from_topk` returns a BOOLEAN mask on that arm
  (`modeling_glm5_next.py:1249-1250`) where the eager arm returns the additive
  `finfo.min` one (`:1252-1256`). On a LEFT-PADDED query row, where every key is
  masked, the two backends disagree: torch's SDPA emits 0.0 and eager's uniform
  softmax emits the mean of the values, measured at 0.0 against 0.509 on this
  fixture. Our port inlines `eager_attention_forward`, which is what W5b-1 gated
  and what `:1227-1228` names as the only interface a 3-D per-(query, key) mask
  can reach, so both generators pin `cfg._attn_implementation = "eager"`. No
  token gate could see this, because the rows that differ are padding.


- **O27 — W5b-2b DISCHARGES the reachability halves of O15, O16, O17, O23, O25
  and O26, and this entry names exactly what is and is not discharged so the
  next reader does not have to re-derive it.**
  [#2337](https://github.com/mudler/vllm.cpp/issues/2337), split out of
  [#2241](https://github.com/mudler/vllm.cpp/issues/2241) because #2241's
  append-only index row is spent on W5b-2a — the same split W5b-1 took to
  [#2324](https://github.com/mudler/vllm.cpp/issues/2324).

  **DISCHARGED.** `ForwardGlm5NextForConditionalGeneration` no longer refuses by
  name. `ModelRegistry::Forward` opens a real `Glm5NextLoadedModel`, bridges each
  decoder layer on demand and runs `TextModelForward`, so the entry-point chain
  is complete and every hop on it is reached:

  ```text
  ModelRegistry::Forward
    -> ForwardGlm5NextForConditionalGeneration   (glm5_next_registry.cpp)
    -> glm5_next::Glm5NextHostForward            (glm5_next_forward.cpp)
    -> Glm5NextGgufLayerSource::Layer            -> BridgeKdaLayer / BridgeDsaLayer
                                                    / BridgeMlp / BridgeMhcSite
                                                    / BridgeMoeLayer + GgufExpertSource
    -> glm5_next::TextModelForward               (glm5_next_layer.cpp)
    -> DecoderLayerForward -> Glm5NextKdaLayerForward (O15)
                           -> MhcPre / MhcPost / HcHeadCollapseMean (O16)
                           -> Attention -> SelectIndexerTopkFromPacked (O17)
                           -> MoeForward (O23)
                           -> BridgedDsaLayer / DecodeOwnedTensorToF32 (O25)
  ```

  **The reachability mutation ANSWERS the question rather than restating it.**
  `.agents/reachability.md` asks for the production call site to be deleted in a
  scratch copy: deleting the `Glm5NextHostForward(...)` call in the registry hook
  and returning an empty `ForwardLogits{}` REDS `test_glm5_next_forward` at 11 of
  118 assertions (M1, tree `91354df62`, mutant proved applied by diff hash, BUILD
  rc=0, restored byte-for-byte). O26 said "there is a production call site to
  delete after W5b-2b and there is none now"; there is one now, and deleting it
  is red.

  **The smallest failing test ENTERS through the entry point.** Every case in
  `tests/vllm/models/test_glm5_next_forward.cpp` reaches the model through
  `ModelRegistry::Load` and `ModelRegistry::Forward` on the synthetic `glm5next`
  miniature. Nothing in that file calls `Glm5NextHostForward` or
  `TextModelForward` to produce the value under test; the one direct call is the
  `lm_head` chunk-size case, which needs a parameter the hook does not expose,
  and its result is cross-checked against the entry-point result.

  **NOT DISCHARGED, and each is a different kind of debt.**

  - **O1 stands, entirely.** No end-to-end token gate for this model exists or
    can exist on this fleet. What is gated is a synthetic 4-layer miniature at
    `hidden_size` 32; nothing here is a claim about the 321.32B model, and no
    token, speed or load number is claimed.
  - **O2, O3, O4, O5, O6, O7, O8, O13, O14, O19 and O18's second half** are
    untouched by this wave.
  - **O25's `shared` indexer arm stays unreached**, unchanged and for the reason
    O25 and O26 both give: the published `config.json` selects `shared` on ZERO
    of its 45 layers. The layer loop threads it and the fixture exercises it;
    no released checkpoint does.
  - **W5b-2a's `LayerCache` binding is now UNREACHED on the production path**,
    and §W5b-2b said this decision had to be taken rather than inherited. It was
    taken the way the two precedents take it: the hook re-runs the whole prefix
    each step and passes `caches = nullptr`, so `DsaCache` and the KDA
    recurrence carried across steps are exercised by
    `test_glm5_next_layer.cpp` alone. Carrying `std::vector<LayerCache>` on
    `Glm5NextLoadedModel` has no precedent in this tree and a model that cannot
    be run end to end on this fleet is the wrong place to invent one. This is
    the staged-slice disclosure AGENTS.md `## Nothing lands dead` requires,
    declared rather than claimed by silence, and it is a NARROWER debt than the
    six this wave discharged: the code is reached, the CACHED path through it is
    not.
  - **RAGGED BATCHING is owed and is refused rather than approximated.** A step
    with `num_reqs > 1` is refused by name. The two house precedents concatenate
    instead, which attends across the request boundary; on this model that is
    fluent wrong text no gate here could detect, so the divergence is
    deliberate and in the safe direction. `attn_meta.query_start_loc` sliced as
    at `kimi_linear_device.cpp` is the shape that closes it.
  - **THE DEVICE ARM is owed and is refused rather than crashed.** Every
    primitive on this row is host f32, so a non-CPU queue is refused by name
    instead of handing `vt::MoeRouterTopK` host pointers on a device queue.
    `glm5_next` is therefore on `scripts/runner-routing-allowlist.txt`. Two
    facts bound what a device port would buy today and both are already
    recorded: O19 / [#2260](https://github.com/mudler/vllm.cpp/issues/2260) —
    the artifact's IQ2_XS and IQ4_XS expert tensors have no CUDA keep-quant
    kernel, so a device decode of THIS checkpoint round-trips every expert GEMM
    to the host cores anyway — and O1, so it would land without a correctness
    gate.
  - **SPEED is not measured and is not claimed.** O6 stands. The forward
    re-runs the whole prefix and re-decodes every layer each step by
    construction; that is a residency decision and not a speed one, and W8 owns
    the axis.

  **`check-runner-routing-consistency.py` MISCLASSIFIES this model on its
  SECOND invariant, disclosed here rather than fixed.** It reports `glm5_next`
  among the "30 keep bf16-resident DBuf activations", which is false: every
  buffer on this path is host f32. The cause is the cross-TU resolution hole the
  `nemotron_h` allowlist entry already names
  ([#1410](https://github.com/mudler/vllm.cpp/issues/1410)) — the invariant reads
  the registry TU's decode file set and this model's f32 residual declarations
  live in `glm5_next_forward.cpp`. Not repaired in flow: it is a semantic
  checker change, which AGENTS.md `## Changing the rules or a checker` routes to
  its own row, spec and red-before.

- **O28 — THE ENGINE CANNOT RUN THIS MODEL, and the reason is a guard ABOVE this
  row's hook rather than anything in it.** Measured on `dgx:gpu0` 2026-08-30 by
  driving the production C ABI at the staged artifact.
  [#2343](https://github.com/mudler/vllm.cpp/issues/2343).

  **What ran.** `vllm-cli --model .../GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
  --device cpu --prompt "The capital of France is" --max-tokens 8`, from a
  `vllm-cli` built at `349df8e9a` in the leased container (Ubuntu 24.04, 20
  cores, 119 GiB total, ~85 GiB free at start), the shards read from the CIFS
  `/workspace` share.

  **THE LOAD SUCCEEDED, and that is the first materialized load of this model
  that has ever happened.** All four shards opened, the tower materialized, and
  the engine sized its caches:

  ```text
  INFO auto-fit max_model_len: reduced from 1048576 to 8192 to fit the KV cache
       (256 blocks x 32 tokens).
  INFO recurrent-state budget: reduced max_num_seqs from 32 to 1. The KV pool
       (256 blocks) holds 1 unified pages of 4288 tokens (one page = one
       4390912-byte GDN state), and each sequence owns 1 of them.
  vllm.cpp: Asynchronous scheduling is enabled (max_concurrent_batches=2)
  ```

  **THE FIRST STEP THREW, one level above this row's hook:**

  ```text
  engine-fatal: EngineCore busy loop threw: vt: model forward: 22 KV cache(s)
  from 2 published group(s) reached this forward, first
  'model.layers.3.self_attn.attn', with block tables gathered for 3 of 3
  published group(s), and no registered forward consumes a cache set keyed by
  layer name. Refusing rather than discarding an allocated KV topology in
  silence (row KV-DSV4-MULTICACHE W5 owns the consuming forward; #1925, #2068)
  ```

  `vllm-cli: completion failed (status 3)`. **No token was generated and none is
  claimed.**

  **This is the `input.multi_kv != nullptr` guard at the TOP of
  `ModelRegistry::Forward`**, landed by KV-DSV4-MULTICACHE W3
  ([#2068](https://github.com/mudler/vllm.cpp/issues/2068)). It fires for ANY
  model that publishes a multi-cache topology, BEFORE dispatch to that model's
  `forward` hook, and GLM-5.3-Flash publishes three groups (W5,
  [#2223](https://github.com/mudler/vllm.cpp/issues/2223)). It is not a defect in
  W5b-2b and W5b-2b cannot close it: the consuming forward is that row's.

  **WHAT THIS DOES AND DOES NOT DO TO O27.** O27's discharge stands in the letter
  it was made in — `ModelRegistry::Forward` dispatches to
  `ForwardGlm5NextForConditionalGeneration` when `multi_kv` is null, the focused
  gate enters through that entry point, and deleting the production call site
  reds it at 11 of 118 assertions. What is now MEASURED and was not before is
  that the ENGINE path stops above it, so **O27 must not be read as "a user can
  generate text with this model"**. It cannot. Two guards stand between the
  registration and a token, and only one of them is this row's: the `multi_kv`
  guard here, and this row's own device-arm and ragged-batching refusals.

  **W5b-2b LANDED TWO SENTENCES THIS MAKES FALSE, and they were in PRODUCT
  OUTPUT.** `docs/FEATURES.md` said "LOADS AND FORWARDS on `--device cpu`, one
  sequence at a time" and `docs/USAGE.md` said "A `glm5next` file LOADS and
  FORWARDS, on the CPU device, one sequence at a time". Both were written from
  the focused gate, which is exactly the reading this measurement corrects, and
  both are repaired in the same change that records the measurement — a record
  correction that leaves the lie in product output is not a correction.

  **WHAT O7 KEEPS AND WHAT IT LOSES.** O7 and the `docs/USAGE.md` weights row
  both said "No materialized load, peak RSS, token or speed number exists for
  this artifact". The FIRST clause is now false and is corrected in place. The
  rest stands: **peak RSS was NOT sampled** — the staging script did not measure
  it, which is a defect in the instrument and not a property of the run — and no
  token and no speed number exists. Load plus engine init took under 26 minutes
  wall (`04:10:07` to the refusal at `04:36:32`), which is a duration and not a
  throughput number and must not be quoted as one.

  **THREE INSTRUMENT DEFECTS were found and fixed on the way, recorded because
  each one produced a job that looked like a product result.** The first attempt
  died at `rc=127` because `/usr/bin/time` is not installed in the leased
  container, so the build never started. The second died in cmake because
  `/tmp/b` still held a cache keyed to the previous attempt's source path — the
  reused-build-dir shape — and cmake refused, which is the good outcome only
  because it refused. Both are fixed in the staging script. The third is open:
  the script does not sample RSS, so the peak this row owes did not come out of
  a run that would otherwise have produced it.

## Now

`ACTIVE`, 2026-08-30. **`ModelRegistry::Forward` reaches this model.** W5b-2b
([#2337](https://github.com/mudler/vllm.cpp/issues/2337), split out of
[#2241](https://github.com/mudler/vllm.cpp/issues/2241),
`CLAIM-GLM53-FLASH-W5B2B`) landed the two pieces that stood between a loaded
`Glm5NextWeights` and a token. The row's lifecycle state does not move, because
O1 does not: no end-to-end token gate for this model exists or can exist on this
fleet, and nothing in this wave is a claim about the 321.32B model.

The first piece was a RESIDENCY problem and not plumbing. `BridgeDsaLayer`
covered the 11 DSA layers; `BridgeKdaLayer`, `BridgeMlp` and `BridgeMhcSite` are
the mechanical three, and `BridgeMoeLayer` is not, because one sparse layer's
three expert banks are 27.0 GiB in f32 and the 42 sparse layers together are
1,134 GiB against ~119.63 GiB usable. `kBridgeTensorF32ByteCeiling` did NOT
move: it correctly refuses one 9.0 GiB bank. What changed is that
`num_experts_per_tok` is 8 of 288, so `MoeLayerWeights` grows a borrowed
`ExpertSource*` and `DecodeOwnedTensorRowsToF32` decodes ONE leading-axis row
range — 32 MiB, 32x under the same ceiling — with the RANGE checked against that
ceiling, so asking for all 288 rows is refused by the arithmetic that refuses
the whole tensor. `MoeForward` visits each HIT expert once, which is upstream's
own order and what bounds the peak at one expert. The second piece is
`glm5_next_forward.{h,cpp}`: `Glm5NextGgufLayerSource` holds ONE layer slot,
`TextModelForward` grows an overload over it so there is one loop and not two,
and the hook follows the `NemotronHForCausalLM` / `KimiLinearForCausalLM`
full-prefix pattern with two narrow refusals of its own — a multi-request step
and a non-CPU queue.

**IT LOADS, AND THE ENGINE STILL CANNOT RUN IT — measured, not assumed.** Driven
at the staged 101.2535 GiB artifact on `dgx:gpu0` on 2026-08-30
([#2343](https://github.com/mudler/vllm.cpp/issues/2343)), all four shards load,
the tower materializes and the engine sizes its caches; the FIRST step then
throws at the `input.multi_kv` guard at the TOP of `ModelRegistry::Forward`,
which KV-DSV4-MULTICACHE W3 ([#2068](https://github.com/mudler/vllm.cpp/issues/2068))
landed and which fires for ANY multi-cache model before dispatch to its hook.
**No token was generated.** O28 carries the run, the two product sentences W5b-2b
landed that it falsifies, and the three staging-script defects found on the way.

**O27 records what that discharges and what it does not.** The reachability
halves of **O15, O16, O17, O23, O25 and O26 are DISCHARGED**: the chain from
`ModelRegistry::Forward` down to the KDA arm, the mHC bricks, the DSA indexer,
the MoE block and W5b-1's two files is complete, and deleting the production
call site reds the focused gate at 11 of 118 assertions. Still owed and named in
place: the `shared` indexer arm (no released config selects it), W5b-2a's
`LayerCache` binding (the full-prefix recompute does not call it), ragged
batching, the device arm, and O1, O6 and O19 unchanged. Two of thirteen
mutations survived the first suite and both were repaired rather than disclosed
— a fixture whose mHC ramps saturate every sigmoid, and a one-token case that
could not see a missing per-expert grouping.

The next actions are W6 (the vision tower, processor and placeholder expansion),
W7b (the first fitting artifact, whenever the developer grants a large-asset
download), and the two narrow debts O27 names: ragged batching and the device
arm.

### Before W5b-2b

`ACTIVE`, 2026-08-29. The row's lifecycle state does not move: W3
([#2213](https://github.com/mudler/vllm.cpp/issues/2213),
`CLAIM-GLM53-FLASH-W3`) landed the critical-path geometry — `MlaBlockDims`
accepts the NoPE layer, discharging O11, and the DSA indexer's k-pool
compression and always-kept ragged tail are ported and gated against the pinned
transformers reference. W5 can now assemble a forward; nothing on this row loads
yet, because the loader and `Forward` still refuse by name (O10) and the wiring
is W5's (O17). The CUDA arm of W3 is committed and unmeasured; O17 carries that debt too,
deliberately, rather than opening an O-number a concurrent wave may be using. The row
reached `ACTIVE` on 2026-08-27 through W7a
([#2011](https://github.com/mudler/vllm.cpp/issues/2011),
`CLAIM-GLM53-FLASH-W7A`), which landed the first product code: the
safetensors→GGUF converter and its synthetic-fixture gate, with the Q2_K, Q6_K
and Q8_0 encoders byte-identical to the pinned llama.cpp `b10451` reference.

W1 ([#2067](https://github.com/mudler/vllm.cpp/issues/2067)) then made the
architecture RESOLVE. `Glm5NextForConditionalGeneration` is registered from its
own translation unit, `glm5next` has its `general.architecture` dispatch row,
and both sources — a `config.json` and a converter-written GGUF — descend
through one `ParseGlm5NextParams` that mirrors upstream's `__post_init__` and
all five `validate_architecture` rejections. **O9 is discharged.**

**THE MODEL LOADS.** W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242))
landed the weight tower, so the GGUF arm of `load_weights` returns a real
`Glm5NextLoadedModel` and this architecture has a `LoadedModel` for the first
time. Driven at the staged `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL through the
production chain, HEADERS ONLY: four shards open, the config resolves to 45
layers / 34 KDA / 11 DSA / NoPE MLA, and all 1383 backbone tensor names resolve
with 0 missing and 0 unexplained at 41 MB peak RSS. `blk.45` is read, counted
and NOT built as a decoder layer. **O10 is half discharged and O7 is narrowed:
the artifact exists, and what W7b still owes is a conversion of OURS.**

**THE DSA ATTENTION BLOCK EXISTS.** W5b-1
([#2324](https://github.com/mudler/vllm.cpp/issues/2324)) landed
`Glm5NextTextAttention` — `q_a_proj`/`q_a_layernorm`/`q_b_proj`,
`kv_a_proj_with_mqa`/`kv_a_layernorm`, `expand_kv` over the checkpoint's SPLIT
half-transposed `kv_b_proj` halves, `build_attention_mask_from_topk` over W3's
selection, and the CROSS-LAYER top-k sharing a `shared` layer needs — gated
against the RUN output of `transformers` v5.16.1 at 14 cases / 160 assertions.
It also landed the `OwnedTensor` -> host f32 bridge, which ANSWERS O22's open
residency question: one DSA layer at a time, 0.4654 GiB, never the tower, whose
expanded form is 426.72 GiB against a ~119.63 GiB box. **W5b-1 is NOT REACHED
from any production entry point** and O25 carries that disclosure; the wiring is
W5b-2's.

**THE STACK ASSEMBLES, AND NOTHING REACHES IT YET.** W5b-2a
([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) landed
`glm5_next_layer.{h,cpp}`: the decoder layer with all four control-flow arms,
the mHC stream threading at both sites, `TextModelForward`, and the binding of
the DSA block to `MakeGlm5NextKVCache`'s groups 0 and 2 through an additive
`DsaCache`. Gated at 10 cases / 1647 assertions against the RUN output of
`transformers` v5.16.1 over a FIVE-layer mixed schedule at the published
`hc_mult` of 4 — including a cached 8-token prefill plus 4-token continuation
that agrees with the 12-token one-shot, and an EARLY-COLLAPSE decoy from the same
oracle run that a port threading `[T, hidden]` would match instead. The KV
binding stores the 512-wide LATENT rather than the 32,768-wide expanded K/V the
reference stores, and the `ExpandKv(a ++ b) == ExpandKv(a) ++ ExpandKv(b)`
identity that makes the two equal is now a case rather than a sentence.
**W5b-2a is NOT REACHED from any production entry point**, O26 carries that
disclosure, and it discharges NONE of O15, O16, O17, O23 or O25 — the five
primitives now have one assembly point instead of five dead ends, and that
assembly point is itself unreached.

**Nothing FORWARDS** (O10's remaining half): the FORWARD still refuses by name
and W5b-2b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) owns it —
the weight bridge for the KDA, MoE, dense-MLP and mHC arms, whose routed expert
banks are ~1,150 GiB in f32 across the 42 sparse layers and therefore need an
on-demand per-expert decode rather than four more `BridgeDsaLayer`s, plus the
engine binding from `ModelForwardInput` to `TextModelForward`. The
KV-CACHE SPEC is NOT part of that debt any more — W5
([#2223](https://github.com/mudler/vllm.cpp/issues/2223)) publishes it through
the production `make_kv_cache` hook, as the paragraph below records. No GPU gate
has moved, no materialized load has been measured (O22), and no correctness
claim about the MODEL has been made. The paragraph this
replaced said "no artifact exists and nothing loads"; both halves of that were
true when written and neither is now.

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
any production entry point (O16); **W5b** owns the wiring. W4 wrote "W5" here
and that was right until W5 split: W5 landed the MoE and the KV-cache spec and
explicitly did not land the decoder layer, because W3 left no assembled
attention block for its DSA arm to call, so the layer that reaches this code is
W5b's ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)). O23 records the
same split for the MoE.

W3 ([#2213](https://github.com/mudler/vllm.cpp/issues/2213)) then made the NoPE
MLA geometry representable and ported the DSA k-pool indexer. **O11 is
discharged.**

W5 ([#2223](https://github.com/mudler/vllm.cpp/issues/2223),
`CLAIM-GLM53-FLASH-W5`) then landed the 288+1 expert MoE and the heterogeneous
KV-cache spec — **and this row has its first REACHED capability.**
`MakeGlm5NextKVCache` replaces a refusal with three published groups, entered
through `ModelRegistry::Resolve` and the production `make_kv_cache` factory hook;
unwiring that hook reds the gate, and deleting the row does not compile, because
`-Werror=unused-function` fires on the function the factory is the only
reference to. The MoE binds rather than reimplements — `vt::MoeRouterTopK`'s
grouped `noaux_tc` arm and `deepseek_v4::ClampedSwiGLU` at `alpha=1, beta=0` —
and is gated at the PUBLISHED 288/top-8 on SET equality of the selected experts
with the separation margin printed, because top-k error is bimodal.

**W5 SPLIT, and the reason is a gap W3 left rather than a scope decision.** The
decoder layer and the assembled `Glm5NextTextModel::Forward` need an assembled
`Glm5NextTextAttention` over W3's indexer, and there is none: the selection
landed, the block did not. They are **W5b**; the weight tower and `load_weights`
are **W5c**, and W5c has since LANDED
([#2242](https://github.com/mudler/vllm.cpp/issues/2242)). The MoE is still not
reached (O23), but the reason is no longer that nothing on this row can be: when
W5 was written `ModelRegistry::Forward` was unreachable by construction because
`load_weights` refused, and that is retired as O24. `load_weights` now returns a
real `LoadedModel`, so what is missing is the decoder layer W5b owes, not a
handle.

**The published artifact was measured, not assumed — and the reading has since
been SUPERSEDED, which is why it is kept as a dated measurement rather than a
state.** `unsloth/GLM-5.3-Flash-GGUF` rev `d425e572f`, arm `UD-Q2_K_XL`, run
through `LoadedEngine::FromModelDir` on 2026-08-29: it opened the file, resolved
`glm5next`, walked the 4-way split, and stopped on
`blk.3.ffn_gate_exps.weight has unknown ggml type id 17` (IQ2_XS). It does not
stop there now — [#2245](https://github.com/mudler/vllm.cpp/issues/2245) landed
that decoder and W5c resolves all 1383 backbone tensors. The arm mixes EIGHT ggml encodings and six are undecodable here —
Q3_K/Q4_K/Q5_K (O8) and IQ2_XS/IQ3_XXS/IQ4_XS (O5). **"Q2_K" is a floor, not a
format**, and O7's premise is superseded by a harder debt than the one it named:
a weight tower alone will not load this file. §W5 carries the census.

W5 also repaired three refusal messages that named landed waves as owing and
denied an artifact that exists
([#2230](https://github.com/mudler/vllm.cpp/issues/2230)); the gate had been
pinning all three, which is why they survived W2, W3 and W4 landing.

No GPU gate has moved: `dgx:gpu0` was held by other sessions throughout W5's
window, `strix:gpu0` cannot hold the artifact or run a CUDA kernel, and W3's
committed CUDA arm remains unmeasured (O17). GPU gates stay `PENDING` with the
reason recorded rather than a result invented.

The next actions are W5b and W5c, then W6, and — whenever the developer grants a
large-asset download or six quant decoders exist — W7b.
