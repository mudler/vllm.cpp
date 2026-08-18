# LTX-2.5 — 21B joint video+audio flow-matching DiT, and the generalized video seam

**Rows:** `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model` (model-matrix),
`ROAD-V1-LTX25` (roadmap portfolio).
**Issue:** [#435](https://github.com/mudler/vllm.cpp/issues/435).
**Branch:** `row/MODEL-DIFFUSION-LTX25` (ONE PR for the whole campaign — developer-directed
2026-08-11; AGENTS.md retired per-class line budgets, so size is a review judgement).
**Upstream (architecture):** Lightricks `LTX-2` — `packages/ltx-core/src/ltx_core/`.
**Upstream (serving oracle):** vLLM-Omni `vllm_omni/diffusion/` — see §3, it does NOT yet
carry 2.5.
**Checkpoints:** `Lightricks/LTX-2.5` (gated: auto), `vonkaiser/LTX-2.5-FP8-NVFP4` (ungated).
**Status:** phases L1–L8. **L8's spec was written AFTER its implementation** (§8.1), which
violates the spec-before-code rule; the operator's error, recorded rather than backfilled
silently.

---

## 0. Honesty statement — what is and is not claimed

LTX-2.5 is **not an autoregressive LLM**. It is a joint video+audio **diffusion
transformer**: one request runs a flow-matching denoise loop in which a 21.00B DiT is
forwarded once per step over two coupled modality streams, and the resulting latents are
decoded to frames plus a waveform by two VAEs. There is no KV cache in the LLM sense, no
sampler, no logits, and **no token-exact gate** — the SACRED near-tie methodology this
project uses for decoders does not apply, exactly as recorded for MiniMax-H3
([minimax-h3](minimax-h3.md) §0).

Three things are recorded as **owed** here, before any work starts, so they cannot be
discovered later:

1. **The speed gate lands `PENDING`.** vLLM-Omni's only route to 2.5 is its
   `DiffusersAdapterPipeline`, which is a black box — `supports_step_execution = False`,
   `supports_request_batch = False`
   (`vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py:68-69`).
   A throughput number taken through it is **not vLLM's production configuration**, which
   AGENTS.md §Gates requires as the denominator. Correctness is gateable through it;
   throughput is not. The axis stays open with a named next step (§9).
2. **DiffVAE is refused, never silently downgraded.** The higher-quality video decoder is
   `NADiffusionDecoder`, built on neighborhood attention. Until its row lands, asking for it
   fails with a message naming the missing piece; it does not quietly fall back to the Conv
   VAE and return a worse render as if it were the requested one.
3. **No render-quality claim.** Structural e2e (correct shapes, finite values, valid MP4)
   is not a quality result. H3 taught this directly: its fp4-resident e2e *ran* and produced
   a valid mp4 while the frames were a non-scene patch grid.

## 1. Architecture — measured, not inferred

Read by HTTP range request from the ungated
`vonkaiser/LTX-2.5-FP8-NVFP4` → `transformer/ltx-2.5-22b-distilled-fp8.safetensors`
(6124 tensors, 881,048-byte header; no payload downloaded — the same technique used for
H3's manifests).

| Field | Value |
|---|---|
| Parameters | **21.00B** — blocks 18.560B + audio connector 2.016B + global 0.427B |
| Blocks | **48**, 386.7M each |
| Video stream | hidden **4096**, 32 heads x 128 |
| Audio stream | hidden **2048**, 32 heads x 64 |
| `in_channels` / `out_channels` | 128 video (`patchify_proj` [4096,128], `proj_out` [128,4096]); 128 audio |
| Video FFN | `ff.net.0.proj` [16384, 4096] → `ff.net.2` [4096, 16384], **NO bias** |
| Audio FFN | `audio_ff.net.0.proj` [8192, 2048] → [2048, 8192], **WITH bias** |
| Activation | `gelu-approximate` (`model_configurator.py:31`) |
| Norms | `standardization_norm=rms_norm`, `qk_norm=rms_norm`, `norm_elementwise_affine=False` |
| Audio connector | 8 x 1-D transformer blocks + `learnable_registers` [128, 2048] |
| Quant (FP8 arm) | F8_E4M3 + **per-tensor F32 `weight_scale`**; biases/norms BF16; 1775 FP8 tensors |

The filename says `22b`; the measured count is **21.00B** and the Diffusers card says ~19B.
The measured number is the one this spec uses.

**The `ff_bias` cross-check.** `ff` carries no bias while `audio_ff` does. That is exactly
`ff_bias=false` / `audio_ff_bias=true`, whose defaults
(`model_configurator.py:78-80`) are documented as *"Default True keeps backwards
compatibility: pre-2.5 checkpoints lack these keys and retain FFN biases. LTX 2.5 (gemma4)
sets ff_bias=false."* Checkpoint and source agree — that agreement, not either alone, is
what AGENTS.md §"Verify against both the running oracle and its source" asks for.

### 1.1 The block — `BasicAVTransformerBlock` (`transformer.py:87`)

Per block, two coupled streams:

| Tensor | Shape | Role |
|---|---|---|
| `scale_shift_table` | [9, 4096] | 3 groups of 3: `slice(0,3)` self-attn pre-mod, `slice(3,6)` FFN pre-mod, `slice(6,9)` cross-attn `shift_q, scale_q, gate` (`transformer.py:240,273,401`) |
| `audio_scale_shift_table` | [9, 2048] | same, audio (`:302,320,410`) |
| `prompt_scale_shift_table` | [2, 4096] | **prompt K/V modulation — see §1.2** |
| `audio_prompt_scale_shift_table` | [2, 2048] | same, audio |
| `scale_shift_table_a2v_ca_video` | [5, 4096] | audio↔video cross-attn modulation (`:337,378`) |
| `scale_shift_table_a2v_ca_audio` | [5, 2048] | same (`:347,369`) |

Attentions per block: `attn1` (video self), `attn2` (video↔text, cross_dim 4096),
`audio_attn1`, `audio_attn2` (cross_dim 2048), plus the two cross-modal
`audio_to_video_attn` (`transformer.py:154`) and `video_to_audio_attn`
(`transformer.py:166`).

**Per-head gated attention** is on for every one of them: `to_gate_logits` is
`torch.nn.Linear(query_dim, heads, bias=True)` (`attention.py:513-514`) — hence the
`[32, dim]` weights in the checkpoint, one logit per head — applied *after* the attention
output as `out = self.gated_attention_function(x, out, self)` (`attention.py:577`). H3 has
no analogue; getting this wrong yields a plausible-but-wrong render rather than an error.

### 1.2 The prompt K/V cache — RETRACTED as a win for the shipped checkpoint

> **RETRACTION, 2026-08-12.** This section previously called a timestep-independent prompt
> K/V "the free win" and made it the headline of this spec, of issue #435 and of the PR.
> **It does not apply to the checkpoint this campaign actually runs.** The mechanism is real
> and correctly implemented; the claim that 2.5 enables it was wrong, and it was wrong
> because the spec author read `transformer.py:441` and stopped two lines early.

**What upstream actually does.** `apply_cross_attention_adaln` (`transformer.py:420-447`):

```python
kv_modulation = prompt_scale_shift_table[None, None].to(...)      # :441
if prompt_timestep is not None:                                   # :442
    kv_modulation = kv_modulation + prompt_timestep.reshape(...)  # :443
```

Line 441 alone was quoted as proof of "no timestep term at all". Lines 442-443 add one
whenever `prompt_timestep` is not None, and the comment immediately above them says so
outright: *"With the prompt-side AdaLN MLP disabled (use_prompt_adaln_single=False),
prompt_timestep is None and only the static per-block table applies, so K/V are
timestep-independent and cacheable across denoising/AR steps. Otherwise the
timestep-conditioned MLP output is added on top."*

`model.py:223-227` builds the MLP conditionally:

```python
self.prompt_adaln_single = (
    AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2)
    if self.cross_attention_adaln and self.use_prompt_adaln_single else None)
```

**What the shipped checkpoint carries.** Read from
`ltx-2.5-22b-distilled-transformer-fp8.safetensors`, 12 tensors that only exist when the MLP
is built:

| Tensor | Shape |
|---|---|
| `prompt_adaln_single.emb.timestep_embedder.linear_1.weight` | [4096, **256**] |
| `prompt_adaln_single.emb.timestep_embedder.linear_2.weight` | [4096, 4096] |
| `prompt_adaln_single.linear.weight` | [8192, 4096] (= 2 x 4096) |
| `audio_prompt_adaln_single.*` | the audio twin |

The **256** is the sinusoidal timestep input width. There is a prompt-side timestep MLP.
Therefore `use_prompt_adaln_single` is TRUE for this checkpoint, `prompt_timestep` is not
None, the cross-attention K/V **do** carry a timestep term, and caching them across denoise
steps would be **wrong**.

**Why the earlier reasoning looked sound and was not.** Three pieces of evidence were
consistent with the wrong conclusion: `model_configurator.py:74-76` genuinely documents
KV-cacheable checkpoints as setting the flag false; the checkpoint genuinely carries the
static `prompt_scale_shift_table [2, dim]` (96 of them); and line 441 genuinely has no
timestep. All three are true. None of them says the flag is false HERE — and the tensor that
settles it was never looked for. The header dump that would have shown it was filtered with
`'adaln_single' not in k`.

**What this costs, and what saves it.** No shipped defect: L2's implementer read upstream's
conditional correctly even though the brief handed it the wrong conclusion, and wrote

```cpp
VT_CHECK(!params.use_prompt_adaln_single,
         "ltx2: the prompt K/V cache is only valid when use_prompt_adaln_single is false "
         "(transformer.py:441-443); with the prompt AdaLN MLP enabled the K/V carry a "
         "timestep term and caching them would be wrong");
```

at `ltx2_dit.cpp:672`, refusing by name before any block runs. So the cache is
correct-and-inapplicable rather than silently wrong. It remains implemented, gated
bit-identical, and prompt-bound (§below) for any checkpoint that does set the flag false.

**The lesson, which is the same one §7.0 keeps teaching.** A claim assembled from three true
facts is not thereby true. The decisive test was one `grep` for `prompt_adaln_single` against
the checkpoint, and it was never run because the conclusion already looked supported.

L2 gates the cache by asserting the cached and recomputed paths are **bit-identical**.

**Correction, 2026-08-12 — the earlier claim here was too strong.** This section previously
said the bit-identity gate meant the cache "cannot silently diverge". L2's fresh review
disproved that, and the distinction is the whole point of the feature:

- Against a changed **timestep**, the cache cannot diverge. That is the property the
  checkpoint gives us, and it is real.
- Against a changed **prompt**, it silently could. The gate ran the forward twice with
  IDENTICAL inputs, so it only ever proved "same in, same out"; the cache carried no prompt
  identity and its only validity check was on SIZE. A probe that swapped in a different
  prompt of equal token count found the cache did not notice.

The failure that implies is not academic: a pipeline or server reusing one cache across two
requests whose prompts differ but tokenize to the same length renders the **second request
with the first request's prompt**, with no error, no shape mismatch and no finiteness
failure. The repair carries a content fingerprint on the cache and refuses by name on
mismatch. Recorded rather than quietly amended, because "we gate that" was written here
before it was true.

### 1.3 How this differs from MiniMax-H3

| | MiniMax-H3 (ported) | LTX-2.5 |
|---|---|---|
| Modalities | ONE packed sequence, per-row token tags | **two streams + explicit audio↔video cross-attn** |
| FFN | SwiGLU 14336 | gelu-approximate 16384, no bias |
| Gated attention | none | **per-head, every attention** |
| Text encoder | Qwen3-VL-32B-derived | **Gemma-4 12B + projections** |
| Video decode | ViT3D | Conv VAE **or** DiffVAE (neighborhood attn) |
| Extras | — | latent spatial/temporal x2 upsamplers, duration head |

### 1.4 Text conditioning is a MULTI-LAYER aggregate, not the last hidden state

Recorded 2026-08-11 while briefing L3, from the real TE checkpoint
(`vonkaiser` `gemma4-12b-with-proj-nvfp4-torchao.safetensors`, 1688 tensors) read against
`text_encoders/gemma/feature_extractor.py`.

`feature_extractor.py` takes hidden states shaped `[batch, seq_len, hidden_dim, num_layers]`,
normalizes them, and concatenates **across the LAYER dimension** to
`[batch, seq_len, hidden_dim * num_layers]`. The checkpoint confirms it: the two projections
take **188160** input features, and 188160 = **3840** x 49 — the model's 3840-wide per-layer
state across its 48 layers plus one.

| Tensor | Stored shape | LOGICAL shape |
|---|---|---|
| `text_embedding_projection.video_aggregate_embed.weight` | U8 [4096, 94080] | **[4096, 188160]** |
| `text_embedding_projection.audio_aggregate_embed.weight` | U8 [2048, 94080] | **[2048, 188160]** |
| `.bias` (both) | BF16 [4096] / [2048] | unchanged, BF16 is unpacked |

**Correction, 2026-08-12 — NVFP4 STORED widths are HALF the logical ones.** This section first
recorded 94080 = 1920 x 49, reading the U8 shapes as logical. L3 caught it. NVFP4 packs **two
values per byte along the last dimension**, so every U8 width in this file is half the real
one. The tell was in the same header all along: `model.norm.weight` is **BF16 [3840]**, and
BF16 is unpacked, so 3840 is authoritative. Three independent confirmations agree:
`model.embed_tokens.weight` U8 [262144, 1920] -> [262144, **3840**];
`model.layers.0.self_attn.o_proj.weight` U8 [3840, 2048] -> [3840, **4096**] (= 16 heads x 256);
and the projections' BF16 biases, [4096] and [2048], match the DiT's two stream widths exactly.

The layer count (48 + 1) was right; only the width moved. **Read a quantized checkpoint's
unpacked tensors to establish the width, never its packed ones** — that is the general lesson,
and it applies again at L6.

L3 also settled the variant question by execution: 2.5 uses **`FeatureExtractorV2`** (per-token
RMS), selected by four `_V2_EXPECTED_CONFIG` marker keys, with `_rescale_norm` applied
SEPARATELY per projection using that projection's own `out_features` over the Gemma hidden
size. `create_caption_projection` is not on this path — 2.5's projections are the two
`aggregate_embed` Linears inside the encoder, as §1.4 assumed.

There are at least TWO normalization variants and the right one is selected from config, never
guessed: `_norm_and_concat_padded_batch` (per-batch, per-layer masked mean and range, an `8 *`
scale, `eps = 1e-6`) and `norm_and_concat_per_token_rms` (per-token RMS, "for V2 models"). Both
are padding-side agnostic and ZERO padded positions.

**Why this is a trap and not a detail:** getting the variant, the mask handling, the reduction
axes or the layer order wrong yields conditioning that is finite, correctly shaped and WRONG.
It renders a plausible video for the wrong prompt, which no shape or finiteness check catches.
L3 gates the variant selection explicitly.

Two further facts the loader must respect, both measured:

- **The tokenizer is embedded AS A TENSOR** — `tokenizer_json` U8 [32,169,626] (~32 MB), plus
  `hf_asset__{chat_template,generation_config,processor_config,tokenizer_config}`. A loader that
  assumes a sibling `tokenizer.json` file fails on this checkpoint.
- **The TE quantization is torchao NVFP4, NOT compressed-tensors.** `weight` U8 packed,
  `weight_scale` F8_E4M3 grouped, `weight_scale_2` F32 scalar, plus a `torchao_nvfp4` U8 [240]
  marker per quantized module. H3's NVFP4 arm is compressed-tensors, so the layouts must be
  verified before any reuse rather than assumed equal (L6).

The checkpoint also carries the FULL multimodal Gemma-4 (`vision_model.*`,
`multi_modal_projector`, `audio_projector`); text-only conditioning is the scope, but the loader
must not choke on their presence.

### 1.5 The audio VAE is NOT end-to-end causal

Recorded 2026-08-12 from L4, which measured it rather than assuming it: its first causality
probes FAILED, and upstream agreed with the failure.

`causality_axis` governs the audio decoder's **convolutions**, but its `AttnBlock`s attend over
the whole (time, mel) map, so a last-frame perturbation reaches every output frame. The Conv
video decoder is not end-to-end causal either, for a different reason: `res_x_y`'s shortcut norm
is a one-group GroupNorm over (C,T,H,W) whose statistics span time.

This matters because "causal" is exactly the kind of property a port assumes and never checks.
Both are now gated in two parts: the shipped config asserts the GLOBAL reach, and a stripped
config isolates the convolution-only reach, with upstream itself supplying the expected windows
([5,8] audio, [3,4] video).

## 2. Scope

**In:** the DiT forward (both streams, gated attention, AV cross-attention, split and
interleaved RoPE); the Gemma-4 12B text encoder with its two caption projections; the Conv
video VAE, the audio VAE and its vocoder; the flow-matching pipeline including the distilled
two-stage recipe and the latent spatial x2 upsampler; the duration head; the FP8 and NVFP4
arms; the generalized `VideoEngine` seam with H3 moved behind it; `/v1/videos`; e2e on
dgx.casa.

**Out (recorded as owed, not silently dropped):** DiffVAE / `NADiffusionDecoder` (own row —
new neighborhood-attention kernel); the temporal x2 upsampler; LoRA fusion; multishot;
`int8-convrot` (ComfyUI-only quantization); multi-GPU / CFG parallelism.

**Four upstream PIPELINES are out too, and this list did not say so until 2026-08-17:**
`TI2VidTwoStagesPipeline`, `HDRICLoraPipeline`, `DubItPipeline` and
`KeyframeInterpolationPipeline`. See `## Owed` for what each is blocked on and its issue.
`TI2VidTwoStagesHQPipeline` (#921) and `DFRPipeline`'s rounds loop (#986) were already filed.

## 3. The oracle problem, and its resolution

vLLM-Omni does **not** support LTX-2.5. Its recipe table keys on
`("one_stage","2")`, `("one_stage","2.3")`, `("distilled_two_stage","2")`, `("dmd2","2")`,
`("dmd2","2.3")` (`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:162-166`), and
`resolve_ltx_pipeline_recipe` raises `ValueError` on anything else. Upstream issues
[vllm-omni#6066 "[New Model]: LTX-2.5"](https://github.com/vllm-project/vllm-omni/issues/6066)
(filed 2026-08-11) and [#4985 "Align and expand LTX support"](https://github.com/vllm-project/vllm-omni/issues/4985)
are open.

But its `DiffusersAdapterPipeline` is fully generic — it calls
`DiffusionPipeline.from_pretrained(model_id, **load_kwargs)`
(`pipeline_diffusers_adapter.py:116`) — so vLLM-Omni **can** execute 2.5 through
`--load-format diffusers` against `Lightricks/LTX-2.5-Diffusers` with diffusers installed
from main.

**Resolution (developer-directed 2026-08-11):**

- **Binding correctness oracle:** vLLM-Omni + diffusers adapter. Keeps AGENTS.md
  §"vLLM is the reference" intact.
- **Immediate cross-check:** Lightricks `ltx-pipelines`, the model author's own runtime.
  Available as soon as the `Lightricks/LTX-2.5` auto-gate is accepted, and it keeps every
  phase unblocked while `-Diffusers` access is pending.

Both are recorded per brick. Where they disagree, the disagreement is the finding.

**BINDING-ORACLE PARITY IS PENDING FOR EVERY BRICK LANDED SO FAR** (recorded 2026-08-12, from
L2's review). L1–L5 gate against the CROSS-CHECK (`ltx_core` executed at reduced dimensions),
not against the binding oracle, because `Lightricks/LTX-2.5-Diffusers` access is still
awaiting manual approval. That is legitimate under §3 and §6 and it is what "immediate
cross-check" is for — but it must be stated, not left implicit. Concretely:

| Axis | State |
|---|---|
| DiT / VAE / text-encoder parity vs `ltx_core` (cross-check) | gated, per-brick max abs diff recorded |
| DiT / VAE / text-encoder parity vs vLLM-Omni (BINDING oracle) | **PENDING** on `-Diffusers` access |
| Throughput vs vLLM's production configuration | **PENDING**, structurally, per §0 |

`docs/BENCHMARKS.md` records only the SPEED axis as pending, which understates it; the
correctness axis against the binding oracle is pending too. Neither is a failure, and neither
is a pass. §3's instruction to "record the vllm-omni SHA inline with every golden" is
therefore N/A so far rather than satisfied, and saying so is the point.

**There is still no vllm-omni parity PIN** — `.agents/upstream-sync.md` covers the vLLM repo
only. This spec inherits H3's open gap (model-matrix, H3 row: *"OPEN: there is no vllm-omni
parity PIN"*) and records the vllm-omni SHA used for every golden inline with that golden.

## 3.1 The two DiTs are NOT interchangeable quantizations of the same weights

Recorded 2026-08-12, after the developer accepted the HF licence and the first-party weights
downloaded. All six files verified: declared payload equals file size exactly, 23.01 GB total.

The ungated `vonkaiser` **FP8** DiT and Lightricks' first-party **NVFP4** DiT are both
"ltx-2.5-22b-distilled-transformer", and every phase before L6 gated against the FP8 copy
alone. They agree on 4348 of 4349 non-scale tensor names — and disagree on one:

| Family | FP8 | NVFP4 |
|---|---|---|
| `prompt_adaln_single` | 12 | 12 |
| `audio_prompt_adaln_single` | 6 | 6 |
| `video_embeddings_connector` | 129 | 129 |
| `audio_embeddings_connector` | 129 | 129 |
| **`keyframes_abs_pos_embedding`** | **1** | **0** |

`model.py:216-219` builds it only when `use_keyframes_abs_pos_embedding` is set, and its
comment reads: *"Marks tokens whose latent encodes a single standalone pixel frame.
Zero-initialized, so a checkpoint that predates it behaves identically until the parameter is
trained."*

**It is trained in the FP8 checkpoint.** Read directly: `F8_E4M3 [1, 4096]`, byte values
spread across `[24, 35, 39, 41, 42, 44, 45, 46, ...]`, with a real
`keyframes_abs_pos_embedding_scale` of `7.68899917602539e-06`. Not zeros.

So the upstream escape hatch — "behaves identically until trained" — **does not apply here**.
The two files differ in a TRAINED parameter that marks single-standalone-frame latents. On any
request that uses keyframe conditioning they are different models, not two precisions of one.

**CORRECTION, 2026-08-13 — the practical reach was OVERSTATED.** A third implementation
settles it. `huggingface/diffusers` main documents `keyframes_abs_pos_embedding` as
*"Zero-initialized in the reference; **unused by the regular distilled forward** until a
dedicated keyframes pipeline applies it after `proj_in`"*
(`src/diffusers/models/transformers/transformer_ltx2.py:1116-1119, 1199-1200`), and defaults
`use_keyframes_abs_pos_embedding` to `False`.

So for the DISTILLED forward this campaign actually runs, that tensor is **not consumed at
all**. The difference between the two files is real and the FP8 copy's parameter is genuinely
trained, but the claim below that they are "different models" on any keyframe request
overstates what it means for our path: no forward we run reads it. What remains true is the
narrow statement — the files differ, and a keyframes pipeline (which we do not have) would
distinguish them.

Consequences, and none of them are optional:

- **A parity number measured on one does not transfer to the other.** Everything gated so far
  used the FP8 copy.
- **L7 must state which DiT produced each artifact**, every time.
- **L6's by-name refusal fires on FP8 and not on NVFP4**, because the family is simply absent
  there. That is correct behaviour in both cases, but it means the two arms take different
  paths, and a test that passes on one proves nothing about the other.
- Quantization coverage differs too: FP8 carries 1775 `F8_E4M3` quantized tensors; NVFP4
  carries 1176 `U8` + 1176 `F8_E4M3` group scales, so **fewer modules are quantized**.

Checked because §1.2's retraction came from reading the shipped checkpoint and finding it
contradicted the spec. The same question asked of the second checkpoint found a second
difference. Ask it of every new artifact.

## 3.2 The THIRD implementation, and the default it disagrees with

Recorded 2026-08-13, after the developer asked whether the diffusers implementation had been
checked. **It had not been**, and that was a gap: §3 named vLLM-Omni's diffusers adapter as the
binding oracle ROUTE and recorded it PENDING on the gated `LTX-2.5-Diffusers` **weights** repo,
after which nobody read the diffusers **source** — which is public on GitHub and needed no gate
at all. AGENTS.md asks for verification against the running oracle *and* its source; this
campaign checked one implementation's source twice instead of two implementations once.

`huggingface/diffusers` main carries `LTX2VideoTransformer3DModel`
(`src/diffusers/models/transformers/transformer_ltx2.py`, 1683 lines) plus the `ltx2` pipelines.
Three results from reading it:

**(a) It confirms §1.2's retraction independently.** `use_prompt_adaln_single` defaults to
**`True`** (`:1185`), and `:677` documents `temb_prompt` as `None` only when it is `False`
("KV-cacheable"). The cacheable case is the exception, exactly as the retraction states.

**(b) It corrects §3.1's reach** — see the correction there. The keyframes parameter is unused
by the regular distilled forward.

**(c) It DISAGREES with our fallback defaults**, and that is the finding worth acting on:

| | ours (`ltx2.h:106,112`) | `ltx_core` (`model_configurator.py:66,68`) | **diffusers** (`:1176,1179`) |
|---|---|---|---|
| `av_ca_timestep_scale_multiplier` | `1` | `1` | **`1000`** |
| rope double precision | `false` | `false` | **`True`** |

We mirrored Lightricks' fallbacks faithfully, so ours are not wrong *as a port of ltx_core*.
But diffusers defaults both to what LTX-2.5 DECLARES. So a checkpoint carrying no metadata —
which is exactly the shipped `vonkaiser` FP8 DiT (§1.2, F2) — resolves under our defaults to a
configuration **neither upstream would produce**. That is weaker than the "differently
configured, not materially wrong" reading recorded at F2, and it is why L7's repair refusing a
config-less DiT unless one is named is the right shape rather than a nicety.

Owed: decide whether our fallbacks should follow `ltx_core`'s (a faithful port of one
reference) or LTX-2.5's declared values (what both references actually run). That is a product
decision, not a porting one, and it wants its own row.

## 4.1 The first-party NVFP4 DiT is SWIZZLED and HIGH-NIBBLE-FIRST

Recorded 2026-08-13 by L9a, which was told the file was linear, measured instead of building,
and returned `NEEDS_DECISION`. **The shape cannot discriminate here, and that is the trap.**

For every quantized layer in that file `N % 128 == 0` and `(K/16) % 4 == 0`, so the linear
shape `[N, K/16]` and the cuBLAS-padded swizzled shape
`[round_up(N,128), round_up(K/16,4)]` are **the same numbers** — `[4096, 256]` for
`attn1.to_q`. Our code knew only the OTHER 2-D framing of those same bytes, `to_blocked`'s
`[32*ceil(N/128), 16*ceil(G/4)]` = `[1024, 1024]`. All three describe one set of 1,048,576
swizzled bytes. A shape test alone can never separate them.

**The discriminating measurement.** The `vonkaiser` FP8 DiT quantizes the SAME base weights, so
it is an independent oracle. Dequantizing one module from both files, rows 0-127:

| reading of the NVFP4 scale | rms | corr vs FP8 | rel rms err |
|---|---|---|---|
| LINEAR / lo-nibble-first (what the brief asked for) | 0.013564 | **0.000414** | 1.786 |
| LINEAR / hi-nibble-first | 0.013564 | 0.257746 | 1.558 |
| SWIZZLED / lo-nibble-first (our current dequant) | 0.009208 | 0.032296 | 1.394 |
| **SWIZZLED / hi-nibble-first** | **0.009208** | **0.995560** | **0.0946** |

FP8 oracle rms 0.009167; swizzled/hi-first matches to 0.4%, linear is 48% high. The 9.46%
relative rms IS NVFP4 4-bit quantization error. Confirmed on 5 modules across 4 distinct
shapes, corr 0.9952-0.9956. **Control**: the same NVFP4 read against OTHER modules' FP8 weights
gives corr +0.021, +0.005, +0.002, +0.001 — so 0.9956 is signal, not method artifact.

**Independently confirmed in the model author's own runtime**, which is what loads this
checkpoint family: `ltx-core/quantization/nvfp4/linear.py:6-7` ("element 2j in the **high
nibble**"; "E4M3 block scales … **cuBLAS 128x4 tiled layout**"), `ltx-kernels/csrc/nvfp4/
quantize.cu:26-31` (`swizzled_offset`, `padded_cols == roundup(K/16,4)`),
`ltx-kernels/docs/NVFP4.md:27-29` ("expected in the default (`hi_first=True`) order").

**Two things this leaves.** `Ltx2UnswizzleNvfp4BlockScale` is already correct and framing-
agnostic — only the shape assertion in `Ltx2DequantTorchaoNvfp4ToBf16` hard-codes the
`to_blocked` framing. But `DequantNvfp4ToBf16` is low-nibble-first and this file is
high-first: **a different byte ENCODING, not a different scale indexing**, in a header shared
with the H3 and Laguna NVFP4 paths. That is why L9a stopped rather than reaching into it.

### 4.2 An open question about work already shipped

The torchao NVFP4 **text encoder** arm reads nibbles low-first, and its gate compares against
goldens produced by **our own low-first helper** — the generator's header even notes its e2m1
LUT is not torch-decoded. So the TE's nibble order has never been checked against an
independent oracle. This is exactly the failure this project has recorded before: *a gate
comparing two arms through the same helper proves consistency, not correctness.*

The DiT result does not transfer — different producer (torchao vs Lightricks `nvfp4-prequant`)
— and no second Gemma-4 checkpoint exists on the NAS to serve as an oracle. **torchao's own
source is public and defines the packing**, and that is the cross-check owed.

## 3.3 Where the two references DISAGREE, and which one we follow

Recorded 2026-08-13 from L11. §3 says: *"Where they disagree, the disagreement is the
finding."* This is the first place they actually do, and it is not a cosmetic difference.

**The noised-state composition.** `ltx_core`'s `latent_cond.py:38-39` leaves the NOISY tensor
untouched and lets the noiser compose it. diffusers writes clean tokens INTO the noisy tensor
(`pipeline_ltx2_condition.py:1002`). **The two agree only at `noise_scale == 1`.**

So at every other noise scale the conditioning differs, and the difference is invisible to any
shape, finiteness or dtype check — it is a correct-looking render of subtly wrong conditioning.
Following diffusers here would have been a silent divergence at every noise scale except one.

**The divergence is LIVE in this project, not hypothetical.** L11's reviewer executed both
compositions on shared tensors and a shared noise draw:

| `noise_scale` | max abs diff, `ltx_core` vs diffusers |
|---|---|
| 1.0 | 2.38e-07 (f32 round-off — they agree) |
| **0.909375** | **1.16e-01** |
| 0.5 | 6.38e-01 |
| 0.0 | 1.28e+00 |

`ltx2_pipeline.cpp:1083` sets `stage2.noise_scale = Stage2DistilledSigmas().front()` =
**0.909375**. So the distilled two-stage recipe — the one this campaign actually runs — sits
exactly at a 1.16e-01 divergence, five orders above any golden tolerance and invisible to every
shape and finiteness check.

**We follow `ltx_core`**, and the port asserts the noisy tensor is byte-identical, so the
choice is pinned rather than incidental. The reviewer's decisive argument is internal
consistency: L5 already landed `Ltx2GaussianNoise` as a direct port of `noisers.py:30-37`, so
adopting the diffusers write WITHOUT also replacing the noiser would double-apply the clean
tokens and be strictly wrong. The two halves must come from one reference, and one of them is
already landed. The reasoning: `ltx_core` is the model author's own
runtime and is what loads this checkpoint family, and §3 already names it the immediate
cross-check while vLLM-Omni access stays pending. This is recorded as a CHOICE with a reason,
not as an unnoticed coincidence.

**Four further divergences**, all following `ltx_core`: diffusers has no frame-count crop (it
dies in `unflatten`); `latent_log_var` is hardcoded to `uniform`; `norm3` is `GroupNorm(1)`
where ours is `LayerNorm(C)` (shape-compatible, statistically different — the kind that passes
every structural check); and diffusers has **no reference-audio conditioning at all**.

**One thing only ONE reference attests.** diffusers has **no mel front-end whatsoever**, so
slaney/slaney normalisation, centered-reflect padding and `power=1.0` rest on `ltx_core`
alone. That is recorded at the code site. A single-source fact is weaker evidence than a
cross-checked one and should be labelled as such rather than blend into the rest.

## 3.4 The tower's oracle existed all along, and BOS is the second disagreement

Recorded 2026-08-13 from L10.

**L3's blocker was a `transformers` VERSION, not a property of the tower.** L3 recorded that
the Gemma-4 tower could not be gated because `gemma4_unified` was absent from `CONFIG_MAPPING`.
Established by execution, both directions: `/usr/bin/python3` at transformers **5.3.0** raises
`KeyError 'gemma4_unified'` (reproducing the blocker exactly), while a venv at **5.12.1**
builds and RUNS it. So the tower is now held to a running upstream instead of to invariants
derived from its own output — `gemma4.h` had said "grounded + compiles" since it landed, and
compiling is not running.

**The tolerance is measured, not chosen.** The generator measures how far UPSTREAM'S OWN answer
moves between f32 and bf16, per hidden state, and emits that as the bound. The worst state
reaches 0.71x of its own floor and state 0 is bit-identical — i.e. we are closer to
upstream-in-bf16 than upstream-in-bf16 is to upstream-in-f32, at every layer. It cannot be
loosened to rescue a failure: loosening it means regenerating it, which means the oracle moved.
That is the shape every tolerance on this campaign should have had.

**The second real disagreement (§3.3 was the first): BOS.** diffusers relies on
`add_special_tokens=True`, which on THIS tokenizer adds nothing — so following it would **drop
token 0 of every prompt**. `ltx_core` does not. We follow `ltx_core`, and the measurement is
recorded in the goldens header. Tokenization is token-exact against HuggingFace over the
shipped 262144-entry vocab, four prompts x 1024 positions, first mismatching index -1 on all
four.

**The Gemma config came from an 84 KB range request.** The `vonkaiser` build has no
`__metadata__` at all, so the config had to come out of band; L10 read it authoritatively from
the OFFICIAL bf16 checkpoint's safetensors header without downloading the payload. It
independently re-confirms §1.4's corrected 188160 width.

**Why `has_encoder()` is still false, and this is the honest part.** Upstream routes each
stream through an `Embeddings1DConnector` before cross-attention. Its math is ported
(`Ltx2ConnectorForward`), but its WEIGHTS are not loaded: they ship inside the DiT file as
`video_embeddings_connector.*` / `audio_embeddings_connector.*` (**372 tensors, measured**) and
are still among the modules the loader refuses as unported. So conditioning has nowhere to go,
and flipping the flag would promise a render that cannot complete. L10 MOVED the refusal rather
than lifting it — `encoder_path` still refuses, but the message now names the connector weights
instead of the tower, because the old message became false.

## 4. Checkpoint access and placement

Verified against the HF API on 2026-08-11 with the session token:

| Repo | `gated` | Consequence |
|---|---|---|
| `Lightricks/LTX-2.5` | `auto` | Accepting the license opens it. Holds the **first-party NVFP4 DiT**, 18.72 GB |
| `Lightricks/LTX-2.5-Diffusers` | restricted (manual) | Needed for the **binding oracle**; request submitted |
| `vonkaiser/LTX-2.5-FP8-NVFP4` | none | **Unblocks L1–L2 today**: FP8 DiT + NVFP4 Gemma-4 TE |

All artifacts land under `$CHECKPOINT_ROOT` = `/mnt/nas_share/checkpoints` (per `.env`), so
dgx.casa and the cluster nodes mount one copy rather than each pulling 30 GB.

**Best GB10 arm** (119 GiB unified):

| Component | Source | Size |
|---|---|---|
| DiT, NVFP4 | `Lightricks/LTX-2.5` `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18.72 GB |
| Gemma-4 12B TE, NVFP4 | `vonkaiser` `gemma4-12b-with-proj-nvfp4-torchao.safetensors` | 7.40 GB |
| Video VAE + audio VAE, bf16 | `Lightricks/LTX-2.5` | 1.83 GB |
| Latent spatial upsampler x2, bf16 | `Lightricks/LTX-2.5` | 1.00 GB |
| | | **~29 GB** |

> **RETRACTION, 2026-08-13 — the diagnosis below is WRONG, and it was the operator's.** The
> correction that follows said the first-party NVFP4 DiT stores `weight_scale` in the LINEAR
> `[N, K/16]` layout. **It does not. It is SWIZZLED, and it additionally uses the OPPOSITE
> NIBBLE ORDER from our dequant.** Phase L9a was dispatched to build a linear arm on that
> premise, measured it first, and stopped without implementing — building it would have made
> the file LOAD and render silently wrong output. See §4.1.

**SUPERSEDED CORRECTION, 2026-08-12 — the DiT in that table does NOT load.** L8 tried it and
the loader refuses by name, before any forward:

> `'transformer_blocks.0.attn1.to_q.weight_scale' is [4096, 256] but a SWIZZLED torchao scale
> for [4096, 4096] is stored as [1024, 1024]. The LINEAR shape [4096, 256] has the same element
> count, so reading one as the other type-checks and permutes every scale within a 128x4 tile.`

The first-party NVFP4 file carries **no `.torchao_nvfp4` marker at all** and stores
`weight_scale` in the LINEAR `[N, K/16]` layout, where L6's dequant expects the swizzled one.
Same element count, so the mistake type-checks — which is exactly why the refusal exists.

This is **L6 loader debt, not a forward defect**: it throws from `Ltx2StreamDitToDevice`. But it
means the arm this table calls "best" is the one that cannot load, and nothing had materialized
a tensor from that file before L8 — L7's shipped-checkpoint test only parsed the manifest.

**What actually ran on the GB10 is the `vonkaiser` FP8 DiT** (21.0 GB, 6124 tensors), and §3.1
records that the two are not interchangeable. Until the loader learns the linear layout, the
NVFP4 arm is aspirational and the FP8 arm is the real one.

Comfortably inside the pool, and materially smaller than H3's ~41 GB GGUF arm.

## 5. Design — the generalized seam

`include/vllm.h` already models this shape: `vllm_video_model_params` carries `dit_path`,
`encoder_path`, `video_vae_path`, `audio_vae_path` as separate artifacts (ABI v12, ROW 2).
It is only the *internals* that are H3-typed — `vllm::multimodal::MiniMaxH3VideoEngine` —
plus two H3-specific fields (`partition` = fl2va/ref2va, and H3's 50-step default).

**L1 introduces `vllm::multimodal::VideoEngine`**, an abstract seam with a
checkpoint-detected registry, and moves H3 behind it **unchanged**. Per AGENTS.md
§"Shared seams", a capability not reachable through the shared surface is not done, and new
models are additive files. ABI goes to **v18 by ADDING fields only** — v12 video callers keep
working byte-identically, which the existing `test_capi` v12 section already guards.

**Correction, 2026-08-11.** Earlier revisions of this section said "v13". That was wrong: it
read the VIDEO SLICE's own v12 label as if it were the ABI counter, when `VLLM_ABI_VERSION`
was already **17** and v13 shipped long ago as `vllm_complete_tokens`. The additive
requirement was always the real one and is unchanged; only the number moves, 17 -> 18. L1
found this while implementing, which is the delegation loop working as intended.

Reuse is the point. Already ours and shared, not re-implemented: the flow-matching denoise
loop, AdaLN block plumbing, 3D RoPE construction, VAE CNN infrastructure
(`minimax_h3_vae_cnn.cpp`), WAV writing, PPM frame serialization, the ffmpeg mux argv
composer, the NVFP4 resident-weight path and Marlin W4A16 dispatch, and — for the text
tower — `gemma4.cpp` / `gemma4_weights.cpp`.

Genuinely new: dual-stream AV cross-attention, per-head gated attention, gelu-approximate
FFN, the audio embeddings connector, the two-stage distilled recipe with its latent
upsampler, and the duration head.

## 6. Phases

All on `row/MODEL-DIFFUSION-LTX25`, one PR.

| Phase | Scope | Gate |
|---|---|---|
| **L0** | This spec; issue #435; checkpoint inventory; oracle stand-up | spec committed |
| **L1** | `VideoEngine` interface + registry; H3 behind it unchanged; ABI v18 additive | H3 frames+WAV **byte-identical** to pre-refactor on the committed fold fixture; v12 `test_capi` green |
| **L2** | DiT layout + forward: dual stream, gated attn, AV cross-attn, split/interleaved RoPE, prompt-KV cache | reduced-dim CPU parity vs upstream modules; cached vs recomputed prompt K/V **bit-identical** |
| **L3** | Gemma-4 12B TE + caption projections (4096 video / 2048 audio) | parity vs upstream TE; reuses `gemma4.cpp` |
| **L4** | Conv video VAE + audio VAE + vocoder | per-brick parity vs upstream decoders |
| **L5** | Pipeline: sigma schedule, distilled two-stage, latent spatial x2 upsampler, duration head | recipe values EXACT vs upstream |
| **L6** | NVFP4 DiT + NVFP4 TE arms; GB10 load-time residency | quantized vs bf16 wiring gate; residency per the ATS finding |
| **L7** | e2e on dgx.casa under `flock`; `/v1/videos` route | valid MP4+WAV; speed axis recorded `PENDING` per §0 |
| **L8** | Device-resident DiT forward on GB10; the CUDA `vt::AttentionCross` it needed | every dispatched op `vt-native` on a CUDA queue, ZERO reference-tier hits; device-vs-host at f32 round-off against the SAME upstream goldens |
| **L9a** | NVFP4 DiT: swizzled + hi-nibble-first (§4.1) | loads AND correlates ~0.9956 against the FP8 file's dequant, not merely finite |
| **L9b** | Real render on shipped weights; `--video-family`, `--video-extra` | frames + WAV + MP4 from the 21B FP8 DiT under its DECLARED config |
| **L9c** | Connector wiring + the missing pool Drain | conditioning real, not synthetic; peak device usage measured per phase |
| **L10** | The Gemma-4 tower, so a prompt works | gated against a RUNNING upstream, tolerance MEASURED from upstream's own f32-to-bf16 spread |
| **L11** | VAE encoders + image/keyframe/reference conditioning | both encoders at f32 round-off; the conditioning items EXACT |

### 8.0 Two phases landed WITHOUT a spec, and both are the operator's failure

L8 and **L10** were dispatched and landed with no spec section, against AGENTS.md's
"committed *before* implementation, never written up afterwards". In both cases a phase brief
stood in for the spec. That is the operator's error twice over, recorded rather than backfilled
as though the order had been kept. §8.1 sets out what L8's spec would have had to say; the
phase table above now carries L9a-L11 so the remaining phases are at least declared.

The cost is measurable and was predicted in §8.1: a written scope would plausibly have caught
the ungated config adoption in L7, and — for L10 — would have forced the question "what is the
oracle for the tower?" up front, which is exactly the question whose answer turned out to be
*a newer transformers*.

### 8.1 L8 — written AFTER the fact, and that is a process failure

**AGENTS.md is unambiguous: the spec "is committed *before* implementation, never written up
afterwards."** L8 was dispatched and landed with no spec. That is the operator's error, not the
implementer's, and it is recorded here rather than backfilled as though the order had been kept.

What the spec would have had to say, had it been written first:

**Scope.** A device-resident LTX-2.5 DiT forward whose activations live in device memory, and
the CUDA `vt::AttentionCross` kernel it requires. Everything else routes through existing shared
seams (`vt::MatmulBT`, `Add`, `RmsNorm`, `LayerNorm`, `GeluTanh`, `Attention`, `AttentionCross`).

**The risk that justified the phase, and that a spec would have named first.** GB10 reports
`UnifiedMemory`, so `RegisterReferenceTier` will serve a CPU kernel to a CUDA queue. With
`vt::AttentionCross` CPU-only, all six cross-attentions per block would have executed on the
host with every gate green and "it ran on the GPU" false. The gate therefore cannot be
"the tests pass" — it has to be *provider identity per op*, which is what
`VT_OP_PROVIDER_STATS=1` reports and what the review used.

**Dtype.** bf16 is the production stream; `kF32` exists only as a gate arm, so the device
forward can be held to `ltx2_goldens.inc` at f32 round-off. Nothing widens a bf16 load.

**Stop condition that should have been written down.** "No 'it ran on the GPU' unless every
dispatched op did, proven by provider identity rather than by a passing suite."

**What the missing spec cost.** Two of the review's five MEDIUM findings are gate defects a
written scope would plausibly have caught up front: the L7 config adoption landed entirely
ungated (§7.0(c) again), and the new CUDA cross-attention's tiling machinery is never reached
by any fixture — every gate geometry is `tiles=1 npl=1 nblk=1 hq==hk`, while a real render puts
S at prompt length and Tq in the thousands, straight into the untested regime.

## 7. Tests

### 7.0 Four findings about the METHOD itself, and how they compound

Recorded 2026-08-12. These came out of the L2/L3/L4 review rounds and they change what the
evidence is evidence *of*. They matter to every future brick, not just LTX-2.5.

**(a) There is a CLASS of constants a reduced-dimension golden cannot see.** Not one
constant, a class. Confirmed independently on two phases. An epsilon, a clamp bound or a
normalize floor only becomes load-bearing in a regime the synthetic fixture never enters, so
it can be changed — sometimes by 100x — with every golden still green:

| Constant | Mutation that stayed GREEN | Why the fixture cannot reach it |
|---|---|---|
| video VAE `pixel_norm_eps` | 1e-8 -> 1e-6 | activations are O(1); the epsilons differ by ~1e-7 relative |
| video VAE `norm_eps` | 1e-6 -> **1e-4** | same |
| BWE mel log clamp | 1e-5 -> 1e-8 | `mel_basis` is built non-negative and well-scaled, so nothing saturates. **Real silence DOES saturate it in production** |
| Snake/SnakeBeta `eps`, `_RMSNorm2D` floor | -> 0.0 | never divides by a zero-norm row |
| text `range_ + eps` | -> 0.0 | only matters when a whole (batch, layer) slice is constant |
| text `denom + eps` | -> 0.0 | only matters when `sequence_lengths == 0` |

The repair is not "tighten the tolerance" — the tolerance is fine. It is a **source-anchored
constant assertion** pinning the value against the upstream line it came from, plus, where
feasible, a golden arm whose input actually enters the regime (L4 built one that attenuates
`mel_basis` so all 384 bins saturate, and it catches the mutation numerically).

**(b) Byte-identical goldens are NOT evidence that the oracle was right.** L4's repair tested
this instead of assuming it: with a decoy `ltx_core` differing from upstream by ONE drifted
constant, and path precedence defeated, the generator imported the decoy, **exited 0, and
emitted goldens whose md5 was identical to the real ones**.

That is (a) and (b) compounding. A drifted constant from the invisible class produces
identical goldens, so "the goldens reproduce byte-for-byte" cannot distinguish the right
oracle from a wrong one. Reproducibility proves determinism; it does not prove provenance.

**Therefore every generator on this campaign owes two things, and they are separate:**
assert the resolved `ltx_core.__file__` lives under the `--ltx2` checkout (identity), and
record the upstream revision SHA in the emitted goldens (provenance). AGENTS.md already
required the revision anchor; the identity assertion is what makes the anchor mean anything.

**(a-bis) The class was swept, and it had FIVE members, not one.** Recorded 2026-08-13
(issue #560). After the fourth recurrence, every stabilizing constant in the LTX-2.5 files was
enumerated and mutated ALONE — 20 constants. Three categories emerged, and the middle one is
the interesting one:

| verdict | count | meaning |
|---|---|---|
| pinned AND numerically reachable | 12 | a mutation moves a golden; the gate genuinely bites |
| **INVISIBLE — no arm read it at all** | **5+1** | a 100x change left every suite green |
| pinned, genuinely unreachable | **1** | upstream discards the value, so a pin is the only honest treatment |
| **MISLABELLED as unreachable** | **2** | live numeric path; the mutation was merely below fixture sensitivity |

**CORRECTION, 2026-08-13.** The row above originally read "3 pinned, correctly unreachable",
taken from the sweep's own table and propagated into this spec by the operator without
independent check. Its review disproved two of the three, and the failure mode is the one this
whole section exists to name: **"upstream discards it" is exactly the label a fixture gap
wears when nobody probes it.**

- `Ltx2ConvVideoDecoderConfig::norm_eps` is NOT discarded. `video_vae/resnet.py:93-97` builds
  `norm3 = nn.GroupNorm(num_groups=1, ..., eps=eps)` **regardless of `norm_layer`** whenever
  `in_channels != out_channels`. A `VT_CHECK` probe at our `norm3` call site fired on FIVE
  goldens. The 100x mutation was simply below that fixture's sensitivity; at 1.0 the goldens
  move by 1.64e-2 and 2.04e-2 against a 5e-6 band. Its ENCODER twin at the SAME 100x moves
  4.39e-5 and goes red — so the decoder's silence was an accident of fixture scale, not a
  property of upstream.
- `kLtx2RmsNorm2dEps` is a `clamp_min` FLOOR in `F.normalize`, not a discarded value. It IS read
  — setting it to `1.0` reds two encoder arms at 5.26e-4, so the path demonstrably executes —
  but the floor itself binds only on an exactly-zero channel vector.

  **SECOND CORRECTION, 2026-08-13.** This entry originally went on to claim the project's own
  BWE-quiet arm is precedent that such a probe IS constructible. A later reviewer disproved
  that, and the distinction is one I should have drawn myself: in the BWE case **silence is a
  legitimate INPUT**, so the probe is just a fixture. Here the floor binds on a **mid-network
  activation vector**, and no legitimate input zeroes one downstream of a biased convolution.
  The precedent does not transfer. So this constant is read-but-never-binding, and a pin really
  is the only honest instrument for it.

  Worth stating plainly: in correcting the sweep for calling a live constant unreachable, I
  overcorrected and called an unreachable one constructible. Both errors are the same failure —
  asserting a reachability verdict without the probe that settles it — and mine had the
  additional defect of citing a precedent I had not checked applied.
- Only `kLtx2EncoderApproxLnZero` is genuinely unreachable: `video_vae.py:325-334` concatenates
  the block and `torch.chunk(...)[0]` discards it. Mutating -30 to -1 leaves the suite green
  because upstream never reads it either.

**And the sweep missed a sixth invisible constant**, `Ltx2AttentionArgs::norm_eps`
(`ltx2.h:365`) — same shape as the DiT case, all ten call sites assign it explicitly, so a
10^6 mutation leaves every suite green. A latent trap rather than live code today, but the
"class swept" claim was incomplete by one.

**The lesson is sharper than the fix.** A change written to close this class mislabelled a live
constant as unreachable, and the operator copied the label into the spec without probing it.
Recording a hole is not removing it: the only evidence that a constant is unreachable is a
probe that FAILS to reach it, not a mutation that happens not to move anything.

The fifth instance is the one worth remembering: **`Ltx2DitParams::norm_eps`**. Every test
passes that value explicitly through `ReducedParams`, so nothing ever read the FIELD DEFAULT —
which is live code, because `ParseLtx2DitParams` falls back to it exactly as upstream's
`config.get("norm_eps", 1e-06)` does. A constant can be invisible not because the fixture
avoids its regime, but because the fixture never lets the default apply.

The repair that matters is not the pin. For the two audio-VAE `norm_eps` holes the fix was a
new gate arm at `norm_type = kGroup` / `causality_axis = kNone`, which makes the constant
**numerically** reachable: the same 100x mutation now moves the goldens by 1.13e-3 and 5.18e-3
against a 5e-6 band. A source-anchored `CHECK` is the floor; a reachable arm is the gate.

**(c) A FIXTURE that cannot separate right from wrong is the same defect, wearing different
clothes.** Recorded 2026-08-12 from L5's review, and it is the sharpest instance so far.

L5's `linspace` mirror — the two-sided walk that makes the last sigma land on EXACTLY 0 —
was declared load-bearing in a comment and was **not gated at all**. Replacing it with a
naive forward walk left 33 cases and 1512 assertions green. It is not cosmetic: for **23 of
the first 198 step counts** the naive walk misses exact zero, and at `steps=41` a 5.96e-08
terminal survives the `sigma == 0` guard, takes the shift transform, and displaces
`last_non_zero`, moving the WHOLE schedule by 0.1 so the denoise loop never reaches zero
noise. A plain `--steps 41` renders confidently and wrongly.

It hid because the fixture exercises `steps in {8,6,5,1,4,7}` — every one of which the broken
walk happens to get right.

Two other instances from the same phase, both caught before landing:

- The duration head's three arms collapsed to within 2.98e-06 at the fixture's scale, **below
  the round-off bound**, so an implementation ignoring one input stream would have passed.
  Widening the scale to 0.35 separates them by 4.9e-02.
- A guider-refusal probe matrix omitted the `B=1` row, so a claim false at batch 1 — the
  ordinary single-request shape — was recorded as a golden.

**The generalization.** (a) is about a constant the fixture never drives into its active
regime; (c) is about an INPUT the fixture never drives into the regime that discriminates.
Same failure, different axis. A golden proves only what its inputs can distinguish, so for
every brick ask: *what input would tell a correct implementation from a plausible wrong one,
and does the fixture contain it?* Sweeping a parameter (step counts, scales, batch sizes)
costs almost nothing and is what turns a golden from a witness into a gate.

**(d) A BROKEN ENVIRONMENT can impersonate a repair.** Recorded 2026-08-13, from the #516
diagnosis, and it is the variant with the worst timing.

`test_ltx2_device`'s shipped-DiT case reads its checkpoint from `LTX2_SHIPPED_DIT`. If that
path is unreadable — which on this project means simply that dgx rebooted and
`mnt-nas_share.mount` lost its boot race again, as it has every single time — the case takes
its `SKIPPED` early-return and the suite reports:

```
[doctest] test cases: 13 | 13 passed | 0 failed
[doctest] Status: SUCCESS!
```

**That is indistinguishable from the defect being fixed**, and it arrives precisely when
someone is hoping to see green. A skip that reads as a pass is worse than a failure.

**The rule, in the diagnosing agent's own formulation:** *a fixture that opts in on an
environment variable must assert the resource it names is USABLE, not merely that the variable
is SET.* `LTX2_SHIPPED_DIT` gates the case on `getenv() != nullptr`, so every downstream claim
rests on a path nothing ever checked. Nothing lies — each layer does exactly what it says — but
the composition manufactures the shape of a repair out of a broken mount.

**`SKIPPED` is only honest when the operator CHOSE not to supply the resource, never when they
supplied one that is gone.** An opt-in fixture therefore owes a readability check on its input
and a refusal by name when that fails.

**How narrowly this was missed, because the margin is the point.** The measurement hold ran
22:31:12-22:55:34; the box went down at 23:18. Twenty minutes slower in the lock queue and the
relaunch would have run against a dead mount and reported a clean green suite — and the most
likely reading of that is "the red went away after a reboot", which is the worst possible
conclusion to draw about an allocator bug that returns **wrong answers silently**.

This is (a)/(c)'s pattern from a new direction. There the INSTRUMENT could not see the defect;
here the instrument is fine and the ENVIRONMENT manufactures the shape of success. Both produce
a green that means nothing, and neither is visible in the summary line — which is why
`Status:` alone is not sufficient evidence either. Ask additionally: *did this run actually
execute the thing it claims to gate?*

### 7.1 Method

Mirroring H3's method, which is what made its bricks trustworthy: upstream's modules are
pure Python, so they are **executed at reduced dimensions on CPU** as the oracle, with both
sides rebuilding weights and inputs from an identical deterministic stream so **no weight
byte is checked in**. A generator script freezes upstream outputs; the C++ test asserts
against them.

Specific traps this port must gate, each of which produces a *plausible but wrong* result
rather than an error:

- **Per-head gate application order** — gating before vs after `to_out` differ silently.
- **`ff` bias presence** — reading a bias that is not there, or skipping one that is.
- **Cross-modal projection asymmetry** — `audio_to_video_attn.to_q` is `[2048, 4096]` while
  `to_k`/`to_v` are `[2048, 2048]` and `to_out` is `[4096, 2048]`. Transposing any of these
  still type-checks against a square assumption.
- **AdaLN slice mapping** — `slice(0,3)` / `slice(3,6)` / `slice(6,9)` are not interchangeable.
- **Prompt-KV caching** — asserted bit-identical against recomputation.
- **RoPE `split` vs `interleaved`** (`LTXRopeType`) and the optional float64 frequency path.
- **F32 `scale`-guard on tolerances** — per this project's doctest finding, `Approx` needs
  `.scale(0.0)` or a 1.19e-5 absolute floor silently accepts anything.

## 8. Risks

| Risk | Mitigation |
|---|---|
| `-Diffusers` manual access never granted | `ltx-pipelines` cross-check keeps every phase unblocked; binding oracle recorded as pending, not faked |
| Upstream lands 2.5 in vllm-omni mid-campaign | Good outcome — re-anchor goldens to the native path and record the SHA; the diffusers-adapter goldens stay as the earlier evidence |
| Gemma-4 TE differs from our ported Gemma-4 | L3 gates the TE against upstream independently before wiring; a delta is a finding, not an adaptation |
| GB10 unified-memory OOM reboots the box | Never run a large oracle alongside ctest; `flock $HOME/gpu.lock`; park `local-ai-worker` |
| Contention with the 3 other coordinators | `flock` on every GPU-executing step; named tmux; single-load steady state |

## 9. Stop conditions

- A brick whose upstream reference cannot be executed → record `NEEDS_CONTEXT`, do not guess
  the semantics from shapes.
- A parity delta that is not round-off → stop and report; never widen a tolerance to pass.
- Any temptation to declare a performance ceiling → forbidden by AGENTS.md; keep the gap open
  and name the next traceable hypothesis.
- Speed axis: stays open until vllm-omni carries native 2.5 (tracked upstream at #6066) or
  another production-configuration denominator is ratified.

## Owed

Recorded 2026-08-17. Four upstream pipelines at Lightricks/LTX-2 `fd4ded7f` had no recipe
row, no refusal, no `Ltx2UnportedPipelineFeature` marker and no issue. §2's "Out" list named
none of them. Per AGENTS.md that is the worse half of the silent/refused split: a refusal
naming a missing part is documented debt, and silence is not. Each now has its own issue,
each saying what is absent, what a future row starts from, and what blocks it.

- [#1093](https://github.com/mudler/vllm.cpp/issues/1093) — `TI2VidTwoStagesPipeline`
  (`ti2vid_two_stages.py:61`). NOT our `distilled_two_stage`: stage 1 is CFG-guided on the
  FULL model (`:247-259`), stage 2 carries the distilled LoRA alone (`:151`), and stage-1
  sigmas are scheduler-derived (`:243-245`) where ours are the fixed `DistilledSigmas()`
  table (`ltx2_pipeline.cpp:1163`). Also NOT `TI2VidTwoStagesHQPipeline`, which
  [#921](https://github.com/mudler/vllm.cpp/issues/921) owns. Blocked on a guided VIDEO
  denoise loop, and on two checkpoints absent from the NAS: the distilled LoRA and the full
  `-dev-` transformer.
  `.agents/specs/ltx25-resolution-envelope.md` already carried this under its own `## Owed`
  and said "not separately filed, because #644 already owns 'close every refused arm'". That
  record is real and is why this one was never silent; it is filed now because an umbrella
  row cannot say what THIS arm is blocked on. That bullet now points here.
- [#1094](https://github.com/mudler/vllm.cpp/issues/1094) — `HDRICLoraPipeline`
  (`hdr_ic_lora.py:229`). Four comments and one `Fail(...)` string literal
  (`ltx2_lora.cpp:246`) cite it; nothing implements it. Blocked on a LogC3 / ACEScct decode
  tail (`ltx-core/hdr.py:37-172`, applied at `hdr_ic_lora.py:624`), which this tree has zero
  of by deliberate exclusion (`ltx25-retire-dead-arms.md:167`, "no — colour science"), plus
  an HDR IC-LoRA and a pre-computed text-embeddings file, neither on the NAS.
- [#1095](https://github.com/mudler/vllm.cpp/issues/1095) — `DubItPipeline` (`dubit.py`).
  `Ltx2ConditionAudioByReference` is ported, gated (`test_ltx2_vae.cpp:2926`) and undriven;
  reference audio is already refused by name (`ltx2_video.cpp:1991-2004`). Blocked on the
  negative RoPE shift (`dubit.py:351-353`), which our one ported shift structurally cannot
  produce because it clamps at zero (`ltx2_conditioning.cpp:596-601`), and on the Dub-It
  IC-LoRA.
- ~~[#1096](https://github.com/mudler/vllm.cpp/issues/1096) —
  `KeyframeInterpolationPipeline` (`keyframe_interpolation.py`).~~ LANDED as row
  `LTX25-KEYFRAME-INTERP` ([`ltx25-keyframe-interp.md`](ltx25-keyframe-interp.md)),
  and two of the three blockers recorded here were stale by the time it was picked
  up. The per-sigma denoiser resolves ONE guider on this pipeline's default path —
  `main()` passes plain `MultiModalGuiderParams`, so
  `create_multimodal_guider_factory` takes `constant()` and builds a single
  `(inf, params)` bin (`guiders.py:312-315`) — and both checkpoints are on the NAS
  with #1148 closed at `40a796aa9`. The multi-keyframe surface is real, is NOT
  what makes this pipeline different, and is now
  [#1187](https://github.com/mudler/vllm.cpp/issues/1187). What WAS different, and
  is named in none of the above, is the conditioning BUILDER: `:211` and `:260`
  call `image_conditionings_by_adding_guiding_latent` (`helpers.py:343-367`), so
  frame 0 is a keyframe that APPENDS rather than a latent that REPLACES.
- [#1097](https://github.com/mudler/vllm.cpp/issues/1097) — `ltx2-gen` silently discards a
  second `--lora`, and `kKnownLoadExtras`' own comment still says "nine of these ten" over a
  twelve-entry array. Product code, so filed rather than fixed in this records change.
- [#1098](https://github.com/mudler/vllm.cpp/issues/1098) — `README.md` has ZERO `LTX`
  mentions (control: `minimax` = 7) and says "37 registered architectures" four times where
  `docs/FEATURES.md` says 40. **Neither can be fixed today, and this change tried.** Two gates
  refuse it independently: `MAX_README_CHARS = 30000` against a measured 29,989, so the
  LTX-2.5 matrix row could only land by deleting another architecture's row; and
  `check-doc-checkpoint.py:346-354`, which refuses any README change not accompanied by a
  LANDING SOURCE edit, per commit (#573). A two-family paragraph was written and MEASURED to
  fit with four characters to spare, then reverted unlanded when the second gate fired; it is
  preserved verbatim in the issue thread. The first blocker is the shared-file lock AGENTS.md
  § Records names; the second has no arm for a CORRECTION as against churn. Both want the
  spec-plus-red-first path, not a drive-by, so neither is touched here.

**One rejected audit finding, recorded because a rejection is a result.** These four were
reported alongside a claim that `Ltx2AudioPatchify` is ported-but-undriven. It is not. It is
called at `ltx2_video.cpp:2595`, unconditionally inside the phase loop of
`Ltx2VideoEngine::Generate`, on every LTX-2.5 render. Its second call site, inside
`Ltx2CreateAudioLatentState` (`ltx2_conditioning.cpp:484`), is the undriven one — which is
how a true statement about one call site became a false statement about a symbol.

## Now

Updated 2026-08-17. **The `## Now` this replaces was three days and roughly seventeen landed
rows stale**, and it is worth saying what it claimed, because the shape recurs: it reported
L1-L6 merged with "L7 and L8 ... reviewed FAIL on five MEDIUM findings; the repair is in
flight". L7 and L8 landed at `cefacd2d0` on 2026-08-13, and `git log --grep LTX25` shows what
followed.

L1-L11 are on `main`. Landed since that paragraph was written, each on its own row and PR:
prompt-side AdaLN (`65e79eee5`), the trained keyframe absolute-position bias (`98f8e046d`,
#658), the temporal x2 upsampler (`2e9d95e74`, gated and undriven), tiled and streaming Conv
VAE decode (`44b14cceb`), the dead-arm retirement (`0785cfc4d`), image conditioning and the
VAE encoder load path (`c629b5d0f`), the device-seam sibling (`d415c931d`), the staged-view
UAF repair (`4880c5715`, #904), audio-to-video (`c2019b0e3`), token-append (`c7cb59fbb`), the
resolution envelope (`e5351776c`, #919), generated keyframe slots (`71b401b15`), IC-LoRA
fusion (`885c96fe6`, #923), retake (`3ce1cf7c7`), the DFR base (`332aed738`, #986), the
decode dtype (`d1b0ea3a8`, #1008), decode threading (`ec0e410b5`, #1009) and text-to-audio
(`0b0b8900f`, #1005).

**The DiT forward runs on the GB10 GPU**, and that claim is unchanged. Independently verified
by the runtime provider announcer, not by reading: all eight dispatched ops resolve
`vt-native` on device 1 with `registered=1`, and ZERO reference-tier hits across eleven logs.
The shipped 21.00B FP8 DiT (6124 tensors) staged and produced one finite forward, reproduced
by the reviewer to the digit (absmax video 0.300781 / audio 2.14062).

**What still qualifies it.** The FP8 DiT carries NO `__metadata__`, so a run without
`--dit-config` takes DEFAULTS (`av_ca_timestep_scale_multiplier = 1`,
`double_precision_rope = false`) against LTX-2.5's declared 1000 / float64. The first-party
NVFP4 DiT no longer fails to load — §4.1's swizzled, high-nibble-first reading landed as L9a
— so §4's superseded correction is history rather than current position.

**A 704x448/25f render completed with a verified stereo track**, on the NVFP4 transformer on
one GB10 at `0b0b8900f`: 25/25 distinct frame md5s, 0 near-uniform frames, 0/24 zero-motion
pairs, 48 kHz stereo (`ltx25-resolution-envelope.md:383,401-407`). **It was NOT prompted** —
it took the embeds path (`docs/USAGE.md`, the resolution section). A prompted render on real
weights is still OWED, and so is the binding-oracle correctness axis (§3) and the speed axis
(§0). None of the three is a failure and none is a pass.
