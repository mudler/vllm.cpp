// LTX-2.5 behind the GENERALIZED video seam — the second family registered with
// `vllm::multimodal::VideoEngine`, and the driving loop that turns the L2-L6
// bricks into frames + a waveform.
//
// Row: MODEL-DIFFUSION-LTX25. Spec: .agents/specs/ltx-2-5.md phase L7. Issue #435.
//
// ─── WHAT THIS TU IS ─────────────────────────────────────────────────────────
//
// Phases L2-L6 shipped a DiT forward, a text feature extractor, two VAEs, a
// vocoder, an upsampler, a duration head, and a pipeline COMPONENT library —
// schedules, noisers, steppers, guiders, patchifiers, recipes. Nothing drove
// them: an `Ltx2UnportedPipelineFeature::kVideoEngineWiring` refusal named the
// composition BY NAME and named this phase as its owner. This TU is that
// composition and nothing else. It adds no numerics; every line either resolves
// a parameter, moves a buffer, or calls a brick that already has a golden. (That
// enumerator was RETIRED in row LTX25-RETIRE-DEAD-ARMS once L7 landed in
// `cefacd2d0` — a refusal whose subject shipped is a false statement, not a
// record of debt. It is named here in the past tense on purpose.)
//
// ─── WHAT IT IS A PORT OF (file:line on BOTH sides) ──────────────────────────
// Upstream: Lightricks/LTX-2 @ fd4ded7, packages/ltx-pipelines/src/ltx_pipelines/
//   OURS                              <-  UPSTREAM
//   Ltx2VideoEngine::Generate         <-  distilled.py:186-300 (DistilledPipeline.__call__)
//   the per-phase stage call          <-  utils/blocks.py:500-582 (DiffusionStage.__call__)
//   the latent state build            <-  utils/helpers.py:428-447 (create_noised_state)
//                                         + ltx-core tools.py:139-184 / :246-280
//   the denoise loop                  <-  utils/samplers.py:39-79 (euler_denoising_loop)
//                                         + :26-36 (_step_state)
//   the X0 conversion                 <-  ltx-core model/transformer/model.py:590-604
//                                         (X0Model.forward) + utils.py:38-50 (to_denoised)
//   the per-step Modality build       <-  utils/helpers.py:466-503
//                                         (modality_from_latent_state, timesteps_from_mask)
//   post_process_latent               <-  utils/helpers.py:462-464
//
// ─── WHAT THIS ENGINE REFUSES, AND WHY EACH WOULD RENDER ────────────────────
//
// Two of the three below are refusals that STAND. Item 2 is a refusal that was
// LIFTED, and it is kept in this list rather than deleted because the interesting
// thing about it is its history: it is the one whose stated reason went stale
// twice, which is this campaign's recurring defect and worth leaving legible.
//
// 1. `device = 1` (CUDA) WITHOUT A CUDA BACKEND. Phase L7 refused every non-zero
//    device outright: L2's forward was f32-only by declaration and L6's
//    `Ltx2StreamDitToDevice` stages bf16 and refuses to widen, so no combination
//    put the DiT on a GPU. **Phase L8 closed that** — `Ltx2DitForwardDevice`
//    (ltx2_device.h) is the same graph with every activation in device memory and
//    the stream in the checkpoint's own bf16 — so a CUDA handle now denotes a
//    CUDA forward and the load succeeds.
//
//    What is still refused is the SUBSTITUTION. If the CUDA backend is not
//    registered in this build, the load is refused BY NAME rather than served the
//    CPU forward behind a CUDA-looking handle, because that substitution is what
//    would make every later timing and every "it ran on the GPU" claim false.
//
// 2. A PROMPT — NO LONGER REFUSED. Phase L13 closed the last hop, and what
//    follows is the record of what each phase actually contributed, because
//    this refusal went stale TWICE before it went away and a reader deserves to
//    be able to re-check the reason rather than trust it.
//
//    L10 made the tower RUN: the embedded tokenizer reaches a prompt string
//    (`Ltx2TokenizeGemmaPrompt`, token-exact against HuggingFace on the shipped
//    262144-entry vocab), the torchao-NVFP4 Gemma-4 tower materializes onto
//    `Gemma4Weights` (`Ltx2LoadGemmaTowerFromSafetensors`), it produces all 49
//    hidden states within the oracle's own bf16 noise floor, and the aggregation
//    and both caption projections turn those into the 4096-wide video and
//    2048-wide audio conditioning streams (`Ltx2EncodePromptToConditioning`).
//
//    L9c put the `Embeddings1DConnector` on the render path with the
//    checkpoint's OWN weights (`Ltx2LoadConnectorWeights`), which is where
//    upstream puts it too: `EmbeddingsProcessor` runs an 8-layer 1-D transformer
//    over the caption projections before the DiT sees them
//    (embeddings_processor.py:70-117), and that module ships INSIDE THE DiT FILE
//    as `video_embeddings_connector.*` / `audio_embeddings_connector.*`.
//
//    The two landed on separate branches and could not see each other. L10's
//    refusal said the connector's weights were "still among the modules
//    `Ltx2LoadDitFromSafetensors` refuses", and by the time the branches met
//    that was FALSE — `ltx2_loader.cpp:417` already recorded them as loaded
//    elsewhere. So `has_encoder()` is now TRUE when `encoder_path` is supplied,
//    and a request's own `prompt` is tokenized, encoded, projected, run through
//    the connector and handed to cross-attention, per request.
//
//    WHAT THIS COSTS, stated rather than discovered: the tower is ~24 GB of host
//    bf16 at the shipped 12B and it stays RESIDENT, because a prompt arrives per
//    request. The connector weights do NOT stay resident — see the Impl comment
//    in the .cpp; they are ~8 GB of f32 and are re-read from the still-present
//    DiT file inside one scope per request, which keeps the steady state at the
//    tower rather than at tower + connector.
//
//    WHAT IS STILL OWED, and it is a real gap rather than a formality: the
//    shipped `vonkaiser` text encoder carries NO `__metadata__` at all, so the
//    Gemma config is an INPUT. It comes from the checkpoint's own
//    `__metadata__["gemma_config"]` when there is one and from the
//    `encoder_config_path` extra when there is not, and an encoder with neither
//    is REFUSED rather than given a default — the same polarity `dit_config_path`
//    already has, and for the same reason: a wrong Gemma config resolves a
//    DIFFERENT MODEL out of a byte-identical tensor set.
//
// 3. Any pipeline kind / model version the recipe table does not carry.
//    `ResolveLtx2PipelineRecipe` already throws rather than defaulting
//    (ltx2_pipeline.h:543-562); this engine passes the checkpoint's OWN
//    `model_version` through to it rather than assuming 2.5, because a checkpoint
//    of another generation resolved onto 2.5's sigmas renders confidently and
//    wrongly.
//
// ─── SCALE, STATED PLAINLY ───────────────────────────────────────────────────
//
// The f32 CPU forward is the parity forward, not a production one: at the shipped
// 21.00B geometry its weights alone are ~76 GB. The bf16 DEVICE forward (L8) is
// the production residency — ~42 GB staged tensor-by-tensor — and it is what
// `device = 1` runs. Neither changes the other number this row owes: a single
// denoise step over a 512x768x121 latent is ~2.6e14 FLOPs, and there is no
// production-configuration oracle to divide by, so no speed figure is claimed
// anywhere in this family (spec §0).
#pragma once

#include <memory>
#include <string>

#include "vllm/multimodal/video_engine.h"

namespace vllm {
struct Ltx2DitParams;
}  // namespace vllm

namespace vllm::multimodal {

// The stable registry name this family is reached under
// (VideoModelParams::family / vllm_video_model_params.family). It is the string
// `.agents/specs/ltx-2-5.md` and the L1 registry refusal test already print.
inline constexpr char kLtx2VideoFamily[] = "ltx-2.5";

// ── the family-specific LOAD extras (VideoModelParams::extras) ──────────────
// Every one of these is a knob upstream reads from somewhere this seam has no
// field for. An extra this family does not define is REFUSED, never ignored.

// The audio stream's prompt-embeds file, the twin of the seam's
// `prompt_embeds_path` (which carries the VIDEO stream). LTX-2.5 conditions two
// streams at two different widths — 4096 and 2048 — and one file cannot hold
// both, so the audio half rides here. Rows of `audio_cross_attention_dim`,
// little-endian f32, and the two files must agree on their ROW COUNT because
// upstream's two encodings come from one tokenization.
inline constexpr char kLtx2AudioPromptEmbedsExtra[] = "audio_prompt_embeds_path";

// `resolve_ltx_pipeline_recipe`'s first key (ltx2_recipes.py:161-175). Defaults
// to "distilled_two_stage", which is what the shipped
// `ltx-2.5-22b-distilled-transformer` is: the file NAMES itself distilled and
// `DistilledPipeline` is the entry point that loads it.
inline constexpr char kLtx2PipelineKindExtra[] = "pipeline_kind";

// Overrides the `model_version` the DiT checkpoint declares in its own
// `__metadata__`. Present for a checkpoint that carries none; it never silently
// replaces one that does, and a mismatch between the two is reported.
inline constexpr char kLtx2ModelVersionExtra[] = "model_version";

// The DiT's `{"transformer": {...}}` configuration, as a JSON FILE, for a
// checkpoint whose `__metadata__` carries none.
//
// MEASURED 2026-08-12, and it is why this extra exists: of the two shipped
// LTX-2.5 DiTs only the first-party NVFP4 one carries `__metadata__` at all.
// `vonkaiser/LTX-2.5-FP8-NVFP4`'s FP8 DiT — the copy every phase before L6 gated
// against and the one L8 ran on the GPU — has NO `__metadata__` key whatsoever.
// Without a config the geometry still resolves from SHAPES, but the values no
// shape encodes fall back to the parser's defaults: `double_precision_rope =
// false` and `av_ca_timestep_scale_multiplier = 1`, against LTX-2.5's declared
// `float64` and `1000`. Both move every RoPE angle and every audio<->video
// modulation, so a silent default is a DIFFERENT MODEL rendering confidently.
//
// So a DiT that declares no config is REFUSED unless this extra names one. The
// file holds the same object the shipped checkpoints put in
// `__metadata__["config"]` — `{"transformer": {...}}` — and it is adopted through
// the IDENTICAL weight-contract check the declared path uses, so a config
// belonging to another checkpoint is refused rather than bound.
inline constexpr char kLtx2DitConfigPathExtra[] = "dit_config_path";

// Proceed past the module families this port does not carry —
// `keyframes_abs_pos_embedding` (ltx2_loader.h). "1" opts in; anything else
// leaves the loader's refusal in place. The shipped DiTs carry it, so this is the
// flag that says "gate the ported subset knowingly".
//
// IT MUST NEVER DISABLE A PORTED FEATURE, and until 2026-08-13 it did:
// `prompt_adaln_single` / `audio_prompt_adaln_single` were on this list, and
// setting the extra reached three loader assignments that cleared
// `use_prompt_adaln_single`, so every real render dropped the timestep half of
// the prompt K/V modulation — finite, same-shaped, and invisible to every gate.
// Those families are ported now (.agents/specs/ltx25-prompt-adaln.md, issue
// #644), the loader asserts the flag against the file rather than clearing it,
// and this extra is scoped to the one module nothing applies. Adding a family
// here only ever means "the forward genuinely has no code for this".
//
// The two `*_embeddings_connector` families are NOT in that set and this extra
// has nothing to do with them: they are outside the DiT contract by design and
// are materialized by `Ltx2LoadConnectorWeights`. An earlier revision of this
// comment listed them here, which is the same stale claim `ltx2_loader.h`
// carried and phase L10's refusal was built on.
inline constexpr char kLtx2AllowUnportedExtra[] = "allow_unported_modules";

// Run only the phases up to and including this index of the resolved recipe
// (0-based). Absent runs every phase. This exists because the two-stage recipe's
// second phase needs the latent spatial upsampler, and a run without one must
// say so rather than skipping the phase silently.
inline constexpr char kLtx2MaxPhaseExtra[] = "max_phase";

// An IC-LoRA adapter to FUSE into the DiT, and its strength
// (ltx-core loader/primitives.py:160-167 `LoraPathStrengthAndSDOps`, fused by
// loader/fuse_loras.py:119-150; the CLI pair is ltx-pipelines utils/args.py:600-611).
//
// LOAD extras rather than generation fields, because upstream takes the LoRAs as
// a `DiffusionStage.from_checkpoint` CONSTRUCTOR argument (ic_lora.py:104-114)
// and fuses them into the weights, so the adapter is a property of the loaded
// model and cannot vary per request. A per-generation field would promise
// something the mechanism cannot do.
//
// `lora_strength` absent is 1.0, upstream's DEFAULT_LORA_STRENGTH. Supplying a
// strength without a path refuses, because a strength alone is a request that
// silently did nothing.
inline constexpr char kLtx2LoraPathExtra[] = "lora_path";
inline constexpr char kLtx2LoraStrengthExtra[] = "lora_strength";

// How many of the supplied prompt-embeds rows are REAL tokens; the rest are
// padding. Absent means every row is real.
//
// WHY A SEAM WITH NO TOKENIZER NEEDS THIS. The embeddings connector substitutes
// its `learnable_registers` table at PADDED positions
// (embeddings_connector.py:139-152), so the padding is not inert — it is what
// decides which of the connector's inputs are learned constants rather than
// caption features. Upstream always knows this, because the tokenizer produced
// the mask. This seam takes prompt embeds from a FILE, which carries no mask, so
// without this extra the padded tail would be conditioned on as if it were text
// and every register would go unused. Recorded as a knob rather than assumed,
// and it is the field the Gemma-4 tower will supply when it lands.
inline constexpr char kLtx2PromptValidRowsExtra[] = "prompt_embeds_valid_rows";

// The Gemma-4 config for the tower `encoder_path` names, as a JSON FILE.
//
// MEASURED, and it is why this extra exists rather than being a convenience:
// the only shipped LTX-2.5 text encoder, `vonkaiser`'s
// `gemma4-12b-with-proj-nvfp4-torchao.safetensors`, has NO `__metadata__` block,
// so upstream's own `GemmaAssets.from_single_file` raises on it before reading a
// tensor (gemma_assets.py:110-114). The official bf16 encoder DOES carry one,
// under `__metadata__["gemma_config"]`, and that is preferred when present.
//
// What a default would get wrong is not a detail. `layer_types` decides which
// layers are full vs sliding and the two have DIFFERENT geometry;
// `global_head_dim` is 512 against `head_dim` 256; `num_global_key_value_heads`
// is ONE against 8; `attention_k_eq_v` is true, so the full layers ship no
// `v_proj` at all. Each of those moves every hidden state while leaving the
// tensor set byte-identical — a wrong config resolves a DIFFERENT MODEL out of
// the same file and nothing downstream can tell. So an encoder with neither
// source is refused, exactly as `dit_config_path`'s case is.
//
// Supplying BOTH a declaring checkpoint and this extra is refused rather than
// resolved in either direction, for the same reason.
inline constexpr char kLtx2EncoderConfigPathExtra[] = "encoder_config_path";

// ── the PER-GENERATION extra (VideoGenParams::extras) ───────────────────────

// The H.264 CRF the image conditioning is re-compressed at, `ImageConditioner`'s
// `resolve_crf` (ltx-pipelines/utils/blocks.py:977-983). Row LTX25-IMAGE-COND,
// issue #644.
//
// ABSENT MEANS "WHAT THE MODEL WAS TRAINED WITH", which for an LTX-2.5
// checkpoint is **18** — `detect_params` maps a version at or above `(2, 4)`
// onto `LTX_2_4_PARAMS` and its `LTX_2_4_IMAGE_CRF` (utils/constants.py:37,
// 124, 130-133). And that round trip is NOT ported: it needs libx264
// (media_io/decode.py:430-434 -> encode_single_frame:386-400) and no codec is
// vendored here. So the DEFAULT REFUSES, by name.
//
// `image_crf=0` is the supported value and is served. It is upstream-legal —
// `preprocess` short-circuits at `if crf == 0: return image` (decode.py:425-426)
// and an explicit 0 is documented as "skip re-compression entirely"
// (utils/args.py:58-59) — and it is OUT OF DISTRIBUTION, because the model was
// trained on images that had been through the codec. Both halves are said out
// loud rather than one of them: a caller has to ask for 0 knowingly, and gets a
// render conditioned on uncompressed pixels rather than a refusal.
inline constexpr char kLtx2ImageCrfExtra[] = "image_crf";

// ── AUDIO-TO-VIDEO: the driving waveform. Row LTX25-A2V-AUDIO-INPUT (#922) ──
//
// Upstream's `--audio-path` (`a2vid_two_stage.py:312-317`, required there
// because that CLI drives the A2V pipeline and nothing else). Here it is
// per-generation and OPTIONAL: supplying it selects the audio-conditioned path
// on a checkpoint that carries audio VAE encoder weights, and leaving it out is
// the ordinary text-to-video render.
//
// A 16-bit PCM RIFF/WAVE file whose channel count matches the checkpoint's audio
// VAE encoder `in_channels` and whose sample rate matches its mel front-end.
// Neither is converted: upstream resamples with an arbitrary-ratio polyphase
// kaiser resampler (`ops.py:40`) this project has not ported, and it feeds the
// file's own channel count straight into a conv that declares 2
// (`model_configurator.py:172`). Both mismatches are refused with both numbers
// in the message, because a resampled-wrong or upmixed-wrong take conditions the
// render on a waveform the caller never supplied and still finishes.
//
// The audio is held FROZEN through every denoise phase — upstream's
// `ModalitySpec(frozen=True, noise_scale=0.0)` at `a2vid_two_stage.py:251-256`
// and `:291-296` — so the video is generated around it and the encoded latent
// is returned unchanged. The rendered soundtrack is the caller's own file, not
// a VAE round trip, which is upstream's deliberate choice at `:301-303`.
inline constexpr char kLtx2AudioPathExtra[] = "audio_path";

// Seconds into the file to start reading (`--audio-start-time`,
// `a2vid_two_stage.py:318-323`, default 0.0).
inline constexpr char kLtx2AudioStartTimeExtra[] = "audio_start_time";

// Seconds of audio to read (`--audio-max-duration`,
// `a2vid_two_stage.py:324-329`). ABSENT MEANS the video's own duration,
// `num_frames / frame_rate` — and note that the default is applied by upstream's
// CLI (`:369-371`), not by the pipeline, whose own default is `None` (`:157`).
// Mirrored at the same layer, so a caller who omits it gets exactly as much
// audio as the clip is long.
inline constexpr char kLtx2AudioMaxDurationExtra[] = "audio_max_duration";

// GENERATED keyframe slots — the OTHER upstream feature called "keyframe".
// Row LTX25-GENERATED-KEYFRAMES (#920) DEFINED this key and refused it; row
// LTX25-DFR-PIPELINE (#986) SERVES it.
//
// Not to be confused with the SUPPLIED keyframe arm. The two differ by one
// argument, and it is the argument that decides whether the trained marker is
// applied at all:
//
//   supplied  `VideoConditionByKeyframeIndex` — the caller hands in an image
//             for a frame index; appended `marked=False` (keyframe_cond.py:84-86)
//   GENERATED `VideoGeneratedKeyframeSlots`   — the MODEL generates extra frames
//             at interior positions; appended `marked=True` (keyframe_slots.py:121)
//
// `extend_keyframes_mask` (conditioning/mask_utils.py:76-107) documents the
// polarity, and `keyframe_slots.py:121` is upstream's ONLY call site that passes
// True. So this is the only user-facing feature that puts
// `keyframes_abs_pos_embedding` on a token other than the target's own first
// latent frame — which `Ltx2FirstFrameKeyframesMask` already marks on every
// render, unconditionally, mirroring `tools.py:184-196`.
//
// THIS KEY IS NOW SERVED. #920 defined it and refused it, naming the READBACK as
// its one blocker: `GeneratedKeyframeLayout`, the extraction into
// `generated_keyframes` before the trim, and a standalone single-frame decode.
// #986 landed the first two, which is what DFR needs, and the refusal it
// replaced is retired rather than widened — a refusal for a served capability is
// worse than none. The third piece is still owed and is a SEPARATE surface: it
// would return slot PIXELS to a caller, and nothing here does that, because
// upstream's own consumers keep the slots in latent space.
//
// Spelled and typed as upstream's CLI spells it: `--num-generated-keyframes`,
// `type=int`, `default=0` (ltx-pipelines/utils/args.py:833-844). It is a
// per-CALL argument upstream, forwarded to the FIRST diffusion stage only, so it
// belongs on the per-generation surface rather than on load. `0` is upstream's
// own default and means OFF (`has_generated_keyframes`, utils/helpers.py:384-391)
// — an explicit 0 must therefore RENDER, not refuse.
//
// The positions are `evenly_spaced_keyframe_positions` (utils/helpers.py:370-381):
// `linspace(0, num_frames - 1, n + 2)` rounded, with the ENDPOINTS DROPPED. A
// negative count and a target shorter than `n + 2` are upstream's own two
// refusals and are mirrored.
//
// ON A `dfr` PIPELINE THIS KEY IS REFUSED, and that is not an omission.
// `DFRPipeline` does not take it: its slot positions come from `resolve_canvas`
// (dfr_pipeline.py:314), which puts one keyframe on every x8-border segment
// boundary, and its CLI exposes no `--num-generated-keyframes` at all. Accepting
// both would let a caller silently override the canvas the whole pipeline is
// built around.
inline constexpr char kLtx2GeneratedKeyframesExtra[] = "num_generated_keyframes";

// DFR's temporal x2/x4 refinement rounds — `temporal_upsample_rounds`,
// `type=int`, `choices=(0, 1, 2)`, `default=0`
// (ltx-pipelines/dfr_pipeline.py:277, :584-590). Row LTX25-DFR-PIPELINE (#986).
//
// DEFINED, and REFUSED above 0. `0` is upstream's default and is the served
// path; a positive count is refused by name, and what it names is the rounds
// LOOP rather than the upsampler. The operator itself is ported and gated
// (row LTX25-TEMPORAL-UPSAMPLER, `.agents/specs/ltx25-temporal-upsampler.md`) —
// `PixelShuffle1d`, the first-frame drop, and the loader arm that reads
// `temporal_upsample` off the checkpoint config all exist and pass.
//
// The key is defined here rather than left to the generic "unknown extra"
// message for the reason #611 established: that message asserts the family does
// not define the key, which is false, and sends the reader looking for a typo
// instead of for the unported loop.
inline constexpr char kLtx2TemporalRoundsExtra[] = "temporal_upsample_rounds";

// ── RETAKE: regenerate a time window of an existing clip. Row LTX25-RETAKE ──
// (#924), spec .agents/specs/ltx25-retake.md.
//
// `RetakePipeline` (retake.py:53, `__call__` at :151) keeps the source clip
// outside `[start_time, end_time)` and regenerates what is inside it from the
// prompt. The SOURCE is `vllm_video_params::ref_video` — a DIRECTORY of
// `frame_%06d.ppm`, which is what that ABI field has always meant
// (include/vllm.h:912) and which the LTX-2.5 engine used to only test for
// emptiness in order to refuse.
//
// A frame DIRECTORY rather than a container, and that is upstream's second
// ingestion arm rather than a local substitute: upstream reads containers with
// PyAV (media_io/decode.py:226) and carries a folder arm for the case where
// there is none. Three consequences follow from upstream's own lines and are
// mirrored, not invented — the frame rate must be supplied because there is no
// container to read it from (decode.py:213-215), the folder has no audio stream
// (utils/helpers.py:261-262), and therefore BOTH of retake's audio predicates
// are false whatever `regenerate_audio` says (retake.py:279,282).
//
// Seconds, inclusive (retake.py:155, noise_mask_cond.py:19). Supplying it
// selects the retake path; leaving it out is the ordinary render.
inline constexpr char kLtx2RetakeStartTimeExtra[] = "retake_start_time";

// Seconds, EXCLUSIVE (retake.py:156, noise_mask_cond.py:20). Required alongside
// the start: `start_time >= end_time` is upstream's first refusal
// (retake.py:211-212) and an absent end would have to be defaulted to something
// upstream never defaults.
inline constexpr char kLtx2RetakeEndTimeExtra[] = "retake_end_time";

// The source folder's frame rate (`--frame-rate`, utils/args.py:865-873).
// REQUIRED for a folder and refused for a container by upstream's own parser
// help; here the container arm does not exist, so it is simply required. The
// whole temporal mask is `pixel_bounds / fps` (noise_mask_cond.py:35), so a
// wrong one regenerates the wrong seconds and still renders.
inline constexpr char kLtx2RetakeFrameRateExtra[] = "retake_frame_rate";

// `regenerate_video` (retake.py:164, default True) and `regenerate_audio`
// (:165, default True), spelled `0` / `1`. `regenerate_video=0` freezes the
// source video and regenerates nothing, which is upstream-legal and is what
// makes the four-way plan four-way rather than two-way.
inline constexpr char kLtx2RegenerateVideoExtra[] = "regenerate_video";
inline constexpr char kLtx2RegenerateAudioExtra[] = "regenerate_audio";

// ── TEXT-TO-AUDIO. Row LTX25-T2A-ONE-STAGE (#1005) ─────────────────────────
//
// These are read ONLY on a `pipeline_kind = t2a_one_stage` engine — that is a
// LOAD extra, so which pipeline runs is fixed before a request arrives. Supplied
// on any other pipeline they are REFUSED, because upstream's other entry points
// have no counterpart for them and a knob that silently does nothing is the
// defect this whole surface refuses by name elsewhere.
//
// `--negative-prompt` (ltx-pipelines utils/args.py:1083-1088). ABSENT MEANS the
// recipe's own default, which is upstream's `DEFAULT_NEGATIVE_PROMPT`
// (utils/constants.py:186) on the 2.4/2.5 rows.
//
// IT IS NOT COSMETIC ON THIS PIPELINE. T2A's CFG scale defaults to 7.0, so the
// negative conditioning is one of the two tensors the guidance delta is computed
// from (`(cfg_scale - 1) * (cond - uncond_text)`, guiders.py:262). An empty one
// is refused rather than substituted with zeros: a zero `uncond_text` turns the
// delta into `cfg_scale * cond`, which is a DIFFERENT render and not a missing
// one.
inline constexpr char kLtx2NegativePromptExtra[] = "negative_prompt";

// The audio guider, one CLI flag each (utils/args.py:1089-1119). ABSENT MEANS the
// params table's own value for the checkpoint's generation — 7.0 / 1.0 / 0.7 and
// block 28 on the 2.3-and-later lineage (utils/constants.py:58-66, :82-87).
//
// `audio_stg_blocks` is a COMMA-SEPARATED list, mirroring `nargs="*"`. An EMPTY
// value is upstream's empty list and means "perturb nothing" — which is refused
// alongside a non-zero STG scale rather than silently running a perturbed pass
// identical to the conditional one. Upstream's `blocks is None` ("every block",
// guidance/perturbations.py:19-33) has no CLI spelling and none is invented here.
//
// There is deliberately NO `modality_scale` knob: the CLI pins it to 1.0 for this
// pipeline and states the reason (t2a_one_stage.py:200-202), so exposing it would
// offer a fourth forward over a modality that does not exist.
inline constexpr char kLtx2AudioCfgScaleExtra[] = "audio_cfg_guidance_scale";
inline constexpr char kLtx2AudioStgScaleExtra[] = "audio_stg_guidance_scale";
inline constexpr char kLtx2AudioRescaleScaleExtra[] = "audio_rescale_scale";
inline constexpr char kLtx2AudioSkipStepExtra[] = "audio_skip_step";
inline constexpr char kLtx2AudioStgBlocksExtra[] = "audio_stg_blocks";

// THE VIDEO GUIDER, row LTX25-GUIDED-VIDEO (#1092). The same row of flags on the
// other stream, from the same parser (`default_1_stage_arg_parser`,
// utils/args.py:947-1066). ABSENT MEANS the params table's own value: 3.0 / 1.0 /
// 0.7 / 3.0 and block 28 on the 2.3-and-later lineage
// (utils/constants.py:40-88).
//
// THE MODALITY KNOBS EXIST HERE AND NOT ON THE T2A ROW ABOVE, and the asymmetry
// is upstream's rather than an oversight on either side. Text-to-audio has no
// video stream, so `t2a_one_stage.py:200-202` pins `modality_scale = 1.0` and
// exposes no flag. A joint render has both streams and the parser exposes
// `--a2v-guidance-scale` and `--v2a-guidance-scale`, which are the video and
// audio guiders' `modality_scale` respectively (utils/args.py:987-996 and its
// audio counterpart). Reaching either turns on a fourth DiT forward per step.
//
// EVERY ONE OF THESE IS REFUSED on a phase whose recipe sets
// `allow_guidance_override = false` — the distilled two-stage and retake
// recipes, whose guidance is distilled INTO the weights. Honouring an override
// there would sample a trajectory the weights were never trained for, which is
// the same argument `fixed_num_inference_steps` already makes for the schedule.
inline constexpr char kLtx2VideoCfgScaleExtra[] = "video_cfg_guidance_scale";
inline constexpr char kLtx2VideoStgScaleExtra[] = "video_stg_guidance_scale";
inline constexpr char kLtx2VideoRescaleScaleExtra[] = "video_rescale_scale";
inline constexpr char kLtx2VideoSkipStepExtra[] = "video_skip_step";
inline constexpr char kLtx2VideoStgBlocksExtra[] = "video_stg_blocks";
inline constexpr char kLtx2A2vGuidanceScaleExtra[] = "a2v_guidance_scale";
inline constexpr char kLtx2V2aGuidanceScaleExtra[] = "v2a_guidance_scale";

// THE NEGATIVE CONDITIONING FOR AN ENGINE WITH NO TEXT TOWER, and a LOCAL
// ADAPTATION recorded as one.
//
// Upstream has no embeds surface at all: every pipeline encodes
// `[prompt, negative_prompt]` in ONE `PromptEncoder` call
// (ti2vid_one_stage.py:166-174) and takes `.video_encoding` / `.audio_encoding`
// from each half. `prompt_embeds_path` and the `audio_prompt_embeds_path` extra
// are this port's own affordance for running the DiT without a 12B tower; these
// two are the SAME affordance applied to the second of upstream's two
// encodings, not a new concept.
//
// They are supplied together with each other, and only alongside the positive
// pair. Without them and without a tower, a guider that asks for the
// unconditional pass is REFUSED BY NAME rather than served the positive context
// twice — which would make `(cfg_scale - 1) * (cond - uncond)` identically zero
// and produce an unguided render wearing a guided render's configuration.
inline constexpr char kLtx2NegativePromptEmbedsExtra[] = "negative_prompt_embeds_path";
inline constexpr char kLtx2NegativeAudioPromptEmbedsExtra[] =
    "negative_audio_prompt_embeds_path";

// WHAT THE LAST `Generate()` ACTUALLY HANDED THE DiT's CROSS-ATTENTION.
//
// Every field is read off the exact f32 buffers `Ltx2ModalityInput::context`
// pointed at, after the connector and immediately before the denoise loop — not
// re-derived from the inputs that produced them.
//
// WHY THIS EXISTS AS A SURFACE. Conditioning is the one thing about a render
// that the render cannot be inspected for. This project has already been burned
// on that exact point twice: L9c's reviewer found that the difference between a
// scene and a colour field was INVISIBLE to the frame analyzer (neighbour
// |dx|/sd 0.093 vs 0.033, block-mean ratios nearly identical) and took contact
// sheets to tell apart; and the campaign's standing lesson is that a golden
// reduced to `isfinite` once hid a 23842x error. A digest over the bytes that
// were actually fed is an instrument that cannot have that blind spot: it is a
// function OF those bytes, so any change to any element changes it, and it
// cannot be satisfied by a plausible-looking wrong tensor.
//
// It is not a quality claim and cannot become one. It answers "did this render
// depend on this prompt, through these weights" and nothing else. A server also
// has a use for it: "which conditioning produced this clip" is otherwise
// unanswerable after the fact.
//
// IT IS A WITNESS, NOT A GATE — and the difference is MEASURED, not argued. A
// digest detects CHANGE; it does not pin VALUES, so nothing at this level says
// the values are the ones upstream would produce. Two mutations applied to the
// composition below, each alone:
//
//   * video conditioning scaled by 1.5 AFTER the connector, and
//   * the conditioning rows REVERSED, putting every caption row on the wrong
//     token,
//
// and BOTH passed `test_ltx2_video` with exit 0 — at 30 cases / 499 assertions
// when the pair was last re-run. The digest moved, as it must — but no assertion
// says WHICH value it should have moved to.
//
// THE COUNT IS DELIBERATELY NOT RESTATED AS A CURRENT FIGURE, and the reason is
// the history: a reviewer first measured the pair at `43aa58377`, where the
// suite stood at 485 assertions; the numbers were carried forward unchanged
// while the suite grew, so the comment named a count no run of it could produce.
// It was re-measured at 499 (CPU Release, mutant recompiled and relinked each
// leg, tree restored byte-for-byte and re-verified green between legs) — and
// then row LTX25-IMAGE-COND (#644) added the image-conditioning cases and the
// suite moved to 32 / 550, which is exactly how the previous number went stale
// the first time. A count in a header is a MEASUREMENT OF ANOTHER FILE stored
// here, which AGENTS.md §Records names as the thing that couples every PR to
// lines it does not own. What survives is the finding — the mutations passed —
// and the SHA-dated measurement above it; `ctest` is the authority on the count.
//
// THE VALUE ORACLE THE COMPOSITION IS OWED. The per-brick oracles are real and
// strong: the Gemma-4 tower against a running `transformers` at a measured bf16
// floor, `Ltx2ConnectorForward` on five arms against executed upstream, and the
// feature extractor and both caption projections against executed upstream. The
// two JOINS between them have none: `Ltx2ConnectorCreateEmbeddings`
// (ltx2_connector.h) and the `Generate` composition that chains it onto
// `Ltx2TextEncoderConditioning`. Both mutations above live in exactly that gap.
//
// The closure is specified rather than left as a wish, because the path is
// already built. `scripts/gen-ltx2-pipeline-goldens.py` imports and EXECUTES
// upstream `text_encoders/gemma/embeddings_connector.py` under a pinned SHA
// (section 10), and the composition's upstream counterpart is one function in
// the same package: `EmbeddingsProcessor.process_hidden_states`
// (embeddings_processor.py:97-117), which is feature extractor -> additive mask
// -> `create_embeddings` -> the two connectors — precisely this chain. A section
// that executes it end-to-end at the reduced dims the script already uses would
// give both joins a real numeric oracle, WITHOUT the "gate through our own
// helper" trap that makes a max|diff| of 0 prove only that two arms agree. The
// script reproduces its current output byte-for-byte (md5 53e2a6ab…9eb4,
// verified 2026-08-13), so the section can be added without disturbing anything
// already gated. Until that lands, this trace is a change detector and the
// composition's VALUES rest on the per-brick oracles either side of it.
struct Ltx2ConditioningTrace {
  // True when the text tower encoded the request's own prompt; false when the
  // conditioning came from `prompt_embeds_path`.
  bool from_prompt = false;
  std::string prompt;  // the exact string that was tokenized ("" for embeds)
  int64_t tokens = 0;  // context rows the DiT cross-attends over
  int64_t video_width = 0, audio_width = 0;
  // FNV-1a over the raw little-endian f32 bytes of each stream.
  uint64_t video_digest = 0, audio_digest = 0;
  // max|x| per stream. A conditioning tensor that collapsed to zeros would give
  // two prompts the SAME digest and RED any dependence check, but it would do so
  // for the wrong reason; this says which happened.
  double video_absmax = 0.0, audio_absmax = 0.0;
  // ── the IMAGE conditioning (row LTX25-IMAGE-COND, issue #644) ────────────
  //
  // Zero everywhere when the request carried no image. `image_tokens` is how
  // many of the video stream's tokens the encoded image REPLACED in the clean
  // latent, and `image_digest` is FNV-1a over THOSE TOKENS' raw f32 bytes — the
  // same instrument, and with the same limits, as the two prompt digests above:
  // it detects CHANGE, it does not pin VALUES.
  //
  // OVER THE TOKENS, NOT OVER THE ENCODER'S OUTPUT, and that choice is load
  // bearing: a digest of the encoder's output answers "was an image encoded",
  // which stays true of a build that encodes one and then never places it — an
  // unconditioned render with a perfectly healthy trace.
  //
  // IT IS ALSO THE ONLY WAY TO ASK WHETHER THE ENCODER WEIGHTS WERE READ. "The
  // conditioning loaded" and "the conditioning was used" are different claims,
  // and a render cannot be inspected for either. Perturbing one encoder tensor
  // and watching this digest move is what separates them, which is exactly the
  // check `test_ltx2_video` runs.
  int64_t image_tokens = 0;
  uint64_t image_digest = 0;
  double image_absmax = 0.0;
  int64_t image_crf = 0;      // the CRF this render actually preprocessed at
  double image_strength = 0.0;  // `ImageConditioningInput.strength` (args.py:64)

  // ── the token-APPEND seam (row LTX25-TOKEN-APPEND, issue #930) ───────────
  //
  // TWO token counts, because after this row they are no longer the same number
  // and the difference between them IS the row.
  //
  // `video_tokens` is the length of the sequence the DiT forward actually ran
  // over on the LAST phase — the target grid plus whatever an appending
  // conditioning item added (keyframe_cond.py:79-82). `schedule_tokens` is the
  // count the sigma schedule read, which upstream fixes at the TARGET: its shift
  // comes from `math.prod(latent.shape[2:])` (schedulers.py:32), the
  // UNPATCHIFIED target latent, and the pipelines compute sigmas before any state
  // exists (ti2vid_one_stage.py:207, distilled.py:200-201). So a render that
  // appends must show `video_tokens > schedule_tokens`, and a build that let the
  // append re-shift the schedule shows them equal.
  //
  // Both are written INSIDE the phase loop, like `image_tokens` and unlike every
  // field above them. That distinction is the whole reason they exist: the rest
  // of this trace is filled before denoise and therefore cannot observe anything
  // the loop does, so a witness built on those fields finds every arm identical
  // and reads as a weak effect rather than as a blind instrument.
  //
  // `schedule_tokens` stays 0 on a recipe that carries its own distilled sigmas
  // (`Ltx2PhaseRecipe::sigmas` non-empty), because on that path no schedule is
  // computed and there is nothing to report. Zero here means "not measured", not
  // "zero tokens".
  int64_t video_tokens = 0;
  int64_t schedule_tokens = 0;

  // ── GENERATED KEYFRAME SLOTS AND THE DFR CANVAS (#986) ────────────────────
  //
  // Zero everywhere when the request asked for no slots, which is every
  // `one_stage` and `distilled_two_stage` render that omits
  // `num_generated_keyframes`.
  //
  // These exist for the reason `image_digest` does, and the reason is sharper
  // here than anywhere else on this struct: a generated keyframe slot is
  // INVISIBLE to the rendered clip. It appends tokens that are trimmed away
  // before unpatchify, so a build that placed no slots, or placed them and threw
  // them away, or placed them UNMARKED, returns a video of exactly the same
  // shape, the same frame count and the same file size. There is no pixel to
  // compare and no digest of the output that moves.
  //
  // `slot_positions` is the resolved PIXEL frame of each slot — from
  // `resolve_canvas` on a `dfr` pipeline (dfr_layout.py:60-81) and from
  // `evenly_spaced_keyframe_positions` on the others (utils/helpers.py:370-381).
  // `slot_tokens_extracted` is how many token-frames `clear_conditioning`
  // actually read back BEFORE the trim (ltx_core/tools.py:97, :203-230), so a
  // build that placed slots and then dropped them reports a positive
  // `slot_positions` beside a zero here.
  //
  // `slot_marked_tokens` is the one that cannot be inferred from the others. It
  // counts the slot tokens that carry the keyframe marker, and the marker is the
  // ONLY thing that distinguishes a generated slot from an ordinary append
  // (`extend_keyframes_mask(..., marked=True)`, keyframe_slots.py:121 —
  // upstream's single marked call site). An unmarked slot costs the same tokens
  // and renders the same clip while silently omitting the trained embedding.
  //
  // `canvas_frames` is the PADDED canvas a `dfr` request denoised, before the
  // trim back to the caller's own count (dfr_pipeline.py:531-540), and
  // `canvas_segment` is the keyframe segment length `choose_segment_length`
  // picked. Zero on every non-DFR pipeline.
  std::vector<int64_t> slot_positions;
  int64_t slot_tokens_extracted = 0;
  int64_t slot_marked_tokens = 0;
  int64_t canvas_frames = 0;
  int64_t canvas_segment = 0;

  // ── AUDIO-TO-VIDEO: the supplied take, as the DiT received it (#922) ───────
  //
  // Zero and false everywhere when the request carried no `audio_path`.
  //
  // These describe the AUDIO STREAM STATE of the last phase that ran, read off
  // the same patchified buffer `Ltx2ModalityInput::latent` pointed at. They
  // exist for the reason `image_digest` does: "the audio latent has the right
  // token count" is satisfied completely by a path that encoded the file and
  // then conditioned on zeros, and no rendered clip can tell the difference.
  //
  // `audio_frozen` is the one that cannot be inferred from the others. Upstream
  // holds the audio at `frozen=True, noise_scale=0.0` through both stages
  // (a2vid_two_stage.py:251-256, :291-296), and a build that seeded the latent
  // correctly and then let the sampler denoise it produces a soundtrack drifting
  // away from the caller's file while every count and digest above still looks
  // populated.
  bool audio_conditioned = false;
  bool audio_frozen = false;
  int64_t audio_tokens = 0;
  uint64_t audio_latent_digest = 0;
  double audio_latent_absmax = 0.0;

  // The LARGEST scalar `Modality.sigma` the audio stream was handed across
  // EVERY step of EVERY phase — upstream's `Modality.sigma`, the second half of
  // `frozen` (utils/types.py:104-106), which is a separate DiT input from the
  // per-token timesteps and which the denoise mask cannot reach.
  //
  // ALL PHASES, and that is deliberate, so it is the ONE field in this block
  // that is not last-phase-only. Every sibling above reports the conditioning as
  // the final phase saw it, because that is the state the rendered clip came
  // from. This one is an assertion about a FREEZE, and a freeze that holds in
  // the last phase and broke in the first is not a freeze — a per-phase reset
  // would make exactly that build read as frozen. There is no reset in the loop
  // and none is wanted; the wider window is the stronger claim.
  //
  // A MAXIMUM rather than the last value, because the last step's schedule sigma
  // is 0 anyway: reporting that would read as frozen on every render. MEASURED:
  // with this field absent, a mutation leaving the audio at the schedule's sigma
  // — telling the DiT the caller's clean take is noisy — left this suite green.
  // The claim was made in a comment and observed by nothing.
  double audio_sigma_max = 0.0;

  // ── RETAKE (row LTX25-RETAKE, #924) ────────────────────────────────────────
  //
  // Observed for the same reason the audio freeze above is: the whole mechanism
  // is a denoise MASK, and a mask is invisible in a frame count, a token count
  // and a rendered clip. A build that read the source clip, encoded it, and then
  // denoised every token anyway produces a finished video of the right length
  // with the right soundtrack and nothing outside the window preserved.
  bool retake_conditioned = false;
  // Tokens the `TemporalRegionMask` left at 1.0 — the regenerated region — and
  // the total it chose from. BOTH are recorded, because a mask that is all ones
  // and a mask that is all zeros are each a plausible-looking failure and a
  // count on its own cannot tell either from a correct one.
  int64_t retake_masked_tokens = 0;
  int64_t retake_total_tokens = 0;
  // The video stream's scalar sigma, max over every step of every phase — the
  // SECOND half of `frozen` on the video side (utils/types.py:104-106), which
  // the denoise mask cannot reach. `regenerate_video=0` must drive this to 0.
  double video_sigma_max = 0.0;
  // The encoded source clip, before any denoising. `absmax` is the lower bound a
  // token- or shape-shaped check cannot make: a zeroed or constant latent has
  // the right size and the right token count.
  uint64_t retake_latent_digest = 0;
  double retake_latent_absmax = 0.0;

  // ── THE SAMPLER (row LTX25-RES2S-LOOP, #921) ──────────────────────────────
  //
  // TWO COUNTERS, BECAUSE THERE ARE TWO QUESTIONS AND ONE NUMBER CANNOT ANSWER
  // BOTH. A render's DiT work is `evaluations x forwards-per-evaluation`. The
  // sampler decides the first factor and the denoiser decides the second, and a
  // build can get either wrong while producing a clip of the same shape, frame
  // count, sample rate and file size.
  //
  // `dit_evaluations` is every DENOISER CALL this render made, across every
  // phase and every step, and it is the only thing that separates the res_2s
  // sampler from the first-order one. The two samplers differ in that one calls
  // the denoiser TWICE per step (samplers.py:301 and :380-386) plus once at the
  // terminal sigma (:437). Serving the HQ preset's 15 steps on the Euler loop
  // would make 15 calls where upstream makes 31, and no output check in this
  // tree could tell.
  //
  // The count is `2 * steps + 1` per res_2s phase when that phase's schedule
  // ends at 0 and `2 * steps` when it does not, against `steps` for the Euler
  // and ancestral arms — so a build that selected the wrong sampler reports a
  // number that is close to half, not a number that is wrong by one.
  //
  // Incremented at ONE site, inside the shared `Evaluate` lambda that every
  // sampler goes through, so no arm can make a call this misses. A second
  // increment beside the res_2s loop's own returned `evaluations` would let the
  // two drift; the engine asserts they agree instead.
  int64_t dit_evaluations = 0;
  // `dit_forwards` is every ACTUAL `Ltx2DitForward` this render ran, counted
  // inside the `Ltx2X0Model` lambda the guided denoiser drives. One evaluation
  // is one to four forwards — `cond`, `uncond`, `ptb`, `mod`
  // (denoisers.py:100-137) — so this is the factor `dit_evaluations` cannot see.
  //
  // IT EXISTS BECAUSE THE EVALUATION COUNT IS BLIND TO GUIDANCE. Upstream's HQ
  // stage 1 runs a `GuidedDenoiser` at cfg 3.0 / 7.0 with modality 3.0
  // (ti2vid_two_stages_hq.py:271-281, constants.py:99-114), which is three
  // forwards per evaluation. An arm that ran the res_2s sampler around a bare
  // unguided forward keeps `dit_evaluations` at exactly `2 * steps + 1`, renders
  // a plausible clip at cfg 1.0 where the preset was tuned at 3.0, and moves no
  // other number in this struct. This one drops from `3 * (2 * steps + 1)` to
  // `2 * steps + 1`, which is why it is asserted rather than described.
  int64_t dit_forwards = 0;
  // Steps on which the bong anchor refinement ran, i.e. on which
  // `bongmath and h < 0.5 and sigma > 0.03` held (samplers.py:357). Zero on
  // every non-res_2s pipeline. It is reported separately from the evaluation
  // count because the refinement changes the latent WITHOUT changing how many
  // forwards ran, so the counter above is blind to it.
  int64_t res2s_bong_steps = 0;
  // The largest deviation from a standardized draw across every noise tensor the
  // res_2s loop was handed: `max(|mean|, |sd - 1|)` over each draw, maximum over
  // all of them. Zero on every non-res_2s pipeline.
  //
  // IT EXISTS BECAUSE THE WIRING IS INVISIBLE OTHERWISE. `_get_new_noise`
  // normalizes (samplers.py:164-170) and `_get_plain_noise` does not
  // (:155-157); the res_2s loop takes the first and the ancestral loop takes
  // the second, ten lines apart in one file. `Ltx2Res2sNormalizeNoise` is gated
  // as a FUNCTION by `test_ltx2_pipeline`, but whether the engine's hook calls
  // it is a different claim, and nothing about a rendered clip, a token count or
  // an evaluation count can answer it. MEASURED: with the engine handing the
  // loop its raw draw, the end-to-end suite stayed GREEN — mutation M10 in
  // .agents/specs/ltx25-res2s-loop.md section 8 — which is why this field was
  // added rather than the wiring being left as a claim.
  //
  // A NORMALIZED draw drives this to ~1e-15 by construction. A raw Gaussian
  // draw cannot: its sample mean is O(1/sqrt(n)) and its sample deviation is
  // O(1/sqrt(n)) away from 1, so on any latent this engine builds the two are
  // orders of magnitude apart rather than close.
  double res2s_noise_moment_error = 0.0;

  // WHAT THE SECOND EVALUATION WAS HANDED, and what it returned for the
  // conditional pass. Empty on every arm but `res2s_two_stage`, and written at
  // phase 0's SECOND evaluation, which on that arm is the substep.
  //
  // IT EXISTS BECAUSE THE SUBSTEP'S x0 CONVERSION HAS NO OTHER OBSERVABLE. The
  // res_2s substep runs over `x_mid` (samplers.py:369-378), a state that never
  // becomes the stream's own latent, so `to_denoised` there must use the latent
  // THAT EVALUATION was handed and not `video.latent`. Those are the same tensor
  // everywhere else in this file, which is what makes the wrong one an easy
  // write and an invisible one.
  //
  // MEASURED: with the conversion reading `video.latent`, the whole
  // `test_ltx2_video` suite stayed GREEN at 74 cases and 2234 assertions. The
  // loop's own arithmetic is gated against upstream with a FIXTURE denoiser, so
  // that gate never sees the engine's conversion; the clip, the evaluation
  // count, the eval sigmas and the bong count are all blind to it.
  //
  // The gate is the per-arm invariant `cond == latent - timesteps * velocity`
  // over THESE tensors — an equation between four recorded vectors, not a
  // magnitude — plus the non-vacuity that `res2s_substep_latent` differs from
  // `video_first_latent`, which is what says the midpoint moved at all.
  std::vector<float> res2s_substep_latent;
  std::vector<float> res2s_substep_timesteps;
  std::vector<float> res2s_substep_cond;
  std::vector<float> res2s_substep_cond_velocity;
  double res2s_substep_sigma = 0.0;

  // ── TEXT-TO-AUDIO: what the audio-only render actually ran (#1005) ────────
  //
  // Zero and false everywhere on a pipeline that is not `t2a_one_stage`.
  //
  // `t2a_video_stream_present` is the one that cannot be inferred from anything
  // else here, and it is the field this block exists for. Upstream's own
  // predicate is `run_v2a = run_ax and (video is not None and vx.numel() > 0)`
  // (transformer.py:269): it tests PRESENCE, not `enabled`. So a build that
  // handed the forward a present-but-DISABLED video stream would still feed
  // video->audio cross attention from a latent T2A never meant to exist — and
  // would return a waveform of exactly the right length, the right channel count
  // and the right sample rate. There is no sample to compare and no digest of
  // the output that says which happened.
  //
  // The three forward counters are incremented AT THE FORWARD, not derived from
  // the guider parameters that were supposed to drive them. A field written off
  // `cfg_scale` would report a healthy uncond count on a build that resolved the
  // params and then ran one forward, which is the instrument failure
  // `audio_sigma_max` above already paid for on this campaign.
  //
  // `t2a_perturbed_blocks` is read off the mask handed to the DiT rather than
  // off the request's `stg_blocks`: a count alone cannot tell "perturbed block
  // 1" from "perturbed block 0", and which block is perturbed is the whole of
  // STG.
  //
  // The `t2a_first_*` block is everything step 0 produced, and it is the only
  // observable that separates upstream's x0-space guidance combination from a
  // velocity-space one (#1039). Every other field here — the forward counts,
  // the perturbed blocks, the latent absmax, the waveform's length, channel
  // count and sample rate — is identical between the two forms, and on a
  // reduced fixture so is the rendered audio, because the guidance deltas are
  // ~1e-5 of the prediction and the rescale factor lands within 1e-5 of 1.0 in
  // BOTH spaces. What is not identical, and is not a matter of degree, is which
  // tensor the guider was handed:
  //
  //     t2a_first_cond == t2a_first_latent - sigma * t2a_first_velocity
  //
  // holds in x0 space (`X0Model.forward`, ltx-core model/transformer/
  // model.py:590-604) and fails in velocity space.
  //
  // ONE (velocity, x0) PAIR PER ARM. The default T2A guider runs THREE forwards
  // per step, and the equation above decides only the pass it names. A build
  // that converts the conditional pass and leaves the UNCONDITIONAL or the
  // PERTURBED one in velocity space renders a different waveform with a healthy
  // forward count, a correct `t2a_first_cond`, and nothing else to see it by —
  // which is #1039 again, one arm over.
  //
  // `t2a_first_next_latent` is `Ltx2EulerStep`'s output, and it makes what the
  // sampler CONSUMED checkable:
  //
  //     next == latent + (latent - denoised)/sigma * (sigma_next - sigma)
  //
  // A second `to_denoised` applied to the guider's result on the way into the
  // step moves this field and no other.
  //
  // The uncond and perturbed vectors are EMPTY when the guider does not ask for
  // that arm, because the forward did not run.
  bool t2a_rendered = false;
  bool t2a_video_stream_present = false;
  int64_t t2a_cond_forwards = 0;
  int64_t t2a_uncond_forwards = 0;
  int64_t t2a_perturbed_forwards = 0;
  std::vector<int64_t> t2a_perturbed_blocks;
  std::vector<float> t2a_first_latent;
  std::vector<float> t2a_first_velocity;
  std::vector<float> t2a_first_cond;
  std::vector<float> t2a_first_uncond_velocity;
  std::vector<float> t2a_first_uncond;
  std::vector<float> t2a_first_perturbed_velocity;
  std::vector<float> t2a_first_perturbed;
  std::vector<float> t2a_first_denoised;
  std::vector<float> t2a_first_next_latent;
  double t2a_first_sigma = 0.0;

  // ── the GUIDED VIDEO denoise, row LTX25-GUIDED-VIDEO (#1092) ──────────────
  //
  // Everything the FIRST step of the FIRST phase produced, and nothing else. One
  // step decides every question below, and recording every step would hold a
  // whole real trajectory in memory.
  //
  // WHY THE VELOCITIES SIT BESIDE THE X0 TENSORS. "Which space was this combined
  // in" is an equation between three tensors — `x0 == latent - sigma*velocity` —
  // and cannot be answered from the x0 tensor alone. It is exact in x0 space and
  // off by the whole sample in velocity space, which is what makes it a gate
  // rather than a tolerance. Recorded PER ARM, because a claim about "every
  // pass" made from one recorded pass is a claim about a quarter of them:
  // #1039's first gate covered the conditional arm alone and three mutations
  // survived it.
  bool video_guided = false;
  int64_t video_cond_forwards = 0;
  int64_t video_uncond_forwards = 0;
  int64_t video_perturbed_forwards = 0;
  int64_t video_modality_forwards = 0;
  // Read off the mask handed to the DiT, not copied from the guider params: a
  // perturbation that is BUILT and not HANDED OVER leaves the params untouched
  // and the render finite.
  std::vector<int64_t> video_perturbed_blocks;
  std::vector<int64_t> video_audio_perturbed_blocks;
  bool video_modality_skipped_a2v = false;
  bool video_modality_skipped_v2a = false;
  // The guidance phase 0 resolved, after the request overrides. The gate replays
  // `Ltx2MultiModalGuidance` over the recorded arms with these, so a build that
  // resolved different params fails the replay instead of agreeing with itself.
  double video_guidance_cfg_scale = 0.0;
  double video_guidance_stg_scale = 0.0;
  double video_guidance_rescale_scale = 0.0;
  double video_guidance_modality_scale = 0.0;
  std::vector<float> video_first_latent;
  std::vector<float> video_first_cond_velocity;
  std::vector<float> video_first_cond;
  std::vector<float> video_first_uncond_velocity;
  std::vector<float> video_first_uncond;
  std::vector<float> video_first_perturbed_velocity;
  std::vector<float> video_first_perturbed;
  std::vector<float> video_first_modality_velocity;
  std::vector<float> video_first_modality;
  // The guider's output BEFORE `post_process_latent`, which is what
  // `Ltx2MultiModalGuidance` returned and what the replay must reproduce
  // exactly.
  std::vector<float> video_first_denoised;
  // And AFTER it, which is what the stepper was handed. Two fields rather than
  // one, because `post_process_latent` is the identity whenever no token is
  // conditioned and a single field could not say which of the two a build passed
  // on.
  std::vector<float> video_first_stepper_input;
  std::vector<float> video_first_next_latent;
  double video_first_sigma = 0.0;
  // The PER-TOKEN timesteps step 0 ran at, so the invariant is checked with the
  // same sigma `ToDenoised` used rather than with the schedule scalar — they
  // differ exactly where a token is conditioned, which is where getting it wrong
  // re-noises a keyframe.
  std::vector<float> video_first_timesteps;

  // ── THE VIDEO DECODE, counted by the RENDER (row LTX25-DEVICE-RESIDENCY W0)
  //
  // How many chunks the streaming video VAE handed back, incremented in the
  // driver's own sink beside `rendered_frames`. It is here because W0's phase
  // table needs ONE number about the decode that the phase table did not
  // produce.
  //
  // WHY THAT MATTERS AND WHY A COUNTER RATHER THAN A LONGER COMMENT. Every
  // assertion W0's containment case makes about `decode.video` — containment,
  // coverage, exclusivity, non-overlap — is a RATIO against the leaf, so an
  // instrument defect that moves the leaf and its sub-scope TOGETHER satisfies
  // all four at once. A count taken by the render is the one quantity such a
  // defect cannot move: emit the chunk scope once instead of once per chunk and
  // the count disagrees, whatever the clock did.
  //
  // ONE PER `emit`, so it is the group count `Ltx2GroupTilesByTemporalSlice`
  // produces for this request's tiling — which the gate re-derives from that
  // function rather than trusting this field.
  int64_t video_decode_chunks = 0;

  // True only once the `Generate` that produced this conditioning RETURNED. The
  // trace is filled immediately after the connector and BEFORE the denoise loop,
  // because that is the only point at which the exact buffers cross-attention
  // will read still exist as such. So a `Generate` that throws in denoise, in a
  // VAE decode or in the muxer leaves a fully populated trace behind for a render
  // that produced no frames. Without this flag the next reader cannot tell that
  // from a completed render, and "which conditioning produced this clip" would
  // answer for a clip that does not exist.
  bool completed = false;
};

// A loaded LTX-2.5 checkpoint set. Construct through
// `vllm::multimodal::LoadVideoEngine` (detection) or by declaring
// `family = kLtx2VideoFamily`; this type is exposed so a test can name it.
class Ltx2VideoEngine : public VideoEngine {
 public:
  static std::unique_ptr<Ltx2VideoEngine> Load(const VideoModelParams& params);

  Ltx2VideoEngine(Ltx2VideoEngine&&) noexcept;
  Ltx2VideoEngine& operator=(Ltx2VideoEngine&&) noexcept;
  ~Ltx2VideoEngine() override;

  std::string family() const override;
  vt::Device device() const override;
  bool has_encoder() const override;
  bool has_prompt_embeds() const override;
  VideoResult Generate(const VideoGenParams& params) override;

  // The `model_version` this engine resolved its recipe with ("2.5"), and the
  // pipeline kind it resolved with. Exposed because "which recipe ran" is the
  // one thing a rendered clip cannot be inspected for.
  const std::string& model_version() const;
  const std::string& pipeline_kind() const;

  // The DiT parameters THIS ENGINE LOADED — the ones its forward actually runs
  // under, after the checkpoint's own declared config has been adopted.
  //
  // Exposed because the values that config decides are exactly the ones a
  // rendered clip cannot be inspected for and a SHAPE cannot see:
  // `double_precision_rope` and `av_ca_timestep_scale_multiplier` move every
  // RoPE angle and every audio<->video modulation while leaving the tensor set
  // byte-identical. A test that re-derives them from the file and asserts on its
  // own local copy proves nothing about what the engine bound; this accessor is
  // what lets it assert on the engine.
  const Ltx2DitParams& dit_params() const;

  // The conditioning the LAST `Generate()` fed cross-attention. `tokens == 0`
  // before the first generation. See `Ltx2ConditioningTrace`.
  //
  // BY VALUE, under the same mutex `Generate` holds. Returning a reference was a
  // data race on the exact use this accessor is FOR: the header offers it to a
  // server ("which conditioning produced this clip"), and a server calls it from
  // a thread that is not the one inside `Generate`. A reference hands the caller
  // a `std::string` and two digests that a concurrent `Generate` is rewriting,
  // and the lock cannot help because it is released before the caller reads. A
  // copy taken while the writer is excluded is the only form that is safe to
  // hand out. Costs one short string copy per call, on a path that renders video.
  Ltx2ConditioningTrace last_conditioning() const;

 private:
  Ltx2VideoEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // `T2AOneStagePipeline.__call__` (ltx2_t2a.h), reached from `Generate` on an
  // `audio_only` recipe and from nowhere else.
  //
  // A PRIVATE STATIC rather than a free function in the .cpp, because it needs
  // `Impl` — a private nested type no non-member can name. The alternatives were
  // a 150-line branch inside a function that is already 1900 lines, or a
  // template whose only purpose is to deduce a type it is not allowed to spell.
  //
  // `audio_context` is the conditioning `Generate` already resolved: the audio
  // half of the prompt encoding, after the connector. Passing it in rather than
  // re-encoding is what keeps the audio-only arm from owning a second copy of
  // the connector composition.
  static VideoResult GenerateAudioOnly(Impl& im, const VideoGenParams& gen,
                                       const float* audio_context, int64_t context_tokens);
};

// Does this checkpoint set hold an LTX-2.5 DiT? Exposed for the registry and for
// a test that wants the answer without a load. See the definition for the
// discriminator and for why it cannot collide with MiniMax-H3's.
bool DetectLtx2Video(const VideoModelParams& params);

}  // namespace vllm::multimodal
