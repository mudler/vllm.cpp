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
// them: `Ltx2RefuseUnportedPipelineFeature(kVideoEngineWiring)` refused the
// composition BY NAME and named this phase as its owner. This TU is that
// composition and nothing else. It adds no numerics; every line either resolves
// a parameter, moves a buffer, or calls a brick that already has a golden.
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
// and BOTH passed `test_ltx2_video` at 30 cases / 499 assertions with exit 0.
// The digest moved, as it must — but no assertion says WHICH value it should
// have moved to.
//
// THAT COUNT IS THIS HEAD'S, and the distinction is the point of writing it
// down. A reviewer first measured the pair at `43aa58377`, where the suite stood
// at 485 assertions; the numbers were carried forward unchanged while the suite
// grew, so the comment named a count no run of it could produce. Re-run here
// (CPU Release, mutant recompiled and relinked each leg, tree restored
// byte-for-byte and re-verified green between legs): 499/499, exit 0, both.
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
};

// Does this checkpoint set hold an LTX-2.5 DiT? Exposed for the registry and for
// a test that wants the answer without a load. See the definition for the
// discriminator and for why it cannot collide with MiniMax-H3's.
bool DetectLtx2Video(const VideoModelParams& params);

}  // namespace vllm::multimodal
