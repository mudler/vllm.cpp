// LTX-2.5 — the 21.00B joint video+audio flow-matching DiT: layout + forward.
//
// Row: MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model. Spec:
// .agents/specs/ltx-2-5.md (phase L2). Issue #435.
//
// UPSTREAM. Lightricks LTX-2, `packages/ltx-core/src/ltx_core/model/transformer/`.
// Every declaration below names the file:line it was ported from. The mapping is:
//
//   Ltx2DitParams                 <- model_configurator.py:19-83 (LTXModelConfigurator
//                                    .from_metadata) + model.py:48-81 (LTXModel.__init__)
//   Ltx2RopeType                  <- rope.py:11-14 (LTXRopeType)
//   Ltx2PrecomputeFreqsCis        <- rope.py:198-224 (precompute_freqs_cis)
//   Ltx2ApplyRotaryEmb            <- rope.py:16-84 (apply_rotary_emb + both variants)
//   Ltx2AdaLayerNormSingle        <- adaln.py:19-45, timestep_embedding.py:6-143
//   Ltx2Attention                 <- attention.py:478-579 (Attention)
//   Ltx2FeedForward               <- feed_forward.py:6-15, gelu_approx.py:4-10
//   Ltx2TransformerBlockForward   <- transformer.py:87-417 (BasicAVTransformerBlock)
//   Ltx2DitForward                <- model.py:492-538 (LTXModel.forward)
//   EnumerateLtx2DitTensors       <- the named_parameters() of the model above
//   ParseLtx2DitParamsFromManifest<- the SHAPES are the config, mirroring
//                                    ParseMiniMaxH3DitParamsFromGgufManifest
//
// WHAT THIS TU IS. The CORRECTNESS forward, gated against the upstream modules
// executed at REDUCED dimensions on CPU (scripts/gen-ltx2-goldens.py ->
// tests/vllm/models/ltx2_goldens.inc), exactly as MiniMax-H3's DiT is. Every
// projection routes through `vt::MatmulBT`; self-attention routes through
// `vt::Attention(causal=false)`; cross-attention routes through the seam this
// row added for it, `vt::AttentionCross` (vt::Attention rejects Tq != S, which no
// cross-attention can satisfy). The elementwise glue — RMSNorm, LayerNorm,
// gelu-tanh, AdaLN modulation, RoPE, the per-head attention gate — runs as
// explicit host loops so the math reads against upstream line by line.
//
// DTYPE. Everything here is f32. That is NOT a widening of a bf16 path: it is the
// PARITY dtype of the L2 gate, which compares the ALGORITHM against upstream run
// in torch float32. Upstream resolves ONE model dtype and every layer inherits it
// (model.py has no per-layer dtype at all), so the production bf16 / FP8 / NVFP4
// arms are a single stream-dtype choice — phase L6 — and are OWED, not shipped:
// `Ltx2DitForward` REFUSES any `compute_dtype` other than `vt::DType::kF32` with a
// message naming the missing phase rather than silently computing in f32.
//
// NOT PORTED IN L2, recorded here so it cannot be discovered later:
//   - PORTED 2026-08-17 by row LTX25-GUIDED-VIDEO (#1092): the CROSS-attention
//     guidance perturbations, `SKIP_A2V_CROSS_ATTN` and `SKIP_V2A_CROSS_ATTN`
//     (guidance/perturbations.py:8-16, transformer.py:335,367
//     `cross_attn_skip_all`). They are `Ltx2DitPerturbation`'s two booleans.
//
//     WHY THE PREVIOUS ENTRY WAS WRONG RATHER THAN MERELY STALE. It refused them
//     on the ground that "nothing upstream that this port serves constructs
//     them — STG is built from `stg_blocks` and reaches the SELF-attention types
//     alone". STG does. The isolated-modality pass does not: `_guided_denoise`
//     builds BOTH cross types with `blocks=None` whenever either guider has
//     `modality_scale != 1.0` (denoisers.py:121-137, guiders.py:283-285). Every
//     VIDEO row of the params table sets it to 3.0
//     (utils/constants.py:54, :64), so this was upstream's default on the video
//     path the whole time. The sentence was true of text-to-audio, which pins
//     the field to 1.0 (t2a_one_stage.py:202), and it was written while
//     text-to-audio was the only guided path here.
//
//     THE SELF-ATTENTION HALF was ported by #1005. `Ltx2DitPerturbation` is
//     upstream's `perturbations` argument (model.py:493) at the one batch size
//     this port serves, and `Ltx2AttentionArgs::all_perturbed` is
//     `use_attention = not all_perturbed` (attention.py:557). `nullptr` remains
//     upstream's `perturbations=None` path (model.py:509-511) and is what an
//     unguided phase still passes.
//
//     The BATCHED form (`BatchedPerturbationConfig`, perturbations.py:53-143,
//     indexed [type, block, SAMPLE]) is ported in `ltx2_pipeline.h` and reached
//     by `Ltx2GuidedDenoise`, which builds one config over the pass list and
//     slices it per pass. What is still unported is batch > 1 and the partial
//     blend it exists for (`out * mask + v * (1 - mask)`, attention.py:572-573).
//     Both are degenerate at `Ltx2ModalityInput::batch == 1`, which is the only
//     batch any path here runs.
//   - The caption projections (text_projection.py:31-38). LTX-2.5 is a 22B-form
//     checkpoint: `caption_proj_before_connector=true` puts them in the TEXT
//     ENCODER, so the DiT has none (model_configurator.py:199-219). They are
//     phase L3.
//
// PORTED 2026-08-13 — `prompt_adaln_single` / `audio_prompt_adaln_single`
// (model.py:222-227, :252-257), which this list previously carried as unported on
// the strength of "LTX-2.5 sets use_prompt_adaln_single=false". It does not: the
// flag defaults TRUE in both references (model.py:77,
// model_configurator.py:76/:138, diffusers transformer_ltx2.py:1185) and the
// shipped DiT carries the module's tensors. See
// .agents/specs/ltx25-prompt-adaln.md and issue #644.
//
// PORTED 2026-08-14 — `keyframes_abs_pos_embedding` (model.py:217-219,
// transformer_args.py:23-43 called once at :269), which this list previously
// carried as unported on the strength of "LTX-2.5's checkpoint does not carry
// the parameter". THAT WAS FALSE for the shipped vonkaiser FP8 DiT, which
// carries it `F8_E4M3 [1, 4096]` with an `F32` scale and 4096 of 4096 bytes
// NON-ZERO — a trained bias added to every token of the target's first latent
// frame, on every forward. See .agents/specs/ltx25-keyframes-abs-pos.md and
// issue #658.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

// rope.py:11-14. SPLIT is LTX-2.5's setting; INTERLEAVED is upstream's documented
// legacy mode and is ported because a checkpoint may still select it.
enum class Ltx2RopeType { kSplit, kInterleaved };

// LTXModel.__init__ defaults (model.py:48-81), which are also
// LTXModelConfigurator.from_metadata's `config.get(...)` fallbacks
// (model_configurator.py:46-83). `ParseLtx2DitParams` overrides any key the
// checkpoint's `transformer` config actually carries, exactly like from_metadata.
struct Ltx2DitParams {
  int64_t num_layers = 48;
  // Video stream. inner_dim = num_attention_heads * attention_head_dim.
  int64_t num_attention_heads = 32;
  int64_t attention_head_dim = 128;
  int64_t in_channels = 128;
  int64_t out_channels = 128;
  int64_t cross_attention_dim = 4096;  // text context width for `attn2`
  // Audio stream. audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim.
  int64_t audio_num_attention_heads = 32;
  int64_t audio_attention_head_dim = 64;
  int64_t audio_in_channels = 128;
  int64_t audio_out_channels = 128;
  // Serves TWO roles upstream: `audio_attn2`'s context width, and the `inner_dim`
  // the audio<->video cross positional embedding is built at
  // (transformer_args.py:364-371). Both are the audio stream width.
  int64_t audio_cross_attention_dim = 2048;

  double norm_eps = 1e-6;
  double positional_embedding_theta = 10000.0;
  std::vector<int64_t> positional_embedding_max_pos = {20, 2048, 2048};
  std::vector<int64_t> audio_positional_embedding_max_pos = {20};
  int64_t timestep_scale_multiplier = 1000;
  int64_t av_ca_timestep_scale_multiplier = 1;
  bool use_middle_indices_grid = true;
  Ltx2RopeType rope_type = Ltx2RopeType::kSplit;
  // model_configurator.py:68 — `frequencies_precision == "float64"`. Selects the
  // numpy float64 frequency ladder (rope.py:87-107) over the torch float32 one
  // (:110-131); the two differ in the last f32 ulps of every RoPE angle.
  bool double_precision_rope = false;
  bool apply_gated_attention = false;
  bool cross_attention_adaln = false;
  // model_configurator.py:74-76 (`config.get("use_prompt_adaln_single", True)`),
  // model.py:77, diffusers transformer_ltx2.py:1185 — TRUE by default in every
  // reference, and TRUE for the shipped LTX-2.5 DiT, which carries the module's
  // tensors. When true a prompt-side AdaLN MLP adds a timestep term to the
  // cross-attention K/V modulation (transformer.py:441-443), which is what makes
  // those K/V timestep-DEPENDENT and so uncacheable — see Ltx2PromptKvCache.
  bool use_prompt_adaln_single = true;
  // model_configurator.py:77-80. LTX-2.5 (gemma4) sets ff_bias=false and leaves
  // audio_ff_bias at its true default; the checkpoint's shapes agree.
  bool ff_bias = true;
  bool audio_ff_bias = true;
  // model_configurator.py:82/:142 (`config.get("use_keyframes_abs_pos_embedding",
  // False)`) and model.py:80/:101. When set, `_init_video` builds a
  // `(1, inner_dim)` parameter (model.py:217-219) that the VIDEO preprocessor —
  // and only the video one (model.py:314 supplies the provider, :333 does not) —
  // adds to every token the keyframes mask marks.
  //
  // RESOLVED, NOT MERELY DECLARED. This field means what
  // `supports_keyframes_abs_pos_embedding` (model.py:166-173) means: the model has
  // a USABLE embedding. A config that declares the flag over a checkpoint carrying
  // no tensor is upstream-LEGAL and resolves FALSE here, because upstream builds
  // on the meta device (loader/helpers.py:84-95) and loads with
  // `strict=False, assign=True`, so the absent parameter stays on `meta` and the
  // add is never reached. That is the shipped first-party NVFP4 DiT exactly, and
  // it is why `Ltx2AdoptDeclaredDitParams` clears the declared flag rather than
  // refusing the file or synthesising a zero. See
  // .agents/specs/ltx25-keyframes-abs-pos.md §2.
  bool use_keyframes_abs_pos_embedding = false;

  int64_t inner_dim() const { return num_attention_heads * attention_head_dim; }
  int64_t audio_inner_dim() const { return audio_num_attention_heads * audio_attention_head_dim; }
  // adaln.py:14-16 — 6 base params, +3 when cross-attention AdaLN is on.
  int64_t adaln_embedding_coefficient() const { return cross_attention_adaln ? 9 : 6; }
  // model.py:130.
  int64_t cross_pe_max_pos() const {
    return positional_embedding_max_pos[0] > audio_positional_embedding_max_pos[0]
               ? positional_embedding_max_pos[0]
               : audio_positional_embedding_max_pos[0];
  }
  // The audio<->video cross attention projects Q from the VIDEO stream but sizes
  // its heads from the AUDIO stream (transformer.py:154-175), so its inner width
  // is the audio one while its query/out width is the video one. This asymmetry is
  // the one a square assumption silently transposes.
  int64_t av_cross_inner_dim() const { return audio_inner_dim(); }
};

// LTXModelConfigurator.from_metadata (model_configurator.py:19-83). `metadata` is
// the checkpoint's metadata mapping; the transformer keys are read from
// `metadata["config"]["transformer"]` when that nesting is present and from the
// object itself otherwise. Every `check_config_value` upstream asserts
// (:26-44) is asserted here with the same expectation.
Ltx2DitParams ParseLtx2DitParams(const nlohmann::json& metadata);

// One parameter of the DiT, keyed by its EXACT upstream `named_parameters()` name.
struct Ltx2TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};

// The weight contract: every parameter LTXModel(model_type=AudioVideo) creates,
// in a deterministic order. This IS the layout — the parity suite compares it
// against the upstream module's own named_parameters().
std::vector<Ltx2TensorSpec> EnumerateLtx2DitTensors(const Ltx2DitParams& params);

// The SHAPES are the config: recover the geometry from a manifest alone, the way
// a ComfyUI-format checkpoint (which carries no transformer config) has to be
// read. Mirrors ParseMiniMaxH3DitParamsFromGgufManifest. Throws by name when a
// tensor the geometry is derived from is missing.
Ltx2DitParams ParseLtx2DitParamsFromManifest(const std::vector<Ltx2TensorSpec>& manifest);

// ---------------------------------------------------------------------------
// Weight views
// ---------------------------------------------------------------------------

// `bias` is Empty() for a bias-free projection (`ff` on LTX-2.5, ff_bias=false).
struct Ltx2LinearWeight {
  vt::Tensor weight;  // [out_features, in_features]
  vt::Tensor bias;    // [out_features] or empty
};

// attention.py:478-518. `q_norm`/`k_norm` are torch.nn.RMSNorm(INNER_DIM) — the
// norm runs over the WHOLE projected width before the head split (:505-506,
// applied at ops.py:32-33), not per head. `to_gate_logits` is [heads, query_dim]
// (:513-514) and is empty when apply_gated_attention is false.
struct Ltx2AttentionWeights {
  Ltx2LinearWeight to_q, to_k, to_v;
  vt::Tensor q_norm, k_norm;  // [inner_dim]
  Ltx2LinearWeight to_gate_logits;
  Ltx2LinearWeight to_out;  // to_out.0
};

// feed_forward.py:6-15. `net.0.proj` -> gelu(tanh) -> `net.2`.
struct Ltx2FeedForwardWeights {
  Ltx2LinearWeight proj_in;   // net.0.proj, [inner, dim]
  Ltx2LinearWeight proj_out;  // net.2, [dim, inner]
};

// adaln.py:19-45 + timestep_embedding.py:118-143.
struct Ltx2AdaLayerNormSingleWeights {
  Ltx2LinearWeight linear_1;  // emb.timestep_embedder.linear_1, [dim, 256]
  Ltx2LinearWeight linear_2;  // emb.timestep_embedder.linear_2, [dim, dim]
  Ltx2LinearWeight linear;    // [coefficient * dim, dim]
};

// transformer.py:87-189 (BasicAVTransformerBlock's parameters).
struct Ltx2BlockWeights {
  Ltx2AttentionWeights attn1, attn2;
  Ltx2AttentionWeights audio_attn1, audio_attn2;
  Ltx2AttentionWeights audio_to_video_attn;  // Q video, K/V audio
  Ltx2AttentionWeights video_to_audio_attn;  // Q audio, K/V video
  Ltx2FeedForwardWeights ff, audio_ff;
  vt::Tensor scale_shift_table;                // [9, dim]  (:125)
  vt::Tensor audio_scale_shift_table;          // [9, adim] (:150)
  vt::Tensor prompt_scale_shift_table;         // [2, dim]  (:185)
  vt::Tensor audio_prompt_scale_shift_table;   // [2, adim] (:187)
  vt::Tensor scale_shift_table_a2v_ca_video;   // [5, dim]  (:178)
  vt::Tensor scale_shift_table_a2v_ca_audio;   // [5, adim] (:177)
};

// model.py:202-287 (the model-level parameters) plus the block stack.
struct Ltx2DitWeights {
  Ltx2LinearWeight patchify_proj, proj_out;
  Ltx2AdaLayerNormSingleWeights adaln_single;
  // model.py:222-227 / :252-257 — built only when `cross_attention_adaln AND
  // use_prompt_adaln_single`, with embedding_coefficient 2 (shift + scale for the
  // prompt K/V), NOT `adaln_embedding_coefficient()`. Left unbound otherwise.
  Ltx2AdaLayerNormSingleWeights prompt_adaln_single;
  Ltx2AdaLayerNormSingleWeights audio_prompt_adaln_single;
  // model.py:217-219 — [1, dim], built and bound only when
  // `use_keyframes_abs_pos_embedding` RESOLVES true (see that field). Left
  // default-constructed otherwise, which is upstream's `None` provider result.
  vt::Tensor keyframes_abs_pos_embedding;
  vt::Tensor scale_shift_table;  // [2, dim] — the OUTPUT table (:230), not the block's [9, dim]
  Ltx2LinearWeight audio_patchify_proj, audio_proj_out;
  Ltx2AdaLayerNormSingleWeights audio_adaln_single;
  vt::Tensor audio_scale_shift_table;  // [2, adim] (:260)
  Ltx2AdaLayerNormSingleWeights av_ca_video_scale_shift;  // (:269)
  Ltx2AdaLayerNormSingleWeights av_ca_audio_scale_shift;  // (:274)
  Ltx2AdaLayerNormSingleWeights av_ca_a2v_gate;           // (:279)
  Ltx2AdaLayerNormSingleWeights av_ca_v2a_gate;           // (:284)
  std::vector<Ltx2BlockWeights> blocks;
};

// Bind every view in `Ltx2DitWeights` from tensors keyed by their upstream
// parameter name. A name the contract requires and the map lacks throws BY NAME
// rather than reading as zeros. Optional parameters (biases a config turns off,
// the prompt tables when cross_attention_adaln is false, the gate logits when
// gating is off) are bound only when the params say they exist.
Ltx2DitWeights BindLtx2DitWeights(const Ltx2DitParams& params,
                                  const std::map<std::string, vt::Tensor>& tensors);

// ---------------------------------------------------------------------------
// RoPE (rope.py)
// ---------------------------------------------------------------------------

// precompute_freqs_cis (rope.py:198-224). `positions` is the middle-indices grid
// [batch, n_pos_dims, tokens, 2] holding each patch's [start, end) bounds when
// `use_middle_indices_grid` is true, and [batch, n_pos_dims, tokens] otherwise.
// `dim` is the width the frequency ladder is built for — inner_dim for a stream's
// own RoPE, audio_cross_attention_dim for the audio<->video cross RoPE.
//
// Output layout mirrors upstream's tensors exactly:
//   SPLIT       -> cos/sin are [batch, heads, tokens, dim/(2*heads)] (:182-183)
//   INTERLEAVED -> cos/sin are [batch, tokens, dim]                  (:187-195)
struct Ltx2FreqsCis {
  std::vector<float> cos, sin;
  std::vector<int64_t> shape;
};
// generate_freq_grid_pytorch (rope.py:110-131) / generate_freq_grid_np (:87-107):
// `theta ** linspace(0, 1, dim / (2 * n_pos_dims)) * pi/2`, built in float32 or,
// when `double_precision` is set (`frequencies_precision == "float64"`), in
// float64 and cast down. Exposed because the two ladders differ only in the last
// f32 ulps of each sample, and the audio ladder multiplies those up by four
// orders of magnitude before RoPE takes their cosine.
std::vector<float> Ltx2FreqGrid(double theta, int64_t n_pos_dims, int64_t dim,
                                bool double_precision);

// `n_pos_dims` is how many leading position axes are CONSUMED (3 for a video
// stream's own RoPE, 1 for the audio stream and for the audio<->video cross RoPE,
// which slices `positions[:, 0:1, :]` at transformer_args.py:365).
// `source_n_pos_dims` is how many the BUFFER actually carries, so a 1-axis read of
// a 3-axis video grid strides correctly.
Ltx2FreqsCis Ltx2PrecomputeFreqsCis(const double* positions, int64_t batch, int64_t tokens,
                                    int64_t n_pos_dims, int64_t source_n_pos_dims,
                                    bool use_middle_indices_grid, int64_t dim,
                                    const std::vector<int64_t>& max_pos, double theta,
                                    int64_t num_attention_heads, Ltx2RopeType rope_type,
                                    bool double_precision);

// apply_rotary_emb (rope.py:16-27) over `x` [batch, tokens, dim] IN PLACE.
// `heads` is the head count the SPLIT layout was built with.
void Ltx2ApplyRotaryEmb(float* x, int64_t batch, int64_t tokens, int64_t dim, int64_t heads,
                        const Ltx2FreqsCis& pe, Ltx2RopeType rope_type);

// ---------------------------------------------------------------------------
// Bricks (each gated on its own so a failure localizes)
// ---------------------------------------------------------------------------

// AdaLayerNormSingle.forward (adaln.py:39-45). `timesteps` is [count] ALREADY
// scaled by timestep_scale_multiplier (transformer_args.py:177). Returns the
// modulation `linear(silu(emb))` in `modulation` [count, coefficient * dim] and
// the embedding itself in `embedded` [count, dim].
struct Ltx2AdalnOut {
  std::vector<float> modulation;
  std::vector<float> embedded;
};
Ltx2AdalnOut Ltx2AdaLayerNormSingle(vt::Device device, const Ltx2AdaLayerNormSingleWeights& w,
                                    const float* timesteps, int64_t count, int64_t dim);

// FeedForward.forward (feed_forward.py:14-15): net.0.proj -> gelu(tanh) -> net.2.
std::vector<float> Ltx2FeedForward(vt::Device device, const Ltx2FeedForwardWeights& w,
                                   const float* x, int64_t rows, int64_t dim, int64_t inner);

// The K/V half of Attention.forward, split out because a checkpoint that sets
// `use_prompt_adaln_single=false` can CACHE it: the prompt modulation then
// carries no timestep term (transformer.py:441-443), so `to_k`/`to_v` over the
// modulated context — and their k_norm, and the absence of RoPE on the text path
// — depend only on the prompt. The denoise loop computes them ONCE PER REQUEST
// and reuses them for every step. Layout: k/v are
// [batch * context_tokens, heads * dim_head], held exactly as the attention op
// consumes them (post-norm, post-RoPE).
//
// THIS DOES NOT APPLY TO THE SHIPPED LTX-2.5 DiT, which sets the flag TRUE
// (.agents/specs/ltx-2-5.md §1.2, and .agents/specs/ltx25-prompt-adaln.md). The
// mechanism stays here, gated bit-identical, for a checkpoint that does set it
// false; `Ltx2DitForward` refuses a cache when the flag is on.
struct Ltx2CrossKv {
  std::vector<float> k, v;
};

// The identity of the PROMPT a cache was filled for.
//
// The cached K/V are timestep-independent, so reusing them across DENOISE STEPS
// is exact. They are NOT prompt-independent, and a geometry check cannot tell
// two different prompts of equal token count apart — which is precisely the case
// a pipeline hits when one cache outlives one request. Serving request 1's K/V
// to request 2 renders request 1's prompt with no shape mismatch, no non-finite
// value and no error, so the cache carries a CONTENT fingerprint over both
// streams' context tensors, their geometry, and their prompt masks, and a
// mismatch is refused by name.
//
// The digests are FNV-1a over the raw bytes: an exact-identity test, never an
// approximate one. Two prompts that differ in one f32 ulp are two prompts.
struct Ltx2PromptIdentity {
  int64_t batch = 0;
  int64_t video_context_tokens = 0, audio_context_tokens = 0;
  int64_t video_context_dim = 0, audio_context_dim = 0;
  uint64_t video_context_digest = 0, audio_context_digest = 0;
  uint64_t video_mask_digest = 0, audio_mask_digest = 0;
};

// Per-request cache of every block's text-cross-attention K/V, both streams.
// `valid` is set by the first forward that fills it; later forwards reuse it —
// but ONLY for the prompt `prompt` records. Gated BIT-IDENTICAL against
// recomputation, and gated to REFUSE a changed prompt: a cache that silently
// diverges is exactly the failure this is here to make impossible.
struct Ltx2PromptKvCache {
  bool valid = false;
  Ltx2PromptIdentity prompt;
  std::vector<Ltx2CrossKv> video;  // one per block
  std::vector<Ltx2CrossKv> audio;  // one per block

  // Unbind the cache from its prompt so the next forward re-fills it for a new
  // one. This is how a pipeline REUSES the allocation across requests; anything
  // else is refused rather than served stale.
  void Reset() {
    valid = false;
    prompt = Ltx2PromptIdentity{};
    video.clear();
    audio.clear();
  }
};

// Attention.forward (attention.py:520-579). `context` is nullptr for
// self-attention (upstream's `context = x if context is None else context`).
// `pe`/`k_pe` are nullptr when no RoPE applies (every text cross-attention).
// `bias` is the optional additive score bias, [tokens, context_tokens] or
// [1, context_tokens], nullptr for none.
struct Ltx2AttentionArgs {
  int64_t batch = 1;
  int64_t tokens = 0;          // Tq
  int64_t context_tokens = 0;  // S; equals `tokens` for self-attention
  int64_t query_dim = 0;
  int64_t context_dim = 0;
  int64_t heads = 0;
  int64_t dim_head = 0;
  // The q/k RMSNorm eps: `Attention.__init__`'s `norm_eps: float = 1e-6`
  // (attention.py:485), handed to both RMSNorms (attention.py:505-506).
  //
  // This DEFAULT is read by nothing today — every construction of this struct
  // assigns it — and a 10^6 mutation of it leaves every suite green. It is a
  // latent trap rather than live code, so the only instrument that can hold it is
  // the pin in tests/vllm/models/test_ltx2.cpp. Keep it equal to
  // `Ltx2DitParams::norm_eps`, which is where every real call site sources it.
  double norm_eps = 1e-6;
  Ltx2RopeType rope_type = Ltx2RopeType::kSplit;
  const Ltx2FreqsCis* pe = nullptr;
  const Ltx2FreqsCis* k_pe = nullptr;
  const float* bias = nullptr;
  int64_t bias_rows = 0;
  // Prompt-K/V cache hooks. `kv_in` supplies K/V to reuse instead of projecting
  // the context again (the `to_v` / `to_k` / k_norm chain is skipped wholesale);
  // `kv_out`, when non-null, receives the K/V this call computed. Legal only on
  // a path whose K/V really are timestep-independent — the caller owns that.
  const Ltx2CrossKv* kv_in = nullptr;
  Ltx2CrossKv* kv_out = nullptr;

  // `all_perturbed` (attention.py:552-553, `use_attention = not all_perturbed`
  // at :557). The STG perturbation: the attention output is REPLACED by the raw
  // value projection, `out = v`, and `to_q` / `to_k` / the q,k RMSNorms / RoPE /
  // the attention itself never run. `to_out` still does (`:579`), and the gate
  // still does (`:576-578`) — the substitution happens INSIDE the attention, not
  // around it, so a port that returned `v` to the caller and skipped `to_out`
  // would be a different operator.
  //
  // Upstream also carries a PARTIAL form, `out = out * mask + v * (1 - mask)`
  // (`:571-572`), which blends per BATCH ROW. It is not ported here and it is not
  // needed: it exists so one batch can mix perturbed and unperturbed samples,
  // and this port runs `max_batch_size = 1` (`Ltx2ModalityInput::batch`), where
  // the mask is all-ones or all-zeros and the blend degenerates to exactly the
  // two cases this flag expresses. Recorded rather than assumed, because a
  // batched arm that reached this field would need the blend and would get the
  // all-or-nothing answer silently.
  bool all_perturbed = false;
};
std::vector<float> Ltx2Attention(vt::Device device, const Ltx2AttentionWeights& w, const float* x,
                                 const float* context, const Ltx2AttentionArgs& args);

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// One modality's inputs — the fields of `Modality` (modality.py:9-63) this phase
// consumes. Row-major host buffers.
struct Ltx2ModalityInput {
  int64_t batch = 1;
  int64_t tokens = 0;          // T
  int64_t context_tokens = 0;  // S
  const float* latent = nullptr;    // [batch, tokens, in_channels]
  const float* timesteps = nullptr; // [batch, tokens]
  const float* sigma = nullptr;     // [batch]
  // [batch, n_pos_dims, tokens, 2] when use_middle_indices_grid, else
  // [batch, n_pos_dims, tokens]. n_pos_dims is 3 for video, 1 for audio.
  const double* positions = nullptr;
  const float* context = nullptr;   // [batch, context_tokens, context_dim]
  // Optional prompt mask in {0, 1} over the context tokens, [batch, context_tokens].
  // Converted by `(mask - 1) * finfo(f32).max` exactly as
  // TransformerArgsPreprocessor._prepare_attention_mask does (transformer_args.py:199-206).
  const int32_t* context_mask = nullptr;
  // Optional self-attention STRENGTH mask in [0, 1], [batch, tokens, tokens] or
  // the key-only broadcast [batch, 1, tokens]. Converted to an additive log-space
  // bias exactly as _prepare_self_attention_mask does (transformer_args.py:208-237).
  const float* attention_mask = nullptr;
  int64_t attention_mask_rows = 0;  // `tokens` for the dense form, 1 for key-only
  // Modality.keyframes_mask (modality.py:63), [batch, tokens] — upstream's
  // (B, T, 1) with its trailing broadcast axis dropped, because the marker is
  // per TOKEN and the embedding it selects is per channel.
  //
  // `nullptr` is upstream's `keyframes_mask is None`, which
  // `apply_keyframes_absolute_embedding` short-circuits (transformer_args.py:37-38).
  // Only the VIDEO stream ever carries one: `_init_preprocessors` gives the video
  // preprocessor a `keyframes_embedding_provider` (model.py:314) and the audio one
  // none (:333), so a mask on the audio input is REFUSED rather than applied.
  //
  // `_keyframes_embedding` is handed over at TWO sites, and BOTH are video. The
  // `:314` above is the joint audio+video arm's `MultiModalTransformerArgsPreprocessor`;
  // the video-ONLY arm builds a plain `TransformerArgsPreprocessor` and passes the
  // same provider at `:349`. Their audio counterparts (`:316-334` and `:352-365`)
  // take none. The split is by MODALITY, not by which arm ran, so a grep for the
  // `:314` anchor that lands on `:349` has found the same rule, not a second one.
  //
  // The rule that fills it is `_first_frame_keyframes_mask` (tools.py:186-196) —
  // the target's first latent frame, marked UNCONDITIONALLY, whether or not any
  // keyframe was supplied. `Ltx2FirstFrameKeyframesMask` (ltx2_conditioning.h) is
  // that rule; do not re-derive it at a call site.
  const float* keyframes_mask = nullptr;
  bool enabled = true;              // Modality.enabled -> TransformerArgs.enabled
};

// The prompt identity a `Ltx2PromptKvCache` is bound to, over the two streams'
// context tensors, their geometry and their prompt masks — every input the
// cached K/V are a function of. Exposed so a caller can key its own per-request
// cache the same way rather than re-deriving the rule.
Ltx2PromptIdentity Ltx2PromptIdentityOf(const Ltx2DitParams& params,
                                        const Ltx2ModalityInput& video,
                                        const Ltx2ModalityInput& audio);

struct Ltx2DitOutputs {
  std::vector<float> video;  // [batch, video tokens, out_channels]
  std::vector<float> audio;  // [batch, audio tokens, audio_out_channels]
};

// `perturbations` on LTXModel.forward (model.py:493), reduced to what this port
// can express. One entry per BLOCK; `true` means that block's self-attention is
// replaced by its value projection, which is STG.
//
// A `BatchedPerturbationConfig` upstream (guidance/perturbations.py:53-143) is
// indexed [type, block, SAMPLE]; this is indexed [block] alone, because
// `Ltx2ModalityInput::batch` is 1 on every path here and the sample axis is a
// degenerate one. `Ltx2BatchedPerturbationConfig` (ltx2_pipeline.h) is the
// batched form, and row LTX25-GUIDED-VIDEO (#1092) is what gave it a product
// caller: `Ltx2GuidedDenoise` builds one config over all four passes and slices
// it per pass, which is `denoisers.py:182-187` and is where the ONE-sample
// flattening below happens.
//
// EMPTY IS NOT "NOTHING PERTURBED BY COINCIDENCE": a vector of the wrong length
// is REFUSED, so a config built for a different layer count cannot silently
// perturb the first N blocks and leave the rest alone.
//
// ALL FOUR upstream perturbation types are represented here. The two CROSS
// directions arrived with #1092 and are booleans rather than per-block vectors,
// because the one thing that builds them asks for ALL blocks
// (`Perturbation(type=..., blocks=None)`, denoisers.py:132-135) and upstream's
// own reader is the per-block scalar `cross_attn_skip_all` (transformer.py:335,
// :367) rather than a mask multiply. A per-block cross vector would be a surface
// with no constructor.
//
// WHAT THIS ENTRY USED TO SAY, kept because the sentence was load-bearing and
// wrong: "Nothing upstream that this port serves constructs them: STG is built
// from `stg_blocks` and reaches the self-attention types only". That was true
// while text-to-audio was the only guided path here — it pins
// `modality_scale = 1.0` (t2a_one_stage.py:202), which is exactly the value
// `do_isolated_modality_generation` reads as OFF. Every VIDEO row defaults it to
// 3.0 (utils/constants.py:54, :64), so the isolated-modality pass is upstream's
// DEFAULT there and these two types are on the reachable path.
struct Ltx2DitPerturbation {
  std::vector<uint8_t> video_self_attn;  // [num_layers], empty = none
  std::vector<uint8_t> audio_self_attn;  // [num_layers], empty = none
  // `cross_attn_skip_all` on the VIDEO args, i.e. SKIP_A2V_CROSS_ATTN: the
  // audio-to-video direction, which WRITES the video stream.
  bool video_cross_attn_skip_all = false;
  // `cross_attn_skip_all` on the AUDIO args, i.e. SKIP_V2A_CROSS_ATTN.
  bool audio_cross_attn_skip_all = false;
};

// LTXModel.forward (model.py:492-538), plus the preprocessors it drives
// (transformer_args.py:263-411).
//
// EXACTLY ONE OF THE TWO STREAMS MAY BE NULL, which is upstream's own
// `video_args = ... if video is not None else None` (model.py:505) and the
// shape `T2AOneStagePipeline` runs (t2a_one_stage.py:167 passes `video=None`).
// Both null is refused: upstream refuses it too (transformer.py:259-260, "At
// least one of video or audio must be provided").
//
// `video = nullptr` IS NOT `video->enabled = false`, and the difference renders
// rather than failing. Upstream's predicate is `run_v2a = run_ax and (video is
// not None and vx.numel() > 0)` (transformer.py:269) — it tests PRESENCE, not
// `enabled` — so a disabled-but-present video stream still feeds video->audio
// cross attention from that stream's latent. This header used to advise
// `enabled` as the way to run one stream, and for the audio-only case that
// advice was wrong. `enabled` remains correct for running one stream of a JOINT
// render, where the cross attention reading the other stream's state is the
// intent.
//
// WHAT IS *NOT* THE REASON, re-derived at this tree rather than inherited: the
// AudioOnly / VideoOnly WEIGHT CONTRACT is not what blocks a one-stream call, and
// this check used to say it was. `T2AOneStagePipeline` loads the ordinary
// AudioVideo checkpoint FILE and merely restricts which keys are read
// (LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP, model_configurator.py:228-239), so
// the contract `EnumerateLtx2DitTensors` describes is the one it satisfies. What
// remains true is a statement about the LOADER: a checkpoint saved with only the
// audio subset still cannot be materialized here, and that refusal lives at
// `Ltx2LoadDitFromSafetensors`, where it is about the file.
//
// `compute_dtype` must be vt::DType::kF32 — see the DTYPE note at the top of this
// file. Anything else is REFUSED with a message naming phase L6.
//
// `cache`, when non-null, is the prompt-K/V cache: filled on the first call and
// reused afterwards. It is only legal when `use_prompt_adaln_single` is false —
// with the prompt AdaLN MLP enabled the K/V carry a timestep term and caching
// them would be wrong, so that combination is refused rather than approximated.
// A filled cache is bound to the PROMPT it was filled for: a call whose context
// tensors, context geometry or prompt masks differ from that prompt is REFUSED
// by name (call `Ltx2PromptKvCache::Reset()` to rebind it to a new request)
// rather than served K/V that would render the previous prompt.
Ltx2DitOutputs Ltx2DitForward(vt::Device device, const Ltx2DitParams& params,
                              const Ltx2DitWeights& weights, const Ltx2ModalityInput* video,
                              const Ltx2ModalityInput* audio, vt::DType compute_dtype,
                              Ltx2PromptKvCache* cache = nullptr,
                              const Ltx2DitPerturbation* perturbations = nullptr);

// One BasicAVTransformerBlock (transformer.py:254-417), exposed so the block is
// gateable on its own. `video_x` / `audio_x` are updated IN PLACE.
struct Ltx2BlockArgs {
  int64_t batch = 1;
  int64_t video_tokens = 0, audio_tokens = 0;
  int64_t video_context_tokens = 0, audio_context_tokens = 0;
  bool video_enabled = true, audio_enabled = true;
  // Per-token AdaLN modulation, [batch, tokens, coefficient * dim].
  const float* video_timestep_modulation = nullptr;
  const float* audio_timestep_modulation = nullptr;
  // The PROMPT-side AdaLN modulation, [batch, 1, 2 * dim] — shift then scale, one
  // row per batch element broadcast over the prompt tokens (transformer.py:443,
  // whose `prompt_timestep` has token dimension 1 because `_prepare_timestep` ran
  // on the modality's per-sample `sigma`). `nullptr` is upstream's
  // `prompt_timestep is None`, i.e. `use_prompt_adaln_single=false`, in which case
  // only the static per-block table applies (:441).
  const float* video_prompt_modulation = nullptr;
  const float* audio_prompt_modulation = nullptr;
  // STG for THIS block (attention.py:552-577). See `Ltx2DitPerturbation`.
  bool video_self_attn_perturbed = false;
  bool audio_self_attn_perturbed = false;
  // `cross_attn_skip_all` (transformer_args.py:70, read at transformer.py:335
  // and :367). THE FLAG RIDES ON THE STREAM BEING WRITTEN, not on the stream
  // being read: `video.cross_attn_skip_all` skips A2V, which writes the VIDEO
  // stream from audio keys, and `audio.cross_attn_skip_all` skips V2A.
  //
  // THIS COMMENT USED TO SAY A TEST COULD NOT SEPARATE THEM, and it was the
  // wrong conclusion from a true premise. The premise: on the SHIPPED path both
  // directions are off together, because `_guided_denoise` builds the
  // isolated-modality pass with BOTH (denoisers.py:125-138), so swapping the two
  // flags renders identically there. The conclusion does not follow, because
  // nothing obliges the separating test to use the shipped combination.
  // `run_a2v` and `run_v2a` read the two streams' `enabled` flags
  // asymmetrically (transformer.py:265-269), so a forward with one stream
  // PRESENT but DISABLED runs exactly one cross direction and each flag becomes
  // observable alone. That is what
  // "ltx2 dit: each CROSS perturbation gates ITS OWN direction and no other"
  // does; it is red against a build that applies only one direction (M12, M13)
  // and against a build that swaps them (M15), all three of which were GREEN
  // over the shipped-path case alone.
  bool video_cross_attn_skip_all = false;
  bool audio_cross_attn_skip_all = false;
  // Audio<->video cross-attention AdaLN inputs (transformer_args.py:388-411).
  const float* video_cross_scale_shift = nullptr;  // [batch, video tokens, 4 * dim]
  const float* video_cross_gate = nullptr;         // [batch, 1, dim]
  const float* audio_cross_scale_shift = nullptr;  // [batch, audio tokens, 4 * adim]
  const float* audio_cross_gate = nullptr;         // [batch, 1, adim]
  const float* video_context = nullptr;
  const float* audio_context = nullptr;
  const float* video_context_bias = nullptr;  // additive, [batch * 1, S] broadcast
  const float* audio_context_bias = nullptr;
  const float* video_self_bias = nullptr;  // additive, [batch * rows, T]
  int64_t video_self_bias_rows = 0;
  const float* audio_self_bias = nullptr;
  int64_t audio_self_bias_rows = 0;
  const Ltx2FreqsCis* video_pe = nullptr;
  const Ltx2FreqsCis* audio_pe = nullptr;
  const Ltx2FreqsCis* video_cross_pe = nullptr;
  const Ltx2FreqsCis* audio_cross_pe = nullptr;
  Ltx2CrossKv* video_prompt_kv = nullptr;  // non-null: use if `filled`, else fill
  Ltx2CrossKv* audio_prompt_kv = nullptr;
  bool prompt_kv_filled = false;
};
void Ltx2TransformerBlockForward(vt::Device device, const Ltx2DitParams& params,
                                 const Ltx2BlockWeights& weights, const Ltx2BlockArgs& args,
                                 float* video_x, float* audio_x);

// ---------------------------------------------------------------------------
// Mask preparation (transformer_args.py:199-237)
// ---------------------------------------------------------------------------

// _prepare_attention_mask (:199-206): a {0,1} prompt mask becomes
// `(mask - 1) * finfo(f32).max`, i.e. 0 for a kept key and -FLT_MAX for a masked
// one. Result is [batch, context_tokens], one broadcast row per batch element.
std::vector<float> Ltx2PrepareContextMask(const int32_t* mask, int64_t batch,
                                          int64_t context_tokens);

// _prepare_self_attention_mask (:208-237): a [0,1] STRENGTH mask becomes
// `log(clamp(m, min=tiny))` for m > 0 and finfo(f32).min for m <= 0.
std::vector<float> Ltx2PrepareSelfAttentionMask(const float* mask, int64_t count);

}  // namespace vllm
