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
//   - Guidance perturbations (guidance/perturbations.py BatchedPerturbationConfig,
//     transformer.py:330-397 `*_perturbation_mask` / `cross_attn_skip_all` and
//     attention.py:545-552 `perturbation_mask`). L2 runs the no-perturbation
//     configuration, whose masks are all-ones and whose flags are all false —
//     upstream's own `perturbations=None` path (model.py:509-511).
//   - `use_keyframes_abs_pos_embedding` (transformer_args.py:23-43). LTX-2.5's
//     checkpoint does not carry the parameter; the enumeration refuses a config
//     that asks for it.
//   - The caption projections (text_projection.py:31-38). LTX-2.5 is a 22B-form
//     checkpoint: `caption_proj_before_connector=true` puts them in the TEXT
//     ENCODER, so the DiT has none (model_configurator.py:199-219). They are
//     phase L3.
//   - `prompt_adaln_single` (model.py:223-227). LTX-2.5 sets
//     `use_prompt_adaln_single=false`; see Ltx2PromptKvCache.
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
  // model_configurator.py:74-76. FALSE on LTX-2.5, which is what makes the
  // cross-attention K/V timestep-independent — see Ltx2PromptKvCache.
  bool use_prompt_adaln_single = true;
  // model_configurator.py:77-80. LTX-2.5 (gemma4) sets ff_bias=false and leaves
  // audio_ff_bias at its true default; the checkpoint's shapes agree.
  bool ff_bias = true;
  bool audio_ff_bias = true;

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

// The K/V half of Attention.forward, split out because LTX-2.5 can CACHE it:
// with `use_prompt_adaln_single=false` the prompt modulation carries no timestep
// term (transformer.py:441), so `to_k`/`to_v` over the modulated context — and
// their k_norm, and the absence of RoPE on the text path — depend only on the
// prompt. The denoise loop computes them ONCE PER REQUEST and reuses them for
// every step. Layout: k/v are [batch * context_tokens, heads * dim_head], held
// exactly as the attention op consumes them (post-norm, post-RoPE).
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

// LTXModel.forward (model.py:492-538) for model_type=AudioVideo, plus the
// preprocessors it drives (transformer_args.py:263-411). BOTH streams are
// required: LTX-2.5 is an AudioVideo checkpoint, and the VideoOnly / AudioOnly
// model types (model.py:31-33) build a different parameter set, so they are
// refused rather than served by an ungated path. To run one stream of an AV model
// — which is what upstream's own pipeline does — clear `enabled` on the other;
// the audio<->video cross attention still reads its state, exactly as
// transformer.py:265-269 does.
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
                              Ltx2PromptKvCache* cache = nullptr);

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
