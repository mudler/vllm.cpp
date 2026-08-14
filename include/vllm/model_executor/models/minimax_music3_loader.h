// MiniMax-Music3 — the modular six-component checkpoint loader (W1 of #672).
//
// Row MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation,
// .agents/specs/minimax-music3.md phase W1. Issue #672.
//
// ─── WHAT THIS PHASE IS, AND WHAT IT IS NOT ─────────────────────────────────
//
// It resolves the shipped `diffusers` layout, parses each component's config,
// enumerates the tensors that config OWES, and accounts every tensor the files
// actually carry against that enumeration. No forward, no scheduler step, no
// generation: those are W2..W6. What it buys is that a wrong-shaped or
// wrong-dtype component is a LOAD-TIME refusal naming the component and the
// tensor, instead of a plausible, finite, wrong song several phases later.
//
// ─── TWO PACKAGINGS, AND ONLY ONE IS LOADED ─────────────────────────────────
//
// `MiniMaxAI/MiniMax-Music3` ships the same weights twice (spec section 2). The
// NATIVE arm is `qwen_7B/qwen_7B/` + `flowmatching_vae.pth` + `dav.pth`
// (convert_minimax_music3_to_diffusers.py:30,34,38) and is what SGLang-Omni
// serves; the DIFFUSERS arm is the six safetensors components this port loads.
//
// The native arm is REFUSED BY NAME. It is not silently mis-loaded, and the
// refusal names the diffusers components that are missing rather than saying
// "unsupported": a `qwen_7B/` tree pointed at this loader has to be told which
// artifact it is and which one this port wants, or the next person re-derives
// it. `MiniMaxMusic3ResolveCheckpoint` is where that happens.
//
// ─── ON-DISK DTYPE AND RUNTIME DTYPE ARE DIFFERENT THINGS HERE ──────────────
//
// WHAT THE FILES STORE, measured from the headers and not in dispute:
// `transformer`, `condition_encoder` and `vocoder` are F32,
// `rvq_depth_decoder` and `language_model` are BF16. That is what
// `convert_minimax_music3_to_diffusers.py` writes — `:267` defaults
// `--dtype float32`, `:208-211` applies it to three components, `:214` forces
// the depth decoder to bf16 — and `EnumerateMiniMaxMusic3*Tensors` states it
// per component so `MiniMaxMusic3AccountTensors` can refuse a file that
// disagrees. Those dtypes are FACTS ABOUT THE FILES and they stay.
//
// WHAT WILL ACTUALLY RUN IS NOT THAT SET, and this correction is the reason
// this section was rewritten. The spec's section 2.1 read the converter's
// output as upstream's resolved RUNTIME policy. Running the oracle refuted it:
// the on-disk set is NOT RUNNABLE through upstream's own pipeline.
//
// Upstream casts in exactly two places and nowhere else:
//
//   denoise.py:83    the condition encoder's OUTPUT -> transformer.dtype
//   decoders.py:84   the latents                    -> vocoder.dtype
//
// Nothing casts on the way IN. `denoise.py:82` hands
// `block_state.frame_hiddens[...].to(device)` to the condition encoder with a
// device move and no dtype move, so the encoder and the depth decoder consume
// the LANGUAGE MODEL's hidden states at the language model's dtype. Load the
// on-disk set and `condition_embedder_minimax_music3.py:64` raises
//
//   RuntimeError: Input type (c10::BFloat16) and bias type (float)
//                 should be the same
//
// because `self.proj` is an fp32 `nn.Conv1d` being fed bf16. The two lines
// above it are the tell: `:61` and `:63` cast `layer_weight_logits` and
// `layer_scale` with `.to(hidden_states.dtype)`, so the learned MIX follows the
// hidden states — but `proj` is a MODULE and is never cast, so it cannot.
//
// THE INVARIANT THAT ACTUALLY HOLDS, and what this loader enforces:
//
//   dtype(language_model) == dtype(rvq_depth_decoder) == dtype(condition_encoder)
//
// The gated configuration is `kBf16ArFp32Acoustic`: the AR half (language
// model, depth decoder, condition encoder) in bf16 and the acoustic half
// (transformer, vocoder) in fp32. That is also the converter's default for the
// DiT and the vocoder, and what SGLang-Omni states it runs.
//
// `kAsStored` is kept SELECTABLE rather than deleted, so the failure stays
// reproducible against the oracle's own `--dtype-policy on-disk`. It is
// reported as not-runnable rather than quietly repaired.
//
// So the fp32 on the acoustic half still needs no apology under AGENTS.md's
// too-wide rule — the oracle runs fp32 there too. What is NOT true, and what
// this file previously said, is that the whole on-disk set is the runtime
// policy. `MiniMaxMusic3CheckRuntimeDtypes` refuses a violating configuration
// BY NAME, naming all three AR components and their dtypes, because a refusal
// that says which three disagree is worth more than the torch type error
// upstream gives from inside a forward pass.
//
// ─── WEIGHT NORM IS FOLDED AT LOAD, NOT REPRODUCED ──────────────────────────
//
// The vocoder ships torch's LEGACY `weight_g`/`weight_v` pair for all 30 of its
// weight-normed convolutions — MEASURED from the header, not counted by eye:
// `conv_in` and `conv_out`, plus per block one `conv_t1` and three residual
// units of two convs each (7 x 4 = 28). With the 2 `dec_in_proj` tensors and
// the 29 snake alphas that is 2 + 30*3 + 29 = 121, the component's whole tensor
// count. (`weight_norm(...)` at minimax_music3_vocoder.py:42,44,55,89,98.)
// `w = g * v / ||v||` with the norm taken over every dimension but dim 0 — a
// pure function of the stored parameters, constant for the whole run.
//
// DECISION: FOLD, do not reproduce the parameterization. Three reasons, in
// order. (1) It is what torch itself computes once and caches; reproducing it
// per forward would recompute a constant every denoise step. (2) The forward
// then consumes ONE weight per convolution, so W5's decoder is the same shape
// as every other conv decoder in this tree and cannot accidentally read `v` as
// if it were `w` — which is precisely the mis-load minimax_h3_vae_loader.cpp:11
// records catching. (3) Folding is checkable in isolation, which is what the
// mutation gate in test_minimax_music3_loader.cpp does; a parameterization
// spread across a forward is not.
//
// The fold itself is NOT a new function. `vocoder1d::MaterializeWeightNorm` is
// the one home, shared with MiniMax-H3's audio VAE, and
// tests/scripts/test_vocoder1d_single_home.py asserts it stays that way.
//
// ─── WHAT THE SHARED SEAMS DO AND DO NOT ALREADY COVER ──────────────────────
//
// `vocoder1d` is a PRIMITIVE library — Conv1d, ConvTranspose1d, Pad1d,
// SnakeActivation, AliasFreeActivation1d, KaiserSincFilter1d — not an
// instantiable decoder. There is no `Vocoder1D` class for Music3's vocoder to
// be a configuration of, so this loader materializes weights and W5 will
// compose the primitives. Two of them fit unchanged and are recorded here so
// W5 does not re-derive it: Music3's `MiniMaxMusic3Snake1d`
// (minimax_music3_vocoder.py:25-34) is `x + (alpha + 1e-9)^-1 * sin^2(alpha*x)`,
// which is EXACTLY `vocoder1d::SnakeActivation` with a null `beta` and
// `logscale=false`, down to `kSnakeEps`; and Music3 uses PLAIN snake with no
// up/downsample, so `AliasFreeActivation1d` does not apply to it.
//
// `multimodal::SpeechEngine` is the TTS engine seam (text + reference clip ->
// mono waveform). It is the right home for W6's end-to-end surface and it is
// not a loader seam: it has no notion of a component set, and nothing in it is
// reachable from a phase that only reads headers. W1 therefore mirrors
// `ltx2_loader.h` — the other multi-component diffusion checkpoint — and adds
// nothing to `SpeechEngine`.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {

class SafetensorsFile;

// ---------------------------------------------------------------------------
// Which packaging is on disk
// ---------------------------------------------------------------------------

// The three files/directories that IDENTIFY the native arm, exactly as
// convert_minimax_music3_to_diffusers.py:30,34,38 resolves them and as
// sglang_omni/models/minimax_music3/checkpoint.py:35-56 serves them.
inline constexpr const char* kMusic3NativeDitFile = "flowmatching_vae.pth";
inline constexpr const char* kMusic3NativeVaeFile = "dav.pth";
inline constexpr const char* kMusic3NativeQwenDir = "qwen_7B";

// The six components of the diffusers arm, in `modular_model_index.json` order.
inline constexpr const char* kMusic3Components[] = {
    "condition_encoder", "language_model", "rvq_depth_decoder",
    "scheduler",         "tokenizer",      "transformer",
};
inline constexpr int kMusic3ComponentCount = 6;
// `vocoder` is a seventh directory that `modular_model_index.json` also lists;
// the array above is the index's own key order and the vocoder sorts last.
inline constexpr const char* kMusic3VocoderComponent = "vocoder";

// Every resolved path a load needs. Produced only for a checkpoint that IS the
// diffusers arm.
struct MiniMaxMusic3Paths {
  std::string root;
  std::string modular_index;       // modular_model_index.json
  std::string transformer_dir;
  std::string condition_encoder_dir;
  std::string rvq_depth_decoder_dir;
  std::string vocoder_dir;
  std::string language_model_dir;
  std::string scheduler_config;    // scheduler/scheduler_config.json
  std::string tokenizer_dir;
  // Shard files, in index order, for the two sharded components.
  std::vector<std::string> transformer_shards;
  std::vector<std::string> language_model_shards;
};

// Resolve `root` as the diffusers arm, or THROW naming what is wrong.
//
// Three refusals, and they are distinguishable on purpose:
//
//   * a NATIVE-arm tree (`qwen_7B/` + `flowmatching_vae.pth` + `dav.pth`, or
//     any of the three) is named as the native arm, told that only the
//     diffusers arm is supported, and told which diffusers components it lacks;
//   * a tree that is neither is refused for the components it lacks;
//   * a diffusers tree missing one component names that component.
//
// The native-arm case is separated because it is the one that would otherwise
// LOOK loadable: it holds every weight this port needs, in a layout nothing
// here reads. Recorded as owed in the spec rather than discovered at W6.
MiniMaxMusic3Paths MiniMaxMusic3ResolveCheckpoint(const std::string& root);

// True when `root` carries any native-arm marker. Never throws: used by the
// refusal to decide WHICH message to raise, and exposed so a caller can ask.
bool MiniMaxMusic3IsNativeArm(const std::string& root);

// ---------------------------------------------------------------------------
// The configs, as the released checkpoint states them
// ---------------------------------------------------------------------------

// transformer/config.json. F32 on disk (spec section 2.1: upstream's resolved
// `--dtype float32`, convert_minimax_music3_to_diffusers.py:208,267).
struct MiniMaxMusic3TransformerConfig {
  int64_t in_channels = 128;
  int64_t condition_dim = 2048;
  int64_t num_layers = 36;
  int64_t num_attention_heads = 32;
  int64_t attention_head_dim = 64;
  int64_t ff_inner_dim = 8192;
  int64_t rotary_dim = 32;
  int64_t fourier_embedding_dim = 256;

  // transformer_minimax_music3.py:177 — `num_attention_heads * attention_head_dim`.
  int64_t inner_dim() const { return num_attention_heads * attention_head_dim; }
  // transformer_minimax_music3.py:178 — the input concatenates
  // [latent, zeros(in_channels), condition] along channels.
  int64_t concat_channels() const { return 2 * in_channels + condition_dim; }
};

// condition_encoder/config.json. F32 on disk (spec section 2.1).
//
// FOUR TENSORS, and that is the finding rather than an omission: the module is
// a learned weighted MIX over `num_condition_layers` language-model hidden
// layers plus one Conv1d, not an encoder tower
// (condition_embedder_minimax_music3.py:44-46).
struct MiniMaxMusic3ConditionEncoderConfig {
  int64_t condition_hidden_dim = 4096;
  int64_t num_condition_layers = 8;
  int64_t out_dim = 2048;
  int64_t input_sampling_rate = 24000;
  int64_t input_hop_length = 960;
  int64_t output_sampling_rate = 44100;
  int64_t output_hop_length = 512;
};

// rvq_depth_decoder/config.json. BF16 on disk — forced regardless of `--dtype`
// (convert_minimax_music3_to_diffusers.py:214), so it is the one component
// whose narrow dtype is upstream's explicit choice rather than the default.
struct MiniMaxMusic3RvqDepthDecoderConfig {
  int64_t hidden_size = 4096;
  int64_t num_layers = 4;
  int64_t num_attention_heads = 16;
  int64_t intermediate_size = 6144;
  int64_t audio_vocab_size = 1024;
  int64_t num_codebooks = 8;
  int64_t max_position_embeddings = 16;

  // minimax_music3_rvq_depth_decoder.py:113,124 — the embedding table covers
  // the RESIDUAL codebooks only, and there is one head per residual codebook.
  int64_t residual_codebooks() const { return num_codebooks - 1; }
};

// vocoder/config.json. F32 on disk (spec section 2.1).
struct MiniMaxMusic3VocoderConfig {
  int64_t latent_channels = 128;
  int64_t decoder_input_dim = 1024;
  int64_t decoder_hidden_dim = 1536;
  std::vector<int64_t> upsampling_ratios{8, 8, 4, 2};
  int64_t sampling_rate = 44100;

  // minimax_music3_vocoder.py:110,115 — the 128 latent channels are folded into
  // TWO streams of 64, which is where the stereo pair comes from.
  int64_t stream_channels() const { return latent_channels / 2; }
  // 8*8*4*2 = 512 waveform samples per latent frame; 44100/512 = 86.133 Hz is
  // the latent frame rate the condition encoder sets (spec section 1.1).
  int64_t hop_length() const;
};

// language_model/config.json — a stock `Qwen3ForCausalLM`, retrained on a
// 200 000-entry music vocabulary. BF16 on disk (its own `"dtype": "bfloat16"`).
struct MiniMaxMusic3LanguageModelConfig {
  int64_t hidden_size = 4096;
  int64_t intermediate_size = 12288;
  int64_t num_hidden_layers = 36;
  int64_t num_attention_heads = 32;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t vocab_size = 200000;
  int64_t max_position_embeddings = 10240;
  double rope_theta = 1000000.0;
  double rms_norm_eps = 1e-6;
  bool tie_word_embeddings = false;
};

// scheduler/scheduler_config.json — `FlowMatchEulerDiscreteScheduler`.
//
// `num_train_timesteps: 1` is not a typo and is read rather than defaulted: the
// conversion script's own comment (convert_minimax_music3_to_diffusers.py:220)
// says it keeps `scheduler.timesteps` equal to the flow-matching time in [0, 1]
// that the transformer's Fourier embedding expects.
struct MiniMaxMusic3SchedulerConfig {
  int64_t num_train_timesteps = 1;
  double shift = 1.0;
  bool invert_sigmas = true;
  bool use_dynamic_shifting = false;
  std::string time_shift_type = "exponential";
};

struct MiniMaxMusic3Config {
  MiniMaxMusic3TransformerConfig transformer;
  MiniMaxMusic3ConditionEncoderConfig condition_encoder;
  MiniMaxMusic3RvqDepthDecoderConfig rvq_depth_decoder;
  MiniMaxMusic3VocoderConfig vocoder;
  MiniMaxMusic3LanguageModelConfig language_model;
  MiniMaxMusic3SchedulerConfig scheduler;
};

// Parse all six configs. Every `_class_name` / `architectures` entry is checked
// against the class this port implements and refused BY NAME on a mismatch: a
// config that silently deserializes to all-defaults is a wrong-shaped model
// with no error, which is exactly what .agents/porting-a-model.md section 1
// forbids.
MiniMaxMusic3Config MiniMaxMusic3LoadConfig(const MiniMaxMusic3Paths& paths);

// ---------------------------------------------------------------------------
// RUNTIME dtype — what each component will RUN in, not what its file stores
// ---------------------------------------------------------------------------

// The five weight-bearing components, in AR-then-acoustic order, so a refusal
// can name them in the order the pipeline uses them.
struct MiniMaxMusic3RuntimeDtypes {
  // The AR half. These three MUST agree: upstream never casts between them
  // (denoise.py:82 moves device only), so a disagreement is a torch type error
  // from inside `condition_embedder_minimax_music3.py:64`.
  std::string language_model;
  std::string rvq_depth_decoder;
  std::string condition_encoder;
  // The acoustic half. Each is reached through an explicit cast, so each may
  // differ from the AR half and from the other freely.
  std::string transformer;  // denoise.py:83 casts the condition into it
  std::string vocoder;      // decoders.py:84 casts the latents into it
};

enum class MiniMaxMusic3DtypePolicy {
  // GATED. AR half bf16, acoustic half fp32. The converter's default for the
  // DiT and the vocoder, and what SGLang-Omni states it runs.
  kBf16ArFp32Acoustic,
  // The dtypes the FILES carry. NOT RUNNABLE — kept selectable so the failure
  // stays reproducible against the oracle's `--dtype-policy on-disk`, never as
  // a default and never silently repaired.
  kAsStored,
};

MiniMaxMusic3RuntimeDtypes MiniMaxMusic3ResolveRuntimeDtypes(MiniMaxMusic3DtypePolicy policy);

// True when the three AR components agree, i.e. when the configuration can run
// at all. Never throws, so a caller can ask before committing to a load.
bool MiniMaxMusic3RuntimeDtypesAreRunnable(const MiniMaxMusic3RuntimeDtypes& dtypes);

// Enforce the invariant, or THROW naming all three AR components with their
// dtypes and the upstream line that would otherwise fail. Refusing here rather
// than at the first Conv1d is the whole point: upstream's message names a bias
// dtype, not which component disagreed with which.
void MiniMaxMusic3CheckRuntimeDtypes(const MiniMaxMusic3RuntimeDtypes& dtypes);

// The dtype each component's FILE stores. A fact about the artifact, and
// deliberately NOT a runtime policy — see the header note.
MiniMaxMusic3RuntimeDtypes MiniMaxMusic3OnDiskDtypes();

// ---------------------------------------------------------------------------
// What each component OWES, derived from its config
// ---------------------------------------------------------------------------

// One required tensor: the name it ships under on disk, the dtype spec section
// 2.1 fixes for its component, and the exact shape its config implies.
struct MiniMaxMusic3TensorSpec {
  std::string name;
  std::string dtype;  // the safetensors header spelling: "F32" / "BF16"
  std::vector<int64_t> shape;
};

// Enumerated by walking upstream's own `__init__`, module for module, so the
// enumeration is traceable rather than transcribed from a header dump:
//
//   transformer         transformer_minimax_music3.py:179-192, :134-138
//   condition_encoder   condition_embedder_minimax_music3.py:44-46
//   rvq_depth_decoder   minimax_music3_rvq_depth_decoder.py:113-124, :78-83
//   vocoder             minimax_music3_vocoder.py:88-98, :54-62, :41-44
//   language_model      transformers `Qwen3ForCausalLM`, with `q_norm`/`k_norm`
//                       per head_dim (Qwen3's per-head RMSNorm)
//
// Returned in a stable, sorted-by-name order so two callers cannot disagree.
std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3TransformerTensors(
    const MiniMaxMusic3TransformerConfig& config);
std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3ConditionEncoderTensors(
    const MiniMaxMusic3ConditionEncoderConfig& config);
std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3RvqDepthDecoderTensors(
    const MiniMaxMusic3RvqDepthDecoderConfig& config);
std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3VocoderTensors(
    const MiniMaxMusic3VocoderConfig& config);
std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3LanguageModelTensors(
    const MiniMaxMusic3LanguageModelConfig& config);

// Every component's enumeration, keyed by component directory name.
std::map<std::string, std::vector<MiniMaxMusic3TensorSpec>>
EnumerateMiniMaxMusic3Tensors(const MiniMaxMusic3Config& config);

// ---------------------------------------------------------------------------
// Accounting: enumerated == present, zero unaccounted
// ---------------------------------------------------------------------------

// One tensor as a FILE declares it. Header metadata only — this is what a
// manifest gate consumes and what a real `SafetensorsFile` is reduced to, so
// one accounting implementation serves both and they cannot disagree.
struct MiniMaxMusic3ManifestEntry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
};

// What the account examined. A gate that cannot say HOW MANY things it looked
// at has not reported, so these are returned rather than logged.
struct MiniMaxMusic3AccountReport {
  int64_t required = 0;
  int64_t present = 0;
  int64_t matched = 0;
};

// Account `present` against `required` for one component, or THROW naming the
// component and the first offending tensor.
//
// FOUR failure modes, each named separately, because collapsing them is how a
// refusal stops being evidence: a required tensor MISSING, a present tensor
// UNACCOUNTED for, a SHAPE mismatch (with both shapes printed), and a DTYPE
// mismatch (with both dtypes printed, and the spec section that fixes it).
//
// Returns counts on success; `matched == required == present` always holds when
// it returns, which is what makes a silent partial match impossible.
MiniMaxMusic3AccountReport MiniMaxMusic3AccountTensors(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const std::vector<MiniMaxMusic3ManifestEntry>& present);

// The same account against a real file's header.
MiniMaxMusic3AccountReport MiniMaxMusic3AccountFile(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const SafetensorsFile& file);

// Reduce a file's header to manifest entries. No payload is touched.
std::vector<MiniMaxMusic3ManifestEntry> MiniMaxMusic3ReadManifest(
    const SafetensorsFile& file);

// ---------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------

// One host-resident tensor with the dtype the FILE carries preserved. Nothing
// on this path widens or narrows: spec section 2.1's policy is upstream's, and
// a loader that "helpfully" converted would be making the measured change the
// spec reserves for its own evidence.
struct MiniMaxMusic3Tensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;

  int64_t numel() const;
};

struct MiniMaxMusic3ComponentWeights {
  std::string component;
  std::map<std::string, MiniMaxMusic3Tensor> tensors;
};

// Materialize one component, accounting it first. Every enumerated tensor must
// exist at its enumerated shape AND dtype before a byte is copied, so a missing
// tensor can never read as zeros.
MiniMaxMusic3ComponentWeights MiniMaxMusic3LoadComponent(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const SafetensorsFile& file);

// The vocoder, with every `weight_g`/`weight_v` pair FOLDED to a single
// `<module>.weight` and no `_g`/`_v` name surviving. F32 throughout — the
// checkpoint's own dtype (spec section 2.1), not a widening.
//
// `folded` counts the pairs collapsed; 20 for the shipped checkpoint, and it is
// returned rather than assumed so a file that lost a pair is visible.
struct MiniMaxMusic3VocoderWeights {
  std::map<std::string, std::vector<float>> tensors;
  std::map<std::string, std::vector<int64_t>> shapes;
  int64_t folded = 0;
};

MiniMaxMusic3VocoderWeights MiniMaxMusic3LoadVocoderWeights(
    const MiniMaxMusic3VocoderConfig& config, const SafetensorsFile& file);

// The `<module>` names whose weight this checkpoint stores weight-normed, in
// enumeration order. Derived from the same walk the enumeration uses, so the
// two cannot drift.
std::vector<std::string> MiniMaxMusic3WeightNormedModules(
    const MiniMaxMusic3VocoderConfig& config);

}  // namespace vllm
