// MiniMax-Music3 — the modular six-component checkpoint loader. See
// minimax_music3_loader.h for the decisions; this file is the mechanism.
//
// The enumerations below walk upstream's own `__init__` module for module, at
// diffusers PR #14456 head c6da9936, so a reviewer can diff them by eye against
// the Python rather than against a header dump. Every `file:line` cited is that
// revision.
#include "vllm/model_executor/models/minimax_music3_loader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

namespace fs = std::filesystem;

// The safetensors header spelling of each component's ON-DISK dtype, as
// convert_minimax_music3_to_diffusers.py writes it (`--dtype float32` default
// at :267, applied at :208-211; bf16 forced on the depth decoder at :214) and
// as the headers measure.
//
// THIS IS WHAT THE FILES STORE, NOT WHAT WILL RUN. The two differ here and the
// distinction is load-bearing -- see the header's dtype section. The on-disk
// set is not a runnable configuration: running the oracle showed the condition
// encoder must match the LANGUAGE MODEL's dtype, because upstream never casts
// on the way in (denoise.py:82 moves device only) and
// condition_embedder_minimax_music3.py:64 then feeds bf16 hidden states to an
// fp32 Conv1d. `MiniMaxMusic3ResolveRuntimeDtypes` owns the runtime answer.
//
// F32 on the ACOUSTIC half still needs no apology under AGENTS.md's too-wide
// rule: the oracle runs fp32 there too (the converter's default, and what
// SGLang-Omni states it runs). Narrowing that is a measured change with its own
// evidence, never a loader's initiative.
constexpr const char* kF32 = "F32";
constexpr const char* kBf16 = "BF16";

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

std::string JoinNames(const std::vector<std::string>& names, const char* separator) {
  std::string out;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i != 0) out += separator;
    out += names[i];
  }
  return out;
}

void SortByName(std::vector<MiniMaxMusic3TensorSpec>& specs) {
  std::sort(specs.begin(), specs.end(),
            [](const MiniMaxMusic3TensorSpec& a, const MiniMaxMusic3TensorSpec& b) {
              return a.name < b.name;
            });
}

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("minimax_music3: cannot open " + path);
  }
  try {
    nlohmann::json parsed;
    in >> parsed;
    return parsed;
  } catch (const std::exception& error) {
    throw std::runtime_error("minimax_music3: " + path + " is not valid JSON: " + error.what());
  }
}

// Read a required key, refusing BY NAME rather than defaulting. A config key
// that silently falls back to a class default is a DIFFERENT model built from
// the same tensors (.agents/porting-a-model.md section 1).
int64_t RequireInt(const nlohmann::json& object, const std::string& key,
                   const std::string& source) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    throw std::runtime_error("minimax_music3: " + source + " has no integer \"" + key + "\"");
  }
  return it->get<int64_t>();
}

double RequireNumber(const nlohmann::json& object, const std::string& key,
                     const std::string& source) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number()) {
    throw std::runtime_error("minimax_music3: " + source + " has no numeric \"" + key + "\"");
  }
  return it->get<double>();
}

bool RequireBool(const nlohmann::json& object, const std::string& key,
                 const std::string& source) {
  const auto it = object.find(key);
  if (it == object.end() || !it->is_boolean()) {
    throw std::runtime_error("minimax_music3: " + source + " has no boolean \"" + key + "\"");
  }
  return it->get<bool>();
}

// `_class_name` (diffusers) or `architectures[0]` (transformers) must be the
// class this port implements. A config for a DIFFERENT class that happens to
// carry compatible keys would otherwise bind silently.
void RequireClassName(const nlohmann::json& object, const std::string& expected,
                      const std::string& source) {
  const auto it = object.find("_class_name");
  if (it != object.end() && it->is_string()) {
    if (it->get<std::string>() != expected) {
      throw std::runtime_error("minimax_music3: " + source + " declares _class_name \"" +
                               it->get<std::string>() + "\", but this port implements \"" +
                               expected + "\"");
    }
    return;
  }
  const auto arch = object.find("architectures");
  if (arch != object.end() && arch->is_array() && !arch->empty() && arch->front().is_string()) {
    if (arch->front().get<std::string>() != expected) {
      throw std::runtime_error("minimax_music3: " + source + " declares architecture \"" +
                               arch->front().get<std::string>() +
                               "\", but this port implements \"" + expected + "\"");
    }
    return;
  }
  throw std::runtime_error(
      "minimax_music3: " + source +
      " declares neither _class_name nor architectures; refusing to assume \"" + expected + "\"");
}

// Shard list for a component, in the index's own order, deduplicated. Reading
// the index rather than globbing means a shard the index does not mention is
// never silently loaded and a shard it does mention is never silently skipped.
std::vector<std::string> ShardsFromIndex(const std::string& directory,
                                         const std::string& index_name,
                                         const std::string& single_name) {
  const fs::path index = fs::path(directory) / index_name;
  std::error_code ec;
  if (!fs::exists(index, ec)) {
    const fs::path single = fs::path(directory) / single_name;
    if (!fs::exists(single, ec)) {
      throw std::runtime_error("minimax_music3: " + directory + " has neither " + index_name +
                               " nor " + single_name);
    }
    return {single.string()};
  }
  const std::map<std::string, std::string> weight_map = LoadSafetensorsIndex(index.string());
  std::set<std::string> unique;
  for (const auto& [tensor, shard] : weight_map) {
    (void)tensor;
    unique.insert(shard);
  }
  std::vector<std::string> out;
  for (const std::string& shard : unique) {
    const fs::path path = fs::path(directory) / shard;
    if (!fs::exists(path, ec)) {
      throw std::runtime_error("minimax_music3: " + index.string() + " names shard " + shard +
                               ", which is not present in " + directory);
    }
    out.push_back(path.string());
  }
  return out;
}

// One weight-normed convolution: the module prefix and the shape of `weight_v`.
// `weight_g` is always [dim0, 1, 1] and the bias always [out_channels], so the
// pair and the bias are derived here rather than repeated at each call.
struct WeightNormedConv {
  std::string module;
  std::vector<int64_t> v_shape;  // [dim0, dim1, kernel]
  int64_t bias;                  // the CONV's output channels
};

// minimax_music3_vocoder.py:88-98, :54-62, :41-44 walked in construction order.
std::vector<WeightNormedConv> VocoderWeightNormedConvs(
    const MiniMaxMusic3VocoderConfig& config) {
  std::vector<WeightNormedConv> out;
  // :89 weight_norm(nn.Conv1d(decoder_input_dim, decoder_hidden_dim, 7)).
  out.push_back({"conv_in",
                 {config.decoder_hidden_dim, config.decoder_input_dim, 7},
                 config.decoder_hidden_dim});

  int64_t last_output = config.decoder_hidden_dim;
  for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
    const int64_t stride = config.upsampling_ratios[index];
    // :91-95 input_dim = hidden >> index, output_dim = hidden >> (index + 1).
    const int64_t input_dim = config.decoder_hidden_dim / (int64_t(1) << index);
    const int64_t output_dim = config.decoder_hidden_dim / (int64_t(1) << (index + 1));
    last_output = output_dim;
    const std::string block = "blocks." + std::to_string(index) + ".";
    // :55-59 weight_norm(nn.ConvTranspose1d(input_dim, output_dim, 2*stride)).
    //
    // NOTE THE AXIS. A ConvTranspose1d weight is [C_in, C_out, K], so dim 0 --
    // the axis torch's weight_norm reduces over -- is the INPUT channel here
    // and the OUTPUT channel for every plain Conv1d in this list. That is the
    // whole reason `vocoder1d::MaterializeWeightNorm` names its argument
    // `dim0`; folding these four over `out_channels` would be finite, correctly
    // shaped and wrong.
    out.push_back({block + "conv_t1", {input_dim, output_dim, 2 * stride}, output_dim});
    // :41-44, dilations 1/3/9 (:60-62). The dilation changes the PADDING, not
    // the stored shape, so it does not appear here.
    for (int unit = 1; unit <= 3; ++unit) {
      const std::string prefix = block + "res_unit" + std::to_string(unit) + ".";
      out.push_back({prefix + "conv1", {output_dim, output_dim, 7}, output_dim});
      out.push_back({prefix + "conv2", {output_dim, output_dim, 1}, output_dim});
    }
  }
  // :98 weight_norm(nn.Conv1d(output_dim, 1, 7)) -- the mono stream head; the
  // stereo pair comes from the two folded 64-channel streams (:110, :115), not
  // from two output channels.
  out.push_back({"conv_out", {1, last_output, 7}, 1});
  return out;
}

// Copy an F32 safetensors tensor out. Byte-wise, NOT a
// `reinterpret_cast<const float*>`: safetensors puts the payload straight after
// a JSON header of ARBITRARY length, so a tensor's first byte is only 4-byte
// aligned if the writer happened to pad, and the format does not require it.
// UBSan caught exactly that cast on this project once already
// (minimax_h3_vae_loader.cpp's ReadSafetensorF32 carries the same note).
std::vector<float> ReadF32(const std::string& component, const std::string& name,
                           const StTensor& tensor) {
  int64_t numel = 1;
  for (int64_t dim : tensor.shape) numel *= dim;
  if (tensor.dtype != kF32 || tensor.nbytes != static_cast<size_t>(numel) * 4) {
    throw std::runtime_error("minimax_music3: " + component + ": tensor " + name +
                             " is not an F32 tensor of its declared shape");
  }
  std::vector<float> out(static_cast<size_t>(numel));
  std::memcpy(out.data(), tensor.data, tensor.nbytes);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Small accessors
// ---------------------------------------------------------------------------

int64_t MiniMaxMusic3VocoderConfig::hop_length() const {
  // minimax_music3_vocoder.py:92-95 — one ConvTranspose1d per ratio, so the
  // total upsample is their product. 8*8*4*2 = 512, and 44100/512 = 86.133 Hz
  // is the latent frame rate the condition encoder sets (spec section 1.1).
  int64_t hop = 1;
  for (int64_t ratio : upsampling_ratios) hop *= ratio;
  return hop;
}

int64_t MiniMaxMusic3Tensor::numel() const {
  int64_t total = 1;
  for (int64_t dim : shape) total *= dim;
  return total;
}

// ---------------------------------------------------------------------------
// RUNTIME dtype
// ---------------------------------------------------------------------------

MiniMaxMusic3RuntimeDtypes MiniMaxMusic3OnDiskDtypes() {
  // Measured from the shipped headers; pinned in tests by
  // minimax_music3_manifest.inc, which the gate cross-checks against this.
  MiniMaxMusic3RuntimeDtypes out;
  out.language_model = kBf16;
  out.rvq_depth_decoder = kBf16;
  out.condition_encoder = kF32;
  out.transformer = kF32;
  out.vocoder = kF32;
  return out;
}

MiniMaxMusic3RuntimeDtypes MiniMaxMusic3ResolveRuntimeDtypes(MiniMaxMusic3DtypePolicy policy) {
  if (policy == MiniMaxMusic3DtypePolicy::kAsStored) {
    // Deliberately returned UNREPAIRED. It is not runnable, and the point of
    // keeping it is that the failure stays reproducible against the oracle's
    // own `--dtype-policy on-disk`; silently promoting the condition encoder
    // to bf16 here would hide exactly the finding this arm exists to preserve.
    return MiniMaxMusic3OnDiskDtypes();
  }
  // kBf16ArFp32Acoustic: the gated configuration. The AR half runs at the
  // language model's dtype because nothing casts between its stages; the
  // acoustic half is fp32, which is the converter's default for the DiT and the
  // vocoder and what SGLang-Omni states it runs.
  MiniMaxMusic3RuntimeDtypes out;
  out.language_model = kBf16;
  out.rvq_depth_decoder = kBf16;
  out.condition_encoder = kBf16;
  out.transformer = kF32;
  out.vocoder = kF32;
  return out;
}

bool MiniMaxMusic3RuntimeDtypesAreRunnable(const MiniMaxMusic3RuntimeDtypes& dtypes) {
  return dtypes.language_model == dtypes.rvq_depth_decoder &&
         dtypes.language_model == dtypes.condition_encoder;
}

void MiniMaxMusic3CheckRuntimeDtypes(const MiniMaxMusic3RuntimeDtypes& dtypes) {
  if (MiniMaxMusic3RuntimeDtypesAreRunnable(dtypes)) return;
  // Name all three and their dtypes. Upstream's own failure names a BIAS dtype
  // from inside a Conv1d and never says which component disagreed with which,
  // which is the whole reason this refusal exists here instead.
  throw std::runtime_error(
      std::string("minimax_music3: this dtype configuration cannot run. The autoregressive half "
                  "must share ONE dtype -- language_model=") +
      dtypes.language_model + ", rvq_depth_decoder=" + dtypes.rvq_depth_decoder +
      ", condition_encoder=" + dtypes.condition_encoder +
      " -- because upstream casts only on the way OUT of the condition encoder "
      "(denoise.py:83) and into the vocoder (decoders.py:84), never on the way in: "
      "denoise.py:82 hands the language model's hidden states over with a device move and no "
      "dtype move. A mismatch therefore fails inside "
      "condition_embedder_minimax_music3.py:64 as \"Input type (c10::BFloat16) and bias type "
      "(float) should be the same\". The gated configuration is bf16 AR half with an fp32 "
      "acoustic half (MiniMaxMusic3DtypePolicy::kBf16ArFp32Acoustic). NOTE: the checkpoint's "
      "ON-DISK dtypes are NOT this set and are not runnable as stored -- "
      "MiniMaxMusic3DtypePolicy::kAsStored reproduces that failure on purpose.");
}

// ---------------------------------------------------------------------------
// Which packaging is on disk
// ---------------------------------------------------------------------------

bool MiniMaxMusic3IsNativeArm(const std::string& root) {
  std::error_code ec;
  const fs::path base(root);
  return fs::exists(base / kMusic3NativeDitFile, ec) ||
         fs::exists(base / kMusic3NativeVaeFile, ec) ||
         fs::is_directory(base / kMusic3NativeQwenDir, ec);
}

MiniMaxMusic3Paths MiniMaxMusic3ResolveCheckpoint(const std::string& root) {
  std::error_code ec;
  const fs::path base(root);
  if (!fs::is_directory(base, ec)) {
    throw std::runtime_error("minimax_music3: " + root + " is not a directory");
  }

  // The five weight-bearing components plus the two config-only ones. Their
  // absence is what the refusals below enumerate.
  const std::vector<std::string> weight_components{
      "transformer", "condition_encoder", "rvq_depth_decoder", "vocoder", "language_model"};
  std::vector<std::string> missing;
  for (const std::string& component : weight_components) {
    if (!fs::is_directory(base / component, ec)) missing.push_back(component);
  }
  if (!fs::is_directory(base / "scheduler", ec)) missing.push_back("scheduler");
  if (!fs::is_directory(base / "tokenizer", ec)) missing.push_back("tokenizer");
  if (!fs::exists(base / "modular_model_index.json", ec)) {
    missing.push_back("modular_model_index.json");
  }

  if (MiniMaxMusic3IsNativeArm(root)) {
    // THE REFUSAL THAT MATTERS. A native tree holds every weight this port
    // needs, in a layout nothing here reads, so it is the one input that would
    // otherwise LOOK loadable. Name the arm, name its markers, name what the
    // diffusers arm wants, and say which one is supported -- spec section 2.
    std::vector<std::string> markers;
    if (fs::exists(base / kMusic3NativeDitFile, ec)) markers.push_back(kMusic3NativeDitFile);
    if (fs::exists(base / kMusic3NativeVaeFile, ec)) markers.push_back(kMusic3NativeVaeFile);
    if (fs::is_directory(base / kMusic3NativeQwenDir, ec)) {
      markers.push_back(std::string(kMusic3NativeQwenDir) + "/");
    }
    throw std::runtime_error(
        "minimax_music3: " + root + " is the NATIVE arm of MiniMaxAI/MiniMax-Music3 (found " +
        JoinNames(markers, ", ") + "; the full native layout is " + kMusic3NativeQwenDir + "/" +
        kMusic3NativeQwenDir + "/, " + kMusic3NativeDitFile + " and " + kMusic3NativeVaeFile +
        "). Only the diffusers arm is supported: it wants the components transformer, "
        "condition_encoder, rvq_depth_decoder, vocoder, language_model, scheduler and tokenizer "
        "beside a modular_model_index.json, and this tree is missing " +
        (missing.empty() ? std::string("none of them") : JoinNames(missing, ", ")) +
        ". Convert it with diffusers' scripts/convert_minimax_music3_to_diffusers.py, or point "
        "this loader at the diffusers-arm checkpoint. Loading the native arm directly is owed and "
        "recorded in .agents/specs/minimax-music3.md section 2 -- it is refused here rather than "
        "mis-loaded.");
  }

  if (!missing.empty()) {
    throw std::runtime_error(
        "minimax_music3: " + root +
        " is not a diffusers-arm MiniMax-Music3 checkpoint; it is missing " +
        JoinNames(missing, ", ") +
        " (expected a modular_model_index.json beside the component directories).");
  }

  MiniMaxMusic3Paths paths;
  paths.root = root;
  paths.modular_index = (base / "modular_model_index.json").string();
  paths.transformer_dir = (base / "transformer").string();
  paths.condition_encoder_dir = (base / "condition_encoder").string();
  paths.rvq_depth_decoder_dir = (base / "rvq_depth_decoder").string();
  paths.vocoder_dir = (base / "vocoder").string();
  paths.language_model_dir = (base / "language_model").string();
  paths.scheduler_config = (base / "scheduler" / "scheduler_config.json").string();
  paths.tokenizer_dir = (base / "tokenizer").string();
  paths.transformer_shards =
      ShardsFromIndex(paths.transformer_dir, "diffusion_pytorch_model.safetensors.index.json",
                      "diffusion_pytorch_model.safetensors");
  paths.language_model_shards =
      ShardsFromIndex(paths.language_model_dir, "model.safetensors.index.json",
                      "model.safetensors");
  return paths;
}

// ---------------------------------------------------------------------------
// Configs
// ---------------------------------------------------------------------------

MiniMaxMusic3Config MiniMaxMusic3LoadConfig(const MiniMaxMusic3Paths& paths) {
  MiniMaxMusic3Config out;

  {
    const std::string source = paths.transformer_dir + "/config.json";
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "MiniMaxMusic3Transformer1DModel", source);
    MiniMaxMusic3TransformerConfig& config = out.transformer;
    config.in_channels = RequireInt(json, "in_channels", source);
    config.condition_dim = RequireInt(json, "condition_dim", source);
    config.num_layers = RequireInt(json, "num_layers", source);
    config.num_attention_heads = RequireInt(json, "num_attention_heads", source);
    config.attention_head_dim = RequireInt(json, "attention_head_dim", source);
    config.ff_inner_dim = RequireInt(json, "ff_inner_dim", source);
    // `rotary_dim` moves every RoPE angle and no tensor shape, so it is READ
    // rather than defaulted: a wrong value renders a plausible, wrong song and
    // no shape gate can see it (transformer_minimax_music3.py:42-56).
    config.rotary_dim = RequireInt(json, "rotary_dim", source);
    config.fourier_embedding_dim = RequireInt(json, "fourier_embedding_dim", source);
  }
  {
    const std::string source = paths.condition_encoder_dir + "/config.json";
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "MiniMaxMusic3ConditionEncoder", source);
    MiniMaxMusic3ConditionEncoderConfig& config = out.condition_encoder;
    config.condition_hidden_dim = RequireInt(json, "condition_hidden_dim", source);
    config.num_condition_layers = RequireInt(json, "num_condition_layers", source);
    config.out_dim = RequireInt(json, "out_dim", source);
    // The four rate keys set the LM-frame -> latent-frame resample ratio
    // (condition_embedder_minimax_music3.py:64-73). Shape-invisible, so read.
    config.input_sampling_rate = RequireInt(json, "input_sampling_rate", source);
    config.input_hop_length = RequireInt(json, "input_hop_length", source);
    config.output_sampling_rate = RequireInt(json, "output_sampling_rate", source);
    config.output_hop_length = RequireInt(json, "output_hop_length", source);
  }
  {
    const std::string source = paths.rvq_depth_decoder_dir + "/config.json";
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "MiniMaxMusic3RVQDepthDecoder", source);
    MiniMaxMusic3RvqDepthDecoderConfig& config = out.rvq_depth_decoder;
    config.hidden_size = RequireInt(json, "hidden_size", source);
    config.num_layers = RequireInt(json, "num_layers", source);
    config.num_attention_heads = RequireInt(json, "num_attention_heads", source);
    config.intermediate_size = RequireInt(json, "intermediate_size", source);
    config.audio_vocab_size = RequireInt(json, "audio_vocab_size", source);
    config.num_codebooks = RequireInt(json, "num_codebooks", source);
    config.max_position_embeddings = RequireInt(json, "max_position_embeddings", source);
  }
  {
    const std::string source = paths.vocoder_dir + "/config.json";
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "MiniMaxMusic3Vocoder", source);
    MiniMaxMusic3VocoderConfig& config = out.vocoder;
    config.latent_channels = RequireInt(json, "latent_channels", source);
    config.decoder_input_dim = RequireInt(json, "decoder_input_dim", source);
    config.decoder_hidden_dim = RequireInt(json, "decoder_hidden_dim", source);
    config.sampling_rate = RequireInt(json, "sampling_rate", source);
    const auto ratios = json.find("upsampling_ratios");
    if (ratios == json.end() || !ratios->is_array() || ratios->empty()) {
      throw std::runtime_error("minimax_music3: " + source +
                               " has no non-empty \"upsampling_ratios\" array");
    }
    config.upsampling_ratios.clear();
    for (const nlohmann::json& ratio : *ratios) {
      if (!ratio.is_number_integer()) {
        throw std::runtime_error("minimax_music3: " + source +
                                 " has a non-integer entry in \"upsampling_ratios\"");
      }
      config.upsampling_ratios.push_back(ratio.get<int64_t>());
    }
  }
  {
    const std::string source = paths.language_model_dir + "/config.json";
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "Qwen3ForCausalLM", source);
    MiniMaxMusic3LanguageModelConfig& config = out.language_model;
    config.hidden_size = RequireInt(json, "hidden_size", source);
    config.intermediate_size = RequireInt(json, "intermediate_size", source);
    config.num_hidden_layers = RequireInt(json, "num_hidden_layers", source);
    config.num_attention_heads = RequireInt(json, "num_attention_heads", source);
    config.num_key_value_heads = RequireInt(json, "num_key_value_heads", source);
    config.head_dim = RequireInt(json, "head_dim", source);
    config.vocab_size = RequireInt(json, "vocab_size", source);
    config.max_position_embeddings = RequireInt(json, "max_position_embeddings", source);
    config.rms_norm_eps = RequireNumber(json, "rms_norm_eps", source);
    // `tie_word_embeddings` is TRI-STATE in the wild and decides whether
    // `lm_head.weight` exists at all. The released config states it, so it is
    // required rather than defaulted (.agents/porting-a-model.md section 1).
    config.tie_word_embeddings = RequireBool(json, "tie_word_embeddings", source);
    // transformers 5.x nests rope under `rope_parameters`; earlier layouts put
    // `rope_theta` at the top level. Both are normalized, and a config with
    // NEITHER is refused rather than silently given 10000.0 -- which would move
    // every position embedding in a 10240-token context.
    const auto nested = json.find("rope_parameters");
    if (nested != json.end() && nested->is_object() && nested->contains("rope_theta")) {
      config.rope_theta = RequireNumber(*nested, "rope_theta", source + " rope_parameters");
    } else {
      config.rope_theta = RequireNumber(json, "rope_theta", source);
    }
  }
  {
    const std::string source = paths.scheduler_config;
    const nlohmann::json json = ReadJson(source);
    RequireClassName(json, "FlowMatchEulerDiscreteScheduler", source);
    MiniMaxMusic3SchedulerConfig& config = out.scheduler;
    config.num_train_timesteps = RequireInt(json, "num_train_timesteps", source);
    config.shift = RequireNumber(json, "shift", source);
    config.invert_sigmas = RequireBool(json, "invert_sigmas", source);
    config.use_dynamic_shifting = RequireBool(json, "use_dynamic_shifting", source);
    const auto shift_type = json.find("time_shift_type");
    if (shift_type == json.end() || !shift_type->is_string()) {
      throw std::runtime_error("minimax_music3: " + source + " has no string \"time_shift_type\"");
    }
    config.time_shift_type = shift_type->get<std::string>();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Enumeration — upstream's `__init__`, module for module
// ---------------------------------------------------------------------------

std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3TransformerTensors(
    const MiniMaxMusic3TransformerConfig& config) {
  const int64_t inner = config.inner_dim();
  const int64_t concat = config.concat_channels();
  // transformer_minimax_music3.py:119-122 — the attention's own inner width.
  const int64_t attn_inner = config.num_attention_heads * config.attention_head_dim;
  std::vector<MiniMaxMusic3TensorSpec> out;

  // :179 MiniMaxMusic3FourierEmbedding(fourier_embedding_dim) -> :35
  // `nn.Parameter(torch.randn(embedding_dim // 2, 1))`.
  out.push_back({"time_proj.weight", kF32, {config.fourier_embedding_dim / 2, 1}});
  // :180 TimestepEmbedding(fourier_embedding_dim, inner_dim), both linears biased.
  out.push_back({"time_embed.linear_1.weight", kF32, {inner, config.fourier_embedding_dim}});
  out.push_back({"time_embed.linear_1.bias", kF32, {inner}});
  out.push_back({"time_embed.linear_2.weight", kF32, {inner, inner}});
  out.push_back({"time_embed.linear_2.bias", kF32, {inner}});
  // :182 nn.Conv1d(concat_channels, concat_channels, 1, bias=False).
  out.push_back({"preprocess_conv.weight", kF32, {concat, concat, 1}});
  // :183 nn.Linear(concat_channels, inner_dim, bias=False).
  out.push_back({"proj_in.weight", kF32, {inner, concat}});

  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = "transformer_blocks." + std::to_string(layer) + ".";
    // :134 nn.LayerNorm(dim) -- weight AND bias, unlike the RMSNorms elsewhere
    // in this model. A LayerNorm read as an RMSNorm silently drops the bias.
    out.push_back({prefix + "norm1.weight", kF32, {inner}});
    out.push_back({prefix + "norm1.bias", kF32, {inner}});
    // :119-122 to_q/to_k/to_v, and to_out as a ModuleList whose [0] is the
    // linear and whose [1] is a Dropout -- hence the `.0.` in the NAME.
    out.push_back({prefix + "attn.to_q.weight", kF32, {attn_inner, inner}});
    out.push_back({prefix + "attn.to_k.weight", kF32, {attn_inner, inner}});
    out.push_back({prefix + "attn.to_v.weight", kF32, {attn_inner, inner}});
    out.push_back({prefix + "attn.to_out.0.weight", kF32, {inner, attn_inner}});
    out.push_back({prefix + "norm2.weight", kF32, {inner}});
    out.push_back({prefix + "norm2.bias", kF32, {inner}});
    // :137 nn.Linear(dim, ff_inner_dim * 2) -- the GATED feed-forward, so the
    // stored width is TWICE ff_inner_dim and :141 chunks it in two.
    out.push_back({prefix + "ff_in.weight", kF32, {config.ff_inner_dim * 2, inner}});
    out.push_back({prefix + "ff_in.bias", kF32, {config.ff_inner_dim * 2}});
    // :138 nn.Linear(ff_inner_dim, dim).
    out.push_back({prefix + "ff_out.weight", kF32, {inner, config.ff_inner_dim}});
    out.push_back({prefix + "ff_out.bias", kF32, {inner}});
  }

  // :191 nn.Linear(inner_dim, in_channels, bias=False).
  out.push_back({"proj_out.weight", kF32, {config.in_channels, inner}});
  // :192 nn.Conv1d(in_channels, in_channels, 1, bias=False).
  out.push_back({"postprocess_conv.weight", kF32, {config.in_channels, config.in_channels, 1}});

  SortByName(out);
  return out;
}

std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3ConditionEncoderTensors(
    const MiniMaxMusic3ConditionEncoderConfig& config) {
  // condition_embedder_minimax_music3.py:44-46. FOUR tensors: the module is a
  // learned weighted MIX over `num_condition_layers` LM hidden layers plus one
  // Conv1d, not an encoder tower.
  std::vector<MiniMaxMusic3TensorSpec> out{
      {"layer_weight_logits", kF32, {config.num_condition_layers}},
      {"layer_scale", kF32, {1}},
      {"proj.weight", kF32, {config.out_dim, config.condition_hidden_dim, 3}},
      {"proj.bias", kF32, {config.out_dim}},
  };
  SortByName(out);
  return out;
}

std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3RvqDepthDecoderTensors(
    const MiniMaxMusic3RvqDepthDecoderConfig& config) {
  const int64_t hidden = config.hidden_size;
  std::vector<MiniMaxMusic3TensorSpec> out;
  // :113 nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size) --
  // the table covers the RESIDUAL codebooks only, so a reader that used
  // `num_codebooks` would allocate one codebook too many and mis-index every
  // frame after the first.
  out.push_back({"audio_embeddings.weight", kBf16,
                 {config.audio_vocab_size * config.residual_codebooks(), hidden}});
  out.push_back({"projection.weight", kBf16, {hidden, hidden}});                             // :114
  out.push_back({"pos_embedding.weight", kBf16, {config.max_position_embeddings, hidden}});  // :115

  for (int64_t layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    // :78-83 -- RMSNorm (weight only, no bias) and four bias-free projections.
    out.push_back({prefix + "input_layernorm.weight", kBf16, {hidden}});
    out.push_back({prefix + "attn.to_q.weight", kBf16, {hidden, hidden}});
    out.push_back({prefix + "attn.to_k.weight", kBf16, {hidden, hidden}});
    out.push_back({prefix + "attn.to_v.weight", kBf16, {hidden, hidden}});
    out.push_back({prefix + "attn.to_out.weight", kBf16, {hidden, hidden}});
    out.push_back({prefix + "post_attention_layernorm.weight", kBf16, {hidden}});
    out.push_back({prefix + "gate_proj.weight", kBf16, {config.intermediate_size, hidden}});
    out.push_back({prefix + "up_proj.weight", kBf16, {config.intermediate_size, hidden}});
    out.push_back({prefix + "down_proj.weight", kBf16, {hidden, config.intermediate_size}});
  }

  out.push_back({"norm.weight", kBf16, {hidden}});  // :122
  // :123-124 -- one head per RESIDUAL codebook (c1..c7), not per codebook.
  for (int64_t head = 0; head < config.residual_codebooks(); ++head) {
    out.push_back({"audio_heads." + std::to_string(head) + ".weight", kBf16,
                   {config.audio_vocab_size, hidden}});
  }
  SortByName(out);
  return out;
}

std::vector<std::string> MiniMaxMusic3WeightNormedModules(
    const MiniMaxMusic3VocoderConfig& config) {
  std::vector<std::string> out;
  for (const WeightNormedConv& conv : VocoderWeightNormedConvs(config)) {
    out.push_back(conv.module);
  }
  return out;
}

std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3VocoderTensors(
    const MiniMaxMusic3VocoderConfig& config) {
  std::vector<MiniMaxMusic3TensorSpec> out;
  // :88 nn.Conv1d(latent_channels // 2, decoder_input_dim, 1) -- the ONE
  // convolution upstream does NOT weight-norm, and the one whose input width is
  // the FOLDED stream (64), not the 128 latent channels.
  out.push_back(
      {"dec_in_proj.weight", kF32, {config.decoder_input_dim, config.stream_channels(), 1}});
  out.push_back({"dec_in_proj.bias", kF32, {config.decoder_input_dim}});

  for (const WeightNormedConv& conv : VocoderWeightNormedConvs(config)) {
    out.push_back({conv.module + ".weight_g", kF32, {conv.v_shape[0], 1, 1}});
    out.push_back({conv.module + ".weight_v", kF32, conv.v_shape});
    out.push_back({conv.module + ".bias", kF32, {conv.bias}});
  }

  // The snakes. :28 nn.Parameter(torch.ones(1, channels, 1)) throughout.
  int64_t last_output = config.decoder_hidden_dim;
  for (size_t index = 0; index < config.upsampling_ratios.size(); ++index) {
    const int64_t input_dim = config.decoder_hidden_dim / (int64_t(1) << index);
    const int64_t output_dim = config.decoder_hidden_dim / (int64_t(1) << (index + 1));
    last_output = output_dim;
    const std::string block = "blocks." + std::to_string(index) + ".";
    out.push_back({block + "snake1.alpha", kF32, {1, input_dim, 1}});  // :54
    for (int unit = 1; unit <= 3; ++unit) {
      const std::string prefix = block + "res_unit" + std::to_string(unit) + ".";
      out.push_back({prefix + "snake1.alpha", kF32, {1, output_dim, 1}});  // :41
      out.push_back({prefix + "snake2.alpha", kF32, {1, output_dim, 1}});  // :43
    }
  }
  out.push_back({"snake_out.alpha", kF32, {1, last_output, 1}});  // :97

  SortByName(out);
  return out;
}

std::vector<MiniMaxMusic3TensorSpec> EnumerateMiniMaxMusic3LanguageModelTensors(
    const MiniMaxMusic3LanguageModelConfig& config) {
  const int64_t hidden = config.hidden_size;
  const int64_t q_dim = config.num_attention_heads * config.head_dim;
  const int64_t kv_dim = config.num_key_value_heads * config.head_dim;
  std::vector<MiniMaxMusic3TensorSpec> out;

  out.push_back({"model.embed_tokens.weight", kBf16, {config.vocab_size, hidden}});
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    out.push_back({prefix + "input_layernorm.weight", kBf16, {hidden}});
    out.push_back({prefix + "post_attention_layernorm.weight", kBf16, {hidden}});
    out.push_back({prefix + "self_attn.q_proj.weight", kBf16, {q_dim, hidden}});
    out.push_back({prefix + "self_attn.k_proj.weight", kBf16, {kv_dim, hidden}});
    out.push_back({prefix + "self_attn.v_proj.weight", kBf16, {kv_dim, hidden}});
    out.push_back({prefix + "self_attn.o_proj.weight", kBf16, {hidden, q_dim}});
    // Qwen3's PER-HEAD RMSNorm on q and k: [head_dim], not [hidden]. A reader
    // that assumed [hidden] would refuse a correct checkpoint, and one that
    // skipped them would drop a normalization the argmax often survives.
    out.push_back({prefix + "self_attn.q_norm.weight", kBf16, {config.head_dim}});
    out.push_back({prefix + "self_attn.k_norm.weight", kBf16, {config.head_dim}});
    out.push_back({prefix + "mlp.gate_proj.weight", kBf16, {config.intermediate_size, hidden}});
    out.push_back({prefix + "mlp.up_proj.weight", kBf16, {config.intermediate_size, hidden}});
    out.push_back({prefix + "mlp.down_proj.weight", kBf16, {hidden, config.intermediate_size}});
  }
  out.push_back({"model.norm.weight", kBf16, {hidden}});
  // The released config sets `tie_word_embeddings: false`, so `lm_head.weight`
  // is a separate 200000 x 4096 tensor -- 1.6 GB of it. Tied, it would not be
  // in the file at all, which is why the flag decides the ENUMERATION and not
  // just a pointer.
  if (!config.tie_word_embeddings) {
    out.push_back({"lm_head.weight", kBf16, {config.vocab_size, hidden}});
  }
  SortByName(out);
  return out;
}

std::map<std::string, std::vector<MiniMaxMusic3TensorSpec>> EnumerateMiniMaxMusic3Tensors(
    const MiniMaxMusic3Config& config) {
  return {
      {"transformer", EnumerateMiniMaxMusic3TransformerTensors(config.transformer)},
      {"condition_encoder",
       EnumerateMiniMaxMusic3ConditionEncoderTensors(config.condition_encoder)},
      {"rvq_depth_decoder",
       EnumerateMiniMaxMusic3RvqDepthDecoderTensors(config.rvq_depth_decoder)},
      {"vocoder", EnumerateMiniMaxMusic3VocoderTensors(config.vocoder)},
      {"language_model", EnumerateMiniMaxMusic3LanguageModelTensors(config.language_model)},
  };
}

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

MiniMaxMusic3AccountReport MiniMaxMusic3AccountTensors(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const std::vector<MiniMaxMusic3ManifestEntry>& present) {
  MiniMaxMusic3AccountReport report;
  report.required = static_cast<int64_t>(required.size());
  report.present = static_cast<int64_t>(present.size());

  std::map<std::string, const MiniMaxMusic3ManifestEntry*> by_name;
  for (const MiniMaxMusic3ManifestEntry& entry : present) {
    const bool inserted = by_name.emplace(entry.name, &entry).second;
    if (!inserted) {
      throw std::runtime_error("minimax_music3: " + component + ": tensor " + entry.name +
                               " appears more than once across the component's shards");
    }
  }

  std::set<std::string> accounted;
  for (const MiniMaxMusic3TensorSpec& spec : required) {
    const auto it = by_name.find(spec.name);
    if (it == by_name.end()) {
      throw std::runtime_error("minimax_music3: " + component + ": required tensor " + spec.name +
                               " " + ShapeToString(spec.shape) + " " + spec.dtype +
                               " is MISSING from the checkpoint");
    }
    const MiniMaxMusic3ManifestEntry& entry = *it->second;
    if (entry.shape != spec.shape) {
      throw std::runtime_error(
          "minimax_music3: " + component + ": tensor " + spec.name + " has shape " +
          ShapeToString(entry.shape) + " but the config implies " + ShapeToString(spec.shape) +
          "; refusing rather than binding a different model");
    }
    if (entry.dtype != spec.dtype) {
      // A dtype that is too WIDE is numerically correct, so no token or golden
      // gate can see it (AGENTS.md, .agents/porting.md "Mirror the memory
      // format"). It is therefore refused structurally, here.
      throw std::runtime_error(
          "minimax_music3: " + component + ": tensor " + spec.name + " has dtype " + entry.dtype +
          " but this component is " + spec.dtype +
          " (.agents/specs/minimax-music3.md section 2.1, "
          "convert_minimax_music3_to_diffusers.py:208-211,214,267); a narrowing or widening is a "
          "MEASURED change, never a default");
    }
    accounted.insert(spec.name);
    ++report.matched;
  }

  for (const MiniMaxMusic3ManifestEntry& entry : present) {
    if (accounted.count(entry.name) == 0) {
      throw std::runtime_error("minimax_music3: " + component + ": tensor " + entry.name + " " +
                               ShapeToString(entry.shape) +
                               " is present in the checkpoint and UNACCOUNTED for by this port");
    }
  }

  VT_CHECK(report.matched == report.required && report.present == report.required,
           "minimax_music3: account returned without a total match");
  return report;
}

std::vector<MiniMaxMusic3ManifestEntry> MiniMaxMusic3ReadManifest(const SafetensorsFile& file) {
  std::vector<MiniMaxMusic3ManifestEntry> out;
  for (const std::string& name : file.Names()) {
    const StTensor& tensor = file.Get(name);
    out.push_back({name, tensor.dtype, tensor.shape});
  }
  return out;
}

MiniMaxMusic3AccountReport MiniMaxMusic3AccountFile(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const SafetensorsFile& file) {
  return MiniMaxMusic3AccountTensors(component, required, MiniMaxMusic3ReadManifest(file));
}

// ---------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------

MiniMaxMusic3ComponentWeights MiniMaxMusic3LoadComponent(
    const std::string& component, const std::vector<MiniMaxMusic3TensorSpec>& required,
    const SafetensorsFile& file) {
  // Account FIRST. Every enumerated tensor exists at its enumerated shape and
  // dtype before a byte is copied, so a missing tensor can never read as zeros.
  MiniMaxMusic3AccountFile(component, required, file);

  MiniMaxMusic3ComponentWeights out;
  out.component = component;
  for (const MiniMaxMusic3TensorSpec& spec : required) {
    const StTensor& source = file.Get(spec.name);
    MiniMaxMusic3Tensor tensor;
    tensor.dtype = source.dtype;  // preserved; see spec section 2.1
    tensor.shape = spec.shape;
    const size_t width = (spec.dtype == std::string(kF32)) ? 4 : 2;
    const size_t expected = static_cast<size_t>(tensor.numel()) * width;
    if (source.nbytes != expected) {
      throw std::runtime_error("minimax_music3: " + component + ": tensor " + spec.name +
                               " spans " + std::to_string(source.nbytes) +
                               " bytes but its shape " + ShapeToString(spec.shape) +
                               " and dtype " + spec.dtype + " imply " + std::to_string(expected));
    }
    tensor.bytes.resize(source.nbytes);
    std::memcpy(tensor.bytes.data(), source.data, source.nbytes);
    MaybeReleaseSourcePages(source.data, source.nbytes);
    out.tensors.emplace(spec.name, std::move(tensor));
  }
  return out;
}

MiniMaxMusic3VocoderWeights MiniMaxMusic3LoadVocoderWeights(
    const MiniMaxMusic3VocoderConfig& config, const SafetensorsFile& file) {
  const std::vector<MiniMaxMusic3TensorSpec> required =
      EnumerateMiniMaxMusic3VocoderTensors(config);
  MiniMaxMusic3AccountFile("vocoder", required, file);

  const std::vector<WeightNormedConv> convs = VocoderWeightNormedConvs(config);
  MiniMaxMusic3VocoderWeights out;
  std::set<std::string> consumed;

  // Fold every weight-normed pair to ONE weight, so the decoder that consumes
  // this never sees `_g` / `_v` and cannot read the direction as the weight.
  for (const WeightNormedConv& conv : convs) {
    const std::string g_name = conv.module + ".weight_g";
    const std::string v_name = conv.module + ".weight_v";
    const std::vector<float> g = ReadF32("vocoder", g_name, file.Get(g_name));
    const std::vector<float> v = ReadF32("vocoder", v_name, file.Get(v_name));
    // dim0 of `weight_v`: the OUTPUT channel for a Conv1d and the INPUT channel
    // for the ConvTranspose1d, which is the axis torch reduces over either way.
    const int64_t dim0 = conv.v_shape[0];
    out.tensors[conv.module + ".weight"] = vocoder1d::MaterializeWeightNorm(g, v, dim0);
    out.shapes[conv.module + ".weight"] = conv.v_shape;
    consumed.insert(g_name);
    consumed.insert(v_name);
    ++out.folded;
  }

  // Everything else verbatim: the biases, the snake alphas, and the one
  // un-weight-normed convolution.
  for (const MiniMaxMusic3TensorSpec& spec : required) {
    if (consumed.count(spec.name) != 0) continue;
    out.tensors[spec.name] = ReadF32("vocoder", spec.name, file.Get(spec.name));
    out.shapes[spec.name] = spec.shape;
  }

  VT_CHECK(out.folded == static_cast<int64_t>(convs.size()),
           "minimax_music3 vocoder: folded fewer pairs than the enumeration declares");
  return out;
}

}  // namespace vllm
