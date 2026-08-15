// MiniMax-Music3 — the quantized arms: DETECTION, REFUSAL, and the one arm that
// is implemented (the RVQ depth decoder's GGUF Q4_K). See
// minimax_music3_quant.h for the decisions and for the survey that establishes
// which of these formats actually ships for this model; this file is the
// mechanism.
#include "vllm/model_executor/models/minimax_music3_quant.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace vllm {
namespace {

namespace fs = std::filesystem;

// How deep the tree walk goes. `diffusion_models/x.gguf` and
// `text_encoders/y.gguf` are depth 1; a component directory's shard is depth 1;
// nothing this port cares about is deeper, and an unbounded walk over a staged
// 28.5 GB checkpoint is a cost with no payer.
constexpr int kMaxTreeDepth = 2;

// At most this many pieces of evidence are quoted. All of them would make a
// 500-file refusal unreadable; one would hide which components are affected.
constexpr size_t kMaxEvidence = 6;

bool EndsWith(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string Join(const std::vector<std::string>& parts, int64_t total) {
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) out += ", ";
    out += parts[i];
  }
  if (total > static_cast<int64_t>(parts.size())) {
    out += " (+" + std::to_string(total - static_cast<int64_t>(parts.size())) + " more)";
  }
  return out;
}

// What each format would COST to implement, named as a concrete missing piece
// rather than as "unsupported". A refusal that does not say what is missing
// makes the next person re-derive it.
const char* MissingPieceFor(MiniMaxMusic3QuantFormat format) {
  switch (format) {
    case MiniMaxMusic3QuantFormat::kNone:
      return "";
    case MiniMaxMusic3QuantFormat::kGguf:
      // Measured, not assumed: the headers of ten published Music3 GGUFs were
      // read by HTTP range request on 2026-08-14 (spec section 9). What they
      // show is that "the GGUF arm" is THREE arms, and a loader that assumed
      // one would silently mis-bind the other two.
      return "a GGUF arm needs a name map, the GGUF-vs-torch dim reversal, a "
             "geometry source, and k-quant dequantization routed through "
             "vllm/model_executor/model_loader/gguf_dequant.h -- never a parallel "
             "path (minimax_h3_gguf.cpp is the in-tree precedent). Note that the "
             "published Music3 GGUFs are THREE MUTUALLY INCOMPATIBLE LINEAGES and "
             "`general.architecture` cannot separate them -- it reads 'audiocpp', "
             "'mm3', 'qwen3' and 'wan' for the same model, and 'wan' collides with "
             "genuine Wan video GGUFs. Key on `audiocpp.model_spec.family == "
             "\"minimax_music3\"` (audio-cpp: EXACT diffusers names, no rename "
             "table, but geometry only in the sibling config.json), on `mm3.model "
             "== \"MiniMax-Music3\"` (scragnog: fully self-describing metadata, but "
             "a rename table PLUS fused QKV and folded weight-norm to invert), or "
             "on the co-occurring `diffusion_transformer.` + `latent_conditioners.` "
             "prefixes (the ComfyUI lineage, which ships the DiT and condition "
             "encoder ONLY -- no language model, no depth decoder, no vocoder -- "
             "and therefore cannot generate audio by itself)";
    case MiniMaxMusic3QuantFormat::kNvfp4:
      return "an NVFP4 arm needs the U8-packed weight, the F8_E4M3 group scale "
             "and the F32 weight_scale_2 global routed through "
             "vllm/model_executor/model_loader/nvfp4_dequant.h, PLUS a resolved "
             "nibble order and scale framing for this producer -- neither is "
             "derivable from the shapes alone (see ltx2_loader.h and "
             ".agents/specs/nvfp4-nibble-order.md). minimax_h3_nvfp4.cpp is the "
             "in-tree precedent";
    case MiniMaxMusic3QuantFormat::kMxfp4:
      return "an MXFP4 arm needs the compressed-tensors weight_packed / "
             "weight_scale (E8M0) pair routed through "
             "vllm/model_executor/model_loader/mxfp4_dequant.h";
    case MiniMaxMusic3QuantFormat::kFp8:
      return "an FP8 arm needs the per-tensor or per-block scale read and applied "
             "for every quantized projection of the affected component; the "
             "shapes are unchanged, so nothing about the bf16 contract can see it";
    case MiniMaxMusic3QuantFormat::kInt8:
      return "an INT8 arm needs the per-channel scale read and applied; note that "
             "Comfy-Org's arm for this model is int8-convrot, which additionally "
             "carries a rotation that must be inverted, and the two published "
             "w4a8 checkpoints are int8-activation with 4-bit weights";
    case MiniMaxMusic3QuantFormat::kAwqGptq:
      return "an AWQ/GPTQ arm needs the qweight / qzeros / scales triple, its "
             "group size and its packing order routed through "
             "vllm/model_executor/model_loader/awq_gptq_dequant.h";
    case MiniMaxMusic3QuantFormat::kBitsAndBytes:
      return "a bitsandbytes arm needs the 4-bit blockwise absmax and quant_map "
             "path, which this project does not implement for ANY model -- so it "
             "is a new shared seam, not a per-model addition";
    case MiniMaxMusic3QuantFormat::kMlx:
      return "an MLX arm needs a reader for Apple MLX's grouped affine "
             "quantization, which this project does not implement for ANY model "
             "-- so it is a new shared seam, not a per-model addition";
    case MiniMaxMusic3QuantFormat::kCompressedTensors:
      return "a compressed-tensors arm needs `quantization_config.config_groups` "
             "resolved per component and each group's scheme routed to the "
             "matching shared dequant seam";
    case MiniMaxMusic3QuantFormat::kUnknownScheme:
      return "the evidence does not identify the scheme: a bare `weight_scale` "
             "with no `weight_scale_2` and no `weight_packed` is consistent with "
             "NVFP4 without its global scale, with an MXFP4 or other "
             "compressed-tensors block scheme, and with a plain per-channel int8 "
             "scale. This port names the candidates rather than picking one, "
             "because picking wrongly yields a finite, correctly shaped, WRONG "
             "result that no shape gate can see (ltx2_loader.h:232-268)";
  }
  return "";
}

}  // namespace

const char* MiniMaxMusic3QuantFormatName(MiniMaxMusic3QuantFormat format) {
  switch (format) {
    case MiniMaxMusic3QuantFormat::kNone:
      return "unquantized bf16/fp32";
    case MiniMaxMusic3QuantFormat::kGguf:
      return "GGUF";
    case MiniMaxMusic3QuantFormat::kNvfp4:
      return "NVFP4";
    case MiniMaxMusic3QuantFormat::kMxfp4:
      return "MXFP4";
    case MiniMaxMusic3QuantFormat::kFp8:
      return "FP8";
    case MiniMaxMusic3QuantFormat::kInt8:
      return "INT8";
    case MiniMaxMusic3QuantFormat::kAwqGptq:
      return "AWQ/GPTQ";
    case MiniMaxMusic3QuantFormat::kBitsAndBytes:
      return "bitsandbytes";
    case MiniMaxMusic3QuantFormat::kMlx:
      return "MLX";
    case MiniMaxMusic3QuantFormat::kCompressedTensors:
      return "compressed-tensors";
    case MiniMaxMusic3QuantFormat::kUnknownScheme:
      return "an UNIDENTIFIED quantization scheme";
  }
  return "an UNIDENTIFIED quantization scheme";
}

// ---------------------------------------------------------------------------
// CONFIG level
// ---------------------------------------------------------------------------

MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantConfig(const std::string& component,
                                                         const std::string& config_json) {
  MiniMaxMusic3QuantFinding out;
  out.component = component;
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(config_json);
  } catch (const std::exception&) {
    // An unparseable config is the config parser's problem to report, not this
    // detector's. Reporting it as "quantized" would be an accusation drawn from
    // an instrument failure.
    return out;
  }
  if (!doc.is_object()) return out;
  out.examined = static_cast<int64_t>(doc.size());

  // transformers / compressed-tensors / ModelOpt. ABSENT and `null` both mean
  // not quantized -- an unquantized `save_pretrained` writes the null form.
  const auto qc = doc.find("quantization_config");
  if (qc != doc.end() && qc->is_object()) {
    std::string method;
    const auto m = qc->find("quant_method");
    if (m != qc->end() && m->is_string()) method = m->get<std::string>();
    out.matched = 1;
    out.evidence = "quantization_config.quant_method = \"" + method + "\"";
    if (method == "awq" || method == "gptq") {
      out.format = MiniMaxMusic3QuantFormat::kAwqGptq;
    } else if (method == "fp8") {
      out.format = MiniMaxMusic3QuantFormat::kFp8;
    } else if (method == "compressed-tensors" || method == "modelopt" ||
               method == "compressed_tensors") {
      out.format = MiniMaxMusic3QuantFormat::kCompressedTensors;
    } else if (method.rfind("bitsandbytes", 0) == 0) {
      out.format = MiniMaxMusic3QuantFormat::kBitsAndBytes;
    } else if (method == "mxfp4") {
      out.format = MiniMaxMusic3QuantFormat::kMxfp4;
    } else if (method == "nvfp4") {
      out.format = MiniMaxMusic3QuantFormat::kNvfp4;
    } else {
      out.format = MiniMaxMusic3QuantFormat::kUnknownScheme;
    }
    return out;
  }

  // Apple MLX. It writes a bare `quantization` object rather than
  // `quantization_config`, so a detector that only read the latter would pass
  // an MLX tree straight into the bf16 contract. Three MLX repositories exist
  // for this model.
  const auto mlx = doc.find("quantization");
  if (mlx != doc.end() && mlx->is_object() &&
      (mlx->contains("bits") || mlx->contains("group_size"))) {
    out.format = MiniMaxMusic3QuantFormat::kMlx;
    out.matched = 1;
    out.evidence = "quantization = " + mlx->dump();
    return out;
  }
  return out;
}

// ---------------------------------------------------------------------------
// MANIFEST level
// ---------------------------------------------------------------------------

MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantManifest(
    const std::string& component, const std::vector<MiniMaxMusic3ManifestEntry>& entries) {
  MiniMaxMusic3QuantFinding out;
  out.component = component;
  out.examined = static_cast<int64_t>(entries.size());

  // One rule = one format plus the predicate that proves it. They are applied
  // in THIS order and the first with any match wins, so the answer cannot
  // depend on the order the entries arrive in -- and so a NVFP4 file, whose
  // group scale is itself F8_E4M3, is called NVFP4 rather than FP8.
  struct Rule {
    MiniMaxMusic3QuantFormat format;
    bool (*matches)(const MiniMaxMusic3ManifestEntry&);
  };
  static const Rule kRules[] = {
      // The global second scale is unique to NVFP4; nothing else writes it.
      {MiniMaxMusic3QuantFormat::kNvfp4,
       [](const MiniMaxMusic3ManifestEntry& e) { return EndsWith(e.name, "weight_scale_2"); }},
      // compressed-tensors packs into `weight_packed`.
      {MiniMaxMusic3QuantFormat::kMxfp4,
       [](const MiniMaxMusic3ManifestEntry& e) { return EndsWith(e.name, "weight_packed"); }},
      {MiniMaxMusic3QuantFormat::kAwqGptq,
       [](const MiniMaxMusic3ManifestEntry& e) {
         return EndsWith(e.name, "qweight") || EndsWith(e.name, "qzeros");
       }},
      {MiniMaxMusic3QuantFormat::kBitsAndBytes,
       [](const MiniMaxMusic3ManifestEntry& e) {
         return EndsWith(e.name, ".absmax") || EndsWith(e.name, ".quant_map") ||
                EndsWith(e.name, ".quant_state");
       }},
      // DTYPE-only formats. Nothing in the name moves, which is exactly why a
      // name-based detector alone cannot see them and why W1's dtype refusal --
      // true, and silent about quantization -- was what a user got before W7.
      {MiniMaxMusic3QuantFormat::kFp8,
       [](const MiniMaxMusic3ManifestEntry& e) {
         return e.dtype == "F8_E4M3" || e.dtype == "F8_E5M2" || e.dtype == "F8_E4M3FN";
       }},
      {MiniMaxMusic3QuantFormat::kInt8,
       [](const MiniMaxMusic3ManifestEntry& e) { return e.dtype == "I8"; }},
      // Last, and deliberately unresolved: a scale sidecar with none of the
      // discriminators above. `weight_scale_2` is excluded because rule 1 owns
      // it; `weight_g`/`weight_v` are NOT scales at all -- they are the
      // vocoder's 30 legacy weight-norm pairs (spec section 2) and matching
      // them would refuse the shipped checkpoint.
      {MiniMaxMusic3QuantFormat::kUnknownScheme,
       [](const MiniMaxMusic3ManifestEntry& e) {
         return EndsWith(e.name, "weight_scale") || EndsWith(e.name, "weight_scale_inv") ||
                EndsWith(e.name, ".scales");
       }},
  };

  for (const Rule& rule : kRules) {
    std::vector<std::string> evidence;
    int64_t matched = 0;
    for (const MiniMaxMusic3ManifestEntry& entry : entries) {
      if (!rule.matches(entry)) continue;
      ++matched;
      if (evidence.size() < kMaxEvidence) evidence.push_back(entry.name);
    }
    if (matched == 0) continue;
    out.format = rule.format;
    out.matched = matched;
    out.evidence = Join(evidence, matched);
    return out;
  }
  return out;
}

// ---------------------------------------------------------------------------
// TREE level
// ---------------------------------------------------------------------------

MiniMaxMusic3QuantFinding MiniMaxMusic3DetectQuantTree(const std::string& root) {
  MiniMaxMusic3QuantFinding out;
  std::error_code ec;
  const fs::path base(root);
  if (!fs::is_directory(base, ec)) return out;

  // Walk once, collecting both kinds of evidence, so the tree is stat'd a
  // single time. Sorted afterwards: two runs over the same directory must not
  // quote different files.
  std::vector<std::string> ggufs;
  std::vector<std::string> configs;
  int64_t examined = 0;
  fs::recursive_directory_iterator it(base, fs::directory_options::skip_permission_denied, ec);
  if (ec) return out;
  for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
    if (ec) break;
    if (it.depth() >= kMaxTreeDepth && it->is_directory(ec)) {
      it.disable_recursion_pending();
    }
    if (!it->is_regular_file(ec)) continue;
    ++examined;
    const fs::path& path = it->path();
    const std::string relative = fs::relative(path, base, ec).string();
    if (path.extension() == ".gguf") {
      ggufs.push_back(relative.empty() ? path.filename().string() : relative);
    } else if (path.filename() == "config.json") {
      configs.push_back(path.string());
    }
  }
  out.examined = examined;

  if (!ggufs.empty()) {
    std::sort(ggufs.begin(), ggufs.end());
    out.format = MiniMaxMusic3QuantFormat::kGguf;
    out.matched = static_cast<int64_t>(ggufs.size());
    std::vector<std::string> quoted(
        ggufs.begin(), ggufs.begin() + std::min(ggufs.size(), kMaxEvidence));
    out.evidence = Join(quoted, out.matched);
    return out;
  }

  // No GGUF: ask each config.json whether it DECLARES a quantization. This is
  // what catches an MLX tree and a compressed-tensors tree at resolve time,
  // before either reaches the component accounting that would misdiagnose it.
  std::sort(configs.begin(), configs.end());
  for (const std::string& config : configs) {
    std::string text;
    {
      std::error_code read_ec;
      const auto size = fs::file_size(config, read_ec);
      // A config.json is kilobytes. Anything larger is not one, and reading it
      // into a refusal path is a cost with no payer.
      if (read_ec || size > (1u << 20)) continue;
      FILE* file = std::fopen(config.c_str(), "rb");
      if (file == nullptr) continue;
      text.resize(static_cast<size_t>(size));
      const size_t got = std::fread(text.data(), 1, text.size(), file);
      std::fclose(file);
      text.resize(got);
    }
    const std::string component =
        fs::path(config).parent_path().filename().string();
    MiniMaxMusic3QuantFinding finding = MiniMaxMusic3DetectQuantConfig(component, text);
    if (finding.format == MiniMaxMusic3QuantFormat::kNone) continue;
    finding.examined = examined;
    finding.evidence =
        fs::relative(config, base, ec).string() + ": " + finding.evidence;
    return finding;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The refusal
// ---------------------------------------------------------------------------

std::string MiniMaxMusic3QuantRefusal(const MiniMaxMusic3QuantFinding& finding) {
  if (finding.format == MiniMaxMusic3QuantFormat::kNone) return std::string();
  const std::string where =
      finding.component.empty() ? std::string() : finding.component + ": ";
  std::string message =
      "minimax_music3: " + where + "this checkpoint is QUANTIZED -- " +
      MiniMaxMusic3QuantFormatName(finding.format) + " (evidence: " + finding.evidence +
      "; " + std::to_string(finding.matched) + " of " + std::to_string(finding.examined) +
      " entries examined carry the marker). NO quantized arm is implemented for "
      "MiniMax-Music3, so this is REFUSED rather than mis-loaded: " +
      MissingPieceFor(finding.format) +
      ". The supported arm is the bf16/fp32 diffusers arm -- bf16 language_model + "
      "rvq_depth_decoder + condition_encoder, fp32 transformer + vocoder, ~28.5 GB "
      "(MiniMaxAI/MiniMax-Music3; .agents/specs/minimax-music3.md section 2.1). "
      "The quantized arms are owed rather than forgotten: phase W7 of "
      ".agents/specs/minimax-music3.md, issue #672.";
  if (finding.format == MiniMaxMusic3QuantFormat::kGguf) {
    // GGUF is the one AGENTS.md names as a standing requirement, so its refusal
    // says so -- a user who reads only this line should learn that the gap is a
    // debt this project has accepted, not a decision it made about their file.
    message +=
        " GGUF k-quants are a STANDING REQUIREMENT under AGENTS.md rather than a "
        "per-model choice, and the ~9 GB Q4_K arm is what makes this model "
        "runnable on one consumer GPU at all -- this is the highest-priority "
        "debt on the row.";
  }
  return message;
}

void MiniMaxMusic3CheckQuantArm(const MiniMaxMusic3QuantFinding& finding) {
  if (finding.format == MiniMaxMusic3QuantFormat::kNone) return;
  throw std::runtime_error(MiniMaxMusic3QuantRefusal(finding));
}

// ---------------------------------------------------------------------------
// The RVQ depth decoder's GGUF Q4_K arm. See the header for why this component
// and only this component.
// ---------------------------------------------------------------------------

namespace {

// The GGUF metadata keys that IDENTIFY the audio-cpp lineage. `general.architecture`
// is deliberately NOT among them: spec section 9.2 measured it reading `audiocpp`,
// `mm3`, `qwen3` and `wan` across GGUFs of this one model, and `wan` collides with
// genuine Wan video checkpoints -- keying on it would bind another model's file.
constexpr const char* kAudioCppFamilyKey = "audiocpp.model_spec.family";
constexpr const char* kAudioCppFamily = "minimax_music3";
constexpr const char* kAudioCppNameFormatKey = "audiocpp.tensor_name_format";

const std::string* KvString(const GgufFile& file, const char* key) {
  const GgufValue* value = file.FindKv(key);
  if (value == nullptr) return nullptr;
  return std::get_if<std::string>(&value->v);
}

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

}  // namespace

bool MiniMaxMusic3GgufIsNativeLineage(const GgufFile& file) {
  const std::string* family = KvString(file, kAudioCppFamilyKey);
  return family != nullptr && *family == kAudioCppFamily;
}

MiniMaxMusic3ComponentWeights MiniMaxMusic3LoadRvqDepthDecoderFromGguf(
    const MiniMaxMusic3RvqDepthDecoderConfig& config, const GgufFile& file,
    MiniMaxMusic3GgufLoadReport* report) {
  // --- 1. LINEAGE. Refuse the other two by name rather than mis-binding. ---
  if (!MiniMaxMusic3GgufIsNativeLineage(file)) {
    const std::string* family = KvString(file, kAudioCppFamilyKey);
    const GgufValue* arch = file.FindKv("general.architecture");
    const std::string* arch_text = arch == nullptr ? nullptr : std::get_if<std::string>(&arch->v);
    throw std::runtime_error(
        std::string("minimax_music3: this GGUF is not the audio-cpp lineage (") +
        kAudioCppFamilyKey + " is " +
        (family == nullptr ? std::string("absent") : "\"" + *family + "\"") +
        ", general.architecture is " +
        (arch_text == nullptr ? std::string("absent") : "\"" + *arch_text + "\"") +
        "). Only the audio-cpp lineage is implemented, because it declares "
        "tensor_name_format=native and its 47 names match this port's enumeration "
        "EXACTLY, so no rename table can hide a mis-binding. The other two published "
        "lineages are OWED and refused here rather than mis-read: the `mm3` lineage "
        "(scragnog) needs a rename table PLUS fused QKV to split and folded "
        "weight-norm to invert, and the ComfyUI lineage (`diffusion_transformer.` + "
        "`latent_conditioners.` prefixes) does not contain the rvq_depth_decoder at "
        "all -- it ships the DiT and condition encoder ONLY, so it can never "
        "generate audio however complete this arm becomes. NOTE that "
        "general.architecture CANNOT identify a Music3 GGUF: it reads `audiocpp`, "
        "`mm3`, `qwen3` and `wan` across files of this one model, and `wan` collides "
        "with genuine Wan video GGUFs. See .agents/specs/minimax-music3.md section "
        "9.2, issue #672.");
  }
  // `native` is the only name format whose tensors bind without a rename table.
  if (const std::string* format = KvString(file, kAudioCppNameFormatKey);
      format != nullptr && *format != "native") {
    throw std::runtime_error(
        std::string("minimax_music3: this GGUF declares ") + kAudioCppNameFormatKey +
        " = \"" + *format +
        "\", but this arm reads only \"native\" -- the format in which the tensor "
        "names are the checkpoint's own. A renamed layout needs a name map that is "
        "owed, not guessed: binding it by position would be a finite, correctly "
        "shaped, WRONG model. Spec section 9.2, issue #672.");
  }

  // --- 2. ACCOUNT: enumerated == present, zero unaccounted. ---
  // The same contract the safetensors arm uses, so a GGUF that lost a tensor
  // cannot read as zeros. SHAPES are required; the per-tensor ggml TYPE is
  // whatever the quantizer chose and is recorded rather than required.
  const std::vector<MiniMaxMusic3TensorSpec> required =
      EnumerateMiniMaxMusic3RvqDepthDecoderTensors(config);
  std::map<std::string, const GgufTensorInfo*> present;
  for (const GgufTensorInfo& info : file.Tensors()) {
    if (!present.emplace(info.name, &info).second) {
      throw std::runtime_error("minimax_music3: rvq_depth_decoder gguf: tensor " + info.name +
                               " appears more than once");
    }
  }
  std::set<std::string> accounted;
  for (const MiniMaxMusic3TensorSpec& spec : required) {
    const auto it = present.find(spec.name);
    if (it == present.end()) {
      throw std::runtime_error(
          "minimax_music3: rvq_depth_decoder gguf: required tensor " + spec.name + " " +
          ShapeText(spec.shape) +
          " is MISSING. This arm reads the audio-cpp `native` layout, whose names are "
          "the checkpoint's own; a file missing one is refused rather than zero-filled.");
    }
    // GgufTensorInfo::shape is ALREADY reversed into torch row-major order by the
    // reader, so it compares directly against the enumeration. Reversing again
    // here would silently transpose every projection.
    if (it->second->shape != spec.shape) {
      throw std::runtime_error("minimax_music3: rvq_depth_decoder gguf: tensor " + spec.name +
                               " has shape " + ShapeText(it->second->shape) + " but the config "
                               "implies " + ShapeText(spec.shape) +
                               "; refusing rather than binding a different model");
    }
    accounted.insert(spec.name);
  }
  for (const auto& entry : present) {
    if (accounted.count(entry.first) == 0) {
      throw std::runtime_error(
          "minimax_music3: rvq_depth_decoder gguf: tensor " + entry.first +
          " is present but UNACCOUNTED for by the config's enumeration; a GGUF carrying "
          "a tensor this port does not know is a different model, not a superset");
    }
  }

  // --- 3. MATERIALIZE to bf16, mirroring the safetensors arm's runtime dtype. ---
  MiniMaxMusic3ComponentWeights out;
  out.component = "rvq_depth_decoder";
  MiniMaxMusic3GgufLoadReport local;
  for (const MiniMaxMusic3TensorSpec& spec : required) {
    const GgufTensorInfo& info = *present.find(spec.name)->second;
    int64_t numel = 1;
    for (const int64_t dim : spec.shape) numel *= dim;

    // NVFP4 (and anything else keeping its scale outside the block) would need a
    // sidecar this arm has not resolved. Refuse rather than apply a scale of 1
    // and hand back values off by ~1e4 that still look perfectly finite.
    if (GgmlTypeNeedsGlobalScale(info.ggml_type)) {
      throw std::runtime_error(
          "minimax_music3: rvq_depth_decoder gguf: tensor " + spec.name + " has ggml type " +
          std::to_string(info.ggml_type) +
          ", whose blocks are NOT self-contained -- it needs a per-tensor scale sidecar "
          "this arm does not resolve. Refused rather than dequantized with an assumed "
          "scale of 1.0, which would return finite values off by orders of magnitude. "
          "Owed: spec section 9.5, issue #672.");
    }

    // The shared seam does the decoding. Never a parallel path: a k-quant block
    // decoder written here would be a second implementation of something
    // gguf_dequant.h already owns and gates.
    const std::vector<uint16_t> bf16 =
        DequantGgufRowToBf16(info.ggml_type, info.data, numel);
    ++local.dequant_calls;
    if (static_cast<int64_t>(bf16.size()) != numel) {
      throw std::runtime_error("minimax_music3: rvq_depth_decoder gguf: tensor " + spec.name +
                               " dequantized to " + std::to_string(bf16.size()) +
                               " elements, expected " + std::to_string(numel));
    }

    MiniMaxMusic3Tensor tensor;
    tensor.dtype = "BF16";
    tensor.shape = spec.shape;
    tensor.bytes.resize(bf16.size() * sizeof(uint16_t));
    std::memcpy(tensor.bytes.data(), bf16.data(), tensor.bytes.size());
    out.tensors.emplace(spec.name, std::move(tensor));

    ++local.tensors;
    local.elements += numel;
    local.resident_type[spec.name] = info.ggml_type;
    if (info.ggml_type == kMusic3GgmlF16 || info.ggml_type == kMusic3GgmlBf16 ||
        info.ggml_type == 0) {
      ++local.unquantized;
    } else {
      ++local.quantized;
    }
  }
  if (report != nullptr) *report = std::move(local);
  return out;
}

}  // namespace vllm
