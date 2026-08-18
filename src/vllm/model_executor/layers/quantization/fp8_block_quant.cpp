#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {
namespace {

// `quantization_config`, top level first and then the `text_config` nesting.
// `HfConfig::raw` holds the FULL top-level document
// (`src/vllm/transformers_utils/hf_config.cpp:564`), so both are reachable from
// here. `Qwen/Qwen3.8-27B-FP8` uses the top-level spelling, measured: its
// `text_config` carries no `quantization_config`.
const nlohmann::json* QuantizationConfigOf(const HfConfig& config) {
  if (!config.raw.is_object()) return nullptr;
  const auto top = config.raw.find("quantization_config");
  if (top != config.raw.end() && top->is_object()) return &*top;
  const auto text = config.raw.find("text_config");
  if (text == config.raw.end() || !text->is_object()) return nullptr;
  const auto nested = text->find("quantization_config");
  if (nested != text->end() && nested->is_object()) return &*nested;
  return nullptr;
}

std::string DimensionList(const std::vector<int64_t>& dims) {
  std::string out = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(dims[i]);
  }
  return out + "]";
}

std::string StringOr(const nlohmann::json& quant, const char* key,
                     const std::string& fallback) {
  const auto it = quant.find(key);
  if (it == quant.end() || !it->is_string()) return fallback;
  return it->get<std::string>();
}

// `ignored_layers` first, then `modules_to_not_convert`, exactly the order
// `Fp8Config.from_config` reads them in (`fp8.py:160,165-168`): the fallback is
// taken only when the first is absent or empty.
std::vector<std::string> IgnoreListOf(const nlohmann::json& quant) {
  const auto read = [&quant](const char* key) {
    std::vector<std::string> out;
    const auto it = quant.find(key);
    if (it == quant.end() || !it->is_array()) return out;
    for (const nlohmann::json& entry : *it)
      if (entry.is_string()) out.push_back(entry.get<std::string>());
    return out;
  };
  std::vector<std::string> ignored = read("ignored_layers");
  if (!ignored.empty()) return ignored;
  return read("modules_to_not_convert");
}

constexpr int64_t kSupportedBlockN = 128;
constexpr int64_t kSupportedBlockK = 128;

const char* kIssue = "https://github.com/mudler/vllm.cpp/issues/1189";

}  // namespace

bool Fp8BlockQuantConfig::ExcludesModule(
    const std::string& module_prefix) const {
  return std::find(modules_to_not_convert.begin(),
                   modules_to_not_convert.end(),
                   module_prefix) != modules_to_not_convert.end();
}

std::vector<int64_t> Fp8WeightBlockSizeOf(const HfConfig& config) {
  const nlohmann::json* quant = QuantizationConfigOf(config);
  if (quant == nullptr) return {};
  const auto it = quant->find("weight_block_size");
  // Absent and null are both "None" upstream (`fp8.py:161` reads it with a
  // default of None), so neither is block quant.
  if (it == quant->end() || it->is_null() || !it->is_array()) return {};
  std::vector<int64_t> dims;
  dims.reserve(it->size());
  for (const nlohmann::json& dim : *it) {
    if (!dim.is_number_integer()) return {};
    dims.push_back(dim.get<int64_t>());
  }
  return dims;
}

Fp8BlockQuantConfig ReadFp8BlockQuantConfig(const HfConfig& config) {
  Fp8BlockQuantConfig out;
  const std::vector<int64_t> block = Fp8WeightBlockSizeOf(config);
  if (block.empty()) return out;

  const nlohmann::json* quant = QuantizationConfigOf(config);
  // `Fp8WeightBlockSizeOf` only returns a non-empty list when it found one, so
  // the object exists. Asserted rather than assumed because a null deref here
  // would be the one failure mode this whole file exists to prevent.
  if (quant == nullptr) return out;

  // `is_checkpoint_fp8_serialized = "fp8" in quant_method` (`fp8.py:158-159`),
  // and block quant requires it (`fp8.py:117-120`).
  const std::string method = StringOr(*quant, "quant_method", "");
  if (method.find("fp8") == std::string::npos) {
    throw std::runtime_error(
        "quantization_config.weight_block_size " + DimensionList(block) +
        " selects block-wise (fine-grained) FP8, but quant_method is \"" +
        method +
        "\", which is not an fp8-serialized checkpoint. Block-wise weight "
        "quantization is defined only for fp8 here, exactly as upstream "
        "requires (vllm fp8.py:117-120). Tracked by " +
        kIssue);
  }

  // `len(weight_block_size) != 2` (`fp8.py:121-126`).
  if (block.size() != 2) {
    throw std::runtime_error(
        "quantization_config.weight_block_size " + DimensionList(block) +
        " has " + std::to_string(block.size()) +
        " dimensions. A block-wise FP8 quantization block must have exactly "
        "2 dimensions, [block_n, block_k], one scale per block of the "
        "[out_features, in_features] weight. Tracked by " +
        kIssue);
  }

  // `activation_scheme != "dynamic"` (`fp8.py:127-131`). The default matches
  // `Fp8Config.__init__`'s own default (`fp8.py:102`).
  const std::string scheme = StringOr(*quant, "activation_scheme", "dynamic");
  if (scheme != "dynamic") {
    throw std::runtime_error(
        "quantization_config.activation_scheme is \"" + scheme +
        "\" beside weight_block_size " + DimensionList(block) +
        ". Block-wise (fine-grained) FP8 supports only the \"dynamic\" "
        "activation scheme, which quantizes activations per token and per "
        "group at run time; a static per-tensor input_scale cannot express "
        "it. Upstream refuses the same combination (vllm fp8.py:127-131). "
        "Tracked by " +
        kIssue);
  }

  // OUR limit, not upstream's, and the message says so. `vt::QuantFp8Group`
  // and `vt::MatmulFp8BlockScaled` are general, but the M5 CUTLASS kernel that
  // will actually execute this is 128x128 and no gate in this tree has run any
  // other shape end to end. Loading a [64, 128] checkpoint would fill a weight
  // nothing can consume.
  if (block[0] != kSupportedBlockN || block[1] != kSupportedBlockK) {
    throw std::runtime_error(
        "quantization_config.weight_block_size " + DimensionList(block) +
        " selects a block-wise (fine-grained) FP8 block shape this build does "
        "not implement. Only [" +
        std::to_string(kSupportedBlockN) + ", " +
        std::to_string(kSupportedBlockK) +
        "] is implemented, which is the shape Qwen3.8-27B-FP8 and DeepSeek-V3 "
        "style checkpoints ship. This is a missing arm in vllm.cpp and not a "
        "problem with the checkpoint. Tracked by " +
        kIssue);
  }

  out.block_quant = true;
  out.block_n = block[0];
  out.block_k = block[1];
  out.activation_scheme = scheme;
  out.modules_to_not_convert = IgnoreListOf(*quant);
  return out;
}

void RefuseUnsupportedFp8BlockQuant(const HfConfig& config) {
  const Fp8BlockQuantConfig block = ReadFp8BlockQuantConfig(config);
  (void)block;
}

void RefuseUnconsumedFp8BlockWeight(const std::string& proj) {
  // Named, not merely refused: the projection so the reader knows this is a
  // real loaded weight rather than a config guess, what is missing, what DOES
  // work today, and the issue that owes the rest.
  throw std::runtime_error(
      "block-wise (fine-grained) 128x128 FP8 weights LOADED for " + proj +
      " and nothing in this build can execute them: there is no block-wise "
      "FP8 linear method, so the checkpoint would run through an empty weight "
      "and produce fluent wrong output. The loader, the weight and the CPU "
      "reference GEMM are implemented; the linear method and the forward "
      "wiring are milestone M4 of " +
      std::string(kIssue) +
      ". Per-tensor FP8 and NVFP4 checkpoints of the same architecture run "
      "today.");
}

}  // namespace vllm
