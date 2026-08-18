#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"

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

}  // namespace

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

void RefuseUnsupportedFp8BlockQuant(const HfConfig& config) {
  const std::vector<int64_t> block = Fp8WeightBlockSizeOf(config);
  if (block.empty()) return;

  // Named, not merely refused. The key so the reader can grep their own
  // config.json, the value so the message is about THIS checkpoint, the arm
  // that is missing, the arm that works, where the scale actually lives, and
  // the issue that owes the port.
  throw std::runtime_error(
      "quantization_config.weight_block_size " + DimensionList(block) +
      " selects block-wise (fine-grained) FP8, which is not implemented. This "
      "build implements per-tensor FP8 only. A block-wise checkpoint stores one "
      "scale for each " +
      (block.size() == 2 ? DimensionList(block) : std::string("block")) +
      " weight block under `weight_scale_inv`, and nothing here reads that "
      "tensor, so the weights cannot be dequantized correctly. This is a "
      "missing arm in vllm.cpp and not a problem with the checkpoint. Tracked "
      "by https://github.com/mudler/vllm.cpp/issues/1166");
}

}  // namespace vllm
