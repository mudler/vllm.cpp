// Production weight loader for cua-s1-forms (MODEL-CUA-S1-FORMS Phase 3).
//
// Reads all F32 tensors from the safetensors checkpoint, parses the model
// params from config.raw (the unwrapped cua-s1-forms.json config envelope),
// and calls cua_s1::Load.
//
// All tensors are F32 — no F16 conversion needed (unlike Laya, which stores
// weights in F16). The loader copies raw F32 bytes straight from the
// safetensors mmap into CheckpointTensors.
#include "vllm/model_executor/models/cua_s1.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// Copy one StTensor into a CheckpointTensors map as F32. All cua-s1
// checkpoint tensors are F32.
void LoadTensor(const StTensor& t, const std::string& name,
                cua_s1::CheckpointTensors& out) {
  VT_CHECK(t.data != nullptr,
           "cua_s1: tensor '" + name + "' has null data");
  if (t.dtype != "F32") {
    throw std::runtime_error(
        "cua_s1: tensor '" + name + "' has unsupported dtype " + t.dtype +
        " (expected F32)");
  }
  VT_CHECK(t.nbytes % sizeof(float) == 0,
           "cua_s1: tensor '" + name + "' nbytes not a multiple of 4");
  size_t numel = t.nbytes / sizeof(float);
  const float* ptr = reinterpret_cast<const float*>(t.data);
  std::vector<float> data(ptr, ptr + numel);
  std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
  out.Set(name, std::move(shape), std::move(data));
}

// Parse cua_s1::Params from config.raw. The cua-s1-forms.json envelope has
// been unwrapped in ParseHfConfigDoc, so the params are at the top level of
// config.raw: width, rank, context_tokens, option_tokens, layers, heads.
cua_s1::Params ParseParams(const HfConfig& config) {
  cua_s1::Params p;
  const auto& raw = config.raw;
  if (raw.contains("width") && raw["width"].is_number()) {
    p.width = raw["width"].get<int64_t>();
  }
  if (raw.contains("rank") && raw["rank"].is_number()) {
    p.rank = raw["rank"].get<int64_t>();
  }
  if (raw.contains("context_tokens") && raw["context_tokens"].is_number()) {
    p.context_tokens = raw["context_tokens"].get<int64_t>();
  }
  if (raw.contains("option_tokens") && raw["option_tokens"].is_number()) {
    p.option_tokens = raw["option_tokens"].get<int64_t>();
  }
  if (raw.contains("layers") && raw["layers"].is_number()) {
    p.layers = raw["layers"].get<int64_t>();
  }
  if (raw.contains("heads") && raw["heads"].is_number()) {
    p.heads = raw["heads"].get<int64_t>();
  }
  return p;
}

}  // namespace

CuaS1ModelWeights LoadCuaS1Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  // Build the name -> shard index so each Get(name) finds the right file.
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;

  // Read all F32 tensors into CheckpointTensors.
  cua_s1::CheckpointTensors tensors;
  for (const auto& [name, shard_ptr] : where) {
    const StTensor& t = shard_ptr->Get(name);
    LoadTensor(t, name, tensors);
  }

  // Parse model params from config.raw (the unwrapped cua-s1-forms.json
  // config envelope).
  cua_s1::Params params = ParseParams(config);

  // Load weights via the existing host reference loader, which looks up
  // tensors by name and validates shapes.
  cua_s1::Weights weights = cua_s1::Load(params, tensors);

  return CuaS1ModelWeights{std::move(params), std::move(weights)};
}

}  // namespace vllm
