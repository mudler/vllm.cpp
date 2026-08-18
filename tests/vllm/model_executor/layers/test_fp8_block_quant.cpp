// Block-wise (fine-grained) FP8: the CONFIG gate at `ModelRegistry::Load`.
//
// This file used to assert that the whole scheme was refused (issue #1166,
// `469f38395`, spec `.agents/specs/fp8-blockwise-refusal.md`). MODEL-FP8-BLOCK-
// WEIGHT (#1189 milestone M3, spec `.agents/specs/model-fp8-block-weight.md`)
// narrowed that: `Qwen/Qwen3.8-27B-FP8`'s `[128, 128]` `dynamic` config now
// PASSES this gate and its projections load through the `weight_scale_inv` rung
// in `qwen3_5_dense_weights.cpp`. What is still refused here is what no build in
// this tree can execute, and each refusal names the part.
//
// EVERY case here enters through `ModelRegistry::Load`, which the production
// loader `src/vllm/entrypoints/model_loader.cpp::FromModelDir` calls, and NOT
// through the predicate. That is deliberate and it is the reachability proof
// AGENTS.md `## Nothing lands dead` asks for: a unit test that called the predicate
// directly would prove the function works and never that a load reaches it.
// Deleting the call site in `ModelRegistry::Load` must red this file.
//
// No checkpoint, no GPU, and no model directory: the refusal fires before
// `factory.load_weights`, so an EMPTY shard vector is all a load needs to reach
// it. That is why the guard sits in `ModelRegistry::Load` rather than at the
// other pre-load refusal site,
// `src/vllm/entrypoints/model_loader.cpp::RefuseUnsupportedWeightOffload`,
// which needs a directory on disk.
//
// The LOADING half -- the rung, the weight, the scale dtype, the config/tensor
// cross-check -- is gated by `test_fp8_block_weight_load`, which needs a
// synthetic checkpoint. This file needs none.
#include <doctest/doctest.h>

#include <string>
#include <vector>

// `model_registry.h` only forward-declares `SafetensorsFile`, and building the
// empty shard vector the loads below hand to `ModelSource` needs the complete
// type.
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

// A registered architecture, so `ModelRegistry::Resolve` succeeds and the load
// gets far enough to be refused for the QUANTIZATION rather than for the
// architecture. Pinned by `tests/vllm/models/test_model_registry.cpp:88`.
constexpr const char* kArch = "Qwen3_5ForConditionalGeneration";

vllm::HfConfig ConfigWithQuant(const nlohmann::json& quant, bool nested) {
  vllm::HfConfig config;
  config.architectures = {kArch};
  config.model_type = "qwen3_5";
  nlohmann::json doc = nlohmann::json::object();
  doc["architectures"] = nlohmann::json::array({kArch});
  if (nested) {
    doc["text_config"] = nlohmann::json::object();
    doc["text_config"]["quantization_config"] = quant;
  } else {
    doc["quantization_config"] = quant;
  }
  config.raw = std::move(doc);
  return config;
}

// The message `ModelRegistry::Load` fails with, or "" when it does not throw.
// The load is EXPECTED to throw in every case here: a config carrying no
// weights cannot produce a model. What each case asserts is WHICH sentence
// comes back, which is the whole subject of issues #1166 and #1189.
std::string LoadFailureMessage(const vllm::HfConfig& config) {
  const std::vector<vllm::SafetensorsFile> shards;
  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  try {
    std::unique_ptr<vllm::LoadedModel> model =
        vllm::ModelRegistry::Load(config, source);
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// The sentence a config that PASSES the quantization gate fails with instead:
// the load reached `factory.load_weights` and found no tensors. Asserting this
// is what proves the gate did not fire, and it is the same marker the negative
// controls below use.
constexpr const char* kReachedLoader = "backbone tensors found";

// The block-wise config the real checkpoint ships, measured from
// `Qwen/Qwen3.8-27B-FP8` at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
// on 2026-08-17: `quant_method` fp8, `weight_block_size` [128, 128],
// `activation_scheme` dynamic, `fmt` e4m3.
nlohmann::json BlockWiseQuantConfig() {
  nlohmann::json quant = nlohmann::json::object();
  quant["quant_method"] = "fp8";
  quant["fmt"] = "e4m3";
  quant["activation_scheme"] = "dynamic";
  quant["weight_block_size"] = nlohmann::json::array({128, 128});
  return quant;
}

}  // namespace

TEST_CASE("fp8 block quant: the SUPPORTED block-wise config reaches the loader") {
  // The whole point of #1189 M3. Before it this config was refused by name;
  // now it passes the quantization gate and the load fails only because this
  // case ships no weights. Without this assertion the file passes for a gate
  // that refuses every block-wise checkpoint, which is the state M3 replaces.
  const std::string message =
      LoadFailureMessage(ConfigWithQuant(BlockWiseQuantConfig(), false));
  REQUIRE_FALSE(message.empty());
  MESSAGE("supported block-wise fp8 load failed with: " << message);
  CHECK(Names(message, kReachedLoader));
  CHECK_FALSE(Names(message, "not implemented"));
  CHECK_FALSE(Names(message, "does not implement"));
}

TEST_CASE("fp8 block quant: a non-dynamic activation scheme is refused by name") {
  // Upstream refuses the same combination
  // (`vllm/model_executor/layers/quantization/fp8.py:127-131` @ `555967922`):
  // a static per-tensor input scale cannot express a per-token per-group
  // dynamic quantization.
  nlohmann::json quant = BlockWiseQuantConfig();
  quant["activation_scheme"] = "static";
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, false));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "activation_scheme"));
  CHECK(Names(message, "static"));   // the VALUE this checkpoint declares
  CHECK(Names(message, "dynamic"));  // the value that works
  CHECK(Names(message, "1189"));     // the owing pointer
  // THE REFUSAL MUST PREEMPT THE WEIGHT LOADER. Asserting the absence of the
  // real checkpoint's `tensor not found` would be a MUTE SWITCH: this config
  // carries no shards, so an unguarded load does not reach that sentence
  // either. It reaches a DIFFERENT one, `kReachedLoader`, which means
  // `factory.load_weights` RAN. Requiring that sentence to be absent proves the
  // guard fires before the loader, and it is exactly what reds when the call
  // site in `ModelRegistry::Load` is deleted.
  CHECK_FALSE(Names(message, kReachedLoader));
}

TEST_CASE("fp8 block quant: a block shape other than 128x128 is refused by name") {
  // OUR limit rather than upstream's, and the message says so: #1189's CUTLASS
  // kernel (M5) is 128x128 and nothing here has run any other shape end to end,
  // so a [64, 128] checkpoint would fill a weight nothing can consume.
  nlohmann::json quant = BlockWiseQuantConfig();
  quant["weight_block_size"] = nlohmann::json::array({64, 128});
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, false));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "weight_block_size"));
  CHECK(Names(message, "[64, 128]"));  // the VALUE quoted back
  CHECK(Names(message, "1189"));
  CHECK_FALSE(Names(message, kReachedLoader));
}

TEST_CASE("fp8 block quant: a weight_block_size that is not 2-D is refused by name") {
  // Upstream refuses it too (`fp8.py:121-126`).
  nlohmann::json quant = BlockWiseQuantConfig();
  quant["weight_block_size"] = nlohmann::json::array({128});
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, false));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "weight_block_size"));
  CHECK(Names(message, "2 dimensions"));
  CHECK_FALSE(Names(message, kReachedLoader));
}

TEST_CASE("fp8 block quant: a non-fp8 quant_method is refused by name") {
  // Mirrors `is_checkpoint_fp8_serialized = "fp8" in quant_method`
  // (`fp8.py:158-159`) and the `__init__` refusal it feeds (`fp8.py:117-120`).
  // Block-wise weight quantization is defined only for fp8 here.
  nlohmann::json quant = BlockWiseQuantConfig();
  quant["quant_method"] = "awq";
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, false));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "quant_method"));
  CHECK(Names(message, "awq"));
  CHECK_FALSE(Names(message, kReachedLoader));
}

TEST_CASE("fp8 block quant: the nested text_config spelling is read too") {
  // A multimodal wrapper can nest `quantization_config` under `text_config`.
  // `Qwen/Qwen3.8-27B-FP8` uses the TOP-LEVEL spelling (measured: `text_config`
  // carries no `quantization_config`), so this covers the shape the next
  // checkpoint can arrive in and would otherwise load straight past the gate.
  nlohmann::json quant = BlockWiseQuantConfig();
  quant["activation_scheme"] = "static";
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, true));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "activation_scheme"));
  CHECK(Names(message, "1189"));
  CHECK_FALSE(Names(message, kReachedLoader));
}

TEST_CASE("fp8 block quant: a per-tensor fp8 config is not refused as block-wise") {
  // WITHOUT this case the gate passes for a refusal that fires on EVERY fp8
  // checkpoint, which would break the per-tensor arm this tree does implement
  // and gates on the 27B and 35B. The load still fails, because there are no
  // weights, but it must fail for a different reason.
  nlohmann::json quant = BlockWiseQuantConfig();
  quant.erase("weight_block_size");
  const std::string message = LoadFailureMessage(ConfigWithQuant(quant, false));
  REQUIRE_FALSE(message.empty());
  // The sentence a NON-block-wise load fails with here is evidence, not noise:
  // it is what the negative controls are asserting the ABSENCE of markers in,
  // and a control that never says what it saw cannot be falsified.
  MESSAGE("per-tensor fp8 load failed with: " << message);
  CHECK(Names(message, kReachedLoader));
  CHECK_FALSE(Names(message, "weight_block_size"));
  CHECK_FALSE(Names(message, "block-wise"));
}

TEST_CASE("fp8 block quant: a null or empty weight_block_size is not block-wise") {
  // Upstream treats the key as absent when it is null
  // (`Fp8Config.from_config`, `vllm/model_executor/layers/quantization/fp8.py:161`
  // reads it with a default of None), so a null must not refuse a checkpoint
  // that is really per-tensor. A `[64, 128]` value is used for the OTHER keys
  // here so that a reader that mistook null for a real list would be refused
  // and this case would notice.
  nlohmann::json null_quant = BlockWiseQuantConfig();
  null_quant["weight_block_size"] = nullptr;
  const std::string null_message =
      LoadFailureMessage(ConfigWithQuant(null_quant, false));
  REQUIRE_FALSE(null_message.empty());
  CHECK(Names(null_message, kReachedLoader));

  nlohmann::json empty_quant = BlockWiseQuantConfig();
  empty_quant["weight_block_size"] = nlohmann::json::array();
  const std::string empty_message =
      LoadFailureMessage(ConfigWithQuant(empty_quant, false));
  REQUIRE_FALSE(empty_message.empty());
  CHECK(Names(empty_message, kReachedLoader));
}
