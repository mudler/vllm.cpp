// Block-wise (fine-grained) FP8 is refused BY NAME at load — issue #1166, spec
// `.agents/specs/fp8-blockwise-refusal.md`.
//
// `Qwen/Qwen3.8-27B-FP8` declares `quantization_config.weight_block_size`
// `[128, 128]` and stores one scale per 128x128 block under `weight_scale_inv`.
// This tree implements PER-TENSOR fp8 only. Before this gate the load still
// stopped, so the defect was never wrong numerics — it stopped on the wrong
// sentence. `LoadFp8Raw` asks for `<proj>.weight_scale`
// (`src/vllm/model_executor/models/qwen3_5_weights.cpp:458`), the checkpoint
// spells that tensor `weight_scale_inv`, and the resolver raised
// `qwen3_5 dense: tensor not found: ...q_proj.weight_scale`. Nothing was missing
// from the checkpoint. The reader was sent after a tensor upstream never writes
// in this mode instead of being told the fine-grained arm is absent.
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
// comes back, which is the whole subject of issue #1166.
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

TEST_CASE("fp8 block quant: a block-wise checkpoint is refused by name") {
  const std::string message =
      LoadFailureMessage(ConfigWithQuant(BlockWiseQuantConfig(), false));
  REQUIRE_FALSE(message.empty());

  // The KEY, so a reader can grep their own config.json for it.
  CHECK(Names(message, "weight_block_size"));
  // The VALUE quoted back, so the message is about THIS checkpoint.
  CHECK(Names(message, "128"));
  // WHAT is missing, named. "unsupported" alone makes the next person
  // re-derive it.
  CHECK(Names(message, "block-wise"));
  // WHAT WOULD work, so the refusal points somewhere.
  CHECK(Names(message, "per-tensor"));
  // The owing pointer. Deleting it reds this line.
  CHECK(Names(message, "1166"));

  // THE REFUSAL MUST PREEMPT THE WEIGHT LOADER, and this is the assertion that
  // can actually fail. Asserting the absence of the real checkpoint's
  // `tensor not found: ...weight_scale` would be a MUTE SWITCH here: this
  // config carries no shards, so without a guard the load does not reach that
  // sentence either. It reaches a different one. Measured on the RED run of
  // this same file, an unguarded load of this config fails with
  // `no Qwen3.5 backbone tensors found` from `qwen3_5_weights.cpp:1190`, which
  // means `factory.load_weights` RAN. So requiring that sentence to be absent
  // proves the guard fires before the loader rather than after it, and it is
  // exactly what reds when the call site is deleted.
  CHECK_FALSE(Names(message, "backbone tensors found"));
}

TEST_CASE("fp8 block quant: the nested text_config spelling is refused too") {
  // A multimodal wrapper can nest `quantization_config` under `text_config`.
  // `Qwen/Qwen3.8-27B-FP8` uses the TOP-LEVEL spelling (measured: `text_config`
  // carries no `quantization_config`), so this case covers the shape the next
  // checkpoint can arrive in and would otherwise load straight past the guard.
  const std::string message =
      LoadFailureMessage(ConfigWithQuant(BlockWiseQuantConfig(), true));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "weight_block_size"));
  CHECK(Names(message, "block-wise"));
  CHECK(Names(message, "1166"));
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
  CHECK_FALSE(Names(message, "weight_block_size"));
  CHECK_FALSE(Names(message, "block-wise"));
  CHECK_FALSE(Names(message, "1166"));
}

TEST_CASE("fp8 block quant: a null or empty weight_block_size is not block-wise") {
  // Upstream treats the key as absent when it is null
  // (`Fp8Config.from_config`, `vllm/model_executor/layers/quantization/fp8.py:161`
  // reads it with a default of None), so a null must not refuse a checkpoint
  // that is really per-tensor.
  nlohmann::json null_quant = BlockWiseQuantConfig();
  null_quant["weight_block_size"] = nullptr;
  const std::string null_message =
      LoadFailureMessage(ConfigWithQuant(null_quant, false));
  REQUIRE_FALSE(null_message.empty());
  CHECK_FALSE(Names(null_message, "block-wise"));

  nlohmann::json empty_quant = BlockWiseQuantConfig();
  empty_quant["weight_block_size"] = nlohmann::json::array();
  const std::string empty_message =
      LoadFailureMessage(ConfigWithQuant(empty_quant, false));
  REQUIRE_FALSE(empty_message.empty());
  CHECK_FALSE(Names(empty_message, "block-wise"));
}
