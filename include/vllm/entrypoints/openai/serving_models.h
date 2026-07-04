// Ported from: vllm/entrypoints/openai/models/serving.py @ e24d1b24
//   (OpenAIModelRegistry / OpenAIServingModels.model_name / is_base_model /
//   check_model / show_available_models) + the ModelCard / ModelList shapes
//   from vllm/entrypoints/openai/engine/protocol.py:86,97.
//
// SCOPE (M3.1 Task 4 / T0): a MINIMAL, read-only served-model registry — the
// backing for GET /v1/models and the request `model`-field validation the
// route handlers do. No LoRA adapters, no engine dependency (LoRA load/unload,
// ModelPermission, root/parent/max_model_len are DEFERRED — matches upstream so
// re-adding is mechanical). T0 serves exactly ONE base model.
#ifndef VLLM_ENTRYPOINTS_OPENAI_SERVING_MODELS_H_
#define VLLM_ENTRYPOINTS_OPENAI_SERVING_MODELS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

// Ported from: engine/protocol.py:86 (ModelCard). object == "model"; T0 subset
// (owned_by "vllm"; root/parent/max_model_len/permission deferred).
struct ModelCard {
  std::string id;
  std::string object = "model";
  int64_t created = 0;
  std::string owned_by = "vllm";
};

// Ported from: engine/protocol.py:97 (ModelList). object == "list".
struct ModelList {
  std::string object = "list";
  std::vector<ModelCard> data;
};

void to_json(nlohmann::json& j, const ModelCard& c);
void to_json(nlohmann::json& j, const ModelList& l);

// Ported from: models/serving.py:30 (OpenAIModelRegistry) — the read-only view.
// Holds the served base-model name(s); the first is the canonical served id.
class OpenAIServingModels {
 public:
  explicit OpenAIServingModels(std::vector<std::string> served_model_names);
  // Convenience single-model ctor.
  explicit OpenAIServingModels(std::string served_model_name);

  // model_name (models/serving.py:47/144): the canonical served id (T0: [0]).
  const std::string& model_name() const { return served_model_names_.front(); }

  // is_base_model (models/serving.py:50): is `name` one of the served models.
  bool is_base_model(const std::string& name) const;

  // check_model (models/serving.py:53): std::nullopt (empty/absent model resolves
  // to the served model), else the name is unknown → the caller returns 404.
  // Returns true when the request model is served (or unset), false otherwise.
  bool check_model(const std::optional<std::string>& model) const;

  // show_available_models (models/serving.py:149): the ModelList for /v1/models.
  ModelList show_available_models() const;

 private:
  std::vector<std::string> served_model_names_;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_SERVING_MODELS_H_
