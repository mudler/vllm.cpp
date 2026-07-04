// Ported from: vllm/entrypoints/openai/models/serving.py @ e24d1b24.
// See serving_models.h for scope + deferrals.
#include "vllm/entrypoints/openai/serving_models.h"

#include <ctime>
#include <stdexcept>
#include <utility>

namespace vllm::entrypoints::openai {

void to_json(nlohmann::json& j, const ModelCard& c) {
  j = nlohmann::json{{"id", c.id},
                     {"object", c.object},
                     {"created", c.created},
                     {"owned_by", c.owned_by}};
}

void to_json(nlohmann::json& j, const ModelList& l) {
  j = nlohmann::json{{"object", l.object}, {"data", l.data}};
}

OpenAIServingModels::OpenAIServingModels(
    std::vector<std::string> served_model_names)
    : served_model_names_(std::move(served_model_names)) {
  if (served_model_names_.empty()) {
    throw std::invalid_argument(
        "OpenAIServingModels requires at least one served model name");
  }
}

OpenAIServingModels::OpenAIServingModels(std::string served_model_name)
    : OpenAIServingModels(
          std::vector<std::string>{std::move(served_model_name)}) {}

bool OpenAIServingModels::is_base_model(const std::string& name) const {
  for (const std::string& n : served_model_names_) {
    if (n == name) return true;
  }
  return false;
}

bool OpenAIServingModels::check_model(
    const std::optional<std::string>& model) const {
  // models/serving.py:53: an absent/empty model name resolves to the served
  // model (no error); otherwise it must be one of the base models.
  if (!model.has_value() || model->empty()) return true;
  return is_base_model(*model);
}

ModelList OpenAIServingModels::show_available_models() const {
  const auto created = static_cast<int64_t>(std::time(nullptr));
  ModelList list;
  for (const std::string& name : served_model_names_) {
    ModelCard card;
    card.id = name;
    card.created = created;
    list.data.push_back(std::move(card));
  }
  return list;
}

}  // namespace vllm::entrypoints::openai
