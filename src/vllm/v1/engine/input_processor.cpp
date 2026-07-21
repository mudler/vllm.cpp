// Ported from: vllm/v1/engine/input_processor.py @ e24d1b24
// See include/vllm/v1/engine/input_processor.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/input_processor.h"

#include <chrono>
#include <set>
#include <utility>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::v1 {
namespace {

// Wall-clock seconds since the epoch, mirroring upstream time.time().
double NowSeconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace

InputProcessor::InputProcessor(const tok::Tokenizer& tokenizer,
                               const HfConfig& config)
    : tokenizer_(tokenizer), config_(config) {
  // model_config.max_model_len. HfConfig has no dedicated max_model_len (rope
  // scaling etc. are deferred), so max_position_embeddings stands in at T0.
  max_model_len_ = config.max_position_embeddings;

  // renderer.get_eos_token_id() + generation_config["eos_token_id"]: derive the
  // primary eos id and the secondary eos-id list from config.json's
  // "eos_token_id" (int OR list), falling back to the tokenizer's own eos.
  bool found = false;
  if (config.raw.is_object()) {
    auto it = config.raw.find("eos_token_id");
    if (it != config.raw.end() && !it->is_null()) {
      if (it->is_number_integer()) {
        const auto id = it->get<int32_t>();
        eos_token_id_ = id;
        generation_config_eos_ids_.push_back(id);
        found = true;
      } else if (it->is_array()) {
        for (const auto& e : *it) {
          if (e.is_number_integer()) {
            generation_config_eos_ids_.push_back(e.get<int32_t>());
          }
        }
        if (!generation_config_eos_ids_.empty()) {
          eos_token_id_ = generation_config_eos_ids_.front();
          found = true;
        }
      }
    }
  }
  if (!found && tokenizer_.EosId() >= 0) {
    eos_token_id_ = tokenizer_.EosId();
    generation_config_eos_ids_.push_back(tokenizer_.EosId());
  }
}

void InputProcessor::ValidateParams(SamplingParams& params) const {
  // Upstream _validate_params calls params.verify(model_config, ...) after
  // __post_init__ already ran at construction. Our SamplingParams deferred
  // __post_init__ to this constructing unit (M1.1), so PostInit() both
  // normalizes the params AND runs Verify() — closing that carry.
  params.PostInit();
}

void InputProcessor::UpdateFromGenerationConfig(SamplingParams& params) const {
  // sampling_params.py:627-655 (T0 subset). _all_stop_token_ids is deferred
  // (M1.1), so its side of this is dropped; the observable effects are setting
  // eos_token_id and merging the SECONDARY eos ids into stop_token_ids.
  if (!params.ignore_eos) {
    params.eos_token_id = eos_token_id_;
  }

  if (generation_config_eos_ids_.empty()) {
    return;
  }
  std::set<int32_t> eos_ids(generation_config_eos_ids_.begin(),
                            generation_config_eos_ids_.end());
  // The primary eos id is handled separately for stopping; don't duplicate it.
  if (eos_token_id_.has_value()) {
    eos_ids.erase(*eos_token_id_);
  }
  if (!eos_ids.empty() && !params.ignore_eos) {
    for (int32_t id : params.stop_token_ids) {
      eos_ids.insert(id);
    }
    params.stop_token_ids.assign(eos_ids.begin(), eos_ids.end());
  }
}

void InputProcessor::UpdateFromTokenizer(SamplingParams& params) const {
  // sampling_params.py:657 only processes bad_words, a deferred SamplingParams
  // field (M1.1) -> no-op at T0.
  (void)params;
}

EngineCoreRequest InputProcessor::process_inputs(
    const std::string& request_id, const std::string& prompt,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // _validate_params: run PostInit()/Verify() on the (cloned) params.
  ValidateParams(params);

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  // input_preprocessor.preprocess -> tokenize (text path only). vLLM tokenizes
  // prompts with HF's default `add_special_tokens=True`, so the tokenizer's
  // post_processor template is APPLIED here. This is a no-op for every Qwen
  // tokenizer (their ByteLevel post_processor declares no bos/eos) and supplies
  // the prepended `</s>` that OPT's TemplateProcessing declares.
  std::vector<int32_t> prompt_token_ids =
      tokenizer_.EncodeWithSpecialTokens(prompt);

  // params is already our clone (passed by value). If unset max_tokens, then
  // generate up to the max_model_len (input_processor.py:317-321).
  if (!params.max_tokens.has_value()) {
    const int64_t seq_len = static_cast<int64_t>(prompt_token_ids.size());
    params.max_tokens = static_cast<int>(max_model_len_ - seq_len);
  }

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  return request;
}

}  // namespace vllm::v1
