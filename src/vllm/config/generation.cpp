// Ported from: vllm/config/model.py @ 5559679229bc961848b121ccdeaa8fa5d79bec98.
// See include/vllm/config/generation.h for scope and deferrals.
#include "vllm/config/generation.h"

#include <sstream>

namespace vllm {

DefaultSamplingParams NarrowGenerationConfigSampling(
    const GenerationConfigSampling& s) {
  // get_diff_sampling_param's `available_params` list, in its order. Upstream
  // keeps a key only when `config.get(p) is not None`, which is exactly what an
  // unset optional carries here, so the copy is unconditional.
  DefaultSamplingParams out;
  out.repetition_penalty = s.repetition_penalty;
  out.temperature = s.temperature;
  out.top_k = s.top_k;
  out.top_p = s.top_p;
  out.min_p = s.min_p;
  // "Huggingface definition of max_new_tokens is equivalent to vLLM's
  // max_tokens" -- the rename happens here and nowhere else.
  out.max_tokens = s.max_new_tokens;
  return out;
}

DefaultSamplingParams GetDiffSamplingParam(const HfConfig& config,
                                           const std::string& generation_config) {
  // `src == "vllm"` -> `config = {}` (config/model.py). No file is read and
  // every knob falls through to the neutral OpenAI default, which is exactly
  // the behaviour this tree had before the checkpoint's file was read at all.
  if (generation_config == kGenerationConfigNone) return {};
  if (generation_config == kGenerationConfigAuto) {
    return NarrowGenerationConfigSampling(config.generation_config_sampling);
  }
  // A directory path: try_get_generation_config(self.generation_config, ...)
  // loads generation_config.json out of the named folder rather than the
  // checkpoint's. A trailing separator is tolerated so `--generation-config
  // /some/dir/` behaves like `/some/dir`.
  std::string dir = generation_config;
  while (dir.size() > 1 && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
  return NarrowGenerationConfigSampling(
      ReadGenerationConfigSamplingFile(dir + "/generation_config.json"));
}

std::string DefaultSamplingParams::ToString() const {
  // The dict repr upstream interpolates into its "Default vLLM sampling
  // parameters have been overridden by %s: `%s`" warning. Order follows
  // get_diff_sampling_param's available_params so two engines' logs line up.
  std::ostringstream os;
  bool first = true;
  const auto put = [&](const char* key, const std::string& value) {
    if (!first) os << ", ";
    first = false;
    os << "'" << key << "': " << value;
  };
  const auto num = [](double v) {
    std::ostringstream t;
    t << v;
    return t.str();
  };
  if (repetition_penalty.has_value()) put("repetition_penalty", num(*repetition_penalty));
  if (temperature.has_value()) put("temperature", num(*temperature));
  if (top_k.has_value()) put("top_k", std::to_string(*top_k));
  if (top_p.has_value()) put("top_p", num(*top_p));
  if (min_p.has_value()) put("min_p", num(*min_p));
  if (max_tokens.has_value()) put("max_tokens", std::to_string(*max_tokens));
  if (first) return {};
  return "{" + os.str() + "}";
}

}  // namespace vllm
