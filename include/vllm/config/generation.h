// Ported from: vllm/config/model.py @ 5559679229bc961848b121ccdeaa8fa5d79bec98
// (ModelConfig.generation_config, ModelConfig.try_get_generation_config,
// ModelConfig.get_diff_sampling_param).
//
// SCOPE: turning a checkpoint's generation_config.json into the server-wide
// sampling defaults an OpenAI request resolves against. Upstream owns this on
// ModelConfig; this tree has no ModelConfig, so the narrowing is a free
// function over the already-parsed HfConfig, which is where the file's keys
// land (transformers_utils/hf_config.h).
//
// DEFERRED, and tracked under `## Owed` in
// .agents/specs/sample-gen-config-and-parallel-gumbel.md:
//   - --override-generation-config (config/model.py:305), the dict that is
//     merged over the file's values before the narrowing.
//   - the server-wide output cap upstream derives from max_new_tokens
//     (completion/serving.py:81-86). max_tokens is CARRIED here so a later row
//     has it, and no caller reads it yet.
#ifndef VLLM_CONFIG_GENERATION_H_
#define VLLM_CONFIG_GENERATION_H_

#include <string>

#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The three forms vLLM's --generation-config takes (config/model.py:298-304).
// "auto" is the DEFAULT and means the checkpoint's own file.
inline constexpr const char* kGenerationConfigAuto = "auto";
// "vllm" means load no file at all and use vLLM's neutral defaults.
inline constexpr const char* kGenerationConfigNone = "vllm";

// ModelConfig.get_diff_sampling_param (config/model.py). `generation_config` is
// the --generation-config selector: "auto" (the checkpoint's own sibling file,
// already parsed into `config`), "vllm" (no file), or a DIRECTORY path holding
// a generation_config.json.
//
// A directory that does not exist, or holds no readable generation_config.json,
// yields an empty result rather than throwing, because upstream's loader
// returns {} on every failure path and never raises.
DefaultSamplingParams GetDiffSamplingParam(
    const HfConfig& config,
    const std::string& generation_config = kGenerationConfigAuto);

// The same narrowing over an already-read file, so a caller that has the six
// keys in hand does not have to reconstruct an HfConfig around them. This is
// where max_new_tokens becomes max_tokens.
DefaultSamplingParams NarrowGenerationConfigSampling(
    const GenerationConfigSampling& sampling);

}  // namespace vllm

#endif  // VLLM_CONFIG_GENERATION_H_
