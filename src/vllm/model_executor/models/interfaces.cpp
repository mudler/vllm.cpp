// Ported from: vllm/model_executor/models/interfaces.py:293 @ 5559679229bc.
// See include/vllm/model_executor/models/interfaces.h for the whole port map.
#include "vllm/model_executor/models/interfaces.h"

#include <string>

namespace vllm {

bool SkipTowerForModalities(const MultiModalConfig* mm_config,
                            std::initializer_list<std::string_view> modalities) {
  if (mm_config == nullptr) return false;
  if (modalities.size() == 0) return false;
  // `all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)`.
  for (const std::string_view modality : modalities) {
    if (mm_config->GetLimitPerPrompt(std::string(modality)) != 0) return false;
  }
  return true;
}

}  // namespace vllm
