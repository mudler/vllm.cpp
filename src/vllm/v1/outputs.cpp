// Ported from: vllm/v1/outputs.py @ e24d1b24
// See include/vllm/v1/outputs.h for scope + the flat-vector layout.

#include "vllm/v1/outputs.h"

#include <cstddef>

namespace vllm::v1 {

LogprobsTensors LogprobsTensors::empty_cpu(int num_positions,
                                           int num_tokens_per_position) {
  LogprobsTensors out;
  out.num_positions = num_positions;
  out.num_tokens_per_position = num_tokens_per_position;
  const size_t area = static_cast<size_t>(num_positions) *
                      static_cast<size_t>(num_tokens_per_position);
  out.logprob_token_ids.resize(area);
  out.logprobs.resize(area);
  out.selected_token_ranks.resize(static_cast<size_t>(num_positions));
  return out;
}

}  // namespace vllm::v1
