// Ported from: vllm/v1/outputs.py @ e24d1b24
//
// The logprobs payload the V1 sampler produces. Upstream `LogprobsTensors` is a
// NamedTuple of three torch tensors (+ an optional cumulative-count list); the
// paired numpy form `LogprobsLists` is what crosses the process boundary. At T0
// there is no torch: each "tensor" is a flat host vector with explicit dims, the
// closest C++ analogue of the fixed [num_positions, k+1] / [num_positions] shapes.
//
// Field NAMES + shapes mirror upstream 1:1:
//   logprob_token_ids   [num_positions, num_tokens_per_position] int32
//   logprobs            [num_positions, num_tokens_per_position] float32
//   selected_token_ranks[num_positions]                          int32
// where num_tokens_per_position == max_num_logprobs + 1 (the sampled token plus
// the top-k). `num_positions == num_reqs x num_generated_tokens` (T0 non-spec
// decode: one position per request).
//
// DEFERRED upstream members, intentionally omitted (marked): the
// `cu_num_generated_tokens` slicing cursor (spec/jump decode — one position per
// req at T0), and the torch/device helpers tolists / to_cpu_nonblocking /
// filter / slice_request. The LogprobsLists numpy twin is represented by the
// same flat-vector layout (the C++ side never re-serializes to numpy), so it is
// not given a separate type here; add it when the OutputProcessor logprobs path
// (M1.8) needs the distinct numpy-vs-tensor split.
#ifndef VLLM_V1_OUTPUTS_H_
#define VLLM_V1_OUTPUTS_H_

#include <cstdint>
#include <vector>

namespace vllm::v1 {

// LogprobsTensors (vllm/v1/outputs.py): the (token_ids, logprobs, ranks) triple
// the sampler's gather_logprobs (M1.7 Task 4) fills. Flat row-major storage;
// element (pos, j) lives at index pos * num_tokens_per_position + j.
struct LogprobsTensors {
  // Leading dim: num_reqs x num_generated_tokens (T0 non-spec: num_reqs).
  int num_positions = 0;
  // Trailing dim: max_num_logprobs + 1 (sampled token + top-k).
  int num_tokens_per_position = 0;

  // [num_positions, num_tokens_per_position]
  std::vector<int32_t> logprob_token_ids;
  // [num_positions, num_tokens_per_position]
  std::vector<float> logprobs;
  // [num_positions]
  std::vector<int32_t> selected_token_ranks;

  // Upstream LogprobsTensors.empty_cpu: allocate the (uninitialized) buffers at
  // the given shape.
  static LogprobsTensors empty_cpu(int num_positions,
                                   int num_tokens_per_position);
};

}  // namespace vllm::v1

#endif  // VLLM_V1_OUTPUTS_H_
