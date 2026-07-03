// Ported from: vllm/config/scheduler.py @ e24d1b24
// See include/vllm/config/scheduler.h for the T0-subset / deferred-field /
// deviation notes.
#include "vllm/config/scheduler.h"

#include <stdexcept>
#include <string>

namespace vllm {

void SchedulerConfig::PostInit(int max_model_len_in, bool is_encoder_decoder) {
  max_model_len = max_model_len_in;

  if (is_encoder_decoder) {
    // Chunked prefill (and prefix caching) are disabled for encoder-decoder
    // models. disable_chunked_mm_input is deferred, so only the two T0 knobs
    // are touched here.
    enable_chunked_prefill = false;
    long_prefill_token_threshold = 0;
  }

  max_num_encoder_input_tokens = max_num_batched_tokens;
  encoder_cache_size = max_num_batched_tokens;

  // The max_num_partial_prefills > 1 branch (which derives
  // long_prefill_token_threshold = int(max_model_len * 0.04)) is DEFERRED —
  // partial prefills are a T1 field, so in T0 the threshold stays as given.

  VerifyMaxModelLen(max_model_len_in);
}

void SchedulerConfig::VerifyMaxModelLen(int max_model_len_in) const {
  if (max_num_batched_tokens < max_model_len_in && !enable_chunked_prefill) {
    throw std::invalid_argument(
        "max_num_batched_tokens (" + std::to_string(max_num_batched_tokens) +
        ") is smaller than max_model_len (" + std::to_string(max_model_len_in) +
        "). This effectively limits the maximum sequence length to "
        "max_num_batched_tokens and makes vLLM reject longer sequences. "
        "Please increase max_num_batched_tokens or decrease max_model_len.");
  }

  if (max_num_batched_tokens < max_num_seqs) {
    throw std::invalid_argument(
        "max_num_batched_tokens (" + std::to_string(max_num_batched_tokens) +
        ") must be greater than or equal to max_num_seqs (" +
        std::to_string(max_num_seqs) + ").");
  }

  // Upstream logs a warning when max_num_batched_tokens > max_num_seqs *
  // max_model_len; there is no logger here, so this is a no-op.

  // The max_num_partial_prefills > 1 and max_long_partial_prefills verify
  // branches are DEFERRED (T1) — no-ops while max_num_partial_prefills == 1.
}

}  // namespace vllm
