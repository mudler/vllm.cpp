// XOR (juspay/xor) decision model — head declarations (MODEL-XOR).
//
// xor is a 35B MoE from Qwen3.6-35B-A3B (~3B activated) with deterministic
// single-token candidate readout. This header declares the decision head
// structure and forward declarations for the text-only path.
//
// Ported from juspay/xor (spec .agents/specs/xor.md).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace xor_model {

// XOR uses deterministic single-token candidate readout: for each candidate
// option, the model reads the logit at a specific token position. The
// decision head is a lightweight projection (or direct lm_head readout).
//
// For the text-only Phase 1 path, we use the lm_head already present in the
// Qwen3.5 MoE weights: the last-token hidden is forwarded through lm_head to
// get vocab logits, and the candidate token IDs' logits are the scores.

struct XorHeadParams {
  int64_t hidden_size = 2048;  // Qwen3.6-35B-A3B hidden size
  float temperature = 1.0F;     // applied before softmax (default 1.0 = no scaling)
};

// Encode a decision question into the prompt format xor expects.
// Returns the full token IDs for the state+question prompt.
struct XorEncoded {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
};

}  // namespace xor_model
}  // namespace vllm
