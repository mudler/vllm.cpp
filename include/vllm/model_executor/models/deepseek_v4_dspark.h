#pragma once

// DSV4-DSPARK-DRAFTER — the DSpark block drafter's ENTRY, as pure host code.
//
// Spec: `.agents/specs/dsv4-dspark-drafter.md`. Oracle: `exllamav3` @ the
// registered pin `2398c05635fbbad01a0a51dce63c85c6c8a8450e`
// (`.agents/oracles/exllamav3.md`); vLLM implements no DSpark, and its
// `nvidia/mtp.py` head is a DIFFERENT architecture.
//
// This is the lever the 44-47 tok/s target actually uses: that figure is measured
// WITH DSpark K5 speculative decoding at acceptance 0.65/0.44/0.31/0.17/0.07,
// about 2.64 accepted tokens per step.

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"

namespace vllm::dspark {

// THE TAP. `exllamav3/modules/transformer.py:198-203`:
//
//     x_ = x.mean(dim = 2) if self.attn_hc else x
//
// With hyperconnections the trunk residual is a STREAM STACK
// `[num_tokens, hc_mult, hidden]`, and the tap the drafter consumes is its mean
// over the STREAM dimension -- "streams start as broadcast copies of the
// embedding", so the mean is the collapsed hidden state. Taken AFTER the layer's
// residual add, and only on `layer_instance == 0` (`transformer.py:144`).
//
// Pinned in code rather than left to the caller because every neighbouring choice
// -- the raw stack, the pre-residual state, the final-norm output -- yields a
// drafter that runs and emits CORRECT tokens (verification is lossless) while
// drafting badly. There is no failing test for a wrong tap, only an acceptance
// rate nobody can account for.
//
//   x    [num_tokens, hc_mult, hidden], stream-major within a token
//   out  [num_tokens, hidden]
std::vector<float> StreamMeanTap(const std::vector<float>& x, int64_t num_tokens,
                                 int64_t hc_mult, int64_t hidden);

// THE ENTRY. `DSparkInputLayer.project_taps`
// (`exllamav3/modules/arch_specific/dspark.py:194-197`): concatenate the taps on
// the last axis, project once through `main_proj`, then `main_norm`.
//
// `main_proj` is `Linear(n_taps * hidden, hidden)` and
// `dspark_target_layer_ids = [40, 41, 42]` gives `n_taps = 3`, which is the
// artifact's measured `[4096, 12288]`. The projection is applied ONCE per set of
// taps, not once per block: all three blocks then read the same `dspark_main_x`.
//
//   taps  n_taps buffers, each [num_tokens, hidden], already stream-meaned
//   out   [num_tokens, hidden]
std::vector<float> ProjectTaps(const std::vector<std::vector<float>>& taps,
                               const DeepseekV4MtpTensorView& main_proj,
                               const std::vector<float>& main_norm_weight, float eps,
                               int64_t num_tokens, int64_t hidden);

}  // namespace vllm::dspark
