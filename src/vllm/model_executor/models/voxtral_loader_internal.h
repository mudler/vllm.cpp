// INTERNAL header (src/, not include/): the two Voxtral loader steps that read
// the mmap'd BF16 safetensors payload directly. Not part of the public ABI and
// deliberately not reachable from `include/vllm.h` — nothing outside this TU and
// its gate calls them.
//
// Exposed at all for the ODD-OFFSET loader gate (issue #772). A safetensors
// payload carries NO alignment guarantee — a tensor's first byte is
// `8 + <JSON header length> + <sum of the preceding tensors' sizes>` and not one
// of those terms is required to be even — so both of these must work over a BF16
// tensor that begins on an odd address. They cannot be reached through
// `LoadVoxtralWeights` at test scale: `VoxtralEncoderConfig()` is FIXED at 32
// layers of d_model 1280 / ffn 5120, so a synthetic checkpoint that satisfies it
// is ~1.2 GiB. Declaring them is what lets the gate drive the production code
// rather than a copy of it. Same shape as the internal resolver header
// `test_modelopt_mixed_precision` reaches through `-I${CMAKE_SOURCE_DIR}/src`.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // StTensor

namespace vllm {

// BF16 StTensor -> host f32 vector (encoder + adapter weights).
std::vector<float> VoxtralStBf16ToF32(const StTensor& t);

// Permute the rows of a BF16 [n_heads*head_dim, K] q/k weight from the
// Meta-interleaved rope layout (mistral consolidated) to the HF NeoX layout
// vLLM's rotary_emb (is_neox_style=True) expects.
std::vector<uint16_t> VoxtralPermuteQKBf16(const StTensor& t, int64_t n_heads);

}  // namespace vllm
