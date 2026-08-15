// ENG-EXPERT-STREAM W2 (issue #912, spec .agents/specs/expert-streaming.md).
//
// The byte range of ONE routed expert inside a stacked GGUF expert tensor.
//
// WHY THIS IS ITS OWN UNIT. A wrong offset here does not fail: the read
// succeeds, the GEMM runs, and the model emits fluent text computed from
// another expert's weights. There is no error anywhere and no gate that would
// notice, because the tokens are still tokens. Arithmetic with that failure
// mode gets its own tested surface rather than living inline in a loader.
//
// THE LAYOUT, which `gguf_keep_quant.cpp:27` already states for the
// `kStackedExpertWeight` role: a stacked expert tensor is `[E, out, in]`, and
// "each expert slice is whole rows of the same K". So expert `e` owns rows
// `[e*out, (e+1)*out)` of an `[E*out, K]` matrix, every row is
// `vt::RowSizeBytes(dtype, K)` bytes, and no block is ever cut. That last part
// is what makes the slice a pure byte offset instead of a repack.
//
// This mirrors the arithmetic `OwnGgufQuantBlocks` already performs at
// `qwen3_5_gguf_weights.cpp:57` for the LOAD path, extracted so the STREAMING
// path cannot drift from it. Streaming asks the same question at a different
// time: the loader asks once per tensor, the stream asks per expert per step.
#ifndef VLLM_MODEL_EXECUTOR_MODEL_LOADER_GGUF_EXPERT_SPAN_H_
#define VLLM_MODEL_EXECUTOR_MODEL_LOADER_GGUF_EXPERT_SPAN_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"

namespace vllm {

// The geometry of a stacked expert tensor, read once per layer rather than
// recomputed per expert per step.
struct GgufExpertLayout {
  int64_t num_experts = 0;   // E
  int64_t rows_per_expert = 0;  // out features of one expert
  int64_t k = 0;             // in features, the block-quantised dimension
  size_t row_bytes = 0;      // vt::RowSizeBytes(dtype, k)
  size_t expert_bytes = 0;   // rows_per_expert * row_bytes

  bool valid() const {
    return num_experts > 0 && rows_per_expert > 0 && k > 0 && row_bytes > 0;
  }
};

// A borrowed view of one expert's bytes inside the mmap'd file. `data` points
// into the mapping and is valid only while the `GgufFile` lives.
struct GgufExpertSpan {
  const uint8_t* data = nullptr;
  size_t bytes = 0;
  bool valid() const { return data != nullptr && bytes > 0; }
};

// Derive the layout of `tensor` given the expert count.
//
// Throws std::invalid_argument when the tensor is not a 2-D or 3-D stacked
// expert weight, when `num_experts` does not divide its rows, or when K is not
// a whole number of quant blocks. Each of those would otherwise yield a span
// that reads the wrong bytes rather than failing.
GgufExpertLayout GgufExpertLayoutOf(const GgufTensorInfo& tensor,
                                    int64_t num_experts);

// The span of expert `expert_index`.
//
// Throws std::out_of_range when the index is outside `[0, num_experts)`, and
// std::invalid_argument when the computed span would leave the tensor. The
// bounds check is the point: `tensor.nbytes` is the only thing standing between
// a bad index and a read of unrelated mapped memory.
GgufExpertSpan GgufExpertSpanOf(const GgufTensorInfo& tensor,
                                const GgufExpertLayout& layout,
                                int64_t expert_index);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODEL_LOADER_GGUF_EXPERT_SPAN_H_
