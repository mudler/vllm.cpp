// ENG-EXPERT-STREAM W2. See gguf_expert_span.h for the layout contract and the
// reason this arithmetic is a tested unit rather than an inline expression.
#include "vllm/model_executor/model_loader/gguf_expert_span.h"

#include <stdexcept>
#include <string>

#include "vt/dtype.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

namespace vllm {

GgufExpertLayout GgufExpertLayoutOf(const GgufTensorInfo& tensor,
                                    int64_t num_experts) {
  if (num_experts <= 0) {
    throw std::invalid_argument("gguf expert span: num_experts must be > 0 for " +
                                tensor.name);
  }

  // The reader reverses ggml dims into torch row-major order, so a stacked
  // expert tensor arrives as [E, out, in] (rank 3) or already flattened to
  // [E*out, in] (rank 2). Both are accepted; anything else is not a stacked
  // expert weight and must not be sliced as one.
  int64_t rows = 0;
  int64_t k = 0;
  if (tensor.shape.size() == 3) {
    if (tensor.shape[0] != num_experts) {
      throw std::invalid_argument(
          "gguf expert span: leading dim " + std::to_string(tensor.shape[0]) +
          " != num_experts " + std::to_string(num_experts) + " for " + tensor.name);
    }
    rows = tensor.shape[0] * tensor.shape[1];
    k = tensor.shape[2];
  } else if (tensor.shape.size() == 2) {
    rows = tensor.shape[0];
    k = tensor.shape[1];
  } else {
    throw std::invalid_argument(
        "gguf expert span: rank " + std::to_string(tensor.shape.size()) +
        " is not a stacked expert weight for " + tensor.name);
  }

  if (rows % num_experts != 0) {
    // A non-divisible row count means the tensor is not E equal slices, so
    // every offset after the first would be wrong by a growing amount.
    throw std::invalid_argument(
        "gguf expert span: " + std::to_string(rows) + " rows is not divisible by " +
        std::to_string(num_experts) + " experts for " + tensor.name);
  }

  vt::DType dt = vt::DType::kF32;
  if (!KeepQuantDType(tensor.ggml_type, &dt)) {
    throw std::invalid_argument(
        "gguf expert span: ggml type " + std::to_string(tensor.ggml_type) +
        " has no keep-quant block encoding, so its rows have no fixed byte size, for " +
        tensor.name);
  }

  GgufExpertLayout out;
  out.num_experts = num_experts;
  out.rows_per_expert = rows / num_experts;
  out.k = k;
  // vt::RowSizeBytes throws when k is not a whole number of blocks (the
  // ggml_row_size contract), which is exactly the case where a byte offset
  // would cut a block in half. It throws vt's own runtime_error; this surface
  // promises std::invalid_argument for every geometry rejection, so the type is
  // normalised here and vt's message is preserved rather than replaced. One
  // exception type is what lets a caller write a single catch and still learn
  // which dimension was wrong.
  try {
    out.row_bytes = vt::RowSizeBytes(dt, k);
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string("gguf expert span: ") + e.what());
  }
  out.expert_bytes = static_cast<size_t>(out.rows_per_expert) * out.row_bytes;
  return out;
}

GgufExpertSpan GgufExpertSpanOf(const GgufTensorInfo& tensor,
                                const GgufExpertLayout& layout,
                                int64_t expert_index) {
  if (!layout.valid()) {
    throw std::invalid_argument("gguf expert span: invalid layout for " + tensor.name);
  }
  if (expert_index < 0 || expert_index >= layout.num_experts) {
    throw std::out_of_range(
        "gguf expert span: expert " + std::to_string(expert_index) + " outside [0, " +
        std::to_string(layout.num_experts) + ") for " + tensor.name);
  }

  const size_t begin = static_cast<size_t>(expert_index) * layout.expert_bytes;
  // The bounds check is the whole point: tensor.nbytes is the only thing
  // between a bad geometry and a read of unrelated mapped memory, which would
  // succeed and produce fluent output from the wrong weights.
  if (begin > tensor.nbytes || layout.expert_bytes > tensor.nbytes - begin) {
    throw std::invalid_argument(
        "gguf expert span: expert " + std::to_string(expert_index) + " span [" +
        std::to_string(begin) + ", " + std::to_string(begin + layout.expert_bytes) +
        ") exceeds the " + std::to_string(tensor.nbytes) + "-byte tensor " + tensor.name);
  }

  GgufExpertSpan out;
  out.data = tensor.data + begin;
  out.bytes = layout.expert_bytes;
  return out;
}

}  // namespace vllm
