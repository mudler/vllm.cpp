// Shared BF16 safetensors loader helpers for dense-transformer weight loaders.
//
// Extracted VERBATIM (behavior-preserving) from the anonymous namespace of
// qwen3_5_dense_weights.cpp so a second dense arch (Qwen3 `Qwen3ForCausalLM`,
// the first additive-model bring-up — qwen3_weights.cpp) reuses the exact same
// BF16 copy/transpose/merge routines rather than re-deriving them. The only
// change vs the qwen3_5-local originals is the diagnostic message prefix, which
// is generalized from "qwen3_5 dense:" to "dense loader:" (a shared helper must
// not name one arch); the LOADED bytes are byte-identical, so the qwen3_5 load
// result is unchanged (.agents/specs/first-additive-model-qwen3-dense.md §3b
// SEAM GAP #3).
//
// Helpers (all in `vllm::dense_loaders`):
//   MakeOwned            — allocate a zero-filled OwnedTensor of dtype+shape.
//   TransposeBf16        — bf16 [rows,cols] -> bf16 [cols,rows].
//   LoadBf16Direct       — copy a BF16 tensor verbatim (optionally reshaped).
//   LoadBf16Transposed   — BF16 [out,in] -> owned bf16 [in,out] (Matmul-B).
//   LoadMergedBf16RawNK  — concat BF16 torch-Linear shards [N_i,K] along output
//                          rows, kept RAW [N,K] with nk=true for vt::MatmulBT.
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // StTensor, MaybeReleaseSourcePages
#include "vllm/model_executor/models/qwen3_5_weights.h"           // OwnedTensor, TensorResolver
#include "vt/dtype.h"

namespace vllm {
namespace dense_loaders {

// Allocate a zero-filled owned host tensor of the given dtype and shape.
inline OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "dense loader: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
inline void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                          uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src_row[c];
  }
}

// BF16 tensor copied verbatim (optionally reshaped).
inline OwnedTensor LoadBf16Direct(const TensorResolver& get,
                                  const std::string& name,
                                  const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "dense loader: expected BF16 for " + name);
  std::vector<int64_t> shape = shape_override.empty() ? t.shape : shape_override;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "dense loader: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  // LOAD-SAFETENSORS: source range now copied-then-dead; drop its resident pages
  // so the owned mirror never double-resides with the mmap (spec §page-lifetime).
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// BF16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
inline OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                                      const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "dense loader: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 2, "dense loader: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(reinterpret_cast<const uint16_t*>(t.data), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Load and concatenate raw BF16 torch-Linear weights `[N_i,K]` along their
// output rows, preserving the exact listed order and setting `nk=true` for
// vt::MatmulBT (the cuBLASLt TN fast path). This is vLLM's physical ownership
// rule for MergedColumnParallelLinear/QKVParallelLinear (one merged param).
inline OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                       const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(),
           "dense loader: merged BF16 projection requires at least one shard");
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "BF16", "dense loader: expected BF16 for " + name);
    VT_CHECK(tensor.shape.size() == 2,
             "dense loader: expected 2-D weight for " + name);
    VT_CHECK(tensor.shape[0] > 0 && tensor.shape[1] > 0,
             "dense loader: merged BF16 shard has an empty dimension: " + name);
    VT_CHECK(tensor.data != nullptr,
             "dense loader: merged BF16 shard has null data: " + name);
    if (in_dim < 0) in_dim = tensor.shape[1];
    VT_CHECK(tensor.shape[1] == in_dim,
             "dense loader: merged BF16 shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "dense loader: merged BF16 output width overflow");
    out_dim += tensor.shape[0];
    shards.push_back(&tensor);
  }

  VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() / in_dim,
           "dense loader: merged BF16 element count overflow");
  const auto elements =
      static_cast<uint64_t>(out_dim) * static_cast<uint64_t>(in_dim);
  VT_CHECK(elements <= std::numeric_limits<size_t>::max() / sizeof(uint16_t),
           "dense loader: merged BF16 byte count overflow");
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {out_dim, in_dim});
  size_t offset = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const size_t expected = static_cast<size_t>(shard.shape[0]) *
                            static_cast<size_t>(in_dim) * sizeof(uint16_t);
    VT_CHECK(shard.nbytes == expected,
             "dense loader: byte-size mismatch for " + names[i]);
    std::memcpy(merged.bytes.data() + offset, shard.data, expected);
    MaybeReleaseSourcePages(shard.data, expected);
    offset += expected;
  }
  VT_CHECK(offset == merged.bytes.size(),
           "dense loader: merged BF16 byte accounting mismatch");
  merged.nk = true;
  return merged;
}

}  // namespace dense_loaders
}  // namespace vllm
