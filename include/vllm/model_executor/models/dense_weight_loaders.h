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
#include <functional>
#include <limits>
#include <string>
#include <utility>
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

// Load and concatenate rank-1 BF16 tensors in the listed order — the vector
// analog of LoadMergedBf16RawNK, for merging the per-shard BIAS terms of a
// MergedColumnParallelLinear/QKVParallelLinear whose weights were merged by it.
// ADDED (append-only, no existing helper touched) by the OPT (`OPTForCausalLM`)
// bring-up: OPT is the first family we port whose projections carry bias
// (`config.enable_bias`, opt.py:90-104), so its merged qkv needs a merged
// [3*H] bias to go with the merged [3*H, H] weight. Generic — every future
// biased family (GPT-2, BLOOM, Falcon, ...) reuses it.
inline OwnedTensor LoadMergedBf16Vector(const TensorResolver& get,
                                        const std::vector<std::string>& names) {
  VT_CHECK(!names.empty(),
           "dense loader: merged BF16 vector requires at least one shard");
  int64_t total = 0;
  std::vector<const StTensor*> shards;
  shards.reserve(names.size());
  for (const std::string& name : names) {
    const StTensor& tensor = get(name);
    VT_CHECK(tensor.dtype == "BF16", "dense loader: expected BF16 for " + name);
    VT_CHECK(tensor.shape.size() == 1,
             "dense loader: expected 1-D vector for " + name);
    VT_CHECK(tensor.shape[0] > 0, "dense loader: merged BF16 vector shard is empty: " + name);
    VT_CHECK(tensor.data != nullptr,
             "dense loader: merged BF16 vector shard has null data: " + name);
    VT_CHECK(total <= std::numeric_limits<int64_t>::max() - tensor.shape[0],
             "dense loader: merged BF16 vector length overflow");
    total += tensor.shape[0];
    shards.push_back(&tensor);
  }
  OwnedTensor merged = MakeOwned(vt::DType::kBF16, {total});
  size_t offset = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const StTensor& shard = *shards[i];
    const size_t expected = static_cast<size_t>(shard.shape[0]) * sizeof(uint16_t);
    VT_CHECK(shard.nbytes == expected,
             "dense loader: byte-size mismatch for " + names[i]);
    std::memcpy(merged.bytes.data() + offset, shard.data, expected);
    MaybeReleaseSourcePages(shard.data, expected);
    offset += expected;
  }
  VT_CHECK(offset == merged.bytes.size(),
           "dense loader: merged BF16 vector byte accounting mismatch");
  return merged;
}

// --- compressed-tensors NVFP4 **W4A16** (`nvfp4-pack-quantized`) ------------
// ADDED (append-only; no existing helper touched) by the Qwen3-32B-NVFP4A16
// bring-up — the QUANT-SCHEME additivity row. These are the quantized analogs of
// LoadBf16Direct / LoadMergedBf16RawNK: same [N=out, K=in] raw orientation, same
// merged-shard ownership rule, but the payload is NVFP4 instead of BF16.
//
// ON-DISK LAYOUT (verified on RedHatAI/Qwen3-32B-NVFP4A16, and the format
// `compressed-tensors` emits for `nvfp4-pack-quantized`, group_size 16,
// num_bits 4, type float, strategy tensor_group, symmetric):
//   <proj>.weight_packed        U8       [N, K/2]    two E2M1 nibbles per byte
//   <proj>.weight_scale         F8_E4M3  [N, K/16]   one fp8 scale per 16 elems
//   <proj>.weight_global_scale  F32      [1]         per-tensor DIVISOR
// and — the discriminator that makes this W4A16 rather than W4A4 — there is NO
// `<proj>.input_global_scale`. That mirrors vLLM exactly: `input_activations`
// null in the config group selects `CompressedTensorsW4A4Fp4(use_a16=True)`
// (compressed_tensors.py:696-698), whose `create_weights` registers
// `input_global_scale` ONLY when `not use_a16`
// (compressed_tensors_w4a4_nvfp4.py:86-91).
//
// SCALE CONVENTION: compressed-tensors stores the global scale as a DIVISOR, so
// scale2 is its RECIPROCAL — vLLM takes it at
// compressed_tensors_w4a4_nvfp4.py:111-114
// (`weight_global_scale = 1.0 / layer.weight_global_scale.max()`). The exact
// on-disk divisor is retained in `weight_global_scale_inv` because a merged
// (qkv / gate_up) linear must take `max()` over the shards' DIVISORS *before*
// reciprocating — reconstructing a divisor from scale2 can lose a float ULP.
// `alpha` is deliberately left 0 so `Nvfp4Weight::IsTrueW4A4()` is false and the
// weight routes to the W4A16 (Marlin, bf16-activation) dispatcher.

// Read a per-tensor f32 scalar (the CT global scales are 1-element F32 tensors).
inline float ReadCtF32Scalar(const StTensor& t, const std::string& name) {
  VT_CHECK(t.data != nullptr && t.nbytes >= sizeof(float),
           "dense loader: scalar tensor too small for f32: " + name);
  float v = 0.0F;
  std::memcpy(&v, t.data, sizeof(float));
  return v;
}

// True when `proj` is stored as a compressed-tensors NVFP4 linear. This is the
// per-layer scheme probe: presence of `.weight_packed` means the config group
// matched this Linear (vLLM resolves the same thing through `find_matched_target`
// + the `ignore` list, compressed_tensors.py:868-880).
inline bool IsCtNvfp4Projection(
    const std::function<bool(const std::string&)>& has, const std::string& proj) {
  return has(proj + ".weight_packed");
}

// One compressed-tensors NVFP4 W4A16 Linear -> raw fp4-resident Nvfp4Weight in
// the on-disk [N=out, K=in] orientation the Marlin/naive W4A16 GEMMs read
// directly (no bf16 materialization). Rejects a W4A4 checkpoint outright: an
// `input_global_scale` here means the scheme is NOT weight-only and belongs to
// the fp4-activation path, which this dense helper does not implement.
inline Nvfp4Weight LoadCtNvfp4W4A16(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "dense loader: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "dense loader: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "dense loader: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "dense loader: expected F8_E4M3 weight_scale for " + proj);
  VT_CHECK(ws.shape.size() == 2 && ws.shape[0] == out_dim &&
               ws.shape[1] == in_dim / 16,
           "dense loader: weight_scale shape must be [N, K/16] for " + proj);
  VT_CHECK(!has(proj + ".input_global_scale"),
           "dense loader: " + proj +
               " carries input_global_scale (W4A4); the dense NVFP4 loader "
               "implements the WEIGHT-ONLY W4A16 scheme only");
  const float wgs_disk =
      ReadCtF32Scalar(get(proj + ".weight_global_scale"), proj);
  VT_CHECK(wgs_disk != 0.0F,
           "dense loader: zero weight_global_scale (divisor) for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs_disk;  // exact divisor, for merged linears
  r.scale2 = 1.0F / wgs_disk;            // CT stores a divisor -> reciprocate
  r.alpha = 0.0F;                        // W4A16: no activation quant
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(packed.nbytes == r.packed.bytes.size(),
           "dense loader: packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "dense loader: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  MaybeReleaseSourcePages(ws.data, ws.nbytes);
  return r;
}

// Load and concatenate compressed-tensors NVFP4 W4A16 shards `[N_i, K]` along
// their output rows — the NVFP4 analog of LoadMergedBf16RawNK, and vLLM's
// physical ownership rule for QKVParallelLinear / MergedColumnParallelLinear
// (ONE merged parameter, qwen3.py:271-274).
//
// `weight_packed` [N_i,K/2] and `weight_scale` [N_i,K/16] are both row-major
// over N, so both concatenate by plain row-stack (packing/grouping runs along K
// and is therefore untouched) — vLLM likewise fuses them by concat along
// output_dim=0 (compressed_tensors_w4a4_nvfp4.py:53-62,73-84).
//
// The GLOBAL scale is the one lossy part, and we mirror vLLM exactly: it keeps a
// PerTensorScaleParameter with one entry per shard and then collapses them with
// `.max()` before reciprocating (compressed_tensors_w4a4_nvfp4.py:111-114,
// warning at :101-108 when the shards disagree). We take the max over the on-disk
// DIVISORS and reciprocate once, which is the identical arithmetic without an
// intermediate rounding. (On RedHatAI/Qwen3-32B-NVFP4A16 the shards' divisors are
// bit-identical within every fused group, so the collapse is exactly lossless
// there; the max is kept for faithfulness on checkpoints where they differ.)
inline Nvfp4Weight LoadMergedCtNvfp4W4A16(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::vector<std::string>& projs) {
  VT_CHECK(!projs.empty(),
           "dense loader: merged NVFP4 projection requires at least one shard");
  std::vector<Nvfp4Weight> shards;
  shards.reserve(projs.size());
  int64_t in_dim = -1;
  int64_t out_dim = 0;
  for (const std::string& proj : projs) {
    Nvfp4Weight s = LoadCtNvfp4W4A16(get, has, proj);
    if (in_dim < 0) in_dim = s.k;
    VT_CHECK(s.k == in_dim,
             "dense loader: merged NVFP4 shards must share input width");
    VT_CHECK(out_dim <= std::numeric_limits<int64_t>::max() - s.n,
             "dense loader: merged NVFP4 output width overflow");
    out_dim += s.n;
    shards.push_back(std::move(s));
  }
  if (shards.size() == 1) return std::move(shards[0]);

  Nvfp4Weight merged;
  merged.n = out_dim;
  merged.k = in_dim;
  merged.alpha = 0.0F;
  // vLLM: weight_global_scale = 1.0 / max(per-shard on-disk global scales).
  float max_divisor = 0.0F;
  for (const Nvfp4Weight& s : shards)
    if (s.weight_global_scale_inv > max_divisor)
      max_divisor = s.weight_global_scale_inv;
  VT_CHECK(max_divisor != 0.0F,
           "dense loader: merged NVFP4 max weight_global_scale is zero");
  merged.weight_global_scale_inv = max_divisor;
  merged.scale2 = 1.0F / max_divisor;

  merged.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  merged.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  size_t p_off = 0;
  size_t s_off = 0;
  for (const Nvfp4Weight& s : shards) {
    std::memcpy(merged.packed.bytes.data() + p_off, s.packed.bytes.data(),
                s.packed.bytes.size());
    p_off += s.packed.bytes.size();
    std::memcpy(merged.scale.bytes.data() + s_off, s.scale.bytes.data(),
                s.scale.bytes.size());
    s_off += s.scale.bytes.size();
  }
  VT_CHECK(p_off == merged.packed.bytes.size() &&
               s_off == merged.scale.bytes.size(),
           "dense loader: merged NVFP4 byte accounting mismatch");
  return merged;
}

}  // namespace dense_loaders
}  // namespace vllm
