// vllm.cpp original; see qwen3_5_dense.h. Loader for the DENSE Qwen3.6-27B text
// gate (compressed-tensors NVFP4 W4A4). Mirrors the 35B loader structure
// (qwen3_5_weights.cpp) but routes each Linear bf16 vs W4A4-materialized-to-bf16
// by name (notes §3.6) and swaps the MoE block for the dense SwiGLU MLP.
#include "vllm/model_executor/models/qwen3_5_dense.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_config.h"
#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {

// The generic BF16 copy/transpose/merge routines now live in the shared
// dense_weight_loaders.h (SEAM GAP #3 extraction) so qwen3_weights.cpp reuses
// them. Re-expose them unqualified for the 27B call sites below — behavior and
// loaded bytes are byte-identical to the former anon-namespace copies.
using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::MakeOwned;
// The per-tensor scale read (#1181). The local copy this replaces checked
// neither the element count nor the dtype.
using dense_loaders::ReadF32Scalar;

namespace {

using TensorExists = std::function<bool(const std::string&)>;

// The GDN in-projections stay RAW in the on-disk torch Linear [out, in]
// orientation (nk=true, via LoadMergedBf16RawNK), consumed by vt::MatmulBT —
// the cuBLASLt TN fast path (K contiguous in both operands, the layout vLLM's
// F.linear hits; nvjet TNNN on GB10). Measured 2026-07-10: the transposed
// [K,N] Matmul-B orientation forces cuBLASLt onto NNNN/sm80-cutlass kernels —
// 27B GDN in_proj site 2.29 vs vLLM 1.80 us/tok. Raw NK also skips the
// load-time host transpose.

// Model-dtype vectors are BF16 in the NVFP4 gate checkpoint but F32 in ordinary
// Qwen3.5 dense checkpoints. Normalize them to the representation consumed by
// the existing forward without changing the quantized path.
OwnedTensor LoadModelBf16Direct(
    const TensorResolver& get, const std::string& name,
    const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  const std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): ONLY the BF16 arm is verbatim. The F32
  // arm below CONVERTS f32 -> bf16 and therefore can never borrow; the dtype
  // test is what keeps the two apart, and BorrowStTensorBytes independently
  // rejects the f32 source anyway (its span is twice the bf16 destination).
  OwnedTensor borrowed;
  if (t.dtype == "BF16" &&
      BorrowStTensorBytes(borrowed, t, vt::DType::kBF16, shape)) {
    return borrowed;
  }
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  if (t.dtype == "BF16") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(o.bytes.data(), t.data, t.nbytes);
  } else {
    VT_CHECK(t.nbytes == static_cast<size_t>(o.Numel()) * sizeof(float),
             "qwen3_5 dense: byte-size mismatch for " + name);
    auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
    const auto* src = static_cast<const uint8_t*>(t.data);
    for (int64_t i = 0; i < o.Numel(); ++i) {
      float value = 0.0F;
      std::memcpy(&value, src + static_cast<size_t>(i) * sizeof(value),
                  sizeof(value));
      dst[i] = vt::F32ToBF16(value);
    }
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

OwnedTensor LoadBf16RawNK(const TensorResolver& get,
                          const std::string& name) {
  OwnedTensor out = LoadBf16Direct(get, name);
  VT_CHECK(out.rank == 2,
           "qwen3_5 dense: expected 2-D weight for " + name);
  out.nk = true;
  return out;
}

// BF16/F32 [n] -> owned f32 [n] (A_log / dt_bias).
OwnedTensor LoadToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 dense: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  if (t.dtype == "F32") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(dst, t.data, t.nbytes);
  } else {
    // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(t.data + i * 2));
    }
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

bool DirectDeviceLoadEligible(vt::Queue* queue) {
  if (queue == nullptr ||
      !platforms::GetPlatform(queue->device.type).needs_weight_staging()) {
    return false;
  }
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  if (direct != nullptr && direct[0] == '0') return false;
  const auto& platform = platforms::GetPlatform(queue->device.type);
  return !platform.is_unified_memory() &&
         platform.residency_policy().release_host_weights_after_upload;
}

void StageAndReleaseLoadedDense(Qwen3_5DenseWeights& weights,
                                vt::Queue& queue) {
  Qwen3_5DenseModel::PrepareBf16Resident(weights, queue);
  vt::GetBackend(queue.device.type).Synchronize(queue);
  (void)ReleaseResidentQwen3_5DenseHostWeights(weights);
}

// VT_MODELOPT_W4A4 (default 0): consume a projection's on-disk activation
// divisor, setting `alpha` and so flipping `IsTrueW4A4()` to the fp4-ACTIVATION
// GEMM (docs/ENVIRONMENT.md). ONE reader, so the spellings cannot drift.
bool ModelOptW4A4OptIn() {
  const char* w4a4 = std::getenv("VT_MODELOPT_W4A4");
  return w4a4 != nullptr && w4a4[0] == '1';
}

// One compressed-tensors NVFP4 W4A4 Linear -> RAW fp4-resident Nvfp4Weight kept
// in the on-disk [N=out, K=in] orientation vt::MatmulNvfp4 reads directly (notes
// §5 step-6a — the throughput path; NO bf16 materialization). Reads the CT
// tensor names `<proj>.weight_packed` (U8 [out,in/2]), `<proj>.weight_scale`
// (F8_E4M3 [out,in/16], LINEAR non-swizzled) and `<proj>.weight_global_scale`
// (F32 scalar). The CT global scale is stored as a DIVISOR, so scale2 is its
// RECIPROCAL (notes §3.3) — the ONLY math delta vs the modelopt LoadNvfp4Raw;
// the byte encoding (E2M1 nibbles + fp8-e4m3 group-16 scale) is identical, so the
// existing M2.7 tensor-core GEMM carries these bit-for-bit as the CPU dequant
// reference (DequantCtNvfp4WeightToF32). Activation-quant is dropped (bf16
// activations, W4A16-style) — the on-disk `input_global_scale` is not read.
Nvfp4Weight LoadCtNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get, proj + ".weight_global_scale");
  VT_CHECK(wgs_disk != 0.0F,
           "qwen3_5 dense: zero weight_global_scale (divisor) for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs_disk;  // retain exact divisor for fused linears
  r.scale2 = 1.0F / wgs_disk;  // CT stores 1/scale (divisor) -> reciprocate
  // TRUE W4A4 (notes §7): the on-disk activation divisor drives vt::ScaledFp4Quant
  // (used DIRECTLY), and alpha folds both reciprocated globals for the fp4xfp4
  // GEMM: alpha = (1/input_divisor)·(1/weight_divisor). input_global_scale is a
  // per-tensor F32 scalar present on every 27B quantized Linear (§3.2).
  const float igs_disk = ReadF32Scalar(get, proj + ".input_global_scale");
  VT_CHECK(igs_disk != 0.0F,
           "qwen3_5 dense: zero input_global_scale (divisor) for " + proj);
  r.input_global_scale_inv = igs_disk;   // on-disk divisor, used directly
  r.alpha = r.scale2 * (1.0F / igs_disk);
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): both payloads are verbatim.
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8,
                           {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(),
             "qwen3_5 dense: packed byte-size mismatch for " + proj);
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
  if (!BorrowStTensorBytes(r.scale, ws, vt::DType::kI8,
                           {out_dim, in_dim / 16})) {
    r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
    VT_CHECK(ws.nbytes == r.scale.bytes.size(),
             "qwen3_5 dense: scale byte-size mismatch for " + proj);
    std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
    MaybeReleaseSourcePages(ws.data, ws.nbytes);
  }
  return r;
}

// --- lm_head dtype dispatch (issue #164) --------------------------------------
// The 27B NVFP4 publishers do NOT agree on the OUTPUT HEAD, and the head is not
// a compressed-tensors Linear, so none of the scheme probes above cover it:
//
//   BF16     `lm_head.weight` [V,H]                                (transpose)
//   F8_E4M3  `lm_head.weight` [V,H] + `.weight_scale` [V,1] or []  (per-row/scalar)
//   U8       `lm_head.weight` [V,H/2] + `.weight_scale` F8 [V,H/16]
//                                     + `.weight_scale_2` f32       (ModelOpt NVFP4)
//
// This loader was written against `unsloth/Qwen3.6-27B-NVFP4` @890bdef7, which
// ships a BF16 head — the snapshot every recorded 27B-NVFP4 benchmark ran on, so
// those numbers are unaffected by this change. @ccdaab7e later re-quantized the
// head to FP8 with a PER-OUTPUT-CHANNEL scale, and nvidia/Qwen3.6-27B-NVFP4 ships
// a ModelOpt NVFP4 head; both hit the old unconditional BF16 assert.
//
// BF16 and FP8 heads land on the SAME bf16 [in, out] Matmul-B operand the logits
// GEMM already consumes, so a BF16 head stays byte-exact. The NVFP4 form no
// longer reaches this function: LoadDenseLmHead routes it to the PACKED
// `lm_head_fp4` (PERF-27B-LMHEAD-FP4, issue #213), and the U8 branch below
// survives ONLY as the VT_LMHEAD_FP4=0 in-binary rollback.
//
// ModelOpt vs compressed-tensors global-scale convention: CT stores the value as
// a DIVISOR and `DequantCtNvfp4WeightToF32` reciprocates it internally, whereas
// ModelOpt's `weight_scale_2` IS the scale (qwen3_5_weights.cpp:272 assigns it to
// `scale2` directly). Passing `1/weight_scale_2` as the "disk divisor" makes the
// shared CT dequant compute the ModelOpt scale exactly.
}  // namespace

OwnedTensor LoadLmHeadAnyDtype(const TensorResolver& get, const TensorExists& has,
                               const std::string& name) {
  const StTensor& w = get(name);
  VT_CHECK(w.shape.size() == 2, "qwen3_5 dense: expected 2-D weight for " + name);

  if (w.dtype == "BF16") {
    return LoadBf16Transposed(get, name);  // unchanged byte-for-byte
  }

  if (w.dtype == "F8_E4M3") {
    const int64_t out_dim = w.shape[0];
    const int64_t in_dim = w.shape[1];
    // Per-output-channel [V,1] (unsloth @ccdaab7e) or a single per-tensor scalar.
    // Stored BF16 there, F32 elsewhere; normalize both to f32 rows.
    std::vector<float> row_scale(static_cast<size_t>(out_dim), 1.0F);
    VT_CHECK(has(name + "_scale"),
             "qwen3_5 dense: FP8 " + name + " requires " + name + "_scale");
    const StTensor& sc = get(name + "_scale");
    const int64_t n_scale =
        static_cast<int64_t>(sc.nbytes) / (sc.dtype == "BF16" ? 2 : 4);
    VT_CHECK(n_scale == out_dim || n_scale == 1,
             "qwen3_5 dense: " + name + "_scale must be per-tensor or [out,1]");
    for (int64_t r = 0; r < out_dim; ++r) {
      const int64_t i = (n_scale == 1) ? 0 : r;
      if (sc.dtype == "BF16") {
        uint16_t h = 0;
        std::memcpy(&h, static_cast<const uint8_t*>(sc.data) + i * 2, 2);
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        std::memcpy(&row_scale[static_cast<size_t>(r)], &bits, sizeof(float));
      } else {
        std::memcpy(&row_scale[static_cast<size_t>(r)],
                    static_cast<const uint8_t*>(sc.data) + i * 4, sizeof(float));
      }
    }
    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    for (int64_t r = 0; r < out_dim; ++r) {
      DequantFp8ToBf16(static_cast<const uint8_t*>(w.data) + r * in_dim,
                       row_scale[static_cast<size_t>(r)], in_dim,
                       dq.data() + static_cast<size_t>(r) * in_dim);
    }
    MaybeReleaseSourcePages(w.data, w.nbytes);
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    dense_loaders::TransposeBf16(dq.data(), out_dim, in_dim,
                                 reinterpret_cast<uint16_t*>(o.bytes.data()));
    return o;
  }

  if (w.dtype == "U8") {
    const int64_t out_dim = w.shape[0];
    const int64_t in_dim = w.shape[1] * 2;
    VT_CHECK(in_dim % 16 == 0,
             "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + name);
    const StTensor& ws = get(name + "_scale");
    VT_CHECK(ws.dtype == "F8_E4M3",
             "qwen3_5 dense: expected F8_E4M3 " + name + "_scale");
    // ModelOpt spells the global scale `weight_scale_2`; compressed-tensors
    // spells it `weight_global_scale` and stores the reciprocal.
    float disk_divisor = 0.0F;
    if (has(name + "_scale_2")) {
      const float ws2 = ReadF32Scalar(get, name + "_scale_2");
      VT_CHECK(ws2 != 0.0F, "qwen3_5 dense: zero " + name + "_scale_2");
      disk_divisor = 1.0F / ws2;  // ModelOpt scale -> CT divisor convention
    } else {
      VT_CHECK(has(name + "_global_scale"),
               "qwen3_5 dense: NVFP4 " + name + " requires " + name +
                   "_scale_2 (ModelOpt) or " + name + "_global_scale (CT)");
      disk_divisor = ReadF32Scalar(get, name + "_global_scale");
      VT_CHECK(disk_divisor != 0.0F,
               "qwen3_5 dense: zero " + name + "_global_scale (divisor)");
    }
    std::vector<float> f32(static_cast<size_t>(out_dim) * in_dim);
    DequantCtNvfp4WeightToF32(static_cast<const uint8_t*>(w.data),
                              static_cast<const uint8_t*>(ws.data), disk_divisor,
                              out_dim, in_dim, f32.data());
    MaybeReleaseSourcePages(w.data, w.nbytes);
    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    for (size_t i = 0; i < f32.size(); ++i) {
      uint32_t bits = 0;
      std::memcpy(&bits, &f32[i], sizeof(bits));
      // round-to-nearest-even f32 -> bf16, matching DequantFp8ToBf16.
      const uint32_t lsb = (bits >> 16) & 1U;
      bits += 0x7FFFU + lsb;
      dq[i] = static_cast<uint16_t>(bits >> 16);
    }
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    dense_loaders::TransposeBf16(dq.data(), out_dim, in_dim,
                                 reinterpret_cast<uint16_t*>(o.bytes.data()));
    return o;
  }

  VT_CHECK(false, "qwen3_5 dense: unsupported dtype '" + w.dtype + "' for " +
                      name + "; supported: BF16, F8_E4M3 (+_scale), "
                      "U8 NVFP4 (+_scale and _scale_2/_global_scale)");
  return OwnedTensor{};
}

namespace {

// compressed-tensors and ModelOpt both ship NVFP4, with different names AND a
// different global-scale convention:
//
//   compressed-tensors  <proj>.weight_packed U8 + .weight_scale F8
//                       + .weight_global_scale F32 (a DIVISOR)
//   ModelOpt            <proj>.weight        U8 + .weight_scale F8
//                       + .weight_scale_2    F32 (the SCALE itself)
//
// nvidia/Qwen3.6-27B-NVFP4 is ModelOpt, so every `has(<proj>.weight_packed)`
// probe missed it and the whole tower fell through to the BF16 path and died at
// the first U8 tensor. LoadCtNvfp4Raw already reciprocates internally, so the
// ModelOpt scale is passed through as 1/weight_scale_2 to land on the same math
// (identical to the lm_head conversion in LoadLmHeadAnyDtype).
bool IsNvfp4Projection(const TensorExists& has, const std::string& proj) {
  return has(proj + ".weight_packed") || has(proj + ".weight_scale_2");
}

Nvfp4Weight LoadNvfp4AnyNaming(const TensorResolver& get, const TensorExists& has,
                               const std::string& proj) {
  if (has(proj + ".weight_packed")) return LoadCtNvfp4Raw(get, proj);

  const StTensor& packed = get(proj + ".weight");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 ModelOpt weight for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D ModelOpt weight for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float ws2 = ReadF32Scalar(get, proj + ".weight_scale_2");
  VT_CHECK(ws2 != 0.0F, "qwen3_5 dense: zero weight_scale_2 for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.scale2 = ws2;                       // ModelOpt stores the scale directly
  r.weight_global_scale_inv = 1.0F / ws2;  // the CT-convention divisor
  // W4A16 unless the checkpoint also carries an activation scale. ModelOpt
  // spells it `input_scale`; leaving alpha at 0 keeps IsTrueW4A4() false so the
  // weight routes to the W4A16 dispatcher, matching vLLM's use_a16 branch.
  // A/B(VT_MODELOPT_W4A4=1): default W4A16. ModelOpt ships `input_scale` on every
  // projection, but consuming it flips IsTrueW4A4() and routes to the
  // fp4-activation GEMM; leaving alpha at 0 keeps the weight-only dispatcher.
  const bool w4a4_opt_in = ModelOptW4A4OptIn();
  if (w4a4_opt_in && has(proj + ".input_scale")) {
    const float is = ReadF32Scalar(get, proj + ".input_scale");
    if (is != 0.0F) {
      r.input_global_scale_inv = is;
      r.alpha = r.scale2 * (1.0F / is);
    }
  }
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(packed.nbytes == r.packed.bytes.size(),
           "qwen3_5 dense: ModelOpt packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 dense: ModelOpt scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  MaybeReleaseSourcePages(ws.data, ws.nbytes);
  return r;
}

// MODEL-FP8-BLOCK-WEIGHT (#1189 M3), spec
// `.agents/specs/model-fp8-block-weight.md`. Does this projection take the
// block-wise (fine-grained 128x128) FP8 arm?
//
// READ THE CONFIG AND THE TENSORS, AND REFUSE WHEN THEY DISAGREE. A dtype probe
// alone answers "which arm" and can never answer "do these two sources agree",
// and the disagreement is where a silent wrong-scale bug lives: #1166 measured a
// `[96, 40]` block grid passing a per-tensor scale reader's byte floor and being
// applied to the whole `[N, K]` weight, stopped only by the tensor NAME. The
// checkpoint's `modules_to_not_convert` is the other half -- 882 entries on
// `Qwen/Qwen3.8-27B-FP8` @`017b9c7a`, 636 of them outside the vision tower --
// which a probe reproduces only by accident (#1614; the "~400" this line
// carried was wrong by more than 2.2x however the list is sliced).
//
// Six combinations, four of them refused by name; the table is in the spec.
bool IsFp8BlockProjection(const TensorExists& has, const std::string& proj,
                          const std::string& weight_dtype,
                          const Fp8BlockQuantConfig& block) {
  const bool has_scale_inv = has(proj + ".weight_scale_inv");
  const bool excluded = block.ExcludesModule(proj);

  if (!block.block_quant) {
    VT_CHECK(!has_scale_inv,
             "qwen3_5 dense: '" + proj +
                 ".weight_scale_inv' is present, which is the block-wise "
                 "(fine-grained) FP8 scale, but the checkpoint's "
                 "quantization_config declares no weight_block_size. The "
                 "tensors and the config disagree and there is no block "
                 "geometry to read the scale with; refusing rather than "
                 "guessing 128x128");
    return false;
  }

  if (excluded) {
    VT_CHECK(!has_scale_inv,
             "qwen3_5 dense: '" + proj +
                 "' is listed in quantization_config.modules_to_not_convert, "
                 "so it must be unquantized, yet it ships '" + proj +
                 ".weight_scale_inv'. The tensors and the config disagree; "
                 "refusing rather than picking one of them");
    return false;
  }

  if (weight_dtype == "F8_E4M3") {
    VT_CHECK(has_scale_inv,
             "qwen3_5 dense: the checkpoint declares "
             "quantization_config.weight_block_size and '" +
                 proj +
                 ".weight' is F8_E4M3, but '" + proj +
                 ".weight_scale_inv' is missing. Block-wise (fine-grained) FP8 "
                 "stores its scale under that name and only that name "
                 "(vllm fp8.py:378-379,511), so this projection cannot be "
                 "dequantized. Either the module belongs in "
                 "modules_to_not_convert or the shard is incomplete");
  }
  if (!has_scale_inv) return false;

  // An `input_scale` cannot coexist with a dynamic activation scheme: upstream
  // registers one only when `act_q_static` (`fp8.py:381-384`), which block quant
  // asserts against outright (`fp8.py:367`). `Qwen/Qwen3.8-27B-FP8` ships zero.
  VT_CHECK(!has(proj + ".input_scale"),
           "qwen3_5 dense: '" + proj +
               ".input_scale' is present beside a block-wise (fine-grained) "
               "FP8 weight whose quantization_config declares "
               "activation_scheme \"dynamic\". A dynamic scheme quantizes "
               "activations at run time and has no static input scale; the "
               "tensors and the config disagree");
  return true;
}

GdnLayerWeights LoadGdnDense(const TensorResolver& get, const TensorExists& has,
                             const std::string& base,
                             const Fp8BlockQuantConfig& block) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // in_proj_{qkv,z,a,b}: bf16 (ignore list, notes §3.6). Kept raw [N,K]
  // (nk=true -> vt::MatmulBT TN fast path). Mirror vLLM's TWO physical
  // MergedColumnParallelLinear owners (qwen3_5.py:203-210 stacked mapping +
  // qwen_gdn_linear_attn.py:481-496): in_proj_qkvz in exact [q,k,v,z] row
  // order (the checkpoint's in_proj_qkv already stacks q|k|v; z appended) and
  // in_proj_ba in exact [b,a] row order. The rollback paths take non-owning
  // row slices of these owners, so the split fields deliberately stay empty.
  // FP8 checkpoints (nvidia/Qwen3.6-27B-NVFP4 is `modelopt_mixed`) keep the GDN
  // shards NATIVE. Merging forces one dtype, and dequantizing fp8 -> bf16 DOUBLES
  // this tower's resident bytes (6.72 -> 13.44 GiB measured), which both slows
  // decode and starves the KV pool. `ProjectGdnQkvz` already carries the
  // separate-fp8 arm the 35B runs and selects it when the merged owner is empty.
  const StTensor& qkv_probe = get(la + "in_proj_qkv.weight");
  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): the block-wise rung goes BEFORE the
  // per-tensor one at every site that probes `F8_E4M3`, because a block-wise
  // weight IS `F8_E4M3` and fell into the per-tensor arm (#1166).
  if (IsFp8BlockProjection(has, la + "in_proj_qkv", qkv_probe.dtype, block)) {
    g.in_proj_qkv_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "in_proj_qkv", block.block_n, block.block_k);
    g.in_proj_z_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "in_proj_z", block.block_n, block.block_k);
  } else if (qkv_probe.dtype == "F8_E4M3") {
    g.in_proj_qkv_fp8 = LoadFp8RawShared(get, la + "in_proj_qkv");
    g.in_proj_z_fp8 = LoadFp8RawShared(get, la + "in_proj_z");
  } else {
    g.in_proj_qkvz = LoadMergedBf16RawNK(
        get, {la + "in_proj_qkv.weight", la + "in_proj_z.weight"});
  }
  g.in_proj_ba = LoadMergedBf16RawNK(
      get, {la + "in_proj_b.weight", la + "in_proj_a.weight"});
  // NVFP4 checkpoints use compressed tensors; ordinary checkpoints use raw
  // torch-Linear BF16 [N,K].
  if (IsNvfp4Projection(has, la + "out_proj")) {
    g.out_proj_fp4 = LoadNvfp4AnyNaming(get, has, la + "out_proj");
  } else if (IsFp8BlockProjection(has, la + "out_proj",
                                  get(la + "out_proj.weight").dtype, block)) {
    g.out_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "out_proj", block.block_n, block.block_k);
  } else if (get(la + "out_proj.weight").dtype == "F8_E4M3") {
    // Same rule as the in_proj shards above, and for the same measured reason:
    // the bf16 arm dequantizes this tower and then runs it as a cuBLAS `gemvx`,
    // which the decode profile shows costing far more than the bytes justify.
    // `ProjectGdnOut` already selects `out_proj_fp8` when it is populated.
    g.out_proj_fp8 = LoadFp8RawShared(get, la + "out_proj");
  } else {
    g.out_proj = LoadBf16RawNK(get, la + "out_proj.weight");
  }
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 dense: unexpected conv1d shape");
  g.conv1d_weight =
      LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadToF32(get, la + "A_log");
  g.dt_bias = LoadToF32(get, la + "dt_bias");
  g.norm_weight = LoadModelBf16Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttnDense(const TensorResolver& get,
                                   const TensorExists& has,
                                   const std::string& base,
                                   const Fp8BlockQuantConfig& block) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // Three forms, not two. `modelopt_mixed` checkpoints quantize this tower to
  // FP8 W8A8 while leaving the MLP NVFP4, and a projection that matches neither
  // the NVFP4 probe nor an FP8 dtype is genuinely BF16. Without the middle
  // branch an FP8 tower fell through to `LoadBf16RawNK`, which dequantizes it:
  // 1.562 GiB of FP8 became 3.12 GiB of BF16 re-read every decode step and
  // executed as cuBLAS `gemvx`. The `*_fp8` slots and their `MatmulFp8Cutlass*`
  // consumers already exist and are what the 35B runs.
  // FOUR forms. MODEL-FP8-BLOCK-WEIGHT (#1189 M3) inserts the block-wise rung
  // BEFORE the per-tensor one, because a block-wise weight is also `F8_E4M3`
  // and therefore entered the per-tensor arm, which then asked for a
  // `weight_scale` the checkpoint spells `weight_scale_inv` (#1166).
  const auto load_projection = [&](const std::string& name, Nvfp4Weight& fp4,
                                   Fp8BlockWeight& fp8_block, Fp8Weight& fp8,
                                   OwnedTensor& plain) {
    if (IsNvfp4Projection(has, name)) {
      fp4 = LoadNvfp4AnyNaming(get, has, name);
      return;
    }
    const std::string dtype = get(name + ".weight").dtype;
    if (IsFp8BlockProjection(has, name, dtype, block)) {
      fp8_block = dense_loaders::LoadFp8BlockRaw(get, name, block.block_n,
                                                 block.block_k);
    } else if (dtype == "F8_E4M3") {
      fp8 = LoadFp8RawShared(get, name);
    } else {
      plain = LoadBf16RawNK(get, name + ".weight");
    }
  };
  load_projection(sa + "q_proj", a.q_proj_fp4, a.q_proj_fp8_block, a.q_proj_fp8,
                  a.q_proj);
  load_projection(sa + "k_proj", a.k_proj_fp4, a.k_proj_fp8_block, a.k_proj_fp8,
                  a.k_proj);
  load_projection(sa + "v_proj", a.v_proj_fp4, a.v_proj_fp8_block, a.v_proj_fp8,
                  a.v_proj);
  load_projection(sa + "o_proj", a.o_proj_fp4, a.o_proj_fp8_block, a.o_proj_fp8,
                  a.o_proj);
  a.q_norm = LoadModelBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadModelBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// Dense SwiGLU MLP: gate/up/down all W4A4-quantized -> fp4-resident (§5 6a).
DenseMlpWeights LoadDenseMlp(const TensorResolver& get, const TensorExists& has,
                             const std::string& base,
                             const Fp8BlockQuantConfig& block) {
  const std::string mlp = base + "mlp.";
  DenseMlpWeights m;
  if (IsNvfp4Projection(has, mlp + "gate_proj")) {
    m.gate_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "gate_proj");
    m.up_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "up_proj");
    m.down_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "down_proj");
  } else if (IsFp8BlockProjection(has, mlp + "gate_proj",
                                  get(mlp + "gate_proj.weight").dtype, block)) {
    // MODEL-FP8-BLOCK-WEIGHT (#1189 M3). This block had NO fp8 rung at all, so
    // a block-wise MLP fell through to `LoadMergedBf16RawNK` and died on
    // "expected BF16". Loaded UNMERGED: block scales concatenate losslessly
    // along N, so merging gate+up is simpler here than in the per-tensor case
    // and #1189 M6 owns it rather than this row guessing at the layout.
    m.gate_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "gate_proj", block.block_n, block.block_k);
    m.up_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "up_proj", block.block_n, block.block_k);
    m.down_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "down_proj", block.block_n, block.block_k);
  } else {
    m.gate_up_proj = dense_loaders::LoadMergedBf16RawNK(
        get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    m.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
  }
  return m;
}

}  // namespace

bool DenseLmHeadFp4Enabled() {
  const char* v = std::getenv("VT_LMHEAD_FP4");
  return v == nullptr || v[0] != '0';
}

void LoadDenseLmHead(const TensorResolver& get, const TensorExists& has,
                     const std::string& proj, OwnedTensor& bf16_out,
                     Nvfp4Weight& fp4_out) {
  fp4_out = Nvfp4Weight{};
  bf16_out = OwnedTensor{};
  // PERF-27B-LMHEAD-FP4 (issue #213). An NVFP4 head stays PACKED, through the SAME
  // LoadNvfp4AnyNaming every other NVFP4 projection takes, so the ModelOpt vs
  // compressed-tensors global-scale convention is handled in exactly one place.
  // vLLM's mixed scheme likewise resolves a quantized head (modelopt.py:2491-2496).
  if (DenseLmHeadFp4Enabled() && IsNvfp4Projection(has, proj)) {
    fp4_out = LoadNvfp4AnyNaming(get, has, proj);
    // The head is W4A16, whatever the naming. `LoadNvfp4AnyNaming` decides
    // activation-quant per SPELLING — the ModelOpt arm ignores `input_scale`
    // unless VT_MODELOPT_W4A4=1, but `LoadCtNvfp4Raw` consumes
    // `input_global_scale` UNCONDITIONALLY, correct for a TOWER projection of the
    // 27B compressed-tensors checkpoint, which really is W4A4. An output head is
    // not one: vLLM resolves it through ModelOptNvFp4W4A16LinearMethod, which
    // DELETES input_scale (modelopt.py:1365; registered at :1358) and pins
    // MarlinNvFp4LinearKernel (modelopt.py:1249,1283-1284). A set alpha would (a)
    // take the fp4-activation GEMM vLLM refuses here and (b) make
    // PrepareLmHeadResident early-return, skipping the pre-capture Marlin build.
    if (!ModelOptW4A4OptIn()) {
      fp4_out.input_global_scale_inv = 0.0F;
      fp4_out.alpha = 0.0F;
    }
    // The ONE weight that opts into a lifetime dequant-B resident where there is
    // no fp4 GEMM (qwen3_5_weights.h): the head is re-read whole every step.
    fp4_out.keep_dequant_b = true;
    return;
  }
  bf16_out = LoadLmHeadAnyDtype(get, has, proj + ".weight");
}

bool DenseCheckpointHasLmHead(const TensorExists& has, const std::string& proj) {
  // A bare `<proj>.weight` probe misses a compressed-tensors head, whose only
  // weight tensor is `<proj>.weight_packed`; no head at all means
  // `tie_word_embeddings`, so missing CT ties the logits to the embedding table.
  return has(proj + ".weight") || IsNvfp4Projection(has, proj);
}

OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                const std::vector<std::string>& names) {
  // Extracted to the shared dense_weight_loaders.h (SEAM GAP #3); this retains
  // the public vllm::LoadMergedBf16RawNK API (used by the 27B GDN loader below
  // and test_qwen27_dense_forward) as a byte-identical thin forward.
  return dense_loaders::LoadMergedBf16RawNK(get, names);
}

GdnLayerWeights LoadQwen3_5DenseGdn(const TensorResolver& get,
                                    const std::string& layer_base) {
  // Public focused-loader seam historically describes the 27B checkpoint, which
  // is NVFP4 and declares no `weight_block_size`, so the default-constructed
  // block config here is the truthful one and the routing is unchanged.
  //
  // `has` ASKS THE RESOLVER (#1256). It used to answer `true` for every name,
  // which was harmless while the only caller of `has` was a dtype probe that
  // went on to `get` the tensor anyway and would throw on a name that was not
  // there. M3's config/tensor cross-check made it harmful: it asks
  // `has(proj + ".weight_scale_inv")` WITHOUT then fetching it, so a stub that
  // says yes to everything reports a block-wise scale on a checkpoint that has
  // none, and `IsFp8BlockProjection` refuses the disagreement it just invented.
  // `TensorResolver` throws on a missing tensor, so probing it is the honest
  // answer and the only one available at this seam.
  //
  // FIX-PROBE-CANNOT-SAY-NO (#1258): the `try`/`catch` this used to spell out
  // now lives ONCE in `dense_loaders`, because the sibling seam below needs the
  // identical probe and two copies of a subtle thing is the next defect.
  const TensorExists has = dense_loaders::ProbeThroughResolver(get);
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5DenseGdn");
  return LoadGdnDense(get, has, layer_base, Fp8BlockQuantConfig{});
}

bool IsQwen27QuantizedLinear(const std::string& name) {
  // Never quantized regardless of suffix (notes §3.6 `ignore`).
  if (name.rfind("mtp.", 0) == 0) return false;
  if (name.find("model.visual.") != std::string::npos) return false;
  if (name.find(".linear_attn.in_proj_") != std::string::npos) return false;
  if (name.find("lm_head") != std::string::npos) return false;
  // Quantized set: dense-MLP gate/up/down, self_attn q/k/v/o, GDN out_proj.
  auto ends_with = [&name](const char* suf) {
    const std::string s(suf);
    return name.size() >= s.size() &&
           name.compare(name.size() - s.size(), s.size(), s) == 0;
  };
  if (name.find(".mlp.") != std::string::npos &&
      (ends_with(".gate_proj") || ends_with(".up_proj") ||
       ends_with(".down_proj")))
    return true;
  if (name.find(".self_attn.") != std::string::npos &&
      (ends_with(".q_proj") || ends_with(".k_proj") || ends_with(".v_proj") ||
       ends_with(".o_proj")))
    return true;
  if (ends_with(".linear_attn.out_proj")) return true;
  return false;
}

OwnedTensor MaterializeCtNvfp4Bf16Transposed(const TensorResolver& get,
                                             const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& wscale = get(proj + ".weight_scale");
  VT_CHECK(wscale.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get, proj + ".weight_global_scale");

  // Dequant to f32 [out, in] (the divisor is reciprocated inside), then round to
  // bf16 while transposing to Matmul-B layout [in, out].
  std::vector<float> f32(static_cast<size_t>(out_dim) * in_dim);
  DequantCtNvfp4WeightToF32(reinterpret_cast<const uint8_t*>(packed.data),
                            reinterpret_cast<const uint8_t*>(wscale.data),
                            wgs_disk, out_dim, in_dim, f32.data());
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  MaybeReleaseSourcePages(wscale.data, wscale.nbytes);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      dst[c * out_dim + r] =
          vt::F32ToBF16(f32[static_cast<size_t>(r) * in_dim + c]);
  return o;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const TensorExists& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix, const Fp8BlockQuantConfig& block) {
  // FIX-PROBE-CANNOT-SAY-NO (#1258): every dense-layer path funnels through this
  // overload, so one check here covers the production loader, both resolver-only
  // seams, and whatever probe a caller supplies next.
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5DenseLayer");
  const std::string base =
      backbone_prefix + "layers." + std::to_string(layer_idx) + ".";
  Qwen3_5DenseLayerWeights layer;
  layer.input_layernorm =
      LoadModelBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadModelBf16Direct(get, base + "post_attention_layernorm.weight");
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdnDense(get, has, base, block);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttnDense(get, has, base, block);
  } else {
    VT_CHECK(false, "qwen3_5 dense: unknown layer_type " + layer_type);
  }
  layer.mlp = LoadDenseMlp(get, has, base, block);
  return layer;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const TensorExists& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix) {
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx, backbone_prefix,
                               Fp8BlockQuantConfig{});
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix) {
  // FIX-PROBE-CANNOT-SAY-NO (#1258). This was the SECOND always-true stub named
  // by #1256; #1257 fixed only the sibling above. It was not latent: the
  // `qwen36_gdn_layer_27b` isolated-layer golden replays through THIS overload
  // (`tests/parity/test_op_parity.cpp`) with `layer_type == "linear_attention"`,
  // which routes to `LoadGdnDense`, whose first act is the
  // `IsFp8BlockProjection` cross-check a constant `true` lies to. That leg is
  // SKIPped without the pinned 27B snapshot, which is why no CPU lane saw it.
  //
  // The compressed-tensors parity fixture this seam was written for really does
  // carry `weight_packed` on every routed projection, so a truthful probe leaves
  // its routing where it was and merely stops inventing the tensors it lacks.
  const TensorExists has = dense_loaders::ProbeThroughResolver(get);
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx,
                               backbone_prefix);
}

Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config,
                                     vt::Queue* load_queue) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  std::vector<std::string> all_names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      where[name] = &shard;
      all_names.push_back(name);
    }
  }
  // ONE namespace decision for the whole checkpoint (qwen3_5_weights.h): the
  // VL-nested spelling for the wrappers we gate, the flat `model.` spelling for
  // a text-only arm, and a refusal for a mixed index.
  const std::string backbone = ResolveQwen3_5BackbonePrefix(all_names);
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3_5 dense: tensor not found: " + name);
    return it->second->Get(name);
  };
  const TensorExists has = [&where](const std::string& name) {
    return where.find(name) != where.end();
  };
  // FIX-PROBE-CANNOT-SAY-NO (#1258). Truthful by construction today; checked
  // anyway, because the guard's value is that it does not depend on how the
  // probe was built — a name index filled from the wrong shard list fails here
  // too.
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5Dense");

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 dense: layer_types size must equal num_hidden_layers");

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): ONE read of the quantization config for
  // the whole checkpoint, validated here rather than per projection. It carries
  // the block geometry the rung needs and the `modules_to_not_convert` list the
  // rung cross-checks against, which a per-tensor dtype probe cannot supply.
  // Default-constructed (`block_quant == false`) on every non-block checkpoint,
  // which leaves the routing below byte-identical to before this row.
  const Fp8BlockQuantConfig block = ReadFp8BlockQuantConfig(config);

  // QUANT-QWEN38-27B-NVFP4-ARM (#821). The SECOND whole-checkpoint read of the
  // quantization config, for the compressed-tensors spelling this file's
  // per-projection dtype probe cannot resolve.
  //
  // `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d...` declares
  // `format: "mixed-precision"`: layers 0-55's MLP is NVFP4 W4A4 and layers
  // 56-63's MLP is FP8 W8A8, under the SAME module names. Which arm a
  // projection takes is a REGEX OVER LAYER INDICES in `config_groups`, so no
  // dtype probe can recover it — and its FP8 group ships a PER-OUTPUT-CHANNEL
  // BF16 `weight_scale` with DYNAMIC per-token activations, which this file's
  // `LoadFp8Raw` rung cannot represent. Before this call the loader entered the
  // per-tensor arm anyway and died on `tensor not found:
  // ...in_proj_qkv.input_scale`, a sentence about a checkpoint that is
  // complete: the artifact ships ZERO `*.input_scale` because its activation
  // scheme is dynamic, exactly as its config says.
  //
  // Empty on every checkpoint that is not compressed-tensors, and on every
  // compressed-tensors checkpoint whose groups this build can execute, so every
  // arm that loaded before this row still loads byte-identically.
  const std::string ct_refusal =
      layers::compressed_tensors::RefusalForHfConfigRaw(config.raw, all_names);
  VT_CHECK(ct_refusal.empty(), "qwen3_5 dense: " + ct_refusal);

  Qwen3_5DenseWeights w;
  w.embed_tokens = LoadBf16Direct(get, backbone + "embed_tokens.weight");
  w.final_norm = LoadModelBf16Direct(get, backbone + "norm.weight");
  // The 27B owns an explicit head; smaller Qwen3.5 checkpoints tie logits to
  // the embedding table and omit lm_head.weight.
  if (DenseCheckpointHasLmHead(has, "lm_head")) {
    LoadDenseLmHead(get, has, "lm_head", w.lm_head, w.lm_head_fp4);
  } else {
    w.tied_lm_head = true;
    w.embed_tokens.nk = true;
  }
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  bool direct_device = DirectDeviceLoadEligible(load_queue);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadQwen3_5DenseLayer(
        get, has, config.layer_types[static_cast<size_t>(l)], l, backbone,
        block));
    if (direct_device) {
      direct_device = IsPlainBf16Qwen3_5Dense(w);
      if (direct_device) StageAndReleaseLoadedDense(w, *load_queue);
    }
  }
  return w;
}

// MODEL-FP8-BLOCK-LINEAR (#1189 M4), the M4/M5 seam, narrowing the M3/M4 one.
//
// M3 refused every LOADED block weight, because the dense forward knew only
// fp4, per-tensor fp8 and bf16 and an unwired block-wise checkpoint would fall
// through to an EMPTY bf16 tensor. The forward reads all ten projections now,
// through THREE entry points rather than one:
// `dense_fp8_block::MatmulFp8BlockScaledD` reads eight of them,
// `MatmulFp8BlockMergedD` reads q/k/v as one operand, and
// `Fp8BlockGateUpSwiGLUD` is the only reader of `gate_proj` and `up_proj`. So
// what is left to refuse is a DEVICE with no kernel. M5 (`489a9a4c0`) added the
// mainloop-scaled CUTLASS kernel for `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a), so
// this now fires for a CUDA arch outside that cell. It stays inert on CPU,
// where `vt::MatmulFp8BlockScaled` is registered as a correctness reference.
// This runs from `PrepareQwen3_5Dense`, i.e.
// `ModelRegistry::Prepare`, which every runner calls unconditionally before the
// first forward and before any graph capture
// (`src/vllm/v1/worker/gpu/runner.cpp:414,455`), so the user is told before a
// capture rather than inside the first GEMM. M5 NARROWED this rather than
// deleting it: an arch outside the CUTLASS cell must still be refused by name.
void RefuseUnrunnableQwen3_5DenseFp8Block(const Qwen3_5DenseWeights& weights,
                                          vt::DeviceType device) {
  if (dense_fp8_block::BlockFp8Runnable(device)) return;
  for (size_t l = 0; l < weights.layers.size(); ++l) {
    const Qwen3_5DenseLayerWeights& layer = weights.layers[l];
    const std::string base = "model.layers." + std::to_string(l) + ".";
    const auto check = [&base, device](const Fp8BlockWeight& w,
                                       const std::string& suffix) {
      if (!w.Empty()) RefuseUnrunnableFp8BlockWeight(base + suffix, device);
    };
    if (layer.is_linear_attention) {
      check(layer.gdn.in_proj_qkv_fp8_block, "linear_attn.in_proj_qkv");
      check(layer.gdn.in_proj_z_fp8_block, "linear_attn.in_proj_z");
      check(layer.gdn.out_proj_fp8_block, "linear_attn.out_proj");
    } else {
      check(layer.attn.q_proj_fp8_block, "self_attn.q_proj");
      check(layer.attn.k_proj_fp8_block, "self_attn.k_proj");
      check(layer.attn.v_proj_fp8_block, "self_attn.v_proj");
      check(layer.attn.o_proj_fp8_block, "self_attn.o_proj");
    }
    check(layer.mlp.gate_proj_fp8_block, "mlp.gate_proj");
    check(layer.mlp.up_proj_fp8_block, "mlp.up_proj");
    check(layer.mlp.down_proj_fp8_block, "mlp.down_proj");
  }
}

bool IsPlainBf16Qwen3_5Dense(const Qwen3_5DenseWeights& weights) {
  // A PACKED head (PERF-27B-LMHEAD-FP4) is not plain bf16: this staging path
  // only knows how to stage OwnedTensors.
  if (!weights.lm_head_fp4.Empty()) return false;
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    if (!layer.mlp.gate_proj_fp4.Empty() || !layer.mlp.up_proj_fp4.Empty() ||
        !layer.mlp.down_proj_fp4.Empty()) {
      return false;
    }
    // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): a block-wise FP8 projection is not
    // plain bf16 either, and the direct-device staging path only knows how to
    // stage OwnedTensors.
    if (!layer.mlp.gate_proj_fp8_block.Empty() ||
        !layer.mlp.up_proj_fp8_block.Empty() ||
        !layer.mlp.down_proj_fp8_block.Empty()) {
      return false;
    }
    if (layer.is_linear_attention) {
      if (!layer.gdn.out_proj_fp4.Empty() ||
          !layer.gdn.in_proj_qkv_fp8.Empty() ||
          !layer.gdn.in_proj_z_fp8.Empty() ||
          !layer.gdn.out_proj_fp8.Empty() ||
          !layer.gdn.in_proj_qkv_fp8_block.Empty() ||
          !layer.gdn.in_proj_z_fp8_block.Empty() ||
          !layer.gdn.out_proj_fp8_block.Empty()) {
        return false;
      }
    } else if (!layer.attn.q_proj_fp8_block.Empty() ||
               !layer.attn.k_proj_fp8_block.Empty() ||
               !layer.attn.v_proj_fp8_block.Empty() ||
               !layer.attn.o_proj_fp8_block.Empty() ||
               !layer.attn.q_proj_fp4.Empty() ||
               !layer.attn.k_proj_fp4.Empty() ||
               !layer.attn.v_proj_fp4.Empty() ||
               !layer.attn.o_proj_fp4.Empty() ||
               !layer.attn.q_proj_fp8.Empty() ||
               !layer.attn.k_proj_fp8.Empty() ||
               !layer.attn.v_proj_fp8.Empty() ||
               !layer.attn.o_proj_fp8.Empty()) {
      return false;
    }
  }
  return true;
}

size_t ReleaseResidentQwen3_5DenseHostWeights(
    Qwen3_5DenseWeights& weights) {
  size_t released = 0;
  const auto release = [&released](OwnedTensor& tensor) {
    // `HostMirrorIsRedundant`, NOT `d_dev || d_dev_f32` (the second instance of
    // the use-after-free W0f introduced; #1299).
    //
    // `d_dev_f32` is a bf16->f32 UPCAST into a separate device allocation. It is
    // not a copy of these bytes and it can never stand in for them, so it never
    // made a host mirror redundant — it only looked like it did while every
    // weight that had one also had a `d_dev`. On the aliasing arm that stopped
    // being true: `PrepareBf16Resident` passes exactly four weights to BOTH
    // `raw()` and `f32()` — `gdn.conv1d_weight`, `gdn.norm_weight`,
    // `attn.q_norm`, `attn.k_norm` — and there `raw()` ALIASES (leaving `d_dev`
    // null) while `f32()` allocates (setting `d_dev_f32`). The disjunction then
    // passes and frees the very bytes the aliased raw tensor points at.
    //
    // Nothing reaches that combination today, and the reason is worth writing
    // down because it is an accident rather than a design:
    // `DirectDeviceLoadEligible` above requires `!platform.is_unified_memory()`,
    // and on CUDA `is_unified_memory()` and `host_memory_is_device_addressable()`
    // are the SAME `pageable && integrated` conjunction, computed independently
    // in `src/vt/cuda/cuda_backend.cu` and `src/vllm/platforms/cuda.cpp`. No
    // rule states that equality, no document records it and no gate holds it, so
    // a platform that ever separates the two arrives here with a live
    // use-after-free. Asking the invariant instead of a proxy for it costs
    // nothing: the disjunct is redundant on the staging arm, where every weight
    // with a `d_dev_f32` has a `d_dev` too.
    if (tensor.HasHostBytes() && HostMirrorIsRedundant(tensor)) {
      released += tensor.bytes.size();
      tensor.ReleaseHost();
    }
  };
  const auto release_gdn = [&release](GdnLayerWeights& gdn) {
    release(gdn.in_proj_qkv);
    release(gdn.in_proj_z);
    release(gdn.in_proj_qkvz);
    release(gdn.in_proj_b);
    release(gdn.in_proj_a);
    release(gdn.in_proj_ba);
    release(gdn.conv1d_weight);
    release(gdn.a_log);
    release(gdn.dt_bias);
    release(gdn.norm_weight);
    release(gdn.out_proj);
  };
  const auto release_attn = [&release](FullAttnLayerWeights& attn) {
    release(attn.q_proj);
    release(attn.k_proj);
    release(attn.v_proj);
    release(attn.o_proj);
    release(attn.q_norm);
    release(attn.k_norm);
  };

  release(weights.embed_tokens);
  release(weights.final_norm);
  release(weights.lm_head);
  for (Qwen3_5DenseLayerWeights& layer : weights.layers) {
    release(layer.input_layernorm);
    release(layer.post_attention_layernorm);
    if (layer.is_linear_attention)
      release_gdn(layer.gdn);
    else
      release_attn(layer.attn);
    release(layer.mlp.gate_proj);
    release(layer.mlp.up_proj);
    release(layer.mlp.gate_up_proj);
    release(layer.mlp.down_proj);
  }
  return released;
}

}  // namespace vllm
