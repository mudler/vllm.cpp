// DeepSeek-V4.1-Flash W3b — the MXFP8 32x32 UE8M0 linear family host reference.
// Ported 1:1 from vLLM @ `e77daef89e` (the V4.1 head; W3b needs no pin advance,
// per `.agents/specs/deepseek-v4-1-flash.md` `## Work breakdown`):
//   - the checkpoint -> runtime scale row expansion:
//     model_executor/layers/quantization/modelopt.py:2186-2201
//     (`KMxfp8Static.get_scale_weight_loader`)
//   - the runtime per-32-column dequant:
//     .../utils/mxfp8_utils.py:230-243 (`dequant_mxfp8_to_bf16`)
//   - the dynamic activation quantizer:
//     .../utils/mxfp8_utils.py:113-142 (`_mxfp8_quant_triton_kernel`), whose
//     amax/bias/clamp are identical to :56-69 (`_mxfp8_e4m3_quantize_torch`)
//   - the emulation linear arm:
//     model_executor/kernels/linear/mxfp8/emulation.py:16-84
//   - the checkpoint scale name split:
//     models/deepseek_v4_1/nvidia/model.py:871-882
// See deepseek_v4_1_mxfp8.h for the full semantics, the three named divergences,
// and the honest-scope note.
#include "vllm/model_executor/models/deepseek_v4_1_mxfp8.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/mxfp4_dequant.h"  // E8M0ToF32
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // F8E4M3ToF32
#include "vt/dtype.h"                                        // VT_CHECK, F32ToBF16

namespace vllm::deepseek_v4_1 {

std::vector<uint8_t> ExpandMxfp8CheckpointScale(const uint8_t* ckpt_scale,
                                                int64_t ckpt_rows, int64_t ckpt_cols,
                                                int64_t block_rows,
                                                int64_t block_cols) {
  // modelopt.py:2189-2192 raises NotImplementedError on exactly these two, and
  // names the offending block in the message. Refuse the same way rather than
  // reading a scale plane this layout does not describe.
  VT_CHECK(block_cols == kMxfp8BlockSize,
           "MXFP8 checkpoint scale block columns must be 32 (MXFP8_BLOCK_SIZE); "
           "the scale group is 32 ELEMENTS ALONG K in both the checkpoint and "
           "the runtime layout, so no other column block is expressible");
  VT_CHECK(block_rows >= 1,
           "MXFP8 checkpoint scale block rows must be at least 1; each checkpoint "
           "row expands into that many runtime rows");
  VT_CHECK(ckpt_scale != nullptr, "MXFP8 checkpoint scale expansion: scale is null");
  VT_CHECK(ckpt_rows >= 0 && ckpt_cols >= 0,
           "MXFP8 checkpoint scale expansion: dims must not be negative");

  // `loaded_weight.repeat_interleave(block_rows, dim=0)` (modelopt.py:2196-2197):
  // each checkpoint row is repeated CONTIGUOUSLY, so output row n reads
  // checkpoint row n / block_rows. `repeat`/`tile` would concatenate whole copies
  // of the plane instead and is the wrong operation.
  std::vector<uint8_t> out(static_cast<size_t>(ckpt_rows * block_rows * ckpt_cols));
  for (int64_t r = 0; r < ckpt_rows; ++r) {
    const uint8_t* src = ckpt_scale + r * ckpt_cols;
    for (int64_t rep = 0; rep < block_rows; ++rep) {
      uint8_t* dst = out.data() + (r * block_rows + rep) * ckpt_cols;
      std::copy(src, src + ckpt_cols, dst);
    }
  }
  return out;
}

namespace {

// The one shared row walk behind both emitters. `store` maps the f32 dequant
// result to the caller's output element, exactly as mxfp4_dequant.cpp does for
// its bf16/f32 pair, so the two emitters cannot drift apart.
template <typename Store>
void DequantMxfp8Rows(const uint8_t* weight_f8, const uint8_t* scale_e8m0, int64_t N,
                      int64_t K, Store store) {
  VT_CHECK(weight_f8 != nullptr && scale_e8m0 != nullptr,
           "MXFP8 dequant: weight or scale is null");
  VT_CHECK(N > 0 && K > 0, "MXFP8 dequant: dims must be positive");
  VT_CHECK(K % kMxfp8BlockSize == 0,
           "MXFP8 dequant: K must divide by 32 (MXFP8_BLOCK_SIZE); upstream "
           "rejects the layer at create_weights when it does not "
           "(modelopt.py:2206-2210)");
  const int64_t scale_cols = K / kMxfp8BlockSize;
  for (int64_t n = 0; n < N; ++n) {
    const uint8_t* wrow = weight_f8 + n * K;
    // Scale row index is `n`, NOT `n / 32`: this is the layout AFTER the row
    // expansion above. mxfp8_utils.py:235-239 broadcasts one scale per
    // (row, 32-column block).
    const uint8_t* srow = scale_e8m0 + n * scale_cols;
    for (int64_t k = 0; k < K; ++k) {
      // `x.to(f32) * exp2(scale.to(f32) - 127)` (mxfp8_utils.py:232,237,239).
      // E8M0ToF32 is that exp2, already in this tree.
      store(n * K + k, F8E4M3ToF32(wrow[k]) * E8M0ToF32(srow[k / kMxfp8BlockSize]));
    }
  }
}

}  // namespace

void DequantMxfp8ToF32(const uint8_t* weight_f8, const uint8_t* scale_e8m0, int64_t N,
                       int64_t K, float* out_f32) {
  VT_CHECK(out_f32 != nullptr, "MXFP8 dequant: output buffer is null");
  DequantMxfp8Rows(weight_f8, scale_e8m0, N, K,
                   [&](int64_t i, float v) { out_f32[i] = v; });
}

void DequantMxfp8ToBf16(const uint8_t* weight_f8, const uint8_t* scale_e8m0, int64_t N,
                        int64_t K, uint16_t* out_bf16) {
  VT_CHECK(out_bf16 != nullptr, "MXFP8 dequant: output buffer is null");
  // `.to(torch.bfloat16)` at mxfp8_utils.py:243 — the narrowing happens ONCE, at
  // the store, after the f32 multiply.
  DequantMxfp8Rows(weight_f8, scale_e8m0, N, K,
                   [&](int64_t i, float v) { out_bf16[i] = vt::F32ToBF16(v); });
}

void QuantizeMxfp8E4m3(const float* x, int64_t M, int64_t K, uint8_t* out_q,
                       uint8_t* out_s) {
  VT_CHECK(x != nullptr && out_q != nullptr && out_s != nullptr,
           "MXFP8 dynamic quant: null buffer");
  VT_CHECK(M > 0 && K > 0, "MXFP8 dynamic quant: dims must be positive");
  VT_CHECK(K % kMxfp8BlockSize == 0,
           "MXFP8 dynamic quant: K must divide by 32 (MXFP8_BLOCK_SIZE) "
           "(mxfp8_utils.py:52)");
  // `TINY = torch.finfo(torch.float32).tiny` (mxfp8_utils.py:182), the smallest
  // NORMAL float, which is what `amax.clamp(min=...)` uses at :60.
  const float kTiny = std::numeric_limits<float>::min();
  const int64_t blocks = K / kMxfp8BlockSize;

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t b = 0; b < blocks; ++b) {
      const float* blk = x + m * K + b * kMxfp8BlockSize;
      // `amax = max(|x|, TINY)` (mxfp8_utils.py:126). Kept for fidelity, and
      // measured as NOT load-bearing on a host: replacing TINY with 0.0F leaves
      // the suite GREEN with the binary proved changed, because `log2(0)` is
      // `-inf` and the LOW CLAMP already maps that to sb == 0. It is a defence
      // against a backend where the intermediate misbehaves, not a value this
      // gate can distinguish. The digest and the build recipe for that run are
      // in the `## Mutations` table of
      // `ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P`.
      float amax = kTiny;
      for (int64_t j = 0; j < kMxfp8BlockSize; ++j)
        amax = std::max(amax, std::fabs(blk[j]));

      // `sb = clamp(ceil(log2(amax / FP8_MAX)) + 127, 0, 254)`
      // (mxfp8_utils.py:127-128). The bias puts the block amax at the TOP of the
      // e4m3 range rather than at 1.0; :123-125 says why — otherwise the small
      // elements of the block land in the fp8 subnormals.
      const float raw = std::ceil(std::log2(amax / kMxfp8Fp8Max)) + 127.0F;
      const int sb = static_cast<int>(std::min(254.0F, std::max(0.0F, raw)));
      out_s[m * blocks + b] = static_cast<uint8_t>(sb);

      // `rescale = exp2(127 - sb); xq = x * rescale` (mxfp8_utils.py:135-136).
      // MULTIPLY by the reciprocal, not divide by `exp2(sb - 127)`: upstream's
      // own comment at :129-134 says a `sb == 0` block makes that divisor
      // `2**-127`, subnormal in fp32, which flushes to zero on CDNA and turns
      // the block's zeros into `0/0 == NaN`. Both are exact powers of two, so
      // the two forms agree bit-for-bit on every host WITHOUT flush-to-zero —
      // which is not "for every reachable sb": at sb == 0 they are exactly the
      // pair upstream's comment separates, and only a CDNA arm can show it.
      // W5 owes that measurement.
      const float rescale = std::ldexp(1.0F, 127 - sb);
      uint8_t* qblk = out_q + m * K + b * kMxfp8BlockSize;
      for (int64_t j = 0; j < kMxfp8BlockSize; ++j)
        qblk[j] = F32ToF8E4M3(blk[j] * rescale);
    }
  }
}

namespace {

// The one shared body behind both emulation emitters. `load_x` and `load_bias`
// widen the caller's element to f32; `store` narrows the f32 accumulator back.
// Accumulation is f32 in both arms, because `F.linear` on bf16 operands
// accumulates in f32 and narrows once at the output (emulation.py:62-63).
template <typename LoadX, typename LoadBias, typename Store>
void Mxfp8LinearEmulationBody(int64_t M, int64_t K, const uint8_t* weight_f8,
                              const uint8_t* scale_e8m0, int64_t N, bool has_bias,
                              LoadX load_x, LoadBias load_bias, Store store) {
  // K is validated HERE and not only inside the dequant below: the weight buffer
  // is sized `N * K` first, and a negative K makes `static_cast<size_t>(N * K)`
  // ask the allocator for ~2^64 elements before any VT_CHECK can refuse.
  VT_CHECK(M > 0 && N > 0 && K > 0,
           "MXFP8 emulation linear: dims must be positive");

  // emulation.py:45-48: `dequant_mxfp8_to_bf16` ONCE, at load, replacing the
  // 1-byte MXFP8 weight with BF16. The narrowing is part of the arm's numerics,
  // not an implementation detail, so it happens here and not inside the dot.
  std::vector<uint16_t> w_bf16(static_cast<size_t>(N * K));
  DequantMxfp8ToBf16(weight_f8, scale_e8m0, N, K, w_bf16.data());

  // emulation.py:59-63: a plain `F.linear(x, weight, bias)` — the emulation arm
  // does NOT quantize the activation, which is why `QuantizeMxfp8E4m3` has no
  // caller on this path (it serves the native backends).
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      const uint16_t* wrow = w_bf16.data() + n * K;
      float acc = 0.0F;
      for (int64_t k = 0; k < K; ++k) acc += load_x(m * K + k) * vt::BF16ToF32(wrow[k]);
      store(m * N + n, has_bias ? acc + load_bias(n) : acc);
    }
  }
}

}  // namespace

void Mxfp8LinearEmulationBf16(const uint16_t* x_bf16, int64_t M, int64_t K,
                              const uint8_t* weight_f8, const uint8_t* scale_e8m0,
                              int64_t N, const uint16_t* bias_bf16,
                              uint16_t* out_bf16) {
  VT_CHECK(x_bf16 != nullptr && out_bf16 != nullptr,
           "MXFP8 emulation linear (bf16): activation or output is null");
  Mxfp8LinearEmulationBody(
      M, K, weight_f8, scale_e8m0, N, bias_bf16 != nullptr,
      [&](int64_t i) { return vt::BF16ToF32(x_bf16[i]); },
      [&](int64_t n) { return vt::BF16ToF32(bias_bf16[n]); },
      // `output.to(x.dtype)` (emulation.py:63): ONE narrowing, at the store.
      [&](int64_t i, float v) { out_bf16[i] = vt::F32ToBF16(v); });
}

void Mxfp8LinearEmulation(const float* x, int64_t M, int64_t K, const uint8_t* weight_f8,
                          const uint8_t* scale_e8m0, int64_t N, const float* bias,
                          float* out_f32) {
  VT_CHECK(x != nullptr && out_f32 != nullptr,
           "MXFP8 emulation linear: activation or output is null");
  Mxfp8LinearEmulationBody(
      M, K, weight_f8, scale_e8m0, N, bias != nullptr, [&](int64_t i) { return x[i]; },
      [&](int64_t n) { return bias[n]; },
      [&](int64_t i, float v) { out_f32[i] = v; });
}

std::string Mxfp8LinearScaleParamName(const std::vector<int64_t>& weight_block_size,
                                      std::string_view expert_dtype) {
  // nvidia/model.py:878-882. BOTH halves of the conjunction are load-bearing:
  // native MXFP8 is `[32, 32]` blocks WITH MXFP4 experts, and only that pairing
  // routes linears through `ModelOptLinearMethod`, which registers
  // `weight_scale`. Block-FP8 linears register `weight_scale_inv`.
  const bool use_mxfp8 =
      weight_block_size.size() == 2 && weight_block_size[0] == kMxfp8BlockSize &&
      weight_block_size[1] == kMxfp8BlockSize && expert_dtype == "fp4";
  return use_mxfp8 ? "weight_scale" : "weight_scale_inv";
}

}  // namespace vllm::deepseek_v4_1
