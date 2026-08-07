// MiniMax-H3 DiT — the DEVICE-RESIDENT forward (brick H3-2b).
//
// `MiniMaxH3DitForward` (minimax_h3.cpp) is the portable reference: it computes
// into host `std::vector<float>` buffers and only reaches for `vt::` at the two
// big ops. This file runs the SAME graph with every activation in device memory,
// so the block stack — and, above it, the 50-step denoise loop — never round-trips
// through the host.
//
// ─── WHAT IS PORTED, AND FROM WHERE ──────────────────────────────────────────
// Structure is 1:1 with the reference; each step below names the reference helper
// it replaces and the upstream line it came from. The port REUSES tuned shared
// ops wherever one exists, which is why only three H3-specific kernels were added
// (minimax_h3_device.h):
//
//   Linear                  -> vt::MatmulBT + vt::Add (rank-1 row-broadcast bias)
//   RmsNormRows             -> vt::RmsNorm
//   qkv split               -> vt::QkvSplit
//   ApplyRope (3-axis)      -> vt::RopeFromCache over a prebuilt per-row cache
//   MLP silu(gate)*up       -> vt::SiluAndMul
//   residual add            -> vt::Add
//   row gather / scatter    -> vt::IndexSelect / vt::IndexCopy
//   packed varlen attention -> vt::DFlashBlockAttention(causal=false)
//   ModulateScaleShift/Gate -> kMiniMaxH3 glue table
//   Silu (ungated)          -> kMiniMaxH3 glue table
//
// ─── NUMERICS: CLOSE, NOT BIT-IDENTICAL ──────────────────────────────────────
// This path is NOT bit-identical to the CPU reference and does not claim to be.
// The divergence is in the SHARED ops, not the H3 kernels: vt::RmsNorm reduces in
// f32 where RmsNormRows accumulates in double, and MatmulBT/attention use their
// own accumulation orders. f32 is what upstream torch RMSNorm does, so the device
// path is arguably the closer mirror. It is gated against the SAME upstream
// goldens at the SAME tolerance as the CPU forward.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"  // W-FP4a: MatmulNvfp4W4A16D
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/minimax_h3_device.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

// The device twin of the reference's StreamDtype -- but where the reference rounds
// f32 buffers in place, this path gives the stream REAL bf16 STORAGE. The rounding
// then happens where upstream's happens (on every store), the activations are half
// the bytes, and the tuned shared ops (RmsNorm / MatmulBT / SiluAndMul / attention)
// run their native bf16 paths instead of an f32 path plus a rounding pass.
//
// `S()` is the stream dtype; the fp32 ISLANDS (both patch projections, the time
// embedder, both output heads) allocate kF32 explicitly and are converted at the
// boundary with vt::CastBf16 / vt::CastF32.
struct DeviceStreamDtype {
  bool bf16 = false;
  DType S() const { return bf16 ? DType::kBF16 : DType::kF32; }
};

// Convert between the f32 ISLANDS and the stream dtype. A same-dtype call is a
// straight copy, so the f32 path costs one memcpy and stays byte-identical.
void CastTo(Dev d, Tensor& out, const Tensor& in) {
  if (out.dtype == in.dtype) {
    vt::GetBackend(d.q.device.type).Copy(d.q, out.data, in.data, in.Bytes());
  } else if (out.dtype == DType::kBF16) {
    vt::CastBf16(d.q, out, in);
  } else {
    vt::CastF32(d.q, out, in);
  }
}

const minimax_h3::MiniMaxH3DeviceKernels* Glue(const Dev& d) {
  VT_CHECK(minimax_h3::MiniMaxH3DeviceKernelsAvailable(d.q.device.type),
           "minimax_h3: no device glue table registered for this backend");
  return minimax_h3::MiniMaxH3Device(d.q.device.type);
}

// DIAGNOSTIC helper: write a per-stage activation/weight fingerprint of a CONTIGUOUS
// device tensor -- summary stats plus a fixed spread of positional sample values that
// catches a transpose/scramble even when the summary matches. Byte-inert unless a
// caller passes a non-null file. Shared by the forward's stage hooks and AttentionDev.
void H3DumpFingerprint(std::FILE* f, vt::Backend& backend, vt::Queue& q, const char* stage,
                       const Tensor& t) {
  if (f == nullptr || t.data == nullptr) return;
  int64_t n = 1;
  for (int r = 0; r < t.rank; ++r) n *= t.shape[r];
  if (n <= 0) return;
  std::vector<float> host(static_cast<size_t>(n));
  if (t.dtype == DType::kF32) {
    backend.Copy(q, host.data(), t.data, static_cast<size_t>(n) * sizeof(float));
    backend.Synchronize(q);
  } else {  // bf16 stream tensor -> widen on the host
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    backend.Copy(q, raw.data(), t.data, static_cast<size_t>(n) * sizeof(uint16_t));
    backend.Synchronize(q);
    for (int64_t i = 0; i < n; ++i) {
      const uint32_t widened = static_cast<uint32_t>(raw[static_cast<size_t>(i)]) << 16;
      std::memcpy(&host[static_cast<size_t>(i)], &widened, sizeof(float));
    }
  }
  double s = 0.0, s2 = 0.0, amax = 0.0;
  bool finite = true;
  for (int64_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(host[static_cast<size_t>(i)]);
    if (!std::isfinite(v)) finite = false;
    s += v;
    s2 += v * v;
    if (std::fabs(v) > amax) amax = std::fabs(v);
  }
  const double denom = static_cast<double>(n);
  std::fprintf(f, "%-26s n=%-9lld dt=%s finite=%d mean=%+.6e rms=%.6e absmax=%.6e |", stage,
               static_cast<long long>(n), t.dtype == DType::kF32 ? "f32" : "bf16", finite ? 1 : 0,
               s / denom, std::sqrt(s2 / denom), amax);
  const int64_t samp[] = {0,         1,     2,         3,         n / 8,     n / 4,
                          3 * n / 8, n / 2, 5 * n / 8, 3 * n / 4, 7 * n / 8, n - 1};
  for (int64_t idx0 : samp) {
    int64_t idx = idx0;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    std::fprintf(f, " %+.5e", host[static_cast<size_t>(idx)]);
  }
  std::fprintf(f, "\n");
  std::fflush(f);
}

// vt::MatmulBT + optional rank-1 bias, the device twin of the reference `Linear`.
// Weight is [out_features, in_features] — every {Column,Row,QKV,MergedColumn}
// ParallelLinear at TP=1.
//
// W-FP4a: when `fp4` is non-null and non-Empty(), the projection is fp4-RESIDENT.
// The GEMM routes through dense_nvfp4::MatmulNvfp4W4A16D instead of vt::MatmulBT —
// the SAME forced-Marlin W4A16 dispatch the Laguna/dense-Qwen3 NVFP4 arms use
// (vLLM __init__.py:879-881: a weight-only NVFP4 scheme forces the Marlin kernel).
// On sm_121a with a bf16 activation that is the native FP4-tensor-core path; in the
// f32 parity stream (or on a backend without the Marlin op) the SAME dispatcher
// falls back to a redundant-dequant GEMM, so the arm is correct either way. The
// dequant lives entirely inside the shared dispatcher — this adds NO quant code.
void LinearDev(Dev d, const Tensor& in, int64_t rows, int64_t in_features, const Tensor& weight,
               const Tensor* bias, Tensor& out, const Nvfp4Weight* fp4 = nullptr) {
  Tensor a = dense_attn::Reshape(in, {rows, in_features});
  if (fp4 != nullptr && !fp4->Empty()) {
    VT_CHECK(fp4->k == in_features,
             "minimax_h3 fp4 linear: packed weight K does not match input width");
    Tensor o = dense_attn::Reshape(out, {rows, fp4->n});
    DBuf r = dense_nvfp4::MatmulNvfp4W4A16D(d, a, *fp4, o.dtype);
    d.b.Copy(d.q, o.data, r.t().data, r.bytes());
    if (bias != nullptr && bias->data != nullptr) {
      vt::Add(d.q, o, o, *bias);  // rank-1 row-broadcast == a nn.Linear bias term
    }
    return;
  }
  VT_CHECK(weight.rank == 2 && weight.shape[1] == in_features,
           "minimax_h3 device linear: weight shape does not match input width");
  Tensor o = dense_attn::Reshape(out, {rows, weight.shape[0]});
  vt::MatmulBT(d.q, o, a, weight);
  if (bias != nullptr && bias->data != nullptr) {
    vt::Add(d.q, o, o, *bias);  // rank-1 row-broadcast == a nn.Linear bias term
  }
}

// MiniMaxH3Rope.forward + _apply_rope (minimax_h3_transformer.py:222-244) as a
// cos/sin CACHE.
//
// H3's rope is plain NeoX rotate_half over the leading rot_dim head dims; only the
// ANGLES are unusual (three axes t/h/w off the fp64 position grid rather than one
// scalar position). vt::RopeFromCache consumes exactly a [S, rot_dim] cache laid
// out as [cos(half) | sin(half)] and indexes it by position, so building that
// cache here reproduces ApplyRope with NO bespoke rope kernel.
//
// The reference duplicates the half-angle block (`memcpy(dst + half, dst, ...)`)
// so its cos_lo/cos_hi are equal; that duplication is exactly what makes the
// rotation NeoX-shaped, and it is why one [S, rot_dim] cache suffices.
std::vector<float> BuildRopeCosSin(const double* position_ids, int64_t seq_len,
                                   const float* inv_freq, int64_t inv_freq_len) {
  const int64_t half = 3 * inv_freq_len;  // rot_dim / 2
  const int64_t rot_dim = 2 * half;
  std::vector<float> cache(static_cast<size_t>(seq_len * rot_dim));
  for (int64_t s = 0; s < seq_len; ++s) {
    float* dst = cache.data() + s * rot_dim;
    for (int64_t axis = 0; axis < 3; ++axis) {
      const float pos = static_cast<float>(position_ids[s * 3 + axis]);
      for (int64_t i = 0; i < inv_freq_len; ++i) {
        const float angle = pos * inv_freq[i];
        dst[axis * inv_freq_len + i] = std::cos(angle);
        dst[half + axis * inv_freq_len + i] = std::sin(angle);
      }
    }
  }
  return cache;
}

struct AttnWeightsDev {
  const Tensor* qkv;
  const Tensor* q_norm;
  const Tensor* k_norm;
  const Tensor* out_proj;
  // W-FP4a: fp4-resident twins of qkv/out_proj (null on the bf16/f32 arms).
  const Nvfp4Weight* qkv_fp4 = nullptr;
  const Nvfp4Weight* out_fp4 = nullptr;
};

// MiniMaxH3Attention.forward (minimax_h3_transformer.py:421-467).
void AttentionDev(Dev d, const MiniMaxH3DitParams& params, const AttnWeightsDev& w,
                  const Tensor& in, int64_t rows, const Tensor* rope_cache,
                  const Tensor* rope_positions, const int32_t* cu_seqlens, int num_reqs,
                  const DeviceStreamDtype& dt, Tensor& out, std::FILE* dbg = nullptr,
                  const char* dbg_pfx = nullptr) {
  const int64_t heads = params.num_attention_heads;
  const int64_t head_dim = params.attention_head_dim;
  const int64_t inner = heads * head_dim;
  auto dbgd = [&](const char* leaf, const Tensor& t) {
    if (dbg == nullptr) return;
    char nm[96];
    std::snprintf(nm, sizeof(nm), "%s.%s", dbg_pfx ? dbg_pfx : "attn", leaf);
    H3DumpFingerprint(dbg, vt::GetBackend(d.q.device.type), d.q, nm, t);
  };

  DBuf qkv(d, dt.S(), {rows, 3 * inner});
  LinearDev(d, in, rows, params.hidden_size, *w.qkv, nullptr, qkv.t(), w.qkv_fp4);
  dbgd("qkv_out", qkv.t());

  DBuf qb(d, dt.S(), {rows, inner});
  DBuf kb(d, dt.S(), {rows, inner});
  DBuf vb(d, dt.S(), {rows, inner});
  vt::QkvSplit(d.q, qb.t(), kb.t(), vb.t(), qkv.t());
  dbgd("q_split", qb.t());
  dbgd("v_split", vb.t());

  // Per-head RMSNorm over head_dim: [rows, heads, head_dim] -> [rows*heads, head_dim].
  Tensor qn = dense_attn::Reshape(qb.t(), {rows * heads, head_dim});
  Tensor kn = dense_attn::Reshape(kb.t(), {rows * heads, head_dim});
  vt::RmsNormArgs norm_args;
  norm_args.eps = static_cast<float>(params.qk_norm_eps);
  vt::RmsNorm(d.q, qn, qn, *w.q_norm, norm_args);
  vt::RmsNorm(d.q, kn, kn, *w.k_norm, norm_args);
  dbgd("q_qknorm", qb.t());

  if (rope_cache != nullptr) {
    Tensor q3 = dense_attn::Reshape(qb.t(), {rows, heads, head_dim});
    Tensor k3 = dense_attn::Reshape(kb.t(), {rows, heads, head_dim});
    vt::RopeArgs rope;
    rope.rotary_dim = static_cast<int>(params.rope_rot_dim());
    rope.is_neox_style = true;
    vt::RopeFromCache(d.q, q3, &k3, *rope_positions, *rope_cache, rope);
  }
  dbgd("q_roped", qb.t());

  Tensor tq = dense_attn::Reshape(qb.t(), {rows, heads, head_dim});
  Tensor tk = dense_attn::Reshape(kb.t(), {rows, heads, head_dim});
  Tensor tv = dense_attn::Reshape(vb.t(), {rows, heads, head_dim});
  DBuf attn(d, dt.S(), {rows, heads, head_dim});
  vt::DFlashBlockAttentionArgs args;
  args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  args.causal = false;  // bidirectional within each packed document
  args.sliding_window = 0;
  args.cu_seqlens = cu_seqlens;
  args.num_reqs = num_reqs;
  vt::DFlashBlockAttention(d.q, attn.t(), tq, tk, tv, args);
  dbgd("core_out", attn.t());

  Tensor flat = dense_attn::Reshape(attn.t(), {rows, inner});
  LinearDev(d, flat, rows, inner, *w.out_proj, nullptr, out, w.out_fp4);
  dbgd("out_proj", out);
}

// MiniMaxH3MLP.forward (minimax_h3_transformer.py:512-517): silu(gate) * up.
// fc1 emits [gate; up] per row, which is exactly vt::SiluAndMul's input layout.
void MlpDev(Dev d, const MiniMaxH3DitParams& params, const Tensor& fc1, const Tensor& fc2,
            const Tensor& in, int64_t rows, const DeviceStreamDtype& dt, Tensor& out,
            const Nvfp4Weight* fc1_fp4 = nullptr, const Nvfp4Weight* fc2_fp4 = nullptr) {
  const int64_t ffn = params.ffn_hidden_size;
  DBuf hidden(d, dt.S(), {rows, 2 * ffn});
  // fc1 is already the merged [gate; up] (SwiGLU) — one W4A16 GEMM to [rows, 2*ffn]
  // then SiluAndMul, so the fused gate_up pair (GateUpFusedMarlinD) does not apply.
  LinearDev(d, in, rows, params.hidden_size, fc1, nullptr, hidden.t(), fc1_fp4);
  DBuf act(d, dt.S(), {rows, ffn});
  vt::SiluAndMul(d.q, act.t(), hidden.t());
  LinearDev(d, act.t(), rows, ffn, fc2, nullptr, out, fc2_fp4);
}

// MiniMaxH3AdalnProj.forward (minimax_h3_transformer.py:555-561):
// silu(t_emb) -> linear. `activated` is the ALREADY-silu'd t_emb, hoisted out of
// the block loop by the caller: t_emb does not change across blocks, so the
// reference's per-block silu is redundant work the device path simply does once.
void AdalnProjectDev(Dev d, const Tensor& activated, int64_t m, int64_t time_embed_dim,
                     const Tensor& weight, const Tensor& bias, Tensor& out,
                     const Nvfp4Weight* fp4 = nullptr) {
  LinearDev(d, activated, m, time_embed_dim, weight, &bias, out, fp4);
}

}  // namespace

MiniMaxH3DitDeviceWeights StageMiniMaxH3DitWeights(vt::Queue& queue,
                                                   const MiniMaxH3DitParams& params,
                                                   const MiniMaxH3DitWeights& host,
                                                   DType compute_dtype) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  MiniMaxH3DitDeviceWeights staged;
  staged.weights = host;  // shapes carry over; every view is rebound below
  const bool bf16 = compute_dtype == DType::kBF16;

  // `as_bf16` marks a weight that upstream STORES in bf16. The fp32 islands pass
  // false and stay f32 even in the bf16 stream -- that split IS the dtype policy
  // (MINIMAX_H3_FP32_PARAM_NAMES, minimax_h3_transformer.py:85-101), and getting it
  // backwards would round tensors upstream deliberately keeps at full precision.
  auto upload = [&](const Tensor& src, Tensor& dst, bool as_bf16) {
    if (src.data == nullptr) {
      dst = src;
      return;
    }
    const std::vector<int64_t> shape(src.shape, src.shape + src.rank);
    const int64_t n = src.Numel();

    // KEEP-QUANT: a block-encoded weight is uploaded VERBATIM and keeps its block
    // dtype -- never dequantized, and never rounded to the stream dtype (its
    // scales already carry the precision). vt::MatmulBT routes it to
    // kMatmulBTQuant on its own, so the forward is unchanged.
    //
    // The upload is the load-bearing part: a keep-quant slice feeding a DEVICE
    // GEMM must point at DEVICE memory. Handing the GEMM a raw host-byte view
    // reads as ALL ZEROS on the GPU, and a CPU-only gate cannot catch it because
    // there the host pointer is valid.
    if (vt::IsBlockQuant(src.dtype)) {
      VT_CHECK(src.rank == 2, "minimax_h3 stage: keep-quant weights must be rank 2");
      const size_t bytes =
          static_cast<size_t>(shape[0]) * vt::RowSizeBytes(src.dtype, shape[1]);
      void* p = backend.Alloc(bytes);
      std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
      backend.Copy(queue, p, src.data, bytes);
      dst = dense_attn::MakeTensor(p, src.dtype, queue.device, shape);
      staged.storage.push_back(std::move(owner));
      return;
    }

    // Already bf16 (the throughput loader): upload verbatim, no conversion.
    if (src.dtype == DType::kBF16) {
      const size_t bytes = static_cast<size_t>(n) * vt::SizeOf(DType::kBF16);
      void* p = backend.Alloc(bytes);
      std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
      backend.Copy(queue, p, src.data, bytes);
      dst = dense_attn::MakeTensor(p, DType::kBF16, queue.device, shape);
      staged.storage.push_back(std::move(owner));
      return;
    }
    VT_CHECK(src.dtype == DType::kF32, "minimax_h3 stage: host weights must be f32, bf16 or block-quant");
    const DType want = (bf16 && as_bf16) ? DType::kBF16 : DType::kF32;
    const size_t bytes = static_cast<size_t>(n) * vt::SizeOf(want);
    void* p = backend.Alloc(bytes);
    std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
    if (want == DType::kF32) {
      backend.Copy(queue, p, src.data, bytes);
    } else {
      // Round on the HOST, then upload: the same round-to-nearest-even the stream
      // uses, so a bf16 weight and a bf16 activation agree bit-for-bit on the rule.
      std::vector<uint16_t> packed(static_cast<size_t>(n));
      const float* in = src.Ptr<float>();
      for (int64_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &in[i], sizeof(bits));
        if ((bits & 0x7F800000u) == 0x7F800000u) {
          bits &= 0xFFFF0000u;
        } else {
          const uint32_t lsb = (bits >> 16) & 1u;
          bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
        }
        packed[static_cast<size_t>(i)] = static_cast<uint16_t>(bits >> 16);
      }
      backend.Copy(queue, p, packed.data(), bytes);
      backend.Synchronize(queue);  // `packed` dies at the end of this scope
    }
    dst = dense_attn::MakeTensor(p, want, queue.device, shape);
    staged.storage.push_back(std::move(owner));
  };

  auto upload_block = [&](const MiniMaxH3DitBlockWeights& src, MiniMaxH3DitBlockWeights& dst) {
    // Every tensor in a refiner/DiT block is bf16-stored upstream.
    upload(src.norm1, dst.norm1, true);
    upload(src.norm2, dst.norm2, true);
    upload(src.qkv_proj, dst.qkv_proj, true);
    upload(src.q_norm, dst.q_norm, true);
    upload(src.k_norm, dst.k_norm, true);
    upload(src.out_proj, dst.out_proj, true);
    upload(src.fc1, dst.fc1, true);
    upload(src.fc2, dst.fc2, true);
    upload(src.adaln_w, dst.adaln_w, true);
    upload(src.adaln_b, dst.adaln_b, true);
  };

  // --- fp32 ISLANDS (never rounded, even in the bf16 stream) ---
  upload(host.video_patch_proj_w, staged.weights.video_patch_proj_w, false);
  upload(host.video_patch_proj_b, staged.weights.video_patch_proj_b, false);
  upload(host.audio_patch_proj_w, staged.weights.audio_patch_proj_w, false);
  upload(host.audio_patch_proj_b, staged.weights.audio_patch_proj_b, false);
  upload(host.time_proj_in_w, staged.weights.time_proj_in_w, false);
  upload(host.time_proj_in_b, staged.weights.time_proj_in_b, false);
  upload(host.time_proj_out_w, staged.weights.time_proj_out_w, false);
  upload(host.time_proj_out_b, staged.weights.time_proj_out_b, false);
  upload(host.video_out_w, staged.weights.video_out_w, false);
  upload(host.video_out_b, staged.weights.video_out_b, false);
  upload(host.audio_out_w, staged.weights.audio_out_w, false);
  upload(host.audio_out_b, staged.weights.audio_out_b, false);
  // rope.inv_freq is consumed on the HOST (it builds the cos/sin cache), so it is
  // deliberately left as the caller's host view rather than uploaded. It is also an
  // fp32 buffer upstream, so no rounding question arises.
  staged.weights.rope_inv_freq = host.rope_inv_freq;

  // --- bf16-STORED modules ---
  upload(host.condition_proj_w, staged.weights.condition_proj_w, true);
  upload(host.condition_proj_b, staged.weights.condition_proj_b, true);
  for (size_t i = 0; i < host.refiner.size(); ++i) {
    upload_block(host.refiner[i], staged.weights.refiner[i]);
  }
  upload(host.refiner_final_norm, staged.weights.refiner_final_norm, true);
  for (size_t i = 0; i < host.blocks.size(); ++i) {
    upload_block(host.blocks[i], staged.weights.blocks[i]);
  }
  upload(host.final_norm, staged.weights.final_norm, true);
  upload(host.final_adaln_w, staged.weights.final_adaln_w, true);
  upload(host.final_adaln_b, staged.weights.final_adaln_b, true);
  (void)params;
  backend.Synchronize(queue);
  return staged;
}

MiniMaxH3DitDeviceWeights StageMiniMaxH3DitWeightsDequantBf16(
    vt::Queue& queue, const MiniMaxH3DitParams& params, const MiniMaxH3GgufDit& gguf) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  MiniMaxH3DitDeviceWeights staged;
  staged.weights = gguf.weights;

  // Reverse of the loader's own name->view binding: find which checkpoint name a
  // given view came from, so its ggml type id can be recovered.
  auto ggml_type_of = [&](const Tensor& t) -> const uint32_t* {
    for (const auto& kv : gguf.quant_storage) {
      if (kv.second.data() == static_cast<const uint8_t*>(t.data)) {
        const auto it = gguf.quant_ggml_type.find(kv.first);
        return it == gguf.quant_ggml_type.end() ? nullptr : &it->second;
      }
    }
    return nullptr;
  };

  auto upload = [&](const Tensor& src, Tensor& dst) {
    if (src.data == nullptr) {
      dst = src;
      return;
    }
    const std::vector<int64_t> shape(src.shape, src.shape + src.rank);
    std::vector<uint16_t> bf16;
    if (vt::IsBlockQuant(src.dtype)) {
      const uint32_t* type = ggml_type_of(src);
      VT_CHECK(type != nullptr,
               "minimax_h3 stage-dequant: a block-quant weight has no recorded ggml type");
      // Straight to bf16: dequantizing to f32 first would double the peak and buy
      // nothing, since bf16 is what the GEMM will consume.
      bf16 = DequantGgufRowToBf16(*type, static_cast<const uint8_t*>(src.data),
                                  src.shape[0] * src.shape[1]);
    } else {
      VT_CHECK(src.dtype == DType::kF32, "minimax_h3 stage-dequant: expected f32 or block-quant");
      const int64_t n = src.Numel();
      bf16.resize(static_cast<size_t>(n));
      const float* in = src.Ptr<float>();
      for (int64_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &in[i], sizeof(bits));
        if ((bits & 0x7F800000u) == 0x7F800000u) {
          bits &= 0xFFFF0000u;
        } else {
          const uint32_t lsb = (bits >> 16) & 1u;
          bits = (bits + 0x7FFFu + lsb) & 0xFFFF0000u;
        }
        bf16[static_cast<size_t>(i)] = static_cast<uint16_t>(bits >> 16);
      }
    }
    const size_t bytes = bf16.size() * sizeof(uint16_t);
    void* p = backend.Alloc(bytes);
    std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
    backend.Copy(queue, p, bf16.data(), bytes);
    backend.Synchronize(queue);  // `bf16` dies at the end of this scope
    dst = dense_attn::MakeTensor(p, DType::kBF16, queue.device, shape);
    staged.storage.push_back(std::move(owner));
  };

  auto upload_block = [&](const MiniMaxH3DitBlockWeights& src, MiniMaxH3DitBlockWeights& dst) {
    upload(src.norm1, dst.norm1);
    upload(src.norm2, dst.norm2);
    upload(src.qkv_proj, dst.qkv_proj);
    upload(src.q_norm, dst.q_norm);
    upload(src.k_norm, dst.k_norm);
    upload(src.out_proj, dst.out_proj);
    upload(src.fc1, dst.fc1);
    upload(src.fc2, dst.fc2);
    upload(src.adaln_w, dst.adaln_w);
    upload(src.adaln_b, dst.adaln_b);
  };

  const MiniMaxH3DitWeights& host = gguf.weights;
  upload(host.video_patch_proj_w, staged.weights.video_patch_proj_w);
  upload(host.video_patch_proj_b, staged.weights.video_patch_proj_b);
  upload(host.audio_patch_proj_w, staged.weights.audio_patch_proj_w);
  upload(host.audio_patch_proj_b, staged.weights.audio_patch_proj_b);
  upload(host.condition_proj_w, staged.weights.condition_proj_w);
  upload(host.condition_proj_b, staged.weights.condition_proj_b);
  upload(host.time_proj_in_w, staged.weights.time_proj_in_w);
  upload(host.time_proj_in_b, staged.weights.time_proj_in_b);
  upload(host.time_proj_out_w, staged.weights.time_proj_out_w);
  upload(host.time_proj_out_b, staged.weights.time_proj_out_b);
  staged.weights.rope_inv_freq = host.rope_inv_freq;  // consumed on the host
  for (size_t i = 0; i < host.refiner.size(); ++i) {
    upload_block(host.refiner[i], staged.weights.refiner[i]);
  }
  upload(host.refiner_final_norm, staged.weights.refiner_final_norm);
  for (size_t i = 0; i < host.blocks.size(); ++i) {
    upload_block(host.blocks[i], staged.weights.blocks[i]);
  }
  upload(host.final_norm, staged.weights.final_norm);
  upload(host.final_adaln_w, staged.weights.final_adaln_w);
  upload(host.final_adaln_b, staged.weights.final_adaln_b);
  upload(host.video_out_w, staged.weights.video_out_w);
  upload(host.video_out_b, staged.weights.video_out_b);
  upload(host.audio_out_w, staged.weights.audio_out_w);
  upload(host.audio_out_b, staged.weights.audio_out_b);
  (void)params;
  backend.Synchronize(queue);
  return staged;
}

MiniMaxH3DitOutputs MiniMaxH3DitForwardDevice(vt::Queue& queue,
                                              const MiniMaxH3DitParams& params,
                                              const MiniMaxH3DitWeights& weights,
                                              const MiniMaxH3DitInputs& inputs,
                                              DType compute_dtype) {
  VT_CHECK(compute_dtype == DType::kF32 || compute_dtype == DType::kBF16,
           "minimax_h3: the device forward computes in f32 (parity) or bf16 (the "
           "production stream policy)");
  VT_CHECK(static_cast<int64_t>(weights.blocks.size()) == params.num_layers,
           "minimax_h3: block weight count does not match num_layers");
  VT_CHECK(static_cast<int64_t>(weights.refiner.size()) == params.token_refiner_num_layers,
           "minimax_h3: refiner weight count does not match token_refiner_num_layers");
  VT_CHECK(inputs.num_cu_seqlens >= 2, "minimax_h3: packed_seq_params.cu_seqlens is required");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const auto* glue = Glue(d);

  // DIAGNOSTIC (env-gated, byte-identical when unset): VT_H3_ACT_DUMP=<file> writes a
  // per-STAGE activation fingerprint of THIS forward -- stats (mean/rms/absmax/finite)
  // plus a fixed set of positional sample values -- so two weight arms that run the
  // SAME graph on the SAME inputs (e.g. the NVFP4-bf16 stream vs the FL2VA-GGUF-bf16
  // control, whose weights differ only by quant noise) can be diffed layer-by-layer to
  // localise WHERE they diverge. A JUMP at a stage names the guilty tensor class; a
  // scramble/transpose is caught by the positional samples even when rms matches. Only
  // the forward whose 0-based call index equals VT_H3_ACT_CALL (default 0) dumps, so a
  // single small render (--denoise-only --steps 1) captures exactly one clean forward.
  std::FILE* act_f = nullptr;
  {
    const char* act_path = std::getenv("VT_H3_ACT_DUMP");
    if (act_path != nullptr) {
      static int act_call_no = -1;  // process-global; resets per process (per arm)
      const int this_call = ++act_call_no;
      const char* want = std::getenv("VT_H3_ACT_CALL");
      const int want_call = (want != nullptr) ? std::atoi(want) : 0;
      if (this_call == want_call) act_f = std::fopen(act_path, "w");
    }
  }
  auto dump_act = [&](const char* stage, const Tensor& t) {
    H3DumpFingerprint(act_f, backend, queue, stage, t);
  };

  // Input-independent WEIGHT fingerprints for the islands + every bias + the output
  // heads -- the classes #94's oracle never sampled (it sampled adaln/qkv/out/fc
  // WEIGHTS only, never the biases or islands). Comparing these between the two arms
  // isolates a bias/island materialization difference (bf16-disk vs f16-disk).
  auto W = [&](const char* n, const Tensor& t) { if (t.data != nullptr) dump_act(n, t); };
  W("Wg.video_patch_w", weights.video_patch_proj_w);
  W("Wg.video_patch_b", weights.video_patch_proj_b);
  W("Wg.audio_patch_w", weights.audio_patch_proj_w);
  W("Wg.audio_patch_b", weights.audio_patch_proj_b);
  W("Wg.condition_b", weights.condition_proj_b);
  W("Wg.time_in_w", weights.time_proj_in_w);
  W("Wg.time_in_b", weights.time_proj_in_b);
  W("Wg.time_out_w", weights.time_proj_out_w);
  W("Wg.time_out_b", weights.time_proj_out_b);
  W("Wg.final_norm", weights.final_norm);
  W("Wg.final_adaln_b", weights.final_adaln_b);
  W("Wg.video_out_w", weights.video_out_w);
  W("Wg.video_out_b", weights.video_out_b);
  W("Wg.audio_out_w", weights.audio_out_w);
  W("Wg.audio_out_b", weights.audio_out_b);

  // kBF16 reproduces upstream's PRODUCTION dtype policy: the block stream is bf16
  // while the patch projections, the time embedder and both output heads stay fp32
  // islands (minimax_h3_transformer.py:85-101). Both dtypes run the SAME code; only
  // the rounding points differ -- exactly as in the CPU reference.
  const DeviceStreamDtype dt{compute_dtype == DType::kBF16};

  const int64_t seq_len = inputs.seq_len;
  const int64_t hidden = params.hidden_size;
  const int64_t video_width = params.video_row_width();
  const int64_t audio_width = params.audio_latents_dim;
  const int64_t m = inputs.num_unique_timesteps;

  // --- indices, uploaded once (i32: what IndexSelect/IndexCopy and the glue take) ---
  auto to_i32 = [](const int64_t* src, int64_t n) {
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = static_cast<int32_t>(src[i]);
    return out;
  };
  const std::vector<int32_t> img_pos = to_i32(inputs.img_pos, inputs.num_img_pos);
  const std::vector<int32_t> audio_pos = to_i32(inputs.audio_pos, inputs.num_audio_pos);
  const std::vector<int32_t> text_pos = to_i32(inputs.text_pos, inputs.num_text_pos);
  const std::vector<int32_t> infer_out_pos = to_i32(inputs.infer_out_pos, inputs.num_infer_out_pos);

  // combined_indices = inverse_indices * modality_num + token_tags.clamp(min=0)
  // (minimax_h3_transformer.py:1057).
  std::vector<int32_t> combined(static_cast<size_t>(seq_len));
  std::vector<int32_t> inverse(static_cast<size_t>(seq_len));
  for (int64_t i = 0; i < seq_len; ++i) {
    const int64_t tag = inputs.token_tags[i] < 0 ? 0 : inputs.token_tags[i];
    combined[static_cast<size_t>(i)] =
        static_cast<int32_t>(inputs.inverse_indices[i] * kMiniMaxH3AdalnModalityNum + tag);
    inverse[static_cast<size_t>(i)] = static_cast<int32_t>(inputs.inverse_indices[i]);
  }
  DBuf d_img_pos(d, DType::kI32, {inputs.num_img_pos}, img_pos.data());
  DBuf d_audio_pos(d, DType::kI32, {inputs.num_audio_pos}, audio_pos.data());
  DBuf d_text_pos(d, DType::kI32, {inputs.num_text_pos}, text_pos.data());
  DBuf d_infer_pos(d, DType::kI32, {inputs.num_infer_out_pos}, infer_out_pos.data());
  DBuf d_combined(d, DType::kI32, {seq_len}, combined.data());
  DBuf d_inverse(d, DType::kI32, {seq_len}, inverse.data());

  // --- RoPE cache over the full packed sequence (:1040-1041) ---
  const std::vector<float> rope_cache_host =
      BuildRopeCosSin(inputs.img_position_ids, seq_len, weights.rope_inv_freq.Ptr<float>(),
                      params.rope_inv_freq_len);
  const int64_t rot_dim = params.rope_rot_dim();
  // vt::RopeFromCache requires q/k/cache to share a dtype, so a bf16 stream gets a
  // bf16 cache -- the same shape the project's other bf16 models use (qwen3_vl
  // builds its cache in f32 then CastBf16's it). The ANGLES are still computed in
  // f32 on the host; only the cached cos/sin are rounded.
  DBuf d_rope_cache_f32(d, DType::kF32, {seq_len, rot_dim}, rope_cache_host.data());
  DBuf d_rope_cache(d, dt.S(), {seq_len, rot_dim});
  CastTo(d, d_rope_cache.t(), d_rope_cache_f32.t());
  dump_act("rope.cache_f32", d_rope_cache_f32.t());
  std::vector<int32_t> arange(static_cast<size_t>(seq_len));
  for (int64_t i = 0; i < seq_len; ++i) arange[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  DBuf d_rope_pos(d, DType::kI32, {seq_len}, arange.data());

  // --- _embed (minimax_h3_transformer.py:944-984) ---
  DBuf d_x(d, DType::kF32, {seq_len, video_width}, inputs.x);
  DBuf video_rows(d, DType::kF32, {inputs.num_img_pos, video_width});
  vt::IndexSelect(d.q, video_rows.t(), d_x.t(), d_img_pos.t());
  DBuf video_embed(d, DType::kF32, {inputs.num_img_pos, hidden});
  LinearDev(d, video_rows.t(), inputs.num_img_pos, video_width, weights.video_patch_proj_w,
            &weights.video_patch_proj_b, video_embed.t());
  dump_act("embed.video_rows_in", video_rows.t());
  dump_act("embed.video_patch_out", video_embed.t());

  DBuf d_audio_x(d, DType::kF32, {seq_len, audio_width}, inputs.audio_x);
  DBuf audio_rows(d, DType::kF32, {inputs.num_audio_pos, audio_width});
  vt::IndexSelect(d.q, audio_rows.t(), d_audio_x.t(), d_audio_pos.t());
  DBuf audio_embed(d, DType::kF32, {inputs.num_audio_pos, hidden});
  LinearDev(d, audio_rows.t(), inputs.num_audio_pos, audio_width, weights.audio_patch_proj_w,
            &weights.audio_patch_proj_b, audio_embed.t());
  dump_act("embed.audio_patch_out", audio_embed.t());

  // text rows enter as the stream dtype before the BF16 condition projection.
  // text rows enter as the stream dtype before the BF16 condition projection.
  DBuf text_rows_f32(d, DType::kF32, {inputs.num_text_pos, params.text_dim}, inputs.prompt_embeds);
  DBuf text_rows(d, dt.S(), {inputs.num_text_pos, params.text_dim});
  CastTo(d, text_rows.t(), text_rows_f32.t());
  DBuf text_embed(d, dt.S(), {inputs.num_text_pos, hidden});
  LinearDev(d, text_rows.t(), inputs.num_text_pos, params.text_dim, weights.condition_proj_w,
            &weights.condition_proj_b, text_embed.t(), &weights.condition_fp4);
  dump_act("embed.text_condition_out", text_embed.t());

  // Token refiner: a plain pre-norm stack, no AdaLN and no RoPE (:564-623), on the
  // REPLICATED text rows, so it uses the refiner's own cu_seqlens.
  {
    const int64_t rows = inputs.num_text_pos;
    DBuf normed(d, dt.S(), {rows, hidden});
    DBuf tmp(d, dt.S(), {rows, hidden});
    vt::RmsNormArgs args;
    args.eps = static_cast<float>(params.norm_eps);
    int64_t ref_idx = -1;
    for (const MiniMaxH3DitBlockWeights& block : weights.refiner) {
      ++ref_idx;
      if (ref_idx == 0) {
        dump_act("W.ref0.norm1", block.norm1);
        if (block.qkv_proj.data != nullptr) dump_act("W.ref0.qkv_proj", block.qkv_proj);
        dump_act("W.ref0.q_norm", block.q_norm);
        dump_act("W.ref0.k_norm", block.k_norm);
        if (block.out_proj.data != nullptr) dump_act("W.ref0.out_proj", block.out_proj);
        if (block.fc1.data != nullptr) dump_act("W.ref0.fc1", block.fc1);
        if (block.fc2.data != nullptr) dump_act("W.ref0.fc2", block.fc2);
        dump_act("W.ref0.norm2", block.norm2);
      }
      vt::RmsNorm(d.q, normed.t(), text_embed.t(), block.norm1, args);
      AttentionDev(d, params,
                   AttnWeightsDev{&block.qkv_proj, &block.q_norm, &block.k_norm, &block.out_proj,
                                  &block.qkv_fp4, &block.out_fp4},
                   normed.t(), rows, nullptr, nullptr, inputs.refiner_cu_seqlens,
                   static_cast<int>(inputs.num_refiner_cu_seqlens - 1), dt, tmp.t());
      // FOLD onto the catalog recipe. This was DECLINED one milestone ago because
      // the recipe casts nothing between its `residual = x + residual` and its
      // `rms_norm(residual)`, which would have dropped a bf16 rounding. TRUE bf16
      // STORAGE removes that objection: the residual operand IS bf16, so the add is
      // rounded on store before the f32 variance (vt::RmsNorm's documented
      // bf16-residual behaviour, matching csrc fused_add_rms_norm). Same cast point,
      // one launch instead of two.
      vt::FusedChain(d.q, normed.t(), tmp.t(), block.norm2, &text_embed.t(),
                     vt::kFusedAddRmsNormStd, args.eps);
      MlpDev(d, params, block.fc1, block.fc2, normed.t(), rows, dt, tmp.t(), &block.fc1_fp4,
             &block.fc2_fp4);
      vt::Add(d.q, text_embed.t(), text_embed.t(), tmp.t());
    }
    vt::RmsNormArgs final_args;
    final_args.eps = static_cast<float>(params.final_norm_eps);
    vt::RmsNorm(d.q, normed.t(), text_embed.t(), weights.refiner_final_norm, final_args);
    backend.Copy(d.q, text_embed.ptr(), normed.ptr(), normed.bytes());
  }
  dump_act("embed.text_refined_out", text_embed.t());

  // index_add_ scatter of the three modality embeddings into the packed stream.
  // The three position sets are DISJOINT by construction (each packed row carries
  // exactly one modality), so torch's index_add_ into a zeroed stream is an
  // index_COPY here — asserted rather than assumed, since a silent overlap would
  // turn a lost add into a wrong answer.
  {
    std::vector<uint8_t> seen(static_cast<size_t>(seq_len), 0);
    auto mark = [&](const std::vector<int32_t>& pos) {
      for (int32_t p : pos) {
        VT_CHECK(p >= 0 && p < seq_len, "minimax_h3: scatter position out of range");
        VT_CHECK(seen[static_cast<size_t>(p)] == 0,
                 "minimax_h3: modality scatter positions overlap; index_copy would drop an add");
        seen[static_cast<size_t>(p)] = 1;
      }
    };
    mark(text_pos);
    mark(img_pos);
    mark(audio_pos);
  }
  // The patch projections are fp32 ISLANDS; their outputs enter the bf16 stream
  // only at this indexed scatter (minimax_h3_transformer.py:978-981), which is
  // exactly where the cast belongs.
  DBuf video_s(d, dt.S(), {inputs.num_img_pos, hidden});
  DBuf audio_s(d, dt.S(), {inputs.num_audio_pos, hidden});
  CastTo(d, video_s.t(), video_embed.t());
  CastTo(d, audio_s.t(), audio_embed.t());
  DBuf stream(d, dt.S(), {seq_len, hidden});
  stream.Zero(d);
  vt::IndexCopy(d.q, stream.t(), text_embed.t(), d_text_pos.t());
  vt::IndexCopy(d.q, stream.t(), video_s.t(), d_img_pos.t());
  vt::IndexCopy(d.q, stream.t(), audio_s.t(), d_audio_pos.t());
  dump_act("embed.stream_post_scatter", stream.t());

  // --- time embedding (minimax_h3_transformer.py:272-285) ---
  DBuf t_emb(d, DType::kF32, {m, params.time_embed_dim});
  {
    const int64_t half = params.timestep_input_dim / 2;
    std::vector<float> t_freq(static_cast<size_t>(m * params.timestep_input_dim));
    for (int64_t r = 0; r < m; ++r) {
      for (int64_t i = 0; i < half; ++i) {
        const double freq =
            std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(half));
        const double arg = static_cast<double>(inputs.unique_timesteps[r]) * freq;
        // Cosine values are concatenated BEFORE sine values.
        t_freq[static_cast<size_t>(r * params.timestep_input_dim + i)] =
            static_cast<float>(std::cos(arg));
        t_freq[static_cast<size_t>(r * params.timestep_input_dim + half + i)] =
            static_cast<float>(std::sin(arg));
      }
    }
    DBuf d_t_freq(d, DType::kF32, {m, params.timestep_input_dim}, t_freq.data());
    DBuf mid(d, DType::kF32, {m, params.time_embed_hidden_size});
    LinearDev(d, d_t_freq.t(), m, params.timestep_input_dim, weights.time_proj_in_w,
              &weights.time_proj_in_b, mid.t());
    glue->silu(d.q, mid.t().Ptr<float>(), m * params.time_embed_hidden_size);
    LinearDev(d, mid.t(), m, params.time_embed_hidden_size, weights.time_proj_out_w,
              &weights.time_proj_out_b, t_emb.t());
  }
  dump_act("time.t_emb", t_emb.t());

  // silu(t_emb), hoisted: t_emb is loop-invariant, so the reference's per-block
  // silu inside AdalnProject is redundant work here.
  DBuf t_emb_act(d, DType::kF32, {m, params.time_embed_dim});
  backend.Copy(d.q, t_emb_act.ptr(), t_emb.ptr(), t_emb.bytes());
  glue->silu(d.q, t_emb_act.t().Ptr<float>(), m * params.time_embed_dim);
  // silu(t_emb) is fp32; cast to the BF16 AdaLN linear's dtype before the GEMM.
  DBuf t_emb_s(d, dt.S(), {m, params.time_embed_dim});
  CastTo(d, t_emb_s.t(), t_emb_act.t());

  // --- the DiT block stack (minimax_h3_transformer.py:645-688) ---
  const int num_reqs = static_cast<int>(inputs.num_cu_seqlens - 1);
  const int64_t adaln_rows = m * kMiniMaxH3AdalnModalityNum;
  DBuf normed(d, dt.S(), {seq_len, hidden});
  DBuf tmp(d, dt.S(), {seq_len, hidden});
  DBuf projected(d, dt.S(), {adaln_rows, 6 * hidden});
  vt::RmsNormArgs block_args;
  block_args.eps = static_cast<float>(params.norm_eps);

  // Chunk c of AdaLN row r lives at (r * expand + c) * hidden within the flat
  // projection; as a TENSOR that is a row-strided [rows, hidden] view, so the
  // chunks need no copy — MatmulBT wrote them where the modulates want them.
  auto chunk_view = [&](DBuf& buf, int64_t rows, int64_t expand, int64_t c) {
    std::byte* base = static_cast<std::byte*>(buf.ptr()) +
                      static_cast<size_t>(c * hidden) * vt::SizeOf(dt.S());
    Tensor t = MakeTensor(base, dt.S(), d.q.device, {rows, hidden});
    t.stride[0] = expand * hidden;  // row stride skips the sibling chunks
    return t;
  };

  int64_t blk_idx = -1;
  for (const MiniMaxH3DitBlockWeights& block : weights.blocks) {
    ++blk_idx;
    // Direct WEIGHT fingerprints for block 0 -- comparing these between the two arms
    // isolates a loader/materialization bug (a weight class that differs) from a
    // compute divergence. The bf16 arm has real bf16 tensors here (fp4-resident leaves
    // the projection Empty, so those print n=0).
    if (blk_idx == 0) {
      dump_act("W.blk0.norm1", block.norm1);
      if (block.qkv_proj.data != nullptr) dump_act("W.blk0.qkv_proj", block.qkv_proj);
      dump_act("W.blk0.q_norm", block.q_norm);
      dump_act("W.blk0.k_norm", block.k_norm);
      if (block.out_proj.data != nullptr) dump_act("W.blk0.out_proj", block.out_proj);
      if (block.fc1.data != nullptr) dump_act("W.blk0.fc1", block.fc1);
      if (block.fc2.data != nullptr) dump_act("W.blk0.fc2", block.fc2);
      dump_act("W.blk0.norm2", block.norm2);
      if (block.adaln_w.data != nullptr) dump_act("W.blk0.adaln_w", block.adaln_w);
      if (block.adaln_b.data != nullptr) dump_act("W.blk0.adaln_b", block.adaln_b);
    }
    AdalnProjectDev(d, t_emb_s.t(), m, params.time_embed_dim, block.adaln_w, block.adaln_b,
                    projected.t(), &block.adaln_fp4);
    if (act_f != nullptr) dump_act(("block." + std::to_string(blk_idx) + ".adaln_proj").c_str(),
                                   projected.t());
    const Tensor shift_msa = chunk_view(projected, adaln_rows, 6, 0);
    const Tensor scale_msa = chunk_view(projected, adaln_rows, 6, 1);
    const Tensor gate_msa = chunk_view(projected, adaln_rows, 6, 2);
    const Tensor shift_mlp = chunk_view(projected, adaln_rows, 6, 3);
    const Tensor scale_mlp = chunk_view(projected, adaln_rows, 6, 4);
    const Tensor gate_mlp = chunk_view(projected, adaln_rows, 6, 5);

    vt::RmsNorm(d.q, normed.t(), stream.t(), block.norm1, block_args);
    glue->modulate_scale_shift(d.q, normed.t().data, shift_msa.data, scale_msa.data,
                               d_combined.t().Ptr<int32_t>(), seq_len, hidden, 6 * hidden,
                               dt.S());
    if (blk_idx == 0) dump_act("block.0.normed_pre_attn", normed.t());
    AttentionDev(d, params,
                 AttnWeightsDev{&block.qkv_proj, &block.q_norm, &block.k_norm, &block.out_proj,
                                &block.qkv_fp4, &block.out_fp4},
                 normed.t(), seq_len, &d_rope_cache.t(), &d_rope_pos.t(), inputs.cu_seqlens,
                 num_reqs, dt, tmp.t(), blk_idx == 0 ? act_f : nullptr, "block.0.attn");
    if (blk_idx == 0) dump_act("block.0.attn_contrib", tmp.t());
    glue->modulate_gate(d.q, stream.t().data, gate_msa.data, tmp.t().data,
                        d_combined.t().Ptr<int32_t>(), seq_len, hidden, 6 * hidden, dt.S());
    if (act_f != nullptr)
      dump_act(("block." + std::to_string(blk_idx) + ".post_attn").c_str(), stream.t());

    vt::RmsNorm(d.q, normed.t(), stream.t(), block.norm2, block_args);
    glue->modulate_scale_shift(d.q, normed.t().data, shift_mlp.data, scale_mlp.data,
                               d_combined.t().Ptr<int32_t>(), seq_len, hidden, 6 * hidden,
                               dt.S());
    MlpDev(d, params, block.fc1, block.fc2, normed.t(), seq_len, dt, tmp.t(), &block.fc1_fp4,
           &block.fc2_fp4);
    glue->modulate_gate(d.q, stream.t().data, gate_mlp.data, tmp.t().data,
                        d_combined.t().Ptr<int32_t>(), seq_len, hidden, 6 * hidden, dt.S());
    if (act_f != nullptr)
      dump_act(("block." + std::to_string(blk_idx) + ".post_mlp").c_str(), stream.t());
  }

  // --- final layer (minimax_h3_transformer.py:724-743) ---
  DBuf final_projected(d, dt.S(), {m, 2 * hidden});
  AdalnProjectDev(d, t_emb_s.t(), m, params.time_embed_dim, weights.final_adaln_w,
                  weights.final_adaln_b, final_projected.t(), &weights.final_adaln_fp4);
  const Tensor final_shift = chunk_view(final_projected, m, 2, 0);
  const Tensor final_scale = chunk_view(final_projected, m, 2, 1);
  vt::RmsNormArgs final_args;
  final_args.eps = static_cast<float>(params.final_norm_eps);
  vt::RmsNorm(d.q, normed.t(), stream.t(), weights.final_norm, final_args);
  // The final layer is single-modality, so it indexes by inverse_indices directly.
  glue->modulate_scale_shift(d.q, normed.t().data, final_shift.data, final_scale.data,
                             d_inverse.t().Ptr<int32_t>(), seq_len, hidden, 2 * hidden, dt.S());
  dump_act("final.adaln_proj", final_projected.t());
  dump_act("final.normed", normed.t());
  // Cast UP before both output heads: they are fp32 ISLANDS.
  DBuf head_in(d, DType::kF32, {seq_len, hidden});
  CastTo(d, head_in.t(), normed.t());

  DBuf video_all(d, DType::kF32, {seq_len, video_width});
  LinearDev(d, head_in.t(), seq_len, hidden, weights.video_out_w, &weights.video_out_b,
            video_all.t());
  DBuf audio_all(d, DType::kF32, {seq_len, audio_width});
  LinearDev(d, head_in.t(), seq_len, hidden, weights.audio_out_w, &weights.audio_out_b,
            audio_all.t());
  dump_act("final.video_all", video_all.t());
  dump_act("final.audio_all", audio_all.t());
  if (act_f != nullptr) std::fclose(act_f);

  // Select the inference-output rows ON DEVICE, so only the selected rows cross
  // the bus (:1087-1101).
  MiniMaxH3DitOutputs out;
  DBuf video_sel(d, DType::kF32, {inputs.num_infer_out_pos, video_width});
  vt::IndexSelect(d.q, video_sel.t(), video_all.t(), d_infer_pos.t());
  out.video_logits.resize(static_cast<size_t>(inputs.num_infer_out_pos * video_width));
  video_sel.Download(d, out.video_logits.data());

  DBuf audio_sel(d, DType::kF32, {inputs.num_audio_pos, audio_width});
  vt::IndexSelect(d.q, audio_sel.t(), audio_all.t(), d_audio_pos.t());
  out.audio_logits.resize(static_cast<size_t>(inputs.num_audio_pos * audio_width));
  audio_sel.Download(d, out.audio_logits.data());

  // Zeroing the pinned condition rows stays on the host: the rows are already
  // here, and a kernel for it would buy nothing.
  if (!inputs.skip_mask_out_condition) {
    VT_CHECK(inputs.update_mask != nullptr, "minimax_h3: update_mask is required");
    for (int64_t r = 0; r < inputs.num_infer_out_pos; ++r) {
      if (inputs.update_mask[r]) continue;
      std::fill_n(out.video_logits.data() + r * video_width, video_width, 0.0f);
    }
    if (inputs.audio_update_mask != nullptr) {
      for (int64_t r = 0; r < inputs.num_audio_pos; ++r) {
        if (inputs.audio_update_mask[r]) continue;
        std::fill_n(out.audio_logits.data() + r * audio_width, audio_width, 0.0f);
      }
    }
  }
  return out;
}


namespace {

// Bind the forward's views over the DEVICE tensors, by the same names the
// checkpoint uses, so a missing tensor still throws BY NAME.
//
// Shared by BOTH streaming stagers (GGUF and NVFP4). One copy deliberately: a
// name-by-name binding this long would drift the moment a tensor is added to one
// path only. `rope.inv_freq` is NOT bound here -- it is the one entry that must
// stay HOST-resident, and each loader reads it from its own container.
void BindStreamedDitViews(const std::map<std::string, Tensor>& views,
                          const MiniMaxH3DitParams& params, MiniMaxH3DitWeights* out) {
  auto view = [&](const std::string& name) -> Tensor {
    const auto it = views.find(name);
    VT_CHECK(it != views.end(), "minimax_h3 stream: checkpoint is missing a required tensor");
    return it->second;
  };
  MiniMaxH3DitWeights& w = *out;
  w.video_patch_proj_w = view("video_patch_proj.weight");
  w.video_patch_proj_b = view("video_patch_proj.bias");
  w.audio_patch_proj_w = view("audio_patch_proj.weight");
  w.audio_patch_proj_b = view("audio_patch_proj.bias");
  w.condition_proj_w = view("condition_proj.weight");
  w.condition_proj_b = view("condition_proj.bias");
  w.time_proj_in_w = view("time_embedder.proj_in.weight");
  w.time_proj_in_b = view("time_embedder.proj_in.bias");
  w.time_proj_out_w = view("time_embedder.proj_out.weight");
  w.time_proj_out_b = view("time_embedder.proj_out.bias");
  auto block = [&](const std::string& prefix, bool adaln) {
    MiniMaxH3DitBlockWeights b;
    b.norm1 = view(prefix + ".norm1.weight");
    b.norm2 = view(prefix + ".norm2.weight");
    b.qkv_proj = view(prefix + ".attn.qkv_proj.weight");
    b.q_norm = view(prefix + ".attn.q_norm.weight");
    b.k_norm = view(prefix + ".attn.k_norm.weight");
    b.out_proj = view(prefix + ".attn.out_proj.weight");
    b.fc1 = view(prefix + ".mlp.fc1.weight");
    b.fc2 = view(prefix + ".mlp.fc2.weight");
    if (adaln) {
      b.adaln_w = view(prefix + ".adaln_proj.linear.weight");
      b.adaln_b = view(prefix + ".adaln_proj.linear.bias");
    }
    return b;
  };
  for (int64_t i = 0; i < params.token_refiner_num_layers; ++i) {
    w.refiner.push_back(block("token_refiner.blocks." + std::to_string(i), false));
  }
  w.refiner_final_norm = view("token_refiner.final_norm.weight");
  for (int64_t i = 0; i < params.num_layers; ++i) {
    w.blocks.push_back(block("blocks." + std::to_string(i), true));
  }
  w.final_norm = view("final_layer.norm.weight");
  w.final_adaln_w = view("final_layer.adaln_proj.linear.weight");
  w.final_adaln_b = view("final_layer.adaln_proj.linear.bias");
  w.video_out_w = view("final_layer.video_out.weight");
  w.video_out_b = view("final_layer.video_out.bias");
  w.audio_out_w = view("final_layer.audio_out.weight");
  w.audio_out_b = view("final_layer.audio_out.bias");
}

// W-FP4a binder: the fp4-resident twin of BindStreamedDitViews. Every non-quantized
// tensor (norms, biases, fp32 islands) is bound from `views` exactly as above; each
// quantized projection is MOVED out of `fp4` into its Nvfp4Weight slot, and its bf16
// `vt::Tensor` slot is left Empty(). A projection absent from `fp4` was stored
// unquantized in the checkpoint and falls back to a `views` bind, so an
// island-only-quantized file still lands on the same contract.
void BindStreamedDitViewsFp4(const std::map<std::string, Tensor>& views,
                             std::map<std::string, Nvfp4Weight>& fp4,
                             const MiniMaxH3DitParams& params, MiniMaxH3DitWeights* out) {
  auto view = [&](const std::string& name) -> Tensor {
    const auto it = views.find(name);
    VT_CHECK(it != views.end(),
             "minimax_h3 nvfp4-fp4: checkpoint is missing a required tensor");
    return it->second;
  };
  // Route a projection: fp4-resident when present in `fp4`, else its bf16 view.
  auto proj = [&](const std::string& name, Nvfp4Weight& wdst, Tensor& tdst) {
    const auto it = fp4.find(name);
    if (it != fp4.end()) {
      wdst = std::move(it->second);
      tdst = Tensor{};
    } else {
      tdst = view(name);
    }
  };
  MiniMaxH3DitWeights& w = *out;
  w.video_patch_proj_w = view("video_patch_proj.weight");
  w.video_patch_proj_b = view("video_patch_proj.bias");
  w.audio_patch_proj_w = view("audio_patch_proj.weight");
  w.audio_patch_proj_b = view("audio_patch_proj.bias");
  proj("condition_proj.weight", w.condition_fp4, w.condition_proj_w);
  w.condition_proj_b = view("condition_proj.bias");
  w.time_proj_in_w = view("time_embedder.proj_in.weight");
  w.time_proj_in_b = view("time_embedder.proj_in.bias");
  w.time_proj_out_w = view("time_embedder.proj_out.weight");
  w.time_proj_out_b = view("time_embedder.proj_out.bias");
  auto block = [&](const std::string& prefix, bool adaln) {
    MiniMaxH3DitBlockWeights b;
    b.norm1 = view(prefix + ".norm1.weight");
    b.norm2 = view(prefix + ".norm2.weight");
    proj(prefix + ".attn.qkv_proj.weight", b.qkv_fp4, b.qkv_proj);
    b.q_norm = view(prefix + ".attn.q_norm.weight");
    b.k_norm = view(prefix + ".attn.k_norm.weight");
    proj(prefix + ".attn.out_proj.weight", b.out_fp4, b.out_proj);
    proj(prefix + ".mlp.fc1.weight", b.fc1_fp4, b.fc1);
    proj(prefix + ".mlp.fc2.weight", b.fc2_fp4, b.fc2);
    if (adaln) {
      proj(prefix + ".adaln_proj.linear.weight", b.adaln_fp4, b.adaln_w);
      b.adaln_b = view(prefix + ".adaln_proj.linear.bias");
    }
    return b;
  };
  for (int64_t i = 0; i < params.token_refiner_num_layers; ++i) {
    w.refiner.push_back(block("token_refiner.blocks." + std::to_string(i), false));
  }
  w.refiner_final_norm = view("token_refiner.final_norm.weight");
  for (int64_t i = 0; i < params.num_layers; ++i) {
    w.blocks.push_back(block("blocks." + std::to_string(i), true));
  }
  w.final_norm = view("final_layer.norm.weight");
  proj("final_layer.adaln_proj.linear.weight", w.final_adaln_fp4, w.final_adaln_w);
  w.final_adaln_b = view("final_layer.adaln_proj.linear.bias");
  w.video_out_w = view("final_layer.video_out.weight");
  w.video_out_b = view("final_layer.video_out.bias");
  w.audio_out_w = view("final_layer.audio_out.weight");
  w.audio_out_b = view("final_layer.audio_out.bias");
}

}  // namespace

MiniMaxH3DitDeviceWeights StreamMiniMaxH3DitToDeviceBf16(vt::Queue& queue, const GgufFile& file,
                                                         MiniMaxH3DitParams* out_params) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const std::vector<MiniMaxH3TensorSpec> manifest = EnumerateMiniMaxH3GgufTensors(file);
  const MiniMaxH3DitParams params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  if (out_params != nullptr) *out_params = params;

  // Opt the mapping into page release: without this the DropSpanResidency calls
  // below are no-ops and the read-once file pages accumulate against the same
  // unified pool the weights live in.
  // NOTE: page release is OFF by default here. It was enabled once and the load was
  // SIGKILLed early at only ~21 GB peak — not a memory problem — so it is gated
  // behind VT_H3_DROP_PAGES until that is understood rather than left on by faith.
  const bool drop_pages = std::getenv("VT_H3_DROP_PAGES") != nullptr;
  if (drop_pages) file.ReleaseExpandedPages(true);
  const bool trace = std::getenv("VT_H3_PROGRESS") != nullptr;
  MiniMaxH3DitDeviceWeights staged;
  std::map<std::string, Tensor> views;
  // Upstream's fp32 ISLANDS (MINIMAX_H3_FP32_PARAM_NAMES / _BUFFER_NAMES): both patch
  // projections, both time-embedder projections and both output heads stay f32 even
  // in a bf16 stream. Their ACTIVATIONS are f32 too, and vt::MatmulBT rejects a
  // mixed (f32 act, bf16 weight) pair — so getting this split wrong is not a
  // precision nuance, it fails loudly at the first island GEMM. Single-sourced in
  // MiniMaxH3IsFp32IslandTensor so every staging path agrees on it.
  auto is_fp32_island = [](const std::string& n) { return MiniMaxH3IsFp32IslandTensor(n); };
  int64_t done = 0;
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const GgufTensorInfo& info = file.Get(spec.name);
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    VT_CHECK(numel > 0, "minimax_h3 stream: tensor has an empty logical shape");
    // Dequantize ONE tensor, upload it, and let the host buffer die before the next.
    // That is the whole point: peak stays at the device copy plus one tensor.
    {
      const bool island = is_fp32_island(spec.name);
      const DType want = island ? DType::kF32 : DType::kBF16;
      std::vector<float> f32;
      std::vector<uint16_t> bf16;
      const void* src = nullptr;
      size_t bytes = 0;
      if (island) {
        f32 = DequantGgufRowToF32(info.ggml_type, static_cast<const uint8_t*>(info.data), numel);
        VT_CHECK(static_cast<int64_t>(f32.size()) == numel,
                 "minimax_h3 stream: island dequant produced the wrong element count");
        src = f32.data();
        bytes = f32.size() * sizeof(float);
      } else {
        bf16 = DequantGgufRowToBf16(info.ggml_type, static_cast<const uint8_t*>(info.data), numel);
        VT_CHECK(static_cast<int64_t>(bf16.size()) == numel,
                 "minimax_h3 stream: dequant produced the wrong element count");
        src = bf16.data();
        bytes = bf16.size() * sizeof(uint16_t);
      }
      void* p = backend.Alloc(bytes);
      std::shared_ptr<void> owner(p, [&backend](void* q) { backend.Free(q); });
      backend.Copy(queue, p, src, bytes);
      backend.Synchronize(queue);  // the host buffer is about to go out of scope
      views[spec.name] = dense_attn::MakeTensor(p, want, queue.device, spec.shape);
      staged.storage.push_back(std::move(owner));
    }
    // Drop the file pages we will never read again — on a unified-memory box the
    // page cache competes with the model for the same pool.
    if (drop_pages) file.DropSpanResidency(static_cast<const uint8_t*>(info.data), info.nbytes);
    if (trace && (++done % 50 == 0 || done == static_cast<int64_t>(manifest.size()))) {
      std::fprintf(stderr, "[h3] streamed %lld/%zu tensors (last: %s)\n",
                   static_cast<long long>(done), manifest.size(), spec.name.c_str());
      std::fflush(stderr);
    }
  }

  MiniMaxH3DitWeights& w = staged.weights;
  // rope.inv_freq is read on the HOST (BuildRopeCosSin runs before any kernel), so
  // it is dequantized to host f32 and kept alive by the staged struct. Binding the
  // device tensor here segfaults on the first forward.
  {
    const GgufTensorInfo& info = file.Get("rope.inv_freq");
    int64_t n = 1;
    for (int64_t d : info.shape) n *= d;
    staged.rope_inv_freq_host =
        DequantGgufRowToF32(info.ggml_type, static_cast<const uint8_t*>(info.data), n);
    w.rope_inv_freq = vt::Tensor::Contiguous(staged.rope_inv_freq_host.data(), DType::kF32,
                                             vt::Device{}, {n});
  }
  BindStreamedDitViews(views, params, &staged.weights);
  return staged;
}


// Stream an NVFP4 (compressed-tensors) DiT STRAIGHT ONTO THE DEVICE as bf16.
//
// The non-streaming LoadMiniMaxH3DitFromNvfp4 materializes every weight as host
// f32, which for this checkpoint is ~132 GB and is an OOM kill on a 122 GiB
// unified box (observed: anon-rss 125 GB, killed during load). It stays as the
// portable/reference loader; this is the one a real run uses, and it mirrors
// StreamMiniMaxH3DitToDeviceBf16 exactly: dequantize ONE tensor, upload it, let
// the host buffer die before the next, so peak is the device copy plus one tensor.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3Nvfp4ToDeviceBf16(vt::Queue& queue,
                                                           const SafetensorsFile& file,
                                                           MiniMaxH3DitParams* out_params) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const bool trace = std::getenv("VT_H3_PROGRESS") != nullptr;

  auto is_sidecar = [](const std::string& n) {
    return (n.size() > 12 && n.compare(n.size() - 12, 12, "weight_scale") == 0) ||
           (n.size() > 14 && n.compare(n.size() - 14, 14, "weight_scale_2") == 0);
  };
  // Same fp32 ISLANDS as the GGUF stream: vt::MatmulBT rejects a mixed
  // (f32 activation, bf16 weight) pair, so this split is load-bearing, not a
  // precision nicety.
  auto is_fp32_island = [](const std::string& n) { return MiniMaxH3IsFp32IslandTensor(n); };

  // Pass 1: LOGICAL shapes only (no payload), so geometry is known before any
  // allocation. A packed [out, in/2] U8 weight is logically [out, in].
  std::vector<MiniMaxH3TensorSpec> manifest;
  for (const std::string& name : file.Names()) {
    if (is_sidecar(name)) continue;
    const StTensor& t = file.Get(name);
    MiniMaxH3TensorSpec spec;
    spec.name = name;
    spec.shape = t.shape;
    if (t.dtype == "U8") {
      VT_CHECK(t.shape.size() == 2, "minimax_h3 nvfp4 stream: a packed weight must be rank 2");
      spec.shape = {t.shape[0], t.shape[1] * 2};
    }
    manifest.push_back(std::move(spec));
  }
  const MiniMaxH3DitParams params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  if (out_params != nullptr) *out_params = params;

  MiniMaxH3DitDeviceWeights staged;
  std::map<std::string, Tensor> views;
  size_t done = 0;
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const StTensor& t = file.Get(spec.name);
    const bool island = is_fp32_island(spec.name);

    // rope.inv_freq is consumed on the HOST (BuildRopeCosSin runs before any
    // kernel); binding a device pointer here segfaults on the first forward.
    if (spec.name == "rope.inv_freq") {
      staged.rope_inv_freq_host = MiniMaxH3ReadSafetensorF32(t);
      staged.weights.rope_inv_freq = vt::Tensor::Contiguous(
          staged.rope_inv_freq_host.data(), DType::kF32, vt::Device{},
          {static_cast<int64_t>(staged.rope_inv_freq_host.size())});
      continue;
    }

    std::vector<uint16_t> bf16;
    std::vector<float> f32;
    const void* src = nullptr;
    size_t bytes = 0;
    DType want = island ? DType::kF32 : DType::kBF16;

    if (t.dtype == "U8") {
      const int64_t out_dim = spec.shape[0], in_dim = spec.shape[1];
      const StTensor& scale = file.Get(spec.name + "_scale");
      const StTensor& global = file.Get(spec.name + "_scale_2");
      VT_CHECK(scale.dtype == "F8_E4M3", "minimax_h3 nvfp4 stream: weight_scale must be F8_E4M3");
      VT_CHECK(global.dtype == "F32", "minimax_h3 nvfp4 stream: weight_scale_2 must be F32");
      VT_CHECK(scale.shape.size() == 2 && scale.shape[0] == out_dim &&
                   scale.shape[1] * 16 == in_dim,
               "minimax_h3 nvfp4 stream: weight_scale must be [out, in/16] (group size 16)");
      VT_CHECK(global.nbytes >= sizeof(float), "minimax_h3 nvfp4 stream: weight_scale_2 too small");
      float global_scale = 0.0f;
      std::memcpy(&global_scale, global.data, sizeof(float));
      bf16.resize(static_cast<size_t>(out_dim * in_dim));
      // Correct the community checkpoint's high-nibble-first fp4 packing before the
      // standard (low-first) dequant. See MiniMaxH3Nvfp4HighNibbleFirst.
      const uint8_t* packed = static_cast<const uint8_t*>(t.data);
      std::vector<uint8_t> swapped;
      if (MiniMaxH3Nvfp4HighNibbleFirst()) {
        swapped.resize(t.nbytes);
        MiniMaxH3Nvfp4SwapNibbles(static_cast<const uint8_t*>(t.data), t.nbytes, swapped.data());
        packed = swapped.data();
      }
      DequantNvfp4ToBf16(packed, static_cast<const uint8_t*>(scale.data), global_scale, out_dim,
                         in_dim, bf16.data());
      if (island) {
        f32.resize(bf16.size());
        for (size_t i = 0; i < bf16.size(); ++i) {
          const uint32_t widened = static_cast<uint32_t>(bf16[i]) << 16;
          std::memcpy(&f32[i], &widened, sizeof(float));
        }
        bf16.clear();
        bf16.shrink_to_fit();
        src = f32.data();
        bytes = f32.size() * sizeof(float);
      } else {
        src = bf16.data();
        bytes = bf16.size() * sizeof(uint16_t);
      }
    } else {
      f32 = MiniMaxH3ReadSafetensorF32(t);
      if (island) {
        src = f32.data();
        bytes = f32.size() * sizeof(float);
      } else {
        bf16.resize(f32.size());
        for (size_t i = 0; i < f32.size(); ++i) {
          uint32_t bits;
          std::memcpy(&bits, &f32[i], sizeof(bits));
          // round-to-nearest-even, the same rounding vt uses on a bf16 store
          const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
          bf16[i] = static_cast<uint16_t>(rounded >> 16);
        }
        f32.clear();
        f32.shrink_to_fit();
        src = bf16.data();
        bytes = bf16.size() * sizeof(uint16_t);
      }
    }

    void* pdev = backend.Alloc(bytes);
    std::shared_ptr<void> owner(pdev, [&backend](void* q) { backend.Free(q); });
    backend.Copy(queue, pdev, src, bytes);
    backend.Synchronize(queue);  // the host buffer dies at the end of this iteration
    views[spec.name] = dense_attn::MakeTensor(pdev, want, queue.device, spec.shape);
    staged.storage.push_back(std::move(owner));

    if (trace && (++done % 50 == 0 || done == manifest.size())) {
      std::fprintf(stderr, "[h3] nvfp4-streamed %zu/%zu tensors (last: %s)\n", done,
                   manifest.size(), spec.name.c_str());
      std::fflush(stderr);
    }
  }

  BindStreamedDitViews(views, params, &staged.weights);
  return staged;
}

// W-FP4a: the fp4-RESIDENT NVFP4 streamer. Structurally identical to
// StreamMiniMaxH3Nvfp4ToDeviceBf16, except a U8-packed projection is KEPT as an
// Nvfp4Weight (host packed + E4M3 scale + f32 global) instead of being dequantized
// to bf16 and uploaded — so the device forward routes it through the Marlin W4A16
// GEMM. Islands stay f32, norms/biases bf16. Peak device memory is ~1/4 of the bf16
// arm because the ~16 GB of packed FP4 never expands to ~66 GB of bf16.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3Nvfp4ToDeviceFp4(vt::Queue& queue,
                                                         const SafetensorsFile& file,
                                                         MiniMaxH3DitParams* out_params) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const bool trace = std::getenv("VT_H3_PROGRESS") != nullptr;

  auto is_sidecar = [](const std::string& n) {
    return (n.size() > 12 && n.compare(n.size() - 12, 12, "weight_scale") == 0) ||
           (n.size() > 14 && n.compare(n.size() - 14, 14, "weight_scale_2") == 0);
  };
  auto is_fp32_island = [](const std::string& n) { return MiniMaxH3IsFp32IslandTensor(n); };

  // Pass 1: logical shapes (U8 packed [out, in/2] is logically [out, in]).
  std::vector<MiniMaxH3TensorSpec> manifest;
  for (const std::string& name : file.Names()) {
    if (is_sidecar(name)) continue;
    const StTensor& t = file.Get(name);
    MiniMaxH3TensorSpec spec;
    spec.name = name;
    spec.shape = t.shape;
    if (t.dtype == "U8") {
      VT_CHECK(t.shape.size() == 2, "minimax_h3 nvfp4-fp4: a packed weight must be rank 2");
      spec.shape = {t.shape[0], t.shape[1] * 2};
    }
    manifest.push_back(std::move(spec));
  }
  const MiniMaxH3DitParams params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  if (out_params != nullptr) *out_params = params;

  MiniMaxH3DitDeviceWeights staged;
  std::map<std::string, Tensor> views;
  std::map<std::string, Nvfp4Weight> fp4;
  size_t done = 0;
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const StTensor& t = file.Get(spec.name);

    if (spec.name == "rope.inv_freq") {
      staged.rope_inv_freq_host = MiniMaxH3ReadSafetensorF32(t);
      staged.weights.rope_inv_freq = vt::Tensor::Contiguous(
          staged.rope_inv_freq_host.data(), DType::kF32, vt::Device{},
          {static_cast<int64_t>(staged.rope_inv_freq_host.size())});
      continue;
    }

    // A packed projection -> keep the FP4 triple RESIDENT (no dequant, no upload;
    // the dispatcher uploads + repacks lazily on first forward).
    if (t.dtype == "U8") {
      const int64_t out_dim = spec.shape[0], in_dim = spec.shape[1];
      const StTensor& scale = file.Get(spec.name + "_scale");
      const StTensor& global = file.Get(spec.name + "_scale_2");
      VT_CHECK(scale.dtype == "F8_E4M3", "minimax_h3 nvfp4-fp4: weight_scale must be F8_E4M3");
      VT_CHECK(global.dtype == "F32", "minimax_h3 nvfp4-fp4: weight_scale_2 must be F32");
      VT_CHECK(scale.shape.size() == 2 && scale.shape[0] == out_dim &&
                   scale.shape[1] * 16 == in_dim,
               "minimax_h3 nvfp4-fp4: weight_scale must be [out, in/16] (group size 16)");
      VT_CHECK(global.nbytes >= sizeof(float), "minimax_h3 nvfp4-fp4: weight_scale_2 too small");
      Nvfp4Weight w;
      w.n = out_dim;
      w.k = in_dim;
      std::memcpy(&w.scale2, global.data, sizeof(float));
      w.packed.dtype = DType::kI8;
      w.packed.rank = 2;
      w.packed.shape[0] = out_dim;
      w.packed.shape[1] = in_dim / 2;
      w.packed.bytes.resize(static_cast<size_t>(out_dim) * (in_dim / 2));
      VT_CHECK(t.nbytes == w.packed.bytes.size(), "minimax_h3 nvfp4-fp4: packed byte-size mismatch");
      // Correct the community checkpoint's high-nibble-first fp4 packing so the
      // Marlin W4A16 kernel (low-first, like every other NVFP4 arm) reads it right.
      // See MiniMaxH3Nvfp4HighNibbleFirst.
      if (MiniMaxH3Nvfp4HighNibbleFirst()) {
        MiniMaxH3Nvfp4SwapNibbles(t.data, t.nbytes, w.packed.bytes.data());
      } else {
        std::memcpy(w.packed.bytes.data(), t.data, t.nbytes);
      }
      w.scale.dtype = DType::kI8;
      w.scale.rank = 2;
      w.scale.shape[0] = out_dim;
      w.scale.shape[1] = in_dim / 16;
      w.scale.bytes.resize(static_cast<size_t>(out_dim) * (in_dim / 16));
      VT_CHECK(scale.nbytes == w.scale.bytes.size(), "minimax_h3 nvfp4-fp4: scale byte-size mismatch");
      std::memcpy(w.scale.bytes.data(), scale.data, scale.nbytes);
      fp4[spec.name] = std::move(w);
    } else {
      // Island (f32) or norm/bias (bf16) -> upload a plain tensor, exactly like the
      // bf16 streamer's non-U8 branch.
      const bool island = is_fp32_island(spec.name);
      const DType want = island ? DType::kF32 : DType::kBF16;
      std::vector<float> f32 = MiniMaxH3ReadSafetensorF32(t);
      const void* src = nullptr;
      size_t bytes = 0;
      std::vector<uint16_t> bf16;
      if (island) {
        src = f32.data();
        bytes = f32.size() * sizeof(float);
      } else {
        bf16.resize(f32.size());
        for (size_t i = 0; i < f32.size(); ++i) {
          uint32_t bits;
          std::memcpy(&bits, &f32[i], sizeof(bits));
          const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
          bf16[i] = static_cast<uint16_t>(rounded >> 16);
        }
        src = bf16.data();
        bytes = bf16.size() * sizeof(uint16_t);
      }
      void* pdev = backend.Alloc(bytes);
      std::shared_ptr<void> owner(pdev, [&backend](void* q) { backend.Free(q); });
      backend.Copy(queue, pdev, src, bytes);
      backend.Synchronize(queue);  // host buffers die at end of iteration
      views[spec.name] = dense_attn::MakeTensor(pdev, want, queue.device, spec.shape);
      staged.storage.push_back(std::move(owner));
    }

    if (trace && (++done % 50 == 0 || done == manifest.size())) {
      std::fprintf(stderr, "[h3] nvfp4-fp4-streamed %zu/%zu tensors (last: %s)\n", done,
                   manifest.size(), spec.name.c_str());
      std::fflush(stderr);
    }
  }

  BindStreamedDitViewsFp4(views, fp4, params, &staged.weights);
  return staged;
}

// ★ The ORIGINAL bf16 release (13 shards, 66.3 GB), streamed STRAIGHT ONTO THE
// DEVICE — the multi-shard twin of StreamMiniMaxH3Nvfp4ToDeviceBf16, and the
// first loader that can open the full-precision DiT at all.
//
// WHY IT MUST STREAM. The box has 122 GiB of UNIFIED memory: host and device
// share ONE pool, so "load to host, then stage to device" holds the model TWICE
// against the same budget. The non-streaming NVFP4 loader was already OOM-KILLED
// at anon-rss 125 GB doing exactly that, and this checkpoint is twice its size.
// So: convert ONE tensor, upload it, let the host buffer die before the next.
//
// PEAK HOST MEMORY, precisely. Two of the three cases cost NOTHING:
//   * BF16 on disk -> bf16 device slot: uploaded DIRECTLY out of the read-only
//     mmap. No host buffer exists at any point. This is the case for essentially
//     the entire 66.3 GB (every norm, every projection).
//   * F32 on disk -> f32 island: same, a direct mmap upload.
//   * anything else (a BF16 island widened to f32, an F32 tensor rounded to
//     bf16, an F16 shard): ONE tensor's conversion buffer, freed at the end of
//     its iteration. `host_peak_bytes` reports the largest such buffer, so the
//     gate asserts on the bound rather than trusting this comment.
// Each source range is handed to MaybeReleaseSourcePages the moment its copy
// returns (the same windowed release every other loader uses, default ON), so
// the page cache does not accumulate against the pool the weights live in.
MiniMaxH3DitDeviceWeights StreamMiniMaxH3ShardedToDeviceBf16(
    vt::Queue& queue, const MiniMaxH3ShardedCheckpoint& ckpt,
    MiniMaxH3DitParams* out_params) {
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const bool trace = std::getenv("VT_H3_PROGRESS") != nullptr;

  // Manifest first: names + shapes only, no payload, so the geometry is known
  // (and a wrong name map is caught) before a single byte is allocated.
  const std::vector<MiniMaxH3TensorSpec> manifest = EnumerateMiniMaxH3ShardedTensors(ckpt);
  const MiniMaxH3DitParams params = ParseMiniMaxH3DitParamsFromGgufManifest(manifest);
  if (out_params != nullptr) *out_params = params;

  MiniMaxH3ShardStreamStats& stats = MutableMiniMaxH3ShardStreamStats();
  stats = MiniMaxH3ShardStreamStats{};
  stats.shards_opened = static_cast<uint64_t>(ckpt.ShardCount());

  MiniMaxH3DitDeviceWeights staged;
  std::map<std::string, Tensor> views;
  size_t done = 0;
  for (const MiniMaxH3TensorSpec& spec : manifest) {
    const StTensor& t = ckpt.Get(spec.name);

    // rope.inv_freq is consumed on the HOST (BuildRopeCosSin runs before any
    // kernel); binding a device pointer here segfaults on the first forward.
    if (spec.name == "rope.inv_freq") {
      staged.rope_inv_freq_host = MiniMaxH3ReadSafetensorF32(t);
      staged.weights.rope_inv_freq = vt::Tensor::Contiguous(
          staged.rope_inv_freq_host.data(), DType::kF32, vt::Device{},
          {static_cast<int64_t>(staged.rope_inv_freq_host.size())});
      MaybeReleaseSourcePages(t.data, t.nbytes);
      ++stats.host_resident;
      continue;
    }

    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    VT_CHECK(numel > 0, "minimax_h3 sharded stream: tensor '" + spec.name +
                            "' has an empty logical shape");

    const bool island = MiniMaxH3IsFp32IslandTensor(spec.name);
    const DType want = island ? DType::kF32 : DType::kBF16;
    const bool already = (island && t.dtype == "F32") || (!island && t.dtype == "BF16");

    std::vector<float> f32;
    std::vector<uint16_t> bf16;
    const void* src = nullptr;
    size_t bytes = 0;
    if (already) {
      // The whole point: the on-disk bytes ARE the device bytes, so there is no
      // host copy to peak on.
      VT_CHECK(t.nbytes == static_cast<size_t>(numel) * (island ? 4u : 2u),
               "minimax_h3 sharded stream: tensor '" + spec.name +
                   "' byte span does not match its shape");
      src = t.data;
      bytes = t.nbytes;
      ++stats.direct_uploads;
    } else {
      // Whatever the shard stored it as (BF16 island, F32 body, F16 either way),
      // land on the dtype the forward needs — one tensor at a time.
      f32 = MiniMaxH3ReadSafetensorF32(t);
      VT_CHECK(static_cast<int64_t>(f32.size()) == numel,
               "minimax_h3 sharded stream: tensor '" + spec.name +
                   "' read produced the wrong element count");
      if (island) {
        src = f32.data();
        bytes = f32.size() * sizeof(float);
      } else {
        bf16.resize(f32.size());
        for (size_t i = 0; i < f32.size(); ++i) {
          uint32_t bits;
          std::memcpy(&bits, &f32[i], sizeof(bits));
          // round-to-nearest-even, the same rounding vt uses on a bf16 store
          const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
          bf16[i] = static_cast<uint16_t>(rounded >> 16);
        }
        // The f32 staging buffer dies HERE, not at the end of the iteration: it
        // is the larger of the two, and holding both is the only way this branch
        // could double its own peak.
        const size_t f32_bytes = f32.size() * sizeof(float);
        f32.clear();
        f32.shrink_to_fit();
        stats.host_peak_bytes = std::max<uint64_t>(
            stats.host_peak_bytes, static_cast<uint64_t>(f32_bytes + bf16.size() * 2));
        src = bf16.data();
        bytes = bf16.size() * sizeof(uint16_t);
      }
      stats.host_peak_bytes = std::max<uint64_t>(stats.host_peak_bytes, bytes);
      ++stats.converted_uploads;
    }

    void* pdev = backend.Alloc(bytes);
    std::shared_ptr<void> owner(pdev, [&backend](void* q) { backend.Free(q); });
    backend.Copy(queue, pdev, src, bytes);
    backend.Synchronize(queue);  // the host buffer dies at the end of this iteration
    views[spec.name] = dense_attn::MakeTensor(pdev, want, queue.device, spec.shape);
    staged.storage.push_back(std::move(owner));
    // Copied-then-dead: drop the shard pages this tensor occupied. On a unified
    // box the page cache competes with the model for the same pool.
    MaybeReleaseSourcePages(t.data, t.nbytes);

    ++stats.tensors_streamed;
    stats.bytes_uploaded += bytes;
    if (trace && (++done % 50 == 0 || done == manifest.size())) {
      std::fprintf(stderr, "[h3] shard-streamed %zu/%zu tensors (last: %s)\n", done,
                   manifest.size(), spec.name.c_str());
      std::fflush(stderr);
    }
  }

  BindStreamedDitViews(views, params, &staged.weights);
  return staged;
}

}  // namespace vllm
