// DeepSeek-V4-Flash forward — W7 ASSEMBLY. The `VT_CHECK(false, "W3-W8 pending")`
// stub is replaced by a REAL `DeepseekV4Model::Forward` that COMPOSES the four
// landed host-reference primitive stacks (W3 DSA Lightning-Indexer + 512-wide MLA
// output seams, W4 compressor + fp8_ds_mla KV state, W5 Manifold Hyper-Connections
// + Sinkhorn, W6 sqrtsoftplus/hash MoE + clamped SwiGLU) into an end-to-end logits
// producer on the portable CPU path at a SMALL synthetic config.
//
// ─── HONEST SCOPE (mirrors W3-W6) ───────────────────────────────────────────
// The fixed-config 167B V4 does NOT fit ONE GB10 (156.7 GiB, see deepseek_v4.h)
// and its weights are NOT materialized (W2b residual), so W7 is DERIVED +
// BUILD-VERIFIED: it assembles the interleave + is STRUCTURALLY gated at a tiny
// synthetic shape (test_deepseek_v4_forward.cpp) — NOT a real-checkpoint token
// gate (that is W8, multi-Spark). This does NOT claim V4 "runs" a real model; it
// claims the forward ASSEMBLES + is structurally gated at tiny shape. The device
// kernels (MHC Sinkhorn, DSA indexer/compressor, sqrtsoftplus router, clamped
// SwiGLU — the expert GEMM REUSES the existing NVFP4/FP8 grouped-GEMM) + the real
// e2e are named residuals (W7-device + W8).
//
// ─── INTERLEAVE (grounded, file:line on both sides, @ pin 555967922) ─────────
// vllm/models/deepseek_v4/nvidia/model.py:1080-1148 (DeepseekV4Model.forward) +
// :866-957 (DeepseekV4DecoderLayer.forward):
//   embed -> for each layer:
//     [first layer] MHC-pre BROADCAST expand [T,H] -> [T,hc,H] (mhc_pre_broadcast,
//        :880-897) ; [else] MHC fused-post-pre = MhcPost(prev-ffn-out) + MhcPre(attn)
//     512-wide MLA attn: q(wq_a->q_norm->wq_b) + kv(wkv->kv_norm), dual-theta RoPE,
//        DSA indexer->topk->compressor->fp8_ds_mla KV, sink softmax, grouped o-LoRA
//     MHC fused-post-pre = MhcPost(attn-out) + MhcPre(ffn)   (:934-957)
//     MoE: sqrtsoftplus/hash router + shared+routed clamped-SwiGLU experts
//   final MhcPost(last-ffn-out) -> hc_head collapse (:1136) -> norm -> lm_head.
//
// Where the tiny-config forward diverges from the 167B structure (documented, not
// silent): (i) the compressor pools a fixed W=2 window rather than the real
// (1+overlap)*compress_ratio window (the state-cache gather addressing is a W7
// device concern, deepseek_v4_compressor.h note); (ii) the MLA value is the full
// decoded latent (the W_UK/W_UV absorption geometry is a shared-MLA-extraction W7
// follow-on); (iii) a single rope_theta is used for all layers (the compressed
// layers' dual compress_rope_theta is a device-RoPE seam); (iv) the fp8_ds_mla
// quant_block == nope_head_dim (one block) at tiny width. Each reuses the SAME
// landed primitive math the device kernels will call.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_probe.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>  // madvise(MADV_DONTNEED) — Phase-2 routed-expert mmap reclaim
#include <unistd.h>    // sysconf(_SC_PAGESIZE)
#endif

#include "vllm/model_executor/models/deepseek_v4_compressor.h"
#include "vllm/model_executor/models/deepseek_v4_device.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/deepseek_v4_moe.h"
#include "vt/dtype.h"
#include "vt/quant.h"  // BlockToFloat (Phase-2 expert discriminator, #188)
#include "vt/ops.h"      // vt::MatmulBT (auto-dispatches kMatmulBTQuant on block weights)
#include "vt/tensor.h"   // vt::Tensor::Contiguous
#include "vt/backend.h"  // vt::GetBackend / Backend::Synchronize (device GEMM drain)
#include "vt/breakable_graph.h"  // ENG-CUDAGRAPH-BREAK W5: the shared capture seam

namespace vllm {
namespace {

void DumpAct(const char* name, const std::vector<float>& v);  // fwd (coherence-debug #188)

using deepseek_v4::ClampedSwiGLU;
using deepseek_v4::CompressorPoolNorm;
using deepseek_v4::CompressorSaveScoreApe;
using deepseek_v4::DsaIndexerLogits;
using deepseek_v4::DsaIndexerWeightFold;
using deepseek_v4::DsaTopkSelect;
using deepseek_v4::Fp8DsMlaDecodeToken;
using deepseek_v4::Fp8DsMlaEncodeToken;
using deepseek_v4::Fp8DsMlaLayout;
using deepseek_v4::HcHeadCollapse;
using deepseek_v4::MakeFp8DsMlaLayout;
using deepseek_v4::MhcPost;
using deepseek_v4::MhcPre;
using deepseek_v4::MhcPreResult;
using deepseek_v4::MoeRouteResult;
using deepseek_v4::SoftmaxWithSink;
using deepseek_v4::SqrtSoftplusRouteTopk;

// ── backend policy: HOST refs (the oracle) OR the W7-device CUDA kernels ───────
// The composition below is written ONCE and run either on the portable host
// references (DeepseekV4ForwardHost, the oracle the device kernels are gated
// against) or on the CUDA kernels through the OpProvider seam
// (DeepseekV4Model::ForwardDevice). Only the four NEW V4 op families
// (MHC / DSA indexer+seams / compressor+fp8_ds_mla / sqrtsoftplus-hash MoE +
// clamped SwiGLU) branch; the small linear projections stay host in both modes
// (in the real device path they REUSE the existing GEMM/MLA/MoE-grouped kernels —
// a documented W7 seam, not re-ported here). device==host at the tiny structural
// shape is the ForwardDevice composition gate (test_cuda_deepseek_v4.cpp).
// ── W2C: keep-quant weight SOURCE ─────────────────────────────────────────────
// When `gguf != nullptr` the big MLA/MoE/lm_head GEMMs consume the keep-quant
// `weights.gguf` OwnedTensor blocks DIRECTLY via vt::MatmulBT (which dispatches to
// the landed CPU kMatmulBTQuant CIQ GEMM for a block-quant weight) — NO per-layer
// f32 tower. The small non-GEMM tensors (norms, sinks, MHC/DSA mixing, ape, the
// hash table, embed) still come from the SMALL `hw` host tower, dequant-f32 exactly
// as our other GGUF models keep them (qwen3_5_gguf_weights.cpp). When `gguf ==
// nullptr` every GEMM reads the f32 `hw` tower (the safetensors/NVFP4 + the tiny
// synthetic structural gate), byte-for-byte the pre-W2C behavior. `device` selects
// the CUDA V4-primitive kernels for the four NEW op families (orthogonal to the
// weight source; the GGUF keep-quant path runs device=false, CPU).
struct V4Backend {
  bool device = false;
  vt::Queue* q = nullptr;
  const DeepseekV4GgufWeights* gguf = nullptr;
  // Incremental-decode KV cache (Stage 1). Null = stateless full-recompute (the
  // default / --gpu path). When set, AttentionBlock appends each token's per-layer
  // `deck` latent to cache.deck[layer] and attends over the full cached KV; the
  // query's global position is kv_base + local_t (kv_base = cache.len at the call).
  DeepseekV4KvCache* kv = nullptr;
  int64_t kv_base = 0;
  // Re-scoped Stage 2: collapse the routed-expert per-expert keep-quant matvecs
  // into ONE grouped kMatmulBTQuantGrouped launch per {gate,up,down} (fewer host
  // launches + higher GB10 occupancy). Default ON for the GGUF keep-quant path;
  // `VT_V4_GROUPED_MOE=0` rolls back to the per-expert GemmRowSlice loop. The CPU
  // grouped provider loops the same kMatmulBTQuant kernel ⇒ byte-identical.
  bool grouped_moe = true;
};

// Drain the queue's stream after a keep-quant GEMM. This host-orchestrated GGUF
// forward reads each GEMM's f32 output vector on the host IMMEDIATELY (the next
// Disp*/glue op runs host-side), but the CUDA kMatmulBTQuant kernel is
// stream-async — so on a device queue the stream must be drained before the host
// touches `out`. On GB10 the GEMM operands are unified-memory views (mmap'd
// keep-quant weight blocks + std::vector activations that the coherent GPU reads
// in place — no H2D/D2H, which keeps the ~91 GiB weights single-copy), so this is
// pure ORDERING, not a transfer. On the CPU queue (the default, coherent path)
// MatmulBT is synchronous and this is a no-op. Mirrors qwen3_5.cpp:746
// (Backend::Synchronize after a device matmul the host then consumes).
inline void SyncDeviceGemm(const V4Backend& be) {
  if (be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU)
    vt::GetBackend(be.q->device).Synchronize(*be.q);
}

// Stage-2 decode-step profiling (env `VT_V4_PROF`): split GEMM-dispatch vs
// stream-drain so the decode bottleneck is MEASURED, not guessed. Host glue =
// step_time - gemm - sync. Accumulators are process-global; the driver resets
// them per step. Inert unless VT_V4_PROF is set.
namespace prof {
double g_gemm_s = 0.0;
double g_sync_s = 0.0;
inline bool On() {
  static const bool e = std::getenv("VT_V4_PROF") != nullptr;
  return e;
}
}  // namespace prof

// Re-scoped Stage 2: grouped routed-expert MoE GEMM default ON; `VT_V4_GROUPED_MOE=0`
// rolls back to the per-expert GemmRowSlice batch. Read once.
inline bool GroupedMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_GROUPED_MOE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Brick A (device-resident decode campaign): run the MLA attention QK/softmax-sink/AV
// on the device kernel instead of the host Dot loop. Default OFF — the host path
// stays default until the decode graph (Brick D) proves out. `VT_V4_DEVICE_ATTN=1`.
inline bool DeviceAttnEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_DEVICE_ATTN");
    return e != nullptr && std::string(e) == "1";
  }();
  return on;
}

// Brick B (device-resident decode campaign): run the MoE/MHC glue on device kernels
// (in place on the unified activations) instead of the host reference. Default OFF
// until the decode graph (Brick D) proves out. `VT_V4_DEVICE_GLUE=1`.
inline bool DeviceGlueEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_DEVICE_GLUE");
    return e != nullptr && std::string(e) == "1";
  }();
  return on;
}

// True when the GGUF keep-quant glue should run on the in-place device kernels:
// VT_V4_DEVICE_GLUE set + a CUDA queue + the V4 device kernels linked/live.
inline bool GlueDev(const V4Backend& be) {
  return be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU && DeviceGlueEnabled() &&
         deepseek_v4::V4DeviceKernelsAvailable();
}

// One MatmulBT + (optional device) drain, timed when profiling is on. When
// `defer_sync` is true the stream is NOT drained here — the caller issues a batch
// of independent GEMMs and drains ONCE via DrainDevice before the host reads them
// (Stage 2: amortize the ~66 us cudaStreamSynchronize over many GEMMs). Same
// stream ⇒ the GEMMs are serialized regardless, so the result is byte-identical.
inline void TimedMatmul(const V4Backend& be, bool on_dev, bool defer_sync, vt::Queue& gq,
                        vt::Tensor& o, const vt::Tensor& a, const vt::Tensor& w) {
  const bool do_sync = on_dev && !defer_sync;
  if (!prof::On()) {
    vt::MatmulBT(gq, o, a, w);
    if (do_sync) SyncDeviceGemm(be);
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  vt::MatmulBT(gq, o, a, w);
  const auto t1 = std::chrono::steady_clock::now();
  prof::g_gemm_s += std::chrono::duration<double>(t1 - t0).count();
  if (do_sync) {
    SyncDeviceGemm(be);
    const auto t2 = std::chrono::steady_clock::now();
    prof::g_sync_s += std::chrono::duration<double>(t2 - t1).count();
  }
}

// Explicit one-shot stream drain for a batch of deferred GEMMs (Stage 2). No-op on
// the CPU queue. Timed into the sync bucket when profiling.
inline void DrainDevice(const V4Backend& be) {
  if (be.q == nullptr || be.q->device.type == vt::DeviceType::kCPU) return;
  if (!prof::On()) {
    vt::GetBackend(be.q->device).Synchronize(*be.q);
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  vt::GetBackend(be.q->device).Synchronize(*be.q);
  prof::g_sync_s +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

deepseek_v4::MhcPreResult DispMhcPre(const V4Backend& be, const std::vector<float>& residual,
                                     const std::vector<float>& fn,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& base, int64_t hc, int64_t hidden,
                                     float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                                     float hc_post_mult, int64_t iters,
                                     const std::vector<float>& norm_weight, float norm_eps) {
  if (be.device)
    return deepseek_v4::MhcDevice()->pre(*be.q, residual, fn, scale, base, hc, hidden, rms_eps,
                                         hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, iters,
                                         norm_weight, norm_eps);
  // Brick B: in-place device MHC-pre via the PARALLEL MhcPreParallelKernel (one
  // block over the H width; the #183 <<<1,1>>> stub regressed decode 10×, so this
  // is the real parallel kernel). Near-tie vs host (width-reduction reorder).
  if (GlueDev(be)) {
    deepseek_v4::MhcPreResult out;
    out.pre_mix.resize(static_cast<size_t>(hc));
    out.post_mix.resize(static_cast<size_t>(hc));
    out.comb_mix.resize(static_cast<size_t>(hc * hc));
    out.layer_input.resize(static_cast<size_t>(hidden));
    std::vector<float> mix(static_cast<size_t>((2 + hc) * hc + 1));  // mixes[hc3] + folded sqrsum slot
    const bool has_norm = !norm_weight.empty();
    deepseek_v4::MhcDevice()->pre_ip(
        *be.q, out.pre_mix.data(), out.post_mix.data(), out.comb_mix.data(),
        out.layer_input.data(), mix.data(), residual.data(), fn.data(), scale.data(), base.data(),
        hc, hidden, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, iters,
        has_norm ? norm_weight.data() : nullptr, has_norm, norm_eps);
    SyncDeviceGemm(be);
    return out;
  }
  return MhcPre(residual, fn, scale, base, hc, hidden, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult, iters, norm_weight, norm_eps);
}
std::vector<float> DispMhcPost(const V4Backend& be, const std::vector<float>& x,
                               const std::vector<float>& residual,
                               const std::vector<float>& post_mix,
                               const std::vector<float>& comb, int64_t hc, int64_t hidden) {
  if (be.device) return deepseek_v4::MhcDevice()->post(*be.q, x, residual, post_mix, comb, hc, hidden);
  if (GlueDev(be)) {  // Brick B: in-place device MHC-post
    std::vector<float> out(static_cast<size_t>(hc * hidden));
    deepseek_v4::MhcDevice()->post_ip(*be.q, out.data(), x.data(), residual.data(),
                                      post_mix.data(), comb.data(), hc, hidden);
    SyncDeviceGemm(be);
    return out;
  }
  return MhcPost(x, residual, post_mix, comb, hc, hidden);
}
std::vector<float> DispHcHead(const V4Backend& be, const std::vector<float>& x,
                              const std::vector<float>& fn, float scale,
                              const std::vector<float>& base, int64_t hc, int64_t hidden,
                              float rms_eps, float hc_eps) {
  if (be.device) return deepseek_v4::MhcDevice()->head(*be.q, x, fn, scale, base, hc, hidden, rms_eps, hc_eps);
  if (GlueDev(be)) {  // Brick B: in-place device hc_head collapse
    std::vector<float> out(static_cast<size_t>(hidden));
    deepseek_v4::MhcDevice()->head_ip(*be.q, out.data(), x.data(), fn.data(), scale, base.data(),
                                      hc, hidden, rms_eps, hc_eps);
    SyncDeviceGemm(be);
    return out;
  }
  return HcHeadCollapse(x, fn, scale, base, hc, hidden, rms_eps, hc_eps);
}
std::vector<float> DispSaveScoreApe(const V4Backend& be, const std::vector<float>& score,
                                    const std::vector<float>& ape,
                                    const std::vector<int64_t>& positions, int64_t T,
                                    int64_t width, int64_t cr) {
  if (be.device) return deepseek_v4::CompressorDevice()->save_score_ape(*be.q, score, ape, positions, T, width, cr);
  return CompressorSaveScoreApe(score, ape, positions, T, width, cr);
}
std::vector<float> DispPoolNorm(const V4Backend& be, const std::vector<float>& kv,
                                const std::vector<float>& score,
                                const std::vector<uint8_t>& valid,
                                const std::vector<float>& rms_w, float eps, int64_t window,
                                int64_t hd) {
  if (be.device) return deepseek_v4::CompressorDevice()->pool_norm(*be.q, kv, score, valid, rms_w, eps, window, hd);
  return CompressorPoolNorm(kv, score, valid, rms_w, eps, window, hd);
}
deepseek_v4::Fp8DsMlaToken DispEncode(const V4Backend& be, const std::vector<float>& head,
                                      const Fp8DsMlaLayout& ly) {
  if (be.device) return deepseek_v4::CompressorDevice()->encode(*be.q, head, ly);
  return Fp8DsMlaEncodeToken(head, ly);
}
std::vector<float> DispDecode(const V4Backend& be, const deepseek_v4::Fp8DsMlaToken& tok,
                              const Fp8DsMlaLayout& ly) {
  if (be.device) return deepseek_v4::CompressorDevice()->decode(*be.q, tok, ly);
  return Fp8DsMlaDecodeToken(tok, ly);
}
std::vector<float> DispWeightFold(const V4Backend& be, const std::vector<float>& wp, int64_t T,
                                  int64_t inh, int64_t ihd) {
  if (be.device) return deepseek_v4::DsaDevice()->weight_fold(*be.q, wp, T, inh, ihd);
  return DsaIndexerWeightFold(wp, T, inh, ihd);
}
std::vector<float> DispLogits(const V4Backend& be, const std::vector<float>& q,
                              const std::vector<float>& k, const std::vector<float>& folded,
                              const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                              int64_t T, int64_t nk, int64_t inh, int64_t ihd) {
  if (be.device) return deepseek_v4::DsaDevice()->logits(*be.q, q, k, folded, ws, we, T, nk, inh, ihd);
  return DsaIndexerLogits(q, k, folded, ws, we, T, nk, inh, ihd);
}
std::vector<int64_t> DispTopk(const V4Backend& be, const std::vector<float>& logits,
                              const std::vector<int64_t>& ws, const std::vector<int64_t>& we,
                              int64_t T, int64_t nk, int64_t topk) {
  if (be.device) return deepseek_v4::DsaDevice()->topk(*be.q, logits, ws, we, T, nk, topk);
  return DsaTopkSelect(logits, ws, we, T, nk, topk);
}
std::vector<float> DispSoftmaxSink(const V4Backend& be, const std::vector<float>& scores,
                                   float sink) {
  if (be.device) return deepseek_v4::DsaDevice()->softmax_sink(*be.q, scores, sink);
  return SoftmaxWithSink(scores, sink);
}
std::vector<float> DispGroupedOLora(const V4Backend& be, const std::vector<float>& o,
                                    const std::vector<float>& wo_a,
                                    const std::vector<float>& wo_b, int64_t T, int64_t nh,
                                    int64_t hd, int64_t ng, int64_t olr, int64_t H) {
  if (be.device) return deepseek_v4::DsaDevice()->grouped_olora(*be.q, o, wo_a, wo_b, T, nh, hd, ng, olr, H);
  return deepseek_v4::GroupedOutputLora(o, wo_a, wo_b, T, nh, hd, ng, olr, H);
}
deepseek_v4::MoeRouteResult DispRoute(const V4Backend& be, const std::vector<float>& gating,
                                      int64_t T, int64_t E, int64_t topk,
                                      const std::vector<float>& bias, bool renorm, float scale,
                                      const std::vector<int64_t>& in_tokens,
                                      const std::vector<int32_t>& hashtab, int64_t vocab) {
  if (be.device)
    return deepseek_v4::MoeDevice()->route(*be.q, gating, T, E, topk, bias, renorm, scale,
                                           in_tokens, hashtab, vocab);
  if (GlueDev(be)) {  // Brick B: in-place device router (softmax + top-k → expert_ids)
    const bool has_bias = !bias.empty();
    const bool is_hash = !hashtab.empty() && !in_tokens.empty();
    deepseek_v4::MoeRouteResult out;
    out.topk_ids.assign(static_cast<size_t>(T * topk), 0);
    out.topk_weights.assign(static_cast<size_t>(T * topk), 0.0f);
    deepseek_v4::MoeDevice()->route_ip(
        *be.q, out.topk_ids.data(), out.topk_weights.data(), gating.data(), T, E, topk,
        has_bias ? bias.data() : nullptr, has_bias, is_hash ? in_tokens.data() : nullptr, is_hash,
        is_hash ? hashtab.data() : nullptr, vocab, renorm, scale);
    SyncDeviceGemm(be);
    return out;
  }
  return SqrtSoftplusRouteTopk(gating, T, E, topk, bias, renorm, scale, in_tokens, hashtab, vocab);
}
std::vector<float> DispClampedSwiGLU(const V4Backend& be, const std::vector<float>& gate_up,
                                     int64_t d, float limit, float alpha, float beta) {
  if (be.device) return deepseek_v4::MoeDevice()->clamped_swiglu(*be.q, gate_up, d, limit, alpha, beta);
  // Brick B: the GGUF keep-quant path (be.device=false) runs the clamped-SwiGLU on
  // the device kernel IN PLACE over the unified activation when VT_V4_DEVICE_GLUE +
  // a CUDA queue + the V4 device kernels are live; bit-identical (elementwise).
  if (be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU && DeviceGlueEnabled() &&
      deepseek_v4::V4DeviceKernelsAvailable()) {
    std::vector<float> out(static_cast<size_t>(d));
    deepseek_v4::MoeDevice()->clamped_swiglu_ip(*be.q, out.data(), gate_up.data(), d, limit,
                                                alpha, beta);
    SyncDeviceGemm(be);  // Brick B drains; Brick D will capture instead
    return out;
  }
  return ClampedSwiGLU(gate_up, d, limit, alpha, beta);
}

// ── small portable linear-algebra helpers ────────────────────────────────────
float Dot(const float* a, const float* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

// y[o] = Σ_i W[o*in + i] * x[i]  (W is [out, in] row-major).
std::vector<float> MatVec(const std::vector<float>& w, const float* x, int64_t out,
                          int64_t in) {
  VT_CHECK(static_cast<int64_t>(w.size()) == out * in, "MatVec weight size mismatch");
  std::vector<float> y(static_cast<size_t>(out));
  for (int64_t o = 0; o < out; ++o) y[static_cast<size_t>(o)] = Dot(&w[o * in], x, in);
  return y;
}

// ── W2C keep-quant GEMM: Y[T,N] = X[T,K] @ W[N,K]^T ───────────────────────────
// `wq` (the keep-quant / bf16 OwnedTensor block, [N,K] nk=true as on GGUF disk) is
// consumed IN PLACE via vt::MatmulBT — a block-quant dtype routes to the CPU
// kMatmulBTQuant CIQ GEMM (cpu_quant_gemm.cpp, quantizes the activation once then
// integer vec_dot per output, weights never expanded); a bf16 dtype (the expand
// oracle) routes to the elementwise MatmulBT. When `wq` is null/absent (host
// source) it falls back to the per-row f32 MatVec — BIT-IDENTICAL to the pre-W2C
// host composition. Grounded in qwen3_5.cpp:786-838 (host MatmulBT off an
// OwnedTensor.View()) + vt/ops.cpp:134-171 (block-quant dispatch).
std::vector<float> Gemm(const V4Backend& be, const OwnedTensor* wq,
                        const std::vector<float>& wf32, const std::vector<float>& x,
                        int64_t T, int64_t N, int64_t K, bool defer_sync = false) {
  if (be.gguf != nullptr && wq != nullptr && !wq->Empty()) {
    VT_CHECK(be.q != nullptr, "deepseek-v4 keep-quant GEMM needs a queue");
    VT_CHECK(wq->rank == 2 && wq->shape[0] == N && wq->shape[1] == K,
             "deepseek-v4 keep-quant GEMM: weight shape mismatch: want [N=" +
                 std::to_string(N) + ",K=" + std::to_string(K) + "] got [" +
                 std::to_string(wq->shape[0]) + "," + std::to_string(wq->shape[1]) +
                 "] rank=" + std::to_string(wq->rank));
    std::vector<float> out(static_cast<size_t>(T) * N);
    // Only BLOCK-QUANT weights (the dominant-FLOP experts / MLA linears / lm_head,
    // IQ2_XXS/IQ3_XXS/Q2_K/Q8_0) go to the device kMatmulBTQuant provider. Small
    // ELEMENTWISE weights the loader dequantized to bf16 (e.g. the [256,H] router
    // gate) stay on the CPU: the CUDA elementwise kMatmulBT has no (f32-act,
    // bf16-weight) combo, and these are negligible FLOPs. Both read the SAME
    // unified-memory tensors, so mixing CPU/GPU GEMMs across the host-orchestrated
    // forward is free (SyncDeviceGemm orders the device ones before host reads).
    vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const bool on_dev =
        be.q->device.type != vt::DeviceType::kCPU && vt::IsBlockQuant(wq->dtype);
    vt::Queue& gq = on_dev ? *be.q : cpuq;
    vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()),
                                          vt::DType::kF32, gq.device, {T, K});
    vt::Tensor o =
        vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
    vt::Tensor w = wq->View();
    // The keep-quant blocks are loaded with the CPU device tag; on a device queue
    // they are unified-memory views the GPU reads in place — retag to the chosen
    // queue's device so MatmulBTQuant's device-consistency check (ops.cpp:198)
    // dispatches to the right provider (mirrors GemmRowSlice's wt.device below).
    w.device = gq.device;
    TimedMatmul(be, on_dev, defer_sync, gq, o, a, w);  // MatmulBT + (device) drain
    return out;
  }
  std::vector<float> out(static_cast<size_t>(T) * N);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> y = MatVec(wf32, &x[t * K], N, K);
    for (int64_t n = 0; n < N; ++n) out[t * N + n] = y[static_cast<size_t>(n)];
  }
  return out;
}

// Keep-quant GEMM against a ROW-SLICE [row_off, row_off+N) of a stacked block
// weight `w` ([E*out, K] nk=true) — the per-expert (moe_*_exps) / per-group (wo_a)
// slice. Rows are whole blocks (RowSizeBytes), so the offset is a byte offset and
// no block is ever cut (mirrors the loader's OwnGgufQuantBlocks row_offset slice,
// qwen3_5_gguf_weights.cpp:57-101, and the kStackedExpertWeight contract). Returns
// [T,N] f32.
std::vector<float> GemmRowSlice(const V4Backend& be, const OwnedTensor& w,
                                const std::vector<float>& x, int64_t T, int64_t N,
                                int64_t K, int64_t row_off, bool defer_sync = false) {
  VT_CHECK(be.q != nullptr, "deepseek-v4 keep-quant expert GEMM needs a queue");
  VT_CHECK(!w.repacked,
           "deepseek-v4 keep-quant expert/group slice requires non-repacked blocks "
           "(disable VT_CPU_QUANT_REPACK for the stacked-expert weights)");
  VT_CHECK(w.rank == 2 && row_off >= 0 && row_off + N <= w.shape[0] && w.shape[1] == K,
           "deepseek-v4 keep-quant expert GEMM: slice out of range");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  std::vector<float> out(static_cast<size_t>(T) * N);
  // Stacked expert/group slices are always block-quant → the device provider; the
  // block-quant guard mirrors Gemm so a stray elementwise weight would fall to CPU
  // rather than hit the CUDA kMatmulBT's missing (f32-act, bf16-weight) combo.
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev =
      be.q->device.type != vt::DeviceType::kCPU && vt::IsBlockQuant(w.dtype);
  vt::Queue& gq = on_dev ? *be.q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x.data()), vt::DType::kF32,
                                        gq.device, {T, K});
  vt::Tensor o =
      vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {T, N});
  vt::Tensor wt;
  wt.data = const_cast<uint8_t*>(w.bytes.data()) +
            static_cast<size_t>(row_off) * row_bytes;
  wt.dtype = w.dtype;
  wt.device = gq.device;
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;  // inert for a block-quant weight; correct for the bf16 oracle
  wt.stride[1] = 1;
  TimedMatmul(be, on_dev, defer_sync, gq, o, a, wt);  // MatmulBT + (device) drain
  return out;
}

// Grouped OUTPUT-LoRA on the keep-quant tower (the GGUF mirror of
// deepseek_v4::GroupedOutputLora): z[t, g*olr+d] = Σ_r wo_a[g,d,r]·o[t,g,r]
// (per-group block-diagonal, so one row-slice quant GEMM per group), then
// out[t] = wo_b @ z. wo_a keep-quant [ng*olr, in_per_group], wo_b keep-quant
// [H, ng*olr]. o_proj.py:58-73.
std::vector<float> GroupedOutputLoraGguf(const V4Backend& be, const OwnedTensor& wo_a,
                                         const OwnedTensor& wo_b,
                                         const std::vector<float>& o, int64_t T,
                                         int64_t nh, int64_t hd, int64_t ng,
                                         int64_t olr, int64_t H) {
  VT_CHECK(ng > 0 && nh % ng == 0, "grouped o-LoRA: n_heads % n_groups != 0");
  const int64_t ipg = nh * hd / ng;  // in_per_group
  const int64_t z_dim = ng * olr;
  std::vector<float> z(static_cast<size_t>(T) * z_dim);
  // Stage 2: the `ng` per-group wo_a GEMMs are independent → separate input/output
  // buffers per group, issue them all with a deferred drain, DRAIN ONCE, then
  // assemble z. Was ng+1 stream drains → 2. Byte-identical (same GEMMs, same stream).
  std::vector<std::vector<float>> og(static_cast<size_t>(ng)), zg(static_cast<size_t>(ng));
  for (int64_t g = 0; g < ng; ++g) {
    og[static_cast<size_t>(g)].resize(static_cast<size_t>(T) * ipg);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t r = 0; r < ipg; ++r)
        og[static_cast<size_t>(g)][t * ipg + r] = o[t * nh * hd + g * ipg + r];
    zg[static_cast<size_t>(g)] = GemmRowSlice(be, wo_a, og[static_cast<size_t>(g)], T, olr, ipg,
                                              /*row_off=*/g * olr, /*defer_sync=*/true);  // [T,olr]
  }
  DrainDevice(be);
  for (int64_t g = 0; g < ng; ++g)
    for (int64_t t = 0; t < T; ++t)
      for (int64_t d = 0; d < olr; ++d)
        z[t * z_dim + g * olr + d] = zg[static_cast<size_t>(g)][t * olr + d];
  return Gemm(be, &wo_b, /*wf32=*/{}, z, T, H, z_dim);  // [T,H] (final; drains normally)
}

// Grouped keep-quant expert GEMM (re-scoped Stage 2): out[P,N] where
// out[p,:] = act[p,:] · weight[expert_ids[p]] (the [e*N,+N) block row-slice of the
// stacked expert weight). ONE vt::MatmulBTQuantGrouped launch replaces P per-expert
// GemmRowSlice matvecs. `act` is [P*K] row-major. Weight retagged to the queue
// device (unified view). Drains before returning (eids is a local buffer the async
// kernel reads, so it must not be deferred past this call). Numerically identical
// to the per-expert path (the grouped kernel is the same integer-dot core, and the
// CPU provider literally loops kMatmulBTQuant per group).
std::vector<float> GemmGroupedExpertsKq(const V4Backend& be, const OwnedTensor& weight,
                                        const std::vector<float>& act,
                                        const std::vector<int32_t>& expert_ids,
                                        int64_t P, int64_t N, int64_t K) {
  VT_CHECK(be.q != nullptr, "deepseek-v4 grouped expert GEMM needs a queue");
  VT_CHECK(!weight.repacked,
           "deepseek-v4 grouped expert GEMM requires non-repacked stacked blocks");
  VT_CHECK(vt::IsBlockQuant(weight.dtype),
           "deepseek-v4 grouped expert GEMM requires a block-quant stacked weight");
  VT_CHECK(static_cast<int64_t>(act.size()) == P * K, "grouped expert GEMM: act size mismatch");
  VT_CHECK(static_cast<int64_t>(expert_ids.size()) == P, "grouped expert GEMM: expert_ids size");
  const bool on_dev = be.q->device.type != vt::DeviceType::kCPU;
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Queue& gq = on_dev ? *be.q : cpuq;
  std::vector<float> out(static_cast<size_t>(P) * N);
  std::vector<int32_t> eids = expert_ids;  // stable buffer for the (unified) tensor
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(act.data()), vt::DType::kF32,
                                        gq.device, {P, K});
  vt::Tensor o = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, gq.device, {P, N});
  vt::Tensor eid =
      vt::Tensor::Contiguous(eids.data(), vt::DType::kI32, gq.device, {P});
  vt::Tensor w = weight.View();
  w.device = gq.device;
  vt::MatmulBTQuantGrouped(gq, o, a, w, eid);
  if (on_dev) SyncDeviceGemm(be);  // drain before eids/out leave scope
  return out;
}

// Weighted RMSNorm (the standard DeepSeek/vLLM RMSNorm).
std::vector<float> RmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                           float eps) {
  const int64_t n = static_cast<int64_t>(x.size());
  double ss = 0.0;
  for (int64_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float r = 1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(n)) + eps);
  std::vector<float> y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    y[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] * r * w[static_cast<size_t>(i)];
  return y;
}

// Decoupled NeoX-free pairwise RoPE over an `r`-wide (r even) rope subvector
// (common/rope.py deepseek_yarn, mscale disabled — the tiny forward uses a single
// theta; the compressed-layer dual compress_rope_theta is a device-RoPE seam).
// Per-layer YaRN RoPE over the last `r` dims of a head (GPT-J adjacent pairs). The
// REAL DeepSeek-V4 uses a DUAL rope: dense layers (compress_ratio==0) rotate with
// base=rope_theta, freq_scale=1, ext_factor=0 (== RopeInplace); COMPRESSED layers
// (41 of 43) rotate with base=compress_rope_theta (160000), freq_scale=1/factor
// (1/16) YaRN interpolation + the beta_fast/beta_slow correction-dim ramp. The net
// magnitude scale is 1 (ds4 cancels the yarn mscale explicitly). 1:1 port of ds4
// `ds4.c:rope_tail_ext_inplace` (+ `rope_yarn_corr_dim`/`rope_yarn_ramp`). Getting
// this wrong scrambles the rope half of q·k on every compressed layer → the model
// loses positional/context structure (degenerate repetition).
double YarnCorrDim(int64_t n_dims, int64_t n_ctx_orig, double beta, double base) {
  return static_cast<double>(n_dims) *
         std::log(static_cast<double>(n_ctx_orig) /
                  (beta * 2.0 * std::numbers::pi_v<double>)) /
         (2.0 * std::log(base));
}
void RopeInplaceLayer(float* v, int64_t r, int64_t pos, double base, double freq_scale,
                      double ext_factor, int64_t n_ctx_orig, double beta_fast,
                      double beta_slow, bool inverse = false) {
  const double theta_scale = std::pow(base, -2.0 / static_cast<double>(r));
  const double sin_sign = inverse ? -1.0 : 1.0;  // inverse rope un-rotates (ds4 sin_sign)
  double corr_lo = 0.0, corr_hi = 0.0;
  if (ext_factor != 0.0) {
    corr_lo = std::max(0.0, std::floor(YarnCorrDim(r, n_ctx_orig, beta_fast, base)));
    corr_hi = std::min(static_cast<double>(r - 1),
                       std::ceil(YarnCorrDim(r, n_ctx_orig, beta_slow, base)));
  }
  double theta_extrap = static_cast<double>(pos);
  for (int64_t i = 0; i < r; i += 2) {
    const double theta_interp = freq_scale * theta_extrap;
    double theta = theta_interp;
    if (ext_factor != 0.0) {
      const double y = (static_cast<double>(i / 2) - corr_lo) / std::max(0.001, corr_hi - corr_lo);
      const double ramp = (1.0 - std::min(1.0, std::max(0.0, y))) * ext_factor;
      theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
    }
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(sin_sign * std::sin(theta));
    const float x0 = v[i], x1 = v[i + 1];
    v[i] = x0 * c - x1 * s;
    v[i + 1] = x0 * s + x1 * c;
    theta_extrap *= theta_scale;
  }
}

std::vector<float> Slice(const std::vector<float>& v, int64_t off, int64_t len) {
  return std::vector<float>(v.begin() + off, v.begin() + off + len);
}

// ── 512-wide MLA attention block (W3 + W4 primitives) : [T,H] -> [T,H] ────────
std::vector<float> AttentionBlock(const DeepseekV4LayerHostWeights& L,
                                  const DeepseekV4GgufLayerWeights* Lq,
                                  const DeepseekV4Params& p,
                                  const std::vector<float>& x,
                                  const std::vector<int32_t>& positions, int64_t layer,
                                  V4Miswire miswire, V4ForwardTrace* trace,
                                  const V4Backend& be) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const int64_t H = p.hidden_size;
  const int64_t nh = p.num_attention_heads;
  const int64_t hd = p.head_dim;
  const int64_t rope = p.qk_rope_head_dim;
  const int64_t nope = hd - rope;
  const int64_t qlr = p.q_lora_rank;
  const float eps = p.rms_norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  // DSA sparse path (the compressor's separate compressed-KV cache + the indexer's
  // learned top-k token selection) is implemented only at the COLLAPSED synthetic
  // geometry, where the compressor projects to `head_dim`. At the REAL DeepSeek-V4
  // geometry the compressor projects to `comp_width = 2*head_dim` (ds4
  // `ds4.c:5016-5021`: `coff=2` for `compress_ratio==4`), so the tiny compressor/
  // indexer code does not apply. The real keep-quant run therefore uses DENSE MLA —
  // which is EXACT, not an approximation, whenever `seq_len <= index_topk` (=512):
  // the indexer cannot select more tokens than exist, so top-k over ≤512 tokens IS
  // the full causal set, and no raw row has yet been evicted into the compressed
  // cache. Every short single-Spark step satisfies this. The full real-geometry DSA
  // sparse path (compressor cache + indexer selection, for contexts > 512) is a
  // NAMED residual. The host/device synthetic path (be.gguf==nullptr) keeps
  // exercising the compressor/indexer primitives at their gated tiny shape.
  const bool dsa_dense = (be.gguf != nullptr);
  const bool is_indexer = p.has_indexer(layer) && !dsa_dense;
  const bool is_comp = p.has_compressor(layer) && !dsa_dense;

  // 1. q [T,nh,hd] and raw kv latent [T,hd] (num_key_value_heads=1 MLA). The MLA
  //    linears (wq_a, wq_b, wkv) run the keep-quant GEMM (Gemm) — the whole batch
  //    at once — then the per-token RMSNorm(q_norm/kv_norm) + per-head RoPE.
  std::vector<float> qa = Gemm(be, Lq != nullptr ? &Lq->wq_a : nullptr, L.wq_a, x, T, qlr, H);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> n = RmsNorm(Slice(qa, t * qlr, qlr), L.q_norm_weight, eps);
    for (int64_t i = 0; i < qlr; ++i) qa[t * qlr + i] = n[static_cast<size_t>(i)];
  }
  // Per-layer DUAL RoPE (ds4 layer_rope_freq_base/scale): compressed layers use
  // base=compress_rope_theta + 1/factor YaRN interpolation; dense layers plain.
  const bool rope_compressed = p.has_compressor(layer);
  const double rope_base = rope_compressed ? p.compress_rope_theta : p.rope_theta;
  const double rope_fscale =
      (rope_compressed && p.rope_scale_factor > 1.0) ? 1.0 / p.rope_scale_factor : 1.0;
  const double rope_ext =
      (rope_compressed && p.rope_scale_factor > 1.0) ? 1.0 : 0.0;
  auto rope_layer = [&](float* v, int64_t pos) {
    RopeInplaceLayer(v, rope, pos, rope_base, rope_fscale, rope_ext, p.rope_orig_ctx,
                     p.rope_beta_fast, p.rope_beta_slow);
  };
  std::vector<float> q =
      Gemm(be, Lq != nullptr ? &Lq->wq_b : nullptr, L.wq_b, qa, T, nh * hd, qlr);
  // Per-head query RMS-norm (ds4 head_rms_norm_inplace, AFTER wq_b, BEFORE RoPE) — the
  // MLA query normalization our forward previously omitted (#188: q was rel-L2 ~0.96
  // vs ds4 at L00 with a bit-exact input; the KV latent already has its attn_kv_a_norm).
  DeepseekV4QHeadRmsNormInplace(q, T * nh, hd, eps);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h)
      rope_layer(&q[t * nh * hd + h * hd + nope], positions[static_cast<size_t>(t)]);
  if (std::getenv("VT_DUMP_ACT") != nullptr && (layer <= 5 || (layer >= 28 && layer <= 34))) {
    char nm[64]; std::snprintf(nm, sizeof(nm), "ours_q_L%02lld", static_cast<long long>(layer));
    DumpAct(nm, Slice(q, 0, nh * hd));  // #188 q operand (post-proj+rope), t=0
  }
  std::vector<float> kraw = Gemm(be, Lq != nullptr ? &Lq->wkv : nullptr, L.wkv, x, T, hd, H);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> kv = RmsNorm(Slice(kraw, t * hd, hd), L.kv_norm_weight, eps);
    rope_layer(&kv[nope], positions[static_cast<size_t>(t)]);
    for (int64_t d = 0; d < hd; ++d) kraw[t * hd + d] = kv[static_cast<size_t>(d)];
  }

  // 2. compressor (compressor layers): softmax-window POOL + save-time APE + RMSNorm
  //    into the cached latent (deepseek_v4_compressor.h : CompressorSaveScoreApe /
  //    CompressorPoolNorm). Non-compressor layers cache the raw latent directly.
  std::vector<float> latent = kraw;
  if (is_comp) {
    const int64_t cr = p.compress_ratio(layer);
    const int64_t win = 2;  // tiny pooling window (device gather addressing = W7 seam)
    // compressor pool-score projection (keep-quant comp_wgate) : [T,H] -> [T,hd].
    std::vector<float> score =
        Gemm(be, Lq != nullptr ? &Lq->comp_wgate : nullptr, L.comp_wgate, x, T, hd, H);
    std::vector<int64_t> pos64(positions.begin(), positions.end());
    score = DispSaveScoreApe(be, score, L.comp_ape, pos64, T, hd, cr);
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> kvwin(static_cast<size_t>(win) * hd, 0.0f);
      std::vector<float> scwin(static_cast<size_t>(win) * hd, 0.0f);
      std::vector<uint8_t> valid(static_cast<size_t>(win), 0);
      for (int64_t i = 0; i < win; ++i) {
        const int64_t row = t - (win - 1) + i;
        if (row < 0) continue;
        valid[static_cast<size_t>(i)] = 1;
        for (int64_t d = 0; d < hd; ++d) {
          kvwin[i * hd + d] = kraw[row * hd + d];
          scwin[i * hd + d] = score[row * hd + d];
        }
      }
      std::vector<float> comp =
          DispPoolNorm(be, kvwin, scwin, valid, L.comp_norm_weight, eps, win, hd);
      for (int64_t d = 0; d < hd; ++d) latent[t * hd + d] = comp[static_cast<size_t>(d)];
    }
    if (trace != nullptr) trace->layer_compressor_ran[static_cast<size_t>(layer)] = 1;
  }

  // 3. KV state. The synthetic path round-trips through the fp8_ds_mla layout to
  //    EXERCISE the paged-cache encoding (W4). The REAL keep-quant run uses the
  //    latent DIRECTLY: MakeFp8DsMlaLayout quantizes the whole `nope` span (448 at
  //    the real geometry) as ONE fp8 block — a single scale over 448 values, whose
  //    dynamic-range loss corrupts the latent and degrades generation. Skipping it
  //    is strictly MORE faithful (ds4 keeps per-sub-block KV-cache scales; matching
  //    that block layout is a named residual). Position-precision for attention is
  //    unaffected (dense recompute; no paged cache in the stateless run).
  std::vector<float> deck;
  if (dsa_dense) {
    deck = latent;
  } else {
    const Fp8DsMlaLayout ly = MakeFp8DsMlaLayout(nope, rope, /*quant_block=*/nope);
    deck.resize(static_cast<size_t>(T) * hd);
    for (int64_t t = 0; t < T; ++t) {
      const auto tok = DispEncode(be, Slice(latent, t * hd, hd), ly);
      const std::vector<float> dec = DispDecode(be, tok, ly);
      for (int64_t d = 0; d < hd; ++d) deck[t * hd + d] = dec[static_cast<size_t>(d)];
    }
  }
  if (std::getenv("VT_DUMP_ACT") != nullptr && (layer <= 5 || (layer >= 28 && layer <= 34))) {
    char nm[64]; std::snprintf(nm, sizeof(nm), "ours_kv_L%02lld", static_cast<long long>(layer));
    DumpAct(nm, Slice(deck, 0, hd));  // #188 kv latent operand (deck), t=0
  }

  // 3b. KV CACHE (Stage 1, incremental decode). When a cache is bound, APPEND this
  //     call's T new `deck` latents to the layer's cache and attend over the FULL
  //     cached KV (global positions 0..kv_base+T-1). `deck[t]` depends only on
  //     token t and its position, so the cached value equals the recomputed one →
  //     incremental decode is token-identical to full-recompute. Null cache = the
  //     stateless path (kv_keys == the local `deck`, base 0).
  const std::vector<float>* kv_keys = &deck;
  int64_t kv_base = 0;
  int64_t n_keys = T;
  if (be.kv != nullptr) {
    VT_CHECK(!is_indexer && !is_comp,
             "kv-cache incremental decode requires dense MLA (no indexer/compressor)");
    VT_CHECK(be.kv->head_dim == hd, "kv cache head_dim mismatch");
    std::vector<float>& lc = be.kv->deck[static_cast<size_t>(layer)];
    kv_base = be.kv_base;
    VT_CHECK(static_cast<int64_t>(lc.size()) == kv_base * hd,
             "kv cache length mismatch (layer cache out of sync with kv_base)");
    lc.insert(lc.end(), deck.begin(), deck.end());  // append the T new latents
    kv_keys = &lc;
    n_keys = kv_base + T;
  }
  (void)n_keys;

  // 4. selection: DSA Lightning-Indexer top-k on indexer layers, else dense causal
  //    over GLOBAL positions [0..kv_base+t] (kv_base==0 in the stateless path).
  std::vector<std::vector<int64_t>> sel(static_cast<size_t>(T));
  if (is_indexer) {
    const int64_t inh = p.index_n_heads, ihd = p.index_head_dim, itopk = p.index_topk;
    // indexer q/k projections keep-quant (idx_wq_b / indexer_compressor_kv); the
    // weights_proj (idx_wproj) is a small V role and stays f32 (host).
    const std::vector<float> iq =
        Gemm(be, Lq != nullptr ? &Lq->idx_wq_b : nullptr, L.idx_wq, x, T, inh * ihd, H);
    const std::vector<float> ik =
        Gemm(be, Lq != nullptr ? &Lq->idx_comp_wkv : nullptr, L.idx_wk, x, T, ihd, H);
    const std::vector<float> wproj = Gemm(be, nullptr, L.idx_wproj, x, T, inh, H);
    const std::vector<float> folded = DispWeightFold(be, wproj, T, inh, ihd);
    std::vector<int64_t> ws(static_cast<size_t>(T)), we(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      ws[static_cast<size_t>(t)] = 0;
      we[static_cast<size_t>(t)] = t + 1;  // causal candidate window
    }
    const std::vector<float> logits =
        DispLogits(be, iq, ik, folded, ws, we, T, T, inh, ihd);
    const std::vector<int64_t> topk = DispTopk(be, logits, ws, we, T, T, itopk);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < itopk; ++j) {
        const int64_t s = topk[t * itopk + j];
        if (s >= 0) sel[static_cast<size_t>(t)].push_back(s);
      }
    if (trace != nullptr) {
      trace->layer_is_indexer[static_cast<size_t>(layer)] = 1;
      trace->layer_indexer_selected[static_cast<size_t>(layer)] =
          T > 0 ? static_cast<int>(sel[static_cast<size_t>(T - 1)].size()) : 0;
    }
  } else {
    for (int64_t t = 0; t < T; ++t) {
      const int64_t g = kv_base + t;  // this query's GLOBAL position
      for (int64_t s = 0; s <= g; ++s) sel[static_cast<size_t>(t)].push_back(s);
    }
  }

  // 5. attention with per-head sink softmax; key = value = the cached latent
  //    (kv_keys, GLOBAL-indexed) — the decoded MLA latent (W3 seams). Brick A:
  //    when VT_V4_DEVICE_ATTN + a CUDA queue + the V4 device kernels are live, this
  //    runs on the device kernel over the unified KV cache (dense-causal only, no
  //    indexer); else the host Dot loop. The device kernel preserves the host
  //    accumulation order (bit-identical target); the caller drains after it.
  std::vector<float> o(static_cast<size_t>(T) * nh * hd, 0.0f);
  const float kNegInf = -std::numeric_limits<float>::infinity();
  const bool dev_attn = be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU &&
                        !is_indexer && DeviceAttnEnabled() &&
                        deepseek_v4::V4DeviceKernelsAvailable();
  if (dev_attn) {
    // kv_keys holds the cached deck [n_keys_total, hd]; sel is dense-causal, so the
    // device kernel derives it from kv_base+t (no per-key index list needed).
    deepseek_v4::DsaDevice()->decode_attn(
        *be.q, o.data(), q.data(), kv_keys->data(), L.attn_sink.data(), nh, hd, kv_base, T,
        scale, /*no_sink=*/miswire == V4Miswire::kNoAttnSink);
    SyncDeviceGemm(be);  // Brick A drains; Brick D will capture instead
  } else {
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<int64_t>& S = sel[static_cast<size_t>(t)];
      for (int64_t h = 0; h < nh; ++h) {
        std::vector<float> sc(S.size());
        const float* qh = &q[(t * nh + h) * hd];
        for (size_t si = 0; si < S.size(); ++si)
          sc[si] = Dot(qh, &(*kv_keys)[S[si] * hd], hd) * scale;
        const float sink = (miswire == V4Miswire::kNoAttnSink)
                               ? kNegInf
                               : L.attn_sink[static_cast<size_t>(h)];
        const std::vector<float> probs = DispSoftmaxSink(be, sc, sink);
        float* oh = &o[(t * nh + h) * hd];
        for (size_t si = 0; si < S.size(); ++si) {
          const float w = probs[si];
          const float* v = &(*kv_keys)[S[si] * hd];
          for (int64_t d = 0; d < hd; ++d) oh[d] += w * v[d];
        }
      }
    }
  }

  // 5b. INVERSE RoPE on the attention output heads (ds4 `rope_tail_layer_inplace(
  //     heads, ..., inverse=true)`, AFTER attention, BEFORE the o-proj). The value =
  //     the rope-rotated latent, so the position rotation must be undone on the output
  //     before the o-LoRA. IDENTITY at pos 0 (so it is invisible on the pos-0 single-
  //     token gate) but load-bearing for multi-token generation (pos>0). #188.
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < nh; ++h)
      RopeInplaceLayer(&o[(t * nh + h) * hd + nope], rope, positions[static_cast<size_t>(t)],
                       rope_base, rope_fscale, rope_ext, p.rope_orig_ctx, p.rope_beta_fast,
                       p.rope_beta_slow, /*inverse=*/true);

  // 6. grouped OUTPUT-LoRA (W3) : [T,nh,hd] -> [T,H]. Keep-quant wo_a/wo_b on the
  //    GGUF source; the host/device-synthetic path keeps the f32 primitive.
  if (be.gguf != nullptr && Lq != nullptr) {
    return GroupedOutputLoraGguf(be, Lq->wo_a, Lq->wo_b, o, T, nh, hd, p.o_groups,
                                 p.o_lora_rank, H);
  }
  return DispGroupedOLora(be, o, L.wo_a, L.wo_b, T, nh, hd, p.o_groups, p.o_lora_rank, H);
}

// ── DeepSeek-V4 MoE block (W6 primitives) : [T,H] -> [T,H] ────────────────────
std::vector<float> MoeBlock(const DeepseekV4LayerHostWeights& L,
                            const DeepseekV4GgufLayerWeights* Lq,
                            const DeepseekV4Params& p, const std::vector<float>& x,
                            const std::vector<int32_t>& token_ids, int64_t layer,
                            V4Miswire miswire, V4ForwardTrace* trace, const V4Backend& be) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t ne = p.n_routed_experts;
  const int64_t topk = p.num_experts_per_tok;
  const int64_t mi = p.moe_intermediate_size;
  const float lim = static_cast<float>(p.swiglu_limit);
  const bool cfg_hash = p.is_hash_layer(layer);
  const bool hash_route = cfg_hash && miswire != V4Miswire::kAllLayersGated;
  const bool kq = be.gguf != nullptr && Lq != nullptr;

  // router gating logits [T, ne] (keep-quant moe_gate).
  const std::vector<float> gating =
      Gemm(be, kq ? &Lq->moe_gate : nullptr, L.gate_weight, x, T, ne, H);
  std::vector<int64_t> in_tokens;
  std::vector<int32_t> hashtab;
  std::vector<float> bias;
  if (hash_route) {
    in_tokens.assign(token_ids.begin(), token_ids.end());
    hashtab = L.tid2eid;
  } else {
    bias = L.gate_bias;  // may be empty (then plain top-k on the unbiased scores)
  }
  if (std::getenv("VT_DUMP_ACT") != nullptr && layer == 34) {  // #188 router logits/bias
    DumpAct("ours_gating_L34", std::vector<float>(gating.begin(), gating.begin() + ne));
    DumpAct("ours_gatebias_L34", bias.empty() ? std::vector<float>(ne, 0.0f) : bias);
    std::fprintf(stderr, "  [router L34] logit[33]=%.4f logit[233]=%.4f bias[33]=%.4f bias[233]=%.4f\n",
                 gating[33], gating[233], bias.empty() ? 0.f : bias[33], bias.empty() ? 0.f : bias[233]);
  }
  const MoeRouteResult route =
      DispRoute(be, gating, T, ne, topk, bias, p.norm_topk_prob,
                static_cast<float>(p.routed_scaling_factor), in_tokens, hashtab, p.vocab_size);
  if (trace != nullptr) {
    trace->layer_is_hash[static_cast<size_t>(layer)] = cfg_hash ? 1 : 0;
    trace->layer_hash_routed[static_cast<size_t>(layer)] = hash_route ? 1 : 0;
  }

  // one clamped-SwiGLU expert on the f32 host tower: w1/w3 [mi,H], w2 [H,mi].
  const auto expert_f32 = [&](const float* w1, const float* w3, const float* w2,
                              const float* xin) -> std::vector<float> {
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = Dot(&w1[r * H], xin, H);
      gate_up[static_cast<size_t>(mi + r)] = Dot(&w3[r * H], xin, H);
    }
    const std::vector<float> act = DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
    std::vector<float> out(static_cast<size_t>(H));
    for (int64_t hh = 0; hh < H; ++hh)
      out[static_cast<size_t>(hh)] = Dot(&w2[hh * mi], act.data(), mi);
    return out;
  };
  // clamped-SwiGLU host activation from separate gate `g` and up `u` [mi] vectors.
  const auto swiglu = [&](const std::vector<float>& g,
                          const std::vector<float>& u) -> std::vector<float> {
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) {
      gate_up[static_cast<size_t>(r)] = g[static_cast<size_t>(r)];
      gate_up[static_cast<size_t>(mi + r)] = u[static_cast<size_t>(r)];
    }
    return DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
  };

  std::vector<float> out(static_cast<size_t>(T) * H, 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> x1(x.begin() + t * H, x.begin() + (t + 1) * H);
    const bool dbg = std::getenv("VT_DUMP_ACT") != nullptr && t == 0;

    if (kq) {
      // ── Stage 2: BATCHED expert GEMMs. The shared + `topk` routed experts are
      //    mutually independent and gate/up feed only their own down, so issue all
      //    gate+up (deferred, no per-GEMM sync), DRAIN ONCE, host clamped-SwiGLU,
      //    issue all down (deferred), DRAIN ONCE, then combine. Was ~21 stream
      //    drains/layer (one per GEMM) → 2. BYTE-IDENTICAL to the per-expert path:
      //    same GEMMs on the same stream (serialized), only the host drain is
      //    amortized. Index 0 = shared expert (weight 1); 1..topk = routed.
      const int64_t A = 1 + topk;
      std::vector<std::vector<float>> g(static_cast<size_t>(A)), u(static_cast<size_t>(A)),
          act(static_cast<size_t>(A)), eo(static_cast<size_t>(A));
      // The grouped kernel requires a BLOCK-QUANT stacked weight (the keep-quant
      // load); the near-tie dequant oracle (kExpandAll) has bf16 expert weights, so
      // fall back to the per-expert path there.
      const bool grouped = be.grouped_moe && vt::IsBlockQuant(Lq->moe_gate_exps.dtype) &&
                           vt::IsBlockQuant(Lq->moe_up_exps.dtype) &&
                           vt::IsBlockQuant(Lq->moe_down_exps.dtype);
      // The topk routed expert ids (i32), shared by phases 1 and 3.
      std::vector<int32_t> eids(static_cast<size_t>(topk));
      for (int64_t j = 0; j < topk; ++j)
        eids[static_cast<size_t>(j)] = static_cast<int32_t>(route.topk_ids[t * topk + j]);

      // phase 1: gate + up. Shared expert stays a per-expert Gemm; the topk routed
      // experts collapse into ONE grouped kMatmulBTQuantGrouped launch each when
      // grouped_moe (else the Stage-2 per-expert GemmRowSlice batch).
      g[0] = Gemm(be, &Lq->shared_gate, {}, x1, 1, mi, H, /*defer_sync=*/true);
      u[0] = Gemm(be, &Lq->shared_up, {}, x1, 1, mi, H, /*defer_sync=*/true);
      if (grouped) {
        std::vector<float> xrep(static_cast<size_t>(topk) * H);  // topk copies of x1
        for (int64_t j = 0; j < topk; ++j)
          std::copy(x1.begin(), x1.end(), xrep.begin() + j * H);
        const std::vector<float> gr =
            GemmGroupedExpertsKq(be, Lq->moe_gate_exps, xrep, eids, topk, mi, H);
        const std::vector<float> ur =
            GemmGroupedExpertsKq(be, Lq->moe_up_exps, xrep, eids, topk, mi, H);
        for (int64_t j = 0; j < topk; ++j) {
          g[1 + j].assign(gr.begin() + j * mi, gr.begin() + (j + 1) * mi);
          u[1 + j].assign(ur.begin() + j * mi, ur.begin() + (j + 1) * mi);
        }
      } else {
        for (int64_t j = 0; j < topk; ++j) {
          const int64_t e = route.topk_ids[t * topk + j];
          g[1 + j] = GemmRowSlice(be, Lq->moe_gate_exps, x1, 1, mi, H, e * mi, /*defer_sync=*/true);
          u[1 + j] = GemmRowSlice(be, Lq->moe_up_exps, x1, 1, mi, H, e * mi, /*defer_sync=*/true);
        }
      }
      DrainDevice(be);
      // phase 2: host clamped-SwiGLU
      for (int64_t a = 0; a < A; ++a) act[static_cast<size_t>(a)] = swiglu(g[a], u[a]);
      // phase 3: down. Shared per-expert; routed grouped when grouped_moe.
      eo[0] = Gemm(be, &Lq->shared_down, {}, act[0], 1, H, mi, /*defer_sync=*/true);
      if (grouped) {
        std::vector<float> adown(static_cast<size_t>(topk) * mi);
        for (int64_t j = 0; j < topk; ++j)
          std::copy(act[static_cast<size_t>(1 + j)].begin(), act[static_cast<size_t>(1 + j)].end(),
                    adown.begin() + j * mi);
        const std::vector<float> er =
            GemmGroupedExpertsKq(be, Lq->moe_down_exps, adown, eids, topk, H, mi);
        for (int64_t j = 0; j < topk; ++j)
          eo[1 + j].assign(er.begin() + j * H, er.begin() + (j + 1) * H);
      } else {
        for (int64_t j = 0; j < topk; ++j) {
          const int64_t e = route.topk_ids[t * topk + j];
          eo[1 + j] = GemmRowSlice(be, Lq->moe_down_exps, act[static_cast<size_t>(1 + j)], 1, H, mi,
                                   e * H, /*defer_sync=*/true);
        }
      }
      DrainDevice(be);
      // phase 4: combine  out = shared + Σ_j w_j · eo_j
      for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += eo[0][static_cast<size_t>(hh)];
      for (int64_t j = 0; j < topk; ++j) {
        const float w = route.topk_weights[t * topk + j];
        for (int64_t hh = 0; hh < H; ++hh)
          out[t * H + hh] += w * eo[static_cast<size_t>(1 + j)][static_cast<size_t>(hh)];
      }
      if (dbg) {
        double sh_r = 0; for (float v : eo[0]) sh_r += (double)v * v; sh_r = std::sqrt(sh_r / H);
        double rt_r = 0, wmax = 0, eo_rmax = 0;
        for (int64_t j = 0; j < topk; ++j) {
          const int64_t e = route.topk_ids[t * topk + j];
          const float w = route.topk_weights[t * topk + j];
          double er = 0; for (float v : eo[1 + j]) er += (double)v * v; er = std::sqrt(er / H);
          wmax = std::max(wmax, (double)std::fabs(w)); eo_rmax = std::max(eo_rmax, er);
          double c = 0; for (float v : eo[1 + j]) c += (double)(w * v) * (w * v); rt_r += c;
          std::fprintf(stderr, "    [moe L%02lld] routed e=%lld w=%.4f eo_rms=%.3f\n",
                       static_cast<long long>(layer), static_cast<long long>(e), w, er);
        }
        std::fprintf(stderr, "  [moe L%02lld] shared_rms=%.3f routed_rms=%.3f wmax=%.3f expert_out_rmax=%.3f\n",
                     static_cast<long long>(layer), sh_r, std::sqrt(rt_r / H), wmax, eo_rmax);
      }
    } else {
      // f32 host path (unchanged): pure host Dot, no device queue / sync.
      const std::vector<float> sh = expert_f32(L.shared_w1.data(), L.shared_w3.data(),
                                               L.shared_w2.data(), &x[t * H]);
      for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += sh[static_cast<size_t>(hh)];
      double sh_r = 0; if (dbg) { for (float v : sh) sh_r += (double)v * v; sh_r = std::sqrt(sh_r / H); }
      double rt_r = 0, wmax = 0, eo_rmax = 0;
      for (int64_t j = 0; j < topk; ++j) {
        const int64_t e = route.topk_ids[t * topk + j];
        const float w = route.topk_weights[t * topk + j];
        const std::vector<float> eo = expert_f32(&L.exp_w1[e * mi * H], &L.exp_w3[e * mi * H],
                                                 &L.exp_w2[e * H * mi], &x[t * H]);
        if (dbg) {
          double er = 0; for (float v : eo) er += (double)v * v; er = std::sqrt(er / H);
          wmax = std::max(wmax, (double)std::fabs(w)); eo_rmax = std::max(eo_rmax, er);
          double c = 0; for (float v : eo) c += (double)(w * v) * (w * v); rt_r += c;
          std::fprintf(stderr, "    [moe L%02lld] routed e=%lld w=%.4f eo_rms=%.3f\n",
                       static_cast<long long>(layer), static_cast<long long>(e), w, er);
        }
        for (int64_t hh = 0; hh < H; ++hh) out[t * H + hh] += w * eo[static_cast<size_t>(hh)];
      }
      if (dbg)
        std::fprintf(stderr, "  [moe L%02lld] shared_rms=%.3f routed_rms=%.3f wmax=%.3f expert_out_rmax=%.3f\n",
                     static_cast<long long>(layer), sh_r, std::sqrt(rt_r / H), wmax, eo_rmax);
    }
  }
  return out;
}

constexpr const char* kHostPending =
    "DeepseekV4 forward: host-float weight tower not materialized — the W7 "
    "tiny-config CPU composition runs off DeepseekV4Weights::host (populated by the "
    "structural gate, test_deepseek_v4_forward.cpp); the real-checkpoint FP8-block + "
    "NVFP4 tower materialization is the named W2b residual and the device forward is "
    "W7-device. See .agents/specs/deepseek-v4-flash.md §W7.";

constexpr const char* kDevicePending =
    "DeepseekV4 DEVICE forward (W7-device) not implemented — the tiny-config CPU "
    "composition lands in DeepseekV4Model::Forward / DeepseekV4ForwardHost; the CUDA "
    "kernels (MHC Sinkhorn, DSA indexer/compressor, sqrtsoftplus router, clamped "
    "SwiGLU; the expert GEMM REUSES the existing NVFP4/FP8 grouped-GEMM) + the real "
    "multi-Spark e2e (W8) are named residuals. See .agents/specs/deepseek-v4-flash.md §W7.";

// VT_DUMP_ACT: env-gated per-layer activation dump (coherence-debug #188). Writes a
// raw little-endian float32 vector to $VT_DUMP_ACT/<name>.bin (inert when unset).
void DumpAct(const char* name, const std::vector<float>& v) {
  const char* dir = std::getenv("VT_DUMP_ACT");
  if (dir == nullptr) return;
  const std::string path = std::string(dir) + "/" + name + ".bin";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f != nullptr) {
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
  }
}

}  // namespace

// ─── Brick C part 2: the DEVICE-RESIDENT T=1 decode forward ───────────────────
// The host-orchestrated ForwardComposeImpl drains the stream ~560×/step (each GEMM
// + each Disp* device kernel syncs so the host can read/copy its output). Here the
// whole 43-layer T=1 step runs as ONE async device chain over the (unified) buffers
// with NO per-op sync: every GEMM defers, the small host primitives (q/kv/final
// RMSNorm, per-head q-RMS, dual+inverse RoPE, MoE combine) run on the Brick-C device
// kernels IN PLACE, and routing stays resident (device router → i32 topk_ids that
// the grouped expert GEMM consumes on-device; topk_weights that the combine kernel
// consumes) — no host gather. The ONLY drain is before the host reads the [V] logits
// for argmax (the step boundary). CUDA + keep-quant GGUF + dense-causal (no compressor/
// indexer) + T==1 only — the decode benchmark path (§7 of the device-decode spec).
//
// DEFAULT ON (validated 2026-07-30): the device-resident decode is the fastest path on
// GB10 (7.96 tok/s vs the host path's ~4.3-7.2, ~1.8× at 256-token context) and is
// token-identical/characterized-coherent-near-tie to host across a 4-prompt × 256-token
// validation (P0 token-identical; open-ended prompts diverge only at genuine near-tie
// positions — bounded kernel noise can only flip host's ~tied top-2 — into coherent,
// deterministic continuations). `VT_V4_RESIDENT_DECODE=0` is the rollback off-switch
// (→ the host ForwardComposeImpl). The guard (CanRunResidentDecode) still falls back to
// host for CPU / non-dense / T>1 / no-KV-cache, so those paths are UNCHANGED.
namespace {

inline bool ResidentDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_RESIDENT_DECODE");
    return e == nullptr || std::string(e) != "0";  // default ON; `=0` rolls back to host
  }();
  return on;
}

// Async device copy on the queue stream (cudaMemcpyAsync — no sync). Used to
// assemble the grouped-GEMM inputs (topk row-copies of x) + the [gate|up] pairs.
inline void AsyncCopyF(const V4Backend& be, float* dst, const float* src, int64_t n) {
  vt::GetBackend(be.q->device).Copy(*be.q, dst, src,
                                    static_cast<size_t>(n) * sizeof(float));
}

// ── VT_V4_RESIDENT_W (default OFF; =1 stages the dense Q8_0 decode projection tower
// TRUE device-resident). MECHANISM transferred from Laguna's VT_LAGUNA_RESIDENT_BF16W
// (laguna.cpp:125 LagunaResidentBf16W): on GB10 the GPU reads system-allocated
// (ATS/unified) host memory slower per-GEMV than cudaMalloc'd device memory. The
// keep-quant MLA / shared-expert / lm_head Q8_0 weights are consumed here via
// `wq.View(); .device=dev` — a unified-memory RETAG of the GGUF mmap's read-only file
// pages — so every decode projection GEMV reads host bytes over ATS. Staging the ~6 GiB
// dense Q8_0 tower cudaMalloc-device ONCE (lazily, on first touch, cached in the
// OwnedTensor's mutable d_dev) and reusing the device copy every step recovers the
// per-call GEMV bandwidth. SAME bytes, SAME kMatmulBTQuant / matmul_q8_0_* kernel, SAME
// invocation ⇒ byte-exact by construction. CAPTURE-SAFE: the first touch is the eager
// prefill / gstate-0 decode-graph warm run, BEFORE any BeginCapture, so the captured
// replay reads a stable device pointer and does ZERO fresh cudaMalloc (mirrors the
// Laguna gstate-0 pattern). The source bytes are the GGUF mmap (borrowed, read-only),
// so this is ADDITIVE (~6 GiB) rather than move-not-duplicate; the clean file pages are
// evictable under pressure. Default OFF as the A/B artifact; recommend-flip only after
// an on-box bit-exact decode win. The routed-expert slabs (the ~70 GiB bulk) are NOT
// staged here — that is the Phase-2 move-semantics surface (GemmGroupedInto).
inline bool V4ResidentWEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_RESIDENT_W");
    return e == nullptr || e[0] == '\0' || std::string(e) != "0";  // default ON (parity enabler, beats ds4); =0 opts out
  }();
  return on;
}

// Upload a keep-quant projection weight's host bytes → a cudaMalloc device copy ONCE
// (cached in the OwnedTensor's mutable d_dev), returning the device base pointer. `gq`
// must be the device queue. Mirrors LagunaResidentBf16W's lazy Alloc+Copy.
inline const uint8_t* V4ResidentBase(vt::Queue& gq, const OwnedTensor& w) {
  if (!w.d_dev) {
    vt::Backend& b = vt::GetBackend(gq.device);
    const size_t nb = w.bytes.size();
    void* p = b.Alloc(nb);
    b.Copy(gq, p, w.bytes.data(), nb);
    vt::Backend* bk = &b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* pp) { bk->Free(pp); });
  }
  return static_cast<const uint8_t*>(w.d_dev.get());
}

// Device operand for a keep-quant projection weight consumed by a decode GEMV. Lever ON:
// a View over the true-device copy. Lever OFF: the legacy unified-memory retag (host
// mmap bytes, device-tagged). Byte-identical either way — only the backing allocation
// (true-device vs ATS host) differs; shape/stride/nk/dtype/q8_0_aligned metadata comes
// straight from w.View(), so it is exactly the retag path's. `gq` must be the device queue.
inline vt::Tensor V4ResidentW(vt::Queue& gq, const OwnedTensor& w) {
  vt::Tensor t = w.View();
  t.device = gq.device;
  if (V4ResidentWEnabled() && !w.bytes.empty())
    t.data = const_cast<uint8_t*>(V4ResidentBase(gq, w));
  return t;
}

// ── VT_V4_RESIDENT_EXPERTS (default OFF; =1 stages the ~70 GiB routed-expert
// keep-quant slabs TRUE device-resident with MOVE semantics — Phase-2, extending
// Phase-1's dense-tower VT_V4_RESIDENT_W). The three stacked expert slabs per layer
// (moe_gate_exps/up/down, IQ2_XXS/Q2_K) are mmap-BORROWED whole-file GGUF pages
// (OwnedBytes::Borrow, deepseek_v4_weights.cpp Sew()) consumed by the grouped GEMM via
// an ATS retag (`weight.View(); .device=dev`) — every routed GEMM reads host bytes over
// ATS/unified, the same per-GEMV penalty Phase-1 fixed for the dense Q8_0 tower.
//
// A NAIVE additive stage (Phase-1's V4ResidentBase, keeping the mmap pages) would
// transiently need ~156 GiB (70 mmap + 70 device) > the 119 GiB unified pool → OOM. So
// this is a MOVE, not an add: each slab is staged cudaMalloc-device ONCE (lazily, on
// first touch — the eager gstate-0 decode-graph warm run, BEFORE any BeginCapture,
// mirroring Phase-1's capture-safe first touch), and its borrowed mmap pages are
// reclaimed with madvise(MADV_DONTNEED) IMMEDIATELY after the copy completes, so the
// transient never exceeds ~one slab (~0.5 GiB) over the resident baseline and PEAK stays
// flat (device grows as host shrinks, in lockstep). SAME bytes, SAME
// MatmulBTQuantGrouped kernel, SAME invocation ⇒ byte-exact by construction. The mmap
// VMA stays mapped (only resident pages drop; the borrow keep-alive still pins the
// mapping for every other tensor); the resident path never reads the host bytes again —
// the consumer reads the device copy. Default OFF as the A/B artifact; recommend-flip
// only after an on-box bit-exact decode win. Rides the mmap-borrow residency (the slabs
// must be borrowed, i.e. VT_GGUF_MMAP on — the production default for this model).
inline bool V4ResidentExpertsEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_RESIDENT_EXPERTS");
    return e != nullptr && e[0] != '\0' && std::string(e) != "0";  // default OFF; =1 opts in
  }();
  return on;
}

// Reclaim the resident pages of a keep-quant slab's BORROWED mmap byte range after it
// has been copied device-resident for the LAST time. INTERIOR whole pages only — a
// boundary page may hold the first/last bytes of a neighbouring tensor still borrowed in
// place, so it is never dropped (partial edge pages stay). This is the exact page math of
// GgufFile::DropSpanResidency (gguf_reader.cpp:602, llama.cpp `unmap_fragment` adapted to
// an interleaved file); replicated here because the consumer holds only the OwnedTensor
// (a type-erased borrow), not the GgufFile. The mapping is PROT_READ / MAP_PRIVATE and
// file-backed, so this is a pure residency hint — a later read (never taken on the
// resident path) simply re-faults the same bytes. No-op for an OWNED (non-borrowed)
// buffer: there are no mmap pages to reclaim.
inline void V4DropBorrowedResidency(const OwnedBytes& b) {
#if defined(__unix__)
  if (!b.borrowed() || b.empty()) return;
  const long ps_l = ::sysconf(_SC_PAGESIZE);
  const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
  const auto begin = reinterpret_cast<uintptr_t>(b.data());
  const uintptr_t end = begin + b.size();
  const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);  // round UP
  const uintptr_t page_end = end & ~(ps - 1);                 // round DOWN
  if (page_end > page_begin) {
    // Best-effort by contract: a failure costs resident pages, never correctness.
    (void)::madvise(reinterpret_cast<void*>(page_begin),
                    static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
  }
#else
  (void)b;
#endif
}

// Stage a routed-expert slab's host bytes → a cudaMalloc device copy ONCE (cached in the
// OwnedTensor's mutable d_dev), then MOVE: synchronize the copy and reclaim the borrowed
// mmap pages. The SYNCHRONIZE is MANDATORY — Backend::Copy is cudaMemcpyAsync, so dropping
// the source pages before the DMA has read them would corrupt the device copy. It fires
// only on the one-time first touch (d_dev null), during the eager warm run; steady-state
// replay reuses d_dev with ZERO Alloc/Copy/Sync/madvise (capture-safe). `gq` must be the
// device queue. Mirrors V4ResidentBase (Phase-1) plus the move-semantics reclaim.
inline const uint8_t* V4ResidentExpertBase(vt::Queue& gq, const OwnedTensor& w) {
  if (!w.d_dev) {
    vt::Backend& b = vt::GetBackend(gq.device);
    const size_t nb = w.bytes.size();
    void* p = b.Alloc(nb);
    b.Copy(gq, p, w.bytes.data(), nb);  // async H2D (reads/faults in the whole slab)
    b.Synchronize(gq);                  // the copy MUST complete before dropping the source
    V4DropBorrowedResidency(w.bytes);   // MOVE: reclaim the mmap pages (PEAK stays flat)
    vt::Backend* bk = &b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* pp) { bk->Free(pp); });
  }
  return static_cast<const uint8_t*>(w.d_dev.get());
}

// Device operand for a stacked routed-expert weight consumed by the grouped GEMM. Lever
// ON: a View over the true-device (moved) copy. Lever OFF: the legacy ATS retag (host mmap
// bytes, device-tagged). Byte-identical either way — only the backing allocation differs;
// shape/stride/nk/dtype come straight from w.View(), so it is exactly the retag path's
// metadata. `gq` must be the device queue.
inline vt::Tensor V4ResidentExpertW(vt::Queue& gq, const OwnedTensor& w) {
  vt::Tensor t = w.View();
  t.device = gq.device;
  if (V4ResidentExpertsEnabled() && !w.bytes.empty())
    t.data = const_cast<uint8_t*>(V4ResidentExpertBase(gq, w));
  return t;
}

// Keep-quant GEMM into a caller-provided unified `out` (T rows), NO sync — the
// resident chain drains once at the end. Mirrors Gemm's device branch; a
// block-quant weight → the device kMatmulBTQuant, a bf16 weight (e.g. the router
// gate) → the synchronous CPU MatmulBT (still writes `out`; the caller drains the
// stream FIRST when such a CPU GEMM reads a device-produced input).
void GemmIntoKq(const V4Backend& be, const OwnedTensor& wq, const float* x, float* out,
                int64_t T, int64_t N, int64_t K) {
  VT_CHECK(be.q != nullptr && !wq.Empty(), "resident GemmInto needs a queue + kq weight");
  VT_CHECK(wq.rank == 2 && wq.shape[0] == N && wq.shape[1] == K,
           "resident GemmInto: weight shape mismatch");
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = be.q->device.type != vt::DeviceType::kCPU && vt::IsBlockQuant(wq.dtype);
  vt::Queue& gq = on_dev ? *be.q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out, vt::DType::kF32, gq.device, {T, N});
  vt::Tensor w = wq.View();
  w.device = gq.device;
  if (on_dev) w = V4ResidentW(gq, wq);  // VT_V4_RESIDENT_W: device copy vs ATS retag
  vt::MatmulBT(gq, o, a, w);  // NO SyncDeviceGemm — resident
}

// Keep-quant GEMM against a block ROW-SLICE [row_off, row_off+N) of a stacked weight,
// into `out`, NO sync (the resident o-LoRA per-group wo_a). Mirrors GemmRowSlice.
void GemmRowSliceInto(const V4Backend& be, const OwnedTensor& w, const float* x, float* out,
                      int64_t T, int64_t N, int64_t K, int64_t row_off) {
  VT_CHECK(!w.repacked && w.rank == 2 && row_off >= 0 && row_off + N <= w.shape[0] &&
               w.shape[1] == K,
           "resident GemmRowSliceInto: slice out of range / repacked");
  const size_t row_bytes = vt::RowSizeBytes(w.dtype, K);
  vt::Queue cpuq{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const bool on_dev = be.q->device.type != vt::DeviceType::kCPU && vt::IsBlockQuant(w.dtype);
  vt::Queue& gq = on_dev ? *be.q : cpuq;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, gq.device, {T, K});
  vt::Tensor o = vt::Tensor::Contiguous(out, vt::DType::kF32, gq.device, {T, N});
  // VT_V4_RESIDENT_W: slice into the true-device copy (base + row offset) vs the ATS host
  // mmap bytes. Byte-identical layout — only the backing allocation differs.
  const uint8_t* wbase = (on_dev && V4ResidentWEnabled() && !w.bytes.empty())
                             ? V4ResidentBase(gq, w)
                             : w.bytes.data();
  vt::Tensor wt;
  wt.data = const_cast<uint8_t*>(wbase) + static_cast<size_t>(row_off) * row_bytes;
  wt.dtype = w.dtype;
  wt.device = gq.device;
  wt.rank = 2;
  wt.shape[0] = N;
  wt.shape[1] = K;
  wt.stride[0] = K;
  wt.stride[1] = 1;
  vt::MatmulBT(gq, o, a, wt);  // NO sync — resident
}

// Brick 12 (ds4-gap "launch consolidation"): pair/consolidate the resident Q8_0 decode
// projection launches (default ON — a parity/perf enabler ships as default). VT_V4_Q8_PAIR=0
// rolls back to the per-projection / per-group launches (bit-identical) for A/B + rollback.
inline bool Q8PairEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_Q8_PAIR");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Brick 12: PAIRED keep-quant decode GEMV — out0=wq0·x, out1=wq1·x over the SAME x in
// ONE launch (the two A-projections that share the layer hidden: MLA wq_a+wkv;
// shared-expert gate+up). Quantizes x ONCE. Falls back to two GemmIntoKq when the flag
// is off or the weights are not both device Q8_0. BIT-IDENTICAL either way (the pair
// kernel preserves each output's integer __dp4a order + warp reduce + scale fold).
void GemmPairIntoKq(const V4Backend& be, const OwnedTensor& wq0, const OwnedTensor& wq1,
                    const float* x, float* out0, float* out1, int64_t N0, int64_t N1, int64_t K) {
  const bool on_dev = be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU;
  const bool pair = Q8PairEnabled() && on_dev && !wq0.Empty() && !wq1.Empty() &&
                    wq0.dtype == vt::DType::kQ8_0 && wq1.dtype == vt::DType::kQ8_0 &&
                    !wq0.repacked && !wq1.repacked;
  if (!pair) {
    GemmIntoKq(be, wq0, x, out0, 1, N0, K);
    GemmIntoKq(be, wq1, x, out1, 1, N1, K);
    return;
  }
  VT_CHECK(wq0.rank == 2 && wq0.shape[0] == N0 && wq0.shape[1] == K && wq1.rank == 2 &&
               wq1.shape[0] == N1 && wq1.shape[1] == K,
           "GemmPairIntoKq: weight shape mismatch");
  vt::Queue& gq = *be.q;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, gq.device, {1, K});
  vt::Tensor o0 = vt::Tensor::Contiguous(out0, vt::DType::kF32, gq.device, {1, N0});
  vt::Tensor o1 = vt::Tensor::Contiguous(out1, vt::DType::kF32, gq.device, {1, N1});
  vt::Tensor w0 = V4ResidentW(gq, wq0);  // VT_V4_RESIDENT_W: device copy vs ATS retag
  vt::Tensor w1 = V4ResidentW(gq, wq1);
  deepseek_v4::DsaDevice()->matmul_q8_0_pair(gq, o0, o1, a, w0, w1);  // NO sync — resident
}

// Brick 12: BLOCK-DIAGONAL consolidation of the ng grouped output-LoRA `wo_a` GEMVs into
// ONE launch. Falls back to the ng-slice loop when the flag is off / not device Q8_0.
// BIT-IDENTICAL (each group's 32-block quant + per-row dot are unchanged; see kernel).
void OloraAIntoKq(const V4Backend& be, const OwnedTensor& wo_a, const float* o, float* z,
                  int64_t ng, int64_t olr, int64_t ipg) {
  const bool on_dev = be.q != nullptr && be.q->device.type != vt::DeviceType::kCPU;
  const bool cons = Q8PairEnabled() && on_dev && !wo_a.Empty() &&
                    wo_a.dtype == vt::DType::kQ8_0 && !wo_a.repacked;
  if (!cons) {
    for (int64_t gp = 0; gp < ng; ++gp)
      GemmRowSliceInto(be, wo_a, o + gp * ipg, z + gp * olr, 1, olr, ipg, gp * olr);
    return;
  }
  VT_CHECK(wo_a.rank == 2 && wo_a.shape[0] == ng * olr && wo_a.shape[1] == ipg,
           "OloraAIntoKq: wo_a shape mismatch");
  vt::Queue& gq = *be.q;
  vt::Tensor a =
      vt::Tensor::Contiguous(const_cast<float*>(o), vt::DType::kF32, gq.device, {1, ng * ipg});
  vt::Tensor zt = vt::Tensor::Contiguous(z, vt::DType::kF32, gq.device, {1, ng * olr});
  vt::Tensor w = V4ResidentW(gq, wo_a);  // VT_V4_RESIDENT_W: device copy vs ATS retag
  deepseek_v4::DsaDevice()->matmul_q8_0_olora_a(gq, zt, a, w, ng);  // NO sync — resident
}

// Grouped keep-quant expert GEMM into `out`, NO sync. `eids` (i32) is a unified
// buffer the device kernel consumes on-stream (the resident routing — no host
// gather). Mirrors GemmGroupedExpertsKq without the drain.
// `act_rows` is the number of rows in `act`: P (one per expert) OR 1 (a shared hidden
// BROADCAST across all P experts — the routed gate/up preq-reuse: quantize x ONCE, feed
// every expert; the provider quantizes a single row and reads it for all p). Bit-exact.
void GemmGroupedInto(const V4Backend& be, const OwnedTensor& weight, const float* act,
                     const int32_t* eids, int64_t P, int64_t N, int64_t K, float* out,
                     int64_t act_rows = -1) {
  VT_CHECK(be.q != nullptr && !weight.repacked && vt::IsBlockQuant(weight.dtype),
           "resident grouped expert GEMM: needs a non-repacked block-quant stacked weight");
  const int64_t Pa = act_rows > 0 ? act_rows : P;
  vt::Queue& gq = *be.q;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(act), vt::DType::kF32, gq.device, {Pa, K});
  vt::Tensor o = vt::Tensor::Contiguous(out, vt::DType::kF32, gq.device, {P, N});
  vt::Tensor eid =
      vt::Tensor::Contiguous(const_cast<int32_t*>(eids), vt::DType::kI32, gq.device, {P});
  // VT_V4_RESIDENT_EXPERTS (Phase-2): device (moved) copy vs the ATS retag. Byte-identical.
  vt::Tensor w = V4ResidentExpertW(gq, weight);
  vt::MatmulBTQuantGrouped(gq, o, a, w, eid);  // NO sync — resident
}

// FUSED routed-MoE gate+up+SwiGLU into `out` (adown[P,mi]), NO sync. Collapses the
// {gate grouped-GEMM + up grouped-GEMM + P×2 AsyncCopyF + P ClampedSwiGLU} chain
// into ONE launch (ds4 moe_gate_up_mid). Bit-identical: the gate/up dots reuse the
// same integer core + warp reduce as GemmGroupedInto's two GEMMs, and the SwiGLU
// epilogue matches ClampedSwiGLUKernel (α=1,β=0). The route weight is NOT folded —
// it stays in moe_combine (post-down). `x` is the shared hidden (act_rows=1
// broadcast: quantize once, feed every expert). Requires both towers block-quant.
void MoeGateUpSwiGLUInto(const V4Backend& be, const OwnedTensor& gate_w, const OwnedTensor& up_w,
                         const float* x, const int32_t* eids, int64_t P, int64_t mi, int64_t H,
                         float limit, float* out) {
  VT_CHECK(be.q != nullptr && !gate_w.repacked && !up_w.repacked &&
               vt::IsBlockQuant(gate_w.dtype) && vt::IsBlockQuant(up_w.dtype),
           "resident fused MoE gate+up: needs non-repacked block-quant gate/up towers");
  vt::Queue& gq = *be.q;
  vt::Tensor a = vt::Tensor::Contiguous(const_cast<float*>(x), vt::DType::kF32, gq.device, {1, H});
  vt::Tensor o = vt::Tensor::Contiguous(out, vt::DType::kF32, gq.device, {P, mi});
  vt::Tensor eid =
      vt::Tensor::Contiguous(const_cast<int32_t*>(eids), vt::DType::kI32, gq.device, {P});
  // VT_V4_RESIDENT_EXPERTS (Phase-2): device (moved) copies vs the ATS retag. Byte-identical.
  vt::Tensor gw = V4ResidentExpertW(gq, gate_w);
  vt::Tensor uw = V4ResidentExpertW(gq, up_w);
  deepseek_v4::MoeDevice()->moe_gate_up_swiglu(gq, o, a, gw, uw, eid, limit);  // NO sync
}

// The fused routed-MoE gate+up+SwiGLU epilogue is the resident-decode default
// (ds4-faithful). VT_V4_FUSED_MOE=0 falls back to the separate gate/up GEMMs +
// per-expert ClampedSwiGLU (bit-identical) for A/B measurement + rollback.
inline bool FusedMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_FUSED_MOE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// Brick 7: the fused per-head RMSNorm+RoPE kernel (NormRopeRowsKernel) is the
// resident-decode default (ds4-faithful; bit-identical). VT_V4_FUSED_ROPE=0 falls
// back to the separate {rms_norm_rows ; rope} launches for A/B measurement + rollback.
inline bool FusedRopeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_FUSED_ROPE");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// True iff the resident T=1 decode path can run: flag on, CUDA queue, keep-quant
// tower live (be.gguf != nullptr ⇒ `dsa_dense`, so every layer runs DENSE MLA and
// the compressor/indexer are OFF regardless of the config compress_ratios — the
// keep-quant decode geometry, AttentionBlock :658-660), the V4 device kernels
// linked, a KV cache bound, single token. (Contexts beyond the decode_attn
// shared-memory KV window are a named residual — the host path shares that bound.)
bool CanRunResidentDecode(const DeepseekV4Params& /*p*/, const V4Backend& be, int64_t T) {
  if (!ResidentDecodeEnabled() || T != 1) return false;
  if (be.q == nullptr || be.q->device.type == vt::DeviceType::kCPU) return false;
  if (be.gguf == nullptr || be.kv == nullptr) return false;
  if (!deepseek_v4::V4DeviceKernelsAvailable()) return false;
  return true;
}

// The resident T=1 decode step. Returns the [V] logits (host-readable after the one
// drain). Token-identical target to ForwardComposeImpl on the keep-quant dense path
// (the wired-in device RMSNorm/RoPE/combine are the characterized part-1 near-ties;
// argmax is robust, as it was for Bricks A/B).
std::vector<float> ForwardResidentDecodeGguf(const DeepseekV4HostWeights& hw,
                                             const DeepseekV4Params& p,
                                             const std::vector<int32_t>& token_ids,
                                             const std::vector<int32_t>& positions,
                                             const V4Backend& be) {
  const int64_t H = p.hidden_size, hc = p.hc_mult, nlayers = p.num_hidden_layers;
  const int64_t V = p.vocab_size, nh = p.num_attention_heads, hd = p.head_dim;
  const int64_t rope = p.qk_rope_head_dim, nope = hd - rope, qlr = p.q_lora_rank;
  const int64_t ne = p.n_routed_experts, topk = p.num_experts_per_tok, mi = p.moe_intermediate_size;
  const int64_t ng = p.o_groups, olr = p.o_lora_rank, ipg = nh * hd / ng, zdim = ng * olr;
  const float eps = p.rms_norm_eps, hc_eps = static_cast<float>(p.hc_eps);
  const float lim = static_cast<float>(p.swiglu_limit);
  const int64_t iters = p.hc_sinkhorn_iters;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const int32_t tok = token_ids[0];
  const int pos = static_cast<int>(positions[0]);
  VT_CHECK(tok >= 0 && tok < V, "resident decode: token id out of range");

  auto* MHC = deepseek_v4::MhcDevice();
  auto* DSA = deepseek_v4::DsaDevice();
  auto* MOE = deepseek_v4::MoeDevice();

  // ── persistent unified scratch (allocated ONCE; never resized → stable pointers
  //    for the deferred async chain). T=1 shapes.
  std::vector<float> x(static_cast<size_t>(H));
  std::vector<float> resA(static_cast<size_t>(hc * H)), resB(static_cast<size_t>(hc * H));
  std::vector<float> post_mix(static_cast<size_t>(hc)), res_mix(static_cast<size_t>(hc * hc));
  std::vector<float> pre_mix(static_cast<size_t>(hc)), mix_scratch(static_cast<size_t>((2 + hc) * hc + 1));
  std::vector<float> qa(static_cast<size_t>(qlr)), q(static_cast<size_t>(nh * hd));
  std::vector<float> kraw(static_cast<size_t>(hd)), o(static_cast<size_t>(nh * hd));
  std::vector<float> z(static_cast<size_t>(zdim));
  std::vector<float> gating(static_cast<size_t>(ne));
  std::vector<int32_t> eids(static_cast<size_t>(topk));
  std::vector<int64_t> in_tokens{static_cast<int64_t>(tok)};
  std::vector<float> weights(static_cast<size_t>(1 + topk));  // [0]=shared(1), [1..]=routed
  std::vector<float> gate_up_s(static_cast<size_t>(2 * mi)), act_s(static_cast<size_t>(mi));
  std::vector<float> gr(static_cast<size_t>(topk * mi)), ur(static_cast<size_t>(topk * mi));
  std::vector<float> gate_up_r(static_cast<size_t>(2 * mi));
  std::vector<float> adown(static_cast<size_t>(topk * mi));
  std::vector<float> eo(static_cast<size_t>((1 + topk) * H));
  std::vector<int> pos_buf(static_cast<size_t>(std::max<int64_t>(nh, 1)), pos);
  std::vector<float> hbuf(static_cast<size_t>(H)), logits(static_cast<size_t>(V));

  float* res_cur = resA.data();
  float* res_nxt = resB.data();

  // embed (host; the token hidden is the only host-written input, before any device op).
  for (int64_t h = 0; h < H; ++h) x[static_cast<size_t>(h)] = hw.embed[tok * H + h];

  // MHC-pre on a (hc*H) residual → writes layer_input(x), post_mix, res_mix; reads `residual`.
  const auto mhc_pre = [&](const DeepseekV4LayerHostWeights& L, bool attn) {
    const std::vector<float>& fn = attn ? L.hc_attn_fn : L.hc_ffn_fn;
    const std::vector<float>& sc = attn ? L.hc_attn_scale : L.hc_ffn_scale;
    const std::vector<float>& ba = attn ? L.hc_attn_base : L.hc_ffn_base;
    const std::vector<float>& nw = attn ? L.attn_norm_weight : L.ffn_norm_weight;
    MHC->pre_ip(*be.q, pre_mix.data(), post_mix.data(), res_mix.data(), x.data(),
                mix_scratch.data(), res_nxt, fn.data(), sc.data(), ba.data(), hc, H, eps, hc_eps,
                hc_eps, 2.0f, iters, nw.empty() ? nullptr : nw.data(), !nw.empty(), eps);
    std::swap(res_cur, res_nxt);  // res_cur := res_t (the residual this pre consumed)
  };

  bool have_residual = false;
  for (int64_t layer = 0; layer < nlayers; ++layer) {
    const DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(layer)];
    const DeepseekV4GgufLayerWeights& Lq = be.gguf->layers[static_cast<size_t>(layer)];
    const bool rope_comp = p.has_compressor(layer);
    const double rbase = rope_comp ? p.compress_rope_theta : p.rope_theta;
    const double rfs = (rope_comp && p.rope_scale_factor > 1.0) ? 1.0 / p.rope_scale_factor : 1.0;
    const double rext = (rope_comp && p.rope_scale_factor > 1.0) ? 1.0 : 0.0;

    // ── attn sub-block MHC pre (layer 0: broadcast x → [hc,H]; else fused post+pre).
    if (!have_residual) {
      for (int64_t i = 0; i < hc; ++i) AsyncCopyF(be, res_nxt + i * H, x.data(), H);
    } else {
      MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
    }
    mhc_pre(L, /*attn=*/true);
    have_residual = true;

    // ── 512-wide MLA attention (dense-causal, T=1).
    // Brick 12: the two A-projections share x → ONE paired launch (qa + kraw).
    GemmPairIntoKq(be, Lq.wq_a, Lq.wkv, x.data(), qa.data(), kraw.data(), qlr, hd, H);
    DSA->rms_norm_rows(*be.q, qa.data(), qa.data(), L.q_norm_weight.data(), 1, qlr, eps, true);
    GemmIntoKq(be, Lq.wq_b, qa.data(), q.data(), 1, nh * hd, qlr);
    if (FusedRopeEnabled()) {  // Brick 7: fused per-head q RMS-norm + fwd RoPE (bit-identical).
      DSA->norm_rope_rows(*be.q, q.data(), q.data(), nullptr, nh, hd, nope, rope, pos_buf.data(),
                          rbase, rfs, rext, p.rope_orig_ctx, p.rope_beta_fast, p.rope_beta_slow,
                          /*inverse=*/false, /*has_w=*/false, /*do_norm=*/true, eps);
    } else {
      DSA->rms_norm_rows(*be.q, q.data(), q.data(), nullptr, nh, hd, eps, false);  // per-head q-RMS
      DSA->rope(*be.q, q.data(), nh, hd, nope, rope, pos_buf.data(), rbase, rfs, rext,
                p.rope_orig_ctx, p.rope_beta_fast, p.rope_beta_slow, /*inverse=*/false);
    }
    // (kraw computed above by the paired A-projection launch — Brick 12.)
    // kv-norm + RoPE written DIRECTLY into the new KV-cache row (the resident append).
    std::vector<float>& lc = be.kv->deck[static_cast<size_t>(layer)];
    const int64_t kv_base = be.kv_base;
    VT_CHECK(static_cast<int64_t>(lc.size()) == kv_base * hd, "resident decode: kv cache out of sync");
    lc.resize(static_cast<size_t>((kv_base + 1) * hd));  // grow by one latent (host; stable this step)
    float* slot = lc.data() + kv_base * hd;
    if (FusedRopeEnabled()) {  // Brick 7: fused kv RMS-norm + fwd RoPE into the KV slot.
      DSA->norm_rope_rows(*be.q, slot, kraw.data(), L.kv_norm_weight.data(), 1, hd, nope, rope,
                          pos_buf.data(), rbase, rfs, rext, p.rope_orig_ctx, p.rope_beta_fast,
                          p.rope_beta_slow, /*inverse=*/false, /*has_w=*/true, /*do_norm=*/true, eps);
    } else {
      DSA->rms_norm_rows(*be.q, slot, kraw.data(), L.kv_norm_weight.data(), 1, hd, eps, true);
      DSA->rope(*be.q, slot, 1, hd, nope, rope, pos_buf.data(), rbase, rfs, rext, p.rope_orig_ctx,
                p.rope_beta_fast, p.rope_beta_slow, /*inverse=*/false);
    }
    DSA->decode_attn(*be.q, o.data(), q.data(), lc.data(), L.attn_sink.data(), nh, hd, kv_base, 1,
                     scale, /*no_sink=*/false);
    if (FusedRopeEnabled())  // Brick 7: parallelized inverse o-RoPE (no norm), bit-identical.
      DSA->norm_rope_rows(*be.q, o.data(), o.data(), nullptr, nh, hd, nope, rope, pos_buf.data(),
                          rbase, rfs, rext, p.rope_orig_ctx, p.rope_beta_fast, p.rope_beta_slow,
                          /*inverse=*/true, /*has_w=*/false, /*do_norm=*/false, eps);
    else
      DSA->rope(*be.q, o.data(), nh, hd, nope, rope, pos_buf.data(), rbase, rfs, rext,
                p.rope_orig_ctx, p.rope_beta_fast, p.rope_beta_slow, /*inverse=*/true);
    // grouped OUTPUT-LoRA: per-group wo_a slice into z, then wo_b → x (the attn out).
    OloraAIntoKq(be, Lq.wo_a, o.data(), z.data(), ng, olr, ipg);  // Brick 12: ONE block-diag launch
    GemmIntoKq(be, Lq.wo_b, z.data(), x.data(), 1, H, zdim);

    // ── ffn sub-block MHC fused post+pre.
    MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
    mhc_pre(L, /*attn=*/false);

    // ── DeepSeek-V4 MoE (shared + topk routed; resident routing).
    const bool cfg_hash = p.is_hash_layer(layer);
    // router gate — DEVICE (Brick D): the [ne,H] BF16 `ffn.gate.weight` × f32 x via
    // RouterGateKernel (sequential f32, exact bf16 upcast → bit-identical to the host
    // CPU MatmulBT). No drain, no host op — the resident step is now 100% device.
    // (A block-quant router gate, if a build ever ships one, stays the device GEMM.)
    if (vt::IsBlockQuant(Lq.moe_gate.dtype))
      GemmIntoKq(be, Lq.moe_gate, x.data(), gating.data(), 1, ne, H);
    else
      MOE->router_gate(*be.q, gating.data(), x.data(), Lq.moe_gate.bytes.data(), ne, H);
    const bool has_bias = !cfg_hash && !L.gate_bias.empty();
    MOE->route_ip(*be.q, eids.data(), weights.data() + 1, gating.data(), 1, ne, topk,
                  has_bias ? L.gate_bias.data() : nullptr, has_bias,
                  cfg_hash ? in_tokens.data() : nullptr, cfg_hash,
                  cfg_hash ? L.tid2eid.data() : nullptr, p.vocab_size, p.norm_topk_prob,
                  static_cast<float>(p.routed_scaling_factor));
    weights[0] = 1.0f;  // shared-expert combine weight (host write; device reads later)
    // shared expert (index 0 of eo).
    // Brick 12: shared-expert gate+up share x → ONE paired launch.
    GemmPairIntoKq(be, Lq.shared_gate, Lq.shared_up, x.data(), gate_up_s.data(),
                   gate_up_s.data() + mi, mi, mi, H);
    MOE->clamped_swiglu_ip(*be.q, act_s.data(), gate_up_s.data(), mi, lim, 1.0f, 0.0f);
    GemmIntoKq(be, Lq.shared_down, act_s.data(), eo.data(), 1, H, mi);
    // routed experts (grouped, eids resident): gate/up read the SHARED hidden x
    // (act_rows=1 ⇒ quantize x ONCE, broadcast across the topk experts — no xrep copy,
    // no per-expert re-quant of an identical row), swiglu, down.
    if (FusedMoeEnabled()) {
      // ds4 moe_gate_up_mid: ONE launch does gate+up+silu → adown (bit-identical).
      MoeGateUpSwiGLUInto(be, Lq.moe_gate_exps, Lq.moe_up_exps, x.data(), eids.data(), topk, mi, H,
                          lim, adown.data());
    } else {
      GemmGroupedInto(be, Lq.moe_gate_exps, x.data(), eids.data(), topk, mi, H, gr.data(), /*act_rows=*/1);
      GemmGroupedInto(be, Lq.moe_up_exps, x.data(), eids.data(), topk, mi, H, ur.data(), /*act_rows=*/1);
      for (int64_t j = 0; j < topk; ++j) {
        AsyncCopyF(be, gate_up_r.data(), gr.data() + j * mi, mi);        // gate half
        AsyncCopyF(be, gate_up_r.data() + mi, ur.data() + j * mi, mi);   // up half
        MOE->clamped_swiglu_ip(*be.q, adown.data() + j * mi, gate_up_r.data(), mi, lim, 1.0f, 0.0f);
      }
    }
    GemmGroupedInto(be, Lq.moe_down_exps, adown.data(), eids.data(), topk, H, mi, eo.data() + H);
    // combine: x := shared + Σ_j w_j·routed_j  (device FMA near-tie).
    MOE->moe_combine(*be.q, x.data(), eo.data(), weights.data(), 1 + topk, H);
  }

  // ── final MhcPost → hc_head collapse → final RMSNorm → lm_head.
  MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
  MHC->head_ip(*be.q, hbuf.data(), res_nxt, hw.hc_head_fn.data(), hw.hc_head_scale,
               hw.hc_head_base.data(), hc, H, eps, hc_eps);
  DSA->rms_norm_rows(*be.q, hbuf.data(), hbuf.data(), hw.final_norm_weight.data(), 1, H, eps, true);
  GemmIntoKq(be, be.gguf->lm_head, hbuf.data(), logits.data(), 1, V, H);
  DrainDevice(be);  // the ONE step-boundary drain — the host now reads logits for argmax
  return logits;
}

inline bool DecodeGraphEnabled() {
  // DEFAULT-ON (parity-enablers-ship-as-defaults): the decode CUDA graph is a MEASURED
  // 2.13x byte-exact decode win on GB10 (VT_V4_DECODE_GRAPH A/B: 6.0 -> 12.8 tok/s,
  // ids byte-identical), and the recorded 13.0 DeepSeek baseline already ran with it —
  // so the shipped default must match the benchmarked config. VT_V4_DECODE_GRAPH=0 opts
  // back out (same-binary A/B escape hatch); anything else (unset or =1) keeps it ON.
  static const bool on = [] {
    const char* e = std::getenv("VT_V4_DECODE_GRAPH");
    return e == nullptr || std::string(e) != "0";
  }();
  return on;
}

// ─── Brick D step 2: the DECODE CUDA GRAPH (the campaign payoff) ──────────────
// Capture the now-100%-device resident T=1 step into ONE graph and replay it —
// collapsing the ~1700 host launches/step into a single cudaGraphLaunch (the
// host-gap idle the eager resident measured at ~45% GPU-idle). The whole per-layer
// chain runs over PERSISTENT member buffers (never resized → stable addresses the
// captured graph bakes); the ONLY per-step-varying inputs (embed x, position,
// token, KV length) live in persistent buffers whose CONTENTS the driver refreshes
// before each replay. The growing KV is handled cudagraph-safely: fixed-capacity
// per-layer cache; the kv-norm+RoPE writes the new token's latent to a FIXED
// `deck_new[layer]` scratch; `decode_attn_g` attends `cache[0..len)` + `deck_new`
// with `len` read from a DEVICE buffer; and BETWEEN replays the driver async-copies
// `deck_new`→`cache[len]` on the stream (NOT inside the captured region) + advances
// `len`. cold→warm→captured: the cold step runs eager (grows the per-stream Q8_K
// GEMM scratch — cudagraph-safe grow-only — so capture does zero fresh allocation).
// [[cudagraph-capture-bakes-stack-addresses]]: EVERY captured input is a member
// buffer, never a stack temporary; verified BY TOKENS (replay == eager == host).
struct V4Graph {
  const DeepseekV4HostWeights* hw;
  const DeepseekV4Params* p;
  int64_t H, hc, nlayers, V, nh, hd, rope, nope, qlr, ne, topk, mi, ng, olr, ipg, zdim, iters,
      max_cap;
  float eps, hc_eps, lim, scale;
  // persistent unified scratch (allocated once; never resized → stable addresses).
  std::vector<float> x, resA, resB, post_mix, res_mix, pre_mix, mix_scratch, qa, qact, kraw, o, z,
      gating, weights, gate_up_s, act_s, gr, ur, gate_up_r, adown, eo, hbuf, logits;
  std::vector<int32_t> eids;
  std::vector<int64_t> in_tokens;
  std::vector<int> pos_buf, len_buf;                 // per-step inputs (device-read)
  std::vector<std::vector<float>> cache, deck_new;   // [layer]: fixed-cap KV + new-row scratch
  float* res_cur = nullptr;
  float* res_nxt = nullptr;
  int64_t kv_base = 0;
  // ENG-CUDAGRAPH-BREAK W5 (#1335): the instantiated graph, its handle ownership
  // and its release live in the SHARED SEAM instead of in a raw `void*` this
  // class destroyed by hand. `vt::BreakableGraph` releases every segment it
  // holds through `Backend::DestroyGraph`, which is the routing that lets
  // ENG-CUDAGRAPH-DEDUP (#1162) interpose at the backend later without editing
  // this file. `gstate` STAYS: it is this driver's cold/warm/captured ladder,
  // not a duplicate of `captured()` — the seam has no notion of the eager
  // warm-run that grows the pool before a capture may allocate nothing.
  vt::BreakableGraph graph;
  int gstate = 0;  // 0 cold (eager warm-run), 1 warm (capture+replay), 2 captured (replay)
  vt::Queue* qu = nullptr;

  V4Graph(const V4Backend& be, const DeepseekV4HostWeights& hw_, const DeepseekV4Params& pp)
      : hw(&hw_), p(&pp) {
    H = pp.hidden_size; hc = pp.hc_mult; nlayers = pp.num_hidden_layers; V = pp.vocab_size;
    nh = pp.num_attention_heads; hd = pp.head_dim; rope = pp.qk_rope_head_dim; nope = hd - rope;
    qlr = pp.q_lora_rank; ne = pp.n_routed_experts; topk = pp.num_experts_per_tok;
    mi = pp.moe_intermediate_size; ng = pp.o_groups; olr = pp.o_lora_rank; ipg = nh * hd / ng;
    zdim = ng * olr; iters = pp.hc_sinkhorn_iters; eps = pp.rms_norm_eps;
    hc_eps = static_cast<float>(pp.hc_eps); lim = static_cast<float>(pp.swiglu_limit);
    scale = 1.0f / std::sqrt(static_cast<float>(hd));
    qu = be.q;
    kv_base = be.kv_base;  // = prefill length (the graph takes over after prefill)
    // fixed capacity: the decode_attn_g shared-mem KV window (≤40 KiB ⇒ ≤10240 keys).
    max_cap = 4096;
    VT_CHECK(kv_base + 1 <= max_cap,
             "deepseek-v4 decode graph: prefill length exceeds the fixed KV capacity "
             "(long-context graph is a named residual)");
    x.assign(static_cast<size_t>(H), 0.0f);
    resA.assign(static_cast<size_t>(hc * H), 0.0f); resB.assign(static_cast<size_t>(hc * H), 0.0f);
    post_mix.assign(static_cast<size_t>(hc), 0.0f); res_mix.assign(static_cast<size_t>(hc * hc), 0.0f);
    pre_mix.assign(static_cast<size_t>(hc), 0.0f);
    mix_scratch.assign(static_cast<size_t>((2 + hc) * hc + 1), 0.0f);
    qa.assign(static_cast<size_t>(qlr), 0.0f); qact.assign(static_cast<size_t>(nh * hd), 0.0f);
    kraw.assign(static_cast<size_t>(hd), 0.0f); o.assign(static_cast<size_t>(nh * hd), 0.0f);
    z.assign(static_cast<size_t>(zdim), 0.0f); gating.assign(static_cast<size_t>(ne), 0.0f);
    weights.assign(static_cast<size_t>(1 + topk), 0.0f); weights[0] = 1.0f;  // shared weight (const)
    gate_up_s.assign(static_cast<size_t>(2 * mi), 0.0f); act_s.assign(static_cast<size_t>(mi), 0.0f);
    gr.assign(static_cast<size_t>(topk * mi), 0.0f); ur.assign(static_cast<size_t>(topk * mi), 0.0f);
    gate_up_r.assign(static_cast<size_t>(2 * mi), 0.0f);
    adown.assign(static_cast<size_t>(topk * mi), 0.0f);
    eo.assign(static_cast<size_t>((1 + topk) * H), 0.0f);
    eids.assign(static_cast<size_t>(topk), 0);
    in_tokens.assign(1, 0);
    pos_buf.assign(static_cast<size_t>(std::max<int64_t>(nh, 1)), 0);
    len_buf.assign(1, 0);
    hbuf.assign(static_cast<size_t>(H), 0.0f); logits.assign(static_cast<size_t>(V), 0.0f);
    cache.assign(static_cast<size_t>(nlayers), {});
    deck_new.assign(static_cast<size_t>(nlayers), {});
    for (int64_t l = 0; l < nlayers; ++l) {
      cache[static_cast<size_t>(l)].assign(static_cast<size_t>(max_cap * hd), 0.0f);
      deck_new[static_cast<size_t>(l)].assign(static_cast<size_t>(hd), 0.0f);
      // seed the fixed-cap cache with the prefill KV (be.kv->deck filled by ForwardComposeImpl).
      const std::vector<float>& pref = be.kv->deck[static_cast<size_t>(l)];
      VT_CHECK(static_cast<int64_t>(pref.size()) == kv_base * hd,
               "deepseek-v4 decode graph: prefill KV size mismatch");
      std::copy(pref.begin(), pref.end(), cache[static_cast<size_t>(l)].begin());
    }
  }
  // No destructor: `graph` releases its own segments through
  // `Backend::DestroyGraph`, so the hand-rolled one this replaced is gone.

  // The per-layer resident chain over the PERSISTENT buffers (the capture region).
  void RunChain(const V4Backend& be) {
    auto* MHC = deepseek_v4::MhcDevice();
    auto* DSA = deepseek_v4::DsaDevice();
    auto* MOE = deepseek_v4::MoeDevice();
    res_cur = resA.data();
    res_nxt = resB.data();
    const auto mhc_pre = [&](const DeepseekV4LayerHostWeights& L, bool attn) {
      const std::vector<float>& fn = attn ? L.hc_attn_fn : L.hc_ffn_fn;
      const std::vector<float>& sc = attn ? L.hc_attn_scale : L.hc_ffn_scale;
      const std::vector<float>& ba = attn ? L.hc_attn_base : L.hc_ffn_base;
      const std::vector<float>& nw = attn ? L.attn_norm_weight : L.ffn_norm_weight;
      MHC->pre_ip(*be.q, pre_mix.data(), post_mix.data(), res_mix.data(), x.data(),
                  mix_scratch.data(), res_nxt, fn.data(), sc.data(), ba.data(), hc, H, eps, hc_eps,
                  hc_eps, 2.0f, iters, nw.empty() ? nullptr : nw.data(), !nw.empty(), eps);
      std::swap(res_cur, res_nxt);
    };
    bool have_residual = false;
    for (int64_t layer = 0; layer < nlayers; ++layer) {
      const DeepseekV4LayerHostWeights& L = hw->layers[static_cast<size_t>(layer)];
      const DeepseekV4GgufLayerWeights& Lq = be.gguf->layers[static_cast<size_t>(layer)];
      const bool rope_comp = p->has_compressor(layer);
      const double rbase = rope_comp ? p->compress_rope_theta : p->rope_theta;
      const double rfs = (rope_comp && p->rope_scale_factor > 1.0) ? 1.0 / p->rope_scale_factor : 1.0;
      const double rext = (rope_comp && p->rope_scale_factor > 1.0) ? 1.0 : 0.0;
      if (!have_residual)
        for (int64_t i = 0; i < hc; ++i) AsyncCopyF(be, res_nxt + i * H, x.data(), H);
      else
        MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
      mhc_pre(L, /*attn=*/true);
      have_residual = true;
      // MLA attention (T=1) — GRAPH variant: kv → deck_new scratch, attn over cache+deck_new+len.
      // Brick 12: the two A-projections share x → ONE paired launch (qa + kraw).
      GemmPairIntoKq(be, Lq.wq_a, Lq.wkv, x.data(), qa.data(), kraw.data(), qlr, hd, H);
      DSA->rms_norm_rows(*be.q, qa.data(), qa.data(), L.q_norm_weight.data(), 1, qlr, eps, true);
      GemmIntoKq(be, Lq.wq_b, qa.data(), qact.data(), 1, nh * hd, qlr);
      if (FusedRopeEnabled()) {  // Brick 7: fused per-head q RMS-norm + fwd RoPE (bit-identical).
        DSA->norm_rope_rows(*be.q, qact.data(), qact.data(), nullptr, nh, hd, nope, rope,
                            pos_buf.data(), rbase, rfs, rext, p->rope_orig_ctx, p->rope_beta_fast,
                            p->rope_beta_slow, /*inverse=*/false, /*has_w=*/false, /*do_norm=*/true,
                            eps);
      } else {
        DSA->rms_norm_rows(*be.q, qact.data(), qact.data(), nullptr, nh, hd, eps, false);
        DSA->rope(*be.q, qact.data(), nh, hd, nope, rope, pos_buf.data(), rbase, rfs, rext,
                  p->rope_orig_ctx, p->rope_beta_fast, p->rope_beta_slow, /*inverse=*/false);
      }
      // (kraw computed above by the paired A-projection launch — Brick 12.)
      float* dn = deck_new[static_cast<size_t>(layer)].data();
      if (FusedRopeEnabled()) {  // Brick 7: fused kv RMS-norm + fwd RoPE into the deck_new slot.
        DSA->norm_rope_rows(*be.q, dn, kraw.data(), L.kv_norm_weight.data(), 1, hd, nope, rope,
                            pos_buf.data(), rbase, rfs, rext, p->rope_orig_ctx, p->rope_beta_fast,
                            p->rope_beta_slow, /*inverse=*/false, /*has_w=*/true, /*do_norm=*/true,
                            eps);
      } else {
        DSA->rms_norm_rows(*be.q, dn, kraw.data(), L.kv_norm_weight.data(), 1, hd, eps, true);
        DSA->rope(*be.q, dn, 1, hd, nope, rope, pos_buf.data(), rbase, rfs, rext, p->rope_orig_ctx,
                  p->rope_beta_fast, p->rope_beta_slow, /*inverse=*/false);
      }
      DSA->decode_attn_g(*be.q, o.data(), qact.data(), cache[static_cast<size_t>(layer)].data(), dn,
                         L.attn_sink.data(), nh, hd, len_buf.data(), max_cap, scale, /*no_sink=*/false);
      if (FusedRopeEnabled())  // Brick 7: parallelized inverse o-RoPE (no norm), bit-identical.
        DSA->norm_rope_rows(*be.q, o.data(), o.data(), nullptr, nh, hd, nope, rope, pos_buf.data(),
                            rbase, rfs, rext, p->rope_orig_ctx, p->rope_beta_fast, p->rope_beta_slow,
                            /*inverse=*/true, /*has_w=*/false, /*do_norm=*/false, eps);
      else
        DSA->rope(*be.q, o.data(), nh, hd, nope, rope, pos_buf.data(), rbase, rfs, rext,
                  p->rope_orig_ctx, p->rope_beta_fast, p->rope_beta_slow, /*inverse=*/true);
      OloraAIntoKq(be, Lq.wo_a, o.data(), z.data(), ng, olr, ipg);  // Brick 12: ONE block-diag launch
      GemmIntoKq(be, Lq.wo_b, z.data(), x.data(), 1, H, zdim);
      // ffn MHC post+pre.
      MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
      mhc_pre(L, /*attn=*/false);
      // MoE (device router gate + resident routing).
      const bool cfg_hash = p->is_hash_layer(layer);
      if (vt::IsBlockQuant(Lq.moe_gate.dtype))
        GemmIntoKq(be, Lq.moe_gate, x.data(), gating.data(), 1, ne, H);
      else
        MOE->router_gate(*be.q, gating.data(), x.data(), Lq.moe_gate.bytes.data(), ne, H);
      const bool has_bias = !cfg_hash && !L.gate_bias.empty();
      MOE->route_ip(*be.q, eids.data(), weights.data() + 1, gating.data(), 1, ne, topk,
                    has_bias ? L.gate_bias.data() : nullptr, has_bias,
                    cfg_hash ? in_tokens.data() : nullptr, cfg_hash,
                    cfg_hash ? L.tid2eid.data() : nullptr, p->vocab_size, p->norm_topk_prob,
                    static_cast<float>(p->routed_scaling_factor));
      // Brick 12: shared-expert gate+up share x → ONE paired launch.
      GemmPairIntoKq(be, Lq.shared_gate, Lq.shared_up, x.data(), gate_up_s.data(),
                     gate_up_s.data() + mi, mi, mi, H);
      MOE->clamped_swiglu_ip(*be.q, act_s.data(), gate_up_s.data(), mi, lim, 1.0f, 0.0f);
      GemmIntoKq(be, Lq.shared_down, act_s.data(), eo.data(), 1, H, mi);
      // gate/up: broadcast the shared hidden x (act_rows=1 ⇒ quantize once, no xrep copy).
      if (FusedMoeEnabled()) {
        MoeGateUpSwiGLUInto(be, Lq.moe_gate_exps, Lq.moe_up_exps, x.data(), eids.data(), topk, mi, H,
                            lim, adown.data());
      } else {
        GemmGroupedInto(be, Lq.moe_gate_exps, x.data(), eids.data(), topk, mi, H, gr.data(), /*act_rows=*/1);
        GemmGroupedInto(be, Lq.moe_up_exps, x.data(), eids.data(), topk, mi, H, ur.data(), /*act_rows=*/1);
        for (int64_t j = 0; j < topk; ++j) {
          AsyncCopyF(be, gate_up_r.data(), gr.data() + j * mi, mi);
          AsyncCopyF(be, gate_up_r.data() + mi, ur.data() + j * mi, mi);
          MOE->clamped_swiglu_ip(*be.q, adown.data() + j * mi, gate_up_r.data(), mi, lim, 1.0f, 0.0f);
        }
      }
      GemmGroupedInto(be, Lq.moe_down_exps, adown.data(), eids.data(), topk, H, mi, eo.data() + H);
      MOE->moe_combine(*be.q, x.data(), eo.data(), weights.data(), 1 + topk, H);
    }
    MHC->post_ip(*be.q, res_nxt, x.data(), res_cur, post_mix.data(), res_mix.data(), hc, H);
    MHC->head_ip(*be.q, hbuf.data(), res_nxt, hw->hc_head_fn.data(), hw->hc_head_scale,
                 hw->hc_head_base.data(), hc, H, eps, hc_eps);
    DSA->rms_norm_rows(*be.q, hbuf.data(), hbuf.data(), hw->final_norm_weight.data(), 1, H, eps, true);
    GemmIntoKq(be, be.gguf->lm_head, hbuf.data(), logits.data(), 1, V, H);
  }

  // One decode token: refresh the per-step inputs, run/replay the step, append this
  // token's deck to the fixed-cap cache (on-stream, between replays), read logits.
  std::vector<float> Step(const V4Backend& be, int32_t token, int32_t pos) {
    VT_CHECK(kv_base + 1 <= max_cap, "deepseek-v4 decode graph: KV capacity exceeded");
    for (int64_t h = 0; h < H; ++h) x[static_cast<size_t>(h)] = hw->embed[token * H + h];  // embed
    std::fill(pos_buf.begin(), pos_buf.end(), pos);
    in_tokens[0] = token;
    len_buf[0] = static_cast<int>(kv_base);
    vt::Backend& b = vt::GetBackend(be.q->device);
    if (gstate == 0) {          // cold: eager warm-run (grow the cudagraph-safe GEMM scratch)
      RunChain(be);
      gstate = 1;
    } else if (gstate == 1) {   // warm: capture the region once, then replay it
      // ENG-CUDAGRAPH-BREAK W5 (#1335): the capture is the SHARED SEAM's, not
      // this driver's hand-rolled `BeginCapture`/`EndCaptureGraph` pair. The
      // scope owns the segment, the handle, its release, the drain a mid-capture
      // throw needs, and the G3 counters.
      //
      // kFULL, INHERITED FROM W2 AND NOT RE-ARGUED. vLLM's v1 default
      // `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:63` @ pin
      // `5559679229`) is documented at `:630-632` as a FULL graph for DECODE
      // batches and a piecewise one for prefill and mixed batches, and
      // `decode_mode()` (`:65-66`) returns the full half. This is the T=1
      // resident decode step, so its capture is ONE segment with the attention
      // calls INSIDE it — byte-identical in shape to the region this replaces.
      {
        vt::GraphCaptureScope scope(b, *be.q, graph, vt::GraphCaptureMode::kFull);
        RunChain(be);
      }  // ~GraphCaptureScope closes the segment and files it on `graph`
      // NOT CAPTURED covers TWO states and they mean opposite things.
      //
      //   * FAILED (`capture_failed() == true`): `Backend::EndCaptureGraph`
      //     threw. Under stream capture NOTHING between `BeginCapture` and the
      //     throw executed — every kernel was RECORDED — so `logits` holds
      //     whatever the pool last left there, and returning it would hand this
      //     step uncomputed device memory as its logits: no fault, and a token
      //     gate cannot see it. It PROPAGATES, carrying the runtime's own
      //     exception, which is what the pre-W5 driver's unguarded
      //     `EndCaptureGraph` did.
      //   * INERT (`capture_failed() == false`): capture is unsupported here, or
      //     `VLLM_CPP_CUDAGRAPH=0`. The scope made no backend call, `RunChain`
      //     ran EAGERLY, and `logits` is a real result — so this step returns
      //     normally and the driver stays in `gstate == 1`, running eager every
      //     step rather than pretending to hold a graph.
      if (!graph.captured()) {
        if (graph.capture_failed()) {
          const std::exception_ptr err = graph.capture_error();
          graph.Reset();  // clear the failure with the graph it described
          if (err) std::rethrow_exception(err);
          VT_CHECK(false,
                   "deepseek-v4 decode graph: the capture was ABANDONED and its logits "
                   "were never computed; refusing to return uncaptured device memory");
        }
      } else {
        graph.Replay(*be.q);
        gstate = 2;
      }
    } else {                    // captured: one cudaGraphLaunch
      // Through the seam's container, never `Backend::ReplayGraph` directly: it
      // replays its segments in order (one, here, because this capture is kFull)
      // and owns the G3 replay counter.
      graph.Replay(*be.q);
    }
    // append this token's deck_new → cache[kv_base] (on the stream, AFTER the step's
    // graph produced deck_new, BEFORE the next replay reads it) — the growing KV.
    for (int64_t l = 0; l < nlayers; ++l)
      AsyncCopyF(be, cache[static_cast<size_t>(l)].data() + kv_base * hd,
                 deck_new[static_cast<size_t>(l)].data(), hd);
    kv_base++;
    DrainDevice(be);  // the one step-boundary drain: logits ready, appends complete
    return logits;
  }
};

}  // namespace

// ── the W7 composition, written ONCE and run on HOST refs OR the device kernels
//    (backend policy `be`). DeepseekV4ForwardHost binds host; ForwardDevice binds
//    the CUDA kernels through the seam. ───────────────────────────────────────
static std::vector<float> ForwardComposeImpl(const DeepseekV4HostWeights& hw,
                                             const DeepseekV4Params& p,
                                             const std::vector<int32_t>& token_ids,
                                             const std::vector<int32_t>& positions,
                                             const std::vector<int32_t>& logits_indices,
                                             V4Miswire miswire, V4ForwardTrace* trace,
                                             const V4Backend& be,
                                             std::vector<float>* mtp_residual_out = nullptr) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_mult;
  const int64_t nlayers = p.num_hidden_layers;
  const int64_t V = p.vocab_size;
  const float eps = p.rms_norm_eps;
  const float hc_eps = static_cast<float>(p.hc_eps);
  const int64_t iters = p.hc_sinkhorn_iters;

  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "positions/token_ids length mismatch");
  VT_CHECK(static_cast<int64_t>(hw.layers.size()) == nlayers, "host layer count mismatch");
  VT_CHECK(hc > 0 && H > 0 && nlayers > 0, "degenerate config");

  if (trace != nullptr) {
    trace->hc_mult = hc;
    trace->hidden = H;
    trace->num_tokens = T;
    trace->residual_stream_elems = T * hc * H;
    trace->layer_is_hash.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_hash_routed.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_is_indexer.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_indexer_selected.assign(static_cast<size_t>(nlayers), 0);
    trace->layer_compressor_ran.assign(static_cast<size_t>(nlayers), 0);
  }

  // embed lookup -> the [T,H] token hidden stream.
  std::vector<float> x(static_cast<size_t>(T) * H);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = token_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < V, "token id out of range");
    for (int64_t h = 0; h < H; ++h) x[t * H + h] = hw.embed[tok * H + h];
  }
  DumpAct("ours_embed", Slice(x, 0, H));  // t=0 embed plain [H] (coherence-debug #188)

  // MHC residual manifold [T,hc,H] + the per-token post/comb mixes.
  std::vector<float> residual(static_cast<size_t>(T) * hc * H, 0.0f);
  std::vector<float> post_mix(static_cast<size_t>(T) * hc, 0.0f);
  std::vector<float> res_mix(static_cast<size_t>(T) * hc * hc, 0.0f);
  bool have_residual = false;

  // Keep-quant weight source (GGUF): the big MLA/MoE/lm_head GEMMs read the
  // compressed `be.gguf` blocks; small tensors stay in the f32 `hw` tower. Null
  // on the safetensors/NVFP4 + tiny-synthetic host path (every GEMM reads `hw`).
  const bool kq_src = be.gguf != nullptr;
  if (kq_src)
    VT_CHECK(static_cast<int64_t>(be.gguf->layers.size()) == nlayers,
             "deepseek-v4 keep-quant: gguf layer count mismatch");

  for (int64_t layer = 0; layer < nlayers; ++layer) {
    const DeepseekV4LayerHostWeights& L = hw.layers[static_cast<size_t>(layer)];
    const DeepseekV4GgufLayerWeights* Lq =
        kq_src ? &be.gguf->layers[static_cast<size_t>(layer)] : nullptr;

    // ── attn sub-block MHC-pre: first layer BROADCAST-expands [T,H] -> [T,hc,H];
    //    subsequent layers fuse MhcPost(prev-ffn-out) + MhcPre(attn) (model.py:878-933).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t(static_cast<size_t>(hc) * H);
      if (!have_residual) {
        for (int64_t i = 0; i < hc; ++i)
          for (int64_t h = 0; h < H; ++h) res_t[i * H + h] = x[t * H + h];
      } else {
        res_t = DispMhcPost(be, Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                            Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc),
                            hc, H);
      }
      const MhcPreResult pre =
          DispMhcPre(be, res_t, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, hc, H, eps,
                     hc_eps, hc_eps, 2.0f, iters, L.attn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }
    have_residual = true;

    // ── 512-wide MLA attention (W3+W4).
    const std::vector<float> x_pre_attn = std::getenv("VT_DUMP_ACT") ? x : std::vector<float>{};
    x = AttentionBlock(L, Lq, p, x, positions, layer, miswire, trace, be);
    if (std::getenv("VT_DUMP_ACT") != nullptr) {
      double ra = 0, rp = 0;
      for (int64_t h = 0; h < H; ++h) { ra += x[h] * x[h]; rp += x_pre_attn[h] * x_pre_attn[h]; }
      std::fprintf(stderr, "[L%02lld] rms pre_attn=%.3f attn_out=%.3f",
                   static_cast<long long>(layer), std::sqrt(rp / H), std::sqrt(ra / H));
      char nm[64]; std::snprintf(nm, sizeof(nm), "ours_attnout_L%02lld", static_cast<long long>(layer));
      DumpAct(nm, Slice(x, 0, H));  // #188 per-sub-op diff: attention output [H]
      std::snprintf(nm, sizeof(nm), "ours_attncur_L%02lld", static_cast<long long>(layer));
      DumpAct(nm, Slice(x_pre_attn, 0, H));  // #188 attention INPUT (post attn MHC-pre)
    }

    // ── ffn sub-block MHC fused-post-pre = MhcPost(attn-out) + MhcPre(ffn).
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> res_t =
          DispMhcPost(be, Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                      Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc, H);
      const MhcPreResult pre =
          DispMhcPre(be, res_t, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, hc, H, eps, hc_eps,
                     hc_eps, 2.0f, iters, L.ffn_norm_weight, eps);
      for (int64_t i = 0; i < hc * H; ++i)
        residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
      for (int64_t i = 0; i < hc * hc; ++i)
        res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
      for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
    }

    // ── DeepSeek-V4 MoE (W6).
    const std::vector<float> x_pre_moe = std::getenv("VT_DUMP_ACT") ? x : std::vector<float>{};
    if (std::getenv("VT_DUMP_ACT") != nullptr) {  // #188: real moe_in (t=0) for the expert probe
      char nm[64]; std::snprintf(nm, sizeof(nm), "ours_moein_L%02lld", static_cast<long long>(layer));
      DumpAct(nm, Slice(x, 0, H));
    }
    x = MoeBlock(L, Lq, p, x, token_ids, layer, miswire, trace, be);
    if (std::getenv("VT_DUMP_ACT") != nullptr) {
      double rm = 0, rp = 0;
      for (int64_t h = 0; h < H; ++h) { rm += x[h] * x[h]; rp += x_pre_moe[h] * x_pre_moe[h]; }
      std::fprintf(stderr, " moe_in=%.3f moe_out=%.3f\n", std::sqrt(rp / H), std::sqrt(rm / H));
      char nm[64]; std::snprintf(nm, sizeof(nm), "ours_moeout_L%02lld", static_cast<long long>(layer));
      DumpAct(nm, Slice(x, 0, H));  // #188 per-sub-op diff: routed+shared MoE output [H]
    }

    // coherence-debug #188: dump the [hc,H] manifold state AFTER this layer (the
    // MoE output folded back through MhcPost, matching ds4's `cur` after
    // layer_forward_self_one). t=0 only (single-token localization run).
    if (std::getenv("VT_DUMP_ACT") != nullptr) {
      const std::vector<float> folded =
          MhcPost(Slice(x, 0, H), Slice(residual, 0, hc * H), Slice(post_mix, 0, hc),
                  Slice(res_mix, 0, hc * hc), hc, H);
      char nm[64];
      std::snprintf(nm, sizeof(nm), "ours_hc_%02lld_afterL%02lld",
                    static_cast<long long>(layer + 1), static_cast<long long>(layer));
      DumpAct(nm, folded);
    }
  }

  // final MhcPost(last-ffn-out) -> hc_head collapse -> norm -> lm_head (model.py:1128-1146).
  // `all_res` captures the PRE-hc_head MHC residual stream [T,hc,H] — the exact
  // state DeepSeek-V4's MTP head consumes as `previous_hidden_states` (nvidia/mtp.py:
  // 139-141: previous_hidden_states.view(-1, hc_mult, H)). Filled only when the
  // caller asks (mtp_residual_out != nullptr — the self-spec residual stash).
  std::vector<float> hidden(static_cast<size_t>(T) * H);
  std::vector<float> all_res;
  if (mtp_residual_out != nullptr)
    all_res.resize(static_cast<size_t>(T) * hc * H);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> res_t;
    if (miswire == V4Miswire::kSkipFinalMhcPost) {
      res_t = Slice(residual, t * hc * H, hc * H);  // skip the fold (RED-first)
    } else {
      res_t = MhcPost(Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                      Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc,
                      H);
    }
    if (mtp_residual_out != nullptr)
      for (int64_t i = 0; i < hc * H; ++i)
        all_res[t * hc * H + i] = res_t[static_cast<size_t>(i)];
    std::vector<float> h = DispHcHead(be, res_t, hw.hc_head_fn, hw.hc_head_scale,
                                      hw.hc_head_base, hc, H, eps, hc_eps);
    h = RmsNorm(h, hw.final_norm_weight, eps);
    for (int64_t d = 0; d < H; ++d) hidden[t * H + d] = h[static_cast<size_t>(d)];
  }

  // gather the requested rows (all rows if logits_indices empty) then project
  // through the lm_head — keep-quant (GGUF) or f32 (host) via the same Gemm.
  std::vector<int32_t> rows = logits_indices;
  if (rows.empty()) {
    rows.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) rows[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  const int64_t R = static_cast<int64_t>(rows.size());
  std::vector<float> hsel(static_cast<size_t>(R) * H);
  if (mtp_residual_out != nullptr)
    mtp_residual_out->resize(static_cast<size_t>(R) * hc * H);
  for (int64_t ri = 0; ri < R; ++ri) {
    const int64_t r = rows[static_cast<size_t>(ri)];
    VT_CHECK(r >= 0 && r < T, "logits index out of range");
    for (int64_t d = 0; d < H; ++d) hsel[ri * H + d] = hidden[r * H + d];
    if (mtp_residual_out != nullptr)
      for (int64_t i = 0; i < hc * H; ++i)
        (*mtp_residual_out)[ri * hc * H + i] = all_res[r * hc * H + i];
  }
  const OwnedTensor* lmq = be.gguf != nullptr ? &be.gguf->lm_head : nullptr;
  return Gemm(be, lmq, hw.lm_head, hsel, R, V, H);
}

// Public host oracle: the composition on the portable host references.
std::vector<float> DeepseekV4ForwardHost(const DeepseekV4HostWeights& hw,
                                         const DeepseekV4Params& p,
                                         const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const std::vector<int32_t>& logits_indices,
                                         V4Miswire miswire, V4ForwardTrace* trace) {
  return ForwardComposeImpl(hw, p, token_ids, positions, logits_indices, miswire, trace,
                            V4Backend{/*device=*/false, /*q=*/nullptr, /*gguf=*/nullptr});
}

// ─── DeepSeek-V4 MTP self-speculative draft head (host oracle, W1) ────────────
// The target's PRE-hc_head MHC residual stream [R, hc*H] for the requested rows —
// the exact state the MTP head consumes as `previous_hidden_states`
// (nvidia/mtp.py:139-141). Reuses the host oracle with the residual-capture arm.
std::vector<float> DeepseekV4TargetMtpResidualHost(
    const DeepseekV4HostWeights& hw, const DeepseekV4Params& p,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int32_t>& logits_indices) {
  std::vector<float> res;
  (void)ForwardComposeImpl(hw, p, token_ids, positions, logits_indices, V4Miswire::kNone,
                           /*trace=*/nullptr,
                           V4Backend{/*device=*/false, /*q=*/nullptr, /*gguf=*/nullptr},
                           /*mtp_residual_out=*/&res);
  return res;
}

// The MTP DRAFT FORWARD + compute_logits — 1:1 with nvidia/mtp.py:128-258,
// reusing the DS4 host composition helpers. The nextn `mtp_block` is a full V4
// decoder layer at layer index `num_hidden_layers` (dense MLA + learned-gate MoE:
// compress_ratio(43)==0, is_hash_layer(43)==false), so it takes the plain dense
// AttentionBlock / learned-gate MoeBlock path.
std::vector<float> DeepseekV4MtpDraftLogitsHost(
    const DeepseekV4MtpHostWeights& mw, const DeepseekV4HostWeights& target,
    const DeepseekV4Params& p, const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions, const std::vector<float>& previous_hidden,
    const std::vector<int32_t>& logits_indices, V4MtpMiswire miswire) {
  const int64_t T = static_cast<int64_t>(input_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_mult;
  const int64_t V = p.vocab_size;
  const int64_t mtp_layer = p.num_hidden_layers;  // the nextn block's layer index
  const float eps = p.rms_norm_eps;
  const float hc_eps = static_cast<float>(p.hc_eps);
  const int64_t iters = p.hc_sinkhorn_iters;
  VT_CHECK(T > 0 && hc > 0 && H > 0, "deepseek-v4 mtp: degenerate shape");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "deepseek-v4 mtp: positions/input_ids length mismatch");
  VT_CHECK(static_cast<int64_t>(previous_hidden.size()) == T * hc * H,
           "deepseek-v4 mtp: previous_hidden must be [T, hc*H]");

  const V4Backend be{/*device=*/false, /*q=*/nullptr, /*gguf=*/nullptr};

  // 1. inputs_embeds = embed(input_ids); mask position 0 -> 0 (nvidia/mtp.py:135);
  //    then enorm (:136).
  std::vector<float> emb(static_cast<size_t>(T) * H);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t tok = input_ids[static_cast<size_t>(t)];
    VT_CHECK(tok >= 0 && tok < V, "deepseek-v4 mtp: token id out of range");
    const bool mask0 = positions[static_cast<size_t>(t)] == 0;
    for (int64_t h = 0; h < H; ++h)
      emb[t * H + h] = mask0 ? 0.0f : target.embed[tok * H + h];
    const std::vector<float> n = RmsNorm(Slice(emb, t * H, H), mw.enorm_weight, eps);
    for (int64_t h = 0; h < H; ++h) emb[t * H + h] = n[static_cast<size_t>(h)];
  }

  // 2. prev = hnorm(previous_hidden.view(T,hc,H)) per hc-stream row (:137).
  std::vector<float> prev = previous_hidden;  // [T,hc,H]
  if (miswire != V4MtpMiswire::kNoHnorm) {
    for (int64_t t = 0; t < T; ++t)
      for (int64_t i = 0; i < hc; ++i) {
        const std::vector<float> n =
            RmsNorm(Slice(prev, t * hc * H + i * H, H), mw.hnorm_weight, eps);
        for (int64_t h = 0; h < H; ++h) prev[t * hc * H + i * H + h] = n[static_cast<size_t>(h)];
      }
  }

  // 3. hidden[T,hc,H] = h_proj(prev) + e_proj(emb).unsqueeze(-2) (:139-141).
  const std::vector<float> e_out = Gemm(be, nullptr, mw.e_proj, emb, T, H, H);       // [T,H]
  const std::vector<float> h_out = Gemm(be, nullptr, mw.h_proj, prev, T * hc, H, H);  // [T*hc,H]
  std::vector<float> hidden(static_cast<size_t>(T) * hc * H);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t i = 0; i < hc; ++i)
      for (int64_t h = 0; h < H; ++h)
        hidden[t * hc * H + i * H + h] =
            (miswire == V4MtpMiswire::kSkipEhProj)
                ? emb[t * H + h]  // RED-first: drop the eh-lift (raw embed only)
                : h_out[(t * hc + i) * H + h] + e_out[t * H + h];

  // 4. mtp_block = ONE V4 decoder layer over the [T,hc,H] residual stream (the
  //    eh-lift is already hc-wide, so no first-layer broadcast). Mirrors
  //    ForwardComposeImpl:1654-1735 for a single layer.
  const DeepseekV4LayerHostWeights& L = mw.mtp_block;
  std::vector<float> residual = hidden;
  std::vector<float> post_mix(static_cast<size_t>(T) * hc, 0.0f);
  std::vector<float> res_mix(static_cast<size_t>(T) * hc * hc, 0.0f);
  std::vector<float> x(static_cast<size_t>(T) * H);
  // attn sub-block MHC-pre.
  for (int64_t t = 0; t < T; ++t) {
    const std::vector<float> res_t = Slice(residual, t * hc * H, hc * H);
    const MhcPreResult pre =
        DispMhcPre(be, res_t, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, hc, H, eps, hc_eps,
                   hc_eps, 2.0f, iters, L.attn_norm_weight, eps);
    for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
    for (int64_t i = 0; i < hc * hc; ++i)
      res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
    for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
  }
  x = AttentionBlock(L, /*Lq=*/nullptr, p, x, positions, mtp_layer, V4Miswire::kNone,
                     /*trace=*/nullptr, be);
  // ffn sub-block MhcPost + MhcPre.
  for (int64_t t = 0; t < T; ++t) {
    std::vector<float> res_t =
        DispMhcPost(be, Slice(x, t * H, H), Slice(residual, t * hc * H, hc * H),
                    Slice(post_mix, t * hc, hc), Slice(res_mix, t * hc * hc, hc * hc), hc, H);
    const MhcPreResult pre =
        DispMhcPre(be, res_t, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, hc, H, eps, hc_eps,
                   hc_eps, 2.0f, iters, L.ffn_norm_weight, eps);
    for (int64_t i = 0; i < hc * H; ++i)
      residual[t * hc * H + i] = res_t[static_cast<size_t>(i)];
    for (int64_t i = 0; i < hc; ++i) post_mix[t * hc + i] = pre.post_mix[static_cast<size_t>(i)];
    for (int64_t i = 0; i < hc * hc; ++i)
      res_mix[t * hc * hc + i] = pre.comb_mix[static_cast<size_t>(i)];
    for (int64_t h = 0; h < H; ++h) x[t * H + h] = pre.layer_input[static_cast<size_t>(h)];
  }
  x = MoeBlock(L, /*Lq=*/nullptr, p, x, input_ids, mtp_layer, V4Miswire::kNone,
               /*trace=*/nullptr, be);

  // 5. compute_logits: final MhcPost -> mw.hc_head collapse -> shared norm -> lm_head
  //    (nvidia/mtp.py:158-170 + :231-258). Gather the requested rows.
  std::vector<int32_t> rows = logits_indices;
  if (rows.empty()) {
    rows.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) rows[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  const int64_t R = static_cast<int64_t>(rows.size());
  std::vector<float> hsel(static_cast<size_t>(R) * H);
  for (int64_t ri = 0; ri < R; ++ri) {
    const int64_t r = rows[static_cast<size_t>(ri)];
    VT_CHECK(r >= 0 && r < T, "deepseek-v4 mtp: logits index out of range");
    const std::vector<float> res_t =
        MhcPost(Slice(x, r * H, H), Slice(residual, r * hc * H, hc * H),
                Slice(post_mix, r * hc, hc), Slice(res_mix, r * hc * hc, hc * hc), hc, H);
    std::vector<float> h;
    if (miswire == V4MtpMiswire::kSkipHcHead) {
      // RED-first: naive mean collapse instead of the learned hc_head.
      h.assign(static_cast<size_t>(H), 0.0f);
      for (int64_t i = 0; i < hc; ++i)
        for (int64_t d = 0; d < H; ++d)
          h[static_cast<size_t>(d)] += res_t[i * H + d] / static_cast<float>(hc);
    } else {
      h = DispHcHead(be, res_t, mw.hc_head_fn, mw.hc_head_scale, mw.hc_head_base, hc, H, eps,
                     hc_eps);
    }
    h = RmsNorm(h, mw.shared_norm_weight, eps);
    for (int64_t d = 0; d < H; ++d) hsel[ri * H + d] = h[static_cast<size_t>(d)];
  }
  return Gemm(be, nullptr, mw.lm_head, hsel, R, V, H);
}

// W2C — the GGUF keep-quant forward. The SAME composition as the host oracle, but
// the big MLA/MoE/lm_head GEMMs consume the COMPRESSED `weights.gguf` blocks in
// place via vt::MatmulBT -> kMatmulBTQuant (no per-layer f32 tower). The small
// tensors (norms/sinks/MHC/DSA mixing/ape/hash/embed) come from the SMALL
// `weights.host` tower the GGUF loader still dequants. Requires a queue (the CPU
// quant GEMM consumer).
std::vector<float> DeepseekV4ForwardGguf(const DeepseekV4Weights& weights,
                                         vt::Queue& queue,
                                         const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const std::vector<int32_t>& logits_indices,
                                         V4Miswire miswire, V4ForwardTrace* trace) {
  VT_CHECK(weights.has_gguf_weights,
           "DeepseekV4ForwardGguf: no keep-quant tower (call LoadDeepseekV4FromGguf)");
  VT_CHECK(weights.has_host_weights,
           "DeepseekV4ForwardGguf: the small f32 host tower (norms/embed/mixing) is "
           "absent");
  V4Backend be{/*device=*/false, /*q=*/&queue, /*gguf=*/&weights.gguf};
  be.grouped_moe = GroupedMoeEnabled();
  return ForwardComposeImpl(weights.host, weights.params, token_ids, positions, logits_indices,
                            miswire, trace, be);
}

// Stage 1 — incremental-decode variant. Same keep-quant composition, but binds a
// KV cache: this call's T tokens append their per-layer `deck` latent to the cache
// and the new queries attend over the full cached KV (global positions
// 0..cache.len+T-1). Prefill = first call (cache.len==0, all prompt tokens);
// decode = later calls (one new token, positions={cache.len}). Token-identical to
// DeepseekV4ForwardGguf run over the growing context (pure equivalence).
std::vector<float> DeepseekV4ForwardGgufCached(const DeepseekV4Weights& weights,
                                               vt::Queue& queue,
                                               DeepseekV4KvCache& cache,
                                               const std::vector<int32_t>& token_ids,
                                               const std::vector<int32_t>& positions,
                                               const std::vector<int32_t>& logits_indices) {
  VT_CHECK(weights.has_gguf_weights,
           "DeepseekV4ForwardGgufCached: no keep-quant tower (call LoadDeepseekV4FromGguf)");
  VT_CHECK(weights.has_host_weights,
           "DeepseekV4ForwardGgufCached: the small f32 host tower is absent");
  const int64_t nlayers = weights.params.num_hidden_layers;
  const int64_t hd = weights.params.head_dim;
  if (cache.deck.empty()) cache.Reset(nlayers, hd);  // lazy init on first call
  VT_CHECK(static_cast<int64_t>(cache.deck.size()) == nlayers && cache.head_dim == hd,
           "DeepseekV4ForwardGgufCached: cache not sized for this model");
  const int64_t T = static_cast<int64_t>(token_ids.size());
  V4Backend be{/*device=*/false, /*q=*/&queue, /*gguf=*/&weights.gguf};
  be.kv = &cache;
  be.kv_base = cache.len;  // same base for every layer this call
  be.grouped_moe = GroupedMoeEnabled();
  // Brick C part 2 / Brick D: the device-resident T=1 decode chain (no per-op sync).
  // Requires logits over the single new row (the decode contract).
  const bool single_row =
      logits_indices.empty() || (logits_indices.size() == 1 && logits_indices[0] == 0);
  const bool resident_ok = CanRunResidentDecode(weights.params, be, T) && single_row;

  // Brick D (VT_V4_DECODE_GRAPH=1): after prefill (kv_base>0), drive the resident
  // step through the captured decode CUDA graph (one cudaGraphLaunch/step). The
  // prefill step (kv_base==0, T>1) still runs the host ForwardComposeImpl, filling
  // cache.deck the graph seeds from. Default OFF; falls back to the eager resident /
  // host path otherwise.
  if (resident_ok && DecodeGraphEnabled() && cache.len > 0) {
    if (!cache.decode_graph) {
      cache.decode_graph = std::shared_ptr<void>(
          new V4Graph(be, weights.host, weights.params),
          [](void* g) { delete static_cast<V4Graph*>(g); });
    }
    auto* g = static_cast<V4Graph*>(cache.decode_graph.get());
    std::vector<float> logits = g->Step(be, token_ids[0], positions[0]);
    cache.len += T;
    return logits;
  }

  std::vector<float> logits =
      resident_ok
          ? ForwardResidentDecodeGguf(weights.host, weights.params, token_ids, positions, be)
          : ForwardComposeImpl(weights.host, weights.params, token_ids, positions, logits_indices,
                               V4Miswire::kNone, /*trace=*/nullptr, be);
  cache.len += T;  // all layers appended their T decks
  return logits;
}

// Stage-2 profiling accessors (env `VT_V4_PROF`; see prof:: above). The driver
// resets before a step and reads after: host-glue = step_time - gemm - sync.
void DeepseekV4ProfReset() {
  prof::g_gemm_s = 0.0;
  prof::g_sync_s = 0.0;
}
double DeepseekV4ProfGemmSeconds() { return prof::g_gemm_s; }
double DeepseekV4ProfSyncSeconds() { return prof::g_sync_s; }

std::vector<float> detail::DeepseekV4ExpertProbeInput(int64_t size,
                                                       float frequency) {
  if (size <= 0) return {};
  std::vector<float> values(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; ++i) {
    values[static_cast<size_t>(i)] =
        0.5f * std::sin(frequency * static_cast<float>(i + 1));
  }
  return values;
}

void DeepseekV4QHeadRmsNormInplace(std::vector<float>& q, int64_t n_head,
                                   int64_t head_dim, float eps) {
  for (int64_t h = 0; h < n_head; ++h) {
    float* head = q.data() + h * head_dim;
    double ss = 0.0;
    for (int64_t i = 0; i < head_dim; ++i) ss += static_cast<double>(head[i]) * head[i];
    const float r =
        1.0f / std::sqrt(static_cast<float>(ss / static_cast<double>(head_dim)) + eps);
    for (int64_t i = 0; i < head_dim; ++i) head[i] *= r;
  }
}

// Coherence-debug #188 Phase-2 DISCRIMINATOR. For each listed routed expert of `layer`,
// compute each projection TWO ways from the SAME keep-quant blocks:
//   (A) the kMatmulBTQuant path (GemmRowSlice, what the forward uses), and
//   (B) dequant those exact block bytes to f32 (vt::BlockToFloat, the loader's own
//       decoder) then a plain f64-accumulated GEMM.
// A != B  => the vec_dot / block-decode NUMERICS are wrong (per quant type: gate/up
//            are IQ2_XXS, down is Q2_K — so it names WHICH type).
// A == B but the rms is grossly larger than a clean expert => the GemmRowSlice ROW
//            OFFSET pulls the wrong rows out of the stacked [E*out,K] tensor.
// Prints to stderr; pure diagnostic (no state change).
void DeepseekV4ExpertProbe(const DeepseekV4Weights& weights, vt::Queue& queue,
                           int64_t layer, const std::vector<int64_t>& experts) {
  const DeepseekV4Params& p = weights.params;
  VT_CHECK(weights.has_gguf_weights && layer >= 0 &&
               layer < static_cast<int64_t>(weights.gguf.layers.size()),
           "DeepseekV4ExpertProbe: bad layer / no keep-quant tower");
  const DeepseekV4GgufLayerWeights& Lq = weights.gguf.layers[static_cast<size_t>(layer)];
  const int64_t H = p.hidden_size, mi = p.moe_intermediate_size;
  const V4Backend be{/*device=*/false, /*q=*/&queue, /*gguf=*/&weights.gguf};
  std::vector<float> xh = detail::DeepseekV4ExpertProbeInput(H, 0.017f);
  std::vector<float> xm = detail::DeepseekV4ExpertProbeInput(mi, 0.013f);
  // If a real moe_in dump exists for this layer (VT_DUMP_ACT), use it — the smooth
  // synthetic input does NOT reproduce the forward's explosion; the real hidden does.
  if (const char* dir = std::getenv("VT_DUMP_ACT")) {
    const std::string path =
        std::string(dir) + "/ours_moein_L" + (layer < 10 ? "0" : "") + std::to_string(layer) + ".bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f != nullptr) {
      const size_t got = std::fread(xh.data(), sizeof(float), static_cast<size_t>(H), f);
      std::fclose(f);
      double r = 0; for (float v : xh) r += (double)v * v;
      std::fprintf(stderr, "[probe] using REAL moe_in L%lld (%zu floats, rms=%.4f, max|x|=%.3f)\n",
                   static_cast<long long>(layer), got, std::sqrt(r / H),
                   [&]{ double m = 0; for (float v : xh) m = std::max(m, (double)std::fabs(v)); return m; }());
    }
  }
  auto ref = [&](const OwnedTensor& w, int64_t N, int64_t K, int64_t roff,
                 const std::vector<float>& x) {
    const size_t rb = vt::RowSizeBytes(w.dtype, K);
    std::vector<float> wf(static_cast<size_t>(N) * K);
    vt::cpu::BlockToFloat(w.dtype)(w.bytes.data() + static_cast<size_t>(roff) * rb, wf.data(), N * K);
    std::vector<float> out(static_cast<size_t>(N));
    for (int64_t n = 0; n < N; ++n) {
      double a = 0;
      for (int64_t k = 0; k < K; ++k) a += static_cast<double>(wf[n * K + k]) * x[static_cast<size_t>(k)];
      out[static_cast<size_t>(n)] = static_cast<float>(a);
    }
    return out;
  };
  auto rms = [](const std::vector<float>& v) {
    double s = 0; for (float x : v) s += static_cast<double>(x) * x; return std::sqrt(s / v.size());
  };
  auto rel = [](const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { double e = a[i] - b[i]; n += e * e; d += static_cast<double>(b[i]) * b[i]; }
    return std::sqrt(n / (d + 1e-30));
  };
  // GATE-GEMM cross-check: feed ds4's EXACT router input to OUR gate GEMM. If our
  // logits then match ds4's, our gate GEMM is correct and the divergence is an
  // upstream (diverged-input) effect; if not, the gate GEMM itself is the bug.
  if (const char* dir = std::getenv("VT_DUMP_ACT")) {
    const std::string ip = std::string(dir) + "/ds4_routerin_L34.bin";
    const std::string gp = std::string(dir) + "/ds4_gating_L34.bin";
    std::FILE* fi = std::fopen(ip.c_str(), "rb");
    if (fi != nullptr && layer == 34) {
      std::vector<float> din(static_cast<size_t>(H));
      if (std::fread(din.data(), sizeof(float), static_cast<size_t>(H), fi) != static_cast<size_t>(H)) { std::fclose(fi); return; }
      std::fclose(fi);
      const int64_t ne = p.n_routed_experts;
      const std::vector<float> myg = Gemm(be, &Lq.moe_gate, {}, din, 1, ne, H);
      double dr = 0; for (float v : din) dr += (double)v * v;
      std::fprintf(stderr, "[gate-xcheck] on ds4's router input (rms=%.4f): OUR logit[33]=%.4f logit[233]=%.4f\n",
                   std::sqrt(dr / H), myg[33], myg[233]);
      std::FILE* fg = std::fopen(gp.c_str(), "rb");
      if (fg != nullptr) {
        std::vector<float> dg(static_cast<size_t>(ne));
        const size_t ngot = std::fread(dg.data(), sizeof(float), static_cast<size_t>(ne), fg);
        std::fclose(fg);
        if (static_cast<int64_t>(ngot) == ne) {
        std::fprintf(stderr, "[gate-xcheck] ds4 logit[33]=%.4f logit[233]=%.4f ; rel-L2(our_gate(ds4_in), ds4_gate)=%.4f\n",
                     dg[33], dg[233], rel(myg, dg));
        }
      }
    }
  }
  std::fprintf(stderr, "== expert probe layer %lld : A=kMatmulBTQuant B=dequant(BlockToFloat)+GEMM ; gate/up=IQ2_XXS down=Q2_K ==\n",
               static_cast<long long>(layer));
  const float lim = static_cast<float>(p.swiglu_limit);
  // full expert (gate->clamped-SwiGLU->down) on input `xin`, projections via `proj`.
  auto full_expert = [&](const std::vector<float>& g, const std::vector<float>& u,
                         auto&& down_proj) {
    std::vector<float> gate_up(static_cast<size_t>(2) * mi);
    for (int64_t r = 0; r < mi; ++r) { gate_up[r] = g[r]; gate_up[mi + r] = u[r]; }
    const std::vector<float> act = DispClampedSwiGLU(be, gate_up, mi, lim, 1.0f, 0.0f);
    return std::make_pair(down_proj(act), rms(act));
  };
  for (int64_t e : experts) {
    const std::vector<float> gA = GemmRowSlice(be, Lq.moe_gate_exps, xh, 1, mi, H, e * mi);
    const std::vector<float> gB = ref(Lq.moe_gate_exps, mi, H, e * mi, xh);
    const std::vector<float> uA = GemmRowSlice(be, Lq.moe_up_exps, xh, 1, mi, H, e * mi);
    const std::vector<float> uB = ref(Lq.moe_up_exps, mi, H, e * mi, xh);
    // full expert output, both ways (the actual thing that exploded in the forward).
    auto [oA, actA] = full_expert(gA, uA, [&](const std::vector<float>& act) {
      return GemmRowSlice(be, Lq.moe_down_exps, act, 1, H, mi, e * H); });
    auto [oB, actB] = full_expert(gB, uB, [&](const std::vector<float>& act) {
      return ref(Lq.moe_down_exps, H, mi, e * H, act); });
    std::fprintf(stderr,
                 "e%03lld gate[A=%8.3f rel=%.4f] up[A=%8.3f rel=%.4f] act[A=%9.3f B=%9.3f] EXPERT_OUT[A=%9.3f B=%9.3f rel=%.4f]\n",
                 static_cast<long long>(e), rms(gA), rel(gA, gB), rms(uA), rel(uA, uB),
                 actA, actB, rms(oA), rms(oB), rel(oA, oB));
  }
}

std::vector<float> DeepseekV4Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;
  // GGUF source: consume the keep-quant tower (memory-bounded — no ~1 TiB f32
  // tower). Safetensors/NVFP4 + the tiny-synthetic gate: the f32 host oracle.
  if (weights.has_gguf_weights) {
    return DeepseekV4ForwardGguf(weights, queue, token_ids, positions, logits_indices);
  }
  (void)queue;
  VT_CHECK(weights.has_host_weights, kHostPending);
  return DeepseekV4ForwardHost(weights.host, weights.params, token_ids, positions,
                               logits_indices);
}

// FRAMEWORK-CONFORMANCE (device-resident logits): wrap the composed
// [rows, vocab] f32 logits as the runner's ForwardLogits, DEVICE-RESIDENT on a
// CUDA queue — the shared-framework contract, mirroring qwen3_moe.cpp
// WrapDeviceLogits (:216). When on_device() the runner samples argmax /
// temperature / top-k/top-p STRAIGHT off `device_tensor` with NO full-[rows,vocab]
// D2H (runner.cpp sample path (A)); only the sampled token ids cross to host.
//
// DeepSeek-V4's keep-quant forward is GB10-UNIFIED by construction: every GEMM
// reads/writes coherent unified-memory views in place (the same property that lets
// the keep-quant weight blocks be read mmap'd + the CLI read the logits on the host
// after ONE drain), and ForwardComposeImpl already drained the lm_head GEMM before
// returning `flat` (TimedMatmul syncs on the non-deferred device path). So the
// composed buffer is already device-addressable + complete; ownership moves into a
// shared_ptr the runner holds across execute_model -> sample_tokens, then returns
// (no per-step alloc/free — the buffer is the vector's own storage). On a CPU queue
// there is no device to sample from, so the logits stay on the byte-identical
// `.host` path (the portable host oracle + every CPU gate).
static ForwardLogits WrapV4DeviceLogits(std::vector<float>&& flat, int64_t rows,
                                        int64_t vocab, vt::Queue& queue) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  if (queue.device.type == vt::DeviceType::kCPU) {
    fl.host = std::move(flat);
    return fl;
  }
  auto buf = std::make_shared<std::vector<float>>(std::move(flat));
  fl.device_tensor = vt::Tensor::Contiguous(buf->data(), vt::DType::kF32,
                                            queue.device, {rows, vocab});
  fl.device_storage = std::move(buf);  // shared_ptr<vector<float>> -> shared_ptr<void>
  return fl;
}

// ── W7-DEVICE: the DEVICE forward. Runs the SAME composition as
//    DeepseekV4ForwardHost but routes the four NEW V4 op families through the
//    CUDA kernels (kDeepseekV4{Mhc,Dsa,Compressor,Moe}) via the OpProvider seam,
//    at the tiny structural config. device==host within near-tie is the
//    ForwardDevice composition gate (test_cuda_deepseek_v4.cpp). The small linear
//    projections stay host in both modes (the real path REUSES the existing
//    MLA/MoE-grouped/NVFP4 GEMM kernels — a documented W7 seam). The full paged
//    engine over a materialized 167B checkpoint is the W8 residual. ────────────
//
// FRAMEWORK-CONFORMANCE: the registry hook (deepseek_v4_registry.cpp) routes the
// runner's ModelRegistry::Forward here on the gather-logits path, and the return is
// now DEVICE-RESIDENT on a CUDA queue (WrapV4DeviceLogits) so the runner's on-device
// sampler consumes it like every framework-conforming model — no host logit
// download. (The device-resident DECODE forward + its KV/scratch off the shared
// AttnBlock is the scoped follow-up; today this shares ForwardComposeImpl.)
ForwardLogits DeepseekV4Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const DeepseekV4Weights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  (void)attn_meta;
  (void)attn_kv;
  VT_CHECK(weights.has_host_weights, kHostPending);
  VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending);
  std::vector<float> flat = ForwardComposeImpl(
      weights.host, weights.params, token_ids, positions, logits_indices,
      V4Miswire::kNone, /*trace=*/nullptr,
      V4Backend{/*device=*/true, /*q=*/&queue, /*gguf=*/nullptr});
  const int64_t vocab = weights.params.vocab_size;
  const int64_t rows =
      vocab > 0 ? static_cast<int64_t>(flat.size()) / vocab : 0;
  return WrapV4DeviceLogits(std::move(flat), rows, vocab, queue);
}

}  // namespace vllm
