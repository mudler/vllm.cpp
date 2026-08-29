// DeepSeek-V4-Flash W7-device — the CUDA kernel seam for the four NEW V4 op
// families. This header declares the DEVICE launchers (1:1 CUDA ports of the
// landed portable HOST references) and the OpProvider-seam resolvers the device
// forward (DeepseekV4Model::ForwardDevice) and the unit gate dispatch through.
//
// ─── WHAT THIS IS A PORT OF (the four families; file:line on BOTH sides) ─────
//   MHC        <- deepseek_v4_mhc.{h,cpp}        (Sinkhorn / MhcPre / MhcPost /
//                                                 HcHeadCollapse) @ kernels/mhc/*.py
//   DSA        <- deepseek_v4_dsa.{h,cpp}        (indexer weight-fold / MQA logits /
//                                                 causal top-k / sink softmax /
//                                                 grouped output-LoRA)
//   Compressor <- deepseek_v4_compressor.{h,cpp} (save-time APE / pool+norm /
//                                                 fp8_ds_mla KV encode+decode)
//   MoE        <- deepseek_v4_moe.{h,cpp}         (sqrtsoftplus / hash+bias router /
//                                                 clamped SwiGLU)
// The 512-wide MLA attention + expert grouped-GEMM REUSE the existing NVFP4/FP8
// CUDA paths (cuda_mla_attn.cu, cuda_moe*.cu) and are NOT re-ported here — only
// these four NEW glue families need dedicated V4 kernels.
//
// ─── HONEST SCOPE (mirrors W3-W7) ────────────────────────────────────────────
// The launchers take/return host std::vectors and upload/run/download internally
// (via the CUDA backend). That is a STRUCTURAL, correctness-grade path — each
// kernel is unit-gated on the DGX GB10 against its landed host reference at a
// SMALL synthetic shape (test_cuda_deepseek_v4.cpp) — NOT a fused/perf path. The
// per-op host round-trip lets a CPU-compiled TU (deepseek_v4.cpp ForwardDevice)
// drive the kernels through the seam without linking CUDA symbols; the real
// paged-engine e2e over a materialized checkpoint stays the W8 residual (the
// fixed-config 167B does not fit ONE GB10 — see deepseek_v4.h).
//
// SEAM: each family registers ONE OpProvider under a dedicated OpId
// (vt/ops.h: kDeepseekV4{Mhc,Dsa,Compressor,Moe}); the provider `fn` points at a
// static kernels-struct of typed device launchers. The resolvers below cast the
// GetOp() result; they THROW on a CPU-only build (nothing registered for kCUDA),
// which is correct — ForwardDevice is a device-only path.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_compressor.h"  // Fp8DsMlaLayout / Fp8DsMlaToken
#include "vllm/model_executor/models/deepseek_v4_mhc.h"          // MhcPreResult
#include "vllm/model_executor/models/deepseek_v4_moe.h"          // MoeRouteResult
#include "vt/device.h"                                           // vt::Queue
#include "vt/tensor.h"                                            // vt::Tensor

namespace vllm::deepseek_v4 {

// ── (1) MHC family device kernels ─────────────────────────────────────────────
struct MhcDeviceKernels {
  std::vector<float> (*sinkhorn)(vt::Queue&, const std::vector<float>& comb_logits,
                                 int64_t hc, int64_t iters, float eps);
  MhcPreResult (*pre)(vt::Queue&, const std::vector<float>& residual,
                      const std::vector<float>& fn, const std::vector<float>& scale,
                      const std::vector<float>& base, int64_t hc, int64_t hidden,
                      float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
                      float hc_post_mult, int64_t iters,
                      const std::vector<float>& norm_weight, float norm_eps);
  std::vector<float> (*post)(vt::Queue&, const std::vector<float>& x,
                             const std::vector<float>& residual,
                             const std::vector<float>& post_layer_mix,
                             const std::vector<float>& comb_res_mix, int64_t hc,
                             int64_t hidden);
  std::vector<float> (*head)(vt::Queue&, const std::vector<float>& x,
                             const std::vector<float>& fn, float scale,
                             const std::vector<float>& base, int64_t hc, int64_t hidden,
                             float rms_eps, float hc_eps);
  // Brick B — IN-PLACE MHC glue (unified memory, no Upload/Download/Sync; caller
  // drains). Same MhcPost/HcHead/MhcPreKernel as above. `mix_scratch` is a
  // caller-provided [(2+hc)*hc] unified buffer for pre's intermediate mixes.
  void (*post_ip)(vt::Queue&, float* out, const float* x, const float* residual,
                  const float* post_mix, const float* comb, int64_t hc, int64_t hidden);
  void (*head_ip)(vt::Queue&, float* out, const float* x, const float* fn, float scale,
                  const float* base, int64_t hc, int64_t hidden, float rms_eps, float hc_eps);
  void (*pre_ip)(vt::Queue&, float* pre_mix, float* post_mix, float* comb_mix,
                 float* layer_input, float* mix_scratch, const float* residual, const float* fn,
                 const float* scale, const float* base, int64_t hc, int64_t hidden, float rms_eps,
                 float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult, int64_t iters,
                 const float* norm_weight, bool has_norm, float norm_eps);
};

// ── (2) DSA family device kernels ─────────────────────────────────────────────
struct DsaDeviceKernels {
  std::vector<float> (*weight_fold)(vt::Queue&, const std::vector<float>& weights_proj,
                                    int64_t num_tokens, int64_t index_n_heads,
                                    int64_t index_head_dim);
  std::vector<float> (*logits)(vt::Queue&, const std::vector<float>& q,
                               const std::vector<float>& k,
                               const std::vector<float>& folded_weights,
                               const std::vector<int64_t>& win_start,
                               const std::vector<int64_t>& win_end, int64_t num_tokens,
                               int64_t num_keys, int64_t index_n_heads,
                               int64_t index_head_dim);
  std::vector<int64_t> (*topk)(vt::Queue&, const std::vector<float>& logits,
                               const std::vector<int64_t>& win_start,
                               const std::vector<int64_t>& win_end, int64_t num_tokens,
                               int64_t num_keys, int64_t topk);
  std::vector<float> (*softmax_sink)(vt::Queue&, const std::vector<float>& scores, float sink);
  // W1d (#2186): the two weights are BF16 bit patterns (`HostBf16`), not f32 --
  // the carried tower's FP8-sourced half is held at the model dtype. A function
  // pointer cannot be a template, so this entry names the ONE dtype the carried
  // tower actually has; the f32 arm lives on in the CPU `GroupedOutputLora<float>`
  // that the ported upstream-parity tests drive.
  std::vector<float> (*grouped_olora)(vt::Queue&, const std::vector<float>& o,
                                      const std::vector<uint16_t>& wo_a,
                                      const std::vector<uint16_t>& wo_b, int64_t num_tokens,
                                      int64_t n_heads, int64_t head_dim, int64_t n_groups,
                                      int64_t o_lora_rank, int64_t hidden_size);
  // Brick A — device MLA decode/prefill attention over the unified KV-cache latent.
  // Unlike the launchers above (host-vector, Upload/Download/Sync), this reads/writes
  // UNIFIED-MEMORY raw pointers IN PLACE on the queue stream (no round-trip) — the
  // first real device V4 forward kernel, toward a capturable decode graph. q
  // [T*nh*hd], kv [n_keys*hd] (cached deck; num KV heads = 1, shared across heads),
  // sink [nh], o [T*nh*hd], all on the queue device. Causal: query t attends keys
  // [0, kv_base+t]. no_sink = the kNoAttnSink miswire (sink -> -inf). Matches
  // SoftmaxWithSink (deepseek_v4_dsa.cpp:116) with host accumulation order preserved.
  // Launches async on q's stream; the CALLER drains (Brick A) or captures (Brick D).
  void (*decode_attn)(vt::Queue&, float* o, const float* q, const float* kv,
                      const float* sink, int64_t nh, int64_t hd, int64_t kv_base,
                      int64_t num_tokens, float scale, bool no_sink);
  // Brick C — folded-in device glue (in place on unified/device buffers, no
  // Upload/Download/Sync; caller drains at Brick C / captures at Brick D).
  // rms_norm: weighted RMSNorm over [n] (has_w=false → the per-head q-RMS). Near-tie
  // (block reduction reorders vs host double-sequential).
  void (*rms_norm)(vt::Queue&, float* out, const float* x, const float* w, int64_t n, float eps,
                   bool has_w);
  // rope: sequential-recurrence RoPE over `num_rows` rows (each v[row*row_stride+off..+r]),
  // per-row position `row_pos[row]`. inverse flips the sin sign. Near-tie (cos/sin lib).
  void (*rope)(vt::Queue&, float* v, int64_t num_rows, int64_t row_stride, int64_t off, int64_t r,
               const int* row_pos, double base, double freq_scale, double ext_factor,
               int64_t n_ctx_orig, double beta_fast, double beta_slow, bool inverse);
  // Brick C part 2 — BATCHED RMSNorm over `rows` independent [n] segments in ONE
  // launch (the per-head q-RMS: rows=nh, has_w=false). Per-row identical to rms_norm
  // above (same block reduction; a shared weight w[n] applies to every row when has_w).
  void (*rms_norm_rows)(vt::Queue&, float* out, const float* x, const float* w, int64_t rows,
                        int64_t n, float eps, bool has_w);
  // Brick D step 2 — GRAPH decode attention (T=1, capturable): reads the KV length
  // from the DEVICE buffer `len_dev` (so ONE captured graph serves every step) and
  // attends `cache[0..len)` + the current token's key `deck_new` (index len), FIXED
  // shmem (max_cap keys). Bit-identical to decode_attn (same key set + order).
  void (*decode_attn_g)(vt::Queue&, float* o, const float* q, const float* cache,
                        const float* deck_new, const float* sink, int64_t nh, int64_t hd,
                        const int* len_dev, int64_t max_cap, float scale, bool no_sink);
  // Brick 7 — FUSED per-row RMSNorm + RoPE (ds4 head_rms_norm_rope_tail_kernel /
  // dsv4_qkv_rms_norm_rows_kv_rope_kernel). Collapses the {rms_norm_rows ; rope}
  // launch pair (q per-head norm+rope; kv norm+rope) into ONE kernel — normalized
  // values never round-trip HBM — and parallelizes the RoPE tail (block-per-row,
  // threads split the r/2 pairs) BIT-IDENTICALLY (double reduction + left-fold theta
  // recurrence). do_norm=false + inverse=true is the standalone post-attention
  // inverse o-RoPE (no norm). `in`/`out` may alias (in-place q/o); for kv in=kraw,
  // out=slot. off/r are the RoPE tail window [off, off+r) within each n-wide row.
  void (*norm_rope_rows)(vt::Queue&, float* out, const float* in, const float* w, int64_t rows,
                         int64_t n, int64_t off, int64_t r, const int* row_pos, double base,
                         double freq_scale, double ext_factor, int64_t n_ctx_orig, double beta_fast,
                         double beta_slow, bool inverse, bool has_w, bool do_norm, float eps);
  // Brick 12 (ds4-gap "launch consolidation") — PAIRED Q8_0 decode GEMV: ONE launch
  // computes out0=w0·act and out1=w1·act over the SAME activation (m==1), quantizing
  // `act` to Q8_0 once. Halves the launch count for the two A-projections that share the
  // layer hidden (MLA q_a+kv_a; shared-expert gate+up). BIT-IDENTICAL to two matmul_q8_0
  // launches (same preq quant + 8×__dp4a dot + warp reduce). out0[1,n0]/out1[1,n1],
  // act[1,K], w0[n0,K]/w1[n1,K] plain Q8_0 blocks — all on the queue device, no drain.
  // Ports ds4 `matmul_q8_0_pair_preq_warp8_kernel` (ds4_cuda.cu:4485).
  void (*matmul_q8_0_pair)(vt::Queue&, vt::Tensor& out0, vt::Tensor& out1, const vt::Tensor& act,
                           const vt::Tensor& w0, const vt::Tensor& w1);
  // Brick 12 (ds4-gap "row-split consolidation") — BLOCK-DIAGONAL grouped Q8_0 decode
  // GEMV: consolidates the ng per-group output-LoRA `wo_a` GEMVs into ONE launch. `w` is
  // the stacked [ng*rows_per_group, ipg] weight (row rr → group rr/rows_per_group); `act`
  // is the full [1, ng*ipg] attention output, quantized ONCE (each group reads its
  // block-aligned ipg-wide slice). BIT-IDENTICAL to the ng separate slice GEMVs. out
  // [1, ng*rows_per_group] contiguous. Ports ds4 `grouped_q8_0_a_preq_warp8_kernel`
  // (ds4_cuda.cu:5509).
  void (*matmul_q8_0_olora_a)(vt::Queue&, vt::Tensor& out, const vt::Tensor& act,
                              const vt::Tensor& w, int64_t n_groups);
};

// ── (3) Compressor family device kernels ──────────────────────────────────────
struct CompressorDeviceKernels {
  std::vector<float> (*save_score_ape)(vt::Queue&, const std::vector<float>& score,
                                       const std::vector<float>& ape,
                                       const std::vector<int64_t>& positions,
                                       int64_t num_tokens, int64_t width,
                                       int64_t compress_ratio);
  std::vector<float> (*pool_norm)(vt::Queue&, const std::vector<float>& kv,
                                  const std::vector<float>& score,
                                  const std::vector<uint8_t>& valid,
                                  const std::vector<float>& rms_weight, float eps,
                                  int64_t window, int64_t head_dim);
  Fp8DsMlaToken (*encode)(vt::Queue&, const std::vector<float>& head,
                          const Fp8DsMlaLayout& layout);
  std::vector<float> (*decode)(vt::Queue&, const Fp8DsMlaToken& token,
                               const Fp8DsMlaLayout& layout);
};

// ── (4) MoE family device kernels ─────────────────────────────────────────────
struct MoeDeviceKernels {
  // Elementwise sqrt(softplus(x)) over an arbitrary buffer (the router score).
  std::vector<float> (*sqrtsoftplus)(vt::Queue&, const std::vector<float>& x);
  MoeRouteResult (*route)(vt::Queue&, const std::vector<float>& gating, int64_t num_tokens,
                          int64_t num_experts, int64_t topk,
                          const std::vector<float>& e_score_correction_bias, bool renormalize,
                          float routed_scaling_factor,
                          const std::vector<int64_t>& input_tokens,
                          const std::vector<int32_t>& hash_indices_table, int64_t vocab_size);
  std::vector<float> (*clamped_swiglu)(vt::Queue&, const std::vector<float>& gate_up,
                                       int64_t d, float limit, float alpha, float beta);
  // Brick B — IN-PLACE clamped-SwiGLU: reads gate_up[2*d], writes out[d] on the
  // queue device (unified memory), NO Upload/Download/Sync (caller drains at Brick
  // B / captures at Brick D). Same ClampedSwiGLUKernel math as clamped_swiglu above
  // ⇒ bit-identical (elementwise, no reduction).
  void (*clamped_swiglu_ip)(vt::Queue&, float* out, const float* gate_up, int64_t d,
                            float limit, float alpha, float beta);
  // Brick B — IN-PLACE router: same RouteKernel; writes topk_ids[T*topk] (i32) +
  // topk_weights[T*topk] on the queue device (unified), no Upload/Download/Sync.
  void (*route_ip)(vt::Queue&, int32_t* topk_ids, float* topk_weights, const float* gating,
                   int64_t T, int64_t E, int64_t topk, const float* bias, bool has_bias,
                   const int64_t* in_tokens, bool is_hash, const int32_t* hashtab,
                   int64_t vocab, bool renorm, float scale);
  // Brick C — MoE combine: out[h] = Σ_a weights[a]*eo[a*H+h] (per-h sequential over
  // the A experts; near-tie vs host — device FMA contraction). In place on the queue.
  void (*moe_combine)(vt::Queue&, float* out, const float* eo, const float* weights, int64_t A,
                      int64_t H);
  // Brick D — DEVICE router gate: gating[e] = Σ_h x[h]·bf16→f32(W[e*H+h]) over the
  // [ne,H] BF16 `ffn.gate.weight`. Sequential f32 dot + exact bf16 upcast ⇒
  // BIT-IDENTICAL to the host CPU MatmulBT — replaces the last non-capturable host
  // op (the f32-act×bf16-weight GEMM the CUDA elementwise MatmulBT lacks) so the
  // resident decode step is 100% device (capturable). w_bf16 = the bf16 weight bytes.
  void (*router_gate)(vt::Queue&, float* gating, const float* x, const void* w_bf16, int64_t ne,
                      int64_t H);
  // FUSED routed-MoE gate+up+SwiGLU (ds4 moe_gate_up_mid). ONE launch computes,
  // per (expert-slot p, mid-row j), gate=gate_w[e,j]·x, up=up_w[e,j]·x (shared
  // Q8_K x, broadcast), then adown[p*mi+j] = silu(min(gate,limit))·clamp(up,±limit)
  // — collapsing {gate grouped-GEMM + up grouped-GEMM + topk×2 copies + topk
  // ClampedSwiGLU} → 1 kernel, gate/up never touching HBM. BIT-IDENTICAL to that
  // chain (same integer dot, warp reduce, FinalFactor, SwiGLU formula α=1,β=0);
  // the route weight stays in moe_combine (post-down). `gate_w`/`up_w` are the
  // stacked [E*mi, H] block-quant expert towers (moe_gate_exps / moe_up_exps).
  // out[P,mi] adown, act[1,H] broadcast, gate_w/up_w[E*mi,H] same block-quant
  // dtype, expert_ids[P] i32 — all on the queue device (unified memory), no drain.
  void (*moe_gate_up_swiglu)(vt::Queue&, vt::Tensor& out, const vt::Tensor& act,
                             const vt::Tensor& gate_w, const vt::Tensor& up_w,
                             const vt::Tensor& expert_ids, float limit);
};

// Resolve a family's device kernels through the vt OpProvider seam. THROWS on a
// CPU-only build (no kCUDA provider registered) — ForwardDevice is device-only.
const MhcDeviceKernels* MhcDevice();
const DsaDeviceKernels* DsaDevice();
const CompressorDeviceKernels* CompressorDevice();
const MoeDeviceKernels* MoeDevice();

// True iff the CUDA backend registered the four V4 families (a device build with
// a live CUDA backend). ForwardDevice checks this before dispatch.
bool V4DeviceKernelsAvailable();

}  // namespace vllm::deepseek_v4
