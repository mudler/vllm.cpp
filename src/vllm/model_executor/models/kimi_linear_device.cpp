// Kimi-Linear W7 — the real DBuf-RESIDENT device COMPUTE forward. This TU is the
// device-compute lane the W6 SEAM (kimi_linear.cpp) documented as its plan: a
// pooled-DBuf forward that routes the whole 27-layer KDA/NoPE-MLA + 256-expert-MoE
// hybrid through the SHARED vt:: device ops, gated on CPU against the W2 host f32
// reference (kimi_linear_forward.cpp). Because the CPU backend executes the SAME
// vt:: dispatch (a pooled DBuf is a device buffer on CPU, ResidentWeight aliases
// the host weight bytes on CPU), a tight CPU match to the reference is a REAL proof
// of the residual-stream / vt-op / MoE-routing / on-device-logits WIRING the GPU
// will run — only the GPU numerics (bf16 activations, the GDN Triton-AOT cubins,
// the paged het-KV, the grouped-MoE slabs) + the e2e SACRED golden stay pending
// (box down). Honest labeling: NOT a DONE claim for the GPU-unverified device path.
//
// ─── DEVICE vs HOST-FALLBACK, precisely (each op CPU+CUDA-registered) ──────────
// ON DEVICE (genuine vt:: dispatch on pooled DBufs, f32 activations to match the
// f32 reference tightly):
//   embed                vt::Embedding
//   add+RMSNorm glue      vt::FusedChain(kFusedAddRmsNormStd)  (the fusion seam)
//   every projection      vt::MatmulBT (host weights are torch [out,in] = [N,K])
//   KDA short convs       vt::CausalConv1dFwd (silu, fresh zero conv-state)
//   KDA q/k L2-norm       vt::L2Norm
//   KDA output gated-norm vt::RmsNormGated (sigmoid gate)
//   MoE router            vt::MoeRouterTopK (sigmoid noaux_tc, group 1/1, bias, scale)
//   SwiGLU activation     vt::MoeSiluMul (dense MLP + each expert + shared expert)
//   MoE weighted combine  vt::MoeCombine (+ shared term)
//   lm_head               vt::MatmulBT
// HOST-FALLBACK ISLANDS (the W7-speed residuals — no portable device op yet):
//   (1) KDA per-k-channel gated-delta RECURRENCE + its decay gate g =
//       -exp(A_log)*softplus(f_b(f_a(x))+dt_bias) + beta=sigmoid(b_proj). vt::Gdn
//       Decode/GdnPrefill carry only a per-HEAD scalar decay g[T,Hv] (ops.h), so
//       they CANNOT express KDA's per-channel g[T,H,D]; the recurrence is computed
//       on host from device-resident q/k/v/g1/beta and uploaded (kimi_kda refs).
//   (2) NoPE-MLA attention CORE (causal scores/softmax/weighted-V). The device
//       path is mla::ForwardMlaAttentionBlock over the runner's PAGED het-KV cache
//       + the load-time W_UK/W_UV absorption + TritonMLAImpl — the born-on-runner
//       residual. This seam keeps every MLA projection + kv_a_layernorm ON DEVICE
//       and computes only the softmax core on host (identical materialized-MHA math
//       as the W2 reference, NoPE so no RoPE).
//
// Grounding: the reuse-wiring plan authored in kimi_linear.cpp; the per-op numerics
// in kimi_linear_forward.cpp (the W2 reference) + kimi_kda.{h,cpp}. Mirrors the
// deepseek_v2.cpp device forward structure (ForwardBody/RunLayer/MoeBlock/DenseMlp
// residual-stream + WrapDeviceLogits).
#include "vllm/model_executor/models/kimi_linear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "vllm/model_executor/models/deepseek_v2.h"        // MlaBatchSplit (ROW 7 fold)
#include "vllm/model_executor/models/dense_device_glue.h"  // dense_attn::{Dev,DBuf,MakeTensor}
#include "vllm/model_executor/models/device_pool.h"        // Pool()
#include "vllm/model_executor/models/kimi_kda.h"
#include "vllm/model_executor/models/mla_attention.h"      // mla::ForwardMlaAttentionBlock
#include "vllm/platforms/interface.h"                       // platforms::GetPlatform (is_cpu)
#include "vllm/v1/attention/backends/gdn_attn.h"           // GDNAttentionMetadata
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // vt::kFusedAddRmsNormStd

namespace vllm {
namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::DType;
using vt::Tensor;

// VT_FUSED_CHAIN_ADOPT gate, mirroring dense_attn::FusedChainAdoptEnabled (a
// local copy so this TU need not pull the heavy dense_attn_block.h). Default ON:
// the residual add+RMSNorm goes through the vt::FusedChain catalog seam; =0 falls
// back to the bit-identical residual RmsNorm overload for an A/B.
bool FusedGlue() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

inline double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ─── STRICT-path numerics knobs (W7-speed lane) ────────────────────────────────
// The correctness vehicle is MORE precise than vLLM (f32 residual stream + f64 host
// islands), which flips near-tie argmaxes where vLLM's deterministic bf16 top-1 has
// a small margin (spec §13 root cause). These two knobs mirror vLLM's bf16 compute
// regime so the near-ties become token-exact. Both default OFF → the f32 vehicle is
// byte-identical (the CPU tiny-config gate stays 13/13·656); flip ON only for the
// full-model on-box token gate vs the STRICT golden.
//
// (1) VT_KIMI_BF16_RESIDUAL — carry the residual stream in bf16 like vLLM. vLLM's
// fused_add_rms_norm stores `residual` as bf16 and `hidden_states`/block-outputs are
// bf16, while it computes the RMSNorm variance over the f32 pre-store sum
// (layernorm.py / fused_add_rms_norm.cu). We keep f32 STORAGE (so the two host-fallback
// islands still consume f32) but round the VALUE to bf16 precision at exactly vLLM's
// rounding points: the embed output, each block output before it re-enters the add,
// and the residual AFTER each add (so the norm still sees the f32 sum — byte-matching
// vLLM's fused-add-rms-norm order). 54 bf16 roundings across 27 layers × 2 norms that
// vLLM does and our f32 vehicle does not.
bool Bf16Residual() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_BF16_RESIDUAL");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (2) VT_KIMI_BF16_ISLANDS — round the host-fallback island INPUTS (KDA recurrence
// q/k/v/g1/beta, MLA softmax q/kv/kpe) to bf16 precision before the f64 recurrence.
// vLLM feeds bf16 activations into the GDN Triton-AOT / FA2 kernels; our islands
// download f32 (bf16-precision projection outputs stored to f32) and recompute in f64.
// Rounding the inputs to bf16 moves the island toward vLLM's kernel precision without a
// new device kernel (the true fix — the device GDN per-channel-decay recurrence + the
// paged FA2 MLA — is the named W7-speed residual, spec §13).
bool Bf16Islands() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_BF16_ISLANDS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Round an f32 value to bf16 precision (round-to-nearest-even), matching torch/vLLM's
// bf16 cast (and vt::CastBf16). Truncate-with-RNE-bias on the top 16 bits; qNaN-safe.
inline float ToBf16Rne(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    u |= 0x00400000u;  // qNaN
  } else {
    u += 0x00007fffu + ((u >> 16) & 1u);  // RNE rounding bias
  }
  u &= 0xffff0000u;
  float r;
  std::memcpy(&r, &u, sizeof(r));
  return r;
}
inline void RoundHostBf16(std::vector<float>& v) {
  if (!Bf16Islands()) return;
  for (float& x : v) x = ToBf16Rne(x);
}

// (3) VT_KIMI_ISLAND_F32ACC — compute the host-fallback island recurrence/softmax in
// f32 accumulation (not f64), matching vLLM's GDN Triton / FA2 kernels (bf16 I/O, f32
// accumulation). Our island defaults to f64 (MORE precise than vLLM); rounding each
// accumulation step to f32 mirrors the device kernel's rounding. Combined with
// VT_KIMI_BF16_ISLANDS (bf16 I/O), this is the closest host approximation of vLLM's
// actual kernel numerics without a new device kernel (the named W7-speed residual).
bool IslandF32Acc() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_ISLAND_F32ACC");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (4) VT_KIMI_DEVICE_KDA — run the KDA per-k-channel gated-delta RECURRENCE through the
// device op vt::KdaGatedDeltaRule (the net-new per-channel-decay GDN kernel, cuda_gdn.cu
// KdaScanKernel) instead of the f64 host recompute. This is the principled path to STRICT
// AND the speed lever (spec §14): the recurrence runs vLLM's actual f32-on-bf16 arithmetic
// (FLA fused_recurrent_gated_delta_rule_fwd_kernel IS_KDA=True) on device rather than a
// host f64 recompute that is MORE precise than vLLM and coin-flips near-ties. The decay
// gate `g = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias)` and beta = sigmoid(b) stay host
// (elementwise, numerically stable — the numerically-sensitive object is the recurrence).
// Default OFF (parity-enabler: flip ON only with the token gate green). Independent of the
// bf16-precision knobs (BF16_ISLANDS still rounds the gate inputs when both are set).
bool DeviceKda() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_KDA");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (4b) VT_KIMI_DEVICE_KDA_CHUNK — process the PROMPT-length KDA with the CHUNKED
// prefill kernel family (vt::KdaChunkPrefill: the vendored FLA Triton-AOT cubins
// kda_gate_cumsum -> kkt -> solve_tril -> recompute_w_u -> chunk_delta_h ->
// chunk_gla_o) instead of the RECURRENT form, exactly as vLLM
// (kimi_gdn_linear_attn.py:141 chunk_kda_with_fused_gate; decode stays recurrent).
// This is the spec §15 STRICT residual (c): vLLM processes the prompt chunked, we
// still recur — a different reduction ORDER coin-flips the p7 near-tie. Requires
// VT_KIMI_DEVICE_KDA=1 (the recurrence is the T==1 / fallback path). Default OFF
// (parity-enabler; flip ON only with the token gate green). The chunk op fuses the
// gate on-device (raw g1 + a_log + dt_bias), so unlike the recurrent branch nothing
// is host-rounded — the chunk kernels carry vLLM's exact bf16 path.
bool DeviceKdaChunk() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_KDA_CHUNK");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// (5) VT_KIMI_DEVICE_MLA — run the NoPE-MLA attention CORE (causal scores/softmax/
// weighted-V) through the shared device op vt::Attention instead of the f64 host
// softmax island. This is the MLA twin of VT_KIMI_DEVICE_KDA (spec §14/§15 residual
// (d)): the host island runs an f64 softmax that is MORE precise than vLLM's FA2 and
// coin-flips near-ties, whereas vt::Attention runs the f32 max-subtracted online-
// softmax accumulation vLLM's kernels use (cpu_ops AttentionKernel / cuda_ops). The
// MLA attention has asymmetric head dims (qk = qk_nope+qk_rope = 192, v = 128), which
// vt::Attention (single head-dim D for q/k/v) does not express directly, so the value
// is PADDED to the qk head-dim with zeros (weighted-sum over the zero tail = 0) and the
// out[:, :, :v] slice is the true attention core — bit-exact to the unpadded math since
// softmax weights depend only on q·k. Requires VT_KIMI_DEVICE_COMPUTE=1. Independent of
// the bf16-precision knobs (BF16_ISLANDS still rounds the q/kv/kpe inputs when both are
// set — applied on device before the attention).
//
// ★ MEASURED NEGATIVE (2026-08-07, GB10 full 48.9B gate, spec §16) — kept as a
// documented-negative A/B knob, DEFAULT OFF. On the device-KDA best config
// (VT_KIMI_DEVICE_KDA=1), adding VT_KIMI_DEVICE_MLA REGRESSES 122→109/128 AND slows
// 4.24→3.89 tok/s: (1) vt::Attention's f32 online max-subtracted softmax is NOT vLLM's
// FA2 reduction ORDER — it is a DIFFERENT approximation, so it coin-flips near-ties (it
// BREAKS p3 16/16→3/16 into the `163586×` repeat loop while p7 stays diverged), the same
// §14 plateau class; (2) the per-(t,h) key/value build copies + the 192-dim pad-V waste
// ADD overhead to the O(n²) recompute path. The principled MLA-half STRICT lever is
// vLLM's ACTUAL FA2 via paged mla::ForwardMlaAttentionBlock (residual d, coupled with
// paged-incremental decode e), NOT this softmax approximation — this negative is the
// measurement that proves the approximation is not enough.
bool DeviceMla() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_MLA");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// Round a running f64 accumulator to f32 precision when the knob is on (identity else).
inline double AccR(double x) {
  static const bool f32 = IslandF32Acc();
  return f32 ? static_cast<double>(static_cast<float>(x)) : x;
}

// In-place round an f32 device buffer to bf16 precision (f32→bf16→f32, on-device, no
// download). The VALUE becomes bf16-exact so the RMSNorm variance and the next residual
// add see the same bf16 numbers vLLM does; the STORAGE stays f32 (the islands read f32).
void RoundDevBf16(const Dev& d, DBuf& x) {
  Tensor xt = x.t();
  std::vector<int64_t> shape(xt.shape, xt.shape + xt.rank);
  DBuf b(d, DType::kBF16, shape);
  vt::CastBf16(d.q, b.t(), xt);
  Tensor out = x.t();
  vt::CastF32(d.q, out, b.t());
}

// (6) VT_KIMI_BF16_STREAM — carry the inter-layer residual stream in bf16 END-TO-END,
// exactly as vLLM (DeepseekV2Model::ForwardBody / deepseek_v2.cpp:479-615: hidden /
// residual / normed-hidden / block-outputs all bf16; fused_add_rms_norm rounds the
// residual store to bf16). This SUPERSEDES the partial VT_KIMI_BF16_RESIDUAL knob (spec
// §14, which rounded f32 STORAGE in place): the STRUCTURAL bf16 storage IS vLLM's
// rounding AND removes the per-GEMM CastBf16 on every residual-fed projection (the
// decode-decomposition's 3% CastBf16, spec §19) plus the f32<->bf16 round-trips. The
// projection OUTPUTS stay f32 (the KDA/MLA islands read f32 unchanged), so the ONLY
// numeric change vs the f32 stream is the residual add/norm rounding to bf16 at each
// layer boundary — the isolated p7-STRICT lever (spec §13/§14 root cause). Default OFF
// (parity-enabler: flip only with the token gate green at the flipped default); when ON,
// VT_KIMI_BF16_RESIDUAL's manual RoundDevBf16 is redundant and skipped.
bool Bf16Stream() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_BF16_STREAM");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
inline DType StreamDType() { return Bf16Stream() ? DType::kBF16 : DType::kF32; }

// Cast a block output DBuf to the residual-stream dtype (identity when already `sdt`;
// f32 -> bf16 round for the bf16 stream — the o_proj/down/combine GEMM's bf16 store,
// done as a separate cast so the block internals stay dtype-agnostic). RNE-equal to a
// (bf16,bf16)->bf16 MatmulBT store, so numerically vLLM-faithful.
DBuf ToStream(const Dev& d, DBuf&& out, DType sdt) {
  if (out.t().dtype == sdt) return std::move(out);
  Tensor ot = out.t();
  std::vector<int64_t> shape(ot.shape, ot.shape + ot.rank);
  DBuf s(d, sdt, shape);
  Tensor st = s.t();
  vt::CastBf16(d.q, st, ot);  // f32 -> bf16 (sdt is bf16 in the non-identity case)
  return s;
}

// Device-resident weight view. On CPU this ALIASES the host f32 bytes exactly as
// dense_attn::ResidentWeight does for a CPU device (host-pointer aliasing is a CPU
// property); the CUDA staging over materialized OwnedTensors is the born-on-runner
// residual (Kimi's device weights are not materialized as OwnedTensors yet — see
// kimi_linear.h). The device-compute lane is CPU-reachable in this brick.
inline Tensor WF32(const Dev& d, const std::vector<float>& v,
                   const std::vector<int64_t>& shape) {
  return MakeTensor(const_cast<float*>(v.data()), DType::kF32, d.q.device, shape);
}

// ─── bf16-RESIDENT weight view + cast-GEMM (§13) ───────────────────────────────
// Device-resident bf16 weight view over an OwnedTensor (mirror laguna.cpp:125-139
// LagunaResidentBf16W / dense_attn::ResidentWeight). CPU: alias the host bf16 bytes
// (host-pointer aliasing is a CPU property). CUDA: return the d_dev copy — pre-staged
// at load (StageKimiResidentBf16 + ReleaseHost, the pool-math path) or, if absent,
// uploaded ONCE here (cudaMalloc + one H2D, byte-exact — no ATS penalty).
inline Tensor ResidentBf16W(const Dev& d, const OwnedTensor& w,
                            const std::vector<int64_t>& shape) {
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu())
    return MakeTensor(const_cast<uint8_t*>(w.bytes.data()), w.dtype, d.q.device, shape);
  if (!w.d_dev) {
    VT_CHECK(w.HasHostBytes(),
             "kimi resident: bf16 weight host bytes released before device staging");
    const size_t nb = w.bytes.size();
    void* p = d.b.Alloc(nb);
    d.b.Copy(d.q, p, w.bytes.data(), nb);
    vt::Backend* bk = &d.b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_dev.get(), w.dtype, d.q.device, shape);
}

// bf16 cast-GEMM: out[T,N] f32 = cast(act[T,K] -> bf16) . w_bf16[N,K]^T (mirror
// laguna.cpp:1939-1946 GemmBf16Into). The (bf16,bf16)->f32 MatmulBT IS vLLM's
// projection numerics (cuda_matmul.cu:3); the residual stream stays f32.
void GemmBf16(const Dev& d, Tensor& out, const Tensor& act, const OwnedTensor& w,
              int64_t N, int64_t K) {
  Tensor wt = ResidentBf16W(d, w, {N, K});
  // bf16 residual stream (VT_KIMI_BF16_STREAM): the activation is ALREADY bf16, so the
  // per-GEMM CastBf16 (the decode-decomposition's 3%, spec §19) is elided — feed the
  // bf16 activation straight into the (bf16,bf16)->out MatmulBT.
  if (act.dtype == DType::kBF16) {
    vt::MatmulBT(d.q, out, act, wt);
    return;
  }
  const int64_t T = act.shape[0];
  DBuf ab(d, DType::kBF16, {T, K});
  vt::CastBf16(d.q, ab.t(), act);
  vt::MatmulBT(d.q, out, ab.t(), wt);
}

// Fused residual add + standard RMSNorm: res += x; out = rmsnorm(res) * w. The
// canonical add+RMSNorm glue seam (identical residual accumulation to the W2
// reference's single-`h` stream — see the deepseek_v2.cpp RunLayer derivation).
void AddRmsNorm(const Dev& d, DBuf& out, const Tensor& x, const Tensor& w, DBuf& res,
                float eps) {
  if (FusedGlue()) {
    vt::FusedChain(d.q, out.t(), x, w, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, out.t(), x, w, vt::RmsNormArgs{eps, false}, &res.t());
  }
}

// Stream-dtype-aware residual add+RMSNorm for the bf16 device forward. The residual-
// stream norm weights are kept host f32 (ReadF32 from the bf16 checkpoint — the value
// IS already a bf16 value); the CUDA RmsNorm/FusedChain kernels REQUIRE
// weight.dtype == x.dtype (cuda_ops.cu:452,3480), so for the bf16 stream the f32-stored
// (= bf16-valued) weight is cast to bf16 (LOSSLESS) before the norm — matching vLLM's
// bf16 norm weight exactly. For the f32 stream this aliases the host f32 bytes (WF32),
// byte-identical to the plain AddRmsNorm above.
void AddRmsNormS(const Dev& d, DBuf& out, const Tensor& x, const std::vector<float>& wv,
                 int64_t H, DBuf& res, float eps, DType sdt) {
  if (sdt == DType::kF32) {
    AddRmsNorm(d, out, x, WF32(d, wv, {H}), res, eps);
    return;
  }
  DBuf wf(d, DType::kF32, {H}, wv.data());
  DBuf wb(d, DType::kBF16, {H});
  vt::CastBf16(d.q, wb.t(), wf.t());
  AddRmsNorm(d, out, x, wb.t(), res, eps);
}

// silu(gate@x) * (up@x) -> down@(...) — a gated SwiGLU MLP via the shared vt:: ops
// on separate gate/up/down host weights (torch [out,in]). MoeSiluMul (not
// SiluAndMul) is used so the merged-GEMM checker is not tripped: the fused
// MlpGateUp merged-GEMM arm needs a fused gate_up weight (a loader residual), so
// the CPU device gate uses the separate-GEMM + MoeSiluMul equivalent.
DBuf SwiGluDevice(const Dev& d, const std::vector<float>& gate,
                  const std::vector<float>& up, const std::vector<float>& down,
                  const Tensor& dh, int64_t H, int64_t I, int64_t T) {
  DBuf dg(d, DType::kF32, {T, I});
  vt::MatmulBT(d.q, dg.t(), dh, WF32(d, gate, {I, H}));
  DBuf du(d, DType::kF32, {T, I});
  vt::MatmulBT(d.q, du.t(), dh, WF32(d, up, {I, H}));
  DBuf da(d, DType::kF32, {T, I});
  vt::MoeSiluMul(d.q, da.t(), dg.t(), du.t());
  DBuf out(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, out.t(), da.t(), WF32(d, down, {H, I}));
  return out;
}

// One depthwise causal short conv (silu), fresh zero conv-state (single sequence).
// Mirrors the qwen3_5.cpp GDN conv call (qwen3_5.cpp:3032-3040): f32 conv_state
// [1,C,K-1] + query_start_loc {0,T} + has_initial_state {0}. weight is [C,K].
DBuf ConvSilu(const Dev& d, const Tensor& x, const std::vector<float>& weight,
              int64_t T, int64_t C, int64_t K) {
  DBuf out(d, DType::kF32, {T, C});
  DBuf state(d, DType::kF32, {1, C, K - 1});
  state.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  vt::CausalConv1dFwd(d.q, out.t(), x, WF32(d, weight, {C, K}), nullptr, state.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
  return out;
}

// ── HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta RECURRENCE.
// vt::GdnDecode carries only a per-HEAD scalar decay (ops.h g/beta[T,Hv]); KDA's
// decay is per-k-channel, so the recurrence is computed on host from the device-
// resident q_n/k_n/v/g1/beta via the landed kimi_kda refs + the reference recurrence
// (kimi_linear_forward.cpp:142-183), then uploaded. THE W7-speed residual. Shared by
// the f32 (KdaLayerDevice) and bf16 (KdaLayerDeviceBf16) paths — the recurrence itself
// runs the IDENTICAL host code on both; only the GEMMs feeding it differ (f32 alias vs
// bf16 cast-GEMM), so extracting it keeps the two paths byte-identical here.
DBuf KdaRecurrenceIsland(const Dev& d, DBuf& qn, DBuf& kn, DBuf& vc, DBuf& g1,
                         DBuf& braw, const std::vector<float>& a_log,
                         const std::vector<float>& dt_bias, const KimiLinearParams& p,
                         int64_t T) {
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;

  // ── DEVICE RECURRENCE (VT_KIMI_DEVICE_KDA): run the per-k-channel gated-delta
  // recurrence on device via vt::KdaGatedDeltaRule (KdaScanKernel), vLLM's actual
  // f32-on-bf16 arithmetic, instead of the host f64 recompute below. q_n/k_n/v are
  // ALREADY device-resident; only the elementwise gate (KdaDecayGate) + beta = sigmoid(b)
  // are computed on host and uploaded (small, numerically stable). Fresh zero state,
  // single sequence, qsl=[0,T] — the stateless full-sequence recurrence the island needs.
  // ── CHUNK-PREFILL (VT_KIMI_DEVICE_KDA_CHUNK): route the whole prompt through the
  // chunked FLA Triton-AOT cubins (vt::KdaChunkPrefill) — vLLM's ACTUAL prefill path
  // — instead of the recurrence. The op fuses the gate on-device from the RAW g1
  // projection + a_log + dt_bias (no host gate compute, no bf16 island rounding); only
  // beta = sigmoid(braw) is the tiny host elementwise. T==1 (a single token) keeps the
  // recurrence below (the op itself also falls back for T==1). Spec §17.
  if (DeviceKda() && DeviceKdaChunk() && T > 1) {
    std::vector<float> hbraw(static_cast<size_t>(T) * nh), hbeta(static_cast<size_t>(T) * nh);
    braw.Download(d, hbraw.data());
    for (size_t i = 0; i < hbeta.size(); ++i) hbeta[i] = static_cast<float>(Sigmoid(hbraw[i]));
    DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());
    DBuf da_log(d, DType::kF32, {nh}, a_log.data());
    DBuf ddt(d, DType::kF32, {static_cast<int64_t>(dt_bias.size())},
             dt_bias.empty() ? nullptr : dt_bias.data());
    DBuf dstate(d, DType::kF32, {1, nh, hd, hd});
    dstate.Zero(d);
    DBuf dcore(d, DType::kF32, {T, proj});
    const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
    DBuf dqsl(d, DType::kI32, {2}, qsl);
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor vc3 = MakeTensor(vc.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor gr3 = MakeTensor(g1.ptr(), DType::kF32, d.q.device, {T, nh, hd});  // RAW gate proj
    Tensor out3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));
    vt::KdaChunkPrefill(d.q, out3, qn3, kn3, vc3, gr3, dbeta.t(), da_log.t(), ddt.t(), dstate.t(),
                        dqsl.t(), vt::GdnArgs{scale});
    return dcore;
  }

  if (DeviceKda()) {
    std::vector<float> dhg1(static_cast<size_t>(T) * proj),
        dhbraw(static_cast<size_t>(T) * nh);
    g1.Download(d, dhg1.data());
    braw.Download(d, dhbraw.data());
    RoundHostBf16(dhg1);   // honor BF16_ISLANDS on the gate inputs
    RoundHostBf16(dhbraw);
    const std::vector<float> gch =
        kimi_kda::KdaDecayGate(dhg1, a_log, dt_bias, T, nh, hd);  // [T,nh,hd] per-channel
    std::vector<float> hbeta(static_cast<size_t>(T) * nh);
    for (size_t i = 0; i < hbeta.size(); ++i)
      hbeta[i] = static_cast<float>(Sigmoid(dhbraw[i]));
    DBuf dg(d, DType::kF32, {T, nh, hd}, gch.data());
    DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());
    DBuf dstate(d, DType::kF32, {1, nh, hd, hd});
    dstate.Zero(d);
    DBuf dcore(d, DType::kF32, {T, proj});
    const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
    DBuf dqsl(d, DType::kI32, {2}, qsl);
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor vc3 = MakeTensor(vc.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor out3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));
    vt::KdaGatedDeltaRule(d.q, out3, qn3, kn3, vc3, dg.t(), dbeta.t(), dstate.t(), dqsl.t(),
                          vt::GdnArgs{scale});
    return dcore;
  }

  std::vector<float> hqn(static_cast<size_t>(T) * proj), hkn(hqn.size()),
      hv(hqn.size()), hg1(hqn.size()), hbraw(static_cast<size_t>(T) * nh);
  qn.Download(d, hqn.data());
  kn.Download(d, hkn.data());
  vc.Download(d, hv.data());
  g1.Download(d, hg1.data());
  braw.Download(d, hbraw.data());
  // VT_KIMI_BF16_ISLANDS: feed bf16-precision inputs to the recurrence (like vLLM's
  // GDN kernel), keeping the f64 accumulation. No-op when the knob is off.
  RoundHostBf16(hqn);
  RoundHostBf16(hkn);
  RoundHostBf16(hv);
  RoundHostBf16(hg1);
  RoundHostBf16(hbraw);

  const std::vector<float> g =
      kimi_kda::KdaDecayGate(hg1, a_log, dt_bias, T, nh, hd);  // [T,nh,hd]
  const double scale = std::pow(static_cast<double>(hd), -0.5);
  std::vector<double> S(static_cast<size_t>(nh) * hd * hd, 0.0);
  std::vector<float> core(static_cast<size_t>(T) * proj, 0.0f);
  std::vector<double> u(static_cast<size_t>(hd));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < nh; ++h) {
      const int64_t base = t * proj + h * hd;
      const float* qnp = &hqn[static_cast<size_t>(base)];
      const float* knp = &hkn[static_cast<size_t>(base)];
      const float* vvp = &hv[static_cast<size_t>(base)];
      const float* gh = &g[static_cast<size_t>(base)];
      const double b = Sigmoid(hbraw[static_cast<size_t>(t * nh + h)]);
      double* Sp = &S[static_cast<size_t>(h) * hd * hd];
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        for (int64_t k = 0; k < hd; ++k) Sr[k] = AccR(Sr[k] * std::exp(static_cast<double>(gh[k])));
      }
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double pred = 0.0;
        for (int64_t k = 0; k < hd; ++k) pred = AccR(pred + Sr[k] * knp[k]);
        u[static_cast<size_t>(vd)] = AccR((static_cast<double>(vvp[vd]) - pred) * b);
      }
      for (int64_t vd = 0; vd < hd; ++vd) {
        double* Sr = &Sp[vd * hd];
        const double uv = u[static_cast<size_t>(vd)];
        for (int64_t k = 0; k < hd; ++k) Sr[k] = AccR(Sr[k] + uv * knp[k]);
      }
      float* cr = &core[static_cast<size_t>(base)];
      for (int64_t vd = 0; vd < hd; ++vd) {
        const double* Sr = &Sp[vd * hd];
        double o = 0.0;
        for (int64_t k = 0; k < hd; ++k)
          o = AccR(o + Sr[k] * (static_cast<double>(qnp[k]) * scale));
        cr[vd] = static_cast<float>(o);  // f32 output (bf16-rounding the output MEASURED -14, reverted)
      }
    }
  }
  return DBuf(d, DType::kF32, {T, proj}, core.data());  // upload back to device
}

// ── DEVICE MLA attention CORE (VT_KIMI_DEVICE_MLA): the NoPE causal softmax over the
// per-head k_nope|k_pe / v, run through the shared device op vt::Attention (f32 online
// softmax — vLLM's FA2 regime) instead of the f64 host recompute. MLA has asymmetric
// head dims (qk = qk_nope+qk_rope, v = v_head_dim), which vt::Attention (one head-dim D
// for q/k/v) does not express, so the value is PADDED to qk with zeros: the weighted sum
// over the zero tail is 0, so out[:, :, :vh] is the exact attention core (softmax weights
// depend only on q·k, which is unaffected by the padded v). q is already laid out per head
// as [q_nope(qn) | q_pe(qr)] so dq views directly as [T,nah,qk]; key is built per (t,h) as
// [k_nope(qn) | k_pe(qr, SHARED across heads)] and value as [v(vh) | 0]. Scale = qk^-0.5,
// matching the host island / kimi_linear_forward.cpp:223. Returns [T, nah*v_head_dim].
DBuf MlaAttnCoreDevice(const Dev& d, DBuf& dq, DBuf& dkv, DBuf& dkpe,
                       const KimiLinearParams& p, int64_t T) {
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (qn + vh);

  DBuf key(d, DType::kF32, {T, nah, qk});
  DBuf val(d, DType::kF32, {T, nah, qk});
  val.Zero(d);  // pad-V: the [vh, qk) tail stays 0
  {
    const size_t qkb = static_cast<size_t>(qk) * sizeof(float);
    const size_t qnb = static_cast<size_t>(qn) * sizeof(float);
    const size_t qrb = static_cast<size_t>(qr) * sizeof(float);
    const size_t vhb = static_cast<size_t>(vh) * sizeof(float);
    const char* kv = static_cast<const char*>(dkv.ptr());
    const char* kpe = static_cast<const char*>(dkpe.ptr());
    char* kp = static_cast<char*>(key.ptr());
    char* vp = static_cast<char*>(val.ptr());
    for (int64_t t = 0; t < T; ++t) {
      const char* kpe_t = kpe + static_cast<size_t>(t) * qrb;
      for (int64_t h = 0; h < nah; ++h) {
        const char* src =
            kv + (static_cast<size_t>(t) * kvw + static_cast<size_t>(h) * (qn + vh)) *
                     sizeof(float);
        char* kdst = kp + (static_cast<size_t>(t) * nah + h) * qkb;
        char* vdst = vp + (static_cast<size_t>(t) * nah + h) * qkb;
        d.b.Copy(d.q, kdst, src, qnb);          // k_nope[qn]
        d.b.Copy(d.q, kdst + qnb, kpe_t, qrb);  // k_pe[qr] (shared across heads)
        d.b.Copy(d.q, vdst, src + qnb, vhb);    // v[vh]; the [vh,qk) tail stays 0
      }
    }
  }
  // VT_KIMI_BF16_ISLANDS: round the attention inputs to bf16 precision on device
  // (matching the host island's RoundHostBf16), before the f32 softmax.
  if (Bf16Islands()) {
    RoundDevBf16(d, dq);
    RoundDevBf16(d, key);
    RoundDevBf16(d, val);
  }
  Tensor query = MakeTensor(dq.ptr(), DType::kF32, d.q.device, {T, nah, qk});
  DBuf attn(d, DType::kF32, {T, nah, qk});
  const float scale = static_cast<float>(std::pow(static_cast<double>(qk), -0.5));
  // VT-ATTN-NAIVE: the whole point of this arm is that the attention core runs on
  // the SAME f32 online max-subtracted softmax as the host reference, as the
  // MlaAttnCoreDevice design note at the top of this file records, and it is
  // behind VT_KIMI_DEVICE_MLA, default OFF, recorded there as a measured
  // negative (4.24 -> 3.89 tok/s). No fast rung is available here either: the
  // padded head_dim is qk=192 in f32, and vt::AttentionDenseFlash would need 96 KB
  // of dynamic shared memory, twice CUDA's default cap (#1544, cuda_ops.cu).
  vt::Attention(d.q, attn.t(), query, key.t(), val.t(), vt::AttentionArgs{scale, true});

  // slice out[:, :, :vh] -> [T, nah*vh] (the pad-V tail is 0 by construction).
  DBuf out(d, DType::kF32, {T, nah * vh});
  {
    const size_t qkb = static_cast<size_t>(qk) * sizeof(float);
    const size_t vhb = static_cast<size_t>(vh) * sizeof(float);
    const char* ap = static_cast<const char*>(attn.ptr());
    char* op = static_cast<char*>(out.ptr());
    for (int64_t t = 0; t < T; ++t)
      for (int64_t h = 0; h < nah; ++h)
        d.b.Copy(d.q, op + (static_cast<size_t>(t) * nah + h) * vhb,
                 ap + (static_cast<size_t>(t) * nah + h) * qkb, vhb);
  }
  return out;
}

// ── HOST-FALLBACK ISLAND: the materialized-MHA attention CORE (causal softmax over
// the per-head k_nope|k_pe / v, NoPE so no RoPE). Identical math to kimi_linear_
// forward.cpp:223-258; shared by the f32 and bf16 MLA paths (only the projections
// feeding dq/dkv/dkpe differ). Returns [T, nah*v_head_dim]. When VT_KIMI_DEVICE_MLA is
// set, the attention core runs on device via vt::Attention (MlaAttnCoreDevice) instead.
DBuf MlaSoftmaxIsland(const Dev& d, DBuf& dq, DBuf& dkv, DBuf& dkpe,
                      const KimiLinearParams& p, int64_t T) {
  if (DeviceMla()) return MlaAttnCoreDevice(d, dq, dkv, dkpe, p, T);
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (qn + vh);
  std::vector<float> hq(static_cast<size_t>(T) * nah * qk),
      hkv(static_cast<size_t>(T) * kvw), hkpe(static_cast<size_t>(T) * qr);
  dq.Download(d, hq.data());
  dkv.Download(d, hkv.data());
  dkpe.Download(d, hkpe.data());
  // VT_KIMI_BF16_ISLANDS: bf16-precision inputs to the softmax core (like vLLM's FA2).
  RoundHostBf16(hq);
  RoundHostBf16(hkv);
  RoundHostBf16(hkpe);
  const double scale = std::pow(static_cast<double>(qk), -0.5);
  std::vector<float> out(static_cast<size_t>(T) * nah * vh, 0.0f);
  std::vector<double> sc(static_cast<size_t>(T));
  for (int64_t h = 0; h < nah; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const float* q_nope = &hq[static_cast<size_t>(t * nah * qk + h * qk)];
      const float* q_pe = q_nope + qn;
      double mx = -INFINITY;
      for (int64_t s = 0; s <= t; ++s) {
        const float* k_nope = &hkv[static_cast<size_t>(s * kvw + h * (qn + vh))];
        const float* kpe = &hkpe[static_cast<size_t>(s * qr)];
        double dot = 0.0;
        for (int64_t dd = 0; dd < qn; ++dd)
          dot = AccR(dot + static_cast<double>(q_nope[dd]) * k_nope[dd]);
        for (int64_t dd = 0; dd < qr; ++dd)
          dot = AccR(dot + static_cast<double>(q_pe[dd]) * kpe[dd]);
        dot = AccR(dot * scale);
        sc[static_cast<size_t>(s)] = dot;
        mx = std::max(mx, dot);
      }
      double sum = 0.0;
      for (int64_t s = 0; s <= t; ++s) {
        const double e = AccR(std::exp(sc[static_cast<size_t>(s)] - mx));
        sc[static_cast<size_t>(s)] = e;
        sum = AccR(sum + e);
      }
      float* ot = &out[static_cast<size_t>(t * nah * vh + h * vh)];
      for (int64_t dd = 0; dd < vh; ++dd) {
        double acc = 0.0;
        for (int64_t s = 0; s <= t; ++s) {
          const float* vs = &hkv[static_cast<size_t>(s * kvw + h * (qn + vh) + qn)];
          acc = AccR(acc + (sc[static_cast<size_t>(s)] / sum) * static_cast<double>(vs[dd]));
        }
        ot[dd] = static_cast<float>(acc);  // f32 output (bf16-rounding the output MEASURED -14, reverted)
      }
    }
  }
  return DBuf(d, DType::kF32, {T, nah * vh}, out.data());  // upload
}

// ─── (1) KDA linear-attention layer (device + one host-fallback island) ───────
// Grounding kimi_linear_forward.cpp:101-190 (the W2 reference this must match).
DBuf KdaLayerDevice(const Dev& d, const KdaLayerHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;

  // q/k/v projections -> silu short convs (ON DEVICE).
  DBuf rq(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rq.t(), dh, WF32(d, w.q_proj, {proj, H}));
  DBuf rk(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rk.t(), dh, WF32(d, w.k_proj, {proj, H}));
  DBuf rv(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, rv.t(), dh, WF32(d, w.v_proj, {proj, H}));
  DBuf qc = ConvSilu(d, rq.t(), w.q_conv, T, proj, K);
  DBuf kc = ConvSilu(d, rk.t(), w.k_conv, T, proj, K);
  DBuf vc = ConvSilu(d, rv.t(), w.v_conv, T, proj, K);  // v (no L2-norm)

  // per-head q/k L2-norm over head_dim (ON DEVICE) — view [T,proj] as [T*nh,hd].
  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc3 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc3 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn3, qc3, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn3, kc3, vt::L2NormArgs{1e-6f});
  }

  // beta_raw = b_proj(x); low-rank decay g1 = f_b(f_a(x)); gate g2 = g_b(g_a(x))
  // (ON DEVICE). The sigmoid(beta_raw) + the decay gate g live in the island.
  DBuf braw(d, DType::kF32, {T, nh});
  vt::MatmulBT(d.q, braw.t(), dh, WF32(d, w.b_proj, {nh, H}));
  DBuf fa(d, DType::kF32, {T, hd});
  vt::MatmulBT(d.q, fa.t(), dh, WF32(d, w.f_a_proj, {hd, H}));
  DBuf g1(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, g1.t(), fa.t(), WF32(d, w.f_b_proj, {proj, hd}));
  DBuf ga(d, DType::kF32, {T, hd});
  vt::MatmulBT(d.q, ga.t(), dh, WF32(d, w.g_a_proj, {hd, H}));
  DBuf g2(d, DType::kF32, {T, proj});
  vt::MatmulBT(d.q, g2.t(), ga.t(), WF32(d, w.g_b_proj, {proj, hd}));

  // HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta recurrence
  // (shared KdaRecurrenceIsland — see its definition above). THE W7-speed residual.
  DBuf dcore = KdaRecurrenceIsland(d, qn, kn, vc, g1, braw, w.a_log, w.dt_bias, p, T);

  // sigmoid-gated output RMSNorm then o_proj (ON DEVICE).
  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, out.t(), dcn.t(), WF32(d, w.o_proj, {H, proj}));
  return out;
}

// ─── (2) NoPE-MLA full-attention layer (device projections + host attn core) ──
// Grounding kimi_linear_forward.cpp:193-260.
DBuf MlaLayerDevice(const Dev& d, const MlaLayerHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);

  // q_proj, kv_a projections (ON DEVICE).
  DBuf dq(d, DType::kF32, {T, nah * qk});
  vt::MatmulBT(d.q, dq.t(), dh, WF32(d, w.q_proj, {nah * qk, H}));
  DBuf dlat(d, DType::kF32, {T, L + qr});
  vt::MatmulBT(d.q, dlat.t(), dh, WF32(d, w.kv_a_proj_with_mqa, {L + qr, H}));

  // Split latent -> kv_c[T,L] (normed), k_pe[T,qr] (shared, not normed) via device
  // row column-slice copies.
  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl, static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  // kv_a_layernorm (ON DEVICE, non-residual) then kv_b_proj (ON DEVICE).
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});
  DBuf dkv(d, DType::kF32, {T, kvw});
  vt::MatmulBT(d.q, dkv.t(), dkvcn.t(), WF32(d, w.kv_b_proj, {kvw, L}));

  // HOST-FALLBACK ISLAND: the materialized-MHA attention core (causal softmax, NoPE)
  // via the shared MlaSoftmaxIsland (see its definition above). The device path is
  // mla::ForwardMlaAttentionBlock over the runner's paged KV — the born-on-runner
  // residual. Identical math to kimi_linear_forward.cpp:223-258.
  DBuf dout = MlaSoftmaxIsland(d, dq, dkv, dkpe, p, T);
  DBuf attn(d, DType::kF32, {T, H});
  vt::MatmulBT(d.q, attn.t(), dout.t(), WF32(d, w.o_proj, {H, nah * vh}));
  return attn;
}

// ─── (3) sigmoid noaux_tc MoE block (device router+experts+combine) ───────────
// Grounding kimi_linear_forward.cpp:263-350 + deepseek_v2.cpp:331-472 MoeBlock.
DBuf MoeBlockDevice(const Dev& d, const MoeHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t k = p.num_experts_per_token;
  const int64_t I = p.moe_intermediate_size;

  // router: logits = gate(x) then grouped sigmoid top-k (ON DEVICE).
  DBuf dlog(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, dlog.t(), dh, WF32(d, w.gate, {E, H}));
  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(k);
  args.renormalize = p.moe_renormalize;
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.num_expert_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);
  DBuf dtw(d, DType::kF32, {T, k});
  DBuf dtid(d, DType::kI32, {T, k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.empty())
    bias = std::make_unique<Tensor>(WF32(d, w.e_score_correction_bias, {E}));
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // routed experts — the CPU-lane per-expert gather / SwiGLU / scatter (the grouped
  // CUDA GEMM is CUDA-only; deepseek_v2.cpp:414-455 REFERENCE path, f32 + MatmulBT).
  DBuf expert_out(d, DType::kF32, {T, k, H});
  expert_out.Zero(d);
  std::vector<int32_t> ids(static_cast<size_t>(T) * k);
  dtid.Download(d, ids.data());
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t * k + j)])].push_back({t, j});
  const size_t row_bytes = static_cast<size_t>(H) * sizeof(float);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, DType::kF32, {n, H});
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
               static_cast<const char*>(dh.data) +
                   static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
               row_bytes);
    const MlpHostWeights& ex = w.experts[static_cast<size_t>(e)];
    DBuf y = SwiGluDevice(d, ex.gate_proj, ex.up_proj, ex.down_proj, xg.t(), H, I, n);
    for (int64_t r = 0; r < n; ++r) {
      const auto& tj = list[static_cast<size_t>(r)];
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(tj.first * k + tj.second) * row_bytes,
               static_cast<const char*>(y.ptr()) + static_cast<size_t>(r) * row_bytes,
               row_bytes);
    }
  }

  // shared expert (always, added to the routed sum) + weighted combine (ON DEVICE).
  DBuf out(d, DType::kF32, {T, H});
  if (w.has_shared) {
    const int64_t shared_i = I * p.num_shared_experts;
    DBuf shared = SwiGluDevice(d, w.shared.gate_proj, w.shared.up_proj,
                               w.shared.down_proj, dh, H, shared_i, T);
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

// ─── (4) dense layer-0 SwiGLU MLP (ON DEVICE) ─────────────────────────────────
DBuf DenseMlpDevice(const Dev& d, const MlpHostWeights& w, const Tensor& dh,
                    const KimiLinearParams& p, int64_t T) {
  return SwiGluDevice(d, w.gate_proj, w.up_proj, w.down_proj, dh, p.hidden_size,
                      p.intermediate_size, T);
}

// Wrap [rows,vocab] f32 device logits as a DEVICE-RESIDENT ForwardLogits — verbatim
// the kimi_linear.cpp / deepseek_v2.cpp:633 seam (return the pooled block to the
// shared DevicePool via the shared_ptr deleter; expose the [rows,vocab] view).
ForwardLogits WrapDeviceLogits(DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  fl.device_storage = dlogits.ReleaseShared();
  return fl;
}

// The whole device-compute forward over a single token sequence — the pre-norm
// residual stream (kimi_linear_forward.cpp HostForwardSeq / deepseek_v2.cpp
// ForwardBody). Returns the DEVICE-RESIDENT [rows,vocab] f32 logits DBuf.
DBuf DeviceForwardBody(const Dev& d, const KimiLinearWeights& weights,
                       const std::vector<int32_t>& token_ids,
                       const std::vector<int32_t>& logits_indices) {
  const KimiLinearHostWeights& host = weights.host;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "KimiLinear device compute: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "KimiLinear device compute: host layer count != num_hidden_layers");

  // embed -> residual-stream delta (ON DEVICE).
  DBuf hidden(d, DType::kF32, {T, H});
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor htab = WF32(d, host.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    vt::Embedding(d.q, hh, htab, dids.t());
  }
  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerHostWeights& lw = host.layers[static_cast<size_t>(l)];
    DBuf dhn(d, DType::kF32, {T, H});
    AddRmsNorm(d, dhn, hcur, WF32(d, lw.input_layernorm, {H}), res, eps);
    DBuf attn = lw.is_kda ? KdaLayerDevice(d, lw.kda, dhn.t(), p, T)
                          : MlaLayerDevice(d, lw.mla, dhn.t(), p, T);
    DBuf dh2(d, DType::kF32, {T, H});
    AddRmsNorm(d, dh2, attn.t(), WF32(d, lw.post_attention_layernorm, {H}), res, eps);
    DBuf mlp = lw.is_moe ? MoeBlockDevice(d, lw.moe, dh2.t(), p, T)
                         : DenseMlpDevice(d, lw.dense, dh2.t(), p, T);
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, DType::kF32, {T, H});
  AddRmsNorm(d, dnorm, hcur, WF32(d, host.final_norm, {H}), res, eps);

  // logits_indices gather-before-lm_head, in REQUEST order (mirrors the reference's
  // `want` construction — gather whenever indices are given).
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kF32,
               logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H});
  if (!logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * sizeof(float);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < logits_indices.size(); ++i) {
      const int32_t idx = logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T,
               "KimiLinear device compute: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || host.lm_head.empty();
  Tensor lm = tied ? WF32(d, host.embed_tokens, {V, H}) : WF32(d, host.lm_head, {V, H});
  DBuf logits(d, DType::kF32, {n_out, V});
  vt::MatmulBT(d.q, logits.t(), src, lm);
  return logits;
}

// ═══ bf16-RESIDENT device COMPUTE (§13) — the FULL-model device forward ═════════
// Byte-for-byte the f32 structure above, with each of the ~20 projection GEMMs
// swapped from WF32+MatmulBT (f32 host alias) to GemmBf16 over the bf16-resident
// OwnedTensor (cast the f32 activation to bf16, MatmulBT bf16xbf16->f32 — vLLM's own
// projection numerics). The two host-fallback islands (KdaRecurrenceIsland /
// MlaSoftmaxIsland) and every small-vector op (short convs, norms, L2Norm,
// RmsNormGated, router topk, weighted combine, SwiGLU activation) are UNCHANGED (f32
// activations; the tiny vectors alias host f32 via WF32). The residual stream stays
// f32. Same call graph as the f32 path, so the tiny-config gate proves the exact
// wiring the full model runs.
DBuf SwiGluDeviceBf16(const Dev& d, const OwnedTensor& gate, const OwnedTensor& up,
                      const OwnedTensor& down, const Tensor& dh, int64_t H, int64_t I,
                      int64_t T) {
  DBuf dg(d, DType::kF32, {T, I});
  GemmBf16(d, dg.t(), dh, gate, I, H);
  DBuf du(d, DType::kF32, {T, I});
  GemmBf16(d, du.t(), dh, up, I, H);
  DBuf da(d, DType::kF32, {T, I});
  vt::MoeSiluMul(d.q, da.t(), dg.t(), du.t());
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), da.t(), down, H, I);
  return out;
}

DBuf KdaLayerDeviceBf16(const Dev& d, const KdaResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;

  // q/k/v projections -> silu short convs.
  DBuf rq(d, DType::kF32, {T, proj});
  GemmBf16(d, rq.t(), dh, w.q_proj, proj, H);
  DBuf rk(d, DType::kF32, {T, proj});
  GemmBf16(d, rk.t(), dh, w.k_proj, proj, H);
  DBuf rv(d, DType::kF32, {T, proj});
  GemmBf16(d, rv.t(), dh, w.v_proj, proj, H);
  DBuf qc = ConvSilu(d, rq.t(), w.q_conv, T, proj, K);
  DBuf kc = ConvSilu(d, rk.t(), w.k_conv, T, proj, K);
  DBuf vc = ConvSilu(d, rv.t(), w.v_conv, T, proj, K);  // v (no L2-norm)

  // per-head q/k L2-norm over head_dim — view [T,proj] as [T*nh,hd].
  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc3 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc3 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn3, qc3, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn3, kc3, vt::L2NormArgs{1e-6f});
  }

  DBuf braw(d, DType::kF32, {T, nh});
  GemmBf16(d, braw.t(), dh, w.b_proj, nh, H);
  DBuf fa(d, DType::kF32, {T, hd});
  GemmBf16(d, fa.t(), dh, w.f_a_proj, hd, H);
  DBuf g1(d, DType::kF32, {T, proj});
  GemmBf16(d, g1.t(), fa.t(), w.f_b_proj, proj, hd);
  DBuf ga(d, DType::kF32, {T, hd});
  GemmBf16(d, ga.t(), dh, w.g_a_proj, hd, H);
  DBuf g2(d, DType::kF32, {T, proj});
  GemmBf16(d, g2.t(), ga.t(), w.g_b_proj, proj, hd);

  // HOST-FALLBACK ISLAND: KDA decay gate + per-k-channel gated-delta recurrence.
  DBuf dcore = KdaRecurrenceIsland(d, qn, kn, vc, g1, braw, w.a_log, w.dt_bias, p, T);

  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), dcn.t(), w.o_proj, H, proj);
  return out;
}

DBuf MlaLayerDeviceBf16(const Dev& d, const MlaResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);

  DBuf dq(d, DType::kF32, {T, nah * qk});
  GemmBf16(d, dq.t(), dh, w.q_proj, nah * qk, H);
  DBuf dlat(d, DType::kF32, {T, L + qr});
  GemmBf16(d, dlat.t(), dh, w.kv_a_proj_with_mqa, L + qr, H);

  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl, static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});
  DBuf dkv(d, DType::kF32, {T, kvw});
  GemmBf16(d, dkv.t(), dkvcn.t(), w.kv_b_proj, kvw, L);

  DBuf dout = MlaSoftmaxIsland(d, dq, dkv, dkpe, p, T);
  DBuf attn(d, DType::kF32, {T, H});
  GemmBf16(d, attn.t(), dout.t(), w.o_proj, H, nah * vh);
  return attn;
}

DBuf MoeBlockDeviceBf16(const Dev& d, const MoeResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t k = p.num_experts_per_token;
  const int64_t I = p.moe_intermediate_size;

  // router: logits = gate(x) (bf16, like vLLM) then grouped sigmoid top-k.
  DBuf dlog(d, DType::kF32, {T, E});
  GemmBf16(d, dlog.t(), dh, w.gate, E, H);
  vt::MoeRouterTopKArgs args{};
  args.top_k = static_cast<int>(k);
  args.renormalize = p.moe_renormalize;
  args.scoring_func = vt::MoeScoringFunc::kSigmoid;
  args.num_expert_group = static_cast<int>(p.num_expert_group);
  args.topk_group = static_cast<int>(p.topk_group);
  args.routed_scaling_factor = static_cast<float>(p.routed_scaling_factor);
  DBuf dtw(d, DType::kF32, {T, k});
  DBuf dtid(d, DType::kI32, {T, k});
  std::unique_ptr<Tensor> bias;
  if (!w.e_score_correction_bias.empty())
    bias = std::make_unique<Tensor>(WF32(d, w.e_score_correction_bias, {E}));
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(), args, bias.get());

  // routed experts — per-expert gather / SwiGLU / scatter (bf16 GEMMs).
  DBuf expert_out(d, DType::kF32, {T, k, H});
  expert_out.Zero(d);
  std::vector<int32_t> ids(static_cast<size_t>(T) * k);
  dtid.Download(d, ids.data());
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t * k + j)])].push_back({t, j});
  // Gather per-expert rows in the STREAM dtype (bf16 stream: dh is bf16, so the row
  // stride is the bf16 width and xg is bf16 — SwiGluDeviceBf16's GemmBf16 reads it
  // straight; f32 stream: unchanged f32 gather). Grounding deepseek_v2.cpp MoeBlock.
  const size_t row_bytes = static_cast<size_t>(H) * vt::SizeOf(dh.dtype);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    DBuf xg(d, dh.dtype, {n, H});
    for (int64_t r = 0; r < n; ++r)
      d.b.Copy(d.q, static_cast<char*>(xg.ptr()) + static_cast<size_t>(r) * row_bytes,
               static_cast<const char*>(dh.data) +
                   static_cast<size_t>(list[static_cast<size_t>(r)].first) * row_bytes,
               row_bytes);
    const MlpResidentWeights& ex = w.experts[static_cast<size_t>(e)];
    DBuf y = SwiGluDeviceBf16(d, ex.gate_proj, ex.up_proj, ex.down_proj, xg.t(), H, I, n);
    for (int64_t r = 0; r < n; ++r) {
      const auto& tj = list[static_cast<size_t>(r)];
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(tj.first * k + tj.second) * row_bytes,
               static_cast<const char*>(y.ptr()) + static_cast<size_t>(r) * row_bytes,
               row_bytes);
    }
  }

  // shared expert (always) + weighted combine.
  DBuf out(d, DType::kF32, {T, H});
  if (w.has_shared) {
    const int64_t shared_i = I * p.num_shared_experts;
    DBuf shared = SwiGluDeviceBf16(d, w.shared.gate_proj, w.shared.up_proj,
                                   w.shared.down_proj, dh, H, shared_i, T);
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared.t());
  } else {
    vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), nullptr);
  }
  return out;
}

DBuf DenseMlpDeviceBf16(const Dev& d, const MlpResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T) {
  return SwiGluDeviceBf16(d, w.gate_proj, w.up_proj, w.down_proj, dh, p.hidden_size,
                          p.intermediate_size, T);
}

DBuf DeviceForwardBodyBf16(const Dev& d, const KimiLinearWeights& weights,
                           const std::vector<int32_t>& token_ids,
                           const std::vector<int32_t>& logits_indices) {
  const KimiLinearResidentWeights& rw = weights.resident;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "KimiLinear bf16 device compute: empty token sequence");
  VT_CHECK(static_cast<int64_t>(rw.layers.size()) == L,
           "KimiLinear bf16 device compute: resident layer count != num_hidden_layers");

  // The residual-stream dtype: bf16 END-TO-END (VT_KIMI_BF16_STREAM, vLLM's regime,
  // mirroring deepseek_v2.cpp:479-615) or f32 (default). The bf16 stream removes the
  // per-GEMM CastBf16 (projections read the bf16 stream directly) AND rounds the
  // residual add/norm to bf16 at every layer boundary (the p7-STRICT lever); the
  // partial VT_KIMI_BF16_RESIDUAL RoundDevBf16 is then redundant and skipped.
  const DType sdt = StreamDType();
  const bool round_res = Bf16Residual() && sdt == DType::kF32;

  // embed (bf16 table -> stream out) -> residual stream.
  DBuf hidden(d, sdt, {T, H});
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor htab = ResidentBf16W(d, rw.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    vt::Embedding(d.q, hh, htab, dids.t());
  }
  if (round_res) RoundDevBf16(d, hidden);  // vLLM embed output is bf16
  DBuf res(d, sdt, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerResidentWeights& lw = rw.layers[static_cast<size_t>(l)];
    DBuf dhn(d, sdt, {T, H});
    AddRmsNormS(d, dhn, hcur, lw.input_layernorm, H, res, eps, sdt);
    if (round_res) RoundDevBf16(d, res);
    DBuf attn = ToStream(d,
                         lw.is_kda ? KdaLayerDeviceBf16(d, lw.kda, dhn.t(), p, T)
                                   : MlaLayerDeviceBf16(d, lw.mla, dhn.t(), p, T),
                         sdt);
    if (round_res) RoundDevBf16(d, attn);
    DBuf dh2(d, sdt, {T, H});
    AddRmsNormS(d, dh2, attn.t(), lw.post_attention_layernorm, H, res, eps, sdt);
    if (round_res) RoundDevBf16(d, res);
    DBuf mlp = ToStream(d,
                        lw.is_moe ? MoeBlockDeviceBf16(d, lw.moe, dh2.t(), p, T)
                                  : DenseMlpDeviceBf16(d, lw.dense, dh2.t(), p, T),
                        sdt);
    if (round_res) RoundDevBf16(d, mlp);
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, sdt, {T, H});
  AddRmsNormS(d, dnorm, hcur, rw.final_norm, H, res, eps, sdt);

  // logits_indices gather-before-lm_head, in REQUEST order.
  Tensor src = dnorm.t();
  DBuf dgather(d, sdt,
               logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H});
  if (!logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * vt::SizeOf(sdt);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < logits_indices.size(); ++i) {
      const int32_t idx = logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T,
               "KimiLinear bf16 device compute: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || rw.lm_head.Empty();
  const OwnedTensor& lm = tied ? rw.embed_tokens : rw.lm_head;
  DBuf logits(d, DType::kF32, {n_out, V});
  GemmBf16(d, logits.t(), src, lm, V, H);
  return logits;
}

// ═══ PAGED-INCREMENTAL DECODE (§18 lever e) ═════════════════════════════════════
// The paged-incremental twin of DeviceForwardBodyBf16: instead of re-running the
// whole [0..prompt+t] sequence every step (the O(n²) recompute vehicle — 4.24 tok/s),
// PREFILL the prompt ONCE (capturing the KDA recurrent+conv state per KDA layer and
// the latent-KV per NoPE-MLA layer into a persistent KimiDecodeCache) then advance one
// token per step from the CARRIED state — vLLM's decode regime
// (kimi_gdn_linear_attn.py prefill=chunk / decode=recurrent). The per-layer/per-token
// compute is byte-IDENTICAL to the recompute path (same vt:: ops, same reduction
// orders); the ONLY structural change is that the KDA recurrence / short conv carry
// their state and the MLA attention reads a growing KV cache instead of re-projecting
// the whole prefix. That makes the incremental path token-EXACT vs ForwardDeviceCompute
// at the same numeric config (the token-identity gate) while doing O(1) projection/MoE
// work per step instead of O(n).

// Short conv (silu) with host-persistent tap carry (mamba conv decode). Prefill: fresh
// zero state (has_initial=0), capture the final K-1 taps into `state`. Decode: upload
// the carried `state` (has_initial=1), advance, rewrite it. Same CausalConv1dFwd op as
// the recompute ConvSilu, so byte-exact over the same token window.
DBuf ConvSiluInc(const Dev& d, const Tensor& x, const std::vector<float>& weight,
                 int64_t T, int64_t C, int64_t K, std::vector<float>& state,
                 bool is_prefill) {
  DBuf out(d, DType::kF32, {T, C});
  DBuf cs(d, DType::kF32, {1, C, K - 1});
  const bool carried = !is_prefill && !state.empty();
  if (carried)
    d.b.Copy(d.q, cs.ptr(), state.data(), state.size() * sizeof(float));
  else
    cs.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {carried ? 1 : 0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  vt::CausalConv1dFwd(d.q, out.t(), x, WF32(d, weight, {C, K}), nullptr, cs.t(), dqsl.t(),
                      dhis.t(), vt::CausalConv1dArgs{true});
  state.resize(static_cast<size_t>(C) * (K - 1));
  cs.Download(d, state.data());  // carry the final K-1 taps
  return out;
}

// KDA per-k-channel gated-delta recurrence with host-persistent recurrent-state carry.
// Prefill: fresh zero state, optional CHUNK path (vt::KdaChunkPrefill — vLLM's prefill,
// VT_KIMI_DEVICE_KDA_CHUNK=1) else the recurrence; capture the final state. Decode: the
// recurrence (vt::KdaGatedDeltaRule, T==1) from the carried state, rewrite it. The gate
// (g = -exp(A_log)*softplus(f_b(f_a(x))+dt_bias)) + beta = sigmoid(braw) are computed
// EXACTLY as the recompute KdaRecurrenceIsland device-KDA branch, so recurrence-prefill
// + recurrence-decode is byte-exact vs the recompute device-KDA path.
DBuf KdaRecurrenceIslandInc(const Dev& d, DBuf& qn, DBuf& kn, DBuf& vc, DBuf& g1,
                            DBuf& braw, const std::vector<float>& a_log,
                            const std::vector<float>& dt_bias, const KimiLinearParams& p,
                            int64_t T, std::vector<float>& rec_state, bool is_prefill,
                            bool use_chunk) {
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));

  DBuf dstate(d, DType::kF32, {1, nh, hd, hd});
  const bool carried = !is_prefill && !rec_state.empty();
  if (carried)
    d.b.Copy(d.q, dstate.ptr(), rec_state.data(), rec_state.size() * sizeof(float));
  else
    dstate.Zero(d);

  DBuf dcore(d, DType::kF32, {T, proj});
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
  Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
  Tensor vc3 = MakeTensor(vc.ptr(), DType::kF32, d.q.device, {T, nh, hd});
  Tensor out3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});

  // beta = sigmoid(braw), rounded like the recompute island (BF16_ISLANDS default off).
  std::vector<float> hbraw(static_cast<size_t>(T) * nh);
  braw.Download(d, hbraw.data());
  RoundHostBf16(hbraw);
  std::vector<float> hbeta(hbraw.size());
  for (size_t i = 0; i < hbeta.size(); ++i) hbeta[i] = static_cast<float>(Sigmoid(hbraw[i]));
  DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());

  if (use_chunk && is_prefill && T > 1) {
    // vLLM's PROMPT path: the chunk kernels fuse the gate on device from the RAW g1.
    DBuf da_log(d, DType::kF32, {nh}, a_log.data());
    DBuf ddt(d, DType::kF32, {static_cast<int64_t>(dt_bias.size())},
             dt_bias.empty() ? nullptr : dt_bias.data());
    Tensor gr3 = MakeTensor(g1.ptr(), DType::kF32, d.q.device, {T, nh, hd});  // RAW gate proj
    vt::KdaChunkPrefill(d.q, out3, qn3, kn3, vc3, gr3, dbeta.t(), da_log.t(), ddt.t(),
                        dstate.t(), dqsl.t(), vt::GdnArgs{scale});
  } else {
    std::vector<float> hg1(static_cast<size_t>(T) * proj);
    g1.Download(d, hg1.data());
    RoundHostBf16(hg1);
    const std::vector<float> gch =
        kimi_kda::KdaDecayGate(hg1, a_log, dt_bias, T, nh, hd);  // [T,nh,hd] per-channel
    DBuf dg(d, DType::kF32, {T, nh, hd}, gch.data());
    vt::KdaGatedDeltaRule(d.q, out3, qn3, kn3, vc3, dg.t(), dbeta.t(), dstate.t(), dqsl.t(),
                          vt::GdnArgs{scale});
  }
  rec_state.resize(static_cast<size_t>(nh) * hd * hd);
  dstate.Download(d, rec_state.data());  // carry the final recurrent state
  return dcore;
}

// NoPE-MLA causal-softmax attention over a GROWING host latent-KV cache — the paged
// incremental twin of MlaSoftmaxIsland. Each of the T query rows sits at global
// position base_pos + t and attends the cached keys [0 .. base_pos+t] (causal). Same
// f64 online-softmax math and same ascending-s reduction order as the recompute
// island, so byte-exact vs it: prefill (base_pos=0, T=P) reproduces the whole-sequence
// island; decode (base_pos=seq_len, T=1) attends the full carried prefix. `cache_kv`
// / `cache_kpe` already hold base_pos+T tokens (appended by the caller).
DBuf MlaSoftmaxIslandInc(const Dev& d, DBuf& dq, const std::vector<float>& cache_kv,
                         const std::vector<float>& cache_kpe, const KimiLinearParams& p,
                         int64_t T, int64_t base_pos) {
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (qn + vh);
  std::vector<float> hq(static_cast<size_t>(T) * nah * qk);
  dq.Download(d, hq.data());
  RoundHostBf16(hq);  // BF16_ISLANDS on the query (cache_kv/kpe rounded at append time)
  const double scale = std::pow(static_cast<double>(qk), -0.5);
  std::vector<float> out(static_cast<size_t>(T) * nah * vh, 0.0f);
  std::vector<double> sc;
  for (int64_t h = 0; h < nah; ++h) {
    for (int64_t t = 0; t < T; ++t) {
      const int64_t nkeys = base_pos + t + 1;  // causal: attends [0 .. base_pos+t]
      sc.assign(static_cast<size_t>(nkeys), 0.0);
      const float* q_nope = &hq[static_cast<size_t>(t * nah * qk + h * qk)];
      const float* q_pe = q_nope + qn;
      double mx = -INFINITY;
      for (int64_t s = 0; s < nkeys; ++s) {
        const float* k_nope = &cache_kv[static_cast<size_t>(s * kvw + h * (qn + vh))];
        const float* kpe = &cache_kpe[static_cast<size_t>(s * qr)];
        double dot = 0.0;
        for (int64_t dd = 0; dd < qn; ++dd)
          dot = AccR(dot + static_cast<double>(q_nope[dd]) * k_nope[dd]);
        for (int64_t dd = 0; dd < qr; ++dd)
          dot = AccR(dot + static_cast<double>(q_pe[dd]) * kpe[dd]);
        dot = AccR(dot * scale);
        sc[static_cast<size_t>(s)] = dot;
        mx = std::max(mx, dot);
      }
      double sum = 0.0;
      for (int64_t s = 0; s < nkeys; ++s) {
        const double e = AccR(std::exp(sc[static_cast<size_t>(s)] - mx));
        sc[static_cast<size_t>(s)] = e;
        sum = AccR(sum + e);
      }
      float* ot = &out[static_cast<size_t>(t * nah * vh + h * vh)];
      for (int64_t dd = 0; dd < vh; ++dd) {
        double acc = 0.0;
        for (int64_t s = 0; s < nkeys; ++s) {
          const float* vs = &cache_kv[static_cast<size_t>(s * kvw + h * (qn + vh) + qn)];
          acc = AccR(acc + (sc[static_cast<size_t>(s)] / sum) * static_cast<double>(vs[dd]));
        }
        ot[dd] = static_cast<float>(acc);
      }
    }
  }
  return DBuf(d, DType::kF32, {T, nah * vh}, out.data());
}

// KDA layer — incremental (state-carrying) form of KdaLayerDeviceBf16. Byte-identical
// projections/convs/L2norm/gated-norm; the convs and the recurrence carry their state
// through `cache`.
DBuf KdaLayerDeviceBf16Inc(const Dev& d, const KdaResidentWeights& w, const Tensor& dh,
                           const KimiLinearParams& p, int64_t T, KimiKdaLayerCache& cache,
                           bool is_prefill, bool use_chunk) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;

  DBuf rq(d, DType::kF32, {T, proj});
  GemmBf16(d, rq.t(), dh, w.q_proj, proj, H);
  DBuf rk(d, DType::kF32, {T, proj});
  GemmBf16(d, rk.t(), dh, w.k_proj, proj, H);
  DBuf rv(d, DType::kF32, {T, proj});
  GemmBf16(d, rv.t(), dh, w.v_proj, proj, H);
  DBuf qc = ConvSiluInc(d, rq.t(), w.q_conv, T, proj, K, cache.conv_q, is_prefill);
  DBuf kc = ConvSiluInc(d, rk.t(), w.k_conv, T, proj, K, cache.conv_k, is_prefill);
  DBuf vc = ConvSiluInc(d, rv.t(), w.v_conv, T, proj, K, cache.conv_v, is_prefill);

  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc3 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn3 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc3 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn3 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn3, qc3, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn3, kc3, vt::L2NormArgs{1e-6f});
  }

  DBuf braw(d, DType::kF32, {T, nh});
  GemmBf16(d, braw.t(), dh, w.b_proj, nh, H);
  DBuf fa(d, DType::kF32, {T, hd});
  GemmBf16(d, fa.t(), dh, w.f_a_proj, hd, H);
  DBuf g1(d, DType::kF32, {T, proj});
  GemmBf16(d, g1.t(), fa.t(), w.f_b_proj, proj, hd);
  DBuf ga(d, DType::kF32, {T, hd});
  GemmBf16(d, ga.t(), dh, w.g_a_proj, hd, H);
  DBuf g2(d, DType::kF32, {T, proj});
  GemmBf16(d, g2.t(), ga.t(), w.g_b_proj, proj, hd);

  DBuf dcore = KdaRecurrenceIslandInc(d, qn, kn, vc, g1, braw, w.a_log, w.dt_bias, p, T,
                                      cache.recurrent, is_prefill, use_chunk);

  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), dcn.t(), w.o_proj, H, proj);
  return out;
}

// NoPE-MLA layer — incremental form of MlaLayerDeviceBf16. Byte-identical projections;
// the per-token latent-KV (kv[kvw] | kpe[qr]) is appended to `cache` and the attention
// runs over the growing cache (query_len=T, key_len=base_pos+T).
DBuf MlaLayerDeviceBf16Inc(const Dev& d, const MlaResidentWeights& w, const Tensor& dh,
                           const KimiLinearParams& p, int64_t T, int64_t base_pos,
                           KimiMlaLayerCache& cache) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);

  DBuf dq(d, DType::kF32, {T, nah * qk});
  GemmBf16(d, dq.t(), dh, w.q_proj, nah * qk, H);
  DBuf dlat(d, DType::kF32, {T, L + qr});
  GemmBf16(d, dlat.t(), dh, w.kv_a_proj_with_mqa, L + qr, H);

  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl, static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});
  DBuf dkv(d, DType::kF32, {T, kvw});
  GemmBf16(d, dkv.t(), dkvcn.t(), w.kv_b_proj, kvw, L);

  // append the T tokens' kv[kvw] and kpe[qr] to the growing cache (bf16-round at append
  // time under BF16_ISLANDS, matching the recompute island's per-token rounding).
  std::vector<float> hkv(static_cast<size_t>(T) * kvw), hkpe(static_cast<size_t>(T) * qr);
  dkv.Download(d, hkv.data());
  dkpe.Download(d, hkpe.data());
  RoundHostBf16(hkv);
  RoundHostBf16(hkpe);
  cache.kv.insert(cache.kv.end(), hkv.begin(), hkv.end());
  cache.kpe.insert(cache.kpe.end(), hkpe.begin(), hkpe.end());

  DBuf dout = MlaSoftmaxIslandInc(d, dq, cache.kv, cache.kpe, p, T, base_pos);
  DBuf attn(d, DType::kF32, {T, H});
  GemmBf16(d, attn.t(), dout.t(), w.o_proj, H, nah * vh);
  return attn;
}

// The whole paged-incremental device forward over `token_ids` (prefill: the prompt at
// base_pos=0; decode: one token at base_pos=cache.seq_len), carrying `cache`. Returns
// the DEVICE-RESIDENT [rows,vocab] logits. Byte-for-byte the DeviceForwardBodyBf16
// residual-stream structure with the KDA/MLA layers swapped for their state-carrying
// Inc forms.
DBuf DeviceForwardBodyBf16Incremental(const Dev& d, const KimiLinearWeights& weights,
                                      const std::vector<int32_t>& token_ids,
                                      int64_t base_pos, KimiDecodeCache& cache,
                                      bool is_prefill,
                                      const std::vector<int32_t>& logits_indices) {
  const KimiLinearResidentWeights& rw = weights.resident;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "KimiLinear incremental: empty token sequence");
  VT_CHECK(rw.resident,
           "KimiLinear incremental: bf16-resident weights required (paged-incremental "
           "decode is the full-model device path)");
  VT_CHECK(static_cast<int64_t>(rw.layers.size()) == L,
           "KimiLinear incremental: resident layer count != num_hidden_layers");
  const bool use_chunk = DeviceKdaChunk();
  // Residual-stream dtype (VT_KIMI_BF16_STREAM), byte-for-byte the same treatment as
  // DeviceForwardBodyBf16 (recompute) so the paged-incremental path stays token-
  // identical to recompute (the Gate A / case-(l) byte-exact state-carry proof).
  const DType sdt = StreamDType();
  const bool round_res = Bf16Residual() && sdt == DType::kF32;

  DBuf hidden(d, sdt, {T, H});
  {
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    Tensor htab = ResidentBf16W(d, rw.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    vt::Embedding(d.q, hh, htab, dids.t());
  }
  if (round_res) RoundDevBf16(d, hidden);
  DBuf res(d, sdt, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  int64_t kda_idx = 0, mla_idx = 0;
  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerResidentWeights& lw = rw.layers[static_cast<size_t>(l)];
    DBuf dhn(d, sdt, {T, H});
    AddRmsNormS(d, dhn, hcur, lw.input_layernorm, H, res, eps, sdt);
    if (round_res) RoundDevBf16(d, res);
    DBuf attn = ToStream(
        d,
        lw.is_kda ? KdaLayerDeviceBf16Inc(d, lw.kda, dhn.t(), p, T,
                                          cache.kda[static_cast<size_t>(kda_idx++)],
                                          is_prefill, use_chunk)
                  : MlaLayerDeviceBf16Inc(d, lw.mla, dhn.t(), p, T, base_pos,
                                          cache.mla[static_cast<size_t>(mla_idx++)]),
        sdt);
    if (round_res) RoundDevBf16(d, attn);
    DBuf dh2(d, sdt, {T, H});
    AddRmsNormS(d, dh2, attn.t(), lw.post_attention_layernorm, H, res, eps, sdt);
    if (round_res) RoundDevBf16(d, res);
    DBuf mlp = ToStream(d,
                        lw.is_moe ? MoeBlockDeviceBf16(d, lw.moe, dh2.t(), p, T)
                                  : DenseMlpDeviceBf16(d, lw.dense, dh2.t(), p, T),
                        sdt);
    if (round_res) RoundDevBf16(d, mlp);
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, sdt, {T, H});
  AddRmsNormS(d, dnorm, hcur, rw.final_norm, H, res, eps, sdt);

  Tensor src = dnorm.t();
  DBuf dgather(d, sdt,
               logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{static_cast<int64_t>(logits_indices.size()), H});
  if (!logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * vt::SizeOf(sdt);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < logits_indices.size(); ++i) {
      const int32_t idx = logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T, "KimiLinear incremental: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || rw.lm_head.Empty();
  const OwnedTensor& lm = tied ? rw.embed_tokens : rw.lm_head;
  DBuf logits(d, DType::kF32, {n_out, V});
  GemmBf16(d, logits.t(), src, lm, V, H);
  return logits;
}

// ═══ ROW 7 — THE SHARED-PAGED-RUNNER FOLD (kimi-linear.md §20.3) ════════════════
// The born-on-the-runner PRODUCTION forward: byte-for-byte the
// DeviceForwardBodyBf16Incremental per-token compute with the single-sequence
// host KimiDecodeCache replaced by the runner's OWN paged state groups —
//   * KDA conv+recurrent state in the MambaSpec `gdn_state` group, keyed by
//     `gdn_meta.non_spec_state_indices_tensor` (mirror vLLM
//     kimi_gdn_linear_attn.py:296-440 `_forward`: `constant_caches` =
//     (conv_state, recurrent_state) indexed by non_spec_state_indices_tensor;
//     prefill = chunk_kda_with_fused_gate, decode = fused_recurrent_kda);
//   * NoPE-MLA latent-KV in the paged `attn_kv` MLA group, written through
//     vt::ConcatAndCacheMla at `attn_meta.slot_mapping` (the MLAAttentionSpec
//     page: ONE 576-wide latent row per token, mla_attention.py:553-620 order).
// Batches are decode-first (the GDN builder's segmentation): nd single-token
// decodes then np prefills. NOT GdnBlockPaged — KDA's per-K-channel decay
// g[T,Hv,Dk] needs the KDA ops (vt::KdaChunkPrefill / vt::KdaGatedDeltaRule);
// the shared per-head GDN kernels stay untouched (qwen3_5 byte-identical).

// VT_KIMI_PAGED_KDA_CHUNK (default ON) — process fresh prefill requests with the
// CHUNKED KDA kernel family (vt::KdaChunkPrefill — vLLM's prompt path and the
// §19 Gate-A-winning config); '0' falls back to the recurrence for A/B. Decode
// and continuing (has_initial_state) prefills always use the recurrence, exactly
// as vLLM (decode: fused_recurrent_kda; our chunk op takes a fresh zero state).
bool PagedKdaChunkEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_PAGED_KDA_CHUNK");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// VT_KIMI_PAGED_MLA_FA2 (default ON — GB10-RULED 2026-08-07): route the 7
// NoPE-MLA layers through mla::ForwardMlaAttentionBlock — vLLM's ACTUAL
// absorbed-MQA decode / FA2 prefill over the paged latent cache (§20.3c),
// identity RoPE, scale qk^-0.5. MEASURED on the real 48.9B (§21): this arm
// reproduces the golden's near-tie profile EXACTLY — 122/128 with p0-p6 16/16
// and p7 10/16, the same 122/128 the CLI reference and the §12/§19 batteries
// carry — so per the parity-enablers-ship-as-defaults policy it IS the
// production arm. '0' selects the diagnostic EXACT arm (the f64 softmax island
// over kv_b-up-projected paged rows — byte-comparable to the CLI on CPU, the
// fold-identity vehicle): on GB10 it measured 111/128, the §19-documented
// GPU M-dimension-tiling near-tie perturbation (re-up-projecting the whole
// prefix at M=S vs the CLI's M=T append-time GEMM flips near-tie tokens:
// p7 flips TOWARD golden 16/16, p4 one flip that recovers, p2's token-1 flip
// cascades to 0/16) — a numeric-regime difference, not a paging bug (the CPU
// gate is byte-exact and FA2 shares every projection/cache write).
// NOT memoized (a cheap getenv per layer): the CPU gates pin the arm per test
// case, and a per-process latch would weld the whole binary to one arm.
bool PagedMlaFa2() {
  const char* e = std::getenv("VT_KIMI_PAGED_MLA_FA2");
  return e == nullptr || e[0] != '0';
}

// Contiguous row-range view over a rank-N device tensor: rows [start, start+len).
inline Tensor RowsView(const Tensor& t, int64_t start, int64_t len,
                       const std::vector<int64_t>& shape) {
  VT_CHECK(!shape.empty() && shape[0] == len,
           "kimi paged: RowsView shape[0] must equal len");
  int64_t row = 1;
  for (int i = 1; i < t.rank; ++i) row *= t.shape[i];
  return MakeTensor(static_cast<char*>(t.data) +
                        static_cast<size_t>(start * row) * vt::SizeOf(t.dtype),
                    t.dtype, t.device, shape);
}

// The decode-first non-spec segmentation the GDN builder emits, validated for
// the Kimi paged forward (no spec rows — this checkpoint has no MTP head).
struct KimiPagedSeg {
  int nd = 0, np = 0;
  int64_t nd_tok = 0, np_tok = 0;
  const std::vector<int32_t>* sidx = nullptr;  // per-request GDN state slots
  const std::vector<int32_t>* qsl = nullptr;   // [nreq+1] cumulative offsets
  const std::vector<uint8_t>* his = nullptr;   // per-request has_initial (np>0)
};
KimiPagedSeg KimiSegment(const v1::GDNAttentionMetadata& gm, int64_t T) {
  KimiPagedSeg s;
  VT_CHECK(gm.num_spec_decodes == 0 && gm.num_spec_decode_tokens == 0,
           "kimi paged: spec-decode rows are not expressible (the 48B-Instruct "
           "checkpoint has no MTP head; num_nextn_predict_layers=0)");
  s.nd = gm.num_decodes;
  s.np = gm.num_prefills;
  s.nd_tok = gm.num_decode_tokens;
  s.np_tok = gm.num_prefill_tokens;
  VT_CHECK(s.nd_tok + s.np_tok == T, "kimi paged: decode+prefill tokens != T");
  VT_CHECK(s.nd_tok == s.nd, "kimi paged: decode segment must be 1 token/request");
  VT_CHECK(gm.non_spec_state_indices_tensor.has_value() &&
               gm.non_spec_query_start_loc.has_value(),
           "kimi paged: GDN metadata is missing state indices / query offsets");
  s.sidx = &*gm.non_spec_state_indices_tensor;
  s.qsl = &*gm.non_spec_query_start_loc;
  VT_CHECK(static_cast<int64_t>(s.sidx->size()) >= s.nd + s.np,
           "kimi paged: state index vector shorter than the batch");
  if (s.np > 0) {
    VT_CHECK(gm.has_initial_state.has_value() &&
                 static_cast<int64_t>(gm.has_initial_state->size()) >= s.nd + s.np,
             "kimi paged: prefill batch is missing has_initial_state");
    s.his = &*gm.has_initial_state;
  }
  return s;
}

// KDA layer over the PAGED conv+recurrent state (the paged form of
// KdaLayerDeviceBf16Inc). The projection/conv/L2/gate/gated-norm op sequence is
// byte-identical; only the state residency changes: conv taps + recurrent state
// are gathered from / scattered to the runner's `gdn_state` group rows named by
// the per-request state slots. Layout of one conv row: [q taps | k taps | v taps]
// each [proj, K-1] — vLLM's `conv_state.chunk(3)` (kimi_gdn_linear_attn.py:331).
DBuf KdaLayerPagedBf16(const Dev& d, const KdaResidentWeights& w, const Tensor& dh,
                       const KimiLinearParams& p, int64_t T, const KimiPagedSeg& seg,
                       const GdnStateCache& state) {
  const int64_t H = p.hidden_size;
  const int64_t nh = p.kda_num_heads;
  const int64_t hd = p.kda_head_dim;
  const int64_t proj = nh * hd;
  const int64_t K = p.kda_short_conv_kernel_size;
  const int64_t conv_dim = 3 * proj;
  const int64_t nreq = seg.nd + seg.np;
  const float scale = static_cast<float>(std::pow(static_cast<double>(hd), -0.5));

  VT_CHECK(state.conv_state.shape[1] == conv_dim &&
               state.conv_state.shape[2] == K - 1 &&
               state.ssm_state.shape[1] == nh && state.ssm_state.shape[2] == hd &&
               state.ssm_state.shape[3] == hd,
           "kimi paged: runner GDN state geometry disagrees with linear_attn_config");

  DBuf rq(d, DType::kF32, {T, proj});
  GemmBf16(d, rq.t(), dh, w.q_proj, proj, H);
  DBuf rk(d, DType::kF32, {T, proj});
  GemmBf16(d, rk.t(), dh, w.k_proj, proj, H);
  DBuf rv(d, DType::kF32, {T, proj});
  GemmBf16(d, rv.t(), dh, w.v_proj, proj, H);

  // ── conv (3 separate q/k/v short convs over the paged conv row) ──
  DBuf didx(d, DType::kI32, {nreq}, seg.sidx->data());
  DBuf dcs(d, DType::kF32, {nreq, conv_dim, K - 1});
  vt::GdnStateGather(d.q, dcs.t(), state.conv_state, didx.t());
  const size_t sec_bytes = static_cast<size_t>(proj) * (K - 1) * sizeof(float);
  const size_t row_bytes = static_cast<size_t>(conv_dim) * (K - 1) * sizeof(float);
  auto section_out = [&](int64_t sec) {
    DBuf s(d, DType::kF32, {nreq, proj, K - 1});
    for (int64_t r = 0; r < nreq; ++r)
      d.b.Copy(d.q, static_cast<char*>(s.ptr()) + static_cast<size_t>(r) * sec_bytes,
               static_cast<char*>(dcs.ptr()) + static_cast<size_t>(r) * row_bytes +
                   static_cast<size_t>(sec) * sec_bytes,
               sec_bytes);
    return s;
  };
  auto section_back = [&](DBuf& s, int64_t sec) {
    for (int64_t r = 0; r < nreq; ++r)
      d.b.Copy(d.q,
               static_cast<char*>(dcs.ptr()) + static_cast<size_t>(r) * row_bytes +
                   static_cast<size_t>(sec) * sec_bytes,
               static_cast<char*>(s.ptr()) + static_cast<size_t>(r) * sec_bytes,
               sec_bytes);
  };
  DBuf cs_q = section_out(0);
  DBuf cs_k = section_out(1);
  DBuf cs_v = section_out(2);

  DBuf qc(d, DType::kF32, {T, proj});
  DBuf kc(d, DType::kF32, {T, proj});
  DBuf vc(d, DType::kF32, {T, proj});
  if (seg.np > 0) {
    // Any prefill: varlen conv over the whole non-spec stream (decodes lead,
    // each a 1-token slice with has_initial=1) — qwen3_5.cpp's np>0 conv branch,
    // with Kimi's three separate convs in place of the merged one.
    std::vector<int32_t> his32(seg.his->begin(), seg.his->begin() + nreq);
    DBuf dqsl(d, DType::kI32, {nreq + 1}, seg.qsl->data());
    DBuf dhis(d, DType::kI32, {nreq}, his32.data());
    vt::CausalConv1dFwd(d.q, qc.t(), rq.t(), WF32(d, w.q_conv, {proj, K}), nullptr,
                        cs_q.t(), dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
    vt::CausalConv1dFwd(d.q, kc.t(), rk.t(), WF32(d, w.k_conv, {proj, K}), nullptr,
                        cs_k.t(), dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
    vt::CausalConv1dFwd(d.q, vc.t(), rv.t(), WF32(d, w.v_conv, {proj, K}), nullptr,
                        cs_v.t(), dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
  } else {
    // Pure decode: single-token conv step per sequence on the compact gathered
    // rows (mamba causal_conv1d_update; numerically the same window sum as
    // CausalConv1dFwd(T=1, has_initial=1) — the CLI ConvSiluInc form).
    vt::CausalConv1dUpdate(d.q, qc.t(), rq.t(), WF32(d, w.q_conv, {proj, K}), nullptr,
                           cs_q.t(), vt::CausalConv1dArgs{true});
    vt::CausalConv1dUpdate(d.q, kc.t(), rk.t(), WF32(d, w.k_conv, {proj, K}), nullptr,
                           cs_k.t(), vt::CausalConv1dArgs{true});
    vt::CausalConv1dUpdate(d.q, vc.t(), rv.t(), WF32(d, w.v_conv, {proj, K}), nullptr,
                           cs_v.t(), vt::CausalConv1dArgs{true});
  }
  section_back(cs_q, 0);
  section_back(cs_k, 1);
  section_back(cs_v, 2);
  {
    Tensor conv_cache = state.conv_state;
    vt::GdnStateScatter(d.q, conv_cache, dcs.t(), didx.t());
  }

  // ── post-conv: q/k L2 norm + the low-rank decay/gate projections (== Inc) ──
  DBuf qn(d, DType::kF32, {T, proj});
  DBuf kn(d, DType::kF32, {T, proj});
  {
    Tensor qc2 = MakeTensor(qc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor qn2 = MakeTensor(qn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kc2 = MakeTensor(kc.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    Tensor kn2 = MakeTensor(kn.ptr(), DType::kF32, d.q.device, {T * nh, hd});
    vt::L2Norm(d.q, qn2, qc2, vt::L2NormArgs{1e-6f});
    vt::L2Norm(d.q, kn2, kc2, vt::L2NormArgs{1e-6f});
  }
  DBuf braw(d, DType::kF32, {T, nh});
  GemmBf16(d, braw.t(), dh, w.b_proj, nh, H);
  DBuf fa(d, DType::kF32, {T, hd});
  GemmBf16(d, fa.t(), dh, w.f_a_proj, hd, H);
  DBuf g1(d, DType::kF32, {T, proj});
  GemmBf16(d, g1.t(), fa.t(), w.f_b_proj, proj, hd);
  DBuf ga(d, DType::kF32, {T, hd});
  GemmBf16(d, ga.t(), dh, w.g_a_proj, hd, H);
  DBuf g2(d, DType::kF32, {T, proj});
  GemmBf16(d, g2.t(), ga.t(), w.g_b_proj, proj, hd);

  // beta = sigmoid(b_proj) — host elementwise, exactly the Inc island's form.
  std::vector<float> hbraw(static_cast<size_t>(T) * nh);
  braw.Download(d, hbraw.data());
  RoundHostBf16(hbraw);
  std::vector<float> hbeta(hbraw.size());
  for (size_t i = 0; i < hbeta.size(); ++i)
    hbeta[i] = static_cast<float>(Sigmoid(hbraw[i]));
  DBuf dbeta(d, DType::kF32, {T, nh}, hbeta.data());

  // Raw per-channel gate projection, downloaded once; the recurrent segments
  // compute the decay gate on host (kimi_kda::KdaDecayGate — the Inc island),
  // the chunk segment hands the RAW g1 to the fused on-device gate.
  std::vector<float> hg1(static_cast<size_t>(T) * proj);
  g1.Download(d, hg1.data());
  RoundHostBf16(hg1);

  DBuf dcore(d, DType::kF32, {T, proj});

  // ── decode segment [0, nd): batched T==1 recurrence over the paged state ──
  if (seg.nd > 0) {
    const int64_t ndt = seg.nd_tok;
    DBuf dss(d, DType::kF32, {seg.nd, nh, hd, hd});
    Tensor didx_dec = RowsView(didx.t(), 0, seg.nd, {seg.nd});
    vt::GdnStateGather(d.q, dss.t(), state.ssm_state, didx_dec);
    const std::vector<float> hg_dec(hg1.begin(),
                                    hg1.begin() + static_cast<size_t>(ndt) * proj);
    const std::vector<float> gch =
        kimi_kda::KdaDecayGate(hg_dec, w.a_log, w.dt_bias, ndt, nh, hd);
    DBuf dg(d, DType::kF32, {ndt, nh, hd}, gch.data());
    std::vector<int32_t> qsl_dec(static_cast<size_t>(seg.nd) + 1);
    for (int64_t i = 0; i <= seg.nd; ++i) qsl_dec[static_cast<size_t>(i)] =
        static_cast<int32_t>(i);
    DBuf dqsl(d, DType::kI32, {seg.nd + 1}, qsl_dec.data());
    Tensor q3v = RowsView(qn.t(), 0, ndt, {ndt, nh, hd});
    Tensor k3v = RowsView(kn.t(), 0, ndt, {ndt, nh, hd});
    Tensor v3v = RowsView(vc.t(), 0, ndt, {ndt, nh, hd});
    Tensor b2v = RowsView(dbeta.t(), 0, ndt, {ndt, nh});
    Tensor o3v = RowsView(dcore.t(), 0, ndt, {ndt, nh, hd});
    vt::KdaGatedDeltaRule(d.q, o3v, q3v, k3v, v3v, dg.t(), b2v, dss.t(), dqsl.t(),
                          vt::GdnArgs{scale});
    Tensor ssm_cache = state.ssm_state;
    vt::GdnStateScatter(d.q, ssm_cache, dss.t(), didx_dec);
  }

  // ── prefill segment: per request — chunk (fresh) or recurrence (continuing) ──
  for (int r = 0; r < seg.np; ++r) {
    const int req = seg.nd + r;
    const int64_t tok0 = (*seg.qsl)[static_cast<size_t>(req)];
    const int64_t tok1 = (*seg.qsl)[static_cast<size_t>(req) + 1];
    const int64_t Tr = tok1 - tok0;
    if (Tr <= 0) continue;
    const bool has_init = seg.his != nullptr && (*seg.his)[static_cast<size_t>(req)] != 0;
    DBuf dss1(d, DType::kF32, {1, nh, hd, hd});
    Tensor didx_r = RowsView(didx.t(), req, 1, {1});
    const int32_t hi32[1] = {has_init ? 1 : 0};
    DBuf dhi(d, DType::kI32, {1}, hi32);
    Tensor dhi_t = dhi.t();
    vt::GdnStateGather(d.q, dss1.t(), state.ssm_state, didx_r, &dhi_t);
    const int32_t qsl1[2] = {0, static_cast<int32_t>(Tr)};
    DBuf dqsl1(d, DType::kI32, {2}, qsl1);
    Tensor q3v = RowsView(qn.t(), tok0, Tr, {Tr, nh, hd});
    Tensor k3v = RowsView(kn.t(), tok0, Tr, {Tr, nh, hd});
    Tensor v3v = RowsView(vc.t(), tok0, Tr, {Tr, nh, hd});
    Tensor b2v = RowsView(dbeta.t(), tok0, Tr, {Tr, nh});
    Tensor o3v = RowsView(dcore.t(), tok0, Tr, {Tr, nh, hd});
    if (PagedKdaChunkEnabled() && !has_init && Tr > 1) {
      // vLLM's PROMPT path: the chunk kernels fuse the gate from the RAW g1.
      DBuf da_log(d, DType::kF32, {nh}, w.a_log.data());
      DBuf ddt(d, DType::kF32, {static_cast<int64_t>(w.dt_bias.size())},
               w.dt_bias.empty() ? nullptr : w.dt_bias.data());
      Tensor gr3 = RowsView(g1.t(), tok0, Tr, {Tr, nh, hd});
      vt::KdaChunkPrefill(d.q, o3v, q3v, k3v, v3v, gr3, b2v, da_log.t(), ddt.t(),
                          dss1.t(), dqsl1.t(), vt::GdnArgs{scale});
    } else {
      const std::vector<float> hg_r(
          hg1.begin() + static_cast<size_t>(tok0) * proj,
          hg1.begin() + static_cast<size_t>(tok1) * proj);
      const std::vector<float> gch =
          kimi_kda::KdaDecayGate(hg_r, w.a_log, w.dt_bias, Tr, nh, hd);
      DBuf dg(d, DType::kF32, {Tr, nh, hd}, gch.data());
      vt::KdaGatedDeltaRule(d.q, o3v, q3v, k3v, v3v, dg.t(), b2v, dss1.t(),
                            dqsl1.t(), vt::GdnArgs{scale});
    }
    Tensor ssm_cache = state.ssm_state;
    vt::GdnStateScatter(d.q, ssm_cache, dss1.t(), didx_r);
  }

  // ── sigmoid-gated RMSNorm output + o_proj (== Inc) ──
  DBuf dcn(d, DType::kF32, {T, proj});
  {
    Tensor x3 = MakeTensor(dcore.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor g3 = MakeTensor(g2.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    Tensor o3 = MakeTensor(dcn.ptr(), DType::kF32, d.q.device, {T, nh, hd});
    vt::RmsNormGated(d.q, o3, x3, g3, WF32(d, w.o_norm, {hd}),
                     vt::RmsNormGatedArgs{p.rms_norm_eps, /*sigmoid_gate=*/true});
  }
  DBuf out(d, DType::kF32, {T, H});
  GemmBf16(d, out.t(), dcn.t(), w.o_proj, H, proj);
  return out;
}

// ── the per-forward shared MLA step state (positions/slot_mapping/FA2 meta) ────
struct KimiMlaStep {
  std::vector<DBuf> owned;
  Tensor positions;     // [T] i32 device
  Tensor slot_mapping;  // [T] i64 device
  // FA2 arm only:
  mla::MlaBlockMetadata meta;
  MlaBatchSplit split;
  DBuf* rope_cache = nullptr;  // identity [rows, qr] bf16 (owned)
  v1::TritonMLAImpl impl;
};

template <typename T>
Tensor KimiUploadInto(const Dev& d, std::vector<DBuf>& owned, DType dt,
                      const std::vector<int64_t>& shape, const T* host) {
  owned.emplace_back(d, dt, shape, host);
  return owned.back().t();
}

// Build the per-step MLA metadata (the Kimi-local, EAGER-only port of
// deepseek_v2.cpp BuildMlaStep — `MLACommonMetadataBuilder.build`
// mla_attention.py:1652-1830, non-DCP; no CUDA-graph constraints because the
// Kimi paged forward is eager).
void BuildKimiMlaStep(const Dev& d, const std::vector<int32_t>& positions,
                      const v1::CommonAttentionMetadata& am,
                      const KimiLinearParams& p, int64_t block_size, bool fa2,
                      KimiMlaStep& s) {
  const int64_t T = static_cast<int64_t>(positions.size());
  s.positions = KimiUploadInto(d, s.owned, DType::kI32, {T}, positions.data());
  s.slot_mapping =
      KimiUploadInto(d, s.owned, DType::kI64, {T}, am.slot_mapping.data());
  if (!fa2) return;

  s.split = BuildMlaBatchSplit(am);
  const MlaBatchSplit& sp = s.split;
  const int64_t cols = am.block_table_num_cols;
  s.meta.num_decode_tokens = sp.num_decode_tokens;
  if (sp.num_decodes > 0) {
    s.meta.decode.block_table = KimiUploadInto(
        d, s.owned, DType::kI32, {sp.num_decodes, cols}, am.block_table_tensor.data());
    s.meta.decode.seq_lens = KimiUploadInto(d, s.owned, DType::kI32,
                                            {sp.num_decodes}, am.seq_lens.data());
    s.meta.decode.max_seq_len = sp.decode_max_seq_len;
  }
  if (sp.num_prefills > 0) {
    s.meta.prefill_cu_seqlens_q =
        KimiUploadInto(d, s.owned, DType::kI32, {sp.num_prefills + 1},
                       sp.prefill_cu_seqlens_q.data());
    s.meta.prefill_block_table = KimiUploadInto(
        d, s.owned, DType::kI32, {sp.num_prefills, cols},
        am.block_table_tensor.data() +
            static_cast<size_t>(sp.num_decodes) * static_cast<size_t>(cols));
    s.meta.max_query_len = sp.prefill_max_query_len;
    if (sp.num_prefills_with_context > 0) {
      const int64_t workspace = mla::DetermineChunkedPrefillWorkspaceSize(
          p.max_position_embeddings, am.num_reqs, block_size);
      const mla::MlaChunkedContextMetadata cm = mla::BuildMlaChunkedContext(
          sp.prefill_context_lens, sp.prefill_cu_seqlens_q, workspace, block_size);
      s.meta.prefill_tokens_with_context = cm.prefill_tokens_with_context;
      s.meta.chunk_workspace_tokens = workspace;
      const int64_t np = cm.num_prefills;
      const int32_t row = std::max<int32_t>(cm.max_token_num_over_chunk, 1);
      for (int32_t i = 0; i < cm.num_chunks; ++i) {
        mla::MlaChunkDeviceMetadata cd;
        cd.cu_seq_lens = KimiUploadInto(
            d, s.owned, DType::kI32, {np + 1},
            cm.cu_seq_lens.data() + static_cast<size_t>(i) * (np + 1));
        cd.starts = KimiUploadInto(d, s.owned, DType::kI32, {np},
                                   cm.starts.data() + static_cast<size_t>(i) * np);
        cd.token_to_seq =
            KimiUploadInto(d, s.owned, DType::kI32, {row},
                           cm.token_to_seq.data() + static_cast<size_t>(i) * row);
        cd.total_tokens = cm.chunk_total_token[static_cast<size_t>(i)];
        cd.max_seq_len = cm.max_seq_lens[static_cast<size_t>(i)];
        s.meta.chunks.push_back(cd);
      }
    }
  }
}

// NoPE-MLA layer over the PAGED latent cache — the EXACT-island arm (default;
// the fold-identity vehicle). Projections + kv_a_layernorm + the cache write are
// on-device; the attention core re-derives each request's per-head K/V from the
// paged 576-wide latent rows (kv_b up-projection — the SAME GemmBf16 the CLI ran
// at append time; the cache stores bf16 exactly as vLLM does) and runs the SAME
// f64 causal-softmax island as MlaSoftmaxIslandInc.
DBuf MlaLayerPagedExact(const Dev& d, const MlaResidentWeights& w, const Tensor& dh,
                        const KimiLinearParams& p, int64_t T,
                        const v1::CommonAttentionMetadata& am,
                        const PagedKvCache& kv, const KimiMlaStep& step) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t kvw = nah * (qn + vh);
  const int64_t head = L + qr;
  VT_CHECK(kv.num_kv_heads == 1 && kv.head_size == head,
           "kimi paged: the MLA cache must be 1-head, kv_lora+qk_rope wide "
           "(MLAAttentionSpec)");

  DBuf dq(d, DType::kF32, {T, nah * qk});
  GemmBf16(d, dq.t(), dh, w.q_proj, nah * qk, H);
  DBuf dlat(d, DType::kF32, {T, L + qr});
  GemmBf16(d, dlat.t(), dh, w.kv_a_proj_with_mqa, L + qr, H);
  DBuf dkvc(d, DType::kF32, {T, L});
  DBuf dkpe(d, DType::kF32, {T, qr});
  {
    const size_t rl = static_cast<size_t>(L + qr) * sizeof(float);
    const char* src = static_cast<const char*>(dlat.ptr());
    char* pc = static_cast<char*>(dkvc.ptr());
    char* pp = static_cast<char*>(dkpe.ptr());
    for (int64_t t = 0; t < T; ++t) {
      d.b.Copy(d.q, pc + static_cast<size_t>(t) * L * sizeof(float),
               src + static_cast<size_t>(t) * rl,
               static_cast<size_t>(L) * sizeof(float));
      d.b.Copy(d.q, pp + static_cast<size_t>(t) * qr * sizeof(float),
               src + static_cast<size_t>(t) * rl + static_cast<size_t>(L) * sizeof(float),
               static_cast<size_t>(qr) * sizeof(float));
    }
  }
  DBuf dkvcn(d, DType::kF32, {T, L});
  vt::RmsNorm(d.q, dkvcn.t(), dkvc.t(), WF32(d, w.kv_a_layernorm, {L}),
              vt::RmsNormArgs{p.rms_norm_eps, false});

  // Write this step's latent rows into the paged cache at slot_mapping (vLLM's
  // concat_and_cache_mla order: BEFORE the attention reads). Cache dtype follows
  // the spec (bf16 default — vLLM's regime; f32 under VT_KV_CACHE_F32).
  Tensor cache_t = MakeTensor(kv.data, kv.dtype, d.q.device,
                              {kv.num_blocks, kv.block_size, head});
  if (kv.dtype == DType::kF32) {
    vt::ConcatAndCacheMla(d.q, dkvcn.t(), dkpe.t(), cache_t, step.slot_mapping);
  } else {
    DBuf ckv(d, kv.dtype, {T, L});
    DBuf cpe(d, kv.dtype, {T, qr});
    vt::CastBf16(d.q, ckv.t(), dkvcn.t());
    vt::CastBf16(d.q, cpe.t(), dkpe.t());
    vt::ConcatAndCacheMla(d.q, ckv.t(), cpe.t(), cache_t, step.slot_mapping);
  }

  // Attention per request over the paged rows (query rows [tok0, tok1) at global
  // positions [base, base+Tq)).
  DBuf dout(d, DType::kF32, {T, nah * vh});
  const size_t es = vt::SizeOf(kv.dtype);
  const size_t row_b = static_cast<size_t>(head) * es;
  std::vector<uint8_t> hrows;
  for (int r = 0; r < am.num_reqs; ++r) {
    const int64_t tok0 = am.query_start_loc[static_cast<size_t>(r)];
    const int64_t tok1 = am.query_start_loc[static_cast<size_t>(r) + 1];
    const int64_t Tq = tok1 - tok0;
    if (Tq <= 0) continue;
    const int64_t base = am.num_computed_tokens_cpu[static_cast<size_t>(r)];
    const int64_t S = base + Tq;
    // Gather the request's S latent rows (block table walk) to host.
    hrows.resize(static_cast<size_t>(S) * row_b);
    const int32_t* bt = am.block_table_tensor.data() +
                        static_cast<size_t>(r) * am.block_table_num_cols;
    for (int64_t s0 = 0; s0 < S; s0 += kv.block_size) {
      const int64_t blk = bt[s0 / kv.block_size];
      const int64_t n = std::min<int64_t>(kv.block_size, S - s0);
      d.b.Copy(d.q, hrows.data() + static_cast<size_t>(s0) * row_b,
               static_cast<const char*>(kv.data) +
                   static_cast<size_t>(blk) * kv.block_size * row_b,
               static_cast<size_t>(n) * row_b);
    }
    d.b.Synchronize(d.q);
    // Split latent | kpe to f32 host.
    std::vector<float> hlat(static_cast<size_t>(S) * L);
    std::vector<float> hkpe(static_cast<size_t>(S) * qr);
    for (int64_t s0 = 0; s0 < S; ++s0) {
      const uint8_t* row = hrows.data() + static_cast<size_t>(s0) * row_b;
      if (kv.dtype == DType::kF32) {
        std::memcpy(&hlat[static_cast<size_t>(s0) * L], row,
                    static_cast<size_t>(L) * sizeof(float));
        std::memcpy(&hkpe[static_cast<size_t>(s0) * qr],
                    row + static_cast<size_t>(L) * sizeof(float),
                    static_cast<size_t>(qr) * sizeof(float));
      } else {
        const uint16_t* rb = reinterpret_cast<const uint16_t*>(row);
        for (int64_t i = 0; i < L; ++i)
          hlat[static_cast<size_t>(s0 * L + i)] = vt::BF16ToF32(rb[i]);
        for (int64_t i = 0; i < qr; ++i)
          hkpe[static_cast<size_t>(s0 * qr + i)] = vt::BF16ToF32(rb[L + i]);
      }
    }
    // Up-project the latent to per-head k_nope|v (the CLI's append-time GemmBf16;
    // the bf16 activation cast of the same latent values feeds the same GEMM).
    DBuf dlat_r(d, DType::kF32, {S, L}, hlat.data());
    DBuf dkv_r(d, DType::kF32, {S, kvw});
    GemmBf16(d, dkv_r.t(), dlat_r.t(), w.kv_b_proj, kvw, L);
    std::vector<float> hkv(static_cast<size_t>(S) * kvw);
    dkv_r.Download(d, hkv.data());
    RoundHostBf16(hkv);
    RoundHostBf16(hkpe);
    // The SAME f64 causal-softmax island as the CLI (MlaSoftmaxIslandInc).
    DBuf dq_r(d, DType::kF32, {Tq, nah * qk});
    d.b.Copy(d.q, dq_r.ptr(),
             static_cast<const char*>(dq.ptr()) +
                 static_cast<size_t>(tok0) * nah * qk * sizeof(float),
             static_cast<size_t>(Tq) * nah * qk * sizeof(float));
    DBuf o_r = MlaSoftmaxIslandInc(d, dq_r, hkv, hkpe, p, Tq, base);
    d.b.Copy(d.q,
             static_cast<char*>(dout.ptr()) +
                 static_cast<size_t>(tok0) * nah * vh * sizeof(float),
             o_r.ptr(), static_cast<size_t>(Tq) * nah * vh * sizeof(float));
  }
  DBuf attn(d, DType::kF32, {T, H});
  GemmBf16(d, attn.t(), dout.t(), w.o_proj, H, nah * vh);
  return attn;
}

// NoPE-MLA layer through mla::ForwardMlaAttentionBlock — vLLM's ACTUAL absorbed-
// MQA decode / FA2 prefill over the paged latent cache (§20.3c / §20.2). Identity
// RoPE (cos=1, sin=0: the GPT-J pair rotation is then the identity — NoPE, no
// positional term), scale qk_head_dim^-0.5, the no-q-lora branch. The block does
// its OWN ConcatAndCacheMla write.
DBuf MlaLayerPagedFa2(const Dev& d, const MlaResidentWeights& w, const Tensor& dh_f32,
                      const KimiLinearParams& p, int64_t T, const PagedKvCache& kv,
                      KimiMlaStep& step, const DBuf& kv_a_ln_bf16) {
  const int64_t H = p.hidden_size;
  const int64_t nah = p.num_attention_heads;
  const int64_t qn = p.qk_nope_head_dim;
  const int64_t qr = p.qk_rope_head_dim;
  const int64_t qk = qn + qr;
  const int64_t vh = p.v_head_dim;
  const int64_t L = p.kv_lora_rank;
  const int64_t head = L + qr;
  VT_CHECK(kv.dtype == DType::kBF16,
           "kimi paged FA2 MLA: the paged latent cache must be bf16 (unset "
           "VT_KV_CACHE_F32 for the FA2 arm)");
  VT_CHECK(!w.w_uk_t.Empty() && !w.w_uv.Empty(),
           "kimi paged FA2 MLA: absorbed W_UK_T/W_UV missing (loader absorption)");

  mla::MlaBlockDims dm;
  dm.hidden_size = H;
  dm.num_heads = nah;
  dm.qk_nope_head_dim = qn;
  dm.qk_rope_head_dim = qr;
  dm.v_head_dim = vh;
  dm.kv_lora_rank = L;
  dm.q_lora_rank = 0;
  dm.rms_norm_eps = p.rms_norm_eps;
  dm.scale = static_cast<float>(std::pow(static_cast<double>(qk), -0.5));

  mla::MlaBlockWeights mw;
  mw.kv_a_proj_with_mqa = ResidentBf16W(d, w.kv_a_proj_with_mqa, {L + qr, H});
  mw.q_proj = ResidentBf16W(d, w.q_proj, {nah * qk, H});
  mw.kv_a_layernorm = kv_a_ln_bf16.t();
  mw.kv_b_proj = ResidentBf16W(d, w.kv_b_proj, {nah * (qn + vh), L});
  mw.w_uk_t = ResidentBf16W(d, w.w_uk_t, {nah, qn, L});
  mw.w_uv = ResidentBf16W(d, w.w_uv, {nah, L, vh});
  mw.o_proj = ResidentBf16W(d, w.o_proj, {H, nah * vh});
  mw.rope_cos_sin_cache = step.rope_cache->t();

  Tensor cache_t = MakeTensor(kv.data, kv.dtype, d.q.device,
                              {kv.num_blocks, kv.block_size, head});
  DBuf dh_bf16(d, DType::kBF16, {T, H});
  vt::CastBf16(d.q, dh_bf16.t(), dh_f32);
  DBuf attn_bf16(d, DType::kBF16, {T, H});
  Tensor attn_t = attn_bf16.t();
  mla::ForwardMlaAttentionBlock(d, dm, mw, dh_bf16.t(), step.positions, cache_t,
                                step.slot_mapping, step.meta, step.impl, attn_t);
  DBuf attn(d, DType::kF32, {T, H});
  Tensor attn_f = attn.t();
  vt::CastF32(d.q, attn_f, attn_bf16.t());
  return attn;
}

// The whole paged-runner device forward — DeviceForwardBodyBf16Incremental's
// skeleton with the paged KDA/MLA layer forms and the runner's own metadata.
DBuf DeviceForwardBodyBf16Paged(const Dev& d, const KimiLinearWeights& weights,
                                const ModelForwardInput& in) {
  const KimiLinearResidentWeights& rw = weights.resident;
  const KimiLinearParams& p = weights.params;
  const int64_t H = p.hidden_size;
  const int64_t V = p.vocab_size;
  const int64_t L = p.num_hidden_layers;
  const int64_t T = static_cast<int64_t>(in.token_ids.size());
  const float eps = p.rms_norm_eps;
  VT_CHECK(T > 0, "kimi paged: empty token batch");
  VT_CHECK(rw.resident, "kimi paged: bf16-resident weights required (§13)");
  VT_CHECK(static_cast<int64_t>(rw.layers.size()) == L,
           "kimi paged: resident layer count != num_hidden_layers");
  VT_CHECK(in.attn_meta.num_actual_tokens == T,
           "kimi paged: attn_meta num_actual_tokens != token count");
  int64_t nkda = 0, nmla = 0;
  for (int64_t l = 0; l < L; ++l) (p.is_kda_layer(l) ? nkda : nmla)++;
  VT_CHECK(static_cast<int64_t>(in.gdn_state.size()) == nkda,
           "kimi paged: one GdnStateCache per KDA layer required");
  VT_CHECK(static_cast<int64_t>(in.attn_kv.size()) == nmla,
           "kimi paged: one MLA PagedKvCache per full-attention layer required");

  const KimiPagedSeg seg = KimiSegment(in.gdn_meta, T);
  const bool fa2 = PagedMlaFa2();
  const int64_t block_size = in.attn_kv.empty() ? 0 : in.attn_kv[0].block_size;
  KimiMlaStep step;
  BuildKimiMlaStep(d, in.positions, in.attn_meta, p, block_size, fa2, step);
  // Per-layer bf16 kv_a_layernorm views + the identity rope cache (FA2 arm).
  std::vector<std::unique_ptr<DBuf>> kv_a_ln_bf16(rw.layers.size());
  std::unique_ptr<DBuf> rope;
  if (fa2) {
    const int64_t rows =
        std::max<int64_t>(in.attn_meta.max_seq_len + 1, 2);
    std::vector<uint16_t> ident(static_cast<size_t>(rows) * p.qk_rope_head_dim);
    const uint16_t one = vt::F32ToBF16(1.0f);
    const int64_t half = p.qk_rope_head_dim / 2;
    for (int64_t rr = 0; rr < rows; ++rr)
      for (int64_t i = 0; i < half; ++i)
        ident[static_cast<size_t>(rr * p.qk_rope_head_dim + i)] = one;  // cos=1|sin=0
    rope = std::make_unique<DBuf>(d, DType::kBF16,
                                  std::vector<int64_t>{rows, p.qk_rope_head_dim},
                                  ident.data());
    step.rope_cache = rope.get();
  }

  DBuf hidden(d, DType::kF32, {T, H});
  {
    Tensor htab = ResidentBf16W(d, rw.embed_tokens, {V, H});
    Tensor hh = hidden.t();
    if (in.device_token_ids != nullptr) {
      // ENG-ASYNC-SCHED W4 (the async device mirror, DEFAULT ON on a real CUDA
      // GPU): the runner patched each decode row's sampled token into ITS device
      // input-id buffer and deliberately left the host `token_ids` STALE — a
      // forward that embeds the host vector reads the previous step's token and
      // decodes garbage (the GB10 9/128 divergence this branch was cut from).
      // Embed from the device pointer, exactly like qwen3_5's
      // DeviceTokenIdsScope consumer.
      Tensor ids = MakeTensor(const_cast<int32_t*>(in.device_token_ids),
                              DType::kI32, d.q.device, {T});
      vt::Embedding(d.q, hh, htab, ids);
    } else {
      DBuf dids(d, DType::kI32, {T}, in.token_ids.data());
      vt::Embedding(d.q, hh, htab, dids.t());
    }
  }
  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);
  Tensor hcur = hidden.t();
  std::shared_ptr<void> hold;

  int64_t kda_idx = 0, mla_idx = 0;
  for (int64_t l = 0; l < L; ++l) {
    const KimiLinearLayerResidentWeights& lw = rw.layers[static_cast<size_t>(l)];
    DBuf dhn(d, DType::kF32, {T, H});
    AddRmsNormS(d, dhn, hcur, lw.input_layernorm, H, res, eps, DType::kF32);
    DBuf attn = [&]() -> DBuf {
      if (lw.is_kda) {
        return KdaLayerPagedBf16(d, lw.kda, dhn.t(), p, T, seg,
                                 in.gdn_state[static_cast<size_t>(kda_idx++)]);
      }
      const PagedKvCache& kv = in.attn_kv[static_cast<size_t>(mla_idx++)];
      if (!fa2) return MlaLayerPagedExact(d, lw.mla, dhn.t(), p, T, in.attn_meta,
                                          kv, step);
      std::unique_ptr<DBuf>& ln = kv_a_ln_bf16[static_cast<size_t>(l)];
      if (!ln) {
        DBuf lnf(d, DType::kF32,
                 {static_cast<int64_t>(lw.mla.kv_a_layernorm.size())},
                 lw.mla.kv_a_layernorm.data());
        ln = std::make_unique<DBuf>(
            d, DType::kBF16,
            std::vector<int64_t>{static_cast<int64_t>(lw.mla.kv_a_layernorm.size())});
        vt::CastBf16(d.q, ln->t(), lnf.t());
      }
      return MlaLayerPagedFa2(d, lw.mla, dhn.t(), p, T, kv, step, *ln);
    }();
    DBuf dh2(d, DType::kF32, {T, H});
    AddRmsNormS(d, dh2, attn.t(), lw.post_attention_layernorm, H, res, eps,
                DType::kF32);
    DBuf mlp = lw.is_moe ? MoeBlockDeviceBf16(d, lw.moe, dh2.t(), p, T)
                         : DenseMlpDeviceBf16(d, lw.dense, dh2.t(), p, T);
    auto* held = new DBuf(std::move(mlp));
    hcur = held->t();
    hold = std::shared_ptr<void>(held, [](void* q) { delete static_cast<DBuf*>(q); });
  }

  DBuf dnorm(d, DType::kF32, {T, H});
  AddRmsNormS(d, dnorm, hcur, rw.final_norm, H, res, eps, DType::kF32);

  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kF32,
               in.logits_indices.empty()
                   ? std::vector<int64_t>{1, 1}
                   : std::vector<int64_t>{
                         static_cast<int64_t>(in.logits_indices.size()), H});
  if (!in.logits_indices.empty()) {
    const size_t rb = static_cast<size_t>(H) * sizeof(float);
    char* dp = static_cast<char*>(dgather.ptr());
    const char* sp = static_cast<const char*>(dnorm.ptr());
    for (size_t i = 0; i < in.logits_indices.size(); ++i) {
      const int32_t idx = in.logits_indices[i];
      VT_CHECK(idx >= 0 && idx < T, "kimi paged: logits index out of range");
      d.b.Copy(d.q, dp + i * rb, sp + static_cast<size_t>(idx) * rb, rb);
    }
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];

  const bool tied = p.tie_word_embeddings || rw.lm_head.Empty();
  const OwnedTensor& lm = tied ? rw.embed_tokens : rw.lm_head;
  DBuf logits(d, DType::kF32, {n_out, V});
  GemmBf16(d, logits.t(), src, lm, V, H);
  return logits;
}

}  // namespace

// ─── per-op device wrappers (host-in / host-out) — the per-op CPU gates ────────
std::vector<float> KimiKdaLayerForwardDevice(const KdaLayerHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = KdaLayerDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiNoPEMlaLayerForwardDevice(const MlaLayerHostWeights& w,
                                                 const std::vector<float>& hidden_normed,
                                                 const KimiLinearParams& p,
                                                 int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = MlaLayerDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

// Device MLA attention CORE only (pad-V + vt::Attention), host-in / host-out — the
// dedicated RED-first CPU gate for the VT_KIMI_DEVICE_MLA wiring, independent of the
// env flag. q [T,nah*qk], kv [T,nah*(qn+vh)], kpe [T,qr] -> [T,nah*vh].
std::vector<float> KimiMlaAttnCoreDevice(const std::vector<float>& q_host,
                                         const std::vector<float>& kv_host,
                                         const std::vector<float>& kpe_host,
                                         const KimiLinearParams& p, int64_t num_tokens,
                                         vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t nah = p.num_attention_heads;
  const int64_t qk = p.qk_nope_head_dim + p.qk_rope_head_dim;
  const int64_t vh = p.v_head_dim;
  const int64_t kvw = nah * (p.qk_nope_head_dim + vh);
  const int64_t qr = p.qk_rope_head_dim;
  DBuf dq(d, DType::kF32, {num_tokens, nah * qk}, q_host.data());
  DBuf dkv(d, DType::kF32, {num_tokens, kvw}, kv_host.data());
  DBuf dkpe(d, DType::kF32, {num_tokens, qr}, kpe_host.data());
  DBuf out = MlaAttnCoreDevice(d, dq, dkv, dkpe, p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * nah * vh);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiMoeBlockForwardDevice(const MoeHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = MoeBlockDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

std::vector<float> KimiDenseMlpForwardDevice(const MlpHostWeights& w,
                                             const std::vector<float>& hidden_normed,
                                             const KimiLinearParams& p,
                                             int64_t num_tokens, vt::Queue& queue) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dh(d, DType::kF32, {num_tokens, p.hidden_size}, hidden_normed.data());
  DBuf out = DenseMlpDevice(d, w, dh.t(), p, num_tokens);
  std::vector<float> h(static_cast<size_t>(num_tokens) * p.hidden_size);
  out.Download(d, h.data());
  return h;
}

bool KimiDeviceComputeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_KIMI_DEVICE_COMPUTE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

ForwardLogits KimiLinearModel::ForwardDeviceCompute(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const KimiLinearWeights& weights,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  // §13: the FULL model loads as bf16-resident (host f32 would OOM the pool), so
  // route through the bf16 device forward whenever resident weights are present; the
  // f32 host path stays for the tiny-config unit gate.
  const bool bf16 = weights.resident.resident;
  VT_CHECK(bf16 || weights.host.materialized,
           "KimiLinear device compute: neither bf16-resident (§13, "
           "LoadKimiLinearResidentBf16Weights) nor host-float (LoadKimiLinearFor"
           "CausalLMWeights) weights are populated. The device compute reads one or "
           "the other; the full model MUST use the bf16-resident path (183 GiB f32 "
           "OOMs the 119 GiB unified pool).");
  // The device compute manages a fresh single-sequence context (NoPE, causal); the
  // runner's paged het-KV / positions are consumed by the born-on-runner residual
  // (the paged incremental decode), not this seam.
  (void)positions;
  (void)attn_meta;
  (void)attn_kv;
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = bf16 ? DeviceForwardBodyBf16(d, weights, token_ids, logits_indices)
                      : DeviceForwardBody(d, weights, token_ids, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

// ─── PAGED-INCREMENTAL DECODE (§18 lever e) — public entry points ──────────────
ForwardLogits KimiLinearModel::ForwardPrefillIncremental(
    const std::vector<int32_t>& prompt, const std::vector<int32_t>& positions,
    const KimiLinearWeights& weights, vt::Queue& queue, KimiDecodeCache& cache,
    const std::vector<int32_t>& logits_indices) {
  (void)positions;  // NoPE-MLA + recurrence: causal masking is by cache length, no RoPE
  VT_CHECK(weights.resident.resident,
           "KimiLinear ForwardPrefillIncremental: bf16-resident weights required (§13)");
  const KimiLinearParams& p = weights.params;
  int64_t nkda = 0, nmla = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l)
    (p.is_kda_layer(l) ? nkda : nmla)++;
  cache.kda.assign(static_cast<size_t>(nkda), KimiKdaLayerCache{});
  cache.mla.assign(static_cast<size_t>(nmla), KimiMlaLayerCache{});
  cache.seq_len = 0;
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DeviceForwardBodyBf16Incremental(d, weights, prompt, /*base_pos=*/0, cache,
                                                  /*is_prefill=*/true, logits_indices);
  cache.seq_len = static_cast<int64_t>(prompt.size());
  cache.prefilled = true;
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, p.vocab_size);
}

// ─── ROW 7 — the shared-paged-runner fold (§20.3) — public entry ───────────────
ForwardLogits KimiLinearModel::ForwardPaged(const ModelForwardInput& input,
                                            const KimiLinearWeights& weights) {
  Dev d{vt::GetBackend(input.queue.device.type), input.queue};
  DBuf dlogits = DeviceForwardBodyBf16Paged(d, weights, input);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

ForwardLogits KimiLinearModel::ForwardDecodeStepIncremental(
    int32_t token, int64_t position, const KimiLinearWeights& weights, vt::Queue& queue,
    KimiDecodeCache& cache) {
  (void)position;  // causal masking is by cache.seq_len; NoPE so no positional term
  VT_CHECK(cache.prefilled,
           "KimiLinear ForwardDecodeStepIncremental: call ForwardPrefillIncremental first");
  VT_CHECK(weights.resident.resident,
           "KimiLinear ForwardDecodeStepIncremental: bf16-resident weights required (§13)");
  const std::vector<int32_t> ids = {token};
  const std::vector<int32_t> li = {0};  // the single decoded token's logits
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DeviceForwardBodyBf16Incremental(d, weights, ids, /*base_pos=*/cache.seq_len,
                                                  cache, /*is_prefill=*/false, li);
  cache.seq_len += 1;
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(std::move(dlogits), n_out, weights.params.vocab_size);
}

}  // namespace vllm
