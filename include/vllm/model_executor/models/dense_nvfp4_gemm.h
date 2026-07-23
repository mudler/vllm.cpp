// Shared NVFP4 **W4A16** (4-bit weights, BF16 activations) dense GEMM glue for
// full-attention models — the QUANT-SCHEME additivity seam.
//
// Extracted (behavior-preserving) from the anonymous namespace of
// src/vllm/model_executor/models/qwen3_5.cpp so a dense model whose forward is
// already token-exact in BF16 (Qwen3 `Qwen3ForCausalLM`, qwen3.cpp) can swap a
// BF16 projection for an NVFP4 one WITHOUT re-deriving the resident-repack,
// align-cache, workspace and dispatch machinery. This mirrors how
// dense_weight_loaders.h, device_pool.h and dense_attn_block.h were extracted
// for reuse (.agents/specs/sweep-qwen3-32b-nvfp4a16.md §Port map SEAM #1).
//
// SCOPE — W4A16 ONLY. This header deliberately does NOT carry the true-W4A4
// (fp4-ACTIVATION) path (`MatmulNvfp4Fp4D`, ScaledFp4Quant, the cutlass
// swizzled-blockscale/alpha residents). W4A4 stays private to qwen3_5.cpp with
// the 27B it serves. Selection is by the SAME predicate vLLM uses: a scheme
// whose `input_activations` is null is W4A16 (`Nvfp4Weight::IsTrueW4A4()` ==
// false, i.e. alpha == 0), and vLLM then FORCES the Marlin kernel —
// vllm/model_executor/kernels/linear/__init__.py:879-881
//   `elif linear_backend == "auto" and use_a16: force_kernel = MarlinNvFp4LinearKernel`
//   ("Force a16 (Marlin) when running weight-only quantization")
// bypassing the capability-based kernel registry entirely. So on sm_121 the
// dispatch below is Marlin, unconditionally, exactly like the oracle's.
//
// UPSTREAM CHAIN (ported FROM, cited per the ground-every-impl rule):
//   * scheme            vllm/model_executor/layers/quantization/compressed_tensors/
//                         schemes/compressed_tensors_w4a4_nvfp4.py:29-32,95-141
//                         (`CompressedTensorsW4A4Fp4(use_a16=True)`; note vLLM
//                         has NO separate W4A16 class — the a16 flag reuses this)
//   * kernel selection  vllm/model_executor/kernels/linear/__init__.py:842,879-892
//   * marlin wrapper    vllm/model_executor/kernels/linear/nvfp4/marlin.py:21-57
//   * repack + scales   vllm/model_executor/layers/quantization/utils/
//                         marlin_utils_fp4.py:61-122 (nvfp4_marlin_process_scales),
//                         :142-154 (nvfp4_marlin_process_global_scale),
//                         :221-306 (prepare_fp4_layer_for_marlin),
//                         :157-218 (apply_fp4_marlin_linear)
//   * CUDA GEMM         csrc/libtorch_stable/quantization/marlin/marlin.cu:545,600-611
//   * global-scale epi  csrc/libtorch_stable/quantization/marlin/marlin_template.h:1655-1657
// Our vendored, bit-exact lift of those primitives is include/vt/cuda/marlin_repack.h
// (MarlinRepackExpertWeight / MarlinProcessExpertScales /
// MarlinNvfp4CombinedScaleFactor / MarlinNvfp4ProcessGlobalScale) driving
// vt::MoeGroupedGemmNvfp4Marlin with num_experts=1 — the SINGLE-EXPERT grouped
// GEMM is how a dense [M,K]x[N,K]^T W4A16 linear runs on the MoE Marlin entry
// point (vLLM reaches the same csrc kernel through `ops.marlin_gemm`).
//
// KNOWN DUPLICATION (recorded, not accidental): qwen3_5.cpp retains its own
// copies of these definitions, exactly as it retains its own Dev/DBuf after the
// dense_attn_block.h extraction (see that header's preamble). Unifying the two
// device-glue families is a separate, gate-model-touching refactor; this header
// deliberately leaves the 27B/35B hot path BYTE-UNTOUCHED so the 235/235 +
// 315/315 regressions cannot move. Tracked in the spike's §Risks/decisions.
#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"  // DequantNvfp4ToBf16
#include "vllm/model_executor/models/dense_device_glue.h"    // Dev/DBuf/MakeTensor
#include "vllm/model_executor/models/qwen3_5_weights.h"      // Nvfp4Weight
#include "vt/backend.h"
#include "vt/dtype.h"  // VT_CHECK
#include "vt/ops.h"

#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {
namespace dense_nvfp4 {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;
using vt::Backend;
using vt::DType;
using vt::Tensor;

// VT_NVFP4_MARLIN (default ON): the vendored Marlin NVFP4 W4A16 GEMM is the
// validated path (35B gate +22%, token-for-token vs the pinned oracle). Only an
// explicit VT_NVFP4_MARLIN=0 opts back out to the naive redundant-dequant kernel
// (kept as an A/B escape hatch). Mirrors qwen3_5.cpp::MarlinMoeEnabled.
inline bool MarlinW4A16Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_MARLIN");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// VT_MOE_FUSED_W13 (default ON): run a gate+up PAIR as ONE Marlin GEMM over the
// N-concatenated operand (size_n = 2N) + one SiluAndMul, instead of two GEMMs.
// This is exactly vLLM's merged `gate_up_proj` MergedColumnParallelLinear, which
// `prepare_fp4_layer_for_marlin` repacks WHOLE as a single Marlin operand
// (marlin_utils_fp4.py:221-306) — so the FUSED layout is the vLLM-faithful one,
// and the split layout is our A/B fallback.
inline bool FusedGateUpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_FUSED_W13");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// --- Execution counters (the "this path actually RAN" positive signal) ------
// A passing correctness gate does NOT prove a new code path was exercised — a
// mis-wired dispatch that silently fell back to the BF16 arm would also pass if
// the weights happened to be BF16. These counters make the W4A16 path
// OBSERVABLE, so the gate can assert on them (mirrors the ArchTacticStats
// pattern in cuda_arch_tactics.h, whose tests assert selections/fallbacks).
struct Nvfp4W4A16Stats {
  uint64_t marlin_gemms = 0;      // MatmulNvfp4MarlinD launches
  uint64_t fused_gate_up = 0;     // GateUpFusedMarlinD launches (one per MLP)
  uint64_t fallback_gemms = 0;    // naive vt::MatmulNvfp4 / CPU dequant launches
};

inline Nvfp4W4A16Stats& MutableW4A16Stats() {
  static Nvfp4W4A16Stats s;
  return s;
}

// Snapshot of the counters (test/diagnostic entry point).
inline Nvfp4W4A16Stats GetW4A16Stats() { return MutableW4A16Stats(); }

inline void ResetW4A16Stats() { MutableW4A16Stats() = Nvfp4W4A16Stats{}; }

// Device-resident views over an Nvfp4Weight's packed + scale buffers, uploaded
// ONCE (lazily) and owned by the (const) weight's shared_ptr for the model's
// lifetime. CUDA path only.
struct Nvfp4Dev {
  Tensor packed;
  Tensor scale;
};

inline Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  if (!w.d_scale) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.d_scale = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  Nvfp4Dev r;
  r.packed = MakeTensor(w.d_packed.get(), DType::kI8, d.q.device, {w.n, w.k / 2});
  r.scale = MakeTensor(w.d_scale.get(), DType::kI8, d.q.device, {w.n, w.k / 16});
  return r;
}

// Host dequant of an fp4 weight to bf16 [K=in, N=out] (Matmul-B layout) — the
// CPU fallback (there is no CPU MatmulNvfp4 kernel). Bit-for-bit
// vllm::DequantNvfp4ToBf16 + transpose.
inline std::vector<uint16_t> DequantNvfp4ToBLayout(const Nvfp4Weight& w) {
  const int64_t out_dim = w.n, in_dim = w.k;
  std::vector<uint16_t> oi(static_cast<size_t>(out_dim) * in_dim);
  DequantNvfp4ToBf16(reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
                     reinterpret_cast<const uint8_t*>(w.scale.bytes.data()),
                     w.scale2, out_dim, in_dim, oi.data());
  std::vector<uint16_t> io(static_cast<size_t>(in_dim) * out_dim);
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      io[static_cast<size_t>(c) * out_dim + r] =
          oi[static_cast<size_t>(r) * in_dim + c];
  return io;
}

#ifdef VT_MARLIN_NVFP4

// --- Resident Marlin operands (repacked ONCE at first use) ------------------
// vLLM does this repack in `process_weights_after_loading`
// (marlin_utils_fp4.py:221-306); we do it lazily on first forward and then FREE
// the fp4 originals, so peak weight memory stays flat.
struct MarlinDenseResident {
  void* w = nullptr;  // i32 [K/16, N*2]  Marlin-interleaved weight
  void* s = nullptr;  // fp8 [K/16, N]    processed S0E5M3 scales
  void* g = nullptr;  // f32 [1]          processed global scale
  int64_t n = 0, k = 0;
  bool ready = false;
};

inline MarlinDenseResident& MarlinDenseResidentFor(const Nvfp4Weight* w) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDenseResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

inline void BuildMarlinDenseResident(Dev d, const Nvfp4Weight& w,
                                     MarlinDenseResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(w.k);
  const int N = static_cast<int>(w.n);
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(N) * 2);
  const size_t s_b = static_cast<size_t>(K / 16) * N;
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = w.n;
  mr.k = w.k;
  std::vector<const uint8_t*> bufs{
      reinterpret_cast<const uint8_t*>(w.scale.bytes.data())};
  std::vector<size_t> lens{w.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index,
                                     static_cast<uint32_t*>(mr.w),
                                     static_cast<const uint8_t*>(dw.packed.data), K, N);
  vt::cuda::MarlinProcessExpertScales(stream,
                                      static_cast<const uint8_t*>(dw.scale.data),
                                      static_cast<uint8_t*>(mr.s), K, N, sf);
  const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
  d.b.Copy(d.q, mr.g, &g, sizeof(float));
  d.b.Synchronize(d.q);  // repack done -> safe to free the fp4 originals
  w.d_packed.reset();
  w.d_scale.reset();
  mr.ready = true;
}

// Trivial single-expert moe_align inputs (all M tokens -> expert 0), cached per
// token count M. Avoids a per-GEMM moe_align launch + allocations.
struct DenseAlignCache {
  void* sorted = nullptr;  // i32 [max_tok]
  void* expert = nullptr;  // i32 [max_blk] (all 0)
  void* numpad = nullptr;  // i32 [1]
  void* topkw = nullptr;   // f32 [M] (ones; unused, mul_topk_weights=false)
  int block = 0, max_tok = 0, max_blk = 0;
};

inline DenseAlignCache& DenseAlignFor(Dev d, int M) {
  static std::mutex mu;
  static std::unordered_map<int, DenseAlignCache> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(M);
  if (it != cache.end()) return it->second;
  DenseAlignCache c;
  c.block = vt::cuda::MarlinMoeAlignBlockSizeSelect(M, 1, 1);
  vt::cuda::MarlinMoeAlignSizes(M, 1, 1, c.block, &c.max_tok, &c.max_blk);
  c.sorted = d.b.Alloc(static_cast<size_t>(c.max_tok) * sizeof(int32_t));
  c.expert = d.b.Alloc(static_cast<size_t>(c.max_blk) * sizeof(int32_t));
  c.numpad = d.b.Alloc(sizeof(int32_t));
  c.topkw = d.b.Alloc(static_cast<size_t>(M) * sizeof(float));
  void* tid = d.b.Alloc(static_cast<size_t>(M) * sizeof(int32_t));
  d.b.Memset(d.q, tid, 0, static_cast<size_t>(M) * sizeof(int32_t));  // -> expert 0
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(tid), M, 1, 1,
                                    c.block, static_cast<int32_t*>(c.sorted),
                                    static_cast<int32_t*>(c.expert),
                                    static_cast<int32_t*>(c.numpad));
  std::vector<float> ones(static_cast<size_t>(M), 1.0F);
  d.b.Copy(d.q, c.topkw, ones.data(), ones.size() * sizeof(float));
  d.b.Synchronize(d.q);
  d.b.Free(tid);
  return cache.emplace(M, c).first->second;
}

// Shared zeroed reduction workspace for the dense Marlin GEMMs (sms*4 i32 locks,
// mirror marlin_make_workspace_new). Memset to zero before each launch.
inline void* DenseMarlinWorkspace(Dev d, int* out_sms) {
  static std::mutex mu;
  static void* ws = nullptr;
  static int sms = 0;
  std::lock_guard<std::mutex> lk(mu);
  if (!ws) {
    sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
    ws = d.b.Alloc(static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  }
  *out_sms = sms;
  return ws;
}

// y[M,N] = x[M,K] bf16 @ dequant(w).T via the single-expert Marlin W4A16 GEMM.
inline DBuf MatmulNvfp4MarlinD(Dev d, const Tensor& x, const Nvfp4Weight& w,
                               DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  MarlinDenseResident& mr = MarlinDenseResidentFor(&w);
  if (!mr.ready) BuildMarlinDenseResident(d, w, mr);
  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  ++MutableW4A16Stats().marlin_gemms;

  // Marlin's output is bf16 (c_type=kBFloat16); an f32 result is the bf16 output
  // upcast (the same value it rounds to).
  DBuf outbf(d, DType::kBF16, {M, N});
  Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, N * 2});
  Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / 16, N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
  Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
  Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
  Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
  vt::MoeGroupedGemmNvfp4Marlin(
      d.q, outbf.t(), x, wq, sc, gg, wst, sorted, expert, numpad, topkw,
      vt::MoeMarlinArgs{ac.block, 1, static_cast<int>(M), static_cast<int>(N),
                        static_cast<int>(K), false});
  if (out_dtype == DType::kBF16) return outbf;
  DBuf out(d, DType::kF32, {M, N});
  vt::CastF32(d.q, out.t(), outbf.t());
  return out;
}

// --- Fused gate_up (one Marlin GEMM over the N-concatenated pair) -----------
struct MarlinDensePairResident {
  void* w = nullptr;     // i32 [K/16, (2N)*2]
  void* s = nullptr;     // fp8 [K/16, 2N]
  void* g = nullptr;     // f32 [1]
  int64_t n = 0, k = 0;  // n = per-shard N; operand size_n = 2n
  bool ready = false;
};

inline MarlinDensePairResident& MarlinDensePairResidentFor(const Nvfp4Weight* gate) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDensePairResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[gate];
}

inline void BuildMarlinDensePairResident(Dev d, const Nvfp4Weight& gw,
                                         const Nvfp4Weight& uw,
                                         MarlinDensePairResident& mr) {
  if (mr.ready) return;
  const int K = static_cast<int>(gw.k);
  const int N = static_cast<int>(gw.n);
  void* stream = d.q.handle;
  const size_t w_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(2 * N) * 2);
  const size_t s_b = static_cast<size_t>(K / 16) * (2 * N);
  const size_t pk_b = static_cast<size_t>(N) * (K / 2);   // one shard's packed bytes
  const size_t sc_b = static_cast<size_t>(N) * (K / 16);  // one shard's scale bytes
  mr.w = d.b.Alloc(w_i32 * 4);
  mr.s = d.b.Alloc(s_b);
  mr.g = d.b.Alloc(sizeof(float));
  mr.n = gw.n;
  mr.k = gw.k;
  // combined_scale_factor over BOTH shards (vLLM computes it over the MERGED
  // gate_up scale tensor — marlin_utils_fp4.py:281-284 operates on the whole
  // parameter, which for gate_up_proj is already the concatenation).
  std::vector<const uint8_t*> bufs{
      reinterpret_cast<const uint8_t*>(gw.scale.bytes.data()),
      reinterpret_cast<const uint8_t*>(uw.scale.bytes.data())};
  std::vector<size_t> lens{gw.scale.bytes.size(), uw.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dg = ResidentNvfp4(d, gw);
  Nvfp4Dev du = ResidentNvfp4(d, uw);
  // Flat row-stack concat (packed [N,K/2] u8 / scales [N,K/16] fp8 are row-major
  // over N; gate rows FIRST — vLLM's merged shard order, qwen3.py:271-274
  // `gate_up_proj: [gate_proj, up_proj]`).
  auto* tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
  auto* tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sc_b));
  d.b.Copy(d.q, tmp_w, dg.packed.data, pk_b);
  d.b.Copy(d.q, tmp_w + pk_b, du.packed.data, pk_b);
  d.b.Copy(d.q, tmp_s, dg.scale.data, sc_b);
  d.b.Copy(d.q, tmp_s + sc_b, du.scale.data, sc_b);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index,
                                     static_cast<uint32_t*>(mr.w), tmp_w, K, 2 * N);
  vt::cuda::MarlinProcessExpertScales(stream, tmp_s, static_cast<uint8_t*>(mr.s), K,
                                      2 * N, sf);
  // ONE global scale for both shards (vLLM's merged parameter has exactly one
  // weight_global_scale — it takes `.max()` across the shards at
  // compressed_tensors_w4a4_nvfp4.py:111-114; equality is guarded by the caller).
  const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(gw.scale2, sf);
  d.b.Copy(d.q, mr.g, &g, sizeof(float));
  d.b.Synchronize(d.q);  // repack done -> safe to free staging + fp4 originals
  d.b.Free(tmp_w);
  d.b.Free(tmp_s);
  gw.d_packed.reset();
  gw.d_scale.reset();
  uw.d_packed.reset();
  uw.d_scale.reset();
  mr.ready = true;
}

// True when a gate/up pair takes the fused Marlin gate_up path. Must be checked
// IDENTICALLY at every call site so exactly ONE resident layout is ever built.
inline bool GateUpFusedEligible(const Nvfp4Weight& gw, const Nvfp4Weight& uw) {
  return FusedGateUpEnabled() && !gw.Empty() && !uw.Empty() && !gw.IsTrueW4A4() &&
         !uw.IsTrueW4A4() && gw.n == uw.n && gw.k == uw.k && gw.scale2 == uw.scale2;
}

// silu(x@gate.T) * (x@up.T) -> bf16 [M,N] via ONE fused Marlin gate_up GEMM.
inline DBuf GateUpFusedMarlinD(Dev d, const Tensor& x, const Nvfp4Weight& gw,
                               const Nvfp4Weight& uw) {
  const int64_t M = x.shape[0], K = x.shape[1], N = gw.n;
  MarlinDensePairResident& mr = MarlinDensePairResidentFor(&gw);
  if (!mr.ready) BuildMarlinDensePairResident(d, gw, uw, mr);
  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  ++MutableW4A16Stats().fused_gate_up;

  DBuf gu(d, DType::kBF16, {M, 2 * N});
  Tensor wq = MakeTensor(mr.w, DType::kI32, d.q.device, {1, K / 16, 2 * N * 2});
  Tensor sc = MakeTensor(mr.s, DType::kI8, d.q.device, {1, K / 16, 2 * N});
  Tensor gg = MakeTensor(mr.g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  Tensor sorted = MakeTensor(ac.sorted, DType::kI32, d.q.device, {ac.max_tok});
  Tensor expert = MakeTensor(ac.expert, DType::kI32, d.q.device, {ac.max_blk});
  Tensor numpad = MakeTensor(ac.numpad, DType::kI32, d.q.device, {1});
  Tensor topkw = MakeTensor(ac.topkw, DType::kF32, d.q.device, {M});
  vt::MoeGroupedGemmNvfp4Marlin(
      d.q, gu.t(), x, wq, sc, gg, wst, sorted, expert, numpad, topkw,
      vt::MoeMarlinArgs{ac.block, 1, static_cast<int>(M), static_cast<int>(2 * N),
                        static_cast<int>(K), false});
  DBuf act(d, DType::kBF16, {M, N});
  vt::SiluAndMul(d.q, act.t(), gu.t());
  return act;
}

#endif  // VT_MARLIN_NVFP4

// --- The W4A16 dispatcher ---------------------------------------------------
// y[M,N] = x[M,K] @ dequant(w).T for an NVFP4 W4A16 weight. Mirrors vLLM's
// forced-Marlin selection for `use_a16` (__init__.py:879-881): on CUDA with a
// BF16 activation take Marlin; otherwise fall back to the naive
// redundant-dequant vt::MatmulNvfp4 (CUDA) or a host dequant + bf16 Matmul (CPU
// reference). `w` MUST be W4A16 (alpha == 0) — a true-W4A4 weight belongs to
// qwen3_5.cpp's private fp4-activation path and is rejected here.
inline DBuf MatmulNvfp4W4A16D(Dev d, const Tensor& x, const Nvfp4Weight& w,
                              DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(!w.IsTrueW4A4(),
           "dense_nvfp4: true-W4A4 weight routed into the W4A16 dispatcher");
#ifdef VT_MARLIN_NVFP4
  // Marlin requires a bf16 activation (vLLM's a16 path is bf16/fp16 too). The
  // device gate is an OP-AVAILABILITY question, not a "== kCUDA" question: ask
  // the vt::OpProvider table whether the Marlin NVFP4 grouped-GEMM is realized
  // for this device (registered only for kCUDA today, so this is byte-identical
  // on the production build — accelerator-seam audit class A, work row S4).
  if (vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type) &&
      MarlinW4A16Enabled() && x.dtype == DType::kBF16)
    return MatmulNvfp4MarlinD(d, x, w, out_dtype);
#endif
  ++MutableW4A16Stats().fallback_gemms;
  DBuf dout(d, out_dtype, {M, N});
  // Same class-A conversion: the naive redundant-dequant NVFP4 GEMM exists only
  // where the op table realizes it (kCUDA); elsewhere fall to the host dequant +
  // bf16 Matmul reference. Byte-identical to the old `device == kCUDA` test.
  if (vt::OpRegistered(vt::OpId::kMatmulNvfp4, d.q.device.type)) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), x, dwb.t());
  }
  return dout;
}

}  // namespace dense_nvfp4
}  // namespace vllm
