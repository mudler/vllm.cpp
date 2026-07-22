// Shared dense self-attention block + device glue for full-attention models.
//
// Extracted VERBATIM (behavior-preserving) from the anonymous namespace of
// src/vllm/model_executor/models/qwen3.cpp so a second full-attention arch
// (Qwen3-Coder `Qwen3MoeForCausalLM`, the first full-attention MoE bring-up —
// qwen3_moe.cpp, W3) reuses the EXACT same attention preamble (merged/3-shard
// QKV GEMM, per-head q/k RMSNorm, NeoX RoPE, paged FA2, o_proj) and the pooled
// device-scratch glue (Dev/DBuf/ResidentWeight/StepInputs) rather than
// re-deriving them. This mirrors how dense_weight_loaders.h + device_pool.h were
// extracted for reuse (.agents/specs/sweep-qwen3-coder-30b.md §3b SEAM GAP #1).
//
// It is a PURE RELOCATION: the env-flag readers, Dev/DBuf/pool-policy glue, the
// resident-weight uploaders, KvSlice, StepInputs/BuildStepInputs and AttnBlock
// are byte-for-byte the qwen3.cpp definitions (now `inline` in namespace
// `vllm::dense_attn`), so the Qwen3-dense (0.6B/4B) forward that includes this
// header and `using namespace dense_attn;`-imports it is BYTE-IDENTICAL — the
// same vt:: op sequence runs in the same order. qwen3.cpp keeps the dense-only
// MLP/decoder-layer/forward-body machinery that composes these pieces.
//
// Numeric contract (mirrors the qwen3_5 full-attention FALLBACK path — the
// token-exact paged==dense anchor): the residual stream is the model dtype (bf16,
// matching vLLM's fused_add_rms_norm residual); the qkv GEMM emits f32 q/k/v so
// the per-head q/k RMSNorm + RoPE run in f32 on the f32 A/B path (default bf16
// mirrors vLLM's per-op bf16 stores); the paged KV cache is written bf16 while
// the query stays f32 into vt::PagedAttention; o_proj flows bf16.
#pragma once

#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev/DBuf/MakeTensor/Reshape
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"   // NVFP4 W4A16 dispatch
#include "vllm/model_executor/models/device_pool.h"  // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3.h"         // Qwen3DenseAttnWeights, PagedKvCache
#include "vllm/platforms/interface.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace dense_attn {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;

// VT_FUSED_CHAIN_ADOPT (default ON, consistent with the framework): route the
// norm and qk-norm-rope preambles through the declared fusion recipes
// (kFusedAddRmsNormStd / kAttnQkNormRope) via vt::FusedChain. The Tier-0
// composite dispatches to the SAME standalone vt:: ops, so it is byte-identical
// to the hand-call fallback (=0). Read once (getenv is cheap; process-stable).
inline bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// VT_QWEN3_ROPE_CACHE (DEFAULT ON) — route the bf16 attention preamble's RoPE
// through the per-step cos/sin cache (RopeFromCache) instead of RopeNeox's
// per-element fp64 pow/cos/sin recompute. MEASURED the dominant dense-TTFT lever
// (Qwen3-4B c1 median TTFT 209->135 ms = median parity vs graphed vLLM; c8 316->
// 144 ms), and vLLM-faithful (bf16 cos/sin cache == vLLM RotaryEmbedding).
//
// FLIPPED default-ON 2026-07-20 after the opt-in rationale was GROUNDED AND
// DISPROVEN on GB10. RopeNeox and RopeFromCache are distinct CUDA __global__s so
// nvcc contracts `x*c - y*sn` to FMA differently — a deterministic 1-ULP shift
// that moves two genuine bf16 near-tie tokens (0.6B p0 tok5, 4B p1 tok6) and so
// requires regenerating the near-tie goldens (done: our_ids + neartie_gap). The
// claimed second reason — that the shift landed on a run-to-run "FA2 split-KV
// combine non-determinism" making the SACRED gate flaky — was NOT reproducible:
// the paged engine is byte-DETERMINISTIC run-to-run on the gate battery (K=4
// RoPE-off + K=3 RoPE-on, identical token ids every run), and for the short gate
// contexts num_splits==1 so the split-KV combine kernel is never even launched.
// With the goldens regenerated the near-tie gate PASSES 16/16 on both sizes
// (0.6B max gap 0.125 nats, 4B 0.250 nats — all divergences are genuine bf16
// near-ties, none over the 0.5-nat band). Opt out with VT_QWEN3_ROPE_CACHE=0 for
// a same-binary A/B against the RopeNeox recompute. Read once (process-stable).
inline bool RopeCacheEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN3_ROPE_CACHE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Merged QKVParallelLinear: issue ONE [q|k|v] GEMM over the merged qkv weight
// (mirror vLLM qwen3.py Qwen3Attention.qkv_proj, one F.linear then split via
// QkvSplit), replacing the three per-shard MatmulBT GEMMs. It folds the two tiny
// GQA k/v GEMMs (N=Hkv*Dh) into one wide, tensor-core-efficient GEMM and cuts 2
// GEMM launches/layer.
//
// MEASURED NEUTRAL (2026-07-21, Qwen3-4B GB10, merge ON vs OFF same binary):
// c1 tput 187.1 vs 185.7 (+0.8%), TPOT 46.8 vs 47.2; c8 tput 1314 vs 1308
// (+0.4%), TPOT 50.0 vs 50.4, P99 ITL 266 vs 258 — all within run-to-run noise.
// c8 decode is 93% GPU-busy (compute-bound); merging does not cut FLOPs and the
// launch saving is negligible against ~82k decode launches, so there is no win.
// It is also byte-AFFECTING (cuBLASLt picks a different K-reduction for the wider
// merged N, flipping ONE Qwen3-0.6B near-tie token — Qwen3-4B stays 16/16 exact),
// which would force regenerating the SACRED 0.6B near-tie golden for no benefit.
//
// So it ships DEFAULT OFF (opt-in): the default path is the byte-identical
// 3-shard GEMM that matches the committed near-tie goldens. Set VT_QWEN3_QKV_MERGE=1
// to exercise the vLLM-structural merged path (the QkvSplit primitive stays
// available for a future decode-fusion that could make it pay). Read once.
inline bool Qwen3QkvMergeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN3_QKV_MERGE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Dev / MakeTensor / Reshape / DevicePoolPolicy / DBuf were RELOCATED VERBATIM
// to dense_device_glue.h (included above) so dense_nvfp4_gemm.h can layer
// beneath AttnBlock without an include cycle. They remain in namespace
// `vllm::dense_attn`, so every consumer resolves them exactly as before.

inline std::vector<float> WeightF32(const OwnedTensor& w) {
  const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  return out;
}

// Device-resident raw-dtype view over an owned weight, uploaded ONCE (lazily) and
// reused across every forward step (mirrors qwen3_5.cpp ResidentWeight).
inline Tensor ResidentWeight(Dev d, const OwnedTensor& w, std::vector<int64_t> shape = {}) {
  if (shape.empty()) shape.assign(w.shape, w.shape + w.rank);
  // HOST-POINTER ALIASING IS A CPU PROPERTY, NOT A "NOT-CUDA" PROPERTY.
  // This read `!is_cuda()`, which is true for kMETAL, kVULKAN and kXPU as well
  // as kCPU — so any DEVICE backend other than CUDA aliased the host weight
  // bytes straight into a tensor and handed a HOST pointer to a DEVICE kernel.
  // Latent only because no model runs on a non-CUDA device backend yet; a hard
  // blocker for the Metal/Vulkan model bring-up
  // (.agents/specs/metal-mlx-reuse-study.md §3.3 item 2). The correct predicate
  // is `is_cpu()`: alias when the "device" IS the host, upload otherwise.
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu())
    return MakeTensor(const_cast<uint8_t*>(w.bytes.data()), w.dtype, d.q.device, shape);
  if (!w.d_dev) {
    const size_t nb = w.bytes.size();
    void* p = d.b.Alloc(nb);
    d.b.Copy(d.q, p, w.bytes.data(), nb);
    Backend* bk = &d.b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_dev.get(), w.dtype, d.q.device, shape);
}

// Device-resident f32 upcast of a bf16 owned weight (per-head q/k norm weights,
// consumed by the f32 RMSNorm), uploaded ONCE.
inline Tensor ResidentWeightF32(Dev d, const OwnedTensor& w, const std::vector<int64_t>& shape) {
  if (!w.d_dev_f32) {
    std::vector<float> f = WeightF32(w);
    // Same defect, same fix as ResidentWeight above: `!is_cuda()` aliased a
    // host std::vector<float> into a tensor for kMETAL/kVULKAN/kXPU.
    if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu()) {
      auto* buf = new std::vector<float>(std::move(f));
      w.d_dev_f32 = std::shared_ptr<void>(buf->data(), [buf](void*) { delete buf; });
    } else {
      const size_t nb = f.size() * sizeof(float);
      void* p = d.b.Alloc(nb);
      d.b.Copy(d.q, p, f.data(), nb);
      Backend* bk = &d.b;
      w.d_dev_f32 = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    }
  }
  return MakeTensor(w.d_dev_f32.get(), DType::kF32, d.q.device, shape);
}

// The two dim-1 unbind slices of the flash KV cache (num_blocks, 2, block_size,
// Hkv, Dh): a rank-4 strided view (block stride 2*bs*Hkv*Dh). Mirrors
// qwen3_5.cpp KvSlice.
inline Tensor KvSlice(const PagedKvCache& kv, vt::Device dev, int which) {
  const int64_t bs = kv.block_size, h = kv.num_kv_heads, dd = kv.head_size;
  Tensor t;
  t.data = static_cast<char*>(kv.data) +
           static_cast<size_t>(which) * static_cast<size_t>(bs * h * dd) *
               vt::SizeOf(kv.dtype);
  t.dtype = kv.dtype;
  t.device = dev;
  t.rank = 4;
  t.shape[0] = kv.num_blocks;
  t.shape[1] = bs;
  t.shape[2] = h;
  t.shape[3] = dd;
  t.stride[0] = 2 * bs * h * dd;
  t.stride[1] = h * dd;
  t.stride[2] = dd;
  t.stride[3] = 1;
  return t;
}

// Per-step device inputs, uploaded ONCE (positions/attention metadata + the
// per-step cos|sin cache), reused by every one of the N full-attn layers.
struct StepInputs {
  DBuf positions;        // i32 [T]
  DBuf slot_mapping;     // i64 [T]
  DBuf block_table;      // i32 [num_reqs, cols]
  DBuf seq_lens;         // i32 [num_reqs]
  DBuf query_start_loc;  // i32 [num_reqs+1]
  DBuf cos_sin;          // f32 [T, rotary_dim]   (per-step cos|sin, f32 A/B path)
  DBuf cos_sin_bf16;     // bf16 [T, rotary_dim]  (vLLM-dtype cache, bf16 rope)
  DBuf rope_row_idx;     // i32 [T] = 0..T-1 (token-index lookup into the cache)
};

inline StepInputs BuildStepInputs(Dev d, const std::vector<int32_t>& positions,
                                  const CommonAttentionMetadata& am, const HfConfig& cfg) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const int rot = static_cast<int>(cfg.rotary_dim);
  // Identity row index 0..T-1: the per-step cos|sin cache is TOKEN-indexed (row t
  // already holds f(positions[t])), so RopeFromCache — which looks the cache up by
  // its `positions` argument — must be fed the identity so it reads cache[t], the
  // correct angle for token t at ANY real position (prefill AND decode). Feeding
  // the real positions would double-apply the position map (and skip decode rows
  // whose position >= T, the cache row count). Mirrors the token-indexed fused
  // AttnQkNormRope(Gate) consumer used by the 27B/35B forward.
  // CUDA-GRAPH SAFETY (W7): the identity row index is uploaded by a host->device
  // copy that a decode CUDA graph CAPTURES as a memcpy node, baking this host
  // SOURCE ADDRESS and re-reading it on every replay. A stack-local vector here
  // (the pre-W7 form) is destroyed the moment BuildStepInputs returns, so every
  // replay would read freed stack memory — MEASURED as a wrong RoPE and a wrong
  // token on the first captured decode step of Qwen3-Coder. It is instead served
  // from a process-persistent per-T table: the contents are a pure function of T
  // (row_idx[i] == i), the storage is created once per distinct T and NEVER
  // resized or moved (a std::map node's vector never reallocates), so a captured
  // pointer stays valid for the process lifetime. Byte-identical contents to the
  // stack-local form, so the eager (non-graph) dense path is unchanged.
  //
  // Same discipline as the graph-safe persistent scratch on the device side
  // (EnsureMoeScratch / EnsureMoePartials, cuda_matmul_nvfp4.cu:767,986: RETIRE,
  // never free/move, because the pointer is baked into a captured graph).
  static std::mutex row_idx_mu;
  static std::map<int64_t, std::vector<int32_t>> row_idx_by_t;
  const std::vector<int32_t>* row_idx = nullptr;
  {
    std::lock_guard<std::mutex> lk(row_idx_mu);
    auto it = row_idx_by_t.find(T);
    if (it == row_idx_by_t.end()) {
      std::vector<int32_t> v(static_cast<size_t>(T));
      for (int64_t i = 0; i < T; ++i) v[static_cast<size_t>(i)] = static_cast<int32_t>(i);
      it = row_idx_by_t.emplace(T, std::move(v)).first;
    }
    row_idx = &it->second;
  }
  StepInputs s{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kF32, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kBF16, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kI32, {T}, row_idx->data()),
  };
  if (rot > 0) {
    // Build the per-step cos|sin cache ONCE (RopeCosSinCache does the fp64
    // pow/cos/sin over T*rot/2 elements, keyed by the REAL positions), so the
    // per-layer attention preamble reads it via RopeFromCache instead of
    // recomputing the fp64 transcendentals N times (the old RopeNeox). The cache
    // is f32; RopeCosSinCacheKernel uses the SAME double-angle math as
    // RopeNeoxKernel, so an f32-precision RoPE off this cache is BIT-IDENTICAL to
    // RopeNeox (cuda_ops.cu:720).
    vt::RopeCosSinCache(d.q, s.cos_sin.t(), s.positions.t(),
                        vt::RopeArgs{static_cast<float>(cfg.rope_theta), rot});
    // The bf16 model-dtype cache is only consumed by the opt-in RopeFromCache
    // path; skip the cast on the default (RopeNeox) path so default-OFF stays
    // perf-neutral vs baseline.
    if (RopeCacheEnabled()) vt::CastBf16(d.q, s.cos_sin_bf16.t(), s.cos_sin.t());
  }
  return s;
}

// One Qwen3 dense self-attention block (qwen3.py::Qwen3Attention.forward). `dhn`
// is the input-normed hidden [T,H] bf16; returns the o_proj output [T,H] bf16.
inline DBuf AttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
                      const Tensor& dhn, const StepInputs& si,
                      const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  VT_CHECK(w.qkv_bias.Empty(),
           "qwen3 dense forward: attention_bias not supported yet (Qwen3-0.6B has none)");
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "qwen3 dense: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "qwen3 dense: KV cache head dims mismatch config");

  // vLLM runs Qwen3-0.6B in BF16: the qkv GEMM, per-head q/k RMSNorm (variance in
  // f32, result rounded to bf16), RoPE, and flash attention all flow bf16. Token-
  // exactness with the bf16 oracle requires mirroring that rounding exactly, so
  // the DEFAULT path keeps q/k/v/query/attn in bf16 (matching vLLM's per-op bf16
  // stores). VT_QWEN3_ATTN_F32=1 selects the f32 A/B path (q/k RMSNorm+RoPE in
  // f32, exercising the kAttnQkNormRope catalog recipe via cos/sin cache) — a
  // more-precise deviation, kept for diagnostics.
  const bool attn_f32 = [] {
    const char* e = std::getenv("VT_QWEN3_ATTN_F32");
    return e != nullptr && e[0] == '1';
  }();
  const DType adt = attn_f32 ? DType::kF32 : DType::kBF16;

  // Merged QKVParallelLinear. The weight owner is ONE raw-NK [qdim+2kdim, H]
  // tensor (vLLM's stacked qkv_proj). Two dispatch paths:
  //  - MERGE (default): one MatmulBT over the whole owner -> [T, qdim+2kdim],
  //    then QkvSplit into the contiguous q/k/v shards. Mirrors vLLM's single
  //    qkv GEMM + .split([q,kv,kv]) and folds the tiny GQA k/v GEMMs into one
  //    wide, tensor-core-efficient GEMM. Byte-affecting (near-tie-gated).
  //  - 3-SHARD (VT_QWEN3_QKV_MERGE=0): slice the owner's output rows and project
  //    each shard separately (the byte-identical baseline for the A/B).
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  if (w.IsNvfp4()) {
    // NVFP4 W4A16 qkv — ALWAYS the merged form: vLLM owns exactly one merged
    // `qkv_proj` parameter and repacks it WHOLE into one Marlin operand
    // (marlin_utils_fp4.py:221-306), so there is no 3-shard analog to A/B here
    // (slicing a Marlin-interleaved operand is not row-addressable). The a16
    // activation must be bf16, which is the default path's dtype.
    VT_CHECK(adt == DType::kBF16,
             "qwen3 dense: NVFP4 W4A16 qkv requires a bf16 activation "
             "(VT_QWEN3_ATTN_F32=1 is not supported on the quantized path)");
    DBuf qkv = dense_nvfp4::MatmulNvfp4W4A16D(d, dhn, w.qkv_proj_fp4, adt);
    vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
  } else {
    Tensor wqkv = ResidentWeight(d, w.qkv_proj);
    if (Qwen3QkvMergeEnabled()) {
      DBuf qkv(d, adt, {T, qdim + 2 * kdim});
      vt::MatmulBT(d.q, qkv.t(), dhn, wqkv);
      vt::QkvSplit(d.q, q.t(), k.t(), v.t(), qkv.t());
    } else {
      Tensor wq = wqkv.Slice(0, 0, qdim);
      Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
      Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
      vt::MatmulBT(d.q, q.t(), dhn, wq);
      vt::MatmulBT(d.q, k.t(), dhn, wk);
      vt::MatmulBT(d.q, v.t(), dhn, wv);
    }
  }

  // Per-head q/k RMSNorm (RMSNorm(head_dim), non-gemma) BEFORE partial NeoX RoPE.
  Tensor q2 = Reshape(q.t(), {T * Hq, Dh});
  Tensor k2 = Reshape(k.t(), {T * Hkv, Dh});
  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  if (attn_f32 && FusedChainAdoptEnabled() && rot > 0) {
    // f32 A/B ADOPT: the whole preamble through vt::FusedChain(kAttnQkNormRope) —
    // the Tier-0 composite = RmsNorm(q,false) + RmsNorm(k,false) + RopeFromCache,
    // byte-identical to the hand-call (test_ops_fused_chain.cpp). Qwen3's reuse of
    // the fusion catalog's non-gated qk-norm-rope recipe (f32 cos/sin cache).
    Tensor wqn = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor wkn = ResidentWeightF32(d, w.k_norm, {Dh});
    vt::FusedBinding b;
    b.op[0] = &q2;
    b.op[1] = &wqn;
    b.op[2] = &k2;
    b.op[3] = &wkn;
    b.op[4] = &q3;
    b.op[5] = &k3;
    b.op[6] = const_cast<Tensor*>(&si.cos_sin.t());
    b.op[7] = const_cast<Tensor*>(&si.positions.t());
    b.n = 8;
    vt::FusedParams p;
    p.eps = eps;
    p.rope = vt::RopeArgs{base, rot};
    vt::FusedChain(d.q, vt::kAttnQkNormRope, b, p);
  } else {
    // BF16 attention preamble: standalone per-head RMSNorm (weight dtype == q
    // dtype) then partial NeoX RoPE. The norm weight follows q's dtype so bf16 q ·
    // bf16 q_norm, matching vLLM's per-op bf16 stores.
    //
    // RoPE has two paths (RopeCacheEnabled / VT_QWEN3_ROPE_CACHE):
    //  - DEFAULT (OFF): in-place RopeNeox — byte-identical + deterministic (the
    //    committed near-tie goldens + the stable SACRED gate).
    //  - OPT-IN (ON): RopeFromCache off the per-step cos|sin cache — MEASURED the
    //    dominant dense-TTFT lever (recomputing fp64 pow/cos/sin per element per
    //    layer, RopeNeox, is 36.5% of prefill GPU-busy; fp64 ≈ 1/64 fp32 on GB10).
    //    The cache is cast to BF16 so bf16 q/k rotate against a bf16 cos|sin —
    //    EXACTLY vLLM's RotaryEmbedding (base.cpp initialize_cache()), the
    //    vLLM-FAITHFUL rope. See RopeCacheEnabled() for why it is opt-in (CUDA
    //    FMA non-byte-identity + engine near-tie nondeterminism → flaky gate).
    //
    // Token-indexed lookup (opt-in path): cache row t already encodes positions[t], so
    // RopeFromCache is fed the identity row index (si.rope_row_idx) — NOT the
    // real positions — so cache[t] is read for token t at ANY position (the
    // position map is already applied when the cache is built), correct for
    // prefill AND decode.
    Tensor wqn = attn_f32 ? ResidentWeightF32(d, w.q_norm, {Dh})
                          : ResidentWeight(d, w.q_norm, {Dh});
    Tensor wkn = attn_f32 ? ResidentWeightF32(d, w.k_norm, {Dh})
                          : ResidentWeight(d, w.k_norm, {Dh});
    vt::RmsNorm(d.q, q2, q2, wqn, vt::RmsNormArgs{eps, false});
    vt::RmsNorm(d.q, k2, k2, wkn, vt::RmsNormArgs{eps, false});
    if (RopeCacheEnabled() && rot > 0) {
      Tensor k3v = k3;
      vt::RopeFromCache(d.q, q3, &k3v, si.rope_row_idx.t(), si.cos_sin_bf16.t(),
                        vt::RopeArgs{base, rot});
    } else {
      // DEFAULT (byte-identical, deterministic): in-place bf16 NeoX RoPE with
      // per-element fp64 cos/sin, mirroring vLLM's rotary_emb bf16 rounding.
      vt::RopeNeox(d.q, q3, k3, si.positions.t(), vt::RopeArgs{base, rot});
    }
  }

  // v [T,Hkv,Dh] view.
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});

  // Write the rope'd K + V into the paged cache, then run causal GQA paged
  // attention over the cache. On the bf16 default the K/V/query are ALREADY bf16
  // (matching the bf16 cache directly, no cast); the f32 A/B down-casts K/V for a
  // bf16 cache.
  // The "auto" ReshapeAndCache copy requires the K/V dtype == cache dtype. Cast
  // K/V to the cache dtype only when they differ (bf16 default + bf16 cache =
  // no-op, the common production case).
  Tensor kw = k3;
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    }
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, adt, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  // o_proj (RowParallelLinear, no bias): [T, Hq*Dh] -> [T,H] bf16. The attention
  // output is already bf16 on the default path (matching vLLM's bf16 flash-attn
  // output); the f32 A/B down-casts it so MatmulBT's inputs share dtype.
  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  DBuf attn_bf(d, DType::kBF16, {T, Hq * Dh});
  if (adt != DType::kBF16) {
    vt::CastBf16(d.q, attn_bf.t(), Reshape(attn.t(), {T, Hq * Dh}));
    o_in = attn_bf.t();
  }
  if (w.IsNvfp4()) {
    // NVFP4 W4A16 o_proj — the bf16 attention output IS the a16 activation.
    DBuf o = dense_nvfp4::MatmulNvfp4W4A16D(d, o_in, w.o_proj_fp4, DType::kBF16);
    (void)rot;
    return o;
  }
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  (void)rot;
  return o;
}

}  // namespace dense_attn
}  // namespace vllm
