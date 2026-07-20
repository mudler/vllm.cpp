// Qwen3 DENSE (`Qwen3ForCausalLM`) forward — the first ADDITIVE-MODEL bring-up
// W3 (the capstone). A pure standard-dense transformer forward COMPOSED from the
// public vt:: ops + the fusion catalog (kFusedAddRmsNormStd / kAttnQkNormRope,
// include/vt/recipes.h), with NO GDN, NO MoE and NO attention output gate. It is
// the Qwen3.6-dense full-attention path (qwen3_5.cpp DenseForwardLayers /
// FullAttnBlockPaged) stripped to the pure-dense subset:
//   - ONE full-attention KV group per layer (no MambaSpec/GDN, no hybrid split);
//   - STANDARD (non-gemma) RMSNorm at input/post/final norms;
//   - per-head q_norm/k_norm (RMSNorm(head_dim), non-gemma) applied BEFORE RoPE;
//   - NO attention gate (Qwen3 has none);
//   - a TIED lm_head (aliases embed_tokens).
//
// Grounding: vllm/model_executor/models/qwen3.py @ e24d1b24
//   Qwen3Attention (:65-168), Qwen3MLP=Qwen2MLP (:58), Qwen3DecoderLayer
//   (:171-242), Qwen3Model=Qwen2Model (:260), tied lm_head (:294-295).
// See .agents/specs/first-additive-model-qwen3-dense.md §2/§4/§6.
//
// Numeric contract (mirrors the qwen3_5 full-attention FALLBACK path — the
// token-exact paged==dense anchor): the residual stream is the model dtype (bf16,
// matching vLLM's fused_add_rms_norm residual); the qkv GEMM emits f32 q/k/v so
// the per-head q/k RMSNorm + RoPE run in f32; the paged KV cache is written bf16
// (down-cast K/V) while the query stays f32 into vt::PagedAttention; o_proj and
// the whole MLP flow bf16. Returns [n_out, vocab] f32 logits.
//
// Self-contained device glue (Dev/DBuf/ResidentWeight): the DBuf here draws its
// scratch from the SHARED DevicePool (include/vllm/model_executor/models/
// device_pool.h — extracted verbatim from qwen3_5.cpp), so the dense forward
// reuses freed blocks instead of a per-op cudaMalloc/cudaFree. This is a pure
// allocation-source change (identical computation ⇒ byte-identical output; all
// gate models unchanged). NOTE: a clean same-binary A/B (Qwen3-4B c1+c8)
// measured the pool PERF-NEUTRAL on this model — the async scheduler already
// overlaps the host-side alloc syncs with GPU compute; it is kept as byte-safe
// hygiene + code sharing, not a measured TTFT lever. The real dense-TTFT lever
// is the RoPE cos|sin cache below.
#include "vllm/model_executor/models/qwen3.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/device_pool.h"     // DevicePool/Pool/ActivePool (shared)
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace {

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
bool FusedChainAdoptEnabled() {
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
bool RopeCacheEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN3_ROPE_CACHE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

struct Dev {
  Backend& b;
  Queue& q;
};

Tensor MakeTensor(void* data, DType dt, vt::Device dev,
                  const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

Tensor Reshape(const Tensor& src, const std::vector<int64_t>& shape) {
  return MakeTensor(src.data, src.dtype, src.device, shape);
}

// The device-scratch residency policy (BACKEND-PLATFORM item 2), resolved from
// the running device's platform. The DevicePool soft cap is platform data (0 ==
// uncapped, GB10 today ⇒ pool behavior byte-for-byte unchanged). Memoized in a
// function-local static: DBuf is a per-op hot path and the process runs on ONE
// device, so the virtual dispatch is paid exactly once. Mirrors qwen3_5.cpp.
struct DevicePoolPolicy {
  size_t cap_bytes = 0;  // residency_policy().device_pool_cap_bytes (0 == uncapped)
};
DevicePoolPolicy ResolveDevicePoolPolicy(const Dev& d) {
  static const DevicePoolPolicy p = [&] {
    const auto rp =
        vllm::platforms::GetPlatform(d.q.device.type).residency_policy();
    return DevicePoolPolicy{rp.device_pool_cap_bytes};
  }();
  return p;
}

// Owned device allocation + tensor view, routed through the SHARED DevicePool so
// the buffer's storage is reused rather than freed to the driver (avoiding the
// per-op cudaMalloc/cudaFree sync). Move-only, RAII. Ported verbatim from the
// qwen3_5.cpp pooled DBuf (device_pool.h Pool()/ActivePool()).
class DBuf {
 public:
  DBuf(Dev d, DType dt, const std::vector<int64_t>& shape,
       const void* host = nullptr)
      : b_(&d.b) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    alloc_bytes_ = bytes_ == 0 ? 1 : bytes_;
    cap_ = ResolveDevicePoolPolicy(d).cap_bytes;
    pool_ = ActivePool();
    p_ = pool_->Get(*b_, alloc_bytes_);
    t_ = MakeTensor(p_, dt, d.q.device, shape);
    if (host != nullptr && bytes_ > 0) b_->Copy(d.q, p_, host, bytes_);
  }
  ~DBuf() { if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept
      : b_(o.b_), pool_(o.pool_), p_(o.p_), bytes_(o.bytes_),
        alloc_bytes_(o.alloc_bytes_), cap_(o.cap_), t_(o.t_) {
    o.p_ = nullptr;
  }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) {
      if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_);
      b_ = o.b_;
      pool_ = o.pool_;
      p_ = o.p_;
      bytes_ = o.bytes_;
      alloc_bytes_ = o.alloc_bytes_;
      cap_ = o.cap_;
      t_ = o.t_;
      o.p_ = nullptr;
    }
    return *this;
  }

  Tensor& t() { return t_; }
  const Tensor& t() const { return t_; }
  void* ptr() { return p_; }
  size_t bytes() const { return bytes_; }
  size_t alloc_bytes() const { return alloc_bytes_; }
  void Zero(Dev d) { b_->Memset(d.q, p_, 0, bytes_); }
  void Download(Dev d, void* host) {
    b_->Copy(d.q, host, p_, bytes_);
    b_->Synchronize(d.q);
  }
  // Relinquish the pool block WITHOUT returning it (dtor becomes a no-op); the
  // caller takes over the Pool().Put obligation for alloc_bytes().
  void* Release() {
    void* p = p_;
    p_ = nullptr;
    return p;
  }

 private:
  Backend* b_;
  DevicePool* pool_ = &Pool();
  void* p_ = nullptr;
  size_t bytes_ = 0;
  size_t alloc_bytes_ = 0;
  size_t cap_ = 0;
  Tensor t_;
};

std::vector<float> WeightF32(const OwnedTensor& w) {
  const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  return out;
}

// Device-resident raw-dtype view over an owned weight, uploaded ONCE (lazily) and
// reused across every forward step (mirrors qwen3_5.cpp ResidentWeight).
Tensor ResidentWeight(Dev d, const OwnedTensor& w, std::vector<int64_t> shape = {}) {
  if (shape.empty()) shape.assign(w.shape, w.shape + w.rank);
  if (!vllm::platforms::GetPlatform(d.q.device.type).is_cuda())
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
Tensor ResidentWeightF32(Dev d, const OwnedTensor& w, const std::vector<int64_t>& shape) {
  if (!w.d_dev_f32) {
    std::vector<float> f = WeightF32(w);
    if (!vllm::platforms::GetPlatform(d.q.device.type).is_cuda()) {
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
Tensor KvSlice(const PagedKvCache& kv, vt::Device dev, int which) {
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

StepInputs BuildStepInputs(Dev d, const std::vector<int32_t>& positions,
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
  std::vector<int32_t> row_idx(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) row_idx[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  StepInputs s{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kF32, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kBF16, {T, rot > 0 ? rot : 1}),
      DBuf(d, DType::kI32, {T}, row_idx.data()),
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
DBuf AttnBlock(Dev d, const Qwen3DenseAttnWeights& w, const HfConfig& cfg,
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

  // Merged QKVParallelLinear executed as its logical [q|k|v] shards: slice the
  // ONE raw-NK [qdim+2kdim, H] owner's output rows (contiguous sub-blocks) and
  // project each into a contiguous buffer via MatmulBT (bf16 in, bf16/f32 out).
  Tensor wqkv = ResidentWeight(d, w.qkv_proj);
  Tensor wq = wqkv.Slice(0, 0, qdim);
  Tensor wk = wqkv.Slice(0, qdim, qdim + kdim);
  Tensor wv = wqkv.Slice(0, qdim + kdim, qdim + 2 * kdim);
  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kdim});
  DBuf v(d, adt, {T, kdim});
  vt::MatmulBT(d.q, q.t(), dhn, wq);
  vt::MatmulBT(d.q, k.t(), dhn, wk);
  vt::MatmulBT(d.q, v.t(), dhn, wv);

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
  Tensor wo = ResidentWeight(d, w.o_proj);
  DBuf o(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, o.t(), o_in, wo);
  (void)rot;
  return o;
}

// Dense SwiGLU MLP (qwen3.py::Qwen3MLP=Qwen2MLP): merged gate_up_proj ->
// SiluAndMul -> down_proj, all bf16. `dh2` is the post-norm hidden [T,H] bf16.
DBuf MlpBlock(Dev d, const Qwen3DenseMlpWeights& w, const HfConfig& cfg,
              const Tensor& dh2, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;
  Tensor wgu = ResidentWeight(d, w.gate_up_proj);  // [2I, H] raw-NK
  DBuf gate_up(d, DType::kBF16, {T, 2 * I});
  vt::MatmulBT(d.q, gate_up.t(), dh2, wgu);
  DBuf act(d, DType::kBF16, {T, I});
  vt::SiluAndMul(d.q, act.t(), gate_up.t());  // silu(gate)*up
  Tensor wdn = ResidentWeight(d, w.down_proj);  // [H, I] raw-NK
  DBuf out(d, DType::kBF16, {T, H});
  vt::MatmulBT(d.q, out.t(), act.t(), wdn);
  return out;
}

// One dense decoder layer (qwen3.py::Qwen3DecoderLayer): input norm (std
// add+RMSNorm) -> attention -> post norm (std add+RMSNorm) -> MLP. `hidden` (bf16
// [T,H]) is the delta; `res` (bf16 [T,H]) the residual accumulator.
void RunLayer(Dev d, const Qwen3DenseLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const StepInputs& si,
              const CommonAttentionMetadata& meta, const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor w_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dhn.t(), hidden.t(), w_in, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dhn.t(), hidden.t(), w_in, vt::RmsNormArgs{eps, false}, &res.t());
  }

  DBuf attn = AttnBlock(d, layer.attn, cfg, dhn.t(), si, meta, kv, T);

  Tensor w_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dh2.t(), attn.t(), w_post, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dh2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());
  }

  hidden = MlpBlock(d, layer.mlp, cfg, dh2.t(), T);
}

// GatherRows: gather the idx-indexed rows of `src` [.,H] into contiguous `dst`.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// embed -> N layers -> final RMSNorm -> lm_head. Returns [n_out, vocab] f32 as a
// device DBuf (no host Download; the caller Downloads or wraps it).
DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions,
                 const CommonAttentionMetadata& attn_meta,
                 const std::vector<PagedKvCache>& attn_kv,
                 const Qwen3DenseWeights& weights, const HfConfig& config,
                 const std::vector<int32_t>& logits_indices) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  const float eps = static_cast<float>(config.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3 dense: positions length must match token_ids");
  VT_CHECK(attn_kv.size() == static_cast<size_t>(config.num_hidden_layers),
           "qwen3 dense: one PagedKvCache per layer required");

  // Embed: hidden[T,H] bf16 = embed_tokens[token_ids].
  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }

  DBuf res(d, DType::kBF16, {T, H});
  res.Zero(d);

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res, si,
             attn_meta, attn_kv[static_cast<size_t>(l)], T);

  // Final RMSNorm over the fused stream (res += hidden; std norm), then lm_head.
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm.t(), hidden.t(), w_fn, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm.t(), hidden.t(), w_fn, vt::RmsNormArgs{eps, false}, &res.t());
  }

  // lm_head. Tied (Qwen3-0.6B): logits = hidden @ embed_tokens^T via MatmulBT
  // over the [vocab,H] embed table (== [N=vocab,K=H]). Untied: the loaded
  // Matmul-B [H,vocab] lm_head via vt::Matmul.
  const bool tied = weights.tie_word_embeddings || weights.lm_head.Empty();
  Tensor lm = tied ? ResidentWeight(d, weights.embed_tokens, {vocab, H})
                   : ResidentWeight(d, weights.lm_head);

  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  Tensor src = dnorm.t();
  DBuf dgather(d, DType::kBF16, do_gather ? std::vector<int64_t>{
                                                static_cast<int64_t>(logits_indices.size()), H}
                                          : std::vector<int64_t>{1, 1});
  if (do_gather) {
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    src = dgather.t();
  }
  const int64_t n_out = src.shape[0];
  DBuf logits(d, DType::kF32, {n_out, vocab});
  if (tied)
    vt::MatmulBT(d.q, logits.t(), src, lm);
  else
    vt::Matmul(d.q, logits.t(), src, lm);
  return logits;
}

ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();
  // The pool block's lifetime moves into a shared_ptr whose deleter returns it to
  // the DevicePool — no per-step cudaMalloc/cudaFree, and the buffer safely
  // outlives sampling (mirrors qwen3_5.cpp WrapDeviceLogits).
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();
  fl.device_storage =
      std::shared_ptr<void>(p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

}  // namespace

std::vector<float> Qwen3DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3DenseModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const std::vector<PagedKvCache>& attn_kv,
    const Qwen3DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights,
                             config, logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  return WrapDeviceLogits(d, std::move(dlogits), n_out, config.vocab_size);
}

}  // namespace vllm
