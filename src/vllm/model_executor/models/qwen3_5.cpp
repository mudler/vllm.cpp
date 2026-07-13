// vllm.cpp original; see qwen3_5.h. Forward math mirrored 1:1 from the pinned
// upstream (qwen3_next.py::Qwen3NextDecoderLayer / Qwen3NextModel.forward,
// qwen_gdn_linear_attn.py, qwen3_next.py::Qwen3NextAttention /
// Qwen3NextSparseMoeBlock @ e24d1b24). References:
// .agents/specs/qwen36-forward-notes.md (assembly, §2 mRoPE->NeoX, §5 attention),
// .agents/specs/gdn-semantics.md (§1 layout, §6 g/beta prep, §7 recurrence),
// .agents/specs/moe-semantics.md (§1-§6 MoE block + activated-expert gather).
#include "vllm/model_executor/models/qwen3_5.h"

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <optional>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {

DenseGateUpGlobals MergeDenseGateUpGlobals(const Nvfp4Weight& gate,
                                           const Nvfp4Weight& up) {
  VT_CHECK(gate.weight_global_scale_inv > 0.0F &&
               up.weight_global_scale_inv > 0.0F,
           "qwen3_5 dense merged gate_up: missing CT weight divisor");
  VT_CHECK(gate.input_global_scale_inv > 0.0F &&
               up.input_global_scale_inv > 0.0F,
           "qwen3_5 dense merged gate_up: missing CT input divisor");
  DenseGateUpGlobals globals;
  globals.input_global_scale_inv =
      std::max(gate.input_global_scale_inv, up.input_global_scale_inv);
  const float weight_global_scale_inv =
      std::max(gate.weight_global_scale_inv, up.weight_global_scale_inv);
  // Preserve vLLM/PyTorch's operation order: reciprocal each selected maximum,
  // then multiply. Do not derive the maximum divisor back from scale2.
  const float input_global_scale = 1.0F / globals.input_global_scale_inv;
  globals.weight_global_scale = 1.0F / weight_global_scale_inv;
  globals.alpha = input_global_scale * globals.weight_global_scale;
  return globals;
}

FullAttnQkvGlobals MergeFullAttnQkvGlobals(const Nvfp4Weight& q,
                                           const Nvfp4Weight& k,
                                           const Nvfp4Weight& v) {
  VT_CHECK(q.weight_global_scale_inv > 0.0F &&
               k.weight_global_scale_inv > 0.0F &&
               v.weight_global_scale_inv > 0.0F,
           "qwen3_5 packed QKV: missing CT weight divisor");
  VT_CHECK(q.input_global_scale_inv > 0.0F &&
               k.input_global_scale_inv > 0.0F &&
               v.input_global_scale_inv > 0.0F,
           "qwen3_5 packed QKV: missing CT input divisor");
  FullAttnQkvGlobals globals;
  globals.input_global_scale_inv =
      std::max({q.input_global_scale_inv, k.input_global_scale_inv,
                v.input_global_scale_inv});
  const float weight_global_scale_inv =
      std::max({q.weight_global_scale_inv, k.weight_global_scale_inv,
                v.weight_global_scale_inv});
  const float input_global_scale = 1.0F / globals.input_global_scale_inv;
  globals.weight_global_scale = 1.0F / weight_global_scale_inv;
  globals.alpha = input_global_scale * globals.weight_global_scale;
  return globals;
}

namespace {

using vt::Backend;
using vt::Device;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using v1::GDNAttentionMetadata;

// Backend + queue bundle threaded through every helper.
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

// Contiguous reinterpret of a device tensor's buffer at a new shape (same numel,
// same dtype/device). Used to view [T,H,D] as [T*H,D] etc. for rank-2 ops.
Tensor Reshape(const Tensor& src, const std::vector<int64_t>& shape) {
  return MakeTensor(src.data, src.dtype, src.device, shape);
}

// Process-wide caching device allocator (M2.x staging elimination). Both
// cudaMalloc AND cudaFree SYNCHRONIZE the whole device, so the per-op DBuf
// alloc/free churn in the forward (thousands of tiny scratch buffers per decode
// step — dominated by the MoE expert loop) was itself a sync storm, larger than
// the Download() syncs. This pool reuses freed blocks (keyed by exact byte size)
// instead of hitting cudaMalloc/cudaFree, so after a brief warm-up almost every
// DBuf lifetime is sync-free.
//
// Reuse is safe under the forward's single-queue (single-stream) ordering: a
// block returned to the pool is only handed back out on the same queue, and CUDA
// stream ordering guarantees the op that last touched the block has completed
// before any reused op runs — no host sync needed. Blocks are never returned to
// the driver (leak at process exit, like the cublasLt workspace); the pool is
// bounded by the forward's peak concurrent scratch. The pool is backend-agnostic
// (CPU malloc/free too — a harmless bounded cache there).
//
// SIZE-CLASS BUCKETING (prefill sync-cudaMalloc kill). Keying the pool on the
// EXACT byte size defeats reuse during prefill: under continuous batching each
// engine step processes a different token count T (prefill-chunk tokens + decode
// tokens), so every [T,*] scratch is a byte size that step-1 never saw — an
// exact-key MISS -> a SYNCHRONOUS cudaMalloc (device-serializing, does NOT
// overlap compute) on the forward's host thread, thousands per prefill. Rounding
// the request UP to a size class (keep the top `kClassBits` significant bits;
// <=1/2^kClassBits ~ 6.25% over-allocation) makes nearby-T scratch of the same
// op share ONE block, so after warm-up almost every prefill DBuf is a pool hit.
// The returned block is >= the requested bytes (the Tensor view uses only the
// logical prefix), and Put/Get round identically so a block always returns to
// its own class bucket. VT_POOL_EXACT=1 restores exact keying (A/B measurement).
class DevicePool {
 public:
  void* Get(Backend& b, size_t bytes) {
    const size_t key = ClassOf(bytes);
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = free_.find(key);
      if (it != free_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        ++hits_;
        return p;
      }
      ++misses_;
    }
    return b.Alloc(key);
  }
  void Put(size_t bytes, void* p) {
    const size_t key = ClassOf(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    free_[key].push_back(p);
  }

  ~DevicePool() {
    if (std::getenv("VT_POOL_STATS") != nullptr) {
      const uint64_t h = hits_.load(), m = misses_.load();
      const double rate = (h + m) ? 100.0 * static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
      std::fprintf(stderr,
                   "[DevicePool] hits=%llu misses(cudaMalloc)=%llu hit-rate=%.2f%% distinct-classes=%zu\n",
                   static_cast<unsigned long long>(h), static_cast<unsigned long long>(m),
                   rate, free_.size());
    }
  }

 private:
  // Round `bytes` up so it keeps at most kClassBits leading significant bits.
  // Exact keying when VT_POOL_EXACT=1 (A/B). Small sizes (< 2^kClassBits) key
  // exactly — there are few of them and the waste would be proportionally large.
  static size_t ClassOf(size_t bytes) {
    static const bool exact = [] {
      const char* e = std::getenv("VT_POOL_EXACT");
      return e != nullptr && e[0] == '1';
    }();
    if (exact || bytes == 0) return bytes == 0 ? 1 : bytes;
    constexpr int kClassBits = 4;  // <=6.25% over-allocation per class
    const int msb = 63 - __builtin_clzll(static_cast<unsigned long long>(bytes));
    if (msb < kClassBits) return bytes;
    const int shift = msb - kClassBits;
    const size_t mask = (static_cast<size_t>(1) << shift) - 1;
    return (bytes + mask) & ~mask;  // round up to a multiple of 2^shift
  }

  std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
};

DevicePool& Pool() {
  static DevicePool p;
  return p;
}

// --- Fused-MoE per-layer resident constants (M2.5 Phase 2, CUDA-graph unblock) -
// MoeBlockFusedCuda used to rebuild + re-upload, EVERY forward step, a set of
// per-layer CONSTANT device buffers: the E fp4-expert device-pointer/scale
// arrays (gate/up/down packed+scale ptrs, scale2) and the pair->token row map
// (tok_map, a function of T only). Those uploads copy from HOST STACK temporaries
// — illegal to have inside a CUDA-graph capture region (their host addresses
// dangle on replay). They are also pure per-step waste (the values never change).
// This process-lifetime cache (keyed by the layer's MoeBlockWeights address)
// uploads them ONCE, during the pre-warm forward, so the captured region only
// READS resident device buffers — no host-sourced copy, nothing to dangle. The
// device buffers leak at process exit (like the cublasLt workspace / the resident
// weights); they are bounded by (num_layers * (9*E + one tok_map per distinct T)).
struct MoeFusedResident {
  void* gp = nullptr;  // i64 [E] device: expert gate packed ptrs
  void* gs = nullptr;  // i64 [E] device: expert gate scale ptrs
  void* up = nullptr;  // i64 [E] device: expert up packed ptrs
  void* us = nullptr;  // i64 [E] device: expert up scale ptrs
  void* dp = nullptr;  // i64 [E] device: expert down packed ptrs
  void* ds = nullptr;  // i64 [E] device: expert down scale ptrs
  void* g2 = nullptr;  // f32 [E] device: expert gate scale2
  void* u2 = nullptr;  // f32 [E] device: expert up scale2
  void* d2 = nullptr;  // f32 [E] device: expert down scale2
  std::unordered_map<int64_t, void*> tok_map;  // T -> i32 [T*top_k] device
  bool ready = false;
};

MoeFusedResident& MoeResidentFor(const MoeBlockWeights* w) {
  static std::mutex mu;
  static std::unordered_map<const MoeBlockWeights*, MoeFusedResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

#ifdef VT_MARLIN_NVFP4
// --- Marlin NVFP4 W4A16 MoE: per-layer resident repacked weights (M0.8 drop-in).
// When VT_NVFP4_MARLIN=1, the routed experts are repacked ONCE at first touch
// into Marlin's interleaved layout (gptq_marlin_moe_repack + S0E5M3 scales +
// processed global scales, all bit-exact to vLLM — tools/marlin/repack_*), and
// the wmma-fp4 fused path is replaced by moe_wna16_marlin_gemm. The original
// per-expert fp4 device copies are freed after repack (the wmma path is unused
// when the gate is on) so peak weight memory stays flat. Buffers leak at exit
// (like the wmma resident / cublasLt workspace).
struct MoeMarlinResident {
  void* w_gate = nullptr;  // i32 [E, K/16, N*2]  (gate: size_k=K, size_n=N)
  void* w_up = nullptr;    // i32 [E, K/16, N*2]
  void* w_down = nullptr;  // i32 [E, N/16, K*2]  (down: size_k=N, size_n=K)
  void* s_gate = nullptr;  // fp8 [E, K/16, N]
  void* s_up = nullptr;
  void* s_down = nullptr;  // fp8 [E, N/16, K]
  void* g_gate = nullptr;  // f32 [E]
  void* g_up = nullptr;
  void* g_down = nullptr;
  // Fused w13 layout (VT_MOE_FUSED_W13): gate+up CONCATENATED along N per expert
  // — rows [0,N) = gate (vLLM w1), rows [N,2N) = up (vLLM w3) — repacked as ONE
  // Marlin B operand of size_n=2N, mirroring vLLM's stacked w13_weight
  // (marlin_utils_fp4.py prepare_nvfp4_moe_layer_for_marlin:374-401 repacks the
  // stacked [E, 2N, K/2] per expert with size_n = num_shards*N). Populated
  // INSTEAD of w_gate/w_up (same total bytes) when fused_w13 is true.
  void* w_gu = nullptr;       // i32 [E, K/16, (2N)*2]
  void* s_gu = nullptr;       // fp8 [E, K/16, 2N]
  void* g_gu = nullptr;       // f32 [E]  (gate scale2 — vLLM w13_weight_scale_2[:, 0])
  bool fused_w13 = false;
  void* workspace = nullptr;  // i32 [sms]
  int sms = 0;
  bool ready = false;
};

MoeMarlinResident& MoeMarlinResidentFor(const MoeBlockWeights* w) {
  static std::mutex mu;
  static std::unordered_map<const MoeBlockWeights*, MoeMarlinResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

bool MarlinMoeEnabled() {
  // Default ON: the vendored Marlin NVFP4 W4A16 GEMM is the validated 35B path
  // (measured gate +22%, decode-heavy +80%, 16/16 token-for-token vs the pinned
  // oracle — see parity-ledger). Only an explicit VT_NVFP4_MARLIN=0 opts back out
  // to the naive redundant-dequant / cublas bf16 GEMM (kept as an escape hatch).
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_MARLIN");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Fused w13 grouped GEMM (VT_MOE_FUSED_W13, DEFAULT ON): run the routed
// experts' gate+up as ONE Marlin grouped GEMM over the N-concatenated w13
// weights (size_n=2I, output [P,2I]) + one SiluAndMul over the halves, instead
// of TWO grouped GEMMs (+2 workspace memsets, 2 schedule passes) + MoeSiluMul;
// same fusion for the shared-expert gate_up (SharedGateUpFusedMarlinD). This is
// exactly vLLM's marlin_moe.py shape: ONE moe_wna16_marlin_gemm with
// size_n = w13_num_shards * N into intermediate_cache1 [M*topk, 2N]
// (fused_moe/experts/marlin_moe.py:133-160), then silu_and_mul on the [:N]/[N:]
// halves (:162-170). At the 35B decode shape (I=512, top_k=8, many tiny
// latency-bound tiles) the second GEMM's fixed costs are pure overhead.
// GATED ON (GB10, 2026-07-10): fused-vs-split BIT-EXACT (test_ops_moe_grouped
// probe), 35B greedy 16/16-vs-oracle BOTH arms, and a clean same-binary A/B win
// (in1024/out128 np200 conc-64, 3+3 interleaved reps: 3166.85 -> 3262.28 tok/s
// = +3.01%, TPOT -3.1%, TTFT -1.4%; MoE-expert fusion alone +0.53%, the rest
// from the shared-expert gate_up fusion). VT_MOE_FUSED_W13=0 opts back out to
// the split two-GEMM layout for A/B.
// The layout choice is made at LOAD (BuildMoeMarlinResident builds either the
// fused or the split resident), so A/B = two runs of the same binary.
bool MoeFusedW13Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_FUSED_W13");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}
#endif  // VT_MARLIN_NVFP4

// Owned device allocation + tensor view. On CPU the backend's Alloc/Copy are
// malloc/memcpy; on CUDA they are cudaMalloc / h2d-d2h on the queue's stream.
// Allocation is routed through the DevicePool so the buffer's storage is reused
// rather than freed to the driver (avoiding the cudaMalloc/cudaFree sync).
class DBuf {
 public:
  DBuf(Dev d, DType dt, const std::vector<int64_t>& shape,
       const void* host = nullptr)
      : b_(&d.b) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    alloc_bytes_ = bytes_ == 0 ? 1 : bytes_;
    p_ = Pool().Get(*b_, alloc_bytes_);
    t_ = MakeTensor(p_, dt, d.q.device, shape);
    if (host != nullptr) b_->Copy(d.q, p_, host, bytes_);
  }
  ~DBuf() { if (p_ != nullptr) Pool().Put(alloc_bytes_, p_); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  // Movable so device-resident block helpers can RETURN a DBuf (the buffer
  // ownership transfers; the moved-from buffer is not returned to the pool).
  DBuf(DBuf&& o) noexcept
      : b_(o.b_), p_(o.p_), bytes_(o.bytes_), alloc_bytes_(o.alloc_bytes_), t_(o.t_) {
    o.p_ = nullptr;
  }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) {
      if (p_ != nullptr) Pool().Put(alloc_bytes_, p_);
      b_ = o.b_;
      p_ = o.p_;
      bytes_ = o.bytes_;
      alloc_bytes_ = o.alloc_bytes_;
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
  // Relinquish ownership of the pool block WITHOUT returning it (the dtor becomes
  // a no-op). The caller takes over the Pool().Put obligation for `alloc_bytes()`.
  // The Tensor view (t()) still holds the raw data pointer after this.
  void* Release() {
    void* p = p_;
    p_ = nullptr;
    return p;
  }
  void Zero(Dev d) { b_->Memset(d.q, p_, 0, bytes_); }
  // Copies the buffer back to host and blocks until the queue is idle.
  void Download(Dev d, void* host) {
    b_->Copy(d.q, host, p_, bytes_);
    b_->Synchronize(d.q);
  }

 private:
  Backend* b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  size_t alloc_bytes_ = 0;
  Tensor t_;
};

float SizeF(int64_t n) { return static_cast<float>(n); }
float Silu(float x) { return x / (1.0F + std::exp(-x)); }

// Upcast a bf16 owned weight to an f32 host buffer (lossless). The CUDA norm /
// conv kernels require the weight dtype to match the activation dtype; where
// activations are f32 (GDN conv/gated-norm, attention qk-norm, final-norm
// replay), the bf16 weight must be presented as f32.
std::vector<float> WeightF32(const OwnedTensor& w) {
  const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  return out;
}

// Device-resident raw-dtype view over an owned weight, uploaded ONCE (lazily)
// and reused across every forward step (mirrors ResidentNvfp4). The forward's
// bf16/f32 weights (embed table, layernorms, attention/GDN projections, router)
// were re-uploaded per op — the ~600MB embed table alone re-copied every step —
// dominating the measured 67.5%-of-wall cudaMemcpyAsync. Caching kills the
// re-upload: the shared_ptr in the (const) weight owns the device buffer for the
// model's lifetime. On CPU the bytes are already host-resident, so a direct view
// avoids the copy. The weight is a read-only matmul-B / norm / embed operand, so
// the const_cast is safe. `shape` defaults to the owned shape.
Tensor ResidentWeight(Dev d, const OwnedTensor& w, std::vector<int64_t> shape = {}) {
  if (shape.empty()) shape.assign(w.shape, w.shape + w.rank);
  if (d.q.device.type != vt::DeviceType::kCUDA)
    return MakeTensor(const_cast<uint8_t*>(w.bytes.data()), w.dtype, d.q.device,
                      shape);
  if (!w.d_dev) {
    const size_t nb = w.bytes.size();
    void* p = d.b.Alloc(nb);
    d.b.Copy(d.q, p, w.bytes.data(), nb);
    Backend* bk = &d.b;
    w.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_dev.get(), w.dtype, d.q.device, shape);
}

// Device-resident f32 upcast of a bf16 owned weight, uploaded ONCE. Matches the
// CUDA norm/conv kernels' requirement that the weight dtype equal the (f32)
// activation dtype (GDN conv1d / gated-norm, attention qk-norm). `shape` is the
// logical view (e.g. {conv_dim, Kw}).
Tensor ResidentWeightF32(Dev d, const OwnedTensor& w,
                         const std::vector<int64_t>& shape) {
  if (!w.d_dev_f32) {
    std::vector<float> f = WeightF32(w);
    if (d.q.device.type != vt::DeviceType::kCUDA) {
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

// y[M,N] f32 = x[M,K] bf16 @ w[K,N] (w owns [K,N] bf16). f32 output keeps the
// GEMM's f32 accumulation for the f32 glue that consumes it.
std::vector<float> MatmulF32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                             int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// y[M,N] bf16 = x[M,K] bf16 @ w[K,N] bf16 (bf16 output mirrors the model's bf16
// hidden states where the result feeds the residual stream / next matmul).
std::vector<uint16_t> MatmulBf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                 int64_t K, const OwnedTensor& w) {
  const int64_t N = w.nk ? w.shape[0] : w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), dx.t(), dw);
  else
    vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}


// --- NVFP4 fp4-resident weight helpers (M2.2b) ------------------------------

// Device-resident views over an Nvfp4Weight's packed + scale buffers. Valid for
// the lifetime of the weight (the buffers are owned by the weight's shared_ptr).
struct Nvfp4Dev {
  Tensor packed;
  Tensor scale;
};

// Upload packed + scale to the device ONCE (lazily, on first use) and keep them
// resident: the shared_ptr in the (const) weight owns the device buffer across
// every forward step, so subsequent calls reuse the resident copy — no per-op
// weight staging. CUDA path only; the deleter frees through the vt Backend.
Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
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

std::vector<int64_t> CutlassFp4ScaleShape(int64_t rows, int64_t k) {
  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return {round_up(rows, 128), round_up(k / 16, 4)};
}

#ifdef VT_CUTLASS_NVFP4
// Resident SWIZZLED weight block scale for the cutlass fp4 GEMM. Computed ONCE
// (lazily) from the resident linear d_scale via vt::SwizzleBlockscale and kept
// on the weight's shared_ptr. Shape [round_up(n,128), round_up(k/16,4)].
Tensor ResidentNvfp4ScaleSwizzled(Dev d, const Nvfp4Weight& w) {
  auto round_up = [](int64_t x, int64_t y) { return (x + y - 1) / y * y; };
  const int64_t Np = round_up(w.n, 128), Kp = round_up(w.k / 16, 4);
  if (!w.d_scale_sw) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);  // ensures d_scale (linear device copy)
    void* p = d.b.Alloc(static_cast<size_t>(Np * Kp));
    Backend* bk = &d.b;
    w.d_scale_sw = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    Tensor sw = MakeTensor(p, DType::kI8, d.q.device, {Np, Kp});
    vt::SwizzleBlockscale(d.q, sw, dw.scale);
  }
  return MakeTensor(w.d_scale_sw.get(), DType::kI8, d.q.device, {Np, Kp});
}

// Device-resident packed operand for the dense CT gate_up_proj. Pinned vLLM
// represents gate/up as one MergedColumnParallelLinear (`qwen2_moe.py:75-115`)
// and loads both checkpoint shards into one N-concatenated parameter
// (`linear.py:580-695`). We retain the split host weights for the diagnostic
// arm, but production uploads one [2I,H/2] packed buffer and swizzles the one
// concatenated [2I,H/16] block-scale buffer exactly once.
struct Nvfp4GateUpDev {
  Tensor packed;
  Tensor scale_sw;
  DenseGateUpGlobals globals;
};

Nvfp4GateUpDev ResidentNvfp4GateUp(Dev d, const DenseMlpWeights& w) {
  const Nvfp4Weight& gate = w.gate_proj_fp4;
  const Nvfp4Weight& up = w.up_proj_fp4;
  VT_CHECK(!gate.Empty() && !up.Empty(),
           "qwen3_5 dense merged gate_up: empty logical shard");
  VT_CHECK(gate.n == up.n && gate.k == up.k,
           "qwen3_5 dense merged gate_up: logical shard shape mismatch");
  const int64_t n = gate.n;
  const int64_t k = gate.k;
  const size_t packed_shard_bytes = gate.packed.bytes.size();
  const size_t scale_shard_bytes = gate.scale.bytes.size();
  VT_CHECK(up.packed.bytes.size() == packed_shard_bytes &&
               up.scale.bytes.size() == scale_shard_bytes,
           "qwen3_5 dense merged gate_up: logical shard byte mismatch");

  if (!w.d_gate_up_packed || !w.d_gate_up_scale_sw) {
    VT_CHECK(!w.d_gate_up_packed && !w.d_gate_up_scale_sw,
             "qwen3_5 dense merged gate_up: partial resident state");
    Backend* backend = &d.b;
    void* packed_data = d.b.Alloc(2 * packed_shard_bytes);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* packed_bytes = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, packed_bytes, gate.packed.bytes.data(), packed_shard_bytes);
    d.b.Copy(d.q, packed_bytes + packed_shard_bytes, up.packed.bytes.data(),
             packed_shard_bytes);

    // Linear scale staging is pool-backed and can return immediately after the
    // swizzle launch: all reuse is on this queue and therefore stream-ordered.
    DBuf scale_linear(d, DType::kI8, {2 * n, k / 16});
    auto* scale_bytes = static_cast<uint8_t*>(scale_linear.ptr());
    d.b.Copy(d.q, scale_bytes, gate.scale.bytes.data(), scale_shard_bytes);
    d.b.Copy(d.q, scale_bytes + scale_shard_bytes, up.scale.bytes.data(),
             scale_shard_bytes);

    const auto round_up = [](int64_t value, int64_t multiple) {
      return (value + multiple - 1) / multiple * multiple;
    };
    const int64_t np = round_up(2 * n, 128);
    const int64_t kp = round_up(k / 16, 4);
    void* scale_sw_data = d.b.Alloc(static_cast<size_t>(np * kp));
    std::shared_ptr<void> scale_sw_owner(
        scale_sw_data, [backend](void* pointer) { backend->Free(pointer); });
    Tensor scale_sw =
        MakeTensor(scale_sw_data, DType::kI8, d.q.device, {np, kp});
    vt::SwizzleBlockscale(d.q, scale_sw, scale_linear.t());

    w.d_gate_up_packed = std::move(packed_owner);
    w.d_gate_up_scale_sw = std::move(scale_sw_owner);
  }

  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return Nvfp4GateUpDev{
      MakeTensor(w.d_gate_up_packed.get(), DType::kI8, d.q.device,
                 {2 * n, k / 2}),
      MakeTensor(w.d_gate_up_scale_sw.get(), DType::kI8, d.q.device,
                 {round_up(2 * n, 128), round_up(k / 16, 4)}),
      MergeDenseGateUpGlobals(gate, up)};
}

// Device-resident packed operand for Qwen3NextAttention's QKVParallelLinear.
// vLLM loads the three logical checkpoint shards into one N-concatenated
// parameter (`qwen3_5.py:279-288`, `qwen3_next.py:252-270`,
// `linear.py:942-1050`) and performs one GEMM before splitting the output.
struct Nvfp4QkvDev {
  Tensor packed;
  Tensor scale_sw;
  FullAttnQkvGlobals globals;
  int64_t qn = 0;
  int64_t kn = 0;
  int64_t vn = 0;
};

Nvfp4QkvDev ResidentNvfp4Qkv(Dev d, const FullAttnLayerWeights& w) {
  const Nvfp4Weight& q = w.q_proj_fp4;
  const Nvfp4Weight& k = w.k_proj_fp4;
  const Nvfp4Weight& v = w.v_proj_fp4;
  VT_CHECK(!q.Empty() && !k.Empty() && !v.Empty(),
           "qwen3_5 packed QKV: empty logical shard");
  VT_CHECK(q.k == k.k && q.k == v.k,
           "qwen3_5 packed QKV: logical shard K mismatch");
  const int64_t total_n = q.n + k.n + v.n;
  const int64_t inner_k = q.k;
  const size_t q_packed_bytes = q.packed.bytes.size();
  const size_t k_packed_bytes = k.packed.bytes.size();
  const size_t v_packed_bytes = v.packed.bytes.size();
  const size_t q_scale_bytes = q.scale.bytes.size();
  const size_t k_scale_bytes = k.scale.bytes.size();
  const size_t v_scale_bytes = v.scale.bytes.size();
  VT_CHECK(q_packed_bytes == static_cast<size_t>(q.n * inner_k / 2) &&
               k_packed_bytes == static_cast<size_t>(k.n * inner_k / 2) &&
               v_packed_bytes == static_cast<size_t>(v.n * inner_k / 2),
           "qwen3_5 packed QKV: packed shard byte mismatch");
  VT_CHECK(q_scale_bytes == static_cast<size_t>(q.n * inner_k / 16) &&
               k_scale_bytes == static_cast<size_t>(k.n * inner_k / 16) &&
               v_scale_bytes == static_cast<size_t>(v.n * inner_k / 16),
           "qwen3_5 packed QKV: scale shard byte mismatch");

  if (!w.d_qkv_packed || !w.d_qkv_scale_sw) {
    VT_CHECK(!w.d_qkv_packed && !w.d_qkv_scale_sw,
             "qwen3_5 packed QKV: partial resident state");
    Backend* backend = &d.b;
    const size_t packed_bytes =
        q_packed_bytes + k_packed_bytes + v_packed_bytes;
    void* packed_data = d.b.Alloc(packed_bytes);
    std::shared_ptr<void> packed_owner(
        packed_data, [backend](void* pointer) { backend->Free(pointer); });
    auto* packed_dst = static_cast<uint8_t*>(packed_data);
    d.b.Copy(d.q, packed_dst, q.packed.bytes.data(), q_packed_bytes);
    d.b.Copy(d.q, packed_dst + q_packed_bytes, k.packed.bytes.data(),
             k_packed_bytes);
    d.b.Copy(d.q, packed_dst + q_packed_bytes + k_packed_bytes,
             v.packed.bytes.data(), v_packed_bytes);

    DBuf scale_linear(d, DType::kI8, {total_n, inner_k / 16});
    auto* scale_dst = static_cast<uint8_t*>(scale_linear.ptr());
    d.b.Copy(d.q, scale_dst, q.scale.bytes.data(), q_scale_bytes);
    d.b.Copy(d.q, scale_dst + q_scale_bytes, k.scale.bytes.data(),
             k_scale_bytes);
    d.b.Copy(d.q, scale_dst + q_scale_bytes + k_scale_bytes,
             v.scale.bytes.data(), v_scale_bytes);

    const auto round_up = [](int64_t value, int64_t multiple) {
      return (value + multiple - 1) / multiple * multiple;
    };
    const int64_t np = round_up(total_n, 128);
    const int64_t kp = round_up(inner_k / 16, 4);
    void* scale_sw_data = d.b.Alloc(static_cast<size_t>(np * kp));
    std::shared_ptr<void> scale_sw_owner(
        scale_sw_data, [backend](void* pointer) { backend->Free(pointer); });
    Tensor scale_sw =
        MakeTensor(scale_sw_data, DType::kI8, d.q.device, {np, kp});
    vt::SwizzleBlockscale(d.q, scale_sw, scale_linear.t());

    w.d_qkv_packed = std::move(packed_owner);
    w.d_qkv_scale_sw = std::move(scale_sw_owner);
  }

  const auto round_up = [](int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };
  return Nvfp4QkvDev{
      MakeTensor(w.d_qkv_packed.get(), DType::kI8, d.q.device,
                 {total_n, inner_k / 2}),
      MakeTensor(w.d_qkv_scale_sw.get(), DType::kI8, d.q.device,
                 {round_up(total_n, 128), round_up(inner_k / 16, 4)}),
      MergeFullAttnQkvGlobals(q, k, v), q.n, k.n, v.n};
}
#endif  // VT_CUTLASS_NVFP4

// Host reference dequant of an fp4 weight to bf16 [K=in, N=out] (Matmul-B
// layout) — the CPU fallback for the fp4 path (no CPU MatmulNvfp4 kernel). Only
// exercised when a real fp4 checkpoint is run on the host device; the CUDA path
// never dequants. Bit-for-bit vllm::DequantNvfp4ToBf16 + transpose.
std::vector<uint16_t> DequantNvfp4ToBLayout(const Nvfp4Weight& w) {
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

// y[M,N] f32 = x[M,K] bf16 @ dequant(w).T, w fp4-resident [N=out, K=in]. Drops
// in for MatmulF32 where the weight is NVFP4 (experts/shared/lm_head).
std::vector<float> MatmulNvfp4F32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                  int64_t K, const Nvfp4Weight& w) {
  const int64_t N = w.n;
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kF32, {M, N});
  if (d.q.device.type == vt::DeviceType::kCUDA) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), dx.t(), dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), dx.t(), dwb.t());
  }
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// y[M,N] bf16 = x[M,K] bf16 @ dequant(w).T, w fp4-resident [N=out, K=in]. Drops
// in for MatmulBf16 (expert down projection).
std::vector<uint16_t> MatmulNvfp4Bf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                      int64_t K, const Nvfp4Weight& w) {
  const int64_t N = w.n;
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dout(d, DType::kBF16, {M, N});
  if (d.q.device.type == vt::DeviceType::kCUDA) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), dx.t(), dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), dx.t(), dwb.t());
  }
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// --- Device-resident matmul helpers (M2.5 Phase 1) --------------------------
// Same GEMMs as MatmulF32/MatmulBf16/MatmulNvfp4* but device-in / device-out:
// the input activation is already a device tensor and the result STAYS on the
// device (returned as a DBuf) — NO host Download/Synchronize. This is what lets
// the whole decode step run async-on-stream (the prerequisite for graph
// capture). x is [M,K] bf16 (device); the returned DBuf owns the [M,N] output.

// Both helpers route on the weight's orientation flag: nk=true (raw torch
// Linear [N,K], LoadBf16RawNK) -> vt::MatmulBT, the cuBLASLt TN fast path;
// nk=false (loader-transposed [K,N]) -> row-major vt::Matmul, unchanged.

DBuf MatmulF32D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.nk ? w.shape[0] : w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), x, dw);
  else
    vt::Matmul(d.q, dout.t(), x, dw);
  return dout;
}

DBuf MatmulBf16D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.nk ? w.shape[0] : w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
  if (w.nk)
    vt::MatmulBT(d.q, dout.t(), x, dw);
  else
    vt::Matmul(d.q, dout.t(), x, dw);
  return dout;
}

// Device-resident view over an Fp8Weight's raw fp8 [N,K] bytes, uploaded ONCE
// (lazily) and reused across every forward step (mirror ResidentNvfp4). The
// shared_ptr in the (const) weight owns the device buffer for the model lifetime.
Tensor ResidentFp8(Dev d, const Fp8Weight& w) {
  if (!w.d_packed) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.d_packed = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
  }
  return MakeTensor(w.d_packed.get(), DType::kI8, d.q.device, {w.n, w.k});
}

// cuBLASLt FP8 dense GEMM toggle (VT_DENSE_CUBLASLT_FP8, DEFAULT ON when the fp8
// weights are resident). Routes the fp8 dense projections through vt::
// MatmulFp8CublasLt (cuBLASLt e4m3 — the native equivalent of vLLM's measured-
// FASTER nvjet_sm121_qqtst fp8 kernels) instead of vt::MatmulFp8Cutlass (our
// cutlass sm120 fp8 GEMM, measured NEUTRAL vs bf16 at M=64/sm_121a). The
// activation quant + fp8-resident weight are IDENTICAL for both — only the GEMM
// backend differs, so both are the same fp8 W8A8 math (vLLM's scheme).
// VT_DENSE_CUBLASLT_FP8=0 restores the cutlass fp8 GEMM (the previous, validated
// path) for the parent's authoritative A/B.
bool DenseCublasLtFp8Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_DENSE_CUBLASLT_FP8");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// y[M,N] = x[M,K] (bf16/f32 device) @ dequant(w).T via a per-tensor W8A8 fp8
// GEMM: static per-tensor activation quant (vt::QuantFp8Static with the
// checkpoint input_scale) then an fp8 GEMM with the folded alpha
// (= input_scale·weight_scale). By DEFAULT the GEMM is cuBLASLt fp8 (vt::
// MatmulFp8CublasLt — mirrors vLLM's nvjet_qqtst fp8 dense); VT_DENSE_CUBLASLT_
// FP8=0 selects the cutlass sm120 fp8 GEMM (vt::MatmulFp8Cutlass). out dtype f32
// (q/k/v, in_proj_qkv/z sinks) or bf16 (o/out_proj residual sinks). CUDA-only
// (the 35B W8A8 path is CUDA-resident — fp8 fields are populated by DEFAULT on
// the CUDA+cutlass load, VT_DENSE_NATIVE).
DBuf MatmulFp8CutlassD(Dev d, const Tensor& x, const Fp8Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(d.q.device.type == vt::DeviceType::kCUDA,
           "MatmulFp8CutlassD: the fp8 W8A8 path is CUDA-only");
  DBuf a_fp8(d, DType::kI8, {M, K});
  vt::QuantFp8Static(d.q, a_fp8.t(), x, w.input_scale);
  Tensor wdev = ResidentFp8(d, w);
  DBuf dout(d, out_dtype, {M, N});
  if (DenseCublasLtFp8Enabled())
    vt::MatmulFp8CublasLt(d.q, dout.t(), a_fp8.t(), wdev, w.alpha);
  else
    vt::MatmulFp8Cutlass(d.q, dout.t(), a_fp8.t(), wdev, w.alpha);
  return dout;
}

// Pre-quantized fp8 analog of MatmulFp8CutlassD: the activation is ALREADY the
// static-quant fp8 [M,K] (produced ONCE — either by RmsNormQuantFp8 or a shared
// quant — and fed to every projection reading it), so this SKIPS the internal
// QuantFp8Static and runs only the fp8 GEMM. The fp8 counterpart of
// MatmulNvfp4Fp4DirectD; each GEMM still applies its own folded alpha (= shared
// input_scale · this projection's weight_scale), so the result is identical to
// MatmulFp8CutlassD(x) when a_fp8 == QuantFp8Static(x, w.input_scale).
DBuf MatmulFp8CutlassPreQuantD(Dev d, const Tensor& a_fp8, const Fp8Weight& w, DType out_dtype) {
  const int64_t M = a_fp8.shape[0], N = w.n;
  VT_CHECK(d.q.device.type == vt::DeviceType::kCUDA,
           "MatmulFp8CutlassPreQuantD: the fp8 W8A8 path is CUDA-only");
  Tensor wdev = ResidentFp8(d, w);
  DBuf dout(d, out_dtype, {M, N});
  if (DenseCublasLtFp8Enabled())
    vt::MatmulFp8CublasLt(d.q, dout.t(), a_fp8, wdev, w.alpha);
  else
    vt::MatmulFp8Cutlass(d.q, dout.t(), a_fp8, wdev, w.alpha);
  return dout;
}

// FUSE fp8 RMSNorm -> static quant + quantize-once (35B W8A8): fold the input-
// layernorm (residual-add + gemma RMSNorm) and the shared activation's fp8 quant
// into ONE pass (vt::RmsNormQuantFp8, mirror vLLM Inductor
// fused_add_rms_norm_static_fp8_quant, rms_quant_fusion.py:124), feeding the SINGLE
// fp8 activation to every projection that reads it (attn q/k/v; GDN in_proj_qkv/z)
// via MatmulFp8CutlassPreQuantD — removing the standalone QuantFp8Static pass + its
// bf16 round-trip AND the redundant per-projection re-quant of the same [T,H].
// Bit-identical (bf16-intermediate form); only fires when the shared projections
// carry ONE input_scale (guarded at the RunLayer site — the real 35B q/k/v and
// in_proj_qkv/z do). DEFAULT ON: token-exact (op-level byte-identical + 35B greedy
// 16/16) and a clean same-binary win at the gate shape (in1024/out128 conc32 np192:
// +0.85% total & prefill tok/s, every ON run > every OFF run; nsys: the input-fed
// QuantFp8Static instances drop 130->40 per step, folded into RmsNormQuantFp8). The
// 27B (no fp8 weights) never fires it. VT_FUSE_RMSNORM_FP8QUANT=0 opts out for an A/B.
bool FuseRmsNormFp8QuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_RMSNORM_FP8QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// TRUE W4A4 toggle (A/B; default ON — mirrors vLLM, which runs this checkpoint as
// use_a16=False true-W4A4, notes §7.1). Set VT_W4A4_TRUE=0 to fall back to the
// W4A16 6a fast path (bf16 activations) for a throughput A/B. Only affects
// weights that carry the activation-quant globals (27B; alpha>0) — the 35B
// (alpha==0) is untouched regardless.
bool TrueW4A4Enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_W4A4_TRUE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// FUSE silu-mul with the down_proj activation quant (mirror vllm
// ActivationQuantFusionPass / silu_and_mul_nvfp4_quant) — one kernel, no bf16
// intermediate. Default ON (opt-out VT_FUSE_SILU_QUANT=0): MEASURED +2.4% prefill-
// heavy, token-exact (the earlier "-2%" was the stale software quant ladder, not the
// fusion). Only the 27B true-W4A4 dense MLP is affected (down_proj quantizes its act).
bool FuseSiluQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_SILU_QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// FUSE the full-attention preamble (prefill-gap-scan dominant lever): the 4-5
// separate f32 kernels before the attention kernel (AttnGateSplit + q-RMSNorm +
// k-RMSNorm + partial NeoX RoPE, the last recomputing cos/sin transcendentals in
// DOUBLE per element/head/layer) collapse into ONE launch that reads a precomputed
// cos_sin cache — mirror of vLLM's fused_qk_rmsnorm_rope (fla
// fused_qk_norm_rope.py:95-102, zero in-kernel transcendentals). Measured 27B
// site (2026-07-10): unfused preamble+rope 1.36 vs vLLM 0.27 us/tok/layer.
//
// PER-ARCH DEFAULT (2026-07-10). The fusion is NOT bit-identical: the gate
// passthrough is exact, but the RoPE'd q/k differ by ONE f32 ULP because the
// fused `ni*cs - nih*cs` contracts to a different FMA than the unfused
// RmsNorm-store + RopeNeox `x*c - y*sn` (pinned by test_ops_attn_preamble). On
// the 35B (fp8 attn) the deterministic greedy decode is 1-ULP-sensitive and
// DIVERGED within 16 tokens (measured 2026-07-09) => fp8/bf16-attn default
// stays OFF. On the 27B (fp4 attn) the token-exact gates PASS with it ON
// (test_qwen27_paged_engine + the dense-logits deterministic span) => fp4-attn
// default ON. Unset env => default ON iff fp4_attn; VT_FUSE_ATTN_PREAMBLE=1/0
// force-overrides either way (same-binary A/B).
bool FuseAttnPreambleOn(bool fp4_attn) {
  static const char* e = std::getenv("VT_FUSE_ATTN_PREAMBLE");
  if (e != nullptr) return e[0] == '1';
  return fp4_attn;
}

// FA-2 PREFILL (the last measured 27B prefill gap): route the full-attn PREFILL
// segment through the vendored FlashAttention-2 flash_fwd_splitkv kernel (the
// exact kernel vLLM runs on GB10; vllm-project/flash-attention @ 2c839c33) by
// making the fused preamble emit bf16 q/k and the attention output bf16 — the
// natively-bf16 combo the FA-2 dispatch gate (cuda_paged_attn.cu) requires, with
// ZERO cast kernels (the earlier f32-glued wiring measured negative; see
// cuda_flash_attn_fa2.cu header). Measured 27B site (2026-07-10): our WMMA
// prefill attention 1.37 vs vLLM's FA-2 0.25 us/tok/layer (~18 us/tok e2e).
// bf16-q/out is NOT bit-identical to the f32-q WMMA path (FA-2 rounds q to bf16
// and accumulates in its own order) but IS vLLM-faithful (vLLM's whole attn
// path is bf16) — validated by the token-exact greedy gate (2026-07-10: 27B
// engine gate PASS ON and OFF, same tie branch; chunked==one-shot holds).
// MEASURED (2026-07-10, GB10, same binary): kernel 3.68x faster (475.3ms ->
// 129.2ms per np16xin1024 profile; ~1.81 -> 0.49 us/tok/layer), 27B e2e
// conc16/np96 752.8 -> 761.6 (+1.2%), conc32/np192 1045.6 -> 1050.4 (+0.5%),
// TTFT -3.4/-3.6%, putting the 27B at/above fresh graphed-vLLM denominators —
// hence DEFAULT ON when compiled (VLLM_CPP_FLASH_ATTN); VT_FA2_PREFILL=0
// restores the WMMA prefill for a same-binary A/B. Decode segments keep f32
// q/out + the graph-captured decode kernels, byte-identical either way.
// Without VLLM_CPP_FLASH_ATTN the env is ignored and the WMMA path runs.
bool Fa2PrefillOn() {
#ifdef VLLM_CPP_FLASH_ATTN
  const char* e = std::getenv("VT_FA2_PREFILL");
  return e == nullptr || e[0] != '0';
#else
  return false;
#endif
}

// QUANTIZE-ONCE: q/k/v (and gate/up) share their input activation AND their on-disk
// input_global_scale (verified: 27B layer-3 q/k/v all 28.75; gate/up 812/476), so we
// can ScaledFp4Quant the shared activation ONCE and feed each projection's fp4xfp4
// GEMM the same packed/scale — removing the 2-3x redundant per-projection quant of
// [T,H] (mirrors vLLM's fused qkv/gate_up MergedColumnParallelLinear = one quant).
// Bit-identical only when the shared input_global_scale_inv match (guarded at the
// call site). DEFAULT ON (bit-identical + measured +0.3-0.5% on the 27B prefill,
// mirrors vLLM's fused qkv/gate_up MergedColumnParallelLinear); VT_FUSE_QUANT_ONCE=0
// restores per-projection quant for an A/B.
bool FuseQuantOnceEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSE_QUANT_ONCE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// BF16 GEMM OUTPUTS (prefill-lever-scan rank 1-2): the fp4 gate/up (and later q/k/v)
// projections currently output f32 (MatmulNvfp4F32D); vLLM keeps them bf16 (model
// dtype). Emitting bf16 halves the GEMM output write + the downstream glue read
// traffic (MoeSiluMul) on the memory-bound prefill. NOT bit-identical (gate/up are
// bf16-rounded before silu) but MORE faithful to vLLM's bf16 model dtype — validated
// by the token-exact gate. DEFAULT ON (measured 27B +5.4% standard / +12.5% prefill-
// heavy, gate 9/9; matches vLLM bf16 dtype); VT_BF16_GEMM_OUT=0 restores f32 for an A/B.
bool Bf16GemmOutEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_BF16_GEMM_OUT");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

#ifdef VT_CUTLASS_NVFP4
// cutlass sm120a fp4xfp4 GEMM path toggle (DEFAULT ON when compiled with
// VT_CUTLASS_NVFP4 — mirrors how the validated 35B fp8/Marlin defaults were
// flipped on: absence of the env selects cutlass; VT_NVFP4_CUTLASS=0 is the
// opt-out to the emulation-grade path). Routes the 27B true-W4A4 projections
// through vt::MatmulNvfp4Cutlass (the lifted vLLM near-peak kernel) instead of
// our emulation-grade MatmulNvfp4Fp4. Mirrors vLLM, which auto-selects the
// cutlass/flashinfer sm120a fp4×fp4 kernel for CT-W4A4
// (compressed_tensors_w4a4_nvfp4.py; notes §7.1). MEASURED same-binary A/B
// 37.21→124.33 tok/s (3.34×), gap vs vLLM 11.2×→3.19× (parity-ledger row 66).
//
// 27B-ONLY BY CONSTRUCTION: NvfpCutlassEnabled() is reached ONLY from
// MatmulNvfp4Fp4D, itself guarded by `w.IsTrueW4A4()` (alpha>0 — the 27B W4A4
// alone). The 35B is W4A16 (alpha==0, IsTrueW4A4()==false) and never enters this
// path; its dense/MoE NVFP4 run the Marlin W4A16 branch. So this default-flip
// cannot affect the 35B.
//
// THROUGHPUT lever, near-emulation correctness. Swapping the GEMM to real cutlass
// does NOT recover vLLM's native flashinfer stream (198) — the 27B still yields
// tok6=271; tok6 is a razor near-tie tipped by the aggregate non-fp4 forward
// numerics, not the fp4 GEMM (measured 2026-07-05). MEASURED 2026-07-06: cutlass
// is ~0.19% off the emulation-grade MatmulNvfp4Fp4 (test_ops_nvfp4_fp4, NOT
// bit-exact), so on the near-tie-dense 27B greedy gate it reproduces emulation on
// the semantic tokens (0-7) then DETERMINISTICALLY flips a later whitespace
// near-tie at tok8 (271 "\n\n" -> 198 "\n"). Output stays coherent; the token-exact
// correctness gate therefore pins the emulation-grade reference (VT_NVFP4_CUTLASS=0)
// while this default carries the throughput win — see
// tests/parity/test_qwen27_paged_engine.cpp. Only meaningful when the cutlass TU
// was compiled (VT_CUTLASS_NVFP4).
bool NvfpCutlassEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_CUTLASS");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// vLLM v0.25's normal and fused NVFP4 quant producers write activation block
// scales directly in the CUTLASS tensor-core atom layout. Default to that
// topology only on the compiled CUDA CUTLASS W4A4 path; the opt-out preserves
// the exact former linear producer -> standalone SwizzleBlockscale sequence for
// same-binary A/B and rollback.
bool DirectFp4ScaleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_DIRECT_SF");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

bool DirectFp4ScaleEligible(Dev d) {
  return DirectFp4ScaleEnabled() && NvfpCutlassEnabled() &&
         d.q.device.type == vt::DeviceType::kCUDA;
}

// vLLM's CompressedTensorsW4A4Nvfp4 retains alpha as a non-trainable device
// parameter and FlashInfer passes that pointer directly into CUTLASS. Default to
// the same ownership; the opt-out is the exact former per-GEMM host-scalar
// staging path for same-binary component measurement and rollback.
bool DeviceFp4AlphaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_DEVICE_ALPHA");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

#ifdef VT_CUTLASS_NVFP4
Tensor ResidentDeviceAlpha(Dev d, const float* host_alpha,
                           std::shared_ptr<void>& owner,
                           const char* context) {
  VT_CHECK(host_alpha != nullptr && *host_alpha > 0.0F, context);
  if (!owner) {
    Backend* backend = &d.b;
    void* data = d.b.Alloc(sizeof(float));
    std::shared_ptr<void> candidate(
        data, [backend](void* pointer) { backend->Free(pointer); });
    d.b.Copy(d.q, data, host_alpha, sizeof(float));
    owner = std::move(candidate);
  }
  return MakeTensor(owner.get(), DType::kF32, d.q.device, {1});
}

Tensor ResidentNvfp4Alpha(Dev d, const Nvfp4Weight& w) {
  VT_CHECK(w.IsTrueW4A4(), "qwen3_5 NVFP4 device alpha: true-W4A4 required");
  VT_CHECK(w.d_packed && w.d_scale && w.d_scale_sw,
           "qwen3_5 NVFP4 device alpha: incomplete weight resident state");
  return ResidentDeviceAlpha(d, &w.alpha, w.d_alpha,
                             "qwen3_5 NVFP4 device alpha: invalid scalar");
}

Tensor ResidentNvfp4GateUpAlpha(Dev d, const DenseMlpWeights& w,
                                float alpha) {
  VT_CHECK(w.d_gate_up_packed && w.d_gate_up_scale_sw,
           "qwen3_5 dense merged gate_up alpha: incomplete resident state");
  if (!w.d_gate_up_alpha) {
    w.gate_up_alpha = alpha;
  } else {
    VT_CHECK(w.gate_up_alpha == alpha,
             "qwen3_5 dense merged gate_up alpha changed after upload");
  }
  return ResidentDeviceAlpha(
      d, &w.gate_up_alpha, w.d_gate_up_alpha,
      "qwen3_5 dense merged gate_up alpha: invalid scalar");
}

Tensor ResidentNvfp4QkvAlpha(Dev d, const FullAttnLayerWeights& w,
                             float alpha) {
  VT_CHECK(w.d_qkv_packed && w.d_qkv_scale_sw,
           "qwen3_5 packed QKV alpha: incomplete resident state");
  if (!w.d_qkv_alpha) {
    w.qkv_alpha = alpha;
  } else {
    VT_CHECK(w.qkv_alpha == alpha,
             "qwen3_5 packed QKV alpha changed after upload");
  }
  return ResidentDeviceAlpha(d, &w.qkv_alpha, w.d_qkv_alpha,
                             "qwen3_5 packed QKV alpha: invalid scalar");
}

void MatmulNvfp4CutlassModel(Dev d, Tensor& out,
                             const Tensor& a_packed,
                             const Tensor& a_sf_sw,
                             const Tensor& b_packed,
                             const Tensor& b_sf_sw, float alpha_host,
                             const Tensor& alpha_device) {
  if (alpha_device.data != nullptr) {
    vt::MatmulNvfp4Cutlass(d.q, out, a_packed, a_sf_sw, b_packed,
                           b_sf_sw, alpha_device);
  } else {
    vt::MatmulNvfp4Cutlass(d.q, out, a_packed, a_sf_sw, b_packed,
                           b_sf_sw, alpha_host);
  }
}
#endif

// SWIZZLE-ONCE dedup (prefill-gap-scan quant-hw-swizzle lever). The fused qkv/
// gate-up projections share ONE ScaledFp4Quant activation + its LINEAR fp8 block
// scale, but each MatmulNvfp4Fp4DirectD re-runs the identical internal
// SwizzleBlockscale on that SAME shared scale — 3x for a fused qkv, 2x for gate/up
// (the nsys "2,856 SwizzleBlockscaleKernel launches" per short prefill). When ON we
// swizzle the shared activation SF exactly ONCE per fuse-site and pass the already-
// swizzled SF into each projection GEMM (skipping its internal re-swizzle), removing
// the redundant per-projection re-swizzles. BIT-IDENTICAL: SwizzleBlockscale is a
// pure deterministic reorder of the linear SF, so one swizzled buffer reused across
// the projections equals each projection swizzling independently. Default OFF until
// GPU-gated (token-exact vs OFF by construction); VT_SWIZZLE_IN_QUANT=1 enables.
bool SwizzleInQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_SWIZZLE_IN_QUANT");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Dense gate/up topology mirror. The full-tactic fallback must remain the
// immutable W1 arm, so VT_FP4_FULL_TACTICS=0 also disables this W2 model-side
// adaptation. VT_FP4_MERGED_GATE_UP=0 isolates the split W2 arm while retaining
// the same 32 raw tactics.
bool MergedGateUpEnabled() {
  static const bool on = [] {
    const char* full_tactics = std::getenv("VT_FP4_FULL_TACTICS");
    if (full_tactics != nullptr && full_tactics[0] == '0') return false;
    const char* merged = std::getenv("VT_FP4_MERGED_GATE_UP");
    return merged == nullptr || merged[0] != '0';
  }();
  return on;
}

// vLLM's QKVParallelLinear is one physical projection even at TP=1. Keep the
// W1 tactic fallback independent and expose a dedicated same-binary W3-D arm.
bool MergedQkvEnabled() {
  static const bool on = [] {
    const char* full_tactics = std::getenv("VT_FP4_FULL_TACTICS");
    if (full_tactics != nullptr && full_tactics[0] == '0') return false;
    const char* merged = std::getenv("VT_FP4_MERGED_QKV");
    return merged == nullptr || merged[0] != '0';
  }();
  return on;
}

// vLLM's ActivationQuantFusionPass consumes the merged BF16 [M,2I] gate_up
// result directly and emits the down-projection NVFP4 operands in one kernel.
// Keep a dedicated same-binary fallback so this W2 sub-iteration can be timed
// independently from the already-gated merged topology and tactic family.
bool MergedSiluQuantEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_MERGED_SILU_QUANT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

bool MergedGateUpEligible(const DenseMlpWeights& w, Dev d) {
  const Nvfp4Weight& gate = w.gate_proj_fp4;
  const Nvfp4Weight& up = w.up_proj_fp4;
  return MergedGateUpEnabled() && NvfpCutlassEnabled() &&
         Bf16GemmOutEnabled() && d.q.device.type == vt::DeviceType::kCUDA &&
         !gate.Empty() && !up.Empty() && gate.IsTrueW4A4() &&
         up.IsTrueW4A4() && TrueW4A4Enabled() && gate.n == up.n &&
         gate.k == up.k && gate.weight_global_scale_inv > 0.0F &&
         up.weight_global_scale_inv > 0.0F;
}

bool MergedQkvEligible(const FullAttnLayerWeights& w, Dev d,
                       bool packed_consumers) {
  const Nvfp4Weight& q = w.q_proj_fp4;
  const Nvfp4Weight& k = w.k_proj_fp4;
  const Nvfp4Weight& v = w.v_proj_fp4;
  // Packed output views are row-strided. The fused preamble consumes Q/K with
  // their real row strides, while Attention/ReshapeAndCache consume the V view.
  // If the preamble is explicitly disabled, retain the fully contiguous split
  // reference rather than materializing split-copy kernels.
  return packed_consumers && MergedQkvEnabled() &&
         NvfpCutlassEnabled() && Bf16GemmOutEnabled() &&
         d.q.device.type == vt::DeviceType::kCUDA && !q.Empty() && !k.Empty() &&
         !v.Empty() && q.IsTrueW4A4() && k.IsTrueW4A4() &&
         v.IsTrueW4A4() && TrueW4A4Enabled() && q.k == k.k && q.k == v.k &&
         q.weight_global_scale_inv > 0.0F &&
         k.weight_global_scale_inv > 0.0F &&
         v.weight_global_scale_inv > 0.0F;
}

// One CT-W4A4 gate_up projection, matching Qwen2MoeMLP.forward and the fused
// scale processing in CompressedTensorsW4A4Fp4. This is deliberately a BF16
// result: vLLM's model dtype is BF16 and SiluAndMul consumes that one [M,2I]
// buffer. The split diagnostic remains in DenseMlpBlock below.
DBuf MergedGateUpCutlassD(Dev d, const Tensor& x, const DenseMlpWeights& w) {
  const int64_t m = x.shape[0];
  const int64_t k = x.shape[1];
  const int64_t n = w.gate_proj_fp4.n;
  Nvfp4GateUpDev gate_up = ResidentNvfp4GateUp(d, w);

  DBuf a_packed(d, DType::kI8, {m, k / 2});
  const bool direct_scale = DirectFp4ScaleEligible(d);
  DBuf a_scale(d, DType::kI8,
               direct_scale ? CutlassFp4ScaleShape(m, k)
                            : std::vector<int64_t>{m, k / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                     gate_up.globals.input_global_scale_inv,
                     direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                                  : vt::Fp4ScaleLayout::kLinear);
  std::optional<DBuf> composed_scale;
  const Tensor* scale_for_gemm = &a_scale.t();
  if (!direct_scale) {
    composed_scale.emplace(d, DType::kI8, CutlassFp4ScaleShape(m, k));
    vt::SwizzleBlockscale(d.q, composed_scale->t(), a_scale.t());
    scale_for_gemm = &composed_scale->t();
  }

  DBuf out(d, DType::kBF16, {m, 2 * n});
  Tensor alpha_device;
  if (DeviceFp4AlphaEnabled()) {
    alpha_device =
        ResidentNvfp4GateUpAlpha(d, w, gate_up.globals.alpha);
  }
  MatmulNvfp4CutlassModel(d, out.t(), a_packed.t(), *scale_for_gemm,
                          gate_up.packed, gate_up.scale_sw,
                          gate_up.globals.alpha, alpha_device);
  return out;
}

// One CT-W4A4 QKVParallelLinear. The result is the standard row-major
// [M,Q+K+V] tensor; consumers use row-strided, inner-contiguous logical views
// exactly like torch.split on vLLM's packed result.
DBuf MergedQkvCutlassD(Dev d, const Tensor& x,
                       const FullAttnLayerWeights& w) {
  const int64_t m = x.shape[0];
  const int64_t inner_k = x.shape[1];
  Nvfp4QkvDev qkv = ResidentNvfp4Qkv(d, w);
  const int64_t total_n = qkv.qn + qkv.kn + qkv.vn;

  DBuf a_packed(d, DType::kI8, {m, inner_k / 2});
  const bool direct_scale = DirectFp4ScaleEligible(d);
  DBuf a_scale(d, DType::kI8,
               direct_scale ? CutlassFp4ScaleShape(m, inner_k)
                            : std::vector<int64_t>{m, inner_k / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                     qkv.globals.input_global_scale_inv,
                     direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                                  : vt::Fp4ScaleLayout::kLinear);
  std::optional<DBuf> composed_scale;
  const Tensor* scale_for_gemm = &a_scale.t();
  if (!direct_scale) {
    composed_scale.emplace(d, DType::kI8,
                           CutlassFp4ScaleShape(m, inner_k));
    vt::SwizzleBlockscale(d.q, composed_scale->t(), a_scale.t());
    scale_for_gemm = &composed_scale->t();
  }

  DBuf out(d, DType::kBF16, {m, total_n});
  Tensor alpha_device;
  if (DeviceFp4AlphaEnabled()) {
    alpha_device = ResidentNvfp4QkvAlpha(d, w, qkv.globals.alpha);
  }
  MatmulNvfp4CutlassModel(d, out.t(), a_packed.t(), *scale_for_gemm,
                          qkv.packed, qkv.scale_sw, qkv.globals.alpha,
                          alpha_device);
  return out;
}
#endif

// TRUE W4A4 (fp4 activations x fp4 weights) device GEMM — the 27B path (notes §7).
// ScaledFp4Quant(x) -> per-token fp4 activations + fp8 block scales, then the
// fp4xfp4 GEMM with the folded alpha (= vllm cutlass_scaled_fp4_mm_sm120a). CUDA
// only; the CPU fp4 path uses the bf16-dequant fallback in the callers. out_dtype
// f32 or bf16.
// The fp4xfp4 GEMM from PRE-QUANTIZED activations (a_packed [M,K/2] + a_scale
// [M,K/16], the ScaledFp4Quant outputs). Factored out of MatmulNvfp4Fp4D so a
// fused producer (SiluMulFp4Quant / RmsNormResFp4Quant) can feed the GEMM directly
// without the bf16 intermediate. out_dtype f32 or bf16.
// `a_sf_sw_pre` may be produced directly by the v0.25-style quant kernel or by
// the older VT_SWIZZLE_IN_QUANT dedup. In either case it is already in CUTLASS
// layout and bypasses the standalone swizzle. When null, `a_scale` is linear
// and the former composition remains byte-identical.
DBuf MatmulNvfp4Fp4DirectD(Dev d, const Tensor& a_packed, const Tensor& a_scale,
                           const Nvfp4Weight& w, DType out_dtype,
                           [[maybe_unused]] const Tensor* a_sf_sw_pre = nullptr) {
  const int64_t M = a_packed.shape[0], N = w.n;
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  DBuf dout(d, out_dtype, {M, N});
#ifdef VT_CUTLASS_NVFP4
  if (NvfpCutlassEnabled()) {
    // Swizzle the per-token activation scale into the cutlass atom layout (the
    // weight scale is swizzled once, cached). Then the lifted sm120a fp4xfp4 GEMM.
    const int64_t K = a_packed.shape[1] * 2;
    auto round_up = [](int64_t v, int64_t y) { return (v + y - 1) / y * y; };
    const int64_t Mp = round_up(M, 128), Kp = round_up(K / 16, 4);
    Tensor b_sf_sw = ResidentNvfp4ScaleSwizzled(d, w);
    Tensor alpha_device;
    if (DeviceFp4AlphaEnabled()) {
      alpha_device = ResidentNvfp4Alpha(d, w);
    }
    // SWIZZLE-ONCE dedup: the shared activation SF was already swizzled ONCE by the
    // fused caller — reuse it (bit-identical to re-swizzling here) and skip our
    // internal SwizzleBlockscale for this projection.
    if (a_sf_sw_pre != nullptr) {
      MatmulNvfp4CutlassModel(d, dout.t(), a_packed, *a_sf_sw_pre,
                              dw.packed, b_sf_sw, w.alpha, alpha_device);
      return dout;
    }
    DBuf a_sf_sw(d, DType::kI8, {Mp, Kp});  // SwizzleBlockscale zero-fills padding
    vt::SwizzleBlockscale(d.q, a_sf_sw.t(), a_scale);
    MatmulNvfp4CutlassModel(d, dout.t(), a_packed, a_sf_sw.t(),
                            dw.packed, b_sf_sw, w.alpha, alpha_device);
    return dout;
  }
#endif
  vt::MatmulNvfp4Fp4(d.q, dout.t(), a_packed, a_scale, dw.packed, dw.scale, w.alpha);
  return dout;
}

#ifdef VT_CUTLASS_NVFP4
// SWIZZLE-ONCE dedup helper (VT_SWIZZLE_IN_QUANT). Swizzle the shared fused-site
// activation block scale [M, groups] into the cutlass atom layout [round_up(M,128),
// round_up(groups,4)] exactly ONCE; the returned buffer is fed to each projection
// GEMM via MatmulNvfp4Fp4DirectD's a_sf_sw_pre. Shape/bytes are identical to the
// buffer MatmulNvfp4Fp4DirectD would build internally (M=a_packed.shape[0],
// groups=K/16), so reuse is bit-identical to per-projection swizzling.
DBuf SwizzleActScaleOnce(Dev d, const Tensor& a_scale) {
  const int64_t M = a_scale.shape[0], groups = a_scale.shape[1];
  auto round_up = [](int64_t v, int64_t y) { return (v + y - 1) / y * y; };
  DBuf a_sf_sw(d, DType::kI8, {round_up(M, 128), round_up(groups, 4)});
  vt::SwizzleBlockscale(d.q, a_sf_sw.t(), a_scale);
  return a_sf_sw;
}
#endif

DBuf MatmulNvfp4Fp4D(Dev d, const Tensor& x, const Nvfp4Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1];
  DBuf a_packed(d, DType::kI8, {M, K / 2});
#ifdef VT_CUTLASS_NVFP4
  if (DirectFp4ScaleEligible(d)) {
    DBuf a_scale(d, DType::kI8, CutlassFp4ScaleShape(M, K));
    vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x,
                       w.input_global_scale_inv,
                       vt::Fp4ScaleLayout::kCutlassSwizzled);
    return MatmulNvfp4Fp4DirectD(d, a_packed.t(), a_scale.t(), w,
                                 out_dtype, &a_scale.t());
  }
#endif
  DBuf a_scale(d, DType::kI8, {M, K / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x, w.input_global_scale_inv);
  return MatmulNvfp4Fp4DirectD(d, a_packed.t(), a_scale.t(), w, out_dtype);
}

// Defined after the W4A16 Marlin selection helpers below; the split QKV
// diagnostic still uses their normal model-side dispatch.
DBuf MatmulNvfp4F32D(Dev d, const Tensor& x, const Nvfp4Weight& w);
DBuf MatmulNvfp4Bf16D(Dev d, const Tensor& x, const Nvfp4Weight& w);

// Owners plus logical views for full-attention Q/K/V. The packed path owns one
// [T,Q+K+V] allocation and exposes row-strided inner-contiguous views; the
// diagnostic/non-FP4 paths own three ordinary contiguous allocations.
struct FullAttnQkvOutput {
  bool fp4 = false;
  std::optional<DBuf> packed_owner;
  std::optional<DBuf> q_owner;
  std::optional<DBuf> k_owner;
  std::optional<DBuf> v_owner;
  Tensor qgate;
  Tensor key;
  Tensor value;
};

FullAttnQkvOutput ProjectFullAttnQkv(Dev d, const FullAttnLayerWeights& w,
                                     const Tensor& h, int64_t t,
                                     const Tensor* h_fp8,
                                     [[maybe_unused]] bool packed_consumers) {
  FullAttnQkvOutput out;
  out.fp4 = !w.q_proj_fp4.Empty();
  const bool fp8 = !w.q_proj_fp8.Empty();
#ifdef VT_CUTLASS_NVFP4
  if (out.fp4 && MergedQkvEligible(w, d, packed_consumers)) {
    out.packed_owner.emplace(MergedQkvCutlassD(d, h, w));
    Tensor all = out.packed_owner->t();
    const int64_t qn = w.q_proj_fp4.n;
    const int64_t kn = w.k_proj_fp4.n;
    const int64_t vn = w.v_proj_fp4.n;
    VT_CHECK(all.shape[1] == qn + kn + vn,
             "qwen3_5 packed QKV: output shape mismatch");
    out.qgate = all.Slice(1, 0, qn);
    out.key = all.Slice(1, qn, qn + kn);
    Tensor value2 = all.Slice(1, qn + kn, qn + kn + vn);
    out.value = value2;
    return out;
  }
#endif

  // Split reference: quantize the shared activation once when the three input
  // divisors are equal, then retain one independently-scaled GEMM per shard.
  const bool fuse_qkv =
      out.fp4 && FuseQuantOnceEnabled() &&
      d.q.device.type == vt::DeviceType::kCUDA &&
      w.q_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled() &&
      w.q_proj_fp4.input_global_scale_inv ==
          w.k_proj_fp4.input_global_scale_inv &&
      w.q_proj_fp4.input_global_scale_inv ==
          w.v_proj_fp4.input_global_scale_inv;
  std::optional<DBuf> qkv_ap;
  std::optional<DBuf> qkv_as;
#ifdef VT_CUTLASS_NVFP4
  const bool qkv_direct_scale =
      fuse_qkv && DirectFp4ScaleEligible(d);
#else
  const bool qkv_direct_scale = false;
#endif
  if (fuse_qkv) {
    const int64_t hidden = h.shape[1];
    qkv_ap.emplace(d, DType::kI8,
                   std::vector<int64_t>{t, hidden / 2});
    qkv_as.emplace(d, DType::kI8,
                   qkv_direct_scale
                       ? CutlassFp4ScaleShape(t, hidden)
                       : std::vector<int64_t>{t, hidden / 16});
    vt::ScaledFp4Quant(d.q, qkv_ap->t(), qkv_as->t(), h,
                       w.q_proj_fp4.input_global_scale_inv,
                       qkv_direct_scale
                           ? vt::Fp4ScaleLayout::kCutlassSwizzled
                           : vt::Fp4ScaleLayout::kLinear);
  }
  const Tensor* qkv_sf_sw_p = nullptr;
#ifdef VT_CUTLASS_NVFP4
  std::optional<DBuf> qkv_sf_sw;
  if (qkv_direct_scale) {
    qkv_sf_sw_p = &qkv_as->t();
  } else if (fuse_qkv && SwizzleInQuantEnabled() &&
             NvfpCutlassEnabled()) {
    qkv_sf_sw.emplace(SwizzleActScaleOnce(d, qkv_as->t()));
    qkv_sf_sw_p = &qkv_sf_sw->t();
  }
#endif
  const DType q_out_dt =
      (Bf16GemmOutEnabled() && out.fp4) ? DType::kBF16 : DType::kF32;
  const auto project = [&](const Nvfp4Weight& fp4_weight,
                           const Fp8Weight& fp8_weight,
                           const OwnedTensor& plain_weight) -> DBuf {
    if (fuse_qkv) {
      return MatmulNvfp4Fp4DirectD(d, qkv_ap->t(), qkv_as->t(),
                                    fp4_weight, q_out_dt, qkv_sf_sw_p);
    }
    if (fp8) {
      return h_fp8 != nullptr
                 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, fp8_weight,
                                              DType::kF32)
                 : MatmulFp8CutlassD(d, h, fp8_weight, DType::kF32);
    }
    if (out.fp4) {
      return Bf16GemmOutEnabled()
                 ? MatmulNvfp4Bf16D(d, h, fp4_weight)
                 : MatmulNvfp4F32D(d, h, fp4_weight);
    }
    return MatmulF32D(d, h, plain_weight);
  };
  out.q_owner.emplace(
      project(w.q_proj_fp4, w.q_proj_fp8, w.q_proj));
  out.k_owner.emplace(
      project(w.k_proj_fp4, w.k_proj_fp8, w.k_proj));
  out.v_owner.emplace(
      project(w.v_proj_fp4, w.v_proj_fp8, w.v_proj));
  out.qgate = out.q_owner->t();
  out.key = out.k_owner->t();
  out.value = out.v_owner->t();
  return out;
}

#ifdef VT_MARLIN_NVFP4
// --- Dense NVFP4 W4A16 Marlin (M0.8 dense sibling of the MoE grouped GEMM).
// The 35B's dense NVFP4 projections (shared-expert gate/up/down + lm_head) run
// at decode (m=8) through MatmulNvfp4's naive one-thread-per-output kernel, which
// re-dequants the whole weight column PER activation row (8x redundant dequant,
// ~19% of decode GPU per the nsys profile). Route them through the SAME vendored,
// bit-exact Marlin W4A16 GEMM as the MoE experts, run as a SINGLE-expert grouped
// GEMM (E=1, top_k=1): y[M,N] = x[M,K] @ dequant(W[N,K]).T. Repack is load-time
// (BuildMarlinDenseResident); the align inputs are trivial (all tokens -> expert
// 0) and cached per token count. Gated by VT_NVFP4_MARLIN (MarlinMoeEnabled()).
struct MarlinDenseResident {
  void* w = nullptr;  // i32 [K/16, N*2]  Marlin-interleaved weight
  void* s = nullptr;  // fp8 [K/16, N]    processed S0E5M3 scales
  void* g = nullptr;  // f32 [1]          processed global scale
  int64_t n = 0, k = 0;
  bool ready = false;
};

MarlinDenseResident& MarlinDenseResidentFor(const Nvfp4Weight* w) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDenseResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[w];
}

// Repack one dense NVFP4 weight into the resident Marlin layout, then free the
// fp4 device originals (Marlin is the only consumer once the gate is on) so peak
// weight memory stays flat. Same repack primitives as BuildMoeMarlinResident;
// the single weight is its own "expert" (E=1), so combined_scale_factor is taken
// over just this scale buffer.
void BuildMarlinDenseResident(Dev d, const Nvfp4Weight& w, MarlinDenseResident& mr) {
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
  std::vector<const uint8_t*> bufs{reinterpret_cast<const uint8_t*>(w.scale.bytes.data())};
  std::vector<size_t> lens{w.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, static_cast<uint32_t*>(mr.w),
                                     static_cast<const uint8_t*>(dw.packed.data), K, N);
  vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dw.scale.data),
                                      static_cast<uint8_t*>(mr.s), K, N, sf);
  const float g = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.scale2, sf);
  d.b.Copy(d.q, mr.g, &g, sizeof(float));
  d.b.Synchronize(d.q);  // repack done -> safe to free the fp4 originals
  w.d_packed.reset();
  w.d_scale.reset();
  mr.ready = true;
}

// Trivial single-expert moe_align inputs (all M tokens -> expert 0), cached per
// token count M (decode M is constant, so this runs once). Avoids a per-GEMM
// moe_align launch + allocations for the ~120 dense Marlin GEMMs of a step.
struct DenseAlignCache {
  void* sorted = nullptr;  // i32 [max_tok]
  void* expert = nullptr;  // i32 [max_blk] (all 0)
  void* numpad = nullptr;  // i32 [1]
  void* topkw = nullptr;   // f32 [M] (ones; unused since mul_topk_weights=false)
  int block = 0, max_tok = 0, max_blk = 0;
};

DenseAlignCache& DenseAlignFor(Dev d, int M) {
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
  d.b.Memset(d.q, tid, 0, static_cast<size_t>(M) * sizeof(int32_t));  // topk_ids = 0 -> expert 0
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(tid), M, 1, 1, c.block,
                                    static_cast<int32_t*>(c.sorted),
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
void* DenseMarlinWorkspace(Dev d, int* out_sms) {
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
DBuf MatmulNvfp4MarlinD(Dev d, const Tensor& x, const Nvfp4Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  MarlinDenseResident& mr = MarlinDenseResidentFor(&w);
  if (!mr.ready) BuildMarlinDenseResident(d, w, mr);
  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));

  // Marlin's output is bf16 (c_type=kBFloat16); an f32 result is the bf16 output
  // upcast (same value it rounds to — mirror of the cutlass f32-scratch cast).
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

// --- Fused shared-expert gate_up Marlin resident (VT_MOE_FUSED_W13, the dense
// sibling of the MoE fused w13). The shared expert's gate/up dense NVFP4
// projections are TWO separate checkpoint weights here, but in vLLM they are
// ONE merged parameter (MergedColumnParallelLinear gate_up_proj — w1 rows
// first, w3 rows second) repacked WHOLE as a single Marlin operand
// (marlin_utils_fp4.py prepare_fp4_layer_for_marlin). Mirroring that: the pair
// is N-concatenated (gate rows [0,Is), up rows [Is,2Is)) and repacked ONCE
// with size_n=2Is + ONE combined_scale_factor over both shards, so the forward
// runs ONE Marlin GEMM [T,2Is] + SiluAndMul instead of two GEMMs (+2 workspace
// memsets, +2 CastF32) + MoeSiluMul. Requires gate.scale2 == up.scale2 (single
// per-GEMM global scale — same rule as the MoE fused w13); the caller guards.
struct MarlinDensePairResident {
  void* w = nullptr;   // i32 [K/16, (2N)*2]
  void* s = nullptr;   // fp8 [K/16, 2N]
  void* g = nullptr;   // f32 [1]
  int64_t n = 0, k = 0;  // n = per-shard N (Is); operand size_n = 2n
  bool ready = false;
};

MarlinDensePairResident& MarlinDensePairResidentFor(const Nvfp4Weight* gate) {
  static std::mutex mu;
  static std::unordered_map<const Nvfp4Weight*, MarlinDensePairResident> cache;
  std::lock_guard<std::mutex> lk(mu);
  return cache[gate];
}

void BuildMarlinDensePairResident(Dev d, const Nvfp4Weight& gw, const Nvfp4Weight& uw,
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
  // combined_scale_factor over BOTH shards (vLLM computes it over the merged
  // gate_up scale tensor).
  std::vector<const uint8_t*> bufs{reinterpret_cast<const uint8_t*>(gw.scale.bytes.data()),
                                   reinterpret_cast<const uint8_t*>(uw.scale.bytes.data())};
  std::vector<size_t> lens{gw.scale.bytes.size(), uw.scale.bytes.size()};
  const float sf = vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
  Nvfp4Dev dg = ResidentNvfp4(d, gw);
  Nvfp4Dev du = ResidentNvfp4(d, uw);
  // Flat row-stack concat (packed [N,K/2] u8 / scales [N,K/16] fp8 are
  // row-major over N; gate rows first — the vLLM merged shard order).
  auto* tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
  auto* tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sc_b));
  d.b.Copy(d.q, tmp_w, dg.packed.data, pk_b);
  d.b.Copy(d.q, tmp_w + pk_b, du.packed.data, pk_b);
  d.b.Copy(d.q, tmp_s, dg.scale.data, sc_b);
  d.b.Copy(d.q, tmp_s + sc_b, du.scale.data, sc_b);
  vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, static_cast<uint32_t*>(mr.w),
                                     tmp_w, K, 2 * N);
  vt::cuda::MarlinProcessExpertScales(stream, tmp_s, static_cast<uint8_t*>(mr.s), K, 2 * N, sf);
  // Single global scale for both shards (gate's; equality guarded by caller —
  // the vLLM merged parameter has exactly one weight_global_scale).
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

// True when the shared-expert gate/up pair takes the fused Marlin gate_up path
// (one GEMM [T,2Is] + SiluAndMul). Must be checked IDENTICALLY at load
// (PrepareMarlinResident) and forward so exactly one resident layout is built.
bool SharedGateUpFusedEligible(const Nvfp4Weight& gw, const Nvfp4Weight& uw) {
  return MoeFusedW13Enabled() && !gw.Empty() && !uw.Empty() && !gw.IsTrueW4A4() &&
         !uw.IsTrueW4A4() && gw.n == uw.n && gw.k == uw.k && gw.scale2 == uw.scale2;
}

// silu(x@gate.T) * (x@up.T) -> bf16 [M,Is] via ONE fused Marlin gate_up GEMM.
DBuf SharedGateUpFusedMarlinD(Dev d, const Tensor& x, const Nvfp4Weight& gw,
                              const Nvfp4Weight& uw) {
  const int64_t M = x.shape[0], K = x.shape[1], N = gw.n;
  MarlinDensePairResident& mr = MarlinDensePairResidentFor(&gw);
  if (!mr.ready) BuildMarlinDensePairResident(d, gw, uw, mr);
  DenseAlignCache& ac = DenseAlignFor(d, static_cast<int>(M));
  int sms = 0;
  void* ws = DenseMarlinWorkspace(d, &sms);
  d.b.Memset(d.q, ws, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));

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

DBuf MatmulNvfp4F32D(Dev d, const Tensor& x, const Nvfp4Weight& w) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  if (d.q.device.type == vt::DeviceType::kCUDA && w.IsTrueW4A4() && TrueW4A4Enabled())
    return MatmulNvfp4Fp4D(d, x, w, DType::kF32);
#ifdef VT_MARLIN_NVFP4
  // NVFP4 W4A16 dense (shared expert / lm_head): the load-time-repacked Marlin
  // GEMM replaces the naive redundant-dequant kernel when VT_NVFP4_MARLIN=1.
  // Marlin requires a bf16 activation (the 35B dense NVFP4 sinks all are).
  if (d.q.device.type == vt::DeviceType::kCUDA && !w.IsTrueW4A4() && MarlinMoeEnabled() &&
      x.dtype == DType::kBF16)
    return MatmulNvfp4MarlinD(d, x, w, DType::kF32);
#endif
  DBuf dout(d, DType::kF32, {M, N});
  if (d.q.device.type == vt::DeviceType::kCUDA) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), x, dwb.t());
  }
  return dout;
}

// Same as MatmulNvfp4F32D but bf16 output (the down/o/out_proj sinks that feed
// the residual add). CUDA: fp4-resident vt::MatmulNvfp4 (bf16 out). CPU: the
// DequantNvfp4ToBLayout fallback (no CPU MatmulNvfp4 kernel).
DBuf MatmulNvfp4Bf16D(Dev d, const Tensor& x, const Nvfp4Weight& w) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  if (d.q.device.type == vt::DeviceType::kCUDA && w.IsTrueW4A4() && TrueW4A4Enabled())
    return MatmulNvfp4Fp4D(d, x, w, DType::kBF16);
#ifdef VT_MARLIN_NVFP4
  if (d.q.device.type == vt::DeviceType::kCUDA && !w.IsTrueW4A4() && MarlinMoeEnabled() &&
      x.dtype == DType::kBF16)
    return MatmulNvfp4MarlinD(d, x, w, DType::kBF16);
#endif
  DBuf dout(d, DType::kBF16, {M, N});
  if (d.q.device.type == vt::DeviceType::kCUDA) {
    Nvfp4Dev dw = ResidentNvfp4(d, w);
    vt::MatmulNvfp4(d.q, dout.t(), x, dw.packed, dw.scale, w.scale2);
  } else {
    std::vector<uint16_t> wb = DequantNvfp4ToBLayout(w);
    DBuf dwb(d, DType::kBF16, {K, N}, wb.data());
    vt::Matmul(d.q, dout.t(), x, dwb.t());
  }
  return dout;
}

// --- Paged-path helpers (M1.8 Task 3) --------------------------------------

// Non-owning strided view over dim0 [row_offset, row_offset+rows) of a
// contiguous device tensor (all trailing dims kept). Used to hand GdnPrefill/
// GdnDecode the decode vs prefill token sub-slices without copying.
Tensor SubView(const Tensor& src, int64_t row_offset, int64_t rows) {
  Tensor t = src;
  int64_t row_elems = 1;
  for (int i = 1; i < src.rank; ++i) row_elems *= src.shape[i];
  t.data = static_cast<char*>(src.data) +
           static_cast<size_t>(row_offset * row_elems) * vt::SizeOf(src.dtype);
  t.shape[0] = rows;
  return t;
}

// One unbind(1) slice of a (num_blocks, 2, block_size, H, D) FlashAttention KV
// buffer: which=0 -> K, which=1 -> V. The result is the rank-4 STRIDED view
// backends read (block stride 2*bs*H*D — the "2" is NOT collapsed), matching the
// M1.6 Task-2 contract (cpu_cache.cpp / cpu_paged_attn.cpp).
Tensor KvSlice(const PagedKvCache& kv, Device dev, int which) {
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

// Gather the `idx`-indexed rows of a persistent state cache `src`
// [Nblk, row_elems] into contiguous `dst` (device-to-device via Backend::Copy;
// on CPU a memcpy, on CUDA a same-device copy). Row-major per row.
void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// Inverse of GatherRows: write the contiguous per-request rows back to their
// persistent cache slots. `dst` is a non-owning view whose buffer is mutable
// through .data even when the view is const.
void ScatterRows(Dev d, const Tensor& dst, const void* src,
                 const std::vector<int32_t>& idx, int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(dst.dtype);
  auto* dp = static_cast<char*>(dst.data);
  const auto* sp = static_cast<const char*>(src);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + static_cast<size_t>(idx[s]) * rb, sp + s * rb, rb);
}

// Gather idx-indexed rows of a persistent GDN state cache into a fresh f32
// working buffer. On CUDA the cache is bf16 (vLLM default mamba_cache_dtype
// auto → model dtype): gather the raw bf16 rows then upcast — fla reads the
// bf16 cache into f32 registers (fused_recurrent.py:102). On CPU the cache is
// f32: gather directly. The f32 buffer is what the f32 GdnPrefill/CausalConv1dFwd
// consume; the bf16 round-trip lives only at the cache boundary.
DBuf GatherStateF32(Dev d, const Tensor& cache, const std::vector<int32_t>& idx,
                    int64_t row_elems, const std::vector<int64_t>& shape) {
  DBuf f32buf(d, DType::kF32, shape);
  if (cache.dtype == DType::kBF16) {
    DBuf bf(d, DType::kBF16, shape);
    GatherRows(d, bf.ptr(), cache, idx, row_elems);
    vt::CastF32(d.q, f32buf.t(), bf.t());
  } else {
    GatherRows(d, f32buf.ptr(), cache, idx, row_elems);
  }
  return f32buf;
}

// Inverse of GatherStateF32: downcast the f32 working buffer to the cache dtype
// (bf16 on CUDA / f32 on CPU) and scatter it back to the idx-indexed slots.
void ScatterStateF32(Dev d, const Tensor& cache, DBuf& f32buf,
                     const std::vector<int32_t>& idx, int64_t row_elems,
                     const std::vector<int64_t>& shape) {
  if (cache.dtype == DType::kBF16) {
    DBuf bf(d, DType::kBF16, shape);
    vt::CastBf16(d.q, bf.t(), f32buf.t());
    ScatterRows(d, cache, bf.ptr(), idx, row_elems);
  } else {
    ScatterRows(d, cache, f32buf.ptr(), idx, row_elems);
  }
}

// W1 indexed state-I/O dispatch. CUDA + device-resident W0 storage defaults to
// the fused indexed gather/scatter operators. Either diagnostic opt-out restores
// the exact row-copy + cast baseline on the same binary. CPU keeps that baseline
// as its reference implementation.
bool IndexedGdnStateIoEnabled(Device device) {
  const char* indexed = std::getenv("VT_GDN_INDEXED_STATE_IO");
  // CPU keeps the row-copy reference by default. An explicit =1 is a test hook
  // that drives the whole model integration through the CPU reference kernels.
  if (device.type != vt::DeviceType::kCUDA)
    return indexed != nullptr && indexed[0] == '1';
  const char* cache = std::getenv("VT_DEVICE_KV_CACHE");
  if (cache != nullptr && cache[0] == '0') return false;
  return indexed == nullptr || indexed[0] != '0';
}

// Prefill launch-gap fusion (perf/glue-fuse): fold the GDN post-conv glue chain
// GdnConvSplit + L2Norm(q) + L2Norm(k) + GdnGBeta (4 launches) into ONE
// vt::GdnPostConv launch, and the gated-RMSNorm + CastBf16 pair (2 launches)
// into a single RmsNormGated writing bf16 directly (layernorm_guard.py:57
// `out.to(dtype)`). Bit-for-bit vs the unfused chain. VT_GLUE_FUSE=0 restores
// the per-op path for A/B measurement (default ON).
bool GlueFuseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_GLUE_FUSE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// Coupled GDN bf16 activations (the measured #1 prefill lever). Default ON. When
// on, the GDN chunk-scan matmul-INPUT activations (q/k/v out of the post-conv
// prep, hence the derived u/w/v_new/hstate scratch — those follow the input
// dtype in cuda_gdn.cu's LaunchChunkedPrefill) are bf16, so the WMMA chunk trio
// (WU Gram / DeltaH / ChunkO) runs on native bf16 tensor-core fragments (2× vs
// TF32) AND moves half the activation/scratch bytes. The recurrent ssm_state,
// the g log-decay + its cumsum, beta, and the WMMA f32 accumulators stay f32 —
// mirroring vLLM FLA's dtype split (chunk_delta_h.py final_state=torch.float32
// while v_new/w are k.dtype=bf16; wy_fast.py u/w = k.new_empty bf16; chunk_o.py
// b_o/b_A f32 accumulators, o stored bf16). The op-level bf16 WMMA path is the
// same one test_ops_gdn's bf16 chunked-vs-sequential case exercises (3e-2 tol).
// A/B: VT_GDN_BF16=0 restores the f32/TF32 activations in the same binary.
DType GdnActDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_GDN_BF16");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}

// GDN INPUT-side bf16 (VT_GDN_IN_BF16, DEFAULT ON — gated). Mirrors vLLM FLA
// carrying the *pre-chunk* activations in bf16, not just the chunk fragments:
// when ON, the GDN in_proj mixed_qkv GEMM emits bf16 (MatmulBf16D instead of
// MatmulF32D), the causal conv1d then runs bf16 in/out (bf16 conv weight,
// f32-accumulated internally, f32 conv_state unchanged), and the post-conv
// split/l2norm reads bf16 conv — halving the traffic of the two big [T,conv_dim]
// activation buffers (mixed write+conv read, conv write+post-conv read) that
// stayed f32 while the chunk trio was already bf16 (GdnActDType). The
// l2norm/softplus math is f32-accumulated regardless (Load() upcasts); g/beta,
// ssm_state, and the a/b GEMMs stay f32 (FLA's split). 27B-only by construction:
// gated on the bf16-weight in_proj branch (the 27B's in_proj_qkv is a plain bf16
// weight); the 35B's fp8-cutlass in_proj branch keeps its f32 output untouched.
// MEASURED (27B, GB10, same-binary A/B): conv kernel -31.5%, post-conv -17.6%
// (chunk trio FLAT — already bf16); e2e +0.68% (conc16, non-overlapping) / +0.83%
// (conc32), TTFT -1.5%; token-exact (27B greedy paged-engine + 35B 16/16). The
// GDN-vs-vLLM gap only 2.07×→~1.9× (the ~1.9× residual is the chunk's codegen
// gap, not dtype). A/B: VT_GDN_IN_BF16=0 restores the byte-identical f32 path.
DType GdnInDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_GDN_IN_BF16");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}

// GDN recurrence-OUTPUT + z-gate in bf16 (27B default ON; 35B keeps its former
// f32 default; VT_GDN_OUT_BF16=0/1 overrides both for diagnostics).
// vLLM keeps core_attn_out and the z gate bf16 (the gated-RMSNorm consumes them):
// FLA chunk_o.py stores o bf16, and Qwen3NextGatedRMSNorm reads bf16 core/gate,
// upcasting to f32 only for the variance reduction (layernorm_guard.py). Our
// `dcore` (recurrence out) + `z` were f32 (a more-precise deviation that doubled
// the [T,Hv,Dv] core traffic in/out of GdnDecode/GdnPrefill AND the gated-norm
// read). When ON, `dcore` is bf16 (GdnDecode/GdnPrefill already dispatch a Tout=bf16
// path so they store bf16 directly), `z` is bf16 (MatmulBf16D), and the gated-norm
// weight is loaded native bf16 (RmsNormGated requires gate/weight dtype == x dtype)
// — the norm's variance/normalize math stays f32-accumulated regardless, so only
// the core/z I/O dtype changes. ssm_state, g (+cumsum), and beta stay f32 (FLA's
// split). Distinct from the earlier VT_BF16_GDN (in_proj/conv/z-gate, neutral):
// this lever is the f32 `dcore` recurrence output that attempt left untouched.
// This is correctness-significant for the 27B: with the repaired full NVFP4
// tactic stack, f32 core/z takes the alternate whitespace near-tie branch while
// bf16 reproduces native vLLM 16/16. Keep every unmeasured 35B arm, including
// GGUF, on its prior f32 default; the explicit env override remains available
// for its later independently gated campaign.
DType GdnOutDType(bool dense_model) {
  static const int override = [] {
    const char* e = std::getenv("VT_GDN_OUT_BF16");
    if (e == nullptr) return -1;
    return e[0] == '0' ? 0 : 1;
  }();
  const bool bf16 = override >= 0 ? override != 0 : dense_model;
  return bf16 ? DType::kBF16 : DType::kF32;
}

// bf16 residual stream (default ON). vLLM runs the 35B in bf16
// (model_config.dtype=bfloat16): qwen3_next.py keeps `residual` as the bf16 hidden
// dtype across all 48 layers, and Qwen3NextRMSNorm==GemmaRMSNorm's fused_add_rms_norm
// adds x into the residual and stores it bf16 while accumulating the variance in f32
// (csrc/layernorm_kernels.cu). OUR residual was f32 — a more-precise accepted
// deviation that doubled the residual memory traffic (read+write per RmsNorm, 2×
// per layer). Making `res` bf16 mirrors vLLM exactly (the RmsNorm kernel keeps its
// f32 variance/normalize accumulation regardless — only the residual load/store
// dtype changes) and halves that traffic. A/B: VT_BF16_RESIDUAL=0 restores the f32
// residual in the same binary.
DType ResidualDType() {
  static const bool bf16 = [] {
    const char* e = std::getenv("VT_BF16_RESIDUAL");
    return e == nullptr || e[0] != '0';
  }();
  return bf16 ? DType::kBF16 : DType::kF32;
}

// --- GDN (linear_attention) block. gdn-semantics.md §1 (layout), §6 (g/beta),
// §7 (recurrence); qwen_gdn_linear_attn.py forward. Device-resident (M2.5
// Phase 1): h [T,H] bf16 (device) -> DBuf [T,H] bf16 (device); no host round-
// trips (the g/beta prep + conv split are device ops, not host loops).
DBuf GdnBlock(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
              const Tensor& h, int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Kw = cfg.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // Input projections (mixed_qkv | z | b | a); kept separate at TP=1 (§6).
  // W8A8 cutlass fp8 (35B) when populated, else bf16 (default / GGUF). qkv/z read
  // the shared pre-quantized fp8 activation (h_fp8, quantize-once) when supplied;
  // a/b stay bf16 GEMMs on h (so h_fp8's producer also emits bf16 h for them).
  // mixed_qkv: bf16 output under VT_GDN_IN_BF16 (27B bf16-weight branch); the
  // fp8-cutlass branch (35B) keeps f32. See GdnInDType().
  const DType indt = GdnInDType();
  DBuf mixed = !w.in_proj_qkv_fp8.Empty()
                   ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_qkv_fp8, DType::kF32)
                            : MatmulFp8CutlassD(d, h, w.in_proj_qkv_fp8, DType::kF32))
               : indt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_qkv)
                                      : MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  const DType outdt = GdnOutDType(cfg.num_experts == 0);
  DBuf z = !w.in_proj_z_fp8.Empty()
               ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_z_fp8, outdt)
                        : MatmulFp8CutlassD(d, h, w.in_proj_z_fp8, outdt))
           : outdt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_z)
                                   : MatmulF32D(d, h, w.in_proj_z);  // [T,value_dim]
  DBuf braw = MatmulF32D(d, h, w.in_proj_b);      // [T,Hv]
  DBuf araw = MatmulF32D(d, h, w.in_proj_a);      // [T,Hv]

  // Causal conv1d over the token stream (silu activation), fresh zero state. conv
  // in/out dtype follows the in_proj output (bf16 under VT_GDN_IN_BF16); f32
  // conv_state + f32-accumulated math unchanged.
  const DType convdt = mixed.t().dtype;
  Tensor dcw = convdt == DType::kBF16 ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
                                      : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dstate(d, DType::kF32, {1, conv_dim, Kw - 1});
  dstate.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  DBuf dconv(d, convdt, {T, conv_dim});
  vt::CausalConv1dFwd(d.q, dconv.t(), mixed.t(), dcw, nullptr, dstate.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});

  // Post-conv prep (gdn-semantics.md §1 layout, §4 l2norm, §6 g/beta): split the
  // conv output into q|k|v, l2-normalize q/k over Dk, derive g/beta. Fused into a
  // single vt::GdnPostConv launch (perf/glue-fuse; mirror fla
  // fused_gdn_prefill_post_conv), or the four per-op launches when disabled.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  // Coupled bf16 (VT_GDN_BF16, default ON): the matmul-input activations q/k/v
  // feed the WMMA chunk trio as native bf16 (halved traffic + bf16 fragments);
  // g/beta/state stay f32 (FLA's split). VT_GDN_BF16=0 keeps f32/TF32.
  const DType actdt = GdnActDType();
  DBuf vf(d, actdt, {T, Hv, Dv});
  DBuf g(d, DType::kF32, {T, Hv});
  DBuf beta(d, DType::kF32, {T, Hv});
  DBuf dql2(d, actdt, {T, Hk, Dk});
  DBuf dkl2(d, actdt, {T, Hk, Dk});
  if (GlueFuseEnabled()) {
    vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), g.t(), beta.t(), dconv.t(), araw.t(),
                    braw.t(), a_log_dev, dt_bias_dev, vt::L2NormArgs{1e-6F});
  } else {
    DBuf qf(d, actdt, {T, Hk, Dk});
    DBuf kf(d, actdt, {T, Hk, Dk});
    Tensor q2 = Reshape(qf.t(), {T, key_dim});
    Tensor k2 = Reshape(kf.t(), {T, key_dim});
    Tensor v2 = Reshape(vf.t(), {T, value_dim});
    vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());
    vt::GdnGBeta(d.q, g.t(), beta.t(), araw.t(), braw.t(), a_log_dev, dt_bias_dev);
    vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
    vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
  }
  // scale = Dk^-0.5, applied to q only inside the gated-delta-rule recurrence.
  DBuf dssm(d, DType::kF32, {1, Hv, Dv, Dk});
  dssm.Zero(d);
  DBuf dcore(d, outdt, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  vt::GdnPrefill(d.q, dcore.t(), dql2.t(), dkl2.t(), vf.t(), g.t(), beta.t(),
                 dssm.t(), dqsl.t(), vt::GdnArgs{scale});

  // Gated RMSNorm over Dv with the z gate (gdn-semantics.md §5), viewing the
  // core output and z as [T*Hv, Dv]; cast to bf16, flatten heads, out-project.
  Tensor dnw = outdt == DType::kBF16 ? ResidentWeight(d, w.norm_weight, {Dv})
                                     : ResidentWeightF32(d, w.norm_weight, {Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = Reshape(z.t(), {T * Hv, Dv});
  // Gated RMSNorm writes bf16 directly (perf/glue-fuse: fold the CastBf16 into
  // the op store, mirror layernorm_guard.py:57 `out.to(dtype)`); VT_GLUE_FUSE=0
  // keeps the f32 RmsNormGated + separate CastBf16 pair.
  DBuf gated_bf16(d, DType::kBF16, {T, value_dim});
  if (GlueFuseEnabled()) {
    Tensor gated2 = Reshape(gated_bf16.t(), {T * Hv, Dv});
    vt::RmsNormGated(d.q, gated2, core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
  } else {
    DBuf dgated(d, DType::kF32, {T * Hv, Dv});
    vt::RmsNormGated(d.q, dgated.t(), core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
    vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  }
  // W8A8 cutlass fp8 (35B) when populated, else fp4-resident W4A4 (27B, notes
  // §3.6), else bf16 (default / GGUF).
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_bf16.t(), w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : MatmulBf16D(d, gated_bf16.t(), w.out_proj);  // [T,H]
}

// PERSISTENT per-step input device buffers (decode host-tax #2): the flattened
// positions + the full-attn metadata (slot_mapping/block_table/seq_lens/
// query_start_loc) + all GDN non-spec/prefill state metadata, uploaded ONCE
// per step and read by EVERY layer — mirrors vLLM's persistent input buffers in
// gpu_model_runner.py (self.input_batch.{positions,slot_mapping,block_table,
// seq_lens,query_start_loc} device tensors, refreshed once per step). Collapses
// the previous per-full-attn-layer (×10) and per-GDN-layer (×30) H2D re-uploads —
// each a BLOCKING pageable cudaMemcpyAsync that serialized the decode stream
// (nsys: ~110 blocking copies/step ≈ 5.1s host-stall in an 8s decode window) — to
// ONE upload per input, per step. The buffers live for the whole layer loop (held
// by the caller). Bit-exact (same bytes, uploaded once instead of per layer).
// Graph-safe: on the num_reqs==1 decode-graph the single upload is captured from
// the persistent host metadata address (same as the per-layer copies were) and
// replayed.
struct StepDevInputs {
  DBuf positions;        // i32 [T]
  DBuf slot_mapping;     // i64 [T]
  DBuf block_table;      // i32 [num_reqs, cols]
  DBuf seq_lens;         // i32 [num_reqs]
  DBuf query_start_loc;  // i32 [num_reqs+1]
  DBuf gdn_state_idx;    // i32 [num_reqs] full non-spec state slots
  bool has_gdn_idx = false;
  DBuf gdn_non_spec_qsl;       // i32 [num_reqs+1]
  DBuf gdn_has_initial;        // i8 [num_reqs], upstream bool mask
  DBuf gdn_prefill_state_idx;  // i32 [num_prefills]
  DBuf gdn_prefill_qsl;        // i32 [num_prefills+1]
  DBuf gdn_prefill_has_initial;  // i8 [num_prefills]
  bool has_gdn_prefill_meta = false;
  bool indexed_gdn_state_io = false;
  // f32 [T, rotary_dim] cos|sin cache for the fused full-attn preamble, built ONCE
  // per step (VT_FUSE_ATTN_PREAMBLE) and reused by every full-attn layer; a 1-elem
  // stub when the toggle is off (has_attn_cos_sin=false).
  DBuf attn_cos_sin;
  bool has_attn_cos_sin = false;
};

StepDevInputs BuildStepDevInputs(Dev d, const std::vector<int32_t>& positions,
                                 const CommonAttentionMetadata& am,
                                 const GDNAttentionMetadata& gm) {
  const int64_t T = static_cast<int64_t>(positions.size());
  const bool indexed_state_io = IndexedGdnStateIoEnabled(d.q.device);
  StepDevInputs s{
      DBuf(d, DType::kI32, {T}, positions.data()),
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {am.num_reqs, am.block_table_num_cols},
           am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {am.num_reqs}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {am.num_reqs + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kI32, {1}),  // non-spec index stub
      false,
      DBuf(d, DType::kI32, {1}),  // non-spec qsl stub
      DBuf(d, DType::kI8, {1}),   // full has-initial stub
      DBuf(d, DType::kI32, {1}),  // prefill index stub
      DBuf(d, DType::kI32, {1}),  // prefill qsl stub
      DBuf(d, DType::kI8, {1}),   // prefill has-initial stub
      false,
      indexed_state_io,
      DBuf(d, DType::kF32, {1}),  // attn cos|sin stub (filled by MaybeBuildAttnCosSin)
      false,
  };
  // Full non-spec state indices are shared by decode and mixed-prefill paths.
  // Decode consumes their leading num_decodes rows; indexed W1 gather/scatter
  // consumes the whole vector. One upload replaces every per-layer row copy.
  if (gm.non_spec_state_indices_tensor.has_value() &&
      (indexed_state_io || gm.num_decodes > 0)) {
    const int64_t index_count =
        indexed_state_io
            ? static_cast<int64_t>(gm.non_spec_state_indices_tensor->size())
            : static_cast<int64_t>(gm.num_decodes);
    s.gdn_state_idx = DBuf(d, DType::kI32,
                           {index_count},
                           gm.non_spec_state_indices_tensor->data());
    s.has_gdn_idx = true;
  }
  if (indexed_state_io && gm.num_prefills > 0 &&
      gm.non_spec_query_start_loc.has_value() &&
      gm.has_initial_state.has_value() &&
      gm.prefill_state_indices.has_value() &&
      gm.prefill_query_start_loc.has_value() &&
      gm.prefill_has_initial_state.has_value()) {
    s.gdn_non_spec_qsl = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.non_spec_query_start_loc->size())},
        gm.non_spec_query_start_loc->data());
    s.gdn_has_initial = DBuf(
        d, DType::kI8,
        {static_cast<int64_t>(gm.has_initial_state->size())},
        gm.has_initial_state->data());
    s.gdn_prefill_state_idx = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.prefill_state_indices->size())},
        gm.prefill_state_indices->data());
    s.gdn_prefill_qsl = DBuf(
        d, DType::kI32,
        {static_cast<int64_t>(gm.prefill_query_start_loc->size())},
        gm.prefill_query_start_loc->data());
    s.gdn_prefill_has_initial = DBuf(
        d, DType::kI8,
        {static_cast<int64_t>(gm.prefill_has_initial_state->size())},
        gm.prefill_has_initial_state->data());
    s.has_gdn_prefill_meta = true;
  }
  return s;
}

// Build the per-step fused-preamble cos|sin cache into `sdi` (once per step,
// reused by every full-attn layer's fused preamble) when VT_FUSE_ATTN_PREAMBLE is
// on. No-op otherwise, so the default forward path is byte-identical. Uses the
// PERSISTENT sdi.positions device buffer (same source the RopeNeox path reads) so
// the fill is a single device kernel — eager and graph-replay identical.
void MaybeBuildAttnCosSin(Dev d, StepDevInputs& sdi, const HfConfig& cfg, int64_t T,
                          bool fp4_attn = false) {
  if (!FuseAttnPreambleOn(fp4_attn)) return;
  const int rot = static_cast<int>(cfg.rotary_dim);
  if (rot <= 0) return;
  sdi.attn_cos_sin = DBuf(d, DType::kF32, {T, rot});
  vt::RopeCosSinCache(d.q, sdi.attn_cos_sin.t(), sdi.positions.t(),
                      vt::RopeArgs{static_cast<float>(cfg.rope_theta), rot});
  sdi.has_attn_cos_sin = true;
}

// --- Batched PAGED GDN block (M1.8 Task 3). Same conv1d + l2norm + q/k/v/g/beta
// prep + gated-norm + out_proj as GdnBlock, but driven by the batched
// GDNAttentionMetadata segmentation over the PERSISTENT ssm_state/conv_state:
// leading num_decode_tokens are decode (vt::GdnDecode + causal_conv1d_update),
// the rest prefill (vt::GdnPrefill + causal_conv1d_fn). Mirrors
// qwen_gdn_linear_attn.py::_forward_core @ e24d1b24 (conv split L1360-1388;
// recurrence split L1480-1559; the ssm gather+ZERO L1513-1514, scatter L1532).
// h [T*H] bf16 -> [T*H] bf16.
DBuf GdnBlockPaged(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
                   const Tensor& h, const StepDevInputs& sdi,
                   const GDNAttentionMetadata& meta,
                   const GdnStateCache& state, int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hk = cfg.linear_num_key_heads;
  const int64_t Hv = cfg.linear_num_value_heads;
  const int64_t Dk = cfg.linear_key_head_dim;
  const int64_t Dv = cfg.linear_value_head_dim;
  const int64_t Kw = cfg.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk;
  const int64_t value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  const int64_t nd = meta.num_decodes;
  const int64_t np = meta.num_prefills;
  const int64_t nd_tok = meta.num_decode_tokens;
  const int64_t np_tok = meta.num_prefill_tokens;
  VT_CHECK(meta.num_actual_tokens == T, "gdn paged: num_actual_tokens != T");
  VT_CHECK(nd_tok + np_tok == T, "gdn paged: decode+prefill tokens != T");

  // Input projections (mixed_qkv | z | b | a); kept separate at TP=1 (§6).
  // W8A8 cutlass fp8 (35B) when populated, else bf16 (default / GGUF). qkv/z read
  // the shared pre-quantized fp8 activation (h_fp8, quantize-once) when supplied;
  // a/b stay bf16 GEMMs on h (so h_fp8's producer also emits bf16 h for them).
  // mixed_qkv: bf16 output under VT_GDN_IN_BF16 (27B bf16-weight branch, halves
  // the conv-input traffic); the fp8-cutlass branch (35B) keeps f32. See
  // GdnInDType().
  const DType indt = GdnInDType();
  DBuf mixed = !w.in_proj_qkv_fp8.Empty()
                   ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_qkv_fp8, DType::kF32)
                            : MatmulFp8CutlassD(d, h, w.in_proj_qkv_fp8, DType::kF32))
               : indt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_qkv)
                                      : MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  // z gate follows the recurrence-output dtype (VT_GDN_OUT_BF16): bf16 when the
  // core is bf16 (the gated-RMSNorm requires gate.dtype == core.dtype), else the
  // byte-identical f32 path.
  const DType outdt = GdnOutDType(cfg.num_experts == 0);
  DBuf z = !w.in_proj_z_fp8.Empty()
               ? (h_fp8 ? MatmulFp8CutlassPreQuantD(d, *h_fp8, w.in_proj_z_fp8, outdt)
                        : MatmulFp8CutlassD(d, h, w.in_proj_z_fp8, outdt))
           : outdt == DType::kBF16 ? MatmulBf16D(d, h, w.in_proj_z)
                                   : MatmulF32D(d, h, w.in_proj_z);  // [T,value_dim]
  DBuf braw = MatmulF32D(d, h, w.in_proj_b);      // [T,Hv]
  DBuf araw = MatmulF32D(d, h, w.in_proj_a);      // [T,Hv]

  // Causal conv1d over the token stream, PERSISTENT conv_state (gathered by the
  // per-request state indices, updated in place, scattered back). conv in/out
  // dtype follows the in_proj output (bf16 under VT_GDN_IN_BF16 → bf16 weight +
  // bf16 dconv halve the conv read/write); the f32 conv_state and the
  // f32-accumulated conv math are unchanged. The post-conv split reads dconv's
  // dtype (GdnPostConv/GdnConvSplit are templated on it).
  const DType convdt = mixed.t().dtype;
  Tensor dcw = convdt == DType::kBF16 ? ResidentWeight(d, w.conv1d_weight, {conv_dim, Kw})
                                      : ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dconv(d, convdt, {T, conv_dim});
  const int64_t conv_row_elems = conv_dim * (Kw - 1);
  const bool indexed_state_io = sdi.indexed_gdn_state_io;
  if (np > 0) {
    // Any prefill: conv over the WHOLE non-spec stream (decodes lead, each with
    // has_initial_state=1). qwen_gdn_linear_attn.py:1360-1375.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    const int64_t nreq = static_cast<int64_t>(sidx.size());
    // Gather the persistent conv_state rows into an f32 working buffer (bf16
    // cache on CUDA → upcast; f32 cache on CPU → direct), run the f32
    // CausalConv1dFwd, then downcast + scatter back to the cache.
    const std::vector<int64_t> cs_shape = {nreq, conv_dim, Kw - 1};
    if (indexed_state_io) {
      VT_CHECK(sdi.has_gdn_idx && sdi.has_gdn_prefill_meta,
               "indexed GDN conv requires persistent non-spec metadata");
      DBuf dcs(d, DType::kF32, cs_shape);
      vt::GdnStateGather(d.q, dcs.t(), state.conv_state,
                         sdi.gdn_state_idx.t());
      vt::CausalConv1dFwd(d.q, dconv.t(), mixed.t(), dcw, nullptr,
                          dcs.t(), sdi.gdn_non_spec_qsl.t(),
                          sdi.gdn_has_initial.t(),
                          vt::CausalConv1dArgs{true});
      Tensor conv_cache = state.conv_state;
      vt::GdnStateScatter(d.q, conv_cache, dcs.t(),
                          sdi.gdn_state_idx.t());
    } else {
      const auto& qsl_full = *meta.non_spec_query_start_loc;
      const auto& his_u8 = *meta.has_initial_state;
      DBuf dcs =
          GatherStateF32(d, state.conv_state, sidx, conv_row_elems, cs_shape);
      std::vector<int32_t> his(his_u8.begin(), his_u8.end());
      DBuf dqsl(d, DType::kI32, {nreq + 1}, qsl_full.data());
      DBuf dhis(d, DType::kI32, {nreq}, his.data());
      vt::CausalConv1dFwd(d.q, dconv.t(), mixed.t(), dcw, nullptr,
                          dcs.t(), dqsl.t(), dhis.t(),
                          vt::CausalConv1dArgs{true});
      ScatterStateF32(d, state.conv_state, dcs, sidx, conv_row_elems,
                      cs_shape);
    }
  } else {
    // Pure decode: single-token conv step per sequence, IN PLACE on the persistent
    // conv_state at each sequence's slot (mirrors mamba causal_conv1d_update
    // conv_state_indices, qwen_gdn_linear_attn.py:1376-1388). Passing the state
    // indices to the op eliminates the per-request gather+scatter — the two
    // host<->device (unified-cache) copies per sequence per layer that dominate
    // the decode memcpy tax (they scale with concurrency, flattening throughput).
    // Upload the decode state-slot indices straight from the PERSISTENT metadata
    // vector (its leading nd entries are the decode slots; decodes lead the batch)
    // — NOT a stack-local copy, so the decode-CUDA-graph replay (num_reqs==1) can
    // re-read this H2D copy from a fixed host address across replays.
    // State slot indices from the PERSISTENT per-step buffer (uploaded once,
    // shared by conv-update + the ssm recurrence across all GDN layers).
    Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
    Tensor conv_cache = state.conv_state;  // mutable view over the shared buffer
    vt::CausalConv1dUpdate(d.q, dconv.t(), mixed.t(), dcw, nullptr, conv_cache,
                           vt::CausalConv1dArgs{true}, &gidx);
  }

  // Post-conv prep (§1 layout, §4 l2norm, §6 g/beta): split q|k|v, l2-normalize
  // q/k over Dk, derive g/beta. Fused into one vt::GdnPostConv launch
  // (perf/glue-fuse; mirror fla fused_gdn_prefill_post_conv), or four per-op
  // launches when disabled. g/beta uniform over all tokens; recurrence segments.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  // Coupled bf16 (VT_GDN_BF16, default ON): matmul-input activations q/k/v are
  // bf16 (native WMMA fragments + halved traffic); g/beta/state f32 (FLA split).
  const DType actdt = GdnActDType();
  DBuf vf(d, actdt, {T, Hv, Dv});
  DBuf dg(d, DType::kF32, {T, Hv});
  DBuf dbeta(d, DType::kF32, {T, Hv});
  DBuf dql2(d, actdt, {T, Hk, Dk});
  DBuf dkl2(d, actdt, {T, Hk, Dk});
  if (GlueFuseEnabled()) {
    vt::GdnPostConv(d.q, dql2.t(), dkl2.t(), vf.t(), dg.t(), dbeta.t(), dconv.t(), araw.t(),
                    braw.t(), a_log_dev, dt_bias_dev, vt::L2NormArgs{1e-6F});
  } else {
    DBuf qf(d, actdt, {T, Hk, Dk});
    DBuf kf(d, actdt, {T, Hk, Dk});
    Tensor q2 = Reshape(qf.t(), {T, key_dim});
    Tensor k2 = Reshape(kf.t(), {T, key_dim});
    Tensor v2 = Reshape(vf.t(), {T, value_dim});
    vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());
    vt::GdnGBeta(d.q, dg.t(), dbeta.t(), araw.t(), braw.t(), a_log_dev, dt_bias_dev);
    vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
    vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
  }
  // scale = Dk^-0.5 (q only, inside the recurrence). dcore (recurrence output /
  // core_attn_out) is bf16 under VT_GDN_OUT_BF16 — GdnDecode/GdnPrefill store the
  // Tout=bf16 path directly (mirror FLA chunk_o.py o bf16); else the f32 arm.
  DBuf dcore(d, outdt, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  const int64_t ssm_row_elems = Hv * Dv * Dk;

  // Recurrence — decode segment first (leading nd_tok tokens), then prefill.
  if (nd > 0) {
    // Decode recurrence IN PLACE on the persistent ssm_state at each sequence's
    // slot (mirrors fla fused_recurrent ssm_state_indices) — no per-request
    // gather+scatter (the other two host<->device copies per sequence per layer).
    // Persistent metadata source (see the conv branch) for graph-replay safety.
    // The persistent W1 buffer covers every non-spec request. A turnover step
    // may have a leading decode subset followed by prefills, so both decode
    // consumers must narrow the view to exactly `nd` state slots.
    Tensor gidx = SubView(sdi.gdn_state_idx.t(), 0, nd);
    Tensor ssm_cache = state.ssm_state;  // mutable view over the shared buffer
    Tensor q_dec = SubView(dql2.t(), 0, nd_tok);
    Tensor k_dec = SubView(dkl2.t(), 0, nd_tok);
    Tensor v_dec = SubView(vf.t(), 0, nd_tok);
    Tensor g_dec = SubView(dg.t(), 0, nd_tok);
    Tensor b_dec = SubView(dbeta.t(), 0, nd_tok);
    Tensor o_dec = SubView(dcore.t(), 0, nd_tok);
    vt::GdnDecode(d.q, o_dec, q_dec, k_dec, v_dec, g_dec, b_dec, ssm_cache,
                  vt::GdnArgs{scale}, &gidx);
  }
  if (np > 0) {
    const auto& pidx = *meta.prefill_state_indices;
    const auto& p_qsl = *meta.prefill_query_start_loc;
    // Gather the persistent ssm_state rows into an f32 working buffer (bf16 cache
    // on CUDA → upcast; f32 cache on CPU → direct), run the f32 chunked GdnPrefill
    // (fla reads the initial_state in f32, writes the final_state), then downcast +
    // scatter back to the cache.
    const std::vector<int64_t> ss_shape = {np, Hv, Dv, Dk};
    DBuf dss(d, DType::kF32, ss_shape);
    if (indexed_state_io) {
      VT_CHECK(sdi.has_gdn_prefill_meta,
               "indexed GDN prefill requires persistent prefill metadata");
      // Fuses indexing, BF16->F32, and the fresh-request zeroing obligation.
      vt::GdnStateGather(d.q, dss.t(), state.ssm_state,
                         sdi.gdn_prefill_state_idx.t(),
                         &sdi.gdn_prefill_has_initial.t());
    } else {
      dss = GatherStateF32(d, state.ssm_state, pidx, ssm_row_elems,
                           ss_shape);
      const auto& p_his = *meta.prefill_has_initial_state;
      const size_t rb = static_cast<size_t>(ssm_row_elems) * sizeof(float);
      for (size_t s = 0; s < p_his.size(); ++s)
        if (p_his[s] == 0)
          d.b.Memset(d.q, static_cast<char*>(dss.ptr()) + s * rb, 0, rb);
    }
    Tensor q_pre = SubView(dql2.t(), nd_tok, np_tok);
    Tensor k_pre = SubView(dkl2.t(), nd_tok, np_tok);
    Tensor v_pre = SubView(vf.t(), nd_tok, np_tok);
    Tensor g_pre = SubView(dg.t(), nd_tok, np_tok);
    Tensor b_pre = SubView(dbeta.t(), nd_tok, np_tok);
    Tensor o_pre = SubView(dcore.t(), nd_tok, np_tok);
    // Hand the CUDA chunked-prefill path the HOST query_start_loc (p_qsl, already
    // materialized on the host by the GDN attention metadata build) so it skips
    // the per-layer D2H copy + cudaStreamSynchronize that forced host↔GPU
    // lockstep every GDN prefill layer (~67% GPU-idle in prefill). p_qsl outlives
    // this call. Device-resident metadata, mirroring the decode StepDevInputs fix.
    vt::GdnArgs gdn_args{scale};
    gdn_args.query_start_loc_host = p_qsl.data();
    if (indexed_state_io) {
      vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre, dss.t(),
                     sdi.gdn_prefill_qsl.t(), gdn_args);
      Tensor ssm_cache = state.ssm_state;
      vt::GdnStateScatter(d.q, ssm_cache, dss.t(),
                          sdi.gdn_prefill_state_idx.t());
    } else {
      DBuf dpqsl(d, DType::kI32, {np + 1}, p_qsl.data());
      vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre, dss.t(),
                     dpqsl.t(), gdn_args);
      ScatterStateF32(d, state.ssm_state, dss, pidx, ssm_row_elems,
                      ss_shape);
    }
  }

  // Gated RMSNorm over Dv with the z gate, cast bf16, flatten heads, out-project.
  // Weight follows core/z dtype (RmsNormGated requires w.dtype == x.dtype): native
  // bf16 under VT_GDN_OUT_BF16 (the norm still accumulates variance in f32), else
  // the f32 upcast. Mirrors the q_norm/k_norm resident-weight dtype gate.
  Tensor dnw = outdt == DType::kBF16 ? ResidentWeight(d, w.norm_weight, {Dv})
                                     : ResidentWeightF32(d, w.norm_weight, {Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = Reshape(z.t(), {T * Hv, Dv});
  // Gated RMSNorm writes bf16 directly (perf/glue-fuse: fold the CastBf16 into
  // the op store, mirror layernorm_guard.py:57 `out.to(dtype)`); VT_GLUE_FUSE=0
  // keeps the f32 RmsNormGated + separate CastBf16 pair.
  DBuf gated_bf16(d, DType::kBF16, {T, value_dim});
  if (GlueFuseEnabled()) {
    Tensor gated2 = Reshape(gated_bf16.t(), {T * Hv, Dv});
    vt::RmsNormGated(d.q, gated2, core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
  } else {
    DBuf dgated(d, DType::kF32, {T * Hv, Dv});
    vt::RmsNormGated(d.q, dgated.t(), core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
    vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  }
  // W8A8 cutlass fp8 (35B) when populated, else fp4-resident W4A4 (27B, notes
  // §3.6), else bf16 (default / GGUF).
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_bf16.t(), w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : MatmulBf16D(d, gated_bf16.t(), w.out_proj);  // [T,H]
}

// --- Dense full_attention block. qwen36-forward-notes.md §5; pinned
// Qwen3NextAttention. h [T*H] bf16 -> [T*H] bf16.
DBuf FullAttnBlock(Dev d, const FullAttnLayerWeights& w, const HfConfig& cfg,
                   const Tensor& h, const std::vector<int32_t>& positions,
                   int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  const bool fp4_attn = !w.q_proj_fp4.Empty();
  const bool packed_consumers =
      FuseAttnPreambleOn(fp4_attn) && rot > 0 &&
      d.q.device.type == vt::DeviceType::kCUDA;
  FullAttnQkvOutput qkv =
      ProjectFullAttnQkv(d, w, h, T, h_fp8, packed_consumers);
  const bool fp4 = qkv.fp4;
  Tensor qgate = qkv.qgate;  // [T,2*Hq*Dh], possibly row-strided packed view
  Tensor kf = qkv.key;       // [T,Hkv*Dh], possibly row-strided packed view
  Tensor vf = qkv.value;     // [T,Hkv*Dh], possibly row-strided packed view

  // Split q|gate + per-head gemma-RMSNorm(q,k) + partial NeoX RoPE + gate
  // passthrough, producing q[T,Hq,Dh]/k[T,Hkv,Dh] (normed+RoPE'd) and the raw
  // gate[T,Hq,Dh]. VT_FUSE_ATTN_PREAMBLE collapses the four ops into ONE launch
  // reading a precomputed cos_sin cache; it emits f32 q/k/gate — byte-identical
  // intermediates to the four-op path (value-exact). OFF keeps the exact original
  // AttnGateSplit + RmsNorm(q) + RmsNorm(k) + RopeNeox sequence.
  DBuf dq3(d, DType::kF32, {T, Hq, Dh});
  DBuf dk3(d, DType::kF32, {T, Hkv, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  if (FuseAttnPreambleOn(fp4) && rot > 0 && d.q.device.type == vt::DeviceType::kCUDA) {
    DBuf dpos(d, DType::kI32, {T}, positions.data());
    DBuf cos_sin(d, DType::kF32, {T, rot});
    vt::RopeCosSinCache(d.q, cos_sin.t(), dpos.t(), vt::RopeArgs{base, rot});
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
    vt::AttnQkNormRopeGate(d.q, dq3.t(), dk3.t(), gatef.t(), qgate,
                           kf, dqw, dkw, cos_sin.t(),
                           vt::RmsNormArgs{eps, true}, vt::RopeArgs{base, rot});
  } else {
    DBuf qf(d, DType::kF32, {T, Hq, Dh});
    vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate);
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dqn2d = Reshape(dq3.t(), {T * Hq, Dh});
    vt::RmsNorm(d.q, dqn2d, Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
    // k-norm weight dtype must equal kf's (RmsNorm requires w.dtype == x.dtype). When
    // kf is bf16 (VT_BF16_GEMM_OUT on the fp4 path) use the raw bf16 on-disk k_norm;
    // otherwise (fp8/35B or toggle off) keep the f32 upcast. bf16 kf · bf16 dkw -> f32.
    Tensor dkw = (kf.dtype == DType::kBF16) ? ResidentWeight(d, w.k_norm, {Dh})
                                           : ResidentWeightF32(d, w.k_norm, {Dh});
    Tensor dkn2d = Reshape(dk3.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, dkn2d, Reshape(kf, {T * Hkv, Dh}), dkw,
                vt::RmsNormArgs{eps, true});
    DBuf dpos(d, DType::kI32, {T}, positions.data());
    vt::RopeNeox(d.q, dq3.t(), dk3.t(), dpos.t(), vt::RopeArgs{base, rot});
  }
  Tensor qn3 = dq3.t();
  Tensor kn3 = dk3.t();

  // Causal GQA scaled-dot-product attention, scale = Dh^-0.5.
  Tensor v3 = vf;
  v3.rank = 3;
  v3.shape[0] = T;
  v3.shape[1] = Hkv;
  v3.shape[2] = Dh;
  v3.stride[1] = Dh;
  v3.stride[2] = 1;
  // vt::Attention requires q/k/v the same float dtype; qn3/kn3 are f32 after
  // norm+rope, so upcast a bf16 V (VT_BF16_GEMM_OUT fp4 path) back to f32. This is
  // the reference (non-paged) path — not perf-critical, so the small cast is fine.
  std::optional<DBuf> v3f32;
  if (v3.dtype == DType::kBF16) {
    v3f32.emplace(d, DType::kF32, std::vector<int64_t>{T, Hkv, Dh});
    vt::CastF32(d.q, v3f32->t(), v3);
    v3 = v3f32->t();
  }
  DBuf dattn(d, DType::kF32, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  vt::Attention(d.q, dattn.t(), qn3, kn3, v3, vt::AttentionArgs{scale, true});

  // Sigmoid output gate on the raw gate split, then o-project (§5) — device op.
  DBuf gated(d, DType::kBF16, {T, Hq * Dh});
  vt::SigmoidGateBf16(d.q, gated.t(), Reshape(dattn.t(), {T, Hq * Dh}),
                      Reshape(gatef.t(), {T, Hq * Dh}));
  return !w.o_proj_fp8.Empty() ? MatmulFp8CutlassD(d, gated.t(), w.o_proj_fp8, DType::kBF16)
         : fp4 ? MatmulNvfp4Bf16D(d, gated.t(), w.o_proj_fp4)
               : MatmulBf16D(d, gated.t(), w.o_proj);  // [T,H]
}

// --- Batched PAGED full_attention block (M1.8 Task 3). Identical q/k/v prep to
// FullAttnBlock (gemma qk-RMSNorm + partial NeoX RoPE + GQA + output gate), but
// replaces vt::Attention with vt::ReshapeAndCache (write new K/V into the paged
// NHD cache at slot_mapping) + vt::PagedAttention (read causal K/V from the
// cache via block_table/seq_lens/query_start_loc). Mirrors
// qwen3_next.py::Qwen3NextAttention.forward @ e24d1b24 (self.attn(q,k,v) is the
// reshape_and_cache + paged read). h [T*H] bf16 -> [T*H] bf16.
DBuf FullAttnBlockPaged(Dev d, const FullAttnLayerWeights& w, const HfConfig& cfg,
                        const Tensor& h, const StepDevInputs& sdi,
                        const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                        int64_t T, const Tensor* h_fp8 = nullptr) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  VT_CHECK(kv.dtype == DType::kBF16 || kv.dtype == DType::kF32,
           "full-attn paged: KV cache must be bf16 or f32");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "full-attn paged: KV cache head dims mismatch config");

  const bool fp4_attn = !w.q_proj_fp4.Empty();
  const bool packed_consumers =
      FuseAttnPreambleOn(fp4_attn) && sdi.has_attn_cos_sin &&
      d.q.device.type == vt::DeviceType::kCUDA;
  FullAttnQkvOutput qkv_out =
      ProjectFullAttnQkv(d, w, h, T, h_fp8, packed_consumers);
  const bool fp4 = qkv_out.fp4;
  Tensor qgate = qkv_out.qgate;
  Tensor kf = qkv_out.key;
  Tensor vf = qkv_out.value;

  // Split q|gate + per-head gemma-RMSNorm(q,k) + partial NeoX RoPE + gate
  // passthrough. VT_FUSE_ATTN_PREAMBLE (default OFF) collapses the four ops into
  // ONE launch reading the per-step cos_sin cache (sdi.attn_cos_sin, built once by
  // MaybeBuildAttnCosSin and reused by all 11 full-attn layers) — mirror of vLLM's
  // fused_qk_rmsnorm_rope (fla fused_qk_norm_rope.py:95-102, zero in-kernel
  // transcendentals). Emits f32 q/k/gate: byte-identical intermediates to the
  // four-op path (the query stays f32 for PagedAttention), so ON is token-exact.
  // positions/slot_mapping/... are the PERSISTENT per-step device buffers; sdi.*
  // Tensors are const views over the shared DBufs (no per-layer H2D re-upload).
  //
  // FA-2 PREFILL (VT_FA2_PREFILL, see Fa2PrefillOn): on an eligible PREFILL step
  // the preamble instead emits bf16 q/k (gate stays f32) and the attention
  // output is bf16 — the natively-bf16 combo the FA-2 dispatch gate requires,
  // with zero cast kernels. bf16 k feeds the bf16 KV-cache write directly
  // (skipping the CastBf16 below — bit-identical, both are the RN round of the
  // same f32 value); bf16 attention out feeds SigmoidGateBf16 (exact upcast).
  // The eligibility MUST mirror cuda_paged_attn.cu's fa2 gate (prefill segment:
  // T > num_reqs; head_dim 256; bf16 KV; CUDA) so the bf16 query is consumed by
  // FA-2. Decode steps keep f32 q/out + the graph-captured decode kernels,
  // byte-identical to today.
  const bool fa2_prefill = Fa2PrefillOn() && FuseAttnPreambleOn(fp4) && sdi.has_attn_cos_sin &&
                           d.q.device.type == vt::DeviceType::kCUDA &&
                           kv.dtype == DType::kBF16 && Dh == 256 && T > meta.num_reqs;
  const DType attn_dt = fa2_prefill ? DType::kBF16 : DType::kF32;
  DBuf dq3(d, attn_dt, {T, Hq, Dh});
  DBuf dk3(d, attn_dt, {T, Hkv, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  if (FuseAttnPreambleOn(fp4) && sdi.has_attn_cos_sin) {
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
    vt::AttnQkNormRopeGate(d.q, dq3.t(), dk3.t(), gatef.t(), qgate,
                           kf, dqw, dkw, sdi.attn_cos_sin.t(),
                           vt::RmsNormArgs{eps, true}, vt::RopeArgs{base, rot});
  } else {
    DBuf qf(d, DType::kF32, {T, Hq, Dh});
    vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate);
    Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
    Tensor dqn2d = Reshape(dq3.t(), {T * Hq, Dh});
    vt::RmsNorm(d.q, dqn2d, Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
    // k-norm weight dtype must equal kf's (RmsNorm requires w.dtype == x.dtype). When
    // kf is bf16 (VT_BF16_GEMM_OUT on the fp4 path) use the raw bf16 on-disk k_norm;
    // otherwise (fp8/35B or toggle off) keep the f32 upcast. bf16 kf · bf16 dkw -> f32.
    Tensor dkw = (kf.dtype == DType::kBF16) ? ResidentWeight(d, w.k_norm, {Dh})
                                           : ResidentWeightF32(d, w.k_norm, {Dh});
    Tensor dkn2d = Reshape(dk3.t(), {T * Hkv, Dh});
    vt::RmsNorm(d.q, dkn2d, Reshape(kf, {T * Hkv, Dh}), dkw,
                vt::RmsNormArgs{eps, true});
    vt::RopeNeox(d.q, dq3.t(), dk3.t(), sdi.positions.t(), vt::RopeArgs{base, rot});
  }
  Tensor qn3 = dq3.t();
  Tensor kn3 = dk3.t();

  Tensor v3 = vf;
  v3.rank = 3;
  v3.shape[0] = T;
  v3.shape[1] = Hkv;
  v3.shape[2] = Dh;
  v3.stride[1] = Dh;
  v3.stride[2] = 1;

  // KV-cache dtype: the production runner allocates a bf16 cache (mirrors vLLM's
  // bf16 flash_attn KV store, halves KV memory); the paged==dense unit anchors
  // allocate an f32 cache to stay bit-exact. The "auto" ReshapeAndCache copy
  // requires cache dtype == k/v dtype, so down-cast the rope'd f32 K and the f32
  // V to bf16 only when the cache is bf16. The query stays f32 either way
  // (Phase 1: f32 query · <cache-dtype> cache, f32-accumulate softmax — the
  // attention kernel converts bf16 cache reads to f32).
  Tensor kw = kn3;
  Tensor vw = v3;
  DBuf kbf(d, DType::kBF16, {T, Hkv, Dh});
  DBuf vbf(d, DType::kBF16, {T, Hkv, Dh});
  if (kv.dtype == DType::kBF16) {
    // K may already be bf16 (the FA-2 prefill preamble emits bf16 k directly —
    // the RN round of the same f32 value this CastBf16 would produce); only
    // down-cast when the preamble/fallback produced f32 K.
    if (kn3.dtype == DType::kBF16) {
      kw = kn3;
    } else {
      vt::CastBf16(d.q, kbf.t(), kn3);
      kw = kbf.t();
    }
    // V may already be bf16 (VT_BF16_GEMM_OUT: the fp4 v_proj GEMM emits bf16
    // directly, removing the cutlass CastBf16ToF32 + this CastBf16 round-trip);
    // only down-cast when the GEMM produced f32 V.
    if (v3.dtype == DType::kBF16) {
      vw = v3;
    } else {
      vt::CastBf16(d.q, vbf.t(), v3);
      vw = vbf.t();
    }
  }

  // Write the new K/V into the paged cache, then read K/V from the cache.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  Tensor dslot = sdi.slot_mapping.t();
  Tensor dblk = sdi.block_table.t();
  Tensor dsl = sdi.seq_lens.t();
  Tensor dqsl = sdi.query_start_loc.t();
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, dslot);

  // bf16 attention out on the FA-2 prefill path (FA-2 writes bf16; the sigmoid
  // gate upcast is exact) — f32 everywhere else, byte-identical to today.
  DBuf dattn(d, attn_dt, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  // Hand the prefill flash/WMMA launchers the HOST query_start_loc (already
  // materialized per step by the attention metadata build) so they size the
  // query-tile grid from these host values + a device meta-kernel, skipping the
  // per-layer D2H copy + cudaStreamSynchronize that drained the pipeline every
  // full-attention prefill layer (~10-12 syncs/step; prefill only 43.7%
  // GPU-busy). meta.query_start_loc outlives this call. Device-resident metadata,
  // mirroring the GDN GdnArgs::query_start_loc_host / decode StepDevInputs fix.
  // max_seq_len is the FA-2 launcher's host grid bound (same pattern).
  vt::PagedAttentionArgs pa_args{scale, meta.causal};
  pa_args.query_start_loc_host = meta.query_start_loc.data();
  pa_args.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, dattn.t(), qn3, k_cache, v_cache, dblk, dsl, dqsl, pa_args);

  // Sigmoid output gate then o-project (§5) — device op.
  DBuf gated(d, DType::kBF16, {T, Hq * Dh});
  vt::SigmoidGateBf16(d.q, gated.t(), Reshape(dattn.t(), {T, Hq * Dh}),
                      Reshape(gatef.t(), {T, Hq * Dh}));
  return !w.o_proj_fp8.Empty() ? MatmulFp8CutlassD(d, gated.t(), w.o_proj_fp8, DType::kBF16)
         : fp4 ? MatmulNvfp4Bf16D(d, gated.t(), w.o_proj_fp4)
               : MatmulBf16D(d, gated.t(), w.o_proj);  // [T,H]
}

// Per-expert silu-mul MLP over the gathered token rows `x` [n, H] bf16 ->
// [n, H] bf16 (moe-semantics.md §4; gate/up/down kept separate at TP=1).
std::vector<uint16_t> ExpertMlp(Dev d, const OwnedTensor& gate,
                                const OwnedTensor& up, const OwnedTensor& down,
                                const std::vector<uint16_t>& x, int64_t n,
                                int64_t H, int64_t I) {
  std::vector<float> hg = MatmulF32(d, x, n, H, gate);  // [n,I]
  std::vector<float> hu = MatmulF32(d, x, n, H, up);    // [n,I]
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(Silu(hg[i]) * hu[i]);
  return MatmulBf16(d, act, n, I, down);  // [n,H]
}

// fp4-resident per-expert silu-mul MLP (M2.2b): identical to ExpertMlp but the
// gate/up/down NVFP4 weights are read on-device via vt::MatmulNvfp4.
std::vector<uint16_t> ExpertMlpNvfp4(Dev d, const Nvfp4Weight& gate,
                                     const Nvfp4Weight& up, const Nvfp4Weight& down,
                                     const std::vector<uint16_t>& x, int64_t n,
                                     int64_t H, int64_t I) {
  std::vector<float> hg = MatmulNvfp4F32(d, x, n, H, gate);  // [n,I]
  std::vector<float> hu = MatmulNvfp4F32(d, x, n, H, up);    // [n,I]
  std::vector<uint16_t> act(static_cast<size_t>(n) * I);
  for (size_t i = 0; i < act.size(); ++i)
    act[i] = vt::F32ToBF16(Silu(hg[i]) * hu[i]);
  return MatmulNvfp4Bf16(d, act, n, I, down);  // [n,H]
}

// Shared-expert MLP (moe-semantics.md §5): silu-mul MLP then sigmoid(x@Wseg) *
// out. Device-resident (M2.5 Phase 1): h [T,H] bf16 (device) -> DBuf [T,H] bf16
// (device); the silu-mul + sigmoid-gate are device ops (not host loops), so the
// shared expert adds no host round-trip to the captured decode step.
// MoE glue-kernel fusion (VT_MOE_GLUE_FUSE, default ON): fold the shared-expert
// sigmoid gate into the weighted MoeCombine (one launch, no shared [T,H]
// round-trip) instead of a separate SharedExpertGate kernel. Bit-identical to
// the two-kernel path (see MoeCombineGate). VT_MOE_GLUE_FUSE=0 restores the
// unfused SharedExpertGate + MoeCombine sequence for A/B.
bool MoeGlueFuseEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_GLUE_FUSE");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

// Shared-expert pre-gate parts: sd [T,H] f32 (down projection) and gl [T,1] f32
// (gate logit), before the sigmoid gate + bf16 round. The unfused SharedExpert
// applies SharedExpertGate to these; the fused MoeBlock passes them straight to
// MoeCombineGate.
struct SharedExpertParts {
  DBuf sd;
  DBuf gl;
};

SharedExpertParts SharedExpertUngated(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                                      const Tensor& h, int64_t T, bool fp4) {
  const int64_t Is = cfg.shared_expert_intermediate_size;
#ifdef VT_MARLIN_NVFP4
  // Fused gate_up (VT_MOE_FUSED_W13, dense sibling of the MoE fused w13): ONE
  // Marlin GEMM [T,2Is] + SiluAndMul — vLLM's merged gate_up_proj layout. The
  // silu input values match the unfused path bit-for-bit given equal GEMM
  // outputs: unfused MatmulNvfp4F32D is the SAME Marlin bf16 GEMM upcast to f32
  // (value-preserving), and MoeSiluMul/SiluAndMul share the f32 silu math.
  if (fp4 && d.q.device.type == vt::DeviceType::kCUDA && MarlinMoeEnabled() &&
      h.dtype == DType::kBF16 &&
      SharedGateUpFusedEligible(w.shared_gate_proj_fp4, w.shared_up_proj_fp4)) {
    DBuf sact = SharedGateUpFusedMarlinD(d, h, w.shared_gate_proj_fp4, w.shared_up_proj_fp4);
    DBuf sd = MatmulNvfp4F32D(d, sact.t(), w.shared_down_proj_fp4);  // [T,H] f32
    DBuf gl = MatmulF32D(d, h, w.shared_gate);                       // [T,1] f32
    return {std::move(sd), std::move(gl)};
  }
#endif
  DBuf sg = fp4 ? MatmulNvfp4F32D(d, h, w.shared_gate_proj_fp4)
                : MatmulF32D(d, h, w.shared_gate_proj);  // [T,Is]
  DBuf su = fp4 ? MatmulNvfp4F32D(d, h, w.shared_up_proj_fp4)
                : MatmulF32D(d, h, w.shared_up_proj);    // [T,Is]
  DBuf sact(d, DType::kBF16, {T, Is});
  vt::MoeSiluMul(d.q, sact.t(), sg.t(), su.t());  // silu(sg) * su -> bf16
  DBuf sd = fp4 ? MatmulNvfp4F32D(d, sact.t(), w.shared_down_proj_fp4)
                : MatmulF32D(d, sact.t(), w.shared_down_proj);  // [T,H] f32
  DBuf gl = MatmulF32D(d, h, w.shared_gate);                    // [T,1] f32
  return {std::move(sd), std::move(gl)};
}

DBuf SharedExpert(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                  const Tensor& h, int64_t T, bool fp4) {
  const int64_t H = cfg.hidden_size;
  SharedExpertParts p = SharedExpertUngated(d, w, cfg, h, T, fp4);
  DBuf shared(d, DType::kBF16, {T, H});
  vt::SharedExpertGate(d.q, shared.t(), p.sd.t(), p.gl.t());  // sigmoid(gl)*sd -> bf16
  return shared;
}

// --- Fused MoE block (M2.4). CUDA + fp4-resident only. Replaces the per-expert
// loop of tiny MatmulNvfp4 launches (each with a host round-trip Download) with
// ~3 GROUPED NVFP4 GEMM launches over ALL (token, activated-expert) pairs, kept
// entirely on-device (no host round-trip in the expert compute). The router
// top-k indices [T,top_k] ARE the per-pair expert ids (viewed as [P=T*top_k]);
// gate/up read the token hidden via a row-map (pair p -> token p/top_k), down
// reads the per-pair silu output. The E fp4-resident expert weights are indexed
// by device pointer arrays. Result is per-pair bit-identical to ExpertMlpNvfp4
// (same on-the-fly NVFP4 decode + f32 accumulation); the combine + shared expert
// match MoeBlock exactly. h [T*H] bf16 -> [T*H] bf16.
DBuf MoeBlockFusedCuda(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                       const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;

  // Hidden states already device-resident (dh [T,H] bf16); router logits +
  // top-k stay on device (the ids [T,top_k] are the pair expert ids, the
  // weights [T,top_k] feed the combine).
  Tensor drg = ResidentWeight(d, w.router_gate);          // [H,E] bf16
  DBuf dlog(d, DType::kBF16, {T, E});
  vt::Matmul(d.q, dlog.t(), dh, drg);                     // logits [T,E]
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  Tensor eids = Reshape(dtid.t(), {P});                   // [P] i32 expert ids

  // Per-layer RESIDENT fp4-expert device pointer/scale arrays + pair->token map
  // (MoeResidentFor cache): uploaded ONCE (here on first touch, during the
  // pre-warm forward), then read on every step — no per-step host-sourced upload,
  // so nothing dangles inside a captured graph. residency of the expert weights
  // themselves is still lazy through ResidentNvfp4 (their shared_ptr owns the
  // device copy); we capture the resulting stable device pointers once.
  MoeFusedResident& mr = MoeResidentFor(&w);
  if (!mr.ready) {
    std::vector<int64_t> gp(E), gs(E), up(E), us(E), dp(E), ds(E);
    std::vector<float> g2(E), u2(E), d2(E);
    for (int64_t e = 0; e < E; ++e) {
      const size_t se = static_cast<size_t>(e);
      Nvfp4Dev g = ResidentNvfp4(d, w.expert_gate_fp4[se]);
      Nvfp4Dev u = ResidentNvfp4(d, w.expert_up_fp4[se]);
      Nvfp4Dev dn = ResidentNvfp4(d, w.expert_down_fp4[se]);
      gp[se] = reinterpret_cast<int64_t>(g.packed.data);
      gs[se] = reinterpret_cast<int64_t>(g.scale.data);
      up[se] = reinterpret_cast<int64_t>(u.packed.data);
      us[se] = reinterpret_cast<int64_t>(u.scale.data);
      dp[se] = reinterpret_cast<int64_t>(dn.packed.data);
      ds[se] = reinterpret_cast<int64_t>(dn.scale.data);
      g2[se] = w.expert_gate_fp4[se].scale2;
      u2[se] = w.expert_up_fp4[se].scale2;
      d2[se] = w.expert_down_fp4[se].scale2;
    }
    const size_t eb = static_cast<size_t>(E) * sizeof(int64_t);
    const size_t fb = static_cast<size_t>(E) * sizeof(float);
    auto up_i64 = [&](const std::vector<int64_t>& h) {
      void* p = d.b.Alloc(eb);
      d.b.Copy(d.q, p, h.data(), eb);
      return p;
    };
    auto up_f32 = [&](const std::vector<float>& h) {
      void* p = d.b.Alloc(fb);
      d.b.Copy(d.q, p, h.data(), fb);
      return p;
    };
    mr.gp = up_i64(gp); mr.gs = up_i64(gs); mr.up = up_i64(up);
    mr.us = up_i64(us); mr.dp = up_i64(dp); mr.ds = up_i64(ds);
    mr.g2 = up_f32(g2); mr.u2 = up_f32(u2); mr.d2 = up_f32(d2);
    mr.ready = true;
  }
  // Resident pair->token row map for this T (function of T + top_k only).
  auto tok_it = mr.tok_map.find(T);
  if (tok_it == mr.tok_map.end()) {
    std::vector<int32_t> tok_map(static_cast<size_t>(P));
    for (int64_t p = 0; p < P; ++p)
      tok_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
    const size_t tb = static_cast<size_t>(P) * sizeof(int32_t);
    void* p = d.b.Alloc(tb);
    d.b.Copy(d.q, p, tok_map.data(), tb);
    tok_it = mr.tok_map.emplace(T, p).first;
  }
  Tensor dgp = MakeTensor(mr.gp, DType::kI64, d.q.device, {E});
  Tensor dgs = MakeTensor(mr.gs, DType::kI64, d.q.device, {E});
  Tensor dup = MakeTensor(mr.up, DType::kI64, d.q.device, {E});
  Tensor dus = MakeTensor(mr.us, DType::kI64, d.q.device, {E});
  Tensor ddp = MakeTensor(mr.dp, DType::kI64, d.q.device, {E});
  Tensor dds = MakeTensor(mr.ds, DType::kI64, d.q.device, {E});
  Tensor dg2 = MakeTensor(mr.g2, DType::kF32, d.q.device, {E});
  Tensor du2 = MakeTensor(mr.u2, DType::kF32, d.q.device, {E});
  Tensor dd2 = MakeTensor(mr.d2, DType::kF32, d.q.device, {E});
  Tensor dtok = MakeTensor(tok_it->second, DType::kI32, d.q.device, {P});

  // Grouped gate/up GEMM over all pairs (one launch each), silu-mul, grouped
  // down GEMM (act = per-pair silu output, identity row-map). expert_out lands
  // as [T,top_k,H] contiguous — exactly what MoeCombine consumes.
  DBuf dgate(d, DType::kF32, {P, I});
  DBuf dup_out(d, DType::kF32, {P, I});
  vt::MoeGroupedGemmNvfp4(d.q, dgate.t(), dh, eids, &dtok, dgp, dgs, dg2);
  vt::MoeGroupedGemmNvfp4(d.q, dup_out.t(), dh, eids, &dtok, dup, dus, du2);
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());
  DBuf ddown(d, DType::kBF16, {P, H});
  vt::MoeGroupedGemmNvfp4(d.q, ddown.t(), dact.t(), eids, nullptr, ddp, dds, dd2);
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  // Shared expert + weighted combine (out = shared + sum_j w_j * expert_out_j),
  // all device-resident (no host round-trip). MoE glue fusion folds the shared
  // sigmoid gate into the combine (one launch, no shared [T,H] round-trip).
  DBuf dout(d, DType::kBF16, {T, H});
  if (MoeGlueFuseEnabled()) {
    SharedExpertParts sp = SharedExpertUngated(d, w, cfg, dh, T, true);
    vt::MoeCombineGate(d.q, dout.t(), expert_out, dtw.t(), sp.sd.t(), sp.gl.t());
  } else {
    DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
  }
  return dout;
}

// --- Sparse-MoE block (moe-semantics.md §1-§6). Router top-k over ALL experts,
// then the ACTIVATED-EXPERT token-gather loop (not O(E)-dense), shared expert
// with sigmoid gate, and the weighted combine. h [T*H] bf16 -> [T*H] bf16.
#ifdef VT_MARLIN_NVFP4
// Build (once) the per-layer resident Marlin repacked experts. Repacks all E
// routed experts' gate/up/down (weight interleave + S0E5M3 scales + processed
// global scales) from the resident fp4 buffers, then frees the fp4 originals.
void BuildMoeMarlinResident(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                            MoeMarlinResident& mr) {
  const int E = static_cast<int>(cfg.num_experts);
  const int K = static_cast<int>(cfg.hidden_size);
  const int N = static_cast<int>(cfg.moe_intermediate_size);
  void* stream = d.q.handle;

  const int sms = vt::cuda::MarlinDeviceSms(d.q.device.index);
  mr.sms = sms;

  const size_t wg_i32 = static_cast<size_t>(K / 16) * (N * 2);  // gate/up weight elems
  const size_t wd_i32 = static_cast<size_t>(N / 16) * (K * 2);  // down weight elems
  const size_t sg_b = static_cast<size_t>(K / 16) * N;          // gate/up scale bytes
  const size_t sd_b = static_cast<size_t>(N / 16) * K;          // down scale bytes

  // Fused w13 (VT_MOE_FUSED_W13): one Marlin B operand per expert with gate+up
  // concatenated along N — needs ONE per-expert global scale for both halves
  // (the grouped GEMM takes global_scale[e], a scalar per expert). Mirror vLLM,
  // which checks allclose(w13_weight_scale_2[:, 0], w13_weight_scale_2[:, 1])
  // and then uses [:, 0] (modelopt.py:1556-1564, "Use a single gscale for
  // w13"). vLLM merely WARNS on mismatch ("Accuracy may be affected"); our
  // token-exact gate forbids that, so on any gate-vs-up scale2 mismatch we fall
  // back to the split two-GEMM layout (and say so) instead of degrading.
  bool fuse = MoeFusedW13Enabled();
  for (int e = 0; fuse && e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    if (w.expert_gate_fp4[se].scale2 != w.expert_up_fp4[se].scale2) {
      std::fprintf(stderr,
                   "vllm.cpp: VT_MOE_FUSED_W13: expert %d gate/up scale2 differ "
                   "(%g vs %g) — falling back to the split w13 layout\n",
                   e, static_cast<double>(w.expert_gate_fp4[se].scale2),
                   static_cast<double>(w.expert_up_fp4[se].scale2));
      fuse = false;
    }
  }
  mr.fused_w13 = fuse;

  if (fuse) {
    // Same total bytes as the split w_gate+w_up / s_gate+s_up pair.
    mr.w_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * wg_i32 * 4);
    mr.s_gu = d.b.Alloc(static_cast<size_t>(E) * 2 * sg_b);
    mr.g_gu = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  } else {
    mr.w_gate = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
    mr.s_gate = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.s_up = d.b.Alloc(static_cast<size_t>(E) * sg_b);
    mr.g_gate = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
    mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  }
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  // marlin_make_workspace_new(device, max_blocks_per_sm=4): sms*4 int32 locks.
  mr.workspace = d.b.Alloc(static_cast<size_t>(sms) * 4 * sizeof(int32_t));

  // combined_scale_factor: w13 = gate+up jointly (mirror vLLM), w2 = down alone.
  std::vector<const uint8_t*> gu_bufs;
  std::vector<size_t> gu_lens;
  std::vector<const uint8_t*> dn_bufs;
  std::vector<size_t> dn_lens;
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_gate_fp4[se].scale.bytes.data()));
    gu_lens.push_back(w.expert_gate_fp4[se].scale.bytes.size());
    gu_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_up_fp4[se].scale.bytes.data()));
    gu_lens.push_back(w.expert_up_fp4[se].scale.bytes.size());
    dn_bufs.push_back(reinterpret_cast<const uint8_t*>(w.expert_down_fp4[se].scale.bytes.data()));
    dn_lens.push_back(w.expert_down_fp4[se].scale.bytes.size());
  }
  const float sf_gu = vt::cuda::MarlinNvfp4CombinedScaleFactor(gu_bufs, gu_lens);
  const float sf_dn = vt::cuda::MarlinNvfp4CombinedScaleFactor(dn_bufs, dn_lens);

  // Fused-w13 concat staging (device, reused across experts — all copies and
  // repack kernels are issued on the SAME stream, so each expert's repack reads
  // its staging bytes before the next expert's copy overwrites them). The fp4
  // source layouts are row-major over N (packed [N, K/2] u8, scales [N, K/16]
  // fp8), so the vLLM w13 stack — w1 (gate) rows first, then w3 (up)
  // (fused_moe layer weight_loader shard order; silu_and_mul reads [:N] as
  // gate) — is a flat back-to-back device copy.
  const size_t pk_b = static_cast<size_t>(N) * (K / 2);  // one shard's packed bytes
  uint8_t* tmp_w = nullptr;
  uint8_t* tmp_s = nullptr;
  if (fuse) {
    tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
    tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sg_b));
  }

  std::vector<float> gg(E), gu(E), gd(E);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    Nvfp4Dev g = ResidentNvfp4(d, w.expert_gate_fp4[se]);
    Nvfp4Dev u = ResidentNvfp4(d, w.expert_up_fp4[se]);
    Nvfp4Dev dn = ResidentNvfp4(d, w.expert_down_fp4[se]);
    auto* wd = static_cast<uint32_t*>(mr.w_down) + se * wd_i32;
    auto* sdp = static_cast<uint8_t*>(mr.s_down) + se * sd_b;
    const auto* pg = static_cast<const uint8_t*>(g.packed.data);
    const auto* pu = static_cast<const uint8_t*>(u.packed.data);
    const auto* pd = static_cast<const uint8_t*>(dn.packed.data);
    if (fuse) {
      // ONE repack + ONE scale-process over the N-concatenated gate|up, with
      // size_n = 2N — the per-expert body of vLLM's repack_weight/
      // permute_scales over the stacked w13 (marlin_utils_fp4.py:388-398,
      // :423-434; size_n = num_shards * N at :375-378 / :413-415).
      auto* wgu = static_cast<uint32_t*>(mr.w_gu) + se * 2 * wg_i32;
      auto* sgup = static_cast<uint8_t*>(mr.s_gu) + se * 2 * sg_b;
      d.b.Copy(d.q, tmp_w, pg, pk_b);
      d.b.Copy(d.q, tmp_w + pk_b, pu, pk_b);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wgu, tmp_w, K, 2 * N);
      d.b.Copy(d.q, tmp_s, g.scale.data, sg_b);
      d.b.Copy(d.q, tmp_s + sg_b, u.scale.data, sg_b);
      vt::cuda::MarlinProcessExpertScales(stream, tmp_s, sgup, K, 2 * N, sf_gu);
      // vLLM w13_weight_scale_2[:, 0] (the gate/w1 scale; equality with up/w3
      // was verified above).
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_gate_fp4[se].scale2, sf_gu);
    } else {
      auto* wg = static_cast<uint32_t*>(mr.w_gate) + se * wg_i32;
      auto* wu = static_cast<uint32_t*>(mr.w_up) + se * wg_i32;
      auto* sgp = static_cast<uint8_t*>(mr.s_gate) + se * sg_b;
      auto* sup = static_cast<uint8_t*>(mr.s_up) + se * sg_b;
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wg, pg, K, N);
      vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wu, pu, K, N);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(g.scale.data), sgp,
                                          K, N, sf_gu);
      vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(u.scale.data), sup,
                                          K, N, sf_gu);
      gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_gate_fp4[se].scale2, sf_gu);
      gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_up_fp4[se].scale2, sf_gu);
    }
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wd, pd, N, K);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dn.scale.data), sdp, N,
                                        K, sf_dn);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_down_fp4[se].scale2, sf_dn);
  }
  if (fuse) {
    d.b.Copy(d.q, mr.g_gu, gg.data(), gg.size() * sizeof(float));
  } else {
    d.b.Copy(d.q, mr.g_gate, gg.data(), gg.size() * sizeof(float));
    d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  }
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // repack done → safe to free fp4 originals
  if (fuse) {
    d.b.Free(tmp_w);
    d.b.Free(tmp_s);
  }

  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    w.expert_gate_fp4[se].d_packed.reset();
    w.expert_gate_fp4[se].d_scale.reset();
    w.expert_up_fp4[se].d_packed.reset();
    w.expert_up_fp4[se].d_scale.reset();
    w.expert_down_fp4[se].d_packed.reset();
    w.expert_down_fp4[se].d_scale.reset();
  }
  mr.ready = true;
}

// Marlin fused MoE block: same router/silu/combine as MoeBlockFusedCuda, but the
// grouped GEMMs are moe_wna16_marlin_gemm over the resident repacked experts —
// 3 of them (gate, up, down), or 2 when VT_MOE_FUSED_W13 built the concatenated
// w13 operand (ONE gate+up GEMM with size_n=2I, vLLM marlin_moe.py:133-170).
// Marlin's per-pair output layout (row = t*top_k+k) matches dgate/dup/ddown, so
// MoeSiluMul/SiluAndMul + MoeCombine are unchanged. Per-pair equivalent to the
// wmma path (same weight-only fp4 dequant), so token-for-token identical.
DBuf MoeBlockFusedMarlinCuda(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                             const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t P = T * top_k;
  void* stream = d.q.handle;

  MoeMarlinResident& mr = MoeMarlinResidentFor(&w);
  if (!mr.ready) BuildMoeMarlinResident(d, w, cfg, mr);

  // Router (identical to MoeBlockFusedCuda).
  Tensor drg = ResidentWeight(d, w.router_gate);  // [H,E] bf16
  DBuf dlog(d, DType::kBF16, {T, E});
  vt::Matmul(d.q, dlog.t(), dh, drg);
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});

  // moe_align_block_size over the router top-k ids.
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(static_cast<int>(T),
                                                            static_cast<int>(top_k),
                                                            static_cast<int>(E));
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(static_cast<int>(T), static_cast<int>(top_k), static_cast<int>(E),
                                block, &max_tok, &max_blk);
  DBuf sorted_ids(d, DType::kI32, {max_tok});
  DBuf expert_ids(d, DType::kI32, {max_blk});
  DBuf num_pad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(
      stream, static_cast<const int32_t*>(dtid.t().data), static_cast<int>(T),
      static_cast<int>(top_k), static_cast<int>(E), block,
      static_cast<int32_t*>(sorted_ids.t().data), static_cast<int32_t*>(expert_ids.t().data),
      static_cast<int32_t*>(num_pad.t().data));

  Tensor wd = MakeTensor(mr.w_down, DType::kI32, d.q.device, {E, I / 16, H * 2});
  Tensor sd = MakeTensor(mr.s_down, DType::kI8, d.q.device, {E, I / 16, H});
  Tensor gd = MakeTensor(mr.g_down, DType::kF32, d.q.device, {E});
  Tensor ws = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  const int bi = block, tki = static_cast<int>(top_k);
  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int Pi = static_cast<int>(P);

  DBuf dact(d, DType::kBF16, {P, I});
  if (mr.fused_w13) {
    // ONE grouped GEMM over the N-concatenated w13 (size_n=2I, output [P,2I])
    // + one SiluAndMul on the halves — vLLM's exact marlin_moe.py shape (ONE
    // moe_wna16_marlin_gemm with size_n = w13_num_shards*N into
    // intermediate_cache1 [M*topk, 2N], fused_moe/experts/marlin_moe.py:133-160,
    // then silu_and_mul at :162-170). Removes the second GEMM's workspace
    // memset, schedule pass, and launch tail.
    Tensor wgu = MakeTensor(mr.w_gu, DType::kI32, d.q.device, {E, H / 16, 2 * I * 2});
    Tensor sgu = MakeTensor(mr.s_gu, DType::kI8, d.q.device, {E, H / 16, 2 * I});
    Tensor ggu = MakeTensor(mr.g_gu, DType::kF32, d.q.device, {E});
    DBuf dgu(d, DType::kBF16, {P, 2 * I});
    d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dgu.t(), dh, wgu, sgu, ggu, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, 2 * Ii, Hi, false});
    // SiluAndMul reads gate = dgu[:, :I], up = dgu[:, I:] (same row) — identical
    // f32 silu math + bf16 store as MoeSiluMul, so per-element it matches the
    // split path bit-for-bit given equal GEMM outputs.
    vt::SiluAndMul(d.q, dact.t(), dgu.t());
  } else {
    Tensor wg = MakeTensor(mr.w_gate, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor wu = MakeTensor(mr.w_up, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor sg = MakeTensor(mr.s_gate, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor su = MakeTensor(mr.s_up, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor gg = MakeTensor(mr.g_gate, DType::kF32, d.q.device, {E});
    Tensor gu = MakeTensor(mr.g_up, DType::kF32, d.q.device, {E});
    DBuf dgate(d, DType::kBF16, {P, I});
    DBuf dup_out(d, DType::kBF16, {P, I});
    d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dgate.t(), dh, wg, sg, gg, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
    d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dup_out.t(), dh, wu, su, gu, ws, sorted_ids.t(),
                                  expert_ids.t(), num_pad.t(), dtw.t(),
                                  vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
    vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());
  }

  DBuf ddown(d, DType::kBF16, {P, H});
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(d.q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted_ids.t(),
                                expert_ids.t(), num_pad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, 1, Pi, Hi, Ii, false});
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  DBuf dout(d, DType::kBF16, {T, H});
  if (MoeGlueFuseEnabled()) {
    // Fuse shared-expert gate into the combine (one launch, no shared round-trip).
    SharedExpertParts sp = SharedExpertUngated(d, w, cfg, dh, T, true);
    vt::MoeCombineGate(d.q, dout.t(), expert_out, dtw.t(), sp.sd.t(), sp.gl.t());
  } else {
    DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
    vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
  }
  return dout;
}
#endif  // VT_MARLIN_NVFP4

DBuf MoeBlock(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
              const Tensor& dh, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  // fp4-resident NVFP4 experts/shared (M2.2b real-ckpt CUDA load) vs bf16
  // (synthetic / GGUF). Exactly one set is populated (see qwen3_5_weights.h).
  const bool fp4 = !w.expert_gate_fp4.empty();

  // M2.4/M2.5 fused MoE: CUDA + fp4-resident does the expert compute in ~3
  // grouped GEMM launches fully on-device (no host round-trip — capturable).
  // The bf16 / CPU / GGUF reference below keeps the per-expert token-gather
  // path on host (not the capture target); the fused output is per-pair
  // bit-identical to it (same NVFP4 decode).
  if (fp4 && d.q.device.type == vt::DeviceType::kCUDA) {
#ifdef VT_MARLIN_NVFP4
    if (MarlinMoeEnabled()) return MoeBlockFusedMarlinCuda(d, w, cfg, dh, T);
#endif
    return MoeBlockFusedCuda(d, w, cfg, dh, T);
  }

  // Reference path: download the hidden once, then gather + per-expert MLP.
  std::vector<uint16_t> h(static_cast<size_t>(T) * H);
  d.b.Copy(d.q, h.data(), dh.data, h.size() * sizeof(uint16_t));
  d.b.Synchronize(d.q);

  // Router: logits = x @ gate.T (bf16, §2), softmax/top-k/renormalize (§3).
  std::vector<uint16_t> logits = MatmulBf16(d, h, T, H, w.router_gate);  // [T,E]
  DBuf dlog(d, DType::kBF16, {T, E}, logits.data());
  DBuf dtw(d, DType::kF32, {T, top_k});
  DBuf dtid(d, DType::kI32, {T, top_k});
  vt::MoeRouterTopK(d.q, dtw.t(), dtid.t(), dlog.t(),
                    vt::MoeRouterTopKArgs{static_cast<int>(top_k), true});
  std::vector<float> weights(static_cast<size_t>(T) * top_k);
  std::vector<int32_t> ids(static_cast<size_t>(T) * top_k);
  dtw.Download(d, weights.data());
  dtid.Download(d, ids.data());

  // Activated-expert gather: per expert, the (token, slot) pairs routed to it.
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
      static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < top_k; ++j)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t) * top_k + j])]
          .push_back({t, j});

  std::vector<uint16_t> expert_out(static_cast<size_t>(T) * top_k * H, 0);
  for (int64_t e = 0; e < E; ++e) {
    const auto& list = lists[static_cast<size_t>(e)];
    if (list.empty()) continue;
    const int64_t n = static_cast<int64_t>(list.size());
    std::vector<uint16_t> xg(static_cast<size_t>(n) * H);
    for (int64_t r = 0; r < n; ++r)
      std::memcpy(xg.data() + static_cast<size_t>(r) * H,
                  h.data() + static_cast<size_t>(list[r].first) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    std::vector<uint16_t> y =
        fp4 ? ExpertMlpNvfp4(d, w.expert_gate_fp4[static_cast<size_t>(e)],
                             w.expert_up_fp4[static_cast<size_t>(e)],
                             w.expert_down_fp4[static_cast<size_t>(e)], xg, n, H, I)
            : ExpertMlp(d, w.expert_gate[static_cast<size_t>(e)],
                        w.expert_up[static_cast<size_t>(e)],
                        w.expert_down[static_cast<size_t>(e)], xg, n, H, I);
    for (int64_t r = 0; r < n; ++r) {
      const int64_t t = list[r].first, j = list[r].second;
      std::memcpy(expert_out.data() + static_cast<size_t>(t * top_k + j) * H,
                  y.data() + static_cast<size_t>(r) * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
  }

  // Shared expert (moe-semantics.md §5): device-resident (takes dh).
  DBuf shared = SharedExpert(d, w, cfg, dh, T, fp4);

  // Combine (moe-semantics.md §6): out = shared + sum_j w_j * expert_out_j.
  DBuf deo(d, DType::kBF16, {T, top_k, H}, expert_out.data());
  DBuf dwt(d, DType::kF32, {T, top_k}, weights.data());
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), deo.t(), dwt.t(), &shared.t());
  return dout;
}

// Returns true (+ fills *scale) when the layer's input-fed fp8 projections share
// ONE static input_scale, so a single RmsNormQuantFp8 can quantize the shared
// activation once and feed them all (attn q/k/v; GDN in_proj_qkv/z). The fp8
// analog of the fp4 fuse_qkv input_global_scale guard; exact float equality (only
// fuse when the checkpoint scales are truly identical).
bool Fp8SharedInputScale(bool is_linear_attention, const GdnLayerWeights& g,
                         const FullAttnLayerWeights& a, float* scale) {
  if (is_linear_attention) {
    if (g.in_proj_qkv_fp8.Empty() || g.in_proj_z_fp8.Empty()) return false;
    if (g.in_proj_qkv_fp8.input_scale != g.in_proj_z_fp8.input_scale) return false;
    *scale = g.in_proj_qkv_fp8.input_scale;
    return true;
  }
  if (a.q_proj_fp8.Empty() || a.k_proj_fp8.Empty() || a.v_proj_fp8.Empty()) return false;
  if (a.q_proj_fp8.input_scale != a.k_proj_fp8.input_scale ||
      a.q_proj_fp8.input_scale != a.v_proj_fp8.input_scale)
    return false;
  *scale = a.q_proj_fp8.input_scale;
  return true;
}

// Input-layernorm producer shared by RunLayer / RunLayerPaged: residual-add +
// gemma RMSNorm -> bf16 `dhn`. With VT_FUSE_RMSNORM_FP8QUANT (35B, shared fp8
// input_scale) it ALSO emits the shared static-quant fp8 activation in the SAME
// pass (vt::RmsNormQuantFp8, mirror vLLM Inductor fused_add_rms_norm_static_fp8_
// quant), returned so the block feeds it to q/k/v (or in_proj_qkv/z) once. GDN's
// in_proj_a/b still read bf16 `dhn`, so it is emitted there; full-attn reads only
// the fp8, so bf16 `dhn` is skipped. std::nullopt (no fp8) = the byte-identical
// plain RmsNorm path.
std::optional<DBuf> InputLayernormFp8(Dev d, const Qwen3_5MoeLayerWeights& layer,
                                      const HfConfig& cfg, DBuf& hidden, DBuf& res, DBuf& dhn,
                                      int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  float fp8_scale = 0.0F;
  const bool fuse = FuseRmsNormFp8QuantEnabled() && d.q.device.type == vt::DeviceType::kCUDA &&
                    Fp8SharedInputScale(layer.is_linear_attention, layer.gdn, layer.attn,
                                        &fp8_scale);
  if (fuse) {
    std::optional<DBuf> dhn_fp8;
    dhn_fp8.emplace(d, DType::kI8, std::vector<int64_t>{T, H});
    Tensor* out_bf16 = layer.is_linear_attention ? &dhn.t() : nullptr;
    vt::RmsNormQuantFp8(d.q, dhn_fp8->t(), out_bf16, hidden.t(), dw_in,
                        vt::RmsNormArgs{eps, true}, &res.t(), fp8_scale);
    return dhn_fp8;
  }
  // Qwen3NextRMSNorm == GemmaRMSNorm (weight applied as 1+w). res += hidden.
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());
  return std::nullopt;
}

// One decoder layer over the fused residual stream. `hidden` (bf16 [T*H]) is
// the previous block's output (the delta); `res` (f32 [T,H], device) is the
// accumulator. Mirrors qwen3_next.py::Qwen3NextDecoderLayer.forward:
//   h  = input_layernorm(hidden, res)          # res += hidden; h = norm(res)
//   a  = attn/gdn(h)
//   h2 = post_attention_layernorm(a, res)      # res += a; h2 = norm(res)
//   hidden = mlp(h2)                            # MoE block; returned as delta
void RunLayer(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
              DBuf& hidden, DBuf& res, const std::vector<int32_t>& positions,
              int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  DBuf dhn(d, DType::kBF16, {T, H});
  std::optional<DBuf> dhn_fp8 = InputLayernormFp8(d, layer, cfg, hidden, res, dhn, T);
  const Tensor* h_fp8 = dhn_fp8 ? &dhn_fp8->t() : nullptr;

  DBuf attn = layer.is_linear_attention
                  ? GdnBlock(d, layer.gdn, cfg, dhn.t(), T, h_fp8)
                  : FullAttnBlock(d, layer.attn, cfg, dhn.t(), positions, T, h_fp8);

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  hidden = MoeBlock(d, layer.moe, cfg, dh2.t(), T);
}

// --- Dense SwiGLU MLP block (the 27B's replacement for the MoE block; notes
// §2). down( silu(gate(x)) * up(x) ), intermediate = cfg.intermediate_size.
// Mirrors the shared-expert silu-mul MLP (no router, no expert gather, no output
// gate). h [T,H] bf16 (device) -> DBuf [T,H] bf16 (device). Reused by the dense
// forward below; the gate/up/down weights are W4A4-materialized-to-bf16 at load.
DBuf DenseMlpBlock(Dev d, const DenseMlpWeights& w, const HfConfig& cfg,
                   const Tensor& dh, int64_t T) {
  const int64_t I = cfg.intermediate_size;
  // fp4-resident W4A4 path (real 27B, notes §5 step-6a) when populated; else the
  // bf16 path (synthetic CPU tests). Exactly one representation is filled.
  const bool fp4 = !w.gate_proj_fp4.Empty();
#ifdef VT_CUTLASS_NVFP4
  // vLLM's production topology is one MergedColumnParallelLinear gate_up_proj,
  // not two independently-scaled linears. Its CT loader takes max(input
  // divisor) and max(weight divisor) across the logical shards, computes one
  // alpha, quantizes once and launches one [T,H]x[2I,H] GEMM. This branch is W2
  // default; VT_FP4_MERGED_GATE_UP=0 restores the split W2 diagnostic and
  // VT_FP4_FULL_TACTICS=0 restores W1 including its split model topology.
  if (fp4 && MergedGateUpEligible(w, d)) {
    DBuf gate_up = MergedGateUpCutlassD(d, dh, w);  // bf16 [T,2I]
    if (FuseSiluQuantEnabled() &&
        w.down_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
      DBuf ap(d, DType::kI8, {T, I / 2});
      const bool direct_scale = DirectFp4ScaleEligible(d);
      DBuf as(d, DType::kI8,
              direct_scale ? CutlassFp4ScaleShape(T, I)
                           : std::vector<int64_t>{T, I / 16});
      const vt::Fp4ScaleLayout scale_layout =
          direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                       : vt::Fp4ScaleLayout::kLinear;
      if (MergedSiluQuantEnabled()) {
        vt::SiluAndMulFp4Quant(d.q, ap.t(), as.t(), gate_up.t(),
                               w.down_proj_fp4.input_global_scale_inv,
                               scale_layout);
      } else {
        DBuf act(d, DType::kBF16, {T, I});
        vt::SiluAndMul(d.q, act.t(), gate_up.t());
        vt::ScaledFp4Quant(d.q, ap.t(), as.t(), act.t(),
                           w.down_proj_fp4.input_global_scale_inv,
                           scale_layout);
      }
      return MatmulNvfp4Fp4DirectD(d, ap.t(), as.t(), w.down_proj_fp4,
                                   DType::kBF16,
                                   direct_scale ? &as.t() : nullptr);
    }
    DBuf act(d, DType::kBF16, {T, I});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());
    return MatmulNvfp4Bf16D(d, act.t(), w.down_proj_fp4);
  }
#endif
  // QUANTIZE-ONCE for gate/up: shared activation dh + shared input_global_scale ->
  // one ScaledFp4Quant feeding both fp4 GEMMs (removes 1 redundant [T,H] quant).
  const bool fuse_gu =
      fp4 && FuseQuantOnceEnabled() && d.q.device.type == vt::DeviceType::kCUDA &&
      w.gate_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled() &&
      w.gate_proj_fp4.input_global_scale_inv == w.up_proj_fp4.input_global_scale_inv;
  std::optional<DBuf> gu_ap, gu_as;
#ifdef VT_CUTLASS_NVFP4
  const bool gu_direct_scale = fuse_gu && DirectFp4ScaleEligible(d);
#else
  const bool gu_direct_scale = false;
#endif
  if (fuse_gu) {
    const int64_t H = dh.shape[1];
    gu_ap.emplace(d, DType::kI8, std::vector<int64_t>{T, H / 2});
    gu_as.emplace(d, DType::kI8,
                  gu_direct_scale
                      ? CutlassFp4ScaleShape(T, H)
                      : std::vector<int64_t>{T, H / 16});
    vt::ScaledFp4Quant(
        d.q, gu_ap->t(), gu_as->t(), dh,
        w.gate_proj_fp4.input_global_scale_inv,
        gu_direct_scale ? vt::Fp4ScaleLayout::kCutlassSwizzled
                        : vt::Fp4ScaleLayout::kLinear);
  }
  // SWIZZLE-ONCE (VT_SWIZZLE_IN_QUANT): swizzle the shared gate/up activation SF
  // ONCE and feed the already-swizzled SF to both GEMMs (skipping each one's
  // internal SwizzleBlockscale). nullptr (OFF / non-cutlass) = per-projection
  // swizzle, byte-identical to the current path.
  const Tensor* gu_sf_sw_p = nullptr;
#ifdef VT_CUTLASS_NVFP4
  std::optional<DBuf> gu_sf_sw;
  if (gu_direct_scale) {
    gu_sf_sw_p = &gu_as->t();
  } else if (fuse_gu && SwizzleInQuantEnabled() && NvfpCutlassEnabled()) {
    gu_sf_sw.emplace(SwizzleActScaleOnce(d, gu_as->t()));
    gu_sf_sw_p = &gu_sf_sw->t();
  }
#endif
  // gate/up output bf16 (VT_BF16_GEMM_OUT, rank-1 lever) — matches vLLM bf16 dtype,
  // halves the GEMM write + MoeSiluMul read; else f32 (current). MoeSiluMul is
  // templated on the input dtype so both work.
  const DType gu_out = Bf16GemmOutEnabled() ? DType::kBF16 : DType::kF32;
  DBuf gate = fuse_gu ? MatmulNvfp4Fp4DirectD(d, gu_ap->t(), gu_as->t(), w.gate_proj_fp4, gu_out,
                                              gu_sf_sw_p)
              : fp4 ? (Bf16GemmOutEnabled() ? MatmulNvfp4Bf16D(d, dh, w.gate_proj_fp4)
                                            : MatmulNvfp4F32D(d, dh, w.gate_proj_fp4))
                    : MatmulF32D(d, dh, w.gate_proj);  // [T,I]
  DBuf up = fuse_gu ? MatmulNvfp4Fp4DirectD(d, gu_ap->t(), gu_as->t(), w.up_proj_fp4, gu_out,
                                            gu_sf_sw_p)
            : fp4 ? (Bf16GemmOutEnabled() ? MatmulNvfp4Bf16D(d, dh, w.up_proj_fp4)
                                          : MatmulNvfp4F32D(d, dh, w.up_proj_fp4))
                  : MatmulF32D(d, dh, w.up_proj);      // [T,I]
  // FUSED silu-mul + fp4-quant → down GEMM (no bf16 intermediate). Only when the
  // down_proj would have quantized its activation anyway (true-W4A4, CUDA) — same
  // guard as MatmulNvfp4Bf16D's MatmulNvfp4Fp4D route. Bit-identical to the else.
  if (fp4 && FuseSiluQuantEnabled() && d.q.device.type == vt::DeviceType::kCUDA &&
      w.down_proj_fp4.IsTrueW4A4() && TrueW4A4Enabled()) {
    DBuf ap(d, DType::kI8, {T, I / 2});
#ifdef VT_CUTLASS_NVFP4
    const bool direct_scale = DirectFp4ScaleEligible(d);
#else
    const bool direct_scale = false;
#endif
    DBuf as(d, DType::kI8,
            direct_scale ? CutlassFp4ScaleShape(T, I)
                         : std::vector<int64_t>{T, I / 16});
    vt::SiluMulFp4Quant(d.q, ap.t(), as.t(), gate.t(), up.t(),
                        w.down_proj_fp4.input_global_scale_inv,
                        direct_scale
                            ? vt::Fp4ScaleLayout::kCutlassSwizzled
                            : vt::Fp4ScaleLayout::kLinear);
    return MatmulNvfp4Fp4DirectD(
        d, ap.t(), as.t(), w.down_proj_fp4, DType::kBF16,
        direct_scale ? &as.t() : nullptr);
  }
  DBuf act(d, DType::kBF16, {T, I});
  vt::MoeSiluMul(d.q, act.t(), gate.t(), up.t());  // silu(gate)*up -> bf16
  return fp4 ? MatmulNvfp4Bf16D(d, act.t(), w.down_proj_fp4)
             : MatmulBf16D(d, act.t(), w.down_proj);  // [T,H] bf16
}

// One dense decoder layer (notes §2). Same residual/norm thread as RunLayer, but
// the MoE block is swapped for the dense SwiGLU MLP; the GDN / full-attention
// blocks are the 35B helpers reused verbatim. `hidden` (bf16 [T,H]) is the delta;
// `res` (f32 [T,H]) the accumulator.
void RunDenseLayer(Dev d, const Qwen3_5DenseLayerWeights& layer,
                   const HfConfig& cfg, DBuf& hidden, DBuf& res,
                   const std::vector<int32_t>& positions, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf attn = layer.is_linear_attention
                  ? GdnBlock(d, layer.gdn, cfg, dhn.t(), T)
                  : FullAttnBlock(d, layer.attn, cfg, dhn.t(), positions, T);

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  hidden = DenseMlpBlock(d, layer.mlp, cfg, dh2.t(), T);
}

// Batched PAGED decoder layer (M1.8 Task 3). Same residual/norm/MoE thread as
// RunLayer, but the attention block reads/writes the paged KV cache
// (full-attn: attn_kv) or the persistent GDN mamba state (GDN: gdn_state).
// Exactly one of {attn_kv, gdn_state} is non-null (per layer type).
void RunLayerPaged(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
                   DBuf& hidden, DBuf& res, const StepDevInputs& sdi,
                   const CommonAttentionMetadata& attn_meta,
                   const GDNAttentionMetadata& gdn_meta,
                   const PagedKvCache* attn_kv, const GdnStateCache* gdn_state,
                   int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  DBuf dhn(d, DType::kBF16, {T, H});
  std::optional<DBuf> dhn_fp8 = InputLayernormFp8(d, layer, cfg, hidden, res, dhn, T);
  const Tensor* h_fp8 = dhn_fp8 ? &dhn_fp8->t() : nullptr;

  DBuf attn = [&] {
    if (layer.is_linear_attention) {
      VT_CHECK(gdn_state != nullptr, "paged layer: GDN layer needs a GdnStateCache");
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), sdi, gdn_meta, *gdn_state, T, h_fp8);
    }
    VT_CHECK(attn_kv != nullptr, "paged layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), sdi, attn_meta,
                              *attn_kv, T, h_fp8);
  }();

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  hidden = MoeBlock(d, layer.moe, cfg, dh2.t(), T);
}

// Batched PAGED dense decoder layer (27B; notes §5). Identical residual/norm
// thread + paged attention wiring as RunLayerPaged, but the MoE block is swapped
// for the dense SwiGLU MLP (DenseMlpBlock) and the layer carries dense weights.
// The GDN / full-attn paged blocks are the 35B helpers reused VERBATIM. Exactly
// one of {attn_kv, gdn_state} is non-null (per layer type).
void RunDenseLayerPaged(Dev d, const Qwen3_5DenseLayerWeights& layer,
                        const HfConfig& cfg, DBuf& hidden, DBuf& res,
                        const StepDevInputs& sdi,
                        const CommonAttentionMetadata& attn_meta,
                        const GDNAttentionMetadata& gdn_meta,
                        const PagedKvCache* attn_kv,
                        const GdnStateCache* gdn_state, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf attn = [&] {
    if (layer.is_linear_attention) {
      VT_CHECK(gdn_state != nullptr,
               "paged dense layer: GDN layer needs a GdnStateCache");
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), sdi, gdn_meta, *gdn_state, T);
    }
    VT_CHECK(attn_kv != nullptr,
             "paged dense layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), sdi, attn_meta,
                              *attn_kv, T);
  }();

  Tensor dw_post = ResidentWeight(d, layer.post_attention_layernorm, {H});
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), attn.t(), dw_post, vt::RmsNormArgs{eps, true}, &res.t());

  hidden = DenseMlpBlock(d, layer.mlp, cfg, dh2.t(), T);
}

}  // namespace

// Embed: hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident table).
// KEPT OUTSIDE THE CUDA-GRAPH (M2.5 Phase 2): the CUDA Embedding op allocates a
// device bounds-check flag (cudaMalloc/cudaFree) and syncs the stream, all of
// which are illegal inside a capture region — and it consumes host token_ids.
// The graph driver runs this per step into its PERSISTENT hidden buffer, then
// captures/replays ForwardLayers over that fixed hidden address.
static void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
                      const Qwen3_5MoeWeights& weights, const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// The CAPTURABLE paged forward region: everything AFTER the embedding — the
// residual stream (res=0), the N paged decoder layers, the final RMSNorm and the
// lm_head — returning the [T,vocab] f32 logits as a device DBuf (NO host
// Download; the caller Downloads, or the graph keeps it resident as the graph's
// output). `hidden_in` is the embedded input (a view over the graph's persistent
// hidden buffer on the replay path); it is COPIED into a working buffer so the
// layers' in-place residual thread never disturbs the persistent embedding.
// Split out of Forward (M2.5 Phase 2) so the exact op sequence is what the graph
// captures/replays; every per-step-varying input is read from a HOST vector
// argument (positions / the metadata vectors), whose host->device copies are
// capturable on GB10 (pageable memory access) and which the graph driver keeps
// persistent + mutates in place so replays pick up each new token's inputs.
static DBuf ForwardLayers(Dev d, const Tensor& hidden_in,
                          const std::vector<int32_t>& positions,
                          const CommonAttentionMetadata& attn_meta,
                          const GDNAttentionMetadata& gdn_meta,
                          const std::vector<PagedKvCache>& attn_kv,
                          const std::vector<GdnStateCache>& gdn_state,
                          const Qwen3_5MoeWeights& weights, const HfConfig& config,
                          const std::vector<int32_t>& logits_indices = {}) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy of the embedded hidden (device->device; captured). RunLayerPaged
  // reassigns `hidden` per layer, so this must NOT alias the persistent buffer.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, ResidualDType(), {T, H});
  res.Zero(d);

  // Upload the per-step inputs ONCE (positions + full-attn metadata + GDN decode
  // state indices) into persistent device buffers all layers read — replaces the
  // per-layer H2D re-uploads (the decode host-stall root; see StepDevInputs).
  StepDevInputs sdi = BuildStepDevInputs(d, positions, attn_meta, gdn_meta);
  // Build the fused-preamble cos|sin cache ONCE; fp4_attn keys the per-arch
  // default (fp8/bf16 attn — the 35B — stays OFF; VT_FUSE_ATTN_PREAMBLE overrides).
  const bool fp4_attn = [&] {
    for (const auto& l : weights.layers)
      if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
    return false;
  }();
  MaybeBuildAttnCosSin(d, sdi, config, T, fp4_attn);

  int64_t fa_idx = 0, gdn_idx = 0;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3_5MoeLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    const PagedKvCache* kv =
        layer.is_linear_attention ? nullptr : &attn_kv[static_cast<size_t>(fa_idx++)];
    const GdnStateCache* gs =
        layer.is_linear_attention ? &gdn_state[static_cast<size_t>(gdn_idx++)] : nullptr;
    RunLayerPaged(d, layer, config, hidden, res, sdi, attn_meta, gdn_meta,
                  kv, gs, T);
  }

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // Logits gather (perf): mirror vLLM's gather-BEFORE-lm_head. On a prefill/mixed
  // step (len(logits_indices) < T) gather only the per-request last-token hidden
  // rows [num_reqs,H] and run lm_head on those → [num_reqs,vocab]. This is the
  // whole win: lm_head over ~num_reqs rows instead of T, and only that tiny
  // logits tensor is materialized/Downloaded. Pure-decode / graph replay pass
  // empty indices (identity) so the full [T,vocab] path is unchanged.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  if (do_gather) {
    const int64_t n_out = static_cast<int64_t>(logits_indices.size());
    DBuf dgather(d, DType::kBF16, {n_out, H});
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    return weights.lm_head_fp4.Empty()
               ? MatmulF32D(d, dgather.t(), weights.lm_head)
               : MatmulNvfp4F32D(d, dgather.t(), weights.lm_head_fp4);
  }

  // lm_head: fp4-resident (M2.2b) when populated, else the bf16/GGUF path.
  return weights.lm_head_fp4.Empty()
             ? MatmulF32D(d, dnorm.t(), weights.lm_head)
             : MatmulNvfp4F32D(d, dnorm.t(), weights.lm_head_fp4);
}

// Full eager paged forward body: embed (host token_ids) then the capturable
// layer region. Used by Qwen3_5Model::Forward and the graph driver's eager
// fallback / cold-shape pre-warm step (one contiguous stream, no capture).
static DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                        const std::vector<int32_t>& positions,
                        const CommonAttentionMetadata& attn_meta,
                        const GDNAttentionMetadata& gdn_meta,
                        const std::vector<PagedKvCache>& attn_kv,
                        const std::vector<GdnStateCache>& gdn_state,
                        const Qwen3_5MoeWeights& weights, const HfConfig& config,
                        const std::vector<int32_t>& logits_indices = {}) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  DBuf hidden(d, DType::kBF16, {T, H});
  EmbedInto(d, hidden, token_ids, weights, config);
  return ForwardLayers(d, hidden.t(), positions, attn_meta, gdn_meta, attn_kv,
                       gdn_state, weights, config, logits_indices);
}

// Shared shape/count validation for the paged forward entry points.
static void CheckPagedForward(const std::vector<int32_t>& token_ids,
                              const std::vector<int32_t>& positions,
                              const CommonAttentionMetadata& attn_meta,
                              const std::vector<PagedKvCache>& attn_kv,
                              const std::vector<GdnStateCache>& gdn_state,
                              const Qwen3_5MoeWeights& weights,
                              const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "qwen3_5 paged forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 paged forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 paged forward: weights.layers size must equal num_hidden_layers");
  VT_CHECK(attn_meta.num_actual_tokens == T,
           "qwen3_5 paged forward: attn_meta.num_actual_tokens must equal T");
  int64_t n_full = 0, n_gdn = 0;
  for (const auto& l : weights.layers)
    (l.is_linear_attention ? n_gdn : n_full) += 1;
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == n_full,
           "qwen3_5 paged forward: attn_kv count must equal full-attn layer count");
  VT_CHECK(static_cast<int64_t>(gdn_state.size()) == n_gdn,
           "qwen3_5 paged forward: gdn_state count must equal GDN layer count");
}

// Transfer a freshly-produced [rows, vocab] device logits DBuf into an OWNING
// ForwardLogits (the sampler-on-device return). The pool block's lifetime moves
// into a shared_ptr whose deleter returns it to the DevicePool — so there is NO
// per-step cudaMalloc/cudaFree and the buffer safely outlives sampling (the
// runner holds the ForwardLogits across execute_model -> sample_tokens).
static ForwardLogits WrapDeviceLogits(Dev d, DBuf&& dlogits, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = dlogits.t().shape[0];
  fl.vocab = vocab;
  fl.device_tensor = dlogits.t();  // view (raw data ptr survives Release)
  const size_t alloc = dlogits.alloc_bytes();
  void* p = dlogits.Release();     // dtor now a no-op; we own the Pool().Put
  fl.device_storage = std::shared_ptr<void>(
      p, [alloc](void* q) { Pool().Put(alloc, q); });
  (void)d;
  return fl;
}

// Wrap the first `rows` rows of an EXTERNALLY-owned (persistent) device logits
// buffer as a NON-owning ForwardLogits view (the decode-graph slot keeps the
// storage alive across steps; each replay overwrites it, so the sampler may
// mutate the view in place). Rows are contiguous (row-major), so the first `rows`
// rows are a plain prefix view over `base`.
static ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows,
                                      int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  // Non-owning: keep on_device() true without taking ownership of `base`.
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  return fl;
}

void Qwen3_5Model::PrepareMarlinResident(const Qwen3_5MoeWeights& weights,
                                         const HfConfig& config, vt::Queue& queue) {
#ifdef VT_MARLIN_NVFP4
  if (queue.device.type != vt::DeviceType::kCUDA || !MarlinMoeEnabled()) return;
  Dev d{vt::GetBackend(queue.device.type), queue};
  for (const auto& layer : weights.layers) {
    const MoeBlockWeights& moe = layer.moe;
    if (!moe.expert_gate_fp4.empty())
      BuildMoeMarlinResident(d, moe, config, MoeMarlinResidentFor(&moe));
    if (SharedGateUpFusedEligible(moe.shared_gate_proj_fp4, moe.shared_up_proj_fp4)) {
      // Fused gate_up pair resident INSTEAD of the two singles (same total
      // bytes; the forward takes the fused path under the identical guard).
      BuildMarlinDensePairResident(d, moe.shared_gate_proj_fp4, moe.shared_up_proj_fp4,
                                   MarlinDensePairResidentFor(&moe.shared_gate_proj_fp4));
    } else {
      if (!moe.shared_gate_proj_fp4.Empty())
        BuildMarlinDenseResident(d, moe.shared_gate_proj_fp4,
                                 MarlinDenseResidentFor(&moe.shared_gate_proj_fp4));
      if (!moe.shared_up_proj_fp4.Empty())
        BuildMarlinDenseResident(d, moe.shared_up_proj_fp4,
                                 MarlinDenseResidentFor(&moe.shared_up_proj_fp4));
    }
    if (!moe.shared_down_proj_fp4.Empty())
      BuildMarlinDenseResident(d, moe.shared_down_proj_fp4,
                               MarlinDenseResidentFor(&moe.shared_down_proj_fp4));
  }
  if (!weights.lm_head_fp4.Empty())
    BuildMarlinDenseResident(d, weights.lm_head_fp4, MarlinDenseResidentFor(&weights.lm_head_fp4));
  d.b.Synchronize(d.q);
#else
  (void)weights;
  (void)config;
  (void)queue;
#endif
}

std::vector<float> Qwen3_5Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state, weights,
                    config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices);
  // dlogits is [n_out, vocab]: n_out == num_reqs when gathered (prefill/mixed),
  // else T. Download exactly the produced rows (the ONE host Download).
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3_5Model::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices) {
  CheckPagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state, weights,
                    config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config, logits_indices);
  // Keep the [n_out, vocab] logits ON DEVICE (no Download) — the sampler reads
  // them directly.
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

std::vector<float> Qwen3_5Model::ForwardDense(const std::vector<int32_t>& token_ids,
                                              const std::vector<int32_t>& positions,
                                              const Qwen3_5MoeWeights& weights,
                                              const HfConfig& config,
                                              vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 forward: weights.layers size must equal num_hidden_layers");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
             positions, T);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  // Final norm is GemmaRMSNorm too (weight applied as 1+w).
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head (the one host Download): fp4-resident (M2.2b) or bf16/GGUF path.
  DBuf dlogits = weights.lm_head_fp4.Empty()
                     ? MatmulF32D(d, dnorm.t(), weights.lm_head)
                     : MatmulNvfp4F32D(d, dnorm.t(), weights.lm_head_fp4);
  std::vector<float> logits(static_cast<size_t>(T) * vocab);
  dlogits.Download(d, logits.data());
  return logits;
}

std::vector<float> Qwen3_5DenseModel::ForwardDense(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 dense forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense forward: weights.layers size must equal num_hidden_layers");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  // For a TEXT-only step the three mRoPE position streams are identical, so the
  // partial NeoX RoPE in FullAttnBlock degenerates to 1-D RoPE over `positions`
  // (notes §2). The vision tower / image-video merger are DEFERRED.
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, ResidualDType(), {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunDenseLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
                  positions, T);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // lm_head is unquantized bf16 in the 27B (notes §3.6): the one host Download.
  DBuf dlogits = MatmulF32D(d, dnorm.t(), weights.lm_head);
  std::vector<float> logits(static_cast<size_t>(T) * vocab);
  dlogits.Download(d, logits.data());
  return logits;
}

Qwen3_5MTPModel::Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                                 const Qwen3_5DenseWeights& target,
                                 const HfConfig& config)
    : weights_(&weights),
      config_(&config),
      embed_tokens_(&target.embed_tokens),
      lm_head_(&target.lm_head) {
  VT_CHECK(weights.kind == Qwen3_5MTPKind::kDense,
           "qwen3_5 MTP: dense target requires dense MTP weights");
}

Qwen3_5MTPModel::Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                                 const Qwen3_5MoeWeights& target,
                                 const HfConfig& config)
    : weights_(&weights),
      config_(&config),
      embed_tokens_(&target.embed_tokens),
      lm_head_(&target.lm_head),
      lm_head_fp4_(&target.lm_head_fp4) {
  VT_CHECK(weights.kind == Qwen3_5MTPKind::kMoe,
           "qwen3_5 MTP: MoE target requires MoE MTP weights");
}

Qwen3_5MTPHiddenStates Qwen3_5MTPModel::Forward(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions,
    const vt::Tensor& target_hidden_states, vt::Queue& queue,
    int64_t spec_step_idx) const {
  const int64_t tokens = static_cast<int64_t>(input_ids.size());
  const int64_t hidden_size = config_->hidden_size;
  const int64_t vocab_size = config_->vocab_size;
  const int64_t num_layers = weights_->NumLayers();
  VT_CHECK(tokens > 0, "qwen3_5 MTP forward: empty input_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == tokens,
           "qwen3_5 MTP forward: positions length must equal token count");
  VT_CHECK(spec_step_idx >= 0 && num_layers > 0,
           "qwen3_5 MTP forward: invalid spec step/layer count");
  VT_CHECK(target_hidden_states.rank == 2 &&
               target_hidden_states.shape[0] == tokens &&
               target_hidden_states.shape[1] == hidden_size &&
               target_hidden_states.dtype == DType::kBF16 &&
               target_hidden_states.IsContiguous() &&
               target_hidden_states.device == queue.device,
           "qwen3_5 MTP forward: target hidden states must be contiguous "
           "bf16 [T,H] on the queue device");
  VT_CHECK(weights_->fc.rank == 2 && weights_->fc.nk &&
               weights_->fc.shape[0] == hidden_size &&
               weights_->fc.shape[1] == 2 * hidden_size,
           "qwen3_5 MTP forward: fc must be raw bf16 [H,2H]");

  Dev device{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config_->rms_norm_eps);

  // Qwen3_5MultiTokenPredictor.forward: shared embedding, then independent
  // Gemma RMSNorms over the embedding and target final hidden states.
  Tensor embedding_table =
      ResidentWeight(device, *embed_tokens_, {vocab_size, hidden_size});
  DBuf device_ids(device, DType::kI32, {tokens}, input_ids.data());
  DBuf embedding(device, DType::kBF16, {tokens, hidden_size});
  vt::Embedding(device.q, embedding.t(), embedding_table, device_ids.t());

  Tensor embedding_norm_weight = ResidentWeight(
      device, weights_->pre_fc_norm_embedding, {hidden_size});
  Tensor hidden_norm_weight =
      ResidentWeight(device, weights_->pre_fc_norm_hidden, {hidden_size});
  DBuf embedding_norm(device, DType::kBF16, {tokens, hidden_size});
  DBuf target_norm(device, DType::kBF16, {tokens, hidden_size});
  vt::RmsNorm(device.q, embedding_norm.t(), embedding.t(),
              embedding_norm_weight, vt::RmsNormArgs{eps, true});
  vt::RmsNorm(device.q, target_norm.t(), target_hidden_states,
              hidden_norm_weight, vt::RmsNormArgs{eps, true});

  // torch.cat([embedding_norm, target_norm], -1). Backend::Copy is used row by
  // row so this remains portable and exact without introducing a one-off MTP
  // kernel; M-mtp-1 may fuse this if profiling makes it material.
  DBuf concatenated(device, DType::kBF16, {tokens, 2 * hidden_size});
  const size_t row_bytes =
      static_cast<size_t>(hidden_size) * vt::SizeOf(DType::kBF16);
  auto* cat = static_cast<uint8_t*>(concatenated.ptr());
  const auto* embed =
      static_cast<const uint8_t*>(embedding_norm.t().data);
  const auto* target = static_cast<const uint8_t*>(target_norm.t().data);
  for (int64_t token = 0; token < tokens; ++token) {
    const size_t source_offset = static_cast<size_t>(token) * row_bytes;
    const size_t target_offset = static_cast<size_t>(token) * 2 * row_bytes;
    device.b.Copy(device.q, cat + target_offset, embed + source_offset,
                  row_bytes);
    device.b.Copy(device.q, cat + target_offset + row_bytes,
                  target + source_offset, row_bytes);
  }

  DBuf hidden = MatmulBf16D(device, concatenated.t(), weights_->fc);
  DBuf residual(device, ResidualDType(), {tokens, hidden_size});
  residual.Zero(device);
  const size_t layer_index =
      static_cast<size_t>(spec_step_idx % num_layers);
  if (weights_->kind == Qwen3_5MTPKind::kDense) {
    RunDenseLayer(device, weights_->dense_layers[layer_index], *config_, hidden,
                  residual, positions, tokens);
  } else {
    RunLayer(device, weights_->moe_layers[layer_index], *config_, hidden,
             residual, positions, tokens);
  }

  Tensor final_norm_weight =
      ResidentWeight(device, weights_->final_norm, {hidden_size});
  DBuf normalized(device, DType::kBF16, {tokens, hidden_size});
  vt::RmsNorm(device.q, normalized.t(), hidden.t(), final_norm_weight,
              vt::RmsNormArgs{eps, true}, &residual.t());

  Qwen3_5MTPHiddenStates out;
  out.tensor = normalized.t();
  const size_t allocation = normalized.alloc_bytes();
  void* storage = normalized.Release();
  out.storage = std::shared_ptr<void>(
      storage, [allocation](void* ptr) { Pool().Put(allocation, ptr); });
  return out;
}

ForwardLogits Qwen3_5MTPModel::ComputeLogits(
    const vt::Tensor& hidden_states, vt::Queue& queue) const {
  const int64_t hidden_size = config_->hidden_size;
  VT_CHECK(hidden_states.rank == 2 &&
               hidden_states.shape[1] == hidden_size &&
               hidden_states.dtype == DType::kBF16 &&
               hidden_states.IsContiguous() &&
               hidden_states.device == queue.device,
           "qwen3_5 MTP logits: hidden states must be contiguous bf16 [T,H] "
           "on the queue device");
  Dev device{vt::GetBackend(queue.device.type), queue};
  DBuf logits = (lm_head_fp4_ != nullptr && !lm_head_fp4_->Empty())
                    ? MatmulNvfp4F32D(device, hidden_states, *lm_head_fp4_)
                    : MatmulF32D(device, hidden_states, *lm_head_);
  return WrapDeviceLogits(device, std::move(logits), config_->vocab_size);
}

std::vector<float> Qwen3_5MTPModel::ForwardLogitsHost(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& positions,
    const vt::Tensor& target_hidden_states, vt::Queue& queue,
    int64_t spec_step_idx) const {
  Qwen3_5MTPHiddenStates hidden =
      Forward(input_ids, positions, target_hidden_states, queue, spec_step_idx);
  ForwardLogits logits = ComputeLogits(hidden.tensor, queue);
  std::vector<float> host(static_cast<size_t>(logits.rows) * logits.vocab);
  Backend& backend = vt::GetBackend(queue.device.type);
  backend.Copy(queue, host.data(), logits.device_tensor.data,
               host.size() * sizeof(float));
  backend.Synchronize(queue);
  return host;
}

// Shared shape/count validation for the dense paged forward entry points (the
// 27B analogue of CheckPagedForward). Same contract, over the dense weights.
static void CheckDensePagedForward(const std::vector<int32_t>& token_ids,
                                   const std::vector<int32_t>& positions,
                                   const CommonAttentionMetadata& attn_meta,
                                   const std::vector<PagedKvCache>& attn_kv,
                                   const std::vector<GdnStateCache>& gdn_state,
                                   const Qwen3_5DenseWeights& weights,
                                   const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "qwen3_5 dense paged forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 dense paged forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 dense paged forward: weights.layers size must equal "
           "num_hidden_layers");
  VT_CHECK(attn_meta.num_actual_tokens == T,
           "qwen3_5 dense paged forward: attn_meta.num_actual_tokens must equal T");
  int64_t n_full = 0, n_gdn = 0;
  for (const auto& l : weights.layers) (l.is_linear_attention ? n_gdn : n_full) += 1;
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == n_full,
           "qwen3_5 dense paged forward: attn_kv count must equal full-attn layers");
  VT_CHECK(static_cast<int64_t>(gdn_state.size()) == n_gdn,
           "qwen3_5 dense paged forward: gdn_state count must equal GDN layers");
}

// Dense embed (27B): hidden[T,H] bf16 = embed_tokens[token_ids] (device-resident
// table). The 27B analogue of EmbedInto — KEPT OUTSIDE THE CUDA-GRAPH for the
// same reasons (the Embedding op allocs a bounds-check flag + syncs, and it
// consumes host token_ids). The dense-graph driver runs this per step into its
// PERSISTENT hidden buffer, then captures/replays DenseForwardLayers over that
// fixed address. Text-only: the three mRoPE streams coincide so the partial NeoX
// RoPE degenerates to 1-D RoPE over `positions` (notes §2); vision tower DEFERRED.
static void DenseEmbedInto(Dev d, DBuf& hidden,
                           const std::vector<int32_t>& token_ids,
                           const Qwen3_5DenseWeights& weights,
                           const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());
}

// The CAPTURABLE dense paged forward region (27B): everything AFTER the embedding
// — the residual stream (res=0), the N paged dense decoder layers, the final
// RMSNorm and the bf16 lm_head — returning the [n_out,vocab] f32 logits as a
// device DBuf (NO host Download). The 27B analogue of ForwardLayers; split out so
// the exact op sequence is what the graph captures/replays. `hidden_in` is the
// embedded input (a view over the graph's persistent hidden buffer on the replay
// path); it is COPIED into a working buffer so the layers' in-place residual
// thread never disturbs the persistent embedding. All per-step-varying inputs are
// read from HOST vector args (positions / metadata), whose host->device copies
// are capturable on GB10; the driver keeps them persistent + mutates in place.
// Every per-call scratch is pool-backed (DevicePool) or resident/StreamScratch-
// pooled (the cutlass/emulation fp4 GEMMs, cublas lm_head) so a cold pre-warm at
// this size makes the capture region do ZERO cudaMalloc.
static DBuf DenseForwardLayers(Dev d, const Tensor& hidden_in,
                               const std::vector<int32_t>& positions,
                               const CommonAttentionMetadata& attn_meta,
                               const GDNAttentionMetadata& gdn_meta,
                               const std::vector<PagedKvCache>& attn_kv,
                               const std::vector<GdnStateCache>& gdn_state,
                               const Qwen3_5DenseWeights& weights,
                               const HfConfig& config,
                               const std::vector<int32_t>& logits_indices = {}) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy of the embedded hidden (device->device; captured). RunDenseLayer
  // Paged reassigns `hidden` per layer, so this must NOT alias the persistent buf.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, ResidualDType(), {T, H});
  res.Zero(d);

  // Per-step inputs uploaded ONCE (see StepDevInputs) — no per-layer re-upload.
  StepDevInputs sdi = BuildStepDevInputs(d, positions, attn_meta, gdn_meta);
  // Build the fused-preamble cos|sin cache ONCE; fp4_attn keys the per-arch
  // default (the real 27B W4A4 => ON; bf16/GGUF dense => OFF; env overrides).
  const bool fp4_attn = [&] {
    for (const auto& l : weights.layers)
      if (!l.is_linear_attention) return !l.attn.q_proj_fp4.Empty();
    return false;
  }();
  MaybeBuildAttnCosSin(d, sdi, config, T, fp4_attn);

  // N paged decoder layers: full-attn layers read/write attn_kv[fa_idx], GDN
  // layers the persistent gdn_state[gdn_idx] (same layer-order indexing as the
  // 35B paged forward).
  int64_t fa_idx = 0, gdn_idx = 0;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3_5DenseLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    const PagedKvCache* kv =
        layer.is_linear_attention ? nullptr : &attn_kv[static_cast<size_t>(fa_idx++)];
    const GdnStateCache* gs =
        layer.is_linear_attention ? &gdn_state[static_cast<size_t>(gdn_idx++)] : nullptr;
    RunDenseLayerPaged(d, layer, config, hidden, res, sdi, attn_meta,
                       gdn_meta, kv, gs, T);
  }

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

  // Logits gather-before-lm_head (prefill/mixed): same semantics as the 35B path.
  // lm_head is unquantized bf16 in the 27B (notes §3.6). Pure-decode / graph
  // replay pass empty indices (identity) → the full [T,vocab] path.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < T;
  if (do_gather) {
    const int64_t n_out = static_cast<int64_t>(logits_indices.size());
    DBuf dgather(d, DType::kBF16, {n_out, H});
    GatherRows(d, dgather.ptr(), dnorm.t(), logits_indices, H);
    return MatmulF32D(d, dgather.t(), weights.lm_head);
  }
  return MatmulF32D(d, dnorm.t(), weights.lm_head);
}

// Full eager dense paged forward body: embed (host token_ids) then the capturable
// dense layer region. Used by Qwen3_5DenseModel::Forward/ForwardDevice and the
// dense-graph driver's eager fallback / cold-shape pre-warm step (one contiguous
// stream, no capture). Returns [n_out,vocab] f32 (n_out == num_reqs when gathered,
// else T). Shared op sequence with the graph so eager output == replay output.
static DBuf DenseForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                             const std::vector<int32_t>& positions,
                             const CommonAttentionMetadata& attn_meta,
                             const GDNAttentionMetadata& gdn_meta,
                             const std::vector<PagedKvCache>& attn_kv,
                             const std::vector<GdnStateCache>& gdn_state,
                             const Qwen3_5DenseWeights& weights,
                             const HfConfig& config,
                             const std::vector<int32_t>& logits_indices) {
  CheckDensePagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state,
                         weights, config);
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  DBuf hidden(d, DType::kBF16, {T, H});
  DenseEmbedInto(d, hidden, token_ids, weights, config);
  return DenseForwardLayers(d, hidden.t(), positions, attn_meta, gdn_meta, attn_kv,
                            gdn_state, weights, config, logits_indices);
}

std::vector<float> Qwen3_5DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices);
  const int64_t n_out = dlogits.t().shape[0];
  std::vector<float> logits(static_cast<size_t>(n_out) * config.vocab_size);
  dlogits.Download(d, logits.data());
  return logits;
}

ForwardLogits Qwen3_5DenseModel::ForwardDevice(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf dlogits = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta,
                                  attn_kv, gdn_state, weights, config,
                                  logits_indices);
  return WrapDeviceLogits(d, std::move(dlogits), config.vocab_size);
}

std::vector<float> Qwen3_5ReplayLayer(const Qwen3_5MoeLayerWeights& layer,
                                      const HfConfig& config,
                                      const std::vector<float>& hidden_in,
                                      const std::vector<int32_t>& positions,
                                      int64_t seqlen, vt::Queue& queue) {
  const int64_t T = seqlen;
  const int64_t H = config.hidden_size;
  VT_CHECK(static_cast<int64_t>(hidden_in.size()) == T * H,
           "qwen3_5 replay: hidden_in must be [T*H]");
  Dev d{vt::GetBackend(queue.device.type), queue};

  // Seed the fused stream with the combined residual input: res = hidden_in,
  // hidden delta = 0. The layer's input_layernorm then normalizes hidden_in.
  DBuf res(d, DType::kF32, {T, H}, hidden_in.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  hidden.Zero(d);
  RunLayer(d, layer, config, hidden, res, positions, T);

  // Combined stream out = residual + hidden (f32), directly comparable to the
  // layer golden's `out`.
  std::vector<float> res_host(static_cast<size_t>(T) * H);
  res.Download(d, res_host.data());
  std::vector<uint16_t> hidden_host(static_cast<size_t>(T) * H);
  hidden.Download(d, hidden_host.data());
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = res_host[i] + vt::BF16ToF32(hidden_host[i]);
  return out;
}

std::vector<float> Qwen3_5ReplayDenseLayer(
    const Qwen3_5DenseLayerWeights& layer, const HfConfig& config,
    const std::vector<float>& hidden_in, const std::vector<int32_t>& positions,
    int64_t seqlen, vt::Queue& queue) {
  const int64_t T = seqlen;
  const int64_t H = config.hidden_size;
  VT_CHECK(static_cast<int64_t>(hidden_in.size()) == T * H,
           "qwen3_5 dense replay: hidden_in must be [T*H]");
  Dev d{vt::GetBackend(queue.device.type), queue};

  // Match Qwen3_5ReplayLayer's fused residual contract: the captured layer
  // input is the combined stream, so seed it as `res` and start the bf16 delta
  // at zero before running the real dense attention + SwiGLU layer.
  DBuf res(d, DType::kF32, {T, H}, hidden_in.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  hidden.Zero(d);
  RunDenseLayer(d, layer, config, hidden, res, positions, T);

  std::vector<float> res_host(static_cast<size_t>(T) * H);
  res.Download(d, res_host.data());
  std::vector<uint16_t> hidden_host(static_cast<size_t>(T) * H);
  hidden.Download(d, hidden_host.data());
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = res_host[i] + vt::BF16ToF32(hidden_host[i]);
  }
  return out;
}

// ─── Qwen3_5DecodeGraph (decode CUDA-graph driver) ──────────────────────────
namespace {

// Overwrite dst's CONTENTS from src WITHOUT changing dst.data() when the sizes
// already match (preserves the fixed address a captured host->device copy reads
// from); reallocate only when the shape actually changed.
template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}
template <typename T>
void CopyInPlace(std::optional<std::vector<T>>& dst,
                 const std::optional<std::vector<T>>& src) {
  if (!src.has_value()) {
    dst.reset();
    return;
  }
  if (!dst.has_value()) dst.emplace();
  CopyInPlace(*dst, *src);
}

// The SET of decode batch sizes we capture a graph for (mirrors vLLM's
// cudagraph_capture_sizes = [1,2,4] + range(8,…) capped to max_num_seqs;
// compilation/cuda_graph.py + config/compilation.py:678-690 @ e24d1b24). A real
// decode batch of B requests is PADDED up to the smallest captured size >= B and
// that size's graph is replayed; the padded rows are inert (see BuildPaddedDecode).
constexpr std::array<int64_t, 7> kDecodeGraphSizes = {1, 2, 4, 8, 16, 32, 64};
constexpr int64_t kMaxDecodeGraphBatch = 64;

// Smallest captured size >= b, or -1 when b exceeds the largest (caller runs
// eager). The result is CAPPED at `cap` (== max_num_seqs == the GDN state-cache
// slot count): a decode batch is never padded beyond max_num_seqs, mirroring
// vLLM's decode cudagraph dispatcher ("already caps batch sizes at max_num_seqs",
// compilation.py:1438-1444 @ e24d1b24). Without this cap a batch of B requests
// (B <= max_num_seqs) could pad up to the next fixed size (e.g. B=24 -> 32) and
// overrun the max_num_seqs-sized conv/ssm state cache, tripping the
// causal_conv1d_update `conv_state.shape[0] >= x.shape[0]` guard. cap is always
// >= b here (the caller only routes num_reqs <= max_num_seqs), so the clamped
// value still satisfies size >= b; `cap` itself becomes a captured size.
int64_t PadToCaptureSize(int64_t b, int64_t cap) {
  if (b > cap) return -1;  // never pad DOWN below the real batch (eager fallback)
  for (int64_t s : kDecodeGraphSizes)
    if (s >= b) return std::min(s, cap);
  return -1;
}

// Build the S-padded PURE-DECODE inputs from the real B-request step (B<=S). The
// decode forward is ROW-INDEPENDENT (paged attn is per-request causal, GDN
// recurrence is per-sequence, MoE/router/norm/lm_head are per-token — no cross-
// row reduction), so appending S-B INERT rows cannot perturb the real rows'
// logits. The padding rows are made inert exactly as vLLM's cudagraph padding:
//   * token id / position 0 (embed row is discarded);
//   * slot_mapping = -1  → ReshapeAndCache skips the KV write (cuda_cache.cu:50);
//   * gdn state index = -1 → causal_conv1d_update / GdnDecode skip the in-place
//     mamba/conv update (cuda_gdn.cu:153,471), so no real state slot is touched;
//   * seq_lens = 1 + block_table row 0 → PagedAttention does a valid in-bounds
//     read of block 0 whose output row is discarded (never returned to the caller).
// The real prefix [0,B) is copied verbatim, so at S==B this is a bit-identical
// rebuild of the eager inputs (the S==B graph output equals Forward exactly).
void BuildPaddedDecode(int64_t S, const std::vector<int32_t>& tok,
                       const std::vector<int32_t>& pos,
                       const v1::CommonAttentionMetadata& am,
                       const v1::GDNAttentionMetadata& gm,
                       std::vector<int32_t>& tok_out,
                       std::vector<int32_t>& pos_out,
                       v1::CommonAttentionMetadata& am_out,
                       v1::GDNAttentionMetadata& gm_out) {
  const int64_t B = static_cast<int64_t>(tok.size());
  const int64_t cols = am.block_table_num_cols;

  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());

  am_out = am;  // carries causal + block_table_num_cols + max_seq_len
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;  // pure decode
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(),
            am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  gm_out = gm;
  gm_out.num_prefills = 0;
  gm_out.num_prefill_tokens = 0;
  gm_out.num_decodes = static_cast<int>(S);
  gm_out.num_decode_tokens = static_cast<int>(S);
  gm_out.num_actual_tokens = static_cast<int>(S);
  {
    std::vector<int32_t> si(static_cast<size_t>(S), -1);  // inert padding slots
    if (gm.non_spec_state_indices_tensor.has_value())
      std::copy(gm.non_spec_state_indices_tensor->begin(),
                gm.non_spec_state_indices_tensor->end(), si.begin());
    gm_out.non_spec_state_indices_tensor = std::move(si);
  }
  {
    std::vector<int32_t> q(static_cast<size_t>(S + 1));
    for (int64_t i = 0; i <= S; ++i) q[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    gm_out.non_spec_query_start_loc = std::move(q);
  }
  // Pure decode (num_prefills==0): the prefill-only fields are unused.
  gm_out.has_initial_state.reset();
  gm_out.prefill_query_start_loc.reset();
  gm_out.prefill_state_indices.reset();
  gm_out.prefill_has_initial_state.reset();
  (void)B;
}

}  // namespace

struct Qwen3_5DecodeGraph::Impl {
  Impl(const Qwen3_5MoeWeights& w, const HfConfig& c, vt::Queue q,
       int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    const char* env = std::getenv("VLLM_CPP_CUDAGRAPH");
    const bool env_on = (env == nullptr) || std::string(env) != "0";
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = env_on && queue.device.type == vt::DeviceType::kCUDA &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      if (kv.second.graph != nullptr) b.DestroyGraph(kv.second.graph);
  }

  // One captured padded batch size. Owns its OWN persistent host inputs (the
  // captured graph's host->device copies bake these addresses, so each size needs
  // its own fixed-address buffers), its persistent embed target + logits output,
  // and its instantiated graph. The state machine per slot mirrors the original
  // single-shape driver: cold (eager pre-warm) → warm (capture+replay) → replay.
  struct SizeSlot {
    std::vector<int32_t> token_ids;   // [S]
    std::vector<int32_t> positions;   // [S]
    v1::CommonAttentionMetadata attn_meta;
    v1::GDNAttentionMetadata gdn_meta;
    std::unique_ptr<DBuf> hidden;     // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;     // [S,vocab] f32 held graph output
    void* graph = nullptr;            // instantiated cudaGraphExec (opaque)
    int fa_cols = -1;                 // captured block-table column count
    bool captured = false;
    bool warm = false;
    int64_t replays = 0;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const v1::CommonAttentionMetadata& am,
                 const v1::GDNAttentionMetadata& gm) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
      CopyInPlace(gdn_meta.non_spec_state_indices_tensor,
                  gm.non_spec_state_indices_tensor);
      CopyInPlace(gdn_meta.non_spec_query_start_loc, gm.non_spec_query_start_loc);
      CopyInPlace(gdn_meta.has_initial_state, gm.has_initial_state);
      CopyInPlace(gdn_meta.prefill_query_start_loc, gm.prefill_query_start_loc);
      CopyInPlace(gdn_meta.prefill_state_indices, gm.prefill_state_indices);
      CopyInPlace(gdn_meta.prefill_has_initial_state, gm.prefill_has_initial_state);
      gdn_meta.num_prefills = gm.num_prefills;
      gdn_meta.num_prefill_tokens = gm.num_prefill_tokens;
      gdn_meta.num_decodes = gm.num_decodes;
      gdn_meta.num_decode_tokens = gm.num_decode_tokens;
      gdn_meta.num_actual_tokens = gm.num_actual_tokens;
    }
  };

  const Qwen3_5MoeWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3_5DecodeGraph::Qwen3_5DecodeGraph(const Qwen3_5MoeWeights& weights,
                                       const HfConfig& config, vt::Queue queue,
                                       int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3_5DecodeGraph::~Qwen3_5DecodeGraph() = default;

bool Qwen3_5DecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3_5DecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3_5DecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state) {
  CheckPagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state,
                    impl_->weights, impl_->config);
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t B = static_cast<int64_t>(token_ids.size());
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // The step returns the [B, vocab] real-row logits ON DEVICE — NO full-logits
  // D2H / Synchronize here (removed: the per-step drain). The captured/warm paths
  // return a NON-owning view over the slot's persistent [S,vocab] logits (the
  // first B rows are the real requests; the padded rows follow). Stream ordering
  // guarantees the sampler's later reads see the replay's writes; the next
  // same-size replay overwrites the buffer, so in-place sampler mutation is safe.
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                          gdn_state, impl_->weights, impl_->config);
    // ForwardBody returns [B,vocab] (owned pool block; hand ownership out).
    return WrapDeviceLogits(d, std::move(lg), vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then
  // refresh THIS size's persistent host buffers in place.
  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  v1::CommonAttentionMetadata pam;
  v1::GDNAttentionMetadata pgm;
  BuildPaddedDecode(S, token_ids, positions, attn_meta, gdn_meta, ptok, ppos,
                    pam, pgm);

  // A block-table column-count change reallocates the persistent block_table (the
  // captured H2D copy's source address moves) → invalidate this slot's graph and
  // re-warm/re-capture.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam, pgm);
  s.fa_cols = cols;
  if (cols_changed && s.graph != nullptr) {
    b.DestroyGraph(s.graph);
    s.graph = nullptr;
    s.captured = false;
    s.warm = false;
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, then relaunch the captured layer region.
  if (s.captured) {
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.ReplayGraph(impl_->queue, s.graph);
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + residency were warmed for this size by the previous (eager)
  // step. CAPTURE the layer region once, instantiate the graph, then launch it.
  if (s.warm) {
    EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.BeginCapture(impl_->queue);
    DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                            s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                            impl_->config);
    s.graph = b.EndCaptureGraph(impl_->queue);
    s.logits = std::make_unique<DBuf>(std::move(lg));
    s.captured = true;
    impl_->any_captured = true;
    b.ReplayGraph(impl_->queue, s.graph);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool + resident weights /
  // fused-MoE constants for this size) and defer capture to the next same-size
  // step. This is a real decode step (its padded output's real rows are used;
  // nothing is wasted). (Re)allocate the persistent hidden buffer to this size.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, s.gdn_meta,
                          attn_kv, gdn_state, impl_->weights, impl_->config);
  s.warm = true;
  s.captured = false;
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), vocab);
  if (fl.rows != B) {
    fl.rows = B;
    fl.device_tensor =
        MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  }
  return fl;
}

// ─── Qwen3_5DenseDecodeGraph (27B dense decode CUDA-graph driver) ────────────
// The 27B DENSE sibling of Qwen3_5DecodeGraph. Same cold→warm→replay state
// machine, same padded-batch capture set (kDecodeGraphSizes), same persistent
// fixed-address host inputs and persistent embed/logits buffers — but it drives
// the dense forward (DenseForwardLayers over DenseEmbedInto) instead of the MoE
// forward. It reuses the weight-agnostic PadToCaptureSize / BuildPaddedDecode
// verbatim. The dense W4A4 projections' per-call scratch is already persistent
// (the cutlass StreamScratch pool / the emulation path's pool-backed DBufs / the
// cublas lm_head), so a cold pre-warm at each size makes the capture region do
// ZERO cudaMalloc — mirroring the MoE path's EnsureMoeScratch/EnsureCtmp/pool
// discipline. The 35B MoE graph is UNTOUCHED.
struct Qwen3_5DenseDecodeGraph::Impl {
  Impl(const Qwen3_5DenseWeights& w, const HfConfig& c, vt::Queue q,
       int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    const char* env = std::getenv("VLLM_CPP_CUDAGRAPH");
    const bool env_on = (env == nullptr) || std::string(env) != "0";
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = env_on && queue.device.type == vt::DeviceType::kCUDA &&
              b.SupportsGraphCapture();
  }
  ~Impl() {
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr, "[DenseDecodeGraph] dense-27B decode graph: %lld total "
                           "replays across %zu captured size(s)\n",
                   static_cast<long long>(replays), slots.size());
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots)
      if (kv.second.graph != nullptr) b.DestroyGraph(kv.second.graph);
  }

  // One captured padded batch size (mirror of Qwen3_5DenseDecodeGraph SizeSlot).
  struct SizeSlot {
    std::vector<int32_t> token_ids;   // [S]
    std::vector<int32_t> positions;   // [S]
    v1::CommonAttentionMetadata attn_meta;
    v1::GDNAttentionMetadata gdn_meta;
    std::unique_ptr<DBuf> hidden;     // [S,H] bf16 persistent embed target
    std::unique_ptr<DBuf> logits;     // [S,vocab] f32 held graph output
    void* graph = nullptr;            // instantiated cudaGraphExec (opaque)
    int fa_cols = -1;                 // captured block-table column count
    bool captured = false;
    bool warm = false;
    int64_t replays = 0;

    // In-place refresh of the persistent host inputs (fixed addresses once the
    // slot's vectors reach size S) so a replay re-reads this step's tokens.
    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const v1::CommonAttentionMetadata& am,
                 const v1::GDNAttentionMetadata& gm) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
      CopyInPlace(gdn_meta.non_spec_state_indices_tensor,
                  gm.non_spec_state_indices_tensor);
      CopyInPlace(gdn_meta.non_spec_query_start_loc, gm.non_spec_query_start_loc);
      CopyInPlace(gdn_meta.has_initial_state, gm.has_initial_state);
      CopyInPlace(gdn_meta.prefill_query_start_loc, gm.prefill_query_start_loc);
      CopyInPlace(gdn_meta.prefill_state_indices, gm.prefill_state_indices);
      CopyInPlace(gdn_meta.prefill_has_initial_state, gm.prefill_has_initial_state);
      gdn_meta.num_prefills = gm.num_prefills;
      gdn_meta.num_prefill_tokens = gm.num_prefill_tokens;
      gdn_meta.num_decodes = gm.num_decodes;
      gdn_meta.num_decode_tokens = gm.num_decode_tokens;
      gdn_meta.num_actual_tokens = gm.num_actual_tokens;
    }
  };

  const Qwen3_5DenseWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;  // == max_num_seqs; padded decode batch cap
  bool enabled = false;

  std::map<int64_t, SizeSlot> slots;  // padded size S -> slot
  int64_t replays = 0;                // total replays (diagnostics)
  bool any_captured = false;          // diagnostics: at least one live graph
};

Qwen3_5DenseDecodeGraph::Qwen3_5DenseDecodeGraph(const Qwen3_5DenseWeights& weights,
                                                 const HfConfig& config,
                                                 vt::Queue queue,
                                                 int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

Qwen3_5DenseDecodeGraph::~Qwen3_5DenseDecodeGraph() = default;

bool Qwen3_5DenseDecodeGraph::captured() const { return impl_->any_captured; }
int64_t Qwen3_5DenseDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits Qwen3_5DenseDecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state) {
  CheckDensePagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state,
                         impl_->weights, impl_->config);
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t B = static_cast<int64_t>(token_ids.size());
  const int64_t vocab = impl_->config.vocab_size;
  const int64_t H = impl_->config.hidden_size;

  // Returns the [B,vocab] real-row logits ON DEVICE (no D2H). The captured/warm
  // paths return a NON-owning view over the slot's persistent [S,vocab] logits
  // (first B rows are the real requests). Stream ordering guarantees the sampler
  // sees the replay's writes; the next same-size replay overwrites the buffer.
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = DenseForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                               gdn_state, impl_->weights, impl_->config, {});
    return WrapDeviceLogits(d, std::move(lg), vocab);
  }

  // Pad this step's real B-request inputs up to S (inert padding rows), then
  // refresh THIS size's persistent host buffers in place.
  Impl::SizeSlot& s = impl_->slots[S];
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  v1::CommonAttentionMetadata pam;
  v1::GDNAttentionMetadata pgm;
  BuildPaddedDecode(S, token_ids, positions, attn_meta, gdn_meta, ptok, ppos,
                    pam, pgm);

  // A block-table column-count change reallocates the persistent block_table (the
  // captured H2D copy's source address moves) → invalidate this slot's graph.
  const bool cols_changed = (s.fa_cols != -1 && s.fa_cols != cols);
  s.Refresh(ptok, ppos, pam, pgm);
  s.fa_cols = cols;
  if (cols_changed && s.graph != nullptr) {
    b.DestroyGraph(s.graph);
    s.graph = nullptr;
    s.captured = false;
    s.warm = false;
  }

  // Fast path: this size's graph is captured. Embed OUTSIDE the graph into the
  // persistent hidden buffer, then relaunch the captured dense layer region.
  if (s.captured) {
    DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.ReplayGraph(impl_->queue, s.graph);
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Warm: the pool + residency were warmed for this size by the previous (eager)
  // step. CAPTURE the dense layer region once, instantiate the graph, launch it.
  if (s.warm) {
    DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
    b.BeginCapture(impl_->queue);
    DBuf lg = DenseForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                                 s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                                 impl_->config);
    s.graph = b.EndCaptureGraph(impl_->queue);
    s.logits = std::make_unique<DBuf>(std::move(lg));
    s.captured = true;
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr, "[DenseDecodeGraph] captured dense-27B decode graph "
                           "for padded size S=%lld (real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    b.ReplayGraph(impl_->queue, s.graph);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, vocab);
  }

  // Cold size: run one EAGER step (pre-warms the DevicePool + resident weights /
  // fp4-GEMM StreamScratch pools for this size) and defer capture to the next
  // same-size step. This is a real decode step (its padded output's real rows are
  // used). (Re)allocate the persistent hidden buffer to this size.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  DenseEmbedInto(d, *s.hidden, s.token_ids, impl_->weights, impl_->config);
  DBuf lg = DenseForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta,
                               s.gdn_meta, attn_kv, gdn_state, impl_->weights,
                               impl_->config);
  s.warm = true;
  s.captured = false;
  // lg is [S,vocab]; hand ownership out but expose only the first B (real) rows.
  ForwardLogits fl = WrapDeviceLogits(d, std::move(lg), vocab);
  if (fl.rows != B) {
    fl.rows = B;
    fl.device_tensor =
        MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, vocab});
  }
  return fl;
}

}  // namespace vllm
