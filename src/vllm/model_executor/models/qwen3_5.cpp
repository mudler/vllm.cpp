// vllm.cpp original; see qwen3_5.h. Forward math mirrored 1:1 from the pinned
// upstream (qwen3_next.py::Qwen3NextDecoderLayer / Qwen3NextModel.forward,
// qwen_gdn_linear_attn.py, qwen3_next.py::Qwen3NextAttention /
// Qwen3NextSparseMoeBlock @ e24d1b24). References:
// .agents/qwen36-forward-notes.md (assembly, §2 mRoPE->NeoX, §5 attention),
// .agents/gdn-semantics.md (§1 layout, §6 g/beta prep, §7 recurrence),
// .agents/moe-semantics.md (§1-§6 MoE block + activated-expert gather).
#include "vllm/model_executor/models/qwen3_5.h"

#include "vllm/model_executor/models/qwen3_5_dense.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {
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
class DevicePool {
 public:
  void* Get(Backend& b, size_t bytes) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = free_.find(bytes);
      if (it != free_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        return p;
      }
    }
    return b.Alloc(bytes);
  }
  void Put(size_t bytes, void* p) {
    std::lock_guard<std::mutex> lk(mu_);
    free_[bytes].push_back(p);
  }

 private:
  std::mutex mu_;
  std::unordered_map<size_t, std::vector<void*>> free_;
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
  void* ptr() { return p_; }
  size_t bytes() const { return bytes_; }
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
  const int64_t N = w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  vt::Matmul(d.q, dout.t(), dx.t(), dw);
  std::vector<float> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

// y[M,N] bf16 = x[M,K] bf16 @ w[K,N] bf16 (bf16 output mirrors the model's bf16
// hidden states where the result feeds the residual stream / next matmul).
std::vector<uint16_t> MatmulBf16(Dev d, const std::vector<uint16_t>& x, int64_t M,
                                 int64_t K, const OwnedTensor& w) {
  const int64_t N = w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
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

DBuf MatmulF32D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  vt::Matmul(d.q, dout.t(), x, dw);
  return dout;
}

DBuf MatmulBf16D(Dev d, const Tensor& x, const OwnedTensor& w) {
  const int64_t M = x.shape[0], N = w.shape[1];
  Tensor dw = ResidentWeight(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
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

// y[M,N] = x[M,K] (bf16/f32 device) @ dequant(w).T via the lifted vLLM cutlass
// W8A8 fp8 GEMM: static per-tensor activation quant (vt::QuantFp8Static with the
// checkpoint input_scale) then vt::MatmulFp8Cutlass with the folded alpha
// (= input_scale·weight_scale). out dtype f32 (q/k/v, in_proj_qkv/z sinks) or
// bf16 (o/out_proj residual sinks). CUDA-only (sm120a; the 35B W8A8 path is
// CUDA-resident — fp8 fields are only populated on the CUDA load, VT_FP8_CUTLASS).
DBuf MatmulFp8CutlassD(Dev d, const Tensor& x, const Fp8Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  VT_CHECK(d.q.device.type == vt::DeviceType::kCUDA,
           "MatmulFp8CutlassD: the fp8 W8A8 cutlass path is CUDA-only");
  DBuf a_fp8(d, DType::kI8, {M, K});
  vt::QuantFp8Static(d.q, a_fp8.t(), x, w.input_scale);
  Tensor wdev = ResidentFp8(d, w);
  DBuf dout(d, out_dtype, {M, N});
  vt::MatmulFp8Cutlass(d.q, dout.t(), a_fp8.t(), wdev, w.alpha);
  return dout;
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

#ifdef VT_CUTLASS_NVFP4
// cutlass sm120a fp4xfp4 GEMM path toggle (opt-in; VT_NVFP4_CUTLASS=1). Routes the
// 27B true-W4A4 projections through vt::MatmulNvfp4Cutlass (the lifted vLLM
// near-peak kernel) instead of our emulation-grade MatmulNvfp4Fp4. Mirrors vLLM's
// actual kernel (notes §7.1). NOTE (measured 2026-07-05): swapping ONLY the GEMM
// to real cutlass does NOT recover vLLM's native token stream (198) — the 27B
// still yields the emulation stream (271); tok6 is a near-tie tipped by the
// aggregate of the non-fp4 forward numerics, not the fp4 GEMM. So this toggle is
// a THROUGHPUT lever (near-peak prefill), correctness-neutral (still reproduces
// vLLM's own emulation stream token-for-token). Only meaningful when the cutlass
// TU was compiled (VT_CUTLASS_NVFP4).
bool NvfpCutlassEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_CUTLASS");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
#endif

// TRUE W4A4 (fp4 activations x fp4 weights) device GEMM — the 27B path (notes §7).
// ScaledFp4Quant(x) -> per-token fp4 activations + fp8 block scales, then the
// fp4xfp4 GEMM with the folded alpha (= vllm cutlass_scaled_fp4_mm_sm120a). CUDA
// only; the CPU fp4 path uses the bf16-dequant fallback in the callers. out_dtype
// f32 or bf16.
DBuf MatmulNvfp4Fp4D(Dev d, const Tensor& x, const Nvfp4Weight& w, DType out_dtype) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
  DBuf a_packed(d, DType::kI8, {M, K / 2});
  DBuf a_scale(d, DType::kI8, {M, K / 16});
  vt::ScaledFp4Quant(d.q, a_packed.t(), a_scale.t(), x, w.input_global_scale_inv);
  Nvfp4Dev dw = ResidentNvfp4(d, w);
  DBuf dout(d, out_dtype, {M, N});
#ifdef VT_CUTLASS_NVFP4
  if (NvfpCutlassEnabled()) {
    // Swizzle the per-token activation scale into the cutlass atom layout (the
    // weight scale is swizzled once, cached). Then the lifted sm120a fp4xfp4 GEMM.
    auto round_up = [](int64_t v, int64_t y) { return (v + y - 1) / y * y; };
    const int64_t Mp = round_up(M, 128), Kp = round_up(K / 16, 4);
    DBuf a_sf_sw(d, DType::kI8, {Mp, Kp});  // SwizzleBlockscale zero-fills padding
    vt::SwizzleBlockscale(d.q, a_sf_sw.t(), a_scale.t());
    Tensor b_sf_sw = ResidentNvfp4ScaleSwizzled(d, w);
    vt::MatmulNvfp4Cutlass(d.q, dout.t(), a_packed.t(), a_sf_sw.t(), dw.packed, b_sf_sw, w.alpha);
    return dout;
  }
#endif
  vt::MatmulNvfp4Fp4(d.q, dout.t(), a_packed.t(), a_scale.t(), dw.packed, dw.scale, w.alpha);
  return dout;
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

// --- GDN (linear_attention) block. gdn-semantics.md §1 (layout), §6 (g/beta),
// §7 (recurrence); qwen_gdn_linear_attn.py forward. Device-resident (M2.5
// Phase 1): h [T,H] bf16 (device) -> DBuf [T,H] bf16 (device); no host round-
// trips (the g/beta prep + conv split are device ops, not host loops).
DBuf GdnBlock(Dev d, const GdnLayerWeights& w, const HfConfig& cfg,
              const Tensor& h, int64_t T) {
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
  // W8A8 cutlass fp8 (35B) when populated, else bf16 (default / GGUF).
  DBuf mixed = !w.in_proj_qkv_fp8.Empty()
                   ? MatmulFp8CutlassD(d, h, w.in_proj_qkv_fp8, DType::kF32)
                   : MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  DBuf z = !w.in_proj_z_fp8.Empty() ? MatmulFp8CutlassD(d, h, w.in_proj_z_fp8, DType::kF32)
                                    : MatmulF32D(d, h, w.in_proj_z);  // [T,value_dim]
  DBuf braw = MatmulF32D(d, h, w.in_proj_b);      // [T,Hv]
  DBuf araw = MatmulF32D(d, h, w.in_proj_a);      // [T,Hv]

  // Causal conv1d over the token stream (silu activation), fresh zero state.
  Tensor dcw = ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dstate(d, DType::kF32, {1, conv_dim, Kw - 1});
  dstate.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  DBuf dconv(d, DType::kF32, {T, conv_dim});
  vt::CausalConv1dFwd(d.q, dconv.t(), mixed.t(), dcw, nullptr, dstate.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});

  // Split conv output into q[T,Hk,Dk] | k[T,Hk,Dk] | v[T,Hv,Dv] (device op).
  DBuf qf(d, DType::kF32, {T, Hk, Dk});
  DBuf kf(d, DType::kF32, {T, Hk, Dk});
  DBuf vf(d, DType::kF32, {T, Hv, Dv});
  Tensor q2 = Reshape(qf.t(), {T, key_dim});
  Tensor k2 = Reshape(kf.t(), {T, key_dim});
  Tensor v2 = Reshape(vf.t(), {T, value_dim});
  vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());

  // g/beta from a/b + A_log/dt_bias (gdn-semantics.md §6) — device op.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  DBuf g(d, DType::kF32, {T, Hv});
  DBuf beta(d, DType::kF32, {T, Hv});
  vt::GdnGBeta(d.q, g.t(), beta.t(), araw.t(), braw.t(), a_log_dev, dt_bias_dev);

  // L2-normalize q,k over Dk (gdn-semantics.md §4), then the gated-delta-rule
  // recurrence (§7). scale = Dk^-0.5, applied to q only inside the op.
  DBuf dql2(d, DType::kF32, {T, Hk, Dk});
  DBuf dkl2(d, DType::kF32, {T, Hk, Dk});
  vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
  vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
  DBuf dssm(d, DType::kF32, {1, Hv, Dv, Dk});
  dssm.Zero(d);
  DBuf dcore(d, DType::kF32, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  vt::GdnPrefill(d.q, dcore.t(), dql2.t(), dkl2.t(), vf.t(), g.t(), beta.t(),
                 dssm.t(), dqsl.t(), vt::GdnArgs{scale});

  // Gated RMSNorm over Dv with the z gate (gdn-semantics.md §5), viewing the
  // core output and z as [T*Hv, Dv]; cast to bf16, flatten heads, out-project.
  Tensor dnw = ResidentWeightF32(d, w.norm_weight, {Dv});
  DBuf dgated(d, DType::kF32, {T * Hv, Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = Reshape(z.t(), {T * Hv, Dv});
  vt::RmsNormGated(d.q, dgated.t(), core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
  DBuf gated_bf16(d, DType::kBF16, {T, value_dim});
  vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
  // W8A8 cutlass fp8 (35B) when populated, else fp4-resident W4A4 (27B, notes
  // §3.6), else bf16 (default / GGUF).
  return !w.out_proj_fp8.Empty()
             ? MatmulFp8CutlassD(d, gated_bf16.t(), w.out_proj_fp8, DType::kBF16)
         : !w.out_proj_fp4.Empty()
             ? MatmulNvfp4Bf16D(d, gated_bf16.t(), w.out_proj_fp4)
             : MatmulBf16D(d, gated_bf16.t(), w.out_proj);  // [T,H]
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
                   const Tensor& h, const GDNAttentionMetadata& meta,
                   const GdnStateCache& state, int64_t T) {
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
  // W8A8 cutlass fp8 (35B) when populated, else bf16 (default / GGUF).
  DBuf mixed = !w.in_proj_qkv_fp8.Empty()
                   ? MatmulFp8CutlassD(d, h, w.in_proj_qkv_fp8, DType::kF32)
                   : MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  DBuf z = !w.in_proj_z_fp8.Empty() ? MatmulFp8CutlassD(d, h, w.in_proj_z_fp8, DType::kF32)
                                    : MatmulF32D(d, h, w.in_proj_z);  // [T,value_dim]
  DBuf braw = MatmulF32D(d, h, w.in_proj_b);      // [T,Hv]
  DBuf araw = MatmulF32D(d, h, w.in_proj_a);      // [T,Hv]

  // Causal conv1d over the token stream, PERSISTENT conv_state (gathered by the
  // per-request state indices, updated in place, scattered back).
  Tensor dcw = ResidentWeightF32(d, w.conv1d_weight, {conv_dim, Kw});
  DBuf dconv(d, DType::kF32, {T, conv_dim});
  const int64_t conv_row_elems = conv_dim * (Kw - 1);
  if (np > 0) {
    // Any prefill: conv over the WHOLE non-spec stream (decodes lead, each with
    // has_initial_state=1). qwen_gdn_linear_attn.py:1360-1375.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    const auto& qsl_full = *meta.non_spec_query_start_loc;
    const auto& his_u8 = *meta.has_initial_state;
    const int64_t nreq = static_cast<int64_t>(sidx.size());
    DBuf dcs(d, DType::kF32, {nreq, conv_dim, Kw - 1});
    GatherRows(d, dcs.ptr(), state.conv_state, sidx, conv_row_elems);
    std::vector<int32_t> his(his_u8.begin(), his_u8.end());
    DBuf dqsl(d, DType::kI32, {nreq + 1}, qsl_full.data());
    DBuf dhis(d, DType::kI32, {nreq}, his.data());
    vt::CausalConv1dFwd(d.q, dconv.t(), mixed.t(), dcw, nullptr, dcs.t(),
                        dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
    ScatterRows(d, state.conv_state, dcs.ptr(), sidx, conv_row_elems);
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
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    DBuf didx_dev(d, DType::kI32, {nd}, sidx.data());
    Tensor conv_cache = state.conv_state;  // mutable view over the shared buffer
    vt::CausalConv1dUpdate(d.q, dconv.t(), mixed.t(), dcw, nullptr, conv_cache,
                           vt::CausalConv1dArgs{true}, &didx_dev.t());
  }

  // Split conv output into q[T,Hk,Dk] | k[T,Hk,Dk] | v[T,Hv,Dv] (device op).
  DBuf qf(d, DType::kF32, {T, Hk, Dk});
  DBuf kf(d, DType::kF32, {T, Hk, Dk});
  DBuf vf(d, DType::kF32, {T, Hv, Dv});
  Tensor q2 = Reshape(qf.t(), {T, key_dim});
  Tensor k2 = Reshape(kf.t(), {T, key_dim});
  Tensor v2 = Reshape(vf.t(), {T, value_dim});
  vt::GdnConvSplit(d.q, q2, k2, v2, dconv.t());

  // g/beta from a/b + A_log/dt_bias (gdn-semantics.md §6) — device op. Uniform
  // over all tokens; the recurrence below segments them.
  Tensor a_log_dev = ResidentWeight(d, w.a_log, {Hv});
  Tensor dt_bias_dev = ResidentWeight(d, w.dt_bias, {Hv});
  DBuf dg(d, DType::kF32, {T, Hv});
  DBuf dbeta(d, DType::kF32, {T, Hv});
  vt::GdnGBeta(d.q, dg.t(), dbeta.t(), araw.t(), braw.t(), a_log_dev, dt_bias_dev);

  // L2-normalize q,k over Dk (gdn-semantics.md §4), scale = Dk^-0.5 (q only).
  DBuf dql2(d, DType::kF32, {T, Hk, Dk});
  DBuf dkl2(d, DType::kF32, {T, Hk, Dk});
  vt::L2Norm(d.q, dql2.t(), qf.t(), vt::L2NormArgs{1e-6F});
  vt::L2Norm(d.q, dkl2.t(), kf.t(), vt::L2NormArgs{1e-6F});
  DBuf dcore(d, DType::kF32, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  const int64_t ssm_row_elems = Hv * Dv * Dk;

  // Recurrence — decode segment first (leading nd_tok tokens), then prefill.
  if (nd > 0) {
    // Decode recurrence IN PLACE on the persistent ssm_state at each sequence's
    // slot (mirrors fla fused_recurrent ssm_state_indices) — no per-request
    // gather+scatter (the other two host<->device copies per sequence per layer).
    // Persistent metadata source (see the conv branch) for graph-replay safety.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    DBuf didx_dev(d, DType::kI32, {nd}, sidx.data());
    Tensor ssm_cache = state.ssm_state;  // mutable view over the shared buffer
    Tensor q_dec = SubView(dql2.t(), 0, nd_tok);
    Tensor k_dec = SubView(dkl2.t(), 0, nd_tok);
    Tensor v_dec = SubView(vf.t(), 0, nd_tok);
    Tensor g_dec = SubView(dg.t(), 0, nd_tok);
    Tensor b_dec = SubView(dbeta.t(), 0, nd_tok);
    Tensor o_dec = SubView(dcore.t(), 0, nd_tok);
    vt::GdnDecode(d.q, o_dec, q_dec, k_dec, v_dec, g_dec, b_dec, ssm_cache,
                  vt::GdnArgs{scale}, &didx_dev.t());
  }
  if (np > 0) {
    const auto& pidx = *meta.prefill_state_indices;
    const auto& p_his = *meta.prefill_has_initial_state;
    const auto& p_qsl = *meta.prefill_query_start_loc;
    DBuf dss(d, DType::kF32, {np, Hv, Dv, Dk});
    GatherRows(d, dss.ptr(), state.ssm_state, pidx, ssm_row_elems);
    // ⚠ GDN-STATE ZEROING (M1.6 caller obligation, qwen_gdn_linear_attn.py:1514):
    // vt::GdnPrefill reads `state` unconditionally, so zero the gathered rows
    // for fresh requests (prefill_has_initial_state==0) — else a fresh request
    // reads a stale mamba block.
    const size_t rb = static_cast<size_t>(ssm_row_elems) * sizeof(float);
    for (size_t s = 0; s < p_his.size(); ++s)
      if (p_his[s] == 0)
        d.b.Memset(d.q, static_cast<char*>(dss.ptr()) + s * rb, 0, rb);
    DBuf dpqsl(d, DType::kI32, {np + 1}, p_qsl.data());
    Tensor q_pre = SubView(dql2.t(), nd_tok, np_tok);
    Tensor k_pre = SubView(dkl2.t(), nd_tok, np_tok);
    Tensor v_pre = SubView(vf.t(), nd_tok, np_tok);
    Tensor g_pre = SubView(dg.t(), nd_tok, np_tok);
    Tensor b_pre = SubView(dbeta.t(), nd_tok, np_tok);
    Tensor o_pre = SubView(dcore.t(), nd_tok, np_tok);
    vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre, dss.t(),
                   dpqsl.t(), vt::GdnArgs{scale});
    ScatterRows(d, state.ssm_state, dss.ptr(), pidx, ssm_row_elems);
  }

  // Gated RMSNorm over Dv with the z gate, cast bf16, flatten heads, out-project.
  Tensor dnw = ResidentWeightF32(d, w.norm_weight, {Dv});
  DBuf dgated(d, DType::kF32, {T * Hv, Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  Tensor z2 = Reshape(z.t(), {T * Hv, Dv});
  vt::RmsNormGated(d.q, dgated.t(), core2, z2, dnw, vt::RmsNormGatedArgs{eps, false});
  DBuf gated_bf16(d, DType::kBF16, {T, value_dim});
  vt::CastBf16(d.q, gated_bf16.t(), dgated.t());
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
                   int64_t T) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  // fp4-resident W4A4 q/k/v/o (real 27B, notes §3.6) when populated; else bf16
  // (35B FP8-dequant path / synthetic). Exactly one representation is filled.
  const bool fp4 = !w.q_proj_fp4.Empty();
  const bool fp8 = !w.q_proj_fp8.Empty();  // 35B W8A8 cutlass (mutually exclusive)
  DBuf qgate = fp8 ? MatmulFp8CutlassD(d, h, w.q_proj_fp8, DType::kF32)
               : fp4 ? MatmulNvfp4F32D(d, h, w.q_proj_fp4)
                     : MatmulF32D(d, h, w.q_proj);  // [T, 2*Hq*Dh]
  DBuf kf = fp8 ? MatmulFp8CutlassD(d, h, w.k_proj_fp8, DType::kF32)
            : fp4 ? MatmulNvfp4F32D(d, h, w.k_proj_fp4)
                  : MatmulF32D(d, h, w.k_proj);     // [T, Hkv*Dh]
  DBuf vf = fp8 ? MatmulFp8CutlassD(d, h, w.v_proj_fp8, DType::kF32)
            : fp4 ? MatmulNvfp4F32D(d, h, w.v_proj_fp4)
                  : MatmulF32D(d, h, w.v_proj);     // [T, Hkv*Dh]

  // Gate split: per q-head the projection lays out [q(Dh) | gate(Dh)] (§5) —
  // device op producing q[T,Hq,Dh] and gate[T,Hq,Dh].
  DBuf qf(d, DType::kF32, {T, Hq, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate.t());

  // Per-head gemma-RMSNorm over Dh, then partial NeoX RoPE on positions[0].
  DBuf dqn(d, DType::kF32, {T * Hq, Dh});
  Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
  vt::RmsNorm(d.q, dqn.t(), Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
  DBuf dkn(d, DType::kF32, {T * Hkv, Dh});
  Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
  vt::RmsNorm(d.q, dkn.t(), Reshape(kf.t(), {T * Hkv, Dh}), dkw, vt::RmsNormArgs{eps, true});

  DBuf dpos(d, DType::kI32, {T}, positions.data());
  Tensor qn3 = Reshape(dqn.t(), {T, Hq, Dh});
  Tensor kn3 = Reshape(dkn.t(), {T, Hkv, Dh});
  vt::RopeNeox(d.q, qn3, kn3, dpos.t(), vt::RopeArgs{base, rot});

  // Causal GQA scaled-dot-product attention, scale = Dh^-0.5.
  Tensor v3 = Reshape(vf.t(), {T, Hkv, Dh});
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
                        const Tensor& h, const std::vector<int32_t>& positions,
                        const CommonAttentionMetadata& meta, const PagedKvCache& kv,
                        int64_t T) {
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  VT_CHECK(kv.dtype == DType::kF32, "full-attn paged: KV cache must be f32 (T0)");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "full-attn paged: KV cache head dims mismatch config");

  // fp4-resident W4A4 q/k/v/o (real 27B, notes §3.6) when populated; else bf16
  // (35B FP8-dequant path / synthetic). Exactly one representation is filled.
  const bool fp4 = !w.q_proj_fp4.Empty();
  const bool fp8 = !w.q_proj_fp8.Empty();  // 35B W8A8 cutlass (mutually exclusive)
  DBuf qgate = fp8 ? MatmulFp8CutlassD(d, h, w.q_proj_fp8, DType::kF32)
               : fp4 ? MatmulNvfp4F32D(d, h, w.q_proj_fp4)
                     : MatmulF32D(d, h, w.q_proj);  // [T, 2*Hq*Dh]
  DBuf kf = fp8 ? MatmulFp8CutlassD(d, h, w.k_proj_fp8, DType::kF32)
            : fp4 ? MatmulNvfp4F32D(d, h, w.k_proj_fp4)
                  : MatmulF32D(d, h, w.k_proj);     // [T, Hkv*Dh]
  DBuf vf = fp8 ? MatmulFp8CutlassD(d, h, w.v_proj_fp8, DType::kF32)
            : fp4 ? MatmulNvfp4F32D(d, h, w.v_proj_fp4)
                  : MatmulF32D(d, h, w.v_proj);     // [T, Hkv*Dh]

  // Gate split: per q-head [q(Dh) | gate(Dh)] (§5) — device op.
  DBuf qf(d, DType::kF32, {T, Hq, Dh});
  DBuf gatef(d, DType::kF32, {T, Hq, Dh});
  vt::AttnGateSplit(d.q, qf.t(), gatef.t(), qgate.t());

  // Per-head gemma-RMSNorm over Dh, then partial NeoX RoPE on positions.
  DBuf dqn(d, DType::kF32, {T * Hq, Dh});
  Tensor dqw = ResidentWeightF32(d, w.q_norm, {Dh});
  vt::RmsNorm(d.q, dqn.t(), Reshape(qf.t(), {T * Hq, Dh}), dqw, vt::RmsNormArgs{eps, true});
  DBuf dkn(d, DType::kF32, {T * Hkv, Dh});
  Tensor dkw = ResidentWeightF32(d, w.k_norm, {Dh});
  vt::RmsNorm(d.q, dkn.t(), Reshape(kf.t(), {T * Hkv, Dh}), dkw, vt::RmsNormArgs{eps, true});

  DBuf dpos(d, DType::kI32, {T}, positions.data());
  Tensor qn3 = Reshape(dqn.t(), {T, Hq, Dh});
  Tensor kn3 = Reshape(dkn.t(), {T, Hkv, Dh});
  vt::RopeNeox(d.q, qn3, kn3, dpos.t(), vt::RopeArgs{base, rot});

  Tensor v3 = Reshape(vf.t(), {T, Hkv, Dh});

  // Write the new K/V into the paged cache, then read K/V from the cache.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  DBuf dslot(d, DType::kI64, {T}, meta.slot_mapping.data());
  DBuf dblk(d, DType::kI32, {meta.num_reqs, meta.block_table_num_cols},
            meta.block_table_tensor.data());
  DBuf dsl(d, DType::kI32, {meta.num_reqs}, meta.seq_lens.data());
  DBuf dqsl(d, DType::kI32, {meta.num_reqs + 1}, meta.query_start_loc.data());
  vt::ReshapeAndCache(d.q, kn3, v3, k_cache, v_cache, dslot.t());

  DBuf dattn(d, DType::kF32, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  vt::PagedAttention(d.q, dattn.t(), qn3, k_cache, v_cache, dblk.t(), dsl.t(),
                     dqsl.t(), vt::PagedAttentionArgs{scale, meta.causal});

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
DBuf SharedExpert(Dev d, const MoeBlockWeights& w, const HfConfig& cfg,
                  const Tensor& h, int64_t T, bool fp4) {
  const int64_t H = cfg.hidden_size;
  const int64_t Is = cfg.shared_expert_intermediate_size;
  DBuf sg = fp4 ? MatmulNvfp4F32D(d, h, w.shared_gate_proj_fp4)
                : MatmulF32D(d, h, w.shared_gate_proj);  // [T,Is]
  DBuf su = fp4 ? MatmulNvfp4F32D(d, h, w.shared_up_proj_fp4)
                : MatmulF32D(d, h, w.shared_up_proj);    // [T,Is]
  DBuf sact(d, DType::kBF16, {T, Is});
  vt::MoeSiluMul(d.q, sact.t(), sg.t(), su.t());  // silu(sg) * su -> bf16
  DBuf sd = fp4 ? MatmulNvfp4F32D(d, sact.t(), w.shared_down_proj_fp4)
                : MatmulF32D(d, sact.t(), w.shared_down_proj);  // [T,H] f32
  DBuf gl = MatmulF32D(d, h, w.shared_gate);                    // [T,1] f32
  DBuf shared(d, DType::kBF16, {T, H});
  vt::SharedExpertGate(d.q, shared.t(), sd.t(), gl.t());  // sigmoid(gl)*sd -> bf16
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
  // all device-resident (no host round-trip).
  DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
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

  mr.w_gate = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
  mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wg_i32 * 4);
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_gate = d.b.Alloc(static_cast<size_t>(E) * sg_b);
  mr.s_up = d.b.Alloc(static_cast<size_t>(E) * sg_b);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_gate = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
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

  std::vector<float> gg(E), gu(E), gd(E);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    Nvfp4Dev g = ResidentNvfp4(d, w.expert_gate_fp4[se]);
    Nvfp4Dev u = ResidentNvfp4(d, w.expert_up_fp4[se]);
    Nvfp4Dev dn = ResidentNvfp4(d, w.expert_down_fp4[se]);
    auto* wg = static_cast<uint32_t*>(mr.w_gate) + se * wg_i32;
    auto* wu = static_cast<uint32_t*>(mr.w_up) + se * wg_i32;
    auto* wd = static_cast<uint32_t*>(mr.w_down) + se * wd_i32;
    auto* sgp = static_cast<uint8_t*>(mr.s_gate) + se * sg_b;
    auto* sup = static_cast<uint8_t*>(mr.s_up) + se * sg_b;
    auto* sdp = static_cast<uint8_t*>(mr.s_down) + se * sd_b;
    const auto* pg = static_cast<const uint8_t*>(g.packed.data);
    const auto* pu = static_cast<const uint8_t*>(u.packed.data);
    const auto* pd = static_cast<const uint8_t*>(dn.packed.data);
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wg, pg, K, N);
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wu, pu, K, N);
    vt::cuda::MarlinRepackExpertWeight(stream, d.q.device.index, wd, pd, N, K);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(g.scale.data), sgp, K, N,
                                        sf_gu);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(u.scale.data), sup, K, N,
                                        sf_gu);
    vt::cuda::MarlinProcessExpertScales(stream, static_cast<const uint8_t*>(dn.scale.data), sdp, N, K,
                                        sf_dn);
    gg[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_gate_fp4[se].scale2, sf_gu);
    gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_up_fp4[se].scale2, sf_gu);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.expert_down_fp4[se].scale2, sf_dn);
  }
  d.b.Copy(d.q, mr.g_gate, gg.data(), gg.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // repack done → safe to free fp4 originals

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
// 3 grouped GEMMs are moe_wna16_marlin_gemm over the resident repacked experts.
// Marlin's per-pair output layout (row = t*top_k+k) matches dgate/dup/ddown, so
// MoeSiluMul + MoeCombine are unchanged. Per-pair equivalent to the wmma path
// (same weight-only fp4 dequant), so token-for-token identical.
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

  Tensor wg = MakeTensor(mr.w_gate, DType::kI32, d.q.device, {E, H / 16, I * 2});
  Tensor wu = MakeTensor(mr.w_up, DType::kI32, d.q.device, {E, H / 16, I * 2});
  Tensor wd = MakeTensor(mr.w_down, DType::kI32, d.q.device, {E, I / 16, H * 2});
  Tensor sg = MakeTensor(mr.s_gate, DType::kI8, d.q.device, {E, H / 16, I});
  Tensor su = MakeTensor(mr.s_up, DType::kI8, d.q.device, {E, H / 16, I});
  Tensor sd = MakeTensor(mr.s_down, DType::kI8, d.q.device, {E, I / 16, H});
  Tensor gg = MakeTensor(mr.g_gate, DType::kF32, d.q.device, {E});
  Tensor gu = MakeTensor(mr.g_up, DType::kF32, d.q.device, {E});
  Tensor gd = MakeTensor(mr.g_down, DType::kF32, d.q.device, {E});
  Tensor ws = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  const int bi = block, tki = static_cast<int>(top_k);
  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int Pi = static_cast<int>(P);

  DBuf dgate(d, DType::kBF16, {P, I});
  DBuf dup_out(d, DType::kBF16, {P, I});
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(d.q, dgate.t(), dh, wg, sg, gg, ws, sorted_ids.t(), expert_ids.t(),
                                num_pad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(d.q, dup_out.t(), dh, wu, su, gu, ws, sorted_ids.t(),
                                expert_ids.t(), num_pad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, tki, Ti, Ii, Hi, false});
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());

  DBuf ddown(d, DType::kBF16, {P, H});
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  vt::MoeGroupedGemmNvfp4Marlin(d.q, ddown.t(), dact.t(), wd, sd, gd, ws, sorted_ids.t(),
                                expert_ids.t(), num_pad.t(), dtw.t(),
                                vt::MoeMarlinArgs{bi, 1, Pi, Hi, Ii, false});
  Tensor expert_out = Reshape(ddown.t(), {T, top_k, H});

  DBuf shared = SharedExpert(d, w, cfg, dh, T, true);
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), expert_out, dtw.t(), &shared.t());
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

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  // Qwen3NextRMSNorm == GemmaRMSNorm (weight applied as 1+w). res += hidden.
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf attn = layer.is_linear_attention
                  ? GdnBlock(d, layer.gdn, cfg, dhn.t(), T)
                  : FullAttnBlock(d, layer.attn, cfg, dhn.t(), positions, T);

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
  DBuf gate = fp4 ? MatmulNvfp4F32D(d, dh, w.gate_proj_fp4)
                  : MatmulF32D(d, dh, w.gate_proj);  // [T,I] f32
  DBuf up = fp4 ? MatmulNvfp4F32D(d, dh, w.up_proj_fp4)
                : MatmulF32D(d, dh, w.up_proj);      // [T,I] f32
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
                   DBuf& hidden, DBuf& res, const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& attn_meta,
                   const GDNAttentionMetadata& gdn_meta,
                   const PagedKvCache* attn_kv, const GdnStateCache* gdn_state,
                   int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  Tensor dw_in = ResidentWeight(d, layer.input_layernorm, {H});
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), hidden.t(), dw_in, vt::RmsNormArgs{eps, true}, &res.t());

  DBuf attn = [&] {
    if (layer.is_linear_attention) {
      VT_CHECK(gdn_state != nullptr, "paged layer: GDN layer needs a GdnStateCache");
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), gdn_meta, *gdn_state, T);
    }
    VT_CHECK(attn_kv != nullptr, "paged layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), positions, attn_meta,
                              *attn_kv, T);
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
                        const std::vector<int32_t>& positions,
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
      return GdnBlockPaged(d, layer.gdn, cfg, dhn.t(), gdn_meta, *gdn_state, T);
    }
    VT_CHECK(attn_kv != nullptr,
             "paged dense layer: full-attn layer needs a PagedKvCache");
    return FullAttnBlockPaged(d, layer.attn, cfg, dhn.t(), positions, attn_meta,
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
                          const Qwen3_5MoeWeights& weights, const HfConfig& config) {
  const int64_t T = hidden_in.shape[0];
  const int64_t H = config.hidden_size;
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Working copy of the embedded hidden (device->device; captured). RunLayerPaged
  // reassigns `hidden` per layer, so this must NOT alias the persistent buffer.
  DBuf hidden(d, DType::kBF16, {T, H});
  d.b.Copy(d.q, hidden.ptr(), hidden_in.data,
           static_cast<size_t>(T) * static_cast<size_t>(H) * vt::SizeOf(DType::kBF16));

  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);

  int64_t fa_idx = 0, gdn_idx = 0;
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const Qwen3_5MoeLayerWeights& layer = weights.layers[static_cast<size_t>(l)];
    const PagedKvCache* kv =
        layer.is_linear_attention ? nullptr : &attn_kv[static_cast<size_t>(fa_idx++)];
    const GdnStateCache* gs =
        layer.is_linear_attention ? &gdn_state[static_cast<size_t>(gdn_idx++)] : nullptr;
    RunLayerPaged(d, layer, config, hidden, res, positions, attn_meta, gdn_meta,
                  kv, gs, T);
  }

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  Tensor dfn = ResidentWeight(d, weights.final_norm, {H});
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), hidden.t(), dfn, vt::RmsNormArgs{eps, true}, &res.t());

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
                        const Qwen3_5MoeWeights& weights, const HfConfig& config) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  DBuf hidden(d, DType::kBF16, {T, H});
  EmbedInto(d, hidden, token_ids, weights, config);
  return ForwardLayers(d, hidden.t(), positions, attn_meta, gdn_meta, attn_kv,
                       gdn_state, weights, config);
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

void Qwen3_5Model::PrepareMarlinResident(const Qwen3_5MoeWeights& weights,
                                         const HfConfig& config, vt::Queue& queue) {
#ifdef VT_MARLIN_NVFP4
  if (queue.device.type != vt::DeviceType::kCUDA || !MarlinMoeEnabled()) return;
  Dev d{vt::GetBackend(queue.device.type), queue};
  for (const auto& layer : weights.layers) {
    const MoeBlockWeights& moe = layer.moe;
    if (!moe.expert_gate_fp4.empty())
      BuildMoeMarlinResident(d, moe, config, MoeMarlinResidentFor(&moe));
    if (!moe.shared_gate_proj_fp4.Empty())
      BuildMarlinDenseResident(d, moe.shared_gate_proj_fp4,
                               MarlinDenseResidentFor(&moe.shared_gate_proj_fp4));
    if (!moe.shared_up_proj_fp4.Empty())
      BuildMarlinDenseResident(d, moe.shared_up_proj_fp4,
                               MarlinDenseResidentFor(&moe.shared_up_proj_fp4));
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
    const HfConfig& config, vt::Queue& queue) {
  CheckPagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state, weights,
                    config);
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf dlogits = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                             gdn_state, weights, config);
  std::vector<float> logits(static_cast<size_t>(T) * config.vocab_size);
  dlogits.Download(d, logits.data());  // the ONE host Download of the forward.
  return logits;
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

  DBuf res(d, DType::kF32, {T, H});
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

  DBuf res(d, DType::kF32, {T, H});
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

std::vector<float> Qwen3_5DenseModel::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state,
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue) {
  // Same shape/count contract as Qwen3_5Model::Forward (CheckPagedForward), over
  // the dense weights. One PagedKvCache per full-attn layer, one GdnStateCache
  // per GDN layer, in layer order.
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
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

  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  // Text-only: the three mRoPE streams coincide so partial NeoX RoPE degenerates
  // to 1-D RoPE over `positions` (notes §2); the vision tower is DEFERRED.
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);

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
    RunDenseLayerPaged(d, layer, config, hidden, res, positions, attn_meta,
                       gdn_meta, kv, gs, T);
  }

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

// The SHAPE key a captured decode graph is valid for. Beyond num_reqs (== T for
// decode) it includes the block-table column count and the GDN mamba-state
// indices, because both are BAKED into the captured op stream (the block-table
// copy size / the per-row GDN gather offsets). Any change re-captures.
struct DecodeShapeKey {
  int64_t T = -1;
  int fa_cols = -1;
  std::vector<int32_t> gdn_state_indices;
  bool operator==(const DecodeShapeKey& o) const {
    return T == o.T && fa_cols == o.fa_cols &&
           gdn_state_indices == o.gdn_state_indices;
  }
};

DecodeShapeKey MakeShapeKey(const std::vector<int32_t>& token_ids,
                            const v1::CommonAttentionMetadata& attn_meta,
                            const v1::GDNAttentionMetadata& gdn_meta) {
  DecodeShapeKey k;
  k.T = static_cast<int64_t>(token_ids.size());
  k.fa_cols = attn_meta.block_table_num_cols;
  if (gdn_meta.non_spec_state_indices_tensor.has_value())
    k.gdn_state_indices = *gdn_meta.non_spec_state_indices_tensor;
  return k;
}

}  // namespace

struct Qwen3_5DecodeGraph::Impl {
  Impl(const Qwen3_5MoeWeights& w, const HfConfig& c, vt::Queue q)
      : weights(w), config(c), queue(q) {
    const char* env = std::getenv("VLLM_CPP_CUDAGRAPH");
    const bool env_on = (env == nullptr) || std::string(env) != "0";
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = env_on && queue.device.type == vt::DeviceType::kCUDA &&
              b.SupportsGraphCapture();
  }

  const Qwen3_5MoeWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  bool enabled = false;

  // Persistent host inputs (fixed addresses across replays of a shape).
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  v1::CommonAttentionMetadata attn_meta;
  v1::GDNAttentionMetadata gdn_meta;

  std::unique_ptr<DBuf> logits;  // held graph-output buffer (not pool-returned)
  std::unique_ptr<DBuf> hidden;  // persistent embed target (fixed address; the
                                 // captured layer region reads it, embedding
                                 // writes it OUTSIDE the graph each step)

  bool captured = false;
  DecodeShapeKey cap_key;
  DecodeShapeKey warm_key;
  bool warm_valid = false;
  int64_t replays = 0;

  // Copy this step's inputs into the persistent host buffers IN PLACE (same
  // addresses on an unchanged shape) so a replay re-reads the new token.
  void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
               const v1::CommonAttentionMetadata& am,
               const v1::GDNAttentionMetadata& gm) {
    CopyInPlace(token_ids, tok);
    CopyInPlace(positions, pos);
    // Attention metadata: in-place the H2D-copied vectors, assign the scalars.
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
    // GDN metadata: in-place the (optional) index/offset vectors, copy scalars.
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

Qwen3_5DecodeGraph::Qwen3_5DecodeGraph(const Qwen3_5MoeWeights& weights,
                                       const HfConfig& config, vt::Queue queue)
    : impl_(std::make_unique<Impl>(weights, config, queue)) {}

Qwen3_5DecodeGraph::~Qwen3_5DecodeGraph() = default;

bool Qwen3_5DecodeGraph::captured() const { return impl_->captured; }
int64_t Qwen3_5DecodeGraph::replay_count() const { return impl_->replays; }

std::vector<float> Qwen3_5DecodeGraph::Step(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state) {
  CheckPagedForward(token_ids, positions, attn_meta, attn_kv, gdn_state,
                    impl_->weights, impl_->config);
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t vocab = impl_->config.vocab_size;
  std::vector<float> out(static_cast<size_t>(T) * static_cast<size_t>(vocab));

  // Eager fallback (capture disabled / non-CUDA): identical to Forward.
  if (!impl_->enabled) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, gdn_meta, attn_kv,
                          gdn_state, impl_->weights, impl_->config);
    lg.Download(d, out.data());
    return out;
  }

  // Update the persistent host inputs (in place) to this step's token.
  impl_->Refresh(token_ids, positions, attn_meta, gdn_meta);
  const DecodeShapeKey key = MakeShapeKey(token_ids, attn_meta, gdn_meta);
  const int64_t H = impl_->config.hidden_size;

  // Fast path: a graph is captured for this shape. Embed OUTSIDE the graph into
  // the persistent hidden buffer, then relaunch the captured layer region.
  if (impl_->captured && key == impl_->cap_key) {
    EmbedInto(d, *impl_->hidden, impl_->token_ids, impl_->weights, impl_->config);
    b.Replay(impl_->queue);
    ++impl_->replays;
    impl_->logits->Download(d, out.data());
    return out;
  }

  // The pool + residency were warmed for this shape by the previous (eager)
  // step: CAPTURE the layer region once, then launch it. Capture RECORDS (does
  // not execute); the Replay is the single real execution that advances the
  // caches. Embedding runs OUTSIDE the capture region (it mallocs/syncs a
  // bounds-check flag and consumes host token_ids).
  if (impl_->warm_valid && key == impl_->warm_key) {
    EmbedInto(d, *impl_->hidden, impl_->token_ids, impl_->weights, impl_->config);
    b.BeginCapture(impl_->queue);
    DBuf lg = ForwardLayers(d, impl_->hidden->t(), impl_->positions,
                            impl_->attn_meta, impl_->gdn_meta, attn_kv, gdn_state,
                            impl_->weights, impl_->config);
    b.EndCapture(impl_->queue);
    impl_->logits = std::make_unique<DBuf>(std::move(lg));
    impl_->captured = true;
    impl_->cap_key = key;
    b.Replay(impl_->queue);
    impl_->replays = 1;
    impl_->logits->Download(d, out.data());
    return out;
  }

  // Cold shape: run one EAGER step (pre-warms the DevicePool + resident weights /
  // fused-MoE constants for this shape) and defer capture to the next step. This
  // is a real decode step (its output is used); nothing is wasted. (Re)allocate
  // the persistent hidden buffer to this shape's T so the next-step capture bakes
  // its fixed address.
  impl_->hidden = std::make_unique<DBuf>(d, DType::kBF16,
                                         std::vector<int64_t>{T, H});
  EmbedInto(d, *impl_->hidden, impl_->token_ids, impl_->weights, impl_->config);
  {
    DBuf lg = ForwardLayers(d, impl_->hidden->t(), impl_->positions,
                            impl_->attn_meta, impl_->gdn_meta, attn_kv, gdn_state,
                            impl_->weights, impl_->config);
    lg.Download(d, out.data());
  }
  impl_->warm_key = key;
  impl_->warm_valid = true;
  impl_->captured = false;  // any prior-shape graph is now stale
  return out;
}

}  // namespace vllm
