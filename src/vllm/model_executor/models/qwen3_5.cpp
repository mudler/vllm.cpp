// vllm.cpp original; see qwen3_5.h. Forward math mirrored 1:1 from the pinned
// upstream (qwen3_next.py::Qwen3NextDecoderLayer / Qwen3NextModel.forward,
// qwen_gdn_linear_attn.py, qwen3_next.py::Qwen3NextAttention /
// Qwen3NextSparseMoeBlock @ e24d1b24). References:
// .agents/qwen36-forward-notes.md (assembly, §2 mRoPE->NeoX, §5 attention),
// .agents/gdn-semantics.md (§1 layout, §6 g/beta prep, §7 recurrence),
// .agents/moe-semantics.md (§1-§6 MoE block + activated-expert gather).
#include "vllm/model_executor/models/qwen3_5.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

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

DBuf MatmulNvfp4F32D(Dev d, const Tensor& x, const Nvfp4Weight& w) {
  const int64_t M = x.shape[0], K = x.shape[1], N = w.n;
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
  DBuf mixed = MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  DBuf z = MatmulF32D(d, h, w.in_proj_z);         // [T,value_dim]
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
  return MatmulBf16D(d, gated_bf16.t(), w.out_proj);  // [T,H]
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
  DBuf mixed = MatmulF32D(d, h, w.in_proj_qkv);  // [T,conv_dim]
  DBuf z = MatmulF32D(d, h, w.in_proj_z);         // [T,value_dim]
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
    // Pure decode: single-token conv step per sequence.
    // qwen_gdn_linear_attn.py:1376-1388.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    std::vector<int32_t> didx(sidx.begin(), sidx.begin() + nd);
    DBuf dcs(d, DType::kF32, {nd, conv_dim, Kw - 1});
    GatherRows(d, dcs.ptr(), state.conv_state, didx, conv_row_elems);
    vt::CausalConv1dUpdate(d.q, dconv.t(), mixed.t(), dcw, nullptr, dcs.t(),
                           vt::CausalConv1dArgs{true});
    ScatterRows(d, state.conv_state, dcs.ptr(), didx, conv_row_elems);
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
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    std::vector<int32_t> didx(sidx.begin(), sidx.begin() + nd);
    DBuf dss(d, DType::kF32, {nd, Hv, Dv, Dk});
    GatherRows(d, dss.ptr(), state.ssm_state, didx, ssm_row_elems);
    Tensor q_dec = SubView(dql2.t(), 0, nd_tok);
    Tensor k_dec = SubView(dkl2.t(), 0, nd_tok);
    Tensor v_dec = SubView(vf.t(), 0, nd_tok);
    Tensor g_dec = SubView(dg.t(), 0, nd_tok);
    Tensor b_dec = SubView(dbeta.t(), 0, nd_tok);
    Tensor o_dec = SubView(dcore.t(), 0, nd_tok);
    vt::GdnDecode(d.q, o_dec, q_dec, k_dec, v_dec, g_dec, b_dec, dss.t(),
                  vt::GdnArgs{scale});
    ScatterRows(d, state.ssm_state, dss.ptr(), didx, ssm_row_elems);
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
  return MatmulBf16D(d, gated_bf16.t(), w.out_proj);  // [T,H]
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

  DBuf qgate = MatmulF32D(d, h, w.q_proj);  // [T, 2*Hq*Dh]
  DBuf kf = MatmulF32D(d, h, w.k_proj);     // [T, Hkv*Dh]
  DBuf vf = MatmulF32D(d, h, w.v_proj);     // [T, Hkv*Dh]

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
  return MatmulBf16D(d, gated.t(), w.o_proj);  // [T,H]
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

  DBuf qgate = MatmulF32D(d, h, w.q_proj);  // [T, 2*Hq*Dh]
  DBuf kf = MatmulF32D(d, h, w.k_proj);     // [T, Hkv*Dh]
  DBuf vf = MatmulF32D(d, h, w.v_proj);     // [T, Hkv*Dh]

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
  return MatmulBf16D(d, gated.t(), w.o_proj);  // [T,H]
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

  // Per-expert fp4-resident device pointer arrays (packed/scale/scale2) for the
  // gate/up/down projections, gathered once (residency is uploaded lazily + kept
  // in each weight's shared_ptr — after warm-up these Get()s are pointer reads).
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
  DBuf dgp(d, DType::kI64, {E}, gp.data()), dgs(d, DType::kI64, {E}, gs.data());
  DBuf dup(d, DType::kI64, {E}, up.data()), dus(d, DType::kI64, {E}, us.data());
  DBuf ddp(d, DType::kI64, {E}, dp.data()), dds(d, DType::kI64, {E}, ds.data());
  DBuf dg2(d, DType::kF32, {E}, g2.data()), du2(d, DType::kF32, {E}, u2.data());
  DBuf dd2(d, DType::kF32, {E}, d2.data());

  // Gate/up read the token hidden: pair p -> token p/top_k.
  std::vector<int32_t> tok_map(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p) tok_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / top_k);
  DBuf dtok(d, DType::kI32, {P}, tok_map.data());

  // Grouped gate/up GEMM over all pairs (one launch each), silu-mul, grouped
  // down GEMM (act = per-pair silu output, identity row-map). expert_out lands
  // as [T,top_k,H] contiguous — exactly what MoeCombine consumes.
  DBuf dgate(d, DType::kF32, {P, I});
  DBuf dup_out(d, DType::kF32, {P, I});
  vt::MoeGroupedGemmNvfp4(d.q, dgate.t(), dh, eids, &dtok.t(), dgp.t(), dgs.t(), dg2.t());
  vt::MoeGroupedGemmNvfp4(d.q, dup_out.t(), dh, eids, &dtok.t(), dup.t(), dus.t(), du2.t());
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeSiluMul(d.q, dact.t(), dgate.t(), dup_out.t());
  DBuf ddown(d, DType::kBF16, {P, H});
  vt::MoeGroupedGemmNvfp4(d.q, ddown.t(), dact.t(), eids, nullptr, ddp.t(), dds.t(), dd2.t());
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
  if (fp4 && d.q.device.type == vt::DeviceType::kCUDA)
    return MoeBlockFusedCuda(d, w, cfg, dh, T);

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

}  // namespace

std::vector<float> Qwen3_5Model::Forward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const CommonAttentionMetadata& attn_meta, const GDNAttentionMetadata& gdn_meta,
    const std::vector<PagedKvCache>& attn_kv,
    const std::vector<GdnStateCache>& gdn_state, const Qwen3_5MoeWeights& weights,
    const HfConfig& config, vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(T > 0, "qwen3_5 paged forward: empty token_ids");
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "qwen3_5 paged forward: positions length must equal token count");
  VT_CHECK(static_cast<int64_t>(weights.layers.size()) == config.num_hidden_layers,
           "qwen3_5 paged forward: weights.layers size must equal num_hidden_layers");
  VT_CHECK(attn_meta.num_actual_tokens == T,
           "qwen3_5 paged forward: attn_meta.num_actual_tokens must equal T");
  Dev d{vt::GetBackend(queue.device.type), queue};
  const float eps = static_cast<float>(config.rms_norm_eps);

  // Count the KV groups' layers so the per-layer cache indexing is checkable.
  int64_t n_full = 0, n_gdn = 0;
  for (const auto& l : weights.layers)
    (l.is_linear_attention ? n_gdn : n_full) += 1;
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == n_full,
           "qwen3_5 paged forward: attn_kv count must equal full-attn layer count");
  VT_CHECK(static_cast<int64_t>(gdn_state.size()) == n_gdn,
           "qwen3_5 paged forward: gdn_state count must equal GDN layer count");

  // Embed: hidden = embed_tokens[token_ids] (bf16, device-resident). res = 0.
  Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf hidden(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, hidden.t(), dtab, dids.t());

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

  // lm_head: fp4-resident (M2.2b) when populated, else the bf16/GGUF path. This
  // is the ONE host Download of the whole forward (the [T,vocab] logits).
  DBuf dlogits = weights.lm_head_fp4.Empty()
                     ? MatmulF32D(d, dnorm.t(), weights.lm_head)
                     : MatmulNvfp4F32D(d, dnorm.t(), weights.lm_head_fp4);
  std::vector<float> logits(static_cast<size_t>(T) * vocab);
  dlogits.Download(d, logits.data());
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

}  // namespace vllm
