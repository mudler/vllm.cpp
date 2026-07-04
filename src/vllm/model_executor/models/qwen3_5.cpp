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

// Owned device allocation + tensor view. On CPU the backend's Alloc/Copy are
// malloc/memcpy; on CUDA they are cudaMalloc / h2d-d2h on the queue's stream.
class DBuf {
 public:
  DBuf(Dev d, DType dt, const std::vector<int64_t>& shape,
       const void* host = nullptr)
      : b_(&d.b) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_->Alloc(bytes_ == 0 ? 1 : bytes_);
    t_ = MakeTensor(p_, dt, d.q.device, shape);
    if (host != nullptr) b_->Copy(d.q, p_, host, bytes_);
  }
  ~DBuf() { b_->Free(p_); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;

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
  Tensor t_;
};

// A view over an owned host weight tensor (already in Matmul-B layout [in,out]).
DBuf WeightBuf(Dev d, const OwnedTensor& w) {
  std::vector<int64_t> shape(w.shape, w.shape + w.rank);
  return DBuf(d, w.dtype, shape, w.bytes.data());
}

float SizeF(int64_t n) { return static_cast<float>(n); }
float Silu(float x) { return x / (1.0F + std::exp(-x)); }
float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

// y[M,N] f32 = x[M,K] bf16 @ w[K,N] (w owns [K,N] bf16). f32 output keeps the
// GEMM's f32 accumulation for the f32 glue that consumes it.
std::vector<float> MatmulF32(Dev d, const std::vector<uint16_t>& x, int64_t M,
                             int64_t K, const OwnedTensor& w) {
  const int64_t N = w.shape[1];
  DBuf dx(d, DType::kBF16, {M, K}, x.data());
  DBuf dw = WeightBuf(d, w);
  DBuf dout(d, DType::kF32, {M, N});
  vt::Matmul(d.q, dout.t(), dx.t(), dw.t());
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
  DBuf dw = WeightBuf(d, w);
  DBuf dout(d, DType::kBF16, {M, N});
  vt::Matmul(d.q, dout.t(), dx.t(), dw.t());
  std::vector<uint16_t> out(static_cast<size_t>(M) * N);
  dout.Download(d, out.data());
  return out;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& x) {
  std::vector<uint16_t> out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = vt::F32ToBF16(x[i]);
  return out;
}

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
// §7 (recurrence); qwen_gdn_linear_attn.py forward. h [T*H] bf16 -> [T*H] bf16.
std::vector<uint16_t> GdnBlock(Dev d, const GdnLayerWeights& w,
                               const HfConfig& cfg,
                               const std::vector<uint16_t>& h, int64_t T) {
  const int64_t H = cfg.hidden_size;
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
  std::vector<float> mixed = MatmulF32(d, h, T, H, w.in_proj_qkv);  // [T,conv_dim]
  std::vector<float> z = MatmulF32(d, h, T, H, w.in_proj_z);        // [T,value_dim]
  std::vector<float> braw = MatmulF32(d, h, T, H, w.in_proj_b);     // [T,Hv]
  std::vector<float> araw = MatmulF32(d, h, T, H, w.in_proj_a);     // [T,Hv]

  // Causal conv1d over the token stream (silu activation), fresh zero state.
  DBuf dmixed(d, DType::kF32, {T, conv_dim}, mixed.data());
  std::vector<float> cw = WeightF32(w.conv1d_weight);  // match f32 x on CUDA
  DBuf dcw(d, DType::kF32, {conv_dim, Kw}, cw.data());
  DBuf dstate(d, DType::kF32, {1, conv_dim, Kw - 1});
  dstate.Zero(d);
  const int32_t qsl[2] = {0, static_cast<int32_t>(T)};
  const int32_t his[1] = {0};
  DBuf dqsl(d, DType::kI32, {2}, qsl);
  DBuf dhis(d, DType::kI32, {1}, his);
  DBuf dconv(d, DType::kF32, {T, conv_dim});
  vt::CausalConv1dFwd(d.q, dconv.t(), dmixed.t(), dcw.t(), nullptr, dstate.t(),
                      dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
  std::vector<float> conv(static_cast<size_t>(T) * conv_dim);
  dconv.Download(d, conv.data());

  // Split conv output into q[T,Hk,Dk] | k[T,Hk,Dk] | v[T,Hv,Dv].
  std::vector<float> qf(static_cast<size_t>(T) * key_dim);
  std::vector<float> kf(static_cast<size_t>(T) * key_dim);
  std::vector<float> vf(static_cast<size_t>(T) * value_dim);
  for (int64_t t = 0; t < T; ++t) {
    const float* row = conv.data() + static_cast<size_t>(t) * conv_dim;
    std::memcpy(qf.data() + static_cast<size_t>(t) * key_dim, row,
                static_cast<size_t>(key_dim) * sizeof(float));
    std::memcpy(kf.data() + static_cast<size_t>(t) * key_dim, row + key_dim,
                static_cast<size_t>(key_dim) * sizeof(float));
    std::memcpy(vf.data() + static_cast<size_t>(t) * value_dim, row + 2 * key_dim,
                static_cast<size_t>(value_dim) * sizeof(float));
  }

  // g/beta from a/b + A_log/dt_bias (gdn-semantics.md §6). softplus stabilized
  // at threshold 20; g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b).
  const float* a_log = reinterpret_cast<const float*>(w.a_log.bytes.data());
  const float* dt_bias = reinterpret_cast<const float*>(w.dt_bias.bytes.data());
  std::vector<float> g(static_cast<size_t>(T) * Hv);
  std::vector<float> beta(static_cast<size_t>(T) * Hv);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t hv = 0; hv < Hv; ++hv) {
      const size_t idx = static_cast<size_t>(t) * Hv + hv;
      const float x = araw[idx] + dt_bias[hv];
      const float sp = x > 20.0F ? x : std::log1p(std::exp(x));
      g[idx] = -std::exp(a_log[hv]) * sp;
      beta[idx] = Sigmoid(braw[idx]);
    }
  }

  // L2-normalize q,k over Dk (gdn-semantics.md §4), then the gated-delta-rule
  // recurrence (§7). scale = Dk^-0.5, applied to q only inside the op.
  DBuf dq(d, DType::kF32, {T, Hk, Dk}, qf.data());
  DBuf dk(d, DType::kF32, {T, Hk, Dk}, kf.data());
  DBuf dql2(d, DType::kF32, {T, Hk, Dk});
  DBuf dkl2(d, DType::kF32, {T, Hk, Dk});
  vt::L2Norm(d.q, dql2.t(), dq.t(), vt::L2NormArgs{1e-6F});
  vt::L2Norm(d.q, dkl2.t(), dk.t(), vt::L2NormArgs{1e-6F});
  DBuf dv(d, DType::kF32, {T, Hv, Dv}, vf.data());
  DBuf dg(d, DType::kF32, {T, Hv}, g.data());
  DBuf dbeta(d, DType::kF32, {T, Hv}, beta.data());
  DBuf dssm(d, DType::kF32, {1, Hv, Dv, Dk});
  dssm.Zero(d);
  DBuf dcore(d, DType::kF32, {T, Hv, Dv});
  const float scale = 1.0F / std::sqrt(SizeF(Dk));
  vt::GdnPrefill(d.q, dcore.t(), dql2.t(), dkl2.t(), dv.t(), dg.t(), dbeta.t(),
                 dssm.t(), dqsl.t(), vt::GdnArgs{scale});

  // Gated RMSNorm over Dv with the z gate (gdn-semantics.md §5), viewing the
  // core output and z as [T*Hv, Dv]; then flatten heads and out-project.
  DBuf dz(d, DType::kF32, {T * Hv, Dv}, z.data());
  std::vector<float> nw = WeightF32(w.norm_weight);  // match f32 x on CUDA
  DBuf dnw(d, DType::kF32, {Dv}, nw.data());
  DBuf dgated(d, DType::kF32, {T * Hv, Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  vt::RmsNormGated(d.q, dgated.t(), core2, dz.t(), dnw.t(),
                   vt::RmsNormGatedArgs{eps, false});
  std::vector<float> gated(static_cast<size_t>(T) * value_dim);
  dgated.Download(d, gated.data());
  std::vector<uint16_t> gated_bf16 = ToBf16(gated);
  return MatmulBf16(d, gated_bf16, T, value_dim, w.out_proj);  // [T,H]
}

// --- Batched PAGED GDN block (M1.8 Task 3). Same conv1d + l2norm + q/k/v/g/beta
// prep + gated-norm + out_proj as GdnBlock, but driven by the batched
// GDNAttentionMetadata segmentation over the PERSISTENT ssm_state/conv_state:
// leading num_decode_tokens are decode (vt::GdnDecode + causal_conv1d_update),
// the rest prefill (vt::GdnPrefill + causal_conv1d_fn). Mirrors
// qwen_gdn_linear_attn.py::_forward_core @ e24d1b24 (conv split L1360-1388;
// recurrence split L1480-1559; the ssm gather+ZERO L1513-1514, scatter L1532).
// h [T*H] bf16 -> [T*H] bf16.
std::vector<uint16_t> GdnBlockPaged(Dev d, const GdnLayerWeights& w,
                                    const HfConfig& cfg,
                                    const std::vector<uint16_t>& h,
                                    const GDNAttentionMetadata& meta,
                                    const GdnStateCache& state, int64_t T) {
  const int64_t H = cfg.hidden_size;
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
  std::vector<float> mixed = MatmulF32(d, h, T, H, w.in_proj_qkv);  // [T,conv_dim]
  std::vector<float> z = MatmulF32(d, h, T, H, w.in_proj_z);        // [T,value_dim]
  std::vector<float> braw = MatmulF32(d, h, T, H, w.in_proj_b);     // [T,Hv]
  std::vector<float> araw = MatmulF32(d, h, T, H, w.in_proj_a);     // [T,Hv]

  // Causal conv1d over the token stream, PERSISTENT conv_state (gathered by the
  // per-request state indices, updated in place, scattered back).
  DBuf dmixed(d, DType::kF32, {T, conv_dim}, mixed.data());
  std::vector<float> cw = WeightF32(w.conv1d_weight);
  DBuf dcw(d, DType::kF32, {conv_dim, Kw}, cw.data());
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
    vt::CausalConv1dFwd(d.q, dconv.t(), dmixed.t(), dcw.t(), nullptr, dcs.t(),
                        dqsl.t(), dhis.t(), vt::CausalConv1dArgs{true});
    ScatterRows(d, state.conv_state, dcs.ptr(), sidx, conv_row_elems);
  } else {
    // Pure decode: single-token conv step per sequence.
    // qwen_gdn_linear_attn.py:1376-1388.
    const auto& sidx = *meta.non_spec_state_indices_tensor;
    std::vector<int32_t> didx(sidx.begin(), sidx.begin() + nd);
    DBuf dcs(d, DType::kF32, {nd, conv_dim, Kw - 1});
    GatherRows(d, dcs.ptr(), state.conv_state, didx, conv_row_elems);
    vt::CausalConv1dUpdate(d.q, dconv.t(), dmixed.t(), dcw.t(), nullptr, dcs.t(),
                           vt::CausalConv1dArgs{true});
    ScatterRows(d, state.conv_state, dcs.ptr(), didx, conv_row_elems);
  }
  std::vector<float> conv(static_cast<size_t>(T) * conv_dim);
  dconv.Download(d, conv.data());

  // Split conv output into q[T,Hk,Dk] | k[T,Hk,Dk] | v[T,Hv,Dv].
  std::vector<float> qf(static_cast<size_t>(T) * key_dim);
  std::vector<float> kf(static_cast<size_t>(T) * key_dim);
  std::vector<float> vf(static_cast<size_t>(T) * value_dim);
  for (int64_t t = 0; t < T; ++t) {
    const float* row = conv.data() + static_cast<size_t>(t) * conv_dim;
    std::memcpy(qf.data() + static_cast<size_t>(t) * key_dim, row,
                static_cast<size_t>(key_dim) * sizeof(float));
    std::memcpy(kf.data() + static_cast<size_t>(t) * key_dim, row + key_dim,
                static_cast<size_t>(key_dim) * sizeof(float));
    std::memcpy(vf.data() + static_cast<size_t>(t) * value_dim, row + 2 * key_dim,
                static_cast<size_t>(value_dim) * sizeof(float));
  }

  // g/beta from a/b + A_log/dt_bias (gdn-semantics.md §6). Uniform over all
  // tokens; the recurrence below segments them.
  const float* a_log = reinterpret_cast<const float*>(w.a_log.bytes.data());
  const float* dt_bias = reinterpret_cast<const float*>(w.dt_bias.bytes.data());
  std::vector<float> g(static_cast<size_t>(T) * Hv);
  std::vector<float> beta(static_cast<size_t>(T) * Hv);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t hv = 0; hv < Hv; ++hv) {
      const size_t idx = static_cast<size_t>(t) * Hv + hv;
      const float x = araw[idx] + dt_bias[hv];
      const float sp = x > 20.0F ? x : std::log1p(std::exp(x));
      g[idx] = -std::exp(a_log[hv]) * sp;
      beta[idx] = Sigmoid(braw[idx]);
    }
  }

  // L2-normalize q,k over Dk (gdn-semantics.md §4), scale = Dk^-0.5 (q only).
  DBuf dq(d, DType::kF32, {T, Hk, Dk}, qf.data());
  DBuf dk(d, DType::kF32, {T, Hk, Dk}, kf.data());
  DBuf dql2(d, DType::kF32, {T, Hk, Dk});
  DBuf dkl2(d, DType::kF32, {T, Hk, Dk});
  vt::L2Norm(d.q, dql2.t(), dq.t(), vt::L2NormArgs{1e-6F});
  vt::L2Norm(d.q, dkl2.t(), dk.t(), vt::L2NormArgs{1e-6F});
  DBuf dv(d, DType::kF32, {T, Hv, Dv}, vf.data());
  DBuf dg(d, DType::kF32, {T, Hv}, g.data());
  DBuf dbeta(d, DType::kF32, {T, Hv}, beta.data());
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
    Tensor v_dec = SubView(dv.t(), 0, nd_tok);
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
    Tensor v_pre = SubView(dv.t(), nd_tok, np_tok);
    Tensor g_pre = SubView(dg.t(), nd_tok, np_tok);
    Tensor b_pre = SubView(dbeta.t(), nd_tok, np_tok);
    Tensor o_pre = SubView(dcore.t(), nd_tok, np_tok);
    vt::GdnPrefill(d.q, o_pre, q_pre, k_pre, v_pre, g_pre, b_pre, dss.t(),
                   dpqsl.t(), vt::GdnArgs{scale});
    ScatterRows(d, state.ssm_state, dss.ptr(), pidx, ssm_row_elems);
  }

  // Gated RMSNorm over Dv with the z gate, flatten heads, out-project.
  DBuf dz(d, DType::kF32, {T * Hv, Dv}, z.data());
  std::vector<float> nw = WeightF32(w.norm_weight);
  DBuf dnw(d, DType::kF32, {Dv}, nw.data());
  DBuf dgated(d, DType::kF32, {T * Hv, Dv});
  Tensor core2 = Reshape(dcore.t(), {T * Hv, Dv});
  vt::RmsNormGated(d.q, dgated.t(), core2, dz.t(), dnw.t(),
                   vt::RmsNormGatedArgs{eps, false});
  std::vector<float> gated(static_cast<size_t>(T) * value_dim);
  dgated.Download(d, gated.data());
  std::vector<uint16_t> gated_bf16 = ToBf16(gated);
  return MatmulBf16(d, gated_bf16, T, value_dim, w.out_proj);  // [T,H]
}

// --- Dense full_attention block. qwen36-forward-notes.md §5; pinned
// Qwen3NextAttention. h [T*H] bf16 -> [T*H] bf16.
std::vector<uint16_t> FullAttnBlock(Dev d, const FullAttnLayerWeights& w,
                                    const HfConfig& cfg,
                                    const std::vector<uint16_t>& h,
                                    const std::vector<int32_t>& positions,
                                    int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  std::vector<float> qgate = MatmulF32(d, h, T, H, w.q_proj);  // [T, 2*Hq*Dh]
  std::vector<float> kf = MatmulF32(d, h, T, H, w.k_proj);     // [T, Hkv*Dh]
  std::vector<float> vf = MatmulF32(d, h, T, H, w.v_proj);     // [T, Hkv*Dh]

  // Gate split: per q-head the projection lays out [q(Dh) | gate(Dh)] (§5).
  std::vector<float> qf(static_cast<size_t>(T) * Hq * Dh);
  std::vector<float> gatef(static_cast<size_t>(T) * Hq * Dh);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t hq = 0; hq < Hq; ++hq) {
      const float* src = qgate.data() + (static_cast<size_t>(t) * Hq + hq) * 2 * Dh;
      float* qd = qf.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      float* gd = gatef.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      std::memcpy(qd, src, static_cast<size_t>(Dh) * sizeof(float));
      std::memcpy(gd, src + Dh, static_cast<size_t>(Dh) * sizeof(float));
    }
  }

  // Per-head gemma-RMSNorm over Dh, then partial NeoX RoPE on positions[0].
  std::vector<float> qnw = WeightF32(w.q_norm);  // match f32 q/k on CUDA
  std::vector<float> knw = WeightF32(w.k_norm);
  DBuf dq(d, DType::kF32, {T * Hq, Dh}, qf.data());
  DBuf dqn(d, DType::kF32, {T * Hq, Dh});
  DBuf dqw(d, DType::kF32, {Dh}, qnw.data());
  vt::RmsNorm(d.q, dqn.t(), dq.t(), dqw.t(), vt::RmsNormArgs{eps, true});
  DBuf dk(d, DType::kF32, {T * Hkv, Dh}, kf.data());
  DBuf dkn(d, DType::kF32, {T * Hkv, Dh});
  DBuf dkw(d, DType::kF32, {Dh}, knw.data());
  vt::RmsNorm(d.q, dkn.t(), dk.t(), dkw.t(), vt::RmsNormArgs{eps, true});

  DBuf dpos(d, DType::kI32, {T}, positions.data());
  Tensor qn3 = Reshape(dqn.t(), {T, Hq, Dh});
  Tensor kn3 = Reshape(dkn.t(), {T, Hkv, Dh});
  vt::RopeNeox(d.q, qn3, kn3, dpos.t(), vt::RopeArgs{base, rot});

  // Causal GQA scaled-dot-product attention, scale = Dh^-0.5.
  DBuf dv(d, DType::kF32, {T, Hkv, Dh}, vf.data());
  DBuf dattn(d, DType::kF32, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  vt::Attention(d.q, dattn.t(), qn3, kn3, dv.t(), vt::AttentionArgs{scale, true});
  std::vector<float> attn(static_cast<size_t>(T) * Hq * Dh);
  dattn.Download(d, attn.data());

  // Sigmoid output gate on the raw gate split, then o-project (§5).
  std::vector<uint16_t> gated(static_cast<size_t>(T) * Hq * Dh);
  for (size_t i = 0; i < gated.size(); ++i)
    gated[i] = vt::F32ToBF16(attn[i] * Sigmoid(gatef[i]));
  return MatmulBf16(d, gated, T, Hq * Dh, w.o_proj);  // [T,H]
}

// --- Batched PAGED full_attention block (M1.8 Task 3). Identical q/k/v prep to
// FullAttnBlock (gemma qk-RMSNorm + partial NeoX RoPE + GQA + output gate), but
// replaces vt::Attention with vt::ReshapeAndCache (write new K/V into the paged
// NHD cache at slot_mapping) + vt::PagedAttention (read causal K/V from the
// cache via block_table/seq_lens/query_start_loc). Mirrors
// qwen3_next.py::Qwen3NextAttention.forward @ e24d1b24 (self.attn(q,k,v) is the
// reshape_and_cache + paged read). h [T*H] bf16 -> [T*H] bf16.
std::vector<uint16_t> FullAttnBlockPaged(Dev d, const FullAttnLayerWeights& w,
                                         const HfConfig& cfg,
                                         const std::vector<uint16_t>& h,
                                         const std::vector<int32_t>& positions,
                                         const CommonAttentionMetadata& meta,
                                         const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int rot = static_cast<int>(cfg.rotary_dim);
  const float base = static_cast<float>(cfg.rope_theta);
  const float eps = static_cast<float>(cfg.rms_norm_eps);
  VT_CHECK(kv.dtype == DType::kF32, "full-attn paged: KV cache must be f32 (T0)");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "full-attn paged: KV cache head dims mismatch config");

  std::vector<float> qgate = MatmulF32(d, h, T, H, w.q_proj);  // [T, 2*Hq*Dh]
  std::vector<float> kf = MatmulF32(d, h, T, H, w.k_proj);     // [T, Hkv*Dh]
  std::vector<float> vf = MatmulF32(d, h, T, H, w.v_proj);     // [T, Hkv*Dh]

  // Gate split: per q-head [q(Dh) | gate(Dh)] (§5).
  std::vector<float> qf(static_cast<size_t>(T) * Hq * Dh);
  std::vector<float> gatef(static_cast<size_t>(T) * Hq * Dh);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t hq = 0; hq < Hq; ++hq) {
      const float* src = qgate.data() + (static_cast<size_t>(t) * Hq + hq) * 2 * Dh;
      float* qd = qf.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      float* gd = gatef.data() + (static_cast<size_t>(t) * Hq + hq) * Dh;
      std::memcpy(qd, src, static_cast<size_t>(Dh) * sizeof(float));
      std::memcpy(gd, src + Dh, static_cast<size_t>(Dh) * sizeof(float));
    }
  }

  // Per-head gemma-RMSNorm over Dh, then partial NeoX RoPE on positions.
  std::vector<float> qnw = WeightF32(w.q_norm);
  std::vector<float> knw = WeightF32(w.k_norm);
  DBuf dq(d, DType::kF32, {T * Hq, Dh}, qf.data());
  DBuf dqn(d, DType::kF32, {T * Hq, Dh});
  DBuf dqw(d, DType::kF32, {Dh}, qnw.data());
  vt::RmsNorm(d.q, dqn.t(), dq.t(), dqw.t(), vt::RmsNormArgs{eps, true});
  DBuf dk(d, DType::kF32, {T * Hkv, Dh}, kf.data());
  DBuf dkn(d, DType::kF32, {T * Hkv, Dh});
  DBuf dkw(d, DType::kF32, {Dh}, knw.data());
  vt::RmsNorm(d.q, dkn.t(), dk.t(), dkw.t(), vt::RmsNormArgs{eps, true});

  DBuf dpos(d, DType::kI32, {T}, positions.data());
  Tensor qn3 = Reshape(dqn.t(), {T, Hq, Dh});
  Tensor kn3 = Reshape(dkn.t(), {T, Hkv, Dh});
  vt::RopeNeox(d.q, qn3, kn3, dpos.t(), vt::RopeArgs{base, rot});

  DBuf dv(d, DType::kF32, {T, Hkv, Dh}, vf.data());

  // Write the new K/V into the paged cache, then read K/V from the cache.
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  DBuf dslot(d, DType::kI64, {T}, meta.slot_mapping.data());
  DBuf dblk(d, DType::kI32, {meta.num_reqs, meta.block_table_num_cols},
            meta.block_table_tensor.data());
  DBuf dsl(d, DType::kI32, {meta.num_reqs}, meta.seq_lens.data());
  DBuf dqsl(d, DType::kI32, {meta.num_reqs + 1}, meta.query_start_loc.data());
  vt::ReshapeAndCache(d.q, kn3, dv.t(), k_cache, v_cache, dslot.t());

  DBuf dattn(d, DType::kF32, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(SizeF(Dh));
  vt::PagedAttention(d.q, dattn.t(), qn3, k_cache, v_cache, dblk.t(), dsl.t(),
                     dqsl.t(), vt::PagedAttentionArgs{scale, meta.causal});
  std::vector<float> attn(static_cast<size_t>(T) * Hq * Dh);
  dattn.Download(d, attn.data());

  // Sigmoid output gate then o-project (§5).
  std::vector<uint16_t> gated(static_cast<size_t>(T) * Hq * Dh);
  for (size_t i = 0; i < gated.size(); ++i)
    gated[i] = vt::F32ToBF16(attn[i] * Sigmoid(gatef[i]));
  return MatmulBf16(d, gated, T, Hq * Dh, w.o_proj);  // [T,H]
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

// --- Sparse-MoE block (moe-semantics.md §1-§6). Router top-k over ALL experts,
// then the ACTIVATED-EXPERT token-gather loop (not O(E)-dense), shared expert
// with sigmoid gate, and the weighted combine. h [T*H] bf16 -> [T*H] bf16.
std::vector<uint16_t> MoeBlock(Dev d, const MoeBlockWeights& w,
                               const HfConfig& cfg,
                               const std::vector<uint16_t>& h, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = cfg.num_experts;
  const int64_t top_k = cfg.num_experts_per_tok;
  const int64_t I = cfg.moe_intermediate_size;
  const int64_t Is = cfg.shared_expert_intermediate_size;
  // fp4-resident NVFP4 experts/shared (M2.2b real-ckpt CUDA load) vs bf16
  // (synthetic / GGUF). Exactly one set is populated (see qwen3_5_weights.h).
  const bool fp4 = !w.expert_gate_fp4.empty();

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

  // Shared expert (moe-semantics.md §5): silu-mul MLP then sigmoid(x@Wseg)*out.
  std::vector<float> sg = fp4 ? MatmulNvfp4F32(d, h, T, H, w.shared_gate_proj_fp4)
                              : MatmulF32(d, h, T, H, w.shared_gate_proj);  // [T,Is]
  std::vector<float> su = fp4 ? MatmulNvfp4F32(d, h, T, H, w.shared_up_proj_fp4)
                              : MatmulF32(d, h, T, H, w.shared_up_proj);    // [T,Is]
  std::vector<uint16_t> sact(static_cast<size_t>(T) * Is);
  for (size_t i = 0; i < sact.size(); ++i)
    sact[i] = vt::F32ToBF16(Silu(sg[i]) * su[i]);
  std::vector<float> sd = fp4 ? MatmulNvfp4F32(d, sact, T, Is, w.shared_down_proj_fp4)
                              : MatmulF32(d, sact, T, Is, w.shared_down_proj);  // [T,H]
  std::vector<float> gl = MatmulF32(d, h, T, H, w.shared_gate);           // [T,1]
  std::vector<uint16_t> shared(static_cast<size_t>(T) * H);
  for (int64_t t = 0; t < T; ++t) {
    const float gate = Sigmoid(gl[static_cast<size_t>(t)]);
    for (int64_t c = 0; c < H; ++c)
      shared[static_cast<size_t>(t) * H + c] =
          vt::F32ToBF16(gate * sd[static_cast<size_t>(t) * H + c]);
  }

  // Combine (moe-semantics.md §6): out = shared + sum_j w_j * expert_out_j.
  DBuf deo(d, DType::kBF16, {T, top_k, H}, expert_out.data());
  DBuf dwt(d, DType::kF32, {T, top_k}, weights.data());
  DBuf dsh(d, DType::kBF16, {T, H}, shared.data());
  DBuf dout(d, DType::kBF16, {T, H});
  vt::MoeCombine(d.q, dout.t(), deo.t(), dwt.t(), &dsh.t());
  std::vector<uint16_t> out(static_cast<size_t>(T) * H);
  dout.Download(d, out.data());
  return out;
}

// One decoder layer over the fused residual stream. `hidden` (bf16 [T*H]) is
// the previous block's output (the delta); `res` (f32 [T,H], device) is the
// accumulator. Mirrors qwen3_next.py::Qwen3NextDecoderLayer.forward:
//   h  = input_layernorm(hidden, res)          # res += hidden; h = norm(res)
//   a  = attn/gdn(h)
//   h2 = post_attention_layernorm(a, res)      # res += a; h2 = norm(res)
//   hidden = mlp(h2)                            # MoE block; returned as delta
void RunLayer(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
              std::vector<uint16_t>& hidden, DBuf& res,
              const std::vector<int32_t>& positions, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  DBuf dh(d, DType::kBF16, {T, H}, hidden.data());
  DBuf dw_in(d, layer.input_layernorm.dtype, {H},
             layer.input_layernorm.bytes.data());
  DBuf dhn(d, DType::kBF16, {T, H});
  // Qwen3NextRMSNorm == GemmaRMSNorm (weight applied as 1+w).
  vt::RmsNorm(d.q, dhn.t(), dh.t(), dw_in.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> hn(static_cast<size_t>(T) * H);
  dhn.Download(d, hn.data());

  std::vector<uint16_t> attn =
      layer.is_linear_attention
          ? GdnBlock(d, layer.gdn, cfg, hn, T)
          : FullAttnBlock(d, layer.attn, cfg, hn, positions, T);

  DBuf datt(d, DType::kBF16, {T, H}, attn.data());
  DBuf dw_post(d, layer.post_attention_layernorm.dtype, {H},
               layer.post_attention_layernorm.bytes.data());
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), datt.t(), dw_post.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> h2(static_cast<size_t>(T) * H);
  dh2.Download(d, h2.data());

  hidden = MoeBlock(d, layer.moe, cfg, h2, T);
}

// Batched PAGED decoder layer (M1.8 Task 3). Same residual/norm/MoE thread as
// RunLayer, but the attention block reads/writes the paged KV cache
// (full-attn: attn_kv) or the persistent GDN mamba state (GDN: gdn_state).
// Exactly one of {attn_kv, gdn_state} is non-null (per layer type).
void RunLayerPaged(Dev d, const Qwen3_5MoeLayerWeights& layer, const HfConfig& cfg,
                   std::vector<uint16_t>& hidden, DBuf& res,
                   const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& attn_meta,
                   const GDNAttentionMetadata& gdn_meta,
                   const PagedKvCache* attn_kv, const GdnStateCache* gdn_state,
                   int64_t T) {
  const int64_t H = cfg.hidden_size;
  const float eps = static_cast<float>(cfg.rms_norm_eps);

  DBuf dh(d, DType::kBF16, {T, H}, hidden.data());
  DBuf dw_in(d, layer.input_layernorm.dtype, {H},
             layer.input_layernorm.bytes.data());
  DBuf dhn(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dhn.t(), dh.t(), dw_in.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> hn(static_cast<size_t>(T) * H);
  dhn.Download(d, hn.data());

  std::vector<uint16_t> attn;
  if (layer.is_linear_attention) {
    VT_CHECK(gdn_state != nullptr, "paged layer: GDN layer needs a GdnStateCache");
    attn = GdnBlockPaged(d, layer.gdn, cfg, hn, gdn_meta, *gdn_state, T);
  } else {
    VT_CHECK(attn_kv != nullptr, "paged layer: full-attn layer needs a PagedKvCache");
    attn = FullAttnBlockPaged(d, layer.attn, cfg, hn, positions, attn_meta,
                              *attn_kv, T);
  }

  DBuf datt(d, DType::kBF16, {T, H}, attn.data());
  DBuf dw_post(d, layer.post_attention_layernorm.dtype, {H},
               layer.post_attention_layernorm.bytes.data());
  DBuf dh2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dh2.t(), datt.t(), dw_post.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> h2(static_cast<size_t>(T) * H);
  dh2.Download(d, h2.data());

  hidden = MoeBlock(d, layer.moe, cfg, h2, T);
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

  // Embed: hidden = embed_tokens[token_ids] (bf16). residual = 0 (f32).
  DBuf dtab(d, weights.embed_tokens.dtype, {vocab, H},
            weights.embed_tokens.bytes.data());
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf dembed(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, dembed.t(), dtab.t(), dids.t());
  std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
  dembed.Download(d, hidden.data());

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
  DBuf dh(d, DType::kBF16, {T, H}, hidden.data());
  DBuf dfn(d, weights.final_norm.dtype, {H}, weights.final_norm.bytes.data());
  DBuf dnorm(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, dnorm.t(), dh.t(), dfn.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> normed(static_cast<size_t>(T) * H);
  dnorm.Download(d, normed.data());

  // lm_head: fp4-resident (M2.2b) when populated, else the bf16/GGUF path.
  return weights.lm_head_fp4.Empty()
             ? MatmulF32(d, normed, T, H, weights.lm_head)
             : MatmulNvfp4F32(d, normed, T, H, weights.lm_head_fp4);
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

  // Embed: hidden = embed_tokens[token_ids] (bf16). residual = 0 (f32).
  DBuf dtab(d, weights.embed_tokens.dtype, {vocab, H},
            weights.embed_tokens.bytes.data());
  DBuf dids(d, DType::kI32, {T}, token_ids.data());
  DBuf dembed(d, DType::kBF16, {T, H});
  vt::Embedding(d.q, dembed.t(), dtab.t(), dids.t());
  std::vector<uint16_t> hidden(static_cast<size_t>(T) * H);
  dembed.Download(d, hidden.data());

  DBuf res(d, DType::kF32, {T, H});
  res.Zero(d);

  for (int64_t l = 0; l < config.num_hidden_layers; ++l)
    RunLayer(d, weights.layers[static_cast<size_t>(l)], config, hidden, res,
             positions, T);

  // Final RMSNorm over the fused stream (res += hidden; norm), then lm_head.
  DBuf dh(d, DType::kBF16, {T, H}, hidden.data());
  DBuf dfn(d, weights.final_norm.dtype, {H}, weights.final_norm.bytes.data());
  DBuf dnorm(d, DType::kBF16, {T, H});
  // Final norm is GemmaRMSNorm too (weight applied as 1+w).
  vt::RmsNorm(d.q, dnorm.t(), dh.t(), dfn.t(), vt::RmsNormArgs{eps, true},
              &res.t());
  std::vector<uint16_t> normed(static_cast<size_t>(T) * H);
  dnorm.Download(d, normed.data());

  // lm_head: fp4-resident (M2.2b) when populated, else the bf16/GGUF path.
  return weights.lm_head_fp4.Empty()
             ? MatmulF32(d, normed, T, H, weights.lm_head)
             : MatmulNvfp4F32(d, normed, T, H, weights.lm_head_fp4);  // [T, vocab]
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
  std::vector<uint16_t> hidden(static_cast<size_t>(T) * H, 0);
  RunLayer(d, layer, config, hidden, res, positions, T);

  // Combined stream out = residual + hidden (f32), directly comparable to the
  // layer golden's `out`.
  std::vector<float> res_host(static_cast<size_t>(T) * H);
  res.Download(d, res_host.data());
  std::vector<float> out(static_cast<size_t>(T) * H);
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = res_host[i] + vt::BF16ToF32(hidden[i]);
  return out;
}

}  // namespace vllm
