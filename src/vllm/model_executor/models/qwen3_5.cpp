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
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

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
        ExpertMlp(d, w.expert_gate[static_cast<size_t>(e)],
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
  std::vector<float> sg = MatmulF32(d, h, T, H, w.shared_gate_proj);  // [T,Is]
  std::vector<float> su = MatmulF32(d, h, T, H, w.shared_up_proj);    // [T,Is]
  std::vector<uint16_t> sact(static_cast<size_t>(T) * Is);
  for (size_t i = 0; i < sact.size(); ++i)
    sact[i] = vt::F32ToBF16(Silu(sg[i]) * su[i]);
  std::vector<float> sd = MatmulF32(d, sact, T, Is, w.shared_down_proj);  // [T,H]
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

}  // namespace

std::vector<float> Qwen3_5Model::Forward(const std::vector<int32_t>& token_ids,
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

  return MatmulF32(d, normed, T, H, weights.lm_head);  // [T, vocab] f32
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
