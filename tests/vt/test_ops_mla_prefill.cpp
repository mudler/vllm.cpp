// vt::MlaPrefillAttention unit tests (MLA campaign W5).
//
// Upstream test modules ported per .agents/test-porting.md:
//   * vllm/tests/v1/attention/test_mla_backends.py @ pin e24d1b24 — "MLA backend
//     correctness vs a reference across batch/seq shapes"; its batch/seq-shape
//     sweep is what the ragged / single-token / chunk-boundary cases below cover.
//   * vllm/tests/v1/attention/test_mla_prefill_quant_output.py — the MLA PREFILL
//     output-vs-reference module, and the source of the REAL DeepSeek-V2-Lite
//     prefill dims: `:158` "v = v_head_dim(128); DeepSeek-V2-Lite has 16 query
//     heads". Its quant (fp8-output) arms are NOT ported: they need
//     `is_device_capability_family(100)` (mla_attention.py:1382-1385) and are
//     unreachable on sm_121 — recorded, not silently dropped.
//
// Kernels under port: vllm/v1/attention/backends/mla/prefill/flash_attn.py:153-248
//   (`FlashAttnPrefillBackend`), the ONLY MLA prefill backend reachable on GB10
//   (`mla/prefill/selector.py:66-76`, OBSERVED at W0), over the vendored FA-2.
//
// THE ORACLE IS INDEPENDENT BY CONSTRUCTION. `RefPrefill` below is a plain
// TWO-PASS softmax in DOUBLE precision (max over the row, then the exp-sum, then
// the weighted sum). FlashAttention is a streaming ONLINE-softmax with a running
// rescale, and our CPU reference — while two-pass — accumulates in f32. So a bug
// in the streaming rescale cannot hide behind a matching reference, which is the
// same independence rule the W4 decode gate follows.
//
// GATE DIMENSIONS ARE THE REAL ONES: DeepSeek-V2-Lite (confirmed at W0)
// qk_nope_head_dim 128 + qk_rope_head_dim 64 = QK 192, v_head_dim 128,
// num_attention_heads 16, and a `scale` carrying the YaRN mscale^2 correction.
// DeepSeek-V3's 128-head shape is covered too.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::MlaPrefillAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {

// ─── DeepSeek-V2-Lite prefill geometry (config.json, confirmed at W0) ───────
constexpr int kQkNope = 128;
constexpr int kQkRope = 64;
constexpr int kQkHeadDim = kQkNope + kQkRope;  // 192
constexpr int kVHeadDim = 128;
constexpr int kHeadsLite = 16;

// `self.scale` as MLAAttentionImpl builds it: qk_head_dim^-0.5 times the YaRN
// mscale^2 correction (deepseek_v2.py:981-996 + :1067-1075). V2-Lite's
// rope_scaling is {factor: 40, mscale: 0.707, mscale_all_dim: 0.707}.
double LiteScale() {
  const double mscale = 0.1 * 0.707 * std::log(40.0) + 1.0;
  return (1.0 / std::sqrt(static_cast<double>(kQkHeadDim))) * mscale * mscale;
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceTensor {
 public:
  DeviceTensor(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = Contig(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
  }
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::F32ToBF16(v[i]);
  return o;
}

std::vector<float> FromBf16(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = vt::BF16ToF32(v[i]);
  return o;
}

// Round-trip through bf16 so the ORACLE sees exactly the bytes the kernel does.
std::vector<float> Bf16Round(const std::vector<float>& v) { return FromBf16(ToBf16(v)); }

// ─── the independent oracle ─────────────────────────────────────────────────
// Two-pass softmax in DOUBLE over varlen q/k/v. FlashAttention's causal is
// BOTTOM-RIGHT aligned: query i of a request with lengths (Lq, Lk) sees keys
// j <= i + (Lk - Lq).
void RefPrefill(std::vector<float>& out, std::vector<float>* lse, const std::vector<float>& q,
                const std::vector<float>& k, const std::vector<float>& v,
                const std::vector<int32_t>& cu_q, const std::vector<int32_t>& cu_k, int heads,
                int dqk, int dv, double scale, bool causal) {
  const int nreq = static_cast<int>(cu_q.size()) - 1;
  const int total_q = cu_q.back();
  out.assign(static_cast<size_t>(total_q) * heads * dv, 0.0f);
  if (lse != nullptr)
    lse->assign(static_cast<size_t>(heads) * total_q,
                -std::numeric_limits<float>::infinity());
  std::vector<double> logits;
  for (int b = 0; b < nreq; ++b) {
    const int q0 = cu_q[static_cast<size_t>(b)];
    const int lq = cu_q[static_cast<size_t>(b) + 1] - q0;
    const int k0 = cu_k[static_cast<size_t>(b)];
    const int lk = cu_k[static_cast<size_t>(b) + 1] - k0;
    for (int iq = 0; iq < lq; ++iq) {
      const int t = q0 + iq;
      const int visible =
          causal ? std::min(lk, std::max(0, iq + (lk - lq) + 1)) : lk;
      for (int h = 0; h < heads; ++h) {
        const float* qp = q.data() + (static_cast<size_t>(t) * heads + h) * dqk;
        logits.assign(static_cast<size_t>(std::max(visible, 0)), 0.0);
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < visible; ++j) {
          const float* kp = k.data() + (static_cast<size_t>(k0 + j) * heads + h) * dqk;
          double dot = 0.0;
          for (int d = 0; d < dqk; ++d) dot += static_cast<double>(qp[d]) * kp[d];
          dot *= scale;
          logits[static_cast<size_t>(j)] = dot;
          mx = std::max(mx, dot);
        }
        double denom = 0.0;
        for (int j = 0; j < visible; ++j) {
          logits[static_cast<size_t>(j)] = std::exp(logits[static_cast<size_t>(j)] - mx);
          denom += logits[static_cast<size_t>(j)];
        }
        float* op = out.data() + (static_cast<size_t>(t) * heads + h) * dv;
        std::vector<double> acc(static_cast<size_t>(dv), 0.0);
        for (int j = 0; j < visible; ++j) {
          const float* vp = v.data() + (static_cast<size_t>(k0 + j) * heads + h) * dv;
          const double p = logits[static_cast<size_t>(j)];
          for (int d = 0; d < dv; ++d) acc[static_cast<size_t>(d)] += p * vp[d];
        }
        for (int d = 0; d < dv; ++d) {
          op[d] = denom > 0.0 ? static_cast<float>(acc[static_cast<size_t>(d)] / denom) : 0.0f;
        }
        if (lse != nullptr) {
          (*lse)[static_cast<size_t>(h) * total_q + t] =
              denom > 0.0 ? static_cast<float>(mx + std::log(denom))
                          : -std::numeric_limits<float>::infinity();
        }
      }
    }
  }
}

std::vector<int32_t> Cumsum(const std::vector<int32_t>& lens) {
  std::vector<int32_t> cu(lens.size() + 1, 0);
  for (size_t i = 0; i < lens.size(); ++i) cu[i + 1] = cu[i] + lens[i];
  return cu;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE_FALSE(std::isnan(a[i]));
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return m;
}

// One end-to-end comparison of an impl against the oracle. `dev` selects CPU
// (f32) or CUDA (bf16, the production dtype).
struct PrefillCase {
  std::vector<int32_t> q_lens;
  std::vector<int32_t> k_lens;
  int heads = kHeadsLite;
  bool causal = true;
};

void RunCpuCase(const PrefillCase& c, uint32_t seed, double tol) {
  const auto cu_q = Cumsum(c.q_lens);
  const auto cu_k = Cumsum(c.k_lens);
  const int total_q = cu_q.back();
  const int total_k = cu_k.back();
  const int h = c.heads;
  const double scale = LiteScale();

  auto q = RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, seed);
  auto k = RandF32(static_cast<size_t>(std::max(total_k, 1)) * h * kQkHeadDim, seed + 7);
  auto v = RandF32(static_cast<size_t>(std::max(total_k, 1)) * h * kVHeadDim, seed + 13);

  std::vector<float> ref;
  std::vector<float> ref_lse;
  RefPrefill(ref, &ref_lse, q, k, v, cu_q, cu_k, h, kQkHeadDim, kVHeadDim, scale, c.causal);

  // NaN-poison the output so a kernel that fails to write FAILS the gate.
  std::vector<float> out(static_cast<size_t>(total_q) * h * kVHeadDim,
                         std::numeric_limits<float>::quiet_NaN());
  std::vector<float> lse(static_cast<size_t>(h) * total_q,
                         std::numeric_limits<float>::quiet_NaN());
  auto cu_q_v = cu_q;
  auto cu_k_v = cu_k;

  Tensor tq = Contig(q.data(), DType::kF32, Cpu(), {total_q, h, kQkHeadDim});
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {std::max(total_k, 1), h, kQkHeadDim});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {std::max(total_k, 1), h, kVHeadDim});
  Tensor to = Contig(out.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
  Tensor tl = Contig(lse.data(), DType::kF32, Cpu(), {h, total_q});
  Tensor tcq =
      Contig(cu_q_v.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cu_q_v.size())});
  Tensor tck =
      Contig(cu_k_v.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cu_k_v.size())});

  MlaPrefillAttentionArgs args;
  args.scale = static_cast<float>(scale);
  args.causal = c.causal;
  Queue q0 = CpuQ();
  vt::MlaPrefillAttention(q0, to, &tl, tq, tk, tv, tcq, tck, args);

  CHECK(MaxAbsDiff(out, ref) < tol);
  for (size_t i = 0; i < lse.size(); ++i) {
    if (std::isinf(ref_lse[i])) {
      CHECK(std::isinf(lse[i]));
      CHECK(lse[i] < 0.0f);
    } else {
      CHECK(std::fabs(lse[i] - ref_lse[i]) < 1e-3);
    }
  }
}

}  // namespace

TEST_CASE("MLA prefill CPU reference matches the independent two-pass oracle") {
  // Ragged varlen, the causal new-tokens form (flash_attn.py:223 causal=True).
  RunCpuCase({{7, 1, 33, 16}, {7, 1, 33, 16}, kHeadsLite, true}, 11u, 2e-4);
  // The NON-causal context-chunk form (flash_attn.py:246, "Context is unmasked")
  // with cu_seqlens_k DIFFERENT from cu_seqlens_q — the shape only the chunked
  // path produces.
  RunCpuCase({{7, 1, 33, 16}, {40, 16, 3, 0}, kHeadsLite, false}, 23u, 2e-4);
  // Single-token queries.
  RunCpuCase({{1, 1, 1}, {5, 1, 9}, 4, false}, 31u, 2e-4);
  // A request with ZERO keys in this chunk — the degenerate path the chunked
  // loop reaches whenever a request's context is shorter than the chunk grid
  // (mla_attention.py:1699 clamp(min=0)); its LSE must be -inf so
  // vt::MergeAttnStates' both(-inf) guard (merge_attn_states.cu:100-134) fires.
  RunCpuCase({{4, 4}, {0, 6}, 3, false}, 37u, 2e-4);
  // A request with ZERO queries (upstream's `if not use_fp8_prefill` loop can
  // produce an empty span; the op must simply skip it).
  RunCpuCase({{0, 5}, {0, 5}, 3, true}, 41u, 2e-4);
}

TEST_CASE("MLA prefill rejects the shapes and dtypes upstream cannot reach") {
  std::vector<float> buf(4096, 0.0f);
  std::vector<int32_t> cu{0, 2};
  Tensor tq = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 8});
  Tensor tk = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 8});
  Tensor tv = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 4});
  Tensor to = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 4});
  Tensor tcu = Contig(cu.data(), DType::kI32, Cpu(), {2});
  MlaPrefillAttentionArgs args;
  args.scale = 0.5f;
  Queue q0 = CpuQ();

  // v_head_dim WIDER than qk_head_dim: upstream only ever pads V UP
  // (flash_attn.py:164-168), so this must be refused, not silently truncated.
  Tensor bad_v = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 16});
  Tensor bad_o = Contig(buf.data(), DType::kF32, Cpu(), {2, 1, 16});
  CHECK_THROWS(vt::MlaPrefillAttention(q0, bad_o, nullptr, tq, tk, bad_v, tcu, tcu, args));
  // scale must be set explicitly.
  MlaPrefillAttentionArgs zero_scale;
  CHECK_THROWS(vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcu, tcu, zero_scale));
  // K/V head count must match Q's (MLA prefill is multi-head on both sides).
  Tensor bad_k = Contig(buf.data(), DType::kF32, Cpu(), {2, 2, 8});
  CHECK_THROWS(vt::MlaPrefillAttention(q0, to, nullptr, tq, bad_k, tv, tcu, tcu, args));
}

TEST_CASE("CUDA MLA prefill matches the oracle at the real DeepSeek geometry") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);

  struct Case {
    std::vector<int32_t> q_lens;
    std::vector<int32_t> k_lens;
    int heads;
    bool causal;
    const char* what;
  };
  const std::vector<Case> cases = {
      // The NEW-TOKENS call: causal, cu_seqlens_k == cu_seqlens_q. Ragged
      // lengths straddling FA-2's kBlockM=64 / kBlockN=64 tiles.
      {{1, 63, 64, 65, 129}, {1, 63, 64, 65, 129}, kHeadsLite, true, "new tokens, ragged"},
      // A single long request (one full page-aligned prefill).
      {{256}, {256}, kHeadsLite, true, "single 256-token prefill"},
      // The CONTEXT-CHUNK call: non-causal, cu_seqlens_k independent of q.
      {{16, 16, 16}, {200, 1, 64}, kHeadsLite, false, "context chunk, non-causal"},
      // A chunk in which one request contributes ZERO keys — the boundary case.
      {{16, 16, 16}, {128, 0, 33}, kHeadsLite, false, "context chunk with an empty request"},
      // DeepSeek-V3's 128 heads.
      {{8, 8}, {32, 8}, 128, false, "V3 128-head geometry"},
      // Single-token queries against a long context (the decode-shaped prefill).
      {{1, 1, 1, 1}, {97, 1, 512, 33}, kHeadsLite, false, "single-token queries"},
  };

  for (const auto& c : cases) {
    CAPTURE(c.what);
    const auto cu_q = Cumsum(c.q_lens);
    const auto cu_k = Cumsum(c.k_lens);
    const int total_q = cu_q.back();
    const int total_k = std::max(cu_k.back(), 1);
    const int h = c.heads;
    const double scale = LiteScale();

    // bf16 is the production dtype; round-trip the inputs so the oracle sees
    // exactly the bytes the kernel does.
    auto qf = Bf16Round(RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, 101u));
    auto kf = Bf16Round(RandF32(static_cast<size_t>(total_k) * h * kQkHeadDim, 211u));
    auto vf = Bf16Round(RandF32(static_cast<size_t>(total_k) * h * kVHeadDim, 307u));

    std::vector<float> ref;
    std::vector<float> ref_lse;
    RefPrefill(ref, &ref_lse, qf, kf, vf, cu_q, cu_k, h, kQkHeadDim, kVHeadDim, scale,
               c.causal);

    auto qb = ToBf16(qf);
    auto kb = ToBf16(kf);
    auto vb = ToBf16(vf);
    // NaN-poison BOTH outputs: a kernel that fails to write fails the gate.
    std::vector<uint16_t> out_poison(static_cast<size_t>(total_q) * h * kVHeadDim,
                                     vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN()));
    std::vector<float> lse_poison(static_cast<size_t>(h) * total_q,
                                  std::numeric_limits<float>::quiet_NaN());
    auto cu_q_v = cu_q;
    auto cu_k_v = cu_k;

    DeviceTensor dq(b, g.q, DType::kBF16, {total_q, h, kQkHeadDim}, qb.data());
    DeviceTensor dk(b, g.q, DType::kBF16, {total_k, h, kQkHeadDim}, kb.data());
    DeviceTensor dv(b, g.q, DType::kBF16, {total_k, h, kVHeadDim}, vb.data());
    DeviceTensor dout(b, g.q, DType::kBF16, {total_q, h, kVHeadDim}, out_poison.data());
    DeviceTensor dlse(b, g.q, DType::kF32, {h, total_q}, lse_poison.data());
    DeviceTensor dcq(b, g.q, DType::kI32, {static_cast<int64_t>(cu_q_v.size())},
                     cu_q_v.data());
    DeviceTensor dck(b, g.q, DType::kI32, {static_cast<int64_t>(cu_k_v.size())},
                     cu_k_v.data());

    MlaPrefillAttentionArgs args;
    args.scale = static_cast<float>(scale);
    args.causal = c.causal;
    vt::MlaPrefillAttention(g.q, dout.tensor(), &dlse.tensor(), dq.tensor(), dk.tensor(),
                            dv.tensor(), dcq.tensor(), dck.tensor(), args);
    b.Synchronize(g.q);

    std::vector<uint16_t> got_b(out_poison.size());
    dout.Download(g.q, got_b.data());
    std::vector<float> got_lse(lse_poison.size());
    dlse.Download(g.q, got_lse.data());
    const auto got = FromBf16(got_b);

    // bf16 accumulation in FA-2 vs a double oracle: the tolerance is the bf16
    // output quantum times a small factor, the same bar the FA-2 prefill
    // regression uses.
    CHECK(MaxAbsDiff(got, ref) < 3e-2);
    for (size_t i = 0; i < got_lse.size(); ++i) {
      if (std::isinf(ref_lse[i])) {
        CHECK(std::isinf(got_lse[i]));
        CHECK(got_lse[i] < 0.0f);
      } else {
        CHECK(std::fabs(got_lse[i] - ref_lse[i]) < 5e-2);
      }
    }
  }
}

TEST_CASE("CUDA MLA prefill is run-to-run bit-exact") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);

  const std::vector<int32_t> q_lens{37, 64, 129};
  const auto cu = Cumsum(q_lens);
  const int total = cu.back();
  const int h = kHeadsLite;
  auto qb = ToBf16(RandF32(static_cast<size_t>(total) * h * kQkHeadDim, 5u));
  auto kb = ToBf16(RandF32(static_cast<size_t>(total) * h * kQkHeadDim, 9u));
  auto vb = ToBf16(RandF32(static_cast<size_t>(total) * h * kVHeadDim, 17u));
  auto cu_v = cu;

  DeviceTensor dq(b, g.q, DType::kBF16, {total, h, kQkHeadDim}, qb.data());
  DeviceTensor dk(b, g.q, DType::kBF16, {total, h, kQkHeadDim}, kb.data());
  DeviceTensor dv(b, g.q, DType::kBF16, {total, h, kVHeadDim}, vb.data());
  DeviceTensor dcu(b, g.q, DType::kI32, {static_cast<int64_t>(cu_v.size())}, cu_v.data());

  MlaPrefillAttentionArgs args;
  args.scale = static_cast<float>(LiteScale());
  args.causal = true;

  std::vector<uint16_t> first;
  for (int rep = 0; rep < 5; ++rep) {
    std::vector<uint16_t> poison(static_cast<size_t>(total) * h * kVHeadDim,
                                 vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN()));
    DeviceTensor dout(b, g.q, DType::kBF16, {total, h, kVHeadDim}, poison.data());
    vt::MlaPrefillAttention(g.q, dout.tensor(), nullptr, dq.tensor(), dk.tensor(),
                            dv.tensor(), dcu.tensor(), dcu.tensor(), args);
    b.Synchronize(g.q);
    std::vector<uint16_t> got(poison.size());
    dout.Download(g.q, got.data());
    if (rep == 0) {
      first = got;
      for (uint16_t x : got) REQUIRE_FALSE(std::isnan(vt::BF16ToF32(x)));
    } else {
      CHECK(got == first);
    }
  }
}

// ───────────────────────────────────────────────────────────────────────────
// THE SLIDING WINDOW (dots3-note W4b-2, #699).
//
// `MlaPrefillAttentionArgs::window_size` is FlashAttention's `(left, right)`
// pair, exactly as upstream hands it: `run_sliding_window` calls
// `_flash_attn_varlen_diff_headdims(..., causal=True,
// window_size=(sliding_window - 1, 0))`
// (`vllm/models/dots3_note/nvidia/attention.py:279-305` @ `bc2d63e650`, the
// pair at `:300`).
//
// THE ORACLE IS THE OP ITSELF, ON A DIFFERENT INPUT. `RefPrefill` is NOT given
// a window, deliberately: a reference that recomputed `iq + (lk - lq) - left`
// would share the bottom-right arithmetic it is supposed to check. Instead the
// windowed multi-query call is compared against an EXPANDED batch in which
// every query becomes its own single-query request carrying only the keys its
// window admits, run through the UNWINDOWED path that `RefPrefill` already
// gates. With `lq == 1` the bottom-right causal bound admits every key handed
// in, so the expansion needs no mask of its own.
namespace {

// Expand `c` into one request per query, each carrying only that query's
// windowed key range, and run the UNWINDOWED op over it. Returns the [total_q,
// heads, dv] output in the original query order.
std::vector<float> RunExpandedWindowCpu(const std::vector<int32_t>& q_lens,
                                        const std::vector<int32_t>& k_lens, int heads,
                                        int win_left, const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v, double scale,
                                        int64_t* keys_dropped) {
  const std::vector<int32_t> cu_q = Cumsum(q_lens);
  const std::vector<int32_t> cu_k = Cumsum(k_lens);
  const int total_q = cu_q.back();
  std::vector<int32_t> sub_q(static_cast<size_t>(total_q), 1);
  std::vector<int32_t> sub_k;
  std::vector<float> kk, vv;
  *keys_dropped = 0;
  for (size_t b = 0; b + 1 < cu_q.size(); ++b) {
    const int lq = cu_q[b + 1] - cu_q[b];
    const int lk = cu_k[b + 1] - cu_k[b];
    for (int iq = 0; iq < lq; ++iq) {
      const int p = iq + (lk - lq);           // the bottom-right query position
      const int hi = std::min(lk, p + 1);     // the causal bound, inclusive of p
      const int lo = std::max(0, p - win_left);
      REQUIRE(hi >= lo);
      sub_k.push_back(hi - lo);
      *keys_dropped += lo;
      for (int j = lo; j < hi; ++j) {
        const size_t kr = (static_cast<size_t>(cu_k[b] + j) * heads) * kQkHeadDim;
        kk.insert(kk.end(), k.begin() + static_cast<long>(kr),
                  k.begin() + static_cast<long>(kr + static_cast<size_t>(heads) * kQkHeadDim));
        const size_t vr = (static_cast<size_t>(cu_k[b] + j) * heads) * kVHeadDim;
        vv.insert(vv.end(), v.begin() + static_cast<long>(vr),
                  v.begin() + static_cast<long>(vr + static_cast<size_t>(heads) * kVHeadDim));
      }
    }
  }
  const std::vector<int32_t> cu_sq = Cumsum(sub_q);
  const std::vector<int32_t> cu_sk = Cumsum(sub_k);
  const int total_k = cu_sk.back();
  std::vector<float> out(static_cast<size_t>(total_q) * heads * kVHeadDim,
                         std::numeric_limits<float>::quiet_NaN());
  std::vector<float> qq = q;
  std::vector<int32_t> a = cu_sq, bb = cu_sk;
  Tensor tq = Contig(qq.data(), DType::kF32, Cpu(), {total_q, heads, kQkHeadDim});
  Tensor tk = Contig(kk.data(), DType::kF32, Cpu(), {std::max(total_k, 1), heads, kQkHeadDim});
  Tensor tv = Contig(vv.data(), DType::kF32, Cpu(), {std::max(total_k, 1), heads, kVHeadDim});
  Tensor to = Contig(out.data(), DType::kF32, Cpu(), {total_q, heads, kVHeadDim});
  Tensor tcq = Contig(a.data(), DType::kI32, Cpu(), {static_cast<int64_t>(a.size())});
  Tensor tck = Contig(bb.data(), DType::kI32, Cpu(), {static_cast<int64_t>(bb.size())});
  MlaPrefillAttentionArgs args;
  args.scale = static_cast<float>(scale);
  args.causal = true;  // lq == 1 everywhere, so this admits every key handed in
  Queue q0 = CpuQ();
  vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcq, tck, args);
  return out;
}

}  // namespace

TEST_CASE("MLA prefill CPU: a sliding window keeps exactly the last W keys per query") {
  const std::vector<int32_t> q_lens{7, 1, 33, 16};
  const std::vector<int32_t> k_lens = q_lens;  // a fresh prompt: seq_len == query_len
  constexpr int kWindow = 5;
  const int h = 4;
  const double scale = LiteScale();
  const std::vector<int32_t> cu_q = Cumsum(q_lens);
  const int total_q = cu_q.back();
  const auto q = RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, 909u);
  const auto k = RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, 911u);
  const auto v = RandF32(static_cast<size_t>(total_q) * h * kVHeadDim, 913u);

  std::vector<float> out(static_cast<size_t>(total_q) * h * kVHeadDim,
                         std::numeric_limits<float>::quiet_NaN());
  std::vector<int32_t> cu_a = cu_q, cu_b = cu_q;
  Tensor tq = Contig(const_cast<float*>(q.data()), DType::kF32, Cpu(),
                     {total_q, h, kQkHeadDim});
  Tensor tk = Contig(const_cast<float*>(k.data()), DType::kF32, Cpu(),
                     {total_q, h, kQkHeadDim});
  Tensor tv = Contig(const_cast<float*>(v.data()), DType::kF32, Cpu(),
                     {total_q, h, kVHeadDim});
  Tensor to = Contig(out.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
  Tensor tcq = Contig(cu_a.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cu_a.size())});
  Tensor tck = Contig(cu_b.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cu_b.size())});
  MlaPrefillAttentionArgs args;
  args.scale = static_cast<float>(scale);
  args.causal = true;
  args.window_size = vt::AttentionWindow{kWindow - 1, 0};
  Queue q0 = CpuQ();
  vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcq, tck, args);

  int64_t dropped = 0;
  const std::vector<float> want =
      RunExpandedWindowCpu(q_lens, k_lens, h, kWindow - 1, q, k, v, scale, &dropped);
  MESSAGE("MLA prefill window " << kWindow << ": " << dropped
                                << " (query, key) pairs dropped across " << total_q
                                << " queries");
  // The fixture has to BITE: 7+1+33+16 = 57 queries, and every query past
  // position 4 in its own request loses at least one key.
  REQUIRE(dropped > 0);
  CHECK(MaxAbsDiff(out, want) < 2e-4);

  // The CONTROL: without the window the same call is a different answer.
  std::vector<float> full(static_cast<size_t>(total_q) * h * kVHeadDim,
                          std::numeric_limits<float>::quiet_NaN());
  Tensor tof = Contig(full.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
  MlaPrefillAttentionArgs no_win = args;
  no_win.window_size = std::nullopt;
  vt::MlaPrefillAttention(q0, tof, nullptr, tq, tk, tv, tcq, tck, no_win);
  CHECK(MaxAbsDiff(out, full) > 1e-3);

  // A window at least as wide as the longest request is BIT-IDENTICAL to no
  // window: the absent state is a not-taken branch, not a mask.
  std::vector<float> wide(static_cast<size_t>(total_q) * h * kVHeadDim,
                          std::numeric_limits<float>::quiet_NaN());
  Tensor tow = Contig(wide.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
  MlaPrefillAttentionArgs wide_args = args;
  wide_args.window_size = vt::AttentionWindow{32, 0};  // == the longest request
  vt::MlaPrefillAttention(q0, tow, nullptr, tq, tk, tv, tcq, tck, wide_args);
  for (size_t i = 0; i < full.size(); ++i) CHECK(full[i] == wide[i]);
}

// The CUDA half of the windowed prefill (dots3-note W4b-2, #699).
//
// WHY IT EXISTS SEPARATELY. The CPU case above gates `cpu_mla_prefill.cpp`, and
// `test_ops_mla_attn`'s "CUDA mla_decode: the sliding window matches the CPU
// reference" gates `cuda_mla_attn.cu`. Neither reaches the FA-2 MLA-prefill
// launcher's `is_local` normalization in `cuda_flash_attn_fa2.cu`, which is the
// OTHER CUDA file W4b-2 changed. Without this case that normalization has no
// gate on any device, and a later GPU lease would discharge the decode half
// while the record read as though it had closed both.
//
// IT SKIPS ON A BOX WITH NO CUDA DEVICE, which is where W4b-2 and this repair
// ran. A skip is not a pass: until the row's designated host is reachable this
// case has never EXECUTED, and the spec's §4.8 and `## Owed` say so.
//
// WHAT IT DOES NOT ASSERT, and why that is not a weaker bar. The CPU case
// asserts that a window at least as wide as the longest request is
// BIT-IDENTICAL to no window, because on CPU the absent state is a not-taken
// branch. That claim is FALSE on the GPU by construction and asserting it would
// be a defect: a finite `window_size` sets `p.is_causal = false` and dispatches
// FA-2's LOCAL specialization, a different compile-time template from the
// causal one, so agreement there is numerical rather than byte-wise.
TEST_CASE("CUDA MLA prefill: the sliding window matches the CPU reference") {
  if (!HasCuda()) return;
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(b);

  // The CPU case's fixture, unchanged, so the two halves are the same
  // experiment: 7+1+33+16 = 57 queries and a window of 5 that cuts every query
  // past position 4 in its own request.
  const std::vector<int32_t> q_lens{7, 1, 33, 16};
  const std::vector<int32_t> k_lens = q_lens;  // a fresh prompt: seq_len == query_len
  constexpr int kWindow = 5;
  const int h = kHeadsLite;
  const double scale = LiteScale();
  const std::vector<int32_t> cu_q = Cumsum(q_lens);
  const int total_q = cu_q.back();

  // The CUDA MLA prefill is bf16-only — the launcher instantiates
  // `cutlass::bfloat16_t` at every head-dim arm. Round-trip the inputs so the
  // CPU arm reads EXACTLY the values the kernel does; otherwise the comparison
  // charges bf16 input quantisation to the window.
  const auto qf = Bf16Round(RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, 909u));
  const auto kf = Bf16Round(RandF32(static_cast<size_t>(total_q) * h * kQkHeadDim, 911u));
  const auto vf = Bf16Round(RandF32(static_cast<size_t>(total_q) * h * kVHeadDim, 913u));

  // ── the CPU arm, windowed and unwindowed ─────────────────────────────────
  auto run_cpu = [&](std::optional<vt::AttentionWindow> win) {
    std::vector<float> q = qf, k = kf, v = vf;
    std::vector<int32_t> ca = cu_q, cb = cu_q;
    std::vector<float> out(static_cast<size_t>(total_q) * h * kVHeadDim,
                           std::numeric_limits<float>::quiet_NaN());
    Tensor tq = Contig(q.data(), DType::kF32, Cpu(), {total_q, h, kQkHeadDim});
    Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {total_q, h, kQkHeadDim});
    Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
    Tensor to = Contig(out.data(), DType::kF32, Cpu(), {total_q, h, kVHeadDim});
    Tensor tcq = Contig(ca.data(), DType::kI32, Cpu(), {static_cast<int64_t>(ca.size())});
    Tensor tck = Contig(cb.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cb.size())});
    MlaPrefillAttentionArgs args;
    args.scale = static_cast<float>(scale);
    args.causal = true;
    args.window_size = win;
    Queue q0 = CpuQ();
    vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcq, tck, args);
    return out;
  };
  const std::vector<float> cpu_win = run_cpu(vt::AttentionWindow{kWindow - 1, 0});
  const std::vector<float> cpu_none = run_cpu(std::nullopt);

  // ── the CUDA arm, windowed and unwindowed ────────────────────────────────
  const auto qb = ToBf16(qf);
  const auto kb = ToBf16(kf);
  const auto vb = ToBf16(vf);
  auto cu_v = cu_q;
  DeviceTensor dq(b, g.q, DType::kBF16, {total_q, h, kQkHeadDim}, qb.data());
  DeviceTensor dk(b, g.q, DType::kBF16, {total_q, h, kQkHeadDim}, kb.data());
  DeviceTensor dv(b, g.q, DType::kBF16, {total_q, h, kVHeadDim}, vb.data());
  DeviceTensor dcu(b, g.q, DType::kI32, {static_cast<int64_t>(cu_v.size())}, cu_v.data());

  auto run_cuda = [&](std::optional<vt::AttentionWindow> win) {
    // NaN-poison the output: a kernel that fails to write FAILS the gate
    // rather than reading back whatever the allocator handed us.
    const std::vector<uint16_t> poison(static_cast<size_t>(total_q) * h * kVHeadDim,
                                       vt::F32ToBF16(std::numeric_limits<float>::quiet_NaN()));
    DeviceTensor dout(b, g.q, DType::kBF16, {total_q, h, kVHeadDim}, poison.data());
    MlaPrefillAttentionArgs args;
    args.scale = static_cast<float>(scale);
    args.causal = true;
    args.window_size = win;
    vt::MlaPrefillAttention(g.q, dout.tensor(), nullptr, dq.tensor(), dk.tensor(),
                            dv.tensor(), dcu.tensor(), dcu.tensor(), args);
    b.Synchronize(g.q);
    std::vector<uint16_t> got(poison.size());
    dout.Download(g.q, got.data());
    return FromBf16(got);
  };
  const std::vector<float> gpu_win = run_cuda(vt::AttentionWindow{kWindow - 1, 0});
  const std::vector<float> gpu_none = run_cuda(std::nullopt);

  // The bar is the file's own CUDA-vs-reference bar: FA-2 accumulates in bf16
  // against a two-pass f32 CPU arm at these dims.
  CHECK(MaxAbsDiff(gpu_win, cpu_win) < 3e-2);

  // THE WINDOW HAS TO BITE ON THE DEVICE, not only on the CPU. Without this the
  // case above would pass on a launcher that dropped `window_size` on the floor
  // — which is the exact defect the `is_local` normalization exists to prevent,
  // because FA-2's `Is_causal` template IGNORES `window_size_left`.
  CHECK(MaxAbsDiff(gpu_win, gpu_none) > 1e-2);
  // …and the unwindowed device call still agrees with the unwindowed CPU one,
  // so the difference above is the window and not a broken device arm.
  CHECK(MaxAbsDiff(gpu_none, cpu_none) < 3e-2);

  // The independent semantic oracle, the same one the CPU case uses: every
  // query re-expressed as its own single-query request carrying only the keys
  // its window admits, run UNWINDOWED. This is what makes the comparison more
  // than "two spellings of one arithmetic".
  int64_t dropped = 0;
  const std::vector<float> expanded =
      RunExpandedWindowCpu(q_lens, k_lens, h, kWindow - 1, qf, kf, vf, scale, &dropped);
  MESSAGE("CUDA MLA prefill window " << kWindow << ": " << dropped
                                     << " (query, key) pairs dropped across " << total_q
                                     << " queries");
  REQUIRE(dropped > 0);
  CHECK(MaxAbsDiff(gpu_win, expanded) < 3e-2);
}

TEST_CASE("MLA prefill rejects a window shape upstream never produces") {
  const std::vector<int32_t> lens{4};
  const int h = 2;
  std::vector<int32_t> cu = Cumsum(lens);
  auto q = RandF32(4 * static_cast<size_t>(h) * kQkHeadDim, 1u);
  auto k = q;
  auto v = RandF32(4 * static_cast<size_t>(h) * kVHeadDim, 2u);
  std::vector<float> out(4 * static_cast<size_t>(h) * kVHeadDim, 0.0f);
  Tensor tq = Contig(q.data(), DType::kF32, Cpu(), {4, h, kQkHeadDim});
  Tensor tk = Contig(k.data(), DType::kF32, Cpu(), {4, h, kQkHeadDim});
  Tensor tv = Contig(v.data(), DType::kF32, Cpu(), {4, h, kVHeadDim});
  Tensor to = Contig(out.data(), DType::kF32, Cpu(), {4, h, kVHeadDim});
  Tensor tcq = Contig(cu.data(), DType::kI32, Cpu(), {static_cast<int64_t>(cu.size())});
  Queue q0 = CpuQ();
  MlaPrefillAttentionArgs args;
  args.scale = 0.1f;
  args.causal = true;
  args.window_size = vt::AttentionWindow{2, 1};
  CHECK_THROWS_WITH_AS(vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcq, tcq, args),
                       doctest::Contains("window_size.right"), std::runtime_error);
  // A NON-causal window has no FlashAttention spelling — the local mask
  // REPLACES the causal specialization, so "everything forward, windowed
  // backward" would need an infinite right bound. Upstream never asks for one.
  args.window_size = vt::AttentionWindow{2, 0};
  args.causal = false;
  CHECK_THROWS_WITH_AS(vt::MlaPrefillAttention(q0, to, nullptr, tq, tk, tv, tcq, tcq, args),
                       doctest::Contains("causal=true"), std::runtime_error);
}
