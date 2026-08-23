// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror —
// upstream's equivalent coverage is FlashAttention-2's own `tests/test_flash_attn.py`
// (vllm-project/flash-attention @ 2c839c33, `test_flash_attn_output` non-causal
// d in {64, 128}), which cannot be ported directly because it is a torch/pytest
// harness comparing against `torch.nn.functional.scaled_dot_product_attention`. The
// adaptation is: same shape family (dense, non-paged, b=1, non-causal, head_dim 64
// or 128, bf16), and the reference is our own `vt::AttentionDenseFlash` instead of
// SDPA. Upstream's 2x-REFERENCE-ERROR RULE IS PORTED rather than replaced, at
// head_dim 128, in case 1c: `RefF64` below stands in for torch's higher-precision
// `out_ref` and `AttentionDenseFlash` for its dtype-matched `out_pt`, and the factor
// of 2 is upstream's. The bf16-envelope rel-L2 bound is kept ALONGSIDE it as the
// cheap arm-to-arm check that runs at every shape.
//
// WHAT THIS FILE GUARDS — `vt::AttentionDenseFa2` (`OpId::kAttentionDenseFa2`,
// multimodal-speed.md §17, issue #432). PR #439's review found the op shipped with
// ZERO tests, and demonstrated two defects that survived every gate in the tree:
//
//   M2a  Corrupt the params filler so the kernel attends only HALF the keys
//        (`p.seqlen_k = t/2`). No exception, no diagnostic, plausible tokens, and
//        the shipping default arm — which does not use this op — stayed 16/16.
//        Killed here by `fa2 dense attends the FULL key range`, which perturbs only
//        the tail of V and REQUIRES the output to move.
//
//   M3   Drop `!args.causal` from the dispatch gate and ask for causal attention.
//        The launcher hardcoded `p.is_causal = false`, so it returned an output
//        BIT-IDENTICAL to the non-causal one and refused nothing. Killed here by
//        `attention-dense-fa2 falls through for CAUSAL (M3)`, which requires
//        bit-equality with the causal reference. The launcher now also takes
//        `bool causal` and throws, so the mutation surfaces as a refusal; with the
//        guard also removed it surfaces as the bit-equality failure. Both are red.
//
// The op is TOTAL: its fast path is bf16 + head_dim in {64, 128} + non-causal + MHA
// + FA-2 compiled, and every other shape must fall through to `AttentionDenseFlash`
// BIT-exactly. Both halves are covered below.
//
// HEAD_DIM 128 was added by #1551 for the LTX-2.5 DiT, whose video stream is 32
// heads x 128 over 2352 tokens and which could not reach a tensor core while this
// op refused every head dim but 64. It is a SECOND compiled instantiation
// (`run_mha_fwd_<bfloat16_t, 128, false>`) with its own kernel traits and its own
// shared-memory request, reached through its own call site, so it is covered by its
// own cases rather than by the hd-64 ones: parity, the full-key-range killer, the
// causal refusal, the A/B knob, and the ported upstream tolerance rule. The last of
// those is the one that makes adoption defensible rather than merely measured — see
// case 1c.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <stdlib.h>  // setenv/unsetenv (POSIX; not in <cstdlib>'s guaranteed set)

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::AttentionArgs;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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
    t_ = MakeT(p_, dt, Gpu(), shape);
  }
  ~DeviceTensor() { b_.Free(p_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;
  Tensor& tensor() { return t_; }
  void Upload(Queue& q, const void* src) { b_.Copy(q, p_, src, bytes_); }
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

// Deterministic LCG in [-1,1) — same shape as test_ops_attention.cpp's RandF32, so
// the two files cannot drift on <random> implementation differences.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
  }
  return v;
}

std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> b(f.size());
  for (size_t i = 0; i < f.size(); ++i) b[i] = vt::F32ToBF16(f[i]);
  return b;
}

std::vector<float> FromBf16(const std::vector<uint16_t>& b) {
  std::vector<float> f(b.size());
  for (size_t i = 0; i < b.size(); ++i) f[i] = vt::BF16ToF32(b[i]);
  return f;
}

double RelL2(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return std::sqrt(num / (den + 1e-30));
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return m;
}

// Number of positions where two results differ at all. Used instead of
// `CHECK(a == b)` on the vectors: doctest stringifies both operands of a failing
// CHECK, and dumping two 65,000-element vectors buries the actual signal (it did,
// on the first M3-silent mutation run).
size_t Mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) ++n;
  return n;
}

double Rms(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  return std::sqrt(s / static_cast<double>(v.empty() ? 1 : v.size()));
}

// A DOUBLE-PRECISION host reference for one dense non-causal MHA attention, from
// the same bf16 inputs the device arms read. This exists because the upstream
// tolerance rule ported below is stated against a more-accurate reference, not
// against the incumbent kernel: `tests/test_flash_attn.py::test_flash_attn_output`
// (vllm-project/flash-attention @ 2c839c33) asserts
// `(out - out_ref).abs().max() <= 2 * (out_pt - out_ref).abs().max()`, where
// `out_ref` is the fp32/fp64 reference and `out_pt` is the reference kernel run at
// the tested dtype. Our adaptation names our own `AttentionDenseFlash` as `out_pt`
// and computes `out_ref` here in `double` instead of calling torch.
//
// COST is why the cases below run it at a REDUCED head count and never at the full
// 32: this is O(T^2 * D) per head in scalar double, and T is 2352.
std::vector<double> RefF64(const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                           const std::vector<uint16_t>& v, int64_t T, int64_t H, int64_t D,
                           float scale) {
  std::vector<double> out(static_cast<size_t>(T * H * D), 0.0);
  std::vector<double> p(static_cast<size_t>(T));
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t i = 0; i < T; ++i) {
      const size_t qo = static_cast<size_t>((i * H + h) * D);
      double m = -std::numeric_limits<double>::infinity();
      for (int64_t j = 0; j < T; ++j) {
        const size_t ko = static_cast<size_t>((j * H + h) * D);
        double dot = 0.0;
        for (int64_t d = 0; d < D; ++d)
          dot += static_cast<double>(vt::BF16ToF32(q[qo + static_cast<size_t>(d)])) *
                 static_cast<double>(vt::BF16ToF32(k[ko + static_cast<size_t>(d)]));
        const double sc = dot * static_cast<double>(scale);
        p[static_cast<size_t>(j)] = sc;
        if (sc > m) m = sc;
      }
      double den = 0.0;
      for (int64_t j = 0; j < T; ++j) {
        p[static_cast<size_t>(j)] = std::exp(p[static_cast<size_t>(j)] - m);
        den += p[static_cast<size_t>(j)];
      }
      const double inv = 1.0 / den;
      for (int64_t j = 0; j < T; ++j) {
        const double w = p[static_cast<size_t>(j)] * inv;
        if (w == 0.0) continue;
        const size_t vo = static_cast<size_t>((j * H + h) * D);
        for (int64_t d = 0; d < D; ++d)
          out[qo + static_cast<size_t>(d)] +=
              w * static_cast<double>(vt::BF16ToF32(v[vo + static_cast<size_t>(d)]));
      }
    }
  }
  return out;
}

double MaxAbsDiffVsRef(const std::vector<float>& a, const std::vector<double>& ref) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - ref[i]));
  return m;
}

double RelL2VsRef(const std::vector<float>& a, const std::vector<double>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return std::sqrt(num / (den + 1e-30));
}

enum class Op { kFa2, kFlash };

// Run one dense attention through either op on bf16 [T,H,D] host data, returning the
// bf16 output decoded to f32.
std::vector<float> RunBf16(Op op, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
                           const std::vector<uint16_t>& v, int64_t T, int64_t Hq, int64_t Hk,
                           int64_t D, float scale, bool causal) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kBF16, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kBF16, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kBF16, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kBF16, {T, Hq, D});
  const AttentionArgs args{scale, causal};
  if (op == Op::kFa2)
    vt::AttentionDenseFa2(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  else
    vt::AttentionDenseFlash(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  std::vector<uint16_t> got(static_cast<size_t>(T * Hq * D), 0);
  dout.Download(g.q, got.data());
  return FromBf16(got);
}

std::vector<float> RunF32(Op op, const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, int64_t T, int64_t Hq, int64_t Hk,
                          int64_t D, float scale, bool causal) {
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard g(gpu);
  DeviceTensor dq(gpu, g.q, DType::kF32, {T, Hq, D}, q.data());
  DeviceTensor dk(gpu, g.q, DType::kF32, {T, Hk, D}, k.data());
  DeviceTensor dv(gpu, g.q, DType::kF32, {T, Hk, D}, v.data());
  DeviceTensor dout(gpu, g.q, DType::kF32, {T, Hq, D});
  const AttentionArgs args{scale, causal};
  if (op == Op::kFa2)
    vt::AttentionDenseFa2(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  else
    vt::AttentionDenseFlash(g.q, dout.tensor(), dq.tensor(), dk.tensor(), dv.tensor(), args);
  std::vector<float> got(static_cast<size_t>(T * Hq * D), 0.0f);
  dout.Download(g.q, got.data());
  return got;
}

// The bf16 envelope. FA-2 and the scalar flash kernel compute the same function with
// different reduction orders AND different softmax formulations (exp2f on a
// log2-scaled score vs expf on a linearly-scaled one), and both store bf16, whose
// relative resolution is 2^-8 = 3.9e-3. A rel-L2 of a few e-3 is the expected
// agreement; anything approaching 1e-2 is a real defect. The M2a mutation (attend
// half the keys) lands at rel-L2 ~ O(1) against these bounds.
constexpr double kRelL2Bound = 1.0e-2;
constexpr double kMaxAbsVsRmsBound = 0.15;

}  // namespace

// ===========================================================================
// 1. The FAST PATH: bf16, head_dim 64, non-causal, MHA — the Whisper encoder shape.
//    FA-2 must agree with the shipping byte-exact scalar kernel inside the bf16
//    envelope.
TEST_CASE("attention-dense-fa2 bf16 hd-64 non-causal MHA matches AttentionDenseFlash") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 parity");
    return;
  }
  // (T, H): the real Voxtral/Whisper-large encoder geometry (1500 positions, 20
  // heads of 64) plus two shapes that are NOT multiples of FA-2's 128-wide K block,
  // so the epilogue masking is exercised too.
  const std::vector<std::pair<int64_t, int64_t>> shapes = {{1500, 20}, {257, 4}, {17, 2}};
  const int64_t D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (const auto& [T, H] : shapes) {
    const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 1000 + static_cast<uint32_t>(T)));
    const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 2000 + static_cast<uint32_t>(T)));
    const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 3000 + static_cast<uint32_t>(T)));

    const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, /*causal=*/false);
    const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, /*causal=*/false);

    const double rel = RelL2(got, ref);
    const double mad = MaxAbsDiff(got, ref);
    const double rms = Rms(ref);
    MESSAGE("T=", T, " H=", H, " D=64 non-causal: rel-L2 ", rel, "  max|diff| ", mad,
            "  (rms(ref) ", rms, ")");
    CHECK(rel < kRelL2Bound);
    CHECK(mad < kMaxAbsVsRmsBound * rms);
  }
}

// ===========================================================================
// 1b. THE FAST PATH AT head_dim 128 — LTX-2.5's DiT self-attention (#1551).
//     Until #1551 this op refused every head dim but 64, so the LTX-2.5 video
//     stream (32 heads x 128 over 2352 tokens at 768x448/49f) fell through to the
//     scalar `AttentionDenseFlash` and never reached a tensor core. The extra
//     `run_mha_fwd_<bfloat16_t, 128, false>` instantiation
//     (src/vt/cuda/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu) is what changed.
//
//     THE GEOMETRY IS THE MODEL'S OWN, not a convenient small one. 2352 is not a
//     multiple of FA-2's K block, so the epilogue masking is exercised at the
//     production length rather than only at a fixture length.
TEST_CASE("attention-dense-fa2 bf16 hd-128 non-causal MHA matches AttentionDenseFlash") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 hd-128 parity");
    return;
  }
  // (T, H): LTX-2.5 at 768x448/49f is 2352 video tokens x 32 heads. The two
  // smaller shapes are NOT multiples of FA-2's 128-wide M block or 64-wide N
  // block, so the ragged epilogue is covered as well.
  const std::vector<std::pair<int64_t, int64_t>> shapes = {{2352, 32}, {577, 3}, {17, 2}};
  const int64_t D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  for (const auto& [T, H] : shapes) {
    const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 5000 + static_cast<uint32_t>(T)));
    const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 6000 + static_cast<uint32_t>(T)));
    const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 7000 + static_cast<uint32_t>(T)));

    const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, /*causal=*/false);
    const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, /*causal=*/false);

    const double rel = RelL2(got, ref);
    const double mad = MaxAbsDiff(got, ref);
    const double rms = Rms(ref);
    MESSAGE("T=", T, " H=", H, " D=128 non-causal: rel-L2 ", rel, "  max|diff| ", mad,
            "  (rms(ref) ", rms, ")");
    // The SAME committed bounds the hd-64 case uses, and deliberately not new
    // numbers. `kRelL2Bound` is a property of the bf16 OUTPUT DTYPE — 1e-2 is
    // about 2.5 bf16 ulps, and bf16's relative resolution is 2^-8 = 3.9e-3 — not
    // a property of a head dim, a sequence length or a measurement. Widening it
    // for this head dim would be fitting the bound to the result.
    CHECK(rel < kRelL2Bound);
    CHECK(mad < kMaxAbsVsRmsBound * rms);
  }
}

// ===========================================================================
// 1c. THE PORTED UPSTREAM TOLERANCE RULE, at head_dim 128.
//     `CHECK(rel < kRelL2Bound)` above compares the two device arms to each other,
//     which bounds how far they DIVERGE and says nothing about which one is right.
//     Both could be wrong together. Upstream FA-2 answers that with a rule stated
//     against a more accurate reference —
//     `tests/test_flash_attn.py::test_flash_attn_output`
//     (vllm-project/flash-attention @ 2c839c33):
//
//         assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
//
//     `out_ref` is the higher-precision reference, `out_pt` the reference kernel at
//     the tested dtype, `out` the FA-2 result. The adaptation is the harness only:
//     `out_ref` is `RefF64` above instead of a torch fp32 SDPA, and `out_pt` is our
//     own `AttentionDenseFlash` instead of torch's bf16 SDPA. The FACTOR OF 2 and
//     the max-abs statistic are upstream's and are not re-derived here.
//
//     This is the rule that makes hd-128 adoption defensible rather than merely
//     measured: FA-2 must be AT LEAST AS ACCURATE as the kernel it replaces, to
//     within upstream's own factor. A tolerance fitted to the observed deviation
//     could not distinguish "agrees closely" from "both drifted the same way".
//
//     REDUCED HEAD COUNT, stated rather than hidden: `RefF64` is O(T^2 * D) scalar
//     double per head and T is the production 2352, so H is 2 rather than 32. The
//     SEQUENCE LENGTH — the axis the reduction order actually accumulates over — is
//     the production one. Heads are independent in this op, so the head count is
//     the cheap axis to reduce and the sequence length is not.
TEST_CASE("attention-dense-fa2 hd-128 is at least as accurate as the kernel it replaces") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 hd-128 upstream tolerance rule");
    return;
  }
  const int64_t T = 2352, H = 2, D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 8101));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 8102));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 8103));

  const auto ref = RefF64(q, k, v, T, H, D, scale);
  const auto pt = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, /*causal=*/false);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, /*causal=*/false);

  const double err_fa2 = MaxAbsDiffVsRef(got, ref);
  const double err_flash = MaxAbsDiffVsRef(pt, ref);
  MESSAGE("T=", T, " H=", H, " D=128 vs f64 reference: max|fa2-ref| ", err_fa2,
          "  max|flash-ref| ", err_flash, "  ratio ",
          (err_flash > 0.0 ? err_fa2 / err_flash : -1.0));
  MESSAGE("  rel-L2 vs f64 reference: fa2 ", RelL2VsRef(got, ref), "  flash ",
          RelL2VsRef(pt, ref));
  // The incumbent must have a NON-ZERO error against the f64 reference, or the
  // rule below is `x <= 0` and passes only for an exactly-equal arm — a bound of
  // zero would be a mute switch dressed as a strict gate.
  REQUIRE(err_flash > 0.0);
  CHECK(err_fa2 <= 2.0 * err_flash);  // upstream's rule, upstream's factor
}

// ===========================================================================
// 2. THE M2a KILLER — the op must attend the ENTIRE key range.
//    Reference-free: perturb ONLY the tail of V and require the output to move by
//    much more than the bf16 envelope. A params filler that clamps `seqlen_k`, or a
//    launcher that mis-strides K/V, silently drops those keys and this case goes red
//    while every token gate in the tree stays green (review of PR #439, M2a).
TEST_CASE("attention-dense-fa2 attends the FULL key range (M2a: p.seqlen_k = t/2)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 key-range coverage");
    return;
  }
  const int64_t T = 1500, H = 4, D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 4242));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 4243));
  auto vf = RandF32(static_cast<size_t>(T * H * D), 4244);
  const auto v_base = ToBf16(vf);

  // Perturb the SECOND HALF of the value rows only, by a large, unmistakable amount.
  for (int64_t t = T / 2; t < T; ++t)
    for (int64_t i = 0; i < H * D; ++i) vf[static_cast<size_t>(t * H * D + i)] += 8.0f;
  const auto v_tail = ToBf16(vf);

  const auto out_base = RunBf16(Op::kFa2, q, k, v_base, T, H, H, D, scale, false);
  const auto out_tail = RunBf16(Op::kFa2, q, k, v_tail, T, H, H, D, scale, false);

  const double moved = RelL2(out_tail, out_base);
  MESSAGE("perturbing V rows [", T / 2, ",", T, ") moved the output by rel-L2 ", moved,
          " (bf16 envelope is ", kRelL2Bound, ")");
  // With random q/k at hd-64 the softmax mass is spread over all 1500 keys, so
  // roughly half of it sits in the perturbed tail and the output must shift by O(1).
  // A kernel attending only keys [0, T/2) would shift by EXACTLY zero.
  CHECK(moved > 100.0 * kRelL2Bound);

  // And the scalar reference must see the same shift, so the case is measuring the
  // key range and not some FA-2-specific artefact.
  const auto ref_base = RunBf16(Op::kFlash, q, k, v_base, T, H, H, D, scale, false);
  const auto ref_tail = RunBf16(Op::kFlash, q, k, v_tail, T, H, H, D, scale, false);
  MESSAGE("  scalar reference moved by rel-L2 ", RelL2(ref_tail, ref_base));
  CHECK(RelL2(out_tail, ref_tail) < kRelL2Bound);
}

// The SAME killer at head_dim 128, and it is not redundant. The two head dims are
// SEPARATE template instantiations reached through separate call sites in
// `LaunchDenseFA2Bf16`, with different kernel traits (`Flash_fwd_kernel_traits<128,
// 128, 64, ...>` against `<64, 128, 128, ...>`) and a shared-memory request that
// crosses CUDA's default 48 KiB cap where hd-64's does not. A stride or seqlen
// defect can exist in one and not the other, so the hd-64 case above does not cover
// this one. LTX-2.5's production length is used rather than Whisper's.
TEST_CASE("attention-dense-fa2 hd-128 attends the FULL key range") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 hd-128 key-range coverage");
    return;
  }
  const int64_t T = 2352, H = 4, D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 9242));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 9243));
  auto vf = RandF32(static_cast<size_t>(T * H * D), 9244);
  const auto v_base = ToBf16(vf);
  for (int64_t t = T / 2; t < T; ++t)
    for (int64_t i = 0; i < H * D; ++i) vf[static_cast<size_t>(t * H * D + i)] += 8.0f;
  const auto v_tail = ToBf16(vf);

  const auto out_base = RunBf16(Op::kFa2, q, k, v_base, T, H, H, D, scale, false);
  const auto out_tail = RunBf16(Op::kFa2, q, k, v_tail, T, H, H, D, scale, false);
  const double moved = RelL2(out_tail, out_base);
  MESSAGE("hd-128: perturbing V rows [", T / 2, ",", T, ") moved the output by rel-L2 ", moved);
  CHECK(moved > 100.0 * kRelL2Bound);

  const auto ref_tail = RunBf16(Op::kFlash, q, k, v_tail, T, H, H, D, scale, false);
  CHECK(RelL2(out_tail, ref_tail) < kRelL2Bound);
}

// The M3 killer at head_dim 128. `run_mha_fwd_<bfloat16_t, 128, false>` is the only
// hd-128 non-split instantiation compiled, so a causal request MUST leave this op
// rather than be answered non-causally. The hd-64 case proves the dispatch gate's
// `!args.causal` term; this one proves the gate is not accidentally head-dim-scoped
// — a gate written as `(d == 64 && !causal) || d == 128` would pass that one and
// fail this one.
TEST_CASE("attention-dense-fa2 hd-128 falls through for CAUSAL") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 hd-128 causal fall-through");
    return;
  }
  const int64_t T = 257, H = 4, D = 128;
  const float s128 = 1.0f / std::sqrt(128.0f);
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 151));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 152));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 153));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s128, /*causal=*/true);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, s128, /*causal=*/true);
  CHECK(Mismatches(got, ref) == 0);
  const auto noncausal = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s128, /*causal=*/false);
  CHECK(RelL2(ref, noncausal) > 100.0 * kRelL2Bound);
}

// ===========================================================================
// 3. TOTALITY / FALL-THROUGH — every shape outside the narrow fast path must reach
//    `AttentionDenseFlash`, and therefore be BIT-identical to calling it directly.
//
//    Four SEPARATE test cases, not four SUBCASEs of one: an uncaught exception ends
//    the whole enclosing TEST_CASE and doctest skips its remaining subcases. The M3
//    mutation throws, and under a single test case that silently dropped the GQA,
//    hd!=64 and f32 coverage from the run (14 assertions became 9). Separate cases
//    keep one failure from hiding three others.
//
//    The causal case is the M3 killer: with the dispatch gate's `!args.causal`
//    removed, FA-2 answered the NON-causal question instead and refused nothing.
TEST_CASE("attention-dense-fa2 falls through for CAUSAL (M3)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 causal fall-through");
    return;
  }
  // bf16, hd 64, MHA — everything the fast path wants EXCEPT non-causality.
  const int64_t T = 257, H = 4, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 51));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 52));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 53));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s64, /*causal=*/true);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, s64, /*causal=*/true);
  CHECK(Mismatches(got, ref) == 0);  // bit-exact: same kernel, same args
  // The causal answer must NOT be the non-causal one — otherwise "bit-exact vs the
  // reference" would be satisfiable by a kernel that ignores the mask entirely.
  const auto noncausal = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s64, /*causal=*/false);
  MESSAGE("causal vs non-causal rel-L2 (must be large): ", RelL2(ref, noncausal));
  CHECK(RelL2(ref, noncausal) > 100.0 * kRelL2Bound);
}

TEST_CASE("attention-dense-fa2 falls through for GQA (h_k != h)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 GQA fall-through");
    return;
  }
  const int64_t T = 96, Hq = 8, Hk = 2, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * Hq * D), 61));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * Hk * D), 62));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * Hk * D), 63));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, Hq, Hk, D, s64, false);
  const auto got = RunBf16(Op::kFa2, q, k, v, T, Hq, Hk, D, s64, false);
  CHECK(Mismatches(got, ref) == 0);
}

TEST_CASE("attention-dense-fa2 falls through for head_dim outside {64, 128}") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 head_dim fall-through");
    return;
  }
  // The served set is the set of COMPILED non-split instantiations, {64, 128}, and
  // it is not an interval. 80 sits between them and 192 above them; both must leave
  // this op BIT-exactly. Testing only 80 would pass a gate written `d <= 128`, and
  // testing only 192 would pass one written `d != 80`.
  const std::vector<int64_t> dims = {80, 192};
  const int64_t T = 96, H = 3;
  for (int64_t D : dims) {
    const float s = 1.0f / std::sqrt(static_cast<float>(D));
    const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 71));
    const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 72));
    const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 73));
    const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, s, false);
    const auto got = RunBf16(Op::kFa2, q, k, v, T, H, H, D, s, false);
    MESSAGE("head_dim ", D, " must fall through bit-exactly");
    CHECK(Mismatches(got, ref) == 0);
  }
}

TEST_CASE("attention-dense-fa2 falls through for f32") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping dense FA-2 f32 fall-through");
    return;
  }
  const int64_t T = 96, H = 3, D = 64;
  const float s64 = 1.0f / std::sqrt(64.0f);
  const auto q = RandF32(static_cast<size_t>(T * H * D), 81);
  const auto k = RandF32(static_cast<size_t>(T * H * D), 82);
  const auto v = RandF32(static_cast<size_t>(T * H * D), 83);
  const auto ref = RunF32(Op::kFlash, q, k, v, T, H, H, D, s64, false);
  const auto got = RunF32(Op::kFa2, q, k, v, T, H, H, D, s64, false);
  CHECK(Mismatches(got, ref) == 0);
}

// ===========================================================================
// 4. THE A/B KNOB — `VT_FA2_DENSE=0` is the same-binary rollback arm recorded in
//    docs/ENVIRONMENT.md and used for every §17 measurement. If it stopped
//    selecting the scalar kernel, every A/B in the record would silently be
//    comparing the FA-2 arm against itself.
TEST_CASE("attention-dense-fa2 VT_FA2_DENSE=0 restores the scalar kernel bit-exactly") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping VT_FA2_DENSE A/B knob");
    return;
  }
  const int64_t T = 257, H = 4, D = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 91));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 92));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 93));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, false);

  const char* prev = getenv("VT_FA2_DENSE");
  const std::string saved = prev == nullptr ? std::string() : std::string(prev);
  const bool had = prev != nullptr;
  (void)setenv("VT_FA2_DENSE", "0", 1);
  const auto off = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, false);
  if (had)
    (void)setenv("VT_FA2_DENSE", saved.c_str(), 1);
  else
    (void)unsetenv("VT_FA2_DENSE");

  CHECK(Mismatches(off, ref) == 0);  // the knob really routes back to AttentionDenseFlash
}

// The same knob at head_dim 128, because #1551's A/B is taken at that head dim and
// a knob that only covered hd-64 would leave that measurement comparing the FA-2
// arm against itself — the exact failure this case family exists to prevent.
TEST_CASE("attention-dense-fa2 VT_FA2_DENSE=0 restores the scalar kernel at hd-128") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend; skipping VT_FA2_DENSE A/B knob at hd-128");
    return;
  }
  const int64_t T = 257, H = 4, D = 128;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto q = ToBf16(RandF32(static_cast<size_t>(T * H * D), 191));
  const auto k = ToBf16(RandF32(static_cast<size_t>(T * H * D), 192));
  const auto v = ToBf16(RandF32(static_cast<size_t>(T * H * D), 193));
  const auto ref = RunBf16(Op::kFlash, q, k, v, T, H, H, D, scale, false);
  const auto on = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, false);

  const char* prev = getenv("VT_FA2_DENSE");
  const std::string saved = prev == nullptr ? std::string() : std::string(prev);
  const bool had = prev != nullptr;
  (void)setenv("VT_FA2_DENSE", "0", 1);
  const auto off = RunBf16(Op::kFa2, q, k, v, T, H, H, D, scale, false);
  if (had)
    (void)setenv("VT_FA2_DENSE", saved.c_str(), 1);
  else
    (void)unsetenv("VT_FA2_DENSE");

  CHECK(Mismatches(off, ref) == 0);
  // And the ON arm must NOT be that same scalar answer, or the knob is measuring
  // one kernel twice at this head dim whatever it does at the other.
  MESSAGE("hd-128 knob: ON-vs-flash mismatching elements = ", Mismatches(on, ref));
  CHECK(Mismatches(on, ref) > 0);
}
