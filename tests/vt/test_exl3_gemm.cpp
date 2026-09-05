// EXL3 (exllamav3 trellis) DEVICE kernels — MODEL-DSV4-EXL3 W2a + W2b.
//
// PORTED FROM exllamav3 @ 2398c05635fbbad01a0a51dce63c85c6c8a8450e (MIT). vLLM
// implements no EXL3 at the parity pin, so exllamav3 is this format's secondary
// oracle; see `.agents/specs/model-dsv4-exl3.md`, whose `## W2 design` section
// states the parity contract this file gates, tier by tier, WITH ITS NUMBERS.
//
// WHAT IS GATED HERE, AND ON WHAT MACHINE.
//
//   * The kernel-SHAPE policy (`exl3_kernel_map.cu:23-91,153-160`) is pure host
//     arithmetic and is gated on any machine, GPU or not. A selection table that
//     only a device run can check is a table nobody checks.
//   * `Exl3HadR128`'s CPU arm is gated against an INDEPENDENT Sylvester-H128
//     reference built here from popcount parity — not from the implementation's
//     own butterfly, which would be a transcription gating its own
//     transcription.
//   * `Exl3Gemm`'s CPU arm is gated against a `double` evaluation of the same
//     chain (decode -> had(x, suh) -> @ W_inner -> had(y, svh)) at the bound the
//     spec states: RMS relative 1.0e-3, elementwise 8 fp16 ulps of the output
//     RMS.
//   * The device arms are gated against the CPU arms — BYTE-identically for
//     `Exl3HadR128` (§1 tier 2: the two run the same f32 operations in the same
//     order) and at the tier-3 bound for `Exl3Gemm`. Those cases SKIP when no
//     CUDA backend is registered, and they say so out loud AND still assert, so
//     the suite can never report `assertions: 0` — a skip wearing a pass is a
//     trap this row has already paid for once.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "exl3_fixture.h"

namespace {

// The fixtures and the INDEPENDENT references now live in one place, shared
// with test_exl3_gemv and test_exl3_moe (tests/vt/exl3_fixture.h). Nothing
// under test moved: SylvesterH is still built from popcount parity and
// HadRefBlock still from the hadamard.cu docstring.
using exl3_test::Exl3ChainF64;
using exl3_test::Exl3Fixture;
using exl3_test::HadRefBlock;
using exl3_test::MakeFixture;
using exl3_test::Rms;
using exl3_test::Rng;
using exl3_test::SylvesterH;
using exl3_test::UlpF16;


bool HasCuda() {
  try {
    (void)vt::GetBackend(vt::DeviceType::kCUDA);
    return vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
}

vt::Queue CpuQueue() { return vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }

}  // namespace

// ─── §3 of the spec: the shape policy, host-side ─────────────────────────────

TEST_CASE("exl3 policy: the shape table is upstream's, value for value") {
  // exl3_kernel_map.cuh:53-60. Transcribed here from the MACROS, independently
  // of the implementation's own table.
  //   #define EXL3_GEMM_SHAPE_1  16, 16, 128,  6, 5
  //   #define EXL3_GEMM_SHAPE_2  16, 32, 128,  4, 3
  //   #define EXL3_GEMM_SHAPE_3  16, 32, 256,  4, 3
  //   #define EXL3_GEMM_SHAPE_4  16, 16, 512,  4, 3
  //   #define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256
  REQUIRE(vt::Exl3GemmNumShapes() == 4);
  struct Row {
    int idx, m, k, n, sh, frag, bd;
  };
  const Row rows[4] = {
      {1, 16, 16, 128, 6, 5, 256},
      {2, 16, 32, 128, 4, 3, 512},
      {3, 16, 32, 256, 4, 3, 512},
      {4, 16, 16, 512, 4, 3, 256},
  };
  for (const Row& r : rows) {
    const vt::Exl3GemmShape s = vt::Exl3GemmShapeParams(r.idx);
    CHECK(s.tile_m == r.m);
    CHECK(s.tile_k == r.k);
    CHECK(s.tile_n == r.n);
    CHECK(s.sh_stages == r.sh);
    CHECK(s.frag_stages == r.frag);
    CHECK(s.block_dim == r.bd);
  }
}

TEST_CASE("exl3 policy: the compute-capability bucket is upstream's five-way split") {
  // exl3_devctx.cu:39-43. GB10 is sm_121 -> major 12 -> Blackwell; the row's
  // gate arch is therefore NOT a new bucket.
  CHECK(vt::Exl3CcFromSm(12, 1) == vt::Exl3Cc::kBlackwell);
  CHECK(vt::Exl3CcFromSm(10, 0) == vt::Exl3Cc::kBlackwell);
  CHECK(vt::Exl3CcFromSm(9, 0) == vt::Exl3Cc::kHopper);
  CHECK(vt::Exl3CcFromSm(8, 9) == vt::Exl3Cc::kAda);
  CHECK(vt::Exl3CcFromSm(8, 6) == vt::Exl3Cc::kAmpere);
  CHECK(vt::Exl3CcFromSm(8, 0) == vt::Exl3Cc::kAmpere);
  CHECK(vt::Exl3CcFromSm(7, 5) == vt::Exl3Cc::kOld);
}

TEST_CASE("exl3 policy: this checkpoint's expert shapes select shape 2 on GB10") {
  // The spec's `## W2 design` §3 table, and the claim a device run must agree
  // with. w1/w3 are k=4096 n=2048; w2 is k=2048 n=4096 (TP1-coalesced, measured
  // from the real shard header). K = 3, cb = mcg, multi = false.
  CHECK(vt::Exl3SelectGemmShape(vt::Exl3Cc::kBlackwell, 1, 4096, 2048, 3, false) == 2);
  CHECK(vt::Exl3SelectGemmShape(vt::Exl3Cc::kBlackwell, 16, 4096, 2048, 3, false) == 2);
  CHECK(vt::Exl3SelectGemmShape(vt::Exl3Cc::kBlackwell, 1, 2048, 4096, 3, false) == 2);
  // Both are compatible with the shape they select (k % 32, n % 128).
  CHECK(vt::Exl3GemmShapeCompat(2, 4096, 2048));
  CHECK(vt::Exl3GemmShapeCompat(2, 2048, 4096));
}

TEST_CASE("exl3 policy: the other branches of select_gemm_shape are upstream's too") {
  // exl3_kernel_map.cu:31-73, transcribed branch by branch. These are the rows
  // a DSV4 run never takes, and they are gated so a later shape edit cannot
  // quietly change them.
  using vt::Exl3Cc;
  using vt::Exl3SelectGemmShape;
  // Ampere, mod_256 && K <= 4: small -> 2, large -> 3.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kAmpere, 1, 2048, 2048, 3, false) == 2);
  CHECK(Exl3SelectGemmShape(Exl3Cc::kAmpere, 1, 4096, 4096, 3, false) == 3);
  // Ampere, K >= 5: mod_512 and a big product -> 4.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kAmpere, 1, 8192, 8192, 6, false) == 4);
  // Ada, K <= 3, small k -> 2; big k -> 3.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kAda, 1, 2048, 8192, 3, false) == 2);
  CHECK(Exl3SelectGemmShape(Exl3Cc::kAda, 1, 16384, 8192, 3, false) == 3);
  // Blackwell, K == 4 and small k, non-multi -> the otherwise-unused shape 1.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kBlackwell, 1, 2048, 4096, 4, false) == 1);
  CHECK(Exl3SelectGemmShape(Exl3Cc::kBlackwell, 1, 2048, 4096, 4, true) == 2);
  // Blackwell, K >= 7, big mod_512 n -> 4.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kBlackwell, 1, 4096, 65536, 7, false) == 4);
  // Blackwell, K = 3, big k and n <= 4096 -> 3.
  CHECK(Exl3SelectGemmShape(Exl3Cc::kBlackwell, 1, 16384, 4096, 3, false) == 3);
}

TEST_CASE("exl3 policy: shape compatibility and the empty-block clamp") {
  // exl3_kernel_map.cu:86-91: k must divide the tile's k, n the tile's n.
  CHECK_FALSE(vt::Exl3GemmShapeCompat(2, 4080, 2048));  // 4080 % 32 != 0
  CHECK_FALSE(vt::Exl3GemmShapeCompat(3, 4096, 2048 + 16));
  CHECK(vt::Exl3GemmShapeCompat(4, 4096, 2048));  // n % 512 == 0
  CHECK_FALSE(vt::Exl3GemmShapeCompat(4, 4096, 2048 + 256));
  // exl3_kernel_map.cu:153-160: never more blocks than k x n tiles, never zero.
  //   shape 2 over k=4096 n=2048: 128 * 16 = 2048 slices, so 48 SMs survive.
  CHECK(vt::Exl3GemmNumSms(2, 4096, 2048, 48) == 48);
  //   a tiny problem clamps DOWN: k=32 n=128 is one slice.
  CHECK(vt::Exl3GemmNumSms(2, 32, 128, 48) == 1);
  //   and never below one, even when the caller offers zero.
  CHECK(vt::Exl3GemmNumSms(2, 4096, 2048, 0) == 1);
}

// ─── §1 tier 2: had_r_128 ────────────────────────────────────────────────────

TEST_CASE("exl3 had_r_128: the CPU arm matches an independent Sylvester H128") {
  vt::Queue q = CpuQueue();
  Rng rng;
  const int64_t rows = 3, cols = 256;
  std::vector<float> in(static_cast<size_t>(rows * cols));
  for (auto& v : in) v = rng.next(2.0f);
  std::vector<float> out(in.size(), 0.0f);

  vt::Tensor ti = vt::Tensor::Contiguous(in.data(), vt::DType::kF32, q.device, {rows, cols});
  vt::Tensor to = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {rows, cols});
  vt::Exl3HadR128(q, to, ti, vt::Exl3HadArgs{});

  const double inv = 1.0 / std::sqrt(128.0);
  std::vector<double> blk(128), ref(128);
  double worst = 0.0, scale = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < cols; b += 128) {
      for (int i = 0; i < 128; ++i)
        blk[static_cast<size_t>(i)] = static_cast<double>(in[static_cast<size_t>(r * cols + b + i)]);
      HadRefBlock(blk.data(), ref.data(), inv);
      for (int i = 0; i < 128; ++i) {
        const double got = static_cast<double>(out[static_cast<size_t>(r * cols + b + i)]);
        worst = std::max(worst, std::fabs(got - ref[static_cast<size_t>(i)]));
        scale = std::max(scale, std::fabs(ref[static_cast<size_t>(i)]));
      }
    }
  }
  // f32 accumulation of a 128-term butterfly vs f64: ~sqrt(128) * 2^-24 of the
  // block magnitude. 1e-5 relative is three orders above that and four below a
  // wrong pairing, which flips a sign and misses by O(1).
  MESSAGE("had_r_128 f32 vs f64 Sylvester: worst=", worst, " scale=", scale);
  REQUIRE(scale > 0.0);
  CHECK(worst <= 1e-5 * scale);
}

TEST_CASE("exl3 had_r_128: pre_scale multiplies BEFORE and post_scale AFTER") {
  vt::Queue q = CpuQueue();
  Rng rng;
  const int64_t cols = 128;
  std::vector<float> in(static_cast<size_t>(cols));
  for (auto& v : in) v = rng.next(1.0f);
  std::vector<uint16_t> sc(static_cast<size_t>(cols));
  for (int64_t i = 0; i < cols; ++i)
    sc[static_cast<size_t>(i)] = vt::F32ToF16((i % 3 == 0) ? -2.0f : 0.5f);

  vt::Tensor ti = vt::Tensor::Contiguous(in.data(), vt::DType::kF32, q.device, {1, cols});
  vt::Tensor ts = vt::Tensor::Contiguous(sc.data(), vt::DType::kF16, q.device, {cols});

  std::vector<float> pre(static_cast<size_t>(cols), 0.0f), post(static_cast<size_t>(cols), 0.0f);
  vt::Tensor tpre = vt::Tensor::Contiguous(pre.data(), vt::DType::kF32, q.device, {1, cols});
  vt::Tensor tpost = vt::Tensor::Contiguous(post.data(), vt::DType::kF32, q.device, {1, cols});
  vt::Exl3HadArgs apre;
  apre.pre_scale = &ts;
  vt::Exl3HadArgs apost;
  apost.post_scale = &ts;
  vt::Exl3HadR128(q, tpre, ti, apre);
  vt::Exl3HadR128(q, tpost, ti, apost);

  const double inv = 1.0 / std::sqrt(128.0);
  std::vector<double> blk(128), ref(128);
  for (int i = 0; i < 128; ++i)
    blk[static_cast<size_t>(i)] = static_cast<double>(in[static_cast<size_t>(i)]) *
                                  static_cast<double>(vt::F16ToF32(sc[static_cast<size_t>(i)]));
  HadRefBlock(blk.data(), ref.data(), inv);
  double worst_pre = 0.0, scale_pre = 0.0;
  for (int i = 0; i < 128; ++i) {
    worst_pre = std::max(worst_pre, std::fabs(static_cast<double>(pre[static_cast<size_t>(i)]) -
                                              ref[static_cast<size_t>(i)]));
    scale_pre = std::max(scale_pre, std::fabs(ref[static_cast<size_t>(i)]));
  }
  CHECK(worst_pre <= 1e-5 * scale_pre);

  for (int i = 0; i < 128; ++i)
    blk[static_cast<size_t>(i)] = static_cast<double>(in[static_cast<size_t>(i)]);
  HadRefBlock(blk.data(), ref.data(), inv);
  double worst_post = 0.0, scale_post = 0.0;
  for (int i = 0; i < 128; ++i) {
    const double want = ref[static_cast<size_t>(i)] *
                        static_cast<double>(vt::F16ToF32(sc[static_cast<size_t>(i)]));
    worst_post = std::max(worst_post,
                          std::fabs(static_cast<double>(post[static_cast<size_t>(i)]) - want));
    scale_post = std::max(scale_post, std::fabs(want));
  }
  CHECK(worst_post <= 1e-5 * scale_post);
  // The two are DIFFERENT tensors — a kernel that ignored the placement and
  // applied the vector on one side only would pass one of the two checks above
  // and fail this one.
  double diff = 0.0;
  for (int i = 0; i < 128; ++i)
    diff = std::max(diff, std::fabs(static_cast<double>(pre[static_cast<size_t>(i)] -
                                                        post[static_cast<size_t>(i)])));
  CHECK(diff > 1e-3);
}

TEST_CASE("exl3 had_r_128: runs in place, and refuses what it cannot express") {
  vt::Queue q = CpuQueue();
  Rng rng;
  const int64_t cols = 128;
  std::vector<float> a(static_cast<size_t>(cols)), b(static_cast<size_t>(cols));
  for (int64_t i = 0; i < cols; ++i) {
    a[static_cast<size_t>(i)] = rng.next(1.0f);
    b[static_cast<size_t>(i)] = a[static_cast<size_t>(i)];
  }
  vt::Tensor ta = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {1, cols});
  vt::Tensor tb_in = vt::Tensor::Contiguous(b.data(), vt::DType::kF32, q.device, {1, cols});
  std::vector<float> outb(static_cast<size_t>(cols), 0.0f);
  vt::Tensor tb_out = vt::Tensor::Contiguous(outb.data(), vt::DType::kF32, q.device, {1, cols});
  vt::Exl3HadR128(q, ta, ta, vt::Exl3HadArgs{});  // in place (hadamard.cu:86)
  vt::Exl3HadR128(q, tb_out, tb_in, vt::Exl3HadArgs{});
  for (int64_t i = 0; i < cols; ++i)
    CHECK(a[static_cast<size_t>(i)] == outb[static_cast<size_t>(i)]);

  // cols % 128 != 0 is TORCH_CHECK_DIV(input, 1, 128) upstream (hadamard.cu:102)
  // and is refused BY NAME here.
  std::vector<float> bad(64, 0.0f);
  vt::Tensor tbad = vt::Tensor::Contiguous(bad.data(), vt::DType::kF32, q.device, {1, 64});
  std::string msg;
  try {
    vt::Exl3HadR128(q, tbad, tbad, vt::Exl3HadArgs{});
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  CHECK(msg.find("128") != std::string::npos);
  CHECK(msg.find("exl3_had_r_128") != std::string::npos);
}

// ─── §1 tier 3: exl3_gemm ────────────────────────────────────────────────────

TEST_CASE("exl3 gemm: the CPU arm matches the f64 chain within the stated bound") {
  vt::Queue q = CpuQueue();
  const int64_t m = 3, k = 256, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x51ED270Bu);

  Rng rng;
  rng.s = 0xB5297A4Du;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  std::vector<float> a_f(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a_h.size(); ++i) {
    a_h[i] = vt::F32ToF16(rng.next(1.0f));
    a_f[i] = vt::F16ToF32(a_h[i]);  // the reference sees EXACTLY the fp16 the kernel does
  }
  std::vector<uint16_t> c_h(static_cast<size_t>(m * n), 0);
  std::vector<uint16_t> a_had(static_cast<size_t>(m * k), 0);

  vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, q.device, {m, k});
  vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, q.device,
                                         {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, q.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, q.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(c_h.data(), vt::DType::kF16, q.device, {m, n});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, q.device, {m, k});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = 1;
  vt::Exl3Gemm(q, tc, ta, tb, tsuh, tsvh, tah, args);

  const std::vector<double> ref = Exl3ChainF64(f, a_f, m);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0, worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double got = static_cast<double>(vt::F16ToF32(c_h[i]));
    const double d = got - ref[i];
    sq += d * d;
    worst = std::max(worst, std::fabs(d));
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  const double ulp = UlpF16(rms);
  MESSAGE("exl3_gemm cpu vs f64: rel_rms=", rel_rms, " worst=", worst, " 8*ulp(rms)=", 8.0 * ulp);
  CHECK(rel_rms <= 1.0e-3);   // spec `## W2 design` §1 tier 3
  CHECK(worst <= 8.0 * ulp);  // spec `## W2 design` §1 tier 3
}

TEST_CASE("exl3 gemm: the fused BASIS agrees with the W1a weight-side dequant") {
  // The algebraic identity the whole format rests on (`exl3.py:183-214` vs
  // `:227-237`): running the two Hadamards on the ACTIVATIONS is the same map as
  // running them on the WEIGHTS. This links the fused kernel to the
  // independently-gated W1a reference. It is a LOOSER bound than tier 3 and says
  // why: `Exl3DequantLinear` rounds the weight to fp16 after each of its four
  // stages (`quantize.py:340-356` `.to(x_dtype)`), roundings the fused path never
  // performs, and each costs up to half an fp16 ulp = 2.4e-4 relative.
  vt::Queue q = CpuQueue();
  const int64_t m = 2, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x2545F491u);

  Rng rng;
  rng.s = 0x27220A95u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  std::vector<uint16_t> c_h(static_cast<size_t>(m * n), 0);
  std::vector<uint16_t> a_had(static_cast<size_t>(m * k), 0);
  vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, q.device, {m, k});
  vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, q.device,
                                         {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, q.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, q.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(c_h.data(), vt::DType::kF16, q.device, {m, n});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, q.device, {m, k});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = 1;
  vt::Exl3Gemm(q, tc, ta, tb, tsuh, tsvh, tah, args);

  std::vector<float> w(static_cast<size_t>(k * n));
  vt::Exl3DequantLinear(f.trellis.data(), f.suh.data(), f.svh.data(), k, n, f.bits, /*codebook=*/1, w.data());
  std::vector<double> ref(static_cast<size_t>(m * n), 0.0);
  for (int64_t r = 0; r < m; ++r)
    for (int64_t i = 0; i < k; ++i) {
      const double xv = static_cast<double>(vt::F16ToF32(a_h[static_cast<size_t>(r * k + i)]));
      for (int64_t j = 0; j < n; ++j)
        ref[static_cast<size_t>(r * n + j)] += xv * static_cast<double>(w[static_cast<size_t>(i * n + j)]);
    }
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(vt::F16ToF32(c_h[i])) - ref[i];
    sq += d * d;
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  MESSAGE("exl3_gemm cpu vs W1a weight basis: rel_rms=", rel_rms);
  // Four weight roundings at 2.4e-4 each, added in quadrature with the fused
  // path's own fp16 store, bound the difference at 4 * 2.4e-4 = 9.6e-4 in the
  // worst case; 2.0e-3 is twice that and still two orders below a decode defect.
  CHECK(rel_rms <= 2.0e-3);
}

TEST_CASE("exl3 gemm: unrepresentable inputs REFUSE BY NAME") {
  vt::Queue q = CpuQueue();
  const int64_t m = 1, k = 128, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x9E3779B9u);
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k), 0), c_h(static_cast<size_t>(m * n), 0),
      a_had(static_cast<size_t>(m * k), 0);
  vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()), vt::DType::kI8, q.device,
                                         {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()), vt::DType::kF16, q.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()), vt::DType::kF16, q.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(c_h.data(), vt::DType::kF16, q.device, {m, n});
  vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, q.device, {m, k});

  auto refusal = [&](vt::Tensor a, vt::Exl3GemmArgs args) {
    std::string msg;
    try {
      vt::Exl3Gemm(q, tc, a, tb, tsuh, tsvh, tah, args);
    } catch (const std::runtime_error& e) {
      msg = e.what();
    }
    return msg;
  };
  vt::Exl3GemmArgs ok;
  ok.bits = 3;
  ok.codebook = 1;

  // An f32 activation: `ldmatrix` + `mma.f16` read fp16 fragments, so this is
  // not a widening we may perform silently.
  std::vector<float> a_f(static_cast<size_t>(m * k), 0.0f);
  vt::Tensor ta_f32 = vt::Tensor::Contiguous(a_f.data(), vt::DType::kF32, q.device, {m, k});
  const std::string m_dtype = refusal(ta_f32, ok);
  CHECK(m_dtype.find("exl3_gemm") != std::string::npos);
  CHECK(m_dtype.find("f16") != std::string::npos);

  // The codebook seam, in BOTH directions.
  //
  // THIS CASE HAS BEEN WRONG ONCE ALREADY, and the history is why it now asserts
  // acceptance as well as refusal. It first used cb 0, asserting "the artifact
  // is mcg (cb 1)" as though cb 0 were exotic; cb 0 is the original QTIP 3INST
  // and is what EVERY stock `turboderp/*-exl3` checkpoint uses, because
  // `LinearEXL3` derives the codebook from tensor PRESENCE (`exl3.py:74-77`) and
  // those artifacts ship no marker. It then moved to cb 2, which was true until
  // cb 2 was ported (QUANT-EXL3-MUL1, #2495).
  //
  // A refusal-only case cannot tell "not yet ported" from "ported and this seam
  // never learned", so the positive half is the load-bearing one: cb 2 must
  // reach the host arm and RUN.
  vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, q.device, {m, k});
  vt::Exl3GemmArgs cb2 = ok;
  cb2.codebook = 2;
  CHECK(refusal(ta, cb2).empty());

  // What is left to refuse is a value UPSTREAM DOES NOT DEFINE.
  // `decode_3inst<cb>` (`codebook.cuh:56-90`) has arms for 0, 1 and 2 and falls
  // off the end for anything else, so cb 3 is not an arm awaiting a port.
  vt::Exl3GemmArgs cb3 = ok;
  cb3.codebook = 3;
  const std::string m_cb = refusal(ta, cb3);
  CHECK(m_cb.find("exl3_gemm") != std::string::npos);
  CHECK(m_cb.find("codebook") != std::string::npos);
  vt::Exl3GemmArgs cbneg = ok;
  cbneg.codebook = -1;
  CHECK_FALSE(refusal(ta, cbneg).empty());
}

// ─── the device arms ─────────────────────────────────────────────────────────

TEST_CASE("exl3 device: had_r_128 is BYTE-identical to the CPU arm") {
  if (!HasCuda()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2a device parity is PENDING. "
        "dgx.casa hung 2026-08-25 03:24Z (GB10 unified-memory OOM-reboot signature) "
        "and needs a manual power cycle. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    // This one asserts the precondition it skipped on, so the counter is honest.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3HadR128, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue dq = cb.CreateQueue();
  vt::Queue hq = CpuQueue();
  Rng rng;
  const int64_t rows = 4, cols = 256;
  std::vector<uint16_t> in(static_cast<size_t>(rows * cols));
  for (auto& v : in) v = vt::F32ToF16(rng.next(2.0f));
  std::vector<uint16_t> host_out(in.size(), 0), dev_out(in.size(), 0);

  vt::Tensor hi = vt::Tensor::Contiguous(in.data(), vt::DType::kF16, hq.device, {rows, cols});
  vt::Tensor ho = vt::Tensor::Contiguous(host_out.data(), vt::DType::kF16, hq.device, {rows, cols});
  vt::Exl3HadR128(hq, ho, hi, vt::Exl3HadArgs{});

  void* d_in = cb.Alloc(in.size() * sizeof(uint16_t));
  void* d_out = cb.Alloc(in.size() * sizeof(uint16_t));
  cb.Copy(dq, d_in, in.data(), in.size() * sizeof(uint16_t));
  vt::Tensor di = vt::Tensor::Contiguous(d_in, vt::DType::kF16, dq.device, {rows, cols});
  vt::Tensor dof = vt::Tensor::Contiguous(d_out, vt::DType::kF16, dq.device, {rows, cols});
  vt::Exl3HadR128(dq, dof, di, vt::Exl3HadArgs{});
  cb.Synchronize(dq);
  cb.Copy(dq, dev_out.data(), d_out, in.size() * sizeof(uint16_t));
  cb.Synchronize(dq);

  int mismatches = 0;
  for (size_t i = 0; i < in.size(); ++i)
    if (host_out[i] != dev_out[i]) ++mismatches;
  CHECK(mismatches == 0);
  cb.Free(d_in);
  cb.Free(d_out);
  cb.DestroyQueue(dq);
}

TEST_CASE("exl3 device: exl3_gemm matches the f64 reference within the stated bound") {
  if (!HasCuda()) {
    MESSAGE(
        "SKIPPED, no CUDA device: MODEL-DSV4-EXL3 W2b device parity is PENDING. "
        "dgx.casa hung 2026-08-25 03:24Z and needs a manual power cycle. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue dq = cb.CreateQueue();
  const int64_t m = 4, k = 512, n = 256;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x1B873593u);
  Rng rng;
  rng.s = 0xCC9E2D51u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  std::vector<float> a_f(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a_h.size(); ++i) {
    a_h[i] = vt::F32ToF16(rng.next(1.0f));
    a_f[i] = vt::F16ToF32(a_h[i]);
  }
  const size_t a_bytes = a_h.size() * sizeof(uint16_t);
  const size_t b_bytes = f.trellis.size() * sizeof(uint16_t);
  const size_t c_bytes = static_cast<size_t>(m * n) * sizeof(uint16_t);
  void* d_a = cb.Alloc(a_bytes);
  void* d_ah = cb.Alloc(a_bytes);
  void* d_b = cb.Alloc(b_bytes);
  void* d_suh = cb.Alloc(f.suh.size() * sizeof(uint16_t));
  void* d_svh = cb.Alloc(f.svh.size() * sizeof(uint16_t));
  void* d_c = cb.Alloc(c_bytes);
  cb.Copy(dq, d_a, a_h.data(), a_bytes);
  cb.Copy(dq, d_b, f.trellis.data(), b_bytes);
  cb.Copy(dq, d_suh, f.suh.data(), f.suh.size() * sizeof(uint16_t));
  cb.Copy(dq, d_svh, f.svh.data(), f.svh.size() * sizeof(uint16_t));

  vt::Tensor ta = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tb = vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device,
                                         {k / 16, n / 16, 32 * f.bits});
  vt::Tensor tsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tc = vt::Tensor::Contiguous(d_c, vt::DType::kF16, dq.device, {m, n});
  vt::Exl3GemmArgs args;
  args.bits = f.bits;
  args.codebook = 1;
  vt::Exl3Gemm(dq, tc, ta, tb, tsuh, tsvh, tah, args);
  cb.Synchronize(dq);
  std::vector<uint16_t> c_h(static_cast<size_t>(m * n), 0);
  cb.Copy(dq, c_h.data(), d_c, c_bytes);
  cb.Synchronize(dq);

  const std::vector<double> ref = Exl3ChainF64(f, a_f, m);
  const double rms = Rms(ref);
  REQUIRE(rms > 0.0);
  double sq = 0.0, worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(vt::F16ToF32(c_h[i])) - ref[i];
    sq += d * d;
    worst = std::max(worst, std::fabs(d));
  }
  const double rel_rms = std::sqrt(sq / static_cast<double>(ref.size())) / rms;
  const double ulp = UlpF16(rms);
  MESSAGE("exl3_gemm CUDA vs f64: rel_rms=", rel_rms, " worst=", worst, " 8*ulp=", 8.0 * ulp);
  CHECK(rel_rms <= 1.0e-3);
  CHECK(worst <= 8.0 * ulp);

  cb.Free(d_a);
  cb.Free(d_ah);
  cb.Free(d_b);
  cb.Free(d_suh);
  cb.Free(d_svh);
  cb.Free(d_c);
  cb.DestroyQueue(dq);
}

// ─── the widened device arms — QUANT-EXL3 W3 (#2181) ─────────────────────────
//
// The device arm was instantiated for ONE (bits, codebook) pair, (3, 1), and
// that pair is the EXCEPTION rather than the rule: `LinearEXL3` derives the
// codebook from tensor PRESENCE (`exl3.py:74-77`), so every stock
// `turboderp/*-exl3` artifact ships no marker and is codebook 0, and the device
// refused all of them one projection at a time. W3 instantiates (3, 0), (3, 1)
// and (6, 0) — a stock body, the DeepSeek-V4 artifact, and the stock 6-bit
// `lm_head`.
//
// These cases gate the NEW arms on the device. The CPU arm is generic over
// widths and codebooks and is gated elsewhere; what is unproven until a GPU runs
// it is that each instantiation LAUNCHES and agrees with that generic arm.
TEST_CASE("exl3 device: the widened (bits, codebook) arms agree with the CPU arm") {
  if (!HasCuda()) {
    MESSAGE(
        "SKIPPED, no CUDA device: QUANT-EXL3 W3's widened arms are PENDING. Reproduce with: "
        "rc run --device dgx:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm -V");
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb_dev = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue dq = cb_dev.CreateQueue();
  vt::Queue hq = CpuQueue();

  struct Arm {
    int bits;
    int codebook;
    const char* what;
  };
  // (3, 1) is covered by the case above; these are the two W3 added, the three
  // QUANT-EXL3-MUL1 added for `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (#2495), and
  // the fourth mul1 width its slice F added (#2574).
  //
  // The cb 2 rows are the ones that carry a NEW DECODE and not only a new width:
  // `decode_mul1_product_2` sums the four bytes of the product instead of adding
  // the two fp16 halves, so agreement with the host arm here is agreement on
  // that decode. The host side of it is gated against hand-computed upstream
  // values in `tests/vt/test_exl3_dequant.cpp`, which is what keeps this
  // cross-check from being two ports of the same mistake.
  //
  // (3, 2) IS THE ROW THAT MATTERS MOST HERE AND IT IS ALSO THE MOST FRAGILE.
  // It is the artifact's single largest width population -- 137 of its 409
  // trellis modules, every MLP projection of the layers quantized at the low end
  // of its 3.5 bpw average -- and it was missing because the census slice D
  // worked from said 272 modules where the artifact has 409. It is also now the
  // NEIGHBOUR of (3, 1): same width, adjacent codebook, same `dq8` route, same
  // tile shapes. A `cb` threaded wrongly between those two compiles, launches,
  // and returns the right shape; only a numeric comparison against an
  // independent decoder sees it, which is what this loop is.
  const Arm arms[] = {{3, 0, "a stock exl3 body"},
                      {6, 0, "a stock exl3 lm_head"},
                      {3, 2, "the Qwen3.8-27B mul1 MLP, 137 of its 409 trellis modules"},
                      {4, 2, "the Qwen3.8-27B mul1 GDN tower and attention, 270 more"},
                      {5, 2, "the Qwen3.8-27B mul1 5-bit tensor, and all 36 of its draft"},
                      {6, 2, "the Qwen3.8-27B mul1 lm_head"}};

  for (const Arm& arm : arms) {
    CAPTURE(arm.bits);
    CAPTURE(arm.codebook);
    const int64_t m = 4, k = 256, n = 256;
    const Exl3Fixture f = MakeFixture(k, n, arm.bits, 0x1D0C0DEu + arm.bits);
    Rng rng;
    rng.s = 0xC0FFEEu + arm.bits;
    std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
    for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

    // The CPU arm is the reference: it decodes every width and every codebook,
    // and `test_exl3_real_decode` ties it to real exllamav3 data.
    std::vector<uint16_t> a_had_h(a_h.size(), 0);
    std::vector<float> c_host(static_cast<size_t>(m * n), 0.0f);
    vt::Exl3GemmArgs args;
    args.bits = arm.bits;
    args.codebook = arm.codebook;
    {
      vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tah =
          vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
      vt::Tensor tc = vt::Tensor::Contiguous(c_host.data(), vt::DType::kF32, hq.device, {m, n});
      vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()),
                                             vt::DType::kI8, hq.device,
                                             {k / 16, n / 16, 32 * arm.bits});
      vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()),
                                               vt::DType::kF16, hq.device, {k});
      vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()),
                                               vt::DType::kF16, hq.device, {n});
      vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, args);
    }

    // Raw device allocations, the same shape the (3, 1) case above uses; there
    // is no DBuf helper in this suite and inventing one here would be a second
    // way to do what that case already does.
    const size_t ab = a_h.size() * sizeof(uint16_t);
    const size_t bb = f.trellis.size() * sizeof(uint16_t);
    const size_t cbytes = static_cast<size_t>(m * n) * sizeof(float);
    void* d_a = cb_dev.Alloc(ab);
    void* d_ah = cb_dev.Alloc(ab);
    void* d_b = cb_dev.Alloc(bb);
    void* d_suh = cb_dev.Alloc(f.suh.size() * sizeof(uint16_t));
    void* d_svh = cb_dev.Alloc(f.svh.size() * sizeof(uint16_t));
    void* d_c = cb_dev.Alloc(cbytes);
    cb_dev.Copy(dq, d_a, a_h.data(), ab);
    cb_dev.Copy(dq, d_b, f.trellis.data(), bb);
    cb_dev.Copy(dq, d_suh, f.suh.data(), f.suh.size() * sizeof(uint16_t));
    cb_dev.Copy(dq, d_svh, f.svh.data(), f.svh.size() * sizeof(uint16_t));

    vt::Tensor tda = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor tdah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor tdb = vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device,
                                            {k / 16, n / 16, 32 * arm.bits});
    vt::Tensor tdsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
    vt::Tensor tdsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
    vt::Tensor tdc = vt::Tensor::Contiguous(d_c, vt::DType::kF32, dq.device, {m, n});
    vt::Exl3Gemm(dq, tdc, tda, tdb, tdsuh, tdsvh, tdah, args);
    cb_dev.Synchronize(dq);
    std::vector<float> c_dev(static_cast<size_t>(m * n), 0.0f);
    cb_dev.Copy(dq, c_dev.data(), d_c, cbytes);
    cb_dev.Synchronize(dq);
    for (void* p : {d_a, d_ah, d_b, d_suh, d_svh, d_c}) cb_dev.Free(p);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < c_dev.size(); ++i) {
      const double d = static_cast<double>(c_dev[i]) - c_host[i];
      num += d * d;
      den += static_cast<double>(c_host[i]) * c_host[i];
    }
    const double rel = std::sqrt(num / den);
    MESSAGE("bits ", arm.bits, " cb ", arm.codebook, " (", arm.what,
            "): device vs CPU rel_rms = ", rel);
    REQUIRE(den > 0.0);  // not vacuous
    // The same bound the (3, 1) device case states: the difference is the
    // tensor-core reduction order, not the decode, which is exact either side.
    CHECK(rel <= 1.0e-3);
  }
  cb_dev.DestroyQueue(dq);
}

// ─── the kernel SHAPE table, forced — QUANT-EXL3 W5 (#2749) ──────────────────
//
// `Exl3SelectGemmShape` picks one of four shapes and `GemmKernelForArm`
// instantiates all four for each of the seven `(bits, codebook)` arms. Before
// this case ONE of the four had ever executed: the device cases above run
// `k 512 n 256` and `k 256 n 256`, which on a Blackwell-class `cc` take
// `mod_256 && size_n <= 4096` with `size_k > 8192` false and return shape 2.
// Shape 4 is not merely unselected at `n = 256`, it is REFUSED -- it tiles `n`
// by 512 and `Exl3GemmShapeCompat` rejects that `n`. Twenty-one instantiations
// compiled into the fat build for every architecture and shipped untested.
//
// NO PRODUCT CODE CHANGES. `Exl3GemmArgs::force_shape_idx` already exists and
// `Exl3GemmKernelCuda` already honours it (`cuda_exl3.cu:2270`), mirroring
// upstream's own `force_shape` parameter. What changes is that the gate stops
// measuring one kernel four times.
//
// THE DISCRIMINATION PROBLEM, AND WHY THE OBVIOUS ANSWER IS UNAVAILABLE. A
// forced shape and the selected shape compute the same product, so no
// comparison against any reference can tell a honoured force from a dropped
// one. The structural discriminator -- pick dimensions at which the SELECTED
// shape would refuse -- does not exist here, and the reason is worth writing
// down so nobody re-derives it: `vt::Exl3Gemm` runs `had_r_128` over `A[m, k]`
// and over `C[m, n]`, and that transform refuses a row length that is not a
// multiple of 128. So `k % 128 == 0` and `n % 128 == 0` hold on EVERY legal
// call, which makes shape 1 (`k % 16`, `n % 128`) and shape 2 (`k % 32`,
// `n % 128`) compatible with every legal call and makes the selector's answer
// compatible in every branch it has.
//
// Two things settle it instead, and this case carries both.
//
//   1. THE REFUSAL PROVES THE FIELD IS READ. Forcing shape 4 at `n = 768`
//      (`768 % 512 == 256`) must throw by name, because shape 4 is
//      incompatible there. A launcher that dropped `force_shape_idx` would
//      select shape 3 -- `768 % 256 == 0` -- and return numbers.
//   2. BYTE INEQUALITY ACROSS SHAPES PROVES EACH KERNEL IS A DIFFERENT ONE.
//      The four shapes decompose the same problem into different tile grids, so
//      `Exl3GemmNumSms` gives them different block counts and the split-K
//      reduction over `k` runs a different tree. At `k 256 n 512` the slice
//      counts are 64, 32, 16 and 16, and shapes 3 and 4 reach their equal count
//      by different routes -- 2 n-tiles of 8 k-tiles against 1 n-tile of 16 --
//      so the per-output reduction chains differ in length. The output is taken
//      in f32 exactly so an fp16 store cannot absorb that. Two shapes returning
//      byte-identical output means the case is measuring one kernel twice,
//      which is the failure the `(3,1)`/`(3,2)` check in `test_exl3_gemv.cpp`
//      exists to catch; it is a stop condition, never an assertion to widen.
//
// NO COMMA IN THE NAME: doctest splits `-tc=` on commas (#2605).
TEST_CASE("exl3 device: every shape in the kernel table is FORCED and agrees with the CPU arm") {
  if (!HasCuda()) {
    MESSAGE(
        "SKIPPED, no CUDA device: QUANT-EXL3 W5's shape coverage is PENDING. Reproduce with: "
        "rc run -d thor:gpu0 -- ctest --test-dir build-cuda -R test_exl3_gemm -V");
    // A skip that asserts NOTHING reports `assertions: 0`, which reads as a pass.
    CHECK_FALSE(vt::OpRegistered(vt::OpId::kExl3Gemm, vt::DeviceType::kCUDA));
    return;
  }
  vt::Backend& cb_dev = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue dq = cb_dev.CreateQueue();
  vt::Queue hq = CpuQueue();

  // `n = 512` is the smallest `n` compatible with all four shapes at once:
  // shape 4 tiles it by 512, shape 3 by 256, shapes 1 and 2 by 128. `k = 256`
  // is a multiple of 128 (had_r_128) and of 32 (shapes 2 and 3).
  const int64_t m = 4, k = 256, n = 512;
  const int kBits = 3, kCb = 1;
  const Exl3Fixture f = MakeFixture(k, n, kBits, 0x27D4EB2Fu);
  Rng rng;
  rng.s = 0x165667B1u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  vt::Exl3GemmArgs base;
  base.bits = kBits;
  base.codebook = kCb;

  // The CPU arm is the reference, as in the widened-arms case above: it decodes
  // every width and codebook and `test_exl3_real_decode` ties it to real
  // exllamav3 output. It has no shape table, so it is the same reference for
  // all four.
  std::vector<uint16_t> a_had_h(a_h.size(), 0);
  std::vector<float> c_host(static_cast<size_t>(m * n), 0.0f);
  {
    vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, hq.device, {m, k});
    vt::Tensor tah = vt::Tensor::Contiguous(a_had_h.data(), vt::DType::kF16, hq.device, {m, k});
    vt::Tensor tc = vt::Tensor::Contiguous(c_host.data(), vt::DType::kF32, hq.device, {m, n});
    vt::Tensor tb = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.trellis.data()),
                                           vt::DType::kI8, hq.device,
                                           {k / 16, n / 16, 32 * kBits});
    vt::Tensor tsuh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.suh.data()),
                                             vt::DType::kF16, hq.device, {k});
    vt::Tensor tsvh = vt::Tensor::Contiguous(const_cast<uint16_t*>(f.svh.data()),
                                             vt::DType::kF16, hq.device, {n});
    vt::Exl3Gemm(hq, tc, ta, tb, tsuh, tsvh, tah, base);
  }
  double den = 0.0;
  for (float v : c_host) den += static_cast<double>(v) * v;
  REQUIRE(den > 0.0);  // a reference of zeros would admit anything

  const size_t ab = a_h.size() * sizeof(uint16_t);
  const size_t bb = f.trellis.size() * sizeof(uint16_t);
  const size_t cbytes = static_cast<size_t>(m * n) * sizeof(float);
  void* d_a = cb_dev.Alloc(ab);
  void* d_ah = cb_dev.Alloc(ab);
  void* d_b = cb_dev.Alloc(bb);
  void* d_suh = cb_dev.Alloc(f.suh.size() * sizeof(uint16_t));
  void* d_svh = cb_dev.Alloc(f.svh.size() * sizeof(uint16_t));
  void* d_c = cb_dev.Alloc(cbytes);
  cb_dev.Copy(dq, d_a, a_h.data(), ab);
  cb_dev.Copy(dq, d_b, f.trellis.data(), bb);
  cb_dev.Copy(dq, d_suh, f.suh.data(), f.suh.size() * sizeof(uint16_t));
  cb_dev.Copy(dq, d_svh, f.svh.data(), f.svh.size() * sizeof(uint16_t));
  vt::Tensor tda = vt::Tensor::Contiguous(d_a, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tdah = vt::Tensor::Contiguous(d_ah, vt::DType::kF16, dq.device, {m, k});
  vt::Tensor tdb = vt::Tensor::Contiguous(d_b, vt::DType::kI8, dq.device,
                                          {k / 16, n / 16, 32 * kBits});
  vt::Tensor tdsuh = vt::Tensor::Contiguous(d_suh, vt::DType::kF16, dq.device, {k});
  vt::Tensor tdsvh = vt::Tensor::Contiguous(d_svh, vt::DType::kF16, dq.device, {n});
  vt::Tensor tdc = vt::Tensor::Contiguous(d_c, vt::DType::kF32, dq.device, {m, n});

  // The loop bound is the TABLE's own count, not a literal 4: a fifth shape
  // added to `kShapes` must arrive with a gate or fail here.
  std::vector<std::vector<float>> per_shape;
  for (int idx = 1; idx <= vt::Exl3GemmNumShapes(); ++idx) {
    CAPTURE(idx);
    // The instrument's own precondition. A shape incompatible with these
    // dimensions would throw instead of running, and a case that never ran the
    // kernel it names is a skip wearing a pass.
    REQUIRE(vt::Exl3GemmShapeCompat(idx, static_cast<int>(k), static_cast<int>(n)));

    std::vector<float> c_dev(static_cast<size_t>(m * n), 0.0f);
    vt::Exl3GemmArgs args = base;
    args.force_shape_idx = idx;
    vt::Exl3Gemm(dq, tdc, tda, tdb, tdsuh, tdsvh, tdah, args);
    cb_dev.Synchronize(dq);
    cb_dev.Copy(dq, c_dev.data(), d_c, cbytes);
    cb_dev.Synchronize(dq);

    double num = 0.0;
    for (size_t i = 0; i < c_dev.size(); ++i) {
      const double d = static_cast<double>(c_dev[i]) - static_cast<double>(c_host[i]);
      num += d * d;
    }
    const double rel = std::sqrt(num / den);
    MESSAGE("forced shape ", idx, ": device vs CPU rel_rms = ", rel);
    // The same bound the (3, 1) device case states. A new shape INHERITS it.
    CHECK(rel <= 1.0e-3);
    per_shape.push_back(c_dev);
  }

  // THE DISCRIMINATION CHECK (2 above). Byte comparison, not a tolerance: two
  // runs of the SAME kernel on the same operands are bit-identical, so any
  // difference at all is proof that a different kernel ran.
  REQUIRE(per_shape.size() == static_cast<size_t>(vt::Exl3GemmNumShapes()));
  for (size_t i = 0; i < per_shape.size(); ++i) {
    for (size_t j = i + 1; j < per_shape.size(); ++j) {
      CAPTURE(i + 1);
      CAPTURE(j + 1);
      size_t differing = 0;
      for (size_t e = 0; e < per_shape[i].size(); ++e)
        if (std::memcmp(&per_shape[i][e], &per_shape[j][e], sizeof(float)) != 0) ++differing;
      MESSAGE("shapes ", i + 1, " and ", j + 1, " differ in ", differing, " of ",
              per_shape[i].size(), " outputs");
      CHECK(differing > 0);
    }
  }

  for (void* p : {d_a, d_ah, d_b, d_suh, d_svh, d_c}) cb_dev.Free(p);

  // THE REFUSAL (1 above). `n = 768` is a legal EXL3 output width -- a multiple
  // of 128, so `had_r_128` accepts it -- and it is compatible with shapes 1, 2
  // and 3 but NOT with shape 4. Forcing 4 there must refuse BY NAME. A launcher
  // that ignored the field would select shape 3 and return numbers.
  {
    const int64_t rn = 768;
    REQUIRE_FALSE(vt::Exl3GemmShapeCompat(4, static_cast<int>(k), static_cast<int>(rn)));
    REQUIRE(vt::Exl3GemmShapeCompat(3, static_cast<int>(k), static_cast<int>(rn)));
    const Exl3Fixture rf = MakeFixture(k, rn, kBits, 0x85EBCA6Bu);
    void* r_a = cb_dev.Alloc(ab);
    void* r_ah = cb_dev.Alloc(ab);
    void* r_b = cb_dev.Alloc(rf.trellis.size() * sizeof(uint16_t));
    void* r_suh = cb_dev.Alloc(rf.suh.size() * sizeof(uint16_t));
    void* r_svh = cb_dev.Alloc(rf.svh.size() * sizeof(uint16_t));
    void* r_c = cb_dev.Alloc(static_cast<size_t>(m * rn) * sizeof(float));
    vt::Tensor ra = vt::Tensor::Contiguous(r_a, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor rah = vt::Tensor::Contiguous(r_ah, vt::DType::kF16, dq.device, {m, k});
    vt::Tensor rb = vt::Tensor::Contiguous(r_b, vt::DType::kI8, dq.device,
                                           {k / 16, rn / 16, 32 * kBits});
    vt::Tensor rsuh = vt::Tensor::Contiguous(r_suh, vt::DType::kF16, dq.device, {k});
    vt::Tensor rsvh = vt::Tensor::Contiguous(r_svh, vt::DType::kF16, dq.device, {rn});
    vt::Tensor rc = vt::Tensor::Contiguous(r_c, vt::DType::kF32, dq.device, {m, rn});
    vt::Exl3GemmArgs bad = base;
    bad.force_shape_idx = 4;
    std::string msg;
    try {
      vt::Exl3Gemm(dq, rc, ra, rb, rsuh, rsvh, rah, bad);
      cb_dev.Synchronize(dq);
    } catch (const std::runtime_error& e) {
      msg = e.what();
    }
    MESSAGE("forced shape 4 at n=768 said: ", msg);
    CHECK(msg.find("exl3_gemm") != std::string::npos);
    CHECK(msg.find("no compatible") != std::string::npos);
    CHECK(msg.find("selected shape 4") != std::string::npos);
    for (void* p : {r_a, r_ah, r_b, r_suh, r_svh, r_c}) cb_dev.Free(p);
  }

  cb_dev.DestroyQueue(dq);
  vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(hq);
}

// ─── The operands' BYTE ADDRESSES — #2558 ────────────────────────────────────
//
// `suh`, `svh` and the trellis are WEIGHTS, and the borrow path hands the kernel
// the mapping's own bytes. A safetensors payload starts at
// `8 + <JSON header length>` (`safetensors_reader.cpp:78`) and a header length is
// arbitrary, so any of the three begins on an ODD byte in roughly half of all
// checkpoints. The trellis is the sharpest case: `vt::Exl3Gemm` types it kI8 --
// "opaque i8 BYTES" -- so its alignment requirement is 1 and an odd base is the
// expected case rather than an exotic one.
//
// `HadRowBlock` (`cpu_exl3_kernels.cpp:130`) read `suh`/`svh` through a
// `const uint16_t*` and `TileWord32` (`cpu_exl3_dequant.cpp:64`) read the trellis
// the same way, and `-fsanitize=alignment` aborted `test_qwen35_exl3` on the
// first of those, reached through `ModelRegistry::Forward`.
//
// THE FIX IS IN THE CONSUMER, NOT THE PRODUCER, and this case is what separates
// the two. Refusing to borrow a misaligned tensor also takes the sanitizer green
// -- and switches direct upload off for every tensor in half of all checkpoints,
// because the parity of the payload base is a coin flip per FILE.
// `tests/vllm/test_load_direct_upload.cpp` and
// `tests/vllm/models/test_qwen3_5_dense_load_residency.cpp` hold that lever; this
// case holds the correctness, and both must be green at once.
//
// The bar is BYTE EQUALITY against the same operands at even addresses. The
// repair replaces word arithmetic with byte arithmetic over the trellis, and a
// stride that lost its `sizeof(uint16_t)` decodes a different weight -- which a
// tolerance over a random-bit trellis could plausibly absorb and an equality
// cannot. Each `REQUIRE` on a parity is the instrument's own precondition: a run
// that quietly landed on an even address would exercise nothing and still pass.
// NO COMMA IN THE NAME. doctest splits `-tc=` on commas, so a case whose name
// carries one can never be selected by it: the filter matches nothing and the
// binary reports `assertions: 0` and `SUCCESS!` at rc 0, which is a SKIP
// wearing a pass for anyone running a focused gate by name.
TEST_CASE("exl3 gemm: the suh/svh scales and the trellis at ODD byte addresses decode the same") {
  vt::Queue q = CpuQueue();
  const int64_t m = 2, k = 256, n = 128;
  const Exl3Fixture f = MakeFixture(k, n, 3, 0x2545F491u);

  Rng rng;
  rng.s = 0x27220A95u;
  std::vector<uint16_t> a_h(static_cast<size_t>(m * k));
  for (auto& v : a_h) v = vt::F32ToF16(rng.next(1.0f));

  // `off` bytes of lead-in before each of the three weight payloads. 0 is the
  // aligned control; 1 is what a 171-byte header produces.
  auto run = [&](size_t off) {
    const size_t tb_bytes = f.trellis.size() * sizeof(uint16_t);
    const size_t su_bytes = f.suh.size() * sizeof(uint16_t);
    const size_t sv_bytes = f.svh.size() * sizeof(uint16_t);
    std::vector<unsigned char> tb_raw(tb_bytes + off, 0), su_raw(su_bytes + off, 0),
        sv_raw(sv_bytes + off, 0);
    std::memcpy(tb_raw.data() + off, f.trellis.data(), tb_bytes);
    std::memcpy(su_raw.data() + off, f.suh.data(), su_bytes);
    std::memcpy(sv_raw.data() + off, f.svh.data(), sv_bytes);
    void* tb_p = tb_raw.data() + off;
    void* su_p = su_raw.data() + off;
    void* sv_p = sv_raw.data() + off;
    REQUIRE(reinterpret_cast<std::uintptr_t>(tb_p) % 2 == off % 2);
    REQUIRE(reinterpret_cast<std::uintptr_t>(su_p) % 2 == off % 2);
    REQUIRE(reinterpret_cast<std::uintptr_t>(sv_p) % 2 == off % 2);

    std::vector<uint16_t> c_h(static_cast<size_t>(m * n), 0);
    std::vector<uint16_t> a_had(static_cast<size_t>(m * k), 0);
    vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, q.device, {m, k});
    vt::Tensor tb = vt::Tensor::Contiguous(tb_p, vt::DType::kI8, q.device,
                                           {k / 16, n / 16, 32 * f.bits});
    vt::Tensor tsuh = vt::Tensor::Contiguous(su_p, vt::DType::kF16, q.device, {k});
    vt::Tensor tsvh = vt::Tensor::Contiguous(sv_p, vt::DType::kF16, q.device, {n});
    vt::Tensor tc = vt::Tensor::Contiguous(c_h.data(), vt::DType::kF16, q.device, {m, n});
    vt::Tensor tah = vt::Tensor::Contiguous(a_had.data(), vt::DType::kF16, q.device, {m, k});
    vt::Exl3GemmArgs args;
    args.bits = f.bits;
    args.codebook = 1;
    vt::Exl3Gemm(q, tc, ta, tb, tsuh, tsvh, tah, args);
    return c_h;
  };

  const std::vector<uint16_t> aligned = run(0);
  // The control must have DECODED something. An all-zero output would satisfy the
  // equality below without either arm reading a single trellis byte.
  size_t nonzero = 0;
  for (const uint16_t v : aligned)
    if (v != 0) ++nonzero;
  REQUIRE(nonzero > aligned.size() / 2);

  CHECK(run(1) == aligned);
}
