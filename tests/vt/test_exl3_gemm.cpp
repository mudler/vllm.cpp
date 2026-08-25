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

namespace {

// ── independent references ───────────────────────────────────────────────────

// The natural-order Sylvester Hadamard: H[i][j] = (-1)^popcount(i & j). Built
// from the DEFINITION, so it shares no code with the butterfly under test.
// `util/hadamard.py:34-42` recurses to `hadamard_1.txt` = "+" for 128, which is
// exactly this matrix.
int SylvesterH(int i, int j) {
  unsigned v = static_cast<unsigned>(i & j);
  int parity = 0;
  while (v) {
    parity ^= static_cast<int>(v & 1u);
    v >>= 1;
  }
  return parity ? -1 : 1;
}

// One 128-block of `had_r_128` in double precision, straight from the docstring
// at `hadamard.cu:83-86`: y = (x.view(-1,128) @ had_128) * scale / sqrt(128).
void HadRefBlock(const double* x, double* y, double r_scale) {
  for (int j = 0; j < 128; ++j) {
    double acc = 0.0;
    for (int i = 0; i < 128; ++i) acc += x[i] * SylvesterH(i, j);
    y[j] = acc * r_scale;
  }
}

struct Rng {
  uint32_t s = 0x243F6A88u;
  float next(float scale) {
    s = s * 1664525u + 1013904223u;
    const float u = (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return u * scale;
  }
};

// fp16 ulp at `v` — the gap between consecutive representables around it. Used
// for the elementwise half of the tier-3 bound.
double UlpF16(double v) {
  const uint16_t h = vt::F32ToF16(static_cast<float>(std::fabs(v)));
  const float here = vt::F16ToF32(h);
  const float up = vt::F16ToF32(static_cast<uint16_t>(h + 1));
  return static_cast<double>(up - here);
}

double Rms(const std::vector<double>& v) {
  double a = 0.0;
  for (double e : v) a += e * e;
  return std::sqrt(a / static_cast<double>(v.size()));
}

// ── a synthetic EXL3 linear ──────────────────────────────────────────────────
//
// The trellis is filled with pseudo-random BITS. That is legitimate: every
// 16-bit codeword decodes to a valid fp16 pair under the MCG codebook, so a
// random bit stream is a valid (if meaningless) quantized weight, and the
// kernels must reproduce whatever it decodes to. suh/svh carry a real
// per-channel scale, not just signs, exactly as the DeepSeek-V4 artifact does.
struct Exl3Fixture {
  int64_t k = 0;
  int64_t n = 0;
  int bits = 3;
  std::vector<uint16_t> trellis;  // [k/16, n/16, 16*bits] words
  std::vector<uint16_t> suh;      // [k] fp16 bits
  std::vector<uint16_t> svh;      // [n] fp16 bits
};

Exl3Fixture MakeFixture(int64_t k, int64_t n, int bits, uint32_t seed) {
  Exl3Fixture f;
  f.k = k;
  f.n = n;
  f.bits = bits;
  Rng rng;
  rng.s = seed;
  f.trellis.resize(static_cast<size_t>(k / 16 * n / 16 * 16 * bits));
  for (auto& w : f.trellis) {
    rng.s = rng.s * 1664525u + 1013904223u;
    w = static_cast<uint16_t>(rng.s >> 13);
  }
  f.suh.resize(static_cast<size_t>(k));
  for (auto& s : f.suh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 0.75f : -0.75f);
  f.svh.resize(static_cast<size_t>(n));
  for (auto& s : f.svh) s = vt::F32ToF16(rng.next(1.0f) >= 0.0f ? 1.25f : -1.25f);
  return f;
}

// The tier-3 REFERENCE: the fused chain evaluated in double.
//   x_had = (x * suh) @ H128 / sqrt(128)   (blockwise over k)
//   y_raw = x_had @ W_inner                (W_inner = reconstruct(trellis))
//   y     = (y_raw @ H128 / sqrt(128)) * svh   (blockwise over n)
std::vector<double> Exl3ChainF64(const Exl3Fixture& f, const std::vector<float>& x_f16_rounded,
                                 int64_t m) {
  const int64_t k = f.k, n = f.n;
  std::vector<float> w_inner(static_cast<size_t>(k * n));
  vt::Exl3ReconstructInner(f.trellis.data(), k, n, f.bits, w_inner.data());

  const double inv = 1.0 / std::sqrt(128.0);
  std::vector<double> y(static_cast<size_t>(m * n), 0.0);
  std::vector<double> blk(128), obk(128);
  for (int64_t r = 0; r < m; ++r) {
    std::vector<double> xh(static_cast<size_t>(k));
    for (int64_t b = 0; b < k; b += 128) {
      for (int i = 0; i < 128; ++i)
        blk[static_cast<size_t>(i)] = static_cast<double>(x_f16_rounded[static_cast<size_t>(r * k + b + i)]) *
                                      static_cast<double>(vt::F16ToF32(f.suh[static_cast<size_t>(b + i)]));
      HadRefBlock(blk.data(), obk.data(), inv);
      for (int i = 0; i < 128; ++i) xh[static_cast<size_t>(b + i)] = obk[static_cast<size_t>(i)];
    }
    std::vector<double> raw(static_cast<size_t>(n), 0.0);
    for (int64_t i = 0; i < k; ++i) {
      const double xv = xh[static_cast<size_t>(i)];
      if (xv == 0.0) continue;
      const float* wrow = &w_inner[static_cast<size_t>(i * n)];
      for (int64_t j = 0; j < n; ++j)
        raw[static_cast<size_t>(j)] += xv * static_cast<double>(wrow[j]);
    }
    for (int64_t b = 0; b < n; b += 128) {
      HadRefBlock(&raw[static_cast<size_t>(b)], obk.data(), inv);
      for (int i = 0; i < 128; ++i)
        y[static_cast<size_t>(r * n + b + i)] =
            obk[static_cast<size_t>(i)] * static_cast<double>(vt::F16ToF32(f.svh[static_cast<size_t>(b + i)]));
    }
  }
  return y;
}

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
  vt::Exl3DequantLinear(f.trellis.data(), f.suh.data(), f.svh.data(), k, n, f.bits, w.data());
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

  // A codebook this row does not decode. The artifact is mcg (cb 1) and the
  // refusal must name what it wanted rather than silently decoding as mcg.
  vt::Tensor ta = vt::Tensor::Contiguous(a_h.data(), vt::DType::kF16, q.device, {m, k});
  vt::Exl3GemmArgs cb0 = ok;
  cb0.codebook = 0;
  const std::string m_cb = refusal(ta, cb0);
  CHECK(m_cb.find("exl3_gemm") != std::string::npos);
  CHECK(m_cb.find("codebook") != std::string::npos);
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
