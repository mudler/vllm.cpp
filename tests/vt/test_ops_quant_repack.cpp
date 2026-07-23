// CIQ G7 — repack-at-load for the q8_0 quant GEMM.
//
// The repack reorders the WEIGHT bytes into the i8mm block_q8_0x4 interleave
// (llama.cpp @ 237ad9b96 make_block_q8_0x4 / repack_q8_0_to_q8_0_4_bl); a wrong
// permutation silently corrupts, so this file proves two things:
//
//   1. PORTABLE (every CI box): the interleave transform matches the upstream
//      make_block_q8_0x4 formula byte-for-byte AND round-trips (interleave then
//      de-interleave == the original rows). Also documents the i8mm gating.
//
//   2. i8mm (dgx aarch64): the repacked GEMM is BYTE-IDENTICAL to the
//      non-repacked kMatmulBTQuant across decode (M=1), leftover (M%4) and
//      prefill (M%4==0) shapes — a permutation of storage the kernel un-permutes
//      exactly — AND agrees with an independent dequant-to-f64 oracle at
//      NMSE <= 5e-4. These cases skip coherently where the tier is not live.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vt/cpu/cpu_quant_blocks.h"  // BlockQ8_0 / BlockQ8_0x4 (via -I src)
#include "vt/cpu/cpu_quant_repack.h"  // InterleaveQ8_0Rows4 (via -I src)
#include "vt/cpu/cpu_threadpool.h"    // Threadpool::SwapForTesting (via -I src)
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/tensor.h"

namespace {

// A plain q8_0 weight of `n` rows x `nblocks` blocks, quantized from a random
// f32 signal through the SAME from_float the kernel uses (so the interleave is
// exercised on realistic quant values, not random bit patterns).
std::vector<uint8_t> MakePlainQ8_0(int64_t n, int64_t nblocks, uint32_t seed) {
  const int64_t k = nblocks * vt::cpu::kQK8_0;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.5F, 1.5F);
  std::vector<uint8_t> w(static_cast<size_t>(n) * nblocks * sizeof(vt::cpu::BlockQ8_0));
  std::vector<float> row(static_cast<size_t>(k));
  const auto from_float = vt::cpu::QuantTraits(vt::DType::kQ8_0).from_float;
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t p = 0; p < k; ++p) row[p] = dist(rng);
    from_float(row.data(),
               w.data() + static_cast<size_t>(r) * nblocks * sizeof(vt::cpu::BlockQ8_0), k);
  }
  return w;
}

// Independent re-statement of make_block_q8_0x4(in, 8) (repack.cpp:2725), so the
// gate does not check the transform against a copy of itself.
vt::cpu::BlockQ8_0x4 RefInterleave(const vt::cpu::BlockQ8_0* in4) {
  vt::cpu::BlockQ8_0x4 out;
  for (int i = 0; i < 4; ++i) out.d[i] = in4[i].d;
  for (int i = 0; i < 16; ++i) {
    const int src_id = i % 4;
    const int src_offset = (i / 4) * 8;
    std::memcpy(&out.qs[i * 8], &in4[src_id].qs[src_offset], 8);
  }
  return out;
}

}  // namespace

TEST_CASE("G7 InterleaveQ8_0Rows4 matches make_block_q8_0x4 and round-trips") {
  const int64_t nblocks = 5;
  std::vector<uint8_t> plain = MakePlainQ8_0(4, nblocks, 0x6117U);
  const auto* rows = reinterpret_cast<const vt::cpu::BlockQ8_0*>(plain.data());

  std::vector<vt::cpu::BlockQ8_0x4> got(static_cast<size_t>(nblocks));
  vt::cpu::InterleaveQ8_0Rows4(rows + 0 * nblocks, rows + 1 * nblocks,
                               rows + 2 * nblocks, rows + 3 * nblocks,
                               got.data(), nblocks);

  // (a) byte-for-byte vs the independent formula, per block.
  for (int64_t x = 0; x < nblocks; ++x) {
    const vt::cpu::BlockQ8_0 in4[4] = {rows[0 * nblocks + x], rows[1 * nblocks + x],
                                       rows[2 * nblocks + x], rows[3 * nblocks + x]};
    const vt::cpu::BlockQ8_0x4 ref = RefInterleave(in4);
    CHECK(std::memcmp(&got[static_cast<size_t>(x)], &ref, sizeof(ref)) == 0);
  }

  // (b) de-interleave (the exact inverse) reconstructs the original rows.
  for (int64_t x = 0; x < nblocks; ++x) {
    for (int r = 0; r < 4; ++r) {
      const vt::cpu::BlockQ8_0& src = rows[r * nblocks + x];
      CHECK(got[static_cast<size_t>(x)].d[r] == src.d);
      for (int so = 0; so < 32; so += 8) {
        const int src_chunk = r + 4 * (so / 8);
        CHECK(std::memcmp(&got[static_cast<size_t>(x)].qs[src_chunk * 8],
                          &src.qs[so], 8) == 0);
      }
    }
  }
}

TEST_CASE("G7 repack eligibility gating is total") {
  // q8_0 only, N%4==0, K%32==0, and only when the i8mm tier is live.
  const bool live = vt::cpu::QuantRepackActive();
  CHECK(vt::cpu::QuantRepackEligible(vt::DType::kQ8_0, 8, 64) == live);
  // Never eligible regardless of the tier:
  CHECK_FALSE(vt::cpu::QuantRepackEligible(vt::DType::kQ4_K, 8, 256));  // wrong dtype
  CHECK_FALSE(vt::cpu::QuantRepackEligible(vt::DType::kQ8_0, 6, 64));   // N%4!=0
  CHECK_FALSE(vt::cpu::QuantRepackEligible(vt::DType::kQ8_0, 8, 48));   // K%32!=0
  if (!live) {
    // On a non-i8mm box the transform is unreachable and must fail loud, never
    // silently produce a mis-laid buffer.
    std::vector<uint8_t> w = MakePlainQ8_0(8, 2, 1U);
    CHECK_THROWS(vt::cpu::QuantRepackWeight(vt::DType::kQ8_0, w.data(), 8, 64));
  }
}

TEST_CASE("G7 repacked GEMM is byte-identical to the plain quant GEMM") {
  if (!vt::cpu::QuantRepackActive()) return;  // i8mm-only; skips coherently

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::mt19937 rng(0x9EAC91U);
  std::uniform_real_distribution<float> adist(-1.0F, 1.0F);

  // Decode (M=1), leftover (M%4 != 0) and prefill (M%4 == 0) row counts; N a
  // multiple of 4 (eligibility); K a multiple of 32.
  for (int64_t nblocks : {int64_t{1}, int64_t{2}, int64_t{4}}) {
    const int64_t k = nblocks * vt::cpu::kQK8_0;
    for (int64_t n : {int64_t{4}, int64_t{8}, int64_t{16}, int64_t{64}}) {
      for (int64_t m : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4}, int64_t{5},
                        int64_t{8}, int64_t{128}}) {
        CAPTURE(m);
        CAPTURE(n);
        CAPTURE(k);
        const std::vector<uint8_t> plain =
            MakePlainQ8_0(n, nblocks, 0xC0DEU + static_cast<uint32_t>(n * 131 + k));
        std::vector<float> a(static_cast<size_t>(m) * k);
        for (float& v : a) v = adist(rng);

        // Reference: the non-repacked kMatmulBTQuant (mmla / portable tier).
        std::vector<float> out_ref(static_cast<size_t>(m) * n, 0.0F);
        {
          vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
          std::vector<uint8_t> wplain = plain;
          vt::Tensor bt = vt::Tensor::Contiguous(wplain.data(), vt::DType::kF32, q.device, {n, k});
          bt.dtype = vt::DType::kQ8_0;
          vt::Tensor ot = vt::Tensor::Contiguous(out_ref.data(), vt::DType::kF32, q.device, {m, n});
          vt::MatmulBTQuant(q, ot, at, bt);
        }

        // Repacked: same bytes, permuted by QuantRepackWeight, b.repacked=true.
        std::vector<float> out_rp(static_cast<size_t>(m) * n, 0.0F);
        {
          std::vector<uint8_t> wrp = plain;
          vt::cpu::QuantRepackWeight(vt::DType::kQ8_0, wrp.data(), n, k);
          vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
          vt::Tensor bt = vt::Tensor::Contiguous(wrp.data(), vt::DType::kF32, q.device, {n, k});
          bt.dtype = vt::DType::kQ8_0;
          bt.repacked = true;
          vt::Tensor ot = vt::Tensor::Contiguous(out_rp.data(), vt::DType::kF32, q.device, {m, n});
          vt::MatmulBTQuant(q, ot, at, bt);
        }

        // The target and the expectation: BYTE-identical, not merely close.
        CHECK(std::memcmp(out_ref.data(), out_rp.data(),
                          out_ref.size() * sizeof(float)) == 0);

        // Belt-and-braces: also within the ported MUL_MAT NMSE bound of the
        // independent dequant-to-f64 oracle (never trusted alone, but if the
        // memcmp above ever regresses to NMSE this documents the fallback bar).
        std::vector<float> w(static_cast<size_t>(n) * k);
        vt::cpu::BlockToFloat(vt::DType::kQ8_0)(plain.data(), w.data(), n * k);
        double num = 0, den = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t j = 0; j < n; ++j) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p) {
              ref += static_cast<double>(a[static_cast<size_t>(i) * k + p]) *
                     static_cast<double>(w[static_cast<size_t>(j) * k + p]);
            }
            const double d = out_rp[static_cast<size_t>(i) * n + j] - ref;
            num += d * d;
            den += ref * ref;
          }
        }
        CHECK((den > 0 ? num / den : num) <= 5e-4);
      }
    }
  }
}

TEST_CASE("G7 repacked GEMM matches plain at real model shapes / dtypes") {
  if (!vt::cpu::QuantRepackActive()) return;

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::mt19937 rng(0x5150U);
  std::uniform_real_distribution<float> adist(-1.0F, 1.0F);

  struct Shape { int64_t m, n, k; };
  const Shape shapes[] = {
      {128, 3072, 2048}, {128, 2048, 6144}, {1, 3072, 2048}, {7, 2048, 2048},
      {128, 2048, 2048}, {1, 2048, 6144},
  };
  for (const Shape& s : shapes) {
    for (vt::DType odt : {vt::DType::kF32, vt::DType::kBF16}) {
      for (bool strided : {false, true}) {
        CAPTURE(s.m); CAPTURE(s.n); CAPTURE(s.k);
        CAPTURE(static_cast<int>(odt)); CAPTURE(strided);
        const int64_t nblocks = s.k / vt::cpu::kQK8_0;
        const std::vector<uint8_t> plain = MakePlainQ8_0(s.n, nblocks, 0x11U);
        // Optionally over-allocate the activation row stride (a column slice of
        // a wider workspace — the relaxed contract MatmulBTQuant accepts).
        const int64_t a_rs = strided ? s.k + 96 : s.k;
        std::vector<float> a(static_cast<size_t>(s.m) * a_rs, 0.0F);
        for (int64_t i = 0; i < s.m; ++i)
          for (int64_t p = 0; p < s.k; ++p)
            a[static_cast<size_t>(i) * a_rs + p] = adist(rng);

        auto make_a = [&]() {
          vt::Tensor t;
          t.data = a.data(); t.dtype = vt::DType::kF32; t.device = q.device;
          t.rank = 2; t.shape[0] = s.m; t.shape[1] = s.k;
          t.stride[0] = a_rs; t.stride[1] = 1;
          return t;
        };
        const size_t osz = static_cast<size_t>(s.m) * s.n;
        std::vector<uint16_t> ref16(osz, 0), rp16(osz, 0);
        std::vector<float> ref32(osz, 0), rp32(osz, 0);

        auto run = [&](bool repack, void* obuf) {
          std::vector<uint8_t> w = plain;
          if (repack) vt::cpu::QuantRepackWeight(vt::DType::kQ8_0, w.data(), s.n, s.k);
          vt::Tensor at = make_a();
          vt::Tensor bt = vt::Tensor::Contiguous(w.data(), vt::DType::kF32, q.device, {s.n, s.k});
          bt.dtype = vt::DType::kQ8_0; bt.repacked = repack;
          vt::Tensor ot = vt::Tensor::Contiguous(obuf, odt, q.device, {s.m, s.n});
          vt::MatmulBTQuant(q, ot, at, bt);
        };
        void* refbuf = odt == vt::DType::kF32 ? (void*)ref32.data() : (void*)ref16.data();
        void* rpbuf = odt == vt::DType::kF32 ? (void*)rp32.data() : (void*)rp16.data();
        run(false, refbuf);
        run(true, rpbuf);
        const size_t bytes = osz * (odt == vt::DType::kF32 ? 4 : 2);
        CHECK(std::memcmp(refbuf, rpbuf, bytes) == 0);
      }
    }
  }
}

TEST_CASE("G7 repacked GEMM is bit-exact across thread counts") {
  if (!vt::cpu::QuantRepackActive()) return;

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t nblocks = 3, k = nblocks * vt::cpu::kQK8_0, n = 16, m = 130;
  const std::vector<uint8_t> plain = MakePlainQ8_0(n, nblocks, 0x7A11U);
  std::vector<float> a(static_cast<size_t>(m) * k);
  std::mt19937 rng(9U);
  std::uniform_real_distribution<float> adist(-1.0F, 1.0F);
  for (float& v : a) v = adist(rng);

  auto run = [&]() {
    std::vector<uint8_t> wrp = plain;
    vt::cpu::QuantRepackWeight(vt::DType::kQ8_0, wrp.data(), n, k);
    std::vector<float> out(static_cast<size_t>(m) * n, 0.0F);
    vt::Tensor at = vt::Tensor::Contiguous(a.data(), vt::DType::kF32, q.device, {m, k});
    vt::Tensor bt = vt::Tensor::Contiguous(wrp.data(), vt::DType::kF32, q.device, {n, k});
    bt.dtype = vt::DType::kQ8_0;
    bt.repacked = true;
    vt::Tensor ot = vt::Tensor::Contiguous(out.data(), vt::DType::kF32, q.device, {m, n});
    vt::MatmulBTQuant(q, ot, at, bt);
    return out;
  };

  const std::vector<float> base = run();
  for (int threads : {1, 2, 4, 20}) {
    CAPTURE(threads);
    vt::cpu::Threadpool tp(threads);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    const std::vector<float> again = run();
    vt::cpu::Threadpool::SwapForTesting(prev);
    CHECK(std::memcmp(again.data(), base.data(), base.size() * sizeof(float)) == 0);
  }
}
