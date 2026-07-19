// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"  // F32ToF8E4M3
#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::RmsNormArgs;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("rmsnorm golden row") {
  // x = [3,4]; mean(x^2) = 12.5; rms = sqrt(12.5); w = [2, 0.5]
  // out = [3/3.53553*2, 4/3.53553*0.5] = [1.697056, 0.565685]
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> w = {2.0f, 0.5f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));
}

TEST_CASE("rmsnorm gemma variant uses (1+w)") {
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> w = {1.0f, -0.5f};  // effective weights [2, 0.5]
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, true});
  CHECK(out[0] == doctest::Approx(1.697056f));
  CHECK(out[1] == doctest::Approx(0.565685f));
}

TEST_CASE("rmsnorm eps matters for zero rows") {
  std::vector<float> x = {0.0f, 0.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(2, -1.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{1e-6f, false});
  CHECK(out[0] == 0.0f);  // no NaN
  CHECK(out[1] == 0.0f);
}

TEST_CASE("fused residual add updates residual stream then normalizes") {
  // x = [1,2], residual = [2,2] → sum = [3,4] (residual becomes [3,4])
  // norm of [3,4] with w=[1,1]: [0.848528, 1.131371]
  std::vector<float> x = {1.0f, 2.0f};
  std::vector<float> res = {2.0f, 2.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false}, &tr);
  CHECK(res[0] == 3.0f);
  CHECK(res[1] == 4.0f);
  CHECK(out[0] == doctest::Approx(0.848528f));
  CHECK(out[1] == doctest::Approx(1.131371f));
}

TEST_CASE("rmsnorm multi-row normalizes independently") {
  std::vector<float> x = {3.0f, 4.0f, 6.0f, 8.0f};
  std::vector<float> w = {1.0f, 1.0f};
  std::vector<float> out(4, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(0.848528f));  // row 0
  CHECK(out[2] == doctest::Approx(0.848528f));  // row 1 same direction
}

TEST_CASE("rmsnorm bf16 output: same golden within bf16 eps") {
  // Same golden as the f32 case: out = [1.697056, 0.565685], stored as bf16.
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> w = {2.0f, 0.5f};
  std::vector<uint16_t> out(2, 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(1.697056f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(0.565685f).epsilon(0.01));
}

TEST_CASE("rmsnorm fused residual: f32 (full precision) or bf16 (vLLM model dtype)") {
  // The residual stream may be f32 (full-precision) OR bf16. A bf16 residual mirrors
  // vLLM's bf16 model_config.dtype: fused_add_rms_norm adds x into the residual,
  // stores it back in the model dtype, and accumulates the variance in f32. Both add
  // x into the residual IN PLACE and normalize that sum. Here 1+2=3 and 2+2=4 are
  // exactly representable in bf16, so f32 and bf16 residuals give the same result.
  std::vector<float> w = {1.0f, 1.0f};

  SUBCASE("f32 residual (byte-identical to the pre-bf16 path)") {
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> res = {2.0f, 2.0f};
    std::vector<float> out(2, 0.0f);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
    Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, Cpu(), {1, 2});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
    Queue q{Cpu(), nullptr};
    vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false}, &tr);
    CHECK(res[0] == doctest::Approx(3.0f));  // new stream, updated in place
    CHECK(res[1] == doctest::Approx(4.0f));
    CHECK(out[0] == doctest::Approx(0.848528f));
    CHECK(out[1] == doctest::Approx(1.131371f));
  }

  SUBCASE("bf16 residual (now valid — mirrors vLLM)") {
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<uint16_t> res = {0x4000, 0x4000};  // bf16(2.0), bf16(2.0)
    std::vector<float> out(2, 0.0f);
    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 2});
    Tensor tr = Tensor::Contiguous(res.data(), DType::kBF16, Cpu(), {1, 2});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {2});
    Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
    Queue q{Cpu(), nullptr};
    vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false}, &tr);
    CHECK(vt::BF16ToF32(res[0]) == doctest::Approx(3.0f));  // stored bf16, in place
    CHECK(vt::BF16ToF32(res[1]) == doctest::Approx(4.0f));
    CHECK(out[0] == doctest::Approx(0.848528f));
    CHECK(out[1] == doctest::Approx(1.131371f));
  }
}

// RmsNormQuantFp8 (fused fp8 RMSNorm -> static quant) must be BIT-IDENTICAL to the
// split path RmsNorm(bf16 out, residual) then static_scaled_fp8_quant of that bf16
// (BF16ToF32(bf16) / input_scale). Validated on CPU against vllm::F32ToF8E4M3 (the
// same RNE-saturating e4m3 the CPU kernel's F32ToFp8 mirrors), covering the two
// residual dtypes and the optional bf16 side-output.
TEST_CASE("rmsnorm_quant_fp8 is bit-identical to RmsNorm(bf16)+static fp8 quant") {
  const int64_t T = 5, H = 96;  // H a multiple of 16 (fp8 GEMM alignment)
  const float eps = 1e-6f, input_scale = 0.035f, inv_scale = 1.0f / input_scale;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> ux(-3.0f, 3.0f);

  auto run = [&](bool bf16_residual, bool want_bf16_out) {
    std::vector<float> x(static_cast<size_t>(T) * H);
    std::vector<float> w(H);
    for (auto& v : x) v = ux(rng);
    for (auto& v : w) v = ux(rng) * 0.1f;

    // Reference: vt::RmsNorm to a bf16 output (its own residual copy), then quant.
    std::vector<uint16_t> ref_bf16(static_cast<size_t>(T) * H, 0);
    std::vector<float> res_f32_ref(static_cast<size_t>(T) * H);
    std::vector<uint16_t> res_bf16_ref(static_cast<size_t>(T) * H, 0);
    for (int64_t i = 0; i < T * H; ++i) {
      const float rv = ux(rng);
      res_f32_ref[static_cast<size_t>(i)] = rv;
      res_bf16_ref[static_cast<size_t>(i)] = vt::F32ToBF16(rv);
    }
    std::vector<float> res_f32_fused = res_f32_ref;    // separate copies (updated in place)
    std::vector<uint16_t> res_bf16_fused = res_bf16_ref;

    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {T, H});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {H});
    Tensor to_bf16 = Tensor::Contiguous(ref_bf16.data(), DType::kBF16, Cpu(), {T, H});
    Tensor tr_ref = bf16_residual
                        ? Tensor::Contiguous(res_bf16_ref.data(), DType::kBF16, Cpu(), {T, H})
                        : Tensor::Contiguous(res_f32_ref.data(), DType::kF32, Cpu(), {T, H});
    Queue q{Cpu(), nullptr};
    vt::RmsNorm(q, to_bf16, tx, tw, RmsNormArgs{eps, true}, &tr_ref);
    std::vector<uint8_t> ref_fp8(static_cast<size_t>(T) * H);
    for (size_t i = 0; i < ref_fp8.size(); ++i)
      ref_fp8[i] = vllm::F32ToF8E4M3(vt::BF16ToF32(ref_bf16[i]) * inv_scale);

    // Fused: RmsNormQuantFp8 producing fp8 (+ optional bf16), same inputs.
    std::vector<uint8_t> got_fp8(static_cast<size_t>(T) * H, 0);
    std::vector<uint16_t> got_bf16(static_cast<size_t>(T) * H, 0);
    Tensor tofp8 = Tensor::Contiguous(got_fp8.data(), DType::kI8, Cpu(), {T, H});
    Tensor tobf16 = Tensor::Contiguous(got_bf16.data(), DType::kBF16, Cpu(), {T, H});
    Tensor tr_fused = bf16_residual
                          ? Tensor::Contiguous(res_bf16_fused.data(), DType::kBF16, Cpu(), {T, H})
                          : Tensor::Contiguous(res_f32_fused.data(), DType::kF32, Cpu(), {T, H});
    vt::RmsNormQuantFp8(q, tofp8, want_bf16_out ? &tobf16 : nullptr, tx, tw,
                        RmsNormArgs{eps, true}, &tr_fused, input_scale);

    CHECK(got_fp8 == ref_fp8);              // fp8 byte-exact vs the split path
    if (want_bf16_out) CHECK(got_bf16 == ref_bf16);  // bf16 side-output byte-exact
    if (bf16_residual)
      CHECK(res_bf16_fused == res_bf16_ref);         // residual stream updated identically
    else
      CHECK(res_f32_fused == res_f32_ref);
  };

  run(/*bf16_residual=*/false, /*want_bf16_out=*/true);
  run(/*bf16_residual=*/false, /*want_bf16_out=*/false);
  run(/*bf16_residual=*/true, /*want_bf16_out=*/true);
  run(/*bf16_residual=*/true, /*want_bf16_out=*/false);
}

// RmsNormGatedQuantFp8 (fused GDN gated-RMSNorm -> static fp8 quant) must be
// BIT-IDENTICAL to the split path RmsNormGated(bf16 out) then static fp8 quant of that
// bf16 (F32ToF8E4M3(BF16ToF32(bf16) / input_scale)) — the 35B GDN out_proj W8A8 path.
// Covers both the silu and sigmoid gate, over a contiguous rank-2 [rows,D] gate.
TEST_CASE("rmsnorm_gated_quant_fp8 is bit-identical to RmsNormGated(bf16)+static fp8 quant") {
  const int64_t rows = 7, D = 128;  // D==128 is the GDN Dv (the production shape)
  const float eps = 1e-6f, input_scale = 0.042f, inv_scale = 1.0f / input_scale;
  std::mt19937 rng(20260719);
  std::uniform_real_distribution<float> ux(-3.0f, 3.0f);

  auto run = [&](bool sigmoid_gate) {
    std::vector<float> x(static_cast<size_t>(rows) * D);
    std::vector<float> gate(static_cast<size_t>(rows) * D);
    std::vector<float> w(D);
    for (auto& v : x) v = ux(rng);
    for (auto& v : gate) v = ux(rng);
    for (auto& v : w) v = ux(rng) * 0.1f;

    Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {rows, D});
    Tensor tg = Tensor::Contiguous(gate.data(), DType::kF32, Cpu(), {rows, D});
    Tensor tw = Tensor::Contiguous(w.data(), DType::kF32, Cpu(), {D});
    Queue q{Cpu(), nullptr};
    vt::RmsNormGatedArgs args{eps, sigmoid_gate};

    // Reference: vt::RmsNormGated to a bf16 output, then the static fp8 quant.
    std::vector<uint16_t> ref_bf16(static_cast<size_t>(rows) * D, 0);
    Tensor to_bf16 = Tensor::Contiguous(ref_bf16.data(), DType::kBF16, Cpu(), {rows, D});
    vt::RmsNormGated(q, to_bf16, tx, tg, tw, args);
    std::vector<uint8_t> ref_fp8(static_cast<size_t>(rows) * D);
    for (size_t i = 0; i < ref_fp8.size(); ++i)
      ref_fp8[i] = vllm::F32ToF8E4M3(vt::BF16ToF32(ref_bf16[i]) * inv_scale);

    // Fused: RmsNormGatedQuantFp8 emitting the fp8 activation directly.
    std::vector<uint8_t> got_fp8(static_cast<size_t>(rows) * D, 0);
    Tensor tofp8 = Tensor::Contiguous(got_fp8.data(), DType::kI8, Cpu(), {rows, D});
    vt::RmsNormGatedQuantFp8(q, tofp8, tx, tg, tw, args, input_scale);

    CHECK(got_fp8 == ref_fp8);  // fp8 byte-exact vs the split RmsNormGated+quant path
  };

  run(/*sigmoid_gate=*/false);  // silu (Qwen GDN default)
  run(/*sigmoid_gate=*/true);   // sigmoid
}

TEST_CASE("rmsnorm accepts bf16 inputs via f32 conversion") {
  // bf16(3.0)=0x4040, bf16(4.0)=0x4080 are exact; w bf16(1.0)=0x3F80
  std::vector<uint16_t> x = {0x4040, 0x4080};
  std::vector<uint16_t> w = {0x3F80, 0x3F80};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kBF16, Cpu(), {1, 2});
  Tensor tw = Tensor::Contiguous(w.data(), DType::kBF16, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{0.0f, false});
  CHECK(out[0] == doctest::Approx(0.848528f));
  CHECK(out[1] == doctest::Approx(1.131371f));
}
