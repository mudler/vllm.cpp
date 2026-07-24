// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "vt/dtype.h"
#include "vt/ops.h"

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {
Device Cpu() { return Device{DeviceType::kCPU, 0}; }
}  // namespace

TEST_CASE("silu_and_mul golden") {
  // x = [1, 2, 3, 4] with D=2: gate=[1,2], up=[3,4]
  // silu(1)=0.731059, silu(2)=1.761594
  // out = [2.193176, 7.046377]
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, to, tx);
  CHECK(out[0] == doctest::Approx(2.193176f));
  CHECK(out[1] == doctest::Approx(7.046377f));
}

TEST_CASE("silu_and_mul bf16 output: same golden within bf16 eps") {
  // Same golden as the f32 case: out = [2.193176, 7.046377], stored as bf16.
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> out(2, 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, to, tx);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(2.193176f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(7.046377f).epsilon(0.01));
}

namespace {
// Independent reference for gelu_pytorch_tanh (the exact math vLLM's
// GeluAndMul(approximate="tanh") applies), computed in f32.
float GeluTanhRef(float g) {
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.0f + std::tanh(inner));
}
}  // namespace

TEST_CASE("gelu_and_mul golden (gelu_pytorch_tanh)") {
  // x=[1,2,3,4], D=2: gate=[1,2], up=[3,4].
  // gelu_tanh(1)=0.841192, gelu_tanh(2)=1.954598 -> out=[2.523575, 7.818392]
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::GeluAndMul(q, to, tx);
  CHECK(out[0] == doctest::Approx(GeluTanhRef(1.0f) * 3.0f));
  CHECK(out[1] == doctest::Approx(GeluTanhRef(2.0f) * 4.0f));
}

TEST_CASE("gelu_and_mul bit-exact vs reference at real Gemma dims") {
  // Gemma-3-1b intermediate_size = 6912; T rows of [2*I] -> [I]. The CPU kernel
  // IS the reference impl, so this asserts the kernel matches the independent
  // formula EXACTLY (f32), bit-for-bit, at the real forward width.
  const int64_t T = 3, I = 6912;
  std::vector<float> x(static_cast<size_t>(T * 2 * I));
  for (size_t n = 0; n < x.size(); ++n)
    x[n] = std::sin(0.013f * static_cast<float>(n) + 0.5f) * 4.0f;  // spread of magnitudes
  std::vector<float> out(static_cast<size_t>(T * I), 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {T, 2 * I});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {T, I});
  Queue q{Cpu(), nullptr};
  vt::GeluAndMul(q, to, tx);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < I; ++j) {
      const float g = x[static_cast<size_t>(i * 2 * I + j)];
      const float up = x[static_cast<size_t>(i * 2 * I + I + j)];
      CHECK(out[static_cast<size_t>(i * I + j)] == GeluTanhRef(g) * up);  // exact
    }
}

TEST_CASE("gelu_and_mul bf16 output within bf16 eps") {
  std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<uint16_t> out(2, 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  vt::GeluAndMul(q, to, tx);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(GeluTanhRef(1.0f) * 3.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(GeluTanhRef(2.0f) * 4.0f).epsilon(0.01));
}

TEST_CASE("gelu_and_mul rejects odd inner dim") {
  std::vector<float> x(3, 0.0f);
  std::vector<float> out(1, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 3});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::GeluAndMul(q, to, tx), std::runtime_error);
}

TEST_CASE("mul_scalar: bf16 embedding-normalizer semantics") {
  // sqrt(1152) rounded to bf16 then multiplied (f32) and rounded to bf16 — the
  // Gemma embed scale. Assert against the same f32 arithmetic.
  const double norm = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(
      std::sqrt(1152.0f))));  // bf16-rounded sqrt(hidden)
  std::vector<float> x = {1.0f, -2.5f, 0.25f, 7.0f};
  std::vector<uint16_t> out(4, 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2, 2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::MulScalar(q, to, tx, norm);
  for (int i = 0; i < 4; ++i)
    CHECK(vt::BF16ToF32(out[i]) ==
          doctest::Approx(vt::BF16ToF32(vt::F32ToBF16(x[i] * static_cast<float>(norm)))));
}

TEST_CASE("mul_scalar: f32 exact") {
  std::vector<float> x = {1.0f, 2.0f, 3.0f};
  std::vector<float> out(3, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {3});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {3});
  Queue q{Cpu(), nullptr};
  vt::MulScalar(q, to, tx, 2.5);
  CHECK(out[0] == 2.5f);
  CHECK(out[1] == 5.0f);
  CHECK(out[2] == 7.5f);
}

namespace {
// Independent reference for the Gemma-2 logit soft-cap cap*tanh(x/cap), f32.
float SoftCapRef(float x, float cap) { return cap * std::tanh(x / cap); }
}  // namespace

TEST_CASE("soft_cap golden (final_logit_softcapping semantics)") {
  // Gemma-2 final_logit_softcapping = 30.0. Monotone squashing: |cap*tanh(x/cap)|
  // < cap, preserving order (greedy argmax invariant).
  const float cap = 30.0f;
  std::vector<float> x = {0.0f, 15.0f, -45.0f, 300.0f};
  std::vector<float> out(4, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {4});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {4});
  Queue q{Cpu(), nullptr};
  vt::SoftCap(q, to, tx, cap);
  for (int i = 0; i < 4; ++i) CHECK(out[i] == doctest::Approx(SoftCapRef(x[i], cap)));
  CHECK(out[0] == 0.0f);            // tanh(0) = 0
  CHECK(std::abs(out[3]) <= cap);  // saturates to cap (tanhf(10) rounds to 1.0 in f32)
}

TEST_CASE("soft_cap bit-exact vs reference at real Gemma-2 dims") {
  // gemma-2-2b vocab_size = 256000; one row of logits soft-capped at cap=30. The
  // CPU kernel IS the reference impl (both cap*std::tanh(x/cap), f32), so this
  // asserts the kernel matches the independent formula EXACTLY at the real width.
  const int64_t V = 256000;
  const float cap = 30.0f;
  std::vector<float> x(static_cast<size_t>(V));
  for (size_t n = 0; n < x.size(); ++n)
    x[n] = std::sin(0.0007f * static_cast<float>(n) + 0.2f) * 80.0f;  // spread past cap
  std::vector<float> out(static_cast<size_t>(V), 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, V});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, V});
  Queue q{Cpu(), nullptr};
  vt::SoftCap(q, to, tx, cap);
  for (int64_t i = 0; i < V; ++i)
    CHECK(out[static_cast<size_t>(i)] == SoftCapRef(x[static_cast<size_t>(i)], cap));  // exact
}

TEST_CASE("soft_cap greedy argmax is invariant (monotone)") {
  // The final soft-cap must not change the greedy token: argmax(cap*tanh(x/cap))
  // == argmax(x). Proves why a greedy gate does not by itself detect it.
  const float cap = 30.0f;
  std::vector<float> x = {2.0f, -7.0f, 41.0f, 40.9f, 5.0f};
  std::vector<float> out(5, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {5});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {5});
  Queue q{Cpu(), nullptr};
  vt::SoftCap(q, to, tx, cap);
  auto argmax = [](const std::vector<float>& v) {
    int a = 0; for (int i = 1; i < static_cast<int>(v.size()); ++i) if (v[i] > v[a]) a = i; return a;
  };
  CHECK(argmax(out) == argmax(x));
}

TEST_CASE("soft_cap bf16 output within bf16 eps") {
  const float cap = 30.0f;
  std::vector<float> x = {15.0f, -45.0f};
  std::vector<uint16_t> out(2, 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2});
  Queue q{Cpu(), nullptr};
  vt::SoftCap(q, to, tx, cap);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(SoftCapRef(15.0f, cap)).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(SoftCapRef(-45.0f, cap)).epsilon(0.01));
}

TEST_CASE("soft_cap rejects non-positive cap") {
  std::vector<float> x(2, 1.0f);
  std::vector<float> out(2, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::SoftCap(q, to, tx, 0.0), std::runtime_error);
}

TEST_CASE("silu_and_mul rejects odd inner dim") {
  std::vector<float> x(3, 0.0f);
  std::vector<float> out(1, 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {1, 3});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 1});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::SiluAndMul(q, to, tx), std::runtime_error);
}

TEST_CASE("embedding gathers rows by id") {
  // table 3x2: [[0,1],[10,11],[20,21]]; ids [2,0] → [[20,21],[0,1]]
  std::vector<float> table = {0, 1, 10, 11, 20, 21};
  std::vector<int32_t> ids = {2, 0};
  std::vector<float> out(4, -1.0f);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {3, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, to, tt, ti);
  CHECK(out[0] == 20.0f);
  CHECK(out[1] == 21.0f);
  CHECK(out[2] == 0.0f);
  CHECK(out[3] == 1.0f);
}

TEST_CASE("embedding bf16 output: same golden within bf16 eps") {
  // Same golden as the f32 case: [[20,21],[0,1]], stored as bf16.
  std::vector<float> table = {0, 1, 10, 11, 20, 21};
  std::vector<int32_t> ids = {2, 0};
  std::vector<uint16_t> out(4, 0xFFFF);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {3, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, to, tt, ti);
  CHECK(vt::BF16ToF32(out[0]) == doctest::Approx(20.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[1]) == doctest::Approx(21.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[2]) == doctest::Approx(0.0f).epsilon(0.01));
  CHECK(vt::BF16ToF32(out[3]) == doctest::Approx(1.0f).epsilon(0.01));
}

TEST_CASE("embedding bounds-checks ids") {
  std::vector<float> table = {0, 1};
  std::vector<int64_t> ids = {5};
  std::vector<float> out(2, 0.0f);
  Tensor tt = Tensor::Contiguous(table.data(), DType::kF32, Cpu(), {1, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI64, Cpu(), {1});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {1, 2});
  Queue q{Cpu(), nullptr};
  CHECK_THROWS_AS(vt::Embedding(q, to, tt, ti), std::runtime_error);
}
