// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
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

TEST_CASE("embedding reads a BF16 table from an unaligned byte address") {
  // Borrowed safetensors payloads are byte-addressed and may begin at an odd
  // offset. Construct that exact storage contract rather than accidentally
  // relying on vector<uint16_t>'s alignment.
  std::vector<uint8_t> storage(1 + 6 * sizeof(uint16_t));
  const uint16_t values[] = {
      vt::F32ToBF16(0.0F), vt::F32ToBF16(1.0F),
      vt::F32ToBF16(10.0F), vt::F32ToBF16(11.0F),
      vt::F32ToBF16(20.0F), vt::F32ToBF16(21.0F),
  };
  std::memcpy(storage.data() + 1, values, sizeof(values));
  REQUIRE(reinterpret_cast<uintptr_t>(storage.data() + 1) % alignof(uint16_t) ==
          1);

  std::vector<int32_t> ids = {2, 0};
  std::vector<float> out(4, -1.0F);
  Tensor tt = Tensor::Contiguous(storage.data() + 1, DType::kBF16, Cpu(), {3, 2});
  Tensor ti = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {2});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {2, 2});
  Queue q{Cpu(), nullptr};
  vt::Embedding(q, to, tt, ti);

  CHECK(out == std::vector<float>{20.0F, 21.0F, 0.0F, 1.0F});
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

// --- vt::GeluMulSeparate: the SEPARATE-buffer GeGLU (issue #377) -------------
// Gemma-4's per-layer-embedding path calls this from the SHARED forward
// (gemma4.cpp, the `ple > 0` block), with gate and up in two distinct buffers
// rather than one [T, 2D] tensor. It had only a ROCm implementation and threw
// "ROCm-only fast path in this build" everywhere else, so Gemma-4 aborted on
// its first layer on CUDA and on CPU. These cases pin the portable path.
#include "vt/fused_ops.h"

TEST_CASE("gelu_mul_separate golden matches gelu_and_mul on the same values") {
  // Same numbers as the gelu_and_mul golden above, but split across two
  // buffers: gate=[1,2], up=[3,4]. The two ops must agree by construction.
  std::vector<float> gate = {1.0f, 2.0f};
  std::vector<float> up = {3.0f, 4.0f};
  std::vector<float> out(2, 0.0f);
  Queue q{Cpu(), nullptr};
  vt::GeluMulSeparate(q, out.data(), gate.data(), up.data(), 2, DType::kF32);
  CHECK(out[0] == doctest::Approx(GeluTanhRef(1.0f) * 3.0f));
  CHECK(out[1] == doctest::Approx(GeluTanhRef(2.0f) * 4.0f));
}

TEST_CASE("gelu_mul_separate agrees with gelu_and_mul bit-for-bit at Gemma-4 PLE width") {
  // gemma-4-E4B per-layer-embedding width; T rows of n = T*ple elements. The
  // interleaved op is the reference: whatever GeluAndMul produces for
  // [gate|up], the separate-buffer form must produce byte-for-byte.
  const int64_t T = 3, ple = 256, n = T * ple;
  std::vector<float> gate(static_cast<size_t>(n)), up(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    gate[static_cast<size_t>(i)] = std::sin(0.017f * static_cast<float>(i) + 0.3f) * 3.0f;
    up[static_cast<size_t>(i)] = std::cos(0.011f * static_cast<float>(i) + 1.1f) * 2.0f;
  }
  // Reference: one [1, 2n] tensor laid out as [gate | up].
  std::vector<float> inter(static_cast<size_t>(2 * n));
  std::copy(gate.begin(), gate.end(), inter.begin());
  std::copy(up.begin(), up.end(), inter.begin() + static_cast<ptrdiff_t>(n));
  std::vector<float> ref(static_cast<size_t>(n), 0.0f);
  Tensor ti = Tensor::Contiguous(inter.data(), DType::kF32, Cpu(), {1, 2 * n});
  Tensor tr = Tensor::Contiguous(ref.data(), DType::kF32, Cpu(), {1, n});
  Queue q{Cpu(), nullptr};
  vt::GeluAndMul(q, tr, ti);

  std::vector<float> got(static_cast<size_t>(n), 0.0f);
  vt::GeluMulSeparate(q, got.data(), gate.data(), up.data(), n, DType::kF32);
  CHECK(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
}

TEST_CASE("gelu_mul_separate bf16 in/out at Gemma-4 PLE width") {
  // The shipped call site is bf16: gemma4.cpp passes DType::kBF16.
  const int64_t n = 512;
  std::vector<uint16_t> gate(static_cast<size_t>(n)), up(static_cast<size_t>(n));
  std::vector<uint16_t> out(static_cast<size_t>(n), 0);
  for (int64_t i = 0; i < n; ++i) {
    gate[static_cast<size_t>(i)] = vt::F32ToBF16(std::sin(0.021f * static_cast<float>(i)) * 3.0f);
    up[static_cast<size_t>(i)] = vt::F32ToBF16(std::cos(0.013f * static_cast<float>(i)) * 2.0f);
  }
  Queue q{Cpu(), nullptr};
  vt::GeluMulSeparate(q, out.data(), gate.data(), up.data(), n, DType::kBF16);
  for (int64_t i = 0; i < n; ++i) {
    const float g = vt::BF16ToF32(gate[static_cast<size_t>(i)]);
    const float u = vt::BF16ToF32(up[static_cast<size_t>(i)]);
    CHECK(vt::BF16ToF32(out[static_cast<size_t>(i)]) ==
          doctest::Approx(GeluTanhRef(g) * u).epsilon(0.01));
  }
}

// --- VT-ACT-ROUND-POLARITY (#1322) -----------------------------------------
// Ported from vLLM `tests/kernels/core/test_activation.py::test_act_and_mul` at
// the parity pin 5559679229bc961848b121ccdeaa8fa5d79bec98, which runs the CUDA
// kernel and `forward_native` over the same input and asserts
// `torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)` for
// silu_and_mul / mul_and_silu / gelu / gelu_tanh / fatrelu, with the comment
// that those implementations "are equivalent to the native PyTorch
// implementations, so we can do exact comparison" (test_activation.py:105-108).
//
// The behaviour that makes them exactly equal is that BOTH narrow act(gate) to
// the INPUT dtype before the multiply: `activation_kernels.cu:156-158`
// `silu_kernel` returns `(T)(((float)x) / (1.0f + expf((float)-x * alpha)))`
// and `:36` `compute` returns
// `(scalar_t)(ACT_FN(gate, alpha) * ((float)up + beta))`, while
// `activation.py:143` `F.silu(x[..., :d]) * x[..., d:]` yields a bf16 tensor
// from `F.silu` on a bf16 tensor. Upstream's dims are D in [512, 13824] and
// NUM_TOKENS in [7, 83, 2048]; the smallest of each is used here.
//
// The rounding target is the INPUT dtype, not the output dtype. Upstream never
// has the two differ (`SiluAndMul.forward_cuda` allocates `out` with
// `dtype=x.dtype`), while this seam permits an f32 input with a bf16 output, so
// the f32-input cases below pin that an f32 input rounds NOTHING and stays
// bit-identical to the pure-f32 expression. A value test cannot infer that.
namespace {
// bf16 round-trip through the tree's own codec (src/vt/dtype.cpp:297-304),
// i.e. what `act(gate)` reads back as once narrowed to a bf16 input's dtype.
float ThroughBf16(float v) { return vt::BF16ToF32(vt::F32ToBF16(v)); }
float SiluRef(float g) { return g / (1.0f + std::exp(-g)); }
}  // namespace

TEST_CASE("silu_and_mul narrows act(gate) to the bf16 INPUT dtype (upstream atol=rtol=0)") {
  const int64_t T = 7, D = 512;
  std::vector<uint16_t> x(static_cast<size_t>(T * 2 * D));
  for (size_t n = 0; n < x.size(); ++n)
    x[n] = vt::F32ToBF16(std::sin(0.0131f * static_cast<float>(n) + 0.37f) * 3.0f);
  std::vector<uint16_t> out(static_cast<size_t>(T * D), 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kBF16, Cpu(), {T, 2 * D});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {T, D});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, to, tx);
  int64_t checked = 0;
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < D; ++j) {
      const float g = vt::BF16ToF32(x[static_cast<size_t>(i * 2 * D + j)]);
      const float u = vt::BF16ToF32(x[static_cast<size_t>(i * 2 * D + D + j)]);
      // act(gate) narrowed to the input dtype, THEN multiplied, THEN stored.
      const uint16_t want = vt::F32ToBF16(ThroughBf16(SiluRef(g)) * u);
      REQUIRE(out[static_cast<size_t>(i * D + j)] == want);  // exact, atol=rtol=0
      ++checked;
    }
  CHECK(checked == T * D);
}

TEST_CASE("silu_and_mul leaves an f32 INPUT bit-identical (the target is x's dtype)") {
  // Same op, f32 input: RoundThrough is the identity on f32, so this must equal
  // the pure-f32 expression bit-for-bit whatever the output width is.
  const int64_t T = 3, D = 512;
  std::vector<float> x(static_cast<size_t>(T * 2 * D));
  for (size_t n = 0; n < x.size(); ++n)
    x[n] = std::sin(0.0113f * static_cast<float>(n) + 0.9f) * 4.0f;
  std::vector<float> out(static_cast<size_t>(T * D), 0.0f);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kF32, Cpu(), {T, 2 * D});
  Tensor to = Tensor::Contiguous(out.data(), DType::kF32, Cpu(), {T, D});
  Queue q{Cpu(), nullptr};
  vt::SiluAndMul(q, to, tx);
  int64_t checked = 0;
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < D; ++j) {
      const float g = x[static_cast<size_t>(i * 2 * D + j)];
      const float u = x[static_cast<size_t>(i * 2 * D + D + j)];
      REQUIRE(out[static_cast<size_t>(i * D + j)] == SiluRef(g) * u);  // exact
      ++checked;
    }
  CHECK(checked == T * D);
}

TEST_CASE("gelu_and_mul narrows act(gate) to the bf16 INPUT dtype (upstream atol=rtol=0)") {
  const int64_t T = 7, D = 512;
  std::vector<uint16_t> x(static_cast<size_t>(T * 2 * D));
  for (size_t n = 0; n < x.size(); ++n)
    x[n] = vt::F32ToBF16(std::cos(0.0091f * static_cast<float>(n) + 0.21f) * 3.0f);
  std::vector<uint16_t> out(static_cast<size_t>(T * D), 0);
  Tensor tx = Tensor::Contiguous(x.data(), DType::kBF16, Cpu(), {T, 2 * D});
  Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, Cpu(), {T, D});
  Queue q{Cpu(), nullptr};
  vt::GeluAndMul(q, to, tx);
  int64_t checked = 0;
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < D; ++j) {
      const float g = vt::BF16ToF32(x[static_cast<size_t>(i * 2 * D + j)]);
      const float u = vt::BF16ToF32(x[static_cast<size_t>(i * 2 * D + D + j)]);
      const uint16_t want = vt::F32ToBF16(ThroughBf16(GeluTanhRef(g)) * u);
      REQUIRE(out[static_cast<size_t>(i * D + j)] == want);  // exact, atol=rtol=0
      ++checked;
    }
  CHECK(checked == T * D);
}
