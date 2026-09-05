// The gamma's dtype is INDEPENDENT of the activation's, and the pairing that
// PRODUCTION reaches is f32 activation against a bf16 gamma (#2477).
//
// vllm.cpp original (vt runtime).
//
// WHICH PAIRING IS REAL, and an earlier revision of this file got it wrong.
// The released `unsloth/Qwen3.8-Flash-Next-GGUF` stores its norm gammas as F32
// ON DISK, but the loader does not hand that to the kernel:
// `qwen4_exp_weights.cpp`'s `LoadNormBf16` runs `DequantAll` into an f32 vector
// and then `Bf16From`, which is `MakeTensor(kBF16, ...)`
// (`glm_moe_dsa_loader.cpp:145-151`). All four QSA gammas take that path, so at
// RUNTIME every gamma is bf16. On-disk dtype is not runtime dtype.
//
// The activation is what diverges. `qwen4_exp_qsa_block.cpp:694` allocates
// `q_f32` at `DType::kF32` and `vt::AttnGateSplit` fills it, where vLLM chunks
// `q` straight off the bf16 QKV GEMM and hands it to `q_norm` unwidened
// (`vllm/model_executor/models/qwen3_next.py:384-440`). So the refusing site is
// `qwen4_exp_qsa_block.cpp:705` -- f32 activation, bf16 gamma, `Dh = 256` -- and
// the sites at `:632` and `:736` pair bf16 with bf16 and always passed.
//
// RED BEFORE GREEN, stated exactly as it was taken: the pre-fix dispatcher
// refused ANY `w.dtype != x.dtype`, so case 1 below could not have run at all.
// That refusal was observed as a server 500 from `#2477`, NOT captured by
// running this file against a pre-fix binary. The transition is therefore
// evidence that the pairing is now accepted, and not evidence about which
// pairing was at fault; case 1 exists to pin the one production actually uses.
//
// WAVE RMSTWINS (#2503) changed case 3's polarity and corrected case 2's
// reachability claim. Case 3 pinned a CUDA REFUSAL of a kF16 gamma; the CUDA
// dispatcher now serves kF16 and the case pins agreement. Case 2's claim that
// glm4/gemma*/deepseek_v2 GGUF arms present an F32 gamma was measured and is
// FALSE -- see the comment above `RunBf16ActF32W`. Every CUDA case here is
// `[SKIP]` on a host without a device, and no CI lane executes GPU tests.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
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

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

uint16_t F32ToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}
float Bf16ToF32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

constexpr int64_t kRows = 5;    // the CPU control's prompt_tokens
constexpr int64_t kProdH = 256; // qwen4_exp head_dim: attn_q_norm.weight is [256]
constexpr int64_t kIdxH = 128;  // indexer head_dim; the shape cases 2 and 3 borrow

// Widen both operands to f32, multiply, round once at the store. This is
// GemmaRMSNorm's own order and is recomputed here rather than taken from the op,
// because a gate that compares the op against itself proves consistency and not
// correctness.
std::vector<float> Reference(const std::vector<float>& x, const std::vector<float>& w,
                             int64_t rows, int64_t h, float eps) {
  std::vector<float> out(static_cast<size_t>(rows * h));
  for (int64_t r = 0; r < rows; ++r) {
    double sumsq = 0.0;
    for (int64_t j = 0; j < h; ++j) {
      const float v = x[static_cast<size_t>(r * h + j)];
      sumsq += static_cast<double>(v) * static_cast<double>(v);
    }
    const float inv =
        1.0f / std::sqrt(static_cast<float>(sumsq / static_cast<double>(h)) + eps);
    for (int64_t j = 0; j < h; ++j)
      out[static_cast<size_t>(r * h + j)] =
          x[static_cast<size_t>(r * h + j)] * inv * (1.0f + w[static_cast<size_t>(j)]);
  }
  return out;
}

std::vector<float> SpreadX(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] =
        std::sin(static_cast<float>(i) * 0.37f) * (1.0f + 0.01f * static_cast<float>(i % 17));
  return v;
}
std::vector<float> SpreadW(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j)
    v[static_cast<size_t>(j)] = 0.05f * std::cos(static_cast<float>(j) * 0.11f);
  return v;
}

// ─── CASE 1 SHAPE: f32 activation, bf16 gamma, bf16 out. THE PRODUCTION ONE.
// Instantiates RmsNormRowKernel<float, __nv_bfloat16, __nv_bfloat16, float>.
std::vector<uint16_t> RunF32ActBf16W(Device dev, const std::vector<float>& x,
                                     const std::vector<uint16_t>& w, int64_t h) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * h), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<float*>(x.data()), DType::kF32, dev, {kRows, h});
    Tensor tw = Tensor::Contiguous(const_cast<uint16_t*>(w.data()), DType::kBF16, dev, {h});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, h});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(float));
  void* wd = b.Alloc(w.size() * sizeof(uint16_t));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(float));
  b.Copy(q, wd, w.data(), w.size() * sizeof(uint16_t));
  Tensor tx = Tensor::Contiguous(xd, DType::kF32, dev, {kRows, h});
  Tensor tw = Tensor::Contiguous(wd, DType::kBF16, dev, {h});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, h});
  vt::RmsNorm(q, to, tx, tw, args);
  b.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
  return out;
}

// ─── CASE 2 SHAPE: bf16 activation, f32 gamma. Newly ADMITTED by #2477, and
// reached by NOTHING. The claim that stood here -- that the GGUF arms of glm4,
// gemma/gemma2/gemma3 and deepseek_v2 keep an F32 gamma -- was #2503's and it is
// FALSE in both halves. Those five architectures have no GGUF arm at all: each
// refuses `source.kind != kSafetensors` by name at its registry door
// (`gemma_registry.cpp:48-51`, `gemma2_registry.cpp:49-52`,
// `gemma3_registry.cpp:50-53`, `glm4_registry.cpp:50-53`,
// `deepseek_v2_registry.cpp:68-71`), and their safetensors loaders take every
// gamma through `dense_loaders::LoadBf16Direct`, which is `kBF16` unconditionally
// (`dense_weight_loaders.h:355-374`).
//
// The f32 gammas this tree DOES build -- kimi-linear (`kimi_linear.h:179,:254`,
// f32 `std::vector`), Laguna (`laguna.cpp:2568`), the MiniMax-H3 encoder
// (`minimax_h3_encoder_sharded.cpp:228`) and Qwen3.5's q-norm
// (`qwen3_5.cpp:5329`) -- all pair the f32 gamma with an f32 ACTIVATION, so they
// satisfied the old equality check too. This pairing is therefore kept as the
// contract `vt::RmsNorm` promises and the CPU arm serves, NOT as a model's.
std::vector<uint16_t> RunBf16ActF32W(Device dev, const std::vector<uint16_t>& x,
                                     const std::vector<float>& w, int64_t h) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * h), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, dev, {kRows, h});
    Tensor tw = Tensor::Contiguous(const_cast<float*>(w.data()), DType::kF32, dev, {h});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, h});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(uint16_t));
  void* wd = b.Alloc(w.size() * sizeof(float));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(uint16_t));
  b.Copy(q, wd, w.data(), w.size() * sizeof(float));
  Tensor tx = Tensor::Contiguous(xd, DType::kBF16, dev, {kRows, h});
  Tensor tw = Tensor::Contiguous(wd, DType::kF32, dev, {h});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, h});
  vt::RmsNorm(q, to, tx, tw, args);
  b.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
  return out;
}

// Gamma values that are EXACTLY representable in f32, binary16 and bfloat16 at
// once: every one is k/64 with |k| <= 8, so it has at most four significant bits
// and an exponent well inside binary16's range. bfloat16 is the binding
// constraint (8 significant bits), and binary16's range is the other one. This is
// what makes the three-dtype comparison in case 3 an EQUALITY rather than a
// tolerance.
std::vector<float> ExactW(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j)
    v[static_cast<size_t>(j)] = static_cast<float>((j % 17) - 8) / 64.0f;
  return v;
}

// ─── CASE 3 SHAPE: bf16 activation, 16-bit gamma of the CALLER'S dtype tag.
// One function for both kF16 and kBF16 so the two arms cannot differ in anything
// but the tag, which is the only variable case 3 is about.
std::vector<uint16_t> RunBf16ActW16(Device dev, const std::vector<uint16_t>& x,
                                    const std::vector<uint16_t>& w, DType wdt, int64_t h) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * h), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, dev, {kRows, h});
    Tensor tw = Tensor::Contiguous(const_cast<uint16_t*>(w.data()), wdt, dev, {h});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, h});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(uint16_t));
  void* wd = b.Alloc(w.size() * sizeof(uint16_t));
  void* od = b.Alloc(out.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(uint16_t));
  b.Copy(q, wd, w.data(), w.size() * sizeof(uint16_t));
  Tensor tx = Tensor::Contiguous(xd, DType::kBF16, dev, {kRows, h});
  Tensor tw = Tensor::Contiguous(wd, wdt, dev, {h});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, h});
  vt::RmsNorm(q, to, tx, tw, args);
  b.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(od); b.DestroyQueue(q);
  return out;
}

size_t BadCount(const std::vector<uint16_t>& got, const std::vector<float>& want) {
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - want[i]) > 0.01f * (std::fabs(want[i]) + 1e-3f)) ++bad;
  return bad;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// CASE 1. The pairing qwen4_exp_qsa_block.cpp:705 actually issues.
TEST_CASE("rmsnorm: f32 activation against a bf16 gamma is the production pairing") {
  const std::vector<float> x = SpreadX(kRows * kProdH);
  const std::vector<float> wf = SpreadW(kProdH);
  std::vector<uint16_t> wb(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wb[static_cast<size_t>(j)] = F32ToBf16(wf[static_cast<size_t>(j)]);
  // The reference must use the ROUNDED gamma, since that is what the op reads.
  std::vector<float> wr(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wr[static_cast<size_t>(j)] = Bf16ToF32(wb[static_cast<size_t>(j)]);

  const std::vector<uint16_t> got = RunF32ActBf16W(Cpu(), x, wb, kProdH);
  CHECK(BadCount(got, Reference(x, wr, kRows, kProdH, 1e-6f)) == 0);

  // The gamma must be LOAD-BEARING or the tolerance above is vacuous.
  std::vector<uint16_t> bumped = wb;
  bumped[7] = F32ToBf16(wr[7] + 4.0f);
  CHECK(got[7] != RunF32ActBf16W(Cpu(), x, bumped, kProdH)[7]);
}

TEST_CASE("rmsnorm: CUDA runs the production f32-activation/bf16-gamma pairing") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: production pairing device arm NOT exercised\n");
    return;
  }
  const std::vector<float> x = SpreadX(kRows * kProdH);
  const std::vector<float> wf = SpreadW(kProdH);
  std::vector<uint16_t> wb(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j) wb[static_cast<size_t>(j)] = F32ToBf16(wf[static_cast<size_t>(j)]);
  const std::vector<uint16_t> want = RunF32ActBf16W(Cpu(), x, wb, kProdH);
  const std::vector<uint16_t> got = RunF32ActBf16W(Device{DeviceType::kCUDA, 0}, x, wb, kProdH);
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - Bf16ToF32(want[i])) >
        0.01f * (std::fabs(Bf16ToF32(want[i])) + 1e-3f)) ++bad;
  CHECK(bad == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 2. The pairing this change newly ADMITS. Not qwen4_exp's, but reachable.
TEST_CASE("rmsnorm: bf16 activation against an f32 gamma, the newly admitted pairing") {
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  std::vector<float> xr(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xr[i] = Bf16ToF32(xb[i]);
  const std::vector<float> w = SpreadW(kIdxH);

  const std::vector<uint16_t> got = RunBf16ActF32W(Cpu(), xb, w, kIdxH);
  CHECK(BadCount(got, Reference(xr, w, kRows, kIdxH, 1e-6f)) == 0);
}

TEST_CASE("rmsnorm: CUDA runs the bf16-activation/f32-gamma pairing") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: admitted-pairing device arm NOT exercised\n");
    return;
  }
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  const std::vector<float> w = SpreadW(kIdxH);
  const std::vector<uint16_t> want = RunBf16ActF32W(Cpu(), xb, w, kIdxH);
  const std::vector<uint16_t> got = RunBf16ActF32W(Device{DeviceType::kCUDA, 0}, xb, w, kIdxH);
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i)
    if (std::fabs(Bf16ToF32(got[i]) - Bf16ToF32(want[i])) >
        0.01f * (std::fabs(Bf16ToF32(want[i])) + 1e-3f)) ++bad;
  CHECK(bad == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 3. The divergence #2503 named, now CLOSED by serving kF16 (wave RMSTWINS).
//
// THIS CASE CHANGED POLARITY. #2493 pinned it as a CUDA REFUSAL:
// `CHECK_THROWS_WITH(..., doctest::Contains("unsupported weight dtype"))`, because
// `vt::RmsNorm` admits `IsFloat(weight.dtype)` (`ops.cpp:1030`, `:24`), which
// includes kF16, the CPU kernel widened a kF16 gamma like any other
// (`cpu_ops.cpp:554-557`), and the CUDA dispatcher refused it --- a device arm
// refusing a dtype its CPU sibling accepts, which `cuda_qwen4_exp.cu:60-62` says
// must be recorded. `cuda_ops.cu` now dispatches `case DType::kF16` to
// `LaunchRmsNorm<Tin, __half>`, so the case pins AGREEMENT instead.
//
// NO PRODUCTION LOADER BUILDS A kF16 GAMMA, and the wave that added the arm
// measured that rather than assuming it: every rank-1 norm weight in this tree is
// kF32 or kBF16, and `nemotron_h_weights.cpp:1044-1050` --- the one config-driven
// dtype channel that could produce one --- refuses `"float16"`/`"half"` by name.
// The arm is served because the SEAM promises it, the CPU arm keeps that promise,
// `CudaPlatform::supported_dtypes` lists kF16 (`platforms/cuda.cpp:111-113`), and
// vLLM upcasts any float gamma (`self.weight.float()`,
// `vllm/models/qwen4_exp/nvidia/ple_layer.py:80`, vLLM origin/main cdefd9d499 ---
// a forward reference, off this project's pin 5559679229, see #2502).
//
// THE INVARIANT IS EQUALITY, NOT A TOLERANCE. The three gammas below hold
// BIT-IDENTICAL values --- every element is k/64 with |k| <= 8, exactly
// representable in f32, binary16 and bfloat16 alike --- so all three runs must
// produce byte-identical output. A tolerance would have passed a build that read
// the f16 gamma through a `const float*`: that reads 2H bytes out of an H*2-byte
// buffer, and the first half of the garbage is small denormals near zero, which a
// 1%-relative bar absorbs. Equality does not absorb it.
TEST_CASE("rmsnorm: the gamma dtype does not change the answer (f32/f16/bf16)") {
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);

  const std::vector<float> w32 = ExactW(kIdxH);
  std::vector<uint16_t> w16(static_cast<size_t>(kIdxH)), wbf(static_cast<size_t>(kIdxH));
  for (int64_t j = 0; j < kIdxH; ++j) {
    const size_t u = static_cast<size_t>(j);
    w16[u] = vt::F32ToF16(w32[u]);
    wbf[u] = F32ToBf16(w32[u]);
    // The premise of the equality bar: all three encodings hold the SAME value.
    REQUIRE(vt::F16ToF32(w16[u]) == w32[u]);
    REQUIRE(Bf16ToF32(wbf[u]) == w32[u]);
  }

  const std::vector<uint16_t> from_f32 = RunBf16ActF32W(Cpu(), xb, w32, kIdxH);
  const std::vector<uint16_t> from_f16 = RunBf16ActW16(Cpu(), xb, w16, DType::kF16, kIdxH);
  const std::vector<uint16_t> from_bf16 = RunBf16ActW16(Cpu(), xb, wbf, DType::kBF16, kIdxH);

  // Against an independent reference first, so the three-way equality below cannot
  // be satisfied by three identically wrong runs.
  std::vector<float> xr(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xr[i] = Bf16ToF32(xb[i]);
  CHECK(BadCount(from_f32, Reference(xr, w32, kRows, kIdxH, 1e-6f)) == 0);

  CHECK(from_f16 == from_f32);
  CHECK(from_bf16 == from_f32);

  // The gamma must be LOAD-BEARING through the f16 arm specifically, or the
  // equality above would also hold for a kernel that ignored `w` entirely.
  std::vector<uint16_t> bumped = w16;
  bumped[7] = vt::F32ToF16(w32[7] + 4.0f);
  CHECK(RunBf16ActW16(Cpu(), xb, bumped, DType::kF16, kIdxH)[7] != from_f16[7]);
}

TEST_CASE("rmsnorm: CUDA serves a kF16 gamma and agrees with CPU byte-for-byte") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: kF16-gamma device arm NOT exercised\n");
    return;
  }
  const std::vector<float> xf = SpreadX(kRows * kIdxH);
  std::vector<uint16_t> xb(xf.size());
  for (size_t i = 0; i < xf.size(); ++i) xb[i] = F32ToBf16(xf[i]);
  const std::vector<float> w32 = ExactW(kIdxH);
  std::vector<uint16_t> w16(static_cast<size_t>(kIdxH));
  for (int64_t j = 0; j < kIdxH; ++j)
    w16[static_cast<size_t>(j)] = vt::F32ToF16(w32[static_cast<size_t>(j)]);

  const Device gpu{DeviceType::kCUDA, 0};
  // Byte equality, not a tolerance: the f32 and f16 gammas hold the same values,
  // so the CUDA f16 arm must produce what the CUDA f32 arm produces. A
  // fall-through that sent kF16 to `LaunchRmsNorm<Tin, float>` would read the
  // [128] half buffer as 128 floats and fail here rather than being absorbed.
  CHECK(RunBf16ActW16(gpu, xb, w16, DType::kF16, kIdxH) ==
        RunBf16ActF32W(gpu, xb, w32, kIdxH));
  // And against the host, which is the divergence #2503 was about.
  CHECK(RunBf16ActW16(gpu, xb, w16, DType::kF16, kIdxH) ==
        RunBf16ActW16(Cpu(), xb, w16, DType::kF16, kIdxH));
}

// ─────────────────────────────────────────────────────────────────────────────
// CASE 4. THE GAMMA'S BYTE ADDRESS. #2540.
//
// A safetensors payload starts at `8 + <JSON header length>`
// (`safetensors_reader.cpp:78`) and a header length is arbitrary, so a bf16
// gamma begins on an ODD byte in roughly half of all checkpoints.
// `BorrowStTensorBytes` hands those bytes to the kernel verbatim -- that is what
// direct upload IS -- and `WidenRowToF32` (`cpu_matmul_elem.cpp:577`) read them
// through a `const uint16_t*`. `-fsanitize=alignment` aborted `test_dots3_note_attn`
// and `test_muse_glimmer_text` on exactly this operand, reached through
// `ModelRegistry::Forward` on both.
//
// TWO THINGS ARE GATED HERE AND THEY FAIL DIFFERENTLY.
//
//   * Under `sanitize-cpu (address,undefined)` the pre-fix binary ABORTS in this
//     case with `load of misaligned address ... for type 'const uint16_t'`. That
//     is the lane's red.
//   * Without a sanitizer the pre-fix binary PASSES on x86, which is why this
//     class survived three UBSan sweeps. The equality below is therefore aimed at
//     the byte-cursor arithmetic the fix introduces: an element stride counted in
//     words instead of bytes reads the wrong half of the buffer and lands here on
//     any lane.
//
// The `REQUIRE` on the parity is not decoration. A case that quietly landed on an
// EVEN address would exercise nothing and still report a pass, which is the
// "instrument that never ran" trap.
TEST_CASE("rmsnorm: a bf16 gamma at an ODD byte address is read as its bytes") {
  const std::vector<float> x = SpreadX(kRows * kProdH);
  const std::vector<float> wf = SpreadW(kProdH);
  std::vector<uint16_t> wb(static_cast<size_t>(kProdH));
  for (int64_t j = 0; j < kProdH; ++j)
    wb[static_cast<size_t>(j)] = F32ToBf16(wf[static_cast<size_t>(j)]);

  // The same gamma bytes, one byte into an over-allocated buffer.
  std::vector<unsigned char> raw(wb.size() * sizeof(uint16_t) + 1, 0);
  std::memcpy(raw.data() + 1, wb.data(), wb.size() * sizeof(uint16_t));
  void* odd = raw.data() + 1;
  REQUIRE(reinterpret_cast<std::uintptr_t>(odd) % 2 == 1);

  std::vector<uint16_t> got(static_cast<size_t>(kRows * kProdH), 0);
  Queue q{Cpu(), nullptr};
  Tensor tx = Tensor::Contiguous(const_cast<float*>(x.data()), DType::kF32, Cpu(), {kRows, kProdH});
  Tensor tw = Tensor::Contiguous(odd, DType::kBF16, Cpu(), {kProdH});
  Tensor to = Tensor::Contiguous(got.data(), DType::kBF16, Cpu(), {kRows, kProdH});
  vt::RmsNorm(q, to, tx, tw, RmsNormArgs{1e-6f, /*gemma=*/true});

  // Byte equality against the SAME gamma at an even address. Not a tolerance: a
  // stride defect moves whole elements, and a tolerance on a smooth gamma could
  // absorb a neighbour's value.
  const std::vector<uint16_t> aligned = RunF32ActBf16W(Cpu(), x, wb, kProdH);
  CHECK(got == aligned);

  // The odd gamma must be LOAD-BEARING, or the equality above would also hold for
  // a kernel that read the aligned copy or ignored `w` entirely.
  std::vector<unsigned char> bumped = raw;
  const uint16_t hot = F32ToBf16(Bf16ToF32(wb[7]) + 4.0f);
  std::memcpy(bumped.data() + 1 + 7 * sizeof(uint16_t), &hot, sizeof(hot));
  std::vector<uint16_t> got2(static_cast<size_t>(kRows * kProdH), 0);
  Tensor tw2 = Tensor::Contiguous(bumped.data() + 1, DType::kBF16, Cpu(), {kProdH});
  Tensor to2 = Tensor::Contiguous(got2.data(), DType::kBF16, Cpu(), {kRows, kProdH});
  vt::RmsNorm(q, to2, tx, tw2, RmsNormArgs{1e-6f, /*gemma=*/true});
  CHECK(got2[7] != got[7]);
}
