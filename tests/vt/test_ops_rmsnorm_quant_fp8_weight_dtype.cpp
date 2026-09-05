// `RmsNormQuantFp8` reads its gamma in the GAMMA's own dtype (#2492).
//
// vllm.cpp original (vt runtime); the behaviour it pins is upstream's.
//
// THE TWIN, AND WHY IT IS LATENT RATHER THAN LIVE. `RmsNormQuantFp8RowKernel`
// took its gamma as `const Tin* w` -- the ACTIVATION's element type -- exactly as
// `RmsNormRowKernel` did before #2477, so `RmsNormQuantFp8KernelCuda` had to hold
// `VT_CHECK(w.dtype == x.dtype, "cuda rmsnorm_quant_fp8: weight dtype must match
// x")`. Its one production consumer is Qwen3.5's fused input layernorm
// (`qwen3_5.cpp:6866` through `vt::FusedChain(vt::kRmsNormQuantFp8, ...)` and
// `:6869` through the op directly), whose gamma is `ResidentWeight(d,
// layer.input_layernorm, {H})` -- bf16, from `LoadBf16Direct`
// (`qwen3_5_weights.cpp:1100`) or from the GGUF arm's `OwnNormMinus1`
// (`qwen3_5_gguf_weights.cpp:1398`), which is `MakeOwned(vt::DType::kBF16, ...)`
// -- against a bf16 activation. So the equality held there and the weld never
// fired in production. It is a defect anyway, for two reasons that do not depend
// on a consumer:
//
//   1. `vt::RmsNormQuantFp8` admits `IsFloat(weight.dtype)` (`ops.cpp:748`), so
//      the SEAM promises f32/f16/bf16 and the CUDA arm kept only one of the three.
//   2. The CPU sibling reads the gamma through `LoadF32(w, j)`
//      (`cpu_ops.cpp:1042`) and therefore accepts all three. A device arm
//      refusing a dtype its CPU sibling accepts is the divergence
//      `cuda_qwen4_exp.cu:60-62` says must not stand unrecorded.
//
// vLLM never couples the two dtypes either: `GemmaRMSNorm` upcasts the gamma on
// its own, `normalized * (1.0 + self.weight.float())` at
// `vllm/models/qwen4_exp/nvidia/ple_layer.py:80`. That citation is read at vLLM
// origin/main `cdefd9d499`, which is AHEAD of this project's pin `5559679229` --
// the pin has no `vllm/models/qwen4_exp/` at all. It is a forward reference and
// is not gateable against the pin (#2502).
//
// WHICH SITE EACH CASE MODELS. None of them models a model. The bf16/bf16 case
// models `qwen3_5.cpp:6869`, the only production caller, and exists so the
// mixed-dtype cases have a known-good anchor. The f32-gamma and f16-gamma cases
// model the SEAM's own admission at `ops.cpp:748` -- the contract every device
// arm owes -- and nothing else. Saying so is the point: a test that claimed a
// consumer here would be claiming one that does not exist.
//
// THE INVARIANT IS BYTE EQUALITY, NOT A TOLERANCE. The three gammas hold
// bit-identical values (every element is k/64 with |k| <= 8, exactly
// representable in f32, binary16 and bfloat16 alike), so all three runs must emit
// byte-identical fp8. A tolerance would have passed a build that read the f16 or
// bf16 gamma through a `const float*`: that reads 2H bytes out of an H*2-byte
// buffer, and much of the garbage is denormal-small, which a relative bar
// absorbs. Byte equality does not.
//
// RED BEFORE GREEN, stated exactly as it was taken: the CUDA arms below are
// `[SKIP]` on every machine this suite has run on, and no CI lane executes GPU
// tests. The transition proved locally is the CPU one, which was ALREADY green --
// the CPU kernel never had the weld. The CUDA transition is asserted from the
// source change and compiled by `cuda-fat-build`; it is not measured here.
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

constexpr int64_t kT = 5;
constexpr int64_t kH = 96;  // a multiple of 16, as the fp8 GEMM alignment wants
constexpr float kEps = 1e-6f;
constexpr float kScale = 0.035f;

// Exactly representable in f32, binary16 and bfloat16 at once (see the header).
std::vector<float> ExactW(int64_t n) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t j = 0; j < n; ++j)
    v[static_cast<size_t>(j)] = static_cast<float>((j % 17) - 8) / 64.0f;
  return v;
}

std::vector<uint16_t> SpreadXBf16(int64_t n) {
  std::vector<uint16_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] = vt::F32ToBF16(
        std::sin(static_cast<float>(i) * 0.37f) * (1.0f + 0.01f * static_cast<float>(i % 17)));
  return v;
}

struct Out {
  std::vector<uint8_t> fp8;
  std::vector<uint16_t> bf16;
  bool operator==(const Out& o) const { return fp8 == o.fp8 && bf16 == o.bf16; }
};

// bf16 activation, f32 residual, gamma bytes reinterpreted at `wdt`. One function
// for every gamma dtype so the arms cannot differ in anything but the tag.
Out RunFp8(Device dev, const std::vector<uint16_t>& x, const void* w, DType wdt, size_t wbytes) {
  Out out;
  out.fp8.assign(static_cast<size_t>(kT * kH), 0);
  out.bf16.assign(static_cast<size_t>(kT * kH), 0);
  std::vector<float> res(static_cast<size_t>(kT * kH));
  for (size_t i = 0; i < res.size(); ++i)
    res[i] = 0.25f * std::cos(static_cast<float>(i) * 0.19f);
  const RmsNormArgs args{kEps, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(x.data()), DType::kBF16, dev, {kT, kH});
    Tensor tw = Tensor::Contiguous(const_cast<void*>(w), wdt, dev, {kH});
    Tensor tr = Tensor::Contiguous(res.data(), DType::kF32, dev, {kT, kH});
    Tensor tf = Tensor::Contiguous(out.fp8.data(), DType::kI8, dev, {kT, kH});
    Tensor tb = Tensor::Contiguous(out.bf16.data(), DType::kBF16, dev, {kT, kH});
    vt::RmsNormQuantFp8(q, tf, &tb, tx, tw, args, &tr, kScale);
    return out;
  }
  vt::Backend& b = vt::GetBackend(dev.type);
  Queue q = b.CreateQueue();
  void* xd = b.Alloc(x.size() * sizeof(uint16_t));
  void* wd = b.Alloc(wbytes);
  void* rd = b.Alloc(res.size() * sizeof(float));
  void* fd = b.Alloc(out.fp8.size());
  void* bd = b.Alloc(out.bf16.size() * sizeof(uint16_t));
  b.Copy(q, xd, x.data(), x.size() * sizeof(uint16_t));
  b.Copy(q, wd, w, wbytes);
  b.Copy(q, rd, res.data(), res.size() * sizeof(float));
  Tensor tx = Tensor::Contiguous(xd, DType::kBF16, dev, {kT, kH});
  Tensor tw = Tensor::Contiguous(wd, wdt, dev, {kH});
  Tensor tr = Tensor::Contiguous(rd, DType::kF32, dev, {kT, kH});
  Tensor tf = Tensor::Contiguous(fd, DType::kI8, dev, {kT, kH});
  Tensor tb = Tensor::Contiguous(bd, DType::kBF16, dev, {kT, kH});
  vt::RmsNormQuantFp8(q, tf, &tb, tx, tw, args, &tr, kScale);
  b.Copy(q, out.fp8.data(), fd, out.fp8.size());
  b.Copy(q, out.bf16.data(), bd, out.bf16.size() * sizeof(uint16_t));
  b.Synchronize(q);
  b.Free(xd); b.Free(wd); b.Free(rd); b.Free(fd); b.Free(bd);
  b.DestroyQueue(q);
  return out;
}

}  // namespace

TEST_CASE("rmsnorm_quant_fp8: the gamma dtype does not change the answer (f32/f16/bf16)") {
  const std::vector<uint16_t> x = SpreadXBf16(kT * kH);
  const std::vector<float> w32 = ExactW(kH);
  std::vector<uint16_t> w16(static_cast<size_t>(kH)), wbf(static_cast<size_t>(kH));
  for (int64_t j = 0; j < kH; ++j) {
    const size_t u = static_cast<size_t>(j);
    w16[u] = vt::F32ToF16(w32[u]);
    wbf[u] = vt::F32ToBF16(w32[u]);
    // The premise of the byte-equality bar: all three encodings hold the SAME value.
    REQUIRE(vt::F16ToF32(w16[u]) == w32[u]);
    REQUIRE(vt::BF16ToF32(wbf[u]) == w32[u]);
  }

  // bf16/bf16 is `qwen3_5.cpp:6869`'s own pairing and the anchor for the other two.
  const Out anchor = RunFp8(Cpu(), x, wbf.data(), DType::kBF16, wbf.size() * 2);
  const Out from_f32 = RunFp8(Cpu(), x, w32.data(), DType::kF32, w32.size() * 4);
  const Out from_f16 = RunFp8(Cpu(), x, w16.data(), DType::kF16, w16.size() * 2);

  CHECK(from_f32 == anchor);
  CHECK(from_f16 == anchor);

  // The gamma must be LOAD-BEARING, or the equalities above would also hold for a
  // kernel that ignored `w`. Bump one element far enough that the e4m3 quantization
  // of that column cannot land on the same byte.
  std::vector<float> bumped = w32;
  bumped[7] += 4.0f;
  const Out moved = RunFp8(Cpu(), x, bumped.data(), DType::kF32, bumped.size() * 4);
  size_t changed = 0;
  for (int64_t r = 0; r < kT; ++r)
    if (moved.fp8[static_cast<size_t>(r * kH + 7)] != anchor.fp8[static_cast<size_t>(r * kH + 7)])
      ++changed;
  CHECK(changed == static_cast<size_t>(kT));
}

TEST_CASE("rmsnorm_quant_fp8: CUDA serves every gamma dtype and agrees with CPU") {
  if (!HasCuda()) {
    std::printf("[SKIP] no CUDA backend: rmsnorm_quant_fp8 gamma-dtype arms NOT exercised\n");
    return;
  }
  const Device gpu{DeviceType::kCUDA, 0};
  const std::vector<uint16_t> x = SpreadXBf16(kT * kH);
  const std::vector<float> w32 = ExactW(kH);
  std::vector<uint16_t> w16(static_cast<size_t>(kH)), wbf(static_cast<size_t>(kH));
  for (int64_t j = 0; j < kH; ++j) {
    w16[static_cast<size_t>(j)] = vt::F32ToF16(w32[static_cast<size_t>(j)]);
    wbf[static_cast<size_t>(j)] = vt::F32ToBF16(w32[static_cast<size_t>(j)]);
  }

  // Before this change the two mixed calls THREW
  // "cuda rmsnorm_quant_fp8: weight dtype must match x" out of the dispatcher,
  // rather than disagreeing.
  const Out g_bf16 = RunFp8(gpu, x, wbf.data(), DType::kBF16, wbf.size() * 2);
  const Out g_f32 = RunFp8(gpu, x, w32.data(), DType::kF32, w32.size() * 4);
  const Out g_f16 = RunFp8(gpu, x, w16.data(), DType::kF16, w16.size() * 2);

  // Byte equality across the three gamma dtypes on the device, which is what a
  // fall-through to the wrong pointer type cannot produce.
  CHECK(g_f32 == g_bf16);
  CHECK(g_f16 == g_bf16);
  // And against the host, which is the divergence this change closes.
  CHECK(g_bf16 == RunFp8(Cpu(), x, wbf.data(), DType::kBF16, wbf.size() * 2));
  CHECK(g_f32 == RunFp8(Cpu(), x, w32.data(), DType::kF32, w32.size() * 4));
  CHECK(g_f16 == RunFp8(Cpu(), x, w16.data(), DType::kF16, w16.size() * 2));
}
