// The gamma's dtype is INDEPENDENT of the activation's (#2477).
//
// vllm.cpp original (vt runtime); the behaviour it pins is upstream's.
// `GemmaRMSNorm` upcasts the gamma on its own —
// `vllm/models/qwen4_exp/nvidia/ple_layer.py:80` at vLLM origin/main 25efcfa788
// reads `normalized * (1.0 + self.weight.float())` — while the activation stays
// at `model_config.dtype`, which `vllm/models/qwen4_exp/nvidia/qsa.py:188-189`
// forces to bf16 for this architecture. So bf16 activation against an f32 gamma
// is upstream's own pairing, not an accident.
//
// It is also the ONLY pairing the released `unsloth/Qwen3.8-Flash-Next-GGUF`
// UD-IQ1_S artifact can present: GGUF stores every norm gamma as F32
// (`blk.N.attn_q_norm.weight` [256] F32, `blk.N.indexer.q_norm.weight` [128]
// F32, measured from the shards' tensor tables), and
// `qwen4_exp_qsa_block.cpp:446` refuses a non-bf16 hidden state.
//
// RED BEFORE GREEN: on the CUDA arm, before `RmsNormRowKernel` took its own
// `Tw`, this case did not compare wrong values — it THREW
// "cuda rmsnorm: weight dtype must match x" out of `RmsNormKernelCuda`, which
// is the refusal that stopped a `--device cuda` forward of qwen4_exp at
// `src/vt/cuda/cuda_ops.cu:463`.
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
  // Round-to-nearest-even, the same rounding the kernels' stores use.
  const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}
float Bf16ToF32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// The production shape: one head_dim row of the QSA indexer q-norm.
constexpr int64_t kRows = 5;   // the CPU control's prompt_tokens
constexpr int64_t kH = 128;    // blk.N.indexer.q_norm.weight is [128]

struct Inputs {
  std::vector<uint16_t> x;  // bf16 activation
  std::vector<float> w;     // f32 gamma, as GGUF stores it
};

Inputs MakeInputs() {
  Inputs in;
  in.x.resize(static_cast<size_t>(kRows * kH));
  in.w.resize(static_cast<size_t>(kH));
  for (int64_t i = 0; i < kRows * kH; ++i) {
    // Spread over a decade so the reduction is not dominated by one term.
    const float v = std::sin(static_cast<float>(i) * 0.37f) * (1.0f + 0.01f * static_cast<float>(i % 17));
    in.x[static_cast<size_t>(i)] = F32ToBf16(v);
  }
  for (int64_t j = 0; j < kH; ++j) {
    // gemma polarity: the stored gamma is the RAW HuggingFace value, near zero.
    in.w[static_cast<size_t>(j)] = 0.05f * std::cos(static_cast<float>(j) * 0.11f);
  }
  return in;
}

// bf16 out, bf16 x, f32 w — exactly what qwen4_exp_qsa_block.cpp:632 hands the op.
std::vector<uint16_t> RunMixed(Device dev, const Inputs& in) {
  std::vector<uint16_t> out(static_cast<size_t>(kRows * kH), 0);
  const RmsNormArgs args{1e-6f, /*gemma=*/true};
  if (dev.type == DeviceType::kCPU) {
    Queue q{dev, nullptr};
    Tensor tx = Tensor::Contiguous(const_cast<uint16_t*>(in.x.data()), DType::kBF16, dev,
                                   {kRows, kH});
    Tensor tw = Tensor::Contiguous(const_cast<float*>(in.w.data()), DType::kF32, dev, {kH});
    Tensor to = Tensor::Contiguous(out.data(), DType::kBF16, dev, {kRows, kH});
    vt::RmsNorm(q, to, tx, tw, args);
    return out;
  }
  vt::Backend& backend = vt::GetBackend(dev.type);
  Queue q = backend.CreateQueue();
  void* xd = backend.Alloc(in.x.size() * sizeof(uint16_t));
  void* wd = backend.Alloc(in.w.size() * sizeof(float));
  void* od = backend.Alloc(out.size() * sizeof(uint16_t));
  backend.Copy(q, xd, in.x.data(), in.x.size() * sizeof(uint16_t));
  backend.Copy(q, wd, in.w.data(), in.w.size() * sizeof(float));
  Tensor tx = Tensor::Contiguous(xd, DType::kBF16, dev, {kRows, kH});
  Tensor tw = Tensor::Contiguous(wd, DType::kF32, dev, {kH});
  Tensor to = Tensor::Contiguous(od, DType::kBF16, dev, {kRows, kH});
  vt::RmsNorm(q, to, tx, tw, args);
  backend.Copy(q, out.data(), od, out.size() * sizeof(uint16_t));
  backend.Synchronize(q);
  backend.Free(xd);
  backend.Free(wd);
  backend.Free(od);
  backend.DestroyQueue(q);
  return out;
}

}  // namespace

TEST_CASE("rmsnorm: an f32 gamma against a bf16 activation is upstream's own pairing") {
  const Inputs in = MakeInputs();
  const std::vector<uint16_t> got = RunMixed(Cpu(), in);

  // The independent reference: widen both operands to f32, as GemmaRMSNorm does,
  // and round once at the store. Recomputed here rather than taken from the op,
  // because a gate that compares the op against itself proves consistency and
  // not correctness.
  size_t mismatches = 0;
  for (int64_t r = 0; r < kRows; ++r) {
    double sumsq = 0.0;
    for (int64_t j = 0; j < kH; ++j) {
      const float v = Bf16ToF32(in.x[static_cast<size_t>(r * kH + j)]);
      sumsq += static_cast<double>(v) * static_cast<double>(v);
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(sumsq / static_cast<double>(kH)) + 1e-6f);
    for (int64_t j = 0; j < kH; ++j) {
      const float v = Bf16ToF32(in.x[static_cast<size_t>(r * kH + j)]);
      const float want = v * inv * (1.0f + in.w[static_cast<size_t>(j)]);
      const float have = Bf16ToF32(got[static_cast<size_t>(r * kH + j)]);
      // One bf16 ulp of headroom: the op rounds at the store, this reference
      // rounds at the compare, and the reduction order may differ.
      if (std::fabs(have - want) > 0.01f * (std::fabs(want) + 1e-3f)) ++mismatches;
    }
  }
  CHECK(mismatches == 0);

  // The gamma must actually be LOAD-BEARING, or the case above is vacuous: a
  // kernel that ignored `w` entirely would pass a tolerance this loose on a
  // near-zero gamma. Perturb one column and require that column to move.
  Inputs bumped = in;
  bumped.w[7] += 4.0f;
  const std::vector<uint16_t> other = RunMixed(Cpu(), bumped);
  CHECK(got[7] != other[7]);
}

TEST_CASE("rmsnorm: the CUDA provider accepts the f32-gamma/bf16-activation pairing") {
  if (!HasCuda()) {
    // Not a gate on a CPU-only lane, and this suite's convention is to say so
    // rather than to score an unexercised pass silently
    // (tests/vt/test_ops_conv3d.cpp carries the same note). The device evidence
    // for this case was taken on thor:gpu0, sm_110.
    std::printf("[SKIP] no CUDA backend: rmsnorm mixed-dtype device arm NOT exercised\n");
    return;
  }
  const Inputs in = MakeInputs();
  const std::vector<uint16_t> want = RunMixed(Cpu(), in);
  // Before the fix this line THREW rather than disagreeing.
  const std::vector<uint16_t> got = RunMixed(Device{DeviceType::kCUDA, 0}, in);
  REQUIRE(got.size() == want.size());
  size_t bad = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (std::fabs(Bf16ToF32(got[i]) - Bf16ToF32(want[i])) >
        0.01f * (std::fabs(Bf16ToF32(want[i])) + 1e-3f))
      ++bad;
  }
  CHECK(bad == 0);
}
