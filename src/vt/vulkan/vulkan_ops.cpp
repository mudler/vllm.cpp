// Vulkan backend — op kernels: descriptor binding + dispatch of the committed
// SPIR-V in vulkan_spirv.h, plus the `RegisterOp` table entries. BACKEND-VULKAN,
// W0 skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp`
// Registrar idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS (deliberately a SEAM PROOF, not a model):
//   kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm, kRmsNorm and the
//   single kFusedChain registration that inherits the portable fusion catalog.
// That set spans every structural class the seam has to get right: flat
// elementwise, a rank-1 broadcast, a dtype-converting copy, TWO different row
// reductions, an optional in-place residual stream, and the recipe interpreter.
// It matches the Metal skeleton's set exactly, so the two backends are directly
// comparable through tests/vt/test_backend_cross_device.cpp.
//
// WHAT IS STILL STUBBED: everything else. `kMatmul`/`kMatmulBT`,
// `kPagedAttention`, `kReshapeAndCache`, the whole quant tier and the sampler
// ops are NOT registered, so `vt::GetOp` throws its normal "no kernel for op N
// on device type 3" for them (src/vt/ops.cpp:104-111 — a partial backend is a
// supported, tested state). NO MODEL RUNS ON THIS BACKEND.
//
// BINDING MODEL: every tensor operand occupies TWO consecutive descriptor
// bindings onto the SAME VkBuffer — a uint32_t view and a uint16_t view — and
// its BYTE OFFSET travels in the push constants. See
// src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL for why.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/ops.h"

namespace vt::vulkan {
namespace {

// Storage dtype -> the shader-side code (vt_common.glsl VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "vulkan: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// Collects the (buffer, byte-offset) pairs for a dispatch. Each Add() appends
// the SAME buffer twice — bindings 2k and 2k+1, the u32 and u16 views — and
// returns the byte offset for the push-constant block.
class Binder {
 public:
  uint32_t Add(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    buffers_.push_back(r.buffer);
    // f32 access indexes a uint32_t[] view, so a f32 operand's byte offset must
    // be 4-byte aligned; 16-bit access only needs 2. Tensor storage always
    // satisfies this (allocations are 64-byte aligned and views advance by whole
    // elements), but a violation would silently read shifted data.
    VT_CHECK(t.dtype != DType::kF32 || r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }
  // A raw buffer bound ONCE (no 16-bit view): the fused-chain step list.
  void AddRaw(void* buffer) { buffers_.push_back(buffer); }

  const void* const* data() const { return buffers_.data(); }
  uint32_t count() const { return static_cast<uint32_t>(buffers_.size()); }

 private:
  std::vector<const void*> buffers_;
};

// ---- Host mirrors of the shaders' push-constant blocks. Field order and types
// must match the GLSL declarations EXACTLY. GLSL `uint`/`float` are 4-byte with
// 4-byte alignment and every block below is a run of 4-byte scalars, so the std430
// push-constant layout coincides with the C++ layout with no padding surprises.
struct AddParams {
  uint32_t n, d, a_dt, b_dt, out_dt, bcast, a_off, b_off, out_off;
};
struct UnaryParams {
  uint32_t n, a_dt, out_dt, a_off, out_off;
};
struct SiluMulParams {
  uint32_t t, d, x_dt, out_dt, x_off, out_off;
};
struct RmsParams {
  uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, x_off, w_off, out_off, res_off;
  float eps;
};
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, x_off, w_off, b_off, out_off;
  float eps;
};
struct FcParams {
  uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, x_off, w_off, res_off, out_off;
  float eps;
};

// Vulkan only GUARANTEES 128 bytes of push-constant space (maxPushConstantsSize);
// staying inside it is what keeps this backend portable without a probe.
static_assert(sizeof(RmsParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(LayerNormParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(FcParams) <= 128, "push constants must fit the guaranteed 128 bytes");

template <typename P>
void Go(const char* name, const Binder& b, const P& p, uint32_t groups) {
  VulkanContext::Get().Dispatch(name, b.data(), b.count(), &p, sizeof(P), groups);
}

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "add: a");
  const uint32_t b_off = bind.Add(b, "add: b");
  const uint32_t out_off = bind.Add(out, "add: out");
  AddParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d),
              DtypeCode(a.dtype),      DtypeCode(b.dtype),
              DtypeCode(out.dtype),    bcast ? 1u : 0u,
              a_off,                   b_off,
              out_off};
  Go("vt_add", bind, p, FlatGroupCount(n));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  Binder bind;
  const uint32_t x_off = bind.Add(x, "relu: x");
  const uint32_t out_off = bind.Add(out, "relu: out");
  UnaryParams p{static_cast<uint32_t>(n), DtypeCode(x.dtype), DtypeCode(out.dtype), x_off,
                out_off};
  Go("vt_relu", bind, p, FlatGroupCount(n));
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  Binder bind;
  const uint32_t in_off = bind.Add(in, "cast: in");
  const uint32_t out_off = bind.Add(out, "cast: out");
  UnaryParams p{static_cast<uint32_t>(n), DtypeCode(in.dtype), DtypeCode(out.dtype), in_off,
                out_off};
  Go("vt_cast", bind, p, FlatGroupCount(n));
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "silu_and_mul: x");
  const uint32_t out_off = bind.Add(out, "silu_and_mul: out");
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype), x_off, out_off};
  Go("vt_silu_and_mul", bind, p, FlatGroupCount(t * d));
}

// cpu_ops.cpp:225-250 RmsNormKernel. One workgroup per token row.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm: x");
  const uint32_t w_off = bind.Add(w, "rmsnorm: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm: out");
  // Bindings 6/7 are always written: a descriptor a shader statically uses must
  // be valid even on the code path that never reads it. With has_res == 0 they
  // alias `out` and are dead.
  const uint32_t res_off =
      residual != nullptr ? bind.Add(*residual, "rmsnorm: residual") : bind.Add(out, "rmsnorm: out");
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              x_off,
              w_off,
              out_off,
              res_off,
              args.eps};
  Go("vt_rms_norm", bind, p, static_cast<uint32_t>(t));
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "layer_norm: x");
  const uint32_t w_off =
      weight != nullptr ? bind.Add(*weight, "layer_norm: weight") : bind.Add(x, "layer_norm: x");
  const uint32_t b_off =
      bias != nullptr ? bind.Add(*bias, "layer_norm: bias") : bind.Add(x, "layer_norm: x");
  const uint32_t out_off = bind.Add(out, "layer_norm: out");
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    x_off,
                    w_off,
                    b_off,
                    out_off,
                    args.eps};
  Go("vt_layer_norm", bind, p, static_cast<uint32_t>(rows));
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "vulkan fused_chain: bad step count");

  // Words per step, matching VT_STEP_WORDS in vt_fused_chain.comp.
  constexpr uint32_t kStepWords = 5;
  std::vector<uint32_t> steps(static_cast<size_t>(r.n) * kStepWords, 0u);
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    uint32_t op = 0;
    switch (st.op) {
      case FOp::kAdd: op = 0; break;
      case FOp::kMul: op = 1; break;
      case FOp::kSilu: op = 2; break;
      case FOp::kSigmoid: op = 3; break;
      case FOp::kRmsNorm:
        // Mirrors the CPU interpreter's assertion (cpu_ops.cpp:1674): the shader
        // hard-codes the mean-square reduction, so any other kind must not reach it.
        VT_CHECK(st.reduce == FReduce::kMeanSquare,
                 "vulkan fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "vulkan fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "vulkan fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "vulkan fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "vulkan fused_chain: recipe touches the residual slot but none was bound");
    const size_t base = static_cast<size_t>(s) * kStepWords;
    steps[base + 0] = op;
    steps[base + 1] = st.out;
    steps[base + 2] = st.in[0];
    steps[base + 3] = st.in[1];
    steps[base + 4] = st.gemma ? 1u : 0u;
  }

  VulkanContext& ctx = VulkanContext::Get();
  const size_t step_bytes = steps.size() * sizeof(uint32_t);
  VT_CHECK(step_bytes <= VulkanContext::kScratchBytes,
           "vulkan fused_chain: step list exceeds the scratch buffer");
  std::memcpy(ctx.ScratchData(), steps.data(), step_bytes);

  Binder bind;
  const uint32_t x_off = bind.Add(x, "fused_chain: x");
  const uint32_t w_off = bind.Add(weight, "fused_chain: weight");
  const uint32_t res_off = residual != nullptr ? bind.Add(*residual, "fused_chain: residual")
                                               : bind.Add(out, "fused_chain: out");
  const uint32_t out_off = bind.Add(out, "fused_chain: out");
  bind.AddRaw(ctx.ScratchBuffer());
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             x_off,
             w_off,
             res_off,
             out_off,
             eps};
  Go("vt_fused_chain", bind, p, static_cast<uint32_t>(t));
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Vulkan-enabled build on a host with
    // no loader or no conformant device registers nothing, so GetOp throws its
    // normal not-registered error.
    if (!VulkanContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kAdd, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::vulkan
