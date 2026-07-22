// Metal backend — op kernels: encode/dispatch of the runtime-compiled MSL in
// metal_msl.h, plus the `RegisterOp` table entries. BACKEND-METAL-MLX, W0
// skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp` Registrar
// idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS (deliberately a SEAM PROOF, not a model):
//   kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm, kRmsNorm, the
//   dense GEMM pair kMatmul/kMatmulBT, and the single kFusedChain registration
//   that inherits the portable fusion catalog.
// That set spans every structural class the seam has to get right: flat
// elementwise, a rank-1 broadcast, a dtype-converting copy, TWO different row
// reductions, an optional in-place residual stream, and the recipe interpreter.
//
// WHAT IS STILL STUBBED: everything else. `kPagedAttention`,
// `kReshapeAndCache`, the whole quant tier and the sampler ops are NOT
// registered, so `vt::GetOp` throws its normal "no kernel for op N on device
// type 2" for them (a partial backend is a supported, tested state). No model
// runs on this backend. NOTE `kPagedAttention` stays OURS regardless of MLX:
// MLX has no paged-KV attention primitive at all
// (.agents/specs/metal-mlx-reuse-study.md §5.3).
//
// DISPATCH MODEL: one command buffer per op, committed and waited. See
// metal_backend.mm § SCOPE for why, and what it costs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>
#include <vector>

#include "metal_buffers.h"
#include "metal_context.h"
#include "vt/ops.h"

namespace vt::metal {
namespace {

// Storage dtype -> the shader-side code (metal_msl.h VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "metal: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// ---- Host mirrors of the metal_msl.h parameter structs. Field order and types
// must match the MSL declarations EXACTLY; MSL scalar `uint`/`float` are 4-byte
// with 4-byte alignment, same as here, and every struct is a multiple of 4 with
// no member exceeding 4 bytes, so the layouts coincide without padding surprises.
struct ElemParams { uint32_t n, d, a_dt, b_dt, out_dt, bcast; };
struct SiluMulParams { uint32_t t, d, x_dt, out_dt; };
struct RmsParams { uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, tg; float eps; };
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, tg;
  float eps;
};
struct GemmParams { uint32_t m, n, k, lda, a_dt, b_dt, out_dt, bt; };
struct FStepGpu { uint32_t op, out, in0, in1, gemma, pad; };
struct FcParams { uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, tg; float eps; };

static_assert(sizeof(ElemParams) == 24, "ElemParams layout must match the MSL struct");
static_assert(sizeof(FStepGpu) == 24, "FStepGpu layout must match the MSL struct");

// A small RAII-ish encode helper: opens a command buffer + compute encoder for
// `fn_name`, lets the caller bind, then dispatches and BLOCKS.
class Encoder {
 public:
  explicit Encoder(const char* fn_name) {
    pool_ = [[NSAutoreleasePool alloc] init];
    auto& ctx = MetalContext::Get();
    id<MTLCommandQueue> q = static_cast<id<MTLCommandQueue>>(ctx.command_queue());
    cmd_ = [q commandBuffer];
    enc_ = [cmd_ computeCommandEncoder];
    auto pso = static_cast<id<MTLComputePipelineState>>(ctx.Pipeline(std::string(fn_name)));
    [enc_ setComputePipelineState:pso];
    max_threads_ = static_cast<size_t>([pso maxTotalThreadsPerThreadgroup]);
  }
  // A bind can THROW (Resolve rejects memory this backend did not allocate),
  // unwinding past the dispatch. Metal asserts and aborts the process if a
  // command encoder is released without endEncoding, which would turn a clean,
  // catchable vt:: error into SIGABRT — so close the encoder here whenever the
  // happy path did not. `finished_` makes this idempotent.
  ~Encoder() {
    if (!finished_ && enc_ != nil) {
      [enc_ endEncoding];
      finished_ = true;
    }
    [pool_ release];
  }

  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;

  // Bind a vt::Tensor's storage, resolving the (possibly interior) pointer to
  // the owning MTLBuffer + offset.
  void BindTensor(const Tensor& t, int index, const char* what) {
    Resolved r = Resolve(t.data, what);
    [enc_ setBuffer:static_cast<id<MTLBuffer>>(r.buffer) offset:r.offset atIndex:index];
  }
  // Bind small host-side data by value (params structs, the recipe step list).
  // setBytes is the documented path for <4 KiB of per-dispatch constants.
  void BindBytes(const void* data, size_t bytes, int index) {
    VT_CHECK(bytes <= 4096, "metal: setBytes payload must stay under 4 KiB");
    [enc_ setBytes:data length:bytes atIndex:index];
  }

  // Flat dispatch: `n` threads, non-uniform threadgroups (Apple GPUs support
  // dispatchThreads:, so no manual grid rounding is needed).
  void DispatchFlat(int64_t n) {
    if (n <= 0) { Finish(); return; }
    // Clamp to THIS pipeline's limit, which the compiler may set below the
    // device max; dispatching past it is a hard Metal error.
    const NSUInteger tg = ChooseThreadgroupSize(n, max_threads_);
    [enc_ dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    Finish();
  }
  // Row dispatch: one THREADGROUP per row, `tg` threads each (llama.cpp
  // kernel_rms_norm shape).
  void DispatchRows(int64_t rows, uint32_t tg) {
    if (rows <= 0) { Finish(); return; }
    [enc_ dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    Finish();
  }
  // 2-D tile dispatch: one threadgroup per (tile x tile) output tile, which is
  // the GEMM shape. dispatchThreadgroups (not dispatchThreads) because the tile
  // loop needs FULL threadgroups — the kernel bounds-checks its own edges.
  void DispatchTiles(int64_t cols, int64_t rows, uint32_t tile) {
    if (cols <= 0 || rows <= 0) { Finish(); return; }
    const NSUInteger gx = static_cast<NSUInteger>((cols + tile - 1) / tile);
    const NSUInteger gy = static_cast<NSUInteger>((rows + tile - 1) / tile);
    [enc_ dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
         threadsPerThreadgroup:MTLSizeMake(tile, tile, 1)];
    Finish();
  }

 private:
  void Finish() {
    [enc_ endEncoding];
    finished_ = true;
    [cmd_ commit];
    [cmd_ waitUntilCompleted];
    VT_CHECK([cmd_ error] == nil,
             std::string("metal: command buffer failed: ") +
                 [[[cmd_ error] localizedDescription] UTF8String]);
  }
  NSAutoreleasePool* pool_ = nil;
  id<MTLCommandBuffer> cmd_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  size_t max_threads_ = 0;
  bool finished_ = false;
};

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  ElemParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d), DtypeCode(a.dtype),
               DtypeCode(b.dtype), DtypeCode(out.dtype), bcast ? 1u : 0u};
  Encoder e("vt_add");
  e.BindTensor(a, 0, "add: a");
  e.BindTensor(b, 1, "add: b");
  e.BindTensor(out, 2, "add: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchFlat(n);
}

// Dense GEMM, the NATIVE MSL provider for kMatmul / kMatmulBT (metal_msl.h
// vt_matmul). Registered at the default priority under vt::kNativeProviderName,
// so it stays the DEFAULT on Metal and the optional MLX provider
// (metal_mlx_provider.mm, VLLM_CPP_MLX) only displaces it when explicitly built
// in — and can be switched back off in the same binary with
// VT_OP_PROVIDER_DISABLE=mlx.
constexpr uint32_t kGemmTile = 16;

void MatmulKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[1];
  GemmParams p{static_cast<uint32_t>(m),       static_cast<uint32_t>(n),
               static_cast<uint32_t>(k),       static_cast<uint32_t>(a.stride[0]),
               DtypeCode(a.dtype),             DtypeCode(b.dtype),
               DtypeCode(out.dtype),           0u};
  Encoder e("vt_matmul");
  e.BindTensor(a, 0, "matmul: a");
  e.BindTensor(b, 1, "matmul: b");
  e.BindTensor(out, 2, "matmul: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchTiles(n, m, kGemmTile);
}

void MatmulBTKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  GemmParams p{static_cast<uint32_t>(m),       static_cast<uint32_t>(n),
               static_cast<uint32_t>(k),       static_cast<uint32_t>(a.stride[0]),
               DtypeCode(a.dtype),             DtypeCode(b.dtype),
               DtypeCode(out.dtype),           1u};
  Encoder e("vt_matmul");
  e.BindTensor(a, 0, "matmul_bt: a");
  e.BindTensor(b, 1, "matmul_bt: b");
  e.BindTensor(out, 2, "matmul_bt: out");
  e.BindBytes(&p, sizeof(p), 3);
  e.DispatchTiles(n, m, kGemmTile);
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  ElemParams p{static_cast<uint32_t>(n), 1u, DtypeCode(x.dtype), 0u, DtypeCode(out.dtype), 0u};
  Encoder e("vt_relu");
  e.BindTensor(x, 0, "relu: x");
  e.BindTensor(out, 1, "relu: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(n);
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  ElemParams p{static_cast<uint32_t>(n), 1u, DtypeCode(in.dtype), 0u, DtypeCode(out.dtype), 0u};
  Encoder e("vt_cast");
  e.BindTensor(in, 0, "cast: in");
  e.BindTensor(out, 1, "cast: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(n);
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype)};
  Encoder e("vt_silu_and_mul");
  e.BindTensor(x, 0, "silu_and_mul: x");
  e.BindTensor(out, 1, "silu_and_mul: out");
  e.BindBytes(&p, sizeof(p), 2);
  e.DispatchFlat(t * d);
}

// cpu_ops.cpp:225-250 RmsNormKernel.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  // Pipeline-specific cap: vt_rms_norm declares threadgroup scratch, so its
  // maxTotalThreadsPerThreadgroup can be LOWER than the device limit and
  // over-dispatching is a hard Metal error. Resolved BEFORE the Encoder is built
  // because `tg` also travels in the params struct — the shader's stride loop and
  // the host dispatch must agree exactly.
  const uint32_t tg =
      ChooseThreadgroupSize(h, MetalContext::Get().PipelineMaxThreads("vt_rms_norm"));
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              tg,
              args.eps};
  Encoder e("vt_rms_norm");
  e.BindTensor(x, 0, "rmsnorm: x");
  e.BindTensor(w, 1, "rmsnorm: weight");
  e.BindTensor(out, 2, "rmsnorm: out");
  // Buffer 3 is always bound: an unbound buffer is undefined even when the
  // shader never reads it. With has_res == 0 it aliases `out` and is dead.
  e.BindTensor(residual != nullptr ? *residual : out, 3, "rmsnorm: residual");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchRows(t, tg);
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  const uint32_t tg =
      ChooseThreadgroupSize(d, MetalContext::Get().PipelineMaxThreads("vt_layer_norm"));
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    tg,
                    args.eps};
  Encoder e("vt_layer_norm");
  e.BindTensor(x, 0, "layer_norm: x");
  e.BindTensor(weight != nullptr ? *weight : x, 1, "layer_norm: weight");
  e.BindTensor(bias != nullptr ? *bias : x, 2, "layer_norm: bias");
  e.BindTensor(out, 3, "layer_norm: out");
  e.BindBytes(&p, sizeof(p), 4);
  e.DispatchRows(rows, tg);
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "metal fused_chain: bad step count");

  std::vector<FStepGpu> steps(static_cast<size_t>(r.n));
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
                 "metal fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "metal fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "metal fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "metal fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "metal fused_chain: recipe touches the residual slot but none was bound");
    steps[static_cast<size_t>(s)] =
        FStepGpu{op, st.out, st.in[0], st.in[1], st.gemma ? 1u : 0u, 0u};
  }

  const uint32_t tg =
      ChooseThreadgroupSize(h, MetalContext::Get().PipelineMaxThreads("vt_fused_chain"));
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             tg,
             eps};
  Encoder e("vt_fused_chain");
  e.BindTensor(x, 0, "fused_chain: x");
  e.BindTensor(weight, 1, "fused_chain: weight");
  e.BindTensor(residual != nullptr ? *residual : out, 2, "fused_chain: residual");
  e.BindTensor(out, 3, "fused_chain: out");
  e.BindBytes(steps.data(), steps.size() * sizeof(FStepGpu), 4);
  e.BindBytes(&p, sizeof(p), 5);
  e.DispatchRows(t, tg);
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Metal-enabled build on a device-less
    // host registers nothing, so GetOp throws its normal not-registered error.
    if (!MetalContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kMatmul, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulKernel)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTKernel)));
    RegisterOp(OpId::kAdd, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kMETAL,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
  }
} registrar;

}  // namespace
}  // namespace vt::metal
