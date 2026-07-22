// Metal backend — the MLX acceleration PROVIDER for the dense GEMM pair
// (BACKEND-METAL-MLX / BACKEND-ACCEL-PROVIDER, .agents/specs/metal-mlx-reuse-study.md
// §5 + §6 work row M5). Compiled ONLY when -DVLLM_CPP_MLX=ON; the native MSL
// GEMM in metal_ops.mm is the default and is what ships.
//
// WHY THIS FILE EXISTS AT ALL. It is the seam's proof obligation: the study's
// §6.1 table claims ONE mechanism serves MLX-on-Metal (a C++ OBJECT-MODEL
// library), cuBLASLt/CUTLASS/flashinfer-on-CUDA (raw C launchers), llama.cpp on
// CPU/Vulkan, and our own arch tactics. Three of those four shapes were already
// expressible; MLX was the one that was not, because
// `.agents/specs/dropin-kernel-abi.md` binds raw pointer/shape/stride/stream
// contracts and MLX's entry points take `const array&` / `Stream`. This file
// registers an object-model library through the same `OpProvider` struct a raw
// launcher uses, which is what makes the claim true rather than asserted.
//
// THE LAZY-EVAL OBJECTION IS REFUTED, and the refutation is what this code
// relies on (.agents/backends.md:84-90 was wrong; study §5.1): MLX's graph
// terminates at `mlx/backend/metal/eval.cpp:32-48`, which calls
// `primitive().eval_gpu()` — an EAGER per-op encode into the stream's command
// buffer. Building one `matmul` node and evaluating it immediately is therefore
// a single kernel launch, not a graph: no tape survives the call, no autodiff,
// no deferred anything. The explicit `eval()` below IS the boundary.
//
// ZERO-COPY, HONESTLY SCOPED (study §5.2, and where we deviate from it):
//   * INPUTS are zero-copy. `mlx::core::allocator::Buffer` is a bare `void*`
//     wrapper (`mlx/allocator.h:12-29`) and on Metal that `void*` IS the
//     `MTL::Buffer*`; `array::set_data(Buffer, size, strides, flags, Deleter)`
//     (`mlx/array.h:417-422`) takes a caller-supplied deleter, so passing a
//     no-op means MLX never owns or frees our memory. metal-cpp's `MTL::Buffer*`
//     and ObjC's `id<MTLBuffer>` are the same pointer, so the bridge is a cast.
//   * OUTPUT is NOT zero-copy, and the study's sketch was optimistic here. The
//     study proposed calling `steel_matmul` directly with a pre-bound output.
//     MEASURED against the shipped wheel: `steel_matmul_axpby<false>` is NOT an
//     exported symbol of `libmlx.dylib` (only the `get_steel_*_kernel` helpers
//     are), so that entry point is not linkable against the prebuilt artifact,
//     and `Matmul::eval_gpu` re-`set_data`s its output from MLX's allocator
//     anyway. We therefore call the exported `mlx::core::matmul`, let MLX
//     allocate the output, and `memcpy` it into ours. On unified memory that is
//     a host memcpy of M*N elements against an O(M*N*K) GEMM. It is a real cost,
//     it is recorded, and it is not described as zero-copy.
//
// WHAT IS NOT DELEGATED, ON PURPOSE. `kPagedAttention` and `kReshapeAndCache`
// stay OURS unconditionally: MLX has no paged-KV primitive anywhere — its
// `ScaledDotProductAttention::use_fallback` gates on contiguous q/k/v and head
// dim, and there is no block-table concept in the file (study §5.3). MLX wins on
// GEMM, not on our KV cache.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <vector>

// MLX public headers. Deliberately NOT mlx/backend/metal/*: those pull in
// metal-cpp and, as noted above, their entry points are not exported anyway.
#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

#include "metal_buffers.h"
#include "metal_context.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace vt::metal {
namespace {

namespace mx = mlx::core;

// The provider identity. It is the key for VT_OP_PROVIDER_DISABLE=mlx (the
// same-binary A/B the benchmark protocol requires) and the string
// GetOpProviderStats() reports, which is how a run PROVES which GEMM executed
// rather than inferring it from a green assertion.
constexpr const char* kMlxProvider = "mlx";
// Above the native MSL GEMM's implicit 0. A single number, not a policy: the
// build only compiles this TU when the user asked for MLX.
constexpr int kMlxPriority = 100;

bool MlxDtype(DType d, mx::Dtype* out) {
  switch (d) {
    case DType::kF32: *out = mx::float32; return true;
    case DType::kF16: *out = mx::float16; return true;
    case DType::kBF16: *out = mx::bfloat16; return true;
    default: return false;
  }
}

// Wraps one of OUR Metal buffers as an MLX array WITHOUT copying. Returns false
// when the tensor's pointer is interior to its allocation: `array::set_data`
// sets `data_ptr` to `buffer.raw_ptr()` (== `contents()`), so a non-zero offset
// is simply not expressible through the public API, and guessing would silently
// read the wrong rows. A false return is a DECLINE, not an error.
bool WrapTensor(const Tensor& t, const std::vector<int64_t>& strides, mx::array* out) {
  mx::Dtype dt = mx::float32;
  if (!MlxDtype(t.dtype, &dt)) return false;
  Resolved r = Resolve(t.data, "mlx provider: tensor");
  if (r.offset != 0) return false;

  mx::Shape shape;
  for (int i = 0; i < t.rank; ++i) shape.push_back(static_cast<int32_t>(t.shape[i]));
  mx::Strides st;
  for (int64_t s : strides) st.push_back(s);

  mx::array a(shape, dt, nullptr, {});
  mx::array::Flags flags{};
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = false;

  size_t elems = 1;
  for (int i = 0; i < t.rank; ++i) elems *= static_cast<size_t>(t.shape[i]);
  // metal-cpp `MTL::Buffer*` and ObjC `id<MTLBuffer>` are the same pointer; MLX
  // stores it opaquely as `void*` (allocator.h:12-29) and only ever hands it
  // back to Metal. The no-op deleter is what keeps ownership ours.
  a.set_data(mx::allocator::Buffer(r.buffer), elems, st, flags,
             [](mx::allocator::Buffer) {});
  // A fresh `array` starts `unscheduled` (array.h:363-377) and `set_data` does
  // not change that, so MLX's evaluator would reach a leaf with no primitive and
  // throw "Attempting to eval an array without a primitive". Our buffer is
  // already filled and our own queue is idle, so the array IS safe to read —
  // which is exactly what `available` means.
  a.set_status(mx::array::Status::available);
  *out = a;
  return true;
}

// The one delegated computation. `bt` selects the weight orientation exactly as
// the native MSL kernel does: false => b is [K,N], true => b is [N,K] (the torch
// Linear / kMatmulBT orientation), expressed to MLX as a transposed VIEW so its
// Matmul primitive picks the transposed steel kernel rather than materializing a
// transpose.
bool TryMlxMatmul(Tensor& out, const Tensor& a, const Tensor& b, bool bt) {
  // --- per-call DECLINE conditions. Each one falls back to our MSL kernel via
  // GetOpFallback below; none of them is an error.
  if (a.dtype != b.dtype) return false;                 // MLX matmul needs one dtype
  if (a.stride[1] != 1 || b.stride[1] != 1) return false;
  if (a.stride[0] != a.shape[1]) return false;          // row-strided activation
  if (!b.IsContiguous() || !out.IsContiguous()) return false;

  mx::Dtype out_dt = mx::float32;
  if (!MlxDtype(out.dtype, &out_dt)) return false;

  mx::array ma({}, mx::float32, nullptr, {});
  mx::array mb({}, mx::float32, nullptr, {});
  if (!WrapTensor(a, {a.shape[1], 1}, &ma)) return false;
  if (!WrapTensor(b, {b.shape[1], 1}, &mb)) return false;

  const mx::Stream s = mx::default_stream(mx::Device::gpu);
  mx::array mc = mx::matmul(ma, bt ? mx::transpose(mb, {1, 0}, s) : mb, s);
  if (mc.dtype() != out_dt) mc = mx::astype(mc, out_dt, s);
  // THE eval BOUNDARY. Everything above built at most three nodes; this is where
  // MLX encodes and runs them, and it returns with the result materialized.
  mx::eval(mc);

  // Unified memory: MLX's output buffer and our destination are both host
  // addressable, and our own Metal queue is idle (metal_ops.mm commits and waits
  // per op), so a plain memcpy is correct. See the header note — this is the
  // part that is NOT zero-copy.
  std::memcpy(out.data, mc.data<void>(), static_cast<size_t>(mc.nbytes()));
  return true;
}

void MlxMatmulKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  if (TryMlxMatmul(out, a, b, /*bt=*/false)) return;
  // DECLINE-AND-FALL-BACK — the seam's second axis. `GetOp` cannot express this
  // because it never sees a shape; the kernel can, so the kernel asks for the
  // provider immediately below itself and forwards. This is what keeps all ~70
  // op entry points in src/vt/ops.cpp free of any edit.
  reinterpret_cast<MatmulFn>(GetOpFallback(OpId::kMatmul, DeviceType::kMETAL, kMlxProvider))(
      q, out, a, b);
}

void MlxMatmulBTKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  if (TryMlxMatmul(out, a, b, /*bt=*/true)) return;
  reinterpret_cast<MatmulFn>(GetOpFallback(OpId::kMatmulBT, DeviceType::kMETAL, kMlxProvider))(
      q, out, a, b);
}

// Capability predicate. MLX only exists here at all if the build asked for it,
// and it requires a real Metal device — the same guard the native registrar
// uses, so a Metal-enabled build on a device-less host registers nothing and
// GetOp reports its normal not-registered error.
bool MlxSupports(const ProviderCaps&) { return MetalContext::Available(); }

struct Registrar {
  Registrar() {
    if (!MetalContext::Available()) return;
    OpProvider p;
    p.name = kMlxProvider;
    p.priority = kMlxPriority;
    p.supports = &MlxSupports;

    p.fn = reinterpret_cast<void*>(static_cast<MatmulFn>(&MlxMatmulKernel));
    RegisterOpProvider(OpId::kMatmul, DeviceType::kMETAL, p);

    p.fn = reinterpret_cast<void*>(static_cast<MatmulFn>(&MlxMatmulBTKernel));
    RegisterOpProvider(OpId::kMatmulBT, DeviceType::kMETAL, p);
  }
} registrar;

}  // namespace
}  // namespace vt::metal
