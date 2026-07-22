// Metal backend — shared device/queue/pipeline context (BACKEND-METAL-MLX, W0
// skeleton). vllm.cpp original (vt runtime, inventory deviation §9.1): vLLM has
// no Metal platform, so the DESIGN is ported from llama.cpp's Metal backend
// (`ggml/src/ggml-metal/` @ pin 237ad9b96) rather than from vLLM. Specifically:
//
//   * one process-wide `MTLDevice` + one `MTLCommandQueue`, created lazily and
//     kept alive for the process — llama.cpp `ggml-metal-device.m`
//     (`ggml_metal_device_get` / the singleton `g_ctx_dev`, and
//     `ggml_metal_device_init` which creates `device.newCommandQueue`);
//   * the MSL library compiled from EMBEDDED SOURCE at RUN TIME via
//     `-[MTLDevice newLibraryWithSource:options:error:]` — llama.cpp
//     `ggml_metal_library_init`'s `newLibraryWithSource` fallback branch (the
//     one it takes when no prebuilt `default.metallib` is available). We take
//     that branch UNCONDITIONALLY and by design: the gate M4 has Command Line
//     Tools only, so there is no offline `metal` compiler to produce a
//     `.metallib` with (spike § Hardware verdict / Risks 2). This costs one
//     compile at first use and needs ZERO installs;
//   * a NAME -> `MTLComputePipelineState` cache so each kernel is specialized
//     once — llama.cpp `ggml_metal_library_compile_pipeline` + its `pipelines`
//     dictionary.
//
// This header is deliberately PLAIN C++ (no Objective-C types) so the op TUs and
// the tests can include it; the ObjC++ definitions live in metal_context.mm.
#ifndef VT_METAL_METAL_CONTEXT_H_
#define VT_METAL_METAL_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "vt/device.h"

namespace vt::metal {

// Opaque handles into the ObjC world. Kept as void* so every non-.mm consumer
// (metal_ops.mm is ObjC++, but ops.cpp/tests are not) can hold them.
using PipelineHandle = void*;  // id<MTLComputePipelineState>

// Process-wide Metal context. Created on first use, never destroyed (the
// process outlives it; matching llama.cpp's `g_ctx_dev` singleton lifetime).
class MetalContext {
 public:
  // Returns the singleton, creating the device + command queue + compiling the
  // embedded MSL library on first call. Throws (VT_CHECK) if there is no Metal
  // device or if the MSL fails to compile — the compiler diagnostics from
  // `newLibraryWithSource:` are included in the message verbatim, which is the
  // whole reason runtime compilation is tolerable here.
  static MetalContext& Get();

  // True iff a system default Metal device exists. Safe to call on a machine
  // with no GPU: it does NOT throw and does NOT build the library. This is the
  // predicate the backend registrar uses so a Metal-enabled BUILD on a machine
  // with no Metal device simply does not register kMETAL (leaving GetOp/
  // GetBackend to throw their normal "not registered" error) rather than
  // aborting during static initialization.
  static bool Available();

  // Look up (and cache) a compute pipeline for an MSL kernel function name.
  // Throws if the function is absent from the compiled library.
  PipelineHandle Pipeline(const std::string& fn_name);

  // `maxTotalThreadsPerThreadgroup` FOR A SPECIFIC PIPELINE, which can be LOWER
  // than the device-wide limit — the compiler lowers it when a kernel's register
  // or threadgroup-memory demand is high, and our row-reducing kernels declare a
  // threadgroup scratch array. Dispatching more threads than the pipeline allows
  // is a hard Metal error, so every threadgroup-size choice must ask the pipeline,
  // not just the device.
  size_t PipelineMaxThreads(const std::string& fn_name);

  void* device() const { return device_; }        // id<MTLDevice>
  void* command_queue() const { return queue_; }  // id<MTLCommandQueue>

  // Capability data mirrored onto the Platform seam (see
  // src/vllm/platforms/metal.cpp). `family` is the highest MTLGPUFamilyApple<N>
  // the device reports (9 on the M4 gate box), which is what we expose as the
  // DeviceCapability major/minor pair {family, 0} — the Apple-silicon analogue
  // of CUDA's sm_XY.
  int gpu_family_apple() const { return gpu_family_apple_; }
  size_t max_threads_per_threadgroup() const { return max_tg_threads_; }
  size_t threadgroup_memory_bytes() const { return tg_mem_bytes_; }
  bool unified_memory() const { return unified_memory_; }

 private:
  MetalContext();
  void* device_ = nullptr;
  void* queue_ = nullptr;
  void* library_ = nullptr;
  void* pipelines_ = nullptr;  // NSMutableDictionary<NSString*, id<MTLCPS>>
  void* pipelines_lock_ = nullptr;
  int gpu_family_apple_ = 0;
  size_t max_tg_threads_ = 0;
  size_t tg_mem_bytes_ = 0;
  bool unified_memory_ = false;
};

// Plain-C++ spelling of MetalContext::Available(), so the engine-side platform
// TU (src/vllm/platforms/metal.cpp) can ask "is there a Metal device?" without
// including this whole header's ObjC-adjacent surface — and, critically, WITHOUT
// depending on static-initialization ORDER. Asking "did the backend registrar
// already run?" from another TU's static initializer is unspecified-order and
// would intermittently skip platform registration; probing the DEVICE is
// order-independent and gives the same answer.
bool MetalDeviceAvailable();

// Dispatch descriptor for the elementwise/row kernels this skeleton registers.
// One threadgroup per ROW, `tg_size` threads per threadgroup, each thread
// striding the row. The row/threadgroup mapping is llama.cpp's for the same op
// shapes (e.g. `ggml_metal_op_norm` dispatches one threadgroup per row with a
// threadgroup-memory reduction).
struct Dispatch {
  int64_t rows = 0;
  uint32_t tg_size = 0;
};

// Choose a threadgroup size for a row of `width` elements: the largest power of
// two <= min(width, cap), floored at 32 (the M4's threadExecutionWidth). `cap`
// defaults to the device limit; row-reducing kernels MUST pass their pipeline's
// PipelineMaxThreads instead (see above). The result is always a power of two,
// which the MSL tree reduction depends on, even when `cap` is not.
uint32_t ChooseThreadgroupSize(int64_t width, size_t cap = 0);

}  // namespace vt::metal

#endif  // VT_METAL_METAL_CONTEXT_H_
