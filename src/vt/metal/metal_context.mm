// Metal backend — device / command-queue / runtime-MSL-library scaffolding.
// See metal_context.h for the port map (llama.cpp `ggml/src/ggml-metal/` @
// 237ad9b96). BACKEND-METAL-MLX, W0 skeleton.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <string>

#include "metal_context.h"
#include "metal_msl.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vt::metal {
namespace {

// The highest MTLGPUFamilyApple<N> the device claims. Mirrors llama.cpp's
// `ggml_metal_device_init` family probe (it walks the same enum downwards). The
// M4 gate box reports 9 (measured in the spike).
int ProbeAppleFamily(id<MTLDevice> dev) {
  for (int n = 9; n >= 1; --n) {
    const auto family = static_cast<MTLGPUFamily>(static_cast<NSInteger>(MTLGPUFamilyApple1) + (n - 1));
    if ([dev supportsFamily:family]) return n;
  }
  return 0;
}

}  // namespace

bool MetalContext::Available() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    const bool ok = dev != nil;
    [dev release];
    return ok;
  }
}

bool MetalDeviceAvailable() { return MetalContext::Available(); }

MetalContext::MetalContext() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    VT_CHECK(dev != nil, "metal: no system default MTLDevice");

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    VT_CHECK(queue != nil, "metal: newCommandQueue failed");

    // Runtime MSL compilation. There is NO offline `metal` compiler on a
    // Command-Line-Tools-only host (the M4 gate box has no full Xcode), so a
    // prebuilt .metallib is not an option and this is the ONLY path — see
    // .agents/specs/backend-fanout-metal-vulkan-xpu.md § Risks/decisions 2.
    // llama.cpp keeps the same call as its no-metallib fallback
    // (`ggml_metal_library_init`); we take it unconditionally.
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    // IEEE semantics, NOT Metal's default fast-math. Our kernels are gated
    // against the CPU reference at NMSE <= 5e-4 and reproduce its exact
    // expression forms (bf16 round-trips, 1/sqrt rather than rsqrt); fast-math
    // would licence reassociation and a reciprocal-sqrt approximation that make
    // that comparison meaningless.
    if (@available(macOS 15.0, *)) {
      opts.mathMode = MTLMathModeSafe;
    } else {
      VT_CHECK(false, "metal: macOS 15+ required (MTLMathModeSafe); "
                      "the backend refuses to build its library under "
                      "unpinned fast-math semantics");
    }

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalSource];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    [opts release];
    if (lib == nil) {
      const char* what = err != nil ? [[err localizedDescription] UTF8String] : "unknown error";
      // The diagnostics are the entire reason runtime compilation is workable:
      // surface them verbatim rather than a generic failure.
      VT_CHECK(false, std::string("metal: newLibraryWithSource failed: ") + what);
    }

    device_ = static_cast<void*>(dev);
    queue_ = static_cast<void*>(queue);
    library_ = static_cast<void*>(lib);
    pipelines_ = static_cast<void*>([[NSMutableDictionary alloc] init]);
    pipelines_lock_ = static_cast<void*>(new std::mutex());

    gpu_family_apple_ = ProbeAppleFamily(dev);
    max_tg_threads_ = static_cast<size_t>(dev.maxThreadsPerThreadgroup.width);
    tg_mem_bytes_ = static_cast<size_t>(dev.maxThreadgroupMemoryLength);
    unified_memory_ = dev.hasUnifiedMemory;
  }
}

MetalContext& MetalContext::Get() {
  // Function-local static: thread-safe initialization, constructed on first use
  // and deliberately never destroyed (process-lifetime, matching llama.cpp's
  // `g_ctx_dev` singleton). Registration happens from a static initializer, so a
  // destructor here would be ordered against other TUs' teardown for no benefit.
  static MetalContext* ctx = new MetalContext();
  return *ctx;
}

PipelineHandle MetalContext::Pipeline(const std::string& fn_name) {
  auto* lock = static_cast<std::mutex*>(pipelines_lock_);
  std::lock_guard<std::mutex> guard(*lock);
  auto* cache = static_cast<NSMutableDictionary*>(pipelines_);
  NSString* key = [NSString stringWithUTF8String:fn_name.c_str()];
  id existing = [cache objectForKey:key];
  if (existing != nil) return static_cast<void*>(existing);

  @autoreleasepool {
    id<MTLLibrary> lib = static_cast<id<MTLLibrary>>(library_);
    id<MTLFunction> fn = [lib newFunctionWithName:key];
    VT_CHECK(fn != nil, std::string("metal: kernel function not found: ") + fn_name);
    NSError* err = nil;
    id<MTLDevice> dev = static_cast<id<MTLDevice>>(device_);
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (pso == nil) {
      const char* what = err != nil ? [[err localizedDescription] UTF8String] : "unknown error";
      VT_CHECK(false, std::string("metal: pipeline creation failed for ") + fn_name + ": " + what);
    }
    [cache setObject:pso forKey:key];
    [pso release];  // the dictionary owns it now; the context outlives the process
    return static_cast<void*>([cache objectForKey:key]);
  }
}

size_t MetalContext::PipelineMaxThreads(const std::string& fn_name) {
  auto pso = static_cast<id<MTLComputePipelineState>>(Pipeline(fn_name));
  return static_cast<size_t>([pso maxTotalThreadsPerThreadgroup]);
}

uint32_t ChooseThreadgroupSize(int64_t width, size_t cap) {
  if (cap == 0) cap = MetalContext::Get().max_threads_per_threadgroup();
  // Also cap at VT_TG_MAX (1024) — the threadgroup scratch array in metal_msl.h
  // is sized for exactly that, so a device or pipeline advertising more must
  // still be clamped.
  if (cap > 1024) cap = 1024;
  uint32_t tg = 32;  // the M4's threadExecutionWidth; never dispatch below one simd
  // Doubling from 32 keeps the result a power of two even when `cap` is not
  // (pipeline limits are frequently values like 768) — the MSL halving
  // reduction in vt_tg_sum requires that.
  while (static_cast<int64_t>(tg) * 2 <= width && static_cast<size_t>(tg) * 2 <= cap) {
    tg *= 2;
  }
  return tg;
}

}  // namespace vt::metal
