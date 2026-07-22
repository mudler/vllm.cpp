// Vulkan backend skeleton unit gates (BACKEND-VULKAN, W0). Newly authored — vLLM
// has no Vulkan tests to port, and llama.cpp's `test-backend-ops` is a ggml
// harness with no vt:: analogue. Mirrors the shape of
// tests/vt/test_metal_backend.cpp (and through it tests/vt/test_backend.cpp) so
// the three are read side by side.
//
// This TU is COMPILED ONLY in a Vulkan build (tests/CMakeLists.txt gates it on
// VLLM_CPP_VULKAN) and every assertion goes through the public vt:: seam — if
// the skeleton needed Vulkan headers in a test to be checkable, the seam would
// be leaking.
//
// Cross-device NUMERIC equality vs an oracle is NOT here; it lives in
// tests/vt/test_backend_cross_device.cpp, which runs against every registered
// non-CPU backend and so covers Vulkan automatically — and which, on the GB10
// box, compares Vulkan against a CUDA build in the SAME binary, the strongest
// cross-backend oracle in the project.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/vulkan/vulkan_context.h"
#include "vt/vulkan/vulkan_spirv.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

namespace {

// A Vulkan-ENABLED build can legitimately run where there is no loader or no
// conformant device (a headless CI container), in which case the registrars stay
// silent by design. Every case below is skipped in that state rather than
// failing — but the skip is REPORTED, so a silently-unregistered backend on a
// box that does have one cannot masquerade as a pass.
bool VulkanPresent() { return vt::vulkan::VulkanDeviceAvailable(); }

}  // namespace

TEST_CASE("the committed SPIR-V table is present and well-formed") {
  // Independent of any device: this is a property of the CHECKED-IN artifact, so
  // it also gates the generator (scripts/gen-vulkan-spirv.py) on a box with no
  // Vulkan at all.
  const size_t n = sizeof(vt::vulkan::kSpirvModules) / sizeof(vt::vulkan::kSpirvModules[0]);
  CHECK(n == 7);
  for (const auto& m : vt::vulkan::kSpirvModules) {
    CAPTURE(m.name);
    REQUIRE(m.word_count > 5);          // a SPIR-V header alone is 5 words
    CHECK(m.words[0] == 0x07230203u);   // SPIR-V magic
  }
  // The eight registered ops are served by exactly these seven modules (kCastBf16
  // and kCastF32 share vt_cast), so a rename in either direction breaks here
  // rather than at pipeline-creation time on a device we might not have.
  for (const char* want : {"vt_add", "vt_cast", "vt_fused_chain", "vt_layer_norm", "vt_relu",
                           "vt_rms_norm", "vt_silu_and_mul"}) {
    bool found = false;
    for (const auto& m : vt::vulkan::kSpirvModules) {
      if (std::strcmp(m.name, want) == 0) found = true;
    }
    CAPTURE(want);
    CHECK(found);
  }
}

TEST_CASE("Vulkan backend is registered on a Vulkan-capable host") {
  if (!VulkanPresent()) {
    MESSAGE("no Vulkan loader or no conformant device on this host; skipping");
    return;
  }
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);

  // GB10 exposes one 89.72 GiB DEVICE_LOCAL|HOST_VISIBLE heap, and llvmpipe is a
  // CPU device, so both report unified. This is load-bearing beyond a hardware
  // fact: vt::Backend's SEVEN async-output primitive defaults
  // (src/vt/backend.cpp:19-32) are documented correct exactly for unified
  // backends, so the skeleton inherits them instead of implementing them.
  CHECK(vk.UnifiedMemory());

  // A pre-recorded VkCommandBuffer is the eventual mapping
  // (include/vt/backend.h:92) but is NOT implemented; the honest answer today is
  // false, and the base class makes BeginCapture throw loudly rather than
  // silently no-op.
  CHECK_FALSE(vk.SupportsGraphCapture());
  Queue q = vk.CreateQueue();
  CHECK_THROWS_AS(vk.BeginCapture(q), std::runtime_error);

  CHECK(q.device.type == DeviceType::kVULKAN);
  CHECK(q.handle != nullptr);  // the shared VkQueue
  CHECK(q.id != 0);            // a live identity for the workspace-key machinery

  // The Vulkan API version as the capability pair. The assertion is deliberately
  // ">= 1.1", not "== 1.4": the gate is that a REAL probe ran AND that the
  // version floor this backend needs (16-bit storage in core) actually holds,
  // not that we are on one specific GPU.
  CHECK(vk.DeviceCapabilityMajor() >= 1);
  CHECK((vk.DeviceCapabilityMajor() > 1 || vk.DeviceCapabilityMinor() >= 1));

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan allocations are 64B-aligned, byte-exact and freeable") {
  if (!VulkanPresent()) return;
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();

  void* p = vk.Alloc(64);
  REQUIRE(p != nullptr);
  // include/vt/backend.h:26 — vt::StepArena depends on >= 64-byte alignment.
  CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);

  vk.Memset(q, p, 0xAB, 64);
  vk.Synchronize(q);
  unsigned char dst[64];
  vk.Copy(q, dst, p, 64);
  vk.Synchronize(q);
  CHECK(dst[0] == 0xAB);
  CHECK(dst[63] == 0xAB);
  vk.Free(p);

  // A zero-byte request still yields a valid, distinct, freeable block (the CPU
  // backend's contract, which the arena relies on).
  void* z = vk.Alloc(0);
  CHECK(z != nullptr);
  vk.Free(z);
  vk.Free(nullptr);  // no-op

  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan resolves INTERIOR pointers (tensor views/slices) to the owning buffer") {
  if (!VulkanPresent()) return;
  // vt::Tensor::Slice / ::View hand out pointers INTO an allocation, while Vulkan
  // binds resources, not pointers. The allocation registry
  // (src/vt/vulkan/vulkan_buffers.h) is what bridges that; this case is its gate.
  // It is a STRONGER gate on Vulkan than on Metal, because Vulkan additionally
  // has a descriptor-offset ALIGNMENT rule that this backend sidesteps by binding
  // whole buffers and passing the byte offset in push constants — if that ever
  // regressed to a descriptor offset, a non-zero interior offset would either
  // fail validation or silently read shifted data, and this case catches both.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  const int64_t rows = 4, cols = 8;
  auto* base = static_cast<float*>(vk.Alloc(rows * cols * sizeof(float)));
  std::vector<float> host(rows * cols);
  for (size_t i = 0; i < host.size(); ++i) host[i] = -1.0f * static_cast<float>(i + 1);
  vk.Copy(q, base, host.data(), host.size() * sizeof(float));

  // Operate on rows [1,3) only — an INTERIOR pointer at byte offset 32, which is
  // NOT a multiple of a typical minStorageBufferOffsetAlignment of 256.
  Tensor sub = Tensor::Contiguous(base + cols, vt::DType::kF32, d, {2, cols});
  vt::Relu(q, sub, sub);
  vk.Synchronize(q);

  std::vector<float> back(host.size());
  vk.Copy(q, back.data(), base, back.size() * sizeof(float));
  vk.Synchronize(q);
  // Rows 0 and 3 untouched (bit-exact); rows 1-2 relu'd to zero (input was all
  // negative), which also proves the buffer OFFSET was applied and not ignored.
  CHECK(back[0] == host[0]);
  CHECK(back[cols * 3] == host[cols * 3]);
  for (int64_t i = cols; i < cols * 3; ++i) CHECK(back[i] == 0.0f);

  vk.Free(base);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan rejects memory it did not allocate, loudly") {
  if (!VulkanPresent()) return;
  // Handing a Vulkan kernel a host std::vector is THE bring-up mistake; it must
  // throw, never read garbage.
  Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};
  std::vector<float> host(64, 1.0f);
  Tensor t = Tensor::Contiguous(host.data(), vt::DType::kF32, d, {8, 8});
  CHECK_THROWS_AS(vt::Relu(q, t, t), std::runtime_error);
  vk.DestroyQueue(q);
}

TEST_CASE("Vulkan platform is registered and reports unified/no-pool residency") {
  if (!VulkanPresent()) return;
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kVULKAN);
  CHECK(p.device_type() == DeviceType::kVULKAN);
  CHECK_FALSE(p.is_cuda());
  CHECK_FALSE(p.is_cpu());
  CHECK(p.is_unified_memory());
  CHECK_FALSE(p.supports_graph_capture());

  CHECK(p.get_device_capability().present());
  CHECK(p.get_device_capability().major >= 1);

  // interface.py:181-187 order — bf16 is the default fallback.
  REQUIRE(p.supported_dtypes().size() == 3);
  CHECK(p.supported_dtypes()[0] == vt::DType::kBF16);

  // Unified memory: never free the only copy, never pool device scratch.
  const auto rp = p.residency_policy();
  CHECK_FALSE(rp.release_host_weights_after_upload);
  CHECK_FALSE(rp.uses_device_memory_pool);

  // No Vulkan attention kernel exists yet, so the priority list is EMPTY by
  // design (see src/vllm/platforms/vulkan.cpp) — selection must fail loudly
  // rather than name a backend whose kernels are absent.
  CHECK(p.get_attn_backend_priority().empty());
}

TEST_CASE("Vulkan registers the W0 op set and NOT the unimplemented rest") {
  if (!VulkanPresent()) return;
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain}) {
    CHECK(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  // Still stubbed — GEMM, attention, KV cache, quant, sampling. A partial backend
  // is a supported state (src/vt/ops.cpp:104-111 throws on lookup). NO MODEL RUNS
  // ON VULKAN.
  for (vt::OpId op : {vt::OpId::kMatmul, vt::OpId::kMatmulBT, vt::OpId::kPagedAttention,
                      vt::OpId::kReshapeAndCache, vt::OpId::kEmbedding,
                      vt::OpId::kGreedyArgmax}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kVULKAN));
  }
  CHECK_THROWS_AS(vt::GetOp(vt::OpId::kMatmul, DeviceType::kVULKAN), std::runtime_error);
}

TEST_CASE("Vulkan float-controls are PROBED and reported, not assumed") {
  if (!VulkanPresent()) return;
  // The relaxed-precision knobs Vulkan leaves implementation-defined. We cannot
  // pin fp32 denormal/signed-zero behaviour from GLSL without
  // SPV_KHR_float_controls execution modes, so the honest gate is to RECORD what
  // this device does. Both outcomes are acceptable — the shaders avoid
  // `inversesqrt` and carry integer bf16/f16 codecs precisely so that neither
  // knob can move a gated result — but a silent change here would be the first
  // clue if a future NMSE regression appeared, so it is printed rather than
  // asserted to a particular value.
  auto& ctx = vt::vulkan::VulkanContext::Get();
  MESSAGE("vulkan device: " << ctx.device_name() << " (API " << ctx.api_major() << "."
                            << ctx.api_minor() << ")");
  MESSAGE("shaderDenormPreserveFloat32 = " << ctx.denorm_preserve_f32());
  MESSAGE("shaderSignedZeroInfNanPreserveFloat32 = "
          << ctx.signed_zero_inf_nan_preserve_f32());
  // What we DO require: the workgroup-count limit must cover the dispatches the
  // skeleton makes (one workgroup per row).
  CHECK(ctx.max_workgroup_count_x() >= 65535u);
}
