// Metal backend skeleton unit gates (BACKEND-METAL-MLX, W0). Newly authored —
// vLLM has no Metal tests to port. Mirrors the shape of tests/vt/test_backend.cpp
// (the CPU backend's own gates) so the two are read side by side.
//
// This TU is COMPILED ONLY in a Metal build (tests/CMakeLists.txt gates it on
// VLLM_CPP_METAL) but is deliberately plain C++: every assertion goes through
// the public vt:: seam, which is the point — if the skeleton needed ObjC in a
// test to be checkable, the seam would be leaking.
//
// Cross-device NUMERIC equality vs the CPU oracle is NOT here; it lives in
// tests/vt/test_backend_cross_device.cpp, which runs against every registered
// non-CPU backend and so covers Metal automatically.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;

TEST_CASE("Metal backend is registered on a Metal-capable host") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);

  // Apple silicon is unified memory. This is load-bearing well beyond a fact
  // about the hardware: vt::Backend's SEVEN async-output primitive defaults
  // (src/vt/backend.cpp:19-32) are documented correct exactly for unified
  // backends, so the skeleton inherits them instead of implementing them.
  CHECK(metal.UnifiedMemory());

  // MTLIndirectCommandBuffer is the eventual mapping (include/vt/backend.h:92)
  // but is NOT implemented; the honest answer today is false, and the base class
  // makes BeginCapture throw loudly rather than silently no-op.
  CHECK_FALSE(metal.SupportsGraphCapture());
  Queue q = metal.CreateQueue();
  CHECK_THROWS_AS(metal.BeginCapture(q), std::runtime_error);

  CHECK(q.device.type == DeviceType::kMETAL);
  CHECK(q.handle != nullptr);  // the shared MTLCommandQueue
  CHECK(q.id != 0);            // a live identity for the workspace-key machinery

  // Apple GPU family as the capability pair; family 9 on the M4 gate box. The
  // assertion is deliberately ">= 1", not "== 9": the gate is that a REAL probe
  // ran, not that we are on one specific Mac.
  CHECK(metal.DeviceCapabilityMajor() >= 1);
  CHECK(metal.DeviceCapabilityMinor() == 0);

  metal.DestroyQueue(q);
}

TEST_CASE("Metal allocations are 64B-aligned, byte-exact and freeable") {
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();

  void* p = metal.Alloc(64);
  REQUIRE(p != nullptr);
  // include/vt/backend.h:26 — vt::StepArena depends on >= 64-byte alignment.
  CHECK(reinterpret_cast<uintptr_t>(p) % 64 == 0);

  metal.Memset(q, p, 0xAB, 64);
  metal.Synchronize(q);
  unsigned char dst[64];
  metal.Copy(q, dst, p, 64);
  metal.Synchronize(q);
  CHECK(dst[0] == 0xAB);
  CHECK(dst[63] == 0xAB);
  metal.Free(p);

  // A zero-byte request still yields a valid, distinct, freeable block (the CPU
  // backend's contract, which the arena relies on).
  void* z = metal.Alloc(0);
  CHECK(z != nullptr);
  metal.Free(z);
  metal.Free(nullptr);  // no-op

  metal.DestroyQueue(q);
}

TEST_CASE("Metal resolves INTERIOR pointers (tensor views/slices) to the owning buffer") {
  // vt::Tensor::Slice / ::View hand out pointers INTO an allocation, while Metal
  // binds resources, not pointers. The allocation registry (src/vt/metal/
  // metal_buffers.h) is what bridges that; this case is its gate.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};

  const int64_t rows = 4, cols = 8;
  auto* base = static_cast<float*>(metal.Alloc(rows * cols * sizeof(float)));
  std::vector<float> host(rows * cols);
  for (size_t i = 0; i < host.size(); ++i) host[i] = -1.0f * static_cast<float>(i + 1);
  metal.Copy(q, base, host.data(), host.size() * sizeof(float));

  // Operate on rows [1,3) only — an INTERIOR pointer at a non-zero byte offset.
  Tensor sub = Tensor::Contiguous(base + cols, vt::DType::kF32, d, {2, cols});
  vt::Relu(q, sub, sub);
  metal.Synchronize(q);

  std::vector<float> back(host.size());
  metal.Copy(q, back.data(), base, back.size() * sizeof(float));
  metal.Synchronize(q);
  // Rows 0 and 3 untouched (bit-exact); rows 1-2 relu'd to zero (input was all
  // negative), which also proves the buffer OFFSET was applied and not ignored.
  CHECK(back[0] == host[0]);
  CHECK(back[cols * 3] == host[cols * 3]);
  for (int64_t i = cols; i < cols * 3; ++i) CHECK(back[i] == 0.0f);

  metal.Free(base);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal rejects memory it did not allocate, loudly") {
  // Handing a Metal kernel a host std::vector is THE bring-up mistake; it must
  // throw, never read garbage.
  Backend& metal = vt::GetBackend(DeviceType::kMETAL);
  Queue q = metal.CreateQueue();
  const Device d{DeviceType::kMETAL, 0};
  std::vector<float> host(64, 1.0f);
  Tensor t = Tensor::Contiguous(host.data(), vt::DType::kF32, d, {8, 8});
  CHECK_THROWS_AS(vt::Relu(q, t, t), std::runtime_error);
  metal.DestroyQueue(q);
}

TEST_CASE("Metal platform is registered and reports unified/no-pool residency") {
  vllm::platforms::Platform& p = vllm::platforms::GetPlatform(DeviceType::kMETAL);
  CHECK(p.device_type() == DeviceType::kMETAL);
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

  // No Metal attention kernel exists yet, so the priority list is EMPTY by
  // design (see src/vllm/platforms/metal.cpp) — selection must fail loudly
  // rather than name a backend whose kernels are absent.
  CHECK(p.get_attn_backend_priority().empty());
}

TEST_CASE("Metal registers the W0 op set and NOT the unimplemented rest") {
  // The skeleton's registered surface, stated as an executable fact so a later
  // work row cannot quietly claim more than it implements.
  for (vt::OpId op : {vt::OpId::kAdd, vt::OpId::kRelu, vt::OpId::kSiluAndMul,
                      vt::OpId::kCastBf16, vt::OpId::kCastF32, vt::OpId::kLayerNorm,
                      vt::OpId::kRmsNorm, vt::OpId::kFusedChain}) {
    CHECK(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  // Still stubbed — GEMM, attention, KV cache, quant, sampling. A partial
  // backend is a supported state (src/vt/ops.cpp:104-111 throws on lookup).
  for (vt::OpId op : {vt::OpId::kMatmul, vt::OpId::kMatmulBT, vt::OpId::kPagedAttention,
                      vt::OpId::kReshapeAndCache, vt::OpId::kEmbedding,
                      vt::OpId::kGreedyArgmax}) {
    CHECK_FALSE(vt::OpRegistered(op, DeviceType::kMETAL));
  }
  CHECK_THROWS_AS(vt::GetOp(vt::OpId::kMatmul, DeviceType::kMETAL), std::runtime_error);
}
