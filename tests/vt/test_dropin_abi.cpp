// Ported FROM vLLM tests/cuda/test_cuda_context.py:54-83 and
// tests/v1/cudagraph/test_cudagraph_dispatch.py:271-354 @
// e24d1b24fe96. The vt-specific workspace role/lifecycle assertions exercise
// the torch-free adapter contract accepted in .agents/specs/dropin-kernel-abi.md.

#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

#ifdef VLLM_CPP_CUDA
  #include <cuda_runtime_api.h>

  #include "vt/cuda/cuda_dropin.h"
#endif

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DropinProbeArgs;
using vt::KernelLayout;
using vt::OpId;
using vt::Queue;
using vt::ScalarTypeId;
using vt::Tensor;
using vt::WorkspaceInit;
using vt::WorkspaceSlot;

constexpr Device Cpu() { return Device{DeviceType::kCPU, 0}; }
constexpr Device Cuda(int index = 0) { return Device{DeviceType::kCUDA, index}; }

void CpuProbe(Queue&, Tensor& output, const Tensor& input,
              const DropinProbeArgs& args) {
  for (int64_t row = 0; row < input.shape[0]; ++row) {
    for (int64_t col = 0; col < input.shape[1]; ++col) {
      output.Ptr<float>()[row * output.stride[0] + col * output.stride[1]] =
          input.Ptr<float>()[row * input.stride[0] + col * input.stride[1]] +
          args.scalar;
    }
  }
}

#ifdef VLLM_CPP_CUDA

bool HasCuda() {
  try {
    (void)vt::GetBackend(DeviceType::kCUDA);
    return vt::cuda::DeviceCount() > 0;
  } catch (const std::runtime_error&) {
    return false;
  }
}

void CopyAsync(const Queue& q, void* output, const void* input, size_t bytes) {
  vt::cuda::DeviceGuard guard(q.device);
  vt::cuda::Check(cudaMemcpyAsync(output, input, bytes, cudaMemcpyDefault,
                                  vt::cuda::Stream(q)),
                  "test cudaMemcpyAsync");
}

void MemsetAsync(const Queue& q, void* output, int value, size_t bytes) {
  vt::cuda::DeviceGuard guard(q.device);
  vt::cuda::Check(cudaMemsetAsync(output, value, bytes, vt::cuda::Stream(q)),
                  "test cudaMemsetAsync");
}

void Sync(const Queue& q) {
  vt::cuda::DeviceGuard guard(q.device);
  vt::cuda::Check(cudaStreamSynchronize(vt::cuda::Stream(q)),
                  "test cudaStreamSynchronize");
}

#endif

}  // namespace

TEST_CASE("dropin ABI scalar IDs match upstream packing and storage stays explicit") {
  // Independent fixed values from vllm::ScalarType::id() field packing.
  CHECK(vt::scalar_type::kF32 == ScalarTypeId{1125899906914056});
  CHECK(vt::scalar_type::kBF16 == ScalarTypeId{1125899906909960});
  CHECK(vt::scalar_type::kFE2M1f == ScalarTypeId{562949953487106});
  CHECK(vt::scalar_type::kFE4M3fn == ScalarTypeId{2814749767172868});

  CHECK(vt::ToScalarType(vt::DType::kF32) == vt::scalar_type::kF32);
  CHECK(vt::ToScalarType(vt::DType::kF16) == vt::scalar_type::kF16);
  CHECK(vt::ToScalarType(vt::DType::kBF16) == vt::scalar_type::kBF16);
  CHECK(vt::ToScalarType(vt::DType::kI8) == vt::scalar_type::kI8);
  CHECK(vt::scalar_type::kI8 != vt::scalar_type::kFE2M1f);
  CHECK(vt::scalar_type::kFE2M1f != vt::scalar_type::kFE4M3fn);
}

TEST_CASE("dropin ABI Describe preserves element geometry and validates packed layouts") {
  std::array<float, 12> data{};
  Tensor base = Tensor::Contiguous(data.data(), vt::DType::kF32, Cpu(), {3, 4});
  Tensor slice = base.Slice(0, 1, 3);
  const vt::KernelTensorDesc desc =
      vt::Describe(slice, vt::scalar_type::kF32, KernelLayout::kStrided);
  CHECK(desc.data == slice.data);
  CHECK(desc.storage_dtype == vt::DType::kF32);
  CHECK(desc.scalar_type == vt::scalar_type::kF32);
  CHECK(desc.rank == 2);
  CHECK(desc.shape[0] == 2);
  CHECK(desc.shape[1] == 4);
  CHECK(desc.stride[0] == 4);
  CHECK(desc.stride[1] == 1);

  std::array<uint8_t, 32> packed{};
  Tensor raw = Tensor::Contiguous(packed.data(), vt::DType::kI8, Cpu(), {4, 8});
  CHECK_NOTHROW(vt::Describe(raw, vt::scalar_type::kFE2M1f,
                             KernelLayout::kPackedTwoFp4PerByte));
  CHECK_NOTHROW(vt::Describe(raw, vt::scalar_type::kFE4M3fn,
                             KernelLayout::kBlockScaleLinear));
  CHECK_NOTHROW(vt::Describe(raw, vt::scalar_type::kFE8M0fnu,
                             KernelLayout::kBlockScaleSwizzled));
  CHECK_NOTHROW(vt::Describe(raw, vt::scalar_type::kI4,
                             KernelLayout::kMarlinInterleaved));
  CHECK_THROWS_AS(vt::Describe(raw, vt::scalar_type::kI8,
                               KernelLayout::kPackedTwoFp4PerByte),
                  std::runtime_error);
  CHECK_THROWS_AS(vt::Describe(raw, vt::scalar_type::kFE2M1f,
                               KernelLayout::kBlockScaleLinear),
                  std::runtime_error);
  CHECK_THROWS_AS(vt::Describe(base, vt::scalar_type::kFE2M1f,
                               KernelLayout::kStrided),
                  std::runtime_error);
}

TEST_CASE("dropin ABI queue key isolates default handles devices roles and recycled IDs") {
  Queue first{Cuda(0), nullptr};
  Queue second{Cuda(0), nullptr};
  REQUIRE(first.id != 0);
  REQUIRE(second.id != 0);
  CHECK(first.id != second.id);

  const vt::WorkspaceKey a =
      vt::MakeWorkspaceKey(first, OpId::kDropinProbe, WorkspaceSlot::kWorkspace);
  const vt::WorkspaceKey same =
      vt::MakeWorkspaceKey(first, OpId::kDropinProbe, WorkspaceSlot::kWorkspace);
  CHECK(a == same);
  CHECK_FALSE(a == vt::MakeWorkspaceKey(second, OpId::kDropinProbe,
                                        WorkspaceSlot::kWorkspace));
  CHECK_FALSE(a == vt::MakeWorkspaceKey(first, OpId::kDropinProbe,
                                        WorkspaceSlot::kOutput));
  CHECK_FALSE(a == vt::MakeWorkspaceKey(first, OpId::kMatmul,
                                        WorkspaceSlot::kWorkspace));

  Queue other_device = first;
  other_device.device = Cuda(1);
  CHECK_FALSE(a == vt::MakeWorkspaceKey(other_device, OpId::kDropinProbe,
                                        WorkspaceSlot::kWorkspace));

  Queue recycled = first;
  recycled.id = vt::NextQueueId();
  CHECK(recycled.handle == first.handle);
  CHECK_FALSE(a == vt::MakeWorkspaceKey(recycled, OpId::kDropinProbe,
                                        WorkspaceSlot::kWorkspace));
}

TEST_CASE("dropin ABI device-explicit CPU resources and typed registration") {
  Queue q = vt::CreateQueue(Cpu());
  CHECK(q.device == Cpu());
  CHECK(q.id != 0);

  void* allocation = vt::Alloc(Cpu(), 128);
  REQUIRE(allocation != nullptr);
  std::memset(allocation, 0x5a, 128);
  CHECK(static_cast<unsigned char*>(allocation)[127] == 0x5a);
  vt::Free(Cpu(), allocation);

  std::array<float, 6> input{1, 2, 3, 4, 5, 6};
  std::array<float, 6> output{};
  Tensor in = Tensor::Contiguous(input.data(), vt::DType::kF32, Cpu(), {2, 3});
  Tensor out = Tensor::Contiguous(output.data(), vt::DType::kF32, Cpu(), {2, 3});
  vt::RegisterTypedOp<vt::DropinProbeFn>(
      OpId::kDropinProbe, DeviceType::kCPU,
      static_cast<vt::DropinProbeFn>(&CpuProbe));
  DropinProbeArgs args;
  args.scalar = 0.75f;
  vt::DropinProbe(q, out, in, args);
  for (size_t i = 0; i < input.size(); ++i) {
    CHECK(output[i] == doctest::Approx(input[i] + 0.75f));
  }

  CHECK_THROWS_AS(vt::CreateQueue(Device{DeviceType::kCPU, 1}),
                  std::runtime_error);
  vt::DestroyQueue(q);
  CHECK(q.id == 0);
  CHECK(q.handle == nullptr);
  CHECK_NOTHROW(vt::DestroyQueue(q));
}

TEST_CASE("dropin ABI CUDA explicit queues isolate default handles and queue cleanup") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CUDA was not compiled; skipping ported CUDA context case");
  return;
#else
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping ported CUDA context case");
    return;
  }

  const size_t before = vt::cuda::WorkspaceEntryCountForTesting();
  Queue default_a{Cuda(0), nullptr};
  Queue default_b{Cuda(0), nullptr};
  auto a = vt::cuda::AcquireWorkspace(default_a, OpId::kDropinProbe,
                                      WorkspaceSlot::kWorkspace, 64, 64,
                                      WorkspaceInit::kUninitialized);
  auto b = vt::cuda::AcquireWorkspace(default_b, OpId::kDropinProbe,
                                      WorkspaceSlot::kWorkspace, 64, 64,
                                      WorkspaceInit::kUninitialized);
  CHECK(a.data != nullptr);
  CHECK(b.data != nullptr);
  CHECK(a.data != b.data);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before + 2);
  vt::DestroyQueue(default_a);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before + 1);
  vt::DestroyQueue(default_b);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before);

  Queue first = vt::CreateQueue(Cuda(0));
  const uint64_t first_id = first.id;
  CHECK(first.handle != nullptr);
  vt::DestroyQueue(first);
  Queue second = vt::CreateQueue(Cuda(0));
  CHECK(second.id != first_id);
  vt::DestroyQueue(second);

  const int target_device = vt::cuda::DeviceCount() > 1 ? 1 : 0;
  auto set_device = std::async(std::launch::async, [target_device] {
    vt::cuda::DeviceGuard guard(Cuda(target_device));
    int current = -1;
    vt::cuda::Check(cudaGetDevice(&current), "ported set-device check");
    return current;
  });
  CHECK(set_device.get() == target_device);
  CHECK_THROWS_AS(vt::cuda::DeviceGuard(Cpu()), std::runtime_error);
#endif
}

TEST_CASE("dropin ABI CUDA workspace growth roles init policies and device scalar") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CUDA was not compiled; skipping workspace runtime case");
  return;
#else
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping workspace runtime case");
    return;
  }

  const size_t before = vt::cuda::WorkspaceEntryCountForTesting();
  Queue q = vt::CreateQueue(Cuda(0));

  auto first = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                          WorkspaceSlot::kWorkspace, 64, 64,
                                          WorkspaceInit::kUninitialized);
  REQUIRE(first.data != nullptr);
  CHECK(first.grew);
  CHECK(reinterpret_cast<uintptr_t>(first.data) % 64 == 0);
  auto reuse = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                          WorkspaceSlot::kWorkspace, 32, 32,
                                          WorkspaceInit::kUninitialized);
  CHECK_FALSE(reuse.grew);
  CHECK(reuse.data == first.data);
  CHECK(reuse.capacity == 64);
  auto grown = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                          WorkspaceSlot::kWorkspace, 512, 256,
                                          WorkspaceInit::kUninitialized);
  CHECK(grown.grew);
  CHECK(grown.capacity == 512);
  CHECK(grown.data != first.data);
  CHECK(reinterpret_cast<uintptr_t>(grown.data) % 256 == 0);
  auto output = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                           WorkspaceSlot::kOutput, 512, 256,
                                           WorkspaceInit::kUninitialized);
  CHECK(output.data != grown.data);

  auto zero_once = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                              WorkspaceSlot::kLse, 32, 16,
                                              WorkspaceInit::kZeroOnFirstUse);
  std::array<unsigned char, 32> host{};
  CopyAsync(q, host.data(), zero_once.data, host.size());
  Sync(q);
  for (unsigned char value : host) CHECK(value == 0);
  MemsetAsync(q, zero_once.data, 0x7b, host.size());
  auto zero_once_reuse = vt::cuda::AcquireWorkspace(
      q, OpId::kDropinProbe, WorkspaceSlot::kLse, 32, 16,
      WorkspaceInit::kZeroOnFirstUse);
  CopyAsync(q, host.data(), zero_once_reuse.data, host.size());
  Sync(q);
  for (unsigned char value : host) CHECK(value == 0x7b);

  auto zero_each = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                              WorkspaceSlot::kSemaphore, 32, 16,
                                              WorkspaceInit::kZeroEachUse);
  MemsetAsync(q, zero_each.data, 0x4d, host.size());
  zero_each = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                         WorkspaceSlot::kSemaphore, 32, 16,
                                         WorkspaceInit::kZeroEachUse);
  CopyAsync(q, host.data(), zero_each.data, host.size());
  Sync(q);
  for (unsigned char value : host) CHECK(value == 0);

  CHECK_THROWS_AS(vt::cuda::AcquireWorkspace(
                      q, OpId::kDropinProbe, WorkspaceSlot::kOutput, 512, 256,
                      WorkspaceInit::kZeroEachUse),
                  std::runtime_error);

  const auto scalar = vt::cuda::StageScalar(q, OpId::kDropinProbe,
                                            WorkspaceSlot::kDeviceScalar0,
                                            3.25f);
  float scalar_host = 0.0f;
  CopyAsync(q, &scalar_host, scalar.data, sizeof(float));
  Sync(q);
  CHECK(scalar_host == doctest::Approx(3.25f));

  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before + 5);
  vt::DestroyQueue(q);
  CHECK(q.id == 0);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before);
#endif
}

TEST_CASE("dropin ABI CUDA raw probe binds pointer geometry semantics workspace and stream") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CUDA was not compiled; skipping raw probe runtime case");
  return;
#else
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping raw probe runtime case");
    return;
  }

  Queue q = vt::CreateQueue(Cuda(0));
  constexpr size_t kCount = 12;
  std::array<float, kCount> input{};
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i) - 2.0f;
  void* device_input = vt::Alloc(Cuda(0), input.size() * sizeof(float));
  void* device_output = vt::Alloc(Cuda(0), input.size() * sizeof(float));
  CopyAsync(q, device_input, input.data(), input.size() * sizeof(float));

  Tensor in = Tensor::Contiguous(device_input, vt::DType::kF32, Cuda(0), {3, 4});
  Tensor out = Tensor::Contiguous(device_output, vt::DType::kF32, Cuda(0), {3, 4});
  DropinProbeArgs args;
  args.scalar = 1.5f;
  args.workspace_init = WorkspaceInit::kUninitialized;
  vt::DropinProbe(q, out, in, args);

  std::array<float, kCount> result{};
  CopyAsync(q, result.data(), device_output, result.size() * sizeof(float));
  auto marker = vt::cuda::AcquireWorkspace(q, OpId::kDropinProbe,
                                           args.workspace_slot,
                                           args.workspace_bytes,
                                           args.workspace_alignment,
                                           args.workspace_init);
  uint32_t marker_host = 0;
  CopyAsync(q, &marker_host, marker.data, sizeof(marker_host));
  Sync(q);
  for (size_t i = 0; i < result.size(); ++i) {
    CHECK(result[i] == doctest::Approx(input[i] + args.scalar));
  }
  CHECK(marker_host ==
        (static_cast<uint32_t>(args.scalar_type) ^ static_cast<uint32_t>(args.layout)));

  vt::Free(Cuda(0), device_input);
  vt::Free(Cuda(0), device_output);
  vt::DestroyQueue(q);
#endif
}

TEST_CASE("dropin ABI CUDA capture reuses addresses and rejects growth before launch") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CUDA was not compiled; skipping capture runtime case");
  return;
#else
  if (!HasCuda()) {
    MESSAGE("no CUDA backend registered; skipping capture runtime case");
    return;
  }

  Queue q = vt::CreateQueue(Cuda(0));
  void* device_input = vt::Alloc(Cuda(0), 4 * sizeof(float));
  void* device_output = vt::Alloc(Cuda(0), 4 * sizeof(float));
  const std::array<float, 4> input{1.0f, 2.0f, 3.0f, 4.0f};
  CopyAsync(q, device_input, input.data(), input.size() * sizeof(float));
  MemsetAsync(q, device_output, 0, 4 * sizeof(float));

  Tensor in = Tensor::Contiguous(device_input, vt::DType::kF32, Cuda(0), {2, 2});
  Tensor out = Tensor::Contiguous(device_output, vt::DType::kF32, Cuda(0), {2, 2});
  DropinProbeArgs args;
  args.scalar = 2.0f;
  args.workspace_bytes = 64;
  args.workspace_alignment = 64;
  args.workspace_init = WorkspaceInit::kZeroEachUse;
  const auto warm_workspace = vt::cuda::AcquireWorkspace(
      q, OpId::kDropinProbe, args.workspace_slot, args.workspace_bytes,
      args.workspace_alignment, args.workspace_init);
  (void)vt::cuda::StageScalar(q, OpId::kDropinProbe, args.scalar_slot, args.scalar);
  Sync(q);

  vt::Backend& backend = vt::GetBackend(DeviceType::kCUDA);
  backend.BeginCapture(q);
  vt::DropinProbe(q, out, in, args);
  const auto captured_workspace = vt::cuda::AcquireWorkspace(
      q, OpId::kDropinProbe, args.workspace_slot, args.workspace_bytes,
      args.workspace_alignment, args.workspace_init);
  CHECK(captured_workspace.data == warm_workspace.data);
  CHECK_FALSE(captured_workspace.grew);
  CHECK_THROWS_AS(vt::cuda::AcquireWorkspace(
                      q, OpId::kDropinProbe, args.workspace_slot,
                      args.workspace_bytes * 2, args.workspace_alignment,
                      args.workspace_init),
                  std::runtime_error);
  void* graph = backend.EndCaptureGraph(q);
  REQUIRE(graph != nullptr);

  std::array<float, 4> result{};
  CopyAsync(q, result.data(), device_output, result.size() * sizeof(float));
  Sync(q);
  for (float value : result) CHECK(value == 0.0f);

  backend.ReplayGraph(q, graph);
  Sync(q);

  CopyAsync(q, result.data(), device_output, result.size() * sizeof(float));
  Sync(q);
  for (size_t i = 0; i < result.size(); ++i) {
    CHECK(result[i] == doctest::Approx(input[i] + args.scalar));
  }
  backend.DestroyGraph(graph);
  vt::Free(Cuda(0), device_input);
  vt::Free(Cuda(0), device_output);
  vt::DestroyQueue(q);
#endif
}

TEST_CASE("dropin ABI CUDA two device indices remain isolated when hardware permits") {
#ifndef VLLM_CPP_CUDA
  MESSAGE("CUDA was not compiled; skipping multi-device runtime case");
  return;
#else
  if (!HasCuda() || vt::cuda::DeviceCount() < 2) {
    MESSAGE("fewer than two CUDA devices; tracked multi-device case skipped");
    return;
  }

  const size_t before = vt::cuda::WorkspaceEntryCountForTesting();
  Queue q0 = vt::CreateQueue(Cuda(0));
  Queue q1 = vt::CreateQueue(Cuda(1));
  CHECK(q0.device == Cuda(0));
  CHECK(q1.device == Cuda(1));
  CHECK(q0.id != q1.id);
  (void)vt::cuda::AcquireWorkspace(q0, OpId::kDropinProbe,
                                   WorkspaceSlot::kWorkspace, 64, 64,
                                   WorkspaceInit::kUninitialized);
  (void)vt::cuda::AcquireWorkspace(q1, OpId::kDropinProbe,
                                   WorkspaceSlot::kWorkspace, 64, 64,
                                   WorkspaceInit::kUninitialized);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before + 2);
  vt::DestroyQueue(q0);
  vt::DestroyQueue(q1);
  CHECK(vt::cuda::WorkspaceEntryCountForTesting() == before);
#endif
}
