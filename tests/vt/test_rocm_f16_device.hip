// BACKEND-ROCM-F16-WEIGHTS (#3092): ordinary operations bind their queue's
// device even when the calling thread has another current device. Allocate
// explicitly with HIP so this test does not depend on the backend's separate
// resource-creation defect, tracked by #3100.
#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/ops.h"
#include "vt/backend.h"
#include "rocm_f16_alloc_observer.h"
#include "../../src/vt/rocm/rocm_f16_conversion.h"

namespace {
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void Hip(hipError_t status, const char* message) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(message) + ": " + hipGetErrorString(status));
}
struct Resources {
  int device;
  hipStream_t stream = nullptr;
  vt::Queue queue;
  std::vector<void*> allocations;
  explicit Resources(int index) : device(index) {
    Hip(hipSetDevice(device), "bind resource device");
    Hip(hipStreamCreate(&stream), "create intended-device stream");
    queue.device = {vt::DeviceType::kROCM, device};
    queue.handle = stream;
  }
  ~Resources() {
    (void)hipSetDevice(device);
    if (queue.handle) vt::DestroyQueue(queue);
    for (void* pointer : allocations) (void)hipFree(pointer);

  }
  void* Allocate(size_t bytes) {
    void* pointer = nullptr;
    Hip(hipSetDevice(device), "bind allocation device");
    Hip(hipMalloc(&pointer, bytes), "allocate operand");
    allocations.push_back(pointer);
    hipPointerAttribute_t attributes{};
    Hip(hipPointerGetAttributes(&attributes, pointer), "query operand provenance");
    Check(attributes.device == device, "operand allocation has the wrong device");
    return pointer;
  }
  template <typename T>
  void Upload(void* destination, const std::vector<T>& values) {
    Hip(hipSetDevice(device), "bind copy device");
    Hip(hipMemcpyAsync(destination, values.data(), values.size() * sizeof(T),
                       hipMemcpyHostToDevice, stream), "upload operand");
    Hip(hipStreamSynchronize(stream), "complete operand upload");
  }
};
float RoundedWeight(float value) {
  return vt::BF16ToF32(vt::F32ToBF16(vt::F16ToF32(vt::F32ToF16(value))));
}
}  // namespace

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count < 2) {
    std::cout << "SKIP: F16 current-device gate requires two visible ROCm devices\n";
    return 77;
  }
  try {
    int operations = 0;
    for (int intended : {0, 1}) {
      for (auto operation : {vt::OpId::kMatmul, vt::OpId::kMatmulBT, vt::OpId::kEmbedding}) {
        Resources resources(intended);
        auto& queue = resources.queue;
        const auto before = vt::rocm::GetF16ScratchStats();
        std::vector<float> weights{1.0009765625f, -1.0029296875f, .5009765625f,
                                   2.005859375f, -.2509765625f, 3.00390625f};
        std::vector<uint16_t> words;
        for (float value : weights) words.push_back(vt::F32ToF16(value));
        const bool bt = operation == vt::OpId::kMatmulBT;
        auto b = vt::Tensor::Contiguous(resources.Allocate(words.size() * 2), vt::DType::kF16,
            queue.device, {bt ? 3 : 2, bt ? 2 : 3});
        b.weight_value_dtype = vt::DType::kBF16;
        resources.Upload(b.data, words);
        std::vector<float> expected;
        vt::Tensor a;
        if (operation == vt::OpId::kEmbedding) {
          a = vt::Tensor::Contiguous(resources.Allocate(2 * sizeof(int64_t)), vt::DType::kI64,
                                    queue.device, {2});
          resources.Upload(a.data, std::vector<int64_t>{1, 0});
          for (int row : {1, 0})
            for (int column = 0; column < 3; ++column)
              expected.push_back(RoundedWeight(weights[static_cast<size_t>(row * 3 + column)]));
        } else {
          // F32 values outside F16's range expose a wrong-device conversion
          // path without allowing an F16 activation cast to hide the defect.
          a = vt::Tensor::Contiguous(resources.Allocate(4 * sizeof(float)), vt::DType::kF32,
                                    queue.device, {2, 2});
          const std::vector<float> activations{131072.f, 2.f, std::ldexp(1.f, -30), 0.f};
          resources.Upload(a.data, activations);
          for (int row = 0; row < 2; ++row)
            for (int column = 0; column < 3; ++column) {
              float sum = 0.f;
              for (int inner = 0; inner < 2; ++inner)
                sum += activations[static_cast<size_t>(row * 2 + inner)] *
                       RoundedWeight(weights[static_cast<size_t>(bt ? column * 2 + inner : inner * 3 + column)]);
              expected.push_back(sum);
            }
        }
        auto output = vt::Tensor::Contiguous(resources.Allocate(6 * sizeof(float)), vt::DType::kF32,
                                             queue.device, {2, 3});
        // Nothing between this call and the operation binds the intended GPU.
        Hip(hipSetDevice(1 - intended), "select opposite ambient device");
        if (operation == vt::OpId::kMatmul) vt::Matmul(queue, output, a, b);
        else if (operation == vt::OpId::kMatmulBT) vt::MatmulBT(queue, output, a, b);
        else vt::Embedding(queue, output, b, a);
        int actual_device = -1;
        Hip(hipGetDevice(&actual_device), "query operation device");
        Check(actual_device == intended, "operation did not bind its queue device");
        Hip(hipStreamSynchronize(resources.stream), "complete operation");
        std::vector<float> actual(6);
        Hip(hipMemcpyAsync(actual.data(), output.data, actual.size() * sizeof(float),
                           hipMemcpyDeviceToHost, resources.stream), "download result");
        Hip(hipStreamSynchronize(resources.stream), "complete result download");
        for (size_t index = 0; index < actual.size(); ++index)
          Check(actual[index] == expected[index], "operation used incorrect weight values");
        const auto provider = vt::GetOpProviderStats(operation, vt::DeviceType::kROCM);
        Check(provider.last_selected && std::string(provider.last_selected) == vt::kNativeProviderName,
              "operation did not select the native ROCm provider");
        std::vector<uint64_t> scratch_ids;
        const auto allocated = vt::rocm::GetF16ScratchStats();
        for (const auto pointer : allocated.allocation_pointers) {
          const auto id = rocm_f16_test::AllocationId(pointer);
          Check(id != 0, "allocator observer did not see the F16 allocation");
          scratch_ids.push_back(id);
        }
        if (operation != vt::OpId::kEmbedding)
          Check(!scratch_ids.empty(), "GEMM did not allocate F16 scratch");
        Hip(hipSetDevice(1 - intended), "select opposite device before scratch release");
        vt::DestroyQueue(queue);
        Hip(hipGetDevice(&actual_device), "query device after scratch release");
        Check(actual_device == 1 - intended, "scratch release did not restore ambient device");
        for (const auto id : scratch_ids) {
          Check(rocm_f16_test::WasFreed(id), "queue destruction did not free the HIP allocation");
          Check(rocm_f16_test::FreedOnAllocationDevice(id),
                "scratch was not freed on its allocation device");
        }
        const auto released = vt::rocm::GetF16ScratchStats();
        Check(released.retained_bytes == before.retained_bytes,
              "queue destruction retained F16 scratch");
        ++operations;
      }
    }
    std::cout << "PASS: " << operations << " NN, BT, and embedding invocations bind both queue devices\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
