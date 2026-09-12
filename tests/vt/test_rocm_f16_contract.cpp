// CPU-runnable ownership/dispatch contract for BACKEND-ROCM-F16-WEIGHTS (#3092).
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/ops.h"

namespace {
using vt::DType;
using vt::Device;
using vt::DeviceType;
using vt::Queue;
using vt::Tensor;
using vllm::OwnedTensor;

OwnedTensor Weight() {
  OwnedTensor w;
  w.dtype = DType::kF16;
  w.weight_value_dtype = DType::kBF16;
  w.rank = 2; w.shape[0] = 2; w.shape[1] = 3;
  w.bytes.resize(12);
  auto* words = reinterpret_cast<uint16_t*>(w.bytes.data());
  for (int i = 0; i < 6; ++i) words[i] = vt::F32ToF16(1.0009765625f + i);
  return w;
}
void CheckView(const Tensor& tensor, const void* expected, Device device) {
  CHECK(tensor.dtype == DType::kF16);
  CHECK(tensor.weight_value_dtype == DType::kBF16);
  CHECK(tensor.data == expected);
  CHECK(tensor.device == device);
  CHECK(tensor.Numel() == 6);
  CHECK(tensor.Bytes() == 12);
}
class HostBackend final : public vt::Backend {
 public:
  int allocations = 0;
  void* Alloc(size_t size) override { ++allocations; return std::malloc(size); }
  void Free(void* ptr) override { std::free(ptr); }
  void Memset(Queue&, void* ptr, int value, size_t size) override { std::memset(ptr, value, size); }
  void Copy(Queue&, void* dst, const void* src, size_t size) override { std::memcpy(dst, src, size); }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  void DestroyQueue(Queue&) override {}
  bool UnifiedMemory() const override { return false; }
  bool DeviceMemoryIsHostAddressable() const override { return false; }
};
HostBackend& Backend() { static HostBackend backend; return backend; }
class HostPlatform final : public vllm::platforms::Platform {
 public:
  bool alias = false;
  DeviceType device_type() const override { return DeviceType::kXPU; }
  vt::Backend& backend() const override { return Backend(); }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {0, 0}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
  bool host_memory_is_device_addressable() const override { return alias; }
};
HostPlatform& Platform() { static HostPlatform platform; return platform; }
struct Registration {
  Registration() {
    vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &Backend());
    vllm::platforms::RegisterPlatform(DeviceType::kXPU, &Platform());
  }
} registration;
int provider_calls = 0;
void FakeGemm(Queue&, Tensor&, const Tensor&, const Tensor&) { ++provider_calls; }
constexpr const char* kFakeEmbeddingProvider = "f16-contract-embedding-unsupported";
int embedding_provider_calls = 0;
void FakeEmbedding(Queue&, Tensor&, const Tensor&, const Tensor&) {
  ++embedding_provider_calls;
}
struct EmbeddingProviderRegistration {
  EmbeddingProviderRegistration() {
    vt::RegisterOpProvider(vt::OpId::kEmbedding, DeviceType::kROCM,
                          {kFakeEmbeddingProvider, 10000, nullptr,
                           reinterpret_cast<void*>(&FakeEmbedding)});
    vt::DisableOpProvider(kFakeEmbeddingProvider, true);
  }
} embedding_provider_registration;
struct ScopedEmbeddingProvider {
  const bool was_disabled = vt::OpProviderDisabled(kFakeEmbeddingProvider);
  ScopedEmbeddingProvider() { vt::DisableOpProvider(kFakeEmbeddingProvider, false); }
  ~ScopedEmbeddingProvider() { vt::DisableOpProvider(kFakeEmbeddingProvider, was_disabled); }
};
}  // namespace

TEST_CASE("F16 weight metadata survives ownership, shape views, slices, and aliases") {
  OwnedTensor original = Weight();
  OwnedTensor copy = original;
  OwnedTensor moved = std::move(copy);
  CHECK(moved.bytes.data() != original.bytes.data());
  CheckView(moved.View(), moved.bytes.data(), Device{});
  CHECK(moved.View().View({3, 2}).weight_value_dtype == DType::kBF16);
  Tensor slice = moved.View().Slice(0, 1, 2);
  CHECK(slice.weight_value_dtype == DType::kBF16);
  CHECK(slice.dtype == DType::kF16);
  CHECK(slice.data == moved.bytes.data() + 6);
  OwnedTensor borrow = vllm::BorrowWholeOwnedTensor(original);
  REQUIRE(borrow.bytes.borrowed());
  CHECK(borrow.bytes.data() == original.bytes.data());
  original = OwnedTensor{};
  CheckView(borrow.View(), borrow.bytes.data(), Device{});
  CHECK(reinterpret_cast<const uint16_t*>(borrow.bytes.data())[0] == vt::F32ToF16(1.0009765625f));
}

TEST_CASE("OwnedTensor views preserve empty and zero-extent descriptors") {
  for (int kind : {0, 1, 2}) {
    CAPTURE(kind);
    OwnedTensor value;
    value.dtype = DType::kF32;
    if (kind != 0) {
      value.rank = 2;
      value.shape[0] = kind == 1 ? 0 : 3;
      value.shape[1] = kind == 1 ? 3 : 0;
    }
    const Tensor view = value.View();
    CHECK(value.Numel() == 0);
    CHECK(view.rank == value.rank);
    // A default OwnedTensor has no elements; its legacy rank-zero vt::Tensor
    // descriptor uses Tensor::Numel's empty product of one. Preserve both.
    CHECK(view.Numel() == (kind != 0 ? 0 : 1));
    CHECK(view.Bytes() == (kind != 0 ? 0 : 4));
    CHECK(view.data == value.bytes.data());
    CHECK(view.dtype == value.dtype);
    CHECK(view.device == Device{});
    CHECK(view.IsContiguous());
    if (kind != 0) {
      CHECK(view.shape[0] == value.shape[0]);
      CHECK(view.shape[1] == value.shape[1]);
      CHECK(view.stride[0] == value.shape[1]);
      CHECK(view.stride[1] == 1);
    }
  }
}

TEST_CASE("F16 value metadata travels through ViewOn; layout markers do not") {
  for (int flag : {0, 1, 2, 3}) {
    CAPTURE(flag);
    OwnedTensor original = Weight();
    original.repacked = flag == 1;
    original.q8_0_aligned = flag == 2;
    original.elem_kn_repacked = flag == 3;
    auto check = [&](const Tensor& view) {
      CHECK(view.weight_value_dtype == DType::kBF16);
      CHECK(view.dtype == DType::kF16);
      // ViewOn carries weight_value_dtype but NOT layout markers. Each
      // ResidentWeight arm owns a different marker set and sets them
      // explicitly after the call. See qwen3_5_weights.cpp ViewOn comment.
      CHECK(view.repacked == false);
      CHECK(view.q8_0_aligned == false);
      CHECK(view.elem_kn_repacked == false);
    };
    OwnedTensor copy = original;
    OwnedTensor moved = std::move(copy);
    check(moved.View());
    check(moved.View().View({3, 2}));
    check(moved.ViewOn(moved.bytes.data(), Device{DeviceType::kXPU, 0}, {3, 2}));
    check(moved.View().Slice(0, 1, 2));
    OwnedTensor borrowed = vllm::BorrowWholeOwnedTensor(original);
    check(borrowed.View());
  }
}

TEST_CASE("F16 metadata reaches every shared and Qwen3.5 resident return") {
  Queue cpu{Device{}, nullptr};
  Queue device = Backend().CreateQueue();
  for (bool shared : {false, true}) {
    CAPTURE(shared);
    auto resident = [&](OwnedTensor& w, Queue& q) {
      if (shared) return vllm::dense_attn::ResidentWeight({Backend(), q}, w, {3, 2});
      return vllm::Qwen3_5EmbeddingTable(Backend(), q, w, 2, 3);
    };
    OwnedTensor host = Weight();
    CheckView(resident(host, cpu), host.bytes.data(), cpu.device);
    Platform().alias = false;
    OwnedTensor staged = Weight();
    const auto expected = std::vector<uint8_t>(staged.bytes.data(), staged.bytes.data() + 12);
    const int before = Backend().allocations;
    Tensor first = resident(staged, device);
    REQUIRE(staged.d_dev != nullptr);
    CheckView(first, staged.d_dev.get(), device.device);
    CHECK(std::memcmp(first.data, expected.data(), 12) == 0);
    CheckView(resident(staged, device), staged.d_dev.get(), device.device);
    CHECK(Backend().allocations == before + 1);
    staged.ReleaseHost();
    CHECK(staged.host_released);
    CHECK(staged.dtype == DType::kF16);
    CHECK(staged.weight_value_dtype == DType::kBF16);
    CHECK_THROWS(staged.View());
    CheckView(resident(staged, device), staged.d_dev.get(), device.device);
    CHECK(Backend().allocations == before + 1);
    if (!shared) {
      Platform().alias = true;
      OwnedTensor aliased = Weight();
      Tensor view = resident(aliased, device);
      CHECK(aliased.d_dev == nullptr);
      CheckView(view, aliased.bytes.data(), device.device);
      // The addressable platform must still serve an already released resident.
      CheckView(resident(staged, device), staged.d_dev.get(), device.device);
      Platform().alias = false;
    }
  }
}

TEST_CASE("F16 weight marker rejects illegal operands and unsupported providers before execution") {
  float bytes[12]{};
  Queue q{Device{DeviceType::kROCM, 0}, nullptr};
  Tensor a = Tensor::Contiguous(bytes, DType::kF32, q.device, {2, 3});
  Tensor b = Tensor::Contiguous(bytes, DType::kF16, q.device, {3, 3});
  Tensor out = Tensor::Contiguous(bytes, DType::kF32, q.device, {2, 3});
  for (bool bt : {false, true}) {
    const auto op = bt ? vt::OpId::kMatmulBT : vt::OpId::kMatmul;
    vt::OpProvider fake{"f16-contract-unsupported", 10000, nullptr,
                         reinterpret_cast<void*>(&FakeGemm)};
    vt::RegisterOpProvider(op, q.device.type, fake);
    vt::DisableOpProvider(fake.name, false);
    b.weight_value_dtype = DType::kBF16;
    const int before = provider_calls;
    auto call = [&] { if (bt) vt::MatmulBT(q, out, a, b); else vt::Matmul(q, out, a, b); };
    CHECK_THROWS_WITH(call(), doctest::Contains("native ROCm provider"));
    CHECK(provider_calls == before);
    b.dtype = DType::kBF16;
    CHECK_THROWS_WITH(call(), doctest::Contains("F16 storage"));
    b.dtype = DType::kF16; b.weight_value_dtype = DType::kF16;
    CHECK_THROWS_WITH(call(), doctest::Contains("BF16 or F32 values"));
    b.weight_value_dtype = DType::kBF16;
    a.weight_value_dtype = DType::kBF16;
    CHECK_THROWS_WITH(call(), doctest::Contains("activation, ID, or output"));
    a.weight_value_dtype.reset();
    out.weight_value_dtype = DType::kBF16;
    CHECK_THROWS_WITH(call(), doctest::Contains("activation, ID, or output"));
    out.weight_value_dtype.reset();
    q.device = Device{}; a.device = q.device; b.device = q.device; out.device = q.device;
    CHECK_THROWS_WITH(call(), doctest::Contains("native ROCm provider"));
    q.device = Device{DeviceType::kROCM, 0}; a.device = q.device; b.device = q.device; out.device = q.device;
    vt::DisableOpProvider(fake.name, true);
  }
}

TEST_CASE("F16 marked embedding rejects unsupported providers before execution") {
  uint16_t table_bytes[9]{};
  int32_t id_bytes[2]{0, 2};
  float out_bytes[6]{};
  Queue q{Device{DeviceType::kROCM, 0}, nullptr};
  Tensor table = Tensor::Contiguous(table_bytes, DType::kF16, q.device, {3, 3});
  Tensor ids = Tensor::Contiguous(id_bytes, DType::kI32, q.device, {2});
  Tensor out = Tensor::Contiguous(out_bytes, DType::kF32, q.device, {2, 3});
  ScopedEmbeddingProvider provider;
  const int before = embedding_provider_calls;
  table.weight_value_dtype = DType::kBF16;
  CHECK_THROWS_WITH(vt::Embedding(q, out, table, ids),
                    doctest::Contains("weight_value_dtype requires the native ROCm provider"));
  CHECK(embedding_provider_calls == before);
}
