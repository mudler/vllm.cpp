// BACKEND-ROCM-QUANT-GATHER (#3093): selected-row decoding and error contracts.
// Primary: plugin d4c1f0d, tests/test_kernels.py::test_gguf_embedding, 102-124.
// Its 2-D IDs flatten to the shared Embedding ABI's contiguous 1-D ID vector.
// Runtime primary and secondary captures remain separate from scalar controls.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "capi/rocm_quant_gather_fixture.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"

#include "iq1_golden_vectors.h"
#include "iq2xs_iq4xs_golden_vectors.h"
#include "iq3s_golden_vectors.h"
#include "iq4nl_q5_0_golden_vectors.h"

#ifdef VLLM_CPP_HIP
#include <hip/hip_runtime.h>
#include "vt/rocm/rocm_embedding_quant.h"
#endif

using vt::DType;
using vt::Tensor;

namespace {

constexpr vt::Device kDevice{vt::DeviceType::kROCM, 0};

class DeviceCase {
 public:
  DeviceCase() : backend(vt::GetBackend(kDevice.type)), queue(backend.CreateQueue()) {}
  ~DeviceCase() {
    for (void* allocation : allocations) backend.Free(allocation);
    backend.DestroyQueue(queue);
  }
  DeviceCase(const DeviceCase&) = delete;
  DeviceCase& operator=(const DeviceCase&) = delete;

  void* Allocate(size_t size) {
    void* pointer = backend.Alloc(std::max(size, size_t{1}));
    allocations.push_back(pointer);
    return pointer;
  }
  Tensor Upload(const void* source, size_t bytes, DType dtype,
                std::initializer_list<int64_t> shape, size_t offset = 0) {
    auto* pointer = static_cast<uint8_t*>(Allocate(bytes + offset)) + offset;
    if (bytes != 0) backend.Copy(queue, pointer, source, bytes);
    return Tensor::Contiguous(pointer, dtype, kDevice, shape);
  }
  Tensor Output(DType dtype, int64_t tokens, int64_t width) {
    const size_t bytes = static_cast<size_t>(tokens * width) * vt::SizeOf(dtype);
    void* pointer = Allocate(bytes);
    backend.Memset(queue, pointer, 0xa5, std::max(bytes, size_t{1}));
    return Tensor::Contiguous(pointer, dtype, kDevice, {tokens, width});
  }
  std::vector<uint8_t> Read(const Tensor& tensor) {
    std::vector<uint8_t> bytes(tensor.Bytes());
    if (!bytes.empty()) backend.Copy(queue, bytes.data(), tensor.data, bytes.size());
    backend.Synchronize(queue);
    return bytes;
  }

  vt::Backend& backend;
  vt::Queue queue;
  std::vector<void*> allocations;
};

std::vector<uint8_t> RandomFiniteBlocks(DType dtype, int64_t rows, int64_t width,
                                      uint32_t seed) {
  const auto decode = vt::cpu::BlockToFloat(dtype);
  REQUIRE(decode != nullptr);
  const auto elements = vt::BlockElems(dtype);
  const auto block_bytes = vt::BlockBytes(dtype);
  std::vector<uint8_t> bytes(static_cast<size_t>(rows) * vt::RowSizeBytes(dtype, width));
  std::vector<float> decoded(static_cast<size_t>(elements));
  std::mt19937 rng(seed);
  for (size_t block = 0; block < bytes.size(); block += static_cast<size_t>(block_bytes)) {
    int attempts = 0;
    do {
      REQUIRE(attempts++ < 200);
      for (int64_t i = 0; i < block_bytes; ++i) {
        bytes[block + static_cast<size_t>(i)] = static_cast<uint8_t>(rng() & 255);
      }
      decode(bytes.data() + block, decoded.data(), elements);
    } while (!std::all_of(decoded.begin(), decoded.end(), [](float v) { return std::isfinite(v); }));
  }
  if (const char* directory = std::getenv("VT_ROCM_GATHER_TEST_CORPUS")) {
    std::filesystem::create_directories(directory);
    const auto name = std::string(vt::Name(dtype)) + "-" + std::to_string(width) +
                      "-seed-" + std::to_string(seed) + ".packed";
    std::ofstream file(std::filesystem::path(directory) / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(file.good());
  }
  return bytes;
}

std::vector<uint8_t> Expected(DType dtype, const uint8_t* packed, int64_t width,
                            const std::vector<int64_t>& ids, DType output_dtype,
                            const uint32_t* golden = nullptr) {
  const size_t row_bytes = vt::RowSizeBytes(dtype, width);
  const auto decode = vt::cpu::BlockToFloat(dtype);
  REQUIRE(decode != nullptr);
  std::vector<uint8_t> expected(ids.size() * static_cast<size_t>(width) * vt::SizeOf(output_dtype));
  std::vector<float> row(static_cast<size_t>(width));
  for (size_t token = 0; token < ids.size(); ++token) {
    if (golden == nullptr) decode(packed + static_cast<size_t>(ids[token]) * row_bytes, row.data(), width);
    else std::memcpy(row.data(), golden + ids[token] * width, row.size() * sizeof(float));
    for (size_t element = 0; element < row.size(); ++element) {
      const size_t index = token * row.size() + element;
      if (output_dtype == DType::kF32) {
        std::memcpy(expected.data() + index * sizeof(float), &row[element], sizeof(float));
      } else {
        const uint16_t word = vt::F32ToBF16(row[element]);
        std::memcpy(expected.data() + index * sizeof(uint16_t), &word, sizeof(uint16_t));
      }
    }
  }
  return expected;
}

void CheckGather(DType dtype, const uint8_t* bytes, int64_t rows, int64_t width,
                 const std::vector<int64_t>& ids, DType id_dtype,
                 DType output_dtype, size_t offset, const uint32_t* golden = nullptr) {
  CAPTURE(std::string(vt::Name(dtype)));
  CAPTURE(width);
  CAPTURE(offset);
  CAPTURE(std::string(vt::Name(id_dtype)));
  CAPTURE(std::string(vt::Name(output_dtype)));
  DeviceCase device;
  auto table = device.Upload(bytes, static_cast<size_t>(rows) * vt::RowSizeBytes(dtype, width),
                             dtype, {rows, width}, offset);
  const std::vector<int32_t> ids32(ids.begin(), ids.end());
  auto indices = id_dtype == DType::kI32
      ? device.Upload(ids32.data(), ids32.size() * sizeof(int32_t), id_dtype,
                       {static_cast<int64_t>(ids.size())})
      : device.Upload(ids.data(), ids.size() * sizeof(int64_t), id_dtype,
                       {static_cast<int64_t>(ids.size())});
  auto output = device.Output(output_dtype, static_cast<int64_t>(ids.size()), width);
  vt::Embedding(device.queue, output, table, indices);
  const auto actual = device.Read(output);
  const auto expected = Expected(dtype, bytes, width, ids, output_dtype, golden);
  REQUIRE(actual.size() == expected.size());
  const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
  CAPTURE(static_cast<size_t>(mismatch.first - actual.begin()));
  CHECK(mismatch.first == actual.end());
}

}  // namespace

TEST_CASE("ROCm gather decoder set covers every admitted dtype through kIQ3_S") {
  std::set<DType> named;
  for (const auto& format : rocm_gather_test::kFormats) named.insert(format.dtype);
  CHECK(named.size() == 19);
  for (int value = 0; value <= static_cast<int>(DType::kIQ3_S); ++value) {
    const auto dtype = static_cast<DType>(value);
    CAPTURE(value);
    const bool decodable = vt::cpu::BlockToFloat(dtype) != nullptr;
    CHECK((named.count(dtype) != 0) == decodable);
#ifdef VLLM_CPP_HIP
    CHECK(vt::rocm::EmbeddingQuantSupported(dtype) == decodable);
#endif
  }
}

TEST_CASE("ROCm gather matches scalar bytes for every codec shape dtype and byte offset") {
  uint32_t seed = 0x524f434d;
  for (const auto& format : rocm_gather_test::kFormats) {
    const int64_t block = vt::BlockElems(format.dtype);
    const std::set<int64_t> widths = {block, 3 * block, 5 * block, 256, 1024};
    for (int64_t width : widths) {
      const uint32_t case_seed = seed++;
      CAPTURE(case_seed);
      const auto packed = RandomFiniteBlocks(format.dtype, 17, width, case_seed);
      const std::vector<int64_t> ids = {16, 8, 0, 8, 16, 4, 3, 2, 1, 0};
      for (auto id_dtype : {DType::kI32, DType::kI64}) {
        for (auto output : {DType::kF32, DType::kBF16}) {
          for (size_t offset = 0; offset < 4; ++offset) {
            CheckGather(format.dtype, packed.data(), 17, width, ids, id_dtype, output, offset);
          }
        }
      }
    }
  }
}

TEST_CASE("ROCm gather matches committed independent decoder vectors") {
  using namespace vllm_test;
  struct Golden { DType dtype; const uint8_t* packed; int64_t rows; int64_t width; const uint32_t* bits; };
  const Golden cases[] = {
      {DType::kIQ4_NL, kIq4nlGoldenBlocks, 2, 160, kIq4nlGoldenBits},
      {DType::kQ5_0, kQ50GoldenBlocks, 2, 64, kQ50GoldenBits},
      {DType::kIQ1_S, kIq1sGoldenBlocks, 4, 256, kIq1sGoldenBits},
      {DType::kIQ1_XXXS, kIq1xxxsGoldenBlocks, 4, 256, kIq1xxxsGoldenBits},
      {DType::kIQ2_XS, kIq2xsGoldenBlocks, 4, 256, kIq2xsGoldenBits},
      {DType::kIQ4_XS, kIq4xsGoldenBlocks, 4, 256, kIq4xsGoldenBits},
      {DType::kIQ3_S, kIq3sGoldenBlocks, 4, 256, kIq3sGoldenBits},
  };
  for (const auto& golden : cases) {
    for (auto output : {DType::kF32, DType::kBF16}) {
      for (auto id_dtype : {DType::kI32, DType::kI64}) {
        CheckGather(golden.dtype, golden.packed, golden.rows, golden.width,
                    {golden.rows - 1, 0, golden.rows / 2, 0}, id_dtype, output, 3, golden.bits);
      }
    }
  }
}

TEST_CASE("ROCm gather crosses launch blocks and the capped grid") {
  for (const auto& format : rocm_gather_test::kFormats) {
    const auto packed = rocm_gather_test::PackedTable(format, 3, 256);
    std::vector<int64_t> ids(257);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int64_t>(i % 3);
    CheckGather(format.dtype, reinterpret_cast<const uint8_t*>(packed.data()), 3, 256,
                ids, DType::kI64, DType::kBF16, 1);
  }
  const auto format = rocm_gather_test::kFormats[0];
  const auto packed = rocm_gather_test::PackedTable(format, 3, 32);
  std::vector<int64_t> ids(1024 * 128 + 3);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int64_t>(i % 3);
  for (auto output : {DType::kF32, DType::kBF16}) {
    CheckGather(format.dtype, reinterpret_cast<const uint8_t*>(packed.data()), 3, 32,
                ids, DType::kI32, output, 3);
  }
}

TEST_CASE("ROCm gather rounds bf16 ties to even including signed ties") {
  // Q8_K stores an f32 scale. A payload of +1 or -1 reproduces these exact
  // halfway values, independently of half-scale precision in other codecs.
  const uint32_t ties[] = {0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000};
  std::vector<uint8_t> packed(4 * 292, 1);
  for (size_t i = 0; i < 4; ++i) std::memcpy(packed.data() + i * 292, &ties[i], 4);
  for (auto id_dtype : {DType::kI32, DType::kI64}) {
    CheckGather(DType::kQ8_K, packed.data(), 4, 256, {0, 1, 2, 3}, id_dtype, DType::kBF16, 1);
  }
}

TEST_CASE("ROCm gather rejects full-width invalid IDs and recovers immediately") {
  for (auto id_dtype : {DType::kI32, DType::kI64}) {
    for (int64_t invalid : {int64_t{-1}, int64_t{3}, int64_t{4294967297},
                            std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
      if (id_dtype == DType::kI32 && (invalid < INT32_MIN || invalid > INT32_MAX)) continue;
      CAPTURE(invalid);
      DeviceCase device;
      const auto format = rocm_gather_test::kFormats[0];
      const auto packed = rocm_gather_test::PackedTable(format, 3, 32);
      auto table = device.Upload(packed.data(), packed.size(), format.dtype, {3, 32}, 3);
      const int32_t narrow = static_cast<int32_t>(invalid);
      auto ids = device.Upload(id_dtype == DType::kI32 ? static_cast<const void*>(&narrow) : &invalid,
                                vt::SizeOf(id_dtype), id_dtype, {1});
      auto output = device.Output(DType::kBF16, 1, 32);
      CHECK_THROWS_WITH_AS(vt::Embedding(device.queue, output, table, ids),
                           doctest::Contains("id " + std::to_string(invalid) + " out of range"),
                           std::runtime_error);
      const int64_t valid = 1;
      device.backend.Copy(device.queue, ids.data, &valid, vt::SizeOf(id_dtype));
      CHECK_NOTHROW(vt::Embedding(device.queue, output, table, ids));
      CHECK(device.Read(output) == Expected(format.dtype,
            reinterpret_cast<const uint8_t*>(packed.data()), 32, {1}, DType::kBF16));
    }
  }
}

TEST_CASE("ROCm gather accepts empty requests and rejects nonempty empty vocabulary") {
  DeviceCase device;
  // Contiguous constructs positive shapes. Empty non-owning views use the
  // public tensor metadata, as the existing shared Embedding tests do.
  auto table = Tensor::Contiguous(nullptr, DType::kQ4_0, kDevice, {1, 32});
  auto ids = Tensor::Contiguous(nullptr, DType::kI32, kDevice, {1});
  auto output = Tensor::Contiguous(nullptr, DType::kBF16, kDevice, {1, 32});
  table.shape[0] = 0;
  ids.shape[0] = 0;
  output.shape[0] = 0;
  CHECK_NOTHROW(vt::Embedding(device.queue, output, table, ids));
  const int32_t id = 0;
  ids = device.Upload(&id, sizeof(id), DType::kI32, {1});
  output = device.Output(DType::kBF16, 1, 32);
  CHECK_THROWS_WITH_AS(vt::Embedding(device.queue, output, table, ids),
                       doctest::Contains("empty table"), std::runtime_error);
}

TEST_CASE("ROCm gather preserves shared rank shape dtype contiguity and device checks") {
  DeviceCase device;
  const int32_t id = 0;
  auto table = Tensor::Contiguous(nullptr, DType::kQ4_0, kDevice, {3, 32});
  auto ids = device.Upload(&id, sizeof(id), DType::kI32, {1});
  auto output = device.Output(DType::kBF16, 1, 32);
  const auto refused = [&](const Tensor& t, const Tensor& i, Tensor o, const char* message) {
    CHECK_THROWS_WITH_AS(vt::Embedding(device.queue, o, t, i), doctest::Contains(message), std::runtime_error);
  };
  auto bad = table; bad.rank = 1; refused(bad, ids, output, "bad ranks");
  // Stock GET_ROWS batched tables exceed the shared rank-two table ABI.
  bad = table; bad.rank = 3; refused(bad, ids, output, "bad ranks");
  bad = table; bad.rank = 4; refused(bad, ids, output, "bad ranks");
  bad = ids; bad.rank = 2; refused(table, bad, output, "bad ranks");
  bad = output; bad.rank = 1; refused(table, ids, bad, "bad ranks");
  bad = output; bad.shape[1] = 31; refused(table, ids, bad, "output shape mismatch");
  bad = output; bad.shape[0] = 2; refused(table, ids, bad, "output shape mismatch");
  bad = ids; bad.dtype = DType::kF32; refused(table, bad, output, "ids i32/i64");
  bad = table; bad.dtype = DType::kI8; refused(bad, ids, output, "table must be float or block-quantized");
  bad = output; bad.dtype = DType::kF16; refused(table, ids, bad, "f32/bf16 out");
  bad = table; bad.shape[1] = 31;
  auto ragged = output; ragged.shape[1] = 31;
  refused(bad, ids, ragged, "block table K must be a whole number of blocks");
  bad = table; bad.stride[1] = 2; refused(bad, ids, output, "contiguous required");
  bad = ids; bad.shape[0] = 2; bad.stride[0] = 2;
  auto two = output; two.shape[0] = 2;
  refused(table, bad, two, "contiguous required");
  bad = output; bad.stride[1] = 2; refused(table, ids, bad, "contiguous required");
  bad = table; bad.device.type = vt::DeviceType::kCPU; refused(bad, ids, output, "device mismatch");
  bad = ids; bad.device.type = vt::DeviceType::kCPU; refused(table, bad, output, "device mismatch");
  bad = output; bad.device.type = vt::DeviceType::kCPU; refused(table, ids, bad, "device mismatch");
}

TEST_CASE("ROCm gather follows the queue device when the ambient device differs") {
#ifdef VLLM_CPP_HIP
  int count = 0;
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
  if (std::getenv("VT_ROCM_GATHER_REQUIRE_TWO_DEVICES") != nullptr) REQUIRE(count >= 2);
  if (count < 2) {
    MESSAGE("The separate ambient-device gate requires two visible ROCm devices");
    return;
  }
  REQUIRE(hipSetDevice(0) == hipSuccess);
  DeviceCase device;
  struct CleanupOnQueueDevice {
    ~CleanupOnQueueDevice() { (void)hipSetDevice(0); }
  } cleanup;
  const auto format = rocm_gather_test::kFormats[0];
  const auto packed = rocm_gather_test::PackedTable(format, 3, 32);
  auto table = device.Upload(packed.data(), packed.size(), format.dtype, {3, 32});
  int64_t id = 2;
  auto ids = device.Upload(&id, sizeof(id), DType::kI64, {1});
  auto output = device.Output(DType::kBF16, 1, 32);
  device.backend.Synchronize(device.queue);
  REQUIRE(hipSetDevice(1) == hipSuccess);
  CHECK_NOTHROW(vt::Embedding(device.queue, output, table, ids));
  REQUIRE(hipSetDevice(0) == hipSuccess);
  CHECK(device.Read(output) == Expected(format.dtype,
        reinterpret_cast<const uint8_t*>(packed.data()), 32, {2}, DType::kBF16));
  id = -1;
  device.backend.Copy(device.queue, ids.data, &id, sizeof(id));
  device.backend.Synchronize(device.queue);
  REQUIRE(hipSetDevice(1) == hipSuccess);
  CHECK_THROWS_WITH_AS(vt::Embedding(device.queue, output, table, ids),
                       doctest::Contains("id -1 out of range"), std::runtime_error);
  REQUIRE(hipSetDevice(0) == hipSuccess);
  id = 1;
  device.backend.Copy(device.queue, ids.data, &id, sizeof(id));
  device.backend.Synchronize(device.queue);
  REQUIRE(hipSetDevice(1) == hipSuccess);
  CHECK_NOTHROW(vt::Embedding(device.queue, output, table, ids));
  REQUIRE(hipSetDevice(0) == hipSuccess);
  CHECK(device.Read(output) == Expected(format.dtype,
        reinterpret_cast<const uint8_t*>(packed.data()), 32, {1}, DType::kBF16));
#endif
}

int main(int argc, char** argv) {
#ifdef VLLM_CPP_HIP
  try { vt::GetBackend(kDevice.type); }
  catch (const std::runtime_error& error) {
    std::printf("SKIPPED: ROCm device unavailable: %s\n", error.what());
    return 77;
  }
  doctest::Context context(argc, argv);
  return context.run();
#else
  (void)argc;
  (void)argv;
  std::puts("SKIPPED: native ROCm gather requires a HIP build");
  return 77;
#endif
}
