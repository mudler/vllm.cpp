// BACKEND-ROCM-ATTN-GATE-SPLIT (#3106).
// Split preparation: vLLM e126687a9a qwen3_next.py:424-433 and
// tests/kernels/test_fused_qk_norm_rope_gate.py:49-67. The fixture retains
// seed 13, BF16, Dh=256, T={1,4,37}, and Hq/Hkv={24/4,16/2}.
// F32 gates are the existing local SigmoidGateBf16 storage adaptation.
#include <doctest/doctest.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/ops.h"

namespace {
using vt::DType;
using vt::Tensor;
constexpr size_t kGuard = 64;

void Hip(hipError_t error) {
  if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
int DeviceCount() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    std::cerr << "SKIP: native split execution requires a ROCm device\n";
    std::exit(77);
  }
  return count;
}
struct DeviceRun {
  int device;
  hipStream_t stream = nullptr;
  vt::Queue queue;
  std::vector<void*> device_buffers, host_buffers;
  explicit DeviceRun(int index = 0) : device(index) {
    REQUIRE(DeviceCount() > device);
    Hip(hipSetDevice(device));
    // Raw HIP allocations avoid depending on the separately owed backend
    // allocation-device policy. Every allocation has an independent witness.
    Hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    queue = vt::Queue{vt::Device{vt::DeviceType::kROCM, device}, stream};
  }
  ~DeviceRun() {
    CHECK(hipSetDevice(device) == hipSuccess);
    CHECK(hipStreamSynchronize(stream) == hipSuccess);
    for (void* p : device_buffers) CHECK(hipFree(p) == hipSuccess);
    for (void* p : host_buffers) CHECK(hipHostFree(p) == hipSuccess);
    CHECK(hipStreamDestroy(stream) == hipSuccess);
  }
  void* DeviceBuffer(size_t bytes) {
    void* p = nullptr;
    Hip(hipMalloc(&p, std::max<size_t>(bytes, 1)));
    device_buffers.push_back(p);
    hipPointerAttribute_t attributes{};
    Hip(hipPointerGetAttributes(&attributes, p));
    REQUIRE(attributes.device == device);
    REQUIRE(attributes.type == hipMemoryTypeDevice);
    return p;
  }
  uint8_t* HostBuffer(size_t bytes) {
    void* p = nullptr;
    Hip(hipHostMalloc(&p, std::max<size_t>(bytes, 1)));
    host_buffers.push_back(p);
    return static_cast<uint8_t*>(p);
  }
  Tensor Make(DType dtype, std::initializer_list<int64_t> shape, size_t extra = 0) {
    // Tensor::Contiguous requires positive dimensions. The split wrapper also
    // accepts an empty token axis, so construct that packed metadata directly.
    Tensor tensor;
    tensor.dtype = dtype;
    tensor.device = queue.device;
    tensor.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    auto dimension = shape.end();
    for (int i = tensor.rank - 1; i >= 0; --i) {
      tensor.shape[i] = *--dimension;
      tensor.stride[i] = stride;
      stride *= tensor.shape[i];
    }
    tensor.data = DeviceBuffer(tensor.Bytes() + extra);
    return tensor;
  }
};

std::vector<uint8_t> Encode(const std::vector<float>& values, DType dtype) {
  std::vector<uint8_t> bytes(values.size() * vt::SizeOf(dtype));
  for (size_t i = 0; i < values.size(); ++i) {
    if (dtype == DType::kF32) {
      std::memcpy(bytes.data() + i * 4, &values[i], 4);
    } else {
      const uint16_t word = vt::F32ToBF16(values[i]);
      std::memcpy(bytes.data() + i * 2, &word, 2);
    }
  }
  return bytes;
}
std::vector<float> Widen(const std::vector<uint8_t>& bytes) {
  REQUIRE(bytes.size() % 2 == 0);
  std::vector<float> out(bytes.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    uint16_t word = 0;
    std::memcpy(&word, bytes.data() + i * 2, 2);
    out[i] = vt::BF16ToF32(word);
  }
  return out;
}

void RunSplit(DeviceRun& run, int64_t tokens, int64_t heads, int64_t width,
              DType input_dtype, DType query_dtype, const std::vector<uint8_t>& input,
              const std::vector<uint8_t>& expected_query,
              const std::vector<uint8_t>& expected_gate) {
  CAPTURE(tokens); CAPTURE(heads); CAPTURE(width);
  CAPTURE(vt::Name(input_dtype)); CAPTURE(vt::Name(query_dtype));
  auto packed = run.Make(input_dtype, {tokens, heads * 2 * width});
  auto query = run.Make(query_dtype, {tokens, heads, width}, kGuard);
  auto gate = run.Make(DType::kF32, {tokens, heads, width}, kGuard);
  REQUIRE(input.size() == packed.Bytes());
  REQUIRE(expected_query.size() == query.Bytes());
  REQUIRE(expected_gate.size() == gate.Bytes());
  CHECK(query.Bytes() == static_cast<size_t>(tokens * heads * width) * vt::SizeOf(query_dtype));
  auto* source = run.HostBuffer(input.size());
  auto* after_input = run.HostBuffer(input.size());
  auto* actual_query = run.HostBuffer(query.Bytes() + kGuard);
  auto* actual_gate = run.HostBuffer(gate.Bytes() + kGuard);
  if (!input.empty()) std::memcpy(source, input.data(), input.size());
  Hip(hipMemsetAsync(query.data, 0xa5, query.Bytes() + kGuard, run.stream));
  Hip(hipMemsetAsync(gate.data, 0xa5, gate.Bytes() + kGuard, run.stream));
  Hip(hipStreamSynchronize(run.stream));
  vt::EnableOpProviderCallStats(true);
  const auto hits = vt::GetReferenceTierHits();
  // Resolve before capture so the missing-provider red leaves no live capture.
  (void)vt::GetOp(vt::OpId::kAttnGateSplit, run.queue.device.type);
  CHECK(std::string(vt::GetOpProviderStats(vt::OpId::kAttnGateSplit,
                                         run.queue.device.type).last_selected) == "vt-native");
  Hip(hipStreamBeginCapture(run.stream, hipStreamCaptureModeGlobal));
  if (!input.empty())
    Hip(hipMemcpyAsync(packed.data, source, input.size(), hipMemcpyHostToDevice, run.stream));
  vt::AttnGateSplit(run.queue, query, gate, packed);
  Hip(hipMemcpyAsync(actual_query, query.data, query.Bytes() + kGuard,
                     hipMemcpyDeviceToHost, run.stream));
  Hip(hipMemcpyAsync(actual_gate, gate.data, gate.Bytes() + kGuard,
                     hipMemcpyDeviceToHost, run.stream));
  if (!input.empty())
    Hip(hipMemcpyAsync(after_input, packed.data, input.size(), hipMemcpyDeviceToHost, run.stream));
  hipGraph_t graph = nullptr;
  Hip(hipStreamEndCapture(run.stream, &graph));
  size_t count = 0;
  Hip(hipGraphGetNodes(graph, nullptr, &count));
  std::vector<hipGraphNode_t> nodes(count);
  Hip(hipGraphGetNodes(graph, nodes.data(), &count));
  size_t kernels = 0;
  for (auto node : nodes) {
    hipGraphNodeType type{};
    Hip(hipGraphNodeGetType(node, &type));
    if (type == hipGraphNodeTypeKernel) ++kernels;
  }
  CHECK(kernels == (tokens == 0 ? 0 : 1));
  hipGraphExec_t executable = nullptr;
  Hip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  Hip(hipGraphLaunch(executable, run.stream));
  Hip(hipStreamSynchronize(run.stream));
  Hip(hipGraphExecDestroy(executable));
  Hip(hipGraphDestroy(graph));
  CHECK(std::equal(expected_query.begin(), expected_query.end(), actual_query));
  CHECK(std::equal(expected_gate.begin(), expected_gate.end(), actual_gate));
  CHECK(std::equal(input.begin(), input.end(), after_input));
  CHECK(std::all_of(actual_query + query.Bytes(), actual_query + query.Bytes() + kGuard,
                     [](uint8_t value) { return value == 0xa5; }));
  CHECK(std::all_of(actual_gate + gate.Bytes(), actual_gate + gate.Bytes() + kGuard,
                     [](uint8_t value) { return value == 0xa5; }));
  CHECK(vt::GetReferenceTierHits() == hits);
}

void SentinelCase(DeviceRun& run, int64_t tokens, int64_t heads, int64_t width,
                  DType input_dtype, DType query_dtype) {
  std::vector<float> input, query, gate;
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t side = 0; side < 2; ++side) {
        for (int64_t d = 0; d < width; ++d) {
          const float value = static_cast<float>(100 * t + 10 * h + 5 * side + d);
          input.push_back(value);
          (side == 0 ? query : gate).push_back(value);
        }
      }
    }
  }
  const auto encoded = Encode(input, input_dtype);
  if (input_dtype == DType::kBF16) {
    query = Widen(Encode(query, DType::kBF16));
    gate = Widen(Encode(gate, DType::kBF16));
  }
  RunSplit(run, tokens, heads, width, input_dtype, query_dtype, encoded,
           Encode(query, query_dtype), Encode(gate, DType::kF32));
}

TEST_CASE("ROCm split smallest native case reaches the shared entry point") {
  DeviceRun run;
  auto input = run.Make(DType::kF32, {2, 8});
  auto query = run.Make(DType::kF32, {2, 2, 2});
  auto gate = run.Make(DType::kF32, {2, 2, 2});
  const std::array<float, 16> data{0, 1, 5, 6, 10, 11, 15, 16,
                                   100, 101, 105, 106, 110, 111, 115, 116};
  Hip(hipMemcpyAsync(input.data, data.data(), sizeof(data), hipMemcpyHostToDevice, run.stream));
  vt::AttnGateSplit(run.queue, query, gate, input);
  std::array<float, 8> actual_query{}, actual_gate{};
  Hip(hipMemcpyAsync(actual_query.data(), query.data, sizeof(actual_query),
                     hipMemcpyDeviceToHost, run.stream));
  Hip(hipMemcpyAsync(actual_gate.data(), gate.data, sizeof(actual_gate),
                     hipMemcpyDeviceToHost, run.stream));
  Hip(hipStreamSynchronize(run.stream));
  CHECK(actual_query == std::array<float, 8>{0, 1, 10, 11, 100, 101, 110, 111});
  CHECK(actual_gate == std::array<float, 8>{5, 6, 15, 16, 105, 106, 115, 116});
}

TEST_CASE("ROCm split native per-head sentinels cover all dtype pairs and tails") {
  DeviceRun run;
  for (auto input : {DType::kF32, DType::kBF16}) {
    for (auto query : {DType::kF32, DType::kBF16}) {
      SentinelCase(run, 2, 2, 2, input, query);
      SentinelCase(run, 3, 3, 5, input, query);
      SentinelCase(run, 0, 3, 5, input, query);
    }
  }
}

TEST_CASE("ROCm split query narrowing uses BF16 nearest even") {
  DeviceRun run;
  // Tie below an even mantissa, tie below an odd mantissa, and both neighbors.
  // Exact BF16 words make truncation and round-away mutations observable.
  const std::array<uint32_t, 8> bits = {
      0x3f808000u, 0x3f818000u, 0x3f807fffu, 0x3f808001u,
      0xbf808000u, 0xbf818000u, 0xbf807fffu, 0xbf808001u};
  const std::array<uint16_t, 8> words = {
      0x3f80, 0x3f82, 0x3f80, 0x3f81, 0xbf80, 0xbf82, 0xbf80, 0xbf81};
  std::vector<float> values(16), gates(8);
  for (size_t i = 0; i < 8; ++i) {
    std::memcpy(&values[i], &bits[i], 4);
    gates[i] = values[8 + i] = static_cast<float>(i) + 0.00390625f;
  }
  std::vector<uint8_t> expected(sizeof(words));
  std::memcpy(expected.data(), words.data(), expected.size());
  RunSplit(run, 1, 1, 8, DType::kF32, DType::kBF16, Encode(values, DType::kF32),
           expected, Encode(gates, DType::kF32));
}

TEST_CASE("ROCm split preserves active pinned upstream preparation bytes") {
  DeviceRun run;
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path() /
                    "fixtures/rocm_attn_gate_split";
  std::ifstream metadata(root / "manifest.json");
  REQUIRE(metadata.good());
  const auto manifest = nlohmann::json::parse(metadata);
  REQUIRE(manifest.at("upstream_revision") == "e126687a9a828d513c01a07cd69f025f27d63280");
  REQUIRE(manifest.at("seed") == 13);
  REQUIRE(manifest.at("dtype") == "bfloat16");
  REQUIRE(manifest.at("cases").size() == 6);
  std::ifstream file(root / "cases.bin", std::ios::binary);
  REQUIRE(file.good());
  const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
  REQUIRE(bytes.size() == manifest.at("data_bytes").get<size_t>());
  for (const auto& item : manifest.at("cases")) {
    const auto slice = [&](const char* name) {
      const auto offset = item.at(name).at("offset").get<size_t>();
      const auto size = item.at(name).at("bytes").get<size_t>();
      REQUIRE(offset <= bytes.size());
      REQUIRE(size <= bytes.size() - offset);
      return std::vector<uint8_t>(bytes.begin() + offset, bytes.begin() + offset + size);
    };
    const auto input = slice("input"), query = slice("query"), gate = slice("gate");
    const int64_t tokens = item.at("tokens"), heads = item.at("query_heads");
    CHECK((tokens == 1 || tokens == 4 || tokens == 37));
    CHECK((heads == 24 || heads == 16));
    CHECK(item.at("kv_heads") == (heads == 24 ? 4 : 2));
    CHECK(item.at("head_dim") == 256);
    for (auto dtype : {DType::kF32, DType::kBF16})
      RunSplit(run, tokens, heads, 256, DType::kBF16, dtype, input,
               dtype == DType::kBF16 ? query : Encode(Widen(query), dtype),
               Encode(Widen(gate), DType::kF32));
  }
}

TEST_CASE("ROCm split shared wrapper preserves every refusal") {
  DeviceRun run;
  auto input = run.Make(DType::kF32, {2, 8});
  auto query = run.Make(DType::kF32, {2, 2, 2});
  auto gate = run.Make(DType::kF32, {2, 2, 2});
  const char* message = nullptr;
  SUBCASE("query rank") { query.rank = 2; message = "q_out/gate_out rank-3"; }
  SUBCASE("gate rank") { gate.rank = 2; message = "q_out/gate_out rank-3"; }
  SUBCASE("input rank") { input.rank = 3; message = "qgate rank-2"; }
  SUBCASE("gate shape") { gate.shape[1] = 3; message = "gate_out must match"; }
  SUBCASE("input shape") { input.shape[1] = 7; message = "qgate must be"; }
  SUBCASE("F16 query") { query.dtype = DType::kF16; message = "q_out must be f32 or bf16"; }
  SUBCASE("F16 input") { input.dtype = DType::kF16; message = "qgate must be f32 or bf16"; }
  SUBCASE("BF16 gate") { gate.dtype = DType::kBF16; message = "gate_out must be f32"; }
  SUBCASE("strided input") { input.stride[0] += 1; message = "contiguous required"; }
  SUBCASE("strided query") { query.stride[0] += 1; message = "contiguous required"; }
  SUBCASE("strided gate") { gate.stride[0] += 1; message = "contiguous required"; }
  SUBCASE("input device") { input.device.index = 1; message = "device mismatch"; }
  SUBCASE("query device") { query.device.index = 1; message = "device mismatch"; }
  SUBCASE("gate device") { gate.device.index = 1; message = "device mismatch"; }
  SUBCASE("queue device") { run.queue.device.index = 1; message = "device mismatch"; }
  REQUIRE(message != nullptr);
  CHECK_THROWS_WITH_AS(vt::AttnGateSplit(run.queue, query, gate, input),
                       doctest::Contains(message), std::runtime_error);
}

TEST_CASE("ROCm split binds the queue device after an independent current-device change" * doctest::skip()) {
  if (DeviceCount() < 2) {
    std::cerr << "SKIP: queue-device binding requires two visible ROCm devices\n";
    std::exit(77);
  }
  for (int device : {0, 1}) {
    DeviceRun run(device);
    auto input = run.Make(DType::kF32, {2, 8});
    auto query = run.Make(DType::kBF16, {2, 2, 2});
    auto gate = run.Make(DType::kF32, {2, 2, 2});
    const std::array<float, 16> data{0, 1, 5, 6, 10, 11, 15, 16,
                                     100, 101, 105, 106, 110, 111, 115, 116};
    Hip(hipMemcpyAsync(input.data, data.data(), sizeof(data), hipMemcpyHostToDevice, run.stream));
    Hip(hipStreamSynchronize(run.stream));
    Hip(hipSetDevice(1 - device));
    vt::AttnGateSplit(run.queue, query, gate, input);
    int current = -1;
    Hip(hipGetDevice(&current));
    CHECK(current == device);
    Hip(hipSetDevice(device));
    std::array<uint16_t, 8> actual_query{};
    std::array<float, 8> actual_gate{};
    Hip(hipMemcpyAsync(actual_query.data(), query.data, sizeof(actual_query),
                       hipMemcpyDeviceToHost, run.stream));
    Hip(hipMemcpyAsync(actual_gate.data(), gate.data, sizeof(actual_gate),
                       hipMemcpyDeviceToHost, run.stream));
    Hip(hipStreamSynchronize(run.stream));
    for (size_t i = 0; i < 8; ++i) {
      CHECK(actual_query[i] == vt::F32ToBF16(data[(i / 2) * 4 + i % 2]));
      CHECK(actual_gate[i] == data[(i / 2) * 4 + 2 + i % 2]);
    }
  }
}

TEST_CASE("ROCm split indexes past signed 32-bit output and input offsets" * doctest::skip()) {
  DeviceRun run;
  constexpr int64_t width = (int64_t{1} << 31) + 1;
  constexpr size_t need = static_cast<size_t>(width) * 10;
  size_t free = 0, total = 0;
  Hip(hipMemGetInfo(&free, &total));
  if (free < need + (size_t{1} << 30)) {
    std::cerr << "SKIP: 64-bit split witness needs 20 GiB plus 1 GiB free margin\n";
    std::exit(77);
  }
  auto input = run.Make(DType::kBF16, {1, 2 * width});
  auto query = run.Make(DType::kBF16, {1, 1, width});
  auto gate = run.Make(DType::kF32, {1, 1, width});
  Hip(hipMemsetAsync(input.data, 0, input.Bytes(), run.stream));
  Hip(hipMemsetAsync(query.data, 0xa5, query.Bytes(), run.stream));
  Hip(hipMemsetAsync(gate.data, 0xa5, gate.Bytes(), run.stream));
  const std::array<int64_t, 4> indices{0, width - 1, width, 2 * width - 1};
  const std::array<uint16_t, 4> words{0x3f81, 0x3f83, 0xbf84, 0xc001};
  for (size_t i = 0; i < indices.size(); ++i)
    Hip(hipMemcpyAsync(input.Ptr<uint16_t>() + indices[i], &words[i], 2,
                       hipMemcpyHostToDevice, run.stream));
  vt::AttnGateSplit(run.queue, query, gate, input);
  std::array<uint16_t, 3> actual_query{};
  std::array<float, 3> actual_gate{};
  const std::array<int64_t, 3> samples{0, width / 2, width - 1};
  for (size_t i = 0; i < samples.size(); ++i) {
    Hip(hipMemcpyAsync(&actual_query[i], query.Ptr<uint16_t>() + samples[i], 2,
                       hipMemcpyDeviceToHost, run.stream));
    Hip(hipMemcpyAsync(&actual_gate[i], gate.Ptr<float>() + samples[i], 4,
                       hipMemcpyDeviceToHost, run.stream));
  }
  Hip(hipStreamSynchronize(run.stream));
  CHECK(actual_query == std::array<uint16_t, 3>{words[0], 0, words[1]});
  CHECK(actual_gate == std::array<float, 3>{vt::BF16ToF32(words[2]), 0.f, vt::BF16ToF32(words[3])});
}
}  // namespace
