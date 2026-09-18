// BACKEND-ROCM-F16-WEIGHTS (#3092). Ordinary fallback port anchor:
// vLLM e126687a9a tests/model_executor/layers/test_rocm_unquantized_gemm.py:152.
// Preserve F16 [6,64] x [128,64] and atol=rtol=1e-3. The C++ seam returns
// F32, so compare its F16-rounded values separately from F32 accumulation.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "support/test_env.h"
#include "rocm_f16_alloc_observer.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"
#include "../../src/vt/rocm/rocm_f16_conversion.h"

namespace {
using vt::DType;
using vt::Tensor;

class ScopedF16Env {
 public:
  ScopedF16Env(const char* name, const char* value) : name_(name) {
    const char* old = std::getenv(name);
    had_value_ = old != nullptr;
    if (old != nullptr) old_value_ = old;
    vllm_test::SetEnv(name, value);
  }
  ~ScopedF16Env() {
    if (!had_value_) {
      vllm_test::UnsetEnv(name_);
    } else {
#if defined(_WIN32)
      vllm_test::SetEnv(name_, old_value_);
#else
      // Preserve an empty POSIX value as distinct from an absent variable.
      if (::setenv(name_, old_value_.c_str(), 1) != 0) std::terminate();
#endif
    }
  }
  ScopedF16Env(const ScopedF16Env&) = delete;
  ScopedF16Env& operator=(const ScopedF16Env&) = delete;
 private:
  const char* name_;
  bool had_value_ = false;
  std::string old_value_;
};

struct DeviceRun {
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kROCM);
  vt::Queue queue = backend.CreateQueue();
  std::vector<void*> allocations;
  ~DeviceRun() {
    backend.Synchronize(queue);
    for (void* ptr : allocations) backend.Free(ptr);
    backend.DestroyQueue(queue);
  }
  void* Allocate(size_t bytes) {
    void* ptr = backend.Alloc(std::max<size_t>(bytes, 4));
    allocations.push_back(ptr);
    return ptr;
  }
  Tensor Make(DType dtype, int64_t rows, int64_t columns) {
    Tensor tensor;
    tensor.data = Allocate(static_cast<size_t>(rows * columns) * vt::SizeOf(dtype));
    tensor.device = queue.device;
    tensor.dtype = dtype;
    tensor.rank = 2;
    tensor.shape[0] = rows; tensor.shape[1] = columns;
    tensor.stride[0] = columns; tensor.stride[1] = 1;
    return tensor;
  }
  void Upload(Tensor& tensor, const std::vector<float>& values) {
    if (tensor.dtype == DType::kF32) {
      backend.Copy(queue, tensor.data, values.data(), values.size() * sizeof(float));
    } else {
      std::vector<uint16_t> words(values.size());
      for (size_t i = 0; i < values.size(); ++i)
        words[i] = tensor.dtype == DType::kF16 ? vt::F32ToF16(values[i])
                                              : vt::F32ToBF16(values[i]);
      backend.Copy(queue, tensor.data, words.data(), words.size() * sizeof(uint16_t));
      backend.Synchronize(queue);
    }
    backend.Synchronize(queue);
  }
  std::vector<float> Download(const Tensor& tensor) {
    std::vector<float> out(static_cast<size_t>(tensor.Numel()));
    if (tensor.dtype == DType::kF32) {
      backend.Copy(queue, out.data(), tensor.data, out.size() * sizeof(float));
      backend.Synchronize(queue);
    } else {
      std::vector<uint16_t> words(out.size());
      backend.Copy(queue, words.data(), tensor.data, words.size() * sizeof(uint16_t));
      backend.Synchronize(queue);
      for (size_t i = 0; i < out.size(); ++i) out[i] = vt::BF16ToF32(words[i]);
    }
    return out;
  }
};
std::vector<uint64_t> NewAllocationIds(const vt::rocm::F16ScratchStats& before) {
  std::vector<uint64_t> ids;
  const auto after = vt::rocm::GetF16ScratchStats();
  for (const auto pointer : after.allocation_pointers) {
    if (std::find(before.allocation_pointers.begin(), before.allocation_pointers.end(), pointer) ==
        before.allocation_pointers.end()) {
      const auto id = rocm_f16_test::AllocationId(pointer);
      REQUIRE(id != 0);
      ids.push_back(id);
    }
  }
  REQUIRE_FALSE(ids.empty());
  return ids;
}
void CheckFreed(const std::vector<uint64_t>& ids, bool freed) {
  for (const auto id : ids) {
    CAPTURE(id);
    CHECK(rocm_f16_test::WasFreed(id) == freed);
  }
}
float Stored(float value, DType dtype) {
  return dtype == DType::kF16 ? vt::F16ToF32(vt::F32ToF16(value))
      : dtype == DType::kBF16 ? vt::BF16ToF32(vt::F32ToBF16(value)) : value;
}

void GemmCase(DeviceRun& run, bool bt, int64_t m, int64_t n, int64_t k,
              DType ad, DType bd, DType od, std::optional<DType> marker = {},
              bool strided = false, bool extreme = false) {
  CAPTURE(bt); CAPTURE(m); CAPTURE(n); CAPTURE(k);
  CAPTURE(vt::Name(ad)); CAPTURE(vt::Name(bd)); CAPTURE(vt::Name(od));
  CAPTURE(strided); CAPTURE(extreme);
  const int64_t pitch = k + (strided ? 3 : 0);
  Tensor a = run.Make(ad, m, pitch), b = run.Make(bd, bt ? n : k, bt ? k : n);
  Tensor out = run.Make(od, m, n);
  std::vector<float> av(static_cast<size_t>(m * pitch));
  std::vector<float> bv(static_cast<size_t>(n * k));
  for (size_t i = 0; i < av.size(); ++i)
    av[i] = std::sin(static_cast<float>(i + 1) * .31f) * .25f;
  for (size_t i = 0; i < bv.size(); ++i)
    bv[i] = std::cos(static_cast<float>(i + 3) * .17f) * .25f + 0.0009765625f;
  if (extreme) {
    std::fill(av.begin(), av.end(), 0.f);
    std::fill(bv.begin(), bv.end(), 1.0009765625f);
    av[0] = 131072.f;
    if (m > 1) av[static_cast<size_t>(pitch)] = std::ldexp(1.f, -30);
  }
  run.Upload(a, av); run.Upload(b, bv);
  a.shape[1] = k;
  b.weight_value_dtype = marker;
  if (bt) vt::MatmulBT(run.queue, out, a, b);
  else vt::Matmul(run.queue, out, a, b);
  const auto actual = run.Download(out);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.f;
      for (int64_t inner = 0; inner < k; ++inner) {
        float w = Stored(bv[static_cast<size_t>(bt ? col * k + inner : inner * n + col)], bd);
        if (marker) w = Stored(w, *marker);
        sum += Stored(av[static_cast<size_t>(row * pitch + inner)], ad) * w;
      }
      const float expected = Stored(sum, od);
      const float got = actual[static_cast<size_t>(row * n + col)];
      if (extreme) {
        CHECK(std::isfinite(got));
        CHECK(got == expected);
      } else {
        const float tolerance = od == DType::kBF16 ? .002f + .008f * std::abs(expected)
                                                  : 2e-5f + 2e-5f * std::abs(expected);
        CHECK(std::abs(got - expected) <= tolerance);
      }
    }
  }
  const auto provider = vt::GetOpProviderStats(bt ? vt::OpId::kMatmulBT : vt::OpId::kMatmul,
                                              vt::DeviceType::kROCM);
  REQUIRE(provider.last_selected != nullptr);
  CHECK(std::string(provider.last_selected) == vt::kNativeProviderName);
}
}  // namespace

TEST_CASE("ROCm F16 ordinary fallback matches the pinned vLLM output") {
  REQUIRE(vt::rocm::DeviceAvailable());
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                    "fixtures/rocm_f16_weights/upstream-fallback.json";
  std::ifstream input(path);
  REQUIRE(input.good());
  const auto fixture = nlohmann::json::parse(input);
  REQUIRE(fixture.at("upstream_revision") == "e126687a9a828d513c01a07cd69f025f27d63280");
  REQUIRE(fixture.at("dtype") == "float16");
  REQUIRE(fixture.at("x_shape") == nlohmann::json::array({6, 64}));
  REQUIRE(fixture.at("weight_shape") == nlohmann::json::array({128, 64}));
  REQUIRE(fixture.at("atol") == .001);
  REQUIRE(fixture.at("rtol") == .001);
  REQUIRE(fixture.at("skinny_calls") == 0);
  const auto x_bits = fixture.at("x_bits").get<std::vector<uint16_t>>();
  const auto weight_bits = fixture.at("weight_bits").get<std::vector<uint16_t>>();
  const auto output_bits = fixture.at("result_bits").get<std::vector<uint16_t>>();
  REQUIRE(x_bits.size() == 6 * 64);
  REQUIRE(weight_bits.size() == 128 * 64);
  REQUIRE(output_bits.size() == 6 * 128);
  std::vector<float> x_values(x_bits.size()), weights(weight_bits.size());
  for (size_t i = 0; i < x_bits.size(); ++i) x_values[i] = vt::F16ToF32(x_bits[i]);
  for (size_t i = 0; i < weight_bits.size(); ++i) weights[i] = vt::F16ToF32(weight_bits[i]);
  DeviceRun run;
  for (bool bt : {false, true}) {
    CAPTURE(bt);
    auto a = run.Make(DType::kF16, 6, 64);
    auto b = run.Make(DType::kF16, bt ? 128 : 64, bt ? 64 : 128);
    auto out = run.Make(DType::kF32, 6, 128);
    run.Upload(a, x_values);
    std::vector<float> arranged(weights.size());
    for (size_t row = 0; row < 128; ++row)
      for (size_t k = 0; k < 64; ++k)
        arranged[bt ? row * 64 + k : k * 128 + row] = weights[row * 64 + k];
    run.Upload(b, arranged);
    if (bt) vt::MatmulBT(run.queue, out, a, b); else vt::Matmul(run.queue, out, a, b);
    const auto actual = run.Download(out);
    for (size_t row = 0; row < 6; ++row)
      for (size_t column = 0; column < 128; ++column) {
        const size_t index = row * 128 + column;
        const float oracle = vt::F16ToF32(output_bits[index]);
        // The seam returns F32. This comparison retains the upstream F16
        // output dtype and its original 1e-3 absolute and relative tolerance.
        CHECK(std::abs(Stored(actual[index], DType::kF16) - oracle) <= .001f + .001f * std::abs(oracle));
        float accumulated = 0.f;
        for (size_t k = 0; k < 64; ++k)
          accumulated += x_values[row * 64 + k] * weights[column * 64 + k];
        CHECK(std::abs(actual[index] - accumulated) <= 2e-5f + 2e-5f * std::abs(accumulated));
      }
    const auto provider = vt::GetOpProviderStats(bt ? vt::OpId::kMatmulBT : vt::OpId::kMatmul,
                                                vt::DeviceType::kROCM);
    REQUIRE(provider.last_selected != nullptr);
    CHECK(std::string(provider.last_selected) == vt::kNativeProviderName);
  }
}

TEST_CASE("ROCm F16 ordinary GEMM preserves storage and model value contracts") {
  REQUIRE_MESSAGE(vt::rocm::DeviceAvailable(), "this registered physical gate requires a ROCm device");
  DeviceRun run;
  for (bool bt : {false, true}) {
    for (int64_t m : {1, 4, 6, 19}) {
      for (auto ad : {DType::kF16, DType::kBF16, DType::kF32}) {
        for (auto bd : {DType::kF16, DType::kBF16, DType::kF32}) {
          for (auto od : {DType::kBF16, DType::kF32})
            GemmCase(run, bt, m, 11, 13, ad, bd, od, {}, bt);
        }
        for (auto marker : {DType::kBF16, DType::kF32}) {
          for (auto od : {DType::kBF16, DType::kF32})
            GemmCase(run, bt, m, 11, 13, ad, DType::kF16, od, marker, bt);
        }
      }
    }
    // BF16 is a local regression parameter, not a newly attributed upstream case.
    GemmCase(run, bt, 6, 128, 64, DType::kBF16, DType::kBF16, DType::kBF16);
    GemmCase(run, bt, 4, 16, 64, DType::kBF16, DType::kF16, DType::kBF16, DType::kBF16);
    for (auto marker : {std::optional<DType>{}, std::optional<DType>{DType::kBF16},
                        std::optional<DType>{DType::kF32}})
      for (auto od : {DType::kBF16, DType::kF32})
        GemmCase(run, bt, 2, 3, 1, DType::kF32, DType::kF16, od, marker, bt, true);
  }
}

TEST_CASE("ROCm F16 zero extents and ordinary validation") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  for (bool bt : {false, true}) {
    for (const auto& dims : {std::vector<int64_t>{0, 3, 2}, {2, 0, 3}, {2, 3, 0}}) {
      const auto m = dims[0], n = dims[1], k = dims[2];
      Tensor a = run.Make(DType::kF32, m, k);
      Tensor b = run.Make(DType::kF16, bt ? n : k, bt ? k : n);
      Tensor out = run.Make(DType::kBF16, m, n);
      run.backend.Memset(run.queue, out.data, 0x7f, static_cast<size_t>(out.Numel()) * 2);
      b.weight_value_dtype = DType::kBF16;
      if (bt) vt::MatmulBT(run.queue, out, a, b); else vt::Matmul(run.queue, out, a, b);
      for (float value : run.Download(out)) CHECK(value == 0.f);
    }
    Tensor a = run.Make(DType::kF32, 2, 3), b = run.Make(DType::kF16, 3, 3);
    Tensor out = run.Make(DType::kF32, 2, 3);
    auto call = [&] { if (bt) vt::MatmulBT(run.queue, out, a, b);
                     else vt::Matmul(run.queue, out, a, b); };
    a.rank = 1; CHECK_THROWS(call()); a.rank = 2;
    b.shape[1] = 2; CHECK_THROWS(call()); b.shape[1] = 3;
    a.stride[1] = 2; CHECK_THROWS(call()); a.stride[1] = 1;
    a.stride[0] = 2; CHECK_THROWS(call()); a.stride[0] = 3;
    b.device.index = run.queue.device.index + 1; CHECK_THROWS(call()); b.device = run.queue.device;
    b.weight_value_dtype = DType::kF16; CHECK_THROWS(call()); b.weight_value_dtype.reset();
    a.weight_value_dtype = DType::kBF16; CHECK_THROWS(call()); a.weight_value_dtype.reset();
    out.weight_value_dtype = DType::kBF16; CHECK_THROWS(call()); out.weight_value_dtype.reset();
    out.dtype = DType::kF16; CHECK_THROWS(call());
  }
}

TEST_CASE("ROCm F16 embedding rounds weights and preserves both ID widths") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  const std::vector<float> values = {1.0009765625f, -1.0009765625f, .00006103515625f,
                                    2.00390625f, -.00390625f, 65504.f,
                                    3.001953125f, 4.00390625f, 0.f};
  Tensor table = run.Make(DType::kF16, 3, 3);
  run.Upload(table, values);
  for (DType id_dtype : {DType::kI32, DType::kI64}) {
    Tensor ids = run.Make(id_dtype, 1, 4);
    ids.rank = 1; ids.shape[0] = 4; ids.stride[0] = 1;
    const std::vector<int32_t> small = {0, 2, 2, 1};
    const std::vector<int64_t> wide(small.begin(), small.end());
    auto upload_ids = [&] {
      run.backend.Copy(run.queue, ids.data,
          id_dtype == DType::kI32 ? static_cast<const void*>(small.data())
                                  : static_cast<const void*>(wide.data()), 4 * vt::SizeOf(id_dtype));
      run.backend.Synchronize(run.queue);
    };
    upload_ids();
    for (auto marker : {std::optional<DType>{}, std::optional<DType>{DType::kBF16},
                        std::optional<DType>{DType::kF32}}) {
      table.weight_value_dtype = marker;
      for (DType od : {DType::kBF16, DType::kF32}) {
        Tensor out = run.Make(od, 4, 3);
        vt::Embedding(run.queue, out, table, ids);
        const auto actual = run.Download(out);
        for (int i = 0; i < 12; ++i) {
          float expected = Stored(values[small[i / 3] * 3 + i % 3], DType::kF16);
          if (marker) expected = Stored(expected, *marker);
          CHECK(actual[i] == Stored(expected, od));
        }
        for (int64_t invalid : {-1LL, 3LL, 4294967296LL}) {
          if (id_dtype == DType::kI32 && invalid > INT32_MAX) continue;
          const int32_t narrow = static_cast<int32_t>(invalid);
          run.backend.Copy(run.queue, ids.data,
              id_dtype == DType::kI32 ? static_cast<const void*>(&narrow)
                                      : static_cast<const void*>(&invalid), vt::SizeOf(id_dtype));
          run.backend.Synchronize(run.queue);
          CHECK_THROWS_WITH(vt::Embedding(run.queue, out, table, ids),
              ("vt rocm: embedding: id " + std::to_string(invalid) + " out of range [0, 3)").c_str());
        }
        upload_ids();
        ids.weight_value_dtype = DType::kBF16;
        CHECK_THROWS(vt::Embedding(run.queue, out, table, ids));
        ids.weight_value_dtype.reset();
      }
    }
    ids.shape[0] = 0;
    Tensor empty = run.Make(DType::kF32, 0, 3);
    CHECK_NOTHROW(vt::Embedding(run.queue, empty, table, ids));
  }
}

TEST_CASE("ROCm F16 scratch reuse remains bounded across streams and growth") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun first, second;
  const auto before = vt::rocm::GetF16ScratchStats();
  GemmCase(first, true, 6, 17, 19, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
  GemmCase(second, true, 6, 23, 29, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
  const auto allocated = vt::rocm::GetF16ScratchStats();
  CHECK(allocated.stream_count >= before.stream_count + 2);
  for (int i = 0; i < 4; ++i) {
    GemmCase(first, true, 6, 17, 19, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
    GemmCase(second, true, 6, 23, 29, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
  }
  CHECK(vt::rocm::GetF16ScratchStats().retained_bytes == allocated.retained_bytes);
  GemmCase(first, true, 6, 67, 71, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
  const auto grown = vt::rocm::GetF16ScratchStats();
  CHECK(grown.high_water_bytes > allocated.high_water_bytes);
  CHECK(grown.retained_bytes < grown.capacity_bytes * 2);
}

TEST_CASE("ROCm F16 queue ordering and captured scratch survive later growth") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run, other;
  Tensor a = run.Make(DType::kBF16, 1, 3), b = run.Make(DType::kF16, 3, 3);
  Tensor first = run.Make(DType::kF32, 1, 3), second = run.Make(DType::kF32, 1, 3);
  Tensor b2 = run.Make(DType::kF16, 3, 3);
  run.Upload(a, {1.f, 2.f, 3.f});
  run.Upload(b, std::vector<float>(9, 1.0009765625f));
  run.Upload(b2, std::vector<float>(9, 2.001953125f));
  b.weight_value_dtype = DType::kBF16;
  b2.weight_value_dtype = DType::kBF16;
  vt::MatmulBT(run.queue, first, a, b);
  vt::MatmulBT(run.queue, second, a, b2);
  for (float value : run.Download(first)) CHECK(value == 6.f);
  for (float value : run.Download(second)) CHECK(value == 12.f);
  REQUIRE(run.backend.SupportsGraphCapture());
  run.backend.BeginCapture(run.queue);
  vt::MatmulBT(run.queue, first, a, b);
  void* graph = run.backend.EndCaptureGraph(run.queue);
  REQUIRE(graph != nullptr);
  struct GraphOwner {
    vt::Backend& backend;
    void* graph;
    ~GraphOwner() { backend.DestroyGraph(graph); }
  } owner{run.backend, graph};
  // A larger eager call grows this queue's slab after the graph captured it.
  GemmCase(run, true, 6, 67, 71, DType::kF32, DType::kF16, DType::kF32, DType::kBF16);
  for (int repeat = 0; repeat < 3; ++repeat) {
    run.backend.Memset(run.queue, first.data, 0, 12);
    run.backend.ReplayGraph(run.queue, graph);
    GemmCase(other, true, 4, 17, 19, DType::kBF16, DType::kF16, DType::kF32, DType::kBF16);
    for (float value : run.Download(first)) CHECK(value == 6.f);
  }
}

TEST_CASE("ROCm F16 ordinary NN and BT refuse unsupported compute overrides") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  Tensor a = run.Make(DType::kBF16, 1, 3);
  Tensor b = run.Make(DType::kF16, 3, 3);
  Tensor out = run.Make(DType::kF32, 1, 3);
  run.Upload(a, {1.f, 2.f, 3.f});
  run.Upload(b, std::vector<float>(9, 1.0009765625f));
  // Unmarked F16 B keeps this BF16/F16 pair outside the allowed 16bf arm.
  REQUIRE_FALSE(b.weight_value_dtype.has_value());
  for (bool bt : {false, true}) {
    CAPTURE(bt);
    for (const char* compute : {"16f", "16bf"}) {
      CAPTURE(std::string(compute));
      ScopedF16Env override("VT_ROCM_GEMM_COMPUTE", compute);
      auto call = [&] {
        if (bt) vt::MatmulBT(run.queue, out, a, b);
        else vt::Matmul(run.queue, out, a, b);
      };
      const char* refusal = std::strcmp(compute, "16f") == 0
          ? "ROCm ordinary GEMM refuses VT_ROCM_GEMM_COMPUTE=16f: F32 scalars required"
          : "ROCm ordinary GEMM refuses VT_ROCM_GEMM_COMPUTE=16bf for this input pair";
      CHECK_THROWS_WITH(call(), doctest::Contains(refusal));
    }
  }
}

TEST_CASE("ROCm F16 scratch releases after repeated queue destruction") {
  REQUIRE(vt::rocm::DeviceAvailable());
  const auto before = vt::rocm::GetF16ScratchStats();
  for (int repeat = 0; repeat < 4; ++repeat) {
    CAPTURE(repeat);
    std::vector<uint64_t> ids;
    {
      DeviceRun run;
      GemmCase(run, true, 4, 17, 19, DType::kF32, DType::kF16,
               DType::kF32, DType::kBF16);
      CHECK(vt::rocm::GetF16ScratchStats().retained_bytes > before.retained_bytes);
      ids = NewAllocationIds(before);
      CheckFreed(ids, false);
    }
    CheckFreed(ids, true);
    const auto released = vt::rocm::GetF16ScratchStats();
    CHECK(released.retained_bytes == before.retained_bytes);
    CHECK(released.capacity_bytes == before.capacity_bytes);
    CHECK(released.stream_count == before.stream_count);
  }
}

TEST_CASE("ROCm F16 graph handles own scratch after their source queue ends") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  const auto before = vt::rocm::GetF16ScratchStats();
  struct QueueOwner {
    vt::Backend& backend;
    vt::Queue queue;
    ~QueueOwner() { if (queue.handle) backend.DestroyQueue(queue); }
  } source{run.backend, run.backend.CreateQueue()};
  Tensor a = run.Make(DType::kBF16, 1, 3), b = run.Make(DType::kF16, 3, 3);
  Tensor out = run.Make(DType::kF32, 1, 3);
  run.Upload(a, {1.f, 2.f, 3.f});
  run.Upload(b, std::vector<float>(9, 1.0009765625f));
  b.weight_value_dtype = DType::kBF16;
  vt::MatmulBT(source.queue, out, a, b);
  run.backend.Synchronize(source.queue);
  const auto warmed = vt::rocm::GetF16ScratchStats();
  REQUIRE(warmed.retained_bytes > before.retained_bytes);
  const auto captured_ids = NewAllocationIds(before);
  run.backend.BeginCapture(source.queue);
  vt::MatmulBT(source.queue, out, a, b);
  struct GraphOwner {
    vt::Backend& backend;
    void* graph;
    ~GraphOwner() { if (graph) backend.DestroyGraph(graph); }
  } graph{run.backend, run.backend.EndCaptureGraph(source.queue)};
  REQUIRE(graph.graph != nullptr);
  run.backend.BeginCapture(source.queue);
  vt::MatmulBT(source.queue, out, a, b);
  GraphOwner second{run.backend, run.backend.EndCaptureGraph(source.queue)};
  REQUIRE(second.graph != nullptr);
  REQUIRE(second.graph != graph.graph);
  Tensor wide_a = run.Make(DType::kF32, 1, 71);
  Tensor wide_b = run.Make(DType::kF16, 67, 71);
  Tensor wide_out = run.Make(DType::kF32, 1, 67);
  run.Upload(wide_a, std::vector<float>(71, 1.f));
  run.Upload(wide_b, std::vector<float>(67 * 71, 1.0009765625f));
  wide_b.weight_value_dtype = DType::kBF16;
  vt::MatmulBT(source.queue, wide_out, wide_a, wide_b);
  run.backend.Synchronize(source.queue);
  CHECK(vt::rocm::GetF16ScratchStats().retained_bytes > warmed.retained_bytes);
  const auto eager_ids = NewAllocationIds(warmed);
  for (float value : run.Download(wide_out)) CHECK(value == 71.f);
  run.backend.DestroyQueue(source.queue);
  CheckFreed(eager_ids, true);
  CheckFreed(captured_ids, false);
  const auto queue_released = vt::rocm::GetF16ScratchStats();
  CHECK(queue_released.stream_count == before.stream_count);
  CHECK(queue_released.capacity_bytes == before.capacity_bytes);
  CHECK(queue_released.retained_bytes == warmed.retained_bytes);
  for (int repeat = 0; repeat < 3; ++repeat) {
    run.backend.Memset(run.queue, out.data, 0, out.Bytes());
    run.backend.ReplayGraph(run.queue, graph.graph);
    for (float value : run.Download(out)) CHECK(value == 6.f);
  }
  run.backend.DestroyGraph(graph.graph);
  graph.graph = nullptr;
  CheckFreed(captured_ids, false);
  run.backend.Memset(run.queue, out.data, 0, out.Bytes());
  run.backend.ReplayGraph(run.queue, second.graph);
  for (float value : run.Download(out)) CHECK(value == 6.f);
  run.backend.DestroyGraph(second.graph);
  second.graph = nullptr;
  CheckFreed(captured_ids, true);
  CHECK(vt::rocm::GetF16ScratchStats().retained_bytes == before.retained_bytes);
}

TEST_CASE("ROCm F16 single graph replacement releases its previous scratch") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  Tensor a = run.Make(DType::kBF16, 1, 3), b = run.Make(DType::kF16, 3, 3);
  Tensor out = run.Make(DType::kF32, 1, 3);
  run.Upload(a, {1.f, 2.f, 3.f});
  run.Upload(b, std::vector<float>(9, 1.0009765625f));
  b.weight_value_dtype = DType::kBF16;
  const auto before = vt::rocm::GetF16ScratchStats();
  size_t one_graph_bytes = 0;
  std::vector<uint64_t> previous_ids;
  for (int repeat = 0; repeat < 4; ++repeat) {
    CAPTURE(repeat);
    const auto previous = vt::rocm::GetF16ScratchStats();
    vt::Queue source = run.backend.CreateQueue();
    vt::MatmulBT(source, out, a, b);
    run.backend.Synchronize(source);
    const auto current_ids = NewAllocationIds(previous);
    run.backend.BeginCapture(source);
    vt::MatmulBT(source, out, a, b);
    run.backend.EndCapture(source);
    CheckFreed(previous_ids, true);
    previous_ids = current_ids;
    run.backend.DestroyQueue(source);
    CheckFreed(current_ids, false);
    run.backend.Memset(run.queue, out.data, 0, out.Bytes());
    run.backend.Replay(run.queue);
    for (float value : run.Download(out)) CHECK(value == 6.f);
    const auto active = vt::rocm::GetF16ScratchStats();
    if (repeat == 0) one_graph_bytes = active.retained_bytes - before.retained_bytes;
    CHECK(one_graph_bytes > 0);
    CHECK(active.retained_bytes == before.retained_bytes + one_graph_bytes);
    CHECK(active.stream_count == before.stream_count);
    CHECK(active.capacity_bytes == before.capacity_bytes);
  }
  // The single-slot API releases a prior graph when its replacement is installed.
  // A capture with no F16 operation requires no F16 scratch ownership.
  run.backend.BeginCapture(run.queue);
  run.backend.Memset(run.queue, out.data, 0, out.Bytes());
  run.backend.EndCapture(run.queue);
  CheckFreed(previous_ids, true);
  run.backend.Replay(run.queue);
  for (float value : run.Download(out)) CHECK(value == 0.f);
  CHECK(vt::rocm::GetF16ScratchStats().retained_bytes == before.retained_bytes);
}

TEST_CASE("ROCm F16 capture refuses unwarmed scratch growth") {
  REQUIRE(vt::rocm::DeviceAvailable());
  DeviceRun run;
  GemmCase(run, true, 1, 3, 3, DType::kBF16, DType::kF16, DType::kF32, DType::kBF16);
  Tensor a = run.Make(DType::kF32, 1, 71);
  Tensor b = run.Make(DType::kF16, 67, 71);
  Tensor out = run.Make(DType::kF32, 1, 67);
  run.Upload(a, std::vector<float>(71, 1.f));
  run.Upload(b, std::vector<float>(67 * 71, 1.f));
  b.weight_value_dtype = DType::kBF16;
  const auto warmed = vt::rocm::GetF16ScratchStats();
  run.backend.BeginCapture(run.queue);
  CHECK_THROWS_WITH(vt::MatmulBT(run.queue, out, a, b),
                    doctest::Contains("warm this scratch size before graph capture"));
  // The refusal occurs before allocating or submitting work, so capture still
  // accepts another operation and the existing allocation remains reusable.
  run.backend.Memset(run.queue, out.data, 0, out.Bytes());
  void* graph = run.backend.EndCaptureGraph(run.queue);
  REQUIRE(graph != nullptr);
  run.backend.ReplayGraph(run.queue, graph);
  for (float value : run.Download(out)) CHECK(value == 0.f);
  run.backend.DestroyGraph(graph);
  const auto unchanged = vt::rocm::GetF16ScratchStats();
  CHECK(unchanged.capacity_bytes == warmed.capacity_bytes);
  CHECK(unchanged.retained_bytes == warmed.retained_bytes);
  CHECK(unchanged.allocation_pointers == warmed.allocation_pointers);
  GemmCase(run, true, 1, 3, 3, DType::kBF16, DType::kF16, DType::kF32, DType::kBF16);
}
