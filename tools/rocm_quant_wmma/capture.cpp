// Native adapter for plugin d4c1f0d tests/test_kernels.py::test_mmq.
// Input generation and the unchanged upstream tolerances live in primary.py.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "vt/backend.h"
#include "vt/ops.h"

#ifdef VLLM_CPP_HIP
namespace vt::rocm {
std::string DeviceArchName(int);
uint64_t KQuantWmmaDispatchCount();
uint64_t KQuantWmmaQ4KDispatchCount();
}  // namespace vt::rocm
#endif

namespace {
using Json = nlohmann::json;
using Path = std::filesystem::path;
std::string Read(const Path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void Write(const Path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("cannot write " + path.string());
}

#ifdef VLLM_CPP_HIP
constexpr vt::Device kDevice{vt::DeviceType::kROCM, 0};
class Storage {
 public:
  Storage() : backend(vt::GetBackend(kDevice.type)), queue(backend.CreateQueue()) {}
  ~Storage() {
    for (auto* allocation : allocations) backend.Free(allocation);
    backend.DestroyQueue(queue);
  }
  void* Allocate(size_t size) {
    auto* pointer = backend.Alloc(size);
    allocations.push_back(pointer);
    return pointer;
  }
  vt::Tensor Upload(const std::string& bytes, vt::DType dtype, int64_t rows, int64_t width) {
    auto* pointer = Allocate(bytes.size());
    backend.Copy(queue, pointer, bytes.data(), bytes.size());
    return vt::Tensor::Contiguous(pointer, dtype, kDevice, {rows, width});
  }
  vt::Backend& backend;
  vt::Queue queue;
  std::vector<void*> allocations;
};

void Capture(const Path& input, const Path& output) {
  const auto arch = vt::rocm::DeviceArchName(0);
  const auto stem = arch.substr(0, arch.find(':'));
  if (stem != "gfx1100" && stem != "gfx1200" && stem != "gfx1201")
    throw std::runtime_error("capture requires an admitted physical WMMA device");
  const auto manifest = Json::parse(Read(input));
  std::filesystem::create_directories(output);
  Json report = {{"architecture", arch}, {"cases", Json::array()}};
  for (const auto& entry : manifest.at("cases")) {
    const auto name = entry.at("name").get<std::string>();
    const int type = entry.at("type").get<int>();
    if (type != 12 && type != 14) throw std::runtime_error("only Q4_K/Q6_K are in scope");
    const auto dtype = type == 12 ? vt::DType::kQ4_K : vt::DType::kQ6_K;
    const int64_t m = entry.at("tokens").get<int64_t>();
    const int64_t n = entry.at("rows").get<int64_t>();
    const int64_t k = entry.at("width").get<int64_t>();
    const auto input_dtype = entry.at("dtype").get<std::string>();
    const auto act_dtype = input_dtype == "f16" ? vt::DType::kF16 :
                           input_dtype == "bf16" ? vt::DType::kBF16 : vt::DType::kF32;
    if (input_dtype != "f16" && input_dtype != "bf16" && input_dtype != "f32")
      throw std::runtime_error("invalid activation dtype");
    // f32 is the harness adaptation for F16 output, absent from the native ABI.
    // BF16 uses its native two-byte output. primary.py narrows only F16 cases.
    const auto out_dtype = act_dtype == vt::DType::kBF16 ? vt::DType::kBF16 : vt::DType::kF32;
    const auto packed = Read(input.parent_path() / entry.at("packed").get<std::string>());
    const auto activation = Read(input.parent_path() / entry.at("activation").get<std::string>());
    if (m <= 0 || n <= 0 || k <= 0 || k % 256 != 0 ||
        packed.size() != static_cast<size_t>(n) * vt::RowSizeBytes(dtype, k) ||
        activation.size() != static_cast<size_t>(m * k) * vt::SizeOf(act_dtype))
      throw std::runtime_error("fixture geometry or bytes differ");
    Storage storage;
    auto a = storage.Upload(activation, act_dtype, m, k);
    auto b = storage.Upload(packed, dtype, n, k);
    std::string result(static_cast<size_t>(m * n) * vt::SizeOf(out_dtype), '\0');
    auto out = vt::Tensor::Contiguous(storage.Allocate(result.size()), out_dtype, kDevice, {m, n});
    const auto count = type == 12 ? vt::rocm::KQuantWmmaQ4KDispatchCount : vt::rocm::KQuantWmmaDispatchCount;
    const auto before = count();
    const auto reference_before = vt::GetReferenceTierHits();
    vt::MatmulBTQuant(storage.queue, out, a, b);
    storage.backend.Copy(storage.queue, result.data(), out.data, result.size());
    storage.backend.Synchronize(storage.queue);
    const auto calls = count() - before;
    const char* knob = std::getenv("VT_ROCM_QUANT_WMMA");
    const bool expected = m >= 16 && n >= 16 && (knob == nullptr || std::string(knob) != "0");
    if (calls != static_cast<uint64_t>(expected) || vt::GetReferenceTierHits() != reference_before)
      throw std::runtime_error("native WMMA dispatch or reference fallback mismatch: " + name);
    const auto result_name = name + ".bin";
    Write(output / result_name, result);
    report["cases"].push_back({{"name", name}, {"dtype", vt::Name(out_dtype)},
                                {"wmma_calls", calls}, {"result", result_name}});
    std::printf("NATIVE %s wmmas=%llu\n", name.c_str(), static_cast<unsigned long long>(calls));
  }
  if (report["cases"].empty()) throw std::runtime_error("empty input manifest");
  Write(output / "report.json", report.dump(2) + "\n");
}
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef VLLM_CPP_HIP
    if (argc != 3) throw std::runtime_error("use MANIFEST OUTPUT");
    Capture(argv[1], argv[2]);
    return 0;
#else
    (void)argc;
    (void)argv;
    throw std::runtime_error("native capture requires a HIP build");
#endif
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CAPTURE FAILED: %s\n", error.what());
    return 1;
  }
}
