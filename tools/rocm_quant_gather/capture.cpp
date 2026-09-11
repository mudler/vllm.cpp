// BACKEND-ROCM-QUANT-GATHER (#3093): same-byte operation captures.
// The operator runs the HIP mode under the recorded GPU mutex. Generate mode
// creates deterministic host inputs without initializing a GPU backend.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/rocm_quant_gather_fixture.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_HIP
#include "vt/rocm/rocm_embedding_quant.h"
#endif

namespace {

using Json = nlohmann::json;
using Path = std::filesystem::path;
constexpr vt::Device kDevice{vt::DeviceType::kROCM, 0};

std::string Read(const Path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void Write(const Path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("cannot write " + path.string());
}

void Generate(const Path& directory) {
  std::filesystem::create_directories(directory);
  Json manifest = {{"version", 1}, {"source", "deterministic-synthetic"},
                   {"seed", rocm_gather_test::kSeed}, {"cases", Json::array()}};
  constexpr int32_t ids[] = {0, 127, 64, 0};
  for (const auto& format : rocm_gather_test::kFormats) {
    for (int width : {256, 1024}) {
      const auto name = std::string(format.name) + "-" + std::to_string(width);
      Write(directory / (name + ".packed"), rocm_gather_test::PackedTable(format, 128, width));
      Write(directory / (name + ".ids"), std::string(reinterpret_cast<const char*>(ids), sizeof(ids)));
      manifest["cases"].push_back({{"name", name}, {"type", format.ggml},
                                   {"rows", 128}, {"width", width},
                                   {"packed", name + ".packed"}, {"ids", name + ".ids"}});
    }
  }
  Write(directory / "manifest.json", manifest.dump(2) + "\n");
}

vt::DType FindType(uint32_t type) {
  for (const auto& format : rocm_gather_test::kFormats) {
    if (format.ggml == type) return format.dtype;
  }
  throw std::runtime_error("input type exceeds the shared gather decoder set");
}

class DeviceStorage {
 public:
  DeviceStorage() : backend(vt::GetBackend(kDevice.type)), queue(backend.CreateQueue()) {}
  ~DeviceStorage() {
    for (auto* allocation : allocations) backend.Free(allocation);
    backend.DestroyQueue(queue);
  }
  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;
  void* Allocate(size_t size) {
    auto* pointer = backend.Alloc(size);
    allocations.push_back(pointer);
    return pointer;
  }
  vt::Tensor Upload(const std::string& bytes, vt::DType dtype,
                    std::initializer_list<int64_t> shape) {
    auto* pointer = Allocate(bytes.size());
    backend.Copy(queue, pointer, bytes.data(), bytes.size());
    return vt::Tensor::Contiguous(pointer, dtype, kDevice, shape);
  }
  vt::Backend& backend;
  vt::Queue queue;
  std::vector<void*> allocations;
};

void Capture(const Path& input, const Path& output, const std::string& selected) {
#ifdef VLLM_CPP_HIP
  const auto manifest = Json::parse(Read(input));
  std::filesystem::create_directories(output);
  Json report = {{"version", 1}, {"cases", Json::array()}};
  bool found = false;
  for (const auto& entry : manifest.at("cases")) {
    const auto name = entry.at("name").get<std::string>();
    if (!selected.empty() && selected != name) continue;
    found = true;
    const auto dtype = FindType(entry.at("type").get<uint32_t>());
    const int64_t rows = entry.at("rows").get<int64_t>();
    const int64_t width = entry.at("width").get<int64_t>();
    const auto packed = Read(input.parent_path() / entry.at("packed").get<std::string>());
    const auto ids = Read(input.parent_path() / entry.at("ids").get<std::string>());
    if (rows <= 0 || width <= 0 || width % vt::BlockElems(dtype) != 0 ||
        packed.size() != static_cast<size_t>(rows) * vt::RowSizeBytes(dtype, width) ||
        ids.empty() || ids.size() % sizeof(int32_t) != 0) {
      throw std::runtime_error("input geometry or byte count mismatch");
    }
    const auto tokens = static_cast<int64_t>(ids.size() / sizeof(int32_t));
    DeviceStorage device;
    auto table = device.Upload(packed, dtype, {rows, width});
    const bool use_i64 = std::getenv("VT_ROCM_GATHER_TRACE_I64_IDS") != nullptr;
    std::string index_bytes = ids;
    if (use_i64) {
      index_bytes.resize(static_cast<size_t>(tokens) * sizeof(int64_t));
      for (int64_t token = 0; token < tokens; ++token) {
        int32_t narrow;
        std::memcpy(&narrow, ids.data() + token * sizeof(narrow), sizeof(narrow));
        const int64_t wide = narrow;
        std::memcpy(index_bytes.data() + token * sizeof(wide), &wide, sizeof(wide));
      }
    }
    const auto index_dtype = use_i64 ? vt::DType::kI64 : vt::DType::kI32;
    auto indices = device.Upload(index_bytes, index_dtype, {tokens});
    // f32 is the explicit primitive oracle mode. Model gathers emit bf16.
    for (auto out_dtype : {vt::DType::kF32, vt::DType::kBF16}) {
      const size_t output_bytes = static_cast<size_t>(tokens * width) * vt::SizeOf(out_dtype);
      auto result = vt::Tensor::Contiguous(device.Allocate(output_bytes), out_dtype,
                                            kDevice, {tokens, width});
      const auto hits = vt::GetReferenceTierHits();
      vt::ResetOpProviderStats(vt::OpId::kEmbeddingQuant, kDevice.type);
      vt::EnableOpProviderCallStats(true);
      vt::Embedding(device.queue, result, table, indices);
      vt::EnableOpProviderCallStats(false);
      const auto stats = vt::GetOpProviderStats(vt::OpId::kEmbeddingQuant, kDevice.type);
      if (stats.selections != 1 || stats.fallbacks != 0 || stats.declines != 0 ||
          vt::GetReferenceTierHits() != hits) {
        throw std::runtime_error("capture did not execute exactly one native gather");
      }
      std::string bytes(output_bytes, '\0');
      device.backend.Copy(device.queue, bytes.data(), result.data, bytes.size());
      device.backend.Synchronize(device.queue);
      const auto result_name = name + "-" + vt::Name(out_dtype) + ".bin";
      Write(output / result_name, bytes);
      report["cases"].push_back({{"name", name}, {"type", entry.at("type")},
                                  {"dtype", vt::Name(out_dtype)}, {"table_bytes", packed.size()},
                                  {"id_dtype", vt::Name(index_dtype)}, {"id_bytes", index_bytes.size()},
                                  {"output_bytes", output_bytes},
                                  {"scratch_bytes", sizeof(vt::rocm::EmbeddingQuantError)},
                                  {"selected_packed_bytes", 0}, {"selections", stats.selections},
                                  {"provider", stats.last_selected == nullptr ? "" : stats.last_selected},
                                  {"result", result_name}});
      std::printf("NATIVE %s %s table=%zu output=%zu scratch=%zu\n", name.c_str(),
                  vt::Name(out_dtype), packed.size(), output_bytes,
                  sizeof(vt::rocm::EmbeddingQuantError));
    }
  }
  if (!found) throw std::runtime_error("case selection matched no manifest entries");
  Write(output / "report.json", report.dump(2) + "\n");
#else
  (void)input;
  (void)output;
  (void)selected;
  throw std::runtime_error("native capture requires a HIP build");
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "generate") Generate(argv[2]);
    else if ((argc == 4 || argc == 5) && std::string(argv[1]) == "capture")
      Capture(argv[2], argv[3], argc == 5 ? argv[4] : "");
    else throw std::runtime_error("use generate DIRECTORY or capture MANIFEST OUTPUT [CASE]");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CAPTURE FAILED: %s\n", error.what());
    return 1;
  }
}
