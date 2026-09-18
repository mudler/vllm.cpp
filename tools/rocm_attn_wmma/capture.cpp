// Capture the production PagedAttention operation for a pinned external oracle.
#include <chrono>
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
#include "vt/rocm/rocm_runtime.h"

namespace {
using Json = nlohmann::json;
using Path = std::filesystem::path;
constexpr vt::Device kDevice{vt::DeviceType::kROCM, 0};

std::string Read(const Path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void Write(const Path& path, const std::string& data) {
  std::ofstream file(path, std::ios::binary);
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!file) throw std::runtime_error("cannot write " + path.string());
}

struct Storage {
  vt::Backend& backend = vt::GetBackend(kDevice.type);
  vt::Queue queue = backend.CreateQueue();
  std::vector<void*> allocations;
  ~Storage() {
    backend.Synchronize(queue);
    for (auto* p : allocations) backend.Free(p);
    backend.DestroyQueue(queue);
  }
  vt::Tensor Upload(const Path& path, vt::DType dtype,
                    std::initializer_list<int64_t> shape) {
    const auto data = Read(path);
    size_t size = vt::SizeOf(dtype);
    for (auto n : shape) {
      if (n <= 0) throw std::runtime_error("nonpositive fixture dimension");
      size *= static_cast<size_t>(n);
    }
    if (data.size() != size) throw std::runtime_error("fixture size differs: " + path.string());
    void* p = backend.Alloc(size);
    allocations.push_back(p);
    backend.Copy(queue, p, data.data(), size);
    backend.Synchronize(queue);
    return vt::Tensor::Contiguous(p, dtype, kDevice, shape);
  }
};

void Capture(const Path& manifest_path, const Path& output) {
  if (!vt::rocm::DeviceAvailable()) throw std::runtime_error("no physical ROCm device");
  const auto manifest = Json::parse(Read(manifest_path));
  if (manifest.at("cases").empty()) throw std::runtime_error("empty fixture manifest");
  std::filesystem::create_directories(output);
  Json report = {{"cases", Json::array()}};
  const int repeats = std::getenv("VT_ATTN_WMMA_REPEATS")
                          ? std::stoi(std::getenv("VT_ATTN_WMMA_REPEATS")) : 1;
  if (repeats < 1 || repeats > 100000) throw std::runtime_error("invalid repeat count");
  for (const auto& entry : manifest.at("cases")) {
    const auto name = entry.at("name").get<std::string>();
    const auto input = manifest_path.parent_path() / name;
    const int64_t tokens = entry.at("tokens"), heads = entry.at("heads");
    const int64_t kv_heads = entry.at("kv_heads"), dim = entry.at("dim");
    const int64_t blocks = entry.at("blocks"), block = entry.at("block");
    const int32_t seq_len = entry.at("seq_len");
    const std::vector<int32_t> qsl{0, static_cast<int32_t>(tokens)};
    Storage storage;
    auto q = storage.Upload(input / "q.bf16", vt::DType::kBF16, {tokens, heads, dim});
    auto k = storage.Upload(input / "k.bf16", vt::DType::kBF16, {blocks, block, kv_heads, dim});
    auto v = storage.Upload(input / "v.bf16", vt::DType::kBF16, {blocks, block, kv_heads, dim});
    auto bt = storage.Upload(input / "blocks.i32", vt::DType::kI32, {1, blocks});
    auto lens = storage.Upload(input / "length.i32", vt::DType::kI32, {1});
    auto starts = storage.Upload(input / "starts.i32", vt::DType::kI32, {2});
    // A finite poison makes deleting the production launch fail output validation.
    auto out = storage.Upload(input / "poison.bf16", vt::DType::kBF16, {tokens, heads, dim});
    vt::PagedAttentionArgs args{entry.at("scale").get<float>(), entry.at("causal").get<bool>()};
    args.query_start_loc_host = qsl.data();
    args.max_seq_len = seq_len;
    args.logits_soft_cap = entry.at("softcap");
    const int32_t left = entry.at("window_left"), right = entry.at("window_right");
    if (left >= 0 || right >= 0) args.window_size = vt::AttentionWindow{left, right};
    auto launch = [&] {
      vt::PagedAttention(storage.queue, out, q, k, v, bt, lens, starts, args);
    };
    const auto reference_before = vt::GetReferenceTierHits();
    launch();
    storage.backend.Synchronize(storage.queue);
    std::string bytes(out.Bytes(), '\0');
    storage.backend.Copy(storage.queue, bytes.data(), out.data, bytes.size());
    storage.backend.Synchronize(storage.queue);
    Write(output / (name + ".bf16"), bytes);
    for (int warmup = 0; warmup < 5; ++warmup) launch();
    storage.backend.Synchronize(storage.queue);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) launch();
    storage.backend.Synchronize(storage.queue);
    const auto end = std::chrono::steady_clock::now();
    if (vt::GetReferenceTierHits() != reference_before)
      throw std::runtime_error("attention entered the CPU reference tier");
    const double us = std::chrono::duration<double, std::micro>(end - start).count() / repeats;
    report["cases"].push_back({{"name", name}, {"microseconds_per_call", us},
                                {"repeats", repeats}, {"output_dtype", "bf16"}});
    std::printf("CAPTURE %s %.3f us/call\n", name.c_str(), us);
  }
  Write(output / "report.json", report.dump(2) + "\n");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("use MANIFEST OUTPUT");
    Capture(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CAPTURE FAILED: %s\n", error.what());
    return 1;
  }
}
