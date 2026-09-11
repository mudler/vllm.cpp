// Public reachability gate for BACKEND-ROCM-QUANT-GATHER (#3093).
// Pinned plugin d4c1f0d: vocal_embeds.py::_apply_gguf_embedding, lines 79-98.
// Oracle model comparisons use the exact generated GGUFs exported by this test.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "capi/rocm_quant_gather_fixture.h"
#include "vllm.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

struct Completion {
  std::vector<int32_t> ids;
  std::vector<float> logits;
  bool capture_overflow = false;
  bool nonfinite_logits = false;
  int callbacks = 0;
};

// This C callback cannot throw. Reserve the complete capture before entering
// the public API, then append within that bound.
void Capture(const int32_t*, int32_t, float* logits, int32_t vocab, void* data) {
  auto& result = *static_cast<Completion*>(data);
  if (vocab != 128 || result.callbacks >= 4) {
    result.capture_overflow = true;
    return;
  }
  for (int32_t i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i])) result.nonfinite_logits = true;
    result.logits.push_back(logits[i]);
  }
  ++result.callbacks;
}

Completion Run(const std::string& path, const std::vector<int32_t>& prompt,
               bool expect_quant) {
  auto model = vllm_model_params_default();
  model.model_path = path.c_str();
  model.device = 0;  // Public AUTO in a HIP-only build, measured by provider stats.
  model.block_size = 16;
  model.num_blocks = 16;
  model.max_model_len = 64;
  model.max_num_seqs = 1;
  model.kv_cache_dtype = "auto";
  vllm_engine* raw = nullptr;
  const vllm_status loaded = vllm_engine_load(&model, &raw);
  INFO("public load: " << std::string(vllm_last_error()));
  REQUIRE(loaded == VLLM_OK);
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, &vllm_engine_free);

  Completion result;
  result.ids.resize(4);
  result.logits.reserve(4 * 128);
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0.0F;
  sampling.max_tokens = 4;
  sampling.ignore_eos = 1;
  sampling.has_seed = 1;
  sampling.seed = rocm_gather_test::kSeed;
  sampling.logits_processor = Capture;
  sampling.logits_processor_user_data = &result;
  int32_t count = 0;
  const auto hits_before = vt::GetReferenceTierHits();
  vt::ResetOpProviderStats(vt::OpId::kEmbeddingQuant, vt::DeviceType::kROCM);
  vt::ResetOpProviderStats(vt::OpId::kEmbedding, vt::DeviceType::kROCM);
  vt::EnableOpProviderCallStats(true);
  const vllm_status completed = vllm_complete_tokens(
      engine.get(), prompt.data(), static_cast<int32_t>(prompt.size()), &sampling,
      result.ids.data(), 4, &count, nullptr);
  vt::EnableOpProviderCallStats(false);
  INFO("public complete: " << std::string(vllm_last_error()));
  REQUIRE(completed == VLLM_OK);
  REQUIRE(count == 4);
  REQUIRE_FALSE(result.capture_overflow);
  REQUIRE_FALSE(result.nonfinite_logits);
  REQUIRE(result.callbacks == 4);
  CHECK(vt::GetReferenceTierHits() == hits_before);
  const auto op = expect_quant ? vt::OpId::kEmbeddingQuant : vt::OpId::kEmbedding;
  const auto stats = vt::GetOpProviderStats(op, vt::DeviceType::kROCM);
  INFO("public load and complete must select the ROCm embedding provider");
  CHECK(stats.selections > 0);
  CHECK(stats.declines == 0);
  CHECK(stats.fallbacks == 0);
  std::printf("PUBLIC prompt_tokens=%zu embedding=%s selections=%llu provider=%s tokens=",
              prompt.size(), expect_quant ? "quant" : "bf16", stats.selections,
              stats.last_selected == nullptr ? "none" : stats.last_selected);
  for (int32_t id : result.ids) std::printf(" %d", id);
  std::printf("\n");
  return result;
}

void Write(const std::filesystem::path& path, const std::string& data) {
  std::ofstream file(path, std::ios::binary);
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
  if (!file) throw std::runtime_error("cannot write gather fixture " + path.string());
}

void Export(const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  for (const auto& format : rocm_gather_test::kFormats) {
    Write(directory / (std::string(format.name) + ".gguf"),
          rocm_gather_test::BuildModel(format, false));
    Write(directory / (std::string(format.name) + "-dense.gguf"),
          rocm_gather_test::BuildModel(format, true));
    Write(directory / (std::string(format.name) + ".packed"),
          rocm_gather_test::PackedTable(format));
  }
}

void RecordCompletion(const char* format, int repeat, size_t prompt,
                      const char* arm, const Completion& result) {
  const char* directory = std::getenv("VT_ROCM_GATHER_PUBLIC_EVIDENCE");
  if (directory == nullptr) return;
  std::filesystem::create_directories(directory);
  const auto prefix = std::filesystem::path(directory) /
      (std::string(format) + "-r" + std::to_string(repeat) + "-p" +
       std::to_string(prompt) + "-" + arm);
  Write(prefix.string() + "-logits-f32.bin",
        std::string(reinterpret_cast<const char*>(result.logits.data()),
                    result.logits.size() * sizeof(float)));
  std::string ids;
  for (int32_t id : result.ids) ids += std::to_string(id) + "\n";
  Write(prefix.string() + "-tokens.txt", ids);
}

}  // namespace

TEST_CASE("ROCm compressed embeddings enter through public load and completion") {
  const std::vector<std::vector<int32_t>> prompts = {{1, 0, 63, 127, 63}, {1, 127, 0, 127}};
  for (const auto& format : rocm_gather_test::kFormats) {
    CAPTURE(std::string(format.name));
    const auto packed = rocm_gather_test::BuildModel(format, false);
    const auto dense = rocm_gather_test::BuildModel(format, true);
    REQUIRE(packed.size() < 8 * 1024 * 1024);
    REQUIRE(dense.size() < 8 * 1024 * 1024);
    CHECK(packed == rocm_gather_test::BuildModel(format, false));
    CHECK(dense == rocm_gather_test::BuildModel(format, true));
    gguf_test::TempFile packed_file(packed), dense_file(dense);
    std::vector<Completion> first;
    for (int repeat = 0; repeat < 3; ++repeat) {
      CAPTURE(repeat);
      for (size_t p = 0; p < prompts.size(); ++p) {
        CAPTURE(p);
        const auto quant_result = Run(packed_file.path(), prompts[p], true);
        const auto dense_result = Run(dense_file.path(), prompts[p], false);
        RecordCompletion(format.name, repeat, p, "quant", quant_result);
        RecordCompletion(format.name, repeat, p, "dense", dense_result);
        CHECK(quant_result.ids == dense_result.ids);
        REQUIRE(quant_result.logits.size() == dense_result.logits.size());
        CHECK(std::memcmp(quant_result.logits.data(), dense_result.logits.data(),
                          quant_result.logits.size() * sizeof(float)) == 0);
        if (repeat == 0) first.push_back(quant_result);
        else {
          CHECK(quant_result.ids == first[p].ids);
          CHECK(std::memcmp(quant_result.logits.data(), first[p].logits.data(),
                            quant_result.logits.size() * sizeof(float)) == 0);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--write-fixtures") {
    Export(argv[2]);
    return 0;
  }
#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA)
  try {
    vt::GetBackend(vt::DeviceType::kROCM);
  } catch (const std::runtime_error& error) {
    std::printf("SKIPPED: ROCm device unavailable: %s\n", error.what());
    return 77;
  }
  doctest::Context context(argc, argv);
  return context.run();
#else
  std::puts("SKIPPED: the public AUTO selection gate requires a HIP-only build");
  return 77;
#endif
}
