// Public regression for ENG-QWEN35-FULL-ATTN-STATE (#3098).
// Pinned vLLM e126687a9a: qwen3_5.py:144-160 constructs recurrent state
// consumers only for linear_attention layers. This no-GDN case is local.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "capi/qwen35_full_attn_fixture.h"
#include "vllm.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA)
namespace vt::rocm {
void EmbeddingKernelRocm(Queue&, Tensor&, const Tensor&, const Tensor&);
void ReshapeAndCacheKernelRocm(Queue&, const Tensor&, const Tensor&, Tensor&,
                             Tensor&, const Tensor&);
void GreedyArgmaxKernelRocm(Queue&, Tensor&, const Tensor&);
}  // namespace vt::rocm
#endif

namespace {

std::atomic<int> native_prefills{0}, native_decodes{0}, native_cache_writes{0};
std::atomic<int> native_samples{0};
std::atomic<bool> wrong_model_dtype{false};
constexpr const char* kObserver = "qwen35-full-attn-state-observer";

#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA)
void ObserveEmbedding(vt::Queue& queue, vt::Tensor& out,
                      const vt::Tensor& table, const vt::Tensor& ids) {
  if (queue.device.type != vt::DeviceType::kROCM ||
      out.dtype != vt::DType::kBF16 || table.dtype != vt::DType::kBF16)
    wrong_model_dtype = true;
  // Call the verified native function, then count the completed native call.
  // RegisterOp cannot replace vt-native: duplicate provider names are ignored.
  vt::rocm::EmbeddingKernelRocm(queue, out, table, ids);
  if (ids.Numel() > 1) ++native_prefills;
  else ++native_decodes;
}

void ObserveCache(vt::Queue& queue, const vt::Tensor& key,
                  const vt::Tensor& value, vt::Tensor& key_cache,
                  vt::Tensor& value_cache, const vt::Tensor& slots) {
  if (key.dtype != vt::DType::kBF16 || value.dtype != vt::DType::kBF16 ||
      key_cache.dtype != vt::DType::kBF16 || value_cache.dtype != vt::DType::kBF16)
    wrong_model_dtype = true;
  vt::rocm::ReshapeAndCacheKernelRocm(queue, key, value, key_cache, value_cache, slots);
  ++native_cache_writes;
}

void ObserveGreedy(vt::Queue& queue, vt::Tensor& ids, const vt::Tensor& logits) {
  // The existing sampler ABI consumes f32 logits and produces i64 token IDs.
  if (queue.device.type != vt::DeviceType::kROCM ||
      logits.device.type != vt::DeviceType::kROCM ||
      ids.device.type != vt::DeviceType::kROCM ||
      logits.dtype != vt::DType::kF32 || ids.dtype != vt::DType::kI64)
    wrong_model_dtype = true;
  vt::rocm::GreedyArgmaxKernelRocm(queue, ids, logits);
  ++native_samples;
}
#endif

void InstallNativeObservers() {
#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA)
  const auto install = [](vt::OpId op, void* native, void* observer) {
    REQUIRE(vt::GetOp(op, vt::DeviceType::kROCM) == native);
    const auto before = vt::GetOpProviderStats(op, vt::DeviceType::kROCM);
    REQUIRE(before.last_selected != nullptr);
    REQUIRE(std::string(before.last_selected) == vt::kNativeProviderName);
    vt::OpProvider provider;
    provider.name = kObserver;
    provider.priority = 1;
    provider.fn = observer;
    vt::RegisterOpProvider(op, vt::DeviceType::kROCM, provider);
    REQUIRE(vt::GetOp(op, vt::DeviceType::kROCM) == observer);
  };
  install(vt::OpId::kEmbedding,
          reinterpret_cast<void*>(&vt::rocm::EmbeddingKernelRocm),
          reinterpret_cast<void*>(&ObserveEmbedding));
  install(vt::OpId::kReshapeAndCache,
          reinterpret_cast<void*>(&vt::rocm::ReshapeAndCacheKernelRocm),
          reinterpret_cast<void*>(&ObserveCache));
  install(vt::OpId::kGreedyArgmax,
          reinterpret_cast<void*>(&vt::rocm::GreedyArgmaxKernelRocm),
          reinterpret_cast<void*>(&ObserveGreedy));
#else
  throw std::runtime_error("native observers require a HIP-only build");
#endif
}

struct Completion {
  std::vector<int32_t> ids;
  // The public logits-processor ABI supplies f32 logits. This is host evidence,
  // not an activation or KV buffer in the model.
  std::vector<float> logits;
  bool capture_overflow = false;
  int callbacks = 0;
};

// Reserve the full callback output before entering the C ABI. This callback
// cannot throw or change the logits that the greedy sampler reads.
void Capture(const int32_t*, int32_t, float* logits, int32_t vocab, void* data) {
  auto& result = *static_cast<Completion*>(data);
  if (vocab != 128 || result.callbacks >= 4) {
    result.capture_overflow = true;
    return;
  }
  for (int32_t i = 0; i < vocab; ++i) result.logits.push_back(logits[i]);
  ++result.callbacks;
}

Completion Run(const std::string& path, const std::vector<int32_t>& prompt,
               int repeat, size_t prompt_index, bool capture_logits) {
  auto model = vllm_model_params_default();
  model.model_path = path.c_str();
  model.device = 0;  // Public AUTO in the required HIP-only build.
  model.block_size = 16;
  model.num_blocks = 4;
  model.max_model_len = 64;
  model.max_num_seqs = 1;
  model.kv_cache_dtype = "auto";
  vllm_engine* raw = nullptr;
  const vllm_status loaded = vllm_engine_load(&model, &raw);
  INFO("public load: " << vllm_last_error());
  REQUIRE(loaded == VLLM_OK);
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, &vllm_engine_free);
  std::printf("FULL_ATTN public load succeeded repeat=%d prompt=%zu\n", repeat, prompt_index);
  std::fflush(stdout);

  Completion result;
  result.ids.resize(4);
  result.logits.reserve(4 * 128);
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0.0F;
  sampling.max_tokens = 4;
  sampling.ignore_eos = 1;
  sampling.has_seed = 1;
  sampling.seed = qwen35_full_attn_test::kSeed;
  if (capture_logits) {
    sampling.logits_processor = Capture;
    sampling.logits_processor_user_data = &result;
  }
  native_prefills = 0;
  native_decodes = 0;
  native_cache_writes = 0;
  native_samples = 0;
  wrong_model_dtype = false;
  const auto hits_before = vt::GetReferenceTierHits();
  vt::ResetOpProviderStats(vt::OpId::kEmbedding, vt::DeviceType::kROCM);
  vt::EnableOpProviderCallStats(true);
  int32_t count = 0;
  const vllm_status completed = vllm_complete_tokens(
      engine.get(), prompt.data(), static_cast<int32_t>(prompt.size()), &sampling,
      result.ids.data(), 4, &count, nullptr);
  vt::EnableOpProviderCallStats(false);
  INFO("public completion: " << vllm_last_error());
  REQUIRE(completed == VLLM_OK);
  REQUIRE(count == 4);
  REQUIRE_FALSE(result.capture_overflow);
  REQUIRE(result.callbacks == (capture_logits ? 4 : 0));
  REQUIRE(result.logits.size() == (capture_logits ? 512 : 0));
  for (float value : result.logits) CHECK(std::isfinite(value));
  for (int32_t id : result.ids) CHECK((id >= 0 && id < 128));
  CHECK(native_prefills.load() == 1);
  CHECK(native_decodes.load() == 3);
  CHECK(native_cache_writes.load() == 4);
  CHECK(native_samples.load() == 4);
  CHECK_FALSE(wrong_model_dtype.load());
  CHECK(vt::GetReferenceTierHits() == hits_before);
  const auto stats = vt::GetOpProviderStats(vt::OpId::kEmbedding, vt::DeviceType::kROCM);
  CHECK(stats.selections >= 4);
  CHECK(stats.declines == 0);
  CHECK(stats.fallbacks == 0);
  REQUIRE(stats.last_selected != nullptr);
  CHECK(std::string(stats.last_selected) == kObserver);
  std::printf("FULL_ATTN repeat=%d prompt=%zu capture_logits=%d fresh_engine=1 concurrency=1 "
              "prompt_tokens=%zu native_prefills=%d native_decodes=%d "
              "native_cache_writes=%d native_samples=%d activation=bf16 kv=bf16 "
              "finite_logits=%zu tokens=",
              repeat, prompt_index, capture_logits, prompt.size(), native_prefills.load(),
              native_decodes.load(), native_cache_writes.load(), native_samples.load(),
              result.logits.size());
  for (int32_t id : result.ids) std::printf(" %d", id);
  std::printf("\n");
  return result;
}

void WriteFixture(const std::filesystem::path& path) {
  const auto bytes = qwen35_full_attn_test::BuildModel();
  if (bytes.size() != qwen35_full_attn_test::kArtifactBytes)
    throw std::runtime_error("full-attention fixture size changed");
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("cannot write full-attention fixture");
}

}  // namespace

TEST_CASE("Qwen3.5 full-attention state enters public load and completion") {
  const auto bytes = qwen35_full_attn_test::BuildModel();
  REQUIRE(bytes.size() == qwen35_full_attn_test::kArtifactBytes);
  CHECK(bytes == qwen35_full_attn_test::BuildModel());
  gguf_test::TempFile fixture(bytes);
  InstallNativeObservers();
  const std::vector<std::vector<int32_t>> prompts = {{1, 0, 63, 127, 63}, {1, 127, 0, 127}};
  std::vector<Completion> first(2 * prompts.size());
  // builtin.cpp::apply_logits_processors stages logits through the host when a
  // callback exists. Both modes still execute the native device argmax, but
  // the callback-free arm proves the default public route without that staging.
  for (bool capture_logits : {false, true}) {
    CAPTURE(capture_logits);
    for (int repeat = 0; repeat < 3; ++repeat) {
      CAPTURE(repeat);
      for (size_t p = 0; p < prompts.size(); ++p) {
        CAPTURE(p);
        const Completion result = Run(fixture.path(), prompts[p], repeat, p, capture_logits);
        Completion& baseline = first[static_cast<size_t>(capture_logits) * prompts.size() + p];
        if (repeat == 0) baseline = result;
        else {
          CHECK(result.ids == baseline.ids);
          REQUIRE(result.logits.size() == baseline.logits.size());
          if (capture_logits)
            CHECK(std::memcmp(result.logits.data(), baseline.logits.data(),
                              result.logits.size() * sizeof(float)) == 0);
        }
        if (capture_logits) CHECK(result.ids == first[p].ids);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--write-fixture") {
    WriteFixture(argv[2]);
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
  std::puts("SKIPPED: public AUTO regression requires a HIP-only build");
  return 77;
#endif
}
