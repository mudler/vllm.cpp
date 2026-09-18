// KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3: public Q4_K/Q6_K prefill reachability.
// A separate scalar process supplies complete logits for the identical GGUF.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/rocm_quant_gather_fixture.h"
#include "vllm.h"
#include "vt/backend.h"
#include "vt/ops.h"

#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA) && defined(__unix__)
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;

namespace vt::rocm {
std::string DeviceArchName(int);
uint64_t KQuantWmmaDispatchCount();
uint64_t KQuantWmmaQ4KDispatchCount();
}  // namespace vt::rocm

namespace {
using Json = nlohmann::json;
struct Completion {
  std::vector<float> logits;
  int callbacks = 0;
  bool invalid = false;
};

void Require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

void Capture(const int32_t*, int32_t, float* logits, int32_t vocab, void* data) {
  auto& capture = *static_cast<Completion*>(data);
  if (vocab != 128 || capture.callbacks >= 4) {
    capture.invalid = true;
    return;
  }
  for (int i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i])) capture.invalid = true;
    capture.logits.push_back(logits[i]);
  }
  ++capture.callbacks;
}

Json Run(bool expect_wmma) {
  const rocm_gather_test::Format q4{vt::DType::kQ4_K, 12, "Q4_K"};
  const rocm_gather_test::Format q6{vt::DType::kQ6_K, 14, "Q6_K"};
  gguf_test::TempFile model_file(rocm_gather_test::BuildModel(q4, true, &q4, &q6));
  Json result = Json::array();
  for (const int tokens : {16, 37}) {
    auto params = vllm_model_params_default();
    params.model_path = model_file.path().c_str();
    params.device = 0;
    params.block_size = 16;
    params.num_blocks = 16;
    params.max_model_len = 64;
    params.max_num_seqs = 1;
    vllm_engine* raw = nullptr;
    Require(vllm_engine_load(&params, &raw) == VLLM_OK,
            "public load: " + std::string(vllm_last_error()));
    std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
    std::vector<int32_t> prompt(static_cast<size_t>(tokens));
    for (int i = 0; i < tokens; ++i) prompt[static_cast<size_t>(i)] = (i * 19 + 1) % 128;
    Completion capture;
    capture.logits.reserve(4 * 128);
    std::vector<int32_t> ids(4);
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0.0F;
    sampling.max_tokens = 4;
    sampling.ignore_eos = 1;
    sampling.has_seed = 1;
    sampling.seed = rocm_gather_test::kSeed;
    sampling.logits_processor = Capture;
    sampling.logits_processor_user_data = &capture;
    int32_t count = 0;
    const auto q4_before = vt::rocm::KQuantWmmaQ4KDispatchCount();
    const auto q6_before = vt::rocm::KQuantWmmaDispatchCount();
    const auto reference_before = vt::GetReferenceTierHits();
    Require(vllm_complete_tokens(engine.get(), prompt.data(), tokens, &sampling,
                                 ids.data(), 4, &count, nullptr) == VLLM_OK,
            "public completion: " + std::string(vllm_last_error()));
    const auto q4_calls = vt::rocm::KQuantWmmaQ4KDispatchCount() - q4_before;
    const auto q6_calls = vt::rocm::KQuantWmmaDispatchCount() - q6_before;
    Require(count == 4 && capture.callbacks == 4 && !capture.invalid,
            "completion must return four tokens and finite full logits");
    Require(vt::GetReferenceTierHits() == reference_before, "unexpected CPU reference fallback");
    Require((q4_calls > 0) == expect_wmma, "Q4_K public WMMA dispatch mismatch");
    Require((q6_calls > 0) == expect_wmma, "Q6_K public WMMA dispatch mismatch");
    std::printf("PUBLIC prompt=%d q4_wmma=%llu q6_wmma=%llu tokens=", tokens,
                static_cast<unsigned long long>(q4_calls),
                static_cast<unsigned long long>(q6_calls));
    for (const auto id : ids) std::printf(" %d", id);
    std::puts("");
    result.push_back({{"prompt_tokens", tokens}, {"ids", ids}, {"logits", capture.logits}});
  }
  return result;
}

void ScalarProcess(const char* executable, const std::string& output) {
  // posix_spawn executes a fresh process because QuantWmmaEnabled caches the
  // environment on first use. The parent's GPU mutex also covers this child.
  std::vector<std::string> variables;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string value(*entry);
    if (value.rfind("VT_ROCM_QUANT_WMMA=", 0) != 0) variables.push_back(value);
  }
  variables.emplace_back("VT_ROCM_QUANT_WMMA=0");
  std::vector<char*> environment;
  for (auto& value : variables) environment.push_back(value.data());
  environment.push_back(nullptr);
  char* args[] = {const_cast<char*>(executable), const_cast<char*>("--scalar"),
                  const_cast<char*>(output.c_str()), nullptr};
  pid_t child = 0;
  Require(posix_spawnp(&child, executable, nullptr, nullptr, args, environment.data()) == 0,
          "cannot start scalar control");
  int status = 0;
  Require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "scalar control failed");
}
}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(VLLM_CPP_HIP) && !defined(VLLM_CPP_CUDA) && defined(__unix__)
  try {
    const std::string arch = vt::rocm::DeviceArchName(0);
    const auto stem = arch.substr(0, arch.find(':'));
    if (stem != "gfx1100" && stem != "gfx1200" && stem != "gfx1201") {
      std::printf("SKIPPED: physical gfx1100 or gfx1200/gfx1201 required, got %s\n", arch.c_str());
      return 77;
    }
    if (argc == 3 && std::string(argv[1]) == "--scalar") {
      const auto report = Run(false);
      std::ofstream file(argv[2]);
      file << report.dump(2) << '\n';
      Require(static_cast<bool>(file), "cannot write scalar result");
      return 0;
    }
    Require(argc == 1, "use no arguments or --scalar OUTPUT");
    const char* knob = std::getenv("VT_ROCM_QUANT_WMMA");
    Require(knob == nullptr || std::string(knob) != "0", "run the public gate with WMMA enabled");
    gguf_test::TempFile scalar_file("");
    ScalarProcess(argv[0], scalar_file.path());
    std::ifstream stream(scalar_file.path());
    const auto expected = Json::parse(stream);
    const auto actual = Run(true);
    Require(actual.size() == expected.size(), "scalar prompt count differs");
    float max_error = 0;
    for (size_t p = 0; p < actual.size(); ++p) {
      Require(actual[p]["ids"] == expected[p]["ids"], "scalar completion tokens differ");
      const auto got = actual[p]["logits"].get<std::vector<float>>();
      const auto ref = expected[p]["logits"].get<std::vector<float>>();
      Require(got.size() == ref.size(), "scalar logits shape differs");
      for (size_t i = 0; i < got.size(); ++i) {
        const float error = std::abs(got[i] - ref[i]);
        max_error = std::max(max_error, error);
        Require(error <= 0.02F + 0.01F * std::abs(ref[i]), "public logits differ from scalar control");
      }
    }
    std::printf("PUBLIC WMMA PASS: 1024 logits, token-exact, max_abs_error=%g\n", max_error);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "PUBLIC WMMA FAILED: %s\n", error.what());
    return 1;
  }
#else
  (void)argc;
  (void)argv;
  std::puts("SKIPPED: the public AUTO selection gate requires a HIP-only Unix build");
  return 77;
#endif
}
