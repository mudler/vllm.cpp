// End-to-end token capture through the public load and completion ABI.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include "vllm.h"

using Json = nlohmann::json;

struct LogitCapture {
  std::filesystem::path directory;
  std::string name;
  bool failed = false;
};

void CaptureLogits(const int32_t*, int32_t step, float* logits, int32_t vocab, void* context) {
  auto& capture = *static_cast<LogitCapture*>(context);
  try {
    std::ofstream output(capture.directory / (capture.name + "-" + std::to_string(step) + ".f32"),
                         std::ios::binary);
    output.write(reinterpret_cast<const char*>(logits), static_cast<size_t>(vocab) * sizeof(float));
    capture.failed = capture.failed || !output;
  } catch (...) {
    capture.failed = true;
  }
}

int main(int argc, char** argv) {
  vllm_engine* engine = nullptr;
  try {
    if (argc != 4) throw std::runtime_error("use MODEL MANIFEST OUTPUT");
    std::ifstream input(argv[2]);
    Json manifest;
    input >> manifest;
    auto mp = vllm_model_params_default();
    mp.model_path = argv[1];
    mp.device = 0;
    mp.max_num_seqs = 1;
    mp.max_model_len = 4096;
    // The pinned primary's production block size is 16. A manifest can
    // request 32 to exercise the native API's separate default configuration.
    mp.block_size = manifest.value("block_size", 16);
    if (mp.block_size <= 0) throw std::runtime_error("block_size must be positive");
    mp.num_blocks = (mp.max_model_len + mp.block_size - 1) / mp.block_size;
    mp.gpu_memory_utilization = 0.;
    if (vllm_engine_load(&mp, &engine) != VLLM_OK)
      throw std::runtime_error(vllm_last_error());
    auto sp = vllm_sampling_params_default();
    sp.temperature = 0.;
    sp.max_tokens = manifest.at("output_len");
    sp.ignore_eos = 1;
    LogitCapture logits;
    if (const char* directory = std::getenv("VT_ATTN_WMMA_LOGITS_DIR")) {
      logits.directory = directory;
      std::filesystem::create_directories(logits.directory);
      sp.logits_processor = CaptureLogits;
      sp.logits_processor_user_data = &logits;
    }
    Json report = {{"block_size", mp.block_size}, {"cases", Json::array()}};
    for (const auto& entry : manifest.at("cases")) {
      logits.name = entry.at("name").get<std::string>();
      auto prompt = entry.at("prompt_token_ids").get<std::vector<int32_t>>();
      std::vector<int32_t> tokens(static_cast<size_t>(sp.max_tokens));
      int32_t count = 0;
      vllm_completion completion{};
      const auto start = std::chrono::steady_clock::now();
      const auto status = vllm_complete_tokens(engine, prompt.data(), prompt.size(),
          &sp, tokens.data(), tokens.size(), &count, &completion);
      const auto end = std::chrono::steady_clock::now();
      if (status != VLLM_OK) throw std::runtime_error(vllm_last_error());
      tokens.resize(static_cast<size_t>(count));
      const double seconds = std::chrono::duration<double>(end - start).count();
      report["cases"].push_back({{"name", entry.at("name")},
          {"prompt_token_ids", prompt}, {"output_token_ids", tokens},
          {"text", completion.text ? completion.text : ""}, {"seconds", seconds}});
      std::printf("PUBLIC %s prompt=%zu output=%d seconds=%.6f\n",
          entry.at("name").get<std::string>().c_str(), prompt.size(), count, seconds);
      std::fflush(stdout);
      vllm_completion_free(&completion);
      if (count != sp.max_tokens) throw std::runtime_error("short completion");
      if (logits.failed) throw std::runtime_error("cannot write diagnostic logits");
    }
    vllm_engine_free(engine);
    engine = nullptr;
    std::ofstream output(argv[3]);
    output << report.dump(2) << '\n';
    if (!output) throw std::runtime_error("cannot write output");
    return 0;
  } catch (const std::exception& error) {
    if (engine) vllm_engine_free(engine);
    std::fprintf(stderr, "PUBLIC FAILED: %s\n", error.what());
    return 1;
  }
}
