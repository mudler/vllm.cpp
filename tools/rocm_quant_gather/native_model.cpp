// Public-ABI capture for BACKEND-ROCM-QUANT-GATHER (#3093).
// The workload matches test_capi_rocm_embedding_quant. This runner accepts the
// original exported GGUF so native and oracle traces load identical bytes.
#include "vllm.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Capture {
  std::vector<float> logits;
  int calls = 0;
  bool invalid = false;
};

void Logits(const int32_t*, int32_t, float* values, int32_t count, void* opaque) {
  auto& capture = *static_cast<Capture*>(opaque);
  if (count != 128 || capture.calls >= 4) {
    capture.invalid = true;
    return;
  }
  for (int32_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) capture.invalid = true;
    capture.logits.push_back(values[i]);
  }
  ++capture.calls;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("MODEL_GGUF PROMPT_INDEX OUTPUT_PREFIX");
    const int index = std::stoi(argv[2]);
    if (index != 0 && index != 1) throw std::runtime_error("prompt index must be 0 or 1");
    const std::vector<int32_t> prompt = index == 0 ? std::vector<int32_t>{1, 0, 63, 127, 63}
                                                 : std::vector<int32_t>{1, 127, 0, 127};
    auto params = vllm_model_params_default();
    params.model_path = argv[1];
    params.device = 0;
    params.block_size = 16;
    params.num_blocks = 16;
    params.max_model_len = 64;
    params.max_num_seqs = 1;
    params.kv_cache_dtype = "auto";
    vllm_engine* raw = nullptr;
    if (vllm_engine_load(&params, &raw) != VLLM_OK) throw std::runtime_error(vllm_last_error());
    std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, &vllm_engine_free);
    Capture capture;
    capture.logits.reserve(512);
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0.0F;
    sampling.max_tokens = 4;
    sampling.ignore_eos = 1;
    sampling.has_seed = 1;
    sampling.seed = 0x524F434D;
    sampling.logits_processor = Logits;
    sampling.logits_processor_user_data = &capture;
    int32_t tokens[4]{};
    int32_t count = 0;
    if (vllm_complete_tokens(engine.get(), prompt.data(), static_cast<int32_t>(prompt.size()),
                             &sampling, tokens, 4, &count, nullptr) != VLLM_OK)
      throw std::runtime_error(vllm_last_error());
    if (count != 4 || capture.calls != 4 || capture.invalid)
      throw std::runtime_error("incomplete or nonfinite public completion");
    std::ofstream ids(std::string(argv[3]) + "-tokens.txt");
    for (const int32_t token : tokens) ids << token << '\n';
    std::ofstream logits(std::string(argv[3]) + "-logits-f32.bin", std::ios::binary);
    logits.write(reinterpret_cast<const char*>(capture.logits.data()),
                 static_cast<std::streamsize>(capture.logits.size() * sizeof(float)));
    if (!ids || !logits) throw std::runtime_error("capture write failed");
    std::printf("PUBLIC model=%s prompt=%d physical_blocks=16 block_size=16 logical_max=64 tokens=",
                argv[1], index);
    for (const int32_t token : tokens) std::printf(" %d", token);
    std::puts("");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "PUBLIC FAILED: %s\n", error.what());
    return 1;
  }
}
