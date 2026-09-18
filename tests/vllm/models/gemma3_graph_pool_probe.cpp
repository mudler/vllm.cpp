// Production graph ownership probe. The model runs only through the public ABI.
// An independent pool user holds canaries across subsequent graph replays.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "vllm.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vt/backend.h"

using Json = nlohmann::json;

class PoolCanaries {
 public:
  explicit PoolCanaries(vt::Backend& backend)
      : backend_(backend), pool_(vllm::ActivePool(backend)), queue_(backend.CreateQueue()) {
    try {
      for (auto [bytes, count] : pool_.StepDemandProfile()) {
        std::vector<unsigned char> fill(bytes, 0xa5);
        for (int64_t i = 0; i < count; ++i) {
          void* data = pool_.Get(backend_, bytes);
          blocks_.emplace_back(bytes, data);
          backend_.Copy(queue_, data, fill.data(), bytes);
          backend_.Synchronize(queue_);
        }
      }
      if (blocks_.empty()) throw std::runtime_error("no production decode pool demand");
    } catch (...) {
      Release();
      throw;
    }
  }
  ~PoolCanaries() { Release(); }
  void Check() {
    for (auto [bytes, data] : blocks_) {
      std::vector<unsigned char> actual(bytes);
      backend_.Copy(queue_, actual.data(), data, bytes);
      backend_.Synchronize(queue_);
      if (!std::all_of(actual.begin(), actual.end(), [](unsigned char x) { return x == 0xa5; }))
        throw std::runtime_error("graph overwrote a live pool allocation, class " +
                                 std::to_string(bytes));
    }
  }
  size_t Count() const { return blocks_.size(); }

 private:
  void Release() {
    backend_.Synchronize(queue_);
    for (auto [bytes, data] : blocks_) pool_.Put(backend_, bytes, data);
    blocks_.clear();
    backend_.DestroyQueue(queue_);
  }
  vt::Backend& backend_;
  vllm::DevicePool& pool_;
  vt::Queue queue_;
  std::vector<std::pair<size_t, void*>> blocks_;
};

int main(int argc, char** argv) {
  vllm_engine* engine = nullptr;
  try {
    if (argc != 5) throw std::runtime_error("use MODEL MANIFEST PRIMARY OUTPUT");
    Json manifest, primary;
    std::ifstream(argv[2]) >> manifest;
    std::ifstream(argv[3]) >> primary;
    auto model = vllm_model_params_default();
    model.model_path = argv[1];
    model.device = 0;
    model.max_num_seqs = 1;
    model.max_model_len = 4096;
    model.block_size = manifest.value("block_size", 16);
    model.num_blocks = 4096 / model.block_size;
    model.gpu_memory_utilization = 0.;
    if (vllm_engine_load(&model, &engine) != VLLM_OK) throw std::runtime_error(vllm_last_error());
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0.;
    sampling.max_tokens = manifest.at("output_len");
    sampling.ignore_eos = 1;
    Json result = {{"cases", Json::array()}};
    std::unique_ptr<PoolCanaries> canaries;
    for (size_t i = 0; i < manifest.at("cases").size(); ++i) {
      const auto& entry = manifest.at("cases").at(i);
      auto prompt = entry.at("prompt_token_ids").get<std::vector<int32_t>>();
      std::vector<int32_t> tokens(static_cast<size_t>(sampling.max_tokens));
      int32_t count = 0;
      vllm_completion completion{};
      auto status = vllm_complete_tokens(engine, prompt.data(), prompt.size(), &sampling,
                                         tokens.data(), tokens.size(), &count, &completion);
      vllm_completion_free(&completion);
      if (status != VLLM_OK) throw std::runtime_error(vllm_last_error());
      tokens.resize(static_cast<size_t>(count));
      if (tokens != primary.at("cases").at(i).at("output_token_ids").get<std::vector<int32_t>>())
        throw std::runtime_error("primary output tokens differ");
      if (canaries) canaries->Check();
      if (!canaries)
        canaries =
            std::make_unique<PoolCanaries>(vt::GetBackend(vt::Device{vt::DeviceType::kROCM, 0}));
      result["cases"].push_back({{"name", entry.at("name")},
                                 {"token_exact", count},
                                 {"intact_live_allocations", canaries->Count()}});
    }
    canaries.reset();
    vllm_engine_free(engine);
    engine = nullptr;
    std::ofstream(argv[4]) << result.dump(2) << '\n';
    std::puts("PUBLIC GRAPH POOL CANARIES AND TOKENS PASS");
    return 0;
  } catch (const std::exception& error) {
    if (engine) vllm_engine_free(engine);
    std::fprintf(stderr, "GRAPH POOL PROBE FAILED: %s\n", error.what());
    return 1;
  }
}
