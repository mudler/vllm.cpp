// Thin public-API benchmark clients. Pins and settings: qualification spec #3053.
#include <nlohmann/json.hpp>
#ifdef STRIX_LLAMA
#include <llama.h>
#else
#include <vllm.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
static json requested() {
  return {{"backend", "rocm"}, {"model_dtype", "bfloat16"}, {"kv_dtype", "bfloat16"},
          {"context_per_sequence", 2048}, {"slots", 4}, {"prefix_caching", false}};
}
static json record(int index, const std::vector<int32_t>& prompt, const std::vector<int32_t>& output,
                   double start, double stop) {
  return {{"index", index}, {"prompt_ids", prompt}, {"output_ids", output}, {"status", "ok"},
          {"finish_reason", "length"}, {"dispatched", start}, {"completed", stop}};
}

class Engine {
  std::vector<std::vector<int32_t>> prompts;
#ifdef STRIX_LLAMA
  llama_model* model = nullptr;
  llama_context* context = nullptr;
#else
  vllm_engine* engine = nullptr;
#endif
public:
  explicit Engine(const json& config) {
    require(config.at("prompt_ids").is_array(), "canonical prompt IDs required");
    prompts = config.at("prompt_ids").get<std::vector<std::vector<int32_t>>>();
    require(prompts.size() == 6, "six prompts required");
    for (const auto& prompt : prompts) {
      require(!prompt.empty() && prompt.size() + 128 <= 2048, "invalid prompt length");
      for (int token : prompt) require(token >= 0 && token < 151936, "invalid token ID");
    }
#ifdef STRIX_LLAMA
    require(config.at("engine") == "llama.cpp", "wrong native engine");
    llama_backend_init();
    require(llama_supports_gpu_offload(), "GPU offload unavailable");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    model = llama_model_load_from_file(config.at("gguf").get<std::string>().c_str(), mp);
    require(model != nullptr, "llama model load failed");
    try {
      const auto* vocab = llama_model_get_vocab(model);
      auto raw = config.at("prompts").get<std::vector<std::string>>();
      require(raw.size() == 6, "six raw prompts required");
      for (int i = 0; i < 6; ++i) {
        std::vector<llama_token> ids(2048);
        int n = llama_tokenize(vocab, raw[i].c_str(), static_cast<int32_t>(raw[i].size()),
                               ids.data(), static_cast<int32_t>(ids.size()), false, true);
        require(n > 0, "llama tokenization failed");
        ids.resize(n);
        require(ids == prompts[i], "llama tokenizer mismatch");
      }
      auto cp = llama_context_default_params();
      require(cp.n_batch == 2048 && cp.n_ubatch == 512 && cp.n_threads == 4 && cp.n_threads_batch == 4 &&
              cp.offload_kqv && cp.op_offload, "llama public-library defaults changed");
      cp.n_ctx = 8192;
      cp.n_seq_max = 4;
      cp.kv_unified = false;
      cp.type_k = GGML_TYPE_BF16;
      cp.type_v = GGML_TYPE_BF16;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
      context = llama_init_from_model(model, cp);
      require(context != nullptr, "llama context load failed");
      require(llama_n_ctx_seq(context) == 2048, "llama per-sequence capacity mismatch");
    } catch (...) {
      if (context) llama_free(context);
      llama_model_free(model);
      throw;
    }
#else
    require(config.at("engine") == "vllm.cpp", "wrong native engine");
    auto mp = vllm_model_params_default();
    std::string path = config.at("model");
    mp.model_path = path.c_str();
    mp.max_model_len = 2048;
    mp.max_num_seqs = 4;
    mp.enable_prefix_caching = 2;
    mp.kv_cache_dtype = "bfloat16";
    // 256 blocks * 32 tokens supplies four independent 2048-token contexts.
    mp.block_size = 32;
    mp.num_blocks = 256;
    require(vllm_engine_load(&mp, &engine) == VLLM_OK, vllm_last_error());
#endif
  }
  ~Engine() {
#ifdef STRIX_LLAMA
    if (context) llama_free(context);
    if (model) llama_model_free(model);
    llama_backend_free();
#else
    if (engine) vllm_engine_free(engine);
#endif
  }
  json configuration() const {
    return {{"prompt_ids", prompts}, {"requested", requested()},
            {"runtime_info", "Requested configuration only; bound operator runtime evidence required"}};
  }
  json run(int concurrency) {
    require(concurrency == 1 || concurrency == 4, "concurrency must be 1 or 4");
    std::vector<json> results(6);
#ifdef STRIX_LLAMA
    struct Slot { int request = -1; int position = 0; int row = -1; double start = 0;
                  std::vector<int32_t> output; llama_sampler* sampler = nullptr; };
    std::vector<Slot> slots(concurrency);
    auto batch = llama_batch_init(2048, 0, 1);
    int next = 0, finished = 0;
    const double start = now();
    try {
      while (finished < 6) {
        batch.n_tokens = 0;
        for (int s = 0; s < concurrency; ++s) {
          auto& slot = slots[s];
          if (slot.request == -1 && next < 6) {
            require(llama_memory_seq_rm(llama_get_memory(context), s, -1, -1), "KV clear failed");
            slot.request = next++;
            slot.position = 0;
            slot.output.clear();
            slot.sampler = llama_sampler_init_greedy();
            require(slot.sampler != nullptr, "sampler allocation failed");
            slot.start = now();
          }
          if (slot.request == -1) continue;
          const auto& prompt = prompts[slot.request];
          const int count = slot.position == 0 ? static_cast<int>(prompt.size()) : 1;
          require(batch.n_tokens + count <= 2048, "batch capacity exceeded");
          for (int i = 0; i < count; ++i) {
            const int row = batch.n_tokens++;
            batch.token[row] = slot.position == 0 ? prompt[i] : slot.output.back();
            batch.pos[row] = slot.position + i;
            batch.n_seq_id[row] = 1;
            batch.seq_id[row][0] = s;
            batch.logits[row] = i == count - 1;
            if (i == count - 1) slot.row = row;
          }
          slot.position += count;
        }
        require(llama_decode(context, batch) == 0, "llama decode failed");
        // Every live row is sampled before the next decode can overwrite logits.
        for (int s = 0; s < concurrency; ++s) {
          auto& slot = slots[s];
          if (slot.request == -1) continue;
          const int32_t token = llama_sampler_sample(slot.sampler, context, slot.row);
          require(token >= 0 && token < 151936, "invalid sampled token");
          // sample() accepts once internally. EOS remains eligible and never stops this corpus.
          slot.output.push_back(token);
          if (slot.output.size() == 128) {
            results[slot.request] = record(slot.request, prompts[slot.request], slot.output, slot.start, now());
            require(llama_memory_seq_rm(llama_get_memory(context), s, -1, -1), "KV clear failed");
            llama_sampler_free(slot.sampler);
            slot.sampler = nullptr;
            slot.request = -1;
            ++finished;
          }
        }
      }
    } catch (...) {
      for (auto& slot : slots) if (slot.sampler) llama_sampler_free(slot.sampler);
      llama_batch_free(batch);
      throw;
    }
    const double stop = now();
    llama_batch_free(batch);
#else
    std::atomic<int> next{0};
    std::mutex errors_mutex;
    std::exception_ptr error;
    const double start = now();
    std::vector<std::thread> workers;
    for (int worker = 0; worker < concurrency; ++worker) {
      workers.emplace_back([&] {
        try {
          for (;;) {
            int index = next.fetch_add(1);
            if (index >= 6) break;
            auto sp = vllm_sampling_params_default();
            sp.temperature = 0;
            sp.top_p = 1;
            sp.max_tokens = 128;
            sp.min_tokens = 0;
            sp.ignore_eos = 1;
            std::vector<int32_t> output(128);
            int32_t count = 0;
            vllm_completion completion{};
            const double dispatched = now();
            auto status = vllm_complete_tokens(engine, prompts[index].data(),
                static_cast<int32_t>(prompts[index].size()), &sp, output.data(), 128, &count, &completion);
            const double completed = now();
            const bool valid = status == VLLM_OK && count == 128 && completion.completion_tokens == 128 &&
                completion.prompt_tokens == static_cast<int32_t>(prompts[index].size()) &&
                completion.finish_reason && std::string(completion.finish_reason) == "length";
            vllm_completion_free(&completion);
            require(valid, "vllm_complete_tokens failed or truncated");
            results[index] = record(index, prompts[index], output, dispatched, completed);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(errors_mutex);
          if (!error) error = std::current_exception();
        }
      });
    }
    for (auto& worker : workers) worker.join();
    const double stop = now();
    if (error) std::rethrow_exception(error);
#endif
    return {{"requests", results}, {"started", start}, {"completed", stop}};
  }
};

int main() {
  FILE* protocol = fdopen(dup(STDOUT_FILENO), "w");
  if (!protocol || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;
  std::unique_ptr<Engine> engine;
  int64_t identifier = 0;
  try {
    std::string line;
    while (std::getline(std::cin, line)) {
      require(line.size() <= 1 << 20, "command size limit exceeded");
      std::vector<std::set<std::string>> keys;
      auto command = json::parse(line, [&](int, json::parse_event_t event, json& item) {
        if (event == json::parse_event_t::object_start) keys.emplace_back();
        if (event == json::parse_event_t::key)
          require(keys.back().insert(item.get<std::string>()).second, "duplicate JSON key");
        if (event == json::parse_event_t::object_end) keys.pop_back();
        return true;
      });
      require(command.at("schema").is_number_integer() && command.at("schema") == 1 && command.at("id").is_number_integer() &&
              command.at("id").get<int64_t>() == identifier + 1, "command identity mismatch");
      ++identifier;
      const std::string action = command.at("command");
      json result;
      if (action == "configure" && !engine) {
        engine = std::make_unique<Engine>(command);
        result = engine->configuration();
      } else if (action == "run" && engine) {
        const std::string phase = command.at("phase");
        require((phase == "qualification" || phase == "warmup" || phase == "measured") &&
                command.at("repetition").is_number_integer(), "invalid run command");
        require(command.at("concurrency").is_number_integer(), "integer concurrency required");
        result = engine->run(command.at("concurrency"));
      } else if (action == "shutdown" && engine) {
        engine.reset();
        result = {{"schema", 1}, {"id", identifier}, {"status", "ok"}};
        fprintf(protocol, "%s\n", result.dump().c_str());
        fclose(protocol);
        return 0;
      } else throw std::runtime_error("out-of-order command");
      result["schema"] = 1;
      result["id"] = identifier;
      result["status"] = "ok";
      fprintf(protocol, "%s\n", result.dump().c_str());
      fflush(protocol);
    }
    throw std::runtime_error("protocol EOF before shutdown");
  } catch (const std::exception& error) {
    json result = {{"schema", 1}, {"id", identifier}, {"status", "error"}, {"error", error.what()}};
    fprintf(protocol, "%s\n", result.dump().c_str());
    fclose(protocol);
    return 1;
  }
}
