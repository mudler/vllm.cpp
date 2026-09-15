// Single-load, concurrency-one benchmark with an untimed warm-up request.
// Reuse the existing benchmark's sampling and metric helpers.
#include "../../examples/bench/bench_core.h"

int main(int argc, char** argv) {
  try {
    using namespace vllm::bench;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    if (argc != 4) throw std::runtime_error("use MODEL MANIFEST OUTPUT");
    std::ifstream input(argv[2]);
    Json manifest;
    input >> manifest;
    BenchConfig cfg;
    cfg.model_path = argv[1];
    cfg.output_len = manifest.at("output_len");
    cfg.concurrency = 1;
    cfg.temperature = 0.;
    cfg.ignore_eos = true;
    if (cfg.output_len < 2 || manifest.at("cases").empty())
      throw std::runtime_error("empty benchmark");
    vllm::entrypoints::EngineParams params;
    params.max_num_seqs = 1;
    params.max_model_len = 4096;
    params.max_num_batched_tokens = 4096;
    params.block_size = manifest.value("block_size", 16);  // Pinned vLLM production default.
    if (params.block_size <= 0) throw std::runtime_error("block_size must be positive");
    params.num_blocks = (4096 + params.block_size - 1) / params.block_size;
    auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(argv[1], params);
    auto& engine = loaded->async_engine();
    const auto epoch = Clock::now();
    auto now = [&] { return std::chrono::duration<double>(Clock::now() - epoch).count(); };
    auto run = [&](const Json& entry, int index) {
      RequestRecord rec;
      rec.prompt_token_ids = entry.at("prompt_token_ids").get<std::vector<int32_t>>();
      rec.arrival_s = now();
      auto request = engine.add_request(entry.at("name"), rec.prompt_token_ids, MakeSampling(cfg, index));
      while (!rec.finished) {
        auto out = engine.get_output(request);
        if (!out.prompt_token_ids.empty() && out.prompt_token_ids != rec.prompt_token_ids)
          throw std::runtime_error("engine prompt IDs differ");
        if (!out.outputs.empty() && !out.outputs[0].token_ids.empty()) {
          if (rec.first_token_s < 0.) rec.first_token_s = now();
          const auto& ids = out.outputs[0].token_ids;
          rec.output_token_ids.insert(rec.output_token_ids.end(), ids.begin(), ids.end());
        }
        if (out.finished) { rec.finished = true; rec.completion_s = now(); }
      }
      if (rec.output_token_ids.size() != static_cast<size_t>(cfg.output_len))
        throw std::runtime_error("incomplete completion");
      return rec;
    };
    const auto warmup = run(manifest.at("warmup"), 0);
    Json report = {{"cases", Json::array()}, {"warmup_output_token_ids", warmup.output_token_ids},
                   {"block_size", params.block_size},
                   {"async_scheduling_enabled", loaded->async_scheduling_enabled()},
                   {"resolved_kv_cache_dtype", ResolvedKvCacheDTypeName(loaded->kv_cache_config())}};
    std::vector<double> ttfts, tpots, e2els;
    int64_t inputs = 0;
    const double start = now();
    int index = 1;
    for (const auto& entry : manifest.at("cases")) {
      const auto rec = run(entry, index++);
      const double ttft = rec.first_token_s - rec.arrival_s;
      const double e2el = rec.completion_s - rec.arrival_s;
      ttfts.push_back(ttft * 1000.);
      tpots.push_back((e2el - ttft) * 1000. / (cfg.output_len - 1));
      e2els.push_back(e2el * 1000.);
      inputs += static_cast<int64_t>(rec.prompt_token_ids.size());
      report["cases"].push_back({{"name", entry.at("name")}, {"prompt_token_ids", rec.prompt_token_ids},
          {"output_token_ids", rec.output_token_ids}, {"ttft_ms", ttfts.back()},
          {"tpot_ms", tpots.back()}, {"e2el_ms", e2els.back()}});
    }
    report["duration_s"] = now() - start;
    report["prefill_tokens_per_second"] = inputs * 1000. / (Mean(ttfts) * ttfts.size());
    report["decode_tokens_per_second"] = 1000. / Mean(tpots);
    report["mean_ttft_ms"] = Mean(ttfts);
    report["median_ttft_ms"] = Percentile(ttfts, 50);
    report["mean_tpot_ms"] = Mean(tpots);
    report["mean_e2el_ms"] = Mean(e2els);
    report["median_e2el_ms"] = Percentile(e2els, 50);
    std::ofstream output(argv[3]);
    output << report.dump(2) << '\n';
    if (!output) throw std::runtime_error("cannot write benchmark");
    std::printf("BENCH %zu requests, %.3f prefill tokens/s, %.3f decode tokens/s\n",
        ttfts.size(), report["prefill_tokens_per_second"].get<double>(),
        report["decode_tokens_per_second"].get<double>());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "BENCH FAILED: %s\n", error.what());
    return 1;
  }
}
