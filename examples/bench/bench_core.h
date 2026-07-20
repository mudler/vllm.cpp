// vllm.cpp benchmark core (M2.1). ORIGINAL tool (no upstream 1:1 mirror), but the
// METRICS and their report format deliberately mirror vLLM's `vllm bench serve`
// (vllm/benchmarks/serve.py @ the pinned oracle, class BenchmarkMetrics +
// benchmark()/save-and-print path) and `vllm bench throughput`
// (vllm/benchmarks/throughput.py) so the numbers this prints are directly
// comparable to a vLLM run on the same box + same model + same workload. See
// examples/bench/main.cpp for the CLI and the dgx invocation.
//
// Mirrored from serve.py:
//   - request_throughput = completed / dur_s               (serve.py:731)
//   - output_throughput  = sum(output_lens) / dur_s        (serve.py:733)
//   - total_token_throughput = (in+out) / dur_s            (serve.py:734)
//   - ttft: time-to-first-token per request                (serve.py:615)
//   - tpot: (latency - ttft) / (output_len - 1)            (serve.py:609-611)
//   - itl:  inter-token latency per streamed chunk         (serve.py:614)
//   - e2el: end-to-end per-request latency                 (serve.py:616)
//   mean/median/std/percentile reduction over each         (serve.py:726-758)
// Our ADDITION (labeled separately): a prefill-vs-decode token-throughput split
// — Input(prefill) tok/s = total_input/dur, Output(decode) tok/s =
// total_output/dur (== serve.py output_throughput) — which is exactly the
// gate #1 measurement ("prefill AND decode throughput at large concurrency",
// .agents/gates.md). We drive the engine in DELTA output mode (like a streaming
// serve client) so TTFT/ITL are observed the same way serve.py observes them.
//
// This header is deliberately header-only + inline so both the CLI
// (examples/bench/main.cpp) and the ctest smoke (tests/examples/test_bench.cpp)
// compile the exact same measurement path.
#ifndef VLLM_EXAMPLES_BENCH_BENCH_CORE_H_
#define VLLM_EXAMPLES_BENCH_BENCH_CORE_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/outputs.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/llm_engine.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::bench {

// ── Config (mirrors the `vllm bench throughput` / `serve` knobs we support) ────
struct BenchConfig {
  // Empty => SYNTHETIC tiny CPU engine (smoke). Otherwise a model dir / .gguf.
  std::string model_path;
  int num_prompts = 8;     // N: total requests to submit.
  int input_len = 16;      // L: target prompt tokens per request.
  int output_len = 16;     // O: max_tokens per request (greedy => exactly O).
  int concurrency = 4;     // C: max in-flight requests admitted to the engine.
  uint64_t seed = 0;       // prompt-generation RNG seed.
  double temperature = 0;  // <= 0 => greedy (deterministic).
  bool quiet = false;      // suppress per-progress logging to stderr.
  // Per-step token budget (chunked-prefill knob). 0 => the engine's bounded
  // default (ResolveMaxNumBatchedTokens). Exposed so the GB10 memory ramp can
  // bound the per-step GDN prefill activation explicitly (the 27B 8x1024 OOM
  // fix): a smaller budget splits a big prefill batch across more steps.
  int max_num_batched_tokens = 0;
  // KV/GDN-state cache blocks to preallocate. 0 => the heuristic below
  // (concurrency * seq_blocks * 2). Exposed for the GB10 memory ramp: the
  // preallocated f32 KV + GDN mamba-state cache scales with num_blocks and is the
  // dominant unified-memory consumer at high concurrency, SEPARATE from the
  // (chunked-prefill-bounded) per-step activation. Setting it to just enough for
  // C concurrent (input+output)-long sequences keeps peak RAM bounded.
  int num_blocks = 0;
};

// ── Per-request timing record (client-side, exactly what serve.py records). ────
struct RequestRecord {
  double arrival_s = 0.0;       // when we admitted it to the engine.
  double first_token_s = -1.0;  // when its first output token was observed.
  double completion_s = 0.0;    // when it finished.
  double last_token_s = 0.0;    // running: previous token arrival (for ITL).
  int prompt_tokens = 0;
  int output_tokens = 0;
  std::vector<double> itls;  // inter-token latencies (s), one per chunk>1st.
  bool finished = false;
};

// ── Aggregated result (field names mirror serve.py BenchmarkMetrics). ──────────
struct BenchResult {
  int completed = 0;
  double duration_s = 0.0;
  int64_t total_input = 0;
  int64_t total_output = 0;
  double request_throughput = 0.0;       // req/s
  double output_throughput = 0.0;        // tok/s  (decode)
  double input_throughput = 0.0;         // tok/s  (prefill) — our split
  double total_token_throughput = 0.0;   // tok/s
  double mean_per_stream_decode = 0.0;   // 1000/mean_tpot, tok/s per stream
  // ms statistics (mean/median/p99), like serve.py.
  double mean_ttft_ms = 0, median_ttft_ms = 0, p99_ttft_ms = 0;
  double mean_tpot_ms = 0, median_tpot_ms = 0, p99_tpot_ms = 0;
  double mean_itl_ms = 0, median_itl_ms = 0, p99_itl_ms = 0;
  double mean_e2el_ms = 0, median_e2el_ms = 0, p99_e2el_ms = 0;
};

// ── numpy-style linear-interpolation percentile (matches np.percentile). ───────
inline double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  if (v.size() == 1) return v.front();
  const double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return v[lo] + (v[hi] - v[lo]) * frac;
}

inline double Mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

// ────────────────────────────── Synthetic model ───────────────────────────────
// A tiny hybrid-MoE Qwen3.6 (mirrors tests/vllm/v1/test_llm_engine.cpp) so the
// harness runs end-to-end on the CPU box with no checkpoint. The NUMBERS from
// this engine are meaningless (toy weights); it proves the harness drives the
// real V1 engine loop + produces sane metrics. Real numbers come from a GB10
// run with --model.
namespace detail {

inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
inline float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
inline OwnedTensor MakeOwned(vt::DType dt, std::vector<int64_t> shape,
                             uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == vt::DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

constexpr int kVocab = 24;  // == the tiny BPE fixture's assigned ids (0..23).

inline HfConfig MakeSyntheticConfig(int max_model_len) {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = kVocab;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = max_model_len;
  c.raw = nlohmann::json::object();  // no eos => runs to max_tokens.
  return c;
}

inline vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(vt::DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(vt::DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(vt::DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(vt::DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(vt::DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(vt::DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(vt::DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(vt::DType::kBF16, {Is, H}, s + 5);
  return m;
}

inline vllm::Qwen3_5MoeWeights MakeSyntheticWeights(const HfConfig& c) {
  vllm::Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(vt::DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(vt::DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(vt::DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(vt::DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(vt::DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(vt::DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(vt::DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(vt::DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(vt::DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(vt::DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(vt::DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(vt::DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(vt::DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(vt::DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(vt::DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(vt::DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(vt::DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(vt::DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(vt::DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(vt::DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The tiny oracle-verified BPE fixture (ids 0..23, no holes) from
// tests/vllm/v1/test_llm_engine.cpp: "hello"=13, " world"=17, "1"=8, "2"=9, ...
inline tok::Tokenizer BuildSyntheticTokenizer() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_bench_tok_" + std::to_string(counter++) + ".json"))
          .string();
  nlohmann::json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = nlohmann::json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       nlohmann::json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  nlohmann::json vocab = {
      {"h", 0},   {"e", 1},   {"l", 2},    {"o", 3},     {"w", 4},
      {"r", 5},   {"d", 6},   {"Ġ", 7}, {"1", 8},   {"2", 9},
      {"ll", 10}, {"he", 11}, {"llo", 12}, {"hello", 13}, {"Ġw", 14},
      {"or", 15}, {"orld", 16}, {"Ġworld", 17}, {"ld", 18}};
  vocab[tok::MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[tok::MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       nlohmann::json::array(
           {nlohmann::json::array({"l", "l"}), nlohmann::json::array({"h", "e"}),
            nlohmann::json::array({"ll", "o"}),
            nlohmann::json::array({"he", "llo"}),
            nlohmann::json::array({"Ġ", "w"}),
            nlohmann::json::array({"o", "r"}), nlohmann::json::array({"l", "d"}),
            nlohmann::json::array({"or", "ld"}),
            nlohmann::json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  tok::Tokenizer t = tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return t;
}

// Build a prompt string that tokenizes to ~target tokens under `tok`. We can't
// hand the engine raw token ids (add_request takes text), so vLLM's exact
// random-token input is approximated with a repeated-word filler; the harness
// reports the MEASURED tokenized counts (from RequestOutput.prompt_token_ids),
// so throughput stays honest regardless of the small over/undershoot.
inline std::string BuildPrompt(const tok::Tokenizer& t, int target,
                               uint64_t seed) {
  // NOTE: for the SYNTHETIC engine, only bytes in the tiny fixture's alphabet
  // (h e l o w r d and digits 1 2) are encodable; a real tokenizer accepts any.
  // These filler words stay within that alphabet so both paths share this code.
  static const char* kWords[] = {"hello", "world", "1", "2"};
  constexpr size_t kNumWords = sizeof(kWords) / sizeof(kWords[0]);
  std::mt19937_64 rng(seed);
  std::string p;
  // Append a few words at a time, re-checking length, to keep this ~linear.
  while (static_cast<int>(t.Encode(p).size()) < target) {
    for (int k = 0; k < 4 && static_cast<int>(t.Encode(p).size()) < target;
         ++k) {
      if (!p.empty()) p += ' ';
      p += kWords[rng() % kNumWords];
    }
  }
  return p;
}

}  // namespace detail

// Build a SamplingParams for one bench request (greedy unless temperature>0).
inline SamplingParams MakeSampling(const BenchConfig& cfg, int req_index) {
  SamplingParams sp;
  sp.temperature = cfg.temperature;  // <= 0 => greedy.
  sp.max_tokens = cfg.output_len;
  // Fixed-length workload: generate EXACTLY output_len tokens (never stop early
  // on EOS), so throughput/latency are measured on the intended token budget and
  // match `vllm bench serve --ignore-eos` apples-to-apples. This makes the
  // harness's documented "greedy => exactly O" contract actually hold.
  sp.ignore_eos = true;
  sp.output_kind = RequestOutputKind::kDelta;  // observe TTFT/ITL like a client.
  if (cfg.temperature > 0.0) {
    sp.seed = static_cast<int64_t>(cfg.seed + static_cast<uint64_t>(req_index));
  }
  return sp;
}

// ─────────────────────────────── The harness ──────────────────────────────────
// Creates the engine (synthetic if cfg.model_path is empty, else loaded from the
// dir/.gguf), builds N prompts, then drives the V1 engine step() loop admitting
// up to C requests at a time until all N finish — timing everything with
// steady_clock. Returns the aggregated metrics.
inline BenchResult RunBench(const BenchConfig& cfg) {
  using Clock = std::chrono::steady_clock;

  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded;
  std::vector<std::string> prompts;
  prompts.reserve(static_cast<size_t>(cfg.num_prompts));

  if (cfg.model_path.empty()) {
    // Synthetic: build tokenizer first (to size the engine + measure prompts),
    // then move it into the LoadedEngine.
    tok::Tokenizer tok = detail::BuildSyntheticTokenizer();
    int max_prompt = 1;
    for (int i = 0; i < cfg.num_prompts; ++i) {
      prompts.push_back(detail::BuildPrompt(
          tok, cfg.input_len, cfg.seed + static_cast<uint64_t>(i)));
      max_prompt = std::max(
          max_prompt, static_cast<int>(tok.Encode(prompts.back()).size()));
    }
    const int seq_budget = max_prompt + cfg.output_len + 4;
    vllm::entrypoints::EngineParams params;
    params.block_size = seq_budget;   // unified block (hybrid-KV constraint).
    params.max_model_len = seq_budget;
    params.max_num_seqs = std::max(cfg.concurrency, 1);
    params.num_blocks = std::max(cfg.concurrency * 4, 16);
    HfConfig c = detail::MakeSyntheticConfig(seq_budget);
    vllm::Qwen3_5MoeWeights w = detail::MakeSyntheticWeights(c);
    loaded = std::make_unique<vllm::entrypoints::LoadedEngine>(
        std::move(c), std::move(w), std::move(tok), params);
  } else {
    vllm::entrypoints::EngineParams params;
    params.max_num_seqs = std::max(cfg.concurrency, 1);
    // Real checkpoint: max_model_len comes from config; blocks sized for C
    // concurrent (input+output)-long sequences at the default block size.
    params.block_size = 32;
    const int seq_blocks = (cfg.input_len + cfg.output_len + 31) / 32 + 1;
    params.num_blocks = cfg.num_blocks > 0
                            ? cfg.num_blocks
                            : std::max(cfg.concurrency * seq_blocks * 2, 256);
    // Chunked-prefill per-step budget (0 => engine bounded default).
    params.max_num_batched_tokens = cfg.max_num_batched_tokens;
    loaded = vllm::entrypoints::LoadedEngine::FromModelDir(cfg.model_path, params);
    for (int i = 0; i < cfg.num_prompts; ++i) {
      prompts.push_back(detail::BuildPrompt(
          loaded->tokenizer(), cfg.input_len, cfg.seed + static_cast<uint64_t>(i)));
    }
  }

  vllm::v1::LLMEngine& engine = loaded->engine();

  std::map<std::string, RequestRecord> records;
  const Clock::time_point t0 = Clock::now();
  auto now_s = [&]() {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };

  int next = 0;       // next prompt index to submit.
  int in_flight = 0;  // requests currently admitted + unfinished.
  int done = 0;

  auto admit = [&]() {
    while (next < cfg.num_prompts && in_flight < cfg.concurrency) {
      const std::string rid = std::to_string(next);
      RequestRecord rec;
      rec.arrival_s = now_s();
      records[rid] = rec;
      engine.add_request(rid, prompts[static_cast<size_t>(next)],
                         MakeSampling(cfg, next));
      ++next;
      ++in_flight;
    }
  };

  admit();
  while (done < cfg.num_prompts) {
    for (RequestOutput& out : engine.step()) {
      RequestRecord& rec = records[out.request_id];
      if (rec.prompt_tokens == 0 && !out.prompt_token_ids.empty()) {
        rec.prompt_tokens = static_cast<int>(out.prompt_token_ids.size());
      }
      if (!out.outputs.empty() && !out.outputs[0].token_ids.empty()) {
        const double t = now_s();
        const int n_new = static_cast<int>(out.outputs[0].token_ids.size());
        if (rec.first_token_s < 0.0) {
          rec.first_token_s = t;  // TTFT reference.
        } else {
          rec.itls.push_back(t - rec.last_token_s);  // one ITL per chunk.
        }
        rec.last_token_s = t;
        rec.output_tokens += n_new;
      }
      if (out.finished && !rec.finished) {
        rec.finished = true;
        rec.completion_s = now_s();
        --in_flight;
        ++done;
      }
    }
    admit();  // keep C in flight as requests finish.
  }

  const double dur_s = now_s();

  // ── Reduce to aggregate metrics (serve.py:726-758 semantics). ────────────────
  std::vector<double> ttfts, tpots, itls, e2els;
  int64_t total_in = 0, total_out = 0;
  double sum_prefill = 0.0, sum_decode = 0.0;
  int64_t decode_tokens = 0;
  for (const auto& kv : records) {
    const RequestRecord& r = kv.second;
    total_in += r.prompt_tokens;
    total_out += r.output_tokens;
    const double ttft = std::max(0.0, r.first_token_s - r.arrival_s);
    const double e2el = std::max(0.0, r.completion_s - r.arrival_s);
    ttfts.push_back(ttft);
    e2els.push_back(e2el);
    if (r.output_tokens > 1) {
      tpots.push_back((e2el - ttft) / static_cast<double>(r.output_tokens - 1));
      sum_decode += (e2el - ttft);
      decode_tokens += (r.output_tokens - 1);
    }
    sum_prefill += ttft;
    for (double x : r.itls) itls.push_back(x);
  }

  BenchResult res;
  res.completed = done;
  res.duration_s = dur_s;
  res.total_input = total_in;
  res.total_output = total_out;
  res.request_throughput = dur_s > 0 ? static_cast<double>(done) / dur_s : 0.0;
  res.output_throughput =
      dur_s > 0 ? static_cast<double>(total_out) / dur_s : 0.0;
  res.input_throughput = dur_s > 0 ? static_cast<double>(total_in) / dur_s : 0.0;
  res.total_token_throughput =
      dur_s > 0 ? static_cast<double>(total_in + total_out) / dur_s : 0.0;
  (void)sum_prefill;
  const double mean_tpot = Mean(tpots);
  res.mean_per_stream_decode = mean_tpot > 0 ? 1.0 / mean_tpot : 0.0;
  (void)sum_decode;
  (void)decode_tokens;

  res.mean_ttft_ms = Mean(ttfts) * 1000.0;
  res.median_ttft_ms = Percentile(ttfts, 50) * 1000.0;
  res.p99_ttft_ms = Percentile(ttfts, 99) * 1000.0;
  res.mean_tpot_ms = mean_tpot * 1000.0;
  res.median_tpot_ms = Percentile(tpots, 50) * 1000.0;
  res.p99_tpot_ms = Percentile(tpots, 99) * 1000.0;
  res.mean_itl_ms = Mean(itls) * 1000.0;
  res.median_itl_ms = Percentile(itls, 50) * 1000.0;
  res.p99_itl_ms = Percentile(itls, 99) * 1000.0;
  res.mean_e2el_ms = Mean(e2els) * 1000.0;
  res.median_e2el_ms = Percentile(e2els, 50) * 1000.0;
  res.p99_e2el_ms = Percentile(e2els, 99) * 1000.0;
  return res;
}

// Print the summary table, mirroring serve.py's "Serving Benchmark Result"
// block + our prefill/decode split. Fixed-width columns like serve.py
// ("{:<40} {:<10.2f}").
inline void PrintReport(const BenchConfig& cfg, const BenchResult& r,
                        std::FILE* out) {
  auto line_i = [&](const char* label, long long v) {
    std::fprintf(out, "%-42s %-12lld\n", label, v);
  };
  auto line_f = [&](const char* label, double v) {
    std::fprintf(out, "%-42s %-12.2f\n", label, v);
  };
  auto sep = [&](const char* title) {
    std::fprintf(out, "%.*s %s %.*s\n", 8,
                 "----------------------------------------", title, 8,
                 "----------------------------------------");
  };

  std::fprintf(out, "\n============= vllm.cpp Benchmark Result =============\n");
  line_i("Successful requests:", r.completed);
  line_i("Maximum request concurrency:", cfg.concurrency);
  line_f("Benchmark duration (s):", r.duration_s);
  line_i("Total input tokens:", static_cast<long long>(r.total_input));
  line_i("Total generated tokens:", static_cast<long long>(r.total_output));
  line_f("Request throughput (req/s):", r.request_throughput);
  line_f("Output token throughput (tok/s):", r.output_throughput);
  line_f("Total token throughput (tok/s):", r.total_token_throughput);
  sep("Time to First Token");
  line_f("Mean TTFT (ms):", r.mean_ttft_ms);
  line_f("Median TTFT (ms):", r.median_ttft_ms);
  line_f("P99 TTFT (ms):", r.p99_ttft_ms);
  sep("Time per Output Token (excl. 1st token)");
  line_f("Mean TPOT (ms):", r.mean_tpot_ms);
  line_f("Median TPOT (ms):", r.median_tpot_ms);
  line_f("P99 TPOT (ms):", r.p99_tpot_ms);
  sep("Inter-token Latency");
  line_f("Mean ITL (ms):", r.mean_itl_ms);
  line_f("Median ITL (ms):", r.median_itl_ms);
  line_f("P99 ITL (ms):", r.p99_itl_ms);
  sep("End-to-end Latency");
  line_f("Mean E2EL (ms):", r.mean_e2el_ms);
  line_f("Median E2EL (ms):", r.median_e2el_ms);
  line_f("P99 E2EL (ms):", r.p99_e2el_ms);
  sep("Prefill vs Decode split (gate #1)");
  line_f("Input (prefill) token throughput (tok/s):", r.input_throughput);
  line_f("Output (decode) token throughput (tok/s):", r.output_throughput);
  line_f("Mean per-stream decode rate (tok/s):", r.mean_per_stream_decode);
  std::fprintf(out, "====================================================\n");
}

}  // namespace vllm::bench

#endif  // VLLM_EXAMPLES_BENCH_BENCH_CORE_H_
