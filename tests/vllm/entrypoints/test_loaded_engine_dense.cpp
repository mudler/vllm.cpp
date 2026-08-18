// LoadedEngine DENSE-arch dispatch test (M0.8 27B CPU wiring) — proves the last
// CPU-side plumbing: the full LoadedEngine / Executor / EngineCore / LLMEngine
// stack loads and dispatches the DENSE 27B arch (Qwen3_5ForConditionalGeneration,
// num_experts==0) through the dense weights + the dense paged forward
// (Qwen3_5DenseModel::Forward), without regressing the MoE 35B path. CPU-only;
// the real 12GB W4A4 checkpoint + GPU are NOT touched (the greedy acceptance gate
// on the real checkpoint is test_qwen27_paged_engine.cpp, still SKIPPING).
//
// The 35B analogue for the MoE stack is tests/vllm/v1/test_llm_engine.cpp; this is
// its dense sibling driven through the packaging LoadedEngine seam. Cases:
//   1. ModelRegistry dispatch decision: explicit architecture IDs select their
//      factories; num_experts is no longer used as a model-class key.
//   2. Dense stack runs end to end: a LoadedEngine built via the DENSE constructor
//      over synthetic dense weights + a small hybrid dense config generates
//      exactly N greedy tokens, terminates, and is deterministic across two fresh
//      stacks — proving the dense weights thread executor -> engine -> runner and
//      the dense forward actually drives the loop.
#include "vllm/entrypoints/model_loader.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "vllm/config/device.h"
#include "vllm/config/scheduler.h"
#include "vllm/platforms/interface.h"
#include "vllm/v1/core/sched/async_scheduler.h"
#include "vllm/v1/engine/input_processor.h"  // InputValidationError

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

using nlohmann::json;
using vllm::DenseMlpWeights;
using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;
using vllm::RequestOutput;
using vllm::RequestOutputKind;
using vllm::SamplingParams;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vt::DType;

namespace {

// ─── Synthetic weights (mirrors test_qwen27_paged_forward.cpp) ───────────────
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}
OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
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

// Vocab == the tiny BPE fixture's assigned ids (0..23), block_size ==
// max_model_len == hash_block_size (hybrid coordinator constraint; prompts far
// shorter than a block keep prefix caching inert), matching test_llm_engine.cpp.
constexpr int kVocab = 24;
constexpr int kMaxModelLen = 32;

// 27B-shaped small DENSE config: layer_types [LA, LA, LA, FA], no experts,
// GQA ratio 3 (Hv/Hk = 6/2), attn_output_gate. num_experts==0 => dense arch.
HfConfig MakeDenseConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = kVocab;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;  // GQA ratio 3
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = kMaxModelLen;
  c.raw = json::object();  // no eos_token_id -> generation runs to max_tokens.
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeDenseWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// The tiny oracle-verified BPE fixture (ids 0..23, no holes) from test_llm_engine.
Tokenizer BuildFixture() {
  static int counter = 0;
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("vllm_dense_engine_tok_" + std::to_string(counter++) + ".json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", 19}, {"content", "<|end|>"}, {"special", true}},
       {{"id", 20}, {"content", "<tool>"}, {"special", false}},
       {{"id", 21}, {"content", "<|end|>of"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
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
  json vocab = {{"h", 0},   {"e", 1},   {"l", 2},     {"o", 3},   {"w", 4},
                {"r", 5},   {"d", 6},   {"Ġ", 7},     {"1", 8},   {"2", 9},
                {"ll", 10}, {"he", 11}, {"llo", 12},  {"hello", 13},
                {"Ġw", 14}, {"or", 15}, {"orld", 16}, {"Ġworld", 17},
                {"ld", 18}};
  vocab[MapBytesToUnicode("\xF0\x9F")] = 22;
  vocab[MapBytesToUnicode("\x8C\x8D")] = 23;
  doc["model"] = {
      {"type", "BPE"},
      {"ignore_merges", false},
      {"vocab", vocab},
      {"merges",
       json::array({json::array({"l", "l"}), json::array({"h", "e"}),
                    json::array({"ll", "o"}), json::array({"he", "llo"}),
                    json::array({"Ġ", "w"}), json::array({"o", "r"}),
                    json::array({"l", "d"}), json::array({"or", "ld"}),
                    json::array({"Ġw", "orld"})})}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  Tokenizer tok = Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

Tokenizer FreshFixture() { return BuildFixture(); }

SamplingParams Greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax) -> deterministic.
  sp.max_tokens = max_tokens;
  sp.output_kind = RequestOutputKind::kCumulative;
  return sp;
}

// FIX-GPU-MEM-UTIL-INERT (#1165): build ONE engine with std::cerr captured and
// return what it wrote. Scope-guarded, mirroring test_async_llm.cpp:130-144 —
// the restore must survive an exception out of the constructor, because two of
// the cases below build engines that log other things (the auto-fit INFO line)
// and a leaked rdbuf swap would silently redirect every later case.
class CerrRedirect {
 public:
  explicit CerrRedirect(std::streambuf* target)
      : previous_(std::cerr.rdbuf(target)) {}
  ~CerrRedirect() { std::cerr.rdbuf(previous_); }
  CerrRedirect(const CerrRedirect&) = delete;
  CerrRedirect& operator=(const CerrRedirect&) = delete;

 private:
  std::streambuf* previous_;
};

// The engine is built INSIDE the capture and destroyed inside it too, so a
// notice emitted from any part of construction is seen.
std::string CerrOfEngineLoad(const HfConfig& c, const EngineParams& params) {
  std::ostringstream captured;
  {
    CerrRedirect guard(captured.rdbuf());
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    std::cerr.flush();
  }
  return captured.str();
}

// The one substring that identifies the notice. Deliberately NOT the whole
// message: the cases assert the individual facts separately, so a reworded
// sentence fails on the fact it dropped rather than on all of them at once.
constexpr const char* kInertNotice = "--gpu-memory-utilization";

}  // namespace

// ─── 1. Arch-select: the FromModelDir dispatch decision ──────────────────────
TEST_CASE("loaded_engine: ModelRegistry routes explicit 27B dense vs 35B MoE IDs") {
  HfConfig dense = MakeDenseConfig();
  // Deliberately contradict the old structural heuristic: the architecture ID
  // remains authoritative.
  dense.num_experts = 4;
  CHECK(ModelRegistry::Resolve(dense).architecture ==
        "Qwen3_5ForConditionalGeneration");

  HfConfig moe = MakeDenseConfig();
  moe.model_type = "qwen3_5_moe_text";
  moe.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  moe.num_experts = 0;
  CHECK(ModelRegistry::Resolve(moe).architecture ==
        "Qwen3_5MoeForConditionalGeneration");
}

// ─── 2. The dense stack drives the full engine loop end to end ───────────────
TEST_CASE("loaded_engine: dense 27B arch generates deterministically through the full stack") {
  const HfConfig c = MakeDenseConfig();
  const std::string prompt = "hello";
  const int kN = 6;
  EngineParams params;  // defaults: block_size 32 == max_model_len 32.

  RequestOutput run1;
  RequestOutput run2;
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    run1 = eng.engine().generate(prompt, Greedy(kN), "req");
    CHECK_FALSE(eng.engine().has_unfinished_requests());  // loop terminated.
  }
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    run2 = eng.engine().generate(prompt, Greedy(kN), "req");
  }

  REQUIRE(run1.finished);
  REQUIRE(run1.outputs.size() == 1);
  // Exactly N tokens (no eos configured -> length finish): the dense forward
  // actually produced a stream through executor -> engine_core -> runner.
  CHECK(static_cast<int>(run1.outputs[0].token_ids.size()) == kN);
  REQUIRE(run1.outputs[0].finish_reason.has_value());
  CHECK(*run1.outputs[0].finish_reason == "length");

  // Deterministic: two fresh dense stacks over the same prompt agree.
  REQUIRE(run2.outputs.size() == 1);
  CHECK(run1.outputs[0].token_ids == run2.outputs[0].token_ids);
  CHECK(run1.outputs[0].text == run2.outputs[0].text);
}

TEST_CASE(
    "loaded_engine: ResolveMaxNumBatchedTokens per-arch default (dense 2048 "
    "flat, MoE concurrency-aware)") {
  EngineParams p;
  const int kBigLen = 262144;  // large max_model_len => the tiny-model ceiling
                               // (max_model_len*seqs) never binds here.

  // DENSE arch: vLLM's scheduler default 2048, FLAT across concurrency
  // (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
  // e24d1b24).
  for (int seqs : {8, 16, 32, 64}) {
    p.max_num_seqs = seqs;
    CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen,
                                                   /*is_dense_arch=*/true) ==
          2048);
  }

  // MoE arch: GB10-tuned concurrency-aware budget (unchanged behavior).
  p.max_num_seqs = 8;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 4096);
  p.max_num_seqs = 16;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 4096);
  p.max_num_seqs = 32;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);
  p.max_num_seqs = 64;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);

  // Explicit override wins for BOTH arches (the CLI --max-num-batched-tokens).
  p.max_num_seqs = 32;
  p.max_num_batched_tokens = 8192;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, true) == 8192);
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, false) == 8192);
  // ... but is still clamped up to the >= max_num_seqs invariant
  // (SchedulerConfig.verify_max_model_len, vllm/config/scheduler.py:87).
  p.max_num_seqs = 64;
  p.max_num_batched_tokens = 4;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, kBigLen, true) == 64);

  // Tiny-model ceiling preservation: whole workload smaller than the default
  // => budget capped at max_model_len*seqs (no behavior change for the small
  // synthetic CPU engines).
  p.max_num_batched_tokens = 0;
  p.max_num_seqs = 4;
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, /*max_model_len=*/64,
                                                 true) == 256);
  CHECK(LoadedEngine::ResolveMaxNumBatchedTokens(p, /*max_model_len=*/64,
                                                 false) == 256);
}

// ─── W3 ENG-ASYNC-SCHED: the construction enable-flip (DEFAULT ON) ────────────
// LoadedEngine resolves async scheduling ONCE at construction from
// runner_.runner_supports_async() (VT_ASYNC_RUNNER, DEFAULT ON since the
// 2026-07-17 flip mirroring vLLM config/vllm.py:992-1044) x VT_ASYNC_SCHED, then
// constructs an AsyncScheduler + max_concurrent_batches=2 when ON (else the
// byte-identical synchronous Scheduler + depth-1). These cases pin the resolution
// matrix: (a) the pure resolution logic (static, no disk load), and (b) the wired
// construction outcome (scheduler runtime TYPE + mcb) over a synthetic dense
// engine, across the production default and the two same-binary rollback arms
// (VT_ASYNC_RUNNER=0 runner-level, VT_ASYNC_SCHED=0 scheduler-level).
TEST_CASE(
    "loaded_engine: async-scheduling resolution matrix (runner x VT_ASYNC_SCHED)") {
  const vllm::SchedulerConfig cfg;  // async_scheduling == nullopt, as MakeSchedulerConfig.

  // VT_ASYNC_SCHED unset: resolution == runner_supports_async (the compat gate);
  // mcb follows (2 under async, else 1).
  ::unsetenv("VT_ASYNC_SCHED");
  CHECK(LoadedEngine::ResolveAsyncEnabled(cfg, /*runner_supports_async=*/false) ==
        false);
  CHECK(LoadedEngine::ResolveAsyncEnabled(cfg, /*runner_supports_async=*/true) ==
        true);
  CHECK(cfg.MaxConcurrentBatches(/*async=*/false) == 1);
  CHECK(cfg.MaxConcurrentBatches(/*async=*/true) == 2);

  // VT_ASYNC_SCHED=0: the same-binary rollback forces OFF regardless of the runner.
  ::setenv("VT_ASYNC_SCHED", "0", /*overwrite=*/1);
  CHECK(LoadedEngine::ResolveAsyncEnabled(cfg, /*runner_supports_async=*/false) ==
        false);
  CHECK(LoadedEngine::ResolveAsyncEnabled(cfg, /*runner_supports_async=*/true) ==
        false);
  ::unsetenv("VT_ASYNC_SCHED");
}

TEST_CASE(
    "loaded_engine: async enable-flip constructs AsyncScheduler + mcb=2 BY "
    "DEFAULT (rollback arms VT_ASYNC_RUNNER=0 / VT_ASYNC_SCHED=0)") {
  const HfConfig c = MakeDenseConfig();
  EngineParams params;  // defaults.

  // Clean env baseline. Every arm restores it so no state leaks between cases.
  ::unsetenv("VT_ASYNC_RUNNER");
  ::unsetenv("VT_ASYNC_SCHED");

  // (1) Production default (all env unset): the runner advertises async by
  // default (VT_ASYNC_RUNNER default ON), resolution defaults ON, so scheduler()
  // is an AsyncScheduler and mcb is 2 (depth-2 step_with_batch_queue engaged for
  // the async engine) — mirroring vLLM's async-scheduling default-on-when-compatible.
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    CHECK(eng.runner().runner_supports_async());
    CHECK(eng.async_scheduling_enabled());
    CHECK(eng.max_concurrent_batches() == 2);
    CHECK(dynamic_cast<const vllm::v1::AsyncScheduler*>(&eng.scheduler()) !=
          nullptr);
  }

  // (2) VT_ASYNC_RUNNER=0: the runner-level rollback — the runner does NOT
  // advertise async, so the flip resolves OFF: synchronous Scheduler, depth-1,
  // byte-identical to the pre-flip production path.
  ::setenv("VT_ASYNC_RUNNER", "0", /*overwrite=*/1);
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    CHECK_FALSE(eng.runner().runner_supports_async());
    CHECK_FALSE(eng.async_scheduling_enabled());
    CHECK(eng.max_concurrent_batches() == 1);
    CHECK(dynamic_cast<const vllm::v1::AsyncScheduler*>(&eng.scheduler()) ==
          nullptr);
  }
  ::unsetenv("VT_ASYNC_RUNNER");

  // (3) VT_ASYNC_SCHED=0 (runner default ON): the scheduler-level same-binary
  // rollback — the runner is still async-capable (runner_supports_async TRUE) but
  // the scheduler is forced back to synchronous (Scheduler + depth-1), so the A/B
  // can compare arms without a rebuild.
  ::setenv("VT_ASYNC_SCHED", "0", /*overwrite=*/1);
  {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    CHECK(eng.runner().runner_supports_async());
    CHECK_FALSE(eng.async_scheduling_enabled());
    CHECK(eng.max_concurrent_batches() == 1);
    CHECK(dynamic_cast<const vllm::v1::AsyncScheduler*>(&eng.scheduler()) ==
          nullptr);
  }

  ::unsetenv("VT_ASYNC_RUNNER");
  ::unsetenv("VT_ASYNC_SCHED");
}

TEST_CASE("loaded_engine: prefix caching mirrors model-capability defaults") {
  EngineParams params;
  vllm::ModelInfo decoder;
  CHECK(LoadedEngine::ResolveEnablePrefixCaching(params, decoder));

  vllm::ModelInfo hybrid;
  hybrid.is_hybrid = true;
  CHECK_FALSE(LoadedEngine::ResolveEnablePrefixCaching(params, hybrid));

  vllm::ModelInfo attention_free;
  attention_free.has_inner_state = true;
  CHECK_FALSE(LoadedEngine::ResolveEnablePrefixCaching(params, attention_free));

  params.enable_prefix_caching = true;
  CHECK(LoadedEngine::ResolveEnablePrefixCaching(params, hybrid));
  params.enable_prefix_caching = false;
  CHECK_FALSE(LoadedEngine::ResolveEnablePrefixCaching(params, decoder));
}

// ─── ARCH-ONE-SURFACE ROW 8: explicit device selection ───────────────────────
// The policy matrix behind SelectQueue's explicit arms, gated PURE over the
// "is the CUDA platform registered" probe answer so the CPU tier pins the
// whole contract — including the CUDA-build half ("explicit cpu beats a
// registered accelerator") that a CPU-only process could otherwise never
// exercise. SelectQueue routes its explicit arms through THIS function
// (model_loader.cpp), so these pins bind the production policy, not a copy.
TEST_CASE("loaded_engine: ResolveExplicitDeviceType uses the named platform without fallback") {
  using vllm::Device;

  // Explicit CPU resolves CPU regardless of what the name lookup found. The
  // non-CPU value is the accelerator-build pin: a registered accelerator must
  // NOT win over an
  // explicit cpu ask (the fold-ROW-8 defect was that an embedder could not ASK
  // for CPU at all).
  CHECK(LoadedEngine::ResolveExplicitDeviceType(
            Device::kCPU, std::optional{vt::DeviceType::kXPU}) ==
        vt::DeviceType::kCPU);
  CHECK(LoadedEngine::ResolveExplicitDeviceType(Device::kCPU, std::nullopt) ==
        vt::DeviceType::kCPU);

  // The explicit named-platform arm returns what the registry found. kXPU is
  // deliberate mutation sensitivity: a hidden CUDA constant cannot satisfy it.
  CHECK(LoadedEngine::ResolveExplicitDeviceType(
            Device::kNamedPlatform, std::optional{vt::DeviceType::kXPU}) ==
        vt::DeviceType::kXPU);

  // Explicit CUDA WITHOUT the platform throws the pinned message — never a
  // silent CPU fallback (mirror of vLLM assigning an explicit device verbatim,
  // vllm/config/device.py:61-66).
  CHECK_THROWS_WITH_AS(
      LoadedEngine::ResolveExplicitDeviceType(Device::kNamedPlatform,
                                              std::nullopt),
      doctest::Contains("device 'cuda' was requested but no CUDA platform"),
      std::runtime_error);

  // kAuto is not an explicit selection: it resolves through the probe inside
  // SelectQueue, and this seam refuses it rather than guessing.
  CHECK_THROWS_AS(
      LoadedEngine::ResolveExplicitDeviceType(Device::kAuto, std::nullopt),
                  std::invalid_argument);
}

TEST_CASE("loaded_engine: DeviceFromString mirrors the vLLM Device names") {
  using vllm::Device;
  // The supported subset of upstream's Device Literal (vllm/config/device.py:13)
  // — the strings the server's --device flag consumes.
  CHECK(vllm::DeviceFromString("auto") == Device::kAuto);
  CHECK(vllm::DeviceFromString("cpu") == Device::kCPU);
  CHECK(vllm::DeviceFromString("cuda") == Device::kNamedPlatform);
  CHECK_THROWS_WITH_AS(vllm::DeviceFromString("tpu"),
                       doctest::Contains("Unknown device: tpu"),
                       std::invalid_argument);
  CHECK_THROWS_AS(vllm::DeviceFromString(""), std::invalid_argument);

  // The wire contract (vllm_model_params.device, ABI v14): 0 MUST stay auto so
  // a zero-initialized struct preserves pre-v14 behaviour; cpu/cuda follow the
  // v12 vllm_video_model_params.device precedent (0 cpu, 1 cuda) shifted by
  // the auto slot.
  CHECK(static_cast<int32_t>(Device::kAuto) == 0);
  CHECK(static_cast<int32_t>(Device::kCPU) == 1);
  CHECK(static_cast<int32_t>(Device::kNamedPlatform) == 2);
  CHECK(std::string(vllm::DeviceName(Device::kAuto)) == "auto");
  CHECK(std::string(vllm::DeviceName(Device::kCPU)) == "cpu");
  CHECK(std::string(vllm::DeviceName(Device::kNamedPlatform)) == "cuda");
}

TEST_CASE("loaded_engine: FromModelDir resolves an explicit absent device BEFORE any path I/O") {
  // The device error must win over the path error (the mirror of vLLM building
  // DeviceConfig at config-creation time, before the model load —
  // arg_utils.py:1878, device.py __post_init__). This is also what makes the
  // EngineParams->FromModelDir plumb pinnable on the CPU tier with no loadable
  // checkpoint: a bogus path + device=cuda must report the DEVICE, not the path.
  if (vllm::platforms::FindPlatformByName("cuda") != nullptr) {
    return;  // CUDA build/box: the explicit-cuda arm resolves; nothing to pin.
  }
  EngineParams params;
  params.device = vllm::Device::kNamedPlatform;
  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/model/dir", params),
      doctest::Contains("device 'cuda' was requested but no CUDA platform"),
      std::runtime_error);

  // An explicit CPU ask is legal and proceeds to the path (the path error, not
  // a device error, surfaces) — the plumb forwards the field, not a constant.
  params.device = vllm::Device::kCPU;
  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/model/dir", params),
      doctest::Contains("not a directory"), std::runtime_error);
}

// ─── KV sizing at startup (issue #83 M4; external PR #227) ───────────────────
// vllm/v1/core/kv_cache_utils.py:2160-2174 @ 555967922 runs both halves at
// engine init. Without them a prompt the pool can never hold is admitted, never
// allocates, and the engine spins at model_executed=0 with an idle GPU.

TEST_CASE(
    "loaded_engine: refuses a pinned --max-model-len the KV pool cannot hold") {
  // _check_enough_kv_cache_memory (kv_cache_utils.py:751-788): the caller asked
  // for 4096 tokens of context out of a 1 x 32-token pool.
  const HfConfig c = MakeDenseConfig();
  EngineParams params;
  params.num_blocks = 1;  // 1 x 32 = 32 tokens of KV
  params.max_model_len = 4096;

  CHECK_THROWS_WITH_AS(
      LoadedEngine(c, MakeDenseWeights(c), FreshFixture(), params),
      doctest::Contains("larger than the available KV cache memory"),
      std::invalid_argument);

  // The message carries upstream's remediation and ours, so the user is left
  // with an action rather than a number.
  try {
    LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
    FAIL("expected the KV sizing check to refuse this configuration");
  } catch (const std::invalid_argument& e) {
    const std::string msg = e.what();
    CHECK(msg.find("max seq len (4096)") != std::string::npos);
    CHECK(msg.find("estimated maximum model length is 32") != std::string::npos);
    CHECK(msg.find("--num-blocks") != std::string::npos);
  }
}

TEST_CASE(
    "ResolveMaxModelLen: a model with no paged KV is never refused by the "
    "sizing check") {
  // kv_cache_utils.py:872-878 guards the check with `if kv_cache_spec:`. A
  // model whose KV state does not scale with the block count (attention-free,
  // or pure Mamba/GDN — KVBytesPerBlock is 0 for both) has nothing to run out
  // of, so a pinned length must pass however small the pool is, and an unpinned
  // one must not be fitted down to nothing.
  const HfConfig c = MakeDenseConfig();
  vllm::v1::KVCacheConfig no_paged_kv{};
  no_paged_kv.num_blocks = 1;  // no groups -> KVBytesPerBlock == 0

  EngineParams pinned;
  pinned.max_model_len = 4096;
  CHECK(LoadedEngine::ResolveMaxModelLen(pinned, c, no_paged_kv,
                                         /*block_size=*/32) == 4096);

  EngineParams unpinned;
  CHECK(LoadedEngine::ResolveMaxModelLen(unpinned, c, no_paged_kv,
                                         /*block_size=*/32) == kMaxModelLen);
}

TEST_CASE(
    "loaded_engine: a pinned --max-model-len the pool CAN hold is served "
    "unchanged") {
  const HfConfig c = MakeDenseConfig();
  EngineParams params;
  params.num_blocks = 1;                // 32 tokens of KV
  params.max_model_len = kMaxModelLen;  // exactly one pool
  LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
  CHECK(eng.max_model_len() == kMaxModelLen);
}

TEST_CASE(
    "loaded_engine: an unpinned max_model_len auto-fits down to the KV pool") {
  // _auto_fit_max_model_len (kv_cache_utils.py:1967-2027): the checkpoint claims
  // 4096 tokens of context and the pool holds 32, so 32 is what gets served —
  // and the admission check then rejects anything longer instead of the
  // scheduler wedging on it.
  HfConfig c = MakeDenseConfig();
  c.max_position_embeddings = 4096;
  EngineParams params;
  params.num_blocks = 1;  // 32 tokens
  LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
  CHECK(eng.max_model_len() == kMaxModelLen);
}

TEST_CASE(
    "loaded_engine: an unpinned max_model_len the pool holds is NOT reduced") {
  // The default path must be untouched: 256 x 32 = 8192 tokens of KV against a
  // 32-token checkpoint context leaves the context alone.
  const HfConfig c = MakeDenseConfig();
  EngineParams params;  // defaults: num_blocks 256, block_size 32
  LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
  CHECK(eng.max_model_len() == kMaxModelLen);
}

TEST_CASE("loaded_engine: an over-long prompt is REFUSED, not left waiting") {
  // The end-to-end point of both guards. Before them this prompt was admitted,
  // could never allocate, and the engine produced no tokens forever.
  HfConfig c = MakeDenseConfig();
  c.max_position_embeddings = 4096;
  EngineParams params;
  params.num_blocks = 1;  // 32 tokens of KV -> max_model_len auto-fits to 32
  LoadedEngine eng(c, MakeDenseWeights(c), FreshFixture(), params);
  REQUIRE(eng.max_model_len() == kMaxModelLen);

  const std::vector<int32_t> long_prompt(kMaxModelLen + 8, 3);
  CHECK_THROWS_AS(eng.engine().generate(long_prompt, Greedy(1), "toolong"),
                  vllm::v1::InputValidationError);
  CHECK_FALSE(eng.engine().has_unfinished_requests());
}

// ─── --gpu-memory-utilization must not be silently inert (#1165) ─────────────
// The flag is parsed, threaded to both engines, carried on the C ABI and then
// read by NOTHING: ResolveNumBlocks falls through to `return 256` under a
// TODO(ROAD-V1-MEM M3). Implementing the utilization path is #83 and is
// dgx-gated. What is gated HERE is only that the engine stops reporting
// success for a budget it discarded.
//
// These cases enter through the LOADER, not through the resolver:
// ResolveNumBlocks is private, and a test that called it would prove the
// function works rather than that anything reaches it. The chain under test is
// LoadedEngine ctor -> MakeKVCacheResolved -> ResolveNumBlocks
// (`src/vllm/entrypoints/model_loader.cpp::MakeKVCacheResolved`,
// `src/vllm/entrypoints/model_loader.cpp::ResolveNumBlocks`).

TEST_CASE(
    "loaded_engine: an EXPLICIT --gpu-memory-utilization says it did not size "
    "the KV pool") {
  const HfConfig c = MakeDenseConfig();
  EngineParams params;
  params.gpu_memory_utilization = 0.85;  // the flag a user actually types

  const std::string logged = CerrOfEngineLoad(c, params);

  // It fired at all.
  REQUIRE(logged.find(kInertNotice) != std::string::npos);
  // It names the value the caller chose, so a reader can tell WHICH knob is
  // being reported when several are set.
  CHECK(logged.find("0.85") != std::string::npos);
  // It names what actually sized the pool instead.
  CHECK(logged.find("256") != std::string::npos);
  // It names the two flags that DO bind today, so the reader is left with an
  // action rather than a complaint.
  CHECK(logged.find("--kv-cache-memory") != std::string::npos);
  CHECK(logged.find("--num-blocks") != std::string::npos);
  // It names the row and the issue that own the real fix, so the notice cannot
  // be mistaken for a permanent limitation.
  CHECK(logged.find("ROAD-V1-MEM") != std::string::npos);
  CHECK(logged.find("83") != std::string::npos);
}

TEST_CASE(
    "loaded_engine: an UNSET --gpu-memory-utilization is silent") {
  // The noise case, and the one that fails if the notice is made
  // unconditional. A default nobody chose has nothing to warn about: the pool
  // resolves to 256 blocks, which is exactly what the default documents.
  const HfConfig c = MakeDenseConfig();
  const EngineParams params;  // gpu_memory_utilization untouched

  CHECK(CerrOfEngineLoad(c, params).find(kInertNotice) == std::string::npos);
}

TEST_CASE(
    "loaded_engine: --gpu-memory-utilization beside --kv-cache-memory is "
    "silent, because vLLM ignores the fraction there too") {
  // cache.py:189 @ 555967922: kv_cache_memory_bytes IGNORES
  // gpu_memory_utilization. A caller who set both gets vLLM's exact semantics,
  // so there is no lie to report. This case fails if the notice is hoisted
  // above ResolveNumBlocks' early returns.
  const HfConfig c = MakeDenseConfig();
  EngineParams params;
  params.gpu_memory_utilization = 0.85;
  // One block's worth of bytes for this model's own geometry, so knob 2
  // resolves to a servable pool rather than throwing.
  params.kv_cache_memory_bytes = 1LL << 30;

  CHECK(CerrOfEngineLoad(c, params).find(kInertNotice) == std::string::npos);
}

TEST_CASE(
    "loaded_engine: --gpu-memory-utilization beside a --num-blocks override is "
    "silent") {
  // Same argument at knob 1 (vLLM num_gpu_blocks_override): the override sized
  // the pool, so the fraction was not the thing that got discarded.
  const HfConfig c = MakeDenseConfig();
  EngineParams params;
  params.gpu_memory_utilization = 0.85;
  params.num_blocks = 4;

  CHECK(CerrOfEngineLoad(c, params).find(kInertNotice) == std::string::npos);
}
