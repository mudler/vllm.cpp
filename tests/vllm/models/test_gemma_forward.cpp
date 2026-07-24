// Gemma-1 (`GemmaForCausalLM`) forward doctest + THE SACRED MODEL correctness
// gate (sweep W5). Two tiers:
//
//  (1) CPU synthetic: the Gemma-1 forward (two fused GemmaRMSNorms per layer,
//      head_dim^-0.5 scaling, GeGLU MLP, sqrt(hidden) embed-scale, tied lm_head)
//      composes, runs finite + deterministic; and the embed-scale is APPLIED.
//  (2) Real-checkpoint CUDA greedy vs the vLLM 0.25.0 oracle (dgx-only), the
//      SACRED gate. TOKENIZER-FREE (feeds vLLM's exact BOS-prefixed ids).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace fs = std::filesystem;

using vllm::GemmaLayerWeights;
using vllm::GemmaModel;
using vllm::GemmaWeights;
using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

namespace {

vt::Queue Qcpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

vllm::OwnedTensor MakeBf16(const std::vector<int64_t>& shape, bool nk, uint32_t seed,
                           float scale = 0.08f) {
  vllm::OwnedTensor o;
  o.dtype = DType::kBF16;
  o.nk = nk;
  o.rank = static_cast<int>(shape.size());
  int64_t numel = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[static_cast<size_t>(i)];
    numel *= shape[static_cast<size_t>(i)];
  }
  o.bytes.resize(static_cast<size_t>(numel) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (int64_t i = 0; i < numel; ++i) p[i] = vt::F32ToBF16(dist(rng));
  return o;
}

// Tiny Gemma-1 config: MQA 4/1, head_dim 16, GeGLU, tied lm_head, no soft-cap.
HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 4;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 1;
  c.head_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000.0;
  c.vocab_size = 100;
  c.raw = nlohmann::json{{"tie_word_embeddings", true}};
  return c;
}

GemmaWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  GemmaWeights w;
  w.tie_word_embeddings = true;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.3f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    GemmaLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

struct CachePool {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> attn_kv;
  CachePool(const HfConfig& c, int64_t num_blocks, int64_t block_size) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l)
      buf.emplace_back(static_cast<size_t>(num_blocks * 2 * block_size * Hkv * Dh), 0.0f);
    for (auto& b : buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = num_blocks;
      kv.block_size = block_size;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
  }
};

CommonAttentionMetadata PrefillMeta(int64_t T, int64_t block_size) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(T)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(T);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t % block_size);
  m.causal = true;
  return m;
}

std::vector<float> RunTinyForward(const HfConfig& c, const GemmaWeights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();
  return GemmaModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("gemma forward: CPU synthetic runs, finite, deterministic") {
  const HfConfig c = TinyConfig();
  const GemmaWeights w = TinyWeights(c);
  const std::vector<float> a = RunTinyForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));
  const std::vector<float> b = RunTinyForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("gemma forward: sqrt(hidden) embed-scale is applied (differs from unscaled)") {
  const HfConfig c = TinyConfig();
  const GemmaWeights scaled = TinyWeights(c);
  const std::vector<float> ls = RunTinyForward(c, scaled);
  GemmaWeights unscaled = TinyWeights(c);
  const float inv = 1.0f / std::sqrt(static_cast<float>(c.hidden_size));
  auto* p = reinterpret_cast<uint16_t*>(unscaled.embed_tokens.bytes.data());
  const int64_t n = unscaled.embed_tokens.Numel();
  for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(vt::BF16ToF32(p[i]) * inv);
  const std::vector<float> lu = RunTinyForward(c, unscaled);
  bool differs = false;
  for (size_t i = 0; i < ls.size() && !differs; ++i)
    if (ls[i] != lu[i]) differs = true;
  CHECK(differs);
}

namespace {
std::string FindSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const char* repos[] = {"models--unsloth--gemma-2b", "models--google--gemma-2b"};
  for (const char* repo : repos) {
    const fs::path snaps = fs::path(home) / ".cache/huggingface/hub" / repo / "snapshots";
    std::error_code ec;
    if (!fs::is_directory(snaps, ec)) continue;
    for (const auto& e : fs::directory_iterator(snaps, ec))
      if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}
std::vector<vllm::SafetensorsFile> OpenShards(const std::string& snap) {
  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  const fs::path single = fs::path(snap) / "model.safetensors";
  if (fs::exists(single, ec)) { shards.push_back(vllm::SafetensorsFile::Open(single.string())); return shards; }
  for (const auto& e : fs::directory_iterator(snap, ec)) {
    const std::string n = e.path().filename().string();
    if (n.rfind("model-", 0) == 0 && n.size() > 12 && n.substr(n.size() - 12) == ".safetensors")
      shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  }
  return shards;
}
}  // namespace

// THE SACRED MODEL CORRECTNESS GATE (dgx-only, GPU), TOKENIZER-FREE. Oracle table
// captured via scripts/gemma-oracle-capture.py --tag GEMMA1.
TEST_CASE("gemma forward: real gemma-2b CUDA greedy vs oracle (dgx-only, SACRED)") {
  const std::string snap = FindSnap();
  if (snap.empty()) { MESSAGE("SKIP: gemma-2b checkpoint absent"); return; }
  vt::Backend* cuda = nullptr;
  try { cuda = &vt::GetBackend(vt::DeviceType::kCUDA); }
  catch (...) { MESSAGE("SKIP: no CUDA backend registered"); return; }

  const HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards = OpenShards(snap);
  REQUIRE_FALSE(shards.empty());
  const GemmaWeights w = vllm::LoadGemmaForCausalLMWeights(shards, cfg);

  struct Case { std::vector<int32_t> prompt; std::vector<std::vector<int32_t>> cont; };
  const std::vector<Case> cases = {
      // GEMMA1_ORACLE_GREEDY_BEGIN (vLLM 0.25.0 greedy, unsloth/gemma-2b;
      // K=5 per-prompt ALL-DETERMINISTIC -> singleton sets == STRICT token-exact)
      {{2, 651, 6037, 576, 6081, 603},  // 'The capital of France is'
       {{476}, {3413}, {576}, {82777}, {235265}, {1165}, {603}, {476}}},
      {{2, 651, 10155, 14908, 575, 1167, 12677, 1812, 603},  // 'The largest planet in our solar system is'
       {{48453}, {235269}, {675}, {476}, {5182}, {576}, {697}, {235274}}},
      {{2, 11586, 115014, 696, 476, 5809, 576},  // 'Water boils at a temperature of'
       {{697}, {235274}, {235276}, {235276}, {3723}, {5837}, {235270}, {730}}},
      {{2, 651, 9408, 10159, 604, 4394, 603},  // 'The chemical symbol for gold is'
       {{2328}, {235265}, {6863}, {603}, {476}, {12068}, {5998}, {235265}}},
      {{2, 651, 1370, 9456, 576, 573, 3520, 3858, 729},  // 'The first president of the United States was'
       {{7565}, {575}, {235248}, {235274}, {235324}, {235304}, {235321}, {575}}},
      {{2, 154240, 708, 3118, 235269, 185737, 708},  // 'Roses are red, violets are'
       {{3868}, {235269}, {590}, {235349}, {235262}, {2319}, {577}, {3337}}},
      // GEMMA1_ORACLE_GREEDY_END
  };
  if (cases.empty()) {
    MESSAGE("SKIP: gemma1 oracle greedy table not captured — run "
            "scripts/gemma-oracle-capture.py --model google/gemma-2b --tag GEMMA1.");
    return;
  }

  const int64_t Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim, V = cfg.vocab_size;
  int total_ok = 0, total_steps = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    REQUIRE_FALSE(cases[ci].prompt.empty());
    CHECK_MESSAGE(cases[ci].prompt.front() == 2,
                  "gemma1 p" << ci << ": vLLM prompt must begin with BOS (2)");
    std::vector<int32_t> tokens = cases[ci].prompt;
    const size_t nsteps = cases[ci].cont.size();
    for (size_t step = 0; step < nsteps; ++step) {
      const int64_t T = static_cast<int64_t>(tokens.size());
      std::vector<int32_t> positions(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
      vt::Queue q = cuda->CreateQueue();
      const int64_t bs = 128;
      const size_t cbytes =
          static_cast<size_t>(1 * 2 * bs * Hkv * Dh) * vt::SizeOf(DType::kBF16);
      std::vector<void*> devbuf;
      std::vector<PagedKvCache> attn_kv;
      for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
        void* p = cuda->Alloc(cbytes);
        cuda->Memset(q, p, 0, cbytes);
        devbuf.push_back(p);
        PagedKvCache kv;
        kv.data = p; kv.dtype = DType::kBF16; kv.num_blocks = 1;
        kv.block_size = bs; kv.num_kv_heads = Hkv; kv.head_size = Dh;
        attn_kv.push_back(kv);
      }
      CommonAttentionMetadata m;
      m.num_reqs = 1;
      m.num_actual_tokens = static_cast<int>(T);
      m.query_start_loc = {0, static_cast<int32_t>(T)};
      m.query_start_loc_cpu = m.query_start_loc;
      m.seq_lens = {static_cast<int32_t>(T)};
      m.seq_lens_cpu = m.seq_lens;
      m.max_query_len = static_cast<int>(T);
      m.max_seq_len = static_cast<int>(T);
      m.block_table_num_cols = 1;
      m.block_table_tensor = {0};
      for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t);
      m.causal = true;

      const std::vector<float> logits =
          GemmaModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
      const float* last = logits.data() + (T - 1) * V;
      int argmax = 0;
      for (int64_t v = 1; v < V; ++v) if (last[v] > last[argmax]) argmax = static_cast<int>(v);
      for (void* p : devbuf) cuda->Free(p);

      const std::vector<int32_t>& want = cases[ci].cont[step];
      const bool member = std::find(want.begin(), want.end(), argmax) != want.end();
      ++total_steps;
      if (member) ++total_ok;
      std::string ws; for (int32_t t : want) ws += std::to_string(t) + " ";
      MESSAGE("gemma1 p" << ci << " step" << step << " argmax=" << argmax
              << " (vLLM greedy set {" << ws << "}) " << (member ? "OK" : "MISMATCH")
              << (want.size() > 1 ? " [near-tie]" : ""));
      CHECK_MESSAGE(member, "gemma1 p" << ci << " step" << step
                    << ": our argmax " << argmax << " not in vLLM greedy set");
      tokens.push_back(want.size() == 1 ? want[0] : argmax);
    }
  }
  MESSAGE("gemma1 CUDA greedy-vs-oracle: " << total_ok << "/" << total_steps
          << " greedy tokens match vLLM (SACRED gate)");
  CHECK(total_ok == total_steps);
}
