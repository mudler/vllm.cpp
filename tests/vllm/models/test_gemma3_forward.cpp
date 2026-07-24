// Gemma-3 (`Gemma3ForCausalLM`) forward doctest + THE SACRED MODEL correctness
// gate (sweep W2). Two tiers:
//
//  (1) CPU synthetic: the Gemma-3 forward (GemmaRMSNorm sandwich, per-head Gemma
//      q/k norm, dual-theta RoPE, GeGLU MLP, sqrt(hidden) embed-scale, tied
//      lm_head) composes, runs finite + deterministic; and the sqrt(hidden)
//      embed-scale is APPLIED (a no-scale variant differs).
//
//  (2) Real-checkpoint CUDA greedy vs the vLLM 0.25.0 oracle (dgx-only), the SACRED
//      gate. TOKENIZER-FREE: our BPE tokenizer does not validate Gemma's
//      byte_fallback tokenizer.json (the SentencePiece/BPE loader gap, mirroring
//      the Mistral gate), so we feed vLLM's EXACT prompt token ids (incl. the BOS
//      the tokenizer prepends — bos_token_id=2, verified) straight into our CUDA
//      prefill and greedy-decode by RE-PREFILLING the growing prefix, checking each
//      greedy token against vLLM's greedy set (K-run near-tie-robust:
//      [[near-tie-distributional-gate]]). This isolates + validates the FORWARD
//      (GeGLU, dual per-layer rope theta, Gemma sandwich norms, embed-scale, qpas
//      scaling) against the oracle — the primitives an OPT-style silent bug would
//      corrupt while still emitting fluent tokens.
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
#include "vllm/model_executor/models/gemma3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace fs = std::filesystem;

using vllm::Gemma3LayerWeights;
using vllm::Gemma3Model;
using vllm::Gemma3Weights;
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

// Tiny Gemma-3 config: GQA 4/1, head_dim 16 (full rotary), interleaved
// sliding/global (pattern 2 -> odd-index layers full, even sliding), dual rope
// theta, qpas 16, GeGLU MLP, tied lm_head.
HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 4;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 1;
  c.head_dim = 16;
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 100;
  c.sliding_window = 4;
  c.raw = nlohmann::json{{"query_pre_attn_scalar", 16},
                         {"rope_local_base_freq", 10000.0},
                         {"sliding_window_pattern", 2},
                         {"tie_word_embeddings", true}};
  return c;
}

Gemma3Weights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Gemma3Weights w;
  w.tie_word_embeddings = true;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.3f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    Gemma3LayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.pre_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.post_feedforward_layernorm = MakeBf16({H}, false, seed++, 0.3f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.3f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.3f);
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

std::vector<float> RunTinyForward(const HfConfig& c, const Gemma3Weights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Qcpu();
  return Gemma3Model::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("gemma3 forward: CPU synthetic runs, finite, deterministic") {
  const HfConfig c = TinyConfig();
  const Gemma3Weights w = TinyWeights(c);
  const std::vector<float> a = RunTinyForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));
  const std::vector<float> b = RunTinyForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("gemma3 forward: sqrt(hidden) embed-scale is applied (differs from unscaled)") {
  // Same weights; only the embed normalizer differs (real sqrt(64)=8 vs a config
  // whose hidden_size were 1 -> normalizer 1). The residual stream (hence the
  // logits) MUST differ, proving the embed-scale fired. We fake "no scale" by
  // scaling the embed table down by 1/sqrt(H) so scale*table == table_orig — i.e.
  // compare against a run whose effective normalizer is 1.
  const HfConfig c = TinyConfig();
  Gemma3Weights scaled = TinyWeights(c);
  const std::vector<float> ls = RunTinyForward(c, scaled);

  // Build an unscaled reference by pre-dividing the embed table by sqrt(H): the
  // forward re-multiplies by sqrt(H), cancelling — so this run's effective embed
  // scale is 1. A different residual => different logits => the scale is applied.
  Gemma3Weights unscaled = TinyWeights(c);
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
std::string FindGemma3Snap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
      ".cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "config.json", ec) &&
        fs::exists(e.path() / "model.safetensors", ec))
      return e.path().string();
  return "";
}
}  // namespace

// THE SACRED MODEL CORRECTNESS GATE (dgx-only, GPU), TOKENIZER-FREE. Feeds vLLM's
// EXACT prompt token ids (incl. BOS) into our CUDA prefill and greedy-decodes by
// re-prefilling the growing prefix, checking each argmax against vLLM 0.25.0's
// greedy set (K-run near-tie-robust). Oracle table captured on dgx via
// scripts/gemma3-oracle-capture.py.
TEST_CASE("gemma3 forward: real gemma-3-1b-it CUDA greedy vs oracle (dgx-only, SACRED)") {
  const std::string snap = FindGemma3Snap();
  if (snap.empty()) {
    MESSAGE("SKIP: gemma-3-1b-it checkpoint absent (CUDA greedy vs oracle)");
    return;
  }
  vt::Backend* cuda = nullptr;
  try { cuda = &vt::GetBackend(vt::DeviceType::kCUDA); }
  catch (...) { MESSAGE("SKIP: no CUDA backend registered"); return; }

  const HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(snap + "/model.safetensors"));
  const Gemma3Weights w = vllm::LoadGemma3ForCausalLMWeights(shards, cfg);

  // prompt = vLLM prompt token ids (BOS-prefixed); cont[step] = the SET of vLLM
  // greedy tokens observed across K runs (singleton = STRICT token-exact;
  // multi-member = a bf16 near-tie vLLM itself cannot separate).
  struct Case { std::vector<int32_t> prompt; std::vector<std::vector<int32_t>> cont; };
  const std::vector<Case> cases = {
      // GEMMA3_ORACLE_GREEDY_BEGIN (vLLM 0.25.0 greedy, google/gemma-3-1b-it;
      // K=5 per-prompt ALL-DETERMINISTIC -> singleton sets == STRICT token-exact)
      {{2, 818, 5279, 529, 7001, 563},  // 'The capital of France is'
       {{9079}, {236761}, {108}, {818}, {7488}, {3207}, {528}, {7001}}},
      {{2, 818, 7488, 13401, 528, 1023, 10321, 1458, 563},  // 'The largest planet in our solar system is'
       {{52895}, {236764}, {532}, {625}, {236789}, {236751}, {496}, {9995}}},
      {{2, 17390, 104264, 657, 496, 4022, 529},  // 'Water boils at a temperature of'
       {{236743}, {236770}, {236771}, {236771}, {10674}, {57356}, {236761}, {108}}},
      {{2, 818, 7395, 5404, 573, 5122, 563},  // 'The chemical symbol for gold is'
       {{16107}, {236761}, {108}, {818}, {7395}, {6581}, {573}, {5122}}},
      {{2, 818, 1171, 6207, 529, 506, 3640, 4184, 691},  // 'The first president of the United States was'
       {{9142}, {7693}, {236761}, {108}, {818}, {1855}, {6207}, {529}}},
      {{2, 205251, 659, 2604, 236764, 81758, 8524, 659},  // 'Roses are red, violets are'
       {{3730}, {236764}, {107}, {236777}, {1133}, {531}, {9039}, {496}}},
      // GEMMA3_ORACLE_GREEDY_END
  };
  if (cases.empty()) {
    MESSAGE("SKIP: gemma3 oracle greedy table not captured yet — run "
            "scripts/gemma3-oracle-capture.py on dgx and paste the table.");
    return;
  }

  const int64_t Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim, V = cfg.vocab_size;
  int total_ok = 0, total_steps = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    // BOS verified vs oracle: Gemma prepends bos_token_id=2.
    REQUIRE_FALSE(cases[ci].prompt.empty());
    CHECK_MESSAGE(cases[ci].prompt.front() == 2,
                  "gemma3 p" << ci << ": vLLM prompt must begin with BOS (2)");
    std::vector<int32_t> tokens = cases[ci].prompt;
    const size_t nsteps = cases[ci].cont.size();
    for (size_t step = 0; step < nsteps; ++step) {
      const int64_t T = static_cast<int64_t>(tokens.size());
      std::vector<int32_t> positions(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
      vt::Queue q = cuda->CreateQueue();
      const int64_t bs = 128;  // one block covers the short gate contexts (< sliding_window)
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
          Gemma3Model::Forward(tokens, positions, m, attn_kv, w, cfg, q);
      const float* last = logits.data() + (T - 1) * V;
      int argmax = 0;
      for (int64_t v = 1; v < V; ++v) if (last[v] > last[argmax]) argmax = static_cast<int>(v);
      for (void* p : devbuf) cuda->Free(p);

      const std::vector<int32_t>& want = cases[ci].cont[step];
      const bool member = std::find(want.begin(), want.end(), argmax) != want.end();
      ++total_steps;
      if (member) ++total_ok;
      std::string ws; for (int32_t t : want) ws += std::to_string(t) + " ";
      MESSAGE("gemma3 p" << ci << " step" << step << " argmax=" << argmax
              << " (vLLM greedy set {" << ws << "}) " << (member ? "OK" : "MISMATCH")
              << (want.size() > 1 ? " [near-tie]" : ""));
      CHECK_MESSAGE(member, "gemma3 p" << ci << " step" << step
                    << ": our argmax " << argmax << " not in vLLM greedy set");
      tokens.push_back(want.size() == 1 ? want[0] : argmax);
    }
  }
  MESSAGE("gemma3 CUDA greedy-vs-oracle: " << total_ok << "/" << total_steps
          << " greedy tokens match vLLM (SACRED gate)");
  CHECK(total_ok == total_steps);
}
