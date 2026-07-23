// vllm.cpp original (fifth-family additive-model W3 forward doctest); no upstream
// mirror.
//
// CPU synthetic forward for the DENSE Mistral (`MistralForCausalLM`) path. Mistral
// reuses the shared dense forward (MistralModel == Qwen3DenseModel) and is SIMPLER
// than Llama — no new primitive at all: plain RoPE (theta 1e6, no scaling) and no
// qk-norm. The two seams it exercises on CPU (no checkpoint, runs in CI) are:
//   (1) qk-norm-ABSENT attention: q_norm/k_norm left EMPTY -> the shared AttnBlock
//       skips the norm step and the forward still runs, finite + deterministic (the
//       qk-norm-optional seam, shared with Llama);
//   (2) UNTIED lm_head is ACTUALLY used: with a distinct standalone lm_head the
//       logits DIFFER from the tied case (lm_head aliasing embed_tokens), proving
//       the untied output-projection path fires — the one config-shape novelty vs
//       Llama-3.2-1B (which is tied);
//   (3) the fusion-catalog ADOPT path is byte-identical to the hand-call fallback
//       for the qk-norm-absent forward.
// The real token-exact vs-oracle bar is tests/parity/test_mistral_paged_engine.cpp
// (dgx-only).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/mistral.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::MistralWeights;
using vllm::PagedKvCache;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

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

// Tiny Mistral config: GQA 4/2, head_dim 16 (rotary over the full head), plain RoPE
// theta 1e6 with NO rope_scaling (rope_type "default"), rms_norm_eps 1e-5, untied.
HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;  // partial_rotary_factor 1.0 (Mistral rotates the whole head)
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-5;  // Mistral rms_norm_eps
  c.rope_theta = 1000000.0;
  c.vocab_size = 100;
  // rope_parameters.rope_type defaults to "default" -> plain RoPE, no llama3 rescale.
  return c;
}

// Mistral dense weights: NO q_norm/k_norm (empty), UNTIED lm_head unless `tied`.
MistralWeights TinyWeights(const HfConfig& c, bool tied) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  MistralWeights w;
  w.tie_word_embeddings = tied;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  // UNTIED: a DISTINCT lm_head [H, V] (Matmul-B). Tied: leave lm_head empty so the
  // forward aliases embed_tokens.
  if (!tied) w.lm_head = MakeBf16({H, V}, /*nk=*/false, 3);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    // NO q_norm/k_norm — Mistral has no qk-norm (the seam this doctest exercises).
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

std::vector<float> RunForward(const HfConfig& c, const MistralWeights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  return vllm::MistralModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("mistral dense forward: qk-norm-absent CPU synthetic runs, finite, deterministic") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const MistralWeights w = TinyWeights(c, /*tied=*/false);
  // Precondition: this is genuinely a qk-norm-absent, untied model.
  REQUIRE(w.layers[0].attn.q_norm.Empty());
  REQUIRE(w.layers[0].attn.k_norm.Empty());
  REQUIRE_FALSE(w.lm_head.Empty());

  const std::vector<float> a = RunForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  const std::vector<float> b = RunForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("mistral dense forward: untied lm_head is applied (differs from tied)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  // Same trunk weights both ways; only the output projection differs (untied
  // standalone lm_head vs tied embed_tokens). The logits MUST differ, proving the
  // untied path — Mistral's one config-shape novelty vs the tied Llama-3.2-1B — fired.
  MistralWeights untied = TinyWeights(c, /*tied=*/false);
  MistralWeights tied = TinyWeights(c, /*tied=*/true);

  const std::vector<float> lu = RunForward(c, untied);
  const std::vector<float> lt = RunForward(c, tied);
  REQUIRE(lu.size() == lt.size());

  bool differs = false;
  for (size_t i = 0; i < lu.size() && !differs; ++i)
    if (lu[i] != lt[i]) differs = true;
  CHECK(differs);

  // Each configuration is itself deterministic.
  const std::vector<float> lu2 = RunForward(c, untied);
  CHECK(std::memcmp(lu.data(), lu2.data(), lu.size() * sizeof(float)) == 0);
}

TEST_CASE("mistral dense forward: fusion-catalog ADOPT == hand-call fallback (byte-exact)") {
  const HfConfig c = TinyConfig();
  const MistralWeights w = TinyWeights(c, /*tied=*/false);

  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const std::vector<float> adopt = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "0", 1);
  const std::vector<float> hand = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);

  REQUIRE(adopt.size() == hand.size());
  CHECK(std::memcmp(adopt.data(), hand.data(), adopt.size() * sizeof(float)) == 0);
}

namespace {
namespace fs = std::filesystem;
std::string FindMistralSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
      ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  return "";
}
std::vector<vllm::SafetensorsFile> OpenShards(const std::string& dir) {
  std::vector<vllm::SafetensorsFile> shards;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    const std::string name = e.path().filename().string();
    if (name.rfind("model-", 0) == 0 && e.path().extension() == ".safetensors")
      shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  }
  return shards;
}
}  // namespace

// THE MODEL CORRECTNESS GATE (dgx-only, GPU), TOKENIZER-FREE. Because our
// ByteLevel-only tokenizer cannot load Mistral's SentencePiece/Metaspace
// tokenizer.json (blocker LOAD-SENTENCEPIECE — see test_mistral_load.cpp),
// the full paged-engine gate is blocked. This case instead feeds vLLM's EXACT prompt
// token ids straight into our CUDA prefill and does greedy decode by RE-PREFILLING
// the growing prefix, checking each greedy token == vLLM's greedy continuation. This
// isolates and validates the FORWARD (embed, GQA attention, PLAIN rope theta 1e6, NO
// qk-norm, SwiGLU, RMSNorm eps 1e-5, UNTIED lm_head, KV growth) against the vLLM
// 0.25.0 oracle, independent of the prompt tokenizer.
//
// The prompt ids are vLLM's tokenization; the `want` continuations are vLLM's greedy
// (temperature 0) decode captured on dgx via scripts/qwen3-oracle-capture.py.
TEST_CASE("mistral dense forward: real Mistral-7B-v0.3 CUDA greedy vs oracle (dgx-only, MODEL GATE)") {
  const std::string snap = FindMistralSnap();
  if (snap.empty()) {
    MESSAGE("SKIP: Mistral-7B-v0.3 checkpoint absent (CUDA greedy vs oracle)");
    return;
  }
  vt::Backend* cuda = nullptr;
  try { cuda = &vt::GetBackend(vt::DeviceType::kCUDA); }
  catch (...) { MESSAGE("SKIP: no CUDA backend registered"); return; }

  const vllm::HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards = OpenShards(snap);
  REQUIRE(shards.size() >= 1);
  const vllm::MistralWeights w = vllm::LoadMistralForCausalLMWeights(shards, cfg);

  // prompt = vLLM prompt token ids; cont[step] = the SET of vLLM greedy tokens
  // observed across K=3 runs (the near-tie-robust bar: our argmax must be a MEMBER —
  // a singleton set is STRICT token-exact; a multi-member set is a bf16 near-tie vLLM
  // itself cannot separate). Captured on dgx (mistral_capture2.py, gpu_mem 0.40).
  // vLLM greedy was DETERMINISTIC on 4/5 prompts (K=3); the sole non-singleton is
  // p3 "E equals m c" at the last step ({1782, 4608}).
  struct Case { std::vector<int32_t> prompt; std::vector<std::vector<int32_t>> cont; };
  const std::vector<Case> cases = {
      // MISTRAL_ORACLE_GREEDY_BEGIN (vLLM 0.25.0 greedy, mistralai/Mistral-7B-v0.3)
      {{1, 1183, 6333, 1070, 5611, 1117},                                       // "The capital of France is"
       {{1032}, {3758}, {1137}, {1117}, {2941}, {1070}}},
      {{1, 1183, 8407, 10641, 1065, 1581, 13792, 2355, 1117},                   // "The largest planet in our solar system is"
       {{1243}, {24538}, {29491}, {1429}, {1117}, {1040}}},
      {{1, 14021, 5936, 1117, 1032, 1851, 2990, 1070},                          // "Machine learning is a subfield of"
       {{19046}, {11663}, {1137}, {21933}, {1124}, {1040}}},
      {{1, 1181, 22356, 1058, 1045},                                            // "E equals m c"
       {{5292}, {2095}, {29491}, {781}, {781}, {1782, 4608}}},
      {{1, 1569, 16950, 1034, 28895, 29500, 29479, 2097},                       // "def fibonacci(n):"
       {{781}, {3055}, {1281}, {1075}, {1627}, {29473}}},
      // MISTRAL_ORACLE_GREEDY_END
  };

  const int64_t Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim;
  const int64_t V = cfg.vocab_size;
  int total_ok = 0, total_steps = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    if (cases[ci].cont.empty()) {
      MESSAGE("SKIP prompt p" << ci << ": oracle continuation not captured yet");
      continue;
    }
    std::vector<int32_t> tokens = cases[ci].prompt;
    const size_t nsteps = cases[ci].cont.size();
    for (size_t step = 0; step < nsteps; ++step) {
      const int64_t T = static_cast<int64_t>(tokens.size());
      std::vector<int32_t> positions(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
      vt::Queue q = cuda->CreateQueue();
      const int64_t bs = 64;
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
          vllm::MistralModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
      const float* last = logits.data() + (T - 1) * V;
      int argmax = 0;
      for (int64_t v = 1; v < V; ++v) if (last[v] > last[argmax]) argmax = static_cast<int>(v);
      for (void* p : devbuf) cuda->Free(p);

      const std::vector<int32_t>& want = cases[ci].cont[step];
      const bool member = std::find(want.begin(), want.end(), argmax) != want.end();
      ++total_steps;
      if (member) ++total_ok;
      std::string ws; for (int32_t t : want) ws += std::to_string(t) + " ";
      MESSAGE("mistral p" << ci << " step" << step << " argmax=" << argmax
              << " (vLLM greedy set {" << ws << "}) " << (member ? "OK" : "MISMATCH")
              << (want.size() > 1 ? " [near-tie]" : ""));
      CHECK_MESSAGE(member, "mistral p" << ci << " step" << step
                    << ": our argmax " << argmax << " not in vLLM greedy set");
      // Advance the greedy prefix with vLLM's deterministic token where well-posed
      // (keeps the teacher-forced prefix aligned even if our near-tie flip differs);
      // at the sole near-tie, our own member token continues.
      tokens.push_back(want.size() == 1 ? want[0] : argmax);
    }
  }
  MESSAGE("mistral CUDA greedy-vs-oracle: " << total_ok << "/" << total_steps
          << " greedy tokens match vLLM");
}
