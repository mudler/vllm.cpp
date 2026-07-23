// vllm.cpp original (cross-family additive-model W3 forward doctest); no upstream
// mirror.
//
// CPU synthetic forward for the DENSE Llama (`LlamaForCausalLM`) path. Llama reuses
// the shared dense forward (LlamaModel == Qwen3DenseModel) with two additive deltas
// vs Qwen3 — NO per-head q/k RMSNorm, and llama3 RoPE frequency scaling — so this
// doctest exercises exactly those seams on CPU (no checkpoint, runs in CI):
//   (1) qk-norm-ABSENT attention: q_norm/k_norm left EMPTY -> the shared AttnBlock
//       skips the norm step and the forward still runs, finite + deterministic;
//   (2) llama3 rope-scaling is ACTUALLY applied: the same weights with rope_type
//       "llama3" vs "default" produce DIFFERENT logits (the piecewise inv_freq
//       rescale changed the RoPE angles), and re-running each is bit-identical;
//   (3) the fusion-catalog ADOPT path (kFusedAddRmsNormStd) is byte-identical to
//       the hand-call fallback for the qk-norm-absent forward.
// The real token-exact vs-oracle bar is tests/parity/test_llama_paged_engine.cpp
// (dgx-only).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <filesystem>
#include <string>
#include <system_error>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/llama.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::LlamaWeights;
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

// Tiny Llama config: GQA 4/2, head_dim 16 (llama-shaped: rotary over the full
// head), rope_theta 500000 + llama3 scaling matching Llama-3.2's rope_scaling.
HfConfig TinyConfig(bool llama3_scaling) {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;  // partial_rotary_factor 1.0 (Llama rotates the whole head)
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-5;  // Llama-3.2 rms_norm_eps
  c.rope_theta = 500000.0;
  c.vocab_size = 100;
  if (llama3_scaling) {
    c.rope_parameters.rope_type = "llama3";
    c.rope_parameters.rope_theta = 500000.0;
    c.rope_parameters.factor = 32.0;
    c.rope_parameters.low_freq_factor = 1.0;
    c.rope_parameters.high_freq_factor = 4.0;
    c.rope_parameters.original_max_position_embeddings = 8192;
  }
  return c;
}

// Llama dense weights: NO q_norm/k_norm (left empty), tied lm_head.
LlamaWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  LlamaWeights w;
  w.tie_word_embeddings = true;
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3DenseLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    // NO q_norm/k_norm — Llama has no qk-norm (the seam this doctest exercises).
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

std::vector<float> RunForward(const HfConfig& c, const LlamaWeights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  return vllm::LlamaModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("llama dense forward: qk-norm-absent CPU synthetic runs, finite, deterministic") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig(/*llama3_scaling=*/true);
  const LlamaWeights w = TinyWeights(c);
  // Precondition: this is genuinely a qk-norm-absent model.
  REQUIRE(w.layers[0].attn.q_norm.Empty());
  REQUIRE(w.layers[0].attn.k_norm.Empty());

  const std::vector<float> a = RunForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  const std::vector<float> b = RunForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("llama dense forward: llama3 rope-scaling is applied (differs from plain rope)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c_scaled = TinyConfig(/*llama3_scaling=*/true);
  const HfConfig c_plain = TinyConfig(/*llama3_scaling=*/false);
  const LlamaWeights w = TinyWeights(c_scaled);  // same weights both ways

  const std::vector<float> scaled = RunForward(c_scaled, w);
  const std::vector<float> plain = RunForward(c_plain, w);
  REQUIRE(scaled.size() == plain.size());

  // The llama3 piecewise inv_freq rescale changes the RoPE angles for the
  // low/mid-frequency bands, so the logits MUST differ from plain RoPE (over the
  // 5-token prefill, positions 1..4 are non-trivially rotated). A no-op scaling
  // would make these identical — this is the positive signal that the new kernel
  // path fired.
  bool differs = false;
  for (size_t i = 0; i < scaled.size() && !differs; ++i)
    if (scaled[i] != plain[i]) differs = true;
  CHECK(differs);

  // Each configuration is itself deterministic.
  const std::vector<float> scaled2 = RunForward(c_scaled, w);
  CHECK(std::memcmp(scaled.data(), scaled2.data(), scaled.size() * sizeof(float)) == 0);
}

namespace {
namespace fs = std::filesystem;
std::string FindLlamaSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
      ".cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "model.safetensors", ec)) return e.path().string();
  return "";
}
}  // namespace

// DIAGNOSTIC (dgx-only, GPU): feed vLLM's EXACT prompt tokens for the tok0-diverging
// prompts through our CUDA prefill and check the last-position argmax == vLLM's first
// greedy token. Isolates the FORWARD from tokenization + decode + sampler. If this
// passes, the paged-gate divergences are a decode/near-tie issue; if it fails, the
// prefill forward (attention head_dim 64 / llama3 rope / weights) is wrong.
TEST_CASE("llama dense forward: real Llama-3.2-1B CUDA prefill argmax vs oracle (dgx-only)") {
  const std::string snap = FindLlamaSnap();
  if (snap.empty()) {
    MESSAGE("SKIP: Llama-3.2-1B checkpoint absent (CUDA prefill argmax check)");
    return;
  }
  vt::Backend* cuda = nullptr;
  try { cuda = &vt::GetBackend(vt::DeviceType::kCUDA); }
  catch (...) { MESSAGE("SKIP: no CUDA backend registered"); return; }

  const vllm::HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(snap + "/model.safetensors"));
  const vllm::LlamaWeights w = vllm::LoadLlamaForCausalLMWeights(shards, cfg);

  struct Case { std::vector<int32_t> toks; int expect; };
  const std::vector<Case> cases = {
      {{128000, 791, 6864, 315, 9822, 374}, 12366},                       // p0
      {{128000, 791, 7928, 11841, 304, 1057, 13238, 1887, 374}, 264},     // p8
      {{128000, 22333, 6975, 374, 264, 1207, 2630, 315}, 21075},          // p9
      {{128000, 36, 17239, 296, 272}, 53363},                             // p13
  };
  const int64_t Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim;
  int pass = 0;
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const std::vector<int32_t>& tokens = cases[ci].toks;
    const int64_t T = static_cast<int64_t>(tokens.size());
    std::vector<int32_t> positions(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    vt::Queue q = cuda->CreateQueue();
    const int64_t bs = 16;
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
        vllm::LlamaModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
    const int64_t V = cfg.vocab_size;
    const float* last = logits.data() + (T - 1) * V;
    int argmax = 0;
    for (int64_t v = 1; v < V; ++v) if (last[v] > last[argmax]) argmax = static_cast<int>(v);
    MESSAGE("llama prefill p" << ci << " argmax=" << argmax << " (want " << cases[ci].expect << ")");
    if (argmax == cases[ci].expect) ++pass;
    for (void* p : devbuf) cuda->Free(p);
  }
  MESSAGE("llama CUDA prefill argmax: " << pass << "/" << cases.size() << " match vLLM greedy tok0");
  CHECK(pass == static_cast<int>(cases.size()));
}

TEST_CASE("llama dense forward: fusion-catalog ADOPT == hand-call fallback (byte-exact)") {
  const HfConfig c = TinyConfig(/*llama3_scaling=*/true);
  const LlamaWeights w = TinyWeights(c);

  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const std::vector<float> adopt = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "0", 1);
  const std::vector<float> hand = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);

  REQUIRE(adopt.size() == hand.size());
  CHECK(std::memcmp(adopt.data(), hand.data(), adopt.size() * sizeof(float)) == 0);
}
