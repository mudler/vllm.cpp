// vllm.cpp original (full-attention MoE W3 forward doctest); no upstream mirror.
//
// CPU synthetic forward for the Qwen3-Coder MoE (`Qwen3MoeForCausalLM`) path
// (Qwen3MoeModel::Forward, src/vllm/model_executor/models/qwen3_moe.cpp). Builds a
// tiny random Qwen3MoeWeights (router + per-expert bf16 SwiGLU experts, NO shared
// expert, UNTIED lm_head) + a single-sequence paged KV cache and runs the whole op
// chain on CPU (no checkpoint needed, runs in CI). It exercises the full wiring —
// embed -> N MoE layers (std add+RMSNorm, dense AttnBlock per-head q/k norm + RoPE
// + causal paged attention, router softmax+top-k+renorm, per-expert bf16 MLP,
// combine) -> final norm -> untied lm_head — and asserts:
//   (1) the forward runs and returns finite [T,vocab] logits;
//   (2) it is deterministic (re-run bit-identical);
//   (3) the fusion-catalog ADOPT path (VT_FUSED_CHAIN_ADOPT=1) is BYTE-IDENTICAL
//       to the hand-call fallback (=0) — the additive recipe reuse is bit-exact.
// The real prefill argmax vs the checkpoint is the dgx-only case below; the
// token-exact vs-oracle bar is W4 (test_qwen3_moe_paged_engine.cpp).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_moe.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3MoeWeights;
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

HfConfig TinyConfig() {
  HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 64;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 16;  // partial_rotary_factor 1.0
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000000.0;  // Qwen3-Coder rope_theta
  c.vocab_size = 100;
  // MoE fields.
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 32;
  c.shared_expert_intermediate_size = 0;  // NO shared expert
  return c;
}

Qwen3MoeWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, V = c.vocab_size;
  const int64_t E = c.num_experts, I = c.moe_intermediate_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3MoeWeights w;
  w.tie_word_embeddings = false;  // UNTIED
  w.attention_bias = false;
  w.embed_tokens = MakeBf16({V, H}, /*nk=*/false, 1);
  w.final_norm = MakeBf16({H}, false, 2, 0.5f);
  w.lm_head = MakeBf16({H, V}, /*nk=*/false, 3);  // Matmul-B [H,V] untied
  uint32_t seed = 100;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    vllm::Qwen3MoeLayerWeights lw;
    lw.input_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.post_attention_layernorm = MakeBf16({H}, false, seed++, 0.5f);
    lw.attn.qkv_proj = MakeBf16({qdim + 2 * kdim, H}, /*nk=*/true, seed++);
    lw.attn.o_proj = MakeBf16({H, qdim}, /*nk=*/true, seed++);
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    // MoE: router gate [H,E] Matmul-B + per-expert bf16 gate/up [H,I], down [I,H].
    lw.moe.router_gate = MakeBf16({H, E}, /*nk=*/false, seed++);
    for (int64_t e = 0; e < E; ++e) {
      lw.moe.expert_gate.push_back(MakeBf16({H, I}, /*nk=*/false, seed++));
      lw.moe.expert_up.push_back(MakeBf16({H, I}, /*nk=*/false, seed++));
      lw.moe.expert_down.push_back(MakeBf16({I, H}, /*nk=*/false, seed++));
    }
    // NO shared expert (shared_* left empty).
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// Single-sequence prefill paged KV cache (all layers full-attention).
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

std::vector<float> RunForward(const HfConfig& c, const Qwen3MoeWeights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  return vllm::Qwen3MoeModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("qwen3-moe forward: CPU synthetic runs, finite, deterministic") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const Qwen3MoeWeights w = TinyWeights(c);

  const std::vector<float> a = RunForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  // Determinism: a re-run is bit-identical.
  const std::vector<float> b = RunForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

TEST_CASE("qwen3-moe forward: fusion-catalog ADOPT == hand-call fallback (byte-exact)") {
  const HfConfig c = TinyConfig();
  const Qwen3MoeWeights w = TinyWeights(c);

  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const std::vector<float> adopt = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "0", 1);
  const std::vector<float> hand = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);

  // kFusedAddRmsNormStd (Tier-0 composite) dispatches to the SAME standalone ops
  // the fallback hand-calls -> byte-identical logits. (The MoE reference path is
  // env-flag-free, so any divergence is the residual add+RMSNorm fusion alone.)
  REQUIRE(adopt.size() == hand.size());
  CHECK(std::memcmp(adopt.data(), hand.data(), adopt.size() * sizeof(float)) == 0);
}

namespace {
namespace fs = std::filesystem;
std::string FindQwen3CoderSnap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/"
      "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "model.safetensors.index.json", ec))
      return e.path().string();
  return "";
}

std::vector<vllm::SafetensorsFile> OpenShards(const std::string& snap) {
  const std::map<std::string, std::string> wmap =
      vllm::LoadSafetensorsIndex(snap + "/model.safetensors.index.json");
  std::unordered_set<std::string> files;
  for (const auto& [tensor, file] : wmap) files.insert(file);
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(files.size());
  for (const std::string& f : files)
    shards.push_back(vllm::SafetensorsFile::Open(snap + "/" + f));
  return shards;
}
}  // namespace

// DIAGNOSTIC (dgx-only): load the REAL Qwen3-Coder-30B-A3B and run a single-seq CPU
// prefill over prompt0 = "The capital of France is" [785,6722,315,9625,374]; assert
// the logits are finite, the run is deterministic, and REPORT the last-token argmax
// + top5 (the W4 token-exact gate owns the exact-token assertion — this isolates
// the MoE forward from the paged engine / sampler and confirms it produces sane,
// non-garbage output).
TEST_CASE("qwen3-moe forward: real Qwen3-Coder prefill argmax sanity (dgx-only)") {
  const std::string snap = FindQwen3CoderSnap();
  if (snap.empty()) {
    MESSAGE("SKIP: Qwen3-Coder-30B-A3B checkpoint absent (dgx-only forward sanity)");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  const std::vector<vllm::SafetensorsFile> shards = OpenShards(snap);
  const vllm::Qwen3MoeWeights w =
      vllm::LoadQwen3MoeForCausalLMWeights(shards, cfg);

  const std::vector<int32_t> tokens = {785, 6722, 315, 9625, 374};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const int64_t T = static_cast<int64_t>(tokens.size());

  const bool bf16_cache = std::getenv("VT_KV_CACHE_F32") == nullptr;
  const DType cdt = bf16_cache ? DType::kBF16 : DType::kF32;
  const int64_t bs = 16, Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim;
  std::vector<std::vector<uint8_t>> buf;
  std::vector<PagedKvCache> attn_kv;
  const size_t cbytes = static_cast<size_t>(1 * 2 * bs * Hkv * Dh) * vt::SizeOf(cdt);
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) buf.emplace_back(cbytes, 0);
  for (auto& b : buf) {
    PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = cdt;
    kv.num_blocks = 1;
    kv.block_size = bs;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
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

  vt::Queue q = Q();
  const std::vector<float> logits =
      vllm::Qwen3MoeModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
  REQUIRE(logits.size() == static_cast<size_t>(T) * cfg.vocab_size);
  for (float x : logits) REQUIRE(std::isfinite(x));

  const int64_t V = cfg.vocab_size;
  const float* last = logits.data() + (T - 1) * V;
  int argmax = 0;
  for (int64_t v = 1; v < V; ++v)
    if (last[v] > last[argmax]) argmax = static_cast<int>(v);
  std::vector<int> idx(static_cast<size_t>(V));
  for (int64_t v = 0; v < V; ++v) idx[static_cast<size_t>(v)] = static_cast<int>(v);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return last[a] > last[b]; });
  MESSAGE("qwen3-moe prefill argmax=" << argmax << " top5=[" << idx[0] << ","
          << idx[1] << "," << idx[2] << "," << idx[3] << "," << idx[4] << "]");

  // Determinism on the real checkpoint (re-run bit-identical).
  const std::vector<float> logits2 =
      vllm::Qwen3MoeModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
  CHECK(std::memcmp(logits.data(), logits2.data(),
                    logits.size() * sizeof(float)) == 0);
}
