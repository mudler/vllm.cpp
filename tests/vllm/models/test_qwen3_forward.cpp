// vllm.cpp original (ADDITIVE-MODEL W3 forward doctest); no upstream mirror.
//
// CPU synthetic forward for the DENSE Qwen3 (`Qwen3ForCausalLM`) path
// (Qwen3DenseModel::Forward, src/vllm/model_executor/models/qwen3.cpp). Builds a
// tiny random Qwen3DenseWeights + a single-sequence paged KV cache and runs the
// whole op chain on CPU (no checkpoint needed, runs in CI). It exercises the full
// wiring — embed -> N layers (std add+RMSNorm, per-head q/k norm + RoPE, causal
// paged attention, SwiGLU MLP) -> final norm -> tied lm_head — and asserts:
//   (1) the forward runs and returns finite [T,vocab] logits;
//   (2) it is deterministic (re-run bit-identical);
//   (3) the fusion-catalog ADOPT path (VT_FUSED_CHAIN_ADOPT=1, kFusedAddRmsNormStd
//       + kAttnQkNormRope via vt::FusedChain) is BYTE-IDENTICAL to the hand-call
//       fallback (=0) — the additive recipe reuse is bit-exact.
// The real token-exact vs-oracle bar is test_qwen3_paged_engine.cpp (dgx-only).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <algorithm>
#include <filesystem>
#include <string>
#include <system_error>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::Qwen3DenseWeights;
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
  c.intermediate_size = 128;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 100;
  return c;
}

Qwen3DenseWeights TinyWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads, Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;
  Qwen3DenseWeights w;
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
    lw.attn.q_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.attn.k_norm = MakeBf16({Dh}, false, seed++, 0.5f);
    lw.mlp.gate_up_proj = MakeBf16({2 * I, H}, /*nk=*/true, seed++);
    lw.mlp.down_proj = MakeBf16({H, I}, /*nk=*/true, seed++);
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

std::vector<float> RunForward(const HfConfig& c, const Qwen3DenseWeights& w) {
  const int64_t T = 5;
  CachePool pool(c, /*num_blocks=*/2, /*block_size=*/8);
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  vt::Queue q = Q();
  return vllm::Qwen3DenseModel::Forward(tokens, positions, am, pool.attn_kv, w, c, q);
}

}  // namespace

TEST_CASE("qwen3 dense forward: CPU synthetic runs, finite, deterministic") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);

  const std::vector<float> a = RunForward(c, w);
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  // Determinism: a re-run is bit-identical.
  const std::vector<float> b = RunForward(c, w);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

namespace {
namespace fs = std::filesystem;
std::string FindQwen3Snap() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec))
    if (fs::exists(e.path() / "model.safetensors", ec)) return e.path().string();
  return "";
}
}  // namespace

// DIAGNOSTIC (dgx-only): load the REAL Qwen3-0.6B and run a single-sequence CPU
// prefill over prompt0 = "The capital of France is" [785,6722,315,9625,374];
// the last-token argmax must be 12095 (" Paris"), the oracle's first greedy
// token. Isolates the forward from the paged engine / sampler / KV growth.
TEST_CASE("qwen3 dense forward: real Qwen3-0.6B prefill argmax vs oracle (dgx-only)") {
  const std::string snap = FindQwen3Snap();
  if (snap.empty()) {
    MESSAGE("SKIP: Qwen3-0.6B checkpoint absent (dgx-only forward argmax check)");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(snap + "/model.safetensors"));
  const vllm::Qwen3DenseWeights w = vllm::LoadQwen3ForCausalLMWeights(shards, cfg);

  const std::vector<int32_t> tokens = {785, 6722, 315, 9625, 374};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const int64_t T = static_cast<int64_t>(tokens.size());

  // Single-seq prefill CPU KV cache: one block, block_size >= T. The cache dtype
  // mirrors the production runner (bf16 by default; VT_KV_CACHE_F32=0). Exercise
  // the SAME bf16 cache path the GPU engine uses.
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
      vllm::Qwen3DenseModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
  REQUIRE(logits.size() == static_cast<size_t>(T) * cfg.vocab_size);

  const int64_t V = cfg.vocab_size;
  const float* last = logits.data() + (T - 1) * V;
  int argmax = 0;
  for (int64_t v = 1; v < V; ++v)
    if (last[v] > last[argmax]) argmax = static_cast<int>(v);
  // top-5 for diagnostics
  std::vector<int> idx(static_cast<size_t>(V));
  for (int64_t v = 0; v < V; ++v) idx[static_cast<size_t>(v)] = static_cast<int>(v);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return last[a] > last[b]; });
  MESSAGE("qwen3 prefill argmax=" << argmax << " top5=[" << idx[0] << "," << idx[1]
          << "," << idx[2] << "," << idx[3] << "," << idx[4] << "] (want 12095)");
  CHECK(argmax == 12095);
}

// DIAGNOSTIC (dgx-only, GPU): the SAME prompt0 prefill but on a CUDA queue with a
// device-resident bf16 KV cache — isolates whether a CUDA-kernel numeric bug (vs
// the engine wiring) breaks the forward. argmax must again be 12095.
TEST_CASE("qwen3 dense forward: real Qwen3-0.6B CUDA prefill argmax (dgx-only)") {
  const std::string snap = FindQwen3Snap();
  if (snap.empty()) {
    MESSAGE("SKIP: Qwen3-0.6B checkpoint absent (CUDA forward argmax check)");
    return;
  }
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(snap + "/config.json");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(snap + "/model.safetensors"));
  const vllm::Qwen3DenseWeights w = vllm::LoadQwen3ForCausalLMWeights(shards, cfg);

  const std::vector<int32_t> tokens = {785, 6722, 315, 9625, 374};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const int64_t T = static_cast<int64_t>(tokens.size());
  vt::Queue q = cuda->CreateQueue();

  const int64_t bs = 16, Hkv = cfg.num_key_value_heads, Dh = cfg.head_dim;
  const size_t cbytes = static_cast<size_t>(1 * 2 * bs * Hkv * Dh) * vt::SizeOf(DType::kBF16);
  std::vector<void*> devbuf;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < cfg.num_hidden_layers; ++l) {
    void* p = cuda->Alloc(cbytes);
    cuda->Memset(q, p, 0, cbytes);
    devbuf.push_back(p);
    PagedKvCache kv;
    kv.data = p;
    kv.dtype = DType::kBF16;
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

  const std::vector<float> logits =
      vllm::Qwen3DenseModel::Forward(tokens, positions, m, attn_kv, w, cfg, q);
  const int64_t V = cfg.vocab_size;
  const float* last = logits.data() + (T - 1) * V;
  int argmax = 0;
  for (int64_t v = 1; v < V; ++v)
    if (last[v] > last[argmax]) argmax = static_cast<int>(v);
  MESSAGE("qwen3 CUDA prefill argmax=" << argmax << " (want 12095)");
  for (void* p : devbuf) cuda->Free(p);
  CHECK(argmax == 12095);
}

TEST_CASE("qwen3 dense forward: fusion-catalog ADOPT == hand-call fallback (byte-exact)") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);

  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const std::vector<float> adopt = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "0", 1);
  const std::vector<float> hand = RunForward(c, w);
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);

  // kFusedAddRmsNormStd + kAttnQkNormRope (Tier-0 composite) dispatch to the SAME
  // standalone ops the fallback hand-calls -> byte-identical logits.
  REQUIRE(adopt.size() == hand.size());
  CHECK(std::memcmp(adopt.data(), hand.data(), adopt.size() * sizeof(float)) == 0);
}
