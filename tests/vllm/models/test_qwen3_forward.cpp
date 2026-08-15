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

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
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

// Qwen3DenseDecodeGraph::Step EAGER-FALLBACK == Qwen3DenseModel::Forward. The
// shared pure-dense decode CUDA-graph driver (qwen3.cpp) that the five dense
// registrations pick up. Its CUDA capture/replay is dgx-only, but on CPU
// support_static_graph_mode() is false, so Step() takes the eager ForwardBody
// fallback — which MUST be bit-identical to Qwen3DenseModel::Forward for the same
// pure-decode step. This exercises the new class end-to-end on CI (its
// construction, BuildPaddedDecodeAttn at S==B, and the eager-fallback wiring +
// device-logits carrier), catching any wiring regression without a GPU.
TEST_CASE("qwen3 dense decode-graph: CPU eager-fallback Step == Forward (pure decode)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();

  // A single-token PURE-DECODE step (num_actual_tokens == num_reqs == 1) against a
  // one-block cache. Two independent, identically zero-initialised caches so the
  // Forward and Step KV writes never alias.
  const int64_t block_size = 8;
  const std::vector<int32_t> tokens = {42};
  const std::vector<int32_t> positions = {3};  // some decode position within block 0
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = 1;
  am.query_start_loc = {0, 1};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {4};  // context length 4 (positions 0..3)
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = 4;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping = {3};
  am.causal = true;

  CachePool pool_fwd(c, /*num_blocks=*/2, block_size);
  CachePool pool_step(c, /*num_blocks=*/2, block_size);

  const std::vector<float> fwd = vllm::Qwen3DenseModel::Forward(
      tokens, positions, am, pool_fwd.attn_kv, w, c, q);
  REQUIRE(fwd.size() == static_cast<size_t>(c.vocab_size));

  vllm::Qwen3DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/8);
  const vllm::ForwardLogits step = graph.Step(tokens, positions, am, pool_step.attn_kv);
  // CPU: the graph is disabled, so Step returns the eager fallback's device-resident
  // [1, vocab] logits (device == host on the CPU backend, directly readable).
  REQUIRE(step.rows == 1);
  REQUIRE(step.vocab == c.vocab_size);
  REQUIRE(step.on_device());
  const auto* sp = static_cast<const float*>(step.device_tensor.data);
  REQUIRE(sp != nullptr);

  // Bit-identical: Step's eager fallback runs the exact same op sequence as Forward.
  CHECK(std::memcmp(fwd.data(), sp, fwd.size() * sizeof(float)) == 0);
  CHECK_FALSE(graph.captured());  // no capture on CPU
}

// --- The `inputs_embeds` ENTRY (MODEL-MUSIC-MUSIC3 W2, #672) -----------------
//
// Qwen3DenseModel::ForwardEmbeds starts the hidden stream from caller-supplied
// [T, H] bf16 rows instead of an embed_tokens lookup — the door
// `Qwen3Model.forward(inputs_embeds=...)` has upstream and that the DENSE
// registration never wired, which MiniMax-Music3's frame feedback needs because
// its input is a SUM of embedding rows and corresponds to no vocabulary entry.
//
// The claim these cases make is ADDITIVITY, and it is the whole reason the entry
// can exist on a token-exact row: feeding the embedding OF THE SAME IDS through
// the new entry must reproduce Forward BIT FOR BIT, in the logits AND in the
// paged KV it wrote, because the only thing that differs is where the first
// [T, H] buffer came from. A non-additive change — an extra scale, a norm, a
// different residual init, a dropped gather — moves those bytes.
namespace {

// embed_tokens[id] for each id, as the raw bf16 rows the entry consumes. The
// table is the checkpoint layout [vocab, H], NOT transposed.
std::vector<uint16_t> EmbedRows(const Qwen3DenseWeights& w, const HfConfig& c,
                                const std::vector<int32_t>& ids) {
  const int64_t H = c.hidden_size;
  const auto* table = reinterpret_cast<const uint16_t*>(w.embed_tokens.bytes.data());
  std::vector<uint16_t> rows(ids.size() * static_cast<size_t>(H));
  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t j = 0; j < H; ++j) {
      rows[t * static_cast<size_t>(H) + static_cast<size_t>(j)] =
          table[static_cast<size_t>(ids[t]) * static_cast<size_t>(H) + static_cast<size_t>(j)];
    }
  }
  return rows;
}

}  // namespace

TEST_CASE("qwen3 dense inputs_embeds: the embedding of the same ids is BIT-IDENTICAL") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();

  const int64_t T = 5;
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const CommonAttentionMetadata am = PrefillMeta(T, 8);

  // Two independent, identically zero-initialised caches so the two forwards'
  // KV writes never alias — the second arm must not read the first's residue.
  CachePool pool_ids(c, /*num_blocks=*/2, /*block_size=*/8);
  CachePool pool_emb(c, /*num_blocks=*/2, /*block_size=*/8);

  const std::vector<float> by_ids =
      vllm::Qwen3DenseModel::Forward(tokens, positions, am, pool_ids.attn_kv, w, c, q);
  std::vector<float> hidden;
  const std::vector<float> by_embeds = vllm::Qwen3DenseModel::ForwardEmbeds(
      EmbedRows(w, c, tokens), positions, am, pool_emb.attn_kv, w, c, q, {}, &hidden);

  REQUIRE(by_ids.size() == static_cast<size_t>(T * c.vocab_size));
  REQUIRE(by_embeds.size() == by_ids.size());
  int64_t compared = 0;
  int64_t differing = 0;
  for (size_t i = 0; i < by_ids.size(); ++i) {
    ++compared;
    if (by_ids[i] != by_embeds[i]) ++differing;
  }
  MESSAGE("inputs_embeds vs token ids: " << compared << " logits compared, " << differing
                                         << " differing");
  CHECK(compared == T * c.vocab_size);
  CHECK(differing == 0);

  // The KV the two arms wrote is the same too — a forward that agreed on logits
  // while writing a different cache would decode differently on the NEXT step,
  // which no logits comparison at this step can see.
  int64_t kv_values = 0;
  int64_t kv_differing = 0;
  for (size_t l = 0; l < pool_ids.buf.size(); ++l) {
    REQUIRE(pool_ids.buf[l].size() == pool_emb.buf[l].size());
    for (size_t i = 0; i < pool_ids.buf[l].size(); ++i) {
      ++kv_values;
      if (pool_ids.buf[l][i] != pool_emb.buf[l][i]) ++kv_differing;
    }
  }
  MESSAGE("paged KV: " << kv_values << " values compared, " << kv_differing << " differing");
  CHECK(kv_values > 0);
  CHECK(kv_differing == 0);

  // `out_hidden` is the SAME post-final-norm tensor ForwardHidden returns, taken
  // from the forward that produced the logits rather than from a second pass.
  CachePool pool_hid(c, /*num_blocks=*/2, /*block_size=*/8);
  const vllm::ForwardLogits pooled = vllm::Qwen3DenseModel::ForwardHidden(
      tokens, positions, am, pool_hid.attn_kv, w, c, q);
  REQUIRE(pooled.rows == T);
  REQUIRE(pooled.vocab == c.hidden_size);
  REQUIRE(hidden.size() == pooled.host.size());
  int64_t hidden_differing = 0;
  for (size_t i = 0; i < hidden.size(); ++i) {
    if (hidden[i] != pooled.host[i]) ++hidden_differing;
  }
  MESSAGE("out_hidden vs ForwardHidden: " << hidden.size() << " values compared, "
                                          << hidden_differing << " differing");
  CHECK(hidden.size() == static_cast<size_t>(T * c.hidden_size));
  CHECK(hidden_differing == 0);
}

TEST_CASE("qwen3 dense inputs_embeds: the logits_indices gather still binds") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();

  const int64_t T = 5;
  const std::vector<int32_t> tokens = {3, 17, 42, 8, 61};
  const std::vector<int32_t> positions = {0, 1, 2, 3, 4};
  const CommonAttentionMetadata am = PrefillMeta(T, 8);
  const std::vector<int32_t> last_only = {4};

  CachePool pool_full(c, 2, 8);
  CachePool pool_gathered(c, 2, 8);
  std::vector<float> hidden_full;
  std::vector<float> hidden_gathered;
  const std::vector<float> full = vllm::Qwen3DenseModel::ForwardEmbeds(
      EmbedRows(w, c, tokens), positions, am, pool_full.attn_kv, w, c, q, {}, &hidden_full);
  const std::vector<float> gathered = vllm::Qwen3DenseModel::ForwardEmbeds(
      EmbedRows(w, c, tokens), positions, am, pool_gathered.attn_kv, w, c, q, last_only,
      &hidden_gathered);

  // The gather returns ONE row, and it is the LAST one of the ungathered set —
  // which is the row Music3's loop reads (`last_hidden_state[:, -1]`).
  REQUIRE(gathered.size() == static_cast<size_t>(c.vocab_size));
  REQUIRE(hidden_gathered.size() == static_cast<size_t>(c.hidden_size));
  int64_t logit_diff = 0;
  for (int64_t v = 0; v < c.vocab_size; ++v) {
    if (gathered[static_cast<size_t>(v)] !=
        full[static_cast<size_t>((T - 1) * c.vocab_size + v)]) {
      ++logit_diff;
    }
  }
  int64_t hidden_diff = 0;
  for (int64_t j = 0; j < c.hidden_size; ++j) {
    if (hidden_gathered[static_cast<size_t>(j)] !=
        hidden_full[static_cast<size_t>((T - 1) * c.hidden_size + j)]) {
      ++hidden_diff;
    }
  }
  MESSAGE("gathered last row: " << c.vocab_size << " logits compared, " << logit_diff
                                << " differing; " << c.hidden_size << " hidden compared, "
                                << hidden_diff << " differing");
  CHECK(logit_diff == 0);
  CHECK(hidden_diff == 0);
}

TEST_CASE("qwen3 dense inputs_embeds: a non-rectangular embedding is refused BY NAME") {
  const HfConfig c = TinyConfig();
  const Qwen3DenseWeights w = TinyWeights(c);
  vt::Queue q = Q();
  CachePool pool(c, 2, 8);
  const CommonAttentionMetadata am = PrefillMeta(2, 8);
  // One value short of two full rows: the shape is unrecoverable, so it is named
  // rather than silently truncated to one row.
  std::vector<uint16_t> ragged(static_cast<size_t>(2 * c.hidden_size - 1), 0);
  CHECK_THROWS_WITH_AS(
      vllm::Qwen3DenseModel::ForwardEmbeds(ragged, {0, 1}, am, pool.attn_kv, w, c, q),
      doctest::Contains("inputs_embeds"), std::runtime_error);
}

// --- NVFP4 W4A16 synthetic forward (the QUANT-SCHEME additivity doctest) -----
// ADDITIVE-QUANT W3 for Qwen3-32B-NVFP4A16 (`Qwen3ForCausalLM` +
// compressed-tensors NVFP4A16). Builds the SAME tiny model twice — once with
// synthetic NVFP4 W4A16 operands, once with those operands DEQUANTED to BF16 —
// and asserts the two forwards agree. That is the semantic contract of a
// weight-only quantization scheme: W4A16 is defined to compute
// `x @ dequant(W).T`, so a correct loader + dispatcher must be numerically
// indistinguishable from running the dequantized weights through the BF16 arm.
//
// This is the CPU-side, checkpoint-free proof that (a) the `.weight_packed`
// probe routes into the fp4 fields, (b) the merged qkv / split gate_up operands
// are laid out consistently with what the forward expects, and (c) the
// divisor-reciprocal scale convention composes end-to-end. The Marlin GEMM
// itself is CUDA-only and is gated on dgx by
// tests/vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp.
namespace {

// One synthetic NVFP4 W4A16 operand [N=out, K=in]: pseudo-random E2M1 nibble
// pairs, pseudo-random fp8-e4m3 group-16 scales near 1.0, and a per-tensor
// DIVISOR (the compressed-tensors convention) whose reciprocal becomes scale2.
vllm::Nvfp4Weight MakeNvfp4(int64_t n, int64_t k, uint32_t seed,
                            float divisor = 448.0f) {
  REQUIRE(k % 16 == 0);
  vllm::Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.weight_global_scale_inv = divisor;
  w.scale2 = 1.0f / divisor;
  w.alpha = 0.0f;  // W4A16 — no activation quant
  w.packed = vllm::dense_loaders::MakeOwned(DType::kI8, {n, k / 2});
  w.scale = vllm::dense_loaders::MakeOwned(DType::kI8, {n, k / 16});
  std::mt19937 rng(seed);
  auto* p = reinterpret_cast<uint8_t*>(w.packed.bytes.data());
  for (size_t i = 0; i < w.packed.bytes.size(); ++i)
    p[i] = static_cast<uint8_t>(rng() & 0xFFu);
  // fp8-e4m3 bytes around 1.0 (0x38 == exponent 7 == 2^0), sign bit clear:
  // compressed-tensors block scales are non-negative by construction.
  auto* s = reinterpret_cast<uint8_t*>(w.scale.bytes.data());
  for (size_t i = 0; i < w.scale.bytes.size(); ++i)
    s[i] = static_cast<uint8_t>(0x30u + (rng() % 0x10u));
  return w;
}

// The bf16 [N,K] raw-NK dequant of an NVFP4 operand — the reference the W4A16
// path must reproduce.
vllm::OwnedTensor DequantToBf16RawNK(const vllm::Nvfp4Weight& w) {
  vllm::OwnedTensor o = vllm::dense_loaders::MakeOwned(DType::kBF16, {w.n, w.k});
  o.nk = true;
  vllm::DequantNvfp4ToBf16(
      reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
      reinterpret_cast<const uint8_t*>(w.scale.bytes.data()), w.scale2, w.n, w.k,
      reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// MXFP4 analog of MakeNvfp4: E2M1 packed [n,k/2] + E8M0 scale [n,k/32] (group 32,
// no global). is_mxfp4=true so the forward routes it through the MXFP4 keep-quant
// path (Marlin on GPU, dequant fallback on CPU).
vllm::Nvfp4Weight MakeMxfp4(int64_t n, int64_t k, uint32_t seed) {
  REQUIRE(k % 32 == 0);
  vllm::Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.group_size = 32;
  w.is_mxfp4 = true;
  w.scale2 = 0.0f;  // no global
  w.alpha = 0.0f;   // W4A16
  w.packed = vllm::dense_loaders::MakeOwned(DType::kI8, {n, k / 2});
  w.scale = vllm::dense_loaders::MakeOwned(DType::kI8, {n, k / 32});
  std::mt19937 rng(seed);
  auto* p = reinterpret_cast<uint8_t*>(w.packed.bytes.data());
  for (size_t i = 0; i < w.packed.bytes.size(); ++i)
    p[i] = static_cast<uint8_t>(rng() & 0xFFu);
  // E8M0 bytes near 127 => scales ~2^-9..2^3 (finite, bf16 normal range).
  auto* s = reinterpret_cast<uint8_t*>(w.scale.bytes.data());
  for (size_t i = 0; i < w.scale.bytes.size(); ++i)
    s[i] = static_cast<uint8_t>(118u + (rng() % 15u));
  return w;
}

vllm::OwnedTensor DequantMxfp4ToBf16RawNK(const vllm::Nvfp4Weight& w) {
  vllm::OwnedTensor o = vllm::dense_loaders::MakeOwned(DType::kBF16, {w.n, w.k});
  o.nk = true;
  vllm::DequantMxfp4ToBf16(
      reinterpret_cast<const uint8_t*>(w.packed.bytes.data()),
      reinterpret_cast<const uint8_t*>(w.scale.bytes.data()), w.n, w.k,
      reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// Merge two raw-NK bf16 [N,K] tensors by output-row concat (the BF16 arm's
// gate_up layout, which the fp4 arm keeps as two separate operands).
vllm::OwnedTensor ConcatRawNK(const vllm::OwnedTensor& a,
                              const vllm::OwnedTensor& b) {
  vllm::OwnedTensor o = vllm::dense_loaders::MakeOwned(
      DType::kBF16, {a.shape[0] + b.shape[0], a.shape[1]});
  o.nk = true;
  std::memcpy(o.bytes.data(), a.bytes.data(), a.bytes.size());
  std::memcpy(o.bytes.data() + a.bytes.size(), b.bytes.data(), b.bytes.size());
  return o;
}

}  // namespace

TEST_CASE("qwen3 dense forward: NVFP4 W4A16 == BF16-on-dequantized (CPU synthetic)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t I = c.intermediate_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;

  // Shared non-Linear weights: the embed table and every norm stay BF16 in a
  // compressed-tensors checkpoint (no config group targets them).
  Qwen3DenseWeights base = TinyWeights(c);

  Qwen3DenseWeights fp4 = TinyWeights(c);
  Qwen3DenseWeights deq = TinyWeights(c);
  uint32_t seed = 7000;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    auto& fl = fp4.layers[static_cast<size_t>(l)];
    auto& dl = deq.layers[static_cast<size_t>(l)];

    vllm::Nvfp4Weight qkv = MakeNvfp4(qdim + 2 * kdim, H, seed++);
    vllm::Nvfp4Weight o = MakeNvfp4(H, qdim, seed++);
    vllm::Nvfp4Weight g = MakeNvfp4(I, H, seed++);
    vllm::Nvfp4Weight u = MakeNvfp4(I, H, seed++);
    vllm::Nvfp4Weight dn = MakeNvfp4(H, I, seed++);

    // The BF16 mirror: exactly dequant(W), so any difference in the result is a
    // W4A16 dispatch/layout bug, not a quantization-error artifact.
    dl.attn.qkv_proj = DequantToBf16RawNK(qkv);
    dl.attn.o_proj = DequantToBf16RawNK(o);
    dl.mlp.gate_up_proj = ConcatRawNK(DequantToBf16RawNK(g), DequantToBf16RawNK(u));
    dl.mlp.down_proj = DequantToBf16RawNK(dn);

    // The QUANTIZED arm: fp4 fields populated, BF16 fields cleared so the
    // forward's `IsNvfp4()` dispatch is the only thing that can select a path.
    fl.attn.qkv_proj = vllm::OwnedTensor{};
    fl.attn.o_proj = vllm::OwnedTensor{};
    fl.mlp.gate_up_proj = vllm::OwnedTensor{};
    fl.mlp.down_proj = vllm::OwnedTensor{};
    fl.attn.qkv_proj_fp4 = std::move(qkv);
    fl.attn.o_proj_fp4 = std::move(o);
    fl.mlp.gate_proj_fp4 = std::move(g);
    fl.mlp.up_proj_fp4 = std::move(u);
    fl.mlp.down_proj_fp4 = std::move(dn);

    REQUIRE(fl.attn.IsNvfp4());
    REQUIRE(fl.mlp.IsNvfp4());
    REQUIRE_FALSE(dl.attn.IsNvfp4());
    REQUIRE_FALSE(dl.mlp.IsNvfp4());
  }

  vllm::dense_nvfp4::ResetW4A16Stats();
  const std::vector<float> a = RunForward(c, fp4);
  const vllm::dense_nvfp4::Nvfp4W4A16Stats st = vllm::dense_nvfp4::GetW4A16Stats();
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));

  // POSITIVE SIGNAL: the W4A16 dispatcher actually ran. On CPU there is no
  // Marlin, so every quantized GEMM lands in the dequant fallback — 5 per layer
  // (qkv, o, gate, up, down).
  MESSAGE("NVFP4 W4A16 CPU counters: fallback_gemms=" << st.fallback_gemms
          << " marlin_gemms=" << st.marlin_gemms);
  CHECK(st.marlin_gemms == 0);  // CPU: Marlin is not available
  CHECK(st.fallback_gemms == static_cast<uint64_t>(5 * c.num_hidden_layers));

  // The semantic contract: W4A16 == BF16 on the dequantized weights. Both arms
  // multiply the SAME bf16 values; only the GEMM's K-reduction order differs
  // (MatmulBT over raw-NK vs Matmul over the transposed dequant), so the logits
  // agree to float round-off rather than bit-exactly.
  const std::vector<float> b = RunForward(c, deq);
  REQUIRE(b.size() == a.size());
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(a[i] - b[i])));
  MESSAGE("NVFP4 W4A16 vs BF16-on-dequantized: max |dlogit| = " << max_abs);
  CHECK(max_abs < 1e-2);

  // Determinism: a re-run of the quantized arm is bit-identical.
  const std::vector<float> a2 = RunForward(c, fp4);
  CHECK(std::memcmp(a.data(), a2.data(), a.size() * sizeof(float)) == 0);
}

// MXFP4 analog: the FULL model forward with MXFP4 W4A16 weights must equal the
// forward on the SAME weights dequantized to BF16 — so any difference is a MXFP4
// dispatch/layout/merge bug in the model integration, NOT a quantization artifact.
// On GPU this exercises the Marlin mxf4 path THROUGH the model (merged qkv, split
// gate_up, o/down) — the exact path the e2e uses, which the op-level unit gate does
// not (it bypasses the loader/merge/model). RED-first for the e2e residual.
TEST_CASE("qwen3 dense forward: MXFP4 W4A16 == BF16-on-dequantized (synthetic)") {
  setenv("VT_FUSED_CHAIN_ADOPT", "1", 1);
  const HfConfig c = TinyConfig();
  const int64_t H = c.hidden_size, Hq = c.num_attention_heads;
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t I = c.intermediate_size;
  const int64_t qdim = Hq * Dh, kdim = Hkv * Dh;

  Qwen3DenseWeights fp4 = TinyWeights(c);
  Qwen3DenseWeights deq = TinyWeights(c);
  uint32_t seed = 9000;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    auto& fl = fp4.layers[static_cast<size_t>(l)];
    auto& dl = deq.layers[static_cast<size_t>(l)];

    vllm::Nvfp4Weight qkv = MakeMxfp4(qdim + 2 * kdim, H, seed++);
    vllm::Nvfp4Weight o = MakeMxfp4(H, qdim, seed++);
    vllm::Nvfp4Weight g = MakeMxfp4(I, H, seed++);
    vllm::Nvfp4Weight u = MakeMxfp4(I, H, seed++);
    vllm::Nvfp4Weight dn = MakeMxfp4(H, I, seed++);

    dl.attn.qkv_proj = DequantMxfp4ToBf16RawNK(qkv);
    dl.attn.o_proj = DequantMxfp4ToBf16RawNK(o);
    dl.mlp.gate_up_proj =
        ConcatRawNK(DequantMxfp4ToBf16RawNK(g), DequantMxfp4ToBf16RawNK(u));
    dl.mlp.down_proj = DequantMxfp4ToBf16RawNK(dn);

    fl.attn.qkv_proj = vllm::OwnedTensor{};
    fl.attn.o_proj = vllm::OwnedTensor{};
    fl.mlp.gate_up_proj = vllm::OwnedTensor{};
    fl.mlp.down_proj = vllm::OwnedTensor{};
    fl.attn.qkv_proj_fp4 = std::move(qkv);
    fl.attn.o_proj_fp4 = std::move(o);
    fl.mlp.gate_proj_fp4 = std::move(g);
    fl.mlp.up_proj_fp4 = std::move(u);
    fl.mlp.down_proj_fp4 = std::move(dn);

    REQUIRE(fl.attn.IsNvfp4());  // emptiness-based dispatch; true for mxfp4 too
    REQUIRE(fl.mlp.IsNvfp4());
    REQUIRE_FALSE(dl.attn.IsNvfp4());
    REQUIRE_FALSE(dl.mlp.IsNvfp4());
  }

  vllm::dense_nvfp4::ResetW4A16Stats();
  const std::vector<float> a = RunForward(c, fp4);
  const vllm::dense_nvfp4::Nvfp4W4A16Stats st = vllm::dense_nvfp4::GetW4A16Stats();
  REQUIRE(a.size() == static_cast<size_t>(5 * c.vocab_size));
  for (float x : a) REQUIRE(std::isfinite(x));
  MESSAGE("MXFP4 W4A16 counters: fallback_gemms=" << st.fallback_gemms
          << " marlin_gemms=" << st.marlin_gemms);

  const std::vector<float> b = RunForward(c, deq);
  REQUIRE(b.size() == a.size());
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(a[i] - b[i])));
  MESSAGE("MXFP4 W4A16 vs BF16-on-dequantized: max |dlogit| = " << max_abs);
  CHECK(max_abs < 1e-2);

  const std::vector<float> a2 = RunForward(c, fp4);
  CHECK(std::memcmp(a.data(), a2.data(), a.size() * sizeof(float)) == 0);
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

#ifdef VT_MARLIN_NVFP4
// QUANT-CT-MXFP4-CLOSERS sliver (a): the shared dense NVFP4/MXFP4 W4A16 route
// forces moe_block_size=8 at M<=8 (dense_nvfp4_gemm.h::DenseAlignFor), matching
// vLLM's dense marlin m_block_size_8 = prob_m<=8, while leaving M>8 on the raw
// MoE-align selection. Direct launch-config assertion: RED against the pre-fix
// code, whose DenseAlignFor returned MarlinMoeAlignBlockSizeSelect(M,1,1) == 16
// at M=8. DenseAlignFor allocs device memory + runs the moe_align kernel, so
// this is CUDA-only. Only dense_nvfp4_gemm.h is changed — the qwen3_5.cpp twin
// (27B/35B gate models' dense attn + shared-expert) is deliberately untouched.
TEST_CASE("dense_nvfp4 DenseAlignFor forces block=8 at M<=8 (dgx-only)") {
  vt::Backend* cuda = nullptr;
  try {
    cuda = &vt::GetBackend(vt::DeviceType::kCUDA);
  } catch (...) {
    MESSAGE("SKIP: no CUDA backend registered");
    return;
  }
  vt::Queue q = cuda->CreateQueue();
  vllm::dense_nvfp4::Dev d{*cuda, q};
  // The pre-fix production route at M=8 was 16 (padded tile); the sliver forces 8.
  CHECK(vt::cuda::MarlinMoeAlignBlockSizeSelect(8, 1, 1) == 16);
  for (int M : {1, 2, 4, 8}) {
    CAPTURE(M);
    CHECK(vllm::dense_nvfp4::DenseAlignFor(d, M).block == 8);
  }
  // M>8 unchanged: still the MoE-align pick (16 at M in (8,16], 32 at M=16).
  for (int M : {12, 16}) {
    CAPTURE(M);
    CHECK(vllm::dense_nvfp4::DenseAlignFor(d, M).block ==
          vt::cuda::MarlinMoeAlignBlockSizeSelect(M, 1, 1));
  }
  cuda->DestroyQueue(q);
}
#endif  // VT_MARLIN_NVFP4

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
