// vllm.cpp original. DFlash D4 (DF-ENGINE-INTEGRATION) propose-brick unit tests.
// Ported semantics: vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
// (DflashSpeculator.propose :300-413, _generate_draft :242-273, greedy sample_draft)
// @ 555967922. These pin the load-bearing D4 invariants of the NON-autoregressive
// whole-block propose brick that swaps in for the MTP k=1 propose in the runner loop:
//   (1) SampleDflashBlockDrafts greedily argmaxes each of the k MASK positions and
//       SKIPS the anchor row (sample_from_anchor=false), assembling [num_reqs][k];
//   (2) RED — the anchor row's argmax never leaks into a draft; a per-request block
//       is isolated (request r's drafts read only request r's rows);
//   (3) DflashProposeBlock == SampleDflashBlockDrafts(ForwardBlockLogitsWithContext)
//       — the brick composes the D3 context-aware forward + the greedy sampler
//       exactly (no re-derivation drift);
//   (4) at EMPTY context the brick degenerates to argmax over the D2 context-free
//       ForwardBlockLogits mask rows (D3's empty-context degeneration, end-to-end);
//   (5) the "dflash" method parses through ParseSpeculativeConfigJson + ResolveDflash
//       (config-select DFlash vs MTP) and the scheduler lookahead is k+1.
//
// Deterministic + RED-first; runs on CPU with synthetic draft weights (the real
// vLLM-parity numerics are the GPU-gated D3 test_qwen3_dflash_kvprep_parity + the
// D5 e2e). No checkpoint, no GPU.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vt/backend.h"
#include "vt/dtype.h"

// SPEC-DFLASH2 (#1919): the draft context store's capacity is a REQUIRED
// argument now, resolved in production from the engine's own `max_model_len`
// (`Qwen3DFlashModel::ResolveCtxStoreSizing`). These unit stores hold single-
// digit context rows, so they name a small capacity directly; before #1919 each
// of them silently allocated the fixed 4096-slot pool.
constexpr int64_t kUnitCtxSlots = 256;

using namespace vllm;
using namespace vllm::v1;

namespace {
vt::Queue Cpu() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// Small deterministic bf16 weight: value(i) = amp * sin(seed + 0.7*i). Mirrors the
// D2 test_qwen3_dflash_forward synthetic-weight builder so the two share numerics.
OwnedTensor MkBf16(const std::vector<int64_t>& shape, double seed, double amp, bool nk) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  t.nk = nk;
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (int64_t i = 0; i < n; ++i)
    p[i] = vt::F32ToBF16(static_cast<float>(amp * std::sin(seed + 0.7 * static_cast<double>(i))));
  return t;
}

struct Dims {
  int64_t H = 4, Hq = 2, Hkv = 1, Dh = 2, I = 6, vocab = 8, layers = 2, taps = 2;
};

HfConfig MakeConfig(const Dims& dm) {
  HfConfig c;
  c.hidden_size = dm.H;
  c.num_attention_heads = dm.Hq;
  c.num_key_value_heads = dm.Hkv;
  c.head_dim = dm.Dh;
  c.rotary_dim = dm.Dh;
  c.rope_theta = 10000.0;
  c.intermediate_size = dm.I;
  c.vocab_size = dm.vocab;
  c.num_hidden_layers = dm.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 64;
  c.layer_types = {"sliding_attention", "full_attention"};
  c.raw = nlohmann::json::object();
  c.raw["dflash_config"] = {{"mask_token_id", 7}};
  return c;
}

Qwen3DFlashWeights MakeWeights(const Dims& dm) {
  Qwen3DFlashWeights w;
  w.num_taps = dm.taps;
  w.mask_token_id = 7;
  w.draft_vocab_size = dm.vocab;
  const int64_t qdim = dm.Hq * dm.Dh, kdim = dm.Hkv * dm.Dh;
  w.embed_tokens = MkBf16({dm.vocab, dm.H}, 0.1, 0.3, false);
  w.fc = MkBf16({dm.H, dm.H * dm.taps}, 0.2, 0.2, true);
  w.hidden_norm = MkBf16({dm.H}, 0.3, 0.5, false);
  w.final_norm = MkBf16({dm.H}, 0.4, 0.5, false);
  w.lm_head = MkBf16({dm.vocab, dm.H}, 0.5, 0.3, true);
  const std::vector<Qwen3DFlashLayerAttnMode> modes = {{true, 64}, {false, 0}};
  for (int64_t l = 0; l < dm.layers; ++l) {
    Qwen3DFlashLayerWeights lw;
    const double s = 1.0 + static_cast<double>(l);
    lw.input_layernorm = MkBf16({dm.H}, s + 0.01, 0.5, false);
    lw.post_attention_layernorm = MkBf16({dm.H}, s + 0.02, 0.5, false);
    lw.qkv_proj = MkBf16({qdim + 2 * kdim, dm.H}, s + 0.03, 0.25, true);
    lw.o_proj = MkBf16({dm.H, qdim}, s + 0.04, 0.25, true);
    lw.q_norm = MkBf16({dm.Dh}, s + 0.05, 0.5, false);
    lw.k_norm = MkBf16({dm.Dh}, s + 0.06, 0.5, false);
    lw.gate_up_proj = MkBf16({2 * dm.I, dm.H}, s + 0.07, 0.2, true);
    lw.down_proj = MkBf16({dm.H, dm.I}, s + 0.08, 0.2, true);
    lw.attn_mode = modes[static_cast<size_t>(l)];
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// Independent argmax over the k mask rows of a [num_reqs*(1+k), vocab] block-logits
// buffer (the reference the brick's sampler must reproduce).
std::vector<std::vector<int32_t>> RefArgmaxMask(const std::vector<float>& logits,
                                                int num_reqs, int k, int64_t vocab) {
  const int64_t block = k + 1;
  std::vector<std::vector<int32_t>> out(static_cast<size_t>(num_reqs));
  for (int r = 0; r < num_reqs; ++r) {
    for (int j = 0; j < k; ++j) {
      const int64_t row = static_cast<int64_t>(r) * block + 1 + j;
      const float* lr = logits.data() + static_cast<size_t>(row) * vocab;
      int best = 0;
      for (int64_t v = 1; v < vocab; ++v)
        if (lr[static_cast<size_t>(v)] > lr[static_cast<size_t>(best)]) best = static_cast<int>(v);
      out[static_cast<size_t>(r)].push_back(best);
    }
  }
  return out;
}
}  // namespace

TEST_CASE("dflash SampleDflashBlockDrafts: greedy argmax of the k mask rows, anchor skipped") {
  // 2 requests, k=2 => block=3, vocab=4. Hand-built logits: the argmax per row is the
  // diagonal token so the expected drafts are exact + the anchor rows are decoys.
  const int num_reqs = 2, k = 2;
  const int64_t vocab = 4;
  // Row layout per request: [anchor, mask0, mask1]. Put a giant value at a DIFFERENT
  // column in the anchor row than any mask row so a leak would be caught.
  std::vector<float> logits(static_cast<size_t>(num_reqs * (k + 1)) * vocab, 0.0f);
  auto set = [&](int row, int col, float v) {
    logits[static_cast<size_t>(row) * vocab + col] = v;
  };
  // req0: anchor->col3 (decoy), mask0->col1, mask1->col2
  set(0, 3, 99.0f);
  set(1, 1, 5.0f);
  set(2, 2, 5.0f);
  // req1: anchor->col0 (decoy), mask0->col3, mask1->col0
  set(3, 0, 99.0f);
  set(4, 3, 5.0f);
  set(5, 0, 5.0f);
  const std::vector<std::vector<int32_t>> d =
      SampleDflashBlockDrafts(logits, num_reqs, k, vocab);
  REQUIRE(d.size() == 2);
  CHECK(d[0] == std::vector<int32_t>{1, 2});
  CHECK(d[1] == std::vector<int32_t>{3, 0});  // anchor decoys (col3, col0) NOT leaked
}

TEST_CASE("dflash SampleDflashBlockDrafts RED: anchor row argmax must not leak into a draft") {
  // If the sampler wrongly read the anchor row (offset 0) it would return col3 for
  // req0's first draft instead of col1. Assert it does NOT.
  const int num_reqs = 1, k = 1;
  const int64_t vocab = 4;
  std::vector<float> logits(static_cast<size_t>(num_reqs * (k + 1)) * vocab, 0.0f);
  logits[0 * vocab + 3] = 99.0f;  // anchor -> col3
  logits[1 * vocab + 1] = 5.0f;   // mask0  -> col1
  const std::vector<std::vector<int32_t>> d =
      SampleDflashBlockDrafts(logits, num_reqs, k, vocab);
  CHECK(d[0] == std::vector<int32_t>{1});
  CHECK(d[0][0] != 3);  // the anchor argmax is excluded
}

TEST_CASE("dflash DflashProposeBlock == sampler(ForwardBlockLogitsWithContext) composition") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int num_reqs = 2, k = 2;
  // Two (1+k) blocks: anchor(2) + 2 mask(7).
  std::vector<int32_t> ids = {2, 7, 7, 3, 7, 7};
  std::vector<int32_t> pos = {5, 6, 7, 4, 5, 6};
  std::vector<int32_t> block_cu = {0, 3, 6};
  // A small per-request context (2 + 1 tokens) of combined features [num_ctx, H].
  const int64_t H = dm.H;
  std::vector<int32_t> ctx_cu = {0, 2, 3};
  std::vector<int32_t> ctx_pos = {0, 1, 0};
  std::vector<float> ctx(static_cast<size_t>(3) * H);
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * std::sin(0.13 * static_cast<double>(i) + 0.4);

  // Independent reference: the raw context-aware block logits, then argmax the mask rows.
  const std::vector<float> ref_logits = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, block_cu, w, cfg, q);
  const std::vector<std::vector<int32_t>> ref =
      RefArgmaxMask(ref_logits, num_reqs, k, dm.vocab);

  const DflashProposeResult res =
      DflashProposeBlock(w, cfg, ctx, ctx_pos, ctx_cu, ids, pos, block_cu, num_reqs, k, q);
  REQUIRE(res.draft_token_ids.size() == 2);
  CHECK(res.draft_token_ids[0].size() == static_cast<size_t>(k));
  CHECK(res.draft_token_ids == ref);  // brick composes forward + sampler exactly
}

TEST_CASE("dflash D9 persistent-KV: incremental append == full recompute, BIT-IDENTICAL (c1)") {
  // The D9 persistent paged draft-KV must be bit-for-bit identical to the D5/D7 full
  // per-step recompute. Build a 3-token context in TWO appends (pos 0,1 then pos 2) via
  // AppendContextKVHost + ForwardBlockLogitsWithPrecomputedKV, and compare to the single
  // ForwardBlockLogitsWithContext recompute over the whole context. Per-row projection
  // independence => IDENTICAL bf16 K/V => IDENTICAL f32 logits (exact equality).
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {3, 4, 5};
  std::vector<int32_t> block_cu = {0, 3};
  std::vector<int32_t> ctx_cu = {0, 3};
  std::vector<int32_t> ctx_pos = {0, 1, 2};
  std::vector<float> ctx(static_cast<size_t>(3) * H);
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * std::sin(0.13 * static_cast<double>(i) + 0.4);

  const std::vector<float> recompute = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, block_cu, w, cfg, q);

  // Persistent: two appends (rows 0-1 @ pos {0,1}, then row 2 @ pos {2}).
  Qwen3DFlashModel::PrecomputedContextKV store;
  std::vector<float> f01(ctx.begin(), ctx.begin() + 2 * H);
  std::vector<float> f2(ctx.begin() + 2 * H, ctx.end());
  Qwen3DFlashModel::AppendContextKVHost(store, f01, {0, 1}, w, cfg, q);
  Qwen3DFlashModel::AppendContextKVHost(store, f2, {2}, w, cfg, q);
  REQUIRE(store.num_ctx == 3);
  const std::vector<float> persistent = Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
      store, ctx_cu, ids, pos, block_cu, w, cfg, q);

  REQUIRE(persistent.size() == recompute.size());
  bool bit_identical = true;
  for (size_t i = 0; i < persistent.size(); ++i)
    if (persistent[i] != recompute[i]) { bit_identical = false; break; }
  CHECK(bit_identical);  // exact equality: persistent paged KV is not an approximation
}

TEST_CASE("dflash D9 persistent-KV: multi-request concat == full recompute, BIT-IDENTICAL") {
  // Two requests (ctx 2 + 1), stores concatenated in ctx_cu order == the recompute over
  // the [req0; req1] combined context. Exercises the runner's per-request store concat.
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;
  const int num_reqs = 2, k = 2;
  std::vector<int32_t> ids = {2, 7, 7, 3, 7, 7};
  std::vector<int32_t> pos = {5, 6, 7, 4, 5, 6};
  std::vector<int32_t> block_cu = {0, 3, 6};
  std::vector<int32_t> ctx_cu = {0, 2, 3};
  std::vector<int32_t> ctx_pos = {0, 1, 0};
  std::vector<float> ctx(static_cast<size_t>(3) * H);
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * std::sin(0.13 * static_cast<double>(i) + 0.4);

  const std::vector<float> recompute = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, block_cu, w, cfg, q);

  Qwen3DFlashModel::PrecomputedContextKV s0, s1;
  std::vector<float> f0(ctx.begin(), ctx.begin() + 2 * H);  // req0 rows @ pos {0,1}
  std::vector<float> f1(ctx.begin() + 2 * H, ctx.end());     // req1 row  @ pos {0}
  Qwen3DFlashModel::AppendContextKVHost(s0, f0, {0, 1}, w, cfg, q);
  Qwen3DFlashModel::AppendContextKVHost(s1, f1, {0}, w, cfg, q);
  Qwen3DFlashModel::PrecomputedContextKV comb;
  comb.k.assign(static_cast<size_t>(dm.layers), {});
  comb.v.assign(static_cast<size_t>(dm.layers), {});
  for (int64_t l = 0; l < dm.layers; ++l) {
    comb.k[static_cast<size_t>(l)].insert(comb.k[static_cast<size_t>(l)].end(),
                                          s0.k[static_cast<size_t>(l)].begin(),
                                          s0.k[static_cast<size_t>(l)].end());
    comb.k[static_cast<size_t>(l)].insert(comb.k[static_cast<size_t>(l)].end(),
                                          s1.k[static_cast<size_t>(l)].begin(),
                                          s1.k[static_cast<size_t>(l)].end());
    comb.v[static_cast<size_t>(l)].insert(comb.v[static_cast<size_t>(l)].end(),
                                          s0.v[static_cast<size_t>(l)].begin(),
                                          s0.v[static_cast<size_t>(l)].end());
    comb.v[static_cast<size_t>(l)].insert(comb.v[static_cast<size_t>(l)].end(),
                                          s1.v[static_cast<size_t>(l)].begin(),
                                          s1.v[static_cast<size_t>(l)].end());
  }
  comb.num_ctx = 3;
  const std::vector<float> persistent = Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
      comb, ctx_cu, ids, pos, block_cu, w, cfg, q);

  REQUIRE(persistent.size() == recompute.size());
  bool bit_identical = true;
  for (size_t i = 0; i < persistent.size(); ++i)
    if (persistent[i] != recompute[i]) { bit_identical = false; break; }
  CHECK(bit_identical);
  (void)num_reqs; (void)k;
}

TEST_CASE("dflash D11 device-KV: append+device forward == full recompute, BIT-IDENTICAL (c1)") {
  // Part A: the DEVICE-RESIDENT append-only store (no host round-trip) must be bit-for-bit
  // identical to the full per-step recompute (and therefore to the D9 host store). Same
  // 3-token / two-append shape as the D9 c1 case, but via MakeDeviceKVStore +
  // AppendContextKVDevice + ForwardBlockLogitsWithDeviceKV. Per-row projection independence
  // + IndexCopy concat in ascending-position order => IDENTICAL bf16 K/V => IDENTICAL logits.
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;
  std::vector<int32_t> ids = {2, 7, 7};
  std::vector<int32_t> pos = {3, 4, 5};
  std::vector<int32_t> block_cu = {0, 3};
  std::vector<int32_t> ctx_cu = {0, 3};
  std::vector<int32_t> ctx_pos = {0, 1, 2};
  std::vector<float> ctx(static_cast<size_t>(3) * H);
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * std::sin(0.13 * static_cast<double>(i) + 0.4);

  const std::vector<float> recompute = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, block_cu, w, cfg, q);

  auto store = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q, kUnitCtxSlots);
  std::vector<float> f01(ctx.begin(), ctx.begin() + 2 * H);
  std::vector<float> f2(ctx.begin() + 2 * H, ctx.end());
  Qwen3DFlashModel::AppendContextKVDevice(*store, f01, {0, 1}, w, cfg, q);
  Qwen3DFlashModel::AppendContextKVDevice(*store, f2, {2}, w, cfg, q);
  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store) == 3);
  std::vector<DflashDeviceKVStore*> stores = {store.get()};
  const std::vector<float> device_kv = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, block_cu, w, cfg, q);

  REQUIRE(device_kv.size() == recompute.size());
  bool bit_identical = true;
  for (size_t i = 0; i < device_kv.size(); ++i)
    if (device_kv[i] != recompute[i]) { bit_identical = false; break; }
  CHECK(bit_identical);  // exact equality: device-resident store is not an approximation
}

TEST_CASE("dflash D11 device-KV: multi-request device forward == full recompute, BIT-IDENTICAL") {
  // Part A multi-request: two device stores (ctx 2 + 1), passed in ctx_cu order to
  // ForwardBlockLogitsWithDeviceKV, must equal the recompute over the [req0; req1] combined
  // context. Exercises the on-device per-request concat (the runner's propose-batch path).
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int64_t H = dm.H;
  std::vector<int32_t> ids = {2, 7, 7, 3, 7, 7};
  std::vector<int32_t> pos = {5, 6, 7, 4, 5, 6};
  std::vector<int32_t> block_cu = {0, 3, 6};
  std::vector<int32_t> ctx_cu = {0, 2, 3};
  std::vector<int32_t> ctx_pos = {0, 1, 0};
  std::vector<float> ctx(static_cast<size_t>(3) * H);
  for (size_t i = 0; i < ctx.size(); ++i)
    ctx[i] = 0.2f * std::sin(0.13 * static_cast<double>(i) + 0.4);

  const std::vector<float> recompute = Qwen3DFlashModel::ForwardBlockLogitsWithContext(
      ctx, ctx_pos, ctx_cu, ids, pos, block_cu, w, cfg, q);

  auto s0 = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q, kUnitCtxSlots);
  auto s1 = Qwen3DFlashModel::MakeDeviceKVStore(cfg, q, kUnitCtxSlots);
  std::vector<float> f0(ctx.begin(), ctx.begin() + 2 * H);  // req0 rows @ pos {0,1}
  std::vector<float> f1(ctx.begin() + 2 * H, ctx.end());     // req1 row  @ pos {0}
  Qwen3DFlashModel::AppendContextKVDevice(*s0, f0, {0, 1}, w, cfg, q);
  Qwen3DFlashModel::AppendContextKVDevice(*s1, f1, {0}, w, cfg, q);
  std::vector<DflashDeviceKVStore*> stores = {s0.get(), s1.get()};
  const std::vector<float> device_kv = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      stores, ctx_cu, ids, pos, block_cu, w, cfg, q);

  REQUIRE(device_kv.size() == recompute.size());
  bool bit_identical = true;
  for (size_t i = 0; i < device_kv.size(); ++i)
    if (device_kv[i] != recompute[i]) { bit_identical = false; break; }
  CHECK(bit_identical);
}

TEST_CASE("dflash DflashProposeBlock: empty context degenerates to context-free ForwardBlockLogits") {
  Dims dm;
  HfConfig cfg = MakeConfig(dm);
  Qwen3DFlashWeights w = MakeWeights(dm);
  vt::Queue q = Cpu();
  const int num_reqs = 2, k = 2;
  std::vector<int32_t> ids = {2, 7, 7, 3, 7, 7};
  std::vector<int32_t> pos = {0, 1, 2, 0, 1, 2};
  std::vector<int32_t> block_cu = {0, 3, 6};
  // Empty context: ctx_cu all-zero boundaries, no context states/positions.
  std::vector<int32_t> ctx_cu = {0, 0, 0};
  std::vector<int32_t> ctx_pos;
  std::vector<float> ctx;

  const std::vector<float> ctxfree_logits =
      Qwen3DFlashModel::ForwardBlockLogits(ids, pos, block_cu, w, cfg, q);
  const std::vector<std::vector<int32_t>> ref =
      RefArgmaxMask(ctxfree_logits, num_reqs, k, dm.vocab);

  const DflashProposeResult res =
      DflashProposeBlock(w, cfg, ctx, ctx_pos, ctx_cu, ids, pos, block_cu, num_reqs, k, q);
  CHECK(res.draft_token_ids == ref);  // D3 empty-context == D2 context-free, end-to-end
}

TEST_CASE("dflash config: ParseSpeculativeConfigJson accepts method dflash + k, lookahead k+1") {
  // SPEC-DFLASH D5 (speculative.cpp:83-90): the DFlash draft is a SEPARATE
  // checkpoint, so `method:"dflash"` now REQUIRES a `model` key naming it (unlike
  // MTP's in-target tensors). The parser only records the string — it does NOT
  // open the path — so this stays a CPU-only, checkpoint-free config test.
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
      R"({"method":"dflash","num_speculative_tokens":15,"model":"z-lab/dflash-draft"})");
  CHECK(cfg.method == "dflash");
  REQUIRE(cfg.draft_model_path.has_value());
  CHECK(*cfg.draft_model_path == "z-lab/dflash-draft");
  CHECK(cfg.use_dflash());
  CHECK(cfg.use_eagle());  // dflash is a draft-hidden-state method
  REQUIRE(cfg.num_speculative_tokens.has_value());
  CHECK(*cfg.num_speculative_tokens == 15);
  // NumLookaheadTokens = k + 1 for dflash (the extra in-fill last-sampled slot).
  CHECK(cfg.NumLookaheadTokens() == 16);

  const SpeculativeConfig rd = SpeculativeConfig::ResolveDflash(15);
  CHECK(rd.method == "dflash");
  CHECK(rd.ResolvedNumSpeculativeTokens() == 15);
  CHECK(rd.NumLookaheadTokens() == 16);

  // A still-unsupported method throws (dspark stays out of scope).
  CHECK_THROWS_AS(ParseSpeculativeConfigJson(R"({"method":"dspark"})"),
                  std::invalid_argument);
}
