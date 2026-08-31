// GLM-5.3's heterogeneous indexer schedule, the `skip_topk` selection reuse, and
// the fp32 router gate GEMM. W4 of `.agents/specs/glm-dsa-latest-deepseek.md`
// §3.7, issue [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHAT IS UNDER GATE, and the upstream line each claim is measured against ─
//   deepseek_v2.py:1092-1103  the derived `_skip_topk` rule (W2 gates this one)
//   deepseek_v2.py:1115       the indexer is CONSTRUCTED only on a full layer
//   deepseek_v2.py:1134-1135  `self.indexer_rope_emb = None; self.indexer = None`
//   deepseek_v2.py:1175       `skip_topk=_skip_topk and not is_mtp_layer`
//   deepseek_v2.py:1372-1377  ONE `topk_indices_buffer` per model
//   deepseek_v2.py:1395       handed to EVERY layer
//   mla.py:120                and on to `MLAAttention` regardless of the layer
//   mla.py:180                `if self.indexer and self.is_sparse and not self.skip_topk:`
//   sparse_mla_attention.py:303-310  "the explicitly-passed buffer covers
//                             backbone skip layers, whose indexer is not
//                             constructed"
//   deepseek_v2.py:123-133    `_get_moe_router_dtype`, the gate's out_dtype
//   deepseek_v2.py:309-314    `GateLinear(..., out_dtype=self.router_dtype)`
//
// ─── THE THING THIS FILE EXISTS TO PREVENT ───────────────────────────────────
// A shared layer that quietly attends DENSELY. It would produce a finite,
// plausible, wrong output on 57 of 79 blocks, and no token gate could see it,
// because the tokens would still be tokens. So the reuse is not asserted by
// reading the flag: `G1` proves the shared layer's output CHANGES when the
// selection it inherits changes, and the tautology guard proves a full layer and
// a shared layer are not computing the same thing in the first place. Either one
// alone passes trivially.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "glm_moe_dsa_config_glm53.inc"
#include "router_dtype_golden.inc"
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/model_executor/models/glm_moe_dsa.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::dense_attn::Dev;
using vllm::mla::AbsorbedKvBProj;
using vllm::mla::AbsorbKvBProjBf16;
using vllm::mla::BuildDeepseekRopeCosSinCache;
using vllm::mla::DeepseekYarnRopeParams;
using vllm::mla::ForwardMlaAttentionBlock;
using vllm::mla::MlaBlockDims;
using vllm::mla::MlaBlockMetadata;
using vllm::mla::MlaBlockWeights;
using vllm::mla::MlaSharedSelection;
using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

nlohmann::json Glm53Doc() {
  return nlohmann::json::parse(glm_moe_dsa_fixture::kGlm53ConfigJson);
}
vllm::HfConfig Glm53Config() {
  return vllm::ParseHfConfig(Glm53Doc(), "zai-org/GLM-5.3 config.json");
}
vllm::HfConfig ConfigFromDoc(const nlohmann::json& doc) {
  return vllm::ParseHfConfig(doc, "test config");
}

// ─── a small MLA block harness with an INDEXER ──────────────────────────────
// The block test next door (`test_mla_attention_block.cpp`) has no indexer
// weights at all, so nothing in this tree drives `ForwardMlaAttentionBlock` on a
// SPARSE step outside the dots3-note model forward. This harness is the smallest
// thing that does.
constexpr int64_t kBlockSize = 16;

std::vector<float> Rand(size_t n, uint32_t seed, float amp) {
  std::vector<float> v(n);
  uint32_t s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    // Round through bf16 so the value the device holds IS the value here.
    v[i] = vt::BF16ToF32(vt::F32ToBF16(
        (static_cast<float>(s % 20001u) / 10000.0f - 1.0f) * amp));
  }
  return v;
}

// The geometry: GLM-shaped (q-LoRA present, v_head_dim < qk_head_dim, an indexer
// head narrower than the main head) at a size a CPU test can run.
MlaBlockDims BaseDims() {
  MlaBlockDims d{};
  d.hidden_size = 32;
  d.num_heads = 2;
  d.q_lora_rank = 16;
  d.kv_lora_rank = 16;
  d.qk_nope_head_dim = 8;
  d.qk_rope_head_dim = 4;
  d.v_head_dim = 8;
  d.scale = 0.25f;
  return d;
}

constexpr int64_t kIdxHeads = 2;
constexpr int64_t kIdxHeadDim = 8;
// STRICTLY smaller than the token count below, or the top-k selects every causal
// candidate and every selection in this file is the same one — the shape in
// which G1 and its mutation both pass while measuring nothing.
constexpr int64_t kIndexTopk = 3;
constexpr int64_t kTokens = 6;

MlaBlockDims FullDims() {
  MlaBlockDims d = BaseDims();
  d.index_n_heads = kIdxHeads;
  d.index_head_dim = kIdxHeadDim;
  d.index_topk = kIndexTopk;
  return d;
}
MlaBlockDims SharedDims() {
  MlaBlockDims d = BaseDims();
  d.skip_topk = true;
  return d;
}

class Harness {
 public:
  Harness(Backend& b, Queue& q, const MlaBlockDims& d, uint32_t seed)
      : b_(b), q_(q), dev_{b, q} {
    const int64_t H = d.hidden_size, N = d.num_heads, L = d.kv_lora_rank;
    const int64_t P = d.qk_nope_head_dim, R = d.qk_rope_head_dim, V = d.v_head_dim;
    const int64_t Dqk = d.qk_head_dim(), ql = d.q_lora_rank;
    w_.fused_qkv_a_proj = Up(Rand(static_cast<size_t>((ql + L + R) * H), seed, 0.1f), {ql + L + R, H});
    w_.q_a_layernorm = Up(Rand(static_cast<size_t>(ql), seed + 1, 0.5f), {ql});
    w_.q_b_proj = Up(Rand(static_cast<size_t>(N * Dqk * ql), seed + 2, 0.1f), {N * Dqk, ql});
    w_.kv_a_layernorm = Up(Rand(static_cast<size_t>(L), seed + 3, 0.5f), {L});
    std::vector<float> kv_b = Rand(static_cast<size_t>(N * (P + V) * L), seed + 4, 0.1f);
    w_.kv_b_proj = Up(kv_b, {N * (P + V), L});
    std::vector<uint16_t> kv_b_bf(kv_b.size());
    for (size_t i = 0; i < kv_b.size(); ++i) kv_b_bf[i] = vt::F32ToBF16(kv_b[i]);
    AbsorbedKvBProj ab = AbsorbKvBProjBf16(kv_b_bf.data(), d);
    std::vector<float> uk(ab.w_uk_t.size()), uv(ab.w_uv.size());
    for (size_t i = 0; i < uk.size(); ++i) uk[i] = vt::BF16ToF32(ab.w_uk_t[i]);
    for (size_t i = 0; i < uv.size(); ++i) uv[i] = vt::BF16ToF32(ab.w_uv[i]);
    w_.w_uk_t = Up(uk, {N, P, L});
    w_.w_uv = Up(uv, {N, L, V});
    w_.o_proj = Up(Rand(static_cast<size_t>(H * N * V), seed + 5, 0.1f), {H, N * V});
    DeepseekYarnRopeParams rp{};
    rp.rotary_dim = R;
    rp.yarn = false;
    std::vector<float> cs = BuildDeepseekRopeCosSinCache(rp, 128);
    w_.rope_cos_sin_cache = Up(cs, {static_cast<int64_t>(cs.size()) / R, R});
    kv_cache_ = Alloc(DType::kF32, {8, kBlockSize, L + R});
  }

  // The five indexer tensors. `seed` selects WHICH indexer, which is how two
  // full layers in this file produce two DIFFERENT selections from one input.
  void AddIndexer(const MlaBlockDims& d, uint32_t seed) {
    const int64_t H = d.hidden_size, ql = d.q_lora_rank;
    const int64_t IH = d.index_n_heads, ID = d.index_head_dim;
    w_.indexer_wq_b = Up(Rand(static_cast<size_t>(IH * ID * ql), seed, 0.4f), {IH * ID, ql});
    w_.indexer_wk = Up(Rand(static_cast<size_t>(ID * H), seed + 1, 0.4f), {ID, H});
    w_.indexer_weights_proj = Up(Rand(static_cast<size_t>(IH * H), seed + 2, 0.4f), {IH, H});
    w_.indexer_k_norm_weight = Up(Rand(static_cast<size_t>(ID), seed + 3, 0.5f), {ID});
    w_.indexer_k_norm_bias = Up(Rand(static_cast<size_t>(ID), seed + 4, 0.2f), {ID});
  }

  Dev dev() { return dev_; }
  MlaBlockWeights& weights() { return w_; }
  Tensor& kv_cache() { return kv_cache_; }

  Tensor Up(const std::vector<float>& v, const std::vector<int64_t>& shape) {
    Tensor t = Alloc(DType::kF32, shape);
    b_.Copy(q_, t.data, v.data(), v.size() * sizeof(float));
    return t;
  }
  Tensor UpI32(const std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
    Tensor t = Alloc(DType::kI32, shape);
    b_.Copy(q_, t.data, v.data(), v.size() * sizeof(int32_t));
    return t;
  }
  Tensor UpI64(const std::vector<int64_t>& v, const std::vector<int64_t>& shape) {
    Tensor t = Alloc(DType::kI64, shape);
    b_.Copy(q_, t.data, v.data(), v.size() * sizeof(int64_t));
    return t;
  }
  Tensor Alloc(DType dt, const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    const size_t bytes = static_cast<size_t>(numel) * vt::SizeOf(dt);
    void* p = b_.Alloc(bytes == 0 ? 1 : bytes);
    owned_.push_back(p);
    Tensor t;
    t.data = p;
    t.dtype = dt;
    t.device = q_.device;
    t.rank = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (int i = t.rank - 1; i >= 0; --i) {
      t.shape[i] = shape[static_cast<size_t>(i)];
      t.stride[i] = stride;
      stride *= shape[static_cast<size_t>(i)];
    }
    return t;
  }
  std::vector<float> DownF32(const Tensor& t) {
    std::vector<float> o(static_cast<size_t>(t.Numel()));
    b_.Copy(q_, o.data(), t.data, o.size() * sizeof(float));
    b_.Synchronize(q_);
    return o;
  }
  std::vector<int32_t> DownI32(const Tensor& t) {
    std::vector<int32_t> o(static_cast<size_t>(t.Numel()));
    b_.Copy(q_, o.data(), t.data, o.size() * sizeof(int32_t));
    b_.Synchronize(q_);
    return o;
  }
  ~Harness() {
    for (void* p : owned_) b_.Free(p);
  }
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

 private:
  Backend& b_;
  Queue& q_;
  Dev dev_;
  MlaBlockWeights w_{};
  Tensor kv_cache_{};
  std::vector<void*> owned_;
};

// One fresh request of `kTokens` tokens, every token routed through MQA — the
// shape `BuildDots3NoteSparseStep` builds (`dots3_note_device.cpp:455-497`) and
// the one `indexer_cu_seqlens_q` being non-empty declares.
MlaBlockMetadata SparseMeta(Harness& h) {
  const int64_t cols = (kTokens + kBlockSize - 1) / kBlockSize;
  std::vector<int32_t> bt(static_cast<size_t>(kTokens * cols), 0);
  std::vector<int32_t> sl(static_cast<size_t>(kTokens));
  for (int64_t t = 0; t < kTokens; ++t) {
    sl[static_cast<size_t>(t)] = static_cast<int32_t>(t + 1);
    for (int64_t c = 0; c < cols; ++c) {
      bt[static_cast<size_t>(t * cols + c)] = static_cast<int32_t>(c);
    }
  }
  MlaBlockMetadata m{};
  m.num_decode_tokens = kTokens;
  m.decode.block_table = h.UpI32(bt, {kTokens, cols});
  m.decode.seq_lens = h.UpI32(sl, {kTokens});
  m.decode.max_seq_len = static_cast<int>(kTokens);
  m.indexer_cu_seqlens_q = {0, static_cast<int32_t>(kTokens)};
  return m;
}

// Every number a float assertion in this file touches must be finite. An
// all-NaN forward compares "equal" to nothing and "different" from everything,
// so both G1 and its mutation would report success on a block that produced no
// numbers at all.
void RequireFinite(const std::vector<float>& v, const char* what) {
  bool ok = true;
  for (float x : v) ok = ok && std::isfinite(x);
  INFO("non-finite values in ", what);
  REQUIRE(ok);
  REQUIRE(!v.empty());
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

// Run one layer. `shared` is upstream's per-model buffer; `dims` decides whether
// this call WRITES it (a full layer) or READS it (a shared layer).
std::vector<float> RunLayer(Backend& b, Queue& q, Harness& h, const MlaBlockDims& dims,
                            MlaSharedSelection* shared) {
  const int64_t H = dims.hidden_size;
  std::vector<float> hidden = Rand(static_cast<size_t>(kTokens * H), 7u, 0.6f);
  std::vector<int32_t> pos(static_cast<size_t>(kTokens));
  std::vector<int64_t> slots(static_cast<size_t>(kTokens));
  for (int64_t t = 0; t < kTokens; ++t) {
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    slots[static_cast<size_t>(t)] = t;
  }
  Tensor th = h.Up(hidden, {kTokens, H});
  Tensor tp = h.UpI32(pos, {kTokens});
  Tensor ts = h.UpI64(slots, {kTokens});
  Tensor out = h.Alloc(DType::kF32, {kTokens, H});
  MlaBlockMetadata meta = SparseMeta(h);
  vllm::v1::TritonMLAImpl impl;
  Dev dev = h.dev();
  Tensor kvc = h.kv_cache();
  // The ELEVENTH argument is `attn_pre_o_proj`, which `KV-DSV4-MULTICACHE` W5
  // added between `out` and the shared selection this case exists to drive
  // (#2323). Naming both keeps the two from swapping again silently: passing the
  // selection positionally is what broke this file's compile on `origin/main`,
  // and the product-code half of the same slip was repaired by `11f34effb`.
  ForwardMlaAttentionBlock(dev, dims, h.weights(), th, tp, kvc, ts, meta, impl,
                           out, /*attn_pre_o_proj=*/nullptr, shared);
  b.Synchronize(q);
  return h.DownF32(out);
}

MlaSharedSelection MakeShared(Harness& h) {
  MlaSharedSelection s;
  s.topk_indices = h.Alloc(DType::kI32, {kTokens, kIndexTopk});
  s.valid_counts = h.Alloc(DType::kI32, {kTokens});
  return s;
}

}  // namespace

// ════════════════════════════════════════════════════════════════════════════
// 1. THE SCHEDULE, FROM THE PARSED CONFIG
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("GLM-5.3's MLA schedule splits the backbone 21 full / 57 shared, from the PARSED config") {
  // NOTHING in this case names a layer index or a count that was typed by hand
  // from the checkpoint. The schedule is read off `GlmMoeDsaParams`, which W2
  // resolved from the committed `config.json` at revision
  // `935644c05e76fc198714f4cca449fd8b970ff6d7`. Re-deriving the rule here would
  // gate this file against itself.
  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  const std::vector<MlaBlockDims> sched = vllm::GlmMoeDsaMlaSchedule(p);

  REQUIRE(static_cast<int64_t>(sched.size()) == p.num_hidden_layers);
  CHECK(sched.size() == 78u);

  int64_t full = 0, shared = 0;
  for (const MlaBlockDims& d : sched) {
    if (d.has_indexer()) {
      ++full;
      CHECK_FALSE(d.skip_topk);
      // The geometry a full layer carries is the config's, not a default.
      CHECK(d.index_n_heads == p.index_n_heads);
      CHECK(d.index_head_dim == p.index_head_dim);
      CHECK(d.index_topk == p.index_topk);
    } else {
      ++shared;
      CHECK(d.skip_topk);
      // CLEARED, not merely unused: a surviving `index_topk` would make the
      // block run an indexer over weights this block does not ship.
      CHECK(d.index_n_heads == 0);
      CHECK(d.index_head_dim == 0);
      CHECK(d.index_topk == 0);
    }
    // Every layer is sparse either way — that is the whole point of `skip_topk`.
    CHECK(d.is_sparse());
  }
  CHECK(full == 21);
  CHECK(shared == 57);
  CHECK(full + shared == 78);
  CHECK(vllm::GlmMoeDsaFullIndexerLayerCount(p) == full);

  // THE 22 OF 79 THE SPEC QUOTES, reassembled from its two halves so a reader
  // meets the arithmetic rather than a number. 78 backbone blocks carry 21
  // indexers; block 78 is the MTP block, which upstream forces FULL at
  // `deepseek_v2.py:1110-1115` regardless of the schedule, and which this row
  // skips through `allow_mtp_tail` (spec O5). 21 + 1 = 22 indexer-bearing
  // blocks of 79, and 57 without.
  CHECK(p.num_nextn_predict_layers == 1);
  CHECK(full + p.num_nextn_predict_layers == 22);
  CHECK(p.num_hidden_layers + p.num_nextn_predict_layers == 79);
  CHECK(shared == 79 - 22);
}

TEST_CASE("Layer 0 is FULL, so the very first shared layer always has something to inherit") {
  // Not a restatement of the schedule: it is the ORDERING precondition the
  // reuse depends on. `mla.py:180` reads whatever is in the shared buffer, and
  // at layer 0 that is uninitialized memory.
  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  const std::vector<MlaBlockDims> sched = vllm::GlmMoeDsaMlaSchedule(p);
  REQUIRE(!sched.empty());
  CHECK(sched.front().has_indexer());
  CHECK_FALSE(sched.front().skip_topk);

  // And a schedule that violated it is REFUSED rather than read past.
  vllm::GlmMoeDsaParams bad = p;
  bad.indexer_types[0] = vllm::GlmMoeDsaIndexerKind::kShared;
  CHECK_THROWS_WITH_AS((void)vllm::GlmMoeDsaMlaSchedule(bad),
                       doctest::Contains("layer 0 is `shared`"), std::runtime_error);
}

TEST_CASE("GlmMoeDsaMlaBlockDims refuses a layer outside the schedule") {
  const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(Glm53Config());
  CHECK_THROWS_AS((void)vllm::GlmMoeDsaMlaBlockDims(p, -1), std::out_of_range);
  CHECK_THROWS_AS((void)vllm::GlmMoeDsaMlaBlockDims(p, p.num_hidden_layers),
                  std::out_of_range);
}

// ── REACHABILITY ────────────────────────────────────────────────────────────
TEST_CASE("REACHABILITY: the registry's config hook RUNS the per-layer schedule") {
  // `ParseGlmMoeDsaConfig` is what `ModelRegistry::Resolve` calls, and W2's own
  // suite proves that hook is reached from `LoadedEngine::FromModelDir`. This
  // case proves the SCHEDULE is inside it: a config whose per-layer MLA geometry
  // `MlaBlockDims::Validate` refuses must be refused AT RESOLVE.
  //
  // THE MUTATION THIS CASE EXISTS FOR: delete `(void)GlmMoeDsaMlaSchedule(p);`
  // from `ParseGlmMoeDsaConfig` and this case goes red, because the config below
  // parses cleanly as fields and only fails as GEOMETRY.
  nlohmann::json doc = Glm53Doc();
  // `v_head_dim > qk_head_dim` has no upstream form (the MLA block refuses it),
  // and every field is individually well-formed, so ONLY the geometry check
  // catches it.
  doc["v_head_dim"] = 4096;
  CHECK_THROWS_AS(vllm::ParseGlmMoeDsaConfig(ConfigFromDoc(doc)), std::exception);

  // The unmodified config still resolves — a hook that threw on everything
  // would pass the assertion above while gating nothing.
  CHECK_NOTHROW(vllm::ParseGlmMoeDsaConfig(Glm53Config()));
}

TEST_CASE("MlaBlockDims refuses skip_topk together with an indexer geometry") {
  MlaBlockDims d = FullDims();
  d.skip_topk = true;
  CHECK_THROWS_WITH_AS(d.Validate(), doctest::Contains("mutually"),
                       std::invalid_argument);
  // Each alone is fine.
  CHECK_NOTHROW(FullDims().Validate());
  CHECK_NOTHROW(SharedDims().Validate());
  CHECK(FullDims().is_sparse());
  CHECK(SharedDims().is_sparse());
  CHECK_FALSE(BaseDims().is_sparse());
}

// ════════════════════════════════════════════════════════════════════════════
// 2. G1 — THE SELECTION REUSE
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("G1: a shared layer attends through the preceding full layer's selection") {
  Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Queue q = b.CreateQueue();

  // ── the instrument's own precondition ─────────────────────────────────────
  // Two DIFFERENT full layers, i.e. two different indexers over the same input,
  // must produce two DIFFERENT selections. If they do not, every assertion
  // below holds vacuously and the mutation at the end proves nothing. This is
  // asserted BEFORE anything is concluded from it.
  Harness ha(b, q, FullDims(), 100u);
  ha.AddIndexer(FullDims(), 1000u);
  Harness hb(b, q, FullDims(), 100u);
  hb.AddIndexer(FullDims(), 5000u);

  MlaSharedSelection sel_a = MakeShared(ha);
  MlaSharedSelection sel_b = MakeShared(hb);

  const std::vector<float> out_full_a = RunLayer(b, q, ha, FullDims(), &sel_a);
  const std::vector<float> out_full_b = RunLayer(b, q, hb, FullDims(), &sel_b);
  RequireFinite(out_full_a, "full layer A output");
  RequireFinite(out_full_b, "full layer B output");

  const std::vector<int32_t> idx_a = ha.DownI32(sel_a.topk_indices);
  const std::vector<int32_t> idx_b = hb.DownI32(sel_b.topk_indices);
  const std::vector<int32_t> cnt_a = ha.DownI32(sel_a.valid_counts);
  REQUIRE(idx_a.size() == static_cast<size_t>(kTokens * kIndexTopk));

  // The selection is a REAL one: every count is bounded by the token's causal
  // range and by index_topk, and at least one token actually pruned. A run in
  // which top-k picked everything is the "discrete selection gate that cannot
  // discriminate" shape, and it must not read as a pass.
  bool pruned = false;
  for (int64_t t = 0; t < kTokens; ++t) {
    const int32_t c = cnt_a[static_cast<size_t>(t)];
    REQUIRE(c >= 0);
    REQUIRE(c <= static_cast<int32_t>(kIndexTopk));
    REQUIRE(c <= static_cast<int32_t>(t + 1));
    if (t + 1 > kIndexTopk) pruned = true;
  }
  INFO("index_topk (", kIndexTopk, ") must be below the token count (", kTokens,
       ") or no token prunes and the selection cannot discriminate");
  REQUIRE(pruned);

  INFO("two different indexers produced the SAME selection; every assertion "
       "below would hold vacuously");
  REQUIRE(idx_a != idx_b);

  // ── G1 proper ────────────────────────────────────────────────────────────
  // The shared layer runs on layer A's weights and layer A's buffer.
  const std::vector<float> out_shared_a = RunLayer(b, q, ha, SharedDims(), &sel_a);
  RequireFinite(out_shared_a, "shared layer output");

  // (a) THE BUFFER IS NOT TOUCHED. Reuse in `mla.py:180` is the absence of a
  //     write, not a copy — a shared layer that rewrote the buffer would break
  //     every LATER shared layer in the same pass.
  const std::vector<int32_t> idx_after = ha.DownI32(sel_a.topk_indices);
  const std::vector<int32_t> cnt_after = ha.DownI32(sel_a.valid_counts);
  CHECK(idx_after == idx_a);
  CHECK(cnt_after == cnt_a);

  // (b) THE SHARED LAYER CONSUMED IT. Same weights, same input, same buffer —
  //     but pointed at layer B's selection instead, the output must MOVE. If it
  //     does not, the shared layer is not reading the buffer at all and is
  //     almost certainly attending densely.
  //
  //     THIS IS G1's MUTATION, and it is a mutation of the DATA rather than of
  //     the source, which is what "re-pointing it at a different layer" means.
  const std::vector<float> out_shared_b = RunLayer(b, q, ha, SharedDims(), &sel_b);
  RequireFinite(out_shared_b, "shared layer output on the foreign selection");
  const double moved = MaxAbsDiff(out_shared_a, out_shared_b);
  INFO("re-pointing the shared layer at a DIFFERENT full layer's selection did "
       "not change its output: max|diff| = ", moved);
  CHECK(moved > 1e-6);

  // (c) AND IT USED EXACTLY THAT SELECTION. Layer A's own full run and the
  //     shared run differ only in who computed the selection, so a shared layer
  //     handed a selection is not required to match the full layer's OUTPUT —
  //     the full layer also ran an indexer. What must match is the selection,
  //     and (a) already established that it is byte-identical. Recorded here so
  //     a later reader does not add a wrong equality.
  CHECK(idx_after.size() == idx_a.size());
}

TEST_CASE("The tautology guard: a full layer and a shared layer do NOT compute the same thing") {
  // Without this, G1 passes trivially on a block in which every layer produces
  // the same output regardless of its selection.
  Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Queue q = b.CreateQueue();

  Harness ha(b, q, FullDims(), 100u);
  ha.AddIndexer(FullDims(), 1000u);
  Harness hb(b, q, FullDims(), 100u);
  hb.AddIndexer(FullDims(), 5000u);

  MlaSharedSelection sel_b = MakeShared(hb);
  (void)RunLayer(b, q, hb, FullDims(), &sel_b);

  MlaSharedSelection sel_a = MakeShared(ha);
  const std::vector<float> full = RunLayer(b, q, ha, FullDims(), &sel_a);
  // The SAME weights, the SAME input, but this layer skips its indexer and
  // inherits a foreign selection instead.
  const std::vector<float> shared = RunLayer(b, q, ha, SharedDims(), &sel_b);
  RequireFinite(full, "full layer output");
  RequireFinite(shared, "shared layer output");

  const double d = MaxAbsDiff(full, shared);
  INFO("a full layer and a shared layer produced the SAME output (max|diff| = ",
       d, "), so this block's attention does not depend on its selection and "
       "G1 measures nothing");
  CHECK(d > 1e-6);
}

TEST_CASE("A shared layer with no selection to inherit is REFUSED, not served densely") {
  // Falling through to the dense contiguous key loop is the silent failure this
  // whole wave is built around: it would be finite, plausible and wrong.
  Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Queue q = b.CreateQueue();
  Harness h(b, q, FullDims(), 100u);
  h.AddIndexer(FullDims(), 1000u);

  CHECK_THROWS_WITH_AS((void)RunLayer(b, q, h, SharedDims(), nullptr),
                       doctest::Contains("nothing to reuse"), std::invalid_argument);

  // An EMPTY buffer is the same refusal, not a different one.
  MlaSharedSelection empty;
  CHECK_THROWS_AS((void)RunLayer(b, q, h, SharedDims(), &empty), std::invalid_argument);

  // A buffer of the wrong width is refused BY SHAPE rather than read.
  MlaSharedSelection narrow;
  narrow.topk_indices = h.Alloc(DType::kI32, {kTokens, kIndexTopk - 1});
  narrow.valid_counts = h.Alloc(DType::kI32, {kTokens});
  CHECK_THROWS_AS((void)RunLayer(b, q, h, FullDims(), &narrow), std::invalid_argument);
}

TEST_CASE("A full layer with NO shared buffer is byte-identical to the pre-W4 block") {
  // Every existing caller of `ForwardMlaAttentionBlock` passes no buffer. The
  // parameter is a default, and a default that changed a number would be a
  // regression in four models this wave does not touch.
  Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Queue q = b.CreateQueue();
  Harness h(b, q, FullDims(), 100u);
  h.AddIndexer(FullDims(), 1000u);

  const std::vector<float> without = RunLayer(b, q, h, FullDims(), nullptr);
  MlaSharedSelection sel = MakeShared(h);
  const std::vector<float> with = RunLayer(b, q, h, FullDims(), &sel);
  RequireFinite(without, "full layer without a shared buffer");
  RequireFinite(with, "full layer with a shared buffer");
  // The destination of the write moved; the arithmetic did not.
  CHECK(MaxAbsDiff(without, with) == 0.0);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. THE fp32 ROUTER GATE GEMM
// ════════════════════════════════════════════════════════════════════════════
TEST_CASE("The router gate's output dtype matches the pinned vLLM oracle on every case") {
  // The expectations are `_get_moe_router_dtype`'s OWN return values, executed
  // from the pinned file's bytes — see the header of `router_dtype_golden.inc`.
  using router_dtype_fixture::kNumRouterDtypeCases;
  using router_dtype_fixture::kRouterDtypeCases;
  REQUIRE(kNumRouterDtypeCases == 8);

  int checked = 0;
  int f32_cases = 0;
  for (int i = 0; i < kNumRouterDtypeCases; ++i) {
    const auto& c = kRouterDtypeCases[i];
    INFO("case ", c.name);
    if (c.upstream_returns_f32) ++f32_cases;

    // GLM-5.3's own resolve. `ParseGlmMoeDsaParams` serves `model_type ==
    // "glm_moe_dsa"` only, so it answers the three glm cases.
    if (c.model_type != nullptr && std::string(c.model_type) == "glm_moe_dsa") {
      nlohmann::json doc = Glm53Doc();
      if (c.moe_router_dtype == nullptr) {
        doc.erase("moe_router_dtype");
      } else {
        doc["moe_router_dtype"] = c.moe_router_dtype;
      }
      const vllm::GlmMoeDsaParams p = vllm::ParseGlmMoeDsaParams(ConfigFromDoc(doc));
      // `:127` fires FIRST and wins even against an explicit "bfloat16".
      CHECK(p.router_dtype_is_f32 == c.upstream_returns_f32);
      ++checked;
      continue;
    }

    // Every other case is `DeepseekV2ForCausalLM`'s generic `:131` arm, which is
    // the one this wave repaired: the tree hardcoded bf16 there.
    nlohmann::json doc = Glm53Doc();
    doc["architectures"] = nlohmann::json::array({"DeepseekV2ForCausalLM"});
    if (c.model_type == nullptr) {
      // A DIVERGENCE, asserted rather than skipped. Upstream reads the field
      // with `getattr(config, "model_type", None)` (`deepseek_v2.py:127`), so a
      // config with no `model_type` reaches `:131` and the oracle answers f32.
      // `ParseHfConfig` refuses such a config outright, one layer earlier and
      // for a reason that has nothing to do with the router — so this row of the
      // golden is UNREACHABLE here, and the honest gate is that the refusal is
      // what happens, not that the case quietly disappears. The generic `:131`
      // arm itself is covered by `deepseek_v3_router_float32` below.
      doc.erase("model_type");
      CHECK(c.upstream_returns_f32);
      CHECK_THROWS_AS((void)ConfigFromDoc(doc), std::exception);
      ++checked;
      continue;
    }
    doc["model_type"] = c.model_type;
    if (c.moe_router_dtype == nullptr) {
      doc.erase("moe_router_dtype");
    } else {
      doc["moe_router_dtype"] = c.moe_router_dtype;
    }
    // DeepSeek-V2's parser refuses this checkpoint's DSA keys by design
    // (`deepseek_v2_weights.cpp:358-364`), which is the wall W2 deliberately
    // left standing, so the router key is read off a config it will accept.
    doc.erase("index_topk");
    doc.erase("index_n_heads");
    doc.erase("index_head_dim");
    doc.erase("indexer_types");
    doc.erase("quantization_config");
    doc["num_nextn_predict_layers"] = 0;
    const vllm::DeepseekV2Params p = vllm::ParseDeepseekV2Params(ConfigFromDoc(doc));
    CHECK(p.router_dtype_is_f32 == c.upstream_returns_f32);
    ++checked;
  }
  // The table must actually DISCRIMINATE. A fixture in which every case answers
  // the same way passes against a function that returns a constant.
  CHECK(checked == kNumRouterDtypeCases);
  CHECK(f32_cases > 0);
  CHECK(f32_cases < kNumRouterDtypeCases);
}

TEST_CASE("The router gate GEMM at f32 reproduces the exact products a bf16 store loses") {
  // Shape-valid is not enough and a tolerance would hide the whole effect: the
  // reference products are exact integers for ANY reduction order, so the f32
  // arm is asserted with `==` and the bf16 arm is asserted to DIFFER.
  using namespace router_dtype_fixture;
  Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  Queue q = b.CreateQueue();

  const int64_t T = kGemmT, H = kGemmH, E = kGemmE;
  std::vector<void*> owned;
  auto alloc = [&](DType dt, int64_t n) {
    void* p = b.Alloc(static_cast<size_t>(n) * vt::SizeOf(dt));
    owned.push_back(p);
    return p;
  };
  auto mk = [&](DType dt, int64_t d0, int64_t d1) {
    Tensor t;
    t.data = alloc(dt, d0 * d1);
    t.dtype = dt;
    t.device = q.device;
    t.rank = 2;
    t.shape[0] = d0;
    t.shape[1] = d1;
    t.stride[0] = d1;
    t.stride[1] = 1;
    return t;
  };
  auto up_bf16 = [&](Tensor& t, const float* v, int64_t n) {
    std::vector<uint16_t> bf(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) bf[static_cast<size_t>(i)] = vt::F32ToBF16(v[i]);
    b.Copy(q, t.data, bf.data(), bf.size() * sizeof(uint16_t));
  };

  Tensor a = mk(DType::kBF16, T, H);
  Tensor w = mk(DType::kBF16, H, E);
  up_bf16(a, kGemmA, T * H);
  up_bf16(w, kGemmB, H * E);

  // The f32 arm — `GateLinear(..., out_dtype=torch.float32)`.
  Tensor of32 = mk(DType::kF32, T, E);
  vt::Matmul(q, of32, a, w);
  b.Synchronize(q);
  std::vector<float> got_f32(static_cast<size_t>(T * E));
  b.Copy(q, got_f32.data(), of32.data, got_f32.size() * sizeof(float));
  b.Synchronize(q);

  // The bf16 arm — what the tree did unconditionally before this wave.
  Tensor obf = mk(DType::kBF16, T, E);
  vt::Matmul(q, obf, a, w);
  b.Synchronize(q);
  std::vector<uint16_t> raw(static_cast<size_t>(T * E));
  b.Copy(q, raw.data(), obf.data, raw.size() * sizeof(uint16_t));
  b.Synchronize(q);
  std::vector<float> got_bf16(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) got_bf16[i] = vt::BF16ToF32(raw[i]);

  RequireFinite(got_f32, "f32 router logits");
  RequireFinite(got_bf16, "bf16 router logits");

  int differ = 0;
  for (int64_t i = 0; i < T * E; ++i) {
    const size_t k = static_cast<size_t>(i);
    INFO("element ", i);
    // EXACT. Not a tolerance.
    CHECK(got_f32[k] == kGemmF32Exact[k]);
    CHECK(got_bf16[k] == kGemmBf16Stored[k]);
    if (kGemmBf16Stored[k] != kGemmF32Exact[k]) ++differ;
  }
  // THE LOWER BOUND a dtype claim needs. If the two stores agreed everywhere,
  // this case could not tell a widened buffer from a relabelled one.
  INFO("the fixture does not discriminate: the bf16 store lost nothing");
  CHECK(differ == 13);
}

TEST_CASE("The parsed router dtype REACHES the DeepSeek-V2 params a forward reads") {
  // The dtype is a field on `DeepseekV2Params`, which `MoeBlock` reads at
  // `deepseek_v2.cpp:363` to size `dlog`. Deleting the read there — i.e.
  // restoring the hardcoded `DType::kBF16` — reds the GEMM case above only if
  // something drives `MoeBlock`; this case pins the PARSE half, which is what
  // decides the buffer's dtype.
  nlohmann::json doc = Glm53Doc();
  doc["architectures"] = nlohmann::json::array({"DeepseekV2ForCausalLM"});
  doc["model_type"] = "deepseek_v3";
  doc.erase("index_topk");
  doc.erase("index_n_heads");
  doc.erase("index_head_dim");
  doc.erase("indexer_types");
  doc.erase("quantization_config");
  doc["num_nextn_predict_layers"] = 0;

  doc["moe_router_dtype"] = "float32";
  CHECK(vllm::ParseDeepseekV2Params(ConfigFromDoc(doc)).router_dtype_is_f32);
  // Upstream compares against the LITERAL string at `:131`, so every other
  // value falls through to the model dtype rather than selecting one.
  doc["moe_router_dtype"] = "bfloat16";
  CHECK_FALSE(vllm::ParseDeepseekV2Params(ConfigFromDoc(doc)).router_dtype_is_f32);
  doc["moe_router_dtype"] = "float64";
  CHECK_FALSE(vllm::ParseDeepseekV2Params(ConfigFromDoc(doc)).router_dtype_is_f32);
  doc.erase("moe_router_dtype");
  CHECK_FALSE(vllm::ParseDeepseekV2Params(ConfigFromDoc(doc)).router_dtype_is_f32);
}
