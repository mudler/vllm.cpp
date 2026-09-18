// DeepSeek-V4.1-Flash W3c (indexer arm) + W3d (`kv_source_layer` topology) gate.
//
// ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB, row
// `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`.
// Oracle: vLLM `e77daef89e`, AHEAD OF our parity pin `e126687a9a`, at which
// `vllm/models/deepseek_v4_1/` does not exist. No run gate against the pinned
// oracle is claimed and none is reachable; W2 owns the pin advance. This is the
// host-reference gate V4's W3-W7 carried: small shapes, an INDEPENDENTLY
// written reference, and exactness where upstream asserts exactness.
//
// ─── WHAT IS PORTED, AND WHAT UPSTREAM HAS NO COUNTERPART FOR ────────────────
//
// PORTED. Every parameter, fixture, tolerance and failure case is preserved
// EXCEPT the four named below as ADAPTED or DROPPED. A blanket "preserved" over
// an axis that was quietly removed is the claim this list exists to prevent, so
// each removal carries its reason here and at the case itself.
//   * `tests/kernels/test_fused_indexer_q_rope_quant.py:358`
//     `test_indexer_k_inserts_only_valid_groups_into_padded_pages`
//     -> `indexer-k store: only group ends land, everything else is untouched`
//     num_tokens {1,17,257} x compress_ratio {1,2} x fp4 {off,on}, rtol=0
//     atol=0, and the same zero-sign normalization upstream applies before
//     comparing (a fused RoPE can flip the sign of a zero without changing its
//     value).
//     DROPPED AXIS, unavoidable: `cache_dtype [torch.float32, torch.bfloat16]`
//     (`:356`), which upstream applies to `cos_sin` alone (`:382`). It gates
//     that the kernel widens a bf16 `cos_sin_cache` on load. Our signature is
//     `const float* cos_sin_cache`, so the bf16 arm cannot be expressed and
//     there is no load-time widening to get wrong; the f32 arm is what a host
//     reference is. Carrying the axis would need the parameter, and adding a
//     parameter no caller sets is how dead code lands. It is recorded here
//     rather than owed, because there is nothing to do later.
//   * `:443` `test_indexer_k_cuda_graph_replay_reads_current_projection`,
//     WITH its `use_fp4 [False, True]` parametrization (`:441`)
//     -> `indexer-k store: the op carries NO state between calls`. ADAPTED:
//     there is no CUDA graph to capture on a host reference. What that test
//     actually asserts is that a replay consumes the CURRENT `k_pre` and slot
//     mapping rather than values captured at trace time, i.e. that the op holds
//     no state across calls. That property is portable and is what is asserted;
//     the capture mechanism is not.
//   * `:496` `test_indexer_k_store_roundtrips_through_rocm_gather`
//     -> `indexer-k store: the ROCm 16x16 tiled layout round-trips`.
//     block_size {16,64,128} x compress_ratio {1,2}, exact. ADAPTED: the reader
//     `cp_gather_indexer_k_quant_cache_triton` is an aiter Triton kernel with no
//     counterpart in this tree, so the gather side is written here from the
//     layout `indexer_k_store.py:189-206` declares, and the round-trip is
//     closed against the ROW-MAJOR write of the same input. A tiled write that
//     permuted bytes would then disagree with the row-major one.
//   * `tests/kernels/core/test_fused_q_kv_rmsnorm.py:36`
//     `test_fused_q_kv_rmsnorm_correctness` -> `fused q/kv rmsnorm`.
//     THREE AXES ADAPTED, each argued at the case: the `num_tokens = 8192`
//     rung (`:34`) is a launch-grid fixture, the `dtype [bfloat16, float16]`
//     axis (`:35`) gates an output cast an f32 signature does not have, and
//     `rtol=1e-2 atol=1e-2` (`:51`) is sized for that cast and is SUBSTITUTED
//     by two narrower checks rather than carried.
//
// UPSTREAM HAS NO COUNTERPART, AND THIS IS THE GAP THE SPEC ASKED TO BE STATED:
//   * `kv_source_layer_ids`, `index_source_layer_ids`,
//     `candidate_source_layer_id` and `candidate_topk_blocks` have ZERO
//     references anywhere under upstream `tests/` at `e77daef89e`. No upstream
//     test constructs a V4.1 attention layer at all. Every W3d case below is
//     therefore OURS, gated against the published `config.json` and against
//     `attention.py` read line by line. It is a real gate and it is not a
//     ported one.
//   * On NVIDIA, `indexer_k_norm_rope_store` is exercised at `block_size = 16`
//     only. The REAL block sizes the V4.1 backend returns are 64 (Hopper) and
//     128 (everything else) — `indexer.py:252-259` — and neither is covered by
//     any upstream CUDA case. Only the ROCm case reaches 64 and 128. The cases
//     below run 16, 64 and 128 on BOTH layouts for that reason.
//   * `tests/kernels/core/test_fused_q_kv_rmsnorm.py:123` and `:171` gate the
//     MXFP8 FUSION (`fused_q_kv_rmsnorm_quant`), which
//     `can_fuse_query_quant` refuses outright off CUDA and then again off a
//     FlashInfer MXFP8 linear. A host reference takes the unfused branch by
//     construction, so those two cases have nothing here to assert. The fusion
//     is owed PERF under the spec's `## Owed`, not a correctness gap.
//
// REACHABILITY. Nothing reaches this code yet. The wiring is owed to WAVE W4
// of row `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`, which is the ROW
// AGENTS.md §"Nothing lands dead" asks to be named; W4 is the wave inside it
// that owns the host forward composing W3a-W3f. Every case below constructs its
// inputs by hand, which that section says proves the class works and never that
// anything reaches it. That is the disclosed staged slice, named in the landing
// commit body and in the spec's `## Owed`, and not an oversight.
#include "vllm/model_executor/models/deepseek_v4_1_indexer.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/models/deepseek_v4_1.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace {

using vllm::DeepseekV41IndexerKFormat;
using vllm::DeepseekV41IndexerKLayout;
using vllm::DeepseekV41Params;

constexpr int64_t kHeadDim = 128;  // index_head_dim, NOT shrunk
constexpr int64_t kRopeDim = 64;   // qk_rope_head_dim, NOT shrunk
constexpr uint8_t kPad = 0xA5;     // upstream's page-padding sentinel
constexpr int64_t kMaxPos = 4096;

// A deterministic, portable generator. `torch.manual_seed` is not reproducible
// here, so the fixtures are generated instead of transcribed; what is preserved
// from upstream is the SHAPE of each fixture (zero rows, denormal rows, NaN
// rows, -1 slots, rows past `num_tokens`), which is what each case is for.
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<uint32_t>(s >> 32);
  }
  // Roughly standard-normal, rounded to bf16 because upstream's `k_pre` is a
  // bf16 tensor and a f32-precision fixture would gate a value the real op
  // never sees.
  float Normal() {
    double u = 0.0;
    for (int i = 0; i < 6; ++i) u += static_cast<double>(Next()) / 4294967296.0;
    return vt::BF16ToF32(vt::F32ToBF16(static_cast<float>((u - 3.0) * 0.8)));
  }
};

inline float Bf16(float x) { return vt::BF16ToF32(vt::F32ToBF16(x)); }

// The INDEPENDENT reference: `_indexer_k_reference`
// (`test_fused_indexer_q_rope_quant.py:333-350`) transcribed. It deliberately
// follows the TORCH shape of the computation — normalize the whole row, slice
// `k[:, 64::2]` / `k[:, 65::2]`, rotate, concatenate, then quantize the
// assembled row — rather than the kernel's even/odd register split, so that a
// defect in the split cannot be reproduced identically on both sides.
//
// IT ACCUMULATES IN f32, BECAUSE UPSTREAM'S REFERENCE DOES. `_indexer_k_reference`
// opens `k = k_pre.float()` (`:336`) and stays in fp32 to the quantizer; the
// kernel it checks says the same thing in words at `indexer_k_store.py:156`.
// An f64 reference here would share NO type with the port, so widening the
// port's accumulation back to f64 would leave this comparison green — which is
// exactly the axis the gate exists to hold, since the later device port
// computes in f32.
// Returns the value bytes and the scale bytes for one token.
// `f64_norm` exists for ONE caller: the accumulation-width pin below, which has
// to be able to compute the f64 answer in order to prove that the fixture can
// tell the two widths apart. Every other caller takes the default and gets the
// faithful fp32 reference.
void ReferenceRow(const std::vector<float>& k_pre_row, int64_t position,
                  const std::vector<float>& cos_sin, const std::vector<float>& w,
                  int64_t compress_ratio, bool use_fp4,
                  std::vector<uint8_t>* values, std::vector<uint8_t>* scales,
                  bool f64_norm = false) {
  std::vector<float> k(static_cast<size_t>(kHeadDim));
  if (f64_norm) {
    double sq64 = 0.0;
    for (int64_t i = 0; i < kHeadDim; ++i)
      sq64 += static_cast<double>(k_pre_row[static_cast<size_t>(i)]) *
              static_cast<double>(k_pre_row[static_cast<size_t>(i)]);
    const double rr64 =
        1.0 / std::sqrt(sq64 / static_cast<double>(kHeadDim) + 1e-20);
    for (int64_t i = 0; i < kHeadDim; ++i) {
      k[static_cast<size_t>(i)] = Bf16(static_cast<float>(
          static_cast<double>(k_pre_row[static_cast<size_t>(i)]) * rr64 *
          static_cast<double>(w[static_cast<size_t>(i)])));
    }
  } else {
  float sq = 0.0F;
  for (int64_t i = 0; i < kHeadDim; ++i)
    sq += k_pre_row[static_cast<size_t>(i)] * k_pre_row[static_cast<size_t>(i)];
  const float rrms = 1.0F / std::sqrt(sq / static_cast<float>(kHeadDim) + 1e-20F);
  for (int64_t i = 0; i < kHeadDim; ++i) {
    k[static_cast<size_t>(i)] =
        Bf16(k_pre_row[static_cast<size_t>(i)] * rrms * w[static_cast<size_t>(i)]);
  }
  }
  // `group_positions = positions // compress_ratio * compress_ratio`
  const int64_t gp = (position / compress_ratio) * compress_ratio;
  const float* cs = cos_sin.data() + gp * kRopeDim;
  std::vector<float> out(k);
  for (int64_t j = 0; j < kRopeDim / 2; ++j) {
    // `even, odd = k[:, 64::2], k[:, 65::2]`
    const float e = k[static_cast<size_t>(kRopeDim + 2 * j)];
    const float o = k[static_cast<size_t>(kRopeDim + 2 * j + 1)];
    const float c = cs[j];
    const float s = cs[kRopeDim / 2 + j];
    out[static_cast<size_t>(kRopeDim + 2 * j)] = e * c - o * s;
    out[static_cast<size_t>(kRopeDim + 2 * j + 1)] = o * c + e * s;
  }
  // `.to(torch.bfloat16)` over the whole assembled row.
  std::vector<float> row(static_cast<size_t>(kHeadDim));
  for (int64_t i = 0; i < kHeadDim; ++i) row[static_cast<size_t>(i)] = Bf16(out[static_cast<size_t>(i)]);

  if (use_fp4) {
    values->assign(static_cast<size_t>(kHeadDim / 2), 0);
    scales->assign(static_cast<size_t>(kHeadDim / 32), 0);
    for (int64_t b = 0; b < kHeadDim / 32; ++b) {
      float amax = 0.0F;
      for (int64_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(row[static_cast<size_t>(32 * b + i)]));
      amax = std::max(amax, 6.0F * std::ldexp(1.0F, -126));
      double lr = std::ceil(std::log2(static_cast<double>(amax) / 6.0));
      lr = std::min(std::max(lr, -127.0), 127.0);
      (*scales)[static_cast<size_t>(b)] = static_cast<uint8_t>(lr + 127.0);
      const double inv = std::exp2(-lr);
      for (int64_t i = 0; i < 16; ++i) {
        const float lo = vllm::CastToFp4(static_cast<float>(row[static_cast<size_t>(32 * b + 2 * i)] * inv));
        const float hi = vllm::CastToFp4(static_cast<float>(row[static_cast<size_t>(32 * b + 2 * i + 1)] * inv));
        (*values)[static_cast<size_t>(16 * b + i)] = static_cast<uint8_t>(
            (vllm::Fp4ToNibble(lo) & 0x0FU) | (vllm::Fp4ToNibble(hi) << 4));
      }
    }
    return;
  }
  values->assign(static_cast<size_t>(kHeadDim), 0);
  scales->assign(4, 0);
  float amax = 0.0F;
  for (int64_t i = 0; i < kHeadDim; ++i) amax = std::max(amax, std::fabs(row[static_cast<size_t>(i)]));
  amax = std::max(amax, 1e-4F);
  const double expo = std::ceil(std::log2(static_cast<double>(amax) / 448.0));
  const double scale = std::exp2(expo);
  for (int64_t i = 0; i < kHeadDim; ++i) {
    double v = static_cast<double>(row[static_cast<size_t>(i)]) / scale;
    v = std::min(std::max(v, -448.0), 448.0);
    (*values)[static_cast<size_t>(i)] = vllm::F32ToF8E4M3(static_cast<float>(v));
  }
  const float sf = static_cast<float>(scale);
  std::memcpy(scales->data(), &sf, sizeof(float));
}

// Upstream normalizes signed zeros away before comparing: "Fused RoPE can
// change zero signs without changing their numerical values"
// (`test_fused_indexer_q_rope_quant.py:425-434`). Same rule, same place.
void NormalizeZeroSigns(std::vector<uint8_t>* v, bool use_fp4) {
  for (uint8_t& b : *v) {
    if (use_fp4) {
      if ((b & 0x07U) == 0) b &= static_cast<uint8_t>(~0x08U);
      if (((b >> 4) & 0x07U) == 0) b &= static_cast<uint8_t>(~0x80U);
    } else if (b == 128) {
      b = 0;
    }
  }
}

DeepseekV41Params PublishedParams() {
  const std::string path = std::string(DEEPSEEK_V4_1_CKPT_FIXTURE_DIR) + "/config.json";
  std::ifstream f(path);
  REQUIRE_MESSAGE(f.good(), "missing fixture config.json at " << path);
  nlohmann::json j;
  f >> j;
  return vllm::ParseDeepseekV41Params(vllm::ParseHfConfig(j, path));
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// W3d — topology
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("v4.1 topology: source resolution is max(s <= L) on the REAL config") {
  const DeepseekV41Params p = PublishedParams();
  REQUIRE(p.kv_source_layer_ids == std::vector<int64_t>{2, 8, 14, 20});
  REQUIRE(p.index_source_layer_ids ==
          std::vector<int64_t>{2, 8, 14, 20, 24, 28, 32, 36});
  REQUIRE(p.num_hidden_layers == 40);

  // THE DECISIVE CASE. `kv_source_layers[-1]` is indistinguishable from
  // `max(s <= L)` for every layer at or above the last source (20-39), so a
  // gate built only from decoder layers cannot see that mutation. These are
  // the layers BELOW the last source — the "encoder" half — where the two
  // rules disagree, and where `max(s <= L)` also has to beat "the previous
  // source" and "the previous layer".
  struct Expect {
    int64_t layer, ratio, kv, index;
  };
  const Expect below[] = {
      {2, 2, 2, 2},    // a source at EXACTLY layer_id resolves to itself
      {3, 2, 2, 2},    {7, 2, 2, 2},    // >= 2 consumers of source 2
      {8, 2, 8, 8},    {9, 2, 8, 8},    {13, 2, 8, 8},
      {14, 2, 14, 14}, {15, 2, 14, 14}, {19, 2, 14, 14},
  };
  for (const Expect& e : below) {
    CAPTURE(e.layer);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, e.layer);
    CHECK(t.compress_ratio == e.ratio);
    CHECK(t.kv_source_layer_id == e.kv);
    CHECK(t.index_source_layer_id == e.index);
    // `kv_source_layers[-1]` would answer 20 for every one of these.
    CHECK(t.kv_source_layer_id != p.kv_source_layer_ids.back());
  }

  // The decoder half: ratio 1 is FULL-LENGTH COMPRESSED, not sliding window,
  // and every one of layers 20-39 resolves its KV to 20 — which IS the release
  // card's "decoder KV projected from the final encoder hidden states".
  for (int64_t l = 20; l < 40; ++l) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.compress_ratio == 1);
    CHECK(t.kv_source_layer_id == 20);
  }
  // ... while the INDEX source advances four times inside that same half.
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 23).index_source_layer_id == 20);
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 24).index_source_layer_id == 24);
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 27).index_source_layer_id == 24);
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 36).index_source_layer_id == 36);
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 39).index_source_layer_id == 36);
  // So the two lists CANNOT be collapsed into one: at layer 24 they disagree.
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 24).kv_source_layer_id !=
        vllm::DeepseekV41ResolveLayerTopology(p, 24).index_source_layer_id);
}

TEST_CASE("v4.1 topology: ratio 0 resolves NO source, and the MTP tail is 0") {
  const DeepseekV41Params p = PublishedParams();
  for (const int64_t l : {int64_t{0}, int64_t{1}}) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.compress_ratio == 0);
    CHECK(t.kv_source_layer_id == -1);     // upstream's None
    CHECK(t.index_source_layer_id == -1);
    CHECK_FALSE(t.is_kv_source);
    CHECK_FALSE(t.is_index_source);
  }
  // The three DSpark/MTP layers: `compress_ratios` has 43 entries against 40
  // layers, so 40-42 read 0 from the list, and `is_backbone` is false for them
  // regardless (attention.py:273).
  REQUIRE(static_cast<int64_t>(p.compress_ratios.size()) == 43);
  for (int64_t l = 40; l < 43; ++l) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.compress_ratio == 0);
    CHECK_FALSE(t.is_backbone);
  }
  // Past the end of the list: the BOUNDS GUARD, not an out-of-range read.
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 43).compress_ratio == 0);
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 999).compress_ratio == 0);
}

TEST_CASE("v4.1 topology: EIGHT indexers, FOUR index-K caches") {
  const DeepseekV41Params p = PublishedParams();
  const auto m = vllm::DeepseekV41ResolveModelTopology(p);
  CHECK(m.compressed_kv_cache_layers == std::vector<int64_t>{2, 8, 14, 20});
  CHECK(m.indexer_layers == std::vector<int64_t>{2, 8, 14, 20, 24, 28, 32, 36});
  // THE GAP. An index source that is not a kv source has no `wk`/`k_norm`, so
  // it allocates nothing and reads the cache of its kv source
  // (attention.py:371-395). Allocating eight caches passes every shape check.
  CHECK(m.index_k_cache_layers == std::vector<int64_t>{2, 8, 14, 20});
  CHECK(m.indexer_layers.size() == 2 * m.index_k_cache_layers.size());

  // The owner is keyed by the KV source, never by the INDEX source: layers 24,
  // 28, 32 and 36 are each their own index source and all four read layer 20's
  // cache.
  for (const int64_t l : {int64_t{24}, int64_t{28}, int64_t{32}, int64_t{36}}) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.is_index_source);
    CHECK_FALSE(t.is_kv_source);
    CHECK_FALSE(t.owns_index_k_cache);
    CHECK(t.index_source_layer_id == l);            // itself
    CHECK(t.index_k_cache_owner_layer_id == 20);    // but NOT its own cache
  }
  for (const int64_t l : {int64_t{2}, int64_t{8}, int64_t{14}, int64_t{20}}) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.is_kv_source);
    CHECK(t.owns_index_k_cache);
    CHECK(t.index_k_cache_owner_layer_id == l);
  }
  // A consumer that is not an index source carries no cache reference at all.
  CHECK(vllm::DeepseekV41ResolveLayerTopology(p, 25).index_k_cache_owner_layer_id == -1);
}

TEST_CASE("v4.1 topology: the candidate pre-filter is scoped to index sources") {
  const DeepseekV41Params p = PublishedParams();
  REQUIRE(p.candidate_source_layer_id == 20);
  REQUIRE(p.candidate_topk_blocks == 2048);
  REQUIRE(p.candidate_block_size == 8);
  const auto src = vllm::DeepseekV41ResolveLayerTopology(p, 20);
  CHECK(src.is_candidate_source);
  CHECK_FALSE(src.uses_candidates);   // the publisher does not mask itself
  for (const int64_t l : {int64_t{24}, int64_t{28}, int64_t{32}, int64_t{36}}) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK(t.uses_candidates);
    CHECK_FALSE(t.is_candidate_source);
  }
  // Below the candidate source: an index source that neither publishes nor
  // consumes.
  for (const int64_t l : {int64_t{2}, int64_t{8}, int64_t{14}}) {
    CAPTURE(l);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, l);
    CHECK_FALSE(t.is_candidate_source);
    CHECK_FALSE(t.uses_candidates);
  }
  // A non-index-source layer above 20 answers neither, because upstream scopes
  // both flags inside `if self.is_index_source:` (attention.py:369,398-399).
  const auto plain = vllm::DeepseekV41ResolveLayerTopology(p, 30);
  CHECK_FALSE(plain.is_index_source);
  CHECK_FALSE(plain.uses_candidates);
  CHECK_FALSE(plain.is_candidate_source);
}

TEST_CASE("v4.1 topology: per-layer RoPE is NOT uniform, and ratio 1 is compressed") {
  const DeepseekV41Params p = PublishedParams();
  REQUIRE(p.rope_theta == doctest::Approx(10000.0));
  REQUIRE(p.compress_rope_theta == doctest::Approx(160000.0));
  REQUIRE(p.rope_scale_factor == doctest::Approx(16.0));
  REQUIRE(p.rope_orig_ctx == 65536);

  // Ratio 0: plain RoPE. theta 10000, YaRN pinned to the identity.
  for (const int64_t l : {int64_t{0}, int64_t{1}}) {
    CAPTURE(l);
    const auto r = vllm::DeepseekV41ResolveLayerRope(p, l);
    CHECK(r.rope_theta == doctest::Approx(10000.0));
    CHECK_FALSE(r.use_yarn);
    CHECK(r.factor == doctest::Approx(1.0));
    CHECK(r.original_max_position_embeddings == p.max_position_embeddings);
  }
  // Ratio 2 AND ratio 1: theta 160000 with YaRN at factor 16. Under V4's
  // `compress_ratio > 1` threshold every ratio-1 layer would take theta 10000
  // with no scaling — half the stack, no shape change (rope.py, whose whole
  // delta against V4's copy is `> 1` becoming `> 0`).
  for (const int64_t l : {int64_t{2}, int64_t{19}, int64_t{20}, int64_t{39}}) {
    CAPTURE(l);
    const auto r = vllm::DeepseekV41ResolveLayerRope(p, l);
    CHECK(r.rope_theta == doctest::Approx(160000.0));
    CHECK(r.use_yarn);
    CHECK(r.factor == doctest::Approx(16.0));
    CHECK(r.original_max_position_embeddings == 65536);
  }
  // The MTP tail falls back to the plain arm through the bounds guard.
  CHECK_FALSE(vllm::DeepseekV41ResolveLayerRope(p, 41).use_yarn);
  CHECK(vllm::DeepseekV41ResolveLayerRope(p, 41).rope_theta == doctest::Approx(10000.0));
}

TEST_CASE("v4.1 topology: the refusals upstream raises") {
  DeepseekV41Params p = PublishedParams();

  SUBCASE("a ratio outside {0,1,2}") {
    p.compress_ratios[4] = 4;  // V4's indexer ratio, which V4.1 has no arm for
    CHECK_THROWS_WITH_AS(vllm::DeepseekV41ResolveLayerTopology(p, 4),
                         doctest::Contains("compress_ratio=4"), std::exception);
  }
  SUBCASE("a compressed layer with empty source lists") {
    p.kv_source_layer_ids.clear();
    CHECK_THROWS_WITH_AS(vllm::DeepseekV41ResolveLayerTopology(p, 5),
                         doctest::Contains("kv_source_layer_ids"), std::exception);
  }
  SUBCASE("a compressed layer BELOW every source") {
    // Python's `max()` over an empty generator raises; folding this to the
    // first source instead would run the model against a cache nobody wrote.
    p.kv_source_layer_ids = {30};
    p.index_source_layer_ids = {30};
    CHECK_THROWS_WITH_AS(vllm::DeepseekV41ResolveLayerTopology(p, 5),
                         doctest::Contains("at or below"), std::exception);
  }
  SUBCASE("is_backbone gates membership: an MTP index in a source list") {
    p.kv_source_layer_ids.push_back(41);
    p.index_source_layer_ids.push_back(41);
    const auto t = vllm::DeepseekV41ResolveLayerTopology(p, 41);
    CHECK_FALSE(t.is_kv_source);
    CHECK_FALSE(t.is_index_source);
    const auto m = vllm::DeepseekV41ResolveModelTopology(p);
    CHECK(m.compressed_kv_cache_layers == std::vector<int64_t>{2, 8, 14, 20});
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// W3c — the indexer arm
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("v4.1 indexer: cache row bytes and the arch-conditional block sizes") {
  // attention.py:82-90 — 132 and 68 at index_head_dim 128, and this is the one
  // place the two byte widths are stated.
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(128, DeepseekV41IndexerKFormat::kFp8) == 132);
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(128, DeepseekV41IndexerKFormat::kMxfp4) == 68);
  // A SECOND index_head_dim, because 128 alone does not gate the FORMULAS.
  // Upstream writes `index_head_dim + index_head_dim // 128 * 4` (`:90`) and
  // `index_head_dim // 2 + index_head_dim // MXFP4_BLOCK_SIZE` (`:87`), and at
  // 128 the scale terms are 4 and 4 — so a port that wrote a constant `+ 4` in
  // either arm answers 132 and 68 and looks right. It is wrong the moment the
  // published `index_head_dim` changes, and a V4.1 variant that changes it is
  // exactly the reader this function has. At 256 the two scale terms become 8
  // and 8, which is where the constant and the formula part.
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(256, DeepseekV41IndexerKFormat::kFp8) == 264);
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(256, DeepseekV41IndexerKFormat::kMxfp4) == 136);
  // And the scale region is a fixed FRACTION of the values, not a constant:
  // doubling the width doubles both scale terms.
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(256, DeepseekV41IndexerKFormat::kFp8) - 256 ==
        2 * (vllm::DeepseekV41IndexerKCacheRowBytes(128, DeepseekV41IndexerKFormat::kFp8) - 128));
  CHECK(vllm::DeepseekV41IndexerKCacheRowBytes(256, DeepseekV41IndexerKFormat::kMxfp4) - 128 ==
        2 * (vllm::DeepseekV41IndexerKCacheRowBytes(128, DeepseekV41IndexerKFormat::kMxfp4) - 64));
  // indexer.py:252-259 — ARCH-CONDITIONAL, against V4's flat [256]. Reading it
  // as a flat 128 is wrong on Hopper and only on Hopper.
  CHECK(vllm::DeepseekV41IndexerBlockSizes(/*is_hopper_family=*/true) ==
        std::vector<int64_t>{64});
  CHECK(vllm::DeepseekV41IndexerBlockSizes(/*is_hopper_family=*/false) ==
        std::vector<int64_t>{128});
}

TEST_CASE("v4.1 indexer: only group ends land, everything else is untouched") {
  // Ported from test_fused_indexer_q_rope_quant.py:358. num_tokens {1,17,257}
  // x compress_ratio {1,2} x use_fp4 {off,on} preserved; the `cache_dtype`
  // axis (`:356`) is the one DROPPED axis, argued in the file header.
  for (const int64_t num_tokens : {int64_t{1}, int64_t{17}, int64_t{257}}) {
    for (const int64_t ratio : {int64_t{1}, int64_t{2}}) {
      for (const bool use_fp4 : {false, true}) {
        CAPTURE(num_tokens);
        CAPTURE(ratio);
        CAPTURE(use_fp4);
        const auto format = use_fp4 ? DeepseekV41IndexerKFormat::kMxfp4
                                    : DeepseekV41IndexerKFormat::kFp8;
        Rng rng(static_cast<uint64_t>(num_tokens * 7 + ratio * 3 + (use_fp4 ? 1 : 0)));
        const int64_t num_rows = num_tokens + 3;

        std::vector<float> k_pre(static_cast<size_t>(num_rows * kHeadDim));
        for (float& v : k_pre) v = rng.Normal();
        std::vector<float> w(static_cast<size_t>(kHeadDim));
        for (float& v : w) v = rng.Normal();
        // `positions = torch.arange(7, num_tokens + 10)`
        std::vector<int64_t> positions(static_cast<size_t>(num_rows));
        for (int64_t i = 0; i < num_rows; ++i) positions[static_cast<size_t>(i)] = 7 + i;
        // A permutation of 2*num_tokens slots, with every 7th from index 5 set
        // to -1 (upstream's `slots[5::7] = -1`).
        std::vector<int64_t> slots(static_cast<size_t>(2 * num_tokens));
        for (int64_t i = 0; i < 2 * num_tokens; ++i) slots[static_cast<size_t>(i)] = i;
        for (int64_t i = 2 * num_tokens - 1; i > 0; --i)
          std::swap(slots[static_cast<size_t>(i)],
                    slots[static_cast<size_t>(rng.Next() % static_cast<uint32_t>(i + 1))]);
        slots.resize(static_cast<size_t>(num_tokens));
        for (int64_t i = 5; i < num_tokens; i += 7) slots[static_cast<size_t>(i)] = -1;

        std::vector<bool> valid(static_cast<size_t>(num_tokens));
        for (int64_t i = 0; i < num_tokens; ++i) {
          valid[static_cast<size_t>(i)] =
              slots[static_cast<size_t>(i)] >= 0 &&
              (positions[static_cast<size_t>(i)] + 1) % ratio == 0;
        }
        // Upstream's four fixture shapes, preserved: exact-zero rows, denormal
        // rows, NaN at every skipped row, and NaN at every row past
        // `num_tokens`. The NaNs are the assertion: a row that reaches an
        // output poisons it.
        for (int64_t i = 3; i < num_rows; i += 7)
          for (int64_t d = 0; d < kHeadDim; ++d) k_pre[static_cast<size_t>(i * kHeadDim + d)] = 0.0F;
        for (int64_t i = 8; i < num_rows; i += 11)
          for (int64_t d = 0; d < kHeadDim; ++d) k_pre[static_cast<size_t>(i * kHeadDim + d)] *= 1e-10F;
        for (int64_t i = 0; i < num_tokens; ++i)
          if (!valid[static_cast<size_t>(i)])
            for (int64_t d = 0; d < kHeadDim; ++d)
              k_pre[static_cast<size_t>(i * kHeadDim + d)] = std::nanf("");
        for (int64_t i = num_tokens; i < num_rows; ++i) {
          for (int64_t d = 0; d < kHeadDim; ++d)
            k_pre[static_cast<size_t>(i * kHeadDim + d)] = std::nanf("");
          positions[static_cast<size_t>(i)] = kMaxPos + 10;  // would fault a read
        }

        std::vector<float> cos_sin(static_cast<size_t>(kMaxPos * kRopeDim));
        for (int64_t pos = 0; pos < kMaxPos; ++pos)
          for (int64_t j = 0; j < kRopeDim / 2; ++j) {
            const double a = rng.Normal();
            cos_sin[static_cast<size_t>(pos * kRopeDim + j)] = static_cast<float>(std::cos(a));
            cos_sin[static_cast<size_t>(pos * kRopeDim + kRopeDim / 2 + j)] =
                static_cast<float>(std::sin(a));
          }

        const int64_t block_size = 16;  // upstream's only CUDA block size
        const int64_t row_bytes = vllm::DeepseekV41IndexerKCacheRowBytes(kHeadDim, format);
        const int64_t value_bytes = use_fp4 ? kHeadDim / 2 : kHeadDim;
        const int64_t num_blocks = (2 * num_tokens + block_size - 1) / block_size + 1;
        std::vector<uint8_t> cache(static_cast<size_t>(num_blocks * block_size * row_bytes), kPad);
        std::vector<uint8_t> expected = cache;

        for (int64_t i = 0; i < num_tokens; ++i) {
          if (!valid[static_cast<size_t>(i)]) continue;
          std::vector<float> row(k_pre.begin() + i * kHeadDim,
                                 k_pre.begin() + (i + 1) * kHeadDim);
          std::vector<uint8_t> v, s;
          ReferenceRow(row, positions[static_cast<size_t>(i)], cos_sin, w, ratio, use_fp4, &v, &s);
          const int64_t slot = slots[static_cast<size_t>(i)];
          uint8_t* page = expected.data() + (slot / block_size) * block_size * row_bytes;
          std::memcpy(page + (slot % block_size) * value_bytes, v.data(), v.size());
          std::memcpy(page + block_size * value_bytes + (slot % block_size) * s.size(),
                      s.data(), s.size());
        }

        vllm::DeepseekV41IndexerKNormRopeStore(
            k_pre.data(), positions.data(), cos_sin.data(), w.data(), 1e-20,
            num_tokens, num_rows, kHeadDim, kRopeDim, ratio, format,
            DeepseekV41IndexerKLayout::kRowMajor, slots.data(), num_blocks,
            block_size, cache.data());

        NormalizeZeroSigns(&cache, use_fp4);
        NormalizeZeroSigns(&expected, use_fp4);
        // rtol=0 atol=0, on every byte of the page INCLUDING the padding.
        CHECK(cache == expected);
        // And explicitly: at least one token was skipped and at least one
        // landed, so neither side of the guard is vacuous.
        const int64_t landed = std::count(valid.begin(), valid.end(), true);
        CHECK(landed > 0);
        // At num_tokens == 1 there is no index-5 slot to set to -1 and, at
        // ratio 2, position 7 is itself a group end -- so the ONE token lands
        // and nothing is skipped. Above that both sides are populated.
        if (num_tokens > 5) CHECK(landed < num_tokens);
      }
    }
  }
}

TEST_CASE("v4.1 indexer: the op carries NO state between calls") {
  // Ported from test_fused_indexer_q_rope_quant.py:443, INCLUDING its
  // `use_fp4 [False, True]` parametrization (`:441`). ADAPTED in one respect
  // only: there is no CUDA graph here. What that test asserts is that a replay
  // consumes the CURRENT projection and slot mapping rather than trace-time
  // values, which for a host reference is exactly "the op is a pure function of
  // its arguments". The fp4 arm is not an adaptation and is not optional: the
  // MXFP4 branch is a second quantizer with its own block scales, and a state
  // defect in it would not show on the fp8 branch at all.
  for (const bool use_fp4 : {false, true}) {
    CAPTURE(use_fp4);
    const auto format = use_fp4 ? DeepseekV41IndexerKFormat::kMxfp4
                                : DeepseekV41IndexerKFormat::kFp8;
    const int64_t n = 19, block_size = 16, num_blocks = 2;
    const int64_t row_bytes = vllm::DeepseekV41IndexerKCacheRowBytes(kHeadDim, format);
    Rng rng(18);
    std::vector<float> k_pre(static_cast<size_t>(n * kHeadDim));
    for (float& v : k_pre) v = rng.Normal();
    std::vector<float> w(static_cast<size_t>(kHeadDim), 1.0F);
    std::vector<int64_t> positions(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) positions[static_cast<size_t>(i)] = i;
    std::vector<float> cos_sin(static_cast<size_t>(32 * kRopeDim), 0.0F);
    for (int64_t p = 0; p < 32; ++p)
      for (int64_t j = 0; j < kRopeDim / 2; ++j) cos_sin[static_cast<size_t>(p * kRopeDim + j)] = 1.0F;
    std::vector<int64_t> slots(17);
    for (int64_t i = 0; i < 17; ++i) slots[static_cast<size_t>(i)] = i;

    const auto run = [&](const std::vector<float>& kp, const std::vector<int64_t>& sl) {
      std::vector<uint8_t> c(static_cast<size_t>(num_blocks * block_size * row_bytes), kPad);
      vllm::DeepseekV41IndexerKNormRopeStore(
          kp.data(), positions.data(), cos_sin.data(), w.data(), 1e-20,
          static_cast<int64_t>(sl.size()), n, kHeadDim, kRopeDim, 2, format,
          DeepseekV41IndexerKLayout::kRowMajor, sl.data(), num_blocks,
          block_size, c.data());
      return c;
    };

    const std::vector<uint8_t> first = run(k_pre, slots);
    // Upstream's exact mutation: `k_pre.neg_()` and `slots.add_(4)`.
    std::vector<float> negated = k_pre;
    for (float& v : negated) v = -v;
    std::vector<int64_t> shifted = slots;
    for (int64_t& s : shifted) s += 4;
    const std::vector<uint8_t> second = run(negated, shifted);
    CHECK(first != second);                  // the call READ the new inputs
    CHECK(run(negated, shifted) == second);  // and is deterministic
    CHECK(run(k_pre, slots) == first);       // with nothing carried forward
  }
}

TEST_CASE("v4.1 indexer: the ROCm 16x16 tiled layout round-trips") {
  // Ported from test_fused_indexer_q_rope_quant.py:496, block sizes and ratios
  // preserved. The 64 and 128 rungs are also the REAL backend block sizes that
  // no upstream CUDA case reaches (indexer.py:252-259).
  for (const int64_t block_size : {int64_t{16}, int64_t{64}, int64_t{128}}) {
    for (const int64_t ratio : {int64_t{1}, int64_t{2}}) {
      CAPTURE(block_size);
      CAPTURE(ratio);
      Rng rng(static_cast<uint64_t>(block_size + ratio));
      const int64_t num_tokens = 3 * block_size + 5;
      const int64_t num_blocks = num_tokens / block_size + 2;
      const int64_t row_bytes =
          vllm::DeepseekV41IndexerKCacheRowBytes(kHeadDim, DeepseekV41IndexerKFormat::kFp8);
      std::vector<float> k_pre(static_cast<size_t>(num_tokens * kHeadDim));
      for (float& v : k_pre) v = rng.Normal();
      std::vector<float> w(static_cast<size_t>(kHeadDim));
      for (float& v : w) v = rng.Normal();
      std::vector<int64_t> positions(static_cast<size_t>(num_tokens)), slots(static_cast<size_t>(num_tokens));
      for (int64_t i = 0; i < num_tokens; ++i) positions[static_cast<size_t>(i)] = slots[static_cast<size_t>(i)] = i;
      std::vector<float> cos_sin(static_cast<size_t>(kMaxPos * kRopeDim));
      for (int64_t p = 0; p < kMaxPos; ++p)
        for (int64_t j = 0; j < kRopeDim / 2; ++j) {
          const double a = rng.Normal();
          cos_sin[static_cast<size_t>(p * kRopeDim + j)] = static_cast<float>(std::cos(a));
          cos_sin[static_cast<size_t>(p * kRopeDim + kRopeDim / 2 + j)] = static_cast<float>(std::sin(a));
        }

      const auto store = [&](DeepseekV41IndexerKLayout layout) {
        std::vector<uint8_t> c(static_cast<size_t>(num_blocks * block_size * row_bytes), 0);
        vllm::DeepseekV41IndexerKNormRopeStore(
            k_pre.data(), positions.data(), cos_sin.data(), w.data(), 1e-20,
            num_tokens, num_tokens, kHeadDim, kRopeDim, ratio,
            DeepseekV41IndexerKFormat::kFp8, layout, slots.data(), num_blocks,
            block_size, c.data());
        return c;
      };
      const std::vector<uint8_t> row_major = store(DeepseekV41IndexerKLayout::kRowMajor);
      const std::vector<uint8_t> tiled = store(DeepseekV41IndexerKLayout::kRocmTiled16x16);
      // The two layouts must DIFFER, or the tiled arm is not tiling.
      CHECK(row_major != tiled);

      // The gather the ROCm readers perform, written from the layout
      // indexer_k_store.py:189-206 declares:
      //   [block_size/16][head_dim/16][16][16]
      for (int64_t t = 0; t < num_tokens; ++t) {
        if ((positions[static_cast<size_t>(t)] + 1) % ratio != 0) continue;
        const int64_t b = slots[static_cast<size_t>(t)] / block_size;
        const int64_t q = slots[static_cast<size_t>(t)] % block_size;
        const uint8_t* page = tiled.data() + b * block_size * row_bytes;
        const uint8_t* ref = row_major.data() + b * block_size * row_bytes + q * kHeadDim;
        for (int64_t lane = 0; lane < kHeadDim; ++lane) {
          const int64_t off = (q / 16) * 16 * kHeadDim + (q % 16) * 16 +
                              (lane / 16) * 16 * 16 + (lane % 16);
          REQUIRE(page[off] == ref[lane]);
        }
        // The SCALE region is identical in both layouts: only values tile.
        for (int64_t s = 0; s < 4; ++s) {
          REQUIRE(page[block_size * kHeadDim + q * 4 + s] ==
                  row_major[static_cast<size_t>(b * block_size * row_bytes + block_size * kHeadDim + q * 4 + s)]);
        }
      }
    }
  }
}

TEST_CASE("v4.1 indexer: the ROCm tiled layout REFUSES what its readers lack") {
  std::vector<uint8_t> cache(4096, 0);
  std::vector<float> k(static_cast<size_t>(kHeadDim), 1.0F), w(static_cast<size_t>(kHeadDim), 1.0F);
  std::vector<float> cs(static_cast<size_t>(kRopeDim), 0.0F);
  std::vector<int64_t> pos{0}, slot{0};
  const auto call = [&](DeepseekV41IndexerKFormat f, int64_t block_size) {
    vllm::DeepseekV41IndexerKNormRopeStore(
        k.data(), pos.data(), cs.data(), w.data(), 1e-20, 1, 1, kHeadDim,
        kRopeDim, 1, f, DeepseekV41IndexerKLayout::kRocmTiled16x16, slot.data(),
        1, block_size, cache.data());
  };
  // indexer_k_store.py:80-84 — MXFP4 has no tiled ROCm layout at all.
  CHECK_THROWS_WITH_AS(call(DeepseekV41IndexerKFormat::kMxfp4, 16),
                       doctest::Contains("MXFP4"), std::exception);
  // indexer_k_store.py:85-90 — the tiling needs block_size % 16 == 0.
  CHECK_THROWS_WITH_AS(call(DeepseekV41IndexerKFormat::kFp8, 24),
                       doctest::Contains("16"), std::exception);
  // indexer_k_store.py:59 — ratio 4 is V4's, and V4.1 deletes that arm.
  CHECK_THROWS_WITH_AS(
      vllm::DeepseekV41IndexerKNormRopeStore(
          k.data(), pos.data(), cs.data(), w.data(), 1e-20, 1, 1, kHeadDim,
          kRopeDim, 4, DeepseekV41IndexerKFormat::kFp8,
          DeepseekV41IndexerKLayout::kRowMajor, slot.data(), 1, 16, cache.data()),
      doctest::Contains("compress_ratio"), std::exception);
}

TEST_CASE("v4.1 indexer: the store's RMSNorm accumulates in f32, not f64") {
  // `indexer_k_store.py:156` titles the norm block "k_norm (fp32 throughout,
  // bf16 roundtrip like the reference)" and `:157-161` loads, reduces and
  // scales in `tl.float32`; upstream's own reference opens `k = k_pre.float()`
  // (`test_fused_indexer_q_rope_quant.py:336`). So fp32 is MIRRORED behaviour,
  // and the device port this host reference will gate computes in fp32 too.
  //
  // WHY THIS CASE EXISTS SEPARATELY FROM THE BIG STORE CASE. A wider
  // accumulation is invisible to a value gate: it agrees with any reference
  // BETTER, and the bf16 round-trip plus the fp8 quantizer then swallow what is
  // left. Measured over the 2,310 random rows the "only group ends land" case
  // already runs, widening the norm to f64 changes NOT ONE stored byte. Only a
  // row that sits on a rounding boundary can see it, and this is one: found by
  // scanning 46,979 rows of the same generator for a row whose stored bytes
  // differ between the two widths. It is carried as literal bf16 bit patterns
  // rather than as a seed, so the case does not depend on `Rng`.
  //
  // The weight is 1, cos is 1 and sin is 0, so the RoPE is the identity and the
  // only thing left that can move a byte is the width of the norm.
  static const uint16_t kRowBf16[kHeadDim] = {
    0xbf4b, 0xbeb3, 0xbf31, 0x3eb6, 0x3f50, 0x3ef0, 0xbf27, 0xbf99,
    0xbf3c, 0x3fab, 0x3db3, 0x3f1d, 0x3ea1, 0x3f5f, 0x3eb2, 0xbeca,
    0x3e84, 0xbbaf, 0xbf85, 0x3e6f, 0xbf3d, 0x3d27, 0x3f28, 0x3e8a,
    0xbe85, 0x3e1f, 0x3e1e, 0xbe2a, 0xbed3, 0xbd67, 0x3f4d, 0xbe10,
    0x3f9a, 0xbf59, 0x3f53, 0x3d8f, 0xbeb5, 0x3dc0, 0xbdfe, 0x3f11,
    0xbf03, 0x3eb4, 0xbec8, 0xbec5, 0x3e44, 0x3ccb, 0x3f9f, 0xbedb,
    0x3df2, 0xbf44, 0xbf0e, 0x3eaf, 0xbec6, 0x3f5e, 0x3e55, 0x3f1e,
    0xbe57, 0xbf0f, 0x3dee, 0xbe28, 0x3f29, 0xbe93, 0xbf4d, 0x3f3f,
    0x3e80, 0x3e0c, 0xbe8e, 0x3f61, 0xbf6c, 0x3e31, 0x3def, 0xbdc0,
    0x3f19, 0xbf5e, 0x3f77, 0x3e4b, 0x3e6e, 0x3eb7, 0xbd85, 0xbf81,
    0x3d92, 0x3f1f, 0x3efa, 0xbeab, 0xbdda, 0x3f0a, 0x3f05, 0xbea5,
    0x3e85, 0xbf0c, 0xbef5, 0xbeeb, 0xbec0, 0xbf93, 0xbf92, 0xbf06,
    0xbe49, 0x3e19, 0xbc04, 0xbd94, 0xbd05, 0x3d4d, 0xbe20, 0xbfb2,
    0xbf26, 0x3f7e, 0xbf6b, 0xbefd, 0xbf0d, 0xbefd, 0xbeec, 0x3f81,
    0xbee9, 0x3db3, 0x3e14, 0x3f6c, 0xbef8, 0xbee3, 0xbf30, 0x3e9c,
    0x3f93, 0x3e5d, 0x3eb1, 0xbf0b, 0x3e26, 0xbf97, 0x3f44, 0x3dc3,
  };
  std::vector<float> k_pre(static_cast<size_t>(kHeadDim));
  for (int64_t i = 0; i < kHeadDim; ++i)
    k_pre[static_cast<size_t>(i)] = vt::BF16ToF32(kRowBf16[i]);
  const std::vector<float> w(static_cast<size_t>(kHeadDim), 1.0F);
  std::vector<float> cos_sin(static_cast<size_t>(kRopeDim), 0.0F);
  for (int64_t j = 0; j < kRopeDim / 2; ++j) cos_sin[static_cast<size_t>(j)] = 1.0F;

  std::vector<uint8_t> v32, s32, v64, s64;
  ReferenceRow(k_pre, 0, cos_sin, w, 1, /*use_fp4=*/false, &v32, &s32);
  ReferenceRow(k_pre, 0, cos_sin, w, 1, /*use_fp4=*/false, &v64, &s64,
               /*f64_norm=*/true);
  // THE INSTRUMENT STATES ITS OWN COMPARISON: the two widths must disagree on
  // this fixture, or the exact check below could never fail and would be
  // decoration. On the found row they differ in exactly one value byte.
  int64_t differing = 0;
  for (size_t i = 0; i < v32.size(); ++i)
    if (v32[i] != v64[i]) ++differing;
  for (size_t i = 0; i < s32.size(); ++i)
    if (s32[i] != s64[i]) ++differing;
  CAPTURE(differing);
  REQUIRE(differing > 0);

  const int64_t block_size = 16, num_blocks = 1;
  const int64_t row_bytes =
      vllm::DeepseekV41IndexerKCacheRowBytes(kHeadDim, DeepseekV41IndexerKFormat::kFp8);
  std::vector<uint8_t> cache(static_cast<size_t>(num_blocks * block_size * row_bytes), kPad);
  const std::vector<int64_t> positions{0}, slots{0};
  vllm::DeepseekV41IndexerKNormRopeStore(
      k_pre.data(), positions.data(), cos_sin.data(), w.data(), 1e-20, 1, 1,
      kHeadDim, kRopeDim, 1, DeepseekV41IndexerKFormat::kFp8,
      DeepseekV41IndexerKLayout::kRowMajor, slots.data(), num_blocks, block_size,
      cache.data());

  // And the port is on the f32 side of that disagreement, byte for byte.
  for (int64_t i = 0; i < kHeadDim; ++i) {
    CAPTURE(i);
    REQUIRE(cache[static_cast<size_t>(i)] == v32[static_cast<size_t>(i)]);
  }
  for (int64_t i = 0; i < 4; ++i) {
    CAPTURE(i);
    REQUIRE(cache[static_cast<size_t>(block_size * kHeadDim + i)] ==
            s32[static_cast<size_t>(i)]);
  }
}

TEST_CASE("v4.1 topology: the model roll-up SORTS and DEDUPES its layer lists") {
  // The header promises "Every vector is sorted ascending and holds DISTINCT
  // layer ids". The published `config.json` is already sorted and already
  // distinct, so every case built from it holds that promise whether the code
  // does or not: deleting both the `std::find` dedupe and the `std::sort` from
  // `BackboneIntersect` leaves them all green. This case feeds the shapes the
  // config cannot: out of order, and repeated.
  DeepseekV41Params p = PublishedParams();
  p.kv_source_layer_ids = {20, 2, 14, 8, 2, 20};
  p.index_source_layer_ids = {36, 2, 24, 8, 32, 14, 28, 20, 24, 36};
  const auto m = vllm::DeepseekV41ResolveModelTopology(p);

  // Sorted AND deduped, not merely "the same set".
  CHECK(m.compressed_kv_cache_layers == std::vector<int64_t>{2, 8, 14, 20});
  CHECK(m.indexer_layers == std::vector<int64_t>{2, 8, 14, 20, 24, 28, 32, 36});
  CHECK(m.index_k_cache_layers == std::vector<int64_t>{2, 8, 14, 20});

  // Stated as the two properties rather than only as an expected vector, so
  // that a reader can see which mutation each one kills.
  const auto is_sorted_distinct = [](const std::vector<int64_t>& v) {
    for (size_t i = 1; i < v.size(); ++i)
      if (v[i] <= v[i - 1]) return false;
    return true;
  };
  CHECK(is_sorted_distinct(m.compressed_kv_cache_layers));
  CHECK(is_sorted_distinct(m.indexer_layers));
  CHECK(is_sorted_distinct(m.index_k_cache_layers));
  // The input really did carry both defects, or the case above proves nothing.
  REQUIRE_FALSE(is_sorted_distinct(p.kv_source_layer_ids));
  REQUIRE_FALSE(is_sorted_distinct(p.index_source_layer_ids));
  CHECK(m.index_k_cache_layers.size() * 2 == m.indexer_layers.size());
}

TEST_CASE("v4.1 indexer: a slot past the end of the paged cache is REFUSED") {
  // This guard is OURS, not a port: `indexer_k_store.py` has no bounds check,
  // because in Triton the slot comes from a runner-built mapping and an
  // out-of-range one is a device-side fault rather than a silent host write.
  // A host reference writing through a raw `uint8_t*` gets neither, so it holds
  // the bound itself. Untested, it is a comment; this case is what makes it a
  // guard.
  const int64_t block_size = 16, num_blocks = 2;
  const int64_t row_bytes =
      vllm::DeepseekV41IndexerKCacheRowBytes(kHeadDim, DeepseekV41IndexerKFormat::kFp8);
  std::vector<uint8_t> cache(static_cast<size_t>(num_blocks * block_size * row_bytes), kPad);
  std::vector<float> k(static_cast<size_t>(kHeadDim), 1.0F), w(static_cast<size_t>(kHeadDim), 1.0F);
  std::vector<float> cs(static_cast<size_t>(kRopeDim), 0.0F);
  const std::vector<int64_t> positions{0};
  const auto call = [&](int64_t slot) {
    const std::vector<int64_t> slots{slot};
    vllm::DeepseekV41IndexerKNormRopeStore(
        k.data(), positions.data(), cs.data(), w.data(), 1e-20, 1, 1, kHeadDim,
        kRopeDim, 1, DeepseekV41IndexerKFormat::kFp8,
        DeepseekV41IndexerKLayout::kRowMajor, slots.data(), num_blocks,
        block_size, cache.data());
  };
  // The LAST legal slot is `num_blocks * block_size - 1` and it must be
  // accepted, so the bound is not off by one in the safe direction either.
  CHECK_NOTHROW(call(num_blocks * block_size - 1));
  // One past it, and far past it, are both refused.
  CHECK_THROWS_WITH_AS(call(num_blocks * block_size),
                       doctest::Contains("past the end"), std::exception);
  CHECK_THROWS_WITH_AS(call(1 << 20), doctest::Contains("past the end"),
                       std::exception);
  // The refusal is what stops the write: nothing outside the buffer was
  // touched, and the legal write above landed inside it.
  CHECK(cache.size() == static_cast<size_t>(num_blocks * block_size * row_bytes));
}

TEST_CASE("v4.1 indexer: fused q/kv rmsnorm") {
  // Ported from tests/kernels/core/test_fused_q_kv_rmsnorm.py:36
  // (`test_fused_q_kv_rmsnorm_correctness`), shapes and token counts preserved
  // (q_size 192 / kv_size 576, num_tokens {1,17,1024}).
  //
  // THREE UPSTREAM AXES ARE ADAPTED, AND EACH ONE IS NAMED HERE RATHER THAN
  // LEFT TO BE DISCOVERED:
  //
  //  1. `num_tokens = 8192` (`:34`) is DROPPED. The whole file exists for a
  //     launch-grid fix — its docstring says the old grid `(2, num_tokens)`
  //     hit CUDA's 65535 grid-y cap — so the large rung gates a grid geometry
  //     a host loop does not have. Nothing it asserts is portable.
  //  2. The `dtype [torch.bfloat16, torch.float16]` axis (`:35`) is DROPPED,
  //     because `DeepseekV41FusedQKvRmsNorm`'s signature is `const float*`.
  //     Upstream's axis is about the STORAGE dtype of `q`/`kv`/the weights:
  //     `_ref_rmsnorm` (`:27-31`) widens to fp32, computes, and casts back to
  //     `x.dtype`, so what the axis gates is that final narrowing cast. A host
  //     reference that carries f32 buffers has no such cast to get wrong. The
  //     ARITHMETIC dtype, which is fp32 on both arms, is gated below and is
  //     the part that is portable.
  //  3. `rtol=1e-2 atol=1e-2` (`:51`) is SUBSTITUTED, not preserved. Those
  //     numbers are sized for a bf16/fp16 round-trip of the OUTPUT, which axis
  //     2 just removed; carrying them here would be a band roughly three
  //     orders of magnitude wider than anything this code can produce, and a
  //     tolerance that loose cannot see a deleted stage. What replaces it is
  //     two checks with two different questions, each stating its own:
  //     an fp32 PAIRWISE reference at a tight bound for the value, and an fp32
  //     SERIAL reference at exact equality for the accumulation WIDTH.
  for (const int64_t num_tokens : {int64_t{1}, int64_t{17}, int64_t{1024}}) {
    CAPTURE(num_tokens);
    const int64_t q_size = 192, kv_size = 576;
    Rng rng(static_cast<uint64_t>(num_tokens));
    std::vector<float> q(static_cast<size_t>(num_tokens * q_size)),
        kv(static_cast<size_t>(num_tokens * kv_size)),
        qw(static_cast<size_t>(q_size)), kvw(static_cast<size_t>(kv_size));
    for (float& v : q) v = rng.Normal();
    for (float& v : kv) v = rng.Normal();
    for (float& v : qw) v = rng.Normal();
    for (float& v : kvw) v = rng.Normal();
    std::vector<float> qo(q.size()), kvo(kv.size());
    vllm::DeepseekV41FusedQKvRmsNorm(q.data(), kv.data(), qw.data(), kvw.data(),
                                     num_tokens, q_size, kv_size, 1e-6, qo.data(),
                                     kvo.data());

    // CHECK ONE — THE VALUE. `_ref_rmsnorm` (`:27-31`) ported faithfully: fp32
    // throughout, which is what `x.to(torch.float32)` and `w.to(torch.float32)`
    // say. The reduction order is NOT upstream's loop order and is not the
    // port's either: `torch.mean` reduces pairwise, so this sums pairwise. That
    // is what keeps the check from being a second spelling of the code it
    // checks — same dtype, independent summation order.
    const auto pairwise = [](const float* v, int64_t n) -> float {
      if (n == 1) return v[0] * v[0];
      std::vector<float> acc(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) acc[static_cast<size_t>(i)] = v[i] * v[i];
      for (int64_t w = n; w > 1;) {
        const int64_t half = (w + 1) / 2;
        for (int64_t i = 0; i < w / 2; ++i)
          acc[static_cast<size_t>(i)] = acc[static_cast<size_t>(2 * i)] +
                                        acc[static_cast<size_t>(2 * i + 1)];
        // An odd width leaves one element unpaired; it moves up untouched.
        if (w % 2 == 1) acc[static_cast<size_t>(half - 1)] = acc[static_cast<size_t>(w - 1)];
        w = half;
      }
      return acc[0];
    };
    const auto ref = [&](const std::vector<float>& x, const std::vector<float>& w,
                         int64_t size, const std::vector<float>& got) {
      double worst = 0.0;
      bool any_nan = false;
      for (int64_t t = 0; t < num_tokens; ++t) {
        const float* row = x.data() + t * size;
        const float rr =
            1.0F / std::sqrt(pairwise(row, size) / static_cast<float>(size) + 1e-6F);
        for (int64_t i = 0; i < size; ++i) {
          const float e = row[i] * rr * w[static_cast<size_t>(i)];
          const double g = got[static_cast<size_t>(t * size + i)];
          if (std::isnan(g)) any_nan = true;
          worst = std::max(worst, std::fabs(g - static_cast<double>(e)));
        }
      }
      // NaN-blindness is the trap a bare worst-difference walks into: an
      // all-NaN output leaves `worst` at 0 and passes. Assert BOTH.
      CHECK_FALSE(any_nan);
      // Two fp32 reductions of the same 576 squares in two different orders.
      // The band is the f32 rounding of that disagreement, not upstream's
      // bf16-sized 1e-2, and it is tight enough that a deleted stage cannot sit
      // inside it: dropping the weight, the rsqrt or the mean moves the result
      // by order 1.
      CHECK(worst < 1e-5);
    };
    ref(q, qw, q_size, qo);
    ref(kv, kvw, kv_size, kvo);

    // CHECK TWO — THE ACCUMULATION WIDTH, which check one cannot see. A port
    // that accumulates in f64 agrees with any reference BETTER, so no tolerance
    // band can detect it; only an exact comparison against the f32 result can.
    // `fused_qk_rmsnorm.py:71-73` is explicit that upstream keeps "x, rrms, and
    // w all in fp32", and the device port this reference will gate computes in
    // fp32, so the width is a mirrored behaviour and not an implementation
    // detail.
    //
    // The instrument states what it compared: it recomputes the row in BOTH
    // widths and REQUIREs that they disagree before asserting anything, so an
    // exact check that could never fail is itself a failure here.
    const auto width = [&](const std::vector<float>& x, const std::vector<float>& w,
                           int64_t size, const std::vector<float>& got) {
      int64_t discriminating_lanes = 0;
      int64_t exact_lanes = 0;
      for (int64_t t = 0; t < num_tokens; ++t) {
        const float* row = x.data() + t * size;
        float sq32 = 0.0F;
        double sq64 = 0.0;
        for (int64_t i = 0; i < size; ++i) {
          sq32 += row[i] * row[i];
          sq64 += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        const float rr32 =
            1.0F / std::sqrt(sq32 / static_cast<float>(size) + 1e-6F);
        const double rr64 =
            1.0 / std::sqrt(sq64 / static_cast<double>(size) + 1e-6);
        for (int64_t i = 0; i < size; ++i) {
          const float e32 = row[i] * rr32 * w[static_cast<size_t>(i)];
          const float e64 = static_cast<float>(static_cast<double>(row[i]) * rr64 *
                                               static_cast<double>(w[static_cast<size_t>(i)]));
          if (e32 != e64) ++discriminating_lanes;
          if (got[static_cast<size_t>(t * size + i)] == e32) ++exact_lanes;
        }
      }
      // The fixture must be able to tell the two widths apart, or the exact
      // check below is decoration.
      CAPTURE(discriminating_lanes);
      REQUIRE(discriminating_lanes > 0);
      // And the port must be on the f32 side of that difference, every lane.
      CHECK(exact_lanes == num_tokens * size);
    };
    width(q, qw, q_size, qo);
    width(kv, kvw, kv_size, kvo);

    // The two halves must not be the same computation: different widths,
    // different weights.
    CHECK(qo.size() != kvo.size());
  }
}
