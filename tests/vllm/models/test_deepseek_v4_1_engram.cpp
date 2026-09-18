// DeepSeek-V4.1-Flash W3a UNIT GATE — Engram: the n-gram hash over a compressed
// token vocabulary, the prime bucket layout, the sharded fp8 / ue8m0 table
// lookup, and the sigmoid gate that injects the lookup into the
// `[T, hc_mult, hidden]` hyper-connection manifold.
//
// Row `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`, issue
// `ISSUE-LOCAL-01M2C40RNXB871VW0E560PVBFA`, spec
// `.agents/specs/deepseek-v4-1-flash.md` (`## Work breakdown`, wave W3a).
//
// ─── WHAT IS PORTED, AND FROM WHERE ─────────────────────────────────────────
// Upstream ships a real gate for this: `tests/kernels/test_engram.py` at vLLM
// `e77daef89e`, 778 lines. Two of its helpers are scalar oracles that port
// directly and they are the spine of this file:
//
//   ReferenceHashes  <-  test_engram.py::_reference_hashes (:188-225)
//   ReferenceLookup  <-  test_engram.py::_reference_lookup (:520-529)
//
// and the cases below carry upstream's own parameters, fixtures and tolerances:
//
//   OURS                                <-  UPSTREAM (test_engram.py)
//   "engram hash cache replay"          <-  ::test_engram_hash_cache_replay
//                                           (:228-356), query_len 1/17/257 x
//                                           large_hashes False/True
//   "lookback window reproduces a
//    single instance"                   <-  ::test_lookback_window_reproduces_
//                                           single_instance (:410-477), both
//                                           runner shapes
//   "engram cache follows kv binding"   <-  ::test_engram_cache_follows_kv_
//                                           cache_binding (:154-176) +
//                                           ::test_engram_without_slot_cache_
//                                           only_gates_on_kv_binding (:179-185)
//   "engram head shards reconstruct
//    the checkpoint"                    <-  ::test_engram_head_shards_
//                                           reconstruct_checkpoint (:562-636),
//                                           head_sizes (17,19,23,29,31,37),
//                                           dim 64, tp 1/2/4/8
//   "engram lookup matches the torch
//    dequant"                           <-  ::test_engram_lookup_matches_torch
//                                           (:639-659), 4096 rows, dim 256,
//                                           block 32, tokens 1/7/256
//   "fused engram post-wkv gate"        <-  ::test_fused_engram_post_wkv_
//                                           matches_reference (:50-139) via
//                                           ::_reference_engram_post_wkv
//                                           (:18-47)
//
// HARNESS ADAPTATIONS, each unavoidable and each named:
//   * CUDA-graph capture, strided slot tensors, CPU-offload/UVA storage and the
//     `background` grid cap are torch/Triton launch concerns with no host
//     meaning. Their parameter axes collapse, and the cases they gated on the
//     device (graph replay reading the same buffers, a non-contiguous slot
//     mapping) are owed to the W5 CUDA port, not silently dropped.
//   * Tensor-parallel collectives are replaced by concatenating each rank's
//     shard, which is exactly what upstream's own monkeypatched `gather`
//     (:613-615) does.
//   * `_reference_engram_post_wkv` is re-derived in DOUBLE precision rather
//     than transcribed from torch, so the port is checked against the
//     mathematical definition and not only against a second copy of itself.
//
// ONE UPSTREAM CASE IS DROPPED, and it is named here because the heading above
// promises that every adaptation is:
//   * `::test_v2_model_state_gathers_lookback_window` (:479-517) is NOT ported.
//     It drives `DeepseekV41ModelState.prepare_inputs`, the runner-side gather
//     that BUILDS `lookback_token_ids` out of `all_token_ids` and
//     `num_computed_tokens` — a model-state surface W3a lands nothing of. What
//     it gates is the NEWEST-AT-COLUMN-0 ordering (:509-514 expects
//     `[22, 21, 16]` for a request whose history ends `..., 16, 21, 22`), which
//     is the contract `NgramHashState::Forward`'s tier 2 CONSUMES. This file
//     therefore assumes that ordering on its own fixtures and cannot prove the
//     producer agrees. Owed to W4, and recorded under `## Owed` in
//     `.agents/specs/deepseek-v4-1-flash.md`.
//
// DELIBERATE EXTENSIONS BEYOND THE PORT, both stated because they are not
// upstream's:
//   * A ue8m0 scale byte of ZERO. Upstream draws its scale bytes from
//     `randint(120, 134)` (:557, :571), so byte 0 never occurs and its own gate
//     cannot tell the BITCAST decode from the ARITHMETIC one. Byte 0 is the
//     only input where they differ by more than an exponent, and this file
//     pins it.
//   * The prime bucket layout at the REAL `engram_vocab_size` (16,000,000),
//     `n_heads` 8 and layer ids (1, 14). `build_compressed_token_map` and
//     `EngramLayout`'s prime loop are UNGATED upstream — there is no test that
//     touches either — and there is no v4_1 end-to-end model test at all. That
//     is recorded as a finding, not papered over: the layout gate below is
//     OURS and has no upstream counterpart to preserve.
//
// WHAT THE FIXTURE STANDS IN FOR, AND WHAT IT THEREFORE DOES NOT GATE. The real
// tables are 384,006,168 and 384,016,682 rows; this gate runs 4096. So it
// proves indexing, head sharding, bucket disjointness and dequant arithmetic,
// and it proves NOTHING about residency, bandwidth or the load path of a
// 384M-row table. Those are owed to W5 (device) and W8 (loader).
#include "vllm/model_executor/models/deepseek_v4_1_engram.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/model_loader/mxfp4_dequant.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/dtype.h"

using namespace vllm::deepseek_v4_1;

namespace {

uint16_t Bf16(float value) { return vt::F32ToBF16(value); }
float FromBf16(uint16_t bits) { return vt::BF16ToF32(bits); }

// `compute_hash_multipliers((1, 14), 4, 99092)`, test_engram.py:252-253, the
// `large_hashes=True` arm. DECISION 2 is that this reference READS its
// multipliers instead of reimplementing NumPy PCG64, so the upstream parameter
// is preserved by BAKING the values the pinned oracle's own generator produces
// rather than by re-deriving them here. Reproduce with, at vLLM `e77daef89e`:
//
//   bound = max(1, (np.iinfo(np.int64).max // 99092) // 2)   # 46539438283891
//   np.random.default_rng(10007 * layer_id).integers(
//       low=0, high=bound, size=(4,), dtype=np.int64) * 2 + 1
//
// NumPy's compatibility policy pins the `default_rng` + `integers` stream, so
// these are stable values and not a snapshot of one numpy build. They matter
// because they are the only fixture with REALISTIC magnitudes: a multiplier
// near the bound is what proves `value * multiplier` stays inside int64 and
// `rolling` therefore stays non-negative, which is the one condition under
// which C++ truncating `%` and Python flooring `%` agree.
const std::vector<int64_t> kUpstreamMultipliers1And14 = {
    76632096046245, 4839876093313, 35959672319349, 73987337458391,
    67716810739261, 51510806800915, 30921347202721, 82619226485591};

// ─── ReferenceHashes <- test_engram.py::_reference_hashes (:188-225) ─────────
//
// The scalar n-gram oracle with persistent, physically addressed token history.
// Upstream hardcodes block size 16 and depth 4; both are parameters here so the
// same oracle serves the `block_size` 4 case too, which is the only adaptation.
//
// It resolves ALL history through the slot cache, with no in-batch and no
// window tier. That is not a simplification of the kernel: in the fixture it
// gates, the cache is written with this very batch's tokens at these very
// slots first, so the three tiers agree by construction. Where they must NOT
// agree, the "lookback window" case below gates them against a single-instance
// run instead.
struct HashOracleTables {
  std::vector<std::vector<int32_t>> block_tables;  // [req][block]
  std::vector<int32_t> starts;                     // query_start_loc
  std::vector<int32_t> token_map;
  std::vector<int64_t> multipliers;  // [layers, max_ngram]
  std::vector<int64_t> primes;       // [layers, n_hash_cols]
  std::vector<int64_t> offsets;      // [layers, n_hash_cols]
  int64_t n_layers = 0;
  int64_t max_ngram = 4;
  int64_t n_heads = 1;
  int64_t block_size = 16;
};

std::vector<int32_t> ReferenceHashes(const std::vector<int32_t>& ids,
                                     const std::vector<int64_t>& positions,
                                     const std::vector<int64_t>& slots,
                                     const std::vector<uint8_t>& dead,
                                     const HashOracleTables& t,
                                     std::vector<int32_t>* cache) {
  const int64_t bs = t.block_size;
  const int64_t n_hash_cols = (t.max_ngram - 1) * t.n_heads;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (slots[i] >= 0) {
      (*cache)[static_cast<size_t>(slots[i])] =
          dead[i] != 0 ? -1 : t.token_map[static_cast<size_t>(ids[i])];
    }
  }
  std::vector<int32_t> result;
  result.reserve(positions.size() * static_cast<size_t>(t.n_layers * n_hash_cols));
  for (size_t i = 0; i < positions.size(); ++i) {
    // `min(bisect_right(starts[1:], i), len(tables) - 1)`, :207.
    int64_t req = 0;
    for (size_t k = 1; k < t.starts.size(); ++k) {
      if (t.starts[k] <= static_cast<int32_t>(i)) ++req;
    }
    req = std::min<int64_t>(req, static_cast<int64_t>(t.block_tables.size()) - 1);
    const std::vector<int32_t>& table = t.block_tables[static_cast<size_t>(req)];

    std::vector<int64_t> history;
    bool blocked = false;
    for (int64_t shift = 0; shift < t.max_ngram; ++shift) {
      const int64_t lookback = positions[i] - shift;
      const int64_t p = std::max<int64_t>(
          0, std::min<int64_t>(lookback,
                               static_cast<int64_t>(table.size()) * bs - 1));
      const int64_t source =
          (*cache)[static_cast<size_t>(table[static_cast<size_t>(p / bs)] * bs +
                                       p % bs)];
      blocked = blocked || lookback < 0 || source == -1;
      history.push_back(blocked ? 0 : source);
    }
    for (int64_t layer = 0; layer < t.n_layers; ++layer) {
      int64_t rolling = history[0] * t.multipliers[static_cast<size_t>(
                                        layer * t.max_ngram)];
      for (int64_t j = 1; j < t.max_ngram; ++j) {
        rolling ^= history[static_cast<size_t>(j)] *
                   t.multipliers[static_cast<size_t>(layer * t.max_ngram + j)];
        for (int64_t head = 0; head < t.n_heads; ++head) {
          const int64_t col = (j - 1) * t.n_heads + head;
          const size_t idx = static_cast<size_t>(layer * n_hash_cols + col);
          result.push_back(static_cast<int32_t>(rolling % t.primes[idx] +
                                                t.offsets[idx]));
        }
      }
    }
  }
  return result;
}

// ─── ReferenceLookup <- test_engram.py::_reference_lookup (:520-529) ─────────
//
// "The torch expression the fused kernel replaces." Rows outside
// `[start, end)` contribute zeros. The `<< 23` view-as-float32 IS the bitcast
// decode; it is reproduced here through `vllm::E8M0BitsToF32` so the test and
// the implementation cannot silently share a wrong helper.
std::vector<uint16_t> ReferenceLookup(const uint8_t* weight, const uint8_t* scales,
                                      const std::vector<int32_t>& ids,
                                      int64_t num_tokens, int64_t cols,
                                      int64_t start, int64_t end, int64_t dim,
                                      int64_t block) {
  std::vector<uint16_t> out(static_cast<size_t>(num_tokens * cols * dim), 0);
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t c = 0; c < cols; ++c) {
      const int64_t id = ids[static_cast<size_t>(t * cols + c)];
      if (id < start || id >= end) continue;
      const int64_t local = id - start;
      for (int64_t d = 0; d < dim; ++d) {
        const float value = vllm::F8E4M3ToF32(
            weight[static_cast<size_t>(local * dim + d)]);
        const float scale = vllm::E8M0BitsToF32(
            scales[static_cast<size_t>(local * (dim / block) + d / block)]);
        out[static_cast<size_t>((t * cols + c) * dim + d)] = Bf16(value * scale);
      }
    }
  }
  return out;
}

// ─── the post-wkv gate, re-derived in double precision ──────────────────────
// test_engram.py::_reference_engram_post_wkv (:18-47), written out from the
// definition rather than transcribed, and in `double` so the port is not
// compared against a second copy of its own rounding.
std::vector<uint16_t> ReferencePostWkv(const EngramGateParams& p,
                                       const std::vector<uint16_t>& hidden,
                                       const std::vector<uint16_t>& kv,
                                       const std::vector<uint16_t>& q_weight,
                                       const std::vector<uint16_t>& k_weight,
                                       const std::vector<uint8_t>* token_mask,
                                       int64_t num_tokens, int64_t num_kv_tokens) {
  const int64_t hc = p.hc_mult;
  const int64_t dim = p.dim;
  std::vector<uint16_t> out(static_cast<size_t>(num_tokens * hc * dim), 0);
  for (int64_t t = 0; t < num_tokens; ++t) {
    const bool source_valid = t < num_kv_tokens;
    for (int64_t h = 0; h < hc; ++h) {
      double hidden_sq = 0.0;
      double key_sq = 0.0;
      double dot = 0.0;
      for (int64_t d = 0; d < dim; ++d) {
        const double hv = FromBf16(hidden[static_cast<size_t>((t * hc + h) * dim + d)]);
        const double kvv =
            source_valid
                ? static_cast<double>(FromBf16(
                      kv[static_cast<size_t>(t * (hc + 1) * dim + h * dim + d)]))
                : 0.0;
        const double q = FromBf16(q_weight[static_cast<size_t>(h * dim + d)]);
        const double k = FromBf16(k_weight[static_cast<size_t>(h * dim + d)]);
        hidden_sq += hv * hv;
        key_sq += kvv * kvv;
        dot += hv * q * k * kvv;
      }
      dot *= 1.0 / std::sqrt(hidden_sq / static_cast<double>(dim) + p.eps);
      dot *= 1.0 / std::sqrt(key_sq / static_cast<double>(dim) + p.eps);
      dot *= 1.0 / std::sqrt(static_cast<double>(dim));
      double gate_input = std::sqrt(std::max(std::abs(dot),
                                             static_cast<double>(p.clamp_value)));
      if (dot < 0.0) gate_input = -gate_input;
      double gate = 1.0 / (1.0 + std::exp(-gate_input));
      if (token_mask != nullptr) {
        const bool active = source_valid && (*token_mask)[static_cast<size_t>(t)] != 0;
        if (!active) gate = 0.0;
      }
      for (int64_t d = 0; d < dim; ++d) {
        const double hv = FromBf16(hidden[static_cast<size_t>((t * hc + h) * dim + d)]);
        const double value =
            source_valid
                ? static_cast<double>(FromBf16(kv[static_cast<size_t>(
                      t * (hc + 1) * dim + hc * dim + d)]))
                : 0.0;
        out[static_cast<size_t>((t * hc + h) * dim + d)] =
            Bf16(static_cast<float>(hv + gate * value));
      }
    }
  }
  return out;
}

// A deterministic fp8-e4m3 byte that is never a NaN encoding (0x7F / 0xFF).
uint8_t Fp8Byte(std::mt19937_64* rng) {
  for (;;) {
    const uint8_t byte = static_cast<uint8_t>((*rng)() & 0xFFU);
    if ((byte & 0x7FU) != 0x7FU) return byte;
  }
}

EngramGeometry ReleasedGeometry() {
  EngramGeometry g;
  // Layer ids (1, 14), NOT (0, 1): `extract_layer_index` gates construction on
  // the real layer index (nvidia/model.py:180-181), and a first-layer id would
  // hide an off-by-one against the hash-index used to slice
  // `engram_hashes[:, layer_hash_index]` (:343, :601).
  g.layer_ids = {1, 14};
  g.num_embeddings = {384006168, 384016682};
  g.max_ngram_size = 4;
  g.n_heads = 8;
  g.head_dim = 256;
  g.vocab_size = 16000000;
  g.compressed_vocab_size = 99092;
  // The pad TOKEN, which `NgramHashState` then compresses (engram.py:422).
  g.pad_token_id = 2;
  g.hidden_size = 5120;
  g.hc_mult = 4;
  g.rms_norm_eps = 1e-20F;
  return g;
}

// A dense first-appearance-order map of the released 129,280-token vocabulary
// onto 99,092 compressed ids. The real map comes from the artifact metadata
// (`deepseek41.engram.token_map`); this one only has to satisfy the contract
// `NgramHashState.__init__:414-421` asserts.
std::vector<int32_t> SyntheticTokenMap(int64_t tokens, int64_t compressed) {
  std::vector<int32_t> map(static_cast<size_t>(tokens));
  for (int64_t i = 0; i < tokens; ++i) {
    map[static_cast<size_t>(i)] =
        static_cast<int32_t>(i < compressed ? i : i % compressed);
  }
  return map;
}

}  // namespace

// ═══ (0) the config surface ═════════════════════════════════════════════════
// NO UPSTREAM COUNTERPART, and it is ours because the claim is ours.
// `EngramGeometry`'s own header says its defaults are "named and defaulted from
// the released `deepseek-ai/DeepSeek-V4.1-Flash` `config.json` at revision
// `dba1be0a`", and nothing held them to it. Two were wrong at W3a's first head:
// `head_dim` read 128 against the artifact's 256, and `pad_token_id` read 0
// against its 2. Neither is reachable from any assertion elsewhere in this file
// — `head_dim` only flows into `EngramLayout::head_dim` and is read by no W3a
// case at all — so a silent default is exactly the kind of value a comment
// cannot hold. The artifact values below are re-derived from TWO independent
// reads of the released checkpoint, neither of them from memory:
//   * `tests/vllm/models/fixtures/deepseek_v4_1/config.json`, the `text_config`
//     block W1 checked in at `3c3c0a9bb` — that fixture lands with W1 and is
//     not on this branch yet, so read it with `git show`. It gives
//     `engram_head_dim = 256` and `engram_pad_token_id = 2` (and its top-level
//     `pad_token_id` is 2 as well, which the row spec quotes at `:626`).
//   * The safetensors header read the row spec records under `## Risks`:
//     `layers.{1,14}.engram.embed.weight` is `F8_E4M3 [384006168, 256]` and
//     `[384016682, 256]`, and `engram.wkv.weight` is `F8_E4M3 [25600, 6144]`.
//     256 is the table row, and 6144 is 24 * 256.
TEST_CASE("engram config: the defaults are the released artifact's") {
  const EngramGeometry defaults;
  CHECK(defaults.max_ngram_size == 4);     // text_config.engram_max_ngram_size
  CHECK(defaults.n_heads == 8);            // text_config.engram_n_heads
  // text_config.engram_head_dim, and also the table ROW width:
  // `engram.embed.weight` is `F8_E4M3 [384006168, 256]`.
  CHECK(defaults.head_dim == 256);
  CHECK(defaults.vocab_size == 16000000);  // text_config.engram_vocab_size
  CHECK(defaults.compressed_vocab_size == 99092);
  CHECK(defaults.pad_token_id == 2);  // text_config.engram_pad_token_id

  // The two shapes those defaults SIZE, and the reason a wrong one is not a
  // cosmetic error. `n_hash_cols * head_dim` is the `wkv` input width and
  // `(hc_mult + 1) * hidden_size` is its output width; the artifact's
  // `layers.{1,14}.engram.wkv.weight` is `F8_E4M3 [25600, 6144]`, which is
  // 5 * 5120 by 24 * 256. With `head_dim` 128 the input width reads 3072 and
  // every W4/W5/W8 buffer derived from it is half the size it must be.
  CHECK(defaults.n_hash_cols() * defaults.head_dim == 6144);
  CHECK((defaults.hc_mult + 1) * defaults.hidden_size == 25600);

  // The fixture this file runs its released-geometry cases on carries the same
  // two values, so neither can drift away from the artifact on its own.
  const EngramGeometry released = ReleasedGeometry();
  CHECK(released.head_dim == defaults.head_dim);
  CHECK(released.pad_token_id == defaults.pad_token_id);
}

// ═══ (1) the prime bucket layout ════════════════════════════════════════════
// NO UPSTREAM COUNTERPART: `EngramLayout`'s prime loop is ungated at
// `e77daef89e`. These cases are ours and they gate the three properties the
// forward actually depends on.
TEST_CASE("engram layout: prime buckets are disjoint, ordered and tiled") {
  const EngramGeometry g = ReleasedGeometry();
  const std::optional<EngramLayout> maybe = EngramLayoutFromConfig(g);
  REQUIRE(maybe.has_value());
  const EngramLayout& layout = *maybe;

  CHECK(layout.n_layers() == 2);
  CHECK(layout.n_hash_cols == 24);  // (4 - 1) * 8
  REQUIRE(layout.primes.size() == 48);
  REQUIRE(layout.offsets.size() == 48);

  // Every prime is prime, and is above `engram_vocab_size - 1`.
  for (int64_t prime : layout.primes) {
    CHECK(IsEngramPrime(prime));
    CHECK(prime >= g.vocab_size);
  }
  // The 48 primes are pairwise distinct: `seen` is threaded across BOTH layers
  // and every n-gram group (engram.py:189-197), which is what keeps the bucket
  // ranges disjoint.
  std::vector<int64_t> sorted = layout.primes;
  std::sort(sorted.begin(), sorted.end());
  CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());

  // The search restarts at `vocab_size - 1` for every n-gram group (:193), and
  // that restart is INERT: `find_next_prime` walks UPWARD and skips everything
  // in `seen`, so it climbs back past every prime already taken. All 48
  // columns therefore come out strictly increasing ACROSS the groups and
  // across both layers, not only within a group. Gate the global shape, since
  // it is the one the tree actually has.
  for (size_t idx = 1; idx < layout.primes.size(); ++idx) {
    CHECK(layout.primes[idx] > layout.primes[idx - 1]);
  }

  // Offsets are the exclusive prefix sum WITHIN a layer (engram.py:203), so
  // layer L's buckets tile [0, sum(primes[L])) with no gap and no overlap.
  for (int64_t layer = 0; layer < layout.n_layers(); ++layer) {
    int64_t running = 0;
    for (int64_t col = 0; col < layout.n_hash_cols; ++col) {
      const size_t idx = static_cast<size_t>(layer * layout.n_hash_cols + col);
      CHECK(layout.offsets[idx] == running);
      running += layout.primes[idx];
    }
    // The layer's table must be able to hold every bucket.
    CHECK(running <= layout.num_embeddings[static_cast<size_t>(layer)]);
  }

  // `HeadSizes` is what the embedding shard is built from (engram.py:913).
  const std::vector<int64_t> sizes = layout.HeadSizes(0);
  REQUIRE(sizes.size() == 24);
  for (int64_t col = 0; col < 24; ++col) {
    CHECK(sizes[static_cast<size_t>(col)] == layout.primes[static_cast<size_t>(col)]);
  }
}

TEST_CASE("engram layout: empty layer ids is the feature switch") {
  // `from_config:206-210` returns None on an empty list; `vllm/config/engram.py`
  // is only `cpu_offload` and `embedding_across_dp` and switches NOTHING.
  EngramGeometry g = ReleasedGeometry();
  g.layer_ids.clear();
  g.num_embeddings.clear();
  CHECK_FALSE(EngramLayoutFromConfig(g).has_value());

  // engram.py:186 `assert len(self.layer_ids) == len(self.num_embeddings)`.
  EngramGeometry mismatched = ReleasedGeometry();
  mismatched.num_embeddings.pop_back();
  CHECK_THROWS(EngramLayoutFromConfig(mismatched));
}

TEST_CASE("engram layout: miller-rabin agrees with trial division") {
  // The bases (2, 7, 61) are exact below 2^32; the small-factor pre-screen at
  // engram.py:67-69 must return `n == p`, so 2..37 are primes and not rejected.
  auto trial = [](int64_t n) {
    if (n < 2) return false;
    for (int64_t d = 2; d * d <= n; ++d) {
      if (n % d == 0) return false;
    }
    return true;
  };
  for (int64_t n = 0; n < 2000; ++n) {
    // 61 is the ONE disagreement, and it is upstream's. `pow(61, d, 61)` is
    // zero, so the witness 61 reports 61 composite. Executing engram.py:63-85
    // at `e77daef89e` gives `_is_prime(61) == False`, and 61 is the only n
    // below 2000 where it differs from trial division. We MIRROR it: a port
    // that quietly disagrees with its oracle on one input has not proved
    // agreement on the others.
    if (n == 61) {
      CHECK_FALSE(IsEngramPrime(61));
      CHECK(trial(61));
      continue;
    }
    CHECK(IsEngramPrime(n) == trial(n));
  }
  for (int64_t n = 15999900; n < 16000100; ++n) CHECK(IsEngramPrime(n) == trial(n));
  // The screen must return `n == p` (engram.py:69), so the twelve screening
  // primes are primes and not rejected by their own divisor.
  for (int64_t p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    CHECK(IsEngramPrime(p));
  }
}

// ═══ (2) the two constants this reference consumes ══════════════════════════
TEST_CASE("engram constants: the multiplier bound is what keeps rolling non-negative") {
  // engram.py:155. The two vocabulary sizes are UNRELATED and swapping them
  // does not raise; it rehashes the tables. 16,000,000 seeds the prime search,
  // 99,092 bounds the multiplier.
  const int64_t bound = HashMultiplierBound(99092);
  CHECK(bound == 46539438283891);
  CHECK(HashMultiplierBound(16000000) != bound);
  // A compressed vocabulary larger than INT64_MAX/2 still yields a usable 1.
  CHECK(HashMultiplierBound(INT64_MAX) == 1);

  ValidateHashMultipliers(kUpstreamMultipliers1And14, 2, 4, 99092);
  for (int64_t m : kUpstreamMultipliers1And14) {
    CHECK(m % 2 == 1);
    CHECK(m < 2 * bound + 1);
    // The property the bound exists for: the largest compressed id times the
    // multiplier stays inside int64, so `rolling` never goes negative and C++
    // truncating `%` agrees with the Python oracle's flooring `%`.
    CHECK(m <= INT64_MAX / 99091);
  }

  std::vector<int64_t> even = kUpstreamMultipliers1And14;
  even[3] += 1;
  CHECK_THROWS(ValidateHashMultipliers(even, 2, 4, 99092));
  std::vector<int64_t> overflowing = kUpstreamMultipliers1And14;
  overflowing[2] = INT64_MAX;
  CHECK_THROWS(ValidateHashMultipliers(overflowing, 2, 4, 99092));
  CHECK_THROWS(ValidateHashMultipliers(kUpstreamMultipliers1And14, 3, 4, 99092));
}

TEST_CASE("engram constants: the compressed token map is validated, not derived") {
  // DECISION 1: the map comes from the artifact. What this reference owes is
  // the check `NgramHashState.__init__:415-421` performs, whose failure mode
  // upstream states in its own message: the tables are silently rehashed.
  const std::vector<int32_t> map = SyntheticTokenMap(129280, 99092);
  CHECK(ValidateCompressedTokenMap(map, 99092) == 99092);
  CHECK_THROWS(ValidateCompressedTokenMap(map, 99091));

  // A map with a hole is not what `build_compressed_token_map` can produce: it
  // assigns `len(key_to_new)` in first-appearance order (engram.py:138-142), so
  // the emitted ids are exactly [0, size).
  std::vector<int32_t> holed = map;
  for (int32_t& id : holed) {
    if (id == 5) id = 6;
  }
  CHECK_THROWS(ValidateCompressedTokenMap(holed, 99092));
  std::vector<int32_t> negative = map;
  negative[17] = -1;
  CHECK_THROWS(ValidateCompressedTokenMap(negative, 99092));
}

// ═══ (3) the n-gram hash ════════════════════════════════════════════════════

TEST_CASE("engram hash: a compressed id above the bound is refused") {
  // The one map property the HASH depends on, as opposed to the stronger
  // checkpoint property the validator above enforces: every compressed id is
  // below `engram_compressed_vocab_size`, because the multiplier bound is
  // derived from that size. One id above it is enough to overflow
  // `value * multiplier`, and the overflow is silent — `rolling` goes negative
  // and C++ truncating `%` stops agreeing with the Python oracle's flooring
  // `%`, so the hash diverges without raising anywhere.
  const EngramLayout layout = *EngramLayoutFromConfig(ReleasedGeometry());
  std::vector<int32_t> map = SyntheticTokenMap(129280, 99092);
  map[9] = 99092;
  CHECK_THROWS(NgramHashState(layout, map, kUpstreamMultipliers1And14, 64, true));
}

TEST_CASE("engram hash: the pad is the COMPRESSED id of the pad token") {
  // NOT AN UPSTREAM CASE. engram.py:422 is
  // `self.pad_id = token_map[layout.pad_token_id]` — the pad that fills a
  // BLOCKED lookback is the pad token's COMPRESSED id, not the token id. Every
  // upstream fixture has `token_map[pad_token_id] == pad_token_id` (its maps
  // send 0 to 0 and its pad is 0, test_engram.py:283, :387), so replacing the
  // lookup with the raw token id survives the entire ported suite. Here they
  // differ, so the substitution cannot.
  EngramLayout layout;
  layout.layer_ids = {1};
  layout.num_embeddings = {1024};
  layout.max_ngram_size = 2;
  layout.n_heads = 1;
  layout.head_dim = 8;
  layout.compressed_vocab_size = 16;
  layout.pad_token_id = 3;
  layout.n_hash_cols = 1;
  layout.primes = {101};
  layout.offsets = {0};
  // token_map[3] = 7, so the pad id is 7 and the pad TOKEN is 3.
  const std::vector<int32_t> token_map = {1, 5, 9, 7, 2, 4, 6, 8};
  const std::vector<int64_t> multipliers = {3, 5};

  // One request, one token at position 0. Shift 0 is in batch and gives
  // token_map[2] = 9; shift 1 has lookback -1, which is BLOCKED, so its value
  // is the pad id. Therefore
  //   rolling = (9 * 3) XOR (pad_id * 5) = 27 XOR 35 = 56,  hash = 56 % 101 + 0
  // with the correct pad id 7, and 27 XOR 15 = 20 with the raw pad token 3.
  NgramHashState state(layout, token_map, multipliers, 16, false);
  CHECK(state.pad_id() == 7);
  REQUIRE(state.EnsureCache(4));

  const std::vector<int32_t> input_ids = {2};
  const std::vector<int64_t> positions = {0};
  const std::vector<int32_t> starts = {0, 1};
  const std::vector<uint8_t> dead = {0};
  const std::vector<int32_t> window = {-1};
  const std::vector<uint8_t> window_dead = {0};
  NgramHashInputs in;
  in.input_ids = input_ids.data();
  in.positions = positions.data();
  in.query_start_loc = starts.data();
  in.dead_mask = dead.data();
  in.lookback_token_ids = window.data();
  in.lookback_dead_mask = window_dead.data();
  in.num_tokens = 1;
  in.num_reqs = 1;
  in.lookback_depth = 1;
  int32_t out = -1;
  state.Forward(in, &out);
  CHECK(out == 56);
}

TEST_CASE("engram hash: the cache follows the kv cache binding") {
  // <- test_engram.py::test_engram_cache_follows_kv_cache_binding (:154-176),
  // num_blocks 4 and 100.
  EngramGeometry g = ReleasedGeometry();
  const EngramLayout layout = *EngramLayoutFromConfig(g);
  const std::vector<int32_t> map = SyntheticTokenMap(129280, 99092);

  for (int64_t num_blocks : {int64_t{4}, int64_t{100}}) {
    NgramHashState state(layout, map, kUpstreamMultipliers1And14, 64, true);
    CHECK_FALSE(state.EnsureCache(0));

    CHECK(state.EnsureCache(4));
    CHECK(state.cache().size() == 4 * 64);
    // Rebinding the SAME shape keeps the history.
    CHECK(state.EnsureCache(4));

    CHECK(state.EnsureCache(num_blocks));
    CHECK(state.cache().size() == static_cast<size_t>(num_blocks * 64));
    // "Graph memory profiling binds a temporary, smaller KV cache first.
    // Rebinding must discard its hash history" (engram.py:455-456).
    for (int32_t v : state.cache()) CHECK(v == 0);

    CHECK_FALSE(state.EnsureCache(0));
    CHECK(state.cache().empty());
  }
}

TEST_CASE("engram hash: without a slot cache only the kv binding is gated") {
  // <- ::test_engram_without_slot_cache_only_gates_on_kv_binding (:179-185).
  // "The V2 runner supplies every lookback, so no slot cache is allocated."
  const EngramLayout layout = *EngramLayoutFromConfig(ReleasedGeometry());
  NgramHashState state(layout, SyntheticTokenMap(129280, 99092),
                       kUpstreamMultipliers1And14, 64, false);
  CHECK_FALSE(state.EnsureCache(0));
  CHECK(state.EnsureCache(4));
  CHECK(state.cache().empty());
}

TEST_CASE("engram hash: cache replay preserves history, dead tokens and padding") {
  // <- ::test_engram_hash_cache_replay (:228-356). query_len 1/17/257 and
  // large_hashes False/True are upstream's parameters and are preserved; the
  // `capture` and `strided_slots` axes are launch concerns with no host meaning
  // and are owed to the W5 CUDA port.
  for (int64_t query_len : {int64_t{1}, int64_t{17}, int64_t{257}}) {
    for (bool large_hashes : {false, true}) {
      CAPTURE(query_len);
      CAPTURE(large_hashes);
      const int64_t num_tokens = 2 * query_len + 3;
      const int64_t blocks = (3 * query_len + 15) / 16;
      std::vector<std::vector<int32_t>> tables(2);
      for (int64_t b = 0; b < blocks; ++b) {
        tables[0].push_back(static_cast<int32_t>(blocks + b));
        tables[1].push_back(static_cast<int32_t>(b));
      }
      const std::vector<int32_t> starts = {0, static_cast<int32_t>(query_len),
                                           static_cast<int32_t>(2 * query_len)};

      std::vector<int32_t> token_map(32);
      for (int64_t i = 0; i < 32; ++i) {
        token_map[static_cast<size_t>(i)] =
            static_cast<int32_t>(large_hashes ? i * 3091 : i % 7);
      }
      const std::vector<int64_t> multipliers =
          large_hashes ? kUpstreamMultipliers1And14
                       : std::vector<int64_t>{3, 5, 7, 9, 11, 13, 15, 17};
      const int64_t n_heads = large_hashes ? 8 : 1;
      const int64_t n_hash_cols = 3 * n_heads;
      std::vector<int64_t> primes;
      for (int64_t layer = 0; layer < 2; ++layer) {
        for (int64_t j = 0; j < 3; ++j) {
          for (int64_t h = 0; h < n_heads; ++h) {
            primes.push_back(large_hashes
                                 ? 16000057 + 2 * (layer * 24 + j * 8 + h)
                                 : (j == 0 ? 19 : (j == 1 ? 23 : 29)));
          }
        }
      }
      std::vector<int64_t> offsets(primes.size());
      for (int64_t layer = 0; layer < 2; ++layer) {
        int64_t running = 0;
        for (int64_t col = 0; col < n_hash_cols; ++col) {
          const size_t idx = static_cast<size_t>(layer * n_hash_cols + col);
          offsets[idx] = running;
          running += primes[idx];
        }
      }

      // A hand-built layout: this case deliberately does NOT use the released
      // prime search, because it is the HASH it gates, not the layout.
      EngramLayout layout;
      layout.layer_ids = {1, 14};
      layout.num_embeddings = {1 << 30, 1 << 30};
      layout.max_ngram_size = 4;
      layout.n_heads = n_heads;
      layout.head_dim = 8;
      // The `large_hashes` map is SPARSE by construction (`i * 3091`,
      // test_engram.py:250) — that is how upstream reaches realistic hash
      // magnitudes. It is not a map `build_compressed_token_map` could produce,
      // so it never passes `ValidateCompressedTokenMap`; the constructor
      // therefore checks only the bound the hash needs, and the fixture keeps
      // the real 99092 so the bound is the real one.
      layout.compressed_vocab_size = large_hashes ? 99092 : 7;
      layout.pad_token_id = 0;
      layout.n_hash_cols = n_hash_cols;
      layout.primes = primes;
      layout.offsets = offsets;

      NgramHashState state(layout, token_map, multipliers, 16, true);
      // Upstream seeds the slot store with -1 (:249, :309), which is DEAD, so
      // an unwritten slot blocks the n-gram instead of reading as compressed id
      // zero. Bind that buffer rather than allocate a zeroed one.
      state.BindCache(std::vector<int32_t>(
          static_cast<size_t>(2 * blocks * 16), -1));
      std::vector<int32_t> expected_cache(static_cast<size_t>(2 * blocks * 16), -1);

      HashOracleTables oracle;
      oracle.block_tables = tables;
      oracle.starts = starts;
      oracle.token_map = token_map;
      oracle.multipliers = multipliers;
      oracle.primes = primes;
      oracle.offsets = offsets;
      oracle.n_layers = 2;
      oracle.max_ngram = 4;
      oracle.n_heads = n_heads;
      oracle.block_size = 16;

      const std::vector<int32_t> lookback(2 * 3, -1);
      const std::vector<uint8_t> lookback_dead(2 * 3, 0);
      std::vector<int32_t> block_table_flat;
      for (const std::vector<int32_t>& row : tables) {
        block_table_flat.insert(block_table_flat.end(), row.begin(), row.end());
      }

      for (int64_t step = 0; step < 3; ++step) {
        std::vector<int32_t> ids(static_cast<size_t>(num_tokens));
        std::vector<int64_t> pos;
        std::vector<int64_t> slots;
        std::vector<uint8_t> dead(static_cast<size_t>(num_tokens));
        for (int64_t i = 0; i < num_tokens; ++i) {
          ids[static_cast<size_t>(i)] = static_cast<int32_t>((i + 3 * step) % 32);
          dead[static_cast<size_t>(i)] = (i + step) % 5 == 0 ? 1 : 0;
        }
        for (int64_t rep = 0; rep < 2; ++rep) {
          for (int64_t p = step * query_len; p < (step + 1) * query_len; ++p) {
            pos.push_back(p);
          }
        }
        for (size_t i = 0; i < pos.size(); ++i) {
          const int64_t p = pos[i];
          slots.push_back(tables[i / static_cast<size_t>(query_len)]
                                [static_cast<size_t>(p / 16)] *
                              16 +
                          p % 16);
        }
        for (int64_t k = 0; k < 3; ++k) {
          pos.push_back(0);
          slots.push_back(-1);
        }
        // "A fully padded replay must leave every cache slot untouched."
        if (step == 2) slots.assign(static_cast<size_t>(num_tokens), -1);

        NgramHashInputs in;
        in.input_ids = ids.data();
        in.positions = pos.data();
        in.query_start_loc = starts.data();
        in.dead_mask = dead.data();
        in.lookback_token_ids = lookback.data();
        in.lookback_dead_mask = lookback_dead.data();
        in.slot_mapping = slots.data();
        in.block_table = block_table_flat.data();
        in.num_tokens = num_tokens;
        in.num_reqs = 2;
        in.num_block_table_rows = 2;
        in.max_blocks = blocks;
        in.lookback_depth = 3;

        std::vector<int32_t> out(
            static_cast<size_t>(num_tokens * 2 * n_hash_cols));
        state.Forward(in, out.data());

        const std::vector<int32_t> expected =
            ReferenceHashes(ids, pos, slots, dead, oracle, &expected_cache);
        if (step != 2) {
          const size_t real =
              static_cast<size_t>(2 * query_len * 2 * n_hash_cols);
          for (size_t i = 0; i < real; ++i) {
            // rtol=0, atol=0 (test_engram.py:347-349).
            REQUIRE(out[i] == expected[i]);
          }
        }
        REQUIRE(state.cache() == expected_cache);
      }

      // An empty batch writes nothing and touches no slot (:352-356).
      NgramHashInputs empty;
      empty.input_ids = nullptr;
      empty.positions = nullptr;
      empty.query_start_loc = starts.data();
      empty.dead_mask = nullptr;
      empty.lookback_token_ids = lookback.data();
      empty.lookback_dead_mask = lookback_dead.data();
      empty.slot_mapping = nullptr;
      empty.block_table = block_table_flat.data();
      empty.num_tokens = 0;
      empty.num_reqs = 2;
      empty.num_block_table_rows = 2;
      empty.max_blocks = blocks;
      empty.lookback_depth = 3;
      state.Forward(empty, nullptr);
      CHECK(state.cache() == expected_cache);
    }
  }
}

namespace {

// <- test_engram.py::_hash_ids (:359-407): run the op on a batch of
// (token_ids, start, window) requests and return one slice per request.
// Token ids 14 and 22 stand in for image tokens that break n-grams.
struct HashRequest {
  std::vector<int32_t> token_ids;
  int64_t start = 0;
  std::vector<int32_t> window;  // empty = all unknown
};

std::vector<std::vector<int32_t>> RunHashIds(
    const std::vector<HashRequest>& requests, const EngramLayout& layout,
    const std::vector<int32_t>& token_map, const std::vector<int64_t>& multipliers,
    const std::vector<int32_t>& block_table, int64_t max_blocks, int64_t block_size,
    std::vector<int32_t>* cache) {
  const int64_t depth = layout.max_ngram_size - 1;
  std::vector<int32_t> input_ids;
  std::vector<int64_t> positions;
  std::vector<int64_t> slots;
  std::vector<int32_t> windows;
  std::vector<int32_t> starts = {0};
  for (size_t req = 0; req < requests.size(); ++req) {
    const HashRequest& r = requests[req];
    for (size_t i = 0; i < r.token_ids.size(); ++i) {
      const int64_t p = r.start + static_cast<int64_t>(i);
      input_ids.push_back(r.token_ids[i]);
      positions.push_back(p);
      slots.push_back(
          block_table[req * static_cast<size_t>(max_blocks) +
                      static_cast<size_t>(p / block_size)] *
              block_size +
          p % block_size);
    }
    if (r.window.empty()) {
      for (int64_t j = 0; j < depth; ++j) windows.push_back(-1);
    } else {
      windows.insert(windows.end(), r.window.begin(), r.window.end());
    }
    starts.push_back(starts.back() + static_cast<int32_t>(r.token_ids.size()));
  }
  std::vector<uint8_t> dead(input_ids.size());
  for (size_t i = 0; i < input_ids.size(); ++i) {
    dead[i] = (input_ids[i] == 14 || input_ids[i] == 22) ? 1 : 0;
  }
  std::vector<uint8_t> window_dead(windows.size());
  for (size_t i = 0; i < windows.size(); ++i) {
    window_dead[i] = (windows[i] == 14 || windows[i] == 22) ? 1 : 0;
  }

  // A FRESH state per call, exactly like upstream's `_hash_ids` (:382), but
  // bound to the SAME slot store across chunks — which is the whole point of
  // the V1 arm: the history a previous chunk wrote is the only place the
  // generated positions live.
  NgramHashState state(layout, token_map, multipliers, block_size, cache != nullptr);
  if (cache != nullptr) state.BindCache(*cache);

  NgramHashInputs in;
  in.input_ids = input_ids.data();
  in.positions = positions.data();
  in.query_start_loc = starts.data();
  in.dead_mask = dead.data();
  in.lookback_token_ids = windows.data();
  in.lookback_dead_mask = window_dead.data();
  in.slot_mapping = cache != nullptr ? slots.data() : nullptr;
  in.block_table = cache != nullptr ? block_table.data() : nullptr;
  in.num_tokens = static_cast<int64_t>(input_ids.size());
  in.num_reqs = static_cast<int64_t>(requests.size());
  in.num_block_table_rows = static_cast<int64_t>(requests.size());
  in.max_blocks = max_blocks;
  in.lookback_depth = depth;

  const int64_t width = layout.n_layers() * layout.n_hash_cols;
  std::vector<int32_t> out(input_ids.size() * static_cast<size_t>(width));
  state.Forward(in, out.data());
  if (cache != nullptr) *cache = state.cache();

  std::vector<std::vector<int32_t>> per_request;
  for (size_t req = 0; req + 1 < starts.size(); ++req) {
    per_request.emplace_back(
        out.begin() + starts[req] * width,
        out.begin() + starts[req + 1] * width);
  }
  return per_request;
}

}  // namespace

TEST_CASE("engram hash: the lookback window reproduces a single instance") {
  // <- ::test_lookback_window_reproduces_single_instance (:410-477).
  // "Decoding on a fresh instance whose prompt KV came from elsewhere (P/D,
  // offload) must hash like the instance that processed the whole sequence."
  const int64_t block_size = 4;
  const std::vector<int32_t> block_table = {3, 1, 5, 0, 2, 4, 6, 7};
  const int64_t max_blocks = 4;

  std::vector<int32_t> token_map(50);
  for (int32_t i = 0; i < 50; ++i) token_map[static_cast<size_t>(i)] = i;

  EngramLayout layout;
  layout.layer_ids = {1};
  layout.num_embeddings = {1 << 20};
  layout.max_ngram_size = 4;
  layout.n_heads = 2;  // >= 2 on purpose: one head cannot see a head-index bug.
  layout.head_dim = 8;
  layout.compressed_vocab_size = 50;
  layout.pad_token_id = 0;
  layout.n_hash_cols = 6;
  layout.primes = {97, 89, 83, 79, 73, 71};
  layout.offsets = {0, 97, 186, 269, 348, 421};
  const std::vector<int64_t> multipliers = {3, 5, 7, 9};

  const std::vector<std::vector<int32_t>> prompts = {{11, 12, 13, 14, 15, 16},
                                                     {31, 32, 33, 34, 35}};
  const std::vector<std::vector<int32_t>> decodes = {{21, 22, 23, 24},
                                                     {41, 42, 43, 44}};
  const std::vector<int64_t> chunk_sizes = {1, 2, 1};
  const int64_t depth = 3;

  auto fresh_cache = [&]() { return std::vector<int32_t>(8 * block_size, 0); };

  std::vector<HashRequest> whole;
  for (size_t i = 0; i < prompts.size(); ++i) {
    HashRequest r;
    r.token_ids = prompts[i];
    r.token_ids.insert(r.token_ids.end(), decodes[i].begin(), decodes[i].end());
    r.start = 0;
    whole.push_back(r);
  }
  std::vector<int32_t> reference_cache = fresh_cache();
  const std::vector<std::vector<int32_t>> reference =
      RunHashIds(whole, layout, token_map, multipliers, block_table, max_blocks,
                 block_size, &reference_cache);

  for (bool v2 : {true, false}) {
    CAPTURE(v2);
    std::vector<std::vector<int32_t>> histories = prompts;
    std::vector<int32_t> cache = fresh_cache();
    int64_t consumed = 0;
    const int64_t width = layout.n_layers() * layout.n_hash_cols;
    for (int64_t chunk : chunk_sizes) {
      std::vector<HashRequest> batch;
      for (size_t i = 0; i < histories.size(); ++i) {
        HashRequest r;
        const int64_t start = static_cast<int64_t>(histories[i].size());
        for (int64_t k = 0; k < chunk; ++k) {
          r.token_ids.push_back(decodes[i][static_cast<size_t>(consumed + k)]);
        }
        r.start = start;
        for (int64_t j = 0; j < depth; ++j) {
          const int64_t idx = start - 1 - j;
          // V2 supplies every lookback from its device token history; V1
          // supplies PROMPT positions only and reads generated positions from
          // the slot cache it fills itself.
          const bool known =
              idx >= 0 && (v2 || idx < static_cast<int64_t>(prompts[i].size()));
          r.window.push_back(known ? histories[i][static_cast<size_t>(idx)] : -1);
        }
        batch.push_back(r);
      }
      const std::vector<std::vector<int32_t>> got =
          RunHashIds(batch, layout, token_map, multipliers, block_table,
                     max_blocks, block_size, v2 ? nullptr : &cache);
      for (size_t req = 0; req < batch.size(); ++req) {
        const int64_t start = batch[req].start;
        for (int64_t k = 0; k < chunk * width; ++k) {
          REQUIRE(got[req][static_cast<size_t>(k)] ==
                  reference[req][static_cast<size_t>(start * width + k)]);
        }
        histories[req].insert(histories[req].end(), batch[req].token_ids.begin(),
                              batch[req].token_ids.end());
      }
      consumed += chunk;
    }
  }

  // The negative control (:471-477): without any window the first decode token
  // hashes stale slots and must NOT match.
  std::vector<HashRequest> unseeded_req(1);
  unseeded_req[0].token_ids = {decodes[0][0]};
  unseeded_req[0].start = static_cast<int64_t>(prompts[0].size());
  std::vector<int32_t> unseeded_cache = fresh_cache();
  const std::vector<std::vector<int32_t>> unseeded =
      RunHashIds(unseeded_req, layout, token_map, multipliers, block_table,
                 max_blocks, block_size, &unseeded_cache);
  const int64_t width = layout.n_layers() * layout.n_hash_cols;
  bool identical = true;
  for (int64_t k = 0; k < width; ++k) {
    if (unseeded[0][static_cast<size_t>(k)] !=
        reference[0][static_cast<size_t>(
            static_cast<int64_t>(prompts[0].size()) * width + k)]) {
      identical = false;
    }
  }
  CHECK_FALSE(identical);
}

// ═══ (4) the sharded fp8 / ue8m0 lookup ═════════════════════════════════════

TEST_CASE("engram lookup: head shards reconstruct the checkpoint") {
  // <- ::test_engram_head_shards_reconstruct_checkpoint (:562-636).
  // head_sizes and dim are upstream's; tp_size 4 and 8 do NOT divide the six
  // head count, which is the whole point of the case — a tp_size that divides
  // it cannot produce a padded head or an empty owner.
  const std::vector<int64_t> head_sizes = {17, 19, 23, 29, 31, 37};
  const int64_t dim = 64;
  const int64_t block = 32;
  int64_t num_rows = 0;
  for (int64_t s : head_sizes) num_rows += s;

  std::mt19937_64 rng(0);
  const int64_t loaded_rows = num_rows + 7;
  std::vector<uint8_t> weight(static_cast<size_t>(loaded_rows * dim));
  std::vector<uint8_t> scales(static_cast<size_t>(loaded_rows * (dim / block)));
  for (uint8_t& b : weight) b = Fp8Byte(&rng);
  for (uint8_t& b : scales) b = static_cast<uint8_t>(120 + rng() % 14);

  // Seven token rows, each column drawn inside its own bucket, with the first
  // and last rows pinned to the bucket's first and last id (:574-578).
  const int64_t num_tokens = 7;
  const int64_t cols = static_cast<int64_t>(head_sizes.size());
  std::vector<int32_t> ids(static_cast<size_t>(num_tokens * cols));
  int64_t start = 0;
  for (int64_t head = 0; head < cols; ++head) {
    const int64_t size = head_sizes[static_cast<size_t>(head)];
    for (int64_t t = 0; t < num_tokens; ++t) {
      ids[static_cast<size_t>(t * cols + head)] =
          static_cast<int32_t>(start + static_cast<int64_t>(rng() % static_cast<uint64_t>(size)));
    }
    ids[static_cast<size_t>(head)] = static_cast<int32_t>(start);
    ids[static_cast<size_t>((num_tokens - 1) * cols + head)] =
        static_cast<int32_t>(start + size - 1);
    start += size;
  }

  const std::vector<uint16_t> expected = ReferenceLookup(
      weight.data(), scales.data(), ids, num_tokens, cols, 0, num_rows, dim, block);

  for (int64_t tp_size : {int64_t{1}, int64_t{2}, int64_t{4}, int64_t{8}}) {
    CAPTURE(tp_size);
    std::vector<std::vector<uint16_t>> shards;
    int64_t total_owned = 0;
    int64_t part_cols = 0;
    for (int64_t rank = 0; rank < tp_size; ++rank) {
      const EngramEmbeddingShard shard =
          MakeEngramEmbeddingShard(loaded_rows, dim, head_sizes, tp_size, rank);
      part_cols = shard.part_n_hash_cols;
      total_owned += shard.part_num_embeddings;
      const std::vector<uint8_t> w =
          LoadEngramHeadShard(shard, weight.data(), loaded_rows, dim);
      const std::vector<uint8_t> s =
          LoadEngramHeadShard(shard, scales.data(), loaded_rows, dim / block);
      std::vector<uint16_t> out(
          static_cast<size_t>(num_tokens * shard.part_n_hash_cols * dim), 0xFFFFU);
      EngramLookup(shard, w.data(), s.data(), ids.data(), num_tokens, out.data());
      shards.push_back(out);
    }
    // "sum(layer.part_num_embeddings for layer in layers) == num_rows" (:611):
    // every bucket is owned exactly once, and the seven spare rows of the
    // loaded tensor belong to nobody.
    CHECK(total_owned == num_rows);

    // Concatenate the ranks on the head axis, which is what upstream's
    // monkeypatched all-gather does (:613-615).
    const int64_t gathered_cols = tp_size * part_cols;
    std::vector<uint16_t> gathered(
        static_cast<size_t>(num_tokens * gathered_cols * dim));
    for (int64_t t = 0; t < num_tokens; ++t) {
      for (int64_t rank = 0; rank < tp_size; ++rank) {
        std::memcpy(&gathered[static_cast<size_t>(
                        (t * gathered_cols + rank * part_cols) * dim)],
                    &shards[static_cast<size_t>(rank)][static_cast<size_t>(
                        t * part_cols * dim)],
                    static_cast<size_t>(part_cols * dim) * sizeof(uint16_t));
      }
    }
    for (int64_t t = 0; t < num_tokens; ++t) {
      for (int64_t c = 0; c < cols; ++c) {
        for (int64_t d = 0; d < dim; ++d) {
          // rtol=0, atol=0 (:609).
          REQUIRE(gathered[static_cast<size_t>((t * gathered_cols + c) * dim + d)] ==
                  expected[static_cast<size_t>((t * cols + c) * dim + d)]);
        }
      }
      // "assert torch.count_nonzero(gathered[:, len(head_sizes):]) == 0" (:610).
      for (int64_t c = cols; c < gathered_cols; ++c) {
        for (int64_t d = 0; d < dim; ++d) {
          REQUIRE(gathered[static_cast<size_t>((t * gathered_cols + c) * dim + d)] ==
                  0);
        }
      }
    }
  }
}

TEST_CASE("engram lookup: matches the torch dequant expression it replaces") {
  // <- ::test_engram_lookup_matches_torch (:639-659). 4096 rows, dim 256,
  // block 32, 24 columns, num_tokens 1/7/256, and a vocabulary window strictly
  // inside the table so unowned rows are exercised (:540-541).
  const int64_t rows = 4096;
  const int64_t dim = 256;
  const int64_t block = 32;
  const int64_t cols = 24;

  EngramEmbeddingShard shard;
  shard.dim = dim;
  shard.quant_block = block;
  shard.n_hash_cols = cols;
  shard.part_n_hash_cols = cols;
  shard.head_start = 0;
  shard.tp_size = 1;
  shard.vocab_start_idx = rows / 4;
  shard.vocab_end_idx = rows / 4 + rows / 2;
  shard.part_num_embeddings = shard.vocab_end_idx - shard.vocab_start_idx;

  std::mt19937_64 rng(0);
  const int64_t owned = shard.part_num_embeddings;
  std::vector<uint8_t> weight(static_cast<size_t>(owned * dim));
  std::vector<uint8_t> scales(static_cast<size_t>(owned * (dim / block)));
  for (uint8_t& b : weight) b = Fp8Byte(&rng);
  for (uint8_t& b : scales) b = static_cast<uint8_t>(120 + rng() % 14);

  for (int64_t num_tokens : {int64_t{1}, int64_t{7}, int64_t{256}}) {
    CAPTURE(num_tokens);
    std::vector<int32_t> ids(static_cast<size_t>(num_tokens * cols));
    for (int32_t& id : ids) id = static_cast<int32_t>(rng() % static_cast<uint64_t>(rows));
    std::vector<uint16_t> out(static_cast<size_t>(num_tokens * cols * dim), 0xFFFFU);
    EngramLookup(shard, weight.data(), scales.data(), ids.data(), num_tokens,
                 out.data());
    const std::vector<uint16_t> expected =
        ReferenceLookup(weight.data(), scales.data(), ids, num_tokens, cols,
                        shard.vocab_start_idx, shard.vocab_end_idx, dim, block);
    REQUIRE(out == expected);  // torch.equal (:659)
  }
}

TEST_CASE("engram lookup: a padded head on a rank that owns rows writes zeros") {
  // NOT AN UPSTREAM CASE, and upstream's fixture CANNOT reach it. With six
  // heads and tp_size 1/2/4/8 (:564, :566), every rank that owns a padded head
  // owns NO rows at all, so its empty vocabulary window masks the row out even
  // if the head-count guard were removed. Mutating
  // `head < shard.n_hash_cols` to `true` survives the whole ported suite.
  //
  // FIVE heads over two ranks is the shape that separates them: rank 1 owns
  // heads 3 and 4 (real, 60 rows) AND head 5 (padded). Without the guard, head
  // 5 indexes `ids` past the end of its row and can emit a non-zero vector into
  // a column the all-gather then keeps.
  const std::vector<int64_t> head_sizes = {17, 19, 23, 29, 31};
  const int64_t dim = 64;
  const int64_t block = 32;
  const int64_t cols = static_cast<int64_t>(head_sizes.size());
  int64_t num_rows = 0;
  for (int64_t s : head_sizes) num_rows += s;

  std::mt19937_64 rng(7);
  std::vector<uint8_t> weight(static_cast<size_t>(num_rows * dim));
  std::vector<uint8_t> scales(static_cast<size_t>(num_rows * (dim / block)));
  for (uint8_t& b : weight) b = Fp8Byte(&rng);
  for (uint8_t& b : scales) b = static_cast<uint8_t>(120 + rng() % 14);

  const int64_t num_tokens = 5;
  std::vector<int32_t> ids(static_cast<size_t>(num_tokens * cols));
  for (int32_t& id : ids) {
    id = static_cast<int32_t>(rng() % static_cast<uint64_t>(num_rows));
  }

  const EngramEmbeddingShard shard =
      MakeEngramEmbeddingShard(num_rows, dim, head_sizes, 2, 1);
  REQUIRE(shard.part_n_hash_cols == 3);
  REQUIRE(shard.head_start == 3);
  REQUIRE(shard.part_num_embeddings == 29 + 31);  // two REAL heads, not zero
  const std::vector<uint8_t> w =
      LoadEngramHeadShard(shard, weight.data(), num_rows, dim);
  const std::vector<uint8_t> s =
      LoadEngramHeadShard(shard, scales.data(), num_rows, dim / block);
  std::vector<uint16_t> out(static_cast<size_t>(num_tokens * 3 * dim), 0xFFFFU);
  EngramLookup(shard, w.data(), s.data(), ids.data(), num_tokens, out.data());
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t d = 0; d < dim; ++d) {
      // Column 2 of this rank IS head 5, which does not exist.
      REQUIRE(out[static_cast<size_t>((t * 3 + 2) * dim + d)] == 0);
    }
  }
}

TEST_CASE("engram lookup: a ue8m0 scale byte of zero decodes to +0.0") {
  // NOT AN UPSTREAM CASE. Upstream draws scale bytes from `randint(120, 134)`
  // (:557, :571), so its gate never reaches byte 0 — the one input where the
  // BITCAST decode engram.py:613-614 performs and the ARITHMETIC decode
  // mxfp8_utils.py:66 performs disagree by more than an exponent.
  CHECK(vllm::E8M0BitsToF32(0) == 0.0F);
  CHECK(vllm::E8M0ToF32(0) == std::ldexp(1.0F, -127));
  CHECK(vllm::E8M0BitsToF32(127) == 1.0F);
  CHECK(vllm::E8M0ToF32(127) == 1.0F);
  // The two agree across the whole interior range, which is why no ported test
  // can separate them.
  for (int byte = 1; byte < 255; ++byte) {
    REQUIRE(vllm::E8M0BitsToF32(static_cast<uint8_t>(byte)) ==
            vllm::E8M0ToF32(static_cast<uint8_t>(byte)));
  }

  const int64_t dim = 64;
  const int64_t block = 32;
  const int64_t rows = 4;
  EngramEmbeddingShard shard;
  shard.dim = dim;
  shard.quant_block = block;
  shard.n_hash_cols = 2;
  shard.part_n_hash_cols = 2;
  shard.head_start = 0;
  shard.tp_size = 1;
  shard.vocab_start_idx = 0;
  shard.vocab_end_idx = rows;
  shard.part_num_embeddings = rows;

  std::vector<uint8_t> weight(static_cast<size_t>(rows * dim), 0x40U);  // 2.0
  std::vector<uint8_t> scales(static_cast<size_t>(rows * (dim / block)), 0U);
  // Row 1's second block keeps a live scale so the case cannot pass by zeroing
  // everything.
  scales[static_cast<size_t>(1 * (dim / block) + 1)] = 128U;  // 2^1
  const std::vector<int32_t> ids = {0, 1};
  std::vector<uint16_t> out(static_cast<size_t>(1 * 2 * dim), 0xFFFFU);
  EngramLookup(shard, weight.data(), scales.data(), ids.data(), 1, out.data());

  for (int64_t d = 0; d < dim; ++d) {
    CHECK(FromBf16(out[static_cast<size_t>(d)]) == 0.0F);           // row 0
    CHECK(FromBf16(out[static_cast<size_t>(dim + d)]) ==            // row 1
          (d < block ? 0.0F : 4.0F));
  }
}

// ═══ (5) the injection gate ═════════════════════════════════════════════════

TEST_CASE("engram gate: the fused post-wkv injection matches the definition") {
  // <- ::test_fused_engram_post_wkv_matches_reference (:50-139). The upstream
  // parameter list is preserved; its `tp_size`/`tp_rank` axis selects which
  // slice of `kv` the SP shard sees, which the host reference expresses as
  // `num_kv_tokens` (the kernel's own `source_valid` bound, engram.py:799).
  struct Case {
    int64_t num_tokens;
    int64_t num_kv_tokens;
    int64_t hc_mult;
    int64_t dim;
    int64_t tp_size;
    int64_t tp_rank;
    bool use_mask;
  };
  const std::vector<Case> cases = {
      {1, 1, 4, 5120, 1, 0, false}, {3, 5, 4, 96, 2, 0, true},
      {3, 5, 2, 257, 2, 1, false},  {3, 5, 4, 96, 2, 1, true},
      {1, 1, 4, 96, 4, 3, true},    {0, 0, 4, 96, 4, 0, false},
  };

  std::mt19937_64 rng(0);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  for (const Case& c : cases) {
    CAPTURE(c.num_tokens);
    CAPTURE(c.dim);
    CAPTURE(c.tp_size);
    CAPTURE(c.tp_rank);
    const bool sp = c.tp_size > 1;
    const int64_t shard_size = (c.num_kv_tokens + c.tp_size - 1) / c.tp_size;
    const int64_t kv_start = sp ? c.tp_rank * shard_size : 0;
    const int64_t local_kv =
        sp ? std::max<int64_t>(0, std::min(shard_size, c.num_kv_tokens - kv_start))
           : c.num_kv_tokens;

    EngramGateParams params;
    params.hc_mult = c.hc_mult;
    params.dim = c.dim;
    params.eps = 1e-20F;
    params.clamp_value = 1e-6F;

    std::vector<uint16_t> hidden(
        static_cast<size_t>(c.num_tokens * c.hc_mult * c.dim));
    for (uint16_t& v : hidden) v = Bf16(normal(rng));
    std::vector<uint16_t> full_kv(
        static_cast<size_t>(c.num_kv_tokens * (c.hc_mult + 1) * c.dim));
    for (uint16_t& v : full_kv) v = Bf16(normal(rng));
    std::vector<uint16_t> q_weight(static_cast<size_t>(c.hc_mult * c.dim));
    std::vector<uint16_t> k_weight(static_cast<size_t>(c.hc_mult * c.dim));
    for (uint16_t& v : q_weight) v = Bf16(normal(rng));
    for (uint16_t& v : k_weight) v = Bf16(normal(rng));
    std::vector<uint8_t> full_mask(static_cast<size_t>(c.num_kv_tokens));
    for (size_t i = 0; i < full_mask.size(); ++i) full_mask[i] = i % 2 == 0 ? 1 : 0;

    // `Engram.forward:984-996` slices both kv and the mask to this rank before
    // the kernel runs, and the kernel then indexes them by the LOCAL token.
    const int64_t row = (c.hc_mult + 1) * c.dim;
    // THE TAIL IS GARBAGE ON PURPOSE, and this is the only thing that pins
    // `source_valid`. Both buffers are sized by `num_tokens` while only the
    // first `local_kv` rows are the rank's own slice; the rest is past the
    // SP-local token count and `engram.py:798-799` makes the kernel read it as
    // zero. Value-initialising the tail to 0 makes the guard INVISIBLE —
    // `source_valid ? load : 0.0F` and an unconditional load return the same
    // zeros, so deleting the guard changes no output and the whole case stays
    // green. Non-zero here, and a kept mask bit beside it, makes a missing
    // guard a numeric difference instead.
    std::vector<uint16_t> kv(static_cast<size_t>(c.num_tokens * row));
    for (uint16_t& v : kv) v = Bf16(normal(rng) + 8.0F);
    std::vector<uint8_t> mask(static_cast<size_t>(std::max<int64_t>(c.num_tokens, 0)), 1U);
    for (int64_t t = 0; t < c.num_tokens && t < local_kv; ++t) {
      std::memcpy(&kv[static_cast<size_t>(t * row)],
                  &full_kv[static_cast<size_t>((kv_start + t) * row)],
                  static_cast<size_t>(row) * sizeof(uint16_t));
      mask[static_cast<size_t>(t)] = full_mask[static_cast<size_t>(kv_start + t)];
    }

    std::vector<uint16_t> out(hidden.size(), 0xFFFFU);
    EngramPostWkv(params, hidden.data(), kv.data(), q_weight.data(),
                  k_weight.data(), c.use_mask ? mask.data() : nullptr,
                  c.num_tokens, local_kv, out.data());
    const std::vector<uint16_t> expected =
        ReferencePostWkv(params, hidden, kv, q_weight, k_weight,
                         c.use_mask ? &mask : nullptr, c.num_tokens, local_kv);
    REQUIRE(out.size() == expected.size());
    for (size_t i = 0; i < out.size(); ++i) {
      // rtol=1e-2, atol=1e-2 (:139) — bf16 in, bf16 out, f32 reduction.
      const float got = FromBf16(out[i]);
      const float want = FromBf16(expected[i]);
      REQUIRE(std::abs(got - want) <= 1e-2F + 1e-2F * std::abs(want));
    }
  }
}

TEST_CASE("engram gate: the clamp amplifies a tiny dot and the sign survives") {
  // The two properties of `sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))`
  // (engram.py:831-833) that a randomized case cannot isolate: the clamp is
  // INSIDE the sqrt and it is a MAXIMUM, so the gate input magnitude floors at
  // 1e-3 rather than at 1e-6; and `copysign` keeps the ORIGINAL sign, so the
  // function is odd through the origin rather than continuous at it.
  const int64_t dim = 32;
  EngramGateParams params;
  params.hc_mult = 1;
  params.dim = dim;
  params.eps = 1e-20F;
  params.clamp_value = 1e-6F;

  // Orthogonal hidden and key make the dot exactly zero, so the clamp decides
  // the answer on its own: gate = sigmoid(+1e-3).
  std::vector<uint16_t> hidden(static_cast<size_t>(dim), 0);
  std::vector<uint16_t> kv(static_cast<size_t>(2 * dim), 0);
  std::vector<uint16_t> q_weight(static_cast<size_t>(dim), Bf16(1.0F));
  std::vector<uint16_t> k_weight(static_cast<size_t>(dim), Bf16(1.0F));
  hidden[0] = Bf16(1.0F);
  kv[1] = Bf16(1.0F);            // key, orthogonal to hidden
  kv[dim] = Bf16(1.0F);          // value[0]
  std::vector<uint16_t> out(static_cast<size_t>(dim), 0);
  EngramPostWkv(params, hidden.data(), kv.data(), q_weight.data(),
                k_weight.data(), nullptr, 1, 1, out.data());
  const float expected_gate = 1.0F / (1.0F + std::exp(-1e-3F));
  CHECK(FromBf16(out[0]) == doctest::Approx(1.0F + expected_gate).epsilon(0.01));

  // Flip the key's sign: with a zero dot the SIGN of zero is positive in
  // `dot < 0.0`, so the gate is unchanged. Give the dot a genuinely negative
  // value instead and the gate must fall below one half.
  kv[0] = Bf16(-1.0F);
  kv[1] = 0;
  EngramPostWkv(params, hidden.data(), kv.data(), q_weight.data(),
                k_weight.data(), nullptr, 1, 1, out.data());
  CHECK(FromBf16(out[0]) < 1.0F + 0.5F);
}
