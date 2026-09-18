// DeepSeek-V4.1-Flash indexer arm (W3c) + `kv_source_layer` topology (W3d).
// See `include/vllm/model_executor/models/deepseek_v4_1_indexer.h` for the port
// map, the upstream anchors and the four facts a wrong port passes every shape
// check on. Upstream revision: vLLM `e77daef89e`.
#include "vllm/model_executor/models/deepseek_v4_1_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vt/dtype.h"  // VT_CHECK, F32ToBF16

namespace vllm {
namespace {

// torch `.to(torch.bfloat16).to(torch.float32)` and triton's
// `.to(tl.bfloat16).to(tl.float32)`: round-to-nearest-even into the top 16 bits,
// then widen exactly. `vt::F32ToBF16` is RNE (`src/vt/dtype.cpp:306-313`) and
// `vt::BF16ToF32` is a shift, so the pair IS the round-trip and not an
// approximation of it.
inline float RoundTripBf16(float x) { return vt::BF16ToF32(vt::F32ToBF16(x)); }

// The `(position + 1) % compress_ratio == 0` group-boundary guard
// (`indexer_k_store.py:151-152`). At ratio 1 this is the identity, which is
// why hoisting it out still passes every ratio-1 case.
inline bool IsGroupBoundary(int64_t position, int64_t compress_ratio) {
  return (position + 1) % compress_ratio == 0;
}

// `max(s for s in sources if s <= layer_id)` (`attention.py:284-289`).
//
// Python's `max()` over an EMPTY generator raises `ValueError`, so upstream
// hard-fails when a compressed layer sits below every source. That is mirrored
// as a throw rather than silently folded to the first source: folding would
// turn a malformed config into a running model that indexes into a cache
// nobody published.
int64_t ResolveSource(const std::vector<int64_t>& sources, int64_t layer_id,
                      const char* what) {
  int64_t best = -1;
  bool found = false;
  for (const int64_t s : sources) {
    if (s <= layer_id && (!found || s > best)) {
      best = s;
      found = true;
    }
  }
  VT_CHECK(found,
           std::string("deepseek-v4.1 topology: layer ") +
               std::to_string(layer_id) + " is compressed but no " + what +
               " source sits at or below it (attention.py:284-289 takes "
               "max(s for s in sources if s <= layer_id), which raises on an "
               "empty selection)");
  return best;
}

std::vector<int64_t> BackboneIntersect(const std::vector<int64_t>& ids,
                                       int64_t num_hidden_layers) {
  std::vector<int64_t> out;
  for (const int64_t id : ids) {
    // `is_backbone = layer_id < config.num_hidden_layers` (attention.py:273).
    if (id >= 0 && id < num_hidden_layers &&
        std::find(out.begin(), out.end(), id) == out.end()) {
      out.push_back(id);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// W3d
// ─────────────────────────────────────────────────────────────────────────────

DeepseekV41LayerTopology DeepseekV41ResolveLayerTopology(
    const DeepseekV41Params& params, int64_t layer_id) {
  VT_CHECK(layer_id >= 0, "deepseek-v4.1 topology: layer_id must be >= 0");
  DeepseekV41LayerTopology t;

  // attention.py:251-256 — the bounds guard, NOT a lookup. `DeepseekV41Params`
  // already owns it (W1), so reading it any other way here would be a second
  // spelling of the same rule.
  t.compress_ratio = params.compress_ratio(layer_id);
  VT_CHECK(t.compress_ratio == 0 || t.compress_ratio == 1 ||
               t.compress_ratio == 2,
           std::string("deepseek-v4.1 topology: layer ") +
               std::to_string(layer_id) + " has compress_ratio=" +
               std::to_string(t.compress_ratio) +
               "; only 0 (sliding window), 1 and 2 are supported "
               "(attention.py:257-262)");

  t.is_backbone = params.is_backbone_layer(layer_id);
  const auto contains = [](const std::vector<int64_t>& v, int64_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
  };
  // attention.py:273-275 — `is_backbone` gates BOTH memberships, so an MTP
  // layer whose index happens to appear in a source list is still not a source.
  t.is_kv_source = t.is_backbone && contains(params.kv_source_layer_ids, layer_id);
  t.is_index_source =
      t.is_backbone && contains(params.index_source_layer_ids, layer_id);

  if (t.compress_ratio > 0) {
    // attention.py:276-283.
    VT_CHECK(!params.kv_source_layer_ids.empty() &&
                 !params.index_source_layer_ids.empty(),
             std::string("deepseek-v4.1 topology: compressed layer ") +
                 std::to_string(layer_id) +
                 " requires kv_source_layer_ids / index_source_layer_ids in "
                 "the config (attention.py:276-283)");
    t.kv_source_layer_id =
        ResolveSource(params.kv_source_layer_ids, layer_id, "kv");
    t.index_source_layer_id =
        ResolveSource(params.index_source_layer_ids, layer_id, "index");
  } else {
    // attention.py:290-292 — upstream's `None`. A sliding-window layer resolves
    // NO source.
    t.kv_source_layer_id = -1;
    t.index_source_layer_id = -1;
  }

  if (t.is_index_source) {
    // attention.py:371-395. The index-K cache is ALLOCATED only where the
    // layer is also a kv source, because only a kv source carries the
    // `wk`/`k_norm` pair that produces keys. Every other index source READS
    // the cache of its kv source. Keyed by `kv_source_layer_id`, never by
    // `index_source_layer_id`.
    t.owns_index_k_cache = t.is_kv_source;
    if (t.is_kv_source) {
      t.index_k_cache_owner_layer_id = layer_id;
    } else {
      VT_CHECK(t.kv_source_layer_id >= 0,
               std::string("deepseek-v4.1 topology: index source ") +
                   std::to_string(layer_id) +
                   " does not own its K cache and has no kv source to read one "
                   "from (attention.py:386 asserts kv_source_layer_id is not "
                   "None)");
      t.index_k_cache_owner_layer_id = t.kv_source_layer_id;
    }
    // attention.py:398-399 — both are scoped to an index source; a layer with
    // no indexer neither publishes nor consumes candidates.
    t.is_candidate_source = layer_id == params.candidate_source_layer_id;
    t.uses_candidates = params.candidate_source_layer_id >= 0 &&
                        params.candidate_source_layer_id < layer_id;
  }
  return t;
}

DeepseekV41ModelTopology DeepseekV41ResolveModelTopology(
    const DeepseekV41Params& params) {
  DeepseekV41ModelTopology m;
  m.compressed_kv_cache_layers =
      BackboneIntersect(params.kv_source_layer_ids, params.num_hidden_layers);
  m.indexer_layers =
      BackboneIntersect(params.index_source_layer_ids, params.num_hidden_layers);
  // The intersection, and the gap between this and `indexer_layers` is the
  // whole of fact 4 in the header: 8 indexers, 4 caches.
  for (const int64_t id : m.indexer_layers) {
    if (std::find(m.compressed_kv_cache_layers.begin(),
                  m.compressed_kv_cache_layers.end(),
                  id) != m.compressed_kv_cache_layers.end()) {
      m.index_k_cache_layers.push_back(id);
    }
  }
  return m;
}

DeepseekV41LayerRope DeepseekV41ResolveLayerRope(const DeepseekV41Params& params,
                                                 int64_t layer_id) {
  const int64_t ratio = params.compress_ratio(layer_id);
  DeepseekV41LayerRope r;
  // rope.py:28-30. THE THRESHOLD IS `> 0` IN V4.1 AND `> 1` IN V4. That is
  // three `>` sites, not one: V4.1's `:23`, `:29` (this one) and `:31`. The
  // whole `diff` against V4's copy is those three plus the one-line comment
  // above `:31`, rewritten into the two lines `:32-33` -- four lines removed,
  // five added.
  r.rope_theta = ratio > 0 ? params.compress_rope_theta : params.rope_theta;
  r.beta_fast = params.rope_beta_fast;
  r.beta_slow = params.rope_beta_slow;
  if (ratio > 0 && params.rope_type != "default") {
    // rope.py:31-38. `apply_yarn_scaling` is absent from the published
    // config and upstream's `.get(..., True)` therefore resolves
    // "deepseek_yarn"; the `deepseek_llama_scaling` arm is unreachable for this
    // checkpoint and W1 parsed no field for it. Recorded, not guessed.
    r.use_yarn = true;
    r.factor = params.rope_scale_factor;
    r.original_max_position_embeddings = params.rope_orig_ctx;
  } else {
    // rope.py:39-43. Plain RoPE, expressed as YaRN pinned to the identity:
    // factor 1.0 and `original_max_position_embeddings =
    // max_position_embeddings`.
    r.use_yarn = false;
    r.factor = 1.0;
    r.original_max_position_embeddings = params.max_position_embeddings;
  }
  return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// W3c
// ─────────────────────────────────────────────────────────────────────────────

int64_t DeepseekV41IndexerKCacheRowBytes(int64_t index_head_dim,
                                         DeepseekV41IndexerKFormat format) {
  VT_CHECK(index_head_dim > 0, "deepseek-v4.1 indexer cache: index_head_dim > 0");
  if (format == DeepseekV41IndexerKFormat::kMxfp4) {
    // attention.py:84-87: head_dim/2 packed + head_dim/32 UE8M0 -> 68 at 128.
    VT_CHECK(index_head_dim % 32 == 0,
             "deepseek-v4.1 indexer cache: the MXFP4 row needs index_head_dim % "
             "32 == 0 (one UE8M0 byte per 32 values)");
    return index_head_dim / 2 + index_head_dim / 32;
  }
  // attention.py:88-90: head_dim fp8 bytes + one f32 scale per 128 -> 132.
  return index_head_dim + index_head_dim / 128 * 4;
}

std::vector<int64_t> DeepseekV41IndexerBlockSizes(bool is_hopper_family) {
  // indexer.py:252-259. ARCH-CONDITIONAL against V4's flat [256].
  return {is_hopper_family ? int64_t{64} : int64_t{128}};
}

void DeepseekV41FusedQKvRmsNorm(const float* q, const float* kv,
                                const float* q_weight, const float* kv_weight,
                                int64_t num_tokens, int64_t q_size,
                                int64_t kv_size, double eps, float* q_out,
                                float* kv_out) {
  VT_CHECK(num_tokens >= 0 && q_size > 0 && kv_size > 0,
           "deepseek-v4.1 q/kv rmsnorm: degenerate shape");
  VT_CHECK(q != nullptr && kv != nullptr && q_weight != nullptr &&
               kv_weight != nullptr && q_out != nullptr && kv_out != nullptr,
           "deepseek-v4.1 q/kv rmsnorm: null buffer");
  // fused_qk_rmsnorm.py's kernel: `rrms = rsqrt(sum(v*v)/SIZE + eps)` then
  // `v * rrms * w`, both branches identical apart from width and weight.
  //
  // EVERY ACCUMULATION IS f32, NOT f64, AND THAT IS THE MIRRORED BEHAVIOUR.
  // `fused_qk_rmsnorm.py:71-73` states it in its own words -- "RMSNorm in fp32
  // throughout ... keep x, rrms, and w all in fp32 and perform a single cast at
  // store" -- and the kernel body at `:85-88` loads `x` and `w` `.to(tl.float32)`
  // and reduces there. A f64 host reference would be a DIFFERENT function that
  // happens to agree at these tolerances, and the later device port, which
  // computes in f32, would then be gated against an oracle upstream does not
  // have. `eps` stays a `double` in the signature because it is a scalar the
  // caller supplies, and upstream passes it as a Python float too; it is
  // narrowed here at the one point it enters the arithmetic.
  const float eps_f32 = static_cast<float>(eps);
  const auto norm = [&](const float* in, const float* w, int64_t size,
                        float* out) {
    for (int64_t t = 0; t < num_tokens; ++t) {
      const float* row = in + t * size;
      float sq = 0.0F;
      for (int64_t i = 0; i < size; ++i) {
        sq += row[i] * row[i];
      }
      const float rrms =
          1.0F / std::sqrt(sq / static_cast<float>(size) + eps_f32);
      for (int64_t i = 0; i < size; ++i) {
        out[t * size + i] = row[i] * rrms * w[i];
      }
    }
  };
  norm(q, q_weight, q_size, q_out);
  norm(kv, kv_weight, kv_size, kv_out);
}

void DeepseekV41IndexerKNormRopeStore(
    const float* k_pre, const int64_t* positions, const float* cos_sin_cache,
    const float* rms_norm_weight, double eps, int64_t num_tokens,
    int64_t num_rows, int64_t index_head_dim, int64_t rope_head_dim,
    int64_t compress_ratio, DeepseekV41IndexerKFormat format,
    DeepseekV41IndexerKLayout layout, const int64_t* slot_mapping,
    int64_t num_blocks, int64_t block_size, uint8_t* k_cache) {
  // indexer_k_store.py:56-61 — the entry point's own asserts, mirrored.
  VT_CHECK(compress_ratio == 1 || compress_ratio == 2,
           "deepseek-v4.1 indexer k store: compress_ratio must be 1 or 2 "
           "(indexer_k_store.py:59). Ratio 0 layers have no indexer and ratio 4 "
           "is V4's, which V4.1 deletes");
  VT_CHECK(num_tokens >= 0 && num_tokens <= num_rows,
           "deepseek-v4.1 indexer k store: num_tokens must be <= the rows of "
           "k_pre (indexer_k_store.py:58)");
  if (num_tokens == 0) return;  // indexer_k_store.py:60-61
  VT_CHECK(index_head_dim > 0 && index_head_dim % 2 == 0,
           "deepseek-v4.1 indexer k store: index_head_dim must be even");
  VT_CHECK(rope_head_dim > 0 && rope_head_dim % 2 == 0 &&
               rope_head_dim <= index_head_dim,
           "deepseek-v4.1 indexer k store: rope_head_dim must be even and <= "
           "index_head_dim (the TRAILING lanes rotate)");
  VT_CHECK(k_pre != nullptr && positions != nullptr && cos_sin_cache != nullptr &&
               rms_norm_weight != nullptr && slot_mapping != nullptr &&
               k_cache != nullptr,
           "deepseek-v4.1 indexer k store: null buffer");
  VT_CHECK(block_size > 0 && num_blocks > 0,
           "deepseek-v4.1 indexer k store: degenerate page geometry");

  const bool use_fp4 = format == DeepseekV41IndexerKFormat::kMxfp4;
  const int64_t token_stride = use_fp4 ? index_head_dim / 2 : index_head_dim;
  const int64_t scale_dim = use_fp4 ? index_head_dim / 32 : 4;
  const int64_t row_bytes = DeepseekV41IndexerKCacheRowBytes(index_head_dim, format);
  VT_CHECK(token_stride + scale_dim == row_bytes,
           "deepseek-v4.1 indexer k store: the value and scale regions must "
           "tile the row exactly");
  const int64_t block_stride = block_size * row_bytes;

  constexpr int64_t kBlockTile = 16;  // indexer_k_store.py:25-26
  constexpr int64_t kHeadTile = 16;
  const bool shuffle = layout == DeepseekV41IndexerKLayout::kRocmTiled16x16;
  if (shuffle) {
    // indexer_k_store.py:80-90 — both refusals, verbatim in intent.
    VT_CHECK(!use_fp4,
             "MXFP4 indexer K cache has no tiled ROCm layout; the ROCm readers "
             "only implement the FP8 one (indexer_k_store.py:80-84)");
    VT_CHECK(block_size % kBlockTile == 0 && index_head_dim % kHeadTile == 0,
             "ROCm tiled indexer K cache needs block_size % 16 == 0 and "
             "head_dim % 16 == 0 (indexer_k_store.py:85-90)");
  }

  const int64_t num_pairs = index_head_dim / 2;
  const int64_t nope_pairs = (index_head_dim - rope_head_dim) / 2;
  const int64_t half_rope = rope_head_dim / 2;
  // `rms_norm_eps` reaches the kernel as a Triton scalar and is added to an
  // fp32 variance (`indexer_k_store.py:159-160`), so it is narrowed here, at
  // the one point it enters the arithmetic, and not carried as f64.
  const float eps_f32 = static_cast<float>(eps);

  std::vector<float> even(static_cast<size_t>(num_pairs));
  std::vector<float> odd(static_cast<size_t>(num_pairs));
  std::vector<float> value(static_cast<size_t>(index_head_dim));

  for (int64_t t = 0; t < num_tokens; ++t) {
    // (a) SKIP. indexer_k_store.py:146-152, in this order: the slot first, the
    // group boundary second.
    const int64_t slot = slot_mapping[t];
    if (slot < 0) continue;
    const int64_t position = positions[t];
    if (!IsGroupBoundary(position, compress_ratio)) continue;
    VT_CHECK(slot < num_blocks * block_size,
             "deepseek-v4.1 indexer k store: slot is past the end of the paged "
             "cache");

    // (b) RMSNORM in f32 with the bf16 round-trip (indexer_k_store.py:156-161).
    // f32 THROUGHOUT, exactly as the kernel says: `indexer_k_store.py:156`
    // titles this block "k_norm (fp32 throughout, bf16 roundtrip like the
    // reference)" and `:157-161` loads, reduces and scales in `tl.float32`.
    // Accumulating in f64 here would give a device port an oracle upstream does
    // not compute, and the bf16 round-trip below would hide the difference on
    // most inputs while moving it on the ties.
    const float* row = k_pre + t * index_head_dim;
    float sq = 0.0F;
    for (int64_t i = 0; i < index_head_dim; ++i) {
      sq += row[i] * row[i];
    }
    const float rrms =
        1.0F / std::sqrt(sq / static_cast<float>(index_head_dim) + eps_f32);
    for (int64_t i = 0; i < index_head_dim; ++i) {
      const float y = row[i] * rrms * rms_norm_weight[i];
      // `.to(tl.bfloat16)` then `.to(tl.float32)` — the rounding boundary the
      // reference materializes. Dropping it changes the stored bytes.
      value[static_cast<size_t>(i)] = RoundTripBf16(y);
    }

    // (c) ROPE at the GROUP's FIRST position, over the TRAILING lanes only
    // (indexer_k_store.py:163-187). GPT-J/interleaved: pair p is
    // (lane 2p, lane 2p+1), and only pairs p >= nope_pairs rotate.
    for (int64_t p = 0; p < num_pairs; ++p) {
      even[static_cast<size_t>(p)] = value[static_cast<size_t>(2 * p)];
      odd[static_cast<size_t>(p)] = value[static_cast<size_t>(2 * p + 1)];
    }
    const int64_t compressed_pos = (position / compress_ratio) * compress_ratio;
    const float* cs = cos_sin_cache + compressed_pos * rope_head_dim;
    for (int64_t p = 0; p < num_pairs; ++p) {
      const int64_t local = p - nope_pairs;
      // `other=1.0` / `other=0.0` for the masked-off NoPE pairs: cos 1, sin 0
      // is the identity, which is why the NoPE lanes still take the bf16
      // round-trip below.
      const float cos_v = local >= 0 ? cs[local] : 1.0F;
      const float sin_v = local >= 0 ? cs[half_rope + local] : 0.0F;
      const float e = even[static_cast<size_t>(p)];
      const float o = odd[static_cast<size_t>(p)];
      even[static_cast<size_t>(p)] = RoundTripBf16(e * cos_v - o * sin_v);
      odd[static_cast<size_t>(p)] = RoundTripBf16(o * cos_v + e * sin_v);
    }

    // (d) QUANTIZE AND STORE into the SEGREGATED page: every value row first,
    // then every scale row (indexer_k_store.py:189-206).
    const int64_t block_idx = slot / block_size;
    const int64_t pos_in_block = slot % block_size;
    uint8_t* page = k_cache + block_idx * block_stride;
    uint8_t* scale_ptr = page + block_size * token_stride + pos_in_block * scale_dim;

    if (use_fp4) {
      // indexer_k_store.py:208-234. Block b of 32 lanes == pairs [16b, 16b+16).
      const int64_t n_quant_blocks = index_head_dim / 32;
      constexpr int64_t kHalfBlock = 16;
      uint8_t* val_ptr = page + pos_in_block * token_stride;
      for (int64_t b = 0; b < n_quant_blocks; ++b) {
        float amax = 0.0F;
        for (int64_t j = 0; j < kHalfBlock; ++j) {
          const size_t p = static_cast<size_t>(b * kHalfBlock + j);
          amax = std::max(amax, std::max(std::fabs(even[p]), std::fabs(odd[p])));
        }
        // 6 * 2^-126, from DeepSeek's own kernel.py:163 via fused_indexer_q.py.
        amax = std::max(amax, 6.0F * std::ldexp(1.0F, -126));
        double log2_ratio = std::ceil(std::log2(static_cast<double>(amax) / 6.0));
        log2_ratio = std::min(std::max(log2_ratio, -127.0), 127.0);
        const double inv_scale = std::exp2(-log2_ratio);
        scale_ptr[b] = static_cast<uint8_t>(log2_ratio + 127.0);
        for (int64_t j = 0; j < kHalfBlock; ++j) {
          const size_t p = static_cast<size_t>(b * kHalfBlock + j);
          // `cvt.rn.satfinite.e2m1x2.f32`: round-to-nearest-EVEN onto
          // {0,.5,1,1.5,2,3,4,6}, saturating at +/-6. `CastToFp4`'s half-open
          // buckets ARE that rule — every tie (0.25, 1.25, 2.5, 5.0) resolves
          // to the even-mantissa neighbour and the top bucket saturates — and
          // `Fp4ToNibble` canonicalizes -0 to +0, which is exactly the zero-sign
          // difference upstream's own test normalizes away before comparing.
          const float lo = CastToFp4(static_cast<float>(even[p] * inv_scale));
          const float hi = CastToFp4(static_cast<float>(odd[p] * inv_scale));
          // "$1 is high nibble, $2 is low nibble" with args [x_hi, x_lo]:
          // LOW nibble = the EVEN lane, HIGH nibble = the ODD lane.
          val_ptr[b * kHalfBlock + j] = static_cast<uint8_t>(
              (Fp4ToNibble(lo) & 0x0FU) | (Fp4ToNibble(hi) << 4));
        }
      }
    } else {
      // indexer_k_store.py:235-253. Re-interleave to lane order, one last bf16
      // round-trip, one per-token f32 scale.
      for (int64_t p = 0; p < num_pairs; ++p) {
        value[static_cast<size_t>(2 * p)] = RoundTripBf16(even[static_cast<size_t>(p)]);
        value[static_cast<size_t>(2 * p + 1)] =
            RoundTripBf16(odd[static_cast<size_t>(p)]);
      }
      float absmax = 0.0F;
      for (int64_t i = 0; i < index_head_dim; ++i) {
        absmax = std::max(absmax, std::fabs(value[static_cast<size_t>(i)]));
      }
      absmax = std::max(absmax, 1e-4F);
      const double exponent = std::ceil(std::log2(static_cast<double>(absmax) / 448.0));
      const double inv_scale = std::exp2(-exponent);
      for (int64_t i = 0; i < index_head_dim; ++i) {
        double x = static_cast<double>(value[static_cast<size_t>(i)]) * inv_scale;
        x = std::min(std::max(x, -448.0), 448.0);
        const uint8_t byte = F32ToF8E4M3(static_cast<float>(x));
        int64_t off = i;
        if (shuffle) {
          // indexer_k_store.py:245-248: tiled = lane/16 * 256 + lane%16, into a
          // [block_size/16, head_dim/16, 16, 16] page.
          off = (i / kHeadTile) * kBlockTile * kHeadTile + (i % kHeadTile);
        }
        uint8_t* val_ptr =
            shuffle ? page + (pos_in_block / kBlockTile) * kBlockTile * token_stride +
                          (pos_in_block % kBlockTile) * kHeadTile
                    : page + pos_in_block * token_stride;
        val_ptr[off] = byte;
      }
      // `tl.store(scale_ptr.to(pointer_type(float32)), exp2(exponent))` — the
      // FORWARD scale, four raw f32 bytes, not the reciprocal the values were
      // multiplied by.
      const float scale = static_cast<float>(std::exp2(exponent));
      std::memcpy(scale_ptr, &scale, sizeof(float));
    }
  }
}

}  // namespace vllm
