// `KV-DSV4-MULTICACHE` W5 (#2323) — DeepSeek-V4's attention IS `vt::MlaDecodeAttention`.
//
// THE CLAIM THIS FILE EXISTS TO TEST. W5 routes V4's decode onto the shared
// paged MLA op instead of its bespoke loop over a contiguous `deck`. That is only
// sound if the op computes the SAME function, and the spec's argument for it is a
// reading of two source files. This checks it by running both.
//
// V4's attention (`deepseek_v4.cpp`, "5. attention with per-head sink softmax"):
//
//     sc[j]   = dot(q[t,h], kv[j]) * scale          over the FULL head_dim
//     prob    = softmax_with_sink(sc, sink[h])      sink in the DENOMINATOR only
//     o[t,h]  = sum_j prob[j] * kv[j]               the FULL row is the value
//
// `vt::MlaDecodeAttention` dots the query over `head_size` columns of the cache
// row and accumulates the value over the LEADING `v_head_dim` columns of that
// same row. So V4 is the case `head_size == v_head_dim == head_dim`: the whole
// latent is both key and value. That degenerate-looking choice is the entire
// reason the op fits a model it was not written for.
//
// NOT BIT-IDENTITY, and the reason is stated rather than hidden: V4's host loop
// is a two-pass softmax that sums in key order; the op's CPU kernel is an ONLINE
// softmax with running rescales. Same arithmetic, different association, so the
// f32 results differ in the last bits. The bound below is relative to the output
// scale and tight enough that a wrong VALUE SOURCE, a missing sink or a wrong
// scale could not pass it.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_dsa.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

std::vector<float> Rand(size_t n, uint32_t seed, float scale) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * scale;
  }
  return v;
}

vt::Tensor Contig(void* p, vt::DType dt, vt::Device dev,
                  std::initializer_list<int64_t> shape) {
  return vt::Tensor::Contiguous(p, dt, dev, shape);
}

}  // namespace

TEST_CASE("W5: vt::MlaDecodeAttention reproduces DeepSeek-V4's sink attention") {
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // V4-Flash's real widths, so the case cannot pass by accident on a toy shape.
  const int64_t hd = 512;   // head_dim — key AND value width
  const int64_t nh = 4;     // a few heads, each with its own sink
  const int64_t n_keys = 37;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 7u, 0.35f);
  const std::vector<float> qv = Rand(static_cast<size_t>(nh * hd), 11u, 0.30f);
  std::vector<float> sink(static_cast<size_t>(nh));
  for (int64_t h = 0; h < nh; ++h) sink[static_cast<size_t>(h)] = -0.3f + 0.2f * static_cast<float>(h);

  // ── (A) V4's own arithmetic, transcribed from the forward ────────────────
  std::vector<float> want(static_cast<size_t>(nh * hd), 0.0f);
  for (int64_t h = 0; h < nh; ++h) {
    std::vector<float> sc(static_cast<size_t>(n_keys));
    for (int64_t j = 0; j < n_keys; ++j) {
      float dot = 0.0f;
      for (int64_t d = 0; d < hd; ++d)
        dot += qv[static_cast<size_t>(h * hd + d)] * kv[static_cast<size_t>(j * hd + d)];
      sc[static_cast<size_t>(j)] = dot * scale;
    }
    // The SAME host reference the forward calls.
    const std::vector<float> prob =
        vllm::deepseek_v4::SoftmaxWithSink(sc, sink[static_cast<size_t>(h)]);
    for (int64_t j = 0; j < n_keys; ++j) {
      const float w = prob[static_cast<size_t>(j)];
      for (int64_t d = 0; d < hd; ++d)
        want[static_cast<size_t>(h * hd + d)] += w * kv[static_cast<size_t>(j * hd + d)];
    }
  }

  // ── (B) the shared op, over a PAGED cache holding the same keys ──────────
  const int64_t block_size = 16;
  const int64_t num_blocks = (n_keys + block_size - 1) / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
  for (int64_t j = 0; j < n_keys; ++j) {
    for (int64_t d = 0; d < hd; ++d)
      cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
  }
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens{static_cast<int32_t>(n_keys)};

  std::vector<float> got(static_cast<size_t>(nh * hd), 0.0f);
  vt::Tensor t_out = Contig(got.data(), vt::DType::kF32, q.device, {1, nh, hd});
  vt::Tensor t_q = Contig(const_cast<float*>(qv.data()), vt::DType::kF32, q.device, {1, nh, hd});
  vt::Tensor t_c = Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});
  vt::Tensor t_bt = Contig(block_table.data(), vt::DType::kI32, q.device, {1, num_blocks});
  vt::Tensor t_sl = Contig(seq_lens.data(), vt::DType::kI32, q.device, {1});
  vt::Tensor t_sink = Contig(sink.data(), vt::DType::kF32, q.device, {nh});

  vt::MlaDecodeAttentionArgs args;
  args.scale = scale;
  args.attn_sink = &t_sink;
  vt::MlaDecodeAttention(q, t_out, nullptr, t_q, t_c, t_bt, t_sl, args);

  // ── the comparison ───────────────────────────────────────────────────────
  double worst = 0.0, mag = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(got[i]));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
    worst = std::max(worst, std::abs(static_cast<double>(want[i] - got[i])));
  }
  // NON-TRIVIAL first: an all-zero pair satisfies any bound and proves nothing.
  REQUIRE(mag > 1e-3);
  CHECK(worst <= 1e-5 * mag);

  // AND THE SINK IS LOAD-BEARING. Without it the op computes a different answer,
  // so the agreement above is evidence about the sink and not only about the dot
  // product and the value source.
  std::vector<float> no_sink(static_cast<size_t>(nh * hd), 0.0f);
  vt::Tensor t_out2 = Contig(no_sink.data(), vt::DType::kF32, q.device, {1, nh, hd});
  vt::MlaDecodeAttentionArgs plain = args;
  plain.attn_sink = nullptr;
  vt::MlaDecodeAttention(q, t_out2, nullptr, t_q, t_c, t_bt, t_sl, plain);
  double diff = 0.0;
  for (size_t i = 0; i < want.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(want[i] - no_sink[i])));
  CHECK(diff > 1e-4 * mag);
}

TEST_CASE("W5: the paged helper reproduces V4's DENSE-CAUSAL step for T > 1") {
  // The single-query case above proves the OP matches V4's attention. This proves
  // the MAPPING that lets one decode op serve a whole step: `batch = T` with
  // `seq_lens[t] = kv_base + t + 1`.
  //
  // WHY THAT IS THE RISKY PART. Query `t` sits at global position `kv_base + t`
  // and may see `[0, kv_base + t]`. Off-by-one in either direction still produces
  // finite, plausible output -- one key too many leaks the FUTURE, one too few
  // drops the token's own key -- and neither shows up as a NaN or a crash. So the
  // case runs several `kv_base` values, including 0, and compares against V4's own
  // `sel`-driven arithmetic rather than a restatement of the formula.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = 512, nh = 3, T = 5;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  std::vector<float> sink(static_cast<size_t>(nh));
  for (int64_t h = 0; h < nh; ++h) sink[static_cast<size_t>(h)] = -0.25f + 0.15f * static_cast<float>(h);

  for (const int64_t kv_base : {int64_t{0}, int64_t{7}, int64_t{31}}) {
    const int64_t n_keys = kv_base + T;  // the cache holds every key this step sees
    const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 3u + static_cast<uint32_t>(kv_base), 0.3f);
    const std::vector<float> qv = Rand(static_cast<size_t>(T * nh * hd), 91u, 0.25f);

    // (A) V4's own step-5 arithmetic, causal `sel` and all.
    std::vector<float> want(static_cast<size_t>(T * nh * hd), 0.0f);
    for (int64_t t = 0; t < T; ++t) {
      const int64_t g = kv_base + t;  // this query's GLOBAL position
      for (int64_t h = 0; h < nh; ++h) {
        std::vector<float> sc(static_cast<size_t>(g + 1));
        const float* qh = &qv[static_cast<size_t>((t * nh + h) * hd)];
        for (int64_t j = 0; j <= g; ++j) {
          float dot = 0.0f;
          for (int64_t d = 0; d < hd; ++d) dot += qh[d] * kv[static_cast<size_t>(j * hd + d)];
          sc[static_cast<size_t>(j)] = dot * scale;
        }
        const std::vector<float> prob =
            vllm::deepseek_v4::SoftmaxWithSink(sc, sink[static_cast<size_t>(h)]);
        float* oh = &want[static_cast<size_t>((t * nh + h) * hd)];
        for (int64_t j = 0; j <= g; ++j)
          for (int64_t d = 0; d < hd; ++d)
            oh[d] += prob[static_cast<size_t>(j)] * kv[static_cast<size_t>(j * hd + d)];
      }
    }

    // (B) the paged helper.
    const int64_t block_size = 16;
    const int64_t num_blocks = (n_keys + block_size - 1) / block_size;
    std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
    for (int64_t j = 0; j < n_keys; ++j)
      for (int64_t d = 0; d < hd; ++d)
        cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
    vt::Tensor t_c =
        Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});
    const std::vector<float> got = vllm::deepseek_v4::PagedCausalMlaAttention(
        q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, sink, scale,
        /*no_sink=*/false);

    REQUIRE(got.size() == want.size());
    double worst = 0.0, mag = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
      REQUIRE(!std::isnan(got[i]));
      mag = std::max(mag, std::abs(static_cast<double>(want[i])));
      worst = std::max(worst, std::abs(static_cast<double>(want[i] - got[i])));
    }
    CAPTURE(kv_base);
    REQUIRE(mag > 1e-3);
    CHECK(worst <= 1e-5 * mag);
  }
}

TEST_CASE("W5: the paged helper's no_sink arm is EXACTLY no sink") {
  // `kNoAttnSink` feeds -inf, which contributes `exp(-inf - m) == 0` to the
  // denominator. This pins that it equals a plain softmax rather than merely
  // approximating one -- and that the SINKED arm differs, so the case is not
  // vacuous.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = 512, nh = 2, T = 3, kv_base = 4;
  const int64_t n_keys = kv_base + T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 55u, 0.3f);
  const std::vector<float> qv = Rand(static_cast<size_t>(T * nh * hd), 56u, 0.25f);
  const std::vector<float> sink(static_cast<size_t>(nh), 0.0f);
  const std::vector<float> neg_inf(static_cast<size_t>(nh),
                                   -std::numeric_limits<float>::infinity());

  const int64_t block_size = 16;
  const int64_t num_blocks = (n_keys + block_size - 1) / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
  for (int64_t j = 0; j < n_keys; ++j)
    for (int64_t d = 0; d < hd; ++d)
      cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
  vt::Tensor t_c = Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});

  const auto no_sink = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, sink, scale, /*no_sink=*/true);
  const auto explicit_neg = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, neg_inf, scale,
      /*no_sink=*/false);
  const auto sinked = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, sink, scale, /*no_sink=*/false);

  REQUIRE(no_sink.size() == explicit_neg.size());
  bool identical = true;
  for (size_t i = 0; i < no_sink.size(); ++i)
    if (no_sink[i] != explicit_neg[i]) identical = false;
  CHECK(identical);

  double diff = 0.0, mag = 0.0;
  for (size_t i = 0; i < sinked.size(); ++i) {
    mag = std::max(mag, std::abs(static_cast<double>(no_sink[i])));
    diff = std::max(diff, std::abs(static_cast<double>(sinked[i] - no_sink[i])));
  }
  REQUIRE(mag > 1e-3);
  CHECK(diff > 1e-4 * mag);
}

TEST_CASE("W5: the SLIDING WINDOW matches upstream's swa_only layer, and bounds the old claim") {
  // `KV-DSV4-MULTICACHE` W5 (#2323). Upstream attends a `compress_ratio <= 1`
  // layer over its 128-token window and NOTHING else -- `flash_mla_with_kvcache`
  // is called with `k_cache=swa_cache` and `extra_k_cache=None` when
  // `swa_only` (`nvidia/flashmla.py`). Our forward attends the FULL causal
  // prefix, so the two diverge above the window.
  //
  // THIS CASE PINS BOTH HALVES OF THAT. The windowed helper must match a windowed
  // host reference, AND it must DIFFER from the full-context answer -- because a
  // window that silently did nothing would pass the first assertion alone, and
  // that is exactly the failure this row's records already made once by carrying
  // a 512-token bound where the real one is 128.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = 512, nh = 2, T = 3, kv_base = 40, win = 8;
  const int64_t n_keys = kv_base + T;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 17u, 0.3f);
  const std::vector<float> qv = Rand(static_cast<size_t>(T * nh * hd), 18u, 0.25f);
  const std::vector<float> sink(static_cast<size_t>(nh), -0.2f);

  const int64_t block_size = 16;
  const int64_t num_blocks = (n_keys + block_size - 1) / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
  for (int64_t j = 0; j < n_keys; ++j)
    for (int64_t d = 0; d < hd; ++d)
      cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
  vt::Tensor t_c = Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});

  // (A) a WINDOWED host reference: query t sees `[g - win + 1, g]`, g = kv_base+t.
  std::vector<float> want(static_cast<size_t>(T * nh * hd), 0.0f);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t g = kv_base + t;
    const int64_t lo = std::max<int64_t>(0, g - win + 1);
    for (int64_t h = 0; h < nh; ++h) {
      std::vector<float> sc;
      const float* qh = &qv[static_cast<size_t>((t * nh + h) * hd)];
      for (int64_t j = lo; j <= g; ++j) {
        float dot = 0.0f;
        for (int64_t d = 0; d < hd; ++d) dot += qh[d] * kv[static_cast<size_t>(j * hd + d)];
        sc.push_back(dot * scale);
      }
      const std::vector<float> prob =
          vllm::deepseek_v4::SoftmaxWithSink(sc, sink[static_cast<size_t>(h)]);
      float* oh = &want[static_cast<size_t>((t * nh + h) * hd)];
      for (int64_t j = lo; j <= g; ++j)
        for (int64_t d = 0; d < hd; ++d)
          oh[d] += prob[static_cast<size_t>(j - lo)] * kv[static_cast<size_t>(j * hd + d)];
    }
  }

  const auto got = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, sink, scale,
      /*no_sink=*/false, /*sliding_window=*/win);
  const auto full = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_c, num_blocks, block_size, T, nh, hd, kv_base, sink, scale,
      /*no_sink=*/false, /*sliding_window=*/0);

  REQUIRE(got.size() == want.size());
  double worst = 0.0, mag = 0.0, vs_full = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(got[i]));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
    worst = std::max(worst, std::abs(static_cast<double>(want[i] - got[i])));
    vs_full = std::max(vs_full, std::abs(static_cast<double>(got[i] - full[i])));
  }
  REQUIRE(mag > 1e-3);
  CHECK(worst <= 1e-5 * mag);
  // THE WINDOW IS LOAD-BEARING: with kv_base=40 and win=8 the full-context answer
  // is a different function, so a no-op window could not pass this.
  CHECK(vs_full > 1e-3 * mag);
}

namespace {

// A `MultiKvCacheIndex` over a name list, the way the runner builds one.
struct FakeIndex {
  std::vector<std::string> names;
  vllm::MultiKvCacheIndex Index() const {
    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;
    return mk;
  }
};

vllm::DeepseekV4Params SwaOnlyParams(int64_t layers, int64_t head_dim) {
  vllm::DeepseekV4Params p;
  p.num_hidden_layers = layers;
  p.head_dim = head_dim;
  p.compress_ratios.assign(static_cast<size_t>(layers), 0);  // every layer SWA-only
  return p;
}

std::vector<vllm::PagedKvCache> FakeCaches(size_t n, int64_t head_dim,
                                           std::vector<std::vector<float>>* storage) {
  std::vector<vllm::PagedKvCache> v(n);
  storage->assign(n, std::vector<float>(static_cast<size_t>(2 * 4 * head_dim), 0.0f));
  for (size_t i = 0; i < n; ++i) {
    v[i].data = (*storage)[i].data();
    v[i].dtype = vt::DType::kF32;
    v[i].num_blocks = 2;
    v[i].block_size = 4;
    v[i].num_kv_heads = 1;
    v[i].head_size = head_dim;
  }
  return v;
}

}  // namespace

TEST_CASE("W5: the SWA page resolver accepts a served shape and REFUSES every other") {
  // `KV-DSV4-MULTICACHE` W5 (#2323). Every safety-critical decision of the paged
  // engine path lives in this pure function, so that it is gateable WITHOUT a
  // runner, a checkpoint or a GPU -- the same reason W1 made the staging budget
  // pure. The registry adapter around it is glue.
  //
  // EACH REFUSAL GUARDS A SILENT WRONG ANSWER, not a crash: batching would attend
  // the wrong history for every request but one, a compressor layer would attend
  // the raw prefix instead of window-plus-compressed-history, and an unresolved
  // name would drop a cache. All three produce plausible tokens.
  const int64_t L = 3, HD = 32;
  const vt::Device dev{vt::DeviceType::kCPU, 0};
  std::vector<std::vector<float>> storage;
  const auto caches = FakeCaches(static_cast<size_t>(L), HD, &storage);
  FakeIndex fx;
  for (int64_t l = 0; l < L; ++l)
    fx.names.push_back("model.layers." + std::to_string(l) + ".attn.swa_cache");
  const auto mk = fx.Index();
  const auto params = SwaOnlyParams(L, HD);

  // THE SERVED SHAPE: one request, every layer SWA-only, every name resolving.
  std::vector<vt::Tensor> pages;
  CHECK(vllm::ResolveDeepseekV4SwaPages(params, mk, caches, 1, dev, &pages).empty());
  REQUIRE(pages.size() == static_cast<size_t>(L));
  for (const auto& t : pages) {
    CHECK(t.rank == 3);
    CHECK(t.shape[2] == HD);
    CHECK(t.data != nullptr);
  }

  // 1. A BATCH refuses -- one `kv_base` cannot serve several context lengths.
  pages.clear();
  const std::string batched =
      vllm::ResolveDeepseekV4SwaPages(params, mk, caches, 2, dev, &pages);
  CHECK(batched.find("one request per step only") != std::string::npos);
  CHECK(pages.empty());

  // 2. A COMPRESSOR layer refuses, naming the row that owns it.
  auto with_comp = params;
  with_comp.compress_ratios[1] = 128;
  const std::string comp =
      vllm::ResolveDeepseekV4SwaPages(with_comp, mk, caches, 1, dev, &pages);
  CHECK(comp.find("has a compressor") != std::string::npos);
  CHECK(comp.find("#2286") != std::string::npos);

  // 3. AN UNRESOLVED NAME refuses rather than silently dropping that layer.
  FakeIndex missing;
  missing.names = fx.names;
  missing.names[1] = "model.layers.1.attn.some_other_cache";
  const auto mk_missing = missing.Index();
  const std::string unresolved =
      vllm::ResolveDeepseekV4SwaPages(params, mk_missing, caches, 1, dev, &pages);
  CHECK(unresolved.find("no published cache named") != std::string::npos);

  // 4. A WRONG head_size refuses -- the cache would be read at the wrong width.
  auto wrong = caches;
  wrong[2].head_size = HD / 2;
  const std::string bad_width =
      vllm::ResolveDeepseekV4SwaPages(params, mk, wrong, 1, dev, &pages);
  CHECK(bad_width.find("head_size") != std::string::npos);
}

TEST_CASE("W1: two LSE-merged passes equal one pass over the union — sink in EXACTLY one") {
  // `MODEL-DSV4-DSA-COMPOSE` W1 (#2286). Upstream attends a sparse layer with ONE
  // fused two-cache kernel (`flash_mla_with_kvcache(k_cache=swa,
  // extra_k_cache=compressed, ...)`). We have no such kernel, so W1 composes two
  // `vt::MlaDecodeAttention` passes over DISJOINT key sets and merges them with
  // `vt::MergeAttnStates`. This case is the proof that the composition is the
  // same function -- written BEFORE the composition, because if it is not, W1
  // needs a different design.
  //
  // AND IT PINS THE RULE THE DESIGN TURNS ON. A sink is one extra logit in the
  // DENOMINATOR. The merge combines by LSEs, each `log sum exp(scores)`, so a
  // sink seeded into BOTH passes is counted TWICE in the merged denominator --
  // giving a plausible, slightly-too-small output that no token gate would catch.
  // The second half of this case is that double-count, asserted to be WRONG.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = 512, nh = 2, n_keys = 24;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> kv = Rand(static_cast<size_t>(n_keys * hd), 71u, 0.3f);
  const std::vector<float> qv = Rand(static_cast<size_t>(nh * hd), 72u, 0.25f);
  std::vector<float> sink(static_cast<size_t>(nh), -0.15f);
  std::vector<float> none(static_cast<size_t>(nh), -std::numeric_limits<float>::infinity());

  const int64_t block_size = 8;
  const int64_t num_blocks = n_keys / block_size;
  std::vector<float> cache(static_cast<size_t>(num_blocks * block_size * hd), 0.0f);
  for (int64_t j = 0; j < n_keys; ++j)
    for (int64_t d = 0; d < hd; ++d)
      cache[static_cast<size_t>(j * hd + d)] = kv[static_cast<size_t>(j * hd + d)];
  vt::Tensor t_c = Contig(cache.data(), vt::DType::kF32, q.device, {num_blocks, block_size, hd});
  std::vector<int32_t> bt(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) bt[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> sl{static_cast<int32_t>(n_keys)};
  vt::Tensor t_bt = Contig(bt.data(), vt::DType::kI32, q.device, {1, num_blocks});
  vt::Tensor t_sl = Contig(sl.data(), vt::DType::kI32, q.device, {1});
  vt::Tensor t_q = Contig(const_cast<float*>(qv.data()), vt::DType::kF32, q.device, {1, nh, hd});

  // DISJOINT key sets, selected through the op's own selected-slot arm: evens
  // and odds. Their union is every key, so a single pass over all of them is the
  // reference.
  std::vector<int32_t> even, odd;
  for (int32_t j = 0; j < n_keys; ++j) (j % 2 == 0 ? even : odd).push_back(j);
  auto run = [&](const std::vector<int32_t>& sel, const std::vector<float>& sk,
                 std::vector<float>* out, std::vector<float>* lse) {
    std::vector<int32_t> idx = sel;
    std::vector<int32_t> cnt{static_cast<int32_t>(idx.size())};
    vt::Tensor t_idx = Contig(idx.data(), vt::DType::kI32, q.device,
                              {1, static_cast<int64_t>(idx.size())});
    vt::Tensor t_cnt = Contig(cnt.data(), vt::DType::kI32, q.device, {1});
    vt::Tensor t_sk = Contig(const_cast<float*>(sk.data()), vt::DType::kF32, q.device, {nh});
    out->assign(static_cast<size_t>(nh * hd), 0.0f);
    lse->assign(static_cast<size_t>(nh), 0.0f);
    vt::Tensor t_o = Contig(out->data(), vt::DType::kF32, q.device, {1, nh, hd});
    vt::Tensor t_l = Contig(lse->data(), vt::DType::kF32, q.device, {1, nh});
    vt::MlaDecodeAttentionArgs a;
    a.scale = scale;
    a.attn_sink = &t_sk;
    a.topk_indices = &t_idx;
    a.valid_counts = &t_cnt;
    vt::MlaDecodeAttention(q, t_o, &t_l, t_q, t_c, t_bt, t_sl, a);
  };

  std::vector<int32_t> all_keys;
  for (int32_t j = 0; j < n_keys; ++j) all_keys.push_back(j);
  std::vector<float> want, want_lse;
  run(all_keys, sink, &want, &want_lse);   // the reference: ONE pass, ONE sink

  auto merged = [&](const std::vector<float>& sink_b) {
    std::vector<float> oa, la, ob, lb;
    run(even, sink, &oa, &la);      // pass A carries the sink
    run(odd, sink_b, &ob, &lb);     // pass B: -inf (correct) or the sink (the bug)
    std::vector<float> out(static_cast<size_t>(nh * hd), 0.0f);
    vt::Tensor t_out = Contig(out.data(), vt::DType::kF32, q.device, {1, nh, hd});
    vt::Tensor t_pa = Contig(oa.data(), vt::DType::kF32, q.device, {1, nh, hd});
    vt::Tensor t_pb = Contig(ob.data(), vt::DType::kF32, q.device, {1, nh, hd});
    vt::Tensor t_la = Contig(la.data(), vt::DType::kF32, q.device, {nh, 1});
    vt::Tensor t_lb = Contig(lb.data(), vt::DType::kF32, q.device, {nh, 1});
    vt::MergeAttnStates(q, t_out, nullptr, t_pa, t_la, t_pb, t_lb, -1);
    return out;
  };

  // CORRECT: the sink in exactly one pass, the other at -inf (no sink).
  const std::vector<float> good = merged(none);
  double worst = 0.0, mag = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(good[i]));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
    worst = std::max(worst, std::abs(static_cast<double>(want[i] - good[i])));
  }
  REQUIRE(mag > 1e-3);
  CHECK(worst <= 1e-5 * mag);

  // THE DOUBLE-COUNT IS WRONG, and this is why the rule is a rule: seeding the
  // sink into BOTH passes still produces finite, plausible output.
  const std::vector<float> doubled = merged(sink);
  double bad = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(doubled[i]));
    bad = std::max(bad, std::abs(static_cast<double>(want[i] - doubled[i])));
  }
  CHECK(bad > 1e-4 * mag);
}

TEST_CASE("W1: window + compressed merged equals ONE pass over the union") {
  // `MODEL-DSV4-DSA-COMPOSE` W1 (#2286). Upstream attends a compressor layer with
  // ONE fused two-cache call; `MergeWindowAndCompressed` composes it from two
  // passes. The claim is that the composition is the same function, so this
  // builds a cache holding BOTH key sets and compares against a single pass over
  // all of them.
  //
  // T == 1, because the LSE layouts `MergeAttnStates` wants coincide with the
  // decode op's only when T or H is 1 -- the function asserts that rather than
  // assuming it, and a general step needs a transpose it does not yet do.
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const int64_t hd = 512, nh = 3, T = 1, n_win = 6, n_comp = 5;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const std::vector<float> win_kv = Rand(static_cast<size_t>(n_win * hd), 301u, 0.3f);
  const std::vector<float> comp = Rand(static_cast<size_t>(n_comp * hd), 302u, 0.28f);
  const std::vector<float> qv = Rand(static_cast<size_t>(T * nh * hd), 303u, 0.25f);
  std::vector<float> sink(static_cast<size_t>(nh));
  for (int64_t h = 0; h < nh; ++h) sink[static_cast<size_t>(h)] = -0.2f + 0.1f * static_cast<float>(h);

  // (A) THE REFERENCE: one cache holding the window keys THEN the compressed
  //     rows, attended in a single pass with the sink applied once.
  const int64_t n_all = n_win + n_comp;
  const int64_t bs = 16, nb = (n_all + bs - 1) / bs;
  std::vector<float> all(static_cast<size_t>(nb * bs * hd), 0.0f);
  std::copy(win_kv.begin(), win_kv.end(), all.begin());
  std::copy(comp.begin(), comp.end(), all.begin() + n_win * hd);
  vt::Tensor t_all = Contig(all.data(), vt::DType::kF32, q.device, {nb, bs, hd});
  const auto want = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_all, nb, bs, T, nh, hd, /*kv_base=*/n_all - 1, sink, scale,
      /*no_sink=*/false, /*sliding_window=*/0);

  // (B) THE COMPOSITION: a window pass over the window keys only (carrying the
  //     sink and its LSE), then the compressed pass merged in.
  const int64_t nbw = (n_win + bs - 1) / bs;
  std::vector<float> wcache(static_cast<size_t>(nbw * bs * hd), 0.0f);
  std::copy(win_kv.begin(), win_kv.end(), wcache.begin());
  vt::Tensor t_w = Contig(wcache.data(), vt::DType::kF32, q.device, {nbw, bs, hd});
  std::vector<float> win_lse;
  const auto win_out = vllm::deepseek_v4::PagedCausalMlaAttention(
      q, qv, t_w, nbw, bs, T, nh, hd, /*kv_base=*/n_win - 1, sink, scale,
      /*no_sink=*/false, /*sliding_window=*/0, &win_lse);
  REQUIRE(win_lse.size() == static_cast<size_t>(T * nh));

  const auto got = vllm::deepseek_v4::MergeWindowAndCompressed(
      q, win_out, win_lse, qv, comp, n_comp, T, nh, hd, scale);

  REQUIRE(got.size() == want.size());
  double worst = 0.0, mag = 0.0, vs_win = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    REQUIRE(!std::isnan(got[i]));
    mag = std::max(mag, std::abs(static_cast<double>(want[i])));
    worst = std::max(worst, std::abs(static_cast<double>(want[i] - got[i])));
    vs_win = std::max(vs_win, std::abs(static_cast<double>(got[i] - win_out[i])));
  }
  REQUIRE(mag > 1e-3);
  CHECK(worst <= 1e-5 * mag);
  // THE COMPRESSED PASS IS LOAD-BEARING: merging must move the answer away from
  // the window-only result, or the case would pass against a merge that dropped
  // the second contributor entirely.
  CHECK(vs_win > 1e-3 * mag);
}
