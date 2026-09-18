// MODEL-MM-QWEN4-EXP — the deferred Gated DeltaNet V-head permutation
// (ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM).
//
// WHAT IS BEING PROVED. The `qwen4exp` converter writes the GDN V heads TILED
// (`t = r*K + k`) and every GDN kernel indexes them GROUPED (`g = k*R + r`,
// because the recurrence pairs value head g with key head `g / R`). The loader
// used to undo that on the WEIGHT, which forces a k-quant superblock to be
// dequantized -- 61% of this model's whole decode weight traffic. It now leaves
// the five projections verbatim and undoes the order on the projection VECTORS
// instead, through `vt::VHeadPermute`.
//
// THE EQUIVALENCE, and it is an identity of the GEMV, not an approximation:
//   ROW permutation     `(P W) x = P (W x)`  -- permute the OUTPUT
//   COLUMN permutation  `(W P) x = W (P x)`  -- permute the INPUT
//
// THESE TWO ARE NOT EQUALLY EXACT AND THE DIFFERENCE IS MEASURED BELOW, not
// assumed. A ROW permutation moves whole dot products: every output element is
// the SAME sum of the SAME products in the SAME order, so it is bit-identical
// by construction. A COLUMN permutation permutes the summation order INSIDE
// each dot product, and floating-point addition is not associative. The
// `out_proj`-only case is what says which of those this tree actually shows.
//
// THE FIXTURE RATIO IS K = 2, R = 3 AND THAT IS DELIBERATE. At K == 1 the map
// is the IDENTITY and at K == R it is its own INVERSE, so neither can tell a
// correct permutation from no permutation or from the permutation run
// backwards. 2-by-3 is neither, and it is the released model's own ratio
// (16 key heads to 48 value heads).
//
// ─── WHAT THIS SUITE DOES NOT REACH, STATED SO IT IS NOT READ AS COVERAGE ────
//
// (a) THE WIDTH AND THE OPERAND. Every case below runs `value_dim` 96 through
//     the bf16 CPU `Matmul`, on an `out_proj` built `[value_dim, H]` with `nk`
//     UNSET. The released path is `value_dim` 6144 on an `[H, value_dim]`
//     `nk = true` Q6_K operand through `QuantDotGemm*`. The `bad == 0` reading
//     below is therefore a measurement of THIS fixture, not a property of the
//     released reduction; `test_qwen4_exp_layer_loop.cpp`'s deferred subcase is
//     the one that runs the released `[H, value_dim]` `nk = true` orientation.
//
// (b) TWO OF THE FIVE PERMUTE SITES. `GdnBlockPagedForTest` enters
//     `GdnBlockPaged` only, so the out-projection permutation in `GdnBlock`
//     (`qwen3_5.cpp:4741`) and in `GdnBlockPagedMixedSpec` (`:5251`) is
//     UNTESTED here. It is also unreached: `GdnBlock` is called only from
//     qwen3_5's own non-paged layer loop, whose loaders all leave
//     `v_head_perm_key_heads == 0`, and `GdnBlockPagedMixedSpec` is reached
//     from `GdnBlockPaged` only under an active speculator at concurrency > 1,
//     which `qwen4_exp_forward.cpp` does not configure. Both carry the site so
//     that a future speculator or a non-paged qwen4exp arm cannot silently drop
//     it; testing them today would mean testing an unreachable path.
//
// (c) THE COST AXIS. The permutation is O(T * N) per layer per forward, so it
//     is net-negative per DECODE step and net-positive for one large prefill
//     step in isolation (breakeven T ~= 817). Nothing here measures either side
//     — see `LoadGdn` for the arithmetic and `## Owed` for the measurement.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::GdnLayerWeights;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::v1::GDNAttentionMetadata;
using vt::DType;

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed, float lo, float hi) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(lo + u * (hi - lo));
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed,
                      float lo = -0.08f, float hi = 0.08f) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i), lo, hi));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = RandV(seed + static_cast<uint64_t>(i), lo, hi);
  }
  return t;
}

// GROUPED -> TILED, the direction that turns a reference (HuggingFace-order)
// weight into the file's own layout: destination tiled head `t` takes grouped
// head `(t % K) * R + (t / K)`. Composed with `vt::VHeadPermute`'s forward map
// it is the identity, which is the whole claim.
int64_t GroupedForTiled(int64_t t, int64_t kk, int64_t rr) {
  return (t % kk) * rr + (t / kk);
}

// Gather `heads` contiguous blocks of `width` elements each, starting at
// `prefix` elements into every one of `outer` equal segments of `seg` elements.
// One helper covers both weight orientations: for `[H, N]` the head axis is
// last (outer = H, seg = N), and for `out_proj`'s `[value_dim, H]` the head
// axis is first, so one head is `Dv` whole rows -- outer = 1, seg = the whole
// tensor, width = Dv * H.
std::vector<uint16_t> PermuteBf16Blocks(const OwnedTensor& src, int64_t outer,
                                        int64_t seg, int64_t prefix,
                                        int64_t heads, int64_t width,
                                        int64_t kk, int64_t rr) {
  REQUIRE(src.dtype == DType::kBF16);
  const auto* in = reinterpret_cast<const uint16_t*>(src.bytes.data());
  const size_t n = src.bytes.size() / 2;
  REQUIRE(static_cast<int64_t>(n) == outer * seg);
  REQUIRE(prefix + heads * width == seg);
  std::vector<uint16_t> out(n);
  for (int64_t o = 0; o < outer; ++o) {
    const uint16_t* s = in + static_cast<size_t>(o * seg);
    uint16_t* dvec = out.data() + static_cast<size_t>(o * seg);
    for (int64_t i = 0; i < prefix; ++i) dvec[i] = s[i];
    for (int64_t t = 0; t < heads; ++t) {
      const int64_t g = GroupedForTiled(t, kk, rr);
      std::memcpy(dvec + prefix + t * width, s + prefix + g * width,
                  static_cast<size_t>(width) * sizeof(uint16_t));
    }
  }
  return out;
}

void StoreBf16(OwnedTensor& t, const std::vector<uint16_t>& v) {
  REQUIRE(t.bytes.size() == v.size() * 2);
  std::memcpy(t.bytes.data(), v.data(), t.bytes.size());
}

struct Dims {
  int64_t hk = 2, rr = 3, dk = 16, dv = 16, kw = 4, h = 32;
  int64_t hv() const { return hk * rr; }
  int64_t key_dim() const { return hk * dk; }
  int64_t value_dim() const { return hv() * dv; }
  int64_t conv_dim() const { return 2 * key_dim() + value_dim(); }
};

HfConfig MakeConfig(const Dims& d) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.hidden_size = d.h;
  c.num_hidden_layers = 1;
  c.num_experts = 0;
  c.linear_num_key_heads = d.hk;
  c.linear_num_value_heads = d.hv();
  c.linear_key_head_dim = d.dk;
  c.linear_value_head_dim = d.dv;
  c.linear_conv_kernel_dim = d.kw;
  c.rms_norm_eps = 1e-6;
  return c;
}

// The REFERENCE layer: every V-indexed weight already in HuggingFace grouped
// order, `v_head_perm_key_heads == 0`. This is exactly what the loader used to
// produce, and it is the arm the new one has to reproduce.
GdnLayerWeights MakeGroupedWeights(const Dims& d) {
  GdnLayerWeights w;
  w.in_proj_qkv = MakeOwned(DType::kBF16, {d.h, d.conv_dim()}, 10);
  w.in_proj_z = MakeOwned(DType::kBF16, {d.h, d.value_dim()}, 20);
  w.in_proj_b = MakeOwned(DType::kBF16, {d.h, d.hv()}, 30);
  w.in_proj_a = MakeOwned(DType::kBF16, {d.h, d.hv()}, 40);
  w.conv1d_weight = MakeOwned(DType::kBF16, {d.conv_dim(), d.kw}, 50);
  w.a_log = MakeOwned(DType::kF32, {d.hv()}, 60, 0.1f, 1.0f);
  w.dt_bias = MakeOwned(DType::kF32, {d.hv()}, 70, -0.5f, 0.5f);
  w.norm_weight = MakeOwned(DType::kBF16, {d.dv}, 80, 0.5f, 1.5f);
  w.out_proj = MakeOwned(DType::kBF16, {d.value_dim(), d.h}, 90);
  return w;
}

// The DEFERRED layer: the same VALUES, re-indexed into the converter's tiled
// V-head order on exactly the five projections the loader now leaves verbatim,
// with the flag set. `conv1d_weight`, `a_log` and `dt_bias` stay GROUPED --
// that is the loader's rule and it is what confines the run-time permutation to
// the projection boundary.
//
// `which_flipped` inverts the direction on EXACTLY ONE tensor and is the
// RED-FIRST: a vector permuted the wrong way must be visible in the output.
// The five indices are 0 `in_proj_qkv`, 1 `in_proj_z`, 2 `in_proj_b`,
// 3 `in_proj_a`, 4 `out_proj`. `b` and `a` used to share index 2 and flip
// TOGETHER, so neither was ever convicted on its own: a block that permuted
// only one of them reddened anyway on the other's contribution.
GdnLayerWeights MakeTiledWeights(const Dims& d, const GdnLayerWeights& ref,
                                 int which_flipped = -1) {
  GdnLayerWeights w = ref;
  const int64_t kk = d.hk, rr = d.rr;
  // `in_proj_qkv` is [H, conv_dim]: head axis LAST, and only the trailing V
  // columns are tiled -- the leading 2*key_dim q and k columns are the prefix.
  auto k0 = (which_flipped == 0) ? rr : kk;
  auto r0 = (which_flipped == 0) ? kk : rr;
  StoreBf16(w.in_proj_qkv,
            PermuteBf16Blocks(ref.in_proj_qkv, d.h, d.conv_dim(),
                              2 * d.key_dim(), d.hv(), d.dv, k0, r0));
  auto k1 = (which_flipped == 1) ? rr : kk;
  auto r1 = (which_flipped == 1) ? kk : rr;
  StoreBf16(w.in_proj_z, PermuteBf16Blocks(ref.in_proj_z, d.h, d.value_dim(), 0,
                                           d.hv(), d.dv, k1, r1));
  auto k2 = (which_flipped == 2) ? rr : kk;
  auto r2 = (which_flipped == 2) ? kk : rr;
  StoreBf16(w.in_proj_b, PermuteBf16Blocks(ref.in_proj_b, d.h, d.hv(), 0,
                                           d.hv(), 1, k2, r2));
  auto k3 = (which_flipped == 3) ? rr : kk;
  auto r3 = (which_flipped == 3) ? kk : rr;
  StoreBf16(w.in_proj_a, PermuteBf16Blocks(ref.in_proj_a, d.h, d.hv(), 0,
                                           d.hv(), 1, k3, r3));
  // `out_proj` is [value_dim, H]: the permuted axis is the INPUT one, so one
  // head is Dv whole rows of H elements. NOTE the orientation — `nk` is unset
  // and the head axis is FIRST here, while the released loader produces
  // `[H, value_dim]` with `nk = true`, head axis LAST. See header note (a).
  auto k4 = (which_flipped == 4) ? rr : kk;
  auto r4 = (which_flipped == 4) ? kk : rr;
  StoreBf16(w.out_proj,
            PermuteBf16Blocks(ref.out_proj, 1, d.value_dim() * d.h, 0, d.hv(),
                              d.dv * d.h, k4, r4));
  w.v_head_perm_key_heads = kk;
  return w;
}

// T INDEPENDENT single-token decode rows, each at its own state slot. The
// non-spec decode path refuses more than one token per request, and one row
// would leave the per-token addressing of every permuted vector untested.
GDNAttentionMetadata DecodeMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_decodes = static_cast<int>(T);
  g.num_decode_tokens = static_cast<int>(T);
  g.num_actual_tokens = static_cast<int>(T);
  std::vector<int32_t> slots(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slots[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  g.non_spec_state_indices_tensor = slots;
  return g;
}

vt::Queue Q(vt::DeviceType dev) { return vt::Queue{vt::Device{dev, 0}, nullptr}; }

// The block MUTATES `ssm` and `conv` in place and `GdnBlockPagedForTest` copies
// them back. Taking them by value and dropping them on return left the WRITE
// side of both persistent caches unasserted, which is exactly where the spec's
// "both persistent state caches see the grouped order they have always seen"
// claim lives. They are returned beside the output now, and the equality case
// compares all three.
struct LayerRun {
  std::vector<float> out, ssm, conv;
};

LayerRun RunLayer(vt::DeviceType dev, const GdnLayerWeights& w,
                  const HfConfig& c, const Dims& d,
                  const std::vector<float>& h, int64_t T,
                  std::vector<float> ssm, std::vector<float> conv) {
  LayerRun r;
  r.out = vllm::GdnBlockPagedForTest(Q(dev), w, c, h, DecodeMeta(T), ssm, conv,
                                     /*slots=*/T, d.kw - 1, T);
  r.ssm = std::move(ssm);
  r.conv = std::move(conv);
  return r;
}

size_t BitDiffs(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++bad;
  return bad;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

struct Inputs {
  std::vector<float> h, ssm, conv;
};

Inputs MakeInputs(const Dims& d, int64_t T) {
  Inputs in;
  in.h.resize(static_cast<size_t>(T * d.h));
  for (size_t i = 0; i < in.h.size(); ++i) in.h[i] = RandV(1000 + i, -1.0f, 1.0f);
  in.ssm.resize(static_cast<size_t>(T * d.hv() * d.dv * d.dk));
  for (size_t i = 0; i < in.ssm.size(); ++i)
    in.ssm[i] = RandV(2000 + i, -0.5f, 0.5f);
  in.conv.resize(static_cast<size_t>(T * d.conv_dim() * (d.kw - 1)));
  for (size_t i = 0; i < in.conv.size(); ++i)
    in.conv[i] = RandV(3000 + i, -1.0f, 1.0f);
  return in;
}

}  // namespace

// ── (1) the op's two index maps ─────────────────────────────────────────────

TEST_CASE("vt::VHeadPermute: the two index maps, at K=2 R=3 (neither degenerate)") {
  const int64_t kk = 2, rr = 3, heads = kk * rr, width = 3, prefix = 4;
  const int64_t n = prefix + heads * width;
  const int64_t rows = 2;
  std::vector<float> src(static_cast<size_t>(rows * n));
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) + 0.5f;

  vt::Queue q = Q(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  using vllm::dense_attn::DBuf;

  auto run = [&](const std::vector<float>& in, bool inverse) {
    DBuf a(d, DType::kF32, {rows, n}, const_cast<float*>(in.data()));
    DBuf b(d, DType::kF32, {rows, n});
    vt::Tensor o = b.t();
    vt::VHeadPermute(d.q, o, a.t(),
                     vt::VHeadPermuteArgs{prefix, kk, rr, width, inverse});
    std::vector<float> out(in.size());
    b.Download(d, out.data());
    return out;
  };

  SUBCASE("tiled -> grouped is the map the loader used to apply to the weight") {
    // g = k*R + r reads t = r*K + k: 0->0, 1->2, 2->4, 3->1, 4->3, 5->5.
    const int64_t want[6] = {0, 2, 4, 1, 3, 5};
    const std::vector<float> got = run(src, /*inverse=*/false);
    for (int64_t row = 0; row < rows; ++row) {
      for (int64_t i = 0; i < prefix; ++i)
        CHECK(got[static_cast<size_t>(row * n + i)] ==
              src[static_cast<size_t>(row * n + i)]);
      for (int64_t g = 0; g < heads; ++g) {
        CAPTURE(g);
        for (int64_t c = 0; c < width; ++c)
          CHECK(got[static_cast<size_t>(row * n + prefix + g * width + c)] ==
                src[static_cast<size_t>(row * n + prefix + want[g] * width + c)]);
      }
    }
    // NOT the identity, and NOT its own inverse: head 1 reads 2, and the
    // backwards map would have it read 3. Both wrong answers are available.
    CHECK(got[static_cast<size_t>(prefix + 1 * width)] !=
          src[static_cast<size_t>(prefix + 1 * width)]);
    CHECK(got[static_cast<size_t>(prefix + 1 * width)] !=
          src[static_cast<size_t>(prefix + 3 * width)]);
  }

  SUBCASE("grouped -> tiled is the exact inverse, and is NOT the same map") {
    const std::vector<float> fwd = run(src, /*inverse=*/false);
    const std::vector<float> back = run(fwd, /*inverse=*/true);
    for (size_t i = 0; i < src.size(); ++i) CHECK(back[i] == src[i]);
    const std::vector<float> inv = run(src, /*inverse=*/true);
    bool differs = false;
    for (size_t i = 0; i < src.size(); ++i) differs = differs || inv[i] != fwd[i];
    CHECK(differs);
  }

  SUBCASE("an aliasing call is REFUSED, not silently half-gathered") {
    DBuf a(d, DType::kF32, {rows, n}, src.data());
    vt::Tensor same = a.t();
    CHECK_THROWS(vt::VHeadPermute(
        d.q, same, a.t(),
        vt::VHeadPermuteArgs{prefix, kk, rr, width, false}));
  }
}

// ── (2) the equivalence, through the production block ───────────────────────

TEST_CASE("GDN deferred V-head permutation == the load-time one (CPU)") {
  const Dims d;
  const HfConfig c = MakeConfig(d);
  const int64_t T = 3;
  const Inputs in = MakeInputs(d, T);
  const GdnLayerWeights grouped = MakeGroupedWeights(d);
  const GdnLayerWeights tiled = MakeTiledWeights(d, grouped);

  REQUIRE(grouped.v_head_perm_key_heads == 0);
  REQUIRE(tiled.v_head_perm_key_heads == d.hk);

  const LayerRun ref =
      RunLayer(vt::DeviceType::kCPU, grouped, c, d, in.h, T, in.ssm, in.conv);
  const LayerRun got =
      RunLayer(vt::DeviceType::kCPU, tiled, c, d, in.h, T, in.ssm, in.conv);

  // THE BAR IS EQUALITY, not a tolerance: this is a re-indexing.
  //
  // The four ROW permutations are bit-identical by construction (whole dot
  // products move, the summands inside each do not). `out_proj`'s COLUMN
  // permutation reorders the summation inside every dot product, so this
  // assertion is where that either holds or is measured.
  //
  // AND IT IS A MEASUREMENT OF THIS FIXTURE, ON BOTH AXES. `value_dim` is 96
  // here, not the released 6144, so the reordered dot product is 64x shorter
  // than the one that ships; and `out_proj` is `[value_dim, H]` with `nk` unset
  // through the bf16 CPU `Matmul`, not the released `[H, value_dim]`
  // `nk = true` Q6_K operand through `QuantDotGemm*`. A `bad == 0` here does
  // not extend to either. Header note (a).
  const size_t bad = BitDiffs(got.out, ref.out);
  const float worst = MaxAbsDiff(got.out, ref.out);
  CAPTURE(bad);
  CAPTURE(worst);
  CHECK(bad == 0);

  // THE WRITE SIDE OF BOTH PERSISTENT CACHES, which the output alone cannot
  // see: the spec's rule is that `conv1d`, `a_log` and `dt_bias` stay permuted
  // at LOAD precisely so the conv state and the SSM state keep the grouped
  // order every older build wrote. If the deferred arm moved either cache's
  // channel order, a resumed request would read its own state re-indexed.
  CHECK(BitDiffs(got.ssm, ref.ssm) == 0);
  CHECK(BitDiffs(got.conv, ref.conv) == 0);
  // And the caches were actually WRITTEN — an all-zero pair would make the two
  // assertions above agree about nothing.
  CHECK(BitDiffs(got.ssm, in.ssm) > 0);
  CHECK(BitDiffs(got.conv, in.conv) > 0);
}

TEST_CASE("GDN deferred V-head permutation: a vector permuted the WRONG way is seen") {
  // RED-FIRST. Each subcase flips ONE tensor's direction, which at K=2, R=3 is
  // a genuinely different map (it is the same map only at K == R). If the block
  // did not actually apply the permutation, every one of these would pass by
  // matching the reference anyway -- so a green here is the proof that the
  // permutation is load-bearing on every one of the five.
  const Dims d;
  const HfConfig c = MakeConfig(d);
  const int64_t T = 3;
  const Inputs in = MakeInputs(d, T);
  const GdnLayerWeights grouped = MakeGroupedWeights(d);
  const LayerRun ref =
      RunLayer(vt::DeviceType::kCPU, grouped, c, d, in.h, T, in.ssm, in.conv);

  // FIVE, NOT FOUR: `in_proj_b` (2) and `in_proj_a` (3) are separate indices
  // now. They shared one index before, flipped together, and a case that flips
  // two tensors convicts neither of them alone.
  for (int which : {0, 1, 2, 3, 4}) {
    CAPTURE(which);
    const GdnLayerWeights bad_w = MakeTiledWeights(d, grouped, which);
    const LayerRun got =
        RunLayer(vt::DeviceType::kCPU, bad_w, c, d, in.h, T, in.ssm, in.conv);
    CHECK(BitDiffs(got.out, ref.out) > 0);
  }
}
