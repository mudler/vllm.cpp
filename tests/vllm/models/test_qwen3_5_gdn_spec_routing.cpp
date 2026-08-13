// SPEC-MTP I5a — GDN layer spec routing (GdnBlockPaged spec branch).
//
// Proves that GdnBlockPaged's num_spec_decodes>0 branch (mirror of
// vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1329-1576 @
// e24d1b24) routes a PURE spec batch through vt::CausalConv1dSpecUpdate +
// vt::GdnSpecDecode with the correct per-request slots / cu_seqlens /
// num_accepted, BIT-FOR-BIT against the shipped non-spec decode path.
//
// THE REFERENCE (the "I4 ops applied directly" chain). A speculative step over a
// single request with num_accepted==1 processes its 1+k query tokens
// sequentially from the request's initial state, snapshotting each timestep.
// That is EXACTLY a token-by-token run of the shipped non-spec decode path
// (vt::GdnDecode + vt::CausalConv1dUpdate — the ops I4 proved the spec kernels
// reduce to at T==1/accepted==1) carrying the state forward. So one spec call
// over T tokens must equal T single-token decode calls over the same weights and
// the same initial state, row for row. A mis-wired split / slot select / conv
// window / merge diverges immediately. Real gate-checkpoint GDN dims (Hv 48/32,
// Dk==Dv 128, conv width 4). On CPU the projection GEMM is batch-invariant so the
// match is BIT-EXACT; the CUDA build additionally proves the spec ops execute
// on-device (asserted within a tight bf16-ULP band, since the M=1 vs M=T
// projection GEMM may retile — the spec routing itself is exact).

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/platforms/interface.h"  // supports_fp8() gates the fp8 GDN tail
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
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
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(lo + u * (hi - lo));
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed, float lo = -0.08f,
                      float hi = 0.08f) {
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
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i), lo, hi);
  }
  return t;
}

struct GdnDims {
  int64_t hk, hv, dk, dv, kw;
  const char* name;
};

// Every case below runs at BOTH gate dims, so a failure log has to say which.
// `CAPTURE(g.name)` does NOT: doctest 2.5.2 stringifies a `const char*` through
// its generic path and prints `g.name := 1` (the pointer decayed to bool), so
// the one thing the capture exists to disambiguate is exactly what is lost.
// Wrap in std::string, which has a real stringifier. Applies to `INFO(... <<
// ptr)` too — the `<<` form goes through the same DOCTEST_STRINGIFY.

// Minimal dense (num_experts==0) GDN config at the real gate-checkpoint GDN dims.
HfConfig MakeConfig(const GdnDims& g, int64_t H) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.hidden_size = H;
  c.num_hidden_layers = 1;
  c.num_experts = 0;
  c.linear_num_key_heads = g.hk;
  c.linear_num_value_heads = g.hv;
  c.linear_key_head_dim = g.dk;
  c.linear_value_head_dim = g.dv;
  c.linear_conv_kernel_dim = g.kw;
  c.rms_norm_eps = 1e-6;
  return c;
}

GdnLayerWeights MakeGdnWeights(const HfConfig& c) {
  const int64_t H = c.hidden_size, Hv = c.linear_num_value_heads;
  const int64_t key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
  const int64_t value_dim = Hv * c.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t Kw = c.linear_conv_kernel_dim;
  GdnLayerWeights w;
  w.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, 10);
  w.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, 20);
  w.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, 30);
  w.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, 40);
  w.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, 50);
  // A_log / dt_bias positive-ish so g = -exp(A_log)*softplus(...) is a sane decay.
  w.a_log = MakeOwned(DType::kF32, {Hv}, 60, 0.1f, 1.0f);
  w.dt_bias = MakeOwned(DType::kF32, {Hv}, 70, -0.5f, 0.5f);
  w.norm_weight = MakeOwned(DType::kBF16, {c.linear_value_head_dim}, 80, 0.5f, 1.5f);
  w.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, 90);
  return w;
}

// One fresh single-token decode step at state slot 0.
GDNAttentionMetadata DecodeMeta() {
  GDNAttentionMetadata g;
  g.num_decodes = 1;
  g.num_decode_tokens = 1;
  g.num_actual_tokens = 1;
  g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  return g;
}

// One PURE spec request over T tokens (k = T-1) writing snapshots into slots
// 0..T-1, reading its initial state from slot 0 (num_accepted == 1).
GDNAttentionMetadata SpecMeta(int64_t T) {
  GDNAttentionMetadata g;
  g.num_spec_decodes = 1;
  g.num_spec_decode_tokens = static_cast<int>(T);
  g.num_actual_tokens = static_cast<int>(T);
  g.spec_state_indices_num_cols = static_cast<int>(T);  // num_spec + 1
  std::vector<int32_t> ssi(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) ssi[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  g.spec_state_indices_tensor = ssi;
  g.spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  g.spec_sequence_masks = std::vector<uint8_t>{1};
  g.spec_token_indx = ssi;  // identity gather over the pure batch
  g.num_accepted_tokens = std::vector<int32_t>{1};
  return g;
}

vt::Queue Q(vt::DeviceType dev) { return vt::Queue{vt::Device{dev, 0}, nullptr}; }

void RunSpecRoutingCase(vt::DeviceType dev, const GdnDims& g, bool bit_exact) {
  const int64_t H = 128;
  const int64_t T = 4;              // 1 + k, k = 3 draft tokens
  const HfConfig c = MakeConfig(g, H);
  const GdnLayerWeights w = MakeGdnWeights(c);
  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;

  // Shared inputs and initial state.
  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(1000 + i, -1.0f, 1.0f);
  std::vector<float> ssm0(static_cast<size_t>(ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(2000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(conv_dim * (Kw - 1)));
  for (size_t i = 0; i < conv0.size(); ++i) conv0[i] = RandV(3000 + i, -1.0f, 1.0f);

  // ── Reference: T single-token decode calls, carrying state (narrow conv). ──
  std::vector<float> ref_out;
  {
    const int64_t slots = 1;
    std::vector<float> ssm(static_cast<size_t>(slots * ssm_row));
    std::memcpy(ssm.data(), ssm0.data(), ssm0.size() * sizeof(float));
    std::vector<float> conv(static_cast<size_t>(slots * conv_dim * (Kw - 1)));
    std::memcpy(conv.data(), conv0.data(), conv0.size() * sizeof(float));
    for (int64_t t = 0; t < T; ++t) {
      std::vector<float> h1(h.begin() + static_cast<std::ptrdiff_t>(t * H),
                            h.begin() + static_cast<std::ptrdiff_t>((t + 1) * H));
      GDNAttentionMetadata dm = DecodeMeta();
      std::vector<float> row = vllm::GdnBlockPagedForTest(Q(dev), w, c, h1, dm, ssm, conv,
                                                          slots, Kw - 1, /*T=*/1);
      ref_out.insert(ref_out.end(), row.begin(), row.end());
    }
  }

  // ── Spec: one call over T tokens, num_accepted == 1, widened conv state. ──
  std::vector<float> spec_out;
  {
    const int64_t slots = T;                 // one slot per timestep snapshot
    const int64_t conv_len = (Kw - 1) + (T - 1);  // widened spec row
    std::vector<float> ssm(static_cast<size_t>(slots * ssm_row), 0.0f);
    // Initial state lives in slot 0 (num_accepted-1 == 0).
    std::memcpy(ssm.data(), ssm0.data(), ssm0.size() * sizeof(float));
    std::vector<float> conv(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
    // Slot 0's first K-1 taps are the initial conv window; the rest are unread.
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv[static_cast<size_t>(ch * conv_len + j)] =
            conv0[static_cast<size_t>(ch * (Kw - 1) + j)];
    GDNAttentionMetadata sm = SpecMeta(T);
    spec_out = vllm::GdnBlockPagedForTest(Q(dev), w, c, h, sm, ssm, conv, slots, conv_len, T);
  }

  REQUIRE(spec_out.size() == ref_out.size());
  if (bit_exact) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < spec_out.size(); ++i)
      if (std::memcmp(&spec_out[i], &ref_out[i], sizeof(float)) != 0) {
        if (bad == 0) first = i;
        ++bad;
      }
    INFO("dims := ", std::string(g.name));
    CAPTURE(bad);
    CAPTURE(first);
    if (bad != 0) {
      CAPTURE(spec_out[first]);
      CAPTURE(ref_out[first]);
    }
    CHECK(bad == 0);
  } else {
    float maxabs = 0.0f;
    for (size_t i = 0; i < spec_out.size(); ++i)
      maxabs = std::max(maxabs, std::fabs(spec_out[i] - ref_out[i]));
    INFO("dims := ", std::string(g.name));
    CAPTURE(maxabs);
    // Tight band: only the M=1-vs-M=T projection GEMM retile differs; the spec
    // conv/recurrence routing is exact. A broken split would blow past this.
    CHECK(maxabs < 0.05f);
  }
}

// ── MIXED spec+non-spec batch metadata: one spec request (2 tokens, k=1, state
// slots 0,1) followed by one PREFILL request (Tp tokens, state slot 2). Mirrors
// the GDN metadata builder's mixed output (gdn_attn.cpp:218-333). ──
GDNAttentionMetadata MixedMeta(int Tp) {
  const int Ts = 2;  // 1 + k, k = 1
  GDNAttentionMetadata g;
  g.num_spec_decodes = 1;
  g.num_spec_decode_tokens = Ts;
  g.num_prefills = 1;
  g.num_prefill_tokens = Tp;
  g.num_decodes = 0;
  g.num_decode_tokens = 0;
  g.num_actual_tokens = Ts + Tp;
  g.spec_state_indices_num_cols = 2;
  g.spec_state_indices_tensor = std::vector<int32_t>{0, 1};
  g.spec_query_start_loc = std::vector<int32_t>{0, Ts};
  g.spec_sequence_masks = std::vector<uint8_t>{1, 0};
  g.spec_token_indx = std::vector<int32_t>{0, 1};
  std::vector<int32_t> nst;
  for (int t = Ts; t < Ts + Tp; ++t) nst.push_back(t);
  g.non_spec_token_indx = nst;
  g.num_accepted_tokens = std::vector<int32_t>{1};
  g.non_spec_state_indices_tensor = std::vector<int32_t>{2};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, Tp};
  g.has_initial_state = std::vector<uint8_t>{0};
  g.prefill_state_indices = std::vector<int32_t>{2};
  g.prefill_query_start_loc = std::vector<int32_t>{0, Tp};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// One fresh PURE prefill request over Tp tokens at state slot `slot`.
GDNAttentionMetadata PrefillMeta(int Tp, int slot) {
  GDNAttentionMetadata g;
  g.num_prefills = 1;
  g.num_prefill_tokens = Tp;
  g.num_actual_tokens = Tp;
  g.non_spec_state_indices_tensor = std::vector<int32_t>{slot};
  g.non_spec_query_start_loc = std::vector<int32_t>{0, Tp};
  g.has_initial_state = std::vector<uint8_t>{0};
  g.prefill_state_indices = std::vector<int32_t>{slot};
  g.prefill_query_start_loc = std::vector<int32_t>{0, Tp};
  g.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv =
      vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
  g.batch_ptr = conv.batch_ptr;
  g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  return g;
}

// THE MIXED-BATCH PROOF (model-independent). The mixed spec+non-spec batch
// processes the spec request and the prefill request over DISJOINT state slots,
// so its per-row output MUST equal the spec request run as a PURE spec batch
// (rows 0..1) followed by the prefill request run as a PURE prefill batch
// (rows 2..). A mis-wired index_select split, a wrong sub-batch metadata feed,
// or a mis-indexed index_copy merge diverges immediately. This is independent
// of any model-level bf16 batch-nondeterminism (the e2e c>1 confound): the
// CPU projection GEMM is row-invariant, so the split/merge is BIT-EXACT.
void RunMixedRoutingCase(vt::DeviceType dev, const GdnDims& g, bool bit_exact) {
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);  // mixed needs widened indexed IO
  const int64_t H = 128;
  const int Ts = 2, Tp = 3, T = Ts + Tp;
  const HfConfig c = MakeConfig(g, H);
  const GdnLayerWeights w = MakeGdnWeights(c);
  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;
  const int64_t conv_len = (Kw - 1) + 1;  // widened spec row (k = 1)
  const int64_t slots = 3;

  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(5000 + i, -1.0f, 1.0f);
  // Initial state: slot 0 = the spec request's initial (read at num_accepted-1);
  // slot 2 = the prefill request (fresh — zeroed by has_initial_state==0).
  std::vector<float> ssm0(static_cast<size_t>(slots * ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(6000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
  for (int64_t s = 0; s < slots; ++s)
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv0[static_cast<size_t>((s * conv_dim + ch) * conv_len + j)] =
            RandV(7000 + (s * conv_dim + ch) * (Kw - 1) + j, -1.0f, 1.0f);

  // ── Mixed run. ──
  std::vector<float> ssm_m = ssm0, conv_m = conv0;
  const std::vector<float> mixed_out = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h, MixedMeta(Tp), ssm_m, conv_m, slots, conv_len, T);

  // ── Reference: spec rows via the PURE spec path (slot 0 initial). ──
  std::vector<float> ssm_s = ssm0, conv_s = conv0;
  std::vector<float> h_spec(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(Ts * H));
  const std::vector<float> spec_ref = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h_spec, SpecMeta(Ts), ssm_s, conv_s, slots, conv_len, Ts);

  // ── Reference: prefill rows via the PURE prefill path (fresh slot 2). ──
  std::vector<float> ssm_p = ssm0, conv_p = conv0;
  std::vector<float> h_pf(h.begin() + static_cast<std::ptrdiff_t>(Ts * H), h.end());
  const std::vector<float> pf_ref = vllm::GdnBlockPagedForTest(
      Q(dev), w, c, h_pf, PrefillMeta(Tp, 2), ssm_p, conv_p, slots, conv_len, Tp);

  REQUIRE(static_cast<int64_t>(mixed_out.size()) == T * H);
  REQUIRE(static_cast<int64_t>(spec_ref.size()) == Ts * H);
  REQUIRE(static_cast<int64_t>(pf_ref.size()) == Tp * H);
  std::vector<float> ref;
  ref.insert(ref.end(), spec_ref.begin(), spec_ref.end());
  ref.insert(ref.end(), pf_ref.begin(), pf_ref.end());

  if (bit_exact) {
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < mixed_out.size(); ++i)
      if (std::memcmp(&mixed_out[i], &ref[i], sizeof(float)) != 0) {
        if (bad == 0) first = i;
        ++bad;
      }
    INFO("dims := ", std::string(g.name));
    CAPTURE(bad);
    CAPTURE(first);
    if (bad != 0) { CAPTURE(mixed_out[first]); CAPTURE(ref[first]); }
    CHECK(bad == 0);
  } else {
    float maxabs = 0.0f;
    for (size_t i = 0; i < mixed_out.size(); ++i)
      maxabs = std::max(maxabs, std::fabs(mixed_out[i] - ref[i]));
    INFO("dims := ", std::string(g.name));
    CAPTURE(maxabs);
    CHECK(maxabs < 0.05f);
  }
}

constexpr GdnDims kGate27B{16, 48, 128, 128, 4, "27B (Hv=48)"};
constexpr GdnDims kGate35B{16, 32, 128, 128, 4, "35B (Hv=32)"};

}  // namespace

TEST_CASE("GDN spec routing (CPU): pure spec batch == token-sequential decode chain") {
  RunSpecRoutingCase(vt::DeviceType::kCPU, kGate27B, /*bit_exact=*/true);
  RunSpecRoutingCase(vt::DeviceType::kCPU, kGate35B, /*bit_exact=*/true);
}

TEST_CASE("GDN MIXED spec+prefill routing (CPU): mixed batch == pure spec + pure prefill") {
  RunMixedRoutingCase(vt::DeviceType::kCPU, kGate27B, /*bit_exact=*/true);
  RunMixedRoutingCase(vt::DeviceType::kCPU, kGate35B, /*bit_exact=*/true);
}

#ifdef VLLM_CPP_CUDA
TEST_CASE("GDN spec routing (CUDA): spec branch runs on-device and matches decode chain") {
  vt::GetBackend(vt::DeviceType::kCUDA);  // skip cleanly if no device
  RunSpecRoutingCase(vt::DeviceType::kCUDA, kGate27B, /*bit_exact=*/false);
  RunSpecRoutingCase(vt::DeviceType::kCUDA, kGate35B, /*bit_exact=*/false);
}

TEST_CASE("GDN MIXED spec+prefill routing (CUDA): mixed batch matches pure spec + prefill") {
  vt::GetBackend(vt::DeviceType::kCUDA);
  RunMixedRoutingCase(vt::DeviceType::kCUDA, kGate27B, /*bit_exact=*/false);
  RunMixedRoutingCase(vt::DeviceType::kCUDA, kGate35B, /*bit_exact=*/false);
}

// PERF-27B-GDN-FP8-QKVZ, the numerical contract: ONE merged fp8 GEMM over the
// N-concatenated [qkv;z] operand must produce EXACTLY the concatenation of the
// two legacy per-shard fp8 GEMM outputs. Both arms run in one process over one
// uploaded activation and one set of resident bytes, so the comparison is
// bitwise. The merged output is f32 (the dtype the split mixed_qkv GEMM already
// emits) and z is cast to the split arm's own output dtype, so nothing about the
// split arithmetic is traded for the shape change.
//
// Two weight-scale regimes, because they take different code paths inside the
// merged GEMM: EQUAL folded alphas fold into the GEMM scalar, DIFFERENT ones go
// through the resident per-output-column alpha vector.
//
// AT THE GATE SHAPES, and only there, per the spec's Tests §2. The shape is
// load-bearing and not an incidental parameter. The production fp8 GEMM is
// cuBLASLt (`VT_DENSE_CUBLASLT_FP8`, default ON), and its kernel is chosen per
// (M,N,K) by cublasLtMatmulAlgoGetHeuristic — a choice that includes the SPLIT-K
// factor, i.e. how many partial K-reductions get summed afterwards. Measured on
// `sm_121a` with VT_GEMM_ALGO_LOG=1:
//
//   27B  M=1  merged n=16384 k=5120 splitK=1 | qkv n=10240 splitK=1 | z n=6144 splitK=1
//   toy  M=3  merged n=320   k=256  splitK=1 | qkv n=192   splitK=4 | z n=128  splitK=8
//
// At the gate shapes all three GEMMs reduce K in ONE pass, so the merge is
// bitwise exact (measured 0/16384 differing at 27B M=1, 0/49152 at M=3). At a
// toy N the heuristic splits K four ways for one shard and eight ways for the
// other while the wider merged N stays at one — three DIFFERENT summation
// orders, and no single merged launch can reproduce two of them at once, so
// bitwise equality is unattainable there for a reason that has nothing to do
// with this code. That was confirmed by rerunning the toy shape on the
// shape-invariant CUTLASS fp8 GEMM (VT_DENSE_CUBLASLT_FP8=0), where merged and
// split agree bitwise in all four regimes; under cuBLASLt they differ by at
// most 128 ULP / 8.4e-06 relative, the size of f32 reassociation and nothing
// more. So the precondition this case rests on is: the gate shapes' N and K are
// large enough that cuBLASLt needs no split-K to fill the device. If a future
// cuBLASLt or driver starts splitting K here, this case goes red — which is the
// correct signal, because the merged and split arms would then genuinely no
// longer be the same arithmetic.
TEST_CASE("GDN merged FP8 qkvz == the two split fp8 GEMMs, bitwise") {
  vt::GetBackend(vt::DeviceType::kCUDA);  // skip cleanly if no device

  // The real GDN input-projection shapes of both gate checkpoints:
  // nvidia/Qwen3.6-27B-NVFP4@0893e160 (hidden 5120, Hk16/Dk128, Hv48/Dv128) and
  // nvidia/Qwen3.6-35B-A3B-NVFP4@491c2f1e (hidden 2048, Hv32) — the same FP8
  // tower, and the one the spec requires be run alongside the 27B.
  struct GateShape {
    GdnDims dims;
    int64_t hidden;
  };
  const GateShape gates[] = {{kGate27B, 5120}, {kGate35B, 2048}};

  const float shared_input_scale = 0.0078125F;
  for (const GateShape& gate : gates) {
    const int64_t H = gate.hidden;
    const int64_t value_dim = gate.dims.hv * gate.dims.dv;
    const int64_t conv_dim = 2 * gate.dims.hk * gate.dims.dk + value_dim;

    const auto make_fp8 = [&](int64_t n, uint64_t seed, float weight_scale) {
      vllm::Fp8Weight f;
      f.n = n;
      f.k = H;
      f.input_scale = shared_input_scale;
      f.weight_scale = weight_scale;
      f.alpha = shared_input_scale * weight_scale;
      f.packed.dtype = DType::kI8;
      f.packed.rank = 2;
      f.packed.shape[0] = n;
      f.packed.shape[1] = H;
      f.packed.bytes.resize(static_cast<size_t>(n * H));
      auto* bytes = f.packed.bytes.data();
      for (int64_t i = 0; i < n * H; ++i) {
        // e4m3 byte patterns with the sign/exponent bits exercised; 0x7f/0xff
        // are NaN in e4m3fn, so keep the mantissa/exponent below that.
        bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(
            Mix(seed + static_cast<uint64_t>(i)) % 0x7EU);
      }
      return f;
    };

    for (const bool same_weight_scale : {true, false}) {
      // One set of resident bytes per weight-scale regime, reused across the
      // token counts below: the shards are tens of megabytes at these shapes.
      GdnLayerWeights w;
      w.in_proj_qkv_fp8 = make_fp8(conv_dim, 101, 0.00390625F);
      w.in_proj_z_fp8 =
          make_fp8(value_dim, 202, same_weight_scale ? 0.00390625F : 0.015625F);

      // T=1 is the decode step this row is measured on; T=3 is a short prefill,
      // which the heuristic sees as a different M and may answer differently.
      for (const int64_t T : {int64_t{1}, int64_t{3}}) {
        std::vector<float> h(static_cast<size_t>(T * H));
        for (size_t i = 0; i < h.size(); ++i)
          h[i] = RandV(9000 + static_cast<uint64_t>(i), -0.5F, 0.5F);
        for (const bool z_bf16 : {true, false}) {
          CAPTURE(H);
          CAPTURE(conv_dim);
          CAPTURE(value_dim);
          CAPTURE(same_weight_scale);
          CAPTURE(T);
          CAPTURE(z_bf16);
          const std::vector<float> merged = vllm::ProjectGdnFp8QkvzForTest(
              Q(vt::DeviceType::kCUDA), w, h, T, conv_dim, value_dim,
              /*merged=*/true, z_bf16);
          const std::vector<float> split = vllm::ProjectGdnFp8QkvzForTest(
              Q(vt::DeviceType::kCUDA), w, h, T, conv_dim, value_dim,
              /*merged=*/false, z_bf16);
          REQUIRE(merged.size() == split.size());
          size_t bad = 0;
          size_t first_bad = 0;
          for (size_t i = 0; i < merged.size(); ++i) {
            if (std::memcmp(&merged[i], &split[i], sizeof(float)) != 0) {
              if (bad == 0) first_bad = i;
              ++bad;
            }
          }
          if (bad != 0) {
            CAPTURE(first_bad);
            CAPTURE(merged[first_bad]);
            CAPTURE(split[first_bad]);
          }
          CHECK(bad == 0);
        }
      }
    }
  }
}

// The merged operand must exist BEFORE the first forward — a resident built
// inside a CUDA-graph capture allocates and copies mid-capture, which aborts
// the capture. PrepareGdnFp8Resident is registered on the dense prepare hook and
// is what guarantees it; deleting that call leaves d_qkvz_fp8_packed null here.
TEST_CASE("GDN merged FP8 qkvz resident is built pre-capture, at prepare") {
  vt::GetBackend(vt::DeviceType::kCUDA);  // skip cleanly if no device
  const int64_t H = 256;
  const HfConfig c = MakeConfig(kGate27B, H);
  const int64_t value_dim = kGate27B.hv * kGate27B.dv;
  const int64_t conv_dim = 2 * kGate27B.hk * kGate27B.dk + value_dim;

  const auto make_fp8 = [&](int64_t n, uint64_t seed) {
    vllm::Fp8Weight f;
    f.n = n;
    f.k = H;
    f.input_scale = 0.0078125F;
    f.weight_scale = 0.00390625F;
    f.alpha = f.input_scale * f.weight_scale;
    f.packed.dtype = DType::kI8;
    f.packed.rank = 2;
    f.packed.shape[0] = n;
    f.packed.shape[1] = H;
    f.packed.bytes.resize(static_cast<size_t>(n * H));
    auto* bytes = f.packed.bytes.data();
    for (int64_t i = 0; i < n * H; ++i)
      bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(
          Mix(seed + static_cast<uint64_t>(i)) % 0x7EU);
    return f;
  };

  vllm::Qwen3_5DenseWeights weights;
  vllm::Qwen3_5DenseLayerWeights layer;
  layer.is_linear_attention = true;
  layer.gdn.in_proj_qkv_fp8 = make_fp8(conv_dim, 11);
  layer.gdn.in_proj_z_fp8 = make_fp8(value_dim, 22);
  weights.layers.push_back(std::move(layer));

  REQUIRE_FALSE(static_cast<bool>(weights.layers[0].gdn.d_qkvz_fp8_packed));
  vt::Queue q = Q(vt::DeviceType::kCUDA);
  vllm::Qwen3_5DenseModel::PrepareGdnFp8Resident(weights, c, q);
  CHECK(static_cast<bool>(weights.layers[0].gdn.d_qkvz_fp8_packed));
  // Equal folded alphas: the alpha vector is folded into the GEMM scalar and no
  // second resident is allocated.
  CHECK_FALSE(static_cast<bool>(weights.layers[0].gdn.d_qkvz_fp8_alpha));
}
#endif

namespace {

// The gate ACTIVATION must reach the PAGED GDN tails, not just the eager one.
//
// Upstream qwen_gdn_linear_attn.py:452-464 @555967922 resolves
// `output_gate_type` once per layer and hands it to RMSNormGated as
// `activation=`; vt::RmsNormGatedArgs::sigmoid_gate is the same switch. Our
// paged forward has TWO independently wired gated-RMSNorm tails --
// GdnBlockPaged's shared tail and GdnBlockPagedMixedSpec's own -- so each is
// driven here. Every gate checkpoint we own resolves to silu, so a silu-only
// corpus cannot see either wiring being absent: the proof has to be that the
// sigmoid arm MOVES the output.
//
// `mixed=false` runs the pure-prefill batch (GdnBlockPaged's tail);
// `mixed=true` runs the spec+prefill batch (GdnBlockPagedMixedSpec's tail).
void RunGateActivationCase(const GdnDims& g, bool mixed) {
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);  // mixed needs widened indexed IO
  const int64_t H = 128;
  const int Ts = 2, Tp = 3;
  const int64_t T = mixed ? Ts + Tp : Tp;
  const HfConfig silu = MakeConfig(g, H);
  // MakeConfig never sets the key: the struct default IS the upstream default.
  REQUIRE(silu.output_gate_type == "silu");
  HfConfig sigmoid = MakeConfig(g, H);
  sigmoid.output_gate_type = "sigmoid";

  const GdnLayerWeights w = MakeGdnWeights(silu);
  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;
  const int64_t conv_len = (Kw - 1) + 1;  // widened spec row (k = 1)
  const int64_t slots = 3;

  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(5000 + i, -1.0f, 1.0f);
  std::vector<float> ssm0(static_cast<size_t>(slots * ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(6000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
  for (int64_t s = 0; s < slots; ++s)
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv0[static_cast<size_t>((s * conv_dim + ch) * conv_len + j)] =
            RandV(7000 + (s * conv_dim + ch) * (Kw - 1) + j, -1.0f, 1.0f);

  const auto run = [&](const HfConfig& c) {
    std::vector<float> ssm = ssm0, conv = conv0;
    const GDNAttentionMetadata meta = mixed ? MixedMeta(Tp) : PrefillMeta(Tp, 2);
    return vllm::GdnBlockPagedForTest(Q(vt::DeviceType::kCPU), w, c, h, meta, ssm,
                                      conv, slots, conv_len, T);
  };

  const std::vector<float> silu_out = run(silu);
  const std::vector<float> sigmoid_out = run(sigmoid);
  REQUIRE(static_cast<int64_t>(silu_out.size()) == T * H);
  REQUIRE(sigmoid_out.size() == silu_out.size());

  double maxd = 0.0;
  for (size_t i = 0; i < silu_out.size(); ++i)
    maxd = std::max(maxd,
                    std::abs(static_cast<double>(silu_out[i]) - sigmoid_out[i]));
  INFO("dims := ", std::string(g.name));
  CAPTURE(mixed);
  CAPTURE(maxd);
  CHECK(maxd > 0.0);

  // The same arm twice is BIT-identical, so the delta above is the gate and not
  // run-to-run noise.
  const std::vector<float> silu_again = run(silu);
  size_t bad = 0;
  for (size_t i = 0; i < silu_out.size(); ++i)
    if (std::memcmp(&silu_out[i], &silu_again[i], sizeof(float)) != 0) ++bad;
  CAPTURE(bad);
  CHECK(bad == 0);
}

// WHICH activation each paged tail applies, not merely that the arms differ.
//
// RunGateActivationCase proves the outputs MOVE; it would pass unchanged with
// the boolean INVERTED, i.e. a silu checkpoint driving the sigmoid kernel --
// this row's bug class turned inside out. The polarity therefore needs a
// reference that never consults our gate.
//
// Arithmetic supplies one: silu(0) = 0 * sigmoid(0) = 0 EXACTLY, while
// sigmoid(0) = 0.5. The gate input is z = h @ in_proj_z, a plain GEMM with no
// bias (qwen3_5.cpp ProjectGdnQkvz), so a zeroed in_proj_z makes z identically
// zero. A SILU tail then computes norm(core) * 0 = 0 and the block returns
// 0 @ out_proj = ZERO, whatever the recurrence produced; a SIGMOID tail
// computes 0.5 * norm(core) and returns something non-zero. Inverting the
// resolution flips both assertions at once.
void RunGatePolarityCase(const GdnDims& g, bool mixed) {
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);  // mixed needs widened indexed IO
  const int64_t H = 128;
  const int Ts = 2, Tp = 3;
  const int64_t T = mixed ? Ts + Tp : Tp;
  const HfConfig silu = MakeConfig(g, H);
  REQUIRE(silu.output_gate_type == "silu");
  HfConfig sigmoid = MakeConfig(g, H);
  sigmoid.output_gate_type = "sigmoid";

  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;
  const int64_t conv_len = (Kw - 1) + 1;
  const int64_t slots = 3;

  GdnLayerWeights w = MakeGdnWeights(silu);
  // z ≡ 0. lo == hi == 0 makes every element exactly +0.0f.
  w.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, 20, 0.0f, 0.0f);

  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(5000 + i, -1.0f, 1.0f);
  std::vector<float> ssm0(static_cast<size_t>(slots * ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(6000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
  for (int64_t s = 0; s < slots; ++s)
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv0[static_cast<size_t>((s * conv_dim + ch) * conv_len + j)] =
            RandV(7000 + (s * conv_dim + ch) * (Kw - 1) + j, -1.0f, 1.0f);

  const auto run = [&](const HfConfig& c) {
    std::vector<float> ssm = ssm0, conv = conv0;
    const GDNAttentionMetadata meta = mixed ? MixedMeta(Tp) : PrefillMeta(Tp, 2);
    return vllm::GdnBlockPagedForTest(Q(vt::DeviceType::kCPU), w, c, h, meta, ssm,
                                      conv, slots, conv_len, T);
  };

  const std::vector<float> silu_out = run(silu);
  const std::vector<float> sigmoid_out = run(sigmoid);
  REQUIRE(static_cast<int64_t>(silu_out.size()) == T * H);
  REQUIRE(sigmoid_out.size() == silu_out.size());
  INFO("dims := ", std::string(g.name));
  CAPTURE(mixed);

  // silu(0) == 0 annihilates the block output, exactly.
  size_t silu_nonzero = 0;
  for (size_t i = 0; i < silu_out.size(); ++i)
    if (silu_out[i] != 0.0f) ++silu_nonzero;
  CAPTURE(silu_nonzero);
  CHECK(silu_nonzero == 0);

  // sigmoid(0) == 0.5 does not. Without this the check above would also pass on
  // a fixture whose core happened to be zero, which would prove nothing.
  double max_sigmoid = 0.0;
  for (size_t i = 0; i < sigmoid_out.size(); ++i)
    max_sigmoid = std::max(max_sigmoid, std::abs(static_cast<double>(sigmoid_out[i])));
  CAPTURE(max_sigmoid);
  CHECK(max_sigmoid > 0.0);
}

}  // namespace

TEST_CASE("GDN output_gate_type=sigmoid reaches GdnBlockPaged's gate (CPU)") {
  RunGateActivationCase(kGate27B, /*mixed=*/false);
  RunGateActivationCase(kGate35B, /*mixed=*/false);
}

TEST_CASE("GDN output_gate_type=sigmoid reaches the MIXED spec batch gate (CPU)") {
  RunGateActivationCase(kGate27B, /*mixed=*/true);
  RunGateActivationCase(kGate35B, /*mixed=*/true);
}

TEST_CASE("GDN gate POLARITY: GdnBlockPaged's silu arm is silu (CPU)") {
  RunGatePolarityCase(kGate27B, /*mixed=*/false);
  RunGatePolarityCase(kGate35B, /*mixed=*/false);
}

TEST_CASE("GDN gate POLARITY: the MIXED spec batch's silu arm is silu (CPU)") {
  RunGatePolarityCase(kGate27B, /*mixed=*/true);
  RunGatePolarityCase(kGate35B, /*mixed=*/true);
}

#ifdef VLLM_CPP_CUDA
namespace {

// THE FP8 GATED-RMSNORM TAIL — the arm no backend had ever executed.
//
// MODEL-GDN-OUTPUT-GATE-TYPE (#489) wired `output_gate_type` into TWELVE
// gate-carrying constructions across the three Qwen3.5 GDN tails. SIX of them
// are fp8: the `vt::kRmsNormGatedQuantFp8` FusedRecipe copy and the direct
// `vt::RmsNormGatedQuantFp8` call in each tail (qwen3_5.cpp:3641-3647,
// :4111-4117, :4539-4545). They are reachable only when `out_proj_fp8` is
// populated AND `GdnOutFp8FuseEnabled()` AND `GlueFuseEnabled()` AND
// `Platform::supports_fp8()`. The last term is FALSE on every non-CUDA
// platform, so the row's CPU gate could not touch them and did not: they
// shipped WIRED BUT NEVER EXECUTED, which is precisely the footing this row
// exists to remove.
//
// The polarity argument is the bf16 one, unchanged (RunGatePolarityCase above):
// silu(0) = 0*sigmoid(0) = 0 EXACTLY while sigmoid(0) = 0.5, and z = h @
// in_proj_z is a bias-free GEMM, so a zeroed in_proj_z makes the SILU tail
// annihilate the block output and leaves the SIGMOID tail non-zero. It survives
// the fp8 store: quantizing 0.0 yields the fp8 zero byte and 0 @ out_proj is
// still exactly 0, while the sigmoid arm's 0.5*norm(core) quantizes well inside
// e4m3 range at these scales.
//
// WHY THIS CANNOT PASS VACUOUSLY. `out_proj_fp8` non-empty also routes the
// UNFUSED bf16 tail to an fp8 GEMM, so "the output is fp8-ish" proves nothing
// about which gated-RMSNorm ran. What pins it is the mutation: hardwiring
// `sigmoid_gate` to `false` at the SIX fp8 constructions ONLY — leaving the six
// bf16 ones untouched — must turn this case RED while every CPU polarity case
// above stays GREEN. That mutation WAS run, on the dgx at b9d172f6, and is
// recorded in `.agents/specs/gdn-output-gate-type.md`. It inverted both fp8
// KINDS at once, so it does not yet demonstrate what the
// `_fused_chain_off` CTest entry adds over the default entry — the default
// entry alone would have caught it. The SPLIT mutation that would demonstrate
// it (the 3 direct `RmsNormGatedQuantFp8` sites alone, then the 3 recipe-copy
// sites alone, each showing exactly one of the two entries fail) is recorded in
// that spec as OWED at the next GPU run. It has not been performed.
vllm::Fp8Weight MakeFp8OutProj(int64_t n, int64_t k, uint64_t seed) {
  vllm::Fp8Weight f;
  f.n = n;
  f.k = k;
  // Per-tensor scales in the shape the 35B loader produces (powers of two, so
  // the dequant introduces no rounding of its own).
  f.input_scale = 0.0078125F;
  f.weight_scale = 0.00390625F;
  f.alpha = f.input_scale * f.weight_scale;
  f.packed.dtype = DType::kI8;
  f.packed.rank = 2;
  f.packed.shape[0] = n;
  f.packed.shape[1] = k;
  f.packed.bytes.resize(static_cast<size_t>(n * k));
  auto* bytes = f.packed.bytes.data();
  // 0x00..0x7D: finite non-negative e4m3 bytes (0x7F is NaN). Same construction
  // as the merged-qkvz resident case above.
  for (int64_t i = 0; i < n * k; ++i)
    bytes[static_cast<size_t>(i)] =
        static_cast<uint8_t>(Mix(seed + static_cast<uint64_t>(i)) % 0x7EU);
  return f;
}

void RunGatePolarityFp8Case(const GdnDims& g, bool mixed) {
  setenv("VT_GDN_INDEXED_STATE_IO", "1", 1);  // mixed needs widened indexed IO
  vt::GetBackend(vt::DeviceType::kCUDA);      // skip cleanly if no device

  // An ABSENT precondition is a SKIP, never a failure — tests/CMakeLists.txt:26-33
  // ("A test whose preconditions are absent exits 77") and the try/catch at
  // tests/vt/test_ops_fp8_cutlass.cpp:33-39. `supports_fp8()` is
  // `has_device_capability(8, 9)` (src/vllm/platforms/cuda.cpp:39), so a CUDA
  // build running on a pre-sm_89 board has NO fp8 gated-RMSNorm tail to reach:
  // the branch under test is unreachable, not broken. The Jetson AGX Orin
  // (sm_87) is a recorded runtime-gate host (.agents/benchmark-record.md:2402-2426),
  // and a `REQUIRE` here turned this whole suite RED on it for a capability the
  // board never claimed.
  //
  // The skip cannot swallow a real defect: it returns BEFORE any of this case's
  // CHECKs and ONLY on the missing capability. Where fp8 IS supported the guard
  // is not taken and every assertion below runs unchanged, so broken gate wiring
  // still fails loudly. The cost is that a skipped run reports Passed with the
  // fp8 cases contributing zero assertions — read the assertion COUNT, not the
  // status, to tell a run that exercised the fp8 tail from one that did not.
  if (!vllm::platforms::GetPlatform(vt::DeviceType::kCUDA).supports_fp8()) {
    MESSAGE(
        "SKIP: this device does not support fp8 (Platform::supports_fp8() == "
        "has_device_capability(8, 9) is false) — the GDN fp8 gated-RMSNorm tail "
        "is unreachable here, so the fp8 gate-polarity case did NOT run");
    return;
  }

  const int64_t H = 128;
  const int Ts = 2, Tp = 3;
  const int64_t T = mixed ? Ts + Tp : Tp;
  const HfConfig silu = MakeConfig(g, H);
  REQUIRE(silu.output_gate_type == "silu");
  HfConfig sigmoid = MakeConfig(g, H);
  sigmoid.output_gate_type = "sigmoid";

  const int64_t Hv = g.hv, Dv = g.dv, Dk = g.dk, Kw = g.kw;
  const int64_t key_dim = g.hk * Dk, value_dim = Hv * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t ssm_row = Hv * Dv * Dk;
  const int64_t conv_len = (Kw - 1) + 1;
  const int64_t slots = 3;

  GdnLayerWeights w = MakeGdnWeights(silu);
  // z == 0. lo == hi == 0 makes every element exactly +0.0f.
  w.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, 20, 0.0f, 0.0f);
  // THE branch selector: a populated W8A8 out_proj is what routes the tail into
  // the fused fp8 gated-RMSNorm (the 35B production shape).
  w.out_proj_fp8 = MakeFp8OutProj(H, value_dim, 700);
  REQUIRE_FALSE(w.out_proj_fp8.Empty());

  std::vector<float> h(static_cast<size_t>(T * H));
  for (size_t i = 0; i < h.size(); ++i) h[i] = RandV(5000 + i, -1.0f, 1.0f);
  std::vector<float> ssm0(static_cast<size_t>(slots * ssm_row));
  for (size_t i = 0; i < ssm0.size(); ++i) ssm0[i] = RandV(6000 + i, -0.5f, 0.5f);
  std::vector<float> conv0(static_cast<size_t>(slots * conv_dim * conv_len), 0.0f);
  for (int64_t s = 0; s < slots; ++s)
    for (int64_t ch = 0; ch < conv_dim; ++ch)
      for (int64_t j = 0; j < Kw - 1; ++j)
        conv0[static_cast<size_t>((s * conv_dim + ch) * conv_len + j)] =
            RandV(7000 + (s * conv_dim + ch) * (Kw - 1) + j, -1.0f, 1.0f);

  const auto run = [&](const HfConfig& c) {
    std::vector<float> ssm = ssm0, conv = conv0;
    const GDNAttentionMetadata meta = mixed ? MixedMeta(Tp) : PrefillMeta(Tp, 2);
    return vllm::GdnBlockPagedForTest(Q(vt::DeviceType::kCUDA), w, c, h, meta, ssm,
                                      conv, slots, conv_len, T);
  };

  const std::vector<float> silu_out = run(silu);
  const std::vector<float> sigmoid_out = run(sigmoid);
  REQUIRE(static_cast<int64_t>(silu_out.size()) == T * H);
  REQUIRE(sigmoid_out.size() == silu_out.size());
  INFO("dims := ", std::string(g.name));
  CAPTURE(mixed);

  // silu(0) == 0 annihilates the block output, exactly — through the fp8 store
  // and the fp8 out_proj GEMM alike.
  size_t silu_nonzero = 0;
  for (size_t i = 0; i < silu_out.size(); ++i)
    if (silu_out[i] != 0.0f) ++silu_nonzero;
  CAPTURE(silu_nonzero);
  CHECK(silu_nonzero == 0);

  // sigmoid(0) == 0.5 does not. Without this the check above would also pass on
  // a fixture whose core happened to be zero, or one whose fp8 quantization
  // flushed everything to zero — neither of which would prove anything.
  double max_sigmoid = 0.0;
  for (size_t i = 0; i < sigmoid_out.size(); ++i)
    max_sigmoid = std::max(max_sigmoid, std::abs(static_cast<double>(sigmoid_out[i])));
  CAPTURE(max_sigmoid);
  CHECK(max_sigmoid > 0.0);
}

}  // namespace

TEST_CASE("GDN gate POLARITY on the FP8 tail: GdnBlockPaged (CUDA)") {
  RunGatePolarityFp8Case(kGate27B, /*mixed=*/false);
  RunGatePolarityFp8Case(kGate35B, /*mixed=*/false);
}

TEST_CASE("GDN gate POLARITY on the FP8 tail: the MIXED spec batch (CUDA)") {
  RunGatePolarityFp8Case(kGate27B, /*mixed=*/true);
  RunGatePolarityFp8Case(kGate35B, /*mixed=*/true);
}
#endif  // VLLM_CPP_CUDA
