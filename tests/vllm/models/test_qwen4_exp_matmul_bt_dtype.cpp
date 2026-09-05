// MODEL-MM-QWEN4-EXP — the `(f32 activation, bf16 weight)` arm of the CUDA
// `matmul_bt` (issue #2452, spec
// `.agents/specs/qwen4-exp-matmul-bt-mixed-dtype.md`).
//
// WHAT WAS BROKEN. With the layer-0 inject weight staged (#2449/#2453), a CUDA
// `ModelRegistry::Forward` over the tiny `qwen4exp` GGUF fixture reached the
// hyper-connection mixer and stopped there with
//
//     vt cuda: matmul_bt: unsupported dtype combo (f32,bf16)->f32
//
// from `cuda_matmul.cu`. The call is `Qwen4ExpGatedResidualKernelCuda`'s first
// projection (`cuda_qwen4_exp.cu:414`, then `:418` and `:428`): the mixer keeps
// its grouped-RMSNorm result in an f32 scratch and projects it through the three
// mix weights, which on this fixture are bf16.
//
// WHY THE ACTIVATION IS NOT THE BUG, WHICH IS THE WHOLE QUESTION.
// `Qwen4ExpTextRMSNorm.forward` ends in `output.type_as(x)`
// (transformers 5.16.0, `models/qwen4_exp/modeling_qwen4_exp.py:173-178`, this
// row's lane oracle), so upstream ROUNDS the normed value back to the model
// dtype before the down projection. This tree deliberately does not: it widens
// on load, computes in f32 and rounds once on the store. That divergence is
// RATIFIED and GATED — `.agents/specs/qwen4-exp-flash-next.md` records it under
// `## Owed`, and reproducing upstream's rounding is that row's mutation **M13**,
// RED in both the host and the device suite. Narrowing the activation here in
// order to satisfy cuBLASLt would BE that mutation.
//
// So the gap was the seam's, not the caller's: three of the four
// implementations behind `vt::MatmulBT` already answered an f32 activation
// against a typed weight — the CPU elementwise kernel, the CPU block-quant
// kernel, and THIS DEVICE'S OWN `kMatmulBTQuant`, which is the arm the released
// Q8_0 checkpoint takes at these very three call sites. Only the cuBLASLt
// elementwise lane refused, and `cuda_deepseek_v4.cu`'s `RouterGateKernel`
// already hand-wrote a private GEMV because of it.
//
// WHAT THIS FILE MEASURES, AND WHAT IT CANNOT.
//
//   * `a CUDA ModelRegistry::Forward` carries the REACH claim. It drives the
//     production entry point over the shared fixture on a real CUDA queue and
//     asserts the forward does not stop on this combo. It ALSO asserts its own
//     precondition — that the forward got past the layer-0 inject weight —
//     because on a tree without #2453 an earlier stop would hide this one and
//     the case would pass having measured nothing.
//   * `matches a double reference` carries the VALUE claim. A CPU-vs-CUDA
//     comparison alone would prove the two arms agree, not that either is
//     right, so the oracle is a host `double` dot product over the same bytes.
//   * The CPU case is a regression guard and CANNOT see the defect: the CPU
//     elementwise kernel has always accepted this pair. It says so rather than
//     letting a green CPU tier read as a device gate.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/platforms/interface.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "support/qwen4_exp_gguf_fixture.h"

using vt::DType;

namespace {

bool HasCudaBackend() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// The refusal this row removes, by the words `cuda_matmul.cu` refuses with. The
// dtype pair is part of the match, so a LATER combo refusal on some other pair
// is not mistaken for this one.
bool IsMixedComboRefusal(const std::string& what) {
  return what.find("matmul_bt: unsupported dtype combo") != std::string::npos &&
         what.find("(f32,bf16)") != std::string::npos;
}

// #2453's refusal. Asserted ABSENT so this file's CUDA case cannot pass by
// stopping earlier than the code it claims to measure.
bool IsInjectResidencyRefusal(const std::string& what) {
  return what.find("block_inject_weight device mismatch") != std::string::npos;
}

uint16_t F32ToBf16Rne(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float Bf16ToF32(uint16_t v) {
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// A cheap deterministic generator; no <random> so the bytes are identical on
// every host and the two arms are fed the same numbers by construction.
float Deterministic(uint32_t i, uint32_t salt) {
  uint32_t x = i * 2654435761u + salt * 40503u + 1u;
  x ^= x >> 15;
  x *= 0x2c1b3c6du;
  x ^= x >> 12;
  return static_cast<float>(static_cast<int32_t>(x % 2001) - 1000) / 1000.0F;
}

// One prefill step's worth of metadata over the tiny fixture.
struct Step {
  std::vector<int32_t> ids;
  std::vector<int32_t> pos;
  vllm::v1::CommonAttentionMetadata am;
  vllm::v1::GDNAttentionMetadata gm;
  std::vector<int32_t> logits_indices;
};

Step MakeStep(int64_t T) {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)
  Step s;
  s.ids.resize(static_cast<size_t>(T));
  s.pos.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    s.ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    s.pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }
  s.am.num_reqs = 1;
  s.am.num_actual_tokens = static_cast<int>(T);
  s.am.block_table_num_cols = 1;
  s.am.block_table_tensor.assign(1, 0);
  s.am.seq_lens.assign(1, static_cast<int32_t>(T));
  s.am.query_start_loc = {0, static_cast<int32_t>(T)};
  s.am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    s.am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  s.gm.num_prefills = 1;
  s.gm.num_prefill_tokens = static_cast<int>(T);
  s.gm.num_actual_tokens = static_cast<int>(T);
  s.gm.has_initial_state = std::vector<uint8_t>{0};
  s.gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  s.gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  s.gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  s.gm.prefill_state_indices = std::vector<int32_t>{0};
  s.gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*s.gm.non_spec_query_start_loc);
    s.gm.batch_ptr = conv.batch_ptr;
    s.gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }
  s.logits_indices.assign(1, static_cast<int32_t>(T - 1));
  return s;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// 1. THE VALUE CLAIM, against a host `double` oracle rather than the CPU arm.
// ═══════════════════════════════════════════════════════════════════════════
//
// THE BOUND IS DERIVED, NOT FITTED. Both arms accumulate the K-length dot in
// f32; they differ only in the ORDER of that reduction (the host reference walks
// k ascending, cuBLASLt tiles and tree-reduces). The classical bound on an
// f32 sum of K terms is `sqrt(K) * eps_f32` relative for a random reordering,
// and the tolerance below is 8x that — checked against the measured margin,
// which the case PRINTS, and scaled by sqrt(K) so it cannot point the wrong way
// as K grows. The bf16 -> f32 upcast the device arm performs is EXACT
// (`bits << 16`), so it contributes no term at all; a mutation that corrupts it
// moves the relative error to O(1), four orders above this bound.
TEST_CASE("CUDA matmul_bt: (f32 act, bf16 weight) matches a double reference") {
  if (!HasCudaBackend()) {
    MESSAGE(
        "no CUDA backend in this build: the mixed-dtype GEMM is UNMEASURED by "
        "this run, and nothing below asserted it");
    return;
  }
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = gpu.CreateQueue();
  vllm::dense_attn::Dev d{gpu, q};

  // (M, K, N). The middle row is the hyper-connection mixer's own shape class
  // (a wide `hc*H` reduction into a small low rank); the last is the `mix_up`
  // direction, wide on N instead.
  const struct {
    int64_t m, k, n;
  } cases[] = {{4, 64, 8}, {6, 1024, 16}, {6, 16, 1024}, {1, 4096, 4}};

  for (const auto& c : cases) {
    CAPTURE(c.m);
    CAPTURE(c.k);
    CAPTURE(c.n);
    std::vector<float> a(static_cast<size_t>(c.m * c.k));
    for (size_t i = 0; i < a.size(); ++i)
      a[i] = Deterministic(static_cast<uint32_t>(i), 1u);
    std::vector<uint16_t> w(static_cast<size_t>(c.n * c.k));
    for (size_t i = 0; i < w.size(); ++i)
      w[i] = F32ToBf16Rne(Deterministic(static_cast<uint32_t>(i), 7u));

    // The oracle: the SAME bytes, in double. `Bf16ToF32` is the exact widening
    // the device kernel performs, so the reference and the device read
    // bit-identical operand values and only the reduction differs.
    std::vector<double> ref(static_cast<size_t>(c.m * c.n), 0.0);
    for (int64_t i = 0; i < c.m; ++i)
      for (int64_t j = 0; j < c.n; ++j) {
        double acc = 0.0;
        for (int64_t t = 0; t < c.k; ++t)
          acc += static_cast<double>(a[static_cast<size_t>(i * c.k + t)]) *
                 static_cast<double>(Bf16ToF32(w[static_cast<size_t>(j * c.k + t)]));
        ref[static_cast<size_t>(i * c.n + j)] = acc;
      }

    vllm::dense_attn::DBuf da(d, DType::kF32, {c.m, c.k}, a.data());
    vllm::dense_attn::DBuf dw(d, DType::kBF16, {c.n, c.k}, w.data());
    vllm::dense_attn::DBuf dout(d, DType::kF32, {c.m, c.n});
    vt::Tensor o = dout.t();
    // THE ARM UNDER TEST. Before this row it threw here, naming the combo.
    REQUIRE_NOTHROW(vt::MatmulBT(q, o, da.t(), dw.t()));
    std::vector<float> got(static_cast<size_t>(c.m * c.n), 0.0F);
    dout.Download(d, got.data());

    double max_abs_ref = 0.0, max_abs_diff = 0.0;
    int finite = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      finite += std::isfinite(got[i]) ? 1 : 0;
      const double r = ref[i];
      const double diff = std::fabs(static_cast<double>(got[i]) - r);
      if (std::fabs(r) > max_abs_ref) max_abs_ref = std::fabs(r);
      if (diff > max_abs_diff) max_abs_diff = diff;
    }
    // FINITENESS FIRST: a max-fold over NaN returns the other operand, so an
    // all-NaN output would otherwise read as a match to any tolerance.
    REQUIRE(finite == static_cast<int>(got.size()));
    REQUIRE(max_abs_ref > 0.0);
    const double rel = max_abs_diff / max_abs_ref;
    const double bound =
        8.0 * std::sqrt(static_cast<double>(c.k)) * 1.1920929e-7;
    MESSAGE("m=", c.m, " k=", c.k, " n=", c.n, "  max|diff|=", max_abs_diff,
            "  rel=", rel, "  bound=", bound, "  margin=", bound / (rel + 1e-30),
            "x");
    CHECK(rel <= bound);
  }
  gpu.DestroyQueue(q);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. THE REACH CLAIM: a CUDA ModelRegistry::Forward is not stopped by it.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE(
    "qwen4_exp: a CUDA ModelRegistry::Forward is not stopped by the mixer's "
    "mixed-dtype matmul_bt") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  if (!HasCudaBackend()) {
    MESSAGE(
        "no CUDA backend in this build: the mixer's GEMM is UNMEASURED by this "
        "run, and nothing below asserted it");
    return;
  }
  // THE INSTRUMENT'S OWN PRECONDITION. `ModelRegistry::Load` takes its device
  // from `platforms::CurrentPlatform()`, so on a build where CUDA registered a
  // backend but not the platform this case would load on the CPU, take the CPU
  // GEMM, and report a pass having measured nothing.
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() ==
          vt::DeviceType::kCUDA);

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);

  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  Step s = MakeStep(T);

  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = gpu.CreateQueue();
  vllm::dense_attn::Dev d{gpu, q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  // RANK 4, NOT 5 (#2435). The paged K/V page is logically
  // `[2, num_blocks, block_size, num_kv_heads, head_size]`, and `vt::Tensor`
  // holds four dimensions (`vt::kMaxRank`). Folding the leading K/V axis into
  // the block axis keeps the same bytes, the same element count and the same
  // `data` pointer -- which is all this fixture takes from the buffer, because
  // `PagedKvCache` carries the five extents as separate scalars. The rank-5
  // spelling wrote past `Tensor::shape` and `Tensor::stride` and set
  // `Tensor::repacked` as a side effect; `dense_attn::MakeTensor` now refuses it.
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim},
                              kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  vllm::ModelForwardInput in{s.ids,   s.pos, s.am,   s.gm,
                             attn_kv, gdn,   config, q,
                             s.logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  vllm::ForwardLogits fl;
  std::string stopped_with;
  try {
    fl = vllm::ModelRegistry::Forward(*model, in);
  } catch (const std::exception& e) {
    stopped_with = e.what();
  }
  MESSAGE("CUDA ModelRegistry::Forward stopped with: ",
          stopped_with.empty() ? std::string("(it did not stop)") : stopped_with);

  // THE PRECONDITION, CHECKED BEFORE ANYTHING IS CLAIMED. The mixer's GEMM runs
  // INSIDE the op whose operand #2449 fixed, so on a tree without that fix the
  // forward stops BEFORE reaching this row's code and the assertion below would
  // pass having measured nothing. That state is REPORTED, never a pass: the
  // case asserts nothing and says which change it is waiting for.
  if (IsInjectResidencyRefusal(stopped_with)) {
    MESSAGE(
        "UNMEASURED: this forward stops earlier, on the layer-0 "
        "block_inject_weight residency (#2449, PR #2453). Until that lands the "
        "mixer's GEMM is not reached and this case asserts NOTHING.");
    WARN_FALSE(IsInjectResidencyRefusal(stopped_with));
    gpu.DestroyQueue(q);
    return;
  }

  // ─── THE ONE ASSERTION THIS ROW OWNS ──────────────────────────────────────
  CHECK_FALSE(IsMixedComboRefusal(stopped_with));

  if (!stopped_with.empty()) {
    // NOT A PASS DRESSED AS ONE. The forward did not complete; the case prints
    // where it went instead and claims nothing beyond the assertion above.
    MESSAGE(
        "the CUDA forward did not complete; the mixer's mixed-dtype GEMM is no "
        "longer the reason");
    gpu.DestroyQueue(q);
    return;
  }

  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);
  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  gpu.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  gpu.Synchronize(q);
  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  CHECK(finite == static_cast<int>(host.size()));
  float lo = host[0], hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  CHECK(hi > lo);
  MESSAGE("qwen4_exp CUDA forward logit range [", lo, ", ", hi, "]");
  gpu.DestroyQueue(q);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. THE CPU GUARD. It cannot see the defect, and says so.
// ═══════════════════════════════════════════════════════════════════════════
//
// `vt::MatmulBT`'s CPU kernel reads both operands through a dtype-generic getter
// and has always answered this pair, so this case was GREEN before the fix and
// is green after. It is here to pin that the shared contract really does admit
// the pair on the tier CI can run — the polarity that makes the CUDA lane, not
// the caller, the thing that was out of line.
TEST_CASE("CPU matmul_bt has always accepted (f32 act, bf16 weight)") {
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = cpu.CreateQueue();
  vllm::dense_attn::Dev d{cpu, q};
  const int64_t M = 3, K = 32, N = 5;
  std::vector<float> a(static_cast<size_t>(M * K));
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = Deterministic(static_cast<uint32_t>(i), 1u);
  std::vector<uint16_t> w(static_cast<size_t>(N * K));
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = F32ToBf16Rne(Deterministic(static_cast<uint32_t>(i), 7u));

  vllm::dense_attn::DBuf da(d, DType::kF32, {M, K}, a.data());
  vllm::dense_attn::DBuf dw(d, DType::kBF16, {N, K}, w.data());
  vllm::dense_attn::DBuf dout(d, DType::kF32, {M, N});
  vt::Tensor o = dout.t();
  REQUIRE_NOTHROW(vt::MatmulBT(q, o, da.t(), dw.t()));
  std::vector<float> got(static_cast<size_t>(M * N), 0.0F);
  dout.Download(d, got.data());
  double worst = 0.0;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t t = 0; t < K; ++t)
        acc += static_cast<double>(a[static_cast<size_t>(i * K + t)]) *
               static_cast<double>(Bf16ToF32(w[static_cast<size_t>(j * K + t)]));
      const double diff =
          std::fabs(static_cast<double>(got[static_cast<size_t>(i * N + j)]) - acc);
      if (diff > worst) worst = diff;
    }
  MESSAGE("CPU (f32,bf16) max|diff| vs double reference: ", worst);
  CHECK(worst < 1e-4);
  MESSAGE(
      "this case CANNOT see the CUDA defect this row fixes: the refusal was in "
      "cuBLASLt, and CI has no GPU");
  cpu.DestroyQueue(q);
}
