// MODEL-MM-QWEN4-EXP — the hyper-connection INJECT weight's residency
// (issue #2449, spec `.agents/specs/qwen4-exp-inject-residency.md`).
//
// WHAT WAS BROKEN. `qwen4_exp_forward.cpp` builds the two per-layer
// hyper-connection calls out of four weight operands each. Three of them —
// `hc_norm`, `down`, `up` — go through `dense_attn::ResidentWeight`, which is
// the seam that puts a weight where the queue's kernels can read it. The fourth,
// `block_inject_weight`, went through `OwnedTensor::View()`, and `View()` ends
// with `t.device = vt::Device{}` (`qwen3_5_weights.cpp:500`): a HOST pointer
// stamped kCPU, whatever the queue is. `vt::Qwen4ExpGatedResidual` validates
// `t.device == q.device` on that operand — through `check_projection` for a
// block-typed weight and through `check_operand` for a float one, both arms —
// so on a CUDA queue every forward of this architecture stopped at decoder
// LAYER 0 with `qwen4_exp_gated_residual: block_inject_weight device mismatch`.
// Not at the QSA layers, not in the MoE: at the first hyper-connection of the
// first layer, before PLE had run.
//
// The refusal fires on BOTH arms and is a refusal on both. #2449 as filed said
// `check_projection` skipped the device check for a block-quantized operand and
// that the released Q8_0 checkpoint would therefore hand a device kernel a host
// pointer; that reading is withdrawn on the issue itself. The `VT_CHECK` at
// `src/vt/ops.cpp:2598` is INSIDE the block-quant branch, so a Q8_0 inject
// produced this same message rather than corrupting memory.
//
// WHAT THIS FILE MEASURES, AND WHAT IT CANNOT.
//
//   * `on a CUDA queue` is the case that carries the claim. It drives
//     `ModelRegistry::Forward` — the production entry point AGENTS.md names —
//     over the shared tiny `qwen4exp` GGUF fixture on a real CUDA queue, and
//     asserts only what this row owns: the forward does not stop on an INJECT
//     residency. It PRINTS where it stopped instead of asserting that it
//     completed, because getting past layer 0 exposes whatever is next and the
//     next stop is not this row's. On a build with no CUDA backend the case says
//     UNMEASURED and asserts nothing; that state is reported, never a pass.
//
//   * `on a CPU queue` is the regression guard the CPU tier can run. The CPU arm
//     of `ResidentWeight` ALIASES the host bytes and carries `repacked` and
//     `elem_kn_repacked` exactly as `View()` does, so the change is meant to be
//     invisible there — and `q8_0_aligned`, the one marker `View()` carries and
//     `ResidentWeight` drops, is never set on this model's weights, because
//     `LoadMatmul` (`qwen4_exp_weights.cpp:130-141`) calls `OwnGgufQuantBlocks`
//     without the `cuda_align` argument and it defaults false
//     (`qwen3_5_gguf_weights.h:100`). The case asserts the CPU forward still
//     completes with finite, non-constant logits. It cannot detect the defect,
//     because on a CPU queue `View()`'s kCPU stamp is the right answer. It is
//     stated here so no reader mistakes a green CPU tier for a device gate.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
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

// The refusal this row removes, by the words `vt::ops` refuses with. Both
// `check_operand` and `check_projection` end their message in
// `<operand> device mismatch`, and the operand is named, so this matches the
// inject weight and NOT a device mismatch on some other operand of some other
// op — which is the distinction that makes a red here attributable.
bool IsInjectResidencyRefusal(const std::string& what) {
  return what.find("block_inject_weight device mismatch") != std::string::npos;
}

// One prefill step's worth of metadata over the tiny fixture, byte for byte the
// shape `test_qwen4_exp_layer_loop.cpp` drives the CPU forward with.
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
// THE CLAIM: a CUDA `ModelRegistry::Forward` is not stopped by this operand.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE(
    "qwen4_exp: a CUDA ModelRegistry::Forward gets past the layer-0 "
    "hyper-connection inject weight") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  if (!HasCudaBackend()) {
    MESSAGE(
        "no CUDA backend in this build: the layer-0 inject residency is "
        "UNMEASURED by this run, and nothing below asserted it");
    return;
  }
  // THE INSTRUMENT'S OWN PRECONDITION. `ModelRegistry::Load` takes its device
  // from `platforms::CurrentPlatform()` rather than from an argument, so on a
  // build where CUDA registered a backend but not the platform this case would
  // load on the CPU, take the aliasing arm of `ResidentWeight`, and report a
  // pass having measured nothing.
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() ==
          vt::DeviceType::kCUDA);

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);

  std::unique_ptr<vllm::LoadedModel> model;
  std::string load_stopped_with;
  try {
    model = LoadThroughRegistry(g);
  } catch (const std::exception& e) {
    load_stopped_with = e.what();
  }
  INFO("CUDA load stopped with: ", load_stopped_with);
  REQUIRE(load_stopped_with.empty());
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

  vllm::ModelForwardInput in{s.ids,   s.pos, s.am,  s.gm,
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

  // ─── THE ONE ASSERTION THIS ROW OWNS ──────────────────────────────────────
  // RED before the fix, with `block_inject_weight device mismatch` printed by
  // the MESSAGE above it; green after, whatever the forward goes on to do.
  MESSAGE("CUDA ModelRegistry::Forward stopped with: ",
          stopped_with.empty() ? std::string("(it did not stop)") : stopped_with);
  CHECK_FALSE(IsInjectResidencyRefusal(stopped_with));

  if (!stopped_with.empty()) {
    // NOT A PASS DRESSED AS ONE. The forward did not complete; the case says so
    // in its own output and claims nothing beyond the assertion above.
    MESSAGE(
        "the CUDA forward did not complete; the layer-0 inject weight is no "
        "longer the reason");
    gpu.DestroyQueue(q);
    return;
  }

  // Reached only once nothing downstream stops either. FINITENESS FIRST: a fold
  // over `std::max` returns the non-NaN operand, so an all-NaN row reads as a
  // match to any tolerance and still argmaxes to an index in range.
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
  // A CONSTANT row is finite, in range, and argmaxes to 0 while meaning the
  // tower contributed nothing.
  CHECK(hi > lo);
  MESSAGE("qwen4_exp CUDA forward logit range [", lo, ", ", hi, "]");
  gpu.DestroyQueue(q);
}

// ═══════════════════════════════════════════════════════════════════════════
// THE CPU REGRESSION GUARD. Cannot see the defect; says so.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE(
    "qwen4_exp: routing the inject weight through ResidentWeight leaves the "
    "CPU forward intact") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  Step s = MakeStep(T);

  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::dense_attn::Dev d{cpu, q};

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
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim},
                              kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  vllm::ModelForwardInput in{s.ids,   s.pos, s.am,  s.gm,
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
  INFO("CPU ModelRegistry::Forward stopped with: ", stopped_with);
  REQUIRE(stopped_with.empty());

  const float* logits = fl.on_device()
                            ? static_cast<const float*>(fl.device_tensor.data)
                            : fl.host.data();
  REQUIRE(logits != nullptr);
  const size_t n = static_cast<size_t>(fl.rows * fl.vocab);
  REQUIRE(n == static_cast<size_t>(kVocab));
  int finite = 0;
  for (size_t i = 0; i < n; ++i) finite += std::isfinite(logits[i]) ? 1 : 0;
  CHECK(finite == static_cast<int>(n));
  float lo = logits[0], hi = logits[0];
  for (size_t i = 0; i < n; ++i) {
    lo = logits[i] < lo ? logits[i] : lo;
    hi = logits[i] > hi ? logits[i] : hi;
  }
  CHECK(hi > lo);
  MESSAGE("qwen4_exp CPU forward logit range [", lo, ", ", hi, "]");
}
