// vllm.cpp original. Native ROCm BF16 MoE production gate (#3094).
// The checkpoint enters through ModelRegistry::Load and every step through Forward.
#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "rocm_moe_fixture.h"
#include "support/rocm_moe_reference_set.h"
#include "support/residual_norm_fixture.h"
#include "support/residual_norm_test.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {
using namespace rocm_moe_fixture;

// The production fixture this file's first two cases replay. Declared up here
// because the case decorators below are evaluated in file order; see the
// "env-gated cases" block for why those cases skip instead of exiting.
bool FixtureAbsent() {
  const char* e = std::getenv("VT_ROCM_MOE_FIXTURE");
  return e == nullptr || e[0] == '\0';
}

vt::MoeRouterTopKFn native_router = nullptr;
vt::RmsNormFn native_norm = nullptr;
vt::ResidualRmsNormFn native_expression = nullptr;
bool observe_residual = false;
int observed_norm_calls = 0;
std::vector<uint16_t> observed_attention, observed_residual, observed_gamma, observed_post;
std::vector<uint16_t> observed_first;
struct ExpressionRecord {
  vt::ResidualRmsNormArgs args;
  std::vector<uint16_t> a, base, delta, gamma, norm, residual;
  uintptr_t attention_pointer = 0;
};
std::vector<ExpressionRecord> observed_expressions;
std::vector<uint16_t> ReadRow(vt::Queue& q, const vt::Tensor& tensor) {
  REQUIRE(tensor.dtype == vt::DType::kBF16);
  REQUIRE(tensor.Numel() >= 128);
  std::vector<uint16_t> row(128);
  auto& backend = vt::GetBackend(q.device);
  backend.Copy(q, row.data(), tensor.data, row.size() * sizeof(uint16_t));
  backend.Synchronize(q);
  return row;
}
void CaptureResidualNorm(vt::Queue& q, vt::Tensor& out, const vt::Tensor& x,
                         const vt::Tensor& weight, const vt::RmsNormArgs& args,
                         vt::Tensor* residual) {
  const bool legacy = vt::GetBackend(q.device).GetResidualNormPolicy() ==
                      vt::ResidualNormPolicy::kMaterialized;
  const bool capture = observe_residual && legacy && observed_norm_calls++ == 3;
  if (capture) {
    REQUIRE(residual != nullptr);
    observed_attention = ReadRow(q, x);
    observed_residual = ReadRow(q, *residual);
    observed_gamma = ReadRow(q, weight);
  }
  native_norm(q, out, x, weight, args, residual);
  if (capture) observed_post = ReadRow(q, out);
  if (observe_residual && !legacy && observed_first.empty()) observed_first = ReadRow(q, out);
}
void CaptureExpression(vt::Queue& q, vt::Tensor& out, const vt::Tensor& a,
                        const vt::Tensor& base, const vt::Tensor* delta,
                        const vt::Tensor& weight, const vt::ResidualRmsNormArgs& args,
                        vt::Tensor* residual_out) {
  const bool first = observe_residual && observed_norm_calls++ == 0;
  ExpressionRecord record;
  if (observe_residual) {
    record.args = args;
    record.a = ReadRow(q, a); record.base = ReadRow(q, base);
    record.gamma = ReadRow(q, weight);
    record.attention_pointer = reinterpret_cast<uintptr_t>(a.data);
    if (delta != nullptr) record.delta = ReadRow(q, *delta);
  }
  if (first) {
    observed_attention = ReadRow(q, a);
    observed_residual = ReadRow(q, base);
    observed_gamma = ReadRow(q, weight);
  }
  native_expression(q, out, a, base, delta, weight, args, residual_out);
  if (first) observed_post = ReadRow(q, out);
  if (observe_residual) {
    record.norm = ReadRow(q, out);
    if (residual_out != nullptr) record.residual = ReadRow(q, *residual_out);
    CHECK(ReadRow(q, a) == record.a);
    CHECK(ReadRow(q, weight) == record.gamma);
    if (!args.descriptor.residual_alias_base) CHECK(ReadRow(q, base) == record.base);
    if (delta != nullptr && !args.descriptor.output_alias_delta) CHECK(ReadRow(q, *delta) == record.delta);
    observed_expressions.push_back(std::move(record));
  }
}
nlohmann::json* current_routes = nullptr;
void CaptureRoutes(vt::Queue& q, vt::Tensor& weights, vt::Tensor& indices,
                    const vt::Tensor& logits, const vt::MoeRouterTopKArgs& args,
                    const vt::Tensor* bias) {
  native_router(q, weights, indices, logits, args, bias);
  if (current_routes == nullptr) return;
  std::vector<int32_t> ids(static_cast<size_t>(indices.Numel()));
  auto& backend = vt::GetBackend(q.device);
  backend.Copy(q, ids.data(), indices.data, ids.size() * sizeof(int32_t));
  backend.Synchronize(q);
  current_routes->push_back(ids);
}
vt::Tensor Tensor(void* data, vt::DType dtype, vt::Device device,
                  std::initializer_list<int64_t> dimensions) {
  vt::Tensor tensor;
  tensor.data = data;
  tensor.dtype = dtype;
  tensor.device = device;
  tensor.rank = static_cast<int>(dimensions.size());
  int axis = 0;
  for (int64_t size : dimensions) tensor.shape[axis++] = size;
  int64_t stride = 1;
  for (int i = tensor.rank - 1; i >= 0; --i) {
    tensor.stride[i] = stride;
    stride *= tensor.shape[i];
  }
  return tensor;
}
struct Buffer {
  vt::Backend& backend;
  void* data;
  Buffer(vt::Backend& b, size_t bytes) : backend(b), data(b.Alloc(bytes)) {}
  ~Buffer() { backend.Free(data); }
};
struct QueueGuard {
  vt::Backend& backend;
  vt::Queue queue;
  explicit QueueGuard(vt::Backend& b) : backend(b), queue(b.CreateQueue()) {}
  ~QueueGuard() { backend.DestroyQueue(queue); }
};

nlohmann::json Generate(const std::filesystem::path& fixture, int length, int concurrency,
                        int steps = 8) {
  auto config = vllm::LoadHfConfig((fixture / "config.json").string());
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((fixture / "model.safetensors").string()));
  auto model = vllm::ModelRegistry::Load(config, vllm::ModelSource::FromSafetensors(shards));
  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard queue(backend);
  auto& q = queue.queue;
  constexpr int64_t block = 64;
  std::vector<std::unique_ptr<Buffer>> allocations;
  std::vector<vllm::PagedKvCache> caches;
  for (int layer = 0; layer < kL; ++layer) {
    const size_t bytes = static_cast<size_t>(concurrency * 2 * block * kHkv * kDh) * 2;
    allocations.push_back(std::make_unique<Buffer>(backend, bytes));
    std::vector<uint16_t> zeros(bytes / 2, 0);
    backend.Copy(q, allocations.back()->data, zeros.data(), bytes);
    vllm::PagedKvCache kv;
    kv.data = allocations.back()->data;
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = concurrency;
    kv.block_size = block;
    kv.num_kv_heads = kHkv;
    kv.head_size = kDh;
    caches.push_back(kv);
    backend.Synchronize(q);
  }
  std::vector<vllm::GdnStateCache> gdn;
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<std::vector<int32_t>> generated(static_cast<size_t>(concurrency));
  nlohmann::json logits = nlohmann::json::array();
  nlohmann::json routes = nlohmann::json::array();
  current_routes = &routes;
  Buffer sampled(backend, static_cast<size_t>(concurrency) * sizeof(int64_t));
  auto ids = Tensor(sampled.data, vt::DType::kI64, q.device, {concurrency});
  for (int step = 0; step < steps; ++step) {
    const int count = step == 0 ? length : 1;
    const int previous = step == 0 ? 0 : length + step - 1;
    std::vector<int32_t> tokens, positions, gather;
    vllm::v1::CommonAttentionMetadata attention;
    attention.num_reqs = concurrency;
    attention.num_actual_tokens = count * concurrency;
    attention.query_start_loc = {0};
    for (int r = 0; r < concurrency; ++r) {
      const auto prompt = Prompt(length, r);
      for (int i = 0; i < count; ++i) {
        tokens.push_back(step == 0 ? prompt[static_cast<size_t>(i)] : generated[r].back());
        positions.push_back(previous + i);
        attention.slot_mapping.push_back(r * block + previous + i);
      }
      attention.query_start_loc.push_back((r + 1) * count);
      attention.seq_lens.push_back(previous + count);
      attention.block_table_tensor.push_back(r);
      gather.push_back((r + 1) * count - 1);
    }
    attention.query_start_loc_cpu = attention.query_start_loc;
    attention.seq_lens_cpu = attention.seq_lens;
    attention.max_query_len = count;
    attention.max_seq_len = previous + count;
    attention.block_table_num_cols = 1;
    attention.causal = true;
    vllm::ModelForwardInput input{tokens, positions, attention, gdn_meta, caches, gdn,
                                 config, q, gather};
    input.num_reqs = concurrency;
    input.gdn_state_slots = concurrency;
    input.pure_decode = step > 0;
    input.uniform_query_len = count;
    const auto result = vllm::ModelRegistry::Forward(*model, input);
    REQUIRE(result.on_device());
    REQUIRE(result.device_tensor.dtype == vt::DType::kF32);
    vt::GreedyArgmax(q, ids, result.device_tensor);
    std::vector<int64_t> host_ids(static_cast<size_t>(concurrency));
    std::vector<float> host_logits(static_cast<size_t>(concurrency * kV));
    backend.Copy(q, host_ids.data(), sampled.data, host_ids.size() * sizeof(int64_t));
    backend.Copy(q, host_logits.data(), result.device_tensor.data,
                 host_logits.size() * sizeof(float));
    backend.Synchronize(q);
    for (float value : host_logits) REQUIRE(std::isfinite(value));
    logits.push_back(host_logits);
    for (int r = 0; r < concurrency; ++r)
      generated[r].push_back(static_cast<int32_t>(host_ids[r]));
  }
  current_routes = nullptr;
  return {{"length", length}, {"concurrency", concurrency}, {"tokens", generated},
          {"logits", logits}, {"expert_ids", routes}};
}
}  // namespace

TEST_CASE("ROCm residual row zero matches the compiled primary through production forward" *
          doctest::skip(FixtureAbsent())) {
  using namespace residual_norm_fixture;
  REQUIRE_MESSAGE(!FixtureAbsent(),
                  "VT_ROCM_MOE_FIXTURE must name the exported production checkpoint");
  const char* directory = std::getenv("VT_ROCM_MOE_FIXTURE");
  native_norm = reinterpret_cast<vt::RmsNormFn>(
      vt::GetOp(vt::OpId::kRmsNorm, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kRmsNorm, vt::DeviceType::kROCM,
      {"test-residual-row-observer", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::RmsNormFn>(&CaptureResidualNorm))});
  native_expression = reinterpret_cast<vt::ResidualRmsNormFn>(
      vt::GetOp(vt::OpId::kResidualRmsNorm, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kResidualRmsNorm, vt::DeviceType::kROCM,
      {"test-residual-expression-observer", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::ResidualRmsNormFn>(&CaptureExpression))});
  observe_residual = true;
  observed_norm_calls = 0;
  observed_first.clear(); observed_expressions.clear();
  Generate(directory, 33, 2, 1);
  observe_residual = false;
  CHECK(observed_attention == std::vector<uint16_t>(kAttention.begin(), kAttention.end()));
  CHECK(observed_residual == std::vector<uint16_t>(kResidual.begin(), kResidual.end()));
  CHECK(observed_gamma == std::vector<uint16_t>(kGamma.begin(), kGamma.end()));
  REQUIRE(observed_post.size() == kPostNorm.size());
  int different = 0;
  for (size_t j = 0; j < kPostNorm.size(); ++j) {
    CAPTURE(j);
    different += observed_post[j] != kPostNorm[j];
    CHECK(observed_post[j] == kPostNorm[j]);
  }
  MESSAGE("Compared 128 production post-attention BF16 words with the compiled primary; differences: ", different);
  CHECK(observed_first == std::vector<uint16_t>(kFirstNorm.begin(), kFirstNorm.end()));
  REQUIRE(observed_expressions.size() == 4);
  auto checkpoint = vllm::SafetensorsFile::Open((std::filesystem::path(directory) / "model.safetensors").string());
  const std::vector<std::string> weights{
      "model.layers.0.post_attention_layernorm.weight", "model.layers.1.input_layernorm.weight",
      "model.layers.1.post_attention_layernorm.weight", "model.norm.weight"};
  for (size_t i = 0; i < observed_expressions.size(); ++i) {
    CAPTURE(i);
    const auto& record = observed_expressions[i];
    const bool triple = i % 2 == 1;
    CHECK(record.args.descriptor.expression == (triple ? vt::ResidualNormExpr::kDeltaPlusAdd : vt::ResidualNormExpr::kAdd));
    CHECK(record.args.descriptor.materialize_residual == (i == 1));
    const auto& gamma = checkpoint.Get(weights[i]);
    REQUIRE(gamma.dtype == "BF16");
    const auto* words = reinterpret_cast<const uint16_t*>(gamma.data);
    CHECK(record.gamma == std::vector<uint16_t>(words, words + 128));
    std::vector<uint16_t> expected_residual;
    const auto expected = residual_norm_test::Reference(record.a, record.base,
        triple ? &record.delta : nullptr, record.gamma, 1, 128, 128,
        record.args.eps, &expected_residual);
    CHECK(record.norm == expected);
    if (i == 1) CHECK(record.residual == expected_residual);
    else CHECK(record.residual.empty());
    if (triple) {
      CHECK(record.a == observed_expressions[i - 1].a);
      CHECK(record.base == observed_expressions[i - 1].base);
      CHECK(record.attention_pointer == observed_expressions[i - 1].attention_pointer);
    }
  }
  CHECK(observed_expressions[2].base == observed_expressions[1].residual);
}

TEST_CASE("ROCm BF16 MoE enters native providers through the production registry" *
          doctest::skip(FixtureAbsent())) {
  REQUIRE_MESSAGE(!FixtureAbsent(),
                  "VT_ROCM_MOE_FIXTURE must name the exported production checkpoint");
  const char* directory = std::getenv("VT_ROCM_MOE_FIXTURE");
  const std::filesystem::path fixture(directory);
  if (std::getenv("VT_ROCM_MOE_EXPORT_ONLY") != nullptr) {
    Export(fixture);
    REQUIRE(std::filesystem::file_size(fixture / "model.safetensors") > 0);
    return;
  }
  const char* oracle_path = std::getenv("VT_ROCM_MOE_ORACLE");
  REQUIRE(oracle_path != nullptr);
  nlohmann::json oracle;
  std::ifstream(oracle_path) >> oracle;
  // Test-only observation of the real device router. Expert providers retain
  // their native registry entries and per-call selection accounting.
  native_router = reinterpret_cast<vt::MoeRouterTopKFn>(
      vt::GetOp(vt::OpId::kMoeRouterTopK, vt::DeviceType::kROCM));
  vt::RegisterOpProvider(vt::OpId::kMoeRouterTopK, vt::DeviceType::kROCM,
      {"test-moe-route-capture", 100, nullptr,
       reinterpret_cast<void*>(static_cast<vt::MoeRouterTopKFn>(&CaptureRoutes))});
  vt::EnableOpProviderCallStats(true);
  for (auto op : {vt::OpId::kMoeGroupedGemmBf16GateUpSiluNative,
                  vt::OpId::kMoeGroupedGemmBf16Weighted, vt::OpId::kMoeCombinePreweighted})
    vt::ResetOpProviderStats(op, vt::DeviceType::kROCM);
  nlohmann::json runs = nlohmann::json::array();
  std::set<std::vector<int32_t>> selected_pairs;
  for (int length : {1, 3, 33}) {
    for (int concurrency : {1, 2}) {
      nlohmann::json reference;
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto run = Generate(fixture, length, concurrency);
        if (repeat == 0) reference = run["tokens"];
        CHECK(run["tokens"] == reference);
        bool compared = false;
        for (const auto& expected : oracle.at("runs")) {
          if (expected["length"] != length || expected["concurrency"] != concurrency ||
              expected["repeat"] != repeat)
            continue;
          CAPTURE(length);
          CAPTURE(concurrency);
          CAPTURE(repeat);
          compared = true;
          const auto native = run["tokens"].get<std::vector<std::vector<int32_t>>>();
          // The compared request count is the captured record's concurrency, not
          // whatever the native run happened to return.
          REQUIRE(native.size() == static_cast<size_t>(concurrency));
          for (size_t request = 0; request < native.size(); ++request) {
            CAPTURE(request);
            // Every captured configuration of this workload forms the reference
            // set for this request. Request 1 exists only in the records at
            // concurrency 2, so a record without it contributes nothing.
            const auto reference_set = rocm_moe_reference_set::Collect(
                oracle.at("runs"), length, repeat, request);
            REQUIRE(!reference_set.empty());
            const auto comparison = rocm_moe_reference_set::Compare(
                reference_set, native[request], concurrency);
            const int matched = comparison.pass() ? comparison.matched : -1;
            // The gate reports the reference set, the matched configuration, the
            // same-configuration outcome, and the reference's own disagreement
            // positions. The same-configuration outcome is a report and not an
            // assertion: the pinned primary disagrees with itself at length 33.
            std::cout << "[production tokens] length " << length << " concurrency "
                      << concurrency << " repeat " << repeat << " request " << request
                      << ": reference set "
                      << rocm_moe_reference_set::Describe(reference_set)
                      << "; matched configuration "
                      << (matched < 0
                              ? std::string("none")
                              : std::to_string(reference_set[static_cast<size_t>(matched)]
                                                   .concurrency))
                      << "; same-configuration match "
                      << (comparison.same_configuration_match ? "true" : "false")
                      << "; reference disagreement positions "
                      << rocm_moe_reference_set::Describe(comparison.disagreements)
                      << std::endl;
            // Whole-sequence membership. A per-position mix of two reference
            // sequences matches no member and fails here.
            CHECK(comparison.pass());
          }
        }
        REQUIRE(compared);
        REQUIRE(run["expert_ids"].size() == static_cast<size_t>(8 * kL));
        for (const auto& call : run["expert_ids"]) {
          const auto ids = call.get<std::vector<int32_t>>();
          for (size_t pair = 0; pair < ids.size(); pair += 2)
            selected_pairs.insert({ids[pair], ids[pair + 1]});
        }
        run["repeat"] = repeat;
        runs.push_back(std::move(run));
      }
    }
  }
  if (const char* output = std::getenv("VT_ROCM_MOE_OUTPUT"))
    std::ofstream(output) << runs.dump(2) << '\n';
  CHECK(selected_pairs.size() > 1);
  CHECK(vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16, vt::DeviceType::kROCM));
  CHECK(vt::OpRegistered(vt::OpId::kMoeGroupedGemmBf16GateUpSilu, vt::DeviceType::kROCM));
  for (auto op : {vt::OpId::kMoeGroupedGemmBf16GateUpSiluNative,
                  vt::OpId::kMoeGroupedGemmBf16Weighted, vt::OpId::kMoeCombinePreweighted}) {
    CAPTURE(static_cast<int>(op));
    const auto stats = vt::GetOpProviderStats(op, vt::DeviceType::kROCM);
    CHECK(vt::OpRegistered(op, vt::DeviceType::kROCM));
    CHECK(stats.selections > 0);
    REQUIRE(stats.last_selected != nullptr);
    CHECK(std::string(stats.last_selected) == vt::kNativeProviderName);
    CHECK(stats.declines == 0);
    CHECK(stats.fallbacks == 0);
    CHECK(vt::GetOpProviderStats(op, vt::DeviceType::kCPU).selections == 0);
  }
  vt::EnableOpProviderCallStats(false);
}

// ---------------------------------------------------------------------------
// BACKEND-ROCM-BF16-MOE (#3115): the native attention boundary.
//
// The first attention output of L33/C2/R0 layer 0 differs from the pinned
// primary's by 2902 of 8448 bf16 words, and the captured numbers alone do not
// separate two candidate causes:
//
//   (a) the Q/K PREAMBLE. The primary keeps normalized Q/K in FP32 through RoPE
//       and narrows at the store; the native block narrows to bf16 before RoPE
//       (dense_attn_block.h:637-653).
//   (b) the KERNEL ARITHMETIC. The primary's executing arm is Triton
//       (prefix_prefill.py, selected because the ROCm custom predicate needs
//       gqa_ratio >= 3 and this fixture has qg == 1) and narrows the softmax
//       probabilities to the V dtype before tl.dot (prefix_prefill.py:471). The
//       native ROCm arm keeps them f32 (rocm_paged_attn.hip:527-531), and its
//       D128/QG1 dispatch has no query-length gate, so the 33-token prefill runs
//       through the decode-geometry kernel (rocm_paged_attn.hip:445,2037-2041).
//
// This instrument separates them with native bytes. It (i) captures the native
// boundary from the unchanged production path, (ii) replays the PRIMARY's own
// captured Q/K/V through the same native op at the primary's cache geometry so
// the kernel term is measured on identical inputs, and (iii) runs the native
// preamble ops over the primary's captured `qkv` so the preamble term is
// measured on identical inputs too.
//
// TEST-ONLY. Every observer below is the provider-seam wrapper this file already
// uses for the residual row (CaptureResidualNorm, :228-237): read the live kernel
// with vt::GetOp, register a priority-100 replacement, forward to the saved
// pointer. Nothing under src/ or include/ changes and no attention arithmetic is
// touched. Everything is gated on VT_ATTN_DUMP; when that is unset no observer is
// registered at all and vt::GetOp dispatch is byte-identical to production.
//
// EAGER ONLY. The capture does device->host copies, which are illegal inside a
// graph capture, and vt::Backend carries no "is capturing" query (gap G5:
// include/vt/backend.h:244-262, include/vt/device.h:131-135). The workload that
// matters never captures: RocmPlatform::support_static_graph_mode() is false
// (src/vllm/platforms/rocm.cpp:91, include/vllm/platforms/interface.h:344), so
// the production registry path does not enter the graph driver.
//
// THE THREE OBSERVERS AND WHY EACH IS ENOUGH.
//  * kPagedAttention — its arguments ARE the post-RoPE boundary: `query` (post-
//    RoPE Q), `k_cache`/`v_cache` (what the kernel reads), `block_table`,
//    `seq_lens`, `query_start_loc` and `out` (include/vt/ops.h:5424-5426,
//    src/vt/ops.cpp:5038-5130).
//  * kRopeFromCache — reached only where the recipe's Tier-0 composite runs
//    (a backend with no fast realisation) or by the standalone fallback. On ROCm
//    the fast realisation IS registered at this head
//    (src/vt/rocm/rocm_ops.hip:331), so with the default VT_FUSED_CHAIN_ADOPT=1
//    the preamble is the fused op and this observer does not fire; with
//    VT_FUSED_CHAIN_ADOPT=0 the bf16 branch calls vt::AttnQkNormRope, which
//    dispatches to that same registered op (dense_attn_block.h:645-649), so it
//    does not fire there either. Where it does fire it sees the post-norm
//    PRE-RoPE q2/k2 on entry, the post-RoPE q3/k3 on return, and the cos|sin
//    table that was actually read — all three without a product-file hook,
//    because the composite reaches the op through GetOp.
//  * kAttnQkNormRope — the recipe's fast realisation, which is the production
//    preamble on this backend for BOTH settings of the adoption switch: ADOPT=1
//    through vt::FusedChain and ADOPT=0 through the hand-call. The runner reads
//    whichever realization executed, so one case measures the boundary under
//    either lever.
//  * kRmsNorm — records every call in order so the pair immediately preceding
//    the captured RopeFromCache call is identified by position in that call
//    sequence rather than by shape (the layer input norms have the same shape).
//
// THE OPERATOR COMMAND. One binary, one process, one run; both cases below skip
// with exit 77 when their environment is absent, so an ordinary test run is
// unchanged. Run it under the host's recorded GPU mutex, with
// `VT_ATTN_DUMP=1` and these six values in the environment:
//
//   VT_ROCM_MOE_FIXTURE        = /home/vikash/.cache/rdna3-moe-impl/preserved/fixture
//   VT_ATTN_PARITY_PRIMARY     = /home/vikash/.cache/rdna3-moe-attention-operator/results
//   VT_ATTN_PARITY_PREAMBLE    = /home/vikash/.cache/residual-norm-impl/oracle-probe-v3/observed
//   VT_ATTN_PARITY_OUT         = a scratch directory for the dumped native bytes
//   HIP_VISIBLE_DEVICES        = 0
//
//   flock /home/vikash/gpu.lock -c '<the five values above> build-attn-hip/tests/test_rocm_moe_bf16 --test-case="ROCm paged attention replays*"'
//
// The CPU-only half needs no device and only VT_ATTN_PARITY_PRIMARY:
//
//   VT_ATTN_PARITY_PRIMARY=<results> build-attn-hip/tests/test_rocm_moe_bf16 --test-case="Primary attention capture decodes*"
//
// MEASURED at 691b7af30 + this instrument (gfx1100, bf16, 66 tokens, hq 1,
// hkv 1, dh 128, L33/C2/R0 layer 0 step 0; all 1322 assertions pass), over the
// primary capture above and the primary preamble capture:
//
//   row                                                        differing / 8448
//   A0 native run output vs primary output (the recorded 2902)          2902
//   B  native run post-RoPE Q vs primary Q                              1569
//   B  native run post-RoPE K vs primary K                              1542
//   C0 replay self-consistency (native Q/K/V at the primary's
//      geometry vs the native run's own output)                             0
//   C  native ROCm kernel on the PRIMARY's own Q/K/V vs its output      2918
//   D  native preamble on the primary's qkv (RmsNorm + RopeFromCache)   1569 Q / 1542 K
//   D' the same through RopeNeox instead of the cos|sin cache           1776 Q / 1719 K
//   E  native cos|sin table vs primary cos_sin at the same positions        0
//   F  primary qkv q/k slices vs the native run's pre-norm q/k              0
//
// VERDICT: BOTH, kernel-dominant. The kernel term alone is 2918 of 8448 words on
// identical inputs (row C, with row C0 proving the replay reproduces the native
// run exactly at the primary's geometry), which is the whole recorded 2902. The
// preamble term is 1569 Q and 1542 K. Rows E and F exclude the cos|sin source
// and the qkv projection, so the preamble term is the pre-RoPE BF16 norm store.
// Row C's shape: rows 0 and 33 (query position 0, a one-key softmax) are exact,
// no differing word lies in a row whose maximum probability is >= 0.999, all
// 2918 lie in rows below 0.9, and 2132 of them are one bf16 code unit apart.
//
// GAP G5 (recorded, not repaired): a device->host read is illegal inside a graph
// capture and vt::Backend carries no "is capturing" query
// (include/vt/backend.h:244-262; vt::Queue is include/vt/device.h:131-135), so
// this instrument cannot decline on its own during a capture. It is declared
// EAGER-ONLY instead: RocmPlatform::support_static_graph_mode() is false, so the
// workload measured here is never captured. A graphed variant needs
// `virtual bool IsCapturing() const` on vt::Backend, overridden in RocmBackend
// beside BeginCapture, and an issue of its own.
// ---------------------------------------------------------------------------
namespace {

// One env var, read once into a function-local static — the tree's convention
// (dense_attn_block.h:65, :93, :154). Unset => no observer is installed, so the
// default path pays nothing and dispatches exactly as it does today.
bool AttnDumpEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ATTN_DUMP");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// --- host file helpers ------------------------------------------------------
std::vector<uint8_t> ReadBytes(const std::filesystem::path& path) {
  REQUIRE_MESSAGE(std::filesystem::exists(path), "missing capture file ", path.string());
  const auto size = static_cast<size_t>(std::filesystem::file_size(path));
  std::vector<uint8_t> bytes(size);
  std::ifstream in(path, std::ios::binary);
  if (size > 0)
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  REQUIRE(static_cast<bool>(in));
  return bytes;
}

template <typename T>
std::vector<T> ReadTyped(const std::filesystem::path& path) {
  const std::vector<uint8_t> bytes = ReadBytes(path);
  REQUIRE(bytes.size() % sizeof(T) == 0);
  std::vector<T> values(bytes.size() / sizeof(T));
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<uint16_t> ReadBf16(const std::filesystem::path& path) {
  return ReadTyped<uint16_t>(path);
}

void WriteWords(const std::filesystem::path& path, const std::vector<uint16_t>& words) {
  if (words.empty()) return;
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(words.data()),
             static_cast<std::streamsize>(words.size() * sizeof(uint16_t)));
}

// --- device read helpers ----------------------------------------------------
std::vector<uint8_t> ReadDevice(vt::Queue& q, const vt::Tensor& tensor) {
  std::vector<uint8_t> host(tensor.Bytes());
  auto& backend = vt::GetBackend(q.device);
  if (!host.empty()) backend.Copy(q, host.data(), tensor.data, host.size());
  backend.Synchronize(q);
  return host;
}

std::vector<uint16_t> ReadDeviceBf16(vt::Queue& q, const vt::Tensor& tensor) {
  REQUIRE(tensor.dtype == vt::DType::kBF16);
  const std::vector<uint8_t> bytes = ReadDevice(q, tensor);
  std::vector<uint16_t> words(bytes.size() / sizeof(uint16_t));
  if (!bytes.empty()) std::memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

template <typename T>
std::vector<T> ReadDeviceTyped(vt::Queue& q, const vt::Tensor& tensor) {
  const std::vector<uint8_t> bytes = ReadDevice(q, tensor);
  REQUIRE(bytes.size() % sizeof(T) == 0);
  std::vector<T> values(bytes.size() / sizeof(T));
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

// The (num_blocks, block_size, hkv, dh) unbind slice is STRIDED: the block
// stride is 2*block_size*hkv*dh because the flash cache interleaves K and V per
// block (dense_attn_block.h:356-373). One linear Copy would read the wrong
// bytes, so read the slice one block at a time.
std::vector<uint16_t> ReadCacheSlice(vt::Queue& q, const vt::Tensor& cache) {
  REQUIRE(cache.rank == 4);
  REQUIRE(cache.dtype == vt::DType::kBF16);
  REQUIRE(cache.stride[3] == 1);
  const int64_t blocks = cache.shape[0], bs = cache.shape[1];
  const int64_t h = cache.shape[2], d = cache.shape[3];
  std::vector<uint16_t> host(static_cast<size_t>(blocks * bs * h * d));
  auto& backend = vt::GetBackend(q.device);
  for (int64_t b = 0; b < blocks; ++b) {
    backend.Copy(q, host.data() + static_cast<size_t>(b * bs * h * d),
                 static_cast<const char*>(cache.data) +
                     static_cast<size_t>(b * cache.stride[0]) * sizeof(uint16_t),
                 static_cast<size_t>(bs * h * d) * sizeof(uint16_t));
  }
  backend.Synchronize(q);
  return host;
}

// The K/V row the kernel reads for request r, key position j:
//   cache[block_table[r, j / block_size], j % block_size, 0, :]
std::vector<uint16_t> GatherKv(const std::vector<uint16_t>& slice, int64_t block_size,
                               int64_t hkv, int64_t dh,
                               const std::vector<int32_t>& block_table,
                               int64_t block_table_cols, int64_t request, int64_t keys) {
  const size_t row = static_cast<size_t>(hkv * dh);
  std::vector<uint16_t> out(static_cast<size_t>(keys) * row);
  for (int64_t j = 0; j < keys; ++j) {
    const int64_t block =
        block_table[static_cast<size_t>(request * block_table_cols + j / block_size)];
    const size_t src = static_cast<size_t>((block * block_size + j % block_size)) * row;
    REQUIRE(src + row <= slice.size());
    std::memcpy(out.data() + static_cast<size_t>(j) * row, slice.data() + src,
                row * sizeof(uint16_t));
  }
  return out;
}

// --- capture records --------------------------------------------------------
struct PagedAttentionCapture {
  int64_t calls = 0;
  int64_t tokens = 0, hq = 0, dh = 0, hkv = 0, block_size = 0, blocks = 0;
  int64_t block_table_cols = 0;
  float scale = 0.0f;
  bool causal = true;
  bool have = false;
  std::vector<uint16_t> query, out, k_cache, v_cache;
  std::vector<int32_t> block_table, seq_lens, query_start_loc;
};

struct RopeCapture {
  bool have = false;
  int64_t tokens = 0, heads = 0, dh = 0, rot = 0, rms_calls_before = 0;
  std::vector<uint16_t> q_pre, k_pre, q_post, k_post, cos_sin;
  std::vector<int32_t> index;
};

struct RmsNormCall {
  int64_t shape0 = 0, shape1 = 0;
  float eps = 0.0f;
  std::vector<uint16_t> in, out, weight;
};

// The fused qk-norm-plus-RoPE preamble op (`vt::OpId::kAttnQkNormRope`). Before
// the #3115 repair ROCm registers no fast realization for the recipe, so the
// production preamble is the Tier-0 composite and this observer never fires.
// After it, the fused op IS the preamble, and the kRopeFromCache/kRmsNorm
// observers above stop firing on the production path. The runner reads whichever
// realization executed, so one case measures one boundary before and after.
struct PreambleCapture {
  bool have = false;
  int64_t tokens = 0, hq = 0, hkv = 0, dh = 0, rot = 0;
  float eps = 0.0f;
  std::vector<uint16_t> q_in, k_in, q_post, k_post;
};

PagedAttentionCapture g_attn;
RopeCapture g_rope;
PreambleCapture g_preamble;
int64_t g_rope_neox_calls = 0;
std::vector<RmsNormCall> g_rms_calls;
bool g_record = false;

vt::PagedAttentionFn g_native_paged_attention = nullptr;
vt::RmsNormFn g_native_rms_norm = nullptr;
vt::RopeFromCacheFn g_native_rope_from_cache = nullptr;
vt::RopeFn g_native_rope_neox = nullptr;
vt::AttnQkNormRopeFn g_native_attn_qk_norm_rope = nullptr;

void CapturePagedAttention(vt::Queue& q, vt::Tensor& out, const vt::Tensor& query,
                           const vt::Tensor& k_cache, const vt::Tensor& v_cache,
                           const vt::Tensor& block_table, const vt::Tensor& seq_lens,
                           const vt::Tensor& query_start_loc,
                           const vt::PagedAttentionArgs& args) {
  const bool record = g_record && g_attn.calls == 0;
  if (record) {
    g_attn.tokens = query.shape[0];
    g_attn.hq = query.shape[1];
    g_attn.dh = query.shape[2];
    g_attn.hkv = k_cache.shape[2];
    g_attn.block_size = k_cache.shape[1];
    g_attn.blocks = k_cache.shape[0];
    g_attn.block_table_cols = block_table.shape[1];
    g_attn.scale = args.scale;
    g_attn.causal = args.causal;
    g_attn.query = ReadDeviceBf16(q, query);
    g_attn.k_cache = ReadCacheSlice(q, k_cache);
    g_attn.v_cache = ReadCacheSlice(q, v_cache);
    g_attn.block_table = ReadDeviceTyped<int32_t>(q, block_table);
    g_attn.seq_lens = ReadDeviceTyped<int32_t>(q, seq_lens);
    g_attn.query_start_loc = ReadDeviceTyped<int32_t>(q, query_start_loc);
  }
  g_native_paged_attention(q, out, query, k_cache, v_cache, block_table, seq_lens,
                           query_start_loc, args);
  if (record) {
    g_attn.out = ReadDeviceBf16(q, out);
    g_attn.have = true;
  }
  g_attn.calls += 1;
}

void CaptureRmsNorm(vt::Queue& q, vt::Tensor& out, const vt::Tensor& x,
                    const vt::Tensor& weight, const vt::RmsNormArgs& args,
                    vt::Tensor* residual) {
  const bool record = g_record && !g_rope.have && g_rms_calls.size() < 32 &&
                      x.Numel() <= 65536 && x.dtype == vt::DType::kBF16 &&
                      out.dtype == vt::DType::kBF16;
  RmsNormCall call;
  if (record) {
    call.shape0 = x.shape[0];
    call.shape1 = x.rank >= 2 ? x.shape[1] : 0;
    call.eps = args.eps;
    call.in = ReadDeviceBf16(q, x);
    call.weight = ReadDeviceBf16(q, weight);
  }
  g_native_rms_norm(q, out, x, weight, args, residual);
  if (record) {
    call.out = ReadDeviceBf16(q, out);
    g_rms_calls.push_back(std::move(call));
  }
}

// The preamble's RoPE call is preceded by RmsNorm(q) then RmsNorm(k) from the
// SAME composite (include/vt/recipes.h kAttnQkNormRope), so the two RmsNorm
// calls immediately before it are the pair, identified by position in the call
// sequence rather than by shape.
void CaptureRopeFromCache(vt::Queue& q, vt::Tensor& q_states, vt::Tensor* k_states,
                          const vt::Tensor& positions, const vt::Tensor& cos_sin_cache,
                          const vt::RopeArgs& args) {
  const bool record = g_record && !g_rope.have;
  if (record) {
    g_rope.tokens = q_states.shape[0];
    g_rope.heads = q_states.shape[1];
    g_rope.dh = q_states.shape[2];
    g_rope.rot = args.rotary_dim;
    g_rope.rms_calls_before = static_cast<int64_t>(g_rms_calls.size());
    g_rope.q_pre = ReadDeviceBf16(q, q_states);
    if (k_states != nullptr) g_rope.k_pre = ReadDeviceBf16(q, *k_states);
    g_rope.cos_sin = ReadDeviceBf16(q, cos_sin_cache);
    g_rope.index = ReadDeviceTyped<int32_t>(q, positions);
  }
  g_native_rope_from_cache(q, q_states, k_states, positions, cos_sin_cache, args);
  if (record) {
    g_rope.q_post = ReadDeviceBf16(q, q_states);
    if (k_states != nullptr) g_rope.k_post = ReadDeviceBf16(q, *k_states);
    g_rope.have = true;
  }
}

void CaptureRopeNeox(vt::Queue& q, vt::Tensor& q_states, vt::Tensor& k_states,
                     const vt::Tensor& positions, const vt::RopeArgs& args) {
  g_rope_neox_calls += 1;
  g_native_rope_neox(q, q_states, k_states, positions, args);
}

// The fused preamble norms and rotates q3/k3 IN PLACE (include/vt/ops.h:2285),
// so the entry bytes ARE the qkv projection's own slice and the return bytes ARE
// the post-RoPE boundary the attention kernel consumes.
void CaptureAttnQkNormRope(vt::Queue& q, vt::Tensor& q3, vt::Tensor& k3,
                           const vt::Tensor& q_norm, const vt::Tensor& k_norm,
                           const vt::Tensor& cos_sin, const vt::Tensor& positions,
                           const vt::RmsNormArgs& norm_args, const vt::RopeArgs& rope_args) {
  const bool record = g_record && !g_preamble.have;
  if (record) {
    g_preamble.tokens = q3.shape[0];
    g_preamble.hq = q3.shape[1];
    g_preamble.dh = q3.shape[2];
    g_preamble.hkv = k3.shape[1];
    g_preamble.rot = rope_args.rotary_dim;
    g_preamble.eps = norm_args.eps;
    g_preamble.q_in = ReadDeviceBf16(q, q3);
    g_preamble.k_in = ReadDeviceBf16(q, k3);
    (void)q_norm;
    (void)k_norm;
    (void)cos_sin;
    (void)positions;
  }
  g_native_attn_qk_norm_rope(q, q3, k3, q_norm, k_norm, cos_sin, positions, norm_args,
                            rope_args);
  if (record) {
    g_preamble.q_post = ReadDeviceBf16(q, q3);
    g_preamble.k_post = ReadDeviceBf16(q, k3);
    g_preamble.have = true;
  }
}

// Function-local static: the provider table is process-global and registering
// the same name twice is refused, so this runs exactly once. GetOp is read here
// and NOWHERE else, so on a re-entry the saved pointer is the live kernel and
// never this wrapper.
void InstallAttnObservers() {
  static const bool installed = [] {
    g_native_paged_attention = reinterpret_cast<vt::PagedAttentionFn>(
        vt::GetOp(vt::OpId::kPagedAttention, vt::DeviceType::kROCM));
    vt::RegisterOpProvider(vt::OpId::kPagedAttention, vt::DeviceType::kROCM,
                           {"test-attention-boundary-paged-attention", 100, nullptr,
                            reinterpret_cast<void*>(static_cast<vt::PagedAttentionFn>(
                                &CapturePagedAttention))});
    g_native_rms_norm = reinterpret_cast<vt::RmsNormFn>(
        vt::GetOp(vt::OpId::kRmsNorm, vt::DeviceType::kROCM));
    vt::RegisterOpProvider(vt::OpId::kRmsNorm, vt::DeviceType::kROCM,
                           {"test-attention-boundary-rmsnorm", 100, nullptr,
                            reinterpret_cast<void*>(
                                static_cast<vt::RmsNormFn>(&CaptureRmsNorm))});
    g_native_rope_from_cache = reinterpret_cast<vt::RopeFromCacheFn>(
        vt::GetOp(vt::OpId::kRopeFromCache, vt::DeviceType::kROCM));
    vt::RegisterOpProvider(vt::OpId::kRopeFromCache, vt::DeviceType::kROCM,
                           {"test-attention-boundary-rope-from-cache", 100, nullptr,
                            reinterpret_cast<void*>(static_cast<vt::RopeFromCacheFn>(
                                &CaptureRopeFromCache))});
    g_native_rope_neox =
        reinterpret_cast<vt::RopeFn>(vt::GetOp(vt::OpId::kRopeNeox, vt::DeviceType::kROCM));
    vt::RegisterOpProvider(vt::OpId::kRopeNeox, vt::DeviceType::kROCM,
                           {"test-attention-boundary-rope-neox", 100, nullptr,
                            reinterpret_cast<void*>(
                                static_cast<vt::RopeFn>(&CaptureRopeNeox))});
    // Registered only when the backend HAS the fused op. Before the #3115 repair
    // ROCm registers none, GetOp would return null, and the wrapper would
    // forward into a null pointer; the composite observers above are the
    // preamble source on that tree.
    if (vt::OpRegistered(vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM)) {
      g_native_attn_qk_norm_rope = reinterpret_cast<vt::AttnQkNormRopeFn>(
          vt::GetOp(vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM));
      vt::RegisterOpProvider(
          vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM,
          {"test-attention-boundary-attn-qk-norm-rope", 100, nullptr,
           reinterpret_cast<void*>(
               static_cast<vt::AttnQkNormRopeFn>(&CaptureAttnQkNormRope))});
    }
    return true;
  }();
  (void)installed;
}

// --- the primary's captured attention boundary ------------------------------
struct PrimaryBoundary {
  int64_t tokens = 0, hq = 0, hkv = 0, dh = 0, num_reqs = 0;
  int64_t block_size = 0, block_table_cols = 0;
  float scale = 0.0f;
  std::vector<uint16_t> query, key, value, output;
  std::vector<int32_t> query_start_loc, seq_lens, block_table;
  std::vector<int64_t> slot_mapping;
};

PrimaryBoundary LoadPrimaryBoundary(const std::filesystem::path& dir, const std::string& label,
                                    int index) {
  const std::string stem = label + "-attention-" + std::to_string(index);
  nlohmann::json record;
  std::ifstream(dir / (stem + ".json")) >> record;
  REQUIRE(!record.is_null());
  PrimaryBoundary b;
  b.scale = record.at("scale").get<float>();
  b.hq = record.at("num_heads").get<int64_t>();
  b.hkv = record.at("num_kv_heads").get<int64_t>();
  b.dh = record.at("head_size").get<int64_t>();
  b.tokens = record.at("num_actual_tokens").get<int64_t>();
  b.block_size = record.at("kv_cache").at("shape").at(2).get<int64_t>();
  b.block_table_cols = record.at("tensors").at("block_table").at("shape").at(1).get<int64_t>();
  b.num_reqs = static_cast<int64_t>(record.at("seq_lens").size());
  b.query = ReadBf16(dir / (stem + "-query.bin"));
  b.key = ReadBf16(dir / (stem + "-key.bin"));
  b.value = ReadBf16(dir / (stem + "-value.bin"));
  b.output = ReadBf16(dir / (stem + "-output.bin"));
  b.query_start_loc = ReadTyped<int32_t>(dir / (stem + "-query_start_loc.bin"));
  b.seq_lens = ReadTyped<int32_t>(dir / (stem + "-seq_lens.bin"));
  b.block_table = ReadTyped<int32_t>(dir / (stem + "-block_table.bin"));
  b.slot_mapping = ReadTyped<int64_t>(dir / (stem + "-slot_mapping.bin"));
  const size_t expected = static_cast<size_t>(b.tokens * b.hq * b.dh);
  REQUIRE(b.query.size() == expected);
  REQUIRE(b.key.size() == expected);
  REQUIRE(b.value.size() == expected);
  REQUIRE(b.output.size() == expected);
  return b;
}

// --- diffs ------------------------------------------------------------------
struct WordDiff {
  int64_t words = 0, different = 0;
  std::vector<int64_t> row_different;
  int64_t distance[5] = {0, 0, 0, 0, 0};
  std::vector<int64_t> first;
};

// bf16 shares f32's ordering, so reading the 16 bits as a signed integer is
// monotone within one sign and the difference is a code-unit (ULP) distance.
int64_t Bf16CodeDistance(uint16_t a, uint16_t b) {
  return std::llabs(static_cast<long long>(static_cast<int16_t>(a)) -
                    static_cast<long long>(static_cast<int16_t>(b)));
}

WordDiff DiffWords(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected,
                   int64_t row) {
  REQUIRE(actual.size() == expected.size());
  WordDiff d;
  d.words = static_cast<int64_t>(actual.size());
  if (row > 0) d.row_different.assign(static_cast<size_t>(d.words / row), 0);
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected[i]) continue;
    d.different += 1;
    const int64_t distance = Bf16CodeDistance(actual[i], expected[i]);
    d.distance[distance >= 4 ? 4 : distance] += 1;
    if (row > 0) d.row_different[i / static_cast<size_t>(row)] += 1;
    if (d.first.size() < 8) d.first.push_back(static_cast<int64_t>(i));
  }
  return d;
}

// Host-side f32 classification of each query row: for query token t the maximum
// softmax probability over the keys it can see. A difference confined to rows
// whose maximum is ~1 (a trivial softmax) points at exp/reduction order; a
// difference spread over rows with a flat distribution points at the
// probability dtype.
std::vector<double> MaxProbPerRow(const PrimaryBoundary& b) {
  std::vector<double> result(static_cast<size_t>(b.tokens), 0.0);
  for (int64_t r = 0; r < b.num_reqs; ++r) {
    const int64_t begin = b.query_start_loc[static_cast<size_t>(r)];
    const int64_t end = b.query_start_loc[static_cast<size_t>(r + 1)];
    const int64_t seq = b.seq_lens[static_cast<size_t>(r)];
    for (int64_t t = begin; t < end; ++t) {
      const int64_t position = seq - (end - begin) + (t - begin);
      std::vector<double> scores(static_cast<size_t>(position + 1), 0.0);
      for (int64_t j = 0; j <= position; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < b.dh; ++d)
          dot += static_cast<double>(vt::BF16ToF32(b.query[static_cast<size_t>(t * b.dh + d)])) *
                 static_cast<double>(vt::BF16ToF32(b.key[static_cast<size_t>(j * b.dh + d)]));
        scores[static_cast<size_t>(j)] = static_cast<double>(b.scale) * dot;
      }
      const double max = *std::max_element(scores.begin(), scores.end());
      double sum = 0.0;
      for (double s : scores) sum += std::exp(s - max);
      result[static_cast<size_t>(t)] = 1.0 / sum;
    }
  }
  return result;
}

void PrintDiff(const std::string& tag, const WordDiff& d) {
  std::cout << "[attn-parity] " << tag << ": " << d.different << " / " << d.words
            << " differing bf16 words; bf16 code-unit distance 1:" << d.distance[1]
            << " 2:" << d.distance[2] << " 3:" << d.distance[3] << " >=4:" << d.distance[4]
            << std::endl;
}

// WHERE the words differ, in the two axes that localize the cause.
void PrintDiffShape(const PrimaryBoundary& b, const WordDiff& d,
                    const std::vector<double>& max_prob) {
  std::cout << "[attn-parity]   per-row differing words (" << d.row_different.size()
            << " rows):";
  for (size_t t = 0; t < d.row_different.size(); ++t)
    std::cout << " " << d.row_different[t];
  std::cout << std::endl;
  const double edges[3] = {0.999, 0.99, 0.9};
  int64_t words[4] = {0, 0, 0, 0}, rows[4] = {0, 0, 0, 0}, total[4] = {0, 0, 0, 0};
  for (size_t t = 0; t < max_prob.size(); ++t) {
    const int bucket = max_prob[t] >= edges[0]   ? 0
                       : max_prob[t] >= edges[1] ? 1
                       : max_prob[t] >= edges[2] ? 2
                                                 : 3;
    total[bucket] += 1;
    if (t < d.row_different.size() && d.row_different[t] > 0) rows[bucket] += 1;
    if (t < d.row_different.size()) words[bucket] += d.row_different[t];
  }
  const char* names[4] = {"maxprob>=0.999", "0.99<=maxprob<0.999", "0.9<=maxprob<0.99",
                          "maxprob<0.9"};
  for (int i = 0; i < 4; ++i)
    std::cout << "[attn-parity]   " << names[i] << ": " << words[i] << " differing words over "
              << rows[i] << " of " << total[i] << " rows" << std::endl;
  if (!d.row_different.empty())
    std::cout << "[attn-parity]   row 0 (position 0, softmax trivial): "
              << d.row_different[0] << " differing words; first differing flat indices:";
  for (int64_t index : d.first) std::cout << " " << index;
  std::cout << std::endl;
  std::cout << "[attn-parity]   requests: " << b.num_reqs;
  for (int64_t r = 0; r < b.num_reqs; ++r) {
    int64_t count = 0;
    for (int64_t t = b.query_start_loc[static_cast<size_t>(r)];
         t < b.query_start_loc[static_cast<size_t>(r + 1)]; ++t)
      if (t < static_cast<int64_t>(d.row_different.size()))
        count += d.row_different[static_cast<size_t>(t)];
    std::cout << " request " << r << "=" << count;
  }
  std::cout << std::endl;
}

// The primary's captured Q/K preamble boundary, beside the attention capture.
struct PrimaryPreamble {
  int64_t tokens = 0, qkv_width = 0, dh = 0, rot = 0;
  std::vector<uint16_t> qkv, q_gamma, k_gamma, cos_sin, q_left, q_right, k_left, k_right;
  std::vector<int64_t> positions;
};

PrimaryPreamble LoadPrimaryPreamble(const std::filesystem::path& dir, const std::string& label,
                                    int index) {
  const std::string stem = label + "-residual-preamble-" + std::to_string(index);
  nlohmann::json record;
  std::ifstream(dir / (stem + ".json")) >> record;
  REQUIRE(!record.is_null());
  PrimaryPreamble p;
  p.tokens = record.at("inputs").at("qkv").at("shape").at(0).get<int64_t>();
  p.qkv_width = record.at("inputs").at("qkv").at("shape").at(1).get<int64_t>();
  p.dh = record.at("inputs").at("q_gamma").at("shape").at(0).get<int64_t>();
  p.rot = record.at("inputs").at("cos_sin").at("shape").at(1).get<int64_t>();
  p.qkv = ReadBf16(dir / (stem + "-input-qkv.bin"));
  p.q_gamma = ReadBf16(dir / (stem + "-input-q_gamma.bin"));
  p.k_gamma = ReadBf16(dir / (stem + "-input-k_gamma.bin"));
  p.cos_sin = ReadBf16(dir / (stem + "-input-cos_sin.bin"));
  p.positions = ReadTyped<int64_t>(dir / (stem + "-input-positions.bin"));
  p.q_left = ReadBf16(dir / (stem + "-output-q_left.bin"));
  p.q_right = ReadBf16(dir / (stem + "-output-q_right.bin"));
  p.k_left = ReadBf16(dir / (stem + "-output-k_left.bin"));
  p.k_right = ReadBf16(dir / (stem + "-output-k_right.bin"));
  REQUIRE(p.qkv.size() == static_cast<size_t>(p.tokens * p.qkv_width));
  REQUIRE(p.cos_sin.size() ==
          static_cast<size_t>(record.at("inputs").at("cos_sin").at("shape").at(0).get<int64_t>() *
                              p.rot));
  REQUIRE(p.positions.size() == static_cast<size_t>(p.tokens));
  return p;
}

// q_left|q_right (each [T,1,Dh/2], stride [Dh,Dh,1]) reassembled into the
// [T,1,Dh] query the attention capture holds.
std::vector<uint16_t> JoinRopeHalves(const PrimaryPreamble& p, const std::vector<uint16_t>& left,
                                     const std::vector<uint16_t>& right) {
  const int64_t half = p.dh / 2;
  std::vector<uint16_t> joined(static_cast<size_t>(p.tokens * p.dh), 0);
  for (int64_t t = 0; t < p.tokens; ++t) {
    for (int64_t d = 0; d < half; ++d) {
      joined[static_cast<size_t>(t * p.dh + d)] = left[static_cast<size_t>(t * half + d)];
      joined[static_cast<size_t>(t * p.dh + half + d)] = right[static_cast<size_t>(t * half + d)];
    }
  }
  return joined;
}

// One head-contiguous row of the flash cache, at the address the capture's own
// slot_mapping names: block = slot / block_size, offset = slot % block_size.
void ScatterCacheSlot(std::vector<uint16_t>& cache, int64_t blocks, int64_t block_size,
                      int64_t hkv, int64_t dh, int which, int64_t slot, const uint16_t* row) {
  const int64_t block = slot / block_size;
  const int64_t offset = slot % block_size;
  REQUIRE(block >= 0);
  REQUIRE(block < blocks);
  const size_t base = static_cast<size_t>((block * 2 + which) * block_size * hkv * dh +
                                          offset * hkv * dh);
  REQUIRE(base + static_cast<size_t>(hkv * dh) <= cache.size());
  std::memcpy(cache.data() + base, row, static_cast<size_t>(hkv * dh) * sizeof(uint16_t));
}

// --- the arm's key walk, on synthesized data --------------------------------
//
// The reference max the probability is narrowed against is not a free parameter:
// the primary takes it per contiguous key tile, and the tile is chosen by the arm
// that executes for that row (## Design 1, .agents/specs/rocm-attn-parity.md).
// The recorded capture cannot witness the choice: its 33-token rows are all
// prefix_prefill with an EMPTY context, so `max(keys 0..31) == max(keys 0..32)`
// and a 32-key tile reproduces the same 8448 bytes. These cases synthesize the
// data that makes the width load-bearing, and compare the device against a host
// transcription of the primary's own key walk for the arm (`PrimaryTiles` +
// `HostArmOutput`) and against the same transcription under neighbouring widths.
//
// Data design. q = e0 and every K row is either 0 or `kArmHighKey` in element 0,
// so every score is exactly 0 or s = kArmHighKey * scale (bf16-exact operands, a
// single nonzero product, so the dot is order-independent). Every V row is a
// single bf16-exact integer in its own lane, so each output lane is fed by one
// key and the narrowed-probability accumulations are sums of bf16-exact values.
// A key that lies in a tile with a high key is narrowed at exp(-s); the same key
// alone in a tile is narrowed at 1. The two differ by a relative 2^-9, which is
// what moves the output word -- and it only exists when the tile is right.
constexpr float kArmHighKey = 5.65625f;  // bf16-exact; s = 0.4999466
constexpr int64_t kArmDh = 128;

// The kernel's FastExp (src/vt/rocm/rocm_paged_attn.hip:196-198) on the host.
// Only exp2(0) = 1 and exp2(-inf) = 0 are load-bearing here.
float ArmFastExp(float x) { return std::exp2(x * 1.4426950408889634f); }

std::vector<uint16_t> ArmWords(const std::vector<float>& values) {
  std::vector<uint16_t> words(values.size());
  for (size_t i = 0; i < values.size(); ++i) words[i] = vt::F32ToBF16(values[i]);
  return words;
}

// The primary's key walk for ONE query row, as absolute [begin, end) key tiles.
// chunked selects prefix_prefill; the widths are its pinned constants:
// TRITON_BLOCK_SIZE = 32 for the cached context (:965, :1007) and BLOCK_N = 64
// for the current chunk when the physical block size is a power of two, 32
// otherwise (:955-966). The decode arm tiles by min(block_size, 128)
// (chunked_prefill_paged_decode.py:444-445), also 32 for a non-power-of-two
// physical block size.
std::vector<std::pair<int64_t, int64_t>> PrimaryTiles(bool chunked, int64_t context,
                                                      int64_t jmax, int64_t block_size) {
  const bool pow2 = block_size > 0 && (block_size & (block_size - 1)) == 0;
  std::vector<std::pair<int64_t, int64_t>> tiles;
  if (chunked) {
    for (int64_t base = 0; base < context; base += 32)
      tiles.emplace_back(base, std::min(base + 32, context));
  }
  const int64_t width = chunked ? (pow2 ? 64 : 32)
                                : (pow2 ? std::min<int64_t>(block_size, 128) : 32);
  for (int64_t base = chunked ? context : 0; base <= jmax; base += width)
    tiles.emplace_back(base, std::min(base + width, jmax + 1));
  return tiles;
}

// A uniform-width tiling from key 0, the geometry this kernel used before the
// repair. Used as the falsifying counterfactual: a case whose device bytes match
// the primary arm's tiles and NOT these is a case the width is load-bearing for.
std::vector<std::pair<int64_t, int64_t>> UniformTiles(int64_t jmax, int64_t width) {
  std::vector<std::pair<int64_t, int64_t>> tiles;
  for (int64_t base = 0; base <= jmax; base += width)
    tiles.emplace_back(base, std::min(base + width, jmax + 1));
  return tiles;
}

// The primary's softmax over one query row, transcribed from
// prefix_prefill.py:442-478 (context loop :231-343, chunk loop :369) and
// chunked_prefill_paged_decode.py:244-268. `key_score[j]` is the f32 q.k for key
// j, `values[j*dh + d]` the f32 V row, `tiles` the arm's key walk.
std::vector<float> HostArmOutput(const std::vector<float>& key_score,
                                 const std::vector<float>& values,
                                 const std::vector<std::pair<int64_t, int64_t>>& tiles,
                                 float scale, int64_t dh) {
  const float ninf = -std::numeric_limits<float>::infinity();
  float m = ninf, l = 0.0f;
  std::vector<float> acc(static_cast<size_t>(dh), 0.0f);
  for (const auto& tile : tiles) {
    float tile_max = ninf;
    for (int64_t j = tile.first; j < tile.second; ++j)
      tile_max = std::max(tile_max, key_score[static_cast<size_t>(j)] * scale);
    const float m_new = std::max(m, tile_max);
    const float alpha = ArmFastExp(m - m_new);
    for (float& a : acc) a *= alpha;
    l *= alpha;
    m = m_new;
    for (int64_t j = tile.first; j < tile.second; ++j) {
      const float p = ArmFastExp(key_score[static_cast<size_t>(j)] * scale - m_new);
      const float pw = vt::BF16ToF32(vt::F32ToBF16(p));  // p.to(v.dtype)
      for (int64_t d = 0; d < dh; ++d)
        acc[static_cast<size_t>(d)] += pw * values[static_cast<size_t>(j * dh + d)];
      l += p;
    }
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  for (float& a : acc) a *= inv;
  return acc;
}

// One synthesized workload: `num_reqs` requests of `query_len` query tokens each
// over a cache that already holds `seq_len - query_len` keys, with exactly one
// high-scoring key at `high_key`. Every request has the same shape, so the arm
// is the same for every row of the batch.
struct SyntheticArm {
  int64_t num_reqs = 0, query_len = 0, seq_len = 0, block_size = 0, high_key = 0;
  int64_t dh = kArmDh, hq = 1, hkv = 1, blocks = 0, block_table_cols = 0, total_q = 0;
  float scale = 0.0f;
  std::vector<int32_t> query_start_loc, seq_lens, block_table;
  std::vector<uint16_t> query, k_cache, v_cache;
  std::vector<float> key_score, values;  // host f32 copies for the model
};

SyntheticArm BuildSyntheticArm(int64_t num_reqs, int64_t query_len, int64_t seq_len,
                               int64_t block_size, int64_t high_key) {
  SyntheticArm a;
  a.num_reqs = num_reqs;
  a.query_len = query_len;
  a.seq_len = seq_len;
  a.block_size = block_size;
  a.high_key = high_key;
  a.scale = 1.0f / std::sqrt(static_cast<float>(kArmDh));
  a.total_q = num_reqs * query_len;
  a.block_table_cols = (seq_len + block_size - 1) / block_size;
  a.blocks = num_reqs * a.block_table_cols;
  const size_t row = static_cast<size_t>(a.dh);
  a.query.assign(static_cast<size_t>(a.total_q) * row, 0);
  a.k_cache.assign(static_cast<size_t>(a.blocks * block_size) * row, 0);
  a.v_cache.assign(static_cast<size_t>(a.blocks * block_size) * row, 0);
  a.key_score.assign(static_cast<size_t>(seq_len), 0.0f);
  a.values.assign(static_cast<size_t>(seq_len) * row, 0.0f);
  a.query_start_loc.assign(static_cast<size_t>(num_reqs) + 1, 0);
  for (int64_t j = 0; j < seq_len; ++j) {
    a.key_score[static_cast<size_t>(j)] = j == high_key ? kArmHighKey : 0.0f;
    // Key j's value row is a single bf16-exact integer in lane j, so each output
    // lane is fed by exactly one key and the narrowing is what the lane shows.
    a.values[static_cast<size_t>(j * a.dh + (j % a.dh))] = static_cast<float>(64 + j % 8);
  }
  for (int64_t r = 0; r < num_reqs; ++r) {
    a.query_start_loc[static_cast<size_t>(r + 1)] = static_cast<int32_t>((r + 1) * query_len);
    a.seq_lens.push_back(static_cast<int32_t>(seq_len));
    for (int64_t c = 0; c < a.block_table_cols; ++c)
      a.block_table.push_back(static_cast<int32_t>(r * a.block_table_cols + c));
    for (int64_t j = 0; j < seq_len; ++j) {
      const int64_t slot = static_cast<int64_t>(
                               a.block_table[static_cast<size_t>(r * a.block_table_cols +
                                                                 j / block_size)]) *
                               block_size +
                           j % block_size;
      const size_t base = static_cast<size_t>(slot) * row;
      a.k_cache[base] = vt::F32ToBF16(a.key_score[static_cast<size_t>(j)]);
      a.v_cache[base + static_cast<size_t>(j % a.dh)] =
          vt::F32ToBF16(a.values[static_cast<size_t>(j * a.dh + j % a.dh)]);
    }
    for (int64_t i = 0; i < query_len; ++i) {
      const size_t t = static_cast<size_t>(r * query_len + i);
      a.query[t * row] = vt::F32ToBF16(1.0f);  // q = e0, so q.k_j = key_score[j]
    }
  }
  return a;
}

// One device forward of the workload through the shared op layer, returning the
// [total_q, hq, dh] BF16 output.
std::vector<uint16_t> RunSyntheticArm(vt::Backend& backend, vt::Queue& q,
                                      const SyntheticArm& a) {
  Buffer qb(backend, a.query.size() * sizeof(uint16_t));
  Buffer kb(backend, a.k_cache.size() * sizeof(uint16_t));
  Buffer vb(backend, a.v_cache.size() * sizeof(uint16_t));
  Buffer ob(backend, a.query.size() * sizeof(uint16_t));
  Buffer tbb(backend, a.block_table.size() * sizeof(int32_t));
  Buffer slb(backend, a.seq_lens.size() * sizeof(int32_t));
  Buffer qlb(backend, a.query_start_loc.size() * sizeof(int32_t));
  backend.Copy(q, qb.data, a.query.data(), a.query.size() * sizeof(uint16_t));
  backend.Copy(q, kb.data, a.k_cache.data(), a.k_cache.size() * sizeof(uint16_t));
  backend.Copy(q, vb.data, a.v_cache.data(), a.v_cache.size() * sizeof(uint16_t));
  backend.Copy(q, tbb.data, a.block_table.data(), a.block_table.size() * sizeof(int32_t));
  backend.Copy(q, slb.data, a.seq_lens.data(), a.seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, qlb.data, a.query_start_loc.data(),
               a.query_start_loc.size() * sizeof(int32_t));
  backend.Synchronize(q);
  vt::Tensor tq = Tensor(qb.data, vt::DType::kBF16, q.device, {a.total_q, a.hq, a.dh});
  vt::Tensor tk = Tensor(kb.data, vt::DType::kBF16, q.device, {a.blocks, a.block_size, a.hkv, a.dh});
  vt::Tensor tv = Tensor(vb.data, vt::DType::kBF16, q.device, {a.blocks, a.block_size, a.hkv, a.dh});
  vt::Tensor to = Tensor(ob.data, vt::DType::kBF16, q.device, {a.total_q, a.hq, a.dh});
  vt::Tensor tt = Tensor(tbb.data, vt::DType::kI32, q.device, {a.num_reqs, a.block_table_cols});
  vt::Tensor ts = Tensor(slb.data, vt::DType::kI32, q.device, {a.num_reqs});
  vt::Tensor tl = Tensor(qlb.data, vt::DType::kI32, q.device, {a.num_reqs + 1});
  vt::PagedAttentionArgs args;
  args.scale = a.scale;
  args.causal = true;
  args.query_start_loc_host = a.query_start_loc.data();
  args.max_seq_len = a.seq_len;
  vt::PagedAttention(q, to, tq, tk, tv, tt, ts, tl, args);
  backend.Synchronize(q);
  return ReadDeviceBf16(q, to);
}

// The model's output for the whole batch, one arm geometry for every row.
std::vector<uint16_t> ArmModelWords(const SyntheticArm& a, bool chunked, int64_t tile_override) {
  std::vector<uint16_t> words;
  for (int64_t r = 0; r < a.num_reqs; ++r) {
    const int64_t context = a.seq_len - a.query_len;
    for (int64_t i = 0; i < a.query_len; ++i) {
      const int64_t jmax = context + i;  // causal: the row sits at position jmax
      const std::vector<std::pair<int64_t, int64_t>> tiles =
          tile_override > 0 ? UniformTiles(jmax, tile_override)
                            : PrimaryTiles(chunked, context, jmax, a.block_size);
      const std::vector<float> out =
          HostArmOutput(a.key_score, a.values, tiles, a.scale, a.dh);
      const std::vector<uint16_t> row_words = ArmWords(out);
      words.insert(words.end(), row_words.begin(), row_words.end());
    }
  }
  return words;
}

void PrintArmDiff(const char* what, const WordDiff& d, int64_t rows) {
  std::cout << "[arm-geometry] " << what << ": " << d.different << " / " << d.words
            << " differing words over " << rows << " rows" << std::endl;
}

// The noise floor is the host/device accumulation-order difference on the f32
// running sum; the signal is the narrowed-probability scale the tile decides.
// They are three orders of magnitude apart on this data, so the thresholds are
// not tuned to the measurement.
constexpr int64_t kArmNoiseWords = 8;
constexpr int64_t kArmSignalWords = 32;

// ─── env-gated cases: per-case skip, process-level 77 ────────────────────────
//
// Four cases in this file need an environment the default run does not have: the
// two production cases need `VT_ROCM_MOE_FIXTURE`, the CPU half of the instrument
// needs `VT_ATTN_PARITY_PRIMARY`, and the device half needs all of that plus
// `VT_ATTN_DUMP=1`. They used to call `std::exit(77)` on the absent variable, and
// doctest runs the cases in file order, so the first absent variable ended the
// PROCESS: the summary was never printed and every later case was silently not
// run (the fresh review's fourth finding).
//
// The cases are now decorated `doctest::skip(...)`, so doctest reports them
// skipped in its own summary and the remaining cases still run, and the process
// still exits 77 -- CTest reports Skipped, the convention `tests/CMakeLists.txt`
// registers -- but ONLY when nothing failed, so a reddened case keeps doctest's
// own non-zero status instead of being folded into the skip. `--no-skip` forces
// a gated case anyway; the in-case guard then FAILS rather than reading an absent
// variable as a path.
bool AttnCaptureAbsent() {
  const char* e = std::getenv("VT_ATTN_PARITY_PRIMARY");
  return e == nullptr || e[0] == '\0';
}
bool AttnInstrumentAbsent() {
  return FixtureAbsent() || AttnCaptureAbsent() || !AttnDumpEnabled();
}

// The four backend-requiring cases near the end of this file measure the ROCm
// arm on data the file builds itself, so they need the ROCm backend registered
// but no captured oracle and no environment variable. On the CPU-only lane
// (VLLM_CPP_HIP=OFF) nothing registers that backend, and a bare REQUIRE would
// FATAL there instead of reporting the CTest Skipped the rest of this file's
// ROCm-only cases report.
bool RocmBackendAbsent() {
  return vt::TryGetBackend(vt::DeviceType::kROCM) == nullptr;
}

bool g_env_cases_skipped = false;
bool g_run_failed = false;

// A listener, not a reporter: listeners are always active whatever `-r=` selects,
// so neither the skip note nor the failure guard can be switched off from the
// command line (the sibling head test registers the same shape).
struct GatedCaseListener : public doctest::IReporter {
  explicit GatedCaseListener(const doctest::ContextOptions&) {}
  void test_run_start() override {}
  void report_query(const doctest::QueryData&) override {}
  void test_run_end(const doctest::TestRunStats& stats) override {
    g_run_failed = stats.numTestCasesFailed != 0 || stats.numAssertsFailed != 0;
  }
  void test_case_start(const doctest::TestCaseData&) override {}
  void test_case_reenter(const doctest::TestCaseData&) override {}
  void test_case_end(const doctest::CurrentTestCaseStats&) override {}
  void test_case_exception(const doctest::TestCaseException&) override {}
  void subcase_start(const doctest::SubcaseSignature&) override {}
  void subcase_end() override {}
  void log_assert(const doctest::AssertData&) override {}
  void log_message(const doctest::MessageData&) override {}
  // Also fires for a case a filter excluded; only the decorator sets `m_skip`,
  // and only that means a prerequisite this case needs is absent -- an unset
  // environment variable for the captured-oracle cases, a registered backend for
  // the synthesized-data ones. `TestCaseData` carries no reason, so the note
  // below has to stay true for either.
  void test_case_skipped(const doctest::TestCaseData& tc) override {
    if (!tc.m_skip) return;
    g_env_cases_skipped = true;
    std::cout << "[rocm-moe-bf16] SKIPPED: " << tc.m_name
              << " (a prerequisite this case needs is absent; see the case's own message)"
              << std::endl;
  }
};
DOCTEST_REGISTER_LISTENER("vt-rocm-moe-env-gated", 1, GatedCaseListener);

// Registered during static initialization, so it runs after doctest's main has
// printed its summary and returned. `std::_Exit` rather than `std::exit`: this IS
// an exit handler, and re-entering the exit sequence is undefined.
void ExitSkippedWhenAGatedCaseDidNotRun() {
  if (!g_env_cases_skipped || g_run_failed) return;
  std::cout.flush();
  std::fflush(nullptr);
  std::fprintf(stderr,
               "\n*** SKIPPED (exit 77): at least one env-gated case did not run. The cases that "
               "did run are reported in the summary above and their result stands; this status "
               "says only that the gated ones did not. ***\n\n");
  std::fflush(stderr);
  std::_Exit(77);
}

struct RegisterExitSkippedWhenAGatedCaseDidNotRun {
  RegisterExitSkippedWhenAGatedCaseDidNotRun() {
    std::atexit(&ExitSkippedWhenAGatedCaseDidNotRun);
  }
};
[[maybe_unused]] const RegisterExitSkippedWhenAGatedCaseDidNotRun
    g_exit_skipped_when_a_gated_case_did_not_run;

}  // namespace

// The replay's cache geometry, checked on the CPU against the recorded capture
// before anything reaches a device. The replay is only meaningful if
// slot_mapping, block_table and block_size agree with each other, and this case
// is what makes that a gate rather than an assumption. It needs no GPU: it is
// the red-first half of the instrument and it runs wherever the artifact is.
TEST_CASE("Primary attention capture decodes to the cache geometry the replay uses" *
          doctest::skip(AttnCaptureAbsent())) {
  REQUIRE_MESSAGE(!AttnCaptureAbsent(),
                  "VT_ATTN_PARITY_PRIMARY must name the primary attention-capture directory");
  const char* results = std::getenv("VT_ATTN_PARITY_PRIMARY");
  const std::filesystem::path dir(results);
  const std::string label = std::getenv("VT_ATTN_PARITY_LABEL") != nullptr
                                ? std::getenv("VT_ATTN_PARITY_LABEL")
                                : "L33-C2-R0";
  const PrimaryBoundary primary = LoadPrimaryBoundary(dir, label, 0);
  CHECK(primary.tokens == 66);
  CHECK(primary.hq == 1);
  CHECK(primary.hkv == 1);
  CHECK(primary.dh == 128);
  CHECK(primary.block_size == 16);
  CHECK(primary.num_reqs == 2);
  CHECK(primary.query_start_loc == std::vector<int32_t>({0, 33, 66}));
  CHECK(primary.seq_lens == std::vector<int32_t>({33, 33}));
  CHECK(primary.block_table.size() ==
        static_cast<size_t>(primary.num_reqs * primary.block_table_cols));
  CHECK(std::vector<int32_t>(primary.block_table.begin(), primary.block_table.begin() + 3) ==
        std::vector<int32_t>({7, 8, 9}));
  CHECK(std::vector<int32_t>(primary.block_table.begin() + primary.block_table_cols,
                             primary.block_table.begin() + primary.block_table_cols + 3) ==
        std::vector<int32_t>({10, 11, 12}));
  // Every token of every request addresses its own block table entry at its own
  // offset. A capture whose slot_mapping disagreed with its block_table would
  // make the replay write K/V somewhere the kernel never reads.
  int64_t addressed = 0, blocks_needed = 0;
  for (int64_t r = 0; r < primary.num_reqs; ++r) {
    const int64_t keys = primary.seq_lens[static_cast<size_t>(r)];
    for (int64_t j = 0; j < keys; ++j) {
      const int64_t token = primary.query_start_loc[static_cast<size_t>(r)] + j;
      const int64_t slot = primary.slot_mapping[static_cast<size_t>(token)];
      CAPTURE(r);
      CAPTURE(j);
      CHECK(slot / primary.block_size ==
            primary.block_table[static_cast<size_t>(r * primary.block_table_cols +
                                                    j / primary.block_size)]);
      CHECK(slot % primary.block_size == j % primary.block_size);
      blocks_needed =
          std::max<int64_t>(blocks_needed,
                            primary.block_table[static_cast<size_t>(r * primary.block_table_cols +
                                                                   j / primary.block_size)] +
                                1);
      addressed += 1;
    }
  }
  CHECK(addressed == primary.tokens);
  CHECK(blocks_needed == 13);
  // The zero-cached-prefix record is what licenses deriving the KV cache from
  // `key`/`value` plus `slot_mapping` instead of capturing it.
  nlohmann::json record;
  std::ifstream(dir / (label + "-attention-0.json")) >> record;
  CHECK(record.at("zero_cached_prefix").get<bool>());
  CHECK(record.at("kv_cache").at("raw_omitted").is_string());
  std::cout << "[attn-parity] capture geometry: " << addressed << " tokens, " << blocks_needed
            << " cache blocks, block_size " << primary.block_size << std::endl;
}

TEST_CASE("ROCm paged attention replays the primary's captured attention boundary" *
          doctest::skip(AttnInstrumentAbsent())) {
  REQUIRE_MESSAGE(
      !AttnInstrumentAbsent(),
      "the attention-boundary instrument needs VT_ROCM_MOE_FIXTURE, "
      "VT_ATTN_PARITY_PRIMARY and VT_ATTN_DUMP=1");
  const char* fixture = std::getenv("VT_ROCM_MOE_FIXTURE");
  const char* results_env = std::getenv("VT_ATTN_PARITY_PRIMARY");
  const std::filesystem::path results(results_env);
  const char* preamble_env = std::getenv("VT_ATTN_PARITY_PREAMBLE");
  const std::filesystem::path out_dir =
      std::getenv("VT_ATTN_PARITY_OUT") != nullptr ? std::filesystem::path(std::getenv("VT_ATTN_PARITY_OUT"))
                                                   : std::filesystem::path();
  const std::string label = std::getenv("VT_ATTN_PARITY_LABEL") != nullptr
                                ? std::getenv("VT_ATTN_PARITY_LABEL")
                                : "L33-C2-R0";

  InstallAttnObservers();
  const PrimaryBoundary primary = LoadPrimaryBoundary(results, label, 0);
  nlohmann::json report;
  report["label"] = label;
  report["layer"] = 0;
  report["primary_dir"] = results.string();

  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard queue(backend);
  auto& q = queue.queue;

  // ---- (i) the native boundary, from the unchanged production path --------
  g_record = true;
  Generate(fixture, 33, 2, 1);
  g_record = false;
  REQUIRE(g_attn.have);
  // The preamble boundary comes from whichever realization executed: the fused
  // op when the backend registers one, the Tier-0 composite otherwise. Both
  // describe the same three tensors, so the rows below read the same thing on
  // either tree.
  if (!g_preamble.have) REQUIRE(g_rope.have);
  const bool preamble_fused = g_preamble.have;
  // The RAW qkv slice the preamble starts from, and the post-RoPE K it ends at.
  // Both realizations norm in place: the fused op reads q3/k3 on entry, and the
  // composite's RmsNorm records its input before it writes. The check in the
  // composite arm pins that its normed output is the RoPE call's input, so the
  // capture is the boundary it claims to be.
  std::vector<uint16_t> raw_q, raw_k;
  if (preamble_fused) {
    raw_q = g_preamble.q_in;
    raw_k = g_preamble.k_in;
  } else {
    REQUIRE(g_rope.rms_calls_before >= 2);
    const RmsNormCall& q_norm_call =
        g_rms_calls[static_cast<size_t>(g_rope.rms_calls_before - 2)];
    const RmsNormCall& k_norm_call =
        g_rms_calls[static_cast<size_t>(g_rope.rms_calls_before - 1)];
    CHECK(q_norm_call.shape1 == primary.dh);
    CHECK(k_norm_call.shape1 == primary.dh);
    const WordDiff normed_vs_pre_rope = DiffWords(q_norm_call.out, g_rope.q_pre, primary.dh);
    CHECK(normed_vs_pre_rope.different == 0);
    raw_q = q_norm_call.in;
    raw_k = k_norm_call.in;
  }
  const std::vector<uint16_t>& post_rope_k = preamble_fused ? g_preamble.k_post : g_rope.k_post;
  REQUIRE(raw_q.size() == g_attn.query.size());
  REQUIRE(raw_k.size() == g_attn.query.size());
  REQUIRE(post_rope_k.size() == g_attn.query.size());
  REQUIRE(g_attn.tokens == primary.tokens);
  CHECK(g_attn.hq == primary.hq);
  CHECK(g_attn.hkv == primary.hkv);
  CHECK(g_attn.dh == primary.dh);
  CHECK(g_attn.block_table_cols >=
        (primary.seq_lens[0] + g_attn.block_size - 1) / g_attn.block_size);
  CHECK(g_attn.causal);
  CHECK(std::fabs(g_attn.scale - primary.scale) < 1e-9f);
  // The metadata the replay reuses must be model-independent, or the replay
  // would compare two different batches.
  CHECK(g_attn.query_start_loc == primary.query_start_loc);
  CHECK(g_attn.seq_lens == primary.seq_lens);
  CHECK(g_attn.calls == 2);  // one PagedAttention per full-attention layer
  std::cout << "[attn-parity] native dispatch: " << g_attn.tokens << " tokens, hq "
            << g_attn.hq << ", hkv " << g_attn.hkv << ", dh " << g_attn.dh
            << ", block_size " << g_attn.block_size << ", blocks " << g_attn.blocks
            << ", scale " << g_attn.scale << std::endl;
  std::cout << "[attn-parity] native preamble: fused op " << (preamble_fused ? "yes" : "no")
            << "; rmsnorm calls before rope " << g_rope.rms_calls_before << " of "
            << g_rms_calls.size() << "; rope_neox calls " << g_rope_neox_calls << "; rot "
            << (preamble_fused ? g_preamble.rot : g_rope.rot) << std::endl;

  const std::vector<double> max_prob = MaxProbPerRow(primary);

  // Row 0: the recorded comparison, reproduced at this boundary.
  const WordDiff run_out = DiffWords(g_attn.out, primary.output, primary.dh);
  PrintDiff("A0 native run output vs primary output", run_out);

  // Row B: the preamble term on the two runs' own inputs.
  const WordDiff run_q = DiffWords(g_attn.query, primary.query, primary.dh);
  const WordDiff run_k = DiffWords(post_rope_k, primary.key, primary.dh);
  PrintDiff("B  native run Q vs primary Q", run_q);
  PrintDiff("B  native run K (post-RoPE) vs primary K", run_k);

  // Row F is compared against the raw slice in the preamble block below.

  // The native cache rows the kernel reads must be the post-RoPE K the preamble
  // produced, or the capture is not the boundary it claims to be.
  std::vector<uint16_t> cache_k;
  for (int64_t r = 0; r < primary.num_reqs; ++r) {
    const int64_t keys = primary.seq_lens[static_cast<size_t>(r)];
    const std::vector<uint16_t> rows =
        GatherKv(g_attn.k_cache, g_attn.block_size, g_attn.hkv, g_attn.dh, g_attn.block_table,
                 g_attn.block_table_cols, r, keys);
    cache_k.insert(cache_k.end(), rows.begin(), rows.end());
  }
  const WordDiff cache_identity = DiffWords(cache_k, post_rope_k, primary.dh);
  std::cout << "[attn-parity] native cache K rows vs post-RoPE k3: " << cache_identity.different
            << " / " << cache_identity.words << " differing" << std::endl;

  // ---- (ii) the native kernel on the PRIMARY's captured Q/K/V -------------
  const int64_t bs = primary.block_size;
  const int64_t hkv = primary.hkv, dh = primary.dh;
  int64_t blocks = 0;
  for (int64_t r = 0; r < primary.num_reqs; ++r)
    for (int64_t c = 0; c < (primary.seq_lens[static_cast<size_t>(r)] + bs - 1) / bs; ++c)
      blocks = std::max<int64_t>(
          blocks, primary.block_table[static_cast<size_t>(r * primary.block_table_cols + c)] + 1);
  const size_t tile_words = static_cast<size_t>(bs * hkv * dh);
  const size_t slice_words = static_cast<size_t>(blocks) * tile_words;
  std::vector<uint16_t> cache(2 * slice_words, 0);
  int64_t scattered = 0;
  auto scatter = [&](int which, const std::vector<uint16_t>& rows) {
    for (int64_t r = 0; r < primary.num_reqs; ++r) {
      for (int64_t j = 0; j < primary.seq_lens[static_cast<size_t>(r)]; ++j) {
        const int64_t token = primary.query_start_loc[static_cast<size_t>(r)] + j;
        ScatterCacheSlot(cache, blocks, bs, hkv, dh, which,
                         primary.slot_mapping[static_cast<size_t>(token)],
                         rows.data() + static_cast<size_t>(token * hkv * dh));
        if (which == 0) scattered += 1;
      }
    }
  };

  Buffer cache_buffer(backend, cache.size() * sizeof(uint16_t));
  Buffer query_buffer(backend, primary.query.size() * sizeof(uint16_t));
  Buffer out_buffer(backend, primary.query.size() * sizeof(uint16_t));
  Buffer qsl_buffer(backend, primary.query_start_loc.size() * sizeof(int32_t));
  Buffer seq_buffer(backend, primary.seq_lens.size() * sizeof(int32_t));
  Buffer table_buffer(backend, primary.block_table.size() * sizeof(int32_t));
  backend.Copy(q, qsl_buffer.data, primary.query_start_loc.data(),
               primary.query_start_loc.size() * sizeof(int32_t));
  backend.Copy(q, seq_buffer.data, primary.seq_lens.data(),
               primary.seq_lens.size() * sizeof(int32_t));
  backend.Copy(q, table_buffer.data, primary.block_table.data(),
               primary.block_table.size() * sizeof(int32_t));
  backend.Synchronize(q);

  auto kv_slice = [&](int which) {
    // The two dim-1 slices of the (blocks, 2, block_size, hkv, dh) flash cache:
    // K starts at the buffer base, V one block-slice further in, and both carry
    // the block stride 2*block_size*hkv*dh (dense_attn_block.h:356-373).
    vt::Tensor t;
    t.data = static_cast<char*>(cache_buffer.data) +
             static_cast<size_t>(which) * tile_words * sizeof(uint16_t);
    t.dtype = vt::DType::kBF16;
    t.device = q.device;
    t.rank = 4;
    t.shape[0] = blocks;
    t.shape[1] = bs;
    t.shape[2] = hkv;
    t.shape[3] = dh;
    t.stride[0] = 2 * bs * hkv * dh;
    t.stride[1] = hkv * dh;
    t.stride[2] = dh;
    t.stride[3] = 1;
    return t;
  };
  vt::Tensor replay_query =
      Tensor(query_buffer.data, vt::DType::kBF16, q.device, {primary.tokens, primary.hq, primary.dh});
  vt::Tensor replay_out =
      Tensor(out_buffer.data, vt::DType::kBF16, q.device, {primary.tokens, primary.hq, primary.dh});
  vt::Tensor replay_qsl = Tensor(qsl_buffer.data, vt::DType::kI32, q.device, {primary.num_reqs + 1});
  vt::Tensor replay_seq = Tensor(seq_buffer.data, vt::DType::kI32, q.device, {primary.num_reqs});
  vt::Tensor replay_table = Tensor(table_buffer.data, vt::DType::kI32, q.device,
                                   {primary.num_reqs, primary.block_table_cols});
  vt::PagedAttentionArgs replay_args;
  replay_args.scale = primary.scale;
  replay_args.causal = true;
  replay_args.query_start_loc_host = primary.query_start_loc.data();
  replay_args.max_seq_len = *std::max_element(primary.seq_lens.begin(), primary.seq_lens.end());

  auto run_replay = [&](const std::vector<uint16_t>& k_rows,
                        const std::vector<uint16_t>& v_rows,
                        const std::vector<uint16_t>& query_rows) {
    std::fill(cache.begin(), cache.end(), static_cast<uint16_t>(0));
    scatter(0, k_rows);
    scatter(1, v_rows);
    backend.Copy(q, cache_buffer.data, cache.data(), cache.size() * sizeof(uint16_t));
    backend.Copy(q, query_buffer.data, query_rows.data(),
                 query_rows.size() * sizeof(uint16_t));
    backend.Synchronize(q);
    vt::PagedAttention(q, replay_out, replay_query, kv_slice(0), kv_slice(1), replay_table,
                       replay_seq, replay_qsl, replay_args);
    backend.Synchronize(q);
    return ReadDeviceBf16(q, replay_out);
  };

  // Self-consistency first. The kernel addresses the cache by logical
  // (request, key position), so the native run's own Q/K/V through the same
  // kernel at block_size 16 must reproduce the native run's own output at
  // block_size 64. A difference here means the replay's cache construction is
  // wrong, and row C below would mean nothing.
  std::vector<uint16_t> native_v;
  for (int64_t r = 0; r < primary.num_reqs; ++r) {
    const int64_t keys = primary.seq_lens[static_cast<size_t>(r)];
    const std::vector<uint16_t> rows =
        GatherKv(g_attn.v_cache, g_attn.block_size, g_attn.hkv, g_attn.dh, g_attn.block_table,
                 g_attn.block_table_cols, r, keys);
    native_v.insert(native_v.end(), rows.begin(), rows.end());
  }
  const std::vector<uint16_t> replay_self = run_replay(post_rope_k, native_v, g_attn.query);
  const WordDiff replay_self_diff = DiffWords(replay_self, g_attn.out, primary.dh);
  CHECK(replay_self_diff.different == 0);
  PrintDiff("C0 replay self-consistency: native Q/K/V at the primary's geometry vs native run out",
            replay_self_diff);

  const std::vector<uint16_t> replay = run_replay(primary.key, primary.value, primary.query);
  const WordDiff replay_diff = DiffWords(replay, primary.output, primary.dh);
  CHECK(scattered == 2 * primary.tokens);
  std::cout << "[attn-parity] replay: " << blocks << " cache blocks of " << bs
            << " at the primary's geometry, " << scattered << " K/V rows written by direct copy"
            << std::endl;
  PrintDiff("C  native PagedAttention on PRIMARY Q/K/V vs primary output", replay_diff);
  PrintDiffShape(primary, replay_diff, max_prob);

  // ---- (iii) the native preamble on the PRIMARY's captured qkv ------------
  WordDiff preamble_q, preamble_k, neox_q, neox_k, table_diff, qkv_q, qkv_k;
  WordDiff production_q, production_k;
  WordDiff handcall_q, handcall_k;
  int64_t hand_fused_identity = -1;
  bool have_preamble = false;
  bool have_production = false;
  bool have_handcall = false;
  if (preamble_env != nullptr) {
    const PrimaryPreamble preamble = LoadPrimaryPreamble(preamble_env, label, 0);
    REQUIRE(preamble.tokens == primary.tokens);
    REQUIRE(preamble.dh == primary.dh);
    REQUIRE(preamble.rot == primary.dh);
    // The two captures must describe the same rows, or the lane compares
    // different tokens.
    const WordDiff join_q =
        DiffWords(JoinRopeHalves(preamble, preamble.q_left, preamble.q_right), primary.query,
                  primary.dh);
    const WordDiff join_k =
        DiffWords(JoinRopeHalves(preamble, preamble.k_left, preamble.k_right), primary.key,
                  primary.dh);
    CHECK(join_q.different == 0);
    CHECK(join_k.different == 0);

    const vllm::HfConfig config =
        vllm::LoadHfConfig((std::filesystem::path(fixture) / "config.json").string());
    CHECK(config.rope_parameters.rope_type != "llama3");
    vt::RopeArgs rope;  // dense_attn::MakeRopeArgs (dense_attn_block.h:108-123), default rope type
    rope.base = static_cast<float>(config.rope_theta);
    rope.rotary_dim = static_cast<int>(config.rotary_dim);
    CHECK(rope.base == 10000000.0f);
    CHECK(rope.rotary_dim == static_cast<int>(primary.dh));
    const float eps = static_cast<float>(config.rms_norm_eps);

    const size_t rows = static_cast<size_t>(preamble.tokens * preamble.dh);
    std::vector<uint16_t> q2(rows), k2(rows);
    for (int64_t t = 0; t < preamble.tokens; ++t) {
      for (int64_t d = 0; d < preamble.dh; ++d) {
        q2[static_cast<size_t>(t * preamble.dh + d)] =
            preamble.qkv[static_cast<size_t>(t * preamble.qkv_width + d)];
        k2[static_cast<size_t>(t * preamble.dh + d)] =
            preamble.qkv[static_cast<size_t>(t * preamble.qkv_width + preamble.dh + d)];
      }
    }
    // Row F: the two runs' qkv projection outputs, so a preamble difference
    // cannot be blamed on the GEMM upstream of it. The residual row already
    // found this boundary exact; this confirms it on this capture pair.
    qkv_q = DiffWords(q2, raw_q, preamble.dh);
    qkv_k = DiffWords(k2, raw_k, preamble.dh);
    PrintDiff("F  primary qkv q slice vs native run pre-norm q", qkv_q);
    PrintDiff("F  primary qkv k slice vs native run pre-norm k", qkv_k);
    std::vector<int32_t> positions(static_cast<size_t>(preamble.tokens));
    std::vector<int32_t> row_index(static_cast<size_t>(preamble.tokens));
    for (int64_t t = 0; t < preamble.tokens; ++t) {
      positions[static_cast<size_t>(t)] = static_cast<int32_t>(preamble.positions[static_cast<size_t>(t)]);
      row_index[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    }
    Buffer q2_buffer(backend, q2.size() * sizeof(uint16_t));
    Buffer k2_buffer(backend, k2.size() * sizeof(uint16_t));
    Buffer qg_buffer(backend, preamble.q_gamma.size() * sizeof(uint16_t));
    Buffer kg_buffer(backend, preamble.k_gamma.size() * sizeof(uint16_t));
    Buffer pos_buffer(backend, positions.size() * sizeof(int32_t));
    Buffer idx_buffer(backend, row_index.size() * sizeof(int32_t));
    Buffer cs32_buffer(backend, rows * sizeof(float));
    Buffer cs16_buffer(backend, rows * sizeof(uint16_t));
    backend.Copy(q, q2_buffer.data, q2.data(), q2.size() * sizeof(uint16_t));
    backend.Copy(q, k2_buffer.data, k2.data(), k2.size() * sizeof(uint16_t));
    backend.Copy(q, qg_buffer.data, preamble.q_gamma.data(),
                 preamble.q_gamma.size() * sizeof(uint16_t));
    backend.Copy(q, kg_buffer.data, preamble.k_gamma.data(),
                 preamble.k_gamma.size() * sizeof(uint16_t));
    backend.Copy(q, pos_buffer.data, positions.data(), positions.size() * sizeof(int32_t));
    backend.Copy(q, idx_buffer.data, row_index.data(), row_index.size() * sizeof(int32_t));
    backend.Synchronize(q);

    vt::Tensor t_q2 = Tensor(q2_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, preamble.dh});
    vt::Tensor t_k2 = Tensor(k2_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, preamble.dh});
    vt::Tensor t_qg = Tensor(qg_buffer.data, vt::DType::kBF16, q.device, {preamble.dh});
    vt::Tensor t_kg = Tensor(kg_buffer.data, vt::DType::kBF16, q.device, {preamble.dh});
    vt::Tensor t_pos = Tensor(pos_buffer.data, vt::DType::kI32, q.device, {preamble.tokens});
    vt::Tensor t_idx = Tensor(idx_buffer.data, vt::DType::kI32, q.device, {preamble.tokens});
    vt::Tensor t_cs32 = Tensor(cs32_buffer.data, vt::DType::kF32, q.device,
                               {preamble.tokens, preamble.rot});
    vt::Tensor t_cs16 = Tensor(cs16_buffer.data, vt::DType::kBF16, q.device,
                               {preamble.tokens, preamble.rot});
    vt::RopeCosSinCache(q, t_cs32, t_pos, rope);
    vt::CastBf16(q, t_cs16, t_cs32);
    const std::vector<uint16_t> native_table = ReadDeviceBf16(q, t_cs16);
    // The native per-step table row t already encodes positions[t]; the primary's
    // cache is indexed by the real position.
    std::vector<uint16_t> expected_table(static_cast<size_t>(preamble.tokens * preamble.rot));
    for (int64_t t = 0; t < preamble.tokens; ++t)
      for (int64_t d = 0; d < preamble.rot; ++d)
        expected_table[static_cast<size_t>(t * preamble.rot + d)] =
            preamble.cos_sin[static_cast<size_t>(preamble.positions[static_cast<size_t>(t)] *
                                                     preamble.rot + d)];
    table_diff = DiffWords(native_table, expected_table, preamble.rot);
    PrintDiff("E  native cos|sin table vs primary cos_sin at positions", table_diff);

    vt::RmsNorm(q, t_q2, t_q2, t_qg, vt::RmsNormArgs{eps, false});
    vt::RmsNorm(q, t_k2, t_k2, t_kg, vt::RmsNormArgs{eps, false});
    backend.Synchronize(q);
    const std::vector<uint16_t> normed_q = ReadDeviceBf16(q, t_q2);
    const std::vector<uint16_t> normed_k = ReadDeviceBf16(q, t_k2);
    REQUIRE(normed_q.size() == rows);

    Buffer q3_buffer(backend, rows * sizeof(uint16_t));
    Buffer k3_buffer(backend, rows * sizeof(uint16_t));
    backend.Copy(q, q3_buffer.data, normed_q.data(), rows * sizeof(uint16_t));
    backend.Copy(q, k3_buffer.data, normed_k.data(), rows * sizeof(uint16_t));
    backend.Synchronize(q);
    vt::Tensor t_q3 = Tensor(q3_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, 1, preamble.dh});
    vt::Tensor t_k3 = Tensor(k3_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, 1, preamble.dh});
    vt::RopeFromCache(q, t_q3, &t_k3, t_idx, t_cs16, rope);
    backend.Synchronize(q);
    preamble_q = DiffWords(ReadDeviceBf16(q, t_q3), primary.query, primary.dh);
    preamble_k = DiffWords(ReadDeviceBf16(q, t_k3), primary.key, primary.dh);
    PrintDiff("D  native preamble (RmsNorm+RopeFromCache) on primary qkv vs primary Q", preamble_q);
    PrintDiff("D  native preamble (RmsNorm+RopeFromCache) on primary qkv vs primary K", preamble_k);

    // The same normed q/k through the non-cached native RoPE, to separate the
    // norm store from the cos|sin source.
    Buffer q4_buffer(backend, rows * sizeof(uint16_t));
    Buffer k4_buffer(backend, rows * sizeof(uint16_t));
    backend.Copy(q, q4_buffer.data, normed_q.data(), rows * sizeof(uint16_t));
    backend.Copy(q, k4_buffer.data, normed_k.data(), rows * sizeof(uint16_t));
    backend.Synchronize(q);
    vt::Tensor t_q4 = Tensor(q4_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, 1, preamble.dh});
    vt::Tensor t_k4 = Tensor(k4_buffer.data, vt::DType::kBF16, q.device,
                             {preamble.tokens, 1, preamble.dh});
    vt::RopeNeox(q, t_q4, t_k4, t_pos, rope);
    backend.Synchronize(q);
    neox_q = DiffWords(ReadDeviceBf16(q, t_q4), primary.query, primary.dh);
    neox_k = DiffWords(ReadDeviceBf16(q, t_k4), primary.key, primary.dh);
    PrintDiff("D' native preamble (RmsNorm+RopeNeox) on primary qkv vs primary Q", neox_q);
    PrintDiff("D' native preamble (RmsNorm+RopeNeox) on primary qkv vs primary K", neox_k);
    have_preamble = true;

    // D2: the PRODUCTION preamble on the primary's own qkv. D above measures the
    // Tier-0 composite, which keeps the pre-RoPE BF16 store on every backend that
    // registers no fused op; D2 measures the path this repair changes. The
    // binding is the fused branch's own BF16 binding (dense_attn_block.h:594-613):
    // bf16 norm weights, the bf16 per-step cos|sin table, the identity row index.
    if (vt::OpRegistered(vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM)) {
      auto fused_preamble_op = reinterpret_cast<vt::AttnQkNormRopeFn>(
          vt::GetOp(vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM));
      REQUIRE(fused_preamble_op != nullptr);
      Buffer q5_buffer(backend, rows * sizeof(uint16_t));
      Buffer k5_buffer(backend, rows * sizeof(uint16_t));
      backend.Copy(q, q5_buffer.data, q2.data(), rows * sizeof(uint16_t));
      backend.Copy(q, k5_buffer.data, k2.data(), rows * sizeof(uint16_t));
      backend.Synchronize(q);
      vt::Tensor t_q5 = Tensor(q5_buffer.data, vt::DType::kBF16, q.device,
                               {preamble.tokens, 1, preamble.dh});
      vt::Tensor t_k5 = Tensor(k5_buffer.data, vt::DType::kBF16, q.device,
                               {preamble.tokens, 1, preamble.dh});
      fused_preamble_op(q, t_q5, t_k5, t_qg, t_kg, t_cs16, t_idx,
                        vt::RmsNormArgs{eps, false}, rope);
      backend.Synchronize(q);
      production_q = DiffWords(ReadDeviceBf16(q, t_q5), primary.query, primary.dh);
      production_k = DiffWords(ReadDeviceBf16(q, t_k5), primary.key, primary.dh);
      PrintDiff("D2 production preamble (fused qk-norm-rope) on primary qkv vs primary Q",
                production_q);
      PrintDiff("D2 production preamble (fused qk-norm-rope) on primary qkv vs primary K",
                production_k);
      have_production = true;
      if (!out_dir.empty()) {
        WriteWords(out_dir / (label + "-native-production-preamble-q.bin"),
                   ReadDeviceBf16(q, t_q5));
      }

      // D3: the HAND-CALL realization on the same primary qkv, through the
      // op-layer entry the ADOPT=0 branch calls (dense_attn_block.h:645-649).
      // Before the stride repair this threw
      // "attn_qk_norm_rope: row stride must be the inner dimension" for every
      // Dh > 1, which made the documented same-binary A/B lever unusable; the two
      // realizations must also agree byte-for-byte, which is the recipe's
      // composite contract and what makes the lever an A/B rather than two arms.
      Buffer q6_buffer(backend, rows * sizeof(uint16_t));
      Buffer k6_buffer(backend, rows * sizeof(uint16_t));
      backend.Copy(q, q6_buffer.data, q2.data(), rows * sizeof(uint16_t));
      backend.Copy(q, k6_buffer.data, k2.data(), rows * sizeof(uint16_t));
      backend.Synchronize(q);
      vt::Tensor t_q6 = Tensor(q6_buffer.data, vt::DType::kBF16, q.device,
                               {preamble.tokens, 1, preamble.dh});
      vt::Tensor t_k6 = Tensor(k6_buffer.data, vt::DType::kBF16, q.device,
                               {preamble.tokens, 1, preamble.dh});
      vt::AttnQkNormRope(q, t_q6, t_k6, t_qg, t_kg, t_cs16, t_idx,
                         vt::RmsNormArgs{eps, false}, rope);
      backend.Synchronize(q);
      const std::vector<uint16_t> hand_q = ReadDeviceBf16(q, t_q6);
      const std::vector<uint16_t> hand_k = ReadDeviceBf16(q, t_k6);
      handcall_q = DiffWords(hand_q, primary.query, primary.dh);
      handcall_k = DiffWords(hand_k, primary.key, primary.dh);
      PrintDiff("D3 hand-call preamble (vt::AttnQkNormRope) on primary qkv vs primary Q",
                handcall_q);
      PrintDiff("D3 hand-call preamble (vt::AttnQkNormRope) on primary qkv vs primary K",
                handcall_k);
      const WordDiff hand_vs_fused_q = DiffWords(hand_q, ReadDeviceBf16(q, t_q5), primary.dh);
      const WordDiff hand_vs_fused_k = DiffWords(hand_k, ReadDeviceBf16(q, t_k5), primary.dh);
      PrintDiff("D3 hand-call vs D2 fused-op, same primary qkv, Q", hand_vs_fused_q);
      PrintDiff("D3 hand-call vs D2 fused-op, same primary qkv, K", hand_vs_fused_k);
      hand_fused_identity = hand_vs_fused_q.different + hand_vs_fused_k.different;
      CHECK(hand_fused_identity == 0);
      have_handcall = true;
    }

    if (!out_dir.empty()) {
      WriteWords(out_dir / (label + "-native-preamble-rope-from-cache-q.bin"),
                 ReadDeviceBf16(q, t_q3));
      WriteWords(out_dir / (label + "-native-preamble-rope-neox-q.bin"), ReadDeviceBf16(q, t_q4));
      WriteWords(out_dir / (label + "-native-cos-sin-table.bin"), native_table);
    }
  } else {
    std::cout << "[attn-parity] preamble lanes SKIPPED: VT_ATTN_PARITY_PREAMBLE unset"
              << std::endl;
  }

  // ---- focused cases for the #3115 repair ---------------------------------
  // KERNEL. Under the primary's arithmetic (the softmax probability narrowed to
  // the value dtype before the value accumulate, prefix_prefill.py:471) the
  // native kernel on the primary's OWN Q/K/V must reproduce the primary's
  // output byte-for-byte. Red at 2918 of 8448 before that repair.
  std::cout << "[attn-parity][focused-kernel] C " << replay_diff.different << " / "
            << replay_diff.words << " differing words (want 0)" << std::endl;
  CHECK(replay_diff.different == 0);

  // PREAMBLE. Under the primary's boundary (f32 carrier through RoPE, narrowed
  // once at the store) the native run and the primary must agree on Q and K up
  // to the CPU model's own 1-word residue against the primary's capture. Red at
  // 1569 Q / 1542 K before that repair.
  std::cout << "[attn-parity][focused-preamble] B " << run_q.different << " Q / "
            << run_k.different << " K differing words (want <= 1)" << std::endl;
  CHECK(run_q.different <= 1);
  CHECK(run_k.different <= 1);
  if (have_production) {
    std::cout << "[attn-parity][focused-preamble-op] D2 " << production_q.different << " Q / "
              << production_k.different << " K differing words (want <= 1)" << std::endl;
    CHECK(production_q.different <= 1);
    CHECK(production_k.different <= 1);
  }
  if (have_handcall) {
    std::cout << "[attn-parity][focused-handcall] D3 " << handcall_q.different << " Q / "
              << handcall_k.different << " K differing words (want <= 1); hand-call vs fused op "
              << hand_fused_identity << " differing words (want 0)" << std::endl;
    CHECK(handcall_q.different <= 1);
    CHECK(handcall_k.different <= 1);
    CHECK(hand_fused_identity == 0);
  }
  // BOTH SETTINGS OF THE ADOPTION SWITCH MUST REACH THE REGISTERED OP. On ROCm the
  // bf16 production preamble is the fused realization either way:
  // VT_FUSED_CHAIN_ADOPT=1 enters it through vt::FusedChain's fast_op dispatch,
  // =0 through the hand-call at dense_attn_block.h:645-649, which is the same
  // vt::AttnQkNormRope entry. `preamble_fused` records that the OP ran, not which
  // branch called it, so the branch is pinned by the second check: the Tier-0
  // composite must NOT have run, or the =0 run would be measuring the standalone
  // RmsNorm+RopeFromCache sequence instead of the hand-call this finding names.
  // The device A/B -- same binary, same workload, one variable -- is the other half.
  const bool adopt = vllm::dense_attn::FusedChainAdoptEnabled();
  std::cout << "[attn-parity][focused-adopt] VT_FUSED_CHAIN_ADOPT=" << (adopt ? 1 : 0)
            << "; registered fused op ran: " << (preamble_fused ? "yes" : "no")
            << "; composite fallback ran: " << (g_rope.have ? "yes" : "no") << std::endl;
  CHECK(preamble_fused);
  CHECK_FALSE(g_rope.have);

  // ---- the verdict --------------------------------------------------------
  // #3115's two hypotheses: the Q/K preamble, the kernel arithmetic, or both.
  std::string verdict;
  if (replay_diff.different == 0) {
    verdict = "PREAMBLE-ONLY";
  } else if (run_q.different == 0 && run_k.different == 0) {
    verdict = "KERNEL-ONLY";
  } else {
    verdict = "BOTH";
  }
  std::cout << "[attn-parity] VERDICT " << verdict << ": native-on-primary-inputs "
            << replay_diff.different << "/" << replay_diff.words << "; preamble (native run Q/K) "
            << run_q.different << "/" << run_q.words << " and " << run_k.different << "/"
            << run_k.words << std::endl;
  if (replay_diff.different == 0)
    std::cout << "[attn-parity] repair site: dense_attn_block.h:637-653 (do not touch the kernel)"
              << std::endl;
  else
    std::cout << "[attn-parity] repair site: rocm_paged_attn.hip:527-531 (probability dtype), "
                 ":196-198 (FastExp), :512 (warp reduction order)"
              << std::endl;

  report["native_dispatch"] = {{"tokens", g_attn.tokens},
                               {"hq", g_attn.hq},
                               {"hkv", g_attn.hkv},
                               {"dh", g_attn.dh},
                               {"block_size", g_attn.block_size},
                               {"blocks", g_attn.blocks},
                               {"scale", g_attn.scale}};
  report["A0_native_run_output_vs_primary_output"] = run_out.different;
  report["B_native_run_q_vs_primary_q"] = run_q.different;
  report["B_native_run_k_vs_primary_k"] = run_k.different;
  report["C_native_kernel_on_primary_inputs_vs_primary_output"] = replay_diff.different;
  report["C_words"] = replay_diff.words;
  report["C_per_row"] = replay_diff.row_different;
  report["C_distance_1"] = replay_diff.distance[1];
  report["C_distance_2"] = replay_diff.distance[2];
  report["C_distance_3"] = replay_diff.distance[3];
  report["C_distance_ge4"] = replay_diff.distance[4];
  report["C0_replay_self_consistency"] = replay_self_diff.different;
  report["cache_k_identity_different"] = cache_identity.different;
  report["preamble_fused"] = preamble_fused;
  if (have_preamble) {
    report["D_preamble_rope_cache_q"] = preamble_q.different;
    report["D_preamble_rope_cache_k"] = preamble_k.different;
    report["D_preamble_rope_neox_q"] = neox_q.different;
    report["D_preamble_rope_neox_k"] = neox_k.different;
    report["E_cos_sin_table"] = table_diff.different;
    report["E_cos_sin_words"] = table_diff.words;
    report["F_qkv_q"] = qkv_q.different;
    report["F_qkv_k"] = qkv_k.different;
  }
  report["D2_production_preamble_measured"] = have_production;
  if (have_production) {
    report["D2_production_preamble_q"] = production_q.different;
    report["D2_production_preamble_k"] = production_k.different;
  }
  report["D3_handcall_preamble_measured"] = have_handcall;
  report["fused_chain_adopt"] = vllm::dense_attn::FusedChainAdoptEnabled();
  if (have_handcall) {
    report["D3_handcall_preamble_q"] = handcall_q.different;
    report["D3_handcall_preamble_k"] = handcall_k.different;
    report["D3_handcall_vs_fused_differing_words"] = hand_fused_identity;
  }
  report["verdict"] = verdict;
  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir);
    std::ofstream(out_dir / (label + "-attn-parity-report.json")) << report.dump(2) << '\n';
    WriteWords(out_dir / (label + "-native-attention-0-query.bin"), g_attn.query);
    WriteWords(out_dir / (label + "-native-attention-0-output.bin"), g_attn.out);
    WriteWords(out_dir / (label + "-native-attention-0-k-cache.bin"), g_attn.k_cache);
    WriteWords(out_dir / (label + "-native-attention-0-v-cache.bin"), g_attn.v_cache);
    WriteWords(out_dir / (label + "-native-attention-0-k3.bin"), post_rope_k);
    WriteWords(out_dir / (label + "-replay-output.bin"), replay);
  }
}

// ---------------------------------------------------------------------------
// The key-walk geometry the primary selects, one case per arm.
//
// The recorded capture measures ONE arm: prefix_prefill over a request with no
// cached prefix, where BLOCK_N = 64 and the 33 keys fit one tile. The same native
// kernel also serves pure decode (the Triton decode arm tiles by
// min(block_size, 128)) and a chunked prefill (a 32-key context tile walk before
// the 64-key chunk walk), and it served both of those with the prefill tile
// before this repair. These cases synthesize the data that separates the arms
// and compare the device against the primary's own key walk for the arm.
// ---------------------------------------------------------------------------
namespace {

void CheckArmGeometry(const char* label, const SyntheticArm& arm, bool chunked,
                      const std::vector<int64_t>& wrong_widths) {
  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard queue(backend);
  auto& q = queue.queue;
  const std::vector<uint16_t> device = RunSyntheticArm(backend, q, arm);
  const WordDiff correct = DiffWords(device, ArmModelWords(arm, chunked, 0), arm.dh);
  std::cout << "[arm-geometry] " << label << ": " << arm.num_reqs << " requests, query_len "
            << arm.query_len << ", seq_len " << arm.seq_len << ", context "
            << (arm.seq_len - arm.query_len) << ", block_size " << arm.block_size << ", high key "
            << arm.high_key << ", " << arm.total_q << " query rows" << std::endl;
  PrintArmDiff("device vs the primary arm's own tiles", correct, arm.total_q);
  CHECK_MESSAGE(correct.different <= kArmNoiseWords,
                label << ": the device must follow the primary arm's key walk ("
                      << correct.different << " differing words, noise floor " << kArmNoiseWords
                      << ")");
  for (int64_t width : wrong_widths) {
    const WordDiff d = DiffWords(device, ArmModelWords(arm, chunked, width), arm.dh);
    PrintArmDiff("device vs a uniform tiling of that width", d, arm.total_q);
    CHECK_MESSAGE(d.different >= kArmSignalWords,
                  label << ": a uniform " << width
                        << "-key tiling must not reproduce the device (" << d.different
                        << " differing words, signal floor " << kArmSignalWords << ")");
  }
}

}  // namespace

// THE PREFILL ARM: max_query_len > 1, no cached prefix -> prefix_prefill with
// BLOCK_N = 64 (prefix_prefill.py:955-966). 65 query tokens so that 33 rows per
// request have a key range that reaches past key 32, and the high key sits AT 32:
// one key beyond the 32-boundary, so a 32-key tile and a 64-key tile take their
// reference max over different key sets and narrow the other 32 keys against
// different scales.
TEST_CASE("ROCm paged attention uses the primary's 64-key prefill tile" *
          doctest::skip(RocmBackendAbsent())) {
  REQUIRE_MESSAGE(!RocmBackendAbsent(),
                  "this case measures the ROCm arm's key tiling and needs the ROCm backend "
                  "registered");
  CheckArmGeometry("prefill, empty context", BuildSyntheticArm(4, 65, 65, 16, 32), true, {32, 16});
}

// THE CHUNKED-CONTEXT ARM: max_query_len > 1 with a cached prefix -> prefix_prefill
// walks the context in TRITON_BLOCK_SIZE = 32 key tiles from key 0 and then the
// chunk in BLOCK_N = 64 tiles anchored at the chunk start (:965, :1007, :231-343,
// :369). 37 cached keys and 3 query tokens; the high key is at 33, inside the
// second context tile, so a 64-key context tile would put it in the same tile as
// keys 0..31 and narrow those at the high scale instead of at 1.
TEST_CASE("ROCm paged attention uses the primary's 32-key context tile" *
          doctest::skip(RocmBackendAbsent())) {
  REQUIRE_MESSAGE(!RocmBackendAbsent(),
                  "this case measures the ROCm arm's key tiling and needs the ROCm backend "
                  "registered");
  CheckArmGeometry("chunked prefill, 37-key context", BuildSyntheticArm(4, 3, 40, 16, 33), true,
                   {64});
}

// THE PURE-DECODE ARM: max_query_len == 1 -> the Triton decode kernel tiles the
// key range by TRITON_BLOCK_SIZE = min(block_size, 128) = 16 for this cache
// (chunked_prefill_paged_decode.py:444-445, :147-149, :244). A 16-wide tile puts
// key 17 in its own tile; a 32- or 64-wide one puts it with keys 0..15, whose
// probability is then narrowed at the high scale.
TEST_CASE("ROCm paged attention uses the primary's decode tile" *
          doctest::skip(RocmBackendAbsent())) {
  REQUIRE_MESSAGE(!RocmBackendAbsent(),
                  "this case measures the ROCm arm's key tiling and needs the ROCm backend "
                  "registered");
  CheckArmGeometry("pure decode, block_size 16", BuildSyntheticArm(8, 1, 40, 16, 17), false,
                   {32, 64});
}

// ---------------------------------------------------------------------------
// The hand-call realization of the qk-norm-RoPE preamble, on synthesized data.
//
// The production bf16 preamble has two realizations that must agree: the recipe's
// fast op through vt::FusedChain (VT_FUSED_CHAIN_ADOPT=1) and the hand-call
// vt::AttnQkNormRope the fallback branch reaches under VT_FUSED_CHAIN_ADOPT=0
// (dense_attn_block.h:645-649). Before the op-layer stride repair the second one
// threw for every Dh > 1, so the documented same-binary A/B lever could not run
// at all on this workload; this case runs both in one process, on identical
// inputs, and requires byte identity.
// ---------------------------------------------------------------------------
TEST_CASE("ROCm bf16 qk-norm-rope: the hand-call realization matches the fused recipe" *
          doctest::skip(RocmBackendAbsent())) {
  REQUIRE_MESSAGE(!RocmBackendAbsent(),
                  "this case measures the ROCm arm's qk-norm-RoPE hand-call realization and "
                  "needs the ROCm backend registered");
  REQUIRE_MESSAGE(vt::OpRegistered(vt::OpId::kAttnQkNormRope, vt::DeviceType::kROCM),
                  "this case measures the Recipe-vs-hand-call pair on a backend that registers "
                  "the recipe's fast realization");
  auto& backend = vt::GetBackend(vt::DeviceType::kROCM);
  QueueGuard queue(backend);
  auto& q = queue.queue;
  const int64_t tokens = 8, dh = 128, rot = 128;
  const size_t rows = static_cast<size_t>(tokens * dh);
  std::vector<uint16_t> q_words(rows), k_words(rows);
  std::vector<uint16_t> q_gamma(static_cast<size_t>(dh)), k_gamma(static_cast<size_t>(dh));
  for (size_t i = 0; i < rows; ++i) {
    q_words[i] = static_cast<uint16_t>(0x3C00 + ((i * 37 + 11) % 1024));   // [1, 8)
    k_words[i] = static_cast<uint16_t>(0x3C00 + ((i * 53 + 29) % 1024));
  }
  // Distinct q/k weights, so a realization that swapped them could not agree.
  for (size_t i = 0; i < q_gamma.size(); ++i) {
    q_gamma[i] = static_cast<uint16_t>(0x3F00 + (i % 64));         // [0.5, 1)
    k_gamma[i] = static_cast<uint16_t>(0x3E80 + ((i * 3) % 96));   // [0.25, 0.625)
  }
  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  Buffer qb(backend, rows * sizeof(uint16_t)), kb(backend, rows * sizeof(uint16_t));
  Buffer qc(backend, rows * sizeof(uint16_t)), kc(backend, rows * sizeof(uint16_t));
  Buffer qgb(backend, q_gamma.size() * sizeof(uint16_t));
  Buffer kgb(backend, k_gamma.size() * sizeof(uint16_t));
  Buffer pb(backend, positions.size() * sizeof(int32_t));
  Buffer cs32(backend, rows * sizeof(float)), cs16(backend, rows * sizeof(uint16_t));
  backend.Copy(q, qb.data, q_words.data(), rows * sizeof(uint16_t));
  backend.Copy(q, qc.data, q_words.data(), rows * sizeof(uint16_t));
  backend.Copy(q, kb.data, k_words.data(), rows * sizeof(uint16_t));
  backend.Copy(q, kc.data, k_words.data(), rows * sizeof(uint16_t));
  backend.Copy(q, qgb.data, q_gamma.data(), q_gamma.size() * sizeof(uint16_t));
  backend.Copy(q, kgb.data, k_gamma.data(), k_gamma.size() * sizeof(uint16_t));
  backend.Copy(q, pb.data, positions.data(), positions.size() * sizeof(int32_t));
  backend.Synchronize(q);

  vt::RopeArgs rope;
  rope.base = 1000000.0f;
  rope.rotary_dim = static_cast<int>(rot);
  const vt::RmsNormArgs norm{1e-6f, false};
  vt::Tensor t_cs32 = Tensor(cs32.data, vt::DType::kF32, q.device, {tokens, rot});
  vt::Tensor t_cs16 = Tensor(cs16.data, vt::DType::kBF16, q.device, {tokens, rot});
  vt::Tensor t_pos = Tensor(pb.data, vt::DType::kI32, q.device, {tokens});
  vt::RopeCosSinCache(q, t_cs32, t_pos, rope);
  vt::CastBf16(q, t_cs16, t_cs32);
  backend.Synchronize(q);

  // Arm 1 (VT_FUSED_CHAIN_ADOPT=1): the recipe's fast realization, with exactly the
  // binding the model builds (dense_attn_block.h:594-613): the 2-D norm view and
  // the 3-D rope view alias one buffer, bf16 weights, the bf16 cache, identity rows.
  vt::Tensor f_q2 = Tensor(qb.data, vt::DType::kBF16, q.device, {tokens, dh});
  vt::Tensor f_k2 = Tensor(kb.data, vt::DType::kBF16, q.device, {tokens, dh});
  vt::Tensor f_q3 = Tensor(qb.data, vt::DType::kBF16, q.device, {tokens, 1, dh});
  vt::Tensor f_k3 = Tensor(kb.data, vt::DType::kBF16, q.device, {tokens, 1, dh});
  vt::Tensor t_qg = Tensor(qgb.data, vt::DType::kBF16, q.device, {dh});
  vt::Tensor t_kg = Tensor(kgb.data, vt::DType::kBF16, q.device, {dh});
  vt::FusedBinding binding;
  binding.op[0] = &f_q2;
  binding.op[1] = &t_qg;
  binding.op[2] = &f_k2;
  binding.op[3] = &t_kg;
  binding.op[4] = &f_q3;
  binding.op[5] = &f_k3;
  binding.op[6] = &t_cs16;
  binding.op[7] = &t_pos;
  binding.n = 8;
  vt::FusedParams params;
  params.eps = norm.eps;
  params.rope = rope;
  vt::FusedChain(q, vt::kAttnQkNormRope, binding, params);
  backend.Synchronize(q);

  // Arm 2 (VT_FUSED_CHAIN_ADOPT=0): the hand-call through the op layer, the same
  // entry the model's fallback branch reaches.
  vt::Tensor h_q3 = Tensor(qc.data, vt::DType::kBF16, q.device, {tokens, 1, dh});
  vt::Tensor h_k3 = Tensor(kc.data, vt::DType::kBF16, q.device, {tokens, 1, dh});
  vt::AttnQkNormRope(q, h_q3, h_k3, t_qg, t_kg, t_cs16, t_pos, norm, rope);
  backend.Synchronize(q);

  const std::vector<uint16_t> fused_q = ReadDeviceBf16(q, f_q3);
  const std::vector<uint16_t> fused_k = ReadDeviceBf16(q, f_k3);
  const WordDiff dq = DiffWords(ReadDeviceBf16(q, h_q3), fused_q, dh);
  const WordDiff dk = DiffWords(ReadDeviceBf16(q, h_k3), fused_k, dh);
  PrintDiff("hand-call vs fused recipe, Q", dq);
  PrintDiff("hand-call vs fused recipe, K", dk);
  const bool any_rotation = [&] {
    for (int64_t t = 0; t < tokens; ++t)
      for (int64_t d = 0; d < dh; ++d)
        if (fused_q[static_cast<size_t>(t * dh + d)] != q_words[static_cast<size_t>(t * dh + d)])
          return true;
    return false;
  }();
  CHECK(any_rotation);  // a no-op preamble would make the identity vacuous
  CHECK(dq.different == 0);
  CHECK(dk.different == 0);
}
