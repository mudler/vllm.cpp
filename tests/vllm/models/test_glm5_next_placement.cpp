// #3019: the production loader's expert placement must survive the forward.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "support/glm5_next_forward_fixture.h"
#include "support/test_env.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vllm/platforms/interface.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {
using namespace glm5_next_fixture;
using glm5_next_forward_fixture::Step;
using glm5_next_forward_fixture::Topology;
using vt::DeviceType;

// The fake substitutes only allocation and device kernels. Load, routing,
// bridging, cache composition and forward dispatch remain production code.
class ExpertBackend final : public vt::Backend {
 public:
  std::map<const void*, int> banks;
  std::map<int, int> uploads;
  std::map<const void*, int> residents;
  std::map<int, int> gate_calls;
  std::map<int, int> down_calls;
  bool room = true;
  mutable int fit_queries = 0;

  void* Alloc(size_t n) override {
    return vt::GetBackend(DeviceType::kCPU).Alloc(n);
  }
  void Free(void* p) override { vt::GetBackend(DeviceType::kCPU).Free(p); }
  void Memset(vt::Queue&, void* p, int v, size_t n) override { std::memset(p, v, n); }
  void Copy(vt::Queue&, void* dst, const void* src, size_t n) override {
    if (const auto it = banks.find(src); it != banks.end()) {
      ++uploads[it->second];
      residents[dst] = it->second;
    }
    std::memcpy(dst, src, n);
  }
  vt::Queue CreateQueue() override { return {{DeviceType::kROCM, 0}, nullptr}; }
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return true; }
  bool DeviceMemoryInfo(size_t* free, size_t* total) const override {
    ++fit_queries;
    *free = room ? size_t{8} << 30 : 0;
    *total = size_t{8} << 30;
    return true;
  }
  void Reset() {
    banks.clear(); uploads.clear(); residents.clear();
    gate_calls.clear(); down_calls.clear(); fit_queries = 0; room = true;
  }
};

ExpertBackend& Backend() {
  // Pools can outlive tests and retain their backend reference.
  static auto* b = new ExpertBackend;
  return *b;
}

class ExpertPlatform final : public vllm::platforms::Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kROCM; }
  vt::Backend& backend() const override { return Backend(); }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {}; }
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kF32, vt::DType::kBF16};
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
};

vt::Tensor Host(vt::Tensor t) {
  t.device = {DeviceType::kCPU, 0};
  return t;
}

void Gate(vt::Queue&, vt::Tensor& out, const vt::Tensor& act,
          const vt::Tensor& gate, const vt::Tensor& up,
          const vt::Tensor& ids, float limit) {
  auto& b = Backend();
  REQUIRE(b.residents.count(gate.data) == 1);
  REQUIRE(b.residents.count(up.data) == 1);
  ++b.gate_calls[b.residents.at(gate.data)];
  vt::Queue q{{DeviceType::kCPU, 0}, nullptr};
  auto h = Host(out);
  vt::MoeGateUpSwiGLUGrouped(q, h, Host(act), Host(gate), Host(up), Host(ids), limit);
}

void Down(vt::Queue&, vt::Tensor& out, const vt::Tensor& act,
          const vt::Tensor& weight, const vt::Tensor& ids) {
  auto& b = Backend();
  REQUIRE(b.residents.count(weight.data) == 1);
  ++b.down_calls[b.residents.at(weight.data)];
  vt::Queue q{{DeviceType::kCPU, 0}, nullptr};
  auto h = Host(out);
  vt::MatmulBTQuantGrouped(q, h, Host(act), Host(weight), Host(ids));
}

struct SavedEnvironment {
  const char* key;
  std::string previous;
  bool present;
  SavedEnvironment(const char* k, const char* value) : key(k) {
    const char* old = std::getenv(key);
    present = old != nullptr;
    if (old) previous = old;
    vllm_test::SetEnv(key, value);
  }
  ~SavedEnvironment() {
    vllm_test::SetEnv(key, present ? previous.c_str() : nullptr);
  }
};

struct Environment {
  SavedEnvironment keep_quant{"VT_GGUF_KEEP_QUANT", "1"};
  SavedEnvironment cpu_ref{"VT_CPU_REF", "0"};
  Environment() {
    Backend().Reset();
    vllm::ResetActiveMoePlacementPlanForTesting();
    static const bool registered = [] {
      static ExpertPlatform platform;
      vllm::platforms::RegisterPlatform(DeviceType::kROCM, &platform);
      vt::RegisterBackend({DeviceType::kROCM, 0}, &Backend());
      // Priority 100 under a test-only name so the fakes outrank real ROCm
      // kernels (priority 0, kNativeProviderName) on a GPU build. RegisterOp
      // would reuse kNativeProviderName and be silently ignored.
      vt::OpProvider fake;
      fake.name = "test-placement";
      fake.priority = 100;
      fake.supports = nullptr;
      fake.fn = reinterpret_cast<void*>(&Gate);
      vt::RegisterOpProvider(vt::OpId::kMoeGateUpSwiGLUGrouped,
                             DeviceType::kROCM, fake);
      fake.fn = reinterpret_cast<void*>(&Down);
      vt::RegisterOpProvider(vt::OpId::kMatmulBTQuantGrouped,
                             DeviceType::kROCM, fake);
      return true;
    }();
    (void)registered;
  }
  ~Environment() { vllm::ResetActiveMoePlacementPlanForTesting(); }
};

void Place(const std::vector<vllm::PlacementOverride>& overrides) {
  vllm::SetActiveMoePlacementPlan(vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides(overrides, DeviceType::kROCM), kLayers));
}

std::unique_ptr<vllm::LoadedModel> Load(const vllm::GgufFile& g,
                                      DeviceType target = DeviceType::kROCM) {
  const auto config = vllm::Glm5NextHfConfigFromGguf(g);
  return vllm::ModelRegistry::Load(config,
      vllm::ModelSource::FromGguf(g, target));
}

void Track(const vllm::LoadedModel& model) {
  const auto& w = vllm::ModelAs<vllm::Glm5NextLoadedModel>(
      model, "Glm5NextForConditionalGeneration").weights();
  for (int l = 0; l < kLayers; ++l) {
    if (IsDense(l)) continue;
    const auto& m = w.layers[l].moe;
    for (const auto* t : {&m.gate_exps, &m.up_exps, &m.down_exps})
      Backend().banks[t->bytes.data()] = l;
  }
}

bool Enabled() {
  const char* e = std::getenv("VT_GLM5_NEXT_DEVICE_EXPERTS");
  return e && std::string(e) == "1";
}

void CheckLogits(const vllm::ForwardLogits& got, const vllm::ForwardLogits& want) {
  REQUIRE(got.rows > 0);
  REQUIRE(got.host.size() == want.host.size());
  CHECK(got.host == want.host);
  CHECK(std::all_of(got.host.begin(), got.host.end(), [](float f) { return std::isfinite(f); }));
  CHECK(std::any_of(got.host.begin(), got.host.end(), [](float f) { return f != 0; }));
}
}  // namespace

TEST_CASE("glm5_next placement: loaded CPU experts never upload to ROCm") {
  Environment env;
  FixtureOpts opts;
  SUBCASE("Q8_0") {}
  SUBCASE("real Q4_0 banks unsupported by ROCm") { opts.q4_0_experts = true; }
  gguf_test::TempFile file(BuildFixture(opts));
  const auto g = vllm::GgufFile::Open(file.path());
  Place({{vllm::kLlmFfnExpsRegex, "cpu"}});
  auto model = Load(g);
  Track(*model);
  // Changing the global plan must not change a model already loaded.
  vllm::ResetActiveMoePlacementPlanForTesting();
  Step device({3, 11, 7});
  device.queue = Backend().CreateQueue();
  if (!Enabled()) {
    CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, device.Get()),
                         doctest::Contains("OPT-IN"), std::runtime_error);
    CHECK(Backend().uploads.empty());
    return;
  }
  const auto actual = vllm::ModelRegistry::Forward(*model, device.Get());
  Step host({3, 11, 7});
  CheckLogits(actual, vllm::ModelRegistry::Forward(*model, host.Get()));
  CHECK(Backend().uploads.empty());
  CHECK(Backend().gate_calls.empty());
  CHECK(Backend().down_calls.empty());
  CHECK(Backend().fit_queries == 0);
}

TEST_CASE("glm5_next placement: mixed and unplaced layers retain their loaded targets") {
  if (!Enabled()) return;
  Environment env;
  bool mixed = false;
  SUBCASE("mixed") { mixed = true; }
  SUBCASE("unplaced positive control") {}
  gguf_test::TempFile file(BuildFixture());
  const auto g = vllm::GgufFile::Open(file.path());
  if (mixed) Place({{vllm::LlmFfnExpsBlockRegex(2), "cpu"}});
  auto model = Load(g);
  Track(*model);
  vllm::ResetActiveMoePlacementPlanForTesting();
  Topology device_pages, host_pages;
  const auto run = [&](std::vector<int32_t> ids, int64_t computed) {
    Step d(ids, {}, computed), h(ids, {}, computed);
    d.Bind(device_pages);
    h.Bind(host_pages);
    d.queue = Backend().CreateQueue();
    const auto got = vllm::ModelRegistry::Forward(*model, d.Get());
    CheckLogits(got, vllm::ModelRegistry::Forward(*model, h.Get()));
  };
  run({3, 11, 7}, 0);
  for (int l = 1; l < kLayers; ++l) {
    CHECK(Backend().uploads[l] == (mixed && l == 2 ? 0 : 3));
    CHECK(Backend().gate_calls[l] == (mixed && l == 2 ? 0 : 1));
    CHECK(Backend().down_calls[l] == (mixed && l == 2 ? 0 : 1));
  }
  // A second model's placement must not change this model on continuation.
  Place({{vllm::kLlmFfnExpsRegex, "cpu"}});
  run({20, 2}, 3);
  for (int l = 1; l < kLayers; ++l) {
    CHECK(Backend().uploads[l] == (mixed && l == 2 ? 0 : 3));
    CHECK(Backend().gate_calls[l] == (mixed && l == 2 ? 0 : 2));
    CHECK(Backend().down_calls[l] == (mixed && l == 2 ? 0 : 2));
  }
}

TEST_CASE("glm5_next placement: expansion and memory fallback remain host calculations") {
  if (!Enabled()) return;
  Environment env;
  FixtureOpts opts;
  SUBCASE("unsupported ROCm format expands normally") { opts.q4_0_experts = true; }
  SUBCASE("insufficient memory") { Backend().room = false; }
  SUBCASE("CPU reference expands normally") { vllm_test::SetEnv("VT_CPU_REF", "1"); }
  SUBCASE("disabled keep-quant expands normally") {
    vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "0");
  }
  gguf_test::TempFile file(BuildFixture(opts));
  const auto g = vllm::GgufFile::Open(file.path());
  auto model = Load(g);
  Track(*model);
  Step device({3, 11, 7}), host({3, 11, 7});
  device.queue = Backend().CreateQueue();
  CheckLogits(vllm::ModelRegistry::Forward(*model, device.Get()),
              vllm::ModelRegistry::Forward(*model, host.Get()));
  CHECK(Backend().uploads.empty());
  CHECK(Backend().gate_calls.empty());
  CHECK(Backend().down_calls.empty());
}

TEST_CASE("glm5_next placement: shared partial-bank refusal and first-match precedence") {
  Environment env;
  CHECK_THROWS_WITH_AS(Place({{"blk\\.1\\.ffn_down_exps", "cpu"}}),
                       doctest::Contains("splits its routed experts"), std::invalid_argument);
  Place({{vllm::LlmFfnExpsBlockRegex(1), "cpu"},
         {vllm::kLlmFfnExpsRegex, "rocm"}});
  CHECK(vllm::ActiveMoePlacementPlan().DeviceForLayer(1) == DeviceType::kCPU);
  CHECK(vllm::ActiveMoePlacementPlan().DeviceForLayer(2) == DeviceType::kROCM);
}

TEST_CASE("glm5_next placement: conflicting captured accelerator refuses before upload") {
  if (!Enabled()) return;
  Environment env;
  gguf_test::TempFile file(BuildFixture());
  const auto g = vllm::GgufFile::Open(file.path());
  auto model = Load(g, DeviceType::kCUDA);
  Track(*model);
  Step step({3, 11, 7});
  step.queue = Backend().CreateQueue();
  std::string error;
  try {
    (void)vllm::ModelRegistry::Forward(*model, step.Get());
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  CHECK(error.find("blk.1") != std::string::npos);
  CHECK(error.find("cuda") != std::string::npos);
  CHECK(error.find("rocm") != std::string::npos);
  CHECK(Backend().uploads.empty());
  CHECK(Backend().gate_calls.empty());
  CHECK(Backend().down_calls.empty());
  CHECK(Backend().fit_queries == 0);
  // No device context requests an explicit host reference, which remains valid.
  Step host({3, 11, 7});
  const auto logits = vllm::ModelRegistry::Forward(*model, host.Get());
  CHECK(std::any_of(logits.host.begin(), logits.host.end(), [](float f) { return f != 0; }));
}

TEST_CASE("glm5_next placement: backstop checks actual sources before device staging") {
  if (!Enabled()) return;
  Environment env;
  FixtureOpts opts;
  opts.q4_0_experts = true;
  gguf_test::TempFile q4_file(BuildFixture(opts)), q8_file(BuildFixture());
  const auto q4 = vllm::GgufFile::Open(q4_file.path());
  const auto q8 = vllm::GgufFile::Open(q8_file.path());
  Place({{vllm::kLlmFfnExpsRegex, "cpu"}});
  auto source_model = Load(q4);
  vllm::ResetActiveMoePlacementPlanForTesting();
  auto model = Load(q8);
  const auto& source = vllm::ModelAs<vllm::Glm5NextLoadedModel>(
      *source_model, "Glm5NextForConditionalGeneration").weights();
  // Deliberately injected invalid runtime state. The actual production load
  // would expand these unsupported accelerator banks. No bytes are retagged.
  auto& owned = const_cast<vllm::Glm5NextWeights&>(
      vllm::ModelAs<vllm::Glm5NextLoadedModel>(
          *model, "Glm5NextForConditionalGeneration").weights());
  const auto dims = vllm::glm5_next::MoeDimsFrom(owned.params);
  auto runtime = vllm::glm5_next::BridgeMoeLayer(owned.layers[1].moe, dims, "blk.1.ffn");
  std::string bank;
  bool through_registry = true;
  SUBCASE("gate and up actual sources") {
    owned.layers[1].moe.gate_exps = source.layers[1].moe.gate_exps;
    owned.layers[1].moe.up_exps = source.layers[1].moe.up_exps;
    bank = "gate_exps";
  }
  SUBCASE("down actual source") {
    owned.layers[1].moe.down_exps = source.layers[1].moe.down_exps;
    bank = "down_exps";
  }
  SUBCASE("up source independently of the cached host view") {
    owned.layers[1].moe.up_exps = source.layers[1].moe.up_exps;
    bank = "up_exps";
    through_registry = false;
    REQUIRE(runtime.quant_banks.up.dtype == vt::DType::kQ8_0);
    REQUIRE(runtime.quant_banks.up_src->dtype == vt::DType::kQ4_0);
  }
  Track(*model);
  std::string error;
  try {
    if (through_registry) {
      Step step({3, 11, 7});
      step.queue = Backend().CreateQueue();
      (void)vllm::ModelRegistry::Forward(*model, step.Get());
    } else {
      vt::Queue host{{DeviceType::kCPU, 0}, nullptr};
      auto queue = Backend().CreateQueue();
      vllm::dense_attn::Dev dev{Backend(), queue};
      std::vector<float> hidden(3 * kH);
      for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = 0.01F * (i + 1);
      (void)vllm::glm5_next::MoeForward(dims, runtime, hidden, 3, host, &dev);
    }
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  CHECK(error.find("blk.1") != std::string::npos);
  CHECK(error.find(bank) != std::string::npos);
  CHECK(error.find(vt::Name(vt::DType::kQ4_0)) != std::string::npos);
  CHECK(error.find("rocm") != std::string::npos);
  CHECK(Backend().uploads.empty());
  CHECK(Backend().gate_calls.empty());
  CHECK(Backend().down_calls.empty());
}
