// CPU-tier contract for the attention-backend REGISTRY + platform-priority
// SELECTION seam (extensibility item 4). Ports the executable spec of
// vllm/v1/attention/backends/registry.py (self-registration) and
// vllm/platforms/cuda.py::get_attn_backend_cls / _get_backend_priorities (the
// capability-ordered priority + first-registered selection) @ pin e24d1b24.
//
// Covers: (1) the gate backends self-register per DeviceType; (2) MakeAttention-
// Backend constructs the named backend / throws when absent; (3) the CPU and
// CUDA (per-capability) priority ORDER matches the vLLM-mirrored lists; (4) the
// selection walk returns the first REGISTERED name (behavior-preserving
// FLASH_ATTN on both, incl. CPU's CPU_ATTN→FLASH_ATTN fallthrough); (5) the
// explicit-override path; (6) sm_100 (major 10) vs sm_121 (major 12) select
// differently ONLY if their preferred backend is registered — here both resolve
// to FLASH_ATTN because FLASHINFER is not implemented, which IS the
// behavior-preserving outcome.
#include <doctest/doctest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/platforms/cuda_attn_priority.h"
#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/attention/registry.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

using vllm::platforms::AttnSelectorConfig;
using vllm::platforms::DeviceCapability;
using vllm::platforms::Platform;
using vllm::platforms::ResidencyPolicy;
using vllm::v1::AttentionBackend;
using vllm::v1::AttentionImpl;
using vllm::v1::AttentionLayer;
using vllm::v1::MLACommonMetadata;
using vllm::v1::TritonMLAImpl;
using vllm::v1::HasAttentionBackend;
using vllm::v1::MakeAttentionBackend;
using vllm::v1::RegisterAttentionBackend;
using vllm::v1::SelectAttentionBackend;
using vllm::v1::SelectAttentionBackendName;
using vt::DeviceType;
using vt::DType;

namespace {

// Host-tensor helpers for the W4 TritonMLAImpl contract cases below.
vt::Tensor MakeTensor(void* data, DType dt, const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = vt::Device{DeviceType::kCPU, 0};
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}
vt::Tensor MakeF32(float* d, const std::vector<int64_t>& s) {
  return MakeTensor(d, DType::kF32, s);
}
vt::Tensor MakeI32(int32_t* d, const std::vector<int64_t>& s) {
  return MakeTensor(d, DType::kI32, s);
}

// A synthetic CUDA platform with a fixed compute capability, so the
// capability-ordered priority + selection can be exercised without a GPU. Since
// W2 it delegates to the REAL shared table (cuda_attn_priority.h) that
// src/vllm/platforms/cuda.cpp uses, so there is no hand-copied duplicate to keep
// in sync — the only thing this class fakes is the capability probe (which needs
// a device) and the composed backend.
class FakeCudaPlatform final : public Platform {
 public:
  explicit FakeCudaPlatform(int major, int minor) : cap_{major, minor} {}
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return cap_; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    return vllm::platforms::LookupAttnPriority(cap_.major, cfg);
  }
  std::vector<std::string> get_mla_prefill_backend_priority() const override {
    return vllm::platforms::LookupMlaPrefillPriority(cap_.major);
  }

 private:
  DeviceCapability cap_;
};

// A platform whose top priority is a test-only backend, to prove the walk stops
// at the first REGISTERED name rather than always falling to FLASH_ATTN. Uses a
// unique name so it never collides with a real backend other cases assert about.
class TopIsTestBackendPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
  DeviceCapability get_device_capability() const override { return {10, 0}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
  ResidencyPolicy residency_policy() const override { return {}; }
  std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg) const override {
    (void)cfg;
    return {"TEST_ONLY_ATTN", "FLASH_ATTN"};
  }
};

}  // namespace

TEST_CASE("gate backends self-register per DeviceType") {
  // FlashAttentionBackend (backend.cpp) registers for CUDA and CPU; GDN
  // (gdn_attn.cpp) registers for CUDA and CPU. Static registrars ran at load.
  CHECK(HasAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCPU, "FLASH_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCUDA, "GDN_ATTN"));
  CHECK(HasAttentionBackend(DeviceType::kCPU, "GDN_ATTN"));

  // Backends we name in the priority lists but do not implement are NOT
  // registered — the selection walk must skip them.
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCUDA, "FLASHINFER"));
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCUDA, "TRITON_ATTN"));
  CHECK_FALSE(HasAttentionBackend(DeviceType::kCPU, "CPU_ATTN"));
}

TEST_CASE("MakeAttentionBackend constructs the named backend / throws when absent") {
  std::unique_ptr<AttentionBackend> flash =
      MakeAttentionBackend(DeviceType::kCUDA, "FLASH_ATTN");
  REQUIRE(flash != nullptr);
  CHECK(flash->get_name() == "FLASH_ATTN");
  // FlashAttention NHD KV layout is unchanged: (num_blocks, 2, block, H, D).
  const std::vector<int64_t> shape = flash->get_kv_cache_shape(10, 16, 2, 128);
  const std::vector<int64_t> expected{10, 2, 16, 2, 128};
  CHECK(shape == expected);

  std::unique_ptr<AttentionBackend> gdn =
      MakeAttentionBackend(DeviceType::kCPU, "GDN_ATTN");
  REQUIRE(gdn != nullptr);
  CHECK(gdn->get_name() == "GDN_ATTN");

  CHECK_THROWS_AS(MakeAttentionBackend(DeviceType::kCUDA, "NOPE"),
                  std::runtime_error);
}

TEST_CASE("CUDA priority order mirrors _get_backend_priorities (non-MLA)") {
  // BEHAVIOR-PRESERVING across W2: these two lists are byte-identical to the
  // pre-W2 assertions; the default AttnSelectorConfig{} is use_mla=false.
  // major != 10 (incl. GB10 sm_121 == major 12): FLASH_ATTN first.
  FakeCudaPlatform sm121(12, 1);
  const std::vector<std::string> else_order{
      "FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"};
  CHECK(sm121.get_attn_backend_priority(AttnSelectorConfig{}) == else_order);

  // major == 10 (Blackwell datacenter): FLASHINFER first.
  FakeCudaPlatform sm100(10, 0);
  const std::vector<std::string> major10_order{
      "FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"};
  CHECK(sm100.get_attn_backend_priority(AttnSelectorConfig{}) == major10_order);
}

// ─── W2: the MLA branch of _get_backend_priorities (cuda.py:93-142) ─────────
// Ported from tests/v1/attention/test_attention_backends_selection.py (its MLA
// cases) @ pin e24d1b24.

TEST_CASE("MLA priority on GB10 (major 12) is exactly [TRITON_MLA, SPARSE_SM120]") {
  // cuda.py:129-133. TWO entries, that order, nothing else — the single most
  // plan-shaping fact of the MLA campaign, and the reason the whole
  // sm90/sm100-only MLA kernel class (FlashMLA / CUTLASS MLA / FlashInfer MLA /
  // TokenSpeed MLA) is out of scope on our hardware.
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig mla;
  mla.use_mla = true;
  const std::vector<std::string> expected{"TRITON_MLA", "FLASHINFER_MLA_SPARSE_SM120"};
  CHECK(sm121.get_attn_backend_priority(mla) == expected);
}

TEST_CASE("MLA priority on sm_100 mirrors the adaptive sparse tail") {
  // cuda.py:117-131 + the :96-115 sparse ordering rule.
  FakeCudaPlatform sm100(10, 0);
  const std::vector<std::string> head{"FLASHINFER_MLA", "TOKENSPEED_MLA",
                                      "CUTLASS_MLA",    "FLASH_ATTN_MLA",
                                      "FLASHMLA",       "TRITON_MLA"};

  // bf16 KV, head count unknown (upstream None) -> FlashMLA leads the tail.
  AttnSelectorConfig plain;
  plain.use_mla = true;
  std::vector<std::string> expected = head;
  expected.emplace_back("FLASHMLA_SPARSE");
  expected.emplace_back("FLASHINFER_MLA_SPARSE");
  CHECK(sm100.get_attn_backend_priority(plain) == expected);

  // fp8 KV cache -> FlashInfer leads (cuda.py:96-102).
  AttnSelectorConfig quant = plain;
  quant.quantized_kv_cache = true;
  std::vector<std::string> expected_q = head;
  expected_q.emplace_back("FLASHINFER_MLA_SPARSE");
  expected_q.emplace_back("FLASHMLA_SPARSE");
  CHECK(sm100.get_attn_backend_priority(quant) == expected_q);

  // bf16 KV with num_heads <= 16 -> FlashInfer leads (cuda.py:105-110). 16 is
  // DeepSeek-V2-Lite's head count, so this is the boundary that matters.
  AttnSelectorConfig few = plain;
  few.num_heads = 16;
  CHECK(sm100.get_attn_backend_priority(few) == expected_q);
  // 17 heads crosses back (strict `<= 16`).
  AttnSelectorConfig many = plain;
  many.num_heads = 17;
  CHECK(sm100.get_attn_backend_priority(many) == expected);
}

TEST_CASE("MLA priority on sm_90 and older is the `else` arm") {
  // cuda.py:134-142.
  FakeCudaPlatform sm90(9, 0);
  AttnSelectorConfig mla;
  mla.use_mla = true;
  const std::vector<std::string> expected{"FLASH_ATTN_MLA", "FLASHMLA",
                                          "FLASHINFER_MLA", "TRITON_MLA",
                                          "FLASH_ATTN_MLA_SPARSE", "FLASHMLA_SPARSE"};
  CHECK(sm90.get_attn_backend_priority(mla) == expected);
}

TEST_CASE("MLA selection on GB10 resolves to TRITON_MLA (the W0 observation)") {
  // THE W2 POSITIVE SIGNAL: the new MLA path is not merely compiled — a
  // use_mla=true request walks the new table, applies the new is_mla/is_sparse
  // filter, and lands on TRITON_MLA, exactly what the vLLM 0.25.0 oracle logged
  // on sm_121 at W0 ("Using TRITON_MLA attention backend out of potential
  // backends: ['TRITON_MLA']").
  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig mla;
  mla.use_mla = true;
  CHECK(SelectAttentionBackendName(sm121, "", mla) == "TRITON_MLA");

  std::unique_ptr<AttentionBackend> b = SelectAttentionBackend(sm121, "", mla);
  REQUIRE(b != nullptr);
  CHECK(b->get_name() == "TRITON_MLA");
  CHECK(b->is_mla());
  CHECK_FALSE(b->is_sparse());
  // mla_attention.py:1216-1224 — THREE dims, no K/V axis. DeepSeek-V2-Lite's
  // real geometry: block 16, one kv head, head_size 512+64 == 576.
  const std::vector<int64_t> shape = b->get_kv_cache_shape(10, 16, 1, 576);
  const std::vector<int64_t> expected{10, 16, 576};
  CHECK(shape == expected);
  // num_kv_heads != 1 is refused rather than silently ignored.
  CHECK_THROWS_AS(b->get_kv_cache_shape(10, 16, 8, 576), std::invalid_argument);
  // triton_mla.py:100-103 supports_block_size.
  CHECK_THROWS_AS(b->get_kv_cache_shape(10, 24, 1, 576), std::invalid_argument);

  // ─── W4: get_impl_cls() is no longer nullptr ─────────────────────────────
  // triton_mla.py:126-128 returns TritonMLAImpl. THE W4 POSITIVE SIGNAL for the
  // engine seam: the backend the selector resolves now hands back a real impl,
  // so the MLA path is SELECTABLE end-to-end, not just compiled.
  std::unique_ptr<AttentionImpl> impl = b->get_impl_cls();
  REQUIRE(impl != nullptr);
  auto* mla_impl = dynamic_cast<TritonMLAImpl*>(impl.get());
  REQUIRE(mla_impl != nullptr);
  CHECK(TritonMLAImpl::kCanReturnLseForDecode);  // triton_mla.py:135

  // Every other registered backend still returns the base nullptr — W4 filled
  // exactly one hook and moved nothing else.
  std::unique_ptr<AttentionBackend> dense = SelectAttentionBackend(sm121, "", AttnSelectorConfig{});
  REQUIRE(dense != nullptr);
  CHECK(dense->get_name() == "FLASH_ATTN");
  CHECK(dense->get_impl_cls() == nullptr);
}

TEST_CASE("TritonMLAImpl refuses what upstream refuses, and names W5 for prefill") {
  // triton_mla.py:165-179 — the impl's constructor rejects alibi_slopes /
  // sliding_window / logits_soft_cap and any non-decoder attention type. Of
  // those, `window_size` is the one our AttentionImpl surface can carry, so it
  // is the one we refuse; the others are unreachable by construction.
  TritonMLAImpl impl;
  impl.num_heads = 16;
  impl.head_size = 576;
  impl.scale = 0.05f;

  std::vector<int32_t> block_ids{0, 1};
  std::vector<int32_t> lens{16, 16};
  MLACommonMetadata md;
  md.block_table = MakeI32(block_ids.data(), {2, 1});
  md.seq_lens = MakeI32(lens.data(), {2});
  md.max_seq_len = 16;

  std::vector<float> cache(static_cast<size_t>(2) * 16 * 576, 0.0f);
  std::vector<float> q(static_cast<size_t>(2) * 16 * 576, 0.0f);
  std::vector<float> out(static_cast<size_t>(2) * 16 * 512, 0.0f);
  vt::Tensor t_cache = MakeF32(cache.data(), {2, 16, 576});
  vt::Tensor t_q = MakeF32(q.data(), {2, 16, 576});
  vt::Tensor t_out = MakeF32(out.data(), {2, 16, 512});

  AttentionLayer layer;
  layer.window_size = vt::AttentionWindow{128, 0};
  CHECK_THROWS_AS(impl.forward_mqa(layer, t_q, t_cache, md, t_out), std::invalid_argument);

  // A prefill-shaped batch (more query tokens than requests) is W5, and says so.
  AttentionLayer plain;
  std::vector<float> q_prefill(static_cast<size_t>(8) * 16 * 576, 0.0f);
  vt::Tensor t_q_prefill = MakeF32(q_prefill.data(), {8, 16, 576});
  CHECK_THROWS_AS(impl.forward(plain, t_q_prefill, t_q_prefill, t_q_prefill, t_cache, md, t_out),
                  std::invalid_argument);
}

TEST_CASE("the sparse/DSA seam: is_sparse() filters SPARSE_SM120 out of the dense walk") {
  // FLASHINFER_MLA_SPARSE_SM120 sits in GB10's list at position 2 but loses on
  // is_sparse() (flashinfer_mla_sparse.py:67) vs use_sparse=false — the exact
  // reason the oracle's candidate list printed a SINGLE entry. Registering a
  // stand-in sparse backend proves the filter is real and that a future DSA
  // backend needs NO selector edit: it is selected purely by declaring
  // is_sparse() == true.
  class FakeSparseMlaBackend final : public AttentionBackend {
   public:
    std::string get_name() const override { return "FLASHINFER_MLA_SPARSE_SM120"; }
    std::vector<int64_t> get_kv_cache_shape(int64_t nb, int64_t bs, int64_t,
                                            int64_t hs,
                                            const std::string& = "auto") const override {
      return {nb, bs, hs};
    }
    bool is_mla() const override { return true; }
    bool is_sparse() const override { return true; }
  };
  RegisterAttentionBackend(DeviceType::kCUDA, "FLASHINFER_MLA_SPARSE_SM120",
                           []() -> std::unique_ptr<AttentionBackend> {
                             return std::make_unique<FakeSparseMlaBackend>();
                           });

  FakeCudaPlatform sm121(12, 1);
  AttnSelectorConfig dense_mla;
  dense_mla.use_mla = true;
  // Even with the sparse backend REGISTERED, a dense-MLA request still gets
  // TRITON_MLA — the filter, not mere absence, is what excludes it.
  CHECK(SelectAttentionBackendName(sm121, "", dense_mla) == "TRITON_MLA");

  // A sparse request skips TRITON_MLA (is_sparse() false) and takes the sparse
  // entry — the seam working end to end, with zero selector code keyed on DSA.
  AttnSelectorConfig sparse_mla;
  sparse_mla.use_mla = true;
  sparse_mla.use_sparse = true;
  CHECK(SelectAttentionBackendName(sm121, "", sparse_mla) == "FLASHINFER_MLA_SPARSE_SM120");

  // An explicit override that contradicts the request is rejected, mirroring
  // upstream validate_configuration rather than being silently honored.
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "TRITON_MLA", sparse_mla),
                  std::invalid_argument);
}

TEST_CASE("a dense request never selects an MLA backend, and vice versa") {
  FakeCudaPlatform sm121(12, 1);
  // Dense (the gate models): unchanged FLASH_ATTN, even though TRITON_MLA is now
  // registered — it is not in the non-MLA priority list AND would fail is_mla().
  CHECK(SelectAttentionBackendName(sm121) == "FLASH_ATTN");

  // MLA on CPU: no CPU MLA backend exists upstream at the pin, so the walk
  // yields nothing and throws rather than falling back to a dense backend.
  class FakeCpuMlaPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kCPU; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    std::vector<std::string> get_attn_backend_priority(
        const AttnSelectorConfig& cfg) const override {
      (void)cfg;
      return {"CPU_ATTN", "FLASH_ATTN"};
    }
  } cpu;
  AttnSelectorConfig mla;
  mla.use_mla = true;
  CHECK_THROWS_AS(SelectAttentionBackendName(cpu, "", mla), std::runtime_error);
}

TEST_CASE("MLA prefill backend priority mirrors the prefill selector") {
  // Ports tests/v1/attention/test_mla_prefill_selector.py +
  // test_mla_prefill_registry.py (the capability-ordering cases) @ e24d1b24.
  // selector.py:66-76. GB10 (major 12) falls in the `else` arm: FLASH_ATTN
  // ALONE — OBSERVED at W0 ("Using FLASH_ATTN MLA prefill backend."). There is
  // no safety net below it: selector.py:191-194 hard-raises rather than falling
  // back, which is why W5's FA-2 generalization is mandatory, not optional.
  FakeCudaPlatform sm121(12, 1);
  const std::vector<std::string> gb10{"FLASH_ATTN"};
  CHECK(sm121.get_mla_prefill_backend_priority() == gb10);

  FakeCudaPlatform sm90(9, 0);
  CHECK(sm90.get_mla_prefill_backend_priority() == gb10);

  FakeCudaPlatform sm100(10, 0);
  const std::vector<std::string> blackwell{"FLASH_ATTN", "TRTLLM_RAGGED",
                                           "FLASHINFER", "TOKENSPEED_MLA"};
  CHECK(sm100.get_mla_prefill_backend_priority() == blackwell);

  // The base Platform default is empty (no MLA prefill preference).
  class NoMlaPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kMETAL; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
  } none;
  CHECK(none.get_mla_prefill_backend_priority().empty());
}

TEST_CASE("selection walks priority and returns the first REGISTERED backend") {
  // GB10 sm_121: FLASH_ATTN is first AND registered → selected (behavior-
  // preserving; the NHD layout the runtime uses today).
  FakeCudaPlatform sm121(12, 1);
  CHECK(SelectAttentionBackendName(sm121) == "FLASH_ATTN");
  std::unique_ptr<AttentionBackend> b = SelectAttentionBackend(sm121);
  REQUIRE(b != nullptr);
  CHECK(b->get_name() == "FLASH_ATTN");

  // sm_100: FLASHINFER is preferred but UNREGISTERED, so the walk falls through
  // to FLASH_ATTN — same behavior-preserving outcome until FLASHINFER lands.
  FakeCudaPlatform sm100(10, 0);
  CHECK(SelectAttentionBackendName(sm100) == "FLASH_ATTN");
}

TEST_CASE("selection stops at the first registered name (not always FLASH_ATTN)") {
  // Prove the walk is real: register a test-only backend for CUDA, then a
  // platform preferring it selects it over FLASH_ATTN. The unique name is not
  // asserted-absent by any other case (registry mutation is global/persistent).
  RegisterAttentionBackend(DeviceType::kCUDA, "TEST_ONLY_ATTN",
                           []() -> std::unique_ptr<AttentionBackend> {
                             return std::make_unique<vllm::v1::FlashAttentionBackend>();
                           });
  TopIsTestBackendPlatform p;
  CHECK(SelectAttentionBackendName(p) == "TEST_ONLY_ATTN");
}

TEST_CASE("CPU selection: CPU_ATTN preference falls through to FLASH_ATTN") {
  // The real CpuPlatform priority is {CPU_ATTN, FLASH_ATTN}; CPU_ATTN is not
  // implemented, so the walk returns FLASH_ATTN (the layout our CPU paged-attn
  // kernel uses). Exercised here through a synthetic CPU platform mirroring
  // cpu.cpp so no GPU/accelerator resolution is involved.
  class FakeCpuPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kCPU; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    std::vector<std::string> get_attn_backend_priority(
        const AttnSelectorConfig& cfg) const override {
      (void)cfg;
      return {"CPU_ATTN", "FLASH_ATTN"};
    }
  } cpu;
  CHECK(SelectAttentionBackendName(cpu) == "FLASH_ATTN");
}

TEST_CASE("explicit backend override is honored / validated") {
  FakeCudaPlatform sm121(12, 1);
  // A registered override is returned as-is (upstream selected_backend arg).
  CHECK(SelectAttentionBackendName(sm121, "FLASH_ATTN") == "FLASH_ATTN");
  // An unregistered override throws (upstream ValueError).
  CHECK_THROWS_AS(SelectAttentionBackendName(sm121, "TRITON_ATTN"),
                  std::invalid_argument);
}

TEST_CASE("empty priority yields no backend (base Platform default)") {
  class NoPriorityPlatform final : public Platform {
   public:
    DeviceType device_type() const override { return DeviceType::kMETAL; }
    vt::Backend& backend() const override { return vt::GetBackend(DeviceType::kCPU); }
    DeviceCapability get_device_capability() const override { return {}; }
    std::vector<DType> supported_dtypes() const override { return {DType::kBF16}; }
    ResidencyPolicy residency_policy() const override { return {}; }
    // Uses the base default get_attn_backend_priority() == {}.
  } none;
  CHECK(none.get_attn_backend_priority(AttnSelectorConfig{}).empty());
  CHECK_THROWS_AS(SelectAttentionBackendName(none), std::runtime_error);
}
