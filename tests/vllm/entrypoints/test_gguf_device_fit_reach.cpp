// ENG-EXPERT-STREAM, issue #1123 — the REACHABILITY gate for the load-time
// device-fit refusal. The arithmetic is gated in test_gguf_device_fit; this file
// answers the different question that tree has carried green before: does the
// production loader actually ask?
//
// A test that constructs the predicate by hand proves the predicate works and
// never proves anything reaches it. So this drives
// `LoadedEngine::FromModelDir`, the loader entry point every consumer uses, and
// asserts the thrown MESSAGE. Deleting the call site in `model_loader.cpp` makes
// the refusing case throw the LATER tokenizer error instead, which is red here.
//
// Why a fake platform. `needs_weight_staging()` is true on exactly one platform
// in this tree (`src/vllm/platforms/cuda.cpp:71`), so on a host with no CUDA device
// the branch is unreachable from the real loader — the untestable-device-branch
// shape this row has hit repeatedly. A fake staging platform registered in the
// CUDA lookup slot reaches it, which is the instrument
// `tests/vllm/entrypoints/test_device_selection.cpp` established for exactly
// this reason. It is a SEPARATE executable for the same reason that one is:
// registering into the global platform/backend registries must not leak into
// other suites.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "support/test_env.h"
#include "vllm/config/device.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/gguf_builder.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"

namespace {

using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

// A backend that allocates on the host. Nothing in this file runs a forward; it
// exists so the fake platform has a `Backend&` to return, and so
// `SelectQueueForModel` has something to create a queue from if it is ever
// asked (this file never gets that far — the refusal fires first, and on the
// permitting arm the tokenizer throws first).
class HostBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    return std::malloc(bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
};

// The one property under test: a platform that STAGES weights, carrying a
// budget on its residency policy exactly as `CudaPlatform` now does.
class StagingPlatform final : public vllm::platforms::Platform {
 public:
  StagingPlatform(HostBackend& backend, size_t budget)
      : backend_(backend), budget_(budget) {}

  vt::DeviceType device_type() const override { return vt::DeviceType::kCUDA; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {12, 1};
  }
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kBF16};
  }
  bool needs_weight_staging() const override { return true; }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    vllm::platforms::ResidencyPolicy p;
    p.device_memory_total_bytes = budget_;
    return p;
  }

 private:
  HostBackend& backend_;
  size_t budget_ = 0;
};

HostBackend& Backend() {
  static HostBackend backend;
  return backend;
}

// The budget is deliberately supplied through the POLICY here (0), and moved by
// `VT_DEVICE_WEIGHT_BUDGET_BYTES` in each case, so both halves of
// `DeviceWeightBudgetBytes` are exercised through the production path: the
// unknown-policy arm and the override arm.
StagingPlatform& Platform() {
  static StagingPlatform platform(Backend(), /*budget=*/0);
  return platform;
}

void RegisterFakeStagingPlatform() {
  vt::RegisterBackend(vt::DeviceType::kCUDA, &Backend());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &Platform());
}

// A synthetic `qwen35moe` GGUF: enough hparams for `HfConfigFromGguf` and
// `ModelRegistry::Resolve` to succeed, so the fit check is reached at its real
// position in the ladder (AFTER architecture resolution) and BEFORE any weight
// I/O. It carries no tokenizer, so the arm that is ALLOWED through fails LATER
// and DIFFERENTLY -- measured, not assumed:
//
//   tokenizer: GGUF missing kv "tokenizer.ggml.model"
//
// That is the NEXT step after the check, and it is what makes the permitting
// case meaningful: the load got past the check.
//
// Its total staged footprint is small and asserted below rather than assumed.
std::string BuildSyntheticMoeGguf() {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "qwen35moe"));
  b.AddKv(U32Kv("qwen35moe.embedding_length", 64));
  b.AddKv(U32Kv("qwen35moe.block_count", 2));
  b.AddKv(U32Kv("qwen35moe.attention.head_count", 4));
  b.AddKv(U32Kv("qwen35moe.attention.head_count_kv", 2));
  b.AddKv(U32Kv("qwen35moe.attention.key_length", 16));
  b.AddKv(U32Kv("qwen35moe.expert_count", 4));
  b.AddKv(U32Kv("qwen35moe.expert_used_count", 2));
  b.AddKv(U32Kv("qwen35moe.expert_feed_forward_length", 32));
  b.AddKv(U32Kv("qwen35moe.expert_shared_feed_forward_length", 32));
  b.AddKv(U32Kv("qwen35moe.ssm.group_count", 2));
  b.AddKv(U32Kv("qwen35moe.ssm.time_step_rank", 4));
  b.AddKv(U32Kv("qwen35moe.ssm.state_size", 8));
  b.AddKv(U32Kv("qwen35moe.ssm.conv_kernel", 4));
  b.AddKv(U32Kv("qwen35moe.full_attention_interval", 4));
  b.AddKv(U32Kv("qwen35moe.context_length", 256));
  // Both of these are REQUIRED by HfConfigFromGguf (`ReqFloat`,
  // qwen3_5_gguf_weights.cpp:843-845), which runs BEFORE the fit check. Omitting
  // them made the load throw "missing metadata key" during the config parse and
  // the refusing case never reached the check at all — caught because the case
  // asserted the MESSAGE rather than merely that something threw.
  b.AddKv(gguf_test::F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(gguf_test::F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  // One F32 tensor of 4096 elements: 16384 bytes on disk, 8192 expanded to
  // bf16, so the staged lower bound is 8192.
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  return b.Build();
}

constexpr size_t kStagedLowerBound = 8192;

std::string ThrownMessage(const std::string& gguf_path, vllm::Device device) {
  vllm::entrypoints::EngineParams params;
  params.device = device;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(gguf_path, params);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

TEST_CASE("device fit: the loader REFUSES a GGUF that exceeds the staging budget") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());

  // One byte under the footprint. Chosen at the boundary so the case cannot pass
  // by accident on an implementation that compares the wrong quantity.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kStagedLowerBound - 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // The refusal, not some other failure: the message names the device, the
  // measured need, the budget it exceeded, and the missing capability.
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kStagedLowerBound)) != std::string::npos);
  CHECK(message.find(std::to_string(kStagedLowerBound - 1)) != std::string::npos);
  CHECK(message.find("HOST-ONLY") != std::string::npos);
  CHECK(message.find("device=cpu") != std::string::npos);
  // And it fires BEFORE the tokenizer and therefore before any weight I/O, which
  // is the whole point of refusing at load: everything after this point is the
  // 26 minutes the refusal exists to avoid paying.
  CHECK(message.find("tokenizer") == std::string::npos);
}

TEST_CASE("device fit: a GGUF that FITS the budget is let through to the next stage") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());

  // Exactly the footprint: the boundary on the permitting side, so a mutation
  // that turns `>` into `>=` is red here rather than merely unnoticed.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kStagedLowerBound));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  // It still throws — this synthetic file carries no tokenizer — and that is what
  // makes the case meaningful: the throw is a DIFFERENT one, from the step AFTER
  // the check, which proves the check let it through rather than that it never
  // ran. Asserting the later message positively is the point: a case that only
  // asserted the absence of the refusal would also pass if the loader had died
  // earlier for an unrelated reason, which is exactly how the first draft of
  // this file passed while the config parse was throwing.
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") == std::string::npos);
  CHECK(message.find("HOST-ONLY") == std::string::npos);
  CHECK(message.find("tokenizer: GGUF missing kv") != std::string::npos);
}

TEST_CASE("device fit: an explicit CPU load is never refused, at any budget") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());

  // A budget of one byte, which every checkpoint exceeds. The CPU platform does
  // not stage weights, so the predicate must not even look — this is the arm
  // that keeps `--device cpu` byte-identical, and it is the arm the measured
  // 370 GiB checkpoint actually serves on.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "1");
  const std::string message = ThrownMessage(f.path(), vllm::Device::kCPU);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  REQUIRE_FALSE(message.empty());  // no tokenizer, as above
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") == std::string::npos);
  CHECK(message.find("tokenizer: GGUF missing kv") != std::string::npos);
}
