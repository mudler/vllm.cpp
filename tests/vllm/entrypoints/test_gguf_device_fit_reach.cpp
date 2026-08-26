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
#include <stdexcept>
#include <string>
#include <vector>

#include "support/test_env.h"
#include "vllm/config/device.h"
#include "vllm/config/weight_residency.h"
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
  // "A platform can be registered while CreateQueue still fails" is the reason
  // `SelectQueueForModel`'s AUTO arm wraps this call in a try/catch and falls
  // back to CPU (`src/vllm/entrypoints/model_loader.cpp::ResolveAutoDevice`). This flag
  // reproduces that box on a
  // host with no CUDA device, so the resolver the fit refusal reads can be
  // checked against the queue the load will actually run on. A flag rather than
  // a second backend, because the registry is global and process-wide: a second
  // registration would leak into the other cases in this file.
  bool create_queue_throws = false;
  vt::Queue CreateQueue() override {
    if (create_queue_throws) {
      throw std::runtime_error("fake backend: no usable device");
    }
    ++queues_created;
    return vt::Queue{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  }
  // Counted so a case can prove the resolver ATTEMPTED a queue rather than
  // merely returning the same answer for another reason, and that it handed the
  // probe queue back. `vt::Queue` is a non-owning handle with no destructor, so a
  // resolver that dropped the value would leak the stream and nothing would say
  // so; these two counters are what makes that observable on a fake backend.
  int queues_created = 0;
  int queues_destroyed = 0;
  void DestroyQueue(vt::Queue&) override { ++queues_destroyed; }
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
  // ENG-EXPERT-STREAM-DEVICE W0d (#1124). The second half of the loader's lane
  // condition. A settable field for the same reason `create_queue_throws` is
  // one: the platform registry is process-global, so a second registration would
  // fight this one, and a flag lets a case move exactly the bit under test.
  bool host_memory_is_device_addressable() const override {
    return host_addressable;
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    vllm::platforms::ResidencyPolicy p;
    p.device_memory_total_bytes = budget_;
    return p;
  }

  bool host_addressable = false;

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
  // qwen3_5_gguf_weights.cpp:843-847), which runs BEFORE the fit check. Omitting
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

// ENG-EXPERT-STREAM-DEVICE W0d (#1124). The same synthetic model plus ONE stacked
// expert tower, so the streaming lane has something to serve.
//
//   blk.0.ffn_gate_exps.weight  Q8_0, ne [32, 2, 4] -> 256 elems, 8 blocks, 272 B
//
// So the lane-OFF bound is 8192 + 272 = 8464, and the lane-ON bound is 8192 plus
// the slot arena. `VT_MOE_EXPERT_STREAM_SLOTS` is pinned to 2 below and the slot
// size resolves to the largest per-expert slice, 272/4 = 68, so the arena is 136
// and the lane-ON bound is 8328. The two bounds STRADDLE the budget the cases
// use, which is what makes them a real comparison rather than two runs of the
// same arithmetic.
constexpr size_t kExpertTowerStaged = 272;
constexpr size_t kLaneOffBound = kStagedLowerBound + kExpertTowerStaged;  // 8464
constexpr size_t kArenaBytes = 136;                                      // 2 x 68
constexpr size_t kLaneOnBound = kStagedLowerBound + kArenaBytes;          // 8328

// GGUF-DEVICE-FIT-EXPAND-POLICY (#1870). The tower's FULL-EXPAND size:
// 256 elems (32 x 2 x 4) x 2 bytes = 512, versus its on-disk 272. `kLaneOffBound`
// above is the MIN-based figure every keep-quant-active case in this file still
// gets (272 is the smaller term); this is what the SAME tower charges once
// `VT_GGUF_KEEP_QUANT=0` leaves no residency-shrinking flag active for it, and
// `RouteGgufTensor`'s totality guarantee makes `kExpandBf16` the only outcome
// rather than one `min()` merely happens to agree with.
constexpr size_t kExpertTowerFullExpand = 512;
constexpr size_t kLaneOffBoundFullExpand =
    kStagedLowerBound + kExpertTowerFullExpand;  // 8704

std::string Q8BlockBytes() {
  std::string b(2, '\0');
  b[1] = '\x3c';  // f16 1.0, little-endian
  for (int i = 0; i < 32; ++i) b.push_back(static_cast<char>(i));
  return b;
}

std::string BuildSyntheticMoeGgufWithExpertTower() {
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
  b.AddKv(gguf_test::F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(gguf_test::F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  std::string tower;
  for (int i = 0; i < 8; ++i) tower += Q8BlockBytes();
  b.AddTensor("blk.0.ffn_gate_exps.weight", {32, 2, 4}, /*ggml_type=*/8, tower);
  return b.Build();
}

// ENG-EXPERT-STREAM-DEVICE W0d repair (#1124). A `deepseek4` GGUF carrying the
// SAME `_exps.weight` suffix, and the same expert tower byte-for-byte, on an
// architecture whose forward never reaches `KqExpertSlice`.
//
// `deepseek_v4_weights.cpp` writes `blk.<n>.ffn_{gate,up,down}_exps.weight`
// exactly as the llama.cpp Qwen MoE export does, and `deepseek_v2.cpp` records at
// its head that this family deliberately does NOT compose `RunMoeBlock` — so no
// slot lane serves it, no arena is ever allocated for it, and every one of its
// towers IS staged. A lane keyed on the suffix alone cannot tell this file from
// the one above, which is the defect this case exists for: with the towers
// dropped from the bound and an arena added that nothing builds, the #1123
// refusal is replaced by the 26-minute-then-`cudaMalloc` death it was created to
// eliminate.
//
// The KV set is the one `tests/vllm/models/test_deepseek_v4_gguf_load.cpp` builds,
// reduced to the keys `DeepseekV4ParamsFromGguf` REQUIRES plus the two its
// self-consistency checks read (`hyper_connection.count`, and
// `attention.compress_ratios` of length >= `block_count`). It carries no
// tokenizer, so the permitted arm dies at the same later step every other case in
// this file uses as its positive control.
std::string BuildSyntheticDeepseekV4GgufWithExpertTower() {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "deepseek4"));
  const std::string p = "deepseek4.";
  b.AddKv(U32Kv(p + "embedding_length", 64));
  b.AddKv(U32Kv(p + "block_count", 2));
  b.AddKv(U32Kv(p + "attention.head_count", 4));
  b.AddKv(U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(U32Kv(p + "attention.key_length", 16));
  b.AddKv(U32Kv(p + "attention.q_lora_rank", 8));
  b.AddKv(U32Kv(p + "attention.output_lora_rank", 8));
  b.AddKv(U32Kv(p + "attention.output_group_count", 1));
  b.AddKv(gguf_test::F32Kv(p + "rope.freq_base", 10000.0F));
  b.AddKv(gguf_test::F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(U32Kv(p + "expert_count", 4));
  b.AddKv(U32Kv(p + "expert_used_count", 2));
  b.AddKv(U32Kv(p + "expert_shared_count", 1));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", 32));
  b.AddKv(gguf_test::F32Kv(p + "swiglu_clamp", 10.0F));
  b.AddKv(U32Kv(p + "hyper_connection.count", 2));
  b.AddKv(gguf_test::I32ArrayKv(p + "attention.compress_ratios",
                                std::vector<int32_t>{1, 1}));
  // Byte-identical to the qwen35moe file above, so the two cases differ in the
  // ARCHITECTURE and in nothing else the bound can see.
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  std::string tower;
  for (int i = 0; i < 8; ++i) tower += Q8BlockBytes();
  b.AddTensor("blk.0.ffn_gate_exps.weight", {32, 2, 4}, /*ggml_type=*/8, tower);
  return b.Build();
}

// ENG-EXPERT-STREAM-DEVICE W0d repair (#1378). The same synthetic `qwen35moe`
// model, with the expert tower stored in NVFP4 (ggml type 40) instead of Q8_0.
//
//   blk.0.ffn_gate_exps.weight  NVFP4, ne [64, 2, 4] -> 512 elems, 8 blocks of
//   64 elements in 36 bytes = 288 B
//
// NVFP4 has no `vt::DType` block encoding by design, so it can never be
// `kKeepQuant`: `RouteGgufTensor` sends it to `kNvfp4Fp4` when the fp4 residency
// is on and to `kExpandBf16` when it is off, and BOTH of those arms reach
// `ResidentWeight` and stage the whole tower. So the lane must stay off and the
// bound must stay whole, on a file whose architecture, tensor names, platform and
// streaming configuration are otherwise identical to the permitted case.
constexpr size_t kNvfp4TowerStaged = 288;                                  // 8 x 36
constexpr size_t kNvfp4LaneOffBound = kStagedLowerBound + kNvfp4TowerStaged;  // 8480
constexpr size_t kNvfp4ArenaBytes = 144;                                   // 2 x 72
constexpr size_t kNvfp4LaneOnBound = kStagedLowerBound + kNvfp4ArenaBytes;  // 8336

std::string BuildSyntheticMoeGgufWithNvfp4ExpertTower() {
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
  b.AddKv(gguf_test::F32Kv("qwen35moe.rope.freq_base", 1000000.0F));
  b.AddKv(gguf_test::F32Kv("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddTensor("token_embd.weight", {64, 64}, /*ggml_type=*/0,
              std::string(4096 * 4, '\0'));
  // 8 NVFP4 blocks: 4 UE4M3 sub-block scales + 32 packed e2m1 bytes each. The
  // BYTES are never decoded by anything this suite runs — the fit check reads the
  // tensor table, and the load dies at the tokenizer long before a GEMM — so a
  // fixed filler is honest here in a way it would not be in a numerics test.
  b.AddTensor("blk.0.ffn_gate_exps.weight", {64, 2, 4}, /*ggml_type=*/40,
              std::string(8 * 36, '\x11'));
  return b.Build();
}

// The lane's environment, set before any case runs. `VT_MOE_EXPERT_STREAM`
// LATCHES on first read, so a case that set it in its own body would work today
// and break the moment a case ordering changed — the same hazard, and the same
// remedy, as the expert-stream suites. It is inert for every case that leaves
// `host_addressable` false, because the loader short-circuits before asking.
//
// `VT_GGUF_KEEP_QUANT` is pinned here for a different reason, and it is a
// deliberate pin rather than a convenience (#1378). The lane now also requires the
// expert towers to take a KEEP residency, and the production default for that is
// `GgufQuantComputeAvailable()`, which asks whether the CURRENT platform has
// `OpId::kMatmulBTQuant` registered. On a real GB10 that is true — `cuda_quant_dot.cu`
// registers it for CUDA — but this suite registers a FAKE platform in the CUDA
// slot of a CPU-only build, where nothing registers that op, so the availability
// default would answer false and every lane-on case would be measuring the
// absence of a CUDA kernel rather than the lane. Pinning the variable states the
// residency the cases are about; the case below moves it to `0` on purpose.
struct EnableExpertStreaming {
  EnableExpertStreaming() {
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM", "1");
    vllm_test::SetEnv("VT_MOE_EXPERT_STREAM_SLOTS", "2");
    vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "1");
  }
};
const EnableExpertStreaming kEnableExpertStreaming;

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

// --- The AUTO arm: the refusal must name the device the load will RUN on ------
//
// `SelectQueueForModel`'s auto arm falls back to CPU when `CreateQueue()` throws,
// and its own comment says why: "a platform can be registered while CreateQueue
// still fails, and CPU must remain reachable". A resolver that only asked
// `CurrentPlatform()` answered `kCUDA` on such a box, so the fit refusal REFUSED
// a checkpoint by naming a device nothing was going to run on — a load that
// previously served on CPU. These two cases are the pair: the same file, the same
// budget, the same platform, differing only in whether the queue can be created.
TEST_CASE("device fit: the AUTO arm refuses when the accelerator queue CAN be created") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());
  Backend().create_queue_throws = false;
  const int created_before = Backend().queues_created;
  const int destroyed_before = Backend().queues_destroyed;

  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kStagedLowerBound - 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kAuto);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  // The POSITIVE control for the case below. Without it, "no refusal" there could
  // mean the auto arm never selects the fake platform at all, and the pair would
  // prove nothing.
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find("HOST-ONLY") != std::string::npos);
  // The resolution went through an ATTEMPTED queue rather than a bare platform
  // query, which is the only way it can agree with the queue selector.
  CHECK(Backend().queues_created == created_before + 1);
  // And it gave the probe queue back. The refusal throws before
  // `SelectQueueForModel` runs, so this load creates exactly one queue and
  // destroys exactly one: a resolver that leaked it reads 1 created, 0 destroyed.
  CHECK(Backend().queues_destroyed == destroyed_before + 1);
}

TEST_CASE("device fit: the AUTO arm refuses NOTHING when the accelerator queue cannot be created") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());
  Backend().create_queue_throws = true;

  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kStagedLowerBound - 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kAuto);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  Backend().create_queue_throws = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // This load runs on CPU, and the CPU platform does not stage weights, so there
  // is nothing to refuse. Asserting the LATER tokenizer error positively rather
  // than merely the absence of the refusal: a case that only checked the absence
  // would also pass if the loader had died earlier for an unrelated reason.
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

// --- ENG-EXPERT-STREAM-DEVICE W0d (#1124): does the LOADER pass the lane? ------
//
// The arithmetic is gated in test_gguf_device_fit. This is the different
// question, and it is the one that decides whether a 369.96 GiB checkpoint loads
// at all: the refusal fires before the tokenizer and before any weight I/O, so
// unless the loader itself computes the lane and hands it to the predicate, the
// device arm never starts. Deleting the `StreamedExpertLane` block in
// `model_loader.cpp` makes the permitting case below refuse, which is red.
//
// All three cases use ONE file, ONE platform and ONE budget, and differ only in
// `host_addressable` — the probed bit W0b added. That is the comparison: same
// checkpoint, same device memory, refused without the predicate and served with
// it.

TEST_CASE("device fit W0d: a device that cannot read host slots is still refused") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = false;
  TempFile f(BuildSyntheticMoeGgufWithExpertTower());

  // Between the two bounds: above the lane-ON figure, below the lane-OFF one.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOnBound + 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  // The WHOLE table, expert tower included — this is the pre-W0d number, and it
  // is what a discrete GPU still gets.
  CHECK(message.find(std::to_string(kLaneOffBound)) != std::string::npos);
  // ...and no lane note, because no lane was computed.
  CHECK(message.find("the expert-stream lane IS active") == std::string::npos);
}

TEST_CASE("device fit W0d: a host-addressable device is LET THROUGH on the same budget") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticMoeGgufWithExpertTower());

  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOnBound + 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  Platform().host_addressable = false;

  // Same file, same budget, same platform: the ONE bit that moved is the probed
  // predicate, and the load now gets past the check to die at the tokenizer —
  // the step immediately after it. The later error is asserted POSITIVELY, so a
  // loader that died earlier for an unrelated reason cannot pass this case.
  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") == std::string::npos);
  CHECK(message.find("tokenizer: GGUF missing kv") != std::string::npos);
}

TEST_CASE("device fit W0d: when the lane is on and it STILL does not fit, the message says so") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticMoeGgufWithExpertTower());

  // One byte under the lane-ON bound. This case is the positive proof that the
  // loader computed and passed the lane rather than merely happening to permit
  // the load: only a footprint built WITH the lane can produce this note and
  // this number, and the number is the arena the two resolvers actually resolve.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOnBound - 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  Platform().host_addressable = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kLaneOnBound)) != std::string::npos);
  CHECK(message.find("the expert-stream lane IS active") != std::string::npos);
  CHECK(message.find(std::to_string(kExpertTowerStaged)) != std::string::npos);
  CHECK(message.find(std::to_string(kArenaBytes)) != std::string::npos);
}

TEST_CASE(
    "device fit W0d: an architecture the slot lane never serves keeps the WHOLE "
    "bound") {
  // The case the first draft of W0d did not have, and the reason the lane grew an
  // architecture term. Everything the loader could see was already identical to
  // the permitted case above — a weight-staging platform, `host_addressable`
  // true, `VT_MOE_EXPERT_STREAM=1` latched on, and a tensor whose name ends in
  // `_exps.weight` — and the ONE thing that differs is the resolved
  // architecture, which decides whether any slot lane exists to serve those
  // towers. `DeepseekV4ForCausalLM`'s forward does not compose `RunMoeBlock`, so
  // it never reaches `KqExpertSlice`, so its towers are staged in full.
  //
  // The direction matters. The two over-counts this bound already documents are
  // conservative: they refuse a checkpoint that would have fitted, and the
  // operator has `VT_DEVICE_WEIGHT_BUDGET_BYTES` to get past them. This one is
  // not. Dropping 335.62 GiB of towers from the bound on a model that stages
  // every one of them REMOVES a refusal that was correct, and what replaces it is
  // the failure #1123 exists to prevent: a 26-minute load and then
  // `cudaMalloc: out of memory` on the first forward.
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticDeepseekV4GgufWithExpertTower());

  // The SAME budget the qwen35moe case is let through on.
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOnBound + 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  Platform().host_addressable = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  // The WHOLE table, expert tower included, and no lane note — the pre-W0d
  // number, which is what every architecture that does not stream must still get.
  CHECK(message.find(std::to_string(kLaneOffBound)) != std::string::npos);
  CHECK(message.find("the expert-stream lane IS active") == std::string::npos);
  // Not the tokenizer: the refusal fired first, which is the point of refusing at
  // load. Asserting this rules out the reading in which the case passes because
  // the deepseek4 config parse died before the check was reached at all.
  CHECK(message.find("tokenizer") == std::string::npos);
}

// --- ENG-EXPERT-STREAM-DEVICE W0d repair (#1378): does the loader ask which -----
// --- RESIDENCY ROUTE the towers will take? --------------------------------------
//
// The two cases below are the architecture case one level down. There the file
// and the platform were identical and the ARCHITECTURE decided; here the file's
// architecture, tensor names, platform, budget and streaming configuration are
// all the permitted case's, and what decides is whether THIS file under THIS
// policy routes its towers to the `expert_*_kq` arm that `KqExpertSlice` serves.
//
// Both are the UNSAFE direction, like the architecture case and unlike the two
// over-counts the bound documents: with the towers dropped from the bound and an
// arena charged that nothing allocates, a load that stages all 335.62 GiB of them
// is permitted, and what replaces the refusal is the 26-minute load and the
// `cudaMalloc: out of memory` first forward #1123 exists to prevent.

TEST_CASE(
    "device fit W0d: the keep-quant OPT-OUT keeps the WHOLE bound, on the same "
    "file the lane serves") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticMoeGgufWithExpertTower());

  // `VT_GGUF_KEEP_QUANT=0` is a DOCUMENTED, supported opt-out, not a corner: it
  // is the two-way override `FromEnv` promises so the historical all-expand load
  // stays reachable in the same binary. With it set, `LoadExpertsOrNvfp4` peeks
  // `kExpandBf16` and fills `expert_gate` rather than `expert_gate_kq`, the
  // forward takes `MoeBlockBf16Cuda`, and every tower is staged through
  // `ResidentWeight`. Byte-for-byte the same file that IS served with the opt-out
  // absent — the case above this one — so this pair isolates the residency route
  // and nothing else.
  vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "0");
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOnBound + 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "1");
  Platform().host_addressable = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  // GGUF-DEVICE-FIT-EXPAND-POLICY (#1870). This case's fixture has no
  // `VT_GGUF_KEEP_F16` or `VT_GGUF_NVFP4_FP4` available either (a CPU-only test
  // binary with no CUDA op registered), so `VT_GGUF_KEEP_QUANT=0` alone leaves
  // EVERY residency flag off and `RouteGgufTensor` can only answer `kExpandBf16`
  // for every tensor. The bound is therefore the FULL-EXPAND figure, not the
  // pre-#1870 min-based one this case used to assert (8464): the expert tower
  // alone moves from its on-disk 272 to its expanded 512.
  CHECK(message.find(std::to_string(kLaneOffBoundFullExpand)) != std::string::npos);
  CHECK(message.find(std::to_string(kLaneOffBound)) == std::string::npos);
  CHECK(message.find("the expert-stream lane IS active") == std::string::npos);
  // The named-cause NOTE #1870 asked for: an operator reading this message is
  // told WHICH knob would shrink the figure, not only which ones suppress the
  // refusal.
  CHECK(message.find("VT_GGUF_KEEP_QUANT=0") != std::string::npos);
  // Not the tokenizer: the refusal fired first, which is the point of refusing at
  // load. This also rules out the reading in which the case passes because the
  // load died earlier for an unrelated reason.
  CHECK(message.find("tokenizer") == std::string::npos);
}

// GGUF-DEVICE-FIT-EXPAND-POLICY, issue #1870's reproduced bug, at the
// reachability level. `kLaneOffBound` (8464) is exactly what the PRE-FIX
// min-based bound reported for this file under `VT_GGUF_KEEP_QUANT=0` — see the
// case above. A budget set to that exact figure is a boundary case for the OLD
// code (`needed == budget` does not refuse) and a genuine refusal for the FIXED
// code, because the load actually stages `kLaneOffBoundFullExpand` (8704) once
// every residency flag is off. This is the shape #1870 reports on real
// hardware: a load the (wrong) bound said fit reaches the point past which
// nothing this suite can observe (a real weight-staging platform) would have
// crashed with a raw allocator failure instead of refusing by name.
TEST_CASE(
    "device fit: VT_GGUF_KEEP_QUANT=0 refuses a load the min-based bound "
    "would have let through") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticMoeGgufWithExpertTower());

  vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "0");
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES", std::to_string(kLaneOffBound));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  vllm_test::SetEnv("VT_GGUF_KEEP_QUANT", "1");
  Platform().host_addressable = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kLaneOffBoundFullExpand)) != std::string::npos);
  CHECK(message.find(std::to_string(kLaneOffBound)) != std::string::npos);  // the budget
  CHECK(message.find("tokenizer") == std::string::npos);
}

TEST_CASE(
    "device fit W0d: an NVFP4 expert tower keeps the WHOLE bound, because fp4 "
    "residency stages it") {
  RegisterFakeStagingPlatform();
  Platform().host_addressable = true;
  TempFile f(BuildSyntheticMoeGgufWithNvfp4ExpertTower());

  // `VT_GGUF_NVFP4_FP4=1` forces the fp4 residency ON, which is the SHIPPED CUDA
  // default (`FromEnv` reads `GgufNvfp4ComputeAvailable()`, true wherever the
  // NVFP4 GEMM is registered) and is not reachable from the availability default
  // in a CPU-only build. It is forced rather than inherited so the route under
  // test is `kNvfp4Fp4` and not the `kExpandBf16` an unregistered kernel would
  // give — both keep the lane off, but only one of them is the arm a GB10 takes.
  vllm_test::SetEnv("VT_GGUF_NVFP4_FP4", "1");
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kNvfp4LaneOnBound + 1));
  const std::string message = ThrownMessage(f.path(), vllm::Device::kNamedPlatform);
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  vllm_test::UnsetEnv("VT_GGUF_NVFP4_FP4");
  Platform().host_addressable = false;

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kNvfp4LaneOffBound)) != std::string::npos);
  CHECK(message.find("the expert-stream lane IS active") == std::string::npos);
  CHECK(message.find("tokenizer") == std::string::npos);
}

// --- The budget as a CONFIG KEY (#1127), through the same loader --------------
//
// The cases above move the budget with `VT_DEVICE_WEIGHT_BUDGET_BYTES`. The three
// below move it with `--offload-config`'s `vllm_cpp.device_fit.weight_budget_bytes`,
// arriving as `EngineParams::weight_residency`. The first two set NO variable at
// all; the third sets one DELIBERATELY, because its subject is the precedence
// between the two inputs rather than the config tier on its own.
//
// This is the reachability half of #1127 and not a second unit test of the
// resolver: `test_weight_residency_config` already builds the config by hand and
// calls `ResolveDeviceWeightBudgetBytes`, which proves the rule and says nothing
// about whether a document reaches it. The chain these cases traverse is
// `parse_weight_residency_extension_json` -> `EngineParams::weight_residency` ->
// `SetWeightResidencyConfig` (the install block at the top of `FromModelDir`) ->
// `DeviceWeightBudgetBytes` (the fit check, later in the SAME call). Measured:
// deleting the install call site in `LoadedEngine::FromModelDir`, and separately
// deleting the delegation inside `DeviceWeightBudgetBytes`, each turn this suite
// RED.
//
// The install and the fit check being in one function is what makes this
// observable without a checkpoint: the install runs before any path or weight
// operation, and the check runs before the tokenizer.

TEST_CASE("device fit: the budget arrives as a CONFIG KEY, with no variable set") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());
  vllm::ResetWeightResidencyConfigForTesting();
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  vllm::entrypoints::EngineParams params;
  params.device = vllm::Device::kNamedPlatform;
  // Parsed from the SAME string the `--offload-config` flag carries, rather than
  // assembled field by field: a case that set the struct member directly would
  // still pass if the parser never learned the key.
  params.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":8191}}})");
  REQUIRE(params.weight_residency->device_weight_budget_bytes.has_value());

  std::string message;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(f.path(), params);
  } catch (const std::exception& e) {
    message = e.what();
  }
  vllm::ResetWeightResidencyConfigForTesting();

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  // One byte under the footprint, exactly as the environment case is, so a
  // comparison against the wrong quantity cannot pass by accident.
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kStagedLowerBound)) != std::string::npos);
  CHECK(message.find(std::to_string(kStagedLowerBound - 1)) != std::string::npos);
  // The refusal names the config form as a way out, not only the variable, so an
  // operator who set the budget with a document is told how to raise it with one.
  CHECK(message.find("weight_budget_bytes") != std::string::npos);
  CHECK(message.find("tokenizer") == std::string::npos);
}

TEST_CASE("device fit: a config budget of ZERO suppresses the refusal") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());
  vllm::ResetWeightResidencyConfigForTesting();
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  vllm::entrypoints::EngineParams params;
  params.device = vllm::Device::kNamedPlatform;
  // The platform's own probe is 0 in this fixture, so a budget of 0 is NOT what
  // distinguishes this case from the default one. What it proves is that a
  // configured zero reaches the check AS a zero rather than being dropped as a
  // falsy value on the way, which is the one direction the merge and the resolver
  // could each have got wrong. The positive control is the case above: the same
  // file and platform refuse when the configured budget is 8191.
  params.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":0}}})");

  std::string message;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(f.path(), params);
  } catch (const std::exception& e) {
    message = e.what();
  }
  // Read what the loader installed BEFORE clearing it: this is the assertion that
  // the document reached the process-global rather than being carried past it.
  const vllm::WeightResidencyConfig installed =
      vllm::ActiveWeightResidencyConfig();
  vllm::ResetWeightResidencyConfigForTesting();

  REQUIRE(installed.device_weight_budget_bytes.has_value());
  CHECK(*installed.device_weight_budget_bytes == 0);

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") == std::string::npos);
  // The LATER error, asserted positively: without it, "no refusal" would also be
  // true of a load that died earlier for an unrelated reason.
  CHECK(message.find("tokenizer: GGUF missing kv") != std::string::npos);
}

TEST_CASE("device fit: the VARIABLE beats the config key, through the loader") {
  RegisterFakeStagingPlatform();
  TempFile f(BuildSyntheticMoeGguf());
  vllm::ResetWeightResidencyConfigForTesting();

  vllm::entrypoints::EngineParams params;
  params.device = vllm::Device::kNamedPlatform;
  // The document suppresses the refusal; the variable puts a refusing budget
  // back. The precedence exists so a benchmark arm is switchable without a
  // restart, and this is that direction: the variable can turn a configured
  // suppression back ON.
  params.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":0}}})");
  vllm_test::SetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES",
                    std::to_string(kStagedLowerBound - 1));
  std::string message;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(f.path(), params);
  } catch (const std::exception& e) {
    message = e.what();
  }
  vllm_test::UnsetEnv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  vllm::ResetWeightResidencyConfigForTesting();

  REQUIRE_FALSE(message.empty());
  CAPTURE(message);
  CHECK(message.find("cannot serve this GGUF") != std::string::npos);
  CHECK(message.find(std::to_string(kStagedLowerBound - 1)) != std::string::npos);
}
