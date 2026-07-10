// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE 27B DENSE W4A4 GREEDY ACCEPTANCE GATE — the counterpart to
// test_qwen36_paged_engine.cpp for the dense 27B gate model
// (unsloth/Qwen3.6-27B-NVFP4, arch Qwen3_5ForConditionalGeneration,
// compressed-tensors NVFP4 W4A4). See .agents/specs/qwen27b-w4a4-notes.md.
//
// It drives the pinned M0-exit prompt through the FULL PAGED LLMEngine stack
// (LoadedEngine::FromModelDir -> InputProcessor -> Scheduler -> paged attention
// + KV-cache growth + batched GDN + Sampler -> OutputProcessor) via the dense
// arch dispatch, and asserts the greedy (temperature-0) decode reproduces the
// deterministic prefix shared by the two pip-vLLM oracle continuations
// (qwen36_logits_27b/{greedy_ids,greedy_ids_emulation}.npy, §5 step-5)
// TOKEN-FOR-TOKEN over that asserted prefix — validating the fp4-resident W4A4
// tensor-core GEMM (§5 step-6a) against the oracle without claiming the tail.
//
// Checkpoint-GATED + dgx-only, mirroring test_qwen36_paged_engine.cpp: on the
// CPU dev box / CI the snapshot is absent, so the body emits a loud SKIP and
// returns — the test compiles + links on CPU but only RUNS on dgx.casa (GB10).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime_api.h>
#endif

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

void CheckDeviceCacheResidency(
    const vllm::entrypoints::LoadedEngine& loaded) {
#ifdef VLLM_CPP_CUDA
  const char* fallback = std::getenv("VT_DEVICE_KV_CACHE");
  if (fallback != nullptr && fallback[0] == '0') {
    CHECK_FALSE(loaded.runner().kv_cache_backend_resident());
    return;
  }
  REQUIRE(loaded.runner().kv_cache_backend_resident());
  const auto check_device_pointer = [](const void* pointer) {
    cudaPointerAttributes attributes{};
    REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
    CHECK(attributes.type == cudaMemoryTypeDevice);
  };
  for (const vllm::PagedKvCache& cache : loaded.runner().attn_kv()) {
    check_device_pointer(cache.data);
  }
  for (const vllm::GdnStateCache& cache : loaded.runner().gdn_state()) {
    check_device_pointer(cache.ssm_state.data);
    check_device_pointer(cache.conv_state.data);
  }
#else
  CHECK_FALSE(loaded.runner().kv_cache_backend_resident());
#endif
}

// Snapshot dir of the 27B checkpoint (contains config.json), or "". Same HF
// cache layout as Find35BSnapshot, for models--unsloth--Qwen3.6-27B-NVFP4.
std::string Find27BSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

// Load an i32 (.npy "<i4") vector from the committed golden.
std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

// Greedy (argmax) sampling params — temperature 0 => deterministic, matching
// the oracle's temperature-0 continuation.
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// W2 mirrors the native production chain completely: merged gate_up projection,
// FlashInfer's full SM12 tactic family with stream-safe eager launches, and BF16
// GDN recurrence/z output. The latter is correctness-significant at the tok6
// whitespace near-tie: the former f32 GDN diagnostic takes vLLM's coherent
// emulation branch, while the shipping BF16 path reproduces the native production
// continuation for all 16 tokens. The gate therefore requires full equality.
constexpr bool kW4A4ForwardReady = true;

}  // namespace

// The M0-exit prompt (pinned oracle: qwen36_logits_27b/manifest.json) and its
// two 16-token vLLM continuations. NOTE: this is NOT a 16/16 parity gate. It
// proves the PAGED dense engine reproduces the six-token deterministic answer
// prefix shared by vLLM production and emulation, end to end from the prompt
// string through the batched serving loop; the near-tied tail is diagnostic.
TEST_CASE("qwen27 paged-engine greedy acceptance gate (dgx-only, 27B W4A4)") {
  const std::string snap = Find27BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "27B checkpoint absent; skipping (dgx-only) — "
        "unsloth/Qwen3.6-27B-NVFP4 snapshot not present");
    return;
  }
  if (!kW4A4ForwardReady) {
    MESSAGE("27B W4A4 forward not yet wired; skipping");
    return;
  }

  // Run the DEFAULT production config with no tactic/kernel/dtype force.
  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_27b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  // The production stream is the acceptance reference. The emulation stream is
  // retained as a negative-control fixture for the known tok6 near-tie.
  const std::vector<int32_t> want_prod = LoadI32Npy(golden / "greedy_ids.npy");
  const std::vector<int32_t> want_emu = LoadI32Npy(golden / "greedy_ids_emulation.npy");
  const int kMaxTokens = static_cast<int>(want_prod.size());  // 16
  REQUIRE(want_emu != want_prod);

  MESSAGE("qwen27_paged_engine: loading full 27B via FromModelDir("
          << snap << ") — dense W4A4 fp4-resident loader + engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  CheckDeviceCacheResidency(*loaded);

  MESSAGE("qwen27_paged_engine: greedy-decoding "
          << kMaxTokens << " tokens through the PAGED dense engine...");
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gate");

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // Diagnostic: the engine's tokenization of the prompt string must match the
  // oracle's prompt_ids (else a greedy divergence would be a tokenizer mismatch,
  // not a forward-pass regression).
  CHECK(out.prompt_token_ids == want_prompt_ids);

  MESSAGE("qwen27_paged_engine M0-EXIT: produced " << got.size() << "/"
          << kMaxTokens << " tokens; continuation=\""
          << out.outputs[0].text << "\"");

  // THE 27B acceptance bar: native-vLLM production equality through prefill,
  // decode, paged KV/GDN state growth and sampling.
  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_prod);
  CHECK(got != want_emu);
  MESSAGE("qwen27_paged_engine: full production stream 16/16 token-exact vs vLLM");
}
