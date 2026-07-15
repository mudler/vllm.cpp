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
// pip-vLLM oracle continuation (qwen36_logits_27b/greedy_ids.npy, §5 step-5)
// TOKEN-FOR-TOKEN — validating the fp4-resident W4A4 tensor-core GEMM (§5
// step-6a) against the oracle.
//
// Checkpoint-GATED + dgx-only, mirroring test_qwen36_paged_engine.cpp: on the
// CPU dev box / CI the snapshot is absent, so the body emits a loud SKIP and
// returns — the test compiles + links on CPU but only RUNS on dgx.casa (GB10).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime_api.h>
#include "vt/cuda/cuda_gdn_internal.h"
#endif

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
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
    if (std::getenv("VT_GDN_STATE_BF16") == nullptr &&
        (cache.ssm_state.dtype != vt::DType::kF32 ||
         cache.conv_state.dtype != vt::DType::kBF16)) {
      throw std::runtime_error(
          "qwen27 default GDN cache ABI must be FP32 SSM + BF16 conv");
    }
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
// greedy continuation (16 tokens). Proves the PAGED dense engine reproduces the
// pip-vLLM oracle's greedy continuation token-for-token, end to end from the
// prompt STRING through the batched serving loop.
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
  loaded->engine().add_request("gate", kPrompt, Greedy(kMaxTokens));
  std::optional<vllm::RequestOutput> final;
  auto consume = [&](std::vector<vllm::RequestOutput> batch) {
    for (vllm::RequestOutput& item : batch)
      if (item.finished) final = std::move(item);
  };

#ifdef VLLM_CPP_CUDA
  // First model step is prefill and must never select packed recurrence.
  vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
#endif
  consume(loaded->engine().step());
#ifdef VLLM_CPP_CUDA
  const uint64_t prefill_packed_launches =
      vt::cuda::testing::GetGdnPackedDecodeDebugStats().launches;
  if (prefill_packed_launches != 0) {
    vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
    throw std::runtime_error(
        "qwen27 packed GDN decode selected during prefill");
  }
  // The next model step is one-token pure non-spec decode. Qwen3.6-27B has 48
  // GDN layers, so default selection is exactly 48 calls; the process-cached
  // rollback arm must issue none.
  vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
#endif
  consume(loaded->engine().step());
#ifdef VLLM_CPP_CUDA
  const uint64_t packed_launches =
      vt::cuda::testing::GetGdnPackedDecodeDebugStats().launches;
  // Packed pure-decode selection is process-coupled to the merged BF16 BA
  // projection (ShouldUsePackedGdnDecode requires merged_ba_enabled + the
  // coupled BF16 dtypes), so EVERY arm that splits BA or reverts a coupled
  // dtype — VT_GDN_PACKED_DECODE=0, master VT_GDN_MERGED_PROJ=0, leaf
  // VT_GDN_MERGED_BA=0, VT_GDN_IN_BF16=0 / VT_GDN_OUT_BF16=0 /
  // VT_GDN_BA_OUT_BF16=0 — runs the decomposed recurrence and must issue
  // ZERO packed launches. (VT_GDN_MERGED_QKVZ is NOT coupled: the qkvz-off
  // arm keeps merged BA and still selects 48.) The env truth table is
  // detail::PackedGdnDecodeEnvSelected, CPU-pinned by
  // test_qwen27_paged_forward.cpp; the DGX VT_GDN_MERGED_PROJ=0 arm at
  // baea3ec proved the old VT_GDN_PACKED_DECODE-only expectation wrong.
  const bool packed_selected = vllm::detail::PackedGdnDecodeEnvSelected(
      vllm::detail::GdnPackedDecodeEnvConfig{
          std::getenv("VT_GDN_PACKED_DECODE"),
          std::getenv("VT_GDN_MERGED_PROJ"),
          std::getenv("VT_GDN_MERGED_BA"), std::getenv("VT_GDN_IN_BF16"),
          std::getenv("VT_GDN_OUT_BF16"),
          std::getenv("VT_GDN_BA_OUT_BF16")});
  const uint64_t expected_packed_launches = packed_selected ? 48U : 0U;
  vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
  if (packed_launches != expected_packed_launches) {
    throw std::runtime_error(
        "qwen27 packed GDN decode dispatch count mismatch");
  }
#endif
  while (loaded->engine().has_unfinished_requests())
    consume(loaded->engine().step());
  if (!final.has_value())
    throw std::runtime_error("qwen27 paged engine produced no final output");
  const vllm::RequestOutput& out = *final;

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
