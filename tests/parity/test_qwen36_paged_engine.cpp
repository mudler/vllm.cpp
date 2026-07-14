// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE PAGED-ENGINE 35B GREEDY ACCEPTANCE GATE — the counterpart to
// test_op_parity.cpp's RunQwen36Logits (which validates the DENSE ForwardDense
// path). This one drives the SAME M0-exit prompt through the FULL PAGED
// LLMEngine stack (InputProcessor -> Scheduler -> paged attention + KV-cache
// growth + batched GDN + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and asserts the greedy (temperature-0) decode
// reproduces the pinned-oracle M0-exit continuation TOKEN-FOR-TOKEN.
//
// Checkpoint-GATED + dgx-only, mirroring RunQwen36Logits exactly: it resolves
// the real nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot under
// ~/.cache/huggingface/hub/... (same resolution as Find35BSnapshot in
// test_op_parity.cpp). On the CPU dev box / CI the snapshot is absent, so the
// body emits a loud SKIP MESSAGE and returns — the test compiles + links on CPU
// but only RUNS on dgx.casa (GB10), where scripts/dgx-bringup.sh invokes it.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
          "qwen36 default GDN cache ABI must be FP32 SSM + BF16 conv");
    }
    check_device_pointer(cache.ssm_state.data);
    check_device_pointer(cache.conv_state.data);
  }
#else
  CHECK_FALSE(loaded.runner().kv_cache_backend_resident());
#endif
}

// Snapshot dir of the pinned 35B checkpoint (contains config.json), or "".
// IDENTICAL resolution to test_op_parity.cpp's Find35BSnapshot — the HF cache
// layout for models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/<rev>/.
std::string Find35BSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots";
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
// the oracle's temperature-0 continuation. PostInit normalizes/validates as the
// engine's InputProcessor would.
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

// The M0-exit prompt (pinned oracle: qwen36_logits_35b/manifest.json) and its
// greedy continuation. RunQwen36Logits proved the DENSE forward reproduces
// greedy_ids token-for-token (16/16); this proves the PAGED engine does too,
// end to end from the prompt STRING through the batched serving loop.
TEST_CASE("qwen36 paged-engine greedy acceptance gate (dgx-only, 35B)") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present");
    return;
  }

  // The pinned M0-exit prompt + its oracle greedy continuation (16 tokens). The
  // golden lives beside the RunQwen36Logits inputs; prompt_ids is the
  // tokenization the oracle used (diagnostic cross-check on the engine's own
  // tokenization), greedy_ids is the temperature-0 continuation to reproduce.
  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16

  MESSAGE("qwen36_paged_engine: loading full 35B via FromModelDir("
          << snap << ") — dequant + transpose + engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  CheckDeviceCacheResidency(*loaded);

  MESSAGE("qwen36_paged_engine: greedy-decoding "
          << kMaxTokens << " tokens through the PAGED engine...");
#ifdef VLLM_CPP_CUDA
  vt::cuda::testing::ResetGdnPackedDecodeDebugStats();
#endif
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gate");
#ifdef VLLM_CPP_CUDA
  const uint64_t packed_launches =
      vt::cuda::testing::GetGdnPackedDecodeDebugStats().launches;
  vt::cuda::testing::DisableGdnPackedDecodeDebugStats();
  if (packed_launches != 0)
    throw std::runtime_error("qwen36 selected dense-only packed GDN decode");
#endif

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // Diagnostic: the engine's tokenization of the prompt string must match the
  // oracle's prompt_ids — otherwise a greedy divergence would be a tokenizer
  // mismatch, not a forward-pass regression. (prompt_token_ids carries the
  // engine-tokenized prompt.)
  CHECK(out.prompt_token_ids == want_prompt_ids);

  MESSAGE("qwen36_paged_engine M0-EXIT: produced " << got.size() << "/"
          << kMaxTokens << " tokens; continuation=\""
          << out.outputs[0].text << "\"");

  // THE M0 EXIT BAR (paged engine): the batched serving loop reproduces the
  // oracle's greedy continuation EXACTLY, token-for-token. Greedy is
  // deterministic and was verified at M0 exit (dense: 16/16); this is the
  // paged-engine counterpart and MUST hold.
  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_greedy_ids);
}

// BATCHED decode CUDA-graph acceptance gate (num_reqs>1). Submits N identical
// greedy requests so the engine, once all are past prefill, runs PURE-DECODE
// steps with num_reqs==N through the batched Qwen3_5DecodeGraph (padded up to the
// nearest captured size — N=6 pads to 8, exercising 2 INERT padding rows). Each
// request is independent (own KV + GDN state), so each MUST reproduce the same
// oracle greedy continuation token-for-token; a corrupted real row (padding
// leaking into a real slot) or a bad batched replay would diverge. This is the
// conc>1 counterpart to the num_reqs==1 gate above.
TEST_CASE("qwen36 paged-engine batched-graph greedy gate (dgx-only, 35B)") {
  const std::string snap = Find35BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "35B checkpoint absent; skipping (dgx-only) — "
        "nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot not present");
    return;
  }

  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_35b";
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16
  const int kN = 6;  // pure-decode num_reqs==6 -> padded to captured size 8

  MESSAGE("qwen36_paged_engine(batched): loading 35B via FromModelDir...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});
  CheckDeviceCacheResidency(*loaded);

  MESSAGE("qwen36_paged_engine(batched): greedy-decoding " << kN
          << " concurrent requests x " << kMaxTokens << " tokens...");
  for (int i = 0; i < kN; ++i)
    loaded->engine().add_request("r" + std::to_string(i), kPrompt,
                                 Greedy(kMaxTokens));

  std::map<std::string, vllm::RequestOutput> finished;
  while (loaded->engine().has_unfinished_requests()) {
    for (vllm::RequestOutput& out : loaded->engine().step())
      if (out.finished) finished[out.request_id] = std::move(out);
  }

  // Every concurrent request reproduces the oracle continuation exactly.
  REQUIRE(static_cast<int>(finished.size()) == kN);
  for (int i = 0; i < kN; ++i) {
    const std::string id = "r" + std::to_string(i);
    REQUIRE(finished.count(id) == 1);
    REQUIRE(finished[id].outputs.size() == 1);
    const std::vector<int32_t>& g = finished[id].outputs[0].token_ids;
    REQUIRE(static_cast<int>(g.size()) == kMaxTokens);
    CHECK(g == want_greedy_ids);
  }
}
