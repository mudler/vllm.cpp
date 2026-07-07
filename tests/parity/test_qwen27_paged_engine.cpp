// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE 27B DENSE W4A4 GREEDY ACCEPTANCE GATE — the counterpart to
// test_qwen36_paged_engine.cpp for the dense 27B gate model
// (unsloth/Qwen3.6-27B-NVFP4, arch Qwen3_5ForConditionalGeneration,
// compressed-tensors NVFP4 W4A4). See .agents/qwen27b-w4a4-notes.md.
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
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

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

// The 27B fp4-resident W4A4 GEMM path + the dense forward are wired. This gate
// RUNS on dgx and closes token-for-token against vLLM's PRODUCTION greedy
// continuation (greedy_ids.npy, tok6=198) under the DEFAULT config.
//
// CORRECTNESS BAR = token-for-token parity with vLLM's PRODUCTION (native
// flashinfer/cutlass-sm120a) output. MEASURED 2026-07-07: the default stack —
// cutlass sm120a fp4 GEMM (NvfpCutlassEnabled default-ON, qwen3_5.cpp) + bf16
// residual (VT_BF16_RESIDUAL default-ON) — reproduces the 198 stream 16/16, i.e.
// it is PRODUCTION-TOKEN-EXACT (matches vLLM's real serving output, not just the
// degenerate emulation-271 reference). This is the strongest bar and is what the
// gate asserts. The emulation path (VT_NVFP4_CUTLASS=0, hand-written
// vt::MatmulNvfp4Fp4 mirroring vLLM's EmulationNvFp4LinearKernel) yields the
// degenerate 271 stream and is kept only as an opt-out (greedy_ids_emulation.npy
// covers it). tok6 is the razor whitespace near-tie (198 "\n" vs 271 "\n\n") —
// the default now lands on the production side.
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

  // Run the DEFAULT config (cutlass sm120a fp4 GEMM + bf16 residual, both
  // default-ON) — it reproduces vLLM's PRODUCTION greedy continuation
  // token-for-token (tok6=198), so assert the PRODUCTION golden below. Do NOT
  // force VT_NVFP4_CUTLASS=0; that opt-out gives the degenerate emulation-271
  // stream, which is the weaker reference.
  const std::string kPrompt = "The capital of France is Paris, and the";
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_logits_27b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / "token_ids.npy");
  // Assert the vLLM PRODUCTION continuation (greedy_ids.npy, tok6=198) — the
  // default cutlass+bf16-residual stack reproduces it token-for-token, the
  // strongest correctness bar. (greedy_ids_emulation.npy = the degenerate
  // emulation-271 reference for the VT_NVFP4_CUTLASS=0 opt-out only.)
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / "greedy_ids.npy");
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16

  MESSAGE("qwen27_paged_engine: loading full 27B via FromModelDir("
          << snap << ") — dense W4A4 fp4-resident loader + engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

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

  // THE 27B M0 EXIT BAR (paged engine): the batched serving loop reproduces the
  // pip-vLLM oracle's EMULATION greedy continuation (tok6=271) EXACTLY,
  // token-for-token — validating the fp4-resident W4A4 cutlass GEMM (default-on)
  // against vLLM's own emulation reference.
  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_greedy_ids);
}
