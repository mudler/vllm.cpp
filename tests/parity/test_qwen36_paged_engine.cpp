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
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

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

  MESSAGE("qwen36_paged_engine: greedy-decoding "
          << kMaxTokens << " tokens through the PAGED engine...");
  const vllm::RequestOutput out =
      loaded->engine().generate(kPrompt, Greedy(kMaxTokens), "gate");

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
