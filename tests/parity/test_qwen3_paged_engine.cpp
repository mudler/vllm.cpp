// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED token-exact gate); no
// upstream mirror.
//
// THE PAGED-ENGINE Qwen3-0.6B GREEDY ACCEPTANCE GATE. Drives a standard battery
// of prompts through the FULL PAGED LLMEngine stack (InputProcessor -> Scheduler
// -> paged attention + KV-cache growth + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and asserts the greedy (temperature-0) decode
// reproduces the pinned vLLM 0.25.0 oracle continuation TOKEN-FOR-TOKEN for every
// prompt. This is the non-negotiable correctness bar for the first ADDITIVE
// model.
//
// Checkpoint-GATED + dgx-only: it resolves the real Qwen--Qwen3-0.6B snapshot
// under ~/.cache/huggingface/hub/. On the CPU dev box / CI the snapshot is
// absent, so the body emits a loud SKIP and returns — the test compiles + links
// on CPU but only RUNS on dgx.casa (GB10), where the oracle goldens live under
// tests/parity/goldens/qwen3_greedy_0_6b/ (captured by
// scripts/qwen3-oracle-capture.py against ~/venvs/vllm-oracle).
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

// The standard prompt battery. Kept in the test (the oracle capture script reads
// the SAME list) so the goldens and the gate never drift.
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "Once upon a time,",
      "In the beginning God created",
      "The quick brown fox jumps over",
      "def fibonacci(n):",
      "Water boils at a temperature of",
      "The theory of relativity was developed by",
      "To be or not to be, that is",
      "The largest planet in our solar system is",
      "Machine learning is a subfield of",
      "The mitochondria is the powerhouse of",
      "Roses are red, violets are",
      "The first president of the United States was",
      "E equals m c",
      "A journey of a thousand miles begins with",
      "The chemical symbol for gold is",
  };
  return p;
}

std::string Find06BSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

}  // namespace

// THE SACRED GATE: every prompt in the battery reproduces the vLLM 0.25.0 oracle
// greedy continuation EXACTLY (16/16 tokens, token-for-token), through the full
// paged engine. greedy_ids.npy is a committed [N, T] i32 oracle golden.
TEST_CASE("qwen3-0.6B dense paged-engine greedy token-exact gate (dgx-only, SACRED)") {
  const std::string snap = Find06BSnapshot();
  if (snap.empty()) {
    MESSAGE("Qwen3-0.6B checkpoint absent; skipping (dgx-only) — "
            "Qwen--Qwen3-0.6B snapshot not present");
    return;
  }
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen3_greedy_0_6b";
  if (!fs::exists(golden / "greedy_ids.npy")) {
    MESSAGE("qwen3 oracle golden absent; skipping — run "
            "scripts/qwen3-oracle-capture.py on dgx to capture it");
    return;
  }

  const parity::NpyArray g = parity::LoadNpy((golden / "greedy_ids.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];  // tokens per prompt (16)
  const auto* gd = reinterpret_cast<const int32_t*>(g.data.data());
  REQUIRE(static_cast<size_t>(N) == Prompts().size());

  MESSAGE("qwen3_paged_engine: loading Qwen3-0.6B via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  int pass = 0;
  for (int64_t i = 0; i < N; ++i) {
    const std::vector<int32_t> want(gd + i * T, gd + (i + 1) * T);
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "gate" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    const bool ok = static_cast<int64_t>(got.size()) == T && got == want;
    if (ok) ++pass;
    else {
      int first = -1;
      for (int64_t j = 0; j < T && j < static_cast<int64_t>(got.size()); ++j)
        if (got[static_cast<size_t>(j)] != want[static_cast<size_t>(j)]) { first = static_cast<int>(j); break; }
      MESSAGE("qwen3 DIVERGED prompt[" << i << "] first-mismatch@tok=" << first
              << " got=" << (first >= 0 ? got[static_cast<size_t>(first)] : -1)
              << " want=" << (first >= 0 ? want[static_cast<size_t>(first)] : -1)
              << " \"" << out.outputs[0].text << "\"");
    }
    CHECK(ok);
  }
  MESSAGE("qwen3-0.6B SACRED gate: " << pass << "/" << N
          << " prompts token-exact (" << T << " tokens each)");
  REQUIRE(pass == static_cast<int>(N));
}
