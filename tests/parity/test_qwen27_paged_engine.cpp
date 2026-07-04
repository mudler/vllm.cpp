// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE 27B DENSE W4A4 GREEDY ACCEPTANCE GATE (scaffold) — the counterpart to
// test_qwen36_paged_engine.cpp for the dense 27B gate model
// (unsloth/Qwen3.6-27B-NVFP4, arch Qwen3_5ForConditionalGeneration,
// compressed-tensors NVFP4 W4A4). See .agents/qwen27b-w4a4-notes.md.
//
// STATUS: intentionally SKIPPING. The 27B model forward + the GB10 fp4xfp4 W4A4
// GEMM are not yet implemented (this session set up only the CPU W4A4 dequant
// reference, nvfp4_emulation.h). This file compiles + links on CPU today so the
// gate exists; when the model + GPU kernel land, flip kW4A4ForwardReady to true
// and fill in the FromModelDir + greedy-continuation checks (mirroring the 35B
// gate above it). The oracle greedy golden is captured on dgx via pip-vLLM on
// the SAME checkpoint (AGENTS.md STANDING DIRECTIVE) as a GPU-gated step.
//
// Like the 35B gate, it is checkpoint-gated + dgx-only: on the CPU dev box / CI
// the snapshot is absent, so the body emits a loud SKIP and returns.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

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

// Flip to true once the 27B dense forward + the W4A4 GPU GEMM exist. Until then
// the gate stays a compile-only scaffold (no GPU touched).
constexpr bool kW4A4ForwardReady = false;

}  // namespace

TEST_CASE("qwen27 paged-engine greedy acceptance gate (dgx-only, 27B W4A4)") {
  const std::string snap = Find27BSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "27B checkpoint absent; skipping (dgx-only) — "
        "unsloth/Qwen3.6-27B-NVFP4 snapshot not present");
    return;
  }
  if (!kW4A4ForwardReady) {
    MESSAGE(
        "27B W4A4 forward not yet wired; skipping — implement the dense "
        "Qwen3_5 forward + GB10 fp4xfp4 GEMM, capture the pip-vLLM greedy "
        "golden, then flip kW4A4ForwardReady (see qwen27b-w4a4-notes.md §5)");
    return;
  }

  // TODO(27B bring-up, GPU-gated): mirror the 35B gate —
  //   auto loaded = LoadedEngine::FromModelDir(snap, EngineParams{});
  //   auto out = loaded->engine().generate(kPrompt, Greedy(N), "gate");
  //   REQUIRE(out.finished);
  //   CHECK(out.outputs[0].token_ids == want_greedy_ids);  // 27B oracle golden
  FAIL("kW4A4ForwardReady is true but the gate body is not implemented");
}
