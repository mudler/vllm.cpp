// vllm.cpp original (checkpoint-gated acceptance gate); no upstream mirror.
//
// THE GGUF 35B GREEDY ACCEPTANCE GATE (MVP requirement #2: the gate models
// load and serve from GGUF, not just safetensors). Drives decisive prompts
// through the FULL PAGED LLMEngine loaded from a REAL APEX k-quant GGUF file
// (single-file config + GGUF-embedded vocab + k-quant dequant -> bf16 weights,
// LoadedEngine::FromModelDir(".gguf")) and asserts the greedy (temperature-0)
// continuations reproduce the pinned per-file oracle TOKEN-FOR-TOKEN.
//
// THE ORACLE (goldens/qwen36_gguf_35b/manifest.json): llama.cpp (mudler's
// qwen35moe fork on dgx.casa) loading the SAME .gguf, greedy, same prompts —
// NOT the safetensors-NVFP4 goldens: the APEX files are k-quant
// re-quantizations of the same base weights, so their greedy continuations
// legitimately differ from the NVFP4 checkpoint's (the oracle itself diverges
// from the NVFP4 golden on APEX-Balanced at token 7). Same-weights greedy
// agreement is the gate.
//
// PROMPT CHOICE: the gate prompts were selected for DECISIVE greedy steps
// (oracle top-2 logprob margins: count >= 1.91 at every step, eiffel >= 0.15).
// The M0-exit prompt is deliberately NOT gated here: the oracle's own first
// generated token is a near-tie (margin 0.040 between " capital" and
// " official"), and our engine (dequant->bf16 + bf16 GEMMs — the same recipe
// vLLM's GGUF loader uses) legitimately takes the other branch of that tie.
// Loader correctness for that class of divergence was established WEIGHT-LEVEL
// instead: every GGUF tensor family was cross-checked against the safetensors
// checkpoint ground truth (norm +1 inversions and ssm_a/dt_bias/conv1d V-head
// reorders EXACT; quantized families corr >= 0.9927 under the documented
// reorder/transpose transforms). See manifest.json.
//
// Checkpoint-GATED + dgx-only, mirroring test_qwen36_paged_engine.cpp: the
// APEX GGUFs live at ~/work/apex/qwen36_35b/ on dgx.casa (override with
// VLLM_GGUF_35B_DIR). Absent files -> loud SKIP; the test compiles + links on
// CPU/CI but only RUNS where the real files exist.
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

// Directory holding the APEX 35B GGUFs (VLLM_GGUF_35B_DIR overrides the
// dgx.casa default ~/work/apex/qwen36_35b), or "" when absent.
std::string FindGgufDir() {
  if (const char* env = std::getenv("VLLM_GGUF_35B_DIR")) {
    std::error_code ec;
    if (fs::is_directory(env, ec)) return env;
    return "";
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path dir = fs::path(home) / "work/apex/qwen36_35b";
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return "";
  return dir.string();
}

// Load an i32 (.npy "<i4") vector from the committed golden.
std::vector<int32_t> LoadI32Npy(const fs::path& p) {
  const parity::NpyArray a = parity::LoadNpy(p.string());
  REQUIRE(a.dtype == "<i4");
  const size_t n = a.data.size() / sizeof(int32_t);
  const auto* src = reinterpret_cast<const int32_t*>(a.data.data());
  return std::vector<int32_t>(src, src + n);
}

// Greedy (argmax) sampling params, PostInit-normalized as the engine would.
vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// One decisive-prompt greedy check against the pinned oracle golden pair
// (<stem>_prompt_ids.npy for the GGUF-embedded-vocab tokenization,
// <stem>_greedy_ids.npy for the continuation).
void CheckPrompt(vllm::entrypoints::LoadedEngine& loaded,
                 const std::string& prompt, const std::string& stem) {
  const fs::path golden = fs::path(PARITY_GOLDENS_DIR) / "qwen36_gguf_35b";
  const std::vector<int32_t> want_prompt_ids =
      LoadI32Npy(golden / (stem + "_prompt_ids.npy"));
  const std::vector<int32_t> want_greedy_ids =
      LoadI32Npy(golden / (stem + "_greedy_ids.npy"));
  const int kMaxTokens = static_cast<int>(want_greedy_ids.size());  // 16

  const vllm::RequestOutput out =
      loaded.engine().generate(prompt, Greedy(kMaxTokens), "gguf-" + stem);

  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;

  // Diagnostic: the GGUF-embedded vocab must tokenize the prompt exactly as
  // the oracle did (llama-server /tokenize) — otherwise a greedy divergence
  // would be a tokenizer mismatch, not a loader/forward regression.
  CHECK(out.prompt_token_ids == want_prompt_ids);

  MESSAGE("qwen36_gguf_engine[" << stem << "]: produced " << got.size() << "/"
          << kMaxTokens << " tokens; continuation=\"" << out.outputs[0].text
          << "\"");

  // THE GGUF GATE: the engine loaded from the real k-quant GGUF reproduces
  // the same-file llama.cpp oracle's greedy continuation token-for-token.
  REQUIRE(static_cast<int>(got.size()) == kMaxTokens);
  CHECK(got == want_greedy_ids);
}

// Loads `gguf_name` once and runs both decisive-prompt gates through it.
void RunGgufGreedyGate(const std::string& gguf_name) {
  const std::string dir = FindGgufDir();
  const fs::path gguf = dir.empty() ? fs::path() : fs::path(dir) / gguf_name;
  std::error_code ec;
  if (dir.empty() || !fs::is_regular_file(gguf, ec)) {
    MESSAGE("APEX GGUF absent; skipping (dgx-only) — " << gguf_name
            << " not present (set VLLM_GGUF_35B_DIR)");
    return;
  }

  MESSAGE("qwen36_gguf_engine: loading " << gguf.string()
          << " via FromModelDir — GGUF metadata/vocab + k-quant dequant...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          gguf.string(), vllm::entrypoints::EngineParams{});

  CheckPrompt(*loaded, "The Eiffel Tower is located in the city of", "eiffel");
  CheckPrompt(*loaded, "1, 2, 3, 4, 5, 6, 7, 8,", "count");
}

}  // namespace

// APEX-Compact: {F32, Q3_K, Q4_K, Q6_K} — exercises the Q3_K/Q4_K/Q6_K
// dequant paths on real 35B tensors (17.3 GB file).
TEST_CASE("qwen36 GGUF paged-engine greedy gate (dgx-only, APEX-Compact)") {
  RunGgufGreedyGate("Qwen3.6-35B-A3B-APEX-Compact.gguf");
}

// APEX-Balanced: {F32, Q8_0, Q5_K, Q6_K} — exercises the Q8_0/Q5_K paths;
// together with Compact this covers every k-quant type our dequant supports
// that appears in the APEX sweep (25.6 GB file).
TEST_CASE("qwen36 GGUF paged-engine greedy gate (dgx-only, APEX-Balanced)") {
  RunGgufGreedyGate("Qwen3.6-35B-A3B-APEX-Balanced.gguf");
}
