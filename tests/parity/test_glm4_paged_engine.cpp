// vllm.cpp original (GLM G2 — THE SACRED correctness gate for the first
// GLM-family model); no upstream mirror (upstream has NO GLM-4 text-correctness
// test — spike §"Tests to port" R6).
//
// THE PAGED-ENGINE Glm4ForCausalLM (GLM-4-9B-0414, BF16) GREEDY CORRECTNESS GATE.
// Drives the standard prompt battery through the FULL PAGED LLMEngine stack
// (InputProcessor -> Scheduler -> paged attention + KV growth -> Sampler ->
// OutputProcessor) via LoadedEngine::FromModelDir, and checks the greedy
// (temperature-0) decode against the pinned vLLM 0.25.0 oracle.
//
// GATE FORM (selected BY MEASUREMENT, near-tie-distributional methodology): run
// vLLM's OWN per-prompt greedy K=5 first (glm4-oracle-capture.py --per-prompt). A
// 9.4B dense model is well above the tiny-dense near-tie regime, so vLLM is
// EXPECTED self-deterministic -> the bar is STRICT token-exact (strict_exact/N).
// The gate still carries the near-tie band as the honest fallback where vLLM's own
// bf16 prefill/decode disagree on a tie; a token OUTSIDE vLLM's own near-tie band
// (gap > kNearTieMnats in vLLM's teacher-forced logits on OUR prefix) is a REAL
// forward divergence the gate FAILS on. This is what proves the partial-rotary
// slice and the sandwich norms are actually applied — a silently-dropped norm or
// unapplied rope emits FLUENT WRONG tokens far outside the band (the OPT BOS mode).
//
// Goldens (tests/parity/goldens/glm4_greedy_9b/, captured on dgx):
//   greedy_ids.npy         [N,T]   i32  vLLM per-prompt deterministic greedy.
//   greedy_dist.npy        [N,T,K] i32  K per-prompt runs (determinism evidence).
//   our_ids.npy            [N,T]   i32  OUR CUDA engine greedy (anchor).
//   neartie_gap_mnats.npy  [N,T]   i32  vLLM teacher-forced gap (milli-nats) for
//                                       OUR token given OUR prefix.
//
// Checkpoint-GATED + dgx-only: resolves ~/.cache/huggingface/hub/models--zai-org
// --GLM-4-9B-0414. On CPU/CI the snapshot + goldens are absent, so the case emits
// a loud SKIP and returns (compiles + links on CPU, RUNS only on dgx).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vt/ops.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats (0.5 nats) — identical to the
// Qwen3-dense gate: vLLM's argmax must beat OUR token by <= this in vLLM's OWN
// teacher-forced logits given OUR prefix for a divergence to count as a bf16
// near-tie rather than a forward bug. Sits far below real-divergence scale.
constexpr int32_t kNearTieMnats = 500;

// The standard prompt battery — MUST match glm4-oracle-capture.py::PROMPTS.
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

std::string FindSnapshot(const std::string& repo_dir) {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) / ".cache/huggingface/hub" / repo_dir / "snapshots";
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

const int32_t* AsI32(const parity::NpyArray& a) {
  return reinterpret_cast<const int32_t*>(a.data.data());
}

void RunGate(const std::string& repo_dir, const std::string& golden_subdir,
             const char* label) {
  const std::string snap = FindSnapshot(repo_dir);
  if (snap.empty()) {
    MESSAGE(label << " checkpoint absent; skipping (dgx-only) — " << repo_dir);
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE(label << " greedy golden absent; skipping — capture on dgx: "
            "glm4-oracle-capture.py --per-prompt");
    return;
  }
  // BOOTSTRAP: dump OUR token ids so glm4-neartie-gap.py can build the gap golden.
  if (dump && !have_gap) {
    MESSAGE(label << ": BOOTSTRAP dump (gap golden absent)...");
    std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
        vllm::entrypoints::LoadedEngine::FromModelDir(
            snap, vllm::entrypoints::EngineParams{});
    const parity::NpyArray gg = parity::LoadNpy((gdir / "greedy_ids.npy").string());
    const int64_t NN = gg.shape[0], TT = gg.shape[1];
    std::vector<int32_t> buf(static_cast<size_t>(NN * TT), -1);
    for (int64_t i = 0; i < NN; ++i) {
      const vllm::RequestOutput out = le->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(TT)),
          "boot" + std::to_string(i));
      const std::vector<int32_t>& got = out.outputs[0].token_ids;
      for (int64_t j = 0; j < TT && j < static_cast<int64_t>(got.size()); ++j)
        buf[static_cast<size_t>(i * TT + j)] = got[static_cast<size_t>(j)];
    }
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) { std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f); std::fclose(f); }
    MESSAGE(label << " BOOTSTRAP dumped our token ids -> " << path);
    return;
  }
  if (!have_gap) {
    MESSAGE(label << " gap goldens absent; skipping — run under VT_DUMP_IDS=1 "
            "then glm4-neartie-gap.py");
    return;
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(o.shape.size() == 2);
  REQUIRE(o.shape[0] == N);
  REQUIRE(o.shape[1] == T);
  REQUIRE(gap.shape.size() == 2);
  REQUIRE(gap.shape[0] == N);
  REQUIRE(gap.shape[1] == T);
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  const int32_t* gd = AsI32(g);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  std::vector<int32_t> our_dump;
  if (dump) our_dump.assign(static_cast<size_t>(N * T), -1);

  MESSAGE(label << ": loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  int strict_exact = 0;
  int neartie_only = 0;
  int fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "gate" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    if (dump) {
      for (int64_t j = 0; j < T; ++j)
        our_dump[static_cast<size_t>(i * T + j)] = got[static_cast<size_t>(j)];
    }

    // Hard anchor: our exact committed deterministic sequence. A drift is a hard
    // REQUIRE — it means the engine changed and the teacher-forced gaps no longer
    // describe our tokens; re-run glm4-neartie-gap.py.
    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    REQUIRE_MESSAGE(first_div < 0,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor=" << (first_div < 0 ? -1 : od[i * T + first_div])
                    << " — re-run glm4-neartie-gap.py to refresh the gap golden");

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = static_cast<int>(j); }
      if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ") \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }

  if (dump) {
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(our_dump.data(), sizeof(int32_t), our_dump.size(), f);
      std::fclose(f);
      MESSAGE(label << " dumped our token ids -> " << path);
    }
  }
  MESSAGE(label << " correctness gate: " << (strict_exact + neartie_only) << "/" << N
          << " prompts PASS  (STRICT token-exact vs vLLM per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}

}  // namespace

// GLM-4-9B-0414 (dense, sandwich norms + partial interleaved rope) — the first
// GLM-family SACRED gate (spike G2). Proves the two new primitives are applied.
TEST_CASE("glm4-9B-0414 dense paged-engine greedy correctness gate (dgx-only, SACRED)") {
  RunGate("models--zai-org--GLM-4-9B-0414", "glm4_greedy_9b", "glm4-9B-0414");
}
