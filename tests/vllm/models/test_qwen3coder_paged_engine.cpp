// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED correctness gate for
// Qwen3-Coder-30B-A3B, `Qwen3MoeForCausalLM`); no upstream mirror.
//
// THE PAGED-ENGINE Qwen3-Coder (full-attention BF16 MoE) GREEDY CORRECTNESS GATE.
// Drives a compact prompt battery through the FULL PAGED LLMEngine stack
// (InputProcessor -> Scheduler -> paged attention + KV-cache growth + MoE forward
// + Sampler -> OutputProcessor) via LoadedEngine::FromModelDir, and checks the
// greedy (temperature-0) decode against the pinned vLLM 0.25.0 oracle. The battery
// is small + decode bounded (16 tokens).
//
// W5 UPDATE: the bf16 MoE now runs the FAST grouped bf16 GEMM path
// (`vt::MoeGroupedGemmBf16`, default ON — VT_MOE_BF16_FAST=0 restores the per-expert
// reference loop for a same-binary A/B). Swapping the per-expert cuBLASLt loop for a
// grouped tensor-core GEMM changes the f32 ACCUMULATION ORDER, which re-resolves
// bf16 near-ties — so the goldens below were re-captured against the fast path. The
// gate got STRICTER, not looser: STRICT token-exact went 4/6 -> 5/6 and the max
// teacher-forced gap 0.125 nats -> 0.0000 nats (our one divergent token IS vLLM's own
// argmax on our prefix). That is expected: vLLM computes these experts with its own
// Triton GROUPED fused_moe GEMM, so a grouped GEMM lands closer to vLLM than a
// per-expert loop does.
//
// GATE (identical ratified methodology to test_qwen3_paged_engine.cpp; see
// [[near-tie-distributional-gate]]). vLLM 0.25.0 greedy on Qwen3-Coder is
// DETERMINISTIC per-prompt (scripts/qwen3coder-oracle-capture.py, K=5 self-
// consistent — 0 multi-valued cells; documented in greedy_dist.npy). But at bf16
// near-ties two independent bf16 decoders (ours vs vLLM) can resolve a next-token
// tie either way, and vLLM's own one-shot PREFILL argmax disagrees with its
// incremental DECODE. So the honest "mirror vLLM" bar is: GIVEN OUR EXACT PREFIX,
// do vLLM's OWN logits place OUR token within kNearTieMnats of vLLM's argmax
// (gap captured by scripts/qwen3coder-neartie-gap.py teacher-forcing vLLM on our
// sequence)? gap 0 => our token IS vLLM's argmax (STRICT); a large gap (or our
// token outside vLLM's top-K) is a REAL forward divergence the gate FAILS on. This
// is strict where the bar is well-posed and near-tie-robust ONLY where vLLM itself
// cannot separate the tokens — it never papers over a real bug.
//
// Goldens (committed under tests/parity/goldens/qwen3coder_greedy/, dgx-captured):
//   greedy_ids.npy         [N,T]   i32  vLLM per-prompt deterministic greedy (run 0).
//   greedy_dist.npy        [N,T,K] i32  K=5 vLLM runs (self-determinism evidence).
//   our_ids.npy            [N,T]   i32  OUR engine's greedy (anchor for the gaps).
//   neartie_gap_mnats.npy  [N,T]   i32  vLLM teacher-forced gap (milli-nats) for OUR
//                                       token given OUR prefix.
//   p{i}_prompt.i32        [Li]    i32  vLLM's tokenization (cross-checked vs ours).
//
// Checkpoint-GATED + dgx-only: resolves the real
// models--Qwen--Qwen3-Coder-30B-A3B-Instruct snapshot under ~/.cache/huggingface/
// hub/. On CPU/CI the snapshot + goldens are absent, so the body emits a loud SKIP
// and returns — compiles + links on CPU, RUNS only on dgx.casa (GB10).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats (identical to the dense gate). vLLM's
// argmax must beat OUR token by <= this, in vLLM's OWN teacher-forced logits given
// OUR prefix, for a divergence to count as a bf16 near-tie rather than a forward
// bug. 500 mnats (0.5 nats) is the Qwen3-established bar.
constexpr int32_t kNearTieMnats = 500;

// The compact prompt battery. Kept in the test (the oracle capture + teacher-forcing
// scripts read the SAME list) so the goldens and the gate never drift. MUST match
// scripts/qwen3coder-oracle-capture.py::PROMPTS exactly.
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "def fibonacci(n):",
      "Once upon a time,",
      "The largest planet in our solar system is",
      "The chemical symbol for gold is",
      "import numpy as np",
  };
  return p;
}

std::string FindQwen3CoderSnapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps =
      fs::path(home) /
      ".cache/huggingface/hub/"
      "models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots";
  std::error_code ec;
  if (!fs::is_directory(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec)) return e.path().string();
  }
  return "";
}

// Greedy (argmax) sampling params — temperature 0 => deterministic.
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

std::vector<int32_t> LoadI32File(const fs::path& p) {
  std::vector<int32_t> out;
  std::FILE* f = std::fopen(p.string().c_str(), "rb");
  if (f == nullptr) return out;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    out.resize(static_cast<size_t>(n) / sizeof(int32_t));
    if (std::fread(out.data(), sizeof(int32_t), out.size(), f) != out.size())
      out.clear();
  }
  std::fclose(f);
  return out;
}

}  // namespace

// Qwen3-Coder-30B-A3B (Qwen3MoeForCausalLM) — the breadth-sweep full-attention MoE
// correctness gate. STRICT where our token IS vLLM's argmax, near-tie-robust only
// where vLLM's own teacher-forced logits (on OUR prefix) cannot separate.
TEST_CASE("qwen3-coder-30B-A3B paged-engine greedy near-tie correctness gate (dgx-only, SACRED)") {
  const std::string snap = FindQwen3CoderSnapshot();
  if (snap.empty()) {
    MESSAGE(
        "Qwen3-Coder-30B-A3B checkpoint absent; skipping (dgx-only) — "
        "Qwen/Qwen3-Coder-30B-A3B-Instruct snapshot not present");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "qwen3coder_greedy";
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE("Qwen3-Coder greedy golden absent; skipping — capture on dgx: "
            "qwen3coder-oracle-capture.py --runs 5");
    return;
  }

  // Document vLLM's self-determinism (from greedy_dist.npy): the count of golden
  // cells vLLM itself resolved multiply over K runs (expected 0 for this A3B MoE).
  if (fs::exists(gdir / "greedy_dist.npy")) {
    const parity::NpyArray d = parity::LoadNpy((gdir / "greedy_dist.npy").string());
    if (d.shape.size() == 3) {
      const int64_t N = d.shape[0], T = d.shape[1], K = d.shape[2];
      const auto* dd = AsI32(d);
      int64_t multi = 0;
      for (int64_t i = 0; i < N; ++i)
        for (int64_t j = 0; j < T; ++j) {
          std::set<int32_t> s;
          for (int64_t k = 0; k < K; ++k) s.insert(dd[(i * T + j) * K + k]);
          if (s.size() > 1) ++multi;
        }
      MESSAGE("qwen3-coder: vLLM self-determinism over K=" << K
              << " runs — multi-valued (prompt,pos) cells = " << multi
              << (multi == 0 ? " (DETERMINISTIC)" : " (NON-DET at those cells)"));
    }
  }

  // BOOTSTRAP: with VT_DUMP_IDS set and no gap golden yet, generate + dump OUR
  // token ids (our_ids.i32) so qwen3coder-neartie-gap.py can build the gap golden.
  if (dump && !have_gap) {
    MESSAGE("qwen3-coder: BOOTSTRAP dump (gap golden absent) via FromModelDir("
            << snap << ")...");
    std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
        vllm::entrypoints::LoadedEngine::FromModelDir(
            snap, vllm::entrypoints::EngineParams{});
    const parity::NpyArray gg =
        parity::LoadNpy((gdir / "greedy_ids.npy").string());
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
    if (f != nullptr) {
      std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f);
      std::fclose(f);
    }
    MESSAGE("qwen3-coder BOOTSTRAP dumped our token ids -> " << path);
    return;
  }
  if (!have_gap) {
    MESSAGE("qwen3-coder gap goldens absent; skipping — run the gate under "
            "VT_DUMP_IDS=1 then qwen3coder-neartie-gap.py");
    return;
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap =
      parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
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

  MESSAGE("qwen3-coder: loading full 30B via FromModelDir(" << snap
          << ") — bf16 MoE (FAST grouped bf16 GEMM, VT_MOE_BF16_FAST default ON) "
             "+ engine stack...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  int strict_exact = 0;   // prompts where our tokens == vLLM greedy exactly
  int neartie_only = 0;   // prompts that pass only via the near-tie band
  int fail = 0;           // prompts with a token beyond the near-tie band
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "coder" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    if (dump) {
      for (int64_t j = 0; j < T; ++j)
        our_dump[static_cast<size_t>(i * T + j)] = got[static_cast<size_t>(j)];
    }

    // Diagnostic: the engine's tokenization must match vLLM's prompt_ids.
    const std::vector<int32_t> want_prompt =
        LoadI32File(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    if (!want_prompt.empty())
      CHECK(out.prompt_token_ids == want_prompt);

    // Anchor: the committed gaps describe OUR engine's exact sequence. A drift
    // here means the engine changed and the gap golden must be re-captured.
    for (int64_t j = 0; j < T; ++j) {
      REQUIRE_MESSAGE(got[static_cast<size_t>(j)] == od[i * T + j],
                      "qwen3-coder anchor drift prompt[" << i << "] tok=" << j
                      << " engine=" << got[static_cast<size_t>(j)]
                      << " committed our_ids=" << od[i * T + j]
                      << " — re-run qwen3coder-neartie-gap.py to refresh the gap golden");
    }

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) {
        worst_gap = mn;
        worst_i = static_cast<int>(i);
        worst_j = static_cast<int>(j);
      }
      if (mn > kNearTieMnats) {
        prompt_ok = false;
        if (first_bad < 0) first_bad = static_cast<int>(j);
      }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE("qwen3-coder FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
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
      MESSAGE("qwen3-coder dumped our token ids -> " << path);
    }
  }
  MESSAGE("qwen3-coder correctness gate: " << (strict_exact + neartie_only) << "/"
          << N << " prompts PASS  (STRICT token-exact vs vLLM per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}
