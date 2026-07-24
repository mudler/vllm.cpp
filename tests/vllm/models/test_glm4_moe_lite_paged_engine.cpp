// vllm.cpp original (GLM/DSA G1 — THE SACRED correctness gate for GLM-4.7-Flash,
// `Glm4MoeLiteForCausalLM`, the SECOND MLA model in this tree and the first to
// exercise the q_lora query branch AND the `noaux_tc` grouped router e2e). Sibling
// of test_deepseek_v2_paged_engine.cpp — same MLA + DeepSeek-MoE stack, GLM config
// values.
//
// Drives a compact prompt battery through the FULL PAGED LLMEngine stack
// (InputProcessor -> Scheduler -> decode-first reorder -> MLA cache write + MLA
// decode/prefill + DeepSeek-MoE forward -> Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and checks the greedy (temperature-0) decode against
// the pinned vLLM 0.25.0 oracle.
//
// ─── THE BAR, and how it was ARRIVED AT (measured, never assumed) ─────────────
// STEP 1 — is the oracle deterministic in the gate regime (batch=1)? The capture
// (scripts/glm4-moe-lite-oracle-capture.py --runs 5, per-prompt batch=1) writes
// greedy_dist.npy [N,T,K]; a ZERO multi-valued-cell count licenses the STRICT
// token-exact bar, re-asserted below. A 31.2B MoE is well above the small-dense
// near-tie regime, so STRICT is the expectation.
// STEP 2/3 — where our greedy diverges from a deterministic oracle, the ratified
// TEACHER-FORCING procedure (scripts/glm4-moe-lite-neartie-gap.py) supplies the
// per-position nats gap (vLLM's OWN argmax vs OUR token, GIVEN OUR PREFIX),
// committed as neartie_gap_mnats.npy + anchored by our_ids.npy. A divergence
// counts as a bf16 near-tie ONLY where that gap is within kNearTieMnats AND our
// token is inside vLLM's top-K; anything beyond FAILS as a real forward bug.
//
// Phase 0 asserts the engine allocated an MLA cache (page = block_size * 576 * 2B,
// NO factor 2). The TOKENIZATION golden (p{i}_prompt.i32) is a REQUIRED check —
// a token comparison against an oracle fed a different prompt is meaningless (the
// OPT BOS lesson).
//
// Goldens (tests/parity/goldens/glm4_moe_lite_greedy/, dgx-captured):
//   greedy_ids.npy / greedy_dist.npy / our_ids.npy / neartie_gap_mnats.npy /
//   p{i}_prompt.i32 — same layout as deepseek_v2_greedy.
//
// Checkpoint-GATED + dgx-only: resolves models--zai-org--GLM-4.7-Flash under
// ~/.cache/huggingface/hub/. On CPU/CI the snapshot + goldens are absent, so the
// body emits a loud SKIP and returns.
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
#include "vllm/model_executor/models/deepseek_v2.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in MILLI-nats — the ratified 0.5-nat band,
// unchanged from the DeepSeek-V2 / Qwen3-dense / Qwen3-Coder gates.
constexpr int32_t kNearTieMnats = 500;

// MUST match scripts/glm4-moe-lite-oracle-capture.py::PROMPTS and
// scripts/glm4-moe-lite-neartie-gap.py::PROMPTS exactly (goldens + gate never drift).
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "def fibonacci(n):",
      "In a shocking finding, scientists discovered a herd of unicorns living in",
      "Q: What is 17 * 23?\nA:",
      "The three laws of robotics are",
      "Once upon a time, in a land far away,",
      "The chemical symbol for gold is",
      "To be or not to be, that is",
  };
  return p;
}

std::string FindGlm47Snapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--zai-org--GLM-4.7-Flash/snapshots";
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

// GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`) — the GLM/DSA G1 SACRED gate.
TEST_CASE("glm4-moe-lite paged-engine greedy token-exact gate (dgx-only, SACRED)") {
  const std::string snap = FindGlm47Snapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP (dgx-only): GLM-4.7-Flash checkpoint absent — "
        "zai-org/GLM-4.7-Flash snapshot not present under "
        "~/.cache/huggingface/hub/");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "glm4_moe_lite_greedy";
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE("SKIP: GLM-4.7-Flash greedy golden absent — capture on dgx: "
            "scripts/glm4-moe-lite-oracle-capture.py --runs 5");
    return;
  }

  // ---- GATE SELECTION: re-assert vLLM's own self-determinism -----------------
  int64_t multi_cells = -1;
  if (fs::exists(gdir / "greedy_dist.npy")) {
    const parity::NpyArray d = parity::LoadNpy((gdir / "greedy_dist.npy").string());
    REQUIRE(d.shape.size() == 3);
    const int64_t DN = d.shape[0], DT = d.shape[1], DK = d.shape[2];
    const auto* dd = AsI32(d);
    multi_cells = 0;
    for (int64_t i = 0; i < DN; ++i)
      for (int64_t j = 0; j < DT; ++j) {
        std::set<int32_t> s;
        for (int64_t k = 0; k < DK; ++k) s.insert(dd[(i * DT + j) * DK + k]);
        if (s.size() > 1) ++multi_cells;
      }
    const std::string verdict =
        multi_cells == 0
            ? std::string(" (DETERMINISTIC -> the STRICT bar is well-posed)")
            : std::string(" (NON-DET: RE-DERIVE the bar via teacher-forcing, do "
                          "not loosen it)");
    MESSAGE("glm4-moe-lite: vLLM self-determinism over K=" << DK
            << " runs (batch=1 capture) — multi-valued (prompt,pos) cells = "
            << multi_cells << verdict);
    CHECK(multi_cells == 0);
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  const int32_t* gd = AsI32(g);

  // BOOTSTRAP: the teacher-forced goldens describe OUR engine's exact sequence,
  // so they cannot exist until our engine has run once. When they are absent, a
  // run with VT_GLM_DUMP_IDS=1 loads the engine, generates the battery, writes
  // our_ids.i32 and returns (DUMP MODE) — then scripts/glm4-moe-lite-neartie-gap.py
  // produces our_ids.npy + neartie_gap_mnats.npy, and a normal run does the
  // STRICT/near-tie comparison below.
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  const bool dump_mode = std::getenv("VT_GLM_DUMP_IDS") != nullptr;
  if (!have_gap && !dump_mode) {
    MESSAGE("SKIP: GLM-4.7-Flash teacher-forced goldens absent — re-run with "
            "VT_GLM_DUMP_IDS=1 to write our_ids.i32, then "
            "scripts/glm4-moe-lite-neartie-gap.py");
    return;
  }

  MESSAGE("glm4-moe-lite: loading via FromModelDir(" << snap
          << ") — MLA (576-wide latent, q_lora branch, QK 256 / V 256 prefill, "
             "MQA-576 decode) + DeepSeek-MoE (64 routed + 1 shared expert, "
             "noaux_tc sigmoid router, routed_scaling 1.8)...");
  vllm::entrypoints::EngineParams params;
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(snap, params);

  // ---- PHASE 0: the engine really allocated an MLA cache -------------------
  const int64_t kMlaHeadSize = 512 + 64;  // kv_lora_rank + qk_rope_head_dim
  const int64_t want_page =
      static_cast<int64_t>(params.block_size) * kMlaHeadSize * 2;  // bf16
  MESSAGE("glm4-moe-lite: runner fa_page_size_bytes = "
          << loaded->runner().fa_page_size_bytes() << " (MLA expects " << want_page
          << " = block " << params.block_size << " x " << kMlaHeadSize
          << " x 2B, NO factor 2)");
  REQUIRE(loaded->runner().fa_page_size_bytes() == want_page);

  // Teacher-forced goldens (only in compare mode; absent during the bootstrap dump).
  parity::NpyArray o, gap;
  const int32_t* od = nullptr;
  const int32_t* gapd = nullptr;
  if (have_gap) {
    o = parity::LoadNpy((gdir / "our_ids.npy").string());
    gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
    REQUIRE(o.dtype == "<i4");
    REQUIRE(gap.dtype == "<i4");
    REQUIRE(o.shape.size() == 2);
    REQUIRE(o.shape[0] == N);
    REQUIRE(o.shape[1] == T);
    REQUIRE(gap.shape.size() == 2);
    REQUIRE(gap.shape[0] == N);
    REQUIRE(gap.shape[1] == T);
    od = AsI32(o);
    gapd = AsI32(gap);
  }

  // ---- DUMP MODE: generate the battery, write our_ids.i32, return -----------
  if (!have_gap) {
    std::vector<int32_t> flat(static_cast<size_t>(N * T), 0);
    for (int64_t i = 0; i < N; ++i) {
      const vllm::RequestOutput out = loaded->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
          "glm47-dump-" + std::to_string(i));
      REQUIRE(out.finished);
      REQUIRE(out.outputs.size() == 1);
      const std::vector<int32_t>& got = out.outputs[0].token_ids;
      REQUIRE(static_cast<int64_t>(got.size()) == T);
      for (int64_t j = 0; j < T; ++j)
        flat[static_cast<size_t>(i * T + j)] = got[static_cast<size_t>(j)];
    }
    const fs::path outp = gdir / "our_ids.i32";
    std::FILE* f = std::fopen(outp.string().c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(flat.data(), sizeof(int32_t), flat.size(), f);
    std::fclose(f);
    MESSAGE("glm4-moe-lite DUMP MODE: wrote " << outp.string() << " [" << N << "," << T
            << "] — now run scripts/glm4-moe-lite-neartie-gap.py to build the gap "
               "goldens, then re-run this gate normally");
    return;
  }

  // ---- PHASE 1: THE CORRECTNESS GATE (batch=1, the oracle's own regime) ----
  vllm::ResetMlaBatchSplitStats();
  int strict_exact = 0;
  int neartie_only = 0;
  int fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  int64_t exact_tokens = 0;
  int64_t total_tokens = 0;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "glm47-" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);

    // TOKENIZATION GOLDEN — REQUIRED (the OPT BOS lesson).
    const std::vector<int32_t> want_prompt =
        LoadI32File(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    REQUIRE_MESSAGE(!want_prompt.empty(),
                    "glm4-moe-lite: tokenization golden p" << i
                    << "_prompt.i32 missing — re-capture.");
    REQUIRE_MESSAGE(out.prompt_token_ids == want_prompt,
                    "glm4-moe-lite TOKENIZATION MISMATCH prompt[" << i << "] ours["
                    << out.prompt_token_ids.size() << "] vs vLLM["
                    << want_prompt.size() << "] — the token comparison below would "
                       "be meaningless");

    // ANCHOR: the committed gaps describe OUR engine's exact sequence.
    for (int64_t j = 0; j < T; ++j) {
      REQUIRE_MESSAGE(got[static_cast<size_t>(j)] == od[i * T + j],
                      "glm4-moe-lite anchor drift prompt[" << i << "] tok=" << j
                      << " engine=" << got[static_cast<size_t>(j)]
                      << " committed our_ids=" << od[i * T + j]
                      << " — re-run scripts/glm4-moe-lite-neartie-gap.py");
    }

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    int first_diff = -1;
    for (int64_t j = 0; j < T; ++j) {
      ++total_tokens;
      if (got[static_cast<size_t>(j)] == gd[i * T + j]) {
        ++exact_tokens;
      } else {
        exact = false;
        if (first_diff < 0) first_diff = static_cast<int>(j);
      }
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
      // A REAL forward divergence. The GLM-specific suspects, in order: the
      // q_lora query branch (fused_qkv_a_proj / q_a_layernorm / q_b_proj), the
      // sigmoid noaux_tc router with e_score_correction_bias and
      // routed_scaling_factor=1.8, the head_dim-256 MLA prefill, and the
      // absorbed-decode/materialized-prefill split.
      ++fail;
      MESSAGE("glm4-moe-lite FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ")  \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
      MESSAGE("glm4-moe-lite NEAR-TIE prompt[" << i << "] first differing tok="
              << first_diff << " our=" << got[static_cast<size_t>(first_diff)]
              << " vLLM_greedy=" << gd[i * T + first_diff] << " gap="
              << (gapd[i * T + first_diff] / 1000.0)
              << " nats (vLLM's OWN argmax on OUR prefix)  \"" << out.outputs[0].text
              << "\"");
    }
    CHECK_MESSAGE(prompt_ok,
                  "glm4-moe-lite prompt[" << i << "] diverges from vLLM 0.25.0 "
                  "BEYOND the near-tie band — a real forward difference");
  }

  const vllm::MlaBatchSplitStats serial_stats = vllm::GetMlaBatchSplitStats();
  MESSAGE("glm4-moe-lite correctness gate: " << (strict_exact + neartie_only) << "/"
          << N << " prompts PASS  (STRICT token-exact " << strict_exact << "/" << N
          << "; near-tie-band only " << neartie_only << "/" << N
          << "; tokens strictly exact " << exact_tokens << "/" << total_tokens
          << "; max teacher-forced gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail
          << " forward-divergent; vLLM self-determinism: " << multi_cells
          << " multi-valued cells)");
  // PROOF THE PATH RAN: the strict phase drove real MLA forwards — N prefill steps
  // plus N*(T-1) decode steps through the MLA block.
  MESSAGE("glm4-moe-lite MLA split stats (phase 1, batch=1): steps="
          << serial_stats.steps
          << " prefill_only=" << serial_stats.prefill_only_steps
          << " decode_only=" << serial_stats.decode_only_steps
          << " mixed=" << serial_stats.mixed_steps
          << " decode_tokens=" << serial_stats.total_decode_tokens
          << " prefill_tokens=" << serial_stats.total_prefill_tokens);
  REQUIRE(serial_stats.steps >= N * T);
  REQUIRE(serial_stats.prefill_only_steps >= N);
  REQUIRE(serial_stats.total_decode_tokens >= N * (T - 1));
}
