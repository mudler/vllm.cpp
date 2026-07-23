// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED correctness gate); no
// upstream mirror.
//
// THE PAGED-ENGINE Qwen3 DENSE GREEDY CORRECTNESS GATE. Drives a standard battery
// of prompts through the FULL PAGED LLMEngine stack (InputProcessor -> Scheduler
// -> paged attention + KV-cache growth + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and checks the greedy (temperature-0) decode against
// the pinned vLLM 0.25.0 oracle. Two dense checkpoints, one gate
// (see [[near-tie-distributional-gate]]):
//
//   * Qwen3-0.6B (BF16, 28L) — the first additive-model gate.
//   * Qwen3-4B   (BF16, 36L, GQA 32/8) — the BIGGER-model complete-correctness
//     proof (different config: layers/hidden/GQA-ratio all exercised).
//
// METHODOLOGY. vLLM 0.25.0 greedy is DETERMINISTIC per-prompt (batch=1, the regime
// this gate decodes in) — the run-to-run "non-determinism" seen earlier was a
// BATCHING artifact. So STRICT token-exact vs vLLM's per-prompt greedy is
// well-posed and is reported (strict_exact/N). BUT at bf16 near-ties vLLM is not
// self-consistent: its one-shot PREFILL argmax disagrees with its incremental
// DECODE token (verified — e.g. Qwen3-4B p13 tok1: prefill->13, decode->11), and
// two independent bf16 decoders resolve a near-tie either way. Chasing a specific
// decode kernel's rounding at a tie where vLLM contradicts itself is not "more
// correct". The honest "mirror vLLM" PASS bar is therefore the near-tie band:
// GIVEN OUR EXACT PREFIX, vLLM's OWN logits place OUR token within kNearTieMnats
// of vLLM's argmax (gap captured by scripts/qwen3-neartie-gap.py teacher-forcing
// vLLM on our sequence). A gap of 0 means our token IS vLLM's argmax; a large gap
// (or our token outside vLLM's top-K) is a REAL forward divergence the gate fails
// on. This is strict where our token is exactly vLLM's argmax, and near-tie-robust
// only where vLLM itself cannot separate the tokens — it never weakens the bar
// where the bar is well-posed.
//
// Goldens (committed under tests/parity/goldens/<subdir>/, captured on dgx):
//   greedy_ids.npy         [N,T]   i32  vLLM per-prompt deterministic greedy.
//   greedy_dist.npy        [N,T,K] i32  K per-prompt runs (determinism evidence).
//   our_ids.npy            [N,T]   i32  OUR engine's greedy (anchor for the gaps).
//   neartie_gap_mnats.npy  [N,T]   i32  vLLM teacher-forced gap (milli-nats) for
//                                       OUR token given OUR prefix.
//
// Checkpoint-GATED + dgx-only: resolves the real HF snapshots under
// ~/.cache/huggingface/hub/. On CPU/CI the snapshots + goldens are absent, so each
// case emits a loud SKIP and returns — compiles + links on CPU, RUNS only on dgx.
#include <doctest/doctest.h>

#include <cmath>
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
#include "vt/op_provider.h"  // the "which backend actually ran" proof (Metal, M3b)
#include "vt/ops.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats. vLLM's argmax must beat OUR token
// by <= this, in vLLM's OWN teacher-forced logits given OUR prefix, for a
// divergence to count as a bf16 near-tie rather than a forward bug. 500 mnats
// (0.5 nats, prob ratio ~1.65) sits comfortably above the measured worst
// (Qwen3-4B 0.25 nats) and far below real-divergence scale (many nats / outside
// top-K). Reported max gap makes any creep visible.
constexpr int32_t kNearTieMnats = 500;

// The standard prompt battery. Kept in the test (the oracle capture scripts read
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

// Resolve the newest HF snapshot dir for a "models--<repo>" cache entry.
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

// The near-tie-robust correctness gate (see file header). Loads the vLLM greedy
// golden + OUR-token anchor + vLLM's teacher-forced per-position gaps, drives our
// paged engine, and PASSES when every one of our tokens is within kNearTieMnats of
// vLLM's argmax in vLLM's own logits (strict where our token IS vLLM's argmax).
// Reports the strict token-exact count and the max gap.
void RunGate(const std::string& repo_dir, const std::string& golden_subdir,
             const char* label) {
  const std::string snap = FindSnapshot(repo_dir);
  if (snap.empty()) {
    MESSAGE(label << " checkpoint absent; skipping (dgx-only) — " << repo_dir
            << " snapshot not present");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE(label << " greedy golden absent; skipping — capture on dgx: "
            "qwen3-oracle-capture.py --per-prompt for " << golden_subdir);
    return;
  }
  // BOOTSTRAP: with VT_DUMP_IDS set and no gap golden yet, generate + dump OUR
  // token ids (our_ids.i32) so qwen3-neartie-gap.py can build the gap golden.
  if (dump && !have_gap) {
    MESSAGE(label << ": BOOTSTRAP dump (gap golden absent) via FromModelDir(" << snap << ")...");
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
    MESSAGE(label << " gap goldens absent; skipping — run the gate under "
            "VT_DUMP_IDS=1 then qwen3-neartie-gap.py for " << golden_subdir);
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

  // Optional: refresh OUR engine's exact token ids (raw [N,T] i32) so the teacher-
  // forcing harness (qwen3-neartie-gap.py) can rebuild the gap golden.
  std::vector<int32_t> our_dump;
  if (dump) our_dump.assign(static_cast<size_t>(N * T), -1);

  MESSAGE(label << ": loading via FromModelDir(" << snap << ")...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          snap, vllm::entrypoints::EngineParams{});

  // Which device did SelectQueue actually pick? On dgx this is kCUDA and the gate
  // below is byte-for-byte the historical CUDA gate (the strict our_ids anchor +
  // near-tie band). On the Apple M4 it is kMETAL, and two things change: (1) the
  // Qwen3-dense ops must PROVE they ran on the Metal provider (selections > 0,
  // declines == 0 — `last_selected` alone is insufficient, fan-out spike Risk 4);
  // (2) the our_ids anchor was captured on CUDA, so a Metal divergence AT a
  // committed near-tie (gap <= kNearTieMnats) is the SAME bf16 tie the gate already
  // tolerates across decoders, not an anchor drift — it is reported, not required
  // away, while a divergence at a well-separated position is still a hard fail.
  const vt::DeviceType run_dev = loaded->runner().device().type;
  const bool metal = run_dev == vt::DeviceType::kMETAL;
  // The forward + greedy ops Qwen3-dense dispatches on the DEFAULT
  // (VT_QWEN3_ROPE_CACHE) path. kRopeCosSinCache + kRopeFromCache are the M3b
  // additions (build the per-step cos|sin cache, then apply it); the rest are
  // shared with OPT. kMatmul is excluded deliberately: Qwen3-0.6B ties its
  // lm_head, so every projection is a kMatmulBT.
  const std::vector<vt::OpId> kQwen3Ops = {
      vt::OpId::kEmbedding,      vt::OpId::kMatmulBT,       vt::OpId::kRmsNorm,
      vt::OpId::kRopeCosSinCache, vt::OpId::kRopeFromCache, vt::OpId::kReshapeAndCache,
      vt::OpId::kPagedAttention, vt::OpId::kSiluAndMul,     vt::OpId::kGreedyArgmax};
  if (metal) {
    for (vt::OpId op : kQwen3Ops) {
      CHECK(vt::OpRegistered(op, run_dev));
      vt::ResetOpProviderStats(op, run_dev);
    }
    vt::EnableOpProviderCallStats(true);
    MESSAGE(label << ": running on device type " << static_cast<int>(run_dev)
            << " (2=METAL) — the our_ids anchor is CUDA-captured, so Metal near-tie "
               "divergences are reported, not required away");
  }

  int metal_neartie_div = 0;  // Metal-only: prompts that diverge from CUDA our_ids
                              // at a committed near-tie position
  int strict_exact = 0;   // prompts where our tokens == vLLM greedy exactly
  int neartie_only = 0;   // prompts that pass only via the near-tie band
  int fail = 0;           // prompts with a token beyond the near-tie band
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

    // Anchor: the committed gaps describe OUR engine's exact (CUDA-captured)
    // sequence. On CUDA/CPU a drift is a hard REQUIRE (the engine changed and the
    // gap golden must be re-captured). On Metal the anchor was captured on a
    // DIFFERENT device, so a divergence is classified against the committed gap:
    // at a near-tie it is the expected cross-decoder bf16 behaviour, at a
    // well-separated position it is a real forward bug.
    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    if (!metal) {
      REQUIRE_MESSAGE(first_div < 0,
                      label << " anchor drift prompt[" << i << "] tok=" << first_div
                      << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                      << " committed our_ids=" << (first_div < 0 ? -1 : od[i * T + first_div])
                      << " — re-run qwen3-neartie-gap.py to refresh the gap golden");
    } else if (first_div >= 0) {
      const int32_t mn = gapd[i * T + first_div];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = first_div; }
      // HONEST device-awareness (repaired): the committed gap is vLLM's logit
      // margin of OUR token BELOW vLLM's argmax, teacher-forced on OUR sequence.
      // A gap of 0 means our token IS vLLM's argmax — it records NO runner-up
      // margin, so it carries ZERO near-tie evidence. A divergence at a gap-0
      // position is therefore a hard REQUIRE failure on EVERY device (CUDA and
      // Metal alike): treating gap-0 as "within the near-tie band" would excuse an
      // arbitrary forward error (the whole golden here is gap-0, so the old
      // `mn <= band` relaxation excused EVERYTHING — a gate with no teeth). The
      // device-awareness may ONLY relax a GENUINE near-tie: a strictly-positive
      // committed gap within the band (0 < gap <= kNearTieMnats), where vLLM's own
      // logits place our token below its argmax by a hair and two bf16 decoders may
      // land either side. That is the sole cross-device latitude; gap-0 and
      // above-band are both real divergences.
      const bool genuine_neartie = (mn > 0 && mn <= kNearTieMnats);
      if (genuine_neartie) {
        ++metal_neartie_div;
        MESSAGE(label << " metal near-tie divergence prompt[" << i << "] tok=" << first_div
                << " metal=" << got[static_cast<size_t>(first_div)]
                << " cuda_our_ids=" << od[i * T + first_div]
                << " gap=" << (mn / 1000.0) << " nats (0 < gap <= band) — accepted (vLLM's "
                   "own logits place our token a hair below its argmax; two bf16 "
                   "decoders may land either side)");
        continue;  // committed gaps past first_div describe a DIFFERENT sequence
      }
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_div
              << " metal=" << got[static_cast<size_t>(first_div)]
              << " cuda_our_ids=" << od[i * T + first_div]
              << " vLLM_greedy=" << gd[i * T + first_div]
              << " gap=" << (mn / 1000.0)
              << (mn == 0 ? " nats (gap-0 = our token IS vLLM's argmax; a divergence "
                            "here is a hard REQUIRE on EVERY device, NOT a near-tie)"
                          : " nats (> band = a real forward divergence)"));
      // Hard failure on gap-0 or above-band, identical bar on every device.
      REQUIRE_MESSAGE(genuine_neartie,
                      label << " gap-0/above-band divergence prompt[" << i << "] tok="
                      << first_div << " is a forward error, not a tolerated near-tie");
      continue;  // committed gaps past first_div describe a DIFFERENT sequence
    }

    // No divergence from the anchor: the committed gaps describe our sequence, so
    // the near-tie band check below is well-posed on every device.
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

  // The backend proof, now that work has been dispatched. Metal computes the same
  // tokens as any other backend, so the token comparison does NOT say the Metal
  // GPU ran; the provider stats do. Every Qwen3-dense op must have been SELECTED
  // on the running device and must NEVER have DECLINED (a decline is a silent
  // forward down the provider stack — fan-out spike Risk 4).
  if (metal) {
    vt::EnableOpProviderCallStats(false);
    for (vt::OpId op : kQwen3Ops) {
      const auto st = vt::GetOpProviderStats(op, run_dev);
      CHECK_MESSAGE(st.selections > 0,
                    label << ": op " << static_cast<int>(op)
                          << " was never dispatched on the Metal device — the token "
                             "result cannot be attributed to this backend");
      CHECK_MESSAGE(st.declines == 0,
                    label << ": op " << static_cast<int>(op)
                          << " DECLINED on Metal and fell back");
    }
    MESSAGE(label << ": BACKEND PROOF — all " << kQwen3Ops.size()
            << " Qwen3-dense ops dispatched on Metal with 0 declines (kRopeFromCache "
               "selections=" << vt::GetOpProviderStats(vt::OpId::kRopeFromCache, run_dev).selections
            << ", kPagedAttention selections="
            << vt::GetOpProviderStats(vt::OpId::kPagedAttention, run_dev).selections << ")");
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
  MESSAGE(label << " correctness gate: "
          << (strict_exact + neartie_only + metal_neartie_div) << "/" << N
          << " prompts PASS  (STRICT token-exact vs vLLM per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; " << (metal ? "metal cross-device near-tie: " : "")
          << (metal ? std::to_string(metal_neartie_div) : std::string())
          << (metal ? "/" + std::to_string(N) + "; " : "; ") << "max gap "
          << (worst_gap / 1000.0) << " nats @ prompt[" << worst_i << "] tok=" << worst_j
          << "; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}

}  // namespace

// Qwen3-0.6B (dense) — first additive-model gate.
TEST_CASE("qwen3-0.6B dense paged-engine greedy near-tie correctness gate (dgx-only, SACRED)") {
  RunGate("models--Qwen--Qwen3-0.6B", "qwen3_greedy_0_6b", "qwen3-0.6B");
}

// Qwen3-4B (dense) — the BIGGER-model complete-correctness proof (36 layers,
// GQA 32/8, hidden 2560; a different config than 0.6B, all exercised by the same
// forward code).
TEST_CASE("qwen3-4B dense paged-engine greedy near-tie correctness gate (dgx-only, SACRED)") {
  RunGate("models--Qwen--Qwen3-4B", "qwen3_greedy_4b", "qwen3-4B");
}
