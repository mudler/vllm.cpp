// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED correctness gate); no
// upstream mirror.
//
// THE PAGED-ENGINE Qwen3.5-0.8B (GDN) GREEDY CORRECTNESS GATE. The GDN sibling of
// test_qwen3_paged_engine.cpp: drives the standard prompt battery through the FULL
// PAGED LLMEngine stack (InputProcessor -> Scheduler -> GDN recurrence + paged
// attention on the full-attention layers + Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and checks greedy (temperature-0) decode against the
// pinned vLLM oracle (commit 555967922, runtime 0.23.1rc1.dev1511+g555967922).
//
// ORACLE PROVENANCE — DIFFERENT FROM THE QWEN3-DENSE GATE. There is no dgx/CUDA
// capture for this model: the 0.8B GDN checkpoint was only ever stood up on the
// gfx1100 box (issue #41 lane), which is also the only board that hosts a
// vLLM-ROCm oracle. The base golden pair here is therefore ROCm-captured:
//   greedy_ids.npy  [N,T]   i32  the pinned ROCm oracle's per-prompt greedy
//                                (K=10 per-prompt runs DETERMINISTIC in every
//                                cell — see greedy_dist.npy).
//   our_ids.npy     [N,T]   i32  OUR ROCm engine's greedy (the anchor).
//   neartie_gap_mnats.npy [N,T]  oracle teacher-forced gap (milli-nats) for OUR
//                                token given OUR prefix.
// On kROCM these base files are the gate; on ANY other device the gate SKIPS
// (exit 77) rather than compare another engine's tokens against ROCm-derived
// goldens — fail-safe by device, not luck — with ONE ratified exception: the
// Tenstorrent lane, which gets the Mistral gate's device-golden pair treatment
// (our_ids_tenstorrent.npy + neartie_gap_mnats_tenstorrent.npy; gap teacher-
// forced via scripts/qwen3-neartie-gap-transformers.py — vLLM has no TT
// backend, so `transformers` is the secondary oracle per AGENTS.md's registry).
//
// PROVENANCE OF THE GREEN: the committed our_ids/near-tie pair is the
// FIXED engine's sequence, oracle-re-derived after the AttnQkNormRopeGate
// output-dtype dispatch fix (row/ROCM-GDN-08B-FIX). The PRE-FIX capture
// (13/16 forward-divergent, max first-divergence gap 1.062 nats) is recorded
// as evidence in .agents/specs/rocm-m4-oracle.md and the parity ledger — the
// gate landed green-shaped per review, with the RED capture kept as history
// rather than as committed goldens.
//
// METHODOLOGY — identical to the Qwen3-dense gate (see that file's header and
// [[near-tie-distributional-gate]]): STRICT token-exact is reported, but the
// PASS bar is the near-tie band: given OUR EXACT PREFIX, the oracle's OWN logits
// must place OUR token within kNearTieMnats of the oracle argmax. Strict where
// well-posed, near-tie-robust only where the oracle itself cannot separate the
// tokens.
//
// BACKEND PROOF — GDN op set (not the Qwen3-dense list): the GDN layers dispatch
// kCausalConv1dFwd/Update (prefill/decode conv), kGdnPrefill/kGdnDecode (the
// recurrence), kGdnPostConv, kRmsNormGated, kSigmoidGateBf16,
// kAttnQkNormRopeGate (the GDN preamble), and the full-attention layers dispatch
// kReshapeAndCache + kPagedAttention + kAttnQkNormRope; shared: kEmbedding,
// kMatmulBT, kRmsNorm, kSiluAndMul, kGreedyArgmax. All must show selections>0
// and declines==0 on the running device.
//
// Checkpoint-gated: resolves the HF snapshot under ~/.cache/huggingface/hub/.
// Absent snapshot (CI/CPU) => loud SKIP.
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

#include "hf_snapshot.h"
#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"
#include "vt/op_provider.h"  // the "which backend actually ran" proof
#include "vt/ops.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats — same bar as the Qwen3-dense
// gate (0.5 nats); see that file for the derivation.
constexpr int32_t kNearTieMnats = 500;
// Stable test-only sentinel: distinct from process success and CTest skip 77.
constexpr int kPrerequisiteProbeCompleteExit = 86;

// The standard prompt battery — MUST match scripts/qwen3-oracle-capture.py (and
// qwen3-neartie-gap.py) exactly; those scripts are model-agnostic and take
// --model/--out-dir/--golden-dir.
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

// A gate that cannot run must not report success: exit 77 makes CTest report
// **Skipped**, never a false green (issue #463; SKIP_RETURN_CODE is wired for
// every parity test in tests/CMakeLists.txt).
[[noreturn]] void SkipGate(const char* label, const std::string& why) {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "*** %s: %s\n\n",
               label, why.c_str());
  std::fflush(stderr);
  std::exit(77);
}

enum class GateArtifactState { kReady, kBootstrap };

GateArtifactState RequireGateArtifacts(const fs::path& gdir, const char* label,
                                       bool dump) {
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    SkipGate(label,
             "required greedy_ids.npy is absent. Capture it with the pinned "
             "ROCm oracle using qwen3-oracle-capture.py --per-prompt");
  }

  const bool have_anchor = fs::exists(gdir / "our_ids.npy");
  const bool have_gap = fs::exists(gdir / "neartie_gap_mnats.npy");
  if (have_anchor && have_gap) return GateArtifactState::kReady;
  if (dump) return GateArtifactState::kBootstrap;

  std::string missing;
  if (!have_anchor) missing = "our_ids.npy";
  if (!have_gap) {
    if (!missing.empty()) missing += " and ";
    missing += "neartie_gap_mnats.npy";
  }
  SkipGate(label, "required " + missing +
                      " is absent. Run with VT_DUMP_IDS=1, then run "
                      "qwen3-neartie-gap.py");
}

void RunGate(const std::string& golden_subdir, const char* label) {
  const char* probe_dir = std::getenv("VT_QWEN35_GATE_PREREQ_PROBE_DIR");
  const bool probe = probe_dir != nullptr;
  const std::string snap = probe ? std::string() : parity::Qwen35_08BSnapshot();
  if (!probe && snap.empty()) {
    SkipGate(label, "models--Qwen--Qwen3.5-0.8B snapshot at the pinned revision "
                    "2fc06364 not cached — this gate runs where the ROCm oracle "
                    "was captured (gfx1100)");
  }
  const fs::path gdir = probe ? fs::path(probe_dir)
                              : fs::path(PARITY_GOLDENS_DIR) / golden_subdir;
  const bool dump = !probe && std::getenv("VT_DUMP_IDS") != nullptr;
  const GateArtifactState artifacts =
      RequireGateArtifacts(gdir, label, dump);
  if (probe) {
    std::fprintf(
        stderr,
        "\n*** TEST PROBE COMPLETE (exit %d), production gate NOT RUN ***\n"
        "*** %s: all three required artifact paths are present ***\n\n",
        kPrerequisiteProbeCompleteExit, label);
    std::fflush(stderr);
    std::exit(kPrerequisiteProbeCompleteExit);
  }
  // BOOTSTRAP: with VT_DUMP_IDS set and no gap golden yet, generate + dump OUR
  // token ids (our_ids.i32) so qwen3-neartie-gap.py can build the gap golden.
  if (artifacts == GateArtifactState::kBootstrap) {
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
    SkipGate(label, "bootstrap does not run the correctness gate. Run "
                    "qwen3-neartie-gap.py, then rerun without VT_DUMP_IDS");
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

  // The base golden pair for this model is ROCm-captured (see the file header).
  // The Tenstorrent device lane carries its OWN oracle-backed golden pair
  // (the Mistral gate's treatment); every other device still skips loudly.
  const vt::DeviceType run_dev = loaded->runner().device().type;
  const bool rocm = run_dev == vt::DeviceType::kROCM;
  const bool tenstorrent = run_dev == vt::DeviceType::kTENSTORRENT;
  const bool device_golden = tenstorrent;
  // FAIL SAFE BY DEVICE: this model's only oracle-backed goldens are
  // ROCm-captured (the pinned vLLM-ROCm oracle on gfx1100 — no dgx/CUDA capture
  // exists). On any device without its OWN pair the anchor would be compared
  // against another engine's sequence, so skip loudly rather than misattribute
  // a drift. Tenstorrent is the one lane with a committed device pair.
  if (!rocm && !device_golden) {
    SkipGate(label, "goldens are ROCm-oracle-captured; this run is on device "
                    "type " + std::to_string(static_cast<int>(run_dev)) +
                    " — capture that device's pair first");
  }

  // The GDN op set this model dispatches — all must be proven on the running
  // device (selections > 0, declines == 0; fan-out spike Risk 4).
  const std::vector<vt::OpId> kGdnOps = {
      vt::OpId::kEmbedding,        vt::OpId::kMatmulBT,
      vt::OpId::kRmsNorm,          vt::OpId::kRmsNormGated,
      vt::OpId::kCausalConv1dFwd,  vt::OpId::kCausalConv1dUpdate,
      vt::OpId::kGdnPrefill,       vt::OpId::kGdnDecode,
      vt::OpId::kGdnPostConv,      vt::OpId::kSigmoidGateBf16,
      vt::OpId::kAttnQkNormRopeGate,
      vt::OpId::kReshapeAndCache,  vt::OpId::kPagedAttention,
      vt::OpId::kSiluAndMul,       vt::OpId::kGreedyArgmax};
  if (rocm || device_golden) {
    for (vt::OpId op : kGdnOps) {
      CHECK(vt::OpRegistered(op, run_dev));
      vt::ResetOpProviderStats(op, run_dev);
    }
    vt::EnableOpProviderCallStats(true);
    MESSAGE(label << ": running on device type " << static_cast<int>(run_dev)
            << " (5=ROCM, 6=TENSTORRENT) — gated against "
               << (device_golden ? "this device's OWN oracle-backed golden"
                                 : "the ROCm oracle-backed golden pair"));
  }

  // Device-appropriate anchor + teacher-forced gap goldens. Base = the ROCm
  // pair. Tenstorrent carries its own pair (captured via VT_DUMP_IDS=1 on the
  // P150, then qwen3-neartie-gap-transformers.py teacher-forces `transformers`
  // on that sequence — NOT vLLM, which has no Tenstorrent backend; same
  // secondary-oracle lane and precedent as the Qwen3-0.6B and Mistral-7B TT
  // goldens).
  const char* ids_name = tenstorrent ? "our_ids_tenstorrent.npy" : "our_ids.npy";
  const char* gap_name =
      tenstorrent ? "neartie_gap_mnats_tenstorrent.npy" : "neartie_gap_mnats.npy";
  parity::NpyArray o_dev, gap_dev;  // keep the device arrays alive for the loop
  bool bootstrap_only = false;
  if (device_golden) {
    const bool have_dev = fs::exists(gdir / ids_name) && fs::exists(gdir / gap_name);
    if (!have_dev && dump) {
      // Bootstrap dump path: generate tokens, write raw i32, skip the gate.
      bootstrap_only = true;
      MESSAGE(label << ": BOOTSTRAP dump (device golden absent) for Tenstorrent...");
    } else {
      REQUIRE_MESSAGE(have_dev,
                      label << ": device oracle golden absent (" << ids_name << " / "
                            << gap_name
                            << ") — capture the sequence with VT_DUMP_IDS=1, then "
                               "teacher-force it: qwen3-neartie-gap-transformers.py "
                               "-> device golden pair");
      o_dev = parity::LoadNpy((gdir / ids_name).string());
      gap_dev = parity::LoadNpy((gdir / gap_name).string());
      REQUIRE(o_dev.dtype == "<i4");
      REQUIRE(gap_dev.dtype == "<i4");
      REQUIRE(o_dev.shape.size() == 2);
      REQUIRE(o_dev.shape[0] == N);
      REQUIRE(o_dev.shape[1] == T);
      REQUIRE(gap_dev.shape.size() == 2);
      REQUIRE(gap_dev.shape[0] == N);
      REQUIRE(gap_dev.shape[1] == T);
      od = AsI32(o_dev);
      gapd = AsI32(gap_dev);
    }
  }

  const int32_t* anchor_ids = od;
  const int32_t* gap_ids = gapd;

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
    if (bootstrap_only) continue;  // dump-only path; no anchor/gap yet

    // Anchor: the committed anchor is the exact deterministic sequence OUR ROCm
    // engine produces. Drift is a hard REQUIRE — no cross-device latitude.
    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != anchor_ids[i * T + j]) {
        first_div = static_cast<int>(j);
        break;
      }
    }
    const bool anchor_ok = dump || first_div < 0;
    // A drift is a REGRESSION until proven otherwise: this anchor is the fixed
    // engine's oracle-verified sequence. Refreshing it is the LAST step of a
    // justified re-capture, never the response to a failure.
    REQUIRE_MESSAGE(anchor_ok,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor="
                    << (first_div < 0 ? -1 : anchor_ids[i * T + first_div])
                    << " — REGRESSION SUSPECTED: bisect the engine change first; "
                       "only re-derive goldens (VT_DUMP_IDS=1 + "
                       "qwen3-neartie-gap.py) after the drift is proven to be a "
                       "justified numerical change, never to silence the gate");

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gap_ids[i * T + j];
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
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
              << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gap_ids[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ") \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }

  // Backend proof: token equality alone does not prove which device ran.
  // The bootstrap dump path does not exercise the full op set to a comparison,
  // so its stats prove reachability only (still selections > 0, declines == 0).
  if (device_golden || rocm) vt::EnableOpProviderCallStats(false);
  if (device_golden || rocm) {
    for (vt::OpId op : kGdnOps) {
      const auto st = vt::GetOpProviderStats(op, run_dev);
      CHECK_MESSAGE(st.selections > 0,
                    label << ": op " << static_cast<int>(op)
                          << " was never dispatched on device type "
                          << static_cast<int>(run_dev));
      CHECK_MESSAGE(st.declines == 0,
                    label << ": op " << static_cast<int>(op)
                          << " DECLINED and fell back");
    }
    MESSAGE(label << ": BACKEND PROOF — Qwen3.5 GDN ops on device type "
            << static_cast<int>(run_dev) << " with 0 declines (kPagedAttention "
               "selections="
            << vt::GetOpProviderStats(vt::OpId::kPagedAttention, run_dev).selections
            << ", kGdnDecode selections="
            << vt::GetOpProviderStats(vt::OpId::kGdnDecode, run_dev).selections
            << ", kCausalConv1dUpdate selections="
            << vt::GetOpProviderStats(vt::OpId::kCausalConv1dUpdate, run_dev).selections
            << ")");
  }

  if (dump) {
    const std::string dump_name =
        tenstorrent ? "our_ids_tenstorrent.i32" : "our_ids.i32";
    const std::string path = (gdir / dump_name).string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(our_dump.data(), sizeof(int32_t), our_dump.size(), f);
      std::fclose(f);
      MESSAGE(label << " dumped our token ids -> " << path);
    }
  }
  if (bootstrap_only) {
    // BOOTSTRAP complete — no correctness bar was applied this run (every
    // prompt hit the `continue` above). The summary below would print
    // "0/16 prompts PASS ... 0 forward-divergent" over a vacuous REQUIRE —
    // the Mistral gate returns here for the same reason.
    MESSAGE(label << ": BOOTSTRAP complete -- ids dumped, NO correctness bar "
                     "was applied this run. Teacher-force the dumped sequence, "
                     "commit the golden pair, then re-run without VT_DUMP_IDS.");
    return;
  }
  MESSAGE(label << " correctness gate: " << (strict_exact + neartie_only) << "/" << N
          << " prompts PASS  (STRICT token-exact vs oracle per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail
          << " forward-divergent)");
  REQUIRE(fail == 0);
}

}  // namespace

// Qwen3.5-0.8B (GDN hybrid: linear-attention recurrence + full-attention
// layers) — the first GDN-architecture gate, ROCm-oracle-backed (issue #41 M4).
TEST_CASE("qwen3.5-0.8B GDN paged-engine greedy near-tie correctness gate (ROCm, SACRED)") {
  RunGate("qwen35_greedy_0_8b", "qwen3.5-0.8B");
}
