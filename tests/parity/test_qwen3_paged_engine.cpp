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
//   our_ids.npy            [N,T]   i32  OUR CUDA engine's greedy (anchor for gaps).
//   neartie_gap_mnats.npy  [N,T]   i32  vLLM teacher-forced gap (milli-nats) for
//                                       OUR CUDA token given OUR CUDA prefix.
//
// DEVICE-AWARE goldens (Metal, Qwen3-0.6B). SelectQueue runs Metal on the Apple M4;
// the Metal forward is a DIFFERENT but equally correct bf16 decoder that resolves
// the model's genuine near-ties the other way (e.g. p0 tok5 France 9625 vs Italy
// 15344, top-2 margin 0.003-0.007 nats; vLLM's OWN teacher-forced argmax on the
// identical prefix flips to Italy at gap 0.0000, contradicting its CUDA-capture
// pick). So Metal is gated against ITS OWN oracle-backed golden, captured exactly
// like the base pair but teacher-forcing vLLM on the METAL sequence:
//   our_ids_metal.npy           [N,T] i32  the Metal forward's deterministic greedy.
//   neartie_gap_mnats_metal.npy [N,T] i32  vLLM teacher-forced gap for the Metal
//                                          token given the Metal prefix (max 0.125
//                                          nats over all 60 Metal-vs-CUDA
//                                          divergences => oracle-confirmed near-tie;
//                                          strict token-exactness on 0.6B is
//                                          ill-posed, it is a near-tie model). The
//                                          gate logic is IDENTICAL on every device
//                                          (hard anchor + <=0.5-nat band); only the
//                                          golden pair is device-appropriate.
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
  // (2) the gate is held against Metal's OWN oracle-backed golden (see below), not
  // against CUDA's — the identical anchor+band logic, device-appropriate goldens.
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
            << " (2=METAL) — gated against Metal's OWN oracle-backed golden "
               "(our_ids_metal.npy + neartie_gap_mnats_metal.npy)");
  }

  // Device-appropriate anchor + teacher-forced gap goldens. The base golden
  // (our_ids.npy / neartie_gap_mnats.npy) is the CUDA-captured sequence: vLLM 0.25.0
  // teacher-forced on OUR CUDA tokens. The Metal forward is a DIFFERENT but equally
  // correct bf16 decoder: it resolves the model's genuine near-ties the other way
  // (e.g. p0 tok5 France 9625 vs Italy 15344, a 0.003-0.007-nat top-2 margin —
  // vLLM's OWN teacher-forced argmax on the identical prefix is Italy here at gap
  // 0.0000, contradicting its CUDA-capture pick of France), so it emits its OWN
  // deterministic sequence with its OWN teacher-forced gaps (our_ids_metal.npy /
  // neartie_gap_mnats_metal.npy, produced by qwen3-neartie-gap.py teacher-forcing
  // vLLM on the METAL prefix; every Metal token there is within 0.5 nats of vLLM's
  // argmax, max 0.125). Gate the running device against ITS device's oracle golden.
  const int32_t* anchor_ids = od;   // hard anchor for THIS device
  const int32_t* gap_ids = gapd;    // vLLM teacher-forced gaps for THIS device
  parity::NpyArray o_metal, gap_metal;  // keep the metal arrays alive for the loop
  if (metal) {
    const bool metal_goldens = fs::exists(gdir / "our_ids_metal.npy") &&
                               fs::exists(gdir / "neartie_gap_mnats_metal.npy");
    REQUIRE_MESSAGE(metal_goldens,
                    label << ": Metal oracle golden absent — capture the Metal "
                          "sequence on the M4 (VT_DUMP_IDS bootstrap), then on dgx "
                          "teacher-force vLLM on it: qwen3-neartie-gap.py over the "
                          "Metal our_ids -> *_metal.npy");
    o_metal = parity::LoadNpy((gdir / "our_ids_metal.npy").string());
    gap_metal = parity::LoadNpy((gdir / "neartie_gap_mnats_metal.npy").string());
    REQUIRE(o_metal.dtype == "<i4");
    REQUIRE(gap_metal.dtype == "<i4");
    REQUIRE(o_metal.shape.size() == 2);
    REQUIRE(o_metal.shape[0] == N);
    REQUIRE(o_metal.shape[1] == T);
    REQUIRE(gap_metal.shape.size() == 2);
    REQUIRE(gap_metal.shape[0] == N);
    REQUIRE(gap_metal.shape[1] == T);
    anchor_ids = AsI32(o_metal);
    gap_ids = AsI32(gap_metal);
  }

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

    // Anchor: the committed anchor is the exact deterministic sequence THIS device
    // produces (CUDA base golden on CUDA/CPU; Metal golden on Metal), and the
    // committed gaps are vLLM 0.25.0 teacher-forced on THAT device's own prefix. A
    // drift from the anchor is a hard REQUIRE on EVERY device — this is what gives
    // the gate teeth: a real Metal forward bug flips a token off the anchor and
    // fails HERE, and the near-tie band below then independently proves each token
    // is one vLLM's own logits cannot separate from its argmax. No cross-device
    // latitude is taken: Metal is gated against Metal's OWN oracle-backed golden,
    // never excused against CUDA's.
    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != anchor_ids[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    REQUIRE_MESSAGE(first_div < 0,
                    label << " anchor drift prompt[" << i << "] tok=" << first_div
                    << " engine=" << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                    << " committed anchor=" << (first_div < 0 ? -1 : anchor_ids[i * T + first_div])
                    << (metal ? " — re-capture the Metal golden (our_ids_metal.npy + "
                                "neartie_gap_mnats_metal.npy) via qwen3-neartie-gap.py"
                              : " — re-run qwen3-neartie-gap.py to refresh the gap golden"));

    // Anchor holds: the committed gaps describe our exact sequence, so the near-tie
    // band check below is well-posed on every device. PASS iff every one of our
    // tokens is within kNearTieMnats of vLLM's teacher-forced argmax on our prefix.
    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gap_ids[i * T + j];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = static_cast<int>(j); }
      if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
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
  MESSAGE(label << " correctness gate: " << (strict_exact + neartie_only) << "/" << N
          << " prompts PASS  (STRICT token-exact vs vLLM per-prompt greedy: "
          << strict_exact << "/" << N << "; near-tie-band only: " << neartie_only
          << "/" << N << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt["
          << worst_i << "] tok=" << worst_j << "; " << fail << " forward-divergent"
          << (metal ? "; anchor+gaps = Metal oracle-backed golden (vLLM teacher-"
                      "forced on the Metal prefix)" : "") << ")");
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
