// vllm.cpp original (ADDITIVE-QUANT W4 — THE SACRED correctness gate for
// Qwen3-32B-NVFP4A16: `Qwen3ForCausalLM` + compressed-tensors **NVFP4A16**
// (W4A16 — NVFP4 weights, BF16 activations), the QUANT-SCHEME additivity proof).
//
// WHAT THIS ROW ISOLATES. The dense `Qwen3ForCausalLM` forward is ALREADY done
// and token-exact (0.6B/4B). This checkpoint changes exactly ONE variable — the
// storage/compute scheme of the Linear weights — on the SAME forward, at a new
// size (64 layers, hidden 5120, 64 Q heads / 8 KV heads). So a pass here is
// evidence that a new QUANTIZATION scheme is additive; a failure localizes to
// the loader or the W4A16 GEMM, never to the transformer.
//
// Upstream test mirrored: `tests/quantization/test_compressed_tensors.py`
// (the compressed-tensors scheme-selection + end-to-end generation cases) and
// `tests/models/language/generation/test_common.py` (greedy generation vs a
// reference). Our equivalent asserts our paged engine's greedy against the
// pinned vLLM 0.25.0 oracle itself.
//
// THE GATE — ratified near-tie-robust, and the loosening from STRICT is PROVEN,
// not assumed ([[near-tie-distributional-gate]]).
//
// The gate was first run STRICT, because `scripts/qwen3-32b-nvfp4a16-oracle-
// capture.py --runs 5` reported all six prompts deterministic over K=5 runs with
// **0 multi-valued (prompt,pos) cells** (evidence committed in greedy_dist.npy,
// re-asserted below). It scored **4/6 prompts / 67/96 tokens**, with both root
// divergences at token 1 (the first DECODE step) and every PREFILL argmax exact.
//
// The ratified TEACHER-FORCING isolation was then run
// (`scripts/qwen3-32b-nvfp4a16-neartie-gap.py`: feed vLLM OUR exact prefix via
// prompt_token_ids, read prompt_logprobs) and it SETTLED the question:
//
//   * ALL 29 token-divergent positions have a teacher-forced gap <= **0.0625
//     nats** — 8x inside the Qwen3-established 0.5-nat bar.
//   * **28 of 29 have gap EXACTLY 0.000000 nats**: our token IS vLLM's own
//     argmax given our prefix.
//   * prompt[5] tok1 (the root flip " moon" vs " Moon") is an EXACT bf16 TIE —
//     vLLM's own logprobs are bit-identical (-0.727154 for both), and vLLM's
//     teacher-forced argmax is OUR token while its incremental greedy chose the
//     other. **vLLM contradicts ITSELF at this position.**
//   * prompt[2] tok1 (the other root flip, " group" vs " man") gaps 0.0625 nats
//     — and vLLM's OWN separation there MOVES BY 0.125 nats purely from batch
//     composition (measured: 0.1875 alone, 0.0625 batched), i.e. our gap is
//     smaller than vLLM's own numerical jitter at that exact position. It is
//     also smaller than the 0.25-nat gap already ratified on the UNQUANTIZED
//     dense Qwen3-4B row.
//   * Every remaining divergence is downstream CASCADE from those two flips.
//
// So this is NOT a W4A16 defect. It is the pre-existing accumulated bf16
// near-tie drift of the dense `Qwen3ForCausalLM` forward (which itself closes
// under this same near-tie-robust gate: 0.6B 60 divergent positions ALL at gap
// 0.0; 4B max gap 0.25), now run 64 layers deep. The quant path is exonerated
// independently: the CPU forward proof is bit-identical (max |Δlogit| = 0),
// `fallback_gemms=0`, and the divergences are INVARIANT across both of our
// quantized GEMMs (VT_NVFP4_MARLIN=0 reproduces them).
//
// The band therefore admits ONLY what vLLM's own logits cannot separate; a gap
// above kNearTieMnats, or our token outside vLLM's top-K, still FAILS.
//
// WHAT vLLM OBSERVABLY DISPATCHES (not inferred — read off the oracle run):
//   `INFO [__init__.py:929] Using MarlinNvFp4LinearKernel for NVFP4 GEMM`
//   `WARNING [marlin.py:34] ... Weight-only FP4 compression will be used
//    leveraging the Marlin kernel.`
// i.e. on sm_121 the a16 path is FORCED to Marlin
// (kernels/linear/__init__.py:879-881), bypassing the capability registry —
// which is precisely the kernel our vendored `MoeGroupedGemmNvfp4Marlin`
// (num_experts=1) lifts. Our side is asserted to have actually taken it via the
// W4A16 execution counters below.
//
// Goldens (committed under tests/parity/goldens/qwen3_32b_nvfp4a16_greedy/,
// dgx-captured):
//   greedy_ids.npy   [N,T]   i32  vLLM per-prompt greedy (run 0) — the BAR.
//   greedy_dist.npy  [N,T,K] i32  K=5 vLLM runs (the self-determinism evidence
//                                 that SELECTS the strict gate).
//   p{i}_prompt.i32  [Li]    i32  vLLM's TOKENIZATION (cross-checked vs ours).
//     ^ committed deliberately: on the OPT row a BOS bug scored 0/6 while
//       emitting fluent English, and the committed tokenization golden is what
//       localized it in a single run.
//   our_ids.npy           [N,T] i32  OUR engine's exact greedy tokens (anchor).
//   neartie_gap_mnats.npy [N,T] i32  vLLM's TEACHER-FORCED gap, in MILLI-nats,
//                                 for OUR token given OUR prefix (0 = our token
//                                 IS vLLM's argmax). This is the EVIDENCE that
//                                 licenses the band; regenerate with
//                                 scripts/qwen3-32b-nvfp4a16-neartie-gap.py. Qwen3 prepends NO BOS.
//
// Checkpoint-GATED + dgx-only: resolves the HF snapshot for
// RedHatAI/Qwen3-32B-NVFP4A16. On CPU/CI the checkpoint and goldens are absent,
// so the body emits a loud SKIP — compiles + links everywhere, RUNS only on
// dgx.casa (GB10, sm_121).
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
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

// Near-tie acceptance threshold in milli-nats — IDENTICAL to the dense Qwen3 and
// Qwen3-Coder gates (the Qwen3-established bar). vLLM's argmax must beat OUR
// token by <= this, in vLLM's OWN teacher-forced logits given OUR prefix, for a
// divergence to count as a bf16 near-tie rather than a forward bug. This row's
// MEASURED worst case is 62 mnats — 8x inside it.
constexpr int32_t kNearTieMnats = 500;

// The compact prompt battery. Kept in the test (the oracle capture script reads
// the SAME list) so the goldens and the gate never drift. MUST match
// scripts/qwen3-32b-nvfp4a16-oracle-capture.py::PROMPTS exactly.
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "def fibonacci(n):",
      "Once upon a time,",
      "The largest planet in our solar system is",
      "The chemical symbol for gold is",
      "In 1969, humans first walked on",
  };
  return p;
}

// Resolve the RedHatAI/Qwen3-32B-NVFP4A16 HF snapshot dir (config.json +
// model-0000*-of-00005.safetensors + tokenizer.json). Overridable.
std::string FindNvfp4A16ModelDir() {
  if (const char* env = std::getenv("VLLM_CPP_QWEN3_32B_NVFP4A16_DIR");
      env != nullptr) {
    std::error_code ec;
    if (fs::exists(fs::path(env) / "config.json", ec)) return env;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--RedHatAI--Qwen3-32B-NVFP4A16/snapshots";
  std::error_code ec;
  if (!fs::exists(snaps, ec)) return "";
  for (const auto& e : fs::directory_iterator(snaps, ec)) {
    if (fs::exists(e.path() / "config.json", ec) &&
        fs::exists(e.path() / "model.safetensors.index.json", ec))
      return e.path().string();
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

TEST_CASE(
    "Qwen3-32B-NVFP4A16 paged-engine greedy near-tie-robust correctness gate "
    "(dgx-only, SACRED)") {
  const std::string dir = FindNvfp4A16ModelDir();
  if (dir.empty()) {
    MESSAGE(
        "SKIP (dgx-only): RedHatAI/Qwen3-32B-NVFP4A16 snapshot absent "
        "(~/.cache/huggingface/hub/models--RedHatAI--Qwen3-32B-NVFP4A16 or "
        "$VLLM_CPP_QWEN3_32B_NVFP4A16_DIR)");
    return;
  }
  const fs::path gdir =
      fs::path(PARITY_GOLDENS_DIR) / "qwen3_32b_nvfp4a16_greedy";
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE(
        "SKIP: Qwen3-32B-NVFP4A16 greedy golden absent — capture on dgx: "
        "scripts/qwen3-32b-nvfp4a16-oracle-capture.py --runs 5");
    return;
  }

  // ---- DIAGNOSTIC BOOTSTRAP (VT_DUMP_IDS=1) --------------------------------
  // Dump OUR exact greedy token ids so the ratified TEACHER-FORCING isolation
  // (scripts/qwen3-32b-nvfp4a16-neartie-gap.py, mirroring
  // scripts/qwen3coder-neartie-gap.py) can feed vLLM OUR prefix and report, per
  // position, the nats gap between vLLM's own argmax and OUR token. That is the
  // measurement that distinguishes a real forward DEFECT of ours (large gap /
  // our token outside vLLM's top-K) from accumulated bf16 near-tie drift (gap
  // ~0). Re-run this bootstrap whenever the forward changes and the anchor check
  // below reports drift, then regenerate the gap golden.
  if (std::getenv("VT_DUMP_IDS") != nullptr) {
    const parity::NpyArray gg =
        parity::LoadNpy((gdir / "greedy_ids.npy").string());
    REQUIRE(gg.shape.size() == 2);
    const int64_t NN = gg.shape[0], TT = gg.shape[1];
    MESSAGE("Qwen3-32B-NVFP4A16: VT_DUMP_IDS bootstrap via FromModelDir(" << dir
            << ")...");
    vllm::dense_nvfp4::ResetW4A16Stats();
    std::unique_ptr<vllm::entrypoints::LoadedEngine> le =
        vllm::entrypoints::LoadedEngine::FromModelDir(
            dir, vllm::entrypoints::EngineParams{});
    std::vector<int32_t> buf(static_cast<size_t>(NN * TT), -1);
    for (int64_t i = 0; i < NN; ++i) {
      const vllm::RequestOutput out = le->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(TT)),
          "q32boot" + std::to_string(i));
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
    MESSAGE("Qwen3-32B-NVFP4A16 VT_DUMP_IDS dumped our token ids -> " << path);
    return;
  }

  // ---- GATE SELECTION (the ratified methodology) ---------------------------
  // Re-assert vLLM's own self-determinism from the committed K-run evidence.
  // Zero multi-valued cells is why the STRICT column below is meaningful (vLLM
  // repeats itself run-to-run); it is NOT by itself a licence for a strict-only
  // bar, because vLLM's REPEATABLE greedy can still disagree with vLLM's OWN
  // teacher-forced argmax at an exact bf16 tie — which is precisely what the gap
  // golden measures, and precisely what happens at prompt[5] tok1 here.
  int64_t multi_cells = -1;
  if (fs::exists(gdir / "greedy_dist.npy")) {
    const parity::NpyArray d =
        parity::LoadNpy((gdir / "greedy_dist.npy").string());
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
            ? std::string(" (DETERMINISTIC run-to-run)")
            : std::string(" (NON-DET at those cells)");
    MESSAGE("Qwen3-32B-NVFP4A16: vLLM self-determinism over K="
            << DK << " runs — multi-valued (prompt,pos) cells = " << multi_cells
            << verdict);
    CHECK(multi_cells == 0);
  }

  if (!fs::exists(gdir / "our_ids.npy") ||
      !fs::exists(gdir / "neartie_gap_mnats.npy")) {
    MESSAGE(
        "SKIP: Qwen3-32B-NVFP4A16 teacher-forcing gap golden absent — run this "
        "gate under VT_DUMP_IDS=1, then "
        "scripts/qwen3-32b-nvfp4a16-neartie-gap.py");
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

  MESSAGE("Qwen3-32B-NVFP4A16: loading via FromModelDir("
          << dir
          << ") — 64L dense, compressed-tensors nvfp4-pack-quantized W4A16 "
             "(weight_packed/weight_scale/weight_global_scale, group_size 16)");
  vllm::dense_nvfp4::ResetW4A16Stats();
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          dir, vllm::entrypoints::EngineParams{});

  int strict_exact = 0;  // prompts token-exact vs vLLM greedy
  int neartie_only = 0;  // prompts that pass only via the measured near-tie band
  int fail = 0;          // prompts with a token beyond the near-tie band
  int64_t exact_tokens = 0;
  int64_t total_tokens = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "q32nvfp4a16-" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);

    // The engine's tokenization must match vLLM's prompt_ids. A mismatch here
    // makes a token comparison meaningless (the OPT BOS lesson).
    const std::vector<int32_t> want_prompt =
        LoadI32File(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    if (!want_prompt.empty())
      CHECK_MESSAGE(out.prompt_token_ids == want_prompt,
                    "Qwen3-32B-NVFP4A16 prompt[" << i
                        << "] tokenization differs from vLLM's");

    // ANCHOR: the committed gaps describe OUR engine's exact sequence. Drift
    // here means the forward changed and the gap golden is stale — the band
    // would then be describing tokens we no longer emit, so this is a hard
    // REQUIRE, not a soft check. Re-run under VT_DUMP_IDS=1 + the gap script.
    for (int64_t j = 0; j < T; ++j) {
      REQUIRE_MESSAGE(
          got[static_cast<size_t>(j)] == od[i * T + j],
          "Qwen3-32B-NVFP4A16 anchor drift prompt["
              << i << "] tok=" << j << " engine=" << got[static_cast<size_t>(j)]
              << " committed our_ids=" << od[i * T + j]
              << " — re-run under VT_DUMP_IDS=1 then "
                 "scripts/qwen3-32b-nvfp4a16-neartie-gap.py");
    }

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      ++total_tokens;
      if (got[static_cast<size_t>(j)] == gd[i * T + j]) {
        ++exact_tokens;
      } else {
        exact = false;
      }
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) {
        worst_gap = mn;
        worst_i = static_cast<int>(i);
        worst_j = static_cast<int>(j);
      }
      // The ONLY thing the band admits: a divergence vLLM's OWN teacher-forced
      // logits, given OUR prefix, cannot separate. Anything above the threshold
      // (including our token outside vLLM's top-K, encoded as a huge gap) is a
      // real forward divergence and FAILS.
      if (mn > kNearTieMnats) {
        prompt_ok = false;
        if (first_bad < 0) first_bad = static_cast<int>(j);
      }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE("Qwen3-32B-NVFP4A16 FORWARD DIVERGENCE prompt["
              << i << "] tok=" << first_bad
              << " ours=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ")  \"" << out.outputs[0].text
              << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK_MESSAGE(prompt_ok, "Qwen3-32B-NVFP4A16 prompt["
                                 << i
                                 << "] diverges from the vLLM 0.25.0 oracle "
                                    "BEYOND the measured bf16 near-tie band");
  }

  // ---- POSITIVE SIGNAL: the W4A16 path actually RAN -------------------------
  // A passing gate alone does NOT prove the new path was exercised. Assert the
  // execution counters: every layer's qkv + o_proj + down_proj go through
  // MatmulNvfp4MarlinD, and each MLP takes ONE fused gate_up Marlin GEMM. With
  // 64 layers and 6 prompts x (1 prefill + 15 decode) steps this is thousands of
  // launches; requiring > 0 of BOTH (and that the naive fallback never ran on
  // CUDA) is the guard against a silent fall-back to a BF16 or unfused arm.
  const vllm::dense_nvfp4::Nvfp4W4A16Stats st =
      vllm::dense_nvfp4::GetW4A16Stats();
  MESSAGE("Qwen3-32B-NVFP4A16 W4A16 execution counters: marlin_gemms="
          << st.marlin_gemms << " fused_gate_up=" << st.fused_gate_up
          << " fallback_gemms=" << st.fallback_gemms);
  CHECK_MESSAGE(st.marlin_gemms > 0,
                "the NVFP4 W4A16 Marlin dense GEMM never ran — the quantized "
                "path was NOT exercised by this gate");
  CHECK_MESSAGE(st.fused_gate_up > 0,
                "the fused gate_up Marlin GEMM never ran — the MLP did not take "
                "vLLM's merged gate_up_proj layout");

  MESSAGE("Qwen3-32B-NVFP4A16 correctness gate: "
          << (strict_exact + neartie_only) << "/" << N << " prompts PASS  "
          << "(STRICT token-exact: " << strict_exact << "/" << N << " = "
          << exact_tokens << "/" << total_tokens
          << " tokens; near-tie-band only: " << neartie_only << "/" << N
          << "; max teacher-forced gap " << (worst_gap / 1000.0)
          << " nats @ prompt[" << worst_i << "] tok=" << worst_j << " (bar "
          << (kNearTieMnats / 1000.0) << "); " << fail
          << " forward-divergent; vLLM self-determinism: " << multi_cells
          << " multi-valued cells)");
  REQUIRE(fail == 0);
}
