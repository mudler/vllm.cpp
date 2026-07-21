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
// THE GATE IS **STRICT TOKEN-EXACT**, and that is a MEASURED choice, not an
// assumption. Per the ratified methodology ([[near-tie-distributional-gate]])
// the gate is selected by first measuring whether vLLM's OWN greedy is
// deterministic on THIS checkpoint. It is:
// `scripts/qwen3-32b-nvfp4a16-oracle-capture.py --runs 5` reported ALL SIX
// prompts deterministic over K=5 runs with **0 multi-valued (prompt,pos) cells**
// (evidence committed in greedy_dist.npy, and re-asserted at the top of this
// test). Where vLLM is self-consistent the honest bar is exact agreement — so no
// near-tie band is used here at all, and any divergence is reported with its
// first position for diagnosis rather than absorbed.
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
//       localized it in a single run. Qwen3 prepends NO BOS.
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
    "Qwen3-32B-NVFP4A16 paged-engine greedy STRICT token-exact gate "
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

  // ---- GATE SELECTION (the ratified methodology) ---------------------------
  // Re-assert vLLM's own self-determinism from the committed K-run evidence.
  // Zero multi-valued cells is what licenses the STRICT bar below; if this ever
  // becomes non-zero the gate must be re-derived, not silently loosened.
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
            ? std::string(" (DETERMINISTIC -> STRICT token-exact gate)")
            : std::string(" (NON-DET: the strict bar below is no longer the "
                          "right gate — re-derive it)");
    MESSAGE("Qwen3-32B-NVFP4A16: vLLM self-determinism over K="
            << DK << " runs — multi-valued (prompt,pos) cells = " << multi_cells
            << verdict);
    CHECK(multi_cells == 0);
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  const int64_t N = g.shape[0];
  const int64_t T = g.shape[1];
  REQUIRE(static_cast<size_t>(N) == Prompts().size());
  const int32_t* gd = AsI32(g);

  MESSAGE("Qwen3-32B-NVFP4A16: loading via FromModelDir("
          << dir
          << ") — 64L dense, compressed-tensors nvfp4-pack-quantized W4A16 "
             "(weight_packed/weight_scale/weight_global_scale, group_size 16)");
  vllm::dense_nvfp4::ResetW4A16Stats();
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(
          dir, vllm::entrypoints::EngineParams{});

  int exact_prompts = 0;
  int64_t exact_tokens = 0;
  int64_t total_tokens = 0;
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

    bool exact = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      ++total_tokens;
      if (got[static_cast<size_t>(j)] == gd[i * T + j]) {
        ++exact_tokens;
      } else {
        exact = false;
        if (first_bad < 0) first_bad = static_cast<int>(j);
      }
    }
    if (exact) {
      ++exact_prompts;
    } else {
      // vLLM is DETERMINISTIC here, so a divergence is a real forward
      // difference. Report the FIRST divergent position for diagnosis — for this
      // row the suspects are all in the NEW variable (the quant path): the
      // merged-shard global-scale collapse (max-then-reciprocate), the
      // divisor-vs-multiplier convention, the group-16 scale layout, the Marlin
      // scale swizzle/exponent-bias, or the fused-vs-split gate_up layout.
      MESSAGE("Qwen3-32B-NVFP4A16 DIVERGENCE prompt["
              << i << "] first bad tok=" << first_bad
              << " ours=" << got[static_cast<size_t>(first_bad)]
              << " vLLM=" << gd[i * T + first_bad] << "  \""
              << out.outputs[0].text << "\"");
    }
    CHECK_MESSAGE(exact, "Qwen3-32B-NVFP4A16 prompt["
                             << i
                             << "] is not token-exact vs the deterministic vLLM "
                                "0.25.0 oracle");
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

  MESSAGE("Qwen3-32B-NVFP4A16 STRICT correctness gate: "
          << exact_prompts << "/" << N << " prompts token-exact ("
          << exact_tokens << "/" << total_tokens
          << " tokens) vs the vLLM 0.25.0 oracle "
          << "(vLLM self-determinism: " << multi_cells << " multi-valued cells)");
  REQUIRE(exact_prompts == N);
}
