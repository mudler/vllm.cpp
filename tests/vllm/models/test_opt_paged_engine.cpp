// vllm.cpp original (ADDITIVE-MODEL W4 — THE SACRED correctness gate for OPT
// (`OPTForCausalLM`, facebook/opt-125m), the CROSS-FAMILY additivity canary).
//
// Upstream test mirrored: `tests/models/language/generation/test_common.py:77`,
// where `facebook/opt-125m` is one of the models whose greedy generation vLLM
// asserts against the HF reference. Our equivalent asserts our paged engine's
// greedy against the pinned vLLM 0.25.0 oracle itself.
//
// THE GATE IS **STRICT TOKEN-EXACT**, and that is a MEASURED choice, not an
// assumption. Per the ratified methodology ([[near-tie-distributional-gate]])
// the gate is selected by first measuring whether vLLM's OWN greedy is
// deterministic on this model. It is: `scripts/opt-oracle-capture.py --runs 5`
// reported ALL SIX prompts deterministic over K=5 runs with **0 multi-valued
// (prompt,pos) cells** (evidence committed in greedy_dist.npy, and re-asserted
// at the top of this test). Where vLLM is self-consistent the honest bar is
// exact agreement — so no near-tie band is used here at all, and any divergence
// is reported with its first position for diagnosis rather than absorbed.
//
// Goldens (committed under tests/parity/goldens/opt_greedy/, dgx-captured):
//   greedy_ids.npy   [N,T]   i32  vLLM per-prompt greedy (run 0) — the BAR.
//   greedy_dist.npy  [N,T,K] i32  K=5 vLLM runs (the self-determinism evidence
//                                 that SELECTS the strict gate).
//   p{i}_prompt.i32  [Li]    i32  vLLM's tokenization (cross-checked vs ours).
//
// DTYPE: both arms run BF16. facebook/opt-125m ships `torch_dtype: float16`,
// but our CUDA compute path is bf16/f32 (kF16 is unimplemented outside
// cuda_gdn.cu), and `--dtype bfloat16` is a first-class vLLM production mode —
// so the oracle was captured under it and our checkpoint was materialized bf16
// with the SAME single fp16->bf16 rounding. See
// .agents/specs/sweep-opt-125m.md decision D1.
//
// Checkpoint-GATED: resolves the materialized bf16-safetensors dir from
// scripts/opt-materialize-checkpoint.py (the HF snapshot ships a torch-pickle
// `pytorch_model.bin` our loader cannot read — decision D2). Where the dir and
// goldens are absent the body emits a loud SKIP, so it compiles + links
// everywhere and RUNS wherever the checkpoint is staged.
//
// MULTI-BACKEND (BACKEND-METAL-MLX work row M3a). This was dgx-only. It is now
// the SAME gate on two accelerators from one source: model_loader.cpp's
// SelectQueue asks the Platform seam instead of hardcoding kCUDA, so this runs
// on CUDA on dgx.casa (GB10) and on Metal on the Apple M4, both against the SAME
// committed dgx-captured vLLM 0.25.0 goldens. That is what makes it a real
// cross-backend gate rather than two separately-baselined tests: the goldens are
// device-INDEPENDENT (they are vLLM's tokens, not ours), so Metal has to match
// the bar CUDA already met, not a bar re-derived on Metal.
//
// The token comparison alone does not say WHICH device executed — every backend
// computes the same function — so the body also asserts the op-provider
// selection/decline counters for all nine OPT ops on the running device.
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
#include "vt/op_provider.h"  // the "which backend actually ran" proof
#include "vt/ops.h"

namespace fs = std::filesystem;

namespace {

// The compact prompt battery. Kept in the test (the oracle capture script reads
// the SAME list) so the goldens and the gate never drift. MUST match
// scripts/opt-oracle-capture.py::PROMPTS exactly. OPT-125m is a 2022 base LM
// with no chat template, so these are plain completion stems.
const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "Once upon a time,",
      "The largest planet in our solar system is",
      "The chemical symbol for gold is",
      "In 1969, humans first walked on",
      "Water boils at a temperature of",
  };
  return p;
}

// The materialized bf16-safetensors OPT-125m dir (config.json +
// model.safetensors + tokenizer.json). Overridable for a non-default location.
std::string FindOptModelDir() {
  if (const char* env = std::getenv("VLLM_CPP_OPT_MODEL_DIR"); env != nullptr) {
    std::error_code ec;
    if (fs::exists(fs::path(env) / "config.json", ec)) return env;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path dir = fs::path(home) / "models/opt-125m-bf16-st";
  std::error_code ec;
  if (fs::exists(dir / "config.json", ec) && fs::exists(dir / "model.safetensors", ec))
    return dir.string();
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
    if (std::fread(out.data(), sizeof(int32_t), out.size(), f) != out.size()) out.clear();
  }
  std::fclose(f);
  return out;
}

}  // namespace

TEST_CASE("opt-125m paged-engine greedy STRICT token-exact gate (CUDA + Metal, SACRED)") {
  const std::string dir = FindOptModelDir();
  if (dir.empty()) {
    MESSAGE(
        "SKIP (dgx-only): materialized OPT-125m checkpoint absent "
        "(~/models/opt-125m-bf16-st or $VLLM_CPP_OPT_MODEL_DIR); build it with "
        "scripts/opt-materialize-checkpoint.py");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "opt_greedy";
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE("SKIP: OPT greedy golden absent — capture on dgx: "
            "scripts/opt-oracle-capture.py --runs 5");
    return;
  }

  // ---- GATE SELECTION (the ratified methodology) ---------------------------
  // Re-assert vLLM's own self-determinism from the committed K-run evidence.
  // Zero multi-valued cells is what licenses the STRICT bar below; if this ever
  // becomes non-zero the gate must be re-derived, not silently loosened.
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
            ? std::string(" (DETERMINISTIC -> STRICT token-exact gate)")
            : std::string(" (NON-DET: the strict bar below is no longer the "
                          "right gate — re-derive it)");
    MESSAGE("opt-125m: vLLM self-determinism over K=" << DK
            << " runs — multi-valued (prompt,pos) cells = " << multi_cells
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

  MESSAGE("opt-125m: loading via FromModelDir(" << dir << ") — bf16 dense, "
          "LEARNED positions (offset 2), biased projections, LayerNorm, ReLU MLP...");
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(dir,
                                                    vllm::entrypoints::EngineParams{});

  // ---- WHICH BACKEND ACTUALLY RAN (BACKEND-METAL-MLX work row M3a) ---------
  // A passing token comparison does NOT prove which device executed: every
  // backend computes the same function, so a silent fall-back to the CPU
  // reference would produce an IDENTICAL green result. Since
  // model_loader.cpp::SelectQueue now asks the Platform seam rather than
  // hardcoding kCUDA, this test runs on CUDA on dgx and on Metal on the M4 from
  // the same source — so the device is recorded, and on an accelerator the
  // op-provider DECLINE counters are asserted zero.
  //
  // `declines == 0` is the load-bearing half. `last_selected` alone is
  // insufficient: a provider can be selected and then decline INSIDE its kernel
  // and forward down the stack, which is exactly the fan-out spike's Risk 4
  // (a probe failing silently into the slow path).
  const vt::DeviceType run_dev = loaded->runner().device().type;
  MESSAGE("opt-125m: the engine selected device type " << static_cast<int>(run_dev)
          << " (0=CPU, 1=CUDA, 2=METAL, 3=VULKAN, 4=XPU)");
  // The nine ops OPT's forward + greedy sampling dispatch. Every one must be
  // REGISTERED for the running device, must actually be SELECTED at least once,
  // and must never DECLINE.
  const std::vector<vt::OpId> kOptOps = {
      vt::OpId::kEmbedding,        vt::OpId::kMatmulBT,       vt::OpId::kAdd,
      vt::OpId::kLayerNorm,        vt::OpId::kRelu,           vt::OpId::kQkvSplit,
      vt::OpId::kReshapeAndCache,  vt::OpId::kPagedAttention, vt::OpId::kGreedyArgmax};
  if (run_dev != vt::DeviceType::kCPU) {
    for (vt::OpId op : kOptOps) {
      CHECK(vt::OpRegistered(op, run_dev));
      vt::ResetOpProviderStats(op, run_dev);
    }
    // Per-call selection counting is OFF by default (the hot path is a cached
    // pointer load); turn it on EXPLICITLY here rather than depending on the
    // environment, so the proof below is a property of the test.
    vt::EnableOpProviderCallStats(true);
  }

  int exact_prompts = 0;
  int64_t exact_tokens = 0;
  int64_t total_tokens = 0;
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "opt" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);

    // Diagnostic: the engine's tokenization must match vLLM's prompt_ids. A
    // mismatch here would make a token comparison meaningless.
    const std::vector<int32_t> want_prompt =
        LoadI32File(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    if (!want_prompt.empty()) CHECK(out.prompt_token_ids == want_prompt);

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
      // difference. Report the FIRST divergent position for diagnosis — the OPT
      // suspects are the learned pos-emb offset-of-2, a dropped bias term, the
      // pre/post-LN switch, the LayerNorm eps, and ReLU-vs-GELU.
      MESSAGE("opt-125m DIVERGENCE prompt[" << i << "] first bad tok=" << first_bad
              << " ours=" << got[static_cast<size_t>(first_bad)]
              << " vLLM=" << gd[i * T + first_bad] << "  \"" << out.outputs[0].text
              << "\"");
    }
    CHECK_MESSAGE(exact, "opt-125m prompt[" << i << "] is not token-exact vs the "
                                               "deterministic vLLM 0.25.0 oracle");
  }

  // ---- the backend proof, now that work has actually been dispatched -------
  if (run_dev != vt::DeviceType::kCPU) {
    vt::EnableOpProviderCallStats(false);
    for (vt::OpId op : kOptOps) {
      const auto st = vt::GetOpProviderStats(op, run_dev);
      // SELECTED at least once: the device kernel really served this op during
      // the 96 generated tokens, rather than the op never being reached.
      CHECK_MESSAGE(st.selections > 0,
                    "opt-125m: op " << static_cast<int>(op)
                                    << " was never dispatched on the running device — the "
                                       "token match cannot be attributed to this backend");
      // NEVER DECLINED: no silent forward down the provider stack.
      CHECK_MESSAGE(st.declines == 0,
                    "opt-125m: op " << static_cast<int>(op)
                                    << " DECLINED on the running device and fell back");
    }
    MESSAGE("opt-125m: BACKEND PROOF — all 9 OPT ops dispatched on device type "
            << static_cast<int>(run_dev) << " with 0 declines "
            << "(kPagedAttention selections="
            << vt::GetOpProviderStats(vt::OpId::kPagedAttention, run_dev).selections << ")");
  }

  MESSAGE("opt-125m STRICT correctness gate: " << exact_prompts << "/" << N
          << " prompts token-exact (" << exact_tokens << "/" << total_tokens
          << " tokens) vs the vLLM 0.25.0 oracle "
          << "(vLLM self-determinism: " << multi_cells << " multi-valued cells)");
  REQUIRE(exact_prompts == N);
}
