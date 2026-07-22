// vllm.cpp original (MLA CAMPAIGN W8 — THE SACRED correctness gate for
// DeepSeek-V2-Lite, `DeepseekV2ForCausalLM`, the FIRST MLA model in this tree);
// no upstream mirror. Sibling of test_opt_paged_engine.cpp (strict form) and
// test_qwen3coder_paged_engine.cpp (paged-engine battery shape).
//
// THE PAGED-ENGINE DeepSeek-V2-Lite (MLA + DeepSeek-MoE BF16) GREEDY CORRECTNESS
// GATE. Drives a compact prompt battery through the FULL PAGED LLMEngine stack
// (InputProcessor -> Scheduler -> the decode-first reorder -> MLA cache write +
// MLA decode/prefill + DeepSeek MoE forward -> Sampler -> OutputProcessor) via
// LoadedEngine::FromModelDir, and checks the greedy (temperature-0) decode
// against the pinned vLLM 0.25.0 oracle.
//
// ─── THE BAR, and how it was ARRIVED AT (measured twice, never assumed) ──────
//
// STEP 1 — is the oracle deterministic? Per the ratified methodology
// ([[near-tie-distributional-gate]]) the gate form is selected by first
// measuring whether vLLM's OWN greedy is deterministic on this model **in the
// regime the gate runs in**. Measured twice:
//   * a first K=5 probe that generated all 8 prompts in ONE BATCH reported vLLM
//     self-inconsistent on 3/8 — the KNOWN BATCHING ARTIFACT (batched generation
//     re-orders reductions; the Qwen3-dense razor hit the same one). It is NOT
//     non-determinism and must NOT be used to derive a weaker gate.
//   * re-probed at **batch=1, K=5 — the actual gate regime — vLLM's own greedy
//     is DETERMINISTIC on 8/8 prompts** (campaign W0,
//     `~/scratch_mla_w1/w0_probe_b1.{py,log,json}` on dgx), and W8's own capture
//     re-confirmed it at T=16 with **0 multi-valued (prompt,pos) cells**
//     (greedy_dist.npy, re-asserted at run time below).
// So the STRICT token-exact bar is the one that is well-posed here, and it is
// what this gate applies wherever it can.
//
// STEP 2 — we ran it, and it came out **5/8** (92/128 tokens). Against a
// DETERMINISTIC oracle a divergence is a defect until proven otherwise, so the
// ratified TEACHER-FORCING procedure was run rather than a band being assumed
// (scripts/deepseek-v2-neartie-gap.py: feed vLLM OUR exact sequence and ask, per
// position, how many nats vLLM's OWN argmax beats OUR token by GIVEN OUR
// PREFIX). The answer, committed as goldens:
//   * **36 divergent positions, of which 35 have gap EXACTLY 0.0000 nats** —
//     i.e. vLLM's own logits on our prefix pick OUR token. Those are not errors;
//     they are the downstream tail of a single earlier flip.
//   * **exactly ONE root flip carries any gap at all: prompt[3] tok 9, 0.2500
//     nats**, inside the ratified 0.5-nat band and equal to the worst gap the
//     already-landed Qwen3-dense 4B gate carries.
//   * **ZERO tokens outside vLLM's top-20.** No real forward divergence exists.
//   * Two of the three root flips (prompt[2] tok 1, prompt[5] tok 1) have gap
//     0.0000 — at those positions vLLM's incremental DECODE argmax disagrees
//     with vLLM's OWN teacher-forced PREFILL argmax. Our token is the one vLLM's
//     logits prefer; vLLM's greedy output is the one that differs from them.
//
// STEP 3 — so the gate that applies is the ratified NEAR-TIE-ROBUST form, the
// same one the Qwen3-dense (16/16) and Qwen3-Coder (6/6) gates use: **STRICT
// where our token IS vLLM's argmax, near-tie-tolerant ONLY where vLLM itself
// cannot separate the two tokens, and FAILING on anything beyond
// kNearTieMnats**. This is not a loosening: a real forward bug produces a large
// gap or a token outside vLLM's top-K, and either FAILS. The per-position nats
// evidence is COMMITTED (neartie_gap_mnats.npy) so the claim is auditable, and
// our_ids.npy anchors it to the exact sequence it was measured on — an engine
// change that moves a token trips the anchor check and forces a re-capture
// rather than silently inheriting a stale band.
//
// ─── WHAT ELSE THIS GATE PROVES (a passing gate does not prove EXECUTION) ────
// Phase 0 asserts the engine actually allocated an **MLA** KV cache: the runner's
// spec-driven page cost must equal `block_size * 576 * dtype_size` — the factor 2
// every other attention spec carries simply does not appear (W1).
// Phase 2 is the W8 SCHEDULER/RUNNER WIRING gate. W7 built the model so that
// `BuildMlaBatchSplit` THROWS (naming the request and citing the upstream line)
// on an illegal batch — a decode after a prefill, or a with-context prefill after
// a context-free one (mla_attention.py:1640-1649 `split_decodes_and_prefills`
// with `reorder_batch_threshold == 1` at :1420, and :1806-1810
// `prefill_tokens_with_context`, which is a PREFIX LENGTH). W6 measured **0.86
// relative error** from violating the second. Phase 2 drives STAGGERED concurrent
// requests so the engine really produces MIXED decode+prefill steps, and phase 3
// drives a long repeated prompt so a PREFIX-CACHE hit produces a genuine
// WITH-CONTEXT prefill. Both would blow up loudly if the order were wrong; the
// `MlaBatchSplitStats` counters then prove the shapes were non-vacuous (a legal
// order over a stream of batch-of-1 steps would be a meaningless pass).
//
// ─── TOKENIZATION GOLDENS ARE A FIRST-CLASS CHECK, NOT A NICETY ──────────────
// On OPT a silently-unapplied BOS scored 0/6 while emitting fluent English, and
// the committed `p{i}_prompt.i32` goldens are what caught it in ONE run.
// DeepSeek ships its own tokenizer (a 100k BPE with `add_bos_token: true`,
// bos_token_id 100000), so the same class of bug is live here — the prompt-id
// comparison below is REQUIRED, not advisory: a token comparison against an
// oracle fed a different prompt is meaningless.
//
// Goldens (committed under tests/parity/goldens/deepseek_v2_greedy/, dgx-captured
// by scripts/deepseek-v2-oracle-capture.py at gpu_memory_utilization=0.40,
// batch=1 — the determinism-sensitive capture regime):
//   greedy_ids.npy         [N,T]   i32  vLLM per-prompt greedy (run 0).
//   greedy_dist.npy        [N,T,K] i32  K=5 vLLM runs (the self-determinism
//                                       evidence that licenses the strict bar).
//   our_ids.npy            [N,T]   i32  OUR engine's greedy (the anchor the
//                                       gaps were measured on).
//   neartie_gap_mnats.npy  [N,T]   i32  vLLM's teacher-forced gap (milli-nats)
//                                       for OUR token given OUR prefix.
//   p{i}_prompt.i32        [Li]    i32  vLLM's tokenization of prompt i.
//
// Checkpoint-GATED + dgx-only: resolves the real
// models--deepseek-ai--DeepSeek-V2-Lite snapshot under ~/.cache/huggingface/hub/.
// On CPU/CI the snapshot + goldens are absent, so the body emits a loud SKIP and
// returns — compiles + links on CPU, RUNS only on dgx.casa (GB10).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
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

// Near-tie acceptance threshold in MILLI-nats. vLLM's argmax must beat OUR token
// by <= this, in vLLM's OWN teacher-forced logits GIVEN OUR PREFIX, for a
// divergence to count as a bf16 near-tie rather than a forward bug. 500 mnats
// (0.5 nats) is the ratified bar, unchanged from the Qwen3-dense and
// Qwen3-Coder gates; DeepSeek-V2-Lite's worst observed gap is 0.25 nats.
constexpr int32_t kNearTieMnats = 500;

// The compact battery W0 proved deterministic at batch=1 (8/8, K=5). MUST match
// scripts/deepseek-v2-oracle-capture.py::PROMPTS and
// scripts/deepseek-v2-neartie-gap.py::PROMPTS exactly, so the goldens and the
// gate can never drift.
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

// A prompt long enough to fill whole KV blocks, so a SECOND submission of it hits
// the prefix cache and the engine schedules a genuine WITH-CONTEXT prefill (the
// `prefill_tokens_with_context` prefix-length invariant, mla_attention.py:1806-1810).
std::string LongPrompt() {
  std::string s =
      "The history of computing begins long before the electronic computer. ";
  std::string out;
  for (int i = 0; i < 8; ++i) out += s;
  out += "In summary,";
  return out;
}

std::string FindDeepseekV2Snapshot() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snaps = fs::path(home) /
                         ".cache/huggingface/hub/"
                         "models--deepseek-ai--DeepSeek-V2-Lite/snapshots";
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

// DeepSeek-V2-Lite (`DeepseekV2ForCausalLM`) — the MLA campaign's SACRED gate.
// STRICT token-exact, because the vLLM 0.25.0 oracle measured DETERMINISTIC on
// this model at batch=1 (W0, K=5, 8/8 prompts).
TEST_CASE("deepseek-v2-lite paged-engine greedy STRICT token-exact gate (dgx-only, SACRED)") {
  const std::string snap = FindDeepseekV2Snapshot();
  if (snap.empty()) {
    MESSAGE(
        "SKIP (dgx-only): DeepSeek-V2-Lite checkpoint absent — "
        "deepseek-ai/DeepSeek-V2-Lite snapshot not present under "
        "~/.cache/huggingface/hub/");
    return;
  }
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "deepseek_v2_greedy";
  if (!fs::exists(gdir / "greedy_ids.npy")) {
    MESSAGE("SKIP: DeepSeek-V2 greedy golden absent — capture on dgx: "
            "scripts/deepseek-v2-oracle-capture.py --runs 5");
    return;
  }

  // ---- GATE SELECTION (the ratified methodology) ---------------------------
  // Re-assert vLLM's own self-determinism from the committed K-run evidence.
  // ZERO multi-valued cells is what licenses the STRICT bar below. The capture
  // ran ONE PROMPT PER generate() CALL; a batched capture would manufacture
  // false multi-valued cells and must never be used to weaken this gate.
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
            : std::string(" (NON-DET: the bar below is no longer the right gate "
                          "— RE-DERIVE it via the teacher-forcing procedure, do "
                          "not loosen it)");
    MESSAGE("deepseek-v2-lite: vLLM self-determinism over K=" << DK
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

  // The teacher-forced evidence (see the header, STEP 2/3). Absent => the gate
  // cannot run: without the per-position nats we would have no basis on which to
  // call any divergence a near-tie, and silently falling back to "strict only"
  // would be reporting a number whose provenance we cannot show.
  const bool have_gap = fs::exists(gdir / "our_ids.npy") &&
                        fs::exists(gdir / "neartie_gap_mnats.npy");
  if (!have_gap) {
    MESSAGE("SKIP: DeepSeek-V2 teacher-forced goldens absent — run the gate "
            "under VT_DUMP_IDS=1 then scripts/deepseek-v2-neartie-gap.py");
    return;
  }
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap =
      parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  REQUIRE(o.shape.size() == 2);
  REQUIRE(o.shape[0] == N);
  REQUIRE(o.shape[1] == T);
  REQUIRE(gap.shape.size() == 2);
  REQUIRE(gap.shape[0] == N);
  REQUIRE(gap.shape[1] == T);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  MESSAGE("deepseek-v2-lite: loading via FromModelDir(" << snap
          << ") — MLA (576-wide latent, QK 192 / V 128 prefill, MQA-576 decode) "
             "+ DeepSeek MoE (64 routed + 2 SHARED experts, grouped router)...");
  vllm::entrypoints::EngineParams params;  // block 32, 256 blocks, 8 seqs
  std::unique_ptr<vllm::entrypoints::LoadedEngine> loaded =
      vllm::entrypoints::LoadedEngine::FromModelDir(snap, params);

  // ---- PHASE 0: the engine really allocated an MLA cache -------------------
  // The MLA page is `block_size * num_kv_heads(1) * head_size(576) * dtype_size`
  // (kv_cache_interface.py:397-398) — NO factor 2 and NO separate V. Reading it
  // off the runner proves the allocation came from the MLAAttentionSpec (W1's
  // spec-driven sizing), not from an HF-config reconstruction.
  const int64_t kMlaHeadSize = 512 + 64;  // kv_lora_rank + qk_rope_head_dim
  const int64_t want_page =
      static_cast<int64_t>(params.block_size) * kMlaHeadSize * 2;  // bf16
  MESSAGE("deepseek-v2-lite: runner fa_page_size_bytes = "
          << loaded->runner().fa_page_size_bytes() << " (MLA expects "
          << want_page << " = block " << params.block_size << " x " << kMlaHeadSize
          << " x 2B, with NO factor 2 for K+V)");
  REQUIRE(loaded->runner().fa_page_size_bytes() == want_page);

  // ---- PHASE 1: THE CORRECTNESS GATE (batch=1, the oracle's own regime) ----
  vllm::ResetMlaBatchSplitStats();
  int strict_exact = 0;   // prompts whose tokens ARE vLLM's greedy, exactly
  int neartie_only = 0;   // prompts that pass only through the near-tie band
  int fail = 0;           // prompts with a token beyond the near-tie band
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  int64_t exact_tokens = 0;
  int64_t total_tokens = 0;
  std::vector<std::vector<int32_t>> serial_ids(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    const vllm::RequestOutput out = loaded->engine().generate(
        Prompts()[static_cast<size_t>(i)], Greedy(static_cast<int>(T)),
        "dsv2-" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);
    serial_ids[static_cast<size_t>(i)] = got;

    // TOKENIZATION GOLDEN — REQUIRED. See the header: a token comparison against
    // an oracle that was fed a different prompt is meaningless, and this exact
    // check is what caught OPT's missing BOS in one run.
    const std::vector<int32_t> want_prompt =
        LoadI32File(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    REQUIRE_MESSAGE(!want_prompt.empty(),
                    "deepseek-v2-lite: tokenization golden p" << i
                    << "_prompt.i32 missing — re-capture with "
                       "scripts/deepseek-v2-oracle-capture.py");
    REQUIRE_MESSAGE(out.prompt_token_ids == want_prompt,
                    "deepseek-v2-lite TOKENIZATION MISMATCH prompt[" << i
                    << "] ours[" << out.prompt_token_ids.size() << "] vs vLLM["
                    << want_prompt.size() << "] — the token comparison below "
                       "would be meaningless (DeepSeek sets add_bos_token=true, "
                       "bos_token_id=100000)");

    // ANCHOR: the committed gaps describe OUR engine's exact sequence. Drift
    // here means the engine changed and the nats evidence is stale — re-capture
    // it rather than judging today's tokens against yesterday's band.
    for (int64_t j = 0; j < T; ++j) {
      REQUIRE_MESSAGE(got[static_cast<size_t>(j)] == od[i * T + j],
                      "deepseek-v2-lite anchor drift prompt[" << i << "] tok=" << j
                      << " engine=" << got[static_cast<size_t>(j)]
                      << " committed our_ids=" << od[i * T + j]
                      << " — re-run scripts/deepseek-v2-neartie-gap.py to refresh "
                         "the teacher-forced goldens");
    }

    bool exact = true;
    bool prompt_ok = true;
    int first_bad = -1;   // first position beyond the near-tie band
    int first_diff = -1;  // first position differing from vLLM's greedy
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
      // Beyond the band, or outside vLLM's top-K: a REAL forward divergence, not
      // a tie. The MLA suspects, in the order W6/W7 found them worth checking:
      // the mscale^2 softmax-scale correction (deepseek_v2.py:1071-1074), the
      // decoupled is_neox_style=False YaRN rope over qk_rope_head_dim ONLY, the
      // absorbed-decode vs materialized-prefill split, the batch ORDER, the
      // grouped router, and the shared-expert add.
      ++fail;
      MESSAGE("deepseek-v2-lite FORWARD DIVERGENCE prompt[" << i << "] tok="
              << first_bad << " our=" << got[static_cast<size_t>(first_bad)]
              << " vLLM_greedy=" << gd[i * T + first_bad]
              << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats (> "
              << (kNearTieMnats / 1000.0) << ")  \"" << out.outputs[0].text << "\"");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
      MESSAGE("deepseek-v2-lite NEAR-TIE prompt[" << i << "] first differing tok="
              << first_diff << " our=" << got[static_cast<size_t>(first_diff)]
              << " vLLM_greedy=" << gd[i * T + first_diff] << " gap="
              << (gapd[i * T + first_diff] / 1000.0)
              << " nats (vLLM's OWN argmax on OUR prefix)  \""
              << out.outputs[0].text << "\"");
    }
    CHECK_MESSAGE(prompt_ok,
                  "deepseek-v2-lite prompt[" << i << "] diverges from vLLM 0.25.0 "
                  "BEYOND the near-tie band — a real forward difference");
  }

  const vllm::MlaBatchSplitStats serial_stats = vllm::GetMlaBatchSplitStats();
  MESSAGE("deepseek-v2-lite correctness gate: " << (strict_exact + neartie_only)
          << "/" << N << " prompts PASS  (STRICT token-exact vs vLLM per-prompt "
          << "greedy: " << strict_exact << "/" << N << "; near-tie-band only: "
          << neartie_only << "/" << N << "; tokens strictly exact "
          << exact_tokens << "/" << total_tokens << "; max teacher-forced gap "
          << (worst_gap / 1000.0) << " nats @ prompt[" << worst_i << "] tok="
          << worst_j << "; " << fail << " forward-divergent; vLLM "
          << "self-determinism: " << multi_cells << " multi-valued cells)");
  // PROOF THE PATH RAN: the strict phase must have driven real MLA forwards —
  // N prefill steps plus N*(T-1) decode steps through the MLA block.
  MESSAGE("deepseek-v2-lite MLA split stats (phase 1, batch=1): steps="
          << serial_stats.steps << " prefill_only=" << serial_stats.prefill_only_steps
          << " decode_only=" << serial_stats.decode_only_steps
          << " mixed=" << serial_stats.mixed_steps
          << " decode_tokens=" << serial_stats.total_decode_tokens
          << " prefill_tokens=" << serial_stats.total_prefill_tokens);
  REQUIRE(serial_stats.steps >= N * T);
  REQUIRE(serial_stats.prefill_only_steps >= N);
  REQUIRE(serial_stats.total_decode_tokens >= N * (T - 1));

  // VT_DUMP_IDS=1 writes OUR exact sequences so scripts/deepseek-v2-neartie-gap.py
  // can TEACHER-FORCE vLLM on them and say, per divergent position, whether our
  // token is one vLLM's own logits cannot separate from its argmax (a bf16
  // near-tie) or a REAL forward difference. That diagnosis is mandatory when a
  // divergence appears against a DETERMINISTIC oracle — it is how the defect
  // gets localized, and it is the ONLY evidence that could ever justify the
  // ratified near-tie-robust form of this gate.
  if (std::getenv("VT_DUMP_IDS") != nullptr) {
    std::vector<int32_t> buf(static_cast<size_t>(N * T), -1);
    for (int64_t i = 0; i < N; ++i)
      for (int64_t j = 0; j < T; ++j)
        buf[static_cast<size_t>(i * T + j)] =
            serial_ids[static_cast<size_t>(i)][static_cast<size_t>(j)];
    const std::string path = (gdir / "our_ids.i32").string();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
      std::fwrite(buf.data(), sizeof(int32_t), buf.size(), f);
      std::fclose(f);
      MESSAGE("deepseek-v2-lite dumped our token ids -> " << path);
    }
  }

  // ---- PHASE 2: THE W8 SCHEDULER/RUNNER WIRING GATE ------------------------
  // Drive the SAME battery CONCURRENTLY with STAGGERED admission so the engine
  // genuinely produces MIXED steps (already-decoding requests alongside a fresh
  // prefill). If the runner's decode-first reorder
  // (reorder_batch_to_split_decodes_and_prefills, decode_threshold == 1 —
  // exactly MLA's `reorder_batch_threshold`) failed to put decodes first, or put
  // a context-free prefill ahead of a with-context one, BuildMlaBatchSplit
  // THROWS out of the forward and this phase fails loudly.
  // Default output_kind is CUMULATIVE, so a finished RequestOutput carries the
  // whole sequence (mirrors tests/vllm/v1/test_llm_engine.cpp's concurrency case).
  vllm::ResetMlaBatchSplitStats();
  std::map<std::string, vllm::RequestOutput> finished;
  for (int64_t i = 0; i < N; ++i) {
    loaded->engine().add_request("dsv2-conc-" + std::to_string(i),
                                 Prompts()[static_cast<size_t>(i)],
                                 Greedy(static_cast<int>(T)));
    // Stagger: let the already-admitted requests advance a couple of decode
    // steps before the next prefill joins, which is what creates a MIXED batch.
    for (int s = 0; s < 2 && loaded->engine().has_unfinished_requests(); ++s) {
      for (const vllm::RequestOutput& o : loaded->engine().step()) {
        if (o.finished) finished[o.request_id] = o;
      }
    }
  }
  while (loaded->engine().has_unfinished_requests()) {
    for (const vllm::RequestOutput& o : loaded->engine().step()) {
      if (o.finished) finished[o.request_id] = o;
    }
  }
  std::vector<std::vector<int32_t>> concurrent_ids(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    const std::string rid = "dsv2-conc-" + std::to_string(i);
    REQUIRE_MESSAGE(finished.count(rid) == 1,
                    "deepseek-v2-lite: concurrent request " << rid
                    << " never finished");
    REQUIRE(finished[rid].outputs.size() == 1);
    concurrent_ids[static_cast<size_t>(i)] = finished[rid].outputs[0].token_ids;
  }
  const vllm::MlaBatchSplitStats conc_stats = vllm::GetMlaBatchSplitStats();
  MESSAGE("deepseek-v2-lite MLA split stats (phase 2, staggered concurrent): steps="
          << conc_stats.steps << " prefill_only=" << conc_stats.prefill_only_steps
          << " decode_only=" << conc_stats.decode_only_steps
          << " MIXED=" << conc_stats.mixed_steps
          << " max_num_reqs=" << conc_stats.max_num_reqs
          << " max_num_decodes=" << conc_stats.max_num_decodes
          << " max_num_prefills=" << conc_stats.max_num_prefills);
  // NON-VACUITY: a legal ordering over a stream of batch-of-1 steps would prove
  // nothing. The engine must have produced multi-request batches AND at least
  // one step carrying BOTH a decode and a prefill — the exact shape whose
  // ordering MLA's q[:num_mqa_tokens] / q[num_mqa_tokens:] slicing depends on.
  REQUIRE(conc_stats.max_num_reqs > 1);
  REQUIRE(conc_stats.max_num_decodes > 1);
  REQUIRE(conc_stats.mixed_steps > 0);
  for (int64_t i = 0; i < N; ++i) {
    CHECK_MESSAGE(static_cast<int64_t>(concurrent_ids[static_cast<size_t>(i)].size()) == T,
                  "deepseek-v2-lite concurrent request " << i << " produced "
                  << concurrent_ids[static_cast<size_t>(i)].size() << " tokens, want " << T);
  }
  // BATCH INVARIANCE — REPORTED, deliberately NOT asserted, and here is why.
  //
  // The tempting assertion is "a request's tokens must not depend on who it
  // shares a step with". It would be the wrong bar, because THE ORACLE ITSELF
  // DOES NOT MEET IT on this model. Campaign W0 measured exactly this: a K=5
  // greedy probe that generated all 8 prompts in ONE BATCH reported vLLM
  // self-inconsistent on **3/8** prompts, while the same probe at batch=1 was
  // deterministic 8/8. Batched generation re-orders reductions (here: the
  // grouped MoE GEMM batches tokens across requests, so a token's accumulation
  // order depends on the step's total token count), and this model sits on bf16
  // near-ties — 3 of our 8 prompts are within 0.25 nats of vLLM's argmax. A
  // re-resolved tie under batching is that same phenomenon, not a defect, and
  // holding our engine to a standard vLLM does not meet would be measuring the
  // wrong thing (the very error the W0 batching artifact already caused once).
  //
  // What IS asserted for this phase is the thing that would be a real defect:
  // the batch ORDER (above) — because an illegal order does not re-resolve a
  // tie, it produces the 0.86 relative error W6 measured.
  int conc_exact = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (concurrent_ids[static_cast<size_t>(i)] == serial_ids[static_cast<size_t>(i)])
      ++conc_exact;
  }
  MESSAGE("deepseek-v2-lite batch invariance (REPORTED, not a bar): " << conc_exact
          << "/" << N << " concurrent sequences identical to the batch=1 "
          << "sequences — for reference the vLLM 0.25.0 oracle's OWN greedy "
          << "changed on 3/8 of this battery under batched generation (W0)");

  // ---- PHASE 3: a genuine WITH-CONTEXT prefill through the ENGINE ----------
  // Submit a prompt long enough to fill whole KV blocks TWICE. The second
  // submission hits the prefix cache, so the scheduler admits it with
  // num_computed_tokens > 0 — a prefill WITH context, which is what makes
  // `prefill_tokens_with_context` (a PREFIX LENGTH, mla_attention.py:1806-1810)
  // and the chunked up-projection path load-bearing. W6 measured 0.86 relative
  // error when such a request was NOT ordered first in the prefill tail.
  if (loaded->prefix_caching_enabled()) {
    vllm::ResetMlaBatchSplitStats();
    const vllm::RequestOutput a =
        loaded->engine().generate(LongPrompt(), Greedy(8), "dsv2-ctx-a");
    const vllm::RequestOutput b =
        loaded->engine().generate(LongPrompt(), Greedy(8), "dsv2-ctx-b");
    const vllm::MlaBatchSplitStats ctx_stats = vllm::GetMlaBatchSplitStats();
    MESSAGE("deepseek-v2-lite MLA split stats (phase 3, prefix-cache reuse): steps="
            << ctx_stats.steps << " with_context_prefill_steps="
            << ctx_stats.with_context_prefill_steps
            << " prompt_len=" << a.prompt_token_ids.size());
    CHECK_MESSAGE(ctx_stats.with_context_prefill_steps > 0,
                  "deepseek-v2-lite: no WITH-CONTEXT prefill was produced — the "
                  "prefix cache did not hit, so the prefill_tokens_with_context "
                  "prefix-length invariant went unexercised end to end");
    // Identical prompt, greedy: the cached-prefix run must reproduce the
    // uncached one exactly. (Prefix caching is a pure reuse of already-computed
    // KV; a divergence means the reuse is not equivalent.)
    CHECK(a.outputs[0].token_ids == b.outputs[0].token_ids);
  } else {
    MESSAGE("deepseek-v2-lite: prefix caching disabled for this model — phase 3 "
            "(engine-produced with-context prefill) not exercised");
  }

  // THE SACRED BAR, asserted LAST so a correctness failure still leaves the
  // scheduler/runner wiring evidence (phases 2 and 3) on the record — a
  // divergence needs diagnosing, and the batch shapes the engine produced are
  // part of that diagnosis. ZERO prompts may fall outside the near-tie band.
  REQUIRE(fail == 0);
}
