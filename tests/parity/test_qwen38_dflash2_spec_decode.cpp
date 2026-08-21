// vllm.cpp original (checkpoint-gated spec-decode gate); no upstream mirror.
//
// THE DFlash2 e2e CORRECTNESS GATE (SPEC-DFLASH2 W6, #1314). This is the case
// the whole row exists to produce: G2 (draft-token identity against the
// beyond-pin oracle) and G3 (acceptance, measured SAME-TRAJECTORY).
//
// WHAT THIS GATES, AND WHY EACH BAR IS SHAPED THE WAY IT IS
//
// (G2) DRAFT-TOKEN identity, not only output identity. A DFlash2 draft that
//      proposed nothing useful still emits the target's tokens, because the
//      verify is lossless -- that is the silent-wrong this whole row is built to
//      remove, and an output-token gate cannot see it. So the bar reads the
//      PRODUCTION `[SPECTRACE]` line and compares the DRAFTS block by block
//      against the drafts upstream's own `DFlash2Speculator._generate_draft`
//      wrote into `self.draft_tokens`.
//
// (G3) ACCEPTANCE, SAME-TRAJECTORY. `SPEC-DFLASH` D8 spent an entire campaign
//      concluding a 0.85x acceptance deficit that D9 refuted as a
//      divergent-trajectory confound. This gate does not repeat that: a prompt
//      contributes to the acceptance comparison ONLY when the two engines
//      emitted the SAME token stream, which makes the trajectories identical by
//      construction rather than by assumption. A prompt that diverges is
//      reported with its divergence index and is EXCLUDED from the acceptance
//      bar rather than silently averaged into it.
//
// (VOID) TOKENIZER identity is a PRECONDITION, not a bar. If our prompt token
//      ids differ from the oracle's, the two engines were not fed the same
//      thing and no comparison below means anything. That is reported as a VOID
//      gate -- a hard failure with its own message -- rather than as a token
//      mismatch, because the two have different causes and different repairs.
//
// THE NEAR-TIE ENVELOPE APPLIES AND A STRICT CLAIM DOES NOT. `SPEC-DFLASH` D6
// established that strict token identity against vLLM is bf16-IRREDUCIBLE on
// portable kernels (an inline context-KV recompute envelope plus a from-scratch
// block attention against vLLM's flashinfer paged one), and the ratified
// near-tie form is what this lane gates. So a divergence is reported with its
// index and its shared prefix, and the bar is that a divergence is a LATE
// near-tie rather than a structural break at index 0.
//
// THE ORACLE IS BEYOND-PIN AND THE BACKEND IS CONSTRAINED. The parity pin
// carries no DFlash2 at all. The gate oracle is vLLM at
// 66e5414c6d75a8529473d977f7458c140bbab8a0 (vllm-project/vllm#52816), and it
// runs on TRITON_ATTN because vllm-flash-attn does not target sm_12x at that
// revision (#1456). Both facts are carried in the golden and asserted here, so
// a golden captured under a different oracle or backend cannot be read as this
// one.
//
// AND THAT HEAD IS A DATED EXCEPTION, NOT THE RULE. vllm#52816 MERGED on
// 2026-08-21 at 05:27:22Z, at head 3406ec1dae9916f920b90f0dbf90dcf54923d042 and
// merge commit b389ac29465b33f9e9c534df221ea3c129e9793f. `## Gates` G2's own
// rule is "66e5414c if #52816 has not merged, and the merge commit if it has",
// so the head the rule selects TODAY is b389ac29. This capture stays pinned to
// 66e5414c because that is the wheel that was built and run on the lease, and
// it predates the merge -- re-labelling a run with a head it never executed
// would be a false pin. Nothing asserted below is false; what a reader has to
// know is that the head asserted here is ONE MERGE BEHIND vLLM's main, and that
// moving it and re-reading G2 and G3 there is owed under
// https://github.com/mudler/vllm.cpp/issues/1561. The spec, the benchmark
// record, docs/STATUS.md and docs/BENCHMARKS.md all carry this; this file is
// the one surface that ASSERTS the head, so it carries it too.
//
// Checkpoint-GATED + dgx-only. On the CPU dev box and in CI the body emits a
// loud SKIP naming exactly what is absent (it still compiles and links).
#include <doctest/doctest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string Env(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? std::string(v) : std::string();
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

std::string Ids(const std::vector<int32_t>& v) {
  std::string s;
  for (const int32_t id : v) s += std::to_string(id) + " ";
  return s;
}

// `VT_SPEC_TRACE` is latched by a function-local `static` on the FIRST propose
// in the process, so setting it inside a case would race whichever case ran
// first. This runs before main.
const bool kSpecTraceEnabled = [] {
  ::setenv("VT_SPEC_TRACE", "1", 1);
  return true;
}();

// REAL fd 2, by dup/dup2, and not a `std::cerr` rdbuf swap: the trace is written
// with `std::fprintf(stderr, ...)`, which an rdbuf swap cannot see. A capture
// that could not see the line it exists to read would come back empty and look
// like "the propose did not run" -- the instrument failing toward a verdict
// about the code.
std::string CaptureStderr(const std::function<void()>& body) {
  std::FILE* cap = std::tmpfile();
  REQUIRE(cap != nullptr);
  std::fflush(stderr);
  const int saved = ::dup(STDERR_FILENO);
  REQUIRE(saved >= 0);
  REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);
  body();
  std::fflush(stderr);
  const int restored = ::dup2(saved, STDERR_FILENO);
  ::close(saved);
  std::rewind(cap);
  std::string out;
  char buf[8192];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) out.append(buf, n);
  std::fclose(cap);
  REQUIRE(restored >= 0);
  return out;
}

// One decode step that proposed, as the production verify-side trace reports it:
//   [SPECTRACE] req=<id> pos=<n> k=<k> ns=<n> acc=<a> draft=[ ... ] emit=[ ... ]
// `pos` is the request's token count BEFORE this step's write-back, so blocks
// from two runs line up by POSITION rather than by ordinal alone.
struct TraceBlock {
  std::string req;
  int pos = -1;
  int k = 0;
  int accepted = 0;
  std::vector<int32_t> draft;
  std::vector<int32_t> emit;
};

std::vector<int32_t> ParseIdList(const std::string& s) {
  std::vector<int32_t> out;
  std::istringstream is(s);
  int64_t v = 0;
  while (is >> v) out.push_back(static_cast<int32_t>(v));
  return out;
}

// Field lookup by NAME rather than by offset, so a future field added to the
// trace line does not silently shift what this reads.
bool ScalarField(const std::string& line, const std::string& key, int* out) {
  const std::string k = " " + key + "=";
  const size_t at = line.find(k);
  if (at == std::string::npos) return false;
  *out = std::atoi(line.c_str() + at + k.size());
  return true;
}

bool StringField(const std::string& line, const std::string& key, std::string* out) {
  const std::string k = " " + key + "=";
  const size_t at = line.find(k);
  if (at == std::string::npos) return false;
  const size_t open = at + k.size();
  const size_t close = line.find(' ', open);
  *out = line.substr(open, close == std::string::npos ? close : close - open);
  return true;
}

bool ListField(const std::string& line, const std::string& key,
               std::vector<int32_t>* out) {
  const std::string k = " " + key + "=[";
  const size_t at = line.find(k);
  if (at == std::string::npos) return false;
  const size_t open = at + k.size();
  const size_t close = line.find(']', open);
  if (close == std::string::npos) return false;
  *out = ParseIdList(line.substr(open, close - open));
  return true;
}

std::vector<TraceBlock> ParseSpecTrace(const std::string& captured) {
  std::vector<TraceBlock> out;
  std::istringstream is(captured);
  std::string line;
  while (std::getline(is, line)) {
    if (line.find("[SPECTRACE]") == std::string::npos) continue;
    if (line.find(" draft=[") == std::string::npos) continue;  // the propose-side line
    TraceBlock b;
    if (!StringField(line, "req", &b.req)) continue;
    if (!ScalarField(line, "pos", &b.pos)) continue;
    if (!ScalarField(line, "k", &b.k)) continue;
    if (!ScalarField(line, "acc", &b.accepted)) continue;
    if (!ListField(line, "draft", &b.draft)) continue;
    if (!ListField(line, "emit", &b.emit)) continue;
    out.push_back(std::move(b));
  }
  return out;
}

struct OurRun {
  std::vector<int32_t> out_ids;
  std::vector<int32_t> prompt_ids;
  std::vector<TraceBlock> blocks;
  int64_t proposed = 0;
  int64_t accepted = 0;
  std::string text;
};

// ONE ENGINE, EVERY PROMPT. Not a style choice: this target is 51.75 GiB off a
// CIFS mount and vLLM measured 522.9 s to read it once, so an engine per prompt
// would spend most of a GPU lease re-reading weights that did not change. The
// DFlash1 gate beside this one loads once for the same reason.
//
// The per-block trace is split back out by REQUEST ID rather than by ordinal,
// because one captured stream now carries every prompt's blocks. `req=` is a
// field of the production line, so the split is read off the instrument rather
// than reconstructed from call order.
std::vector<OurRun> RunDflash2All(const std::string& target, const std::string& draft,
                                  int k, const std::vector<std::string>& prompts,
                                  int max_tokens) {
  vllm::entrypoints::EngineParams params;
  params.max_num_seqs = 1;  // concurrency 1: the pooled acceptance IS per-request.
  params.speculative_config = vllm::ParseSpeculativeConfigJson(
      std::string("{\"method\":\"dflash\",\"model\":\"") + draft +
      "\",\"num_speculative_tokens\":" + std::to_string(k) + "}");

  std::vector<OurRun> runs(prompts.size());
  std::vector<std::string> ids(prompts.size());
  const std::string captured = CaptureStderr([&] {
    auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(target, params);
    for (size_t i = 0; i < prompts.size(); ++i) {
      ids[i] = "dflash2_" + std::to_string(i);
      const int64_t acc_before = loaded->runner().spec_drafts_accepted();
      const int64_t prop_before = loaded->runner().spec_drafts_proposed();
      const vllm::RequestOutput out =
          loaded->engine().generate(prompts[i], Greedy(max_tokens), ids[i]);
      OurRun& r = runs[i];
      r.out_ids = out.outputs.empty() ? std::vector<int32_t>{}
                                      : out.outputs[0].token_ids;
      r.text = out.outputs.empty() ? std::string() : out.outputs[0].text;
      r.prompt_ids = out.prompt_token_ids;
      // DELTAS, not totals. The counters are cumulative over the engine's life,
      // and reading a total as a per-prompt count is how a second prompt gets
      // credited with the first one's acceptance.
      r.accepted = loaded->runner().spec_drafts_accepted() - acc_before;
      r.proposed = loaded->runner().spec_drafts_proposed() - prop_before;
    }
  });

  const std::vector<TraceBlock> all = ParseSpecTrace(captured);
  for (size_t i = 0; i < runs.size(); ++i)
    for (const TraceBlock& b : all)
      if (b.req == ids[i]) runs[i].blocks.push_back(b);
  return runs;
}

// ACCEPTANCE, RECONSTRUCTED FROM THE DRAFTS AND THE OUTPUT ALONE.
//
// The verify here and upstream is ACCEPT-IFF-EQUAL under greedy sampling
// (`include/vllm/v1/spec_decode/rejection_sampler.h`), so a block's accepted
// count is a FUNCTION of what it proposed and what the request emitted: the
// longest prefix of the draft that the output actually took, at the position the
// block started from. That makes the ORACLE's per-block acceptance recoverable
// from a capture that recorded only drafts and output -- which is what this
// row's capture recorded, because vLLM exposes no per-block counter.
//
// The recovery is not trusted on the strength of that argument. It is run
// against OUR OWN blocks first, where the true per-block count is printed by the
// production trace, and the gate refuses to read the oracle's derived numbers
// unless the derivation reproduces ours exactly.
//
// `len` starts at 1: the prefill step emits one token before any block proposes.
// The last block can be TRUNCATED by max_tokens, so matching stops at the end of
// the output rather than at k -- measured, not assumed: on the first W6 capture
// `4 + 50 + 209 = 263` against 256 tokens actually generated, and the 7 missing
// are exactly that truncation.
struct Reconstructed {
  std::vector<int> per_block;
  int64_t total = 0;
  // Blocks that STARTED inside the output, i.e. the ones the run actually
  // verified. A capture records every `propose` call, and the last one or two of
  // a request are proposals the run never consumed because `max_tokens` had
  // already been reached. MEASURED 2026-08-21 on the oracle capture: 55 blocks
  // recorded, 47 of them starting inside the output, and vLLM's own
  // `spec_decode_num_drafts` reads exactly 47. So this -- not the raw count --
  // is the quantity vLLM counts, and the cross-check below is EXACT rather than
  // banded.
  int64_t verified = 0;
};

Reconstructed ReconstructAcceptance(const std::vector<std::vector<int32_t>>& drafts,
                                    const std::vector<int32_t>& out) {
  Reconstructed r;
  size_t len = 1;  // the prefill token
  for (const std::vector<int32_t>& d : drafts) {
    if (len < out.size()) ++r.verified;
    int acc = 0;
    for (size_t j = 0; j < d.size(); ++j) {
      if (len + j >= out.size()) break;                 // truncated by max_tokens
      if (out[len + j] != d[j]) break;                  // the first rejection
      ++acc;
    }
    r.per_block.push_back(acc);
    r.total += acc;
    len += 1 + static_cast<size_t>(acc);
  }
  return r;
}

// ── THE GOLDEN'S OWN LIVENESS PRECONDITION ──────────────────────────────────
//
// `draft_hook_installed: true` IS NOT LIVENESS, and reading it as liveness is
// how a drafts-less golden passes the one check that exists to stop exactly
// that. MEASURED 2026-08-21 on the committed FLASH_ATTN arm: it carries
// `draft_hook_installed: true` and `blocks: []` on all four records, because it
// was captured before the hook reached the REPLAYED graph (`## Owed` O23's third
// failure). Selecting it with `VLLM_DFLASH2_EXPECT_BACKEND=FLASH_ATTN` then
// drove the gate through the flag, found no block to pair, and reported
// "STRUCTURAL: not a single block pair was anchor-aligned inside a shared
// prefix" -- a sentence about OUR engine, on a run where our engine was never
// the thing missing.
//
// So the golden is asked whether it can answer G2b and G3 at all, BEFORE either
// is read out of it. A golden that carries no drafts makes those two gates VOID.
// It does NOT make them fail, and it does not make G2a (output identity) void,
// which such a golden can still answer.
struct GoldenDrafts {
  size_t records = 0;
  size_t with_blocks = 0;   // records carrying at least one block
  size_t total_blocks = 0;
  bool live = false;        // every record carries drafts
};

GoldenDrafts InspectGoldenDrafts(const json& golden) {
  GoldenDrafts g;
  if (!golden.contains("records")) return g;
  for (const auto& rec : golden.at("records")) {
    ++g.records;
    const size_t n = rec.contains("blocks") ? rec.at("blocks").size() : 0;
    g.total_blocks += n;
    if (n > 0) ++g.with_blocks;
  }
  // EVERY record, not the total. One drafts-less record among four would leave
  // the total positive and silently contribute nothing to G2b on that prompt.
  g.live = (g.records > 0 && g.with_blocks == g.records);
  return g;
}

size_t SharedPrefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

}  // namespace

TEST_CASE("qwen38 DFlash2 e2e gate: draft-token identity + same-trajectory acceptance") {
  (void)kSpecTraceEnabled;
  const std::string target = Env("VLLM_DFLASH2_TARGET");
  const std::string draft = Env("VLLM_DFLASH2_DRAFT");
  std::string golden_env = Env("VLLM_DFLASH2_GOLDEN");
  const fs::path golden_path =
      golden_env.empty()
          ? fs::path(PARITY_GOLDENS_DIR) / "dflash2_27b" / "dflash2_27b_spec_on.json"
          : fs::path(golden_env);

  if (target.empty() || draft.empty() || !fs::exists(golden_path)) {
    MESSAGE("SKIP (dgx-only): the DFlash2 e2e gate needs VLLM_DFLASH2_TARGET "
            "(the Qwen3.8-27B bf16 safetensors dir, HF revision "
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), VLLM_DFLASH2_DRAFT "
            "(z-lab/Qwen3.8-27B-DFlash2 @ "
            "50307d4c4cde6860d4eee73e2547cd786fe8e8a4, a directory or a .gguf), "
            "and the committed golden. Got target="
            << (target.empty() ? "ABSENT" : target)
            << " draft=" << (draft.empty() ? "ABSENT" : draft)
            << " golden=" << (fs::exists(golden_path) ? golden_path.string()
                                                      : std::string("ABSENT")));
    return;
  }

  std::ifstream gf(golden_path.string());
  REQUIRE(gf.good());
  json golden;
  gf >> golden;

  // THE GOLDEN'S OWN IDENTITY, asserted before anything is read out of it. A
  // golden captured against a different oracle revision or a different attention
  // backend describes a different measurement, and reading it as this one is how
  // a parity claim quietly stops meaning what it says.
  const std::string oracle_version = golden.value("oracle_version", std::string());
  const std::string backend = golden.value("attention_backend", std::string());
  // The DECLARED backend defaults to TRITON_ATTN, which is the developer's
  // recorded decision for this row's denominator (#1456). It is an env override
  // rather than a widened set: this row measured BOTH arms, and a run that wants
  // the FLASH_ATTN golden has to say so, so a golden from the other arm can
  // never be read silently as this one.
  std::string want_backend = Env("VLLM_DFLASH2_EXPECT_BACKEND");
  if (want_backend.empty()) want_backend = "TRITON_ATTN";
  MESSAGE("golden oracle_version=" << oracle_version << " backend=" << backend
          << " (expected " << want_backend << ")");
  // 66e5414c is the DATED exception recorded in the header above and owed under
  // #1561, not the head `## Gates` G2's rule selects today: #52816 merged
  // 2026-08-21T05:27:22Z at b389ac29, and this capture predates that merge.
  MESSAGE("ORACLE HEAD IS DATED: 66e5414c is vllm#52816's pre-merge head, the "
          "wheel this capture actually ran. #52816 MERGED 2026-08-21T05:27:22Z "
          "at merge commit b389ac29; moving this gate onto it and re-reading G2 "
          "and G3 there is owed under vllm.cpp#1561.");
  CHECK(oracle_version.find("66e5414c") != std::string::npos);
  CHECK(backend == want_backend);
  CHECK(golden.value("spec", std::string()) == "on");
  CHECK(golden.value("draft_hook_installed", false));

  // THE GOLDEN'S LIVENESS, read from the golden's own records rather than from
  // its `draft_hook_installed` flag. See `InspectGoldenDrafts`.
  const GoldenDrafts gd = InspectGoldenDrafts(golden);
  MESSAGE("golden drafts: " << gd.with_blocks << "/" << gd.records
          << " records carry blocks, " << gd.total_blocks << " blocks total -> "
          << (gd.live ? "G2b and G3 are TAKEABLE" : "G2b and G3 are VOID"));
  if (!gd.live) {
    MESSAGE("VOID for G2b/G3: this golden carries no per-block drafts on "
            << (gd.records - gd.with_blocks) << " of " << gd.records
            << " records, so there is nothing to pair a draft block against. "
               "That is a property of the CAPTURE, not a finding about this "
               "engine: it is what a capture taken before the hook reached the "
               "replayed graph looks like (`## Owed` O23). G2a (output "
               "identity) is still taken below; the draft-identity and "
               "acceptance bars are NOT, and reporting them as failures would "
               "name our engine for the oracle's missing instrument.");
  }

  // AND THE CAPTURE'S OWN BOOKKEEPING IS BOUNDED. `hook_stats` counts what the
  // instrument saw; the records are what it wrote down. A golden that lost
  // records between the two is a golden that under-reports the oracle's drafts,
  // which reads as a draft-identity finding. MEASURED on the committed
  // TRITON_ATTN arm: 59 propose calls, 1 skipped dummy, 0 skipped in capture,
  // and 55 recorded blocks -- so THREE calls are unaccounted for and the
  // residual is printed rather than absorbed (#1562). The bound is one-sided
  // because a skipped-but-uncounted call can only lose a record, never invent
  // one, and no wave has explained the residual well enough to close it.
  if (golden.contains("hook_stats")) {
    const auto& hs = golden.at("hook_stats");
    const int64_t calls = hs.value("propose_calls", static_cast<int64_t>(-1));
    const int64_t sk_dummy = hs.value("skipped_dummy", static_cast<int64_t>(0));
    const int64_t sk_cap = hs.value("skipped_capture", static_cast<int64_t>(0));
    if (calls >= 0) {
      const int64_t recorded = calls - sk_dummy - sk_cap;
      MESSAGE("golden hook_stats: " << calls << " propose calls - " << sk_dummy
              << " dummy - " << sk_cap << " capture = " << recorded
              << " expected records, against " << gd.total_blocks
              << " written (residual " << (recorded - static_cast<int64_t>(gd.total_blocks))
              << ", UNEXPLAINED, #1562)");
      CHECK(static_cast<int64_t>(gd.total_blocks) <= recorded);
    }
  }

  const int k = golden.at("num_speculative_tokens").get<int>();
  const int max_tokens = golden.at("max_tokens").get<int>();
  const auto& records = golden.at("records");
  REQUIRE(records.size() > 0);

  std::vector<std::string> prompts;
  for (const auto& rec : records) prompts.push_back(rec.at("prompt").get<std::string>());

  MESSAGE("dflash2 gate: loading the 27B target " << target << " + DFlash2 draft "
          << draft << " (k=" << k << ", max_tokens=" << max_tokens << ", "
          << prompts.size() << " prompts, ONE engine)...");
  const std::vector<OurRun> runs =
      RunDflash2All(target, draft, k, prompts, max_tokens);
  REQUIRE(runs.size() == records.size());

  int total = 0, exact = 0, same_traj = 0;
  int64_t draft_blocks_compared = 0, draft_blocks_identical = 0;
  int64_t our_acc_sum = 0, their_acc_sum = 0;
  int64_t oracle_acc_all = 0, oracle_blocks_all = 0;
  int64_t our_recon_sum = 0;
  // OUR verified block count, derived by the SAME routine from OUR drafts and
  // OUR output. `draft_blocks_compared` is NOT this number: it is bounded by the
  // `g > shared` cut below, so it counts blocks the gate could PAIR, not blocks
  // this engine verified. Recording both separately is what lets the ours-vs-
  // theirs block claim be asserted instead of printed.
  int64_t our_blocks_all = 0;
  int prompt_idx = 0;

  for (const auto& rec : records) {
    const std::string prompt = rec.at("prompt").get<std::string>();
    const std::vector<int32_t> want_prompt_ids =
        rec.at("prompt_token_ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> want_out =
        rec.at("output_token_ids").get<std::vector<int32_t>>();
    const OurRun& r = runs[static_cast<size_t>(prompt_idx)];
    ++total;
    ++prompt_idx;

    // ---- PRECONDITION: same input, or the whole comparison is VOID ----------
    if (r.prompt_ids != want_prompt_ids) {
      MESSAGE("VOID prompt[" << (prompt_idx - 1) << "]: our prompt token ids ("
              << Ids(r.prompt_ids) << ") differ from the oracle's ("
              << Ids(want_prompt_ids) << "). The two engines were not fed the "
                 "same thing, so nothing below this line is a parity result. "
                 "Repair the tokenizer before reading any bar.");
    }
    REQUIRE(r.prompt_ids == want_prompt_ids);

    // ---- PRECONDITION: the drafter is ALIVE --------------------------------
    // Token identity ALONE passes on a dead drafter (the I5e dead-drafter trap),
    // and a DFlash2 draft that fell back to the DFlash1 per-slot argmax would
    // also emit the target's tokens. So liveness is asserted before identity.
    CHECK(r.proposed > 0);
    CHECK(!r.blocks.empty());
    for (const TraceBlock& b : r.blocks) CHECK(b.k == k);

    // ---- G2a: OUTPUT tokens, under the ratified near-tie envelope -----------
    const bool is_exact = (r.out_ids == want_out);
    const size_t shared = SharedPrefix(r.out_ids, want_out);
    if (is_exact) ++exact;
    if (!is_exact) {
      // A structural break diverges at index 0; a bf16 near-tie does not. This
      // is the discriminator, and it is the only thing the envelope licenses.
      MESSAGE("prompt[" << (prompt_idx - 1) << "] output divergence at index "
              << shared << " of " << want_out.size()
              << "; ours=" << Ids(r.out_ids) << " theirs=" << Ids(want_out));
      CHECK(shared > 0);
    }

    // ---- G2b: DRAFT tokens, block by block ---------------------------------
    //
    // ALIGNMENT IS PROVED, NOT ASSUMED. Two engines that emit the same tokens
    // can still reach block i from different acceptance patterns (accept 3 then
    // 4 against 4 then 3), so pairing purely by ordinal would compare two blocks
    // that started at different positions and report the difference as a draft
    // defect. Every pair is therefore checked on the ANCHOR -- the verified
    // token the walk starts from, which the oracle's own `_generate_draft` reads
    // out of `input_buffers.input_ids[self._anchor_indices[:num_reqs]]` and which
    // our block's `pos` names in the same stream. A mismatch STOPS the walk for
    // that prompt and is reported; it is not silently skipped, because "the
    // blocks stopped lining up" and "the drafts differ" are different findings.
    //
    // And the comparison never runs past the SHARED PREFIX. A draft is a
    // function of the context before it, so a block starting at generated index
    // g is comparable only while g <= the shared prefix length: past that the
    // two engines are drafting from different text, and `SPEC-DFLASH` D8 spent a
    // campaign mistaking exactly that for an acceptance deficit.
    const auto& their_blocks = rec.at("blocks");
    const int64_t plen = static_cast<int64_t>(want_prompt_ids.size());
    std::vector<int32_t> full = want_prompt_ids;  // prompt + OUR continuation
    full.insert(full.end(), r.out_ids.begin(), r.out_ids.end());

    size_t identical_here = 0, compared_here = 0;
    bool anchor_broke = false;
    size_t stopped_at = 0;
    for (size_t b = 0; b < r.blocks.size() && b < their_blocks.size(); ++b) {
      const TraceBlock& ours = r.blocks[b];
      const int64_t g = static_cast<int64_t>(ours.pos) - plen;  // generated index
      stopped_at = b;
      if (g < 0 || g > static_cast<int64_t>(shared)) break;     // past the prefix
      // The anchor is the last token BEFORE this block, at absolute pos-1.
      if (ours.pos <= 0 || static_cast<size_t>(ours.pos) > full.size()) break;
      const int32_t our_anchor = full[static_cast<size_t>(ours.pos) - 1];
      const int64_t their_anchor =
          their_blocks[b].value("anchor", static_cast<int64_t>(-1));
      if (their_anchor >= 0 && their_anchor != static_cast<int64_t>(our_anchor)) {
        MESSAGE("  ALIGNMENT LOST at block[" << b << "] pos=" << ours.pos
                << ": our anchor " << our_anchor << " vs the oracle's "
                << their_anchor << ". Blocks past here are NOT compared, and "
                   "this is an alignment finding rather than a draft one.");
        anchor_broke = true;
        break;
      }
      const std::vector<int32_t> theirs =
          their_blocks[b].at("drafts").get<std::vector<int32_t>>();
      ++compared_here;
      if (ours.draft == theirs) {
        ++identical_here;
      } else if (compared_here - identical_here <= 3) {
        MESSAGE("  draft block[" << b << "] pos=" << ours.pos << " g=" << g
                << " ours=[" << Ids(ours.draft) << "] theirs=[" << Ids(theirs)
                << "]");
      }
      stopped_at = b + 1;
    }
    draft_blocks_compared += static_cast<int64_t>(compared_here);
    draft_blocks_identical += static_cast<int64_t>(identical_here);
    (void)anchor_broke;
    (void)stopped_at;

    int64_t our_recon_here = 0;

    // ---- THE RECONSTRUCTION IS VALIDATED ON OUR OWN BLOCKS FIRST -----------
    // If it cannot reproduce a count the production trace printed, it cannot be
    // trusted to produce the oracle's, and the gate says so instead of quoting a
    // derived number as a measured one.
    //
    // THE LAST BLOCK IS EXCLUDED FROM THE EQUALITY, and the reason is a real
    // state rather than a hedge. A request that hits `max_tokens` mid-block has
    // its emitted stream CUT, so the reconstruction -- which can only see the
    // tokens that survived -- reads fewer accepted than the counter, which
    // counted them before the cut. Measured on the first W6 capture: vLLM's own
    // arithmetic gives `4 prefills + 50 blocks + 209 accepted = 263` against 256
    // tokens actually returned, and the 7 missing are exactly this. So every
    // INTERIOR block must reproduce EXACTLY, the last may only UNDERCOUNT, and
    // the deficit is printed rather than absorbed.
    {
      std::vector<std::vector<int32_t>> our_drafts;
      for (const TraceBlock& b : r.blocks) our_drafts.push_back(b.draft);
      const Reconstructed check = ReconstructAcceptance(our_drafts, r.out_ids);
      REQUIRE(check.per_block.size() == r.blocks.size());
      const size_t interior = r.blocks.empty() ? 0 : r.blocks.size() - 1;
      size_t agree = 0;
      for (size_t b = 0; b < interior; ++b)
        if (check.per_block[b] == r.blocks[b].accepted) ++agree;
      const int64_t deficit = r.accepted - check.total;
      MESSAGE("  reconstruction check on OUR blocks: " << agree << "/" << interior
              << " INTERIOR per-block counts reproduced exactly; last block "
              << (r.blocks.empty() ? 0 : check.per_block.back()) << " against "
              << (r.blocks.empty() ? 0 : r.blocks.back().accepted)
              << "; total " << check.total << " against the trace's "
              << r.accepted << " (deficit " << deficit << ", expected 0.."
              << k << " from max_tokens truncation)");
      CHECK(agree == interior);
      if (!r.blocks.empty()) CHECK(check.per_block.back() <= r.blocks.back().accepted);
      CHECK(deficit >= 0);
      CHECK(deficit <= k);
      our_recon_here = check.total;
      our_blocks_all += check.verified;
    }

    // ---- G3: acceptance, SAME-TRAJECTORY ONLY ------------------------------
    // A prompt contributes here only when the two engines emitted the SAME
    // stream. That is what makes this a same-trajectory measurement rather than
    // the D8 confound, and a diverging prompt is EXCLUDED and named rather than
    // averaged in.
    //
    // The oracle's count is RECONSTRUCTED from its own drafts and its own
    // output by the routine validated three lines above, because vLLM exposes no
    // per-block counter and its aggregate is pooled over the whole run.
    std::vector<std::vector<int32_t>> their_drafts;
    for (const auto& tb : their_blocks)
      their_drafts.push_back(tb.at("drafts").get<std::vector<int32_t>>());
    const Reconstructed theirs_acc = ReconstructAcceptance(their_drafts, want_out);
    const int64_t their_acc = theirs_acc.total;
    oracle_acc_all += their_acc;
    oracle_blocks_all += theirs_acc.verified;
    if (is_exact) {
      ++same_traj;
      our_acc_sum += r.accepted;
      their_acc_sum += their_acc;
      our_recon_sum += our_recon_here;
    }

    // LIKE FOR LIKE, PER PROMPT. Our COUNTER sees the accepted tokens of a
    // block that `max_tokens` then truncated; a RECONSTRUCTION from the emitted
    // stream cannot, on either side. So the counter is only comparable to the
    // other engine's counter, and the reconstruction only to the other engine's
    // reconstruction -- and mixing them subtracts the truncation from one arm
    // alone. That is a small confound and it is exactly the shape of the one
    // `SPEC-DFLASH` D8 got wrong at a larger scale, so it is named and both
    // pairs are reported.
    if (is_exact && gd.live) {
      MESSAGE("  G3 prompt[" << (prompt_idx - 1) << "] RECONSTRUCTION vs "
              "RECONSTRUCTION: ours " << our_recon_here << " against the "
              "oracle's " << their_acc << std::string(
                  our_recon_here == their_acc ? "  (IDENTICAL)" : "  (DIFFER)"));
      // `std::string(...)`, not the bare ternary: doctest's stringifier prints a
      // `const char*` as its BOOL conversion, so the first run of this line read
      // "the oracle's 491" -- the count and a `1` for the message. Cosmetic, and
      // the CHECK below is what binds, but a garbled evidence line is how a
      // number gets misread later.
      CHECK(our_recon_here == their_acc);
    }

    MESSAGE("dflash2 prompt[" << (prompt_idx - 1) << "] \"" << prompt
            << "\": exact=" << (is_exact ? "YES" : "NO")
            << " shared=" << shared
            << "  our drafts " << r.accepted << "/" << r.proposed
            << " accepted over " << r.blocks.size() << " blocks"
            << "  (oracle blocks " << their_blocks.size()
            << ", oracle accepted " << their_acc << ")"
            << "  draft-blocks identical " << identical_here << "/"
            << compared_here
            << "  text=\"" << r.text << "\"");
  }

  MESSAGE("DFLASH2 G2 OUTPUT: " << exact << "/" << total << " prompts token-exact "
          "against the beyond-pin oracle (66e5414c, TRITON_ATTN) -- vllm#52816's "
          "pre-merge head, one merge behind vLLM main since "
          "2026-08-21T05:27:22Z; re-reading at the merged head is vllm.cpp#1561.");
  MESSAGE("DFLASH2 G2 DRAFTS: " << draft_blocks_identical << "/"
          << draft_blocks_compared << " draft blocks identical on the "
          << same_traj << " same-trajectory prompts.");
  MESSAGE("DFLASH2 G3 ACCEPTANCE (same-trajectory only, " << same_traj << "/"
          << total << " prompts). RECONSTRUCTION vs RECONSTRUCTION -- the "
          "comparable pair -- ours " << our_recon_sum << " against the oracle's "
          << their_acc_sum << ". Our COUNTER reads " << our_acc_sum
          << ", which includes the accepted tokens of blocks max_tokens then "
             "truncated and is comparable only to vLLM's own counter, below.");

  // THE RECONSTRUCTION, CROSS-CHECKED AGAINST vLLM'S OWN AGGREGATE COUNTER.
  // The per-block derivation was validated against OUR trace prompt by prompt;
  // this validates the same routine on the ORACLE's side, against a number vLLM
  // measured itself. Without it the oracle's per-prompt acceptance would rest
  // entirely on an argument about the verify rule.
  //
  // The counter is POOLED over the whole capture, which is exactly comparable
  // here because the capture ran at max_num_seqs 1 and generated the same four
  // prompts in the same order.
  if (golden.contains("metrics") && gd.live) {
    const auto& m = golden.at("metrics");
    const int64_t counted_acc =
        m.value("vllm:spec_decode_num_accepted_tokens", static_cast<int64_t>(-1));
    const int64_t counted_drafts =
        m.value("vllm:spec_decode_num_drafts", static_cast<int64_t>(-1));
    // The VERIFIED block count must match EXACTLY -- that is the quantity vLLM
    // counts, and `Reconstructed::verified` records why. The ACCEPTED count may
    // only undercount, by at most one truncated block per prompt, for the reason
    // recorded on our own side above.
    const int64_t max_deficit = static_cast<int64_t>(records.size()) * k;
    MESSAGE("ORACLE RECONSTRUCTION vs vLLM's OWN COUNTER: reconstructed "
            << oracle_acc_all << " accepted over " << oracle_blocks_all
            << " blocks; vLLM counted " << counted_acc << " over "
            << counted_drafts << " (deficit "
            << (counted_acc >= 0 ? counted_acc - oracle_acc_all : -1)
            << ", expected 0.." << max_deficit << " from max_tokens truncation)");
    if (counted_drafts >= 0) CHECK(oracle_blocks_all == counted_drafts);
    if (counted_acc >= 0) {
      CHECK(oracle_acc_all <= counted_acc);
      CHECK(counted_acc - oracle_acc_all <= max_deficit);
    }

    // OURS AGAINST THEIRS, ASSERTED RATHER THAN PRINTED. Through W6 both of
    // these were `MESSAGE` only, and the spec then read them as ours-vs-theirs
    // results. They are asserted here, and BOTH are guarded on
    // `same_traj == total`: vLLM's counters are POOLED over the whole capture,
    // so they are comparable to our per-prompt sums only when every prompt
    // contributed on both sides. On a partially diverging capture the numbers
    // measure different populations and the comparison is skipped and said so.
    if (same_traj == total) {
      // Our verified block count against the oracle's, both from the same
      // routine on each engine's own drafts and output. Chained with the
      // `oracle_blocks_all == counted_drafts` line above, this is what makes
      // "our 47 and vLLM's 47" an asserted identity rather than two printed
      // numbers that happen to agree.
      MESSAGE("OURS vs THEIRS, blocks: our reconstruction verifies "
              << our_blocks_all << " blocks against the oracle's "
              << oracle_blocks_all << " and vLLM's own counter's "
              << counted_drafts);
      CHECK(our_blocks_all == oracle_blocks_all);
      // COUNTER against COUNTER -- the other matched pair. Our runner's
      // cumulative `acc` sees the accepted tokens of a block `max_tokens` then
      // truncated, and so does vLLM's, which is exactly why these two are
      // comparable to each other and neither is comparable to a reconstruction.
      if (counted_acc >= 0) {
        MESSAGE("OURS vs THEIRS, counters: our runner's cumulative acceptance "
                << our_acc_sum << " against vLLM's "
                "spec_decode_num_accepted_tokens " << counted_acc);
        CHECK(our_acc_sum == counted_acc);
      }
    } else {
      MESSAGE("OURS vs THEIRS counter comparison SKIPPED: only " << same_traj
              << " of " << total << " prompts walked the same trajectory, and "
                 "vLLM's counters are pooled over the whole capture, so the two "
                 "sides would count different populations.");
    }
  }

  // HARD BARS.
  //
  // (1) At least one prompt walks the same trajectory END TO END. Without one,
  //     G3 IS NOT TAKEN -- and that is what this failure means. It is NOT an
  //     acceptance deficit, and the message says so, because reporting "G3
  //     failed" for "G3 could not be measured" is the confusion `SPEC-DFLASH`
  //     D8/D9 already paid for once.
  if (same_traj == 0) {
    MESSAGE("G3 NOT TAKEN: no prompt produced an identical token stream on both "
            "engines, so there is no end-to-end same-trajectory pair to compare "
            "acceptance over. This is a MEASUREMENT gap, not an acceptance "
            "result; the per-prompt shared prefixes above say how close each got, "
            "and the draft-block bar below still ran over the shared prefixes.");
  }
  CHECK(same_traj > 0);
  // (2) On a same-trajectory prompt the draft blocks are the oracle's drafts.
  //     This is the bar that a DFlash1-argmax fallback fails and an output-token
  //     gate cannot see. It is stated as a majority rather than as identity for
  //     the D6 reason: the lattice op is a REDUCTION and is specified
  //     within-envelope across backends, unlike the walk, so a late block can
  //     legitimately flip on a bf16 near-tie in the selector's rank contraction.
  //     A fallback does not produce a majority; it produces near-zero.
  // Zero comparable blocks means the two engines diverged before the FIRST
  // block's context, which is a structural break rather than a near-tie.
  //
  // AND BOTH RUN ONLY ON A GOLDEN THAT CARRIES DRAFTS. Without that guard a
  // drafts-less capture reports "STRUCTURAL" -- a verdict about this engine --
  // when what is missing is the oracle's instrument. `gd.live` is decided from
  // the golden's own records before any of this is read.
  if (!gd.live) {
    MESSAGE("G2b and G3 NOT TAKEN on this golden: it carries no per-block "
            "drafts, so draft identity and acceptance cannot be measured from "
            "it. This is a MEASUREMENT gap in the CAPTURE. G2a above still ran "
            "and is a result.");
  } else {
    if (draft_blocks_compared == 0) {
      MESSAGE("STRUCTURAL: not a single block pair was anchor-aligned inside a "
              "shared prefix. A bf16 near-tie diverges LATE; diverging before "
              "the first block does not.");
    }
    CHECK(draft_blocks_compared > 0);
    CHECK(draft_blocks_identical * 2 > draft_blocks_compared);
  }
  // (3) The drafter is alive on every prompt (checked in the loop) and our
  //     acceptance on the same-trajectory prompts is nonzero.
  CHECK(our_acc_sum > 0);
}

// ── THE GOLDENS' OWN LIVENESS, GATED ON EVERY BOX ───────────────────────────
//
// The e2e case above is dgx-only, so the rule it now enforces -- a golden that
// carries no drafts makes G2b and G3 VOID rather than failed -- would otherwise
// be unexecuted everywhere the gate actually runs. This case reads the two
// COMMITTED goldens directly and holds `InspectGoldenDrafts` against what each
// one really contains. It needs no checkpoint, no GPU and no engine.
//
// WHY IT IS WORTH A CASE. `draft_hook_installed` is `true` in BOTH goldens and
// one of them has `blocks: []` on every record, so the flag and the drafts
// disagree in the committed tree, today. A gate that read the flag as liveness
// would call that golden live.
TEST_CASE("dflash2 gate: a golden's drafts decide whether G2b and G3 are takeable") {
  const fs::path dir = fs::path(PARITY_GOLDENS_DIR) / "dflash2_27b";
  const fs::path triton = dir / "dflash2_27b_spec_on.json";
  const fs::path flash = dir / "dflash2_27b_spec_on_flash_attn.json";
  REQUIRE(fs::exists(triton));
  REQUIRE(fs::exists(flash));

  json gt, gf;
  {
    std::ifstream f(triton.string());
    REQUIRE(f.good());
    f >> gt;
  }
  {
    std::ifstream f(flash.string());
    REQUIRE(f.good());
    f >> gf;
  }

  // THE FLAG IS TRUE ON BOTH, which is the whole point of this case.
  CHECK(gt.value("draft_hook_installed", false));
  CHECK(gf.value("draft_hook_installed", false));

  // The TRITON_ATTN arm is the one the gates were read from: four records, all
  // carrying drafts, 55 blocks. LIVE.
  const GoldenDrafts t = InspectGoldenDrafts(gt);
  CHECK(t.records == 4);
  CHECK(t.with_blocks == 4);
  CHECK(t.total_blocks == 55);
  CHECK(t.live);

  // The FLASH_ATTN arm carries NO drafts at all, on any record, because it was
  // captured before the hook reached the replayed graph (`## Owed` O23). It can
  // still answer G2a; it cannot answer G2b or G3, and `live` says so.
  const GoldenDrafts f = InspectGoldenDrafts(gf);
  CHECK(f.records == 4);
  CHECK(f.with_blocks == 0);
  CHECK(f.total_blocks == 0);
  CHECK(!f.live);

  // LIVENESS IS PER RECORD, NOT A TOTAL. One drafts-less record among four
  // leaves the total positive while that prompt contributes nothing to G2b, so
  // a total-based rule would read a partly blind capture as fully live. Built
  // from the real golden so the shape is the shape the gate meets.
  json partial = gt;
  partial["records"][2]["blocks"] = json::array();
  const GoldenDrafts pg = InspectGoldenDrafts(partial);
  CHECK(pg.records == 4);
  CHECK(pg.with_blocks == 3);
  CHECK(pg.total_blocks > 0);   // a total-based rule would call this live
  CHECK(!pg.live);              // and this one does not

  // AND THE hook_stats RESIDUAL IS REAL, not a rounding of it. 59 propose calls
  // less 1 dummy less 0 capture-skips is 58 expected records against 55
  // written: three unaccounted for, tracked by #1562. The gate's bound is
  // one-sided, and what pins it is the three `hs.value(...)` CHECKs -- they read
  // the golden and red if any of the three numbers moves.
  //
  // WHAT THE RESIDUAL LINE DOES AND DOES NOT ADD, stated exactly because an
  // earlier revision of this comment said the bound and the residual were pinned
  // so "neither can drift unnoticed", which is one assertion too generous.
  // `expected_records` is a LITERAL (59 - 1 - 0), written out so the arithmetic
  // is readable, not re-derived from `hs`. So mutating the golden's
  // `propose_calls` from 59 to 58 reds exactly ONE assertion -- the
  // `hs.value("propose_calls", -1) == 59` above -- and the two lines below stay
  // green, because against a literal 58 they only restate `t.total_blocks == 55`,
  // which the case already checks. The drift is caught; it is caught once, by the
  // `hs` pin, and not twice.
  REQUIRE(gt.contains("hook_stats"));
  const auto& hs = gt.at("hook_stats");
  CHECK(hs.value("propose_calls", -1) == 59);
  CHECK(hs.value("skipped_dummy", -1) == 1);
  CHECK(hs.value("skipped_capture", -1) == 0);
  const int64_t expected_records = 59 - 1 - 0;
  CHECK(static_cast<int64_t>(t.total_blocks) <= expected_records);
  CHECK(expected_records - static_cast<int64_t>(t.total_blocks) == 3);
}

// ── THE INSTRUMENT'S OWN GATE ───────────────────────────────────────────────
//
// `ParseSpecTrace` is the whole of G2's draft-token evidence, and it runs on a
// dgx-only path where a silent defect in it would present as a verdict about
// the CODE: an empty block list reads as "the propose did not run", and a
// mis-parsed field reads as "the drafts differ". So it is gated here, on every
// box, against literals rather than against a live engine.
//
// THE FIXTURE SPANS THE FIELDS IT DECODES. Every scalar takes a DIFFERENT value
// (pos 41, k 7, ns 5, acc 4) so a parser that read the wrong one cannot answer
// correctly by coincidence; the draft ids are not a prefix of the emit ids and
// neither list is the identity sequence; and one id is negative and one is past
// 2^16, because the production field is an int32 token id and a fixture that
// only ever holds small positives cannot detect a narrow parse.
TEST_CASE("dflash2 gate instrument: the production SPECTRACE line parses field by field") {
  // Byte-for-byte the shape `GPUModelRunner::sample_tokens_with_rejection`
  // writes (`src/vllm/v1/worker/gpu/runner.cpp`), trailing space inside each
  // bracket included.
  const std::string captured =
      "some unrelated stderr chatter\n"
      "[SPECTRACE] req=r0 pos=41 k=7 ns=5 acc=4 draft=[ 11 -3 70000 5 5 9 12 ] "
      "emit=[ 11 -3 70000 5 88 ]\n"
      "[SPECTRACE] req=r0 pos=46 k=7 ns=1 acc=0 draft=[ 1 2 3 4 5 6 7 ] "
      "emit=[ 99 ]\n";

  const std::vector<TraceBlock> blocks = ParseSpecTrace(captured);
  REQUIRE(blocks.size() == 2);

  // The REQUEST ID is load-bearing now that one captured stream carries every
  // prompt's blocks: the gate splits by it, so a parser that dropped it would
  // hand prompt 0 every prompt's drafts.
  CHECK(blocks[0].req == "r0");
  CHECK(blocks[1].req == "r0");
  CHECK(blocks[0].pos == 41);
  CHECK(blocks[0].k == 7);
  CHECK(blocks[0].accepted == 4);
  CHECK(blocks[0].draft == std::vector<int32_t>{11, -3, 70000, 5, 5, 9, 12});
  CHECK(blocks[0].emit == std::vector<int32_t>{11, -3, 70000, 5, 88});

  CHECK(blocks[1].pos == 46);
  CHECK(blocks[1].accepted == 0);
  CHECK(blocks[1].draft == std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7});
  CHECK(blocks[1].emit == std::vector<int32_t>{99});

  // A line that is not a propose trace contributes nothing. Both halves matter:
  // a parser that accepted the first would inflate the block count, and one that
  // rejected the second by looking for the wrong key would report zero blocks
  // from a live run and read as "the propose did not run".
  CHECK(ParseSpecTrace("[SPECTRACE] req=r0 something else\n").empty());
  CHECK(ParseSpecTrace("").empty());

  // NAME-KEYED, not offset-keyed. `k=` must not be answered by the `k` inside
  // `req=`, and a field appended after `emit` must not shift anything.
  const std::vector<TraceBlock> reordered = ParseSpecTrace(
      "[SPECTRACE] req=knsacc pos=3 k=2 ns=2 acc=1 draft=[ 7 8 ] emit=[ 7 9 ] "
      "extra=[ 1 2 3 ]\n");
  REQUIRE(reordered.size() == 1);
  CHECK(reordered[0].pos == 3);
  CHECK(reordered[0].k == 2);
  CHECK(reordered[0].accepted == 1);
  CHECK(reordered[0].draft == std::vector<int32_t>{7, 8});
  CHECK(reordered[0].emit == std::vector<int32_t>{7, 9});

  // TWO REQUESTS INTERLEAVED, which is the shape the gate's one captured stream
  // actually has. The ids differ only in their last character, and one is a
  // PREFIX of nothing else, so a split that compared loosely would mix them.
  const std::vector<TraceBlock> two = ParseSpecTrace(
      "[SPECTRACE] req=dflash2_0 pos=5 k=2 ns=3 acc=2 draft=[ 4 5 ] emit=[ 4 5 6 ]\n"
      "[SPECTRACE] req=dflash2_1 pos=5 k=2 ns=1 acc=0 draft=[ 7 8 ] emit=[ 9 ]\n"
      "[SPECTRACE] req=dflash2_0 pos=8 k=2 ns=2 acc=1 draft=[ 1 2 ] emit=[ 1 3 ]\n");
  REQUIRE(two.size() == 3);
  int n0 = 0, n1 = 0;
  for (const TraceBlock& b : two) {
    if (b.req == "dflash2_0") ++n0;
    if (b.req == "dflash2_1") ++n1;
  }
  CHECK(n0 == 2);
  CHECK(n1 == 1);
  CHECK(two[2].draft == std::vector<int32_t>{1, 2});
}

// ── THE RECONSTRUCTION'S OWN GATE ───────────────────────────────────────────
//
// `ReconstructAcceptance` is the only source of the ORACLE's per-block accepted
// counts, because vLLM exposes no per-block counter. In the e2e case it is
// validated twice against real measurements -- against our production trace per
// prompt, and against vLLM's own aggregate counter -- but both of those live on
// a dgx-only path. This pins it everywhere, on literals.
//
// THE FIXTURE SPANS THE OUTCOMES rather than repeating one. Block 0 accepts
// EVERY draft (so the `len` advance is k+1 and an off-by-one shows), block 1
// accepts NONE (so a routine that credited the bonus token would read 1), block
// 2 accepts a strict middle prefix, and block 3 is TRUNCATED by the output
// ending mid-draft -- which is a real state, measured at 7 tokens on the first
// W6 capture, not a hypothetical. No two blocks share an accepted count.
TEST_CASE("dflash2 gate instrument: acceptance reconstructs from drafts and output") {
  // len starts at 1 (the prefill token).
  //   out[0]            = 100                     prefill
  //   block0 drafts 1 2 -> out[1]=1, out[2]=2      accept 2, then bonus out[3]=7
  //   block1 drafts 9 9 -> out[4]=5                accept 0, bonus is out[4]
  //   block2 drafts 5 8 -> out[5]=5 then out[6]!=8 accept 1, bonus out[7]=11
  //   block3 drafts 4 4 -> out[8]=4 and the output ENDS: accept 1, truncated
  //   out[0]            = 100                     prefill, len = 1
  //   block0 drafts 1 2 -> out[1]=1 out[2]=2       accept 2, bonus out[3]=7, len 4
  //   block1 drafts 9 9 -> out[4]=5 != 9           accept 0, bonus out[4],  len 5
  //   block2 drafts 5 8 -> out[5]=5, out[6]=6 != 8 accept 1, bonus out[6],  len 7
  //   block3 drafts 4 4 -> out[7]=4, then the OUTPUT ENDS: accept 1, TRUNCATED
  const std::vector<int32_t> out = {100, 1, 2, 7, 5, 5, 6, 4};
  const std::vector<std::vector<int32_t>> drafts = {
      {1, 2}, {9, 9}, {5, 8}, {4, 4}};

  const Reconstructed r = ReconstructAcceptance(drafts, out);
  REQUIRE(r.per_block.size() == 4);
  CHECK(r.per_block[0] == 2);
  CHECK(r.per_block[1] == 0);
  CHECK(r.per_block[2] == 1);
  CHECK(r.per_block[3] == 1);
  CHECK(r.total == 4);
  // All four started inside an 8-token output, so all four are verified.
  CHECK(r.verified == 4);

  // A block that starts PAST the end of the output is a proposal the run never
  // consumed. It contributes 0 accepted and, critically, does NOT count as
  // verified -- which is what makes the cross-check against vLLM's own
  // `spec_decode_num_drafts` exact rather than banded.
  const Reconstructed tail = ReconstructAcceptance(
      {{1, 2}, {9, 9}, {5, 8}, {4, 4}, {6, 6}, {7, 7}}, out);
  CHECK(tail.per_block.size() == 6);
  CHECK(tail.verified == 4);
  CHECK(tail.total == 4);

  // THE BOUNDARY ITSELF, which the `tail` case above does NOT reach. `verified`
  // is the whole basis of `our_blocks_all == oracle_blocks_all`, and it turns on
  // `len < out.size()`. In `tail` the unconsumed blocks start at len 9 against
  // an 8-token output, so `<` and `<=` answer identically there and the
  // comparison itself was never gated -- MEASURED by the fresh review, which
  // mutated `<` to `<=` and left the whole suite green.
  //
  // This case puts a block at EXACTLY `out.size()`. Block 0 accepts both drafts,
  // so len advances 1 -> 1 + 1 + 2 = 4, and the output is 4 tokens long: block 1
  // proposed from a position the run never emitted from, and it is NOT verified.
  // With `<` this reads verified 1; with `<=` it reads 2, and the CHECK below
  // reds.
  const std::vector<int32_t> at_end = {100, 1, 2, 7};
  const Reconstructed edge = ReconstructAcceptance({{1, 2}, {9, 9}}, at_end);
  REQUIRE(edge.per_block.size() == 2);
  CHECK(edge.verified == 1);
  CHECK(edge.per_block[0] == 2);
  CHECK(edge.per_block[1] == 0);
  CHECK(edge.total == 2);

  // The `len` advance is 1 + accepted, not accepted. A routine that forgot the
  // bonus token would read block 2's draft against out[4] instead of out[5] and
  // answer 0 there; the literals above are built so that difference is visible.
  // MEASURED, not asserted: with `len += 1 + accepted` this fixture reads
  // [2, 0, 2] / total 4, and with `len += accepted` it reads [2, 0, 0] / total 2.
  const std::vector<int32_t> shifted = {100, 1, 2, 7, 5, 5, 8, 11, 4};
  const Reconstructed r2 = ReconstructAcceptance({{1, 2}, {9, 9}, {5, 8}}, shifted);
  REQUIRE(r2.per_block.size() == 3);
  CHECK(r2.per_block[0] == 2);
  CHECK(r2.per_block[1] == 0);
  CHECK(r2.per_block[2] == 2);
  CHECK(r2.total == 4);

  // An empty draft list and an empty output are both real edge states on a
  // prompt that never proposed, and neither may throw.
  CHECK(ReconstructAcceptance({}, out).total == 0);
  CHECK(ReconstructAcceptance(drafts, {}).total == 0);
}
