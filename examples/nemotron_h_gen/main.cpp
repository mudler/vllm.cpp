// nemotron-h-gen — THIN PUBLIC-ABI CLIENT (ONE SURFACE / ARCH-ONE-SURFACE).
//
// The Nemotron-3.5-Lightning-30B (`NemotronHForCausalLM`) greedy token battery
// against the PINNED vLLM ORACLE golden, driven ENTIRELY through the flat C ABI
// (include/vllm.h): `vllm_engine_load` builds the full engine — the A2 weight
// loader's 18487 tensors in the format the checkpoint ships them in, plus the
// shared paged runner whose caches select `NemotronHPagedForward` — and
// `vllm_complete_tokens` (ABI v13) generates from the golden's pre-tokenized
// prompts. This file includes "vllm.h" and NOTHING else from the project, links
// `vllm::shared`, and is deliberately NOT on `scripts/example-abi-allowlist.txt`:
// per .agents/specs/nemotron-h-abi-e2e.md §6.1 it is modelled on
// examples/kimi_linear_gen, never on deepseek_v4_gen or laguna_gen, both of
// which drive a bespoke forward through internal headers and are the transition
// state that allowlist exists to retire.
//
//   nemotron-h-gen --model <hf-snapshot-dir> --golden <oracle.json>
//                  [--steps N] [--prompts M] [--max-model-len N] [--load-only]
//
// The golden is the A3 gate's operand:
// tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json, captured from
// the pinned oracle (`vllm 0.23.1rc1.dev1511+g555967922`, the setuptools_scm
// spelling of pin 5559679229bc) at `temperature 0.0, max_tokens 32` over three
// prompts. It carries `prompt_token_ids` per prompt, so this driver needs no
// tokenizer agreement established first, and `token_ids`, the 32 tokens each
// prompt must reproduce.
//
// ── WHY THIS PRINTS COUNTS, LOUDLY ──────────────────────────────────────────
// A comparison over zero elements reports a perfect score. This driver
// therefore refuses rather than passes when it compared nothing: it asserts a
// NON-ZERO prompt count and, per prompt, that the number of tokens actually
// compared equals the golden's own width. `--steps` below that width is a
// short run and is reported as such, never as a match.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm.h"

namespace {

// ── A deliberately small reader for THIS golden's shape ─────────────────────
// The golden is a fixed, committed document, not arbitrary JSON: an object with
// a "golden" array whose entries carry "prompt", "prompt_token_ids" and
// "token_ids". A full JSON parser in an example would be a second
// implementation to maintain; a scanner that finds a named key and reads the
// integer array or string that follows it is enough, and it FAILS LOUDLY on a
// shape it does not recognise rather than returning an empty vector that would
// read downstream as "nothing to compare".

std::string ReadWholeFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Position just past the closing quote of the key `"<key>"` at or after `from`.
size_t FindKey(const std::string& s, const std::string& key, size_t from) {
  const std::string needle = "\"" + key + "\"";
  return s.find(needle, from);
}

// Read the integer array that follows `"<key>":` starting the search at `from`.
// Returns the parsed values and sets `end` past the closing bracket.
std::vector<int32_t> ReadIntArrayAfter(const std::string& s,
                                       const std::string& key, size_t from,
                                       size_t* end) {
  const size_t k = FindKey(s, key, from);
  if (k == std::string::npos)
    throw std::runtime_error("golden: key \"" + key + "\" not found");
  const size_t open = s.find('[', k);
  if (open == std::string::npos)
    throw std::runtime_error("golden: \"" + key + "\" is not an array");
  const size_t close = s.find(']', open);
  if (close == std::string::npos)
    throw std::runtime_error("golden: \"" + key + "\" array is unterminated");
  std::vector<int32_t> out;
  size_t i = open + 1;
  while (i < close) {
    while (i < close && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' ||
                         s[i] == '\t' || s[i] == ','))
      ++i;
    if (i >= close) break;
    size_t j = i;
    if (s[j] == '-') ++j;
    while (j < close && s[j] >= '0' && s[j] <= '9') ++j;
    if (j == i)
      throw std::runtime_error("golden: non-numeric entry in \"" + key + "\"");
    out.push_back(static_cast<int32_t>(std::atoll(s.substr(i, j - i).c_str())));
    i = j;
  }
  if (end != nullptr) *end = close + 1;
  return out;
}

// Read the string value that follows `"<key>":`. Used for the provenance lines
// this driver echoes so the run's evidence names the oracle and the revision it
// is being held to, rather than only the tokens.
std::string ReadStringAfter(const std::string& s, const std::string& key,
                            size_t from) {
  const size_t k = FindKey(s, key, from);
  if (k == std::string::npos) return "";
  const size_t colon = s.find(':', k);
  if (colon == std::string::npos) return "";
  const size_t open = s.find('"', colon);
  if (open == std::string::npos) return "";
  const size_t close = s.find('"', open + 1);
  if (close == std::string::npos) return "";
  return s.substr(open + 1, close - open - 1);
}

// Read the boolean value that follows `"<key>":`, as a tri-state: 1 true,
// 0 false, -1 the key is absent. Absent is its own answer here and must not
// collapse into false -- "the golden says its configuration is unrecorded" and
// "the golden says nothing at all" are exactly the two states #926 separates.
int ReadBoolAfter(const std::string& s, const std::string& key, size_t from) {
  const size_t k = FindKey(s, key, from);
  if (k == std::string::npos) return -1;
  const size_t colon = s.find(':', k);
  if (colon == std::string::npos) return -1;
  const size_t t = s.find("true", colon);
  const size_t f = s.find("false", colon);
  const size_t stop = s.find(',', colon);
  const bool t_ok = t != std::string::npos && (stop == std::string::npos || t < stop);
  const bool f_ok = f != std::string::npos && (stop == std::string::npos || f < stop);
  if (t_ok && (!f_ok || t < f)) return 1;
  if (f_ok) return 0;
  return -1;
}

struct GoldenEntry {
  std::vector<int32_t> prompt_token_ids;
  std::vector<int32_t> token_ids;
};

struct Golden {
  std::string vllm_version;
  std::string model;
  std::string revision;
  // #926: whether the golden records the ENGINE CONFIGURATION it was captured
  // under. 1 recorded, 0 recorded-as-unrecorded, -1 the file does not say.
  int capture_recorded = -1;
  std::string capture_issue;
  std::vector<GoldenEntry> entries;
};

Golden ReadGolden(const std::string& path) {
  const std::string s = ReadWholeFile(path);
  Golden g;
  g.vllm_version = ReadStringAfter(s, "vllm", 0);
  g.model = ReadStringAfter(s, "model", 0);
  g.revision = ReadStringAfter(s, "revision", 0);
  const size_t cap = FindKey(s, "capture", 0);
  if (cap != std::string::npos) {
    g.capture_recorded = ReadBoolAfter(s, "engine_config_recorded", cap);
    g.capture_issue = ReadStringAfter(s, "issue", cap);
  }
  const size_t arr = FindKey(s, "golden", 0);
  if (arr == std::string::npos)
    throw std::runtime_error("golden: no \"golden\" array in " + path);
  size_t cursor = arr;
  // Each entry is delimited by its own "prompt_token_ids"; when that key stops
  // appearing the array is exhausted. This is why a truncated golden yields
  // FEWER entries rather than a silently empty one.
  while (true) {
    const size_t p = FindKey(s, "prompt_token_ids", cursor);
    if (p == std::string::npos) break;
    GoldenEntry e;
    size_t after_prompt = 0;
    e.prompt_token_ids = ReadIntArrayAfter(s, "prompt_token_ids", p, &after_prompt);
    size_t after_tokens = 0;
    e.token_ids = ReadIntArrayAfter(s, "token_ids", after_prompt, &after_tokens);
    if (e.prompt_token_ids.empty())
      throw std::runtime_error("golden: an entry has an EMPTY prompt_token_ids");
    if (e.token_ids.empty())
      throw std::runtime_error("golden: an entry has an EMPTY token_ids");
    g.entries.push_back(std::move(e));
    cursor = after_tokens;
  }
  return g;
}

std::string JoinIds(const std::vector<int32_t>& v, int n) {
  std::string out;
  for (int i = 0; i < n && i < static_cast<int>(v.size()); ++i) {
    out += std::to_string(v[static_cast<size_t>(i)]);
    if (i + 1 < n) out += ",";
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, golden_path;
  // The golden's own sampling block is `{temperature: 0.0, max_tokens: 32}`;
  // 32 is the default here so the gate's shape comes from the oracle rather
  // than from a flag somebody has to remember to pass.
  int steps = 32;
  int prompts = 0;  // 0 => every entry the golden carries
  // The battery's longest prompt is 13 tokens plus 32 continuations. A bounded
  // max_model_len keeps the engine's per-request token tables sized for the
  // battery rather than for the checkpoint's full context.
  int max_model_len = 4096;
  bool load_only = false;
  // Parse the golden and print its geometry WITHOUT loading a model. It exists
  // because the alternative way to find out whether this driver reads the
  // golden correctly is a 20.1 GiB load, and because the geometry it prints —
  // entry count and per-entry widths — is the number every count assertion
  // below is measured against.
  bool golden_info = false;
  // ── #1157 DISCRIMINATOR ────────────────────────────────────────────────────
  // Generate the SAME token stream WITHOUT ever taking a decode step: ask for
  // one token at a time, each from a FRESH request whose prompt is the original
  // prompt plus every token generated so far. Every token then comes out of a
  // PREFILL, and the recurrent state and paged KV each start empty.
  //
  // It is the same engine, the same weights and the same public entry point, so
  // the two runs differ in exactly one thing: whether a token was produced by
  // continuing a sequence or by recomputing it. A stream that is right this way
  // and wrong the normal way localises the defect to the CARRY and rules out
  // the tower; a stream that is wrong BOTH ways rules the carry out instead.
  bool fresh_prefill = false;
  // Run BOTH modes over one engine load, which is the only affordable shape
  // when a load is minutes long: the two streams then differ in nothing except
  // whether a token came from a decode step or from a re-prefill.
  bool both_modes = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--golden") golden_path = next();
    else if (a == "--steps") steps = std::atoi(next());
    else if (a == "--prompts") prompts = std::atoi(next());
    else if (a == "--max-model-len") max_model_len = std::atoi(next());
    else if (a == "--load-only") load_only = true;
    else if (a == "--golden-info") golden_info = true;
    else if (a == "--fresh-prefill") fresh_prefill = true;
    else if (a == "--both-modes") both_modes = true;
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (model.empty() && !golden_info) {
    std::fprintf(stderr,
                 "usage: --model <hf-snapshot-dir> --golden <oracle.json> "
                 "[--steps N] [--prompts M] [--max-model-len N] [--load-only]\n"
                 "       --golden <oracle.json> --golden-info   (parse only, no "
                 "model)\n");
    return 2;
  }
  if (golden_info && golden_path.empty()) {
    std::fprintf(stderr, "--golden-info needs --golden <oracle.json>\n");
    return 2;
  }

  std::fprintf(stderr, "[nemotron-h] libvllm %s (ABI %d, header %d)\n",
               vllm_version(), vllm_abi_version(), VLLM_ABI_VERSION);
  if (!model.empty())
    std::fprintf(stderr, "[nemotron-h] model dir: %s\n", model.c_str());

  Golden gold;
  if (!golden_path.empty()) {
    try {
      gold = ReadGolden(golden_path);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[nemotron-h] golden FAILED to parse: %s\n", e.what());
      return 2;
    }
    // Provenance, printed BEFORE the run: a token match is only evidence about
    // the oracle if the reader can see which oracle and which revision.
    std::fprintf(stderr,
                 "[nemotron-h] golden %s\n"
                 "             oracle vllm=%s  oracle model=%s  revision=%s\n"
                 "             entries=%zu\n",
                 golden_path.c_str(), gold.vllm_version.c_str(),
                 gold.model.c_str(), gold.revision.c_str(), gold.entries.size());
    // ── What this run is being held to (#926) ──────────────────────────────
    // The tokens below are a difference from a REFERENCE, and a difference is
    // only a defect once the reference can be regenerated. This golden's
    // capture configuration was never recorded and its capture host has since
    // been reimaged, so this line prints beside every score taken against it.
    if (gold.capture_recorded == 1) {
      std::fprintf(stderr,
                   "             capture: engine configuration RECORDED\n");
    } else if (gold.capture_recorded == 0) {
      std::fprintf(stderr,
                   "             capture: engine configuration UNRECORDED — a "
                   "token difference below is UNATTRIBUTABLE, not yet a defect%s%s\n",
                   gold.capture_issue.empty() ? "" : "; owed by ",
                   gold.capture_issue.c_str());
    } else {
      // Two different files reach this arm and the message must not name
      // either: a golden with NO `capture` block at all (the af8170154 shape),
      // and a golden that HAS one whose `engine_config_recorded` flag is
      // missing or unreadable. The tri-state is what separates them from
      // "unrecorded"; it does not separate them from each other, and a
      // parenthetical that picked one was wrong for the other.
      std::fprintf(stderr,
                   "             capture: the golden does not SAY whether its "
                   "engine configuration was recorded (no `capture` block, or "
                   "one with no readable `engine_config_recorded` flag)\n");
    }
    if (gold.entries.empty()) {
      std::fprintf(stderr,
                   "[nemotron-h] REFUSING: the golden carries ZERO entries, so "
                   "a comparison here would report a perfect score over nothing\n");
      return 2;
    }
    // The geometry every count assertion below is measured against, printed
    // per entry so a truncated or mis-parsed golden is visible BEFORE a load
    // rather than as a suspiciously small "compared" number afterwards.
    for (size_t i = 0; i < gold.entries.size(); ++i) {
      std::fprintf(stderr,
                   "             entry %zu: prompt_token_ids=%zu golden_width=%zu\n",
                   i, gold.entries[i].prompt_token_ids.size(),
                   gold.entries[i].token_ids.size());
    }
  }
  if (golden_info) return 0;

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = model.c_str();
  mp.max_model_len = max_model_len;
  vllm_engine* eng = nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  const vllm_status lst = vllm_engine_load(&mp, &eng);
  const auto t1 = std::chrono::steady_clock::now();
  if (lst != VLLM_OK) {
    std::fprintf(stderr, "[nemotron-h] engine load FAILED: %s\n",
                 vllm_last_error());
    return 1;
  }
  std::fprintf(stderr, "[nemotron-h] engine loaded in %.1fs\n",
               std::chrono::duration<double>(t1 - t0).count());
  if (load_only) {
    vllm_engine_free(eng);
    return 0;
  }
  if (golden_path.empty()) {
    std::fprintf(stderr,
                 "[nemotron-h] no --golden given: loaded only, nothing compared\n");
    vllm_engine_free(eng);
    return 0;
  }

  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;  // greedy — the oracle's own sampling block
  sp.max_tokens = steps;
  // The golden runs to a fixed max_tokens, so an EOS inside the window would
  // truncate our side and compare fewer tokens than the oracle recorded.
  sp.ignore_eos = 1;

  const int n_prompts =
      (prompts > 0) ? std::min<int>(prompts, static_cast<int>(gold.entries.size()))
                    : static_cast<int>(gold.entries.size());

  int total_compared = 0, total_matched = 0, rows_full = 0, rows_short = 0;
  const int n_modes = both_modes ? 2 : 1;
  for (int mi = 0; mi < n_modes; ++mi) {
  if (both_modes) {
    fresh_prefill = (mi == 1);
    total_compared = 0; total_matched = 0; rows_full = 0; rows_short = 0;
    std::fprintf(stderr, "\n[nemotron-h] ===== MODE %s =====\n",
                 fresh_prefill ? "fresh-prefill" : "decode");
  }
  for (int pi = 0; pi < n_prompts; ++pi) {
    const GoldenEntry& e = gold.entries[static_cast<size_t>(pi)];
    std::vector<int32_t> gen(static_cast<size_t>(steps), 0);
    int32_t n_gen = 0;
    const auto ts = std::chrono::steady_clock::now();
    if (fresh_prefill) {
      // One token per completion, from a prompt that grows by the token the
      // previous completion returned. `n_gen` still counts what THIS driver
      // produced, so every count assertion below is unchanged.
      std::vector<int32_t> ctx = e.prompt_token_ids;
      vllm_sampling_params one = sp;
      one.max_tokens = 1;
      for (int t = 0; t < steps; ++t) {
        int32_t got = 0, n_one = 0;
        const vllm_status s1 = vllm_complete_tokens(
            eng, ctx.data(), static_cast<int32_t>(ctx.size()), &one, &got, 1,
            &n_one, nullptr);
        if (s1 != VLLM_OK || n_one != 1) {
          std::fprintf(stderr,
                       "[nemotron-h] prompt %d fresh-prefill step %d FAILED "
                       "(status=%d n=%d): %s\n",
                       pi, t, static_cast<int>(s1), static_cast<int>(n_one),
                       vllm_last_error());
          vllm_engine_free(eng);
          return 1;
        }
        gen[static_cast<size_t>(t)] = got;
        ctx.push_back(got);
        ++n_gen;
      }
    } else {
    const vllm_status st = vllm_complete_tokens(
        eng, e.prompt_token_ids.data(),
        static_cast<int32_t>(e.prompt_token_ids.size()), &sp, gen.data(),
        static_cast<int32_t>(gen.size()), &n_gen, nullptr);
    if (st != VLLM_OK) {
      std::fprintf(stderr, "[nemotron-h] prompt %d FAILED: %s\n", pi,
                   vllm_last_error());
      vllm_engine_free(eng);
      return 1;
    }
    }
    const auto te = std::chrono::steady_clock::now();

    const int expected = static_cast<int>(e.token_ids.size());
    const int n = std::min(expected, static_cast<int>(n_gen));
    int row_match = 0;
    for (int t = 0; t < n; ++t) {
      ++total_compared;
      if (gen[static_cast<size_t>(t)] == e.token_ids[static_cast<size_t>(t)]) {
        ++total_matched;
        ++row_match;
      }
    }
    // The element count asserted against the expected geometry, per prompt.
    // `compared` short of `golden_width` is a SHORT RUN and is named as one.
    const bool full = (n == expected);
    if (full) ++rows_full; else ++rows_short;
    std::fprintf(stderr,
                 "[nemotron-h] prompt %d: prompt_tokens=%zu generated=%d "
                 "golden_width=%d compared=%d matched=%d wall=%.2fs%s\n",
                 pi, e.prompt_token_ids.size(), static_cast<int>(n_gen), expected,
                 n, row_match, std::chrono::duration<double>(te - ts).count(),
                 full ? "" : "  <-- SHORT RUN, not a full row");
    if (row_match != n || !full) {
      std::fprintf(stderr, "        got: %s\n        exp: %s\n",
                   JoinIds(gen, n).c_str(), JoinIds(e.token_ids, n).c_str());
    }
  }

  std::fprintf(stderr,
               "\n[nemotron-h] TOKEN MATCH: %d/%d over %d prompt(s) "
               "(full rows=%d, short rows=%d, mode=%s)\n",
               total_matched, total_compared, n_prompts, rows_full, rows_short,
               fresh_prefill ? "fresh-prefill" : "decode");
  }
  vllm_engine_free(eng);

  // A pass needs three things to be true at once, and each is checked here
  // rather than left to the reader of the log: something was compared, every
  // compared token matched, and every row was compared to its FULL golden
  // width. Dropping any one of them is how a mute instrument reports a pass.
  if (total_compared == 0) {
    std::fprintf(stderr,
                 "[nemotron-h] REFUSING: ZERO tokens compared — a mute "
                 "instrument, not a pass\n");
    return 3;
  }
  if (rows_short != 0) {
    std::fprintf(stderr,
                 "[nemotron-h] SHORT: %d row(s) compared fewer tokens than the "
                 "golden carries; not a full-width result\n",
                 rows_short);
    return 4;
  }
  if (total_matched != total_compared) {
    std::fprintf(stderr, "[nemotron-h] DIVERGENCE\n");
    // #926: say it HERE too, not only in the header 400 lines up. A reader who
    // sees "DIVERGENCE" and stops reading is the reader this line is for.
    if (gold.capture_recorded != 1) {
      const std::string owed =
          gold.capture_issue.empty() ? std::string() : (" (" + gold.capture_issue + ")");
      std::fprintf(stderr,
                   "[nemotron-h] ...against a golden whose engine configuration "
                   "is NOT recorded, so this is a difference from an "
                   "unattributable reference%s\n",
                   owed.c_str());
    }
    return 1;
  }
  std::fprintf(stderr, "[nemotron-h] STRICT PASS\n");
  return 0;
}
