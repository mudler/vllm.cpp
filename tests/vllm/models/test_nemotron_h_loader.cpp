// Nemotron-H (`NemotronHForCausalLM`) WEIGHT LOADER gate — issue #517, spec
// `.agents/specs/nemotron-h-model.md` (§5b, §6b "Still owed after W4").
//
// W3 made the architecture KNOWN and enumerated all 18487 released tensors; W4
// made it COMPUTE. Nothing MATERIALIZED those tensors, so every checkpoint load
// left `NemotronHHostWeights` unmaterialized and the forward refused by name.
// This gate is that missing brick, and it gates it STRUCTURALLY rather than by
// tokens:
//
//   (1) every one of the 18487 released tensors is either MATERIALIZED into a
//       named host slot or DEFERRED BY NAME with the W that owns it — "nobody
//       thought of it" is not a state;
//   (2) the per-scheme composition is asserted against the checkpoint's own
//       memory format: 5935 NVFP4 W4A16 group-16 triples, 46 FP8 W8A8 static
//       triples, and the bf16/f32 remainder. A checkpoint read as UNIFORM NVFP4
//       is still numerically plausible and still matches tokens while moving the
//       wrong bytes, which is exactly what a token gate cannot see;
//   (3) the loaded DTYPES are the shipped ones, not wider. A too-WIDE dtype is
//       numerically correct, invisible to a token comparison, and doubles the
//       bytes (AGENTS.md). The three deliberate widenings (`A_log`, `D`,
//       `dt_bias`, bf16 on disk -> f32 in host memory) are upstream's own
//       polarity and are asserted INDIVIDUALLY so they cannot spread;
//   (4) the scale tensors are bound to the right consumers — a group scale on
//       the wrong projection is a x1.10-class error a token gate absorbs;
//   (5) the forward reached through the SHARED `ModelRegistry::Forward` seam
//       produces finite, non-degenerate logits over the real vocabulary.
//
// The checkpoint is resolved through `parity::Nemotron35LightningSnapshot()`,
// which pins by CONTENT (#569): it sweeps every staged file's
// `.cache/huggingface/download/<file>.metadata` `commit_hash` against revision
// 29f2d174. The resolved directory is PRINTED, because `VT_NEMOTRON35_SNAPSHOT`
// is an ungated escape hatch and a gate that does not say which directory it
// read cannot be reproduced. Absent checkpoint => a loud SKIP, never a
// substitution.
//
// This is NOT the W6 token gate, and it makes no speed claim. It DOES consume
// the committed `nemotron_35_lightning_greedy/oracle.json`, but only for its
// smallest possible claim: ONE forward per prompt, and the argmax of the last
// position against that prompt's FIRST generated token (case (6) below). The
// full 32-token greedy decode against the pinned oracle — identical prompts,
// counts, batching and sampling, with the oracle identity asserted — stays
// W6's. The first-token arm is here because it is the only check that can fail
// for a reason the structural gate cannot see: every count can be right while a
// group scale is transposed or a nibble order is flipped.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "hf_snapshot.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, *KvCache
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"            // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"  // GDNAttentionMetadata
#include "vt/device.h"
#include "vt/dtype.h"

#ifndef NEMOTRON_H_GOLDENS_DIR
#define NEMOTRON_H_GOLDENS_DIR \
  "tests/parity/goldens/nemotron_35_lightning_greedy"
#endif

namespace {

// Peak resident set size of this process, in KiB, straight out of the kernel.
// VmHWM is the high-water mark, so it survives a buffer that has already been
// freed by the time the check runs — which is the number a load has to report.
int64_t VmHwmKiB() {
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmHWM:") {
      int64_t kib = 0;
      in >> kib;
      return kib;
    }
    std::string rest;
    std::getline(in, rest);
  }
  return -1;
}

// Every checkpoint-gated case in this TU, by name. A skip is a RESULT and has
// to look like one: as this file shipped, a `ctest` run with no
// `CHECKPOINT_ROOT` recorded `test_nemotron_h_loader ... Passed 0.00 sec` with
// ZERO assertions, which is byte-for-byte what a real pass looks like from
// outside. An instrument that cannot say how many things it examined has not
// reported ([[the-state-was-not-the-one-you-believed]]).
//
// So the skip path runs assertions of its own — the case name is in this
// registry, the refusal named a reason, and the count of skipped cases is
// stated — and the closing case below asserts that every registered case
// reached exactly one verdict. Modelled on `PendingRunnerOps()`
// (tests/parity/test_op_parity.cpp:1834): a listed thing SKIPS loudly, an
// unlisted one hard-FAILS.
const std::set<std::string>& CheckpointGatedCases() {
  static const std::set<std::string> kCases = {
      "real_checkpoint_loads_and_forwards",
  };
  return kCases;
}

// Verdicts recorded so far, by case name: "RAN" or "SKIPPED(<reason>)".
std::map<std::string, std::string>& Verdicts() {
  static std::map<std::string, std::string> v;
  return v;
}

// Record a LOUD, NAMED, COUNTED skip and return the number of cases skipped so
// far. Asserts the case is one this TU declared, so a renamed case cannot skip
// itself into invisibility.
int NoteSkip(const std::string& case_name, const std::string& why) {
  REQUIRE_MESSAGE(CheckpointGatedCases().count(case_name) == 1,
                  "'" << case_name
                      << "' is not a declared checkpoint-gated case -- add it to "
                         "CheckpointGatedCases() before skipping it");
  REQUIRE_MESSAGE(!why.empty(), "a skip must state WHY; an empty reason is not one");
  Verdicts()[case_name] = "SKIPPED(" + why + ")";
  int n = 0;
  for (const auto& [name, verdict] : Verdicts()) {
    (void)name;
    if (verdict.rfind("SKIPPED", 0) == 0) ++n;
  }
  return n;
}

void NoteRan(const std::string& case_name) {
  REQUIRE(CheckpointGatedCases().count(case_name) == 1);
  Verdicts()[case_name] = "RAN";
}

// #926: the provenance contract's notion of "prose", matching
// `_is_prose`/`len(value.strip())` in scripts/nemotron-h-oracle-capture.py
// EXACTLY.
//
// This exists because the two copies disagreed on whitespace. Python has always
// spelled these `value.strip()`; this file spelled them `.empty()` and `.size()`
// on the raw string. So a golden carrying 200 SPACES in `unrecoverable_reason`,
// `evidence.never_reproduced` and `evidence.gate_form` was refused by `--check`
// with 3 problems and accepted here at 70 assertions, 0 failed, `SUCCESS!`. A
// blank paragraph is the record going missing exactly as surely as a deleted
// one, and two copies of one contract disagreeing about what satisfies it is
// the drift this three-copy design exists to prevent.
//
// Scope is deliberate and is NOT "trim everything". Python trims in `_is_prose`
// and in the argument floor, and it does NOT trim `capture.batch.shape`, which
// it tests for truthiness. Trimming that one here would repair this divergence
// by opening the mirror image of it.
std::string TrimmedProse(const nlohmann::json& node) {
  if (!node.is_string()) return std::string();
  const std::string raw = node.get<std::string>();
  const char* kSpace = " \t\n\r\f\v";
  const size_t first = raw.find_first_not_of(kSpace);
  if (first == std::string::npos) return std::string();
  return raw.substr(first, raw.find_last_not_of(kSpace) - first + 1);
}

}  // namespace

// ── The golden's PROVENANCE, asserted before any token is compared (#926) ───
// Everything below this line that touches `oracle.json` compares tokens against
// it. This case asks the prior question: can that reference be regenerated, and
// does it say what configuration produced it?
//
// As `af8170154` committed it, the answer was no and the file was silent about
// it. `oracle.json` recorded the model, the revision, `temperature` and
// `max_tokens` and the vllm/transformers/flashinfer versions, and NOT ONE engine
// knob; the capture ran from an uncommitted driver in `$HOME` on `dgx.casa`,
// which was reimaged two days later. `enforce_eager`, `max_model_len`,
// `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization` and the
// batch shape each move a greedy argmax at a near-tie, and two later runs under
// fully recorded configurations reproduce prompts 0 and 1 exactly and neither
// reproduces prompt 2 (32/32, 32/32, 26/32 and 32/32, 32/32, 29/32).
//
// So the contract this case holds is not "the configuration is recorded" -- it
// cannot be, and inventing one would be worse than the silence. It is that the
// golden STATES which of the two it is. A golden that records its configuration
// must record ALL of it; a golden that cannot must say so in the file and name
// the issue that owes the re-derivation. The third state, silence, is the defect,
// and it is what this case removes.
//
// It needs no checkpoint and no GPU: the golden is committed, so this runs on
// every runner. That is deliberate. The provenance defect is a records defect,
// and a gate for it must not need the hardware whose absence caused it.
// #926: the whitespace half of the contract, pinned HERE rather than only in
// the Python suite.
//
// The provenance case below reads the one committed golden, so it can only ever
// exercise the shape that golden happens to have. That made this arm unable to
// hold its own half: a blank-paragraph golden was refused by `--check` and
// accepted here. `test_the_floor_is_not_met_by_whitespace` in
// tests/scripts/test_nemotron_h_oracle_capture.py pins the Python side; this
// case is its counterpart, so neither copy depends on the other to notice.
//
// The values are the reviewer's: 200 spaces, which is over the 80-character
// floor by every untrimmed measure and is not one word of a record.
TEST_CASE("NemotronH golden: a blank paragraph is not prose") {
  CHECK(TrimmedProse(nlohmann::json(std::string(200, ' '))).empty());
  CHECK(TrimmedProse(nlohmann::json(std::string("\t\n\r\f\v  "))).empty());
  CHECK(TrimmedProse(nlohmann::json(std::string())).empty());

  // 200 spaces measures 200 raw and 0 trimmed. The floor must read the second.
  const nlohmann::json blank = std::string(200, ' ');
  CHECK(blank.get<std::string>().size() == 200);
  CHECK(TrimmedProse(blank).size() == 0);

  // A non-string is not prose either, matching `_is_prose`'s isinstance check —
  // this is the `"unrecoverable_reason": 123` divergence, held on both sides.
  CHECK(TrimmedProse(nlohmann::json(123)).empty());
  CHECK(TrimmedProse(nlohmann::json()).empty());

  // And the other side of the rule, so the case cannot pass by refusing
  // everything: real prose survives, and surrounding whitespace is not counted
  // toward the floor.
  CHECK(TrimmedProse(nlohmann::json(std::string("  a real reason  "))) ==
        "a real reason");
  const std::string padded = "   " + std::string(80, 'x') + "   ";
  CHECK(TrimmedProse(nlohmann::json(padded)).size() == 80);
}

TEST_CASE("NemotronH golden: the reference says whether it can be regenerated") {
  const std::string path = std::string(NEMOTRON_H_GOLDENS_DIR) + "/oracle.json";
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot read the committed golden at " << path);
  nlohmann::json doc = nlohmann::json::parse(in);

  // Anti-vacuity first. A comparison over zero elements reports a perfect
  // score, so the width every later case will compare over is asserted here
  // rather than trusted.
  REQUIRE(doc.contains("sampling"));
  REQUIRE(doc["sampling"].contains("max_tokens"));
  const int width = doc["sampling"]["max_tokens"].get<int>();
  REQUIRE(width > 0);
  REQUIRE(doc.contains("golden"));
  REQUIRE(doc["golden"].is_array());
  REQUIRE(!doc["golden"].empty());
  for (const auto& entry : doc["golden"]) {
    REQUIRE(entry.contains("prompt_token_ids"));
    REQUIRE(!entry["prompt_token_ids"].empty());
    REQUIRE(entry.contains("token_ids"));
    CHECK(static_cast<int>(entry["token_ids"].size()) == width);
  }
  MESSAGE("golden: " << doc["golden"].size() << " prompts x " << width
                     << " tokens = " << doc["golden"].size() * width
                     << " token comparisons available");

  REQUIRE_MESSAGE(doc.contains("capture"),
                  "the golden carries no `capture` block, so nothing in this "
                  "tree can say what configuration produced it -- see #926 and "
                  "scripts/nemotron-h-oracle-capture.py");
  const nlohmann::json& capture = doc["capture"];
  REQUIRE(capture.contains("engine_config_recorded"));
  REQUIRE(capture["engine_config_recorded"].is_boolean());
  for (const std::string& key : {std::string("schema"), std::string("generator"),
                                 std::string("captured_utc"), std::string("host")}) {
    // std::string, not const char*: doctest stringifies a bare char* as a
    // BOOL, and this message read "capture is missing '1'" until it did not.
    REQUIRE_MESSAGE(capture.contains(key), "capture is missing '" << key << "'");
  }

  if (!capture["engine_config_recorded"].get<bool>()) {
    // UNATTRIBUTED, and saying so is the whole contract. The golden is kept --
    // deleting evidence to make a gate green is never the repair -- but every
    // token score taken against it is a difference from an unattributable
    // reference, and this run says that out loud.
    REQUIRE(capture.contains("unrecoverable_reason"));
    CHECK(capture["unrecoverable_reason"].is_string());
    CHECK(!TrimmedProse(capture["unrecoverable_reason"]).empty());
    REQUIRE(capture.contains("issue"));
    const std::string issue = capture["issue"].get<std::string>();
    CHECK(issue.rfind("https://github.com/mudler/vllm.cpp/issues/", 0) == 0);
    // "unrecorded" and "here is the record" cannot both be true.
    REQUIRE(capture.contains("engine"));
    CHECK(capture["engine"].is_null());

    // ── The substance, not only the shape ──────────────────────────────────
    // Everything above is satisfied by a file that says "unrecorded", names an
    // issue and argues NOTHING. The attributed arm below is gated by STRUCTURE
    // -- twenty keys are there or they are not -- but this arm's whole record
    // is prose, and a check that asks only whether the prose is non-empty gates
    // the shape and not the substance. Gut `evidence`,
    // `forced_by_checkpoint_or_device` and `captured_utc_is`, put the word
    // "dunno" in `unrecoverable_reason`, and the file still passes as a record
    // while being one. That is the state #926 filed, reached from the other
    // side.
    //
    // `forced_by_checkpoint_or_device` names the terms COMMON to every
    // unoverridden run of this checkpoint, which is what narrows
    // "unrecoverable" to the knobs a driver passes. `evidence` carries whether
    // anything ever reproduced this golden and which gate form its behaviour
    // licenses. The floor below detects REMOVAL of an argument; it does not and
    // cannot claim the prose is true, and it is set from the shortest real
    // field so that an honest rewording does not red it.
    //
    // Mirrored from REQUIRED_FORCED_TERM_KEYS, REQUIRED_EVIDENCE_KEYS and
    // MIN_ARGUMENT_CHARS in scripts/nemotron-h-oracle-capture.py;
    // tests/scripts/test_nemotron_h_oracle_capture.py parses these very lists
    // out of this file and refuses to let the copies drift.
    const size_t kMinArgumentChars = 80;
    const std::vector<std::string> kForcedTermKeys = {
        "kv_cache_dtype", "moe_backend", "dtype", "quantization"};
    const std::vector<std::string> kEvidenceKeys = {"never_reproduced",
                                                    "gate_form"};

    REQUIRE_MESSAGE(capture.contains("captured_utc_is"),
                    "capture is missing 'captured_utc_is': a commit's author "
                    "date read as a capture time is a fabricated provenance, so "
                    "the file has to say which one this is");
    CHECK(!TrimmedProse(capture["captured_utc_is"]).empty());

    for (const auto& block : {std::make_pair(std::string("forced_by_checkpoint_or_device"),
                                             kForcedTermKeys),
                              std::make_pair(std::string("evidence"), kEvidenceKeys)}) {
      REQUIRE_MESSAGE(capture.contains(block.first),
                      "capture is missing '" << block.first
                                             << "': an unrecoverable "
                                                "configuration is a claim, and "
                                                "a claim without its supporting "
                                                "record is silence");
      REQUIRE(capture[block.first].is_object());
      for (const std::string& key : block.second) {
        REQUIRE_MESSAGE(capture[block.first].contains(key),
                        "capture." << block.first << " is missing '" << key
                                   << "', so an argument this golden's "
                                      "admissibility rests on is gone");
        CHECK_MESSAGE(!TrimmedProse(capture[block.first][key]).empty(),
                      "capture." << block.first << "['" << key
                                 << "'] is empty or blank");
      }
    }

    // The four fields whose content is an ARGUMENT rather than a value.
    for (const std::vector<std::string>& path :
         std::vector<std::vector<std::string>>{
             {"unrecoverable_reason"},
             {"forced_by_checkpoint_or_device", "kv_cache_dtype"},
             {"evidence", "never_reproduced"},
             {"evidence", "gate_form"}}) {
      const nlohmann::json* node = &capture;
      bool reachable = true;
      std::string dotted;
      for (const std::string& step : path) {
        dotted += (dotted.empty() ? "" : ".") + step;
        if (!node->is_object() || !node->contains(step)) {
          reachable = false;
          break;
        }
        node = &(*node)[step];
      }
      REQUIRE_MESSAGE(reachable, "capture." << dotted << " is absent");
      REQUIRE_MESSAGE(node->is_string(), "capture." << dotted << " is not prose");
      CHECK_MESSAGE(TrimmedProse(*node).size() >= kMinArgumentChars,
                    "capture." << dotted << " is "
                               << TrimmedProse(*node).size()
                               << " characters, under the " << kMinArgumentChars
                               << " an ARGUMENT needs: a one-word answer here is "
                                  "the record going missing while the file keeps "
                                  "its shape");
    }

    MESSAGE(
        "UNATTRIBUTED GOLDEN: this reference records no engine configuration, "
        "so a token difference against it is not yet a defect. Owed by "
        << issue
        << ". Re-derive with scripts/nemotron-h-oracle-capture.py --capture "
           "--profile nhspeed-a.");
    return;
  }

  // ATTRIBUTED: then it is attributed COMPLETELY. Each key below can move a
  // greedy argmax at a near-tie, and a key that is absent is not "the default",
  // it is unrecorded -- which is the state this whole case exists to refuse.
  // The list is duplicated in scripts/nemotron-h-oracle-capture.py
  // (REQUIRED_ENGINE_KEYS) and tests/scripts/test_nemotron_h_oracle_capture.py
  // asserts the three agree, so neither copy can drift alone.
  REQUIRE(capture.contains("engine"));
  REQUIRE(capture["engine"].is_object());
  REQUIRE(capture["engine"].contains("resolved"));
  const nlohmann::json& resolved = capture["engine"]["resolved"];
  const std::vector<std::string> required = {
      "attention_backend", "block_size",       "compilation_mode",
      "cudagraph_capture_sizes", "cudagraph_mode", "dtype",
      "enable_chunked_prefill", "enable_prefix_caching", "enforce_eager",
      "gpu_memory_utilization", "kv_cache_dtype", "max_model_len",
      "max_num_batched_tokens", "max_num_seqs", "moe_backend",
      "num_gpu_blocks", "num_gpu_blocks_override", "quantization",
      "seed", "tensor_parallel_size"};
  for (const std::string& key : required) {
    REQUIRE_MESSAGE(resolved.contains(key),
                    "capture.engine.resolved is missing '"
                        << key << "', so this golden is only partly attributed");
    if (key != "num_gpu_blocks_override") {
      CHECK_MESSAGE(!resolved[key].is_null(),
                    "capture.engine.resolved['"
                        << key
                        << "'] is null: a value that could not be read is not a "
                           "value that was default");
    }
  }
  REQUIRE(capture.contains("batch"));
  REQUIRE(capture["batch"].contains("shape"));
  CHECK(!capture["batch"]["shape"].get<std::string>().empty());
  REQUIRE(capture.contains("legs"));
  CHECK_MESSAGE(capture["legs"].get<int>() >= 2,
                "one leg cannot show the configuration is deterministic");
  REQUIRE(capture.contains("legs_agree"));
  CHECK(capture["legs_agree"].get<bool>());
  MESSAGE("ATTRIBUTED GOLDEN: profile "
          << capture["engine"].value("profile", "<unnamed>") << ", "
          << required.size() << " engine keys recorded, "
          << capture["legs"].get<int>() << " agreeing legs");
}

TEST_CASE("NemotronH: the REAL checkpoint loads and the forward produces logits") {
  const std::string kCase = "real_checkpoint_loads_and_forwards";
  std::string why;
  const std::string dir = parity::Nemotron35LightningSnapshot(&why);
  if (dir.empty()) {
    const int skipped = NoteSkip(kCase, why);
    MESSAGE("SKIPPED " << skipped << " of " << CheckpointGatedCases().size()
                       << " checkpoint-gated case(s). '" << kCase
                       << "': no Nemotron-3.5-Lightning checkpoint at the pinned "
                          "revision 29f2d1746d8f41e316523194b19018707749b1b1 -- "
                       << why
                       << ". Export CHECKPOINT_ROOT (set -a; . ./.env; set +a) or "
                          "point VT_NEMOTRON35_SNAPSHOT at the staged directory. "
                          "Both gate hosts mount it at "
                          "/usr/local/nas_share/checkpoints/"
                          "nemotron-3.5-lightning-30b-nvfp4.");
    // The load-bearing assertion of the skip path: this run examined ZERO
    // checkpoint tensors, and says so with a check rather than with silence.
    CHECK(skipped == 1);
    CHECK(Verdicts().at(kCase).rfind("SKIPPED", 0) == 0);
    return;
  }
  NoteRan(kCase);
  // The resolved directory is EVIDENCE, not a debug aid: VT_NEMOTRON35_SNAPSHOT
  // is deliberately never revision-checked, so a run that does not name the
  // directory it read cannot be reproduced or falsified.
  MESSAGE("resolved checkpoint directory: " << dir);

  const vllm::HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const vllm::NemotronHParams params = vllm::ParseNemotronHParams(config);
  const std::vector<vllm::NemotronHTensor> enumerated =
      vllm::EnumerateNemotronHTensors(params);
  CHECK(enumerated.size() == 18487);

  // Open every shard. The index is the authority on which files exist; opening
  // them here (rather than inside the loader) is the SHARED ModelSource seam
  // every other architecture's loader consumes.
  const std::map<std::string, std::string> weight_map =
      vllm::LoadSafetensorsIndex(dir + "/model.safetensors.index.json");
  CHECK(weight_map.size() == 18487);
  std::set<std::string> shard_names;
  for (const auto& [tensor, shard] : weight_map) {
    (void)tensor;
    shard_names.insert(shard);
  }
  MESSAGE("shards: " << shard_names.size());

  const int64_t rss_before_kib = VmHwmKiB();
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(shard_names.size());
  for (const std::string& name : shard_names) {
    shards.push_back(vllm::SafetensorsFile::Open(dir + "/" + name));
  }

  const vllm::ModelSource source = vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  const int64_t rss_after_kib = VmHwmKiB();
  MESSAGE("peak RSS before load: " << rss_before_kib / 1024 << " MiB");
  MESSAGE("peak RSS after load:  " << rss_after_kib / 1024 << " MiB");

  // ─── (1) EVERY tensor is materialized or deferred BY NAME ──────────────────
  const vllm::NemotronHLoadReport& rep = vllm::NemotronHLoadReportOf(*model);
  MESSAGE("host bytes: " << rep.host_bytes / (1024 * 1024) << " MiB, source "
                         << rep.source_bytes / (1024 * 1024) << " MiB");
  CHECK(rep.enumerated == 18487);
  CHECK(rep.in_index == 18487);
  CHECK(rep.materialized + rep.deferred == 18487);
  // The MTP tower (W5): 270 unquantized bf16 tensors the `ignore` list's `mtp*`
  // wildcard leaves unquantized. Deferred, counted, and NAMED — never silently
  // dropped.
  CHECK(rep.deferred == 270);
  CHECK(rep.materialized == 18217);
  REQUIRE(!rep.deferred_by_name.empty());
  for (const std::string& tag : rep.deferred_by_name) {
    CHECK(tag.find("W5") != std::string::npos);
  }

  // ─── (2) the per-scheme composition, against the CHECKPOINT's own format ───
  //
  // Reading MIXED_PRECISION as uniform NVFP4 is numerically plausible and
  // token-invisible. These five rows are the scheme table of spec §1 turned into
  // numbers, and they must sum to every materialized tensor with nothing left
  // over.
  //
  // NVFP4 W4A16 g16: 23*128*2 routed + 23*2 shared + lm_head = 5935 projections,
  // three tensors each (weight / weight_scale / weight_scale_2). Exactly the
  // `{W4A16_NVFP4: 5935}` half of the histogram W1 measured over all 5981
  // `quantized_layers` entries.
  CHECK(rep.nvfp4_weights == 5935);
  CHECK(rep.nvfp4_tensors == 5935 * 3);
  // FP8 W8A8 static: the 23 mamba `in_proj` + 23 `out_proj` = 46 targets, the
  // other half of that histogram. If this reads 0 the loader took the whole
  // checkpoint as NVFP4; if it reads 5981 it took the whole thing as FP8.
  CHECK(rep.fp8_weights == 46);
  CHECK(rep.fp8_tensors == 46 * 3);
  // The fp8 KV scheme: one k_scale + one v_scale on each of the 6 attention
  // layers.
  CHECK(rep.fp8_kv_scale_tensors == 12);
  // The unquantized remainder, by the dtype it SHIPS in: 216 bf16 (embeddings,
  // norm_f, 52 layer norms, and the 6 mamba/attention tensor families) and 46
  // f32 (the 23 routers and their 23 score-correction biases).
  CHECK(rep.bf16_tensors == 216);
  CHECK(rep.f32_tensors == 46);
  CHECK(rep.nvfp4_tensors + rep.fp8_tensors + rep.fp8_kv_scale_tensors +
            rep.bf16_tensors + rep.f32_tensors ==
        rep.materialized);

  // ─── (3) the loaded dtypes are the SHIPPED ones, not wider ────────────────
  //
  // A too-WIDE dtype is numerically correct, invisible to a token comparison,
  // and moves twice the bytes. The only widening the released checkpoint asks
  // for is the three f32-by-contract SSM scalars on each of the 23 mamba
  // layers — upstream's own polarity (`-torch.exp(self.A_log.float())`) and
  // what `vt::Mamba2ChunkScan` validates. 23 * 3 = 69, and no more.
  CHECK(rep.widened_tensors == 69);

  // The two arithmetic facts that make keeping the quantized forms mandatory
  // rather than tidy: the host mirror is within a few percent of the on-disk
  // bytes, and the routed experts alone would be 58.7 GB at bf16.
  CHECK(rep.host_bytes < 24LL * 1024 * 1024 * 1024);
  CHECK(rep.host_bytes > 16LL * 1024 * 1024 * 1024);

  // ─── (3b) THE PAYLOAD IS WHOLE, TO THE BYTE ───────────────────────────────
  //
  // PRINTING A NUMBER IS NOT GATING IT. The MiB line above was printed and read
  // by nothing: truncating every payload produced by the production
  // `CopyDenseOwned` (`nemotron_h_weights.cpp:532`) to half moved `host_bytes`
  // by 470 MiB and left this suite 2/2 green WITH 3/3 oracle goldens matching —
  // a shrunk `std::vector` keeps its buffer, so `View()` reads the same values
  // back and every token stays identical. Only the accounting moved, and
  // nothing asserted on the accounting.
  //
  // So both totals are asserted EXACTLY. They can be exact because the
  // checkpoint is pinned BY CONTENT (revision 29f2d174, asserted below), so
  // `source_bytes` is a property of that revision and `host_bytes` is a
  // property of that revision plus this loader. A range would not catch a
  // truncation of one weight family; these do.
  //
  // THEY ARE NOT EQUAL, and the 15324-byte gap is fully accounted for rather
  // than tolerated — asserting `host_bytes == source_bytes` would have been
  // wrong. Two effects, in opposite directions:
  //
  //   -24156  scalars READ but not STORED as bytes: 5935 NVFP4 `weight_scale_2`
  //           + 46*2 FP8 `input_scale`/`weight_scale` + 12 fp8-KV k/v scales
  //           = 6039 f32 scalars * 4 B. `Loader::Need` counts them into
  //           `source_bytes` (`nemotron_h_weights.cpp:410`); they land in float
  //           members, not in a payload `HostBytesOf` walks.
  //   + 8832  the 69 f32-by-contract SSM widenings asserted just above: 23
  //           layers * 3 tensors * 64 heads * (4 - 2) B.
  //   ───────
  //    -15324 = 18888922112 - 18888937436.
  //
  // If either literal moves, the loader's memory format changed — say so in the
  // commit and re-derive it, never widen it into a range.
  MESSAGE("host bytes exact: " << rep.host_bytes
                               << ", source bytes exact: " << rep.source_bytes);
  CHECK(rep.source_bytes == 18888937436LL);
  CHECK(rep.host_bytes == 18888922112LL);
  CHECK(rep.source_bytes - rep.host_bytes == 6039LL * 4 - 69LL * 64 * 2);
  // Peak RSS is the number a unified-memory box lives or dies by.
  MESSAGE("peak RSS after load (GiB): " << static_cast<double>(rss_after_kib) /
                                               (1024.0 * 1024.0));

  // The forward, through the SHARED registry seam — never a private entry point.
  //
  // A2-R: this is also where the loader's recorded MATMUL ORIENTATION is gated.
  // `nemotron_h.cpp:306` refuses an `OwnedTensor` whose `nk` is not the
  // [out, in] torch-Linear orientation `vt::MatmulBT` consumes, and the four
  // attention projections reach that call from HERE. The synthetic fixture in
  // `test_nemotron_h_forward.cpp` sets its own `nk`, so it cannot see the
  // loader get it wrong; this can, and does — flipping
  // `nemotron_h_weights.cpp:686-689` throws out of this call.
  const std::vector<int32_t> token_ids{1, 2, 3, 4};
  const std::vector<int32_t> positions{0, 1, 2, 3};
  const std::vector<int32_t> logits_indices{3};
  const vllm::v1::CommonAttentionMetadata attn_meta{};
  const vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const vllm::ModelForwardInput input{.token_ids = token_ids,
                                      .positions = positions,
                                      .attn_meta = attn_meta,
                                      .gdn_meta = gdn_meta,
                                      .attn_kv = attn_kv,
                                      .gdn_state = gdn_state,
                                      .config = config,
                                      .queue = queue,
                                      .logits_indices = logits_indices,
                                      .num_reqs = 1};
  const vllm::ForwardLogits logits =
      vllm::ModelRegistry::Forward(*model, input);
  CHECK(logits.rows == 1);
  CHECK(logits.vocab == params.vocab_size);
  REQUIRE(logits.host.size() ==
          static_cast<size_t>(logits.rows * logits.vocab));

  // Finite AND non-degenerate: a loader that materialized zeros would return a
  // perfectly finite constant row, and an argmax over it is still a token.
  // Aggregated rather than one assertion per vocabulary entry: 131072 REQUIREs
  // swamp the assertion count, and a changed assertion COUNT is itself the
  // signal a mutation is read by ([[doctest-assertions-line-hides-thrown-cases]]).
  double lo = logits.host[0];
  double hi = logits.host[0];
  int64_t nonfinite = 0;
  for (float v : logits.host) {
    if (!std::isfinite(v)) ++nonfinite;
    lo = std::min<double>(lo, v);
    hi = std::max<double>(hi, v);
  }
  MESSAGE("logits range: [" << lo << ", " << hi << "]");
  CHECK(nonfinite == 0);
  CHECK(hi - lo > 1.0);

  // ─── (6) EVIDENCE, not the W6 token gate: the first greedy token of each of
  //         the three committed oracle prompts ────────────────────────────────
  //
  // W6 formally owns the token gate — identical prompts, counts, batching and
  // sampling against the pinned oracle, with the oracle identity asserted. This
  // is a much smaller claim on the same artifact: for each committed prompt,
  // ONE forward over its `prompt_token_ids` and the argmax of the last position
  // against the golden's FIRST generated token.
  //
  // It is here because it is the one check that can fail for a reason the
  // structural gate above cannot see. Every count can be right — 5935 NVFP4
  // triples bound to the right projections, 46 FP8 triples, 69 widenings and no
  // more — while a group scale is transposed or a nibble order is flipped, and
  // the answer stays finite and correctly shaped. A wrong argmax on all three
  // prompts is what that looks like from outside.
  const std::filesystem::path goldens =
      std::filesystem::path(NEMOTRON_H_GOLDENS_DIR) / "oracle.json";
  std::ifstream gin(goldens.string());
  REQUIRE_MESSAGE(gin.good(), "cannot open " << goldens.string());
  nlohmann::json oracle;
  gin >> oracle;
  // The goldens name the revision they belong to; the checkpoint resolver above
  // pinned the same one by CONTENT. Assert they agree rather than assuming it.
  CHECK(oracle.at("revision").get<std::string>() ==
        std::string(parity::kNemotron35LightningNvfP4Revision));

  int matched = 0;
  int total = 0;
  for (const nlohmann::json& g : oracle.at("golden")) {
    const std::vector<int32_t> prompt =
        g.at("prompt_token_ids").get<std::vector<int32_t>>();
    const int32_t want = g.at("token_ids").at(0).get<int32_t>();
    std::vector<int32_t> pos(prompt.size());
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int32_t>(i);
    const std::vector<int32_t> last{static_cast<int32_t>(prompt.size()) - 1};
    const vllm::ModelForwardInput in{.token_ids = prompt,
                                     .positions = pos,
                                     .attn_meta = attn_meta,
                                     .gdn_meta = gdn_meta,
                                     .attn_kv = attn_kv,
                                     .gdn_state = gdn_state,
                                     .config = config,
                                     .queue = queue,
                                     .logits_indices = last,
                                     .num_reqs = 1};
    const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, in);
    REQUIRE(out.host.size() == static_cast<size_t>(params.vocab_size));
    const int32_t got = static_cast<int32_t>(
        std::max_element(out.host.begin(), out.host.end()) - out.host.begin());
    ++total;
    if (got == want) ++matched;
    // `std::string`, NOT a `const char*` ternary: doctest 2.5.2 has no
    // stringifier for a `const char*` lvalue and prints it as `1`, which is
    // exactly what this line did on its first run — "oracle 69931" instead of
    // "oracle 6993  MATCH". The same trap this row already repaired once (spec
    // §"W1 land-prep", LOW-2).
    const std::string verdict = got == want ? "  MATCH" : "  DIFFERS";
    MESSAGE("prompt \"" << g.at("prompt").get<std::string>()
                        << "\": argmax " << got << ", oracle " << want
                        << verdict);
  }
  MESSAGE("first-token agreement with the committed oracle goldens: "
          << matched << "/" << total);
  CHECK(matched == total);

  MESSAGE("peak RSS at end: " << VmHwmKiB() / 1024 << " MiB");
}

// The accounting. Declared LAST so it runs after every checkpoint-gated case
// above (doctest registers in declaration order within a TU). It exists so this
// suite always states what it did: with the checkpoint present it reports every
// case RAN, and without it every case SKIPPED, with the reason. Neither is
// `Passed 0.00 sec` with nothing on the record.
//
// It presumes the WHOLE TU ran, which is how ctest invokes this binary. Running
// it alone under `--test-case=` deliberately FAILS: "no verdict recorded" is
// exactly the state it exists to refuse, and making it pass in that case would
// hand back the vacuous green it was written to remove.
TEST_CASE("NemotronH loader: every checkpoint-gated case reached exactly one verdict") {
  const std::set<std::string>& declared = CheckpointGatedCases();
  CHECK(Verdicts().size() == declared.size());
  int ran = 0;
  int skipped = 0;
  for (const std::string& name : declared) {
    const auto it = Verdicts().find(name);
    REQUIRE_MESSAGE(it != Verdicts().end(),
                    "checkpoint-gated case '"
                        << name
                        << "' recorded NO verdict -- it neither ran nor skipped, "
                           "which is the state this registry exists to make "
                           "impossible");
    MESSAGE(name << ": " << it->second);
    if (it->second == "RAN") {
      ++ran;
    } else {
      ++skipped;
    }
  }
  MESSAGE("checkpoint-gated cases: " << ran << " ran, " << skipped << " skipped, of "
                                     << declared.size());
  CHECK(ran + skipped == static_cast<int>(declared.size()));
}
