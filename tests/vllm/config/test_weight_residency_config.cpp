// `ENG-RESIDENCY-CONFIG` (issue #1110) — the `vllm_cpp` extension of
// `--offload-config`: the host-RAM -> DISK weight-residency tier as a config
// surface instead of environment variables only.
//
// THERE IS NOTHING UPSTREAM TO PORT. vLLM offloads weights device -> host RAM and
// stops: `OffloadBackend` is `Literal["auto","uva","prefetch"]`
// (vllm/config/offload.py:12 @ 555967922), `offloader/uva.py:21` is a CPU-blanket
// UVA offloader, `offloader/prefetch.py:557-560` is cpu-only, and nothing reads a
// weight off a file at inference time. Upstream's own offload test
// (tests/basic_correctness/test_cpu_offload.py:19-21) covers the mirrored tier and
// is already carried by tests/vllm/config/test_offload_config.cpp. So this file
// gates a vllm.cpp-original surface, and the one upstream obligation it DOES carry
// is negative: the mirrored structs must come out of a document carrying a
// `vllm_cpp` key byte-identical to what they were before, which the
// "mirror is untouched" cases below assert directly.
//
// THE THREE GUARANTEES, each of which has its own mutation:
//   1. PARSE + REFUSE. Every field round-trips, and an unknown key is an ERROR at
//      EVERY level of the document: the top level, inside `vllm_cpp` and its two
//      sub-objects, and inside the MIRRORED `uva` and `prefetch` objects. The refusal
//      is the load-bearing half: parse_offload_config_json ignores a key it does
//      not know (which is what lets this extension share the flag), so
//      `{"vllm-cpp":...}` or `{"vllm_cpp":{"mmapp":...}}` would otherwise SILENTLY
//      disable the tier that keeps a 370 GiB model inside 119 GB, and
//      `{"uva":{"cpu_offload_GB":10}}` would silently offload nothing. The hyphen is
//      the likeliest of those typos, because every flag around it is hyphenated.
//   2. PRECEDENCE: env > config > built-in default, in both directions. `VT_X=0`
//      must beat a config `true`, because an override that cannot turn a thing
//      OFF is not one, and that is the direction a benchmark arm needs.
//   3. THE LATCH, and only where there IS one, and only for what a document SAYS.
//      `expert_stream` is read through a function-local static and the slot store is
//      built once per process, so a document that would CHANGE either after the fact
//      must THROW rather than be ignored. Three things must NOT throw, and each was a
//      real failure on a legal two-model load: `mmap` and `prefault` latch nothing
//      (`GgufLoadPolicy::FromEnv()` runs per load and the prefault site no longer
//      caches); a document that OMITS a decided field is not a change to it; and a
//      document asking for exactly what the process decided is not one either. The
//      install also MERGES rather than replacing, so a partial second document does
//      not drop the first engine's fields. Every one of those is asserted below, and
//      the two-partial-documents case is where the last two were caught (#1133).
#include <doctest/doctest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vllm/config/offload.h"
#include "vllm/config/weight_residency.h"

namespace {

// Every case owns the process-global, so clear it AND the latch on entry.
struct ResidencyFixture {
  ResidencyFixture() { Clear(); }
  ~ResidencyFixture() { Clear(); }
  static void Clear() {
    vllm::ResetWeightResidencyConfigForTesting();
    ::unsetenv("VT_RESIDENCY_TEST_BOOL");
    ::unsetenv("VT_RESIDENCY_TEST_COUNT");
  }
};

// The refusal MESSAGE, not merely the fact of a throw. A parser can refuse the
// right document for the wrong reason, and one of these did: `{"vllm_cpp": 5}`
// reported `"vllm_cpp.vllm_cpp" must be a JSON object` from a hardcoded prefix,
// which CHECK_THROWS_AS cannot see. The operator reads the message, so the
// message is the guarantee.
std::string RefusalMessage(const char* doc) {
  try {
    vllm::parse_weight_residency_extension_json(doc);
  } catch (const std::invalid_argument& e) {
    return e.what();
  } catch (const std::exception& e) {
    return std::string("WRONG EXCEPTION TYPE: ") + e.what();
  }
  return "ACCEPTED (no throw)";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("residency config: every field parses out of the vllm_cpp key") {
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(R"({
        "uva": {"cpu_offload_gb": 4},
        "vllm_cpp": {
          "mmap": {"enabled": true, "prefault": false},
          "expert_stream": {"enabled": true, "slots": 8000,
                            "slot_bytes": 12582912}
        }
      })");
  REQUIRE(c.mmap.has_value());
  CHECK(*c.mmap == true);
  REQUIRE(c.prefault.has_value());
  CHECK(*c.prefault == false);
  REQUIRE(c.expert_stream.has_value());
  CHECK(*c.expert_stream == true);
  REQUIRE(c.expert_stream_slots.has_value());
  CHECK(*c.expert_stream_slots == 8000);
  REQUIRE(c.expert_stream_slot_bytes.has_value());
  CHECK(*c.expert_stream_slot_bytes == 12582912);
  CHECK_FALSE(c.empty());

  // The install line names WHAT THE OPERATOR SET, field by field. It is not the
  // resolved value — a variable can still override any of these, which is what the
  // second line of the install reports — and this string is the half that says which
  // fields of a two-tier document reached this tier.
  const std::string described = c.Describe();
  CHECK(described.find("mmap=on") != std::string::npos);
  CHECK(described.find("prefault=off") != std::string::npos);
  CHECK(described.find("expert_stream=on") != std::string::npos);
  CHECK(described.find("expert_stream_slots=8000") != std::string::npos);
}

TEST_CASE("residency config: an absent extension is the inert default") {
  // Each of these is a document that reaches --offload-config today.
  for (const char* doc : {"", "   ", "{}", R"({"uva":{"cpu_offload_gb":4}})",
                          R"({"offload_backend":"uva"})",
                          R"({"vllm_cpp":{}})"}) {
    const vllm::WeightResidencyConfig c =
        vllm::parse_weight_residency_extension_json(doc);
    CHECK(c.empty());
    CHECK(c.Describe().empty());
    CHECK_FALSE(c.mmap.has_value());
    CHECK_FALSE(c.expert_stream_slots.has_value());
  }
}

TEST_CASE("residency config: a partial extension leaves the rest unchanged") {
  // The reproduction case from the issue is exactly this shape: mmap on, prefault
  // off (a model larger than memory cannot prefault its own tower), streaming on
  // with a real slot count. `slot_bytes` is left to the computed default.
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(R"({
        "vllm_cpp": {"mmap": {"prefault": false},
                     "expert_stream": {"enabled": true, "slots": 8000}}
      })");
  CHECK_FALSE(c.mmap.has_value());
  REQUIRE(c.prefault.has_value());
  CHECK(*c.prefault == false);
  REQUIRE(c.expert_stream.has_value());
  CHECK(*c.expert_stream == true);
  REQUIRE(c.expert_stream_slots.has_value());
  CHECK(*c.expert_stream_slots == 8000);
  CHECK_FALSE(c.expert_stream_slot_bytes.has_value());
}

TEST_CASE("residency config: an unknown or mistyped key is REFUSED, never ignored") {
  // Each of these would be silently accepted by a parser that only looks its own
  // keys up, and each one leaves the field the operator meant to set at its DEFAULT
  // while the operator believes the document set it.
  const char* refused[] = {
      R"({"vllm_cpp":{"mmapp":{"enabled":true}}})",
      R"({"vllm_cpp":{"mmap":{"enable":true}}})",
      R"({"vllm_cpp":{"mmap":{"enabled":true,"prefaultt":false}}})",
      R"({"vllm_cpp":{"expert_stream":{"slot":8000}}})",
      R"({"vllm_cpp":{"expert_stream":{"enabled":true,"stats_every":16}}})",
      R"({"vllm_cpp":{"expert_streaming":{"enabled":true}}})",
      R"({"vllm_cpp":{"disk":{"enabled":true}}})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::parse_weight_residency_extension_json(doc),
                    std::invalid_argument);
  }

  // `stats_every` above is not an oversight: it is environment-only BY DECISION
  // (it changes only how often a diagnostic line prints, moves no byte and
  // reserves nothing), so the config surface must refuse it rather than accept
  // and drop it.

  // Refused BY NAME, and the name is the one the operator typed. A message that
  // invented a prefix would send them hunting through a document that does not
  // contain the key it names.
  const std::string mmapp =
      RefusalMessage(R"({"vllm_cpp":{"mmapp":{"enabled":true}}})");
  CHECK(Mentions(mmapp, "unknown key \"vllm_cpp.mmapp\""));
  CHECK(Mentions(mmapp, "expected one of: mmap expert_stream"));
}

TEST_CASE("residency config: a misspelled TOP-LEVEL key is REFUSED, never ignored") {
  // THE ONE THE FIRST PASS MISSED, and the worst of the set. The extension
  // enumerated its own keys but nothing enumerated the DOCUMENT, so a typo in
  // `vllm_cpp` itself parsed to an empty config and started a server that does not
  // borrow its weights — met by the operator as an out-of-memory kill rather than
  // as an error. The hyphen is the likeliest spelling of all: every flag around it
  // (`--offload-config`, `--kv-transfer-config`) is hyphenated.
  //
  // Refusing is also the MIRROR-FAITHFUL polarity, not a local invention. Upstream
  // has no `--offload-config` flag at all (there is no such string anywhere in the
  // vLLM tree at the pin, so no upstream-legal document exists to break), and vLLM
  // builds its config dataclasses with the `@config` decorator, whose body sets
  // `ConfigDict(extra="forbid")` under the comment "Extra fields are forbidden by
  // default" (vllm/config/utils.py:68-69 @ 555967922). `OffloadConfig`
  // (offload.py:80) and `KVTransferConfig` (kv_transfer.py:22-23) both carry it, so
  // `--kv-transfer-config` refuses an unknown key. It is this parser's silence that
  // was the deviation.
  const char* refused[] = {
      // The hyphen, the case, two transpositions, and the run-together spelling.
      R"({"vllm-cpp":{"mmap":{"enabled":true}}})",
      R"({"VLLM_CPP":{"mmap":{"enabled":true}}})",
      R"({"vllm_ccp":{"mmap":{"enabled":true}}})",
      R"({"vllmcpp":{"mmap":{"enabled":true}}})",
      R"({"vllm.cpp":{"mmap":{"enabled":true}}})",
      // And a typo in the MIRRORED half, which the same document carries and which
      // has exactly the same consequence: a budget the operator believes is set.
      R"({"uvaa":{"cpu_offload_gb":10}})",
      R"({"prefetchh":{"offload_group_size":8}})",
      R"({"offload_backends":"uva"})",
      // ...and a typo INSIDE the mirrored half, which the enumeration used to stop
      // one level short of. `parse_offload_config_json` reads `uva.cpu_offload_gb`
      // and the four `prefetch.*` fields BY NAME with a fallback
      // (src/vllm/config/offload.cpp:272-281), so each of these left the field at
      // its default while the operator believed the document set it — a budget of 0
      // GiB, or a group size of 0, i.e. no offloading at all. Upstream refuses them:
      // `UVAOffloadConfig` (offload.py:15-16) and `PrefetchOffloadConfig` (:47-48)
      // each carry `@config`, whose body sets `ConfigDict(extra="forbid")`
      // (utils.py:68-69).
      R"({"uva":{"cpu_offload_gbb":1}})",
      R"({"uva":{"cpu_offload_GB":10}})",
      R"({"uva":{"cpu_offload_gb":10,"cpu_offload_param":["experts"]}})",
      R"({"prefetch":{"offload_groupsize":8}})",
      R"({"prefetch":{"offload_group_size":8,"offload_num_in_groups":2}})",
      R"({"prefetch":{"offload_group_size":8,"offload_prefetch_steps":1}})",
      R"({"prefetch":{"offload_group_size":8,"offload_param":["experts"]}})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::parse_weight_residency_extension_json(doc),
                    std::invalid_argument);
  }

  // The message names the offender WITHOUT a phantom prefix, and lists the four
  // keys the document may carry — the three mirrored ones and the extension.
  const std::string hyphen =
      RefusalMessage(R"({"vllm-cpp":{"mmap":{"enabled":true}}})");
  CHECK(Mentions(hyphen, "unknown key \"vllm-cpp\""));
  CHECK_FALSE(Mentions(hyphen, "vllm-cpp.vllm-cpp"));
  CHECK(Mentions(hyphen, "vllm_cpp"));  // the spelling that was meant

  // The nested ones name the DOTTED path, so the operator is told which key of
  // which sub-object is wrong, and are listed against the mirrored parser's own
  // spelling rather than this extension's.
  const std::string nested_uva =
      RefusalMessage(R"({"uva":{"cpu_offload_gbb":1}})");
  CHECK(Mentions(nested_uva, "unknown key \"uva.cpu_offload_gbb\""));
  CHECK(Mentions(nested_uva, "expected one of: cpu_offload_gb cpu_offload_params"));
  CHECK(Mentions(RefusalMessage(R"({"prefetch":{"offload_groupsize":8}})"),
                 "unknown key \"prefetch.offload_groupsize\""));

  // ...and the four legal top-level keys still parse, in every combination, with
  // every field of the two mirrored sub-objects spelled correctly, so the
  // enumeration refuses a typo rather than the document.
  CHECK_NOTHROW(vllm::parse_weight_residency_extension_json(
      R"({"offload_backend":"uva",)"
      R"("uva":{"cpu_offload_gb":10,"cpu_offload_params":["experts"]},)"
      R"("prefetch":{"offload_group_size":8,"offload_num_in_group":2,)"
      R"("offload_prefetch_step":1,"offload_params":["experts"]},)"
      R"("vllm_cpp":{"mmap":{"enabled":true}}})"));
}

TEST_CASE("residency config: a wrong-typed or non-positive field is REFUSED") {
  const char* refused[] = {
      "{not json",
      "[]",
      R"({"vllm_cpp": 5})",
      R"({"vllm_cpp":{"mmap": true}})",
      R"({"vllm_cpp":{"expert_stream": 8000}})",
      R"({"vllm_cpp":{"mmap":{"enabled":"yes"}}})",
      R"({"vllm_cpp":{"mmap":{"prefault":1}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":"8000"}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":8000.5}}})",
      // A zero or negative count is TOLERATED by the environment readers, which
      // parse with atol and cannot report. A config is parsed where a message
      // still reaches the operator, so a slot count that would silently have
      // become 64 is refused instead.
      R"({"vllm_cpp":{"expert_stream":{"slots":0}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":-1}}})",
      R"({"vllm_cpp":{"expert_stream":{"slot_bytes":0}}})",
      // The two MIRRORED sub-objects, which this parser now has to OPEN in order to
      // enumerate their keys. Opening them means it meets a non-object `uva` where it
      // previously ignored the key entirely, so the message it produces for one is
      // asserted here: it must name the key with no phantom prefix, and it must be
      // the same sentence the mirrored parser has always produced, because both
      // parsers read the same string and either may reach it first.
      R"({"uva": 5})",
      R"({"prefetch": []})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    CHECK_THROWS_AS(vllm::parse_weight_residency_extension_json(doc),
                    std::invalid_argument);
  }

  // The PATH in each message is the path in the document. The first pass built the
  // top-level one from a hardcoded `vllm_cpp.` prefix and reported
  // `"vllm_cpp.vllm_cpp" must be a JSON object`, which every CHECK_THROWS_AS above
  // passed over: a refusal for the right document with the wrong reason.
  const std::string scalar_ext = RefusalMessage(R"({"vllm_cpp": 5})");
  CHECK(Mentions(scalar_ext, "\"vllm_cpp\" must be a JSON object"));
  CHECK_FALSE(Mentions(scalar_ext, "vllm_cpp.vllm_cpp"));

  const std::string scalar_sub = RefusalMessage(R"({"vllm_cpp":{"mmap": true}})");
  CHECK(Mentions(scalar_sub, "\"vllm_cpp.mmap\" must be a JSON object"));

  CHECK(Mentions(RefusalMessage(R"({"vllm_cpp":{"mmap":{"enabled":"yes"}}})"),
                 "\"vllm_cpp.mmap.enabled\" must be a boolean"));
  CHECK(Mentions(RefusalMessage(R"({"vllm_cpp":{"expert_stream":{"slots":0}}})"),
                 "\"vllm_cpp.expert_stream.slots\" must be positive (got 0)"));

  // The mirrored sub-object, named without a prefix — and, since both parsers read
  // the same string, WORD FOR WORD what `parse_offload_config_json` says about it, so
  // whichever one reaches the document first the operator reads one sentence.
  const std::string scalar_uva = RefusalMessage(R"({"uva": 5})");
  CHECK(Mentions(scalar_uva, "\"uva\" must be a JSON object"));
  CHECK_FALSE(Mentions(scalar_uva, "vllm_cpp.uva"));
  std::string mirrored_uva = "ACCEPTED (no throw)";
  try {
    vllm::parse_offload_config_json(R"({"uva": 5})");
  } catch (const std::invalid_argument& e) {
    mirrored_uva = e.what();
  }
  CHECK(scalar_uva == mirrored_uva);
}

TEST_CASE("residency config: the MIRRORED offload config is untouched by the extension") {
  // The whole reason the extension is a namespaced key rather than a field on
  // OffloadConfig: include/vllm/config/offload.h is a transcription of
  // vllm/config/offload.py @ 555967922 and must stay one. So the same document
  // has to parse through BOTH parsers, each seeing only its own half.
  const char* both = R"({
    "offload_backend": "uva",
    "uva": {"cpu_offload_gb": 10, "cpu_offload_params": ["experts"]},
    "vllm_cpp": {"mmap": {"enabled": true},
                 "expert_stream": {"enabled": true, "slots": 8000}}
  })";
  vllm::OffloadConfig off = vllm::parse_offload_config_json(both);
  off.Validate();
  CHECK(off.offload_backend == vllm::OffloadBackend::kUva);
  CHECK(off.uva.cpu_offload_gb == doctest::Approx(10.0));
  CHECK(off.uva.cpu_offload_params.count("experts") == 1);
  CHECK(off.is_offloading_enabled());
  CHECK(off.warnings.empty());

  const vllm::WeightResidencyConfig res =
      vllm::parse_weight_residency_extension_json(both);
  REQUIRE(res.mmap.has_value());
  CHECK(*res.mmap == true);
  REQUIRE(res.expert_stream_slots.has_value());
  CHECK(*res.expert_stream_slots == 8000);

  // And a document carrying ONLY the extension leaves the mirrored config
  // completely inert — no backend selected, nothing offloaded to host RAM.
  vllm::OffloadConfig only_ext = vllm::parse_offload_config_json(
      R"({"vllm_cpp":{"expert_stream":{"enabled":true}}})");
  only_ext.Validate();
  CHECK_FALSE(only_ext.ResolvedBackend().has_value());
  CHECK_FALSE(only_ext.is_offloading_enabled());
  CHECK(only_ext.uva.cpu_offload_gb == doctest::Approx(0.0));
  CHECK(only_ext.prefetch.offload_group_size == 0);
}

TEST_CASE("residency config: precedence is env > config > built-in default") {
  ResidencyFixture fx;

  // 1. Neither set: the built-in default, both polarities. This case is also the
  // inertness proof — an engine with no config and no environment resolves
  // exactly what getenv resolved before this row existed.
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", std::nullopt,
                                   true) == true);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", std::nullopt,
                                   false) == false);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", std::nullopt,
                                    64) == 64);

  // 2. Config set, environment unset: the config wins over the default, in BOTH
  // directions — a config has to be able to turn a default-on knob off.
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, true) ==
        false);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", true, false) ==
        true);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) ==
        8000);

  // 3. Both set: the ENVIRONMENT wins. This is the constraint the row exists
  // under: these variables are how a benchmark arm is switched without restarting
  // the server with a new document, and an A/B in flight depends on it.
  ::setenv("VT_RESIDENCY_TEST_BOOL", "1", 1);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, false) ==
        true);
  ::setenv("VT_RESIDENCY_TEST_COUNT", "128", 1);
  CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) == 128);

  // 4. And the override can turn a configured knob OFF, which is the direction
  // that matters: an env var that could only enable things would be useless for
  // the arm that measures the feature disabled. Every falsy spelling the tree
  // already honours is checked, because this resolver replaced `EnvOn` and must
  // not narrow it.
  for (const char* off : {"0", "", "false", "off"}) {
    CAPTURE(off);
    ::setenv("VT_RESIDENCY_TEST_BOOL", off, 1);
    CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", true, true) ==
          false);
  }
  // Anything else is on, including a value that is not a recognised word.
  ::setenv("VT_RESIDENCY_TEST_BOOL", "yes", 1);
  CHECK(vllm::ResolveResidencyBool("VT_RESIDENCY_TEST_BOOL", false, false) ==
        true);

  // 5. A garbage or non-positive COUNT in the environment falls THROUGH to the
  // config rather than to the default. The existing readers ignore such a value
  // (atol, then `if (v > 0)`), and this row must not change what an
  // environment-only run resolves.
  for (const char* junk : {"0", "-5", "banana", ""}) {
    CAPTURE(junk);
    ::setenv("VT_RESIDENCY_TEST_COUNT", junk, 1);
    CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", 8000, 64) ==
          8000);
    CHECK(vllm::ResolveResidencyCount("VT_RESIDENCY_TEST_COUNT", std::nullopt,
                                      64) == 64);
  }
}

TEST_CASE("residency config: each knob's own resolver keeps its own env name and polarity") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");

  // Defaults first, with NOTHING installed and nothing exported. These are the
  // values the tree resolved before this row existed, and they are the inertness
  // floor: `VT_GGUF_MMAP` rides the caller's availability predicate,
  // `VT_GGUF_PREFAULT` is ON when unset (which docs/ENVIRONMENT.md got backwards
  // until #1109), slots is 64, slot_bytes is whatever the caller computed.
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == true);
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/false) == false);
  CHECK(vllm::ResolveGgufPrefault() == true);
  CHECK(vllm::ResolveExpertStreamSlots() == 64);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 12582912);

  // Now the config, which must reach every one of them. Installing after those
  // resolves is LEGAL and that is itself part of the contract: not one of them
  // latches anything. `GgufLoadPolicy::FromEnv()` is called per load, the prefault
  // site no longer caches, and the two sizes are frozen by the slot STORE being
  // built rather than by being read. The reset below only clears the installed
  // document so the assertions start from a known state.
  vllm::ResetWeightResidencyConfigForTesting();
  vllm::WeightResidencyConfig cfg;
  cfg.mmap = true;
  cfg.prefault = false;
  cfg.expert_stream_slots = 8000;
  cfg.expert_stream_slot_bytes = 33554432;
  vllm::SetWeightResidencyConfig(cfg);

  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/false) == true);
  CHECK(vllm::ResolveGgufPrefault() == false);
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 33554432);

  // And the environment must beat it, per knob, using each knob's OWN variable —
  // a resolver wired to the wrong name would pass every case above and fail here.
  ::setenv("VT_GGUF_MMAP", "0", 1);
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == false);
  ::setenv("VT_GGUF_PREFAULT", "1", 1);
  CHECK(vllm::ResolveGgufPrefault() == true);
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "128", 1);
  CHECK(vllm::ResolveExpertStreamSlots() == 128);
  ::setenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES", "4096", 1);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 4096);

  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");
}

TEST_CASE("residency config: the expert-stream FIRST-CHARACTER env rule survives") {
  // `VT_MOE_EXPERT_STREAM` does NOT use the tree's whole-value polarity: only the
  // first character is examined, so `false` and `off` read as ON. That is what
  // docs/ENVIRONMENT.md documents and what `qwen3_5.cpp` did, so a row whose
  // subject is where a value COMES FROM must not also change what a value MEANS.
  //
  // Tested through the pure form because the resolver latches and can be
  // exercised only once per process — which is exactly how a normalisation here
  // would have escaped notice.
  CHECK(vllm::ExpertStreamRequestedFrom("1", std::nullopt) == true);
  CHECK(vllm::ExpertStreamRequestedFrom("0", std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("", std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("0abc", std::nullopt) == false);
  // The deliberately odd ones. Under the tree's ordinary polarity these would be
  // OFF; here they are ON, and that is the documented contract.
  CHECK(vllm::ExpertStreamRequestedFrom("false", std::nullopt) == true);
  CHECK(vllm::ExpertStreamRequestedFrom("off", std::nullopt) == true);

  // Config supplies the answer only when the variable is UNSET, and the variable
  // beats the config in both directions.
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, std::nullopt) == false);
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, true) == true);
  CHECK(vllm::ExpertStreamRequestedFrom(nullptr, false) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("0", true) == false);
  CHECK(vllm::ExpertStreamRequestedFrom("1", false) == true);
}

TEST_CASE("residency config: an override is reported only when it would WIN") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");

  vllm::WeightResidencyConfig cfg;
  cfg.mmap = true;
  cfg.expert_stream_slots = 8000;
  // Installed as well as described, so the report below can be checked against what
  // the resolver actually does with the same variable.
  vllm::SetWeightResidencyConfig(cfg);

  // Nothing exported: nothing shadowed. This is the line the overwhelming majority
  // of runs never print.
  CHECK(cfg.DescribeEnvOverrides().empty());

  // A BOOLEAN takes any value, so presence is exact.
  ::setenv("VT_GGUF_MMAP", "0", 1);
  CHECK(Mentions(cfg.DescribeEnvOverrides(), "VT_GGUF_MMAP (mmap)"));
  ::unsetenv("VT_GGUF_MMAP");

  // A COUNT does not. Under the tolerant parse the existing readers have always
  // used, an empty, garbage or non-positive value falls THROUGH to the config, so it
  // overrides NOTHING and announcing it sends the operator after a line the resolver
  // ignores. Each of these is checked against the resolver in the same breath, so
  // the report and the resolution cannot drift.
  for (const char* junk : {"banana", "0", "-5", ""}) {
    CAPTURE(junk);
    ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", junk, 1);
    // The resolver keeps the CONFIG's value, so the variable overrode nothing...
    CHECK(vllm::ResolveExpertStreamSlots() == 8000);
    // ...and the report says nothing.
    CHECK(cfg.DescribeEnvOverrides().empty());
  }

  // ...and a value that DOES win is reported, by the same predicate.
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "128", 1);
  CHECK(vllm::ResolveExpertStreamSlots() == 128);
  CHECK(Mentions(cfg.DescribeEnvOverrides(),
                 "VT_MOE_EXPERT_STREAM_SLOTS (expert_stream_slots)"));
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");

  // A variable that shadows a field the document did NOT set is not an override
  // either: there is nothing for it to override.
  vllm::WeightResidencyConfig only_mmap;
  only_mmap.mmap = true;
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "128", 1);
  CHECK(only_mmap.DescribeEnvOverrides().empty());
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
}

TEST_CASE("residency config: install is readable, and a LATE install of a LATCHED knob throws") {
  ResidencyFixture fx;

  CHECK(vllm::ActiveWeightResidencyConfig().empty());
  CHECK_FALSE(vllm::WeightResidencyLatched());

  vllm::WeightResidencyConfig cfg;
  cfg.expert_stream = true;
  cfg.expert_stream_slots = 8000;
  vllm::SetWeightResidencyConfig(cfg);
  CHECK(vllm::ActiveWeightResidencyConfig() == cfg);
  CHECK_FALSE(vllm::WeightResidencyLatched());

  // Resolving the STREAMING knob latches, because that answer is cached in a
  // function-local static and it decides both whether an ~18 GiB slot store is
  // built and whether the grouped-MoE path is disabled. Reading it through the
  // production resolver is what marks the latch, so this is not a test hook.
  CHECK(vllm::ResolveExpertStreamRequested() == true);
  CHECK(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));
  CHECK(vllm::WeightResidencyLatched());
  // The GEOMETRY is a separate latch: the store is built once per process, and
  // nothing is frozen until it is.
  CHECK_FALSE(
      vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStreamGeometry));

  // A config that would CHANGE the latched answer cannot be honoured, so it throws
  // instead of being recorded and ignored, and the message names the field.
  vllm::WeightResidencyConfig later;
  later.expert_stream = false;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(later), std::logic_error);
  CHECK(vllm::ActiveWeightResidencyConfig() == cfg);
  try {
    vllm::SetWeightResidencyConfig(later);
    FAIL("a late expert_stream change must throw");
  } catch (const std::logic_error& e) {
    CHECK(Mentions(e.what(), "expert_stream"));
  }

  // But a knob that latched NOTHING is still settable — this is the two-model
  // process the coarse check used to fail. `mmap` and `prefault` are resolved per
  // load, so a second engine may change them even after streaming latched.
  //
  // NOTE THE COPY, and note what it therefore does NOT cover. `mmap_too` carries
  // `cfg`'s `expert_stream` and `expert_stream_slots` as well as the two new fields,
  // so the second install sets the latched field to the value it already had. That is
  // a real shape — one document, two engines — but it is not the shape a second
  // engine with its OWN partial document has, and both of the #1133 behaviour defects
  // lived in the difference. "TWO DIFFERENT PARTIAL documents in one process", below,
  // is the case that covers it.
  vllm::WeightResidencyConfig mmap_too = cfg;
  mmap_too.mmap = true;
  mmap_too.prefault = false;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(mmap_too));
  CHECK(vllm::ActiveWeightResidencyConfig() == mmap_too);

  // Re-installing the SAME config is fine — that is what a process loading two
  // engines with one configuration does.
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(mmap_too));

  // And an EMPTY install is always fine: it is the no-op every default load
  // performs, and refusing it would break every engine that has no config.
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(vllm::WeightResidencyConfig{}));
  // It must be a NO-OP rather than an overwrite. A second engine in the same
  // process carries no residency config of its own, and clearing the first one's
  // would change what the expert slot store reads — it is built lazily, on the
  // first slice taken, which can be long after a second engine loaded. Since the
  // install merges field by field, this now holds by construction rather than by a
  // special case for the empty document.
  CHECK(vllm::ActiveWeightResidencyConfig() == mmap_too);
}

TEST_CASE("residency config: the SLOT GEOMETRY latches when the store is built, and only then") {
  ResidencyFixture fx;

  // Reading the two sizes freezes nothing. The store does, and it reports what it
  // was built with — which is the only observable the sizes have.
  CHECK(vllm::ResolveExpertStreamSlots() == 64);
  CHECK(vllm::ResolveExpertStreamSlotBytes(12582912) == 12582912);
  CHECK_FALSE(
      vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStreamGeometry));

  vllm::WeightResidencyConfig sizes;
  sizes.expert_stream_slots = 8000;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(sizes));
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);

  // The store's constructor reports the geometry it used. From here the reservation
  // exists and its size cannot change.
  vllm::NoteExpertStreamGeometry(8000, 12582912);
  CHECK(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStreamGeometry));
  CHECK(vllm::BuiltExpertStreamGeometry().slots == 8000);

  vllm::WeightResidencyConfig resize;
  resize.expert_stream_slots = 96;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(resize), std::logic_error);

  // ...while mmap, which the geometry does not freeze, still installs. Another COPY
  // (`= sizes`), so like `mmap_too` above it re-states the frozen `slots` at its
  // existing value instead of omitting it; the two-partial-documents case below is
  // the one that omits it.
  vllm::WeightResidencyConfig mmap_only = sizes;
  mmap_only.mmap = true;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(mmap_only));
  CHECK(vllm::ActiveWeightResidencyConfig() == mmap_only);
}

TEST_CASE("residency config: TWO DIFFERENT PARTIAL documents in one process") {
  ResidencyFixture fx;
  ::unsetenv("VT_MOE_EXPERT_STREAM");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOT_BYTES");
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");

  // THE SHAPE NO CASE IN THIS FILE HAD, and both #1133 behaviour defects hid in
  // exactly the gap it leaves. Every other latch case here either re-installs a
  // COPY of the first document (`mmap_too = cfg`, `mmap_only = sizes`) or installs
  // the empty one, so the second install always carried the latched field with the
  // same value it already had — and the two ways `optional` was misused are both
  // invisible to that. `FrozenFields` compared `in.expert_stream` against the
  // STORED optional, and `nullopt != engaged` is true, so a document that merely
  // OMITS the latched field was refused. The install then assigned wholesale, so a
  // document that omitted a field CLEARED it. A copy of the first document triggers
  // neither. A genuinely partial one triggers both.
  //
  // Engine A: the reproduction document from #1110, parsed rather than hand-built.
  const vllm::WeightResidencyConfig a =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},)"
          R"("expert_stream":{"enabled":true,"slots":8000}}})");
  vllm::SetWeightResidencyConfig(a);

  // A's load takes both decisions: the streaming answer on the first routed slice,
  // the geometry when the slot store is constructed. Both are taken through the
  // production functions, so this is the state a real first engine leaves behind.
  //
  // `decided` is READ rather than assumed. The streaming answer is a per-process
  // static, so its value depends on which case in this binary reached it first, and
  // a case that hardcoded `true` would pass or fail on test ORDER rather than on the
  // code. Every assertion below is relative to it.
  const bool decided = vllm::ResolveExpertStreamRequested();
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));
  vllm::NoteExpertStreamGeometry(vllm::ResolveExpertStreamSlots(),
                                 vllm::ResolveExpertStreamSlotBytes(12582912));
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStreamGeometry));
  REQUIRE(vllm::BuiltExpertStreamGeometry().slots == 8000);

  // Engine B: mmap only. Not a copy of A — it sets one field and omits four.
  vllm::WeightResidencyConfig b;
  b.mmap = false;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(b));
  {
    // ACCEPTED (#1133 H1: this threw std::logic_error out of FromModelDir, so a
    // legal second engine returned VLLM_ERR_MODEL_LOAD)...
    const vllm::WeightResidencyConfig now = vllm::ActiveWeightResidencyConfig();
    REQUIRE(now.mmap.has_value());
    CHECK(*now.mmap == false);
    // ...and A's four other fields SURVIVED (#1133 H2: `expert_stream=on` with
    // 8000 slots became OFF with 64, with no diagnostic).
    REQUIRE(now.prefault.has_value());
    CHECK(*now.prefault == false);
    REQUIRE(now.expert_stream.has_value());
    CHECK(*now.expert_stream == true);
    REQUIRE(now.expert_stream_slots.has_value());
    CHECK(*now.expert_stream_slots == 8000);
  }
  // The resolvers are what the slot store reads, and it reads them LAZILY — on the
  // first slice taken, which can be long after a second engine loaded. So the drop
  // is asserted where it would be felt, not only on the stored struct.
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == false);

  // Engine C: prefault only. THE SECOND, DIFFERENT partial document — this is the
  // half that makes the case the shape the review named, because it also proves the
  // merge accumulates rather than remembering only the first two installs.
  vllm::WeightResidencyConfig c;
  c.prefault = true;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(c));
  {
    const vllm::WeightResidencyConfig now = vllm::ActiveWeightResidencyConfig();
    REQUIRE(now.prefault.has_value());
    CHECK(*now.prefault == true);
    // B's field survived C, as A's survived B.
    REQUIRE(now.mmap.has_value());
    CHECK(*now.mmap == false);
    REQUIRE(now.expert_stream.has_value());
    CHECK(*now.expert_stream == true);
    REQUIRE(now.expert_stream_slots.has_value());
    CHECK(*now.expert_stream_slots == 8000);
  }
  CHECK(vllm::ResolveGgufPrefault() == true);
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);

  // And the refusal still fires on what it can justify: a document that would make
  // a resolver return something OTHER than the decision already taken.
  vllm::WeightResidencyConfig flip;
  flip.expert_stream = !decided;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(flip), std::logic_error);
  vllm::WeightResidencyConfig resize;
  resize.expert_stream_slots = 96;
  try {
    vllm::SetWeightResidencyConfig(resize);
    FAIL("resizing the built slot store must throw");
  } catch (const std::logic_error& e) {
    CHECK(Mentions(e.what(), "expert_stream_slots"));
    // The message quotes THE DECISION THE ENGINE TOOK, which is the number the
    // operator has to reconcile their document against. It used to quote the stored
    // document while asserting the engine was not running it.
    CHECK(Mentions(e.what(), "expert_stream_slots=8000"));
  }
  // Neither refused install recorded anything.
  const vllm::WeightResidencyConfig after = vllm::ActiveWeightResidencyConfig();
  REQUIRE(after.expert_stream_slots.has_value());
  CHECK(*after.expert_stream_slots == 8000);
  REQUIRE(after.expert_stream.has_value());
  CHECK(*after.expert_stream == true);
}

TEST_CASE("residency config: a document that AGREES with the decision taken is accepted") {
  ResidencyFixture fx;
  ::unsetenv("VT_MOE_EXPERT_STREAM");

  // THE PRODUCTION SHAPE: streaming turned on by the ENVIRONMENT, so engine A
  // installs no document at all and the decision is taken with the stored config
  // EMPTY. Engine B then arrives with `{"vllm_cpp":{"expert_stream":{"enabled":
  // true}}}` — asking for exactly what the process resolved — and was refused with
  // a message saying "accepting this would record a configuration the engine is not
  // running". The engine WAS running it.
  //
  // Reproduced without the variable, because the streaming answer is a per-process
  // static that this binary can observe only once: the reset clears the stored
  // config and both latch flags, the second call re-marks the latch and returns the
  // SAME cached answer, and that pair — a decision taken, nothing stored — is the
  // state the variable produces. `decided` is read, never assumed, for the same
  // reason as in the case above.
  const bool decided = vllm::ResolveExpertStreamRequested();
  vllm::ResetWeightResidencyConfigForTesting();
  REQUIRE(vllm::ActiveWeightResidencyConfig().empty());
  REQUIRE(vllm::ResolveExpertStreamRequested() == decided);
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));

  // The DISAGREEING document first, while nothing is stored, because that is the one
  // moment the refusal MESSAGE can be checked against the right thing. With the
  // stored document empty, quoting it produced "environment/default" — no number and
  // no answer for the operator to reconcile against. It now quotes the DECISION.
  vllm::WeightResidencyConfig disagrees;
  disagrees.expert_stream = !decided;
  try {
    vllm::SetWeightResidencyConfig(disagrees);
    FAIL("a document that would change the decided answer must throw");
  } catch (const std::logic_error& e) {
    CHECK(Mentions(e.what(), decided ? "expert_stream=on" : "expert_stream=off"));
    CHECK_FALSE(Mentions(e.what(), "environment/default"));
  }

  // ...and the AGREEING one installs. The pair is the point: "equal to the stored
  // document" and "equal to the decision" are different predicates, and only the
  // second is what the refusal can justify.
  vllm::WeightResidencyConfig agrees;
  agrees.expert_stream = decided;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(agrees));
  REQUIRE(vllm::ActiveWeightResidencyConfig().expert_stream.has_value());
  CHECK(*vllm::ActiveWeightResidencyConfig().expert_stream == decided);
}

TEST_CASE("residency config: a document the ENVIRONMENT overrides is not a change") {
  ResidencyFixture fx;
  ::unsetenv("VT_MOE_EXPERT_STREAM");
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");

  // The refusal asks what the document WOULD RESOLVE TO, not what it says, and this
  // case is the difference. With the knob's variable exported the resolver's answer
  // cannot change whatever the document says, so a document that contradicts the
  // decision changes nothing and there is nothing to refuse. Refusing it would fail a
  // legal load for a document with no effect.
  const bool decided = vllm::ResolveExpertStreamRequested();
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));

  ::setenv("VT_MOE_EXPERT_STREAM", decided ? "1" : "0", 1);
  vllm::WeightResidencyConfig flip;
  flip.expert_stream = !decided;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(flip));
  ::unsetenv("VT_MOE_EXPERT_STREAM");
  // ...and WITHOUT the variable the same document is refused, which is what shows the
  // acceptance above came from the variable rather than from an absent check.
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(flip), std::logic_error);

  // The same for a COUNT, through the same tolerant predicate the resolver uses.
  vllm::NoteExpertStreamGeometry(8000, 12582912);
  REQUIRE(vllm::BuiltExpertStreamGeometry().slots == 8000);
  vllm::WeightResidencyConfig resize;
  resize.expert_stream_slots = 96;
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "8000", 1);
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(resize));
  // A garbage value is NOT an override — the resolver falls through to the config —
  // so the document takes effect again and is refused again.
  ::setenv("VT_MOE_EXPERT_STREAM_SLOTS", "banana", 1);
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(resize), std::logic_error);
  ::unsetenv("VT_MOE_EXPERT_STREAM_SLOTS");
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(resize), std::logic_error);
}

TEST_CASE("residency config: reading mmap or prefault latches NOTHING, so a second engine may set them") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");

  // THE CASE THE COARSE LATCH FAILED. The first pass marked a process-wide latch
  // inside the shared resolvers, so ANY resolve — including the `GgufLoadPolicy`
  // read that every GGUF load performs — refused every later non-empty install.
  // Measured through the public ABI at the time: load model A with no residency
  // config, then load model B carrying `vllm_cpp`, and B could not load. Neither of
  // these two knobs freezes anything, so neither may block.
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == true);
  CHECK(vllm::ResolveGgufPrefault() == true);
  CHECK_FALSE(vllm::WeightResidencyLatched());

  vllm::WeightResidencyConfig cfg;
  cfg.mmap = false;
  cfg.prefault = false;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(cfg));
  // And the new document is what the next load reads, which is the whole reason
  // accepting it is correct rather than merely permissive.
  CHECK(vllm::ResolveGgufMmap(/*builtin_default=*/true) == false);
  CHECK(vllm::ResolveGgufPrefault() == false);
}

// ─── W2: `vllm_cpp.device_fit.weight_budget_bytes` (issue #1127) ──────────────
//
// The sixth knob, and the only one whose legal range includes ZERO. `slots` and
// `slot_bytes` are sizes, so a zero there is refused: a slot count that silently
// became 64 is a cache the operator does not have. `0` on the budget is the
// DOCUMENTED spelling of "suppress the device-fit refusal", identical to
// `VT_DEVICE_WEIGHT_BUDGET_BYTES=0`, because `CheckDeviceWeightFit` reads a zero
// budget as UNKNOWN and decides nothing. Refusing it would remove the escape hatch
// the key exists to give, so it parses through its own non-negative helper.

TEST_CASE("residency config: the device-fit budget parses, and ZERO is legal") {
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":137438953472}}})");
  REQUIRE(c.device_weight_budget_bytes.has_value());
  CHECK(*c.device_weight_budget_bytes == 137438953472LL);
  CHECK_FALSE(c.empty());
  // It reaches the install line under its own name, so an operator reading the
  // line can tell a budget was set from a budget that fell through to the probe.
  CHECK(Mentions(c.Describe(), "device_weight_budget_bytes=137438953472"));

  // ZERO. This is the suppression spelling and it must survive the parse as an
  // ENGAGED optional: `nullopt` would fall through to the probe, which is the
  // opposite of what the operator asked for.
  const vllm::WeightResidencyConfig zero =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":0}}})");
  REQUIRE(zero.device_weight_budget_bytes.has_value());
  CHECK(*zero.device_weight_budget_bytes == 0);
  CHECK_FALSE(zero.empty());

  // An absent `device_fit` object leaves it unset, and an empty one does too.
  CHECK_FALSE(vllm::parse_weight_residency_extension_json(
                  R"({"vllm_cpp":{"mmap":{"enabled":true}}})")
                  .device_weight_budget_bytes.has_value());
  const vllm::WeightResidencyConfig empty_obj =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{}}})");
  CHECK_FALSE(empty_obj.device_weight_budget_bytes.has_value());
  CHECK(empty_obj.empty());
}

TEST_CASE("residency config: a misspelled device_fit key is REFUSED, never ignored") {
  // The whole reason this parser enumerates: the mirrored parser ignores what it
  // does not know, so a typo in a new level would start a server whose device-fit
  // budget is the probe while the operator believes it is the number typed. On a
  // box where the probe is smaller than the checkpoint, that typo is the refusal
  // the operator was trying to suppress.
  const std::string level =
      RefusalMessage(R"({"vllm_cpp":{"device_fitt":{"weight_budget_bytes":1}}})");
  CHECK(Mentions(level, "unknown key \"vllm_cpp.device_fitt\""));
  CHECK(Mentions(level, "expected one of: mmap expert_stream device_fit"));

  // The hyphenated spelling of the new level, for the same reason the hyphenated
  // `vllm-cpp` is pinned: every flag around it is hyphenated.
  CHECK(Mentions(
      RefusalMessage(R"({"vllm_cpp":{"device-fit":{"weight_budget_bytes":1}}})"),
      "unknown key \"vllm_cpp.device-fit\""));

  // And the field inside it. `weight_budget_byte` is the likeliest of these,
  // because the singular reads correctly in English and the plural is the name.
  const std::string field =
      RefusalMessage(R"({"vllm_cpp":{"device_fit":{"weight_budget_byte":1}}})");
  CHECK(Mentions(field, "unknown key \"vllm_cpp.device_fit.weight_budget_byte\""));
  CHECK(Mentions(field, "expected one of: weight_budget_bytes"));

  // The name a reader might carry over from the environment variable.
  CHECK(Mentions(RefusalMessage(
                     R"({"vllm_cpp":{"device_fit":{"device_weight_budget_bytes":1}}})"),
                 "unknown key \"vllm_cpp.device_fit.device_weight_budget_bytes\""));
}

TEST_CASE("residency config: a wrong-typed or NEGATIVE budget is REFUSED") {
  CHECK(Mentions(
      RefusalMessage(R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":"10"}}})"),
      "\"vllm_cpp.device_fit.weight_budget_bytes\" must be an integer"));
  CHECK(Mentions(
      RefusalMessage(R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":1.5}}})"),
      "\"vllm_cpp.device_fit.weight_budget_bytes\" must be an integer"));
  CHECK(Mentions(
      RefusalMessage(R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":true}}})"),
      "\"vllm_cpp.device_fit.weight_budget_bytes\" must be an integer"));

  // NEGATIVE is refused and ZERO is not, which is the whole distinction between
  // this field's helper and the one `slots` uses. A message that said "positive"
  // here would be lying about a value the parser accepts.
  const std::string neg = RefusalMessage(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":-1}}})");
  CHECK(Mentions(neg, "\"vllm_cpp.device_fit.weight_budget_bytes\""));
  CHECK(Mentions(neg, "must not be negative"));
  CHECK(Mentions(neg, "-1"));

  // `device_fit` itself must be an object, and the message names the DOCUMENT
  // path rather than a hardcoded prefix.
  CHECK(Mentions(RefusalMessage(R"({"vllm_cpp":{"device_fit":5}})"),
                 "\"vllm_cpp.device_fit\" must be a JSON object"));
}

TEST_CASE("residency config: the budget resolves env > config > probed total") {
  ResidencyFixture fx;
  constexpr size_t kProbed = 128ULL * 1024 * 1024 * 1024;

  // Neither input: the probe stands, byte-for-byte as before this key existed.
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == kProbed);

  // Config only.
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":4096}}})"));
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == 4096U);

  // A config ZERO reaches the resolver as zero, which `CheckDeviceWeightFit`
  // reads as UNKNOWN and therefore as "do not refuse". The suppression spelling
  // has to survive the resolver as well as the parser.
  vllm::ResetWeightResidencyConfigForTesting();
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":0}}})"));
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == 0U);

  // Environment beats config, in BOTH directions. A benchmark arm switched by an
  // exported variable is why this precedence exists, and an override that could
  // not raise the budget back would not be one.
  ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "8192", 1);
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == 8192U);
  ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "0", 1);
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == 0U);

  // THE ENVIRONMENT GRAMMAR IS UNCHANGED: decimal digits only. A signed, spaced
  // or garbage value is IGNORED, and what it now falls through to is the CONFIG
  // rather than the probe. Reading "-1" as a budget would wrap to ULLONG_MAX and
  // silently disable the guard, which is why the grammar is explicit.
  vllm::ResetWeightResidencyConfigForTesting();
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":4096}}})"));
  for (const char* bad : {"-1", " 64", "64x", "", "banana", "+64"}) {
    ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", bad, 1);
    CAPTURE(bad);
    CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == 4096U);
  }
  // ...and with no config either, the same bad values leave the probe standing.
  vllm::ResetWeightResidencyConfigForTesting();
  for (const char* bad : {"-1", " 64", "64x", "", "banana"}) {
    ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", bad, 1);
    CAPTURE(bad);
    CHECK(vllm::ResolveDeviceWeightBudgetBytes(kProbed) == kProbed);
  }
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
}

TEST_CASE("residency config: the budget's override note asks whether the variable WINS") {
  ResidencyFixture fx;
  const vllm::WeightResidencyConfig c =
      vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":4096}}})");

  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
  CHECK(c.DescribeEnvOverrides().empty());

  ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "8192", 1);
  const std::string note = c.DescribeEnvOverrides();
  CHECK(Mentions(note, "VT_DEVICE_WEIGHT_BUDGET_BYTES"));
  CHECK(Mentions(note, "device_weight_budget_bytes"));

  // `0` IS an override — it is the suppression value, not an absent one — so it
  // must be announced. A predicate that tested for a positive number would drop
  // exactly the value an operator is most surprised to have inherited.
  ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "0", 1);
  CHECK(Mentions(c.DescribeEnvOverrides(), "VT_DEVICE_WEIGHT_BUDGET_BYTES"));

  // A value the resolver IGNORES is not an override, and announcing it would send
  // the operator after a line that decides nothing (the #1122 L7 shape).
  for (const char* bad : {"-1", " 64", "64x", "banana"}) {
    ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", bad, 1);
    CAPTURE(bad);
    CHECK(c.DescribeEnvOverrides().empty());
  }

  // And a document that does not SET the budget is never reported for it,
  // whatever the variable says.
  ::setenv("VT_DEVICE_WEIGHT_BUDGET_BYTES", "8192", 1);
  CHECK_FALSE(Mentions(vllm::parse_weight_residency_extension_json(
                           R"({"vllm_cpp":{"mmap":{"enabled":true}}})")
                           .DescribeEnvOverrides(),
                       "VT_DEVICE_WEIGHT_BUDGET_BYTES"));
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");
}

TEST_CASE("residency config: ABSENT MEANS UNCHANGED for the budget, at both ends") {
  ResidencyFixture fx;
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  // Two DIFFERENT partial documents, which is the shape that caught #1133 H2: a
  // second document that restates the first's field cannot show a dropped field.
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":4096},)"
      R"("expert_stream":{"slots":8000}}})"));
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"mmap":{"enabled":true}}})"));

  const vllm::WeightResidencyConfig installed =
      vllm::ActiveWeightResidencyConfig();
  REQUIRE(installed.device_weight_budget_bytes.has_value());
  CHECK(*installed.device_weight_budget_bytes == 4096);
  REQUIRE(installed.expert_stream_slots.has_value());
  CHECK(*installed.expert_stream_slots == 8000);
  REQUIRE(installed.mmap.has_value());
  CHECK(*installed.mmap == true);
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(999) == 4096U);

  // And a later document that DOES set it overwrites it, because that is what
  // "set" means. `0` is a set value, so it must overwrite too rather than read as
  // "unchanged" — the one place where the suppression spelling and the absent
  // spelling would be confusable.
  vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":0}}})"));
  REQUIRE(vllm::ActiveWeightResidencyConfig()
              .device_weight_budget_bytes.has_value());
  CHECK(*vllm::ActiveWeightResidencyConfig().device_weight_budget_bytes == 0);
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(999) == 0U);
}

TEST_CASE("residency config: the budget LATCHES NOTHING, so a late install is accepted") {
  ResidencyFixture fx;
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  // Read it, which is what a completed GGUF load does at its fit check. Nothing
  // caches the answer, so nothing freezes.
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(4096) == 4096U);
  CHECK_FALSE(vllm::WeightResidencyLatched());

  // A second engine may therefore still set it, and the refusal must not fire.
  // `expert_stream` and the slot geometry are the only two decisions that freeze,
  // and this key is in neither.
  CHECK_NOTHROW(
      vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":8192}}})")));
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(4096) == 8192U);

  // Even once the streaming decision HAS been taken, a budget-only document is
  // not a change to it and is accepted.
  (void)vllm::ResolveExpertStreamRequested();
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));
  CHECK_NOTHROW(
      vllm::SetWeightResidencyConfig(vllm::parse_weight_residency_extension_json(
          R"({"vllm_cpp":{"device_fit":{"weight_budget_bytes":16384}}})")));
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(4096) == 16384U);
}

TEST_CASE("residency config: the SETTER refuses what the PARSER refuses") {
  ResidencyFixture fx;
  ::unsetenv("VT_DEVICE_WEIGHT_BUDGET_BYTES");

  // THE OTHER DOOR. Every production caller reaches the process-global through
  // `parse_weight_residency_extension_json`, which refuses a negative budget and a
  // non-positive slot size. But `SetWeightResidencyConfig` is declared in a public
  // header and takes the STRUCT, so a hand-built config is a legal way in that
  // skips those rules entirely, and the resolver's own comment used to justify its
  // cast by trusting a parser that had not necessarily run.
  //
  // The budget is the dangerous one rather than merely the wrong one:
  // `ResolveDeviceWeightBudgetBytes` casts to `size_t`, so an installed `-1`
  // resolves to SIZE_MAX — an effectively infinite budget that switches the
  // load-time device-fit refusal OFF without a word. That is the exact
  // "a budget the operator believes is set" failure the parser's own refusal text
  // names.
  vllm::WeightResidencyConfig negative_budget;
  negative_budget.device_weight_budget_bytes = -1;
  std::string message;
  try {
    vllm::SetWeightResidencyConfig(negative_budget);
    message = "ACCEPTED (no throw)";
  } catch (const std::invalid_argument& e) {
    message = e.what();
  } catch (const std::exception& e) {
    message = std::string("WRONG EXCEPTION TYPE: ") + e.what();
  }
  CAPTURE(message);
  CHECK(Mentions(message, "device_weight_budget_bytes"));
  CHECK(Mentions(message, "must not be negative"));
  CHECK(Mentions(message, "-1"));

  // ...and nothing was installed, so the check the refusal protects still reads
  // the probe. Without this the case would pass on an implementation that threw
  // AFTER writing the value.
  CHECK_FALSE(
      vllm::ActiveWeightResidencyConfig().device_weight_budget_bytes.has_value());
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(4096) == 4096U);

  // The two sizes get the parser's rule at this door too, so the two doors into
  // one struct state one rule.
  vllm::WeightResidencyConfig zero_slots;
  zero_slots.expert_stream_slots = 0;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(zero_slots),
                  std::invalid_argument);
  vllm::WeightResidencyConfig negative_slot_bytes;
  negative_slot_bytes.expert_stream_slot_bytes = -8;
  CHECK_THROWS_AS(vllm::SetWeightResidencyConfig(negative_slot_bytes),
                  std::invalid_argument);

  // Everything the PARSER accepts, the setter still accepts — `0` for the budget
  // above all, because it is this field's suppression spelling and a guard that
  // refused it would delete the escape hatch the key exists to give.
  vllm::WeightResidencyConfig zero_budget;
  zero_budget.device_weight_budget_bytes = 0;
  CHECK_NOTHROW(vllm::SetWeightResidencyConfig(zero_budget));
  CHECK(vllm::ResolveDeviceWeightBudgetBytes(4096) == 0U);
}
