// SPEC-DRAFTER-CHAIN W1 (#1522) — the `vllm_cpp.drafter_chain` field of
// `--speculative-config`: its parsing, its validation and its refusals.
//
// W1 lands NO chain behaviour. Nothing resolves a chain, nothing changes which
// speculator drafts, and with the field absent the engine behaves exactly as it
// did. That inertness is the point of the wave, and it is asserted through this
// parser and through a production entry point in
// `tests/vllm/entrypoints/test_drafter_chain_reach.cpp` (which owns G1 and the
// "before any weight I/O" half of G5, because neither is a property of a
// parser).
//
// WHY THE KEY IS NESTED UNDER `vllm_cpp` (.agents/specs/drafter-chain.md D6).
// The field must not collide with a name vLLM's `SpeculativeConfig` declares
// "now or plausibly". A bare `drafter_chain` satisfies "now" and cannot satisfy
// "plausibly" — nothing stops upstream from choosing that spelling, and if it
// ever did, a document would silently mean two different things in the two
// engines. Nesting the extension under one `vllm_cpp` object makes the
// collision surface exactly ONE name, and a collision on THAT name would be a
// loud, visible one rather than a silent reinterpretation. It is also the
// landed convention of this tree: `--offload-config` carries this engine's
// non-vLLM residency tier under exactly the same `vllm_cpp` key
// (src/vllm/config/weight_residency.cpp:451-478).
//
// EVERY NEGATIVE CASE ASSERTS ON THE MESSAGE, never on the bare fact of a
// throw. A `CHECK_THROWS` here would be satisfied by the NEIGHBOURING guard:
// this document has a method check, an admission check, a value-gate check and
// four required-key checks, all throwing the same `std::invalid_argument`, so
// "it threw" is evidence for nothing about which guard ran.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "vllm/config/speculative.h"

using vllm::ParseSpeculativeConfigJson;
using vllm::SpeculativeConfig;

namespace {

// The refusal message, or "" when the parse SUCCEEDED — the same helper shape
// `test_speculative_unknown_keys.cpp` uses, and for the same reason: a case has
// to be able to assert the offending name is IN the message.
std::string RefusalMessage(const std::string& json_text) {
  try {
    ParseSpeculativeConfigJson(json_text);
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

bool Mentions(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// A chain document. `entries` is the raw JSON array body.
std::string Chain(const std::string& entries) {
  return R"({"vllm_cpp":{"drafter_chain":[)" + entries + "]}}";
}

constexpr const char* kMtp = R"({"method":"mtp"})";
constexpr const char* kNgram = R"({"method":"ngram","num_speculative_tokens":4})";
constexpr const char* kDflash =
    R"({"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash",)"
    R"("num_speculative_tokens":16})";

}  // namespace

TEST_CASE("a two-entry chain parses into the entries, in the order given") {
  const SpeculativeConfig cfg =
      ParseSpeculativeConfigJson(Chain(std::string(kDflash) + "," + kNgram));

  REQUIRE(cfg.use_drafter_chain());
  REQUIRE(cfg.drafter_chain.size() == 2);

  // ORDER IS THE SEMANTICS. A suite that only checked the SET would pass under
  // the reversed chain too, and a preference list whose order is untested is
  // not a preference list (.agents/specs/drafter-chain.md `## Tests to port`).
  CHECK(cfg.drafter_chain[0].method == "dflash");
  CHECK(cfg.drafter_chain[1].method == "ngram");

  // Each entry carries its OWN configuration. This is why the shape is a list
  // of objects and not llama.cpp's comma-separated name string.
  REQUIRE(cfg.drafter_chain[0].draft_model_path.has_value());
  CHECK(*cfg.drafter_chain[0].draft_model_path == "z-lab/Qwen3.6-27B-DFlash");
  REQUIRE(cfg.drafter_chain[0].num_speculative_tokens.has_value());
  CHECK(*cfg.drafter_chain[0].num_speculative_tokens == 16);
  CHECK_FALSE(cfg.drafter_chain[1].draft_model_path.has_value());
  REQUIRE(cfg.drafter_chain[1].num_speculative_tokens.has_value());
  CHECK(*cfg.drafter_chain[1].num_speculative_tokens == 4);

  // The top-level `method` is EMPTY, not "the first entry". Nothing here
  // silently promotes an entry into the single-method field, because that would
  // make a chain document parse into a config a one-drafter engine would
  // happily run (D7).
  CHECK(cfg.method == "");
}

TEST_CASE("a chain entry carries its own n-gram window") {
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(Chain(
      std::string(kMtp) +
      R"(,{"method":"ngram","num_speculative_tokens":4,)"
      R"("prompt_lookup_min":3,"prompt_lookup_max":5})"));
  REQUIRE(cfg.drafter_chain.size() == 2);
  REQUIRE(cfg.drafter_chain[1].prompt_lookup_min.has_value());
  CHECK(*cfg.drafter_chain[1].prompt_lookup_min == 3);
  REQUIRE(cfg.drafter_chain[1].prompt_lookup_max.has_value());
  CHECK(*cfg.drafter_chain[1].prompt_lookup_max == 5);
}

TEST_CASE("a SINGLE-entry chain is legal") {
  // D8, recorded. A one-entry chain is the degenerate preference list, exactly
  // as `--spec-type mtp` is in llama.cpp, and refusing it would force every
  // generator of these documents to special-case n == 1 into a different
  // spelling. It is not equivalent to a top-level `method`, and this case is
  // what pins that: the entry lands in the chain, and the top-level method
  // stays empty, so the loader's chain refusal still sees it.
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(Chain(kMtp));
  REQUIRE(cfg.use_drafter_chain());
  REQUIRE(cfg.drafter_chain.size() == 1);
  CHECK(cfg.drafter_chain[0].method == "mtp");
  CHECK(cfg.method == "");
}

TEST_CASE("INERT: no vllm_cpp key leaves every existing document unchanged") {
  // The parser half of G1. The engine half — a REAL document driven through the
  // production loader — is `test_drafter_chain_reach.cpp`, because "byte-identical
  // behaviour" is not something a parser can assert about itself.
  SUBCASE("mtp, method only") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(R"({"method":"mtp"})");
    CHECK(cfg.method == "mtp");
    CHECK_FALSE(cfg.use_drafter_chain());
    CHECK(cfg.drafter_chain.empty());
  }
  SUBCASE("dflash with a draft checkpoint") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash",)"
        R"("num_speculative_tokens":16})");
    CHECK(cfg.method == "dflash");
    CHECK(*cfg.draft_model_path == "z-lab/Qwen3.6-27B-DFlash");
    CHECK(*cfg.num_speculative_tokens == 16);
    CHECK(cfg.drafter_chain.empty());
  }
  SUBCASE("dspark with the two engine-wide sampling keys spelled out") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7",)"
        R"("num_speculative_tokens":7,"draft_sample_method":"greedy",)"
        R"("rejection_sample_method":"standard"})");
    CHECK(cfg.method == "dspark");
    CHECK(cfg.drafter_chain.empty());
  }
  SUBCASE("ngram with the full lookup window") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"ngram","num_speculative_tokens":4,)"
        R"("prompt_lookup_min":3,"prompt_lookup_max":5})");
    CHECK(*cfg.prompt_lookup_min == 3);
    CHECK(cfg.drafter_chain.empty());
  }
  SUBCASE("draft_model with a separate checkpoint") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
        R"({"method":"draft_model","model":"Qwen/Qwen3.6-0.6B",)"
        R"("num_speculative_tokens":3})");
    CHECK(cfg.method == "draft_model");
    CHECK(cfg.drafter_chain.empty());
  }
}

TEST_CASE("the two ENGINE-WIDE sampling keys may sit beside a chain") {
  // `draft_sample_method` and `rejection_sample_method` describe the DRAFT
  // SAMPLING RULE and the VERIFY, neither of which is per drafter: the verify
  // is one verify however many speculators propose into it. So unlike the
  // per-speculator keys below they are NOT ambiguous beside a chain, and they
  // keep their existing value gate.
  const SpeculativeConfig cfg = ParseSpeculativeConfigJson(
      R"({"draft_sample_method":"greedy","rejection_sample_method":"standard",)"
      R"("vllm_cpp":{"drafter_chain":[{"method":"mtp"}]}})");
  CHECK(cfg.drafter_chain.size() == 1);

  // And the gate still fires beside a chain — a chain must not become a way to
  // smuggle an unimplemented acceptance rule past #1160's refusal.
  const std::string msg = RefusalMessage(
      R"({"draft_sample_method":"probabilistic",)"
      R"("vllm_cpp":{"drafter_chain":[{"method":"mtp"}]}})");
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "draft_sample_method"));
  CHECK(Mentions(msg, "probabilistic"));
  CHECK(Mentions(msg, "SPEC-ACCEPT-VARIANTS"));
}

TEST_CASE("G5: a malformed extension object is refused by name") {
  SUBCASE("vllm_cpp is not an object") {
    const std::string msg =
        RefusalMessage(R"({"vllm_cpp":["drafter_chain"]})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp"));
    CHECK(Mentions(msg, "must be a JSON object"));
  }
  SUBCASE("an explicit null names the extension and asks for nothing") {
    // Presence is what is judged, exactly as #1160 established for every other
    // key: a null on a name we DO implement must not read as "absent" and fall
    // through to "a string method is required", which sends the user hunting
    // for a key they deliberately spelled.
    const std::string msg = RefusalMessage(R"({"vllm_cpp":null})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp"));
    CHECK(Mentions(msg, "must be a JSON object"));
    CHECK_FALSE(Mentions(msg, "a string \"method\" is required"));
  }
  SUBCASE("an unknown key inside vllm_cpp is refused by name") {
    const std::string msg = RefusalMessage(
        R"({"vllm_cpp":{"drafter_chian":[{"method":"mtp"}]}})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "drafter_chian"));   // the typo, quoted back
    CHECK(Mentions(msg, "drafter_chain"));   // and the correct spelling
  }
  SUBCASE("vllm_cpp with no drafter_chain is refused rather than ignored") {
    // An empty extension object is a document that names this engine's
    // extension surface and configures nothing on it. Accepting it would run a
    // one-drafter engine under a document its author believes says otherwise.
    const std::string msg = RefusalMessage(R"({"vllm_cpp":{}})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "carries no \"drafter_chain\""));
    // And it is NOT the "must be a JSON object" refusal, which an empty object
    // must not take: `{}` IS an object, and saying otherwise would be wrong.
    CHECK_FALSE(Mentions(msg, "must be a JSON object"));
  }
  SUBCASE("drafter_chain is not an array") {
    const std::string msg =
        RefusalMessage(R"({"vllm_cpp":{"drafter_chain":{"method":"mtp"}}})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
    CHECK(Mentions(msg, "must be a JSON array"));
  }
  SUBCASE("an EMPTY chain is refused") {
    // D9. A chain with no entries asks for a preference order over nothing. The
    // only two readings are "no speculation" and "whatever the engine would
    // have done" — both silently different from what the document says, which
    // is the class of defect #1160 exists to end.
    const std::string msg = RefusalMessage(R"({"vllm_cpp":{"drafter_chain":[]}})");
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
    CHECK(Mentions(msg, "empty"));
  }
  SUBCASE("an entry is not an object") {
    const std::string msg = RefusalMessage(Chain(R"("mtp")"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
    CHECK(Mentions(msg, "must be a JSON object"));
  }
  SUBCASE("an entry has no method") {
    const std::string msg =
        RefusalMessage(Chain(std::string(kMtp) + R"(,{"num_speculative_tokens":4})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[1]"));
    // The PHRASE, not the bare word: "method" appears in almost every refusal
    // this parser emits, so it distinguishes nothing on its own.
    CHECK(Mentions(msg, "requires a string \"method\""));
  }
  SUBCASE("an entry's method is not a string") {
    const std::string msg = RefusalMessage(Chain(R"({"method":7})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
    CHECK(Mentions(msg, "requires a string \"method\""));
  }
}

TEST_CASE("G5: an entry naming a method this engine does not implement") {
  SUBCASE("a method vLLM declares and this engine does not") {
    // `eagle3` is a real vLLM `SpeculativeMethod`. It must be refused BY NAME
    // and BEFORE any weight I/O, not skipped as though the chain simply had one
    // fewer entry — a silently shortened chain is a different feature running
    // under the user's document.
    const std::string msg =
        RefusalMessage(Chain(std::string(kMtp) + R"(,{"method":"eagle3"})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[1]"));
    CHECK(Mentions(msg, "eagle3"));
    // The accepted set, so the message closes the search.
    CHECK(Mentions(msg, "\"mtp\""));
    CHECK(Mentions(msg, "\"dflash\""));
    CHECK(Mentions(msg, "\"dspark\""));
    CHECK(Mentions(msg, "\"ngram\""));
  }
  SUBCASE("a name nobody declares") {
    const std::string msg = RefusalMessage(Chain(R"({"method":"dflash2"})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "dflash2"));
  }
  SUBCASE("draft_model is accepted at top level and NOT as a chain entry") {
    // D10. `draft_model` IS implemented as a top-level method, so its refusal
    // here must not read as "this engine has no such method". The row scopes
    // four chain entry methods (`.agents/specs/drafter-chain.md` `## Scope`),
    // and the fifth is owed rather than quietly admitted.
    const std::string msg = RefusalMessage(Chain(
        R"({"method":"draft_model","model":"Qwen/Qwen3.6-0.6B",)"
        R"("num_speculative_tokens":3})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "draft_model"));
    CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
  }
}

TEST_CASE("G5: an entry key is admitted by name, in the same THREE classes as #1160") {
  SUBCASE("a vLLM SpeculativeConfig field this engine does not implement") {
    const std::string msg =
        RefusalMessage(Chain(R"({"method":"mtp","quantization":"fp8"})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
    CHECK(Mentions(msg, "quantization"));
    // Not a typo, and the message must not send the user hunting for one.
    CHECK_FALSE(Mentions(msg, "unknown key"));
  }
  SUBCASE("a name nobody declares") {
    const std::string msg =
        RefusalMessage(Chain(R"({"method":"mtp","num_speculatve_tokens":4})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "num_speculatve_tokens"));
    CHECK(Mentions(msg, "unknown key"));
    CHECK(Mentions(msg, "num_speculative_tokens"));  // the right spelling
  }
  // #1598 — the class this admission loop originally LOST.
  //
  // `draft_sample_method` and `rejection_sample_method` are #1160's CLASS 2: a
  // vLLM `SpeculativeConfig` field this engine implements at exactly one point
  // of its value space (`speculative.py:77,283` and `:78,216` @ 555967922).
  // Both are ENGINE-WIDE — one describes how a draft is sampled and the other
  // describes the verify, and the verify is one verify however many speculators
  // propose into it — so the top level HONOURS them beside a chain, and the case
  // above asserts that it does.
  //
  // Inside an ENTRY they were in neither the honoured set nor the
  // upstream-unimplemented set, so they fell through to the typo branch and got
  // a message BYTE-FOR-BYTE of the shape a misspelling gets. That is #1160's own
  // failure inverted: the split exists precisely so a key vLLM declares does not
  // read as "unknown", and telling a user that a key this engine honours twelve
  // characters up the same document is unknown sends them hunting for a spelling
  // mistake that is not there.
  //
  // The correct refusal is neither "honoured" nor "not implemented at this pin",
  // because it IS implemented: it is not PER-DRAFTER. So the message names the
  // key, says the engine honours it, and says where to spell it.
  SUBCASE("an ENGINE-WIDE sampling key belongs at the top level, not in an entry") {
    for (const char* key : {"draft_sample_method", "rejection_sample_method"}) {
      const std::string value =
          std::string(key) == "draft_sample_method" ? "greedy" : "standard";
      const std::string msg = RefusalMessage(Chain(
          R"({"method":"mtp",")" + std::string(key) + R"(":")" + value + R"("})"));
      REQUIRE(msg != "");
      CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
      CHECK(Mentions(msg, key));
      // NOT a typo. This is the assertion the defect failed.
      CHECK_FALSE(Mentions(msg, "unknown key"));
      // NOT "unimplemented" either — the engine honours it, one line up.
      CHECK_FALSE(Mentions(msg, "does not implement"));
      // It has to say WHERE, or the user has nowhere to go.
      CHECK(Mentions(msg, "TOP LEVEL"));
      CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
    }
  }
  SUBCASE("the extension key does not nest inside an entry") {
    const std::string msg =
        RefusalMessage(Chain(R"({"method":"mtp","vllm_cpp":{}})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
  }
  SUBCASE("a bad value on an admitted entry key") {
    CHECK(Mentions(RefusalMessage(Chain(
                       R"({"method":"mtp","num_speculative_tokens":0})")),
                   "num_speculative_tokens"));
    CHECK(Mentions(RefusalMessage(Chain(
                       R"({"method":"ngram","num_speculative_tokens":4,)"
                       R"("prompt_lookup_min":0})")),
                   "prompt_lookup_min"));
  }
}

TEST_CASE("G5: an entry missing the key its method requires is refused by name") {
  // The SAME required-key rules the top-level document has, so an entry is not
  // a weaker place to spell a config than the field it replaces.
  SUBCASE("dflash without a draft checkpoint") {
    const std::string msg = RefusalMessage(Chain(R"({"method":"dflash"})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
    CHECK(Mentions(msg, "method \"dflash\""));
    CHECK(Mentions(msg, "requires a \"model\" key"));
  }
  SUBCASE("dspark without a draft checkpoint") {
    const std::string msg = RefusalMessage(Chain(
        std::string(kMtp) + R"(,{"method":"dspark","num_speculative_tokens":7})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[1]"));
    CHECK(Mentions(msg, "method \"dspark\""));
    CHECK(Mentions(msg, "requires a \"model\" key"));
  }
  SUBCASE("ngram without a k") {
    const std::string msg = RefusalMessage(Chain(R"({"method":"ngram"})"));
    REQUIRE(msg != "");
    CHECK(Mentions(msg, "vllm_cpp.drafter_chain[0]"));
    CHECK(Mentions(msg, "method \"ngram\""));
    CHECK(Mentions(msg, "requires \"num_speculative_tokens\""));
  }
  SUBCASE("mtp requires neither, and still parses") {
    const SpeculativeConfig cfg = ParseSpeculativeConfigJson(Chain(kMtp));
    CHECK(cfg.drafter_chain.size() == 1);
  }
}

TEST_CASE("G5: a method named twice is refused by name, with both positions") {
  // D11. Resolution (W3) records a per-sequence WINNER and W2 keys the
  // per-drafter counters on the method name, so two entries with the same
  // method make the attribution the gates depend on ambiguous. Refusing now is
  // reversible; shipping ambiguous counters is not.
  const std::string msg = RefusalMessage(
      Chain(std::string(kDflash) + "," + kMtp + "," +
            R"({"method":"dflash","model":"z-lab/other"})"));
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "dflash"));
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
  // BOTH positions, as a PHRASE. Asserting the bare characters "0" and "2"
  // would be satisfied by any digit anywhere in the message, and this message
  // quotes upstream line numbers elsewhere.
  CHECK(Mentions(msg, "entries 0 and 2"));
  CHECK(Mentions(msg, "SPEC-DRAFTER-CHAIN"));
}

TEST_CASE("G5: a top-level method cannot coexist with a chain") {
  // D7. `method` names ONE speculator and the chain names an ordered list.
  // There is no rule here that makes a top-level `method` the head of the
  // chain, its tail, or a default, and every such rule would silently
  // reinterpret a document. So the two are mutually exclusive, and the refusal
  // says which key to move.
  const std::string msg = RefusalMessage(
      R"({"method":"mtp","vllm_cpp":{"drafter_chain":[{"method":"ngram",)"
      R"("num_speculative_tokens":4}]}})");
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "\"method\" and \"vllm_cpp.drafter_chain\" are"));
  CHECK(Mentions(msg, "mutually exclusive"));
  // The value the user typed, so the message is about their document.
  CHECK(Mentions(msg, "got method \"mtp\""));
}

TEST_CASE("G5: a top-level PER-SPECULATOR key cannot coexist with a chain") {
  // The same argument as D7, one key at a time. `model` beside a two-entry
  // chain names a checkpoint with no entry to belong to; silently applying it
  // to the first, to all, or to none are three different engines.
  for (const std::string key_and_value :
       {R"("model":"z-lab/Qwen3.6-27B-DFlash")",
        R"("num_speculative_tokens":16)", R"("prompt_lookup_min":3)",
        R"("prompt_lookup_max":5)"}) {
    const std::string msg = RefusalMessage(
        "{" + key_and_value +
        R"(,"vllm_cpp":{"drafter_chain":[{"method":"mtp"},)"
        R"({"method":"ngram","num_speculative_tokens":4}]}})");
    REQUIRE_MESSAGE(msg != "", key_and_value);
    // The OFFENDING key, quoted in place. A bare word would pass on a message
    // that named a different one, because several are listed in the tail.
    const std::string key = key_and_value.substr(1, key_and_value.find('"', 1) - 1);
    CHECK_MESSAGE(Mentions(msg, "\"" + key + "\""), key_and_value);
    CHECK_MESSAGE(Mentions(msg, "vllm_cpp.drafter_chain"), key_and_value);
  }
}

TEST_CASE("REGRESSION: the landed refusals are unchanged beside the new field") {
  // The chain checks must not displace an existing error. `method` still
  // selects everything else on a chain-free document, so its refusal is still
  // the first one a user sees.
  CHECK(Mentions(RefusalMessage(R"({"method":"eagle3"})"), "eagle3"));
  CHECK(Mentions(RefusalMessage(R"({"num_speculative_tokens":4})"), "method"));
  CHECK(Mentions(RefusalMessage(R"({"method":"dflash"})"), "model"));
  CHECK(Mentions(RefusalMessage(R"({"method":"ngram"})"), "num_speculative_tokens"));
  CHECK(Mentions(RefusalMessage("not json"), "invalid JSON"));
  CHECK(Mentions(RefusalMessage("[1,2]"), "expected a JSON object"));
  // And an unknown TOP-LEVEL key is still refused with the accepted list, which
  // now has to name the extension too or the list stops closing the search.
  const std::string msg = RefusalMessage(R"({"method":"mtp","not_a_vllm_key":1})");
  REQUIRE(msg != "");
  CHECK(Mentions(msg, "not_a_vllm_key"));
  CHECK(Mentions(msg, "unknown key"));
  CHECK(Mentions(msg, "vllm_cpp.drafter_chain"));
}
