// SPEC-DFLASH2 W1 (#1314) — REACHABILITY of the DFlash2 refusal.
//
// WHAT THIS ROW SHIPS. Before W1 nothing in this tree routed the
// `DFlash2DraftModel` architecture string. Upstream adds it beside
// `DFlashDraftModel` in the model registry and selects a DIFFERENT speculator on
// it (vllm-project/vllm#52816 @ `66e5414c6d75a8529473d977f7458c140bbab8a0`,
// which superseded `19c9351904df4c63042671bc67a866ca48dc7d6f` on 2026-08-19,
// `vllm/model_executor/models/registry.py:628` and
// `vllm/v1/worker/gpu/spec_decode/__init__.py:12-17`). Here the draft path was
// selected by the CLI method string alone, so a DFlash2 checkpoint pointed at
// `--speculative-config` loaded through the DFlash1 loader: its tensor set is
// DFlash1's plus the conv and selector tensors, so nothing is missing, nothing
// throws, and the engine drafts with the grouped convolution and the candidate
// selector simply ABSENT. That draft proposes worse tokens, the verify is
// lossless, and the output is still the target's — so a token gate sees nothing
// and only acceptance falls. AGENTS.md: an unimplemented arm refuses with a
// message that names the missing part. The refusal is what W1 ships.
//
// TWO PRODUCTION ENTRY POINTS, and both are entered below.
//
// `LoadedEngine::ResolveSpecConfig` is the resolution the `LoadedEngine`
// constructor runs on `EngineParams` (src/vllm/entrypoints/model_loader.cpp,
// `resolved_spec_config_(...)`). It is public for exactly this reason, argued at
// include/vllm/entrypoints/model_loader.h beside the declaration and landed by
// SPEC-DSPARK-BLOCK-SIZE-GUARD (#1225): it is one of the class's pure
// config-resolution statics, and a test that entered at
// `SpeculativeConfig::IsDflash2Draft` instead would prove the predicate works
// rather than that any draft reaches it — the distinction
// .agents/reachability.md draws.
//
// `LoadedEngine::FromModelDir` is the one a server or `include/vllm.h` caller
// actually arrives through, and it needs its own guard: it loads the dflash
// draft BEFORE it constructs the engine, from the CLI config rather than through
// `ResolveSpecConfig`, so a refusal only in the constructor would arrive after
// the draft checkpoint had been read. The two cases at the end enter there
// against a nonexistent target directory, which is what proves the refusal lands
// ahead of every path, config, tokenizer and weight operation.
//
// THE REACHABILITY MUTATIONS delete each of those two call sites in turn and
// require this suite RED.
//
// RED before this row: every refusal case below returns a resolved `dflash`
// config, because nothing reads the draft's architectures at all.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../gguf_builder.h"
#include "support/process_id.h"

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace fs = std::filesystem;
using vllm::SpeculativeConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// A scratch draft checkpoint directory holding just a config.json, which is all
// the classification reads. `ResolveDflashDraftDir` accepts any directory that
// contains one, so this is the real production read. No draft weight is ever
// opened on this path — that is the point: the refusal has to land BEFORE any
// weight I/O.
class ScratchDraft {
 public:
  explicit ScratchDraft(const std::string& config_json) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dflash2_route_" + std::to_string(vllm_test::ProcessId()) + "_" +
            std::to_string(counter++));
    fs::create_directories(dir_);
    std::ofstream(dir_ / "config.json") << config_json;
  }
  ~ScratchDraft() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchDraft(const ScratchDraft&) = delete;
  ScratchDraft& operator=(const ScratchDraft&) = delete;
  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// An EMPTY scratch directory: no config.json at all. Stands for the two shapes
// that reach the classification with nothing to read — a `.gguf` draft file and
// an HF repo id that is not in the local cache.
class EmptyDir {
 public:
  EmptyDir() {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_dflash2_empty_" + std::to_string(vllm_test::ProcessId()) + "_" +
            std::to_string(counter++));
    fs::create_directories(dir_);
  }
  ~EmptyDir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  EmptyDir(const EmptyDir&) = delete;
  EmptyDir& operator=(const EmptyDir&) = delete;
  std::string path() const { return dir_.string(); }

 private:
  fs::path dir_;
};

// `z-lab/Qwen3.8-27B-DFlash2`, reduced to the keys the classification reads.
// `is_causal false` beside five `sliding_attention` layers is the OTHER half of
// this wave and is gated in tests/vllm/v1/spec_decode/test_dflash_causality.cpp.
constexpr const char* kDflash2DraftConfig =
    R"({"architectures":["DFlash2DraftModel"],"model_type":"qwen3",)"
    R"("num_hidden_layers":5,"block_size":8,"is_causal":false,)"
    R"("conv_kernel_size":2,"conv_group_size":16,"selector_rank":256,)"
    R"("selector_top_k":16})";

// `z-lab/Qwen3.6-27B-DFlash`, the lane that ships today. The regression guard:
// the classification must not refuse the draft that already loads.
constexpr const char* kDflash1DraftConfig =
    R"({"architectures":["DFlashDraftModel"],"model_type":"qwen3",)"
    R"("num_hidden_layers":5,"block_size":16})";

// A draft that declares no architecture at all. Classifying on the ABSENCE of
// evidence would refuse a lane that loads today, so it must not be refused.
constexpr const char* kNoArchitecturesDraftConfig =
    R"({"model_type":"qwen3","num_hidden_layers":5,"block_size":16})";

// A `dflash`-arch GGUF drafter, reduced to the keys the classification reads.
//
// THE GGUF AXIS IS WHY THIS IS NOT THE SAME QUESTION as the config.json one. A
// GGUF carries no `architectures` array at all, and
// `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `57ab3265056d4024870b0621cfc2c127537020ed`
// declares `general.architecture = "dflash"` -- byte-identical to a DFlash1
// drafter. `src/vllm/model_executor/models/qwen3_dflash_gguf.cpp` discriminates
// on exactly that string, so the architecture cannot tell the two apart and the
// DFlash2 file loads as DFlash1 with no conv, no selector and no error. The
// discriminator is therefore the DFlash2-ONLY metadata a DFlash1 file does not
// carry, verified against both published artifacts on 2026-08-19: the DFlash2
// GGUF has `conv_kernel_size`, `conv_group_size`, `selector_rank` and
// `selector_top_k`; `muse-glimmer-30b-gguf/dflash-kquant.gguf` has none of them.
std::string DflashGgufBytes(bool dflash2) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "dflash"));
  b.AddKv(gguf_test::U32Kv("dflash.block_count", 5));
  b.AddKv(gguf_test::U32Kv("dflash.embedding_length", 5120));
  b.AddKv(gguf_test::U32Kv("dflash.block_size", dflash2 ? 8 : 16));
  b.AddKv(gguf_test::I32ArrayKv("dflash.target_layers", {6, 20, 34, 48, 62}));
  if (dflash2) {
    b.AddKv(gguf_test::BoolKv("dflash.attention.causal", false));
    b.AddKv(gguf_test::U32Kv("dflash.conv_kernel_size", 2));
    b.AddKv(gguf_test::U32Kv("dflash.conv_group_size", 16));
    b.AddKv(gguf_test::U32Kv("dflash.selector_rank", 256));
    b.AddKv(gguf_test::U32Kv("dflash.selector_top_k", 16));
  }
  return b.Build();
}

EngineParams DflashParams(const std::string& draft_path, int k) {
  EngineParams params;
  SpeculativeConfig cli;
  cli.method = "dflash";
  cli.draft_model_path = draft_path;
  cli.num_speculative_tokens = k;
  params.speculative_config = cli;
  return params;
}

// The refusal text the loader produced, or "" when resolution got PAST the
// classification. Returning the empty string rather than asserting keeps the RED
// legible: before this row the refusal is ABSENT, not wrong.
std::string RefusalForDraft(const std::string& draft_path) {
  try {
    LoadedEngine::ResolveSpecConfig(DflashParams(draft_path, 8), vllm::HfConfig{});
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("W2: a safetensors DFlash2 draft is ADMITTED, and as of W4 it DRAFTS") {
  // W1 REFUSED this draft here, before any weight was read, because BOTH
  // mechanisms were missing. SPEC-DFLASH2 W2 implements one of them -- the
  // grouped dynamic depthwise convolution -- and a startup refusal would leave
  // every line of it unreachable from any production entry point, which is what
  // AGENTS.md `## Nothing lands dead` forbids. So the draft is admitted here and
  // refused one step later, AFTER the conv has run. W3 moved that boundary to
  // the PATH WALK, and W4 lands the walk -- so on this arm there is no boundary
  // left, and the guard now points the other way: the DFlash1 per-slot argmax is
  // refused for a DFlash2 block (`RefuseDflash1ArgmaxOnDflash2Block`, gated in
  // tests/vllm/v1/spec_decode/test_dflash2_argmax_guard.cpp, and reached from
  // the runner in tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp).
  //
  // The STARTUP notice stays, and what it says moved with the boundary each
  // time. It is asserted below, because a notice that goes stale is exactly what
  // W3's fresh review found in the loader's other copy.
  const ScratchDraft draft(kDflash2DraftConfig);
  CHECK(RefusalForDraft(draft.path()).empty());
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(DflashParams(draft.path(), 8), vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->method == "dflash");
  CHECK(cfg->ResolvedNumSpeculativeTokens() == 8);
}

TEST_CASE("W4: admitting the DFlash2 draft STATES what runs and what is owed") {
  // Through W3 this notice existed so that a user admitted silently and refused
  // at the first generated token had been told something. W4 removes the
  // refusal, and the notice stays for the two things a user cannot read off the
  // checkpoint: that the port is BEYOND the parity pin, and that no throughput
  // number has been taken for it. Every wave that moved the boundary had to move
  // this text, and W3's review found the loader's OTHER copy still naming the
  // wave that had just shipped -- so it is asserted, not assumed.
  const ScratchDraft draft(kDflash2DraftConfig);
  std::ostringstream captured;
  std::streambuf* const previous = std::cerr.rdbuf(captured.rdbuf());
  try {
    (void)LoadedEngine::ResolveSpecConfig(DflashParams(draft.path(), 8), vllm::HfConfig{});
  } catch (...) {
    std::cerr.rdbuf(previous);
    throw;
  }
  std::cerr.rdbuf(previous);
  const std::string notice = captured.str();
  INFO("notice: ", notice);
  CHECK(notice.find("DFlash2DraftModel") != std::string::npos);
  CHECK(notice.find("grouped dynamic") != std::string::npos);
  // W4: all three mechanisms run, and the notice says so rather than naming a
  // wave that has landed.
  CHECK(notice.find("PATH WALK are all implemented") != std::string::npos);
  CHECK(notice.find("this draft DRAFTS") != std::string::npos);
  CHECK(notice.find("PATH WALK is not implemented") == std::string::npos);
  // What IS still owed, named: the GGUF drafter arm and the absent number.
  CHECK(notice.find("wave W5") != std::string::npos);
  CHECK(notice.find("no throughput number") != std::string::npos);
  CHECK(notice.find("52816") != std::string::npos);
  CHECK(notice.find("SPEC-DFLASH2") != std::string::npos);
  CHECK(notice.find("#1314") != std::string::npos);
}

TEST_CASE("the loader still admits a DFlashDraftModel draft") {
  const ScratchDraft draft(kDflash1DraftConfig);
  CHECK(RefusalForDraft(draft.path()).empty());
  const std::optional<SpeculativeConfig> cfg =
      LoadedEngine::ResolveSpecConfig(DflashParams(draft.path(), 8), vllm::HfConfig{});
  REQUIRE(cfg.has_value());
  CHECK(cfg->method == "dflash");
  CHECK(cfg->ResolvedNumSpeculativeTokens() == 8);
}

TEST_CASE("the loader does not classify a draft that declares no architecture") {
  const ScratchDraft draft(kNoArchitecturesDraftConfig);
  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("a draft with no config.json to read resolves exactly as before") {
  // A `.gguf` draft and an uncached HF repo id both land here. Both already have
  // their own precise error further down the load, and replacing it with a
  // classification failure would be a regression.
  const EmptyDir draft;
  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("W2: the early FromModelDir guard no longer refuses a safetensors DFlash2 draft") {
  // The mirror of the case above at the OTHER production call site. `FromModelDir`
  // loads the dflash draft before it builds the `LoadedEngine`, so W1 guarded it
  // separately; W2 admits the safetensors arm at both sites, and the failure a
  // user now sees for a nonexistent target is the target error, not a DFlash2
  // refusal. The GGUF arm below still refuses at this same site.
  const ScratchDraft draft(kDflash2DraftConfig);
  EngineParams params = DflashParams(draft.path(), 8);
  std::ostringstream sink;
  std::streambuf* const previous = std::cerr.rdbuf(sink.rdbuf());
  std::string what;
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dflash2/target", params);
  } catch (const std::exception& e) {
    what = e.what();
  }
  std::cerr.rdbuf(previous);
  INFO("what: ", what);
  CHECK(what.find("model path is not a directory") != std::string::npos);
  CHECK(what.find("not implemented") == std::string::npos);
}

TEST_CASE("the loader refuses a DFlash2 GGUF draft BEFORE it touches the model directory") {
  // The SECOND production call site, and the one the constructor's resolution
  // cannot cover. `FromModelDir` loads a dflash draft (`maybe_load_dflash`)
  // BEFORE it builds the `LoadedEngine`, and that site resolves the draft from
  // the CLI config directly rather than through `ResolveSpecConfig` — so without
  // an early guard the DFlash2 checkpoint would already have been read through
  // the DFlash1 loader by the time the constructor got an opinion. This is the
  // same ordering assertion SPEC-DSPARK-BLOCK-SIZE-GUARD makes at
  // tests/vllm/entrypoints/test_dspark_block_size_guard.cpp: a nonexistent target
  // directory, so the refusal is proven to land ahead of every path, config,
  // tokenizer and weight operation. RED without the guard: the failure is
  // "model path is not a directory" and the draft classification never runs.
  const gguf_test::TempFile draft(DflashGgufBytes(/*dflash2=*/true));
  EngineParams params = DflashParams(draft.path(), 8);
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dflash2/target", params);
    FAIL("expected a refusal for a DFlash2 GGUF draft");
  } catch (const std::exception& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find("GGUF drafter ARM") != std::string::npos);
    CHECK(what.find("model path is not a directory") == std::string::npos);
  }
}

TEST_CASE("the early guard leaves a DFlashDraftModel target error unchanged") {
  // The instrument's own precondition. A guard that refused every dflash draft
  // would satisfy the case above while breaking the lane that ships, and the
  // assertion there could not tell the two apart.
  const ScratchDraft draft(kDflash1DraftConfig);
  EngineParams params = DflashParams(draft.path(), 8);
  try {
    (void)LoadedEngine::FromModelDir("/nonexistent/vllm-cpp/dflash2/target", params);
    FAIL("expected the missing target directory to be the failure");
  } catch (const std::exception& e) {
    const std::string what = e.what();
    INFO("what: ", what);
    CHECK(what.find("model path is not a directory") != std::string::npos);
    CHECK(what.find("DFlash2") == std::string::npos);
  }
}

TEST_CASE("IsDflash2Draft answers on the architecture upstream selects on") {
  // The predicate itself, kept beside the reachability cases because it
  // localizes a failure. It is not the proof; the cases above are.
  CHECK(SpeculativeConfig::IsDflash2Draft({"DFlash2DraftModel"}));
  CHECK(SpeculativeConfig::IsDflash2Draft({"DFlashDraftModel", "DFlash2DraftModel"}));
  CHECK_FALSE(SpeculativeConfig::IsDflash2Draft({"DFlashDraftModel"}));
  CHECK_FALSE(SpeculativeConfig::IsDflash2Draft({}));
}

TEST_CASE("the loader refuses a DFlash2 GGUF drafter, which declares no architecture") {
  // The published GGUF drafter is the case the config.json-keyed arm CANNOT
  // see: no `architectures` key exists in a GGUF, and its
  // `general.architecture` is the same "dflash" a DFlash1 drafter writes. Only
  // the DFlash2-only metadata separates them.
  const gguf_test::TempFile draft(DflashGgufBytes(/*dflash2=*/true));
  const std::string message = RefusalForDraft(draft.path());
  REQUIRE_FALSE(message.empty());  // RED: the architecture arm cannot answer here.
  INFO("what: ", message);
  CHECK(message.find("grouped dynamic") != std::string::npos);
  CHECK(message.find("candidate selector") != std::string::npos);
  CHECK(message.find("path walk") != std::string::npos);
  CHECK(message.find("not implemented") != std::string::npos);
  CHECK(message.find("SPEC-DFLASH2") != std::string::npos);
  CHECK(message.find("#1314") != std::string::npos);
  // The KEYS it matched on, because the GGUF arm has no architecture string to
  // quote and a reader otherwise cannot tell what identified the file. This is
  // also what separates the two refusal arms in a message assertion.
  CHECK(message.find("dflash.selector_rank") != std::string::npos);
}

TEST_CASE("the loader still admits a DFlash1 GGUF drafter") {
  // The regression guard on the axis that ships today. A DFlash1 GGUF carries
  // none of the DFlash2 metadata, so it must pass through untouched --
  // `muse-glimmer-30b-gguf/dflash-kquant.gguf` is exactly this shape.
  const gguf_test::TempFile draft(DflashGgufBytes(/*dflash2=*/false));
  CHECK(RefusalForDraft(draft.path()).empty());
}

TEST_CASE("REAL published GGUF drafters classify as they must") {
  // ASSET-GATED, and it reports what it did rather than skipping in silence: a
  // suite whose cases all skip prints SUCCESS. The synthetic cases above are the
  // gate; this one checks the synthetic shape against the artifacts it was
  // written from, which is the failure mode a hand-built fixture cannot see.
  //
  //   VLLM_DFLASH2_GGUF_MODEL -> z-lab/Qwen3.8-27B-DFlash2-GGUF
  //                              @ 57ab3265056d4024870b0621cfc2c127537020ed
  //   VLLM_DFLASH_GGUF_MODEL  -> any DFlash1 drafter GGUF
  const char* dflash2 = std::getenv("VLLM_DFLASH2_GGUF_MODEL");
  const char* dflash1 = std::getenv("VLLM_DFLASH_GGUF_MODEL");
  if (dflash2 == nullptr && dflash1 == nullptr) {
    MESSAGE("asset-gated: neither VLLM_DFLASH2_GGUF_MODEL nor "
            "VLLM_DFLASH_GGUF_MODEL is set; 0 real-artifact assertions ran");
    return;
  }
  if (dflash2 != nullptr) {
    const std::string message = RefusalForDraft(dflash2);
    INFO("what: ", message);
    REQUIRE_FALSE(message.empty());
    CHECK(message.find("dflash.selector_rank") != std::string::npos);
  }
  if (dflash1 != nullptr) {
    CHECK(RefusalForDraft(dflash1).empty());
  }
}
