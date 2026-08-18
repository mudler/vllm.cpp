// `ENG-RESIDENCY-CONFIG` (issue #1110) — REACHABILITY of the `vllm_cpp` residency
// extension, not its parsing. `test_weight_residency_config.cpp` proves the struct
// parses; this file proves something a user can arrive through actually installs
// it.
//
// WHY THIS FILE EXISTS SEPARATELY. AGENTS.md `## Nothing lands dead`, and
// .agents/reachability.md's opening case: tensor parallelism landed with a green
// focused gate over a handle no production caller ever passed. A unit test that
// builds a `WeightResidencyConfig` by hand and installs it proves the struct
// works; it says nothing about whether `--offload-config` reaches the loader. So
// every case below enters through a production entry point and never constructs
// the config itself:
//
//   * `LoadedEngine::FromModelDir` — the loader, taking EngineParams as the
//     server and the C ABI both fill it in;
//   * `vllm_engine_load` — the public C ABI (include/vllm.h), taking the SAME
//     `offload_config` JSON string the server flag carries.
//
// THE MODEL DIRECTORY IS DELIBERATELY NONEXISTENT. The install sits in the first
// statement block of FromModelDir, ahead of every path, config and weight
// operation — which is where it has to be, because each knob it feeds is read
// through a function-local static that latches on first use. A load that fails on
// a missing checkpoint therefore still runs the install, and that is exactly what
// makes the install observable without a 370 GiB checkpoint.
//
// THE REACHABILITY MUTATION for this row deletes the install call site in
// `LoadedEngine::FromModelDir` and requires this suite RED.
#include <doctest/doctest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vllm.h"
#include "vllm/config/weight_residency.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

namespace {

constexpr const char* kMissingModel = "/nonexistent/vllm-cpp/residency-reach";

// The reproduction shape from issue #1110: borrow the tower out of the mapping,
// do NOT prefault it (a model larger than memory cannot), and stream the routed
// experts through a bounded slot cache.
constexpr const char* kResidencyJson =
    R"({"vllm_cpp":{"mmap":{"enabled":true,"prefault":false},)"
    R"("expert_stream":{"enabled":true,"slots":8000}}})";

struct ResidencyFixture {
  ResidencyFixture() { vllm::ResetWeightResidencyConfigForTesting(); }
  ~ResidencyFixture() { vllm::ResetWeightResidencyConfigForTesting(); }
};

}  // namespace

TEST_CASE("residency reach: FromModelDir installs the config before it loads anything") {
  ResidencyFixture fx;
  REQUIRE(vllm::ActiveWeightResidencyConfig().empty());

  // EngineParams is what the server flag and the C ABI both produce. The test
  // does NOT call SetWeightResidencyConfig; the loader must.
  vllm::entrypoints::EngineParams params;
  params.weight_residency =
      vllm::parse_weight_residency_extension_json(kResidencyJson);

  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel,
                                                             params));

  // The load failed on the missing checkpoint, AFTER the install. If the install
  // were anywhere below the path resolution this would still be empty, and the
  // knobs — every one of which is read during weight load — would resolve from
  // the environment as though no config had been given.
  const vllm::WeightResidencyConfig& active = vllm::ActiveWeightResidencyConfig();
  REQUIRE_FALSE(active.empty());
  REQUIRE(active.mmap.has_value());
  CHECK(*active.mmap == true);
  REQUIRE(active.prefault.has_value());
  CHECK(*active.prefault == false);
  REQUIRE(active.expert_stream.has_value());
  CHECK(*active.expert_stream == true);
  REQUIRE(active.expert_stream_slots.has_value());
  CHECK(*active.expert_stream_slots == 8000);
}

TEST_CASE("residency reach: a load with no extension leaves the global inert") {
  ResidencyFixture fx;

  // The default path, and the inertness half of the guarantee. An engine whose
  // --offload-config carries only the MIRRORED keys — or nothing at all — must
  // install nothing, so every knob resolves exactly as it did before this row.
  for (const char* doc : {"", R"({"uva":{"cpu_offload_gb":4}})"}) {
    CAPTURE(doc);
    vllm::entrypoints::EngineParams params;
    params.weight_residency = vllm::parse_weight_residency_extension_json(doc);
    CHECK_THROWS(
        vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, params));
    CHECK(vllm::ActiveWeightResidencyConfig().empty());
  }

  // And an engine that sets no residency field at all — the optional unset, which
  // is what every existing caller produces.
  vllm::entrypoints::EngineParams bare;
  CHECK_FALSE(bare.weight_residency.has_value());
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, bare));
  CHECK(vllm::ActiveWeightResidencyConfig().empty());
}

TEST_CASE("residency reach: the C ABI's offload_config string carries the extension") {
  ResidencyFixture fx;
  REQUIRE(vllm::ActiveWeightResidencyConfig().empty());

  // include/vllm.h is the public surface, and `offload_config` is ONE string: the
  // mirrored keys and the vllm_cpp extension travel together, exactly as they do
  // on the server's --offload-config. So this case is also the proof that the
  // extension needed no new ABI field.
  vllm_model_params p = vllm_model_params_default();
  p.model_path = kMissingModel;
  p.offload_config = kResidencyJson;
  vllm_engine* eng = nullptr;
  // MODEL_LOAD, never INVALID_ARGUMENT: the document was ACCEPTED and the load
  // then failed on the missing checkpoint.
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);

  const vllm::WeightResidencyConfig& active = vllm::ActiveWeightResidencyConfig();
  REQUIRE_FALSE(active.empty());
  REQUIRE(active.expert_stream_slots.has_value());
  CHECK(*active.expert_stream_slots == 8000);
  REQUIRE(active.prefault.has_value());
  CHECK(*active.prefault == false);
}

TEST_CASE("residency reach: the GGUF load POLICY consults the installed config") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_CPU_REF");

  // Installing is not the same as being READ, and this is the one knob whose
  // reader is reachable from a test: `GgufLoadPolicy::FromEnv()` is what every
  // GGUF loader in the tree calls, and `mmap_residency` is the field that decides
  // whether a 370 GiB tower is borrowed out of the mapping or copied into RAM it
  // does not fit in. A config the policy ignored would pass every other case in
  // this file.
  vllm::WeightResidencyConfig off;
  off.mmap = false;
  vllm::SetWeightResidencyConfig(off);
  CHECK_FALSE(vllm::GgufLoadPolicy::FromEnv().mmap_residency);

  vllm::ResetWeightResidencyConfigForTesting();
  vllm::WeightResidencyConfig on;
  on.mmap = true;
  vllm::SetWeightResidencyConfig(on);
  CHECK(vllm::GgufLoadPolicy::FromEnv().mmap_residency);

  // The environment still wins at the site, not merely in the resolver.
  ::setenv("VT_GGUF_MMAP", "0", 1);
  CHECK_FALSE(vllm::GgufLoadPolicy::FromEnv().mmap_residency);
  ::unsetenv("VT_GGUF_MMAP");

  // ...and `VT_CPU_REF` still beats BOTH. The oracle switch is not a residency
  // preference: it exists so a reference load reproduces byte for byte, and a
  // config that could turn borrowing back on under it would silently void an
  // oracle comparison.
  ::setenv("VT_CPU_REF", "1", 1);
  CHECK_FALSE(vllm::GgufLoadPolicy::FromEnv().mmap_residency);
  ::unsetenv("VT_CPU_REF");
}

TEST_CASE("residency reach: a SECOND engine can still install, because a first load latches nothing") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_CPU_REF");

  // THE TWO-MODEL PROCESS, driven through the loader. Engine A carries no residency
  // document and stops on the missing checkpoint BEFORE any weight I/O, so the case
  // calls `GgufLoadPolicy::FromEnv()` itself two lines below to stand in for the read
  // a real load would perform. That is the point of the case rather than a weakness
  // in it: the resolver being read is what the first implementation latched on, so
  // reading it here is precisely the state that made engine B throw. What the case
  // does NOT prove is that a completed weight load reads it — no checkpoint small
  // enough to load lives in this suite.
  //
  // The first pass refused engine B. One process-wide flag was marked by every
  // resolver, so the ordinary `GgufLoadPolicy::FromEnv()` of load A made load B
  // throw — a hard failure on a legal second load, for a reason that was untrue of
  // the knob that had been read: `FromEnv()` runs per load and caches nothing.
  //
  // Reading a knob is not taking a decision. The two decisions that CANNOT be
  // retaken are the streaming answer (a function-local static) and the slot store's
  // geometry, and neither is reached by a load that fails on a missing checkpoint. So
  // engine B's whole document — including `expert_stream`, which is latchABLE but
  // has not been latched in this process — must install.
  vllm::entrypoints::EngineParams a;
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, a));
  const vllm::GgufLoadPolicy policy_a = vllm::GgufLoadPolicy::FromEnv();
  (void)policy_a;

  vllm::entrypoints::EngineParams b;
  b.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"mmap":{"enabled":false,"prefault":false},)"
      R"("expert_stream":{"enabled":true,"slots":8000}}})");
  // It fails on the missing checkpoint, NOT on a refused install...
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, b));
  // ...and the whole document took, which is what makes accepting it correct rather
  // than merely permissive: engine B's own load reads the values it asked for.
  const vllm::WeightResidencyConfig active = vllm::ActiveWeightResidencyConfig();
  REQUIRE(active.mmap.has_value());
  CHECK(*active.mmap == false);
  REQUIRE(active.expert_stream.has_value());
  CHECK(*active.expert_stream == true);
  REQUIRE(active.expert_stream_slots.has_value());
  CHECK(*active.expert_stream_slots == 8000);
  CHECK_FALSE(vllm::GgufLoadPolicy::FromEnv().mmap_residency);
}

TEST_CASE("residency reach: two DIFFERENT PARTIAL documents after a streaming latch") {
  ResidencyFixture fx;
  ::unsetenv("VT_GGUF_MMAP");
  ::unsetenv("VT_GGUF_PREFAULT");
  ::unsetenv("VT_CPU_REF");
  ::unsetenv("VT_MOE_EXPERT_STREAM");

  // The #1133 H1/H2 shape, driven through the two production entry points rather
  // than through `SetWeightResidencyConfig` directly. The unit case in
  // `test_weight_residency_config.cpp` proves the merge; this one proves an
  // operator can reach it, because both defects were reported against
  // `LoadedEngine::FromModelDir` and `vllm_engine_load`.
  //
  // Engine A carries the full document.
  vllm::entrypoints::EngineParams a;
  a.weight_residency =
      vllm::parse_weight_residency_extension_json(kResidencyJson);
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, a));
  REQUIRE(vllm::ActiveWeightResidencyConfig().expert_stream_slots.has_value());

  // A's decode then takes the streaming decision. A load that stops on a missing
  // checkpoint never reaches a routed expert slice, so the decision is taken here
  // through the same resolver `Qwen35ExpertStreamRequested` calls.
  const bool decided = vllm::ResolveExpertStreamRequested();
  (void)decided;
  REQUIRE(vllm::WeightResidencyLatched(vllm::ResidencyLatch::kExpertStream));

  // Engine B arrives through the public C ABI with a GENUINELY PARTIAL document:
  // `mmap.enabled` and nothing else.
  //
  // THE RETURN CODE CANNOT TELL THE TWO OUTCOMES APART, and that is why the
  // assertions below read the global. A refused install throws `std::logic_error`
  // out of `FromModelDir`, which leaves `vllm_engine_load` by the same path a
  // missing checkpoint does, so both spell VLLM_ERR_MODEL_LOAD. The discriminator
  // is `mmap`: B asks for FALSE where A installed TRUE, so the value is A's when
  // the install was refused and B's when it ran.
  vllm_model_params p = vllm_model_params_default();
  p.model_path = kMissingModel;
  p.offload_config = R"({"vllm_cpp":{"mmap":{"enabled":false}}})";
  vllm_engine* eng = nullptr;
  CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_MODEL_LOAD);
  CHECK(eng == nullptr);
  {
    const vllm::WeightResidencyConfig now = vllm::ActiveWeightResidencyConfig();
    REQUIRE(now.mmap.has_value());
    CHECK(*now.mmap == false);
    // ...and engine A's streaming fields, which B said nothing about, are still
    // there for the slot store to read when it is finally built.
    REQUIRE(now.expert_stream.has_value());
    CHECK(*now.expert_stream == true);
    REQUIRE(now.expert_stream_slots.has_value());
    CHECK(*now.expert_stream_slots == 8000);
    REQUIRE(now.prefault.has_value());
    CHECK(*now.prefault == false);
  }
  CHECK_FALSE(vllm::GgufLoadPolicy::FromEnv().mmap_residency);

  // Engine C: a SECOND, DIFFERENT partial document, this time through the loader.
  vllm::entrypoints::EngineParams c;
  c.weight_residency = vllm::parse_weight_residency_extension_json(
      R"({"vllm_cpp":{"mmap":{"prefault":true}}})");
  CHECK_THROWS(vllm::entrypoints::LoadedEngine::FromModelDir(kMissingModel, c));
  {
    const vllm::WeightResidencyConfig now = vllm::ActiveWeightResidencyConfig();
    REQUIRE(now.prefault.has_value());
    CHECK(*now.prefault == true);
    REQUIRE(now.mmap.has_value());
    CHECK(*now.mmap == false);
    REQUIRE(now.expert_stream_slots.has_value());
    CHECK(*now.expert_stream_slots == 8000);
  }
  CHECK(vllm::ResolveExpertStreamSlots() == 8000);
}

TEST_CASE("residency reach: the C ABI REFUSES a mistyped extension at the boundary") {
  ResidencyFixture fx;

  // The same contract the mirrored half already holds (tests/capi/test_capi.cpp,
  // "offload_config defaults to NULL and is parsed+validated"): a caller error is
  // INVALID_ARGUMENT, reported before any model I/O, never MODEL_LOAD. Without
  // this the typo would be swallowed and the run would quietly not stream.
  const char* refused[] = {
      R"({"vllm_cpp":{"mmapp":{"enabled":true}}})",
      R"({"vllm_cpp":{"expert_stream":{"slots":0}}})",
      R"({"vllm_cpp":{"expert_stream":{"enabled":"yes"}}})",
      // The TOP-LEVEL typo, through the public ABI. This is the one that used to
      // return MODEL_LOAD — i.e. the document was accepted, empty — so a library
      // client got an engine running this tier at its defaults and no diagnostic.
      R"({"vllm-cpp":{"mmap":{"enabled":true}}})",
      R"({"VLLM_CPP":{"mmap":{"enabled":true}}})",
  };
  for (const char* doc : refused) {
    CAPTURE(doc);
    vllm_model_params p = vllm_model_params_default();
    p.model_path = kMissingModel;
    p.offload_config = doc;
    vllm_engine* eng = nullptr;
    CHECK(vllm_engine_load(&p, &eng) == VLLM_ERR_INVALID_ARGUMENT);
    CHECK(eng == nullptr);
    CHECK(vllm::ActiveWeightResidencyConfig().empty());
  }
}
