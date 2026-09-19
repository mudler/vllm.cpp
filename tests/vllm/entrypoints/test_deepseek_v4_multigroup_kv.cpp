// MODEL-MM-deepseek-v4 (#2411) — CAN A REAL DeepSeek-V4 CHECKPOINT BE SERVED
// WHEN ITS KV CACHE GROUPS DISAGREE ABOUT THEIR BLOCK SIZE?
//
// `test_serve_deepseek_v4_mm` drives the real `VllmServerMain` and passes, and
// it cannot see this defect: its fixture leaves `attention.compress_ratios` all
// zero, so every layer clamps to ratio 1, no layer carries a compressor or an
// indexer, and `MakeDeepseekV4KVCache` publishes exactly ONE group — the SWA
// cache at the hard-coded 64 tokens (`sparse_swa.py:76-83`). One group takes the
// `UnitaryKVCacheCoordinator`, where the scheduler's hash granularity and the
// group's block size are trivially the same number.
//
// A REAL Flash checkpoint SETS those ratios. Then the factory publishes up to
// seven groups at block sizes 256, 64, 4 and 8, no single granularity is every
// group's block size, and the engine takes the `HybridKVCacheCoordinator`
// instead. That is the shape nothing in this tree exercised, which is exactly
// why it survived.
//
// WHAT UPSTREAM DOES, because this is a mirror and not a design decision.
// `resolve_kv_cache_block_sizes` resolves TWO different quantities from the same
// group set: the scheduler's alignment invariant is their LCM, and the
// prefix-hash granularity is their GCD (`vllm/v1/core/kv_cache_utils.py:678-770`
// @ `e126687a9a`). A group whose block size is a MULTIPLE of that granularity
// then reads its hashes through a converting view rather than refusing:
// `BlockHashListWithBlockSize` takes the last fine hash inside each coarse block,
// which is already chained over that block's whole prefix
// (`kv_cache_utils.py:2358-2464`). Upstream's own coordinator asserts only
// DIVISIBILITY (`kv_cache_coordinator.py:608-613`) — it never requires equality.
// So prefix caching stays ON for such a model, and the engine serves.
//
// This suite enters at `LoadedEngine::FromModelDir`, which is the loader entry
// every server and command line takes for a `.gguf` argument, on its DEFAULT
// configuration. A test that built the coordinator by hand would prove the class
// works and say nothing about whether a checkpoint can be served.
#include <doctest/doctest.h>

// The child-process machinery below needs these. `tests/` is outside the
// Windows source contract (`scripts/check-windows-portability.py` sweeps only
// the shipped-server sources plus `src/vllm/platform/`), and two suites in this
// tree already re-exec themselves this way: `test_none_hash_determinism.cpp`
// and `test_serve_hf_model.cpp:371`.
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/registry.h"
#include "vllm/v1/worker/gpu/runner.h"

namespace {

// The ratios that give the multi-group topology on the fixture's three layers:
// one ratio-4 layer (compressed latent + indexer key cache + the attention and
// indexer compressor states), one ratio-128 layer (latent + one compressor
// state) and one plain layer that has only the SWA cache. Upstream accepts 1, 4
// and 128 and nothing else (`sparse_swa.py:44-55`); a raw 0 is upstream's own
// "no DSA on this layer" and clamps to 1 (`attention.py:205-212`).
const std::vector<int32_t> kFlashRatios = {4, 128, 0};

// DeepSeek-V4's REAL config, in the shape `MakeDeepseekV4KVCache` reads it. The
// same three-layer ratio vector as the checkpoint above, so the group set this
// asserts is the group set the engine case below builds.
vllm::HfConfig FlashLikeConfig() {
  vllm::HfConfig cfg;
  cfg.architectures = {"DeepseekV4ForCausalLM"};
  cfg.hidden_size = 32;
  cfg.num_hidden_layers = 3;
  cfg.vocab_size = 16;
  cfg.num_attention_heads = 2;
  cfg.num_key_value_heads = 1;
  cfg.head_dim = 512;
  cfg.rms_norm_eps = 1e-6;
  cfg.max_position_embeddings = 4096;
  nlohmann::json ratios = nlohmann::json::array();
  for (const int32_t r : kFlashRatios) ratios.push_back(r);
  cfg.raw = {
      {"hidden_size", 32},          {"num_hidden_layers", 3},
      {"vocab_size", 16},           {"num_attention_heads", 2},
      {"num_key_value_heads", 1},   {"head_dim", 512},
      {"qk_rope_head_dim", 64},     {"q_lora_rank", 32},
      {"o_lora_rank", 32},          {"o_groups", 2},
      {"sliding_window", 128},      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 4096},
      {"n_routed_experts", 4},      {"num_experts_per_tok", 2},
      {"moe_intermediate_size", 32},{"n_shared_experts", 1},
      {"norm_topk_prob", true},     {"routed_scaling_factor", 1.0},
      {"swiglu_limit", 10.0},       {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},  {"num_hash_layers", 1},
      {"expert_dtype", "fp4"},      {"hc_mult", 2},
      {"hc_sinkhorn_iters", 3},     {"hc_eps", 1e-6},
      {"index_head_dim", 32},       {"index_n_heads", 2},
      {"index_topk", 3},            {"compress_rope_theta", 160000},
      {"rope_theta", 10000},        {"tie_word_embeddings", false},
      {"compress_ratios", ratios},
  };
  return cfg;
}

// IS THE COORDINATOR'S DEFERRAL ASSERT COMPILED INTO THIS BUILD? This is the
// ONE place in the suite that reads `NDEBUG`, and it reads it into a value
// rather than into a branch around an assertion. Every case below runs in both
// configurations and asserts the same invariant; this flag only selects WHICH
// observed outcome of the child process is the correct one, because the
// production behaviour genuinely differs between the two builds and a test that
// hid that difference would be describing neither.
#ifdef NDEBUG
constexpr bool kDeferralAssertLive = false;
#else
constexpr bool kDeferralAssertLive = true;
#endif

// What the parent learned about the child that tried to construct the engine.
struct ChildOutcome {
  bool exited = false;       // terminated normally rather than by a signal
  int exit_code = -1;        // meaningful only when `exited`
  bool aborted = false;      // killed by SIGABRT, which is what a live assert does
  int signal_number = 0;     // the signal, when one killed it
  std::string output;        // the child's stdout and stderr, interleaved
};

// Run THIS binary again, on the skip-decorated child case, and report how it
// died. The child writes both streams into `capture` because glibc prints the
// failed assertion to stderr immediately before it raises SIGABRT: that text is
// the only evidence that says WHICH assert fired, and a run that reported a
// bare "aborted" could not tell the deferral marker apart from any other abort.
ChildOutcome RunConstructChild() {
  char exe[4096];
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE(n > 0);
  exe[n] = '\0';

  const std::filesystem::path capture =
      std::filesystem::temp_directory_path() /
      ("dsv4_multigroup_child." + std::to_string(::getpid()) + ".log");

  const pid_t pid = ::fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    const int fd = ::open(capture.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) std::_Exit(126);
    ::dup2(fd, 1);
    ::dup2(fd, 2);
    ::close(fd);
    // `--no-skip` is required: the child case is skip-decorated, so a normal run
    // of this suite never executes it and it costs one engine load only here.
    const char* child_argv[] = {
        exe, "--no-skip", "--test-case=dsv4_multigroup_construct_child",
        nullptr};
    ::execv(exe, const_cast<char* const*>(child_argv));
    std::_Exit(127);
  }

  int status = 0;
  REQUIRE(::waitpid(pid, &status, 0) == pid);

  ChildOutcome out;
  out.exited = WIFEXITED(status) != 0;
  if (out.exited) out.exit_code = WEXITSTATUS(status);
  if (WIFSIGNALED(status) != 0) {
    out.signal_number = WTERMSIG(status);
    out.aborted = out.signal_number == SIGABRT;
  }
  std::ifstream in(capture);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  out.output = buffer.str();
  std::error_code ignored;
  std::filesystem::remove(capture, ignored);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) THE TOPOLOGY, AND THE TWO NUMBERS IT RESOLVES TO.
//
// This is the WHY for the case below, taken through the factory pointer the
// loader dereferences rather than by calling `MakeDeepseekV4KVCache` by name.
// It is not the capability gate: it establishes that the group set really does
// disagree about its block size, and that the resolver answers with a hash
// granularity that is SMALLER than most of those groups — which is the input
// the coordinator then has to be able to accept.
// ---------------------------------------------------------------------------
TEST_CASE("a ratio-bearing DeepSeek-V4 publishes groups that no single block size covers") {
  const vllm::HfConfig cfg = FlashLikeConfig();
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(cfg);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);

  // 256 is the architecture's own floor (`kv_block_size_floor`), which is what a
  // default-configured engine resolves to.
  const vllm::v1::KVCacheConfig kv =
      reg.factory->make_kv_cache(cfg, /*block_size=*/256, /*num_blocks=*/4);

  // Seven groups: C4A latent, C128A latent, indexer key, SWA, and the three
  // compressor-state populations.
  REQUIRE(kv.kv_cache_groups.size() == 7);
  std::vector<int> sizes;
  for (const auto& group : kv.kv_cache_groups) {
    REQUIRE(group.kv_cache_spec != nullptr);
    sizes.push_back(group.kv_cache_spec->block_size);
  }
  CHECK(sizes == std::vector<int>{256, 256, 256, 64, 4, 4, 8});

  // The LCM schedules and the GCD hashes (`kv_cache_utils.py:678-770`). The
  // second number is the one that matters here: it is 4, so FIVE of the seven
  // groups page COARSER than the granularity their hashes are computed at.
  const auto [scheduler_block_size, hash_block_size] =
      vllm::v1::resolve_kv_cache_block_sizes(
          kv, /*cache_block_size=*/4, /*prefix_match_unit=*/std::nullopt,
          /*enable_prefix_caching=*/true, /*connector_enabled=*/false,
          /*dcp_world_size=*/1);
  CHECK(scheduler_block_size == 256);
  CHECK(hash_block_size == 4);
}

// ---------------------------------------------------------------------------
// (2) WHAT ACTUALLY HAPPENS TO SUCH A CHECKPOINT TODAY. THIS CASE HAS BEEN
// INVERTED ONCE ALREADY, AND THE HISTORY IS THE POINT.
//
// HISTORY, so a reader can tell which behaviour is current and which is merely
// guarded:
//
//   PREDICTION (never true): seven groups at four different block sizes reach
//   `HybridKVCacheCoordinator` and die on its LOCAL deferral assert that every
//   group's block size EQUALS the hash granularity
//   (`kv_cache_coordinator.cpp:386`).
//
//   MEASURED 2026-09-13, BEFORE #2455/W8: the engine never reached the
//   coordinator. `ApplyCacheDType` runs while `kv_cfg_` is initialized, which
//   PRECEDES `scheduler_block_size_` and `scheduler_` in the `LoadedEngine`
//   initializer list, and `RetypeAttentionSpec` refused any `MLAAttentionSpec`
//   BY NAME (`src/vllm/v1/kv_cache_interface.cpp:398`). This case asserted that
//   refusal, and it was written to go RED the moment W8 landed.
//
//   MEASURED 2026-09-13, AFTER #2455/W8 (this case): THE REFUSAL NO LONGER
//   FIRES ON THE DEFAULT PATH, AND THE ENGINE CONSTRUCTS. The tripwire did its
//   job and is inverted here rather than deleted.
//
// WHAT ACTUALLY REMOVED THE REFUSAL IS RESOLUTION, NOT A WIDER GUARD, and that
// distinction is the whole reason case (3) below still exists. `ApplyCacheDType`
// now returns immediately when the resolved cache dtype is `auto`
// (`kv_cache_interface.cpp`, the `if (resolved.is_auto) return;` short-circuit
// added by W8 slice 6): `auto` MEANS "use the dtype the model resolved", so
// there is nothing to apply, and DeepSeek-V4's own factory publishing
// `fp8_ds_mla` specs is that model's resolution rather than an operator
// override. `RetypeAttentionSpec` is UNTOUCHED and still refuses every EXPLICIT
// override, which case (3) pins.
//
// SO THE COORDINATOR ASSERT IS NOW REACHED, AND IT DOES NOT FIRE. Three
// measured facts, none of them assumed:
//   - DeepSeek-V4 registers `is_hybrid = false` and `has_inner_state = false`
//     (`deepseek_v4_registry.cpp:54-55`), so `ResolveEnablePrefixCaching`
//     returns TRUE and `get_kv_cache_coordinator` does NOT take the
//     `KVCacheCoordinatorNoPrefixCache` arm;
//   - seven groups is `num_groups != 1`, so it takes `HybridKVCacheCoordinator`,
//     whose constructor DOES evaluate the equality at `:386`;
//   - that line is a plain `assert`, and every shipping configuration compiles
//     with `-DNDEBUG` (`CMAKE_CXX_FLAGS_RELEASE = -O3 -DNDEBUG`), so it is
//     compiled out of Release and of CI, which builds Release throughout.
//
// A DEBUG BUILD OF THIS SUITE THEREFORE ABORTS, AND THAT WAS MEASURED RATHER
// THAN REASONED. Any build without `-DNDEBUG` — which INCLUDES the repository's
// DEFAULT configure, `cmake -S . -B build -G Ninja` with `CMAKE_BUILD_TYPE`
// EMPTY, not only `-DCMAKE_BUILD_TYPE=Debug` — gives SIGABRT while the engine is
// constructed, verbatim:
//
//   kv_cache_coordinator.cpp:386: vllm::v1::HybridKVCacheCoordinator::
//   HybridKVCacheCoordinator(...): Assertion `g.kv_cache_spec->block_size ==
//   hash_block_size && "differing group/hash block sizes are DEFERRED (M1.3
//   Task 3)"' failed.
//
// AND THAT IS THE TRAP THIS COMMENT EXISTS TO NAME: **AN ASSERT-BASED WALL IS
// INVISIBLE UNDER NDEBUG, SO A TEST THAT RELIES ON THE SIGABRT SILENTLY PASSES
// IN RELEASE.** `:386` is a bare `assert` and nothing in that file guards it, so
// under NDEBUG it — and `:382`, `:390`, `:391` — are deleted outright and the
// load proceeds INTO the `BlockHashListWithBlockSize` path that the comment
// directly above `:386` calls DEFERRED, with the invariant violated. A run that
// then emits plausible text has not shown the path is correct; it has shown the
// check was removed. Any load result on this topology must therefore state its
// `CMAKE_BUILD_TYPE` and whether NDEBUG was defined, or it means nothing.
//
// SO THIS CASE RUNS THE LOAD IN A CHILD PROCESS, and that is the whole repair
// (ISSUE-LOCAL-01M2EHJ5N35KCEC05R3VWHM0VH). Until 2026-09-13 the load ran
// in-process and asserted construction, which made the SUITE'S VERDICT A
// FUNCTION OF `-DNDEBUG`: green in Release, and a SIGABRT that killed the runner
// on a default checkout. A gate whose answer is a build flag is not a gate. An
// abort still cannot be caught in-process, so the observation moved OUT of the
// process instead: the child (`dsv4_multigroup_construct_child`, skip-decorated,
// re-exec'd by name exactly as `test_none_hash_determinism.cpp` does) performs
// the load and its death is read as an exit status plus its captured stderr.
// BOTH BUILDS THEN RUN THE SAME CASE AND ASSERT THE SAME INVARIANT — that
// NOTHING REFUSES THIS TOPOLOGY BY NAME — and only the outcome that follows it
// differs, because the product itself differs. `kDeferralAssertLive` is the one
// place `NDEBUG` is read, and it selects which observed death is correct; it
// does not delete, weaken or widen any assertion, and `:386` is untouched.
//
// The coordinator arithmetic still says `:386` is the only wall INSIDE the
// coordinator: for `{256,256,256,64,4,4,8}` at scheduler 256 / hash 4, the
// divisibility guards at `:138` (256%4), `:140` (256 % each group) and `:382`
// (each group % 4) all PASS, and only the strict equality at `:386` fails.
// Upstream asserts divisibility ALONE (`kv_cache_coordinator.py:608-613`), so
// our extra equality is a local deferral marker and an abort naming it is the
// CORRECT outcome rather than a defect to route around.
//
// **IT IS NO LONGER THE ONLY WALL A REAL CHECKPOINT MEETS, AND THAT WAS
// MEASURED.** This comment claimed it was, and the claim is now false. On
// 2026-09-13, rc job `b622dd45-d763-41d6-9fe3-c1822100163d` on `dgx:gpu0`, a
// RELEASE build of base `7a62a7fca` served the real 82,438,622,112-byte
// DeepSeek-V4-Flash-Vision-Exp UD-IQ1_S GGUF through `vllm-cli`. The engine
// CONSTRUCTED (`:386` compiled out, as this comment predicts), auto-fit
// `max_model_len` from 1048576 to 65536 for 256 blocks x 256 tokens, enabled
// asynchronous scheduling, and then died in the FORWARD after 1026 s with zero
// output bytes:
//
//   engine-fatal: EngineCore busy loop threw: vt: DeepseekV4 DEVICE forward
//   (W7-device) not implemented — ... at
//   src/vllm/model_executor/models/deepseek_v4.cpp:4658
//
// That is `VT_CHECK(deepseek_v4::V4DeviceKernelsAvailable(), kDevicePending)` in
// `DeepseekV4Model::ForwardDevice`. So on the PRODUCTION path the current wall
// is W7-device, not the coordinator, and this suite gates the door the engine
// now walks through rather than the one it stops at. The run also passed
// `--device cpu` and took the DEVICE forward anyway, because
// `ForwardDeepseekV4ForCausalLM` selects that arm on `input.gather_logits` alone
// (`deepseek_v4_registry.cpp:253`) and reads nothing about `input.queue.device`;
// that is filed as ISSUE-LOCAL-01M2EHJFGT76K4DVBN31HAEY9E and is NOT repaired
// here, because moving a forward route is a production change with its own spec
// and review.
//
// A NAMED REFUSAL WOULD BE A BETTER GUARD THAN THIS ASSERT, for exactly the
// reason above: a refusal is visible in the configuration that ships, and an
// assert is not. That change is NOT made here — it would move a production
// refusal and belongs to the wave that owns the hash-granularity port, not to a
// test repair.
// The hash-granularity port (`BlockHashListWithBlockSize`) recorded under
// `## Owed` in `.agents/specs/deepseek-v4-flash-vision.md` is therefore STILL
// owed: what changed is that nothing refuses first any more, not that the
// converting view arrived. Upstream asserts DIVISIBILITY only
// (`kv_cache_coordinator.py:608-613`); our extra equality is a local deferral
// marker, and `{256,256,256,64,4,4,8}` against a granularity of 4 satisfies
// upstream's rule while violating ours.
// ---------------------------------------------------------------------------

// THE CHILD. Skip-decorated, so a normal run never executes it and the engine
// load it performs costs nothing until the parent case asks for it by name. It
// reports through a MARKER LINE rather than through an exit code alone, so a
// named refusal (which is a message, not a death) stays distinguishable from a
// construction and from an abort. `std::_Exit` keeps doctest's own teardown and
// summary out of the captured stream, so what the parent reads after a marker
// is the product's output and nothing else.
TEST_CASE("dsv4_multigroup_construct_child" * doctest::skip()) {
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/false, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true, /*vision_bias_scale=*/1.0f,
      /*sliding_window=*/128, /*hash_layers=*/dsv4_lang_test::kHashLayers,
      /*compress_ratios=*/kFlashRatios));

  // DEFAULT params: `kv_cache_dtype == "auto"`, which is what `vllm-cli` and
  // `vllm-server` pass when no `--kv-cache-dtype` is given. This is the
  // production configuration, not a contrived one.
  vllm::entrypoints::EngineParams params;
  REQUIRE(params.kv_cache_dtype == "auto");

  std::unique_ptr<vllm::entrypoints::LoadedEngine> engine;
  try {
    engine = vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
  } catch (const std::exception& e) {
    std::printf("CONSTRUCT=refused:%s\n", e.what());
    std::fflush(stdout);
    std::_Exit(0);
  }
  // MODEL-MM-deepseek-v4 per-group attention-backend dispatch
  // (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F): report the backend the runner
  // RESOLVED for each published cache, beside that cache's own block size. This
  // is read from `LoadedEngine::runner()`, so it is the engine's own resolution
  // on the DEFAULT configuration and not a second computation of it.
  if (engine != nullptr) {
    const vllm::v1::GPUModelRunner& runner = engine->runner();
    const std::vector<std::string>& names = runner.attn_backend_names();
    const std::vector<vllm::PagedKvCache>& caches = runner.attn_kv();
    std::string line = "BACKENDS=";
    for (size_t i = 0; i < names.size(); ++i) {
      if (i != 0) line += ";";
      line += (names[i].empty() ? std::string("-") : names[i]) + ":" +
              std::to_string(i < caches.size() ? caches[i].block_size : -1);
    }
    std::printf("%s\n", line.c_str());
  }
  std::printf("CONSTRUCT=%s\n", engine != nullptr ? "ok" : "null");
  std::fflush(stdout);
  std::_Exit(0);
}

TEST_CASE("serve: a multi-group DeepSeek-V4 checkpoint reaches the coordinator, and nothing refuses it BY NAME") {
  const ChildOutcome child = RunConstructChild();
  INFO("child output:\n" << child.output);

  // (i) THE INVARIANT, asserted identically in both configurations, because it
  // is true in both: every NAMED door — `RetypeAttentionSpec`'s `fp8_ds_mla`
  // refusal above all, which case (3) still pins on an explicit override — is
  // reached BEFORE the coordinator is built, so a refusal would appear here
  // whether or not the deferral assert survives the compiler.
  CHECK(child.output.find("CONSTRUCT=refused:") == std::string::npos);
  // ...and the child really ran: 127 is a failed `execv` and 126 a capture file
  // that could not be opened, both of which would otherwise read as "no refusal"
  // from a process that never loaded anything.
  CHECK(child.exit_code != 127);
  CHECK(child.exit_code != 126);
  CHECK_FALSE(child.output.empty());

  // (ii) WHAT THEN HAPPENS, which the build genuinely decides. Each arm pins its
  // own evidence, so neither can be satisfied by the other's outcome.
  if (kDeferralAssertLive) {
    // No NDEBUG: `:386` is compiled in, the seven groups violate it, and the
    // load dies there. The text is asserted because "it aborted" alone would
    // accept an abort from anywhere else in the loader.
    CHECK(child.aborted);
    CHECK(child.output.find(
              "differing group/hash block sizes are DEFERRED") !=
          std::string::npos);
    CHECK(child.output.find("kv_cache_coordinator.cpp") != std::string::npos);
    CHECK(child.output.find("CONSTRUCT=ok") == std::string::npos);
  } else {
    // NDEBUG: `:386` is gone, and the engine constructs — the production
    // outcome, and the one the rc-job measurement above then carried into the
    // forward.
    CHECK_FALSE(child.aborted);
    CHECK(child.exited);
    CHECK(child.exit_code == 0);
    CHECK(child.output.find("CONSTRUCT=ok") != std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// (3) THE GUARD IS STILL THERE, AND THIS IS THE CASE THAT PROVES IT.
//
// Case (2) records that the DEFAULT path no longer refuses. That must not be
// read as "the fp8_ds_mla store/read gap stopped mattering". An EXPLICIT
// `--kv-cache-dtype` is the operator asking for a different page format instead
// of delegating the choice to the model, `resolved.is_auto` is then false, the
// short-circuit does not apply, and `RetypeAttentionSpec` refuses the
// `MLAAttentionSpec` groups BY NAME exactly as before.
//
// WITHOUT THIS CASE the suite could not tell "W8 landed" apart from "somebody
// deleted the guard", because both look identical from case (2) alone. That is
// the regression this case exists to make loud.
// ---------------------------------------------------------------------------
TEST_CASE("serve: an EXPLICIT --kv-cache-dtype on the same topology is still refused BY NAME") {
  gguf_test::TempFile lang(dsv4_lang_test::BuildDeepseek4Gguf(
      /*vision=*/false, dsv4_lang_test::BiasWidths{}, /*vision_from=*/0,
      /*head_dim=*/512, /*with_tokenizer=*/true, /*vision_bias_scale=*/1.0f,
      /*sliding_window=*/128, /*hash_layers=*/dsv4_lang_test::kHashLayers,
      /*compress_ratios=*/kFlashRatios));

  vllm::entrypoints::EngineParams params;
  // An operator naming a page format the MLA store cannot write. Not "auto", so
  // the W8 short-circuit is bypassed and the guard is the next thing reached.
  params.kv_cache_dtype = "fp8";

  std::string message;
  try {
    std::unique_ptr<vllm::entrypoints::LoadedEngine> engine =
        vllm::entrypoints::LoadedEngine::FromModelDir(lang.path(), params);
    FAIL_CHECK(
        "an explicit --kv-cache-dtype was ACCEPTED on an MLA topology: the "
        "fp8_ds_mla guard in RetypeAttentionSpec has been removed or widened, "
        "which lets a 584-byte packed page be written as though it were float "
        "(#2455, KV-DSV4-MULTICACHE)");
  } catch (const std::exception& e) {
    message = e.what();
  }

  // It is a refusal that names the layout it cannot serve...
  CHECK(message.find("fp8_ds_mla") != std::string::npos);
  // ...and the issue that owes the store and the read.
  CHECK(message.find("2455") != std::string::npos);
  // ...and it is the CACHE DTYPE door, not the coordinator's. If this ever reads
  // as a hash-block-size complaint instead, the refusal order moved and every
  // comment above it is stale.
  CHECK(message.find("hash_block_size") == std::string::npos);
}

// ---------------------------------------------------------------------------
// (4) THE WALL A REAL CHECKPOINT MEETS ON A CUDA BOX, AND THE UPSTREAM SHAPE
//     THAT REMOVES IT. ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F.
//
// MEASURED 2026-09-14 on `dgx:gpu0` (GB10, sm_121a, Release/NDEBUG, rc job
// fc593c9f) against the real 82,438,622,112-byte
// `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` @ `b977d3c0ea2d`: the tower
// materialized (VmHWM 79.6 GiB, 886 s) and the engine then died, verbatim:
//
//   vllm-cli: model load failed (status 2): vllm_engine_load: Block size must
//   be a multiple of 16.
//
// WHICH BACKEND FIRED, since the string is emitted byte-identically at THREE
// sites (`backend.cpp:253`, `:268`, `:279`) and the log names none of them.
// Every group this factory publishes is FUSED — `MLAAttentionSpec` or
// `SlidingWindowMLASpec` — so `runner.cpp:1412` marks all nine caches MLA and
// the view loop takes the `is_mla` arm for every one of them. That arm resolved
// the MLA backend ONCE and cached it (`mla_backend_resolved`), so the first
// group (256 tokens) resolved `TRITON_MLA` and the 4/4/8 groups inherited it:
// `backend.cpp:279`. On a CPU box the same arm resolves NOTHING (no MLA backend
// is registered for `kCPU`), the name is empty, and `CheckKvCacheShape` is
// skipped entirely — which is why this suite was green on a defect that stops
// every CUDA load. BOTH OUTCOMES ARE THE SAME DEFECT: one backend is asked
// about a group it does not serve.
//
// UPSTREAM DISPATCHES PER GROUP, and that is what this case pins. The LAYER
// names its own backend (`gpu_model_runner.py:7150`,
// `layers[layer_name].get_attn_backend()`, keyed at `:7170`): the compressor
// states name `CompressorBackend`, which declares `[MultipleOf(1)]`
// (`compressor.py:66-68`, `:189-190`), the SWA cache names
// `DeepseekSparseSWABackend` at `[MultipleOf(64)]` (`sparse_swa.py:116-118`,
// `:126-128`), and the indexer key cache names `DeepseekV4IndexerBackend` at
// `[256]` (`indexer.py:194-196`). Upstream has NO global 16-multiple rule at
// all: the default is `[MultipleOf(1)]` (`backend.py:72-74`) and the 16 is a
// per-backend override in exactly three backends (`triton_attn.py:314`,
// `triton_mla.py:152`, `rocm_aiter_unified_attn.py:53`).
//
// READ THE COUNTS, NOT THE BANNER, and read this case's FAILURE MESSAGE: a
// green here means the engine resolved a backend that ACCEPTS each group's own
// block size. It does not mean that backend has a kernel. Nothing dispatches on
// `attn_backend_names_` yet (`runner.cpp`, owed to #1332 M4), and the DSA
// forward is the model's own op path.
// ---------------------------------------------------------------------------
namespace {

// Parse `BACKENDS=name:block;name:block;...` out of the child's stream.
std::vector<std::pair<std::string, int>> ParseBackends(const std::string& out) {
  std::vector<std::pair<std::string, int>> parsed;
  const std::size_t at = out.find("BACKENDS=");
  if (at == std::string::npos) return parsed;
  const std::size_t eol = out.find('\n', at);
  const std::string line =
      out.substr(at + 9, eol == std::string::npos ? std::string::npos
                                                  : eol - (at + 9));
  std::size_t pos = 0;
  while (pos <= line.size() && !line.empty()) {
    const std::size_t sep = line.find(';', pos);
    const std::string entry =
        line.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
    const std::size_t colon = entry.rfind(':');
    if (colon != std::string::npos) {
      parsed.emplace_back(entry.substr(0, colon),
                          std::atoi(entry.c_str() + colon + 1));
    }
    if (sep == std::string::npos) break;
    pos = sep + 1;
  }
  return parsed;
}

}  // namespace

TEST_CASE("serve: every DeepSeek-V4 KV group resolves a backend that ACCEPTS its own block size") {
  const ChildOutcome child = RunConstructChild();
  INFO("child output:\n" << child.output);

  // This case is about what the engine resolved, so it needs an engine. In a
  // build where the deferral assert is live the child dies BEFORE it can report,
  // and that outcome is case (2)'s to assert, not this one's.
  if (kDeferralAssertLive) return;
  REQUIRE(child.output.find("CONSTRUCT=ok") != std::string::npos);

  const std::vector<std::pair<std::string, int>> resolved =
      ParseBackends(child.output);
  // Nine caches: 2 latent + 1 indexer key + 3 SWA + 3 compressor states, in the
  // factory's publication order (`deepseek_v4_registry.cpp:546-556`).
  REQUIRE(resolved.size() == 9);

  // (i) THE PROPERTY, per cache. A backend the engine RESOLVED for a group must
  // accept that group's block size. Asked of the backend by name, through the
  // same registry `CheckKvCacheShape` uses, so a name that no device registers
  // fails here rather than at a customer's engine construction.
  const vt::DeviceType device = vt::DeviceType::kCPU;
  for (std::size_t i = 0; i < resolved.size(); ++i) {
    const std::string& name = resolved[i].first;
    const int block = resolved[i].second;
    CAPTURE(i);
    CAPTURE(name);
    CAPTURE(block);
    if (name == "-") continue;  // op-driven: no backend was consulted
    REQUIRE(vllm::v1::HasAttentionBackend(device, name));
    const std::unique_ptr<vllm::v1::AttentionBackend> backend =
        vllm::v1::MakeAttentionBackend(device, name);
    CHECK_MESSAGE(backend->supports_block_size(block),
                  "the engine resolved backend '"
                      << name << "' for a KV cache group whose block size is "
                      << block << ", and that backend REFUSES that block size. "
                      "This is the defect that killed a real "
                      "DeepSeek-V4-Flash-Vision load at engine construction "
                      "with 'Block size must be a multiple of 16.' — one "
                      "backend is being asked about a group it does not serve "
                      "(ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F).");
  }

  // (ii) THE MIRROR. The three DSA group kinds resolve the backends upstream's
  // own layers name, and they resolve them BY NAME rather than by winning a
  // capability walk — upstream never puts these in a priority list either.
  CHECK(resolved[2].first == "DEEPSEEK_V4_INDEXER");   // indexer key, 256
  CHECK(resolved[2].second == 256);
  for (std::size_t i = 3; i < 6; ++i) {
    CAPTURE(i);
    CHECK(resolved[i].first == "DEEPSEEK_SPARSE_SWA");  // sparse SWA, 64
    CHECK(resolved[i].second == 64);
  }
  for (std::size_t i = 6; i < 9; ++i) {
    CAPTURE(i);
    CHECK(resolved[i].first == "CompressorBackend");    // compressor states
  }
  CHECK(resolved[6].second == 4);
  CHECK(resolved[7].second == 4);
  CHECK(resolved[8].second == 8);

  // (iii) The two MLA latent groups are NOT given a name by the factory: they go
  // through the ordinary selector, exactly as upstream's MLA attention layer
  // does. On a CPU box no MLA backend is registered, so they read as op-driven.
  CHECK(resolved[0].first != "CompressorBackend");
  CHECK(resolved[1].first != "CompressorBackend");
}

// ---------------------------------------------------------------------------
// (5) THE GUARD THAT MUST SURVIVE THE FIX.
//
// The 16-multiple refusal is CORRECT for the backends that declare it. Removing
// or widening it would make case (4) green while letting a 4-token group land on
// a backend that genuinely cannot page it. This case fails if anyone "fixes"
// DeepSeek-V4 that way.
// ---------------------------------------------------------------------------
TEST_CASE("the 16-multiple refusal is intact, and the DSA backends declare upstream's own lists") {
  using vllm::v1::AttentionBackend;
  using vllm::v1::MakeAttentionBackend;
  const vt::DeviceType kCpu = vt::DeviceType::kCPU;

  // FLASH_ATTN: flash_attn.py:82-84, MultipleOf(16). Still refuses 4, with the
  // verbatim string a real checkpoint met.
  const std::unique_ptr<AttentionBackend> flash =
      MakeAttentionBackend(kCpu, "FLASH_ATTN");
  CHECK_FALSE(flash->supports_block_size(4));
  CHECK(flash->supports_block_size(16));
  std::string flash_message;
  try {
    flash->get_kv_cache_shape(4, 4, 1, 512);
  } catch (const std::exception& e) {
    flash_message = e.what();
  }
  CHECK(flash_message == "Block size must be a multiple of 16.");

  // CompressorBackend: compressor.py:66-68 [MultipleOf(1)] and :70-72
  // head sizes [512, 1024]. 4 and 8 are the two sizes the model publishes
  // (compressor.py:152-167).
  const std::unique_ptr<AttentionBackend> compressor =
      MakeAttentionBackend(kCpu, "CompressorBackend");
  CHECK(compressor->get_name() == "CompressorBackend");
  CHECK(compressor->get_supported_kernel_block_sizes() == std::vector<int>{1});
  CHECK(compressor->supports_block_size(4));
  CHECK(compressor->supports_block_size(8));
  CHECK(compressor->supports_block_size(1));
  CHECK(compressor->get_supported_head_sizes() ==
        std::vector<int>{512, 1024});
  CHECK(compressor->is_mla());
  // ONE vector per token, not K+V: the fused rank-3 view CheckKvCacheShape
  // expects for every group this model publishes.
  CHECK(compressor->get_kv_cache_shape(7, 4, 1, 2048) ==
        std::vector<int64_t>{7, 4, 2048});

  // DeepseekSparseSWABackend: sparse_swa.py:126-128 [MultipleOf(64)], :133-136
  // head sizes [512]. 64 is what the model publishes (sparse_swa.py:82-87);
  // 4 is refused, which is the whole point of a PER-GROUP answer.
  const std::unique_ptr<AttentionBackend> swa =
      MakeAttentionBackend(kCpu, "DEEPSEEK_SPARSE_SWA");
  CHECK(swa->get_name() == "DEEPSEEK_SPARSE_SWA");
  CHECK(swa->get_supported_kernel_block_sizes() == std::vector<int>{64});
  CHECK(swa->supports_block_size(64));
  CHECK(swa->supports_block_size(256));
  CHECK_FALSE(swa->supports_block_size(4));
  CHECK_FALSE(swa->supports_block_size(16));
  CHECK(swa->get_supported_head_sizes() == std::vector<int>{512});
  CHECK(swa->is_mla());

  // DeepseekV4IndexerBackend: indexer.py:194-196, the EXACT size 256 and not a
  // multiple-of rule, so 64 and 512 are both refused.
  const std::unique_ptr<AttentionBackend> indexer =
      MakeAttentionBackend(kCpu, "DEEPSEEK_V4_INDEXER");
  CHECK(indexer->get_name() == "DEEPSEEK_V4_INDEXER");
  CHECK(indexer->get_supported_kernel_block_sizes() == std::vector<int>{256});
  CHECK(indexer->supports_block_size(256));
  CHECK_FALSE(indexer->supports_block_size(64));
  CHECK(indexer->is_mla());

  // ...and all three are registered for CUDA too, because the load this row
  // exists to unblock is a CUDA load.
  CHECK(vllm::v1::HasAttentionBackend(vt::DeviceType::kCUDA,
                                      "CompressorBackend"));
  CHECK(vllm::v1::HasAttentionBackend(vt::DeviceType::kCUDA,
                                      "DEEPSEEK_SPARSE_SWA"));
  CHECK(vllm::v1::HasAttentionBackend(vt::DeviceType::kCUDA,
                                      "DEEPSEEK_V4_INDEXER"));
  // ...and none of them is in any platform priority list, so a capability walk
  // can never land on one. Upstream's are named by a layer, never selected.
  CHECK(vllm::v1::MakeAttentionBackend(vt::DeviceType::kCUDA, "TRITON_MLA")
            ->get_supported_kernel_block_sizes() == std::vector<int>{16});

  // THE CUDA REFUSAL AND ITS REPAIR, at the exact function that threw on
  // dgx:gpu0. `CheckKvCacheShape` (`registry.cpp:150`) is what
  // `GPUModelRunner::initialize_kv_cache` calls per cache, and it is the call at
  // `registry.cpp:160` that reached `TritonMLABackend::get_kv_cache_shape`.
  // This box has no GPU, but the throw is host metadata and needs none: asking
  // the same question the CUDA engine asked reproduces the same answer.
  //
  // (a) The OLD routing — TRITON_MLA asked about a 4-token compressor group —
  //     still refuses, verbatim. This is the string a real 82 GB checkpoint
  //     died on, and it must stay reachable.
  std::string cuda_message;
  try {
    vllm::v1::CheckKvCacheShape(vt::DeviceType::kCUDA, "TRITON_MLA",
                                /*num_blocks=*/256, /*block_size=*/4,
                                /*num_kv_heads=*/1, /*head_size=*/2048,
                                /*is_mla=*/true);
  } catch (const std::exception& e) {
    cuda_message = e.what();
  }
  CHECK(cuda_message == "Block size must be a multiple of 16.");

  // (b) The NEW routing — the group's OWN backend asked the same question —
  //     accepts it, and declares the fused rank-3 view the engine allocates.
  //     Nothing was widened to get here: TRITON_MLA's answer in (a) is
  //     unchanged.
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(
      vt::DeviceType::kCUDA, "CompressorBackend", /*num_blocks=*/256,
      /*block_size=*/4, /*num_kv_heads=*/1, /*head_size=*/2048,
      /*is_mla=*/true));
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(
      vt::DeviceType::kCUDA, "CompressorBackend", 256, 8, 1, 1024, true));
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(
      vt::DeviceType::kCUDA, "DEEPSEEK_SPARSE_SWA", 256, 64, 1, 512, true));
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(
      vt::DeviceType::kCUDA, "DEEPSEEK_V4_INDEXER", 256, 256, 1, 132, true));
  // ...and the two latent groups keep the answer they always had: 256 is a
  // multiple of 16, so TRITON_MLA serves them exactly as before.
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(vt::DeviceType::kCUDA, "TRITON_MLA",
                                            256, 256, 1, 512, true));
}
