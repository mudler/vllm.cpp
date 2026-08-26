// MiniMax-Music3 — the ENGINE'S OWN CALL to the DEPTH arm selector, gated
// (issue #1839, row MUSIC3-DEPTH-ARM-REACH, lane #672).
//
// ─── THE TWIN THIS FILE IS, AND WHY THERE ARE TWO ───────────────────────────
//
// `test_minimax_music3_device_arm_real.cpp` is the same file for the DiT arm.
// #1131 named BOTH device-arm twins; row `MUSIC3-DIT-ARM-REACH` closed the DiT
// one on `thor:gpu0` and closed #1131 with it, and #1839 is the depth half that
// change does not touch. The two arms ride ONE switch — `--speech-device 1` —
// and are selected by two separate call sites in `Music3SpeechEngine::Synthesize`
// (`minimax_music3_speech.cpp:638` and the DiT block one screen below it). Two
// call sites on one condition is exactly how one of them drifts, which is why
// each gets its own gate rather than one gate asserting both.
//
// ─── WHAT THIS FILE IS FOR, AND WHAT IT IS NOT FOR ──────────────────────────
//
// It is not for the depth decoder, its kernels, its staging or any number.
// Those are `.agents/specs/minimax-music3.md` §19: the arm is held to the host
// reference inside a measured bf16 band, its residency is asserted bit by bit,
// and `test_minimax_music3_ar.cpp` drives `Music3SelectDepthArm` on BOTH sides
// of its condition from a CPU runner. Nothing here touches any of it.
//
// It is for ONE CALL: the two lines where the shipped engine invokes
// `Music3SelectDepthArm`. §19.5 measured what deleting them costs — nothing.
// `test_minimax_music3_ar` stayed 37/37 · 640/640 and `test_minimax_music3_speech`
// stayed 9/9 · 223/223, because the 0.646 B decoder returns to its host loops
// and produces the same song, later. A 0.646 B model that costs 6.3x the 8.6 B
// language model beside it is what that regression looks like from outside, and
// no token gate, golden or tolerance in this tree can see it.
//
// ─── WHY IT ENTERS THROUGH THE C ABI ────────────────────────────────────────
//
// `AGENTS.md` `## Nothing lands dead` lists what counts as a production entry
// point and `include/vllm.h` is the first item on it. `vllm_speech_engine_load`
// plus `vllm_synthesize` is the pair `examples/minimax_music3_gen` is a thin
// client of, and the bundled server's `/v1/audio/speech` route is a SECOND
// production path onto the same engine through `ApiServer::handle_audio_speech`.
// So this file constructs no engine, calls no `Music3DepthStage`, stages no
// weights and builds no arm by hand. Every one of those is `test_minimax_music3_ar`'s
// subject and is gated there. Here the only thing that may select the device arm
// is the shipped engine executing the line under test.
//
// ─── WHICH ARM RAN, READ OFF THE INSTRUMENTS AND NEVER OFF THE AUDIO ────────
//
// The host and device arms agree by design — §19.4a measures their separation
// and §19.5 asserts the drawn codes match — so neither the waveform nor the
// codes can answer the question this file exists to ask. Buckets from the
// engine's own `profile::Report` answer it, and they answer different halves:
//
//   ar.depth_staging   lives INSIDE `Music3SelectDepthArm`, past its CPU early
//                      return. Present iff the engine CALLED the selector and
//                      the selector took the device branch. THIS is the
//                      instrument that answers #1839, and it is the most direct
//                      of the three.
//   ar.depth_device    the production `append` lambda in `Music3DepthStage`
//                      SELECTED the device branch, once per appended position.
//   ar.depth_host      ABSENT. The control that makes the two above mean
//                      something rather than merely be present.
//
// **They are NOT independent of one another on the call-site mutation**, and
// saying otherwise would be the easy overclaim. `ar.depth_device` is emitted
// under `device_arm.engaged()`, and an arm is engaged only through fields
// `Music3SelectDepthArm` sets, so deleting the engine's call reds both at once.
// `ar.depth_staging` is the direct answer; `ar.depth_device` corroborates. They
// separate on OTHER defects, which is why both are asserted: an arm that stages
// and is then dropped by the loop moves the first and not the second, and a
// partial fallback that serves some positions from the host moves the third.
//
// The body-executed question — the `dit.pack` role in the DiT twin — is not
// asked here and the reason is that it is already answered somewhere better.
// `ar.depth_device` sits in the same branch as the `DepthDecoderAppendDevice`
// call it labels, with nothing between them, so there is no room for a
// mislabelled bracket; and `test_minimax_music3_ar.cpp`'s composed-stage case
// holds `Music3DepthDeviceForwardCount()` to an EXACT `num_codebooks` per frame,
// which is a class assertion in the place a class assertion belongs.
//
// ─── WHERE IT RUNS ──────────────────────────────────────────────────────────
//
// Nowhere CI owns. It needs an accelerator and the 28.5 GB checkpoint, and
// `SpeechEngineDeviceType` refuses device 1 on a CPU-only build before a queue
// exists. It runs inside an `rc` lease on a fleet device; `docs/USAGE.md` has
// the recipe. Its CTest entry carries `LABELS "gpu;checkpoint;music3"` so it can
// be selected with `ctest -L gpu`, and that selection is pinned by name in
// `scripts/check-test-registration.py` because `ctest -L` returns 0 over an
// empty selection. A missing precondition EXITS 77 — CTest reports **Skipped**,
// never Passed. A doctest case that merely returned early would print
// `assertions: 0` and `Status: SUCCESS!`, which is the trap this repository has
// already hit twice (issue #463).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm.h"
#include "vllm/model_executor/models/music3_profile.h"
#include "vllm/multimodal/speech_engine.h"
#include "vt/device.h"

namespace fs = std::filesystem;
namespace m3profile = vllm::models::music3::profile;

namespace {

constexpr const char* kFamily = "minimax-music3";

// The smallest request that still generates audio frames. A routing assertion
// needs a frame, not a song — and the reduced request is what makes the
// reachability mutation affordable, because with the engine's call deleted the
// depth decoder returns to the host loops §19.1 measured at 48.4 % of a run.
constexpr double kDurationSeconds = 0.24;
constexpr int32_t kSteps = 2;
constexpr int64_t kSeed = 7;

// A gate that CANNOT RUN must never report success. See the file header.
[[noreturn]] void SkipGate(const std::string& why) {
  std::fprintf(stderr,
               "\n[SKIP] *** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "[SKIP] test_minimax_music3_depth_arm_real: %s\n\n",
               why.c_str());
  std::fflush(stderr);
  std::exit(77);
}

// The checkpoint SET root. `VLLM_CPP_MUSIC3_CHECKPOINT` names it directly;
// `CHECKPOINT_ROOT` names the shared store the whole fleet reads, and the
// family's directory inside it. Same resolution the DiT twin and
// `tests/parity/test_minimax_music3_e2e_real.cpp` use, deliberately, so a leased
// job that can run one can run all three with no second variable.
std::string CheckpointRoot() {
  if (const char* direct = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT")) {
    if (direct[0] != '\0') return direct;
  }
  if (const char* root = std::getenv("CHECKPOINT_ROOT")) {
    if (root[0] != '\0') return (fs::path(root) / "minimax-music3").string();
  }
  return std::string();
}

// Arms the profile table and restores it. Only the OUTER flag: the intra-DiT
// spans insert a `Backend::Synchronize` at every bracket and this gate needs
// none of them — every bucket it reads is emitted unconditionally once the
// instrument is on.
struct ArmedProfile {
  ArmedProfile() : profile_(m3profile::EnabledFlag()) { m3profile::EnabledFlag() = true; }
  ~ArmedProfile() { m3profile::EnabledFlag() = profile_; }
  bool profile_;
};

const m3profile::Bucket* FindBucket(const char* name) {
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    if (bucket.name == name) return &bucket;
  }
  return nullptr;
}

// Every bucket the run produced, printed on the way past. An instrument that
// does not narrate what it compared cannot be audited from its own log, and a
// reader who has to open the source is reviewing the intent rather than the run
// (`.agents/verification.md`).
void PrintTable() {
  std::fprintf(stderr, "\n[music3 depth-arm] the buckets this run produced:\n");
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    std::fprintf(stderr, "[music3 depth-arm]   %-28s calls=%lld seconds=%.3f%s\n",
                 bucket.name.c_str(), static_cast<long long>(bucket.calls), bucket.seconds,
                 bucket.span ? " (span)" : "");
  }
  std::fflush(stderr);
}

}  // namespace

TEST_CASE("music3 depth arm: the ENGINE'S OWN CALL puts the RVQ depth decoder on the"
          " accelerator (GPU + checkpoint)") {
  // ── PRECONDITION 1: an accelerator, resolved exactly the way the engine
  //    resolves it. Not `#ifdef VLLM_CPP_CUDA` and not a backend probe of our
  //    own: `SpeechEngineDeviceType` is the function `Music3SpeechEngine`'s
  //    constructor calls, so asking it is asking the same question the
  //    production path asks, including the partial-backend refusal.
  try {
    const vt::DeviceType resolved = vllm::multimodal::SpeechEngineDeviceType(1, kFamily);
    std::fprintf(stderr, "[music3 depth-arm] device 1 resolves to '%s'\n",
                 vt::DeviceTypeName(resolved));
  } catch (const std::exception& e) {
    SkipGate(std::string("device 1 is refused by this build — ") + e.what());
  }

  // ── PRECONDITION 2: the checkpoint set.
  const std::string root = CheckpointRoot();
  if (root.empty()) {
    SkipGate("neither VLLM_CPP_MUSIC3_CHECKPOINT nor CHECKPOINT_ROOT is set");
  }
  if (!fs::exists(fs::path(root) / "modular_model_index.json")) {
    SkipGate(root + " has no modular_model_index.json (not a Music3 checkpoint SET)");
  }
  std::fprintf(stderr, "[music3 depth-arm] checkpoint set: %s\n", root.c_str());

  const ArmedProfile armed;

  // ── THE PRODUCTION ENTRY POINT, both halves of it ──────────────────────────
  vllm_speech_model_params model = vllm_speech_model_params_default();
  model.path = root.c_str();
  model.family = kFamily;
  // 1, not 0. This is the whole switch: it is what resolves `queue_` to the
  // platform's device, and it is the only thing that makes the engine's call to
  // `Music3SelectDepthArm` return an engaged arm. There is no separate flag and
  // no environment variable, because a capability behind an option nothing turns
  // on is the shape `.agents/reachability.md` calls dead.
  model.device = 1;

  vllm_speech_engine* engine = nullptr;
  const vllm_status load_status = vllm_speech_engine_load(&model, &engine);
  REQUIRE_MESSAGE(load_status == VLLM_OK,
                  "vllm_speech_engine_load(device=1) failed: "
                      << (vllm_last_error() == nullptr ? "(no detail)" : vllm_last_error()));
  REQUIRE(engine != nullptr);

  // GRANTED, not requested. A caller that echoed back its own request could not
  // tell a device arm from a CPU arm with a flag set.
  CHECK_MESSAGE(vllm_speech_engine_device(engine) == 1,
                "the engine GRANTED device " << vllm_speech_engine_device(engine)
                                             << ", so nothing below could have been the "
                                                "device arm");
  CHECK(std::string(vllm_speech_engine_family(engine)) == kFamily);

  vllm_speech_params request = vllm_speech_params_default();
  request.lyrics = "[verse]\nOne short line\n";
  request.description = "Genre: acoustic pop. BPM: 96.";
  request.audio_duration_s = kDurationSeconds;
  request.num_inference_steps = kSteps;
  request.seed = kSeed;

  vllm_speech_result result;
  const vllm_status run_status = vllm_synthesize(engine, &request, &result);
  const std::string run_detail =
      vllm_last_error() == nullptr ? std::string("(no detail)") : std::string(vllm_last_error());
  if (run_status != VLLM_OK) {
    vllm_speech_engine_free(engine);
    FAIL("vllm_synthesize failed: " << run_detail);
  }

  PrintTable();

  // ── THE QUANTITIES THE COUNTS ARE BUILT FROM, taken from the run ───────────
  //
  // `ar.frames` is the number of audio frames the autoregressive loop actually
  // produced, counted by the loop itself; `ar.depth_stage` is the number of
  // times the engine entered `Music3DepthStage`; `ar.depth_forward` brackets
  // every appended position on BOTH arms, so it is the denominator that makes
  // "every append took the device branch" an equality rather than a floor.
  const m3profile::Bucket* frames_bucket = FindBucket("ar.frames");
  const m3profile::Bucket* stage_bucket = FindBucket("ar.depth_stage");
  const m3profile::Bucket* forward_bucket = FindBucket("ar.depth_forward");
  REQUIRE_MESSAGE(frames_bucket != nullptr,
                  "no ar.frames counter: the profile table was not armed, so every "
                  "assertion below would have read absence as a verdict");
  REQUIRE_MESSAGE(stage_bucket != nullptr, "no ar.depth_stage span");
  REQUIRE_MESSAGE(forward_bucket != nullptr, "no ar.depth_forward bucket");
  const int64_t frames = frames_bucket->calls;
  const int64_t stages = stage_bucket->calls;
  const int64_t forwards = forward_bucket->calls;
  REQUIRE_MESSAGE(frames >= 1, "the language model generated " << frames << " audio frames, so "
                                                               << "the depth decoder this gate "
                                                                  "believes it drove never ran");
  REQUIRE_MESSAGE(stages >= 1, "the engine entered Music3DepthStage " << stages << " times");
  REQUIRE_MESSAGE(forwards >= stages,
                  "ar.depth_forward " << forwards << " is below ar.depth_stage " << stages
                                      << ", which cannot happen: every stage appends at least "
                                         "its own prefix position");
  // `num_codebooks` appends per stage: one batch-2 prefix append, then one per
  // residual codebook. It is a checkpoint property (8 at the shipped geometry),
  // so it is DERIVED from the run rather than written here, and only its
  // divisibility is asserted — a run that appended a ragged number of positions
  // is a defect in a place this gate does not own, and it must not read as one
  // here.
  CHECK_MESSAGE(forwards % stages == 0,
                "ar.depth_forward " << forwards << " is not a whole multiple of ar.depth_stage "
                                    << stages);
  MESSAGE("frames=" << frames << " depth stages=" << stages << " appends=" << forwards << " ("
                    << (forwards / stages) << " per stage)");

  // ── #1839's RESIDUAL, asserted: the engine CALLED the selector and the
  //    selector STAGED. This bucket is emitted from inside
  //    `Music3SelectDepthArm`, after its CPU early return, so it is present if
  //    and only if the engine's own call site ran and took the device branch.
  //    Deleting that call site is the reachability mutation
  //    `.agents/reachability.md` prescribes, and this REQUIRE is what it reds.
  const m3profile::Bucket* staging = FindBucket("ar.depth_staging");
  REQUIRE_MESSAGE(staging != nullptr,
                  "no ar.depth_staging bucket: the engine never called "
                  "Music3SelectDepthArm, or the selector declined the queue. The RVQ depth "
                  "decoder ran on the HOST and the song would be correct and far later");
  CHECK_MESSAGE(staging->calls == 1,
                "the depth decoder staged " << staging->calls
                                            << " times; it is staged ONCE per request, outside "
                                               "every loop");

  // ── which branch the production append lambda SELECTED, for every position ─
  const m3profile::Bucket* device_bucket = FindBucket("ar.depth_device");
  REQUIRE_MESSAGE(device_bucket != nullptr,
                  "no ar.depth_device bucket: the production depth stage did NOT take the "
                  "device arm at any position");
  CHECK_MESSAGE(device_bucket->calls == forwards,
                "ar.depth_device took " << device_bucket->calls << " of " << forwards
                                        << " appends, so some position fell back to the host "
                                           "arm");
  CHECK_MESSAGE(FindBucket("ar.depth_host") == nullptr,
                "the run ALSO emitted ar.depth_host, so some position ran the host reference "
                "loop without saying so");

  // ── A CONTROL, NOT A GATE. The two arms agree by design, so no property of
  //    this waveform can say which one ran; what it CAN catch is an arm that
  //    threw halfway and left a plausible buffer behind. No tolerance here is a
  //    claim about correctness — that is `test_minimax_music3_ar_real.cpp` and
  //    `test_minimax_music3_e2e_real.cpp`.
  CHECK(result.sample_rate == vllm_speech_engine_sample_rate(engine));
  REQUIRE(result.n_samples > 0);
  REQUIRE(result.channels > 0);
  REQUIRE(result.samples != nullptr);
  const int64_t total = result.n_samples * result.channels;
  int64_t finite = 0;
  int64_t nonzero = 0;
  double peak = 0.0;
  for (int64_t i = 0; i < total; ++i) {
    const double v = static_cast<double>(result.samples[i]);
    if (std::isfinite(v)) ++finite;
    if (v != 0.0) ++nonzero;
    peak = std::max(peak, std::abs(v));
  }
  CHECK_MESSAGE(finite == total, finite << " of " << total << " samples are finite");
  CHECK_MESSAGE(nonzero > total / 100,
                "only " << nonzero << " of " << total
                        << " samples are non-zero: the run returned near-silence");
  CHECK_MESSAGE(peak > 1e-4, "waveform peak is " << peak << ", which is silence");
  MESSAGE("waveform: " << result.n_samples << " samples x " << result.channels << " ch at "
                       << result.sample_rate << " Hz, peak " << peak);

  vllm_speech_result_free(&result);
  vllm_speech_engine_free(engine);
}
