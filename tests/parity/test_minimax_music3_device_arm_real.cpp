// MiniMax-Music3 — the ENGINE'S OWN CALL to the DiT arm selector, gated
// (issue #1131, row MUSIC3-DIT-ARM-REACH, lane #672).
//
// ─── WHAT THIS FILE IS FOR, AND WHAT IT IS NOT FOR ──────────────────────────
//
// It is not for the DiT, the kernels, the staging, or any number. Those are
// `.agents/specs/minimax-music3.md` §14 and §21, they are gated against
// upstream's own goldens on both arms, and nothing here touches them.
//
// It is for ONE LINE: the call at
// `src/vllm/model_executor/models/minimax_music3_speech.cpp` where
// `Music3SpeechEngine::Synthesize` invokes `Music3SelectDitArm`. Wave 1 of this
// row moved the RULE out of the engine into that function so a CPU gate could
// drive both sides of its condition, and
// `tests/vllm/models/test_minimax_music3_acoustic.cpp` does exactly that. What
// wave 1 could not reach was the CALL, and it said so: deleting it left every
// suite in the tree green, and `music3-dit-arm-reachability.md` `## Owed`
// carried the residual with the mutation that measured it.
//
// A change that deleted that line would return a 2.4B fp32 DiT to the host
// reference loops. The song would still be correct — the two arms agree by
// design — and it would arrive many hours later. That is the failure a token
// gate, a golden gate and a tolerance are all structurally unable to see.
//
// ─── WHY IT ENTERS THROUGH THE C ABI ────────────────────────────────────────
//
// `AGENTS.md` `## Nothing lands dead` lists what counts as a production entry
// point, and `include/vllm.h` is the first item on it. `vllm_speech_engine_load`
// plus `vllm_synthesize` is the pair that `examples/minimax_music3_gen` and the
// bundled server's `/v1/audio/speech` route are both thin clients of. So this
// file constructs no engine, calls no `Music3DenoiseChunks`, and builds no arm
// by hand. Every one of those is wave 1's subject and is gated there. Here the
// only thing that may select the device arm is the shipped engine.
//
// ─── WHICH ARM RAN, READ OFF THE INSTRUMENTS AND NEVER OFF THE AUDIO ────────
//
// The host and device arms agree by design, so output equality cannot answer
// the question this file exists to ask. Three buckets from the engine's own
// `profile::Report` answer it, and they answer different halves of it:
//
//   acoustic.dit_staging   lives INSIDE `Music3SelectDitArm`, past the CPU
//                          early return. Present iff the engine CALLED the
//                          selector and the selector took the device branch.
//                          This is the instrument that answers the residual.
//   denoise.dit_device     the production denoise loop SELECTED the device
//                          branch. `calls` is steps x windows, because one
//                          bracket spans both classifier-free-guidance
//                          branches.
//   dit.pack               lives inside `DitForwardDevice`, so the device
//                          forward's BODY executed. A mislabelled bucket
//                          cannot fake it. `calls` is 2 x steps x windows.
//   denoise.dit_host       ABSENT. The control that makes the three above mean
//                          something rather than merely be present.
//
// `windows` is not a constant this file asserts against itself: it is read from
// the engine's own `denoise.windows` counter, which is the length of the chunk
// vector the loop returned, and `steps` from the `request.steps` counter the
// request resolver emitted. The count assertion is therefore arithmetic over
// two independently produced quantities.
//
// ─── WHERE IT RUNS ──────────────────────────────────────────────────────────
//
// Nowhere CI owns. It needs an accelerator and the 28.5 GB checkpoint, and
// `SpeechEngineDeviceType` refuses device 1 on a CPU-only build before a queue
// exists. It runs inside an `rc` lease on a fleet device; `docs/USAGE.md` has
// the recipe. Its CTest entry carries `LABELS "gpu;checkpoint;music3"` so it can
// be selected with `ctest -L gpu` and excluded with `ctest -LE gpu`, and a
// missing precondition EXITS 77 — CTest reports **Skipped**, never Passed. A
// doctest case that merely returned early would print `assertions: 0` and
// `Status: SUCCESS!`, which is the trap this repository has already hit twice
// (issue #463).
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

// The smallest request that still produces a denoise window. A routing
// assertion needs a window, not a song — and the reduced request is what makes
// the reachability mutation affordable, because with the engine's call deleted
// the run takes the HOST arm and a host arm at full duration is the thirty-hour
// failure this row exists to prevent.
constexpr double kDurationSeconds = 0.24;
constexpr int32_t kSteps = 2;
constexpr int64_t kSeed = 7;

// A gate that CANNOT RUN must never report success. See the file header.
[[noreturn]] void SkipGate(const std::string& why) {
  std::fprintf(stderr,
               "\n[SKIP] *** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "[SKIP] test_minimax_music3_device_arm_real: %s\n\n",
               why.c_str());
  std::fflush(stderr);
  std::exit(77);
}

// The checkpoint SET root. `VLLM_CPP_MUSIC3_CHECKPOINT` names it directly;
// `CHECKPOINT_ROOT` names the shared store the whole fleet reads, and the
// family's directory inside it. Same resolution
// `tests/parity/test_minimax_music3_e2e_real.cpp` uses, deliberately, so a
// leased job that can run one can run the other with no second variable.
std::string CheckpointRoot() {
  if (const char* direct = std::getenv("VLLM_CPP_MUSIC3_CHECKPOINT")) {
    if (direct[0] != '\0') return direct;
  }
  if (const char* root = std::getenv("CHECKPOINT_ROOT")) {
    if (root[0] != '\0') return (fs::path(root) / "minimax-music3").string();
  }
  return std::string();
}

// Arms BOTH profile flags and restores them. The outer one turns the table on;
// the inner one turns on the intra-forward spans, which is what `dit.pack`
// needs. The spans insert a `Backend::Synchronize` at every bracket inside the
// device forward and therefore perturb timing — this gate makes no timing claim
// and pays nothing for that, and it buys the one instrument a mislabelled
// bucket cannot fake.
struct ArmedProfile {
  ArmedProfile() : profile_(m3profile::EnabledFlag()), spans_(m3profile::DitSpansFlag()) {
    m3profile::EnabledFlag() = true;
    m3profile::DitSpansFlag() = true;
  }
  ~ArmedProfile() {
    m3profile::EnabledFlag() = profile_;
    m3profile::DitSpansFlag() = spans_;
  }
  bool profile_;
  bool spans_;
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
  std::fprintf(stderr, "\n[music3 device-arm] the buckets this run produced:\n");
  for (const m3profile::Bucket& bucket : m3profile::Buckets()) {
    std::fprintf(stderr, "[music3 device-arm]   %-28s calls=%lld seconds=%.3f%s\n",
                 bucket.name.c_str(), static_cast<long long>(bucket.calls), bucket.seconds,
                 bucket.span ? " (span)" : "");
  }
  std::fflush(stderr);
}

}  // namespace

TEST_CASE("music3 device arm: the ENGINE'S OWN CALL puts the DiT on the accelerator"
          " (GPU + checkpoint)") {
  // ── PRECONDITION 1: an accelerator, resolved exactly the way the engine
  //    resolves it. Not `#ifdef VLLM_CPP_CUDA` and not a backend probe of our
  //    own: `SpeechEngineDeviceType` is the function `Music3SpeechEngine`'s
  //    constructor calls, so asking it is asking the same question the
  //    production path asks, including the partial-backend refusal.
  try {
    const vt::DeviceType resolved = vllm::multimodal::SpeechEngineDeviceType(1, kFamily);
    std::fprintf(stderr, "[music3 device-arm] device 1 resolves to '%s'\n",
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
  std::fprintf(stderr, "[music3 device-arm] checkpoint set: %s\n", root.c_str());

  const ArmedProfile armed;

  // ── THE PRODUCTION ENTRY POINT, both halves of it ──────────────────────────
  vllm_speech_model_params model = vllm_speech_model_params_default();
  model.path = root.c_str();
  model.family = kFamily;
  // 1, not 0. This is the whole switch: it is what resolves `queue_` to the
  // platform's device, and it is the only thing that makes the engine's call to
  // `Music3SelectDitArm` return an engaged arm. There is no separate flag and no
  // environment variable, because a capability behind an option nothing turns on
  // is the shape `.agents/reachability.md` calls dead.
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

  // ── THE TWO QUANTITIES THE COUNTS ARE BUILT FROM, taken from the run ───────
  const m3profile::Bucket* steps_bucket = FindBucket("request.steps");
  const m3profile::Bucket* windows_bucket = FindBucket("denoise.windows");
  REQUIRE_MESSAGE(steps_bucket != nullptr,
                  "no request.steps counter: the profile table was not armed, so every "
                  "assertion below would have read absence as a verdict");
  REQUIRE_MESSAGE(windows_bucket != nullptr, "no denoise.windows counter");
  const int64_t steps = steps_bucket->calls;
  const int64_t windows = windows_bucket->calls;
  CHECK_MESSAGE(steps == kSteps, "the engine resolved " << steps << " steps, asked for " << kSteps);
  REQUIRE_MESSAGE(windows >= 1, "the denoise produced " << windows << " windows, so the loop this "
                                                        << "gate believes it drove never ran");
  const int64_t expected_brackets = steps * windows;
  const int64_t expected_forwards = 2 * expected_brackets;
  MESSAGE("steps=" << steps << " windows=" << windows << " => expected denoise.dit_device calls="
                   << expected_brackets << ", dit.pack calls=" << expected_forwards);

  // ── #1131's RESIDUAL, asserted: the engine CALLED the selector and the
  //    selector STAGED. This bucket is emitted from inside `Music3SelectDitArm`,
  //    after its CPU early return, so it is present if and only if the engine's
  //    own call site ran and took the device branch. Deleting that call site is
  //    the reachability mutation `.agents/reachability.md` prescribes, and this
  //    REQUIRE is what it reds.
  const m3profile::Bucket* staging = FindBucket("acoustic.dit_staging");
  REQUIRE_MESSAGE(staging != nullptr,
                  "no acoustic.dit_staging bucket: the engine never called "
                  "Music3SelectDitArm, or the selector declined the queue. The DiT ran on "
                  "the HOST and the song would be correct and hours late");
  CHECK_MESSAGE(staging->calls == 1,
                "the DiT staged " << staging->calls << " times; it is staged ONCE per request, "
                                                       "outside every loop");

  // ── which branch the production denoise loop SELECTED ──────────────────────
  const m3profile::Bucket* device_bucket = FindBucket("denoise.dit_device");
  REQUIRE_MESSAGE(device_bucket != nullptr,
                  "no denoise.dit_device bucket: the production denoise loop did NOT take "
                  "the device arm");
  CHECK_MESSAGE(device_bucket->calls == expected_brackets,
                "denoise.dit_device bracketed " << device_bucket->calls << " times, expected "
                                                << expected_brackets << " = steps x windows");
  CHECK_MESSAGE(FindBucket("denoise.dit_host") == nullptr,
                "the run ALSO emitted denoise.dit_host, so some window fell back to the host "
                "arm without saying so");

  // ── and that the device forward's BODY ran ─────────────────────────────────
  const m3profile::Bucket* pack = FindBucket("dit.pack");
  REQUIRE_MESSAGE(pack != nullptr,
                  "no dit.pack span: the denoise loop labelled its bracket 'device' and "
                  "nothing inside DitForwardDevice executed");
  CHECK_MESSAGE(pack->calls == expected_forwards,
                "dit.pack ran " << pack->calls << " times, expected " << expected_forwards
                                << " = 2 x steps x windows (one bracket spans both CFG "
                                   "branches)");

  // ── A CONTROL, NOT A GATE. The two arms agree by design, so no property of
  //    this waveform can say which one ran; what it CAN catch is an arm that
  //    threw halfway and left a plausible buffer behind. No tolerance here is a
  //    claim about correctness — that is `test_minimax_music3_acoustic_real.cpp`
  //    and `test_minimax_music3_e2e_real.cpp`.
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
                        << " samples are non-zero: the device arm returned near-silence");
  CHECK_MESSAGE(peak > 1e-4, "waveform peak is " << peak << ", which is silence");
  MESSAGE("waveform: " << result.n_samples << " samples x " << result.channels << " ch at "
                       << result.sample_rate << " Hz, peak " << peak);

  vllm_speech_result_free(&result);
  vllm_speech_engine_free(engine);
}
