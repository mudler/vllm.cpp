// LTX-2.5 phase L7 gate — the family behind `vllm::multimodal::VideoEngine`,
// and the driving loop that turns the L2-L6 bricks into artifacts.
//
// Spec: .agents/specs/ltx-2-5.md §6 (L7). Issue #435.
//
// WHAT THIS CAN AND CANNOT SHOW. It runs the REAL path over a reduced-dimension
// checkpoint set written in the SHIPPED FILE FORMAT: the ComfyUI-prefixed FP8
// DiT through phase L6's quantized loader, the two VAEs through their own
// embedded `__metadata__["config"]`, the recipe table, the denoise loop, both
// decoders, the vocoder, and the artifact writers. So it gates COMPOSITION.
//
// It is NOT a render-quality result and nothing here should be read as one. The
// weights are a deterministic stream, so the frames are what those weights
// produce and no more — MiniMax-H3's fp4-resident e2e RAN and emitted a valid
// mp4 of a non-scene patch grid, which is exactly the failure a structural gate
// cannot see. What is asserted is what a structural gate CAN see: the geometry
// the pipeline resolved, that every value is finite, that the artifacts exist at
// the sizes the result reports, and that every refusal fires by name.
#include "vllm/multimodal/ltx2_video.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

#include <sys/stat.h>
#include <unistd.h>

#include "ltx2_video_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — the seam the engine asks
#include "vllm.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vllm/multimodal/video_engine.h"
#include "vllm/multimodal/render_phase_log.h"

namespace {

struct Workspace {
  std::string root, fixture;
  ltx2_fixture::Paths paths;
  Workspace() {
    static int counter = 0;
    root = "/tmp/vllm_ltx2_video_" + std::to_string(::getpid()) + "_" + std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
    fixture = root + "/fixture";
    paths = ltx2_fixture::WriteFixture(fixture);
  }
  // `VLLM_KEEP_TEST_ARTIFACTS=1` leaves the workspace on disk. The e2e evidence
  // this phase owes is a FRAME SET and a WAV somebody can open, and a test that
  // deletes them can only ever report numbers about files nobody saw.
  ~Workspace() {
    const char* keep = std::getenv("VLLM_KEEP_TEST_ARTIFACTS");
    if (keep != nullptr && keep[0] == '1') {
      std::printf("[ltx2] kept workspace: %s\n", root.c_str());
      return;
    }
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

// The class a fixture load must DECLARE for a given pipeline kind — row
// LTX25-CHECKPOINT-CLASS, issue #1137.
//
// RESOLVED FROM THE RECIPE TABLE rather than listed here on purpose. This is
// fixture SETUP and not an assertion: the table's own gate is
// `test_ltx2_pipeline`, and a per-case list here would be a second copy of it
// that could disagree without anything saying so.
//
// `kFullOrDistilled` declares `distilled`, which is the arm that needs no
// adapter. `kUnstated` declares nothing, which is what `dmd2` accepts.
std::string FixtureCheckpointClass(const std::string& kind) {
  // The fixture DiT declares `model_version` 2.5.0, so 2.5 is tried first; the
  // older keys are there for the cases that override the version extra.
  for (const char* version : {"2.5", "2.4", "2.3", "2"}) {
    vllm::Ltx2PipelineRecipe recipe;
    try {
      recipe = vllm::ResolveLtx2PipelineRecipe(kind, version);
    } catch (const std::exception&) {
      continue;
    }
    switch (recipe.checkpoint_class) {
      case vllm::Ltx2RequiredCheckpointClass::kFull:
        return "full";
      case vllm::Ltx2RequiredCheckpointClass::kDistilled:
        return "distilled";
      case vllm::Ltx2RequiredCheckpointClass::kKeyframeSlotSft:
        return "keyframe_slot_sft";
      case vllm::Ltx2RequiredCheckpointClass::kFullOrDistilled:
        return "distilled";
      case vllm::Ltx2RequiredCheckpointClass::kUnstated:
        return "";
    }
  }
  // A kind no version resolves is the caller's own refusal case; declare nothing
  // and let the recipe table refuse it by name, which is what those cases assert.
  return "";
}

vllm::multimodal::VideoModelParams FixtureParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.video_vae_path = paths.video_vae;
  mp.audio_vae_path = paths.audio_vae;
  mp.prompt_embeds_path = paths.video_embeds;
  mp.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = paths.audio_embeds;
  // The NEGATIVE half (row LTX25-GUIDED-VIDEO, #1092), on the SHARED helper
  // rather than only on the cases that gate guidance. A `one_stage` engine's own
  // recipe resolves `cfg_scale = 3.0`, so upstream's unconditional forward is
  // this fixture's DEFAULT configuration, and an engine that could not run it
  // would make every one_stage case here a refusal. `distilled_two_stage` — the
  // default kind, and what most cases below load — resolves `cfg_scale = 1.0`
  // and never reads these.
  mp.extras[vllm::multimodal::kLtx2NegativePromptEmbedsExtra] = paths.negative_video_embeds;
  mp.extras[vllm::multimodal::kLtx2NegativeAudioPromptEmbedsExtra] = paths.negative_audio_embeds;
  mp.device = 0;
  // THE CHECKPOINT CLASS (row LTX25-CHECKPOINT-CLASS, #1137). Every kind this
  // family resolves except `dmd2` now refuses a load that does not declare which
  // class of transformer it was handed, because nothing in a checkpoint header
  // separates a distilled LTX-2.5 DiT from a full one. The default kind here is
  // `distilled_two_stage`, whose upstream row is `Distilled only`.
  //
  // DERIVED, NOT LISTED, by `FixtureCheckpointClass` above: a fixture that
  // hard-coded a class per case would be a second copy of the recipe table, and
  // the two would drift.
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("distilled_two_stage");
  return mp;
}

// The ONE guider field a `one_stage` render on this fixture has to override, and
// the reason is the fixture rather than the row: the reduced DiT has TWO blocks
// (ltx2_video_fixture.h `ReducedDitParams`), so the params table's own
// `stg_blocks = [28]` (utils/constants.py:83-88) names a block this checkpoint
// does not have. Left alone, the perturbed forward would perturb nothing and
// `stg_scale * (cond - perturbed)` would be exactly zero — which the engine now
// refuses by name rather than rendering, so this is what turns that refusal into
// a render. Named explicitly rather than by setting the STG scale to 0, because
// turning the perturbed pass OFF is a different configuration and would vacate
// every assertion about it.
//
// Nothing else is overridden: `cfg_scale`, `rescale_scale` and `modality_scale`
// stay at the recipe's own 3.0 / 0.7 / 3.0, which is what makes the guided cases
// below sit on the DEFAULT arm.
void OneStageFixtureGuidance(vllm::multimodal::VideoGenParams* gen) {
  gen->extras[vllm::multimodal::kLtx2VideoStgBlocksExtra] = "1";
  gen->extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "1";
}

// The smallest request the fixture's own scale factors admit: (8, 32, 32) means
// 64x64 pixels is a 2x2 latent and 9 frames is 2 latent frames.
vllm::multimodal::VideoGenParams FixtureGen(const std::string& out_dir) {
  vllm::multimodal::VideoGenParams gen;
  gen.num_frames = 9;
  gen.height = 64;
  gen.width = 64;
  gen.has_seed = true;
  gen.seed = 7;
  gen.output_dir = out_dir;
  return gen;
}

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// A prompt-embeds file as floats, and back. The register cases need to change
// ONE row of a supplied stream and leave every other byte alone, which the
// fixture's seeded writer cannot express.
std::vector<float> ReadFloats(const std::string& path) {
  const std::string bytes = ReadAll(path);
  REQUIRE(bytes.size() % sizeof(float) == 0);
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

void WriteBytes(const std::string& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  REQUIRE_MESSAGE(out.good(), "cannot write ", path);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteFloats(const std::string& path, const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  REQUIRE_MESSAGE(out.good(), "cannot write ", path);
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

bool Registered(const std::string& family) {
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  return std::find(all.begin(), all.end(), family) != all.end();
}

// A binary PPM's declared width/height, so the test reads the FILE rather than
// trusting the result struct's own numbers about it.
void ParsePpmHeader(const std::string& bytes, int* width, int* height, size_t* payload_at) {
  REQUIRE(bytes.size() > 2);
  REQUIRE(bytes.compare(0, 2, "P6") == 0);
  size_t at = 2;
  int values[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    while (at < bytes.size() && (std::isspace(static_cast<unsigned char>(bytes[at])) != 0)) ++at;
    int v = 0;
    while (at < bytes.size() && (std::isdigit(static_cast<unsigned char>(bytes[at])) != 0)) {
      v = v * 10 + (bytes[at] - '0');
      ++at;
    }
    values[i] = v;
  }
  ++at;  // the single whitespace byte before the payload
  *width = values[0];
  *height = values[1];
  *payload_at = at;
}

}  // namespace

// ─── registration and detection ─────────────────────────────────────────────

TEST_CASE("ltx2 video: the family self-registers under its stable name") {
  CHECK(Registered(vllm::multimodal::kLtx2VideoFamily));
  CHECK(Registered("minimax-h3"));
  const std::vector<std::string> all = vllm::multimodal::RegisteredVideoFamilies();
  CHECK(std::is_sorted(all.begin(), all.end()));
  CHECK(std::adjacent_find(all.begin(), all.end()) == all.end());
}

TEST_CASE("ltx2 video: detection resolves the checkpoint by what it HOLDS") {
  Workspace ws;
  const std::vector<std::string> got =
      vllm::multimodal::DetectVideoFamilies(FixtureParams(ws.paths));
  REQUIRE(got.size() == 1);
  CHECK(got[0] == vllm::multimodal::kLtx2VideoFamily);

  // The VAE is a perfectly good safetensors file carrying no DiT signature. If
  // it claimed a family, "there is only one family so it must be that one" would
  // be back by another door.
  vllm::multimodal::VideoModelParams not_a_dit = FixtureParams(ws.paths);
  not_a_dit.dit_path = ws.paths.video_vae;
  CHECK(vllm::multimodal::DetectVideoFamilies(not_a_dit).empty());

  // And it must be found WITHOUT the ComfyUI prefix, because which prefix a
  // re-export kept is the repackager's choice and says nothing about the model.
  SUBCASE("a de-prefixed re-export is still detected") {
    const std::string stripped = ws.root + "/stripped.safetensors";
    // Rewrite the header with the prefix removed, payload untouched.
    const std::string bytes = ReadAll(ws.paths.dit);
    uint64_t n = 0;
    std::memcpy(&n, bytes.data(), sizeof(n));
    std::string header = bytes.substr(sizeof(n), n);
    const std::string prefix = vllm::kLtx2DitCheckpointPrefix;
    for (size_t at = header.find(prefix); at != std::string::npos;
         at = header.find(prefix, at)) {
      header.erase(at, prefix.size());
    }
    header.append(prefix.size() * 0, ' ');
    std::string rebuilt;
    const uint64_t m = header.size();
    rebuilt.append(reinterpret_cast<const char*>(&m), sizeof(m));
    rebuilt += header;
    rebuilt += bytes.substr(sizeof(n) + n);
    std::ofstream out(stripped, std::ios::binary);
    out.write(rebuilt.data(), static_cast<std::streamsize>(rebuilt.size()));
    out.close();

    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = stripped;
    const std::vector<std::string> claimed = vllm::multimodal::DetectVideoFamilies(mp);
    REQUIRE(claimed.size() == 1);
    CHECK(claimed[0] == vllm::multimodal::kLtx2VideoFamily);
  }
}

// ─── the composition gate ───────────────────────────────────────────────────

TEST_CASE("ltx2 video: an auto-detected load renders frames, a WAV and a mux argv") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  // Stop after phase 0: the recipe's second phase needs the latent spatial
  // upsampler, and running without one is a REFUSAL (gated in its own case
  // below), not a silently shorter render.
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == vllm::multimodal::kLtx2VideoFamily);
  // The text tower is OWED, so this must read false rather than "we have an
  // encoder_path field".
  CHECK(!engine->has_encoder());
  CHECK(engine->has_prompt_embeds());

  const std::string out_dir = ws.root + "/out";
  const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));

  // Phase 0 of the distilled two-stage recipe runs at HALF the requested size
  // (spatial_downscale 2), which is upstream's own stage-1 geometry — so a
  // 64x64 request decodes a 32x32 clip here. Asserting the halved size is the
  // point: a port that ignored `spatial_downscale` would return 64x64 and look
  // more correct while sampling the wrong stage.
  CHECK(result.frame_count == 9);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.fps == 24);
  CHECK(result.sample_rate == 48000);
  CHECK(result.frame_dir == out_dir);
  CHECK(result.mux_output_path == out_dir + "/video.mp4");

  // Every frame exists, at the size the RESULT claims, read off the file.
  //
  // AND CARRIES MORE THAN ONE BYTE VALUE. That second assertion is the one that
  // earns its place: the PPM writer clamps, and `std::max(-1.0, NaN)` returns
  // -1.0, so an ALL-NaN decode serializes as a perfectly well-formed, uniformly
  // black frame of exactly the right size. Every size/geometry check above would
  // pass over it. This is a floor, not a quality claim — it separates "the
  // pipeline produced something" from "the pipeline produced NaN", and nothing
  // more.
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    const std::string bytes = ReadAll(out_dir + name);
    int w = 0, h = 0;
    size_t at = 0;
    ParsePpmHeader(bytes, &w, &h, &at);
    INFO("frame ", f);
    CHECK(w == static_cast<int>(result.width));
    CHECK(h == static_cast<int>(result.height));
    CHECK(bytes.size() == at + static_cast<size_t>(w) * h * 3);
    size_t distinct = 0;
    bool seen[256] = {false};
    for (size_t i = at; i < bytes.size(); ++i) {
      const unsigned char v = static_cast<unsigned char>(bytes[i]);
      if (!seen[v]) {
        seen[v] = true;
        ++distinct;
      }
    }
    CHECK_MESSAGE(distinct > 1, "frame ", f, " is a single flat value (",
                  static_cast<int>(static_cast<unsigned char>(bytes[at])),
                  "), which is what an all-NaN decode serializes as");
  }
  // One more frame than the clip must NOT exist.
  {
    char extra[64];
    std::snprintf(extra, sizeof(extra), "/frame_%06lld.ppm",
                  static_cast<long long>(result.frame_count));
    std::ifstream beyond(out_dir + extra, std::ios::binary);
    CHECK_MESSAGE(!beyond.good(), "the render produced more frames than it reported");
  }

  // The WAV: RIFF, 16-bit PCM, stereo, at the vocoder's OUTPUT rate, and long
  // enough to be the clip's own duration rather than a stub. 9 frames at 24 fps
  // is 0.375 s; the causal audio decoder trims 3 mel frames, so the waveform is
  // shorter than the nominal duration by a known amount, and the assertion is a
  // BAND rather than an equality for exactly that reason.
  const std::string wav = ReadAll(result.audio_path);
  REQUIRE(wav.size() > 44);
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  CHECK(wav.compare(8, 4, "WAVE") == 0);
  uint16_t channels = 0;
  uint32_t rate = 0;
  std::memcpy(&channels, wav.data() + 22, sizeof(channels));
  std::memcpy(&rate, wav.data() + 24, sizeof(rate));
  CHECK(channels == 2);
  CHECK(rate == 48000u);
  const double seconds =
      static_cast<double>(wav.size() - 44) / (2.0 * 2.0 * static_cast<double>(rate));
  INFO("wav seconds = " << seconds);
  CHECK(seconds > 0.2);
  CHECK(seconds < 0.4);
  // The same floor as the frames': a NaN waveform serializes as digital silence
  // through the WAV writer's own cast, and silence is a valid RIFF file of
  // exactly the right length.
  int64_t nonzero = 0;
  for (size_t i = 44; i + 1 < wav.size(); i += 2) {
    int16_t sample = 0;
    std::memcpy(&sample, wav.data() + i, sizeof(sample));
    if (sample != 0) ++nonzero;
  }
  CHECK_MESSAGE(nonzero > 0, "the waveform is digital silence, which is what a NaN decode writes");

  // The mux argv the CALLER execs. The library spawns nothing, so what is gated
  // is that the argv names the artifacts that were actually written.
  REQUIRE(!result.mux_argv.empty());
  CHECK(result.mux_argv[0] == "ffmpeg");
  std::string joined;
  for (const std::string& a : result.mux_argv) joined += a + " ";
  CHECK(joined.find(out_dir + "/frame_%06d.ppm") != std::string::npos);
  CHECK(joined.find(result.audio_path) != std::string::npos);
  CHECK(joined.find(result.mux_output_path) != std::string::npos);
}

// ─── the UNQUANTIZED DiT, reached from the production entry point ───────────
//
// AGENTS.md §"Nothing lands dead": the bf16 arm is not done because
// `Ltx2LoadDitFromSafetensors` accepts the file; it is done because
// `LoadVideoEngine` — what `include/vllm.h`'s `vllm_video_engine_load`, the
// server and `ltx2-gen` all reach — loads it on its DEFAULT configuration and
// renders through it. A loader unit test constructing the checkpoint by hand
// would prove the branch works and NOT that anything reaches it.
//
// The value claim is the loader suite's ("the UNQUANTIZED bf16 DiT materializes
// onto the same contract", bit-exact over every contract weight). This case owns
// the reachability half, and its floor is the same as the FP8 render's: frames
// at the size the result claims, carrying more than one byte value, plus a
// waveform that is not digital silence.
TEST_CASE("ltx2 video: an UNQUANTIZED bf16 DiT loads through the ENGINE and renders") {
  Workspace ws;
  // Rewrite the workspace's DiT with no quantized weight and no scale sidecar,
  // which is the shape the shipped dev transformer is in. Everything else about
  // the fixture — the VAEs, the upsampler, the embeds — is untouched, so the
  // ONLY difference from the FP8 case above is the DiT's dtype.
  ltx2_fixture::ReducedDitOptions bf16;
  bf16.unquantized = true;
  ltx2_fixture::WriteReducedDit(ltx2_fixture::ReducedDitParams(), ws.paths.dit, bf16);

  // The file really is unquantized, asserted before the engine sees it: a
  // fixture flag that silently did nothing would make everything below a second
  // run of the FP8 case.
  {
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(ws.paths.dit);
    int64_t quantized = 0, bf16_tensors = 0, sidecars = 0;
    for (const std::string& name : file.Names()) {
      const std::string& dtype = file.Get(name).dtype;
      if (dtype == "U8" || dtype == "F8_E4M3") ++quantized;
      if (dtype == "BF16") ++bf16_tensors;
      if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_scale") == 0) ++sidecars;
    }
    INFO("quantized = " << quantized << " bf16 = " << bf16_tensors
                        << " sidecars = " << sidecars);
    CHECK(quantized == 0);
    CHECK(sidecars == 0);
    CHECK(bf16_tensors > 0);
  }

  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  CHECK(engine->family() == vllm::multimodal::kLtx2VideoFamily);

  // The geometry the engine RESOLVED from the unquantized file. A bf16 header
  // carries no packed width to double, so a loader that still halved or doubled
  // one would land here rather than in the pixels.
  const auto* ltx2 = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  const vllm::Ltx2DitParams reference = ltx2_fixture::ReducedDitParams();
  CHECK(ltx2->dit_params().num_layers == reference.num_layers);
  CHECK(ltx2->dit_params().inner_dim() == reference.inner_dim());
  CHECK(ltx2->dit_params().audio_inner_dim() == reference.audio_inner_dim());
  CHECK(ltx2->dit_params().in_channels == reference.in_channels);
  CHECK(ltx2->dit_params().audio_in_channels == reference.audio_in_channels);

  const std::string out_dir = ws.root + "/out_bf16";
  const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));
  CHECK(result.frame_count == 9);
  CHECK(result.width == 32);
  CHECK(result.height == 32);

  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    const std::string bytes = ReadAll(out_dir + name);
    int w = 0, h = 0;
    size_t at = 0;
    ParsePpmHeader(bytes, &w, &h, &at);
    INFO("frame ", f);
    CHECK(w == static_cast<int>(result.width));
    CHECK(h == static_cast<int>(result.height));
    CHECK(bytes.size() == at + static_cast<size_t>(w) * h * 3);
    size_t distinct = 0;
    bool seen[256] = {false};
    for (size_t i = at; i < bytes.size(); ++i) {
      const unsigned char v = static_cast<unsigned char>(bytes[i]);
      if (!seen[v]) {
        seen[v] = true;
        ++distinct;
      }
    }
    CHECK_MESSAGE(distinct > 1, "frame ", f, " is a single flat value, which is what an "
                                "all-NaN decode serializes as");
  }

  const std::string wav = ReadAll(result.audio_path);
  REQUIRE(wav.size() > 44);
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  int64_t nonzero = 0;
  for (size_t i = 44; i + 1 < wav.size(); i += 2) {
    int16_t sample = 0;
    std::memcpy(&sample, wav.data() + i, sizeof(sample));
    if (sample != 0) ++nonzero;
  }
  CHECK_MESSAGE(nonzero > 0, "the waveform is digital silence, which is what a NaN decode writes");
}

TEST_CASE("ltx2 video: a MULTI-CHUNK render numbers its frames globally, and clears the last one") {
  // The fixture above is 9 frames — ONE temporal chunk — so the render path's
  // `chunk.first_frame + f` (ltx2_video.cpp, the streaming sink) was never driven
  // through the PPM writer with more than one chunk. Per-chunk numbering would
  // restart at frame_000000 for the second chunk, overwrite the first chunk's
  // files and leave the clip's tail missing, and every assertion in the 9-frame
  // case would still pass.
  //
  // 81 frames is the smallest request that chunks, and that is not a coincidence:
  // `latent_t = (81 - 1) / 8 + 1 = 11` against the AUTO layout's 80-frame / 10
  // latent-frame temporal tile. It is asserted below rather than assumed, because
  // a test that silently stopped chunking would be green and vacuous.
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);

  // The fixture's factors, as documented at `FixtureGen`: (8, 32, 32). Phase 0
  // halves the spatial request, so 64x64 decodes at 32x32 — one latent cell each
  // way — and only the temporal axis can chunk here.
  const vllm::Ltx2ScaleFactors factors{8, 32, 32};
  const vllm::Ltx2TileSizeConfig layout = vllm::Ltx2AutoTileSizeConfig(32, 32, factors);
  const int64_t latent_t = (81 - 1) / factors.time + 1;
  CHECK(latent_t == 11);
  const std::vector<vllm::Ltx2Tile> tiles =
      vllm::Ltx2CreateTiles(latent_t, 1, 1, layout, factors);
  const size_t groups = vllm::Ltx2GroupTilesByTemporalSlice(tiles).size();
  REQUIRE_MESSAGE(groups > 1u, "this request no longer chunks; the case below proves nothing");

  const std::string out_dir = ws.root + "/multichunk";
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.num_frames = 81;
  const vllm::multimodal::VideoResult result = engine->Generate(gen);
  REQUIRE(result.frame_count == 81);

  // EVERY global index exists exactly once, and the one past the end does not.
  // Per-chunk numbering fails here on both counts at the same time.
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    std::ifstream in(out_dir + name, std::ios::binary);
    CHECK_MESSAGE(in.good(), "frame ", f, " of a ", result.frame_count,
                  "-frame streamed render is missing");
  }
  {
    std::ifstream beyond(out_dir + "/frame_000081.ppm", std::ios::binary);
    CHECK(!beyond.good());
  }
  // ...and the last frame is not a copy of the first, which is what a writer that
  // reused chunk-local indices would leave behind after the overwrite.
  CHECK(ReadAll(out_dir + "/frame_000080.ppm") != ReadAll(out_dir + "/frame_000000.ppm"));

  // THE STALE TAIL. Rendering a SHORTER clip into the same directory must not
  // leave the longer render's frames behind: `mux.frame_pattern` is
  // `frame_%06d.ppm` with no count, so ffmpeg would mux 72 frames past the end of
  // a clip that reported 9. Pre-existing, and widened by streaming — a chunk that
  // throws mid-render leaves a partial clip on disk too.
  const vllm::multimodal::VideoResult shorter = engine->Generate(FixtureGen(out_dir));
  REQUIRE(shorter.frame_count == 9);
  for (int64_t f = 9; f < 81; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    std::ifstream stale(out_dir + name, std::ios::binary);
    CHECK_MESSAGE(!stale.good(), "frame ", f, " survived a shorter re-render into the same "
                                              "directory and would be muxed past its end");
  }
  for (int64_t f = 0; f < 9; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    std::ifstream in(out_dir + name, std::ios::binary);
    CHECK_MESSAGE(in.good(), "the cleanup removed frame ", f, ", which the new render owns");
  }
}

TEST_CASE("ltx2 video: the second phase upsamples, and refuses when it cannot") {
  Workspace ws;
  SUBCASE("without an upsampler the phase is refused BY NAME, not skipped") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/no_ups"));
      FAIL("a missing spatial upsampler must be refused, not skipped");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("upsampler_path") != std::string::npos);
      CHECK(msg.find("refine") != std::string::npos);
    }
  }
  SUBCASE("a TEMPORAL upsampler checkpoint is refused BY NAME, not shape-mismatched") {
    // The temporal x2 arm is ported and gated (test_ltx2_pipeline.cpp, "ltx2 the
    // latent temporal upsampler reproduces upstream") but NOTHING drives it. It
    // shares the class name and the `upsampler.0.*` tensor names with the
    // spatial arm, so this checkpoint loads and runs; what it returns is
    // [c, 2f-1, h, w] where the phase needs [c, f, 2h, 2w]. Without the guard
    // the caller gets a shape mismatch and has to work out that they handed over
    // the wrong file.
    vllm::Ltx2UpsamplerConfig temporal =
        ltx2_fixture::ReducedUpsamplerConfig(ltx2_fixture::ReducedDitParams().in_channels);
    temporal.spatial_upsample = false;
    temporal.temporal_upsample = true;
    const std::string path = ws.root + "/temporal_upsampler.safetensors";
    ltx2_fixture::WriteReducedUpsampler(temporal, path);

    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = path;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/temporal_ups"));
      FAIL("a temporal upsampler checkpoint must be refused, not run for this phase");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("temporal_upsample=true") != std::string::npos);
      CHECK(msg.find("SPATIAL") != std::string::npos);
      // The message must not be the generic shape complaint — that is the
      // failure mode this guard exists to replace.
      CHECK(msg.find("the upsampled latent is") == std::string::npos);
    }
  }
  // THE ARM THE GUARD ABOVE WAS SHADOWING, and the reason this subcase exists at
  // all. `if (im.upsampler_cfg.temporal_upsample)` is satisfied by a BOTH-flags
  // checkpoint as well as a temporal-only one, so a genuine SPATIOTEMPORAL
  // checkpoint was told it is the temporal x2 upsampler and pointed at the spatial
  // one. Wrong on both counts: it is neither, it is the third arm, and the ledger
  // refusal that names it (`ltx2_upsampler.cpp:465`) sat behind a guard that could
  // not be reached from a request.
  //
  // The defect is an IMPLICATION between two guards over one variable, which no
  // fixture could see because nothing drove a both-flags config through
  // `LoadVideoEngine` — a review could prove the ledger refusal unmutated but not
  // separate "unreachable" from "untested". This subcase closes that: it is the
  // both-flags checkpoint, driven through the product path, asserting the caller
  // is told which arm they actually supplied.
  SUBCASE("a SPATIOTEMPORAL upsampler checkpoint is refused as SPATIOTEMPORAL, not as temporal") {
    vllm::Ltx2UpsamplerConfig spatiotemporal =
        ltx2_fixture::ReducedUpsamplerConfig(ltx2_fixture::ReducedDitParams().in_channels);
    spatiotemporal.spatial_upsample = true;
    spatiotemporal.temporal_upsample = true;
    const std::string path = ws.root + "/spatiotemporal_upsampler.safetensors";
    ltx2_fixture::WriteReducedUpsampler(spatiotemporal, path);

    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = path;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(FixtureGen(ws.root + "/spatiotemporal_ups"));
      FAIL("a spatiotemporal upsampler checkpoint must be refused by name");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      // The arm they ACTUALLY supplied, which is the whole repair.
      CHECK(msg.find("SPATIOTEMPORAL") != std::string::npos);
      // ...and NOT the temporal-only diagnosis, which is what the shadowing guard
      // produced. Asserted on the sentence that only that guard emits, because
      // both messages legitimately contain the word `temporal`.
      CHECK(msg.find("it is the TEMPORAL x2 upsampler") == std::string::npos);
      CHECK(msg.find("Supply the spatial upsampler") == std::string::npos);
      // Not a shape complaint either: the refusal has to land before any weight
      // is touched, which is what the ledger arm promises.
      CHECK(msg.find("the upsampled latent is") == std::string::npos);
    }
  }
  SUBCASE("with one, the render lands at the FULL requested size") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    const std::string out_dir = ws.root + "/two_stage";
    const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));
    CHECK(result.width == 64);
    CHECK(result.height == 64);
    CHECK(result.frame_count == 9);
  }
}

// A request whose size does not divide the latent grid used to RENDER at a size
// nobody asked for (#919). `ltx2_video.cpp` integer-divides the request into
// `Ltx2VideoLatentShape` and only ever checked the LOWER bound, so the floor was
// silent and the call returned success.
//
// MEASURED on this fixture before the guard existed, which is why these sizes and
// not the obvious ones: a two-stage request of width 80 rendered 64x64, and a
// one-stage request of width 100 rendered 96x64. Width 96 on the two-stage arm
// does NOT reach that state — stage 1 floors 48 to one latent cell while stage 2
// needs three — so the upsampler's shape check catches it and reports "the
// upsampled latent is 4x2x2x2 but phase 'refine' needs 4x2x2x3", a true statement
// about latents and no help at all to a caller who passed a width. The defect has
// two faces, a silent floor and an unreadable downstream throw, and one guard at
// the entry point closes both.
//
// Upstream raises instead, at the top of a pipeline's `__call__` and before any
// work is paid for: `assert_resolution` (ltx-pipelines utils/helpers.py:540-551)
// takes a divisor of 64 for a two-stage pipeline and 32 for a one-stage one,
// across NINE invocations including ti2vid_two_stages.py:184 and
// ti2vid_two_stages_hq.py:199. Nine, counted at the pin: a grep for the name
// returns 21 lines, which are 9 invocations + 1 definition + 10 imports + 1
// `__all__` string. Nor is the guard on every pipeline — 13 pipeline `__call__`s
// take a height and a width, and distilled_mgpu.py:143,
// ti2vid_two_stages_mgpu.py:163, ti2vid_two_stages_hq_mgpu.py:164 and
// hdr_ic_lora.py:352 do not call it.
//
// Those two divisors are NOT two constants. They are the VAE spatial factor (32,
// ltx_core/types.py:31-33) times the worst spatial downscale any phase applies —
// a two-stage pipeline runs stage 1 at `width // 2` (ti2vid_two_stages.py:226-228),
// so the request must survive being halved and still divide the grid. The last two
// subcases gate that DERIVATION rather than the numbers, by driving the same width
// 96 through both recipes and requiring opposite answers.
//
// EVERY AXIS ASSERTION HERE IS A PHRASE AND NOT A WORD, and that is the whole
// point of the spelling. The message carries the literal "(width x height)" label
// in every refusal it ever emits, so `msg.find("width")` and `msg.find("height")`
// are both satisfied by that constant no matter which axis the guard named — a
// mutation swapping the two names in `Ltx2AssertResolution` stayed green against
// needles spelt that way, and a height-80 request would have reported "the width
// is not" with nothing to see it.
TEST_CASE("ltx2 video: a size that does not divide the latent grid is REFUSED, per recipe") {
  Workspace ws;

  SUBCASE("a two-stage width that is not a multiple of 64 is refused BY VALUE") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/w80");
    gen.width = 80;
    try {
      (void)engine->Generate(gen);
      FAIL("width 80 rendered 64x64 before the guard; it must be refused, not floored");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      // The offending value, the divisor, and the axis. A message that says only
      // "bad resolution" leaves the caller to guess which of the two numbers they
      // passed is wrong and what a right one would be.
      CHECK(msg.find("80") != std::string::npos);
      CHECK(msg.find("64") != std::string::npos);
      // The PHRASE. Only the width offends here, so the message must say so and
      // must not name the height.
      CHECK(msg.find("the width is not") != std::string::npos);
      CHECK(msg.find("the height is not") == std::string::npos);
      // The suggested size, and that it is the NEAREST legal one rather than the
      // request echoed back. Without this the arithmetic is unmeasured: a mutation
      // replacing `(width / divisor) * divisor` with `width` left all fifteen
      // other assertions green, and a wrong suggestion here sends the caller
      // straight to another illegal size.
      CHECK(msg.find("Nearest legal size at or below the request: 64x64") != std::string::npos);
    }
  }

  SUBCASE("a two-stage height that is not a multiple of 64 is refused BY VALUE") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/h80");
    gen.height = 80;
    try {
      (void)engine->Generate(gen);
      FAIL("height 80 must be refused, not floored to 64");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("80") != std::string::npos);
      // The axis the guard NAMED, not the axis label the message always carries.
      // This is the assertion the swap mutation has to move: the width is 64 here
      // and legal, so a message that blames it is wrong.
      CHECK(msg.find("the height is not") != std::string::npos);
      CHECK(msg.find("the width is not") == std::string::npos);
      CHECK(msg.find("Nearest legal size at or below the request: 64x64") != std::string::npos);
    }
  }

  // BOTH axes off the grid. This branch of the axis phrase is executed by no
  // other subcase, and it is the one a caller passing a square off-grid size
  // reaches — the commonest shape of the mistake.
  SUBCASE("a two-stage request with BOTH axes off the grid names both") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/wh80");
    gen.width = 80;
    gen.height = 80;
    try {
      (void)engine->Generate(gen);
      FAIL("80x80 is off the grid on both axes and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("the width and height are not") != std::string::npos);
      CHECK(msg.find("Nearest legal size at or below the request: 64x64") != std::string::npos);
    }
  }

  // A SUB-DIVISOR axis, where the suggestion the refusal makes is the thing under
  // test. `(width / divisor) * divisor` is 0 for any width below the divisor, and
  // 0 is not a legal size: a caller who followed the old "Nearest legal size at or
  // below the request: 0x64" landed on the LOWER-bound refusal in the phase loop,
  // one illegal size handed out in place of another. No subcase reached this
  // branch, because every measured size in this case is above its divisor.
  SUBCASE("a two-stage width BELOW the divisor is not told to render at zero") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/w32");
    gen.width = 32;
    try {
      (void)engine->Generate(gen);
      FAIL("width 32 is below the two-stage divisor of 64 and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("the width is not") != std::string::npos);
      // What it must NOT say. `0x64` is the old suggestion, and `x0` catches the
      // mirrored case on the height axis.
      CHECK(msg.find("0x64") == std::string::npos);
      CHECK(msg.find("x0") == std::string::npos);
      // What it must say instead: that no legal size at or below the request
      // exists, and what the smallest legal one is.
      CHECK(msg.find("No legal size at or below the request exists") != std::string::npos);
      CHECK(msg.find("Smallest legal size: 64x64") != std::string::npos);
    }
  }

  // The same branch on the other recipe, so the smallest legal size is proven to
  // follow the DERIVED divisor rather than being a second hardcoded 64.
  SUBCASE("a one-stage width BELOW the divisor names 32 as the smallest legal size") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/os16");
    gen.width = 16;
    try {
      (void)engine->Generate(gen);
      FAIL("width 16 is below the one-stage divisor of 32 and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("the width is not") != std::string::npos);
      CHECK(msg.find("No legal size at or below the request exists") != std::string::npos);
      CHECK(msg.find("Smallest legal size: 32x32") != std::string::npos);
      CHECK(msg.find("x0") == std::string::npos);
    }
  }

  // The other face of the same defect: 96 reaches the upsampler's shape check
  // today. After the guard it is refused at the entry point, by the value the
  // caller actually passed.
  SUBCASE("a two-stage width of 96 is refused BY VALUE, not by latent shape") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/w96");
    gen.width = 96;
    try {
      (void)engine->Generate(gen);
      FAIL("96 is not a multiple of 64 and must be refused at the entry point");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("96") != std::string::npos);
      CHECK(msg.find("the width is not") != std::string::npos);
      CHECK(msg.find("the height is not") == std::string::npos);
      CHECK(msg.find("upsampled latent") == std::string::npos);
    }
  }

  SUBCASE("a one-stage width that is not a multiple of 32 is refused BY VALUE") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/os100");
    gen.width = 100;
    try {
      (void)engine->Generate(gen);
      FAIL("one-stage width 100 rendered 96x64 before the guard; it must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("100") != std::string::npos);
      CHECK(msg.find("32") != std::string::npos);
      CHECK(msg.find("the width is not") != std::string::npos);
      CHECK(msg.find("the height is not") == std::string::npos);
      // 96, not 64: the nearest legal size follows the recipe's own divisor.
      CHECK(msg.find("Nearest legal size at or below the request: 96x64") != std::string::npos);
    }
  }

  // The guard is not a blanket one. Without this, every subcase above is
  // satisfied by a check that refuses everything.
  SUBCASE("a multiple of 64 still renders on the two-stage recipe") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(ws.root + "/ok64"));
    CHECK(result.width == 64);
    CHECK(result.height == 64);
  }

  // The SAME width 96 the two-stage arm refuses, on a recipe whose only phase
  // runs at the requested size. 96 = 3 * 32, so it divides that grid and must be
  // served. A hardcoded 64 would refuse it here too: this is the assertion that
  // separates the derivation from the constant.
  SUBCASE("96 is a legal ONE-STAGE size, because that divisor is 32") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/one_stage_96");
    gen.width = 96;
    OneStageFixtureGuidance(&gen);
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.width == 96);
    CHECK(result.height == 64);
  }
}

// ─── the refusals, each of which would otherwise RENDER ─────────────────────

// PHASE L8 CHANGED WHAT THIS CASE ASSERTS, and the change is the phase.
//
// L7 had to refuse `device = 1` outright: the forward was f32-only by
// declaration and the staging was bf16 and refused to widen, so no combination
// put the DiT on an accelerator. L8 is the device-resident forward that closes
// that (`Ltx2DitForwardDevice`), so a non-zero handle now denotes a
// device-resident forward and the load must SUCCEED where an accelerator backend
// exists.
//
// The DSR repair (#553) changed WHICH accelerator from a hardcoded `kCUDA` to
// the platform seam's own answer, so this case now asks the seam the same
// question the engine asks instead of naming a device itself. On a CUDA box that
// resolves to kCUDA and the assertions below are the ones L8 shipped.
//
// What must never come back is the substitution: on a build with no accelerator
// backend the load is still refused, and the refusal must name the missing
// BACKEND. If it ever again names the f32/bf16 gap, the device forward has been
// un-wired; if it silently succeeds with a CPU device, the engine is lying about
// where it ran.
TEST_CASE("ltx2 video: device 1 runs on the resolved accelerator, refused by name without one") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.device = 1;
  const vt::DeviceType accelerator = vllm::platforms::CurrentPlatform().device_type();
  const bool have_accelerator = accelerator != vt::DeviceType::kCPU &&
                                vt::TryGetBackend(accelerator) != nullptr;
  if (!have_accelerator) {
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a device-1 load must be refused when no accelerator backend is registered");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("no accelerator backend") != std::string::npos);
      // The L7 gap must NOT be what is named any more; naming it would mean the
      // device forward is no longer wired in.
      CHECK(msg.find("kF32") == std::string::npos);
    }
    return;
  }
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  // The handle means what it says: the accelerator the seam resolved, never a
  // CPU device behind it.
  CHECK(engine->device().type == accelerator);
  CHECK(engine->device().type != vt::DeviceType::kCPU);
  CHECK(engine->family() == std::string(vllm::multimodal::kLtx2VideoFamily));
}

TEST_CASE("ltx2 video: a prompt with no text tower is refused, never quietly ignored") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/prompted");
  gen.prompt = "a cat riding a bicycle";
  try {
    (void)engine->Generate(gen);
    FAIL("a prompt that cannot be encoded must be refused, not replaced by the embeds");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("text tower") != std::string::npos);
  }

  // The engine loaded above has NO encoder, so `has_encoder()` says so and the
  // refusal above is the whole behaviour. Phase L13 lifted the LOAD-time refusal
  // of `encoder_path` itself — see "a typed PROMPT conditions the render" below.
  CHECK_FALSE(engine->has_encoder());

  // What is NOT lifted: an `encoder_path` that does not name a text encoder is
  // still refused at LOAD, where the caller can still act on it, rather than at
  // every request. The video VAE is a real safetensors file with none of the
  // encoder's tensors in it.
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.encoder_path = ws.paths.video_vae;
  mp.extras[vllm::multimodal::kLtx2EncoderConfigPathExtra] = ws.paths.encoder_config;
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("an encoder_path that is not a text encoder must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    // `Ltx2LoadGemmaAssets` keys the pack on `tokenizer_json`, which is the
    // first thing a non-encoder file cannot produce.
    CHECK(msg.find("tokenizer") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: one prompt-embeds file alone leaves a stream unconditioned") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras.erase(vllm::multimodal::kLtx2AudioPromptEmbedsExtra);
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("the video embeds alone must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("TWO streams") != std::string::npos);
  }

  // Two files whose ROW COUNTS disagree are two different prompts.
  SUBCASE("mismatched row counts are two prompts, and are refused") {
    const std::string short_audio = ws.root + "/short_audio.f32";
    ltx2_fixture::WritePromptEmbeds(short_audio, "ltx2.embeds.audio.short", 2,
                                    ltx2_fixture::ReducedDitParams().audio_cross_attention_dim);
    vllm::multimodal::VideoModelParams two = FixtureParams(ws.paths);
    two.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = short_audio;
    try {
      (void)vllm::multimodal::LoadVideoEngine(two);
      FAIL("two prompt-embeds files of different lengths must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("ONE tokenization") != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 video: an unknown extra is refused, not ignored") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras["partition"] = "fl2va";  // H3's knob, meaningless here
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("an extra this family does not define must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("partition") != std::string::npos);
  }
}

// A key this family DEFINES but does not serve is the worse half of the same
// defect, and the one an "unknown extra" check cannot see. `duration_head_path`
// was in `kKnownLoadExtras` and read by NOTHING (#611): supplying a duration head
// loaded no head, opened no file, and handed back the recipe default with no
// diagnostic. AGENTS.md requires an unimplemented arm to be refused with a
// message naming the missing piece, so it is refused rather than accepted.
//
// Dropping the key from `kKnownLoadExtras` instead would produce "unknown load
// extra", which is a DIFFERENT and wrong claim — the family defines the key and
// understands what it means; what is missing is the head. Hence the assertion on
// the missing piece and the alternative, not only on the key.
TEST_CASE("ltx2 video: duration_head_path is REFUSED by name, not silently ignored") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  // Any path at all: the point is that NOTHING opens it. Naming a file that does
  // exist keeps a not-found error from standing in for the refusal.
  mp.extras["duration_head_path"] = ws.paths.dit;
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("duration_head_path is served by no code; accepting it substitutes the recipe default");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("duration_head_path") != std::string::npos);
    // The MISSING PIECE, which is what separates this from "unknown key".
    CHECK(msg.find("duration head") != std::string::npos);
    // And what to use instead, so the refusal is actionable.
    CHECK(msg.find("num_frames") != std::string::npos);
    // Not the unknown-key message: that one would say the family does not define
    // it, and this family does.
    CHECK(msg.find("unknown load extra") == std::string::npos);
  }
}

// The INVENTORY, so the defect above cannot come back as a different key. Every
// extra this family accepts is either read by something or refused by name; a
// tenth decorative key fails this case rather than waiting to be discovered by
// the caller who supplies it.
//
// The audit behind it is in .agents/specs/ltx25-retire-dead-arms.md §2.1: nine of
// the ten keys have a reader and `duration_head_path` was the only one with none.
// The reader LINES are deliberately not repeated here — they moved twice while
// this row was in review. They live in one place, the READER ANCHORS comment in
// `ltx2_video.cpp`, and the case below derives them and holds that comment to it.
TEST_CASE("ltx2 video: every accepted load extra is READ by something") {
  Workspace ws;
  // The keys with a reader.
  const std::vector<std::string> served = {
      vllm::multimodal::kLtx2AudioPromptEmbedsExtra, vllm::multimodal::kLtx2PipelineKindExtra,
      vllm::multimodal::kLtx2ModelVersionExtra,      vllm::multimodal::kLtx2AllowUnportedExtra,
      vllm::multimodal::kLtx2MaxPhaseExtra,          vllm::multimodal::kLtx2DitConfigPathExtra,
      vllm::multimodal::kLtx2PromptValidRowsExtra,   vllm::multimodal::kLtx2EncoderConfigPathExtra,
      "upsampler_path",
      // Row LTX25-IC-LORA (#923): the IC-LoRA adapter and its strength. Both
      // have readers -- `lora_path` builds an `Ltx2LoraSpec` and `lora_strength`
      // is parsed into it -- so they belong here and not in `refused`.
      vllm::multimodal::kLtx2LoraPathExtra,          vllm::multimodal::kLtx2LoraStrengthExtra,
      // Row LTX25-GUIDED-VIDEO (#1092): the NEGATIVE half of the embeds
      // fallback. Both are read where the positive pair is, and both are read
      // again by the guided denoise loop when a guider asks for the
      // unconditional forward.
      vllm::multimodal::kLtx2NegativePromptEmbedsExtra,
      vllm::multimodal::kLtx2NegativeAudioPromptEmbedsExtra,
      // Row LTX25-CHECKPOINT-CLASS (#1137): the class the caller declares. Read
      // where the recipe resolves, and the only thing that reads it is the
      // refusal, which is what this key exists to produce.
      vllm::multimodal::kLtx2CheckpointClassExtra,
      // Row LTX25-DFR-ROUNDS (#986): the TEMPORAL x2 latent upsampler, DFR's
      // second upsampler slot. It has three readers, which is why it is served
      // rather than refused: the load parses its config and refuses the wrong
      // arm by name, the request surface refuses positive rounds without it, and
      // the rounds loop calls it once per round.
      vllm::multimodal::kLtx2TemporalUpsamplerPathExtra,
  };
  // The keys the family defines and does NOT serve. Growing this list is a
  // deliberate act; growing it silently is the defect #611 records.
  const std::vector<std::string> refused = {"duration_head_path"};

  // THE HANDLE ON THE REAL ARRAY. The unknown-extra refusal builds its listing
  // from `kKnownLoadExtras` itself, so parsing that listing gates the ACTUAL
  // accepted set rather than a copy of it maintained here. Without this the two
  // vectors above would be true by construction and would gate nothing.
  std::string listing;
  {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["definitely_not_a_key"] = "1";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("an unknown extra must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      const size_t at = msg.find("This family defines: ");
      REQUIRE(at != std::string::npos);
      listing = msg.substr(at + std::string("This family defines: ").size());
    }
  }
  INFO("listing = " << listing);
  // Every name this row inventoried is still accepted...
  for (const std::string& key : served) CHECK(listing.find(key) != std::string::npos);
  for (const std::string& key : refused) CHECK(listing.find(key) != std::string::npos);
  // ...and there is no name past the end of this inventory that it has never
  // seen. The separator is ", ", so the count is one more than the separators.
  size_t names = 1;
  for (size_t at = listing.find(", "); at != std::string::npos; at = listing.find(", ", at + 2)) {
    ++names;
  }
  CHECK_MESSAGE(names == served.size() + refused.size(),
                "kKnownLoadExtras grew; add the key to `served` (with its reader) or to "
                "`refused` (with a by-name refusal), per .agents/specs/ltx25-retire-dead-arms.md");

  // And the unserved half is refused rather than accepted.
  for (const std::string& key : refused) {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[key] = ws.paths.dit;
    INFO("key = " << key);
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("an accepted-but-unread extra must be refused by name");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find(key) != std::string::npos);
      CHECK(msg.find("unknown load extra") == std::string::npos);
    }
  }
}

// THE ANCHORS FOR THAT INVENTORY, DERIVED RATHER THAN TRUSTED.
//
// The case above proves each key is accepted or refused; it cannot prove WHERE a
// key is read, and "nine of ten reach a reader at these lines" is the claim this
// row rests on. That claim shipped wrong: the recorded anchors were nine lines
// that named no reader at all, in the very file they were recorded in. Then, in
// review, a merge of `origin/main` moved all nine again. A `file:line` written by
// hand is stale by the next commit, so this derives them from the source and
// holds the recorded list to what it finds. When it fails it prints the answer.
//
// It does NOT assert absolute line numbers of its own — nothing here to rot. The
// only obligation it creates is on whoever moves a reader: update the one comment
// block in the same file they are already editing.
namespace {

std::string ReadSourceFile(const char* path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open " << path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

std::vector<std::string> SplitLines(const std::string& text) {
  std::vector<std::string> lines;
  size_t at = 0;
  while (at <= text.size()) {
    const size_t end = text.find('\n', at);
    lines.push_back(text.substr(at, end == std::string::npos ? std::string::npos : end - at));
    if (end == std::string::npos) break;
    at = end + 1;
  }
  return lines;
}

// 1-based index of the ONLY line containing `needle`, or 0. Uniqueness is the
// point: an anchor that matches twice anchors nothing, and existence alone is
// what let the stale numbers survive.
size_t UniqueLineWith(const std::vector<std::string>& lines, const std::string& needle) {
  size_t found = 0;
  size_t count = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i].find(needle) != std::string::npos) {
      ++count;
      found = i + 1;
    }
  }
  return count == 1 ? found : 0;
}

std::string JoinNumbers(const std::vector<size_t>& v) {
  std::string s;
  for (size_t n : v) s += (s.empty() ? "" : " ") + std::to_string(n);
  return s;
}

}  // namespace

TEST_CASE("ltx2 video: the recorded reader anchors are the ones in the source") {
  const std::string source = ReadSourceFile(LTX2_VIDEO_SOURCE_PATH);
  const std::vector<std::string> lines = SplitLines(source);
  REQUIRE(lines.size() > 500);

  // Everything is measured relative to the accepted-keys array, so a stray
  // mention in the file header cannot be mistaken for a reader.
  const size_t array_line = UniqueLineWith(lines, "const char* const kKnownLoadExtras[] = {");
  REQUIRE_MESSAGE(array_line != 0, "kKnownLoadExtras[] declaration is not unique in the source");
  size_t array_end = 0;
  for (size_t i = array_line; i < lines.size(); ++i) {
    if (lines[i] == "};") {
      array_end = i + 1;
      break;
    }
  }
  REQUIRE(array_end > array_line);

  // The fourteen SERVED keys, by the token each is spelled with in the source.
  // Order is irrelevant — the comparison is on the sorted set — so this list is
  // not a second place the anchors live. Two arrived with row
  // LTX25-GUIDED-VIDEO (#1092) and the last with LTX25-CHECKPOINT-CLASS (#1137).
  const std::vector<std::string> served_tokens = {
      "kLtx2AudioPromptEmbedsExtra", "kLtx2PipelineKindExtra",  "kLtx2ModelVersionExtra",
      "kLtx2AllowUnportedExtra",     "kLtx2MaxPhaseExtra",      "kLtx2DitConfigPathExtra",
      "kLtx2PromptValidRowsExtra",   "kLtx2EncoderConfigPathExtra", "\"upsampler_path\"",
      "kLtx2LoraPathExtra",          "kLtx2LoraStrengthExtra",
      "kLtx2NegativePromptEmbedsExtra", "kLtx2NegativeAudioPromptEmbedsExtra",
      "kLtx2CheckpointClassExtra",
  };
  std::vector<size_t> derived;
  for (const std::string& token : served_tokens) {
    size_t first = 0;
    for (size_t i = array_end; i < lines.size(); ++i) {
      if (lines[i].find(token) != std::string::npos) {
        first = i + 1;
        break;
      }
    }
    INFO("token = " << token);
    CHECK_MESSAGE(first != 0, "no reader found after kKnownLoadExtras for " << token);
    if (first != 0) derived.push_back(first);
  }
  REQUIRE(derived.size() == served_tokens.size());
  std::sort(derived.begin(), derived.end());

  // The RECORDED list, parsed out of the one comment line that carries it.
  const size_t marker = UniqueLineWith(lines, "READER ANCHORS (derived and gated by");
  REQUIRE_MESSAGE(marker != 0,
                  "the READER ANCHORS comment marker is missing or not unique in the source");
  const std::string recorded_line = lines[marker];  // the line AFTER the marker (1-based)
  std::vector<size_t> recorded;
  for (size_t i = 0; i < recorded_line.size();) {
    if (std::isdigit(static_cast<unsigned char>(recorded_line[i]))) {
      size_t j = i;
      while (j < recorded_line.size() && std::isdigit(static_cast<unsigned char>(recorded_line[j]))) {
        ++j;
      }
      recorded.push_back(static_cast<size_t>(std::stoul(recorded_line.substr(i, j - i))));
      i = j;
    } else {
      ++i;
    }
  }
  std::sort(recorded.begin(), recorded.end());
  CHECK_MESSAGE(recorded == derived, "the reader anchors recorded in ltx2_video.cpp are STALE. "
                                     "Recorded: ["
                                         << JoinNumbers(recorded) << "]. Actual: ["
                                         << JoinNumbers(derived)
                                         << "]. Paste the actual list into the READER ANCHORS "
                                            "comment; the spec's §2.1 table is dated and stays.");

  // And the UNSERVED key is touched only by the refusal, never by a reader. This
  // is the half a "has a reader" sweep cannot express.
  const size_t refuse_line = UniqueLineWith(lines, "void CheckUnservedExtras(");
  REQUIRE(refuse_line != 0);
  size_t refuse_end = 0;
  for (size_t i = refuse_line; i < lines.size(); ++i) {
    if (lines[i] == "}") {
      refuse_end = i + 1;
      break;
    }
  }
  REQUIRE(refuse_end > refuse_line);
  size_t duration_hits = 0;
  for (size_t i = array_end; i < lines.size(); ++i) {
    if (lines[i].find("kLtx2DurationHeadPathExtra") == std::string::npos) continue;
    ++duration_hits;
    const size_t at = i + 1;
    const bool inside_refusal = (at >= refuse_line) && (at <= refuse_end);
    CHECK_MESSAGE(inside_refusal,
                  "ltx2_video.cpp:" << at
                                    << " touches the duration-head extra OUTSIDE "
                                       "CheckUnservedExtras; if it now has a real reader, move "
                                       "it to the served list and drop the refusal (#611)");
  }
  CHECK(duration_hits > 0);
}

// ─── the config the SHAPES cannot see ───────────────────────────────────────
//
// WHY THESE ASSERT ON THE ENGINE AND NOT ON A LOCAL. The L7 repair is one line in
// `Ltx2VideoEngine::Load` — `im.dit.params = declared;` — and before this case
// existed, DELETING it left the whole suite green: 13 cases / 142 assertions
// SUCCESS with the engine silently back to `double_precision_rope = false` and
// `av_ca_timestep_scale_multiplier = 1`. The only case that touched those values
// re-derived them from the file into its own local `declared` and asserted on
// THAT, which is a statement about `ParseLtx2DitParams`, not about what the
// engine bound. Spec §7.0(c) exactly: a fixture that cannot separate right from
// wrong. `Ltx2VideoEngine::dit_params()` exists so this can assert on the loaded
// engine, and the manifest control below is what makes the assertion non-vacuous.
TEST_CASE("ltx2 video: the engine ADOPTS the checkpoint's declared config") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> seam =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  const auto* engine = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(seam.get());
  REQUIRE(engine != nullptr);

  // THE CONTROL, first: what the SHAPES alone resolve from the very same file.
  // Without this the assertions below could be satisfied by the defaults, and
  // the case would gate nothing.
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(ws.paths.dit);
  vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kNvfp4;
  const vllm::Ltx2DitParams from_shapes =
      vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
  REQUIRE_FALSE(from_shapes.double_precision_rope);
  REQUIRE(from_shapes.av_ca_timestep_scale_multiplier == 1);

  // ... and what the ENGINE actually loaded, which is the whole point.
  const vllm::Ltx2DitParams& got = engine->dit_params();
  CHECK(got.double_precision_rope);                     // "frequencies_precision": "float64"
  CHECK(got.av_ca_timestep_scale_multiplier == 1000);   // declared 1000.0, default 1
  CHECK(got.timestep_scale_multiplier == 1000);
  CHECK(got.positional_embedding_theta == doctest::Approx(10000.0).scale(0.0));
  CHECK(got.positional_embedding_max_pos == std::vector<int64_t>{20, 2048, 2048});
  CHECK(got.audio_positional_embedding_max_pos == std::vector<int64_t>{20});
  CHECK(got.norm_eps == doctest::Approx(1e-6).scale(0.0));
  // The geometry must still be the file's, or the config was bound to the wrong
  // checkpoint and the contract check below did not do its job.
  CHECK(got.num_layers == from_shapes.num_layers);
  CHECK(got.inner_dim() == from_shapes.inner_dim());
  CHECK(got.audio_inner_dim() == from_shapes.audio_inner_dim());
}

TEST_CASE("ltx2 video: a declared config that disagrees with the SHAPES is refused") {
  // The other half of the L7 repair, and it was unexercised: adoption is allowed
  // only when the config reproduces the IDENTICAL weight contract, because a
  // config pasted in from another checkpoint would otherwise bind the wrong
  // tensors while looking like a more precise answer.
  Workspace ws;
  const std::string mismatched = ws.root + "/config_disagrees.safetensors";
  ltx2_fixture::ReducedDitOptions options;
  // The TENSORS are written from `ReducedDitParams()`; the CONFIG claims one
  // more layer. Nothing about the file's bytes changes.
  options.transformer_overrides["num_layers"] = ltx2_fixture::ReducedDitParams().num_layers + 1;
  ltx2_fixture::WriteReducedDit(ltx2_fixture::ReducedDitParams(), mismatched, options);
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.dit_path = mismatched;
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("a config describing a different weight contract must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("DIFFERENT weight contract") != std::string::npos);
  }
}

// The FP8 arm. MEASURED from the NAS 2026-08-12: `vonkaiser/LTX-2.5-FP8-NVFP4`'s
// `ltx-2.5-22b-distilled-fp8.safetensors` has NO `__metadata__` key at all, while
// the first-party NVFP4 DiT carries `['config','gemma_source_checkpoint',
// 'license','model_version']`. So for the copy L1-L6 gated against and L8 ran on
// the GPU, the adoption branch never executed and the DiT took the shape defaults
// — `double_precision_rope = false`, `av_ca_timestep_scale_multiplier = 1` —
// against LTX-2.5's declared float64 and 1000.
TEST_CASE("ltx2 video: a DiT that declares NO config is refused, never defaulted") {
  Workspace ws;
  const vllm::Ltx2DitParams params = ltx2_fixture::ReducedDitParams();
  const std::string bare = ws.root + "/no_metadata.safetensors";
  ltx2_fixture::ReducedDitOptions options;
  options.declare_config = false;
  options.declare_model_version = false;  // the shipped FP8 file carries neither
  ltx2_fixture::WriteReducedDit(params, bare, options);
  // The file really is in that shape, or this case proves nothing.
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(bare);
  REQUIRE(file.Metadata().count("config") == 0);
  REQUIRE(file.Metadata().count("model_version") == 0);

  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.dit_path = bare;

  SUBCASE("with nothing supplied, the load refuses BY NAME") {
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a DiT with no declared config must be refused, not given the defaults");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find(vllm::multimodal::kLtx2DitConfigPathExtra) != std::string::npos);
      CHECK(msg.find("double_precision_rope") != std::string::npos);
      CHECK(msg.find("av_ca_timestep_scale_multiplier") != std::string::npos);
    }
  }

  SUBCASE("supplied explicitly, it loads and the ENGINE runs under it") {
    const std::string cfg = ws.root + "/dit_config.json";
    ltx2_fixture::WriteDitConfigJson(params, cfg);
    mp.extras[vllm::multimodal::kLtx2DitConfigPathExtra] = cfg;
    mp.extras[vllm::multimodal::kLtx2ModelVersionExtra] = "2.5";
    const std::unique_ptr<vllm::multimodal::VideoEngine> seam =
        vllm::multimodal::LoadVideoEngine(mp);
    const auto* engine = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(seam.get());
    REQUIRE(engine != nullptr);
    CHECK(engine->dit_params().double_precision_rope);
    CHECK(engine->dit_params().av_ca_timestep_scale_multiplier == 1000);
  }

  SUBCASE("a supplied config that disagrees with the SHAPES is refused too") {
    const std::string cfg = ws.root + "/dit_config_wrong.json";
    nlohmann::json overrides;
    overrides["num_layers"] = params.num_layers + 1;
    ltx2_fixture::WriteDitConfigJson(params, cfg, overrides);
    mp.extras[vllm::multimodal::kLtx2DitConfigPathExtra] = cfg;
    mp.extras[vllm::multimodal::kLtx2ModelVersionExtra] = "2.5";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a supplied config binding a different contract must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("DIFFERENT weight contract") != std::string::npos);
      CHECK(msg.find(vllm::multimodal::kLtx2DitConfigPathExtra) != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 video: a checkpoint config AND a supplied one are refused, not ordered") {
  // Both present is a genuine ambiguity between two answers to a question no
  // shape can settle, so it is refused for the same reason two disagreeing
  // `model_version`s are.
  Workspace ws;
  const std::string cfg = ws.root + "/dit_config_dup.json";
  ltx2_fixture::WriteDitConfigJson(ltx2_fixture::ReducedDitParams(), cfg);
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2DitConfigPathExtra] = cfg;
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
    FAIL("two configs must be refused rather than silently ordered");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("Refusing rather than preferring one") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: the recipe comes from the CHECKPOINT's own model_version") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> seam =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  const auto* engine = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(seam.get());
  REQUIRE(engine != nullptr);
  // "2.5.0" in the file, reduced to the table's two-component key.
  CHECK(engine->model_version() == "2.5");
  CHECK(engine->pipeline_kind() == "distilled_two_stage");

  SUBCASE("a checkpoint of another generation is refused, never defaulted onto 2.5") {
    const std::string other = ws.root + "/v2_3.safetensors";
    ltx2_fixture::WriteReducedDit(ltx2_fixture::ReducedDitParams(), other, "2.3.0");
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = other;
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("distilled_two_stage/2.3 is not in the recipe table and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("2.3") != std::string::npos);
    }
  }

  SUBCASE("a declared version and an overriding extra that disagree are refused") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2ModelVersionExtra] = "2.3";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("two disagreeing model versions must be refused, not silently ordered");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("DIFFERENT sigma schedules") != std::string::npos);
    }
  }
}

// A binary PPM the engine can actually condition on. Deliberately NOT the
// generation's own resolution: `load_image_and_preprocess` aspect-fills and
// centre-crops to the phase's height/width (media_io/resize.py:41-73), and an
// image that already fits would leave that untested.
std::string ConditioningPpm(int height, int width, unsigned seed) {
  std::string out = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
  for (int i = 0; i < height * width * 3; ++i) {
    out.push_back(static_cast<char>((i * 37 + static_cast<int>(seed) * 101) % 251));
  }
  return out;
}

// BOTH phases, which for image conditioning is not a detail: the two-stage
// recipe renders its stages at DIFFERENT resolutions, so the image is decoded,
// resized and encoded once per phase against that phase's own height and width
// (ltx-pipelines/utils/helpers.py:274-275 are the parameters; distilled.py:251,
// :255-256 and :285-286 are where the two stages pass different values). A
// `max_phase = 0` fixture would
// leave the second encode — and the whole reason the conditioning lives inside
// the phase loop — untested.
vllm::multimodal::VideoModelParams ConditioningParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp = FixtureParams(paths);
  mp.extras["upsampler_path"] = paths.upsampler;
  return mp;
}

TEST_CASE("ltx2 video: keyframe and reference conditioning is refused BY WHAT IS MISSING") {
  // Row LTX25-IMAGE-COND (#644) SPLIT this refusal. It used to cover every
  // conditioning kind with one message whose reason was "no encoder weights can
  // be materialized here" — true when written, and no longer: this engine now
  // loads them through `Ltx2VideoVaeEncoderKeyRules`, and the first-frame arm is
  // served (see the case below).
  //
  // So each surviving refusal is held to naming a DIFFERENT missing piece. The
  // point is not that the message is long; it is that a later reader can go and
  // check the named symbol and find out whether the reason still holds — which
  // is the thing five refusals in this campaign failed at.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));

  auto refusal = [&](const char* what,
                     void (*arm)(vllm::multimodal::VideoGenParams&, const Workspace&)) {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/refused");
    arm(gen, ws);
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK(what << " must be refused, never dropped");
      return std::string();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };

  SUBCASE("a LAST-frame keyframe is no longer refused — it is SERVED") {
    // THIS SUBCASE USED TO ASSERT A REFUSAL, and before that it asserted a FALSE
    // one: it required the message to blame `keyframes_abs_pos_embedding`, which
    // at pin `fd4ded7f` is not what blocks a supplied keyframe — `apply_to`
    // appends it with `marked=False` (keyframe_cond.py:84-86) and the sole
    // consumer adds `mask * embedding` (transformer_args.py:42-43, called at
    // :269), so the embedding contributes nothing to those tokens.
    //
    // The reason it then named — the token-APPEND machinery — was the true one,
    // and row LTX25-TOKEN-APPEND (#930) built it. So the arm is checked here for
    // NOT refusing, and what it actually does is gated by "a LAST-frame keyframe
    // is APPENDED, and the sequence is trimmed back", which compares rendered
    // bytes against a no-op control.
    //
    // The check is kept in THIS case rather than only in that one because this
    // is the case a reader consults to ask "which conditioning arms are refused
    // today", and an arm that silently disappeared from it would leave that
    // question answered wrongly.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/served_last_frame");
    const std::string ppm = ws.root + "/served_last_frame.ppm";
    WriteBytes(ppm, ConditioningPpm(20, 28, 9));
    gen.last_frame_path = ppm;
    gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.frame_count == 9);
  }
  SUBCASE("a reference video may not blame a seam THIS ENGINE demonstrably has") {
    // WHAT THIS CASE USED TO DO, AND WHY THAT WAS THE DEFECT. It asserted five
    // SUBSTRINGS of the refusal: `reference_video_cond.py`, `clear_conditioning`,
    // `TOKEN-APPEND`, `NOT the IC-LoRA metadata`, and the absence of one retired
    // phrase. Two of those five are UPSTREAM symbol names, which are present in
    // the pinned checkout whatever this engine can do, and the other three are
    // literals the message declares about itself. So not one of them could go
    // red when the ENGINE changed — and the engine did change, twice, in two
    // days: #923 made the metadata readable and #930 (`c7cb59fbb`) built the
    // token-append seam. A reviewer replaced the local-cause sentence with a
    // self-declared falsehood, kept all five substrings, and the whole suite
    // stayed green.
    //
    // SO THIS CASE MEASURES THE ENGINE FIRST and only then constrains the
    // message. The measurement is the same instrument the token-append row
    // gates itself with: `video_tokens` is written INSIDE the phase loop, so it
    // can observe what the loop does, unlike every field filled before denoise.
    const vllm::multimodal::VideoModelParams cond_params = ConditioningParams(ws.paths);
    const std::string kf = ws.root + "/append_witness.ppm";
    WriteBytes(kf, ConditioningPpm(20, 28, 31));

    auto tokens_of = [&](const std::string& tag, const std::string& keyframe) {
      const std::unique_ptr<vllm::multimodal::VideoEngine> own =
          vllm::multimodal::LoadVideoEngine(cond_params);
      auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(own.get());
      REQUIRE(ltx2 != nullptr);
      vllm::multimodal::VideoGenParams g = FixtureGen(ws.root + "/" + tag);
      if (!keyframe.empty()) {
        g.last_frame_path = keyframe;
        g.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
      }
      const vllm::multimodal::VideoResult result = own->Generate(g);
      // THE TRIM, observed from outside: the volume handed to unpatchify is the
      // target grid, so the clip comes back at the requested length whether or
      // not anything was appended.
      CHECK(result.frame_count == 9);
      return ltx2->last_conditioning().video_tokens;
    };

    const int64_t plain = tokens_of("ref_witness_plain", "");
    const int64_t grown = tokens_of("ref_witness_grown", kf);
    // THE GROWTH. Both numbers are measured; pinning either to a literal would
    // pass on a build that never grew anything.
    REQUIRE_MESSAGE(grown > plain,
                    "this engine's phase loop did not grow its token sequence for an appending "
                    "conditioning item ("
                        << grown << " against a target of " << plain
                        << "), so the rest of this case cannot say what the refusal may claim");

    const std::string msg = refusal("a reference video",
                                    [](vllm::multimodal::VideoGenParams& g, const Workspace& w) {
                                      g.ref_video_dir = w.root;
                                    });
    INFO(msg);

    // BECAUSE THE TWO MEASUREMENTS ABOVE HOLD, the refusal may not CLAIM the
    // phase loop. It may still MENTION it — the message's own convention is to
    // record a ruled-out cause under `WHAT IS *NOT* THE REASON` so the next
    // reader re-checks rather than re-derives — so the property asserted here is
    // positional: every occurrence of a closed cause sits after that marker.
    //
    // That is what makes this case red for the mutation that motivated it.
    // Restoring the pre-repair message leaves no marker at all AND puts
    // `TOKEN-APPEND` in the first sentence, so both halves fire.
    const size_t ruled_out = msg.find("WHAT IS *NOT* THE REASON");
    REQUIRE_MESSAGE(ruled_out != std::string::npos,
                    "the refusal carries no `WHAT IS *NOT* THE REASON` section, so a cause this "
                    "engine has already closed cannot be told apart from one it still has");
    for (const char* closed : {"TOKEN-APPEND", "fixed at the target grid's token count",
                               "nowhere to go", "nothing to trim"}) {
      const size_t at = msg.find(closed);
      const bool only_as_ruled_out = (at == std::string::npos) || (at > ruled_out);
      CHECK_MESSAGE(only_as_ruled_out,
                    "the refusal states '"
                        << std::string(closed)
                        << "' as a cause rather than as a ruled-out one, and this case has just "
                           "MEASURED that the loop grows ("
                        << plain << " -> " << grown << ") and trims back to the target grid");
    }
    // The metadata half, same shape: the factors printed are READ from the
    // adapter at load, so this asserts the read happened rather than asserting
    // a sentence about it. `factors` says "no adapter was supplied" here.
    CHECK(msg.find("no adapter was supplied") != std::string::npos);
    CHECK(msg.find("which this project does not read") == std::string::npos);
    // And the two causes that DO remain are named, by the upstream anchors a
    // reader can go and check.
    CHECK(msg.find("iclora_utils.py:116-117") != std::string::npos);
    CHECK(msg.find("ic_lora.py:108") != std::string::npos);
    CHECK(msg.find("ref_video_dir") != std::string::npos);
  }
  SUBCASE("reference audio names the AUDIO encoder, which this row did not build") {
    const std::string msg = refusal("reference audio",
                                    [](vllm::multimodal::VideoGenParams& g, const Workspace& w) {
                                      g.ref_audio_path = w.paths.audio_embeds;
                                    });
    INFO(msg);
    CHECK(msg.find("audio VAE") != std::string::npos);
    CHECK(msg.find("encode_audio") != std::string::npos);
  }
  SUBCASE("a non-zero CRF names the codec round trip, and says 0 is supported") {
    // AND THIS IS THE DEFAULT PATH. An LTX-2.5 checkpoint resolves
    // `default_image_crf = 18` (constants.py:37/124/130-133), so a caller who
    // says nothing about the CRF lands here — which is what makes the
    // out-of-distribution `crf = 0` arm a deliberate request rather than a
    // silent downgrade.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/crf");
    gen.first_frame_ppm = ConditioningPpm(20, 28, 3);
    try {
      (void)engine->Generate(gen);
      FAIL("an unset CRF resolves 18 for a 2.5 checkpoint and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("CRF 18") != std::string::npos);
      CHECK(msg.find("encode_single_frame") != std::string::npos);
      CHECK(msg.find("CRF 0 IS supported") != std::string::npos);
    }
  }
  SUBCASE("an explicit non-zero CRF is refused just as an unset one is") {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/crf33");
    gen.first_frame_ppm = ConditioningPpm(20, 28, 4);
    gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "33";
    CHECK_THROWS_WITH_AS((void)engine->Generate(gen), doctest::Contains("CRF 33"),
                         std::runtime_error);
  }
  SUBCASE("a mistyped per-generation extra is refused, not ignored") {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/typo");
    gen.first_frame_ppm = ConditioningPpm(20, 28, 5);
    gen.extras["image_crf_"] = "0";
    CHECK_THROWS_WITH_AS((void)engine->Generate(gen), doctest::Contains("image_crf_"),
                         std::runtime_error);
  }
}

TEST_CASE("ltx2 video: GENERATED keyframe slots are SERVED, and the marker reaches them") {
  // Row LTX25-DFR-PIPELINE (#986). This case REPLACES the refusal case row
  // LTX25-GENERATED-KEYFRAMES (#920) landed, and the replacement is itself the
  // record of what changed.
  //
  // #920 defined `num_generated_keyframes` and refused any positive count,
  // naming ONE blocker: the READBACK. Its message declared `ABSENT HERE:
  // GeneratedKeyframe, generated_keyframe` and this suite re-derived those names
  // against `ltx2_conditioning.h` — a tripwire whose own spec (section 4a of
  // `.agents/specs/ltx25-generated-keyframes.md`) said in terms: "If the readback
  // lands, `GeneratedKeyframe` appears in the header, ABSENT goes red, and
  // whoever landed it is told the refusal is now false."
  //
  // It landed, the tripwire fired, and the assertions are RETIRED WITH THE
  // REFUSAL THEY DESCRIBED rather than widened. `AGENTS.md` forbids making a red
  // gate green by deleting an assertion; this is the other case, where the
  // SUBJECT of the assertion no longer exists, so each one is replaced by an
  // assertion about what replaced it.
  //
  // This is still the OTHER feature called "keyframe", and the distinction is
  // still the point:
  //
  //   supplied  -> `VideoConditionByKeyframeIndex`, content from the caller,
  //                appended with `marked=False` (keyframe_cond.py:84-86)
  //   GENERATED -> `VideoGeneratedKeyframeSlots`, content from the MODEL,
  //                appended with `marked=True` (keyframe_slots.py:121)
  //
  // WHY THIS CASE LEANS ON THE TRACE AND NOT ON PIXELS. A generated keyframe
  // slot is INVISIBLE to the rendered clip: its tokens are appended, denoised,
  // read back, and then trimmed away before unpatchify. A build that placed no
  // slots at all returns a video of the same shape, the same frame count and the
  // same byte size. There is nothing in the artifact to compare, which is why
  // the engine reports what it did to the STATE.
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  // Phase 0 only: the second phase needs the latent spatial upsampler, which is
  // a separate and unrelated refusal.
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);

  auto refusal = [&](const char* key, const char* value, const char* dir) {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + dir);
    gen.extras[key] = value;
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK(key << "=" << value << " must be refused, never dropped");
      return std::string();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };

  SUBCASE("a positive count places slots, reads them back, and MARKS every token") {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/gk2");
    gen.extras[vllm::multimodal::kLtx2GeneratedKeyframesExtra] = "2";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    // The clip is unchanged in every dimension a caller can see. Asserted rather
    // than assumed, because it is what makes the trace the only instrument.
    CHECK(result.frame_count == 9);
    CHECK(result.width == 32);
    CHECK(result.height == 32);

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    REQUIRE(trace.slot_positions.size() == 2);
    // `evenly_spaced_keyframe_positions` (utils/helpers.py:370-381) on 9 frames
    // with n=2 is `linspace(0, 8, 4).round()[1:-1]` = {3, 5}. Written out here
    // rather than recomputed from the implementation, so the two are independent
    // expressions and a mutation of one does not move the other.
    CHECK(trace.slot_positions[0] == 3);
    CHECK(trace.slot_positions[1] == 5);
    // THE ENDPOINTS ARE DROPPED (`[1:-1]`, helpers.py:381). Frame 0 already spans
    // a single pixel frame under causal encoding and the terminal frame is the
    // clip's own end, so a slot at either buys nothing. A port that kept them
    // would place 4 slots and still render.
    for (int64_t position : trace.slot_positions) {
      CHECK(position > 0);
      CHECK(position < result.frame_count - 1);
    }

    // One latent frame of tokens per slot (keyframe_slots.py:83-84). At 32x32
    // with VIDEO_SCALE_FACTORS (8, 32, 32) the latent grid is 1x1, so one latent
    // frame is one token and two slots are two tokens. Derived from the geometry
    // rather than hard-coded, so a fixture resized later moves both sides.
    const int64_t per_frame = (result.width / 32) * (result.height / 32);
    REQUIRE(per_frame >= 1);
    CHECK(trace.slot_marked_tokens == 2 * per_frame);

    // THE READBACK RAN, and returned one latent frame per slot. This is the half
    // #920 named as its blocker: `clear_conditioning` extracts into
    // `generated_keyframes` BEFORE it trims (ltx_core/tools.py:97, :115, and
    // `extract_generated_keyframes` at :203-230). A build that trimmed first
    // reports slots placed and ZERO extracted, with a byte-identical video.
    CHECK(trace.slot_tokens_extracted == 2);

    // AND THE MARKER REACHED THEM. `extend_keyframes_mask(..., marked=True)`
    // (keyframe_slots.py:121) is upstream's single marked call site and is the
    // ONLY thing separating a generated slot from an ordinary append. An
    // unmarked slot costs the same tokens, renders the same clip, and silently
    // omits #658's trained embedding — so this is the assertion with no
    // observable proxy anywhere else in the tree.
    CHECK(trace.slot_marked_tokens > 0);
  }

  SUBCASE("zero is upstream's DEFAULT and places nothing") {
    // args.py:836 is `default=0`, and `has_generated_keyframes`
    // (utils/helpers.py:384-391) reads 0 as off. A caller that plumbs the
    // default through must get an ordinary render. This is the half a naive port
    // breaks — "the key is present, so act" is one line shorter and wrong, and
    // it stays wrong now that the arm is served rather than refused.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/gk0");
    gen.extras[vllm::multimodal::kLtx2GeneratedKeyframesExtra] = "0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.frame_count == 9);
    CHECK(result.width == 32);
    CHECK(result.height == 32);
    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    CHECK(trace.slot_positions.empty());
    CHECK(trace.slot_marked_tokens == 0);
    CHECK(trace.slot_tokens_extracted == 0);
  }

  SUBCASE("slots applied AFTER a supplied keyframe still describe their own tokens") {
    // THE ORDERING HALF, and it had no case until
    // [#1219](https://github.com/mudler/vllm.cpp/issues/1219). Upstream applies
    // conditioning items in list order and DFR appends the slot item LAST
    // (dfr_pipeline.py:320-330, :353-365), so a supplied keyframe stands in front
    // of the slots whenever the caller pins one. Every slot assertion the engine
    // makes has to locate the slot tokens, and this is the arrangement in which
    // "the tokens after the target grid" and "the tokens this item appended" are
    // DIFFERENT windows.
    //
    // A closing keyframe is the reachable way to get an append in front of the
    // slots on this kind: `wants_last_frame` takes `VideoConditionByKeyframeIndex`
    // and appends (keyframe_cond.py:79-82) on every pipeline, where the first
    // frame only appends on the `image_conditionings_by_adding_guiding_latent`
    // recipes. The two features are unrelated and compose, which is the point.
    const std::string closing = ws.root + "/gk_closing.ppm";
    WriteBytes(closing, ConditioningPpm(20, 28, 51));

    auto request = [&](const char* dir, bool slots, bool keyframe) {
      vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + dir);
      if (slots) gen.extras[vllm::multimodal::kLtx2GeneratedKeyframesExtra] = "2";
      if (keyframe) {
        gen.last_frame_path = closing;
        gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
      }
      const vllm::multimodal::VideoResult result = engine->Generate(gen);
      CHECK(result.frame_count == 9);
      return ltx2->last_conditioning();
    };

    // Four renders, so the count below is a RELATION between measured lengths
    // rather than a literal: each item's own cost is measured on its own, and
    // the claim is that together they cost the sum. `image_tokens` cannot serve
    // here — the engine fills it on the FIRST-frame arm only, and this closing
    // keyframe leaves it 0.
    const vllm::multimodal::Ltx2ConditioningTrace bare = request("gk_bare2", false, false);
    const vllm::multimodal::Ltx2ConditioningTrace slots_only = request("gk_alone", true, false);
    const vllm::multimodal::Ltx2ConditioningTrace kf_only = request("gk_kf_only", false, true);
    const vllm::multimodal::Ltx2ConditioningTrace both = request("gk_after_kf", true, true);

    REQUIRE(slots_only.slot_positions.size() == 2);
    REQUIRE(slots_only.slot_marked_tokens > 0);
    REQUIRE(slots_only.video_tokens > bare.video_tokens);
    REQUIRE(kf_only.video_tokens > bare.video_tokens);

    MESSAGE("video_tokens  bare " << bare.video_tokens << "  slots " << slots_only.video_tokens
                                  << "  keyframe " << kf_only.video_tokens << "  both "
                                  << both.video_tokens);

    // The slots are placed and MARKED exactly as they are alone: a keyframe in
    // front changes WHERE they sit in the sequence and nothing about what they
    // are. This is the assertion the engine's own slot checks stand behind, and
    // the arrangement that makes "past the target grid" and "what this item
    // appended" two different windows.
    CHECK(both.slot_positions == slots_only.slot_positions);
    CHECK_MESSAGE(both.slot_marked_tokens == slots_only.slot_marked_tokens,
                  "a supplied keyframe in front of the slots changed how many slot tokens "
                  "carry the trained marker — expected " << slots_only.slot_marked_tokens
                                                         << ", got " << both.slot_marked_tokens);
    CHECK_MESSAGE(both.video_tokens == slots_only.video_tokens +
                                           (kf_only.video_tokens - bare.video_tokens),
                  "the two appending items must cost what each costs alone: "
                      << slots_only.video_tokens << " + "
                      << kf_only.video_tokens - bare.video_tokens << " against "
                      << both.video_tokens);
  }

  SUBCASE("a negative count gets upstream's OWN reason") {
    // `evenly_spaced_keyframe_positions` raises "num_keyframes must be
    // non-negative" (utils/helpers.py:372-373) before anything looks at the
    // checkpoint.
    const std::string msg =
        refusal(vllm::multimodal::kLtx2GeneratedKeyframesExtra, "-1", "gkneg");
    INFO(msg);
    CHECK(msg.find("non-negative") != std::string::npos);
    CHECK(msg.find("helpers.py:372-373") != std::string::npos);
  }

  SUBCASE("a count the clip is too short for gets upstream's SECOND reason") {
    // `num_frames < num_keyframes + 2` (helpers.py:374-378). Every slot is an
    // INTERIOR position, so the two endpoints are not available to it and a
    // request for 8 slots in a 9-frame clip has nowhere to put them. Distinct
    // from the negative refusal: a port that collapsed the two would tell a
    // caller who asked for too many that they asked for a negative number.
    const std::string msg =
        refusal(vllm::multimodal::kLtx2GeneratedKeyframesExtra, "8", "gktoomany");
    INFO(msg);
    CHECK(msg.find("num_keyframes + 2") != std::string::npos);
    CHECK(msg.find("non-negative") == std::string::npos);
  }

  SUBCASE("the key is still DEFINED, so a neighbouring typo is answered differently") {
    // Carried over from the refusal case this replaces, because the property it
    // guards survives the change: the family DEFINES this key, so a caller who
    // mistypes it must not get the same answer as one who typed it correctly.
    // That is the distinction `CheckUnservedExtras` was written for on the load
    // side (#611).
    const std::string msg = refusal("num_generated_keyframe", "1", "gktypo");
    INFO(msg);
    CHECK(msg.find("unknown per-generation extra") != std::string::npos);
    CHECK(msg.find("num_generated_keyframes") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: the DFR pipeline pads its canvas, places slots on it, and trims back") {
  // Row LTX25-DFR-PIPELINE (#986), and this is the row's REACHABILITY PROOF.
  //
  // It enters through the production entry point — `LoadVideoEngine` then
  // `VideoEngine::Generate`, the chain `vllm_video_generate` takes — and reaches
  // `Ltx2DfrResolveCanvas` and `Ltx2ConditionVideoByGeneratedKeyframeSlots`
  // through the `pipeline_kind` load extra, which `ltx2-gen --pipeline-kind`
  // and the ABI's `extra_keys`/`extra_values` both carry.
  // `.agents/reachability.md` asks for exactly this: the smallest failing test
  // starts at the entry point rather than constructing the type, because a unit
  // test proves the class works and never that anything reaches it. The unit
  // suite `test_ltx2_dfr` is kept beside this one — it localizes a failure — and
  // it is not the proof.
  //
  // WHAT THE CANVAS DOES TO A 9-FRAME REQUEST, derived rather than asserted from
  // the implementation: `resolve_canvas` (dfr_layout.py:60-81) works on
  // `content = num_frames - 1 = 8`. `choose_segment_length` pads 8 by 16 against
  // segment 24 and by 24 against 32, so 24 wins outright, and the canvas becomes
  // `8 + 16 + 1 = 25` frames with one keyframe position at 24 — the terminal
  // frame. The pipeline therefore DENOISES 25 frames and hands the caller back
  // 9, which is the whole point of `dfr_pipeline.py:531-540` and the single
  // most surprising thing about this pipeline.
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "dfr";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("dfr");
  // Phase 0 only: phase 1 needs the latent spatial upsampler, and this case is
  // about the canvas and the slots rather than about the detailing stage.
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  CHECK(ltx2->pipeline_kind() == "dfr");

  SUBCASE("the canvas pads, the slots land on its segment grid, and the caller gets 9 frames") {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr");
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    // THE CONTRACT. `(requested - 1) * 2**0 + 1 == requested`
    // (dfr_pipeline.py:534). The caller asked for 9 and gets 9, even though the
    // pipeline denoised more.
    CHECK(result.frame_count == 9);

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    // AND THE CANVAS WAS BIGGER, which is the half no output can show. A build
    // that skipped the pad renders 9 frames too, from a 9-frame canvas, with a
    // keyframe grid that does not divide it.
    CHECK(trace.canvas_frames == 25);
    CHECK(trace.canvas_segment == 24);
    CHECK(trace.canvas_frames > result.frame_count);

    // The slots sit on the segment grid `resolve_canvas` returned: `[S, 2S, ...,
    // N'-1]`, which for this canvas is the single terminal position 24. Frame 0
    // is EXCLUDED — it already spans a single pixel frame under causal encoding
    // (dfr_layout.py:66-68).
    REQUIRE(trace.slot_positions.size() == 1);
    CHECK(trace.slot_positions[0] == 24);
    CHECK(trace.slot_positions[0] == trace.canvas_frames - 1);
    CHECK(trace.slot_positions[0] % 8 == 0);

    // The slots were placed, MARKED and read back. Each of the three is a
    // separate failure mode and none of them moves a pixel: see the
    // generated-keyframe case above for why the trace is the only instrument.
    CHECK(trace.slot_marked_tokens > 0);
    CHECK(trace.slot_tokens_extracted == 1);
  }

  SUBCASE("a frame count off the x8 border is REFUSED, where the ordinary pipeline floors it") {
    // The asymmetry is upstream's and it is the reason DFR needs its own guard.
    // `resolve_num_frames` returns an explicit count verbatim and
    // `VideoLatentShape.from_pixel_shape` FLOORS it, which `docs/USAGE.md`
    // documents as this engine's behaviour (#919). DFR cannot live with that:
    // every keyframe position it emits has to land on a latent border, and a
    // floored count moves the terminal position off one. So `resolve_canvas`
    // raises (dfr_layout.py:71-72) and so does this.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfroff");
    gen.num_frames = 10;
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("a frame count off the x8 border must be refused on the dfr pipeline");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("dfr_layout.py:71-72") != std::string::npos);
      CHECK(msg.find("#919") != std::string::npos);
    }
  }

  SUBCASE("BOTH stages run, and stage 2's slots are SEEDED from stage 1's") {
    // The two-stage DFR flow, through the production entry point, with the
    // fixture spatial upsampler. Upstream's stage 2 passes
    // `initial_keyframes=upsampled_slot_keyframes` (dfr_pipeline.py:364), which
    // are stage 1's own denoised slots run through the SPATIAL latent upsampler
    // (:348) — the same object and the same call as the video latent on the line
    // after it.
    //
    // WHAT THIS CASE DOES AND DOES NOT ESTABLISH, stated because the first
    // version of this comment claimed more. It REACHES the seeded path: without
    // it, `initial_keyframes` is null on every engine test, because phase 0 has
    // no previous phase to seed from. It does NOT DETECT the seed's content.
    // Measured, not assumed: mutation M5, which stops the seed reaching
    // `latent`, leaves this suite at 52/52, exit 0, WITH this case present.
    //
    // The reason is a property of the pipeline rather than a hole in this file.
    // Stage 2 re-noises to `stage_2_sigmas[0]`, about 0.909, so the seed is
    // almost entirely replaced by noise before the first step and the denoise
    // loop generates the rest; the assertions available here are structural —
    // counts, positions, resolutions — and the seed moves none of them. Where
    // the seed IS gated is `test_ltx2_dfr`, which checks that it lands in
    // `latent` and not in `clean`, and where M5 goes RED.
    //
    // A separate engine is loaded because `max_phase` is a LOAD extra and the
    // outer one pins phase 0.
    vllm::multimodal::VideoModelParams two = FixtureParams(ws.paths);
    two.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "dfr";
    two.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("dfr");
    two.extras["upsampler_path"] = ws.paths.upsampler;
    const std::unique_ptr<vllm::multimodal::VideoEngine> full =
        vllm::multimodal::LoadVideoEngine(two);
    REQUIRE(full != nullptr);
    auto* ltx2_full = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(full.get());
    REQUIRE(ltx2_full != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr2");
    const vllm::multimodal::VideoResult result = full->Generate(gen);

    // Stage 2 renders at FULL resolution, where phase 0 alone renders at half.
    // That is what proves both phases ran, and it is read off the artifact
    // rather than off a flag.
    CHECK(result.frame_count == 9);
    CHECK(result.width == 64);
    CHECK(result.height == 64);

    // W0 (#1010) names "any latent-upsampler stage", and this leaf is the answer
    // to it. The latent SPATIAL x2 upsampler used to run inside `phase.prepare`,
    // a leaf whose name says nothing about it, and this two-phase render is the
    // ONLY one in this file that reaches it — every other case pins `max_phase`
    // at 0, where the input transform is never the spatial upsample.
    REQUIRE_MESSAGE(!result.phase_log_path.empty(),
                    "the two-phase render wrote no phase table");
    const nlohmann::json upsample_table =
        nlohmann::json::parse(ReadAll(result.phase_log_path));
    bool named_upsample = false;
    for (const nlohmann::json& entry : upsample_table["phases"]) {
      if (entry["name"].get<std::string>() == "phase.upsample_latent") named_upsample = true;
    }
    CHECK_MESSAGE(named_upsample,
                  "the two-phase render's table names no 'phase.upsample_latent' leaf, so the "
                  "latent upsampler is back inside 'phase.prepare' with nothing saying so");

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2_full->last_conditioning();
    // The trace describes the LAST phase that ran, so a slot count here means
    // stage 2 placed slots of its own rather than inheriting stage 1's tokens.
    REQUIRE(trace.slot_positions.size() == 1);
    CHECK(trace.slot_positions[0] == 24);
    CHECK(trace.canvas_frames == 25);
    // And stage 2 read its slots back too, which is what the next round would
    // carry forward if the rounds were served.
    CHECK(trace.slot_tokens_extracted == 1);
    // Stage 2's slot is 4 tokens where stage 1's is 1: the latent grid is 2x2 at
    // full resolution against 1x1 at half. Derived from the result's own size so
    // a mutation of the phase geometry moves this and not just the count.
    const int64_t per_frame = (result.width / 32) * (result.height / 32);
    CHECK(per_frame == 4);
    CHECK(trace.slot_marked_tokens == per_frame);
  }

  SUBCASE("the soundtrack is CUT to the picture, not to the padded canvas") {
    // `dfr_pipeline.py:552-560`, whose own reason is the consequence rather than
    // the mechanism: "Audio was generated for the padded canvas, so cut it to
    // the video's duration or the muxed container outlasts the picture."
    //
    // This case exists because the first version of this row got it wrong and
    // said so in a comment. The video trim moves `frames`; the audio latent's
    // frame count was derived from the PADDED `frames` inside the phase loop and
    // the vocoder runs over all of it, so trimming the video touches nothing
    // about the sound. A 9-frame DFR request emitted 9 frames of picture beside
    // 25 frames' worth of audio, and NOTHING about the render's shape, its frame
    // count or its exit status could see it — it shows up only in a muxed
    // container this library does not produce.
    //
    // The bound is derived from the RESULT's own fields rather than from the
    // canvas, so a mutation of the cut cannot move both sides together.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfraudio");
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    REQUIRE(result.frame_count == 9);
    REQUIRE(result.sample_rate > 0);
    REQUIRE(result.fps > 0);

    const std::string wav = ReadAll(result.audio_path);
    REQUIRE(wav.size() > 44);  // canonical RIFF/WAVE header
    const double seconds_of_picture =
        static_cast<double>(result.frame_count) / static_cast<double>(result.fps);

    // The duration is READ OUT OF THE FILE rather than assumed, and the first
    // draft of this case got that wrong in a way worth recording: it took the
    // channel count as 1, calling that "the loosest reading and therefore the
    // safe direction". It is the opposite. Dividing by too FEW channels reports
    // a longer file than exists, so the bound false-fails on a correct cut,
    // which is exactly what it did — 0.375 s of picture against a reported
    // 0.75 s of sound on a 2-channel file that was already the right length.
    auto u16 = [&](size_t at) {
      return static_cast<int64_t>(static_cast<unsigned char>(wav[at])) |
             (static_cast<int64_t>(static_cast<unsigned char>(wav[at + 1])) << 8);
    };
    auto u32 = [&](size_t at) { return u16(at) | (u16(at + 2) << 16); };
    REQUIRE(wav.compare(0, 4, "RIFF") == 0);
    REQUIRE(wav.compare(8, 4, "WAVE") == 0);
    const int64_t channels = u16(22);
    const int64_t rate = u32(24);
    const int64_t bits = u16(34);
    const int64_t data_bytes = u32(40);
    REQUIRE(channels >= 1);
    REQUIRE(bits == 16);
    REQUIRE(rate == result.sample_rate);
    REQUIRE(data_bytes > 0);
    REQUIRE(static_cast<size_t>(data_bytes) <= wav.size() - 44);
    const double seconds_of_sound = static_cast<double>(data_bytes) /
                                    (static_cast<double>(bits / 8) *
                                     static_cast<double>(channels) * static_cast<double>(rate));
    INFO("picture ", seconds_of_picture, "s, sound ", seconds_of_sound, "s, ", channels,
         " channels at ", rate, " Hz");
    // One frame of slack for the rounding in `llround`, and no more. Before the
    // cut this was the PADDED canvas's duration, about 2.8x the picture.
    CHECK(seconds_of_sound <= seconds_of_picture + (1.0 / static_cast<double>(result.fps)) + 1e-6);
    // And it is not EMPTY, which a cut that computed zero would also satisfy.
    CHECK(seconds_of_sound > 0.0);
  }

  SUBCASE("a request wrong on BOTH axes hears about the RESOLUTION first") {
    // Upstream's order, and it is an order rather than a set: `assert_resolution`
    // is at dfr_pipeline.py:291 and `resolve_canvas` at :314. A request that is
    // wrong on both the resolution and the frame count therefore gets the
    // resolution refusal upstream, and must get it here too.
    //
    // This is gated because it is invisible otherwise. Both refusals are
    // correct, both name a real defect in the request, and a port that resolved
    // the canvas first would look completely healthy to every other case in this
    // file while giving a different answer from the reference to the same input.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfrboth");
    gen.width = 96;      // not a multiple of 64, which a two-stage recipe needs
    gen.num_frames = 10; // and not on the x8 border either
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("a request wrong on both axes must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      // The RESOLUTION refusal, not the canvas one.
      CHECK(msg.find("96") != std::string::npos);
      CHECK(msg.find("dfr_layout.py:71-72") == std::string::npos);
    }
  }

  SUBCASE("num_generated_keyframes is refused on dfr, which owns its own grid") {
    // `DFRPipeline.__call__` takes no such argument and its CLI exposes no
    // `--num-generated-keyframes` (dfr_pipeline.py:268-283, :565-591). Honouring
    // an override would detach the slots from the canvas that indexes them —
    // the tile ranges and the carry-forward bag are built on that same grid —
    // and the render would still finish.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfrgk");
    gen.extras[vllm::multimodal::kLtx2GeneratedKeyframesExtra] = "2";
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("num_generated_keyframes must be refused on the dfr pipeline");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("resolve_canvas") != std::string::npos);
      CHECK(msg.find("not accepted on the 'dfr' pipeline") != std::string::npos);
      // Explicit 0 is upstream's default and must still pass through.
      CHECK(msg.find("unknown per-generation extra") == std::string::npos);
    }
  }

  SUBCASE("an explicit 0 for num_generated_keyframes still renders on dfr") {
    // The refusal above is keyed on `count != 0`, not on the key's presence. A
    // caller that plumbs upstream's own default through must not be refused —
    // the same half a naive port breaks on the ordinary pipelines.
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfrgk0");
    gen.extras[vllm::multimodal::kLtx2GeneratedKeyframesExtra] = "0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.frame_count == 9);
    // And the canvas still placed ITS OWN slot, which is what distinguishes
    // "generated keyframes off" from "DFR without its keyframe grid". A port
    // that read the 0 as "no slots" would render a DFR clip with no slots at
    // all, at the right size, and nothing else would show it.
    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    CHECK(trace.slot_positions.size() == 1);
  }
}

TEST_CASE("ltx2 video: temporal rounds are refused where they mean nothing") {
  // Row LTX25-DFR-ROUNDS (#986). THE REFUSAL THIS REPLACES, and why it is retired
  // rather than widened.
  //
  // Until this row the whole knob was refused above 0, and the message named the
  // ROUNDS LOOP as what was missing while ruling three other causes out by name.
  // The loop has landed, so that message is now false about this tree and the
  // assertions that re-derived it go WITH it — the same disposal #920's tripwire
  // got when the readback landed (see the retirement record above, around the
  // `GENERATED keyframe slots are SERVED` case). Each of its assertions is
  // replaced below by one about what replaced it, never deleted and never
  // loosened to keep a green.
  //
  // ONE OF ITS FOUR STATED CAUSES WAS ALREADY FALSE WHEN THIS ROW READ IT, and
  // that is the more useful record. The message ended with "AND ONE THING IS
  // MISSING THAT NO CODE CAN SUPPLY: the checkpoint", asserting that
  // `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` is not on the NAS,
  // "re-verified 2026-08-16". It is on the NAS, and it arrived 2026-08-17 — one
  // day after that check. Nothing in the tree could notice: a refusal's stated
  // reason is prose, and the only instrument on it is a reader who goes and
  // looks. This suite is that reader, for the claims it CAN reach.
  //
  // What survives the retirement is the part that is still true: a value outside
  // upstream's own `{0, 1, 2}` is a malformed request and is still refused first,
  // and the knob still belongs to `DFRPipeline` alone.
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);

  auto refusal = [&](const char* value, const char* dir) {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + dir);
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = value;
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("temporal_upsample_rounds=" << value << " must be refused, never dropped");
      return std::string();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };

  SUBCASE("a value outside {0,1,2} gets upstream's OWN refusal first") {
    // `dfr_pipeline.py:284-285` raises at the top of `__call__`, before any work
    // is paid for. UNCHANGED by this row: a malformed request is still a
    // different answer from a legal one, and it still comes first.
    const std::string msg = refusal("3", "tr3");
    INFO(msg);
    CHECK(msg.find("must be 0, 1, or 2") != std::string::npos);
    CHECK(msg.find("dfr_pipeline.py:284-285") != std::string::npos);
  }

  SUBCASE("a legal count on a NON-DFR pipeline is refused, and says whose knob it is") {
    // THE BRANCH THAT BECAME REACHABLE IN THE SAME CHANGE THAT MADE IT TRUE.
    // While the loop was unported this check could not fire — every caller was
    // already refused by the "not served" message whatever their pipeline — so
    // landing it then would have been the unselected-branch shape
    // `.agents/reachability.md` enumerates. This engine is loaded on the DEFAULT
    // `distilled_two_stage` kind, so the refusal below is the pipeline one.
    const std::string msg = refusal("1", "trkind");
    INFO(msg);
    CHECK(msg.find("dfr_pipeline.py:277") != std::string::npos);
    CHECK(msg.find("distilled_two_stage") != std::string::npos);
    // And it says what silently dropping the knob would have cost, which is the
    // half that makes "refused rather than ignored" a reason instead of a slogan.
    CHECK(msg.find("2**rounds") != std::string::npos);
    // NOT the retired message. A caller must not be told the loop is unported.
    CHECK(msg.find("are not served") == std::string::npos);
  }

  SUBCASE("the canvas layout is still declared where the rounds loop reads it from") {
    // THE ASSERTION THE #920 REPAIR EXISTS FOR, RETARGETED rather than retired
    // with the message it used to guard. Its subject changed and its reason did
    // not: every other check in this file is on an UPSTREAM symbol name, and no
    // change to THIS tree can move one, so a suite built only on them stays green
    // through exactly the event that falsifies what we claim about ourselves.
    //
    // What it guarded before: a refusal claiming the canvas layout was ported
    // here. What it guards now: the rounds loop actually CALLING that layout.
    // Those four symbols were the first production callers this row added, so a
    // rename or a removal is no longer a stale sentence — it is a build failure
    // in `ltx2_video.cpp`. This check is what makes the failure say why, and it
    // reads a DIFFERENT file from the one that uses them, which is what stops it
    // being the tautology #911 records.
    const std::vector<std::string> header_lines =
        SplitLines(ReadSourceFile(LTX2_DFR_HEADER_PATH));
    REQUIRE(header_lines.size() > 100);
    std::string declarations;
    for (const std::string& line : header_lines) {
      const size_t first = line.find_first_not_of(" \t");
      if (first != std::string::npos && line.compare(first, 2, "//") == 0) continue;
      declarations += line;
      declarations += '\n';
    }
    // Positive control on the instrument: the header must still be mostly
    // declarations after stripping, or a reformat that turned it into one
    // comment block would answer every check below for free.
    REQUIRE_MESSAGE(declarations.find("struct Ltx2DfrTileRange") != std::string::npos,
                    "comment stripping removed the declarations it was meant to keep");
    for (const char* name : {"Ltx2DfrResolveCanvas", "Ltx2DfrTileRanges",
                             "Ltx2DfrStitchTileLatents", "Ltx2DfrMergeCarryForwardKeyframes",
                             "Ltx2DfrSliceLatentFrames", "Ltx2DfrConcatLatentFrames"}) {
      INFO("called by the rounds loop: " << name);
      CHECK_MESSAGE(declarations.find(name) != std::string::npos,
                    "the DFR rounds loop calls '"
                        << name
                        << "' and it is no longer declared in ltx2_dfr.h");
    }
  }
}

TEST_CASE("ltx2 video: DFR's temporal rounds DRIVE the temporal x2 latent upsampler") {
  // Row LTX25-DFR-ROUNDS (#986). THIS CASE IS THE ROW'S REACHABILITY PROOF, and
  // the reason the row exists at all.
  //
  // The temporal x2 latent upsampler has been ported, loader-parsed and gated
  // against executed upstream at reduced dimensions since #644, and until this
  // row NOTHING CALLED IT. `docs/FEATURES.md` carried that as
  // `Temporal x2 ups gated, UNDRIVEN`, and `.agents/reachability.md` names the
  // shape: a capability whose only driver is the test written for it. Its single
  // upstream consumer is DFR's rounds loop (`dfr_pipeline.py:407`), so porting
  // the loop is what retires the debt — and a unit test that hands the operator a
  // latent by hand proves the class works, never that anything reaches it.
  //
  // So every subcase here enters at `LoadVideoEngine` then `VideoEngine::Generate`
  // — the chain `vllm_video_generate` takes — and asks for the rounds through the
  // same `temporal_upsample_rounds` extra a caller has.
  //
  // WHY A FIXTURE CHECKPOINT, stated rather than left to be discovered: DFR's
  // base needs a `keyframe_slot_sft` transformer and NOBODY PUBLISHES ONE (#1137
  // established that by authenticated reads at the pinned revision), so `dfr` is
  // refused in practice on real weights. The reduced fixture below declares the
  // class it genuinely implements, which is not the false declaration
  // `docs/USAGE.md` warns about — that is declaring `keyframe_slot_sft` for a
  // `dev` or `distilled` FILE to get past the refusal. Every claim in this case
  // is therefore a reduced-dimension claim, and no real-weight temporal result
  // exists or is implied. The real temporal upsampler checkpoint IS on the NAS as
  // of 2026-08-17; the base it would run beside is what is missing.
  Workspace ws;

  // The TEMPORAL arm as its own checkpoint. DFR holds two upsamplers at once
  // upstream — `spatial_upsampler_path` (dfr_pipeline.py:174) for stage 2's input
  // transform and `temporal_upsampler_path` (:177) for the rounds — so this
  // engine has two slots rather than one reinterpreted one.
  vllm::Ltx2UpsamplerConfig temporal =
      ltx2_fixture::ReducedUpsamplerConfig(ltx2_fixture::ReducedDitParams().in_channels);
  temporal.spatial_upsample = false;
  temporal.temporal_upsample = true;
  const std::string temporal_path = ws.root + "/dfr_temporal_upsampler.safetensors";
  ltx2_fixture::WriteReducedUpsampler(temporal, temporal_path);

  auto load = [&](bool with_temporal) {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "dfr";
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("dfr");
    mp.extras["upsampler_path"] = ws.paths.upsampler;  // stage 2, the SPATIAL arm
    if (with_temporal) {
      mp.extras[vllm::multimodal::kLtx2TemporalUpsamplerPathExtra] = temporal_path;
    }
    return vllm::multimodal::LoadVideoEngine(mp);
  };

  SUBCASE("one round doubles the caller's frame contract AND invokes the upsampler") {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(true);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_r1");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "1";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    // THE CONTRACT: `(requested - 1) * 2**rounds + 1` (dfr_pipeline.py:534). The
    // fixture asks for 9, so one round owes 17 — and the canvas it was denoised
    // on is larger again, which is the half no output can show.
    CHECK(result.frame_count == 17);

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    CHECK(trace.temporal_rounds == 1);
    // THE REACHABILITY ASSERTION. Everything else in this subcase — the frame
    // count, the shape, the exit status — is satisfied by a loop that resized the
    // latent and never called the operator. This is the only line that separates
    // "the rounds ran" from "the rounds drove the upsampler", which is exactly
    // the distinction `.agents/reachability.md` says a green gate cannot make for
    // you. Deleting the `Ltx2UpsampleVideoLatent` call in the rounds loop must
    // turn this red; that mutation is the row's headline evidence.
    CHECK(trace.temporal_upsample_calls == 1);
    // (:415) `2**round_idx` windows — AND THE CLAMP, which this expectation got
    // wrong on the first pass and which is worth recording rather than quietly
    // fixing. `tile_ranges` takes `min(num_tiles, n_segments)`
    // (dfr_layout.py:171), and this fixture's canvas has exactly ONE segment: a
    // 9-frame request pads to 25 with its single keyframe position at 24, so the
    // boundaries are [0, 24] and round 1 asks for 2 windows over 1 segment.
    //
    // So the value here is 1, and it is upstream's value rather than a shortfall.
    // STATED PLAINLY BECAUSE IT BOUNDS THE CLAIM: this fixture cannot exercise
    // the UNCLAMPED `2**round` tiling, and no assertion in this file gates it.
    // What is gated is that the count comes from the layout at all — a loop that
    // denoised the canvas whole reports 1 here too, which is why the round-2
    // subcase below carries the assertion that separates them.
    REQUIRE(trace.round_tile_counts.size() == 1);
    CHECK(trace.round_tile_counts[0] == 1);
    // PLAYBACK fps doubles (:542). Reporting the conditioning fps instead hands
    // back a clip of the right length that plays at half speed, which no frame
    // count can see.
    CHECK(result.fps == 48);
  }

  SUBCASE("two rounds quadruple it, and the tile count doubles with the round") {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(true);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_r2");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "2";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    CHECK(result.frame_count == 33);  // (9 - 1) * 4 + 1

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    CHECK(trace.temporal_rounds == 2);
    // ONE UPSAMPLE PER ROUND, not one per render and not one per tile. A loop
    // that upsampled once and then re-tiled twice produces a clip of the right
    // length at half the temporal detail.
    CHECK(trace.temporal_upsample_calls == 2);
    // THE TILE COUNT GROWS WITH THE ROUND, which is the assertion that separates
    // a real re-tiling from a loop that denoises the canvas whole. Both values
    // are clamped by `min(num_tiles, n_segments)` (dfr_layout.py:171) — see the
    // round-1 subcase — but they are clamped to DIFFERENT numbers, and that is
    // what makes the pair informative where either alone is not.
    //
    // Derived rather than observed: round 1 runs on boundaries [0, 24] (one
    // segment, so `min(2, 1) = 1`) and invents a mid-segment slot at 12. Round 2
    // therefore carries TWO positions, its seams double to [48, 96] against a
    // 97-frame canvas, the boundaries become [0, 48, 96] — two segments — and it
    // asks for 4, taking `min(4, 2) = 2`. A loop that re-tiled without carrying
    // the round's new slots forward would report 1 again here.
    REQUIRE(trace.round_tile_counts.size() == 2);
    CHECK(trace.round_tile_counts[0] == 1);
    CHECK(trace.round_tile_counts[1] == 2);
    CHECK(trace.round_tile_counts[1] > trace.round_tile_counts[0]);
    CHECK(result.fps == 96);
  }

  SUBCASE("the CONDITIONING fps is capped at 60 while playback is not") {
    // dfr_pipeline.py:414 caps the conditioning fps at 60.0, and :74-78 says why
    // in terms: RoPE time is `pixel_frame / fps`, so a 120 fps time base halves
    // every token's temporal span against the trained distribution and the model
    // can no longer lay out 8 pixel frames inside one latent token — it decodes
    // as a motion spike at each latent border followed by a stall.
    //
    // THIS IS INVISIBLE TO EVERY OTHER ASSERTION IN THIS FILE. An uncapped port
    // renders the right number of frames, at the right size, at the right
    // playback rate, and reports success. The fixture runs at 24 fps, so round 1
    // conditions at 48 (under the cap, unchanged) and round 2 would reach 96 and
    // is held at 60 — which is what makes this two values rather than one.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(true);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_fps");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "2";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    REQUIRE(trace.round_conditioning_fps.size() == 2);
    CHECK(trace.round_conditioning_fps[0] == doctest::Approx(48.0));
    CHECK(trace.round_conditioning_fps[1] == doctest::Approx(60.0));
    // And PLAYBACK went past the cap, which is the other half of the claim: one
    // value capped and the other not. A port that capped both reports 60 here.
    CHECK(result.fps == 96);
  }

  SUBCASE("the four PER-TILE guarantees, none of which the rendered clip can show") {
    // Four values upstream's tile call overrides, mirrored in the rounds loop
    // and, until this subcase, gated by NOTHING. A fresh review mutated each one
    // and the full suite stayed green at 102 cases / 4141 assertions, which is
    // the shape `.agents/reachability.md` calls unasserted-live code: the lines
    // execute, they change the pixels, and no assertion in the tree reads them.
    //
    // They are gathered in one subcase because they are one render's worth of
    // observations about the same three tile invocations — round 1's single tile
    // and round 2's two — and separating them would run the same fixture three
    // more times to assert on different fields of the same trace.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(true);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_tileguards");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "2";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.frame_count == 33);  // the render still has to be the render

    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    // THREE tile invocations: round 1 clamps to 1 window, round 2 to 2. Read off
    // the tile counts rather than restated, so this stays the same claim if the
    // clamp ever changes.
    REQUIRE(trace.round_tile_counts.size() == 2);
    const size_t tiles = static_cast<size_t>(trace.round_tile_counts[0] + trace.round_tile_counts[1]);
    REQUIRE(tiles == 3);

    // ── 1. THE PER-TILE SEED (dfr_pipeline.py:496-498) ──────────────────────
    // `seed + 1000 * round + tile`, and the seed is what the loop generator was
    // CONSTRUCTED with rather than the offset that was asked for. Upstream names
    // the failure: the tiles are positionally identical, so a shared seed injects
    // byte-identical noise into every window and the clip renders with the same
    // grain pattern repeating across it — right length, right shape, exit 0.
    //
    // `FixtureGen` seeds at 7, so round 1's single tile owes 1007 and round 2's
    // two owe 2007 and 2008. THE LOAD-BEARING PART IS THAT THEY DIFFER, and the
    // `+ tile` term is the only thing that separates the last two: dropping it
    // leaves 2007 twice.
    REQUIRE(trace.round_tile_seeds.size() == tiles);
    CHECK(trace.round_tile_seeds[0] == 1007u);
    CHECK(trace.round_tile_seeds[1] == 2007u);
    CHECK(trace.round_tile_seeds[2] == 2008u);
    CHECK(trace.round_tile_seeds[2] != trace.round_tile_seeds[1]);

    // ── 2. THE ANCESTRAL ETA (dfr_pipeline.py:495) ──────────────────────────
    // Recorded inside the ancestral branch when it takes a step, so a tile that
    // fell onto the deterministic Euler arm contributes no entry at all and the
    // size check below is red before any value is compared. At eta 0 the
    // ancestral step degenerates to plain Euler: the round re-runs the denoiser
    // over the upsampled latent and injects no new detail, which is the entire
    // reason the round exists and is invisible in every count above.
    REQUIRE(trace.round_stepper_eta.size() == tiles);
    for (size_t i = 0; i < trace.round_stepper_eta.size(); ++i) {
      INFO("tile ", i);
      CHECK(trace.round_stepper_eta[i] == doctest::Approx(0.5));
    }

    // ── 3. THE SEAM ANCHOR STRENGTH (dfr_pipeline.py:466, :70-72) ───────────
    // 0.95, "pinned just short of fully clean so a tile can still settle its seam
    // frame". At 1.0 the seam frame is frozen and cannot move to meet the frames
    // either side of it, so each window border keeps a discontinuity — and the
    // token count, the tile count and the stitched length are all unchanged.
    //
    // The count is a floor rather than an equality: every tile pins at least its
    // own local frame 0 (:453-468), so fewer entries than tiles means a tile
    // denoised its window with no seam pinned at all.
    REQUIRE(trace.round_anchor_strengths.size() >= tiles);
    for (size_t i = 0; i < trace.round_anchor_strengths.size(); ++i) {
      INFO("anchor ", i);
      CHECK(trace.round_anchor_strengths[i] == doctest::Approx(0.95));
    }

    // ── 4. THE DEDUPE ORDER (dfr_pipeline.py:519-522) ───────────────────────
    // `first_index.setdefault(position, index)` — FIRST WINS. A lead-in segment
    // makes a later tile re-emit a slot an earlier tile already denoised, and the
    // earlier copy is the one denoised inside the window that OWNS the slot
    // rather than inside a lead-in.
    //
    // THE POSITIONS ARE IDENTICAL UNDER EITHER RULE. Both rules keep exactly one
    // entry per position and both sort the bag, so the position list, its length,
    // the carry-forward count and every downstream shape are byte-for-byte the
    // same whichever copy is kept. The winning TILE is the only observable the
    // rule moves, which is why it is recorded.
    //
    // This fixture: round 1 emits one slot at 24; round 2's tile 0 re-emits 24
    // and its tile 1 emits 24 again as a lead-in plus 72 of its own. Four
    // emissions, three survivors — so a duplicate genuinely occurred here and the
    // rule genuinely decided something.
    CHECK(trace.round_slots_emitted == 4);
    REQUIRE(trace.round_merged_slot_positions.size() == 3);
    CHECK(trace.round_slots_emitted > static_cast<int64_t>(trace.round_merged_slot_positions.size()));
    CHECK(trace.round_merged_slot_positions[0] == 24);
    CHECK(trace.round_merged_slot_positions[1] == 24);
    CHECK(trace.round_merged_slot_positions[2] == 72);
    // The assertion the rule lives or dies on. Under last-wins this reads
    // `0 1 1`.
    REQUIRE(trace.round_merged_slot_tiles.size() == 3);
    CHECK(trace.round_merged_slot_tiles[0] == 0);
    CHECK(trace.round_merged_slot_tiles[1] == 0);
    CHECK(trace.round_merged_slot_tiles[2] == 1);
  }

  SUBCASE("rounds without a temporal upsampler are refused, not silently skipped") {
    // Upstream's second refusal, raised at the top of `__call__` beside the
    // malformed-value one (dfr_pipeline.py:286-287). Refused rather than run at
    // 0 rounds because the rounds are what set the frame count: a skipped round
    // returns a clip a fraction of the requested length, and nothing else about
    // the render can flag it.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(false);
    REQUIRE(engine != nullptr);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_nots");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "1";
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("rounds without a temporal upsampler must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("temporal_upsampler_path") != std::string::npos);
      CHECK(msg.find("dfr_pipeline.py:286-287") != std::string::npos);
    }
  }

  SUBCASE("zero rounds is unchanged, which is the refactor's floor") {
    // THE CONTROL, and it matters more than any assertion above it. Landing the
    // rounds meant turning this engine's inline per-phase denoise into a callable
    // the tiles could enter — a refactor of the shipped render path that EVERY
    // pipeline kind now runs through. A rounds loop that works while quietly
    // moving the base path is a worse outcome than no rounds loop.
    //
    // At 0 rounds the seam must be a no-op: same frame count, same canvas, same
    // slot grid, and no upsample call. The wider control is the other 100 cases
    // in this file plus `test_ltx2_pipeline`, `test_ltx2_retake` and
    // `test_ltx2_dfr`, all of which run through the new seam unchanged.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine = load(true);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/dfr_r0");
    gen.extras[vllm::multimodal::kLtx2TemporalRoundsExtra] = "0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);

    CHECK(result.frame_count == 9);  // (9 - 1) * 2**0 + 1
    CHECK(result.fps == 24);
    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
    CHECK(trace.temporal_rounds == 0);
    // The upsampler is LOADED and still not called, which is the distinction
    // between "the checkpoint is present" and "the rounds ran".
    CHECK(trace.temporal_upsample_calls == 0);
    CHECK(trace.round_tile_counts.empty());
    CHECK(trace.canvas_frames == 25);  // the canvas still pads, exactly as before
  }
}

TEST_CASE("ltx2 video: an image at crf 0 conditions the render, and the ENCODER weights are read") {
  // The arm row LTX25-IMAGE-COND (#644) opened. Two separate claims are made
  // here and they are NOT the same claim:
  //
  //   1. the conditioning REACHES the render — the trace reports the encoded
  //      image, and a different image gives a different digest; and
  //   2. the ENCODER WEIGHTS are READ — perturbing ONE encoder tensor in the
  //      checkpoint moves the digest, with every byte of the REQUEST identical.
  //
  // (2) is the one that is easy to fake. A path that loaded the weights and then
  // conditioned on something else — zeros, the raw pixels, a re-used decoder
  // tensor — satisfies (1) completely.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);

  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/img");
  gen.first_frame_ppm = ConditioningPpm(20, 28, 1);
  gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
  const vllm::multimodal::VideoResult result = engine->Generate(gen);

  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
  CHECK(trace.completed);
  CHECK(trace.image_crf == 0);
  CHECK(trace.image_strength == 1.0);  // noise_aug defaults to 1.0 => the frame is PINNED
  // WHICH PHASE this describes is the claim, and `image_tokens > 0` did not make
  // it. MEASURED: changing the guard to `wants_image && phase_index == 0` — the
  // shape of an obvious refactor that hoists the per-phase decode+encode out of
  // the loop — left this whole binary at 32 cases / 550 assertions / exit 0
  // while `refine`, the phase whose latent is actually rendered, ran with the
  // pinned frame re-noised away. The design's own reason for living inside the
  // loop (spec section 8.5) was gated by nothing.
  //
  // So the count is pinned to the LAST phase's per-latent-frame token count.
  // This fixture's two-stage recipe runs `generate_lowres` at
  // `spatial_downscale = 2` and `refine` at 1, so the latent grid doubles in
  // each spatial dimension and the placed count is 1 then 4 — a per-phase value,
  // which is exactly why `image_tokens == 4` falsifies a stage-1-only build.
  constexpr int64_t kRefineImageTokens = 4;
  CHECK(trace.image_tokens == kRefineImageTokens);
  CHECK(trace.image_digest != 0);
  // A conditioning that collapsed to zeros would give every image the same
  // digest and still satisfy every check below it, so the magnitude is asked for
  // separately — the same reason `video_absmax` exists next to `video_digest`.
  CHECK(trace.image_absmax > 0.0);
  // The render still produced its artifacts; conditioning is not a bypass.
  CHECK(result.frame_count == 9);

  SUBCASE("the trace describes the LAST phase, and stage 1 is a different count") {
    // The other half of the same claim, and the half a literal cannot make: the
    // number above is not a constant of the fixture, it TRACKS the phase that
    // ran last. Same request, capped at phase 0, must report stage 1's smaller
    // count — and the ratio is checked between two MEASURED values rather than
    // between two compile-time constants, which would assert nothing.
    // `max_phase` is a LOAD-time extra, not a per-generation one, so the cap
    // needs its own engine over the same fixture.
    vllm::multimodal::VideoModelParams capped = ConditioningParams(ws.paths);
    capped.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
    const std::unique_ptr<vllm::multimodal::VideoEngine> stage1_engine =
        vllm::multimodal::LoadVideoEngine(capped);
    auto* stage1_ltx2 =
        dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(stage1_engine.get());
    REQUIRE(stage1_ltx2 != nullptr);
    vllm::multimodal::VideoGenParams lowres = FixtureGen(ws.root + "/img_stage1");
    lowres.first_frame_ppm = ConditioningPpm(20, 28, 1);
    lowres.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    (void)stage1_engine->Generate(lowres);
    const vllm::multimodal::Ltx2ConditioningTrace stage1 = stage1_ltx2->last_conditioning();
    CHECK(stage1.image_tokens == 1);
    CHECK(trace.image_tokens == 4 * stage1.image_tokens);
    // And it is a DIFFERENT encode, not the same one carried forward: the image
    // is resized and encoded against each phase's own height and width
    // (ltx-pipelines/utils/helpers.py:274-275, per-stage h/w at
    // distilled.py:251, 255-256, 285-286).
    CHECK(stage1.image_digest != trace.image_digest);
  }

  SUBCASE("a DIFFERENT image is a different conditioning") {
    vllm::multimodal::VideoGenParams other = FixtureGen(ws.root + "/img2");
    other.first_frame_ppm = ConditioningPpm(20, 28, 2);
    other.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    (void)engine->Generate(other);
    CHECK(ltx2->last_conditioning().image_digest != trace.image_digest);
  }

  SUBCASE("the SAME image is the same conditioning") {
    vllm::multimodal::VideoGenParams again = FixtureGen(ws.root + "/img3");
    again.first_frame_ppm = ConditioningPpm(20, 28, 1);
    again.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    (void)engine->Generate(again);
    CHECK(ltx2->last_conditioning().image_digest == trace.image_digest);
  }

  SUBCASE("the ENCODER's own weights decide the conditioning") {
    // ONE tensor of the encoder half, perturbed in a SECOND fixture, with the
    // request byte-identical. If the engine were conditioning on anything but
    // the encoder's output — or had loaded the DECODER's tensors under the
    // encoder's names — this digest would not move.
    Workspace mutated;
    const std::string path = mutated.paths.video_vae;
    std::string bytes = ReadAll(path);
    // The PAYLOAD is what gets perturbed, and its position is READ from the
    // safetensors header rather than guessed at. An earlier revision of this
    // case searched for the tensor's NAME and flipped a byte a fixed distance
    // past it, which lands inside the JSON header of whatever tensor happens to
    // be stored next — the file still parsed, the render still ran, and the
    // digest did not move. The case failed, which is the only reason that is a
    // footnote and not a false green.
    REQUIRE(bytes.size() > 8);
    uint64_t header_len = 0;
    std::memcpy(&header_len, bytes.data(), sizeof(header_len));
    REQUIRE(8 + header_len <= bytes.size());
    const nlohmann::json header =
        nlohmann::json::parse(bytes.substr(8, static_cast<size_t>(header_len)));
    const std::string needle = "encoder.conv_in.conv.weight";
    REQUIRE_MESSAGE(header.contains(needle),
                    "the fixture must carry an encoder half for this to prove anything");
    const size_t data_start =
        8 + static_cast<size_t>(header_len) +
        header.at(needle).at("data_offsets").at(0).get<size_t>();
    REQUIRE(data_start + 1 < bytes.size());
    // bf16 is stored little-endian, so byte 0 of a word carries the mantissa's
    // top bits; flipping 0x40 there moves that ONE weight by ~50% without any
    // risk of manufacturing an Inf or a NaN out of the exponent — which would
    // change the digest for a reason that has nothing to do with this claim.
    bytes[data_start] = static_cast<char>(bytes[data_start] ^ 0x40);
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      REQUIRE(out.good());
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    const std::unique_ptr<vllm::multimodal::VideoEngine> other =
        vllm::multimodal::LoadVideoEngine(ConditioningParams(mutated.paths));
    auto* other_ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(other.get());
    REQUIRE(other_ltx2 != nullptr);
    vllm::multimodal::VideoGenParams same = FixtureGen(mutated.root + "/img");
    same.first_frame_ppm = ConditioningPpm(20, 28, 1);
    same.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    (void)other->Generate(same);
    CHECK(other_ltx2->last_conditioning().image_digest != trace.image_digest);
  }
}

TEST_CASE("ltx2 video: a request WITHOUT an image leaves the trace's image fields empty") {
  // Otherwise "this render was conditioned on an image" and "this render was
  // not" would be indistinguishable after the fact, which is the one question
  // `Ltx2ConditioningTrace` exists to answer.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  (void)engine->Generate(FixtureGen(ws.root + "/plain"));
  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
  CHECK(trace.completed);
  CHECK(trace.image_tokens == 0);
  CHECK(trace.image_digest == 0);
  CHECK(trace.image_absmax == 0.0);
  CHECK(trace.image_strength == 0.0);
}


// ─── the floor under everything above ───────────────────────────────────────
//
// THIS CASE EARNED ITS PLACE BY FIRING. The fixture's first FP8 encoder let a
// value below 2^-6 fall through its subnormal branch into the normal one with a
// mantissa under 1.0; the resulting negative 3-bit fraction OR'd into the byte,
// and 0xFF is one of E4M3's two NaNs. 460 of 40,452 dequantized weights came
// back NaN, the DiT emitted NaN, and every geometry assertion above still
// passed: the PPM writer clamps, and `std::max(-1.0, NaN)` is -1.0, so the
// render serialized as nine perfectly well-formed uniformly black frames.
//
// It is a gate on the FIXTURE as much as on the loader, and that is the point —
// a fixture that cannot tell a right implementation from a wrong one is the same
// defect wearing different clothes.
TEST_CASE("ltx2 video: the loaded DiT and its forward are finite, weight by weight") {
  Workspace ws;
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(ws.paths.dit);
  vllm::Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;
  const vllm::Ltx2DitCheckpoint checkpoint =
      vllm::Ltx2LoadDitFromSafetensors(file, options);

  int64_t total = 0, nans = 0;
  for (const auto& entry : checkpoint.views) {
    const vt::Tensor& view = entry.second;
    const float* values = view.Ptr<float>();
    for (int64_t i = 0; i < view.Numel(); ++i) {
      ++total;
      if (!(values[i] == values[i])) ++nans;
    }
  }
  INFO("dequantized weights: " << total);
  REQUIRE(total > 0);
  CHECK_MESSAGE(nans == 0, nans, " of ", total, " dequantized DiT weights are NaN");

  // And the forward over them, at the geometry the engine drives it with.
  const vllm::Ltx2DitParams& params = checkpoint.params;
  const int64_t video_tokens = 2, audio_tokens = 4, context_tokens = 3;
  std::vector<float> video_latent(video_tokens * params.in_channels, 0.1F);
  std::vector<float> audio_latent(audio_tokens * params.audio_in_channels, 0.1F);
  std::vector<float> video_timesteps(video_tokens, 1.0F);
  std::vector<float> audio_timesteps(audio_tokens, 1.0F);
  float sigma = 1.0F;
  std::vector<double> video_positions(3 * video_tokens * 2);
  std::vector<double> audio_positions(audio_tokens * 2);
  for (int64_t d = 0; d < 3; ++d) {
    for (int64_t t = 0; t < video_tokens; ++t) {
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2)] = static_cast<double>(t);
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2 + 1)] =
          static_cast<double>(t + 1);
    }
  }
  for (int64_t t = 0; t < audio_tokens; ++t) {
    audio_positions[static_cast<size_t>(t * 2)] = static_cast<double>(t);
    audio_positions[static_cast<size_t>(t * 2 + 1)] = static_cast<double>(t + 1);
  }
  std::vector<float> video_context(context_tokens * params.cross_attention_dim, 0.05F);
  std::vector<float> audio_context(context_tokens * params.audio_cross_attention_dim, 0.05F);

  vllm::Ltx2ModalityInput video;
  video.tokens = video_tokens;
  video.context_tokens = context_tokens;
  video.latent = video_latent.data();
  video.timesteps = video_timesteps.data();
  video.sigma = &sigma;
  video.positions = video_positions.data();
  video.context = video_context.data();
  vllm::Ltx2ModalityInput audio;
  audio.tokens = audio_tokens;
  audio.context_tokens = context_tokens;
  audio.latent = audio_latent.data();
  audio.timesteps = audio_timesteps.data();
  audio.sigma = &sigma;
  audio.positions = audio_positions.data();
  audio.context = audio_context.data();

  const vllm::Ltx2DitOutputs out = vllm::Ltx2DitForward(
      vt::Device{}, params, checkpoint.weights, &video, &audio, vt::DType::kF32);
  REQUIRE(out.video.size() == static_cast<size_t>(video_tokens * params.out_channels));
  REQUIRE(out.audio.size() == static_cast<size_t>(audio_tokens * params.audio_out_channels));
  int64_t video_nans = 0, audio_nans = 0;
  for (const float v : out.video) {
    if (!(v == v)) ++video_nans;
  }
  for (const float v : out.audio) {
    if (!(v == v)) ++audio_nans;
  }
  CHECK(video_nans == 0);
  CHECK(audio_nans == 0);
}

// ─── reachable through include/vllm.h, which is the actual "done" bar ───────
//
// AGENTS.md §"Shared seams": a capability that is not reachable through the
// shared surface is not done, and for this project the shared surface is the C
// ABI. L1 made the video slice family-generic at ABI v18 (`family`,
// `extra_keys` / `extra_values`, `vllm_video_engine_family`), so registering a
// family is supposed to be all it takes — this asserts that it IS, by driving a
// whole generation through the ABI and nothing else.
TEST_CASE("ltx2 video: an ABI client loads, detects and generates through vllm.h") {
  Workspace ws;
  const std::string audio_embeds = ws.paths.audio_embeds;
  const std::string max_phase = "0";
  // The CHECKPOINT CLASS rides the ABI like every other load extra (row
  // LTX25-CHECKPOINT-CLASS, #1137). No kind is declared here, so the recipe is
  // `distilled_two_stage`, whose upstream row is `Distilled only`.
  const std::string checkpoint_class = FixtureCheckpointClass("distilled_two_stage");
  const char* keys[] = {vllm::multimodal::kLtx2AudioPromptEmbedsExtra,
                        vllm::multimodal::kLtx2MaxPhaseExtra,
                        vllm::multimodal::kLtx2CheckpointClassExtra};
  const char* values[] = {audio_embeds.c_str(), max_phase.c_str(), checkpoint_class.c_str()};

  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = ws.paths.dit.c_str();
  mp.video_vae_path = ws.paths.video_vae.c_str();
  mp.audio_vae_path = ws.paths.audio_vae.c_str();
  mp.prompt_embeds_path = ws.paths.video_embeds.c_str();
  mp.extra_keys = keys;
  mp.extra_values = values;
  mp.n_extras = 3;
  mp.device = 0;  // the family refuses 1 by name; the ABI carries the refusal

  vllm_video_engine* engine = nullptr;
  const vllm_status loaded = vllm_video_engine_load(&mp, &engine);
  const std::string load_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(load_error);
  REQUIRE(loaded == VLLM_OK);
  REQUIRE(engine != nullptr);
  // Detection ran through the ABI, with no `family` declared.
  REQUIRE(vllm_video_engine_family(engine) != nullptr);
  CHECK(std::string(vllm_video_engine_family(engine)) == vllm::multimodal::kLtx2VideoFamily);

  const std::string out_dir = ws.root + "/abi_out";
  vllm_video_params gen = vllm_video_params_default();
  gen.width = 64;
  gen.height = 64;
  gen.num_frames = 9;
  gen.seed = 7;
  gen.has_seed = 1;
  gen.output_dir = out_dir.c_str();

  vllm_video_result result;
  std::memset(&result, 0, sizeof(result));
  const vllm_status generated = vllm_video_generate(engine, &gen, &result);
  const std::string gen_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(gen_error);
  REQUIRE(generated == VLLM_OK);
  CHECK(result.frame_count == 9);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.sample_rate == 48000);
  REQUIRE(result.mux_argc > 0);
  REQUIRE(result.mux_argv != nullptr);
  CHECK(std::string(result.mux_argv[0]) == "ffmpeg");
  // execvp-ready: the argv is NULL-terminated past mux_argc.
  CHECK(result.mux_argv[result.mux_argc] == nullptr);
  const std::string first_frame = out_dir + "/frame_000000.ppm";
  std::ifstream frame(first_frame, std::ios::binary);
  CHECK_MESSAGE(frame.good(), "the ABI render wrote no ", first_frame);

  // The artifact INVENTORY, printed rather than only asserted: a phase whose
  // deliverable is "frames + a WAV + the mux argv the caller execs" owes the
  // reader those three things, not a claim that they were checked.
  std::string joined;
  for (int32_t i = 0; i < result.mux_argc; ++i) {
    joined += (i == 0 ? "" : " ") + std::string(result.mux_argv[i]);
  }
  const std::string wav_bytes = ReadAll(result.audio_path);
  const double wav_seconds =
      static_cast<double>(wav_bytes.size() - 44) / (2.0 * 2.0 * static_cast<double>(result.sample_rate));
  MESSAGE("ABI artifacts: family=" << std::string(vllm_video_engine_family(engine)) << " frames="
          << result.frame_count << " " << result.width << "x" << result.height << " fps="
          << result.fps << " wav=" << wav_bytes.size() << "B " << wav_seconds << "s @"
          << result.sample_rate << "Hz dir=" << std::string(result.frame_dir));
  MESSAGE("ABI mux argv: " << joined);

  vllm_video_result_free(&result);
  vllm_video_engine_free(engine);
}


// ─── W0 of LTX25-DEVICE-RESIDENCY: the phase instrument (issue #1010) ────────
//
// WHAT THIS GATES, and why the campaign put it first. Before this case the whole
// LTX-2.5 render path emitted ONE line per render — the example's `wrote N
// frames` — so a run that spent 1731 s somewhere could not say WHERE
// (#1087 measures that phase and says in its own text "Do not guess it from the
// duration"), and #1024's GPU-zero window is known NOT to be the denoise
// (`ltx2_video.cpp` selects `Ltx2StreamDitToDevice` / `Ltx2DitForwardDevice` on
// `on_device`) and NOT to be the decode (no frame had been written). A campaign
// staged on that profile would be spending months against a subject nobody had
// named. `.agents/specs/ltx25-device-residency.md` §Gates, stage W0.
//
// WHAT MAKES IT NON-VACUOUS, which is the whole difficulty of a "the parts sum
// to the whole" gate. A single leaf called `render` covering everything would
// sum to wall exactly and measure nothing, so the sum is only half of what is
// asserted here: the table must ALSO carry the named boundaries below (the DiT
// load, the denoise, the video decode, the frame writer). One without the other
// is satisfiable by an instrument that names nothing, or by an instrument that
// names everything and misses the time between the names.
//
// THE TOLERANCE IS STATED BEFORE THE RUN, per the spec's §Gates: the named
// leaves must account for at least 95% of the instrumented wall. The residue is
// emitted as `unaccounted_seconds` rather than distributed over the leaves,
// because a phase table whose parts do not add up is a table with a phase nobody
// named, and the spec's stop condition for W0 is that a partial table is worse
// than none.
//
// It runs over the REDUCED fixture on the CPU arm, so its `peak_device_bytes`
// column is the "no device probe on this arm" sentinel (-1). What that column
// carries on a device render is owed to W1's lease; what is gated here is that
// every entry carries a monotone timestamp and a host byte count, which is what
// #1010 asks for.
TEST_CASE("ltx2 video: a render through the ABI emits a phase table that SUMS to wall") {
  Workspace ws;
  const std::string audio_embeds = ws.paths.audio_embeds;
  const std::string max_phase = "0";
  // The CHECKPOINT CLASS rides the ABI like every other load extra (row
  // LTX25-CHECKPOINT-CLASS, #1137). No kind is declared here, so the recipe is
  // `distilled_two_stage`, whose upstream row is `Distilled only`.
  const std::string checkpoint_class = FixtureCheckpointClass("distilled_two_stage");
  const char* keys[] = {vllm::multimodal::kLtx2AudioPromptEmbedsExtra,
                        vllm::multimodal::kLtx2MaxPhaseExtra,
                        vllm::multimodal::kLtx2CheckpointClassExtra};
  const char* values[] = {audio_embeds.c_str(), max_phase.c_str(), checkpoint_class.c_str()};

  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = ws.paths.dit.c_str();
  mp.video_vae_path = ws.paths.video_vae.c_str();
  mp.audio_vae_path = ws.paths.audio_vae.c_str();
  mp.prompt_embeds_path = ws.paths.video_embeds.c_str();
  mp.extra_keys = keys;
  mp.extra_values = values;
  mp.n_extras = 3;
  mp.device = 0;

  vllm_video_engine* engine = nullptr;
  const vllm_status loaded = vllm_video_engine_load(&mp, &engine);
  const std::string load_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(load_error);
  REQUIRE(loaded == VLLM_OK);
  REQUIRE(engine != nullptr);

  const std::string out_dir = ws.root + "/phase_out";
  vllm_video_params gen = vllm_video_params_default();
  gen.width = 64;
  gen.height = 64;
  gen.num_frames = 9;
  gen.seed = 7;
  gen.has_seed = 1;
  gen.output_dir = out_dir.c_str();

  vllm_video_result result;
  std::memset(&result, 0, sizeof(result));
  const vllm_status generated = vllm_video_generate(engine, &gen, &result);
  const std::string gen_error = vllm_last_error() == nullptr ? "" : vllm_last_error();
  INFO(gen_error);
  REQUIRE(generated == VLLM_OK);

  // (1) THE FILE. The deliverable is retrievable evidence, not a line on a
  // console somebody has to have been watching (#1040 is made of exactly the
  // evidence that existed only on a host that stopped answering), so the shipped
  // default WRITES it beside the frames it explains.
  const std::string log_path = out_dir + "/phase-log.json";
  std::ifstream probe(log_path, std::ios::binary);
  REQUIRE_MESSAGE(probe.good(), "the render wrote no phase table at ", log_path);
  probe.close();
  const nlohmann::json table = nlohmann::json::parse(ReadAll(log_path));

  // THE CAVEAT TRAVELS WITH THE FILE. A phase log is read by somebody who did
  // not run it, months later, out of a directory named `benchmarks/`; the row's
  // whole subject is evidence that got separated from its context. The emitter
  // therefore writes what it knows about how far these numbers carry, and this
  // is the assertion that keeps it there.
  REQUIRE(table.contains("notice"));
  CHECK(table["notice"].get<std::string>().find("NOT A BENCHMARK") != std::string::npos);
  REQUIRE(table.contains("sum_rule"));
  CHECK(!table["sum_rule"].get<std::string>().empty());
  REQUIRE(table.contains("sampler_enabled"));

  REQUIRE(table.contains("wall_seconds"));
  REQUIRE(table.contains("phases"));
  REQUIRE(table["phases"].is_array());
  REQUIRE(!table["phases"].empty());
  const double wall = table["wall_seconds"].get<double>();
  CHECK(wall > 0.0);

  // (2) EVERY ENTRY carries a monotone timestamp and a byte count. What the
  // non-decreasing `start` sequence pins is the EMITTER'S SORT, and this comment
  // used to say the opposite — that the sequence holds *because* the records are
  // appended in completion order. Completion order is what would BREAK it: the
  // `load` span closes at start 0.0001 and is appended AFTER
  // `load.prompt_embeds`, which starts at 0.0625, so the appended order is not
  // monotone in `start` at all. `ByStart`'s `stable_sort` in
  // `render_phase_log.cpp` is what makes it monotone, and deleting that sort is
  // what this line detects.
  double previous_start = -1.0;
  double leaves = 0.0;
  std::vector<std::string> names;
  for (const nlohmann::json& entry : table["phases"]) {
    REQUIRE(entry.contains("name"));
    REQUIRE(entry.contains("start_seconds"));
    REQUIRE(entry.contains("end_seconds"));
    REQUIRE(entry.contains("peak_host_bytes"));
    REQUIRE(entry.contains("peak_device_bytes"));
    const std::string name = entry["name"].get<std::string>();
    const double start = entry["start_seconds"].get<double>();
    const double end = entry["end_seconds"].get<double>();
    INFO("phase = " << name);
    CHECK(!name.empty());
    CHECK_MESSAGE(start >= previous_start,
                  "the table is emitted as a TIMELINE, sorted by start; the records are "
                  "APPENDED in completion order, and an emitter that forgot to sort would "
                  "hand a reader a table whose rows interleave two renders");
    CHECK(start <= wall);
    CHECK(end <= wall + 1e-6);
    CHECK(end >= start);
    previous_start = start;
    // A byte count, not a placeholder: this process is resident and every phase
    // of it holds the fixture's weights, so a zero here is an instrument that
    // read nothing rather than a phase that allocated nothing.
    CHECK(entry["peak_host_bytes"].get<int64_t>() > 0);
    CHECK(entry["peak_device_bytes"].get<int64_t>() >= -1);
    names.push_back(name);
    // SPAN **AND** NESTED, which is what `render_phase_log.cpp`'s own `Sum`
    // skips. Skipping only spans here made this loop silently assert something
    // else — "no nested leaf has a non-trivial duration" — although the header
    // advertises nesting as supported and recorded. A fresh review split the mel
    // decode out as a nested leaf and this case failed the reconciliation below
    // with `CHECK( 0.0118791 < 1e-06 )`, a message that says nothing about
    // nesting and reads as "the emitter does not reconcile".
    if (entry.value("span", false) || entry.value("nested", false)) continue;
    leaves += end - start;
  }

  // (3) THE NAMED BOUNDARIES. Without these the sum below is satisfied by one
  // leaf called `render`, which is an instrument that measures a stopwatch.
  for (const std::string required : {"load.dit", "denoise", "decode.video", "decode.audio",
                                    "artifacts.frames", "artifacts.audio"}) {
    // `std::string`, not `const char*`: doctest stringifies a `char*` streamed
    // into a message as a BOOL, so a failing case here would have printed
    // "names no '1' phase" and said nothing about which name was missing.
    CHECK_MESSAGE(std::find(names.begin(), names.end(), required) != names.end(),
                  "the phase table names no '" << required << "' phase");
  }

  // (4) THE SUM. Stated as the tolerance the spec fixed before the run: the
  // named leaves account for >= 95% of the instrumented wall, and whatever is
  // left is reported as its own quantity rather than smeared over the leaves.
  REQUIRE(table.contains("unaccounted_seconds"));
  const double unaccounted = table["unaccounted_seconds"].get<double>();
  MESSAGE("phase table: wall=" << wall << "s leaves=" << leaves << "s unaccounted="
                               << unaccounted << "s over " << names.size() << " entries");
  CHECK(std::fabs((wall - leaves) - unaccounted) < 1e-6);
  CHECK_MESSAGE(leaves >= 0.95 * wall,
                "the phase table does not sum: " << leaves << "s of named leaves against " << wall
                                                 << "s of wall. The missing time is a phase "
                                                    "nobody named, and W0 iterates until it is");

  // (5) THE ABI CARRIES IT. `examples/ltx2_gen` is a client of `vllm.h` and
  // nothing else, so a client that never guesses a filename beside the frames
  // can still name the table it was handed.
  REQUIRE(vllm_video_last_phase_log(engine) != nullptr);
  CHECK(std::string(vllm_video_last_phase_log(engine)) == log_path);

  vllm_video_result_free(&result);
  vllm_video_engine_free(engine);
}

// ─── W0 repair: the table ATTRIBUTES its time, which a SUM cannot say ────────
//
// WHY A SECOND CASE, when the one above already passes. The case above pins two
// things: that six names EXIST, and that the leaves ADD UP to wall within 5%.
// A fresh review of this row's pull request built a table that satisfies both
// and is worse than useless. Leave the `decode.video` leaf open across the audio
// decode and give `decode.audio` its name with no work beneath it, and the case
// above reports **1 case passed, 273 assertions, 0 failed, exit 0** at 99.9%
// accounted — over a table that says the video decode is 32% of the render (it
// is 2.4%) and that the audio decode is free (it is a quarter of it). W1 ranks
// this campaign's levers off this table and W5 is its largest stage. Existence
// plus a sum is NECESSARY and it is not SUFFICIENT, because neither ties a NAME
// to the work beneath it.
//
// WHAT WAS TRIED FIRST AND REJECTED, MEASURED RATHER THAN ARGUED. The obvious
// repair is a differential: render at two frame counts and require the phases
// whose work scales with the clip to grow. It was written, built and run here,
// and **this box cannot carry it**. Two renders of the same binary, minutes
// apart, 9 frames against 33: the 9-frame render measured 8.03 s of leaves and
// the 33-frame render 3.80 s — the 2.5x LONGER clip cost HALF the time, and an
// earlier run of the same 9-frame render measured 4.80 s. Wall-clock noise here
// is a factor of two in both directions, against a 2.5x signal, so a
// seconds-differential is a coin flip wearing a gate. That is the same finding
// `## Outcome — W0` already records at 76x, and it is why W1 is written as a
// lease on an idle box.
//
// WHAT IS ASSERTED INSTEAD NEEDS NO CLOCK AT ALL. A phase's name is tied to its
// work STRUCTURALLY: a leaf that claims to cover a piece of work must ENCLOSE
// the scopes that sit on that work. Every number involved comes from one clock
// in one run, and every threshold is a ratio of two intervals measured inside
// the same contention window, so contention cannot move the verdict.
//
// AND IT IS ASSERTED FOR ALL THREE CARRYING PHASES, which is the second fresh
// review's finding. The first repair gave `decode.audio` two sub-scopes and left
// `denoise` and `decode.video` held by a share floor alone. The reviewer then ran
// M4's shape one level over, on the phase #1024 and #1087 are actually about:
// close `denoise` after the first sampler step and open `phase.finish` there. No
// overlap, no nesting, the sum untouched — **exit 0 on both cases, 99.94%
// accounted, and 82% of the denoise charged to a neighbour**, with the emitted
// table reporting `phase.finish` at 55.0% against a `denoise` of 6.0% on a run
// whose honest denoise was 73.4%. A gate blind to that is a gate this campaign
// cannot rank levers with, because W1 ranks them off this very table.
//
// AND A THIRD FRESH REVIEW BROKE IT AGAIN, one level over from that. All four
// assertions below are RATIOS taken against the sub-scope anchor, so an anchor
// that moves WITH the leaf defeats all four at once. Two mutations, both
// building clean, both exit 0 on both cases:
//
//   M12 emits `denoise.step` for `step_index == 0` only, closes `denoise` after
//       step 0 and opens `phase.finish` over steps 1-7. **exit 0, 106/106 and
//       350/350 assertions**, over a table reporting `phase.finish` at 28.73%
//       against a `denoise` of 6.11% on a run whose honest denoise was 39.3%.
//       That is M7's defect again, on M7's phase, through the gate M7 produced.
//   M11 moves the two `Close`s in the driver's chunk sink to after the PPM
//       write. **exit 0, 124/124 and 422/422**, coverage still 99.68%, with the
//       whole PPM writer charged to `decode.video` — the leaf W5's lever is
//       measured from — because a swallowed neighbour turns `nested`, leaves the
//       sum, and overlaps nothing.
//
// So the case now asserts the list below, over TWO renders. Read it by what each
// item is measured against, because that is what decides which defect it can
// see. (0), (1b) and (5) are measured against nothing the instrument controls —
// a count the render kept, an order the driver runs in, a flag the emitter sets
// — and they are the three that survive an anchor moving with its leaf. (1),
// (2), (3) and (4) are ratios against the anchor. (6) is a ratio between two
// anchors, which is a weaker guarantee than (0) and a stronger one than nothing,
// and nothing was what held it before.
//
//   0. RECORD COUNT   the anchor runs once per unit of work the RENDER counted.
//                     `denoise.step` once per `Ltx2ConditioningTrace::
//                     dit_evaluations`, `decode.video.chunk` once per
//                     `video_decode_chunks` plus the reopen after the last one,
//                     the audio pair once each. Neither counter is derived from
//                     the phase table, so no placement of a scope can move them.
//                     This is what M12 fails.
//
//   1. CONTAINMENT  every sub-scope interval lies inside one of the leaf's own
//                   records. `denoise.step` wraps the denoiser evaluation inside
//                   `Evaluate`, which every sampler arm reaches, so a `denoise`
//                   that stops short of its loop no longer contains its steps.
//                   `decode.video.chunk` ends when the streaming decoder hands a
//                   chunk BACK, which is a production event and not an
//                   instrument statement. `decode.audio.mel` and
//                   `decode.audio.vocoder` wrap the two calls #1010 names.
//  1b. SIBLING ORDER two sub-scopes of one leaf appear in the order the driver
//                   runs them, compared on each name's first record. NOTHING
//                   ELSE TIES A SUB-SCOPE NAME TO A POSITION — the counts are
//                   per name, containment holds for either, the intervals are
//                   sorted before the overlap check, and the nested check is a
//                   set membership — so swapping the `decode.audio.mel` and
//                   `decode.audio.vocoder` literals moved 96.8% of that leaf's
//                   decomposed seconds onto the wrong name with everything
//                   green. This is what N1 fails.
//   2. COVERAGE     the sub-scopes together account for nearly all of the leaf,
//                   so a leaf that encloses its own work PLUS a phase nobody
//                   named still fails. The threshold is per phase and is set
//                   from what the anchor can see, not from a round number.
//   3. EXCLUSIVITY  no OTHER leaf overlaps the window the sub-scopes span, apart
//                   from a partner the driver genuinely interleaves with
//                   (`artifacts.frames` alternates with `decode.video`, because
//                   the decoder streams chunks into the writer's callback). This
//                   is the assertion that catches the transfer directly: M7's
//                   `phase.finish` opens in the middle of the denoise window.
//   4. NON-OVERLAP  the carrying leaves and the writer never overlap each other.
//   5. NOTHING BUT AN ANCHOR IS NESTED, plus the writer is never enclosed by a
//                     `decode.video` leaf. A leaf that grows over a NEIGHBOUR
//                     does not overlap it and does not break the sum — the
//                     neighbour turns `nested` and leaves the timeline — so this
//                     is the assertion M11 fails, and it needs no clock.
//                     `decode.video` also gains `decode.video.vae`, opened
//                     inside `Ltx2ConvVideoDecodeTiled` around
//                     `AccumulateTemporalGroup`, so that leaf has one sub-scope
//                     whose ends are both production events rather than
//                     statements adjacent to its own.
//   6. VAE COVERAGE   `decode.video.vae` must cover at least half of this
//                     render's `decode.video.chunk` seconds. Until this landed
//                     the vae anchor was held by cardinality, `nested` and
//                     containment alone and its DURATION was compared against
//                     nothing, so moving it one statement up onto
//                     `buffer.Allocate` reported `decode.video.vae = 0.000 s`
//                     beside a five-millisecond `decode.video` with the tile
//                     decode inside no sub-scope, both gates green. N6.
//   7. THE COUNTS AT N > 1. The case renders twice — nine frames, which decode
//                     in one temporal chunk, and 81, which chunk — and runs
//                     assertions 0 through 6 over each render of one table.
//                     Every decode-side count was previously 1 or 1+1, which
//                     cannot distinguish "once per unit of work" from "once per
//                     render". Both lookups are filtered by the record's
//                     `render` field; see `RecordsNamed`.
//
// WHAT WAS TRIED FIRST AND REJECTED, MEASURED RATHER THAN ARGUED. The obvious
// repair is a differential: render at two frame counts and require the phases
// whose work scales with the clip to grow. It was written, built and run here,
// and **this box cannot carry it**. Two renders of the same binary, minutes
// apart, 9 frames against 33: the 9-frame render measured 8.03 s of leaves and
// the 33-frame render 3.80 s — the 2.5x LONGER clip cost HALF the time, and an
// earlier run of the same 9-frame render measured 4.80 s. Wall-clock noise here
// is a factor of two in both directions, against a 2.5x signal, so a
// seconds-differential is a coin flip wearing a gate. That is the same finding
// `## Outcome — W0` already records at 76x, and it is why W1 is written as a
// lease on an idle box.
//
// AND THE SHARE FLOOR STAYS, WITH ITS MARGIN NAMED RATHER THAN TIGHTENED. Each
// carrying phase must still hold at least 0.05% of the leaf sum. That is a
// backstop against a name with NOTHING beneath it — a detached leaf measures two
// function calls, about a microsecond, five orders of magnitude below it — and
// it is deliberately NOT tightened toward the measured share, because the
// measured share is exactly the quantity this box destroys. The same binary at
// the same geometry has reported `denoise` at 38.1% and at 73.4%,
// `decode.audio` at 50.9% and at 16.7%, and `decode.video` at 5.27% and at
// 2.13%. A floor set at "a fraction of the measured value" would be a flake, and
// under M7 `denoise` sat at 6.0% and cleared any floor a quiet box could
// justify. The floor is therefore a weak assertion ON PURPOSE, and the three
// assertions above are what now bind. Read the floor as "this name is not
// detached", never as "this phase is honest".
namespace {

// "any render", for a lookup that genuinely spans the whole table.
constexpr int64_t kAnyRender = -1;

// Every record of one name IN ONE RENDER, in emitted (start-sorted) order.
//
// THE RENDER ARGUMENT IS NOT DECORATION, and a fourth fresh review named the
// hole it closes. `PhaseLog` is a PROCESS timeline that resets at LOAD
// (`ltx2_video.cpp`, `Begin()` in the load path); `Ltx2ConditioningTrace` resets
// on every `Generate` (`ltx2_video.cpp`, `im.trace = Ltx2ConditioningTrace{}`).
// So a count taken over every record of a name is process-wide while every
// denominator it is compared against is per-render, and the two agreed only
// while this case rendered exactly once. It now renders twice into one table,
// and without this filter the second render's counts would be the sum of both.
std::vector<nlohmann::json> RecordsNamed(const nlohmann::json& table, const std::string& name,
                                         int64_t render = kAnyRender) {
  std::vector<nlohmann::json> out;
  for (const nlohmann::json& e : table["phases"]) {
    if (e["name"].get<std::string>() != name) continue;
    if (render != kAnyRender && e.value("render", static_cast<int64_t>(0)) != render) continue;
    out.push_back(e);
  }
  return out;
}

// The render index of the LAST generation in a table. The load is render 0 and a
// table written by the second `Generate` of a process holds the load's records
// and BOTH renders', so "this render" is a value the reader has to resolve
// rather than assume.
int64_t LastRender(const nlohmann::json& table) {
  int64_t last = 0;
  for (const nlohmann::json& e : table["phases"]) {
    last = std::max(last, e.value("render", static_cast<int64_t>(0)));
  }
  return last;
}

// Leaf seconds per NAME, summed: `decode.video` alternates with
// `artifacts.frames` and appears more than once per render, so a per-name total
// is the only quantity that means anything. Leaves only — spans enclose leaves
// and nested records decompose one, and the emitter's own `Sum` skips both.
std::map<std::string, double> LeafSecondsByName(const nlohmann::json& table) {
  std::map<std::string, double> out;
  for (const nlohmann::json& e : table["phases"]) {
    if (e.value("span", false) || e.value("nested", false)) continue;
    out[e["name"].get<std::string>()] += e["duration_seconds"].get<double>();
  }
  return out;
}

struct NamedInterval {
  std::string name;
  double start = 0.0;
  double end = 0.0;
};

// Every SUMMED leaf of one render as an interval, in start order.
std::vector<NamedInterval> LeafIntervals(const nlohmann::json& table,
                                         int64_t render = kAnyRender) {
  std::vector<NamedInterval> out;
  for (const nlohmann::json& e : table["phases"]) {
    if (e.value("span", false) || e.value("nested", false)) continue;
    if (render != kAnyRender && e.value("render", static_cast<int64_t>(0)) != render) continue;
    NamedInterval iv;
    iv.name = e["name"].get<std::string>();
    iv.start = e["start_seconds"].get<double>();
    iv.end = e["end_seconds"].get<double>();
    out.push_back(iv);
  }
  std::sort(out.begin(), out.end(),
            [](const NamedInterval& a, const NamedInterval& b) { return a.start < b.start; });
  return out;
}

// One carrying phase and the anchor that ties its name to its work.
struct Carrying {
  std::string leaf;                    // the name that claims the seconds
  std::vector<std::string> parts;      // the nested sub-scopes sitting on the work
  double min_coverage = 0.0;           // of the leaf's own seconds
  std::vector<std::string> partners;   // leaves the driver genuinely interleaves with
  // HOW MANY RECORDS OF EACH PART, from a number the RENDER produced. One entry
  // per name in `parts`. This is the only assertion here that is not a ratio
  // against the leaf, and it is why the other four are worth making — see the
  // note above this case.
  std::vector<int64_t> part_counts;
  std::string count_source;            // where those numbers came from, for the message
  // EACH PART'S OWN FLOOR, as a share of the LEAF's seconds. One entry per name
  // in `parts`, and 0.0 means "no floor, and the row's spec says which direction
  // that leaves open".
  //
  // WHY IT IS PER PART AND NOT ONE NUMBER. `min_coverage` above is checked on
  // the SUM of the parts, so ONE part covering everything satisfies it and the
  // others may measure nothing. That is not a hypothetical: mutation R1b leaves
  // `decode.audio.mel` open across the vocoder call and lets
  // `decode.audio.vocoder` open and close EMPTY beside it — identical semantics,
  // identical call order, two lines — and it moves ~100% of `decode.audio`'s
  // decomposed seconds onto the wrong name, which is the same end state the
  // NAME-SWAP mutation N1 produces. Every other assertion here survives it:
  // (1b) passes because mel still OPENS first, so it proves an ORDER and not an
  // IDENTITY; the counts are {1, 1} either way; containment, `nested` and
  // non-overlap all hold. This field is what makes the two names measurable
  // SEPARATELY.
  //
  // IT IS REQUIRED RATHER THAN DEFAULTED, and deliberately sits before `render`
  // so that a future leaf cannot acquire a second part without answering the
  // question. `decode.audio{mel, vocoder}` is the ONLY multi-part leaf today —
  // `denoise`, `decode.video` and `artifacts.frames` have one part each, which
  // is why (1b) and this assertion are both vacuous for them and why the hole
  // above existed on exactly one pair.
  std::vector<double> part_min_coverage;
  // WHICH RENDER. The table is a process timeline and every count above is a
  // per-render quantity; see `RecordsNamed`.
  int64_t render = kAnyRender;
};

// The four assertions above, for one carrying phase. A free function rather than
// three copies, because `decode.video` has N leaves and M sub-scopes while
// `decode.audio` has one of each, and a form that only handles the second is how
// `denoise` came to be ungated in the first place.
void CheckCarryingPhase(const nlohmann::json& table, const Carrying& c) {
  INFO("carrying phase = " << c.leaf);
  const std::vector<nlohmann::json> leaves = RecordsNamed(table, c.leaf, c.render);
  REQUIRE_MESSAGE(!leaves.empty(), "the table names no '" << c.leaf << "' leaf at all");
  double leaf_seconds = 0.0;
  for (const nlohmann::json& r : leaves) {
    REQUIRE_FALSE(r.value("span", false));
    REQUIRE_FALSE(r.value("nested", false));
    leaf_seconds += r["end_seconds"].get<double>() - r["start_seconds"].get<double>();
  }
  REQUIRE(leaf_seconds > 0.0);

  // (0) THE RECORD COUNT, and it is FIRST because it is the only one of the five
  // that is not a ratio against `leaf_seconds`. See the note above this case:
  // containment, coverage, exclusivity and non-overlap all measure the anchor
  // AGAINST the leaf, so an anchor that moves WITH the leaf satisfies all four
  // at once, and mutation M12 did exactly that — emit `denoise.step` for step 0
  // only, close `denoise` there, open `phase.finish` over the remaining seven
  // steps, and both cases in this file report exit 0 over a table that charges
  // 82% of the largest phase to its neighbour.
  //
  // The counts come from `Ltx2ConditioningTrace`, which the RENDER maintains: an
  // evaluation increments `dit_evaluations` inside the shared `Evaluate` lambda
  // and the driver's own sink increments `video_decode_chunks` beside
  // `rendered_frames`. Neither is derived from the phase table, so no placement
  // of a phase scope can move them, and an anchor that runs a different number
  // of times than the work it claims to wrap is caught whatever the clock did.
  REQUIRE(c.part_counts.size() == c.parts.size());
  for (size_t i = 0; i < c.parts.size(); ++i) {
    const int64_t emitted =
        static_cast<int64_t>(RecordsNamed(table, c.parts[i], c.render).size());
    INFO("sub-scope = " << c.parts[i]);
    CHECK_MESSAGE(emitted == c.part_counts[i],
                  "'" << c.parts[i] << "' was emitted " << emitted << " time(s) and the render "
                      << "did the work " << c.part_counts[i] << " time(s) (" << c.count_source
                      << "). An anchor that does not run once per unit of work it names is not "
                         "anchoring the name to the work — it is measuring whichever unit it "
                         "happened to be opened around");
  }

  // (1) CONTAINMENT, and the sub-scopes' own intervals.
  std::vector<std::pair<double, double>> subs;
  // ...and the FIRST start of each part name, in the order `parts` lists them,
  // which is assertion (1b) below.
  std::vector<double> part_first;
  // ...and each part's OWN seconds, kept separate from `subs` because `subs`
  // collapses every part into one pool and assertion (2b) is precisely the
  // question that pool cannot answer.
  std::vector<double> part_seconds;
  for (const std::string& part : c.parts) {
    INFO("sub-scope = " << part);
    const std::vector<nlohmann::json> found = RecordsNamed(table, part, c.render);
    double seconds = 0.0;
    REQUIRE_MESSAGE(!found.empty(),
                    "the table names no '" << part << "' record. It is the anchor that ties the '"
                        << c.leaf << "' name to the work beneath it, and without it this phase is "
                                     "held by a share floor alone");
    for (const nlohmann::json& r : found) {
      // Recorded as a decomposition, not as a partition: a nested record is
      // excluded from `sum_leaf_seconds`, which is what lets the split exist at
      // all without changing what the table adds up to.
      CHECK_MESSAGE(r.value("nested", false),
                    "'" << part << "' is not marked nested, so it is being SUMMED as well as the "
                                   "leaf that contains it, and the table double counts");
      const double start = r["start_seconds"].get<double>();
      const double end = r["end_seconds"].get<double>();
      bool inside = false;
      for (const nlohmann::json& leaf : leaves) {
        if (start >= leaf["start_seconds"].get<double>() - 1e-9 &&
            end <= leaf["end_seconds"].get<double>() + 1e-9) {
          inside = true;
        }
      }
      CHECK_MESSAGE(inside, "'" << part << "' runs [" << start << ", " << end << "] and NO '"
                                << c.leaf << "' leaf encloses it. The name and the work are in "
                                             "different places, which is the whole defect this "
                                             "case exists for");
      subs.push_back({start, end});
      seconds += end - start;
    }
    part_seconds.push_back(seconds);
    double first = found.front()["start_seconds"].get<double>();
    for (const nlohmann::json& r : found) {
      first = std::min(first, r["start_seconds"].get<double>());
    }
    part_first.push_back(first);
  }
  REQUIRE(!subs.empty());

  // (1b) THE SIBLING SUB-SCOPES RUN IN THE ORDER `parts` LISTS THEM, which is
  // the order the driver runs them in.
  //
  // WHY IT EXISTS, and it is the same defect assertion (6) was added for one
  // review earlier. NOTHING ABOVE TIES A SUB-SCOPE NAME TO A POSITION: the
  // counts are per name, containment holds for either name, `subs` is sorted by
  // start BEFORE the overlap check, and `CheckOnlyAnchorsAreNested` is a
  // set-membership test. So swapping the two string literals at the
  // `decode.audio.mel` / `decode.audio.vocoder` scopes in the driver — two
  // lines — moved 96.8% of `decode.audio`'s decomposed seconds onto the wrong
  // name with every assertion in this file green, on a leaf that held 47.8% of
  // that render's named leaf seconds and that #1010 asks for BY NAME. W1 would
  // have ranked the wrong model.
  //
  // WHAT IT PROVES AND WHAT IT DOES NOT. It proves that the names appear in the
  // driver's own order, so a swapped PAIR is a red. It does NOT prove that
  // either name covers the call it is named after; that would need an anchor
  // inside `Ltx2AudioDecoderForward` / `Ltx2VocoderWithBweForward`, which is the
  // same shape `load.dit` is owed and is recorded beside it in the row's spec
  // under `### Owed out of W0`. Compared on the FIRST record of each name, so a
  // part that legitimately repeats (`decode.video.chunk`) is unaffected, and the
  // assertion is vacuous for a leaf with a single part name.
  //
  // AND COMPARING FIRST RECORDS IS THE WHOLE ORDER ONLY BECAUSE THESE PARTS DO
  // NOT INTERLEAVE. `decode.audio.mel` and `decode.audio.vocoder` emit one
  // record each, so `first(A) <= first(B)` is exact. A future leaf with two
  // REPEATING parts would satisfy this assertion while running `A, B, B, A, B,
  // A` — the first pair is in order and every pair after it is not. That leaf
  // owes a per-record pairing, and this one does not have to pay for it: the
  // (0) count is what would make the difference visible, since `{1, 1}` is the
  // premise the line above rests on.
  for (size_t i = 1; i < part_first.size(); ++i) {
    INFO("sub-scope pair = " << c.parts[i - 1] << " then " << c.parts[i]);
    CHECK_MESSAGE(part_first[i] >= part_first[i - 1] - 1e-9,
                  "'" << c.parts[i] << "' first runs at " << part_first[i] << " and '"
                      << c.parts[i - 1] << "' at " << part_first[i - 1] << ", so the table "
                      << "reports the sub-scopes of '" << c.leaf << "' in an order the driver "
                         "does not run them in. Two SIBLING scope names that were swapped move "
                         "their seconds onto each other and change nothing else in this file");
  }

  std::sort(subs.begin(), subs.end());
  // The sub-scopes decompose one leaf, so they run in sequence and never overlap
  // each other. Two that did would double count into the coverage below.
  for (size_t i = 1; i < subs.size(); ++i) {
    CHECK_MESSAGE(subs[i].first >= subs[i - 1].second - 1e-9,
                  "two sub-scopes of '" << c.leaf << "' overlap: [" << subs[i - 1].first << ", "
                      << subs[i - 1].second << "] and [" << subs[i].first << ", " << subs[i].second
                      << "]");
  }

  // (1c) THE LEAF DOES NOT EXTEND BEYOND ITS ANCHORS, and this assertion is the
  // one that catches what (2) below was written for -- down to its own
  // resolution, which is stated here rather than left to be discovered.
  //
  // WHAT IT DOES NOT CATCH, MEASURED. At fixture scale this bounds a swallow at
  // 30 ms per leaf record and sees nothing smaller. A mutation that opens
  // `denoise` at the top of `phase.prepare`, so the whole prepare becomes
  // un-named time inside the leaf, moves the span slack from 26.6 us to 124.6 us
  // and the case stays GREEN, because `phase.prepare` is only about 125 us on
  // this fixture. The coverage floor (2) misses it too -- it moved 0.4 points --
  // so this is not a regression against the shape that shipped, but it IS the
  // limit of the instrument at this geometry, and `### Owed out of W0` is where
  // the leaf-by-leaf version lives.
  //
  // AND THE RESOLUTION IS NOW 30 ms RATHER THAN 0.25 ms, which is a real
  // loss and is the price of a bound that survives a two-core contended runner.
  // The 0.25 ms constant was inside its own honest distribution: at head
  // 6b48edb2c, unmutated, it reddened 3 of 33 runs on a shared
  // 20-core box and 2 of 10 under an added eight-way spin load.
  // Nothing about the tree moved between a green run and a red one; the
  // scheduler did. See `### The span-slack bound, fourth shape` in the row's
  // spec for the whole distribution.
  //
  // MEASURED, because the previous shape of this gate was argued rather than
  // measured and it decided by box load for two days (#1439, #1494). The
  // un-covered seconds inside a leaf are TWO structurally different things and
  // the old single share floor charged them to one number:
  //
  //   * the HEAD and TAIL -- the leaf's own seconds lying outside the span its
  //     sub-scopes occupy. This is exactly where a leaf that opened early over a
  //     phase nobody named, or stayed open over one, puts the whole swallow. It
  //     is TWO instrument boundaries, so it does not grow with the render and it
  //     does not grow with the step count.
  //   * the INTERIOR gaps -- the time BETWEEN consecutive sub-scopes. For
  //     `denoise` that is the sampler's per-step update, which is real work, runs
  //     once per evaluation, and scales with the latent.
  //
  // The probe that separated them, run on the fixture's two renders: `denoise`
  // is one leaf record with 8 sub-scopes, head 3.2-15.9 us and tail 19.6-102.5
  // us, against SEVEN interior gaps of 62-67 us at nine frames and 463-549 us at
  // 81. So 95% of the residue is interior and 5% is boundary, and the interior
  // half is what moves with the box: `denoise` coverage measures 99.55% down to
  // 88.85% across three boxes on an UNCHANGED tree, while head-plus-tail stays
  // between 0.10% and 0.30% of the leaf in every one of those runs.
  //
  // THAT is why the 0.95 floor sat inside its own measurement and this one does
  // not. A share floor on the SUM cannot separate "the sampler updated its state
  // between evaluations" from "a phase nobody named is wearing this label"; the
  // span slack can, because only the second one lands here.
  //
  // ONE CONSTANT PER LEAF RECORD, AND DELIBERATELY NOT A SHARE OF THE LEAF. This bound was
  // first written `max(1 ms, 2% of the leaf)`, on the reasoning that a share is
  // what keeps an assertion meaningful at production scale. A fresh review showed
  // that reasoning inverts here, and the measurement agrees with the review:
  //
  //   * A share inside a `max` never tightens, so it is pure slack and it is
  //     LARGEST exactly where the leaf is largest. Under load `denoise` reached
  //     3.82 s with 38 us of slack against a 76 ms bound, and `decode.audio`
  //     3.07 s with 39 us against 61 ms -- 500x to 2200x. At production scale a
  //     swallowed phase of several seconds would have passed.
  //   * The quantity does not grow with the render. It is two instrument
  //     boundaries, and every measurement taken of it -- eight renders and four
  //     leaves here, plus a fresh review's own runs under an eight-way spin load
  //     where the denoise leaf ballooned to 3.82 s -- is at or under 135 us
  //     whatever the leaf did. A constant therefore holds on all of them and is
  //     strictly tighter than the share at every leaf size.
  //
  // AND IT IS CONFIGURATION-AWARE, which is the second thing a fresh review broke
  // and the more interesting one. The flat 0.25 ms below was set from plain and
  // spin-loaded builds only. NO measurement behind it came from a sanitizer
  // build, and the premise "this quantity does not scale" turned out to be right
  // about RENDER SIZE and silently wrong about BUILD CONFIGURATION. It reds on
  // both sanitizer lanes.
  //
  // The reason is not a fudge and it is worth stating plainly: the slack is the
  // cost of opening and closing a phase scope, and a sanitizer instruments
  // exactly that path. ASan checks a shadow byte on every access in it; TSan
  // additionally keeps per-access happens-before state and does it under a lock.
  // A scope boundary therefore genuinely costs more in those builds -- it is a
  // different machine, not a noisier one. Measured on this one test:
  //
  //   | lane | that test's wall | worst span slack observed |
  //   |---|---|---|
  //   | plain | 259 s | 121 us |
  //   | `address,undefined` | 1165 s | 730 us |
  //   | `thread` | 2116 s | 1658 us |
  //
  // (CI figures, which are the worst because those runners are two-core and
  // contended. The same probe on a 20-core box read 94 us / 569 us / 895 us in
  // the same order, so the ordering is the build and not the box.)
  //
  // A NORMALISED BOUND WOULD BE BETTER AND THERE IS NO NORMALISER. The old 0.95
  // coverage floor tolerated instrumentation ACCIDENTALLY, by being a ratio of two
  // quantities that inflate together: when TSan makes everything 8x slower both
  // `covered` and `leaf_seconds` grow and the ratio survives. The span slack is an
  // ABSOLUTE quantity, so the overhead lands on it undiluted -- which is the real
  // reason "does not grow with the render" did not generalise. It does not grow
  // with the RENDER; it grows with the INSTRUMENT.
  //
  // So the right question is whether some in-run reference scales the same way.
  // Five candidates were tested over 24 observations -- four leaves x two renders
  // x three build configurations -- and the spread of each, worst over best:
  //
  //   | candidate | spread | undefined |
  //   |---|---|---|
  //   | `slack / submin` | 78630x | 0 |
  //   | `slack / leaf_seconds` | 4598x | 0 |
  //   | `slack / mean sub-scope` | 2377x | 0 |
  //   | `slack / mean interior gap` | 418x | 3 of 24 |
  //   | **`slack` itself** | **44.6x** | 0 |
  //
  // EVERY RATIO IS WORSE THAN THE RAW QUANTITY, and the two most obvious ones are
  // worse by two orders of magnitude. The gap normaliser is also undefined for a
  // leaf with one sub-scope, which `artifacts.frames` is. Conditioned on the build
  // instead, the raw quantity is tight: 20.1-93.6 us plain, 128-569 us under ASan,
  // 52.1-895 us under TSan -- about 4.7x, 4.4x and 17.2x. So configuration is the
  // variable that actually explains this quantity, and a per-configuration
  // constant is the most stable bound available rather than a fallback.
  //
  // WHY NOT A SINGLE FLAT 4 ms COVERING ALL THREE: it would be ~30x slack on the
  // plain lane, where the honest value is ~121 us and the assertion currently
  // works, and that lane is the only one giving this file a clean signal today.
  // Two numbers keep each lane held to its own instrument.
  //
  // 0.25 ms plain is ~2.1x the worst plain measurement; 3 ms under a sanitizer is
  // ~1.8x the worst sanitizer measurement. Both are multiples of a measurement
  // rather than numbers fitted to make a run pass. The sanitizer margin is the
  // thinner of the two and that is disclosed rather than papered over: under TSan
  // the instrument's own noise is within about 4x of the smallest leaf it
  // measures, which is the honest state of this fixture at this scale.
  //
  // ONE BOUND PER LEAF RECORD, COMPARED AGAINST THAT RECORD ALONE. This line
  // first shipped summing the slack of every record of the leaf and comparing the
  // SUM against `kSpanSlackPerRecord * leaves.size()`. Both halves of that were
  // defects, and the comment that stood here asserted the opposite of one of
  // them (#1559):
  //
  //   * THE MULTIPLIER WAS READ FROM THE ARTIFACT UNDER TEST. `leaves.size()` is
  //     whatever `RecordsNamed(table, c.leaf, c.render)` returned, guarded only
  //     by `REQUIRE(!leaves.empty())` and tied to no independent number. A defect
  //     that emitted one EXTRA leaf record therefore enlarged its own budget by a
  //     whole constant. Assertion (0) counts the SUB-SCOPE records against
  //     `Ltx2ConditioningTrace`; nothing counts the leaf's own.
  //   * "STRICTER ON MULTI-RECORD LEAVES RATHER THAN LOOSER" WAS FALSE. A sum
  //     against N constants lets ONE record spend the whole budget, so a swallow
  //     concentrated in a single record of a three-record leaf got 3x the bound
  //     it would have got alone. That looseness was load-bearing rather than
  //     academic: under ASan `artifacts.frames` r2 measured 3.354 ms of slack
  //     against a 6 ms bound and passed only because that leaf has two records.
  //
  // Checking each record against one constant removes both. It is what the
  // failure message already claimed -- that ONE record's head and tail are
  // bounded -- and it is strictly tighter than the sum on every leaf that has
  // more than one record.
  //
  // AND THERE IS ONE CONSTANT NOW, NOT ONE PER BUILD CONFIGURATION. The third
  // shape carried 0.25 ms plain and 3 ms under either sanitizer, on the measured
  // ground that a sanitizer instruments the scope boundary this quantity IS. That
  // ground is still true and it is no longer the biggest term. Within the PLAIN
  // lane alone, on one host, one leaf record's own head-plus-tail spans 25.0 us
  // to 10.032 ms -- 402x -- so the scheduler moves this quantity further than the
  // sanitizer does, a second constant would have to clear the same tail anyway,
  // and 30 ms against 33 ms is a distinction without a difference. The sanitizer
  // lanes therefore get LOOSER here than the 3 ms they carried, which can only
  // turn a red green; the worst sanitizer slack anybody has recorded is 3.354 ms.
  //
  // THE NESTED-`#if` FORM GOES WITH THE SECOND CONSTANT, and what it recorded is
  // worth keeping, so it is kept in the row's spec and here: GCC does not define
  // `__has_feature`, and `defined(__has_feature) && __has_feature(...)` still has
  // to PARSE the call, which is a hard preprocessor error there. The one-line
  // `||` form compiled on the TSan leg only because `||` short-circuits before
  // reaching it, which is the kind of green that means nothing. Anyone
  // reintroducing a per-configuration constant needs the nested form again.
  const double kSpanSlackPerRecord = 0.03;  // 30 ms, every build configuration
  // THE COVERAGE FLOOR (2) IS WHAT HOLDS A SINGLE-RECORD LEAF THE CAP BELOW
  // SWALLOWS -- and ONLY a single-record leaf, which is the scope this line
  // used to leave out. Where the cap binds, this bound degrades to "head plus
  // tail is at most half the record". For a leaf of ONE record `covered >= 0.5 *
  // leaf_seconds` says exactly that, so (2) is a genuine precondition of this
  // assertion and the REQUIRE below keeps it available.
  //
  // ON A MULTI-RECORD LEAF THE HAND-OFF IS MUCH WEAKER AND THAT IS NOT HIDDEN
  // HERE. (2) is a statement about the leaf SUM, so a short record of a long
  // leaf draws on a budget it could never fill. Measured: `decode.video`'s
  // reopen records are 0.019-5.66 ms inside a 0.17-1.39 s leaf whose 0.90 floor
  // permits 17-139 ms, so a swallow that grows one of those records to just
  // under the resolution floor -- up to 94x its honest size -- is seen by
  // neither assertion. `### Owed out of W0` carries that escape as a ceiling
  // rather than as the name of an assertion, and the fix for it is an anchor
  // inside the reopen, not a wider bound.
  REQUIRE_MESSAGE(c.min_coverage >= 0.5,
                  "'" << c.leaf << "' carries a " << (100.0 * c.min_coverage)
                      << "% coverage floor, under the 50% that (1c) hands its "
                         "below-resolution records of a SINGLE-record leaf to");
  // AND EACH RECORD'S BOUND IS CAPPED AT HALF OF THAT RECORD, WITH `min` --
  // never `max`.
  //
  // The polarity is the whole point, and it is the inversion of the
  // `max(constant, share)` an earlier review removed:
  //
  //   * under `max`, the share was a FLOOR, so it only ever made the bound
  //     LOOSER, and it was loosest where the leaf was biggest -- 2200x slack on a
  //     multi-second leaf, where a swallowed phase would have passed;
  //   * under `min`, the share is a CEILING, so it only ever makes the bound
  //     TIGHTER, and it makes `span_bound < record_seconds` true BY
  //     CONSTRUCTION, which is what the REQUIRE below records rather than risks.
  //
  // WHERE THE CAP BINDS, THE RECORD IS BELOW THIS INSTRUMENT'S RESOLUTION, AND IT
  // IS REPORTED RATHER THAN CHECKED. That is the one line in this block a reader
  // should be suspicious of, so here is the whole argument.
  //
  // A capped bound is no longer the absolute quantity this assertion is about. It
  // is "head plus tail is at most half of this record", which on a SINGLE-record
  // leaf is exactly what `covered >= 0.5 * leaf_seconds` already says, and the
  // REQUIRE above holds every leaf here to a floor of at least 0.5. So on that
  // shape the capped arm adds nothing (2) does not already carry. On a
  // multi-record leaf it would add something, and what is given up there is
  // priced in the note beside the REQUIRE above rather than assumed away.
  //
  // AND IT IS WORSE THAN NOTHING, MEASURED. On the records where the cap binds,
  // the honest head-plus-tail is 4.6% to 72.3% of the record's own duration
  // (53 runs at head 6b48edb2c): `decode.video`'s reopen-after-the-last-chunk
  // record is 19-89 us of which 8-32 us is its own two boundaries. A 50% share is
  // inside that distribution, so the capped arm would decide by scheduler and not
  // by tree -- the exact defect the 0.95 coverage floor had and that this whole
  // assertion was added to replace.
  //
  // THE SKIP CANNOT HIDE A SWALLOW, and the direction is why. A swallowed phase
  // makes a leaf record LONGER, never shorter, so it moves a record TOWARD the
  // gated set and never out of it. What escapes is bounded and stated: a swallow
  // that leaves the record under 60 ms in total. On this fixture that is
  // `artifacts.frames` (0.90-61.0 ms over both renders) and `decode.video`'s two
  // short reopen records (0.019-5.66 ms), and both are named in
  // `### Owed out of W0` rather than left to be found.
  size_t span_checked = 0;
  size_t span_unresolvable = 0;
  double worst_span_slack = 0.0;
  for (size_t li = 0; li < leaves.size(); ++li) {
    INFO("leaf record " << (li + 1) << " of " << leaves.size());
    const double lo = leaves[li]["start_seconds"].get<double>();
    const double hi = leaves[li]["end_seconds"].get<double>();
    const double record_seconds = hi - lo;
    double first_in = hi;
    double last_in = lo;
    bool any = false;
    for (const std::pair<double, double>& iv : subs) {
      if (iv.first < lo - 1e-9 || iv.second > hi + 1e-9) continue;
      if (!any || iv.first < first_in) first_in = iv.first;
      if (!any || iv.second > last_in) last_in = iv.second;
      any = true;
    }
    // A leaf record containing NO sub-scope is entirely un-anchored, and every
    // one of its seconds is slack. That is the honest reading and it is also the
    // strictest one.
    const double span_slack = any ? (first_in - lo) + (hi - last_in) : record_seconds;
    worst_span_slack = std::max(worst_span_slack, span_slack);
    const double span_bound = std::min(kSpanSlackPerRecord, 0.5 * record_seconds);
    // THE BOUND MUST BE ABLE TO FAIL, and this is a REQUIRE because a check that
    // cannot fail is worse than no check. `span_slack <= record_seconds` holds by
    // construction above, so any bound at or above the record makes the CHECK
    // below vacuous -- it passes at 100% slack. The 1 ms version this line first
    // carried did exactly that on `artifacts.frames`, which measured 0.826-0.983
    // ms at the time: ten of twelve checks reddened under a forced-strict probe
    // and that leaf's two passed AT 100% SLACK.
    //
    // It is now true BY CONSTRUCTION, because the bound is capped at half the
    // record. It is kept as a cheap invariant on the FORMULA: an edit that
    // removes the cap, or reinstates a floor, reds here instead of silently
    // restoring a vacuous check. A zero-length leaf record reds here too, which
    // is the only way this line can fire today and is a defect either way.
    REQUIRE_MESSAGE(span_bound < record_seconds,
                    "record " << (li + 1) << " of the '" << c.leaf << "' leaf is "
                        << record_seconds << "s, at or under the " << span_bound
                        << "s span-slack bound, so a check against it could not fail: "
                           "`span_slack <= record_seconds` already holds by construction");
    if (span_bound < kSpanSlackPerRecord) {
      ++span_unresolvable;
      MESSAGE("  " << c.leaf << " record " << (li + 1) << " is " << record_seconds
                   << "s, under the " << (2.0 * kSpanSlackPerRecord)
                   << "s this bound can resolve: span slack " << span_slack << "s ("
                   << (100.0 * span_slack / record_seconds)
                   << "%) is NOT checked here. On a single-record leaf (2) below "
                      "is exactly as strict; on a multi-record leaf it is a share "
                      "of the SUM and much weaker -- see `### Owed out of W0` for "
                      "what escapes on this leaf");
      continue;
    }
    ++span_checked;
    MESSAGE("  " << c.leaf << " record " << (li + 1) << " span slack = " << span_slack << "s of "
                 << record_seconds << "s (" << (100.0 * span_slack / record_seconds)
                 << "%), bound " << span_bound << "s");
    CHECK_MESSAGE(span_slack <= span_bound,
                  "record " << (li + 1) << " of '" << c.leaf << "' spends " << span_slack
                      << "s inside its own leaf record but OUTSIDE the span its sub-scopes "
                         "occupy, over a bound of " << span_bound
                      << "s. A leaf that opened before its work began, or stayed open after it "
                         "ended, is carrying a phase nobody named under this one's name. This "
                         "quantity is TWO INSTRUMENT BOUNDARIES and it does not grow with the "
                         "render -- but it does move with the box and with the build: measured "
                         "over one build configuration on one host it spans 402x, so read "
                         "a red here against the run's own load before reading it as a swallowed "
                         "phase");
  }
  MESSAGE("  " << c.leaf << " span slack: " << span_checked << " of " << leaves.size()
               << " leaf record(s) checked at " << kSpanSlackPerRecord << "s, "
               << span_unresolvable << " below this instrument's resolution; worst "
               << worst_span_slack << "s");

  // (2) COVERAGE. A leaf that encloses its own sub-scopes AND a phase nobody
  // named would satisfy containment while still hiding time.
  //
  // WHAT THIS NUMBER IS NOW, AND WHY IT IS NOT THE 0.95 IT WAS. It bounds the
  // INTERIOR gaps -- the residue (1c) above cannot see -- and the interior is
  // real work whose share of the leaf is a property of the BOX, not of the tree.
  // The population, all on an unchanged `denoise`: 99.28%, 99.38%, 99.55% and
  // 99.228% on the row's own box; 98.52%, 98.84%, 98.23%, 98.77% and 94.14% on a
  // second x86 box as its load moved; 94.60%, 96.85% and 94.60% on the box #1494
  // measured; and 92.39% and 88.85% on the GitHub runner that has to stay green.
  // The 81-frame arm runs 85.85% to 97.09% over the same set. A 0.95 floor is
  // therefore BELOW half of its own honest distribution, which is why `main`
  // carried this red, and no floor near the measured share can be set without
  // reopening it.
  //
  // So this floor is deliberately loose, on exactly the argument the
  // `decode.video.vae` and `decode.audio.vocoder` floors beside it are set by:
  // it is worth having where honest and defective separate by a wide margin, and
  // it is not the assertion that carries this leaf. What carries it is (0), the
  // record count taken from `Ltx2ConditioningTrace`, and (1c) above, which bounds
  // the head and the tail of each leaf record at 30 ms where the record is
  // long enough to resolve it. (1c) does NOT hold this leaf's short records --
  // see the resolution note there -- so for `artifacts.frames` on the nine-frame
  // render this floor is the only thing under the head and the tail.
  // NOTHING WAS DELETED to get here: this number moved and an assertion was added
  // beside it.
  double covered = 0.0;
  for (const std::pair<double, double>& iv : subs) covered += iv.second - iv.first;
  MESSAGE("  " << c.leaf << " = " << leaf_seconds << "s over " << leaves.size()
               << " leaf record(s), of which " << subs.size() << " sub-scope(s) cover " << covered
               << "s (" << (100.0 * covered / leaf_seconds) << "%)");
  CHECK_MESSAGE(covered >= c.min_coverage * leaf_seconds,
                "the named parts of '" << c.leaf << "' cover only " << covered << "s of its "
                    << leaf_seconds << "s, under the " << (100.0 * c.min_coverage)
                    << "% this phase's anchor is expected to reach. The rest is a phase nobody "
                       "named, wearing this one's label");

  // (2b) AND EACH PART'S OWN SHARE, because (2) is checked on the SUM.
  //
  // WHAT IT CLOSES, and it is the fifth transfer shape this case has been shown.
  // Assertion (2) permits ONE part to cover the whole leaf while its siblings
  // measure nothing, and the SIBLING BOUNDARY is enough to do it: leave
  // `decode.audio.mel` open across the vocoder call and open `decode.audio.
  // vocoder` empty after it (mutation R1b, two lines, identical semantics and
  // identical call order) and the vocoder — the model #1010 asks for BY NAME —
  // reports 16 microseconds of a half-second leaf, with (0), (1b), (1), (2),
  // (3), the `nested` invariant and the non-overlap check all green. It is the
  // same 100% attribution transfer N1 achieves by swapping the two literals,
  // reached without touching either literal.
  //
  // ONLY THE VOCODER CARRIES A FLOOR, and the mel's 0.0 is a MEASURED debt
  // rather than an oversight. Honest, over the two renders of this case,
  // `decode.audio.vocoder` holds 99.46% and 94.10% of its leaf (97.85%, 91.57%,
  // 90.6% and 89.7% on earlier runs of this box); on the 21B artifact this row
  // shipped it holds 97.0% and 99.99%. Under R1b it holds 0.00091% and 0.0035%.
  // Between four and five orders of separation, which is the property the 0.50
  // `decode.video.vae` floor below exploits, and why this number is 0.50 rather
  // than tightened toward a measured share this box destroys. It also catches
  // the NAME SWAP: under N1 the vocoder name lands on the mel's work and reads
  // 2.76% and 7.68%, so that mutation now reds here as well as on (1b).
  //
  // THE MEL CANNOT HAVE ONE, AND THE REASON IS THAT THE TWO DISTRIBUTIONS
  // OVERLAP. Honest, `decode.audio.mel` holds 0.53% and 5.90% here (2.13%,
  // 8.43%, 9.4% and 10.3% on earlier runs) and 0.0004% to 2.9% on the artifact,
  // because the audio VAE decode's cost RELATIVE to the vocoder is a property of
  // the geometry. Under the MIRROR of R1b — mel emptied, the vocoder scope
  // covering both calls — it holds 0.00063% and 0.0051%. The honest 21B render's
  // 0.0004% is SMALLER than the mirror's 0.0051%, so no threshold separates
  // them: any floor that reddens the mirror also reddens an honest render this
  // row has already produced. The mirror is therefore measured, GREEN, and
  // recorded as open in `### Owed out of W0` rather than closed by a number that
  // cannot be justified.
  //
  // The two directions are not equally expensive, and that is why one is closed
  // and one is disclosed. R1b moves the VOCODER's 90-99.99% onto the mel, which
  // is the 100% attribution transfer that would have made W1 rank the wrong
  // model. The mirror moves the MEL's 0.0004-10% onto the vocoder, so it
  // overstates the name #1010 already asks for by at most the smaller share.
  //
  // AND IT IS A FLOOR, NOT A BAND. A partial transfer — an anchor covering half
  // its call — passes 0.50 and is not detected here. Closing that needs a scope
  // INSIDE the callee, which is the same debt the third row of the anchor table
  // in `### Owed out of W0` carries for all six anchors.
  REQUIRE(c.part_min_coverage.size() == c.parts.size());
  REQUIRE(part_seconds.size() == c.parts.size());
  for (size_t i = 0; i < c.parts.size(); ++i) {
    INFO("sub-scope = " << c.parts[i]);
    MESSAGE("    " << c.parts[i] << " = " << part_seconds[i] << "s, "
                   << (100.0 * part_seconds[i] / leaf_seconds) << "% of " << c.leaf);
    if (c.part_min_coverage[i] <= 0.0) continue;
    CHECK_MESSAGE(part_seconds[i] >= c.part_min_coverage[i] * leaf_seconds,
                  "'" << c.parts[i] << "' holds " << part_seconds[i] << "s of the "
                      << leaf_seconds << "s its '" << c.leaf << "' leaf spent, under the "
                      << (100.0 * c.part_min_coverage[i])
                      << "% this name is expected to reach. A SIBLING scope that opened around "
                         "this one's call carries its seconds under the sibling's name, and "
                         "every other assertion in this case survives that");
  }

  // (3) EXCLUSIVITY. Nothing else may run in the window the anchor spans. This
  // is the assertion that sees a TRANSFER directly rather than through a sum:
  // M7 opens `phase.finish` in the middle of the denoise, and the denoise's own
  // steps keep running around it.
  const double window_start = subs.front().first;
  const double window_end = subs.back().second;
  for (const NamedInterval& iv : LeafIntervals(table, c.render)) {
    if (iv.name == c.leaf) continue;
    if (std::find(c.partners.begin(), c.partners.end(), iv.name) != c.partners.end()) continue;
    // One bool, because doctest refuses to decompose a `||` inside an assertion.
    const bool outside = iv.end <= window_start + 1e-9 || iv.start >= window_end - 1e-9;
    CHECK_MESSAGE(outside,
                  "the leaf '" << iv.name << "' runs [" << iv.start << ", " << iv.end
                      << "], INSIDE the window that '" << c.leaf << "' work occupies ["
                      << window_start << ", " << window_end
                      << "]. Either the work is not where the name says, or a second name is "
                         "charging itself for this phase's seconds");
  }
}

// NOTHING BUT AN ANCHOR IS NESTED, and this is the assertion that closes the
// GROWING direction the four above cannot see.
//
// A record is `nested` when it opened while another leaf was already open
// (`render_phase_log.cpp`, `PhaseLog::Open`), and a nested record is dropped from
// `sum_leaf_seconds`. So a carrying leaf that grows over one of its NEIGHBOURS
// does not overlap anything and does not break any sum — the neighbour simply
// turns nested and leaves the timeline. Mutation M11 is exactly that: move the
// two `Close`s in the driver's chunk sink to after the PPM write and the whole
// writer is charged to `decode.video`, which is the leaf W5's lever is measured
// from. Containment held, coverage held at 99.68%, non-overlap held because the
// leaves stayed disjoint, and exclusivity held because `artifacts.frames` is a
// DECLARED PARTNER of `decode.video` and is skipped by name.
//
// The list is the anchors and nothing else, so the assertion needs no clock and
// no threshold: any other name appearing nested means a leaf was left open
// across work it does not name. The driver's own comment above the sink
// (`ltx2_video.cpp`, "Folding the writes into `decode.video` would charge W5's
// lever with the cost of a `write(2)`; nesting them would take the writer out of
// the sum entirely") states both halves of this as a requirement; before this
// line no gate held it.
void CheckOnlyAnchorsAreNested(const nlohmann::json& table,
                               const std::vector<std::string>& anchors) {
  for (const nlohmann::json& e : table["phases"]) {
    if (!e.value("nested", false)) continue;
    const std::string name = e["name"].get<std::string>();
    INFO("nested record = " << name);
    CHECK_MESSAGE(std::find(anchors.begin(), anchors.end(), name) != anchors.end(),
                  "'" << name << "' is emitted NESTED, and it is not one of this render's "
                      "anchors. A nested record is excluded from `sum_leaf_seconds`, so a leaf "
                      "that grew across this one moved its seconds onto itself and took it out "
                      "of the table at the same time — with no overlap, no gap and no change to "
                      "what the phases add up to");
  }
}

// ...and the converse for the pair the driver INTERLEAVES. `artifacts.frames`
// alternates with `decode.video` rather than nesting inside it, which is what
// keeps the `write(2)` off W5's lever. Asserted over the INTERVALS as well as
// over the `nested` flag, so a change in what the emitter means by `nested`
// cannot quietly retire it.
void CheckWriterIsBesideTheDecode(const nlohmann::json& table, int64_t render) {
  const std::vector<nlohmann::json> writes = RecordsNamed(table, "artifacts.frames", render);
  REQUIRE_MESSAGE(!writes.empty(), "the render wrote frames and named no 'artifacts.frames' leaf");
  const std::vector<nlohmann::json> decodes = RecordsNamed(table, "decode.video", render);
  REQUIRE(!decodes.empty());
  for (const nlohmann::json& w : writes) {
    const double ws = w["start_seconds"].get<double>();
    const double we = w["end_seconds"].get<double>();
    INFO("artifacts.frames = [" << ws << ", " << we << "]");
    CHECK_FALSE_MESSAGE(w.value("nested", false),
                        "the PPM writer is NESTED, so its seconds are charged to whatever leaf "
                        "encloses it and dropped from the sum");
    for (const nlohmann::json& d : decodes) {
      const double ds = d["start_seconds"].get<double>();
      const double de = d["end_seconds"].get<double>();
      const bool outside = we <= ds + 1e-9 || ws >= de - 1e-9;
      CHECK_MESSAGE(outside,
                    "a 'decode.video' leaf runs [" << ds << ", " << de << "] and ENCLOSES the "
                        << "writer at [" << ws << ", " << we << "]. W5 ranks this campaign's "
                        << "levers off the `decode.video` number and it would be reading the "
                        << "cost of a write(2)");
    }
  }
}

// THE PER-RENDER HALF OF THE CONTAINMENT CASE, for one render of the table.
//
// A free function because the case now runs it TWICE, and that is the repair for
// a fourth fresh review's third finding: every decode-side count it made was 1
// or 1+1, because the fixture renders nine frames as ONE temporal chunk. The
// count assertion is the one that is not a ratio, and a count demonstrated only
// at N=1 cannot tell "once per unit of work" from "once per render". The second
// call renders 81 frames, which is the smallest request this fixture chunks (the
// MULTI-CHUNK case above derives that number), and `min_chunks` makes a geometry
// that silently stopped chunking a RED rather than a vacuous pass.
//
// `render` is the index the table's `render` field carries. It is threaded
// through every lookup here because two renders now share one table; see
// `RecordsNamed`.
void CheckRenderPhases(const nlohmann::json& table,
                       const vllm::multimodal::Ltx2ConditioningTrace& trace,
                       const vllm::multimodal::VideoResult& result, int64_t latent_channels,
                       int64_t render, int64_t min_chunks, double denoise_min_coverage) {
  INFO("render = " << render);

  // ── WHAT THE RENDER COUNTED, which is not what the instrument counted ──────
  //
  // Two numbers, both maintained by the render itself. `dit_evaluations` is
  // incremented inside the shared `Evaluate` lambda every sampler arm goes
  // through; `video_decode_chunks` is incremented in the driver's streaming sink
  // beside `rendered_frames`. They are the anchors' denominators below.
  REQUIRE(trace.completed);
  REQUIRE(trace.dit_evaluations > 0);
  REQUIRE(trace.video_decode_chunks > 0);

  // AND THE CHUNK COUNT IS RE-DERIVED rather than taken on faith, in the shape
  // the multi-chunk case above already uses: the tiling algebra the decoder runs
  // is public, so the group count for this request is computable here. A counter
  // that was incremented in the wrong place would agree with the phase table and
  // disagree with this.
  //
  // EVERY INPUT COMES FROM THE FIXTURE OR FROM THE RESULT, which is the repair
  // for the second half of the same finding. This block used to hardcode
  // `Ltx2ScaleFactors{8, 32, 32}`, `Ltx2AutoTileSizeConfig(32, 32, ...)` and
  // `Ltx2CreateTiles(latent_t, 1, 1, ...)` — three constants copied out of the
  // fixture header's comment, which is a re-derivation of the algebra and not of
  // the geometry. The factors now come from the fixture's OWN decoder block list
  // through the same public helper the tiled decode calls, and the tile layout
  // from the size the render reported. A fixture whose blocks changed would move
  // both sides together before; now it moves only the engine's.
  const vllm::Ltx2ConvVideoDecoderConfig vae_cfg =
      ltx2_fixture::ReducedVideoDecoderConfig(latent_channels);
  const vllm::Ltx2ScaleFactors factors =
      vllm::Ltx2VideoScaleFactorsFromBlocks(vae_cfg.decoder_blocks, vae_cfg.patch_size);
  const vllm::Ltx2TileSizeConfig layout =
      vllm::Ltx2AutoTileSizeConfig(result.height, result.width, factors);
  const int64_t latent_t = (result.frame_count - 1) / factors.time + 1;
  const int64_t latent_h = result.height / factors.height;
  const int64_t latent_w = result.width / factors.width;
  REQUIRE(latent_h >= 1);
  REQUIRE(latent_w >= 1);
  const std::vector<vllm::Ltx2Tile> tiles =
      vllm::Ltx2CreateTiles(latent_t, latent_h, latent_w, layout, factors);
  const int64_t groups =
      static_cast<int64_t>(vllm::Ltx2GroupTilesByTemporalSlice(tiles).size());
  REQUIRE_MESSAGE(groups >= min_chunks,
                  "this request decodes in " << groups << " temporal group(s) and this call "
                      << "exists to exercise at least " << min_chunks
                      << ". The counts below would be demonstrated at a chunk count the "
                         "geometry no longer produces");
  CHECK_MESSAGE(trace.video_decode_chunks == groups,
                "the render counted " << trace.video_decode_chunks << " decoded chunk(s) and "
                    << "this request's tiling produces " << groups << " temporal group(s)");

  // (1)-(3) THE THREE PHASES THAT CARRY THIS RENDER, each with its anchor.
  //
  // The coverage thresholds are NOT round numbers and are not the same, because
  // what each anchor can see differs:
  //
  //  * `decode.audio` is exactly two calls and the leaf holds nothing else, so
  //    mel+vocoder measure 99.997%, 99.9995% and 99.995% of it across three
  //    runs, over a leaf of 0.1 s to 1.2 s where a scheduling hiccup at a scope
  //    boundary is noise. 0.99 is the threshold, and the 0.90 this case shipped
  //    with was a HOLE the second review named: at 0.90 `decode.audio` could
  //    open 11% early and swallow 0.13 s, which is 87% of this render's entire
  //    `decode.video`, while passing everything in this file.
  //  * `denoise.step` covers the denoiser EVALUATIONS, measured at 99.37% and
  //    99.67% over 8 of them. The uncovered part is real work — the post-process
  //    and the Euler or res_2s step, which no anchor wraps — PLUS two instrument
  //    boundaries per evaluation. 0.95.
  //
  //    THE FALSE RED THE PREVIOUS COMMENT PREDICTED HAPPENED, AND ITS CAUSE IS
  //    NOT THE ONE IT NAMED (#1494, and #1439 for the sibling assertion). It
  //    reasoned that the uncovered part is 16 boundary samples whose cost is
  //    fixed, so the ratio worsens as the hardware gets faster, put the crossing
  //    point at a denoise leaf of about 3.1 ms, and left the number at 0.95.
  //    `main` then carried this case RED for two days at leaves of 13.7 ms and
  //    27.5 ms, which are four to nine times above that crossing point.
  //
  //    MEASURED, THE RESIDUE IS NOT BOUNDARIES. Probing the intervals over the
  //    fixture's two renders: head 3.2-15.9 us and tail 19.6-102.5 us, against
  //    SEVEN interior gaps of 62-67 us at nine frames and 463-549 us at 81. The
  //    boundaries are about 5% of the residue; the other 95% is the sampler's
  //    per-step update, which is real work, runs once per evaluation, and scales
  //    with the latent. Its share of the leaf is a property of the BOX and not
  //    of the tree, because a short serial stretch between two long parallel
  //    ones is what loses to a contended scheduler. On an UNCHANGED `denoise`
  //    the nine-frame arm measures 99.55%, 99.38%, 99.28% and 99.228% (the row's
  //    own box), 98.84%, 98.77%, 98.52%, 98.23% and 94.14% (a second x86 box as
  //    its load moved), 96.85%, 94.60% and 94.60% (the box #1494 measured), and
  //    92.39% and 88.85% (the GitHub runner this lane has to stay green on). The
  //    81-frame arm spans 97.09% down to 85.85% over the same set.
  //
  //    SO THE SHARE FLOOR CANNOT BE THE ASSERTION THAT CARRIES THIS LEAF. A
  //    0.95 floor sits below half of its own honest distribution, and any floor
  //    set near the measured share reopens the red on the next contended runner.
  //    It is 0.75 on both arms: 10.85 points below the worst honest observation,
  //    and worth having only where honest and defective separate widely, which
  //    is the same argument the 0.50 floors on `decode.video.vae` and
  //    `decode.audio.vocoder` are set by. The geometry dependence the previous
  //    comment parameterised is real and is SMALLER than the box dependence, so
  //    one number below both arms replaces two numbers sitting inside each.
  //
  //    WHAT REPLACES IT IS ASSERTION (1c), the span slack, which bounds the head
  //    and the tail -- the only place a swallowed phase can land -- at
  //    30 ms PER LEAF RECORD, on the records long enough to resolve it.
  //    NO CLAIM IS MADE THAT THIS IS TIGHTER THAN 0.95 IN NUMBERS, because it is
  //    not: 5% of a 13.7 ms leaf is 685 us. The point is a different one. The old
  //    bound covered the head, the tail AND the interior together, while the
  //    interior alone measured 1046 us on the runner that reported it -- so the
  //    budget was already spent before any swallow, no honest tree could satisfy
  //    it, and it therefore bounded nothing at all. (1c) is spent by the
  //    boundaries alone, which do not grow with the render.
  //
  //    THEY DO MOVE WITH THE BOX, and the first two shapes of this bound were
  //    calibrated as though they did not. Measured unmutated at head 6b48edb2c,
  //    one build configuration, one host: 25.0 us to 10.032 ms on one leaf record, 402x. That is why the
  //    constant is 30 ms and not the 0.25 ms this note used to quote, and
  //    it is why a red here is read against the run's own load first (#1559).
  //
  //    NAMING THE UN-NAMED TIME WOULD SETTLE IT PROPERLY, which is what #1439
  //    asks for first. A `denoise.update` scope over the sampler's per-step
  //    update would put the interior residue under a name and make a tight share
  //    floor honest again. That is a production scope in `ltx2_video.cpp` inside
  //    both sampler arms, it makes `denoise` a multi-part leaf and so pulls in
  //    (1b) and `part_min_coverage`, and it owes the phase names published in
  //    `docs/models/ltx-2-5.md`. It stays owed rather than being folded into the
  //    repair of a standing red, which is the same judgement #1439 and #1494
  //    each recorded for themselves.
  //  * `decode.video.chunk` runs from the leaf's own open to the moment the
  //    decoder hands a chunk back, so the only uncovered part is TWO instrument
  //    boundaries — measured 99.44% — and that is why its threshold is the
  //    loosest of the three rather than the tightest. This leaf is 1.6 ms on
  //    this fixture. One preemption between `Close(chunk)` and `Close(decode)`
  //    is a millisecond, i.e. 60% of it, so a threshold set near the measured
  //    value would be a coin flip and not a gate. 0.90 here permits 0.16 ms to
  //    go unnamed, which is three orders of magnitude below the 0.13 s the same
  //    0.90 permitted on the audio decode: the fraction is the same and the
  //    stake is not, which is why these three numbers are not one number.
  //
  // AND EACH CARRIES ITS RECORD COUNT, which is assertion (0) and the one that
  // is not a ratio. `denoise.step` runs once per denoiser evaluation.
  // `decode.video.chunk` opens with the leaf and reopens after every chunk the
  // sink is handed, so it emits one more record than there are chunks — the last
  // one is the empty window between the final chunk and the end of the decode.
  // The audio decode is exactly one mel pass and one vocoder pass.
  const std::vector<Carrying> carrying = {
      {"denoise",
       {"denoise.step"},
       denoise_min_coverage,
       {},
       {trace.dit_evaluations},
       "Ltx2ConditioningTrace::dit_evaluations",
       // ONE part, so (2) already IS the per-part assertion and a second copy of
       // it would only be noise. `denoise.step`'s own placement debt is the
       // third row of the anchor table in `### Owed out of W0`.
       {0.0},
       render},
      {"decode.video",
       {"decode.video.chunk"},
       0.90,
       {"artifacts.frames"},
       {trace.video_decode_chunks + 1},
       "Ltx2ConditioningTrace::video_decode_chunks + 1, the reopen after the last chunk",
       // One part; and this leaf's real per-part floor is the `decode.video.vae`
       // block below, which is a ratio between two ANCHORS rather than between an
       // anchor and its leaf.
       {0.0},
       render},
      {"decode.audio",
       {"decode.audio.mel", "decode.audio.vocoder"},
       0.99,
       {},
       {1, 1},
       "the two calls #1010 names, once each per render",
       // THE ONLY MULTI-PART LEAF, and therefore the only place (2b) is not
       // vacuous. The vocoder carries 0.50 and the mel carries none; the note on
       // `part_min_coverage` and on (2b) argues both numbers, and
       // `### Owed out of W0` records what the mel's 0.0 leaves open.
       {0.0, 0.50},
       render},
      // AND THE WRITER, which shipped with no anchor at all and is the leaf the
      // `decode.video` number is protected by. `artifacts.frames.ppm` wraps the
      // `WriteFileBytes` loop, so it moves with the WRITES: a mutation that
      // empties `artifacts.frames` and leaves the `write(2)`s inside
      // `decode.video` — the reviewer's M11, which reported four microseconds
      // for nine PPM files — no longer has a leaf that contains this record.
      //
      // 0.50 is the threshold and it is loose ON PURPOSE. This leaf is about
      // 0.2 ms on the fixture and holds nine scope boundaries of its own, so the
      // uncovered part is instrument cost against a sub-millisecond leaf, which
      // is the regime the `decode.video` note above already explains.
      //
      // WHAT BINDS HERE IS THE COUNT AND THE FACT THAT THE WRITER IS NOT
      // NESTED, and this note used to say "the containment and the count", which
      // a fourth fresh review showed is false. A count of one plus containment
      // permits this leaf to be TWICE its anchor and still pass: the leaf may
      // grow over any adjacent time nobody named, up to its coverage slack. It
      // is harmless on this fixture only because nothing adjacent to the writer
      // is stealable — `decode.video` is a declared partner that the `nested`
      // assertion and `CheckWriterIsBesideTheDecode` both hold — and that is a
      // property of the fixture rather than of this threshold.
      {"artifacts.frames",
       {"artifacts.frames.ppm"},
       0.50,
       {"decode.video"},
       {trace.video_decode_chunks},
       "Ltx2ConditioningTrace::video_decode_chunks, one write callback per chunk",
       // One part, so (2) already is it.
       {0.0},
       render},
  };
  for (const Carrying& c : carrying) CheckCarryingPhase(table, c);

  // (3b) THE VAE'S OWN SUB-SCOPE, which is where the video decode spends its
  // seconds and the reason `decode.video.chunk` is not the whole answer.
  //
  // `decode.video.chunk` opens two statements after `Open("decode.video")` and
  // closes one statement before `Close("decode.video")`, so containment and
  // coverage over it hold BY CONSTRUCTION and cannot constrain the leaf in the
  // growing direction — which is what M11 exploited. `decode.video.vae` is
  // opened inside `Ltx2ConvVideoDecodeTiled` around `AccumulateTemporalGroup`,
  // i.e. around the tile decode itself, so both of its ends are production
  // events. One record per temporal group, which is one per chunk, and each one
  // inside the chunk window that produced it.
  //
  // AND IT HAS A COVERAGE FLOOR, which is the repair for a fourth fresh review's
  // second finding. Until this line the vae anchor was checked for cardinality,
  // for `nested` and for containment in a chunk window, and its DURATION was
  // compared against nothing: `Carrying{"decode.video", ...}` lists only
  // `decode.video.chunk` in `parts`, so the vae contributed ZERO to any
  // coverage. Moving the scope off `AccumulateTemporalGroup` and onto the
  // `buffer.Allocate` beside it — seven lines — left the table reporting
  // `decode.video.vae = 0.000 s` beside a five-millisecond `decode.video`, with
  // the tile decode inside no sub-scope at all, and both cases in this file
  // green. The claim that this leaf "finally has a sub-scope whose ends are both
  // production events" was true of the source and unheld by the gate.
  //
  // THE DENOMINATOR IS `decode.video.chunk`, not the leaf, AND THE REOPEN IS NOT
  // THE REASON. This note used to say the chunk windows are the tightest
  // denominator available "without a second threshold for the reopen", which
  // claims more than the numbers do. The reopen — the empty window between the
  // final chunk and the end of the decode — is itself a `decode.video.chunk`
  // record, so it sits inside BOTH candidate denominators and separates neither.
  // Measured on the head this lands at, render 1: the leaf is 0.191059 s against
  // 0.191044 s of chunk, a 0.0079% difference that is two scope boundaries;
  // render 2: 0.408779 s against 0.408758 s, 0.0051%. An earlier run of the same
  // binary at the same geometry read 1.06% and 0.39%, because the boundary cost
  // is fixed while the leaf is not — which is the box, not the structure. Either
  // way the two denominators are within about a percent of each other and the
  // LEAF is the marginally stricter of the two, so nothing is loosened by this
  // choice.
  //
  // The chunk is used because it is the same window the containment assertion
  // below encloses each vae record IN, which keeps this ratio between two
  // quantities at one granularity rather than comparing a per-group anchor
  // against a whole-decode leaf. Measured at 91.1%, 93.6%, 97.4%, 98.2%, 98.5% and 98.6% of
  // the chunk seconds across six renders here — the low end is the nine-frame
  // render, whose single chunk is about a millisecond. The floor is 0.50
  // because what is NOT covered is the per-group
  // allocation, the overlap blend and the emit — real work whose share of a
  // chunk grows as the tile shrinks — plus two instrument boundaries per group
  // against a chunk of about a millisecond on this fixture. It is a floor on
  // "the tile decode is where this leaf's seconds are", not on the fraction.
  {
    const std::vector<nlohmann::json> vae = RecordsNamed(table, "decode.video.vae", render);
    CHECK_MESSAGE(static_cast<int64_t>(vae.size()) == trace.video_decode_chunks,
                  "the tiled decode emitted " << vae.size() << " 'decode.video.vae' record(s) "
                      << "against " << trace.video_decode_chunks << " chunk(s) the render "
                      << "counted");
    const std::vector<nlohmann::json> chunks = RecordsNamed(table, "decode.video.chunk", render);
    double vae_seconds = 0.0;
    double chunk_seconds = 0.0;
    for (const nlohmann::json& ch : chunks) {
      chunk_seconds += ch["end_seconds"].get<double>() - ch["start_seconds"].get<double>();
    }
    for (const nlohmann::json& v : vae) {
      CHECK_MESSAGE(v.value("nested", false),
                    "'decode.video.vae' is not marked nested, so the tile decode is being SUMMED "
                    "as well as the leaf that contains it");
      const double vs = v["start_seconds"].get<double>();
      const double ve = v["end_seconds"].get<double>();
      vae_seconds += ve - vs;
      bool inside = false;
      for (const nlohmann::json& ch : chunks) {
        if (vs >= ch["start_seconds"].get<double>() - 1e-9 &&
            ve <= ch["end_seconds"].get<double>() + 1e-9) {
          inside = true;
        }
      }
      CHECK_MESSAGE(inside, "a 'decode.video.vae' record runs [" << vs << ", " << ve
                                << "] and no 'decode.video.chunk' window encloses it");
    }
    REQUIRE(chunk_seconds > 0.0);
    const double kVaeMinCoverage = 0.50;
    MESSAGE("  decode.video.vae = " << vae_seconds << "s over " << vae.size()
                                    << " record(s), of " << chunk_seconds
                                    << "s of decode.video.chunk ("
                                    << (100.0 * vae_seconds / chunk_seconds) << "%)");
    CHECK_MESSAGE(vae_seconds >= kVaeMinCoverage * chunk_seconds,
                  "'decode.video.vae' covers " << vae_seconds << "s of the " << chunk_seconds
                      << "s this render spent inside 'decode.video.chunk', under the "
                      << (100.0 * kVaeMinCoverage)
                      << "% the tile decode is expected to reach. The anchor is somewhere the "
                         "work is not, and the leaf W5 ranks its levers from is held by a "
                         "cardinality alone");
  }

  CheckWriterIsBesideTheDecode(table, render);

  // (5) AND THE CARRYING LEAVES DO NOT OVERLAP. `decode.video` and
  // `artifacts.frames` ALTERNATE, because the decoder streams chunks into the
  // writer's callback; `decode.audio` follows both; `denoise` precedes all of
  // them and is in this list because the review found it was in NO list — the
  // loop that shipped covered the decode and the writer and left the largest
  // phase of the render out of it. Any overlap would mean a leaf was left open
  // across work it does not name, and it would double count into
  // `sum_leaf_seconds`.
  std::vector<std::pair<double, double>> spans;
  for (const std::string name :
       {"denoise", "decode.video", "artifacts.frames", "decode.audio"}) {
    for (const nlohmann::json& r : RecordsNamed(table, name, render)) {
      spans.push_back({r["start_seconds"].get<double>(), r["end_seconds"].get<double>()});
    }
  }
  std::sort(spans.begin(), spans.end());
  for (size_t i = 1; i < spans.size(); ++i) {
    CHECK_MESSAGE(spans[i].first >= spans[i - 1].second - 1e-9,
                  "two of the carrying/writer leaves overlap: [" << spans[i - 1].first << ", "
                      << spans[i - 1].second << "] and [" << spans[i].first << ", "
                      << spans[i].second << "]");
  }
}

}  // namespace

TEST_CASE("ltx2 video: the three carrying phases contain their work and the load keeps its order") {
  Workspace ws;
  // `max_phase = 0` is a LOAD extra and is what keeps this on one recipe phase:
  // the default `distilled_two_stage` kind would need the latent spatial
  // upsampler for its second phase and refuse without one.
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  vllm::multimodal::Ltx2VideoEngine* ltx2 =
      dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  // The fixture writes its video VAE from `ReducedVideoDecoderConfig(dit.in_channels)`
  // (`ltx2_video_fixture.h`), and `CheckRenderPhases` re-derives the tiling from
  // that same config rather than from constants.
  const int64_t latent_channels = ltx2->dit_params().in_channels;

  // ── RENDER 1: the nine-frame fixture, which decodes in ONE temporal chunk ──
  const vllm::multimodal::VideoResult result =
      engine->Generate(FixtureGen(ws.root + "/attribution"));
  REQUIRE(result.frame_count == 9);
  REQUIRE_MESSAGE(!result.phase_log_path.empty(), "the render wrote no phase table");
  const nlohmann::json first = nlohmann::json::parse(ReadAll(result.phase_log_path));
  const int64_t render_one = LastRender(first);
  REQUIRE(render_one > 0);
  // TAKEN BEFORE THE SECOND GENERATION OVERWRITES IT: `im.trace` is reset on
  // every `Generate`, which is half of why the counts here need a render filter.
  const vllm::multimodal::Ltx2ConditioningTrace trace_one = ltx2->last_conditioning();
  CheckRenderPhases(first, trace_one, result, latent_channels, render_one, /*min_chunks=*/1,
                    /*denoise_min_coverage=*/0.75);

  // ── RENDER 2: 81 frames, which CHUNKS, and every count above is checked at
  // N > 1 ───────────────────────────────────────────────────────────────────
  //
  // WHY A SECOND RENDER. Every decode-side count in the first one is 1 or 1+1 —
  // `video_decode_chunks == 1`, `decode.video.chunk == 2`, `artifacts.frames.ppm
  // == 1`, `decode.video.vae == 1` — and the re-derivation is `1 == 1`. A count
  // demonstrated only at one cannot tell "once per unit of work" from "once per
  // render", which is exactly the distinction assertion (0) exists to make. 81
  // frames is the smallest request this fixture chunks; the MULTI-CHUNK case
  // above derives that number and asserts it, and `min_chunks` below makes a
  // geometry that stopped chunking a red rather than a quiet regression to N=1.
  //
  // AND IT IS WHAT MAKES THE `render` FILTER LOAD-BEARING. `PhaseLog` resets at
  // LOAD, so this second table holds the load AND both renders. Every count
  // below is per render and would otherwise be the sum of two.
  vllm::multimodal::VideoGenParams multi = FixtureGen(ws.root + "/attribution_multichunk");
  multi.num_frames = 81;
  const vllm::multimodal::VideoResult multi_result = engine->Generate(multi);
  REQUIRE(multi_result.frame_count == 81);
  REQUIRE_MESSAGE(!multi_result.phase_log_path.empty(), "the render wrote no phase table");
  const nlohmann::json table = nlohmann::json::parse(ReadAll(multi_result.phase_log_path));
  const int64_t render_two = LastRender(table);
  CHECK_MESSAGE(render_two == render_one + 1,
                "the second generation carries render index " << render_two << " and the first "
                    << render_one << "; the table's `render` field is not the per-generation "
                                     "slice the counts below are taken over");
  CheckRenderPhases(table, ltx2->last_conditioning(), multi_result, latent_channels, render_two,
                    /*min_chunks=*/2, /*denoise_min_coverage=*/0.75);

  // ...and the FIRST render's records are still in the second render's table and
  // still hold, which is what says the filter above selects rather than hides.
  CheckRenderPhases(table, trace_one, result, latent_channels, render_one, /*min_chunks=*/1,
                    /*denoise_min_coverage=*/0.75);

  // (3c) AND NOTHING BUT AN ANCHOR IS NESTED. The assertion that sees a leaf
  // swallow a NEIGHBOUR, which the four above cannot: see the note on
  // `CheckOnlyAnchorsAreNested`.
  CheckOnlyAnchorsAreNested(table, {"denoise.step", "decode.video.chunk", "decode.video.vae",
                                    "decode.audio.mel", "decode.audio.vocoder",
                                    "artifacts.frames.ppm"});

  // (4) THE FLOOR, read as "this name is not detached" and nothing more. See the
  // note above this case for why it is not tightened toward the measured share.
  const std::map<std::string, double> leaves = LeafSecondsByName(table);
  double leaf_total = 0.0;
  for (const std::pair<const std::string, double>& kv : leaves) leaf_total += kv.second;
  REQUIRE(leaf_total > 0.0);
  const double kFloor = 0.0005;
  for (const std::string required : {"denoise", "decode.audio", "decode.video"}) {
    INFO("phase = " << required);
    REQUIRE(leaves.count(required) == 1);
    const double share = leaves.at(required) / leaf_total;
    MESSAGE("  " << required << " = " << leaves.at(required) << "s, " << (100.0 * share)
                 << "% of " << leaf_total << "s of leaves");
    CHECK_MESSAGE(share >= kFloor,
                  "'" << required << "' holds " << (100.0 * share) << "% of this render's named "
                      << "leaf seconds, under the " << (100.0 * kFloor)
                      << "% floor. A phase that carries a render and measures nothing is a name "
                         "with the work somewhere else");
  }

  // (6) THE LOAD LEAVES SIT IN THE DRIVER'S OWN ORDER, and this is the weakest
  // assertion in the case as well as the only one the load has.
  //
  // WHY IT EXISTS. The three assertions above cover the three phases that carry
  // a RENDER. They say nothing about the six the LOAD is made of, and a third
  // fresh review showed what that costs: swap the `load.dit` and
  // `load.prompt_embeds` scope NAMES in the driver — two lines — and every gate
  // in this file stays green while 96% of the load's seconds move onto the wrong
  // name. On the shipped 21B that name holds about seven and a half minutes of
  // DiT staging, and it is the phase W2 and W3 both act on.
  //
  // WHY IT IS ONLY AN ORDER. An anchor for `load.dit` would have to open inside
  // `Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice`
  // (`ltx2_loader.cpp`), which this stage's authority does not reach; the driver
  // has exactly one statement between the scope's open and its close, so a
  // sub-scope placed HERE would be the adjacent-statement anchor whose
  // circularity M11 and M12 already demonstrated. The order is what remains
  // checkable from the table alone: the driver loads the DiT, then the video
  // VAE, then the audio VAE, then the upsampler, then the text encoder, then the
  // prompt embeds, and a swapped pair of names shows up here as a swapped pair
  // of start times. What is still NOT proven is that each of these names covers
  // the work it is named after; `### Owed out of W0` in the row's spec lists
  // every leaf in that state, and W2 and W3 must read it before they act on
  // `load.dit`.
  {
    const std::vector<std::string> load_order = {"load.dit",       "load.video_vae",
                                                 "load.audio_vae", "load.upsampler",
                                                 "load.text_encoder", "load.prompt_embeds"};
    double previous = -1.0;
    std::string previous_name;
    int64_t seen = 0;
    for (const std::string& name : load_order) {
      const std::vector<nlohmann::json> found = RecordsNamed(table, name, 0);
      // A load phase that did not run emits nothing; this fixture carries no
      // upsampler and no text encoder.
      if (found.empty()) continue;
      CHECK_MESSAGE(found.size() == 1u,
                    "'" << name << "' was emitted " << found.size()
                        << " times and the driver opens it once per load");
      const double start = found.front()["start_seconds"].get<double>();
      CHECK_MESSAGE(start >= previous,
                    "'" << name << "' starts at " << start << " and '" << previous_name
                        << "' starts at " << previous << ", so the table reports the load in an "
                           "order the driver does not run it in. Two scope names that were "
                           "swapped move their seconds onto each other and change nothing else "
                           "in this file");
      previous = start;
      previous_name = name;
      ++seen;
    }
    REQUIRE_MESSAGE(seen >= 4, "this fixture load names only " << seen
                                   << " load phases; the order it checks is vacuous");
  }
}

// ─── the CONSOLE copy is not a passenger of the FILE write ──────────────
//
// `VLLM_RENDER_PHASE_LOG_STDERR=1` exists for exactly one situation: a render
// whose table cannot reach a file. An unwritable `--output-dir`, a read-only
// mount, a full disk. If the console block is emitted only after the file write
// has already succeeded, the lane is silent in the one case it was added for,
// and `docs/ENVIRONMENT.md`'s "also prints" describes something the code does
// not do. This case pins the ORDER rather than the text: the block reaches the
// real fd 2, and `WriteJson` still reports the IO failure by name instead of
// letting the console copy launder an error into a success.
//
// Real fd 2 by `dup`/`dup2`, not a test-only sink, for the reason the live
// progress lane gives: a sink is a seam that can be armed in the test and absent
// in production.
TEST_CASE("ltx2 phase log: the stderr copy is emitted when the FILE write FAILS") {
  namespace phase = vllm::multimodal::phase;
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Begin();
  { phase::Scope denoise("denoise"); }
  { phase::Scope decode("decode.video"); }

  // A directory nobody created, so the `ofstream` is not `good()` at open. The
  // precondition is asserted rather than assumed — a path that happened to exist
  // would turn this case into a test of the success path.
  const char* kMissingDir = "/vllm-cpp-no-such-directory-1010";
  struct stat missing {};
  REQUIRE(::stat(kMissingDir, &missing) != 0);
  const std::string path = std::string(kMissingDir) + "/phase-log.json";

  const char* prev = std::getenv("VLLM_RENDER_PHASE_LOG_STDERR");
  const std::string saved_env = prev == nullptr ? std::string() : std::string(prev);
  ::setenv("VLLM_RENDER_PHASE_LOG_STDERR", "1", 1);

  char tmpl[] = "/tmp/vllm_phase_stderr_XXXXXX";
  const int sink = ::mkstemp(tmpl);
  REQUIRE(sink >= 0);
  std::fflush(stderr);
  const int saved_fd = ::dup(2);
  REQUIRE(saved_fd >= 0);
  REQUIRE(::dup2(sink, 2) >= 0);

  std::string why;
  const bool ok = log.WriteJson(path, "ltx2-video", "cpu", &why);

  std::fflush(stderr);
  ::dup2(saved_fd, 2);
  ::close(saved_fd);
  ::close(sink);

  if (prev == nullptr) {
    ::unsetenv("VLLM_RENDER_PHASE_LOG_STDERR");
  } else {
    ::setenv("VLLM_RENDER_PHASE_LOG_STDERR", saved_env.c_str(), 1);
  }

  std::ifstream in(tmpl, std::ios::binary);
  std::stringstream buf;
  buf << in.rdbuf();
  const std::string captured = buf.str();
  in.close();
  ::unlink(tmpl);

  // The console copy never converts an IO error into a success.
  CHECK_FALSE(ok);
  CHECK(why.find("cannot open") != std::string::npos);

  MESSAGE("captured " << captured.size() << " bytes of stderr on a FAILED file write");
  REQUIRE_MESSAGE(!captured.empty(),
                  "VLLM_RENDER_PHASE_LOG_STDERR printed nothing when the file write failed, "
                  "which is the one case the lane exists for");
  CHECK(captured.find("denoise") != std::string::npos);
  CHECK(captured.find("decode.video") != std::string::npos);
  log.Reset();
}

// ─── F10: the sampler STOPS when the last scope closes ───────────────────────
//
// THE FIRST UNIT CASE THIS INSTRUMENT HAS. Every other gate in this file reaches
// `PhaseLog` through a complete render, which costs about 40 s and cannot see a
// thread that is still running after the render ended. The row's spec said F10
// was ungated because "F10 is an absence — a thread that keeps running — which
// no assertion in this suite is positioned to observe". A fresh review refuted
// that in twelve lines, and this is those twelve lines: the sample COUNT is
// public (`PhaseLog::Samples()`), the worker increments it every 100 ms, and a
// worker that outlived its timeline therefore moves a number a test can read.
// No injected scheduler, no timing threshold that contention can cross — an idle
// process either takes samples or it does not.
//
// F10 itself is `render_phase_log.cpp`'s last `Close`: when nothing is live any
// more it takes the worker and joins it. Before the repair, only `Reset()` and
// the destructor ever stopped the thread, so a SERVER that rendered one clip
// kept a `/proc/self/statm` read under the process-wide mutex for the rest of
// its life and folded that idle time into the next table's sample count.
TEST_CASE("ltx2 phase log: the last Close stops the sampler") {
  using vllm::multimodal::phase::PhaseLog;
  PhaseLog& log = PhaseLog::Instance();
  log.Reset();
  log.Begin();
  const int64_t before = log.Samples();
  {
    const vllm::multimodal::phase::Scope work("f10.probe");
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  const int64_t at_close = log.Samples();
  // THE INSTRUMENT'S OWN PRECONDITION. With `VLLM_RENDER_PHASE_SAMPLER=0` no
  // worker is ever started, and "the count did not move" would then be true for
  // the wrong reason — a mute switch wearing a pass. 250 ms at a 100 ms period
  // is at least one tick on top of the Open and Close samples.
  REQUIRE_MESSAGE(at_close - before >= 3,
                  "the sampler took " << (at_close - before) << " samples across a 250 ms scope, "
                      "so it was not running and this case cannot observe whether it STOPS. Is "
                      "VLLM_RENDER_PHASE_SAMPLER=0 set?");
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  const int64_t idle = log.Samples();
  CHECK_MESSAGE(idle == at_close,
                "the sample count moved by " << (idle - at_close)
                    << " over 600 ms in which NOTHING was open, so the 100 ms sampler outlived "
                       "the timeline that started it. A process that renders once then serves "
                       "keeps that read under the process-wide mutex for life, and the next "
                       "render's table inherits the idle samples");
  log.Reset();
}

// ─── W0-live: the lane that runs WHILE the render is alive (issue #1413) ─────
//
// WHAT THIS GATES, and why the table above does not already gate it. That case
// reads `phase-log.json`, which `Ltx2VideoEngine::Generate` writes after
// `im.trace.completed = true` — the SUCCESS PATH ONLY. A render that is killed,
// aborted by a lease governor, or still running writes no table at all, and that
// is the population this campaign actually has: #1375 is `ABORT[92] PROJECTED
// OVERRUN` / `child exit=-15` / 0 frames, and `ltx25-decode-speed.md`'s two
// rungs are `EXIT=137` / 0 frames and `EXIT=1` / 0 frames. An instrument that
// reports on completion reported on none of them.
//
// So the deliverable here is different in kind: a line emitted AS EACH PHASE
// STARTS, so that the last line printed names the phase that is currently
// running. That is the whole difference between a working render and a hung one,
// which were byte-identical from outside for 2.5 hours at a time, and it is the
// difference a close-only emitter would still not make — the 3002 s conditioning
// stretch never closed.
//
// AND IT CAPTURES REAL STDERR, with `dup`/`dup2`, rather than installing a
// test-only sink. A sink would be a seam that could be armed here and absent in
// production, which is the exact defect class `.agents/reachability.md` is
// about; what is asserted below is what a user sees on the shipped default with
// nothing set.
//
// THE NON-VACUOUS HALF is the strictly-increasing forward index. An emitter that
// printed a constant `dit forward 1` would satisfy "the marker appears" and would
// report nothing — which is what the spike's external sampler did, and why
// #1375's `dit_runs=0` is not evidence of absence. The count has to MOVE.
//
// Nothing between the redirect and its restore may assert: a doctest `REQUIRE`
// that aborted the case in there would leave the process with stderr pointed at
// a temporary file for every case that follows.
#if !defined(_WIN32)
TEST_CASE("ltx2 video: a render through the ABI PRINTS its progress while it runs") {
  Workspace ws;
  const std::string audio_embeds = ws.paths.audio_embeds;
  const std::string max_phase = "0";
  // The CHECKPOINT CLASS rides the ABI like every other load extra (row
  // LTX25-CHECKPOINT-CLASS, #1137). No kind is declared here, so the recipe is
  // `distilled_two_stage`, whose upstream row is `Distilled only`.
  const std::string checkpoint_class = FixtureCheckpointClass("distilled_two_stage");
  const char* keys[] = {vllm::multimodal::kLtx2AudioPromptEmbedsExtra,
                        vllm::multimodal::kLtx2MaxPhaseExtra,
                        vllm::multimodal::kLtx2CheckpointClassExtra};
  const char* values[] = {audio_embeds.c_str(), max_phase.c_str(), checkpoint_class.c_str()};

  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = ws.paths.dit.c_str();
  mp.video_vae_path = ws.paths.video_vae.c_str();
  mp.audio_vae_path = ws.paths.audio_vae.c_str();
  mp.prompt_embeds_path = ws.paths.video_embeds.c_str();
  mp.extra_keys = keys;
  mp.extra_values = values;
  mp.n_extras = 3;
  mp.device = 0;

  const std::string out_dir = ws.root + "/progress_out";
  vllm_video_params gen = vllm_video_params_default();
  gen.width = 64;
  gen.height = 64;
  gen.num_frames = 9;
  gen.seed = 7;
  gen.has_seed = 1;
  gen.output_dir = out_dir.c_str();

  vllm_video_engine* engine = nullptr;
  vllm_video_result result;
  std::memset(&result, 0, sizeof(result));
  vllm_status loaded = VLLM_OK;
  vllm_status generated = VLLM_OK;
  std::string engine_error;
  std::string captured;

  // The LOAD is inside the capture as well as the generation: the timeline
  // starts at the load because the DiT staging is minutes at 22B, and a lane
  // that only spoke after the load would be silent for the phase #1021 is about.
  {
    std::FILE* cap = std::tmpfile();
    REQUIRE(cap != nullptr);
    std::fflush(stderr);
    const int saved = ::dup(STDERR_FILENO);
    REQUIRE(saved >= 0);
    REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);

    loaded = vllm_video_engine_load(&mp, &engine);
    if (loaded == VLLM_OK && engine != nullptr) {
      generated = vllm_video_generate(engine, &gen, &result);
    }
    if (vllm_last_error() != nullptr) engine_error = vllm_last_error();

    std::fflush(stderr);
    const int restored = ::dup2(saved, STDERR_FILENO);
    ::close(saved);
    std::rewind(cap);
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) captured.append(buf, n);
    std::fclose(cap);
    REQUIRE(restored >= 0);
  }

  INFO(engine_error);
  REQUIRE(loaded == VLLM_OK);
  REQUIRE(engine != nullptr);
  REQUIRE(generated == VLLM_OK);

  // ── the lines ────────────────────────────────────────────────────────────
  std::vector<std::string> lines;
  {
    std::istringstream stream(captured);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.rfind("[render] ", 0) == 0) lines.push_back(line);
    }
  }
  MESSAGE("captured " << lines.size() << " [render] lines of " << captured.size()
                      << " bytes of stderr");
  REQUIRE_MESSAGE(!lines.empty(),
                  "the render printed NOTHING while it ran; a working render and a hung one "
                  "are still the same observation (#1413)");

  // Every line carries the elapsed clock, and it never goes backwards. A `t=`
  // that reset would mean two timelines interleaved in one capture.
  double previous_t = -1.0;
  for (const std::string& line : lines) {
    INFO(line);
    const size_t at = line.find(" t=");
    REQUIRE_MESSAGE(at != std::string::npos, "a progress line carries no elapsed clock");
    const double t = std::atof(line.c_str() + at + 3);
    CHECK(t >= previous_t);
    previous_t = t;
  }

  // ── the boundaries, opened AND closed ────────────────────────────────────
  //
  // The OPEN marker is the one that matters and it is asserted by name: a close
  // marker alone names only phases that finished, and every render this row
  // exists for stopped inside a phase that never did.
  auto index_of = [&lines](const std::string& needle) -> long {
    for (size_t i = 0; i < lines.size(); ++i) {
      if (lines[i].find(needle) != std::string::npos) return static_cast<long>(i);
    }
    return -1;
  };
  // MATCH THE WHOLE NAME FIELD, NOT A PREFIX OF IT. The emitter writes the name
  // with `%-24s`, so every phase shorter than 24 characters is followed by at
  // least one space, and `"- denoise "` cannot match `- denoise.step`. Written as
  // a helper because a prefix match here is not a stylistic point: W0's anchor
  // scopes added `denoise.step` and `decode.video.chunk` beneath the two leaves
  // this case asserts against, and a bare `find("- denoise")` then resolved to
  // the FIRST sub-scope close, several lines before the forwards it was supposed
  // to bracket. The case failed on seven true statements about the wrong line.
  auto phase_line = [&index_of](const char* sign, const std::string& name) -> long {
    return index_of(std::string(sign) + " " + name + " ");
  };
  for (const std::string name : {"load.dit", "denoise", "decode.video", "artifacts.frames"}) {
    // `std::string`, never `const char*`: doctest stringifies a streamed `char*`
    // as a BOOL, so a failure here would have printed "no '1' line".
    CHECK_MESSAGE(phase_line("+", name) >= 0, "no OPEN line for phase '" << name << "'");
    CHECK_MESSAGE(phase_line("-", name) >= 0, "no CLOSE line for phase '" << name << "'");
    CHECK_MESSAGE(phase_line("-", name) > phase_line("+", name),
                  "phase '" << name << "' closed before it opened");
  }

  // A close line reports what the phase cost, which is the number a reader takes
  // off a killed run's log without waiting for a table that will never be
  // written.
  const long dit_close = phase_line("-", "load.dit");
  REQUIRE(dit_close >= 0);
  CHECK_MESSAGE(lines[static_cast<size_t>(dit_close)].find(" dur=") != std::string::npos,
                "a phase closed without saying what it cost: "
                    << lines[static_cast<size_t>(dit_close)]);

  // ── the per-forward counter, and the proof that it MOVES ─────────────────
  std::vector<long> forward_indices;
  std::vector<size_t> forward_lines;
  for (size_t i = 0; i < lines.size(); ++i) {
    const size_t at = lines[i].find("dit forward ");
    if (at == std::string::npos) continue;
    forward_indices.push_back(std::atol(lines[i].c_str() + at + 12));
    forward_lines.push_back(i);
  }
  REQUIRE_MESSAGE(forward_indices.size() >= 2,
                  "the denoise loop printed " << forward_indices.size()
                                              << " forward lines; #1375 measures ~162 s per "
                                                 "forward and 60 of them is structural, so a "
                                                 "loop that prints once is a loop that hangs");
  for (size_t i = 1; i < forward_indices.size(); ++i) {
    INFO(lines[forward_lines[i]]);
    CHECK_MESSAGE(forward_indices[i] > forward_indices[i - 1],
                  "the forward counter did not move: " << forward_indices[i - 1] << " then "
                                                       << forward_indices[i]);
    // `last=` is the per-forward cost #1375 could only get as a wall-clock
    // interval between GPU busy/idle edges. Absent on the first tick by
    // construction, required on every one after it.
    CHECK_MESSAGE(lines[forward_lines[i]].find(" last=") != std::string::npos,
                  "a forward tick carries no per-forward cost: " << lines[forward_lines[i]]);
  }
  // The sampler step, as a FRACTION. Without the denominator a reader knows the
  // render moved and not whether it is a tenth or nine tenths of the way.
  //
  // Two statements rather than one `&&`: doctest's expression decomposer
  // static-asserts "Expression Too Complex Please Rewrite As Binary Comparison"
  // on a conjunction, so the compound form does not build — and a mutation that
  // fails to build reads exactly like a passing test.
  const std::string first_tick = lines[forward_lines[0]];
  CHECK_MESSAGE(first_tick.find(" step ") != std::string::npos,
                "a forward tick names no sampler step: " << first_tick);
  CHECK_MESSAGE(first_tick.find("/") != std::string::npos,
                "the sampler step carries no /N denominator, so a reader knows the render "
                "moved and not whether it is a tenth or nine tenths of the way: "
                    << first_tick);

  // ── the ticks are INSIDE the phase they claim ────────────────────────────
  const long denoise_open = phase_line("+", "denoise");
  const long denoise_close = phase_line("-", "denoise");
  REQUIRE(denoise_open >= 0);
  REQUIRE(denoise_close >= 0);
  for (size_t i = 0; i < forward_lines.size(); ++i) {
    const long at = static_cast<long>(forward_lines[i]);
    // Two comparisons, never one `&&`: doctest static-asserts on a conjunction.
    CHECK_MESSAGE(at > denoise_open,
                  "a DiT forward tick was printed BEFORE the denoise phase opened: "
                      << lines[forward_lines[i]]);
    CHECK_MESSAGE(at < denoise_close,
                  "a DiT forward tick was printed AFTER the denoise phase closed: "
                      << lines[forward_lines[i]]);
  }

  vllm_video_result_free(&result);
  vllm_video_engine_free(engine);
}
#endif  // !_WIN32

// ─── the tick's OWN contract: emitted at the call, not at the return ────────
//
// WHY THIS CASE EXISTS. The case above renders through the ABI and asserts that
// the ticks appear, that the counter moves and that they sit inside `denoise`.
// It cannot see the decision the design rests on. On a render that COMPLETES,
// calling `Tick` before the forward and calling it after produce the same lines,
// the same intervals and the same ordering against the phase boundaries — moving
// the production call below the forward leaves that case green. The two
// placements differ only on a run that dies mid-unit, and no gate here can stage
// a `SIGKILL` inside `Ltx2DitForwardDevice` through `vllm.h`.
//
// So this pins the half that IS testable, which is the half the production
// placement rests on: `Tick` reaches the real fd 2 and FLUSHES before the
// caller's next statement runs. If the line were buffered, or emitted lazily, or
// deferred to the end of the phase, then announcing a unit before starting it
// would buy nothing — the announcement would die in the buffer with the process.
// Below, a fake unit of work writes its own marker to the same stream, and the
// tick has to be there first every time.
//
// AND THE SHAPE THAT MATTERS IS THE LAST ONE. The loop announces unit 4 and then
// stops without finishing it, which is what a killed render looks like from
// stderr: the log's last word is the unit that was in flight. Under an
// after-the-work emitter that capture would end at unit 3, and #1375 would still
// be un-attributed.
//
// Nothing between the redirect and its restore may assert, for the reason the
// case above gives: a `REQUIRE` that aborted in there would leave every later
// case writing stderr into a temporary file.
#if !defined(_WIN32)
TEST_CASE("ltx2 phase log: a live tick is FLUSHED BEFORE the work it announces") {
  namespace phase = vllm::multimodal::phase;
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Begin();

  const int kFinished = 3;  // units 1..3 complete; unit 4 is announced and never does
  std::string captured;
  {
    std::FILE* cap = std::tmpfile();
    REQUIRE(cap != nullptr);
    std::fflush(stderr);
    const int saved = ::dup(STDERR_FILENO);
    REQUIRE(saved >= 0);
    REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);

    for (int k = 1; k <= kFinished; ++k) {
      phase::Tick("unit", k, "step " + std::to_string(k) + "/4");
      // The work. It writes its own marker to the SAME stream, so the ordering
      // question is answered by the bytes rather than by a clock.
      std::fprintf(stderr, "[work] unit %d finished\n", k);
      std::fflush(stderr);
    }
    // Announced, then the "process dies" here. No marker follows.
    phase::Tick("unit", kFinished + 1, "step 4/4");

    std::fflush(stderr);
    const int restored = ::dup2(saved, STDERR_FILENO);
    ::close(saved);
    std::rewind(cap);
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) captured.append(buf, n);
    std::fclose(cap);
    REQUIRE(restored >= 0);
  }
  log.Reset();

  MESSAGE("captured " << captured.size() << " bytes of stderr from the tick contract case");
  REQUIRE_MESSAGE(!captured.empty(), "phase::Tick emitted nothing at all");

  std::vector<std::string> lines;
  {
    std::istringstream stream(captured);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.rfind("[render] ", 0) == 0) lines.push_back(line);
      if (line.rfind("[work] ", 0) == 0) lines.push_back(line);
    }
  }
  auto index_of = [&lines](const std::string& needle) -> long {
    for (size_t i = 0; i < lines.size(); ++i) {
      if (lines[i].find(needle) != std::string::npos) return static_cast<long>(i);
    }
    return -1;
  };

  // Each announcement precedes the work it announces. `std::string` and never a
  // `char*` in a stream: doctest stringifies a streamed `char*` as a bool.
  for (int k = 1; k <= kFinished; ++k) {
    const std::string tick = "unit " + std::to_string(k) + " ";
    const std::string work = "[work] unit " + std::to_string(k) + " finished";
    const long at_tick = index_of(tick);
    const long at_work = index_of(work);
    CHECK_MESSAGE(at_tick >= 0, "no tick line for unit " << k);
    CHECK_MESSAGE(at_work >= 0, "no work marker for unit " << k);
    CHECK_MESSAGE(at_tick < at_work,
                  "the tick for unit " << k << " was not flushed before the work it announces");
  }

  // The killed-run shape: unit 4 was announced, never finished, and the capture
  // still names it — and names it LAST.
  const long announced = index_of("unit 4 ");
  CHECK_MESSAGE(announced >= 0,
                "the unit that was in flight when the run stopped is missing from the log, "
                "which is what an after-the-work emitter produces");
  CHECK_MESSAGE(index_of("[work] unit 4 finished") < 0,
                "the case did not stage an unfinished unit, so it proves nothing");
  REQUIRE(!lines.empty());
  CHECK_MESSAGE(lines.back().find("unit 4 ") != std::string::npos,
                "the last line of the capture is not the unit in flight: " << lines.back());

  // `last=` is the per-unit interval, absent on the first tick by construction
  // and present on every one after it.
  const long first = index_of("unit 1 ");
  REQUIRE(first >= 0);
  CHECK(lines[static_cast<size_t>(first)].find(" last=") == std::string::npos);
  const long second = index_of("unit 2 ");
  REQUIRE(second >= 0);
  CHECK_MESSAGE(lines[static_cast<size_t>(second)].find(" last=") != std::string::npos,
                "the second tick of a unit carries no interval: "
                    << lines[static_cast<size_t>(second)]);
}
#endif  // !_WIN32

// ─── `last=` belongs to ONE generation ───────────────────────────────────────
//
// The per-unit tick clock is keyed by unit NAME and lives in the `PhaseLog`, so
// its lifetime is a decision rather than an accident. `Begin` and `Reset` clear
// it. `SetRender` — the call a driver makes between two generations in one
// process — did not, so the second render's first DiT forward would have carried
// a `last=` measuring the gap between two renders. That number is shaped exactly
// like one very slow forward, on the lane whose entire deliverable is per-forward
// seconds, in a process (a server) that renders more than once.
//
// The wait is what makes this refutable. Without it the stale interval would be
// microseconds and indistinguishable from a fresh one; 150 ms of sleep between
// the two renders means an uncleared clock reports a number nobody could mistake
// for a forward.
#if !defined(_WIN32)
TEST_CASE("ltx2 phase log: a SECOND render's first tick carries no interval from the first") {
  namespace phase = vllm::multimodal::phase;
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Begin();

  std::string captured;
  {
    std::FILE* cap = std::tmpfile();
    REQUIRE(cap != nullptr);
    std::fflush(stderr);
    const int saved = ::dup(STDERR_FILENO);
    REQUIRE(saved >= 0);
    REQUIRE(::dup2(::fileno(cap), STDERR_FILENO) >= 0);

    log.SetRender(1);
    phase::Tick("dit forward", 1, "render one");
    phase::Tick("dit forward", 2, "render one");
    // The gap between two generations, long enough to be unmistakable.
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    log.SetRender(2);
    phase::Tick("dit forward", 1, "render two");

    std::fflush(stderr);
    const int restored = ::dup2(saved, STDERR_FILENO);
    ::close(saved);
    std::rewind(cap);
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), cap)) > 0) captured.append(buf, n);
    std::fclose(cap);
    REQUIRE(restored >= 0);
  }
  log.Reset();

  std::vector<std::string> lines;
  {
    std::istringstream stream(captured);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.rfind("[render] ", 0) == 0) lines.push_back(line);
    }
  }
  MESSAGE("captured " << lines.size() << " tick lines across two renders");
  REQUIRE(lines.size() == 3);

  // THE INSTRUMENT'S OWN PRECONDITION: the second tick of render one DOES carry
  // an interval, so "no `last=`" below is a cleared clock and not a lane that is
  // switched off. `VLLM_RENDER_PROGRESS=0` would have failed the size check above
  // rather than passing this one silently.
  REQUIRE_MESSAGE(lines[1].find(" last=") != std::string::npos,
                  "the second tick of the FIRST render carries no interval, so this case cannot "
                  "observe whether the clock is cleared: " << lines[1]);
  CHECK_MESSAGE(lines[2].find(" last=") == std::string::npos,
                "the first tick of the SECOND render carries an interval spanning the gap since "
                "the first render ended, which reads as one very slow forward: " << lines[2]);
}
#endif  // !_WIN32

// ─── the SHIPPED checkpoints, when the box has them ─────────────────────────
//
// Everything above runs over a reduced fixture, which proves the composition and
// says nothing about the 23 GB of bytes Lightricks actually ships. This case
// reads those bytes when `LTX2_CHECKPOINT_ROOT` points at them, and is SKIPPED
// otherwise rather than silently passing: a gate that quietly does nothing when
// its input is absent is how "we tested the real checkpoint" becomes untrue.
//
// HOW FAR THAT ANNOUNCEMENT CARRIES, stated so nobody over-reads it. The
// `MESSAGE` below is printed by the doctest binary's own run, so `test_ltx2_video`
// executed directly says SKIPPED on its own output. `ctest` captures a passing
// test's output and prints only the pass line, so under `ctest` this case is
// indistinguishable from one that ran. Read "the shipped checkpoints were read"
// off the binary's output or off an explicitly set env, never off a green ctest
// row.
//
// The env is deliberately NOT the tree-wide `CHECKPOINT_ROOT`. That one "declares
// an INTENT, not a behaviour" and nothing in the tree reads it (`.env.example`),
// while this needs a path to one specific publisher tree whose internal layout
// (`diffusion_models/…`) the case walks. Pointing it at the shared root would
// make this its first reader and change what that variable means.
//
// It deliberately stops short of the DiT FORWARD. At the shipped 21.00B geometry
// the f32 parity forward needs ~76 GB of weights and ~2.6e14 FLOPs per step, and
// the device path cannot feed it (see the `device = 1` refusal). What is checked
// is everything before that line: that the first-party NVFP4 DiT's header
// resolves onto the L2 contract, and that BOTH VAEs and the upsampler load and
// configure from their own embedded metadata.
//
// WHICH DiT: `Lightricks/LTX-2.5` `ltx-2.5-22b-distilled-transformer-nvfp4`.
// It is NOT interchangeable with the `vonkaiser` FP8 copy — they differ in a
// TRAINED `keyframes_abs_pos_embedding` (spec section 3.1) — so the file this
// case reads is named here and in every report of its result.
// WHEN QUOTING THIS SUITE'S ASSERTION COUNT, QUOTE THE CONFIGURATION WITH IT.
// This case skips by default, and it is most of the suite: with the variable
// UNSET the binary measures 37 cases / 784 assertions, and with it SET, 37 /
// 9031 — measured 2026-08-15, exit 0 both ways. So the CASE count is identical in
// both configurations and only the assertion count moves; an unchanged case count
// across a change therefore says nothing about whether the real headers were read.
// Quote the number WITH its configuration and its date: these figures move
// whenever a case is added here, and they already have (they read 30 / 502 and
// 30 / 8734 before the keyframe-bias port, issue #658).
TEST_CASE("ltx2 video: the SHIPPED Lightricks checkpoints parse and load") {
  const char* root_env = std::getenv("LTX2_CHECKPOINT_ROOT");
  if (root_env == nullptr) {
    MESSAGE("SKIPPED: set LTX2_CHECKPOINT_ROOT to the Lightricks/LTX-2.5 tree to run this");
    return;
  }
  const std::string root = root_env;

  SUBCASE("the first-party NVFP4 DiT resolves onto the L2 contract") {
    const std::string path =
        root + "/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kFp8;
    vllm::Ltx2DitParams from_shapes = vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
    // MEASURED from this file's own header: it carries `prompt_adaln_single`, so
    // the manifest parser resolves `use_prompt_adaln_single = TRUE` — and nothing
    // clears it any more (.agents/specs/ltx25-prompt-adaln.md, issue #644). This
    // line used to force it false on BOTH sides of the comparison below, which is
    // what made a config/shape disagreement about it unobservable.
    CHECK(from_shapes.use_prompt_adaln_single);
    CHECK(quant == vllm::Ltx2DitQuant::kNvfp4);
    CHECK(from_shapes.num_layers == 48);
    CHECK(from_shapes.inner_dim() == 4096);
    CHECK(from_shapes.audio_inner_dim() == 2048);
    CHECK(from_shapes.in_channels == 128);
    CHECK(from_shapes.audio_in_channels == 128);

    // The declared config, which is what the engine adopts and what the SHAPES
    // cannot see. The two must agree on the weight contract.
    CHECK(vllm::Ltx2ReadCheckpointModelVersion(file) == "2.5.0");
    nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    // The shipped DiT DECLARES `use_keyframes_abs_pos_embedding: true` while
    // carrying NO tensor for it — which is upstream-LEGAL and means "apply
    // nothing": the parameter is built on the meta device and
    // `supports_keyframes_abs_pos_embedding` is False both before and after the
    // load (model.py:166-173; reproduce with
    // scripts/measure-ltx2-keyframes-meta.py).
    //
    // This test used to CLEAR the flag by hand here, mirroring an engine that
    // cleared it under `allow_unported_modules`. Both are gone (row
    // LTX25-KEYFRAMES-ABS-POS, issue #658): `Ltx2AdoptDeclaredDitParams` resolves
    // it against the file's own shapes, so the declared config is adopted
    // VERBATIM and the contracts agree with no hand edit at all.
    REQUIRE(config["transformer"]["use_keyframes_abs_pos_embedding"].get<bool>());
    REQUIRE_FALSE(from_shapes.use_keyframes_abs_pos_embedding);  // the file carries no tensor
    const vllm::Ltx2DitParams declared = vllm::Ltx2AdoptDeclaredDitParams(
        config, from_shapes, "the shipped NVFP4 DiT's own __metadata__[\"config\"]");
    // RESOLVED to FALSE, which is `supports_...`. Not a refusal, and not a
    // synthesised zero — the two failure modes spec §6 names.
    CHECK_FALSE(declared.use_keyframes_abs_pos_embedding);
    // The shipped config OMITS `use_prompt_adaln_single`, so it resolves to
    // upstream's TRUE default (model_configurator.py:76) — which is what the
    // file's own tensors say. Asserted rather than forced: the two sides of the
    // contract comparison below must AGREE about it, not be made to.
    CHECK(declared.use_prompt_adaln_single);
    CHECK(declared.double_precision_rope);              // frequencies_precision float64
    CHECK(declared.av_ca_timestep_scale_multiplier == 1000);
    CHECK(declared.apply_gated_attention);
    CHECK(declared.cross_attention_adaln);
    CHECK(!declared.ff_bias);
    const std::vector<vllm::Ltx2TensorSpec> a = vllm::EnumerateLtx2DitTensors(from_shapes);
    const std::vector<vllm::Ltx2TensorSpec> b = vllm::EnumerateLtx2DitTensors(declared);
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
      INFO("tensor " << i);
      CHECK(a[i].name == b[i].name);
      CHECK(a[i].shape == b[i].shape);
    }
    MESSAGE("shipped NVFP4 DiT: " << a.size() << " contract tensors, "
            << file.Names().size() << " in the file");
  }

  // THE CLAIM THIS ROW EXISTS TO MAKE TRUE (row LTX25-KEYFRAMES-ABS-POS, issue
  // #658): the shipped DiT loads INSIDE THE CONTRACT — no `allow_unported_modules`
  // — which neither shipped copy could do before. `Ltx2LoadDitFromSafetensors`
  // with default options is precisely "no opt-in".
  //
  // The full load reads ~19 GB. It is the same file the subcase above parses; if
  // this box cannot hold it, that shows up as a load failure and not as a pass.
  SUBCASE("the first-party NVFP4 DiT loads with NO allow_unported_modules") {
    const std::string path =
        root + "/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
    CHECK(ck.unported.empty());
    // Absent from the file, so nothing is bound and nothing will be applied.
    CHECK_FALSE(ck.params.use_keyframes_abs_pos_embedding);
    CHECK(ck.weights.keyframes_abs_pos_embedding.data == nullptr);
    MESSAGE("shipped NVFP4 DiT loaded inside the contract, unported=" << ck.unported.size());
  }

  // THE FULL (dev) TRANSFORMER — the arm issue #1148 could not read at all.
  //
  // HEADER ONLY, DELIBERATELY. `SafetensorsFile::Open` mmaps, and
  // `Ltx2ParseDitParamsFromCheckpoint` touches the 677,616-byte JSON header and
  // no payload, so this subcase costs kilobytes. A full
  // `Ltx2LoadDitFromSafetensors` on this file materializes ~42 GB of host bf16,
  // which is not something the CPU gate can hold, so the real-weights
  // MATERIALIZATION stays owed and is recorded as such in the row's spec rather
  // than skipped quietly. What this DOES establish is that `PlanDit` resolves
  // the real file — 4349 tensors, 4059 BF16 / 290 F32, zero scale sidecars — to
  // `kNone` and recovers LTX-2.5's geometry from shapes that were never packed.
  SUBCASE("the FULL bf16 dev DiT resolves onto the L2 contract") {
    const std::string path =
        root + "/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    vllm::Ltx2DitQuant quant = vllm::Ltx2DitQuant::kFp8;  // never the expected value
    const vllm::Ltx2DitParams from_shapes =
        vllm::Ltx2ParseDitParamsFromCheckpoint(file, &quant);
    CHECK(quant == vllm::Ltx2DitQuant::kNone);
    CHECK(from_shapes.num_layers == 48);
    CHECK(from_shapes.inner_dim() == 4096);
    CHECK(from_shapes.audio_inner_dim() == 2048);
    CHECK(from_shapes.in_channels == 128);
    CHECK(from_shapes.audio_in_channels == 128);
    CHECK(from_shapes.use_prompt_adaln_single);
    // TRAINED here, unlike the first-party NVFP4 copy which declares the flag and
    // carries no tensor: the dev file stores `keyframes_abs_pos_embedding` as
    // BF16 [1, 4096].
    CHECK(from_shapes.use_keyframes_abs_pos_embedding);
    CHECK_FALSE(from_shapes.ff_bias);
    CHECK(from_shapes.audio_ff_bias);

    // Not one U8 and not one F8_E4M3 anywhere, which is what made the old
    // refusal fire. Counted from the file rather than asserted from the row.
    int64_t u8 = 0, f8 = 0, bf16 = 0, f32 = 0, sidecars = 0;
    for (const std::string& name : file.Names()) {
      const std::string& dtype = file.Get(name).dtype;
      if (dtype == "U8") ++u8;
      if (dtype == "F8_E4M3") ++f8;
      if (dtype == "BF16") ++bf16;
      if (dtype == "F32") ++f32;
      if (name.size() >= 6 && name.compare(name.size() - 6, 6, "_scale") == 0) ++sidecars;
    }
    CHECK(u8 == 0);
    CHECK(f8 == 0);
    CHECK(sidecars == 0);
    CHECK(bf16 == 4059);
    CHECK(f32 == 290);

    // And the declared config describes the SAME weight contract as the shapes,
    // which is the check the engine performs before adopting it.
    CHECK(vllm::Ltx2ReadCheckpointModelVersion(file) == "2.5.0");
    const nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    const vllm::Ltx2DitParams declared = vllm::Ltx2AdoptDeclaredDitParams(
        config, from_shapes, "the dev bf16 DiT's own __metadata__[\"config\"]");
    CHECK(declared.double_precision_rope);
    CHECK(declared.av_ca_timestep_scale_multiplier == 1000);
    MESSAGE("shipped dev bf16 DiT: quant=kNone, " << file.Names().size()
            << " tensors, " << bf16 << " BF16 / " << f32 << " F32");
  }
}

// The OTHER shipped DiT, which lives under a different publisher root and so
// takes its own env. It is the one that carries the TRAINED
// `keyframes_abs_pos_embedding` — `F8_E4M3 [1, 4096]` with a scalar `F32` scale,
// 4096 of 4096 bytes non-zero — and it was refused from the opposite direction:
// "the checkpoint carries modules this port does NOT carry".
//
// CI SETS NEITHER ENV (issue #673), so this is host-local evidence, not a gate.
TEST_CASE("ltx2 video: the SHIPPED vonkaiser FP8 DiT loads with NO allow_unported_modules") {
  const char* dit_env = std::getenv("LTX2_FP8_DIT");
  if (dit_env == nullptr) {
    MESSAGE("SKIPPED: set LTX2_FP8_DIT to the vonkaiser ltx-2.5-22b-distilled-fp8.safetensors");
    return;
  }
  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(std::string(dit_env));
  // It carries no `__metadata__` at all, which is why its config always arrives
  // separately and why the manifest is the only evidence about this flag.
  CHECK(file.Metadata().count("config") == 0);

  const vllm::Ltx2DitCheckpoint ck = vllm::Ltx2LoadDitFromSafetensors(file);
  CHECK(ck.unported.empty());
  // RESOLVED TRUE from the file's own shapes — `supports_...` holds here.
  CHECK(ck.params.use_keyframes_abs_pos_embedding);
  REQUIRE(ck.weights.keyframes_abs_pos_embedding.data != nullptr);
  REQUIRE(ck.weights.keyframes_abs_pos_embedding.rank == 2);
  CHECK(ck.weights.keyframes_abs_pos_embedding.shape[0] == 1);
  CHECK(ck.weights.keyframes_abs_pos_embedding.shape[1] == ck.params.inner_dim());
  // Dequantized through the ONE existing FP8 convention — F8_E4M3 plus a scalar
  // F32 `<name>_scale`, `DequantFp8ToBf16` — so the view is BF16.
  CHECK(ck.weights.keyframes_abs_pos_embedding.dtype == vt::DType::kBF16);
  // TRAINED, not `torch.zeros`. A zero bias would be an exact no-op because the
  // term is ADDED, so this is the assertion that makes the port matter at all.
  {
    const uint16_t* p = ck.weights.keyframes_abs_pos_embedding.Ptr<uint16_t>();
    int64_t nonzero = 0;
    for (int64_t i = 0; i < ck.params.inner_dim(); ++i) {
      if (p[i] != 0) ++nonzero;
    }
    MESSAGE("shipped FP8 keyframes_abs_pos_embedding: " << nonzero << " of "
            << ck.params.inner_dim() << " non-zero");
    CHECK(nonzero > 0);
  }
}

TEST_CASE("ltx2 video: the SHIPPED Lightricks VAEs and upsampler load") {
  const char* root_env = std::getenv("LTX2_CHECKPOINT_ROOT");
  if (root_env == nullptr) {
    MESSAGE("SKIPPED: set LTX2_CHECKPOINT_ROOT to the Lightricks/LTX-2.5 tree to run this");
    return;
  }
  const std::string root = root_env;

  SUBCASE("the Conv video VAE loads and configures from its own metadata") {
    const std::string path = root + "/vae/ltx-2.5-video-vae-conv-bf16.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    vllm::Ltx2VideoDecoderKind kind = vllm::Ltx2VideoDecoderKind::kDiffusion;
    const vllm::Ltx2ConvVideoDecoderConfig cfg =
        vllm::Ltx2ParseConvVideoDecoderConfig(vllm::Ltx2ReadCheckpointConfig(file), &kind);
    CHECK(kind == vllm::Ltx2VideoDecoderKind::kConv);
    CHECK(cfg.in_channels == 128);
    CHECK(cfg.out_channels == 3);
    CHECK(cfg.patch_size == 4);
    CHECK(cfg.base_channels == 128);
    CHECK(!cfg.timestep_conditioning);
    CHECK(cfg.decoder_blocks.size() == 9);
    // The block list has to multiply out to VIDEO_SCALE_FACTORS, because that
    // constant is what the pipeline derives every latent shape from.
    // `multiplier` is the CHANNEL reduction, not the stride: the resample factor
    // is fixed per block KIND (ltx2_video_vae.cpp:662-668 — compress_space is a
    // 2x2 spatial stride, compress_time a 2x temporal one, compress_all both).
    // Reading `multiplier` as the stride gives 16 and 4 here, which is what this
    // check first did, and which would have made every latent shape wrong by 2x.
    int64_t spatial = cfg.patch_size, temporal = 1;
    for (const vllm::Ltx2VideoDecoderBlock& b : cfg.decoder_blocks) {
      if (b.name == "compress_space") spatial *= 2;
      if (b.name == "compress_time") temporal *= 2;
      if (b.name == "compress_all") {
        spatial *= 2;
        temporal *= 2;
      }
    }
    CHECK(spatial == 32);
    CHECK(temporal == 8);
    const vllm::Ltx2VaeWeights weights =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeDecoderKeyRules());
    CHECK(weights.Has("conv_in.conv.weight"));
    CHECK(weights.Has("conv_out.conv.weight"));
    CHECK(weights.Has("per_channel_statistics.std-of-means"));
    // The encoder half of the file is DROPPED by the key rules, exactly as
    // upstream's SDOps drop it.
    CHECK(!weights.Has("encoder.conv_in.conv.weight"));
    MESSAGE("shipped conv video VAE: " << weights.tensors.size() << " decoder tensors");

    // ...and the ENCODER half of the SAME file resolves through the other
    // filter (row LTX25-IMAGE-COND, #644). This is the only place the encoder
    // load path meets a real shipped checkpoint rather than the fixture, so it
    // is the only place the CHANNEL arithmetic can be wrong in a way the fixture
    // agrees with: `latent_channels` is 128 while the top-level `out_channels`
    // is 3, and reading the second builds a 3-channel-latent encoder that runs.
    REQUIRE(vllm::Ltx2CheckpointHasVideoEncoder(file.Names()));
    const vllm::Ltx2ConvVideoEncoderConfig enc =
        vllm::Ltx2ParseConvVideoEncoderConfig(vllm::Ltx2ReadCheckpointConfig(file));
    CHECK(enc.out_channels == 128);
    CHECK(enc.in_channels == 3);
    CHECK(enc.patch_size == cfg.patch_size);
    // The encoder's block list must multiply out to the SAME scale factors the
    // decoder's does, or an encoded image does not fit the grid it is placed in.
    CHECK(vllm::Ltx2VideoSpatialScaleFactor(enc.encoder_blocks, enc.patch_size) == spatial);
    CHECK(vllm::Ltx2VideoTemporalScaleFactor(enc.encoder_blocks) == temporal);
    const vllm::Ltx2VaeWeights enc_weights =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeEncoderKeyRules());
    CHECK(enc_weights.Has("conv_in.conv.weight"));
    CHECK(enc_weights.Has("conv_out.conv.weight"));
    // The encoder normalizes its output by these (video_vae.py:336), so the
    // filter has to carry them even though they are not `encoder.*` keys.
    CHECK(enc_weights.Has("per_channel_statistics.std-of-means"));
    // And the DECODER's half must be dropped, or the two bags would collide on
    // names like `conv_in.conv.weight` and bind half a model to the other half.
    CHECK(!enc_weights.Has("decoder.conv_in.conv.weight"));
    MESSAGE("shipped conv video VAE: " << enc_weights.tensors.size() << " encoder tensors");
  }

  SUBCASE("the audio VAE and its BWE vocoder load and configure") {
    const std::string path = root + "/vae/ltx-2.5-audio-vae-bf16.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const nlohmann::json config = vllm::Ltx2ReadCheckpointConfig(file);
    const vllm::Ltx2AudioDecoderConfig audio = vllm::Ltx2ParseAudioDecoderConfig(config);
    CHECK(audio.z_channels == 8);   // == the audio latent's channel count
    CHECK(audio.mel_bins == 64);    // == the vocoder's hardcoded 128 / 2 channels
    CHECK(audio.out_ch == 2);
    CHECK(!audio.mid_block_add_attention);
    const vllm::Ltx2VocoderBweConfig voc = vllm::Ltx2ParseVocoderBweConfig(config);
    CHECK(voc.input_sampling_rate == 16000);
    CHECK(voc.output_sampling_rate == 48000);
    CHECK(voc.hop_length == 80);
    CHECK(voc.n_mel_channels == 64);
    CHECK(voc.vocoder.amp);
    CHECK(voc.vocoder.snakebeta);
    CHECK(voc.vocoder.apply_final_activation);
    CHECK(!voc.bwe_generator.apply_final_activation);
    const vllm::Ltx2VaeWeights decoder =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2AudioVaeDecoderKeyRules());
    const vllm::Ltx2VaeWeights vocoder =
        vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VocoderKeyRules());
    CHECK(decoder.Has("conv_in.conv.weight"));
    CHECK(decoder.Has("per_channel_statistics.std-of-means"));
    CHECK(vocoder.Has("vocoder.conv_pre.weight"));
    CHECK(vocoder.Has("bwe_generator.conv_pre.weight"));
    CHECK(vocoder.Has("mel_stft.mel_basis"));
    MESSAGE("shipped audio VAE: " << decoder.tensors.size() << " decoder + "
            << vocoder.tensors.size() << " vocoder tensors");
  }

  SUBCASE("the latent spatial x2 upsampler loads and configures") {
    const std::string path =
        root + "/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
    const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
    const vllm::Ltx2UpsamplerConfig cfg =
        vllm::Ltx2ParseUpsamplerConfig(vllm::Ltx2ReadCheckpointConfig(file));
    CHECK(cfg.in_channels == 128);
    CHECK(cfg.mid_channels == 1024);
    CHECK(cfg.spatial_upsample);
    CHECK(!cfg.temporal_upsample);
    CHECK(cfg.spatial_scale == doctest::Approx(2.0).scale(0.0));
    const vllm::Ltx2VaeWeights weights = vllm::Ltx2LoadVaeWeights(file);
    // Every tensor the contract asks for is in the file, by NAME.
    int64_t missing = 0;
    std::string first_missing;
    for (const vllm::Ltx2UpsamplerTensorSpec& spec :
         vllm::EnumerateLtx2UpsamplerTensors(cfg)) {
      if (!weights.Has(spec.name)) {
        ++missing;
        if (first_missing.empty()) first_missing = spec.name;
      }
    }
    INFO("first missing: " << first_missing);
    CHECK(missing == 0);
    MESSAGE("shipped upsampler: " << weights.tensors.size() << " tensors");
  }
}

// ─── the payload has NO alignment guarantee (issue #674) ────────────────────
//
// A tensor's first byte sits at `8 + <JSON header length> + <sum of the
// preceding tensors' sizes>`. Not one of those three terms is required to be
// even, so a BF16 tensor beginning on an ODD address is an ordinary safetensors
// file and not a corrupt one. `Ltx2LoadVaeWeights` formed a `const uint16_t*`
// over that address and dereferenced it, which is undefined behaviour on every
// target and a real fault on the strict-alignment ones this project builds for
// (`build-test-cpu-arm64`, Jetson/Orin sm_110). UBSan reported it as
//
//   ltx2_loader.cpp:1288:91: runtime error: load of misaligned address ...
//   for type 'const uint16_t', which requires 2 byte alignment
//
// and the `sanitize-cpu (address,undefined)` lane had been red on it since
// `cefacd2d0`. Third recurrence of one class: #301 (closed, and the source of
// the `vt::LoadUnaligned` seam) and #627 (`qwen3_5_weights.cpp`) are the others,
// and `minimax_h3_vae_loader.cpp:87-101` already carries the repair AND the
// reason in prose.
//
// WHY THIS CASE EXISTS AT ALL, given the suite above already reached the defect:
// it reached it BY ACCIDENT. `ltx2_fixture`'s JSON header happens to land one
// VAE tensor on an odd byte today, and any rename or reshape in that fixture
// silently retires the coverage while leaving every assertion green. So here the
// odd offset is FORCED, and the parity it depends on is ASSERTED — an edit that
// makes the address even fails the REQUIRE instead of passing while covering
// nothing.
namespace {

// A bare temp directory. Deliberately NOT `Workspace`: writing the whole LTX-2.5
// fixture here would make this case depend on the very fixture whose accidental
// coverage it exists to replace.
struct TempDir {
  std::string root;
  TempDir() {
    static int counter = 0;
    root = "/tmp/vllm_ltx2_align_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
  }
  ~TempDir() {
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

// safetensors written by hand, so the header length — and with it the payload's
// parity — is ours to choose. `header_pad` spaces are appended INSIDE the
// counted header; trailing whitespace is legal JSON and padding the header is
// exactly how real writers align their payloads. Returns the absolute file
// offset of the single tensor's first byte.
size_t WriteOneBf16Safetensors(const std::string& path, const std::string& name,
                               const std::vector<float>& values, size_t header_pad) {
  std::string header = "{\"" + name + "\":{\"dtype\":\"BF16\",\"shape\":[" +
                       std::to_string(values.size()) + "],\"data_offsets\":[0," +
                       std::to_string(values.size() * sizeof(uint16_t)) + "]}}";
  header.append(header_pad, ' ');
  std::string payload;
  for (const float v : values) {
    const uint16_t b = ltx2_fixture::F32ToBf16(v);
    payload.append(reinterpret_cast<const char*>(&b), sizeof(b));
  }
  std::string out;
  const uint64_t len = header.size();
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((len >> (8 * i)) & 0xFFU));
  out += header;
  out += payload;
  ltx2_fixture::WriteFileBytes(path, out);
  return 8 + header.size();
}

}  // namespace

TEST_CASE("ltx2 VAE weights load from an ODD safetensors payload offset (#674)") {
  TempDir ws;
  // Values chosen so every one survives a bf16 store EXACTLY, which is what lets
  // the check below be equality rather than a tolerance: a wrong-by-one-byte
  // read is then a hard failure and not something a band could absorb.
  const std::vector<float> values = {1.0F, -2.0F, 0.5F, 384.0F, -0.125F, 3.0F, -48.0F};
  const std::string name = "decoder.conv_in.conv.weight";

  // The unpadded header lands the payload on some parity; one space flips it.
  // Write both and keep whichever is ODD, so this does not depend on the exact
  // length of the JSON above.
  const std::string a = ws.root + "/odd_offset_a.safetensors";
  const std::string b = ws.root + "/odd_offset_b.safetensors";
  const size_t off_a = WriteOneBf16Safetensors(a, name, values, 0);
  const size_t off_b = WriteOneBf16Safetensors(b, name, values, 1);
  REQUIRE((off_a % 2) != (off_b % 2));
  const std::string path = (off_a % 2 == 1) ? a : b;

  const vllm::SafetensorsFile file = vllm::SafetensorsFile::Open(path);
  // The fixture really is what this case claims: the tensor's mapped address is
  // ODD, so the loader below cannot satisfy a `uint16_t`'s alignment by luck.
  // Page-aligned mmap base means file-offset parity IS address parity, but assert
  // the address rather than infer it.
  REQUIRE((reinterpret_cast<uintptr_t>(file.Get(name).data) % 2) == 1);

  const vllm::Ltx2VaeWeights weights =
      vllm::Ltx2LoadVaeWeights(file, vllm::Ltx2VideoVaeDecoderKeyRules());
  // `decoder.` is rewritten away by the video-VAE rules, exactly as upstream's
  // SDOps do.
  REQUIRE(weights.Has("conv_in.conv.weight"));
  const std::vector<float>& got = weights.Get("conv_in.conv.weight");
  REQUIRE(got.size() == values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    INFO("element " << i);
    CHECK(got[i] == doctest::Approx(values[i]).scale(0.0));
  }
}

// ─── the embeddings connector (phase L9c) ───────────────────────────────────
//
// WHAT THESE ARE FOR. Until L9c the conditioning this engine handed the DiT was
// the prompt-embeds file VERBATIM: `Ltx2ConnectorForward` had landed at L5, had
// been gated against upstream on five arms, and was called by NOTHING but its
// own test, while the 129-tensor `*_embeddings_connector` family the shipped DiT
// carries was refused as unported and stepped over. Every case below fails on
// the pre-L9c engine, and the first two fail for the reason that matters:
// deleting the connector call leaves the render byte-identical to a render with
// DIFFERENT connector weights, because nothing read them.

namespace {

// Every frame byte of a render, so two renders can be compared as a whole rather
// than through a statistic that might not move.
//
// Phase 0 only, for the same reason the first render case gives: the recipe's
// second phase needs the latent spatial upsampler and running without one is a
// REFUSAL. Phase 0 already carries the conditioning through cross-attention in
// every one of its blocks, which is what these cases are about.
std::string RenderBytes(vllm::multimodal::VideoModelParams mp, const std::string& out_dir) {
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  const vllm::multimodal::VideoResult result = engine->Generate(FixtureGen(out_dir));
  std::string all;
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    all += ReadAll(out_dir + name);
  }
  all += ReadAll(result.audio_path);
  return all;
}

// The same render, driven by a REQUEST the caller chose. `RenderBytes` fixes the
// request at `FixtureGen`, which is what the connector cases want and what a
// conditioning case cannot use.
std::string RenderBytesWithGen(vllm::multimodal::VideoModelParams mp,
                               const vllm::multimodal::VideoGenParams& gen) {
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  const vllm::multimodal::VideoResult result = engine->Generate(gen);
  std::string all;
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    all += ReadAll(gen.output_dir + name);
  }
  all += ReadAll(result.audio_path);
  return all;
}

std::string RefusalOf(const vllm::multimodal::VideoModelParams& mp) {
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
  } catch (const std::exception& e) {
    return e.what();
  }
  return std::string();
}

}  // namespace

TEST_CASE("ltx2 video: the render READS the checkpoint's connector weights") {
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();

  // A second DiT, byte-identical except that its connector's parameter stream is
  // seeded differently. Same shapes, same config, same DiT weights.
  ltx2_fixture::ReducedDitOptions other;
  other.connector.tag = "b";
  const std::string other_dit = ws.root + "/dit_connector_b.safetensors";
  ltx2_fixture::WriteReducedDit(dit, other_dit, other);

  vllm::multimodal::VideoModelParams a = FixtureParams(ws.paths);
  vllm::multimodal::VideoModelParams b = a;
  b.dit_path = other_dit;

  const std::string frames_a = RenderBytes(a, ws.root + "/conn_a");
  const std::string frames_b = RenderBytes(b, ws.root + "/conn_b");
  CHECK(frames_a.size() == frames_b.size());
  // BEFORE L9c these are EQUAL: the connector weights were loaded by nobody, so
  // the only thing that differed between the two files was never read.
  CHECK(frames_a != frames_b);

  // ...and the same render twice is byte-identical, which is what makes the
  // inequality above a statement about the connector rather than about noise.
  CHECK(RenderBytes(a, ws.root + "/conn_a2") == frames_a);
}

// ─── the keyframe marker, measured at the PIXELS ────────────────────────────
//
// WHY THIS CASE EXISTS, and it is not a duplicate of the DiT-level goldens.
// Every other check in this file is RELATIVE — `digest != digest`, `absmax > 0`,
// "the frames are not one flat value" — and none of them is anchored to what the
// render is SUPPOSED to contain. That is why a defect measured on this head was
// invisible: the engine builds `video.keyframes_mask` unconditionally, and the
// guard beside it asserts the VECTOR is populated, so making the mask
// conditional REDs 11 cases here while making the HANDOVER conditional — one
// line lower, `if (wants_image) vin.keyframes_mask = ...` — compiled clean and
// left all five LTX-2.5 suites GREEN. The pixels moved (frame 0 flat 127 → flat
// 130) and nothing in the tree said so.
//
// The fix in `ltx2_video.cpp` closes that one line. THIS CASE CLOSES THE CLASS,
// because it does not look at any line: it compares a render whose DiT carries
// the parameter against a render whose DiT does not, on a request with NO image
// and NO keyframe — upstream's unconditional case. Any route by which the
// trained term fails to reach the forward collapses the two renders into one and
// REDs here, whether the drop is in the mask, the handover, the binding, or the
// add.
//
// The two checkpoints differ in exactly one thing. `Param()` seeds every tensor
// from its own NAME, so dropping `keyframes_abs_pos_embedding` from the contract
// perturbs no other value; the DiT config's flag follows the shapes because both
// come from the same `Ltx2DitParams`.
TEST_CASE("ltx2 video: the keyframe marker reaches the PIXELS with no image supplied") {
  Workspace ws;
  const vllm::Ltx2DitParams marked = ltx2_fixture::ReducedDitParams();
  // The fixture's own default, asserted rather than assumed: with the flag off
  // this whole case would compare two identical renders and pass vacuously.
  REQUIRE(marked.use_keyframes_abs_pos_embedding);

  vllm::Ltx2DitParams unmarked = marked;
  unmarked.use_keyframes_abs_pos_embedding = false;
  const std::string unmarked_dit = ws.root + "/dit_no_keyframes.safetensors";
  ltx2_fixture::WriteReducedDit(unmarked, unmarked_dit, ltx2_fixture::ReducedDitOptions{});

  vllm::multimodal::VideoModelParams with_marker = FixtureParams(ws.paths);
  vllm::multimodal::VideoModelParams without_marker = with_marker;
  without_marker.dit_path = unmarked_dit;

  // `FixtureGen` supplies no image and no keyframe, which is the whole point:
  // upstream marks the first latent frame "independently of whether any keyframe
  // slots exist" (tools.py:186-196). A port that marked it only when something
  // was conditioned would be silently wrong on every plain text-to-video render,
  // and that is the render this case takes.
  const std::string with = RenderBytes(with_marker, ws.root + "/kf_marked");
  const std::string without = RenderBytes(without_marker, ws.root + "/kf_unmarked");
  REQUIRE(with.size() == without.size());

  size_t differing = 0;
  for (size_t i = 0; i < with.size(); ++i) {
    if (with[i] != without[i]) ++differing;
  }
  MESSAGE("keyframe marker moves " << differing << " of " << with.size()
                                   << " artifact bytes");
  // Strictly greater than zero, and no count floor above it. A count-based
  // tolerance would bound nothing — it would red on unrelated numerical drift and
  // still admit a term applied to the wrong frame — and the frame the marker
  // belongs on is gated by the DiT goldens, which mark one frame and check the
  // others are untouched. What this case owns is the ENGINE-to-PIXEL route, and
  // for that the question is binary: did the trained term reach the render at
  // all.
  CHECK_MESSAGE(differing > 0,
                "the DiT that carries keyframes_abs_pos_embedding rendered the same bytes as the "
                "DiT that does not, so the trained term never reached the forward");

  // ...and the same DiT twice is byte-identical, which is what makes the
  // inequality a statement about the marker rather than about noise.
  CHECK(RenderBytes(with_marker, ws.root + "/kf_marked2") == with);
}

// ─── the token-APPEND seam (row LTX25-TOKEN-APPEND, issue #930) ─────────────
//
// UPSTREAM SHIPS NO TESTS at pin `fd4ded7f` — `find /home/mudler/_git/LTX-2
// -name 'test_*.py'` returns 0 across the whole repository — so nothing is
// ported here. Every assertion cites the upstream `file:line` that justifies it
// instead.
//
// THE WITNESS IS ON RENDERED BYTES, and that is the whole design. `Ltx2ConditioningTrace`
// is filled before the denoise loop for every field except the handful written
// inside it, so a witness built on the trace cannot observe what the loop does —
// a sibling row's first attempt at exactly this found every arm identical for
// that reason.
//
// AND IT CARRIES A NO-OP CONTROL, which is the correction that made the sibling's
// result diagnosable. Their arms came out identical INCLUDING the control, which
// is what said "the instrument is blind" rather than "the feature is weak".
// Without the control those two read the same, and the wrong one is the one that
// ships. So the comparison set below is {no keyframe, keyframe A, keyframe B}:
//
//   * every arm equal, control included  => the instrument is blind;
//   * kf_a != noop                       => the append reached the maths;
//   * kf_a != kf_b                       => the appended CONTENT reached it,
//                                           not merely the token count.
TEST_CASE("ltx2 video: a LAST-frame keyframe is APPENDED, and the sequence is trimmed back") {
  Workspace ws;

  // Deliberately not the render's own resolution: `load_image_and_preprocess`
  // aspect-fills and centre-crops to the phase's height/width
  // (media_io/resize.py:41-73).
  const std::string kf_a_path = ws.root + "/kf_a.ppm";
  const std::string kf_b_path = ws.root + "/kf_b.ppm";
  WriteBytes(kf_a_path, ConditioningPpm(20, 28, 21));
  WriteBytes(kf_b_path, ConditioningPpm(20, 28, 22));

  auto request = [&](const std::string& tag, const std::string& keyframe) {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + tag);
    if (!keyframe.empty()) {
      gen.last_frame_path = keyframe;
      // The codec round trip is unported and an LTX-2.5 checkpoint RESOLVES 18,
      // so the supported arm has to be asked for. Same rule as the first-frame
      // arm, because upstream resolves the CRF once for the whole `images` list
      // (blocks.py:966-983).
      gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    }
    return gen;
  };

  const vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  const std::string noop = RenderBytesWithGen(mp, request("kf_noop", ""));
  const std::string kf_a = RenderBytesWithGen(mp, request("kf_a", kf_a_path));
  const std::string kf_b = RenderBytesWithGen(mp, request("kf_b", kf_b_path));
  REQUIRE(noop.size() == kf_a.size());
  REQUIRE(noop.size() == kf_b.size());

  // THE CONTROL FIRST. A re-render of the no-keyframe request must be byte
  // identical, otherwise every inequality below is noise and this case says
  // nothing about appends.
  REQUIRE_MESSAGE(RenderBytesWithGen(mp, request("kf_noop2", "")) == noop,
                  "the same request rendered twice is not byte-identical, so this instrument "
                  "cannot measure anything");

  auto differing = [](const std::string& a, const std::string& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) ++n;
    }
    return n;
  };
  MESSAGE("kf_a vs noop: " << differing(kf_a, noop) << " of " << noop.size() << " bytes; "
                           << "kf_a vs kf_b: " << differing(kf_a, kf_b));

  // The appended tokens take part in self-attention over the WHOLE sequence, so
  // a keyframe that reached the maths moves the target tokens' own output. This
  // is the claim the refusal that stood here was about: the engine could not
  // grow the sequence through the DiT.
  CHECK_MESSAGE(differing(kf_a, noop) > 0,
                "a last-frame keyframe rendered the same bytes as a render with no keyframe at "
                "all, so the appended tokens never reached the forward");
  // ...and it is the keyframe's CONTENT that reached it. Two keyframes append
  // the same NUMBER of tokens, so a build that grew the sequence with zeros —
  // or that appended the wrong buffer — passes the check above and fails this
  // one.
  CHECK_MESSAGE(differing(kf_a, kf_b) > 0,
                "two DIFFERENT last-frame keyframes rendered identical bytes, so the appended "
                "tokens carry no content from the keyframe");

  SUBCASE("the sequence GROWS through the DiT and comes back to the target grid") {
    vllm::multimodal::VideoModelParams capped = FixtureParams(ws.paths);
    capped.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";

    auto tokens_of = [&](const std::string& tag, const std::string& keyframe) {
      const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
          vllm::multimodal::LoadVideoEngine(capped);
      auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
      REQUIRE(ltx2 != nullptr);
      const vllm::multimodal::VideoResult result = engine->Generate(request(tag, keyframe));
      // The artifact is the other half of the claim: the trim is what lets
      // `Ltx2VideoUnpatchify` produce a target-shaped volume, and the frame
      // count is that shape observed from outside.
      CHECK(result.frame_count == 9);
      return ltx2->last_conditioning().video_tokens;
    };

    const int64_t plain = tokens_of("tok_noop", "");
    const int64_t with_kf = tokens_of("tok_kf", kf_a_path);

    // The fixture's phase 0 runs at `spatial_downscale = 2`, so 64x64 pixels is a
    // 1x1 latent grid and 9 frames is 2 latent frames: 2 target tokens. One
    // encoded keyframe is one latent frame at that grid, so it appends exactly
    // `tokens_per_latent_frame` = 1 (tools.py:198-201).
    CHECK(plain == 2);
    CHECK_MESSAGE(with_kf == plain + 1,
                  "a keyframe must append one latent frame's worth of tokens "
                  "(keyframe_cond.py:79-82); got " << with_kf << " against a target of " << plain);
  }

  SUBCASE("the sigma schedule keeps reading the TARGET count, not the grown one") {
    // The distilled two-stage recipe carries its own frozen sigmas
    // (distilled.py:200-201 defaults both stages to the `DISTILLED_SIGMAS` /
    // `STAGE_2_DISTILLED_SIGMAS` constants of utils/constants.py:17-23), so it
    // never computes a schedule and cannot show
    // this. `one_stage` does: `phase.sigmas` is empty, so the engine calls
    // `Ltx2SigmaSchedule`, whose shift is a function of the token count
    // (schedulers.py:37-39).
    //
    // Upstream fixes that count at the TARGET twice over: the argument is
    // `math.prod(latent.shape[2:])` of the UNPATCHIFIED target, which cannot
    // contain appended tokens, and `ti2vid_one_stage.py:207` computes the
    // schedule before any state exists. A port that read the grown count would
    // re-shift the entire trajectory the moment a keyframe was supplied.
    vllm::multimodal::VideoModelParams one_stage = FixtureParams(ws.paths);
    one_stage.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
    one_stage.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
        FixtureCheckpointClass("one_stage");

    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(one_stage);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);

    vllm::multimodal::VideoGenParams gen = request("one_stage_kf", kf_a_path);
    gen.steps = 2;  // one_stage admits a step override; 50 would gate nothing extra
    OneStageFixtureGuidance(&gen);
    (void)engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();

    // Both numbers are MEASURED, and the statement is the relation between them.
    // Pinning either to a literal would pass on a build that read the grown
    // count everywhere.
    CHECK(trace.schedule_tokens > 0);
    CHECK_MESSAGE(trace.video_tokens > trace.schedule_tokens,
                  "the DiT ran over " << trace.video_tokens << " tokens and the schedule was "
                                      << "built for " << trace.schedule_tokens
                                      << "; equal means the append re-shifted the schedule");
  }
}

TEST_CASE("ltx2 video: the connector's positional bound comes from the CONFIG") {
  // `connector_positional_embedding_max_pos` divides every token index
  // (rope.py:132-141). LTX-2.5 declares [4096]; the class default is [1], which
  // is 4096x. Nothing in a SHAPE can see the difference, so a config that is
  // parsed but not USED renders confidently at the wrong RoPE — the invisible
  // constant class spec section 7.0(a) names.
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();
  ltx2_fixture::ReducedDitOptions defaulted;
  defaulted.transformer_overrides["connector_positional_embedding_max_pos"] =
      std::vector<int64_t>{1};
  const std::string other_dit = ws.root + "/dit_maxpos_1.safetensors";
  ltx2_fixture::WriteReducedDit(dit, other_dit, defaulted);

  vllm::multimodal::VideoModelParams a = FixtureParams(ws.paths);
  vllm::multimodal::VideoModelParams b = a;
  b.dit_path = other_dit;
  CHECK(RenderBytes(a, ws.root + "/maxpos_4096") != RenderBytes(b, ws.root + "/maxpos_1"));
}

TEST_CASE("ltx2 video: a connector config that disagrees with the FILE is refused") {
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();

  SUBCASE("fewer layers than the file carries binds a PREFIX, so it is refused") {
    ltx2_fixture::ReducedDitOptions shrunk;
    shrunk.transformer_overrides["connector_num_layers"] = 1;
    const std::string path = ws.root + "/dit_conn_1layer.safetensors";
    ltx2_fixture::WriteReducedDit(dit, path, shrunk);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = path;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    CHECK(msg.find("embeddings_connector") != std::string::npos);
    CHECK(msg.find("does not name") != std::string::npos);
  }

  SUBCASE("more layers than the file carries names the MISSING tensor") {
    ltx2_fixture::ReducedDitOptions grown;
    grown.transformer_overrides["connector_num_layers"] = 3;
    const std::string path = ws.root + "/dit_conn_3layer.safetensors";
    ltx2_fixture::WriteReducedDit(dit, path, grown);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = path;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    CHECK(msg.find("transformer_1d_blocks.2") != std::string::npos);
  }

  SUBCASE("a register count the file's TABLE does not carry is refused") {
    // `connector_num_learnable_registers` is the ONE key `Ltx2ParseConnectorConfig`
    // reads that NEITHER upstream configurator does: `Embeddings1DConnectorConfigurator`
    // (embeddings_connector.py:194-219) and its audio sibling (:222-256) both leave
    // it at the class default of 128, so "mirrors both configurators key for key" is
    // not literally true of this one key and the divergence is deliberate — a
    // checkpoint declaring something else must not be silently run at 128.
    //
    // A read that nothing can falsify is a claim, not a gate, and this is what
    // falsifies it: the fixture's stored table stays `[2, dim]` while the config
    // declares 4, and `Ltx2LoadConnectorWeights`' shape check is what has to fire.
    ltx2_fixture::ReducedDitOptions relabelled;
    relabelled.transformer_overrides["connector_num_learnable_registers"] = 4;
    const std::string path = ws.root + "/dit_conn_registers.safetensors";
    ltx2_fixture::WriteReducedDit(dit, path, relabelled);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = path;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    CHECK(msg.find("learnable_registers") != std::string::npos);
  }

  SUBCASE("gating the config declares but the file does not carry is refused") {
    ltx2_fixture::ReducedDitOptions ungated;
    ungated.connector.gated = false;  // writes NO to_gate_logits tensors...
    ungated.transformer_overrides["connector_apply_gated_attention"] = true;  // ...but claims them
    const std::string path = ws.root + "/dit_conn_gate.safetensors";
    ltx2_fixture::WriteReducedDit(dit, path, ungated);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.dit_path = path;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    CHECK(msg.find("to_gate_logits") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: half a connector conditions two modalities differently") {
  Workspace ws;
  ltx2_fixture::ReducedDitOptions video_only;
  video_only.connector.audio = false;
  const std::string path = ws.root + "/dit_video_connector_only.safetensors";
  ltx2_fixture::WriteReducedDit(ltx2_fixture::ReducedDitParams(), path, video_only);
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.dit_path = path;
  const std::string msg = RefusalOf(mp);
  INFO(msg);
  CHECK(msg.find("embeddings connector") != std::string::npos);
  CHECK(msg.find("audio") != std::string::npos);
}

TEST_CASE("ltx2 video: prompt rows the register table cannot tile are refused") {
  // `seq_len % num_learnable_registers == 0` is upstream's own assert
  // (embeddings_connector.py:144), because the table is TILED across the
  // sequence rather than indexed by which positions were padded.
  Workspace ws;
  const std::string odd = ws.root + "/three_rows.f32";
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();
  ltx2_fixture::WritePromptEmbeds(odd, "ltx2.embeds.video.odd", 3, dit.cross_attention_dim);
  const std::string odd_audio = ws.root + "/three_rows_audio.f32";
  ltx2_fixture::WritePromptEmbeds(odd_audio, "ltx2.embeds.audio.odd", 3,
                                  dit.audio_cross_attention_dim);
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.prompt_embeds_path = odd;
  mp.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = odd_audio;
  const std::string msg = RefusalOf(mp);
  INFO(msg);
  CHECK(msg.find("learnable registers") != std::string::npos);
}

TEST_CASE("ltx2 video: the valid-row count decides which positions are registers") {
  // With no text tower there is no tokenizer mask, so `prompt_embeds_valid_rows`
  // is what says which supplied rows are caption and which are padding — and
  // padding is not inert here, it is REPLACED by the learnable register table.
  // A render that ignored the extra would be identical whatever it said.
  Workspace ws;
  vllm::multimodal::VideoModelParams all_valid = FixtureParams(ws.paths);
  vllm::multimodal::VideoModelParams half = all_valid;
  half.extras[vllm::multimodal::kLtx2PromptValidRowsExtra] = "2";
  CHECK(RenderBytes(all_valid, ws.root + "/rows_4") != RenderBytes(half, ws.root + "/rows_2"));

  SUBCASE("a count past the end of the file is refused") {
    vllm::multimodal::VideoModelParams over = all_valid;
    over.extras[vllm::multimodal::kLtx2PromptValidRowsExtra] = "99";
    const std::string msg = RefusalOf(over);
    INFO(msg);
    CHECK(msg.find(vllm::multimodal::kLtx2PromptValidRowsExtra) != std::string::npos);
  }
}

TEST_CASE("ltx2 video: the register boundary sits EXACTLY at the valid-row count") {
  // WHY THE CASE ABOVE IS NOT ENOUGH, and this one exists. It asserts only that
  // a render at valid=4 differs from one at valid=2. ANY monotone corruption of
  // the boundary keeps that true — `prompt_valid_rows + 1` still renders
  // differently from valid=2 — so the whole family passes it. The defect it
  // cannot see is ONE padded row conditioned on caller junk instead of the
  // connector's TRAINED register: finite, correctly shaped, plausible, wrong.
  //
  // WHAT MAKES THE BOUNDARY NUMERICALLY REACHABLE is the substitution itself.
  // `binary_mask = additive_attention_mask[:, 0, 0, :] >= 0` then
  // `binary_mask * hidden_states + (1 - binary_mask) * registers`
  // (embeddings_connector.py:148-150) REPLACES a masked position's features
  // outright, so what the caller supplied there cannot reach the render at all.
  // Positions therefore separate by whether perturbing them moves a byte:
  //
  //   rows [valid, N) are REGISTERS -> perturbing them must change NOTHING
  //   rows [0, valid)  are CAPTION  -> perturbing the last one must change something
  //
  // Together those pin the boundary from both sides, which is what "which
  // positions are registers" means when the conditioning itself is not
  // observable from outside the engine. `+ 1` breaks the first (row `valid`
  // keeps caller junk); `- 1` breaks the second (row `valid - 1` becomes a
  // register and its content stops mattering). The reviewer's mutation was the
  // former and the suite stayed fully green.
  //
  // THE POLARITY AND THE CONTIGUITY ARE BOTH UPSTREAM'S.
  // `_prepare_attention_mask` builds `(attention_mask - 1) * finfo.max`
  // (transformer_args.py:203-206), so 0.0 is kept and -finfo.max is padded; and
  // `_compute_right_pad_order`'s stable descending argsort
  // (embeddings_processor.py:33-38) makes the padded set the CONTIGUOUS SUFFIX
  // `[valid, N)` that `prompt_embeds_valid_rows` names. Neither the mask
  // comparison nor the returned mask can carry this: with registers on, :152
  // returns `torch.zeros_like(mask)` and `_to_binary_mask`'s `< 0.000001`
  // (:46-48) is satisfied by BOTH values an additive mask holds, so the mask the
  // DiT receives is all ones either way. The substitution is the only carrier.
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();
  const int64_t width = dit.cross_attention_dim;
  const int64_t rows = 4;   // `ltx2_fixture::WriteFixture`'s `prompt_tokens`
  const int64_t valid = 2;  // two caption rows, two padded ones

  const std::vector<float> base = ReadFloats(ws.paths.video_embeds);
  REQUIRE(base.size() == static_cast<size_t>(rows * width));

  // A perturbation big enough that no 8-bit frame byte can absorb it, and
  // column-varying so it cannot turn one row into a copy of another.
  auto perturb_rows = [&](const std::string& name, int64_t first, int64_t last) {
    std::vector<float> values = base;
    for (int64_t r = first; r < last; ++r) {
      for (int64_t c = 0; c < width; ++c) {
        values[static_cast<size_t>(r * width + c)] += 3.0f + 0.25f * static_cast<float>(c);
      }
    }
    const std::string path = ws.root + "/" + name + ".f32";
    WriteFloats(path, values);
    return path;
  };

  vllm::multimodal::VideoModelParams half = FixtureParams(ws.paths);
  half.extras[vllm::multimodal::kLtx2PromptValidRowsExtra] = std::to_string(valid);
  const std::string reference = RenderBytes(half, ws.root + "/reg_ref");
  // Without this an `==` below could pass on two empty reads rather than on two
  // renders.
  REQUIRE(reference.size() > 1000);
  // ...and the same request twice is byte-identical, which is what makes the
  // equality an assertion about the registers rather than about noise.
  REQUIRE(RenderBytes(half, ws.root + "/reg_ref2") == reference);

  SUBCASE("every row at or past the count is a register, so its content is inert") {
    vllm::multimodal::VideoModelParams padded = half;
    padded.prompt_embeds_path = perturb_rows("perturbed_padding", valid, rows);
    // REDs at `prompt_valid_rows + 1`: row `valid` would keep the caller's junk
    // instead of becoming the trained register.
    CHECK(RenderBytes(padded, ws.root + "/reg_padded") == reference);
  }

  SUBCASE("the row just BEFORE the count is caption, so its content is not") {
    vllm::multimodal::VideoModelParams caption = half;
    caption.prompt_embeds_path = perturb_rows("perturbed_last_caption", valid - 1, valid);
    // REDs at `prompt_valid_rows - 1`: row `valid - 1` would be replaced by a
    // register and the perturbation would stop reaching the DiT.
    CHECK(RenderBytes(caption, ws.root + "/reg_caption") != reference);
  }

  SUBCASE("the FIRST row is caption too, so the boundary is not the whole prompt") {
    // Guards the degenerate reading where nothing is caption: a mask of all
    // -finfo.max renders the register table and only the register table.
    vllm::multimodal::VideoModelParams first = half;
    first.prompt_embeds_path = perturb_rows("perturbed_first", 0, 1);
    CHECK(RenderBytes(first, ws.root + "/reg_first") != reference);
  }
}

// ─── the prompt hop (phase L13) ─────────────────────────────────────────────
//
// WHAT THESE GATE, AND WHY THE INSTRUMENT IS WHAT IT IS.
//
// The claim is narrow and it is the whole point of the phase: a TYPED PROMPT
// conditions the render. That is a DEPENDENCE claim — did these bytes come from
// that string, through those weights — and not a quality claim, which nothing
// here could support and nothing here asserts.
//
// The instrument is `Ltx2VideoEngine::last_conditioning()`: an FNV-1a digest and
// an absmax over the exact f32 buffers `Ltx2ModalityInput::context` pointed at,
// taken after the connector and immediately before the denoise loop. It is
// chosen over a frame statistic deliberately, and the campaign already paid for
// the lesson: L9c's reviewer found that a scene and a colour field were
// INDISTINGUISHABLE to the existing frame analyzer (neighbour |dx|/sd 0.093 vs
// 0.033, block-mean ratios nearly identical) and needed contact sheets to be
// told apart. A digest cannot have that blind spot, because it is a function of
// the bytes themselves rather than a summary of them: any change to any element
// changes it, and no plausible-looking wrong tensor can satisfy it by accident.
//
// The absmax is the companion the digest needs. Two prompts whose conditioning
// both collapsed to zeros would have EQUAL digests and RED the dependence check
// — but for the wrong reason, and `video_absmax > 0` is what separates "the
// tower ran" from "the tower returned nothing".
//
// The frame-byte comparison is then the second half: the digest says the
// conditioning depends on the prompt, and the frames say the conditioning
// reaches the render.

namespace {

// The fixture pointed at its own text tower. `max_phase = 0` for the same reason
// `RenderBytes` gives: phase 1 needs the latent upsampler and running without one
// is a refusal, while phase 0 already carries the conditioning through
// cross-attention in every block.
vllm::multimodal::VideoModelParams EncoderParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.video_vae_path = paths.video_vae;
  mp.audio_vae_path = paths.audio_vae;
  mp.encoder_path = paths.encoder;
  mp.extras[vllm::multimodal::kLtx2EncoderConfigPathExtra] = paths.encoder_config;
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  // This helper builds its params from scratch rather than from `FixtureParams`,
  // so it declares the checkpoint class itself (row LTX25-CHECKPOINT-CLASS,
  // #1137). No kind is set, so the recipe is `distilled_two_stage`.
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("distilled_two_stage");
  mp.device = 0;
  return mp;
}

vllm::multimodal::VideoGenParams PromptedGen(const std::string& out_dir,
                                             const std::string& prompt) {
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.prompt = prompt;
  return gen;
}

// One prompted render, returning both halves of the evidence: what the DiT was
// conditioned on, and what came out.
struct Rendered {
  vllm::multimodal::Ltx2ConditioningTrace trace;
  std::string bytes;
};

Rendered RenderPrompt(const vllm::multimodal::VideoModelParams& mp, const std::string& out_dir,
                      const std::string& prompt) {
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  const vllm::multimodal::VideoResult result = engine->Generate(PromptedGen(out_dir, prompt));
  Rendered out;
  // The trace is read off the ENGINE, not re-derived: a test that recomputed the
  // conditioning from the same inputs would prove its own arithmetic and say
  // nothing about what the engine bound.
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  out.trace = ltx->last_conditioning();
  // `completed` separates "this conditioning produced that clip" from "this
  // conditioning was built for a render that then threw". The trace is filled
  // BEFORE the denoise loop — the only point at which the buffers
  // cross-attention reads still exist as such — so everything from there to the
  // muxer can fail with the trace already populated. `Generate` returned here,
  // so it must be true; the refusal cases below cover the other polarity.
  CHECK(out.trace.completed);
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    out.bytes += ReadAll(out_dir + name);
  }
  out.bytes += ReadAll(result.audio_path);
  return out;
}

}  // namespace

TEST_CASE("ltx2 video: a typed PROMPT conditions the render") {
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();

  {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    REQUIRE(engine != nullptr);
    // The seam's own flag. Before L13 this was false BY DECLARATION even though
    // the tower ran, because its output could not reach the DiT.
    CHECK(engine->has_encoder());
    // ...and no prompt-embeds file was supplied, so the conditioning below
    // cannot be coming from one.
    CHECK_FALSE(engine->has_prompt_embeds());
  }

  const Rendered fox = RenderPrompt(mp, ws.root + "/p_fox", "a b c");
  const Rendered whale = RenderPrompt(mp, ws.root + "/p_whale", "c a b a");

  // 1. THE TOWER RAN, on the request's own string.
  CHECK(fox.trace.from_prompt);
  CHECK(fox.trace.prompt == "a b c");
  // 2. THE GEOMETRY IS UPSTREAM'S. Every prompt is padded to
  //    TOKENIZER_MAX_LENGTH (gemma_assets.py:162) and the DiT cross-attends over
  //    the full padded width, at the two stream widths its config declares.
  CHECK(fox.trace.tokens == vllm::kLtx2GemmaTokenizerMaxLength);
  CHECK(fox.trace.video_width == dit.cross_attention_dim);
  CHECK(fox.trace.audio_width == dit.audio_cross_attention_dim);
  // 3. IT IS NOT ZEROS. Without this the dependence check below would pass for a
  //    tower that returned nothing at all.
  CHECK(fox.trace.video_absmax > 1e-6);
  CHECK(fox.trace.audio_absmax > 1e-6);

  // 4. THE CONDITIONING DEPENDS ON THE PROMPT — both streams, because LTX-2.5
  //    conditions two and one of them silently constant is a soundtrack that
  //    ignores the caption.
  CHECK(whale.trace.tokens == fox.trace.tokens);
  CHECK(whale.trace.video_digest != fox.trace.video_digest);
  CHECK(whale.trace.audio_digest != fox.trace.audio_digest);

  // 5. AND IT REACHES THE RENDER. The digests could differ while the frames did
  //    not, if the conditioning were computed and then dropped — which is
  //    precisely the shape of every bug this campaign has hit on this path.
  CHECK(fox.bytes.size() == whale.bytes.size());
  CHECK(fox.bytes != whale.bytes);

  // 6. AND IT IS DETERMINISTIC, which is what makes 4 and 5 statements about the
  //    prompt rather than about noise.
  const Rendered again = RenderPrompt(mp, ws.root + "/p_fox2", "a b c");
  CHECK(again.trace.video_digest == fox.trace.video_digest);
  CHECK(again.trace.audio_digest == fox.trace.audio_digest);
  CHECK(again.bytes == fox.bytes);
}

TEST_CASE("ltx2 video: the prompt's conditioning goes through the CONNECTOR") {
  // The prompt path is a DIFFERENT caller of `RunConnector` from the
  // prompt-embeds path, so "the connector is wired" proved for one proves
  // nothing about the other. A second DiT, byte-identical except that its
  // connector's parameter stream is seeded differently, is what separates
  // "the projections went straight to cross-attention" from "they went through
  // the checkpoint's own Embeddings1DConnector".
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();
  ltx2_fixture::ReducedDitOptions other;
  other.connector.tag = "b";
  const std::string other_dit = ws.root + "/dit_connector_b.safetensors";
  ltx2_fixture::WriteReducedDit(dit, other_dit, other);

  const vllm::multimodal::VideoModelParams a = EncoderParams(ws.paths);
  vllm::multimodal::VideoModelParams b = a;
  b.dit_path = other_dit;

  const Rendered ra = RenderPrompt(a, ws.root + "/pc_a", "a b c");
  const Rendered rb = RenderPrompt(b, ws.root + "/pc_b", "a b c");
  // Same prompt, same tower, same DiT weights — only the connector differs. If
  // the prompt path skipped the connector these would be EQUAL.
  CHECK(ra.trace.video_digest != rb.trace.video_digest);
  CHECK(ra.trace.audio_digest != rb.trace.audio_digest);
  CHECK(ra.bytes != rb.bytes);
}

TEST_CASE("ltx2 video: the prompt's conditioning READS the text TOWER's weights") {
  // The twin of the case above, one level up: a second text encoder whose tower
  // and projections come from a different parameter stream, at identical shapes
  // and with an identical tokenizer. A conditioning that ignored the tower —
  // returning, say, the projection of a constant — would be equal here.
  Workspace ws;
  const vllm::Ltx2DitParams dit = ltx2_fixture::ReducedDitParams();
  ltx2_fixture::ReducedTextEncoderOptions other;
  other.tag = "b";
  const std::string other_encoder = ws.root + "/text_encoder_b.safetensors";
  ltx2_fixture::WriteReducedTextEncoder(dit, other_encoder, other);

  const vllm::multimodal::VideoModelParams a = EncoderParams(ws.paths);
  vllm::multimodal::VideoModelParams b = a;
  b.encoder_path = other_encoder;

  const Rendered ra = RenderPrompt(a, ws.root + "/pt_a", "a b c");
  const Rendered rb = RenderPrompt(b, ws.root + "/pt_b", "a b c");
  CHECK(ra.trace.tokens == rb.trace.tokens);
  CHECK(ra.trace.video_digest != rb.trace.video_digest);
  CHECK(ra.trace.audio_digest != rb.trace.audio_digest);
  CHECK(ra.bytes != rb.bytes);
}

TEST_CASE("ltx2 video: the Gemma config is never defaulted") {
  Workspace ws;

  SUBCASE("an encoder that declares none, with no extra, is refused BY NAME") {
    vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
    mp.extras.erase(vllm::multimodal::kLtx2EncoderConfigPathExtra);
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    REQUIRE(!msg.empty());
    // MEASURED WHILE MUTATING THIS CASE, and worth stating because it is the
    // failure mode this whole campaign keeps hitting: an earlier version of this
    // SUBCASE asserted only that the message named `encoder_config_path`, and
    // deleting the refusal left it GREEN — the load still failed, but with
    // "encoder_config_path: cannot open " from a fallthrough that tried to read
    // the empty path. A refusal that fires for the wrong reason is exactly what
    // this phase exists to stop shipping, and an assertion that cannot tell the
    // two apart is not a gate. So the two clauses below are the DISCRIMINATING
    // ones: the fallthrough message contains neither.
    CHECK(msg.find("declares no") != std::string::npos);
    CHECK(msg.find("layer_types") != std::string::npos);
  }

  SUBCASE("an encoder that DOES declare one, plus the extra, is refused too") {
    // Never resolved in either direction — the two could describe different
    // towers and there is no way to tell which the tensors belong to.
    ltx2_fixture::ReducedTextEncoderOptions declaring;
    declaring.declare_gemma_config = true;
    const std::string path = ws.root + "/text_encoder_declaring.safetensors";
    ltx2_fixture::WriteReducedTextEncoder(ltx2_fixture::ReducedDitParams(), path, declaring);
    vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
    mp.encoder_path = path;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    REQUIRE(!msg.empty());
    CHECK(msg.find("gemma_config") != std::string::npos);
  }

  SUBCASE("an encoder that declares one needs no extra") {
    ltx2_fixture::ReducedTextEncoderOptions declaring;
    declaring.declare_gemma_config = true;
    const std::string path = ws.root + "/text_encoder_declaring2.safetensors";
    ltx2_fixture::WriteReducedTextEncoder(ltx2_fixture::ReducedDitParams(), path, declaring);
    vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
    mp.encoder_path = path;
    mp.extras.erase(vllm::multimodal::kLtx2EncoderConfigPathExtra);
    CHECK(RefusalOf(mp).empty());
  }

  SUBCASE("the extra with no encoder_path at all is refused, not ignored") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2EncoderConfigPathExtra] = ws.paths.encoder_config;
    const std::string msg = RefusalOf(mp);
    INFO(msg);
    REQUIRE(!msg.empty());
    CHECK(msg.find("no encoder_path") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: a tower with no prompt and no embeds is refused") {
  // The seam admits both conditioning sources, so a request that supplies
  // NEITHER must say so rather than render an unconditioned clip — which is
  // exactly what "the tower is not working" looks like from the outside.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(EncoderParams(ws.paths));
  REQUIRE(engine != nullptr);
  try {
    (void)engine->Generate(FixtureGen(ws.root + "/no_prompt"));
    FAIL("a request with no prompt and no embeds must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("no prompt") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: a trace for a render that never completed says so") {
  // THE DEFECT. `Ltx2ConditioningTrace` is filled immediately after the
  // connector and BEFORE the denoise loop, because that is the only point at
  // which the exact f32 buffers cross-attention will read still exist as such.
  // Everything after it — denoise, both VAE decodes, the WAV, the muxer — can
  // throw with the trace already fully populated. A reader that then asked
  // "which conditioning produced this clip" would be answered for a clip that
  // was never produced, and every field would look entirely healthy: real
  // prompt, non-zero absmax, plausible digests.
  //
  // THE PROBE IS A REAL REFUSAL, not an injected one. Reference conditioning is
  // refused by name (ltx2_video.cpp, the `ImageConditioner` note) and that
  // refusal sits AFTER the trace is written, so a prompted request carrying a
  // reference image walks the whole encode path, fills the trace, and then
  // fails — exactly the shape this flag exists to report.
  //
  // IT IS STILL A REFUSAL AFTER ROW LTX25-IMAGE-COND (#644), which served the
  // first-frame arm and would have made a `first_frame_ppm` probe stop
  // refusing. The reference arm stays refused for a reason this row did not
  // touch (the IC-LoRA scale factors), so the probe was moved to it rather than
  // to whatever happened to still throw.
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);

  vllm::multimodal::VideoGenParams gen = PromptedGen(ws.root + "/thrown", "a b c");
  gen.ref_image_paths.push_back(ws.root + "/nonexistent-reference.png");
  try {
    (void)engine->Generate(gen);
    FAIL("reference conditioning must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("reference-image / reference-video conditioning") != std::string::npos);
  }

  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx->last_conditioning();
  // THE CONTROL, and the reason this case is not vacuous. The encode really did
  // run to completion before the refusal: the prompt was tokenized, the tower
  // and both projections ran, the connector ran, and the trace holds the result.
  // Without these three the case would also pass on an engine that threw BEFORE
  // ever writing a trace, which is a different bug with the same symptom.
  CHECK(trace.from_prompt);
  CHECK(trace.prompt == "a b c");
  CHECK(trace.video_absmax > 0.0);
  // ...and yet no render came out of it. This is the whole assertion.
  CHECK_FALSE(trace.completed);
}

// ─────────────────────────────────────────────────────────────────────────────
// IC-LoRA reachability (row LTX25-IC-LORA, issue #923)
// ─────────────────────────────────────────────────────────────────────────────
//
// THE QUESTION THIS ANSWERS is not "does the fuser work" — test_ltx2_lora
// answers that by calling it. It is "does anything a USER can do reach it",
// which a unit test constructing an `Ltx2LoraAdapter` by hand cannot establish
// (.agents/reachability.md).
//
// So this enters through the production entry point, `LoadVideoEngine`, with a
// `lora_path` LOAD EXTRA — the same path `vllm_video_engine_load` and
// `ltx2-gen --lora` take — and asserts the RENDER moves. The reachability
// mutation is deleting the `dit_options.loras.push_back` call site in
// `ltx2_video.cpp`; that leaves the fuser and its whole unit suite green and
// REDs the cases below, which is the difference between measuring a class and
// measuring a capability.
namespace {

// Write an IC-LoRA adapter targeting one REAL tensor of the reduced DiT
// contract, with its shape derived from the contract rather than hard-coded, so
// a fixture geometry change cannot leave this silently targeting nothing.
std::string WriteFixtureLora(const std::string& path, const std::string& target,
                             float scale,
                             const std::map<std::string, std::string>& metadata = {}) {
  const vllm::Ltx2DitParams params = ltx2_fixture::ReducedDitParams();
  std::vector<int64_t> shape;
  for (const vllm::Ltx2TensorSpec& spec : vllm::EnumerateLtx2DitTensors(params)) {
    if (spec.name == target) shape = spec.shape;
  }
  REQUIRE_MESSAGE(shape.size() == 2,
                  "the fixture LoRA target '", target,
                  "' is not a rank-2 tensor of the reduced DiT contract");
  const int64_t out_features = shape[0];
  const int64_t in_features = shape[1];
  const int64_t rank = 2;

  // B [out, rank] and A [rank, in], both constant, so the delta is a uniform
  // `scale * rank` on every element — large enough that the render cannot be
  // numerically indistinguishable from the unfused one.
  std::vector<ltx2_fixture::Entry> entries = {
      {"diffusion_model." + target.substr(0, target.size() - std::string(".weight").size()) +
           ".lora_A.weight",
       "BF16",
       {rank, in_features},
       std::vector<float>(static_cast<size_t>(rank * in_features), 1.0F),
       {}},
      {"diffusion_model." + target.substr(0, target.size() - std::string(".weight").size()) +
           ".lora_B.weight",
       "BF16",
       {out_features, rank},
       std::vector<float>(static_cast<size_t>(out_features * rank), scale),
       {}},
  };
  std::string metadata_json;
  if (!metadata.empty()) {
    metadata_json = "{";
    bool first = true;
    for (const auto& kv : metadata) {
      if (!first) metadata_json += ",";
      first = false;
      metadata_json += "\"" + kv.first + "\":\"" + kv.second + "\"";
    }
    metadata_json += "}";
  }
  ltx2_fixture::WriteSafetensors(entries, metadata_json, path);
  return path;
}

// The target every case uses: the first block's query projection, which every
// render must read.
const char* const kFixtureLoraTarget = "transformer_blocks.0.attn1.to_q.weight";

}  // namespace

TEST_CASE("ltx2 video: an IC-LoRA supplied through the LOAD EXTRA reaches the PIXELS") {
  // THE WITNESS IS THE RENDERED ARTIFACT, not `last_conditioning()`. The
  // conditioning trace is filled BEFORE the denoise loop runs, so it is a
  // function of the prompt and the conditioning items and cannot see a fused
  // weight at all — a first version of this case compared `video_digest` and
  // found every arm identical, which reads exactly like "the LoRA does nothing"
  // and was in fact "the instrument cannot see it". `RenderBytes` takes the
  // decoded output, which is downstream of the DiT weights.
  Workspace ws;

  const std::string lora =
      WriteFixtureLora(ws.root + "/ic.safetensors", kFixtureLoraTarget, 1.0F);
  vllm::multimodal::VideoModelParams fused = ConditioningParams(ws.paths);
  fused.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;

  const std::string plain = RenderBytes(ConditioningParams(ws.paths), ws.root + "/plain");
  const std::string with_lora = RenderBytes(fused, ws.root + "/fused");
  REQUIRE(plain.size() == with_lora.size());
  REQUIRE(plain.size() > 0);

  size_t differing = 0;
  for (size_t i = 0; i < plain.size(); ++i) {
    if (plain[i] != with_lora[i]) ++differing;
  }
  MESSAGE("the IC-LoRA moves " << differing << " of " << plain.size() << " artifact bytes");
  // THE REACHABILITY CLAIM. Every byte of the REQUEST is identical; the only
  // difference is the `lora_path` LOAD EXTRA. Deleting the
  // `dit_options.loras.push_back` call site in ltx2_video.cpp leaves the whole
  // of test_ltx2_lora green and REDs this, which is the difference between
  // measuring a class and measuring a capability (.agents/reachability.md).
  //
  // Strictly greater than zero and no count floor above it: a count-based
  // tolerance would bound nothing.
  CHECK(differing > 0);
}

TEST_CASE("ltx2 video: the IC-LoRA strength reaches the PIXELS, and 0 is a no-op") {
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/ic.safetensors", kFixtureLoraTarget, 1.0F);

  const auto render = [&](const char* strength, const char* out) {
    vllm::multimodal::VideoModelParams mp = ConditioningParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
    if (strength != nullptr) {
      mp.extras[vllm::multimodal::kLtx2LoraStrengthExtra] = strength;
    }
    return RenderBytes(mp, std::string(ws.root) + "/" + out);
  };

  // A no-adapter control, so "strength 0 renders the base model" is asserted
  // against the base model rather than against itself.
  const std::string baseline = RenderBytes(ConditioningParams(ws.paths), ws.root + "/plain");
  const std::string full = render(nullptr, "full");
  const std::string half = render("0.5", "half");
  const std::string zero = render("0.0", "zero");

  // Strength 0 fuses a zero delta, so the weights are the base model's again.
  // This is what proves the strength is READ rather than accepted and dropped:
  // an implementation that ignored it would give `zero == full != baseline`.
  CHECK(zero == baseline);
  CHECK(full != baseline);
  CHECK(half != full);
  CHECK(half != baseline);
}

TEST_CASE("ltx2 video: the IC-LoRA load extras refuse by name on misuse") {
  Workspace ws;

  SUBCASE("a strength with no adapter refuses rather than doing nothing") {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2LoraStrengthExtra] = "0.5";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a strength with no adapter must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("lora_strength") != std::string::npos);
      CHECK(msg.find("lora_path") != std::string::npos);
    }
  }

  SUBCASE("a non-numeric strength refuses BY NAME") {
    const std::string lora = WriteFixtureLora(ws.root + "/ic.safetensors", kFixtureLoraTarget,
                                              1.0F);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
    mp.extras[vllm::multimodal::kLtx2LoraStrengthExtra] = "strong";
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("a non-numeric strength must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("lora_strength") != std::string::npos);
      CHECK(msg.find("not a finite number") != std::string::npos);
    }
  }

  SUBCASE("an adapter naming a module the DiT does not bind refuses BY NAME") {
    // The divergence from upstream's silent skip (fuse_loras.py:135-137),
    // observed through the PRODUCTION load rather than through the reader.
    const std::string path = ws.root + "/bad.safetensors";
    ltx2_fixture::WriteSafetensors(
        {
            {"diffusion_model.transformer_blocks.0.not_a_module.lora_A.weight",
             "BF16",
             {2, 4},
             std::vector<float>(8, 1.0F),
             {}},
            {"diffusion_model.transformer_blocks.0.not_a_module.lora_B.weight",
             "BF16",
             {4, 2},
             std::vector<float>(8, 1.0F),
             {}},
        },
        std::string(), path);
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = path;
    try {
      (void)vllm::multimodal::LoadVideoEngine(mp);
      FAIL("an adapter targeting an unbound module must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("transformer_blocks.0.not_a_module.weight") != std::string::npos);
      CHECK(msg.find("does not bind") != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 video: the IC-LoRA reference factors are read from the adapter's metadata") {
  // The two numbers the reference refusal used to name as unreadable. Read
  // through the PRODUCTION load, and reported back out through the refusal
  // itself, which is the only user-visible surface that carries them today.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/ic.safetensors", kFixtureLoraTarget, 1.0F,
                       {{"reference_downscale_factor", "2"},
                        {"reference_temporal_scale_factor", "4"}});
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);

  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/ref");
  gen.ref_video_dir = ws.root;
  try {
    (void)engine->Generate(gen);
    FAIL("the reference-video arm is still refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    // The factors the adapter declared, echoed back — so this asserts the READ
    // happened, not merely that a refusal fired.
    CHECK(msg.find("downscale=2") != std::string::npos);
    CHECK(msg.find("temporal=4") != std::string::npos);
    CHECK(msg.find("fused into") != std::string::npos);
    // And it does NOT reintroduce the reason this row closed. The cause that
    // genuinely remains is gated by "a reference video may not blame a seam THIS
    // ENGINE demonstrably has", which measures the engine before it reads the
    // message; asserting the remaining cause by NAME here as well would be the
    // second copy of the mistake that case exists to correct.
    CHECK(msg.find("which this project does not read") == std::string::npos);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// AUDIO-TO-VIDEO — row LTX25-A2V-AUDIO-INPUT, issue #922.
//
// Upstream: `A2VidPipelineTwoStage` (Lightricks/LTX-2 @ fd4ded7f,
// ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py:53, called at :143).
//
// These cases enter through the PRODUCTION path — `LoadVideoEngine` +
// `VideoEngine::Generate`, which is what `vllm_video_generate` calls straight
// through (`vllm_c.cpp:1646`) — and not by constructing `Ltx2DecodeAudioWav` or
// `Ltx2AudioEncoderForward` by hand. Per `.agents/reachability.md`, a unit test
// over those two proves the functions work and never that a request can arrive
// at them; `Ltx2AudioEncoderForward` has had exactly that shape since
// `cefacd2d0`, with six test call sites and no production one.
// ───────────────────────────────────────────────────────────────────────────

namespace {

// The rate the FIXTURE's audio VAE declares on `audio_vae.model.params`, which
// is deliberately NOT the shipped 16000 and not the parser's default either —
// see the long note beside it in `ltx2_video_fixture.h`. Named here so a reader
// who changes one changes the other, and so no `24000` in this file reads as a
// magic number.
constexpr int64_t kFixtureAudioRate = 24000;

// A canonical 16-bit PCM RIFF/WAVE buffer, CHANNEL-INTERLEAVED as the format
// requires. Deterministic and NOT silent: an all-zero take encodes to a latent
// dominated by the mel log clamp (log(1e-5), `ltx2_audio_vae_encoder.h:142`) and
// would satisfy a digest check while proving nothing about whether the samples
// were read at all.
std::string MakeWavPcm16(int64_t channels, int64_t sample_rate, double seconds) {
  const int64_t frames = static_cast<int64_t>(sample_rate * seconds);
  const int64_t data_bytes = frames * channels * 2;
  std::string out;
  auto le = [&](uint32_t value, int n) {
    for (int i = 0; i < n; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  };
  out += "RIFF";
  le(static_cast<uint32_t>(36 + data_bytes), 4);
  out += "WAVE";
  out += "fmt ";
  le(16, 4);
  le(1, 2);  // WAVE_FORMAT_PCM
  le(static_cast<uint32_t>(channels), 2);
  le(static_cast<uint32_t>(sample_rate), 4);
  le(static_cast<uint32_t>(sample_rate * channels * 2), 4);  // byte rate
  le(static_cast<uint32_t>(channels * 2), 2);                // block align
  le(16, 2);                                                 // bits per sample
  out += "data";
  le(static_cast<uint32_t>(data_bytes), 4);
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t c = 0; c < channels; ++c) {
      // A tone plus a per-channel offset, so the two channels DIFFER and a build
      // that read one of them twice cannot match a build that read both.
      const double t = static_cast<double>(f) / static_cast<double>(sample_rate);
      const double v = 0.4 * std::sin(6.2831853 * 220.0 * t) + 0.1 * static_cast<double>(c);
      const auto s = static_cast<int16_t>(std::lround(v * 32767.0));
      le(static_cast<uint32_t>(static_cast<uint16_t>(s)), 2);
    }
  }
  return out;
}

std::string WriteWav(const std::string& path, int64_t channels, int64_t sample_rate,
                     double seconds) {
  const std::string bytes = MakeWavPcm16(channels, sample_rate, seconds);
  std::ofstream out(path, std::ios::binary);
  REQUIRE(out.good());
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  out.close();
  return path;
}

}  // namespace

TEST_CASE("ltx2 video: a supplied audio file CONDITIONS the render, and stays FROZEN") {
  // The claim is in three parts, and only the third is hard to fake:
  //   (1) the request is accepted and renders;
  //   (2) the audio stream carries the ENCODED FILE rather than zeros;
  //   (3) it is FROZEN — upstream's `ModalitySpec(frozen=True, noise_scale=0.0)`
  //       at a2vid_two_stage.py:251-256 and :291-296, identical on both stages.
  //
  // A build that encoded the file and then let the sampler denoise it satisfies
  // (1) and (2) completely, and produces a soundtrack drifting away from the
  // caller's own take that no frame count can see.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);

  const std::string wav = WriteWav(ws.root + "/drive.wav", 2, kFixtureAudioRate, 2.0);
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/a2v");
  gen.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
  const vllm::multimodal::VideoResult result = engine->Generate(gen);

  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
  CHECK(trace.completed);
  CHECK(trace.audio_conditioned);
  CHECK(trace.audio_frozen);
  CHECK(trace.audio_tokens > 0);
  // A latent that collapsed to zeros would give every take the same digest and
  // still satisfy both flags, so the MAGNITUDE is asked for separately — the
  // same reason `image_absmax` sits beside `image_digest`. This is the lower
  // bound the spec's §5 requires: a silently zeroed or constant tensor fails it.
  CHECK(trace.audio_latent_absmax > 0.0);
  CHECK(trace.audio_latent_digest != 0);
  // The SECOND half of `frozen`, and a separate DiT input from the denoise mask
  // (utils/types.py:104-106). Asserted on its own because the mask cannot reach
  // it: a build that zeroed the mask and still handed the forward the schedule's
  // sigma tells the model the caller's clean take is noisy, and every other
  // assertion in this case passes.
  CHECK(trace.audio_sigma_max == 0.0);
  // The render still produced its artifacts; conditioning is not a bypass.
  CHECK(result.frame_count == 9);

  // THE SOUNDTRACK IS THE CALLER'S OWN FILE, not a VAE round trip of it.
  // Upstream states the reason outright — "Return the original input audio
  // instead of VAE-decoded audio to preserve fidelity"
  // (a2vid_two_stage.py:301-303) — and the observable consequence is the SAMPLE
  // RATE: the vocoder's BWE arm emits 48 kHz and the take went in at the audio
  // VAE's own rate. A build that ran the decode and the vocoder anyway would
  // report 48000 here and still hand back a plausible clip with a plausible
  // soundtrack.
  CHECK(result.sample_rate == kFixtureAudioRate);
  const std::string rendered_audio = ReadAll(std::string(result.audio_path));
  const std::string source_audio = ReadAll(wav);
  // It is the take WINDOWED to the clip, not the whole 2 s file: upstream
  // returns `decoded_audio.waveform`, which is what `decode_audio_from_file`
  // produced AFTER `max_duration` (a2vid_two_stage.py:196, :303), and
  // `audio_max_duration` defaults to `num_frames / frame_rate` (:369-371). This
  // fixture is 9 frames at 24 fps, so 0.375 s of stereo 16-bit at the audio
  // VAE's rate.
  constexpr size_t kWindowSamples = static_cast<size_t>(kFixtureAudioRate) * 375 / 1000;
  constexpr size_t kPcmBytes = kWindowSamples * 2 * 2;
  constexpr size_t kHeader = 44;
  INFO("rendered " << rendered_audio.size() << " B, source " << source_audio.size() << " B");
  CHECK(rendered_audio.size() == kHeader + kPcmBytes);
  REQUIRE(source_audio.size() >= kHeader + kPcmBytes);
  // And those bytes are the SOURCE's leading samples, unaltered. This is the
  // assertion the sample rate alone cannot make: a build that resampled or
  // re-encoded the take would still report the same rate and the same byte
  // count. Counted rather than compared with `!=`, so a failure prints a number
  // instead of a screenful of PCM.
  size_t audio_differing = 0;
  for (size_t i = 0; i < kPcmBytes; ++i) {
    if (rendered_audio[kHeader + i] != source_audio[kHeader + i]) ++audio_differing;
  }
  INFO("audio PCM bytes differing: " << audio_differing << " of " << kPcmBytes);
  CHECK(audio_differing == 0);

  SUBCASE("the truncation keeps the HEAD of the encode, not the tail") {
    // THE LINE THIS ROW EXISTS TO PORT, and until this subcase existed its
    // DIRECTION was asserted by nothing. Upstream slices
    // `encoded_audio_latent[:, :, : audio_shape.frames]` (a2vid_two_stage.py:202)
    // — the LEADING frames. Every assertion above survives a tail slice
    // unchanged: `audio_latent_digest != 0` and `audio_latent_absmax > 0.0` hold
    // for either window, and the "different start time gives a different digest"
    // control holds too, because BOTH windows shift by the same amount. MEASURED:
    // a build truncating to the tail passed the whole 484-test gate.
    //
    // The reference is the SAME take encoded from a LONGER window. The take is
    // 2 s; the clip is 0.375 s; the render above read exactly the clip's worth
    // and this one reads all 2 s. Both are truncated to the SAME target frame
    // count, so under a head slice both must be the take's first
    // `target_frames` latent frames — bit-identical — and under a tail slice
    // this one is the last frames of a five-times-longer encode, which is a
    // different second of the file entirely.
    //
    // Bit-identical is not a hope, it is a property of the chain: `Ltx2SlaneyMel`
    // framing is `center=True` with a FIXED n_fft/2 lookahead, the encoder's
    // convolutions pad on the leading edge only (`causality_axis = height`,
    // ltx2_audio_vae.cpp:146-151 and :926-931), this config carries no attention
    // at any level, and PixelNorm is per-(t, f) location — so latent frame `t`
    // is a function of a bounded prefix of the waveform and NOT of how much
    // audio follows it. The one length-dependent effect is the reflect pad at
    // the waveform's right edge, and it cannot reach the kept frames: it touches
    // mel frames within n_fft/2 of the end, and the target frames span only the
    // first ~4 * target_frames mel frames of a 57-frame short encode.
    vllm::multimodal::VideoGenParams whole = FixtureGen(ws.root + "/a2v_whole");
    whole.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
    whole.extras[vllm::multimodal::kLtx2AudioMaxDurationExtra] = "2.0";
    (void)engine->Generate(whole);
    const vllm::multimodal::Ltx2ConditioningTrace whole_trace = ltx2->last_conditioning();
    CHECK(whole_trace.audio_conditioned);
    // Same grid either way: the token count comes from the CLIP (types.py:164-181).
    CHECK(whole_trace.audio_tokens == trace.audio_tokens);
    // AND THE SAME CONTENT. This is the whole assertion: a tail slice cannot
    // produce it, because the two encodes it slices are of different lengths.
    CHECK(whole_trace.audio_latent_digest == trace.audio_latent_digest);
    CHECK(whole_trace.audio_latent_absmax == trace.audio_latent_absmax);

    // And the equality is not vacuous — the digest DOES move when the head of
    // the window moves. Without this control, a build that hashed a constant
    // would satisfy the equality above for the wrong reason.
    vllm::multimodal::VideoGenParams later = FixtureGen(ws.root + "/a2v_later");
    later.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
    later.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = "1.0";
    later.extras[vllm::multimodal::kLtx2AudioMaxDurationExtra] = "1.0";
    (void)engine->Generate(later);
    const vllm::multimodal::Ltx2ConditioningTrace later_trace = ltx2->last_conditioning();
    CHECK(later_trace.audio_tokens == trace.audio_tokens);
    CHECK(later_trace.audio_latent_digest != trace.audio_latent_digest);
  }

  SUBCASE("a DIFFERENT window of the take produces a DIFFERENT latent") {
    // The half a single digest cannot make. Without it, a build that ignored the
    // file and hashed a constant passes every assertion above. The shapes stay
    // equal and only the SAMPLES move, so this isolates "were the samples read".
    vllm::multimodal::VideoGenParams shifted = FixtureGen(ws.root + "/a2v_shifted");
    shifted.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
    shifted.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = "0.5";
    (void)engine->Generate(shifted);
    const vllm::multimodal::Ltx2ConditioningTrace shifted_trace = ltx2->last_conditioning();
    CHECK(shifted_trace.audio_conditioned);
    CHECK(shifted_trace.audio_tokens == trace.audio_tokens);
    CHECK(shifted_trace.audio_latent_digest != trace.audio_latent_digest);
  }

  SUBCASE("without an audio file the stream is neither conditioned nor frozen") {
    // The negative control. `audio_frozen` has to TRACK the request; pinned true
    // it would make the case above assert a constant, and pinned false it would
    // silently denoise a supplied take.
    vllm::multimodal::VideoGenParams plain = FixtureGen(ws.root + "/plain");
    const vllm::multimodal::VideoResult plain_result = engine->Generate(plain);
    const vllm::multimodal::Ltx2ConditioningTrace plain_trace = ltx2->last_conditioning();
    CHECK_FALSE(plain_trace.audio_conditioned);
    CHECK_FALSE(plain_trace.audio_frozen);
    // And the sigma control is a CONTROL: a build that pinned the audio sigma to
    // zero unconditionally would pass the frozen case above while silently
    // changing every ordinary joint render.
    CHECK(plain_trace.audio_sigma_max > 0.0);
    // Same token count either way: the audio grid comes from the CLIP's duration
    // (types.py:164-181), not from the file, so a differing count here would mean
    // the supplied take had resized the stream.
    CHECK(plain_trace.audio_tokens == trace.audio_tokens);
    CHECK(plain_trace.audio_latent_digest != trace.audio_latent_digest);

    // AND THE PIXELS MOVED. Everything above this line is read off a trace, and
    // a trace is a change detector: a build that encoded the file, recorded it
    // faithfully and then handed the DiT the zero latent it always had would
    // satisfy every assertion so far. LTX-2.5 joins the two streams by explicit
    // audio<->video cross-attention, so a conditioned audio stream MUST move the
    // video — same seed, same prompt, same geometry, and the only difference is
    // the take. This is the assertion that says the conditioning reached the
    // forward rather than reaching a struct.
    const std::string with_audio = ReadAll(std::string(result.frame_dir) + "/frame_000000.ppm");
    const std::string without =
        ReadAll(std::string(plain_result.frame_dir) + "/frame_000000.ppm");
    CHECK(with_audio.size() == without.size());
    // Compared as a COUNT of differing bytes rather than as two strings, because
    // doctest prints the operands of a failing CHECK and these are binary PPMs:
    // a plain `!=` turns one regression into a screenful of raw pixel bytes and,
    // worse, non-UTF-8 output that a harness reading the log can choke on.
    size_t differing = 0;
    const size_t common = std::min(with_audio.size(), without.size());
    for (size_t i = 0; i < common; ++i) {
      if (with_audio[i] != without[i]) ++differing;
    }
    INFO("frame_000000.ppm differs in " << differing << " of " << common << " bytes");
    CHECK(differing > 0);
  }
}

TEST_CASE("ltx2 video: every audio-input mismatch is refused BY WHAT IS WRONG") {
  // Each of these renders a finished clip if it is accepted, which is why every
  // one is a refusal rather than a conversion. The assertions hold each message
  // to naming the numbers a reader needs, not merely to throwing.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));

  auto refusal = [&](const std::string& dir,
                     void (*arm)(vllm::multimodal::VideoGenParams&, const std::string&),
                     const std::string& wav) {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + dir);
    arm(gen, wav);
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("expected a refusal for " << dir);
      return std::string();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };
  auto just_path = [](vllm::multimodal::VideoGenParams& g, const std::string& w) {
    g.extras[vllm::multimodal::kLtx2AudioPathExtra] = w;
  };

  SUBCASE("a sample rate the mel front-end does not target") {
    // Upstream RESAMPLES here (ops.py:40) with an arbitrary-ratio polyphase
    // kaiser resampler this project has not ported. Reading 44.1 kHz samples at
    // the checkpoint's rate pitches and time-shifts the conditioning while every
    // shape checks out, so both rates go in the message.
    //
    // The TARGET rate asserted here is the fixture's 24000, which is neither the
    // shipped 16000 nor `Ltx2ParseAudioEncoderConfig`'s own default. That is the
    // point: with the fixture at 16000 this assertion passed against a parser
    // that never read `params.sampling_rate` at all.
    const std::string wav = WriteWav(ws.root + "/44k.wav", 2, 44100, 2.0);
    const std::string message = refusal("rate", just_path, wav);
    INFO("refusal: " << message);
    CHECK(message.find("44100") != std::string::npos);
    CHECK(message.find(std::to_string(kFixtureAudioRate)) != std::string::npos);
    CHECK(message.find("ops.py:40") != std::string::npos);
  }

  SUBCASE("a channel count the encoder does not declare") {
    // `MiniMaxH3ReadWav` would REPEAT a mono take across both channels
    // (minimax_h3.h:1839-1845). That is H3's contract; LTX-2 hands the file's own
    // channel count to a conv declaring 2, so a mono file is an error upstream
    // too, and duplicating it would condition on audio nobody supplied.
    const std::string wav = WriteWav(ws.root + "/mono.wav", 1, kFixtureAudioRate, 2.0);
    const std::string message = refusal("channels", just_path, wav);
    INFO("refusal: " << message);
    CHECK(message.find("1 audio channel") != std::string::npos);
    CHECK(message.find("in_channels = 2") != std::string::npos);
  }

  SUBCASE("a take SHORTER than the clip") {
    // The subtle one, and the reason the truncation and the assertion have to be
    // read together. `a2vid_two_stage.py:202` truncates and never pads, and
    // `tools.py:253-255` then asserts the latent matches the target shape — so a
    // short take is an ERROR upstream, not a short latent. Padding it here would
    // weld silence onto the end of the take and still render.
    const std::string wav = WriteWav(ws.root + "/short.wav", 2, kFixtureAudioRate, 0.04);
    const std::string message = refusal("short", just_path, wav);
    INFO("refusal: " << message);
    CHECK(message.find("tools.py:253-255") != std::string::npos);
    CHECK(message.find("a2vid_two_stage.py:202") != std::string::npos);
  }

  SUBCASE("the take is measured against the checkpoint's OWN FFT size") {
    // This case exists to make `stft.filter_length` observable, and it took a
    // surviving mutation to find out that nothing else can. MEASURED: reading
    // `stft.n_fft` — the name torchaudio uses, the name a reader writes from
    // memory, and NOT upstream's key — instead of `stft.filter_length` passed
    // the focused gate at 3 cases / 95 assertions with the decoy already in the
    // fixture. The decoy is necessary and it is not sufficient.
    //
    // The reason is worth writing down: the FFT size moves the mel VALUES and
    // not the frame count, because `frames = 1 + (samples + 2 * (n_fft / 2) -
    // n_fft) / hop` and the pad cancels the window exactly. Every other audio
    // assertion in this file is DIFFERENTIAL — one render against another — so a
    // wrong FFT size moves both sides together and nothing notices.
    //
    // What it does change is which takes are decodable at all. `center=True`
    // reflect-pads by `n_fft / 2` and torch.stft requires the waveform to be
    // longer than that pad. 360 samples sits BETWEEN the two half-windows: it is
    // longer than the fixture's declared 512/2 and shorter than the decoy's
    // 1024/2. So a build reading the decoy refuses this take at the mel front
    // end, where the correct one carries it through to the clip-length check —
    // and THAT difference is the assertion.
    const std::string wav = WriteWav(ws.root + "/hairline.wav", 2, kFixtureAudioRate, 0.015);
    const std::string message = refusal("hairline", just_path, wav);
    INFO("refusal: " << message);
    // It reached the LENGTH check, which means the mel front-end accepted it.
    CHECK(message.find("tools.py:253-255") != std::string::npos);
    // And it did NOT stop at the front end's own reflect-pad constraint.
    CHECK(message.find("center=True") == std::string::npos);
  }

  SUBCASE("a file that is not RIFF/WAVE") {
    const std::string path = ws.root + "/not.wav";
    {
      std::ofstream out(path, std::ios::binary);
      out << "ID3\x04\x00\x00 this is not a wav file at all, honestly, not even close";
    }
    const std::string message = refusal("container", just_path, path);
    INFO("refusal: " << message);
    CHECK(message.find("RIFF/WAVE") != std::string::npos);
    CHECK(message.find("demuxer") != std::string::npos);
  }

  SUBCASE("a window that starts past the end of the take") {
    const std::string wav = WriteWav(ws.root + "/late.wav", 2, kFixtureAudioRate, 1.0);
    const std::string message = refusal(
        "late",
        [](vllm::multimodal::VideoGenParams& g, const std::string& w) {
          g.extras[vllm::multimodal::kLtx2AudioPathExtra] = w;
          g.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = "5.0";
        },
        wav);
    INFO("refusal: " << message);
    CHECK(message.find("past the end") != std::string::npos);
    // The REASON is pinned as well as the fact, because a refusal that names the
    // wrong upstream line sends the next reader to the wrong file and no test
    // that checks only "it threw" can see that. Read at the pin: every decoded
    // frame ends before `start_time`, so the loop's `if frame_end < start_time:
    // continue` drops all of them and `if not samples: return None`
    // (decode.py:271-272, :281-282) fires — and THAT is what makes
    // `decoded_audio is None` true at a2vid_two_stage.py:197-198.
    CHECK(message.find("decode.py:271-272") != std::string::npos);
    CHECK(message.find("a2vid_two_stage.py:197-198") != std::string::npos);
  }

  SUBCASE("a window whose DURATION rounds to no samples") {
    // The sibling of the case above, and a DIFFERENT upstream path — which is
    // the whole reason it has its own case. `max_samples = round(max_duration *
    // sample_rate)` is 0 here, and `audio[..., :0]` (decode.py:295-296) is an
    // `Audio` carrying zero samples rather than `None`, so `decoded_audio is
    // None` is FALSE and a2vid_two_stage.py:197-198 never fires. Upstream fails
    // one or two hops later instead. We refuse at the window, which is STRICTER
    // than upstream, and the message has to SAY that rather than borrowing the
    // other path's citation.
    const std::string wav = WriteWav(ws.root + "/tiny.wav", 2, kFixtureAudioRate, 1.0);
    const std::string message = refusal(
        "tiny",
        [](vllm::multimodal::VideoGenParams& g, const std::string& w) {
          g.extras[vllm::multimodal::kLtx2AudioPathExtra] = w;
          g.extras[vllm::multimodal::kLtx2AudioMaxDurationExtra] = "0.00001";
        },
        wav);
    INFO("refusal: " << message);
    CHECK(message.find("no audio samples") != std::string::npos);
    CHECK(message.find("STRICTER") != std::string::npos);
    CHECK(message.find("decode.py:295-296") != std::string::npos);
    // And it must NOT claim the raise the other path gets.
    CHECK(message.find("does not") != std::string::npos);
  }

  SUBCASE("a window knob WITHOUT a file cannot do anything, so it is refused") {
    // The silent-ignore shape this whole extras surface exists to prevent: an
    // accepted knob that cannot affect the render reads as "the feature does not
    // work" rather than as "it needs the other flag".
    const std::string message = refusal(
        "orphan",
        [](vllm::multimodal::VideoGenParams& g, const std::string&) {
          g.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = "1.0";
        },
        std::string());
    INFO("refusal: " << message);
    CHECK(message.find("audio_path") != std::string::npos);
    CHECK(message.find("ignored") != std::string::npos);
  }

  SUBCASE("a start time that is not a number is refused, not truncated") {
    // `ExtraInt` would take "0.5s" apart differently; a seconds knob parsed as an
    // integer would window the wrong second of the take and still render.
    const std::string wav = WriteWav(ws.root + "/num.wav", 2, kFixtureAudioRate, 2.0);
    const std::string message = refusal(
        "nan",
        [](vllm::multimodal::VideoGenParams& g, const std::string& w) {
          g.extras[vllm::multimodal::kLtx2AudioPathExtra] = w;
          g.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = "0.5s";
        },
        wav);
    INFO("refusal: " << message);
    CHECK(message.find("audio_start_time") != std::string::npos);
  }
}

TEST_CASE("ltx2 video: audio_path on a decoder-only checkpoint names the missing WEIGHTS") {
  // The refusal that separates "this feature is unported" from "this CHECKPOINT
  // cannot do it" — a distinction #758 records this project as repeatedly
  // getting wrong. `Ltx2AudioEncoderForward` and its mel front-end have been
  // ported and gated since `cefacd2d0`; a decoder-only audio VAE is missing the
  // WEIGHTS, and the message has to say so, or a reader goes looking for code
  // that is already there.
  Workspace ws;
  // The audio VAE as EVERY LTX-2.5 checkpoint looked before this row.
  ltx2_fixture::WriteReducedAudioVae(ltx2_fixture::ReducedAudioDecoderConfig(),
                                     ltx2_fixture::ReducedVocoderBweConfig(),
                                     ws.paths.audio_vae, /*with_encoder=*/false);
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));

  const std::string wav = WriteWav(ws.root + "/drive.wav", 2, kFixtureAudioRate, 2.0);
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/noenc");
  gen.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
  std::string message;
  try {
    (void)engine->Generate(gen);
    FAIL_CHECK("a decoder-only audio VAE must refuse audio_path");
  } catch (const std::exception& e) {
    message = e.what();
  }
  INFO("refusal: " << message);
  CHECK(message.find("audio_vae.encoder.") != std::string::npos);
  CHECK(message.find("WEIGHTS") != std::string::npos);
  // And it must NOT send the reader after unwritten code: the forward is ported.
  CHECK(message.find("audio_vae.py:190-246") != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// RETAKE — row LTX25-RETAKE (#924), spec .agents/specs/ltx25-retake.md
//
// These are the REACHABILITY cases. They enter through `LoadVideoEngine` and
// `Generate`, which is where `vllm_video_generate` arrives, and not through
// `Ltx2TemporalRegionMaskVideo`. The value-level cases for the ported pieces are
// `test_ltx2_retake`; per .agents/reachability.md those localize a failure and
// are not the proof, because they stay green when the production call site is
// deleted.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// A `retake` engine. The kind is a LOAD extra and it has to be this one: the
// distilled two-stage recipe renders its first stage at half resolution
// (`spatial_downscale = 2`), and a full-resolution source latent does not fit
// that grid.
vllm::multimodal::VideoModelParams RetakeParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp = FixtureParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "retake";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("retake");
  return mp;
}

// The source clip: `frame_%06d.ppm` numbered from 0, which is what
// `vllm_video_params::ref_video` has always meant and what `minimax-h3-gen`
// writes. 9 frames satisfies 8k+1 and 64x64 is a multiple of 32, so the
// fixture's (8, 32, 32) factors give a 2 x 2 x 2 latent — 8 tokens.
std::string WriteRetakeClip(const std::string& dir, int frames, int height, int width,
                            unsigned seed_base = 41) {
  ::mkdir(dir.c_str(), 0755);
  for (int i = 0; i < frames; ++i) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06d.ppm", i);
    WriteBytes(dir + name, ConditioningPpm(height, width, seed_base + static_cast<unsigned>(i)));
  }
  return dir;
}

// NOT `FixtureGen`: a retake takes its geometry from the source clip and refuses
// an explicit width, height, frame count or duration, because upstream's own
// parser omits all of them ("no height/width/num-frames; resolution comes from
// input video", utils/args.py:851-854).
vllm::multimodal::VideoGenParams RetakeGen(const std::string& out_dir, const std::string& clip,
                                           double start, double end) {
  vllm::multimodal::VideoGenParams gen;
  gen.has_seed = true;
  gen.seed = 7;
  gen.output_dir = out_dir;
  gen.ref_video_dir = clip;
  gen.extras[vllm::multimodal::kLtx2RetakeStartTimeExtra] = std::to_string(start);
  gen.extras[vllm::multimodal::kLtx2RetakeEndTimeExtra] = std::to_string(end);
  gen.extras[vllm::multimodal::kLtx2RetakeFrameRateExtra] = "24";
  return gen;
}

}  // namespace

TEST_CASE("ltx2 retake: a time window REGENERATES and the render carries the source clip") {
  // THE RED-FIRST CASE. Before this row the retake extras were refused as
  // unknown per-generation extras and `ref_video_dir` was refused outright, so
  // nothing here could reach the new code at all.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(RetakeParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);

  const std::string clip = WriteRetakeClip(ws.root + "/clip", 9, 64, 64);
  // 24 fps against the fixture's temporal factor 8. With `causal_fix` — the CALL
  // SITE's default at noise_mask_cond.py:33, not `get_pixel_coords`'s own —
  // latent frame 0 spans pixel frames [0, 1) and latent frame 1 spans [1, 9),
  // i.e. [0, 1/24) s and [1/24, 9/24) s. The window [0.05, 0.10) therefore
  // overlaps frame 1 alone: 4 of the 8 tokens.
  const vllm::multimodal::VideoResult result =
      engine->Generate(RetakeGen(ws.root + "/out", clip, 0.05, 0.10));

  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
  CHECK(trace.completed);
  CHECK(trace.retake_conditioned);
  // BOTH numbers, because an all-ones mask and an all-zeros mask are each a
  // plausible failure that one count cannot tell from a correct one.
  CHECK(trace.retake_total_tokens == 8);
  CHECK(trace.retake_masked_tokens == 4);
  // The lower bound: a source clip that encoded to zeros has the right shape,
  // the right token count and the right mask, and renders.
  CHECK(trace.retake_latent_absmax > 0.0);
  CHECK(trace.retake_latent_digest != 0);
  // The geometry came from the CLIP, not from the recipe's params table and not
  // from the request, which carried none.
  CHECK(result.frame_count == 9);

  // THE RENDER DEPENDS ON THE SOURCE CLIP'S PIXELS, and this is the assertion
  // the trace above cannot make. `retake_latent_absmax` observes the ENCODE; it
  // says nothing about whether the encoded latent was ever handed to the phase.
  // A build that read the clip, encoded it, recorded the digest and then seeded
  // the stream with zeros satisfies every other check in this case and renders a
  // clip of the right length with the right mask. Two DIFFERENT sources at the
  // same seed and the same window must therefore produce different pixels; on
  // that build both start from zeros and produce identical ones.
  //
  // This survived as a mutation before it was written (M2 in the row's table),
  // which is why it is here rather than in a comment.
  const std::string other = WriteRetakeClip(ws.root + "/clip_b", 9, 64, 64, /*seed_base=*/113);
  REQUIRE(ReadAll(clip + "/frame_000000.ppm") != ReadAll(other + "/frame_000000.ppm"));
  const vllm::multimodal::VideoResult from_other =
      engine->Generate(RetakeGen(ws.root + "/out_b", other, 0.05, 0.10));
  const std::string frame_a = ReadAll(std::string(result.frame_dir) + "/frame_000000.ppm");
  const std::string frame_b = ReadAll(std::string(from_other.frame_dir) + "/frame_000000.ppm");
  REQUIRE(frame_a.size() == frame_b.size());
  size_t differing = 0;
  for (size_t i = 0; i < frame_a.size(); ++i) {
    if (frame_a[i] != frame_b[i]) ++differing;
  }
  CHECK_MESSAGE(differing > 0,
                "two different source clips rendered byte-identical frames at the same seed and "
                "window, so the encoded source latent never reached the phase");
}

TEST_CASE("ltx2 retake: the mask takes the CALL SITE's causal_fix, not the function's") {
  // The two upstream defaults disagree — `get_pixel_coords` declares False
  // (patchifiers.py:140) and `TemporalRegionMask` calls it with True
  // (noise_mask_cond.py:33) — and the case above cannot tell them apart: at 24
  // fps the window [0.05, 0.10) selects one latent frame either way. This one
  // picks a window where the two DISAGREE about the count.
  //
  // 9 pixel frames is a 2-frame latent. WITH the fix, frame 0 spans pixel frames
  // [0, 1) and frame 1 spans [1, 9), i.e. [0, 0.0417) s and [0.0417, 0.375) s.
  // WITHOUT it, frame 0 spans [0, 8) and frame 1 spans [8, 16), i.e. [0, 0.333) s
  // and [0.333, 0.667) s. The window [0.30, 0.40) therefore selects ONE frame
  // with the fix and BOTH without it.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(RetakeParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  const std::string clip = WriteRetakeClip(ws.root + "/clip", 9, 64, 64);

  (void)engine->Generate(RetakeGen(ws.root + "/late", clip, 0.30, 0.40));
  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx2->last_conditioning();
  CHECK(trace.retake_total_tokens == 8);
  CHECK_MESSAGE(trace.retake_masked_tokens == 4,
                "the production call site passed causal_fix=false: without the causal rewrite "
                "this window overlaps BOTH latent frames and masks all 8 tokens");
}

TEST_CASE("ltx2 retake: regenerate_video=0 FREEZES the clip, mask and scalar sigma both") {
  // `frozen = not regenerate_video` (retake.py:274) has two halves and upstream
  // says so in as many words: it "zeros the denoise mask and marks the resulting
  // LatentState so Modality.sigma is forced to 0 (not only per-token
  // timesteps)" (utils/types.py:104-106). A build that zeroed only the mask
  // tells the DiT the clean source clip is noisy, and still renders.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(RetakeParams(ws.paths));
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  const std::string clip = WriteRetakeClip(ws.root + "/clip", 9, 64, 64);

  vllm::multimodal::VideoGenParams gen = RetakeGen(ws.root + "/frozen", clip, 0.05, 0.10);
  gen.extras[vllm::multimodal::kLtx2RegenerateVideoExtra] = "0";
  const vllm::multimodal::VideoResult result = engine->Generate(gen);
  const vllm::multimodal::Ltx2ConditioningTrace frozen = ltx2->last_conditioning();
  CHECK(frozen.completed);
  CHECK_FALSE(frozen.retake_conditioned);
  CHECK(frozen.video_sigma_max == 0.0);
  CHECK(result.frame_count == 9);

  // The CONTROL, so the assertion above is not satisfied by a build that never
  // raises the video sigma at all.
  const vllm::multimodal::VideoResult live =
      engine->Generate(RetakeGen(ws.root + "/live", clip, 0.05, 0.10));
  CHECK(live.frame_count == 9);
  CHECK(ltx2->last_conditioning().video_sigma_max > 0.0);
}

TEST_CASE("ltx2 retake: every refusal names what is wrong") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(RetakeParams(ws.paths));
  const std::string clip = WriteRetakeClip(ws.root + "/clip", 9, 64, 64);

  auto refusal = [&](const char* what,
                     const std::function<void(vllm::multimodal::VideoGenParams&)>& tweak) {
    vllm::multimodal::VideoGenParams gen =
        RetakeGen(ws.root + "/ref_" + what, clip, 0.05, 0.10);
    tweak(gen);
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("expected a refusal for " << what);
      return std::string();
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
  };

  SUBCASE("an inverted window reports BOTH values, as upstream does") {
    const std::string msg = refusal("window", [](vllm::multimodal::VideoGenParams& g) {
      g.extras[vllm::multimodal::kLtx2RetakeEndTimeExtra] = "0.010000";
    });
    INFO(msg);
    CHECK(msg.find("must be less than end_time") != std::string::npos);
  }

  SUBCASE("a frame count off the 8k+1 grid NAMES the snapped value") {
    const std::string other = WriteRetakeClip(ws.root + "/ragged", 10, 64, 64);
    const std::string msg = refusal("frames", [&](vllm::multimodal::VideoGenParams& g) {
      g.ref_video_dir = other;
    });
    INFO(msg);
    CHECK(msg.find("8k+1") != std::string::npos);
    CHECK(msg.find("use a video with 9 frames") != std::string::npos);
  }

  SUBCASE("a resolution off the 32 grid names both axes") {
    const std::string other = WriteRetakeClip(ws.root + "/odd", 9, 64, 48);
    const std::string msg = refusal("resolution", [&](vllm::multimodal::VideoGenParams& g) {
      g.ref_video_dir = other;
    });
    INFO(msg);
    CHECK(msg.find("multiples of 32") != std::string::npos);
    CHECK(msg.find("48x64") != std::string::npos);
  }

  SUBCASE("a missing frame rate names the knob and why it cannot be defaulted") {
    const std::string msg = refusal("fps", [](vllm::multimodal::VideoGenParams& g) {
      g.extras.erase(vllm::multimodal::kLtx2RetakeFrameRateExtra);
    });
    INFO(msg);
    CHECK(msg.find(vllm::multimodal::kLtx2RetakeFrameRateExtra) != std::string::npos);
    CHECK(msg.find("decode.py:213-215") != std::string::npos);
  }

  SUBCASE("no source clip refuses the CONTAINER by name") {
    const std::string msg = refusal("noclip", [](vllm::multimodal::VideoGenParams& g) {
      g.ref_video_dir.clear();
    });
    INFO(msg);
    CHECK(msg.find("frame_%06d.ppm") != std::string::npos);
    CHECK(msg.find("no demuxer is vendored") != std::string::npos);
  }

  SUBCASE("an explicit geometry is refused rather than silently preferred") {
    const std::string msg = refusal("geometry", [](vllm::multimodal::VideoGenParams& g) {
      g.height = 64;
      g.width = 64;
    });
    INFO(msg);
    CHECK(msg.find("takes its geometry from the SOURCE clip") != std::string::npos);
  }

  SUBCASE("retake plus an audio file refuses rather than picking one soundtrack") {
    const std::string wav = WriteWav(ws.root + "/both.wav", 2, kFixtureAudioRate, 1.0);
    const std::string msg = refusal("both", [&](vllm::multimodal::VideoGenParams& g) {
      g.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
    });
    INFO(msg);
    CHECK(msg.find("retake.py:250-256") != std::string::npos);
  }

  SUBCASE("a retake knob with no retake window is refused rather than ignored") {
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/orphan");
    gen.extras[vllm::multimodal::kLtx2RegenerateAudioExtra] = "0";
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("a regenerate_audio with no retake window must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("no retake for it to configure") != std::string::npos);
    }
  }
}

TEST_CASE("ltx2 retake: the wrong recipe refuses, and the reference arm still does") {
  Workspace ws;
  // The DEFAULT recipe, not the retake one.
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));
  const std::string clip = WriteRetakeClip(ws.root + "/clip", 9, 64, 64);

  SUBCASE("the distilled two-stage recipe cannot carry a retake") {
    try {
      (void)engine->Generate(RetakeGen(ws.root + "/wrongkind", clip, 0.05, 0.10));
      FAIL_CHECK("a retake on distilled_two_stage must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("'retake' pipeline recipe") != std::string::npos);
      CHECK(msg.find("first stage at half") != std::string::npos);
    }
  }

  SUBCASE("a reference clip WITHOUT the retake knobs is still refused, and #975 stays open") {
    // THE LOCAL FACT this row's refusal edit turns on: the LTX side now reads
    // `ref_video_dir`, so the sentence claiming nothing did is gone. A case that
    // asserted only upstream symbol names could not see that going stale — which
    // is exactly how the sentence survived (#987).
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/refstill");
    gen.ref_video_dir = clip;
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("reference-video conditioning must still be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      const size_t ruled_out = msg.find("WHAT IS *NOT* THE REASON");
      REQUIRE(ruled_out != std::string::npos);
      const size_t stale = msg.find("nothing reads `ref_video_dir` at all");
      const bool only_as_ruled_out = (stale == std::string::npos) || (stale > ruled_out);
      CHECK_MESSAGE(only_as_ruled_out,
                    "the refusal states as a CAUSE that nothing reads ref_video_dir, and this "
                    "suite has just rendered a retake that reads it");
      // And it names the reader that DOES exist, so the next reader goes looking
      // for the reference item's geometry rather than for a file walker.
      CHECK(msg.find("Ltx2ReadFrameDirectory") != std::string::npos);
    }
  }
}

// ─── TEXT-TO-AUDIO (row LTX25-T2A-ONE-STAGE, issue #1005) ────────────────────
//
// `T2AOneStagePipeline` (ltx-pipelines t2a_one_stage.py:43, `__call__` at :109)
// at Lightricks/LTX-2 @ fd4ded7f. UPSTREAM SHIPS NO TESTS at that pin — `find
// /home/mudler/_git/LTX-2 -name 'test_*.py'` returns 0 — so there is nothing to
// port, and what follows pins upstream's BEHAVIOURS against `file:line` anchors
// instead. At least one assertion in each refusal case is tied to a LOCAL fact,
// because a case that asserts only upstream symbol names cannot see a refusal
// whose claim about THIS tree has gone stale.
namespace {

// A t2a engine on the shipped fixture. NO `video_vae_path`, which is the load
// half of the row: upstream's `T2AOneStagePipeline.__init__` never calls
// `model_paths.video_vae()` (t2a_one_stage.py:68-107).
vllm::multimodal::VideoModelParams T2aParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.audio_vae_path = paths.audio_vae;
  mp.encoder_path = paths.encoder;
  mp.extras[vllm::multimodal::kLtx2EncoderConfigPathExtra] = paths.encoder_config;
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "t2a_one_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("t2a_one_stage");
  mp.device = 0;
  return mp;
}

vllm::multimodal::VideoGenParams T2aGen(const std::string& out_dir, const std::string& prompt) {
  vllm::multimodal::VideoGenParams gen;
  gen.prompt = prompt;
  gen.num_frames = 25;
  gen.steps = 2;  // two sigma intervals is enough to exercise the loop
  gen.has_seed = true;
  gen.seed = 11;
  gen.output_dir = out_dir;
  // The reduced DiT has TWO blocks (ltx2_video_fixture.h `ReducedDitParams`), so
  // the params table's own `stg_blocks = [28]` is out of range here. Named
  // explicitly rather than by turning STG off, because the default-block refusal
  // is its own case below and this one is about the render.
  gen.extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "1";
  // The recipe's own default negative prompt is upstream's
  // `DEFAULT_NEGATIVE_PROMPT` — an English sentence — and this fixture's
  // tokenizer carries a three-token vocabulary. Overriding it here is what the
  // `--negative-prompt` flag is for (utils/args.py:1083-1088), and the DEFAULT's
  // reachability is asserted separately on the recipe rather than by pushing an
  // out-of-vocabulary string through a reduced tokenizer.
  gen.extras[vllm::multimodal::kLtx2NegativePromptExtra] = "c b a";
  return gen;
}

}  // namespace

TEST_CASE("ltx2 t2a: an audio-only render returns a waveform and NO picture") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(T2aParams(ws.paths));
  REQUIRE(engine != nullptr);
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  CHECK(ltx->pipeline_kind() == "t2a_one_stage");

  const std::string out = ws.root + "/t2a";
  const vllm::multimodal::VideoResult result = engine->Generate(T2aGen(out, "a b c"));

  // 1. NO PICTURE, said three ways, because each catches a different build.
  //    `frame_count` catches a render that produced frames; the empty
  //    `frame_dir` and `mux_argv` catch one that produced none and still handed
  //    the caller a directory and an ffmpeg command over a pattern matching no
  //    file.
  CHECK(result.frame_count == 0);
  CHECK(result.frame_dir.empty());
  CHECK(result.mux_argv.empty());
  CHECK(result.mux_output_path.empty());
  CHECK(result.width == 0);
  CHECK(result.height == 0);
  // ...AND NO FRAME ON DISK. This is the half the fields cannot make: a build
  // that wrote `frame_000000.ppm` and reported zero passes every check above.
  {
    std::ifstream frame(out + "/frame_000000.ppm", std::ios::binary);
    CHECK_MESSAGE(!frame.good(), "an audio-only render wrote a frame");
  }

  // 2. THERE IS SOUND, at the vocoder's own output rate, and it is not silence.
  //    The lower bound is the assertion a size or a rate cannot make: a
  //    zero-initialized decode produces a perfectly well-formed WAV of exactly
  //    the right length.
  CHECK(result.sample_rate == 48000);
  const std::string wav = ReadAll(result.audio_path);
  REQUIRE(wav.size() > 44);
  CHECK(wav.compare(0, 4, "RIFF") == 0);
  {
    int peak = 0;
    for (size_t i = 44; i + 1 < wav.size(); i += 2) {
      int16_t s = 0;
      std::memcpy(&s, wav.data() + i, sizeof(s));
      peak = std::max(peak, s < 0 ? -static_cast<int>(s) : static_cast<int>(s));
    }
    CHECK_MESSAGE(peak > 0, "the rendered waveform is digital silence");
  }

  const vllm::multimodal::Ltx2ConditioningTrace trace = ltx->last_conditioning();
  CHECK(trace.completed);
  CHECK(trace.t2a_rendered);

  // 3. NO VIDEO STREAM EVER REACHED THE DiT. This is the whole reason the field
  //    exists: upstream's `run_v2a` tests PRESENCE, not `enabled`
  //    (transformer.py:269), so a build that handed the forward a
  //    present-but-disabled video stream would feed video->audio cross attention
  //    from a latent this pipeline never meant to exist — and would return a
  //    waveform of exactly the right length, channel count and sample rate.
  CHECK_FALSE(trace.t2a_video_stream_present);

  // 4. THE GUIDER RAN, arm by arm. The counters are incremented at the forward,
  //    so this is a statement about the passes that were issued rather than
  //    about the parameters that were meant to drive them.
  CHECK(trace.t2a_cond_forwards == 2);
  CHECK(trace.t2a_uncond_forwards == 2);
  CHECK(trace.t2a_perturbed_forwards == 2);
  //    ...and STG perturbed the block that was ASKED for. A count alone cannot
  //    tell block 1 from block 0, and which block is perturbed is the whole of
  //    STG.
  REQUIRE(trace.t2a_perturbed_blocks.size() == 1);
  CHECK(trace.t2a_perturbed_blocks[0] == 1);

  // 5. The latent is populated. A digest alone is stable across a collapse to
  //    zeros; the absmax is the bound that is not.
  CHECK(trace.audio_tokens > 0);
  CHECK(trace.audio_latent_absmax > 1e-6);
}

TEST_CASE("ltx2 t2a: the guidance ARMS are separable, and each one moves the render") {
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = T2aParams(ws.paths);

  struct Out {
    vllm::multimodal::Ltx2ConditioningTrace trace;
    std::string wav;
  };
  const auto render = [&](const std::string& tag,
                          const std::map<std::string, std::string>& overrides) {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    REQUIRE(engine != nullptr);
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/" + tag, "a b c");
    for (const auto& kv : overrides) gen.extras[kv.first] = kv.second;
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    return Out{ltx->last_conditioning(), ReadAll(result.audio_path)};
  };

  const Out full = render("g_full", {});
  // `cfg_scale = 1.0` is `math.isclose(cfg_scale, 1.0)` — upstream's OWN
  // predicate for "no unconditional generation" (guiders.py:275-277), and NOT an
  // exact `!= 1.0` comparison.
  const Out no_cfg = render("g_nocfg", {{vllm::multimodal::kLtx2AudioCfgScaleExtra, "1.0"}});
  const Out no_stg = render("g_nostg", {{vllm::multimodal::kLtx2AudioStgScaleExtra, "0.0"}});

  // Each arm turns off exactly its own forward, and leaves the others alone.
  CHECK(no_cfg.trace.t2a_uncond_forwards == 0);
  CHECK(no_cfg.trace.t2a_perturbed_forwards == full.trace.t2a_perturbed_forwards);
  CHECK(no_stg.trace.t2a_perturbed_forwards == 0);
  CHECK(no_stg.trace.t2a_uncond_forwards == full.trace.t2a_uncond_forwards);
  CHECK(no_cfg.trace.t2a_cond_forwards == full.trace.t2a_cond_forwards);

  // AND EACH ONE CHANGES THE RENDER. Without this, a build that issued the extra
  // forwards and then discarded them would pass every counter above — which is
  // exactly the "recorded value is not a reached one" failure this campaign has
  // already paid for once.
  CHECK(full.wav.size() == no_cfg.wav.size());
  CHECK(full.wav != no_cfg.wav);
  CHECK(full.wav != no_stg.wav);
  CHECK(no_cfg.wav != no_stg.wav);

  // The STG DELTA depends on WHICH block is perturbed. Two builds that perturb
  // different blocks issue the same three forwards and differ only here, so a
  // port that ignored `stg_blocks` and perturbed everything (or nothing) would
  // pass every assertion above.
  const Out block0 = render("g_b0", {{vllm::multimodal::kLtx2AudioStgBlocksExtra, "0"}});
  REQUIRE(block0.trace.t2a_perturbed_blocks.size() == 1);
  CHECK(block0.trace.t2a_perturbed_blocks[0] == 0);
  CHECK(block0.wav != full.wav);
}

TEST_CASE("ltx2 t2a: the refusals name what is missing, and each is checked HERE") {
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = T2aParams(ws.paths);
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);

  const auto refuses = [&](vllm::multimodal::VideoGenParams gen,
                           const char* needle) -> std::string {
    // `needle` is a `const char*`, and doctest stringifies a bare `char*` as a
    // BOOL — a failure would print `true` instead of the string that was looked
    // for. Bound to a std::string before it reaches any doctest macro.
    const std::string want(needle);
    INFO("needle = " << want);
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("expected a refusal naming: " << want);
      return std::string();
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO("refusal = " << msg);
      CHECK_MESSAGE(msg.find(want) != std::string::npos, "the refusal did not name the needle");
      return msg;
    }
  };

  SUBCASE("a resolution is refused rather than accepted and ignored") {
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/r_res", "a b c");
    gen.width = 64;
    gen.height = 64;
    refuses(gen, "height/width are unused");
  }

  SUBCASE("the params table's own STG block is out of range on THIS DiT") {
    // The LOCAL fact, and it is what makes this case able to see staleness. The
    // fixture's DiT has two blocks; upstream's default `stg_blocks` is [28]. The
    // refusal must name the range it checked against, so a build that silently
    // clamped or ignored the index would not produce this message.
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/r_stg", "a b c");
    gen.extras.erase(vllm::multimodal::kLtx2AudioStgBlocksExtra);
    const std::string msg = refuses(gen, "STG block index 28 is outside [0, ");
    // Derived from the tree rather than restated: the range in the message is
    // the DiT the engine actually loaded, not a literal this test also knows.
    const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    CHECK(msg.find("[0, " + std::to_string(ltx->dit_params().num_layers) + ")") !=
          std::string::npos);
  }

  SUBCASE("a perturbed pass over NO block is refused, not run as a no-op") {
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/r_empty", "a b c");
    gen.extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "";
    refuses(gen, "the STG delta would be exactly zero");
  }

  SUBCASE("isolated-modality guidance has no second modality to run over") {
    // Not reachable through an extra by design — there is no `modality_scale`
    // knob — so this asserts the RECIPE pinned it, which is the thing that keeps
    // the refusal unreachable. `Ltx2DetectPipelineParams("2.5")` carries 3.0.
    const vllm::Ltx2PipelineRecipe t2a = vllm::ResolveLtx2PipelineRecipe("t2a_one_stage", "2.5");
    REQUIRE(t2a.audio_only);
    REQUIRE(t2a.phases.size() == 1);
    CHECK(t2a.phases[0].audio_guidance.modality_scale == 1.0);
    CHECK(vllm::Ltx2DetectPipelineParams("2.5").audio_guider.modality_scale == 3.0);
  }

  SUBCASE("a video-only knob on a t2a engine is refused rather than ignored") {
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/r_crf", "a b c");
    gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    refuses(gen, "no meaning on a text-to-audio render");
  }

  SUBCASE("a keyframe has no stream to condition") {
    vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/r_kf", "a b c");
    gen.first_frame_ppm = "P6\n1 1\n255\n\x01\x02\x03";
    refuses(gen, "`video=None`");
  }
}

TEST_CASE("ltx2: the AUDIO guider knobs are NOT t2a-only, and this case used to say they were") {
  // WHAT THIS CASE ASSERTED UNTIL ROW LTX25-GUIDED-VIDEO (#1092): that
  // `audio_cfg_guidance_scale` on a video pipeline is refused "text-to-audio's
  // alone". The premise behind that refusal was that "no other pipeline
  // `__call__` upstream takes a guider argument at all", and it is FALSE about
  // upstream: `default_1_stage_arg_parser` carries the whole audio guider row
  // beside the video one (ltx-pipelines utils/args.py:1011-1075 @ fd4ded7f) and
  // `TI2VidOneStagePipeline` builds `audio_guider_params` from it
  // (ti2vid_one_stage.py:215-218). It was harmless only because the joint
  // render here was unguided, so nothing could have read the knob.
  //
  // The correction is kept as an executable statement rather than a deletion,
  // because "this used to be refused" is exactly what a later reader needs.
  Workspace ws;
  // A `one_stage` engine with the fixture's own text tower, which is the
  // configuration these knobs describe: `distilled_two_stage` fixes its guidance
  // and refuses every override, so asking it would test the other guard.
  vllm::multimodal::VideoModelParams mp = EncoderParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);

  // FIRST, the negative prompt through the TOWER, which is the path the embeds
  // fallback exists to stand in for. The recipe's own default negative prompt is
  // upstream's `DEFAULT_NEGATIVE_PROMPT` -- an English sentence -- and this
  // fixture's tokenizer carries a three-token vocabulary, so it is overridden
  // here exactly as `--negative-prompt` is for (utils/args.py:937-946).
  {
    vllm::multimodal::VideoGenParams gen = PromptedGen(ws.root + "/tower_negative", "a b c");
    gen.steps = 2;
    OneStageFixtureGuidance(&gen);
    gen.extras[vllm::multimodal::kLtx2NegativePromptExtra] = "c b a";
    (void)engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    CHECK(t.video_uncond_forwards == 1);
    // And it was a DIFFERENT forward, so the tower's negative encoding reached
    // the DiT rather than the positive one being handed over twice.
    REQUIRE(!t.video_first_uncond.empty());
    CHECK(t.video_first_uncond != t.video_first_cond);
  }

  // SECOND, the knob itself. At 1.0 on both streams there is no unconditional
  // pass at all (guiders.py:275-277) and nothing to encode. Without this the
  // case would pass on a build that accepted the knob and ignored it, which is
  // the defect the extras surface exists to refuse.
  {
    vllm::multimodal::VideoGenParams gen = PromptedGen(ws.root + "/audio_knob", "a b c");
    gen.steps = 2;
    OneStageFixtureGuidance(&gen);
    gen.extras[vllm::multimodal::kLtx2AudioCfgScaleExtra] = "1.0";
    gen.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "1.0";
    (void)engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    CHECK(t.completed);
    CHECK(t.video_uncond_forwards == 0);
  }

  // The direction that SURVIVES: a knob that describes a picture, on an engine
  // that renders none.
  const std::unique_ptr<vllm::multimodal::VideoEngine> t2a =
      vllm::multimodal::LoadVideoEngine(T2aParams(ws.paths));
  REQUIRE(t2a != nullptr);
  vllm::multimodal::VideoGenParams t2a_gen = T2aGen(ws.root + "/video_knob", "a b c");
  t2a_gen.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "5.0";
  try {
    (void)t2a->Generate(t2a_gen);
    FAIL_CHECK("a VIDEO guider knob must be refused on a text-to-audio engine");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("no meaning on a text-to-audio render") != std::string::npos);
    CHECK(msg.find(vllm::multimodal::kLtx2VideoCfgScaleExtra) != std::string::npos);
  }
}

TEST_CASE("ltx2 t2a: the DiT forward runs ONE stream, and the old guard's reason was wrong") {
  // The lifted refusal, checked against a LOCAL fact rather than against
  // upstream symbol names alone. `Ltx2DitForward` used to demand BOTH streams
  // and blamed the AudioOnly weight contract; the contract is not what blocked
  // it, and the way to see that is that the AV weights this fixture writes are
  // enough to run the audio stream by itself.
  Workspace ws;
  const vllm::SafetensorsFile dit_file = vllm::SafetensorsFile::Open(ws.paths.dit);
  vllm::Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;  // `Ltx2DitForward` is f32 by declaration
  const vllm::Ltx2DitCheckpoint ckpt = vllm::Ltx2LoadDitFromSafetensors(dit_file, options);

  const int64_t tokens = 3;
  // VARYING PER TOKEN, and that is load bearing rather than tidy. A latent whose
  // rows are all equal makes self-attention return a weighted average of
  // identical values — which is exactly the value projection — so the STG
  // perturbation below would be a numeric no-op and the case would report
  // "the perturbation changed nothing" about a build that applies it correctly.
  // Measured: with a constant 0.25 fill this assertion failed on the working
  // implementation.
  std::vector<float> latent(static_cast<size_t>(tokens * ckpt.params.audio_in_channels));
  for (size_t i = 0; i < latent.size(); ++i) {
    latent[i] = 0.25F + 0.01F * static_cast<float>(i % 7) - 0.02F * static_cast<float>(i % 3);
  }
  std::vector<float> timesteps(static_cast<size_t>(tokens), 0.5F);
  std::vector<double> positions(static_cast<size_t>(tokens * 2));
  for (int64_t t = 0; t < tokens; ++t) {
    positions[static_cast<size_t>(t * 2)] = static_cast<double>(t) * 0.04;
    positions[static_cast<size_t>(t * 2 + 1)] = static_cast<double>(t + 1) * 0.04;
  }
  const int64_t ctx = 4;
  std::vector<float> context(
      static_cast<size_t>(ctx * ckpt.params.audio_cross_attention_dim), 0.1F);
  const float sigma = 0.5F;

  vllm::Ltx2ModalityInput ain;
  ain.tokens = tokens;
  ain.context_tokens = ctx;
  ain.latent = latent.data();
  ain.timesteps = timesteps.data();
  ain.sigma = &sigma;
  ain.positions = positions.data();
  ain.context = context.data();

  const vllm::Ltx2DitOutputs out = vllm::Ltx2DitForward(
      vt::Device{}, ckpt.params, ckpt.weights, /*video=*/nullptr, &ain, vt::DType::kF32);
  CHECK(out.video.empty());
  REQUIRE(out.audio.size() == static_cast<size_t>(tokens * ckpt.params.audio_out_channels));
  for (const float v : out.audio) REQUIRE(std::isfinite(v));

  // BOTH null is still refused, which is upstream's own refusal
  // (transformer.py:259-260) rather than a leftover of the old one.
  CHECK_THROWS(vllm::Ltx2DitForward(vt::Device{}, ckpt.params, ckpt.weights, nullptr, nullptr,
                                    vt::DType::kF32));

  // And the STG perturbation MOVES the forward. Without this the flag would be
  // a field nothing reads: a build that plumbed it and never applied it returns
  // the same finite tensor of the same shape.
  vllm::Ltx2DitPerturbation p;
  p.audio_self_attn.assign(static_cast<size_t>(ckpt.params.num_layers), 0);
  p.audio_self_attn[0] = 1;
  const vllm::Ltx2DitOutputs perturbed =
      vllm::Ltx2DitForward(vt::Device{}, ckpt.params, ckpt.weights, nullptr, &ain,
                           vt::DType::kF32, /*cache=*/nullptr, &p);
  REQUIRE(perturbed.audio.size() == out.audio.size());
  bool moved = false;
  for (size_t i = 0; i < out.audio.size(); ++i) {
    if (perturbed.audio[i] != out.audio[i]) moved = true;
  }
  CHECK_MESSAGE(moved, "the STG perturbation changed nothing");

  // A vector of the wrong length is refused rather than indexed defensively.
  vllm::Ltx2DitPerturbation bad;
  bad.audio_self_attn.assign(static_cast<size_t>(ckpt.params.num_layers + 1), 0);
  CHECK_THROWS(vllm::Ltx2DitForward(vt::Device{}, ckpt.params, ckpt.weights, nullptr, &ain,
                                    vt::DType::kF32, /*cache=*/nullptr, &bad));
}

TEST_CASE("ltx2 dit: each CROSS perturbation gates ITS OWN direction and no other (#1092)") {
  // WHY THIS CASE EXISTS, and it is a review mutation result rather than a
  // symmetry a reader would ask for.
  //
  // Three mutations were run against the case above's sibling — the end-to-end
  // `one_stage` guidance case — each built clean and each with its exit status
  // captured directly:
  //
  //   M12  the DiT ignores `video_cross_attn_skip_all`          GREEN, exit 0
  //   M13  the DiT ignores `audio_cross_attn_skip_all`          GREEN, exit 0
  //   M14  the DiT ignores BOTH                                 RED,   exit 1
  //   M15  the DiT SWAPS which flag gates which direction       GREEN, exit 0
  //
  // A build that plumbs the flags and applies NEITHER was caught. A build that
  // applies exactly one, or applies both to the wrong directions, was not — and
  // the half-wrong build renders, on the DEFAULT video arm, whose
  // `modality_scale` is 3.0. The end-to-end assertions cannot separate them:
  // `MaxAbsDiffOf(video_first_modality, video_first_cond)` still fires with one
  // direction applied, because the modality pass still differs from `cond`; and
  // `Ltx2ConditioningTrace::video_modality_skipped_{a2v,v2a}` is assigned from
  // the perturbation struct THE SEAM BUILT (ltx2_denoisers.cpp:315-316), which
  // says what was handed over and nothing about what the DiT did with it.
  //
  // HOW ONE DIRECTION IS ISOLATED AT ALL, on a DiT with more than one block.
  // Within a block the two directions are independent: both read the pre-cross
  // snapshots `vx_pre` / `ax_pre` (transformer.py:333). ACROSS blocks they are
  // not — block 1's V2A reads the video state block 0's A2V wrote — so a
  // both-streams-enabled forward cannot attribute a change to a direction, and
  // this fixture's DiT has two blocks. The separation therefore comes from
  // upstream's own predicates (transformer.py:265-269):
  //
  //   run_a2v = run_vx and audio is present     run_vx = video.ENABLED and ...
  //   run_v2a = run_ax and video is present     run_ax = audio.ENABLED and ...
  //
  // so `audio->enabled = false` with the audio stream still PRESENT runs A2V and
  // not V2A, and `video->enabled = false` runs V2A and not A2V. That is the
  // configuration `ltx2.h` already documents as rendering rather than failing,
  // and it makes each direction observable alone.
  Workspace ws;
  const vllm::SafetensorsFile dit_file = vllm::SafetensorsFile::Open(ws.paths.dit);
  vllm::Ltx2DitLoadOptions options;
  options.widen_to_f32 = true;  // `Ltx2DitForward` is f32 by declaration
  const vllm::Ltx2DitCheckpoint ckpt = vllm::Ltx2LoadDitFromSafetensors(dit_file, options);
  const vllm::Ltx2DitParams& params = ckpt.params;

  const int64_t video_tokens = 2, audio_tokens = 4, context_tokens = 3;
  // VARYING PER TOKEN. A latent whose rows are all equal makes attention return
  // a weighted average of identical values, which is the value projection again,
  // and a perturbation that removes the whole branch would still be measurable —
  // but the SELF-attention case above measured a constant fill turning its own
  // assertion into a false negative, so the same discipline is applied here.
  const auto fill = [](std::vector<float>* v, float base) {
    for (size_t i = 0; i < v->size(); ++i) {
      (*v)[i] = base + 0.01F * static_cast<float>(i % 7) - 0.02F * static_cast<float>(i % 3);
    }
  };
  std::vector<float> video_latent(static_cast<size_t>(video_tokens * params.in_channels));
  std::vector<float> audio_latent(static_cast<size_t>(audio_tokens * params.audio_in_channels));
  fill(&video_latent, 0.25F);
  fill(&audio_latent, 0.30F);
  std::vector<float> video_timesteps(static_cast<size_t>(video_tokens), 0.5F);
  std::vector<float> audio_timesteps(static_cast<size_t>(audio_tokens), 0.5F);
  const float sigma = 0.5F;
  std::vector<double> video_positions(static_cast<size_t>(3 * video_tokens * 2));
  std::vector<double> audio_positions(static_cast<size_t>(audio_tokens * 2));
  for (int64_t d = 0; d < 3; ++d) {
    for (int64_t t = 0; t < video_tokens; ++t) {
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2)] =
          static_cast<double>(t) * 0.04;
      video_positions[static_cast<size_t>((d * video_tokens + t) * 2 + 1)] =
          static_cast<double>(t + 1) * 0.04;
    }
  }
  for (int64_t t = 0; t < audio_tokens; ++t) {
    audio_positions[static_cast<size_t>(t * 2)] = static_cast<double>(t) * 0.04;
    audio_positions[static_cast<size_t>(t * 2 + 1)] = static_cast<double>(t + 1) * 0.04;
  }
  std::vector<float> video_context(static_cast<size_t>(context_tokens * params.cross_attention_dim),
                                   0.05F);
  std::vector<float> audio_context(
      static_cast<size_t>(context_tokens * params.audio_cross_attention_dim), 0.07F);

  struct Streams {
    vllm::Ltx2ModalityInput video;
    vllm::Ltx2ModalityInput audio;
  };
  const auto make = [&](bool video_enabled, bool audio_enabled) {
    Streams s;
    s.video.tokens = video_tokens;
    s.video.context_tokens = context_tokens;
    s.video.enabled = video_enabled;
    s.video.latent = video_latent.data();
    s.video.timesteps = video_timesteps.data();
    s.video.sigma = &sigma;
    s.video.positions = video_positions.data();
    s.video.context = video_context.data();
    s.audio.tokens = audio_tokens;
    s.audio.context_tokens = context_tokens;
    s.audio.enabled = audio_enabled;
    s.audio.latent = audio_latent.data();
    s.audio.timesteps = audio_timesteps.data();
    s.audio.sigma = &sigma;
    s.audio.positions = audio_positions.data();
    s.audio.context = audio_context.data();
    return s;
  };
  const auto run = [&](Streams& io, const vllm::Ltx2DitPerturbation* p) {
    return vllm::Ltx2DitForward(vt::Device{}, params, ckpt.weights, &io.video, &io.audio,
                                vt::DType::kF32, /*cache=*/nullptr, p);
  };
  // `MaxAbsOf` / `MaxAbsDiffOf` are defined further down this file, after this
  // case, so the two measurements are local rather than moved — moving them
  // would churn a block three other cases read.
  const auto max_abs = [](const std::vector<float>& v) {
    double m = 0.0;
    for (const float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
    return m;
  };
  const auto moved = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m > 0.0;
  };

  vllm::Ltx2DitPerturbation skip_a2v;
  skip_a2v.video_cross_attn_skip_all = true;  // SKIP_A2V_CROSS_ATTN
  vllm::Ltx2DitPerturbation skip_v2a;
  skip_v2a.audio_cross_attn_skip_all = true;  // SKIP_V2A_CROSS_ATTN

  SUBCASE("SKIP_A2V_CROSS_ATTN moves the VIDEO stream and SKIP_V2A does not") {
    // `audio->enabled = false`, audio still present: A2V runs, V2A does not.
    auto io = make(/*video_enabled=*/true, /*audio_enabled=*/false);
    const vllm::Ltx2DitOutputs base = run(io, nullptr);
    REQUIRE(base.video.size() == static_cast<size_t>(video_tokens * params.out_channels));
    // NON-VACUITY. A zero output would make both comparisons below trivially
    // equal, so the "did not move" half would pass on a forward that computed
    // nothing at all.
    REQUIRE(max_abs(base.video) > 1e-6);

    const vllm::Ltx2DitOutputs a2v_off = run(io, &skip_a2v);
    REQUIRE(a2v_off.video.size() == base.video.size());
    CHECK_MESSAGE(moved(a2v_off.video, base.video),
                  "`video_cross_attn_skip_all` changed nothing on a forward where A2V is the only "
                  "cross direction running, so the flag reaches no guard "
                  "(transformer.py:335). The isolated-modality pass is then the conditional pass "
                  "again in the audio->video direction, on a recipe whose modality_scale is 3.0");

    const vllm::Ltx2DitOutputs v2a_off = run(io, &skip_v2a);
    REQUIRE(v2a_off.video.size() == base.video.size());
    CHECK_MESSAGE(v2a_off.video == base.video,
                  "`audio_cross_attn_skip_all` moved the VIDEO stream on a forward that runs no "
                  "V2A at all, so the two flags are wired to each other's directions. The flag "
                  "rides on the stream being WRITTEN (transformer.py:335, :367)");
  }

  SUBCASE("SKIP_V2A_CROSS_ATTN moves the AUDIO stream and SKIP_A2V does not") {
    // `video->enabled = false`, video still present: V2A runs, A2V does not.
    auto io = make(/*video_enabled=*/false, /*audio_enabled=*/true);
    const vllm::Ltx2DitOutputs base = run(io, nullptr);
    REQUIRE(base.audio.size() == static_cast<size_t>(audio_tokens * params.audio_out_channels));
    REQUIRE(max_abs(base.audio) > 1e-6);

    const vllm::Ltx2DitOutputs v2a_off = run(io, &skip_v2a);
    REQUIRE(v2a_off.audio.size() == base.audio.size());
    CHECK_MESSAGE(moved(v2a_off.audio, base.audio),
                  "`audio_cross_attn_skip_all` changed nothing on a forward where V2A is the only "
                  "cross direction running, so the flag reaches no guard (transformer.py:367)");

    const vllm::Ltx2DitOutputs a2v_off = run(io, &skip_a2v);
    REQUIRE(a2v_off.audio.size() == base.audio.size());
    CHECK_MESSAGE(a2v_off.audio == base.audio,
                  "`video_cross_attn_skip_all` moved the AUDIO stream on a forward that runs no "
                  "A2V at all, so the two flags are wired to each other's directions");
  }

  SUBCASE("the isolated-modality pass's OWN configuration moves both streams") {
    // Both directions off with both streams enabled, which is what
    // `_guided_denoise` builds for the `mod` pass (denoisers.py:125-138,
    // `blocks=None` on both types). This is the shipped combination; the two
    // subcases above are what separates its halves.
    auto io = make(/*video_enabled=*/true, /*audio_enabled=*/true);
    const vllm::Ltx2DitOutputs base = run(io, nullptr);
    REQUIRE(max_abs(base.video) > 1e-6);
    REQUIRE(max_abs(base.audio) > 1e-6);
    vllm::Ltx2DitPerturbation both;
    both.video_cross_attn_skip_all = true;
    both.audio_cross_attn_skip_all = true;
    const vllm::Ltx2DitOutputs off = run(io, &both);
    CHECK(moved(off.video, base.video));
    CHECK(moved(off.audio, base.audio));
  }
}

TEST_CASE("ltx2 t2a: a SKIPPED step runs no forward and reuses the last prediction") {
  // `should_skip_step` is `step % (skip_step + 1) != 0` (guiders.py:287-291), so
  // `skip_step = 1` skips every ODD step. Upstream then returns
  // `DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)`
  // (utils/denoisers.py:85-91) BEFORE it assembles a pass, so a skipped step
  // costs no DiT forward at all.
  //
  // WHY A COUNT AND NOT A DIGEST. A build that "skipped the guidance" by running
  // the conditional forward and using it — which is the plausible misreading,
  // and what this port did on its first draft — produces a finished waveform of
  // exactly the right length on a different trajectory. Nothing about the output
  // separates the two. The FORWARD COUNT does, and it is the only thing that
  // does.
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = T2aParams(ws.paths);
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);

  vllm::multimodal::VideoGenParams gen = T2aGen(ws.root + "/skip", "a b c");
  gen.steps = 4;  // four sigma intervals: steps 0..3, so 1 and 3 skip
  gen.extras[vllm::multimodal::kLtx2AudioSkipStepExtra] = "1";
  (void)engine->Generate(gen);
  const vllm::multimodal::Ltx2ConditioningTrace skipped = ltx->last_conditioning();

  // Two of the four steps ran, and each ran all three arms.
  CHECK(skipped.t2a_cond_forwards == 2);
  CHECK(skipped.t2a_uncond_forwards == 2);
  CHECK(skipped.t2a_perturbed_forwards == 2);

  // The control: the same request with no skipping runs all four.
  vllm::multimodal::VideoGenParams full = T2aGen(ws.root + "/noskip", "a b c");
  full.steps = 4;
  (void)engine->Generate(full);
  const vllm::multimodal::Ltx2ConditioningTrace every = ltx->last_conditioning();
  CHECK(every.t2a_cond_forwards == 4);
  CHECK(every.t2a_uncond_forwards == 4);
  CHECK(every.t2a_perturbed_forwards == 4);

  // A negative skip is refused rather than taken modulo a non-positive divisor.
  vllm::multimodal::VideoGenParams bad = T2aGen(ws.root + "/badskip", "a b c");
  bad.extras[vllm::multimodal::kLtx2AudioSkipStepExtra] = "-1";
  CHECK_THROWS(engine->Generate(bad));
}

TEST_CASE("ltx2 t2a: the schedule starts at exactly 1.0") {
  // WHY THIS EXISTS, and it is a mutation result rather than a tidiness rule. A
  // mutation that scaled the initial latent by `sigmas[0]` — the thing a reader
  // coming from another flow-matching sampler expects to see — SURVIVED the
  // focused gate at 6 cases / 484 assertions / exit 0. It survived because it is
  // an IDENTITY, not because the gate is blind: `LTX2Scheduler` starts at
  // `linspace(1, 0, steps + 1)[0] == 1`, the shift map sends 1 to exactly 1
  // (schedulers.py:41-45) and the stretch sends it to `1 - (1 - 1)/scale`, again
  // exactly 1 (`:47-55`).
  //
  // Pinning the identity is what turns "a mutation survived" into a checked
  // fact. If upstream ever moves the first sigma off 1, this fires and the two
  // forms stop agreeing.
  // NOT `steps = 1`, and the exclusion is upstream's arithmetic rather than a
  // convenience. At one step the non-zero sigma list is `[1.0]`, so
  // `one_minus_z` is `[0.0]`, `scale_factor = 0 / (1 - terminal)` is 0, and the
  // stretch computes `1 - 0/0` — NaN, on both sides (schedulers.py:49-54).
  // Measured here: `Ltx2SigmaSchedule(1, 0).front()` is `-nan`. Pinning it would
  // be pinning a division by zero as if it were a value; a one-step schedule is
  // a separate question and is recorded in the row spec rather than asserted.
  for (const int64_t steps : {2, 4, 30, 40}) {
    INFO("steps = " << steps);
    const std::vector<float> sigmas = vllm::Ltx2SigmaSchedule(steps, /*tokens=*/0);
    REQUIRE(sigmas.size() == static_cast<size_t>(steps) + 1);
    CHECK(sigmas.front() == 1.0F);
    CHECK(sigmas.back() == 0.0F);
  }
}

TEST_CASE("ltx2 t2a: the one_stage recipes noise their initial latent (#1013)") {
  // A one_stage render used to start from ZEROS: `OneStagePhase` left
  // `noise_scale` at the struct default of 0.0, and `Ltx2GaussianNoise` is
  // `latent + noise_scale * (noise - latent)`, so at 0.0 the state stays exactly
  // as `create_initial_state` wrote it. Upstream's `ModalitySpec.noise_scale`
  // defaults to 1.0 (ltx-pipelines/utils/types.py:110) and
  // `TI2VidOneStagePipeline.__call__` constructs both specs without it
  // (ti2vid_one_stage.py:233-239).
  //
  // Gated on the RECIPE rather than on a render, because the value is what the
  // engine reads and a render's own noise is not separable from it by eye.
  for (const char* version : {"2", "2.3", "2.4", "2.5"}) {
    INFO("version = ", version);
    const vllm::Ltx2PipelineRecipe one = vllm::ResolveLtx2PipelineRecipe("one_stage", version);
    REQUIRE(one.phases.size() == 1);
    CHECK(one.phases[0].noise_scale == 1.0);
    // And the t2a rows inherit it, which is the reason they are built FROM the
    // one_stage recipe rather than beside it.
    const vllm::Ltx2PipelineRecipe t2a =
        vllm::ResolveLtx2PipelineRecipe("t2a_one_stage", version);
    REQUIRE(t2a.phases.size() == 1);
    CHECK(t2a.phases[0].noise_scale == 1.0);
    CHECK(t2a.audio_only);
    CHECK_FALSE(one.audio_only);
  }
}

TEST_CASE("ltx2 t2a: the guider is handed x0 predictions and not raw velocities") {
  // #1039. Upstream hands the denoiser an `X0Model` (ltx-pipelines
  // utils/blocks.py:480-482), so `_guided_denoise` combines DENOISED tensors:
  // `all_v, all_a = transformer(...)` at utils/denoisers.py:188 and
  // `audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)` at `:203`, over an
  // `X0Model.forward` that already applied `to_denoised(latent, v, timesteps)`
  // (ltx-core model/transformer/model.py:590-604, `to_denoised` at
  // ltx-core utils.py:39-52 — `sample - velocity * sigma`).
  //
  // This port combined raw DiT VELOCITIES and converted once afterwards. That is
  // the same function only while `rescale_scale == 0`, because `calculate`'s
  // linear terms are invariant under `x0 = latent - sigma*v`. The rescale branch
  // is not: scaling the x0 by `factor` gives `factor*(latent - sigma*v)`,
  // scaling the velocity gives `latent - sigma*factor*v`, and the two differ by
  // `(factor - 1) * latent`.
  //
  // WHAT THIS CASE ASSERTS, AND WHY IT IS NOT THE RESCALE ARITHMETIC ITSELF.
  // The rescale's numeric consequence is NOT resolvable on the reduced fixture,
  // and that was MEASURED rather than assumed. The first draft of this case
  // computed both candidate step-0 predictions in full — `factor * x0_pred` and
  // `latent - sigma*factor_v*v_pred` — and its own separation guard refused
  // them: this fixture's DiT responds to the conditioning at ~1e-5 of its own
  // output, so `std(cond)/std(pred)` is 1.0 to 1e-5 in BOTH spaces, both
  // factors land within 1e-5 of 1.0, and the two candidates sit 7.6e-07 apart
  // against a span of 3.41. An assertion on that difference would be an
  // assertion about f32 noise, and it would have been GREEN either way.
  //
  // So the rescale's consequence is gated at the seam by the case below, which
  // measures 0.35 relative disagreement at `rescale_scale = 0.7` against
  // 1.5e-07 at 0.0. This case gates what the fixture CAN decide exactly, which
  // is the same defect one step earlier: WHICH TENSOR THE GUIDER WAS HANDED.
  //
  // WHAT MAKES THAT UNREACHABLE BY ACCIDENT — the sibling trap on this campaign
  // was a test whose expectation a zero-filled stub also met.
  // `cond == latent - sigma*velocity` is an equation between three recorded
  // tensors, not a magnitude. It is exact in x0 space; in velocity space `cond`
  // IS the velocity and the residual is `|latent - 2*sigma*velocity|`, i.e. the
  // whole sample. No fixture scale satisfies it by accident, a zeroed velocity
  // collapses it to `cond == latent` and is refused by the lower bound below,
  // and a zeroed `cond` fails it outright.
  //
  // ALL THREE ARMS, AND THE STEP THAT CONSUMES THEM. An earlier draft of this
  // case asserted the equation for the CONDITIONAL pass alone. The default T2A
  // arm runs three forwards per step, so that draft held the file's own
  // "applied to EVERY PASS" claim for one third of the passes, and three
  // mutations survived it at 10 cases / 526 assertions / exit 0:
  //
  //   A1  the PERTURBED pass alone left in velocity space
  //   A2  the UNCONDITIONAL pass alone left in velocity space
  //   R1b `ToDenoised` applied a SECOND time to the guider's output, between the
  //       step-0 recording and the Euler step
  //
  // and a fourth found while closing them:
  //
  //   R1c the same double application placed ABOVE the step-0 recording, so the
  //       recorded `t2a_first_denoised` is itself doubly converted
  //
  // Each renders a different waveform of exactly the right length, through a
  // guider whose `cond` term is impeccable. So the equation is applied to every
  // recorded arm; the guider's own output is reproduced from the three recorded
  // arms through the shipped seam, which is what R1c moves; and the Euler step's
  // input is recovered from the latent it wrote, which is what R1b moves.
  Workspace ws;
  const vllm::multimodal::VideoModelParams mp = T2aParams(ws.paths);

  // The arm this case sits on, pinned as a LOCAL fact before anything is read
  // off a render: `rescale_scale = 0.7` on the 2.3/2.4/2.5 lineage
  // (ltx-pipelines utils/constants.py:63, and the `--audio-rescale-scale`
  // default at utils/args.py:1101-1106).
  const vllm::Ltx2PipelineRecipe t2a_recipe =
      vllm::ResolveLtx2PipelineRecipe("t2a_one_stage", "2.5");
  REQUIRE(t2a_recipe.phases.size() == 1);
  CHECK(t2a_recipe.phases[0].audio_guidance.rescale_scale == 0.7);

  // Through the production entry point — `LoadVideoEngine` then
  // `VideoEngine::Generate`, which is what `vllm_video_generate` calls. Nothing
  // here constructs a guider, a DiT or a modality by hand. `rescale_scale` is
  // the recipe's own 0.7, pinned just above and left untouched by `T2aGen`,
  // which is the field this case turns on. (`T2aGen` does set
  // `audio_stg_blocks`, and that IS a guider field — the two-block fixture
  // cannot take the params table's `[28]` — but it selects WHICH block the
  // perturbed forward skips, not how the arms are combined.)
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  engine->Generate(T2aGen(ws.root + "/x0_space", "a b c"));
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);
  REQUIRE(t.t2a_rendered);

  const size_t n = t.t2a_first_latent.size();
  REQUIRE(n > 0);
  REQUIRE(t.t2a_first_velocity.size() == n);
  REQUIRE(t.t2a_first_cond.size() == n);
  REQUIRE(t.t2a_first_denoised.size() == n);
  REQUIRE(t.t2a_first_next_latent.size() == n);
  const double sigma = t.t2a_first_sigma;
  REQUIRE(sigma > 0.0);

  double latent_span = 0.0;
  for (size_t i = 0; i < n; ++i) {
    latent_span = std::max(latent_span, std::abs(static_cast<double>(t.t2a_first_latent[i])));
  }
  // THE FIXTURE CAN DECIDE THIS AT ALL. The two candidate tensors for every arm
  // are `latent - sigma*velocity` and `velocity`, and they coincide when the
  // sample is zero. A REQUIRE, because nothing below discriminates once it
  // fails. (Its per-arm partner, "the DiT returned no velocity", is next to each
  // arm's own check: a zero velocity makes `to_denoised` the identity for THAT
  // arm alone.)
  REQUIRE_MESSAGE(latent_span > 1e-3,
                  "the step-0 sample is zero, so the two candidate tensors coincide and nothing "
                  "below discriminates");

  // ── the equation, once per guidance pass ──────────────────────────────────
  //
  // EVERY ARM THE RENDER RAN, not only the conditional one. The default T2A
  // guider runs three forwards per step (ltx2_t2a.h item 2), `x0_model` claims
  // to convert EVERY PASS, and a conditional-only assertion holds that claim for
  // one of the three. `t2a_first_uncond` / `t2a_first_perturbed` are empty when
  // the guider did not ask for that arm; this render asks for both, which is
  // asserted rather than assumed — an arm silently skipped would otherwise
  // vacate its own check.
  REQUIRE(t.t2a_uncond_forwards > 0);
  REQUIRE(t.t2a_perturbed_forwards > 0);
  struct Arm {
    const char* name;
    const std::vector<float>& velocity;
    const std::vector<float>& x0;
  };
  const Arm arms[] = {
      {"cond", t.t2a_first_velocity, t.t2a_first_cond},
      {"uncond", t.t2a_first_uncond_velocity, t.t2a_first_uncond},
      {"perturbed", t.t2a_first_perturbed_velocity, t.t2a_first_perturbed},
  };
  for (const Arm& arm : arms) {
    INFO("arm = " << std::string(arm.name));
    REQUIRE(arm.velocity.size() == n);
    REQUIRE(arm.x0.size() == n);

    double velocity_span = 0.0;
    double err_x0 = 0.0;  // |x0 - (latent - sigma*velocity)|  -> 0 in x0 space
    double err_v = 0.0;   // |x0 - velocity|                   -> 0 in velocity space
    for (size_t i = 0; i < n; ++i) {
      const double lat = static_cast<double>(t.t2a_first_latent[i]);
      const double vel = static_cast<double>(arm.velocity[i]);
      const double x0 = static_cast<double>(arm.x0[i]);
      velocity_span = std::max(velocity_span, std::abs(vel));
      err_x0 = std::max(err_x0, std::abs(x0 - (lat - sigma * vel)));
      err_v = std::max(err_v, std::abs(x0 - vel));
    }
    INFO("sigma = " << sigma << "  max|latent| = " << latent_span
                    << "  max|velocity| = " << velocity_span
                    << "  |x0 - (latent - sigma*velocity)| = " << err_x0
                    << "  |x0 - velocity| = " << err_v << "  elements = " << n);

    // 1. `to_denoised` IS NOT THE IDENTITY ON THIS ARM. The second half of the
    //    non-vacuity guard, per arm: a zeroed velocity collapses the equation to
    //    `x0 == latent` and would let a stub satisfy it.
    REQUIRE_MESSAGE(sigma * velocity_span > 1e-6,
                    "the DiT returned no velocity on this arm, so `to_denoised` is the identity "
                    "here and the two candidate tensors coincide");
    // 2. THE GUIDER WAS HANDED THE X0 PREDICTION, exactly — `to_denoised` on the
    //    way out of the forward, which is `X0Model.forward` (model.py:602-603).
    CHECK_MESSAGE(err_x0 <= 1e-5 * latent_span,
                  "the tensor handed to `Ltx2MultiModalGuidance` on this arm is not "
                  "`latent - sigma*velocity`, which is what `X0Model.forward` returns (#1039): "
                  "residual "
                      << err_x0 << " against a tolerance of " << (1e-5 * latent_span));
    // 3. AND IT WAS NOT THE RAW VELOCITY. Said separately from check 2, because a
    //    build handing the guider some THIRD tensor fails 2 and would pass a lone
    //    "not the velocity" check; the pair says which of the two happened.
    CHECK_MESSAGE(err_v > 1e-2 * latent_span,
                  "the tensor handed to `Ltx2MultiModalGuidance` on this arm IS the raw DiT "
                  "velocity, so the guidance is combined in velocity space and converted once "
                  "afterwards (#1039)");
  }

  // ── the guider's output is the guider's output ────────────────────────────
  //
  // The three recorded arms, through the SHIPPED `Ltx2MultiModalGuidance` on the
  // recipe's own params, must reproduce `t2a_first_denoised` bit for bit. This
  // does not gate the guider's arithmetic — `Ltx2Rescale`'s own cases and the
  // seam case below do that — it gates that the pipeline handed the guider these
  // tensors and passed its result on UNTOUCHED. A second `to_denoised` applied
  // to the combination is invisible in every per-arm check above, because it
  // moves nothing the guider was handed.
  //
  // `stg_blocks` is the one guider field `T2aGen` overrides and the one
  // `Ltx2MultiModalGuidance` does not read (it selects the perturbed forward's
  // blocks, not the combination), so the recipe's params are the render's params
  // for this call.
  {
    const std::vector<float> replayed = vllm::Ltx2MultiModalGuidance(
        t2a_recipe.phases[0].audio_guidance, t.t2a_first_cond.data(), t.t2a_first_uncond.data(),
        t.t2a_first_perturbed.data(), /*uncond_modality=*/nullptr, static_cast<int64_t>(n));
    REQUIRE(replayed.size() == n);
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(replayed[i]) -
                                       static_cast<double>(t.t2a_first_denoised[i])));
    }
    INFO("max|replayed guidance - t2a_first_denoised| = " << worst);
    // EXACT, not a tolerance: it is the same function over the same f32 inputs,
    // so any non-zero residual is another operation this pipeline applied.
    CHECK_MESSAGE(worst == 0.0,
                  "`t2a_first_denoised` is not `Ltx2MultiModalGuidance` over the three recorded "
                  "arms, so something else was applied to the guider's result (#1039)");
    // And the combination MOVED what it was handed, so the arms checked above are
    // real inputs to it rather than recorded values beside one.
    CHECK(t.t2a_first_denoised != t.t2a_first_cond);
  }

  // ── and the sampler consumed exactly that ─────────────────────────────────
  //
  // `Ltx2EulerStep` is `x + (x - denoised)/sigma * (sigma_next - sigma)`
  // (ltx2_pipeline.cpp, `EulerDiffusionStep` at ltx-pipelines
  // utils/blocks.py:524-527). Recovering `t2a_first_next_latent` from
  // `t2a_first_denoised` pins WHICH tensor the step was handed. `ToDenoised`
  // applied a second time between the recording and the step leaves every field
  // above untouched and moves only this one.
  //
  // The schedule is re-derived from the shared seam rather than read off the
  // render, and tied to it by the sigma the render recorded.
  {
    const std::vector<float> sigmas = vllm::Ltx2SigmaSchedule(/*steps=*/2, /*tokens=*/0);
    REQUIRE(sigmas.size() == 3);  // `T2aGen` renders two steps
    REQUIRE(static_cast<double>(sigmas[0]) == sigma);
    const double dt = static_cast<double>(sigmas[1]) - static_cast<double>(sigmas[0]);
    REQUIRE_MESSAGE(std::abs(dt) > 1e-3,
                    "the first two sigmas coincide, so the Euler step is the identity and this "
                    "check cannot see what it consumed");
    double worst = 0.0;
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double lat = static_cast<double>(t.t2a_first_latent[i]);
      const double den = static_cast<double>(t.t2a_first_denoised[i]);
      const double expected = lat + (lat - den) / sigma * dt;
      worst = std::max(worst, std::abs(static_cast<double>(t.t2a_first_next_latent[i]) - expected));
      scale = std::max(scale, std::abs(expected));
    }
    INFO("sigma = " << sigma << " -> " << sigmas[1] << "  max|next - Euler(latent, denoised)| = "
                    << worst << "  scale = " << scale);
    REQUIRE_MESSAGE(scale > 1e-3,
                    "the recomputed Euler output is zero, so the residual below bounds nothing");
    CHECK_MESSAGE(worst <= 1e-5 * scale,
                  "the latent `Ltx2EulerStep` wrote is not the step over `t2a_first_denoised`, so "
                  "the sampler was handed some other tensor (#1039): residual "
                      << worst << " against a tolerance of " << (1e-5 * scale));
  }
}

TEST_CASE("ltx2 t2a: rescale_scale 0 is the control because both spaces agree there") {
  // #1039's control, executable rather than asserted in prose. The case above
  // would be testing something OTHER than the defect if it also fired at
  // `rescale_scale = 0`, because `MultiModalGuider.calculate`'s linear terms
  // (guiders.py:261-266) are invariant under `x0 = latent - sigma*v`:
  //
  //   latent - sigma*(c + a(c-u) + b(c-p))  ==  x0c + a(x0c-x0u) + b(x0c-x0p)
  //
  // The rescale at `:268-271` is the only part that is not. This case measures
  // both, on the real seam, with a latent that makes the difference visible.
  const int64_t n = 512;
  std::vector<float> latent(static_cast<size_t>(n));
  std::vector<float> v_cond(static_cast<size_t>(n));
  std::vector<float> v_uncond(static_cast<size_t>(n));
  std::vector<float> v_ptb(static_cast<size_t>(n));
  // Deterministic and NON-CONSTANT. A zero latent erases `(factor - 1) * latent`
  // entirely and a constant one reduces it to a uniform offset; either would
  // make the disagreement below unmeasurable and the control meaningless.
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  const auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>(static_cast<double>(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
  };
  for (int64_t i = 0; i < n; ++i) {
    const size_t j = static_cast<size_t>(i);
    latent[j] = 2.0F * next();
    v_cond[j] = next();
    v_uncond[j] = next();
    v_ptb[j] = next();
  }
  const float sigma = 0.83F;
  std::vector<float> x_cond(static_cast<size_t>(n));
  std::vector<float> x_uncond(static_cast<size_t>(n));
  std::vector<float> x_ptb(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const size_t j = static_cast<size_t>(i);
    x_cond[j] = latent[j] - sigma * v_cond[j];
    x_uncond[j] = latent[j] - sigma * v_uncond[j];
    x_ptb[j] = latent[j] - sigma * v_ptb[j];
  }

  vllm::Ltx2MultiModalGuiderParams params;
  params.cfg_scale = 7.0;  // the T2A defaults (utils/constants.py:58-66)
  params.stg_scale = 1.0;
  params.modality_scale = 1.0;
  params.skip_step = 0;

  const auto compare = [&](double rescale) {
    params.rescale_scale = rescale;
    // Upstream's shape: combine the X0 predictions.
    const std::vector<float> x0_space = vllm::Ltx2MultiModalGuidance(
        params, x_cond.data(), x_uncond.data(), x_ptb.data(), /*uncond_modality=*/nullptr, n);
    // The shape this port shipped: combine the VELOCITIES and convert once after.
    const std::vector<float> v_space = vllm::Ltx2MultiModalGuidance(
        params, v_cond.data(), v_uncond.data(), v_ptb.data(), /*uncond_modality=*/nullptr, n);
    double worst = 0.0;
    double scale = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const size_t j = static_cast<size_t>(i);
      const double converted = static_cast<double>(latent[j]) -
                               static_cast<double>(sigma) * static_cast<double>(v_space[j]);
      worst = std::max(worst, std::abs(static_cast<double>(x0_space[j]) - converted));
      scale = std::max(scale, std::abs(static_cast<double>(x0_space[j])));
    }
    REQUIRE(scale > 1e-3);
    return worst / scale;
  };

  const double at_zero = compare(0.0);
  const double at_default = compare(0.7);
  INFO("relative disagreement: at rescale 0.0 = " << at_zero
                                                  << "  at rescale 0.7 = " << at_default);
  // AT 0.0 THE TWO SPACES ARE THE SAME FUNCTION, to f32 rounding. An assertion
  // that fires here is not about #1039.
  CHECK(at_zero < 1e-4);
  // AT THE SHIPPED 0.7 THEY ARE NOT, by orders of magnitude more. That is the
  // whole of the defect, and it is why the case above can sit on the default.
  CHECK(at_default > 1e-2);
  CHECK(at_default > 100.0 * at_zero);
}

// ─── the HQ arm reaches the res_2s sampler (row LTX25-RES2S-LOOP, #921) ─────
//
// THIS IS THE REACHABILITY CASE, and it is deliberately not a unit test of the
// loop — `test_ltx2_pipeline` already gates the arithmetic against upstream's
// own output. This one enters through the production path a user arrives on:
// `LoadVideoEngine` with the `pipeline_kind` LOAD extra, then
// `VideoEngine::Generate`, which is what `vllm_video_generate`, `ltx2-gen` and
// the server all call. Deleting the `kRes2s` dispatch in `ltx2_video.cpp`'s
// phase loop must red this case; a unit test of `Ltx2Res2sDenoisingLoop` would
// stay green, because it proves the class works and never that anything
// reaches it.
//
// WHAT IT ASSERTS IS A COUNT, because a count is the only thing that separates
// the two samplers. The rendered clip, its shape, its frame count and its
// sample rate are identical whichever one ran.
TEST_CASE("ltx2 video: the HQ pipeline evaluates the DiT twice per step") {
  Workspace ws;

  // `steps` -> forwards, for each arm. The res_2s loop runs two evaluations per
  // step plus one at the terminal sigma the schedule injects (samplers.py:281,
  // :437), and the first-order loop runs one per step. TWO step counts, so an
  // off-by-one cannot satisfy both, and the ratio is close to two rather than a
  // difference of one.
  const auto forwards = [&ws](const std::string& kind, int64_t steps, const std::string& tag) {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    // Stage 1 only. Both recipes' second phase needs the latent spatial
    // upsampler, which the fixture does not carry and which is refused BY NAME
    // in its own case above — that refusal is not what this case is about.
    mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + tag);
    gen.steps = steps;
    // `one_stage` resolves `stg_blocks = [28]` (constants.py:86-87) and this
    // fixture's DiT has two blocks, so its PERTURBED pass is refused by name
    // unless the request names a block that exists. The HQ preset ships
    // `stg_blocks = []` beside `stg_scale = 0.0` (constants.py:105, :113) and
    // asks for no perturbed pass at all, so it needs no override — and giving it
    // one would put a request override on the arm this case is measuring.
    if (kind == "one_stage") OneStageFixtureGuidance(&gen);
    (void)engine->Generate(gen);
    return ltx2->last_conditioning();
  };

  const vllm::multimodal::Ltx2ConditioningTrace hq3 = forwards("res2s_two_stage", 3, "hq3");
  const vllm::multimodal::Ltx2ConditioningTrace hq5 = forwards("res2s_two_stage", 5, "hq5");
  const vllm::multimodal::Ltx2ConditioningTrace euler3 = forwards("one_stage", 3, "e3");
  const vllm::multimodal::Ltx2ConditioningTrace euler5 = forwards("one_stage", 5, "e5");

  INFO("res2s: 3 steps -> " << hq3.dit_evaluations << " forwards, 5 steps -> "
                            << hq5.dit_evaluations << "; euler: 3 -> "
                            << euler3.dit_evaluations << ", 5 -> " << euler5.dit_evaluations);
  // 2 * steps + 1. The schedule `Ltx2SigmaSchedule` builds terminates at exactly
  // 0 (gated in test_ltx2_pipeline), so the terminal evaluation always happens.
  CHECK(hq3.dit_evaluations == 7);
  CHECK(hq5.dit_evaluations == 11);
  // ...against the first-order arm on the SAME request. Both numbers are read
  // off a real render rather than one being computed from the other, so the
  // comparison cannot be satisfied by both arms sharing a defect.
  CHECK(euler3.dit_evaluations == 3);
  CHECK(euler5.dit_evaluations == 5);
  CHECK(hq3.dit_evaluations > 2 * euler3.dit_evaluations);
  CHECK(hq5.dit_evaluations > 2 * euler5.dit_evaluations);
  // A ZERO WOULD ALSO BE "not equal to the Euler count", and zero is what a
  // build that never ran the loop reports. Ruled out explicitly.
  CHECK(euler3.dit_evaluations > 0);

  // THE BONG REFINEMENT IS REACHED ON THE PRODUCTION SCHEDULE, not only on the
  // hand-built fixtures in test_ltx2_pipeline. It changes the latent without
  // changing how many forwards ran, so the counter above is blind to it and this
  // is the only place a real render says it happened.
  CHECK(hq3.res2s_bong_steps > 0);
  CHECK(hq5.res2s_bong_steps > 0);
  // ...and never on a first-order arm, which has no anchor to refine.
  CHECK(euler3.res2s_bong_steps == 0);
  CHECK(euler5.res2s_bong_steps == 0);

  // THE NOISE THE ENGINE HANDED THE LOOP WAS NORMALIZED. `_get_new_noise`
  // (samplers.py:164-170) is what the res_2s loop takes, against the ancestral
  // loop's un-normalized `_get_plain_noise` (:155-157) ten lines away. That the
  // FUNCTION normalizes is gated in test_ltx2_pipeline; that this engine calls
  // it is a different claim, and MEASURED: with the hook handing over its raw
  // draw instead, every assertion above stayed green.
  //
  // 1e-9 is unreachable for a raw Gaussian draw, whose sample moments miss by
  // O(1/sqrt(n)) on any latent this fixture builds, and trivial for a
  // normalized one, which is exact to rounding.
  INFO("res2s noise moment error = " << hq3.res2s_noise_moment_error);
  CHECK(hq3.res2s_noise_moment_error < 1e-9);
  CHECK(hq5.res2s_noise_moment_error < 1e-9);
  // Zero — not "small" — on an arm that runs no res_2s draw at all, so the
  // field cannot read as satisfied by never having been written.
  CHECK(euler3.res2s_noise_moment_error == 0.0);

  // BOTH ARMS BUILT THEIR SCHEDULE THE SAME WAY, which is what lets the two
  // counts be compared at all: each recipe leaves stage 1's sigmas empty and
  // therefore derives them from `steps` through `Ltx2SigmaSchedule`, so the
  // difference between 7 and 3 is the SAMPLER and not a different schedule.
  // Their token counts differ — the HQ stage 1 halves the request
  // (ti2vid_two_stages_hq.py:238-243) and `one_stage` does not — which is why
  // the counts above are asserted absolutely rather than only as a ratio.
  CHECK(hq3.schedule_tokens > 0);
  CHECK(euler3.schedule_tokens > 0);
  CHECK(hq3.video_tokens < euler3.video_tokens);
}

// ─── the HQ arm is GUIDED, and the evaluation count cannot see that ─────────
//
// THIS IS A SEPARATE CASE FROM THE ONE ABOVE BECAUSE IT IS A SEPARATE DEFECT,
// and the one above is blind to it. A render's DiT work is
// `evaluations x forwards-per-evaluation`. The sampler decides the first factor
// and the denoiser decides the second, and `dit_evaluations` — the whole
// instrument of the case above — is exactly the first factor. Route the res_2s
// loop around a bare `Ltx2DitForward` instead of `Ltx2GuidedDenoise` and
// `dit_evaluations` stays at 2n+1, `res2s_bong_steps` stays right, the eval
// sigmas stay right, the clip keeps its shape, frame count, sample rate and file
// size, and the preset renders at cfg 1.0 where upstream tuned it at 3.0.
//
// Upstream's HQ stage 1 runs a `GuidedDenoiser` (ti2vid_two_stages_hq.py:271-281)
// built from `LTX_2_3_HQ_PARAMS` — cfg 3.0 video / 7.0 audio, rescale 0.45,
// modality 3.0, stg 0.0, stg_blocks [] (utils/constants.py:99-114). So each of
// stage 1's evaluations is THREE transformer forwards: `cond` always
// (denoisers.py:100), `uncond` because cfg != 1.0 (:102-109, guiders.py:275-277)
// and `mod` because modality_scale != 1.0 (:121-137, guiders.py:283-285). No
// `ptb`, because stg_scale is 0.0.
TEST_CASE("ltx2 video: the HQ pipeline stage 1 is GUIDED, three forwards per evaluation") {
  Workspace ws;

  const auto render = [&ws](const std::string& kind, int64_t steps, const std::string& tag) {
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    // Stage 1 only, for the reason the case above gives: the second phase needs
    // the latent spatial upsampler the fixture does not carry.
    mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    REQUIRE(engine != nullptr);
    auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx2 != nullptr);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/" + tag);
    gen.steps = steps;
    // The HQ preset ships `stg_blocks = []` on both modalities beside
    // `stg_scale = 0.0`, so unlike `one_stage` it needs no block override to run
    // on a reduced-block fixture — the perturbed pass is not requested at all.
    (void)engine->Generate(gen);
    return ltx2->last_conditioning();
  };

  const vllm::multimodal::Ltx2ConditioningTrace hq3 = render("res2s_two_stage", 3, "ghq3");
  const vllm::multimodal::Ltx2ConditioningTrace hq5 = render("res2s_two_stage", 5, "ghq5");

  // THE GUIDER THE PHASE RESOLVED, so a recipe that quietly lost `LTX_2_3_HQ_PARAMS`
  // fails here rather than rendering at the defaults.
  CHECK(hq3.video_guidance_cfg_scale == 3.0);
  CHECK(hq3.video_guidance_stg_scale == 0.0);
  CHECK(hq3.video_guidance_rescale_scale == 0.45);
  CHECK(hq3.video_guidance_modality_scale == 3.0);
  // ...and the seam RAN, recorded at the call rather than copied from the params
  // above. `RecordFirstGuidedStep` reads `pass_ran`, which the denoiser sets when
  // it issues the forward.
  REQUIRE(hq3.video_guided);
  CHECK(hq3.video_cond_forwards == 1);
  CHECK(hq3.video_uncond_forwards == 1);
  CHECK(hq3.video_perturbed_forwards == 0);
  CHECK(hq3.video_modality_forwards == 1);

  // THE COUNT THAT MOVES WHEN GUIDANCE IS DROPPED, and the one that does not.
  //
  // `dit_evaluations` is 2n+1 whether or not the arm is guided; `dit_forwards`
  // is three times that when it is and equal to it when it is not. Both are
  // asserted EXACTLY and on TWO step counts, so neither an off-by-one nor a
  // constant factor can satisfy both.
  INFO("hq3: evaluations = " << hq3.dit_evaluations << " forwards = " << hq3.dit_forwards);
  INFO("hq5: evaluations = " << hq5.dit_evaluations << " forwards = " << hq5.dit_forwards);
  CHECK(hq3.dit_evaluations == 7);
  CHECK(hq5.dit_evaluations == 11);
  CHECK(hq3.dit_forwards == 21);
  CHECK(hq5.dit_forwards == 33);
  // The relation, derived rather than only read off the two numbers, so a change
  // to one of the four constants above cannot be absorbed by changing another.
  CHECK(hq3.dit_forwards == 3 * hq3.dit_evaluations);
  CHECK(hq5.dit_forwards == 3 * hq5.dit_evaluations);
  // AN UNGUIDED ARM IS EXACTLY `forwards == evaluations`, which is the mutation
  // this case exists for. Stated as its own assertion rather than left implicit
  // in the multiplier, because that is the sentence the RED has to print.
  CHECK(hq3.dit_forwards != hq3.dit_evaluations);

  // ...against the arm whose guidance this tree already gated. `one_stage`
  // resolves cfg 3.0, stg 1.0 AND modality 3.0, so it runs all FOUR passes and
  // the two arms differ in the pass SET as well as in the sampler. Read off a
  // real render rather than computed from the HQ numbers.
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/g1s");
  gen.steps = 3;
  OneStageFixtureGuidance(&gen);
  (void)engine->Generate(gen);
  const vllm::multimodal::Ltx2ConditioningTrace euler3 = ltx2->last_conditioning();
  CHECK(euler3.dit_evaluations == 3);
  CHECK(euler3.dit_forwards == 12);
  CHECK(euler3.video_perturbed_forwards == 1);
  // The HQ arm runs FEWER forwards per evaluation and MORE evaluations, so
  // neither counter on its own separates the two arms and both are needed.
  CHECK(hq3.dit_evaluations > euler3.dit_evaluations);
  CHECK(hq3.dit_forwards > euler3.dit_forwards);
}

// ─── the SUBSTEP evaluation converts against the MIDPOINT it was handed ─────
//
// The res_2s second evaluation runs over `x_mid` (samplers.py:369-378), a state
// that never becomes the stream's own latent. Everywhere else in `ltx2_video.cpp`
// "the latent" and "the latent this evaluation was handed" are the same tensor,
// which is what makes `ToDenoised(video.latent, ...)` an easy write here and an
// invisible one: MEASURED, with that substitution in place this whole file
// stayed GREEN at 74 cases and 2234 assertions. The clip, the evaluation count,
// the forward count, the eval sigmas and the bong count are all blind to it, and
// the loop's own arithmetic is gated with a FIXTURE denoiser that never performs
// this conversion at all.
TEST_CASE("ltx2 video: the res_2s SUBSTEP converts x0 against the midpoint, not the state") {
  Workspace ws;
  vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "res2s_two_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("res2s_two_stage");
  mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(mp);
  REQUIRE(engine != nullptr);
  auto* ltx2 = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx2 != nullptr);
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/sub");
  gen.steps = 3;
  (void)engine->Generate(gen);
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx2->last_conditioning();

  // The substep ran at all, and it ran on the res_2s arm.
  REQUIRE(t.res2s_substep_latent.size() == t.video_first_latent.size());
  REQUIRE(!t.res2s_substep_latent.empty());
  REQUIRE(t.res2s_substep_cond.size() == t.res2s_substep_latent.size());
  REQUIRE(t.res2s_substep_cond_velocity.size() == t.res2s_substep_latent.size());
  // ONE TIMESTEP PER TOKEN, not per element: `timesteps_from_mask` is per token
  // and `to_denoised` broadcasts it across the token's whole row. A conditioned
  // token sits at timestep 0, which is why the scalar sigma cannot stand in.
  const size_t tokens = t.res2s_substep_timesteps.size();
  REQUIRE(tokens > 0);
  REQUIRE(t.res2s_substep_latent.size() % tokens == 0);
  const size_t width = t.res2s_substep_latent.size() / tokens;

  // NON-VACUITY, twice, because both zeros make the assertion below trivially
  // true. The midpoint MOVED — `x_mid = x_anchor + h * a21 * eps_1`
  // (samplers.py:322) is not the anchor — so a build that evaluated the substep
  // over the unmoved state would satisfy the invariant against either tensor and
  // this case would prove nothing.
  const auto abs_max = [](const std::vector<float>& v) {
    double m = 0.0;
    for (const float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
    return m;
  };
  const auto abs_diff = [](const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m;
  };
  const double moved = abs_diff(t.res2s_substep_latent, t.video_first_latent);
  INFO("midpoint moved by " << moved);
  REQUIRE(moved > 1e-6);
  REQUIRE(abs_max(t.res2s_substep_cond_velocity) > 1e-6);
  // ...and the substep sigma is the geometric mean, not the step's own
  // (samplers.py:314-315), so this really is the second evaluation.
  CHECK(t.res2s_substep_sigma < t.video_first_sigma);
  CHECK(t.res2s_substep_sigma > 0.0);

  // THE INVARIANT: `x0 == latent - timesteps * velocity` (model.py:590-604),
  // over the latent THIS evaluation was handed. An equation between four
  // recorded vectors, not a magnitude, so no fixture scale satisfies it by
  // accident. With the conversion reading `video.latent` the residual is
  // exactly `video_first_latent - res2s_substep_latent`, whose max is the
  // `moved` printed above.
  double worst = 0.0;
  for (size_t token = 0; token < tokens; ++token) {
    const double sigma = static_cast<double>(t.res2s_substep_timesteps[token]);
    for (size_t w = 0; w < width; ++w) {
      const size_t i = token * width + w;
      const double want = static_cast<double>(t.res2s_substep_latent[i]) -
                          sigma * static_cast<double>(t.res2s_substep_cond_velocity[i]);
      worst = std::max(worst, std::abs(static_cast<double>(t.res2s_substep_cond[i]) - want));
    }
  }
  INFO("substep |x0 - (latent - t*v)| = " << worst << " against a midpoint that moved " << moved);
  CHECK(worst < 1e-5);
  // And the residual is orders of magnitude below the displacement it would be
  // if the wrong latent had been used, so the tolerance above cannot be
  // absorbing the defect.
  CHECK(worst < 0.01 * moved);
}

// ─── row LTX25-GUIDED-VIDEO (#1092): the guided VIDEO denoiser ──────────────
//
// The video denoise loop ran ONE unguided forward per step and applied
// `ToDenoised` to it, while every recipe resolved a video guider that nothing
// read. These cases gate the four passes upstream's `_guided_denoise` assembles
// (ltx-pipelines utils/denoisers.py:97-137 @ fd4ded7f) and, for each of them,
// WHICH SPACE it was combined in.
//
// They enter through the production entry point — `LoadVideoEngine` then
// `VideoEngine::Generate`, which is what `vllm_video_generate` calls — on
// `pipeline_kind = one_stage`, whose OWN recipe resolves `cfg_scale = 3.0`,
// `stg_scale = 1.0`, `rescale_scale = 0.7` and `modality_scale = 3.0`. Nothing
// below constructs a guider, a DiT, a modality or a perturbation by hand.

namespace {

// `one_stage` on the shipped fixture, guided by its own recipe. The only guider
// field overridden is the STG block list, and `OneStageFixtureGuidance` says why.
vllm::multimodal::VideoModelParams OneStageParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp = FixtureParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
  return mp;
}

vllm::multimodal::VideoGenParams OneStageGen(const std::string& out_dir) {
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.steps = 2;  // two sigma intervals is enough to exercise the loop
  OneStageFixtureGuidance(&gen);
  return gen;
}

double MaxAbsOf(const std::vector<float>& v) {
  double m = 0.0;
  for (const float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
  return m;
}

double MaxAbsDiffOf(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

}  // namespace

TEST_CASE("ltx2 one_stage: all four guidance arms are combined in X0 space (#1092)") {
  // THE DEFECT THIS CASE EXISTS FOR, in two layers.
  //
  // The outer one is that the video path ran no guidance at all. That is caught
  // by the pass counts below, which are read off the seam's own record of which
  // forwards it issued rather than inferred from an output.
  //
  // The inner one is #1039's, on a path that never had it: `MultiModalGuider`
  // combines DENOISED predictions, because `DiffusionStage` hands the loop an
  // `X0Model` (utils/blocks.py:480-482) and not the raw velocity model. The
  // guider's LINEAR terms are invariant under `x0 = latent - sigma*v`, so the
  // difference is entirely in the rescale at guiders.py:268-271 — and
  // `rescale_scale` is 0.7 on this recipe, which is the DEFAULT arm.
  //
  // WHAT MAKES THAT UNREACHABLE BY ACCIDENT. `cond == latent - sigma*velocity` is
  // an EQUATION between three recorded tensors, not a magnitude. It is exact in
  // x0 space; in velocity space `cond` IS the velocity and the residual is the
  // whole sample. No fixture scale satisfies it by accident, a zeroed velocity
  // collapses it to `cond == latent` and is refused by the lower bound below,
  // and a zeroed `cond` fails it outright.
  //
  // ALL FOUR ARMS. #1039's first gate asserted the equation for the conditional
  // pass alone; the T2A arm runs three forwards and three mutations survived
  // that draft. This arm runs FOUR.
  Workspace ws;

  // The arm this case sits on, pinned as a LOCAL fact before anything is read off
  // a render. `rescale_scale = 0.7` on the 2.4/2.5 lineage (ltx-pipelines
  // utils/constants.py:53 video / :63 audio, reached through `_PARAMS_SINCE_VERSION` at
  // :130-133).
  const vllm::Ltx2PipelineRecipe recipe = vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5");
  REQUIRE(recipe.phases.size() == 1);
  const vllm::Ltx2MultiModalGuiderParams row = recipe.phases[0].video_guidance;
  CHECK(row.cfg_scale == 3.0);
  CHECK(row.stg_scale == 1.0);
  CHECK(row.rescale_scale == 0.7);
  CHECK(row.modality_scale == 3.0);

  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(OneStageParams(ws.paths));
  REQUIRE(engine != nullptr);
  (void)engine->Generate(OneStageGen(ws.root + "/guided_x0"));
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);

  // ── the render ran the guidance its recipe asked for ──────────────────────
  //
  // Counts, not tensors. An arm silently skipped changes a counter and changes no
  // output shape, no frame count and no sample rate.
  REQUIRE_MESSAGE(t.video_guided, "the video denoise did not go through the guided seam at all");
  CHECK(t.video_cond_forwards == 1);
  CHECK_MESSAGE(t.video_uncond_forwards == 1,
                "no unconditional forward ran, so `(cfg_scale - 1) * (cond - uncond)` is "
                "identically zero on a recipe whose cfg_scale is 3.0");
  CHECK_MESSAGE(t.video_perturbed_forwards == 1,
                "no perturbed forward ran, so `stg_scale * (cond - perturbed)` is identically "
                "zero on a recipe whose stg_scale is 1.0");
  CHECK_MESSAGE(t.video_modality_forwards == 1,
                "no isolated-modality forward ran, so `(modality_scale - 1) * (cond - mod)` is "
                "identically zero on a recipe whose modality_scale is 3.0");
  // The guidance the ENGINE resolved, which is what the replay below uses. A
  // build that resolved different scales fails the replay rather than agreeing
  // with itself.
  CHECK(t.video_guidance_cfg_scale == row.cfg_scale);
  CHECK(t.video_guidance_stg_scale == row.stg_scale);
  CHECK(t.video_guidance_rescale_scale == row.rescale_scale);
  CHECK(t.video_guidance_modality_scale == row.modality_scale);

  // The perturbations REACHED the DiT, read off the mask that was handed over
  // rather than off the guider params. A config that is BUILT and not HANDED
  // OVER leaves the params untouched and renders.
  CHECK(t.video_perturbed_blocks == std::vector<int64_t>{1});
  CHECK(t.video_audio_perturbed_blocks == std::vector<int64_t>{1});
  // WHAT THESE TWO MEASURE, said exactly, because the message they used to carry
  // claimed more. `video_modality_skipped_{a2v,v2a}` is assigned from the
  // `Ltx2DitPerturbation` THE SEAM BUILT and handed over
  // (ltx2_denoisers.cpp:315-316), so it says the seam asked for both directions
  // — which is `blocks=None` on both types (denoisers.py:125-138) — and says
  // NOTHING about what the DiT did with the request. What the DiT does with each
  // flag is gated separately and per direction by
  // "ltx2 dit: each CROSS perturbation gates ITS OWN direction and no other",
  // which exists because mutations that applied exactly one direction, or
  // swapped the two, survived this case.
  CHECK_MESSAGE(t.video_modality_skipped_a2v,
                "the seam built the isolated-modality pass WITHOUT asking for SKIP_A2V_CROSS_ATTN, "
                "so that pass is the conditional pass again in the audio->video direction "
                "(denoisers.py:125-138)");
  CHECK_MESSAGE(t.video_modality_skipped_v2a,
                "the seam built the isolated-modality pass WITHOUT asking for "
                "SKIP_V2A_CROSS_ATTN");

  const size_t n = t.video_first_latent.size();
  REQUIRE(n > 0);
  const size_t tokens = t.video_first_timesteps.size();
  REQUIRE(tokens > 0);
  const size_t width = n / tokens;
  REQUIRE(width * tokens == n);

  // THE FIXTURE CAN DECIDE THIS AT ALL. The two candidate tensors for every arm
  // are `latent - sigma*velocity` and `velocity`, and they coincide when the
  // sample is zero. A REQUIRE, because nothing below discriminates once it fails.
  const double latent_span = MaxAbsOf(t.video_first_latent);
  REQUIRE_MESSAGE(latent_span > 1e-3,
                  "the step-0 sample is zero, so the two candidate tensors coincide and nothing "
                  "below discriminates");

  // ── the equation, once per guidance pass ──────────────────────────────────
  struct Arm {
    const char* name;
    const std::vector<float>& velocity;
    const std::vector<float>& x0;
  };
  const Arm arms[] = {
      {"cond", t.video_first_cond_velocity, t.video_first_cond},
      {"uncond", t.video_first_uncond_velocity, t.video_first_uncond},
      {"perturbed", t.video_first_perturbed_velocity, t.video_first_perturbed},
      {"modality", t.video_first_modality_velocity, t.video_first_modality},
  };
  for (const Arm& arm : arms) {
    INFO("arm = " << std::string(arm.name));
    REQUIRE(arm.velocity.size() == n);
    REQUIRE(arm.x0.size() == n);

    double velocity_span = 0.0;
    double sigma_velocity_span = 0.0;
    double err_x0 = 0.0;  // |x0 - (latent - sigma*velocity)|  -> 0 in x0 space
    double err_v = 0.0;   // |x0 - velocity|                   -> 0 in velocity space
    for (size_t token = 0; token < tokens; ++token) {
      // The PER-TOKEN sigma, which is what `X0Model.forward` uses
      // (model.py:601-604 passes `video.timesteps`). Using the schedule scalar
      // here would pass on a build that used it too, and that build re-noises
      // every conditioned token.
      const double sigma = static_cast<double>(t.video_first_timesteps[token]);
      for (size_t c = 0; c < width; ++c) {
        const size_t i = token * width + c;
        const double lat = static_cast<double>(t.video_first_latent[i]);
        const double vel = static_cast<double>(arm.velocity[i]);
        const double x0 = static_cast<double>(arm.x0[i]);
        velocity_span = std::max(velocity_span, std::abs(vel));
        sigma_velocity_span = std::max(sigma_velocity_span, std::abs(sigma * vel));
        err_x0 = std::max(err_x0, std::abs(x0 - (lat - sigma * vel)));
        err_v = std::max(err_v, std::abs(x0 - vel));
      }
    }
    INFO("max|latent| = " << latent_span << "  max|velocity| = " << velocity_span
                          << "  max|sigma*velocity| = " << sigma_velocity_span
                          << "  |x0 - (latent - sigma*velocity)| = " << err_x0
                          << "  |x0 - velocity| = " << err_v << "  elements = " << n);

    // 1. `to_denoised` IS NOT THE IDENTITY ON THIS ARM. The second half of the
    //    non-vacuity guard, per arm: a zeroed velocity collapses the equation to
    //    `x0 == latent` and would let a stub satisfy it.
    REQUIRE_MESSAGE(sigma_velocity_span > 1e-6,
                    "the DiT returned no velocity on this arm, so `to_denoised` is the identity "
                    "here and the two candidate tensors coincide");
    // 2. THE GUIDER WAS HANDED THE X0 PREDICTION, exactly.
    CHECK_MESSAGE(err_x0 <= 1e-5 * latent_span,
                  "the tensor handed to `Ltx2MultiModalGuidance` on this arm is not "
                  "`latent - sigma*velocity`, which is what `X0Model.forward` returns "
                  "(model.py:590-604, #1039): residual "
                      << err_x0 << " against a tolerance of " << (1e-5 * latent_span));
    // 3. AND IT WAS NOT THE RAW VELOCITY. Said separately from check 2, because a
    //    build handing the guider some THIRD tensor fails 2 and would pass a lone
    //    "not the velocity" check; the pair says which of the two happened.
    CHECK_MESSAGE(err_v > 1e-2 * latent_span,
                  "the tensor handed to `Ltx2MultiModalGuidance` on this arm IS the raw DiT "
                  "velocity, so the guidance is combined in velocity space and converted once "
                  "afterwards (#1039)");
  }

  // ── each arm is a DIFFERENT forward ───────────────────────────────────────
  //
  // Without these, an arm whose CONTEXT or PERTURBATION never reached the DiT
  // satisfies every check above: it is a perfectly converted x0 prediction of the
  // conditional pass, and its guidance term is exactly zero.
  CHECK_MESSAGE(MaxAbsDiffOf(t.video_first_uncond, t.video_first_cond) > 1e-6 * latent_span,
                "the unconditional pass returned the conditional pass's own tensor, so the "
                "negative context did not reach the forward");
  CHECK_MESSAGE(MaxAbsDiffOf(t.video_first_perturbed, t.video_first_cond) > 1e-6 * latent_span,
                "the perturbed pass returned the conditional pass's own tensor, so the "
                "self-attention perturbation did not reach the forward");
  CHECK_MESSAGE(MaxAbsDiffOf(t.video_first_modality, t.video_first_cond) > 1e-6 * latent_span,
                "the isolated-modality pass returned the conditional pass's own tensor, so the "
                "cross-attention perturbation did not reach the forward (transformer.py:335,367)");

  // ── the guider's output is the guider's output ────────────────────────────
  //
  // The four recorded arms, through the SHIPPED `Ltx2MultiModalGuidance` on the
  // recipe's own params, must reproduce `video_first_denoised` bit for bit. This
  // does not gate the guider's arithmetic — the seam case below does that — it
  // gates that the pipeline handed the guider these tensors and passed its result
  // on UNTOUCHED. A second `to_denoised` applied to the combination is invisible
  // in every per-arm check above, because it moves nothing the guider was handed.
  {
    const std::vector<float> replayed = vllm::Ltx2MultiModalGuidance(
        row, t.video_first_cond.data(), t.video_first_uncond.data(),
        t.video_first_perturbed.data(), t.video_first_modality.data(), static_cast<int64_t>(n));
    REQUIRE(replayed.size() == n);
    const double worst = MaxAbsDiffOf(replayed, t.video_first_denoised);
    INFO("max|replayed guidance - video_first_denoised| = " << worst);
    // EXACT, not a tolerance: it is the same function over the same f32 inputs,
    // so any non-zero residual is another operation this pipeline applied.
    CHECK_MESSAGE(worst == 0.0,
                  "`video_first_denoised` is not `Ltx2MultiModalGuidance` over the four recorded "
                  "arms, so something else was applied to the guider's result (#1039)");
    // And the combination MOVED what it was handed, so the arms checked above are
    // real inputs to it rather than recorded values beside one.
    CHECK(t.video_first_denoised != t.video_first_cond);
  }

  // ── the same combination, over arms REBUILT FROM THE RAW VELOCITIES ───────
  //
  // WHY THIS IS NOT THE PREVIOUS CHECK AGAIN. The replay above is fed the arms
  // the seam recorded, so anything applied to EVERY arm on the way out of the
  // forward is invisible to it: the replay and the pipeline agree because they
  // agree about the same altered inputs. `post_process_latent` applied per arm
  // instead of once to the guider's result is exactly that shape, and it is not
  // hypothetical -- upstream applies it in the LOOP (utils/samplers.py:35), one
  // level above the denoiser, and applying it a level lower is the obvious
  // simplification.
  //
  // It is also invisible to the per-arm invariant, and that took working out.
  // `post_process_latent` is `x*mask + clean*(1-mask)`, so it only moves tokens
  // whose denoise mask is 0 -- and on such a token the schedule sigma is 0 too
  // (`timesteps_from_mask`, utils/helpers.py:494-503), so the invariant reads
  // `x0 == latent`, and a conditioned token's `latent` IS its clean value. The
  // two placements therefore agree token by token and differ only through
  // `cond.std()` and `pred.std()`, which the rescale computes over the WHOLE
  // tensor and which change for every element at once.
  //
  // Rebuilding the arms from `latent` and the raw velocities is independent of
  // anything applied to the arms, so it sees that. It is exact rather than
  // approximate because it repeats `ToDenoised`'s own arithmetic: the subtraction
  // in double, the store in f32.
  {
    const auto rebuild = [&](const std::vector<float>& velocity) {
      std::vector<float> out(n);
      for (size_t token = 0; token < tokens; ++token) {
        const double sigma = static_cast<double>(t.video_first_timesteps[token]);
        for (size_t c = 0; c < width; ++c) {
          const size_t i = token * width + c;
          out[i] = static_cast<float>(static_cast<double>(t.video_first_latent[i]) -
                                      sigma * static_cast<double>(velocity[i]));
        }
      }
      return out;
    };
    const std::vector<float> c = rebuild(t.video_first_cond_velocity);
    const std::vector<float> u = rebuild(t.video_first_uncond_velocity);
    const std::vector<float> p = rebuild(t.video_first_perturbed_velocity);
    const std::vector<float> m = rebuild(t.video_first_modality_velocity);
    const std::vector<float> replayed = vllm::Ltx2MultiModalGuidance(
        row, c.data(), u.data(), p.data(), m.data(), static_cast<int64_t>(n));
    const double worst = MaxAbsDiffOf(replayed, t.video_first_denoised);
    INFO("max|guidance over rebuilt arms - video_first_denoised| = " << worst);
    CHECK_MESSAGE(worst == 0.0,
                  "the guider's result is not `Ltx2MultiModalGuidance` over `latent - sigma*v` "
                  "for the four RAW velocities, so something was applied to the arms between the "
                  "forward and the combination");
  }

  // ── `post_process_latent` came AFTER the guider, and the sampler consumed
  //    exactly what it produced ─────────────────────────────────────────────
  //
  // `_step_state` applies `post_process_latent(denoised, ...)` to the DENOISER's
  // result (utils/samplers.py:35), never per arm inside it. On this render no
  // token is conditioned, so the two tensors coincide — asserted rather than
  // assumed, because it is what makes the Euler recovery below a statement about
  // `video_first_denoised`.
  REQUIRE(t.video_first_stepper_input.size() == n);
  CHECK(t.video_first_stepper_input == t.video_first_denoised);

  {
    // `Ltx2EulerStep` is `x + (x - denoised)/sigma * (sigma_next - sigma)`
    // (`EulerDiffusionStep`, ltx-pipelines utils/blocks.py:524-527). Recovering
    // `video_first_next_latent` from `video_first_stepper_input` pins WHICH
    // tensor the step was handed: `ToDenoised` applied a second time between the
    // recording and the step leaves every field above untouched and moves only
    // this one.
    //
    // The schedule is re-derived from the shared seam rather than read off the
    // render, and tied to it by the sigma the render recorded.
    REQUIRE(t.schedule_tokens > 0);
    const std::vector<float> sigmas = vllm::Ltx2SigmaSchedule(/*steps=*/2, t.schedule_tokens);
    REQUIRE(sigmas.size() == 3);
    const double sigma = t.video_first_sigma;
    REQUIRE(sigma > 0.0);
    REQUIRE(static_cast<double>(sigmas[0]) == sigma);
    const double dt = static_cast<double>(sigmas[1]) - static_cast<double>(sigmas[0]);
    REQUIRE_MESSAGE(std::abs(dt) > 1e-3,
                    "the first two sigmas coincide, so the Euler step is the identity and this "
                    "check cannot see what it consumed");
    REQUIRE(t.video_first_next_latent.size() == n);
    double worst = 0.0;
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double lat = static_cast<double>(t.video_first_latent[i]);
      const double den = static_cast<double>(t.video_first_stepper_input[i]);
      const double expected = lat + (lat - den) / sigma * dt;
      worst =
          std::max(worst, std::abs(static_cast<double>(t.video_first_next_latent[i]) - expected));
      scale = std::max(scale, std::abs(expected));
    }
    INFO("sigma = " << sigma << " -> " << sigmas[1]
                    << "  max|next - Euler(latent, denoised)| = " << worst
                    << "  scale = " << scale);
    REQUIRE_MESSAGE(scale > 1e-3,
                    "the recomputed Euler output is zero, so the residual below bounds nothing");
    CHECK_MESSAGE(worst <= 1e-5 * scale,
                  "the latent `Ltx2EulerStep` wrote is not the step over the recorded denoised "
                  "prediction, so the sampler was handed some other tensor (#1039): residual "
                      << worst << " against a tolerance of " << (1e-5 * scale));
  }
}

TEST_CASE("ltx2 one_stage: rescale_scale 0 is the control and the modality term is INERT in it") {
  // #1039's control on the VIDEO row. It runs with `modality_scale = 3.0`, which
  // the T2A control could not carry because that pipeline pins it to 1.0
  // (t2a_one_stage.py:202) — so the isolated-modality arm is inside a space
  // control here for the first time.
  //
  // WHAT THAT IS WORTH, measured rather than implied, and the case's own title
  // said more than the number supports until 2026-08-17. Presence is coverage,
  // not discriminating power: the third measurement below pins `modality_scale`
  // to 1.0 and the disagreement at the shipped rescale barely moves. A reader
  // must not lean on this control for modality coverage. THE MODALITY ARM'S GATE
  // IS THE PER-ARM INVARIANT in the case above, whose `modality` row is the one
  // mutation M4 (the `mod` pass left in velocity space) turns red; this control
  // gates the RESCALE, on a guider that happens to have four terms.
  //
  // The case above would be testing something OTHER than the defect if it also
  // fired at `rescale_scale = 0`, because `MultiModalGuider.calculate`'s linear
  // terms (guiders.py:261-266) are invariant under `x0 = latent - sigma*v`:
  //
  //   latent - sigma*(c + a(c-u) + b(c-p) + d(c-m))
  //     == x0c + a(x0c-x0u) + b(x0c-x0p) + d(x0c-x0m)
  //
  // The rescale at `:268-271` is the only part that is not.
  const int64_t n = 512;
  std::vector<float> latent(static_cast<size_t>(n));
  std::vector<float> v_cond(static_cast<size_t>(n));
  std::vector<float> v_uncond(static_cast<size_t>(n));
  std::vector<float> v_ptb(static_cast<size_t>(n));
  std::vector<float> v_mod(static_cast<size_t>(n));
  // Deterministic and NON-CONSTANT. A zero latent erases `(factor - 1) * latent`
  // entirely and a constant one reduces it to a uniform offset; either would make
  // the disagreement below unmeasurable and the control meaningless.
  uint64_t s = 0x9E3779B97F4A7C15ULL;
  const auto next = [&s]() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>(static_cast<double>(s >> 11) / 9007199254740992.0 * 2.0 - 1.0);
  };
  for (int64_t i = 0; i < n; ++i) {
    const size_t j = static_cast<size_t>(i);
    latent[j] = 2.0F * next();
    v_cond[j] = next();
    v_uncond[j] = next();
    v_ptb[j] = next();
    v_mod[j] = next();
  }
  const float sigma = 0.83F;
  std::vector<float> x_cond(static_cast<size_t>(n));
  std::vector<float> x_uncond(static_cast<size_t>(n));
  std::vector<float> x_ptb(static_cast<size_t>(n));
  std::vector<float> x_mod(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const size_t j = static_cast<size_t>(i);
    x_cond[j] = latent[j] - sigma * v_cond[j];
    x_uncond[j] = latent[j] - sigma * v_uncond[j];
    x_ptb[j] = latent[j] - sigma * v_ptb[j];
    x_mod[j] = latent[j] - sigma * v_mod[j];
  }

  // The 2.4/2.5 VIDEO row, read from the shared recipe table rather than typed.
  const vllm::Ltx2PipelineRecipe recipe = vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5");
  REQUIRE(recipe.phases.size() == 1);
  vllm::Ltx2MultiModalGuiderParams params = recipe.phases[0].video_guidance;
  REQUIRE(params.rescale_scale == 0.7);
  REQUIRE(params.modality_scale == 3.0);

  const auto compare = [&](double rescale) {
    params.rescale_scale = rescale;
    // Upstream's shape: combine the X0 predictions.
    const std::vector<float> x0_space = vllm::Ltx2MultiModalGuidance(
        params, x_cond.data(), x_uncond.data(), x_ptb.data(), x_mod.data(), n);
    // The shape a port reaches for by accident: combine the VELOCITIES and
    // convert once after.
    const std::vector<float> v_space = vllm::Ltx2MultiModalGuidance(
        params, v_cond.data(), v_uncond.data(), v_ptb.data(), v_mod.data(), n);
    double worst = 0.0;
    double scale = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const size_t j = static_cast<size_t>(i);
      const double converted = static_cast<double>(latent[j]) -
                               static_cast<double>(sigma) * static_cast<double>(v_space[j]);
      worst = std::max(worst, std::abs(static_cast<double>(x0_space[j]) - converted));
      scale = std::max(scale, std::abs(static_cast<double>(x0_space[j])));
    }
    REQUIRE(scale > 1e-3);
    return worst / scale;
  };

  const double at_zero = compare(0.0);
  const double at_default = compare(0.7);
  // The same pair with the modality term switched OFF, which is what T2A's
  // control already measured. Restored afterwards so nothing below reads a
  // mutated params object.
  const double shipped_modality = params.modality_scale;
  params.modality_scale = 1.0;
  const double at_zero_no_modality = compare(0.0);
  const double at_default_no_modality = compare(0.7);
  params.modality_scale = shipped_modality;

  INFO("relative disagreement: at rescale 0.0 = "
       << at_zero << "  at rescale 0.7 = " << at_default
       << "   |  modality_scale pinned to 1.0: at 0.0 = " << at_zero_no_modality
       << "  at 0.7 = " << at_default_no_modality);
  // AT 0.0 THE TWO SPACES ARE THE SAME FUNCTION, to f32 rounding — with the
  // modality term present, which is the arm this control adds over T2A's.
  CHECK(at_zero < 1e-4);
  // AT THE SHIPPED 0.7 THEY ARE NOT, by orders of magnitude more.
  CHECK(at_default > 1e-2);
  CHECK(at_default > 100.0 * at_zero);

  // AND THE MODALITY TERM IS NOT WHAT SEPARATES THEM. Asserted rather than left
  // in prose, because the case's own comment implied the opposite and a later
  // reader would otherwise treat this control as modality coverage. The two
  // `0.7` numbers agree to well inside a factor of two: adding a fourth linear
  // term changes what the rescale is computed over and does not change whether
  // the rescale is the term that breaks the equivalence.
  CHECK(at_zero_no_modality < 1e-4);
  CHECK(at_default_no_modality > 1e-2);
  CHECK_MESSAGE(at_default_no_modality > 0.5 * at_default,
                "the modality term turned out to carry the disagreement after all, which would "
                "make this control modality coverage rather than rescale coverage");
  CHECK_MESSAGE(at_default_no_modality < 2.0 * at_default,
                "the modality term turned out to carry the disagreement after all");
}

TEST_CASE("ltx2 one_stage: post_process_latent runs AFTER the guider, not per arm (#1092)") {
  // WHERE `post_process_latent` IS APPLIED, gated on a render that has something
  // for it to move. The unconditioned case above cannot see this at all: every
  // denoise mask entry is 1 there, so `x*mask + clean*(1-mask)`
  // (utils/helpers.py:462-464) is a literal no-op and any placement of it passes.
  //
  // TWO THINGS ARE TRUE HERE AND THEY ARE EASY TO CONFUSE, so both are asserted.
  //
  // (1) Applying it to each ARM is an IDENTITY, and that is not a gap in this
  //     case -- it is arithmetic. A conditioned token arrives with its per-token
  //     sigma at 0 (`timesteps_from_mask`, utils/helpers.py:494-503), so
  //     `X0Model` returns `latent - 0*v`, which is `latent`; and a conditioned
  //     token's `latent` IS its clean value, which is what the conditioner wrote
  //     and what the Euler step preserves. So every arm already equals what
  //     post-processing would write. MEASURED: adding it per arm runs the whole
  //     suite to 71 cases / 2145 assertions / exit 0, and the arm assertion below
  //     is what says WHY rather than leaving the green unexplained.
  //
  // (2) Applying it after the GUIDER is emphatically not an identity, and that is
  //     the thing worth gating. The guider's rescale (guiders.py:268-271) is a
  //     scalar over the WHOLE tensor, so it multiplies the conditioned tokens too
  //     -- `pred = latent * factor` there, because every guidance term is zero on
  //     a token where all four arms agree. `post_process_latent` is what pins
  //     them back to `clean`. Take it out, or move it a level down into the
  //     denoiser, and the conditioned tokens leave the step scaled by a number
  //     nobody asked for, on a render that finishes.
  //
  // So this case asserts that the arms were NOT touched and that the guider's
  // result WAS, on exactly the mask-0 tokens.
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(OneStageParams(ws.paths));
  REQUIRE(engine != nullptr);
  vllm::multimodal::VideoGenParams gen = OneStageGen(ws.root + "/conditioned");
  gen.first_frame_ppm = ConditioningPpm(20, 28, 1);
  gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
  (void)engine->Generate(gen);
  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);
  REQUIRE(t.video_guided);
  // The same four arms as the case above, not a degenerate set.
  CHECK(t.video_uncond_forwards == 1);
  CHECK(t.video_perturbed_forwards == 1);
  CHECK(t.video_modality_forwards == 1);

  const size_t tokens = t.video_first_timesteps.size();
  REQUIRE(tokens > 0);
  const size_t n = t.video_first_latent.size();
  REQUIRE(n > 0);
  const size_t width = n / tokens;
  REQUIRE(width * tokens == n);

  // NON-VACUITY, both ends. With no conditioned token this case is the one above
  // again; with every token conditioned there is nothing left to denoise.
  size_t conditioned = 0;
  for (size_t token = 0; token < tokens; ++token) {
    if (t.video_first_timesteps[token] == 0.0F) ++conditioned;
  }
  INFO("conditioned tokens = " << conditioned << " of " << tokens);
  REQUIRE_MESSAGE(conditioned > 0,
                  "no token arrived at the denoiser with a zero timestep, so the image "
                  "conditioning did not reach the denoise mask and this case tests nothing");
  REQUIRE_MESSAGE(conditioned < tokens,
                  "EVERY token is conditioned, so there is nothing left to denoise");

  // (2), AND THE POSITIVE CONTROL FOR THE WHOLE CASE. `post_process_latent` MOVES
  // something on this render: the tensor the stepper was handed is not the
  // guider's own output. Without this, every assertion here would be satisfied by
  // a render where post-processing happened to be a no-op, which is exactly what
  // the unconditioned case above is.
  REQUIRE(t.video_first_stepper_input.size() == n);
  REQUIRE_MESSAGE(t.video_first_stepper_input != t.video_first_denoised,
                  "`post_process_latent` changed nothing on this render, so it cannot matter "
                  "WHERE it was applied and this case discriminates nothing");

  // AND IT MOVED ONLY THE CONDITIONED TOKENS, which is what makes the next
  // assertion a statement about placement rather than about some third operation.
  for (size_t token = 0; token < tokens; ++token) {
    const bool is_conditioned = t.video_first_timesteps[token] == 0.0F;
    for (size_t c = 0; c < width; ++c) {
      const size_t i = token * width + c;
      const bool moved = t.video_first_stepper_input[i] != t.video_first_denoised[i];
      if (moved == is_conditioned) continue;
      INFO("token = " << token << " channel = " << c);
      FAIL_CHECK("`post_process_latent` moved a token whose denoise mask does not match: it is "
                 "`x*mask + clean*(1-mask)` and must move exactly the mask-0 tokens");
      break;
    }
  }

  // (1). Every arm the forward returned is `latent - sigma*velocity`, INCLUDING
  // on the conditioned tokens, where that is `latent` itself. This is what makes
  // the per-arm placement an identity rather than an undetected defect, and it is
  // asserted rather than argued because the argument depends on a conditioned
  // token's `latent` being its clean value -- a property of the CONDITIONER, one
  // file away, that nothing here would otherwise hold.
  const std::vector<float>* arms[] = {&t.video_first_cond, &t.video_first_uncond,
                                      &t.video_first_perturbed, &t.video_first_modality};
  const std::vector<float>* velocities[] = {
      &t.video_first_cond_velocity, &t.video_first_uncond_velocity,
      &t.video_first_perturbed_velocity, &t.video_first_modality_velocity};
  const char* names[] = {"cond", "uncond", "perturbed", "modality"};
  for (size_t k = 0; k < 4; ++k) {
    INFO("arm = " << std::string(names[k]));
    REQUIRE(arms[k]->size() == n);
    REQUIRE(velocities[k]->size() == n);
    double worst = 0.0;
    for (size_t token = 0; token < tokens; ++token) {
      const double sigma = static_cast<double>(t.video_first_timesteps[token]);
      for (size_t c = 0; c < width; ++c) {
        const size_t i = token * width + c;
        // `ToDenoised` subtracts in double and STORES f32, so the expectation is
        // rounded the same way. Comparing against the unrounded double leaves one
        // ULP of disagreement -- measured at 5.96e-08, which is 2^-24 -- and a
        // tolerance wide enough to absorb it would also absorb a real defect an
        // order of magnitude away.
        const float expected = static_cast<float>(static_cast<double>(t.video_first_latent[i]) -
                                                  sigma * static_cast<double>((*velocities[k])[i]));
        worst = std::max(worst, std::abs(static_cast<double>((*arms[k])[i]) -
                                         static_cast<double>(expected)));
      }
    }
    INFO("max|arm - (latent - sigma*velocity)| = " << worst);
    CHECK_MESSAGE(worst == 0.0,
                  "this arm is not `latent - sigma*velocity` on every token, so something was "
                  "applied to it between the forward and the guider -- and if that something is "
                  "`post_process_latent`, it has stopped being an identity on the arms and the "
                  "per-arm placement is now a real divergence rather than a harmless one");
  }
}

TEST_CASE("ltx2 guided video: the refusals that would otherwise RENDER (#1092)") {
  Workspace ws;

  SUBCASE("an unconditional forward with no negative conditioning is refused BY NAME") {
    // The positive embeds alone, which is what every engine here loaded before
    // this row. `cfg_scale = 3.0` asks for a forward whose context does not
    // exist; serving the POSITIVE context twice would make the whole CFG term
    // exactly zero and render an unguided clip wearing a guided configuration.
    vllm::multimodal::VideoModelParams mp = OneStageParams(ws.paths);
    mp.extras.erase(vllm::multimodal::kLtx2NegativePromptEmbedsExtra);
    mp.extras.erase(vllm::multimodal::kLtx2NegativeAudioPromptEmbedsExtra);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    try {
      (void)engine->Generate(OneStageGen(ws.root + "/no_negative"));
      FAIL("a cfg_scale of 3.0 with no negative conditioning must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("unconditional forward") != std::string::npos);
      CHECK(msg.find(vllm::multimodal::kLtx2NegativePromptEmbedsExtra) != std::string::npos);
    }
  }

  SUBCASE("cfg_scale 1.0 turns the unconditional pass off instead of needing one") {
    // The other half of the branch above, and what makes it a statement about the
    // GUIDER rather than a blanket requirement: `do_unconditional_generation` is
    // `not isclose(cfg_scale, 1.0)` (guiders.py:275-277), so at 1.0 there is no
    // pass and nothing to encode.
    vllm::multimodal::VideoModelParams mp = OneStageParams(ws.paths);
    mp.extras.erase(vllm::multimodal::kLtx2NegativePromptEmbedsExtra);
    mp.extras.erase(vllm::multimodal::kLtx2NegativeAudioPromptEmbedsExtra);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = OneStageGen(ws.root + "/cfg_one");
    gen.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "1.0";
    gen.extras[vllm::multimodal::kLtx2AudioCfgScaleExtra] = "1.0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    CHECK(result.width == 64);
    const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    CHECK(t.video_uncond_forwards == 0);
    // And the OTHER two passes still ran, so this subcase turned off exactly one
    // arm rather than the guidance.
    CHECK(t.video_perturbed_forwards == 1);
    CHECK(t.video_modality_forwards == 1);
  }

  SUBCASE("an STG block this checkpoint does not have is refused, not silently ignored") {
    // `Perturbation.is_perturbed` is `block in self.blocks`
    // (guidance/perturbations.py:26-33), so a block index past the end perturbs
    // NOTHING: the perturbed forward returns the conditional pass's own tensor
    // and `stg_scale * (cond - perturbed)` is exactly zero. The render is finite,
    // the right size, and carries no spatio-temporal guidance whatever.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(OneStageParams(ws.paths));
    vllm::multimodal::VideoGenParams gen = OneStageGen(ws.root + "/stg_oob");
    gen.extras[vllm::multimodal::kLtx2VideoStgBlocksExtra] = "28";
    try {
      (void)engine->Generate(gen);
      FAIL("block 28 on a two-block DiT perturbs nothing and must be refused");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("stg_blocks") != std::string::npos);
      CHECK(msg.find("exactly zero") != std::string::npos);
    }
  }

  SUBCASE("an EMPTY stg_blocks is SERVED - it is upstream's own way to disable STG") {
    // THIS SUBCASE ASSERTED A REFUSAL UNTIL 2026-08-17. Measured at
    // Lightricks/LTX-2 `fd4ded7f`: `docs/multimodal-guidance.md:13` documents
    // "Set to `[]` to disable STG"; `MultiModalGuiderParams.stg_blocks` defaults
    // to `[]` (guiders.py:204); the flags are `nargs="*"` (args.py:979-985) so
    // the empty list has a CLI spelling; `LTX_2_3_HQ_PARAMS` ships it on both
    // modalities (constants.py:105, :113); and nothing in that tree validates
    // the list at all. Refusing it made this port reject a configuration its
    // reference documents, ships and cannot express any other way.
    //
    // Upstream does NOT skip the pass either: `do_perturbed_generation` reads
    // `stg_scale` alone (guiders.py:279-281), so the "ptb" entry is appended and
    // the batch carries a sample whose result equals `cond`. The forward count
    // below is that fact, and it is why an empty list disables the STG SIGNAL
    // and not the STG COST.
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(OneStageParams(ws.paths));
    REQUIRE(engine != nullptr);
    vllm::multimodal::VideoGenParams gen = OneStageGen(ws.root + "/stg_empty");
    gen.extras[vllm::multimodal::kLtx2VideoStgBlocksExtra] = "";
    gen.extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "";
    (void)engine->Generate(gen);  // it RENDERS; a throw fails the case
    const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    REQUIRE(t.video_guided);
    // The pass still ran, because the scale still asks for it.
    CHECK(t.video_perturbed_forwards == 1);
    // And it perturbed nothing, read off the mask handed to the DiT.
    CHECK(t.video_perturbed_blocks.empty());
    CHECK(t.video_audio_perturbed_blocks.empty());
    // So the STG term is not merely small, it is EXACTLY zero: the perturbed arm
    // is the conditional arm bit for bit. An exact comparison, because a
    // tolerance here would also pass on a build that perturbed a block and
    // happened to move little.
    REQUIRE(!t.video_first_cond.empty());
    CHECK_MESSAGE(t.video_first_perturbed == t.video_first_cond,
                  "an empty stg_blocks perturbed something, so PRESENT-and-empty was collapsed "
                  "onto some other value (`blocks=None` is EVERY block upstream, "
                  "perturbations.py:26-33)");
    // The control that this is about EMPTINESS and not about the extra being
    // read at all: the same render with a real block moves the arm.
    vllm::multimodal::VideoGenParams named = OneStageGen(ws.root + "/stg_named");
    named.extras[vllm::multimodal::kLtx2VideoStgBlocksExtra] = "1";
    named.extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "1";
    (void)engine->Generate(named);
    const vllm::multimodal::Ltx2ConditioningTrace n = ltx->last_conditioning();
    CHECK(n.video_perturbed_blocks == std::vector<int64_t>{1});
    CHECK(n.video_first_perturbed != n.video_first_cond);
  }

  SUBCASE("a recipe that fixes its guidance refuses the override rather than applying it") {
    // `allow_guidance_override = false` on the distilled two-stage recipe
    // (ltx2_recipes.py:125-158), whose scales are distilled INTO the weights.
    // Until this row nothing read that field at all.
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;  // the two-stage recipe's phase 1
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/fixed_guidance");
    gen.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "5.0";
    try {
      (void)engine->Generate(gen);
      FAIL("the distilled recipe fixes its guidance and must refuse the override");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find("fixes its own guidance") != std::string::npos);
      CHECK(msg.find(vllm::multimodal::kLtx2VideoCfgScaleExtra) != std::string::npos);
    }
  }

  SUBCASE("the DISTILLED recipe runs ONE forward, which is what SimpleDenoiser is") {
    // The guided seam is on every video render now, so the recipes upstream
    // denoises with `SimpleDenoiser` (distilled.py:266,295) must still issue one
    // forward per step. Their guiders are `Ltx2MultiModalGuiderParams`'s own
    // defaults, which is `_POSITIVE_ONLY_GUIDER` (denoisers.py:25-28).
    vllm::multimodal::VideoModelParams mp = FixtureParams(ws.paths);
    mp.extras["upsampler_path"] = ws.paths.upsampler;  // the two-stage recipe's phase 1
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    (void)engine->Generate(FixtureGen(ws.root + "/distilled_simple"));
    const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.video_guided);
    CHECK(t.video_cond_forwards == 1);
    CHECK(t.video_uncond_forwards == 0);
    CHECK(t.video_perturbed_forwards == 0);
    CHECK(t.video_modality_forwards == 0);
    // And the guider was the identity over that one pass, so this recipe's
    // trajectory is unchanged by the seam.
    CHECK(t.video_first_denoised == t.video_first_cond);
  }
}

// ─── LTX25-A2VID-RECIPE (#1117) ──────────────────────────────────────────────

namespace {

// An `a2vid_two_stage` engine on the shipped fixture. Both load-side
// requirements the recipe carries are met here: the spatial upsampler stage 2
// needs (through `ConditioningParams`) and the distilled adapter upstream's
// `--distilled-lora required=True` demands.
vllm::multimodal::VideoModelParams A2VidParams(const ltx2_fixture::Paths& paths,
                                               const std::string& lora) {
  vllm::multimodal::VideoModelParams mp = ConditioningParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "a2vid_two_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("a2vid_two_stage");
  mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
  return mp;
}

// The request. Two things beside the take, and each is a property of the FIXTURE
// rather than of this row:
//
//   * `steps = 2`, because stage 1's schedule is DERIVED from the step count
//     (a2vid_two_stage.py:225-227) and two sigma intervals exercise the loop.
//     That this is accepted at all is part of what the case asserts — the
//     distilled recipe refuses a `steps` override.
//   * the STG block list, because the reduced DiT has TWO blocks and the params
//     row this recipe resolves names block 28 — LTX_2_3_PARAMS overrides 2.0's
//     [29] to [28] (utils/constants.py:86) and 2.4, the row 2.5 resolves onto,
//     inherits it (:124). `OneStageFixtureGuidance` carries the whole
//     argument; the override reaching stage 1 and being IGNORED by stage 2 is
//     itself gated below.
vllm::multimodal::VideoGenParams A2VidGen(const std::string& out_dir, const std::string& wav,
                                          double start_time = 0.0) {
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.steps = 2;
  OneStageFixtureGuidance(&gen);
  gen.extras[vllm::multimodal::kLtx2AudioPathExtra] = wav;
  if (start_time != 0.0) {
    gen.extras[vllm::multimodal::kLtx2AudioStartTimeExtra] = std::to_string(start_time);
  }
  return gen;
}

// Every artifact a render wrote, concatenated. Downstream of the DiT weights and
// of every guidance decision, which is what makes it able to see a pass that ran
// on a phase the trace does not record.
std::string A2VidArtifacts(const std::string& out_dir,
                           const vllm::multimodal::VideoResult& result) {
  std::string bytes;
  for (int64_t f = 0; f < result.frame_count; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
    bytes += ReadAll(out_dir + name);
  }
  bytes += ReadAll(std::string(result.audio_path));
  return bytes;
}

}  // namespace

TEST_CASE("ltx2 a2vid: the pipeline renders through vllm.h and CONSUMES its take") {
  // THE REACHABILITY CLAIM, and it is the point of this case rather than a note
  // beside it. Entry point: `LoadVideoEngine` with a documented value of the
  // documented `pipeline_kind` LOAD extra, then `Generate` with the documented
  // `audio_path` per-generation extra. Nothing here constructs a recipe, a
  // guider, a phase or a modality by hand. Deleting the `a2vid_two_stage`
  // dispatch row in `ResolveLtx2PipelineRecipe` REDs this case at the load,
  // which is what separates measuring a capability from measuring a class
  // (.agents/reachability.md).
  //
  // `ltx2-gen --pipeline-kind a2vid_two_stage --audio-path ...` is the same two
  // calls through the ABI, as a thin client that includes no internal header.
  // The `/v1/videos` route CANNOT drive it: `VideoGenParamsFromRequest` never
  // writes `gen.extras` (#928), so no per-generation extra reaches any engine
  // over HTTP. Stated here because the reach claim has to exclude it.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);
  const std::string wav = WriteWav(ws.root + "/take.wav", 2, kFixtureAudioRate, 2.0);

  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(A2VidParams(ws.paths, lora));
  REQUIRE(engine != nullptr);
  auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  CHECK(ltx->pipeline_kind() == "a2vid_two_stage");

  const vllm::multimodal::VideoResult result = engine->Generate(A2VidGen(ws.root + "/a2v", wav));
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);
  CHECK(result.frame_count == 9);
  // Upstream returns the caller's own waveform rather than a VAE round trip of
  // it (`:301-303`), and the observable consequence is the SAMPLE RATE: the
  // vocoder's BWE arm emits 48 kHz where the take went in at the audio VAE's own
  // rate.
  CHECK(result.sample_rate == kFixtureAudioRate);

  // ── the take was CONSUMED, not merely carried ─────────────────────────────
  //
  // A recipe-level assertion proves `noise_scale = 0.0` and `frozen = True` are
  // SET. These four say the DiT saw the consequence, and they are read off the
  // LAST phase — so stage 2's own `noise_scale` of 0.909375, which the loop
  // applies to both streams, is inside what they measure.
  CHECK(t.audio_conditioned);
  CHECK_MESSAGE(t.audio_frozen,
                "the audio denoise mask was not all zeros at the last phase, so the sampler was "
                "free to move the caller's take (utils/types.py:104-106)");
  CHECK_MESSAGE(t.audio_sigma_max == 0.0,
                "the scalar `Modality.sigma` was left at the schedule's value on some step; the "
                "zeroed mask cannot reach that input, and a DiT told its clean conditioning is "
                "noisy still renders");
  CHECK(t.audio_latent_absmax > 0.0);
  CHECK(t.audio_latent_digest != 0);

  // THE CONTROL THAT MAKES THOSE MEAN SOMETHING. Same take, same request, a
  // DIFFERENT seed: the audio latent must be BIT-IDENTICAL, because it is the
  // encoded file and not a sample. A build that noised the audio stream — or
  // that generated it and let the take decorate the trace — moves this digest,
  // and moves nothing a caller can see.
  vllm::multimodal::VideoGenParams reseeded = A2VidGen(ws.root + "/a2v_seed", wav);
  reseeded.seed = 99;
  (void)engine->Generate(reseeded);
  const vllm::multimodal::Ltx2ConditioningTrace t_seed = ltx->last_conditioning();
  REQUIRE(t_seed.completed);
  CHECK_MESSAGE(t_seed.audio_latent_digest == t.audio_latent_digest,
                "the audio latent changed with the SEED, so it is being sampled rather than "
                "taken from the caller's file");
  // ...and the second control, so the first cannot be passing because the latent
  // is a constant: a different WINDOW of the same file gives a different latent.
  const vllm::multimodal::VideoResult windowed =
      engine->Generate(A2VidGen(ws.root + "/a2v_window", wav, 0.5));
  (void)windowed;
  const vllm::multimodal::Ltx2ConditioningTrace t_window = ltx->last_conditioning();
  REQUIRE(t_window.completed);
  CHECK_MESSAGE(t_window.audio_latent_digest != t.audio_latent_digest,
                "windowing the take 0.5s later produced the SAME latent, so the samples are not "
                "reaching the encoder");

  // ── stage 1 ran upstream's GUIDED denoiser, in x0 space, on every arm ──────
  //
  // The trace's guided fields are recorded at step 0 of phase 0, which is
  // a2vid's stage 1. Its guider is the params table's video row — cfg 3.0,
  // stg 1.0, rescale 0.7, modality 3.0 — so all four passes run and the rescale
  // branch, the one term that is NOT invariant between the two spaces, is live.
  REQUIRE_MESSAGE(t.video_guided, "stage 1 did not go through the guided seam at all");
  CHECK(t.video_cond_forwards == 1);
  CHECK(t.video_uncond_forwards == 1);
  CHECK(t.video_perturbed_forwards == 1);
  CHECK(t.video_modality_forwards == 1);
  CHECK(t.video_guidance_cfg_scale == 3.0);
  CHECK(t.video_guidance_stg_scale == 1.0);
  CHECK(t.video_guidance_rescale_scale == 0.7);
  CHECK(t.video_guidance_modality_scale == 3.0);

  // AND STAGE 1'S SCHEDULE WAS DERIVED, not read off a frozen list.
  // `schedule_tokens` is written only on the branch that calls
  // `Ltx2SigmaSchedule`, and stays 0 on a recipe carrying its own distilled
  // sigmas — which is the difference between upstream's
  // `self._scheduler.execute(steps=num_inference_steps)` (a2vid_two_stage.py:225-227)
  // and the eight-step distilled list. Without this the recipe case is the only
  // thing that can see a stage 1 handed the wrong schedule, and a wrong schedule
  // renders.
  CHECK_MESSAGE(t.schedule_tokens > 0,
                "stage 1 did not derive its schedule from the step count, so it is running a "
                "frozen sigma list upstream does not give it");

  const size_t n = t.video_first_latent.size();
  REQUIRE(n > 0);
  const size_t tokens = t.video_first_timesteps.size();
  REQUIRE(tokens > 0);
  const size_t width = n / tokens;
  REQUIRE(width * tokens == n);
  // THE FIXTURE CAN DECIDE THIS AT ALL: `latent - sigma*velocity` and `velocity`
  // coincide when the sample is zero. A REQUIRE, because nothing below
  // discriminates once it fails.
  double latent_span = 0.0;
  for (const float x : t.video_first_latent) {
    latent_span = std::max(latent_span, std::abs(static_cast<double>(x)));
  }
  REQUIRE_MESSAGE(latent_span > 1e-3, "the step-0 sample is zero, so the two candidate tensors "
                                      "coincide and nothing below discriminates");

  struct Arm {
    const char* name;
    const std::vector<float>& velocity;
    const std::vector<float>& x0;
  };
  const Arm arms[] = {
      {"cond", t.video_first_cond_velocity, t.video_first_cond},
      {"uncond", t.video_first_uncond_velocity, t.video_first_uncond},
      {"perturbed", t.video_first_perturbed_velocity, t.video_first_perturbed},
      {"modality", t.video_first_modality_velocity, t.video_first_modality},
  };
  for (const Arm& arm : arms) {
    INFO("arm = " << std::string(arm.name));
    REQUIRE(arm.velocity.size() == n);
    REQUIRE(arm.x0.size() == n);
    // A zeroed velocity makes `to_denoised` the identity on this arm alone and
    // would satisfy the equation while proving nothing.
    double velocity_span = 0.0;
    for (const float x : arm.velocity) {
      velocity_span = std::max(velocity_span, std::abs(static_cast<double>(x)));
    }
    REQUIRE_MESSAGE(velocity_span > 1e-6, "this arm's velocity is zero, so the equation below "
                                          "holds for a reason that is not the one it tests");
    // `x0 = latent - sigma_token * velocity` (model.py:590-604), with the
    // PER-TOKEN timestep and not the schedule scalar.
    double residual = 0.0;
    double against_velocity = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double sigma = static_cast<double>(t.video_first_timesteps[i / width]);
      const double expected = static_cast<double>(t.video_first_latent[i]) -
                              sigma * static_cast<double>(arm.velocity[i]);
      residual = std::max(residual, std::abs(static_cast<double>(arm.x0[i]) - expected));
      against_velocity = std::max(
          against_velocity,
          std::abs(static_cast<double>(arm.x0[i]) - static_cast<double>(arm.velocity[i])));
    }
    INFO("max|x0 - (latent - sigma*v)| = " << residual);
    INFO("max|x0 - velocity| = " << against_velocity);
    // In VELOCITY space the first number is the whole sample and the second is
    // exactly 0, which is what the RED prints.
    CHECK(residual < 1e-4);
    CHECK(against_velocity > 1e-6);
  }
}

TEST_CASE("ltx2 a2vid: every requirement the recipe adds refuses BY WHAT IS MISSING") {
  // Three refusals, and each one guards a configuration that would otherwise
  // RENDER — a finished clip at the right size, frame count and sample rate,
  // with nothing in any output to show what was dropped.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);
  const std::string wav = WriteWav(ws.root + "/take.wav", 2, kFixtureAudioRate, 2.0);

  // ── no distilled adapter, refused at LOAD (utils/args.py:1140-1155) ────────
  {
    vllm::multimodal::VideoModelParams mp = A2VidParams(ws.paths, lora);
    mp.extras.erase(vllm::multimodal::kLtx2LoraPathExtra);
    try {
      const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
          vllm::multimodal::LoadVideoEngine(mp);
      FAIL_CHECK("an a2vid load with no distilled LoRA must be refused");
    } catch (const std::exception& e) {
      const std::string message = e.what();
      INFO("message = " << message);
      CHECK(message.find("distilled LoRA") != std::string::npos);
      CHECK(message.find("lora_path") != std::string::npos);
      CHECK(message.find(":1140-1155") != std::string::npos);
      // THE ANCHOR IS THE SHARED PARSER, not this pipeline's own stage 2. The
      // refusal is keyed on `requires_distilled_lora` so that every recipe off
      // `default_2_stage_arg_parser` inherits it, and it used to hard-code
      // `a2vid_two_stage.py`'s line numbers — which named the caller's pipeline
      // in one sentence and cited a different one in the next as soon as
      // `ti2vid_two_stage` arrived (#1151).
      CHECK(message.find("default_2_stage_arg_parser") != std::string::npos);
      // AND IT NO LONGER ADVERTISES #1118, which closed at `4ae0f54ab`. The
      // message used to end by telling the caller this engine fuses once at
      // load so stage 1 sees the adapter too; `Ltx2PhaseRecipe::loras` made
      // every clause of that false. A refusal that describes a divergence which
      // no longer exists is worse than one that says nothing (#1151).
      CHECK(message.find("1118") == std::string::npos);
      CHECK(message.find("fuses once at load") == std::string::npos);
    }
    // THE CONTROL: the same load on the DEFAULT kind is fine without an adapter,
    // so this is the recipe's requirement and not a new global one.
    vllm::multimodal::VideoModelParams distilled = ConditioningParams(ws.paths);
    CHECK_NOTHROW((void)vllm::multimodal::LoadVideoEngine(distilled));
  }

  // ── no take, refused at GENERATE (a2vid_two_stage.py:312-317) ──────────────
  {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(A2VidParams(ws.paths, lora));
    vllm::multimodal::VideoGenParams gen = A2VidGen(ws.root + "/no_take", wav);
    gen.extras.erase(vllm::multimodal::kLtx2AudioPathExtra);
    try {
      (void)engine->Generate(gen);
      FAIL_CHECK("an a2vid render with no audio_path must be refused");
    } catch (const std::exception& e) {
      const std::string message = e.what();
      INFO("message = " << message);
      CHECK(message.find("audio_path") != std::string::npos);
      CHECK(message.find("a2vid_two_stage.py:312-317") != std::string::npos);
      CHECK(message.find("GENERATED") != std::string::npos);
    }
    // THE CONTROL: the take is what the refusal is about, and supplying it on
    // the same engine renders.
    CHECK_NOTHROW((void)engine->Generate(A2VidGen(ws.root + "/with_take", wav)));
  }

  // ── the guider override REACHES stage 1 and is IGNORED by stage 2 ──────────
  //
  // Upstream's `--video-cfg-guidance-scale` exists on this pipeline's parser
  // (a2vid_two_stage.py:311 -> utils/args.py:947-1006) and reaches stage 1's
  // guider alone (`:233-236`), because stage 2 is `SimpleDenoiser(...)` (`:278`).
  // A build that REFUSED it would reject a request upstream accepts; a build
  // that applied it to stage 2 would run an unconditional forward upstream's
  // stage 2 does not, and neither shows up in any output.
  {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(A2VidParams(ws.paths, lora));
    auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    vllm::multimodal::VideoGenParams gen = A2VidGen(ws.root + "/override", wav);
    gen.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "1.0";
    CHECK_NOTHROW((void)engine->Generate(gen));
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    // It reached STAGE 1: the trace's guidance fields are phase 0's, and the
    // recipe's own value is 3.0.
    CHECK_MESSAGE(t.video_guidance_cfg_scale == 1.0,
                  "the override did not reach stage 1's guider, so a2vid's caller-configured "
                  "guidance is unreachable");
    CHECK(t.video_uncond_forwards == 0);

    // AND IT DID NOT REACH STAGE 2, measured on the artifacts because no trace
    // field records what the second phase did.
    //
    // The instrument is a pair of renders whose difference is a value that is
    // ALREADY stage 1's. `video_stg_scale = 1.0` is exactly what this recipe's
    // stage 1 carries (`utils/constants.py:52`), so applying it there changes
    // nothing; stage 2's own STG scale is 0.0, so applying it THERE adds a
    // perturbed forward per step and moves every pixel downstream of it. Equal
    // bytes therefore mean the override stopped at stage 1, and that is a claim
    // an `allow_guidance_override` boolean cannot make either way.
    const vllm::Ltx2PipelineRecipe recipe =
        vllm::ResolveLtx2PipelineRecipe("a2vid_two_stage", "2.5");
    REQUIRE(recipe.phases.size() == 2u);
    REQUIRE_MESSAGE(recipe.phases[0].video_guidance.stg_scale == 1.0,
                    "the value below is no longer stage 1's own, so the two renders differ for "
                    "a second reason and the comparison proves nothing");
    REQUIRE(recipe.phases[1].video_guidance.stg_scale == 0.0);

    vllm::multimodal::VideoGenParams plain = A2VidGen(ws.root + "/stg_plain", wav);
    const vllm::multimodal::VideoResult plain_result = engine->Generate(plain);
    const std::string plain_bytes = A2VidArtifacts(ws.root + "/stg_plain", plain_result);

    vllm::multimodal::VideoGenParams restated = A2VidGen(ws.root + "/stg_restated", wav);
    restated.extras[vllm::multimodal::kLtx2VideoStgScaleExtra] = "1.0";
    const vllm::multimodal::VideoResult restated_result = engine->Generate(restated);
    const std::string restated_bytes = A2VidArtifacts(ws.root + "/stg_restated", restated_result);

    REQUIRE(plain_bytes.size() > 0);
    REQUIRE(plain_bytes.size() == restated_bytes.size());
    // A COUNT of differing bytes, never the two buffers. These are PPM pixels
    // and a WAV, so a failing `CHECK(a == b)` dumps raw binary into the report —
    // which killed a mutation harness on the sibling row between applying a
    // mutation and restoring it, and left the tree mutated.
    size_t differing = 0;
    for (size_t i = 0; i < plain_bytes.size(); ++i) {
      if (plain_bytes[i] != restated_bytes[i]) ++differing;
    }
    CHECK_MESSAGE(differing == 0,
                  "restating stage 1's OWN stg_scale moved " << differing << " of "
                      << plain_bytes.size()
                      << " artifact bytes, so the override reached stage 2 — which runs "
                         "`SimpleDenoiser` upstream (a2vid_two_stage.py:278) and has no "
                         "guidance to switch on");
    // THE CONTROL for the same request on a recipe that FIXES its guidance: the
    // distilled kind refuses the identical extra, so the acceptance above is
    // this recipe's and not a weakening of that refusal.
    const std::unique_ptr<vllm::multimodal::VideoEngine> fixed =
        vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths));
    vllm::multimodal::VideoGenParams gen_fixed = FixtureGen(ws.root + "/override_fixed");
    gen_fixed.extras[vllm::multimodal::kLtx2VideoCfgScaleExtra] = "1.0";
    CHECK_THROWS((void)fixed->Generate(gen_fixed));
  }
}

// ─── LTX25-PHASE-LORA (#1118) ────────────────────────────────────────────────

TEST_CASE("ltx2 a2vid: the distilled adapter rides stage 2 ALONE") {
  // THE ROW, and the one case that separates a PER-PHASE adapter set from
  // fusion at load. A gate that only asserted "a LoRA was applied" passes on the
  // defect this row fixes, because the defect DOES apply the LoRA — to every
  // phase.
  //
  // Upstream, read at Lightricks/LTX-2 fd4ded7f: two `DiffusionStage`s are built
  // from the SAME `model_paths.transformer()` (a2vid_two_stage.py:104, :116) and
  // differ only in their adapter tuple — stage 1 takes `loras=tuple(loras)`
  // (`:107`) and stage 2 takes `(*tuple(loras), *tuple(distilled_lora))`
  // (`:114`, passed at `:119`). `ltx-pipelines/CLAUDE.md:48` states the same
  // convention in prose: the distilled adapter is "applied to stage 2 only in
  // TI2Vid/A2Vid/Keyframe".
  //
  // ENTRY POINT: `LoadVideoEngine` with the documented `pipeline_kind`,
  // `lora_path`, `lora_strength` and `max_phase` LOAD extras, then `Generate`.
  // Nothing here constructs a recipe, a phase or a checkpoint by hand.
  //
  // WHY STRENGTH 0 IS THE CONTROL and not "no adapter": `requires_distilled_lora`
  // refuses an a2vid load carrying no `lora_path` at all (upstream's
  // `--distilled-lora required=True`, utils/args.py:1140-1155), so the base-
  // weights arm has to be spelled some other way. Strength 0 fuses a ZERO delta,
  // and that it reproduces the base model is already gated independently by
  // "ltx2 video: the IC-LoRA strength reaches the PIXELS, and 0 is a no-op".
  Workspace ws;
  const std::string wav = WriteWav(ws.root + "/take.wav", 2, kFixtureAudioRate, 2.0);
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  // `max_phase` is a LOAD extra, so each arm is its own engine.
  const auto render = [&](const char* strength, const char* max_phase, const char* out) {
    vllm::multimodal::VideoModelParams mp = A2VidParams(ws.paths, lora);
    mp.extras[vllm::multimodal::kLtx2LoraStrengthExtra] = strength;
    if (max_phase != nullptr) mp.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = max_phase;
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    REQUIRE(engine != nullptr);
    const std::string dir = std::string(ws.root) + "/" + out;
    const vllm::multimodal::VideoResult result = engine->Generate(A2VidGen(dir, wav));
    return A2VidArtifacts(dir, result);
  };

  // ── stage 1 ALONE, adapter at full strength against a zero delta ───────────
  const std::string s1_full = render("1.0", "0", "s1_full");
  const std::string s1_zero = render("0.0", "0", "s1_zero");
  REQUIRE(s1_full.size() > 0);
  REQUIRE(s1_full.size() == s1_zero.size());

  size_t s1_differing = 0;
  for (size_t i = 0; i < s1_full.size(); ++i) {
    if (s1_full[i] != s1_zero[i]) ++s1_differing;
  }
  MESSAGE("stage 1 alone: the adapter moves " << s1_differing << " of " << s1_full.size()
                                              << " artifact bytes");
  // THE HALF THAT REDS ON TODAY'S DEFECT. Under fusion at load, stage 1 runs
  // base + distilled and this count is non-zero. `loras=tuple(loras)` at
  // `:107` names no distilled adapter, so stage 1 must be the base model and
  // the adapter's strength must be invisible to it.
  CHECK(s1_differing == 0);

  // ── both stages, the same two strengths ───────────────────────────────────
  const std::string both_full = render("1.0", nullptr, "both_full");
  const std::string both_zero = render("0.0", nullptr, "both_zero");
  REQUIRE(both_full.size() == both_zero.size());

  size_t both_differing = 0;
  for (size_t i = 0; i < both_full.size(); ++i) {
    if (both_full[i] != both_zero[i]) ++both_differing;
  }
  MESSAGE("both stages: the adapter moves " << both_differing << " of " << both_full.size()
                                            << " artifact bytes");
  // THE HALF THAT REDS ON "STOPPED FUSING ALTOGETHER". `stage_2_loras` at `:114`
  // DOES carry the distilled adapter, so it must reach the pixels through stage
  // 2. Without this line the case above is satisfied by an engine that ignores
  // `lora_path` entirely, which is the same shape of green-but-proves-nothing
  // the row's spec rejects.
  //
  // Strictly greater than zero and no count floor above it: a count-based
  // tolerance would bound nothing.
  CHECK(both_differing > 0);

  // ── and the two arms are not the same render ──────────────────────────────
  // Stage 2 upsamples, so a stage-1-only artifact cannot equal a two-stage one.
  // This is what proves `max_phase = 0` actually stopped after stage 1 rather
  // than the whole comparison having run twice on the same pixels.
  CHECK(s1_full != both_full);
}

TEST_CASE("ltx2 a2vid: the rebind leaves the DiT where the NEXT generation expects it") {
  // A phase-scoped adapter mutates weights the engine keeps across calls, so the
  // question load-time fusion never had to answer is whether generation N+1 sees
  // what generation N left behind. `Ltx2RebindDitLoras` is driven from the TOP
  // of the phase loop and keys off the checkpoint's own state, so every
  // generation re-establishes stage 1's before it denoises anything.
  //
  // Rendered through the ABI twice on ONE engine, which is the shape a server
  // runs and the shape no single-generation case can see.
  Workspace ws;
  const std::string wav = WriteWav(ws.root + "/take.wav", 2, kFixtureAudioRate, 2.0);
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(A2VidParams(ws.paths, lora));
  REQUIRE(engine != nullptr);

  const vllm::multimodal::VideoResult first =
      engine->Generate(A2VidGen(ws.root + "/gen_a", wav));
  const std::string a = A2VidArtifacts(ws.root + "/gen_a", first);
  const vllm::multimodal::VideoResult second =
      engine->Generate(A2VidGen(ws.root + "/gen_b", wav));
  const std::string b = A2VidArtifacts(ws.root + "/gen_b", second);

  REQUIRE(a.size() > 0);
  // Same request, same seed, same engine: byte-identical. A rebind that left the
  // DiT unfused after the first render would make the second render's stage 2
  // run on base weights, and these would differ.
  CHECK(a == b);
}

// ─── LTX25-TI2VID-RECIPE (#1093) ─────────────────────────────────────────────

namespace {

// A `ti2vid_two_stage` engine on the shipped fixture. Two load-side
// requirements: the spatial upsampler stage 2 needs, which `ConditioningParams`
// supplies, and the distilled adapter upstream's `--distilled-lora
// required=True` demands (utils/args.py:1140-1155).
//
// NO `audio_path`, and its absence is the point rather than an omission. This
// pipeline GENERATES its soundtrack; `a2vid_two_stage` denoises video around a
// take the caller supplies and sets `requires_audio_input` for it.
vllm::multimodal::VideoModelParams Ti2VidParams(const ltx2_fixture::Paths& paths,
                                                const std::string& lora) {
  vllm::multimodal::VideoModelParams mp = ConditioningParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "ti2vid_two_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("ti2vid_two_stage");
  mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
  return mp;
}

// `steps = 2`, because stage 1's schedule is DERIVED from the step count
// (ti2vid_two_stages.py:243-245) and two sigma intervals exercise the loop. The
// STG block list is the fixture's, not this row's: the reduced DiT has TWO
// blocks and the params row this recipe resolves names block 28.
vllm::multimodal::VideoGenParams Ti2VidGen(const std::string& out_dir, int64_t size = 64) {
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.steps = 2;
  gen.height = size;
  gen.width = size;
  OneStageFixtureGuidance(&gen);
  return gen;
}

}  // namespace

TEST_CASE("ltx2 ti2vid: the pipeline renders through vllm.h, guided on the UNADAPTED stage 1") {
  // THE REACHABILITY CLAIM, and it is the point of this case rather than a note
  // beside it. Entry point: `LoadVideoEngine` with a documented value of the
  // documented `pipeline_kind` LOAD extra plus `lora_path`, then `Generate`.
  // Nothing here constructs a recipe, a guider, a phase or a modality by hand.
  // Deleting the `ti2vid_two_stage` dispatch row in `ResolveLtx2PipelineRecipe`
  // REDs this case at the LOAD, which is what separates measuring a capability
  // from measuring a class (.agents/reachability.md).
  //
  // `ltx2-gen --pipeline-kind ti2vid_two_stage --lora-path ... --upsampler-path
  // ...` is the same two calls through the ABI, as a thin client that includes
  // no internal header.
  //
  // AND #928 DOES NOT EXCLUDE THE HTTP ROUTE HERE, unlike on `a2vid_two_stage`.
  // That recipe needs `audio_path`, a PER-GENERATION extra, and
  // `VideoGenParamsFromRequest` writes none. All three knobs THIS recipe needs —
  // `pipeline_kind`, `lora_path`, `upsampler_path` — are LOAD extras, which a
  // server supplies through `--video-extra KEY=VALUE`, and `requires_audio_input`
  // is false, which the recipe case gates.
  //
  // That is a statement about the REQUEST SURFACE and not a second reach claim:
  // no case here drives the HTTP route end to end, so it is not measured and is
  // not asserted.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(Ti2VidParams(ws.paths, lora));
  REQUIRE(engine != nullptr);
  auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  CHECK(ltx->pipeline_kind() == "ti2vid_two_stage");

  const vllm::multimodal::VideoResult result = engine->Generate(Ti2VidGen(ws.root + "/ti2v"));
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);
  CHECK(result.frame_count == 9);
  // The soundtrack is GENERATED and decoded through the vocoder, so it comes
  // back at the BWE arm's rate rather than at the audio VAE's — which is the
  // observable difference from `a2vid_two_stage`, where upstream hands back the
  // caller's own waveform (a2vid_two_stage.py:301-303).
  CHECK(result.sample_rate > 0);

  // ── stage 1 ran upstream's GUIDED denoiser, on all four arms ───────────────
  //
  // AN EVALUATION COUNT CANNOT SEE THIS. A denoiser call is ONE evaluation
  // whether guidance ran or not, so only the FORWARD counters — actual
  // `Ltx2DitForward` calls — distinguish a guided stage 1 from a
  // `SimpleDenoiser` one. The trace's guided fields are recorded at step 0 of
  // phase 0, which is this recipe's stage 1, and its guider is the params
  // table's video row (cfg 3.0, stg 1.0, rescale 0.7, modality 3.0), so all four
  // passes run and the rescale branch — the one term that is NOT invariant
  // between x0 and velocity space — is live on the DEFAULT path.
  REQUIRE_MESSAGE(t.video_guided, "stage 1 did not go through the guided seam at all");
  CHECK(t.video_cond_forwards == 1);
  CHECK(t.video_uncond_forwards == 1);
  CHECK(t.video_perturbed_forwards == 1);
  CHECK(t.video_modality_forwards == 1);
  CHECK(t.video_guidance_cfg_scale == 3.0);
  CHECK(t.video_guidance_stg_scale == 1.0);
  CHECK(t.video_guidance_rescale_scale == 0.7);
  CHECK(t.video_guidance_modality_scale == 3.0);
  CHECK(t.dit_forwards > t.dit_evaluations);

  // ── the four arms are combined in X0 SPACE, not velocity space ────────────
  //
  // `_guided_denoise` converts each pass to x0 BEFORE it combines them
  // (ltx-core utils.py:39-52, `sample - velocity * sigma`). Every LINEAR term is
  // invariant under that change of variable, so cfg, stg and modality cannot
  // see the difference; the RESCALE branch is not invariant, and
  // `rescale_scale` defaults to 0.7 here, so a space error lands on the default
  // path (#1039, #1092).
  //
  // A MAGNITUDE ASSERTION CANNOT GATE IT. On a reduced fixture
  // `std(cond)/std(pred)` is 1.0 to 1e-5 in BOTH spaces. `x0 == latent -
  // sigma*velocity` is an equation between three RECORDED tensors instead: exact
  // in x0 space, and in velocity space `x0` IS the velocity, so the residual
  // becomes the whole sample and `|x0 - velocity|` collapses to exactly 0 —
  // which is what the RED prints.
  const size_t n = t.video_first_latent.size();
  REQUIRE(n > 0);
  const size_t tokens = t.video_first_timesteps.size();
  REQUIRE(tokens > 0);
  const size_t width = n / tokens;
  REQUIRE(width * tokens == n);
  // THE FIXTURE CAN DECIDE THIS AT ALL: the two candidate tensors coincide when
  // the sample is zero. A REQUIRE, because nothing below discriminates once it
  // fails.
  double latent_span = 0.0;
  for (const float x : t.video_first_latent) {
    latent_span = std::max(latent_span, std::abs(static_cast<double>(x)));
  }
  REQUIRE_MESSAGE(latent_span > 1e-3, "the step-0 sample is zero, so the two candidate tensors "
                                      "coincide and nothing below discriminates");

  struct Ti2VidArm {
    const char* name;
    const std::vector<float>& velocity;
    const std::vector<float>& x0;
  };
  const Ti2VidArm arms[] = {
      {"cond", t.video_first_cond_velocity, t.video_first_cond},
      {"uncond", t.video_first_uncond_velocity, t.video_first_uncond},
      {"perturbed", t.video_first_perturbed_velocity, t.video_first_perturbed},
      {"modality", t.video_first_modality_velocity, t.video_first_modality},
  };
  for (const Ti2VidArm& arm : arms) {
    INFO("arm = " << std::string(arm.name));
    REQUIRE(arm.velocity.size() == n);
    REQUIRE(arm.x0.size() == n);
    // PER ARM, because a zeroed velocity makes `to_denoised` the identity on
    // THIS arm alone and would satisfy the equation while proving nothing.
    double velocity_span = 0.0;
    for (const float x : arm.velocity) {
      velocity_span = std::max(velocity_span, std::abs(static_cast<double>(x)));
    }
    REQUIRE_MESSAGE(velocity_span > 1e-6, "this arm's velocity is zero, so the equation below "
                                          "holds for a reason that is not the one it tests");
    // `x0 = latent - sigma_token * velocity` (model.py:590-604), with the
    // PER-TOKEN timestep and not the schedule scalar.
    double residual = 0.0;
    double against_velocity = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double sigma = static_cast<double>(t.video_first_timesteps[i / width]);
      const double expected = static_cast<double>(t.video_first_latent[i]) -
                              sigma * static_cast<double>(arm.velocity[i]);
      residual = std::max(residual, std::abs(static_cast<double>(arm.x0[i]) - expected));
      against_velocity = std::max(
          against_velocity,
          std::abs(static_cast<double>(arm.x0[i]) - static_cast<double>(arm.velocity[i])));
    }
    INFO("max|x0 - (latent - sigma*v)| = " << residual);
    INFO("max|x0 - velocity| = " << against_velocity);
    CHECK(residual < 1e-4);
    CHECK(against_velocity > 1e-6);
  }
}

TEST_CASE("ltx2 ti2vid: stage 1's sigma shift takes the 4096 anchor, not the target grid") {
  // THE DIVERGENCE THIS ROW RESOLVED, gated through the production entry point
  // rather than on the recipe struct — the recipe case proves the field is SET,
  // and this one proves it is CONSUMED (#1013).
  //
  // `LTX2Scheduler.execute` takes an OPTIONAL latent and `schedulers.py:31` is
  // `tokens = math.prod(latent.shape[2:]) if latent is not None else
  // default_number_of_tokens`, with `default_number_of_tokens` = MAX_SHIFT_ANCHOR
  // = 4096 (`:11`, `:29`). `ti2vid_two_stages.py:243-245` passes NO latent;
  // `ti2vid_two_stages_hq.py:267` — our `res2s_two_stage` — passes
  // `latent=empty_latent` and is the ONE upstream site that does.
  //
  // A 2x2 OVER (recipe, geometry), because neither half alone is load-bearing.
  // The equalities alone pass on a build that hard-codes 4096 for everything;
  // the inequalities alone pass on today's `target_tokens`-everywhere tree. Only
  // the pair says that the anchor is per-phase AND selected correctly.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  // Two geometries whose stage-1 target grids differ. Both divide 64, which is
  // `Ltx2AssertResolution`'s divisor on a `spatial_downscale = 2` recipe.
  const int64_t kSmall = 64;
  const int64_t kLarge = 128;

  // TWO NUMBERS OUT OF EACH RENDER, AND THE SECOND ONE IS WHAT KEEPS THE
  // TRAJECTORY HALF OF THIS CASE HONEST. The recomputations at the end run at
  // the step count the RENDER used. Restating that count as a literal down
  // there decouples it from `gen.steps`: lowering the render to 2 steps would
  // then leave every assertion in this case green while making the trajectory
  // claim vacuous, which is precisely the degeneracy the comment beside
  // `gen.steps` warns about.
  struct Rendered {
    int64_t schedule_tokens;
    int64_t steps;
  };

  auto rendered_for = [&](const char* kind, int64_t size,
                          const std::string& tag) -> Rendered {
    vllm::multimodal::VideoModelParams mp = Ti2VidParams(ws.paths, lora);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    vllm::multimodal::VideoGenParams gen = Ti2VidGen(ws.root + "/" + tag, size);
    // THREE STEPS, NOT THE FIXTURE'S TWO, and the reason is measured below: the
    // scheduler's `stretch` pins sigma[0] at 1.0 and the LAST non-zero sigma at
    // `terminal` = 0.1 (schedulers.py:48-55), so a 2-step schedule is
    // {1, 0.1, 0} for EVERY token count and the anchor cannot reach the
    // trajectory at all. Three steps is the shortest schedule with an interior
    // sigma for the shift to move. Lowering this number is REFUSED rather than
    // deprecated: it comes back out of this lambda and the trajectory
    // assertions recompute at it, so 2 here reds them by name.
    gen.steps = 3;
    // `ti2vid_two_stage` resolves `stg_blocks = [28]` (constants.py:86-87) and
    // this fixture's DiT has TWO blocks, so its perturbed pass is refused by
    // name unless the request names a block that exists. The HQ preset ships
    // `stg_blocks = []` beside `stg_scale = 0.0` (constants.py:105, :113), asks
    // for no perturbed pass at all, and FIXES its stage-2 guidance — so giving
    // it the same override is refused outright. `Ti2VidGen` carries the
    // override, so the control has to take it back off.
    if (std::string(kind) != "ti2vid_two_stage") {
      gen.extras.erase(vllm::multimodal::kLtx2VideoStgBlocksExtra);
      gen.extras.erase(vllm::multimodal::kLtx2AudioStgBlocksExtra);
    }
    (void)engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    // Written only on the branch that CALLS `Ltx2SigmaSchedule`, so a phase
    // carrying frozen sigmas leaves it 0. Stage 2 of both recipes does, which is
    // why what lands here is stage 1's.
    REQUIRE(t.schedule_tokens > 0);
    return Rendered{t.schedule_tokens, gen.steps};
  };

  const Rendered ti_small_r = rendered_for("ti2vid_two_stage", kSmall, "ti_small");
  const Rendered ti_large_r = rendered_for("ti2vid_two_stage", kLarge, "ti_large");
  const Rendered hq_small_r = rendered_for("res2s_two_stage", kSmall, "hq_small");
  const Rendered hq_large_r = rendered_for("res2s_two_stage", kLarge, "hq_large");

  const int64_t ti_small = ti_small_r.schedule_tokens;
  const int64_t ti_large = ti_large_r.schedule_tokens;
  const int64_t hq_small = hq_small_r.schedule_tokens;
  const int64_t hq_large = hq_large_r.schedule_tokens;

  // The step count the four renders ACTUALLY ran at, read back out of them
  // rather than restated below. All four have to agree, or "the step count" is
  // not one number and nothing below can be recomputed at it.
  const int64_t rendered_steps = ti_small_r.steps;
  REQUIRE(ti_large_r.steps == rendered_steps);
  REQUIRE(hq_small_r.steps == rendered_steps);
  REQUIRE(hq_large_r.steps == rendered_steps);

  MESSAGE("ti2vid: " << ti_small << " / " << ti_large << "   res2s: " << hq_small << " / "
                     << hq_large);

  // ── this arm's schedule is RESOLUTION-INDEPENDENT, at upstream's constant ──
  const int64_t anchor = vllm::Ltx2SchedulerParams{}.default_number_of_tokens;
  CHECK(anchor == 4096);  // schedulers.py:11 — pinned, not read back from the build
  CHECK_MESSAGE(ti_small == anchor,
                "stage 1's sigma shift was fitted on " << ti_small << " tokens, but upstream "
                    "passes no latent (ti2vid_two_stages.py:243-245) and gets " << anchor);
  CHECK(ti_large == anchor);

  // ── and the HQ arm's is NOT, which is what stops the above being a constant ─
  CHECK_MESSAGE(hq_small != hq_large,
                "the res_2s arm reported the same anchor at two resolutions, so this fixture "
                "cannot tell the two branches apart and the equalities above prove nothing");
  CHECK(hq_small != anchor);
  CHECK(hq_large != anchor);

  // ── and the two anchors really do produce DIFFERENT sigmas ────────────────
  //
  // The strongest half: a claim about the trajectory rather than about the
  // counter that reports it. `sigma_shift = tokens*mm + b` (schedulers.py:35-39),
  // so two token counts give two schedules — unless the shift arithmetic has
  // been flattened, in which case selecting the anchor would be inert and every
  // assertion above would still pass.
  // AT `rendered_steps`, NOT AT A LITERAL. The step count comes back out of the
  // renders above, so this comparison cannot drift away from what they sampled.
  // A literal here would let someone lower `gen.steps` and keep this green.
  const std::vector<float> at_anchor = vllm::Ltx2SigmaSchedule(rendered_steps, anchor);
  const std::vector<float> at_target = vllm::Ltx2SigmaSchedule(rendered_steps, hq_small);
  REQUIRE(at_anchor.size() == at_target.size());
  CHECK_MESSAGE(at_anchor != at_target,
                "the 4096 anchor and the target grid produce the SAME schedule on this fixture, "
                "so nothing above measures which one was taken");

  // AND THE RENDER'S STEP COUNT IS LOAD-BEARING, WHICH IS WORTH AN ASSERTION
  // RATHER THAN A COMMENT. `stretch` renormalises so that sigma[0] is 1.0 and
  // the LAST non-zero sigma is exactly `terminal` = 0.1 (schedulers.py:48-55).
  // A 2-step schedule has only those two non-zero entries, so it is {1, 0.1, 0}
  // for EVERY token count and the shift is entirely absorbed. This case was
  // written at the fixture's usual 2 steps and went RED here, on this line's
  // ancestor — the counters were already correct and the trajectory claim was
  // vacuous.
  //
  // TWO ASSERTIONS, BECAUSE THEY FAIL FOR DIFFERENT REASONS. The first names
  // the render's own step count, so lowering `gen.steps` to make this case
  // faster fails HERE, by name, rather than silently turning the comparison
  // above into a value against itself. The second pins the degeneracy itself,
  // so it fails if a scheduler change ever makes a 2-step schedule
  // token-dependent and this whole paragraph stops being true.
  CHECK_MESSAGE(rendered_steps > 2,
                "the renders above ran at " << rendered_steps << " steps, and a schedule that "
                "short is {1, 0.1, 0} for EVERY token count (schedulers.py:48-55), so the "
                "trajectory comparison above compares a value with itself");
  CHECK_MESSAGE(vllm::Ltx2SigmaSchedule(/*steps=*/2, anchor) ==
                    vllm::Ltx2SigmaSchedule(/*steps=*/2, hq_small),
                "a 2-step schedule now DOES depend on the token count, so the stretch no longer "
                "pins both of its non-zero sigmas and the comment above is wrong");
  CHECK(vllm::Ltx2SigmaSchedule(/*steps=*/2, anchor).size() == 3u);

  // `Ltx2SigmaSchedule` reads 0 as "take the default", so the concrete 4096 the
  // engine now passes and the sentinel are the same schedule. Pinned because the
  // engine deliberately passes the concrete value, so that the trace reports an
  // anchor rather than a sentinel.
  CHECK(vllm::Ltx2SigmaSchedule(rendered_steps, 0) == at_anchor);
}

TEST_CASE("ltx2 ti2vid: the distilled-LoRA requirement refuses BY WHAT IS MISSING") {
  // `--distilled-lora` is `required=True` (utils/args.py:1140-1155) on the
  // parser ti2vid_two_stages.py:319 selects, and stage 2's three-sigma
  // refinement (`:178`) is what that adapter was trained for. Without it the
  // render FINISHES: a clip of the right size, frame count and sample rate, with
  // a distilled schedule run on undistilled weights.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  vllm::multimodal::VideoModelParams mp = Ti2VidParams(ws.paths, lora);
  mp.extras.erase(vllm::multimodal::kLtx2LoraPathExtra);
  try {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    FAIL_CHECK("a ti2vid load with no distilled LoRA must be refused");
  } catch (const std::exception& e) {
    const std::string message = e.what();
    INFO("message = " << message);
    // It names the PIPELINE that was asked for, which is what makes the refusal
    // actionable rather than generic.
    CHECK(message.find("ti2vid_two_stage") != std::string::npos);
    CHECK(message.find("distilled LoRA") != std::string::npos);
    CHECK(message.find("lora_path") != std::string::npos);
    // ...and the anchor it cites is the SHARED parser, not `a2vid_two_stage.py`.
    // This recipe is the second user of `requires_distilled_lora`, so it is the
    // first caller the old hard-coded anchors would have pointed at the wrong
    // pipeline's source (#1151).
    CHECK(message.find("default_2_stage_arg_parser") != std::string::npos);
    CHECK(message.find("a2vid_two_stage.py") == std::string::npos);
  }

  // THE CONTROL: the same load WITH the adapter renders, so the case is about
  // the requirement and not about any load failure.
  CHECK_NOTHROW((void)vllm::multimodal::LoadVideoEngine(Ti2VidParams(ws.paths, lora)));
  // THE SECOND CONTROL: the DEFAULT kind is fine without an adapter, so this is
  // this recipe's requirement and not a new global one.
  CHECK_NOTHROW((void)vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths)));

  // ── and NO audio take is demanded, which `a2vid_two_stage` does ────────────
  //
  // The two recipes come off the same parser and share `requires_distilled_lora`,
  // so a recipe written by copying that one would inherit `requires_audio_input`
  // and refuse every render. There is no `--audio-path` on this pipeline:
  // ti2vid_two_stages.py:159-181 takes `images`, not a waveform.
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(Ti2VidParams(ws.paths, lora));
  CHECK_NOTHROW((void)engine->Generate(Ti2VidGen(ws.root + "/no_take")));
}

// ─── LTX25-KEYFRAME-INTERP (#1096) ───────────────────────────────────────────

namespace {

// A `keyframe_interpolation` engine on the shipped fixture. The same two
// load-side requirements `ti2vid_two_stage` has: the spatial upsampler stage 2
// needs, which `ConditioningParams` supplies, and the distilled adapter, which
// upstream makes a POSITIONAL non-defaulted parameter
// (keyframe_interpolation.py:68) as well as `--distilled-lora required=True`
// (utils/args.py:1140-1155).
//
// NO `audio_path`: this pipeline generates its soundtrack, and unlike
// `ti2vid_two_stage` the take that leaves is STAGE 2's (`:271`, `:293`).
vllm::multimodal::VideoModelParams KeyframeParams(const ltx2_fixture::Paths& paths,
                                                  const std::string& lora) {
  vllm::multimodal::VideoModelParams mp = ConditioningParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "keyframe_interpolation";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] =
      FixtureCheckpointClass("keyframe_interpolation");
  mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = lora;
  return mp;
}

// `steps = 2`, because stage 1's schedule is DERIVED from the step count
// (keyframe_interpolation.py:199-200) and two sigma intervals exercise the loop.
// The case that gates the schedule ANCHOR raises this to 3 and says why. The STG
// block list is the fixture's, not this row's: the reduced DiT has TWO blocks
// and the params row this recipe resolves names block 28.
vllm::multimodal::VideoGenParams KeyframeGen(const std::string& out_dir, int64_t size = 64) {
  vllm::multimodal::VideoGenParams gen = FixtureGen(out_dir);
  gen.steps = 2;
  gen.height = size;
  gen.width = size;
  OneStageFixtureGuidance(&gen);
  return gen;
}

}  // namespace

TEST_CASE("ltx2 keyframe: the first frame is a KEYFRAME that APPENDS, not a latent that replaces") {
  // THE ROW'S IDENTITY, and the one thing that separates
  // `image_conditionings_by_adding_guiding_latent` (helpers.py:343-367) from
  // `combined_image_conditionings` (:272-308): the second sends `frame_idx == 0`
  // to `VideoConditionByLatentIndex`, which REPLACES latent frame 0's clean
  // tokens and never changes the token count (latent_cond.py:38-39); the first
  // has NO branch and sends it to `VideoConditionByKeyframeIndex`, which APPENDS
  // a latent frame of tokens (keyframe_cond.py:79-82).
  //
  // GATED THROUGH `LoadVideoEngine` + `Generate`, not on the recipe struct: the
  // recipe case in test_ltx2_pipeline proves the field is SET, and this one
  // proves it is CONSUMED (#1013).
  //
  // AND NO PIXEL COMPARISON CAN SEE IT. Both arms condition on the same image,
  // both return a clip of the right size, the right frame count and the right
  // sample rate. The only observable is the sequence LENGTH the DiT ran over,
  // which is `Ltx2ConditioningTrace::video_tokens`, recorded before the trim.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  struct Conditioned {
    int64_t video_tokens;
    int64_t image_tokens;
    std::string bytes;
  };

  // ONE image, ONE geometry, ONE seed. The only thing that differs between the
  // two loads is `pipeline_kind`, so nothing but the builder can move the
  // numbers below.
  auto render = [&](const char* kind, bool with_image, const std::string& tag) -> Conditioned {
    vllm::multimodal::VideoModelParams mp = KeyframeParams(ws.paths, lora);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    vllm::multimodal::VideoGenParams gen = KeyframeGen(ws.root + "/" + tag);
    if (with_image) {
      gen.first_frame_ppm = ConditioningPpm(20, 28, 1);
      // CRF 0 EXPLICITLY, because an LTX-2.5 checkpoint RESOLVES 18 when the
      // caller leaves it unset (`ImageConditioner.resolve_crf`, blocks.py:977-983)
      // and the H.264 round trip that 18 needs is refused by name in this tree.
      // Every image-conditioning case here asks for 0 for the same reason; it is
      // orthogonal to which builder places the image.
      gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    }
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    std::string all;
    for (int64_t f = 0; f < result.frame_count; ++f) {
      char name[64];
      std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(f));
      all += ReadAll(gen.output_dir + name);
    }
    return Conditioned{t.video_tokens, t.image_tokens, all};
  };

  const Conditioned kf = render("keyframe_interpolation", /*with_image=*/true, "kf");
  const Conditioned ti = render("ti2vid_two_stage", /*with_image=*/true, "ti");
  // THE THIRD RENDER, and without it the equality below passes on a tree where
  // BOTH arms append: it pins what the bare target grid is on this geometry, so
  // "the plain arm placed its image without growing the sequence" is a claim
  // about the grid rather than a comparison of two unknowns.
  const Conditioned bare = render("ti2vid_two_stage", /*with_image=*/false, "bare");
  const Conditioned kf_bare = render("keyframe_interpolation", /*with_image=*/false, "kf_bare");

  MESSAGE("video_tokens  keyframe " << kf.video_tokens << "  ti2vid " << ti.video_tokens
                                    << "  bare " << bare.video_tokens);

  // NEITHER ARM PLACED NOTHING. A build that encoded the image and never
  // conditioned on it would satisfy every count comparison below by making them
  // all equal, so this is a REQUIRE.
  REQUIRE(kf.image_tokens > 0);
  REQUIRE(ti.image_tokens > 0);
  // The same image at the same geometry encodes to the same number of tokens
  // under both builders — only WHERE they go differs.
  CHECK(kf.image_tokens == ti.image_tokens);

  // ── the plain arm REPLACED: its sequence is the bare target grid ───────────
  CHECK_MESSAGE(ti.video_tokens == bare.video_tokens,
                "`combined_image_conditionings` sends frame 0 to "
                "`VideoConditionByLatentIndex`, which replaces tokens that already exist "
                "(latent_cond.py:38-39), so the count must not move — it went from "
                    << bare.video_tokens << " to " << ti.video_tokens);
  CHECK(kf_bare.video_tokens == bare.video_tokens);

  // ── and this arm APPENDED, by exactly one image's worth of tokens ──────────
  CHECK_MESSAGE(kf.video_tokens == bare.video_tokens + kf.image_tokens,
                "`image_conditionings_by_adding_guiding_latent` has no frame-0 branch "
                "(helpers.py:343-367), so frame 0 takes `VideoConditionByKeyframeIndex` and "
                "APPENDS (keyframe_cond.py:79-82). Expected "
                    << bare.video_tokens + kf.image_tokens << " tokens, got " << kf.video_tokens);
  CHECK(kf.video_tokens > ti.video_tokens);

  // ── and the render CHANGED, which is what a token count alone cannot say ───
  //
  // Against the SAME kind with no image, deliberately, and not against the plain
  // arm: the two arms place the same content, so a pixel difference between THEM
  // says only that they placed it differently, which the counts above already
  // say with a reason. What this adds is that the conditioning reached the
  // pixels at all on this arm.
  CHECK_MESSAGE(kf.bytes != kf_bare.bytes,
                "the appended keyframe left the rendered clip unchanged, so it was placed into "
                "a state the decode never read");
  CHECK_FALSE(kf.bytes.empty());
}

TEST_CASE("ltx2 keyframe: BOTH ends pin at once - two appends, and each is located at its own") {
  // THE PIPELINE'S HEADLINE REQUEST, and it aborted the render
  // ([#1219](https://github.com/mudler/vllm.cpp/issues/1219)).
  // `KeyframeInterpolationPipeline` exists to generate the motion BETWEEN
  // pinned keyframes; `docs/USAGE.md`'s worked example for this kind passes
  // `--first-frame open.ppm --last-frame close.ppm`, and `ltx2-gen --help` says
  // to use the two together. So this is not an edge case, it is the case.
  //
  // WHAT BROKE, and it is a shape worth keeping in view: the last-frame arm
  // located its own appended tokens at `positions[target_tokens * 2]`, which is
  // the first token PAST THE TARGET GRID — correct only while that arm owned the
  // first append. Row LTX25-KEYFRAME-INTERP put a second appending item in front
  // of it on the `image_conditionings_by_adding_guiding_latent` recipes, so the
  // index then named the FIRST frame's keyframe at temporal 0, and the arm's own
  // positional assertion threw. A derived index that was right by ORDER rather
  // than by construction, and the item that changed the order was two hundred
  // lines above it.
  //
  // THE CONTROL IS THE OTHER BUILDER, and it is what makes this case say
  // "ordering" rather than "keyframes". `ti2vid_two_stage` takes the same two
  // images through `combined_image_conditionings`, whose frame-0 item REPLACES
  // (helpers.py:295-300), so it has exactly ONE append and never reached the
  // defect. It stayed green throughout and must stay green here.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/both.safetensors", kFixtureLoraTarget, 1.0F);

  const std::string closing = ws.root + "/closing.ppm";
  WriteBytes(closing, ConditioningPpm(20, 28, 41));

  struct Pinned {
    int64_t video_tokens;
    int64_t image_tokens;
  };

  // `ends` selects which of the two slots are filled, so one lambda produces the
  // bare grid, each single end, and both — and the counts below are then
  // relations between renders of one geometry rather than literals.
  auto render = [&](const char* kind, bool first, bool last, const std::string& tag) -> Pinned {
    vllm::multimodal::VideoModelParams mp = KeyframeParams(ws.paths, lora);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    vllm::multimodal::VideoGenParams gen = KeyframeGen(ws.root + "/" + tag);
    if (first) gen.first_frame_ppm = ConditioningPpm(20, 28, 40);
    if (last) gen.last_frame_path = closing;
    // One CRF for both slots, which is the surface: upstream resolves the CRF
    // once for the whole `images` list (blocks.py:966-983), and the H.264 round
    // trip an LTX-2.5 checkpoint would otherwise resolve is refused by name here.
    if (first || last) gen.extras[vllm::multimodal::kLtx2ImageCrfExtra] = "0";
    const vllm::multimodal::VideoResult result = engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    // The artifact, not only the trace: the render has to come back at the
    // requested length, because the appended tokens are trimmed back off before
    // the unpatchify and a clip of the wrong length is the visible half of a
    // trim that read the wrong count.
    CHECK(result.frame_count == 9);
    return Pinned{t.video_tokens, t.image_tokens};
  };

  // ── THE CONTROL, FIRST, so a red below is attributable ─────────────────────
  //
  // Replace + one append. This is the arm that was already green, and running it
  // first means a failure here says "both arms are broken" rather than letting
  // the new arm's red stand for the pair.
  const Pinned ti_both = render("ti2vid_two_stage", true, true, "ti_both");
  const Pinned ti_bare = render("ti2vid_two_stage", false, false, "ti_bare");
  REQUIRE(ti_both.image_tokens > 0);
  CHECK_MESSAGE(ti_both.video_tokens == ti_bare.video_tokens + ti_both.image_tokens,
                "`combined_image_conditionings` REPLACES at frame 0 and APPENDS at the closing "
                "frame, so pinning both ends grows the sequence by exactly ONE image — expected "
                    << ti_bare.video_tokens + ti_both.image_tokens << ", got "
                    << ti_both.video_tokens);

  // ── AND THE ARM THIS ROW ADDS: two appends, both located ───────────────────
  const Pinned kf_both = render("keyframe_interpolation", true, true, "kf_both");
  const Pinned kf_bare = render("keyframe_interpolation", false, false, "kf_bare");
  const Pinned kf_first = render("keyframe_interpolation", true, false, "kf_first");
  const Pinned kf_last = render("keyframe_interpolation", false, true, "kf_last");

  MESSAGE("video_tokens  bare " << kf_bare.video_tokens << "  first " << kf_first.video_tokens
                                << "  last " << kf_last.video_tokens << "  both "
                                << kf_both.video_tokens << "  image_tokens "
                                << kf_both.image_tokens);

  REQUIRE(kf_both.image_tokens > 0);
  CHECK(kf_bare.video_tokens == ti_bare.video_tokens);
  // Each end alone appends one image's worth...
  CHECK(kf_first.video_tokens == kf_bare.video_tokens + kf_both.image_tokens);
  CHECK(kf_last.video_tokens == kf_bare.video_tokens + kf_both.image_tokens);
  // ...and the two together append BOTH, which is the count the defect could
  // never produce because the render never finished.
  CHECK_MESSAGE(kf_both.video_tokens == kf_bare.video_tokens + 2 * kf_both.image_tokens,
                "`image_conditionings_by_adding_guiding_latent` has no frame-0 branch "
                "(helpers.py:343-367), so BOTH pinned ends take `VideoConditionByKeyframeIndex` "
                "and append (keyframe_cond.py:79-82) — expected "
                    << kf_bare.video_tokens + 2 * kf_both.image_tokens << ", got "
                    << kf_both.video_tokens);
  // ...and it is strictly more than the builder that replaces at frame 0, which
  // is what the two arms differ by.
  CHECK(kf_both.video_tokens > ti_both.video_tokens);
}

TEST_CASE("ltx2 keyframe: the pipeline renders through vllm.h, guided on the UNADAPTED stage 1") {
  // THE REACHABILITY CLAIM, and it is the point of this case rather than a note
  // beside it. Entry point: `LoadVideoEngine` with a documented value of the
  // documented `pipeline_kind` LOAD extra plus `lora_path`, then `Generate`.
  // Nothing here constructs a recipe, a guider, a phase or a modality by hand.
  // Deleting the `keyframe_interpolation` dispatch row in
  // `ResolveLtx2PipelineRecipe` REDs this case at the LOAD, which is what
  // separates measuring a capability from measuring a class
  // (.agents/reachability.md).
  //
  // `ltx2-gen --pipeline-kind keyframe_interpolation --lora-path ...
  // --upsampler-path ...` is the same two calls through the ABI, as a thin
  // client that includes no internal header.
  //
  // AND #928 DOES NOT EXCLUDE THE HTTP ROUTE HERE, unlike on `a2vid_two_stage`.
  // That recipe needs `audio_path`, a PER-GENERATION extra, and
  // `VideoGenParamsFromRequest` writes none. All three knobs THIS recipe needs —
  // `pipeline_kind`, `lora_path`, `upsampler_path` — are LOAD extras, which a
  // server supplies through `--video-extra KEY=VALUE`, and `requires_audio_input`
  // is false, which the recipe case gates. That is a statement about the REQUEST
  // SURFACE and not a second reach claim: no case here drives the HTTP route end
  // to end, so it is not measured and is not asserted.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(KeyframeParams(ws.paths, lora));
  REQUIRE(engine != nullptr);
  auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  CHECK(ltx->pipeline_kind() == "keyframe_interpolation");

  const vllm::multimodal::VideoResult result = engine->Generate(KeyframeGen(ws.root + "/kf"));
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);
  CHECK(result.frame_count == 9);
  // The soundtrack is GENERATED and decoded through the vocoder, so it comes
  // back at the BWE arm's rate rather than at the audio VAE's.
  CHECK(result.sample_rate > 0);

  // ── stage 1 ran upstream's GUIDED denoiser, on all four arms ───────────────
  //
  // AN EVALUATION COUNT CANNOT SEE THIS. A denoiser call is ONE evaluation
  // whether guidance ran or not, so only the FORWARD counters — actual
  // `Ltx2DitForward` calls — distinguish a guided stage 1 from a
  // `SimpleDenoiser` one. The trace's guided fields are recorded at step 0 of
  // phase 0, which is this recipe's stage 1, and its guider is the params
  // table's video row (cfg 3.0, stg 1.0, rescale 0.7, modality 3.0), so all four
  // passes run and the rescale branch — the one term that is NOT invariant
  // between x0 and velocity space — is live on the DEFAULT path.
  REQUIRE_MESSAGE(t.video_guided, "stage 1 did not go through the guided seam at all");
  CHECK(t.video_cond_forwards == 1);
  CHECK(t.video_uncond_forwards == 1);
  CHECK(t.video_perturbed_forwards == 1);
  CHECK(t.video_modality_forwards == 1);
  CHECK(t.video_guidance_cfg_scale == 3.0);
  CHECK(t.video_guidance_stg_scale == 1.0);
  CHECK(t.video_guidance_rescale_scale == 0.7);
  CHECK(t.video_guidance_modality_scale == 3.0);
  CHECK(t.dit_forwards > t.dit_evaluations);

  // ── the four arms are combined in X0 SPACE, not velocity space ────────────
  //
  // `_guided_denoise` converts each pass to x0 BEFORE it combines them
  // (ltx-core utils.py:39-52, `sample - velocity * sigma`). Every LINEAR term is
  // invariant under that change of variable, so cfg, stg and modality cannot see
  // the difference; the RESCALE branch is not invariant, and `rescale_scale`
  // defaults to 0.7 here, so a space error lands on the default path
  // (#1039, #1092).
  //
  // A MAGNITUDE ASSERTION CANNOT GATE IT. On a reduced fixture
  // `std(cond)/std(pred)` is 1.0 to 1e-5 in BOTH spaces. `x0 == latent -
  // sigma*velocity` is an equation between three RECORDED tensors instead: exact
  // in x0 space, and in velocity space `x0` IS the velocity, so the residual
  // becomes the whole sample and `|x0 - velocity|` collapses to exactly 0 —
  // which is what the RED prints.
  const size_t n = t.video_first_latent.size();
  REQUIRE(n > 0);
  const size_t tokens = t.video_first_timesteps.size();
  REQUIRE(tokens > 0);
  const size_t width = n / tokens;
  REQUIRE(width * tokens == n);
  // THE FIXTURE CAN DECIDE THIS AT ALL: the two candidate tensors coincide when
  // the sample is zero. A REQUIRE, because nothing below discriminates once it
  // fails.
  double latent_span = 0.0;
  for (const float x : t.video_first_latent) {
    latent_span = std::max(latent_span, std::abs(static_cast<double>(x)));
  }
  REQUIRE_MESSAGE(latent_span > 1e-3, "the step-0 sample is zero, so the two candidate tensors "
                                      "coincide and nothing below discriminates");

  struct KeyframeArm {
    const char* name;
    const std::vector<float>& velocity;
    const std::vector<float>& x0;
  };
  const KeyframeArm arms[] = {
      {"cond", t.video_first_cond_velocity, t.video_first_cond},
      {"uncond", t.video_first_uncond_velocity, t.video_first_uncond},
      {"perturbed", t.video_first_perturbed_velocity, t.video_first_perturbed},
      {"modality", t.video_first_modality_velocity, t.video_first_modality},
  };
  for (const KeyframeArm& arm : arms) {
    INFO("arm = " << std::string(arm.name));
    REQUIRE(arm.velocity.size() == n);
    REQUIRE(arm.x0.size() == n);
    // PER ARM, because a zeroed velocity makes `to_denoised` the identity on
    // THIS arm alone and would satisfy the equation while proving nothing.
    double velocity_span = 0.0;
    for (const float x : arm.velocity) {
      velocity_span = std::max(velocity_span, std::abs(static_cast<double>(x)));
    }
    REQUIRE_MESSAGE(velocity_span > 1e-6, "this arm's velocity is zero, so the equation below "
                                          "holds for a reason that is not the one it tests");
    // `x0 = latent - sigma_token * velocity` (model.py:590-604), with the
    // PER-TOKEN timestep and not the schedule scalar.
    double residual = 0.0;
    double against_velocity = 0.0;
    for (size_t i = 0; i < n; ++i) {
      const double sigma = static_cast<double>(t.video_first_timesteps[i / width]);
      const double expected = static_cast<double>(t.video_first_latent[i]) -
                              sigma * static_cast<double>(arm.velocity[i]);
      residual = std::max(residual, std::abs(static_cast<double>(arm.x0[i]) - expected));
      against_velocity = std::max(
          against_velocity,
          std::abs(static_cast<double>(arm.x0[i]) - static_cast<double>(arm.velocity[i])));
    }
    INFO("max|x0 - (latent - sigma*v)| = " << residual);
    INFO("max|x0 - velocity| = " << against_velocity);
    CHECK(residual < 1e-4);
    CHECK(against_velocity > 1e-6);
  }
}

TEST_CASE("ltx2 keyframe: stage 1's sigma shift takes the 4096 anchor, not the target grid") {
  // `LTX2Scheduler.execute` takes an OPTIONAL latent and `schedulers.py:31` is
  // `tokens = math.prod(latent.shape[2:]) if latent is not None else
  // default_number_of_tokens`, with `default_number_of_tokens` = MAX_SHIFT_ANCHOR
  // = 4096 (`:11`, `:29`). `keyframe_interpolation.py:199-200` passes NO latent;
  // `ti2vid_two_stages_hq.py:267` — our `res2s_two_stage` — passes
  // `latent=empty_latent` and is the ONE upstream site that does.
  //
  // A 2x2 OVER (recipe, geometry), because neither half alone is load-bearing.
  // The equalities alone pass on a build that hard-codes 4096 for everything;
  // the inequalities alone pass on a `target_tokens`-everywhere tree. Only the
  // pair says that the anchor is per-phase AND selected correctly.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  // Two geometries whose stage-1 target grids differ. Both divide 64, which is
  // `Ltx2AssertResolution`'s divisor on a `spatial_downscale = 2` recipe.
  const int64_t kSmall = 64;
  const int64_t kLarge = 128;

  // TWO NUMBERS OUT OF EACH RENDER, AND THE SECOND ONE IS WHAT KEEPS THE
  // TRAJECTORY HALF OF THIS CASE HONEST. The recomputations at the end run at
  // the step count the RENDER used. Restating that count as a literal down there
  // decouples it from `gen.steps`: lowering the render to 2 steps would then
  // leave every assertion in this case green while making the trajectory claim
  // vacuous. That is not hypothetical — `ltx25-ti2vid-recipe.md` shipped this
  // case with a literal and a fresh review found the mutation undetected.
  struct Rendered {
    int64_t schedule_tokens;
    int64_t steps;
  };

  auto rendered_for = [&](const char* kind, int64_t size, const std::string& tag) -> Rendered {
    vllm::multimodal::VideoModelParams mp = KeyframeParams(ws.paths, lora);
    mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
    mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass(kind);
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    auto* ltx = dynamic_cast<vllm::multimodal::Ltx2VideoEngine*>(engine.get());
    REQUIRE(ltx != nullptr);
    vllm::multimodal::VideoGenParams gen = KeyframeGen(ws.root + "/" + tag, size);
    // THREE STEPS, NOT THE FIXTURE'S TWO, and the reason is asserted below: the
    // scheduler's `stretch` pins sigma[0] at 1.0 and the LAST non-zero sigma at
    // `terminal` = 0.1 (schedulers.py:48-55), so a 2-step schedule is
    // {1, 0.1, 0} for EVERY token count and the anchor cannot reach the
    // trajectory at all. Three steps is the shortest schedule with an interior
    // sigma for the shift to move. Lowering this number is REFUSED rather than
    // deprecated: it comes back out of this lambda and the trajectory assertions
    // recompute at it, so 2 here reds them by name.
    gen.steps = 3;
    // `keyframe_interpolation` resolves `stg_blocks = [28]` (constants.py:86-87)
    // and this fixture's DiT has TWO blocks, so its perturbed pass is refused by
    // name unless the request names a block that exists. The HQ preset ships
    // `stg_blocks = []` beside `stg_scale = 0.0` (constants.py:105, :113), asks
    // for no perturbed pass at all, and FIXES its stage-2 guidance — so giving
    // it the same override is refused outright. `KeyframeGen` carries the
    // override, so the control has to take it back off.
    if (std::string(kind) != "keyframe_interpolation") {
      gen.extras.erase(vllm::multimodal::kLtx2VideoStgBlocksExtra);
      gen.extras.erase(vllm::multimodal::kLtx2AudioStgBlocksExtra);
    }
    (void)engine->Generate(gen);
    const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
    REQUIRE(t.completed);
    // Written only on the branch that CALLS `Ltx2SigmaSchedule`, so a phase
    // carrying frozen sigmas leaves it 0. Stage 2 of both recipes does, which is
    // why what lands here is stage 1's.
    REQUIRE(t.schedule_tokens > 0);
    return Rendered{t.schedule_tokens, gen.steps};
  };

  const Rendered kf_small_r = rendered_for("keyframe_interpolation", kSmall, "kf_small");
  const Rendered kf_large_r = rendered_for("keyframe_interpolation", kLarge, "kf_large");
  const Rendered hq_small_r = rendered_for("res2s_two_stage", kSmall, "hq_small");
  const Rendered hq_large_r = rendered_for("res2s_two_stage", kLarge, "hq_large");

  const int64_t kf_small = kf_small_r.schedule_tokens;
  const int64_t kf_large = kf_large_r.schedule_tokens;
  const int64_t hq_small = hq_small_r.schedule_tokens;
  const int64_t hq_large = hq_large_r.schedule_tokens;

  // The step count the four renders ACTUALLY ran at, read back out of them
  // rather than restated below. All four have to agree, or "the step count" is
  // not one number and nothing below can be recomputed at it.
  const int64_t rendered_steps = kf_small_r.steps;
  REQUIRE(kf_large_r.steps == rendered_steps);
  REQUIRE(hq_small_r.steps == rendered_steps);
  REQUIRE(hq_large_r.steps == rendered_steps);

  MESSAGE("keyframe: " << kf_small << " / " << kf_large << "   res2s: " << hq_small << " / "
                       << hq_large);

  // ── this arm's schedule is RESOLUTION-INDEPENDENT, at upstream's constant ──
  const int64_t anchor = vllm::Ltx2SchedulerParams{}.default_number_of_tokens;
  CHECK(anchor == 4096);  // schedulers.py:11 — pinned, not read back from the build
  CHECK_MESSAGE(kf_small == anchor,
                "stage 1's sigma shift was fitted on " << kf_small << " tokens, but upstream "
                    "passes no latent (keyframe_interpolation.py:199-200) and gets " << anchor);
  CHECK(kf_large == anchor);

  // ── and the HQ arm's is NOT, which is what stops the above being a constant ─
  CHECK_MESSAGE(hq_small != hq_large,
                "the res_2s arm reported the same anchor at two resolutions, so this fixture "
                "cannot tell the two branches apart and the equalities above prove nothing");
  CHECK(hq_small != anchor);
  CHECK(hq_large != anchor);

  // ── and the two anchors really do produce DIFFERENT sigmas ────────────────
  //
  // The strongest half: a claim about the trajectory rather than about the
  // counter that reports it. `sigma_shift = tokens*mm + b` (schedulers.py:35-39),
  // so two token counts give two schedules — unless the shift arithmetic has
  // been flattened, in which case selecting the anchor would be inert and every
  // assertion above would still pass.
  // AT `rendered_steps`, NOT AT A LITERAL.
  const std::vector<float> at_anchor = vllm::Ltx2SigmaSchedule(rendered_steps, anchor);
  const std::vector<float> at_target = vllm::Ltx2SigmaSchedule(rendered_steps, hq_small);
  REQUIRE(at_anchor.size() == at_target.size());
  CHECK_MESSAGE(at_anchor != at_target,
                "the 4096 anchor and the target grid produce the SAME schedule on this fixture, "
                "so nothing above measures which one was taken");

  // AND THE RENDER'S STEP COUNT IS LOAD-BEARING, WHICH IS WORTH AN ASSERTION
  // RATHER THAN A COMMENT. TWO assertions, because they fail for different
  // reasons. The first names the render's own step count, so lowering
  // `gen.steps` to make this case faster fails HERE, by name, rather than
  // silently turning the comparison above into a value against itself. The
  // second pins the degeneracy itself, so it fails if a scheduler change ever
  // makes a 2-step schedule token-dependent.
  CHECK_MESSAGE(rendered_steps > 2,
                "the renders above ran at " << rendered_steps << " steps, and a schedule that "
                "short is {1, 0.1, 0} for EVERY token count (schedulers.py:48-55), so the "
                "trajectory comparison above compares a value with itself");
  CHECK_MESSAGE(vllm::Ltx2SigmaSchedule(/*steps=*/2, anchor) ==
                    vllm::Ltx2SigmaSchedule(/*steps=*/2, hq_small),
                "a 2-step schedule now DOES depend on the token count, so the stretch no longer "
                "pins both of its non-zero sigmas and the comment above is wrong");
  CHECK(vllm::Ltx2SigmaSchedule(/*steps=*/2, anchor).size() == 3u);
}

TEST_CASE("ltx2 keyframe: the distilled-LoRA requirement refuses BY WHAT IS MISSING") {
  // `distilled_lora` is a POSITIONAL, non-defaulted parameter
  // (keyframe_interpolation.py:68) and `--distilled-lora` is `required=True`
  // (utils/args.py:1140-1155) on the parser `:301` selects. Stage 2's three-sigma
  // refinement (`:166`) is what that adapter was trained for. Without it the
  // render FINISHES: a clip of the right size, frame count and sample rate, with
  // a distilled schedule run on undistilled weights.
  Workspace ws;
  const std::string lora =
      WriteFixtureLora(ws.root + "/distilled.safetensors", kFixtureLoraTarget, 1.0F);

  vllm::multimodal::VideoModelParams mp = KeyframeParams(ws.paths, lora);
  mp.extras.erase(vllm::multimodal::kLtx2LoraPathExtra);
  try {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    FAIL_CHECK("a keyframe_interpolation load with no distilled LoRA must be refused");
  } catch (const std::exception& e) {
    const std::string message = e.what();
    INFO("message = " << message);
    // It names the PIPELINE that was asked for, which is what makes the refusal
    // actionable rather than generic. This recipe is the THIRD user of
    // `requires_distilled_lora`, and #1151 is the record of what the message
    // said when it had only one: the previous pipeline's own line numbers.
    CHECK(message.find("keyframe_interpolation") != std::string::npos);
    CHECK(message.find("distilled LoRA") != std::string::npos);
    CHECK(message.find("lora_path") != std::string::npos);
    CHECK(message.find("default_2_stage_arg_parser") != std::string::npos);
    CHECK(message.find("a2vid_two_stage.py") == std::string::npos);
    CHECK(message.find("ti2vid_two_stages.py") == std::string::npos);
  }

  // THE CONTROL: the same load WITH the adapter renders, so the case is about
  // the requirement and not about any load failure.
  CHECK_NOTHROW((void)vllm::multimodal::LoadVideoEngine(KeyframeParams(ws.paths, lora)));
  // THE SECOND CONTROL: the DEFAULT kind is fine without an adapter, so this is
  // this recipe's requirement and not a new global one.
  CHECK_NOTHROW((void)vllm::multimodal::LoadVideoEngine(ConditioningParams(ws.paths)));

  // ── and NO audio take is demanded, which `a2vid_two_stage` does ────────────
  //
  // The two recipes come off the same parser and share `requires_distilled_lora`,
  // so a recipe written by copying that one would inherit `requires_audio_input`
  // and refuse every render. There is no `--audio-path` on this pipeline:
  // keyframe_interpolation.py:147-168 takes `images`, not a waveform.
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(KeyframeParams(ws.paths, lora));
  CHECK_NOTHROW((void)engine->Generate(KeyframeGen(ws.root + "/no_take")));
}


// The conditioning half of the same instrument, which the case above cannot
// reach: that engine conditions from a prompt-embeds FILE, so the tower and the
// connector never run and a table naming them would be lying. This one types a
// prompt, which is the arm #1269 and stage W4 are about — the spike bounds the
// caption projection at 39-100% of a ~1731 s phase and cannot say which,
// because nothing separated the tower from the connector.
TEST_CASE("ltx2 video: a PROMPTED render names the tower and the connector separately") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(EncoderParams(ws.paths));
  const std::string out_dir = ws.root + "/prompted_phases";
  const vllm::multimodal::VideoResult result =
      engine->Generate(PromptedGen(out_dir, "a b c"));

  REQUIRE_MESSAGE(!result.phase_log_path.empty(),
                  "the seam-level render returned no phase table path");
  CHECK(result.phase_log_path == out_dir + "/phase-log.json");
  const nlohmann::json table = nlohmann::json::parse(ReadAll(result.phase_log_path));
  std::vector<std::string> names;
  for (const nlohmann::json& entry : table["phases"]) {
    names.push_back(entry["name"].get<std::string>());
  }
  for (const std::string required : {"conditioning.tower", "conditioning.connector"}) {
    CHECK_MESSAGE(std::find(names.begin(), names.end(), required) != names.end(),
                  "the phase table names no '" << required << "' phase");
  }
  // The two are SEPARATE leaves, not one bracket counted twice: the sum still
  // holds, which is what says the split did not double count the tower.
  CHECK(table["unaccounted_seconds"].get<double>() <=
        0.05 * table["wall_seconds"].get<double>());
}

// ─── the CHECKPOINT CLASS, through the PRODUCTION ENTRY POINT ────────────────
//
// Row LTX25-CHECKPOINT-CLASS, issue #1137.
//
// THE DEFECT IS SILENCE, so the case that matters is a load that SUCCEEDED
// before this row and must refuse after it. `one_stage` is `TI2VidOneStagePipeline`,
// whose upstream `Model` column reads `Full` (ltx-pipelines CLAUDE.md:19 at
// fd4ded7f). Before this row the engine never asked which transformer it was
// handed, so pointing it at the distilled 22B file rendered a clip of the
// requested size, frame count and sample rate, sampled in a regime those weights
// were never trained for — and every output check this suite owns passed on it.
//
// EVERY CASE HERE ENTERS THROUGH `vllm_video_engine_load` or `LoadVideoEngine`,
// which is what `include/vllm.h`, the server and `ltx2-gen` all reach.
// `test_ltx2_pipeline` gates the recipe table and the pure decision; a case that
// called `Ltx2CheckpointClassRefusal` by hand would prove the function works and
// never that anything reaches it. Deleting the one call site in
// `Ltx2VideoEngine::Load` reds every case below.

namespace {

// A fixture load with the class declaration REMOVED, which is what a caller who
// has not heard of #1137 sends.
vllm::multimodal::VideoModelParams UndeclaredParams(const ltx2_fixture::Paths& paths,
                                                    const std::string& kind) {
  vllm::multimodal::VideoModelParams mp = FixtureParams(paths);
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = kind;
  mp.extras.erase(vllm::multimodal::kLtx2CheckpointClassExtra);
  return mp;
}

// The message of the refusal a load produced, or the empty string if it loaded.
// Returning the message rather than a bool is deliberate: "it threw" is the
// assertion that let a wrong refusal pass for a right one.
std::string LoadRefusal(const vllm::multimodal::VideoModelParams& mp) {
  try {
    const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
        vllm::multimodal::LoadVideoEngine(mp);
    return engine == nullptr ? std::string("LoadVideoEngine returned nullptr") : std::string();
  } catch (const std::exception& e) {
    return std::string(e.what());
  }
}

}  // namespace

TEST_CASE("ltx2 checkpoint class: a FULL-model arm refuses a distilled declaration") {
  Workspace ws;

  // THE MISMATCH. `one_stage` needs the full transformer and the caller says it
  // handed over the distilled one.
  vllm::multimodal::VideoModelParams mismatch = UndeclaredParams(ws.paths, "one_stage");
  mismatch.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "distilled";
  const std::string msg = LoadRefusal(mismatch);
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  // The message names WHICH class was supplied, WHICH is needed, and the
  // upstream row — the three things that make it actionable rather than a bare
  // "wrong checkpoint".
  CHECK(msg.find("'distilled'") != std::string::npos);
  CHECK(msg.find("FULL") != std::string::npos);
  CHECK(msg.find("one_stage") != std::string::npos);
  CHECK(msg.find("CLAUDE.md:17-30") != std::string::npos);
  // ...and it says WHY it refuses rather than renders, which is the whole reason
  // the row exists: the render would finish.
  CHECK(msg.find("no output check") != std::string::npos);

  // THE CONTROL: the same load with the honest declaration goes through, so the
  // case above measures the class check and not a broken `one_stage` fixture.
  vllm::multimodal::VideoModelParams honest = UndeclaredParams(ws.paths, "one_stage");
  honest.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "full";
  honest.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  CHECK(LoadRefusal(honest).empty());

  // AND THE OTHER POLARITY, so this is not a one-way test. `distilled_two_stage`
  // is `DistilledPipeline`, `Distilled only` (CLAUDE.md:25), and it refuses the
  // full model just as hard.
  vllm::multimodal::VideoModelParams reversed =
      UndeclaredParams(ws.paths, "distilled_two_stage");
  reversed.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "full";
  const std::string reversed_msg = LoadRefusal(reversed);
  INFO(reversed_msg);
  REQUIRE_FALSE(reversed_msg.empty());
  CHECK(reversed_msg.find("'full'") != std::string::npos);
  CHECK(reversed_msg.find("DISTILLED") != std::string::npos);
}

TEST_CASE("ltx2 checkpoint class: an UNDECLARED load refuses instead of rendering") {
  Workspace ws;

  // THE RED-FIRST CASE. Before row LTX25-CHECKPOINT-CLASS this load SUCCEEDED
  // and rendered. Nothing about the checkpoint changed; what changed is that the
  // engine now asks.
  //
  // THE ADAPTER FLAG DECIDES WHICH REFUSAL A CALLER READS, and the second column
  // is where that is stated. The class check sits AFTER the
  // `requires_distilled_lora` one deliberately: a two-stage load with no adapter
  // fails both, and the adapter message names a missing file the caller can go
  // and fetch, while a class message would send the same reader to a different
  // question. So the three kinds that demand an adapter get a REAL one here.
  //
  // A real file and not a placeholder path, and that is measured rather than
  // assumed: the load builds and READS the adapter before it resolves the
  // recipe, so a non-existent path refuses with `safetensors: open failed` and
  // this case would be gating the wrong refusal.
  const std::string adapter =
      WriteFixtureLora(ws.root + "/class_adapter.safetensors", kFixtureLoraTarget, 1.0F);
  struct Arm {
    const char* kind;
    bool needs_adapter;
  };
  for (const Arm arm : {Arm{"one_stage", false}, Arm{"distilled_two_stage", false},
                        Arm{"t2a_one_stage", false}, Arm{"a2vid_two_stage", true},
                        Arm{"ti2vid_two_stage", true}, Arm{"keyframe_interpolation", true},
                        Arm{"res2s_two_stage", false}, Arm{"dfr", false},
                        Arm{"retake", false}}) {
    INFO("kind = " << std::string(arm.kind));
    vllm::multimodal::VideoModelParams mp = UndeclaredParams(ws.paths, arm.kind);
    if (arm.needs_adapter) {
      mp.extras[vllm::multimodal::kLtx2LoraPathExtra] = adapter;
    }
    const std::string msg = LoadRefusal(mp);
    INFO("refusal = " << msg);
    REQUIRE_FALSE(msg.empty());
    // It has to be THIS refusal and not some other one the kind happens to
    // produce first — several of these kinds refuse for other reasons too.
    CHECK(msg.find("checkpoint_class") != std::string::npos);
    // The message carries the MEASUREMENT, because "declare the class" without
    // it reads as bureaucracy rather than as a fact about these two files.
    CHECK(msg.find("42,018,190,584") != std::string::npos);
    CHECK(msg.find("byte-identical") != std::string::npos);
    // ...and the three spellings, so the caller can act on it.
    CHECK(msg.find("'full'") != std::string::npos);
    CHECK(msg.find("'distilled'") != std::string::npos);
    CHECK(msg.find("'keyframe_slot_sft'") != std::string::npos);
  }

  // THE DEFAULT KIND IS NOT EXEMPT. `pipeline_kind` defaults to
  // `distilled_two_stage`, so a caller who names no kind at all still declares.
  vllm::multimodal::VideoModelParams defaulted = FixtureParams(ws.paths);
  defaulted.extras.erase(vllm::multimodal::kLtx2CheckpointClassExtra);
  CHECK_FALSE(LoadRefusal(defaulted).empty());

  // AN UNPARSEABLE VALUE is refused too, and with a DIFFERENT message: refused
  // rather than defaulted, because every default here picks a sampling regime
  // for the caller. `dev` is the realistic typo — it is what the full file is
  // named after.
  vllm::multimodal::VideoModelParams typo = UndeclaredParams(ws.paths, "one_stage");
  typo.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "dev";
  const std::string typo_msg = LoadRefusal(typo);
  INFO(typo_msg);
  REQUIRE_FALSE(typo_msg.empty());
  CHECK(typo_msg.find("'dev'") != std::string::npos);
  CHECK(typo_msg.find("'full'") != std::string::npos);
  // NOT the absent-declaration message, which quotes the measurement.
  CHECK(typo_msg.find("42,018,190,584") == std::string::npos);
}

TEST_CASE("ltx2 checkpoint class: `Full or distilled` is a CONDITION, not a hole") {
  Workspace ws;

  // `retake` is `RetakePipeline`, the one `Full or distilled` row
  // (CLAUDE.md:29). This recipe mirrors upstream's `distilled=True` arm
  // (retake.py:336, :359), which takes DISTILLED_SIGMAS (:287), and upstream's
  // stated condition for that arm is "using distilled model or passing
  // distillation lora with full model" (retake.py:71-73).
  //
  // So `full` with no adapter is refused: it would run the distilled schedule on
  // undistilled weights, which is #1137 again on the one permissive row.
  vllm::multimodal::VideoModelParams full_no_lora = UndeclaredParams(ws.paths, "retake");
  full_no_lora.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "full";
  const std::string msg = LoadRefusal(full_no_lora);
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("retake.py:71-73") != std::string::npos);
  CHECK(msg.find("lora_path") != std::string::npos);

  // ...and `distilled` on the same recipe is accepted with no adapter at all,
  // which is what makes the case above a condition on the CLASS rather than a
  // blanket adapter requirement on the kind.
  vllm::multimodal::VideoModelParams distilled = UndeclaredParams(ws.paths, "retake");
  distilled.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "distilled";
  CHECK(LoadRefusal(distilled).empty());
}

TEST_CASE("ltx2 checkpoint class: the refusal names the extras by the keys the engine reads") {
  Workspace ws;

  // The refusal text is assembled in `ltx2_pipeline.cpp`, which does not include
  // a `multimodal` header and therefore spells `checkpoint_class` and
  // `lora_path` as literals. This case is what makes that duplication GATED
  // rather than trusted: it holds a REAL refusal against the constants the
  // engine actually reads, so a rename on either side fails here instead of
  // producing a message that names a key nothing consumes.
  const std::string undeclared = LoadRefusal(UndeclaredParams(ws.paths, "one_stage"));
  INFO(undeclared);
  REQUIRE_FALSE(undeclared.empty());
  CHECK(undeclared.find(vllm::multimodal::kLtx2CheckpointClassExtra) != std::string::npos);

  vllm::multimodal::VideoModelParams retake_full = UndeclaredParams(ws.paths, "retake");
  retake_full.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "full";
  const std::string retake_msg = LoadRefusal(retake_full);
  INFO(retake_msg);
  REQUIRE_FALSE(retake_msg.empty());
  CHECK(retake_msg.find(vllm::multimodal::kLtx2LoraPathExtra) != std::string::npos);
}

TEST_CASE("ltx2 checkpoint class: the declaration is a CLAIM, and the engine says so") {
  Workspace ws;

  // WHAT THIS ROW DOES NOT BUY, asserted rather than left to the reader. The
  // fixture DiT is neither the full nor the distilled 22B transformer, and the
  // engine accepts BOTH declarations for it, because it never opens the file to
  // check. There is nothing in the header to check against — that is the
  // measurement in `.agents/specs/ltx25-checkpoint-class.md` section 2.
  //
  // What the row buys is that the wrong sampling regime now needs a deliberate
  // false statement instead of silence. A case asserting the opposite would be
  // asserting a detector this engine does not have.
  vllm::multimodal::VideoModelParams as_full = UndeclaredParams(ws.paths, "one_stage");
  as_full.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "full";
  as_full.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  CHECK(LoadRefusal(as_full).empty());

  vllm::multimodal::VideoModelParams as_distilled =
      UndeclaredParams(ws.paths, "distilled_two_stage");
  as_distilled.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = "distilled";
  as_distilled.extras[vllm::multimodal::kLtx2MaxPhaseExtra] = "0";
  CHECK(LoadRefusal(as_distilled).empty());

  // The SAME file, two contradictory declarations, both accepted. The pair is
  // the assertion: it is one file and the two loads disagree about what it is.
  CHECK(as_full.dit_path == as_distilled.dit_path);
}
