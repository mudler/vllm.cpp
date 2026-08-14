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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "ltx2_video_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/ltx2_loader.h"
#include "vllm/model_executor/models/ltx2_text_encoder.h"
#include "vllm/model_executor/models/ltx2_upsampler.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — the seam the engine asks
#include "vllm.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vllm/multimodal/video_engine.h"

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

vllm::multimodal::VideoModelParams FixtureParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.video_vae_path = paths.video_vae;
  mp.audio_vae_path = paths.audio_vae;
  mp.prompt_embeds_path = paths.video_embeds;
  mp.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = paths.audio_embeds;
  mp.device = 0;
  return mp;
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

TEST_CASE("ltx2 video: keyframe and reference conditioning is refused by name") {
  Workspace ws;
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(FixtureParams(ws.paths));
  vllm::multimodal::VideoGenParams gen = FixtureGen(ws.root + "/keyframed");
  gen.first_frame_path = ws.paths.video_embeds;  // any path: the refusal precedes the read
  try {
    (void)engine->Generate(gen);
    FAIL("keyframe conditioning must be refused while no encoder is reachable from here");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("ImageConditioner") != std::string::npos);
    // A refusal whose stated REASON has gone stale is worse than a vague one: it
    // sends the next reader to build something that already exists. Phase L11
    // ported the video VAE encoder, so the message may no longer claim the
    // encoder is missing, and these two assertions hold it to the pieces that
    // actually are — the loader path that would put encoder weights in memory,
    // and the CRF re-compression upstream applies before encoding.
    CHECK(msg.find("VAE_ENCODER_COMFY_KEYS_FILTER") != std::string::npos);
    CHECK(msg.find("default_image_crf") != std::string::npos);
    // And the QUALIFIER on that re-compression, which the two substrings above do
    // not reach: `preprocess` returns the image UNTOUCHED at `crf == 0`
    // (media_io/decode.py:413-435, the `if crf == 0:` early return at :425-426 —
    // NOT the one at :427-428, which is the degenerate-size guard), so "re-compresses
    // before encoding" is only true of a nonzero resolved CRF. Naming the round
    // trip without naming its exception overstates what is unported and sends the
    // next reader to build an H.264 path for a case that needs none — the same
    // failure mode as a stale reason, one step subtler. Gated here so deleting the
    // qualifier goes RED rather than quietly restoring the overstatement.
    CHECK(msg.find("unless that CRF is 0") != std::string::npos);
  }
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
  const char* keys[] = {vllm::multimodal::kLtx2AudioPromptEmbedsExtra,
                        vllm::multimodal::kLtx2MaxPhaseExtra};
  const char* values[] = {audio_embeds.c_str(), max_phase.c_str()};

  vllm_video_model_params mp = vllm_video_model_params_default();
  mp.dit_path = ws.paths.dit.c_str();
  mp.video_vae_path = ws.paths.video_vae.c_str();
  mp.audio_vae_path = ws.paths.audio_vae.c_str();
  mp.prompt_embeds_path = ws.paths.video_embeds.c_str();
  mp.extra_keys = keys;
  mp.extra_values = values;
  mp.n_extras = 2;
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
    // The shipped DiT DECLARES `use_keyframes_abs_pos_embedding: true`, which
    // `ParseLtx2DitParams` refuses by name because the module is unported. The
    // engine clears it in a copy under `allow_unported_modules`; this mirrors
    // that, and asserting the file declares it is the point.
    REQUIRE(config["transformer"]["use_keyframes_abs_pos_embedding"].get<bool>());
    config["transformer"]["use_keyframes_abs_pos_embedding"] = false;
    nlohmann::json wrapper;
    wrapper["config"] = config;
    const vllm::Ltx2DitParams declared = vllm::ParseLtx2DitParams(wrapper);
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
  // THE PROBE IS A REAL REFUSAL, not an injected one. Keyframe / reference
  // conditioning is refused by name (ltx2_video.cpp, the `ImageConditioner`
  // note) and that refusal sits AFTER the trace is written, so a prompted
  // request carrying a reference image walks the whole encode path, fills the
  // trace, and then fails — exactly the shape this flag exists to report.
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
    FAIL("keyframe / reference conditioning must be refused");
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    INFO(msg);
    CHECK(msg.find("reference conditioning") != std::string::npos);
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
