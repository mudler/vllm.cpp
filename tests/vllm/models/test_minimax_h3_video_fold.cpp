// ARCH-ONE-SURFACE ROW 2 fold gate — MiniMax-H3 video+audio generation.
//
// Three arms over the SAME deterministic tiny fixture
// (minimax_h3_video_fold_fixture.h), all held to the committed goldens in
// fixtures/minimax_h3_video_fold/ — which are the frames + WAV the PRE-fold
// `minimax-h3-gen` binary rendered at the branch base (fc636c76), BEFORE any
// refactor:
//
//   A. the LIBRARY SEAM (MiniMaxH3VideoEngine::Load + Generate) — the entry
//      point the C ABI, the server's /v1/videos and the thin-client example
//      now drive — must reproduce the goldens BYTE-IDENTICALLY;
//   B. the REPLICATED OLD PIPELINE — the pre-fold example's exact assembly
//      (keep-quant GGUF load -> request build -> the dual-stream splitmix64
//      Gaussian noise -> MiniMaxH3GenerateT2va -> PPM/WAV serialization),
//      re-stated here line for line — must also reproduce them, proving the
//      goldens still describe the OLD pipeline on this tree;
//   C. the committed goldens themselves.
//
// A == B == C means the fold moved the capability without moving a single
// output byte. The rewritten thin-client binary is held to the same goldens by
// the operator-run binary gate (recorded in the spec); the mux argv the seam
// composes is pinned against the pre-fold `minimax-h3-mux --print-only`
// capture the same way.
#include "vllm/multimodal/minimax_h3_video.h"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/video_api.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform() — the seam device 1 resolves through
#include "vllm/support/platform_compat.h"
#include "minimax_h3_video_fold_fixture.h"
#include "vt/backend.h"

namespace {

namespace fs = std::filesystem;

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE_MESSAGE(in.good(), "cannot open ", path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string GoldenDir() { return MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR; }

// A fresh fixture + output workspace per test run (pid-unique under /tmp).
struct FoldWorkspace {
  std::string root;
  FoldWorkspace() {
    static int counter = 0;
    root = (fs::temp_directory_path() /
            ("vllm_h3_video_fold_" +
             std::to_string(vllm::support::CurrentProcessId()) + "_" +
             std::to_string(counter++)))
               .string();
    fs::create_directories(root);
    fixture = root + "/fixture";
    minimax_h3_fold::WriteFoldFixture(fixture);
  }
  ~FoldWorkspace() {
    // Best-effort cleanup; a leftover temp dir on abort is diagnosable, not
    // harmful.
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  std::string fixture;
};

vllm::multimodal::MiniMaxH3VideoModelParams FixtureModelParams(const std::string& dir) {
  vllm::multimodal::MiniMaxH3VideoModelParams mp;
  mp.dit_path = dir + "/dit.gguf";
  mp.video_vae_path = dir + "/video_vae.safetensors";
  mp.video_vae_config_path = dir + "/video_vae_config.json";
  mp.audio_vae_path = dir + "/audio_vae.safetensors";
  mp.audio_vae_config_path = dir + "/audio_vae_config.json";
  mp.prompt_embeds_path = dir + "/prompt_embeds.f32";
  mp.partition = "fl2va";
  mp.device = 0;        // the CPU-feasible fold arm
  mp.dequant_bf16 = 0;  // keep-quant, the arm the goldens were captured on
  return mp;
}

vllm::multimodal::MiniMaxH3VideoGenParams FixtureGenParams(const std::string& out_dir) {
  vllm::multimodal::MiniMaxH3VideoGenParams gen;
  gen.num_frames = 5;
  gen.height = 32;
  gen.width = 32;
  gen.steps = 3;
  gen.output_dir = out_dir;
  return gen;
}

// The number of golden frames committed (the pre-fold binary wrote 8: latent_t
// 2 x the ViT3D decoder's fixed patch_size_t 4).
constexpr int kGoldenFrames = 8;

class CountingCudaBackend final : public vt::Backend {
 public:
  void* Alloc(size_t) override { return nullptr; }
  void Free(void*) override {}
  void Memset(vt::Queue&, void*, int, size_t) override {}
  void Copy(vt::Queue&, void*, const void*, size_t) override {}
  vt::Queue CreateQueue() override {
    ++create_queue_calls;
    return vt::Queue{vt::Device{vt::DeviceType::kCPU, 7}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }

  int create_queue_calls = 0;
};

// The PLATFORM half of the same fake. #660: device 1 used to be
// `static_cast<vt::DeviceType>(1)`, so registering a backend alone was enough to
// fake an accelerator — the engine never asked whether this build HAS one. It
// asks now, through `CurrentPlatform()`, so a test that wants the accelerator
// path has to supply a platform as well as a backend. That is not a harness
// concession; it is the defect being visible: a build with a CUDA backend
// registered and no CUDA platform is not a build that runs on CUDA.
class FakeCudaPlatform final : public vllm::platforms::Platform {
 public:
  explicit FakeCudaPlatform(vt::Backend& backend) : backend_(backend) {}
  vt::DeviceType device_type() const override { return vt::DeviceType::kCUDA; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {}; }
  std::vector<vt::DType> supported_dtypes() const override { return {vt::DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }

 private:
  vt::Backend& backend_;
};

class ScopedCudaBackendRegistration {
 public:
  explicit ScopedCudaBackendRegistration(vt::Backend* replacement)
      : previous_(vt::TryGetBackend(vt::DeviceType::kCUDA)),
        previous_platform_(vllm::platforms::HasPlatform(vt::DeviceType::kCUDA)
                               ? &vllm::platforms::GetPlatform(vt::DeviceType::kCUDA)
                               : nullptr),
        platform_(*replacement) {
    vt::RegisterBackend(vt::DeviceType::kCUDA, replacement);
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &platform_);
  }
  ~ScopedCudaBackendRegistration() {
    if (previous_ != nullptr) vt::RegisterBackend(vt::DeviceType::kCUDA, previous_);
    if (previous_platform_ != nullptr) {
      vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, previous_platform_);
    }
  }

  ScopedCudaBackendRegistration(const ScopedCudaBackendRegistration&) = delete;
  ScopedCudaBackendRegistration& operator=(const ScopedCudaBackendRegistration&) = delete;

 private:
  vt::Backend* previous_;
  vllm::platforms::Platform* previous_platform_;
  FakeCudaPlatform platform_;
};

void CheckAgainstGoldens(const std::string& out_dir) {
  for (int f = 0; f < kGoldenFrames; ++f) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06d.ppm", f);
    const std::string got = ReadAll(out_dir + name);
    const std::string want = ReadAll(GoldenDir() + name);
    INFO("frame ", f);
    REQUIRE(got.size() == want.size());
    CHECK_MESSAGE(got == want, "frame ", f, " diverged from the pre-fold golden");
  }
  const std::string got_wav = ReadAll(out_dir + "/audio.wav");
  const std::string want_wav = ReadAll(GoldenDir() + "/audio.wav");
  REQUIRE(got_wav.size() == want_wav.size());
  CHECK_MESSAGE(got_wav == want_wav, "audio.wav diverged from the pre-fold golden");
  // And there is no NINTH frame: a pipeline that silently rendered a different
  // clip length would otherwise pass every per-frame compare above.
  char extra[64];
  std::snprintf(extra, sizeof(extra), "/frame_%06d.ppm", kGoldenFrames);
  std::ifstream ninth(out_dir + extra, std::ios::binary);
  CHECK_MESSAGE(!ninth.good(), "the render produced more frames than the golden clip");
}

}  // namespace

TEST_CASE("minimax_h3 video fold: ABI device selectors map through DeviceType") {
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(0) == vt::DeviceType::kCPU);
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(-1));
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(2));

  // Selector 1 is RESOLVED through the platform seam, not cast from the integer
  // (#660: `static_cast<vt::DeviceType>(1)` was kCUDA only because kCUDA happens
  // to be enum value 1). So the answer depends on what this build registered,
  // and this case asserts BOTH arms rather than skipping either — a skip here
  // would leave the CPU-only build, which is the one the defect hid on,
  // unmeasured.
  const vllm::platforms::Platform& platform = vllm::platforms::CurrentPlatform();
  const vt::DeviceType accelerator = platform.device_type();
  const bool have_backend =
      accelerator != vt::DeviceType::kCPU && vt::TryGetBackend(accelerator) != nullptr;
  // The THIRD question, asked here because the SOURCE asks it. A predicate that
  // asks only the first two describes a different function than the one under
  // test: on a PARTIAL backend (Metal, Tenstorrent) both of the first two are
  // true, the source correctly refuses BY NAME, and a two-question predicate
  // routes that correct refusal into the `== accelerator` arm — where it
  // surfaces as an uncaught exception. That is a false RED on precisely the
  // build class #659 exists to serve, and it is invisible on the CPU and CUDA
  // boxes that run the gates, which is this row's own thesis about #659.
  const bool accepts_architecture =
      have_backend &&
      platform.supports_model_architecture(vllm::multimodal::kMiniMaxH3VideoFamily);

  if (accepts_architecture) {
    // On the CUDA box this is byte-for-byte the old answer.
    CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(1) == accelerator);
  } else {
    // The assertion the cast could never make: device 1 is REFUSED by name
    // instead of returning kCUDA and failing one step later inside
    // `vt::GetBackend(kCUDA)`. WHICH refusal is itself asserted — a partial
    // backend must be told it is partial, not told its backend is missing, and
    // a right refusal for a wrong reason is a wrong diagnosis that reads as a
    // right one.
    const std::string want =
        have_backend ? "DECLINES" : "no accelerator backend is registered";
    try {
      (void)vllm::multimodal::MiniMaxH3VideoDeviceType(1);
      FAIL("device 1 must be refused when this build cannot honour it");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      INFO(msg);
      CHECK(msg.find(want) != std::string::npos);
    }
  }
}

TEST_CASE("minimax_h3 video fold: CUDA load creates exactly one queue") {
  FoldWorkspace ws;
  static CountingCudaBackend backend;
  backend.create_queue_calls = 0;
  const ScopedCudaBackendRegistration registration(&backend);

  vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
  mp.device = 1;
  auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(mp);
  CHECK(backend.create_queue_calls == 1);
  CHECK(engine->device() == (vt::Device{vt::DeviceType::kCPU, 7}));
}

// ─── ARM A: the library seam reproduces the pre-fold binary byte for byte ────
TEST_CASE("minimax_h3 video fold: the library seam reproduces the pre-fold goldens") {
  FoldWorkspace ws;
  auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(FixtureModelParams(ws.fixture));
  CHECK(!engine->has_encoder());
  CHECK(engine->has_prompt_embeds());

  const std::string out_dir = ws.root + "/seam_out";
  const vllm::multimodal::MiniMaxH3VideoResult result =
      engine->Generate(FixtureGenParams(out_dir));

  CHECK(result.frame_dir == out_dir);
  CHECK(result.frame_count == kGoldenFrames);
  CHECK(result.width == 32);
  CHECK(result.height == 32);
  CHECK(result.fps == 24);
  CHECK(result.sample_rate == 32000);
  CheckAgainstGoldens(out_dir);

  // The mux argv the seam composes must be EXACTLY what the pre-fold
  // `minimax-h3-mux --print-only` printed (with W/ as the directory).
  std::string joined;
  for (size_t i = 0; i < result.mux_argv.size(); ++i) {
    joined += (i == 0 ? "" : " ") + result.mux_argv[i];
  }
  std::string golden_argv = ReadAll(GoldenDir() + "/golden_mux_argv.txt");
  while (!golden_argv.empty() && (golden_argv.back() == '\n' || golden_argv.back() == '\r')) {
    golden_argv.pop_back();
  }
  // The golden was captured with the placeholder directory `W`; substitute the
  // real output dir to compare whole-argv.
  size_t pos = 0;
  while ((pos = golden_argv.find("W/", pos)) != std::string::npos) {
    golden_argv.replace(pos, 1, out_dir);
    pos += out_dir.size() + 1;
  }
  CHECK(joined == golden_argv);
  CHECK(result.mux_output_path == out_dir + "/video.mp4");

  // A SECOND generation from the same engine handle must be byte-identical
  // too (staged state is reused, never mutated).
  const std::string out2 = ws.root + "/seam_out2";
  (void)engine->Generate(FixtureGenParams(out2));
  CheckAgainstGoldens(out2);
}

// ─── ARM B: the replicated OLD pipeline still produces the same bytes ────────
TEST_CASE("minimax_h3 video fold: the replicated pre-fold pipeline matches the goldens") {
  FoldWorkspace ws;

  // This block restates examples/minimax_h3_gen/main.cpp@fc636c76 (the
  // pre-fold assembly) against the same fixture: any behavioural drift in the
  // seam OR a stale golden makes exactly one of the two arms diverge.
  const vllm::GgufFile f = vllm::GgufFile::Open(ws.fixture + "/dit.gguf");
  const vllm::MiniMaxH3GgufDit dit = vllm::LoadMiniMaxH3DitFromGguf(f, /*keep_quant=*/true);

  vllm::MiniMaxH3LatentStats video_stats, audio_stats;
  vllm::MiniMaxH3VideoVaeDecoderConfig video_cfg;
  vllm::MiniMaxH3AudioVaeConfig audio_cfg;
  {
    std::ifstream in(ws.fixture + "/video_vae_config.json");
    nlohmann::json j;
    in >> j;
    video_cfg = vllm::ParseMiniMaxH3VideoVaeDecoderConfig(j, &video_stats);
  }
  {
    std::ifstream in(ws.fixture + "/audio_vae_config.json");
    nlohmann::json j;
    in >> j;
    audio_cfg = vllm::ParseMiniMaxH3AudioVaeConfig(j, &audio_stats);
  }
  const vllm::SafetensorsFile vfile =
      vllm::SafetensorsFile::Open(ws.fixture + "/video_vae.safetensors");
  const vllm::MiniMaxH3AudioVaeWeights video_weights =
      vllm::LoadMiniMaxH3VideoVaeDecoderWeights(vfile);
  const vllm::SafetensorsFile afile =
      vllm::SafetensorsFile::Open(ws.fixture + "/audio_vae.safetensors");
  const vllm::MiniMaxH3AudioVaeWeights audio_weights =
      vllm::LoadMiniMaxH3AudioVaeWeights(afile);

  std::vector<float> prompt_embeds;
  {
    std::ifstream in(ws.fixture + "/prompt_embeds.f32", std::ios::binary | std::ios::ate);
    const std::streamsize bytes = in.tellg();
    in.seekg(0);
    prompt_embeds.resize(static_cast<size_t>(bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(prompt_embeds.data()), bytes);
  }

  vllm::MiniMaxH3T2vaRequest request;
  request.num_steps = 3;
  const vllm::MiniMaxH3ShapePlan plan =
      vllm::MiniMaxH3ResolveShape("t2va", 0.0, 5, 32, 32, 0, 0);
  request.latent_t = plan.latent_t;
  request.num_frames = plan.num_frames;
  request.latent_h = plan.height / vllm::kMiniMaxH3VaeRatio;
  request.latent_w = plan.width / vllm::kMiniMaxH3VaeRatio;
  request.audio_t = plan.audio_t;
  request.audio_channel = vllm::kMiniMaxH3AudioChannels;
  request.video_latents_mean = video_stats.mean;
  request.video_latents_std = video_stats.std_dev;
  request.audio_latents_mean = audio_stats.mean;
  request.audio_latents_std = audio_stats.std_dev;
  request.partition = vllm::MiniMaxH3PartitionFromFlag("fl2va");
  request.text_len = static_cast<int64_t>(prompt_embeds.size()) / dit.params.text_dim;

  // The pre-fold noise draw, restated verbatim (splitmix64 + Box-Muller,
  // Gaussian default, fixed per-modality seeds).
  const int64_t frame_rows = (request.latent_h / dit.params.patch_size_h) *
                             (request.latent_w / dit.params.patch_size_w);
  const int64_t video_rows = request.latent_t * frame_rows;
  const int64_t audio_rows = request.audio_t * request.audio_channel;
  auto fill = [](std::vector<float>& out, uint64_t seed) {
    uint64_t x = seed;
    auto u01 = [&x]() {
      x += 0x9E3779B97F4A7C15ULL;
      uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z ^= z >> 31;
      return (z >> 11) * 0x1.0p-53;
    };
    for (size_t i = 0; i < out.size(); ++i) {
      double u1 = u01(), u2 = u01();
      if (u1 < 1e-12) u1 = 1e-12;
      out[i] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2));
    }
  };
  std::vector<float> noise_video(
      static_cast<size_t>(video_rows * dit.params.video_row_width()));
  std::vector<float> noise_audio(
      static_cast<size_t>(audio_rows * dit.params.audio_latents_dim));
  fill(noise_video, 0x5EED1234ULL);
  fill(noise_audio, 0x5EED5678ULL);

  const vllm::MiniMaxH3T2vaResult out = vllm::MiniMaxH3GenerateT2va(
      vt::Device{}, request, dit.params, dit.weights, video_cfg, video_weights, audio_cfg,
      audio_weights, prompt_embeds, noise_video, noise_audio, vt::DType::kBF16);

  REQUIRE(out.frame_shape.t == kGoldenFrames);
  for (int64_t fr = 0; fr < out.frame_shape.t; ++fr) {
    char name[64];
    std::snprintf(name, sizeof(name), "/frame_%06lld.ppm", static_cast<long long>(fr));
    const std::string want = ReadAll(GoldenDir() + name);
    const std::string got = vllm::MiniMaxH3WritePpmFrame(out.frames, out.frame_shape, fr);
    INFO("frame ", fr);
    CHECK_MESSAGE(got == want, "replicated pre-fold pipeline diverged on frame ", fr);
  }
  const std::string want_wav = ReadAll(GoldenDir() + "/audio.wav");
  const std::string got_wav = vllm::MiniMaxH3WriteWav(
      out.waveform, out.audio_channels, out.audio_samples_per_channel, out.sample_rate);
  CHECK_MESSAGE(got_wav == want_wav, "replicated pre-fold pipeline diverged on audio.wav");

  // And the silent mux argv is still the pre-fold `minimax-h3-mux` contract.
  vllm::MiniMaxH3MuxRequest mux;
  mux.frame_pattern = "frames_%06d.ppm";
  mux.output_path = "silent.mp4";
  const std::vector<std::string> argv = vllm::MiniMaxH3BuildMp4MuxArgs(mux);
  std::string joined;
  for (size_t i = 0; i < argv.size(); ++i) joined += (i == 0 ? "" : " ") + argv[i];
  std::string golden = ReadAll(GoldenDir() + "/golden_mux_argv_silent.txt");
  while (!golden.empty() && (golden.back() == '\n' || golden.back() == '\r')) golden.pop_back();
  CHECK(joined == golden);
}

// ─── the guards the seam must keep (the #77 partition catch + inputs) ────────
TEST_CASE("minimax_h3 video fold: the seam honors the partition/task guard and refuses bad input") {
  FoldWorkspace ws;

  SUBCASE("an undeclared partition refuses a full render, naming the fix") {
    vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
    mp.partition = "";  // declared-but-unknown: the guard refuses every task
    auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(mp);
    try {
      (void)engine->Generate(FixtureGenParams(ws.root + "/out"));
      FAIL("an undeclared partition must refuse to render");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      CHECK(msg.find("partition") != std::string::npos);
    }
  }

  SUBCASE("the ref2va partition refuses a t2va render") {
    vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
    mp.partition = "ref2va";
    auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(mp);
    try {
      (void)engine->Generate(FixtureGenParams(ws.root + "/out"));
      FAIL("the ref2va partition must refuse t2va");
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      CHECK(msg.find("ref2va") != std::string::npos);
    }
  }

  SUBCASE("an unknown partition string is refused at load") {
    vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
    mp.partition = "nonsense";
    CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoEngine::Load(mp));
  }

  SUBCASE("no conditioning is a loud refusal, not a garbage render") {
    vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
    mp.prompt_embeds_path = "";  // no encoder either
    auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(mp);
    try {
      (void)engine->Generate(FixtureGenParams(ws.root + "/out"));
      FAIL("no conditioning must refuse");
    } catch (const std::exception& e) {
      CHECK(std::string(e.what()).find("conditioning") != std::string::npos);
    }
  }

  SUBCASE("a missing output_dir is refused before any compute") {
    auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(FixtureModelParams(ws.fixture));
    vllm::multimodal::MiniMaxH3VideoGenParams gen = FixtureGenParams("");
    CHECK_THROWS(engine->Generate(gen));
  }

  SUBCASE("keyframes and ref2va references are exclusive") {
    auto engine = vllm::multimodal::MiniMaxH3VideoEngine::Load(FixtureModelParams(ws.fixture));
    vllm::multimodal::MiniMaxH3VideoGenParams gen = FixtureGenParams(ws.root + "/out");
    gen.first_frame_path = ws.fixture + "/nonexistent.ppm";
    gen.ref_video_dir = ws.fixture;
    CHECK_THROWS(engine->Generate(gen));
  }

  SUBCASE("a missing DiT path is refused at load") {
    vllm::multimodal::MiniMaxH3VideoModelParams mp = FixtureModelParams(ws.fixture);
    mp.dit_path = "";
    CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoEngine::Load(mp));
  }
}

// ─── the /v1/videos mapping (library-owned so HTTP and FFI cannot drift) ─────
TEST_CASE("minimax_h3 video fold: the VideoRequest mapping carries every field") {
  vllm::openai::VideoRequest req;
  req.prompt = "a cat";
  req.task = "t2va";
  req.duration_seconds = 2.5;
  req.num_frames = 29;
  req.height = 320;
  req.width = 640;
  req.num_inference_steps = 7;
  req.flow_shift = 11.0;
  req.audio_flow_shift = 2.5;
  req.seed = 42;
  req.has_seed = true;
  req.input_reference_path = "/x/ref.ppm";
  req.input_reference_video_dir = "/x/clip";
  req.input_reference_audio_bytes = {1, 2, 3};

  const vllm::multimodal::MiniMaxH3VideoGenParams gen =
      vllm::multimodal::MiniMaxH3VideoGenParamsFromRequest(req, "/x/job0");
  CHECK(gen.prompt == "a cat");
  CHECK(gen.task == "t2va");
  CHECK(gen.duration_seconds == 2.5);
  CHECK(gen.num_frames == 29);
  CHECK(gen.height == 320);
  CHECK(gen.width == 640);
  CHECK(gen.steps == 7);
  CHECK(gen.flow_shift == 11.0);
  CHECK(gen.audio_flow_shift == 2.5);
  CHECK(gen.seed == 42);
  CHECK(gen.has_seed);
  CHECK(gen.first_frame_path == "/x/ref.ppm");
  CHECK(gen.noise_aug == 1.0);  // input_reference pins frame 0 exactly
  CHECK(gen.ref_video_dir == "/x/clip");
  CHECK(gen.ref_audio_wav == std::string("\x01\x02\x03"));
  CHECK(gen.output_dir == "/x/job0");

  // The inline data:-URL spelling lands on the BYTES member, not the path.
  vllm::openai::VideoRequest inline_req;
  inline_req.input_reference_bytes = {'P', '6'};
  const vllm::multimodal::MiniMaxH3VideoGenParams inline_gen =
      vllm::multimodal::MiniMaxH3VideoGenParamsFromRequest(inline_req, "/x/job1");
  CHECK(inline_gen.first_frame_ppm == "P6");
  CHECK(inline_gen.first_frame_path.empty());
}
