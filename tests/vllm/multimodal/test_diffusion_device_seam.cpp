// #659 / #660 — the diffusion lane's device seam, held against a PARTIAL
// backend.
//
// The two lanes ask "where does this engine run" and the answer has three parts:
// is there an accelerator, is a backend registered for it, and can that backend
// actually run THIS model. Only the first two were being asked. The third,
// `Platform::supports_model_architecture` (include/vllm/platforms/interface.h),
// exists precisely so a partial backend can decline BY NAME — Metal registers 15
// of 75 ops and Tenstorrent a comparable slice, and both name exactly two
// architectures (src/vllm/platforms/metal.cpp:70,
// src/vllm/platforms/tenstorrent.cpp:55). On such a build a `device = 1`
// diffusion load was NO LONGER refused by name, and became a queue bind that
// dies later inside a kernel with a shape error.
//
// CUDA cannot see any of this: `supports_model_architecture` defaults to true,
// so on the box that runs the gates all three questions have always passed. That
// is exactly why the guard needs a platform this test brings with it.
//
// A SEPARATE EXECUTABLE, for the reason tests/vllm/entrypoints/
// test_device_selection.cpp is one: it registers a fake platform and backend in
// the otherwise-unused XPU slot AND over the CUDA slot, so `CurrentPlatform()`
// resolves to the fake whether or not this build has a real CUDA platform.
// Process isolation keeps those global registries out of every other suite.
//
// WHY THE FAKE REPORTS kXPU. `static_cast<vt::DeviceType>(1)` is `kCUDA`
// (include/vt/device.h: kCPU = 0, kCUDA = 1). `kXPU` is 4. So an assertion that
// device 1 resolves to kXPU is one the old integer cast could not satisfy under
// any circumstances — it is the seam being asked, not a constant being returned.
//
// WHERE EACH CASE ENTERS (AGENTS.md `## Nothing lands dead`). Both lanes are
// entered through a PRODUCTION entry point, not by calling the resolver:
// `vllm_video_engine_load` (include/vllm.h) → `LoadVideoEngine`
// (src/vllm/multimodal/video_engine.cpp) → the family registration → the
// engine's `Load`, which is where each lane's device question is asked. The
// direct `MiniMaxH3VideoDeviceType(1)` cases below stay, because they localise a
// failure — but they are NOT the reach proof, and on their own they measure a
// free function rather than a capability. Measured: replacing H3's `Load`-time
// `MiniMaxH3VideoDeviceType(params.device)` with the pre-row
// `params.device == 0 ? kCPU : kCUDA` left every direct case, and the whole of
// test_minimax_h3_video_fold, GREEN. The case that enters at `LoadVideoEngine`
// is the one that goes RED.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/multimodal/ltx2_video.h"
#include "vllm/multimodal/minimax_h3_video.h"
#include "vllm/multimodal/video_engine.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

// The backend COUNTS ITS TRANSFERS, and classifies each by direction, because
// that count is the only instrument this project has for the residency claim in
// #1451 and it must be readable on a box with no GPU.
//
// Direction is derived from the live allocation table rather than passed in:
// `vt::Backend::Copy` takes two raw pointers and no direction argument, so a
// pointer inside a range this backend handed out is DEVICE memory and anything
// else is host memory. That is exact here — `Alloc` is the only source of device
// pointers — and it is what lets the test say "the volume is never downloaded
// mid-decode" rather than the much weaker "some copies happened".
class FakeXpuBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    const size_t n = bytes == 0 ? 1 : bytes;
    void* p = std::malloc(n);
    ++allocs;
    live_[p] = n;
    return p;
  }
  void Free(void* p) override {
    live_.erase(p);
    std::free(p);
  }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    const bool dst_dev = IsDevice(dst);
    const bool src_dev = IsDevice(src);
    if (dst_dev && !src_dev) ++h2d;
    else if (!dst_dev && src_dev) ++d2h;
    else if (dst_dev && src_dev) ++d2d;
    else ++h2h;
    std::memcpy(dst, src, bytes);
  }

  void ResetCounters() { h2d = d2h = d2d = h2h = allocs = 0; }

  unsigned h2d = 0, d2h = 0, d2d = 0, h2h = 0, allocs = 0;
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }

 private:
  bool IsDevice(const void* p) const {
    const char* c = static_cast<const char*>(p);
    for (const auto& kv : live_) {
      const char* base = static_cast<const char*>(kv.first);
      if (c >= base && c < base + kv.second) return true;
    }
    return false;
  }
  std::map<void*, size_t> live_;
};

// A PARTIAL backend, shaped exactly like MetalPlatform and TenstorrentPlatform:
// a real accelerator with a registered backend that names the architectures
// whose kernels it has, and declines everything else. `accepts_everything` flips
// it into a COMPLETE backend so the same process can prove the guard does not
// refuse a configuration that works.
class PartialXpuPlatform final : public vllm::platforms::Platform {
 public:
  explicit PartialXpuPlatform(FakeXpuBackend& backend) : backend_(backend) {}

  vt::DeviceType device_type() const override { return vt::DeviceType::kXPU; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {}; }
  std::vector<vt::DType> supported_dtypes() const override { return {vt::DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }

  bool supports_model_architecture(std::string_view architecture) const override {
    if (accepts_everything) return true;
    return architecture == "OPTForCausalLM";
  }

  bool accepts_everything = false;

 private:
  FakeXpuBackend& backend_;
};

FakeXpuBackend& Backend() {
  static FakeXpuBackend backend;
  return backend;
}

PartialXpuPlatform& Platform() {
  static PartialXpuPlatform platform(Backend());
  return platform;
}

// Registered into the CUDA slot as well as its own: `CurrentPlatform()` walks
// {kCUDA, kROCM, kXPU, …} and returns the first REGISTERED entry
// (src/vllm/platforms/platform.cpp), so without this a CUDA build would resolve
// to the real CUDA platform and the guard would never be exercised.
void RegisterPartialAccelerator(bool accepts_everything) {
  vt::RegisterBackend(vt::DeviceType::kXPU, &Backend());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kXPU, &Platform());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &Platform());
  Platform().accepts_everything = accepts_everything;
}

// A path that does not exist. If the capability guard were reverted, the LTX-2.5
// load would sail past it and die opening this instead — a DIFFERENT message,
// which is what makes the assertions below fail for the intended reason rather
// than by accident.
vllm::multimodal::VideoModelParams DeviceOneParams() {
  vllm::multimodal::VideoModelParams mp;
  mp.family = vllm::multimodal::kLtx2VideoFamily;
  mp.dit_path = "/nonexistent/ltx2-dit-that-is-never-opened.safetensors";
  mp.device = 1;
  return mp;
}

// The H3 sibling of DeviceOneParams(), entered through the SAME generic struct
// so it can go in at `LoadVideoEngine` with the family declared. The path does
// not exist for the same reason: a load that got past the capability clause
// would die opening it, with a different message.
vllm::multimodal::VideoModelParams H3DeviceOneParams() {
  vllm::multimodal::VideoModelParams mp;
  mp.family = vllm::multimodal::kMiniMaxH3VideoFamily;
  mp.dit_path = "/nonexistent/h3-dit-that-is-never-opened.gguf";
  mp.device = 1;
  return mp;
}

// The PRODUCTION entry, for BOTH lanes: `vllm_video_engine_load`
// (include/vllm.h) calls `LoadVideoEngine` through src/capi/vllm_c.cpp, and it
// dispatches on the declared family to the registration in the engine's own TU,
// which calls that engine's `Load`. Nothing in this helper names an engine type,
// which is the point — it is the path a user arrives on, and entering at
// `Ltx2VideoEngine::Load` or `MiniMaxH3VideoEngine::Load` instead would skip the
// registry hop and prove one link less.
std::string LoadError(const vllm::multimodal::VideoModelParams& mp) {
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// The architecture name in the refusal's QUOTED SLOT, rather than anywhere in
// the message.
//
// `Fail()` in src/vllm/multimodal/ltx2_video.cpp prefixes EVERY message it
// throws with "ltx-2.5 video: ", and `kLtx2VideoFamily` is the string
// "ltx-2.5". So `msg.find(kLtx2VideoFamily)` is satisfied by that boilerplate on
// every refusal this file can produce — including the two refusals the
// assertions below exist to distinguish this one FROM — and no defect in the
// message can make it fail. Measured: replacing the family name with
// "<redacted>" inside the DECLINES `Fail` left it GREEN.
//
// The row's thesis for #659 is that a partial backend declines BY NAME, so the
// assertion has to be that the architecture appears where the refusal names it.
// The H3 pair happens not to collide — its prefix spells the family with an
// underscore ("minimax_h3 video: ") while the family is hyphenated
// ("minimax-h3") — but that is a coincidence of spelling and not a property, so
// the H3 cases are built the same way.
std::string QuotedArchitecture(std::string_view family) {
  return std::string("architecture '") + std::string(family) + "'";
}

// --- W5 (#1007): the smallest conv video decoder that decodes ---------------
//
// `decoder_blocks = {}`, patch_size 1, no timestep conditioning, so the decode
// is exactly `conv_in` -> PixelNorm -> SiLU -> `conv_out` -> unpatchify: TWO
// convolutions and no noise draw. The weights are drawn from a deterministic LCG
// rather than set to constants, because a symmetric weight can hide a
// transposed axis and this case's whole job is to compare two arms element by
// element.
struct TinyDecoder {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  vllm::Ltx2VaeWeights weights;
  std::vector<float> latent;
  int64_t lt = 3, lh = 5, lw = 4;
};

TinyDecoder MakeTinyDecoder() {
  TinyDecoder d;
  d.cfg.prefix = "w5.dev.";
  d.cfg.in_channels = 1;
  d.cfg.out_channels = 1;
  d.cfg.patch_size = 1;
  d.cfg.base_channels = 8;
  d.cfg.causal = false;
  d.cfg.timestep_conditioning = false;
  d.cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  d.cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  d.cfg.decoder_blocks = {};

  uint64_t seed = 20260820ULL;
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((seed >> 33) % 20001) / 10000.0f - 1.0f;
  };
  auto fill = [&next](size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = next();
    return v;
  };

  const std::string p = d.cfg.prefix;
  d.weights.tensors[p + "per_channel_statistics.std-of-means"] = {1.0f};
  d.weights.tensors[p + "per_channel_statistics.mean-of-means"] = {0.0f};
  d.weights.tensors[p + "conv_in.conv.weight"] = fill(8 * 1 * 27);
  d.weights.tensors[p + "conv_in.conv.bias"] = fill(8);
  d.weights.tensors[p + "conv_out.conv.weight"] = fill(1 * 8 * 27);
  d.weights.tensors[p + "conv_out.conv.bias"] = fill(1);
  d.latent = fill(static_cast<size_t>(d.lt * d.lh * d.lw));
  return d;
}

}  // namespace

// A decoder with STAGES BETWEEN ITS CONVOLUTIONS, which the W5 fixture above
// deliberately does not have (`decoder_blocks = {}`). The residency claim is
// about what happens BETWEEN two convolutions, so a fixture with one stage
// cannot discriminate: it needs resnet blocks, a norm, an ada-LN-free SiLU and a
// depth-to-space upsample in the walk.
TinyDecoder MakeStagedDecoder() {
  TinyDecoder d = MakeTinyDecoder();
  d.cfg.prefix = "r1451.dev.";
  d.cfg.base_channels = 4;
  // GROUPNORM, TIMESTEP CONDITIONING AND NOISE INJECTION ARE ALL ON, and each
  // one is here because its absence hid a defect. With `kPixelNorm` and no
  // conditioning the fixture never reaches `group_norm`, `ada_ln` or
  // `spatial_noise` at all, and the first draft of this change passed every one
  // of them a HOST pointer into a kernel dispatched on the queue's device --
  // which the `FakeXpuBackend` below executes without complaint, because it is a
  // `memcpy` over `malloc` and its "device" memory IS host memory. A discrete
  // GPU would have read unmapped memory. A fixture that does not enter a path
  // cannot gate it.
  d.cfg.norm_layer = vllm::Ltx2NormLayer::kGroupNorm;
  d.cfg.norm_num_groups = 2;
  d.cfg.timestep_conditioning = true;
  d.cfg.decode_timestep = 0.05;
  d.cfg.decode_noise_scale = 0.025;
  d.cfg.decoder_blocks = {{"res_x", 2, 0, true, false}, {"compress_space", 1, 1, false, false}};

  uint64_t seed = 14512026ULL;
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((seed >> 33) % 20001) / 10000.0f - 1.0f;
  };
  auto fill = [&next](size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = next();
    return v;
  };
  const std::string p = d.cfg.prefix;
  const int64_t mid = 4;  // base_channels * multiplier, with compress_space's multiplier 1
  d.weights.tensors.clear();
  d.weights.tensors[p + "per_channel_statistics.std-of-means"] = {1.0f};
  d.weights.tensors[p + "per_channel_statistics.mean-of-means"] = {0.0f};
  d.weights.tensors[p + "timestep_scale_multiplier"] = {1000.0f};
  d.weights.tensors[p + "conv_in.conv.weight"] = fill(static_cast<size_t>(mid * 1 * 27));
  d.weights.tensors[p + "conv_in.conv.bias"] = fill(static_cast<size_t>(mid));
  // up_blocks.0 is the LAST entry of decoder_blocks, walked in reverse: compress_space.
  d.weights.tensors[p + "up_blocks.0.conv.conv.weight"] =
      fill(static_cast<size_t>(mid * 4 * mid * 27));
  d.weights.tensors[p + "up_blocks.0.conv.conv.bias"] = fill(static_cast<size_t>(mid * 4));
  // up_blocks.1 is res_x: a per-block timestep embedder, then two res_blocks that
  // each carry two GroupNorms, an ada-LN table and two per-channel noise scales.
  const std::string tb = p + "up_blocks.1.time_embedder.timestep_embedder.";
  d.weights.tensors[tb + "linear_1.weight"] = fill(static_cast<size_t>(mid * 4 * 256));
  d.weights.tensors[tb + "linear_1.bias"] = fill(static_cast<size_t>(mid * 4));
  d.weights.tensors[tb + "linear_2.weight"] = fill(static_cast<size_t>(mid * 4 * mid * 4));
  d.weights.tensors[tb + "linear_2.bias"] = fill(static_cast<size_t>(mid * 4));
  for (int i = 0; i < 2; ++i) {
    const std::string b = p + "up_blocks.1.res_blocks." + std::to_string(i);
    d.weights.tensors[b + ".norm1.weight"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".norm1.bias"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".norm2.weight"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".norm2.bias"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".scale_shift_table"] = fill(static_cast<size_t>(4 * mid));
    d.weights.tensors[b + ".per_channel_scale1"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".per_channel_scale2"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".conv1.conv.weight"] = fill(static_cast<size_t>(mid * mid * 27));
    d.weights.tensors[b + ".conv1.conv.bias"] = fill(static_cast<size_t>(mid));
    d.weights.tensors[b + ".conv2.conv.weight"] = fill(static_cast<size_t>(mid * mid * 27));
    d.weights.tensors[b + ".conv2.conv.bias"] = fill(static_cast<size_t>(mid));
  }
  d.weights.tensors[p + "conv_norm_out.weight"] = fill(static_cast<size_t>(mid));
  d.weights.tensors[p + "conv_norm_out.bias"] = fill(static_cast<size_t>(mid));
  const std::string lb = p + "last_time_embedder.timestep_embedder.";
  d.weights.tensors[lb + "linear_1.weight"] = fill(static_cast<size_t>(mid * 2 * 256));
  d.weights.tensors[lb + "linear_1.bias"] = fill(static_cast<size_t>(mid * 2));
  d.weights.tensors[lb + "linear_2.weight"] = fill(static_cast<size_t>(mid * 2 * mid * 2));
  d.weights.tensors[lb + "linear_2.bias"] = fill(static_cast<size_t>(mid * 2));
  d.weights.tensors[p + "last_scale_shift_table"] = fill(static_cast<size_t>(2 * mid));
  d.weights.tensors[p + "conv_out.conv.weight"] = fill(static_cast<size_t>(1 * mid * 27));
  d.weights.tensors[p + "conv_out.conv.bias"] = fill(1);
  d.latent = fill(static_cast<size_t>(d.lt * d.lh * d.lw));
  return d;
}

// The two arms must see the SAME draws, or the pixels differ for a reason that
// has nothing to do with residency. `Ltx2NoiseStream` is the reproducibility
// seam and it stays on the host on both arms, so a deterministic sequence here
// is the whole of what the comparison needs.
class CountingNoise final : public vllm::Ltx2NoiseStream {
 public:
  std::vector<float> Draw(int64_t count) override {
    ++draws;
    std::vector<float> v(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      seed_ = seed_ * 6364136223846793005ULL + 1442695040888963407ULL;
      v[static_cast<size_t>(i)] =
          static_cast<float>((seed_ >> 33) % 20001) / 10000.0f - 1.0f;
    }
    return v;
  }
  int draws = 0;

 private:
  uint64_t seed_ = 987654321ULL;
};

// The COMPLEMENT of MakeStagedDecoder, and it exists because measurement said the
// first fixture was not enough. Instrumenting each CPU arm with its queue's
// device type showed only six of the ten kernels ever reached a non-CPU queue:
// `pixel_norm`, `frame_slice`, `channel_repeat` and `linear_cn` were exercised by
// the goldens on the host and by NOTHING on a device. A kernel table whose arms
// are never dispatched on the device they were written for is the `## Nothing
// lands dead` failure one level down from the usual one.
//
// This config reaches all four:
//   * `norm_layer = kPixelNorm`          -> pixel_norm (and NO norm weights, because
//                                           PixelNorm is parameter-free)
//   * a `res_x_y` block that HALVES the channels -> the shortcut path, so
//                                           `norm3` (a one-group GroupNorm) and
//                                           `Linear3d` -> linear_cn
//   * `compress_all` with `residual` and a temporal stride of 2
//                                        -> channel_repeat AND frame_slice
TinyDecoder MakeShortcutDecoder() {
  TinyDecoder d = MakeTinyDecoder();
  d.cfg.prefix = "r1451.alt.";
  d.cfg.base_channels = 4;
  d.cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  d.cfg.timestep_conditioning = false;
  // Walked in REVERSE, so up_blocks.0 is the LAST entry here: compress_all runs
  // FIRST, while the volume is still 8 channels. That order is forced, not
  // cosmetic -- `compress_all`'s residual rearranges the INPUT by the same
  // 2x2x2, so the input channel count must be divisible by 8 (sampling.py:98-110).
  // Running it after res_x_y halved the volume to 4 makes `expand` divide 4 by 8
  // and produce a zero-channel skip, which the shape check catches by name.
  d.cfg.decoder_blocks = {{"res_x_y", 1, 2, false, false},
                          {"compress_all", 1, 1, false, /*residual=*/true}};

  uint64_t seed = 20260825ULL;
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((seed >> 33) % 20001) / 10000.0f - 1.0f;
  };
  auto fill = [&next](size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = next();
    return v;
  };
  const std::string p = d.cfg.prefix;
  // conv_in widens to base_channels * multiplier; multiplier is 2 (res_x_y)
  // times 1 (compress_all), so the bottleneck is 8 and res_x_y halves it to 4.
  const int64_t wide = 8, narrow = 4;
  d.weights.tensors.clear();
  d.weights.tensors[p + "per_channel_statistics.std-of-means"] = {1.0f};
  d.weights.tensors[p + "per_channel_statistics.mean-of-means"] = {0.0f};
  d.weights.tensors[p + "conv_in.conv.weight"] = fill(static_cast<size_t>(wide * 1 * 27));
  d.weights.tensors[p + "conv_in.conv.bias"] = fill(static_cast<size_t>(wide));
  // up_blocks.0 = compress_all, stride 2x2x2 over the 8-channel volume,
  // reduction 1: conv widens to 8*8/1 = 64, expand divides by 8 back to 8, and
  // the residual repeats its own 1-channel expansion 8 times to match.
  d.weights.tensors[p + "up_blocks.0.conv.conv.weight"] =
      fill(static_cast<size_t>(64 * wide * 27));
  d.weights.tensors[p + "up_blocks.0.conv.conv.bias"] = fill(64);
  // up_blocks.1 = res_x_y: 8 -> 4, so the shortcut branch is taken.
  const std::string b = p + "up_blocks.1";
  d.weights.tensors[b + ".conv1.conv.weight"] = fill(static_cast<size_t>(narrow * wide * 27));
  d.weights.tensors[b + ".conv1.conv.bias"] = fill(static_cast<size_t>(narrow));
  d.weights.tensors[b + ".conv2.conv.weight"] = fill(static_cast<size_t>(narrow * narrow * 27));
  d.weights.tensors[b + ".conv2.conv.bias"] = fill(static_cast<size_t>(narrow));
  d.weights.tensors[b + ".norm3.weight"] = fill(static_cast<size_t>(wide));
  d.weights.tensors[b + ".norm3.bias"] = fill(static_cast<size_t>(wide));
  d.weights.tensors[b + ".conv_shortcut.weight"] = fill(static_cast<size_t>(narrow * wide));
  d.weights.tensors[b + ".conv_shortcut.bias"] = fill(static_cast<size_t>(narrow));
  d.weights.tensors[p + "conv_out.conv.weight"] = fill(static_cast<size_t>(1 * narrow * 27));
  d.weights.tensors[p + "conv_out.conv.bias"] = fill(1);
  d.latent = fill(static_cast<size_t>(d.lt * d.lh * d.lw));
  return d;
}

TEST_CASE("ltx2 vae: the SHORTCUT and RESIDUAL-UPSAMPLE stages are resident too") {
  // #1451. Companion to the case below, covering the four kernels that one does
  // not reach on a device queue. THREE assertions matter here, and the third is
  // the one a fresh review had to add: the volume comes back exactly once, the
  // pixels match the host arm bit for bit, and no stage fell back to the CPU
  // queue. The first two cannot see the third -- see the note beside it.
  const TinyDecoder d = MakeShortcutDecoder();

  const vllm::Ltx2VideoFrames host =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                /*noise=*/nullptr, /*timestep=*/nullptr, /*queue=*/nullptr);
  REQUIRE(!host.data.empty());

  RegisterPartialAccelerator(/*accepts_everything=*/true);
  vt::RegisterOp(vt::OpId::kConv3d, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kConv3d, vt::DeviceType::kCPU));
  vt::RegisterOp(vt::OpId::kLtx2, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kLtx2, vt::DeviceType::kCPU));
  vt::RegisterOp(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU));
  vt::RegisterOp(vt::OpId::kAdd, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kAdd, vt::DeviceType::kCPU));
  Backend().ResetCounters();
  vt::EnableOpProviderCallStats(true);
  vt::ResetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU);
  vt::ResetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU);

  vt::Queue q{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  const vllm::Ltx2VideoFrames dev =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                /*noise=*/nullptr, /*timestep=*/nullptr, &q);

  const vt::OpProviderStats xpu =
      vt::GetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU);
  const vt::OpProviderStats cpu =
      vt::GetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU);
  vt::EnableOpProviderCallStats(false);

  INFO("host<-device transfers: " << Backend().d2h << ", host->device: " << Backend().h2d);
  CHECK(Backend().d2h == 1u);

  // EXCLUSIVE DISPATCH, and this case needs it as much as its sibling does.
  // Without the `kCPU == 0` half, a stage that quietly took the CPU queue --
  // `PixelNorm(nullptr, ...)` instead of `PixelNorm(config.queue, ...)`, say --
  // still produces the right pixels on a unified-memory backend and still
  // downloads exactly once, so `d2h` and the `memcmp` both stay green. A fresh
  // review proved that against the first draft of this case: that one mutation
  // left the whole seam suite passing 9 of 9. The stages this fixture owns are
  // pixel-norm, the one-group GroupNorm of the shortcut, `Linear3d`,
  // depth-to-space, the channel repeat and the frame slice, and this pair is
  // what keeps them on the queue the decode was handed.
  INFO("kLtx2Vae dispatches: xpu=" << xpu.selections << " cpu=" << cpu.selections);
  CHECK(xpu.selections > 0u);
  CHECK(cpu.selections == 0u);

  REQUIRE(dev.channels == host.channels);
  REQUIRE(dev.frames == host.frames);
  REQUIRE(dev.data.size() == host.data.size());
  CHECK(std::memcmp(dev.data.data(), host.data.data(), host.data.size() * sizeof(float)) == 0);
}

TEST_CASE("ltx2 vae: the video decode's VOLUME IS NEVER DOWNLOADED between two convolutions") {
  // #1451, LTX25-VAE-DEVICE-RESIDENCY. W5 (#1007) put the CONVOLUTION on the
  // device and left every stage between two convolutions as a host loop, so the
  // decode uploaded its input, its weight and its bias and DOWNLOADED ITS
  // OUTPUT once per `nn.Conv3d` call.
  //
  // THIS IS A DIVERGENCE AND NOT ONLY A COST, and that is why the assertion is
  // on the DOWNLOAD count rather than on a wall clock. Upstream never moves the
  // tensor back: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
  // builds the decoder onto a device once
  // (packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273),
  // the latent follows the weights
  // (packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py:283-284),
  // and a grep for `.cpu()` over that whole package returns only the checkpoint
  // loader and the DIFFUSION decoder's timestep schedule -- the conv decoder's
  // forward contains no host round-trip at all.
  //
  // WHAT THIS CASE CAN AND CANNOT ESTABLISH. `FakeXpuBackend` is a `memcpy` over
  // `malloc`, so the count below is a STRUCTURAL fact and its cost is zero. This
  // case proves the volume stays in device memory across the whole walk. It
  // proves NOTHING about speed, and nothing about any CUDA kernel: no lease was
  // taken for this row and #1452 records that no `.cu` on this lane has ever been
  // compiled or executed in this project's reach. See
  // .agents/specs/ltx25-vae-device-residency.md
  // `## What a CPU-only run can and cannot establish`.
  const TinyDecoder d = MakeStagedDecoder();

  CountingNoise host_noise;
  const vllm::Ltx2VideoFrames host =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                &host_noise, /*timestep=*/nullptr, /*queue=*/nullptr);
  REQUIRE(!host.data.empty());
  // The fixture must actually REACH the noise-injection and timestep paths -- a
  // config that quietly skipped them would make the whole comparison vacuous.
  REQUIRE(host_noise.draws > 0);

  RegisterPartialAccelerator(/*accepts_everything=*/true);
  vt::RegisterOp(vt::OpId::kConv3d, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kConv3d, vt::DeviceType::kCPU));
  vt::RegisterOp(vt::OpId::kLtx2, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kLtx2, vt::DeviceType::kCPU));
  vt::RegisterOp(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU));
  // The two residual accumulates route through the SHARED `vt::Add` rather than
  // an eleventh VAE kernel, so the fake accelerator needs it too. Its absence is
  // not a silent fallback here -- `vt::GetOp` refuses by name, because this
  // backend reports unified memory but NOT host-addressable device memory, and
  // the portable CPU reference tier correctly declines to dereference what it
  // did not allocate.
  vt::RegisterOp(vt::OpId::kAdd, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kAdd, vt::DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU));

  vt::EnableOpProviderCallStats(true);
  vt::ResetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU);
  vt::ResetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU);
  Backend().ResetCounters();

  vt::Queue q{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  CountingNoise dev_noise;
  const vllm::Ltx2VideoFrames dev =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                &dev_noise, /*timestep=*/nullptr, &q);
  REQUIRE(dev_noise.draws == host_noise.draws);

  const unsigned d2h = Backend().d2h;
  const unsigned h2d = Backend().h2d;
  const vt::OpProviderStats xpu =
      vt::GetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kXPU);
  const vt::OpProviderStats cpu =
      vt::GetOpProviderStats(vt::OpId::kLtx2Vae, vt::DeviceType::kCPU);
  vt::EnableOpProviderCallStats(false);

  // THE RESIDENCY ASSERTION. Exactly ONE device-to-host transfer for the whole
  // decode: the finished frames. Any stage that computes on the host needs the
  // volume back, so a second download is the defect this issue names, stated as
  // a number a box with no GPU can read.
  INFO("host<-device transfers: " << d2h << ", host->device transfers: " << h2d);
  CHECK(d2h == 1u);

  // AND THE WEIGHTS ARE STAGED ONCE. The upload count is a function of the
  // DISTINCT TENSOR COUNT, not of the convolution count or the stage count, and
  // the arithmetic is written out so the number is checkable rather than
  // recorded:
  //
  //    1  the latent, uploaded once after the host prologue
  //    2  conv_in                  .conv.weight + .conv.bias
  //    2  up_blocks.0              .conv.conv.weight + .conv.conv.bias
  //   22  up_blocks.1.res_blocks   two blocks x eleven tensors each
  //          (norm1 w+b, norm2 w+b, scale_shift_table,
  //           per_channel_scale1+2, conv1 w+b, conv2 w+b)
  //    2  conv_norm_out            .weight + .bias
  //    1  last_scale_shift_table
  //    2  conv_out                 .conv.weight + .conv.bias
  //    5  the timestep embeddings, which are COMPUTED per ada-LN call on the
  //          host and so are per-call scratch rather than cached weights:
  //          four inside the two res_blocks, one for the tail
  //    4  the spatial-noise planes, drawn per injection site by
  //          `Ltx2NoiseStream`, which stays on the host by design
  //  ---
  //   41
  //
  // THE CACHE IS GENUINELY EXERCISED HERE, and it was not by the first version
  // of this fixture. `scale_shift_table` is fetched TWICE per res_block --
  // rows (0,1) before the first convolution and rows (2,3) before the second
  // (resnet.py:135-148) -- so four fetches resolve to two uploads. A fixture in
  // which no tensor is fetched twice cannot tell a cache from no cache at all,
  // and the mutation that deletes the cache lookup passed against exactly such a
  // fixture before this one replaced it.
  //
  // W5 re-sent a weight AND its bias on every `nn.Conv3d` call, and uploaded the
  // whole volume as well, so this number grew with the convolution count.
  // Upstream stages the decoder's parameters at BUILD time and never moves them
  // again (single_gpu_model_builder.py:273).
  CHECK(h2d == 41u);

  // Every between-convolution stage dispatched on the QUEUE'S device. Asserting
  // only that the xpu counter moved would pass an implementation that ran both
  // arms; the cpu counter staying at zero is what makes it exclusive. That is
  // the same argument the W5 case above makes for kConv3d.
  INFO("kLtx2Vae dispatches: xpu=" << xpu.selections << " cpu=" << cpu.selections);
  CHECK(xpu.selections > 0u);
  CHECK(cpu.selections == 0u);

  // And the pixels are the same ones. A resident arm that is fast and wrong is
  // not a port.
  REQUIRE(dev.data.size() == host.data.size());
  CHECK(std::memcmp(dev.data.data(), host.data.data(), host.data.size() * sizeof(float)) == 0);
}

TEST_CASE("ltx2 vae: the video decode RUNS ITS CONVOLUTION on a non-CPU queue, byte-identically") {
  // W5, #1007. The LTX-2.5 video VAE decode had no device arm at all: `vt` had
  // no 3-D convolution on any device, so the ~7.25 TFLOP of dense 3x3x3
  // convolution ran as host loops while every oracle runs the same decode
  // GPU-resident (Lightricks/LTX-2 @ fd4ded7f2
  // packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273
  // defaults the decoder onto CUDA; vLLM-Omni @ a4ea67a21
  // vllm_omni/diffusion/models/interface.py:92 says "VAE(s) (always on GPU)").
  //
  // WHAT THIS CASE MEASURES, AND WHAT IT DOES NOT. It executes the whole device
  // arm — the upload through `vt::Backend::Alloc`/`Copy`, a `vt::Conv3d`
  // dispatch on a device that is NOT `kCPU`, and the download — and requires the
  // pixels to be `memcmp`-identical to the host arm. It does NOT prove a GPU
  // runs it: that is hardware, this box has none, and it is owed. What it
  // removes is the failure mode where the device branch exists, compiles, and
  // has never once been entered.
  //
  // WHY THIS EXECUTABLE. `vt::DeviceType` is a closed enum
  // (include/vt/device.h), so a fake accelerator must impersonate a real vendor
  // and would flip `CurrentPlatform()` for every case in its process. This file
  // is already a separate executable for exactly that reason, and it already
  // carries a unified-memory `FakeXpuBackend`.
  //
  // WHY REGISTERING THE CPU KERNEL FOR kXPU IS THE RIGHT INSTRUMENT AND NOT A
  // TAUTOLOGY. The kernel is not what is under test here — `test_ops_conv3d`
  // gates the arithmetic against an independent scalar reference. What is under
  // test is the MARSHALLING: that the decode allocates on the queue's backend,
  // copies both operands and the bias in the right shapes, dispatches on the
  // queue's device rather than on kCPU, and copies the result back. A wrong
  // extent, a missing bias upload, or a dispatch that silently fell back to the
  // host would all show here. `vt::RegisterOp` is the public registration API
  // (include/vt/op_provider.h:127) and this is a NATIVE registration, the same
  // call `src/vt/cpu/cpu_conv3d.cpp` makes.
  const TinyDecoder d = MakeTinyDecoder();

  const vllm::Ltx2VideoFrames host =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                /*noise=*/nullptr, /*timestep=*/nullptr, /*queue=*/nullptr);
  REQUIRE(!host.data.empty());

  vt::RegisterBackend(vt::DeviceType::kXPU, &Backend());
  REQUIRE(vt::TryGetBackend(vt::Device{vt::DeviceType::kXPU, 0}) != nullptr);
  vt::RegisterOp(vt::OpId::kConv3d, vt::DeviceType::kXPU,
                 vt::GetOp(vt::OpId::kConv3d, vt::DeviceType::kCPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kConv3d, vt::DeviceType::kXPU));

  vt::EnableOpProviderCallStats(true);
  vt::ResetOpProviderStats(vt::OpId::kConv3d, vt::DeviceType::kXPU);
  vt::ResetOpProviderStats(vt::OpId::kConv3d, vt::DeviceType::kCPU);

  vt::Queue q{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  const vllm::Ltx2VideoFrames dev =
      vllm::Ltx2ConvVideoDecode(d.cfg, d.weights, d.latent, d.cfg.in_channels, d.lt, d.lh, d.lw,
                                /*noise=*/nullptr, /*timestep=*/nullptr, &q);

  const vt::OpProviderStats xpu =
      vt::GetOpProviderStats(vt::OpId::kConv3d, vt::DeviceType::kXPU);
  const vt::OpProviderStats cpu =
      vt::GetOpProviderStats(vt::OpId::kConv3d, vt::DeviceType::kCPU);
  vt::EnableOpProviderCallStats(false);

  // The dispatch went to the QUEUE'S device. Asserting only that the xpu counter
  // moved would pass on an implementation that ran both arms; the cpu counter
  // staying at zero is what makes it exclusive.
  INFO("kConv3d dispatches: xpu=" << xpu.selections << " cpu=" << cpu.selections);
  CHECK(xpu.selections == 2u);
  CHECK(cpu.selections == 0u);

  REQUIRE(dev.channels == host.channels);
  REQUIRE(dev.frames == host.frames);
  REQUIRE(dev.height == host.height);
  REQUIRE(dev.width == host.width);
  REQUIRE(dev.data.size() == host.data.size());
  CHECK(std::memcmp(dev.data.data(), host.data.data(), host.data.size() * sizeof(float)) == 0);
}

TEST_CASE("ltx2 video: a platform that DECLINES the architecture refuses device 1 by name") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  // Both of the questions this file already asked PASS here: the platform is an
  // accelerator, and a backend is registered for it. Without the third question
  // the load proceeds into the DiT.
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kXPU);
  REQUIRE(vt::TryGetBackend(vt::DeviceType::kXPU) != nullptr);
  REQUIRE_FALSE(vllm::platforms::CurrentPlatform().supports_model_architecture(
      vllm::multimodal::kLtx2VideoFamily));

  const std::string msg = LoadError(DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  // BY NAME: the platform, the architecture, and the fact that the backend
  // declined rather than that something was malformed.
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kLtx2VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("supports_model_architecture") != std::string::npos);
  // And it must not be the OTHER refusal: a backend IS registered here, so
  // blaming a missing one would be a wrong diagnosis that reads as a right one.
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);
  // Nor the shape of failure the guard exists to prevent: opening the DiT.
  CHECK(msg.find("/nonexistent/") == std::string::npos);
}

TEST_CASE("ltx2 video: a COMPLETE backend is not refused by the capability guard") {
  // The risk the guard carries is refusing a configuration that works. A
  // platform that accepts the architecture must get past this clause and fail
  // only on the thing that is genuinely wrong — the missing checkpoint.
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  const std::string msg = LoadError(DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DECLINES") == std::string::npos);
  CHECK(msg.find("supports_model_architecture") == std::string::npos);
  // POSITIVELY: the load got PAST the capability clause and failed on the thing
  // that is genuinely wrong — the checkpoint that does not exist. Asserting only
  // the two absences above would pass on any OTHER wrong failure (a refusal
  // reworded, a throw from an earlier clause, a message that never mentions the
  // path), which would leave "the guard does not refuse a working configuration"
  // unproven while reading green. Case 1 at the top of this file asserts the
  // mirror image of this line; the pair is what pins the guard to exactly one
  // arm each.
  CHECK(msg.find("/nonexistent/ltx2-dit-that-is-never-opened.safetensors") !=
        std::string::npos);
}

TEST_CASE("minimax_h3 video: device 1 RESOLVES through the seam, it is not enum value 1") {
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  // kXPU is 4. The integer cast this replaced returned kCUDA (1) for any build,
  // any platform, forever; it could not produce this answer.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(1) == vt::DeviceType::kXPU);
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(1) != vt::DeviceType::kCUDA);

  // The rest of the public contract is untouched by the change.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(0) == vt::DeviceType::kCPU);
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(-1));
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(2));
}

TEST_CASE("minimax_h3 video: a platform that DECLINES the architecture refuses device 1") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  std::string msg;
  try {
    (void)vllm::multimodal::MiniMaxH3VideoDeviceType(1);
    FAIL("device 1 must be refused when the resolved platform declines the architecture");
  } catch (const std::exception& e) {
    msg = e.what();
  }
  INFO(msg);
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kMiniMaxH3VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);

  // 0 still resolves, because the CPU asks none of these questions.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(0) == vt::DeviceType::kCPU);
}

// ── REACH (AGENTS.md `## Nothing lands dead`) ────────────────────────────────
//
// The two cases above call `MiniMaxH3VideoDeviceType` directly, which measures
// the free function and NOT whether anything routes to it. The H3 engine's
// `Load` is the only production caller (src/vllm/multimodal/minimax_h3_video.cpp,
// the `MiniMaxH3VideoDeviceType(params.device)` line), and it is reached from
// `vllm_video_engine_load` → `LoadVideoEngine` → the `minimax_h3` registration.
// Delete that one call and every other H3 device assertion in this tree stays
// green; this case is what turns red.
//
// It is the H3 mirror of the two LTX cases at the top of this file, and it
// exists because the LTX half already had this proof and the H3 half did not.
TEST_CASE("minimax_h3 video: a DECLINING platform refuses device 1 through LoadVideoEngine") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kXPU);
  REQUIRE(vt::TryGetBackend(vt::DeviceType::kXPU) != nullptr);
  REQUIRE_FALSE(vllm::platforms::CurrentPlatform().supports_model_architecture(
      vllm::multimodal::kMiniMaxH3VideoFamily));

  const std::string msg = LoadError(H3DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kMiniMaxH3VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("supports_model_architecture") != std::string::npos);
  // Not the OTHER refusal: a backend IS registered for the resolved platform, so
  // blaming a missing one is a wrong diagnosis that reads as a right one.
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);
  // Nor the failure the guard exists to prevent, one step further on. Reverting
  // the `Load` call site to the integer cast lands here instead: `kCUDA` with no
  // CUDA backend registered in this process.
  CHECK(msg.find("/nonexistent/") == std::string::npos);
}

TEST_CASE("minimax_h3 video: a COMPLETE backend is not refused through LoadVideoEngine") {
  // The mirror arm, for the same reason the LTX pair has one: two absences pass
  // on any OTHER wrong failure, so the guard's "does not refuse a working
  // configuration" half needs a POSITIVE assertion about where the load got to.
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  const std::string msg = LoadError(H3DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DECLINES") == std::string::npos);
  CHECK(msg.find("supports_model_architecture") == std::string::npos);
  CHECK(msg.find("/nonexistent/h3-dit-that-is-never-opened.gguf") != std::string::npos);
}
