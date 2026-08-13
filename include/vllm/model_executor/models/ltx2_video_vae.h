// LTX-2.5 CONV VIDEO VAE — the convolutional video decoder, and the explicit
// refusal of the diffusion one.
//
// LTX-2.5 ships TWO video decoders behind one checkpoint field
// (`config.vae._class_name`, video_vae/model_configurator.py:18-34):
//
//   "CausalVideoAutoencoder" -> ConvVideoDecoder   — ported here
//   anything else            -> NADiffusionDecoder — NOT ported, REFUSED BY NAME
//
// The diffusion decoder is a neighborhood-attention model with its own row. Per
// .agents/specs/ltx-2-5.md section 0 item 2 it is refused with a message naming
// the missing piece and NEVER silently downgraded to the conv decoder — a
// downgrade would return a lower-quality render as if it were the requested one,
// which no gate in this project can detect.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                          <-  UPSTREAM
//   Ltx2ConvVideoDecode           <-  model/video_vae/conv_video_decoder.py:263-357
//   (block construction)          <-  model/video_vae/conv_video_decoder.py:61-143
//   (ResnetBlock3D)               <-  model/video_vae/resnet.py:12-186
//   (UNetMidBlock3D)              <-  model/video_vae/resnet.py:189-277
//   (CausalConv3d)                <-  model/video_vae/convolution.py:266-317
//   (DepthToSpaceUpsample)        <-  model/video_vae/sampling.py:68-123
//   (AttnBlock3D / _RMSNorm2D)    <-  model/video_vae/attention.py:11-69
//   (unpatchify)                  <-  model/video_vae/ops.py:35-60
//   (per-channel statistics)      <-  model/video_vae/ops.py:63-84
//   (PixArt timestep embedding)   <-  model/transformer/timestep_embedding.py:6-141
//   Ltx2ParseVideoDecoderKind     <-  model/video_vae/model_configurator.py:18-34
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * TEMPORAL PADDING IS A REPLICATED FIRST FRAME, NOT ZEROS. CausalConv3d
//    prepends `k_t - 1` copies of frame 0 (convolution.py:306-307). MiniMax-H3's
//    causal Conv3d zero-pads instead, so the two are NOT interchangeable even
//    though both put every temporal pad on the LEFT.
//  * TWO DIFFERENT PixelNorm EPSILONS. video_vae/resnet.py:46 and
//    conv_video_decoder.py:243 construct `PixelNorm()` with its DEFAULT eps of
//    1e-8, while the audio VAE reaches PixelNorm through
//    `build_normalization_layer`, which passes 1e-6 (normalization.py:58). Using
//    one value for both is a silent, tiny, everywhere-bias.
//  * DEPTH-TO-SPACE UNPACKS `(c p1 p2 p3)`, AND THE TEMPORAL STRIDE DROPS THE
//    FIRST FRAME (sampling.py:112-120). Getting the channel order wrong shuffles
//    pixels inside every 2x2 block; keeping the first frame shifts the whole clip.
//  * `unpatchify` DECOMPOSES CHANNELS AS `(c p r q)` WITH `h` TAKING q AND `w`
//    TAKING r (ops.py:50-58) — r and q are NOT interchangeable.
//
// ─── DTYPE ───────────────────────────────────────────────────────────────────
// Every buffer this header names is f32, because this is the CPU REFERENCE arm.
// Upstream runs the decoder in the CHECKPOINT's dtype instead
// (`sample.to(weights_dtype)` in, `sample.to(output_dtype)` out —
// conv_video_decoder.py:283-286, 355-356), and it has none of the float32 pin the
// audio tower carries. The bf16/NVFP4 arm that inherits the checkpoint dtype is
// owed by phase L6; see ltx2_video_vae.cpp for why no gate here can catch a dtype
// that is merely too WIDE.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"  // Ltx2VaeWeights

namespace vllm {

// video_vae/enums.py:4-6.
enum class Ltx2NormLayer { kGroupNorm, kPixelNorm };

// video_vae/enums.py:16-20. Only the modes a decoder can actually select are
// listed; `make_conv_nd` forwards the rest to torch, which this port refuses.
enum class Ltx2PaddingMode { kZeros, kReflect, kReplicate };

// Which decoder a checkpoint asks for.
enum class Ltx2VideoDecoderKind { kConv, kDiffusion };

// `_RMSNorm2D` is `F.normalize(x, dim=1) * (sqrt(C) * gamma)` (attention.py:11-30),
// so the denominator floor is torch's `F.normalize` DEFAULT eps of 1e-12 — an L2
// normalize, not a mean-square RMS, and not this project's usual rms_norm epsilon.
// Named so it can be pinned: mutation proves 1e-12 -> 0.0 leaves every golden
// green, because the reduced-dimension activations are O(1) and the floor never
// binds at that magnitude. It is still READ on every element, so the goldens are
// not blind to it in the other direction — 1e-12 -> 1.0 reds two encoder goldens
// at 0.000525832. And it decides whether an all-zero channel vector divides or
// produces NaN.
inline constexpr double kLtx2RmsNorm2dEps = 1e-12;

// `config.vae._class_name` -> the decoder kind, mirroring
// `_vae_class_name_from_metadata` + `VideoDecoderConfigurator.from_metadata`
// (video_vae/model_configurator.py:18-34, 242-250): the conv decoder is selected
// by the exact string "CausalVideoAutoencoder" and by an ABSENT field (upstream's
// default); anything else is the diffusion decoder.
Ltx2VideoDecoderKind Ltx2ParseVideoDecoderKind(const std::string& vae_class_name);

// One entry in `decoder_blocks`. `multiplier` 0 means "the upstream default for
// this block kind" — 2 for `res_x_y` (conv_video_decoder.py:98), 1 for every
// `compress_*` (conv_video_decoder.py:111, 120, 129).
struct Ltx2VideoDecoderBlock {
  std::string name;
  int64_t num_layers = 1;
  int64_t multiplier = 0;
  bool inject_noise = false;
  bool residual = false;
};

// ─── THE INVISIBLE-CONSTANT CLASS ────────────────────────────────────────────
// An HONEST LIMIT of these goldens, and it is a CLASS, not one instance. Any
// epsilon or floor that exists to stabilize a division is, by construction,
// hard for a reduced-dimension parity gate to reach DOWNWARD: the deterministic
// stream produces O(1) activations, so shrinking the term it guards changes
// nothing the tensor comparison can see. That is the honest form of the claim.
// "Accepts any value at all" is what this paragraph used to say, and it is FALSE
// even of its own members — the epsilon is still READ on every element, so a
// large enough value moves the output. Only a probe that FAILS TO REACH proves
// unreachable; a mutation that happens not to move anything proves nothing, and
// the direction and MAGNITUDE of the mutation are therefore part of the verdict.
// MEASURED, by mutating each in turn, with the bound each number actually holds:
//
//   kMiniMaxH3SnakeEps      1e-9  -> 0.0   green
//   kLtx2RmsNorm2dEps       1e-12 -> 0.0   green ...but NOT green upward:
//     escalating it to 1.0 REDS "the video ENCODER (*_res family)" and "the video
//     encoder CROPS a frame count that is not 1 + k*factor", both at 0.000525832
//     against the 5e-6 band. It never BINDS at the shipped value, and it is read
//     regardless — the two are different statements and only the first is true
//     of this constant.
//
// `kLtx2BweMelLogClamp` was listed with them and NO LONGER belongs — the third
// entry to leave this list for the same reason, which is why the verdict is now
// stated per-entry with the number that proves it. The arm that made it
// reachable, "ltx2 vae: the BWE mel log clamp is gated where it actually binds",
// landed with the pin itself; the line calling it invisible was written in the
// same change and was false the moment it shipped. 1e-5 -> 1e-8 REDS that arm at
// max|diff| = 0.144965 against the 5e-6 band (36 cases: 34 passed, 2 failed —
// the golden, and the constant assertion below it). What made it look invisible
// was the SCALE of the ordinary arm, not the constant's nature: that arm's raw
// mel minimum is ~4.4e-3 and never approaches the floor, so the reachable arm
// attenuates mel_basis by 1e-4 until every bin lands under it — and asserts the
// saturated-bin count rather than assuming it.
//
// `Ltx2ConvVideoDecoderConfig::pixel_norm_eps` was listed with them and NO LONGER
// belongs. The arm added to make `norm_eps` reachable — "ltx2 vae: the video
// decoder's norm_eps is gated where it BINDS" — runs its latent at a tenth of the
// usual scale, and that makes this epsilon a first-order term too: 1e-8 -> 1e-6
// now REDS that arm at max|diff| = 1.69305e-04 against the 5e-6 band. The fixture
// built to close one hole closed its neighbour with it, and the line claiming
// otherwise survived the change that falsified it. It stays pinned, in "ltx2 vae:
// the two PixelNorm epsilons stay different", for the reason a pin always earns:
// a regeneration that moves the constant and the goldens together.
//
// `Ltx2ConvVideoDecoderConfig::norm_eps` was listed here and DOES NOT BELONG. It
// is read on every arm that has a `res_x_y` block, PixelNorm included, because
// `norm3` is a GroupNorm built whenever `in_channels != out_channels`
// (resnet.py:93-97) and applied at resnet.py:178. Its 1e-6 -> 1e-4 mutation
// stayed green only because the norm3 in the shipped fixture divides by a
// variance of ~0.2 five blocks deep; at 1e-6 -> 1.0 the same golden moves 1.6e-2.
// That is a sensitivity property of one fixture, not invisibility, and it is now
// gated numerically by a fixture where the epsilon is a first-order term.
//
// So every member of the class is held by a SOURCE-ANCHORED CONSTANT ASSERTION in
// tests/vllm/models/test_ltx2_vae.cpp, cited to the upstream line that sets it,
// rather than by the tensor comparison — and a constant that is added later and
// left unpinned is a new hole, not a covered one. The three names above that LEFT
// the class keep their assertions as well: their goldens now move under a
// mutation, but a golden still cannot catch a regeneration that shifts the
// constant and the expected tensors together, and only the pin can.
struct Ltx2ConvVideoDecoderConfig {
  // Defaults mirror `_build_conv_video_decoder`
  // (video_vae/model_configurator.py:81-94).
  int64_t in_channels = 128;
  int64_t out_channels = 3;
  // In CHECKPOINT (encoder) order. The decoder walks it REVERSED, exactly as
  // conv_video_decoder.py:222 does.
  std::vector<Ltx2VideoDecoderBlock> decoder_blocks;
  int64_t patch_size = 4;
  Ltx2NormLayer norm_layer = Ltx2NormLayer::kPixelNorm;
  bool causal = false;
  bool timestep_conditioning = true;
  Ltx2PaddingMode spatial_padding_mode = Ltx2PaddingMode::kReflect;
  int64_t base_channels = 128;
  int64_t norm_num_groups = 32;
  double decode_noise_scale = 0.025;
  double decode_timestep = 0.05;
  // The GroupNorm arm's eps, and the one `res_x_y`'s shortcut norm3 uses.
  // `ResnetBlock3D.__init__` declares `eps: float = 1e-6` (video_vae/resnet.py:31)
  // and hands it to every nn.GroupNorm it builds (resnet.py:44, 65, 94);
  // `UNetMidBlock3D` carries the same value as `resnet_eps` (resnet.py:216).
  //
  // norm3 is the reason this is LIVE on a PixelNorm checkpoint too: it is built
  // whenever `in_channels != out_channels` (resnet.py:93-97) and applied to the
  // residual at resnet.py:178, and `norm_layer` does not gate it. Neither does a
  // checkpoint key — `_make_decoder_block` passes `eps=1e-6` / `resnet_eps=1e-6`
  // literally (conv_video_decoder.py:78, 103), so this field exists to be pinned
  // to that literal, and is gated numerically by the norm_eps arm in
  // tests/vllm/models/test_ltx2_vae.cpp.
  double norm_eps = 1e-6;
  // `PixelNorm()`'s DEFAULT (normalization.py:22), reached bare from
  // video_vae/resnet.py:46 and conv_video_decoder.py:243 — NOT the 1e-6 the audio
  // VAE gets through build_normalization_layer.
  double pixel_norm_eps = 1e-8;
  std::string prefix;
};

// The deterministic source for every `torch.randn` upstream draws, consumed in
// CALL ORDER. That is precisely the guarantee an upstream `torch.Generator`
// gives, and it is what makes the decoder's noise injection reproducible on both
// sides. A null stream means "no noise is available", which is an ERROR whenever
// the config asks for noise rather than a silent zero fill.
class Ltx2NoiseStream {
 public:
  virtual ~Ltx2NoiseStream() = default;
  virtual std::vector<float> Draw(int64_t count) = 0;
};

// A (channels, frames, height, width) clip in [-1, 1]-ish pixel space (upstream
// maps it to [0, 1] outside the decoder, conv_video_decoder.py:497-499).
struct Ltx2VideoFrames {
  int64_t channels = 0;
  int64_t frames = 0;
  int64_t height = 0;
  int64_t width = 0;
  std::vector<float> data;
};

// ConvVideoDecoder.forward at batch 1. `latent` is
// [latent_channels, latent_t, latent_h, latent_w], channel-major.
//
// `timestep` overrides `decode_timestep` when non-null (the decoder's own
// default is used otherwise, conv_video_decoder.py:304-305). `noise` must be
// non-null whenever `timestep_conditioning` is set or any block sets
// `inject_noise`.
Ltx2VideoFrames Ltx2ConvVideoDecode(const Ltx2ConvVideoDecoderConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const std::vector<float>& latent, int64_t latent_channels,
                                    int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                    Ltx2NoiseStream* noise, const double* timestep = nullptr);

// The seam a caller reaches for when it holds a checkpoint rather than a decided
// kind. `kConv` forwards to Ltx2ConvVideoDecode; `kDiffusion` THROWS, naming
// NADiffusionDecoder and its missing neighborhood-attention kernel. It never
// falls back.
Ltx2VideoFrames Ltx2VideoDecode(Ltx2VideoDecoderKind kind,
                                const Ltx2ConvVideoDecoderConfig& config,
                                const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                                int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                                int64_t latent_w, Ltx2NoiseStream* noise,
                                const double* timestep = nullptr);

}  // namespace vllm
