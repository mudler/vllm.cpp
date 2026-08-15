// LTX-2.5 CONV VIDEO VAE ENCODER — the LOAD path, row LTX25-IMAGE-COND (#644).
//
// Spec: .agents/specs/ltx25-image-conditioning.md §3.1.
//
// ─── WHY THIS IS ITS OWN TU ──────────────────────────────────────────────────
// The decoder's twin of this code lives in `ltx2_loader.cpp`, which is a 1400-line
// file two concurrent rows of the #644 campaign both need to touch. Putting the
// encoder's four functions there would have made that file the lock AGENTS.md
// §Records names. It is additive here instead, next to the header that declares
// the encoder it serves, and it duplicates only the three tiny config accessors
// (`ConfigGet` / `ConfigObject` / the two enum parsers) that `ltx2_loader.cpp`
// keeps in its own anonymous namespace and does not export.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f,
// packages/ltx-core/src/ltx_core/model/video_vae/
//   OURS                              <-  UPSTREAM
//   Ltx2VideoVaeEncoderKeyRules       <-  model_configurator.py:267-276
//   Ltx2ParseConvVideoEncoderConfig   <-  model_configurator.py:37-69, 72-78
//   Ltx2CheckpointHasVideoEncoder     <-  (the SDOps `with_matching` prefixes of
//                                          the same filter, asked as a question)
#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("ltx2 video vae encoder: " + why);
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

// `config.get(key, fallback)`, refusing a present-but-wrong-typed value rather
// than falling back — the same polarity `ltx2_loader.cpp:845-856` states for the
// decoder, and for the same reason: a checkpoint that says `"patch_size": "4"`
// means something, and treating it as absent builds a different latent grid.
template <typename T>
T ConfigGet(const nlohmann::json& config, const std::string& key, T fallback,
            const std::string& where) {
  const auto it = config.find(key);
  if (it == config.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (const std::exception&) {
    Fail("'" + where + "." + key + "' is present but not the expected type (" + it->dump() + ")");
  }
}

const nlohmann::json& ConfigObject(const nlohmann::json& parent, const std::string& key,
                                   const std::string& where) {
  static const nlohmann::json kEmpty = nlohmann::json::object();
  const auto it = parent.find(key);
  if (it == parent.end() || it->is_null()) return kEmpty;
  if (!it->is_object()) Fail("'" + where + "." + key + "' is not a JSON object");
  return *it;
}

Ltx2NormLayer ParseNormLayer(const std::string& name, const std::string& where) {
  if (name == "pixel_norm") return Ltx2NormLayer::kPixelNorm;
  if (name == "group_norm") return Ltx2NormLayer::kGroupNorm;
  Fail(where + ".norm_layer is '" + name + "'; only 'pixel_norm' and 'group_norm' are ported");
}

Ltx2PaddingMode ParsePaddingMode(const std::string& name, const std::string& where) {
  if (name == "zeros") return Ltx2PaddingMode::kZeros;
  if (name == "reflect") return Ltx2PaddingMode::kReflect;
  if (name == "replicate") return Ltx2PaddingMode::kReplicate;
  Fail(where + ".spatial_padding_mode is '" + name +
       "'; only 'zeros', 'reflect' and 'replicate' are ported");
}

// `LogVarianceType` (video_vae/enums.py:9-13). The DEFAULT is `uniform`
// (model_configurator.py:62), and it is not cosmetic: it decides how many
// channels `conv_out` emits and therefore WHICH half of them the mean split
// keeps. See ltx2_video_vae_encoder.h.
Ltx2LogVarianceType ParseLogVariance(const std::string& name, const std::string& where) {
  if (name == "per_channel") return Ltx2LogVarianceType::kPerChannel;
  if (name == "uniform") return Ltx2LogVarianceType::kUniform;
  if (name == "constant") return Ltx2LogVarianceType::kConstant;
  if (name == "none") return Ltx2LogVarianceType::kNone;
  Fail(where + ".latent_log_var is '" + name + "'; it is not one of the four (enums.py:9-13)");
}

// `[["res_x", {"num_layers": 4}], ["compress_all", {"multiplier": 2}], ...]` —
// the same pair form the decoder's list uses, read by `_make_encoder_block`
// (video_vae.py:39-145), which asks for exactly `num_layers` and `multiplier`.
//
// The encoder's list has NO `inject_noise` and NO `residual`: `_make_encoder_block`
// never reads them (contrast `_make_decoder_block`), so they are deliberately
// not parsed here. A checkpoint carrying them on an encoder block would be
// declaring something upstream ignores, and silently honouring it would build a
// module upstream cannot.
std::vector<Ltx2VideoEncoderBlock> ParseEncoderBlocks(const nlohmann::json& blocks,
                                                      const std::string& where) {
  if (!blocks.is_array()) Fail("'" + where + ".encoder_blocks' is not an array");
  std::vector<Ltx2VideoEncoderBlock> out;
  for (const nlohmann::json& entry : blocks) {
    if (!entry.is_array() || entry.size() != 2 || !entry[0].is_string() || !entry[1].is_object()) {
      Fail("an '" + where + ".encoder_blocks' entry is not a [name, {params}] pair: " +
           entry.dump());
    }
    Ltx2VideoEncoderBlock block;
    block.name = entry[0].get<std::string>();
    const nlohmann::json& params = entry[1];
    // `num_layers` is REQUIRED, and required by exactly one block kind. Upstream
    // SUBSCRIPTS it — `num_layers=block_config["num_layers"]` (video_vae.py:55) —
    // which raises `KeyError` when it is absent, and no other branch of
    // `_make_encoder_block` reads it at all (:61-145). This defaulted to 1, so a
    // `res_x` config upstream refuses outright built a silent one-layer
    // `UNetMidBlock3D` here — the same class of wrong-shape-without-a-word defect
    // that `multiplier`'s sentinel two lines below exists to prevent, resolved the
    // opposite way. Made consistent 2026-08-13 (review of #657, row LTX25-IMAGE-COND).
    if (block.name == "res_x" && !params.contains("num_layers")) {
      Fail("a '" + where + ".encoder_blocks' res_x entry carries no 'num_layers': " +
           entry.dump() +
           ". Upstream subscripts it (video_vae.py:55) and raises KeyError, so a checkpoint "
           "without it is one upstream cannot load either; the layer count is not derivable "
           "from anything else in the config");
    }
    block.num_layers = ConfigGet<int64_t>(params, "num_layers", 1, "encoder_block");
    // 0 is the sentinel `Ltx2VideoEncoderBlock` documents for "the upstream
    // default for this block kind" (2 for every `*_x_y` / `*_res`). An ABSENT
    // multiplier must stay 0 rather than become 1, which would quietly halve
    // every widening block's output width.
    block.multiplier = ConfigGet<int64_t>(params, "multiplier", 0, "encoder_block");
    out.push_back(std::move(block));
  }
  if (out.empty()) Fail("'" + where + ".encoder_blocks' is empty; there is no encoder to build");
  return out;
}

}  // namespace

std::vector<Ltx2VaeKeyRule> Ltx2VideoVaeEncoderKeyRules() {
  // `VAE_ENCODER_COMFY_KEYS_FILTER` (model_configurator.py:267-276), TRANSLATED
  // into this port's rule shape — not copied rule for rule, and the difference is
  // worth stating rather than glossing.
  //
  // Upstream's `SDOps` is two passes (loader/sd_ops.py:101-122): an `any()` over
  // its four `with_matching` PREFIXES decides whether the key is admitted at all,
  // and then every `with_replacement` is applied in order as a chained SUBSTRING
  // replace. Upstream declares four matchings and only three replacements,
  // because the bare `per_channel_statistics.` form needs admitting but not
  // rewriting.
  //
  // This port has ONE pass: a first-match-wins prefix loop where the matched
  // rule's replacement is substituted (`Ltx2LoadVaeTensors`). So the fourth rule
  // below is an IDENTITY — it exists to carry upstream's fourth matching, since a
  // key admitted by no rule here is dropped. ORDER MATTERS for the same reason
  // upstream lists its `vae.`-prefixed spellings first: `vae.encoder.conv_in...`
  // does not start with `encoder.`, so a bare-first ordering would drop it.
  //
  // The two shapes agree on every key any shipped checkpoint carries: all four
  // upstream matchings are prefixes, no admitted key contains a second
  // replacement's substring after the first has fired, and no replacement is a
  // substring of another's output. They would diverge on a key needing TWO
  // rewrites in one pass, which no LTX-2 VAE file produces — behaviourally
  // equivalent, therefore, not verbatim.
  return {
      {"vae.encoder.", ""},
      {"vae.per_channel_statistics.", "per_channel_statistics."},
      {"encoder.", ""},
      {"per_channel_statistics.", "per_channel_statistics."},
  };
}

bool Ltx2CheckpointHasVideoEncoder(const std::vector<std::string>& tensor_names) {
  // Only the two ENCODER prefixes count. `per_channel_statistics.` is in the
  // filter but is carried by decoder-only files too, so treating it as evidence
  // of an encoder would report every Comfy-split decoder as encodable and then
  // fail deep inside `Ltx2ConvVideoEncode` on a missing `conv_in.conv.weight`.
  for (const std::string& name : tensor_names) {
    if (StartsWith(name, "vae.encoder.") || StartsWith(name, "encoder.")) return true;
  }
  return false;
}

Ltx2ConvVideoEncoderConfig Ltx2ParseConvVideoEncoderConfig(const nlohmann::json& config) {
  const nlohmann::json& vae = ConfigObject(config, "vae", "config");
  if (vae.empty()) Fail("the video VAE config carries no 'vae' object");

  // `_prepare_video_encoder_kwargs`'s two layouts (model_configurator.py:46-53).
  const bool nested = vae.contains("encoder") && !vae.at("encoder").is_null();
  const nlohmann::json& enc = nested ? ConfigObject(vae, "encoder", "vae") : vae;
  const std::string where = nested ? "vae.encoder" : "vae";

  Ltx2ConvVideoEncoderConfig out;

  // `convolution_dimensions` (:56). Asserted rather than stored, exactly as the
  // decoder's parser does: this port is 3-D only.
  const int64_t dims = ConfigGet<int64_t>(enc, "dims", ConfigGet<int64_t>(vae, "dims", 3, "vae"),
                                          where);
  if (dims != 3) Fail("vae.dims is " + std::to_string(dims) + "; only the 3-D encoder is ported");

  out.in_channels = ConfigGet<int64_t>(enc, "in_channels", 3, where);

  // THE LATENT WIDTH (:48 nested / :52 flat). Never the top-level `out_channels`.
  out.out_channels = nested ? ConfigGet<int64_t>(enc, "out_channels",
                                                 ConfigGet<int64_t>(vae, "latent_channels", 128,
                                                                    "vae"),
                                                 where)
                            : ConfigGet<int64_t>(vae, "latent_channels", 128, "vae");

  // `encoder_blocks`: NESTED reads `encoder.blocks` then `encoder.encoder_blocks`
  // (:49); FLAT reads `vae.encoder_blocks` (:53). Neither layout falls back to
  // the other's object, so neither does this.
  const nlohmann::json* blocks = nullptr;
  if (nested) {
    if (enc.contains("blocks") && !enc.at("blocks").is_null()) {
      blocks = &enc.at("blocks");
    } else if (enc.contains("encoder_blocks") && !enc.at("encoder_blocks").is_null()) {
      blocks = &enc.at("encoder_blocks");
    }
  } else if (vae.contains("encoder_blocks") && !vae.at("encoder_blocks").is_null()) {
    blocks = &vae.at("encoder_blocks");
  }
  if (blocks == nullptr) {
    // Upstream's default is `[]`, and `VideoEncoder([])` is an encoder with no
    // down-blocks at all: it would run, produce a latent at the WRONG scale
    // factors, and every shape downstream would still check out because the
    // pipeline derives its shapes from VIDEO_SCALE_FACTORS rather than from this
    // list. Refused instead.
    Fail(
        "the video VAE config declares no 'encoder_blocks', and upstream's default for it is an "
        "EMPTY list (model_configurator.py:49, 53). An encoder with no down-blocks still runs and "
        "still returns a latent — at scale factors of (1, patch_size, patch_size) rather than the "
        "checkpoint's — so refusing is the only way this is visible. A decoder-only Comfy-split "
        "file is the expected case here and is reported separately by "
        "Ltx2CheckpointHasVideoEncoder");
  }
  out.encoder_blocks = ParseEncoderBlocks(*blocks, where);

  out.patch_size = ConfigGet<int64_t>(enc, "patch_size", 4, where);
  out.norm_layer =
      ParseNormLayer(ConfigGet<std::string>(enc, "norm_layer", "pixel_norm", where), where);
  out.latent_log_var =
      ParseLogVariance(ConfigGet<std::string>(enc, "latent_log_var", "uniform", where), where);

  // `encoder_spatial_padding_mode` (:63-68): the ENCODER key first, then the
  // top-level `encoder_spatial_padding_mode`, then `zeros` — NOT the decoder's
  // `reflect`. All three levels are mirrored because upstream's chained
  // `.get(a, config.get(b, "zeros"))` is what decides which checkpoint spelling
  // wins, and collapsing it to one lookup changes that.
  out.spatial_padding_mode = ParsePaddingMode(
      ConfigGet<std::string>(
          enc, "spatial_padding_mode",
          ConfigGet<std::string>(vae, "encoder_spatial_padding_mode", "zeros", "vae"), where),
      where);

  // norm_num_groups, norm_eps and pixel_norm_eps are NOT checkpoint keys: they
  // are `VideoEncoder._DEFAULT_NORM_NUM_GROUPS` and the literals
  // `_make_encoder_block` passes (video_vae.py:56, 66, 240). They keep the
  // values ltx2_video_vae_encoder.h pins to those lines. Reading them from
  // config would let a file move a constant no golden can see.
  return out;
}

}  // namespace vllm
