// Binding the converted S2Mel checkpoint. See indextts2_s2mel_loader.h.
//
// The synthetic cases build a real .safetensors byte stream at reduced dims, so
// CI runs them with no checkpoint. The last case runs against the SHIPPED
// checkpoint when VLLM_CPP_INDEXTTS2_S2MEL points at the converted file, and is
// skipped otherwise -- skipped LOUDLY, so "the real checkpoint was never
// checked" cannot read as "the real checkpoint passed".
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/indextts2_config.h"
#include "vllm/model_executor/models/indextts2_s2mel_loader.h"

#include "support/process_id.h"
namespace {

std::string U64Le(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
  }
  return out;
}

struct Builder {
  std::string header = "{";
  std::string data;
  bool first = true;

  void Add(const std::string& name, const std::vector<int64_t>& shape,
           const std::string& dtype = "F32") {
    int64_t n = 1;
    for (const int64_t d : shape) {
      n *= d;
    }
    const size_t width = (dtype == "I64") ? 8 : 4;
    const size_t begin = data.size();
    for (int64_t i = 0; i < n; ++i) {
      if (dtype == "I64") {
        const int64_t v = i;
        data.append(reinterpret_cast<const char*>(&v), 8);
      } else {
        const float v = 0.125F * static_cast<float>((i % 7) - 3);  // i is int64_t here
        data.append(reinterpret_cast<const char*>(&v), 4);
      }
    }
    const size_t end = begin + static_cast<size_t>(n) * width;
    if (!first) {
      header += ",";
    }
    first = false;
    header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (size_t i = 0; i < shape.size(); ++i) {
      header += (i ? "," : "") + std::to_string(shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," +
              std::to_string(end) + "]}";
  }

  std::string Finish() { return U64Le(header.size() + 1) + header + "}" + data; }
};

constexpr const char* kP = "net.cfm.estimator.";

// A minimal but COMPLETE S2Mel tail: hidden 8, in_channels 4, wavenet 8 wide,
// 2 layers of kernel 3, frequency width 16.
std::string BuildTail(int64_t hidden = 8, int64_t in_ch = 4, int64_t wn = 8,
                      int64_t layers = 2, int64_t kernel = 3, int64_t freq = 16,
                      const std::string& wrong_dtype_for = {}) {
  Builder b;
  const std::string p = kP;
  // Everything is F32 except, optionally, one named tensor.
  auto dt = [&](const std::string& suffix) {
    return suffix == wrong_dtype_for ? std::string("I32") : std::string("F32");
  };
  b.Add(p + "skip_linear.weight", {hidden, hidden + in_ch});
  b.Add(p + "skip_linear.bias", {hidden}, dt("skip_linear.bias"));
  b.Add(p + "conv1.weight", {wn, hidden});
  b.Add(p + "conv1.bias", {wn});
  b.Add(p + "res_projection.weight", {wn, hidden});
  b.Add(p + "res_projection.bias", {wn});
  b.Add(p + "conv2.weight", {in_ch, wn, 1});
  b.Add(p + "conv2.bias", {in_ch});
  b.Add(p + "t_embedder2.mlp.0.weight", {wn, freq});
  b.Add(p + "t_embedder2.mlp.0.bias", {wn});
  b.Add(p + "t_embedder2.mlp.2.weight", {wn, wn});
  b.Add(p + "t_embedder2.mlp.2.bias", {wn});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_g", {2 * wn * layers, 1, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_v", {2 * wn * layers, wn, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.bias", {2 * wn * layers});
  for (int64_t i = 0; i < layers; ++i) {
    const std::string idx = std::to_string(i);
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_g", {2 * wn, 1, 1});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_v", {2 * wn, wn, kernel});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.bias", {2 * wn});
    const int64_t rs = (i < layers - 1) ? 2 * wn : wn;
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_g", {rs, 1, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_v", {rs, wn, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.bias", {rs});
  }
  b.Add(p + "final_layer.adaLN_modulation.1.weight", {2 * wn, wn});
  b.Add(p + "final_layer.adaLN_modulation.1.bias", {2 * wn});
  b.Add(p + "final_layer.linear.weight_g", {wn, 1});
  b.Add(p + "final_layer.linear.weight_v", {wn, wn});
  b.Add(p + "final_layer.linear.bias", {wn});
  return b.Finish();
}

std::string WriteTemp(const std::string& bytes, const std::string& tag) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("indextts2_s2mel_" + tag + "_" + std::to_string(vllm_test::ProcessId()) + ".safetensors");
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  out.close();
  return path.string();
}

}  // namespace

TEST_CASE("dimensions come from the WEIGHTS, not from a config") {
  const std::string path = WriteTemp(BuildTail(), "dims");
  const auto tail = vllm::models::indextts2::LoadS2MelTail(path);
  CHECK(tail.config.hidden == 8);
  CHECK(tail.config.in_channels == 4);  // derived: 12 - 8
  CHECK(tail.config.wn_hidden == 8);
  CHECK(tail.config.freq_size == 16);
  CHECK(tail.config.wn.layers == 2);
  CHECK(tail.config.wn.kernel == 3);
  CHECK(tail.config.wn.gin == 8);
  CHECK(tail.weights.wn.in_layers.size() == 2);
  CHECK(tail.weights.wn.res_skip_layers.size() == 2);
  // The last res_skip layer is the narrow one.
  CHECK(tail.weights.wn.res_skip_layers[0].g.size() == 16);
  CHECK(tail.weights.wn.res_skip_layers[1].g.size() == 8);
  std::filesystem::remove(path);
}

TEST_CASE("the loaded weights DRIVE the ported tail") {
  const std::string path = WriteTemp(BuildTail(), "drive");
  auto tail = vllm::models::indextts2::LoadS2MelTail(path);
  tail.config.frames = 5;
  const std::vector<float> x_res(static_cast<size_t>(5 * 8), 0.25F);
  const std::vector<float> x(static_cast<size_t>(5 * 4), -0.5F);
  const std::vector<float> t1(8, 0.1F);

  const std::vector<float> out = vllm::models::dit_tail::Forward(
      tail.config, tail.weights, x_res, x, 0.3F, t1, {});
  REQUIRE(out.size() == static_cast<size_t>(4 * 5));
  for (const float v : out) {
    CHECK(std::isfinite(v));
  }
  std::filesystem::remove(path);
}

TEST_CASE("a MISSING tensor is refused BY NAME, not defaulted") {
  // Drop conv2.bias by rebuilding without it: an incomplete checkpoint must not
  // load as if it were whole.
  Builder b;
  const std::string p = kP;
  b.Add(p + "skip_linear.weight", {8, 12});
  b.Add(p + "skip_linear.bias", {8});
  b.Add(p + "conv1.weight", {8, 8});
  b.Add(p + "conv1.bias", {8});
  b.Add(p + "t_embedder2.mlp.0.weight", {8, 16});
  b.Add(p + "wavenet.in_layers.0.conv.conv.weight_v", {16, 8, 3});
  const std::string path = WriteTemp(b.Finish(), "missing");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelTail(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("a tensor read on the DATA path is refused when absent") {
  // The case above happens to fail inside the SHAPE lookup. This one omits a
  // tensor that is only ever read for its VALUES, so it exercises the other
  // throw. Mutation M4 -- returning an empty vector instead of throwing --
  // survived until this existed.
  Builder b;
  const std::string p = kP;
  const std::string full = BuildTail();
  (void)full;
  // Rebuild everything except final_layer.linear.bias.
  const int64_t hidden = 8, in_ch = 4, wn = 8, layers = 2, kernel = 3, freq = 16;
  b.Add(p + "skip_linear.weight", {hidden, hidden + in_ch});
  b.Add(p + "skip_linear.bias", {8});
  b.Add(p + "conv1.weight", {wn, hidden});
  b.Add(p + "conv1.bias", {wn});
  b.Add(p + "res_projection.weight", {wn, hidden});
  b.Add(p + "res_projection.bias", {wn});
  b.Add(p + "conv2.weight", {in_ch, wn, 1});
  b.Add(p + "conv2.bias", {in_ch});
  b.Add(p + "t_embedder2.mlp.0.weight", {wn, freq});
  b.Add(p + "t_embedder2.mlp.0.bias", {wn});
  b.Add(p + "t_embedder2.mlp.2.weight", {wn, wn});
  b.Add(p + "t_embedder2.mlp.2.bias", {wn});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_g", {2 * wn * layers, 1, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_v", {2 * wn * layers, wn, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.bias", {2 * wn * layers});
  for (int64_t i = 0; i < layers; ++i) {
    const std::string idx = std::to_string(i);
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_g", {2 * wn, 1, 1});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_v", {2 * wn, wn, kernel});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.bias", {2 * wn});
    const int64_t rs = (i < layers - 1) ? 2 * wn : wn;
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_g", {rs, 1, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_v", {rs, wn, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.bias", {rs});
  }
  b.Add(p + "final_layer.adaLN_modulation.1.weight", {2 * wn, wn});
  b.Add(p + "final_layer.adaLN_modulation.1.bias", {2 * wn});
  b.Add(p + "final_layer.linear.weight_g", {wn, 1});
  b.Add(p + "final_layer.linear.weight_v", {wn, wn});
  // final_layer.linear.bias DELIBERATELY omitted.
  const std::string path = WriteTemp(b.Finish(), "nobias");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelTail(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("a tensor of the WRONG DTYPE is refused, not reinterpreted") {
  // The checkpoint is COMPLETE except that one tensor is I32. That matters
  // twice over. I32 is FOUR bytes, exactly F32's width, so the byte-length
  // check cannot catch it and only the dtype check can; and the file is
  // otherwise whole, so the load cannot throw for some unrelated reason and
  // let the mutation live. An earlier version of this case got both wrong and
  // passed while the dtype check was deleted.
  const std::string path = WriteTemp(BuildTail(8, 4, 8, 2, 3, 16, "skip_linear.bias"),
                                     "dtype");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelTail(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("a file that is not an S2Mel checkpoint is refused") {
  Builder b;
  b.Add("something.else.weight", {4, 4});
  const std::string path = WriteTemp(b.Finish(), "wrong");
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelTail(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST_CASE("the SHIPPED checkpoint loads with the dimensions we recorded") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_S2MEL to the converted "
            "s2mel.safetensors to check the real checkpoint");
    return;
  }
  const auto tail = vllm::models::indextts2::LoadS2MelTail(std::string(env));
  // From config.yaml AND from the checkpoint header; both agree.
  CHECK(tail.config.hidden == 512);
  CHECK(tail.config.in_channels == 80);
  CHECK(tail.config.wn_hidden == 512);
  CHECK(tail.config.freq_size == 256);
  CHECK(tail.config.wn.layers == 8);
  CHECK(tail.config.wn.kernel == 5);
  CHECK(tail.weights.wn.in_layers.size() == 8);
  CHECK(tail.weights.skip_linear.weight.size() == 512UL * 592UL);
  CHECK(tail.weights.conv2.weight.size() == 80UL * 512UL);
}

TEST_CASE("the SHIPPED weights actually RUN through the ported tail") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_S2MEL, so the real weights were "
            "never run");
    return;
  }
  auto tail = vllm::models::indextts2::LoadS2MelTail(std::string(env));
  const int64_t frames = 8;
  tail.config.frames = frames;

  // Deterministic, non-constant inputs: a constant input can be mapped to a
  // constant output by a broken port and still look finite.
  // NOTE the int64_t cast. `(i % 37) - 18` on a size_t is UNSIGNED arithmetic:
  // every value below 18 wraps to ~1.8e19, and the first version of this case
  // fed the tower 9e16 and then blamed the port for the 1e16 it returned.
  std::vector<float> x_res(static_cast<size_t>(frames * tail.config.hidden));
  for (size_t i = 0; i < x_res.size(); ++i) {
    x_res[i] = 0.01F * static_cast<float>(static_cast<int64_t>(i % 37) - 18);
  }
  std::vector<float> x(static_cast<size_t>(frames * tail.config.in_channels));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.02F * static_cast<float>(static_cast<int64_t>(i % 23) - 11);
  }
  std::vector<float> t1(static_cast<size_t>(tail.config.hidden));
  for (size_t i = 0; i < t1.size(); ++i) {
    t1[i] = 0.005F * static_cast<float>(static_cast<int64_t>(i % 13) - 6);
  }
  // Guard the inputs before trusting anything the tower says about them.
  for (const float v : x_res) {
    REQUIRE(std::fabs(v) < 1.0F);
  }

  const std::vector<float> mel = vllm::models::dit_tail::Forward(
      tail.config, tail.weights, x_res, x, 0.37F, t1, {});

  REQUIRE(mel.size() == static_cast<size_t>(80 * frames));
  double sum = 0.0;
  float lo = mel[0];
  float hi = mel[0];
  for (const float v : mel) {
    REQUIRE(std::isfinite(v));
    sum += static_cast<double>(v);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  // A mel frame from a real 512-wide tower: finite, not all one value, and not
  // absurdly scaled. These are sanity bounds, NOT a correctness gate -- parity
  // needs the vLLM-Omni oracle, still unpinned (#633).
  CHECK(hi > lo);
  CHECK(std::fabs(sum / static_cast<double>(mel.size())) < 1e3);
  MESSAGE("real S2Mel tail ran: 80 x " << frames << " mel, range ["
          << lo << ", " << hi << "]");
}

namespace {

// A COMPLETE small estimator: the tail plus a front end and a one-layer stack,
// at dimensions deliberately unlike the shipped ones so a hardcoded constant
// cannot pass. hidden 8, in_channels 4, STYLE 5, heads 2, intermediate 12.
std::string BuildEstimator(int64_t hidden = 8, int64_t in_ch = 4, int64_t style = 5,
                           int64_t layers = 1, int64_t inter = 12) {
  // Reuse the tail builder, then append the front and stack tensors.
  const std::string tail = BuildTail(hidden, in_ch, hidden, 2, 3, 16);
  // Rebuild from scratch: the two headers cannot simply be concatenated.
  Builder b;
  const std::string p = kP;
  const int64_t wn = hidden;
  b.Add(p + "skip_linear.weight", {hidden, hidden + in_ch});
  b.Add(p + "skip_linear.bias", {hidden});
  b.Add(p + "conv1.weight", {wn, hidden});
  b.Add(p + "conv1.bias", {wn});
  b.Add(p + "res_projection.weight", {wn, hidden});
  b.Add(p + "res_projection.bias", {wn});
  b.Add(p + "conv2.weight", {in_ch, wn, 1});
  b.Add(p + "conv2.bias", {in_ch});
  b.Add(p + "t_embedder2.mlp.0.weight", {wn, 16});
  b.Add(p + "t_embedder2.mlp.0.bias", {wn});
  b.Add(p + "t_embedder2.mlp.2.weight", {wn, wn});
  b.Add(p + "t_embedder2.mlp.2.bias", {wn});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_g", {2 * wn * 2, 1, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.weight_v", {2 * wn * 2, wn, 1});
  b.Add(p + "wavenet.cond_layer.conv.conv.bias", {2 * wn * 2});
  for (int64_t i = 0; i < 2; ++i) {
    const std::string idx = std::to_string(i);
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_g", {2 * wn, 1, 1});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.weight_v", {2 * wn, wn, 3});
    b.Add(p + "wavenet.in_layers." + idx + ".conv.conv.bias", {2 * wn});
    const int64_t rs = (i < 1) ? 2 * wn : wn;
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_g", {rs, 1, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.weight_v", {rs, wn, 1});
    b.Add(p + "wavenet.res_skip_layers." + idx + ".conv.conv.bias", {rs});
  }
  b.Add(p + "final_layer.adaLN_modulation.1.weight", {2 * wn, wn});
  b.Add(p + "final_layer.adaLN_modulation.1.bias", {2 * wn});
  b.Add(p + "final_layer.linear.weight_g", {wn, 1});
  b.Add(p + "final_layer.linear.weight_v", {wn, wn});
  b.Add(p + "final_layer.linear.bias", {wn});
  // Front end. `wide` = hidden + 2 * in_channels + style.
  b.Add(p + "cond_projection.weight", {hidden, hidden});
  b.Add(p + "cond_projection.bias", {hidden});
  b.Add(p + "cond_x_merge_linear.weight", {hidden, hidden + 2 * in_ch + style});
  b.Add(p + "cond_x_merge_linear.bias", {hidden});
  // Stack.
  for (int64_t i = 0; i < layers; ++i) {
    const std::string lp = p + "transformer.layers." + std::to_string(i) + ".";
    b.Add(lp + "attention.wqkv.weight", {3 * hidden, hidden});
    b.Add(lp + "attention.wo.weight", {hidden, hidden});
    b.Add(lp + "feed_forward.w1.weight", {inter, hidden});
    b.Add(lp + "feed_forward.w2.weight", {hidden, inter});
    b.Add(lp + "feed_forward.w3.weight", {inter, hidden});
    b.Add(lp + "attention_norm.project_layer.weight", {2 * hidden, hidden});
    b.Add(lp + "attention_norm.project_layer.bias", {2 * hidden});
    b.Add(lp + "attention_norm.norm.weight", {hidden});
    b.Add(lp + "ffn_norm.project_layer.weight", {2 * hidden, hidden});
    b.Add(lp + "ffn_norm.project_layer.bias", {2 * hidden});
    b.Add(lp + "ffn_norm.norm.weight", {hidden});
    b.Add(lp + "skip_in_linear.weight", {hidden, 2 * hidden});
    b.Add(lp + "skip_in_linear.bias", {hidden});
  }
  b.Add(p + "transformer.norm.project_layer.weight", {2 * hidden, hidden});
  b.Add(p + "transformer.norm.project_layer.bias", {2 * hidden});
  b.Add(p + "transformer.norm.norm.weight", {hidden});
  (void)tail;
  return b.Finish();
}

}  // namespace

TEST_CASE("the style width comes from cond_x_merge_linear, not a constant") {
  // Style 5, unlike the shipped 192, so a hardcoded value cannot pass.
  const std::string path = WriteTemp(BuildEstimator(), "style");
  const auto est = vllm::models::indextts2::LoadS2MelEstimator(path, 2);
  CHECK(est.front_config.style == 5);
  CHECK(est.front_config.hidden == 8);
  CHECK(est.front_config.in_channels == 4);
  CHECK(est.stack_config.intermediate == 12);
  CHECK(est.stack.layers.size() == 1);
  std::filesystem::remove(path);
}

TEST_CASE("a head count that does not divide the hidden width is REFUSED") {
  const std::string path = WriteTemp(BuildEstimator(), "heads");  // hidden 8
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelEstimator(path, 3),
                  std::runtime_error);
  CHECK_THROWS_AS(vllm::models::indextts2::LoadS2MelEstimator(path, 0),
                  std::runtime_error);
  CHECK_NOTHROW(vllm::models::indextts2::LoadS2MelEstimator(path, 4));
  std::filesystem::remove(path);
}

TEST_CASE("the WHOLE S2Mel estimator loads at its declared geometry") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_S2MEL, so the estimator was never loaded");
    return;
  }
  const auto est = vllm::models::indextts2::LoadS2MelEstimator(
      std::string(env), vllm::models::indextts2::kDitNumHeads);
  // Every one of these is in config.yaml; every one here came from the WEIGHTS.
  CHECK(est.stack_config.dim == vllm::models::indextts2::kDitHiddenDim);
  CHECK(est.front_config.in_channels == vllm::models::indextts2::kDitInChannels);
  CHECK(est.front_config.style == vllm::models::indextts2::kStyleDim);
  CHECK(static_cast<int64_t>(est.stack.layers.size()) ==
        vllm::models::indextts2::kDitDepth);
  CHECK(est.stack_config.heads == vllm::models::indextts2::kDitNumHeads);
  CHECK(est.stack_config.head_dim == 64);
  CHECK(est.stack_config.intermediate == 1536);
  // Every layer carries a skip_in_linear upstream, even the ones that never
  // receive one.
  for (const auto& l : est.stack.layers) {
    CHECK(!l.skip_in_w.empty());
  }
}

TEST_CASE("the real estimator turns conditioning into a MEL") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_S2MEL, so no mel was produced");
    return;
  }
  auto est = vllm::models::indextts2::LoadS2MelEstimator(
      std::string(env), vllm::models::indextts2::kDitNumHeads);
  const int64_t frames = 8;
  est.front_config.frames = frames;
  est.stack_config.frames = frames;
  est.tail_config.frames = frames;

  auto ramp = [](size_t n, float amp) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = amp * std::sin(0.03F * static_cast<float>(i));
    }
    return v;
  };
  const int64_t D = est.stack_config.dim;
  const int64_t C = est.front_config.in_channels;
  const auto x = ramp(static_cast<size_t>(C * frames), 0.4F);
  const auto t1 = ramp(static_cast<size_t>(D), 0.02F);

  const auto x_in = vllm::models::dit_front::BuildXIn(
      est.front_config, est.front, x, ramp(static_cast<size_t>(C * frames), 0.2F),
      ramp(static_cast<size_t>(frames * D), 0.1F),
      ramp(static_cast<size_t>(est.front_config.style), 0.05F), false);
  REQUIRE(x_in.size() == static_cast<size_t>(frames * D));

  std::vector<float> freqs(static_cast<size_t>(frames * est.stack_config.head_dim), 0.0F);
  const auto x_res = vllm::models::dit_stack::Forward(est.stack_config, est.stack,
                                                      x_in, t1, freqs);
  const auto mel = vllm::models::dit_tail::Forward(est.tail_config, est.tail, x_res, x,
                                                   0.37F, t1, {});
  REQUIRE(mel.size() == static_cast<size_t>(C * frames));
  float lo = mel[0];
  float hi = mel[0];
  for (const float v : mel) {
    REQUIRE(std::isfinite(v));
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  CHECK(hi > lo);
  // A log-mel, so plausibly negative and not enormous. Sanity bounds, NOT a
  // correctness gate: parity needs the vLLM-Omni oracle (#633).
  CHECK(std::fabs(static_cast<double>(lo)) < 1e3);
  MESSAGE("real S2Mel estimator produced " << mel.size() << " mel values, range ["
          << lo << ", " << hi << "]");
}
