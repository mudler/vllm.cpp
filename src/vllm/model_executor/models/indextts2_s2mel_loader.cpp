// S2Mel checkpoint binding. See indextts2_s2mel_loader.h for the anchors.
#include "vllm/model_executor/models/indextts2_s2mel_loader.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace indextts2 {
namespace {

constexpr const char* kPrefix = "net.cfm.estimator.";

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("IndexTTS-2.5 s2mel: " + what);
}

int64_t Elems(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) {
    n *= d;
  }
  return n;
}

// Every S2Mel tensor is F32 in the shipped checkpoint. Refuse anything else by
// name rather than reinterpreting bytes: a silently misread dtype is finite,
// plausible and wrong.
std::vector<float> Read(const SafetensorsFile& file, const std::string& suffix) {
  const std::string name = kPrefix + suffix;
  const StTensor* t = nullptr;
  try {
    t = &file.Get(name);
  } catch (const std::exception&) {
    Fail("missing tensor '" + name + "'");
  }
  if (t->dtype != "F32") {
    Fail("tensor '" + name + "' is " + t->dtype + ", expected F32");
  }
  const int64_t n = Elems(t->shape);
  if (t->nbytes != static_cast<size_t>(n) * sizeof(float)) {
    Fail("tensor '" + name + "' byte length disagrees with its shape");
  }
  std::vector<float> out(static_cast<size_t>(n));
  std::memcpy(out.data(), t->data, t->nbytes);
  return out;
}

// Returned BY VALUE: a shape is a handful of ints, and returning a reference
// out of a try/catch is what -Werror=dangling-reference exists to stop.
std::vector<int64_t> Shape(const SafetensorsFile& file, const std::string& suffix) {
  const std::string name = kPrefix + suffix;
  try {
    return file.Get(name).shape;
  } catch (const std::exception&) {
    Fail("missing tensor '" + name + "'");
  }
}

dit_tail::Linear ReadLinear(const SafetensorsFile& file, const std::string& suffix) {
  dit_tail::Linear l;
  l.weight = Read(file, suffix + ".weight");
  l.bias = Read(file, suffix + ".bias");
  return l;
}

wavenet::ConvWeights ReadWeightNormConv(const SafetensorsFile& file,
                                        const std::string& suffix) {
  wavenet::ConvWeights c;
  // torch's legacy weight_norm stores g with the reduced dims kept, so the
  // shipped `weight_g` is [out, 1, 1] rather than [out]. Flattening is correct
  // precisely because every trailing dim is 1; assert that instead of assuming.
  const std::vector<int64_t> gs = Shape(file, suffix + ".weight_g");
  for (size_t i = 1; i < gs.size(); ++i) {
    if (gs[i] != 1) {
      Fail("weight_g for '" + suffix + "' is not one magnitude per output channel");
    }
  }
  c.g = Read(file, suffix + ".weight_g");
  c.v = Read(file, suffix + ".weight_v");
  c.bias = Read(file, suffix + ".bias");
  return c;
}

int64_t CountWavenetLayers(const SafetensorsFile& file) {
  int64_t n = 0;
  const std::string stem = std::string(kPrefix) + "wavenet.in_layers.";
  for (const std::string& name : file.Names()) {
    if (name.rfind(stem, 0) == 0 && name.size() > stem.size() &&
        name.find(".conv.conv.weight_v") != std::string::npos) {
      ++n;
    }
  }
  return n;
}

}  // namespace

S2MelTail LoadS2MelTail(const SafetensorsFile& file) {
  S2MelTail out;

  // Resolve the dimensions from the weights themselves.
  const std::vector<int64_t> skip_w = Shape(file, "skip_linear.weight");
  if (skip_w.size() != 2) {
    Fail("skip_linear.weight must be 2-D");
  }
  const int64_t hidden = skip_w[0];
  const int64_t in_channels = skip_w[1] - hidden;
  if (in_channels <= 0) {
    Fail("skip_linear.weight is not [hidden, hidden + in_channels]");
  }

  const std::vector<int64_t> conv1_w = Shape(file, "conv1.weight");
  if (conv1_w.size() != 2) {
    Fail("conv1.weight must be 2-D");
  }
  const int64_t wn_hidden = conv1_w[0];

  const std::vector<int64_t> te2 = Shape(file, "t_embedder2.mlp.0.weight");
  if (te2.size() != 2) {
    Fail("t_embedder2.mlp.0.weight must be 2-D");
  }
  const int64_t freq_size = te2[1];

  const int64_t layers = CountWavenetLayers(file);
  if (layers <= 0) {
    Fail("no wavenet in_layers found; this is not an S2Mel checkpoint");
  }

  // in_layers.0 is [2 * wn_hidden, wn_hidden, kernel].
  const std::vector<int64_t> in0 = Shape(file, "wavenet.in_layers.0.conv.conv.weight_v");
  if (in0.size() != 3) {
    Fail("wavenet in_layer weight_v must be 3-D");
  }
  const int64_t kernel = in0[2];

  // dilation_rate is not stored. Upstream ships 1, and the shapes cannot
  // distinguish it, so it is recorded as an ASSUMPTION here rather than
  // presented as something the checkpoint proved.
  out.config.hidden = hidden;
  out.config.wn_hidden = wn_hidden;
  out.config.in_channels = in_channels;
  out.config.freq_size = freq_size;
  out.config.frames = 0;  // per call
  out.config.wn.hidden = wn_hidden;
  out.config.wn.kernel = kernel;
  out.config.wn.dilation_rate = 1;
  out.config.wn.layers = layers;
  out.config.wn.gin = wn_hidden;

  out.weights.skip_linear = ReadLinear(file, "skip_linear");
  out.weights.conv1 = ReadLinear(file, "conv1");
  out.weights.res_projection = ReadLinear(file, "res_projection");
  out.weights.conv2 = ReadLinear(file, "conv2");  // [in_channels, wn_hidden, 1]
  out.weights.t_embedder2_mlp0 = ReadLinear(file, "t_embedder2.mlp.0");
  out.weights.t_embedder2_mlp2 = ReadLinear(file, "t_embedder2.mlp.2");

  out.weights.wn.cond = ReadWeightNormConv(file, "wavenet.cond_layer.conv.conv");
  for (int64_t i = 0; i < layers; ++i) {
    const std::string idx = std::to_string(i);
    out.weights.wn.in_layers.push_back(
        ReadWeightNormConv(file, "wavenet.in_layers." + idx + ".conv.conv"));
    out.weights.wn.res_skip_layers.push_back(
        ReadWeightNormConv(file, "wavenet.res_skip_layers." + idx + ".conv.conv"));
  }

  out.weights.final_layer.ada_w = Read(file, "final_layer.adaLN_modulation.1.weight");
  out.weights.final_layer.ada_b = Read(file, "final_layer.adaLN_modulation.1.bias");
  out.weights.final_layer.linear_g = Read(file, "final_layer.linear.weight_g");
  out.weights.final_layer.linear_v = Read(file, "final_layer.linear.weight_v");
  out.weights.final_layer.linear_bias = Read(file, "final_layer.linear.bias");

  // The coupling the shipped config hides; dit_tail refuses it too, but failing
  // at LOAD names the checkpoint rather than the call.
  if (wn_hidden != hidden) {
    Fail("this checkpoint has a wavenet width (" + std::to_string(wn_hidden) +
         ") different from its DiT hidden width (" + std::to_string(hidden) +
         "), which the wavenet final layer cannot compose");
  }
  return out;
}

S2MelTail LoadS2MelTail(const std::string& path) {
  const SafetensorsFile file = SafetensorsFile::Open(path);
  return LoadS2MelTail(file);
}

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
