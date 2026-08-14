// BigVGAN checkpoint binding. See bigvgan_loader.h for the anchors.
#include "vllm/model_executor/models/bigvgan_loader.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/vocoder1d.h"

namespace vllm {
namespace models {
namespace bigvgan {
namespace {

[[noreturn]] void Fail(const std::string& what) {
  throw std::runtime_error("BigVGAN: " + what);
}

// Returned BY VALUE: a shape is a handful of ints, and returning a reference
// out of a try/catch is what -Werror=dangling-reference exists to stop.
std::vector<int64_t> ShapeOf(const SafetensorsFile& file, const std::string& name) {
  try {
    return file.Get(name).shape;
  } catch (const std::exception&) {
    Fail("missing tensor '" + name + "'");
  }
}

std::vector<float> Read(const SafetensorsFile& file, const std::string& name) {
  const StTensor* t = nullptr;
  try {
    t = &file.Get(name);
  } catch (const std::exception&) {
    Fail("missing tensor '" + name + "'");
  }
  if (t->dtype != "F32") {
    Fail("tensor '" + name + "' is " + t->dtype + ", expected F32");
  }
  std::vector<float> out(t->nbytes / sizeof(float));
  std::memcpy(out.data(), t->data, t->nbytes);
  return out;
}

bool Has(const SafetensorsFile& file, const std::string& name) {
  try {
    (void)file.Get(name);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// A weight-normed conv, folded at load. `dim0` is dimension 0 of `weight_v`,
// which for a ConvTranspose1d is the INPUT channel -- see the note on
// vocoder1d::MaterializeWeightNorm.
ConvSpec FoldedConv(const SafetensorsFile& file, const std::string& prefix) {
  const std::vector<int64_t> v = ShapeOf(file, prefix + ".weight_v");
  if (v.empty()) {
    Fail("'" + prefix + ".weight_v' has no shape");
  }
  ConvSpec c;
  c.weight = vocoder1d::MaterializeWeightNorm(Read(file, prefix + ".weight_g"),
                                              Read(file, prefix + ".weight_v"), v[0]);
  if (Has(file, prefix + ".bias")) {
    c.bias = Read(file, prefix + ".bias");
  }
  return c;
}

int64_t CountPrefixed(const SafetensorsFile& file, const std::string& stem,
                      const std::string& leaf) {
  int64_t n = 0;
  for (const std::string& name : file.Names()) {
    if (name.rfind(stem, 0) == 0 && name.size() > stem.size() &&
        name.find(leaf) != std::string::npos) {
      ++n;
    }
  }
  return n;
}

}  // namespace

Loaded Load(const SafetensorsFile& file) {
  Loaded out;

  const std::vector<int64_t> pre = ShapeOf(file, "conv_pre.weight_v");
  if (pre.size() != 3) {
    Fail("conv_pre.weight_v must be [channels, mels, kernel]");
  }
  out.config.init_channels = pre[0];
  out.config.mels = pre[1];
  out.config.snake_logscale = true;
  out.config.tanh_at_final = false;  // the shipped config says so

  const int64_t stages = CountPrefixed(file, "ups.", ".0.weight_v");
  if (stages <= 0) {
    Fail("no upsample stages found; this is not a BigVGAN generator");
  }
  const int64_t resblocks = CountPrefixed(file, "resblocks.", ".convs1.0.weight_v");
  if (resblocks <= 0 || resblocks % stages != 0) {
    Fail("resblock count is not a whole multiple of the upsample stages");
  }
  out.config.num_kernels = resblocks / stages;

  out.weights.conv_pre = FoldedConv(file, "conv_pre");
  for (int64_t i = 0; i < stages; ++i) {
    const std::string p = "ups." + std::to_string(i) + ".0";
    const std::vector<int64_t> w = ShapeOf(file, p + ".weight_v");
    if (w.size() != 3) {
      Fail("'" + p + ".weight_v' must be 3-D");
    }
    out.config.up_kernels.push_back(w[2]);
    // ConvTranspose1d(ch, ch/2, k, stride=rate, padding=(k-rate)/2): upstream
    // sets k = 2 * rate for every stage of this generator, so the rate is
    // recoverable from the kernel. Asserted rather than assumed.
    if (w[2] % 2 != 0) {
      Fail("'" + p + "' has an odd kernel; the stride cannot be recovered");
    }
    out.config.up_rates.push_back(w[2] / 2);
    out.weights.ups.push_back(FoldedConv(file, p));
  }

  for (int64_t r = 0; r < resblocks; ++r) {
    const std::string p = "resblocks." + std::to_string(r) + ".";
    AmpBlock b;
    const std::vector<int64_t> c1 = ShapeOf(file, p + "convs1.0.weight_v");
    if (c1.size() != 3) {
      Fail("'" + p + "convs1.0.weight_v' must be 3-D");
    }
    b.kernel = c1[2];
    const int64_t pairs = CountPrefixed(file, p + "convs1.", ".weight_v");
    for (int64_t d = 0; d < pairs; ++d) {
      const std::string ds = std::to_string(d);
      b.convs1.push_back(FoldedConv(file, p + "convs1." + ds));
      b.convs2.push_back(FoldedConv(file, p + "convs2." + ds));
      // AMPBlock1 dilations are 1, 3, 5 for every kernel in this generator.
      b.dilations.push_back(d == 0 ? 1 : (d == 1 ? 3 : 5));
    }
    for (int64_t a = 0; a < 2 * pairs; ++a) {
      const std::string as = std::to_string(a);
      b.alpha.push_back(Read(file, p + "activations." + as + ".act.alpha"));
      b.beta.push_back(Read(file, p + "activations." + as + ".act.beta"));
    }
    out.weights.resblocks.push_back(std::move(b));
  }

  out.weights.post_alpha = Read(file, "activation_post.act.alpha");
  out.weights.post_beta = Read(file, "activation_post.act.beta");
  out.weights.conv_post = FoldedConv(file, "conv_post");
  return out;
}

Loaded Load(const std::string& path) {
  const SafetensorsFile file = SafetensorsFile::Open(path);
  return Load(file);
}

}  // namespace bigvgan
}  // namespace models
}  // namespace vllm
