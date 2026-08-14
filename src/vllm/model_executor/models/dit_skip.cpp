// DiT U-Net skip connections. See dit_skip.h for the upstream anchors.
#include "vllm/model_executor/models/dit_skip.h"

#include <cstddef>
#include <vector>

#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace dit_skip {

Schedule Plan(int64_t layers) {
  VT_CHECK(layers > 0, "dit_skip: a transformer needs at least one layer");
  Schedule s;
  s.source.assign(static_cast<size_t>(layers), -1);

  // model.py:154-155. Note the ASYMMETRY: emitters use `<` and receivers use
  // `>`, both against `layers / 2` with integer division, so at odd depth the
  // middle layer is on neither list and the two halves balance, while at even
  // depth there is one more emitter than receiver.
  const int64_t half = layers / 2;
  for (int64_t i = 0; i < layers; ++i) {
    if (i < half) {
      s.emit.push_back(i);
    }
    if (i > half) {
      s.receive.push_back(i);
    }
  }

  // model.py:181-190: emitters append AFTER their layer runs, receivers pop the
  // most recent one, so the pairing is LIFO.
  std::vector<int64_t> stack;
  for (int64_t i = 0; i < layers; ++i) {
    // Receive happens BEFORE this layer runs...
    if (i > half) {
      VT_CHECK(!stack.empty(),
               "dit_skip: a receiving layer found the skip stack empty");
      s.source[static_cast<size_t>(i)] = stack.back();
      stack.pop_back();
    }
    // ...and emit happens AFTER, pushing this layer's own output.
    if (i < half) {
      stack.push_back(i);
    }
  }
  s.orphaned = static_cast<int64_t>(stack.size());
  return s;
}

std::vector<float> ApplySkip(const std::vector<float>& x, const std::vector<float>& skip,
                             int64_t frames, int64_t dim,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias) {
  VT_CHECK(frames > 0 && dim > 0, "dit_skip: frames and dim must be positive");
  VT_CHECK(x.size() == static_cast<size_t>(frames * dim) &&
               skip.size() == static_cast<size_t>(frames * dim),
           "dit_skip: x and skip must both be [frames, dim]");
  VT_CHECK(weight.size() == static_cast<size_t>(dim * 2 * dim),
           "dit_skip: skip_in_linear weight must be [dim, 2 * dim]");
  VT_CHECK(bias.empty() || bias.size() == static_cast<size_t>(dim),
           "dit_skip: skip_in_linear bias must be [dim]");

  std::vector<float> out(static_cast<size_t>(frames * dim));
  for (int64_t f = 0; f < frames; ++f) {
    for (int64_t o = 0; o < dim; ++o) {
      double acc = bias.empty() ? 0.0 : static_cast<double>(bias[static_cast<size_t>(o)]);
      // cat([x, skip], -1): x occupies columns [0, dim), skip [dim, 2 * dim).
      for (int64_t i = 0; i < dim; ++i) {
        acc += static_cast<double>(x[static_cast<size_t>(f * dim + i)]) *
               static_cast<double>(weight[static_cast<size_t>(o * 2 * dim + i)]);
      }
      for (int64_t i = 0; i < dim; ++i) {
        acc += static_cast<double>(skip[static_cast<size_t>(f * dim + i)]) *
               static_cast<double>(weight[static_cast<size_t>(o * 2 * dim + dim + i)]);
      }
      out[static_cast<size_t>(f * dim + o)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace dit_skip
}  // namespace models
}  // namespace vllm
