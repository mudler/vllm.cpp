// FactorizedVectorQuantize. See fvq.h for the upstream anchors.
#include "vllm/model_executor/models/fvq.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "vllm/model_executor/models/vocoder1d.h"
#include "vt/dtype.h"

namespace vllm {
namespace models {
namespace fvq {

namespace {

// 1x1 convolution over [C_in, T] -> [C_out, T].
std::vector<float> Pointwise(const std::vector<float>& x, int64_t in_dim, int64_t frames,
                             int64_t out_dim, const std::vector<float>& w,
                             const std::vector<float>& b) {
  std::vector<float> out(static_cast<size_t>(out_dim * frames));
  for (int64_t o = 0; o < out_dim; ++o) {
    for (int64_t t = 0; t < frames; ++t) {
      double acc = b.empty() ? 0.0 : static_cast<double>(b[static_cast<size_t>(o)]);
      for (int64_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(w[static_cast<size_t>(o * in_dim + i)]) *
               static_cast<double>(x[static_cast<size_t>(i * frames + t)]);
      }
      out[static_cast<size_t>(o * frames + t)] = static_cast<float>(acc);
    }
  }
  return out;
}

}  // namespace

QuantizeResult Quantize(const std::vector<float>& z, int64_t frames, int64_t input_dim,
                        int64_t codebook_dim, int64_t codebook_size, const Weights& wts) {
  VT_CHECK(z.size() == static_cast<size_t>(input_dim * frames), "fvq: z shape");

  const std::vector<float> in_w = vocoder1d::MaterializeWeightNorm(wts.in_g, wts.in_v, codebook_dim);
  const std::vector<float> z_e = Pointwise(z, input_dim, frames, codebook_dim, in_w, wts.in_bias);

  // Codebook rows are normalized ONCE, for the search only.
  std::vector<double> cb_norm(static_cast<size_t>(codebook_size * codebook_dim));
  for (int64_t c = 0; c < codebook_size; ++c) {
    double n = 0.0;
    for (int64_t d = 0; d < codebook_dim; ++d) {
      const double x = static_cast<double>(wts.codebook[static_cast<size_t>(c * codebook_dim + d)]);
      n += x * x;
    }
    n = std::sqrt(n);
    // torch's F.normalize guards with eps=1e-12 rather than dividing by zero.
    const double inv = 1.0 / std::max(n, 1e-12);
    for (int64_t d = 0; d < codebook_dim; ++d) {
      cb_norm[static_cast<size_t>(c * codebook_dim + d)] =
          static_cast<double>(wts.codebook[static_cast<size_t>(c * codebook_dim + d)]) * inv;
    }
  }

  QuantizeResult r;
  r.indices.resize(static_cast<size_t>(frames));
  std::vector<float> picked(static_cast<size_t>(codebook_dim * frames));
  for (int64_t t = 0; t < frames; ++t) {
    std::vector<double> e(static_cast<size_t>(codebook_dim));
    double n = 0.0;
    for (int64_t d = 0; d < codebook_dim; ++d) {
      e[static_cast<size_t>(d)] = static_cast<double>(z_e[static_cast<size_t>(d * frames + t)]);
      n += e[static_cast<size_t>(d)] * e[static_cast<size_t>(d)];
    }
    // Kept for fidelity to upstream, though it is provably a NO-OP for the
    // SEARCH: dist = |e|^2 - 2 e.c + |c|^2, and for a fixed frame |e|^2 is
    // constant across candidates while a normalized codebook makes |c|^2 = 1,
    // so argmin reduces to argmax(e.c) -- which scaling e by 1/||e|| cannot
    // change. Mutating this line away therefore CANNOT be caught, and that is
    // an algebraic identity rather than a gap in the gate.
    const double inv = 1.0 / std::max(std::sqrt(n), 1e-12);
    for (double& v : e) v *= inv;

    int64_t best = 0;
    double best_dist = std::numeric_limits<double>::infinity();
    for (int64_t c = 0; c < codebook_size; ++c) {
      double dist = 0.0;
      for (int64_t d = 0; d < codebook_dim; ++d) {
        const double diff = e[static_cast<size_t>(d)] -
                            cb_norm[static_cast<size_t>(c * codebook_dim + d)];
        dist += diff * diff;
      }
      if (dist < best_dist) { best_dist = dist; best = c; }
    }
    r.indices[static_cast<size_t>(t)] = best;
    // decode_code returns the RAW row, not the normalized one used for search.
    for (int64_t d = 0; d < codebook_dim; ++d) {
      picked[static_cast<size_t>(d * frames + t)] =
          wts.codebook[static_cast<size_t>(best * codebook_dim + d)];
    }
  }

  const std::vector<float> out_w = vocoder1d::MaterializeWeightNorm(wts.out_g, wts.out_v, input_dim);
  r.z_q = Pointwise(picked, codebook_dim, frames, input_dim, out_w, wts.out_bias);
  return r;
}

}  // namespace fvq
}  // namespace models
}  // namespace vllm
