// GLM-5.3-Flash — the env-gated forward diagnostic. See `glm5_next_diag.h` for
// what it is for and for the two properties it has to have: it says that it
// ran, and it does not change what it measures.
#include "vllm/model_executor/models/glm5_next_diag.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace vllm::glm5_next::diag {
namespace {

int ParseLevel() {
  const char* e = std::getenv("VT_GLM5_DIAG");
  if (e == nullptr || e[0] == '\0') return 0;
  const int v = std::atoi(e);
  return v > 0 ? v : 0;
}

const char* RawEnv() {
  const char* e = std::getenv("VT_GLM5_DIAG");
  return (e == nullptr) ? "(unset)" : e;
}

}  // namespace

int Level() {
  static const int level = ParseLevel();
  return level;
}

void Banner(const char* where) {
  if (Level() <= 0) return;
  static bool done = false;
  if (done) return;
  done = true;
  std::fprintf(stderr,
               "[glm5-diag] INSTRUMENT ACTIVE at %s: VT_GLM5_DIAG=\"%s\" "
               "level=%d. Level 1 traces the step, the embedding, one line per "
               "decoder layer, the final hidden state and the logits; level 2 "
               "adds the sublayer trace. With the variable unset every probe is "
               "a no-op and the numerics are byte-identical.\n",
               where, RawEnv(), Level());
  std::fflush(stderr);
}

void Stats(const char* what, const float* v, size_t n) {
  if (Level() <= 0) return;
  if (v == nullptr || n == 0) {
    std::fprintf(stderr, "[glm5-diag] %-38s EMPTY (n=%zu)\n", what, n);
    std::fflush(stderr);
    return;
  }
  size_t nan = 0;
  size_t inf = 0;
  size_t zero = 0;
  double lo = 0.0;
  double hi = 0.0;
  bool seen = false;
  double sum = 0.0;
  double sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const float x = v[i];
    if (std::isnan(x)) {
      ++nan;
      continue;
    }
    if (std::isinf(x)) {
      ++inf;
      continue;
    }
    if (x == 0.0F) ++zero;
    const double d = static_cast<double>(x);
    if (!seen) {
      lo = d;
      hi = d;
      seen = true;
    } else {
      lo = std::min(lo, d);
      hi = std::max(hi, d);
    }
    sum += d;
    sq += d * d;
  }
  const size_t finite = n - nan - inf;
  const double mean = finite > 0 ? sum / static_cast<double>(finite) : 0.0;
  const double var =
      finite > 0 ? std::max(0.0, sq / static_cast<double>(finite) - mean * mean)
                 : 0.0;
  std::fprintf(stderr,
               "[glm5-diag] %-38s n=%zu nan=%zu inf=%zu zero=%zu min=%.6g "
               "max=%.6g mean=%.6g sd=%.6g l2=%.6g v[0..3]=%.6g,%.6g,%.6g,%.6g\n",
               what, n, nan, inf, zero, lo, hi, mean, std::sqrt(var),
               std::sqrt(sq), static_cast<double>(v[0]),
               n > 1 ? static_cast<double>(v[1]) : 0.0,
               n > 2 ? static_cast<double>(v[2]) : 0.0,
               n > 3 ? static_cast<double>(v[3]) : 0.0);
  std::fflush(stderr);
}

void Stats(const char* what, const std::vector<float>& v) {
  Stats(what, v.data(), v.size());
}

void TopK(const char* what, const float* v, size_t n, int k) {
  if (Level() <= 0) return;
  Stats(what, v, n);
  if (v == nullptr || n == 0 || k <= 0) return;
  const size_t want = std::min<size_t>(static_cast<size_t>(k), n);
  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), size_t{0});
  // A PARTIAL sort with a total order: NaN never compares greater, so a
  // NaN-filled buffer sorts to index order and its top-k reads 0, 1, 2, ...,
  // which is exactly the signature the argmax produced. That is deliberate --
  // the top-k has to reproduce the sampler's own tie behaviour to be evidence
  // about it.
  std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(want),
                    idx.end(), [v](size_t a, size_t b) {
                      const float xa = v[a];
                      const float xb = v[b];
                      if (xa == xb) return a < b;
                      return xa > xb;
                    });
  std::string line = "[glm5-diag] " + std::string(what) + " top" +
                     std::to_string(want) + ":";
  char buf[96];
  for (size_t i = 0; i < want; ++i) {
    std::snprintf(buf, sizeof(buf), " (%zu, %.8g)", idx[i],
                  static_cast<double>(v[idx[i]]));
    line += buf;
  }
  // The MARGIN, printed beside the ids, because a discrete selection has
  // bimodal error: a top-1 that wins by 0 is a tie the argmax broke by index
  // and is a different fact from a top-1 that wins by 3.
  if (want >= 2) {
    std::snprintf(buf, sizeof(buf), "  margin(top1-top2)=%.8g",
                  static_cast<double>(v[idx[0]]) - static_cast<double>(v[idx[1]]));
    line += buf;
  }
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

void Ids(const char* what, const std::vector<int32_t>& ids) {
  if (Level() <= 0) return;
  std::string line = "[glm5-diag] " + std::string(what) + " n=" +
                     std::to_string(ids.size()) + ":";
  for (size_t i = 0; i < ids.size() && i < 64; ++i) {
    line += " " + std::to_string(ids[i]);
  }
  if (ids.size() > 64) line += " ...";
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

}  // namespace vllm::glm5_next::diag
