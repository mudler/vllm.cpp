// #838: widen Gemma-4 FP8 indexed MoE from T==1 to T<=63.
// Host-injectable dispatch, tensor oracle, single-scale fallback, scratch retire.
// No HIP.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace vllm {

constexpr int64_t kGemma4PrefillBatchMinT = 64;
constexpr int64_t kGemma4DecodeIndexedMaxTDefault = 63;
constexpr int64_t kGemma4DecodeIndexedMaxTLo = 1;
constexpr int64_t kGemma4DecodeIndexedMaxTHi = 63;
constexpr float kGemma4IndexedOraclePow2 = 7.0f;  // abs_tol = 2^-7 * max_abs(ref)

inline int64_t ParseGemma4DecodeIndexedMaxT(const char* e) {
  if (e == nullptr || e[0] == '\0') return kGemma4DecodeIndexedMaxTDefault;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (end == e) return kGemma4DecodeIndexedMaxTDefault;
  if (v < kGemma4DecodeIndexedMaxTLo) return kGemma4DecodeIndexedMaxTLo;
  if (v > kGemma4DecodeIndexedMaxTHi) return kGemma4DecodeIndexedMaxTHi;
  return static_cast<int64_t>(v);
}

inline int64_t Gemma4DecodeIndexedMaxT() {
  static const int64_t n = ParseGemma4DecodeIndexedMaxT(std::getenv("VT_GEMMA4_DECODE_INDEXED_MAX_T"));
  return n;
}

inline bool Gemma4IndexedOkT(int64_t T, int64_t indexed_max_t, int top_k, bool fp8_res) {
  return T >= 1 && T <= indexed_max_t && T < kGemma4PrefillBatchMinT && fp8_res &&
         top_k <= 8 && top_k > 0;
}

struct Gemma4IndexedTokenOff {
  int64_t x_elems = 0;
  int64_t y_elems = 0;
  int64_t route = 0;
};

inline Gemma4IndexedTokenOff Gemma4IndexedTokenOffsets(int64_t t, int64_t H, int top_k) {
  return Gemma4IndexedTokenOff{t * H, t * H, t * static_cast<int64_t>(top_k)};
}

inline std::atomic<uint64_t>& Gemma4IndexedHelperHits() {
  static std::atomic<uint64_t> n{0};
  return n;
}

template <typename YT = uint16_t, typename XT = uint16_t>
struct Gemma4IndexedCall {
  int64_t t = 0;
  Gemma4IndexedTokenOff off{};
  bool peer = false;
  const XT* x = nullptr;
  YT* y = nullptr;
  const int32_t* ri = nullptr;
  const float* rw = nullptr;
};

struct Gemma4IndexedDispatchResult {
  bool ok = false;
  uint64_t hits = 0;
  int restores = 0;
  void* y_owner = nullptr;
  bool enqueued = false;
};

// Production T-loop. `fn(call)` is the same-dev or peer helper. `restore()` runs
// after every helper return (success or fail). Does not free `y`.
template <typename YT, typename XT, typename Fn, typename Restore>
inline Gemma4IndexedDispatchResult Gemma4IndexedDispatchTokens(
    int64_t T, int64_t H, int top_k, bool peer, YT* y, const XT* x, const int32_t* ri,
    const float* rw, Fn fn, Restore restore) {
  Gemma4IndexedDispatchResult r;
  r.y_owner = y;
  if (T <= 0 || !y || !x || !ri || !rw) return r;
  for (int64_t t = 0; t < T; ++t) {
    const auto off = Gemma4IndexedTokenOffsets(t, H, top_k);
    const Gemma4IndexedCall<YT, XT> c{t, off, peer, x + off.x_elems, y + off.y_elems,
                                      ri + off.route, rw + off.route};
    const bool one = fn(c);
    r.enqueued = true;
    restore();
    ++r.restores;
    if (!one) {
      r.ok = false;
      return r;
    }
    ++r.hits;
    Gemma4IndexedHelperHits().fetch_add(1, std::memory_order_relaxed);
  }
  r.ok = true;
  return r;
}

// Serial fallback scale. already_scaled=true skips (indexed must not leave rw mutated).
inline void Gemma4ApplyHostExpertScaleOnce(float* hw, const int32_t* hi, const float* hscale,
                                           int64_t T, int top_k, int64_t E, bool already_scaled) {
  if (already_scaled || !hw || !hi || !hscale || T <= 0 || top_k <= 0) return;
  for (int64_t t = 0; t < T; ++t) {
    for (int i = 0; i < top_k; ++i) {
      const size_t o = static_cast<size_t>(t * top_k + i);
      const int e = hi[o];
      if (e >= 0 && e < static_cast<int>(E)) hw[o] *= hscale[static_cast<size_t>(e)];
    }
  }
}

// Independent serial / product-loop token math (host oracle).
inline void Gemma4IndexedHostApplyToken(float* y, const float* x, const int32_t* ri, const float* rw,
                                        int64_t H, int top_k) {
  for (int64_t h = 0; h < H; ++h) y[h] = 0.f;
  for (int g = 0; g < top_k; ++g) {
    const int32_t e = ri[g];
    const float s = static_cast<float>(e + 1) * rw[g];
    for (int64_t h = 0; h < H; ++h) y[h] += x[h] * s;
  }
}

// Independent of Gemma4IndexedHostApplyToken — do not call it here.
inline void Gemma4IndexedHostSerialRef(const float* x, const int32_t* idx, const float* wts,
                                       float* y, int64_t T, int64_t H, int top_k) {
  for (int64_t t = 0; t < T; ++t) {
    float* yt = y + t * H;
    const float* xt = x + t * H;
    const int32_t* idt = idx + t * top_k;
    const float* wt = wts + t * top_k;
    for (int64_t h = 0; h < H; ++h) yt[h] = 0.f;
    for (int g = 0; g < top_k; ++g) {
      const int32_t e = idt[g];
      const float s = static_cast<float>(e + 1) * wt[g];
      for (int64_t h = 0; h < H; ++h) yt[h] += xt[h] * s;
    }
  }
}

inline bool Gemma4IndexedOracleClose(const float* cand, const float* ref, int64_t n,
                                     float* max_abs_out) {
  float max_abs_ref = 0.f;
  float max_abs_diff = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(cand[i]) || !std::isfinite(ref[i])) return false;
    max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));
    max_abs_diff = std::max(max_abs_diff, std::fabs(cand[i] - ref[i]));
    if (ref[i] == 0.f && cand[i] != 0.f) return false;
  }
  if (max_abs_out) *max_abs_out = max_abs_diff;
  const float tol = std::ldexp(max_abs_ref, -static_cast<int>(kGemma4IndexedOraclePow2));
  return max_abs_diff <= tol;
}

enum class Gemma4IndexedArm { SameDev, Peer, None };

inline Gemma4IndexedArm Gemma4IndexedSelectArm(bool fp8_res_same, bool fp8_res_peer) {
  if (fp8_res_same) return Gemma4IndexedArm::SameDev;
  if (fp8_res_peer) return Gemma4IndexedArm::Peer;
  return Gemma4IndexedArm::None;
}

template <typename SameFn, typename PeerFn>
inline bool Gemma4IndexedRunSelectedArm(Gemma4IndexedArm arm, SameFn same, PeerFn peer) {
  if (arm == Gemma4IndexedArm::SameDev) return same();
  if (arm == Gemma4IndexedArm::Peer) return peer();
  return false;
}

struct Gemma4IndexedHelperArgs {
  void* y = nullptr;
  const void* x = nullptr;
  const int32_t* ri = nullptr;
  const float* rw = nullptr;
};

inline Gemma4IndexedHelperArgs Gemma4IndexedPackArgs(void* y, const void* x, const int32_t* ri,
                                                     const float* rw) {
  return Gemma4IndexedHelperArgs{y, x, ri, rw};
}

inline bool Gemma4IndexedArgsEq(const Gemma4IndexedHelperArgs& a, const Gemma4IndexedHelperArgs& b) {
  return a.y == b.y && a.x == b.x && a.ri == b.ri && a.rw == b.rw;
}

enum class Gemma4IndexedScratchKind { TlsT1, OwnedTH };

inline Gemma4IndexedScratchKind Gemma4IndexedScratchKindFor(int64_t T) {
  return T == 1 ? Gemma4IndexedScratchKind::TlsT1 : Gemma4IndexedScratchKind::OwnedTH;
}

struct Gemma4IndexedScratchChoice {
  Gemma4IndexedScratchKind kind = Gemma4IndexedScratchKind::OwnedTH;
  void* y = nullptr;
  int64_t elems = 0;
};

inline bool Gemma4IndexedScratchValidForT(const Gemma4IndexedScratchChoice& c, int64_t T, int64_t H) {
  if (T <= 0 || H <= 0 || c.y == nullptr) return false;
  if (T == 1) return c.kind == Gemma4IndexedScratchKind::TlsT1 && c.elems >= H;
  return c.kind == Gemma4IndexedScratchKind::OwnedTH && c.elems >= T * H;
}

// Release to pool is illegal unless retirement was observed.
inline bool Gemma4IndexedMayReleaseToPool(bool enqueued, bool retire_observed) {
  return !enqueued || retire_observed;
}

// Host model of the production fail path: retire while buffer is still owned,
// then release only if retirement was observed. Release-before-retire is RED.
template <typename Retire>
inline bool Gemma4IndexedFailPathRetireThenMaybeRelease(bool enqueued, bool& owned, bool& released,
                                                        bool& retire_ok, Retire retire) {
  if (!owned) return false;
  retire_ok = true;
  if (enqueued) retire_ok = retire();
  if (!Gemma4IndexedMayReleaseToPool(enqueued, retire_ok)) {
    owned = true;  // quarantine: keep ownership, do not release
    released = false;
    return false;
  }
  owned = false;
  released = true;
  return true;
}

}  // namespace vllm
