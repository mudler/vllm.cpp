// Shared #785 P1 fixture / oracle / exclusive trace classifier.
// Included by the host suite and the GPU product-seam binary.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace vt_785_p1 {

constexpr int64_t kT = 64;
constexpr int64_t kHq = 2;
constexpr int64_t kHk = 1;
constexpr int64_t kD = 256;
constexpr int64_t kBlock = 16;
constexpr float kScale = 0.0625f;
constexpr bool kCausal = true;
constexpr int64_t kWindowLeft = 32;
constexpr int64_t kWindowRight = 0;
constexpr uint32_t kQSeed = 78525601u;
constexpr uint32_t kKSeed = 78525602u;
constexpr uint32_t kVSeed = 78525603u;

constexpr double kAbsFloor = 1.5e-2;
constexpr double kRel = 1.0e-2;
constexpr double kMinCorr = 0.999;

constexpr const char* kQHash =
    "27f164e220edf8d37ddfd1783f0c390968b9d2314ff5d89b7aea3ef50f0eba72";
constexpr const char* kKHash =
    "d803e64df022e1cb2049b06d65fc36da02a79ac09f77831e967237e47d49f7ca";
constexpr const char* kVHash =
    "910548159fd5a5a478573dac6de96f374014cb0b76dac7389071f265dfe1cae7";

inline uint16_t F32ToBf16Bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>((x + rounding) >> 16);
}
inline float Bf16BitsToF32(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

inline std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

struct Sha256 {
  uint32_t s[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t nbits = 0;
  uint8_t buf[64]{};
  size_t fill = 0;

  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void Block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
             (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6],
             h = s[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = h + S1 + ch + K[i] + w[i];
      const uint32_t S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    s[0] += a;
    s[1] += b;
    s[2] += c;
    s[3] += d;
    s[4] += e;
    s[5] += f;
    s[6] += g;
    s[7] += h;
  }
  void Update(const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    nbits += uint64_t(n) * 8u;
    while (n) {
      const size_t take = std::min(n, size_t(64 - fill));
      std::memcpy(buf + fill, p, take);
      fill += take;
      p += take;
      n -= take;
      if (fill == 64) {
        Block(buf);
        fill = 0;
      }
    }
  }
  std::string Hex() {
    uint8_t tail[64 + 8];
    std::memset(tail, 0, sizeof(tail));
    std::memcpy(tail, buf, fill);
    tail[fill] = 0x80;
    size_t used = fill + 1;
    if (used > 56) {
      Block(tail);
      std::memset(tail, 0, 64);
    }
    for (int i = 0; i < 8; ++i) tail[63 - i] = static_cast<uint8_t>(nbits >> (8 * i));
    Block(tail);
    std::string out(64, '0');
    static const char* hex = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
      for (int b = 0; b < 4; ++b) {
        const uint8_t v = static_cast<uint8_t>(s[i] >> (24 - 8 * b));
        out[static_cast<size_t>(i * 8 + b * 2)] = hex[v >> 4];
        out[static_cast<size_t>(i * 8 + b * 2 + 1)] = hex[v & 0xf];
      }
    }
    return out;
  }
};

inline std::string Sha256U16Le(const std::vector<uint16_t>& v) {
  Sha256 h;
  std::vector<uint8_t> raw(v.size() * 2);
  for (size_t i = 0; i < v.size(); ++i) {
    raw[2 * i] = static_cast<uint8_t>(v[i] & 0xff);
    raw[2 * i + 1] = static_cast<uint8_t>((v[i] >> 8) & 0xff);
  }
  h.Update(raw.data(), raw.size());
  return h.Hex();
}

struct Fixture {
  std::vector<uint16_t> q_bf16, k_bf16, v_bf16;
  std::vector<float> q_f32, k_f32, v_f32;
  std::vector<int32_t> block_table, seq_lens, qsl;
  int64_t num_blocks = 0;
};

inline Fixture MakeFixture() {
  Fixture f;
  f.num_blocks = (kT + kBlock - 1) / kBlock;
  auto q = RandF32(static_cast<size_t>(kT * kHq * kD), kQSeed);
  auto k = RandF32(static_cast<size_t>(f.num_blocks * kBlock * kHk * kD), kKSeed);
  auto v = RandF32(static_cast<size_t>(f.num_blocks * kBlock * kHk * kD), kVSeed);
  f.q_bf16.resize(q.size());
  f.k_bf16.resize(k.size());
  f.v_bf16.resize(v.size());
  f.q_f32.resize(q.size());
  f.k_f32.resize(k.size());
  f.v_f32.resize(v.size());
  for (size_t i = 0; i < q.size(); ++i) {
    f.q_bf16[i] = F32ToBf16Bits(q[i]);
    f.q_f32[i] = Bf16BitsToF32(f.q_bf16[i]);
  }
  for (size_t i = 0; i < k.size(); ++i) {
    f.k_bf16[i] = F32ToBf16Bits(k[i]);
    f.k_f32[i] = Bf16BitsToF32(f.k_bf16[i]);
    f.v_bf16[i] = F32ToBf16Bits(v[i]);
    f.v_f32[i] = Bf16BitsToF32(f.v_bf16[i]);
  }
  f.block_table.resize(static_cast<size_t>(f.num_blocks));
  for (int64_t i = 0; i < f.num_blocks; ++i)
    f.block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  f.seq_lens = {static_cast<int32_t>(kT)};
  f.qsl = {0, static_cast<int32_t>(kT)};
  return f;
}

inline std::vector<float> Oracle(const Fixture& f) {
  const int64_t qpk = kHq / kHk;
  std::vector<float> out(static_cast<size_t>(kT * kHq * kD), 0.0f);
  for (int64_t local = 0; local < kT; ++local) {
    const int64_t p = local;
    const int64_t jmin = std::max<int64_t>(0, p - kWindowLeft);
    int64_t jmax = kCausal ? p : kT - 1;
    jmax = std::min(jmax, p + kWindowRight);
    jmax = std::min(jmax, kT - 1);
    for (int64_t h = 0; h < kHq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (local * kHq + h) * kD;
      std::vector<float> sc(static_cast<size_t>(jmax - jmin + 1));
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        const int64_t blk = f.block_table[static_cast<size_t>(j / kBlock)];
        const int64_t off = j % kBlock;
        const int64_t kbase = ((blk * kBlock + off) * kHk + g) * kD;
        float dot = 0.0f;
        for (int64_t e = 0; e < kD; ++e)
          dot += f.q_f32[static_cast<size_t>(qoff + e)] * f.k_f32[static_cast<size_t>(kbase + e)];
        dot *= kScale;
        sc[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (float& s : sc) {
        s = std::exp(s - m);
        denom += s;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < kD; ++e) {
        float a = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const int64_t blk = f.block_table[static_cast<size_t>(j / kBlock)];
          const int64_t off = j % kBlock;
          const int64_t vbase = ((blk * kBlock + off) * kHk + g) * kD;
          a += sc[static_cast<size_t>(j - jmin)] * inv * f.v_f32[static_cast<size_t>(vbase + e)];
        }
        out[static_cast<size_t>(qoff + e)] = a;
      }
    }
  }
  return out;
}

struct OracleStats {
  double max_abs = 0;
  double corr = 0;
  size_t nonfinite = 0;
  size_t violations = 0;
  bool ok = false;
};

inline OracleStats Score(const std::vector<float>& got, const std::vector<float>& ref) {
  OracleStats s;
  if (got.size() != ref.size() || got.empty()) return s;
  double mean_g = 0, mean_r = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) ++s.nonfinite;
    mean_g += got[i];
    mean_r += ref[i];
  }
  mean_g /= static_cast<double>(ref.size());
  mean_r /= static_cast<double>(ref.size());
  double num = 0, dg = 0, dr = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double err = std::abs(got[i] - ref[i]);
    if (err > s.max_abs) s.max_abs = err;
    const double thr = kAbsFloor + kRel * std::abs(static_cast<double>(ref[i]));
    if (err > thr) ++s.violations;
    const double ag = got[i] - mean_g;
    const double ar = ref[i] - mean_r;
    num += ag * ar;
    dg += ag * ag;
    dr += ar * ar;
  }
  s.corr = (dg > 0.0 && dr > 0.0) ? (num / std::sqrt(dg * dr)) : 0.0;
  s.ok = s.nonfinite == 0 && s.violations == 0 && s.corr >= kMinCorr;
  return s;
}

inline constexpr const char* kExactWmma[] = {
    "PagedAttnPrefillSharedKWmma<2, 8, 16, 32, false>",
    "PagedAttnPrefillSharedKWmma<2,8,16,32,false>",
    "PagedAttnPrefillSharedKWmmaILi2ELi8ELi16ELi32ELb0E",
};
inline constexpr const char* kExactScalar[] = {
    "PagedAttnPrefillSharedK<2, 8, 32, 32>",
    "PagedAttnPrefillSharedK<2,8,32,32>",
    "PagedAttnPrefillSharedKILi2ELi8ELi32ELi32EE",
};

inline std::string ScalarLines(const std::string& text) {
  std::string filtered;
  filtered.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    const size_t nl = text.find('\n', i);
    const size_t end = nl == std::string::npos ? text.size() : nl;
    const auto line = text.substr(i, end - i);
    if (line.find("SharedKWmma") == std::string::npos) {
      filtered.append(line);
      filtered.push_back('\n');
    }
    i = end == text.size() ? end : end + 1;
  }
  return filtered;
}

inline bool HasAny(const std::string& text, const char* const* ms, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (text.find(ms[i]) != std::string::npos) return true;
  }
  return false;
}

inline std::string StripExact(const std::string& text, const char* const* ms, size_t n) {
  std::string out = text;
  for (size_t i = 0; i < n; ++i) {
    for (;;) {
      const auto pos = out.find(ms[i]);
      if (pos == std::string::npos) break;
      out.erase(pos, std::strlen(ms[i]));
    }
  }
  return out;
}

// Family vs exact specialization:
// A = exact WMMA <2,8,16,32,false> AND no SharedK family AND no other WMMA.
// B = exact scalar <2,8,32,32> AND no WMMA family AND no other scalar.
// Wrong BM/BN, wrong qg/d, mixed, or none => '?'.
inline char ClassifyArm(const std::string& text) {
  const auto scalar_blob = ScalarLines(text);
  const bool exact_wmma =
      HasAny(text, kExactWmma, sizeof(kExactWmma) / sizeof(kExactWmma[0]));
  const bool exact_scalar = HasAny(
      scalar_blob, kExactScalar, sizeof(kExactScalar) / sizeof(kExactScalar[0]));
  const bool wmma_family = text.find("PagedAttnPrefillSharedKWmma") != std::string::npos;
  const bool scalar_family =
      scalar_blob.find("PagedAttnPrefillSharedK") != std::string::npos;
  const bool other_wmma =
      StripExact(text, kExactWmma, sizeof(kExactWmma) / sizeof(kExactWmma[0]))
          .find("PagedAttnPrefillSharedKWmma") != std::string::npos;
  const bool other_scalar =
      StripExact(scalar_blob, kExactScalar,
                 sizeof(kExactScalar) / sizeof(kExactScalar[0]))
          .find("PagedAttnPrefillSharedK") != std::string::npos;
  if (exact_wmma && !scalar_family && !other_wmma) return 'A';
  if (exact_scalar && !wmma_family && !other_scalar) return 'B';
  return '?';
}

}  // namespace vt_785_p1
