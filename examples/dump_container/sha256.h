// Compact SHA-256 (FIPS 180-4) for the dump_container example. Self-contained
// public-domain-style implementation (derived from the algorithm spec, no
// external code); kept local to examples/ on purpose — not a library API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace examples {

class Sha256 {
 public:
  Sha256() { Reset(); }

  void Reset() {
    static constexpr uint32_t kInit[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                          0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                          0x1f83d9abu, 0x5be0cd19u};
    std::memcpy(state_, kInit, sizeof(state_));
    total_len_ = 0;
    buf_len_ = 0;
  }

  void Update(const uint8_t* data, size_t len) {
    total_len_ += len;
    while (len > 0) {
      size_t take = 64 - buf_len_;
      if (take > len) take = len;
      std::memcpy(buf_ + buf_len_, data, take);
      buf_len_ += take;
      data += take;
      len -= take;
      if (buf_len_ == 64) {
        Compress(buf_);
        buf_len_ = 0;
      }
    }
  }

  // Finalizes and returns the lowercase hex digest. The object must be
  // Reset() before reuse.
  std::string HexDigest() {
    uint64_t bit_len = total_len_ * 8;
    uint8_t pad = 0x80;
    Update(&pad, 1);
    uint8_t zero = 0x00;
    while (buf_len_ != 56) Update(&zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i)
      len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    // Update() also advances total_len_, but bit_len was already captured.
    Update(len_be, 8);

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 8; ++i) {
      for (int b = 0; b < 4; ++b) {
        uint8_t byte = static_cast<uint8_t>(state_[i] >> (24 - 8 * b));
        out[static_cast<size_t>(i * 8 + b * 2)] = kHex[byte >> 4];
        out[static_cast<size_t>(i * 8 + b * 2 + 1)] = kHex[byte & 0xf];
      }
    }
    return out;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void Compress(const uint8_t* block) {
    static constexpr uint32_t kK[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
             (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + s1 + ch + kK[i] + w[i];
      uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  uint32_t state_[8];
  uint64_t total_len_ = 0;
  uint8_t buf_[64];
  size_t buf_len_ = 0;
};

}  // namespace examples
