// SPEC-DFLASH2 W5 (#1314) — the GGUF DFlash2 DRAFTER ARM, and its LOWER BOUND.
//
// WHAT THIS WAVE SHIPS. Through W4 a GGUF DFlash2 checkpoint was refused at
// startup by name: `MakeDflashGgufConfig` read none of the DFlash2 metadata
// beyond the keys that identify the file, and `LoadQwen3DFlashFromGguf` had no
// name for a conv or a selector tensor, so admitting the file would have loaded
// a DFlash1 draft out of a DFlash2 checkpoint. This wave gives both a name. The
// tensor names below are READ off `z-lab/Qwen3.8-27B-DFlash2-GGUF` @
// `57ab3265056d4024870b0621cfc2c127537020ed` rather than guessed, and the
// asset-gated case at the end holds them against the published artifacts.
//
// WHY A TOKEN GATE CANNOT GATE THIS ARM, and what replaces it. The DFlash GGUF
// lane DEQUANTIZES wholesale to bf16 by design — the draft is a handful of
// layers, so the resolver hands `LoadQwen3DFlash` bf16 views and the entire
// safetensors body is reused unchanged (see the file comment on
// src/vllm/model_executor/models/qwen3_dflash_gguf.cpp). That makes the standing
// trap acute rather than absent: if a k-quant tensor never took the quantized
// path, every token would still match and every golden would still pass, because
// the arm that is claimed shipped is the arm that never ran. So the quantized
// cases below carry a LOWER BOUND, and none of its parts is a token comparison.
// THE PARTS ARE NOT THREE EQUAL LEGS, and the W5 review was right to say so
// (#1314 F3): only L2 bounds the decode.
//
//   L1  a PRECONDITION on the FIXTURE, not a bound on the decode. It reads the
//       ggml type and the byte count off the tensor table of the file this test
//       has just written, so what it proves is that the fixture really is
//       block-encoded — 144 bytes per 256 elements for Q4_K, 34 per 32 for
//       Q8_0, strictly under the 2 bytes/element bf16 costs. That matters,
//       because a fixture that had quietly become bf16 would make L2 and L3
//       compare the wrong thing. It says NOTHING about the loader: a loader
//       that hashed the bytes and returned garbage passes L1 unchanged.
//   L2  THE BOUND. The VALUES, bit-for-bit, against an expectation this file
//       COMPUTES from the block integers it chose, never by calling the
//       production dequantizer. The encoder here is the inverse of the format,
//       so the comparison is a round trip through two independent
//       implementations and not a shared helper agreeing with itself. This is
//       the leg every decoder mutation lands on: truncating either 6-bit field
//       of `get_scale_min_k4` reddens here and nowhere else.
//   L3  a COROLLARY of L2 that is cheap and reaches further: DIFFERENCE from the
//       bf16 arm, tensor by tensor and again at the block logits. Every arm is
//       encoded from the SAME source values, so a loader that ignored the
//       quantized bytes and produced the dense answer is caught end to end
//       rather than only at the tensor. A byte-hashing loader passes L1 and L3
//       — the hash differs from bf16 — and fails L2 hard, which is the ordering
//       these three have.
//
// AND THE NAME MAP IS GATED BY CONSTRUCTION, not by reading it. The bf16 GGUF
// arm is required BIT-IDENTICAL to the same draft written as safetensors under
// the published HF names. A GGUF name mapped to the wrong tensor, transposed, or
// silently dropped moves those bytes; only a correct map reproduces them.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "../gguf_builder.h"

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/models/qwen3_dflash_gguf.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash/speculator.h"
#include "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.h"
#include "vt/dtype.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using vllm::HfConfig;
using vllm::Qwen3DFlashModel;
using vllm::Qwen3DFlashWeights;

namespace {

// ---------------------------------------------------------------------------
// Geometry. Every inner (ne0) extent is a multiple of 256, which is what a
// k-quant row requires and what the published file has at H = 5120 / rank = 256.
// Shrinking H below that would make the fixture unencodable in Q4_K and would
// therefore gate a shape the container cannot hold.
// ---------------------------------------------------------------------------
struct Dims {
  int64_t H = 256;
  int64_t Hq = 4, Hkv = 1, Dh = 64;
  int64_t I = 256;
  int64_t vocab = 256;
  int64_t layers = 2;
  int64_t taps_fc = 2;      // len(target_layer_ids)
  int64_t conv_taps = 2;    // dflash.conv_kernel_size
  int64_t conv_group = 16;  // dflash.conv_group_size
  int64_t rank = 256;       // dflash.selector_rank
  int64_t top_k = 4;        // dflash.selector_top_k
  int64_t block = 8;        // 1 + k
  // The three OUTPUT SCALARS, negative meaning "declare no key at all" -- which
  // is the shape of BOTH published DFlash2 GGUFs and therefore the arm a gate
  // built from them alone would silently measure (#1327, `## Risks/decisions` D9).
  double output_multiplier = -1.0;
  double final_logit_softcapping = -1.0;
  double input_embedding_scale = -1.0;
  int64_t qdim() const { return Hq * Dh; }
  int64_t kdim() const { return Hkv * Dh; }
  int64_t groups() const { return H / conv_group; }
  int64_t proj_out() const { return 2 * conv_taps * groups(); }
};

// Deterministic source values, in [-0.3, 0.3]. The band is chosen so that the
// Q4_K encoder below covers it without clamping: its representable interval at
// d = 2^-6 / dmin = 2^-7 is [-63/128, 210/128].
std::vector<float> Src(int64_t n, double seed) {
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i)
    v[static_cast<size_t>(i)] =
        static_cast<float>(0.3 * std::sin(seed + 0.7 * static_cast<double>(i)));
  return v;
}

std::string F32Bytes(const std::vector<float>& v) {
  std::string s(v.size() * 4, '\0');
  std::memcpy(s.data(), v.data(), v.size() * 4);
  return s;
}

std::string Bf16Bytes(const std::vector<float>& v) {
  std::vector<uint16_t> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = vt::F32ToBF16(v[i]);
  std::string s(b.size() * 2, '\0');
  std::memcpy(s.data(), b.data(), b.size() * 2);
  return s;
}

std::vector<float> RoundTripBf16(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = vt::BF16ToF32(vt::F32ToBF16(v[i]));
  return out;
}

// ---------------------------------------------------------------------------
// The two encoders. Both are the INVERSE of the ggml block format, written here
// so that the expectation is produced by a different implementation from the one
// under test. Neither aims to be llama.cpp's optimal quantizer — the container
// does not care how the scales were chosen, only that the decode reproduces
// them, which is exactly what is being gated.
// ---------------------------------------------------------------------------

// Q4_K only: WHAT THE ENCODER ACTUALLY DROVE, per packed field. Every number
// here is accumulated while encoding and asserted by `CheckQuantArm`; none of it
// is a claim in a comment. Index 0 is sub-blocks 0..3 (`get_scale_min_k4`'s
// whole-field half) and index 1 is sub-blocks 4..7 (its split half); for the
// nibbles, index 0 is the LOW nibble of a byte (even sub-blocks) and index 1 the
// HIGH nibble (odd sub-blocks).
struct Q4KCoverage {
  int scale_above_15[2] = {0, 0};  // sub-blocks whose 6-bit scale exceeds 15
  int min_above_15[2] = {0, 0};    // sub-blocks whose 6-bit min exceeds 15
  int scale_bits[2] = {0, 0};      // OR of every 6-bit scale -> 63 covers the field
  int min_bits[2] = {0, 0};        // OR of every 6-bit min
  int nibble_bits[2] = {0, 0};     // OR of every 4-bit q -> 15 covers the field
  void Merge(const Q4KCoverage& o) {
    for (int i = 0; i < 2; ++i) {
      scale_above_15[i] += o.scale_above_15[i];
      min_above_15[i] += o.min_above_15[i];
      scale_bits[i] |= o.scale_bits[i];
      min_bits[i] |= o.min_bits[i];
      nibble_bits[i] |= o.nibble_bits[i];
    }
  }
};

// Q8_0 only: WHAT THE ENCODER ACTUALLY DROVE in the block header's fp16 `d`.
// Index i counts the blocks written at `kQ8Dfp16[i]`. Both must be nonzero on
// EVERY compared tensor, which is what makes a decoder that reads `d` once per
// TENSOR rather than once per BLOCK visible; see `EncodeQ8_0`.
struct Q8Coverage {
  int blocks_at[2] = {0, 0};
};

// A blob plus the values it decodes to, in the tensor's flat order.
struct Encoded {
  std::string bytes;
  std::vector<float> expect;  // exact; every value below is a small n / 2^p
  Q4KCoverage q4k;            // Q4_K only; see above
  Q8Coverage q8;              // Q8_0 only; see below
};

// Q8_0: { fp16 d; int8 qs[32] } per 32 elements, y = qs * d.
//
// `d` is CHOSEN rather than derived from the block maximum, which is a legal
// Q8_0 block and removes the only step that would need an fp16 rounding helper.
//
// IT ALTERNATES BY BLOCK between two powers of two, and the alternation is
// COUNTED. Through W5 it was one fixed value in every block of every tensor,
// which is the #1314 F1 class of instrument gap one field up from the packed
// integers: `DequantQ8_0` reads `d` from EACH block header, and a fixture whose
// blocks all carry the same `d` cannot tell that apart from a decoder that reads
// it ONCE per tensor and reuses it. Measured 2026-08-21 on the pre-repair
// fixture: rewriting `const float d = ReadF16(blk)` to `ReadF16(data)` -- the
// whole tensor decoded at block 0's scale -- left this suite at 9 cases / 4730
// assertions / `Status: SUCCESS!` / rc 0. After the alternation the same
// mutation reddens 1 case / 7 assertions.
//
// IT COSTS NO BIT-EXACTNESS, which is why W5's recorded reason for leaving this
// open (O18) was wrong. Both values are POWERS OF TWO, so every decoded value is
// still q/2^p: at |x| <= 0.3 the magnitude is |q| <= 77 at 2^-8 and |q| <= 39 at
// 2^-7, both well inside int8 and inside the eight significant bits a bf16 store
// keeps. The comparison therefore stays equality rather than a tolerance, and
// the suite passing it bit-for-bit after the change is the measurement of that,
// rather than the argument above.
constexpr uint16_t kQ8Dfp16[2] = {0x1C00, 0x2000};  // 2^-8, 2^-7
constexpr float kQ8D[2] = {1.0f / 256.0f, 1.0f / 128.0f};

Encoded EncodeQ8_0(const std::vector<float>& x) {
  REQUIRE(x.size() % 32 == 0);
  Encoded e;
  e.expect.resize(x.size());
  e.bytes.reserve(x.size() / 32 * 34);
  for (size_t b = 0; b < x.size(); b += 32) {
    const int w = static_cast<int>((b / 32) % 2);
    ++e.q8.blocks_at[w];
    e.bytes.push_back(static_cast<char>(kQ8Dfp16[w] & 0xff));
    e.bytes.push_back(static_cast<char>(kQ8Dfp16[w] >> 8));
    for (size_t l = 0; l < 32; ++l) {
      int q = static_cast<int>(std::lround(x[b + l] / kQ8D[w]));
      q = std::max(-127, std::min(127, q));
      e.bytes.push_back(static_cast<char>(static_cast<int8_t>(q)));
      e.expect[b + l] = static_cast<float>(q) * kQ8D[w];
    }
  }
  return e;
}

// Q4_K: { fp16 d; fp16 dmin; uint8 scales[12]; uint8 qs[128] } per 256 elements.
//
// The subtle half is `scales`: eight 6-bit scales and eight 6-bit mins packed
// into twelve bytes, four of them split across two bytes. This is the inverse of
// llama.cpp's `get_scale_min_k4`, and it is the reason this arm needs its own
// case at all — a wrong unpack yields finite, plausible weights.
//
// `d` and `dmin` are FIXED at 2^-6 and 2^-7, and the per-sub-block sc and m are
// derived from the data. Every decoded value is then (2*sc*q - m)/128, and the
// numerator is kept within eight significant bits so it survives the loader's
// bf16 store EXACTLY and the comparison can be bit-for-bit.
//
// THE PACKED FIELDS ARE DRIVEN ACROSS THEIR OWN WIDTH, AND THE COVERAGE IS
// COUNTED. `sc` and `m` are SIX-bit fields and `q` is a FOUR-bit one, and a
// fixture whose values do not span a field cannot detect a decoder that ignores
// part of it: every assertion still passes, because the bits that were dropped
// were never set. W5 fixed one instance of that and left two. This is the whole
// set, driven here and asserted in `CheckQuantArm` rather than described:
//
//   sc, sub-blocks 4..7 — the SPLIT half. `get_scale_min_k4` assembles these
//       from a LOW nibble in `scales[j+4]` and a HIGH pair of bits in a
//       DIFFERENT byte, so a scale never above 15 leaves that assembly untested:
//       `sc >> 4` is 0, the high pair is zero, and an unpack that dropped it
//       reproduces every value. Lifted by `16*(sb-3)`, capped at 63.
//   sc, sub-blocks 0..3 — the WHOLE-field half, and the W5 review's F1. `j < 4`
//       reads `q[j] & 63` in one piece, and the data-derived scale here is
//       EXACTLY 3 for every low sub-block of every tensor this fixture writes:
//       two bits of six. A production `q[j] & 15` therefore reproduced every
//       value and the suite stayed green at 9/9. Sub-block 0 is lifted by 16 and
//       sub-block 1 saturates at 63, so the OR over the half is all six bits and
//       no dropped bit survives.
//   m, both halves. Derived from the data it lands at 36..38 — above 15, so a
//       `& 15` mutation is caught, but bits 3 and 4 are never set. `m` is the
//       encoder's FREE CHOICE as long as it reaches the block floor, so
//       sub-blocks 2 and 6 take 63 and the OR over each half is again six bits.
//       The scale is derived AFTER this, from a span that includes it.
//   q, both nibble positions — and this is why sub-blocks 2 and 3 KEEP their
//       data-derived scale. A large scale costs resolution: q falls to 0..2, and
//       lifting all four low sub-blocks the way the high half is lifted would
//       have collapsed the low nibble AND the high one to that range, trading
//       the repair for the same defect one field down. An EVEN sub-block writes
//       the LOW nibble of its byte and the ODD one above it writes the HIGH
//       nibble, so one fine sub-block of each parity keeps both spanning 0..15.
//
// Nothing here aims to be llama.cpp's quantizer. The container does not care how
// the scales and mins were chosen, only that the decode reproduces them, which
// is exactly what is being gated.
constexpr uint16_t kQ4KDfp16 = 0x2400;     // 2^-6
constexpr uint16_t kQ4KDminfp16 = 0x2000;  // 2^-7

Encoded EncodeQ4_K(const std::vector<float>& x) {
  REQUIRE(x.size() % 256 == 0);
  Encoded e;
  e.expect.resize(x.size());
  for (size_t sb0 = 0; sb0 < x.size(); sb0 += 256) {
    int sc[8], mn[8];
    uint8_t qs[128] = {0};
    for (int sb = 0; sb < 8; ++sb) {
      const size_t base = sb0 + static_cast<size_t>(sb) * 32;
      float lo = x[base], hi = x[base];
      for (size_t l = 1; l < 32; ++l) {
        lo = std::min(lo, x[base + l]);
        hi = std::max(hi, x[base + l]);
      }
      // decoded(q) = (2*sc*q - m)/128, q in [0,15]: m sets the floor, sc the span.
      // `m` need only REACH the floor, so two sub-blocks take the widest legal
      // value instead and the scale below adapts to the span that results.
      mn[sb] = std::max(0, std::min(63, static_cast<int>(std::lround(-lo * 128.0f))));
      if (sb == 2 || sb == 6) mn[sb] = 63;
      const float span = hi * 128.0f + static_cast<float>(mn[sb]);
      sc[sb] = std::max(1, std::min(7, static_cast<int>(std::ceil(span / 30.0f))));
      if (sb >= 4) {
        sc[sb] = std::min(63, sc[sb] + 16 * (sb - 3));  // the SPLIT half
      } else if (sb == 0) {
        sc[sb] = std::min(63, sc[sb] + 16);  // the whole-field half, past 15
      } else if (sb == 1) {
        sc[sb] = std::min(63, sc[sb] + 60);  // ...and saturated, so its OR is 63
      }
      const int half = sb < 4 ? 0 : 1;
      if (sc[sb] > 15) ++e.q4k.scale_above_15[half];
      if (mn[sb] > 15) ++e.q4k.min_above_15[half];
      e.q4k.scale_bits[half] |= sc[sb];
      e.q4k.min_bits[half] |= mn[sb];
      for (size_t l = 0; l < 32; ++l) {
        int q = static_cast<int>(
            std::lround((x[base + l] * 128.0f + static_cast<float>(mn[sb])) /
                        (2.0f * static_cast<float>(sc[sb]))));
        q = std::max(0, std::min(15, q));
        e.q4k.nibble_bits[sb % 2] |= q;
        // Nibble placement, straight off `dequantize_row_q4_K`: an EVEN
        // sub-block reads the low nibbles of its 32 bytes and the ODD one above
        // it reads the high nibbles of the SAME bytes.
        const size_t byte = static_cast<size_t>(sb / 2) * 32 + l;
        if (sb % 2 == 0) {
          qs[byte] = static_cast<uint8_t>((qs[byte] & 0xf0) | q);
        } else {
          qs[byte] = static_cast<uint8_t>((qs[byte] & 0x0f) | (q << 4));
        }
        e.expect[base + l] =
            static_cast<float>(2 * sc[sb] * q - mn[sb]) / 128.0f;
      }
    }
    // The 6-bit pack. scales[0..3] carry sc[0..3] plus the HIGH two bits of
    // sc[4..7]; scales[4..7] carry m[0..3] plus the HIGH two bits of m[4..7];
    // scales[8..11] carry the LOW nibbles of sc[4..7] and m[4..7].
    uint8_t scales[12] = {0};
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<uint8_t>((sc[i] & 63) | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] = static_cast<uint8_t>((mn[i] & 63) | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] =
          static_cast<uint8_t>((sc[i + 4] & 0xF) | ((mn[i + 4] & 0xF) << 4));
    }
    e.bytes.push_back(static_cast<char>(kQ4KDfp16 & 0xff));
    e.bytes.push_back(static_cast<char>(kQ4KDfp16 >> 8));
    e.bytes.push_back(static_cast<char>(kQ4KDminfp16 & 0xff));
    e.bytes.push_back(static_cast<char>(kQ4KDminfp16 >> 8));
    e.bytes.append(reinterpret_cast<const char*>(scales), 12);
    e.bytes.append(reinterpret_cast<const char*>(qs), 128);
  }
  return e;
}

// ---------------------------------------------------------------------------
// The draft, as one table of tensors: the HF name the safetensors loader asks
// for, the GGUF name llama.cpp writes, the torch shape, and whether the arm
// quantizes it. Norms and BOTH conv base kernels stay F32 in every published
// arm, which is measured rather than assumed — see the asset-gated case.
// ---------------------------------------------------------------------------
struct TSpec {
  std::string hf;
  std::string gguf;
  std::vector<int64_t> shape;  // torch order
  double seed;
  bool quantized;              // false => F32 in every arm
};

std::vector<TSpec> DraftTensors(const Dims& d) {
  std::vector<TSpec> t;
  t.push_back({"fc.weight", "fc.weight", {d.H, d.H * d.taps_fc}, 0.2, true});
  t.push_back({"hidden_norm.weight", "enc.output_norm.weight", {d.H}, 0.3, false});
  t.push_back({"norm.weight", "output_norm.weight", {d.H}, 0.4, false});
  for (int64_t l = 0; l < d.layers; ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    const std::string g = "blk." + std::to_string(l) + ".";
    const double s = 1.0 + static_cast<double>(l);
    t.push_back({b + "input_layernorm.weight", g + "attn_norm.weight", {d.H}, s + 0.1, false});
    t.push_back({b + "post_attention_layernorm.weight", g + "ffn_norm.weight", {d.H}, s + 0.2, false});
    t.push_back({b + "self_attn.q_proj.weight", g + "attn_q.weight", {d.qdim(), d.H}, s + 0.3, true});
    t.push_back({b + "self_attn.k_proj.weight", g + "attn_k.weight", {d.kdim(), d.H}, s + 0.4, true});
    t.push_back({b + "self_attn.v_proj.weight", g + "attn_v.weight", {d.kdim(), d.H}, s + 0.5, true});
    t.push_back({b + "self_attn.o_proj.weight", g + "attn_output.weight", {d.H, d.qdim()}, s + 0.6, true});
    t.push_back({b + "self_attn.q_norm.weight", g + "attn_q_norm.weight", {d.Dh}, s + 0.7, false});
    t.push_back({b + "self_attn.k_norm.weight", g + "attn_k_norm.weight", {d.Dh}, s + 0.8, false});
    t.push_back({b + "mlp.gate_proj.weight", g + "ffn_gate.weight", {d.I, d.H}, s + 0.9, true});
    t.push_back({b + "mlp.up_proj.weight", g + "ffn_up.weight", {d.I, d.H}, s + 1.1, true});
    t.push_back({b + "mlp.down_proj.weight", g + "ffn_down.weight", {d.H, d.I}, s + 1.2, true});
    // SPEC-DFLASH2 W5: the four conv tensors, under the names the published
    // GGUF stores them under. `attn_` maps to `attention_conv` and `ffn_` to
    // `mlp_conv`, which is llama.cpp's own sublayer vocabulary.
    t.push_back({b + "attention_conv.base_kernel", g + "attn_conv_base",
                 {2, d.conv_taps, d.H}, s + 2.0, false});
    t.push_back({b + "attention_conv.kernel_projection.weight", g + "attn_conv_proj.weight",
                 {d.proj_out(), d.H}, s + 2.1, true});
    t.push_back({b + "mlp_conv.base_kernel", g + "ffn_conv_base",
                 {2, d.conv_taps, d.H}, s + 2.2, false});
    t.push_back({b + "mlp_conv.kernel_projection.weight", g + "ffn_conv_proj.weight",
                 {d.proj_out(), d.H}, s + 2.3, true});
  }
  // The selector's three tensors. The two codebooks share a shape, so they are
  // filled from DIFFERENT seeds: a map that swapped them produces different
  // scores rather than the same ones.
  t.push_back({"candidate_selector.hidden_projection.weight", "selector_hidden.weight",
               {d.rank, d.H}, 3.3, true});
  t.push_back({"candidate_selector.predecessor_codebook", "selector_predecessor.weight",
               {d.vocab, d.rank}, 4.4, true});
  t.push_back({"candidate_selector.successor_codebook", "selector_successor.weight",
               {d.vocab, d.rank}, 5.5, true});
  return t;
}

int64_t Numel(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

enum class Arm { kBf16, kQ8_0, kQ4_K };

const char* ArmName(Arm a) {
  switch (a) {
    case Arm::kBf16: return "BF16";
    case Arm::kQ8_0: return "Q8_0";
    case Arm::kQ4_K: return "Q4_K";
  }
  return "?";
}

// ggml type ids, as the reader reports them.
constexpr uint32_t kGgmlF32 = 0, kGgmlQ8_0 = 8, kGgmlQ4_K = 12, kGgmlBf16 = 30;

// One built draft file plus, per HF tensor name, the values it must decode to.
struct BuiltGguf {
  std::string bytes;
  std::map<std::string, std::vector<float>> expect;
  // Coverage is kept PER TENSOR, keyed by the HF name, and never pre-summed.
  // A total over every quantized tensor in the file would let a tensor L2 does
  // NOT compare satisfy the precondition for the ones it does: the counters
  // would say the field was driven while the compared set never drove it. The
  // merge therefore happens in `CheckQuantArm`, over `Dflash2Names()` alone.
  std::map<std::string, Q4KCoverage> q4k_by_hf;
  std::map<std::string, Q8Coverage> q8_by_hf;
};

BuiltGguf BuildDraftGguf(const Dims& d, Arm arm, bool dflash2) {
  gguf_test::GgufModelBuilder b;
  const std::string p = "dflash.";
  b.AddKv(gguf_test::StrKv("general.architecture", "dflash"));
  b.AddKv(gguf_test::U32Kv(p + "block_count", static_cast<uint32_t>(d.layers)));
  b.AddKv(gguf_test::U32Kv(p + "embedding_length", static_cast<uint32_t>(d.H)));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count", static_cast<uint32_t>(d.Hq)));
  b.AddKv(gguf_test::U32Kv(p + "attention.head_count_kv", static_cast<uint32_t>(d.Hkv)));
  b.AddKv(gguf_test::U32Kv(p + "attention.key_length", static_cast<uint32_t>(d.Dh)));
  b.AddKv(gguf_test::U32Kv(p + "attention.value_length", static_cast<uint32_t>(d.Dh)));
  b.AddKv(gguf_test::U32Kv(p + "feed_forward_length", static_cast<uint32_t>(d.I)));
  b.AddKv(gguf_test::F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(gguf_test::F32Kv(p + "rope.freq_base", 1e7f));
  b.AddKv(gguf_test::U32Kv(p + "attention.sliding_window", 2048));
  b.AddKv(gguf_test::U32Kv(p + "block_size", static_cast<uint32_t>(d.block)));
  b.AddKv(gguf_test::BoolArrayKv(p + "attention.sliding_window_pattern",
                                 std::vector<bool>(static_cast<size_t>(d.layers), true)));
  {
    std::vector<int32_t> tl;
    for (int64_t i = 0; i < d.taps_fc; ++i)
      tl.push_back(static_cast<int32_t>(3 * i + 1));  // stored +1-offset
    b.AddKv(gguf_test::I32ArrayKv(p + "target_layers", tl));
  }
  b.AddKv(gguf_test::U32Kv("tokenizer.ggml.mask_token_id",
                           static_cast<uint32_t>(d.vocab - 1)));
  // The draft GGUF's OWN vocabulary. It is what sizes the selector's codebooks
  // at load, because a DFlash draft declares no vocab key -- it shares the
  // target's embedding and head, so `MakeDflashGgufConfig` deliberately leaves
  // `vocab_size` 0 and the caller fills it from the target.
  {
    std::vector<std::string> toks;
    toks.reserve(static_cast<size_t>(d.vocab));
    for (int64_t i = 0; i < d.vocab; ++i) toks.push_back("t" + std::to_string(i));
    b.AddKv(gguf_test::StrArrayKv("tokenizer.ggml.tokens", toks));
  }
  if (dflash2) {
    b.AddKv(gguf_test::BoolKv(p + "attention.causal", false));
    b.AddKv(gguf_test::U32Kv(p + "conv_kernel_size", static_cast<uint32_t>(d.conv_taps)));
    b.AddKv(gguf_test::U32Kv(p + "conv_group_size", static_cast<uint32_t>(d.conv_group)));
    b.AddKv(gguf_test::U32Kv(p + "selector_rank", static_cast<uint32_t>(d.rank)));
    b.AddKv(gguf_test::U32Kv(p + "selector_top_k", static_cast<uint32_t>(d.top_k)));
    if (d.output_multiplier >= 0.0)
      b.AddKv(gguf_test::F32Kv(p + "output_multiplier",
                               static_cast<float>(d.output_multiplier)));
    if (d.final_logit_softcapping >= 0.0)
      b.AddKv(gguf_test::F32Kv(p + "final_logit_softcapping",
                               static_cast<float>(d.final_logit_softcapping)));
    if (d.input_embedding_scale >= 0.0)
      b.AddKv(gguf_test::F32Kv(p + "input_embedding_scale",
                               static_cast<float>(d.input_embedding_scale)));
  }

  BuiltGguf out;
  for (const TSpec& t : DraftTensors(d)) {
    const bool is_conv_or_sel =
        t.hf.find("_conv.") != std::string::npos ||
        t.hf.rfind("candidate_selector.", 0) == 0;
    if (!dflash2 && is_conv_or_sel) continue;
    const std::vector<float> src = Src(Numel(t.shape), t.seed);
    // ggml dims are the REVERSE of the torch shape (ne0 = fastest).
    std::vector<uint64_t> dims;
    for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it)
      dims.push_back(static_cast<uint64_t>(*it));
    if (!t.quantized || arm == Arm::kBf16) {
      if (!t.quantized) {
        b.AddTensor(t.gguf, dims, kGgmlF32, F32Bytes(src));
        out.expect[t.hf] = src;  // f32 -> bf16 at load; the value is unchanged
      } else {
        b.AddTensor(t.gguf, dims, kGgmlBf16, Bf16Bytes(src));
        out.expect[t.hf] = RoundTripBf16(src);
      }
    } else if (arm == Arm::kQ8_0) {
      const Encoded e = EncodeQ8_0(src);
      b.AddTensor(t.gguf, dims, kGgmlQ8_0, e.bytes);
      out.expect[t.hf] = e.expect;
      out.q8_by_hf[t.hf] = e.q8;
    } else {
      const Encoded e = EncodeQ4_K(src);
      b.AddTensor(t.gguf, dims, kGgmlQ4_K, e.bytes);
      out.expect[t.hf] = e.expect;
      out.q4k_by_hf[t.hf] = e.q4k;
    }
  }
  out.bytes = b.Build();
  return out;
}

// The same draft written as safetensors under the published HF names, bf16. The
// twin the bf16 GGUF arm must reproduce BIT-FOR-BIT.
class ScratchSafetensors {
 public:
  ScratchSafetensors(const Dims& d, bool dflash2) {
    static int counter = 0;
    dir_ = fs::temp_directory_path() /
           ("vllmcpp_df2gguf_st_" + std::to_string(counter++) + "_" +
            std::to_string(static_cast<long long>(::getpid())));
    fs::create_directories(dir_);
    json header = json::object();
    std::string payload;
    size_t offset = 0;
    for (const TSpec& t : DraftTensors(d)) {
      const bool is_conv_or_sel =
          t.hf.find("_conv.") != std::string::npos ||
          t.hf.rfind("candidate_selector.", 0) == 0;
      if (!dflash2 && is_conv_or_sel) continue;
      const std::string bytes = Bf16Bytes(Src(Numel(t.shape), t.seed));
      header[t.hf] = {{"dtype", "BF16"},
                      {"shape", t.shape},
                      {"data_offsets", json::array({offset, offset + bytes.size()})}};
      offset += bytes.size();
      payload += bytes;
    }
    const std::string hs = header.dump();
    std::ofstream out((dir_ / "model.safetensors").string(), std::ios::binary);
    const uint64_t hlen = hs.size();
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
    out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
  ~ScratchSafetensors() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  ScratchSafetensors(const ScratchSafetensors&) = delete;
  ScratchSafetensors& operator=(const ScratchSafetensors&) = delete;
  std::string shard() const { return (dir_ / "model.safetensors").string(); }

 private:
  fs::path dir_;
};

// The HfConfig a safetensors draft of this shape declares, so the two containers
// are loaded through the same body with the same numbers.
HfConfig SafetensorsConfig(const Dims& d, bool dflash2) {
  HfConfig c;
  c.model_type = "dflash";
  c.hidden_size = d.H;
  c.num_attention_heads = d.Hq;
  c.num_key_value_heads = d.Hkv;
  c.head_dim = d.Dh;
  c.rotary_dim = d.Dh;
  c.rope_theta = 1e7;
  c.intermediate_size = d.I;
  c.num_hidden_layers = d.layers;
  c.rms_norm_eps = 1e-6;
  c.sliding_window = 2048;
  c.vocab_size = d.vocab;
  c.layer_types = std::vector<std::string>(static_cast<size_t>(d.layers), "sliding_attention");
  c.raw = json::object();
  c.raw["block_size"] = d.block;
  json dcfg = json::object();
  dcfg["mask_token_id"] = d.vocab - 1;
  json ids = json::array();
  for (int64_t i = 0; i < d.taps_fc; ++i) ids.push_back(3 * i);
  dcfg["target_layer_ids"] = ids;
  if (dflash2) {
    c.raw["is_causal"] = false;
    dcfg["conv_kernel_size"] = d.conv_taps;
    dcfg["conv_group_size"] = d.conv_group;
    dcfg["selector_rank"] = d.rank;
    dcfg["selector_top_k"] = d.top_k;
  }
  c.raw["dflash_config"] = dcfg;
  return c;
}

// A GGUF file on disk, removed on scope exit.
using gguf_test::TempFile;

// Load a draft GGUF through the PRODUCTION pair: the config reader and the
// weight path. This is exactly the sequence `LoadDflashDraft` runs.
Qwen3DFlashWeights LoadGgufDraft(const std::string& path, HfConfig* config_out) {
  const vllm::GgufFile g = vllm::GgufFile::Open(path);
  const HfConfig c = vllm::MakeDflashGgufConfig(g);
  const json& dcfg = c.raw.at("dflash_config");
  const int64_t num_taps = static_cast<int64_t>(dcfg.at("target_layer_ids").size());
  const int32_t mask_id = dcfg.at("mask_token_id").get<int32_t>();
  Qwen3DFlashWeights w = vllm::LoadQwen3DFlashFromGguf(g, c, num_taps, mask_id);
  if (config_out != nullptr) *config_out = c;
  return w;
}

Qwen3DFlashWeights LoadSafetensorsDraft(const ScratchSafetensors& ck, const Dims& d,
                                        bool dflash2) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(ck.shard()));
  return vllm::LoadQwen3DFlash(shards, SafetensorsConfig(d, dflash2), d.taps_fc,
                               static_cast<int32_t>(d.vocab - 1));
}

// What `LoadDflashDraft` does after the container-specific load: the draft
// SHARES the target's embedding and head, and the conv's block is 1 + k.
void ShareTargetHeads(Qwen3DFlashWeights* w, const Dims& d, HfConfig* c) {
  const std::vector<float> emb = Src(d.vocab * d.H, 6.6);
  const std::vector<float> head = Src(d.vocab * d.H, 7.7);
  auto fill = [&](vllm::OwnedTensor* t, const std::vector<float>& v, bool nk) {
    t->dtype = vt::DType::kBF16;
    t->rank = 2;
    t->shape[0] = d.vocab;
    t->shape[1] = d.H;
    t->nk = nk;
    const std::string bytes = Bf16Bytes(v);
    t->bytes.resize(bytes.size());
    std::memcpy(t->bytes.data(), bytes.data(), bytes.size());
  };
  fill(&w->embed_tokens, emb, false);
  fill(&w->lm_head, head, true);
  w->draft_vocab_size = d.vocab;
  if (c->vocab_size == 0) c->vocab_size = d.vocab;
  if (w->IsDflash2()) w->conv_block_size = d.block;
}

// Every DFlash2 tensor of a loaded draft, keyed by its HF name, as raw bf16
// bit patterns. The comparison currency for every case below.
std::map<std::string, std::vector<uint16_t>> Bf16Of(const Qwen3DFlashWeights& w,
                                                    const Dims& d) {
  std::map<std::string, std::vector<uint16_t>> out;
  auto take = [&out](const std::string& name, const vllm::OwnedTensor& t) {
    std::vector<uint16_t> v(t.bytes.size() / 2);
    if (!v.empty()) std::memcpy(v.data(), t.bytes.data(), t.bytes.size());
    out[name] = std::move(v);
  };
  take("fc.weight", w.fc);
  take("hidden_norm.weight", w.hidden_norm);
  take("norm.weight", w.final_norm);
  for (int64_t l = 0; l < static_cast<int64_t>(w.layers.size()); ++l) {
    const std::string b = "layers." + std::to_string(l) + ".";
    const auto& L = w.layers[static_cast<size_t>(l)];
    take(b + "input_layernorm.weight", L.input_layernorm);
    take(b + "post_attention_layernorm.weight", L.post_attention_layernorm);
    take(b + "self_attn.qkv_proj", L.qkv_proj);
    take(b + "self_attn.o_proj.weight", L.o_proj);
    take(b + "self_attn.q_norm.weight", L.q_norm);
    take(b + "self_attn.k_norm.weight", L.k_norm);
    take(b + "mlp.gate_up_proj", L.gate_up_proj);
    take(b + "mlp.down_proj.weight", L.down_proj);
    if (w.IsDflash2()) {
      take(b + "attention_conv.base_kernel", L.attention_conv.base_kernel);
      take(b + "attention_conv.kernel_projection.weight", L.attention_conv.kernel_projection);
      take(b + "mlp_conv.base_kernel", L.mlp_conv.base_kernel);
      take(b + "mlp_conv.kernel_projection.weight", L.mlp_conv.kernel_projection);
    }
  }
  if (w.IsDflash2()) {
    take("candidate_selector.hidden_projection.weight", w.candidate_selector.hidden_projection);
    take("candidate_selector.predecessor_codebook", w.candidate_selector.predecessor_codebook);
    take("candidate_selector.successor_codebook", w.candidate_selector.successor_codebook);
  }
  (void)d;
  return out;
}

// The DFlash2-specific names, which is what this wave adds and what the
// difference legs below are measured over.
const std::vector<std::string>& Dflash2Names() {
  static const std::vector<std::string> names = {
      "layers.0.attention_conv.base_kernel",
      "layers.0.attention_conv.kernel_projection.weight",
      "layers.0.mlp_conv.base_kernel",
      "layers.0.mlp_conv.kernel_projection.weight",
      "layers.1.attention_conv.base_kernel",
      "layers.1.attention_conv.kernel_projection.weight",
      "layers.1.mlp_conv.base_kernel",
      "layers.1.mlp_conv.kernel_projection.weight",
      "candidate_selector.hidden_projection.weight",
      "candidate_selector.predecessor_codebook",
      "candidate_selector.successor_codebook",
  };
  return names;
}

std::vector<uint16_t> ExpectBf16(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = vt::F32ToBF16(v[i]);
  return out;
}

// One draft block through the context-free body, which is what the propose path
// and the runner both reach.
std::vector<float> BlockLogits(const Qwen3DFlashWeights& w, const HfConfig& c,
                               const Dims& d, std::vector<float>* hidden) {
  std::vector<int32_t> ids(static_cast<size_t>(d.block)), pos(static_cast<size_t>(d.block));
  for (int64_t i = 0; i < d.block; ++i) {
    ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 5 + 3) % c.vocab_size);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  const std::vector<int32_t> cu = {0, static_cast<int32_t>(d.block)};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  return Qwen3DFlashModel::ForwardBlockLogits(ids, pos, cu, w, c, q, nullptr, hidden);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. The metadata.
// ---------------------------------------------------------------------------
TEST_CASE("dflash2 gguf: the config reader carries the DFlash2 metadata into dflash_config") {
  // RED before W5: `MakeDflashGgufConfig` reads none of these keys, so
  // `LoadQwen3DFlash` sees a DFlash1 config and builds a DFlash1 draft out of a
  // DFlash2 checkpoint -- which is what the startup refusal existed to prevent.
  const Dims d;
  const TempFile f(BuildDraftGguf(d, Arm::kBf16, /*dflash2=*/true).bytes);
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const HfConfig c = vllm::MakeDflashGgufConfig(g);
  REQUIRE(c.raw.contains("dflash_config"));
  const json& dcfg = c.raw.at("dflash_config");

  CHECK(dcfg.contains("conv_kernel_size"));
  CHECK(dcfg.contains("conv_group_size"));
  CHECK(dcfg.contains("selector_rank"));
  CHECK(dcfg.contains("selector_top_k"));
  CHECK(dcfg.value("conv_kernel_size", -1) == d.conv_taps);
  CHECK(dcfg.value("conv_group_size", -1) == d.conv_group);
  CHECK(dcfg.value("selector_rank", -1) == d.rank);
  CHECK(dcfg.value("selector_top_k", -1) == d.top_k);

  // W1's `dflash.attention.causal` still resolves, now that the lane RUNS rather
  // than refuses: the published DFlash2 GGUF declares it false beside an
  // all-true sliding-window pattern, so the pattern alone would answer CAUSAL
  // for every layer and the drafter would lose acceptance with nothing to see.
  REQUIRE(c.raw.contains("is_causal"));
  CHECK(c.raw.at("is_causal").get<bool>() == false);
  const std::vector<vllm::Qwen3DFlashLayerAttnMode> modes =
      vllm::ResolveQwen3DFlashAttnModes(c);
  REQUIRE(static_cast<int64_t>(modes.size()) == d.layers);
  for (const vllm::Qwen3DFlashLayerAttnMode& m : modes) CHECK_FALSE(m.causal);

  // The DFlash1 contract is UNCHANGED: no DFlash2 key appears, and the vocab
  // stays 0 so the caller fills it from the target it shares a head with.
  const TempFile f1(BuildDraftGguf(d, Arm::kBf16, /*dflash2=*/false).bytes);
  const vllm::GgufFile g1 = vllm::GgufFile::Open(f1.path());
  const HfConfig c1 = vllm::MakeDflashGgufConfig(g1);
  const json& d1 = c1.raw.at("dflash_config");
  CHECK_FALSE(d1.contains("conv_kernel_size"));
  CHECK_FALSE(d1.contains("conv_group_size"));
  CHECK_FALSE(d1.contains("selector_rank"));
  CHECK_FALSE(d1.contains("selector_top_k"));
  CHECK_FALSE(c1.raw.contains("is_causal"));
  CHECK(c1.vocab_size == 0);
  CHECK(c.vocab_size == 0);
}

// ---------------------------------------------------------------------------
// 2. The name map, gated by construction.
// ---------------------------------------------------------------------------
TEST_CASE("dflash2 gguf: a bf16 GGUF draft loads BIT-IDENTICALLY to its safetensors twin") {
  // RED before W5: `LoadQwen3DFlashFromGguf` throws
  // "dflash gguf: no GGUF name for 'attention_conv.base_kernel'".
  //
  // This is the name map's gate, and it is a construction rather than a reading:
  // both files are written from the SAME source values under the two containers'
  // own names, so a GGUF name pointed at the wrong tensor, transposed, or
  // dropped moves the bytes. Nothing here restates the mapping.
  const Dims d;
  const TempFile gf(BuildDraftGguf(d, Arm::kBf16, /*dflash2=*/true).bytes);
  const ScratchSafetensors st(d, /*dflash2=*/true);

  HfConfig gc;
  const Qwen3DFlashWeights gw = LoadGgufDraft(gf.path(), &gc);
  const Qwen3DFlashWeights sw = LoadSafetensorsDraft(st, d, /*dflash2=*/true);
  REQUIRE(gw.IsDflash2());
  REQUIRE(sw.IsDflash2());
  CHECK(gw.conv_taps == d.conv_taps);
  CHECK(gw.conv_group_size == d.conv_group);
  CHECK(gw.candidate_selector.rank == d.rank);
  CHECK(gw.candidate_selector.top_k == d.top_k);

  const auto gm = Bf16Of(gw, d);
  const auto sm = Bf16Of(sw, d);
  REQUIRE(gm.size() == sm.size());
  size_t compared = 0;
  for (const auto& kv : gm) {
    const auto it = sm.find(kv.first);
    REQUIRE(it != sm.end());
    INFO("tensor ", kv.first, " gguf ", kv.second.size(), " st ", it->second.size());
    REQUIRE(kv.second.size() == it->second.size());
    CHECK(kv.second == it->second);
    compared += kv.second.size();
  }
  // The instrument's own precondition: a comparison over an empty set is
  // satisfied by everything, so the extent is asserted EXACTLY rather than
  // bounded below, and the two numbers are the ones the records quote. MEASURED
  // at this fixture: the DFlash2 draft writes 36 GGUF tensors, of which the
  // loader's q/k/v and gate/up merges leave 30 comparable slots holding
  // 1 120 000 bf16 elements, 11 of the slots DFlash2's own. (25 is the DFLASH1
  // fixture's tensor count and was quoted here in error through W5, #1314 F6.)
  INFO("bf16 elements compared across the two containers: ", compared,
       " over ", gm.size(), " tensors");
  CHECK(gm.size() == 30);
  CHECK(compared == 1120000);
  for (const std::string& n : Dflash2Names()) {
    REQUIRE(gm.count(n) == 1);
    CHECK_FALSE(gm.at(n).empty());
  }
}

// ---------------------------------------------------------------------------
// 3-4. The quantized arms, with the lower bound.
// ---------------------------------------------------------------------------
namespace {

// L1 + L2 + L3 for one quantized arm, over every tensor the arm encodes.
void CheckQuantArm(Arm arm, uint32_t ggml_type, int64_t block_elems,
                   int64_t block_bytes) {
  const Dims d;
  const BuiltGguf built = BuildDraftGguf(d, arm, /*dflash2=*/true);
  const TempFile qf(built.bytes);
  const TempFile bf(BuildDraftGguf(d, Arm::kBf16, /*dflash2=*/true).bytes);

  // --- L1: the BYTES. Read off the tensor table the loader itself reads.
  const vllm::GgufFile g = vllm::GgufFile::Open(qf.path());
  int64_t quantized_tensors = 0, quantized_elems = 0, quantized_bytes = 0;
  for (const TSpec& t : DraftTensors(d)) {
    if (!t.quantized) {
      const vllm::GgufTensorInfo& info = g.Get(t.gguf);
      // The published arms keep norms and BOTH conv base kernels dense; a build
      // that quantized them would change what this file gates.
      CHECK(info.ggml_type == kGgmlF32);
      continue;
    }
    const vllm::GgufTensorInfo& info = g.Get(t.gguf);
    const int64_t numel = Numel(t.shape);
    INFO("tensor ", t.gguf, " type ", info.ggml_type, " bytes ", info.nbytes,
         " numel ", numel);
    CHECK(info.ggml_type == ggml_type);
    CHECK(static_cast<int64_t>(info.nbytes) == numel / block_elems * block_bytes);
    // The lower bound proper: a tensor that had quietly become bf16 would need
    // two bytes per element, and this arm stores strictly fewer.
    CHECK(static_cast<int64_t>(info.nbytes) < 2 * numel);
    ++quantized_tensors;
    quantized_elems += numel;
    quantized_bytes += static_cast<int64_t>(info.nbytes);
  }
  INFO(std::string(ArmName(arm)), ": ", quantized_tensors, " quantized tensors, ",
       quantized_elems, " elements in ", quantized_bytes, " bytes");
  REQUIRE(quantized_tensors > 0);
  CHECK(quantized_bytes < quantized_elems * 2);  // strictly denser than bf16

  // --- L2: the VALUES, against this file's own encoder.
  HfConfig qc, bc;
  const Qwen3DFlashWeights qw = LoadGgufDraft(qf.path(), &qc);
  const Qwen3DFlashWeights bw = LoadGgufDraft(bf.path(), &bc);
  REQUIRE(qw.IsDflash2());
  const auto qm = Bf16Of(qw, d);
  const auto bm = Bf16Of(bw, d);
  int64_t value_assertions = 0;
  for (const std::string& n : Dflash2Names()) {
    const auto e = built.expect.find(n);
    REQUIRE(e != built.expect.end());
    const std::vector<uint16_t> want = ExpectBf16(e->second);
    REQUIRE(qm.count(n) == 1);
    INFO("tensor ", n, " got ", qm.at(n).size(), " want ", want.size());
    REQUIRE(qm.at(n).size() == want.size());
    CHECK(qm.at(n) == want);
    value_assertions += static_cast<int64_t>(want.size());
  }
  INFO(std::string(ArmName(arm)), ": values checked over ", value_assertions, " elements");
  CHECK(value_assertions > 0);
  // The Q4_K arm's own preconditions, ONE PER PACKED FIELD. L2 above compares
  // values; these say that the values it compared actually drove each field
  // across its width, which is the difference between a bound and a bound that
  // cannot see half of what it names. Both halves of `get_scale_min_k4` are
  // covered, because `j < 4` reads the scale WHOLE and `j >= 4` assembles it
  // from two bytes, and a fixture that only ever exercised one of the two would
  // pass every assertion here with the other deleted.
  if (arm == Arm::kQ4_K) {
    // Over the COMPARED tensors only, not over every Q4_K tensor in the file.
    Q4KCoverage cov;
    int cov_tensors = 0;
    for (const std::string& n : Dflash2Names()) {
      const auto it = built.q4k_by_hf.find(n);
      if (it == built.q4k_by_hf.end()) continue;  // F32 in every arm
      cov.Merge(it->second);
      ++cov_tensors;
    }
    INFO("Q4_K coverage merged over ", cov_tensors, " COMPARED tensors of ",
         built.q4k_by_hf.size(), " encoded");
    REQUIRE(cov_tensors > 0);
    for (int half = 0; half < 2; ++half) {
      INFO("Q4_K sub-blocks ", half == 0 ? "0..3 (whole field)" : "4..7 (split field)",
           ": ", cov.scale_above_15[half], " scales above 15, ",
           cov.min_above_15[half], " mins above 15, scale bits ",
           cov.scale_bits[half], ", min bits ", cov.min_bits[half]);
      CHECK(cov.scale_above_15[half] > 0);
      CHECK(cov.min_above_15[half] > 0);
      CHECK(cov.scale_bits[half] == 63);
      CHECK(cov.min_bits[half] == 63);
    }
    // ...and the 4-bit quant itself, in BOTH nibble positions. Driving the
    // scales up costs resolution, so this is the field the repair above could
    // have broken while every other assertion stayed green.
    INFO("Q4_K nibble bits: low ", cov.nibble_bits[0], " high ", cov.nibble_bits[1]);
    CHECK(cov.nibble_bits[0] == 15);
    CHECK(cov.nibble_bits[1] == 15);
  }

  // The Q8_0 arm's own precondition, on the field ITS blocks carry: the fp16
  // `d` in each block header. Asserted PER COMPARED TENSOR rather than summed,
  // because the defect it exists to expose -- a `d` read once and reused for a
  // whole tensor -- is invisible unless the tensor being compared itself holds
  // blocks at more than one `d`. A total would be satisfied by two tensors that
  // each carry only one.
  if (arm == Arm::kQ8_0) {
    static_assert(kQ8Dfp16[0] != kQ8Dfp16[1],
                  "the two block scales must differ, or the alternation counts "
                  "nothing and the per-block read is untested");
    int q8_tensors = 0;
    for (const std::string& n : Dflash2Names()) {
      const auto it = built.q8_by_hf.find(n);
      if (it == built.q8_by_hf.end()) continue;  // F32 in every arm
      INFO("Q8_0 tensor ", n, ": ", it->second.blocks_at[0], " blocks at 2^-8, ",
           it->second.blocks_at[1], " blocks at 2^-7");
      CHECK(it->second.blocks_at[0] > 0);
      CHECK(it->second.blocks_at[1] > 0);
      ++q8_tensors;
    }
    INFO("Q8_0 block-scale coverage checked over ", q8_tensors,
         " COMPARED tensors of ", built.q8_by_hf.size(), " encoded");
    REQUIRE(q8_tensors > 0);
  }

  // --- L3: DIFFERENCE from the bf16 arm, which is encoded from the SAME source.
  // A loader that produced the dense answer for a quantized file is caught here
  // even if L1 and L2 were both somehow satisfied.
  int differing = 0, checked = 0;
  for (const std::string& n : Dflash2Names()) {
    if (n.find("base_kernel") != std::string::npos) continue;  // F32 in every arm
    ++checked;
    if (qm.at(n) != bm.at(n)) ++differing;
  }
  INFO(std::string(ArmName(arm)), ": ", differing, " of ", checked,
       " quantized DFlash2 tensors differ from the bf16 arm");
  CHECK(checked > 0);
  CHECK(differing == checked);
  // ...and the F32 tensors, which no arm quantizes, must be IDENTICAL. Without
  // this the case above is satisfied by a loader that garbled everything.
  for (const std::string& n : Dflash2Names()) {
    if (n.find("base_kernel") == std::string::npos) continue;
    INFO("dense tensor ", n);
    CHECK(qm.at(n) == bm.at(n));
  }

  // --- L3 again, end to end: the BLOCK LOGITS move. A quant arm that silently
  // loaded bf16 weights would answer identically here, and a token gate over a
  // lossless verify could never see it.
  Qwen3DFlashWeights qq = qw, bb = bw;
  ShareTargetHeads(&qq, d, &qc);
  ShareTargetHeads(&bb, d, &bc);
  std::vector<float> qh, bh;
  const std::vector<float> ql = BlockLogits(qq, qc, d, &qh);
  const std::vector<float> bl = BlockLogits(bb, bc, d, &bh);
  REQUIRE(ql.size() == bl.size());
  REQUIRE(!ql.empty());
  int moved = 0;
  for (size_t i = 0; i < ql.size(); ++i)
    if (ql[i] != bl[i]) ++moved;
  INFO(std::string(ArmName(arm)), ": ", moved, " of ", ql.size(), " block logits differ from bf16");
  CHECK(moved > 0);
  for (float v : ql) CHECK(std::isfinite(v));
}

}  // namespace

TEST_CASE("dflash2 gguf: the Q8_0 arm decodes the BLOCKS and cannot be the bf16 bytes") {
  CheckQuantArm(Arm::kQ8_0, kGgmlQ8_0, /*block_elems=*/32, /*block_bytes=*/34);
}

TEST_CASE("dflash2 gguf: the Q4_K arm decodes the PACKED 6-bit scales and cannot be the bf16 bytes") {
  CheckQuantArm(Arm::kQ4_K, kGgmlQ4_K, /*block_elems=*/256, /*block_bytes=*/144);
}

// ---------------------------------------------------------------------------
// 5. It DRAFTS, on every arm.
// ---------------------------------------------------------------------------
TEST_CASE("dflash2 gguf: every published arm DRAFTS through the DFlash2 propose") {
  // The end of the mechanism, entered from a GGUF rather than from safetensors.
  // "It returned tokens" is NOT the assertion: deleting the DFlash2 branch also
  // returns tokens -- the DFlash1 per-slot argmax -- and they are well formed,
  // the verify is lossless, and only acceptance falls. So each arm is required
  // to draft from the SELECTOR's own candidate set, which is 4 wide over a
  // vocabulary of 256.
  const Dims d;
  const int k = static_cast<int>(d.block - 1);
  for (Arm arm : {Arm::kBf16, Arm::kQ8_0, Arm::kQ4_K}) {
    INFO("arm ", std::string(ArmName(arm)));
    const TempFile f(BuildDraftGguf(d, arm, /*dflash2=*/true).bytes);
    HfConfig c;
    Qwen3DFlashWeights w = LoadGgufDraft(f.path(), &c);
    ShareTargetHeads(&w, d, &c);
    REQUIRE(w.IsDflash2());
    REQUIRE(w.conv_block_size == d.block);

    std::vector<int32_t> ids(static_cast<size_t>(d.block)), pos(static_cast<size_t>(d.block));
    for (int64_t i = 0; i < d.block; ++i) {
      ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 5 + 3) % c.vocab_size);
      pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const std::vector<int32_t> cu = {0, static_cast<int32_t>(d.block)};
    const std::vector<int32_t> ctx_cu = {0, 0};
    vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

    const vllm::v1::DflashProposeResult got = vllm::v1::DflashProposeBlock(
        w, c, {}, {}, ctx_cu, ids, pos, cu, /*num_reqs=*/1, k, q);
    REQUIRE(got.draft_token_ids.size() == 1);
    REQUIRE(got.draft_token_ids[0].size() == static_cast<size_t>(k));

    std::vector<float> hidden;
    const std::vector<float> logits = BlockLogits(w, c, d, &hidden);
    const std::vector<int32_t> anchors = {ids[0]};
    const vllm::v1::Dflash2ProposeState sel = vllm::v1::Dflash2SelectCandidates(
        logits, hidden, anchors, /*num_reqs=*/1, k, w, c, q);
    REQUIRE(sel.top_k == d.top_k);
    // The instrument's precondition: the candidate set must be a PROPER subset
    // of the vocabulary, or membership is satisfied by every token.
    REQUIRE(sel.top_k < c.vocab_size);
    for (int step = 0; step < k; ++step) {
      bool member = false;
      for (int64_t cand = 0; cand < sel.top_k; ++cand)
        if (sel.candidates.ids[static_cast<size_t>(step * sel.top_k + cand)] ==
            got.draft_token_ids[0][static_cast<size_t>(step)])
          member = true;
      INFO("step ", step, " drafted ", got.draft_token_ids[0][static_cast<size_t>(step)]);
      CHECK(member);
    }
    // ...and it is not the DFlash1 argmax's draft over the same block logits,
    // which is the fallback the whole lane exists to refuse.
    //
    // MEASURED OVER A SWEEP rather than asserted on one block, and the counts
    // are logged rather than assumed. The walk and the argmax CAN agree on a
    // given block -- the walk's best path often runs through the per-slot
    // maximum -- so a single block that happened to coincide would fail this for
    // a reason that says nothing about the arm. What must be true is that the
    // two answers are DISTINGUISHABLE on this fixture at all; a fixture on which
    // they never differ cannot tell a working walk from a deleted one, and that
    // is a defect in the instrument.
    int blocks_run = 0, blocks_differing = 0, slots_differing = 0;
    for (int shift = 0; shift < 6; ++shift) {
      std::vector<int32_t> sids(static_cast<size_t>(d.block));
      for (int64_t i = 0; i < d.block; ++i)
        sids[static_cast<size_t>(i)] =
            static_cast<int32_t>((i * 7 + shift * 11 + 3) % c.vocab_size);
      const vllm::v1::DflashProposeResult a = vllm::v1::DflashProposeBlock(
          w, c, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
      std::vector<float> h2;
      const std::vector<float> l2 =
          Qwen3DFlashModel::ForwardBlockLogits(sids, pos, cu, w, c, q, nullptr, &h2);
      const std::vector<std::vector<int32_t>> argmax = vllm::v1::SampleDflashBlockDrafts(
          l2, /*num_reqs=*/1, k, w.draft_vocab_size);
      REQUIRE(argmax.size() == 1);
      ++blocks_run;
      if (argmax[0] != a.draft_token_ids[0]) ++blocks_differing;
      for (int step = 0; step < k; ++step)
        if (argmax[0][static_cast<size_t>(step)] !=
            a.draft_token_ids[0][static_cast<size_t>(step)])
          ++slots_differing;
      // The instrument's own precondition, per block: the same arm against
      // itself must never differ, or the counter above measures nondeterminism.
      const vllm::v1::DflashProposeResult again = vllm::v1::DflashProposeBlock(
          w, c, {}, {}, ctx_cu, sids, pos, cu, /*num_reqs=*/1, k, q);
      CHECK(again.draft_token_ids[0] == a.draft_token_ids[0]);
    }
    INFO("blocks where the walk differs from the DFlash1 argmax: ", blocks_differing,
         " of ", blocks_run, "; differing slots ", slots_differing);
    CHECK(blocks_run == 6);
    CHECK(blocks_differing > 0);
  }
}

// ---------------------------------------------------------------------------
// 5a. D9's OUTPUT SCALARS, on this container.
// ---------------------------------------------------------------------------
TEST_CASE("dflash2 gguf: the OUTPUT SCALARS are read from the GGUF, and default when absent") {
  // `## Risks/decisions` D9. Both scalars are applied to the candidate VALUES
  // before the selector scores them, so a wrong one reorders the top-K and moves
  // acceptance without raising anything. NEITHER published DFlash2 GGUF declares
  // either, so a gate built from those files alone measures the DEFAULT path and
  // reports it as coverage -- which is the whole reason this case declares them.
  //
  // The values are `z-lab/Muse-Glimmer-30B-DFlash2`'s, the checkpoint that SETS
  // them in safetensors (#1327). The GGUF spelling is the measured
  // `dflash.<hf key>` convention this reader uses for the other five keys.
  Dims declared;
  declared.output_multiplier = 0.19611613513818404;
  declared.final_logit_softcapping = 20.0;
  const TempFile df(BuildDraftGguf(declared, Arm::kBf16, /*dflash2=*/true).bytes);
  HfConfig dc;
  const Qwen3DFlashWeights dw = LoadGgufDraft(df.path(), &dc);
  REQUIRE(dw.IsDflash2());
  CHECK(dw.candidate_selector.output_multiplier ==
        doctest::Approx(static_cast<float>(declared.output_multiplier)));
  CHECK(dw.candidate_selector.final_logit_softcapping == doctest::Approx(20.0f));

  // ...and ABSENT is upstream's own default rather than zero, which is the arm
  // both published files take. Without this half the case above cannot tell a
  // reader that works from one that always returns the declared value.
  const Dims absent;
  const TempFile af(BuildDraftGguf(absent, Arm::kBf16, /*dflash2=*/true).bytes);
  HfConfig ac;
  const Qwen3DFlashWeights aw = LoadGgufDraft(af.path(), &ac);
  CHECK(aw.candidate_selector.output_multiplier == doctest::Approx(1.0f));
  CHECK(aw.candidate_selector.final_logit_softcapping == doctest::Approx(0.0f));
  CHECK_FALSE(ac.raw.at("dflash_config").contains("output_multiplier"));
  CHECK_FALSE(ac.raw.at("dflash_config").contains("final_logit_softcapping"));

  // `input_embedding_scale` is REFUSED BY NAME on this container too, because
  // this engine does not apply it and ignoring it would run a quietly different
  // model (spec `## Owed` O9). A value equal to upstream's default is upstream's
  // own no-op and must NOT be refused -- without that half the refusal could be
  // "refuse the key" rather than "refuse the behaviour".
  Dims scaled;
  scaled.input_embedding_scale = 2.0;
  const TempFile sf(BuildDraftGguf(scaled, Arm::kBf16, /*dflash2=*/true).bytes);
  CHECK_THROWS_WITH_AS(LoadGgufDraft(sf.path(), nullptr),
                       doctest::Contains("input_embedding_scale"),
                       std::runtime_error);
  Dims unit;
  unit.input_embedding_scale = 1.0;
  const TempFile uf(BuildDraftGguf(unit, Arm::kBf16, /*dflash2=*/true).bytes);
  CHECK_NOTHROW(LoadGgufDraft(uf.path(), nullptr));
}

// ---------------------------------------------------------------------------
// 6a. The pairing this container makes possible to get wrong.
// ---------------------------------------------------------------------------
TEST_CASE("dflash2 gguf: a codebook that does not span the TARGET vocabulary is REFUSED") {
  // W5 is the first wave where the two numbers can disagree. The selector's
  // codebooks are indexed by TARGET token id -- `draft_vocab_size`, which the
  // loader takes from the target's head -- while their extent comes from the
  // DRAFT. On the safetensors arm both trace back to one `config.json` and could
  // not differ; a GGUF draft declares no vocabulary at all and is sized from its
  // own tokenizer, so a drafter pointed at the wrong target reaches the selector
  // with a codebook SHORTER than the ids that index it. That is an out-of-range
  // read of a 127 MB tensor on the published shapes, not a wrong answer.
  //
  // Matched on the MESSAGE and the type, not with a bare CHECK_THROWS: every
  // guard on this path throws `std::runtime_error`, so a bare form is satisfied
  // by the neighbouring shape checks and would pass with this one deleted.
  const Dims d;
  const TempFile f(BuildDraftGguf(d, Arm::kBf16, /*dflash2=*/true).bytes);
  HfConfig c;
  Qwen3DFlashWeights w = LoadGgufDraft(f.path(), &c);
  ShareTargetHeads(&w, d, &c);
  const int k = static_cast<int>(d.block - 1);
  const int64_t nq = d.block;
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<int32_t> anchors = {0};

  // The MATCHED pairing runs, which is the instrument's own precondition: a case
  // that threw for both widths would prove nothing about the guard.
  {
    const std::vector<float> logits(static_cast<size_t>(nq * d.vocab), 0.5f);
    const std::vector<float> hidden(static_cast<size_t>(nq * d.H), 0.25f);
    CHECK_NOTHROW(vllm::v1::Dflash2SelectCandidates(logits, hidden, anchors,
                                                    /*num_reqs=*/1, k, w, c, q));
  }
  // A target one row NARROWER than the codebooks. One row is deliberate: the
  // failure this refuses is a mispaired checkpoint, and the interesting case is
  // the near miss rather than an obviously different model.
  {
    Qwen3DFlashWeights narrow = w;
    narrow.draft_vocab_size = d.vocab - 1;
    HfConfig nc = c;
    nc.vocab_size = d.vocab - 1;
    const std::vector<float> logits(static_cast<size_t>(nq * (d.vocab - 1)), 0.5f);
    const std::vector<float> hidden(static_cast<size_t>(nq * d.H), 0.25f);
    CHECK_THROWS_WITH_AS(
        vllm::v1::Dflash2SelectCandidates(logits, hidden, anchors, /*num_reqs=*/1, k,
                                          narrow, nc, q),
        doctest::Contains("candidate-selector codebooks"), std::runtime_error);
    CHECK_THROWS_WITH_AS(
        vllm::v1::Dflash2SelectCandidates(logits, hidden, anchors, /*num_reqs=*/1, k,
                                          narrow, nc, q),
        doctest::Contains("not trained against this target"), std::runtime_error);
  }
}

// ---------------------------------------------------------------------------
// 6. The lane that ships today is unaffected.
// ---------------------------------------------------------------------------
TEST_CASE("dflash1 gguf: a draft with no DFlash2 metadata is UNCHANGED") {
  // The regression guard on the axis that ships. A DFlash1 GGUF carries none of
  // the DFlash2 keys and none of the conv or selector tensors, and must still
  // load exactly as it did -- bit-for-bit against its own safetensors twin,
  // which is the comparison that would move if the DFlash2 branch leaked into
  // the DFlash1 path.
  const Dims d;
  const TempFile gf(BuildDraftGguf(d, Arm::kQ4_K, /*dflash2=*/false).bytes);
  const ScratchSafetensors st(d, /*dflash2=*/false);
  HfConfig gc;
  const Qwen3DFlashWeights gw = LoadGgufDraft(gf.path(), &gc);
  const Qwen3DFlashWeights sw = LoadSafetensorsDraft(st, d, /*dflash2=*/false);
  CHECK_FALSE(gw.IsDflash2());
  CHECK(gw.conv_taps == 0);
  CHECK(gw.candidate_selector.Empty());
  CHECK(gw.layers.size() == static_cast<size_t>(d.layers));
  for (const auto& L : gw.layers) {
    CHECK(L.attention_conv.Empty());
    CHECK(L.mlp_conv.Empty());
  }
  // The DFlash1 tensors are the SHAPE the quantized arm decodes; a bit-identical
  // comparison against safetensors would be false here because the DFlash1 twin
  // is bf16 and this file is Q4_K. So the shapes and the finiteness are what is
  // asserted, plus the absence of every DFlash2 slot above.
  CHECK(sw.layers.size() == gw.layers.size());
  CHECK(gw.fc.bytes.size() == sw.fc.bytes.size());
  CHECK(gw.hidden_norm.bytes == sw.hidden_norm.bytes);  // F32 in both -> identical
  CHECK(gw.final_norm.bytes == sw.final_norm.bytes);
}

// ---------------------------------------------------------------------------
// 7. The published artifacts.
// ---------------------------------------------------------------------------
TEST_CASE("REAL published DFlash2 GGUF drafters carry the names and types this arm reads") {
  // ASSET-GATED, and it REPORTS what it did rather than skipping in silence: a
  // suite whose cases all return quietly prints SUCCESS with zero assertions,
  // which is issue #1382 on the sibling suite. The synthetic cases above are the
  // gate; this one holds the fixture against the artifacts it was written from,
  // which is the failure a hand-built fixture cannot see.
  //
  //   VLLM_DFLASH2_GGUF_MODEL -> any file of z-lab/Qwen3.8-27B-DFlash2-GGUF
  //                              @ 57ab3265056d4024870b0621cfc2c127537020ed.
  //                              Every sibling *.gguf in its directory is read
  //                              too, so one variable covers all three arms.
  //   VLLM_DFLASH_GGUF_MODEL  -> a DFlash1 drafter GGUF, asserted to carry NONE
  //                              of the DFlash2 metadata.
  //
  // Only the HEADER of each file is read -- the KV block and the tensor table --
  // so this costs kilobytes rather than the 7 GB the three arms weigh.
  const char* dflash2 = std::getenv("VLLM_DFLASH2_GGUF_MODEL");
  const char* dflash1 = std::getenv("VLLM_DFLASH_GGUF_MODEL");
  // REPORTED PER VARIABLE, not once for the pair. The combined form said
  // nothing whenever EITHER was set, so a run with only `VLLM_DFLASH2_GGUF_MODEL`
  // exercised the DFlash2 half, printed a line about reading three arms, and
  // left the DFlash1 regression half silently unexercised -- the #1382 shape one
  // level in, since the reader sees a loud line and infers the whole case ran.
  if (dflash2 == nullptr) {
    MESSAGE("asset-gated: VLLM_DFLASH2_GGUF_MODEL unset; 0 published-DFlash2 "
            "assertions ran");
  }
  if (dflash1 == nullptr) {
    MESSAGE("asset-gated: VLLM_DFLASH_GGUF_MODEL unset; 0 published-DFlash1 "
            "regression assertions ran");
  }
  if (dflash2 != nullptr) {
    std::vector<std::string> files;
    std::error_code ec;
    const fs::path first(dflash2);
    for (const fs::directory_entry& de : fs::directory_iterator(first.parent_path(), ec))
      if (de.path().extension() == ".gguf") files.push_back(de.path().string());
    if (files.empty()) files.push_back(dflash2);
    std::sort(files.begin(), files.end());
    MESSAGE("reading ", files.size(), " published DFlash2 GGUF arm(s)");
    for (const std::string& path : files) {
      INFO("file ", path);
      const vllm::GgufFile g = vllm::GgufFile::Open(path);
      std::string matched;
      REQUIRE(vllm::IsDflash2Gguf(g, &matched));
      const HfConfig c = vllm::MakeDflashGgufConfig(g);
      const json& dcfg = c.raw.at("dflash_config");
      // The four keys this wave started reading, at the published values.
      CHECK(dcfg.value("conv_kernel_size", -1) == 2);
      CHECK(dcfg.value("conv_group_size", -1) == 16);
      CHECK(dcfg.value("selector_rank", -1) == 256);
      CHECK(dcfg.value("selector_top_k", -1) == 16);
      REQUIRE(c.raw.contains("is_causal"));
      CHECK(c.raw.at("is_causal").get<bool>() == false);
      CHECK(c.hidden_size == 5120);
      CHECK(c.num_hidden_layers == 5);
      const int64_t groups = c.hidden_size / dcfg.at("conv_group_size").get<int64_t>();
      const int64_t proj_out = 2 * dcfg.at("conv_kernel_size").get<int64_t>() * groups;
      const int64_t rank = dcfg.at("selector_rank").get<int64_t>();
      // Every tensor name and shape the weight path depends on, at the real
      // geometry. ggml dims are the reverse of the torch shape.
      auto want = [&](const std::string& name, const std::vector<int64_t>& ggml_dims) {
        INFO("tensor ", name);
        const vllm::GgufTensorInfo& t = g.Get(name);  // throws if absent
        std::vector<int64_t> got_ggml(t.shape.rbegin(), t.shape.rend());
        CHECK(got_ggml == ggml_dims);
        // A DFlash2 tensor is stored dense (F32 for the conv bases) or in one of
        // the two encodings this arm decodes. A published re-quant that moved
        // one of them to a third encoding is a change this gate must report.
        //
        // The FILE is mixed and the arm name understates it: measured on
        // `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` on 2026-08-21, 32 F32 / 45 Q4_K /
        // FOUR Q6_K (`blk.{2,4}.{attn_v,ffn_down}.weight`, llama.cpp's usual
        // `_M` mixture). None of the four is a DFlash2 tensor, which is why this
        // list stays at two encodings; Q6_K reaches the shared `DequantQ6_K`
        // and is gated by `tests/vllm/test_gguf_dequant.cpp`, not here
        // (#1314 F5).
        CHECK((t.ggml_type == kGgmlF32 || t.ggml_type == kGgmlBf16 ||
               t.ggml_type == kGgmlQ8_0 || t.ggml_type == kGgmlQ4_K));
      };
      for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        want(b + "attn_conv_base", {c.hidden_size, 2, 2});
        want(b + "attn_conv_proj.weight", {c.hidden_size, proj_out});
        want(b + "ffn_conv_base", {c.hidden_size, 2, 2});
        want(b + "ffn_conv_proj.weight", {c.hidden_size, proj_out});
      }
      want("selector_hidden.weight", {c.hidden_size, rank});
      want("selector_predecessor.weight", {rank, 248320});
      want("selector_successor.weight", {rank, 248320});
      // The conv BASES are dense in every published arm, which is what lets the
      // synthetic fixture above keep them F32.
      for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        CHECK(g.Get(b + "attn_conv_base").ggml_type == kGgmlF32);
        CHECK(g.Get(b + "ffn_conv_base").ggml_type == kGgmlF32);
      }
    }
  }
  if (dflash1 != nullptr) {
    INFO("dflash1 file ", dflash1);
    const vllm::GgufFile g = vllm::GgufFile::Open(dflash1);
    std::string matched;
    CHECK_FALSE(vllm::IsDflash2Gguf(g, &matched));
    const HfConfig c = vllm::MakeDflashGgufConfig(g);
    const json& dcfg = c.raw.at("dflash_config");
    CHECK_FALSE(dcfg.contains("conv_kernel_size"));
    CHECK_FALSE(dcfg.contains("selector_rank"));
    CHECK_FALSE(c.raw.contains("is_causal"));
  }
}
