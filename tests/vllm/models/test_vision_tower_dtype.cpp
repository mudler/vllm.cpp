// #1359 — the vision tower's HOST STORAGE DTYPE, gated in bytes.
//
// Spec: `.agents/specs/vision-tower-dtype-polarity.md`. Every vision tower in
// this tree is read from an all-BF16 checkpoint and every one of them narrows
// back to bf16 before its first GEMM, so host f32 storage is a carrier that
// costs exactly 2x the checkpoint's bytes and buys nothing.
//
// WHY THIS SUITE EXISTS AT ALL. A token gate cannot see a dtype that is too
// wide: the tokens match, the tower goldens pass, and the path moves twice the
// bytes (AGENTS.md, "Inherit vLLM defaults"). The only instrument that can see
// it is one that counts bytes, so that is what every case below does — the
// `scripts/mm/tower_skip_rss.sh` peak-RSS gate at unit scale, over the
// production loader `LoadQwen3VLVisionWeights`, with no host or checkpoint
// needed. ONE loader, not two: an earlier draft of this suite also drove the
// Muse Glimmer loader, and that half was reverted with the rest of the Muse
// Glimmer change (#2166 owns it). Gemma-4's tower loader is not covered here
// either, and its own reason is different — nothing in production calls it
// (#2173).
//
// THE MEASUREMENT, NOT A RESTATEMENT OF THE TYPE. `StoreBytes` and `Bits` are
// written generically so that every case compiles against EITHER storage type.
// A helper that spelled `uint16_t` would make the resident-bytes case true by
// construction and the bit-identity case unwritable on the pre-fix tree, and
// the pre-fix red is the whole evidence that the case measures anything.
//
// Upstream anchor (parity pin 5559679229bc): `qwen3_vl.py:633-634` defines
// `Qwen3_VisionTransformer.dtype` AS `patch_embed.proj.weight.dtype`, and
// `base_loader.py:53` wraps construction in `set_default_torch_dtype(
// model_config.dtype)`. The tower has no dtype of its own upstream; it is
// whatever the checkpoint loaded as. Ours must be too.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "muse_glimmer_tiny_fixture.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vt/dtype.h"

namespace {

using muse_glimmer_tiny::Bf16;
using muse_glimmer_tiny::BuildSt;
using muse_glimmer_tiny::Fx;
using muse_glimmer_tiny::TempFile;

// Resident bytes of one host weight store, whatever its element type.
template <typename V>
size_t StoreBytes(const V& v) {
  return v.size() * sizeof(typename V::value_type);
}

// The bf16 bit pattern a stored weight holds, whatever its element type. Two
// overloads, so the value cases below assert the SAME bits against f32 storage
// and against bf16 storage. That is the §3.1 claim stated as a test: on an
// all-BF16 checkpoint the widen/narrow round trip is the identity, so the bits
// a consumer eventually uploads must not move when the storage narrows.
uint16_t Bits(float f) { return vt::F32ToBF16(f); }
uint16_t Bits(uint16_t b) { return b; }

// The bf16 bits `muse_glimmer_tiny::Bf16` wrote at element `i` of the tensor
// built with `seed`. The fixture's own generator, read back.
uint16_t FixtureBits(uint32_t seed, int64_t i) {
  return vt::F32ToBF16(muse_glimmer_tiny::Val(seed, i));
}

// ── the Qwen3-VL vehicle ────────────────────────────────────────────────────
//
// The smallest checkpoint `LoadQwen3VLVisionWeights` accepts: one block, one
// deepstack merger, and geometry small enough that the byte totals below are
// checkable by hand. `num_position_embeddings` is a perfect square because
// `num_grid_per_side()` takes its integer square root.
namespace qwen3vl_tiny {

constexpr int64_t kHidden = 8, kHeads = 2, kInter = 12, kOut = 6;
constexpr int64_t kPatch = 2, kTemporal = 2, kChannels = 3, kPosEmb = 16;

inline vllm::multimodal::Qwen3VLVisionConfig Config() {
  vllm::multimodal::Qwen3VLVisionConfig c;
  c.hidden_size = kHidden;
  c.num_heads = kHeads;
  c.depth = 1;
  c.intermediate_size = kInter;
  c.out_hidden_size = kOut;
  c.patch_size = kPatch;
  c.temporal_patch_size = kTemporal;
  c.spatial_merge_size = 2;
  c.num_position_embeddings = kPosEmb;
  c.in_channels = kChannels;
  c.deepstack_visual_indexes = {0};
  return c;
}

// Every `model.visual.*` tensor the loader above reaches for, in the real
// on-disk spelling (`qwen3_vl.cpp`, `LoadQwen3VLVisionWeights`).
inline std::vector<Fx> Tensors() {
  const int64_t patch_dim = kChannels * kTemporal * kPatch * kPatch;
  const int64_t ctx4 = kHidden * 4;  // spatial_merge_size^2 * hidden
  const std::string V = "model.visual.";
  std::vector<Fx> t;
  uint32_t s = 101;
  t.push_back(Bf16(V + "patch_embed.proj.weight", {kHidden, patch_dim}, s++));
  t.push_back(Bf16(V + "patch_embed.proj.bias", {kHidden}, s++));
  t.push_back(Bf16(V + "pos_embed.weight", {kPosEmb, kHidden}, s++));
  const std::string b = V + "blocks.0.";
  t.push_back(Bf16(b + "norm1.weight", {kHidden}, s++));
  t.push_back(Bf16(b + "norm1.bias", {kHidden}, s++));
  t.push_back(Bf16(b + "norm2.weight", {kHidden}, s++));
  t.push_back(Bf16(b + "norm2.bias", {kHidden}, s++));
  t.push_back(Bf16(b + "attn.qkv.weight", {3 * kHidden, kHidden}, s++));
  t.push_back(Bf16(b + "attn.qkv.bias", {3 * kHidden}, s++));
  t.push_back(Bf16(b + "attn.proj.weight", {kHidden, kHidden}, s++));
  t.push_back(Bf16(b + "attn.proj.bias", {kHidden}, s++));
  t.push_back(Bf16(b + "mlp.linear_fc1.weight", {kInter, kHidden}, s++));
  t.push_back(Bf16(b + "mlp.linear_fc1.bias", {kInter}, s++));
  t.push_back(Bf16(b + "mlp.linear_fc2.weight", {kHidden, kInter}, s++));
  t.push_back(Bf16(b + "mlp.linear_fc2.bias", {kHidden}, s++));
  for (const std::string& m : {V + "merger", V + "deepstack_merger_list.0"}) {
    const int64_t nd = (m.find("deepstack") == std::string::npos) ? kHidden : ctx4;
    t.push_back(Bf16(m + ".norm.weight", {nd}, s++));
    t.push_back(Bf16(m + ".norm.bias", {nd}, s++));
    t.push_back(Bf16(m + ".linear_fc1.weight", {ctx4, ctx4}, s++));
    t.push_back(Bf16(m + ".linear_fc1.bias", {ctx4}, s++));
    t.push_back(Bf16(m + ".linear_fc2.weight", {kOut, ctx4}, s++));
    t.push_back(Bf16(m + ".linear_fc2.bias", {kOut}, s++));
  }
  return t;
}

// On-disk bytes of every tensor above, and of the pos-embed table alone. Both
// are computed from the fixture rather than written down, so a geometry edit
// cannot drift them away from the tensors they describe.
inline size_t OnDiskBytes() {
  size_t n = 0;
  for (const Fx& f : Tensors()) n += f.bytes.size();
  return n;
}
inline size_t PosEmbedOnDiskBytes() {
  for (const Fx& f : Tensors())
    if (f.name == "model.visual.pos_embed.weight") return f.bytes.size();
  return 0;
}

// Resident bytes of a loaded tower: every store summed, element type included.
inline size_t ResidentBytes(const vllm::multimodal::Qwen3VLVisionWeights& w) {
  size_t n = StoreBytes(w.patch_proj_w) + StoreBytes(w.patch_proj_b) +
             StoreBytes(w.pos_embed_w);
  for (const vllm::multimodal::VisionBlockWeights& b : w.blocks)
    n += StoreBytes(b.norm1_w) + StoreBytes(b.norm1_b) + StoreBytes(b.norm2_w) +
         StoreBytes(b.norm2_b) + StoreBytes(b.qkv_w) + StoreBytes(b.qkv_b) +
         StoreBytes(b.proj_w) + StoreBytes(b.proj_b) + StoreBytes(b.fc1_w) +
         StoreBytes(b.fc1_b) + StoreBytes(b.fc2_w) + StoreBytes(b.fc2_b);
  auto merger = [](const vllm::multimodal::VisionMergerWeights& m) {
    return StoreBytes(m.norm_w) + StoreBytes(m.norm_b) + StoreBytes(m.fc1_w) +
           StoreBytes(m.fc1_b) + StoreBytes(m.fc2_w) + StoreBytes(m.fc2_b);
  };
  n += merger(w.merger);
  for (const vllm::multimodal::VisionMergerWeights& m : w.deepstack_mergers)
    n += merger(m);
  return n;
}

}  // namespace qwen3vl_tiny

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// 1. THE STORAGE GATE — Qwen3-VL, through the shared production loader.
// ─────────────────────────────────────────────────────────────────────────────
//
// `LoadQwen3VLVisionWeights` is the reader Qwen3-VL-4B (`qwen3_vl.cpp:431`),
// the Qwen3.5/3.6-27B dense path and the Qwen3.6-35B MoE path
// (`qwen3_5_weights.cpp:1770`) all share, so this one case covers all three.
TEST_CASE("vision tower dtype: the Qwen3-VL tower resides in the checkpoint's own bytes (#1359)") {
  const TempFile f(BuildSt(qwen3vl_tiny::Tensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(f.path()));

  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionWeights(shards, qwen3vl_tiny::Config());

  // The pos-embed table is the ONE deliberate f32 exception (spec §4.3): its
  // values reach host arithmetic in `VisionPosEmbedInterpolate` without being
  // narrowed first, so it costs 2x its on-disk bytes on purpose and says so.
  const size_t pos = qwen3vl_tiny::PosEmbedOnDiskBytes();
  REQUIRE(pos > 0);
  CHECK(StoreBytes(w.pos_embed_w) == 2 * pos);

  // Everything else must be exactly what the checkpoint ships. Pre-#1359 this
  // reads 2x, which is the defect stated in bytes.
  const size_t on_disk = qwen3vl_tiny::OnDiskBytes();
  INFO("resident " << qwen3vl_tiny::ResidentBytes(w) << " B, on disk " << on_disk
                   << " B, pos_embed " << pos << " B");
  CHECK(qwen3vl_tiny::ResidentBytes(w) == on_disk + pos);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. BIT-IDENTITY — the invariant the storage change must NOT move.
// ─────────────────────────────────────────────────────────────────────────────
//
// Green before the change and green after: that is the point. Every consumer
// narrows with `vt::F32ToBF16` before its first GEMM, so what a consumer sees
// is `Bits(stored)`, and `Bits(stored)` must be the checkpoint's own bits on
// both trees. If this case ever moves, the change stopped being storage-only.
TEST_CASE("vision tower dtype: the stored Qwen3-VL weights carry the checkpoint's bf16 bits") {
  const TempFile f(BuildSt(qwen3vl_tiny::Tensors()));
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(f.path()));
  const vllm::multimodal::Qwen3VLVisionWeights w =
      vllm::LoadQwen3VLVisionWeights(shards, qwen3vl_tiny::Config());

  // seed 101 is `patch_embed.proj.weight`, seed 108 is `blocks.0.attn.qkv.weight`
  // (the order in `qwen3vl_tiny::Tensors`). A row-address error moves the value.
  REQUIRE(w.patch_proj_w.size() ==
          static_cast<size_t>(qwen3vl_tiny::kHidden * qwen3vl_tiny::kChannels *
                              qwen3vl_tiny::kTemporal * qwen3vl_tiny::kPatch *
                              qwen3vl_tiny::kPatch));
  for (size_t i = 0; i < w.patch_proj_w.size(); ++i) {
    INFO("patch_proj_w[" << i << "]");
    CHECK(Bits(w.patch_proj_w[i]) == FixtureBits(101, static_cast<int64_t>(i)));
  }
  REQUIRE(w.blocks.size() == 1);
  const auto& qkv = w.blocks[0].qkv_w;
  REQUIRE(qkv.size() ==
          static_cast<size_t>(3 * qwen3vl_tiny::kHidden * qwen3vl_tiny::kHidden));
  for (size_t i = 0; i < qkv.size(); ++i) {
    INFO("blocks.0.attn.qkv.weight[" << i << "]");
    CHECK(Bits(qkv[i]) == FixtureBits(108, static_cast<int64_t>(i)));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. THE ROUND TRIP IS THE IDENTITY — exhaustively, over all 65,536 patterns.
// ─────────────────────────────────────────────────────────────────────────────
//
// The mechanical argument in spec §3.1, executed rather than asserted.
// `BF16ToF32` (`src/vt/dtype.cpp:317`) is `b << 16`, so the low 16 mantissa bits
// are zero; `F32ToBF16` (`:319-326`) adds at most `0x8000` to a value whose low
// 16 bits are zero, and the carry therefore never reaches bit 16. Narrowing a
// value that ORIGINATED bf16 is exact, which is why the storage change moves no
// output bit — and why §6.4's truncation mutation is detectable at all.
//
// THE ONE EXCEPTION IS NAMED RATHER THAN EXCLUDED SILENTLY. `F32ToBF16` takes an
// explicit NaN branch (`dtype.cpp:320-322`) that FORCES the quiet bit, so a
// SIGNALLING bf16 NaN comes back quieted. That is 126 of the 65,536 patterns —
// both signs x the 63 mantissas that do not already carry `0x40` — and it is not
// reachable from a weight tensor, but a case that swept it under a filter would
// be hiding the one place the round trip is not the identity. So it is counted,
// and both what moved and HOW it moved are asserted.
TEST_CASE("vision tower dtype: F32ToBF16(BF16ToF32(b)) == b for every bf16 pattern") {
  size_t moved = 0, moved_nan = 0, moved_quiet_bit_only = 0;
  uint32_t first = 0x10000u;
  for (uint32_t b = 0; b <= 0xFFFFu; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    // Deliberately spelled through the SAME `Bits` overload set the storage
    // cases read with: `Bits(f32 store) == Bits(bf16 store)` for every pattern
    // below is precisely the reason narrowing the store moves no consumer bit.
    const uint16_t back = Bits(vt::BF16ToF32(bits));
    if (back == Bits(bits)) continue;
    ++moved;
    if ((b & 0x7F80u) == 0x7F80u && (b & 0x007Fu) != 0) ++moved_nan;
    if (back == static_cast<uint16_t>(b | 0x0040u)) ++moved_quiet_bit_only;
    if (first == 0x10000u) first = b;
  }
  INFO("first pattern that moved: 0x" << std::hex << first);
  CHECK(moved == moved_nan);              // nothing but NaN moved
  CHECK(moved == moved_quiet_bit_only);   // and it moved only by being quieted
  CHECK(moved == 126);                    // 2 signs x 63 signalling payloads
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. THE NARROWING ROUNDS TO NEAREST EVEN — the arm the GGUF towers need.
// ─────────────────────────────────────────────────────────────────────────────
//
// Case 3 above CANNOT see a truncating narrow, and that is not an oversight: on
// a value that originated bf16 the low 16 bits are zero, so truncation and
// round-to-nearest-even agree by construction. That is the very reason the
// safetensors towers are bit-identical. It also means the round-trip arm is the
// wrong instrument for the risk the spec's §5.1 names.
//
// The GGUF towers ARE exposed to it. `clip_mmproj_gguf.cpp` and
// `minimax_h3_vision_gguf.cpp` dequantize a k-quant row to a GENUINE f32 and
// narrow it once at load time, so a second, truncating rounding function written
// into either loader would move those towers' numbers — and no token gate would
// see it. Neither can the value checks in `test_clip_mmproj_gguf` or
// `test_minimax_h3`: both sides of those comparisons call `vt::F32ToBF16`, so
// they pin CONSISTENCY and would stay green under any rounding rule.
//
// This case is the one that does not. It pins the rule itself, on four f32 bit
// patterns whose low 16 bits are NOT zero, so it is stated independently of the
// function under test. Under `u >> 16` two of the four move: 0x3F80C000 (which
// must round UP to 0x3F81 and truncates to 0x3F80) and 0x3F818000 (the odd-tie,
// which must round up to 0x3F82 and truncates to 0x3F81). The other two already
// truncate to their correct answer, which is why a four-case table is the
// smallest one that separates the rules rather than the smallest one that
// exercises them.
TEST_CASE("vision tower dtype: F32ToBF16 rounds to nearest EVEN, it does not truncate") {
  struct Case {
    uint32_t f32_bits;
    uint16_t want;
    const char* why;
  };
  // mantissa 0x008000 is the exact halfway point between two bf16 neighbours.
  const Case cases[] = {
      {0x3F804000u, 0x3F80u, "below the halfway point: stays"},
      {0x3F80C000u, 0x3F81u, "above the halfway point: rounds up"},
      {0x3F808000u, 0x3F80u, "exact tie, candidate EVEN: stays"},
      {0x3F818000u, 0x3F82u, "exact tie, candidate ODD: rounds up"},
  };
  for (const Case& c : cases) {
    float f = 0.0F;
    std::memcpy(&f, &c.f32_bits, sizeof(f));
    INFO(c.why << " (f32 bits 0x" << std::hex << c.f32_bits << ")");
    CHECK(vt::F32ToBF16(f) == c.want);
  }
}
