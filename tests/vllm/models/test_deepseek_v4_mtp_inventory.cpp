// `CLAIM-DEEPSEEK-V4-MTP` R1 (#1314) — the MTP tail is CLASSIFIED, not just counted.
//
// The shapes below are not invented. They are read from the safetensors headers of
// `/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3` (2026-08-31), so this gate
// states the real artifact as data and needs no 100 GB file to run.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"

using vllm::ClassifyDeepseekV4MtpTail;
using vllm::DeepseekV4MtpTensorDesc;

namespace {

// One head as the artifact stores it: fp8-block attention at exactly 128x128,
// MXFP4 routed experts at group 32, and plain norms.
std::vector<DeepseekV4MtpTensorDesc> RealHead(int64_t l, int64_t experts = 2) {
  const std::string p = "mtp." + std::to_string(l) + ".";
  std::vector<DeepseekV4MtpTensorDesc> v{
      {p + "attn.attn_sink", "F32", {64}},
      {p + "attn.kv_norm.weight", "BF16", {512}},
      {p + "attn_norm.weight", "BF16", {4096}},
      {p + "attn.wkv.weight", "F8_E4M3", {512, 4096}},
      {p + "attn.wkv.weight.scale", "F8_E8M0", {4, 32}},
      {p + "attn.wq_b.weight", "F8_E4M3", {32768, 1024}},
      {p + "attn.wq_b.weight.scale", "F8_E8M0", {256, 8}},
      {p + "ffn.shared_experts.w1.weight", "F8_E4M3", {2048, 4096}},
      {p + "ffn.shared_experts.w1.weight.scale", "F8_E8M0", {16, 32}},
      {p + "ffn.gate.weight", "BF16", {216, 4096}},
  };
  for (int64_t e = 0; e < experts; ++e) {
    const std::string b = p + "ffn.experts." + std::to_string(e) + ".w1.weight";
    v.push_back({b, "I8", {2048, 2048}});          // [N, K/2], so K = 4096
    v.push_back({b + ".scale", "F8_E8M0", {2048, 128}});  // K/32 == 128
  }
  return v;
}

}  // namespace

TEST_CASE("R1: the real MTP tail classifies with no refusal") {
  const auto inv = ClassifyDeepseekV4MtpTail(RealHead(0));
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 1);
  CHECK(inv.plain == 4);      // attn_sink, kv_norm, attn_norm, ffn.gate
  CHECK(inv.fp8_block == 3);  // wkv, wq_b, shared_experts.w1
  CHECK(inv.mxfp4 == 2);      // the two routed experts
}

TEST_CASE("R1: THREE heads are counted, which is the artifact's own shape") {
  // The config says `num_nextn_predict_layers = 1`; this artifact carries three,
  // and three is what makes a K5 draft possible. The count must come from the
  // TENSORS, never from the config, or the extra heads are silently invisible.
  std::vector<DeepseekV4MtpTensorDesc> all;
  for (int64_t l = 0; l < 3; ++l) {
    const auto h = RealHead(l);
    all.insert(all.end(), h.begin(), h.end());
  }
  const auto inv = ClassifyDeepseekV4MtpTail(all);
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 3);
  CHECK(inv.mxfp4 == 6);
  CHECK(inv.fp8_block == 9);
}

TEST_CASE("R1: an absent tail is EMPTY, not a refusal") {
  // The two shipped GGUFs have no tail at all. That is a legitimate checkpoint,
  // and it must read as "no head here" rather than as a broken one.
  const auto inv = ClassifyDeepseekV4MtpTail({});
  CHECK(inv.refusal.empty());
  CHECK(inv.num_heads == 0);
  CHECK(inv.plain + inv.fp8_block + inv.mxfp4 == 0);
}

TEST_CASE("R1: NVFP4's group of 16 is REFUSED, not read as MXFP4") {
  // The 156.7 GiB checkpoint's tail uses NVFP4: group 16, a double scale. Reading
  // it with the MXFP4 group-32 path would dequantize every weight against the
  // wrong exponent and produce a plausible, entirely wrong draft head.
  auto v = RealHead(0, /*experts=*/1);
  for (auto& d : v) {
    if (d.name.find("experts.0.w1.weight.scale") != std::string::npos) {
      d.shape = {2048, 256};  // K/16, the NVFP4 grouping
    }
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("experts.0.w1") != std::string::npos);
  CHECK(inv.mxfp4 == 0);
}

TEST_CASE("R1: a quantized weight with NO scale is refused by name") {
  auto v = RealHead(0, /*experts=*/1);
  for (size_t i = 0; i < v.size(); ++i) {
    if (v[i].name.find("attn.wkv.weight.scale") != std::string::npos) {
      v.erase(v.begin() + static_cast<long>(i));
      break;
    }
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("attn.wkv.weight") != std::string::npos);
  // wq_b AND shared_experts.w1 still classify; only wkv is refused.
  CHECK(inv.fp8_block == 2);
}

TEST_CASE("R1: a scale that does not tile its weight at 128x128 is refused") {
  // A block size that is not 128 is a DIFFERENT fp8 layout. Accepting it would
  // walk the scale array with the wrong stride.
  auto v = RealHead(0, /*experts=*/1);
  for (auto& d : v) {
    if (d.name == "mtp.0.attn.wkv.weight.scale") d.shape = {8, 32};  // 64x128
  }
  const auto inv = ClassifyDeepseekV4MtpTail(v);
  CHECK(!inv.refusal.empty());
  CHECK(inv.refusal.find("128x128") != std::string::npos);
}

// ── R1b: the tail is ROUTED to borrowed views and dequantized ON DEMAND ──────
//
// The expected values below are computed BY HAND from the formats, not read back
// from the same helper the code calls. A gate that only compares the production
// path against `DequantMxfp4ToF32` would prove the two agree and would still pass
// if this code chose the wrong reader for the layout, which is the actual risk.

using vllm::DequantizeDeepseekV4MtpTensor;
using vllm::DeepseekV4MtpFormat;
using vllm::DeepseekV4MtpTensorView;
using vllm::RouteDeepseekV4MtpTail;

TEST_CASE("R1b: an MXFP4 view dequantizes to HAND-COMPUTED values") {
  // One row, one group of 32 => 16 packed bytes, one E8M0 scale byte.
  // scale 128 => 2^(128-127) = 2. LUT is {0, .5, 1, 1.5, 2, 3, 4, 6}; bit 3 signs.
  std::vector<uint8_t> packed(16, 0x00);
  packed[0] = 0x21;  // low  = 1 -> 0.5 ; high = 2 -> 1.0
  packed[1] = 0x9C;  // low  = 0xC -> -2.0 ; high = 0x9 -> -0.5
  const std::vector<uint8_t> scale{128};

  DeepseekV4MtpTensorView v;
  v.format = DeepseekV4MtpFormat::kMxfp4;
  v.dtype = "I8";
  v.shape = {1, 16};
  v.data = packed.data();
  v.scale = scale.data();
  v.out_dim = 1;
  v.in_dim = 32;

  const std::vector<float> got = DequantizeDeepseekV4MtpTensor(v);
  REQUIRE(got.size() == 32u);
  CHECK(got[0] == doctest::Approx(1.0f));   // 0.5 * 2
  CHECK(got[1] == doctest::Approx(2.0f));   // 1.0 * 2
  CHECK(got[2] == doctest::Approx(-4.0f));  // -2.0 * 2
  CHECK(got[3] == doctest::Approx(-1.0f));  // -0.5 * 2
  for (size_t i = 4; i < got.size(); ++i) CHECK(got[i] == doctest::Approx(0.0f));
}

TEST_CASE("R1b: the MXFP4 view's in_dim is the LOGICAL width, not the stored one") {
  // Two e2m1 nibbles share a byte, so a [N, K/2] store is a K-wide weight.
  // Reading `shape[1]` as the width would halve every routed expert and still
  // produce finite numbers -- the failure a shape check cannot see.
  std::vector<vllm::DeepseekV4MtpTensorDesc> tail{
      {"mtp.0.ffn.experts.0.w1.weight", "I8", {64, 128}},
      {"mtp.0.ffn.experts.0.w1.weight.scale", "F8_E8M0", {64, 8}},
  };
  std::vector<uint8_t> w(64 * 128, 0x11), sc(64 * 8, 127);
  const std::vector<const uint8_t*> pay{w.data(), sc.data()};
  const auto r = RouteDeepseekV4MtpTail(tail, pay);
  REQUIRE(r.refusal.empty());
  REQUIRE(r.heads.size() == 1u);
  const auto it = r.heads[0].tensors.find("ffn.experts.0.w1.weight");
  REQUIRE(it != r.heads[0].tensors.end());
  CHECK(it->second.out_dim == 64);
  CHECK(it->second.in_dim == 256);  // 128 stored bytes -> 256 e2m1 values
  CHECK(it->second.format == DeepseekV4MtpFormat::kMxfp4);
  CHECK(it->second.scale == sc.data());
}

TEST_CASE("R1b: an FP8-block view dequantizes to HAND-COMPUTED values") {
  // E4M3 is 1-4-3: value = 2^(e-7) * (1 + m/8). 0x38 -> e=7,m=0 -> 1.0;
  // 0x40 -> e=8 -> 2.0; 0x3C -> e=7,m=4 -> 1.5. E8M0 scale 128 -> 2.
  std::vector<uint8_t> w(128 * 128, 0x00);
  w[0] = 0x38;
  w[1] = 0x40;
  w[2] = 0x3C;
  const std::vector<uint8_t> scale{128};

  DeepseekV4MtpTensorView v;
  v.format = DeepseekV4MtpFormat::kFp8Block;
  v.dtype = "F8_E4M3";
  v.shape = {128, 128};
  v.data = w.data();
  v.scale = scale.data();
  v.out_dim = 128;
  v.in_dim = 128;

  const std::vector<float> got = DequantizeDeepseekV4MtpTensor(v);
  REQUIRE(got.size() == 128u * 128u);
  CHECK(got[0] == doctest::Approx(2.0f));  // 1.0 * 2
  CHECK(got[1] == doctest::Approx(4.0f));  // 2.0 * 2
  CHECK(got[2] == doctest::Approx(3.0f));  // 1.5 * 2
}

TEST_CASE("R1b: routing REFUSES mismatched descriptor and payload arrays") {
  std::vector<vllm::DeepseekV4MtpTensorDesc> tail{
      {"mtp.0.attn_norm.weight", "BF16", {8}},
  };
  const auto r = RouteDeepseekV4MtpTail(tail, {});
  CHECK(!r.refusal.empty());
  CHECK(r.refusal.find("payload") != std::string::npos);
}

TEST_CASE("R1b: a tail this arm cannot read is refused BEFORE any view is built") {
  // Routing runs the classifier first, so an unreadable layout never becomes a
  // borrowed pointer that a later consumer would dequantize as something else.
  std::vector<vllm::DeepseekV4MtpTensorDesc> tail{
      {"mtp.0.ffn.experts.0.w1.weight", "I8", {64, 128}},
      {"mtp.0.ffn.experts.0.w1.weight.scale", "F8_E8M0", {64, 16}},  // group 16
  };
  std::vector<uint8_t> w(64 * 128, 0x11), sc(64 * 16, 127);
  const auto r = RouteDeepseekV4MtpTail(tail, {w.data(), sc.data()});
  CHECK(!r.refusal.empty());
  CHECK(r.heads.empty());
}

TEST_CASE("R1b: the FP8 block extent is proven by a SECOND block row") {
  // The first version of the case above read only row 0, where every block
  // extent picks scale[0] -- so a mutation changing the extent from 128 to 64
  // survived it. Two block rows carrying DIFFERENT scales is what makes the
  // extent observable: with 128 the boundary is at row 128, with 64 it is at 64.
  const int64_t N = 256, K = 128;
  std::vector<uint8_t> w(static_cast<size_t>(N * K), 0x00);
  w[0] = 0x38;                                        // row 0   -> 1.0
  w[static_cast<size_t>(64 * K)] = 0x38;              // row 64  -> 1.0
  w[static_cast<size_t>(128 * K)] = 0x38;             // row 128 -> 1.0
  const std::vector<uint8_t> scale{128, 130};         // 2 and 8

  DeepseekV4MtpTensorView v;
  v.format = DeepseekV4MtpFormat::kFp8Block;
  v.dtype = "F8_E4M3";
  v.shape = {N, K};
  v.data = w.data();
  v.scale = scale.data();
  v.out_dim = N;
  v.in_dim = K;

  const std::vector<float> got = DequantizeDeepseekV4MtpTensor(v);
  REQUIRE(got.size() == static_cast<size_t>(N * K));
  CHECK(got[0] == doctest::Approx(2.0f));                        // scale[0] = 2
  CHECK(got[static_cast<size_t>(64 * K)] == doctest::Approx(2.0f));   // STILL scale[0]
  CHECK(got[static_cast<size_t>(128 * K)] == doctest::Approx(8.0f));  // scale[1] = 8
}
