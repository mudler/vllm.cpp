// W5d-2 (row MODEL-MM-QWEN4-EXP, issue #2249 item 5): the BEHAVIOUR-NEUTRALITY
// gate for giving `BuildMropeCosSinHost` external linkage.
//
// WHAT THIS FILE IS FOR. `BuildMropeCosSinHost` was `static` in
// src/vllm/model_executor/models/qwen3_5.cpp, so the interleaved-mRoPE cos|sin
// tables Qwen4-Exp's QSA block needs could not be built from another
// translation unit. The change gives it external linkage and declares it in
// include/vllm/model_executor/models/qwen3_5_mrope.h. Nothing else moves — the
// body is byte-identical, and `git show 94de63ff5:...qwen3_5.cpp | sed -n
// '9473,9514p'` sha256-matches the body on this branch.
//
// A byte-identical body is not on its own a value guarantee, because the
// keyword that changed is exactly the one that decides which definition a
// caller binds to. So the expected tables below are the ones the FILE-STATIC
// definition produced at base SHA 94de63ff5, captured by compiling its text
// verbatim (`sed`-extracted, not retyped) in a standalone harness and dumping
// the f32 BIT PATTERNS. The comparison here is bitwise on those patterns and
// NOT an epsilon: this is a pure host computation over `std::cos`/`std::pow`
// with no reduction-order freedom, so a tolerance would hide the one defect an
// extraction can introduce.
//
// THIS FILE IS ALSO THE SEAM CASE. It is a FOREIGN translation unit that reaches
// the definition only through the new public header — the property qwen4_exp
// needs and the property `static` denied. It is NOT the reachability proof:
// that is `test_qwen3_5_moe_vision`'s
// `qwen3_5_moe_vl_image_forward_uses_MRoPE_positions_not_plain_1d`, which runs
// the production `Qwen3_5MoeVLGenerateGreedy` driver over this cache.
//
// WHAT THE FOUR CASES SEPARATE. C1 and C2 are the SAME config and the SAME
// positions under the two layouts, so they differ if and only if the
// interleaved/chunked branch is read. C3 is T == 1 at a different rotary_dim and
// a different rope_theta. C4 shrinks the t section so the `pair <= 3 * sec[k]`
// boundary — upstream's `<=`, mrope.py:60-61 — decides differently from C1.
// Every case carries non-zero and per-axis-DISTINCT positions, so an axis
// selection that collapses to axis 0 cannot pass.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_mrope.h"
#include "vllm/transformers_utils/hf_config.h"

namespace {

using vllm::HfConfig;

HfConfig Cfg(int64_t rot, double theta, bool interleaved,
             std::vector<int64_t> section) {
  HfConfig c;
  c.rotary_dim = rot;
  c.rope_theta = theta;
  c.rope_parameters.mrope_interleaved = interleaved;
  c.rope_parameters.mrope_section = std::move(section);
  return c;
}

// Bitwise comparison against the base-SHA capture. `expect` holds IEEE-754
// binary32 bit patterns, so a value that differs in the last ulp fails here.
void CheckBits(const std::vector<float>& got, const uint32_t* expect,
               size_t n) {
  REQUIRE(got.size() == n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &got[i], sizeof(bits));
    INFO("index ", i);
    CHECK(bits == expect[i]);
  }
}

}  // namespace

TEST_CASE("qwen3_5_mrope_interleaved_table_is_bit_identical_to_the_file_static") {
  // C1: interleaved, sections {4,2,2}, T = 3, positions offset off zero and
  // distinct per axis (t = 100.., h = 7.., w = 3..).
  static const uint32_t kC1[] = {
    0x3f5cc0eeu, 0x3ea3f8b9u, 0x3f7ed94fu, 0x3f58940eu, 0x3f7ffe65u, 0x3f7ffffeu,
    0x3f7fffacu, 0x3f7ffffdu, 0xbf01a12eu, 0x3f728444u, 0x3dc1ffc1u, 0x3f087dbau,
    0x3be55fc7u, 0x3a0bd97eu, 0x3b4f3e20u, 0x3a136a16u, 0x3f645a6eu, 0x3ea3f8b9u,
    0x3f7df469u, 0x3f57ceb0u, 0x3f7ffe65u, 0x3f7ffffcu, 0x3f7fffaau, 0x3f7ffffdu,
    0x3ee76fedu, 0x3f728444u, 0x3e012e88u, 0x3f09b4f5u, 0x3be55fc7u, 0x3a3a7752u,
    0x3b5150abu, 0x3a14e377u, 0x3dd00c2au, 0x3e172ca2u, 0x3f7cce81u, 0x3f570792u,
    0x3f7ffde7u, 0x3f7ffff9u, 0x3f7fffa9u, 0x3f7ffffdu, 0x3f7eacf8u, 0x3f7d31e3u,
    0x3e213c1eu, 0x3f0aeb13u, 0x3c031213u, 0x3a691526u, 0x3b536335u, 0x3a165cd9u,
  };
  const HfConfig c = Cfg(16, 1000000.0, /*interleaved=*/true, {4, 2, 2});
  const std::vector<int32_t> p = {100, 101, 102, 7, 7, 8, 3, 4, 5};
  CheckBits(vllm::BuildMropeCosSinHost(p, 3, c), kC1, 48);
}

TEST_CASE("qwen3_5_mrope_chunked_table_is_bit_identical_and_differs_from_interleaved") {
  // C2: the SAME config and positions as C1 with mrope_interleaved false. The
  // two tables must both match their capture AND differ from each other, which
  // is what shows the layout branch is read rather than defaulted.
  static const uint32_t kC2[] = {
    0x3f5cc0eeu, 0x3ef746d4u, 0xbf7ff1fbu, 0x3f58940eu, 0x3f7ffe65u, 0x3f7ffff3u,
    0x3f800000u, 0x3f800000u, 0xbf01a12eu, 0xbf602a05u, 0xbca97082u, 0x3f087dbau,
    0x3be55fc7u, 0x3aa32866u, 0x38c6f40cu, 0x378d8490u, 0x3f645a6eu, 0x3f21576bu,
    0xbf7fa65du, 0x3f57ceb0u, 0x3f7ffe65u, 0x3f7ffff3u, 0x3f800000u, 0x3f800000u,
    0x3ee76fedu, 0xbf46c22bu, 0xbd5627adu, 0x3f09b4f5u, 0x3be55fc7u, 0x3aa32866u,
    0x3904a2b3u, 0x37bcb0c1u, 0x3dd00c2au, 0x3f41f4bbu, 0xbf7f194eu, 0x3f570792u,
    0x3f7ffde7u, 0x3f7fffefu, 0x3f800000u, 0x3f800000u, 0x3f7eacf8u, 0xbf271584u,
    0xbdabb024u, 0x3f0aeb13u, 0x3c031213u, 0x3aba774fu, 0x3925cb5fu, 0x37ebdcf1u,
  };
  const HfConfig c = Cfg(16, 1000000.0, /*interleaved=*/false, {4, 2, 2});
  const std::vector<int32_t> p = {100, 101, 102, 7, 7, 8, 3, 4, 5};
  const std::vector<float> got = vllm::BuildMropeCosSinHost(p, 3, c);
  CheckBits(got, kC2, 48);

  const HfConfig ci = Cfg(16, 1000000.0, /*interleaved=*/true, {4, 2, 2});
  CHECK(got != vllm::BuildMropeCosSinHost(p, 3, ci));
}

TEST_CASE("qwen3_5_mrope_single_token_table_is_bit_identical_to_the_file_static") {
  // C3: T == 1, rotary_dim 24, rope_theta 10000, sections {6,3,3}, large t
  // offset. Separates the T > 1 row stride from the per-pair angle math.
  static const uint32_t kC3[] = {
    0x3f0ff813u, 0xbf2e7fc9u, 0xbeb848feu, 0x3f5cc0eeu, 0x3f7922ffu, 0x3f7b33e1u,
    0xbf56cd64u, 0x3f7fee5au, 0x3f7ff3aeu, 0x3f0a5140u, 0x3f64ea30u, 0x3f7a14ebu,
    0x3f53ae61u, 0x3f3b500cu, 0x3f6ed7afu, 0xbf01a12eu, 0x3e6b8592u, 0x3e454f96u,
    0xbf0b44f8u, 0x3cbe1a38u, 0x3c9ed50cu, 0x3f576aa4u, 0x3ee5351du, 0x3e5ae952u,
  };
  const HfConfig c = Cfg(24, 10000.0, /*interleaved=*/true, {6, 3, 3});
  const std::vector<int32_t> p = {1000, 5, 9};
  CheckBits(vllm::BuildMropeCosSinHost(p, 1, c), kC3, 24);
}

TEST_CASE("qwen3_5_mrope_small_t_section_boundary_is_bit_identical_to_the_file_static") {
  // C4: sections {2,3,3}, so `3 * sec[1]` and `3 * sec[2]` are 9 rather than 6
  // and the interleaved axis choice for pairs 7 and 8 flips relative to C1.
  static const uint32_t kC4[] = {
    0xbf7cc244u, 0x3ef72f34u, 0x3f70aae4u, 0x3f7939d3u, 0x3f7ffed2u, 0x3f7fffe0u,
    0x3f7ffff2u, 0x3f800000u, 0xbe226dfbu, 0x3f603089u, 0x3eae878bu, 0x3e6a01dau,
    0x3bc49b59u, 0x3b003204u, 0x3aa9f072u, 0x380d8490u, 0xbecccae0u, 0x3ea3f8b9u,
    0x3f6aabc3u, 0x3f78e499u, 0x3f7ffe65u, 0x3f7fffd3u, 0x3f7ffff1u, 0x3f800000u,
    0xbf6aa128u, 0x3f728444u, 0x3ecc9a40u, 0x3e6f9c08u, 0x3be55fc7u, 0x3b1780ebu,
    0x3aae1587u, 0x38251aa8u,
  };
  const HfConfig c = Cfg(16, 1000000.0, /*interleaved=*/true, {2, 3, 3});
  const std::vector<int32_t> p = {41, 42, 6, 7, 11, 13};
  CheckBits(vllm::BuildMropeCosSinHost(p, 2, c), kC4, 32);
}
