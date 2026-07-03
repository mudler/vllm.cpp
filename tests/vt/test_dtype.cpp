// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
#include <doctest/doctest.h>

#include "vt/dtype.h"

TEST_CASE("DType sizes and names") {
  CHECK(vt::SizeOf(vt::DType::kF32) == 4);
  CHECK(vt::SizeOf(vt::DType::kF16) == 2);
  CHECK(vt::SizeOf(vt::DType::kBF16) == 2);
  CHECK(vt::SizeOf(vt::DType::kI8) == 1);
  CHECK(vt::SizeOf(vt::DType::kI32) == 4);
  CHECK(vt::SizeOf(vt::DType::kI64) == 8);
  CHECK(std::string(vt::Name(vt::DType::kBF16)) == "bf16");
}

TEST_CASE("f16 round-trip on exactly representable values") {
  for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -65504.0f, 65504.0f}) {
    CHECK(vt::F16ToF32(vt::F32ToF16(v)) == v);
  }
}

TEST_CASE("f16 known bit patterns") {
  CHECK(vt::F32ToF16(1.0f) == 0x3C00);
  CHECK(vt::F32ToF16(-2.0f) == 0xC000);
  CHECK(vt::F16ToF32(0x3555) == doctest::Approx(0.333252f));  // ~1/3 in f16
}

TEST_CASE("bf16 round-trip and truncation semantics") {
  for (float v : {0.0f, 1.0f, -1.0f, 0.5f, 3.0f, 1024.0f}) {
    CHECK(vt::BF16ToF32(vt::F32ToBF16(v)) == v);
  }
  CHECK(vt::F32ToBF16(1.0f) == 0x3F80);
  // round-to-nearest-even: 1.00390625 (0x3F808000) is exactly halfway
  // between bf16 1.0 (0x3F80) and 1.0078125 (0x3F81); even mantissa wins.
  CHECK(vt::F32ToBF16(1.00390625f) == 0x3F80);
  // just above the midpoint rounds up
  CHECK(vt::F32ToBF16(1.00390637f) == 0x3F81);
}

TEST_CASE("NaN and infinity survive conversion") {
  CHECK(std::isnan(vt::F16ToF32(vt::F32ToF16(std::nanf("")))));
  CHECK(std::isnan(vt::BF16ToF32(vt::F32ToBF16(std::nanf("")))));
  CHECK(std::isinf(vt::F16ToF32(vt::F32ToF16(std::numeric_limits<float>::infinity()))));
  CHECK(vt::F16ToF32(vt::F32ToF16(1e10f)) == std::numeric_limits<float>::infinity());  // overflow → inf
}

TEST_CASE("VT_CHECK throws with message") {
  CHECK_THROWS_WITH_AS(VT_CHECK(false, "boom"), doctest::Contains("boom"), std::runtime_error);
}
