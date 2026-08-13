#include <array>
#include <bit>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vt/cpu/cpu_isa_x86.h"

namespace {

vt::cpu::X86IsaCaps FullAvx512() {
  return {.sse2 = true,
          .f16c = true,
          .avx = true,
          .osxsave = true,
          .avx2 = true,
          .avx512f = true,
          .avx512bw = true,
          .avx512vl = true,
          .xcr0 = 0xe6};
}

}  // namespace

TEST_CASE("device pool size classes preserve power-of-two alignment") {
  CHECK(vllm::DevicePool::SizeClassForTest(0) == 1);
  for (std::size_t bytes = 1; bytes <= 4096; ++bytes) {
    const std::size_t size_class =
        vllm::DevicePool::SizeClassForTest(bytes);
    CHECK(size_class >= bytes);
    const int msb = static_cast<int>(std::bit_width(bytes)) - 1;
    if (msb >= 4) {
      const std::size_t alignment = std::size_t{1} << (msb - 4);
      CHECK((size_class % alignment) == 0);
    } else {
      CHECK(size_class == bytes);
    }
  }
}

TEST_CASE("device pool size classes reject unrepresentable rounding") {
  constexpr int kClassBits = 4;
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const int msb = static_cast<int>(std::bit_width(maximum)) - 1;
  const int shift = msb - kClassBits;
  const std::size_t mask = (std::size_t{1} << shift) - 1;
  const std::size_t maximum_aligned = maximum & ~mask;

  CHECK(vllm::DevicePool::SizeClassForTest(0) == 1);
  CHECK(vllm::DevicePool::SizeClassForTest(16) == 16);
  CHECK(vllm::DevicePool::SizeClassForTest(32) == 32);
  CHECK(vllm::DevicePool::SizeClassForTest(maximum_aligned) == maximum_aligned);
  CHECK_THROWS_AS(vllm::DevicePool::SizeClassForTest(maximum_aligned + 1),
                  std::overflow_error);
  CHECK_THROWS_AS(vllm::DevicePool::SizeClassForTest(maximum - mask + 1),
                  std::overflow_error);
  CHECK_THROWS_AS(vllm::DevicePool::SizeClassForTest(maximum),
                  std::overflow_error);
}

TEST_CASE("x86 ISA selection requires CPU bits and exact OS-enabled state") {
  using vt::cpu::SelectX86IsaTier;
  using vt::cpu::X86IsaTier;
  X86IsaTier selected{};
  std::string error;

  vt::cpu::X86IsaCaps caps{.sse2 = true};
  CHECK(SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(selected == X86IsaTier::kSse2);

  caps = FullAvx512();
  caps.xcr0 = 0x6;
  CHECK(SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(selected == X86IsaTier::kAvx2);
  CHECK_FALSE(SelectX86IsaTier(caps, "avx512", &selected, &error));

  caps = FullAvx512();
  for (bool* bit : std::array{&caps.avx512f, &caps.avx512bw, &caps.avx512vl}) {
    *bit = false;
    CHECK_FALSE(SelectX86IsaTier(caps, "avx512", &selected, &error));
    *bit = true;
  }
  CHECK(SelectX86IsaTier(caps, "avx512", &selected, &error));
  CHECK(selected == X86IsaTier::kAvx512);
}

TEST_CASE("x86 forced tiers fail closed instead of silently narrowing") {
  using vt::cpu::SelectX86IsaTier;
  using vt::cpu::X86IsaTier;
  X86IsaTier selected{};
  std::string error;
  const vt::cpu::X86IsaCaps baseline{.sse2 = true};

  CHECK(SelectX86IsaTier(baseline, "portable", &selected, &error));
  CHECK(selected == X86IsaTier::kPortable);
  CHECK(SelectX86IsaTier(baseline, "sse2", &selected, &error));
  CHECK(selected == X86IsaTier::kSse2);
  CHECK_FALSE(SelectX86IsaTier(baseline, "sse2+f16c", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "avx2", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "avx512", &selected, &error));
  CHECK_FALSE(SelectX86IsaTier(baseline, "amx", &selected, &error));
  const bool has_reason = error.find("unsupported") != std::string::npos ||
                          error.find("unknown") != std::string::npos;
  CHECK(has_reason);
}

TEST_CASE("x86 release inventory lists only compiled elementwise GEMM tiers") {
  const auto inventory = vt::cpu::X86IsaTierInventory();
  REQUIRE(inventory.size() == 5);
  CHECK(inventory[0].name == "portable");
  CHECK(inventory[1].name == "sse2");
  CHECK(inventory[2].name == "sse2+f16c");
  CHECK(inventory[3].name == "avx2");
  CHECK(inventory[4].name == "avx512");
  for (const auto& tier : inventory) {
    CHECK(tier.kernel_family == "elementwise-gemm");
    CHECK(tier.cpu_features.find("vnni") == std::string_view::npos);
    CHECK(tier.cpu_features.find("amx") == std::string_view::npos);
  }
}

TEST_CASE("detected host resolves to a supported tier") {
  vt::cpu::X86IsaTier selected{};
  std::string error;
  const auto caps = vt::cpu::DetectX86IsaCaps();
  CHECK(vt::cpu::SelectX86IsaTier(caps, "", &selected, &error));
  CHECK(vt::cpu::X86IsaTierSupported(caps, selected));
}
