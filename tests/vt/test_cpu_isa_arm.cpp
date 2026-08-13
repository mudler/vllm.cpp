#include <ostream>
#include <string>
#include <string_view>

#include "doctest/doctest.h"
#include "vt/cpu/cpu_isa_arm.h"

TEST_CASE("Arm ISA selection requires every CPU capability in a tier") {
  using vt::cpu::ArmIsaTier;
  using vt::cpu::SelectArmIsaTier;
  ArmIsaTier selected{};
  std::string error;

  vt::cpu::ArmIsaCaps caps{};
  CHECK(SelectArmIsaTier(caps, "", &selected, &error));
  CHECK(selected == ArmIsaTier::kPortable);

  caps.neon = true;
  CHECK(SelectArmIsaTier(caps, "", &selected, &error));
  CHECK(selected == ArmIsaTier::kNeon);

  caps.dotprod = true;
  CHECK(SelectArmIsaTier(caps, "", &selected, &error));
  CHECK(selected == ArmIsaTier::kDotProd);

  caps.i8mm = true;
  CHECK(SelectArmIsaTier(caps, "", &selected, &error));
  CHECK(selected == ArmIsaTier::kI8mm);

  caps.dotprod = false;
  CHECK_FALSE(SelectArmIsaTier(caps, "i8mm", &selected, &error));
}

TEST_CASE("Arm forced tiers fail closed instead of silently narrowing") {
  using vt::cpu::ArmIsaTier;
  ArmIsaTier selected{};
  std::string error;
  const vt::cpu::ArmIsaCaps baseline{.neon = true};

  CHECK(vt::cpu::SelectArmIsaTier(baseline, "portable", &selected, &error));
  CHECK(selected == ArmIsaTier::kPortable);
  CHECK(vt::cpu::SelectArmIsaTier(baseline, "neon", &selected, &error));
  CHECK(selected == ArmIsaTier::kNeon);
  CHECK_FALSE(vt::cpu::SelectArmIsaTier(baseline, "dotprod", &selected, &error));
  CHECK_FALSE(vt::cpu::SelectArmIsaTier(baseline, "i8mm", &selected, &error));
  CHECK_FALSE(vt::cpu::SelectArmIsaTier(baseline, "sve2", &selected, &error));
  const bool has_reason = error.find("unsupported") != std::string::npos ||
                          error.find("unknown") != std::string::npos;
  CHECK(has_reason);
}

TEST_CASE("independent Arm feature toggles force execution or fail closed") {
  bool enabled = false;
  std::string error;
  const vt::cpu::ArmIsaCaps baseline{.neon = true};
  const vt::cpu::ArmIsaCaps rich{.neon = true, .dotprod = true, .i8mm = true};

  CHECK(vt::cpu::ResolveArmIsaToggle(
      baseline, vt::cpu::ArmIsaTier::kI8mm, "portable", &enabled, &error));
  CHECK_FALSE(enabled);
  CHECK_FALSE(vt::cpu::ResolveArmIsaToggle(
      baseline, vt::cpu::ArmIsaTier::kI8mm, "i8mm", &enabled, &error));
  CHECK(vt::cpu::ResolveArmIsaToggle(
      rich, vt::cpu::ArmIsaTier::kI8mm, "i8mm", &enabled, &error));
  CHECK(enabled);
  CHECK(vt::cpu::ResolveArmIsaToggle(
      rich, vt::cpu::ArmIsaTier::kDotProd, "auto", &enabled, &error));
  CHECK(enabled);
  CHECK_FALSE(vt::cpu::ResolveArmIsaToggle(
      rich, vt::cpu::ArmIsaTier::kDotProd, "sve2", &enabled, &error));
}

TEST_CASE("Arm release inventory lists only kernels that exist") {
  const auto inventory = vt::cpu::ArmIsaTierInventory();
  REQUIRE(inventory.size() == 4);
  CHECK(inventory[0].name == "portable");
  CHECK(inventory[1].name == "neon");
  CHECK(inventory[2].name == "dotprod");
  CHECK(inventory[3].name == "i8mm");
  CHECK(inventory[2].kernel_families == "q8-dot");
  CHECK(inventory[2].linux_probe == "getauxval:AT_HWCAP:HWCAP_ASIMDDP");
  CHECK(inventory[2].darwin_probe == "sysctl:hw.optional.arm.FEAT_DotProd");
  CHECK(inventory[3].kernel_families == "quant-dot,quant-repack");
  CHECK(inventory[3].linux_probe == "getauxval:AT_HWCAP2:HWCAP2_I8MM");
  CHECK(inventory[3].darwin_probe == "sysctl:hw.optional.arm.FEAT_I8MM");
  for (const auto& tier : inventory) {
    CHECK(tier.cpu_features.find("sve") == std::string_view::npos);
    CHECK(tier.cpu_features.find("sme") == std::string_view::npos);
    CHECK(tier.cpu_features.find("bf16") == std::string_view::npos);
  }
}

TEST_CASE("detected Arm host resolves to a supported tier") {
  vt::cpu::ArmIsaTier selected{};
  std::string error;
  const auto caps = vt::cpu::DetectArmIsaCaps();
  CHECK(vt::cpu::SelectArmIsaTier(caps, "", &selected, &error));
  CHECK(vt::cpu::ArmIsaTierSupported(caps, selected));
}
