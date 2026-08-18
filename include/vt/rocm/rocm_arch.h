// GCN arch string -> (major, minor) capability, ported 1:1 from
// vllm/platforms/rocm.py:223-291 `_capability_from_gcn_arch` @ pin 55596792.
// Upstream's own docstring states what it mirrors: "how HIP derives
// hipDeviceProp_t.major / .minor".
//
// WHY THIS IS A PLAIN HEADER AND NOT PART OF rocm_backend.hip. It is the only
// piece of the ROCm skeleton that contains a DECISION rather than an API call,
// and it is the piece a wrong answer breaks silently (a mis-parsed capability
// picks the wrong kernel tactic later, it does not throw). Keeping it free of
// <hip/hip_runtime.h> means it compiles and is unit-tested in the ordinary CPU
// build — on CI, and on a machine with no AMD GPU and no ROCm installed — while
// the HIP translation unit stays thin glue that only a real ROCm box can check.
// tests/vt/test_rocm_arch.cpp carries upstream's own worked examples.
#pragma once

#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>

namespace vt::rocm {

// (major, minor) for a gcnArchName like "gfx1100" or "gfx942:sramecc+:xnack-".
//
// Returns nullopt in exactly two cases, which upstream splits into a return and
// a raise: the string is not gfx-prefixed at all (upstream returns None and the
// caller falls back), or it looks like a gfx string but does not match a known
// layout (upstream raises ValueError). We collapse both to nullopt because the
// caller is a `noexcept` static-init path that must degrade to "capability
// unknown" rather than abort a process at load time; the caller reports the
// unparsed string in its own error text, so no diagnostic is lost.
constexpr std::optional<std::pair<int, int>> CapabilityFromGcnArch(std::string_view gcn_arch) {
  // Upstream's `re.match(r"gfx(\d+)", gcn_arch)`: anchored at the start, digits
  // run until the first non-digit, which is what strips the ":sramecc+:xnack-"
  // feature suffix HIP appends to gcnArchName on gfx9 parts.
  constexpr std::string_view kPrefix = "gfx";
  if (gcn_arch.substr(0, kPrefix.size()) != kPrefix) return std::nullopt;

  size_t n = 0;
  while (n < gcn_arch.size() - kPrefix.size()) {
    const char c = gcn_arch[kPrefix.size() + n];
    if (c < '0' || c > '9') break;
    ++n;
  }
  const std::string_view digits = gcn_arch.substr(kPrefix.size(), n);
  const auto digit = [&](size_t i) { return static_cast<int>(digits[i] - '0'); };

  int major = 0;
  int minor = 0;
  if (n == 2 || n == 3) {
    // 1-digit major, the gfx9 family: major + minor (+ stepping).
    // "gfx90a" -> the regex captures "90" -> (9, 0). "gfx942" -> (9, 4).
    major = digit(0);
    minor = digit(1);
  } else if (n == 4) {
    // 2-digit major, gfx10xx/11xx/12xx: major(2) + minor(1) + stepping(1).
    // "gfx1100" -> (11, 0). "gfx1151" (Strix Halo) -> (11, 5).
    major = digit(0) * 10 + digit(1);
    minor = digit(2);
  } else {
    // n < 2: too few digits to split. n >= 5: beyond the known MMms layout, so
    // the major/minor split would be a guess. Upstream raises on both.
    return std::nullopt;
  }

  // Upstream's two sanity rails, same bounds, same reason: nothing below gfx9 is
  // a supported AMD part, and above 12 is a generation that does not exist yet,
  // so either answer means the layout assumption above did not hold.
  if (major < 9 || major > 12) return std::nullopt;
  return std::pair<int, int>{major, minor};
}

// Host launch gate for Prefill SharedK WMMA (#785). True only when `gcn_arch`
// is the literal HIP gcnArchName prefix `gfx1200` or `gfx1201`.
//
// Prefix, not substring: `foogfx1201` is false. After the six-char stem the
// next character must be end-of-string or a non-digit so `gfx1201:xnack-`
// matches and `gfx12010` does not. CapabilityFromGcnArch(12,0) is too wide
// (gfx1202..gfx1209).
constexpr bool GcnArchNameIsGfx12PrefillWmma(std::string_view gcn_arch) {
  auto prefix_ok = [](std::string_view s, std::string_view stem) {
    if (s.size() < stem.size()) return false;
    if (s.substr(0, stem.size()) != stem) return false;
    if (s.size() == stem.size()) return true;
    const char c = s[stem.size()];
    return c < '0' || c > '9';
  };
  return prefix_ok(gcn_arch, "gfx1200") || prefix_ok(gcn_arch, "gfx1201");
}

}  // namespace vt::rocm
