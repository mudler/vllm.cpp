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


// --- The memory-model decision (BACKEND-ROCM, issue #2511) -------------------
//
// Which allocator `RocmBackend::Alloc` uses, and whether the backend may claim
// its allocations are host-addressable. It lives HERE, beside
// `CapabilityFromGcnArch`, for the reason the header comment above already
// gives: it is a DECISION, not an API call, and a wrong answer breaks silently
// rather than throwing. Keeping it HIP-free means the whole policy table is
// unit-tested in the ordinary CPU build (tests/vt/test_rocm_arch.cpp) on a
// machine with no AMD GPU, while `rocm_backend.hip` stays glue that reads four
// attributes and calls this.

// The four probed device attributes this decision reads, and nothing else.
// Named rather than passed as four bools so a caller cannot transpose two.
struct RocmMemoryAttributes {
  bool integrated = false;
  bool managed_memory = false;
  bool concurrent_managed_access = false;
  // hipDeviceAttributePageableMemoryAccess. THE attribute #2511 turns on: a
  // device reporting 0 cannot take a RECOVERABLE page fault, so a migratable
  // managed page that the driver moves out from under a live queue is a hard
  // memory violation rather than a fault-and-retry.
  bool pageable_memory_access = false;
};

// `VT_ROCM_MANAGED_ALLOC`. Tri-state because a two-state knob cannot express
// the pre-fix arm once the default moves, and a single-binary interleaved A/B
// is the only shape that measures anything on a board that GPU-resets between
// legs (.agents/specs/rocm-gfx1151-q4k-hang.md, W9/W10).
enum class RocmManagedAllocOverride {
  kUnset = 0,   // the narrowed default: managed only where the part can fault and recover
  kForceOff,    // never managed, on any device
  kForceOn,     // managed wherever the ATTRIBUTES allow it: the pre-#2511 behaviour
};

// Parses the environment value; `nullptr` and every unrecognised string mean
// kUnset, so a typo takes the shipped default rather than a surprising arm.
constexpr RocmManagedAllocOverride ParseManagedAllocOverride(std::string_view v) {
  if (v == "0") return RocmManagedAllocOverride::kForceOff;
  if (v == "1") return RocmManagedAllocOverride::kForceOn;
  return RocmManagedAllocOverride::kUnset;
}

// What the registrar needs, resolved in one place so the allocator and the
// capability claim cannot drift apart.
struct RocmMemoryPolicy {
  // Take the hipMallocManaged branch in Backend::Alloc.
  bool managed_alloc = false;
  // UnifiedMemory() AND DeviceMemoryIsHostAddressable(). Written OVER
  // `managed_alloc`, never over the managed CAPABILITY: the sentence the
  // capability makes is "every block Alloc handed out is host-addressable",
  // and reading it off a branch the allocator did not take is exactly how the
  // two answers could disagree.
  bool unified_memory = false;
  // Non-null only when the managed branch was WITHHELD from a device that has
  // the managed attributes, which is the one case where a false
  // host-addressability answer is a deliberate narrowing rather than a
  // discrete card's ordinary state. `Backend::HostAddressabilityNote()`
  // returns it and the reference tier's refusal path quotes it, so a reader
  // who loses a CPU fallback learns why at the point of failure.
  const char* host_addressability_note = nullptr;
};

// The note text, exported so a test can assert the exact string rather than a
// substring it chose itself.
inline constexpr const char* kRocmManagedWithheldNote =
    "this ROCm device reports hipDeviceAttributePageableMemoryAccess = 0, so it "
    "cannot take a recoverable page fault; vllm.cpp therefore allocates with "
    "hipMalloc instead of hipMallocManaged and makes no host-addressability "
    "claim (issue #2511, which measured 17 GPU faults in 21 legs on the managed "
    "branch and 0 in 21 without it). Set VT_ROCM_MANAGED_ALLOC=1 to restore the "
    "previous managed behaviour, at that risk";

constexpr RocmMemoryPolicy ResolveMemoryPolicy(const RocmMemoryAttributes& a,
                                               RocmManagedAllocOverride ov) {
  RocmMemoryPolicy p;
  // The managed API contract needs all three attributes. The override cannot
  // conjure a capability, so a discrete card (Integrated = 0) keeps the branch
  // provably dead whatever the environment says.
  const bool managed_capable =
      a.integrated && a.managed_memory && a.concurrent_managed_access;
  if (managed_capable) {
    switch (ov) {
      case RocmManagedAllocOverride::kForceOff:
        p.managed_alloc = false;
        break;
      case RocmManagedAllocOverride::kForceOn:
        p.managed_alloc = true;
        break;
      case RocmManagedAllocOverride::kUnset:
        // THE #2511 NARROWING. Managed memory is migratable, and migration
        // under a live queue is only survivable where the device can fault and
        // recover. gfx1151 and gfx1103 report this 0.
        p.managed_alloc = a.pageable_memory_access;
        break;
    }
  }
  // Two independent grounds, both API-anchored: the hipMallocManaged contract,
  // or the W0 CUDA-shaped conjunction (cuda_backend.cu). Under kUnset the first
  // now IMPLIES the second, so the unified claim collapses to exactly the W0
  // conjunction and the managed branch no longer widens it at all.
  p.unified_memory = p.managed_alloc || (a.pageable_memory_access && a.integrated);
  if (managed_capable && !p.managed_alloc && !p.unified_memory) {
    p.host_addressability_note = kRocmManagedWithheldNote;
  }
  return p;
}

}  // namespace vt::rocm
