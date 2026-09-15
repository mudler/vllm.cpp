#pragma once

#include <string_view>

#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {

// The gfx1100 decode prerequisite is independent of the prefill control.
constexpr bool GcnArchNameHasGemmaDecodeWmma(std::string_view arch) {
  return arch == "gfx1100" || arch.starts_with("gfx1100:");
}

// Attention admission is separate from quantized WMMA admission. Every admitted
// target must compile the attention body as well as pass this host predicate.
constexpr bool GcnArchNameHasSharedKAttentionWmma(std::string_view arch) {
  return GcnArchNameHasGemmaDecodeWmma(arch) || GcnArchNameIsGfx12PrefillWmma(arch);
}

// Default on for validated gfx1100 and the existing gfx12 targets. Preserve
// the environment control's first-character rule.
constexpr bool SharedKAttentionWmmaEnabled(std::string_view arch, const char* override_value) {
  return GcnArchNameHasSharedKAttentionWmma(arch) && (!override_value || override_value[0] != '0');
}

}  // namespace vt::rocm
