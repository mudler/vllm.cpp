// Quantized prefill admission stays separate from the attention policy.
#pragma once

#include <string_view>

namespace vt::rocm {

// rocWMMA owns the different gfx11/gfx12 operand layouts. Admit only the
// measured gfx1100 device and existing gfx1200/gfx1201 devices. Attention
// keeps its gfx12 policy because its gfx11 device body is still excluded.
constexpr bool GcnArchNameHasQuantWmma(std::string_view gcn_arch) {
  const auto colon = gcn_arch.find(':');
  const auto stem = gcn_arch.substr(0, colon);
  // Require an exact stem followed by HIP target features with explicit signs.
  if (stem != "gfx1100" && stem != "gfx1200" && stem != "gfx1201") return false;
  if (colon == std::string_view::npos) return true;
  auto features = gcn_arch.substr(colon + 1);
  bool xnack = false, sramecc = false;
  while (!features.empty()) {
    const auto next = features.find(':');
    const auto feature = features.substr(0, next);
    if ((feature == "xnack+" || feature == "xnack-") && !xnack) xnack = true;
    else if ((feature == "sramecc+" || feature == "sramecc-") && !sramecc) sramecc = true;
    else return false;
    if (next == std::string_view::npos) return true;
    features.remove_prefix(next + 1);
  }
  return false;
}

}  // namespace vt::rocm
