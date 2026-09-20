// vllm.cpp ORIGINAL — the Tenstorrent WEIGHT-RESIDENCY lever. One env
// variable selects which device-side weight residency the staging seam
// converts an admitted weight to, so a new variant (bfp4, per-role splits)
// is a new VALUE here, not a new flag. Spec:
// .agents/specs/tenstorrent-bfp-weight-residency.md.
//
// VT_TT_WEIGHT_RESIDENCY takes:
//   off   (default; also unset or empty) — byte-identical bf16 staging
//   bfp8  — bf16 upload, device typecast to BFLOAT8_B at first staging
//   bfp4  — RESERVED, not implemented; refused by name at the staging seam
//
// Header-only and dependency-free so both the loader policy
// (gguf_keep_quant.cpp) and the vt Tenstorrent staging seam
// (tenstorrent_residency.cpp) parse the SAME value with the SAME code.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vllm {

enum class TtWeightResidency {
  kOff,
  kBfp8,
  kBfp4,  // reserved; every consumer refuses it by name for now
};

// Parse one VT_TT_WEIGHT_RESIDENCY value. A value this build does not know is
// a hard, loud abort naming the offending string — a silently-ignored lever
// value is how an operator believes a residency is live when it is not.
inline TtWeightResidency ParseTtWeightResidency(const char* s) {
  if (s == nullptr || s[0] == '\0' || std::strcmp(s, "off") == 0 ||
      std::strcmp(s, "0") == 0)
    return TtWeightResidency::kOff;
  if (std::strcmp(s, "bfp8") == 0) return TtWeightResidency::kBfp8;
  if (std::strcmp(s, "bfp4") == 0) return TtWeightResidency::kBfp4;
  std::fprintf(stderr,
               "vllm.cpp: VT_TT_WEIGHT_RESIDENCY: unknown value '%s' "
               "(expected off|bfp8|bfp4)\n",
               s);
  std::abort();
}

}  // namespace vllm
