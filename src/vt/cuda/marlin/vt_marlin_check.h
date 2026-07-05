// vt marlin lift — torch-free replacement for STD_TORCH_CHECK.
//
// The vendored Marlin csrc (core/scalar_type.hpp + the moe_wna16 dispatcher)
// uses STD_TORCH_CHECK from <torch/headeronly/util/Exception.h> — the ONLY torch
// coupling in the whole slice (marlin-dropin-feasibility.md §2). We replace it
// with a throwing macro so the vendored code stays bit-for-bit otherwise.
#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace vt_marlin_detail {
inline void append_all(std::ostringstream&) {}
template <typename T, typename... Rest>
inline void append_all(std::ostringstream& os, const T& v, const Rest&... rest) {
  os << v;
  append_all(os, rest...);
}
template <typename... Args>
inline std::string make_msg(Args&&... args) {
  std::ostringstream os;
  append_all(os, args...);
  return os.str();
}
}  // namespace vt_marlin_detail

#ifndef STD_TORCH_CHECK
  #define STD_TORCH_CHECK(cond, ...)                                             \
    do {                                                                         \
      if (!(cond)) {                                                             \
        throw std::runtime_error(                                                \
            std::string("vt marlin: check failed (" #cond "): ") +              \
            vt_marlin_detail::make_msg(__VA_ARGS__));                            \
      }                                                                          \
    } while (0)
#endif
