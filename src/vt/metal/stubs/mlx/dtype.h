// Syntax-check stub for mlx/dtype.h. See ../README.md.
//
// The provider names exactly three of MLX's dtypes and compares two Dtypes for
// inequality (`mc.dtype() != out_dt`). Nothing else of `Dtype` is used, so
// nothing else is declared.
#ifndef VT_METAL_STUBS_MLX_DTYPE_H_
#define VT_METAL_STUBS_MLX_DTYPE_H_

#include <cstdint>

namespace mlx::core {

struct Dtype {
  enum class Val : uint8_t { kFloat32, kFloat16, kBfloat16 };
  Val val = Val::kFloat32;
  constexpr bool operator==(const Dtype& other) const { return val == other.val; }
  constexpr bool operator!=(const Dtype& other) const { return val != other.val; }
};

inline constexpr Dtype float32{Dtype::Val::kFloat32};
inline constexpr Dtype float16{Dtype::Val::kFloat16};
inline constexpr Dtype bfloat16{Dtype::Val::kBfloat16};

}  // namespace mlx::core

#endif  // VT_METAL_STUBS_MLX_DTYPE_H_
