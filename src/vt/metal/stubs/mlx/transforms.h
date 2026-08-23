// Syntax-check stub for mlx/transforms.h. See ../README.md.
//
// `eval` is THE boundary: everything the provider builds is at most three graph
// nodes, and this is where MLX encodes and runs them. The real signature is a
// variadic template over arrays, which is why it is declared that way here --
// a plain `void eval(array&)` would also compile the one call site, but would
// stop matching MLX's shape for no gain.
#ifndef VT_METAL_STUBS_MLX_TRANSFORMS_H_
#define VT_METAL_STUBS_MLX_TRANSFORMS_H_

#include "mlx/array.h"

namespace mlx::core {

template <class... Arrays>
void eval(Arrays&&... outputs);

}  // namespace mlx::core

#endif  // VT_METAL_STUBS_MLX_TRANSFORMS_H_
