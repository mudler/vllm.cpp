// Syntax-check stub for mlx/ops.h. See ../README.md.
//
// Three ops. `matmul` is the delegated computation; `transpose` expresses the
// kMatmulBT weight orientation as a VIEW rather than a materialized transpose;
// `astype` converts when MLX's output dtype is not ours.
#ifndef VT_METAL_STUBS_MLX_OPS_H_
#define VT_METAL_STUBS_MLX_OPS_H_

#include <vector>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"

namespace mlx::core {

// MLX spells the trailing argument `StreamOrDevice`, a variant. The provider
// only ever passes a Stream, so the alias is narrowed to the case in use rather
// than modelling a variant the gate would not exercise.
using StreamOrDevice = Stream;

array matmul(const array& a, const array& b, StreamOrDevice s);
array transpose(const array& a, std::vector<int> axes, StreamOrDevice s);
array astype(const array& a, Dtype dtype, StreamOrDevice s);

}  // namespace mlx::core

#endif  // VT_METAL_STUBS_MLX_OPS_H_
