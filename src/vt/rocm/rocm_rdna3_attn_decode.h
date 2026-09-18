#pragma once

#include "vt/ops.h"

namespace vt::rocm {
void Rdna3AttentionDecode(Queue& queue, Tensor& out, const Tensor& query, const Tensor& key,
                          const Tensor& value, const Tensor& blocks, const Tensor& lengths,
                          const PagedAttentionArgs& args, int window_left, int window_right);
}
