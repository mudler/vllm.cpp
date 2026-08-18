// The `vt::CopyOutput` customization point for a break point whose destination
// is a POOLED DEVICE BUFFER — the port of `_copy_output`
// (`breakable_cuda_graph.py:172-201` @ SGLang pin `f63458b5be`).
//
// Row ENG-CUDAGRAPH-BREAK W1, issue #1192, parent #1163.
//
// It lives in a header rather than beside its one call site for two reasons.
// Argument-dependent lookup has to find it wherever a dense model registers an
// attention break point, and W2 through W5 migrate eight more drivers that share
// this exact destination type. And a customization point that only one
// translation unit can see is one nobody can ASSERT is visible: the G2 gate
// static-asserts `vt::detail::HasCopyOutput<std::optional<DBuf>>` against this
// declaration, which is what turns "the writeback branch is selected" from a
// claim into a compile-time fact.
#pragma once

#include <optional>

#include "vllm/model_executor/models/dense_device_glue.h"  // DBuf
#include "vt/backend.h"
#include "vt/breakable_graph.h"

namespace vllm {
namespace dense_attn {

// Why it cannot be a rebind. On replay N the attention call returns a FRESH
// `DBuf` from the device pool, whose address is not the one the FOLLOWING
// segment baked at capture time. Rebinding the destination would leave that
// segment reading capture-time data forever while the break wrote elsewhere —
// wrong numerics, not a fault, and invisible to `compute-sanitizer` (spec D9).
// So the bytes are copied INTO the destination and the destination's address
// never moves.
inline void CopyOutput(vt::Backend& b, vt::Queue& q, std::optional<DBuf>& dst,
                       const std::optional<DBuf>& src) {
  if (!dst.has_value() || !src.has_value()) return;
  vt::CopyOutput(b, q, dst->t(), src->t());
}

}  // namespace dense_attn
}  // namespace vllm
