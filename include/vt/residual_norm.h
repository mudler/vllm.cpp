// vllm.cpp shared representation of the compiled vLLM residual expression.
// vllm/ir/ops/layernorm.py:44-62 @ e126687a9a828d513c01a07cd69f025f27d63280.
#pragma once
#include <cstdint>

namespace vt {

// Arithmetic is FP32 in this order. Every activation operand remains BF16.
enum class ResidualNormExpr : uint8_t {
  kAdd,           // a + base: two activation operands
  kDeltaPlusAdd,  // delta + (a + base): three activation operands
};

struct ResidualNormDesc {
  ResidualNormExpr expression = ResidualNormExpr::kAdd;
  bool materialize_residual = false;
  // All inputs are read-only unless an exact, explicitly permitted output alias
  // names them. Partial overlaps and overlap between outputs are always invalid.
  bool output_alias_delta = false;
  bool residual_alias_base = false;
};

// A reference registration does not opt a production backend into this policy.
enum class ResidualNormPolicy : uint8_t {
  kMaterialized,
  kCompiledExpression,
};

}  // namespace vt
