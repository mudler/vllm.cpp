// Compiled Gemma expressions from vLLM e126687a9a828d513c01a07cd69f025f27d63280.
// See model_executor/models/gemma3.py, layers/layernorm.py and layers/activation.py.
#pragma once
#include "vt/ops.h"

namespace vt {

// The embedding scale participates in the variance before BF16 narrowing.
// The normalized numerator and residual both read the narrowed scaled input.
struct ScaledRmsNormArgs {
  float scale = 1.f;
  float eps = 1e-6f;
};

// N(a,wa)+base, or N(delta,wd)+(N(a,wa)+base), then another Gemma norm.
// N keeps FP32 arithmetic through both products. All operands remain BF16.
struct SandwichNormInputs {
  const Tensor& a;
  const Tensor& base;
  const Tensor& a_weight;
  const Tensor& weight;
  const Tensor* delta = nullptr;
  const Tensor* delta_weight = nullptr;
};

using ScaledRmsNormFn = void (*)(Queue&, Tensor&, const Tensor&, const Tensor&,
                                 const ScaledRmsNormArgs&, Tensor&);
using SandwichRmsNormFn = void (*)(Queue&, Tensor&, const SandwichNormInputs&, float, Tensor*);

// Typed fusion bindings. The optional sandwich residual can alias base exactly;
// all other inputs are read-only and both outputs must be disjoint.
void FusedChain(Queue& q, Tensor& out, const Tensor& input, const Tensor& weight,
                const ScaledRmsNormArgs& args, Tensor& residual);
void FusedChain(Queue& q, Tensor& out, const SandwichNormInputs& inputs, float eps,
                Tensor* residual = nullptr);

// ROCm's compiled GeluAndMul.forward_native uses erf and narrows after the
// product with up. The existing tanh/rounded operation retains its contract.
void CompiledGeluErfMul(Queue& q, Tensor& out, const Tensor& packed);

}  // namespace vt
