// The QuantizationConfig / QuantizeMethod seam — the coarse per-accelerator
// policy layer vLLM keeps in `model_executor/layers/quantization/` and every
// model + platform reuses, ported to vt::-native C++.
//
// UPSTREAM (ported FROM, ground-every-impl rule):
//   vllm/model_executor/layers/quantization/base_config.py:20-46
//     class QuantizeMethodBase (create_weights + apply)
//   vllm/model_executor/layers/quantization/base_config.py:87-180
//     class QuantizationConfig + get_quant_method(layer, prefix)
//
// WHY THIS EXISTS (accelerator-seam audit §5, work row S4). Scheme selection
// used to be a per-model TENSOR-NAME PROBE (`qwen3.h:60 IsNvfp4()`), with the
// compute path chosen by inline `device == kCUDA && !weight.Empty()` checks
// scattered through the model forwards. That is exactly the "adding an
// accelerator costs an edit at every device gate in every model" tangle the
// user asked to remove. vLLM answers "which kernel family does this
// (scheme, device) use" ONCE, in a QuantizeMethod selected at load, so the
// model forward calls `linear_method.apply(x)` and never asks which scheme or
// which device. This header is that seam.
//
// SEPARATION OF CONCERNS (audit Risks/decisions 6, binding rule):
//   * POLICY — which SCHEME (bf16 / nvfp4-w4a16 / …) — lives HERE, in the
//     QuantizeMethod selected by the QuantizationConfig. This mirrors an
//     upstream QuantizationConfig.get_quant_method.
//   * IMPLEMENTATION — which KERNEL realizes an op on this device — lives in
//     the `vt::OpProvider` table. A method's apply() asks the op table
//     `vt::OpRegistered(op, device)` whether the fast kernel is available here
//     instead of testing `device == kCUDA`, so the same code is correct on any
//     accelerator that registers the op. The two seams never overlap.
#pragma once

#include "vt/dtype.h"  // DType

namespace vllm {
namespace layers {

// QuantizeMethodBase (base_config.py:20). The common base of every
// (maybe-quantized) method that owns how a layer's weights are laid out and
// applied. `create_weights` in vLLM lands at LOAD time in our existing
// safetensors/GGUF loaders (they already materialize the packed/scale buffers),
// so the base here carries only the compute-time contract that the model
// forward calls; concrete apply signatures are added by the typed sub-bases
// (LinearMethodBase below) because our tensors are strongly typed vt::Tensor
// rather than a duck-typed torch.nn.Module.
class QuantizeMethodBase {
 public:
  virtual ~QuantizeMethodBase() = default;
};

}  // namespace layers
}  // namespace vllm
