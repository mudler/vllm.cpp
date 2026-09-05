// See `include/vllm/model_executor/models/dense_exl3_linear.h`. This
// translation unit exists so that the scheme header, and through it
// `dense_attn_block.h`, is included HERE and not in `qwen3_5.cpp`.
#include "vllm/model_executor/models/dense_exl3_linear.h"

#include "vllm/model_executor/layers/quantization/exl3.h"

namespace vllm {
namespace dense_exl3 {

dense_attn::DBuf Linear(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_w, const Exl3Weight& exl3_w,
                        vt::DType out_dtype) {
  return layers::MakeLinearMethod(bf16_w, exl3_w)->Apply(d, x, out_dtype);
}

dense_attn::DBuf GateUp(dense_attn::Dev d, const vt::Tensor& x,
                        const OwnedTensor& bf16_gate_up, const Exl3Weight& gate,
                        const Exl3Weight& up, int64_t intermediate) {
  return layers::MakeMlpGateUpMethod(bf16_gate_up, gate, up, intermediate)
      ->Apply(d, x);
}

}  // namespace dense_exl3
}  // namespace vllm
