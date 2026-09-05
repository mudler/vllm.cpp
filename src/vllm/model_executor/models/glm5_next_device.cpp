// GLM-5.3-Flash W9c-0 — the OpProvider-seam probe for the k-pool indexer's
// device ops. Always compiled (CPU + CUDA); it holds NO CUDA code. It only asks
// the op table whether `src/vt/cuda/cuda_glm5_next.cu` registered the pair under
// `kCUDA`. See glm5_next_device.h.
#include "vllm/model_executor/models/glm5_next_device.h"

#include "vt/ops.h"  // OpId, OpRegistered

namespace vllm::glm5_next {

bool KpoolDeviceOpsAvailable() {
  return vt::OpRegistered(vt::OpId::kGlm5NextKpoolCompress, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kGlm5NextKpoolSelect, vt::DeviceType::kCUDA);
}

}  // namespace vllm::glm5_next
