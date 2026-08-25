// The (kLtx2Vae, device) resolver, in its own TU so the vt::GetOp cast links in
// CPU-only builds as well as CUDA ones — the same reason ltx2_device_resolve.cpp
// and minimax_h3_device_resolve.cpp are separate files.
//
// Row: LTX25-VAE-DEVICE-RESIDENCY, #1451.
#include "vllm/model_executor/models/ltx2_video_vae_kernels.h"

#include "vt/ops.h"

namespace vllm::ltx2_vae {

const Ltx2VaeDeviceKernels* Ltx2VaeDevice(vt::DeviceType device) {
  return static_cast<const Ltx2VaeDeviceKernels*>(vt::GetOp(vt::OpId::kLtx2Vae, device));
}

}  // namespace vllm::ltx2_vae
