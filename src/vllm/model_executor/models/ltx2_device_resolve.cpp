// The (kLtx2, device) resolver, in its own TU so the vt::GetOp cast links in
// CPU-only builds as well as CUDA ones — exactly why minimax_h3_device_resolve.cpp
// is a separate file too.
#include "vllm/model_executor/models/ltx2_device.h"

#include "vt/ops.h"

namespace vllm::ltx2 {

const Ltx2DeviceKernels* Ltx2Device(vt::DeviceType device) {
  return static_cast<const Ltx2DeviceKernels*>(vt::GetOp(vt::OpId::kLtx2, device));
}

bool Ltx2DeviceKernelsAvailable(vt::DeviceType device) {
  return vt::OpRegistered(vt::OpId::kLtx2, device);
}

}  // namespace vllm::ltx2
