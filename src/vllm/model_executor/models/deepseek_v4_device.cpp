// DeepSeek-V4-Flash W7-device — the OpProvider-seam resolvers for the four V4
// device kernel families. Always compiled (CPU + CUDA + ROCm); it holds NO device
// code — it only looks up the per-family kernels-struct a device TU registered
// under kDeepseekV4{Mhc,Dsa,Compressor,Moe}. MHC resolves on kCUDA OR kROCm
// (O34); the other three families stay kCUDA-only. On a CPU-only build nothing
// is registered, so GetOp() throws and ForwardDevice surfaces a clean
// device-only error. See deepseek_v4_device.h.
#include "vllm/model_executor/models/deepseek_v4_device.h"

#include "vt/ops.h"  // OpId, GetOp, OpRegistered

namespace vllm::deepseek_v4 {

const MhcDeviceKernels* MhcDevice() {
  if (vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA))
    return static_cast<const MhcDeviceKernels*>(
        vt::GetOp(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA));
  return static_cast<const MhcDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kROCM));
}
const DsaDeviceKernels* DsaDevice() {
  return static_cast<const DsaDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Dsa, vt::DeviceType::kCUDA));
}
const CompressorDeviceKernels* CompressorDevice() {
  return static_cast<const CompressorDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Compressor, vt::DeviceType::kCUDA));
}
const MoeDeviceKernels* MoeDevice() {
  return static_cast<const MoeDeviceKernels*>(
      vt::GetOp(vt::OpId::kDeepseekV4Moe, vt::DeviceType::kCUDA));
}

bool V4DeviceKernelsAvailable() {
  const bool mhc = vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kCUDA) ||
                   vt::OpRegistered(vt::OpId::kDeepseekV4Mhc, vt::DeviceType::kROCM);
  return mhc &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Dsa, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Compressor, vt::DeviceType::kCUDA) &&
         vt::OpRegistered(vt::OpId::kDeepseekV4Moe, vt::DeviceType::kCUDA);
}

}  // namespace vllm::deepseek_v4
