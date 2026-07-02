#include "vllm/version.h"

namespace vllm {

std::string Version() {
  std::string v = std::to_string(VLLM_CPP_VERSION_MAJOR) + "." +
                  std::to_string(VLLM_CPP_VERSION_MINOR) + "." +
                  std::to_string(VLLM_CPP_VERSION_PATCH);
#ifdef VLLM_CPP_CUDA
  v += "+cuda";
#endif
  return v;
}

}  // namespace vllm
