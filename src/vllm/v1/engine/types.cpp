// Ported from: vllm/v1/engine/__init__.py + vllm/v1/outputs.py @ e24d1b24
//
// The EngineCore I/O types are header-only value carriers (see
// include/vllm/v1/engine/types.h). This translation unit exists so the header
// is compiled standalone (self-containment check) and has a home in the build.
#include "vllm/v1/engine/types.h"

#include <cstdlib>

namespace vllm::v1 {

// The async-decode toggle, read once. Shared by the runner and the EngineCore so
// they never disagree on whether the depth-2 pipeline is engaged. Default OFF.
bool AsyncDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_ASYNC_DECODE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

}  // namespace vllm::v1
