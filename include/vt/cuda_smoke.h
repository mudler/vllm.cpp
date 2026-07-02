#pragma once

namespace vt::cuda {
// Adds two ints on the GPU. Build-pipeline smoke only; not part of the vt API.
int SmokeAdd(int a, int b);
}  // namespace vt::cuda
