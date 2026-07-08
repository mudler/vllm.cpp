// Force-included (nvcc -include) ahead of every vendored FA-2 translation unit.
// vLLM's flash-attention sources rely on a handful of standard headers being
// pulled in transitively by the torch/ATen includes we replaced with torch-free
// stubs (hardware_info.h uses fprintf/stderr/exit; utils.h uses std::optional and
// std::min/std::max in the paged-KV slice helper). Providing them here keeps the
// upstream FA sources byte-for-byte pristine (they are auto-generated) while the
// build stays torch-free. Source tree: vllm-project/flash-attention @ 2c839c33.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <vector>
#include <optional>
#include <algorithm>
