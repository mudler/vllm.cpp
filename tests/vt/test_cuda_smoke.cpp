#include <doctest/doctest.h>
#ifdef VLLM_CPP_CUDA
#include "vt/cuda_smoke.h"

TEST_CASE("CUDA smoke: kernel launch on device") { CHECK(vt::cuda::SmokeAdd(2, 40) == 42); }
#else
TEST_CASE("CUDA smoke: skipped (CPU-only build)") { CHECK(true); }
#endif
