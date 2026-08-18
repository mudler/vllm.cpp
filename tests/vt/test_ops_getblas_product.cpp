// #837 live product GetBlas probe (research 4d82/e819).
// Calls vt::rocm::ProductGetBlasHandle → file-local GetBlas.
// Missing HIP_VISIBLE_DEVICES or <2 devices: exit 77 (CTest SKIP), never SUCCESS.
#include <cstdio>
#include <cstdlib>

#include <doctest/doctest.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "vt/rocm/rocm_getblas_product.h"

namespace {

[[noreturn]] void SkipNotRun(const char* why) {
  std::fprintf(stderr,
               "\n*** GATE NOT RUN — SKIPPED (exit 77), this is NOT a pass ***\n"
               "%s\n",
               why);
  std::exit(77);
}

void RequireHip(hipError_t err, const char* what) {
  REQUIRE(err == hipSuccess);
  (void)what;
}

}  // namespace

TEST_CASE("getblas product GetBlas 0-1-0-1 handle identity") {
  const char* env = std::getenv("HIP_VISIBLE_DEVICES");
  if (env == nullptr || env[0] == '\0') {
    SkipNotRun("HIP_VISIBLE_DEVICES empty");
  }
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 2) {
    SkipNotRun("need >= 2 HIP devices");
  }

  hipStream_t s0 = nullptr;
  hipStream_t s1 = nullptr;
  RequireHip(hipSetDevice(0), "set0");
  RequireHip(hipStreamCreate(&s0), "s0");
  RequireHip(hipSetDevice(1), "set1");
  RequireHip(hipStreamCreate(&s1), "s1");

  const hipblasHandle_t h0a = vt::rocm::ProductGetBlasHandle(0, s0);
  int cur = -1;
  RequireHip(hipGetDevice(&cur), "get after 0");
  REQUIRE(h0a != nullptr);
  CHECK(cur == 0);
  hipStream_t bound = nullptr;
  REQUIRE(hipblasGetStream(h0a, &bound) == HIPBLAS_STATUS_SUCCESS);
  CHECK(bound == s0);

  const hipblasHandle_t h1a = vt::rocm::ProductGetBlasHandle(1, s1);
  RequireHip(hipGetDevice(&cur), "get after 1");
  REQUIRE(h1a != nullptr);
  CHECK(h1a != h0a);
  CHECK(cur == 1);
  REQUIRE(hipblasGetStream(h1a, &bound) == HIPBLAS_STATUS_SUCCESS);
  CHECK(bound == s1);

  const hipblasHandle_t h0b = vt::rocm::ProductGetBlasHandle(0, s0);
  RequireHip(hipGetDevice(&cur), "get after 0 revisit");
  CHECK(h0b == h0a);
  CHECK(cur == 0);
  REQUIRE(hipblasGetStream(h0b, &bound) == HIPBLAS_STATUS_SUCCESS);
  CHECK(bound == s0);

  const hipblasHandle_t h1b = vt::rocm::ProductGetBlasHandle(1, s1);
  RequireHip(hipGetDevice(&cur), "get after 1 revisit");
  CHECK(h1b == h1a);
  CHECK(cur == 1);
  REQUIRE(hipblasGetStream(h1b, &bound) == HIPBLAS_STATUS_SUCCESS);
  CHECK(bound == s1);

  RequireHip(hipSetDevice(0), "cleanup0");
  RequireHip(hipStreamDestroy(s0), "ds0");
  RequireHip(hipSetDevice(1), "cleanup1");
  RequireHip(hipStreamDestroy(s1), "ds1");
}

TEST_CASE("getblas product capture hook is load-bearing") {
  const char* env = std::getenv("HIP_VISIBLE_DEVICES");
  if (env == nullptr || env[0] == '\0') {
    SkipNotRun("HIP_VISIBLE_DEVICES empty");
  }
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    SkipNotRun("need >= 1 HIP device");
  }

  hipStream_t s0 = nullptr;
  RequireHip(hipSetDevice(0), "set0");
  RequireHip(hipStreamCreate(&s0), "s0");
  const hipblasHandle_t h0 = vt::rocm::ProductGetBlasHandle(0, s0);
  REQUIRE(h0 != nullptr);
  CHECK(vt::rocm::ProductGetBlasStreamIsCapturing(s0) == false);

  RequireHip(hipStreamBeginCapture(s0, hipStreamCaptureModeGlobal), "begin capture");
  CHECK(vt::rocm::ProductGetBlasStreamIsCapturing(s0) == true);
  const hipblasHandle_t h0c = vt::rocm::ProductGetBlasHandle(0, s0);
  CHECK(h0c == h0);
  CHECK(vt::rocm::ProductGetBlasStreamIsCapturing(s0) == true);

  hipGraph_t graph = nullptr;
  RequireHip(hipStreamEndCapture(s0, &graph), "end capture");
  CHECK(vt::rocm::ProductGetBlasStreamIsCapturing(s0) == false);
  if (graph != nullptr) {
    RequireHip(hipGraphDestroy(graph), "destroy graph");
  }
  RequireHip(hipStreamDestroy(s0), "ds0");
}
