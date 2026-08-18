// CPU seam gate for #317 Gemma4 ROCm FP8 MoE stack.
// Does NOT require a ROCm device: proves portable fused_ops declarations link
// and the documented VT_GEMMA4_ / VT_ATTN_ env knobs parse inertly on CPU.
#include <doctest/doctest.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/fused_ops.h"

namespace {

struct EnvRestorer {
  const char* key;
  bool had = false;
  std::string prev;
  explicit EnvRestorer(const char* k) : key(k) {
    if (const char* v = std::getenv(k)) {
      had = true;
      prev = v;
    }
  }
  ~EnvRestorer() {
    if (had) ::setenv(key, prev.c_str(), 1);
    else ::unsetenv(key);
  }
};

int EnvInt(const char* key, int def) {
  const char* e = std::getenv(key);
  if (e == nullptr || e[0] == '\0') return def;
  char* end = nullptr;
  long v = std::strtol(e, &end, 10);
  if (end == e) return def;
  return static_cast<int>(v);
}

}  // namespace

TEST_CASE("gemma4 rocm fp8 seams: fused_ops ExpertGeGLU symbols link on CPU") {
  // Taking addresses forces the linker to resolve the portable wrappers.
  // Bodies may no-op or refuse without ROCm — that is fine for this gate.
  auto* p0 = &vt::ExpertGeGLUFp8TopKM1;
  auto* p1 = &vt::ExpertGeGLUFp8TopKIndexed;
  auto* p2 = &vt::MatmulBTFp8Channel;
  auto* p3 = &vt::DequantFp8ChannelBf16;
  auto* p4 = &vt::PrewarmExpertGeGLUFp8TopK;
  CHECK(p0 != nullptr);
  CHECK(p1 != nullptr);
  CHECK(p2 != nullptr);
  CHECK(p3 != nullptr);
  CHECK(p4 != nullptr);
}

TEST_CASE("gemma4 rocm fp8 seams: recipe env knobs parse inert defaults") {
  // Defaults match docs/ENVIRONMENT.md / lab recipe when unset.
  {
    EnvRestorer a("VT_GEMMA4_FP8_HW_CVT");
    EnvRestorer b("VT_ATTN_DECODE_KV_SPLITS");
    EnvRestorer c("VT_ATTN_DECODE_SLIDE_SPLITS");
    EnvRestorer d("VT_ATTN_DECODE_SPLIT_WARPS");
    ::unsetenv("VT_GEMMA4_FP8_HW_CVT");
    ::unsetenv("VT_ATTN_DECODE_KV_SPLITS");
    ::unsetenv("VT_ATTN_DECODE_SLIDE_SPLITS");
    ::unsetenv("VT_ATTN_DECODE_SPLIT_WARPS");
    // Unset → code defaults are env-read at runtime inside HIP; here we only
    // document the *recipe* integers the lab pins (not process-wide defaults).
    CHECK(EnvInt("VT_GEMMA4_FP8_HW_CVT", 1) == 1);
    CHECK(EnvInt("VT_ATTN_DECODE_KV_SPLITS", 16) == 16);
    CHECK(EnvInt("VT_ATTN_DECODE_SLIDE_SPLITS", 8) == 8);
    CHECK(EnvInt("VT_ATTN_DECODE_SPLIT_WARPS", 12) == 12);
  }
  {
    EnvRestorer a("VT_GEMMA4_FP8_HW_CVT");
    ::setenv("VT_GEMMA4_FP8_HW_CVT", "0", 1);
    CHECK(EnvInt("VT_GEMMA4_FP8_HW_CVT", 1) == 0);
  }
}

// #1205: `vt::MatmulBTAlphaBeta` has NO CUDA implementation. The only one in the
// tree is `rocm::MatmulBTAlphaBetaRocm` (rocm_matmul_hipblaslt.hip:516), reached
// through a `#if defined(VLLM_CPP_HIP)` + `kROCM` guard in fused_ops.cpp; every
// other device falls through to a throw. That is the actual blocker under #1126:
// adding `CudaBackend::DeviceMemoryInfo` alone would have let the Gemma4 expert
// LRU admit (gemma4_moe.cpp:587,597), take ExpertGeGLUDeviceAccum at :1509, and
// reach this call at :90 — outside the upload's try/catch (:585-607), so
// mid-decode. `EnsureGemma4Fp8ExpertOnDevice` now refuses that upload at :571
// when `vt::HasMatmulBTAlphaBeta` is false, so the throw below is the backstop
// and not the guard; the capability gate for the refusal-before-upload is
// tests/vllm/models/test_gemma4_moe_device_arm_guard.cpp, which enters through
// `vllm::RunGemma4Moe`. What THIS file pins is the message contract a CUDA
// implementation has to satisfy — a unit contract, since with the guard in place
// no production run off ROCm reaches the throw.
//
// AGENTS.md requires an unimplemented arm to refuse with a message that NAMES the
// missing part. "ROCm-only in this build" names neither the device that asked nor
// where the implementation would go, so a caller who hits it on CUDA cannot tell a
// missing kernel from a missing build flag. This pins the contract; when the CUDA
// arm is written, this is what it has to satisfy.
TEST_CASE("gemma4 rocm fp8 seams: MatmulBTAlphaBeta refuses a non-ROCm queue by name") {
  // The refusal reads `q.device.type` and nothing else, so a CPU build can pose
  // as any device here. Buffers are never dereferenced on the refusing path.
  auto refusal_for = [](vt::DeviceType type) -> std::string {
    vt::Queue q;
    q.device.type = type;
    q.device.index = 0;
    try {
      vt::MatmulBTAlphaBeta(q, nullptr, nullptr, nullptr, /*M=*/1, /*N=*/1, /*K=*/1,
                            /*alpha=*/1.f, /*beta=*/0.f, vt::DType::kBF16);
    } catch (const std::runtime_error& e) {
      return e.what();
    }
    return std::string();
  };

  // CUDA is the device #1126 would wake, so it is the one the message must name.
  const std::string cuda_msg = refusal_for(vt::DeviceType::kCUDA);
  REQUIRE_FALSE(cuda_msg.empty());
  INFO("cuda refusal: " << cuda_msg);
  CHECK(cuda_msg.find("MatmulBTAlphaBeta") != std::string::npos);
  CHECK(cuda_msg.find("cuda") != std::string::npos);
  CHECK(cuda_msg.find("rocm") != std::string::npos);
  CHECK(cuda_msg.find("1205") != std::string::npos);

  // Not a CUDA special case: every non-ROCm device refuses, naming itself.
  for (const auto type : {vt::DeviceType::kCPU, vt::DeviceType::kVULKAN,
                          vt::DeviceType::kMETAL}) {
    const std::string msg = refusal_for(type);
    // std::string, not the char* — doctest stringifies a bare char* as a bool.
    INFO("refusal for " << std::string(vt::DeviceTypeName(type)) << ": " << msg);
    REQUIRE_FALSE(msg.empty());
    CHECK(msg.find(vt::DeviceTypeName(type)) != std::string::npos);
    CHECK(msg.find("1205") != std::string::npos);
  }

  // kROCM is the one device the four above make the message look right FOR and
  // wrong ABOUT. A kROCM queue in a build without -DVLLM_CPP_HIP falls through
  // the same `#if` and refuses too — but the 'rocm' kernel EXISTS and was
  // compiled out, so telling that caller "no implementation ... see #1205" would
  // send them to write a kernel the tree already has. The two absences get two
  // messages, and this pins the difference in both directions.
  {
    vt::Queue rocm_q;
    rocm_q.device.type = vt::DeviceType::kROCM;
    rocm_q.device.index = 0;
    // Build-agnostic: on a HIP build the arm is present and the call would run a
    // real GEMM over these null pointers, so ask the same predicate the dispatch
    // asks rather than repeating the `#if` here.
    if (!vt::HasMatmulBTAlphaBeta(rocm_q)) {
      const std::string rocm_msg = refusal_for(vt::DeviceType::kROCM);
      REQUIRE_FALSE(rocm_msg.empty());
      INFO("rocm refusal: " << rocm_msg);
      CHECK(rocm_msg.find("MatmulBTAlphaBeta") != std::string::npos);
      CHECK(rocm_msg.find("rocm") != std::string::npos);
      // Names the ABSENT BUILD FLAG, which is what is actually missing here.
      CHECK(rocm_msg.find("VLLM_CPP_HIP") != std::string::npos);
      // And does NOT claim the kernel is unwritten, nor point at the CUDA issue.
      CHECK(rocm_msg.find("1205") == std::string::npos);
      CHECK(rocm_msg.find("no implementation for device") == std::string::npos);
    }
  }
}

// The predicate the guard in `EnsureGemma4Fp8ExpertOnDevice` keys on. It has to
// agree with the dispatch EXACTLY — a caller that trusts `true` and then throws
// is worse than no predicate — so this asserts the agreement over every device
// rather than reading the two conditions and calling them equal.
TEST_CASE("gemma4 rocm fp8 seams: HasMatmulBTAlphaBeta agrees with the dispatch") {
  int checked = 0;
  for (const auto type : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA, vt::DeviceType::kROCM,
                          vt::DeviceType::kVULKAN, vt::DeviceType::kMETAL,
                          vt::DeviceType::kXPU}) {
    vt::Queue q;
    q.device.type = type;
    q.device.index = 0;
    const bool has = vt::HasMatmulBTAlphaBeta(q);
    INFO("device " << std::string(vt::DeviceTypeName(type)) << " has=" << has);
    if (has) {
      // Do NOT call through: a present arm would run a real GEMM over null
      // pointers. The absence direction is the one this guard depends on.
      ++checked;
      continue;
    }
    bool threw = false;
    try {
      vt::MatmulBTAlphaBeta(q, nullptr, nullptr, nullptr, /*M=*/1, /*N=*/1, /*K=*/1,
                            /*alpha=*/1.f, /*beta=*/0.f, vt::DType::kBF16);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);
    ++checked;
  }
  // A loop that ran zero times would report SUCCESS with nothing examined.
  CHECK(checked == 6);
}
