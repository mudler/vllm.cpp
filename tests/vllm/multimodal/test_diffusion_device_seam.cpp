// #659 / #660 — the diffusion lane's device seam, held against a PARTIAL
// backend.
//
// The two lanes ask "where does this engine run" and the answer has three parts:
// is there an accelerator, is a backend registered for it, and can that backend
// actually run THIS model. Only the first two were being asked. The third,
// `Platform::supports_model_architecture` (include/vllm/platforms/interface.h),
// exists precisely so a partial backend can decline BY NAME — Metal registers 15
// of 75 ops and Tenstorrent a comparable slice, and both name exactly two
// architectures (src/vllm/platforms/metal.cpp:70,
// src/vllm/platforms/tenstorrent.cpp:55). On such a build a `device = 1`
// diffusion load was NO LONGER refused by name, and became a queue bind that
// dies later inside a kernel with a shape error.
//
// CUDA cannot see any of this: `supports_model_architecture` defaults to true,
// so on the box that runs the gates all three questions have always passed. That
// is exactly why the guard needs a platform this test brings with it.
//
// A SEPARATE EXECUTABLE, for the reason tests/vllm/entrypoints/
// test_device_selection.cpp is one: it registers a fake platform and backend in
// the otherwise-unused XPU slot AND over the CUDA slot, so `CurrentPlatform()`
// resolves to the fake whether or not this build has a real CUDA platform.
// Process isolation keeps those global registries out of every other suite.
//
// WHY THE FAKE REPORTS kXPU. `static_cast<vt::DeviceType>(1)` is `kCUDA`
// (include/vt/device.h: kCPU = 0, kCUDA = 1). `kXPU` is 4. So an assertion that
// device 1 resolves to kXPU is one the old integer cast could not satisfy under
// any circumstances — it is the seam being asked, not a constant being returned.
//
// WHERE EACH CASE ENTERS (AGENTS.md `## Nothing lands dead`). Both lanes are
// entered through a PRODUCTION entry point, not by calling the resolver:
// `vllm_video_engine_load` (include/vllm.h) → `LoadVideoEngine`
// (src/vllm/multimodal/video_engine.cpp) → the family registration → the
// engine's `Load`, which is where each lane's device question is asked. The
// direct `MiniMaxH3VideoDeviceType(1)` cases below stay, because they localise a
// failure — but they are NOT the reach proof, and on their own they measure a
// free function rather than a capability. Measured: replacing H3's `Load`-time
// `MiniMaxH3VideoDeviceType(params.device)` with the pre-row
// `params.device == 0 ? kCPU : kCUDA` left every direct case, and the whole of
// test_minimax_h3_video_fold, GREEN. The case that enters at `LoadVideoEngine`
// is the one that goes RED.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/multimodal/ltx2_video.h"
#include "vllm/multimodal/minimax_h3_video.h"
#include "vllm/multimodal/video_engine.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

class FakeXpuBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override { std::free(p); }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
};

// A PARTIAL backend, shaped exactly like MetalPlatform and TenstorrentPlatform:
// a real accelerator with a registered backend that names the architectures
// whose kernels it has, and declines everything else. `accepts_everything` flips
// it into a COMPLETE backend so the same process can prove the guard does not
// refuse a configuration that works.
class PartialXpuPlatform final : public vllm::platforms::Platform {
 public:
  explicit PartialXpuPlatform(FakeXpuBackend& backend) : backend_(backend) {}

  vt::DeviceType device_type() const override { return vt::DeviceType::kXPU; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {}; }
  std::vector<vt::DType> supported_dtypes() const override { return {vt::DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }

  bool supports_model_architecture(std::string_view architecture) const override {
    if (accepts_everything) return true;
    return architecture == "OPTForCausalLM";
  }

  bool accepts_everything = false;

 private:
  FakeXpuBackend& backend_;
};

FakeXpuBackend& Backend() {
  static FakeXpuBackend backend;
  return backend;
}

PartialXpuPlatform& Platform() {
  static PartialXpuPlatform platform(Backend());
  return platform;
}

// Registered into the CUDA slot as well as its own: `CurrentPlatform()` walks
// {kCUDA, kROCM, kXPU, …} and returns the first REGISTERED entry
// (src/vllm/platforms/platform.cpp), so without this a CUDA build would resolve
// to the real CUDA platform and the guard would never be exercised.
void RegisterPartialAccelerator(bool accepts_everything) {
  vt::RegisterBackend(vt::DeviceType::kXPU, &Backend());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kXPU, &Platform());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &Platform());
  Platform().accepts_everything = accepts_everything;
}

// A path that does not exist. If the capability guard were reverted, the LTX-2.5
// load would sail past it and die opening this instead — a DIFFERENT message,
// which is what makes the assertions below fail for the intended reason rather
// than by accident.
vllm::multimodal::VideoModelParams DeviceOneParams() {
  vllm::multimodal::VideoModelParams mp;
  mp.family = vllm::multimodal::kLtx2VideoFamily;
  mp.dit_path = "/nonexistent/ltx2-dit-that-is-never-opened.safetensors";
  mp.device = 1;
  return mp;
}

// The H3 sibling of DeviceOneParams(), entered through the SAME generic struct
// so it can go in at `LoadVideoEngine` with the family declared. The path does
// not exist for the same reason: a load that got past the capability clause
// would die opening it, with a different message.
vllm::multimodal::VideoModelParams H3DeviceOneParams() {
  vllm::multimodal::VideoModelParams mp;
  mp.family = vllm::multimodal::kMiniMaxH3VideoFamily;
  mp.dit_path = "/nonexistent/h3-dit-that-is-never-opened.gguf";
  mp.device = 1;
  return mp;
}

// The PRODUCTION entry, for BOTH lanes: `vllm_video_engine_load`
// (include/vllm.h) calls `LoadVideoEngine` through src/capi/vllm_c.cpp, and it
// dispatches on the declared family to the registration in the engine's own TU,
// which calls that engine's `Load`. Nothing in this helper names an engine type,
// which is the point — it is the path a user arrives on, and entering at
// `Ltx2VideoEngine::Load` or `MiniMaxH3VideoEngine::Load` instead would skip the
// registry hop and prove one link less.
std::string LoadError(const vllm::multimodal::VideoModelParams& mp) {
  try {
    (void)vllm::multimodal::LoadVideoEngine(mp);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// The architecture name in the refusal's QUOTED SLOT, rather than anywhere in
// the message.
//
// `Fail()` in src/vllm/multimodal/ltx2_video.cpp prefixes EVERY message it
// throws with "ltx-2.5 video: ", and `kLtx2VideoFamily` is the string
// "ltx-2.5". So `msg.find(kLtx2VideoFamily)` is satisfied by that boilerplate on
// every refusal this file can produce — including the two refusals the
// assertions below exist to distinguish this one FROM — and no defect in the
// message can make it fail. Measured: replacing the family name with
// "<redacted>" inside the DECLINES `Fail` left it GREEN.
//
// The row's thesis for #659 is that a partial backend declines BY NAME, so the
// assertion has to be that the architecture appears where the refusal names it.
// The H3 pair happens not to collide — its prefix spells the family with an
// underscore ("minimax_h3 video: ") while the family is hyphenated
// ("minimax-h3") — but that is a coincidence of spelling and not a property, so
// the H3 cases are built the same way.
std::string QuotedArchitecture(std::string_view family) {
  return std::string("architecture '") + std::string(family) + "'";
}

}  // namespace

TEST_CASE("ltx2 video: a platform that DECLINES the architecture refuses device 1 by name") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  // Both of the questions this file already asked PASS here: the platform is an
  // accelerator, and a backend is registered for it. Without the third question
  // the load proceeds into the DiT.
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kXPU);
  REQUIRE(vt::TryGetBackend(vt::DeviceType::kXPU) != nullptr);
  REQUIRE_FALSE(vllm::platforms::CurrentPlatform().supports_model_architecture(
      vllm::multimodal::kLtx2VideoFamily));

  const std::string msg = LoadError(DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  // BY NAME: the platform, the architecture, and the fact that the backend
  // declined rather than that something was malformed.
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kLtx2VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("supports_model_architecture") != std::string::npos);
  // And it must not be the OTHER refusal: a backend IS registered here, so
  // blaming a missing one would be a wrong diagnosis that reads as a right one.
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);
  // Nor the shape of failure the guard exists to prevent: opening the DiT.
  CHECK(msg.find("/nonexistent/") == std::string::npos);
}

TEST_CASE("ltx2 video: a COMPLETE backend is not refused by the capability guard") {
  // The risk the guard carries is refusing a configuration that works. A
  // platform that accepts the architecture must get past this clause and fail
  // only on the thing that is genuinely wrong — the missing checkpoint.
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  const std::string msg = LoadError(DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DECLINES") == std::string::npos);
  CHECK(msg.find("supports_model_architecture") == std::string::npos);
  // POSITIVELY: the load got PAST the capability clause and failed on the thing
  // that is genuinely wrong — the checkpoint that does not exist. Asserting only
  // the two absences above would pass on any OTHER wrong failure (a refusal
  // reworded, a throw from an earlier clause, a message that never mentions the
  // path), which would leave "the guard does not refuse a working configuration"
  // unproven while reading green. Case 1 at the top of this file asserts the
  // mirror image of this line; the pair is what pins the guard to exactly one
  // arm each.
  CHECK(msg.find("/nonexistent/ltx2-dit-that-is-never-opened.safetensors") !=
        std::string::npos);
}

TEST_CASE("minimax_h3 video: device 1 RESOLVES through the seam, it is not enum value 1") {
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  // kXPU is 4. The integer cast this replaced returned kCUDA (1) for any build,
  // any platform, forever; it could not produce this answer.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(1) == vt::DeviceType::kXPU);
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(1) != vt::DeviceType::kCUDA);

  // The rest of the public contract is untouched by the change.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(0) == vt::DeviceType::kCPU);
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(-1));
  CHECK_THROWS(vllm::multimodal::MiniMaxH3VideoDeviceType(2));
}

TEST_CASE("minimax_h3 video: a platform that DECLINES the architecture refuses device 1") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  std::string msg;
  try {
    (void)vllm::multimodal::MiniMaxH3VideoDeviceType(1);
    FAIL("device 1 must be refused when the resolved platform declines the architecture");
  } catch (const std::exception& e) {
    msg = e.what();
  }
  INFO(msg);
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kMiniMaxH3VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);

  // 0 still resolves, because the CPU asks none of these questions.
  CHECK(vllm::multimodal::MiniMaxH3VideoDeviceType(0) == vt::DeviceType::kCPU);
}

// ── REACH (AGENTS.md `## Nothing lands dead`) ────────────────────────────────
//
// The two cases above call `MiniMaxH3VideoDeviceType` directly, which measures
// the free function and NOT whether anything routes to it. The H3 engine's
// `Load` is the only production caller (src/vllm/multimodal/minimax_h3_video.cpp,
// the `MiniMaxH3VideoDeviceType(params.device)` line), and it is reached from
// `vllm_video_engine_load` → `LoadVideoEngine` → the `minimax_h3` registration.
// Delete that one call and every other H3 device assertion in this tree stays
// green; this case is what turns red.
//
// It is the H3 mirror of the two LTX cases at the top of this file, and it
// exists because the LTX half already had this proof and the H3 half did not.
TEST_CASE("minimax_h3 video: a DECLINING platform refuses device 1 through LoadVideoEngine") {
  RegisterPartialAccelerator(/*accepts_everything=*/false);

  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kXPU);
  REQUIRE(vt::TryGetBackend(vt::DeviceType::kXPU) != nullptr);
  REQUIRE_FALSE(vllm::platforms::CurrentPlatform().supports_model_architecture(
      vllm::multimodal::kMiniMaxH3VideoFamily));

  const std::string msg = LoadError(H3DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("xpu") != std::string::npos);
  CHECK(msg.find(QuotedArchitecture(vllm::multimodal::kMiniMaxH3VideoFamily)) !=
        std::string::npos);
  CHECK(msg.find("DECLINES") != std::string::npos);
  CHECK(msg.find("supports_model_architecture") != std::string::npos);
  // Not the OTHER refusal: a backend IS registered for the resolved platform, so
  // blaming a missing one is a wrong diagnosis that reads as a right one.
  CHECK(msg.find("no accelerator backend is registered") == std::string::npos);
  // Nor the failure the guard exists to prevent, one step further on. Reverting
  // the `Load` call site to the integer cast lands here instead: `kCUDA` with no
  // CUDA backend registered in this process.
  CHECK(msg.find("/nonexistent/") == std::string::npos);
}

TEST_CASE("minimax_h3 video: a COMPLETE backend is not refused through LoadVideoEngine") {
  // The mirror arm, for the same reason the LTX pair has one: two absences pass
  // on any OTHER wrong failure, so the guard's "does not refuse a working
  // configuration" half needs a POSITIVE assertion about where the load got to.
  RegisterPartialAccelerator(/*accepts_everything=*/true);

  const std::string msg = LoadError(H3DeviceOneParams());
  INFO(msg);
  REQUIRE_FALSE(msg.empty());
  CHECK(msg.find("DECLINES") == std::string::npos);
  CHECK(msg.find("supports_model_architecture") == std::string::npos);
  CHECK(msg.find("/nonexistent/h3-dit-that-is-never-opened.gguf") != std::string::npos);
}
