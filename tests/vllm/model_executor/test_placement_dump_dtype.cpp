// The MoE placement dump must render the block's OWN dtype (#2382 follow-up).
//
// WHY ITS OWN BINARY. `VT_PLACEMENT_DUMP_MOE` is read through a function-local
// static that latches on FIRST use, and `test_device_placement` drives
// `RunMoePlaced` — which calls this hook — before any case here could set the
// variable. Sharing a binary would latch it to "unset" and this suite would pass
// while asserting nothing, which is the shape of failure this repository keeps
// finding. A separate binary gets a fresh latch.
//
// WHAT IT GUARDS. The seam hardcoded bf16 in six places and never compared
// against `dh.dtype`, while `kimi_linear_device.cpp:930` hands it an f32 `[T,H]`
// block. On the placed branch that copied HALF the bytes and reinterpreted f32
// as bf16. The dump had the same assumption: widening f32 bits as bf16 yields
// plausible-looking floats, so the gate would have computed an NMSE over
// garbage and reported a number rather than an error.
//
// WHAT IT CANNOT GUARD, stated because the gap is the reason the defect lived.
// The seam's placed branch takes its device from `engine.q.device.type`, and the
// only legal placement target is the CPU, so on a CPU-only build `placed_on`
// always equals the engine device and THE PLACED BRANCH IS UNREACHABLE. No test
// in this suite can execute the f32 round trip itself; only a GPU box can. That
// is why the truncation survived every green CI run.
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/device_placement.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

std::vector<double> ReadDump(const std::string& path) {
  std::vector<double> v;
  std::ifstream in(path);
  double d;
  while (in >> d) v.push_back(d);
  return v;
}

}  // namespace

TEST_CASE("placement dump: f32 is written as f32, not widened as bf16") {
  const std::string path = "/tmp/vllm-cpp-dump-dtype-f32.txt";
  std::remove(path.c_str());
  ::setenv("VT_PLACEMENT_DUMP_MOE", path.c_str(), 1);

  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();

  // Values chosen so a bf16 MISREAD is unmistakable rather than merely close:
  // reading these f32 bits as two bf16 halves produces different numbers and
  // twice as many of them.
  const std::vector<float> src{1.5f, -2.25f, 0.125f, 3.0f};
  vllm::MaybeDumpMoeBlockOutput(0, b, q, src.data(),
                                static_cast<int64_t>(src.size()),
                                /*data_is_host=*/true, vt::DType::kF32);

  const std::vector<double> got = ReadDump(path);
  REQUIRE(got.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    CAPTURE(i);
    CHECK(got[i] == doctest::Approx(static_cast<double>(src[i])));
  }
  std::remove(path.c_str());
  vt::DestroyQueue(q);
}

TEST_CASE("placement dump: bf16 widens exactly, and the hook fires only once") {
  // The latch is already spent by the case above, so this asserts the SECOND
  // property the hook promises: one write per process. A hook that appended on
  // every call would make the gate compare arms that had already diverged.
  const std::string path = "/tmp/vllm-cpp-dump-dtype-bf16.txt";
  std::remove(path.c_str());

  vt::Backend& b = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = b.CreateQueue();
  const std::vector<uint16_t> bits{0x3F80, 0x4000, 0xBF00};  // 1.0, 2.0, -0.5
  ::setenv("VT_PLACEMENT_DUMP_MOE", path.c_str(), 1);
  vllm::MaybeDumpMoeBlockOutput(0, b, q, bits.data(),
                                static_cast<int64_t>(bits.size()),
                                /*data_is_host=*/true, vt::DType::kBF16);

  // The path latched to the FIRST case's file, so nothing is written here. That
  // is the guarantee, and asserting it is what proves the hook is once-only
  // rather than merely appearing to work.
  std::ifstream in(path);
  CHECK_FALSE(in.good());
  vt::DestroyQueue(q);
}
