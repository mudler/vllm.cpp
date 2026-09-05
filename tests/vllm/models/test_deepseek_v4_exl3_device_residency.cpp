// MODEL-DSV4-EXL3 W2 (#2442): the routed-expert tower's DEVICE RESIDENCY.
//
// THE DEFECT. `DeepseekV4Exl3Linear` holds the coalesced trellis tower in host
// `std::vector`s and the forward built `vt::Tensor`s straight over those host
// pointers while stamping them with the forward's device. A device kernel
// dereferencing that is the #844 / #1435 crash, so both `Exl3Linear` and the
// fused MoE arm refused every non-CPU queue -- which meant all 216 routed
// experts of the real artifact ran on a CPU queue.
//
// WHAT THIS FILE CAN AND CANNOT PROVE, stated up front because the split is the
// whole design of the gate. Staging is a NO-OP on a CPU queue by construction:
// `dense_attn::ResidentWeight` aliases host bytes when the "device" IS the host,
// so there is nothing to upload and the host vectors must survive. So a CPU box
// can prove the CPU branch, the refusal that replaced the old one, and the
// vacuous-truth guard -- and it CANNOT prove the upload. That half is
// `test_cuda_deepseek_v4.cpp`'s staged-residency case and runs under an `rc`
// lease on a GPU. A file that claimed otherwise would be a skip wearing a pass.
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_exl3_device.h"

#include "dsv4_exl3_fixture.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using namespace dsv4_exl3_fixture;

namespace {

// A CPU queue, which is the only device a host-only lane has. Owned by the
// caller through the RAII wrapper below so a failing REQUIRE cannot leak it.
struct CpuQueue {
  vt::Queue q;
  CpuQueue() { q = vt::GetBackend(vt::DeviceType::kCPU).CreateQueue(); }
  ~CpuQueue() { vt::GetBackend(vt::DeviceType::kCPU).DestroyQueue(q); }
};

}  // namespace

TEST_CASE("W2: staging is a NO-OP on a CPU queue, and says so by leaving the host bytes") {
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.has_exl3_weights);
  REQUIRE(w.exl3.layers.size() > 0);
  REQUIRE(w.exl3.layers[0].experts.size() > 0);

  const vllm::DeepseekV4Exl3Linear& w1 = w.exl3.layers[0].experts[0].w1;
  const size_t trellis_before = w1.trellis.size();
  const size_t suh_before = w1.suh.size();
  const size_t svh_before = w1.svh.size();
  REQUIRE(trellis_before > 0);

  CpuQueue cpu;
  const int64_t uploaded = vllm::StageDeepseekV4Exl3TowerToDevice(cpu.q, w.exl3);

  // ZERO bytes moved, and the host tower is untouched. Both halves matter: a
  // staging call that uploaded on CPU would double the residency of the one arm
  // that has no device memory to spare, and one that freed the host vectors
  // would leave the CPU forward reading nothing.
  CHECK(uploaded == 0);
  CHECK(w1.trellis.size() == trellis_before);
  CHECK(w1.suh.size() == suh_before);
  CHECK(w1.svh.size() == svh_before);
  CHECK_FALSE(w1.device_staged);
  CHECK_FALSE(vllm::DeepseekV4Exl3TowerIsDeviceStaged(w.exl3));
}

TEST_CASE("W2: the PER-LINEAR entry is a no-op on CPU too, not just the tower walk") {
  // Found by mutation: deleting the CPU guard inside
  // `StageDeepseekV4Exl3LinearToDevice` left this file GREEN, because the tower
  // walk has its own guard and returns before ever calling the linear. The two
  // guards are independent -- the per-linear entry is public and the CUDA gate
  // calls it directly -- so each needs its own case or one of them is untested.
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.exl3.layers.size() > 0);
  REQUIRE(w.exl3.layers[0].experts.size() > 0);
  const vllm::DeepseekV4Exl3Linear& w1 = w.exl3.layers[0].experts[0].w1;
  const size_t before = w1.trellis.size();
  REQUIRE(before > 0);

  CpuQueue cpu;
  vllm::StageDeepseekV4Exl3LinearToDevice(cpu.q, w1);

  CHECK_FALSE(w1.device_staged);
  CHECK(w1.trellis.size() == before);
  CHECK(w1.d_trellis.d_dev == nullptr);
}

TEST_CASE("W2: an EMPTY tower is not a staged tower") {
  // The vacuous-truth guard. `DeepseekV4Exl3TowerIsDeviceStaged` walks every
  // expert and returns false on the first unstaged one; without the explicit
  // `any` it would return TRUE for a tower with no experts at all, and the
  // forward would read that as "a device queue is safe here".
  vllm::DeepseekV4Exl3Weights empty{};
  CHECK_FALSE(vllm::DeepseekV4Exl3TowerIsDeviceStaged(empty));

  vllm::DeepseekV4Exl3Weights no_experts{};
  no_experts.layers.resize(3);  // three layers, zero experts between them
  CHECK_FALSE(vllm::DeepseekV4Exl3TowerIsDeviceStaged(no_experts));
}

TEST_CASE("W2: a PARTIALLY staged tower does not read as staged") {
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  REQUIRE(w.exl3.layers.size() > 0);
  REQUIRE(w.exl3.layers[0].experts.size() > 1);

  // Mark every expert staged EXCEPT one projection of one expert. That is the
  // state a partial or interrupted stage leaves, and it is the state a
  // tower-level "did we stage?" flag would have called safe -- the kernel
  // dereferences whichever expert the router picked, so one host expert among
  // 216 is the same crash as all of them.
  for (auto& layer : w.exl3.layers)
    for (auto& e : layer.experts) {
      e.w1.device_staged = true;
      e.w2.device_staged = true;
      e.w3.device_staged = true;
    }
  CHECK(vllm::DeepseekV4Exl3TowerIsDeviceStaged(w.exl3));

  w.exl3.layers[0].experts[1].w2.device_staged = false;
  CHECK_FALSE(vllm::DeepseekV4Exl3TowerIsDeviceStaged(w.exl3));
}

TEST_CASE("W2: staging is idempotent, so a per-forward call costs one predicate") {
  auto f = BuildFixture();
  vllm::DeepseekV4Weights w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config);
  CpuQueue cpu;
  // The production call sites invoke this on EVERY forward. It has to be safe to
  // repeat, or the first token would pay for the tower and the second would pay
  // again -- or worse, upload a second copy beside the first.
  CHECK(vllm::StageDeepseekV4Exl3TowerToDevice(cpu.q, w.exl3) == 0);
  CHECK(vllm::StageDeepseekV4Exl3TowerToDevice(cpu.q, w.exl3) == 0);
  CHECK(vllm::StageDeepseekV4Exl3TowerToDevice(cpu.q, w.exl3) == 0);
}

// ─── #2458: the fused arm's flag, on every queue ─────────────────────────────
//
// `vt::Exl3MoeMlp` faulted on a device queue until #2458, so the fused arm
// carried a second predicate that kept it off there unless an operator wrote an
// explicit '1'. The kernel is fixed -- the port had dropped upstream's
// `shmem_out_had` template parameter, so the MoE epilogue dereferenced the NULL
// `post_scale` its own bands pass -- and the second predicate is gone with it.
// One flag now selects the arm on the host and on a device queue alike, and
// this pins its parse so a later edit cannot make silence read as OFF.
TEST_CASE("W2/#2458: one flag selects the fused arm on every queue") {
  // Silence and every non-'0' value read as ON. Silence is the DEFAULT path, so
  // this is the case that says the device arm ships enabled.
  CHECK(vllm::Dsv4Exl3FusedMoeFlagIsOn(nullptr));
  CHECK(vllm::Dsv4Exl3FusedMoeFlagIsOn("1"));
  CHECK(vllm::Dsv4Exl3FusedMoeFlagIsOn("yes"));
  CHECK(vllm::Dsv4Exl3FusedMoeFlagIsOn(""));

  // Only a LEADING '0' turns it off, which is the shape the other
  // `VT_DSV4_EXL3_*` knobs use and not the repository-wide flag rule.
  CHECK_FALSE(vllm::Dsv4Exl3FusedMoeFlagIsOn("0"));
  CHECK_FALSE(vllm::Dsv4Exl3FusedMoeFlagIsOn("0ff"));
}
