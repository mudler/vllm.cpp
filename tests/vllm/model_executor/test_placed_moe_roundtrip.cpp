// `ENG-HYBRID-PLACEMENT` W3b (issue #2026) — the ROUND TRIP itself.
//
// `test_device_placement.cpp` gates the decision: which device a name resolves
// to, and which layers a plan places. Nothing there executes
// `RunMoeBlockPlaced`, so without this file the routing would land dead: the
// plan would be right and the code that acts on it would never have run once.
//
// WHAT THIS CAN AND CANNOT PROVE. It drives the real production helper over a
// real `MoeBlockWeights`, so every line of the round trip executes — the copy
// down, the block on the placement queue, the copy back, and the ownership
// hand-off of the returned pool block. With engine and placement BOTH on the CPU
// the transfers are memcpys, which is what makes the comparison exact and lets
// the equality below be byte-for-byte rather than a tolerance.
//
// It does NOT prove the cross-device arm. That needs the engine device and the
// placement device to differ, which on this box means a Vulkan build against
// lavapipe, and it is the gate W3b still owes.
#include <doctest/doctest.h>

#include <cstring>
#include <string>
#include <vector>

#include "support/expert_stream_model.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/models/qwen3_5_moe_block.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

// A host-resident bf16 view, built the way the fixture's own helpers do rather
// than by reaching into a model TU's private glue.
vt::Tensor MakeHostTensor(void* data, vt::DType dt, vt::Device dev,
                          const std::vector<int64_t>& shape) {
  vt::Tensor t{};
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

// The hidden state the block reads, filled deterministically so a difference in
// the output is a difference in the PATH and never in the input.
std::vector<uint16_t> MakeHidden(int64_t T, int64_t H, uint64_t seed) {
  std::vector<uint16_t> v(static_cast<size_t>(T * H));
  uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (size_t i = 0; i < v.size(); ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    // A small bf16 magnitude: the point is a reproducible pattern, not a range.
    v[i] = static_cast<uint16_t>((x >> 48) & 0x3F00u);
  }
  return v;
}

}  // namespace

TEST_CASE("placed moe: the round trip preserves the block's output EXACTLY") {
  vllm::ResetActiveMoePlacementPlanForTesting();

  const vllm::HfConfig cfg = expert_stream_test::MakeConfig();
  const vllm::MoeBlockWeights moe = expert_stream_test::MakeKqMoe(cfg, /*seed=*/7);
  const int64_t H = cfg.hidden_size;
  const int64_t T = 3;

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = cpu.CreateQueue();

  const std::vector<uint16_t> hidden = MakeHidden(T, H, 42);
  const size_t bytes = hidden.size() * sizeof(uint16_t);

  // The DIRECT call: what an unplaced layer runs today.
  vllm::MoeBlockOutput direct;
  std::vector<uint16_t> direct_out(static_cast<size_t>(T * H));
  {
    vt::Tensor dh = MakeHostTensor(
        const_cast<uint16_t*>(hidden.data()), vt::DType::kBF16, q.device, {T, H});
    direct = vllm::RunMoeBlock(q, moe, cfg, dh, T);
    cpu.Copy(q, direct_out.data(), direct.tensor.data, bytes);
    cpu.Synchronize(q);
  }

  // The PLACED call: the same block, reached through the round trip. Every line
  // of `RunMoeBlockPlaced` runs here.
  vllm::MoeBlockOutput placed;
  std::vector<uint16_t> placed_out(static_cast<size_t>(T * H));
  {
    vt::Tensor dh = MakeHostTensor(
        const_cast<uint16_t*>(hidden.data()), vt::DType::kBF16, q.device, {T, H});
    placed = vllm::RunMoeBlockPlaced(q, vt::DeviceType::kCPU, moe, cfg, dh, T);
    cpu.Copy(q, placed_out.data(), placed.tensor.data, bytes);
    cpu.Synchronize(q);
  }

  // BYTE-FOR-BYTE, not a tolerance. Placement is a scheduling decision: it moves
  // where a value is computed and must never change the value. A tolerance here
  // would hide exactly the defect this row must not ship.
  REQUIRE(direct_out.size() == placed_out.size());
  CHECK(std::memcmp(direct_out.data(), placed_out.data(), bytes) == 0);

  // The returned block is owned the way an unplaced block's output is owned —
  // it comes from the ENGINE's pool and survives the helper returning, which is
  // what the composing forward relies on when it keeps computing over `tensor`.
  CHECK(placed.storage != nullptr);
  CHECK(placed.tensor.shape[0] == T);
  CHECK(placed.tensor.shape[1] == H);
  CHECK(placed.tensor.dtype == vt::DType::kBF16);

  vt::DestroyQueue(q);
}

TEST_CASE("placed moe: the fp4-resident arm is REFUSED, not served slowly") {
  // Its device residents are built EAGERLY at load, so placing it would upload
  // every expert and then compute across the bus — slower than not placing, and
  // INVISIBLE to a token gate because the tokens would still be right. That is
  // the trap this row must not ship, so it is refused by name instead.
  const vllm::HfConfig cfg = expert_stream_test::MakeConfig();
  vllm::MoeBlockWeights moe = expert_stream_test::MakeKqMoe(cfg, /*seed=*/9);
  // One non-empty fp4 expert is enough to select that arm.
  moe.expert_gate_fp4.resize(1);

  vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = cpu.CreateQueue();
  const int64_t H = cfg.hidden_size, T = 1;
  const std::vector<uint16_t> hidden = MakeHidden(T, H, 1);
  vt::Tensor dh = MakeHostTensor(
      const_cast<uint16_t*>(hidden.data()), vt::DType::kBF16, q.device, {T, H});

  std::string msg;
  try {
    vllm::RunMoeBlockPlaced(q, vt::DeviceType::kCPU, moe, cfg, dh, T);
    msg = "ACCEPTED (no throw)";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  CHECK(msg.find("fp4-resident") != std::string::npos);
  CHECK(msg.find("across the bus") != std::string::npos);
  vt::DestroyQueue(q);
}
