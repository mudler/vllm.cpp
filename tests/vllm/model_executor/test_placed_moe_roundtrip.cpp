// `ENG-HYBRID-PLACEMENT` W3i (issue #2714) — the MoE placement round trip,
// executed and compared BYTE-FOR-BYTE, on a build with no GPU in it.
//
// WHAT WAS HERE BEFORE. A file of this name gated the round trip until
// `9164b6cb7` (merge `6416aab85`) deleted it: it called `vllm::RunMoeBlockPlaced`,
// which `866075b2f` had already removed, and `main` would not build with it in
// the tree. Nothing replaced it, so the seam every MoE architecture in this tree
// routes through — `deepseek_v2.cpp`, `kimi_linear_device.cpp`,
// `nemotron_h_device.cpp`, `dots3_note_device.cpp`, `glm_moe_dsa_forward.cpp`,
// `qwen3_5.cpp`, `qwen3_moe.cpp`, `qwen4_exp_forward.cpp` — has had no
// value-preservation gate since 2026-08-30.
//
// WHY IT WAS NOT SIMPLY REWRITTEN, and why that reasoning was wrong. The old
// file forced the transfer by handing `RunMoeBlockPlaced` an explicit `kCPU`
// placement argument. The shared seam has no such parameter: it short-circuits
// on `placed_on == engine_device` and returns the body's value untouched. Issue
// #2714, and this row's spec in three places, concluded from that the branch
// needs a CUDA/ROCm box or a Vulkan/lavapipe build.
//
// It does not. `PlacementQueue` constrains the DESTINATION — `kCPU` is the only
// legal placement target, and that is a real limit. It says nothing about the
// ENGINE, which `RunMoePlaced` reads from `engine.q.device.type`, a field on a
// queue the caller supplies. The branch fires whenever the engine identifies as
// something else, and the engine does not have to BE an accelerator to do that.
//
// So the engine here is a LOOPBACK backend registered on `kXPU` — the device
// type `test_device_pool.cpp:476` documents as having no `RegisterPlatform` call
// anywhere in the tree — whose allocation and transfer methods delegate to the
// real CPU backend. Five files already build a fake backend this way
// (`test_device_selection.cpp:72`, `test_gguf_device_fit_reach.cpp:150`,
// `test_expert_stream_device_slot.cpp:141`, `test_resident_weight_f32_copy_retires.cpp:152`,
// `test_resident_weight_host_addressable.cpp:142`). This one adds no new
// technique; it points the existing one at the other end of the seam.
//
// WHAT THIS PROVES. Every line of the placed branch runs: the down copy sized by
// `dh.dtype`, the block on the placement queue, the up copy sized by the dtype
// the BODY produced, and the hand-back into an engine-pool `DBuf`. Because both
// devices are host memory the arms are comparable byte-for-byte rather than
// within a tolerance, which is the correct bar — placement decides WHERE a value
// is computed and must never change WHAT it is.
//
// WHAT IT DOES NOT PROVE, stated because the complement is what makes the pair
// honest: that a real cross-bus transfer works. The GB10 run recorded in the
// spec's W3h (NMSE 5.239e-06 against the CPU oracle, 48/48 layers placed) is
// that evidence. It needs a leased GPU and a 72 GB checkpoint and runs when
// somebody asks for it; this file runs on every merge.
//
// The loopback COUNTS its transfers, and that is the instrument rather than a
// diagnostic. The seam's f32 truncation (#2383) was exactly a byte count: it
// hardcoded bf16 and copied half of an f32 block. A test that only compares
// values would pass an arm that moved the right bytes for the wrong reason, so
// the byte counts are asserted directly.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "support/expert_stream_model.h"
#include "vllm/model_executor/device_placement.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5_moe_block.h"
#include "vllm/model_executor/moe_placement_seam.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// A SECOND DEVICE IDENTITY over the first device's memory.
//
// It is not a mock: every method does the real thing, by delegating to the CPU
// backend that is already registered. What it changes is the ANSWER TO "which
// device is this", which is the only input the seam's branch reads. A block it
// allocates is freed through it, so the `DevicePool` — which keys on backend
// identity — stays consistent.
class LoopbackBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override { return Cpu().Alloc(bytes); }
  void Free(void* p) override { Cpu().Free(p); }
  void Memset(Queue& q, void* p, int value, size_t bytes) override {
    Cpu().Memset(q, p, value, bytes);
  }
  void Copy(Queue& q, void* dst, const void* src, size_t bytes) override {
    copies.push_back(bytes);
    Cpu().Copy(q, dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  bool UnifiedMemory() const override { return true; }

  // Every transfer the ENGINE side of the seam made, in order and in bytes. The
  // placed side runs on the real CPU backend and is deliberately not counted
  // here: the two sides must be distinguishable or a wrong-sized copy on one
  // could be read off the other's total.
  std::vector<size_t> copies;

 private:
  static vt::Backend& Cpu() { return vt::GetBackend(DeviceType::kCPU); }
};

LoopbackBackend& Loopback() {
  static LoopbackBackend b;
  return b;
}

// The platform the engine's `DBuf` allocations resolve their pool policy
// through. `ResolveDevicePoolPolicy` calls `platforms::GetPlatform(kXPU)` and
// throws if none is registered, so the engine needs one to exist at all.
class LoopbackPlatform final : public vllm::platforms::Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  vt::Backend& backend() const override { return Loopback(); }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {12, 1};
  }
  std::vector<DType> supported_dtypes() const override {
    return {DType::kBF16, DType::kF32};
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
};

LoopbackPlatform& LoopbackPlat() {
  static LoopbackPlatform p;
  return p;
}

struct Registrar {
  Registrar() {
    vt::RegisterBackend(Device{DeviceType::kXPU, 0}, &Loopback());
    vllm::platforms::RegisterPlatform(DeviceType::kXPU, &LoopbackPlat());
  }
};
const Registrar kRegistrar;

Queue& EngineQueue() {
  static Queue q{Device{DeviceType::kXPU, 0}, nullptr};
  return q;
}

Dev EngineDev() { return Dev{Loopback(), EngineQueue()}; }

// The routed-expert regex `-ot` / `-cmoe` resolve to, and the one every MoE
// architecture's tensors are named against.
constexpr const char* kExps = "\\.ffn_(gate|up|down)_exps";

// Install a plan that places EVERY layer's experts on the CPU, against an engine
// that is not the CPU. This is the `cpu_moe` arm, which is the one operators run.
void InstallCpuMoePlan(int64_t layers) {
  const vllm::DevicePlacement p = vllm::DevicePlacement::FromOverrides(
      {vllm::PlacementOverride{kExps, "cpu"}}, DeviceType::kXPU);
  vllm::SetActiveMoePlacementPlan(vllm::MoePlacementPlan::Resolve(p, layers));
}

Tensor HostView(void* data, DType dt, Device dev,
                const std::vector<int64_t>& shape) {
  Tensor t{};
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

// A reproducible hidden state. The pattern matters only in that a difference in
// the output must be a difference in the PATH and never in the input.
std::vector<uint16_t> MakeHiddenBf16(int64_t T, int64_t H, uint64_t seed) {
  std::vector<uint16_t> v(static_cast<size_t>(T * H));
  uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (size_t i = 0; i < v.size(); ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    v[i] = static_cast<uint16_t>((x >> 48) & 0x3F00u);
  }
  return v;
}

}  // namespace

// THE GATE #2714 ASKS FOR. The same block, over the same weights, computed
// directly and then through the placed branch, compared byte-for-byte.
TEST_CASE("placed moe: the round trip preserves the block's output EXACTLY") {
  vllm::ResetActiveMoePlacementPlanForTesting();

  const vllm::HfConfig cfg = expert_stream_test::MakeConfig();
  const vllm::MoeBlockWeights moe = expert_stream_test::MakeKqMoe(cfg, /*seed=*/7);
  const int64_t H = cfg.hidden_size;
  const int64_t T = 3;
  const size_t bytes = static_cast<size_t>(T * H) * sizeof(uint16_t);

  vt::Backend& cpu = vt::GetBackend(DeviceType::kCPU);
  Queue cpu_q = cpu.CreateQueue();
  const std::vector<uint16_t> hidden = MakeHiddenBf16(T, H, 42);

  // THE DIRECT CALL: what an unplaced layer runs, on the CPU, today.
  std::vector<uint16_t> direct_out(static_cast<size_t>(T * H));
  {
    Tensor dh = HostView(const_cast<uint16_t*>(hidden.data()), DType::kBF16,
                         cpu_q.device, {T, H});
    vllm::MoeBlockOutput o = vllm::RunMoeBlock(cpu_q, moe, cfg, dh, T);
    cpu.Copy(cpu_q, direct_out.data(), o.tensor.data, bytes);
    cpu.Synchronize(cpu_q);
  }

  // THE PLACED CALL: the identical block, reached across a device boundary. The
  // body is `qwen3_moe.cpp:110` verbatim, so this holds the production adapter
  // rather than a paraphrase of it.
  InstallCpuMoePlan(cfg.num_hidden_layers);
  Loopback().copies.clear();
  DeviceType body_ran_on = DeviceType::kXPU;
  std::vector<uint16_t> placed_out(static_cast<size_t>(T * H));
  {
    Tensor dh = HostView(const_cast<uint16_t*>(hidden.data()), DType::kBF16,
                         EngineQueue().device, {T, H});
    vllm::MoePlacedOutput moe_out = vllm::RunMoePlacedPair(
        EngineDev(), /*layer_index=*/0, dh, T, H,
        [&](Dev p, const Tensor& h) {
          body_ran_on = p.q.device.type;
          vllm::MoeBlockOutput o = vllm::RunMoeBlock(p.q, moe, cfg, h, T);
          return vllm::MoePlacedOutput{o.tensor, std::move(o.storage)};
        });

    // THE BRANCH ACTUALLY TAKEN, asserted before anything is compared. A test
    // that fell back into the short circuit would agree byte-for-byte for the
    // one reason that proves nothing, and this repository has shipped exactly
    // that shape of green.
    CHECK(body_ran_on == DeviceType::kCPU);
    REQUIRE(Loopback().copies.size() == 2);

    std::memcpy(placed_out.data(), moe_out.tensor.data, bytes);

    // The result is owned the way an unplaced block's output is owned: from the
    // ENGINE's pool, outliving the call, which is what the composing forward
    // relies on when it keeps computing over `tensor`.
    CHECK(moe_out.storage != nullptr);
    CHECK(moe_out.tensor.shape[0] == T);
    CHECK(moe_out.tensor.shape[1] == H);
    CHECK(moe_out.tensor.dtype == DType::kBF16);
    CHECK(moe_out.tensor.device.type == DeviceType::kXPU);
  }

  // BYTE-FOR-BYTE, not a tolerance. A tolerance here would admit precisely the
  // class of defect the seam exists to make impossible.
  CHECK(std::memcmp(direct_out.data(), placed_out.data(), bytes) == 0);

  vllm::ResetActiveMoePlacementPlanForTesting();
  vt::DestroyQueue(cpu_q);
}

// The f32 defect (#2383), as a BYTE COUNT rather than as a value comparison.
// Widening f32 bits into bf16 yields plausible floats, so a value check can pass
// over a half-copied buffer; the transfer size cannot.
TEST_CASE("placed moe: an f32 block moves 4 bytes per element, not 2") {
  vllm::ResetActiveMoePlacementPlanForTesting();
  InstallCpuMoePlan(4);

  const int64_t T = 2, H = 8;
  std::vector<float> src(static_cast<size_t>(T * H));
  for (size_t i = 0; i < src.size(); ++i) src[i] = 1.0f + static_cast<float>(i);

  Loopback().copies.clear();
  Tensor dh = HostView(src.data(), DType::kF32, EngineQueue().device, {T, H});
  DBuf out = vllm::RunMoePlaced(
      EngineDev(), /*layer_index=*/0, dh, T, H, [&](Dev p, const Tensor& h) {
        // An f32 identity block: it produces the value it was given, so any
        // difference downstream is the TRANSFER's and not the arithmetic's.
        DBuf r(p, DType::kF32, {T, H});
        p.b.Copy(p.q, r.t().data, h.data, static_cast<size_t>(T * H) * 4);
        p.b.Synchronize(p.q);
        return r;
      });

  REQUIRE(Loopback().copies.size() == 2);
  CHECK(Loopback().copies[0] == static_cast<size_t>(T * H) * 4);  // down
  CHECK(Loopback().copies[1] == static_cast<size_t>(T * H) * 4);  // back up
  CHECK(out.t().dtype == DType::kF32);

  // The values survive too, which is the half a byte count cannot see.
  std::vector<float> got(static_cast<size_t>(T * H));
  std::memcpy(got.data(), out.t().data, got.size() * 4);
  CHECK(std::memcmp(got.data(), src.data(), got.size() * 4) == 0);

  vllm::ResetActiveMoePlacementPlanForTesting();
}

// The OUTPUT dtype need not equal the input's, and the copy back is sized by
// what the body produced. This is a separate assumption from the one above and
// the seam gets it wrong independently: sizing the return by `dh.dtype` would
// copy twice the bytes out of a bf16 result.
TEST_CASE("placed moe: the copy back is sized by the OUTPUT dtype") {
  vllm::ResetActiveMoePlacementPlanForTesting();
  InstallCpuMoePlan(4);

  const int64_t T = 2, H = 8;
  std::vector<float> src(static_cast<size_t>(T * H), 1.0f);

  Loopback().copies.clear();
  Tensor dh = HostView(src.data(), DType::kF32, EngineQueue().device, {T, H});
  DBuf out = vllm::RunMoePlaced(
      EngineDev(), /*layer_index=*/0, dh, T, H, [&](Dev p, const Tensor& h) {
        (void)h;
        DBuf r(p, DType::kBF16, {T, H});
        p.b.Memset(p.q, r.t().data, 0, static_cast<size_t>(T * H) * 2);
        p.b.Synchronize(p.q);
        return r;
      });

  REQUIRE(Loopback().copies.size() == 2);
  CHECK(Loopback().copies[0] == static_cast<size_t>(T * H) * 4);  // f32 in
  CHECK(Loopback().copies[1] == static_cast<size_t>(T * H) * 2);  // bf16 out
  CHECK(out.t().dtype == DType::kBF16);

  vllm::ResetActiveMoePlacementPlanForTesting();
}

// The fp4-resident refusal (#2309), asserted POSITIVELY for the first time.
// `test_device_placement.cpp:433` can only check that our refusal is NOT what
// fires, because on a CPU engine the branch carrying it is unreachable.
TEST_CASE("placed moe: an unplaceable layer is REFUSED, not served slowly") {
  vllm::ResetActiveMoePlacementPlanForTesting();
  InstallCpuMoePlan(4);

  const int64_t T = 1, H = 4;
  std::vector<uint16_t> src(static_cast<size_t>(T * H), 0);
  Tensor dh = HostView(src.data(), DType::kBF16, EngineQueue().device, {T, H});

  bool body_ran = false;
  CHECK_THROWS_AS(
      vllm::RunMoePlaced(
          EngineDev(), /*layer_index=*/0, dh, T, H,
          [&](Dev p, const Tensor& h) {
            (void)h;
            body_ran = true;
            return DBuf(p, DType::kBF16, {T, H});
          },
          /*placeable=*/false, "the routed experts are fp4-resident"),
      std::invalid_argument);

  // REFUSED rather than quietly run unplaced. Running the body anyway would be
  // the invisible-fallback shape: correct tokens, twice the bytes, no signal.
  CHECK_FALSE(body_ran);

  vllm::ResetActiveMoePlacementPlanForTesting();
}

// INERTNESS, on the same engine, so the two arms differ only in the plan. With
// nothing placed the seam must be the bare call: the body sees the ENGINE device
// and the engine backend performs no transfer at all.
TEST_CASE("placed moe: an unplaced layer costs no transfer and no copy") {
  vllm::ResetActiveMoePlacementPlanForTesting();

  const int64_t T = 2, H = 8;
  std::vector<uint16_t> src(static_cast<size_t>(T * H), 0);

  Loopback().copies.clear();
  Tensor dh = HostView(src.data(), DType::kBF16, EngineQueue().device, {T, H});
  DeviceType body_ran_on = DeviceType::kCPU;
  DBuf out = vllm::RunMoePlaced(
      EngineDev(), /*layer_index=*/0, dh, T, H, [&](Dev p, const Tensor& h) {
        (void)h;
        body_ran_on = p.q.device.type;
        return DBuf(p, DType::kBF16, {T, H});
      });

  CHECK(body_ran_on == DeviceType::kXPU);
  CHECK(Loopback().copies.empty());
  CHECK(out.t().dtype == DType::kBF16);
}
