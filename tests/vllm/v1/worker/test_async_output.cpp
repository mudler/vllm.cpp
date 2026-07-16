// Tests for AsyncGPUModelRunnerOutput + AsyncOutputPool (ENG-ASYNC-SCHED W3
// sampler-OUTPUT half + the throughput lever). Ports the intent of
// vllm/v1/worker/gpu/async_utils.py's AsyncOutput + gpu_model_runner.py:242-332
// AsyncGPUModelRunnerOutput contract: a device-resident sampled-id snapshot is
// D2H'd on a copy queue with an event, and get_output() blocks only that event
// and returns the materialized ModelRunnerOutput. On CPU (unified memory) the
// copy is a memcpy and events are no-ops, so get_output() must be token-exact.
//
// The pool cases lock in the throughput fix: sample_tokens_async must NOT do a
// per-step raw device/pinned alloc/free (those device-sync on CUDA and serialize
// the depth-2 overlap). The pool reuses persistent slots; the async output only
// BORROWS one and releases it on consume, so the slot count stays bounded and
// the buffers are recycled across steps (mirrors torch's caching allocators).
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <set>

#include "vllm/v1/engine/types.h"
#include "vllm/v1/worker/gpu/async_output.h"
#include "vt/backend.h"
#include "vt/device.h"

using vllm::v1::AsyncGPUModelRunnerOutput;
using vllm::v1::AsyncOutputPool;
using vllm::v1::AsyncOutputSlot;
using vllm::v1::ModelRunnerOutput;

namespace {

// Acquire a pool slot, write the sampled ids into its device buffer exactly as
// the runner's sample_tokens_async does, and wrap it in an
// AsyncGPUModelRunnerOutput (the async output BORROWS the slot).
std::unique_ptr<AsyncGPUModelRunnerOutput> MakeAsyncOutput(
    const std::vector<int64_t>& ids, AsyncOutputPool& pool, vt::Queue& main_q,
    vt::Queue& copy_q) {
  const vt::Device dev{vt::DeviceType::kCPU, 0};
  const int num_reqs = static_cast<int>(ids.size());
  AsyncOutputSlot* slot = pool.Acquire();
  if (!ids.empty()) {
    vt::GetBackend(dev.type).Copy(main_q, slot->device_sampled_ids, ids.data(),
                                  ids.size() * sizeof(int64_t));
  }
  ModelRunnerOutput skeleton;
  for (int i = 0; i < num_reqs; ++i) {
    const std::string id = "R" + std::to_string(i);
    skeleton.req_ids.push_back(id);
    skeleton.req_id_to_index[id] = i;
  }
  return std::make_unique<AsyncGPUModelRunnerOutput>(
      std::move(skeleton), dev, pool, slot, num_reqs, main_q, copy_q);
}

}  // namespace

TEST_CASE("AsyncGPUModelRunnerOutput materializes the sampled ids via get_output") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  AsyncOutputPool pool(vt::Device{vt::DeviceType::kCPU, 0}, /*capacity_elems=*/8,
                       /*initial_slots=*/2);

  const std::vector<int64_t> ids = {42, 7, 1001, 0};
  auto async = MakeAsyncOutput(ids, pool, main_q, copy_q);
  CHECK(async->num_reqs() == 4);

  // get_output blocks on the copy event (no-op on CPU), then returns tokens.
  ModelRunnerOutput out = async->get_output();
  REQUIRE(out.sampled_token_ids.size() == 4);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{42});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{7});
  CHECK(out.sampled_token_ids[2] == std::vector<int32_t>{1001});
  CHECK(out.sampled_token_ids[3] == std::vector<int32_t>{0});
  // Ordering fields survive.
  CHECK(out.req_ids.size() == 4);
  CHECK(out.req_id_to_index.at("R2") == 2);

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}

TEST_CASE("AsyncGPUModelRunnerOutput handles a zero-request flush step") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  AsyncOutputPool pool(vt::Device{vt::DeviceType::kCPU, 0}, /*capacity_elems=*/8,
                       /*initial_slots=*/2);

  auto async = MakeAsyncOutput({}, pool, main_q, copy_q);
  CHECK(async->num_reqs() == 0);
  ModelRunnerOutput out = async->get_output();
  CHECK(out.sampled_token_ids.empty());
  CHECK(out.req_ids.empty());

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}

// The copy is issued on the COPY queue (which first waits the MAIN queue), so a
// value written to the device snapshot BEFORE construction is what get_output
// returns — the main queue is never synchronized by the async output.
TEST_CASE("AsyncGPUModelRunnerOutput copy reflects the pre-construction snapshot") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  AsyncOutputPool pool(vt::Device{vt::DeviceType::kCPU, 0}, /*capacity_elems=*/8,
                       /*initial_slots=*/2);

  auto a = MakeAsyncOutput({555, 666}, pool, main_q, copy_q);
  ModelRunnerOutput out = a->get_output();
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{555});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{666});

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}

// THROUGHPUT-LEVER GUARD: a long stream of consume-then-produce steps (depth-2
// discipline: at most 2 outputs alive) must NOT grow the pool past its seed —
// the persistent slots are recycled, so no per-step allocation ever happens.
// RED before the fix: AsyncGPUModelRunnerOutput owned+freed per step (no pool).
TEST_CASE("AsyncOutputPool recycles slots across many steps (no per-step growth)") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  AsyncOutputPool pool(vt::Device{vt::DeviceType::kCPU, 0}, /*capacity_elems=*/16,
                       /*initial_slots=*/4);
  const int seed_slots = pool.num_slots();
  CHECK(seed_slots == 4);

  // Depth-2 pattern: keep one prior output alive while producing the next, then
  // consume the prior — exactly the engine's step_with_batch_queue discipline.
  std::unique_ptr<AsyncGPUModelRunnerOutput> prev;
  std::set<const void*> device_ptrs;
  for (int step = 0; step < 64; ++step) {
    auto cur = MakeAsyncOutput({static_cast<int64_t>(step), step + 1000}, pool,
                               main_q, copy_q);
    if (prev) {
      ModelRunnerOutput out = prev->get_output();
      CHECK(out.sampled_token_ids[0] ==
            std::vector<int32_t>{static_cast<int32_t>(step - 1)});
    }
    prev = std::move(cur);
  }
  if (prev) (void)prev->get_output();

  // The pool never grew past its seed: every step reused a persistent slot.
  CHECK(pool.num_slots() == seed_slots);

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}

// A released slot is handed back out on the next Acquire (buffer recycled), and
// an un-consumed (destroyed) output also releases its slot — no leak.
TEST_CASE("AsyncOutputPool Acquire reuses a released slot's buffers") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  AsyncOutputPool pool(vt::Device{vt::DeviceType::kCPU, 0}, /*capacity_elems=*/8,
                       /*initial_slots=*/1);
  CHECK(pool.num_slots() == 1);

  // First output consumed -> slot released.
  {
    auto a = MakeAsyncOutput({11}, pool, main_q, copy_q);
    (void)a->get_output();
  }
  // Second output DESTROYED without get_output -> slot still released (no leak,
  // no growth).
  { auto b = MakeAsyncOutput({22}, pool, main_q, copy_q); }
  CHECK(pool.num_slots() == 1);

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}
