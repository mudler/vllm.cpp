// Tests for AsyncGPUModelRunnerOutput (ENG-ASYNC-SCHED W3 sampler-OUTPUT half).
// Ports the intent of vllm/v1/worker/gpu/async_utils.py's AsyncOutput +
// gpu_model_runner.py:242-332 AsyncGPUModelRunnerOutput contract: a device-
// resident sampled-id snapshot is D2H'd on a copy queue with an event, and
// get_output() blocks only that event and returns the materialized
// ModelRunnerOutput. On CPU (unified memory) the copy is a memcpy and events are
// no-ops, so get_output() must be token-exact.
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>

#include "vllm/v1/engine/types.h"
#include "vllm/v1/worker/gpu/async_output.h"
#include "vt/backend.h"
#include "vt/device.h"

using vllm::v1::AsyncGPUModelRunnerOutput;
using vllm::v1::ModelRunnerOutput;

namespace {

// Build a device (unified == host) int64 buffer of sampled ids and wrap it in an
// AsyncGPUModelRunnerOutput exactly as the runner's sample_tokens_async does.
std::unique_ptr<AsyncGPUModelRunnerOutput> MakeAsyncOutput(
    const std::vector<int64_t>& ids, vt::Queue& main_q, vt::Queue& copy_q) {
  const vt::Device dev{vt::DeviceType::kCPU, 0};
  const int num_reqs = static_cast<int>(ids.size());
  void* dev_ids = vt::Alloc(dev, ids.empty() ? 1 : ids.size() * sizeof(int64_t));
  if (!ids.empty()) {
    vt::GetBackend(dev.type).Copy(main_q, dev_ids, ids.data(),
                                  ids.size() * sizeof(int64_t));
  }
  ModelRunnerOutput skeleton;
  for (int i = 0; i < num_reqs; ++i) {
    const std::string id = "R" + std::to_string(i);
    skeleton.req_ids.push_back(id);
    skeleton.req_id_to_index[id] = i;
  }
  return std::make_unique<AsyncGPUModelRunnerOutput>(
      std::move(skeleton), dev, dev_ids, num_reqs, main_q, copy_q);
}

}  // namespace

TEST_CASE("AsyncGPUModelRunnerOutput materializes the sampled ids via get_output") {
  vt::Queue main_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});
  vt::Queue copy_q = vt::CreateQueue(vt::Device{vt::DeviceType::kCPU, 0});

  const std::vector<int64_t> ids = {42, 7, 1001, 0};
  auto async = MakeAsyncOutput(ids, main_q, copy_q);
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

  auto async = MakeAsyncOutput({}, main_q, copy_q);
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

  auto a = MakeAsyncOutput({555, 666}, main_q, copy_q);
  ModelRunnerOutput out = a->get_output();
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{555});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{666});

  vt::DestroyQueue(main_q);
  vt::DestroyQueue(copy_q);
}
