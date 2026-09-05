// The HOST arm of `ModelForwardInput::device_token_ids`. See
// `include/vllm/model_executor/models/host_token_ids.h` for the contract, why
// the existing device consumer cannot serve these callers, and what the
// synchronise costs.
#include "vllm/model_executor/models/host_token_ids.h"

#include <string>

#include "vllm/model_executor/models/model_registry.h"
#include "vt/backend.h"
#include "vt/dtype.h"  // VT_CHECK

namespace vllm {

const std::vector<int32_t>& ResolveHostTokenIds(const ModelForwardInput& input,
                                                std::vector<int32_t>* storage,
                                                const char* what) {
  VT_CHECK(storage != nullptr,
           std::string(what) +
               ": ResolveHostTokenIds needs the caller's storage buffer");
  // THE NULL PATH IS THE COMMON PATH and it does nothing at all. Every non-async
  // and every non-CUDA step arrives here, and returning the caller's own vector
  // by reference is what makes those builds byte-identical rather than merely
  // equivalent.
  if (input.device_token_ids == nullptr) return input.token_ids;

  const int64_t count = static_cast<int64_t>(input.token_ids.size());
  // A published buffer over an EMPTY host vector can only mean the runner and
  // the model disagree about this step's shape. Refused by name rather than
  // silently reduced to a no-op, because a no-op here is exactly the defect
  // this file exists to remove.
  VT_CHECK(count > 0,
           std::string(what) +
               ": the runner published device input ids for a step whose host "
               "token_ids is empty, so the two disagree about the step's shape "
               "(#2544)");

  *storage = input.token_ids;
  vt::Backend& backend = vt::GetBackend(input.queue.device.type);
  // ON THE MAIN QUEUE, so this read is ordered after the combine that produced
  // the source instead of racing it — the same ordering argument
  // `detail::ApplyDeviceTokenIds` makes for the device arm.
  backend.Copy(input.queue, storage->data(), input.device_token_ids,
               static_cast<size_t>(count) * sizeof(int32_t));
  // AND AWAITED, because the caller's very next act is a HOST dereference of
  // these bytes. `Copy` is enqueued; without this the gather would read whatever
  // the host vector held before the copy landed, which is the stale value the
  // whole change is about.
  backend.Synchronize(input.queue);
  return *storage;
}

}  // namespace vllm
