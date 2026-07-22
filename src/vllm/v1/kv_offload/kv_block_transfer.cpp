// Ported from: vllm/v1/kv_offload/cpu/gpu_worker.py:73-451 @ e24d1b24
// See include/vllm/v1/kv_offload/kv_block_transfer.h for the ported structure,
// the R4 source-access-ordering note, and the recorded deviations.
#include "vllm/v1/kv_offload/kv_block_transfer.h"

#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

namespace vllm::v1::kv_offload {

CPUBackingStore::CPUBackingStore(vt::Device device, size_t num_layers,
                                 size_t num_blocks, size_t page_size_bytes)
    : device_(device),
      num_layers_(num_layers),
      num_blocks_(num_blocks),
      page_size_bytes_(page_size_bytes) {
  const size_t bytes = total_bytes();
  // Page-locked: a copy engine can DMA straight into it with no staging
  // bounce. On unified-memory backends AllocPinned degrades to a plain host
  // allocation, which is already the right thing there.
  data_ = static_cast<uint8_t*>(
      vt::GetBackend(device_.type).AllocPinned(bytes == 0 ? 1 : bytes));
  if (bytes != 0) {
    std::memset(data_, 0, bytes);
  }
}

CPUBackingStore::~CPUBackingStore() {
  if (data_ != nullptr) {
    vt::GetBackend(device_.type).FreePinned(data_);
  }
}

void* CPUBackingStore::slot(size_t layer, size_t block_id) {
  return const_cast<void*>(
      static_cast<const CPUBackingStore*>(this)->slot(layer, block_id));
}

const void* CPUBackingStore::slot(size_t layer, size_t block_id) const {
  if (layer >= num_layers_ || block_id >= num_blocks_) {
    throw std::out_of_range("CPUBackingStore: slot out of range");
  }
  return data_ + ((layer * num_blocks_ + block_id) * page_size_bytes_);
}

struct KVBlockTransferWorker::Impl {
  vt::Device device;
  vt::Backend* backend = nullptr;
  vt::Queue* compute_queue = nullptr;
  // The dedicated side queue. Every transfer runs here so offloading can never
  // serialize behind (or in front of) the model.
  vt::Queue side_queue;
  std::vector<KVLayerRegion> layers;
  CPUBackingStore* store = nullptr;

  struct InFlight {
    int64_t id = 0;
    vt::Event done;
  };
  std::deque<InFlight> in_flight;
  int64_t next_job_id = 1;
};

KVBlockTransferWorker::KVBlockTransferWorker(vt::Device device,
                                             vt::Queue& compute_queue,
                                             std::vector<KVLayerRegion> layers,
                                             CPUBackingStore& store)
    : impl_(std::make_unique<Impl>()) {
  impl_->device = device;
  impl_->backend = &vt::GetBackend(device.type);
  impl_->compute_queue = &compute_queue;
  impl_->side_queue = vt::CreateQueue(device);
  impl_->layers = std::move(layers);
  impl_->store = &store;

  if (impl_->layers.size() != store.num_layers()) {
    throw std::invalid_argument(
        "KVBlockTransferWorker: layer count does not match the backing store");
  }
  for (const KVLayerRegion& layer : impl_->layers) {
    if (layer.page_size_bytes != store.page_size_bytes()) {
      // A mismatch here would silently transfer a partial page. The page size
      // is a per-spec quantity (full attention vs MLA differ by more than a
      // factor of 2), so it must agree exactly.
      throw std::invalid_argument(
          "KVBlockTransferWorker: layer page size does not match the backing "
          "store page size");
    }
  }
}

KVBlockTransferWorker::~KVBlockTransferWorker() {
  // Drain: destroying a queue with outstanding copies into memory we are about
  // to free is a use-after-free, not a leak.
  for (Impl::InFlight& job : impl_->in_flight) {
    impl_->backend->SynchronizeEvent(job.done);
    impl_->backend->DestroyEvent(job.done);
  }
  impl_->in_flight.clear();
  vt::DestroyQueue(impl_->side_queue);
}

int64_t KVBlockTransferWorker::submit(const TransferJob& job) {
  if (job.device_block_ids.size() != job.host_block_ids.size()) {
    throw std::invalid_argument(
        "KVBlockTransferWorker: device and host block id counts differ");
  }

  const bool device_to_host = job.direction == TransferDirection::kDeviceToHost;

  // Order the side queue AFTER the work already submitted to the compute
  // queue. For a STORE this is mandatory: the model may still be writing the
  // very pages we are about to read (gpu_worker.py:289-296). For a LOAD it is
  // equally mandatory in the other sense — we must not overwrite a page the
  // model is still reading.
  vt::Event start = impl_->backend->CreateEvent();
  impl_->backend->RecordEvent(start, *impl_->compute_queue);
  impl_->backend->QueueWaitEvent(impl_->side_queue, start);
  impl_->backend->DestroyEvent(start);

  const size_t page = impl_->store->page_size_bytes();
  for (size_t layer = 0; layer < impl_->layers.size(); ++layer) {
    const KVLayerRegion& region = impl_->layers[layer];
    auto* region_base = static_cast<uint8_t*>(region.base);
    for (size_t i = 0; i < job.device_block_ids.size(); ++i) {
      const int64_t device_block = job.device_block_ids[i];
      const int64_t host_block = job.host_block_ids[i];
      if (device_block < 0 ||
          static_cast<size_t>(device_block) >= region.num_blocks) {
        throw std::out_of_range(
            "KVBlockTransferWorker: device block id out of range");
      }
      void* device_page =
          region_base + static_cast<size_t>(device_block) * page;
      void* host_page = impl_->store->slot(layer, static_cast<size_t>(host_block));
      // Per-block copy loop: the correct first implementation, and the one
      // upstream's own C++ op falls back to (cache_kernels.cu:69-76).
      if (device_to_host) {
        impl_->backend->Copy(impl_->side_queue, host_page, device_page, page);
      } else {
        impl_->backend->Copy(impl_->side_queue, device_page, host_page, page);
      }
    }
  }

  Impl::InFlight entry;
  entry.id = impl_->next_job_id++;
  entry.done = impl_->backend->CreateEvent();
  impl_->backend->RecordEvent(entry.done, impl_->side_queue);
  impl_->in_flight.push_back(entry);

  // A LOAD's destination is device KV the model will read next step, so the
  // compute queue must wait for it. A STORE's destination is host memory the
  // model never reads, so the compute queue is deliberately NOT held up — that
  // is the whole point of running stores on a side queue.
  if (!device_to_host) {
    impl_->backend->QueueWaitEvent(*impl_->compute_queue, entry.done);
  }
  return entry.id;
}

std::vector<int64_t> KVBlockTransferWorker::poll() {
  std::vector<int64_t> finished;
  // Jobs complete in submission order on a single queue, so stop at the first
  // outstanding one rather than scanning the whole deque.
  while (!impl_->in_flight.empty()) {
    Impl::InFlight& front = impl_->in_flight.front();
    if (!impl_->backend->QueryEvent(front.done)) {
      break;
    }
    finished.push_back(front.id);
    impl_->backend->DestroyEvent(front.done);
    impl_->in_flight.pop_front();
  }
  return finished;
}

void KVBlockTransferWorker::wait(int64_t job_id) {
  while (!impl_->in_flight.empty() && impl_->in_flight.front().id <= job_id) {
    Impl::InFlight& front = impl_->in_flight.front();
    impl_->backend->SynchronizeEvent(front.done);
    impl_->backend->DestroyEvent(front.done);
    impl_->in_flight.pop_front();
  }
}

size_t KVBlockTransferWorker::num_in_flight() const {
  return impl_->in_flight.size();
}

}  // namespace vllm::v1::kv_offload
