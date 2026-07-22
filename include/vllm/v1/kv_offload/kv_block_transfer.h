// Ported from: vllm/v1/kv_offload/cpu/gpu_worker.py:73-451 @ e24d1b24
//               (structure only — see the deviations below)
//
// Scope: the byte movement half of the CPU offload tier — the pinned host
// backing store that holds offloaded blocks, and the batched device<->host
// transfer worker that fills and drains it.
//
// STRUCTURE PORTED (this is the part that is load-bearing):
//   * transfers run on a SIDE queue, never the compute queue, so offloading
//     cannot serialize behind the model (gpu_worker.py:153-163);
//   * the side queue WAITS an event recorded on the compute queue before
//     reading device KV, so it can never read a page the model is still
//     writing (gpu_worker.py:289-296);
//   * completion is EVENT-POLLED, not synchronized: `poll()` never blocks, and
//     only the explicit `wait()` blocks the host (gpu_worker.py:395-404);
//   * the host destination is PAGE-LOCKED (vt::Backend::AllocPinned) so a copy
//     engine can DMA into it without a staging bounce (gpu_worker.py:501-506).
//
// R4 — SOURCE-ACCESS ORDERING IS A CORRECTNESS REQUIREMENT WEARING THE COSTUME
// OF A TUNING KNOB (.agents/specs/kv-persistence-lmcache.md §Risks R4).
// Upstream sets `is_src_access_order_any = not gpu_to_cpu`
// (gpu_worker.py:388-394): a CPU->GPU copy may use
// CU_MEMCPY_SRC_ACCESS_ORDER_ANY, but a GPU->CPU copy MUST keep STREAM source
// ordering, because the compute stream is still writing the source. Getting it
// backwards produces TORN BLOCKS under load and nothing at all under a light
// test. That flag exists only on `cuMemcpyBatchAsync`; `vt::Backend::Copy` is
// stream-ordered in BOTH directions, which is the SAFE side of the asymmetry
// (we give up an optimization on the H2D leg, never a guarantee on the D2H
// leg). If a batched-memcpy fast path is ever added it MUST re-derive this
// flag from the direction, never expose it as a knob.
//
// DEVIATIONS, recorded:
//   * upstream's `ops.swap_blocks_batch` / `cuMemcpyBatchAsync` /
//     Triton-under-28KiB kernel selection (gpu_worker.py:35-58) is NOT ported;
//     the correct first implementation upstream itself falls back to is a
//     per-block `cudaMemcpyAsync` loop (cache_kernels.cu:69-76), which is what
//     `vt::Backend::Copy` gives us on every backend. A batched fast path is a
//     later optimization, gated on a measurement.
//   * the /dev/shm multi-worker shared region (cpu/shared_offload_region.py) is
//     not ported: it exists to share one CPU tier across ranks, and we are
//     single-rank today.
#ifndef VLLM_V1_KV_OFFLOAD_KV_BLOCK_TRANSFER_H_
#define VLLM_V1_KV_OFFLOAD_KV_BLOCK_TRANSFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vt/backend.h"

namespace vllm::v1::kv_offload {

// One KV cache layer's flat byte region. Block `b` of the layer occupies
// [base + b * page_size_bytes, base + (b + 1) * page_size_bytes).
//
// The region is treated as OPAQUE BYTES. No layout is interpreted here — which
// is exactly what lets one code path serve full attention (rank-4, interleaved
// [K|V], a factor of 2 in the block stride) and MLA (rank-3, ONE 576-wide
// latent, no separate V) without a special case. See
// .agents/specs/kv-persistence-lmcache.md §Risks R9.
struct KVLayerRegion {
  void* base = nullptr;
  size_t page_size_bytes = 0;
  size_t num_blocks = 0;
};

// Pinned host memory holding `num_blocks` slots of `page_size_bytes` for EACH
// layer. Slot `s` of layer `l` is `slot(l, s)`.
class CPUBackingStore {
 public:
  CPUBackingStore(vt::Device device, size_t num_layers, size_t num_blocks,
                  size_t page_size_bytes);
  ~CPUBackingStore();
  CPUBackingStore(const CPUBackingStore&) = delete;
  CPUBackingStore& operator=(const CPUBackingStore&) = delete;

  void* slot(size_t layer, size_t block_id);
  const void* slot(size_t layer, size_t block_id) const;

  size_t num_layers() const { return num_layers_; }
  size_t num_blocks() const { return num_blocks_; }
  size_t page_size_bytes() const { return page_size_bytes_; }
  size_t total_bytes() const {
    return num_layers_ * num_blocks_ * page_size_bytes_;
  }

 private:
  vt::Device device_;
  size_t num_layers_ = 0;
  size_t num_blocks_ = 0;
  size_t page_size_bytes_ = 0;
  uint8_t* data_ = nullptr;
};

// Direction of a transfer job.
enum class TransferDirection : int {
  // Device KV -> pinned host slots (a STORE). Must stay stream-ordered on the
  // source; see the R4 note above.
  kDeviceToHost = 0,
  // Pinned host slots -> device KV (a LOAD).
  kHostToDevice = 1,
};

// One batched transfer: `device_block_ids[i]` <-> `host_block_ids[i]`, applied
// to every layer of `layers`.
struct TransferJob {
  TransferDirection direction = TransferDirection::kDeviceToHost;
  std::vector<int64_t> device_block_ids;
  std::vector<int64_t> host_block_ids;
};

// Batched device<->host KV block mover on a dedicated side queue.
class KVBlockTransferWorker {
 public:
  // `compute_queue` is the queue the model runs on; every device-to-host job
  // is ordered after the work already submitted to it. `layers` must outlive
  // the worker.
  KVBlockTransferWorker(vt::Device device, vt::Queue& compute_queue,
                        std::vector<KVLayerRegion> layers,
                        CPUBackingStore& store);
  ~KVBlockTransferWorker();
  KVBlockTransferWorker(const KVBlockTransferWorker&) = delete;
  KVBlockTransferWorker& operator=(const KVBlockTransferWorker&) = delete;

  // Submit a job; returns its id. Never blocks.
  int64_t submit(const TransferJob& job);

  // Non-blocking: return the ids of every job that has completed since the last
  // call. Upstream polls `end_event.query()` (gpu_worker.py:395-404).
  std::vector<int64_t> poll();

  // Block the HOST until the given job has completed. The ONLY blocking call.
  void wait(int64_t job_id);

  // Jobs submitted but not yet reported complete.
  size_t num_in_flight() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::v1::kv_offload

#endif  // VLLM_V1_KV_OFFLOAD_KV_BLOCK_TRANSFER_H_
