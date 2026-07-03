// Ported from: vllm/v1/worker/block_table.py @ e24d1b24
//
// Scope (M1.5 Task 1): the per-request KV-cache block-id storage + slot mapping
// the persistent InputBatch (M1.5 Task 2) and step-input build (M1.5 Task 3)
// consume. Two ported types:
//   * BlockTable: a 2D block-id table [max_num_reqs, max_num_blocks_per_req]
//     plus a per-row length (num_blocks_per_row); rows are added / appended /
//     moved / swapped / cleared as the batch admits / condenses requests, and
//     compute_slot_mapping turns (block table row, token positions) into the
//     flat KV-cache slot ids the attention op writes to.
//   * MultiGroupBlockTable: one BlockTable per KV cache group (the hybrid gate
//     models have 2 groups — full-attention + GDN/mamba state); every mutation
//     fans out to each group's table with that group's block_size.
// Behavioral only: no CUDA, no model. The "tensors" are plain host arrays.
//
// ─── STALE-BRIEF / PATH DEVIATION (recorded) ────────────────────────────────
// The M1.5 brief points at `vllm/v1/worker/gpu/block_table.py` ("the MRV2 gpu/
// package") and asks for `BlockTable` + `MultiGroupBlockTable` with
// add_row / append_row / move_row / swap_row / clear / compute_slot_mapping /
// commit_block_table. Verified against the PINNED e24d1b24 tree, that API does
// NOT live in gpu/block_table.py: at e24d1b24 `gpu/block_table.py` is a
// DIFFERENT, later design — a single `BlockTables` (plural) class built on
// StagedWriteTensor / UvaBackedTensor + fused Triton writers, with
// append_block_ids / apply_staged_writes / gather_block_tables /
// compute_slot_mappings. That `BlockTables` is what the true MRV2 runner
// (`vllm/v1/worker/gpu/model_runner.py`) uses. The `BlockTable` /
// `MultiGroupBlockTable` API the brief describes — 2D table + num_blocks_per_row,
// add_row/append_row/move_row/swap_row/clear, compute_slot_mapping (block_id *
// block_size + within-block offset), commit_block_table — lives in the V1 GPU
// worker file `vllm/v1/worker/block_table.py`, and is consumed by
// `vllm/v1/worker/gpu_input_batch.py` (and tpu_input_batch.py). The brief's
// detailed scope (method names, fields, the [48,112,113] slot oracle, the
// per-group fanout) matches THAT file exactly, so it is what is ported here.
// The C++ file location follows the brief's `worker/gpu/` path (the project's
// committed layout for the InputBatch that consumes this); the `Ported from`
// ref above cites the ACTUAL upstream source.
//
// ─── HOST-ARRAY-FOR-DEVICE-TENSOR DEVIATION (recorded) ──────────────────────
// Upstream stores each table in a `CpuGpuBuffer` (a paired host `.cpu`/`.np`
// numpy array + a device `.gpu` torch tensor); writes land on `.np`, and
// `commit_block_table(num_reqs)` copies `.np -> .gpu` (CpuGpuBuffer.copy_to_gpu)
// before the kernel reads `.gpu`. At T0 there is no device: BOTH buffers are
// host `std::vector`s here. The cpu-vs-device SPLIT is preserved for fidelity —
//   * writes (add_row/append_row/move_row/swap_row/clear_row) touch the "cpu"
//     buffer (get_cpu_tensor);
//   * the "device" buffer (get_device_tensor) is STALE until
//     commit_block_table(num_reqs) copies the first num_reqs rows across;
//   * compute_slot_mapping reads the "device" buffer, exactly as upstream reads
//     `.gpu` — so a real slot mapping requires a prior commit_block_table.
// The actual host->device copy is the runner's job later; here both are host.
//
// ─── OMISSIONS (torch/distributed-only; documented) ─────────────────────────
//   - pin_memory / torch.device constructor args: pure GPU-transfer knobs, dropped.
//   - pcp/dcp process groups: at T0 there are no distributed groups, so
//     total_cp_world_size == 1 and total_cp_rank == 0 (as upstream falls back to
//     in testing). The full context-parallel slot formula IS ported (interleave
//     + is_local masking) but reduces to block_id*block_size+offset at world 1.
//   - get_numpy_array(): folded into get_cpu_tensor() (the host buffer IS "np").
#ifndef VLLM_V1_WORKER_GPU_BLOCK_TABLE_H_
#define VLLM_V1_WORKER_GPU_BLOCK_TABLE_H_

#include <cstdint>
#include <optional>
#include <vector>

namespace vllm::v1 {

// The per-request KV-cache block table for a single KV cache group.
// (Upstream: vllm/v1/worker/block_table.py BlockTable.)
class BlockTable {
 public:
  // Upstream positional order: block_size, max_num_reqs, max_num_blocks_per_req,
  // max_num_batched_tokens, (pin_memory, device dropped), kernel_block_size,
  // cp_kv_cache_interleave_size. `kernel_block_size` drives the hybrid-block
  // split (allocation block size != attention-kernel block size).
  BlockTable(int block_size, int max_num_reqs, int max_num_blocks_per_req,
             int max_num_batched_tokens, int kernel_block_size,
             int cp_kv_cache_interleave_size = 1);

  // Append block_ids to row_idx (extends the row from its current length).
  void append_row(const std::vector<int>& block_ids, int row_idx);
  // Reset row_idx then append block_ids (a fresh row).
  void add_row(const std::vector<int>& block_ids, int row_idx);
  // Zero row_idx's blocks and reset its length to 0.
  void clear_row(int row_idx);
  // Copy row src -> row tgt (used by InputBatch.condense).
  void move_row(int src, int tgt);
  // Swap rows src and tgt.
  void swap_row(int src, int tgt);

  // Fill slot_mapping()[0 .. num_tokens) with block_id*block_size+offset for
  // every token, then pad the rest to PAD_SLOT_ID. Reads the "device" buffer
  // (requires a prior commit_block_table). query_start_loc has num_reqs+1
  // cumulative per-request token offsets; positions has one entry per token.
  void compute_slot_mapping(int num_reqs,
                            const std::vector<int32_t>& query_start_loc,
                            const std::vector<int64_t>& positions);

  // Copy the first num_reqs rows from the "cpu" buffer to the "device" buffer.
  void commit_block_table(int num_reqs);
  // Zero both buffers.
  void clear();

  // Convert kv-manager block ids to kernel block ids for a hybrid table:
  // each block b expands to [b*bpk, b*bpk+1, ..., b*bpk+(bpk-1)].
  // (Upstream map_to_kernel_blocks; kernel_block_arange folded in.)
  static std::vector<int> map_to_kernel_blocks(const std::vector<int>& block_ids,
                                               int blocks_per_kv_block);

  // Accessors mirroring get_cpu_tensor / get_device_tensor / get_numpy_array.
  // Flat, row-major; the row stride is max_num_blocks_per_req.
  const std::vector<int32_t>& get_cpu_tensor() const { return block_table_cpu_; }
  const std::vector<int32_t>& get_device_tensor() const {
    return block_table_device_;
  }
  // Convenience (row, col) reads for the ported tests.
  int32_t cpu_block_id(int row, int col) const {
    return block_table_cpu_[static_cast<size_t>(row) * max_num_blocks_per_req +
                            col];
  }
  int32_t device_block_id(int row, int col) const {
    return block_table_device_[static_cast<size_t>(row) * max_num_blocks_per_req +
                               col];
  }
  const std::vector<int64_t>& slot_mapping() const { return slot_mapping_; }

  // Public state (mirrors upstream's accessible attributes).
  int block_size;
  int max_num_reqs;
  int max_num_batched_tokens;
  int max_num_blocks_per_req;  // post blocks_per_kv_block multiply (row stride)
  int blocks_per_kv_block;
  bool use_hybrid_blocks;
  std::vector<int32_t> num_blocks_per_row;

 private:
  int cp_kv_cache_interleave_size_;
  int total_cp_world_size_;  // 1 at T0 (no dcp/pcp groups)
  int total_cp_rank_;        // 0 at T0
  std::vector<int32_t> block_table_cpu_;     // CpuGpuBuffer .cpu / .np
  std::vector<int32_t> block_table_device_;  // CpuGpuBuffer .gpu (stale pre-commit)
  std::vector<int64_t> slot_mapping_;        // CpuGpuBuffer .gpu of slot_mapping
};

// One BlockTable per KV cache group; every mutation fans out to each group.
// (Upstream: vllm/v1/worker/block_table.py MultiGroupBlockTable.)
class MultiGroupBlockTable {
 public:
  // Upstream positional order: max_num_reqs, max_model_len,
  // max_num_batched_tokens, (pin_memory, device dropped), block_sizes,
  // kernel_block_sizes, max_num_blocks=None, cp_kv_cache_interleave_size.
  // When max_num_blocks is nullopt it is derived per group as
  // cdiv(max_model_len, block_size) then aligned up to a multiple of
  // (128 / block_size) for block_size <= 128.
  MultiGroupBlockTable(int max_num_reqs, int max_model_len,
                       int max_num_batched_tokens, std::vector<int> block_sizes,
                       std::vector<int> kernel_block_sizes,
                       std::optional<std::vector<int>> max_num_blocks = std::nullopt,
                       int cp_kv_cache_interleave_size = 1);

  // block_ids[i] are the blocks for the i-th group.
  void append_row(const std::vector<std::vector<int>>& block_ids, int row_idx);
  void add_row(const std::vector<std::vector<int>>& block_ids, int row_idx);
  void clear_row(int row_idx);
  void move_row(int src, int tgt);
  void swap_row(int src, int tgt);
  void compute_slot_mapping(int num_reqs,
                            const std::vector<int32_t>& query_start_loc,
                            const std::vector<int64_t>& positions);
  void commit_block_table(int num_reqs);
  void clear();

  // The BlockTable for the i-th KV cache group (upstream __getitem__).
  BlockTable& operator[](int idx) { return block_tables[static_cast<size_t>(idx)]; }
  const BlockTable& operator[](int idx) const {
    return block_tables[static_cast<size_t>(idx)];
  }

  // Public state (mirrors upstream self.block_tables).
  std::vector<BlockTable> block_tables;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_BLOCK_TABLE_H_
