// Ported from: vllm/v1/kv_offload/tiering/manager.py:123-706 @ e24d1b24
// See include/vllm/v1/kv_offload/tiering_manager.h for the scope, the tier model
// and the deviation on async job completion.
//
// The ORDERING mirrored here is the correctness content, not the data
// structures. Every method's upstream anchor is cited inline.
#include "vllm/v1/kv_offload/tiering_manager.h"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace vllm::v1::kv_offload {

// --- HeapPrimaryByteView -----------------------------------------------------

HeapPrimaryByteView::HeapPrimaryByteView(int64_t num_blocks,
                                         std::size_t page_size_bytes)
    : page_size_bytes_(page_size_bytes),
      data_(static_cast<std::size_t>(num_blocks) * page_size_bytes, 0) {}

void* HeapPrimaryByteView::slot(int64_t cpu_block_id) {
  return data_.data() +
         static_cast<std::size_t>(cpu_block_id) * page_size_bytes_;
}
const void* HeapPrimaryByteView::slot(int64_t cpu_block_id) const {
  return data_.data() +
         static_cast<std::size_t>(cpu_block_id) * page_size_bytes_;
}

// --- TieringOffloadingManager ------------------------------------------------

namespace {
// The CPU block id a CPULoadStoreSpec carries for the i-th key.
int64_t spec_block_id(const std::shared_ptr<LoadStoreSpec>& spec, size_t i) {
  auto* cpu = dynamic_cast<CPULoadStoreSpec*>(spec.get());
  if (cpu == nullptr) {
    throw std::logic_error(
        "TieringOffloadingManager: primary tier did not return a CPU spec");
  }
  if (i >= cpu->block_ids.size()) {
    throw std::logic_error(
        "TieringOffloadingManager: CPU spec block id out of range");
  }
  return cpu->block_ids[i];
}
}  // namespace

struct TieringOffloadingManager::Impl {
  std::unique_ptr<CPUOffloadingManager> primary;
  std::unique_ptr<FileSystemTier> secondary;
  PrimaryByteView* view = nullptr;

  // Upstream PendingPromotion (manager.py:57-63): per-request accumulator of the
  // secondary->primary loads deferred until on_schedule_end.
  struct PendingPromotion {
    ReqContext ctx;
    std::vector<OffloadKey> keys;
    std::vector<int64_t> block_ids;  // reserved primary slots to load into
  };
  std::unordered_map<std::string, PendingPromotion> pending_promotions;

  // Upstream _transfer_jobs (manager.py:164): in-flight secondary jobs.
  struct JobMeta {
    OffloadKey key;
    int64_t block_id = 0;
    bool is_promotion = false;  // true: secondary->primary; false: cascade
    ReqContext ctx;
    int64_t fs_job_id = 0;
  };
  std::unordered_map<int64_t, JobMeta> transfer_jobs;
  int64_t next_job_id = 0;

  // Upstream _processed_jobs_this_step (manager.py:176): the once-per-step gate.
  bool processed_this_step = false;

  // Upstream RequestState (manager.py:66-71).
  struct ReqState {
    ReqContext ctx;
    int pending_primary_stores = 0;
    bool is_finished = false;
  };
  std::unordered_map<std::string, ReqState> req_state;

  // §B2 evidence: promotions that REFUSED a foreign/corrupt disk block.
  int64_t refusals = 0;
  int64_t promotions = 0;
  int64_t cascades = 0;

  // manager.py:189-200. First call per step polls; the rest no-op.
  void maybe_process_finished_jobs() {
    if (processed_this_step) return;
    processed_this_step = true;
    process_finished_jobs();
  }

  // manager.py:202-235. Poll each in-flight secondary job to completion and
  // finalize the primary-tier bookkeeping. (Our FS completion primitive is a
  // blocking wait(); see the header deviation.)
  void process_finished_jobs() {
    if (transfer_jobs.empty()) return;
    std::vector<int64_t> done;
    for (auto& [id, job] : transfer_jobs) {
      finalize_one(job);
      done.push_back(id);
    }
    for (int64_t id : done) transfer_jobs.erase(id);
  }

  void finalize_one(JobMeta& job) {
    if (job.is_promotion) {
      // Secondary -> primary completed. complete_write == complete_store makes
      // the promoted block readable (manager.py:225-229). A REFUSAL (foreign /
      // truncated file) throws from wait(): roll the reserved slot back and
      // treat the block as ABSENT — never trust the lower tier blindly.
      bool ok = true;
      try {
        secondary->wait(job.fs_job_id);
      } catch (const std::exception&) {
        ok = false;
        refusals += 1;
      }
      primary->complete_store({job.key}, job.ctx, /*success=*/ok);
      if (ok) promotions += 1;
    } else {
      // Cascade (primary -> secondary) completed. complete_read == complete_load
      // releases the pin taken before the write (manager.py:233-235). A failed
      // disk write still unpins.
      try {
        secondary->wait(job.fs_job_id);
        cascades += 1;
      } catch (const std::exception&) {
        // Store failed (e.g. ENOSPC); the block simply is not on disk. Unpin.
      }
      primary->complete_load({job.key}, job.ctx);
    }
  }

  // manager.py:282-329. Reserve a primary slot (marks HIT_PENDING this step) and
  // defer the batched load. Returns false when the primary is full -> MISS.
  bool initiate_promotion(const OffloadKey& key, const ReqContext& ctx) {
    auto reserved = primary->prepare_store({key}, ctx);
    if (!reserved.has_value()) return false;              // manager.py:311-314
    if (reserved->keys_to_store.empty()) {
      // Already resident/in-flight in the primary (a concurrent promotion won
      // the race). Nothing to defer.
      return true;
    }
    const int64_t block_id = spec_block_id(reserved->store_spec, 0);
    auto& p = pending_promotions[ctx.req_id];
    p.ctx = ctx;
    p.keys.push_back(key);
    p.block_ids.push_back(block_id);
    return true;                                          // manager.py:329
  }

  // manager.py:331-353. Submit every deferred promotion's load; one FS job per
  // block. The load reads the disk file into the reserved primary slot.
  void flush_pending_promotions() {
    if (pending_promotions.empty()) return;
    for (auto& [req_id, p] : pending_promotions) {
      for (size_t i = 0; i < p.keys.size(); ++i) {
        void* dst = view->slot(p.block_ids[i]);
        const int64_t fs_id =
            secondary->submit_load(p.keys[i], dst, view->page_size_bytes());
        const int64_t id = next_job_id++;
        transfer_jobs[id] = JobMeta{p.keys[i], p.block_ids[i],
                                    /*is_promotion=*/true, p.ctx, fs_id};
      }
    }
    pending_promotions.clear();
  }
};

TieringOffloadingManager::TieringOffloadingManager(
    std::unique_ptr<CPUOffloadingManager> primary,
    std::unique_ptr<FileSystemTier> secondary, PrimaryByteView& primary_view)
    : impl_(std::make_unique<Impl>()) {
  impl_->primary = std::move(primary);
  impl_->secondary = std::move(secondary);
  impl_->view = &primary_view;
  if (impl_->view->page_size_bytes() == 0) {
    throw std::invalid_argument(
        "TieringOffloadingManager: primary byte view has zero page size");
  }
}

TieringOffloadingManager::~TieringOffloadingManager() {
  // Drain quietly so no background FS job outlives the primary store it reads.
  try {
    impl_->process_finished_jobs();
  } catch (...) {
  }
}

// manager.py:237-280.
LookupResult TieringOffloadingManager::lookup(const OffloadKey& key,
                                              const ReqContext& req_context) {
  impl_->maybe_process_finished_jobs();  // manager.py:260

  const LookupResult primary_hit = impl_->primary->lookup(key, req_context);
  if (primary_hit == LookupResult::kHit) return LookupResult::kHit;
  if (primary_hit == LookupResult::kHitPending)
    return LookupResult::kHitPending;  // primary write in-flight

  // Primary miss -> probe the disk tier. Existence IS the index.
  const LookupResults present = impl_->secondary->lookup({key});
  if (!present.empty() && present[0]) {
    if (!impl_->initiate_promotion(key, req_context)) {
      return LookupResult::kMiss;  // primary full (manager.py:272-273)
    }
    return LookupResult::kRetry;  // ask again next step (manager.py:274)
  }
  return LookupResult::kMiss;
}

// manager.py:355-378.
std::shared_ptr<LoadStoreSpec> TieringOffloadingManager::prepare_load(
    const std::vector<OffloadKey>& keys, const ReqContext& req_context) {
  impl_->maybe_process_finished_jobs();  // manager.py:376
  return impl_->primary->prepare_load(keys, req_context);
}

// manager.py:380-391. Touch every tier so a shared prefix is the last evicted.
void TieringOffloadingManager::touch(const std::vector<OffloadKey>& keys,
                                     const ReqContext& req_context) {
  impl_->primary->touch(keys, req_context);
  // The FS tier's recency is its CachePolicy (bounded case); a no-op otherwise.
}

// manager.py:393-405.
void TieringOffloadingManager::complete_load(
    const std::vector<OffloadKey>& keys, const ReqContext& req_context) {
  impl_->primary->complete_load(keys, req_context);
}

// manager.py:407-459. A GPU->primary store; the cascade to disk is deferred to
// complete_store (docstring 434-437).
std::optional<PrepareStoreOutput> TieringOffloadingManager::prepare_store(
    const std::vector<OffloadKey>& keys, const ReqContext& req_context) {
  impl_->maybe_process_finished_jobs();  // manager.py:432
  auto result = impl_->primary->prepare_store(keys, req_context);
  if (!result.has_value()) return std::nullopt;  // manager.py:440-441
  if (!result->keys_to_store.empty()) {
    // Track outstanding GPU->primary stores so finalization waits (manager.py
    // :443-445).
    auto it = impl_->req_state.find(req_context.req_id);
    if (it != impl_->req_state.end()) it->second.pending_primary_stores += 1;
  }
  return result;
}

// manager.py:497-556. Finalize the primary store, then cascade the newly-stored
// blocks to every secondary tier while KEEPING them pinned until the write lands.
void TieringOffloadingManager::complete_store(
    const std::vector<OffloadKey>& keys, const ReqContext& req_context,
    bool success) {
  impl_->primary->complete_store(keys, req_context, success);  // manager.py:523

  if (success) {
    // Which of `keys` are now HIT (readable) in the primary? Only those can be
    // cascaded. prepare_load pins them (ref_cnt++) for the duration of the disk
    // write (manager.py:532).
    std::vector<OffloadKey> ready;
    for (const OffloadKey& k : keys) {
      if (impl_->primary->lookup(k, req_context) == LookupResult::kHit) {
        ready.push_back(k);
      }
    }
    if (!ready.empty()) {
      auto spec = impl_->primary->prepare_load(ready, req_context);
      for (size_t i = 0; i < ready.size(); ++i) {
        const int64_t block_id = spec_block_id(spec, i);
        const void* src = impl_->view->slot(block_id);
        const int64_t fs_id = impl_->secondary->submit_store(
            ready[i], src, impl_->view->page_size_bytes());
        const int64_t id = impl_->next_job_id++;
        impl_->transfer_jobs[id] = Impl::JobMeta{
            ready[i], block_id, /*is_promotion=*/false, req_context, fs_id};
      }
    }
  }

  // Decrement the outstanding-store count and finalize if the request is done
  // (manager.py:552-556).
  auto it = impl_->req_state.find(req_context.req_id);
  if (it != impl_->req_state.end() && it->second.pending_primary_stores > 0) {
    it->second.pending_primary_stores -= 1;
    if (it->second.is_finished && it->second.pending_primary_stores == 0) {
      impl_->req_state.erase(it);
    }
  }
}

// manager.py:642-681. THE reset ordering: drain secondaries first, do NOT reset
// them (persistent stores survive), then reset the primary.
void TieringOffloadingManager::reset_cache() {
  // 1. Drain the secondary's in-flight jobs FIRST (manager.py:657-658).
  drain_jobs();
  // 2. Deferred promotions reserved primary slots the reset invalidates; their
  //    load never ran, so no disk IO touches that memory (manager.py:666).
  impl_->pending_promotions.clear();
  // 3. Per-request cleanup (manager.py:668-675): active requests are retained;
  //    finished ones are dropped.
  std::vector<std::string> finished;
  for (auto& [req_id, state] : impl_->req_state) {
    state.pending_primary_stores = 0;
    if (state.is_finished) finished.push_back(req_id);
  }
  for (const std::string& r : finished) impl_->req_state.erase(r);
  // 4. Only NOW reset the primary (manager.py:677). The disk tier is DELIBERATELY
  //    NOT reset — a persisted cache survives a prefix-cache reset.
  impl_->primary->reset_cache();
  impl_->processed_this_step = false;
}

std::vector<OffloadingEvent> TieringOffloadingManager::take_events() {
  // manager.py:629-640: manager-level events then the primary's.
  return impl_->primary->take_events();
}

// manager.py:558-580. No request-level secondary tiers today, so this only
// establishes the per-request state that gates cascade finalization.
void TieringOffloadingManager::on_new_request(const ReqContext& req_context) {
  Impl::ReqState state;
  state.ctx = req_context;
  impl_->req_state[req_context.req_id] = state;
}

// manager.py:582-604.
void TieringOffloadingManager::on_request_finished(
    const ReqContext& req_context) {
  auto it = impl_->req_state.find(req_context.req_id);
  if (it == impl_->req_state.end()) return;
  it->second.is_finished = true;
  if (it->second.pending_primary_stores == 0) {
    impl_->req_state.erase(it);  // manager.py:599-604
  }
}

// manager.py:606-618. Once per step.
void TieringOffloadingManager::on_schedule_end() {
  impl_->maybe_process_finished_jobs();      // manager.py:614
  impl_->processed_this_step = false;        // manager.py:615
  impl_->flush_pending_promotions();         // manager.py:616
}

// manager.py:620-627.
bool TieringOffloadingManager::has_pending_work() const {
  return !impl_->transfer_jobs.empty() || !impl_->pending_promotions.empty();
}

void TieringOffloadingManager::drain_jobs() {
  // Wait every in-flight secondary job to completion and finalize it. Called by
  // reset_cache and by the correctness tests to drive promotion deterministically.
  impl_->process_finished_jobs();
}

TieringStats TieringOffloadingManager::get_stats() {
  TieringStats s;
  s.cpu_cache_usage = impl_->primary->get_stats().cpu_cache_usage;
  s.fs_num_blocks = impl_->secondary->num_blocks();
  s.fs_bytes_used = impl_->secondary->bytes_used();
  s.fs_num_evicted = impl_->secondary->num_evicted();
  s.promotions = impl_->promotions;
  s.cascades = impl_->cascades;
  s.refusals = impl_->refusals;
  return s;
}

}  // namespace vllm::v1::kv_offload
