// Ported from: vllm/v1/kv_offload/tiering/fs/{manager,thread_pool}.py @ e24d1b24
// See include/vllm/v1/kv_offload/fs_tier.h for scope and the two deliberate
// excesses over upstream (the verified header, the byte budget).
#include "vllm/v1/kv_offload/fs_tier.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

#include "vllm/v1/kv_offload/cache_policy.h"

namespace vllm::v1::kv_offload {
namespace {

// Upstream: DualQueueThreadPool (thread_pool.py:41-180). Separate read and
// write deques with dedicated workers, but each worker may drain the OTHER
// deque when its own is empty — so a burst of writes cannot starve reads and
// vice versa. That cross-draining is the whole point; a single shared queue
// would let a long write batch delay every load behind it.
class DualQueueThreadPool {
 public:
  using Task = std::function<void()>;

  DualQueueThreadPool(int num_read_threads, int num_write_threads) {
    const int readers = std::max(1, num_read_threads);
    const int writers = std::max(1, num_write_threads);
    for (int i = 0; i < readers; ++i) {
      threads_.emplace_back([this] { Run(/*prefer_reads=*/true); });
    }
    for (int i = 0; i < writers; ++i) {
      threads_.emplace_back([this] { Run(/*prefer_reads=*/false); });
    }
  }

  ~DualQueueThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : threads_) {
      t.join();
    }
  }

  void Submit(Task task, bool is_read) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      (is_read ? reads_ : writes_).push_back(std::move(task));
    }
    cv_.notify_one();
  }

 private:
  void Run(bool prefer_reads) {
    for (;;) {
      Task task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] {
          return stopping_ || !reads_.empty() || !writes_.empty();
        });
        auto& first = prefer_reads ? reads_ : writes_;
        auto& second = prefer_reads ? writes_ : reads_;
        if (!first.empty()) {
          task = std::move(first.front());
          first.pop_front();
        } else if (!second.empty()) {
          // Drain the other queue rather than idling — the anti-starvation
          // property.
          task = std::move(second.front());
          second.pop_front();
        } else {
          // stopping_ with both queues drained.
          return;
        }
      }
      task();
    }
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Task> reads_;
  std::deque<Task> writes_;
  std::vector<std::thread> threads_;
  bool stopping_ = false;
};

}  // namespace

struct FileSystemTier::Impl {
  FileSystemTierOptions options;
  std::unique_ptr<FileMapper> mapper;
  std::string identity_digest;
  int64_t payload_size = 0;
  // Bytes one block actually occupies on disk (header included) — what the
  // byte budget must account against, not the payload alone.
  int64_t bytes_per_block = 0;
  int64_t capacity_blocks = 0;  // 0 == unbounded

  // The §B3 index: which blocks this tier holds, in replacement order. Nothing
  // upstream corresponds to this — upstream has no accounting at all.
  mutable std::mutex mu;
  std::unique_ptr<CachePolicy> policy;
  int64_t num_blocks = 0;
  int64_t num_evicted = 0;

  std::unique_ptr<DualQueueThreadPool> pool;

  // Async job bookkeeping.
  std::mutex jobs_mu;
  std::condition_variable jobs_cv;
  struct Job {
    bool done = false;
    bool result = false;
    std::exception_ptr error;
  };
  std::unordered_map<int64_t, Job> jobs;
  int64_t next_job_id = 1;

  BlockFileHeader MakeHeader(const OffloadKey& key) const {
    BlockFileHeader h;
    h.format_version = options.identity.format_version;
    h.payload_size = static_cast<uint64_t>(payload_size);
    h.identity_digest = identity_digest;
    h.key = key;
    return h;
  }

  // Reclaim room for `needed` more blocks. Caller holds `mu`. Returns false
  // when the budget cannot be met (the store is then skipped — a control path,
  // not an error, exactly like the CPU tier's prepare_store).
  bool MakeRoom(int64_t needed) {
    if (capacity_blocks == 0) {
      return true;  // unbounded (upstream's behaviour)
    }
    const int64_t over = num_blocks + needed - capacity_blocks;
    if (over <= 0) {
      return true;
    }
    OffloadKeySet none;
    auto evicted = policy->evict(over, none);
    if (!evicted.has_value()) {
      return false;
    }
    for (const auto& [key, block] : *evicted) {
      (void)block;
      std::error_code ec;
      std::filesystem::remove(mapper->file_name(key), ec);
      num_blocks -= 1;
      num_evicted += 1;
    }
    return true;
  }
};

FileSystemTier::FileSystemTier(FileSystemTierOptions options)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = std::move(options);
  impl_->payload_size = impl_->options.identity.page_size_bytes;
  if (impl_->payload_size <= 0) {
    throw std::runtime_error(
        "kv_offload: cache identity has a non-positive page_size_bytes");
  }
  impl_->bytes_per_block =
      impl_->payload_size + static_cast<int64_t>(kBlockHeaderBytes);
  impl_->capacity_blocks =
      impl_->options.capacity_bytes > 0
          ? impl_->options.capacity_bytes / impl_->bytes_per_block
          : 0;
  if (impl_->options.capacity_bytes > 0 && impl_->capacity_blocks == 0) {
    throw std::runtime_error(
        "kv_offload: capacity_bytes is smaller than a single block");
  }

  impl_->mapper = std::make_unique<FileMapper>(impl_->options.root_dir,
                                               impl_->options.identity);
  impl_->identity_digest = impl_->options.identity.Digest();
  // Reads config.json back and REFUSES on any field mismatch.
  impl_->mapper->OpenOrCreate();

  impl_->policy = make_cache_policy(
      impl_->options.eviction_policy,
      impl_->capacity_blocks > 0 ? impl_->capacity_blocks : 1);

  // Rebuild the index from what is already on disk, so a restart honours the
  // byte budget instead of starting from a false zero. Files are visited in
  // modification-time order, which is the best cross-restart approximation of
  // recency available without a sidecar index.
  const std::string rank_root =
      impl_->mapper->base_path() + "_r" +
      std::to_string(impl_->options.identity.rank);
  std::error_code ec;
  if (std::filesystem::exists(rank_root, ec)) {
    std::vector<std::pair<std::filesystem::file_time_type, OffloadKey>> found;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(rank_root, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file() || entry.path().extension() != ".bin") {
        continue;
      }
      // Recover the key from the file's own header rather than from its path:
      // the header is the authority, and a file whose header is unreadable is
      // exactly the kind of debris the self-healing path should drop.
      std::ifstream in(entry.path(), std::ios::binary);
      char header_buf[kBlockHeaderBytes];
      if (!in.read(header_buf, kBlockHeaderBytes)) {
        continue;
      }
      try {
        const BlockFileHeader h =
            BlockFileHeader::Decode(header_buf, kBlockHeaderBytes);
        if (h.identity_digest != impl_->identity_digest) {
          continue;
        }
        found.emplace_back(entry.last_write_time(), h.key);
      } catch (const std::exception&) {
        continue;
      }
    }
    std::sort(found.begin(), found.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [when, key] : found) {
      (void)when;
      BlockStatus block(0);
      block.ref_cnt = 0;  // on disk and idle
      impl_->policy->insert(key, block);
      impl_->num_blocks += 1;
    }
  }

  impl_->pool = std::make_unique<DualQueueThreadPool>(
      impl_->options.num_read_threads, impl_->options.num_write_threads);
}

FileSystemTier::~FileSystemTier() {
  // Destroy the pool first: outstanding jobs reference impl_ state.
  impl_->pool.reset();
}

const std::string& FileSystemTier::base_path() const {
  return impl_->mapper->base_path();
}

LookupResults FileSystemTier::lookup(const std::vector<OffloadKey>& keys) const {
  LookupResults results(keys.size(), false);
  std::lock_guard<std::mutex> lock(impl_->mu);
  for (size_t i = 0; i < keys.size(); ++i) {
    // The in-memory index is authoritative for this process; existence on disk
    // is what it was built from.
    results[i] = impl_->policy->get(keys[i]) != nullptr;
  }
  return results;
}

void FileSystemTier::store(const OffloadKey& key, const void* payload,
                           size_t payload_size) {
  if (static_cast<int64_t>(payload_size) != impl_->payload_size) {
    throw std::runtime_error(
        "kv_offload: block payload size does not match the cache identity's "
        "page_size_bytes");
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->policy->get(key) != nullptr) {
      return;  // already stored (existence-skip)
    }
    if (!impl_->MakeRoom(1)) {
      // Budget cannot be met: SKIP the store. Losing a cache entry is
      // semantically transparent.
      return;
    }
  }

  store_block(impl_->mapper->file_name(key), impl_->MakeHeader(key), payload,
              payload_size);

  std::lock_guard<std::mutex> lock(impl_->mu);
  if (impl_->policy->get(key) == nullptr) {
    BlockStatus block(0);
    block.ref_cnt = 0;
    impl_->policy->insert(key, block);
    impl_->num_blocks += 1;
  }
}

bool FileSystemTier::load(const OffloadKey& key, void* out,
                          size_t out_capacity) {
  const std::string path = impl_->mapper->file_name(key);
  bool hit = false;
  try {
    hit = load_block(path, impl_->MakeHeader(key), out, out_capacity);
  } catch (...) {
    // load_block already unlinked the offending file (self-healing); drop it
    // from the index too so the next lookup is a clean miss rather than a
    // repeating throw.
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->policy->get(key) != nullptr) {
      impl_->policy->remove(key);
      impl_->num_blocks -= 1;
    }
    throw;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);
  if (!hit) {
    if (impl_->policy->get(key) != nullptr) {
      impl_->policy->remove(key);
      impl_->num_blocks -= 1;
    }
    return false;
  }
  // A read is a use: refresh recency so a hot prefix is the last thing evicted.
  impl_->policy->touch({key});
  return true;
}

int64_t FileSystemTier::submit_store(const OffloadKey& key, const void* payload,
                                     size_t payload_size) {
  int64_t id;
  {
    std::lock_guard<std::mutex> lock(impl_->jobs_mu);
    id = impl_->next_job_id++;
    impl_->jobs[id] = Impl::Job{};
  }
  impl_->pool->Submit(
      [this, id, key, payload, payload_size] {
        Impl::Job job;
        try {
          store(key, payload, payload_size);
          job.result = true;
        } catch (...) {
          job.error = std::current_exception();
        }
        job.done = true;
        {
          std::lock_guard<std::mutex> lock(impl_->jobs_mu);
          impl_->jobs[id] = std::move(job);
        }
        impl_->jobs_cv.notify_all();
      },
      /*is_read=*/false);
  return id;
}

int64_t FileSystemTier::submit_load(const OffloadKey& key, void* out,
                                    size_t out_capacity) {
  int64_t id;
  {
    std::lock_guard<std::mutex> lock(impl_->jobs_mu);
    id = impl_->next_job_id++;
    impl_->jobs[id] = Impl::Job{};
  }
  impl_->pool->Submit(
      [this, id, key, out, out_capacity] {
        Impl::Job job;
        try {
          job.result = load(key, out, out_capacity);
        } catch (...) {
          job.error = std::current_exception();
        }
        job.done = true;
        {
          std::lock_guard<std::mutex> lock(impl_->jobs_mu);
          impl_->jobs[id] = std::move(job);
        }
        impl_->jobs_cv.notify_all();
      },
      /*is_read=*/true);
  return id;
}

bool FileSystemTier::wait(int64_t job_id) {
  std::unique_lock<std::mutex> lock(impl_->jobs_mu);
  impl_->jobs_cv.wait(lock, [&] {
    auto it = impl_->jobs.find(job_id);
    return it == impl_->jobs.end() || it->second.done;
  });
  auto it = impl_->jobs.find(job_id);
  if (it == impl_->jobs.end()) {
    throw std::runtime_error("kv_offload: unknown job id");
  }
  Impl::Job job = std::move(it->second);
  impl_->jobs.erase(it);
  lock.unlock();
  if (job.error) {
    std::rethrow_exception(job.error);
  }
  return job.result;
}

void FileSystemTier::reset_cache() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->policy->clear();
  impl_->num_blocks = 0;
  std::error_code ec;
  const std::string rank_root =
      impl_->mapper->base_path() + "_r" +
      std::to_string(impl_->options.identity.rank);
  std::filesystem::remove_all(rank_root, ec);
}

int64_t FileSystemTier::num_blocks() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->num_blocks;
}

int64_t FileSystemTier::bytes_used() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->num_blocks * impl_->bytes_per_block;
}

int64_t FileSystemTier::capacity_bytes() const {
  return impl_->options.capacity_bytes;
}

int64_t FileSystemTier::num_evicted() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->num_evicted;
}

}  // namespace vllm::v1::kv_offload
