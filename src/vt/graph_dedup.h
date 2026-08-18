// vllm.cpp original — graph-executable dedup behind the vt capture seam.
//
// Row ENG-CUDAGRAPH-DEDUP, issue #1162, spec .agents/specs/eng-cudagraph-dedup.md.
//
// WHAT THIS REPLACES. Backend::EndCaptureGraph instantiates one graph executable per
// capture and destroys the raw graph immediately (src/vt/cuda/cuda_backend.cu, and the
// same shape on hipGraph). A model therefore holds one executable per padded decode
// bucket — 7 of them at max_num_seqs=32 and 11 at 64 — and nine hand-rolled drivers each
// hold their own set (count corrected 2026-08-18, issue #1179). The decode graphs of two
// padded batch sizes are usually the same node topology with different parameters, which
// is exactly the case cudaGraphExecUpdate exists for: re-point ONE executable instead of
// instantiating a second.
//
// This is a MEMORY and CAPTURE-TIME change, not a throughput change. A deduped replay
// launches the same nodes with the same parameters; if it launched anything else that
// would be a correctness defect, which is why the gate on this file is byte-identity
// and not a ratio.
//
// WHY THE SIGNATURE IS ONLY A LOOKUP KEY. Register groups captures by a structural
// signature, but never trusts it: every candidate match is PROBED with the real driver
// update before it is honoured, on a throwaway executable, and a probe the driver
// refuses gives that capture its own executable. So a signature that is too coarse
// costs a wasted probe, or at worst the loud Replay-time VT_CHECK explained at that
// call site when update compatibility turns out not to be transitive across a group.
// A signature can cost a fold; it can never make a replay wrong.
// The alternative — a signature exhaustive enough to be trusted — would make this file
// a correctness surface that has to track every node-parameter class the driver knows.
//
// WHY THE PROBE IS TRANSIENT. SGLang keeps a second `compat_exec` per group alive for
// the whole capture phase and frees it in seal() at end_cuda_graph_capture
// (cuda_graph_dedup_mixin.py:236-238,244-251 @ f63458b5be). OUR SEAM HAS NO CAPTURE
// PHASE: the drivers capture lazily, the first time a padded bucket is seen, interleaved
// with replays of buckets already captured. A persistent probe would therefore never be
// sealed and would hold TWO executables per group for the process, which is worse than
// the one-per-bucket baseline this file exists to reduce. Instantiating the probe inside
// Register and destroying it there keeps steady state at one executable per group, and
// costs exactly what the pre-dedup path already paid: one instantiate per capture.
//
// THREADING. A group's executable is shared state. Two handles that resolve to one
// executable must not be replayed concurrently on two streams, because the second
// replay's update would re-point the executable out from under the first. Today's
// drivers replay decode graphs from one runner thread on one queue, so the constraint
// holds; it is stated here because it is a constraint the seam did not previously carry.
//
// The constraint is wider than "one runner thread", though, and the reason is where the
// registry LIVES: `CudaBackend::dedup_` belongs to a process-singleton per-device
// backend (src/vt/cuda/cuda_backend.cu:355-365), so ONE registry serves every model on
// a device. Its two `unordered_map`s carry no synchronisation, so a second model
// capturing or destroying a graph concurrently with the first is a data race on the
// containers themselves, before any question about a shared executable arises. Correct
// today because the engine drives one device from one thread; a lock belongs here the
// moment that stops being true.
#ifndef VT_GRAPH_DEDUP_H_
#define VT_GRAPH_DEDUP_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vt {

// The device operations the registry needs, supplied by whichever accelerator backend
// owns the capture. Free function pointers rather than virtuals so this header stays
// device-free and unit-testable on every platform, the way
// src/vt/cuda/graph_safe_scratch.h keeps its retire bookkeeping portable.
struct GraphDedupOps {
  // A structural signature of a raw captured graph. Equal signatures are CANDIDATES
  // for sharing one executable; the driver update below is the authority.
  std::string (*signature)(void* raw_graph) = nullptr;
  // cudaGraphInstantiate / hipGraphInstantiate.
  void* (*instantiate)(void* raw_graph) = nullptr;
  // cudaGraphExecUpdate / hipGraphExecUpdate. False on any non-success, with a
  // human-readable reason in `detail` when it is not null.
  bool (*update)(void* exec, void* raw_graph, std::string* detail) = nullptr;
  void (*destroy_exec)(void* exec) = nullptr;
  void (*destroy_graph)(void* raw_graph) = nullptr;
  void (*launch)(void* exec, void* stream) = nullptr;
};

// The polarity of VT_CUDA_GRAPH_DEDUP, split out so it is testable without touching the
// process environment. Anything that is not exactly "1" leaves the capture path
// byte-identical to the pre-dedup one.
inline bool GraphDedupEnabledFor(const char* value) {
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// Read once: the drivers capture lazily and repeatedly, and a knob that could change
// between two captures of the same model would produce a mixed set of handles for no
// stated reason.
inline bool GraphDedupEnabled() {
  static const bool enabled = GraphDedupEnabledFor(std::getenv("VT_CUDA_GRAPH_DEDUP"));
  return enabled;
}

// Owns every captured raw graph handed to it and every executable it instantiates.
// Register returns the opaque handle the vt seam's void* contract already promises; the
// backend routes ReplayGraph / DestroyGraph back here for the handles Owns() claims, and
// down the plain path for the ones it does not.
class GraphDedupRegistry {
 public:
  // `log` receives the capture-count / executable-count line after each registration.
  // Null silences it, which is what the unit suite wants and what a caller that only
  // reads Stats() programmatically can ask for.
  explicit GraphDedupRegistry(const GraphDedupOps& ops, std::FILE* log = stderr)
      : ops_(ops), log_(log) {
    VT_CHECK(ops_.signature != nullptr && ops_.instantiate != nullptr &&
                 ops_.update != nullptr && ops_.destroy_exec != nullptr &&
                 ops_.destroy_graph != nullptr && ops_.launch != nullptr,
             "graph dedup: incomplete device op table");
  }
  ~GraphDedupRegistry() { Close(); }

  GraphDedupRegistry(const GraphDedupRegistry&) = delete;
  GraphDedupRegistry& operator=(const GraphDedupRegistry&) = delete;

  // Takes ownership of `raw_graph` and returns the handle to replay and destroy it by.
  void* Register(void* raw_graph) {
    VT_CHECK(raw_graph != nullptr, "graph dedup: cannot register a null graph");
    const std::string signature = ops_.signature(raw_graph);
    std::vector<std::unique_ptr<Group>>& candidates = groups_[signature];

    Group* group = nullptr;
    for (const std::unique_ptr<Group>& candidate : candidates) {
      if (ProbeAccepts(*candidate, raw_graph)) {
        group = candidate.get();
        break;
      }
    }
    if (group == nullptr) {
      auto created = std::make_unique<Group>();
      created->exec = ops_.instantiate(raw_graph);
      // FAIL HERE, at the capture site, exactly where the pre-dedup path did:
      // EndCaptureGraph wrapped cudaGraphInstantiate in Check(), so an instantiate
      // failure threw with the driver's error code while the caller was still inside
      // the capture it belongs to. Accepting a null executable instead would mint a
      // valid-looking handle over nothing, count it live in ExecCount(), and defer the
      // failure to the first Replay as "graph launch failed" -- the wrong site, the
      // wrong message, and the driver's reason already discarded. The capture's
      // ownership transferred to us at the top of Register, so release it before
      // unwinding rather than leaking it on the way out.
      const bool instantiated = created->exec != nullptr;
      if (!instantiated) ops_.destroy_graph(raw_graph);
      VT_CHECK(instantiated,
               "graph dedup: instantiate failed for the graph being registered at this "
               "capture site; the capture cannot be replayed");
      created->current_raw = raw_graph;
      group = created.get();
      candidates.push_back(std::move(created));
    }
    group->raws.push_back(raw_graph);

    auto handle = std::make_unique<Handle>();
    handle->group = group;
    handle->raw = raw_graph;
    handle->signature = signature;
    void* key = handle.get();
    handles_.emplace(key, std::move(handle));

    if (log_ != nullptr) {
      // Mirrors SGLang's "captured %d CUDA graphs, deduped to %d execs"
      // (cuda_graph_dedup_mixin.py:358). Emitted per registration rather than once at
      // the end because there is no end: see the capture-phase note at the top.
      std::fprintf(log_, "vt graph dedup: captured %zu graphs, deduped to %zu execs\n",
                   CapturedCount(), ExecCount());
    }
    return key;
  }

  bool Owns(void* handle) const { return handles_.find(handle) != handles_.end(); }

  void Replay(void* handle, void* stream) {
    Handle& entry = Lookup(handle);
    Group& group = *entry.group;
    if (group.current_raw != entry.raw) {
      std::string detail;
      const bool ok = ops_.update(group.exec, entry.raw, &detail);
      // THIS IS NOT THE FOLD Register PROBED. Register probes (group.raws.front(),
      // raw_graph); this update is (group.current_raw, entry.raw), and from the third
      // member of a group onwards those pairs differ. Accepting the fold on the probe's
      // word therefore treats cudaGraphExecUpdate compatibility as TRANSITIVE across a
      // group's members -- if the driver can re-point A onto B and A onto C, then it can
      // re-point B onto C. Neither the CUDA nor the HIP documentation says so, and this
      // file does not assert it; it is an assumption, recorded as such under
      // `## Risks/decisions` in .agents/specs/eng-cudagraph-dedup.md and owed there.
      // The reason it is safe to hold it here anyway is the polarity of the failure: a
      // refusal is either an invariant violation or that transitivity not holding, and
      // both land on this VT_CHECK. What must never happen is the alternative --
      // launching the executable's PREVIOUS contents under this handle, which is a wrong
      // answer rather than a loud one.
      VT_CHECK(ok, std::string("graph dedup: replay update refused (") + detail + ")");
      group.current_raw = entry.raw;
    }
    ops_.launch(group.exec, stream);
  }

  void Destroy(void* handle) {
    auto found = handles_.find(handle);
    if (found == handles_.end()) return;
    Handle& entry = *found->second;
    Group& group = *entry.group;

    group.raws.erase(std::remove(group.raws.begin(), group.raws.end(), entry.raw),
                     group.raws.end());
    // The executable outlives the graph it was last pointed at, and a later capture can
    // land on the freed address. Forgetting the address is what forces the next replay
    // to re-point rather than assume.
    if (group.current_raw == entry.raw) group.current_raw = nullptr;
    ops_.destroy_graph(entry.raw);

    if (group.raws.empty()) {
      ops_.destroy_exec(group.exec);
      auto bucket = groups_.find(entry.signature);
      if (bucket != groups_.end()) {
        std::vector<std::unique_ptr<Group>>& candidates = bucket->second;
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&group](const std::unique_ptr<Group>& owned) {
                                          return owned.get() == &group;
                                        }),
                         candidates.end());
        if (candidates.empty()) groups_.erase(bucket);
      }
    }
    handles_.erase(found);
  }

  // Live captures, and the executables they share between them. The pre-dedup path
  // would report these equal.
  std::size_t CapturedCount() const { return handles_.size(); }
  std::size_t ExecCount() const {
    std::size_t count = 0;
    for (const auto& bucket : groups_) count += bucket.second.size();
    return count;
  }

  void Close() {
    for (auto& bucket : groups_) {
      for (std::unique_ptr<Group>& group : bucket.second) {
        for (void* raw : group->raws) ops_.destroy_graph(raw);
        group->raws.clear();
        ops_.destroy_exec(group->exec);
        group->exec = nullptr;
      }
    }
    groups_.clear();
    handles_.clear();
  }

 private:
  struct Group {
    void* exec = nullptr;
    // The raw graph the executable currently reflects, or null when that graph has been
    // destroyed. Never dereferenced here — only compared, and only against a live raw.
    void* current_raw = nullptr;
    std::vector<void*> raws;
  };
  struct Handle {
    Group* group = nullptr;
    void* raw = nullptr;
    std::string signature;
  };

  // Can `candidate`'s executable be re-pointed onto `raw_graph`? Answered on a
  // throwaway executable so a refusal cannot leave the live one in a state the driver
  // documentation does not define.
  bool ProbeAccepts(const Group& candidate, void* raw_graph) const {
    if (candidate.raws.empty()) return false;
    void* probe = ops_.instantiate(candidate.raws.front());
    // The probe instantiates a SECOND executable, so it can fail on its own -- most
    // plausibly under the memory pressure this row exists to relieve. cudaGraphExecUpdate
    // has no defined behaviour for a null executable, so answer "cannot fold" instead of
    // asking the driver about nothing. Unlike the instantiate above this is not fatal:
    // declining to fold degrades to exactly today's one-executable-per-capture path.
    if (probe == nullptr) return false;
    std::string detail;
    const bool ok = ops_.update(probe, raw_graph, &detail);
    ops_.destroy_exec(probe);
    return ok;
  }

  Handle& Lookup(void* handle) {
    auto found = handles_.find(handle);
    VT_CHECK(found != handles_.end(), "graph dedup: handle does not belong to this registry");
    return *found->second;
  }

  GraphDedupOps ops_;
  std::FILE* log_ = nullptr;
  std::unordered_map<std::string, std::vector<std::unique_ptr<Group>>> groups_;
  std::unordered_map<void*, std::unique_ptr<Handle>> handles_;
};

}  // namespace vt

#endif  // VT_GRAPH_DEDUP_H_
