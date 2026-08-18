// vllm.cpp original test helper (ENG-CUDAGRAPH-BREAK W1, #1192); no upstream
// mirror, because upstream has none to mirror.
//
// THE HARNESS ADAPTATION, stated once and stated exactly. Every class in
// SGLang's BCG suite
// (`test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py` @
// `f63458b5be`) raises `unittest.SkipTest` without CUDA (`:34-36`, `:177-178`,
// `:235-236`), so upstream runs its eleven unit cases on a real device. This box
// and continuous integration have no NVIDIA GPU, so the ported cases run against
// a backend that RECORDS the capture call sequence and SIMULATES a graph.
//
// What "simulates a graph" means here, because a recording backend that only
// logged would have cost the upstream cases their whole point. `Record(fn)`
// models one captured operation: during capture the work is filed against the
// currently open segment and NOT executed, exactly as a real stream capture
// files a kernel without running it, and `ReplayGraph` runs the filed work in
// order. That is what makes upstream's post-replay arithmetic chains portable —
// `x.fill_(5); graph.replay(); assert y == 6` (`:49-63`) is the same assertion
// here — instead of degrading them into capture-time value checks, which assert
// something upstream never asserted.
//
// What is NOT modelled: real device memory, asynchrony, and any error a real
// runtime would raise. A case that needs those is a GPU case and is recorded as
// owed in the spec rather than approximated here.
//
// A recording backend is also strictly STRONGER than a real one for the two
// questions these cases ask. "How many segments did this capture produce?" and
// "in what ORDER did replay emit segments and break functions?" are answered by
// the log directly; on a real device both are inferred from a composed value,
// which a wrong order can still satisfy for a commutative chain (spec
// `## Tests to port` test 12). `Note()` is what puts break functions into the
// SAME log as segments, so one assertion covers the interleaving rather than two
// independent ones that a batched replay satisfies separately.
#pragma once

#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"

namespace vt_test {

// Host-memory backend that implements the capture vocabulary
// (`include/vt/backend.h:208-222`) by appending to a call log. A "graph handle"
// is a small heap tag, which is what makes DestroyGraph observable: the seam
// must route every acquisition and release through EndCaptureGraph and
// DestroyGraph so ENG-CUDAGRAPH-DEDUP (#1162) can later interpose at the
// backend (spec `## Risks/decisions` D4).
class RecordingCaptureBackend final : public vt::Backend {
 public:
  explicit RecordingCaptureBackend(bool supports_capture = true)
      : supports_capture_(supports_capture) {}
  ~RecordingCaptureBackend() override {
    for (void* p : owned_) std::free(p);
    for (void* p : tags_) std::free(p);
  }

  // ---- vt::Backend ----
  void* Alloc(size_t bytes) override {
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    owned_.push_back(p);
    return p;
  }
  // Deliberately does not std::free: cases compare pointer identity across a
  // free, and a freed pointer is not a value you may reason about.
  void Free(void*) override {}
  void Memset(vt::Queue&, void* p, int v, size_t bytes) override {
    std::memset(p, v, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
    ++copies_;
  }
  vt::Queue CreateQueue() override { return vt::Queue{}; }
  bool UnifiedMemory() const override { return true; }

  bool SupportsGraphCapture() const override { return supports_capture_; }
  void BeginCapture(vt::Queue&) override {
    log_.push_back("Begin");
    open_work_.clear();
    capturing_ = true;
  }
  void* EndCaptureGraph(vt::Queue&) override {
    if (fail_next_end_) {
      fail_next_end_ = false;
      capturing_ = false;
      throw std::runtime_error("recording backend: EndCaptureGraph refused");
    }
    log_.push_back("EndCaptureGraph");
    void* tag = std::malloc(1);
    tags_.push_back(tag);
    live_graphs_.push_back(tag);
    graph_work_[tag] = std::move(open_work_);
    open_work_.clear();
    capturing_ = false;
    return tag;
  }
  void ReplayGraph(vt::Queue&, void* graph) override {
    log_.push_back("ReplayGraph");
    replayed_.push_back(graph);
    auto it = graph_work_.find(graph);
    if (it != graph_work_.end())
      for (const auto& fn : it->second) fn();
  }
  void DestroyGraph(void* graph) override {
    log_.push_back("DestroyGraph");
    destroyed_.push_back(graph);
    graph_work_.erase(graph);
    for (size_t i = 0; i < live_graphs_.size(); ++i) {
      if (live_graphs_[i] == graph) {
        live_graphs_.erase(live_graphs_.begin() + static_cast<long>(i));
        break;
      }
    }
  }

  // ---- the instrument ----

  // File one captured operation against the open segment. NOT executed now: a
  // real capture records the kernel and runs nothing, which is why upstream's
  // value assertions all sit AFTER `graph.replay()`.
  void Record(std::function<void()> fn) {
    if (capturing_)
      open_work_.push_back(std::move(fn));
    else
      fn();  // outside a capture there is nothing to record: run it, as eager work does
  }

  // Append a marker into the CALL LOG itself, so a break function's position is
  // comparable with the segments around it in one sequence. Two independent
  // sequences — segments in the backend log, breaks in a private vector — are
  // exactly what a batched replay satisfies while interleaving nothing.
  void Note(const std::string& what) { log_.push_back(what); }

  // Arm a single EndCaptureGraph refusal, for the drain case: the one failure a
  // destructor cannot let propagate.
  void FailNextEndCapture() { fail_next_end_ = true; }

  const std::vector<std::string>& log() const { return log_; }
  const std::vector<void*>& replayed() const { return replayed_; }
  const std::vector<void*>& destroyed() const { return destroyed_; }
  size_t live_graphs() const { return live_graphs_.size(); }
  int copies() const { return copies_; }
  void ClearLog() {
    log_.clear();
    replayed_.clear();
  }
  // The whole call sequence as one string, so a case can assert the ORDER
  // rather than only the counts.
  std::string Trace() const {
    std::string s;
    for (const std::string& e : log_) {
      if (!s.empty()) s += " ";
      s += e;
    }
    return s;
  }
  int Count(const char* what) const {
    int n = 0;
    for (const std::string& e : log_)
      if (e == what) ++n;
    return n;
  }

 private:
  bool supports_capture_;
  bool capturing_ = false;
  bool fail_next_end_ = false;
  std::vector<std::string> log_;
  std::vector<void*> owned_;
  std::vector<void*> tags_;
  std::vector<void*> live_graphs_;
  std::vector<void*> replayed_;
  std::vector<void*> destroyed_;
  std::vector<std::function<void()>> open_work_;
  std::map<void*, std::vector<std::function<void()>>> graph_work_;
  int copies_ = 0;
};

}  // namespace vt_test
