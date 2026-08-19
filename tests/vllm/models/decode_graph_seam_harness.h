// THE SHARED HARNESS FOR THE ENG-CUDAGRAPH-BREAK G2 REACHABILITY GATES
// (row `ENG-CUDAGRAPH-BREAK`, spec `.agents/specs/eng-cudagraph-break.md`;
// W2 #1261, W3 #1291, parent #1163).
//
// WHY IT IS A HEADER AND NOT A FOURTH COPY. This row exists because nine decode
// drivers each re-derived one capture machine, and W2 landed the first gate that
// holds one of them to the shared seam. W3 migrates three more drivers of the
// same shape, and each needs the SAME two swapped seams to run at all. Copying
// them per gate would reproduce inside `tests/` exactly the duplication the row
// is removing from `src/`, and a divergence between two copies of a harness is
// invisible: both files stay green while they measure different things.
//
// WHAT THE TWO SEAMS ARE FOR. This box and continuous integration have no NVIDIA
// GPU, and every migrated driver admits itself only where
// `Platform::support_static_graph_mode()` and `Backend::SupportsGraphCapture()`
// are BOTH true — CUDA and Tenstorrent, never CPU. So for the duration of a case
// the CPU backend is swapped for one that delegates every memory and compute
// call to the real CPU backend and implements the capture vocabulary
// (`include/vt/backend.h:208-222`) by logging, and the CPU platform for one that
// delegates everything and answers `support_static_graph_mode()` true. It has to
// DELEGATE rather than simulate, because the model's arithmetic must be the real
// CPU arithmetic for a bit-exactness assertion over it to mean anything. Both
// registries are restored in a destructor: a doctest binary runs every case in
// one process, so a leaked override would make an unrelated case observe a CPU
// device that captures.
//
// WHAT THIS HARNESS CANNOT SEE, named rather than claimed away. CPU kernels are
// direct function calls, not backend submissions, so a "captured" region here
// EXECUTES eagerly and a "replay" recomputes nothing. A gate built on it holds
// the ROUTING and the capture step's numerics; it does NOT hold that a replayed
// segment reproduces the eager forward. That is G1, it needs a real device, and
// the spec's `## Owed` records it as owed rather than implying it ran here.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm_test {

// Delegates every real operation to the CPU backend and implements the capture
// vocabulary by logging.
class CaptureCapableCpuBackend final : public vt::Backend {
 public:
  // `supports_capture` is the CONTROL LANE switch. Passing false makes this the
  // same delegating backend that answers `SupportsGraphCapture()` FALSE, so a
  // driver's own predicate sends it down its eager path — which is the lane a
  // gate has to compare against to tell "the switch chose eager" apart from
  // "the switch made the capture inert and the forward ran twice" (#1352).
  explicit CaptureCapableCpuBackend(vt::Backend& inner, bool supports_capture = true)
      : inner_(inner), supports_capture_(supports_capture) {}

  // ALLOCATIONS ARE COUNTED, and the reason is LANE IDENTITY rather than memory.
  // ENG-CUDAGRAPH-BREAK W5 (#1352) needed to tell apart two versions of a driver
  // that produce IDENTICAL logits and BOTH report `segments_captured == 0` — the
  // one that takes its eager path under `VLLM_CPP_CUDAGRAPH=0`, and the one that
  // takes the CAPTURE path and runs the forward twice inside an inert scope.
  // Neither a token gate nor the seam's counters can see that difference. What
  // can is the comparison against a backend that CANNOT capture: with the switch
  // off the two must allocate exactly the same, because they must be running the
  // same code. The count is an EQUALITY against a control, never a magic number,
  // so it does not go stale when the driver's allocation pattern changes.
  void* Alloc(size_t bytes) override {
    ++allocs_;
    return inner_.Alloc(bytes);
  }
  void Free(void* p) override { inner_.Free(p); }
  void Memset(vt::Queue& q, void* p, int v, size_t bytes) override {
    inner_.Memset(q, p, v, bytes);
  }
  void Copy(vt::Queue& q, void* dst, const void* src, size_t bytes) override {
    ++copies_;
    inner_.Copy(q, dst, src, bytes);
  }
  vt::Queue CreateQueue() override { return inner_.CreateQueue(); }
  void DestroyQueue(vt::Queue& q) override { inner_.DestroyQueue(q); }
  void Synchronize(vt::Queue& q) override { inner_.Synchronize(q); }
  void FlushPending() override { inner_.FlushPending(); }
  bool UnifiedMemory() const override { return inner_.UnifiedMemory(); }
  bool DeviceMemoryIsHostAddressable() const override {
    return inner_.DeviceMemoryIsHostAddressable();
  }
  bool SupportsAsyncSampledTokenReadback() const override {
    return inner_.SupportsAsyncSampledTokenReadback();
  }
  bool SupportsCompressedConvState() const override {
    return inner_.SupportsCompressedConvState();
  }
  bool SupportsCompressedGdnState() const override {
    return inner_.SupportsCompressedGdnState();
  }
  bool SupportsAuxStream() const override { return inner_.SupportsAuxStream(); }

  bool SupportsGraphCapture() const override { return supports_capture_; }
  void BeginCapture(vt::Queue&) override { log_.push_back("Begin"); }
  void* EndCaptureGraph(vt::Queue&) override {
    // Arm-once refusal, the shape of a real `cudaStreamEndCapture` returning
    // `cudaErrorStreamCaptureInvalidated` / `WrongThread`, or a failing
    // `cudaGraphInstantiate` — all three of which `Check()`
    // (`src/vt/cuda/cuda_backend.cu:50,229`) turns into a throw.
    if (fail_next_end_) {
      fail_next_end_ = false;
      throw std::runtime_error("capture-capable CPU backend: EndCaptureGraph refused");
    }
    log_.push_back("EndCaptureGraph");
    void* tag = std::malloc(1);
    tags_.push_back(tag);
    return tag;
  }
  void ReplayGraph(vt::Queue&, void*) override { log_.push_back("ReplayGraph"); }
  void DestroyGraph(void* g) override {
    if (g == nullptr) return;
    log_.push_back("DestroyGraph");
  }

  ~CaptureCapableCpuBackend() override {
    for (void* p : tags_) std::free(p);
  }

  void FailNextEndCapture() { fail_next_end_ = true; }

  // COPIES, not allocations, and the difference is the DevicePool. A pooled
  // allocator serves the second of two identical forwards out of its free list
  // without calling `Alloc` at all, so an allocation count reads 11 for the
  // first lane measured and 0 for the second REGARDLESS of which lanes they
  // were — an instrument that reports the ORDER of the runs rather than their
  // shape. Copies are per-operation work the pool cannot absorb, so they count
  // the forward rather than the allocator's history (#1352).
  int64_t copies() const { return copies_; }
  int64_t allocs() const { return allocs_; }
  void ResetCounters() {
    copies_ = 0;
    allocs_ = 0;
  }

  size_t Count(std::string_view what) const {
    size_t n = 0;
    for (const auto& e : log_)
      if (e == what) ++n;
    return n;
  }

 private:
  vt::Backend& inner_;
  bool supports_capture_ = true;
  bool fail_next_end_ = false;
  int64_t allocs_ = 0;
  int64_t copies_ = 0;
  std::vector<std::string> log_;
  std::vector<void*> tags_;
};

// Answers `support_static_graph_mode()` true and delegates the rest, so a
// driver's OWN admission predicate is the thing under test rather than a
// predicate a gate reimplements.
class StaticGraphCpuPlatform final : public vllm::platforms::Platform {
 public:
  StaticGraphCpuPlatform(vllm::platforms::Platform& inner, vt::Backend& backend)
      : inner_(inner), backend_(backend) {}

  vt::DeviceType device_type() const override { return inner_.device_type(); }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return inner_.get_device_capability();
  }
  bool supports_fp8() const override { return inner_.supports_fp8(); }
  bool cutlass_fp4_supported() const override { return inner_.cutlass_fp4_supported(); }
  bool opaque_attention_op() const override { return inner_.opaque_attention_op(); }
  bool is_integrated_gpu() const override { return inner_.is_integrated_gpu(); }
  bool needs_weight_staging() const override { return inner_.needs_weight_staging(); }
  bool supports_fa2_attention() const override { return inner_.supports_fa2_attention(); }
  std::vector<vt::DType> supported_dtypes() const override {
    return inner_.supported_dtypes();
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    return inner_.residency_policy();
  }
  bool supports_model_architecture(std::string_view a) const override {
    return inner_.supports_model_architecture(a);
  }
  std::vector<std::string> get_attn_backend_priority(
      const vllm::platforms::AttnSelectorConfig& cfg) const override {
    return inner_.get_attn_backend_priority(cfg);
  }
  std::vector<std::string> get_mla_prefill_backend_priority() const override {
    return inner_.get_mla_prefill_backend_priority();
  }

  // THE ONE ANSWER THAT DIFFERS.
  bool support_static_graph_mode() const override { return true; }

 private:
  vllm::platforms::Platform& inner_;
  vt::Backend& backend_;
};

// Swaps both registries for the life of the object and puts back exactly what
// was there.
class StaticGraphCpu {
 public:
  explicit StaticGraphCpu(bool supports_capture = true)
      : prev_backend_(&vt::GetBackend(vt::DeviceType::kCPU)),
        prev_platform_(&vllm::platforms::GetPlatform(vt::DeviceType::kCPU)),
        backend_(*prev_backend_, supports_capture),
        platform_(*prev_platform_, backend_) {
    vt::RegisterBackend(vt::DeviceType::kCPU, &backend_);
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, &platform_);
  }
  ~StaticGraphCpu() {
    vt::RegisterBackend(vt::DeviceType::kCPU, prev_backend_);
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, prev_platform_);
  }
  CaptureCapableCpuBackend& backend() { return backend_; }

 private:
  vt::Backend* prev_backend_;
  vllm::platforms::Platform* prev_platform_;
  CaptureCapableCpuBackend backend_;
  StaticGraphCpuPlatform platform_;
};

}  // namespace vllm_test
