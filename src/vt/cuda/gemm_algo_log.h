// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Env-gated cuBLASLt GEMM algo-selection diagnostic — the flag plumbing and the
// "log once per unique key" bookkeeping, split into this pure-C++ (CUDA-free)
// header so the portable logic is unit-testable on the CPU tier
// (tests/vt/test_gemm_algo_log.cpp). The cuBLASLt emit that consumes these lives
// in cuda_matmul.cu (it needs <cublasLt.h> to read the selected algo config).
//
// This is OUR diagnostic. Upstream logs cuBLASLt/algo selection under its own
// debug switches (CUBLASLT_LOG_LEVEL, torch `_scaled_mm` verbose paths); we have
// no torch, so we emit an equivalent under our own flag to compare arm-wise
// cuBLASLt algo LATCHING on the packed GDN BF16-BA GEMM vs the F32-BA/decomposed
// arm, per the 2026-07-15 forensic record (a constant ~0.2% packed steady
// per-token tax whose one un-instrumented per-process variable is the BF16 GEMM
// algo selection; see .agents/specs/gdn-packed-decode.md and .agents/state.md).
// Default OFF; when the flag is unset the gate is a cached bool and the whole
// emit block is skipped, so the hot GEMM path pays nothing.
#ifndef VT_CUDA_GEMM_ALGO_LOG_H_
#define VT_CUDA_GEMM_ALGO_LOG_H_

#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

namespace vt::cuda {

// Pure predicate for the VT_GEMM_ALGO_LOG contract: enabled iff the value is
// exactly "1". nullptr (unset) and every other value are OFF. Kept separate from
// the cached getter below so the parse is unit-testable without touching the
// process-global cache.
inline bool GemmAlgoLogFlagIsOn(const char* env_value) {
  return env_value != nullptr && std::string_view(env_value) == "1";
}

// Process-cached gate, read from the environment exactly once (getenv runs on
// the first call only; every later hot-path call pays a single bool load — no
// per-GEMM getenv). Kept out of the CPU unit test because the cache latches on
// first read; the parse itself is covered via GemmAlgoLogFlagIsOn.
inline bool GemmAlgoLogEnabled() {
  static const bool enabled = GemmAlgoLogFlagIsOn(std::getenv("VT_GEMM_ALGO_LOG"));
  return enabled;
}

// Thread-safe "have I already logged this key?" set. ShouldLog(key) returns true
// exactly once per distinct key (its first sighting) and false on every repeat,
// so a hot call site emits exactly one line per unique (shape, dtype-combo,
// epilogue) selection even under concurrent queues. The guarded set is only ever
// touched when the diagnostic flag is on (the caller checks GemmAlgoLogEnabled
// first), so it adds nothing to the default hot path.
class LogOncePerKey {
 public:
  bool ShouldLog(const std::string& key) {
    std::lock_guard<std::mutex> lock(mu_);
    return seen_.insert(key).second;  // .second is true only on first insertion
  }
  // Test-only introspection: number of distinct keys logged so far.
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return seen_.size();
  }

 private:
  mutable std::mutex mu_;
  std::unordered_set<std::string> seen_;
};

}  // namespace vt::cuda

#endif  // VT_CUDA_GEMM_ALGO_LOG_H_
