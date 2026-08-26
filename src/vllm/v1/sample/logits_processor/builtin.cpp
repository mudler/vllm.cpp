// Ported from: vllm/v1/sample/logits_processor/builtin.py @ e24d1b24.
#include "vllm/v1/sample/logits_processor/builtin.h"

#include <cstddef>
#include <vector>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm::v1 {

void apply_min_tokens(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, MinTokensState>& min_tokens,
                      const std::vector<std::vector<int32_t>>& output_token_ids) {
  VT_CHECK(logits.rank == 2, "apply_min_tokens: logits must be [num_reqs, vocab]");
  if (min_tokens.empty()) return;

  // Build the (request, stop-token) -inf slice. Upstream pre-filters min_toks so
  // only under-floor requests remain; we re-check output_len < min_tokens so the
  // function is correct on its own (the same set upstream masks).
  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  for (const auto& [i, state] : min_tokens) {
    VT_CHECK(i >= 0 && static_cast<size_t>(i) < output_token_ids.size(),
             "apply_min_tokens: request index out of range");
    const int output_len = static_cast<int>(output_token_ids[static_cast<size_t>(i)].size());
    if (output_len >= state.min_tokens) continue;
    for (int32_t tok : state.stop_token_ids) {
      rows.push_back(i);
      cols.push_back(tok);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  vt::ApplyTokenMask(q, logits, r.tensor(), c.tensor());
}

void apply_logit_bias(vt::Queue& q, vt::Tensor& logits,
                      const std::map<int, std::map<int32_t, float>>& logit_bias) {
  VT_CHECK(logits.rank == 2, "apply_logit_bias: logits must be [num_reqs, vocab]");
  if (logit_bias.empty()) return;

  std::vector<int32_t> rows;
  std::vector<int32_t> cols;
  std::vector<float> biases;
  for (const auto& [i, tok_bias] : logit_bias) {
    for (const auto& [tok, bias] : tok_bias) {
      rows.push_back(i);
      cols.push_back(tok);
      biases.push_back(bias);
    }
  }

  if (rows.empty()) return;
  const int64_t m = static_cast<int64_t>(rows.size());
  DeviceScratch r(logits.device, q, rows.data(), vt::DType::kI32, {m});
  DeviceScratch c(logits.device, q, cols.data(), vt::DType::kI32, {m});
  DeviceScratch b(logits.device, q, biases.data(), vt::DType::kF32, {m});
  vt::ApplyLogitBias(q, logits, r.tensor(), c.tensor(), b.tensor());
}

void apply_min_p(vt::Queue& q, vt::Tensor& logits, const std::vector<float>& min_p) {
  const int64_t n = logits.shape[0];
  VT_CHECK(logits.rank == 2, "apply_min_p: logits must be [num_reqs, vocab]");
  VT_CHECK(static_cast<int64_t>(min_p.size()) == n, "apply_min_p: min_p must have num_reqs rows");
  DeviceScratch mp(logits.device, q, min_p.data(), vt::DType::kF32, {n});
  vt::ApplyMinP(q, logits, mp.tensor());
}

void apply_logits_processors(
    vt::Queue& q, vt::Tensor& logits,
    const std::map<int, LogitsProcessorCallback>& procs,
    const std::vector<std::vector<int32_t>>& output_token_ids) {
  if (procs.empty()) return;
  VT_CHECK(logits.rank == 2, "apply_logits_processors: logits must be [num_reqs, vocab]");
  const int64_t n = logits.shape[0];
  const int64_t vocab = logits.shape[1];

  vt::Backend& b = vt::GetBackend(logits.device.type);
  // The callbacks read+mutate the logits on the HOST, so any prior async op that
  // produced/mutated the logits must complete first (a no-op on the synchronous
  // CPU backend; a real sync on CUDA).
  b.Synchronize(q);

  // Obtain a host-addressable view of the [n, vocab] logits. The callbacks are
  // ABI code that LOADS AND STORES through whatever pointer we hand them, so the
  // question is whether the HOST MAY DEREFERENCE what `Backend::Alloc` returned
  // -- `DeviceMemoryIsHostAddressable()` -- and NOT whether host and device
  // happen to sit on the same physical RAM, which is the wider `UnifiedMemory()`.
  //
  // THIS USED TO ASK `UnifiedMemory()`, AND THAT IS #1746. It is the same
  // mistake `src/vt/op_provider.cpp` records beside `ReferenceTierEligible`,
  // where it cost two crashes (#844, #1435) and a third report (#960). CUDA
  // reports unified memory on GB10 because host and device address the same
  // physical RAM, yet `CudaBackend::Alloc` returns a plain `cudaMalloc` pointer
  // and CUDA never overrides the narrow predicate, so it keeps the `false`
  // default in include/vt/backend.h. Asking the wide question here therefore
  // handed the ABI callback a device pointer to store through, on the one box
  // the old comment named as safe.
  //
  // A backend that answers the narrow predicate `true` -- Vulkan, Metal
  // StorageModeShared, integrated ROCm -- keeps the zero-copy in-place wrap:
  // `logits.data` IS host memory there. Every other backend, CPU included,
  // stages down, runs the callbacks, and copies the edited logits back. CPU
  // pays that bounce because `CpuBackend` has never opted in to the narrow
  // predicate; that is a correct-but-conservative cost, and it is charged only
  // when a request actually registers a processor (the empty map returns above).
  const bool host_addressable = b.DeviceMemoryIsHostAddressable();
  const size_t total = static_cast<size_t>(n) * static_cast<size_t>(vocab);
  std::vector<float> staging;
  float* host = nullptr;
  if (host_addressable) {
    host = static_cast<float*>(logits.data);
  } else {
    staging.resize(total);
    if (total != 0) b.Copy(q, staging.data(), logits.data, total * sizeof(float));
    b.Synchronize(q);
    host = staging.data();
  }

  static const std::vector<int32_t> kNoTokens;
  for (const auto& [i, cb] : procs) {
    if (cb.fn == nullptr) continue;
    VT_CHECK(i >= 0 && static_cast<int64_t>(i) < n,
             "apply_logits_processors: request index out of range");
    const std::vector<int32_t>& toks =
        (static_cast<size_t>(i) < output_token_ids.size())
            ? output_token_ids[static_cast<size_t>(i)]
            : kNoTokens;
    float* row = host + static_cast<size_t>(i) * static_cast<size_t>(vocab);
    cb.fn(toks.data(), static_cast<int32_t>(toks.size()), row,
          static_cast<int32_t>(vocab), cb.user_data);
  }

  if (!host_addressable) {
    if (total != 0) b.Copy(q, logits.data, staging.data(), total * sizeof(float));
    b.Synchronize(q);
  }
}

}  // namespace vllm::v1
