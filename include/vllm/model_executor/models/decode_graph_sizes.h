// Ported from: vllm/config/vllm.py::_set_cudagraph_sizes @ e24d1b24
//   (+ vllm/config/compilation.py:683-684,1438-1444 — the full-decode-cudagraph
//    dispatcher caps decode batches at max_num_seqs).
//
// The decode CUDA-graph capture-size set + pad-to-capture selector, extracted
// from qwen3_5.cpp so both weight-family graph drivers share one definition and
// it is unit-testable.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_DECODE_GRAPH_SIZES_H_
#define VLLM_MODEL_EXECUTOR_MODELS_DECODE_GRAPH_SIZES_H_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vllm {

// The decode CUDA-graph capture-size set for a given max_num_seqs, mirroring
// vLLM's _set_cudagraph_sizes reduced to the FULL-decode-cudagraph regime.
//
// vLLM (no speculative decode): max_cudagraph_capture_size =
//   min(max_num_seqs*2, 512, max_num_batched_tokens); the candidate sizes are
//   [1,2,4] + range(8, min(max_cap+1, 256), 8) + range(256, max_cap+1, 16). The
//   full-decode-cudagraph dispatcher then CAPS decode batches at max_num_seqs
//   (compilation.py:1438-1444), and for hybrid/Mamba models each decode seq
//   needs one state block, so only sizes <= max_num_seqs are ever selected or
//   captured. Filtering the candidate set to <= max_num_seqs (which is < 256 for
//   every gate config, so the range(256,...) tail never applies) yields exactly
//   [1,2,4] + range(8, max_num_seqs+1, 8), plus max_num_seqs itself when it is
//   off the step-8 grid (vLLM likewise appends the max size so the largest batch
//   always has a graph). For max_num_seqs=32 -> {1,2,4,8,16,24,32}; for 64 ->
//   {1,2,4,8,16,24,32,40,48,56,64}.
inline std::vector<int64_t> DecodeGraphSizes(int64_t max_num_seqs) {
  std::vector<int64_t> sizes;
  if (max_num_seqs < 1) return sizes;
  for (int64_t s : {int64_t{1}, int64_t{2}, int64_t{4}}) {
    if (s <= max_num_seqs) sizes.push_back(s);
  }
  for (int64_t s = 8; s <= max_num_seqs; s += 8) sizes.push_back(s);
  if (sizes.empty() || sizes.back() != max_num_seqs) sizes.push_back(max_num_seqs);
  return sizes;
}

// Smallest captured size >= b in DecodeGraphSizes(max_num_seqs), or -1 when b
// exceeds max_num_seqs (caller runs eager). Equivalent to selecting from the set
// without allocating: round up to the step-8 grid, clamped to max_num_seqs
// (which is itself always a captured size).
inline int64_t PadToCaptureSize(int64_t b, int64_t max_num_seqs) {
  if (b > max_num_seqs) return -1;  // never pad DOWN below the real batch
  if (b <= 1) return 1;
  if (b <= 2) return 2;
  if (b <= 4) return 4;
  const int64_t rounded = ((b + 7) / 8) * 8;  // next multiple of 8
  return std::min(rounded, max_num_seqs);
}

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_DECODE_GRAPH_SIZES_H_
