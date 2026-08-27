// vllm.cpp original — the primitives the sampling ops must agree on BYTE FOR
// BYTE across backends, in one place.
//
// WHY THIS FILE EXISTS. `SplitMix64` and `ExpNoise` were written out three
// times: src/vt/cpu/cpu_sample.cpp, src/vt/cuda/cuda_sample.cu and
// src/vt/rocm/rocm_sample.hip. The CPU copy is the reference the device copies
// are gated against, so a divergence between any two of them is a token
// difference that no test in the tree looks for -- the copies are compared only
// through their results, and only on rows where the result happens to differ.
// The CPU and CUDA copies now come from here. The ROCm copy does not yet, and
// that is recorded under `## Owed` in
// .agents/specs/sample-gen-config-and-parallel-gumbel.md rather than changed on
// hardware nobody could run the gate on.
//
// Everything here is a `__host__ __device__` inline so the SAME expression is
// compiled for the device kernel and for the host test that checks it. A host
// test that re-types the expression proves the transcription, not the function.
#ifndef VT_SAMPLE_COMMON_H_
#define VT_SAMPLE_COMMON_H_

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define VT_SAMPLE_HD __host__ __device__
#else
#define VT_SAMPLE_HD
#endif

namespace vt::sample {

// Deterministic integer mixing. Bit-identical on host and device: it is 64-bit
// integer arithmetic only, with no libm and no floating point, so there is no
// rounding freedom for a platform to spend.
VT_SAMPLE_HD inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// One Exp(1) draw for cell (row, col) under `seed`, mirroring the tensor vLLM
// fills with `q.exponential_()` (vllm/v1/sample/ops/topk_topp_sampler.py).
// Upstream draws a whole tensor from a torch Generator; we hash the coordinate
// instead, so a row's draw does not depend on how many rows share the batch.
// Exact torch-Philox parity is the documented M1.7 T1 carry.
//
// The `-log(u)` is evaluated in DOUBLE. That is WIDER than upstream's default
// (`use_fp64_gumbel: bool = False`, sampler.py), and it is the reason host and
// device can disagree by an ULP: IEEE-754 does not require a correctly-rounded
// `log`, so glibc's and libdevice's may differ in the last bit. Narrowing it to
// f32 would mirror upstream and cost less on a part with 1:64 f64 throughput,
// and it would change which token is drawn -- so it is a separate row with its
// own gate, recorded under `## Owed` in
// .agents/specs/sample-gen-config-and-parallel-gumbel.md.
VT_SAMPLE_HD inline double ExpNoise(uint64_t seed, int64_t row, int64_t col) {
  const uint64_t row_key = SplitMix64(seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(row));
  const uint64_t r = SplitMix64(row_key + static_cast<uint64_t>(col));
  const double u = static_cast<double>((r >> 11) + 1ULL) * (1.0 / 9007199254740993.0);
  return -log(u);
}

// One element of `probs.div_(q)` (topk_topp_sampler.py::
// sample_with_exponential_noise). The argmax over a row of these IS the sample.
VT_SAMPLE_HD inline float GumbelScore(float prob, uint64_t seed, int64_t row, int64_t col) {
  return prob / static_cast<float>(ExpNoise(seed, row, col));
}

// The "no real index yet" marker, INT64_MAX, so that any real index beats an
// unfilled lane on the tie-break below.
constexpr int64_t kArgSentinel = 0x7fffffffffffffffLL;

// (value, index) argmax with the LOWEST index winning a tie -- torch.argmax's
// rule, and the rule the CPU reference's `score > best_v` serial scan produces.
//
// THIS OPERATOR IS ORDER-INDEPENDENT, and that property is the whole reason a
// row can be reduced in parallel at all. It compares the true GLOBAL index
// rather than thread or block order, so any partition of a row and any order of
// combination yield the same answer as the serial left-to-right scan. Drop the
// `bi < ai` clause and the reduction becomes order-DEPENDENT: reducing
// right-to-left then returns the HIGHEST tied index, which is the defect a
// careless parallelisation introduces and which
// tests/vt/test_ops_sample.cpp's order-independence case exists to catch.
//
// NaN propagates the way the serial scan does: `bv > av` and `bv == av` are both
// false for a NaN, so a NaN never displaces a real candidate.
VT_SAMPLE_HD inline void ArgReduce(float& av, int64_t& ai, float bv, int64_t bi) {
  if (bv > av || (bv == av && bi < ai)) {
    av = bv;
    ai = bi;
  }
}

// How many blocks cover one row of `v` elements at `block` threads each, capped
// so the second pass can reduce every partial of a row with a single block of
// `block` threads. Shared by the launcher and by the host test that mirrors the
// launch, so the test cannot check a partition the kernel does not use.
inline int ArgBlocksPerRow(int64_t v, int block) {
  int64_t bpr = (v + block - 1) / block;
  if (bpr > block) bpr = block;
  if (bpr < 1) bpr = 1;
  return static_cast<int>(bpr);
}

}  // namespace vt::sample

#endif  // VT_SAMPLE_COMMON_H_
