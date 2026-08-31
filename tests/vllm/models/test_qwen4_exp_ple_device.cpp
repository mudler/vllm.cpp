// Qwen4-Exp (Qwen3.8-Flash-Next) W5b-3 DEVICE-ARM GATE — `vt::Qwen4ExpPleConv`,
// the PLE dilated depthwise causal convolution and its persistent state as a
// `vt::` op over `vt::Tensor`.
// Issue #2156, campaign issue #1978, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. The op is compared
// DIRECTLY against the lane-pinned oracle, not against this repository's host
// reference: `qwen4_exp_ple_goldens.inc` is dumped by
// `scripts/gen-qwen4-exp-ple-goldens.py`, which lifts `Qwen4ExpTextPLELayer`
// (:1117-1189) and `LinearAttentionLayer.update_conv_state`
// (cache_utils.py:1036-1075) VERBATIM by line range out of transformers
// **v5.16.0** and EXECUTES them under torch. The same file already gates the W2
// host reference (`test_qwen4_exp_ple.cpp`), so the two arms are held to ONE
// oracle rather than to each other.
//
// THE DILATION IS THE VARIABLE, AND THE ORACLE SUPPLIES BOTH SIDES OF IT. A
// fixture in which dilation 3 and dilation 1 give the same answer would gate
// nothing at all, so the generator runs upstream's own `_short_conv` at
// dilations 1, 2 and 3 over the SAME input and the SAME weight — swapping only
// the `nn.Conv1d` and the `short_conv_state_len` upstream derives from it — and
// this file requires the op to reproduce each one. The measured pairwise
// separation between those three answers is 0.44 to 0.72 against a 1e-5
// tolerance, and `kConvDilationsSeparate` re-measures it here so a future
// regeneration that collapsed the three could not pass in silence.
//
// WHAT THE EXISTING GOLDENS COULD NOT DO. `kTapExpected` (section F) is an
// impulse through a ONE-HOT-PER-CHANNEL weight: it pins the lag SET {9,6,3,0}
// beautifully and it cannot see an accumulation defect, because no output ever
// sums more than one term. `kPleExpectedOutput` (section G) is the whole layer,
// which sees everything and localises nothing. The dense conv-only goldens this
// wave adds sit between them.
//
// SCOPE, HONESTLY. This file is the CPU arm's gate. A CUDA arm of this op DOES
// now exist (`src/vt/cuda/cuda_qwen4_exp_ple.cu`, W6-CUDA) and is gated in
// `test_qwen4_exp_cuda.cpp`, against these same goldens and against this arm;
// nothing below runs on a device. The text that stood here said no CUDA arm
// exists, and it read: CPU only — no CUDA arm of this op exists, and one written on
// this CPU-only host could not be gated on it. Nothing calls this op from a
// production entry point yet: `ModelRegistry::Forward` has no `qwen4_exp` arm,
// the wiring is owned by row `MODEL-MM-QWEN4-EXP` and tracked by #2031 under
// campaign #1978, and the spec's `## Owed` records it. No token claim and no
// speed claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vllm/model_executor/models/qwen4_exp_ple.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm_test::MaxAbsDiff;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Qwen4ExpPleConvArgs;
using vt::Queue;
using vt::Tensor;

namespace {

#include "qwen4_exp_ple_goldens.inc"  // NOLINT — golden literals

// The goldens are fp32 out of torch; the op's interior is fp32 with a double
// four-term tap accumulation. At C = 16 and K = 4 the two are bit-identical or
// within an ulp of the silu, so this bound is loose by orders of magnitude for
// everything except a real defect — and the defects it has to separate sit at
// 4.4e-1 (see `kConvDilationsSeparate`).
constexpr double kTol = 1e-5;

// The tiny config's stream width, `hc_count * hidden_size` = 2 * 8 = 16, which
// is the CHANNEL count of a depthwise conv over the hyper-connection stream.
constexpr int64_t kChannels = 16;
constexpr int64_t kKernel = 4;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue CpuQ() { return Queue{Cpu(), nullptr}; }

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = Cpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= t.shape[i];
  }
  return t;
}

int64_t StateLen(int64_t dilation) { return (kKernel - 1) * dilation; }

const float* ExpectedFor(int64_t dilation) {
  switch (dilation) {
    case 1: return &kConvExpectedD1[0];
    case 2: return &kConvExpectedD2[0];
    case 3: return &kConvExpectedD3[0];
    default: return nullptr;
  }
}

// One single-shot run of `tokens` rows starting at `kConvInput[from]`, through a
// freshly zeroed one-row cache. Returns the [tokens, kChannels] output.
std::vector<float> RunSingleShot(int64_t dilation, int64_t from, int64_t tokens) {
  Queue q = CpuQ();
  const int64_t state_len = StateLen(dilation);
  std::vector<float> x(kConvInput + from * kChannels,
                       kConvInput + (from + tokens) * kChannels);
  std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  std::vector<float> out(static_cast<size_t>(tokens * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(kChannels * state_len), 0.0f);
  std::vector<int32_t> qsl{0, static_cast<int32_t>(tokens)};

  Tensor t_x = MakeT(x.data(), DType::kF32, {tokens, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {tokens, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {1, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});

  Qwen4ExpPleConvArgs args;
  args.dilation = dilation;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);
  return out;
}

}  // namespace

TEST_CASE("vt::Qwen4ExpPleConv reproduces the pinned oracle at the model's dilation") {
  // dilation == `config.ngram_size` == 3, so the taps are lags {9, 6, 3, 0} and
  // the state is nine columns deep.
  const std::vector<float> got = RunSingleShot(3, 0, kConvSeqLen);
  const std::vector<float> want(kConvExpectedD3,
                                kConvExpectedD3 + kConvSeqLen * kChannels);
  INFO("max|diff| ", MaxAbsDiff(got, want));
  CHECK(MaxAbsDiff(got, want) < kTol);
}

TEST_CASE("vt::Qwen4ExpPleConv: the DILATION is honoured, and it is load-bearing") {
  // THE POINT OF THIS WHOLE OP. `CausalConv1dFwd` reads a contiguous `K - 1`
  // history; this reads `(K - 1) * dilation` columns at stride `dilation`. A
  // kernel that ignored `args.dilation` — or that read the state contiguously —
  // would still reproduce the dilation-1 golden and nothing else.
  for (int64_t i = 0; i < 3; ++i) {
    const int64_t dilation = kConvDilations[i];
    const std::vector<float> got = RunSingleShot(dilation, 0, kConvSeqLen);
    const std::vector<float> want(ExpectedFor(dilation),
                                  ExpectedFor(dilation) + kConvSeqLen * kChannels);
    INFO("dilation ", dilation, " max|diff| ", MaxAbsDiff(got, want));
    CHECK(MaxAbsDiff(got, want) < kTol);
  }

  SUBCASE("kConvDilationsSeparate: the three oracle answers actually differ") {
    // Re-measured HERE, not trusted from the generator's assert. If a future
    // regeneration drew an input on which the three dilations happened to agree,
    // every check above would still pass while gating nothing, and this is the
    // only line that would notice.
    double worst_pair = 1e30;
    for (int64_t a = 0; a < 3; ++a) {
      for (int64_t b = a + 1; b < 3; ++b) {
        const std::vector<float> lhs(ExpectedFor(kConvDilations[a]),
                                     ExpectedFor(kConvDilations[a]) +
                                         kConvSeqLen * kChannels);
        const std::vector<float> rhs(ExpectedFor(kConvDilations[b]),
                                     ExpectedFor(kConvDilations[b]) +
                                         kConvSeqLen * kChannels);
        worst_pair = std::min(worst_pair, MaxAbsDiff(lhs, rhs));
      }
    }
    // The measured value is 0.443272 (d2 vs d3, the closest pair). The bound is
    // 1e-2: four orders above `kTol`, two below the measurement.
    INFO("closest pair of oracle answers: ", worst_pair);
    CHECK(worst_pair > 1e-2);
  }
}

TEST_CASE("vt::Qwen4ExpPleConv: chunked prefill and decode equal the single shot") {
  // The state is the whole reason this op is not one of the three stateless
  // dilatable convs already in `vt::`. `kConvChunks` is prefill(7) + decode(1) +
  // prefill(4): at dilation 3 the second chunk's first output needs lag 9, i.e.
  // two positions BEFORE anything it has seen, so a state that kept fewer than
  // nine columns cannot produce it.
  //
  // BIT-IDENTICAL, not within a tolerance. Each output position sums the same
  // four products in the same order either way, so any difference at all is the
  // state path disagreeing with the contiguous one.
  for (int64_t i = 0; i < 3; ++i) {
    const int64_t dilation = kConvDilations[i];
    const int64_t state_len = StateLen(dilation);
    Queue q = CpuQ();
    std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
    std::vector<float> state(static_cast<size_t>(kChannels * state_len), 0.0f);
    Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
    Tensor t_state = MakeT(state.data(), DType::kF32, {1, kChannels, state_len});
    Qwen4ExpPleConvArgs args;
    args.dilation = dilation;

    std::vector<float> chunked;
    int64_t lo = 0;
    for (int64_t c = 0; c < 3; ++c) {
      const int64_t n = kConvChunks[c];
      std::vector<float> x(kConvInput + lo * kChannels,
                           kConvInput + (lo + n) * kChannels);
      std::vector<float> out(static_cast<size_t>(n * kChannels), 0.0f);
      std::vector<int32_t> qsl{0, static_cast<int32_t>(n)};
      Tensor t_x = MakeT(x.data(), DType::kF32, {n, kChannels});
      Tensor t_out = MakeT(out.data(), DType::kF32, {n, kChannels});
      Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
      vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);
      chunked.insert(chunked.end(), out.begin(), out.end());
      lo += n;
    }
    REQUIRE(static_cast<int64_t>(chunked.size()) == kConvSeqLen * kChannels);

    const std::vector<float> single = RunSingleShot(dilation, 0, kConvSeqLen);
    INFO("dilation ", dilation);
    CHECK(std::memcmp(chunked.data(), single.data(),
                      sizeof(float) * chunked.size()) == 0);
    // And the chunked arm still has to match the ORACLE, not merely itself.
    const std::vector<float> want(ExpectedFor(dilation),
                                  ExpectedFor(dilation) + kConvSeqLen * kChannels);
    CHECK(MaxAbsDiff(chunked, want) < kTol);
  }
}

TEST_CASE("vt::Qwen4ExpPleConv: a bf16 RING carries state across calls, and rounds ONCE") {
  // W5k (#2031). The oracle stores this cache slot at the MODEL dtype — each slot
  // is allocated with the dtype of the tensor that first reaches it
  // (`cache_utils.py:1019-1023`), and the tensor reaching it is `hidden_states`
  // (`modeling_qwen4_exp.py:1157-1159`). Run at `dtype=torch.bfloat16` the pinned
  // oracle reports `conv_states[1] dtype=torch.bfloat16`. Before this wave the op
  // refused that ring outright, which is why the model's recurrent group (bf16)
  // and this op (f32) could not be connected and `past_len > 0` was unreachable.
  //
  // THREE THINGS ARE ASSERTED, and the third is what makes this a gate rather
  // than a smoke test:
  //   1. the ring's stored bytes are EXACTLY the bf16 rounding of the raw inputs
  //      — bit-exact, so an off-by-one column or a dropped store is visible;
  //   2. a chunked run over a bf16 ring reproduces the oracle within bf16's own
  //      resolution, so the carried state is being READ and not merely written;
  //   3. the bf16 ring's answer DIFFERS from the same chunked run over an f32
  //      ring. Without (3) the case would pass on an op that silently kept an f32
  //      ring behind the caller's back, which is precisely the defect a "does it
  //      run" assertion cannot see.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;

  // Chunk the sequence so the SECOND call must read what the FIRST call stored.
  const int64_t kSplit = 7;
  REQUIRE(kSplit < kConvSeqLen);
  REQUIRE(kConvSeqLen - kSplit > 0);

  auto run_chunked = [&](DType ring_dt, std::vector<unsigned char>* ring_out) {
    Queue q = CpuQ();
    std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
    Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
    std::vector<unsigned char> ring(
        static_cast<size_t>(kChannels * state_len) * vt::SizeOf(ring_dt), 0);
    Tensor t_state = MakeT(ring.data(), ring_dt, {1, kChannels, state_len});
    std::vector<float> all;
    int64_t lo = 0;
    for (int64_t n : {kSplit, kConvSeqLen - kSplit}) {
      std::vector<float> x(kConvInput + lo * kChannels,
                           kConvInput + (lo + n) * kChannels);
      std::vector<float> out(static_cast<size_t>(n * kChannels), 0.0f);
      std::vector<int32_t> qsl{0, static_cast<int32_t>(n)};
      Tensor t_x = MakeT(x.data(), DType::kF32, {n, kChannels});
      Tensor t_out = MakeT(out.data(), DType::kF32, {n, kChannels});
      Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
      vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);
      all.insert(all.end(), out.begin(), out.end());
      lo += n;
    }
    if (ring_out != nullptr) *ring_out = ring;
    return all;
  };

  std::vector<unsigned char> ring_bf16;
  const std::vector<float> got_bf16 = run_chunked(DType::kBF16, &ring_bf16);
  std::vector<unsigned char> ring_f32;
  const std::vector<float> got_f32 = run_chunked(DType::kF32, &ring_f32);
  REQUIRE(got_bf16.size() == static_cast<size_t>(kConvSeqLen * kChannels));

  // (1) THE STORED BYTES, bit-exact against the bf16 rounding of the raw input.
  // `update_conv_state` keeps the last `state_len` RAW columns
  // (`cache_utils.py:1068`), so every stored value has a known preimage.
  {
    const auto* half = reinterpret_cast<const uint16_t*>(ring_bf16.data());
    for (int64_t ch = 0; ch < kChannels; ++ch) {
      for (int64_t c = 0; c < state_len; ++c) {
        const int64_t token = kConvSeqLen - state_len + c;
        INFO("channel ", ch, " state column ", c);
        CHECK(half[static_cast<size_t>(ch * state_len + c)] ==
              vt::F32ToBF16(kConvInput[token * kChannels + ch]));
      }
    }
  }

  // (2) THE ORACLE, within bf16's own resolution of it. A ring that was never
  // read back would lose the second chunk's first `state_len` lags entirely and
  // miss this by orders of magnitude, not by a rounding.
  const std::vector<float> want(ExpectedFor(kDil),
                                ExpectedFor(kDil) + kConvSeqLen * kChannels);
  for (float v : got_bf16) REQUIRE(std::isfinite(v));
  const double sep_bf16 = MaxAbsDiff(got_bf16, want);
  INFO("bf16 ring vs the oracle: ", sep_bf16);
  CHECK(sep_bf16 < 5e-2);

  // (3) THE RING DTYPE IS LOAD-BEARING. The f32 arm is bit-exact against the
  // oracle and the bf16 arm cannot be, because the carried columns round on the
  // store — upstream rounds them too. If these two agreed exactly, the ring dtype
  // the caller asked for would not be the ring dtype the op used.
  CHECK(MaxAbsDiff(got_f32, want) < kTol);
  CHECK(std::memcmp(got_bf16.data(), got_f32.data(),
                    sizeof(float) * got_bf16.size()) != 0);
  CHECK(sep_bf16 > MaxAbsDiff(got_f32, want));
}

TEST_CASE("vt::Qwen4ExpPleConv: the state write-back is the last (K-1)*dilation raw inputs") {
  // `update_conv_state` keeps `full_conv_states[..., -conv_kernel_size:]`
  // (cache_utils.py:1068), i.e. the last `state_len` columns of
  // [old state, this chunk] — the RAW normed inputs, never the conv output and
  // never the activation. Checked structurally rather than through a golden,
  // because a state that is off by one column produces a plausible answer on the
  // NEXT chunk and none of the golden comparisons above run after it.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  Queue q = CpuQ();
  std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
  std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  std::vector<float> out(static_cast<size_t>(kConvSeqLen * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(kChannels * state_len), 0.0f);
  std::vector<int32_t> qsl{0, static_cast<int32_t>(kConvSeqLen)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {1, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);

  for (int64_t ch = 0; ch < kChannels; ++ch) {
    for (int64_t s = 0; s < state_len; ++s) {
      const int64_t token = kConvSeqLen - state_len + s;
      INFO("channel ", ch, " state column ", s);
      CHECK(state[static_cast<size_t>(ch * state_len + s)] ==
            kConvInput[token * kChannels + ch]);
    }
  }
}

TEST_CASE("vt::Qwen4ExpPleConv: the batch axis, and per-sequence state rows") {
  // Two sequences of different lengths in ONE call, each with its own cache row.
  // A kernel that walked `x` with a single running cursor, or that shared one
  // state across the batch, matches neither single-sequence answer.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  const int64_t n0 = 5, n1 = 7;
  REQUIRE(n0 + n1 == kConvSeqLen);
  Queue q = CpuQ();
  std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
  std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  std::vector<float> out(static_cast<size_t>(kConvSeqLen * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(2 * kChannels * state_len), 0.0f);
  std::vector<int32_t> qsl{0, static_cast<int32_t>(n0),
                           static_cast<int32_t>(kConvSeqLen)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {2, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {3});
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);

  const std::vector<float> seq0 = RunSingleShot(kDil, 0, n0);
  const std::vector<float> seq1 = RunSingleShot(kDil, n0, n1);
  CHECK(std::memcmp(out.data(), seq0.data(), sizeof(float) * seq0.size()) == 0);
  CHECK(std::memcmp(out.data() + n0 * kChannels, seq1.data(),
                    sizeof(float) * seq1.size()) == 0);

  // Sequence 1 is the SECOND cache row, so a kernel that wrote every sequence's
  // state to row 0 leaves row 1 zeroed and row 0 holding the wrong tail.
  for (int64_t ch = 0; ch < kChannels; ++ch) {
    for (int64_t s = 0; s < state_len; ++s) {
      INFO("row 1, channel ", ch, " column ", s);
      // Sequence 1 is 7 tokens against a 9-column state, so it is left-padded
      // with two zeros and then carries its own 7 inputs.
      const int64_t token = n1 - state_len + s;
      const float want =
          token < 0 ? 0.0f : kConvInput[(n0 + token) * kChannels + ch];
      CHECK(state[static_cast<size_t>((kChannels + ch) * state_len + s)] == want);
    }
  }
}

TEST_CASE("vt::Qwen4ExpPleConv: an EMPTY segment is an identity on its cache row") {
  // A padded batch row arrives as `qsl[s] == qsl[s+1]`. It must leave its cache
  // row byte-for-byte alone: the window has nothing to consume, so anything the
  // op wrote there would be a shift of somebody's real context by zero tokens.
  // The kernel's early-out claims to be an identity rather than a guard, and
  // this is the line that holds it to that.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  Queue q = CpuQ();
  std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
  std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  std::vector<float> out(static_cast<size_t>(kConvSeqLen * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(2 * kChannels * state_len));
  for (size_t i = 0; i < state.size(); ++i) state[i] = 0.5f + 0.001f * static_cast<float>(i);
  const std::vector<float> before = state;
  // Sequence 0 is empty; sequence 1 carries every token.
  std::vector<int32_t> qsl{0, 0, static_cast<int32_t>(kConvSeqLen)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {2, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {3});
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);

  const int64_t row_stride = kChannels * state_len;
  CHECK(std::memcmp(state.data(), before.data(),
                    sizeof(float) * static_cast<size_t>(row_stride)) == 0);
  // The non-empty neighbour still has to be right, so the empty row cannot be
  // "identity" because the whole call did nothing.
  bool moved = false;
  for (int64_t i = 0; i < row_stride; ++i) {
    if (state[static_cast<size_t>(row_stride + i)] !=
        before[static_cast<size_t>(row_stride + i)]) {
      moved = true;
    }
  }
  CHECK(moved);
}

TEST_CASE("vt::Qwen4ExpPleConv: conv_state_indices addresses cache rows independently") {
  // THE `conv_state_indices` AXIS, and why the op has one at all. A PLE layer owns THREE
  // conv states (`number_of_conv_states = 3`: the GDN conv, this conv, and the
  // n-gram token history), and a batched engine owns one shared cache whose rows
  // are handed out per sequence. Both need a row selector that is not "row s for
  // sequence s". Decoys in every other row make a kernel that ignored the
  // selector visible: it would corrupt row 0 and leave row 2 at its decoy.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  const int64_t rows = 4, pick = 2;
  Queue q = CpuQ();
  std::vector<float> x(kConvInput, kConvInput + kConvSeqLen * kChannels);
  std::vector<float> w(kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel);
  std::vector<float> out(static_cast<size_t>(kConvSeqLen * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(rows * kChannels * state_len));
  for (size_t i = 0; i < state.size(); ++i) {
    state[i] = 1000.0f + static_cast<float>(i);  // a decoy nothing can produce
  }
  const int64_t row_stride = kChannels * state_len;
  std::fill(state.begin() + pick * row_stride,
            state.begin() + (pick + 1) * row_stride, 0.0f);
  const std::vector<float> before = state;

  std::vector<int32_t> qsl{0, static_cast<int32_t>(kConvSeqLen)};
  std::vector<int32_t> idx{static_cast<int32_t>(pick)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {rows, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
  Tensor t_idx = MakeT(idx.data(), DType::kI32, {1});
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, &t_idx, args);

  const std::vector<float> want(kConvExpectedD3,
                                kConvExpectedD3 + kConvSeqLen * kChannels);
  CHECK(MaxAbsDiff(out, want) < kTol);
  for (int64_t r = 0; r < rows; ++r) {
    if (r == pick) continue;
    INFO("untouched cache row ", r);
    CHECK(std::memcmp(state.data() + r * row_stride, before.data() + r * row_stride,
                      sizeof(float) * static_cast<size_t>(row_stride)) == 0);
  }
  bool moved = false;
  for (int64_t i = 0; i < row_stride; ++i) {
    if (state[static_cast<size_t>(pick * row_stride + i)] !=
        before[static_cast<size_t>(pick * row_stride + i)]) {
      moved = true;
    }
  }
  CHECK(moved);
}

TEST_CASE("vt::Qwen4ExpPleConv: bf16 storage rounds ONCE, on the store") {
  // vt's house dtype contract, the same identity `test_qwen4_exp_hc_device.cpp`
  // states: widen on load, compute in f32, round once on the store. Feed inputs
  // that are already bf16-exact and the bf16 outputs must be EXACTLY
  // `F32ToBF16` of the f32 outputs. Strictly stronger than a bf16-eps tolerance,
  // which would absorb a narrowed accumulator.
  constexpr int64_t kDil = 3;
  const int64_t state_len = StateLen(kDil);
  Queue q = CpuQ();
  const auto bf16_exact = [](std::vector<float> v) {
    for (float& e : v) e = vt::BF16ToF32(vt::F32ToBF16(e));
    return v;
  };
  const auto to_bf16 = [](const std::vector<float>& src) {
    std::vector<uint16_t> o(src.size());
    for (size_t i = 0; i < src.size(); ++i) o[i] = vt::F32ToBF16(src[i]);
    return o;
  };
  const std::vector<float> x =
      bf16_exact({kConvInput, kConvInput + kConvSeqLen * kChannels});
  const std::vector<float> w =
      bf16_exact({kPleConv1dWeight, kPleConv1dWeight + kChannels * kKernel});

  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  std::vector<int32_t> qsl{0, static_cast<int32_t>(kConvSeqLen)};
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});

  std::vector<float> x32 = x, w32 = w;
  std::vector<float> out32(static_cast<size_t>(kConvSeqLen * kChannels), 0.0f);
  std::vector<float> st32(static_cast<size_t>(kChannels * state_len), 0.0f);
  Tensor t_x32 = MakeT(x32.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_w32 = MakeT(w32.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_o32 = MakeT(out32.data(), DType::kF32, {kConvSeqLen, kChannels});
  Tensor t_s32 = MakeT(st32.data(), DType::kF32, {1, kChannels, state_len});
  vt::Qwen4ExpPleConv(q, t_o32, t_x32, t_w32, t_s32, t_qsl, nullptr, args);

  std::vector<uint16_t> xbf = to_bf16(x), wbf = to_bf16(w);
  std::vector<uint16_t> outbf(static_cast<size_t>(kConvSeqLen * kChannels), 0);
  // The STATE stays f32 in both arms: it is a cache this op owns end to end and
  // the dispatcher refuses any other dtype, so the only store under test is out.
  std::vector<float> stbf(static_cast<size_t>(kChannels * state_len), 0.0f);
  Tensor t_xbf = MakeT(xbf.data(), DType::kBF16, {kConvSeqLen, kChannels});
  Tensor t_wbf = MakeT(wbf.data(), DType::kBF16, {kChannels, kKernel});
  Tensor t_obf = MakeT(outbf.data(), DType::kBF16, {kConvSeqLen, kChannels});
  Tensor t_sbf = MakeT(stbf.data(), DType::kF32, {1, kChannels, state_len});
  vt::Qwen4ExpPleConv(q, t_obf, t_xbf, t_wbf, t_sbf, t_qsl, nullptr, args);

  bool any_rounded = false;
  for (int64_t i = 0; i < kConvSeqLen * kChannels; ++i) {
    INFO("element ", i);
    CHECK(outbf[static_cast<size_t>(i)] ==
          vt::F32ToBF16(out32[static_cast<size_t>(i)]));
    if (vt::BF16ToF32(outbf[static_cast<size_t>(i)]) != out32[static_cast<size_t>(i)]) {
      any_rounded = true;
    }
  }
  // The store must actually lose something, or the identity holds for the
  // uninteresting reason that every value was representable.
  CHECK(any_rounded);
  // The state keeps the RAW input, so a bf16 x arm writes back bf16-exact values
  // and the two caches agree bit for bit.
  CHECK(std::memcmp(st32.data(), stbf.data(), sizeof(float) * st32.size()) == 0);
}

TEST_CASE("vt::Qwen4ExpPleConv agrees with the host reference at MODEL WIDTH") {
  // `hc_count * hidden_size` = 4 * 2560 = 10240 channels, the released config's
  // stream width, where the ~180 KiB-per-sequence state the spec accounts for
  // actually lives. Neither arm has a golden at this width; this is an agreement
  // check between two independently written implementations that are each gated
  // against the same oracle at the golden width.
  constexpr int64_t kWidth = 10240, kDil = 3, kT = 4;
  const int64_t state_len = StateLen(kDil);
  Queue q = CpuQ();
  uint64_t rng = 0x9E3779B97F4A7C15ULL;
  const auto next = [&rng]() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<int32_t>(rng >> 33)) / 2147483648.0f;
  };
  std::vector<float> x(static_cast<size_t>(kT * kWidth));
  for (float& v : x) v = next();
  std::vector<float> w(static_cast<size_t>(kWidth * kKernel));
  for (float& v : w) v = 0.5f * next();

  std::vector<float> out(static_cast<size_t>(kT * kWidth), 0.0f);
  std::vector<float> state(static_cast<size_t>(kWidth * state_len), 0.0f);
  std::vector<int32_t> qsl{0, static_cast<int32_t>(kT)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kT, kWidth});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kWidth, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kT, kWidth});
  Tensor t_state = MakeT(state.data(), DType::kF32, {1, kWidth, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
  Qwen4ExpPleConvArgs args;
  args.dilation = kDil;
  vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, args);

  vllm::qwen4_exp::PleGeometry geom;
  geom.hidden_size = 2560;
  geom.hc_count = 4;
  geom.ple_conv_kernel_size = kKernel;
  geom.ngram_size = kDil;
  vllm::qwen4_exp::PleSequenceState host;
  host.conv.assign(static_cast<size_t>(kWidth * state_len), 0.0f);
  host.tokens.assign(static_cast<size_t>(geom.context_len()), 0);
  std::vector<float> host_out(static_cast<size_t>(kT * kWidth), 0.0f);
  vllm::qwen4_exp::PleShortConv(geom, w.data(), x.data(), kT, &host, host_out.data());

  // Four terms accumulated in double on both sides, in the same order, then one
  // f32 store: BIT-IDENTICAL is the honest claim, and it is what the seam has to
  // deliver if the device arm is to replace the host one.
  CHECK(std::memcmp(out.data(), host_out.data(), sizeof(float) * out.size()) == 0);
  CHECK(std::memcmp(state.data(), host.conv.data(), sizeof(float) * state.size()) == 0);
}

TEST_CASE("vt::Qwen4ExpPleConv refuses by name") {
  constexpr int64_t kDil = 3, kT = 4;
  const int64_t state_len = StateLen(kDil);
  Queue q = CpuQ();
  std::vector<float> x(static_cast<size_t>(kT * kChannels), 0.25f);
  std::vector<float> w(static_cast<size_t>(kChannels * kKernel), 0.1f);
  std::vector<float> out(static_cast<size_t>(kT * kChannels), 0.0f);
  std::vector<float> state(static_cast<size_t>(kChannels * state_len), 0.0f);
  std::vector<int32_t> qsl{0, static_cast<int32_t>(kT)};
  Tensor t_x = MakeT(x.data(), DType::kF32, {kT, kChannels});
  Tensor t_w = MakeT(w.data(), DType::kF32, {kChannels, kKernel});
  Tensor t_out = MakeT(out.data(), DType::kF32, {kT, kChannels});
  Tensor t_state = MakeT(state.data(), DType::kF32, {1, kChannels, state_len});
  Tensor t_qsl = MakeT(qsl.data(), DType::kI32, {2});
  Qwen4ExpPleConvArgs ok;
  ok.dilation = kDil;

  SUBCASE("a Mamba-shaped K-1 state under a dilated conv") {
    // THE REFUSAL THIS OP EXISTS FOR. A caller that sized its cache with
    // `CausalConv1dFwd`'s formula gets both numbers in the message instead of an
    // answer computed off three columns where nine belong.
    std::vector<float> narrow(static_cast<size_t>(kChannels * (kKernel - 1)), 0.0f);
    Tensor bad = MakeT(narrow.data(), DType::kF32, {1, kChannels, kKernel - 1});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, bad, t_qsl, nullptr, ok),
                         doctest::Contains("(K-1)*dilation"), std::exception);
  }
  SUBCASE("dilation below 1") {
    Qwen4ExpPleConvArgs bad = ok;
    bad.dilation = 0;
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, nullptr, bad),
        doctest::Contains("dilation must be >= 1"), std::exception);
  }
  SUBCASE("a kernel width of one, which has no history at all") {
    std::vector<float> w1(static_cast<size_t>(kChannels), 0.1f);
    Tensor bad = MakeT(w1.data(), DType::kF32, {kChannels, 1});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, bad, t_state, t_qsl, nullptr, ok),
        doctest::Contains("kernel width"), std::exception);
  }
  SUBCASE("query_start_loc that does not end at T") {
    std::vector<int32_t> bad_qsl{0, static_cast<int32_t>(kT - 1)};
    Tensor bad = MakeT(bad_qsl.data(), DType::kI32, {2});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, bad, nullptr, ok),
        doctest::Contains("query_start_loc must run from 0 to T"), std::exception);
  }
  SUBCASE("a conv_state_indices entry past the end of the cache") {
    std::vector<int32_t> bad_idx{7};
    Tensor bad = MakeT(bad_idx.data(), DType::kI32, {1});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, t_qsl, &bad, ok),
        doctest::Contains("conv_state_indices out of range"), std::exception);
  }
  SUBCASE("more sequences than cache rows, with no conv_state_indices to place them") {
    std::vector<int32_t> two{0, 2, static_cast<int32_t>(kT)};
    Tensor bad = MakeT(two.data(), DType::kI32, {3});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, t_state, bad, nullptr, ok),
        doctest::Contains("one row per sequence"), std::exception);
  }
  SUBCASE("an i64 conv_state, which is not a state this conv can hold at all") {
    // WHAT REPLACED THE bf16 REFUSAL, AND WHY (W5k, #2031). This subcase used to
    // assert `conv_state must be f32` against a bf16 ring, on the argument that
    // "no arm of this op can write" one. That argument was about which KERNELS
    // existed here; the pinned oracle answers the different question of what
    // upstream STORES, and it stores the model dtype — each cache slot is
    // allocated with the dtype of the tensor that first reaches it
    // (`cache_utils.py:1019-1023`), which for this slot is `hidden_states`
    // (`modeling_qwen4_exp.py:1157-1159`). Run at `dtype=torch.bfloat16` the
    // oracle reports `conv_states[1] dtype=torch.bfloat16`. So the refusal was on
    // the wrong side and the bf16 case below is now a POSITIVE one.
    //
    // The refusal that remains is real and is not the same statement: an INTEGER
    // ring is not a rounding of the state, it is not a state. It is kept because
    // deleting the negative case entirely would leave the widened contract with
    // no lower edge at all.
    std::vector<int64_t> ints(static_cast<size_t>(kChannels * state_len), 0);
    Tensor bad = MakeT(ints.data(), DType::kI64, {1, kChannels, state_len});
    CHECK_THROWS_WITH_AS(vt::Qwen4ExpPleConv(q, t_out, t_x, t_w, bad, t_qsl, nullptr, ok),
                         doctest::Contains("conv_state must be f32 or bf16"),
                         std::exception);
  }
  SUBCASE("a weight whose channel count is not the stream's") {
    std::vector<float> w2(static_cast<size_t>((kChannels - 1) * kKernel), 0.1f);
    Tensor bad = MakeT(w2.data(), DType::kF32, {kChannels - 1, kKernel});
    CHECK_THROWS_WITH_AS(
        vt::Qwen4ExpPleConv(q, t_out, t_x, bad, t_state, t_qsl, nullptr, ok),
        doctest::Contains("weight channel dim mismatch"), std::exception);
  }
}
