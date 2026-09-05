// Qwen4-Exp (Qwen3.8-Flash-Next) W5e-2 GATE — `RunQwen4ExpPleBlock`, the PLE
// layer as ONE production composition and the LAST of the three block seams.
// Issue #2336, wave issue #2031, campaign issue #1978, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHAT IS UNDER TEST, AND WHAT IT IS COMPARED AGAINST. The block is compared
// DIRECTLY against the lane-pinned oracle, never against this repository's host
// reference: `kPleExpectedOutput` and `kPleMaskedExpectedOutput` in
// `qwen4_exp_ple_goldens.inc` are `Qwen4ExpTextPLELayer.forward` (:1169-1189)
// EXECUTED under torch by `scripts/gen-qwen4-exp-ple-goldens.py`, which lifts
// the upstream class verbatim by line range out of transformers **v5.16.0**
// (sha256 77fec77d…c459, re-fetched and re-hashed for this wave). The W2 host
// reference `PleForward` is gated on the SAME two arrays, so the composition and
// the reference answer to ONE oracle rather than to each other.
//
// NO GOLDEN WAS ADDED. The generator already carried an end-to-end PLE forward,
// its incremental twin and its masked arm, which is exactly what a block gate
// needs; extending it would have been a regeneration risk taken for nothing.
//
// WHAT THE GOLDEN CAN AND CANNOT SEE, MEASURED RATHER THAN ASSERTED. Three
// `fixture separates …` cases below run the W2 host reference with one defect
// injected and print the distance from the oracle. They gate the FIXTURE, not
// the block: a golden on which a defect is invisible is a mute switch, and this
// row has already shipped one (#2272, the eps probe that bound at two of four
// goldens). Each of the three is the paired measurement for one mutation in the
// spec's W5e-2 battery.
//
//   * the n-gram history seeded with ZERO instead of `eos_token_id` — the
//     contract that produces fluent wrong text, because 0 is a VALID token id
//     and `CacheBuffer` zero-fills;
//   * two conv taps swapped (lag 9 <-> lag 6) — the dilated conv's taps are its
//     discriminating axis, and a fixture whose history was uniform or whose ring
//     never wrapped could not tell one lag from another. `T = 12` against a
//     9-column ring, so the ring both fills and wraps;
//   * the `+ 1` dropped from one grouped-norm gamma — #2218's polarity, where
//     every `qwen4_exp` gamma is stored RAW and every consumer adds the 1.
//
// FINITENESS BEFORE TOLERANCE. `MaxAbsDiff` folds with `std::max`, which returns
// the NON-NaN operand, so an all-NaN output reads as a PERFECT match (#2272,
// #449). Every comparison here runs `support/max_abs_diff.h`'s guarded helper
// and every block output is checked finite before any bound is applied.
//
// THE ONE AXIS THE TINY FIXTURE CANNOT GATE, AND WHAT IS DONE ABOUT IT.
// `vt::BatchedMatmul` accumulates the `:1180` dot in f32 with a sequential-over-K
// loop. At the released width (H = 2560) an accumulator-order difference inside
// the clamp band |g| < 1e-6 FLIPS THE SIGN of the gate, which W5e-1's `## Owed`
// measured at 68-98 flips in 201 adversarial pairs. `the f32 score accumulator
// at MODEL width` gates that band ANALYTICALLY rather than by tolerance: the
// clamp floors |gate| at sqrt(1e-6) = 1e-3, so a sign flip can move the sigmoid
// by at most 2*(sigmoid(1e-3) - 1/2) ~= 5.0e-4 of |value| and by nothing at all
// outside the band. The case asserts both halves and prints the measured
// numbers, so the bound is executable rather than a paragraph.
//
// SCOPE, HONESTLY. CPU only — every `vt::` op this block is the first production
// caller of is registered on `kCPU` alone, and the n-gram hash is a host int64
// computation with no `vt::` op behind it. Nothing calls this BLOCK from a
// production entry point: `ForwardQwen4ExpForConditionalGeneration` still refuses
// by name, the layer loop `Qwen4ExpTextModel::Forward` is W5f under #2336 and
// #2031, and the spec's `## Owed` records it. No token claim and no speed claim.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <new>
#include <set>
#include <string>
#include <vector>

#include "support/max_abs_diff.h"
#include "vllm/platforms/interface.h"
#include "vllm/model_executor/models/qwen4_exp_ple.h"
#include "vllm/model_executor/models/qwen4_exp_ple_block.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using vllm::OwnedTensor;
using vllm::Qwen4ExpParams;
using vllm::Qwen4ExpPleBlockOutput;
using vllm::Qwen4ExpPleCaches;
using vllm::Qwen4ExpPleWeights;
using vllm::RunQwen4ExpPleBlock;
using vllm_test::MaxAbsDiff;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// `PleGeometry`, `NGramTableLayout`, `PleSequenceState` and `PleForward` are the
// W2 host reference's own names, in `vllm::qwen4_exp`. Aliased rather than
// `using namespace`d so every use below still says which side it is on.
namespace qwen4_exp = vllm::qwen4_exp;

namespace {

#include "qwen4_exp_ple_goldens.inc"  // NOLINT — golden literals

// The goldens are an f32 torch run; the block's interior is f32 with a double
// accumulator only inside `vt::Qwen4ExpPleConv` and `vt::Qwen4ExpPleGate`. This
// bound is the one every other suite on this row uses, and the smallest defect
// it has to separate — a swapped conv tap — sits three orders of magnitude
// above it; `the fixture separates …` cases re-measure that rather than assume
// it.
constexpr double kTol = 1e-5;

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

OwnedTensor OwnedFrom(DType dt, const std::vector<int64_t>& shape, const float* src) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= t.shape[i];
  }
  // Raw torch `nn.Linear` [N, K] order, which is what `vt::MatmulBT` consumes
  // and what the GGUF loader hands the forward.
  t.nk = t.rank == 2;
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(src[i]);
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = src[i];
  }
  return t;
}

constexpr int64_t kW = kTinyHcCount * kTinyHiddenSize;  // the hc-wide stream: 16
constexpr int64_t kHeads = (kTinyNgramSize - 1) * kTinyHeadsPerNgram;
constexpr int64_t kHd = kTinyPleEmbedDim / kHeads;
constexpr int64_t kCtx = kTinyNgramSize - 1;

// The tiny geometry the fixture was captured at. `head_vocab_sizes` is LEFT
// EMPTY on purpose: a `config.json` states none and the prime chain derives
// them, which is the arm the golden was produced under. The STATED arm is
// exercised by its own case below.
Qwen4ExpParams GoldenParams() {
  Qwen4ExpParams p;
  p.hidden_size = kTinyHiddenSize;
  p.hc_count = kTinyHcCount;
  p.vocab_size = kTinyVocabSize;
  p.eos_token_id = kTinyEosTokenId;
  p.rms_norm_eps = 1e-6;  // the generator's TINY config
  p.ple.layer_ids_zero_based = {0};
  p.ple.embed_dim = kTinyPleEmbedDim;
  p.ple.conv_kernel_size = kTinyConvKernel;
  p.ple.ngram_size = kTinyNgramSize;
  p.ple.heads_per_ngram = kTinyHeadsPerNgram;
  p.ple.ngram_vocab_size_base = kTinyNgramVocabBase;
  // The golden config is a `config.json`-shaped source: it STATES the base, so
  // the prime chain has the file's own inputs and the cross-check against a
  // stated head-size set is a comparison between two things the source said.
  // A GGUF-derived config sets this false, and W5g narrows the cross-check to
  // the sources that can support it. Set explicitly rather than left at the
  // default so this file's cases keep driving the refusal.
  p.ple.ngram_vocab_size_base_stated = true;
  p.ple.make_ngram_vocab_size_divisible_by = kTinyVocabDivisor;
  p.ple.seed = kTinySeed;
  return p;
}

qwen4_exp::PleGeometry GoldenGeometry() {
  return vllm::Qwen4ExpPleGeometry(GoldenParams());
}

Qwen4ExpPleWeights GoldenWeights(DType dt) {
  Qwen4ExpPleWeights w;
  w.key_proj = OwnedFrom(dt, {kW, kTinyPleEmbedDim}, kPleKeyProjWeight);
  w.value_proj = OwnedFrom(dt, {kTinyHiddenSize, kTinyPleEmbedDim}, kPleValueProjWeight);
  w.norm_key = OwnedFrom(dt, {kW}, kPleNormKeyWeight);
  w.norm_query = OwnedFrom(dt, {kW}, kPleNormQueryWeight);
  w.norm_conv = OwnedFrom(dt, {kW}, kPleNormConvWeight);
  w.conv1d = OwnedFrom(dt, {kW, kTinyConvKernel}, kPleConv1dWeight);
  return w;
}

// The W2 host reference's weight view over the same golden arrays. Used ONLY by
// the three `fixture separates …` cases, which measure what the fixture can
// see; nothing under test is ever compared against it.
struct HostWeights {
  std::vector<float> ngram, key, value, nk, nq, nc, conv;
  qwen4_exp::PleWeights View() const {
    qwen4_exp::PleWeights w;
    w.ngram_embedding = ngram.data();
    w.key_proj = key.data();
    w.value_proj = value.data();
    w.norm_key = nk.data();
    w.norm_query = nq.data();
    w.norm_conv = nc.data();
    w.conv1d = conv.data();
    return w;
  }
};

HostWeights GoldenHostWeights() {
  HostWeights h;
  h.ngram.assign(kPleNgramEmbeddingWeight,
                 kPleNgramEmbeddingWeight + kTinyPaddedVocabSize * kHd);
  h.key.assign(kPleKeyProjWeight, kPleKeyProjWeight + kW * kTinyPleEmbedDim);
  h.value.assign(kPleValueProjWeight,
                 kPleValueProjWeight + kTinyHiddenSize * kTinyPleEmbedDim);
  h.nk.assign(kPleNormKeyWeight, kPleNormKeyWeight + kW);
  h.nq.assign(kPleNormQueryWeight, kPleNormQueryWeight + kW);
  h.nc.assign(kPleNormConvWeight, kPleNormConvWeight + kW);
  h.conv.assign(kPleConv1dWeight, kPleConv1dWeight + kW * kTinyConvKernel);
  return h;
}

// One sequence's two caches, host-resident, sized for the tiny geometry and
// ZERO-FILLED — which is what `CacheBuffer` hands a layer, and therefore the
// state the block's own EOS seeding has to correct.
// THE RING CARRIES THE STREAM'S DTYPE, which is what upstream stores: each cache
// slot is allocated with the dtype of the tensor that first reaches it
// (`cache_utils.py:1019-1023`), and the tensor reaching this one is
// `hidden_states` (`modeling_qwen4_exp.py:1157-1159`). Running the pinned oracle
// at `dtype=torch.bfloat16` reports `conv_states[1] dtype=torch.bfloat16`; at
// `float32` it reports `float32`. A fixture that pinned the ring to f32 while the
// stream ran bf16 was asserting a pair upstream cannot produce, which is exactly
// how the f32 requirement survived four waves (W5k, #2031).
//
// The storage is BYTES so one buffer serves both dtypes; `View(dt)` is what says
// how wide an element is. Zero-filled either way, and all-zero bits is +0.0 in
// f32 and bf16 alike.
struct Caches {
  std::vector<unsigned char> ring;
  std::vector<int64_t> tokens;
  Caches()
      : ring(static_cast<size_t>(kW * kTinyShortConvStateLen) * sizeof(float), 0),
        tokens(static_cast<size_t>(kCtx), 0) {}
  Qwen4ExpPleCaches View(DType dt = DType::kF32) {
    Qwen4ExpPleCaches c;
    c.conv_state = MakeT(ring.data(), dt, {1, kW, kTinyShortConvStateLen});
    c.tokens = MakeT(tokens.data(), DType::kI64, {1, kCtx});
    c.state_row = 0;
    return c;
  }
};

bool AllFinite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// Run the block over one chunk and DOWNLOAD its output. `hidden` and the block
// output carry `dt`; the returned values are f32 either way, so the caller
// compares against the f32 golden in one place.
std::vector<float> RunChunkOn(vllm::dense_attn::Dev d, DType dt,
                              const Qwen4ExpPleWeights& w,
                              const OwnedTensor& table, const Qwen4ExpParams& p,
                              const qwen4_exp::NGramTableLayout& layout,
                              const float* hidden_rows, const int64_t* ids,
                              const unsigned char* mask, int64_t tokens,
                              Qwen4ExpPleCaches& caches, int64_t past_len) {
  std::vector<float> h32(hidden_rows, hidden_rows + tokens * kW);
  std::vector<uint16_t> hbf;
  Tensor hidden;
  if (dt == DType::kBF16) {
    hbf.resize(static_cast<size_t>(tokens * kW));
    for (int64_t i = 0; i < tokens * kW; ++i) hbf[static_cast<size_t>(i)] = vt::F32ToBF16(h32[static_cast<size_t>(i)]);
    hidden = MakeT(hbf.data(), DType::kBF16, {tokens, kW});
  } else {
    hidden = MakeT(h32.data(), DType::kF32, {tokens, kW});
  }
  // The step's activations live where the step runs. `MakeT` tags kCPU because
  // every other arm here IS kCPU; on a device queue this is the same host bytes
  // under the queue's own device, which is what the block is handed in
  // production.
  hidden.device = d.q.device;
  Qwen4ExpPleBlockOutput out =
      RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, ids, mask, caches, past_len);
  std::vector<float> got(static_cast<size_t>(tokens * kW));
  if (dt == DType::kBF16) {
    const auto* src = out.tensor.Ptr<uint16_t>();
    for (int64_t i = 0; i < tokens * kW; ++i)
      got[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  } else {
    std::memcpy(got.data(), out.tensor.data,
                static_cast<size_t>(tokens * kW) * sizeof(float));
  }
  return got;
}

std::vector<float> RunChunk(DType dt, const Qwen4ExpPleWeights& w,
                            const OwnedTensor& table, const Qwen4ExpParams& p,
                            const qwen4_exp::NGramTableLayout& layout,
                            const float* hidden_rows, const int64_t* ids,
                            const unsigned char* mask, int64_t tokens,
                            Qwen4ExpPleCaches& caches, int64_t past_len) {
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  return RunChunkOn(d, dt, w, table, p, layout, hidden_rows, ids, mask, tokens, caches,
                    past_len);
}

// ─── A BACKEND WHOSE HOST-BOUND `Copy` ENQUEUES, WHICH IS WHAT CUDA IS (#2496) ─
// The CPU backend's `Copy` is a `memcpy` that has already happened when it
// returns, so NO CPU-only test can see an ordering defect in a caller that reads
// the destination without draining the queue. CUDA's `Copy` is
// `cudaMemcpyAsync` on the queue's stream (src/vt/cuda/cuda_backend.cu:116-118)
// and only ENQUEUES. `CUDA_LAUNCH_BLOCKING=1` does not change that -- it
// serialises kernel LAUNCHES -- which is why #2496 reproduces under the
// serialisation that suppresses its sibling #2476.
//
// WHICH COPIES DEFER, AND WHY IT IS NOT "ALL OF THEM". A copy INTO device memory
// is stream-ordered ahead of every kernel that later reads it, so on real
// hardware it is safe and a fake that deferred it would manufacture a failure
// the device does not have -- an instrument that injects the defect it reports.
// `dense_attn::ResidentWeight` stages every weight that way. The hazard is the
// other direction: a copy into HOST memory that the HOST then dereferences, with
// nothing draining the queue in between. That is the one this defers, and it is
// exactly the branch the PLE block takes to read the n-gram history.
//
// `Memset` is performed eagerly for the same reason: it is stream-ordered
// against the kernel that reads the ring.
class DeferringBackend : public vt::Backend {
 public:
  ~DeferringBackend() override = default;
  void* Alloc(size_t bytes) override {
    void* p = ::operator new(bytes, std::align_val_t(64));
    device_.insert(p);
    sizes_[p] = bytes;
    return p;
  }
  void Free(void* p) override {
    device_.erase(p);
    sizes_.erase(p);
    ::operator delete(p, std::align_val_t(64));
  }
  void Memset(Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    if (IsDevice(dst)) {  // stream-ordered ahead of its readers on real hardware
      std::memcpy(dst, src, bytes);
      ++eager_;
      return;
    }
    // Host-bound: ENQUEUED. The source is snapshotted rather than read at flush,
    // so this instrument measures the READ ordering alone and a green result
    // cannot be credited to some other half of the change.
    pending_.push_back(Pending{dst, std::vector<unsigned char>(
                                       static_cast<const unsigned char*>(src),
                                       static_cast<const unsigned char*>(src) + bytes)});
    ++deferred_;
  }
  Queue CreateQueue() override { return Queue{Device{DeviceType::kXPU, 0}, nullptr}; }
  void Synchronize(Queue&) override {
    for (const Pending& c : pending_) std::memcpy(c.dst, c.bytes.data(), c.bytes.size());
    drained_ += pending_.size();
    pending_.clear();
    ++syncs_;
  }
  bool UnifiedMemory() const override { return true; }
  // Host memory throughout, so the portable reference tier may install the CPU
  // kernels for this device and every `vt::` op in the block runs the SAME code
  // the arms above gate. Only the transfer ordering differs.
  bool DeviceMemoryIsHostAddressable() const override { return true; }

  bool IsDevice(const void* p) const {
    for (const auto& kv : sizes_) {
      const auto* b = static_cast<const unsigned char*>(kv.first);
      const auto* q = static_cast<const unsigned char*>(p);
      if (q >= b && q < b + kv.second) return true;
    }
    return false;
  }
  size_t deferred() const { return deferred_; }
  size_t eager() const { return eager_; }
  size_t drained() const { return drained_; }
  size_t syncs() const { return syncs_; }
  size_t outstanding() const { return pending_.size(); }

 private:
  struct Pending {
    void* dst;
    std::vector<unsigned char> bytes;
  };
  std::vector<Pending> pending_;
  std::set<void*> device_;
  std::map<void*, size_t> sizes_;
  size_t deferred_ = 0, eager_ = 0, drained_ = 0, syncs_ = 0;
};

}  // namespace

TEST_CASE("RunQwen4ExpPleBlock reproduces the pinned oracle, single-shot prefill") {
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  // The layout the block derives must be the one the oracle was captured under.
  // Asserted against the goldens' own dump of upstream's resolved sizes, not
  // read back out of the object under test.
  for (int64_t h = 0; h < kHeads; ++h) {
    CHECK(layout.head_vocab_sizes[static_cast<size_t>(h)] == kTinyHeadVocabSizes[h]);
    CHECK(layout.head_offsets[static_cast<size_t>(h)] == kTinyHeadOffsets[h]);
  }
  CHECK(layout.padded_vocab_size == kTinyPaddedVocabSize);

  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);
  Caches c;
  Qwen4ExpPleCaches cv = c.View();
  const std::vector<float> got =
      RunChunk(DType::kF32, w, table, p, layout, kPleHiddenStates, kNgramTokens, nullptr,
               kNgramTotalLen, cv, /*past_len=*/0);
  REQUIRE(AllFinite(got));
  CHECK(MaxAbsDiff(got, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW)) < kTol);
}

TEST_CASE("prefill(10) + decode + decode equals the single shot, so the RING WRAPS") {
  // The nine-column ring FILLS at t = 9 and is still being rewritten at t = 11,
  // so this arm is the one that carries state across calls. Upstream asserts the
  // same equality inside the generator before dumping the golden.
  REQUIRE(kNgramTotalLen > kTinyShortConvStateLen);
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);

  Caches c;
  Qwen4ExpPleCaches cv = c.View();
  std::vector<float> got;
  int64_t lo = 0;
  for (int64_t count : {kNgramPrefillLen, static_cast<int64_t>(1), static_cast<int64_t>(1)}) {
    const std::vector<float> part =
        RunChunk(DType::kF32, w, table, p, layout, kPleHiddenStates + lo * kW,
                 kNgramTokens + lo, nullptr, count, cv, /*past_len=*/lo);
    got.insert(got.end(), part.begin(), part.end());
    lo += count;
  }
  REQUIRE(lo == kNgramTotalLen);
  REQUIRE(AllFinite(got));
  CHECK(MaxAbsDiff(got, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW)) < kTol);

  // The history the block leaves behind is the last `ngram_size - 1` ids, and it
  // is the SEED for whatever comes next. Compared against the input ids, not
  // against a value read back from the cache before the run.
  for (int64_t i = 0; i < kCtx; ++i) {
    CHECK(c.tokens[static_cast<size_t>(i)] == kNgramTokens[kNgramTotalLen - kCtx + i]);
  }
}

TEST_CASE(
    "DECODE over a device cache whose host-bound Copy ENQUEUES: the n-gram "
    "history must be drained before it is hashed (#2496)") {
  // WHAT THIS SEPARATES, AND WHY NO EXISTING CASE COULD. The case above runs the
  // same prefill(10)+decode+decode over a HOST-resident history, where the block
  // takes `std::memcpy` and no ordering question exists. On the production CUDA
  // path the history is the engine's own cache on `input.queue.device`, the block
  // takes `Backend::Copy`, and that call only ENQUEUES. Every arm on this row was
  // CPU-only, so the tree had never run this branch with a Copy that is not
  // already complete when it returns.
  //
  // THE BRANCH IS DECODE-ONLY. At `past_len == 0` the block seeds the history
  // with EOS on the host and reads nothing; only `past_len != 0` takes the
  // device read. A defect there leaves a prefill exactly right and every decode
  // step wrong, which is the shape #2496 measured on `thor:gpu0`: token 0 equal
  // to the CPU control and tokens 1-7 not.
  REQUIRE(kNgramTotalLen > kTinyShortConvStateLen);
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);

  // kXPU is unused by this build and by this binary; registering the double here
  // gives the block a device queue whose ops resolve to the CPU kernels through
  // the documented portable reference tier (include/vt/op_provider.h), which
  // installs only because this backend's memory really is host-addressable.
  static DeferringBackend b;
  vt::RegisterBackend(DeviceType::kXPU, &b);
  // The block's weight staging consults the platform registry. kXPU has no
  // platform in a CPU build, and the double's memory IS host memory, so the CPU
  // platform is the truthful answer for it rather than a stub.
  vllm::platforms::RegisterPlatform(DeviceType::kXPU,
                                    &vllm::platforms::GetPlatform(DeviceType::kCPU));
  const int installed = vt::RegisterReferenceTier(DeviceType::kXPU);
  REQUIRE(installed > 0);  // the ops resolve; a 0 here would run nothing

  Queue q = b.CreateQueue();
  vllm::dense_attn::Dev d{b, q};

  // BOTH caches live on the device, as they do on the production by-name path.
  // Their storage comes from the backend's own allocator, which is what makes
  // the write-back a device-bound copy and the read a host-bound one.
  const size_t ring_bytes = static_cast<size_t>(kW * kTinyShortConvStateLen) * sizeof(float);
  const size_t tok_bytes = static_cast<size_t>(kCtx) * sizeof(int64_t);
  void* d_ring = b.Alloc(ring_bytes);
  void* d_tok = b.Alloc(tok_bytes);
  std::memset(d_ring, 0, ring_bytes);
  std::memset(d_tok, 0, tok_bytes);
  Qwen4ExpPleCaches cv;
  cv.conv_state = MakeT(d_ring, DType::kF32, {1, kW, kTinyShortConvStateLen});
  cv.tokens = MakeT(d_tok, DType::kI64, {1, kCtx});
  cv.conv_state.device = q.device;
  cv.tokens.device = q.device;
  cv.state_row = 0;

  std::vector<float> got;
  int64_t lo = 0;
  for (int64_t count : {kNgramPrefillLen, static_cast<int64_t>(1), static_cast<int64_t>(1)}) {
    const std::vector<float> part =
        RunChunkOn(d, DType::kF32, w, table, p, layout, kPleHiddenStates + lo * kW,
                   kNgramTokens + lo, nullptr, count, cv, /*past_len=*/lo);
    got.insert(got.end(), part.begin(), part.end());
    lo += count;
  }
  REQUIRE(lo == kNgramTotalLen);

  // THE INSTRUMENT RAN, as a counted property rather than an assumption. The two
  // decode steps are the only calls that read the history back to the host, so
  // exactly two host-bound copies are enqueued; a harness that quietly took the
  // host-resident branch would report zero and its PASS would mean nothing.
  CHECK(b.deferred() == 2);
  CHECK(b.drained() == b.deferred());
  CHECK(b.outstanding() == 0);
  CHECK(b.eager() > 0);  // the weights and the write-backs did move

  REQUIRE(AllFinite(got));
  // THE SAME oracle the host-resident arm above is held to. Residency is not a
  // value, so a device cache cannot move one.
  CHECK(MaxAbsDiff(got, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW)) < kTol);
  const auto* left = static_cast<const int64_t*>(d_tok);
  for (int64_t i = 0; i < kCtx; ++i) {
    CHECK(left[i] == kNgramTokens[kNgramTotalLen - kCtx + i]);
  }
  b.Free(d_ring);
  b.Free(d_tok);
}

TEST_CASE("the STREAM DTYPE is inherited: a bf16 arm moves bf16 bytes end to end") {
  // AGENTS.md "Inherit vLLM defaults": vLLM resolves ONE model dtype and every
  // layer inherits it, so a bf16 model must move bf16 bytes through this block
  // and not f32 ones. A token gate cannot see a dtype that is too wide — the
  // goldens would still pass — so what is asserted is that the bf16 arm RUNS and
  // lands within bf16's own resolution of the same oracle output.
  //
  // The ONE f32 buffer in the block is the [T, hc] score, and it is f32 because
  // `vt::Qwen4ExpPleGate` refuses a bf16 score: it is the argument of a sigmoid
  // and a transcendental's input must not be rounded.
  //
  // THE CONV RING IS bf16 ON THIS ARM, and until W5k this case passed an f32 ring
  // into a bf16 stream — the very widening the case's own title says it gates.
  // The comment that stood here said the ring was "f32 for its own reason:
  // `vt::Qwen4ExpPleConv` admits no other state dtype, because no kernel exists
  // that would write one". That was a fact about which KERNELS this tree had, not
  // about what upstream stores, and the pinned oracle stores the model dtype
  // (`cache_utils.py:1019-1023`; observed at `conv_states[1] dtype=torch.bfloat16`
  // on a bf16 run). The op now admits it and the CPU kernel writes it.
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kBF16);
  const OwnedTensor table =
      OwnedFrom(DType::kBF16, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);
  Caches c;
  Qwen4ExpPleCaches cv = c.View(DType::kBF16);
  const std::vector<float> got =
      RunChunk(DType::kBF16, w, table, p, layout, kPleHiddenStates, kNgramTokens, nullptr,
               kNgramTotalLen, cv, /*past_len=*/0);
  REQUIRE(AllFinite(got));
  const double sep =
      MaxAbsDiff(got, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW));
  INFO("bf16 arm vs the f32 oracle: " << sep);
  // THE BOUND SITS BETWEEN TWO MEASURED NUMBERS, which is what makes it a gate
  // rather than a shape check. Clean it is 1.007e-2, bf16's own ~2^-8 relative
  // resolution on values of O(1); with the `sqrt(hidden_size)` gate divisor
  // dropped (spec mutation M8) it is 9.171e-2. A bound of 0.2 admitted BOTH and
  // this one does not, so the bf16 arm convicts the same defect population the
  // f32 arm does instead of merely proving that bf16 bytes flow.
  CHECK(sep < 0.05);
}

TEST_CASE("the conv_mask masks BOTH tensors, single-shot and incremental") {
  // Zeros at 3 and 4 are INTERIOR, so the dilation carries them to t = 6, 9 and
  // 12 as well as their own rows: a trailing-pad-only mask would leave the conv
  // path almost untouched, and this fixture is not that.
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);

  SUBCASE("single-shot") {
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    const std::vector<float> got =
        RunChunk(DType::kF32, w, table, p, layout, kPleHiddenStates, kPleMaskTokens,
                 kPleConvMask, kNgramTotalLen, cv, /*past_len=*/0);
    REQUIRE(AllFinite(got));
    CHECK(MaxAbsDiff(got, kPleMaskedExpectedOutput,
                     static_cast<size_t>(kNgramTotalLen * kW)) < kTol);
  }
  SUBCASE("prefill(10) + decode + decode, so the mask reaches the nine-column state") {
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    std::vector<float> got;
    int64_t lo = 0;
    for (int64_t count : {kNgramPrefillLen, static_cast<int64_t>(1), static_cast<int64_t>(1)}) {
      const std::vector<float> part =
          RunChunk(DType::kF32, w, table, p, layout, kPleHiddenStates + lo * kW,
                   kPleMaskTokens + lo, kPleConvMask + lo, count, cv, /*past_len=*/lo);
      got.insert(got.end(), part.begin(), part.end());
      lo += count;
    }
    REQUIRE(AllFinite(got));
    CHECK(MaxAbsDiff(got, kPleMaskedExpectedOutput,
                     static_cast<size_t>(kNgramTotalLen * kW)) < kTol);
  }
  SUBCASE("the masked rows are EXACTLY zero in the skip term's own contribution") {
    // A masked row's output is `0 + conv(masked history)`, so it is NOT zero —
    // asserting that it were would be wrong. What IS pinned: the golden's masked
    // rows differ from the unmasked run's, so the mask is not inert here.
    const size_t n = static_cast<size_t>(kNgramTotalLen * kW);
    const std::vector<float> unmasked(kPleExpectedOutput, kPleExpectedOutput + n);
    const double sep = MaxAbsDiff(unmasked, kPleMaskedExpectedOutput, n);
    INFO("masked vs unmasked separation " << sep);
    CHECK(sep > 1e2 * kTol);
  }
}

TEST_CASE("the conv_mask's PAIRED obligation is refused by name") {
  // The hash reads token IDS, not activations, so a masked position whose id is
  // not EOS leaks padding into the hash — every id in range, the gather
  // successful, the answer wrong. This half of the pair has never had an
  // enforcer anywhere in this tree.
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);
  Caches c;
  Qwen4ExpPleCaches cv = c.View();
  std::vector<int64_t> bad(kPleMaskTokens, kPleMaskTokens + kNgramTotalLen);
  REQUIRE(kPleConvMask[3] == 0);
  bad[3] = 9;  // in range, not EOS
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  std::vector<float> h(kPleHiddenStates, kPleHiddenStates + kNgramTotalLen * kW);
  Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
  CHECK_THROWS_WITH_AS(
      RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, bad.data(), kPleConvMask, cv, 0),
      doctest::Contains("PAIRED obligation"), std::runtime_error);
}

TEST_CASE("the fixture separates a ZERO-seeded n-gram history from the EOS-seeded one") {
  // ZERO IS A VALID TOKEN ID and `CacheBuffer` zero-fills, so this is the defect
  // a port gets by trusting the cache it was handed. The distance is measured
  // through the W2 HOST reference, which is gated on the same golden, so the
  // number below is a property of the FIXTURE and not of the block.
  const qwen4_exp::PleGeometry geom = GoldenGeometry();
  const qwen4_exp::NGramTableLayout layout =
      qwen4_exp::BuildNGramTableLayout(geom, 0);
  const HostWeights hw = GoldenHostWeights();

  qwen4_exp::PleSequenceState eos_state;
  eos_state.Reset(geom);
  std::vector<float> eos_out(static_cast<size_t>(kNgramTotalLen * kW));
  qwen4_exp::PleForward(geom, layout, hw.View(), kPleHiddenStates, kNgramTokens,
                        kNgramTotalLen, nullptr, &eos_state, eos_out.data());
  REQUIRE(AllFinite(eos_out));
  // The EOS-seeded reference is the oracle's own answer; that it is says the
  // measurement below is against a live fixture rather than a broken one.
  REQUIRE(MaxAbsDiff(eos_out, kPleExpectedOutput,
                     static_cast<size_t>(kNgramTotalLen * kW)) < kTol);

  qwen4_exp::PleSequenceState zero_state;
  zero_state.Reset(geom);
  for (int64_t i = 0; i < kCtx; ++i) zero_state.tokens[static_cast<size_t>(i)] = 0;
  std::vector<float> zero_out(static_cast<size_t>(kNgramTotalLen * kW));
  qwen4_exp::PleForward(geom, layout, hw.View(), kPleHiddenStates, kNgramTokens,
                        kNgramTotalLen, nullptr, &zero_state, zero_out.data());
  REQUIRE(AllFinite(zero_out));
  const double sep =
      MaxAbsDiff(zero_out, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW));
  INFO("zero-seeded vs EOS-seeded separation " << sep);
  CHECK(sep > 1e2 * kTol);
}

TEST_CASE("the fixture separates SWAPPED conv taps, so the lag set is gateable") {
  // The dilated conv's taps are its discriminating axis. `T = 12` against a
  // nine-column ring, so lag 9 and lag 6 both read live history at t >= 9 and a
  // swap has somewhere to show. A fixture whose ring never filled could not tell
  // the two apart at all, which is why the wrap is asserted here too.
  REQUIRE(kNgramTotalLen > kTinyShortConvStateLen);
  const qwen4_exp::PleGeometry geom = GoldenGeometry();
  const qwen4_exp::NGramTableLayout layout = qwen4_exp::BuildNGramTableLayout(geom, 0);
  HostWeights hw = GoldenHostWeights();
  for (int64_t c = 0; c < kW; ++c) {
    std::swap(hw.conv[static_cast<size_t>(c * kTinyConvKernel + 0)],
              hw.conv[static_cast<size_t>(c * kTinyConvKernel + 1)]);
  }
  qwen4_exp::PleSequenceState st;
  st.Reset(geom);
  std::vector<float> out(static_cast<size_t>(kNgramTotalLen * kW));
  qwen4_exp::PleForward(geom, layout, hw.View(), kPleHiddenStates, kNgramTokens,
                        kNgramTotalLen, nullptr, &st, out.data());
  REQUIRE(AllFinite(out));
  const double sep =
      MaxAbsDiff(out, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW));
  INFO("lag-9 <-> lag-6 swap separation " << sep);
  CHECK(sep > 1e2 * kTol);
}

TEST_CASE("the fixture separates a DROPPED `+ 1` on a grouped-norm gamma") {
  // #2218's polarity: every `qwen4_exp` gamma is stored RAW and every consumer
  // adds the 1. `GroupedRmsNorm` applies `(1 + w)`, so passing `w - 1` is
  // exactly the consumer that forgot it.
  const qwen4_exp::PleGeometry geom = GoldenGeometry();
  const qwen4_exp::NGramTableLayout layout = qwen4_exp::BuildNGramTableLayout(geom, 0);
  HostWeights hw = GoldenHostWeights();
  for (int64_t i = 0; i < kW; ++i) hw.nc[static_cast<size_t>(i)] -= 1.0F;
  qwen4_exp::PleSequenceState st;
  st.Reset(geom);
  std::vector<float> out(static_cast<size_t>(kNgramTotalLen * kW));
  qwen4_exp::PleForward(geom, layout, hw.View(), kPleHiddenStates, kNgramTokens,
                        kNgramTotalLen, nullptr, &st, out.data());
  REQUIRE(AllFinite(out));
  const double sep =
      MaxAbsDiff(out, kPleExpectedOutput, static_cast<size_t>(kNgramTotalLen * kW));
  INFO("dropped `+1` on norm_conv separation " << sep);
  CHECK(sep > 1e2 * kTol);
}

TEST_CASE("the f32 score accumulator at MODEL width: the clamp band is BOUNDED") {
  // `vt::BatchedMatmul` accumulates the `:1180` dot in f32 with a
  // sequential-over-K loop. At the released width (H = 2560) that disagrees with
  // a double accumulator, and inside the clamp band |g| < 1e-6 the disagreement
  // can FLIP THE SIGN of the gate — a discrete outcome, which no tolerance
  // gates: W5e-1's `## Owed` measured 68-98 flips in 201 adversarial pairs.
  //
  // WHAT IS GATED IS THE CONSEQUENCE, AND IT IS BOUNDED ANALYTICALLY IN TWO
  // PARTS, each asserted separately so either can be false:
  //
  //   1. THE JUMP. The clamp floors |gate| at sqrt(1e-6) = 1e-3, so the two
  //      answers a sign flip can give are sigmoid(+1e-3) and sigmoid(-1e-3) and
  //      they differ by 2*(sigmoid(1e-3) - 1/2) ~= 5.0e-4 of |value|,
  //      INDEPENDENT of H. That is the whole cost of a flip, and it is why the
  //      band is a bounded hazard rather than an unbounded one.
  //   2. THE CONTINUOUS PART. Off the jump, d/dg sigmoid(sign(g) sqrt(|g|)) is
  //      sigmoid'(gate) / (2 sqrt(|g|)), which is largest at the band edge:
  //      0.25 / (2 * 1e-3) = 125. So a score disagreement of `dg` costs at most
  //      125 * dg, and `dg` is measured here against a double accumulator and
  //      held under `kScoreAccumBound`.
  //
  // The oracle cannot be run at this width on this host — nothing published fits
  // any device this project owns — so the reference is a second accumulator and
  // not the pin. The pin gates the tiny width in the cases above; this case
  // gates the axis the tiny width cannot see, which is the whole reason it
  // exists.
  Queue q = CpuQ();
  const int64_t H = 2560, hc = 4, T = 3;
  const int64_t Wm = hc * H;
  std::vector<float> key(static_cast<size_t>(T * Wm)), query(static_cast<size_t>(T * Wm));
  // ADVERSARIAL, and only for HALF the pairs, so both populations exist and both
  // assertions below can be false.
  //
  // `j == 0` sums a large positive PREFIX and then subtracts it back term for
  // term, so the exact dot is zero while every partial sum is O(H/2) — which is
  // what makes an f32 accumulator disagree at all. A construction that
  // alternated +x, -x adjacently would cancel EXACTLY in f32 too, report a
  // disagreement of zero, and gate nothing; an earlier draft of this case did
  // exactly that and is the reason this comment names the shape.
  //
  // `j > 0` is `query == key`, a dot of O(H/2) nowhere near the band.
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < hc; ++j) {
      const int64_t base = t * Wm + j * H;
      for (int64_t dd = 0; dd < H; ++dd) {
        const int64_t k = dd < H / 2 ? dd : dd - H / 2;
        const double x = std::sin(0.911 * static_cast<double>(k + 7 * j + 31 * t)) +
                         1.25 * std::cos(0.37 * static_cast<double>(k));
        key[static_cast<size_t>(base + dd)] = static_cast<float>(x);
        query[static_cast<size_t>(base + dd)] =
            static_cast<float>(j == 0 && dd >= H / 2 ? -x : x);
      }
    }
  }
  std::vector<float> score(static_cast<size_t>(T * hc), 0.0F);
  Tensor a = MakeT(key.data(), DType::kF32, {T * hc, 1, H});
  Tensor b = MakeT(query.data(), DType::kF32, {T * hc, H, 1});
  Tensor o = MakeT(score.data(), DType::kF32, {T * hc, 1, 1});
  vt::BatchedMatmul(q, o, a, b);

  const double divisor = std::sqrt(static_cast<double>(H));
  // The f32 accumulator's disagreement with a double one, in `g` units, at this
  // width and on this fixture. A stated BOUND and not a value read back: the
  // measurement is 1.06e-4 and this sits an order above it, which is what makes
  // the assertion falsifiable in both directions rather than a transcription.
  const double kScoreAccumBound = 1e-3;
  // THE ANALYTIC BOUND ON THE CONSEQUENCE, derived rather than measured. For two
  // scores at most `dg` apart, the worst case is the pair (+dg/2, -dg/2)
  // straddling the origin: `gate` is then +/- sqrt(max(dg/2, 1e-6)) — the clamp
  // is what puts the `max` there — and `sigmoid` is 1/4-Lipschitz, so
  //
  //     |sigmoid(gate_a) - sigmoid(gate_b)| <= 0.5 * sqrt(max(dg/2, 1e-6)).
  //
  // Away from the origin the square root is Lipschitz with a constant of
  // 1/(2 sqrt|g|) and the difference is strictly smaller, so this is a GLOBAL
  // bound and not a local one.
  const double kSigmoidBound =
      0.5 * std::sqrt(std::max(kScoreAccumBound / 2.0, 1e-6));

  auto gate_of = [](double g) {
    if (g == 0.0) return 0.0;  // sign(0) == 0: the origin is not on the floor
    const double root = std::sqrt(std::abs(g) < 1e-6 ? 1e-6 : std::abs(g));
    return g > 0.0 ? root : -root;
  };
  auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };

  int in_band = 0, out_band = 0, flips = 0;
  double worst_in = 0.0, worst_out = 0.0, worst_dg = 0.0;
  for (int64_t i = 0; i < T * hc; ++i) {
    double exact = 0.0;
    const int64_t t = i / hc, j = i % hc;
    const int64_t base = t * Wm + j * H;
    for (int64_t dd = 0; dd < H; ++dd) {
      exact += static_cast<double>(key[static_cast<size_t>(base + dd)]) *
               static_cast<double>(query[static_cast<size_t>(base + dd)]);
    }
    const double g_f32 = static_cast<double>(score[static_cast<size_t>(i)]) / divisor;
    const double g_dbl = exact / divisor;
    REQUIRE(std::isfinite(g_f32));
    REQUIRE(std::isfinite(g_dbl));
    worst_dg = std::max(worst_dg, std::abs(g_f32 - g_dbl));
    const double delta = std::abs(sig(gate_of(g_f32)) - sig(gate_of(g_dbl)));
    if ((g_f32 > 0.0) != (g_dbl > 0.0)) ++flips;
    // NEAR-NULL, not "inside the band": the disagreement measured here is ~100x
    // the band's own width, so the f32 score does not stay inside it — which is
    // itself the finding, and is why the bound above is the straddling one and
    // not the flip-only 2*(sigmoid(1e-3) - 1/2) = 5.0e-4.
    if (std::abs(g_f32) < kScoreAccumBound || std::abs(g_dbl) < kScoreAccumBound) {
      ++in_band;
      worst_in = std::max(worst_in, delta);
    } else {
      ++out_band;
      worst_out = std::max(worst_out, delta);
    }
  }
  INFO("near-null pairs " << in_band << " (sign flips " << flips
                          << ", worst sigmoid delta " << worst_in << "); dense "
                          << out_band << " worst " << worst_out
                          << "; worst |dg| " << worst_dg);
  // The fixture must actually PROBE both populations, or each bound below is a
  // mute switch on the other's side.
  CHECK(in_band > 0);
  CHECK(out_band > 0);
  // A sign flip must actually HAPPEN here, or this case is measuring an axis it
  // never reached. Two of three near-null pairs flip on this fixture.
  CHECK(flips > 0);
  // The disagreement itself, which is what the bound below is derived from.
  CHECK(worst_dg < kScoreAccumBound);
  // The consequence near the origin. THIS IS 1e2 to 1e3 TIMES `kTol`, and that
  // is the honest statement rather than a failure: upstream's own `:1181` is
  // discontinuous at the origin and its `:1180` sum is an f32 reduction too
  // (torch's `acc_type` for a float sum is float), so two legitimate summation
  // ORDERS differ here by construction. No port of this line can be gated to
  // 1e-5 near the origin at model width, ours or anyone's, and the spec's
  // `## Owed` records that rather than leaving it to be rediscovered as a bug.
  CHECK(worst_in <= kSigmoidBound);
  // Away from the origin the gate is a well-conditioned function of a
  // well-conditioned dot, and the f32 accumulator costs ordinary rounding —
  // which is what makes the near-null population the ONLY place this matters.
  CHECK(worst_out < kTol);
}

TEST_CASE("Qwen4ExpPleLayout refuses STATED head vocabulary sizes that disagree") {
  // A `qwen4exp` GGUF states them and a `config.json` does not, so both arms
  // exist. Where both exist they are the same set or the table is addressed
  // wrong, and no shape check downstream can see the difference.
  Qwen4ExpParams p = GoldenParams();
  SUBCASE("agreeing sizes pass through") {
    p.ple.head_vocab_sizes.assign(kTinyHeadVocabSizes, kTinyHeadVocabSizes + kHeads);
    const qwen4_exp::NGramTableLayout l = vllm::Qwen4ExpPleLayout(p, 0);
    CHECK(l.padded_vocab_size == kTinyPaddedVocabSize);
  }
  SUBCASE("one disagreeing size is refused, and the message names both numbers") {
    p.ple.head_vocab_sizes.assign(kTinyHeadVocabSizes, kTinyHeadVocabSizes + kHeads);
    p.ple.head_vocab_sizes[1] += 2;
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpPleLayout(p, 0),
                         doctest::Contains("vocabulary size disagrees"),
                         std::runtime_error);
  }
  SUBCASE("a short stated set is refused") {
    p.ple.head_vocab_sizes.assign(kTinyHeadVocabSizes, kTinyHeadVocabSizes + kHeads - 1);
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpPleLayout(p, 0),
                         doctest::Contains("per-head vocabulary sizes"),
                         std::runtime_error);
  }
}

// ─── W5g (#2031) — THE STATED SET IS THE AUTHORITY ───────────────────────────
//
// The case above drives the arm where the SOURCE stated the base, which is a
// `config.json`. This one drives the arm a `qwen4exp` GGUF actually takes: the
// resolved head sizes are stated, `ngram_vocab_size_base` is NOT, and the value
// sitting in the field is upstream's 20,000,000 default rather than anything the
// file said.
//
// Both halves matter and each is asserted against the OTHER's failure mode. The
// derived chain from 20,000,000 opens at 20,000,003, so a layout that still
// derived would be off by seven orders of magnitude on head 0's offset and the
// table's row count — visible in the numbers below, not merely in a throw.
TEST_CASE("Qwen4ExpPleLayout takes the STATED sizes when the source stated no base") {
  Qwen4ExpParams p = GoldenParams();
  p.ple.head_vocab_sizes.assign(kTinyHeadVocabSizes, kTinyHeadVocabSizes + kHeads);

  SUBCASE("no stated base: the stated set builds the layout and nothing is refused") {
    p.ple.ngram_vocab_size_base_stated = false;
    p.ple.ngram_vocab_size_base = 20000000;  // upstream's default, as a GGUF leaves it
    const qwen4_exp::NGramTableLayout l = vllm::Qwen4ExpPleLayout(p, 0);
    // The VALUES, not merely "it did not throw". Every one of these is what the
    // stated set gives and none of them is what the chain from 20,000,000 gives.
    REQUIRE(l.head_vocab_sizes.size() == static_cast<size_t>(kHeads));
    int64_t running = 0;
    for (int64_t h = 0; h < kHeads; ++h) {
      CHECK(l.head_vocab_sizes[static_cast<size_t>(h)] == kTinyHeadVocabSizes[h]);
      CHECK(l.head_offsets[static_cast<size_t>(h)] == running);
      running += kTinyHeadVocabSizes[h];
    }
    CHECK(l.total_vocab_size == running);
    CHECK(l.padded_vocab_size == kTinyPaddedVocabSize);
    // The chain's own head 0 is the FIRST prime after 19,999,999, which is what
    // a layout that ignored the stated set would have used. Asserted as ABSENT
    // so this case cannot pass on a coincidence.
    CHECK(l.head_offsets[0] == 0);
    CHECK(l.padded_vocab_size < 20000000);
  }

  SUBCASE("the container's own head_offsets are cross-checked against its sizes") {
    p.ple.ngram_vocab_size_base_stated = false;
    p.ple.head_offsets.assign(static_cast<size_t>(kHeads), 0);
    int64_t running = 0;
    for (int64_t h = 0; h < kHeads; ++h) {
      p.ple.head_offsets[static_cast<size_t>(h)] = running;
      running += kTinyHeadVocabSizes[h];
    }
    // Agreeing offsets pass through and reach the layout unchanged.
    const qwen4_exp::NGramTableLayout ok = vllm::Qwen4ExpPleLayout(p, 0);
    CHECK(ok.head_offsets == p.ple.head_offsets);
    // ONE wrong offset is refused, and the message names both numbers. This is
    // the defect no shape check downstream can see: the row COUNT is unchanged,
    // so a wrong offset gathers another head's rows out of a correctly sized
    // table.
    p.ple.head_offsets[static_cast<size_t>(kHeads - 1)] += 1;
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpPleLayout(p, 0),
                         doctest::Contains("offset disagrees"),
                         std::runtime_error);
  }

  SUBCASE("a stated set on a PLE layer OTHER than index 0 is refused") {
    // Upstream derives a different vocabulary for every PLE layer, from the
    // GLOBAL head index. The container states one flat array and says nothing
    // about which layer it describes, so only index 0 is unambiguous.
    p.ple.ngram_vocab_size_base_stated = false;
    CHECK_THROWS_WITH_AS(vllm::Qwen4ExpPleLayout(p, 1),
                         doctest::Contains("cannot be known to describe it"),
                         std::runtime_error);
  }
}

TEST_CASE("RunQwen4ExpPleBlock refuses by name") {
  const Qwen4ExpParams p = GoldenParams();
  const qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
  const Qwen4ExpPleWeights w = GoldenWeights(DType::kF32);
  const OwnedTensor table =
      OwnedFrom(DType::kF32, {kTinyPaddedVocabSize, kHd}, kPleNgramEmbeddingWeight);
  Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};
  std::vector<float> h(kPleHiddenStates, kPleHiddenStates + kNgramTotalLen * kW);

  SUBCASE("hidden must be the hc-WIDE stream, not the collapsed hidden state") {
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    Tensor bad = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kTinyHiddenSize});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table, p, layout, bad, kNgramTokens, nullptr, cv, 0),
        doctest::Contains("hc-WIDE hyper-connection stream"), std::runtime_error);
  }
  SUBCASE("the conv ring is (kernel-1)*dilation deep, not kernel-1") {
    Caches c;
    std::vector<float> shallow(static_cast<size_t>(kW * (kTinyConvKernel - 1)), 0.0F);
    Qwen4ExpPleCaches cv = c.View();
    cv.conv_state = MakeT(shallow.data(), DType::kF32, {1, kW, kTinyConvKernel - 1});
    Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, kNgramTokens, nullptr, cv, 0),
        doctest::Contains("the conv is DILATED"), std::runtime_error);
  }
  SUBCASE("the n-gram history must be i64, because it holds TOKEN IDS") {
    Caches c;
    std::vector<float> wrong(static_cast<size_t>(kCtx), 0.0F);
    Qwen4ExpPleCaches cv = c.View();
    cv.tokens = MakeT(wrong.data(), DType::kF32, {1, kCtx});
    Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, kNgramTokens, nullptr, cv, 0),
        doctest::Contains("would ROUND them"), std::runtime_error);
  }
  SUBCASE("the table height must be the height the layout addresses") {
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    OwnedTensor short_table = table;
    short_table.shape[0] -= kTinyVocabDivisor;
    short_table.bytes.resize(static_cast<size_t>(short_table.shape[0] * kHd) * 4);
    Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table.bytes.empty() ? table : short_table, p, layout,
                            hidden, kNgramTokens, nullptr, cv, 0),
        doctest::Contains("is a different table"), std::runtime_error);
  }
  SUBCASE("an out-of-range token id is refused, because the mix would overflow int64") {
    // `std::invalid_argument` and not `std::runtime_error`: this refusal comes
    // from W2's `qwen4_exp::BuildNGramIds`, THROUGH the block, which is the
    // point of asserting it here — the block does not re-implement the bound.
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    std::vector<int64_t> bad(kNgramTokens, kNgramTokens + kNgramTotalLen);
    bad[2] = kTinyVocabSize;
    Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, bad.data(), nullptr, cv, 0),
        doctest::Contains("diverge in silence"), std::invalid_argument);
  }
  SUBCASE("state_row outside the caches") {
    Caches c;
    Qwen4ExpPleCaches cv = c.View();
    cv.state_row = 1;
    Tensor hidden = MakeT(h.data(), DType::kF32, {kNgramTotalLen, kW});
    CHECK_THROWS_WITH_AS(
        RunQwen4ExpPleBlock(d, w, table, p, layout, hidden, kNgramTokens, nullptr, cv, 0),
        doctest::Contains("state_row is outside"), std::runtime_error);
  }
}
