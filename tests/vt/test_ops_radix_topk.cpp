// vllm.cpp original (vt runtime). Unit tests for `include/vt/radix_topk.h` —
// the RADIX top-k selection core that `src/vt/cuda/cuda_sample.cu` runs for
// `vt::TopKValuesIndices` (SPEC-DFLASH2 W3 / D2 revisited,
// [#1867](https://github.com/mudler/vllm.cpp/issues/1867)).
//
// WHY THIS FILE EXISTS AT ALL, given that `test_ops_topk_values_indices.cpp`
// already gates the op. That file gates the OP, and on this host the op has only
// a CPU arm: there is no `nvcc` here and there has not been since W3, so its two
// device cases print `no CUDA backend; skipping` and the CUDA kernel's own
// arithmetic is gated nowhere a CPU-only gate can reach. #1867 replaces that
// kernel's search wholesale. Landing a new float-ordering rule, a new tie rule
// and a new k-th-largest search with NOTHING that runs them on the authoring
// host would put the entire change behind a GPU lease.
//
// So the change is split. `include/vt/radix_topk.h` holds every value the answer
// depends on and no parallelism; `cuda_sample.cu` holds the block-cooperative
// loops and no arithmetic. This file runs the first half here, today, including
// on #1867's own production shape (8 x 248320, K = 16). What it CANNOT run is
// the second half — the shared-memory histogram, the candidate compaction, the
// launch — and that is stated plainly rather than implied: see `## Owed` of
// .agents/specs/dflash2-spec-decode.md, which owes the device run to the same
// GPU lease that owes #1867's timing.
//
// WHAT IS GATED HERE.
//
//  * THE KEY IS MONOTONE, over the float bit patterns that actually break such
//    transforms: both zeros, both infinities, subnormals, the exponent
//    boundaries, and NaN of both signs. Monotonicity is the ONLY property that
//    makes a radix digit search over keys equal a value search, so it is checked
//    as an ordering over a sorted ladder rather than at a handful of points.
//  * THE TWO NORMALIZATIONS the CPU reference forces and FlashInfer's raw
//    transform does not have: `-0.0f` shares `+0.0f`'s key, and EVERY NaN maps
//    to the maximum. Each has its own case, and each case names the answer the
//    raw transform would have given, so it separates the normalization from the
//    transform instead of merely exercising it.
//  * THE ROUNDS FIND THE EXACT k-TH LARGEST. `SelectRow` below composes the
//    header's primitives the way the kernel does — four histogram rounds, then a
//    tie-broken collect — and every case compares it against `FullSortRow`, a
//    `std::stable_sort` of the whole row under the CONTRACT'S OWN comparator
//    written in FLOAT terms. That reference shares no line of code with the
//    header: it never forms a key, never forms a histogram and never picks a
//    bucket. It is the independent oracle, and it is the reason a defect in the
//    key would not simply agree with itself.
//  * THE ADVERSARIAL ROWS, which are where the two can disagree: exact ties,
//    duplicated maxima, an all-equal row, a row that is entirely -inf, ties
//    straddling the k-th boundary, k equal to the row width, and a row whose
//    distinct values number fewer than k.
//  * THE PRODUCTION SHAPE, 8 x 248320 at K = 16 — the shape #1857's kernel table
//    measured at 683 us/step — against the same full-sort oracle, and with a
//    deliberately TIE-DENSE variant, because the LCG rows in the op's own test
//    file were measured to contain no duplicate value at all and so cannot reach
//    the tie rule.
//  * THAT THE SEARCH IS BOUNDED BY FOUR ROUNDS. `SelectRow` counts them and the
//    production-shape case asserts the count. This is the whole of #1867: the
//    kernel it replaces bisected the threshold in float VALUE space under a
//    `kThreshMaxIter = 64` budget, and each of those iterations was a full pass
//    over the 248320-wide row. A future edit that reintroduces an unbounded or
//    data-dependent iteration count reds here rather than on a GPU lease.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vt/radix_topk.h"

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

float FromBits(uint32_t b) {
  float f = 0.0f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
}

// THE INDEPENDENT ORACLE. `src/vt/cpu/cpu_ops.cpp::TopKValuesIndicesKernel`'s
// comparator, restated in FLOAT terms and applied to the WHOLE row by a stable
// sort — no key, no histogram, no bucket. It answers the same question by a
// different route, which is what makes an agreement with `SelectRow` evidence.
std::vector<int64_t> FullSortRow(const std::vector<float>& row, int64_t k) {
  std::vector<int64_t> order(row.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int64_t>(i);
  std::stable_sort(order.begin(), order.end(), [&row](int64_t a, int64_t b) {
    const float x = row[static_cast<size_t>(a)], y = row[static_cast<size_t>(b)];
    const bool nx = std::isnan(x), ny = std::isnan(y);
    if (nx != ny) return nx;
    if (!nx && x != y) return x > y;
    return a < b;
  });
  order.resize(static_cast<size_t>(k));
  return order;
}

// THE COMPOSITION THE KERNEL RUNS, driven serially. Four rounds of histogram and
// `RadixTopKPickBucket` to fix the pivot key exactly, then the collect: every
// key strictly above the pivot, then the LOWEST-INDEXED keys equal to it until k
// slots are full, then the k ordered by `RadixTopKOutranks`.
//
// The kernel spreads the histogram loop across a block and the collect across a
// shared candidate buffer. Neither changes a value: histogram addition is
// commutative, and the collect's tie rule is stated by index, not by arrival.
// What this driver therefore gates is the arithmetic; what it does not gate is
// the parallel plumbing, and the file header says so.
struct RowResult {
  std::vector<int64_t> indices;
  std::vector<float> values;
  int rounds = 0;
};

RowResult SelectRow(const std::vector<float>& row, int64_t k, int64_t usable) {
  RowResult out;
  uint32_t prefix = 0;
  uint32_t remaining_k = static_cast<uint32_t>(k);
  for (int round = 0; round < vt::kRadixTopKRounds; ++round) {
    std::vector<uint32_t> hist(static_cast<size_t>(vt::kRadixTopKRadix), 0u);
    for (int64_t j = 0; j < usable; ++j) {
      const uint32_t key = vt::RadixTopKKey(row[static_cast<size_t>(j)]);
      if (!vt::RadixTopKPrefixMatches(key, prefix, round)) continue;
      ++hist[vt::RadixTopKBucket(key, round)];
    }
    uint32_t next_remaining = 0;
    const uint32_t bucket = vt::RadixTopKPickBucket(hist.data(), remaining_k, &next_remaining);
    prefix |= bucket << (32 - vt::kRadixTopKBits * (round + 1));
    remaining_k = next_remaining;
    ++out.rounds;
  }
  const uint32_t pivot = prefix;

  for (int64_t j = 0; j < usable; ++j) {
    if (vt::RadixTopKKey(row[static_cast<size_t>(j)]) > pivot) out.indices.push_back(j);
  }
  for (int64_t j = 0; j < usable && static_cast<int64_t>(out.indices.size()) < k; ++j) {
    if (vt::RadixTopKKey(row[static_cast<size_t>(j)]) == pivot) out.indices.push_back(j);
  }
  std::sort(out.indices.begin(), out.indices.end(), [&row](int64_t a, int64_t b) {
    return vt::RadixTopKOutranks(vt::RadixTopKKey(row[static_cast<size_t>(a)]), a,
                                 vt::RadixTopKKey(row[static_cast<size_t>(b)]), b);
  });
  for (int64_t idx : out.indices) out.values.push_back(row[static_cast<size_t>(idx)]);
  return out;
}

void CheckAgainstFullSort(const char* name, const std::vector<float>& row, int64_t k) {
  const RowResult got = SelectRow(row, k, static_cast<int64_t>(row.size()));
  const std::vector<int64_t> want = FullSortRow(row, k);
  REQUIRE(static_cast<int64_t>(got.indices.size()) == k);
  for (int64_t j = 0; j < k; ++j) {
    INFO("row \"", std::string(name), "\" slot ", j);
    CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    const float wv = row[static_cast<size_t>(want[static_cast<size_t>(j)])];
    const float gv = got.values[static_cast<size_t>(j)];
    if (std::isnan(wv)) CHECK(std::isnan(gv));
    else CHECK(gv == wv);
  }
}

// Deterministic LCG in [-2,2). Same generator the op's own test file uses, for
// the same reason: <random> diverges across standard libraries.
std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

}  // namespace

TEST_CASE("radix-topk: the key is MONOTONE over the float order") {
  // The single property the whole algorithm rests on. A radix digit search over
  // keys is a value search ONLY if the key preserves order, so this walks an
  // ASCENDING ladder of the bit patterns that break such transforms and asserts
  // the keys ascend with it. Checking a handful of pairs would miss a transform
  // that inverts one exponent octave; a ladder does not.
  const std::vector<float> ladder = {
      kNegInf,
      -3.4028235e38f,          // -FLT_MAX
      -1.0e30f,
      -1.0f,
      -1.1754944e-38f,         // -FLT_MIN, the smallest negative normal
      -1.4e-45f,               // a negative subnormal
      -0.0f,
      0.0f,
      1.4e-45f,                // a positive subnormal
      1.1754944e-38f,          // FLT_MIN
      1.0f,
      1.0000001f,              // one ulp above 1.0
      2.0f,                    // an exponent boundary
      1.0e30f,
      3.4028235e38f,           // FLT_MAX
      kPosInf,
  };
  for (size_t i = 1; i < ladder.size(); ++i) {
    const uint32_t lo = vt::RadixTopKKey(ladder[i - 1]);
    const uint32_t hi = vt::RadixTopKKey(ladder[i]);
    INFO("ladder step ", i, ": ", ladder[i - 1], " -> ", ladder[i]);
    // -0.0f and 0.0f are the one ADJACENT pair that must be EQUAL rather than
    // ascending; every other step is strict.
    if (ladder[i - 1] == ladder[i]) CHECK(lo == hi);
    else CHECK(lo < hi);
  }
  // And NaN is above the top of the ladder, which is +inf.
  CHECK(vt::RadixTopKKey(std::numeric_limits<float>::quiet_NaN()) > vt::RadixTopKKey(kPosInf));
}

TEST_CASE("radix-topk: NEGATIVE ZERO shares positive zero's key") {
  // The CPU reference's comparator reaches `-0.0f != 0.0f` as FALSE and falls
  // through to ascending index, so the two are EQUIVALENT there and must be
  // equivalent here. FlashInfer's raw transform is not: it would send +0.0f to
  // 0x80000000 and -0.0f to 0x7FFFFFFF, ranking the pair strictly. Asserting the
  // raw value the normalization removes is what separates this case from a
  // restatement of the code.
  CHECK(vt::RadixTopKKey(-0.0f) == vt::RadixTopKKey(0.0f));
  CHECK(vt::RadixTopKKey(0.0f) == 0x80000000u);
  // The unnormalized answer, spelled out: without the fixup, -0.0f's key is the
  // bitwise complement of 0x80000000.
  CHECK(0x7FFFFFFFu == ~0x80000000u);
  // And the ordering the pair therefore gets is by INDEX, either way round.
  const std::vector<float> zeros_neg_first = {-0.0f, 0.0f, -1.0f};
  const std::vector<float> zeros_pos_first = {0.0f, -0.0f, -1.0f};
  CheckAgainstFullSort("-0.0 then +0.0", zeros_neg_first, 2);
  CheckAgainstFullSort("+0.0 then -0.0", zeros_pos_first, 2);
}

TEST_CASE("radix-topk: EVERY NaN maps to the maximum key") {
  // `torch.topk(largest=True)` orders NaN first, whatever its sign or payload.
  // FlashInfer's raw transform does not: a positive NaN lands above +inf and a
  // NEGATIVE NaN lands below -inf, so half of them would sort last. Both signs
  // are asserted here for that reason, and a signalling payload with them.
  const uint32_t kMax = 0xFFFFFFFFu;
  CHECK(vt::RadixTopKKey(std::numeric_limits<float>::quiet_NaN()) == kMax);
  CHECK(vt::RadixTopKKey(-std::numeric_limits<float>::quiet_NaN()) == kMax);
  CHECK(vt::RadixTopKKey(std::numeric_limits<float>::signaling_NaN()) == kMax);
  CHECK(vt::RadixTopKKey(FromBits(0xFFC00000u)) == kMax);  // a negative quiet NaN
  CHECK(vt::RadixTopKKey(FromBits(0x7F800001u)) == kMax);  // the smallest NaN payload
  // The bit patterns either side of the NaN range are NOT NaN and must not be
  // caught: an all-ones exponent with a ZERO mantissa is an infinity.
  CHECK(vt::RadixTopKKey(FromBits(0x7F800000u)) != kMax);  // +inf
  CHECK(vt::RadixTopKKey(FromBits(0xFF800000u)) != kMax);  // -inf
  // NaN-first in a row, for both signs, against the full-sort oracle.
  const std::vector<float> row = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f,
                                  -std::numeric_limits<float>::quiet_NaN(), 2.0f};
  CheckAgainstFullSort("both NaN signs sort first", row, 4);
}

TEST_CASE("radix-topk: the ADVERSARIAL rows agree with a full sort") {
  // Every row here contains a tie, because a tie is the only thing that can
  // separate the radix search's answer from "something sorted". The oracle is
  // the full stable sort under the contract's float comparator.
  CheckAgainstFullSort("plain distinct", {0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f}, 3);
  CheckAgainstFullSort("ties inside the kept set", {5.0f, 5.0f, 1.0f, 5.0f}, 2);
  CheckAgainstFullSort("DUPLICATED MAXIMUM", {9.0f, 1.0f, 9.0f, 9.0f, 2.0f}, 2);
  // The case the search has to resolve: the k-th largest value is attained THREE
  // times for ONE remaining slot, so only the index rule decides.
  CheckAgainstFullSort("tie straddling the k-th boundary", {9.0f, 4.0f, 4.0f, 4.0f, 1.0f}, 2);
  CheckAgainstFullSort("tie group larger than k", {4.0f, 4.0f, 4.0f, 4.0f}, 3);
  // An ALL-EQUAL row: the answer is the index rule and nothing else, and the
  // pivot's tie group is the entire row.
  CheckAgainstFullSort("all equal", std::vector<float>(64, 1.25f), 16);
  CheckAgainstFullSort("all equal, k == width", std::vector<float>(16, -7.0f), 16);
  // A row that is ENTIRELY -inf, which is what a fully masked vocabulary looks
  // like, and a row with fewer FINITE entries than k.
  CheckAgainstFullSort("all -inf", std::vector<float>(32, kNegInf), 8);
  CheckAgainstFullSort("-inf saturated", {kNegInf, 5.0f, kNegInf, kNegInf}, 3);
  // Fewer DISTINCT values than k, which is not the same as fewer elements.
  CheckAgainstFullSort("two distinct values, k = 6", {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f}, 6);
  // Both infinities and both zeros in one row.
  CheckAgainstFullSort("infinities and zeros",
                       {kPosInf, -0.0f, kNegInf, 0.0f, kPosInf, kNegInf, 1.0f}, 5);
  // k == 1 and k == the row width, the two boundaries of the k range the op
  // admits (`src/vt/ops.cpp::TopKValuesIndices` refuses k outside [1, V - pad]).
  CheckAgainstFullSort("k == 1", {3.0f, 8.0f, 8.0f, 1.0f}, 1);
  CheckAgainstFullSort("k == width", {3.0f, 8.0f, 8.0f, 1.0f}, 4);
}

TEST_CASE("radix-topk: the USABLE window excludes the org-vocab padding tail") {
  // `num_org_vocab_padding` trailing columns can never contribute a candidate —
  // upstream's `logits[..., -num_pad:] = -inf`, done by restricting the search
  // rather than by writing to a read-only input. The row's two LARGEST values
  // sit inside the tail, so an implementation that ignored `usable` returns
  // them.
  const std::vector<float> row = {1.0f, 2.0f, 99.0f, 98.0f};
  const RowResult masked = SelectRow(row, /*k=*/2, /*usable=*/2);
  CHECK(masked.indices[0] == 1);
  CHECK(masked.indices[1] == 0);
  CHECK(masked.values[0] == 2.0f);
  CHECK(masked.values[1] == 1.0f);
  // The unmasked answer, asserted separately so the case above separates the
  // mask from the search instead of merely exercising the parameter.
  const RowResult unmasked = SelectRow(row, /*k=*/2, /*usable=*/4);
  CHECK(unmasked.indices[0] == 2);
  CHECK(unmasked.indices[1] == 3);
}

TEST_CASE("radix-topk: #1867's PRODUCTION shape, 8 x 248320 at K = 16") {
  // The shape #1857's kernel table measured `TopKValuesIndicesRowKernel` at
  // 683 us/step on: `num_reqs` 8 rows over the published DFlash2 checkpoint's
  // 248320-entry vocabulary, `selector_top_k` 16. This case does not measure
  // anything — the timing is owed to a GPU lease — it asserts that the search
  // ANSWERS this shape, and that it does so in four rounds.
  constexpr int64_t kRows = 8, kVocab = 248320, kK = 16;
  for (int64_t r = 0; r < kRows; ++r) {
    const std::vector<float> row =
        RandF32(static_cast<size_t>(kVocab), 0x9E3779B9u + static_cast<uint32_t>(r));
    const RowResult got = SelectRow(row, kK, kVocab);
    const std::vector<int64_t> want = FullSortRow(row, kK);
    // FOUR ROUNDS, always — this is #1867 itself. The kernel this replaces
    // bisected the threshold in float VALUE space under a `kThreshMaxIter = 64`
    // budget, each iteration a full pass over the row. A data-dependent or
    // unbounded count reds here.
    CHECK(got.rounds == vt::kRadixTopKRounds);
    CHECK(got.rounds == 4);
    for (int64_t j = 0; j < kK; ++j) {
      INFO("row ", r, " slot ", j);
      CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("radix-topk: the production shape with the vocabulary QUANTISED to ties") {
  // The LCG rows above hold no duplicate value at all — reproducing the
  // generator in exact float32 gives 248320 distinct values per row — so they
  // cannot reach the tie rule at this width, and the op's own test file records
  // the same measurement about its bulk rows. Rounding the same rows onto a
  // 512-value grid puts roughly 485 columns on every value, which makes the
  // k-th largest a tie group of hundreds and the answer entirely the index
  // rule. This is the production shape's adversarial twin.
  constexpr int64_t kRows = 4, kVocab = 248320, kK = 16;
  for (int64_t r = 0; r < kRows; ++r) {
    std::vector<float> row =
        RandF32(static_cast<size_t>(kVocab), 0x85EBCA6Bu + static_cast<uint32_t>(r));
    for (float& x : row) x = std::floor(x * 128.0f) / 128.0f;
    const RowResult got = SelectRow(row, kK, kVocab);
    const std::vector<int64_t> want = FullSortRow(row, kK);
    // The tie group at the k-th largest really is large: assert it rather than
    // assume it, because a case whose premise silently stopped holding is a case
    // that stopped testing the tie rule.
    const float kth = row[static_cast<size_t>(want[kK - 1])];
    int64_t at_kth = 0;
    for (float x : row) {
      if (x == kth) ++at_kth;
    }
    INFO("row ", r, ": ", at_kth, " columns attain the k-th largest value");
    CHECK(at_kth > kK);
    for (int64_t j = 0; j < kK; ++j) {
      INFO("row ", r, " slot ", j);
      CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("radix-topk: the bucket walk answers the k-th digit, not the largest") {
  // `RadixTopKPickBucket` directly, because the composed driver above can only
  // show it agreeing with a sort and not WHAT it returns. The histogram here is
  // hand-written: 3 candidates in bucket 200, 5 in bucket 100, 2 in bucket 7.
  std::vector<uint32_t> hist(static_cast<size_t>(vt::kRadixTopKRadix), 0u);
  hist[200] = 3;
  hist[100] = 5;
  hist[7] = 2;
  uint32_t rem = 0;
  // k = 1..3 are inside the top bucket, and none of them consumes it.
  CHECK(vt::RadixTopKPickBucket(hist.data(), 1, &rem) == 200u);
  CHECK(rem == 1u);
  CHECK(vt::RadixTopKPickBucket(hist.data(), 3, &rem) == 200u);
  CHECK(rem == 3u);
  // k = 4 crosses into bucket 100 and the remainder RESTARTS inside it: three
  // are already placed above, so one is still owed here.
  CHECK(vt::RadixTopKPickBucket(hist.data(), 4, &rem) == 100u);
  CHECK(rem == 1u);
  CHECK(vt::RadixTopKPickBucket(hist.data(), 8, &rem) == 100u);
  CHECK(rem == 5u);
  // k = 9 reaches the lowest bucket.
  CHECK(vt::RadixTopKPickBucket(hist.data(), 9, &rem) == 7u);
  CHECK(rem == 1u);
  CHECK(vt::RadixTopKPickBucket(hist.data(), 10, &rem) == 7u);
  CHECK(rem == 2u);
  // A histogram whose mass is entirely in bucket 0 — an all-equal row's last
  // round — still answers bucket 0 rather than walking off the array.
  std::vector<uint32_t> zero_bucket(static_cast<size_t>(vt::kRadixTopKRadix), 0u);
  zero_bucket[0] = 40;
  CHECK(vt::RadixTopKPickBucket(zero_bucket.data(), 16, &rem) == 0u);
  CHECK(rem == 16u);
}

TEST_CASE("radix-topk: the round digits RECONSTRUCT the key") {
  // The prefix the rounds build is the key itself, byte by byte, and the mask
  // that decides which candidates survive a round has to agree with the digit
  // that round reads. A mask off by one round admits the wrong candidate set and
  // still terminates, so this is checked directly rather than left to the
  // composed driver.
  const std::vector<float> probes = {0.0f, -0.0f, 1.0f, -1.0f, kPosInf, kNegInf, 3.5e-38f,
                                     -2.7182818f, 1.0e30f};
  for (float x : probes) {
    const uint32_t key = vt::RadixTopKKey(x);
    uint32_t rebuilt = 0;
    for (int round = 0; round < vt::kRadixTopKRounds; ++round) {
      // Before round r fixes its digit, the key matches the prefix built so far.
      INFO("value ", x, " round ", round);
      CHECK(vt::RadixTopKPrefixMatches(key, rebuilt, round));
      rebuilt |= vt::RadixTopKBucket(key, round) << (32 - vt::kRadixTopKBits * (round + 1));
    }
    INFO("value ", x);
    CHECK(rebuilt == key);
  }
  // Round 0's mask is empty, so EVERY key matches every prefix there — the
  // property that makes the first round histogram the whole row.
  CHECK(vt::RadixTopKPrefixMask(0) == 0u);
  CHECK(vt::RadixTopKPrefixMatches(0xDEADBEEFu, 0x12345678u, 0));
  // And the last round's mask has fixed all but the final byte.
  CHECK(vt::RadixTopKPrefixMask(vt::kRadixTopKRounds - 1) == 0xFFFFFF00u);
}

TEST_CASE("radix-topk: the ORDER predicate is descending key then ascending index") {
  // The tie rule, alone. Slot order is the one thing a value comparison cannot
  // check: two equal values are equal whichever index wins, so a wrong rule
  // moves the DFlash2 selector's candidate slots and moves acceptance without
  // raising anything.
  const uint32_t hi = vt::RadixTopKKey(5.0f), lo = vt::RadixTopKKey(1.0f);
  CHECK(vt::RadixTopKOutranks(hi, 9, lo, 0));       // value wins over index
  CHECK(!vt::RadixTopKOutranks(lo, 0, hi, 9));
  CHECK(vt::RadixTopKOutranks(hi, 3, hi, 4));       // equal value: LOWER index wins
  CHECK(!vt::RadixTopKOutranks(hi, 4, hi, 3));
  CHECK(!vt::RadixTopKOutranks(hi, 3, hi, 3));      // irreflexive
}

// --- THE BLOCK COMPOSITION, AND WHAT GATING IT CAN AND CANNOT MEAN ---------
//
// `SelectRow` above drives the header's primitives the way the ALGORITHM runs.
// It has no candidate buffer, so it cannot tell the kernel's three arms apart,
// and until #1929's review said so this file's strongest claim about the kernel
// came from a scratch harness that was never committed. `BlockSim` below is the
// committed replacement: a serial transcription of
// `src/vt/cuda/cuda_sample.cu::TopKValuesIndicesRadixRowKernel`, preserving
// every branch, guard, buffer bound and emission site, driven one "thread" at a
// time because the kernel's own composition is order-independent by
// construction — histogram addition is commutative and the collect's tie rule is
// stated by index rather than by arrival.
//
// WHAT THIS GATES. That the composition ANSWERS: that the three arms agree with
// a full stable sort of the same row, that the arm the kernel takes on a given
// row and cap is the one it claims, and that `RadixTopKCandCap` sizes the buffer
// the way the launcher needs. Deleting the overflow escape, mis-sizing the cap,
// dropping the second stage's winner emission, or letting the global arm
// re-emit what the second stage already emitted each red these cases.
//
// WHAT IT CANNOT GATE, stated because a transcription that claimed otherwise
// would be gating itself. It does not run the `.cu`. `atomicAdd` and `atomicOr`
// on shared memory, `__syncthreads`, `BlockRedMinI`, the `extern __shared__`
// layout and the launcher's `cudaFuncSetAttribute` opt-in have no host runner
// here and no `nvcc` to compile them; CI's `cuda-fat-build` compiles that file
// and runs nothing from it. A device run closes that and is owed with the
// timing: `## Owed` O34 of .agents/specs/dflash2-spec-decode.md.
namespace {

enum class SimArm { kCompact, kRestage, kGlobal };

struct SimResult {
  std::vector<int64_t> indices;
  std::vector<float> values;
  SimArm arm = SimArm::kGlobal;
  int cand_count = 0;
  int round0_bucket_pop = 0;
  int round1_bucket_pop = 0;
};

SimResult BlockSim(const std::vector<float>& row, int64_t usable, int k, int cand_cap) {
  const float* r = row.data();
  std::vector<uint32_t> s_hist(static_cast<size_t>(vt::kRadixTopKRadix), 0u);
  std::vector<uint32_t> s_cand_key(static_cast<size_t>(cand_cap));
  std::vector<int> s_cand_idx(static_cast<size_t>(cand_cap));
  std::vector<uint32_t> s_pairs(static_cast<size_t>(k), 0u);
  std::vector<int> s_pair_idx(static_cast<size_t>(k), 0);
  uint32_t s_prefix = 0u;
  uint32_t s_remaining = static_cast<uint32_t>(k);
  int s_cand_count = 0, s_overflow = 0, s_filled = 0;

  SimResult out;

  // ROUND 0.
  for (int64_t j = 0; j < usable; ++j) ++s_hist[vt::RadixTopKBucket(vt::RadixTopKKey(r[j]), 0)];
  uint32_t next = 0u;
  uint32_t bucket = vt::RadixTopKPickBucket(s_hist.data(), s_remaining, &next);
  const int top_bucket = static_cast<int>(bucket);
  out.round0_bucket_pop = static_cast<int>(s_hist[bucket]);
  s_prefix = bucket << (32 - vt::kRadixTopKBits);
  s_remaining = next;
  std::fill(s_hist.begin(), s_hist.end(), 0u);

  // ROUND 1: outright winners, candidate compaction, round-1 histogram.
  for (int64_t j = 0; j < usable; ++j) {
    const uint32_t key = vt::RadixTopKKey(r[j]);
    const int b = static_cast<int>(vt::RadixTopKBucket(key, 0));
    if (b > top_bucket) {
      const int slot = s_filled++;
      if (slot < k) {
        s_pairs[static_cast<size_t>(slot)] = key;
        s_pair_idx[static_cast<size_t>(slot)] = static_cast<int>(j);
      }
    } else if (b == top_bucket) {
      ++s_hist[vt::RadixTopKBucket(key, 1)];
      const int slot = s_cand_count++;
      if (slot < cand_cap) {
        s_cand_key[static_cast<size_t>(slot)] = key;
        s_cand_idx[static_cast<size_t>(slot)] = static_cast<int>(j);
      } else {
        s_overflow = 1;
      }
    }
  }
  next = 0u;
  bucket = vt::RadixTopKPickBucket(s_hist.data(), s_remaining, &next);
  const int top_bucket1 = static_cast<int>(bucket);
  out.round1_bucket_pop = static_cast<int>(s_hist[bucket]);
  s_prefix |= bucket << (32 - vt::kRadixTopKBits * 2);
  s_remaining = next;
  // The re-stage decision, taken from the histogram the pass above already
  // built, so a second stage that runs is a second stage that fits.
  const bool restage = s_overflow != 0 && s_hist[bucket] <= static_cast<uint32_t>(cand_cap);

  // STAGE 2.
  if (restage) {
    s_cand_count = 0;
    for (int64_t j = 0; j < usable; ++j) {
      const uint32_t key = vt::RadixTopKKey(r[j]);
      if (static_cast<int>(vt::RadixTopKBucket(key, 0)) != top_bucket) continue;
      const int b1 = static_cast<int>(vt::RadixTopKBucket(key, 1));
      if (b1 > top_bucket1) {
        const int slot = s_filled++;
        if (slot < k) {
          s_pairs[static_cast<size_t>(slot)] = key;
          s_pair_idx[static_cast<size_t>(slot)] = static_cast<int>(j);
        }
      } else if (b1 == top_bucket1) {
        const int slot = s_cand_count++;
        // Unreachable: the re-stage decision above already proved this fits.
        // The bound is transcribed because the kernel carries it, and the kernel
        // carries it because the alternative is a silent shared-memory
        // out-of-bounds write on a device.
        if (slot < cand_cap) {
          s_cand_key[static_cast<size_t>(slot)] = key;
          s_cand_idx[static_cast<size_t>(slot)] = static_cast<int>(j);
        }
      }
    }
  }
  const bool compacted = s_overflow == 0 || restage;
  const int cand_count = s_cand_count < cand_cap ? s_cand_count : cand_cap;
  out.arm = compacted ? (restage ? SimArm::kRestage : SimArm::kCompact) : SimArm::kGlobal;
  out.cand_count = cand_count;

  // ROUNDS 2 and 3.
  for (int round = 2; round < vt::kRadixTopKRounds; ++round) {
    std::fill(s_hist.begin(), s_hist.end(), 0u);
    const uint32_t prefix = s_prefix;
    if (compacted) {
      for (int i = 0; i < cand_count; ++i) {
        const uint32_t key = s_cand_key[static_cast<size_t>(i)];
        if (vt::RadixTopKPrefixMatches(key, prefix, round))
          ++s_hist[vt::RadixTopKBucket(key, round)];
      }
    } else {
      for (int64_t j = 0; j < usable; ++j) {
        const uint32_t key = vt::RadixTopKKey(r[j]);
        if (vt::RadixTopKPrefixMatches(key, prefix, round))
          ++s_hist[vt::RadixTopKBucket(key, round)];
      }
    }
    next = 0u;
    bucket = vt::RadixTopKPickBucket(s_hist.data(), s_remaining, &next);
    s_prefix |= bucket << (32 - vt::kRadixTopKBits * (round + 1));
    s_remaining = next;
  }
  const uint32_t pivot = s_prefix;

  // WINNERS STRICTLY ABOVE THE PIVOT.
  if (compacted) {
    for (int i = 0; i < cand_count; ++i) {
      if (s_cand_key[static_cast<size_t>(i)] > pivot) {
        const int slot = s_filled++;
        if (slot < k) {
          s_pairs[static_cast<size_t>(slot)] = s_cand_key[static_cast<size_t>(i)];
          s_pair_idx[static_cast<size_t>(slot)] = s_cand_idx[static_cast<size_t>(i)];
        }
      }
    }
  } else {
    for (int64_t j = 0; j < usable; ++j) {
      const uint32_t key = vt::RadixTopKKey(r[j]);
      if (key > pivot && static_cast<int>(vt::RadixTopKBucket(key, 0)) == top_bucket) {
        const int slot = s_filled++;
        if (slot < k) {
          s_pairs[static_cast<size_t>(slot)] = key;
          s_pair_idx[static_cast<size_t>(slot)] = static_cast<int>(j);
        }
      }
    }
  }
  int filled = s_filled < k ? s_filled : k;

  // THE TIE FILL, one block-wide min-index pass per remaining slot.
  int taken = -1;
  while (filled < k) {
    int pick = INT32_MAX;
    if (compacted) {
      for (int i = 0; i < cand_count; ++i) {
        if (s_cand_key[static_cast<size_t>(i)] == pivot &&
            s_cand_idx[static_cast<size_t>(i)] > taken)
          pick = std::min(pick, s_cand_idx[static_cast<size_t>(i)]);
      }
    } else {
      for (int64_t j = 0; j < usable; ++j) {
        if (vt::RadixTopKKey(r[j]) == pivot && static_cast<int>(j) > taken) {
          pick = static_cast<int>(j);
          break;
        }
      }
    }
    if (pick == INT32_MAX) break;
    s_pairs[static_cast<size_t>(filled)] = pivot;
    s_pair_idx[static_cast<size_t>(filled)] = pick;
    taken = pick;
    ++filled;
  }

  // THE ORDERING, by the kernel's own selection sort.
  for (int a = 0; a < filled; ++a) {
    int best = a;
    for (int b = a + 1; b < filled; ++b) {
      if (vt::RadixTopKOutranks(s_pairs[static_cast<size_t>(b)], s_pair_idx[static_cast<size_t>(b)],
                                s_pairs[static_cast<size_t>(best)],
                                s_pair_idx[static_cast<size_t>(best)]))
        best = b;
    }
    std::swap(s_pairs[static_cast<size_t>(a)], s_pairs[static_cast<size_t>(best)]);
    std::swap(s_pair_idx[static_cast<size_t>(a)], s_pair_idx[static_cast<size_t>(best)]);
  }

  out.indices.resize(static_cast<size_t>(k));
  out.values.resize(static_cast<size_t>(k));
  for (int j = 0; j < k; ++j) {
    const int idx = j < filled ? s_pair_idx[static_cast<size_t>(j)] : 0;
    out.indices[static_cast<size_t>(j)] = j < filled ? static_cast<int64_t>(idx) : 0;
    out.values[static_cast<size_t>(j)] = j < filled ? r[static_cast<size_t>(idx)] : kNegInf;
  }
  return out;
}

const char* ArmName(SimArm a) {
  return a == SimArm::kCompact ? "compact" : (a == SimArm::kRestage ? "restage" : "global");
}

// Every arm must answer what the full sort answers. `want_arm` is asserted too,
// because a case that only checked the ANSWER would stay green if the cap
// silently sent every row down the global fallback.
void CheckSim(const char* name, const std::vector<float>& row, int64_t usable, int k, int cand_cap,
              SimArm want_arm) {
  const SimResult got = BlockSim(row, usable, k, cand_cap);
  const std::vector<int64_t> want = FullSortRow(
      std::vector<float>(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(usable)), k);
  INFO("row \"", std::string(name), "\" cap ", cand_cap, " arm ", std::string(ArmName(got.arm)),
       " want ", std::string(ArmName(want_arm)));
  CHECK(got.arm == want_arm);
  for (int j = 0; j < k; ++j) {
    INFO("slot ", j);
    CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    const float wv = row[static_cast<size_t>(want[static_cast<size_t>(j)])];
    const float gv = got.values[static_cast<size_t>(j)];
    if (std::isnan(wv)) CHECK(std::isnan(gv));
    else CHECK(gv == wv);
  }
}

}  // namespace

TEST_CASE("radix-topk: the candidate cap is sized from the DEVICE's shared budget") {
  // The launcher's arithmetic, gated where the launcher itself cannot be. Both
  // the k winner pairs and the candidate buffer come out of one dynamic
  // allocation, so the cap is what is left after the pairs.
  constexpr uint32_t kPair = static_cast<uint32_t>(vt::kRadixTopKCandBytes);
  CHECK(kPair == 8u);
  // Nothing left over is a legal answer: the kernel's global arm needs no
  // candidate buffer at all.
  CHECK(vt::RadixTopKCandCap(0u, 16) == 0);
  CHECK(vt::RadixTopKCandCap(16u * kPair, 16) == 0);
  CHECK(vt::RadixTopKCandCap(17u * kPair, 16) == 1);
  CHECK(vt::RadixTopKCandCap(2064u * kPair, 16) == 2048);
  // The 48 KB a block gets WITHOUT an opt-in, less about 2 KB of static shared.
  CHECK(vt::RadixTopKCandCap(46u * 1024u, 16) == 5872);
  // And the ceiling is FlashInfer's `FILTERED_TOPK_SMEM_INPUT_SIZE`, which is
  // what makes the buffer 128 KB and therefore dynamic rather than static.
  CHECK(vt::kRadixTopKCandCapMax == 16 * 1024);
  CHECK(vt::RadixTopKCandCap(0xFFFFFFF0u, 16) == vt::kRadixTopKCandCapMax);
  CHECK(vt::RadixTopKDynamicSmemBytes(vt::kRadixTopKCandCapMax, 16) == 131200u);
  CHECK(vt::RadixTopKDynamicSmemBytes(0, 16) == 128u);
  // The round trip the launcher relies on: whatever cap a budget yields, the
  // allocation that cap asks for fits back inside that budget. The k winner
  // pairs are NOT optional, so the floor is their own 128 bytes and a budget
  // below that is a device this kernel cannot run on at all rather than a cap
  // to round down.
  for (uint32_t budget : {128u, 129u, 4096u, 46u * 1024u, 99u * 1024u, 227u * 1024u}) {
    const int cap = vt::RadixTopKCandCap(budget, 16);
    INFO("budget ", budget, " cap ", cap);
    CHECK(vt::RadixTopKDynamicSmemBytes(cap, 16) <= budget);
  }
  CHECK(vt::RadixTopKDynamicSmemBytes(vt::RadixTopKCandCap(100u, 16), 16) == 128u);
}

TEST_CASE("radix-topk: the block composition answers on EVERY arm") {
  const float qnan = std::numeric_limits<float>::quiet_NaN();

  // The literal rows, at a cap wide enough that the first stage compacts.
  CheckSim("plain distinct", {0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f}, 6, 3, 64, SimArm::kCompact);
  CheckSim("ties inside kept set", {5.0f, 5.0f, 1.0f, 5.0f}, 4, 2, 64, SimArm::kCompact);
  CheckSim("tie straddling the boundary", {9.0f, 4.0f, 4.0f, 4.0f, 1.0f}, 5, 2, 64,
           SimArm::kCompact);
  CheckSim("tie group larger than k", {4.0f, 4.0f, 4.0f, 4.0f}, 4, 3, 64, SimArm::kCompact);
  CheckSim("both zeros and both infinities",
           {kPosInf, -0.0f, kNegInf, 0.0f, kPosInf, kNegInf, 1.0f}, 7, 5, 64, SimArm::kCompact);
  CheckSim("NaN of both signs", {1.0f, qnan, 3.0f, -qnan, 2.0f}, 5, 4, 64, SimArm::kCompact);
  CheckSim("k == 1", {3.0f, 8.0f, 8.0f, 1.0f}, 4, 1, 64, SimArm::kCompact);
  CheckSim("k == usable", {3.0f, 8.0f, 8.0f, 1.0f}, 4, 4, 64, SimArm::kCompact);
  // The org-vocab padding tail, which `usable` excludes: a masked column must
  // not reach the candidate buffer on any arm.
  CheckSim("padding tail masked", {1.0f, 2.0f, 99.0f, 98.0f}, 2, 2, 64, SimArm::kCompact);

  // A CAP OF ZERO drives every row down the global fallback, and the answers do
  // not move. This is the arm a device that refuses the shared-memory opt-in
  // takes, and it is why a refusal is handled rather than raised.
  CheckSim("plain distinct, no buffer", {0.5f, -1.0f, 3.0f, 2.0f, 7.0f, 0.0f}, 6, 3, 0,
           SimArm::kGlobal);
  CheckSim("ties inside kept set, no buffer", {5.0f, 5.0f, 1.0f, 5.0f}, 4, 2, 0, SimArm::kGlobal);
  CheckSim("NaN of both signs, no buffer", {1.0f, qnan, 3.0f, -qnan, 2.0f}, 5, 4, 0,
           SimArm::kGlobal);
  CheckSim("padding tail masked, no buffer", {1.0f, 2.0f, 99.0f, 98.0f}, 2, 2, 0, SimArm::kGlobal);

  // ROWS THAT DEFEAT BOTH STAGES. Every column equal puts the whole row in the
  // round-0 bucket AND in the round-1 bucket, so no cap short of the row width
  // saves it and the kernel re-reads global for rounds 2, 3, the winner collect
  // and the tie fill. The answer is still exact, which is the point of keeping
  // the escape.
  CheckSim("every column equal", std::vector<float>(4096, 1.25f), 4096, 16, 2048, SimArm::kGlobal);
  CheckSim("every column -inf", std::vector<float>(4096, kNegInf), 4096, 16, 2048,
           SimArm::kGlobal);
  // ... and the same row compacts when the cap DOES cover it.
  CheckSim("every column equal, cap covers", std::vector<float>(4096, 1.25f), 4096, 16, 4096,
           SimArm::kCompact);
}

TEST_CASE("radix-topk: the cap BOUNDARY selects the arm, and neither arm moves the answer") {
  // Rows built so that exactly `n` columns share the round-0 bucket, walked
  // across the cap. This is the case that reds if the cap changes value or if
  // the overflow escape stops firing: at n = cap the first stage still fits and
  // at n = cap + 1 it does not.
  //
  // WHICH ARM the overflow then takes is decided by the ROUND-1 population, and
  // both possibilities are covered here rather than one, because they are what
  // separates a three-pass row from a six-pass one.
  constexpr int kCap = 2048;
  for (int n : {kCap - 1, kCap, kCap + 1, 2 * kCap}) {
    // SPREAD across the round-1 buckets: n values filling the octave [1, 2), so
    // one round-0 bucket holds all n and the round-1 buckets hold about n/256.
    std::vector<float> spread(static_cast<size_t>(n) + 32, -1000.0f);
    for (int i = 0; i < n; ++i)
      spread[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i) / static_cast<float>(n);
    const std::string sname = "cap boundary, spread, n=" + std::to_string(n);
    CheckSim(sname.c_str(), spread, static_cast<int64_t>(spread.size()), 16, kCap,
             n <= kCap ? SimArm::kCompact : SimArm::kRestage);

    // PILED into one round-1 bucket: n values within a few ulps of 1.0f, so the
    // second stage would not narrow anything and the kernel does not pay for it.
    std::vector<float> piled(static_cast<size_t>(n) + 32, -1000.0f);
    for (int i = 0; i < n; ++i)
      piled[static_cast<size_t>(i)] = 1.0f + static_cast<float>(i % 7) * 1e-6f;
    const std::string pname = "cap boundary, piled, n=" + std::to_string(n);
    CheckSim(pname.c_str(), piled, static_cast<int64_t>(piled.size()), 16, kCap,
             n <= kCap ? SimArm::kCompact : SimArm::kGlobal);
  }
}

TEST_CASE("radix-topk: the PRODUCTION shape needs the second compaction stage") {
  // #1867's own shape, through the composition rather than only through the
  // arithmetic. Round 0's digit is the sign plus the top SEVEN exponent bits, so
  // one bucket spans two exponents and holds about 93000 of these 248320
  // columns -- 45x the 2048 the kernel first sized and 5.7x
  // `kRadixTopKCandCapMax`, so NO shared buffer can hold it. One round later the
  // same rows hold about 490. Both facts are asserted, because they are the
  // reason the kernel has a second stage at all.
  constexpr int64_t kRows = 8, kVocab = 248320, kK = 16;
  for (int64_t r = 0; r < kRows; ++r) {
    const std::vector<float> row =
        RandF32(static_cast<size_t>(kVocab), 0x9E3779B9u + static_cast<uint32_t>(r));
    const SimResult got = BlockSim(row, kVocab, kK, /*cand_cap=*/2048);
    INFO("row ", r, " round0 ", got.round0_bucket_pop, " round1 ", got.round1_bucket_pop);
    CHECK(got.round0_bucket_pop > 90000);
    CHECK(got.round0_bucket_pop > vt::kRadixTopKCandCapMax);
    CHECK(got.round1_bucket_pop < 1024);
    CHECK(got.arm == SimArm::kRestage);
    CHECK(got.cand_count == got.round1_bucket_pop);
    const std::vector<int64_t> want = FullSortRow(row, kK);
    for (int64_t j = 0; j < kK; ++j) {
      INFO("row ", r, " slot ", j);
      CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("radix-topk: the tie-dense production twin, through the composition") {
  // The same width with the values rounded onto a 512-value grid, so the k-th
  // largest is a tie group of hundreds and the answer is entirely the index
  // rule. It takes the same second stage.
  constexpr int64_t kRows = 4, kVocab = 248320, kK = 16;
  for (int64_t r = 0; r < kRows; ++r) {
    std::vector<float> row =
        RandF32(static_cast<size_t>(kVocab), 0x85EBCA6Bu + static_cast<uint32_t>(r));
    for (float& x : row) x = std::floor(x * 128.0f) / 128.0f;
    const SimResult got = BlockSim(row, kVocab, kK, /*cand_cap=*/2048);
    INFO("row ", r, " round0 ", got.round0_bucket_pop, " round1 ", got.round1_bucket_pop);
    CHECK(got.round0_bucket_pop > 90000);
    CHECK(got.arm == SimArm::kRestage);
    const std::vector<int64_t> want = FullSortRow(row, kK);
    for (int64_t j = 0; j < kK; ++j) {
      INFO("row ", r, " slot ", j);
      CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
    }
  }
}

TEST_CASE("radix-topk: randomized tie-dense fuzz over all three arms") {
  // Small widths, few distinct values, and both zeros, -inf and NaN mixed in, at
  // three caps so that each row is answered by the compact, the re-staged and
  // the global arm in turn. Every one of the three must give the full sort's
  // answer, which is what makes the escape and the second stage interchangeable
  // in RESULT and not only in intent.
  //
  // The ordinary values are drawn from inside ONE octave, [1, 2). That is what
  // makes the re-staged arm reachable at all: distinct values in different
  // octaves land in different round-0 buckets, so an overflow of the round-0
  // bucket would then be a single repeated value whose round-1 bucket is exactly
  // as wide. Values sharing an octave share the round-0 digit and differ in the
  // round-1 one, which is the shape the second stage exists for.
  uint32_t s = 0xC0FFEEu;
  auto next = [&s]() { s = s * 1664525u + 1013904223u; return s >> 8; };
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float kOctave[5] = {1.0f, 1.125f, 1.25f, 1.5f, 1.75f};
  int mismatches = 0, arms_seen = 0;
  bool saw[3] = {false, false, false};
  for (int t = 0; t < 3000; ++t) {
    const int n = 1 + static_cast<int>(next() % 48u);
    const int k = 1 + static_cast<int>(next() % static_cast<uint32_t>(n));
    const int levels = 1 + static_cast<int>(next() % 5u);
    std::vector<float> row(static_cast<size_t>(n));
    for (float& x : row) {
      const uint32_t z = next() % static_cast<uint32_t>(levels + 3);
      if (z == static_cast<uint32_t>(levels)) x = kNegInf;
      else if (z == static_cast<uint32_t>(levels) + 1u) x = (next() % 2u) ? 0.0f : -0.0f;
      else if (z == static_cast<uint32_t>(levels) + 2u) x = qnan;
      else x = kOctave[z];
    }
    const std::vector<int64_t> want = FullSortRow(row, k);
    for (int cap : {0, 4, n}) {
      const SimResult got = BlockSim(row, n, k, cap);
      saw[static_cast<int>(got.arm)] = true;
      for (int j = 0; j < k; ++j) {
        if (got.indices[static_cast<size_t>(j)] != want[static_cast<size_t>(j)]) {
          INFO("t ", t, " n ", n, " k ", k, " cap ", cap, " slot ", j, " arm ",
               std::string(ArmName(got.arm)));
          CHECK(got.indices[static_cast<size_t>(j)] == want[static_cast<size_t>(j)]);
          ++mismatches;
          break;
        }
      }
    }
  }
  CHECK(mismatches == 0);
  for (bool b : saw) arms_seen += b ? 1 : 0;
  // The fuzz is worthless if it only ever reached one arm, so it REPORTS which
  // it reached, in its own output and in words, rather than leaving the reader
  // to assume that "3000 rows passed" means the second stage ever ran.
  // `std::string`, not a string literal: doctest's stringifier renders a bare
  // `const char*` as a BOOL, which would print `1` and say nothing.
  MESSAGE("fuzz arms reached -- compact: " << std::string(saw[0] ? "yes" : "NO")
          << ", restage: " << std::string(saw[1] ? "yes" : "NO") << ", global: "
          << std::string(saw[2] ? "yes" : "NO"));
  CHECK(arms_seen == 3);
}
