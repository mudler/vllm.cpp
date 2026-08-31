// `ENG-HYBRID-PLACEMENT` W4 (#2384) — the `fit` resolver's ARITHMETIC.
//
// A pure function, so every case here runs on any box with no device, no
// checkpoint and no loader. What it decides: how many trailing MoE layers must
// run their routed experts on the CPU for the rest of the model to fit the
// device budget.
//
// The cases are chosen for what fails SILENTLY, not for coverage of the happy
// path. An UNKNOWN budget in particular is a zero, and a resolver that compares
// against it naively places every layer on a box that was merely not measured —
// which looks like a working `--fit` and is a bug.
#include <doctest/doctest.h>

#include <cstddef>
#include <vector>

#include "vllm/model_executor/device_placement.h"

namespace {
constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;
}  // namespace

TEST_CASE("fit: a model that already fits places nothing") {
  // RESOLVED and zero layers is NOT the same as "did not resolve". The operator
  // asked, the resolver answered, and the answer was that nothing needs moving.
  const std::vector<size_t> per_layer(8, 1 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(/*footprint=*/10 * kGiB,
                                              /*budget=*/24 * kGiB, per_layer);
  CHECK(r.resolved);
  CHECK(r.placed_layers == 0);
  CHECK(r.placed_bytes == 0);
  CHECK_FALSE(r.still_exceeds);
}

TEST_CASE("fit: places the fewest TRAILING layers that bring it under budget") {
  // 8 layers x 2 GiB of experts inside a 20 GiB model against a 16 GiB budget:
  // 4 GiB must go, which is exactly two layers. Three would be more than asked.
  const std::vector<size_t> per_layer(8, 2 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(20 * kGiB, 16 * kGiB, per_layer);
  REQUIRE(r.resolved);
  CHECK(r.placed_layers == 2);
  CHECK(r.placed_bytes == 4 * kGiB);
  CHECK_FALSE(r.still_exceeds);
  CHECK(r.footprint_bytes == 20 * kGiB);
  CHECK(r.budget_bytes == 16 * kGiB);
}

TEST_CASE("fit: a partial saving still counts, so the boundary layer is included") {
  // 20.5 GiB against 20 GiB with 1 GiB layers: half a layer would do, but the
  // resolver places WHOLE layers, so it takes one. This is the coarser
  // granularity the spec sanctions, and the test pins it rather than leaving it
  // to be discovered from a memory figure.
  const std::vector<size_t> per_layer(4, 1 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(20 * kGiB + kGiB / 2, 20 * kGiB,
                                              per_layer);
  REQUIRE(r.resolved);
  CHECK(r.placed_layers == 1);
}

TEST_CASE("fit: when every layer is placed and it STILL does not fit, say so") {
  // Upstream also reduces context here; we do not. Placing everything and
  // reporting the shortfall is honest. Silently reporting success would send the
  // operator into an allocation failure with no clue why.
  const std::vector<size_t> per_layer(4, 1 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(100 * kGiB, 8 * kGiB, per_layer);
  CHECK(r.resolved);
  CHECK(r.placed_layers == 4);
  CHECK(r.still_exceeds);
}

TEST_CASE("fit: an UNKNOWN budget places NOTHING and says why") {
  // THE SILENT ONE. `ResidencyPolicy::device_memory_total_bytes` is 0 on every
  // platform that does not probe one. Compared naively, 0 makes everything
  // overflow and the resolver places every layer — on a box it never measured.
  const std::vector<size_t> per_layer(8, 2 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(20 * kGiB, /*budget=*/0, per_layer);
  CHECK_FALSE(r.resolved);
  CHECK(r.placed_layers == 0);
  CHECK_FALSE(r.reason.empty());
}

TEST_CASE("fit: an unknown FOOTPRINT places nothing too") {
  // The mirror of the case above: a file the footprint pass could not price.
  const std::vector<size_t> per_layer(8, 2 * kGiB);
  const auto r = vllm::ResolveMoeFitFromSizes(/*footprint=*/0, 16 * kGiB, per_layer);
  CHECK_FALSE(r.resolved);
  CHECK(r.placed_layers == 0);
  CHECK_FALSE(r.reason.empty());
}

TEST_CASE("fit: a model with no MoE layers places nothing, and does not divide by zero") {
  const auto r = vllm::ResolveMoeFitFromSizes(40 * kGiB, 16 * kGiB, {});
  CHECK_FALSE(r.resolved);
  CHECK(r.placed_layers == 0);
  CHECK_FALSE(r.reason.empty());

  // Dense layers among MoE ones contribute nothing and must not be counted as
  // placed: a plan that "placed" a dense layer would claim a saving of zero and
  // report a layer moved that did not move. The TRAILING entry here is dense, so
  // a resolver that walked back-to-front without skipping would report one
  // placed layer having freed nothing.
  const std::vector<size_t> mixed{2 * kGiB, 0, 2 * kGiB, 0};
  const auto m = vllm::ResolveMoeFitFromSizes(20 * kGiB, 18 * kGiB + kGiB / 2,
                                              mixed);
  REQUIRE(m.resolved);
  CHECK(m.placed_layers == 1);
  CHECK(m.placed_bytes == 2 * kGiB);

  // And the shortfall drives the COUNT: 3 GiB to free against 2 GiB layers needs
  // two of them, not one. Under-placing would leave the model over budget while
  // reporting success.
  const auto two = vllm::ResolveMoeFitFromSizes(20 * kGiB, 17 * kGiB, mixed);
  REQUIRE(two.resolved);
  CHECK(two.placed_layers == 2);
  CHECK(two.placed_bytes == 4 * kGiB);
}
