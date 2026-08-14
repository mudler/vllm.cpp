// The DiT's U-Net skip routing. See dit_skip.h.
//
// The expected schedules below were RECORDED from upstream's own Transformer by
// `scripts/gen-dit-skip-schedule.py`, not derived from reading the formula:
//
//   depth= 2  emit=[0]              receive=[]            pairs=[]
//   depth= 3  emit=[0]              receive=[2]           pairs=[(2,0)]
//   depth= 4  emit=[0,1]            receive=[3]           pairs=[(3,1)]
//   depth= 5  emit=[0,1]            receive=[3,4]         pairs=[(3,1),(4,0)]
//   depth=12  emit=[0..5]           receive=[7..11]       pairs=[(7,5)..(11,1)]
//   depth=13  emit=[0..5]           receive=[7..12]       pairs=[(7,5)..(12,0)]
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_skip.h"

using vllm::models::dit_skip::ApplySkip;
using vllm::models::dit_skip::Plan;

TEST_CASE("the SHIPPED depth 13 routes exactly as upstream does") {
  const auto s = Plan(13);
  CHECK(s.emit == std::vector<int64_t>{0, 1, 2, 3, 4, 5});
  CHECK(s.receive == std::vector<int64_t>{7, 8, 9, 10, 11, 12});
  // LIFO: the LAST emitter feeds the FIRST receiver.
  CHECK(s.source[7] == 5);
  CHECK(s.source[8] == 4);
  CHECK(s.source[9] == 3);
  CHECK(s.source[10] == 2);
  CHECK(s.source[11] == 1);
  CHECK(s.source[12] == 0);
  // The middle layer is on neither list.
  CHECK(s.source[6] == -1);
  for (int64_t i = 0; i <= 6; ++i) {
    CHECK(s.source[i] == -1);
  }
  CHECK(s.orphaned == 0);
}

TEST_CASE("an EVEN depth leaves one skip unconsumed, and we do not hide it") {
  const auto s = Plan(12);
  CHECK(s.emit == std::vector<int64_t>{0, 1, 2, 3, 4, 5});
  CHECK(s.receive == std::vector<int64_t>{7, 8, 9, 10, 11});
  CHECK(s.source[7] == 5);
  CHECK(s.source[11] == 1);
  // Six emitters, five receivers: layer 0's output is pushed and never popped.
  CHECK(s.orphaned == 1);
}

TEST_CASE("small depths match the recorded upstream schedules") {
  const auto d2 = Plan(2);
  CHECK(d2.emit == std::vector<int64_t>{0});
  CHECK(d2.receive.empty());
  CHECK(d2.orphaned == 1);

  const auto d3 = Plan(3);
  CHECK(d3.emit == std::vector<int64_t>{0});
  CHECK(d3.receive == std::vector<int64_t>{2});
  CHECK(d3.source[2] == 0);
  CHECK(d3.orphaned == 0);

  const auto d4 = Plan(4);
  CHECK(d4.emit == std::vector<int64_t>{0, 1});
  CHECK(d4.receive == std::vector<int64_t>{3});
  CHECK(d4.source[3] == 1);
  CHECK(d4.orphaned == 1);

  const auto d5 = Plan(5);
  CHECK(d5.emit == std::vector<int64_t>{0, 1});
  CHECK(d5.receive == std::vector<int64_t>{3, 4});
  CHECK(d5.source[3] == 1);
  CHECK(d5.source[4] == 0);
  CHECK(d5.orphaned == 0);
}

TEST_CASE("a single layer neither emits nor receives") {
  const auto s = Plan(1);
  CHECK(s.emit.empty());
  CHECK(s.receive.empty());
  CHECK(s.source == std::vector<int64_t>{-1});
  CHECK(s.orphaned == 0);
}

TEST_CASE("skip_in_linear concatenates x FIRST, then the skip") {
  // dim 2, one frame. The weight reads the two halves with distinguishable
  // multipliers, so a reversed concat gives a different, still-finite answer.
  //   cat = [x0, x1, s0, s1] = [1, 2, 10, 20]
  //   row0 = [1, 0, 0, 0] . cat + 100 = 101      (sees x0)
  //   row1 = [0, 0, 1, 0] . cat + 200 = 210      (sees s0)
  const std::vector<float> x{1.0F, 2.0F};
  const std::vector<float> skip{10.0F, 20.0F};
  const std::vector<float> weight{1.0F, 0.0F, 0.0F, 0.0F,
                                  0.0F, 0.0F, 1.0F, 0.0F};
  const std::vector<float> bias{100.0F, 200.0F};

  const std::vector<float> got = ApplySkip(x, skip, 1, 2, weight, bias);
  REQUIRE(got.size() == 2);
  CHECK(got[0] == 101.0F);
  CHECK(got[1] == 210.0F);

  // Reversing the arguments must NOT give the same answer, or the order is not
  // actually being honoured.
  const std::vector<float> swapped = ApplySkip(skip, x, 1, 2, weight, bias);
  CHECK(swapped[0] == 110.0F);
  CHECK(swapped[1] == 201.0F);
}

TEST_CASE("skip_in_linear runs per frame") {
  const std::vector<float> x{1.0F, 0.0F, 3.0F, 0.0F};     // 2 frames, dim 2
  const std::vector<float> skip{0.0F, 5.0F, 0.0F, 7.0F};
  const std::vector<float> weight{1.0F, 0.0F, 0.0F, 1.0F,
                                  0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<float> bias{0.0F, 0.0F};

  const std::vector<float> got = ApplySkip(x, skip, 2, 2, weight, bias);
  REQUIRE(got.size() == 4);
  CHECK(got[0] == 6.0F);  // x0 + s1 = 1 + 5
  CHECK(got[1] == 0.0F);
  CHECK(got[2] == 10.0F);  // 3 + 7
  CHECK(got[3] == 0.0F);
}
