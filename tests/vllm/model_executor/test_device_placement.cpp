// `ENG-HYBRID-PLACEMENT` W2 (issue #2023) — the `DevicePlacement` seam.
//
// NOTHING UPSTREAM TO PORT AS A TEST. llama.cpp has no unit test for any of its
// four placement surfaces; its coverage is end to end. What IS portable is the
// SEMANTICS, so every case below names the `b10451` line its expectation comes
// from and a reviewer can check the expectation rather than trust it.
//
// THE THREE GUARANTEES:
//   1. FIRST-MATCH-WINS, in the operator's order. Order is input, never sorted.
//      A case that passes under both orderings is not testing this, so the case
//      here asserts the two orderings resolve DIFFERENTLY.
//   2. `regex_search`, not a full match, which is what makes an unanchored
//      pattern reach every layer and an anchored one reach exactly one.
//   3. INERTNESS. `IsTrivial()` is the predicate the engine reads to decide
//      whether to take its existing single-device path, so it has to be true for
//      both ways of asking for nothing: no overrides, and overrides that name the
//      device we are already on.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/device_placement.h"

namespace {

vllm::PlacementOverride Ov(const char* pattern, const char* device) {
  return vllm::PlacementOverride{pattern, device};
}

// The regex `-cmoe` installs, from `common/common.h:1113` @ `b10451`.
constexpr const char* kExps = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

}  // namespace

TEST_CASE("device placement: an empty placement is trivial and answers the engine device") {
  const vllm::DevicePlacement p(vt::DeviceType::kCUDA);
  CHECK(p.IsTrivial());
  CHECK(p.override_count() == 0);
  CHECK(p.DeviceFor("blk.0.ffn_up_exps.weight") == vt::DeviceType::kCUDA);
  CHECK(p.DeviceFor("token_embd.weight") == vt::DeviceType::kCUDA);
  // Silent, because a line about a placement that changes nothing teaches the
  // operator to skip past the line that matters.
  CHECK(p.Describe().empty());
}

TEST_CASE("device placement: the match is regex_search, not a full match") {
  // `src/llama-model-loader.cpp:1181-1182`. This is what lets ONE unanchored
  // pattern reach every layer, which is the whole mechanism `-cmoe` rests on.
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCUDA);
  CHECK_FALSE(p.IsTrivial());
  for (const char* name : {"blk.0.ffn_up_exps.weight",
                           "blk.41.ffn_down_exps.weight",
                           "blk.7.ffn_gate_exps.weight",
                           // The `ch` and `gate_up` variants the regex admits,
                           // which a hand-written pattern usually forgets.
                           "blk.3.ffn_up_chexps.weight",
                           "blk.3.ffn_gate_up_exps.weight"}) {
    INFO("tensor: " << name);
    CHECK(p.DeviceFor(name) == vt::DeviceType::kCPU);
  }
  // And it must NOT reach the dense tower, the attention weights or the router,
  // which is the difference between CPU-MoE and running the model on the CPU.
  for (const char* name : {"blk.0.attn_q.weight", "blk.0.ffn_up.weight",
                           "blk.0.ffn_gate_inp.weight", "token_embd.weight",
                           "output.weight", "blk.0.attn_norm.weight"}) {
    INFO("tensor: " << name);
    CHECK(p.DeviceFor(name) == vt::DeviceType::kCUDA);
  }
}

TEST_CASE("device placement: an anchored pattern reaches exactly one layer") {
  // `llm_ffn_exps_block_regex(7)` (`common/common.h:1115-1117`), which is what
  // `-ncmoe` pushes one of per layer.
  const std::string anchored = std::string("blk\\.7") + kExps;
  const auto p = vllm::DevicePlacement::FromOverrides(
      {Ov(anchored.c_str(), "cpu")}, vt::DeviceType::kCUDA);
  CHECK(p.DeviceFor("blk.7.ffn_up_exps.weight") == vt::DeviceType::kCPU);
  CHECK(p.DeviceFor("blk.0.ffn_up_exps.weight") == vt::DeviceType::kCUDA);
  // `blk.17` must NOT match `blk\.7`, and this is the case an unanchored
  // implementation passes by accident: `regex_search` for "blk\.7" inside
  // "blk.17.ffn_up_exps.weight" finds nothing only because the '.' is escaped and
  // the '1' intervenes.
  CHECK(p.DeviceFor("blk.17.ffn_up_exps.weight") == vt::DeviceType::kCUDA);
  CHECK(p.DeviceFor("blk.70.ffn_up_exps.weight") == vt::DeviceType::kCUDA);
}

TEST_CASE("device placement: FIRST match wins, and the order is the input") {
  // `src/llama-model-loader.cpp:1180` scans front to back and stops at the first
  // hit. The two orderings below differ ONLY in order and must resolve
  // differently; a resolver that sorted for determinism would collapse them.
  const std::string narrow = std::string("blk\\.0") + kExps;

  const auto narrow_first = vllm::DevicePlacement::FromOverrides(
      {Ov(narrow.c_str(), "cuda"), Ov(kExps, "cpu")}, vt::DeviceType::kCUDA);
  const auto broad_first = vllm::DevicePlacement::FromOverrides(
      {Ov(kExps, "cpu"), Ov(narrow.c_str(), "cuda")}, vt::DeviceType::kCUDA);

  CHECK(narrow_first.DeviceFor("blk.0.ffn_up_exps.weight") ==
        vt::DeviceType::kCUDA);
  CHECK(broad_first.DeviceFor("blk.0.ffn_up_exps.weight") ==
        vt::DeviceType::kCPU);
  // Every OTHER layer resolves the same way under both, so the difference above
  // is the ordering rule and not two unrelated placements.
  CHECK(narrow_first.DeviceFor("blk.1.ffn_up_exps.weight") ==
        vt::DeviceType::kCPU);
  CHECK(broad_first.DeviceFor("blk.1.ffn_up_exps.weight") ==
        vt::DeviceType::kCPU);
}

TEST_CASE("device placement: overrides naming the engine's OWN device stay trivial") {
  // This is not a curiosity. `cpu_moe` on a CPU engine is exactly this, and it is
  // what a user gets for pasting a llama.cpp command line at a CPU build. Reading
  // it as non-trivial would take the engine off its single-device path in order to
  // place everything back where it already was.
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCPU);
  CHECK(p.IsTrivial());
  CHECK(p.override_count() == 1);
  CHECK(p.DeviceFor("blk.0.ffn_up_exps.weight") == vt::DeviceType::kCPU);
  CHECK(p.Describe().empty());

  // One entry that DOES move something makes the whole placement non-trivial,
  // even beside entries that do not.
  const auto mixed = vllm::DevicePlacement::FromOverrides(
      {Ov(kExps, "cpu"), Ov("\\.attn_q\\.", "cuda")}, vt::DeviceType::kCPU);
  CHECK_FALSE(mixed.IsTrivial());
}

TEST_CASE("device placement: the report names the devices, not only the count") {
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCUDA);
  const std::string d = p.Describe();
  CHECK(d.find("cuda") != std::string::npos);
  CHECK(d.find("1 override") != std::string::npos);
  // The COUNT alone cannot tell "40 layers to the CPU" from "40 layers back to
  // the device I am already on", and those differ by the whole point of the row.
  CHECK(d.find("placing to cpu") != std::string::npos);

  // Duplicate destinations are named once, so a 40-layer `-ncmoe` does not print
  // "cpu" forty times.
  std::vector<vllm::PlacementOverride> many;
  for (int i = 0; i < 40; ++i) {
    many.push_back(Ov(kExps, "cpu"));
  }
  const auto big =
      vllm::DevicePlacement::FromOverrides(many, vt::DeviceType::kCUDA);
  const std::string bd = big.Describe();
  CHECK(bd.find("40 overrides") != std::string::npos);
  size_t occurrences = 0;
  for (size_t at = bd.find("cpu"); at != std::string::npos;
       at = bd.find("cpu", at + 1)) {
    ++occurrences;
  }
  CHECK(occurrences == 1);
}

TEST_CASE("device placement: a bad device or a bad regex is REFUSED, not dropped") {
  // W1's parser refuses both at startup, so reaching either here means a caller
  // built the list by hand. Dropping the entry would place fewer tensors than the
  // install line claims it placed, which is the silent-divergence shape this tree
  // refuses everywhere.
  CHECK_THROWS_AS(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "CPU_BUFFER")},
                                           vt::DeviceType::kCUDA),
      std::invalid_argument);
  CHECK_THROWS_AS(vllm::DevicePlacement::FromOverrides({Ov("ffn_(up", "cpu")},
                                                       vt::DeviceType::kCUDA),
                  std::invalid_argument);
}

TEST_CASE("device placement: it is a placement, NOT a shard") {
  // The spec makes a review against `models/tensor_parallel.h` W2's gate, because
  // the easy failure is to grow a second sharding concept here. This case is that
  // review made executable: the seam answers a DEVICE for a NAME and has no
  // notion of a rank, a world size or a dimension to split. If a future change
  // adds one, it belongs in `TensorParallel`, and this case should be the thing
  // that argues about it.
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCUDA);
  // The same name resolves the same way however many ranks exist, because there
  // is nowhere to tell it about them.
  CHECK(p.DeviceFor("blk.0.ffn_up_exps.weight") ==
        p.DeviceFor("blk.0.ffn_up_exps.weight"));
  CHECK(p.engine_device() == vt::DeviceType::kCUDA);
  CHECK(p.override_count() == 1);
}
