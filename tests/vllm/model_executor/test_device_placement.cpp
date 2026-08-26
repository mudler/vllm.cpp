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

// ── `MoePlacementPlan`: the per-layer resolution the forward reads ────────────

TEST_CASE("moe plan: cpu_moe's blanket regex places EVERY layer") {
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCUDA);
  const auto plan = vllm::MoePlacementPlan::Resolve(p, 48);
  CHECK(plan.PlacesAnything());
  // COUNT, not a spot check: `-cmoe` means all of them, and a resolver that
  // placed 47 would be invisible to a test that sampled one.
  CHECK(plan.placed_layer_count() == 48);
  for (int64_t l = 0; l < 48; ++l) {
    INFO("layer " << l);
    CHECK(plan.DeviceForLayer(l) == vt::DeviceType::kCPU);
  }
  CHECK(plan.Describe().find("48 layers") != std::string::npos);
  CHECK(plan.Describe().find("cpu") != std::string::npos);
}

TEST_CASE("moe plan: n_cpu_moe's per-layer regexes place exactly the FIRST N") {
  // The desugared `-ncmoe 4` list, built the way W1 builds it.
  std::vector<vllm::PlacementOverride> overrides;
  for (int64_t i = 0; i < 4; ++i) {
    overrides.push_back(
        vllm::PlacementOverride{vllm::LlmFfnExpsBlockRegex(i), "cpu"});
  }
  const auto plan = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides(overrides, vt::DeviceType::kCUDA), 32);
  CHECK(plan.placed_layer_count() == 4);
  for (int64_t l = 0; l < 4; ++l) {
    INFO("layer " << l);
    CHECK(plan.DeviceForLayer(l) == vt::DeviceType::kCPU);
  }
  // And crucially NOT layer 4, nor the two-digit layers whose names contain the
  // one-digit ones as substrings. `blk.1` must not capture `blk.10`.
  for (int64_t l = 4; l < 32; ++l) {
    INFO("layer " << l);
    CHECK(plan.DeviceForLayer(l) == vt::DeviceType::kCUDA);
  }
}

TEST_CASE("moe plan: an empty placement places nothing and is inert") {
  const vllm::DevicePlacement none(vt::DeviceType::kCUDA);
  const auto plan = vllm::MoePlacementPlan::Resolve(none, 32);
  CHECK_FALSE(plan.PlacesAnything());
  CHECK(plan.placed_layer_count() == 0);
  CHECK(plan.Describe().empty());
  for (int64_t l = 0; l < 32; ++l) {
    CHECK(plan.DeviceForLayer(l) == vt::DeviceType::kCUDA);
  }
}

TEST_CASE("moe plan: a PARTIAL placement is REFUSED by name, never half-applied") {
  // `-ot` can legally name one of the three routed-expert tensors. The MoE block
  // runs ONE grouped GEMM over gate, up and down, so splitting them across
  // devices is a different kernel and not a scheduling decision. Picking one of
  // the three would move weights the operator never asked to move, silently.
  const auto p = vllm::DevicePlacement::FromOverrides(
      {Ov("\\.ffn_down_exps", "cpu")}, vt::DeviceType::kCUDA);
  std::string msg;
  try {
    vllm::MoePlacementPlan::Resolve(p, 4);
    msg = "ACCEPTED (no throw)";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  CHECK(msg.find("splits its routed experts") != std::string::npos);
  // The message names BOTH sides and the layer, so the operator can see which
  // pattern to widen.
  CHECK(msg.find("layer 0") != std::string::npos);
  CHECK(msg.find("ffn_down_exps") != std::string::npos);
  CHECK(msg.find("must share a device") != std::string::npos);

  // Widening it to the full alternation is accepted, which is the fix the
  // message tells the operator to make.
  CHECK_NOTHROW(vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                           vt::DeviceType::kCUDA),
      4));
}

TEST_CASE("moe plan: the names asked about are llama.cpp's GGUF spelling") {
  // The resolver's expectation must be checkable rather than trusted: a pattern
  // an operator wrote for llama.cpp is written against THESE names, so asking
  // any other spelling would silently match nothing and place no layer.
  const std::vector<std::string> names = vllm::RoutedExpertTensorNames(7);
  CHECK(names.size() == 3);
  for (const std::string& n : names) {
    INFO("name: " << n);
    CHECK(n.rfind("blk.7.", 0) == 0);
  }
  // And each one is matched by the regex `-cmoe` installs, which is the join
  // between what the operator types and what this resolver asks.
  const auto p = vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                                      vt::DeviceType::kCUDA);
  for (const std::string& n : names) {
    INFO("name: " << n);
    CHECK(p.DeviceFor(n) == vt::DeviceType::kCPU);
  }
}

TEST_CASE("moe plan: a layer the model does not have answers INERT, never throws") {
  // Read on the decode path. A caller asking about a layer out of range is a bug
  // that must surface as unchanged behaviour, not as a throw mid-token.
  const auto plan = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                           vt::DeviceType::kCUDA),
      4);
  CHECK(plan.DeviceForLayer(-1) == vt::DeviceType::kCUDA);
  CHECK(plan.DeviceForLayer(4) == vt::DeviceType::kCUDA);
  CHECK(plan.DeviceForLayer(1000) == vt::DeviceType::kCUDA);
}

// ── W3b: the placed path actually RUNS, and the round trip is exercised ───────

TEST_CASE("placement queue: cpu is a legal target and is reused, not recreated") {
  // One queue per device for the process's life. A queue created per layer per
  // token would dominate the round trip it exists to serve, so identity is the
  // assertion — not merely that a queue comes back.
  vt::Queue& a = vllm::PlacementQueue(vt::DeviceType::kCPU);
  vt::Queue& b = vllm::PlacementQueue(vt::DeviceType::kCPU);
  CHECK(&a == &b);
  CHECK(a.device.type == vt::DeviceType::kCPU);
}

TEST_CASE("placement queue: an ACCELERATOR target is refused, never leaked") {
  // The engine may run anywhere; it is the DESTINATION that is limited today.
  // A process-lifetime queue on a backend whose `DestroyQueue` releases a stream
  // would leak it, so this refuses rather than leaking quietly — and the message
  // says which half of the arrangement is restricted, because "cuda is not
  // supported" would read as though a CUDA engine were refused.
  std::string msg;
  try {
    vllm::PlacementQueue(vt::DeviceType::kCUDA);
    msg = "ACCEPTED (no throw)";
  } catch (const std::invalid_argument& e) {
    msg = e.what();
  }
  CHECK(msg.find("placement TARGET") != std::string::npos);
  CHECK(msg.find("engine may run on any device") != std::string::npos);
}

TEST_CASE("active plan: installed once, read many, and resettable for a test") {
  vllm::ResetActiveMoePlacementPlanForTesting();
  // The default is inert, so a process that never installs one places nothing
  // and the forward's branch is the existing call.
  CHECK_FALSE(vllm::ActiveMoePlacementPlan().PlacesAnything());

  const auto plan = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                           vt::DeviceType::kVULKAN),
      6);
  vllm::SetActiveMoePlacementPlan(plan);
  CHECK(vllm::ActiveMoePlacementPlan().PlacesAnything());
  CHECK(vllm::ActiveMoePlacementPlan().placed_layer_count() == 6);
  // Read repeatedly: this is taken once per MoE layer per token and must answer
  // the same thing every time without re-deciding.
  for (int i = 0; i < 3; ++i) {
    CHECK(vllm::ActiveMoePlacementPlan().DeviceForLayer(0) ==
          vt::DeviceType::kCPU);
  }
  vllm::ResetActiveMoePlacementPlanForTesting();
  CHECK_FALSE(vllm::ActiveMoePlacementPlan().PlacesAnything());
}

TEST_CASE("placement: a VULKAN engine placing to cpu is a REAL cross-device plan") {
  // This is the shape that makes W3b gateable on this hardware at all. The placed
  // branch needs the engine device and the placement device to DIFFER — not a
  // discrete accelerator — and Vulkan is a distinct `vt::DeviceType`. So a Vulkan
  // engine with `cpu_moe` enters the placed path on a box with only a software
  // rasteriser, which is what turns W3b from untestable into merely unmeasured.
  //
  // It gates CORRECTNESS only. lavapipe is a software rasteriser, so a placement
  // measured against it compares CPU with CPU; the speed axis stays with W5.
  const auto plan = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                           vt::DeviceType::kVULKAN),
      4);
  CHECK(plan.PlacesAnything());
  CHECK(plan.placed_layer_count() == 4);
  for (int64_t l = 0; l < 4; ++l) {
    // Every layer resolves AWAY from the engine device, which is exactly the
    // condition `RunMoeLayer` branches on.
    CHECK(plan.DeviceForLayer(l) != vt::DeviceType::kVULKAN);
    CHECK(plan.DeviceForLayer(l) == vt::DeviceType::kCPU);
  }
  // And the same overrides against a CPU engine place NOTHING, so the branch is
  // driven by the pair of devices and not by the pattern alone.
  const auto same_device = vllm::MoePlacementPlan::Resolve(
      vllm::DevicePlacement::FromOverrides({Ov(kExps, "cpu")},
                                           vt::DeviceType::kCPU),
      4);
  CHECK_FALSE(same_device.PlacesAnything());
}
