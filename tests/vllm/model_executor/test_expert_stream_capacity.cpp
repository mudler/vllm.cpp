// MODEL-TEXT-GLM-MOE-DSA W3 (#2214, spec .agents/specs/glm-dsa-latest-deepseek.md
// §3.3) — a slot budget below one decode step's working set REFUSES, by name.
//
// WHAT THIS REPLACES. Nothing refused. `ExpertSlotCache::Acquire` marks every
// entry it serves protected for the step and only `EndStep` clears it, so one
// step's working set is every slice it touches, all resident at once. Below that
// budget `Acquire` returns -1, `Slice` returns nullptr, and the caller reads the
// tower IN PLACE out of the mmap. That is counted on stderr and reported as a
// successful run. On the checkpoint this row targets — 75 MoE layers, 3 towers
// each, 8 experts per token, so 1800 slices against a default of 64 slots — the
// silent fallback is a 187 GiB random read per token through the page cache, and
// a benchmark measuring it would publish a page-cache number under a streaming
// label. Spec §3.3: "W3 therefore owes a refusal, not a fallback".
//
// WHY BOTH HALVES ARE HERE. The refusal is only worth having if the arithmetic
// it fires on is the arithmetic the lane actually runs, so this file gates the
// working-set formula and the geometry read off a GGUF header as well as the
// refusal itself. A refusal whose input is wrong refuses the wrong loads.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/expert_stream_seam.h"
#include "vllm/model_executor/model_loader/gguf_device_fit.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"

using vllm::GgufExpertLaneGeometry;
using vllm::GgufStreamedExpertLaneGeometry;
using vllm::expert_stream::RequireSlotCapacity;
using vllm::expert_stream::WorkingSetSlots;

namespace {

// The model this row targets, as §3.3 states it: 75 MoE layers x 3 towers = 225
// streamed towers, 8 experts per token, so 1800 slices in one step.
constexpr int64_t kGlmTowers = 225;
constexpr int64_t kGlmExpertsPerTok = 8;
constexpr int64_t kGlmWorkingSet = 1800;

std::string RefusalMessage(const std::string& model, int64_t towers,
                           int64_t per_tok, int64_t slots) {
  try {
    RequireSlotCapacity(model, towers, per_tok, slots);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return "";
}

}  // namespace

TEST_CASE("the working set is towers x experts-per-token, which is the protected set") {
  // The number §3.3 computes, reproduced by the function the refusal uses. Not a
  // transcription of the constant: the inputs are the model's geometry and the
  // product is asserted against the spec's own figure.
  CHECK(WorkingSetSlots(kGlmTowers, kGlmExpertsPerTok) == kGlmWorkingSet);

  // A gate/up/down MoE's tower count IS `moe_layers * 3`, which is why the
  // caller passes towers rather than layers: the seam never has to be told how
  // many projections an architecture merges.
  CHECK(WorkingSetSlots(4 * 3, 2) == 24);

  // UNKNOWN, not zero. A caller that could not determine either term must not
  // have that turn into a working set of zero that every budget satisfies, nor
  // into one that refuses everything.
  CHECK(WorkingSetSlots(0, 8) == 0);
  CHECK(WorkingSetSlots(225, 0) == 0);
  CHECK(WorkingSetSlots(-1, 8) == 0);
}

TEST_CASE("a budget BELOW one step's working set is refused, and the message is actionable") {
  // THE DEFAULT IS THE CASE THAT MATTERS. `weight_residency.cpp`'s default slot
  // count is 64; the target model needs 1800. Before this change that
  // combination loaded and degraded silently.
  const std::string what = RefusalMessage("glm-dsa", kGlmTowers, kGlmExpertsPerTok, 64);
  REQUIRE_FALSE(what.empty());

  // BY NAME, and every term an operator needs to act is present: which model,
  // how many slots it needs, what it was given, and which knob to turn. "Too
  // small" without the budget it needed is not actionable — the same standard
  // the slot-BYTES refusal already meets.
  CHECK(what.find("glm-dsa") != std::string::npos);
  CHECK(what.find("1800") != std::string::npos);
  CHECK(what.find("64") != std::string::npos);
  CHECK(what.find("VT_MOE_EXPERT_STREAM_SLOTS") != std::string::npos);
  // And it says WHY silence was the wrong alternative, because an operator who
  // raises the budget without knowing that will raise it to the first number
  // that stops the message rather than to the working set.
  CHECK(what.find("mmap") != std::string::npos);
}

TEST_CASE("a budget AT or ABOVE the working set is accepted") {
  // The boundary is inclusive: exactly the working set is exactly enough,
  // because that is the count of entries protected at once.
  CHECK(RefusalMessage("glm-dsa", kGlmTowers, kGlmExpertsPerTok, kGlmWorkingSet).empty());
  CHECK(RefusalMessage("glm-dsa", kGlmTowers, kGlmExpertsPerTok, kGlmWorkingSet + 1).empty());
  CHECK(RefusalMessage("glm-dsa", kGlmTowers, kGlmExpertsPerTok, 4096).empty());

  // The 4x4 test model against the default budget: 24 against 64. This is the
  // case that proves the refusal does not fire on the models already gated —
  // if it did, every existing streaming binary would red.
  CHECK(RefusalMessage("qwen35moe", 4 * 3, 2, 64).empty());

  // And the wide G3 model, 6 layers x 3 towers x 3 per token = 54, against the
  // 256 slots its binary sets.
  CHECK(RefusalMessage("qwen35moe", 6 * 3, 3, 256).empty());
}

TEST_CASE("an UNKNOWN geometry is inert, not a refusal") {
  // A caller that read a file it did not understand hands over a zero. That
  // must not become a refusal of every budget: the check would then fire on
  // architectures whose metadata this seam has never seen, which is a refusal
  // invented out of a missing value rather than derived from one. The decision
  // about whether a missing geometry is itself an error belongs to the caller,
  // which knows what it was reading.
  CHECK(RefusalMessage("glm-dsa", 0, 8, 1).empty());
  CHECK(RefusalMessage("glm-dsa", 225, 0, 1).empty());
  // A non-positive budget is likewise the resolver's "unset", not "zero slots".
  CHECK(RefusalMessage("glm-dsa", kGlmTowers, kGlmExpertsPerTok, 0).empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// The refusal's INPUT, read off a real GGUF header rather than hand-passed.
//
// A refusal whose arithmetic is right and whose inputs are wrong refuses the
// wrong loads, and both terms here come from a file the loader opens. This is
// the same standard §3.7 sets for W1's dtype: the geometry must arrive through
// `GgufFile::Open` on a real header, not through a hand-built struct.

namespace {

constexpr uint32_t kGgmlF32 = 0;
constexpr std::string_view kExpsSuffix = "_exps.weight";

// A minimal MoE GGUF: `layers` blocks x 3 stacked expert towers, plus the two
// kv the geometry reads. The tensor DATA is irrelevant to the count, so the
// towers are 1-element f32 to keep the fixture small.
std::string MoeGguf(const std::string& arch, int layers, uint32_t experts_used,
                    bool write_used_count = true) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", arch));
  if (write_used_count)
    b.AddKv(gguf_test::U32Kv(arch + ".expert_used_count", experts_used));
  for (int l = 0; l < layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".ffn_";
    for (const char* proj : {"gate", "up", "down"}) {
      b.AddTensor(p + proj + "_exps.weight", {1, 1, 1}, kGgmlF32,
                  std::string(4, '\0'));
    }
    // A per-layer tensor that must NOT be counted: the suffix predicate is what
    // separates the streamed class from everything else, and a count that
    // included this would over-state the working set on every real checkpoint.
    b.AddTensor(p + "norm.weight", {1}, kGgmlF32, std::string(4, '\0'));
  }
  return b.Build();
}

GgufExpertLaneGeometry GeometryOf(const std::string& bytes) {
  gguf_test::TempFile f(bytes);
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  return GgufStreamedExpertLaneGeometry(g, kExpsSuffix);
}

}  // namespace

TEST_CASE("the geometry is read off the GGUF header the lane will serve") {
  const GgufExpertLaneGeometry g = GeometryOf(MoeGguf("glm-dsa", 5, 8));
  // 5 layers x 3 towers. The per-layer `ffn_norm.weight` in the same fixture is
  // NOT counted, which is what makes this the streamed class and not the
  // tensor table.
  CHECK(g.streamed_tower_count == 15);
  CHECK(g.experts_per_tok == 8);
  // And the product is the working set the refusal fires on.
  CHECK(WorkingSetSlots(g.streamed_tower_count, g.experts_per_tok) == 120);
  CHECK_FALSE(RefusalMessage("glm-dsa", g.streamed_tower_count,
                             g.experts_per_tok, 64)
                  .empty());
  CHECK(RefusalMessage("glm-dsa", g.streamed_tower_count, g.experts_per_tok,
                       120)
            .empty());
}

TEST_CASE("a header with no expert_used_count reports UNKNOWN, not zero") {
  // The term is left at 0 and the refusal is inert on it. A file whose metadata
  // this reader did not understand must not be refused by a number the reader
  // invented for it.
  const GgufExpertLaneGeometry g =
      GeometryOf(MoeGguf("glm-dsa", 5, 8, /*write_used_count=*/false));
  CHECK(g.streamed_tower_count == 15);
  CHECK(g.experts_per_tok == 0);
  CHECK(RefusalMessage("glm-dsa", g.streamed_tower_count, g.experts_per_tok, 1)
            .empty());
}

TEST_CASE("a DENSE header has no streamed towers at all") {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "llama"));
  b.AddTensor("blk.0.ffn_gate.weight", {1}, kGgmlF32, std::string(4, '\0'));
  const GgufExpertLaneGeometry g = GeometryOf(b.Build());
  CHECK(g.streamed_tower_count == 0);
  // Inert: a model with nothing to stream cannot have too small a budget.
  CHECK(RefusalMessage("llama", g.streamed_tower_count, g.experts_per_tok, 1)
            .empty());
}
