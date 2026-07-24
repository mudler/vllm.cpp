// Gemma-4 (`Gemma4ForCausalLM`) HONESTY PASS — sweep W6. No e2e gate exists: the
// bare text row has NO standalone checkpoint (every public Gemma-4 is
// multimodal-wrapped, Gemma4*ForConditionalGeneration / Gemma4Unified…, >=12B),
// and the text backbone needs a large NEW-primitive stack (Per-Layer Embeddings,
// YOCO KV-sharing, the Gemma-4 MoE router, k_eq_v, double-wide MLP, per-layer
// scalar) that no other row uses. Mirroring the GLM spike's blocked-row protocol,
// this test records the GATEABLE SUBSET and claims NOTHING beyond it:
//
//   * We do NOT register Gemma4ForCausalLM — resolving it fails LOUDLY with the
//     exact "not supported for now" oracle message (we never over-claim a row we
//     have not ported). Meanwhile the three ported Gemma text siblings DO resolve.
//   * The measured facts (0.25.0 oracle registry, MM-wrapped-only checkpoints, the
//     PLE/YOCO/MoE primitive stack) are recorded here and in the Gemma spec as
//     HW/DEP-BLOCKED. The config text_config extraction (gemma4.py::_get_text_config)
//     is verified at the ORACLE/Python level (scripts, no weights) — see state.md.
//
// Grounding: vllm/model_executor/models/registry.py:110 (Gemma4ForCausalLM),
// gemma4.py (1715 lines: routing :92-203, Gemma4Router :251-298, Gemma4MoE
// :301-365, Gemma4Attention/k_eq_v/KV-share :368-539, PLE :965-1037, YOCO
// :757-950). See .agents/specs/sweep-gemma.md §0.1, §W6.
#include "vllm/model_executor/models/model_registry.h"

#include <doctest/doctest.h>

#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using vllm::HfConfig;
using vllm::ModelRegistry;

namespace {
HfConfig Config(std::vector<std::string> architectures) {
  HfConfig c;
  c.architectures = std::move(architectures);
  return c;
}
}  // namespace

TEST_CASE("gemma4 honesty: Gemma4ForCausalLM is NOT registered (we do not over-claim)") {
  // The row is gate-BLOCKED (MM-wrapped-only checkpoints + a PLE/YOCO/MoE campaign
  // + oracle-support unverified for e2e). We must NOT register it: resolving must
  // fail loudly rather than silently resolve to a wrong factory.
  const HfConfig cfg = Config({"Gemma4ForCausalLM"});
  CHECK_THROWS_AS(ModelRegistry::Resolve(cfg), std::runtime_error);

  const std::vector<std::string_view> supported = ModelRegistry::SupportedArchs();
  const auto has = [&](std::string_view a) {
    for (std::string_view s : supported) if (s == a) return true;
    return false;
  };
  CHECK_FALSE(has("Gemma4ForCausalLM"));
  // The three ported Gemma text siblings ARE supported (the family is real; only
  // the Gemma-4 backbone is a scoped follow-on campaign).
  CHECK(has("GemmaForCausalLM"));
  CHECK(has("Gemma2ForCausalLM"));
  CHECK(has("Gemma3ForCausalLM"));
}

TEST_CASE("gemma4 honesty: recorded blocked-row facts (measured, no e2e claim)") {
  // Documentation asserts — the honesty subset, stated so a reader of the suite
  // sees exactly why Gemma-4 e2e is deferred. No numeric behaviour is claimed.
  MESSAGE("gemma4 W6: 0.25.0 oracle registry LISTS 'Gemma4ForCausalLM', but every "
          "PUBLIC checkpoint is multimodal-wrapped (Gemma4*ForConditionalGeneration / "
          "Gemma4Unified, >=12B) — the bare text row has NO standalone checkpoint.");
  MESSAGE("gemma4 W6: the text backbone adds a large NEW-primitive stack (Per-Layer "
          "Embeddings, YOCO self/cross KV-sharing, Gemma-4 MoE router softmax-over-all "
          "+ per_expert_scale, k_eq_v, double-wide MLP, per-layer scalar) that no other "
          "row uses — a scoped follow-on campaign, not this bring-up.");
  MESSAGE("gemma4 W6 VERDICT: HW/DEP-BLOCKED — recorded, NOTHING claimed beyond the "
          "gateable subset (registry non-over-claim above + Python-level text_config "
          "resolution in state.md). No implementation of the multimodal stack.");
  CHECK(true);
}
