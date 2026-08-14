// The DiT's U-Net (uvit) skip connections across transformer depth (#634).
//
// Upstream `indextts/s2mel/modules/gpt_fast/model.py:152-190` and `:213-232`,
// index-tts @4f8792ff120cd3ea470dd511e997a17c86cddd10. The shipped config sets
// `s2mel.DiT.uvit_skip_connection: true`, and the checkpoint carries one
// `layers.N.skip_in_linear` per receiving layer.
//
// The first half of the stack pushes its OUTPUT onto a stack; the second half
// pops one before running. So at the shipped depth 13 layer 7 receives layer 5's
// output, layer 8 receives layer 4's, and layer 6 in the middle neither emits
// nor receives.
//
// Every plausible variant of this routing still produces a running model: FIFO
// instead of LIFO, `>=` instead of `>`, pushing the layer's INPUT rather than
// its output. `scripts/gen-dit-skip-schedule.py` records what upstream actually
// does by driving its own Transformer, and the gate compares against that.
//
// One upstream asymmetry is preserved deliberately: at EVEN depth there is one
// more emitter than receiver, so the earliest emitted skip is never consumed.
// It is reported rather than "corrected".
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace dit_skip {

struct Schedule {
  std::vector<int64_t> emit;     // layer indices that push their output
  std::vector<int64_t> receive;  // layer indices that pop one before running
  // source[i] is the layer whose output layer i receives, or -1 for none.
  std::vector<int64_t> source;
  int64_t orphaned = 0;  // emitted but never consumed; non-zero at even depth
};

// The routing for a stack of `layers` transformer blocks.
Schedule Plan(int64_t layers);

// skip_in_linear(cat([x, skip], dim=-1)).
//
// THE CONCATENATION ORDER IS THE DETAIL: x first, then the skip. Reversing it
// reads the same weights against the wrong halves and still returns a tensor of
// the right shape.
//
// x and skip are [frames, dim]; weight is [dim, 2 * dim]; returns [frames, dim].
std::vector<float> ApplySkip(const std::vector<float>& x, const std::vector<float>& skip,
                             int64_t frames, int64_t dim,
                             const std::vector<float>& weight,
                             const std::vector<float>& bias);

}  // namespace dit_skip
}  // namespace models
}  // namespace vllm
