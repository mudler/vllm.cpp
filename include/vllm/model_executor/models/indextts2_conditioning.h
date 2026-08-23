#pragma once

// The two pure steps that turn a reference clip's log-mel into the speaker
// conditioning IndexTTS-2.5 feeds to BOTH halves of the model: the talker's
// conditioning row 0, and the S2Mel front end's style input.
//
// They live here rather than inline in the engine because the engine's own path
// needs a 5 GB checkpoint to run, and neither of these steps does. Extracting
// them is what lets a gate see them at all.

#include <cstdint>
#include <vector>

namespace vllm::indextts2 {

// Subtract each mel column's own mean across frames, in place. Upstream does
// this immediately before CAMPPlus (`infer_v2_5.py:647`); CAMPPlus's first
// batch-norm assumes it, so skipping it biases every embedding.
// `mel` is frame-major, `frames * bins` entries. Refuses a shape that does not
// describe the buffer.
void MeanCentreColumns(std::vector<float>& mel, int64_t bins, int64_t frames);

// `w` is [out_dim, style_dim] row-major, `b` is empty or `out_dim` long.
// Returns `out_dim` entries. Refuses a weight that does not match the widths.
std::vector<float> ProjectSpeaker(const std::vector<float>& style,
                                  const std::vector<float>& w,
                                  const std::vector<float>& b, int64_t out_dim);

}  // namespace vllm::indextts2
