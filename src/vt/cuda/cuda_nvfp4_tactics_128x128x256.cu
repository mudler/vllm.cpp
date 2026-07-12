#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x128x256() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 128, 256, true, false>(kFullTacticDescriptors[20]),
      MakeCandidate<128, 128, 256, false, false>(kFullTacticDescriptors[21]),
      MakeCandidate<128, 128, 256, true, true>(kFullTacticDescriptors[22]),
      MakeCandidate<128, 128, 256, false, true>(kFullTacticDescriptors[23]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
