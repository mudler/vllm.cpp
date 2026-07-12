#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x256x128() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 256, 128, true, false>(kFullTacticDescriptors[28]),
      MakeCandidate<128, 256, 128, false, false>(kFullTacticDescriptors[29]),
      MakeCandidate<128, 256, 128, true, true>(kFullTacticDescriptors[30]),
      MakeCandidate<128, 256, 128, false, true>(kFullTacticDescriptors[31]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
