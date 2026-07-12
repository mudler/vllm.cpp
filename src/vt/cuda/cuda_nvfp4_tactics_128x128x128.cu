#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x128x128() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 128, 128, true, false>(kFullTacticDescriptors[16]),
      MakeCandidate<128, 128, 128, false, false>(kFullTacticDescriptors[17]),
      MakeCandidate<128, 128, 128, true, true>(kFullTacticDescriptors[18]),
      MakeCandidate<128, 128, 128, false, true>(kFullTacticDescriptors[19]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
