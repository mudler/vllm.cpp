#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics256x128x128() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<256, 128, 128, true, false>(kFullTacticDescriptors[24]),
      MakeCandidate<256, 128, 128, false, false>(kFullTacticDescriptors[25]),
      MakeCandidate<256, 128, 128, true, true>(kFullTacticDescriptors[26]),
      MakeCandidate<256, 128, 128, false, true>(kFullTacticDescriptors[27]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
