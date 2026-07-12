#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x32x128() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 32, 128, true, false>(kFullTacticDescriptors[0]),
      MakeCandidate<128, 32, 128, false, false>(kFullTacticDescriptors[1]),
      MakeCandidate<128, 32, 128, true, true>(kFullTacticDescriptors[2]),
      MakeCandidate<128, 32, 128, false, true>(kFullTacticDescriptors[3]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
