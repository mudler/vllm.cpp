#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x32x256() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 32, 256, true, false>(kFullTacticDescriptors[4]),
      MakeCandidate<128, 32, 256, false, false>(kFullTacticDescriptors[5]),
      MakeCandidate<128, 32, 256, true, true>(kFullTacticDescriptors[6]),
      MakeCandidate<128, 32, 256, false, true>(kFullTacticDescriptors[7]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
