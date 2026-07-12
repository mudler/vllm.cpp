#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x64x256() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 64, 256, true, false>(kFullTacticDescriptors[12]),
      MakeCandidate<128, 64, 256, false, false>(kFullTacticDescriptors[13]),
      MakeCandidate<128, 64, 256, true, true>(kFullTacticDescriptors[14]),
      MakeCandidate<128, 64, 256, false, true>(kFullTacticDescriptors[15]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
