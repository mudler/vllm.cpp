#include "vt/cuda/nvfp4_cutlass_tactic_impl.cuh"

namespace vt::cuda::nvfp4 {

const CandidateGroup& FullTactics128x64x128() {
  using detail::MakeCandidate;
  static const CandidateGroup group{{
      MakeCandidate<128, 64, 128, true, false>(kFullTacticDescriptors[8]),
      MakeCandidate<128, 64, 128, false, false>(kFullTacticDescriptors[9]),
      MakeCandidate<128, 64, 128, true, true>(kFullTacticDescriptors[10]),
      MakeCandidate<128, 64, 128, false, true>(kFullTacticDescriptors[11]),
  }};
  return group;
}

}  // namespace vt::cuda::nvfp4
