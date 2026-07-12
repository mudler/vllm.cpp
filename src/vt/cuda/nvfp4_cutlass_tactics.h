// vllm.cpp — internal raw-pointer ABI for split SM12 NVFP4 CUTLASS tactics.
#pragma once

#include <array>
#include <cstddef>

#include <cuda_runtime_api.h>

#include "vt/cuda/nvfp4_tactic_ids.h"

namespace vt::cuda::nvfp4 {

struct LaunchParams {
  void* d = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  const void* a_sf = nullptr;
  const void* b_sf = nullptr;
  const float* alpha = nullptr;
  int m = 0;
  int n = 0;
  int k = 0;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  cudaStream_t stream = nullptr;
};

using WorkspaceSizeFn = size_t (*)(const LaunchParams&);
using RunFn = void (*)(const LaunchParams&);

struct Candidate {
  TacticDescriptor descriptor;
  WorkspaceSizeFn workspace_size;
  RunFn run;
};

using CandidateGroup = std::array<Candidate, 4>;

const CandidateGroup& FullTactics128x32x128();
const CandidateGroup& FullTactics128x32x256();
const CandidateGroup& FullTactics128x64x128();
const CandidateGroup& FullTactics128x64x256();
const CandidateGroup& FullTactics128x128x128();
const CandidateGroup& FullTactics128x128x256();
const CandidateGroup& FullTactics256x128x128();
const CandidateGroup& FullTactics128x256x128();
const CandidateGroup& W1Tactics();

}  // namespace vt::cuda::nvfp4
