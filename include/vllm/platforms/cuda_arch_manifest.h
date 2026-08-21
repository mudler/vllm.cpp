// The COMPILED-ARCHITECTURE manifest matcher (issue #1357, umbrella #1332 M2).
//
// WHY THIS EXISTS. A capability predicate that answers from the DEVICE cannot
// answer a question about the BINARY. The reference engine's
// FlashAttentionBackend declares `capability >= (8,0)`; on a GB10 (12,1) that is
// true, its FA2 binary carries sm_80 SASS plus compute_80 PTX and nothing else,
// and every launch fails the driver JIT with cudaErrorUnsupportedPtxVersion.
// Measured on that board, same wheel and same prompt: requesting FLASHINFER
// generates text and exits 0 while the default resolves FLASH_ATTN and dies at
// the first attention call. This tree had written the same class of claim —
// `CudaPlatform::supports_fa2_attention()` returned true for every CUDA device
// while the default build compiles FA2 for one architecture.
//
// The manifest string comes from `cuda_compiled_archs.h`, which CMake GENERATES
// from `VT_FA2_ARCHS` — the same variable `vt_cuda_set_source_gencode` turns
// into nvcc's `-gencode` options. It is never hand-written: a parallel list is
// `CUDA_SUPPORTED_ARCHS`, the shape that caused this.
//
// This header is pure host code with no CUDA dependency, so the CPU test tier
// exercises the real matcher against INJECTED manifests rather than a copy.
#ifndef VLLM_PLATFORMS_CUDA_ARCH_MANIFEST_H_
#define VLLM_PLATFORMS_CUDA_ARCH_MANIFEST_H_

#include <string>
#include <vector>

namespace vllm::platforms {

// One entry of the manifest: a target nvcc was actually asked to emit SASS for.
struct CompiledArch {
  int major = 0;
  int minor = 0;
  // The CMake/nvcc arch-specific suffix: 'a', 'f', or '\0' for a base target.
  // Load-bearing — `sm_121a` code runs on sm_121 hardware ONLY, so an 'a' target
  // never satisfies a request for the base arch and vice versa.
  char suffix = '\0';
};

// Parse the generated manifest, which is the CMake CUDA_ARCHITECTURES form that
// `vt_cuda_archs_denormalize` produces: "121a", "80,86,87,89,120a,121a", or
// empty when the feature is not compiled at all. Unparseable entries are DROPPED
// rather than guessed, because a guessed entry is a claim.
std::vector<CompiledArch> ParseCompiledArchs(const std::string& manifest);

// Does the build contain SASS that this device can execute?
//
// THE RULE, and its polarity is the point. An entry serves the device when:
//   * it matches (major, minor) exactly AND carries the same suffix; or
//   * it is a BASE target (no suffix) of the same major whose minor is <= the
//     device minor — CUDA's SASS minor-version compatibility, which is what lets
//     an sm_80 cubin run on sm_86.
// Everything else is NOT compiled.
//
// PTX IS DELIBERATELY NOT A YES, and this is the whole lesson of #1332. The
// manifest carries no PTX entries by construction (`vt_cuda_archs_denormalize`
// strips `+PTX`), and that is the correct input: "a driver JIT could in
// principle produce code" is exactly what the reference engine believed on the
// GB10 before every launch failed.
//
// A false negative falls back to a slower path that produces correct output. A
// false positive launches a kernel that has no code for the device. So where
// this rule is uncertain it answers no, on purpose.
bool ArchIsCompiled(const std::vector<CompiledArch>& compiled, int device_major,
                    int device_minor);

// Convenience over the raw manifest string.
bool ArchIsCompiled(const std::string& manifest, int device_major,
                    int device_minor);

}  // namespace vllm::platforms

#endif  // VLLM_PLATFORMS_CUDA_ARCH_MANIFEST_H_
