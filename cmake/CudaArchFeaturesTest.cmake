# Configure-tier test for the CUDA per-arch FEATURE TABLE.
#
# Run standalone, with no CUDA toolkit and no GPU:
#   cmake -P cmake/CudaArchFeaturesTest.cmake
# (scripts/check-cuda-arch-features.sh wraps this; CI runs the wrapper.)
#
# WHY THIS EXISTS. cmake/CudaArchFeatures.cmake is what decides, per requested
# architecture, whether fp4-mma / cutlass-nvfp4 / cutlass-fp8 / marlin-nvfp4 /
# fa2 are compiled in. Its predecessor was four `MATCHES "12[01]a"` regexes over
# the whole arch string, whose failure mode was a SILENT capability drop — the
# build stayed green and simply stopped emitting the fp4 MMA. A resolution table
# with that failure mode must be asserted, not eyeballed in STATUS output, so
# every case below is a hard expectation and a mismatch fails the script.
#
# The `sm_120a` cases are the load-bearing ones for BACKEND-CUDA-SM120: consumer
# Blackwell (RTX 50-series) is the same sm_12x family as GB10's sm_121, the
# feature-table cells already name `12.0a`, and these assertions pin that
# resolution so a future table edit cannot quietly un-support it.
cmake_minimum_required(VERSION 3.24)

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
include("${_here}/CudaArchFeatures.cmake")

set(_failures 0)

# expect_feature(<target-list> <feature> <expected-archs>)
#   Resolves FEATURE against the requested target list and compares the result
#   with the expectation. Both are CMake CUDA_ARCHITECTURES-form lists.
function(expect_feature TARGETS FEATURE EXPECTED)
  set(VLLM_CPP_CUDA_ARCHITECTURES "${TARGETS}")
  vt_cuda_feature_archs(_got "${FEATURE}")
  if(NOT "${_got}" STREQUAL "${EXPECTED}")
    message(SEND_ERROR
      "FEATURE-TABLE MISMATCH for targets [${TARGETS}] feature '${FEATURE}': "
      "expected [${EXPECTED}], got [${_got}]")
    math(EXPR _failures "${_failures} + 1")
    set(_failures "${_failures}" PARENT_SCOPE)
  else()
    message(STATUS "ok  targets=[${TARGETS}] ${FEATURE} -> [${_got}]")
  endif()
endfunction()

set(_ALL_FEATURES fp4-mma cutlass-nvfp4 cutlass-fp8 marlin-nvfp4 fa2)
# The fp4/cutlass/marlin fast paths are sm_12x-only bodies; FA2 is the ONE feature
# that also has an Ampere (major-8) body (the vendored `_sm80.cu` FA2 kernels,
# WA-1). Cases where an Ampere/cross-family target must resolve EMPTY therefore
# iterate this narrower list and FA2 gets its own explicit Ampere assertions.
set(_NON_FA2_FEATURES fp4-mma cutlass-nvfp4 cutlass-fp8 marlin-nvfp4)

# --- GB10, the production default. The full sm_12x capability set. -----------
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("121a" "${_f}" "121a")
endforeach()

# --- Consumer Blackwell alone (RTX 5070/5080/5090 class), BACKEND-CUDA-SM120.
# Same family, same kernel bodies; every feature must resolve exactly as 121a.
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("120a" "${_f}" "120a")
endforeach()

# --- The same-family FAT binary. Both archs keep every capability; nothing is
# dropped and nothing is enabled for an arch that did not request it.
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("120a;121a" "${_f}" "120a;121a")
endforeach()

# --- Base (non-'a') targets must DISABLE the arch-specific features. `mma.sync
# ... kind::mxf4nvf4` is rejected on base sm_120/sm_121, so enabling the define
# would emit an instruction the build cannot produce. This is the deliberate
# deviation from vLLM's cross-suffix-lenient intersection (documented in
# vt_cuda_feature_archs) and it is asserted here so it cannot silently regress.
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("120" "${_f}" "")
  expect_feature("121" "${_f}" "")
endforeach()

# --- A cross-family target has NO tactic body here and must resolve EMPTY for
# itself while leaving the sm_12x arch fully enabled. W1's source-scoped
# gencode now makes this heterogeneous list compilable without widening the
# feature body; the resolution remains the authority for which arch provides it.
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("90a;121a" "${_f}" "121a")
endforeach()
# For a single-arch cross-family sm_80 target, the sm_12x-only features resolve
# EMPTY; FA2 now resolves ENABLED for sm_80 (WA-1, Ampere body) and is asserted
# with the rest of the Ampere fan-out below.
foreach(_f IN LISTS _NON_FA2_FEATURES)
  expect_feature("80" "${_f}" "")
endforeach()

# --- Cross-family SINGLE-ARCH targets (Hopper sm_90a, datacenter Blackwell
# sm_100a, Ampere sm_80) resolve EVERY fp4/cutlass/marlin/fa2 feature to EMPTY:
# vllm.cpp has NO Hopper wgmma / sm_100 tcgen05 / Ampere kernel body, so the
# FEATURE TABLE (which lists only archs with a BUILT+VALIDATED body — deviation
# #2) names none of them. A single-arch `90a` build therefore compiles ONLY the
# portable C++/CUDA kernels for sm_90a and is a build-supported, feature-degraded
# target — NOT runtime-validated (no Hopper board here). Pinned here so a future
# table edit cannot silently claim a Hopper fast path we do not have; widening a
# cell without the matching tactic body is the exact silent-capability failure
# this suite exists to catch. See backend-matrix.md BACKEND-CUDA-SM090 and
# .agents/specs/cuda-arch-additivity.md §W9.
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("90a" "${_f}" "")
  expect_feature("100a" "${_f}" "")
  expect_feature("90" "${_f}" "")
endforeach()

# --- The CROSS-FAMILY BUILD-SUPPORTED FAN-OUT (BACKEND-CUDA-ARCH-EXPANSION,
# spec §W10). Every architecture vLLM builds kernels for that is compilable with
# the pinned nvcc 13.0 and whose portable bf16-WMMA path compiles — Ampere
# sm_80/86/89 and Jetson sm_87 (major 8), datacenter Blackwell sm_100a/sm_103a
# (major 10), and sm_110 (major 11) — resolves EVERY fp4/cutlass/marlin/fa2
# feature to EMPTY, exactly like sm_90a: none of these families has a
# BUILT+VALIDATED fast-path body here, so the FEATURE TABLE (deviation #2) names
# none of them and only the portable C++/CUDA kernels compile. These are
# BUILD-supported, portable-kernels-only, runtime-UNVERIFIED targets (no such
# board here). sm_80 (major 8) and sm_100a (major 10) were compiled end to end
# `-Werror` 0-warn on dgx as the per-major representatives; the same-major
# siblings share the identical portable kernel bodies and gencode-compatible
# SASS. Pinned so a future table edit cannot silently claim a fast path we do not
# have. See backend-matrix.md BACKEND-CUDA-SM0{80,86,87,89}/SM10{0,3}/SM110 and
# .agents/specs/cuda-arch-additivity.md §W10.
foreach(_f IN LISTS _NON_FA2_FEATURES)
  expect_feature("80" "${_f}" "")      # (already asserted above; kept for locality)
  expect_feature("86" "${_f}" "")
  expect_feature("87" "${_f}" "")
  expect_feature("89" "${_f}" "")
  expect_feature("100a" "${_f}" "")    # (already asserted above; kept for locality)
  expect_feature("103a" "${_f}" "")
  # sm_110 is the ONE exception in this fan-out: `marlin-nvfp4` was built and
  # validated on Thor silicon (see MARLIN NVFP4 SM110 below), so it is asserted
  # separately and excluded here. Every OTHER fast path stays EMPTY for 110.
  if(NOT _f STREQUAL "marlin-nvfp4")
    expect_feature("110" "${_f}" "")
  endif()
endforeach()

# --- MARLIN NVFP4 ON SM_110 (BACKEND-CUDA-SM110, issue #325). The vendored
# Marlin slice needed no source change for Thor: every __CUDA_ARCH__ guard under
# src/vt/cuda/marlin/ selects, for 1100, the same side an already-supported arch
# selects. The cell gained `11.0` only after the kernel built (403/403), ran on
# real sm_110 silicon and matched the incumbent vt::MatmulNvfp4 numerically.
# Pinned here so the enablement cannot be silently reverted, and so the sm_12x
# production resolution is proven byte-unchanged by the widened cell.
expect_feature("110" "marlin-nvfp4" "110")
expect_feature("121a" "marlin-nvfp4" "121a")
expect_feature("120a" "marlin-nvfp4" "120a")
expect_feature("120a;121a" "marlin-nvfp4" "120a;121a")
# The widened cell must not leak onto the OTHER major-11 or cross-family
# targets: `11.0` is a single exact arch, not a family claim.
expect_feature("103a" "marlin-nvfp4" "")
expect_feature("100a" "marlin-nvfp4" "")
expect_feature("80" "marlin-nvfp4" "")

# --- FA2 AMPERE ENABLEMENT (WA-1, BACKEND-CUDA-SM080/086/087/089 + COMP-FA).
# The vendored FlashAttention-2 kernel bodies are `#if __CUDA_ARCH__ >= 800`
# (src/vt/cuda/flash_attn/src/flash_fwd_split_hdim*_sm80.cu); the `fa2` FEATURE
# cell now lists 8.0,8.6,8.7,8.9,12.0a,12.1a, so a single-arch Ampere build
# resolves the feature ENABLED and compiles + emits real per-arch SASS. LABEL:
# DERIVED+BUILD-VERIFIED (testing-welcome) — no Ampere board ran it here. This is
# the mutation check that a future edit cannot silently un-support Ampere FA2, and
# cannot claim FA2 on the non-Ampere cross-family targets (which have no FA2 body).
expect_feature("80" "fa2" "80")
expect_feature("86" "fa2" "86")
expect_feature("87" "fa2" "87")
expect_feature("89" "fa2" "89")
expect_feature("100a" "fa2" "")
expect_feature("103a" "fa2" "")
expect_feature("110" "fa2" "")
# A same-major-8 FAT build keeps FA2 for every requested Ampere arch; nothing is
# dropped and no non-requested arch is enabled.
expect_feature("80;86;87;89" "fa2" "80;86;87;89")
# The sm_12x production/consumer targets are UNCHANGED by the widened cell.
expect_feature("121a" "fa2" "121a")
expect_feature("120a" "fa2" "120a")
expect_feature("120a;121a" "fa2" "120a;121a")

# --- DATACENTER-BLACKWELL sm_100a NVFP4 tcgen05 BUILD-VERIFY (BACKEND-CUDA-SM100,
# ROAD-V1-D1-CUDA). The `cutlass-nvfp4-sm100` cell is a DISTINCT feature from the
# consumer `cutlass-nvfp4` sm_12x cell: it gates the ArchTag=Sm100 tcgen05 body
# (cuda_matmul_nvfp4_sm100.cu), which is DERIVED+BUILD-VERIFIED — compiled +
# cuobjdump-proven sm_100a SASS on GB10, NO B200 board ran it. It must resolve
# ENABLED for 100a ALONE. Pinned so (a) a future edit cannot widen it onto the
# gate arch, and (b) the gate arch sm_121a / consumer 120a stay EMPTY → the sm_12x
# `cutlass-nvfp4` tactic sweep is byte-unchanged. sm_103a/sm_110 stay EMPTY
# (separate later bricks); cross-family 90a/80 stay EMPTY (no body).
expect_feature("100a" "cutlass-nvfp4-sm100" "100a")
expect_feature("121a" "cutlass-nvfp4-sm100" "")
expect_feature("120a" "cutlass-nvfp4-sm100" "")
expect_feature("120a;121a" "cutlass-nvfp4-sm100" "")
expect_feature("103a" "cutlass-nvfp4-sm100" "")
expect_feature("110" "cutlass-nvfp4-sm100" "")
expect_feature("90a" "cutlass-nvfp4-sm100" "")
expect_feature("80" "cutlass-nvfp4-sm100" "")
# RED-preserving companion: the CONSUMER sm_12x `cutlass-nvfp4` (and native
# `fp4-mma`) cells must stay EMPTY for 100a — the sm_120 tcgen05-less bodies (the
# tactic sweep + the mma.sync kind::mxf4nvf4 native path) cannot compile for a
# datacenter target. The separate sm100 cell is exactly what keeps them off it.
expect_feature("100a" "cutlass-nvfp4" "")
expect_feature("100a" "fp4-mma" "")

# --- HOPPER sm_90a CUTLASS C3x FP8 scaled-mm BUILD-VERIFY (BACKEND-CUDA-SM090,
# ROAD-V1-D1-CUDA, datacenter fast-path §9 DC2). The `scaledmm-c3x-sm90` cell is a
# DISTINCT feature from the consumer `cutlass-fp8` sm_12x cell: it gates the
# ArchTag=Sm90 wgmma/TMA body (cuda_scaled_mm_c3x_sm90.cu), which is
# DERIVED+BUILD-VERIFIED — compiled + cuobjdump-proven sm_90a SASS on GB10, NO
# H100/H200 board ran it. It must resolve ENABLED for 90a ALONE. Pinned so (a) a
# future edit cannot widen it onto the gate arch, and (b) the gate arch sm_121a /
# consumer 120a stay EMPTY → the sm_12x `cutlass-fp8` scaled-mm is byte-unchanged.
# The datacenter sm_100a stays EMPTY (its C3x FP8 leg is a separate later brick);
# cross-family 80 stays EMPTY (no body).
expect_feature("90a" "scaledmm-c3x-sm90" "90a")
expect_feature("121a" "scaledmm-c3x-sm90" "")
expect_feature("120a" "scaledmm-c3x-sm90" "")
expect_feature("120a;121a" "scaledmm-c3x-sm90" "")
expect_feature("100a" "scaledmm-c3x-sm90" "")
expect_feature("80" "scaledmm-c3x-sm90" "")
# RED-preserving companion: the CONSUMER sm_12x `cutlass-fp8` cell must stay EMPTY
# for 90a — the sm_120 scaled-mm body cannot compile for a Hopper target. The
# separate sm90 cell is exactly what keeps the sm_12x sweep off compute_90a.
expect_feature("90a" "cutlass-fp8" "")

# --- DATACENTER-BLACKWELL sm_100a CUTLASS C3x FP8 scaled-mm BUILD-VERIFY
# (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA, datacenter fast-path §9 DC3). The
# `scaledmm-c3x-sm100` cell is a DISTINCT feature from BOTH the consumer sm_12x
# `cutlass-fp8` cell AND the Hopper `scaledmm-c3x-sm90` cell: it gates the
# ArchTag=Sm100 tcgen05 body (cuda_scaled_mm_c3x_sm100.cu), which is
# DERIVED+BUILD-VERIFIED — compiled + cuobjdump-proven sm_100a SASS on GB10, NO
# B200/sm_100 board ran it. It must resolve ENABLED for 100a ALONE. Pinned so (a) a
# future edit cannot widen it onto the gate arch, and (b) the gate arch sm_121a /
# consumer 120a stay EMPTY → the sm_12x `cutlass-fp8` scaled-mm is byte-unchanged.
# The Hopper sm_90a stays EMPTY (its own separate cell); sm_103a/sm_110 stay EMPTY
# (separate later bricks); cross-family 80 stays EMPTY (no body). It coexists with
# the sm100 NVFP4 cell: both resolve for 100a (independent TUs) but neither widens
# onto the other's arch.
expect_feature("100a" "scaledmm-c3x-sm100" "100a")
expect_feature("121a" "scaledmm-c3x-sm100" "")
expect_feature("120a" "scaledmm-c3x-sm100" "")
expect_feature("120a;121a" "scaledmm-c3x-sm100" "")
expect_feature("90a" "scaledmm-c3x-sm100" "")
expect_feature("103a" "scaledmm-c3x-sm100" "")
expect_feature("110" "scaledmm-c3x-sm100" "")
expect_feature("80" "scaledmm-c3x-sm100" "")
# RED-preserving companion: the CONSUMER sm_12x `cutlass-fp8` (and the Hopper
# `scaledmm-c3x-sm90`) cells must stay EMPTY for 100a — neither the sm_120
# scaled-mm body nor the Sm90 wgmma body can compile for a datacenter tcgen05
# target. The separate sm100 cell is exactly what keeps them off compute_100a.
expect_feature("100a" "cutlass-fp8" "")
expect_feature("100a" "scaledmm-c3x-sm90" "")
# The gate arch sm_121a resolution of the OTHER cutlass cells is byte-unchanged by
# this addition (neutrality): the sm_12x consumer cells stay ENABLED for 121a and
# the sm100 NVFP4 build-verify cell stays DISABLED for 121a.
expect_feature("121a" "cutlass-fp8" "121a")
expect_feature("121a" "cutlass-nvfp4" "121a")
expect_feature("121a" "cutlass-nvfp4-sm100" "")

if(_failures GREATER 0)
  message(FATAL_ERROR "${_failures} CUDA feature-table expectation(s) failed")
endif()
message(STATUS "CUDA feature-table expectations: ALL PASS")
