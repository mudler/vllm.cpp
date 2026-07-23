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
# itself while leaving the sm_12x arch fully enabled. (Such a heterogeneous list
# still cannot COMPILE — the sources are gencode'd for the whole list — but the
# resolution must be honest about which arch provides what; see
# .agents/specs/cuda-arch-additivity.md Risks #1.)
foreach(_f IN LISTS _ALL_FEATURES)
  expect_feature("90a;121a" "${_f}" "121a")
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

if(_failures GREATER 0)
  message(FATAL_ERROR "${_failures} CUDA feature-table expectation(s) failed")
endif()
message(STATUS "CUDA feature-table expectations: ALL PASS")
