# CUDA architecture -> capability FEATURE TABLE (BACKEND-CUDA-ARCH-ADDITIVITY).
#
# PURPOSE. Before this module the build asked four separate hardcoded questions
# of the form `if(VLLM_CPP_CUDA_ARCHITECTURES MATCHES "12[01]a")` — one each for
# the native fp4 MMA define, the CUTLASS NVFP4 TUs, the CUTLASS FP8 TU and the
# vendored Marlin NVFP4 MoE GEMM. Those regexes were evaluated over the WHOLE
# arch list as one string, so a legitimate multi-arch (fat-binary) request such
# as `-DVLLM_CPP_CUDA_ARCHITECTURES="90a;121a"` still MATCHES and, worse, any
# list whose textual form drifts stops matching and SILENTLY disables fp4 /
# cutlass-nvfp4 / cutlass-fp8 / Marlin for EVERY arch in the build, including
# GB10 — a capability regression with no diagnostic at all.
#
# THE FIX. One table, `VT_CUDA_FEATURE_TABLE` below, maps each capability to the
# set of architectures that provide it. Every feature is resolved by INTERSECTING
# its supported-arch set with the requested target list, per arch, so a fat build
# enables each feature for exactly the archs that support it and reports which.
# Adding a CUDA architecture is then a TABLE-ROW edit (widen the cell) plus a
# runtime tactic registration (src/vt/cuda/cuda_arch_tactics.h) — not a hunt for
# scattered regexes.
#
# UPSTREAM. This mirrors vLLM's own build-side answer to exactly this problem.
# `cuda_archs_loose_intersection()` below is a 1:1 port of
#   /home/mudler/_git/vllm/cmake/utils.cmake:376-485 @ pin e24d1b24
# and the per-feature `cuda_archs_loose_intersection(<FEATURE>_ARCHS "<srcs>"
# "${CUDA_ARCHS}")` + `if(<FEATURE>_ARCHS)` idiom is vLLM's, e.g.
#   FP4_SM120_ARCHS   /home/mudler/_git/vllm/CMakeLists.txt:949-953,963
#   SCALED_MM_ARCHS   /home/mudler/_git/vllm/CMakeLists.txt:775-787   (sm120 fp8)
#   MARLIN_ARCHS      /home/mudler/_git/vllm/CMakeLists.txt:556-558
#   CUTLASS_MOE_DATA  /home/mudler/_git/vllm/CMakeLists.txt:918-926
# Arch spellings differ only in form: vLLM uses gencode `<major>.<minor>[af]`
# ("12.1a"), CMake's CUDA_ARCHITECTURES uses `<major><minor>[af]` ("121a"). The
# normalize/denormalize helpers below bridge the two so the table can be read
# side-by-side with upstream's lists.

# ---------------------------------------------------------------------------
# 1:1 PORT — vLLM cmake/utils.cmake:376-485 @ e24d1b24.
#
# For the given `SRC_CUDA_ARCHS` list of gencode versions in the form
#  `<major>.<minor>[letter]` compute the "loose intersection" with the
#  `TGT_CUDA_ARCHS` list of gencodes. We also support the `+PTX` suffix in
#  `SRC_CUDA_ARCHS` which indicates that the PTX code should be built when there
#  is a CUDA_ARCH in `TGT_CUDA_ARCHS` that is equal to or larger than the
#  architecture in `SRC_CUDA_ARCHS`.
# The loose intersection is defined as:
#   { max{ x \in tgt | x <= y } | y \in src, { x \in tgt | x <= y } != {} }
#  where `<=` is the version comparison operator.
# The result is stored in `OUT_CUDA_ARCHS`.
#
# Example:
#   SRC_CUDA_ARCHS="7.5;8.0;8.6;9.0;9.0a"
#   TGT_CUDA_ARCHS="8.0;8.9;9.0"
#   -> OUT_CUDA_ARCHS="8.0;8.6;9.0;9.0a"
# ---------------------------------------------------------------------------
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  set(_SRC_CUDA_ARCHS "${SRC_CUDA_ARCHS}")
  set(_TGT_CUDA_ARCHS ${TGT_CUDA_ARCHS})

  # handle +PTX suffix: separate base arch for matching, record PTX requests
  set(_PTX_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "\\+PTX$")
      string(REPLACE "+PTX" "" _base "${_arch}")
      list(APPEND _PTX_ARCHS "${_base}")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      list(APPEND _SRC_CUDA_ARCHS "${_base}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _PTX_ARCHS)
  list(REMOVE_DUPLICATES _SRC_CUDA_ARCHS)

  # Handle architecture-specific suffixes (a/f) for SRC entries.
  # First try exact base match (x.y), then cross-suffix match (x.ya / x.yf).
  # For 'f' (family) suffix: if no exact/cross match, fall back to major-version
  # match — e.g. SRC="12.0f" matches TGT="12.1a" since SM121 is in the SM12x
  # family. The output uses TGT's value to preserve the user's compilation flags.
  set(_CUDA_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "[af]$")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      string(REGEX REPLACE "[af]$" "" _base "${_arch}")
      if ("${_base}" IN_LIST TGT_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}")
        list(APPEND _CUDA_ARCHS "${_arch}")
      elseif("${_base}a" IN_LIST _TGT_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}a")
        list(APPEND _CUDA_ARCHS "${_base}a")
      elseif("${_base}f" IN_LIST _TGT_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}f")
        list(APPEND _CUDA_ARCHS "${_base}f")
      elseif(_arch MATCHES "f$")
        # Family suffix: match any TGT entry in the same major version family.
        string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" _src_major "${_base}")
        foreach(_tgt ${_TGT_CUDA_ARCHS})
          string(REGEX REPLACE "[af]$" "" _tgt_base "${_tgt}")
          string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" _tgt_major "${_tgt_base}")
          if(_tgt_major STREQUAL _src_major)
            list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_tgt}")
            list(APPEND _CUDA_ARCHS "${_tgt}")
            break()
          endif()
        endforeach()
      endif()
    endif()
  endforeach()

  # Symmetric handling: if TGT has x.ya/f and SRC has x.y (without suffix),
  # preserve TGT's suffix in the output.
  set(_tgt_copy ${_TGT_CUDA_ARCHS})
  foreach(_arch ${_tgt_copy})
    if(_arch MATCHES "[af]$")
      string(REGEX REPLACE "[af]$" "" _base "${_arch}")
      if ("${_base}" IN_LIST _SRC_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_arch}")
        list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_base}")
        list(APPEND _CUDA_ARCHS "${_arch}")
      endif()
    endif()
  endforeach()

  list(SORT _SRC_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)

  # for each ARCH in TGT_CUDA_ARCHS find the highest arch in SRC_CUDA_ARCHS that
  # is less or equal to ARCH (but has the same major version since SASS binary
  # compatibility is only forward compatible within the same major version).
  foreach(_ARCH ${_TGT_CUDA_ARCHS})
    set(_TMP_ARCH)
    # Extract the major version of the target arch
    string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" TGT_ARCH_MAJOR "${_ARCH}")
    foreach(_SRC_ARCH ${_SRC_CUDA_ARCHS})
      # Extract the major version of the source arch
      string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" SRC_ARCH_MAJOR "${_SRC_ARCH}")
      # Check version-less-or-equal, and allow PTX arches to match across majors
      if (_SRC_ARCH VERSION_LESS_EQUAL _ARCH)
        if (_SRC_ARCH IN_LIST _PTX_ARCHS OR SRC_ARCH_MAJOR STREQUAL TGT_ARCH_MAJOR)
          set(_TMP_ARCH "${_SRC_ARCH}")
        endif()
      else()
        # If we hit a version greater than the target, we can break
        break()
      endif()
    endforeach()

    # If we found a matching _TMP_ARCH, append it to _CUDA_ARCHS
    if (_TMP_ARCH)
      list(APPEND _CUDA_ARCHS "${_TMP_ARCH}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHS)

  # reapply +PTX suffix to architectures that requested PTX
  set(_FINAL_ARCHS)
  foreach(_arch ${_CUDA_ARCHS})
    if(_arch IN_LIST _PTX_ARCHS)
      list(APPEND _FINAL_ARCHS "${_arch}+PTX")
    else()
      list(APPEND _FINAL_ARCHS "${_arch}")
    endif()
  endforeach()
  set(_CUDA_ARCHS ${_FINAL_ARCHS})

  set(${OUT_CUDA_ARCHS} ${_CUDA_ARCHS} PARENT_SCOPE)
endfunction()
# --------------------------- end 1:1 port ----------------------------------

# CMake CUDA_ARCHITECTURES form ("121a", "90a", "80", "100f") -> vLLM gencode
# form ("12.1a", "9.0a", "8.0", "10.0f"). The minor version is always the last
# digit; everything before it is the major.
function(vt_cuda_archs_normalize OUT_VAR IN_ARCHS)
  set(_out)
  foreach(_a IN LISTS IN_ARCHS)
    string(STRIP "${_a}" _a)
    if(_a STREQUAL "")
      continue()
    endif()
    if(_a MATCHES "^([0-9]+)([0-9])([af]?)$")
      list(APPEND _out "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
    elseif(_a MATCHES "^[0-9]+\\.[0-9]+[af]?$")
      list(APPEND _out "${_a}")  # already normalized
    else()
      message(FATAL_ERROR
        "VLLM_CPP_CUDA_ARCHITECTURES: cannot parse CUDA architecture '${_a}'. "
        "Use CMake's CUDA_ARCHITECTURES form, e.g. 121a, 120a, 90a, 100f, 80.")
    endif()
  endforeach()
  set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

# vLLM gencode form -> CMake CUDA_ARCHITECTURES form (for STATUS messages).
function(vt_cuda_archs_denormalize OUT_VAR IN_ARCHS)
  set(_out)
  foreach(_a IN LISTS IN_ARCHS)
    string(REPLACE "+PTX" "" _a "${_a}")
    string(REPLACE "." "" _a "${_a}")
    list(APPEND _out "${_a}")
  endforeach()
  set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

# ===========================================================================
# THE FEATURE TABLE — arch -> capability set.
#
# Each row is `FEATURE|SUPPORTED_ARCHS|DESCRIPTION`. SUPPORTED_ARCHS is in vLLM
# gencode form (comma-separated inside the row so the row itself stays one CMake
# list element) and lists the architectures for which WE HAVE A KERNEL BODY that
# has been built and validated. It is deliberately NOT vLLM's full supported set:
# a green fatbinary link is not execution evidence, and this project only claims
# an arch once a tactic exists for it. The `upstream` note on each row records
# vLLM's superset so widening a cell is a mechanical follow-up once the matching
# tactic body lands (see src/vt/cuda/cuda_arch_tactics.h).
#
# TO ADD A CUDA ARCHITECTURE: widen the cells whose kernels you ported, add the
# numeric target to VLLM_CPP_CUDA_ARCHITECTURES, and register the arch's tactic
# in the runtime registry. No other build edit is required.
# ===========================================================================
set(VT_CUDA_FEATURE_TABLE
  # native block-scaled fp4xfp4 mma.sync kind::mxf4nvf4 (cuda_matmul_nvfp4.cu).
  # upstream: vLLM FP4_SM120_ARCHS "12.0a;12.1a" (CMakeLists.txt:951).
  "fp4-mma|12.0a,12.1a|native block-scaled fp4xfp4 MMA (VT_FP4_MMA_SM120A)"
  # CUTLASS NVFP4 block-scaled GEMM TUs (cuda_matmul_nvfp4_cutlass.cu + tactics).
  # upstream: vLLM FP4_SM120_ARCHS "12.0a;12.1a" (CMakeLists.txt:951).
  "cutlass-nvfp4|12.0a,12.1a|CUTLASS NVFP4 block-scaled GEMM (VT_CUTLASS_NVFP4)"
  # CUTLASS FP8 scaled-mm (cuda_matmul_fp8_cutlass.cu, ArchTag=Sm120).
  # upstream: vLLM sm120 SCALED_MM_ARCHS "12.0a;12.1a" (CMakeLists.txt:777).
  "cutlass-fp8|12.0a,12.1a|CUTLASS FP8 scaled-mm (VT_CUTLASS_FP8)"
  # Vendored Marlin NVFP4 W4A16 grouped-MoE GEMM (src/vt/cuda/marlin/).
  # upstream: vLLM MARLIN_ARCHS "8.0+PTX;12.0a;12.1a" (CMakeLists.txt:558) — the
  # sm80+PTX leg is NOT claimed here: our vendored slice is the bf16 NVFP4
  # instantiation only and has never been built or run outside sm_12x.
  "marlin-nvfp4|12.0a,12.1a|vendored Marlin NVFP4 W4A16 MoE GEMM (VT_MARLIN_NVFP4)"
  # Vendored FlashAttention-2 prefill/decode split-KV kernels.
  # upstream: vllm-project/flash-attention builds sm80..sm90; our vendored slice
  # is compiled and validated for sm_12x only (cuda_flash_attn_fa2.cu:428).
  "fa2|12.0a,12.1a|vendored FlashAttention-2 prefill/decode (VLLM_CPP_FLASH_ATTN)")

# vt_cuda_feature_archs(<OUT_ARCHS> <FEATURE>)
#   Resolves FEATURE against the requested VLLM_CPP_CUDA_ARCHITECTURES and sets
#   OUT_ARCHS (CMake CUDA_ARCHITECTURES form) to the subset of requested archs
#   that provide it — empty when no requested arch does. Mirrors vLLM's
#   `cuda_archs_loose_intersection(<F>_ARCHS "<srcs>" "${CUDA_ARCHS}")`.
function(vt_cuda_feature_archs OUT_ARCHS FEATURE)
  set(_src "")
  set(_found OFF)
  foreach(_row IN LISTS VT_CUDA_FEATURE_TABLE)
    string(REPLACE "|" ";" _cells "${_row}")
    list(GET _cells 0 _name)
    if(_name STREQUAL "${FEATURE}")
      list(GET _cells 1 _src)
      string(REPLACE "," ";" _src "${_src}")
      set(_found ON)
      break()
    endif()
  endforeach()
  if(NOT _found)
    message(FATAL_ERROR "vt_cuda_feature_archs: unknown CUDA feature '${FEATURE}' "
      "(add a row to VT_CUDA_FEATURE_TABLE in cmake/CudaArchFeatures.cmake)")
  endif()
  vt_cuda_archs_normalize(_tgt "${VLLM_CPP_CUDA_ARCHITECTURES}")
  cuda_archs_loose_intersection(_hit "${_src}" "${_tgt}")
  # DEVIATION FROM UPSTREAM (recorded, deliberate). vLLM's loose intersection is
  # cross-suffix LENIENT: SRC "12.1a" matches TGT "12.1" and the result carries
  # the 'a'. vLLM can afford that because it emits per-source `-gencode` flags
  # itself, so it re-adds the arch-specific target it just decided it needs. We
  # pass VLLM_CPP_CUDA_ARCHITECTURES straight through to CMAKE_CUDA_ARCHITECTURES,
  # so a target the user did NOT request is never compiled — and for our fp4 rows
  # the 'a' suffix is LOAD-BEARING: `mma.sync ... kind::mxf4nvf4` is rejected on
  # base sm_121 (the reason VLLM_CPP_CUDA_ARCHITECTURES defaults to "121a"). So
  # keep only hits that are literally among the requested targets; a base-arch
  # request reports the feature DISABLED rather than enabling a define whose
  # instruction the build will not emit.
  set(_kept)
  foreach(_h IN LISTS _hit)
    string(REPLACE "+PTX" "" _h_base "${_h}")
    if("${_h_base}" IN_LIST _tgt)
      list(APPEND _kept "${_h}")
    endif()
  endforeach()
  vt_cuda_archs_denormalize(_hit_cmake "${_kept}")
  set(${OUT_ARCHS} "${_hit_cmake}" PARENT_SCOPE)
endfunction()

# vt_cuda_report_feature(<FEATURE> <RESOLVED_ARCHS>)
#   Emits the configure-time capability report. This STATUS output is the
#   evidence surface for the multi-arch gate: it names, per feature, exactly
#   which of the requested archs provide it, so a fat build can never silently
#   drop a capability the way the old whole-list regexes did.
function(vt_cuda_report_feature FEATURE RESOLVED_ARCHS)
  vt_cuda_archs_normalize(_tgt "${VLLM_CPP_CUDA_ARCHITECTURES}")
  vt_cuda_archs_denormalize(_tgt_cmake "${_tgt}")
  if(RESOLVED_ARCHS)
    message(STATUS "  CUDA feature ${FEATURE}: ENABLED for [${RESOLVED_ARCHS}]")
    # A strict subset means the fat build contains archs with no tactic for this
    # feature. Today that is a LOUD build-time compile failure on the untargeted
    # arch (there are no cross-family tactic bodies yet), never a silent capability
    # drop. Narrowing per-source gencode (vLLM's set_gencode_flags_for_srcs,
    # cmake/utils.cmake:265-345) lands with the first cross-family tactic.
    set(_missing "${_tgt_cmake}")
    list(REMOVE_ITEM _missing ${RESOLVED_ARCHS})
    if(_missing)
      message(WARNING
        "CUDA feature '${FEATURE}' has no tactic for requested arch(es) [${_missing}]. "
        "It stays ENABLED for [${RESOLVED_ARCHS}]; the sources are still compiled for the "
        "whole target list, so building this heterogeneous fat binary requires the per-arch "
        "tactic bodies (and per-source gencode narrowing) first. See "
        ".agents/specs/cuda-arch-additivity.md.")
    endif()
  else()
    message(STATUS "  CUDA feature ${FEATURE}: DISABLED (no requested arch in [${_tgt_cmake}] provides it)")
  endif()
endfunction()
