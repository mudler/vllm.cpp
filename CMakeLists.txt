cmake_minimum_required(VERSION 3.24)
project(vllm_cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Host C++ floating point is pinned to IEEE as-written semantics: no FMA
# contraction. GCC/Clang default to -ffp-contract=fast, which lets the compiler
# fuse `a*b + c` into a single-rounding fma PER LOOP, PER FUNCTION — so two
# textually identical reductions (e.g. the vt::RmsNorm golden vs the FusedChain
# Tier-1 interpreter, both `sumsq += v*v`) can compile one contracted and one
# not, breaking the bit-identity contracts the vt:: CPU reference kernels and
# their tests rely on (test_ops_fused_chain, the fused-vs-split fp8 pairs, ...).
# This bites wherever FMA is baseline ISA (aarch64 always; x86 with -march=*).
# Pinning contract=off gives every TU (kernels AND tests) the exact numerics of
# the plain x86-64 Release build on every platform. CUDA device code is NOT
# affected (nvcc flags are separate; GPU parity tests compare GPU-vs-GPU).
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

option(VLLM_CPP_BUILD_TESTS "Build tests" ON)
option(VLLM_CPP_BUILD_EXAMPLES "Build examples" ON)
# The OpenAI HTTP server (M3.1 Task 4) is gated so the core build stays lean when
# the vendored cpp-httplib transport header is absent. cpp-httplib is a
# header-only MIT HTTP TRANSPORT lib (llama.cpp's choice) — a transport-dep
# deviation, NOT a compute/ML dependency (consistent with the no-pytorch/no-ggml
# rule).
option(VLLM_CPP_SERVER "Build the OpenAI HTTP server (needs third_party/httplib/httplib.h)" ON)
if(VLLM_CPP_SERVER AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/httplib/httplib.h")
  message(WARNING "VLLM_CPP_SERVER=ON but third_party/httplib/httplib.h is missing; "
    "disabling the server target. Vendor cpp-httplib to enable it.")
  set(VLLM_CPP_SERVER OFF)
endif()
set(VLLM_CPP_CUDA "AUTO" CACHE STRING "Build the CUDA backend: ON, OFF, or AUTO (detect)")
# Tri-state exactly like VLLM_CPP_CUDA above (BACKEND-METAL-MLX W0). AUTO means
# "on when the host is Apple and OBJCXX is usable" — i.e. it turns itself on for
# a macOS developer and is invisible everywhere else. Resolved to a plain ON/OFF
# a few lines below the CUDA resolution.
set(VLLM_CPP_METAL "AUTO" CACHE STRING "Build the Metal backend: ON, OFF, or AUTO (detect)")
# Tri-state like the two above (BACKEND-VULKAN W0), but AUTO deliberately
# resolves OFF rather than probing for a Vulkan loader. Rationale, recorded
# because it is a DIFFERENT choice from Metal's AUTO: Metal AUTO-enables on an
# Apple host because there is no other GPU backend there, so turning it on cannot
# perturb anything. Vulkan OVERLAPS CUDA on the very box the CUDA gate models run
# on (GB10 enumerates as a Vulkan device), and silently enabling a second GPU
# backend in the gate build would register kVULKAN into the op/platform tables
# and into tests/vt/test_backend_cross_device.cpp during the CUDA regressions it
# must not touch. So Vulkan is opt-in: -DVLLM_CPP_VULKAN=ON.
set(VLLM_CPP_VULKAN "AUTO" CACHE STRING "Build the Vulkan backend: ON, OFF, or AUTO (=OFF)")
# Tri-state like the three above (BACKEND-ROCM W0). AUTO resolves OFF for the
# same reason Vulkan's does, plus one specific to this skeleton: it has NEVER
# BEEN COMPILED by anyone (no AMD GPU on the authoring machine), so AUTO-enabling
# it wherever /opt/rocm happens to exist would turn a green build red on a
# machine whose owner asked for nothing. Opt in with -DVLLM_CPP_HIP=ON, which is
# exactly what a contributor on issue #41 is asked to do first.
set(VLLM_CPP_HIP "AUTO" CACHE STRING "Build the ROCm/HIP backend: ON, OFF, or AUTO (=OFF)")
# Target gfx architecture(s), semicolon-separated, e.g. "gfx1100" or
# "gfx1100;gfx1151". EMPTY (the default) means: pass no --offload-arch at all and
# let hipcc target the installed GPU. That is the right default for a contributor
# building on the machine they will run on, which is every case today. The
# validated set is upstream's HIP_SUPPORTED_ARCHS (vllm/CMakeLists.txt:52 @ pin
# 555967922); a value outside it configures with a WARNING rather than an error,
# because a newer ROCm may know an arch that list does not.
set(VLLM_CPP_HIP_ARCHITECTURES "" CACHE STRING
  "Target gfx arch(es) for the ROCm backend, e.g. gfx1100 (empty = hipcc default)")
# ROCm installation prefix. Mirrors upstream's ROCM_PATH (vllm/CMakeLists.txt:54-60)
# with the same default and the same override, so a TheRock/nightly install in a
# non-standard prefix is one flag away.
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
# The MLX acceleration PROVIDER for the Metal dense GEMM (BACKEND-ACCEL-PROVIDER
# work row M5, .agents/specs/metal-mlx-reuse-study.md §6.2). Default OFF and
# deliberately NOT tri-state: unlike a PLATFORM, a provider is a CONFIGURATION
# CHOICE, and it drags in a ~19 MB libmlx.dylib plus a ~105 MB mlx.metallib —
# a real deviation from discipline.md's header-only preference, accepted only as
# an opt-in exactly like VLLM_CPP_TRITON. Native MSL stays the default and is
# what ships. Point -DMLX_ROOT at an MLX install (the pip wheel's
# `site-packages/mlx` directory works: it ships include/, lib/ and the metallib).
set(VLLM_CPP_MLX OFF CACHE BOOL "Build the optional MLX GEMM provider for the Metal backend")
set(MLX_ROOT "" CACHE PATH "Root of an MLX install (include/ + lib/) for VLLM_CPP_MLX")
# Supported values today are the sm_12x (consumer/GB10 Blackwell) family:
#   121a          GB10 / DGX Spark — the default and the only arch with a GATED
#                 runtime here (every gate model, every benchmark).
#   120a          consumer Blackwell (RTX 5070/5080/5090 class). Same family, same
#                 kernel bodies, same FEATURE-TABLE cells; BUILD-supported and
#                 verified to emit sm_120a SASS, but NOT runtime-gated here — no
#                 sm_120 board exists on this hardware. See backend-matrix.md
#                 BACKEND-CUDA-SM120 for exactly what is and is not proven.
#   120a;121a     same-family FAT binary; builds clean and carries both SASS
#                 targets. NOTE: incompatible with -DVLLM_CPP_TRITON=ON, because a
#                 vendored AOT cubin is single-arch (TritonAOT.cmake explains and
#                 diagnoses this); the portable CUDA kernels cover the fat build.
#   90a           Hopper (H100/H200) — a DIFFERENT family. Single-arch only, and
#                 PORTABLE-KERNELS-ONLY: all five accelerated features resolve
#                 DISABLED (we have no Hopper wgmma/TMA kernel body, so no
#                 FEATURE-TABLE cell names 9.0a), so only the portable C++/CUDA
#                 kernels compile — build-supported, NOT vLLM-competitive, and
#                 runtime-UNPROVEN (no Hopper board here). See backend-matrix.md
#                 BACKEND-CUDA-SM090 and cuda-arch-additivity.md §W9.
#   80 86 87 89   Ampere (A100 / consumer 30-series / Jetson Orin) and Ada
#   100a 103a     (sm_89), datacenter Blackwell (sm_100/sm_103), and sm_110 — the
#   110           CROSS-FAMILY BUILD-SUPPORTED FAN-OUT (spec §W10). Single-arch,
#                 PORTABLE-KERNELS-ONLY exactly like 90a: every fp4/cutlass/
#                 marlin/fa2 feature resolves DISABLED (no fast-path body for
#                 these families), so only the portable C++/CUDA kernels compile.
#                 BUILD-supported and runtime-UNVERIFIED (no such board here).
#                 sm_80 and sm_100a were compiled end to end `-Werror` 0-warn on
#                 dgx as the per-major representatives; the same-major siblings
#                 (86/87/89, 103a) share the identical portable bodies. sm_70
#                 (Volta, dropped by nvcc 13) and sm_75 (Turing, no bf16 tensor
#                 cores → the portable bf16-WMMA path will not compile) are NOT
#                 build-supported — they need a real fp16/non-tensor-core kernel
#                 body, a scoped follow-up. See backend-matrix.md
#                 BACKEND-CUDA-SM0{80,86,87,89}/SM10{0,3}/SM110 and §W10.
# A cross-FAMILY FAT list (e.g. "90a;121a") resolves its features honestly but
# does NOT compile — the sources are gencode'd for the whole list, so the sm12x
# fp4 PTX is rejected on compute_90a; per-source gencode narrowing (spec W7) is
# what a fat cross-family binary needs. See .agents/specs/cuda-arch-additivity.md.
set(VLLM_CPP_CUDA_ARCHITECTURES "121a" CACHE STRING "Target CUDA architectures (GB10=121a, consumer Blackwell=120a, same-family fat binary=120a;121a; the 'a' arch-specific target is REQUIRED for the native block-scaled fp4xfp4 MMA — mma.sync kind::mxf4nvf4 is rejected on base sm_120/sm_121)")
option(VLLM_CPP_BENCH_PROFILE_CONTROL
  "Compile trace-only SIGUSR2/cudaProfiler replay control plus FIFO graceful shutdown (never for production timing builds)"
  OFF)

# Resolve the tri-state VLLM_CPP_CUDA into a plain ON/OFF. The normal variable
# set below intentionally shadows the cache entry so later if(VLLM_CPP_CUDA)
# checks see the resolved value while the user's cached choice is preserved.
include(CheckLanguage)
if(VLLM_CPP_CUDA STREQUAL "AUTO")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolchain found; enabling CUDA backend")
    set(VLLM_CPP_CUDA ON)
  else()
    set(VLLM_CPP_CUDA OFF)
  endif()
elseif(VLLM_CPP_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "VLLM_CPP_CUDA=ON but no CUDA compiler was found. "
      "Install the CUDA toolkit or configure with -DVLLM_CPP_CUDA=OFF (or AUTO).")
  endif()
  set(VLLM_CPP_CUDA ON)
else()
  set(VLLM_CPP_CUDA OFF)
endif()

# ─── Host instrumentation lanes ───────────────────────────────────────────────
# One default build is shallow coverage: it reaches neither the dynamic
# memory/UB detectors nor the race detector, so a whole defect class (bad index
# arithmetic, use-after-free across the engine's owning containers, unsynchronized
# access from the serving threads) can only be caught by a failing parity test,
# if at all. This selects an instrumentation lane for the HOST C++ build so those
# detectors run as their own CI jobs alongside the plain build.
#
# ASan/UBSan and TSan are deliberately SEPARATE lanes: their runtimes are
# mutually exclusive and their environment requirements differ. The lane is
# HOST-ONLY — nvcc-compiled device TUs are not instrumented, and a host sanitizer
# runtime reports false positives against the CUDA driver — so anything other
# than OFF requires the resolved VLLM_CPP_CUDA to be OFF. The CUDA tier's
# equivalent detector is compute-sanitizer, run against the normal build.
#
# -fno-sanitize-recover=all makes a finding ABORT instead of print-and-continue,
# so ctest FAILS the lane rather than burying the report inside passing output.
# Frame pointers plus -g1 keep file/line traces attributable without emitting
# full type/local-variable DWARF. Every test statically links the instrumented
# library, so full -g duplicates hundreds of MiB into each executable; the
# ASan+UBSan GitHub runner exhausted its filesystem before the suite finished
# linking. The lane keeps the project's -ffp-contract=off numerics so a
# bit-identity test still means what it means in the production build.
set(VLLM_CPP_SANITIZE "OFF" CACHE STRING
  "Host sanitizer lane: OFF | address | undefined | address,undefined | thread")
set_property(CACHE VLLM_CPP_SANITIZE PROPERTY STRINGS
  OFF address undefined address,undefined thread)
if(NOT VLLM_CPP_SANITIZE STREQUAL "OFF")
  set(_vllm_cpp_sanitize_allowed address undefined address,undefined thread)
  if(NOT VLLM_CPP_SANITIZE IN_LIST _vllm_cpp_sanitize_allowed)
    message(FATAL_ERROR
      "VLLM_CPP_SANITIZE='${VLLM_CPP_SANITIZE}' is not one of: "
      "OFF;${_vllm_cpp_sanitize_allowed}")
  endif()
  if(VLLM_CPP_CUDA)
    message(FATAL_ERROR
      "VLLM_CPP_SANITIZE requires VLLM_CPP_CUDA=OFF: the host sanitizer runtime "
      "does not instrument nvcc device TUs and reports false positives against "
      "the CUDA driver. Use compute-sanitizer for the CUDA tier instead.")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${VLLM_CPP_SANITIZE}>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-g1>)
  # -Werror itself is dropped for this lane in cmake/CompilerWarnings.cmake, where
  # the per-target warning flags are set; see the rationale there.
  add_link_options(-fsanitize=${VLLM_CPP_SANITIZE})
  message(STATUS "Host sanitizer lane: ${VLLM_CPP_SANITIZE} (CUDA backend off)")
endif()

# Resolve the tri-state VLLM_CPP_METAL the same way (BACKEND-METAL-MLX W0). The
# Metal backend is ObjC++ against the Metal/Foundation frameworks, so it needs
# APPLE plus a working OBJCXX compiler; on any other host AUTO resolves OFF and
# nothing about the build changes.
if(VLLM_CPP_METAL STREQUAL "AUTO")
  if(APPLE)
    check_language(OBJCXX)
    if(CMAKE_OBJCXX_COMPILER)
      message(STATUS "Apple host with an ObjC++ compiler; enabling Metal backend")
      set(VLLM_CPP_METAL ON)
    else()
      set(VLLM_CPP_METAL OFF)
    endif()
  else()
    set(VLLM_CPP_METAL OFF)
  endif()
elseif(VLLM_CPP_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "VLLM_CPP_METAL=ON but this is not an Apple host. "
      "Configure with -DVLLM_CPP_METAL=OFF (or AUTO).")
  endif()
  check_language(OBJCXX)
  if(NOT CMAKE_OBJCXX_COMPILER)
    message(FATAL_ERROR "VLLM_CPP_METAL=ON but no ObjC++ compiler was found. "
      "Install the Xcode Command Line Tools, or configure with -DVLLM_CPP_METAL=OFF.")
  endif()
  set(VLLM_CPP_METAL ON)
else()
  set(VLLM_CPP_METAL OFF)
endif()
# Resolve the tri-state VLLM_CPP_VULKAN (BACKEND-VULKAN W0). There is nothing to
# DETECT at configure time: the backend needs no SDK, no shader compiler and no
# link-time library — the Khronos headers are vendored (third_party/vulkan/), the
# SPIR-V is committed (src/vt/vulkan/vulkan_spirv.h), and libvulkan.so.1 is
# opened with dlopen at run time. Whether a Vulkan DEVICE exists is therefore a
# RUNTIME question, answered by vt::vulkan::VulkanContext::Available(), which is
# what decides whether kVULKAN registers. So ON always succeeds and AUTO means
# OFF (see the option's comment for why AUTO does not probe).
if(VLLM_CPP_VULKAN STREQUAL "AUTO")
  set(VLLM_CPP_VULKAN OFF)
elseif(VLLM_CPP_VULKAN)
  set(VLLM_CPP_VULKAN ON)
else()
  set(VLLM_CPP_VULKAN OFF)
endif()

# Resolve the tri-state VLLM_CPP_HIP (BACKEND-ROCM W0). Unlike Vulkan there IS
# something to detect — a HIP toolchain — but AUTO still resolves OFF (see the
# option's comment). ON is where detection happens, and it FAILS LOUDLY rather
# than silently downgrading: a contributor who typed -DVLLM_CPP_HIP=ON and got a
# green CPU-only build would have no way to notice the backend was not built,
# which is the exact failure mode the CUTLASS note in docs/BUILD.md exists for.
if(VLLM_CPP_HIP STREQUAL "AUTO")
  set(VLLM_CPP_HIP OFF)
elseif(VLLM_CPP_HIP)
  # CMake's own HIP language support (>= 3.21; we require 3.24) understands .hip
  # sources, which is why upstream calls enable_language(HIP) explicitly rather
  # than leaning on torch's setup (vllm/CMakeLists.txt:196-202).
  include(CheckLanguage)
  check_language(HIP)
  if(NOT CMAKE_HIP_COMPILER)
    message(FATAL_ERROR
      "VLLM_CPP_HIP=ON but no HIP compiler was found.\n"
      "  Install ROCm (hipcc), or point at it with -DROCM_PATH=<prefix> / "
      "-DCMAKE_HIP_COMPILER=<path to hipcc>, or configure with -DVLLM_CPP_HIP=OFF.\n"
      "  Looked with ROCM_PATH=${ROCM_PATH}.")
  endif()
  enable_language(HIP)
  # Matches upstream's CMAKE_HIP_STANDARD 20 (vllm/CMakeLists.txt:20-21). The vt
  # headers the .hip TUs include are C++20.
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  # The same IEEE-as-written pin every other language in this build gets. It is
  # NOT cosmetic here: tests/vt/test_backend_cross_device.cpp compares this
  # backend against the CPU kernels at NMSE <= 5e-4, and a contracted multiply-add
  # in the RmsNorm reduction is exactly the kind of drift that budget is meant to
  # catch rather than absorb.
  add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
  set(VLLM_CPP_HIP ON)
else()
  set(VLLM_CPP_HIP OFF)
endif()

if(VLLM_CPP_HIP)
  # Upstream's validated arch list, verbatim (vllm/CMakeLists.txt:52 @ pin
  # 555967922). Used only to WARN: an unknown value is passed through to hipcc,
  # which is the component that actually knows what it supports.
  set(VLLM_CPP_HIP_SUPPORTED_ARCHS
    "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
  foreach(arch IN LISTS VLLM_CPP_HIP_ARCHITECTURES)
    if(NOT arch IN_LIST VLLM_CPP_HIP_SUPPORTED_ARCHS)
      message(WARNING "VLLM_CPP_HIP_ARCHITECTURES contains '${arch}', which is not in "
        "upstream vLLM's validated set. Passing it to hipcc anyway.")
    endif()
  endforeach()
  if(VLLM_CPP_HIP_ARCHITECTURES)
    message(STATUS "ROCm backend: ENABLED for arch(es) [${VLLM_CPP_HIP_ARCHITECTURES}]")
  else()
    message(STATUS "ROCm backend: ENABLED, hipcc default arch (set "
      "-DVLLM_CPP_HIP_ARCHITECTURES=gfx#### to target explicitly)")
  endif()
  message(STATUS "ROCm backend: NEVER COMPILED UPSTREAM OF YOU — this skeleton has no "
    "build report from any machine. Please report success or failure on "
    "https://github.com/mudler/vllm.cpp/issues/41")
endif()

if(VLLM_CPP_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  # Same IEEE-as-written pin the CXX sources get (see the -ffp-contract note at
  # the top of this file): the Metal HOST code shares the vt:: numeric contracts.
  add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-ffp-contract=off>)
endif()

if(VLLM_CPP_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES ${VLLM_CPP_CUDA_ARCHITECTURES})
  # Per-arch capability resolution (BACKEND-CUDA-ARCH-ADDITIVITY). Every
  # arch-gated feature below asks the FEATURE TABLE instead of regex-matching the
  # whole arch list, so adding an architecture cannot silently disable fp4 /
  # cutlass-nvfp4 / cutlass-fp8 / Marlin / FA2 for the archs that DO support them.
  # Mirrors vLLM's cuda_archs_loose_intersection idiom; see the module header.
  include(cmake/CudaArchFeatures.cmake)
  message(STATUS "CUDA target architectures: ${VLLM_CPP_CUDA_ARCHITECTURES}")
  vt_cuda_feature_archs(VT_FP4_MMA_ARCHS       "fp4-mma")
  vt_cuda_feature_archs(VT_CUTLASS_NVFP4_ARCHS "cutlass-nvfp4")
  vt_cuda_feature_archs(VT_CUTLASS_NVFP4_SM100_ARCHS "cutlass-nvfp4-sm100")
  vt_cuda_feature_archs(VT_CUTLASS_FP8_ARCHS   "cutlass-fp8")
  vt_cuda_feature_archs(VT_SCALEDMM_C3X_SM90_ARCHS "scaledmm-c3x-sm90")
  vt_cuda_feature_archs(VT_SCALEDMM_C3X_SM100_ARCHS "scaledmm-c3x-sm100")
  vt_cuda_feature_archs(VT_MARLIN_NVFP4_ARCHS  "marlin-nvfp4")
  vt_cuda_feature_archs(VT_FA2_ARCHS           "fa2")
  vt_cuda_report_feature("fp4-mma"       "${VT_FP4_MMA_ARCHS}")
  vt_cuda_report_feature("cutlass-nvfp4" "${VT_CUTLASS_NVFP4_ARCHS}")
  vt_cuda_report_feature("cutlass-nvfp4-sm100" "${VT_CUTLASS_NVFP4_SM100_ARCHS}")
  vt_cuda_report_feature("cutlass-fp8"   "${VT_CUTLASS_FP8_ARCHS}")
  vt_cuda_report_feature("scaledmm-c3x-sm90" "${VT_SCALEDMM_C3X_SM90_ARCHS}")
  vt_cuda_report_feature("scaledmm-c3x-sm100" "${VT_SCALEDMM_C3X_SM100_ARCHS}")
  vt_cuda_report_feature("marlin-nvfp4"  "${VT_MARLIN_NVFP4_ARCHS}")
  vt_cuda_report_feature("fa2"           "${VT_FA2_ARCHS}")
  # The native block-scaled fp4xfp4 MMA (cuda_matmul_nvfp4.cu) needs the
  # architecture-specific sm_12xa target. Advertise it to the code so the host
  # launcher enables the native path only when it was actually compiled in.
  if(VT_FP4_MMA_ARCHS)
    add_compile_definitions(VT_FP4_MMA_SM120A=1)
  endif()
  # CUTLASS (header-only, >=v4.5.0) for the sm120a NVFP4 block-scaled fp4xfp4 GEMM
  # drop-in (cuda_matmul_nvfp4_cutlass.cu). Vendored slice under third_party/cutlass
  # by default; override with -DVLLM_CPP_CUTLASS_DIR=<cutlass root> (must contain
  # include/ + tools/util/include/). The TU is enabled only when the dir exists and
  # the arch is sm_12xa (the block-scaled fp4 collective is Blackwell-only).
  set(VLLM_CPP_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass"
      CACHE PATH "CUTLASS source root (include/ + tools/util/include/)")
  # Opt-in FetchContent fallback. FlashInfer 0.6.12's installed runtime bundles
  # CUTLASS 4.5.0; its SM12 scale-factor TMA changes are required for the traced
  # 128x32/128x64 tactics (4.4.2 rejects them at compile time).
  # no vendored/pointed cutlass tree is present, -DVLLM_CPP_CUTLASS_FETCH=ON pulls
  # v4.5.0 (header-only). Default OFF so a tight-disk box (dgx) builds against an
  # existing clone via -DVLLM_CPP_CUTLASS_DIR instead of a 200MB download.
  option(VLLM_CPP_CUTLASS_FETCH "FetchContent CUTLASS v4.5.0 if not found locally" OFF)
  # CUTLASS is needed by TWO independent consumers with DIFFERENT arch scopes:
  #   (1) the sm_12x NVFP4 block-scaled GEMM  -> gated on VT_CUTLASS_NVFP4_ARCHS;
  #   (2) the vendored FlashAttention-2 kernels -> gated on VT_FA2_ARCHS, which now
  #       ALSO covers Ampere sm_8x (CudaArchFeatures.cmake fa2 cell, WA-1). FA2 only
  #       `-isystem`s the CUTLASS include tree (see the FA2 target below); it does
  #       NOT use the NVFP4 collective. So "CUTLASS headers present + >=4.5.0" is an
  #       arch-INDEPENDENT fact (VLLM_CPP_CUTLASS_HEADERS), while VLLM_CPP_CUTLASS
  #       stays the strict sm_12x NVFP4 feature flag. This is the additive host
  #       predicate the Ampere FA2 fast path needs
  #       (.agents/specs/cuda-arch-ampere-fastpath.md WA-1). sm_121 is UNCHANGED:
  #       there VT_CUTLASS_NVFP4_ARCHS resolves, so both flags are ON exactly as before.
  if((VT_CUTLASS_NVFP4_ARCHS OR VT_CUTLASS_NVFP4_SM100_ARCHS OR VT_SCALEDMM_C3X_SM90_ARCHS OR VT_SCALEDMM_C3X_SM100_ARCHS OR VT_FA2_ARCHS)
     AND NOT EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h"
     AND VLLM_CPP_CUTLASS_FETCH)
    include(FetchContent)
    FetchContent_Declare(cutlass GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG v4.5.0 GIT_SHALLOW ON)
    FetchContent_MakeAvailable(cutlass)
    set(VLLM_CPP_CUTLASS_DIR "${cutlass_SOURCE_DIR}")
  endif()
  if((VT_CUTLASS_NVFP4_ARCHS OR VT_CUTLASS_NVFP4_SM100_ARCHS OR VT_SCALEDMM_C3X_SM90_ARCHS OR VT_SCALEDMM_C3X_SM100_ARCHS OR VT_FA2_ARCHS)
     AND EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    file(STRINGS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/version.h"
      _CUTLASS_VERSION_LINES
      REGEX "^#define CUTLASS_(MAJOR|MINOR|PATCH) [0-9]+")
    foreach(_CUTLASS_VERSION_LINE IN LISTS _CUTLASS_VERSION_LINES)
      if(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_MAJOR ([0-9]+)")
        set(_CUTLASS_MAJOR "${CMAKE_MATCH_1}")
      elseif(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_MINOR ([0-9]+)")
        set(_CUTLASS_MINOR "${CMAKE_MATCH_1}")
      elseif(_CUTLASS_VERSION_LINE MATCHES "^#define CUTLASS_PATCH ([0-9]+)")
        set(_CUTLASS_PATCH "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    if(NOT DEFINED _CUTLASS_MAJOR OR NOT DEFINED _CUTLASS_MINOR OR
       NOT DEFINED _CUTLASS_PATCH)
      message(FATAL_ERROR "Could not read CUTLASS version from ${VLLM_CPP_CUTLASS_DIR}")
    endif()
    math(EXPR _CUTLASS_VERSION_NUMBER
      "${_CUTLASS_MAJOR} * 10000 + ${_CUTLASS_MINOR} * 100 + ${_CUTLASS_PATCH}")
    if(_CUTLASS_VERSION_NUMBER LESS 40500)
      message(FATAL_ERROR
        "CUTLASS >=4.5.0 is required for SM12 narrow-N NVFP4 tactics and the "
        "vendored FlashAttention-2 kernels; found "
        "${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH} at "
        "${VLLM_CPP_CUTLASS_DIR}")
    endif()
    # Headers are present + new enough: usable by FA2 on ANY arch it resolves for.
    set(VLLM_CPP_CUTLASS_HEADERS ON)
    # Datacenter sm_100a NVFP4 tcgen05 build-verify TU (BACKEND-CUDA-SM100). Its
    # own flag, independent of the sm_12x VLLM_CPP_CUTLASS gate: resolves ON only
    # for a 100a target (cutlass-nvfp4-sm100 FEATURE-TABLE cell). DERIVED+BUILD-
    # VERIFIED — compiled + SASS-proven here, no B200 board ran it.
    if(VT_CUTLASS_NVFP4_SM100_ARCHS)
      set(VLLM_CPP_CUTLASS_NVFP4_SM100 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_100a NVFP4 "
                     "tcgen05 build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_CUTLASS_NVFP4_SM100 OFF)
    endif()
    # Hopper sm_90a CUTLASS C3x FP8 scaled-mm wgmma/TMA build-verify TU
    # (BACKEND-CUDA-SM090). Its own flag, independent of the sm_12x
    # VLLM_CPP_CUTLASS gate: resolves ON only for a 90a target (scaledmm-c3x-sm90
    # FEATURE-TABLE cell). DERIVED+BUILD-VERIFIED — compiled + SASS-proven here, no
    # Hopper board ran it.
    if(VT_SCALEDMM_C3X_SM90_ARCHS)
      set(VLLM_CPP_SCALEDMM_C3X_SM90 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_90a C3x FP8 "
                     "scaled-mm wgmma/TMA build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_SCALEDMM_C3X_SM90 OFF)
    endif()
    # Datacenter-Blackwell sm_100a CUTLASS C3x FP8 scaled-mm tcgen05 build-verify TU
    # (BACKEND-CUDA-SM100). Its own flag, independent of the sm_12x VLLM_CPP_CUTLASS
    # gate AND the Hopper VLLM_CPP_SCALEDMM_C3X_SM90 gate: resolves ON only for a
    # 100a target (scaledmm-c3x-sm100 FEATURE-TABLE cell). DERIVED+BUILD-VERIFIED —
    # compiled + SASS-proven here, no B200 board ran it.
    if(VT_SCALEDMM_C3X_SM100_ARCHS)
      set(VLLM_CPP_SCALEDMM_C3X_SM100 ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm_100a C3x FP8 "
                     "scaled-mm tcgen05 build-verify TU (DERIVED+BUILD-VERIFIED, no board ran it)")
    else()
      set(VLLM_CPP_SCALEDMM_C3X_SM100 OFF)
    endif()
    if(VT_CUTLASS_NVFP4_ARCHS)
      set(VLLM_CPP_CUTLASS ON)
      message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm120a NVFP4 cutlass GEMM")
    else()
      # Ampere / non-sm_12x single-arch build: headers available for FA2, but the
      # sm_12x NVFP4 GEMM is not requested for this arch, so keep that flag OFF.
      set(VLLM_CPP_CUTLASS OFF)
      message(STATUS "CUTLASS headers found at ${VLLM_CPP_CUTLASS_DIR} "
                     "(available to FA2 for arch(es) [${VT_FA2_ARCHS}]); "
                     "sm120a NVFP4 GEMM not requested for this target")
    endif()
  else()
    set(VLLM_CPP_CUTLASS OFF)
    set(VLLM_CPP_CUTLASS_HEADERS OFF)
    set(VLLM_CPP_CUTLASS_NVFP4_SM100 OFF)
    set(VLLM_CPP_SCALEDMM_C3X_SM90 OFF)
    set(VLLM_CPP_SCALEDMM_C3X_SM100 OFF)
    # Two very different situations reach this branch, and only one of them is
    # benign. If VT_FA2_ARCHS resolved, the requested arch DOES support FA2 and
    # the build is about to drop it purely because no CUTLASS tree was found —
    # a silently slower binary, which is the failure a first-time builder hits
    # (they read "CUDA feature fa2: ENABLED for [86]" in the table above and
    # never notice the STATUS line that takes it away). Warn there. If no arch
    # wanted a CUTLASS-dependent feature in the first place, nothing is lost.
    if(VT_FA2_ARCHS)
      message(WARNING
        "CUTLASS headers NOT found, so FlashAttention-2 will NOT be built even "
        "though arch(es) [${VT_FA2_ARCHS}] support it. Attention falls back to "
        "the portable path and decode/prefill will be slower. Fix with "
        "-DVLLM_CPP_CUTLASS_FETCH=ON (downloads CUTLASS 4.5.0) or "
        "-DVLLM_CPP_CUTLASS_DIR=<path to a CUTLASS >=4.5.0 checkout).")
    else()
      message(STATUS "CUTLASS not found / no CUTLASS-dependent feature for this arch; "
                     "NVFP4 GEMM + FA2 disabled "
                     "(set -DVLLM_CPP_CUTLASS_DIR=<cutlass> or -DVLLM_CPP_CUTLASS_FETCH=ON)")
    endif()
  endif()
  # Marlin NVFP4 W4A16 grouped-MoE GEMM drop-in (the 35B expert kernel). Vendored
  # torch-free slice of vLLM's moe/marlin_moe_wna16 under src/vt/cuda/marlin/
  # (marlin-dropin-feasibility.md). NO external dependency — the kernel is
  # standalone CUDA + inline PTX. Enabled on sm_12xa (compiles to native sm_121
  # SASS). The two vendored TUs + the vt launcher are isolated with their own
  # compile options (see below); gated by VT_MARLIN_NVFP4.
  option(VLLM_CPP_MARLIN "Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa)" ON)
  if(VLLM_CPP_MARLIN AND VT_MARLIN_NVFP4_ARCHS
     AND EXISTS "${CMAKE_SOURCE_DIR}/src/vt/cuda/marlin/core/scalar_type.hpp")
    set(VLLM_CPP_MARLIN ON)
    message(STATUS "Marlin NVFP4 W4A16 MoE GEMM enabled (vendored, sm_12xa)")
  else()
    set(VLLM_CPP_MARLIN OFF)
    message(STATUS "Marlin NVFP4 MoE GEMM disabled (needs sm_12xa + vendored slice)")
  endif()
endif()

include(cmake/CompilerWarnings.cmake)
# Triton AOT kernels (cubins embedded in C launchers, CUDA-only), VENDORED
# per-arch under src/vt/cuda/triton_aot_vendored/. Defines the gated option
# VLLM_CPP_TRITON (default OFF; consumes the vendored artifacts — NO Python) +
# the maintainer knob VLLM_CPP_TRITON_REGEN (regenerates them with Python+Triton;
# scripts/regen-triton-aot.sh) + add_triton_kernel()/triton_aot_finalize(); with
# the option OFF this include is inert (no sources added). Wired below under
# CUDA for the SANCTIONED GDN fast-path (see .agents/discipline.md).
include(cmake/TritonAOT.cmake)

configure_file(include/vllm/version.h.in include/vllm/version.h @ONLY)

add_library(vllm STATIC
  src/vllm/version.cpp
  src/vllm/sampling_params.cpp
  src/vllm/lora/punica_cpu.cpp
  src/vllm/config/scheduler.cpp
  src/vllm/config/kv_transfer.cpp
  src/vllm/config/speculative.cpp
  src/vllm/outputs.cpp
  src/vllm/transformers_utils/hf_config.cpp
  src/vllm/model_executor/model_loader/safetensors_reader.cpp
  src/vllm/model_executor/model_loader/gguf_reader.cpp
  src/vllm/model_executor/model_loader/gguf_dequant.cpp
  src/vllm/model_executor/model_loader/gguf_keep_quant.cpp
  src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  src/vllm/model_executor/model_loader/awq_gptq_dequant.cpp
  src/vllm/model_executor/model_loader/mxfp4_dequant.cpp
  src/vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.cpp
  src/vllm/model_executor/models/qwen3_5_weights.cpp
  src/vllm/model_executor/models/qwen3_5_dense_weights.cpp
  src/vllm/model_executor/models/qwen3_5_mtp.cpp
  src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp
  src/vllm/model_executor/models/qwen3_5.cpp
  src/vllm/model_executor/models/qwen3_5_common.cpp
  src/vllm/model_executor/models/qwen3_5_dense.cpp
  src/vllm/model_executor/models/qwen3_5_moe.cpp
  src/vllm/model_executor/models/qwen3_dense.cpp
  src/vllm/model_executor/models/qwen3_weights.cpp
  src/vllm/model_executor/models/qwen3.cpp
  src/vllm/model_executor/models/qwen3_dflash.cpp
  src/vllm/model_executor/models/qwen3_dflash_gguf.cpp
  src/vllm/model_executor/models/qwen3_dflash_weights.cpp
  src/vllm/model_executor/models/qwen3_moe_registry.cpp
  src/vllm/model_executor/models/qwen3_moe_weights.cpp
  src/vllm/model_executor/models/qwen3_moe.cpp
  src/vllm/model_executor/models/deepseek_v2_registry.cpp
  src/vllm/model_executor/models/deepseek_v2_weights.cpp
  src/vllm/model_executor/models/deepseek_v2.cpp
  src/vllm/model_executor/models/deepseek_v4_registry.cpp
  src/vllm/model_executor/models/deepseek_v4_weights.cpp
  src/vllm/model_executor/models/deepseek_v4.cpp
  src/vllm/model_executor/models/deepseek_v4_dsa.cpp
  src/vllm/model_executor/models/deepseek_v4_compressor.cpp
  src/vllm/model_executor/models/deepseek_v4_mhc.cpp
  src/vllm/model_executor/models/deepseek_v4_moe.cpp
  src/vllm/model_executor/models/deepseek_v4_device.cpp
  src/vllm/model_executor/models/laguna_registry.cpp
  src/vllm/model_executor/models/laguna_weights.cpp
  src/vllm/model_executor/models/laguna_ops.cpp
  src/vllm/model_executor/models/laguna_device.cpp
  src/vllm/model_executor/models/laguna_shared_fp4.cpp
  src/vllm/model_executor/models/laguna.cpp
  src/vllm/model_executor/models/kimi_k3_registry.cpp
  src/vllm/model_executor/models/kimi_k3_weights.cpp
  src/vllm/model_executor/models/kimi_k3.cpp
  src/vllm/model_executor/models/minimax_h3.cpp
  src/vllm/model_executor/models/minimax_h3_packing.cpp
  src/vllm/model_executor/models/minimax_h3_planner.cpp
  src/vllm/model_executor/models/minimax_h3_gguf.cpp
  src/vllm/model_executor/models/minimax_h3_nvfp4.cpp
  src/vllm/model_executor/models/minimax_h3_sharded.cpp
  src/vllm/model_executor/models/minimax_h3_audio_vae.cpp
  src/vllm/model_executor/models/minimax_h3_video_vae.cpp
  src/vllm/model_executor/models/minimax_h3_video_vae_device.cpp
  src/vllm/model_executor/models/minimax_h3_vae_cnn.cpp
  src/vllm/model_executor/models/minimax_h3_encoder.cpp
  src/vllm/model_executor/models/minimax_h3_pipeline.cpp
  src/vllm/model_executor/models/minimax_h3_wav.cpp
  src/vllm/model_executor/models/minimax_h3_mux.cpp
  src/vllm/model_executor/models/minimax_h3_device_resolve.cpp
  src/vllm/model_executor/models/minimax_h3_device.cpp
  src/vllm/model_executor/models/minimax_h3_vae_loader.cpp
  src/vllm/model_executor/models/minimax_h3_encoder_gguf.cpp
  src/vllm/model_executor/models/minimax_h3_encoder_device.cpp
  src/vllm/model_executor/models/minimax_h3_vision_gguf.cpp
  src/vllm/entrypoints/openai/video_api.cpp
  src/vllm/model_executor/models/kimi_kda.cpp
  src/vllm/model_executor/models/kimi_linear_registry.cpp
  src/vllm/model_executor/models/kimi_linear_weights.cpp
  src/vllm/model_executor/models/kimi_linear_forward.cpp
  src/vllm/model_executor/models/kimi_linear.cpp
  src/vllm/model_executor/models/kimi_linear_device.cpp
  src/vllm/model_executor/models/glm4_registry.cpp
  src/vllm/model_executor/models/glm4_weights.cpp
  src/vllm/model_executor/models/glm4.cpp
  src/vllm/model_executor/models/glm4_moe_lite_registry.cpp
  src/vllm/model_executor/models/opt_registry.cpp
  src/vllm/model_executor/models/opt_weights.cpp
  src/vllm/model_executor/models/opt.cpp
  src/vllm/model_executor/models/llama_registry.cpp
  src/vllm/model_executor/models/llama_weights.cpp
  src/vllm/model_executor/models/mistral_registry.cpp
  src/vllm/model_executor/models/mistral_weights.cpp
  src/vllm/model_executor/models/gemma4_registry.cpp
  src/vllm/model_executor/models/gemma4_weights.cpp
  src/vllm/model_executor/models/gemma4.cpp
  src/vllm/model_executor/models/gemma4_mm.cpp
  src/vllm/model_executor/models/gemma3_registry.cpp
  src/vllm/model_executor/models/gemma3_weights.cpp
  src/vllm/model_executor/models/gemma3.cpp
  src/vllm/model_executor/models/gemma2_registry.cpp
  src/vllm/model_executor/models/gemma2_weights.cpp
  src/vllm/model_executor/models/gemma2.cpp
  src/vllm/model_executor/models/gemma_registry.cpp
  src/vllm/model_executor/models/gemma_weights.cpp
  src/vllm/model_executor/models/gemma.cpp
  src/vllm/model_executor/models/olmo2_registry.cpp
  src/vllm/model_executor/models/olmo2_weights.cpp
  src/vllm/model_executor/models/olmo2.cpp
  src/vllm/model_executor/models/phi3_registry.cpp
  src/vllm/model_executor/models/phi3_weights.cpp
  src/vllm/model_executor/models/phi3.cpp
  src/vllm/model_executor/models/granite_registry.cpp
  src/vllm/model_executor/models/granite_weights.cpp
  src/vllm/model_executor/models/granite.cpp
  src/vllm/model_executor/models/stablelm_registry.cpp
  src/vllm/model_executor/models/stablelm_weights.cpp
  src/vllm/model_executor/models/stablelm.cpp
  src/vllm/model_executor/models/minicpm_registry.cpp
  src/vllm/model_executor/models/minicpm_weights.cpp
  src/vllm/model_executor/models/minicpm.cpp
  src/vllm/model_executor/models/minicpm3_registry.cpp
  src/vllm/model_executor/models/minicpm3_weights.cpp
  src/vllm/model_executor/models/minicpm3.cpp
  src/vllm/model_executor/models/internlm2_registry.cpp
  src/vllm/model_executor/models/internlm2_weights.cpp
  src/vllm/model_executor/models/commandr_registry.cpp
  src/vllm/model_executor/models/commandr_weights.cpp
  src/vllm/model_executor/models/commandr.cpp
  src/vllm/model_executor/models/phi_registry.cpp
  src/vllm/model_executor/models/phi_weights.cpp
  src/vllm/model_executor/models/phi.cpp
  src/vllm/model_executor/models/model_registry.cpp
  src/vllm/plugins/plugins.cpp
  src/vllm/model_executor/models/qwen3_vl_vision.cpp
  src/vllm/model_executor/models/gemma4_vision.cpp
  src/vllm/model_executor/models/gemma4_audio.cpp
  src/vllm/model_executor/models/whisper_audio.cpp
  src/vllm/model_executor/models/qwen3_vl_text.cpp
  src/vllm/model_executor/models/qwen3_vl.cpp
  src/vllm/model_executor/models/qwen3_vl_registry.cpp
  src/vllm/model_executor/models/voxtral.cpp
  src/vllm/multimodal/hasher.cpp
  src/vllm/multimodal/qwen3vl_processor.cpp
  src/vllm/multimodal/audio_processor.cpp
  src/vllm/v1/core/encoder_cache_manager.cpp
  src/vllm/tokenizer/unicode_data.cpp
  src/vllm/tokenizer/pretokenizer.cpp
  src/vllm/tokenizer/bpe.cpp
  src/vllm/tokenizer/tokenizer.cpp
  src/vllm/v1/request.cpp
  src/vllm/v1/outputs.cpp
  src/vllm/v1/sample/metadata.cpp
  src/vllm/v1/sample/ops/penalties.cpp
  src/vllm/v1/sample/ops/bad_words.cpp
  src/vllm/v1/sample/logits_processor/builtin.cpp
  src/vllm/v1/sample/sampler.cpp
  src/vllm/v1/spec_decode/rejection_sampler.cpp
  src/vllm/v1/spec_decode/ngram_proposer.cpp
  src/vllm/v1/spec_decode/draft_model_proposer.cpp
  src/vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_prefill_inputs.cpp
  src/vllm/v1/worker/gpu/spec_decode/mtp/speculator.cpp
  src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.cpp
  src/vllm/v1/kv_cache_interface.cpp
  src/vllm/v1/kv_cache_spec_registry.cpp
  src/vllm/v1/core/kv_cache_utils.cpp
  src/vllm/distributed/kv_events.cpp
  src/vllm/v1/core/block_pool.cpp
  src/vllm/v1/core/single_type_kv_cache_manager.cpp
  src/vllm/v1/core/kv_cache_coordinator.cpp
  src/vllm/v1/core/kv_cache_manager.cpp
  src/vllm/v1/metrics/prometheus.cpp
  src/vllm/v1/metrics/loggers.cpp
  src/vllm/v1/kv_offload/cache_identity.cpp
  src/vllm/v1/kv_offload/cache_policy.cpp
  src/vllm/v1/kv_offload/cpu_manager.cpp
  src/vllm/v1/kv_offload/kv_block_transfer.cpp
  src/vllm/v1/kv_offload/fs_io.cpp
  src/vllm/v1/kv_offload/fs_tier.cpp
  src/vllm/v1/kv_offload/tiering_manager.cpp
  src/vllm/v1/kv_offload/kv_connector.cpp
  # LMCache MODE-1 (lm://) wire codec + blocking TCP client — KV-EXTERNAL-CACHE
  # W1 (codec) + W2 (client).  Pure CPU; the W3 LMCacheConnector subclass
  # (below) is the KVConnector that drives them over the W5 seam.
  src/vllm/v1/kv_offload/lmcache/memory_format.cpp
  src/vllm/v1/kv_offload/lmcache/remote_protocol.cpp
  src/vllm/v1/kv_offload/lmcache/cache_engine_key.cpp
  src/vllm/v1/kv_offload/lmcache/token_hasher.cpp
  # W4 peer-agreeing key derivation (ChunkedTokenDatabase: vLLM sha256_cbor,
  # chunk_size 256) — byte-identical CacheEngineKeys vs a real Python peer.
  src/vllm/v1/kv_offload/lmcache/chunked_token_database.cpp
  src/vllm/v1/kv_offload/lmcache/remote_client.cpp
  # LMCache lm:// KVConnector subclass over the W5 seam — KV-EXTERNAL-CACHE W3.
  src/vllm/v1/kv_offload/lmcache/lmcache_connector.cpp
  src/vllm/v1/core/sched/request_queue.cpp
  src/vllm/v1/core/sched/output.cpp
  src/vllm/v1/core/sched/scheduler.cpp
  src/vllm/v1/core/sched/async_scheduler.cpp
  src/vllm/v1/core/sched/utils.cpp
  src/vllm/v1/worker/gpu/block_table.cpp
  src/vllm/v1/worker/gpu/input_batch.cpp
  src/vllm/v1/worker/gpu/prepare_inputs.cpp
  src/vllm/v1/worker/gpu/async_output.cpp
  src/vllm/v1/worker/gpu/runner.cpp
  src/vllm/platforms/platform.cpp
  src/vllm/platforms/cpu.cpp
  src/vllm/v1/attention/backend.cpp
  src/vllm/v1/attention/registry.cpp
  src/vllm/v1/attention/backends/utils.cpp
  src/vllm/v1/attention/backends/gdn_attn.cpp
  src/vllm/v1/structured_output/backend_types.cpp
  src/vllm/v1/structured_output/backend_native.cpp
  src/vllm/v1/structured_output/backend_xgrammar.cpp
  src/vllm/v1/structured_output/xgrammar_json_schema.cpp
  src/vllm/v1/structured_output/json_schema_to_gbnf.cpp
  src/vllm/v1/structured_output/jump_forward.cpp
  src/vllm/v1/structured_output/request.cpp
  src/vllm/v1/structured_output/manager.cpp
  src/vllm/v1/engine/types.cpp
  src/vllm/v1/engine/detokenizer.cpp
  src/vllm/v1/engine/core.cpp
  src/vllm/v1/engine/core_proc.cpp
  src/vllm/v1/engine/core_client.cpp
  src/vllm/v1/engine/async_llm.cpp
  src/vllm/v1/engine/input_processor.cpp
  src/vllm/v1/engine/parallel_sampling.cpp
  src/vllm/v1/engine/output_processor.cpp
  src/vllm/v1/engine/logprobs.cpp
  src/vllm/v1/engine/llm_engine.cpp
  src/vllm/entrypoints/beam_search.cpp
  src/vllm/v1/executor/executor.cpp
  src/vllm/model_executor/layers/attention/attention.cpp
  src/vllm/model_executor/layers/attention/chunked_local_attention.cpp
  src/vllm/model_executor/layers/pooler/methods.cpp
  src/vllm/model_executor/layers/pooler/activations.cpp
  src/vllm/model_executor/layers/pooler/heads.cpp
  src/vllm/model_executor/layers/pooler/poolers.cpp
  src/vllm/model_executor/layers/pooler/dispatch_pooler.cpp
  src/vllm/v1/worker/gpu/pool/pooling_runner.cpp
  src/vllm/model_executor/layers/attention/mla_attention.cpp
  src/vllm/model_executor/layers/rotary_embedding/base.cpp
  src/vllm/model_executor/layers/rotary_embedding/common.cpp
  src/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_alpha_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/dynamic_ntk_scaling_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/llama3_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/phi3_long_rope_scaled_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.cpp
  src/vllm/model_executor/layers/rotary_embedding/mrope.cpp
  src/vllm/entrypoints/chat_template.cpp
  src/vllm/entrypoints/model_loader.cpp
  src/capi/vllm_c.cpp
  src/capi/chat_prompt.cpp
  src/vllm/entrypoints/openai/protocol.cpp
  src/vllm/entrypoints/openai/chat_mm.cpp
  src/vllm/entrypoints/openai/serving_utils.cpp
  src/vllm/entrypoints/openai/serving_completion.cpp
  src/vllm/entrypoints/openai/serving_chat.cpp
  src/vllm/entrypoints/openai/serving_models.cpp
  src/vllm/entrypoints/openai/run_batch.cpp
  src/vllm/entrypoints/openai/tool_parsers/abstract.cpp
  src/vllm/entrypoints/openai/tool_parsers/structural_tags.cpp
  src/vllm/entrypoints/openai/tool_parsers/detect.cpp
  src/vllm/entrypoints/openai/tool_parsers/utils.cpp
  src/vllm/entrypoints/openai/tool_parsers/hermes.cpp
  src/vllm/entrypoints/openai/tool_parsers/mistral.cpp
  src/vllm/entrypoints/openai/tool_parsers/llama.cpp
  src/vllm/entrypoints/openai/tool_parsers/llama4_pythonic.cpp
  src/vllm/entrypoints/openai/tool_parsers/pythonic_core.cpp
  src/vllm/entrypoints/openai/tool_parsers/pythonic.cpp
  src/vllm/entrypoints/openai/tool_parsers/qwen3.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v3.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v31.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v32.cpp
  src/vllm/entrypoints/openai/tool_parsers/deepseek_v4.cpp
  src/vllm/entrypoints/openai/tool_parsers/longcat.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite4.cpp
  src/vllm/entrypoints/openai/tool_parsers/granite_20b_fc.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/abstract.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/detect.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/think_auto.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/basic.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/deepseek_r1.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/identity.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/deepseek_v3.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/mistral.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/minimax_m2.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/step3.cpp
  src/vllm/entrypoints/openai/reasoning_parsers/olmo3.cpp
  src/vllm/entrypoints/openai/tool_parsers/xlam.cpp
  src/vllm/entrypoints/openai/tool_parsers/phi4_mini.cpp
  src/vllm/entrypoints/openai/tool_parsers/internlm.cpp
  src/vllm/entrypoints/openai/tool_parsers/jamba.cpp
  src/vllm/entrypoints/openai/tool_parsers/step3.cpp
  src/vllm/entrypoints/openai/tool_parsers/step3p5.cpp
  src/vllm/entrypoints/openai/tool_parsers/qwen3_coder.cpp
  src/vllm/entrypoints/openai/tool_parsers/minicpm5.cpp
  src/vllm/entrypoints/openai/tool_parsers/hy_v3.cpp
  src/vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.cpp
  src/vllm/entrypoints/openai/tool_parsers/apertus.cpp
  src/vllm/entrypoints/openai/tool_parsers/ernie45.cpp
  src/vllm/entrypoints/openai/tool_parsers/gigachat3.cpp
  src/vllm/entrypoints/openai/tool_parsers/olmo3.cpp
  src/vllm/entrypoints/openai/tool_parsers/lfm2.cpp
  src/vllm/entrypoints/openai/tool_parsers/poolside_v1.cpp
  src/vllm/entrypoints/openai/tool_parsers/functiongemma.cpp
  src/vllm/entrypoints/openai/tool_parsers/kimi_k2.cpp
  src/vllm/entrypoints/openai/tool_parsers/glm47.cpp
  src/vllm/entrypoints/openai/tool_parsers/minimax_m2.cpp
  src/vllm/entrypoints/openai/tool_parsers/gemma4.cpp
  src/vllm/entrypoints/openai/tool_parsers/seed_oss.cpp
  src/vllm/parser/engine/incremental_lexer.cpp
  src/vllm/parser/engine/token_id_scanner.cpp
  src/vllm/parser/engine/streaming_parser_engine.cpp
  src/vllm/parser/engine/configs.cpp
  src/vllm/parser/engine/registry.cpp
  src/vllm/parser/engine/parser_engine.cpp
  src/vllm/parser/kimi_k2.cpp
  src/vllm/parser/glm47_moe.cpp
  src/vllm/parser/gemma4.cpp
  src/vllm/parser/inkling.cpp
  src/vllm/parser/parser_manager.cpp
  src/vt/dtype.cpp
  src/vt/backend.cpp
  src/vt/tensor.cpp
  src/vt/arena.cpp
  src/vt/op_provider.cpp
  src/vt/communicator.cpp
  src/vt/ops.cpp
  src/vt/merged_gemm.cpp
  src/vt/cuda/nvfp4_persistent_cache.cpp
  src/vt/cpu/cpu_backend.cpp
  src/vt/cpu/cpu_threadpool.cpp
  src/vt/cpu/cpu_ops.cpp
  src/vt/cpu/cpu_minimax_h3.cpp
  src/vt/cpu/cpu_matmul_elem.cpp
  src/vt/cpu/cpu_quant_dequant.cpp
  src/vt/cpu/cpu_quant_act.cpp
  src/vt/cpu/cpu_quant_dot.cpp
  src/vt/cpu/cpu_quant_dot_arm.cpp
  src/vt/cpu/cpu_quant_repack.cpp
  src/vt/cpu/cpu_quant_repack_arm.cpp
  src/vt/cpu/cpu_quant_traits.cpp
  src/vt/cpu/cpu_quant_gemm.cpp
  src/vt/cpu/cpu_layernorm.cpp
  src/vt/cpu/cpu_cache.cpp
  src/vt/cpu/cpu_mla_attn.cpp
  src/vt/cpu/cpu_mla_prefill.cpp
  src/vt/cpu/cpu_paged_attn.cpp
  src/vt/cpu/cpu_sample.cpp)
add_library(vllm::vllm ALIAS vllm)

# QUANT-GGUF-CIQ-GEMM G6: the Arm i8mm (mmla) quant-GEMM tier needs the ISA
# extension enabled so <arm_neon.h> exposes `vmmlaq_s32` and defines
# `__ARM_FEATURE_MATMUL_INT8`. The project sets no global -march, so build ONLY
# this translation unit with armv8.2-a+i8mm (surgical, exactly like the x86
# tier's per-function `target("f16c")`): every other TU stays on the default
# baseline, and the kernels are additionally runtime-gated on HWCAP2_I8MM so the
# binary still runs on non-i8mm aarch64 (portable tier serves those shapes).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # G6 mmla tier AND G7 repack tier (cpu_quant_repack_arm.cpp) both need i8mm.
  # The global -ffp-contract=off (top of this file) still applies on top of this
  # per-file -march, which is what keeps the repack gemm/gemv's non-fused
  # `vmlaq_f32` bit-identical to the tier-0 path.
  set_source_files_properties(
    src/vt/cpu/cpu_quant_dot_arm.cpp
    src/vt/cpu/cpu_quant_repack_arm.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:-march=armv8.2-a+i8mm+dotprod>")
endif()
# Static-init registrars (e.g. the CPU backend, the platform table, the op
# table, the attention backends) live in archive members no consumer references
# directly; force-link the whole vllm archive so every consumer (tests, C API
# shared lib, benchmarks) gets registration.
#
# BACKEND-METAL-MLX W0 item 1: this was `if(UNIX AND NOT APPLE)`, which is TRUE
# on Linux and FALSE on macOS (Apple is UNIX), so on macOS the force-link was
# silently skipped and EVERY static registrar was dropped by ld64 — measured as
# `test_backend` failing 5/7 with `vt: no backend registered for device type 0`
# (.agents/specs/backend-fanout-metal-vulkan-xpu.md § Our baseline). ld64 has no
# `--whole-archive`; the equivalent is `-force_load <archive>`. This unblocks the
# CPU backend on macOS as well as any future Metal backend.
if(APPLE)
  target_link_options(vllm INTERFACE "LINKER:-force_load,$<TARGET_FILE:vllm>")
elseif(UNIX)
  target_link_options(vllm INTERFACE "LINKER:--whole-archive,$<TARGET_FILE:vllm>,--no-whole-archive")
endif()
target_include_directories(vllm
  PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/include)
# The C ABI impl reaches its internal test-hook header as "capi/engine_handle.h"
# (an internal, non-installed header under src/); src is a PRIVATE search path.
target_include_directories(vllm PRIVATE src)
# third_party/ is a SYSTEM include so vendored headers stay off the -Werror path.
# It carries the header-only deps reached as <dir/header>: <httplib/httplib.h>,
# <nlohmann/json.hpp>, <doctest/doctest.h>, and the vendored google/minja Jinja
# engine <minja/chat-template.hpp> / <minja/minja.hpp> that backs
# src/vllm/entrypoints/chat_template.cpp.
target_include_directories(vllm SYSTEM PUBLIC third_party)
vllm_cpp_set_warnings(vllm)
# SYSTEM covers the FRONT-END diagnostics from those vendored headers, but not
# the ones the OPTIMIZER raises after inlining: those are attributed to the TU
# being compiled, not to the header, so -isystem cannot reach them. GCC 15
# inlines minja::Parser::parseValueExpression (third_party/minja/minja.hpp),
# loses track of the allocation base of a std::vector<std::string> it is
# destroying, and reports -Wfree-nonheap-object against correct vendored code.
# Demote just that diagnostic, just here: the warning stays visible, -Werror
# stays in force for every other diagnostic and every other TU, and the
# vendored source stays unmodified. Ordered after vllm_cpp_set_warnings so the
# target-level -Werror cannot promote it again (the lesson of the MLX header
# exception below). cmake/CompilerWarnings.cmake documents the same GCC
# false-positive class for the sanitizer lanes.
set_source_files_properties(src/vllm/entrypoints/chat_template.cpp PROPERTIES
  COMPILE_OPTIONS
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-error=free-nonheap-object>")
# --- Vendored BLAKE3 (LMCache MODE-1 lm:// token hash, KV-EXTERNAL-CACHE W1) --
# The official BLAKE3 C implementation (third_party/blake3, upstream tag 1.5.5
# / commit 81f772a; dual CC0-1.0 / Apache-2.0), the same algorithm the `blake3`
# PyPI package LMCache imports binds. Built as a SEPARATE static lib with the
# portable backend forced (every SIMD path disabled via BLAKE3_NO_*): the output
# is byte-identical on x86-64 AND aarch64 (BLAKE3 is defined little-endian; the
# portable load32 in blake3_impl.h assembles words byte-wise, endian-independent)
# and this keeps the vendored C sources UNMODIFIED and off the -Werror path.
# PIC so vllm_shared can link it. Provenance: third_party/README.md,
# .agents/porting-inventory.md §9.
enable_language(C)  # blake3 is plain C; the project is otherwise CXX-only
add_library(blake3_vendored STATIC
  third_party/blake3/blake3.c
  third_party/blake3/blake3_dispatch.c
  third_party/blake3/blake3_portable.c)
target_compile_definitions(blake3_vendored PRIVATE
  BLAKE3_NO_AVX512 BLAKE3_NO_AVX2 BLAKE3_NO_SSE41 BLAKE3_NO_SSE2
  BLAKE3_USE_NEON=0)
target_include_directories(blake3_vendored PUBLIC third_party/blake3)
set_target_properties(blake3_vendored PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vllm PUBLIC blake3_vendored)
# --- Metal backend (BACKEND-METAL-MLX W0 skeleton) --------------------------
# Purely ADDITIVE, exactly like the CUDA block below: a `target_sources` list, a
# PUBLIC compile definition tests can gate on, and two framework links. No
# existing source file, and no line of the CPU/CUDA build, is touched by turning
# it on. The MSL is compiled at RUN TIME from an embedded string
# (src/vt/metal/metal_msl.h), so there is NO offline shader-compilation step and
# no dependency on a `metal` compiler this build could fail to find.
if(VLLM_CPP_METAL)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_METAL)
  target_sources(vllm PRIVATE
    src/vllm/platforms/metal.cpp
    src/vt/metal/metal_context.mm
    src/vt/metal/metal_backend.mm
    src/vt/metal/metal_ops.mm)
  target_link_libraries(vllm PUBLIC "-framework Metal" "-framework Foundation")
  # --- optional MLX GEMM provider (BACKEND-ACCEL-PROVIDER M5) ----------------
  # ONE additive block: one source file, one define, one include dir, one link.
  # Turning it OFF leaves the Metal build byte-for-byte what it was — which is
  # the point of the provider seam. The provider registers ALONGSIDE the native
  # MSL GEMM and wins by priority, deterministically, instead of racing it
  # through static-init order the way the old flat op table would have.
  if(VLLM_CPP_MLX)
    if(NOT MLX_ROOT)
      message(FATAL_ERROR "VLLM_CPP_MLX=ON requires -DMLX_ROOT=<mlx install root> "
        "(the pip wheel's site-packages/mlx directory works).")
    endif()
    find_library(MLX_LIBRARY NAMES mlx PATHS "${MLX_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT MLX_LIBRARY OR NOT EXISTS "${MLX_ROOT}/include/mlx/array.h")
      message(FATAL_ERROR "VLLM_CPP_MLX=ON but no libmlx + include/mlx/array.h under "
        "MLX_ROOT=${MLX_ROOT}")
    endif()
    include(cmake/MLXDependency.cmake)
    vllm_cpp_import_mlx("${MLX_ROOT}" "${MLX_LIBRARY}")
    target_compile_definitions(vllm PUBLIC VLLM_CPP_MLX)
    target_sources(vllm PRIVATE src/vt/metal/metal_mlx_provider.mm)
    # MLX is a third-party dependency target: its public headers are SYSTEM,
    # while vllm's own sources remain subject to the target-wide -Werror policy.
    target_link_libraries(vllm PUBLIC vllm_cpp::mlx)
    message(STATUS "MLX GEMM provider enabled: ${MLX_LIBRARY}")
  endif()
endif()
# --- Vulkan backend (BACKEND-VULKAN W0 skeleton) -----------------------------
# Purely ADDITIVE, exactly like the Metal block above: a `target_sources` list, a
# PUBLIC compile definition tests can gate on, and `${CMAKE_DL_LIBS}` for dlopen.
# No existing source file, and no line of the CPU/CUDA/Metal build, is touched by
# turning it on.
#
# NOTHING is needed at build time: the Khronos headers are vendored under
# third_party/ (single-header-adjacent, Apache-2.0, see third_party/README.md),
# the GLSL is compiled AHEAD OF TIME into the committed
# src/vt/vulkan/vulkan_spirv.h (regenerate with scripts/gen-vulkan-spirv.py), and
# libvulkan.so.1 is resolved with dlopen at run time rather than linked. That is
# a deliberate divergence from llama.cpp, which requires `glslc` on every build
# machine; neither of our boxes has one and neither grants sudo to install one.
if(VLLM_CPP_VULKAN)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_VULKAN)
  target_sources(vllm PRIVATE
    src/vllm/platforms/vulkan.cpp
    src/vt/vulkan/vulkan_loader.cpp
    src/vt/vulkan/vulkan_context.cpp
    src/vt/vulkan/vulkan_backend.cpp
    src/vt/vulkan/vulkan_ops.cpp
    # GENERATED by scripts/gen-vulkan-spirv.py, alongside vulkan_spirv.h. The
    # SPIR-V words are here rather than in the header so that adding shaders costs
    # ONE TU's compile time instead of every TU that includes the table.
    src/vt/vulkan/vulkan_spirv.cpp)
  target_link_libraries(vllm PUBLIC ${CMAKE_DL_LIBS})
endif()
# --- ROCm/HIP backend (BACKEND-ROCM W0 skeleton) ------------------------------
# Purely ADDITIVE like the Metal and Vulkan blocks above: a source list, a PUBLIC
# compile definition the tests gate on, and one link library. No existing source
# file and no line of the CPU/CUDA/Metal/Vulkan build changes when it is on.
#
# **NEVER COMPILED.** These four TUs have no build report from any machine. The
# arch parse they depend on (include/vt/rocm/rocm_arch.h) is deliberately NOT
# here: it is a HIP-free header compiled and unit-tested in every build,
# including CPU-only CI. See tests/vt/test_rocm_arch.cpp.
if(VLLM_CPP_HIP)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_HIP)
  target_sources(vllm PRIVATE
    src/vllm/platforms/rocm.cpp
    src/vt/rocm/rocm_backend.hip
    src/vt/rocm/rocm_rmsnorm.hip
    src/vt/rocm/rocm_ops.hip)
  if(VLLM_CPP_HIP_ARCHITECTURES)
    set_source_files_properties(
      src/vt/rocm/rocm_backend.hip
      src/vt/rocm/rocm_rmsnorm.hip
      src/vt/rocm/rocm_ops.hip
      PROPERTIES HIP_ARCHITECTURES "${VLLM_CPP_HIP_ARCHITECTURES}")
  endif()
  # Prefer the absolute path inside ${ROCM_PATH}/lib, fall back to the bare name,
  # which is upstream's own resolution order (vllm/CMakeLists.txt:335-341).
  find_library(VLLM_CPP_AMDHIP64_LIB NAMES amdhip64 libamdhip64.so
    PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64)
  if(VLLM_CPP_AMDHIP64_LIB)
    target_link_libraries(vllm PUBLIC ${VLLM_CPP_AMDHIP64_LIB})
  else()
    target_link_libraries(vllm PUBLIC amdhip64)
  endif()
else()
  # BIT-ROT GUARD (BACKEND-ROCM W0). The ROCm platform leg is deliberately plain
  # C++ with no HIP header, which means it CAN be compiled on a machine with no
  # ROCm — and if we do not, nothing on any maintainer machine ever compiles it,
  # so the next change to the `Platform` interface breaks it silently and the
  # breakage is discovered by a contributor mid-bring-up. Compiling it as an
  # OBJECT library that is never linked keeps it honest at the cost of one TU.
  #
  # Not linked ANYWHERE on purpose: the object is built and discarded, so the
  # platform's static registrar cannot run and a non-HIP build is byte-identical
  # to what it was. This is a syntax/type gate, not a build of the backend — the
  # HIP TUs still have no build report from anyone.
  # The HIP-gated TEST is in here for the same reason: it is the first thing a
  # contributor runs, it is also plain C++, and a stale assertion in it would
  # read as a backend bug on their machine rather than as our drift.
  add_library(vllm_rocm_platform_syntax_check OBJECT
    src/vllm/platforms/rocm.cpp
    tests/vt/test_rocm_backend.cpp)
  target_include_directories(vllm_rocm_platform_syntax_check PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
  target_include_directories(vllm_rocm_platform_syntax_check SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party)
  target_compile_features(vllm_rocm_platform_syntax_check PRIVATE cxx_std_20)
  vllm_cpp_set_warnings(vllm_rocm_platform_syntax_check)
endif()
if(VLLM_CPP_CUDA)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_CUDA)
  if(VLLM_CPP_BENCH_PROFILE_CONTROL)
    target_compile_definitions(vllm PUBLIC VT_BENCH_PROFILE_CONTROL=1)
  endif()
  target_sources(vllm PRIVATE
    src/vllm/platforms/cuda.cpp
    src/vt/cuda/cuda_arch_tactics.cu
    src/vt/cuda/cuda_backend.cu
    src/vt/cuda/nccl_communicator.cu
    src/vt/cuda/cuda_dropin.cu
    src/vt/cuda/cuda_matmul.cu
    src/vt/cuda/cuda_quant_dot.cu
    src/vt/cuda/cuda_matmul_nvfp4.cu
    src/vt/cuda/cuda_ops.cu
    src/vt/cuda/cuda_gdn.cu
    src/vt/cuda/cuda_moe.cu
    src/vt/cuda/cuda_glue.cu
    src/vt/cuda/cuda_layernorm.cu
    src/vt/cuda/cuda_cache.cu
    src/vt/cuda/cuda_mla_attn.cu
    src/vt/cuda/cuda_mla_prefill.cu
    src/vt/cuda/cuda_paged_attn.cu
    src/vt/cuda/cuda_combine_tokens.cu
    src/vt/cuda/cuda_sample.cu
    src/vt/cuda/cuda_deepseek_v4.cu
    src/vt/cuda/cuda_laguna.cu
    src/vt/cuda/cuda_minimax_h3.cu)
  find_package(CUDAToolkit REQUIRED)
  # cublasLt is linked now so the Task 4 matmul lands without a build change.
  target_link_libraries(vllm PUBLIC CUDA::cudart CUDA::cublasLt)
  # NCCL tensor-parallel transport (BACKEND-DISTRIBUTED-COMM W2, DERIVE-AND-SHIP).
  # OFF by default so the standing single-GPU CUDA gate is untouched: without it
  # src/vt/cuda/nccl_communicator.cu compiles to a stub (no kCUDA collective
  # provider). Turn ON on a >=2-GPU host to build-verify + run the real
  # ncclAllReduce/Send/Recv path (pynccl.py mirror). AUTO-detect could be added,
  # but an explicit opt-in keeps the default CUDA build reproducible.
  option(VLLM_CPP_NCCL "Build the NCCL tensor-parallel transport (needs nccl.h + libnccl)" OFF)
  if(VLLM_CPP_NCCL)
    find_library(NCCL_LIBRARY NAMES nccl REQUIRED)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h REQUIRED)
    target_compile_definitions(vllm PRIVATE VT_NCCL=1)
    target_include_directories(vllm PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(vllm PUBLIC ${NCCL_LIBRARY})
    message(STATUS "NCCL TP transport: ON (${NCCL_LIBRARY})")
  endif()
  # The cutlass sm120a NVFP4 GEMM TU is isolated: its own heavy cutlass include
  # tree + relaxed-constexpr/extended-lambda, so it never touches the general
  # build. Gated by VT_CUTLASS_NVFP4 so the host op path no-ops if not compiled.
  if(VLLM_CPP_CUTLASS)
    # The 32 NVFP4 tactics are split four instantiations per tile TU. A single
    # monolithic nvcc process for all swap/scheduler variants exceeds practical
    # compiler memory on GB10; the dispatcher and immutable W1 fallback remain
    # separate so VT_FP4_FULL_TACTICS=0 is an exact same-binary comparison.
    set(_NVFP4_FLASHINFER_TACTIC_SOURCES
      src/vt/cuda/cuda_nvfp4_tactics_128x32x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x32x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x64x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x64x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x128x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x128x256.cu
      src/vt/cuda/cuda_nvfp4_tactics_256x128x128.cu
      src/vt/cuda/cuda_nvfp4_tactics_128x256x128.cu)
    set(_NVFP4_CUTLASS_SOURCES
      src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu
      src/vt/cuda/cuda_nvfp4_tactics_w1.cu
      ${_NVFP4_FLASHINFER_TACTIC_SOURCES})
    # Persistent tactic caches must become incompatible whenever executable
    # tactic source/ABI changes, including a dirty/source-archive build with no
    # meaningful git revision. Hash the exact compile inputs deterministically.
    set(_NVFP4_CACHE_ABI_SOURCES
      ${_NVFP4_CUTLASS_SOURCES}
      src/vt/cuda/nvfp4_cutlass_tactics.h
      src/vt/cuda/nvfp4_plan_cache.h
      src/vt/cuda/nvfp4_tactic_ids.h)
    set(_NVFP4_CACHE_BUILD_INPUT
      "cutlass=${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${_NVFP4_CACHE_ABI_SOURCES})
    foreach(_NVFP4_CACHE_ABI_SOURCE IN LISTS _NVFP4_CACHE_ABI_SOURCES)
      file(SHA256
        "${CMAKE_CURRENT_SOURCE_DIR}/${_NVFP4_CACHE_ABI_SOURCE}"
        _NVFP4_CACHE_ABI_SHA)
      string(APPEND _NVFP4_CACHE_BUILD_INPUT
        "${_NVFP4_CACHE_ABI_SOURCE}=${_NVFP4_CACHE_ABI_SHA};")
    endforeach()
    string(SHA256 _NVFP4_CACHE_BUILD_ID "${_NVFP4_CACHE_BUILD_INPUT}")
    # cutlass-fp8 is its own FEATURE-TABLE row: its ArchTag=Sm120 scaled-mm has a
    # different upstream arch set from the NVFP4 block-scaled collective (vLLM
    # keeps them as separate SCALED_MM_ARCHS / FP4_SM120_ARCHS intersections), so
    # it is resolved separately here rather than riding on the NVFP4 gate.
    set(_FP8_CUTLASS_SOURCES)
    if(VT_CUTLASS_FP8_ARCHS)
      set(_FP8_CUTLASS_SOURCES src/vt/cuda/cuda_matmul_fp8_cutlass.cu)
    endif()
    target_sources(vllm PRIVATE
      ${_NVFP4_CUTLASS_SOURCES}
      ${_FP8_CUTLASS_SOURCES})
    target_compile_definitions(vllm PUBLIC VT_CUTLASS_NVFP4=1)
    if(VT_CUTLASS_FP8_ARCHS)
      target_compile_definitions(vllm PUBLIC VT_CUTLASS_FP8=1)
    endif()
    target_compile_definitions(vllm PRIVATE
      VT_CUTLASS_VERSION_STRING="${_CUTLASS_MAJOR}.${_CUTLASS_MINOR}.${_CUTLASS_PATCH}"
      VT_NVFP4_CACHE_BUILD_ID="${_NVFP4_CACHE_BUILD_ID}")
    # cutlass headers reach via -isystem so their internal deprecation warnings
    # are exempt from the global -Werror=all-warnings (the general build stays
    # strict). --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them.
    set_source_files_properties(
      ${_NVFP4_CUTLASS_SOURCES}
      ${_FP8_CUTLASS_SOURCES} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
    # Keep the split 32-tactic family on FlashInfer's semantic CUDA compile
    # envelope. These definitions are source-scoped: the general CUDA backend,
    # immutable W1 fallback and FP8 adapter retain the project defaults.
    set_property(SOURCE ${_NVFP4_FLASHINFER_TACTIC_SOURCES} APPEND PROPERTY
      COMPILE_DEFINITIONS
        CUTLASS_ENABLE_GDC_FOR_SM100=1
        FLASHINFER_ENABLE_FP8_E8M0
        FLASHINFER_ENABLE_FP4_E2M1
        FLASHINFER_ENABLE_F16
        FLASHINFER_ENABLE_BF16
        FLASHINFER_ENABLE_FP8_E4M3
        FLASHINFER_ENABLE_FP8_E5M2
        ENABLE_BF16
        ENABLE_FP4)
    set_property(SOURCE ${_NVFP4_FLASHINFER_TACTIC_SOURCES} APPEND PROPERTY
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false;-use_fast_math>")
  endif()
  # Datacenter-Blackwell sm_100a NVFP4 tcgen05 GEMM BUILD-VERIFY TU
  # (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA). Compiled ONLY for a single-arch 100a
  # target — VT_CUTLASS_NVFP4_SM100 is set (above) exclusively when the
  # `cutlass-nvfp4-sm100` FEATURE-TABLE cell resolves for the requested arch, which
  # is 100a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the consumer sm_12x tactic
  # sweep): the datacenter body uses ArchTag=Sm100 + KernelScheduleAuto (tcgen05),
  # a different collective/tile/cluster than the sm_120 body, so the two never
  # share a compile (widening the sm_12x gate to 100a would ptxas-reject the sm_120
  # PTX for compute_100a). On the gate arch sm_121a this flag is OFF, so the TU is
  # not added to the build at all → byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them;
  #   * -diag-suppress=20012: benign cutlass sm100_static_tile_scheduler.hpp
  #     "__host__/__device__ annotation ignored on defaulted function" front-end
  #     diagnostic (mirrors the marlin -diag-suppress=20280 precedent below).
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_100a
  # SASS is NOT execution evidence — no B200/sm_100 board ran this.
  if(VLLM_CPP_CUTLASS_NVFP4_SM100)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_matmul_nvfp4_sm100.cu)
    target_compile_definitions(vllm PUBLIC VT_CUTLASS_NVFP4_SM100=1)
    set_source_files_properties(
      src/vt/cuda/cuda_matmul_nvfp4_sm100.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-diag-suppress=20012;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Hopper sm_90a CUTLASS C3x FP8 scaled-mm wgmma/TMA GEMM BUILD-VERIFY TU
  # (BACKEND-CUDA-SM090, ROAD-V1-D1-CUDA, datacenter fast-path §9 DC2). Compiled
  # ONLY for a single-arch 90a target — VLLM_CPP_SCALEDMM_C3X_SM90 is set (above)
  # exclusively when the `scaledmm-c3x-sm90` FEATURE-TABLE cell resolves for the
  # requested arch, which is 90a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the
  # consumer sm_12x scaled-mm): the Hopper body uses ArchTag=Sm90 +
  # KernelTmaWarpSpecialized*FP8FastAccum (wgmma/TMA), a different collective than
  # the sm_120 body, so the two never share a compile (widening the sm_12x gate to
  # 90a would ptxas-reject the sm_120 PTX for compute_90a). On the gate arch
  # sm_121a this flag is OFF, so the TU is not added to the build at all →
  # byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x/sm_100a cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them.
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_90a
  # SASS is NOT execution evidence — no H100/H200/sm_90 board ran this.
  if(VLLM_CPP_SCALEDMM_C3X_SM90)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu)
    target_compile_definitions(vllm PUBLIC VT_SCALEDMM_C3X_SM90=1)
    set_source_files_properties(
      src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Datacenter-Blackwell sm_100a CUTLASS C3x FP8 scaled-mm tcgen05 GEMM BUILD-VERIFY
  # TU (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA, datacenter fast-path §9 DC3). Compiled
  # ONLY for a single-arch 100a target — VLLM_CPP_SCALEDMM_C3X_SM100 is set (above)
  # exclusively when the `scaledmm-c3x-sm100` FEATURE-TABLE cell resolves for the
  # requested arch, which is 100a alone. It is SEPARATE from VLLM_CPP_CUTLASS (the
  # consumer sm_12x scaled-mm) AND from VLLM_CPP_SCALEDMM_C3X_SM90 (the Hopper leg):
  # the datacenter body uses ArchTag=Sm100 + KernelScheduleAuto (tcgen05), a
  # different collective than the sm_120 body and the Sm90 wgmma body, so no two
  # share a compile (widening the sm_12x gate to 100a would ptxas-reject the sm_120
  # PTX for compute_100a). On the gate arch sm_121a this flag is OFF, so the TU is
  # not added to the build at all → byte-zero impact on GB10.
  #   * -isystem cutlass so its headers are off the global -Werror=all-warnings
  #     path (mirrors the sm_12x/sm_90a/sm_100a-NVFP4 cutlass TUs above);
  #   * --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them;
  #   * -diag-suppress=20012: benign cutlass sm100_static_tile_scheduler.hpp
  #     "__host__/__device__ annotation ignored on defaulted function" front-end
  #     diagnostic (same tcgen05 header the sm_100a NVFP4 TU suppresses).
  # DERIVED+BUILD-VERIFIED (testing-welcome): a green compile + cuobjdump sm_100a
  # SASS is NOT execution evidence — no B200/sm_100 board ran this.
  if(VLLM_CPP_SCALEDMM_C3X_SM100)
    target_sources(vllm PRIVATE src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu)
    target_compile_definitions(vllm PUBLIC VT_SCALEDMM_C3X_SM100=1)
    set_source_files_properties(
      src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-diag-suppress=20012;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
  endif()
  # Marlin NVFP4 MoE: two vendored TUs (the marlin_mm dispatcher + the generated
  # bf16 NVFP4 kernel instantiations) + the vt::Tensor launcher. All three need
  #  * -static-global-template-stub=false so the __global__ Marlin<> template
  #    instantiations get external linkage across the two vendored TUs (nvcc 13
  #    defaults to static stubs -> undefined refs at link; verified on GB10);
  #  * -isystem the vendored root so kernel-side warnings (e.g. #20280) are
  #    exempt from the global -Werror=all-warnings; the general build stays strict;
  #  * --expt-relaxed-constexpr (scalar_type.hpp constexpr).
  if(VLLM_CPP_MARLIN)
    set(_MARLIN_ROOT "${CMAKE_SOURCE_DIR}/src/vt/cuda/marlin")
    set(_MARLIN_SRCS
      src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/marlin_mm_moe.cu
      src/vt/cuda/marlin/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
      src/vt/cuda/cuda_moe_marlin.cu
      src/vt/cuda/cuda_marlin_repack.cu
      # DENSE marlin (row KERNEL-MARLIN-DENSE-PORT): the dense marlin_mm dispatcher
      # (`namespace marlin`) + its bf16 NVFP4/MXFP4 kernel instantiations (SAME 12-param
      # Marlin<> template as the MoE TUs, distinct namespace + direct-A kernel body) +
      # the vt::Tensor launcher. Same nvcc options as the MoE TUs below.
      src/vt/cuda/marlin/libtorch_stable/quantization/marlin/marlin_mm_dense.cu
      src/vt/cuda/marlin/libtorch_stable/quantization/marlin/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
      src/vt/cuda/cuda_marlin_dense.cu)
    target_sources(vllm PRIVATE ${_MARLIN_SRCS})
    target_compile_definitions(vllm PUBLIC VT_MARLIN_NVFP4=1)
    set_source_files_properties(${_MARLIN_SRCS} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false;--expt-relaxed-constexpr;-diag-suppress=20280;-isystem;${_MARLIN_ROOT}>")
  endif()

  # ── SANCTIONED Triton AOT fast-path: GDN delta_h (VLLM_CPP_TRITON, default OFF)
  # The FLA chunk_delta_h state-recurrence kernel as an embedded cubin in libvllm
  # (VENDORED per-arch — the build needs no Python; VLLM_CPP_TRITON_REGEN=ON is
  # the maintainer regen); runtime dispatch (behind VT_GDN_DELTAH_TRITON) is in
  # cuda_gdn.cu with the hand-C++ GdnChunkDeltaHRegRingKernel preserved as the
  # default fallback. OFF => none of this runs and the build is byte-identical.
  # Two specializations, one per gate-model GDN shape (both K=V=128, Hg=16, BT=64;
  # only H=num_value_heads differs): 27B H=48, 35B H=32. Signature order matches
  # triton_kernels/chunk_delta_h.py arg list; pinned flags USE_G=1,USE_GK=0,
  # USE_INITIAL_STATE=1,STORE_FINAL_STATE=1,SAVE_NEW_VALUE=1,IS_VARLEN=1,USE_EXP2=0.
  if(VLLM_CPP_TRITON)
    enable_language(C)  # the AOT launchers are plain C (CUDA driver API)
    include(cmake/TritonAOTKernels.cmake)
    vllm_triton_aot_declare_all()
    vllm_triton_aot_expected_lines(_TRITON_AOT_CONTRACT_LINES)
    set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_CONTRACT_LINES
      "${_TRITON_AOT_CONTRACT_LINES}")
    # Launch-config pins, PER KERNEL and PER GDN SHAPE (H=48 27B, H=32 35B),
    # set to the winning config of FLA's OWN @triton.autotune run at the exact
    # engine shapes on GB10 (tools: ~/work/finisher_fla_autotune.py; see
    # .agents/parity-ledger.md 2026-07-10 for the measured best_configs).
    # Overridable cache vars to re-sweep without editing this file.
    # gridX = cdiv(V=128, BV) baked per spec where the kernel is BV-tiled.
    foreach(_H 48 32)
      set(VT_GDN_DELTAH_BV_${_H}     64 CACHE STRING "delta_h H=${_H} Triton BV (32 or 64)")
      set(VT_GDN_DELTAH_WARPS_${_H}  4  CACHE STRING "delta_h H=${_H} Triton num_warps")
      set(VT_GDN_DELTAH_STAGES_${_H} 3  CACHE STRING "delta_h H=${_H} Triton num_stages")
    endforeach()
    set(_DELTAH_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_delta_h.py")
    set(_DELTAH_KFN chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    # Signature (26 args): k v w v_new g gk h h0 ht cu_seqlens chunk_offsets T NH
    #   H Hg K V BT BV USE_G USE_GK USE_INITIAL_STATE STORE_FINAL_STATE
    #   SAVE_NEW_VALUE IS_VARLEN USE_EXP2. dtypes: k/v/w/v_new/h bf16, g/h0/ht fp32.
    set(_DELTAH_SIG_HEAD "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32")
    foreach(_H 48 32)
      set(_DELTAH_SIG_TAIL "*bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, ${_H}, 16, 128, 128, 64, ${VT_GDN_DELTAH_BV_${_H}}, 1, 0, 1, 1, 1, 1, 0")
      math(EXPR _DELTAH_GX "(128 + ${VT_GDN_DELTAH_BV_${_H}} - 1) / ${VT_GDN_DELTAH_BV_${_H}}")
      add_triton_kernel(VT_GDN_DELTAH_${_H}_SRCS
        "${_DELTAH_KPY}" ${_DELTAH_KFN} gdn_deltah_h${_H}
        "${_DELTAH_SIG_HEAD}, ${_DELTAH_SIG_TAIL}" "${_DELTAH_GX},NH,1"
        ${VT_GDN_DELTAH_WARPS_${_H}} ${VT_GDN_DELTAH_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_DELTAH_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_DELTAH_${_H}_SRCS_INCLUDE_DIR})
    endforeach()

    # ── SANCTIONED Triton AOT: GDN chunk_o (the output kernel) ──────────────────
    # FLA chunk_o.py chunk_fwd_kernel_o -> triton_kernels/chunk_o.py. Dispatch
    # TryTritonChunkO (VT_GDN_CHUNKO_TRITON) in cuda_gdn.cu; hand GdnChunkOWmmaKernel
    # preserved as fallback. Two specs per output dtype (H=48 27B, H=32 35B);
    # K=V=128, Hg=16, BT=64, BK=BV=64 (grid.x=cdiv(V,BV)=2). USE_G=1,
    # IS_VARLEN=1. o(=out) is f32 for the default GDN recurrence-output dtype and
    # bf16 for the vLLM-faithful core/z dtype follow-up. Grid (cdiv(V,BV), NT, H);
    # H=B*H baked (varlen B=1), NT is the runtime grid-carrier scalar. scale is
    # PINNED to Dk^-0.5 inside the kernel (Triton AOT mis-packs fp32 scalars), NOT
    # a sig arg.
    # Args: q k v h g o cu_seqlens chunk_indices T NT | H Hg K V BT BK BV USE_G IS_VARLEN.
    foreach(_H 48 32)
      set(VT_GDN_CHUNKO_BK_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BK")
      set(VT_GDN_CHUNKO_BV_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BV")
      set(VT_GDN_CHUNKO_WARPS_${_H}  4  CACHE STRING "chunk_o H=${_H} Triton num_warps")
      set(VT_GDN_CHUNKO_STAGES_${_H} 3  CACHE STRING "chunk_o H=${_H} Triton num_stages")
    endforeach()
    set(_CHUNKO_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_o.py")
    set(_CHUNKO_SIG_HEAD_F32 "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    set(_CHUNKO_SIG_HEAD_BF16 "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
    foreach(_H 48 32)
      set(_CHUNKO_SIG_TAIL "16, 128, 128, 64, ${VT_GDN_CHUNKO_BK_${_H}}, ${VT_GDN_CHUNKO_BV_${_H}}, 1, 1")
      math(EXPR _CHUNKO_GX "(128 + ${VT_GDN_CHUNKO_BV_${_H}} - 1) / ${VT_GDN_CHUNKO_BV_${_H}}")
      set(_sig "${_CHUNKO_SIG_HEAD_F32}, ${_H}, ${_CHUNKO_SIG_TAIL}")
      add_triton_kernel(VT_GDN_CHUNKO_${_H}_SRCS
        "${_CHUNKO_KPY}" chunk_fwd_kernel_o gdn_chunko_h${_H}
        "${_sig}" "${_CHUNKO_GX},NT,${_H}"
        ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_CHUNKO_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_CHUNKO_${_H}_SRCS_INCLUDE_DIR})

      set(_sig "${_CHUNKO_SIG_HEAD_BF16}, ${_H}, ${_CHUNKO_SIG_TAIL}")
      add_triton_kernel(VT_GDN_CHUNKO_BF16_${_H}_SRCS
        "${_CHUNKO_KPY}" chunk_fwd_kernel_o gdn_chunko_bf16_h${_H}
        "${_sig}" "${_CHUNKO_GX},NT,${_H}"
        ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS_INCLUDE_DIR})
    endforeach()
    target_compile_definitions(vllm PUBLIC VLLM_CPP_TRITON_CHUNKO_BF16=1)

    # ── SANCTIONED Triton AOT: GDN packed pure-decode recurrence ────────────────
    # FLA fused_recurrent.py:256-336 fused_recurrent_gated_delta_rule_packed_decode
    # _kernel -> triton_kernels/fused_recurrent_packed_decode.py. Dispatch
    # TryTritonPackedDecode (VT_GDN_PACKED_DECODE_TRITON, default ON — vLLM's exact
    # token-identical FLA kernel, MIRROR policy; =0 rolls back to the hand
    # GdnPackedDecodeKernel in the same binary, which also stays the portable
    # fallback for any unsupported shape). MEASURED codegen-bound (dgx phase1 2026-07-16): the
    # register-resident [BV=32,BK=128] fp32 state tile is REG:205/0-spill under
    # Triton but REG:255+STACK:48 (spills) as hand-CUDA. Dense packed-decode
    # specializations cover Hk=16, Hv in {48,32}, K=V=128, BK=128, BV=32; the
    # 35B MoE model does not select packed decode. num_warps=1,
    # num_stages=3 (FLA launch :465-466). grid.x = cdiv(V,BV) = 4 baked; grid.y =
    # NBH (= B*HV) is the runtime carrier. scale PINNED to Dk^-0.5 in-kernel
    # (Triton AOT mis-packs fp32 scalars). Strides/dims are BAKED to each dense
    # call site and GUARDED in TryTritonPackedDecode. State-index ABI adapter:
    # skip `state_idx < 0` (our
    # cache ABI; FLA skips <= 0). Args: mixed_qkv a b A_log dt_bias o h0 ht
    # ssm_state_indices NBH | strides... H HV K V BK BV SOFTPLUS_THRESHOLD
    # USE_QK_L2NORM_IN_KERNEL. dtypes: mixed_qkv/a/b/o bf16, A_log/dt_bias/h0/ht
    # fp32, indices i32.
    set(_DECODE_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/fused_recurrent_packed_decode.py")
    set(_DECODE_KFN fused_recurrent_gated_delta_rule_packed_decode_kernel)
    foreach(_HV 48 32)
      set(VT_GDN_DECODE_WARPS_${_HV} 1 CACHE STRING
          "packed decode Hv=${_HV} Triton num_warps")
      set(VT_GDN_DECODE_STAGES_${_HV} 3 CACHE STRING
          "packed decode Hv=${_HV} Triton num_stages")
      if(_HV EQUAL 48)
        set(_DECODE_MIXED_STRIDE 10240)
        set(_DECODE_BA_STRIDE 96)
        set(_DECODE_STATE_STRIDE 786432)
      else()
        set(_DECODE_MIXED_STRIDE 8192)
        set(_DECODE_BA_STRIDE 64)
        set(_DECODE_STATE_STRIDE 524288)
      endif()
      set(_DECODE_SIG "*bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, i32, ${_DECODE_MIXED_STRIDE}, ${_DECODE_BA_STRIDE}, ${_DECODE_BA_STRIDE}, ${_DECODE_STATE_STRIDE}, ${_DECODE_STATE_STRIDE}, 1, 16, ${_HV}, 128, 128, 128, 32, 20, 1")
      add_triton_kernel(VT_GDN_DECODE_${_HV}_SRCS
        "${_DECODE_KPY}" ${_DECODE_KFN} gdn_decode_h${_HV}
        "${_DECODE_SIG}" "4,NBH,1"
        ${VT_GDN_DECODE_WARPS_${_HV}} ${VT_GDN_DECODE_STAGES_${_HV}})
      target_sources(vllm PRIVATE ${VT_GDN_DECODE_${_HV}_SRCS})
      target_include_directories(vllm PRIVATE
        ${VT_GDN_DECODE_${_HV}_SRCS_INCLUDE_DIR})
    endforeach()

    # ── SANCTIONED Triton AOT: GDN WU pipeline (kkt -> solve_tril -> recompute_w_u)
    # The 3 FLA kernels our single fused GdnChunkWUWmmaVecKernel mirrors; ported so
    # the WU (biggest ~25% GDN) share runs on Triton codegen. Dispatch TryTritonWU
    # (VT_GDN_WU_TRITON) in cuda_gdn.cu runs all 3 into scratch A(f32)/Ai(bf16) and
    # produces w,u; hand fused kernel preserved as fallback. All grid-y = H (varlen
    # B=1, baked), grid-x = NT (runtime carrier). H=48 (27B) / 32 (35B).
    # Pins verified against FLA's OWN autotuner (GB10 2026-07-10, exact engine
    # shapes, T-stable 1024..8192; ~/work/finisher_autotune.log) AND per-kernel
    # nsys on OUR AOT variants (.agents/parity-ledger.md 2026-07-10):
    #  * delta_h BV=64/w4/s3, chunk_o BK=BV=64/w4/s3: FLA winners == old pins
    #    (CONFIRMED optimal, both H).
    #  * kkt H=48: FLA winner BK=128/w8/s3 measured −2.7% vs BK=64/w4/s3 → pinned.
    #  * tril/wu H=48: FLA winners (w8/s5, w2/s2) measured SLOWER on our AOT
    #    variants (+6.8%, +1.7% — the AOT specialization shifts the optimum) →
    #    keep the proven w4/s3 pins. Recorded as a measured dead-end.
    #  * H=32 (35B): kkt FLA winner == old pin; tril w8/s3 + wu w4/s2 are FLA
    #    winners but UNMEASURED on our AOT (and the H=48 measurement shows FLA
    #    winners don't transfer blindly) → keep proven w4/s3 pins; candidates
    #    recorded in the ledger.
    set(VT_GDN_KKT_BK_48      128 CACHE STRING "kkt H=48 Triton BK (32/64/128)")
    set(VT_GDN_KKT_WARPS_48   8   CACHE STRING "kkt H=48 Triton num_warps")
    set(VT_GDN_KKT_STAGES_48  3   CACHE STRING "kkt H=48 Triton num_stages")
    set(VT_GDN_KKT_BK_32      64  CACHE STRING "kkt H=32 Triton BK (32/64/128)")
    set(VT_GDN_KKT_WARPS_32   4   CACHE STRING "kkt H=32 Triton num_warps")
    set(VT_GDN_KKT_STAGES_32  3   CACHE STRING "kkt H=32 Triton num_stages")
    set(VT_GDN_TRIL_WARPS_48  4   CACHE STRING "solve_tril H=48 Triton num_warps")
    set(VT_GDN_TRIL_STAGES_48 3   CACHE STRING "solve_tril H=48 Triton num_stages")
    set(VT_GDN_TRIL_WARPS_32  4   CACHE STRING "solve_tril H=32 Triton num_warps")
    set(VT_GDN_TRIL_STAGES_32 3   CACHE STRING "solve_tril H=32 Triton num_stages")
    set(VT_GDN_WU_WARPS_48    4   CACHE STRING "recompute_w_u H=48 Triton num_warps")
    set(VT_GDN_WU_STAGES_48   3   CACHE STRING "recompute_w_u H=48 Triton num_stages")
    set(VT_GDN_WU_WARPS_32    4   CACHE STRING "recompute_w_u H=32 Triton num_warps")
    set(VT_GDN_WU_STAGES_32   3   CACHE STRING "recompute_w_u H=32 Triton num_stages")
    # (1) chunk_scaled_dot_kkt_fwd_kernel: k beta g A cu_seqlens chunk_indices T NT |
    #     H Hg K BT BK IS_VARLEN USE_G. A out f32.
    set(_KKT_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/chunk_scaled_dot_kkt.py")
    set(_KKT_HEAD "*bf16:16, *fp32:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    # (2) merge_16x16_to_64x64_inverse_kernel: A Ai cu_seqlens chunk_indices T NT |
    #     H BT IS_VARLEN. A in f32, Ai out bf16. USE_TMA/DOT_PRECISION hardcoded.
    set(_TRIL_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/solve_tril.py")
    set(_TRIL_HEAD "*fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
    set(_TRIL_TAIL "64, 1")
    # (3) recompute_w_u_fwd_kernel: k v beta w u A g cu_seqlens chunk_indices T NT |
    #     H Hg K V BT BK BV IS_VARLEN. BK=BV=64 (pinned by the FLA wrapper,
    #     wy_fast.py:135-136 — NOT autotuned there). A in bf16(=Ai); w,u out bf16.
    set(_WU_KPY "${CMAKE_SOURCE_DIR}/triton_kernels/wy_fast.py")
    set(_WU_HEAD "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32")
    set(_WU_TAIL "16, 128, 128, 64, 64, 64, 1")
    foreach(_H 48 32)
      set(_KKT_TAIL "16, 128, 64, ${VT_GDN_KKT_BK_${_H}}, 1, 1")
      add_triton_kernel(VT_GDN_KKT_${_H}_SRCS
        "${_KKT_KPY}" chunk_scaled_dot_kkt_fwd_kernel gdn_kkt_h${_H}
        "${_KKT_HEAD}, ${_H}, ${_KKT_TAIL}" "NT,${_H},1"
        ${VT_GDN_KKT_WARPS_${_H}} ${VT_GDN_KKT_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_KKT_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_KKT_${_H}_SRCS_INCLUDE_DIR})

      add_triton_kernel(VT_GDN_TRIL_${_H}_SRCS
        "${_TRIL_KPY}" merge_16x16_to_64x64_inverse_kernel gdn_tril_h${_H}
        "${_TRIL_HEAD}, ${_H}, ${_TRIL_TAIL}" "NT,${_H},1"
        ${VT_GDN_TRIL_WARPS_${_H}} ${VT_GDN_TRIL_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_TRIL_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_TRIL_${_H}_SRCS_INCLUDE_DIR})

      add_triton_kernel(VT_GDN_WU_${_H}_SRCS
        "${_WU_KPY}" recompute_w_u_fwd_kernel gdn_wu_h${_H}
        "${_WU_HEAD}, ${_H}, ${_WU_TAIL}" "NT,${_H},1"
        ${VT_GDN_WU_WARPS_${_H}} ${VT_GDN_WU_STAGES_${_H}})
      target_sources(vllm PRIVATE ${VT_GDN_WU_${_H}_SRCS})
      target_include_directories(vllm PRIVATE ${VT_GDN_WU_${_H}_SRCS_INCLUDE_DIR})
    endforeach()

    # Vendored-tree bookkeeping (call ONCE, after the LAST add_triton_kernel):
    # REGEN writes the MANIFEST (generator versions + triton_kernels/*.py hashes
    # + per-base generation params); the normal builder path verifies those
    # hashes against the vendored tree and WARNS loudly on drift.
    triton_aot_finalize()

    target_compile_definitions(vllm PUBLIC VLLM_CPP_TRITON=1)
    target_link_libraries(vllm PUBLIC CUDA::cuda_driver)
  endif()

  # Vendored FlashAttention-2 sm_121a prefill + pure-decode split-KV. The exact
  # FA-2 source vLLM builds
  # as _vllm_fa2_C (vllm-project/flash-attention @ 2c839c33) — its
  # flash_fwd_splitkv kernel is what vLLM's nsys trace shows for prefill on GB10.
  # Compiled torch-free against our CUTLASS: 3 stub headers replace the ATen/c10
  # surface (PhiloxCudaState POD + C10_CUDA_CHECK), and fa2_compat_prelude.h is
  # force-included for the std headers ATen pulled in transitively — the upstream
  # (auto-generated) FA sources stay byte-for-byte pristine. DEFAULT ON (like
  # Marlin/cutlass) — it only actually builds when CUTLASS is present
  # (VLLM_CPP_CUTLASS, sm_12xa); -DVLLM_CPP_FLASH_ATTN=OFF opts out. Runtime
  # toggles VT_FA2_PREFILL and VT_FA2_DECODE (each default ON when compiled;
  # =0 restores its same-binary fallback). Prefill was validated token-for-token
  # + over the throughput bar on GB10 2026-07-10; ratio-6 decode remains GATING.
  option(VLLM_CPP_FLASH_ATTN "Build vendored FlashAttention-2 prefill/decode kernels (needs CUTLASS headers)" ON)
  # Gate on VLLM_CPP_CUTLASS_HEADERS (arch-independent header availability), NOT
  # VLLM_CPP_CUTLASS (the sm_12x NVFP4 feature flag): FA2 also resolves for Ampere
  # sm_8x now (WA-1), where the NVFP4 GEMM is absent but the headers still compile.
  if(VLLM_CPP_FLASH_ATTN AND VLLM_CPP_CUTLASS_HEADERS AND VT_FA2_ARCHS)
    set(_FA2_ROOT "${CMAKE_SOURCE_DIR}/src/vt/cuda/flash_attn")
    set(_FA2_KERNEL_SRCS
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu
      # d128 bf16 split-KV: the exact varlen (non-swap) reduction vLLM's
      # flash_attn_varlen_func runs for Qwen3-dense DECODE (paged KV forces the
      # split kernel even at num_splits==1). MODEL-TEXT-qwen3 strict bit-match.
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim128_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu
      # d192 bf16 split-KV (MLA campaign W5): the MLA PREFILL QK head dim
      # (qk_nope 128 + qk_rope 64). V is zero-padded 128 -> 192 exactly as
      # upstream does (mla/prefill/flash_attn.py:164-168), so the kernel stays a
      # symmetric instantiation of the SAME generic template — no template edit,
      # hence the d128/d256 kernels the 27B/35B/dense paths call are unchanged.
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim192_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu)
    target_sources(vllm PRIVATE
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS})
    target_compile_definitions(vllm PUBLIC VLLM_CPP_FLASH_ATTN)
    # NOTE (QUANT-CT-MXFP4-FLASH-AUDIT #68 + FLASH-OCCUPANCY #75): the c8 decode
    # flash gap vs vLLM (ours ~168-174 vs vLLM ~156 us/call) is a COMPILER-CODEGEN
    # gap, MEASURED to a verdict — NOT occupancy, NOT L2, NOT num_splits, and NOT a
    # flag we can set. The owed matched-c8 ncu (both engines, IDENTICAL grid
    # 1x3x64) shows occupancy is IDENTICAL 8.33% (both smem-limited to 1 CTA/SM by
    # the byte-identical 81.92 KB kSmemSize — the 216-vs-241 reg delta is moot),
    # L2 hit ~1% on BOTH (KV streamed from DRAM: no adjacency/warmth to win), and
    # the same ~60% short-scoreboard stall structure. The ONLY difference: vLLM's
    # SASS executes ~13% FEWER instructions (3.27M vs 3.70M) in this latency-bound
    # ~1-warp/scheduler regime — but that instruction count is NOT the cause. #75
    # built the exact vLLM recipe: compiling the kernel TUs for compute_80 PTX +
    # -use_fast_math REPRODUCES vLLM's SASS profile EXACTLY (241 reg, 17,008 instr
    # vs vLLM 241/17,020) and is STILL ~167 us — the SAME as our native ~165 us and
    # still ~10 us slower than vLLM's ~157 us. So matching vLLM's arch, fast-math,
    # register AND instruction counts does NOT close the gap: the kernel is not
    # instruction-bound, and the residual is vLLM's wheel-`ptxas` SASS-SCHEDULING
    # quality, not reachable from our nvcc 13.0. (native+fast-math REGRESSES to
    # 189.8 us, #68; compute_80 PTX driver-JIT alone is neutral, #75 Build B.) So
    # we keep the plain native sm_121a compile and pass no -use_fast_math. #82
    # (FLASH-PTXAS) then ARBITRATED the "ptxas lineage" claim to a NO and CORRECTED
    # it: the wheel ships NO sm_12x cubin — only CUDA-13.0 (PTX ISA 9.0) sm_80 PTX
    # driver-JIT'd on GB10 — so there is no old-CUDA-12.x ptxas in vLLM's path (and
    # ptxas 12.8 cannot even target sm_121a); a same-params cuModule A/B ties our
    # PTX and vLLM's OWN PTX at ~144 us across driver-JIT / ptxas 13.0 / 13.2, so the
    # +10 us/call gap is ENGINE CONTEXT (neighbour L2/orchestration), not this
    # kernel's codegen. See .agents/benchmark-record.md (#75, #82).
    set_source_files_properties(
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-DFLASH_NAMESPACE=flash;-include;${_FA2_ROOT}/fa2_compat_prelude.h;-isystem;${_FA2_ROOT}/src;-isystem;${_FA2_ROOT}/stubs;-isystem;${VLLM_CPP_CUTLASS_DIR}/include>")
    message(STATUS "FlashAttention-2 prefill/decode: ENABLED for arch(es) [${VT_FA2_ARCHS}] (runtime toggles VT_FA2_PREFILL, VT_FA2_DECODE)")
  endif()
endif()

# OpenAI HTTP server (M3.1 Task 4): the api_server.cpp translation unit pulls in
# the vendored cpp-httplib transport header, so it is compiled into the library
# only when VLLM_CPP_SERVER is enabled. httplib needs threads (and sockets).
if(VLLM_CPP_SERVER)
  find_package(Threads REQUIRED)
  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_SERVER)
  target_link_libraries(vllm PUBLIC Threads::Threads)
  # third_party/httplib/httplib.h is reached as <httplib/httplib.h> (third_party
  # is already a SYSTEM include on the vllm target).
endif()

# ── Packaged shared/static libvllm (M3.5 Task 3, library-first packaging) ─────
# The static `vllm` core lib above stays as-is (tests + examples link it and
# reach internal C++ symbols). Here we ALSO package a SHARED libvllm.so that
# exports ONLY the stable C ABI (include/vllm.h): a linker version script keeps
# every `vllm_*` symbol global and localizes every C++ internal, so an FFI /
# dlopen consumer (LocalAI via purego/cgo, any C host) sees a clean, ABI-stable
# surface — `nm -D libvllm.so` shows the C ABI and nothing else. The static
# `libvllm.a` is the existing `vllm` archive; both are installed.
include(GNUInstallDirs)

# A generated stub gives the SHARED target its (empty) own source and forces the
# CXX linker driver; the real content is the whole `vllm` archive, force-linked
# below via vllm's INTERFACE --whole-archive option.
set(_vllm_shared_stub "${CMAKE_CURRENT_BINARY_DIR}/vllm_shared_stub.cpp")
file(WRITE "${_vllm_shared_stub}"
  "// Generated by CMake: linker stub for the packaged shared libvllm.\n"
  "// The C ABI + engine come from the force-linked static `vllm` archive.\n")

add_library(vllm_shared SHARED "${_vllm_shared_stub}")
add_library(vllm::shared ALIAS vllm_shared)
set_target_properties(vllm_shared PROPERTIES
  OUTPUT_NAME vllm
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
# PUBLIC include so a consumer linking vllm_shared gets include/vllm.h ONLY (no
# internal C++ headers).
target_include_directories(vllm_shared PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Force-link the whole `vllm` archive (the C ABI + engine + the CPU-backend
# static registrar) and inherit its PUBLIC deps (CUDA::cudart, Threads, ...).
target_link_libraries(vllm_shared PRIVATE vllm)
# Export only the C ABI: `vllm_*` stays global, everything else is localized.
# UNLIKE the force-link guard above, `UNIX AND NOT APPLE` is CORRECT here: a
# linker version script is a GNU-ld/ELF feature with no ld64 spelling (ld64 uses
# -exported_symbols_list). The macOS dylib therefore exports its default symbol
# set; tests/CMakeLists.txt gates the matching `capi_shared_exports_only_abi`
# check on the same condition for the same reason.
if(UNIX AND NOT APPLE)
  target_link_options(vllm_shared PRIVATE
    "LINKER:--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cmake/vllm_export.map")
endif()

# Sanitizer-only test linkage. Every normal test force-links the complete static
# `vllm` archive so its registrars are retained. With ASan+UBSan that makes each
# executable hundreds of MiB and duplicates the same instrumented engine across
# the whole suite; GitHub's hosted runner ran out of disk while linking. Keep one
# unversioned internal shared image of the fully instrumented archive instead.
# This is deliberately separate from packaged `vllm_shared`: that library's
# version script must continue exporting only the stable C ABI, while the tests
# need the C++ internals. Production/default builds never create this target.
if(NOT VLLM_CPP_SANITIZE STREQUAL "OFF")
  add_library(vllm_sanitize_test_shared SHARED "${_vllm_shared_stub}")
  add_library(vllm::sanitize_test_shared ALIAS vllm_sanitize_test_shared)
  target_link_libraries(vllm_sanitize_test_shared PRIVATE vllm)
  target_include_directories(vllm_sanitize_test_shared INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_INCLUDE_DIRECTORIES>")
  target_include_directories(vllm_sanitize_test_shared SYSTEM INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>")
  target_compile_definitions(vllm_sanitize_test_shared INTERFACE
    "$<TARGET_PROPERTY:vllm,INTERFACE_COMPILE_DEFINITIONS>")
endif()

# make install => a consumable libvllm.so + libvllm.a + vllm.h (llama.cpp-style).
install(TARGETS vllm vllm_shared
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/vllm.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(VLLM_CPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VLLM_CPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
