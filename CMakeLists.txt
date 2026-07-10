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
set(VLLM_CPP_CUDA_ARCHITECTURES "121a" CACHE STRING "Target CUDA architectures (GB10=121a; the 'a' arch-specific target is REQUIRED for the native block-scaled fp4xfp4 MMA — mma.sync kind::mxf4nvf4 is rejected on base sm_121)")

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
if(VLLM_CPP_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES ${VLLM_CPP_CUDA_ARCHITECTURES})
  # The native block-scaled fp4xfp4 MMA (cuda_matmul_nvfp4.cu) needs the
  # architecture-specific sm_12xa target. Advertise it to the code so the host
  # launcher enables the native path only when it was actually compiled in.
  if(VLLM_CPP_CUDA_ARCHITECTURES MATCHES "12[01]a")
    add_compile_definitions(VT_FP4_MMA_SM120A=1)
  endif()
  # CUTLASS (header-only, v4.4.2) for the sm120a NVFP4 block-scaled fp4xfp4 GEMM
  # drop-in (cuda_matmul_nvfp4_cutlass.cu). Vendored slice under third_party/cutlass
  # by default; override with -DVLLM_CPP_CUTLASS_DIR=<cutlass root> (must contain
  # include/ + tools/util/include/). The TU is enabled only when the dir exists and
  # the arch is sm_12xa (the block-scaled fp4 collective is Blackwell-only).
  set(VLLM_CPP_CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass"
      CACHE PATH "CUTLASS source root (include/ + tools/util/include/)")
  # Opt-in FetchContent fallback (mirrors vLLM's own CUTLASS_REVISION v4.4.2): if
  # no vendored/pointed cutlass tree is present, -DVLLM_CPP_CUTLASS_FETCH=ON pulls
  # v4.4.2 (header-only). Default OFF so a tight-disk box (dgx) builds against an
  # existing clone via -DVLLM_CPP_CUTLASS_DIR instead of a 200MB download.
  option(VLLM_CPP_CUTLASS_FETCH "FetchContent CUTLASS v4.4.2 if not found locally" OFF)
  if(VLLM_CPP_CUDA_ARCHITECTURES MATCHES "12[01]a"
     AND NOT EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h"
     AND VLLM_CPP_CUTLASS_FETCH)
    include(FetchContent)
    FetchContent_Declare(cutlass GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
      GIT_TAG v4.4.2 GIT_SHALLOW ON)
    FetchContent_MakeAvailable(cutlass)
    set(VLLM_CPP_CUTLASS_DIR "${cutlass_SOURCE_DIR}")
  endif()
  if(VLLM_CPP_CUDA_ARCHITECTURES MATCHES "12[01]a"
     AND EXISTS "${VLLM_CPP_CUTLASS_DIR}/include/cutlass/cutlass.h")
    set(VLLM_CPP_CUTLASS ON)
    message(STATUS "CUTLASS found at ${VLLM_CPP_CUTLASS_DIR}; enabling sm120a NVFP4 cutlass GEMM")
  else()
    set(VLLM_CPP_CUTLASS OFF)
    message(STATUS "CUTLASS not found / arch not sm_12xa; cutlass NVFP4 GEMM disabled "
                   "(set -DVLLM_CPP_CUTLASS_DIR=<cutlass> or -DVLLM_CPP_CUTLASS_FETCH=ON)")
  endif()
  # Marlin NVFP4 W4A16 grouped-MoE GEMM drop-in (the 35B expert kernel). Vendored
  # torch-free slice of vLLM's moe/marlin_moe_wna16 under src/vt/cuda/marlin/
  # (marlin-dropin-feasibility.md). NO external dependency — the kernel is
  # standalone CUDA + inline PTX. Enabled on sm_12xa (compiles to native sm_121
  # SASS). The two vendored TUs + the vt launcher are isolated with their own
  # compile options (see below); gated by VT_MARLIN_NVFP4.
  option(VLLM_CPP_MARLIN "Build the vendored Marlin NVFP4 W4A16 MoE GEMM (sm_12xa)" ON)
  if(VLLM_CPP_MARLIN AND VLLM_CPP_CUDA_ARCHITECTURES MATCHES "12[01]a"
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
  src/vllm/config/scheduler.cpp
  src/vllm/outputs.cpp
  src/vllm/transformers_utils/hf_config.cpp
  src/vllm/model_executor/model_loader/safetensors_reader.cpp
  src/vllm/model_executor/model_loader/gguf_reader.cpp
  src/vllm/model_executor/model_loader/gguf_dequant.cpp
  src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  src/vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.cpp
  src/vllm/model_executor/models/qwen3_5_weights.cpp
  src/vllm/model_executor/models/qwen3_5_dense_weights.cpp
  src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp
  src/vllm/model_executor/models/qwen3_5.cpp
  src/vllm/model_executor/models/registry.cpp
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
  src/vllm/v1/kv_cache_interface.cpp
  src/vllm/v1/core/kv_cache_utils.cpp
  src/vllm/v1/core/block_pool.cpp
  src/vllm/v1/core/single_type_kv_cache_manager.cpp
  src/vllm/v1/core/kv_cache_coordinator.cpp
  src/vllm/v1/core/kv_cache_manager.cpp
  src/vllm/v1/core/sched/request_queue.cpp
  src/vllm/v1/core/sched/output.cpp
  src/vllm/v1/core/sched/scheduler.cpp
  src/vllm/v1/core/sched/utils.cpp
  src/vllm/v1/worker/gpu/block_table.cpp
  src/vllm/v1/worker/gpu/input_batch.cpp
  src/vllm/v1/worker/gpu/prepare_inputs.cpp
  src/vllm/v1/worker/gpu/runner.cpp
  src/vllm/v1/attention/backend.cpp
  src/vllm/v1/attention/backends/gdn_attn.cpp
  src/vllm/v1/structured_output/backend_types.cpp
  src/vllm/v1/structured_output/backend_native.cpp
  src/vllm/v1/structured_output/json_schema_to_gbnf.cpp
  src/vllm/v1/structured_output/request.cpp
  src/vllm/v1/structured_output/manager.cpp
  src/vllm/v1/engine/types.cpp
  src/vllm/v1/engine/detokenizer.cpp
  src/vllm/v1/engine/core.cpp
  src/vllm/v1/engine/core_proc.cpp
  src/vllm/v1/engine/core_client.cpp
  src/vllm/v1/engine/input_processor.cpp
  src/vllm/v1/engine/output_processor.cpp
  src/vllm/v1/engine/llm_engine.cpp
  src/vllm/v1/executor/executor.cpp
  src/vllm/entrypoints/chat_template.cpp
  src/vllm/entrypoints/model_loader.cpp
  src/capi/vllm_c.cpp
  src/vllm/entrypoints/openai/protocol.cpp
  src/vllm/entrypoints/openai/serving_utils.cpp
  src/vllm/entrypoints/openai/serving_completion.cpp
  src/vllm/entrypoints/openai/serving_chat.cpp
  src/vllm/entrypoints/openai/serving_models.cpp
  src/vllm/entrypoints/openai/tool_parsers/abstract.cpp
  src/vllm/entrypoints/openai/tool_parsers/hermes.cpp
  src/vllm/entrypoints/openai/tool_parsers/qwen3.cpp
  src/vt/dtype.cpp
  src/vt/backend.cpp
  src/vt/tensor.cpp
  src/vt/arena.cpp
  src/vt/ops.cpp
  src/vt/cpu/cpu_backend.cpp
  src/vt/cpu/cpu_ops.cpp
  src/vt/cpu/cpu_cache.cpp
  src/vt/cpu/cpu_paged_attn.cpp
  src/vt/cpu/cpu_sample.cpp)
add_library(vllm::vllm ALIAS vllm)
# Static-init registrars (e.g. the CPU backend) live in archive members no
# consumer references directly; force-link the whole vllm archive so every
# consumer (tests, C API shared lib, benchmarks) gets backend registration.
if(UNIX AND NOT APPLE)
  target_link_options(vllm INTERFACE "LINKER:--whole-archive,$<TARGET_FILE:vllm>,--no-whole-archive")
endif()
target_include_directories(vllm
  PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/include)
# The C ABI impl reaches its internal test-hook header as "capi/engine_handle.h"
# (an internal, non-installed header under src/); src is a PRIVATE search path.
target_include_directories(vllm PRIVATE src)
target_include_directories(vllm SYSTEM PUBLIC third_party)
vllm_cpp_set_warnings(vllm)
if(VLLM_CPP_CUDA)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_CUDA)
  target_sources(vllm PRIVATE
    src/vt/cuda/cuda_backend.cu
    src/vt/cuda/cuda_matmul.cu
    src/vt/cuda/cuda_matmul_nvfp4.cu
    src/vt/cuda/cuda_ops.cu
    src/vt/cuda/cuda_gdn.cu
    src/vt/cuda/cuda_moe.cu
    src/vt/cuda/cuda_glue.cu
    src/vt/cuda/cuda_cache.cu
    src/vt/cuda/cuda_paged_attn.cu
    src/vt/cuda/cuda_sample.cu)
  find_package(CUDAToolkit REQUIRED)
  # cublasLt is linked now so the Task 4 matmul lands without a build change.
  target_link_libraries(vllm PUBLIC CUDA::cudart CUDA::cublasLt)
  # The cutlass sm120a NVFP4 GEMM TU is isolated: its own heavy cutlass include
  # tree + relaxed-constexpr/extended-lambda, so it never touches the general
  # build. Gated by VT_CUTLASS_NVFP4 so the host op path no-ops if not compiled.
  if(VLLM_CPP_CUTLASS)
    # Both cutlass GEMM TUs share the same isolated heavy-include discipline.
    # NVFP4 W4A4 (VT_CUTLASS_NVFP4) + per-tensor W8A8 fp8 (VT_CUTLASS_FP8).
    target_sources(vllm PRIVATE
      src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu
      src/vt/cuda/cuda_matmul_fp8_cutlass.cu)
    target_compile_definitions(vllm PUBLIC VT_CUTLASS_NVFP4=1 VT_CUTLASS_FP8=1)
    # cutlass headers reach via -isystem so their internal deprecation warnings
    # are exempt from the global -Werror=all-warnings (the general build stays
    # strict). --expt-relaxed-constexpr/--expt-extended-lambda: cutlass needs them.
    set_source_files_properties(
      src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu
      src/vt/cuda/cuda_matmul_fp8_cutlass.cu PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-isystem;${VLLM_CPP_CUTLASS_DIR}/include;-isystem;${VLLM_CPP_CUTLASS_DIR}/tools/util/include>")
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
      src/vt/cuda/cuda_marlin_repack.cu)
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
    set(_CHUNKO_BF16_AOT_AVAILABLE ${VLLM_CPP_TRITON_REGEN})
    if(NOT _CHUNKO_BF16_AOT_AVAILABLE)
      _triton_aot_arch_dir(_TRITON_AOT_ARCH_DIR)
      if(EXISTS "${_TRITON_AOT_ARCH_DIR}/gdn_chunko_bf16_h48.c" AND
         EXISTS "${_TRITON_AOT_ARCH_DIR}/gdn_chunko_bf16_h48.h" AND
         EXISTS "${_TRITON_AOT_ARCH_DIR}/gdn_chunko_bf16_h32.c" AND
         EXISTS "${_TRITON_AOT_ARCH_DIR}/gdn_chunko_bf16_h32.h")
        set(_CHUNKO_BF16_AOT_AVAILABLE ON)
      endif()
    endif()
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

      if(_CHUNKO_BF16_AOT_AVAILABLE)
        set(_sig "${_CHUNKO_SIG_HEAD_BF16}, ${_H}, ${_CHUNKO_SIG_TAIL}")
        add_triton_kernel(VT_GDN_CHUNKO_BF16_${_H}_SRCS
          "${_CHUNKO_KPY}" chunk_fwd_kernel_o gdn_chunko_bf16_h${_H}
          "${_sig}" "${_CHUNKO_GX},NT,${_H}"
          ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}})
        target_sources(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS})
        target_include_directories(vllm PRIVATE ${VT_GDN_CHUNKO_BF16_${_H}_SRCS_INCLUDE_DIR})
      endif()
    endforeach()
    if(_CHUNKO_BF16_AOT_AVAILABLE)
      target_compile_definitions(vllm PUBLIC VLLM_CPP_TRITON_CHUNKO_BF16=1)
    else()
      message(STATUS
        "Triton AOT bf16 chunk_o artifacts not present for this arch; "
        "VT_GDN_OUT_BF16 will use the portable chunk_o fallback until regen")
    endif()

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

  # Vendored FlashAttention-2 sm_121a prefill. The exact FA-2 source vLLM builds
  # as _vllm_fa2_C (vllm-project/flash-attention @ 2c839c33) — its
  # flash_fwd_splitkv kernel is what vLLM's nsys trace shows for prefill on GB10.
  # Compiled torch-free against our CUTLASS: 3 stub headers replace the ATen/c10
  # surface (PhiloxCudaState POD + C10_CUDA_CHECK), and fa2_compat_prelude.h is
  # force-included for the std headers ATen pulled in transitively — the upstream
  # (auto-generated) FA sources stay byte-for-byte pristine. DEFAULT ON (like
  # Marlin/cutlass) — it only actually builds when CUTLASS is present
  # (VLLM_CPP_CUTLASS, sm_12xa); -DVLLM_CPP_FLASH_ATTN=OFF opts out. Runtime
  # toggle VT_FA2_PREFILL (default ON when compiled; =0 restores the WMMA
  # prefill for a same-binary A/B). Validated token-for-token + over the
  # throughput bar on GB10 2026-07-10 (see .agents/parity-ledger.md).
  option(VLLM_CPP_FLASH_ATTN "Build vendored FlashAttention-2 sm_121a prefill kernel (needs CUTLASS)" ON)
  if(VLLM_CPP_FLASH_ATTN AND VLLM_CPP_CUTLASS)
    set(_FA2_ROOT "${CMAKE_SOURCE_DIR}/src/vt/cuda/flash_attn")
    set(_FA2_KERNEL_SRCS
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu
      src/vt/cuda/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu)
    target_sources(vllm PRIVATE
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS})
    target_compile_definitions(vllm PUBLIC VLLM_CPP_FLASH_ATTN)
    set_source_files_properties(
      src/vt/cuda/cuda_flash_attn_fa2.cu
      ${_FA2_KERNEL_SRCS} PROPERTIES
      COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--expt-extended-lambda;-DFLASH_NAMESPACE=flash;-include;${_FA2_ROOT}/fa2_compat_prelude.h;-isystem;${_FA2_ROOT}/src;-isystem;${_FA2_ROOT}/stubs;-isystem;${VLLM_CPP_CUTLASS_DIR}/include>")
    message(STATUS "FlashAttention-2 sm_121a prefill: ENABLED (runtime toggle VT_FA2_PREFILL)")
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
if(UNIX AND NOT APPLE)
  target_link_options(vllm_shared PRIVATE
    "LINKER:--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cmake/vllm_export.map")
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
