cmake_minimum_required(VERSION 3.24)
project(vllm_cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VLLM_CPP_BUILD_TESTS "Build tests" ON)
option(VLLM_CPP_BUILD_EXAMPLES "Build examples" ON)
set(VLLM_CPP_CUDA "AUTO" CACHE STRING "Build the CUDA backend: ON, OFF, or AUTO (detect)")
set(VLLM_CPP_CUDA_ARCHITECTURES "121" CACHE STRING "Target CUDA architectures (GB10=121)")

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
endif()

include(cmake/CompilerWarnings.cmake)

configure_file(include/vllm/version.h.in include/vllm/version.h @ONLY)

add_library(vllm STATIC
  src/vllm/version.cpp
  src/vllm/sampling_params.cpp
  src/vllm/config/scheduler.cpp
  src/vllm/outputs.cpp
  src/vllm/transformers_utils/hf_config.cpp
  src/vllm/model_executor/model_loader/safetensors_reader.cpp
  src/vllm/model_executor/model_loader/gguf_reader.cpp
  src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  src/vllm/model_executor/models/qwen3_5_weights.cpp
  src/vllm/model_executor/models/qwen3_5.cpp
  src/vllm/model_executor/models/registry.cpp
  src/vllm/tokenizer/unicode_data.cpp
  src/vllm/tokenizer/pretokenizer.cpp
  src/vllm/tokenizer/bpe.cpp
  src/vllm/tokenizer/tokenizer.cpp
  src/vllm/v1/request.cpp
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
  src/vllm/v1/engine/types.cpp
  src/vllm/v1/engine/detokenizer.cpp
  src/vt/dtype.cpp
  src/vt/backend.cpp
  src/vt/tensor.cpp
  src/vt/arena.cpp
  src/vt/ops.cpp
  src/vt/cpu/cpu_backend.cpp
  src/vt/cpu/cpu_ops.cpp)
add_library(vllm::vllm ALIAS vllm)
# Static-init registrars (e.g. the CPU backend) live in archive members no
# consumer references directly; force-link the whole vllm archive so every
# consumer (tests, C API shared lib, benchmarks) gets backend registration.
if(UNIX AND NOT APPLE)
  target_link_options(vllm INTERFACE "LINKER:--whole-archive,$<TARGET_FILE:vllm>,--no-whole-archive")
endif()
target_include_directories(vllm
  PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(vllm SYSTEM PUBLIC third_party)
vllm_cpp_set_warnings(vllm)
if(VLLM_CPP_CUDA)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_CUDA)
  target_sources(vllm PRIVATE
    src/vt/cuda/cuda_backend.cu
    src/vt/cuda/cuda_matmul.cu
    src/vt/cuda/cuda_ops.cu
    src/vt/cuda/cuda_gdn.cu
    src/vt/cuda/cuda_moe.cu)
  find_package(CUDAToolkit REQUIRED)
  # cublasLt is linked now so the Task 4 matmul lands without a build change.
  target_link_libraries(vllm PUBLIC CUDA::cudart CUDA::cublasLt)
endif()

if(VLLM_CPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VLLM_CPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
