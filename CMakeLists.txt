cmake_minimum_required(VERSION 3.24)
project(vllm_cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
  src/vllm/model_executor/model_loader/gguf_dequant.cpp
  src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  src/vllm/model_executor/models/qwen3_5_weights.cpp
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
