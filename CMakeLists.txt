cmake_minimum_required(VERSION 3.24)
project(vllm_cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(VLLM_CPP_BUILD_TESTS "Build tests" ON)
option(VLLM_CPP_CUDA "Build the CUDA backend (auto-detected if unset)" OFF)
set(VLLM_CPP_CUDA_ARCHITECTURES "121" CACHE STRING "Target CUDA architectures (GB10=121)")

include(CheckLanguage)
if(NOT VLLM_CPP_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolchain found; enabling VLLM_CPP_CUDA")
    set(VLLM_CPP_CUDA ON)
  endif()
endif()
if(VLLM_CPP_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES ${VLLM_CPP_CUDA_ARCHITECTURES})
endif()

include(cmake/CompilerWarnings.cmake)

configure_file(include/vllm/version.h.in include/vllm/version.h)

add_library(vllm STATIC
  src/vllm/version.cpp)
add_library(vllm::vllm ALIAS vllm)
target_include_directories(vllm
  PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(vllm SYSTEM PUBLIC third_party)
vllm_cpp_set_warnings(vllm)
if(VLLM_CPP_CUDA)
  target_compile_definitions(vllm PUBLIC VLLM_CPP_CUDA)
endif()

if(VLLM_CPP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
