add_library(vllm_test_main STATIC doctest_main.cpp)
target_include_directories(vllm_test_main SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  # Static-init registrars (e.g. the CPU backend) live in archive members no
  # test references directly; force-link the whole vllm archive so they run.
  if(UNIX AND NOT APPLE)
    target_link_options(${name} PRIVATE "LINKER:--whole-archive,$<TARGET_FILE:vllm>,--no-whole-archive")
  endif()
  vllm_cpp_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_version vllm/test_version.cpp)
vllm_cpp_add_test(test_cuda_smoke vt/test_cuda_smoke.cpp)
vllm_cpp_add_test(test_dtype vt/test_dtype.cpp)
vllm_cpp_add_test(test_backend vt/test_backend.cpp)
