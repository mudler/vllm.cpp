add_executable(dump_container dump_container/main.cpp)
target_link_libraries(dump_container PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dump_container)

add_executable(tokenize tokenize/main.cpp)
target_link_libraries(tokenize PRIVATE vllm::vllm)
vllm_cpp_set_warnings(tokenize)

add_executable(dequant_nvfp4 dequant_nvfp4/main.cpp)
target_link_libraries(dequant_nvfp4 PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dequant_nvfp4)

# vllm-bench (M2.1): the throughput/latency benchmark harness — the gate-#1
# measurement tool. Links the full C++ engine (vllm::vllm) so it can build both a
# synthetic CPU engine (no --model) and a real loaded engine (--model on dgx).
add_executable(vllm-bench bench/main.cpp)
target_link_libraries(vllm-bench PRIVATE vllm::vllm)
target_include_directories(vllm-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
vllm_cpp_set_warnings(vllm-bench)

# vllm-cli: a llama.cpp-style client driven ENTIRELY through the C ABI. It links
# the packaged SHARED libvllm (vllm_shared) using only the public include/vllm.h,
# proving library-first consumption (M3.5 Task 3).
add_executable(vllm-cli cli/main.cpp)
target_link_libraries(vllm-cli PRIVATE vllm::shared)
vllm_cpp_set_warnings(vllm-cli)

# The OpenAI HTTP server example is gated on the vendored cpp-httplib transport.
if(VLLM_CPP_SERVER)
  add_executable(server server/main.cpp)
  target_link_libraries(server PRIVATE vllm::vllm)
  vllm_cpp_set_warnings(server)
endif()
