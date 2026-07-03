add_executable(dump_container dump_container/main.cpp)
target_link_libraries(dump_container PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dump_container)

add_executable(tokenize tokenize/main.cpp)
target_link_libraries(tokenize PRIVATE vllm::vllm)
vllm_cpp_set_warnings(tokenize)

add_executable(dequant_nvfp4 dequant_nvfp4/main.cpp)
target_link_libraries(dequant_nvfp4 PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dequant_nvfp4)
