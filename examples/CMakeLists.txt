add_executable(dump_container dump_container/main.cpp)
target_link_libraries(dump_container PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dump_container)
