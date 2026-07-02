function(vllm_cpp_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
endfunction()
