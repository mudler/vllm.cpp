function(vllm_cpp_set_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
    # OBJCXX (.mm — the Metal backend) is a SEPARATE COMPILE_LANGUAGE from CXX,
    # so the CXX genex above does not reach it. Without this line the Metal TUs
    # would be the only unwarned code in the tree (BACKEND-METAL-MLX W0).
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings>)
endfunction()
