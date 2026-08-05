function(vllm_cpp_import_mlx mlx_root mlx_library)
  if(TARGET vllm_cpp::mlx)
    return()
  endif()

  add_library(vllm_cpp_mlx UNKNOWN IMPORTED)
  set_target_properties(vllm_cpp_mlx PROPERTIES
    IMPORTED_LOCATION "${mlx_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${mlx_root}/include"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${mlx_root}/include")
  add_library(vllm_cpp::mlx ALIAS vllm_cpp_mlx)
endfunction()
