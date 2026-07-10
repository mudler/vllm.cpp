if(NOT DEFINED OUTPUT OR OUTPUT STREQUAL "")
  message(FATAL_ERROR "DumpTritonAOTContract.cmake requires -DOUTPUT=<path>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/TritonAOTKernels.cmake")
vllm_triton_aot_declare_all()
vllm_triton_aot_expected_lines(_lines)

file(WRITE "${OUTPUT}" "")
foreach(_line IN LISTS _lines)
  file(APPEND "${OUTPUT}" "${_line}\n")
endforeach()
