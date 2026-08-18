if(NOT DEFINED TEST_EXE OR NOT DEFINED PROBE_ROOT)
  message(FATAL_ERROR "TEST_EXE and PROBE_ROOT are required")
endif()

set(_required greedy_ids.npy our_ids.npy neartie_gap_mnats.npy)
set(_failures "")
foreach(_missing IN LISTS _required)
  set(_case_dir "${PROBE_ROOT}/${_missing}")
  file(REMOVE_RECURSE "${_case_dir}")
  file(MAKE_DIRECTORY "${_case_dir}")
  foreach(_present IN LISTS _required)
    if(NOT _present STREQUAL _missing)
      file(WRITE "${_case_dir}/${_present}" "present")
    endif()
  endforeach()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
            "VT_QWEN35_GATE_PREREQ_PROBE_DIR=${_case_dir}"
            "${TEST_EXE}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  message(STATUS "missing ${_missing}: child exit ${_rc}")
  if(NOT _rc EQUAL 77)
    string(APPEND _failures
      "\nmissing ${_missing}: expected exit 77, got ${_rc}"
      "\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
  endif()
endforeach()

file(REMOVE_RECURSE "${PROBE_ROOT}")
if(_failures)
  message(FATAL_ERROR "Qwen3.5 gate prerequisite failures:${_failures}")
endif()
