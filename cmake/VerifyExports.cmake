# Export-set verification for the packaged shared libvllm.so (M3.5 Task 3).
#
# Runs `nm -D --defined-only <lib>` and fails if ANY defined dynamic symbol is
# not part of the stable `vllm_*` C ABI — i.e. proves no C++ internals leak out
# of the shared library. A handful of linker-synthesized symbols (_init/_fini,
# section bounds, __gmon_start__, ...) are always emitted global regardless of
# the version script; those are allow-listed. Invoked as a ctest:
#   cmake -DVLLM_SHARED_LIB=<path> -P cmake/VerifyExports.cmake
if(NOT DEFINED VLLM_SHARED_LIB)
  message(FATAL_ERROR "VerifyExports: VLLM_SHARED_LIB not set")
endif()

find_program(NM_EXE NAMES nm)
if(NOT NM_EXE)
  message(WARNING "VerifyExports: nm not found; skipping export check")
  return()
endif()

execute_process(
  COMMAND ${NM_EXE} -D --defined-only ${VLLM_SHARED_LIB}
  OUTPUT_VARIABLE nm_out
  RESULT_VARIABLE nm_rc)
if(NOT nm_rc EQUAL 0)
  message(FATAL_ERROR "VerifyExports: nm failed on ${VLLM_SHARED_LIB}")
endif()

# Linker-synthesized globals that appear regardless of the version script.
set(allow
  "_init" "_fini" "_edata" "_end" "__bss_start" "__gmon_start__"
  "_IO_stdin_used" "__data_start" "data_start" "__dso_handle"
  "__TMC_END__" "_ITM_deregisterTMCloneTable" "_ITM_registerTMCloneTable"
  "__cxa_finalize" "_start")

string(REPLACE "\n" ";" lines "${nm_out}")
set(leaked "")
set(abi_exports "")
foreach(line IN LISTS lines)
  if(line STREQUAL "")
    continue()
  endif()
  # nm -D line: "<hexaddr> <type> <name>"; capture type + name.
  if(NOT line MATCHES "^[0-9a-fA-F]+[ \t]+([A-Za-z])[ \t]+(.+)$")
    continue()
  endif()
  set(type "${CMAKE_MATCH_1}")
  set(name "${CMAKE_MATCH_2}")
  # Absolute symbols (type 'A') are version-script nodes (e.g. VLLM_ABI_1), not
  # real code/data exports — skip them.
  if(type STREQUAL "A")
    continue()
  endif()
  if(name MATCHES "^vllm_")
    list(APPEND abi_exports "${name}")
  elseif(name IN_LIST allow)
    # benign linker symbol
  else()
    list(APPEND leaked "${type} ${name}")
  endif()
endforeach()

list(LENGTH abi_exports n_abi)
if(n_abi EQUAL 0)
  message(FATAL_ERROR
    "VerifyExports: no vllm_* symbols exported from ${VLLM_SHARED_LIB} — the "
    "version script or force-link is wrong")
endif()

if(leaked)
  string(REPLACE ";" "\n  " leaked_str "${leaked}")
  message(FATAL_ERROR
    "VerifyExports: ${VLLM_SHARED_LIB} leaks non-ABI dynamic symbols:\n  "
    "${leaked_str}")
endif()

message(STATUS
  "VerifyExports: OK — ${n_abi} vllm_* C ABI symbols exported, no C++ internals")
