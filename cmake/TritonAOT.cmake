# cmake/TritonAOT.cmake — build-time Triton AOT toolchain (CUDA-only, gated).
#
# SANCTIONED CUDA-ONLY accelerator (user decision). A build-time step compiles
# @triton.jit kernels to cubins that are EMBEDDED (as C byte arrays) in generated
# C launchers and linked into libvllm. At runtime there is NO Triton/Python — the
# launcher uses the CUDA driver API (cuModuleLoadData + cuLaunchKernel). The hand
# C++/CUDA path is always kept as the fallback; a Triton realization is selected
# only behind both a compile gate (VLLM_CPP_TRITON) and a runtime toggle.
#
# Mechanism (proven on a dgx PoC):
#   python -m triton.tools.compile <kernel.py> --kernel-name <fn>
#       --out-name <base> --out-path <dir>/<base>
#       --signature "<sig>" --grid "gx,gy,gz" --num-warps N --num-stages S
#   emits, per specialization:
#     <base>.<hash>_<suffix>.c  launcher + EMBEDDED cubin (plain C; no Triton at
#                               runtime). The grid gx/gy/gz is baked in from
#                               --grid (a C expression over the signature scalars);
#                               num_warps/num_stages are baked into the cubin.
#     <base>.<hash>_<suffix>.h  declares void load_<sym>(void); void unload_<sym>(void);
#                               and CUresult <sym>(CUstream, <typed args...>) where
#                               <sym> is HASHED (<base>_<hash>_<suffix>).
#   python -m triton.tools.link <base>.*.h -o <dir>/<base>
#   emits STABLE-named dispatchers into <base>.{c,h}:
#     CUresult <base>(CUstream, <full args...>, int algo_id);
#     CUresult <base>_default(CUstream, <full args...>);   // algo_id = 0
#     void load_<base>(void); void unload_<base>(void);
#   so our C++ calls a STABLE symbol (<base>_default) and never touches the hash.
#   We ALWAYS run the linker (even for a single specialization): it yields the
#   same stable ABI for 1..N specs, so no hash-parsing is ever required.
#
# VLLM_CPP_TRITON=OFF (the default) => this file only DEFINES the option + cache
# vars + the add_triton_kernel() function and does nothing else. No Python is
# invoked, no sources are generated or added, no compile definitions change: a
# build without it is byte-identical to a tree that never included this file.

option(VLLM_CPP_TRITON
  "Compile Triton kernels AOT to cubins linked into libvllm (CUDA-only)" OFF)

# The Python interpreter that has Triton installed (used ONLY at configure time,
# ONLY when VLLM_CPP_TRITON=ON). Defaults to the project's oracle venv on dgx.
set(VLLM_CPP_TRITON_PYTHON "$ENV{HOME}/venvs/vllm-oracle/bin/python"
  CACHE FILEPATH "Python interpreter with Triton installed (for the AOT step)")

# Optional explicit Triton target "<backend>:<arch>:<warp-size>" (e.g. cuda:121:32
# for GB10 sm_121). Empty => Triton auto-detects the GPU present at configure
# time (the dgx case; the AOT step requires a visible GPU + ptxas regardless).
set(VLLM_CPP_TRITON_TARGET "" CACHE STRING
  "Triton AOT target '<backend>:<arch>:<warpsize>' (empty = autodetect active GPU)")

# add_triton_kernel(RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID
#                   [NUM_WARPS] [NUM_STAGES])
#
# Compiles KERNEL_NAME (a @triton.jit fn defined in KERNEL_PY) AOT and links a
# stable dispatcher. SIGNATURE is one Triton signature, or several separated by
# '|' to build multiple specializations of the SAME kernel (they share OUT_BASE
# so triton.tools.link groups them under one stable dispatcher that picks the
# right specialization at runtime). GRID is the "gx,gy,gz" launch-grid expression
# (may reference the signature's scalar args, e.g. "n_rows,1,1"). Defaults:
# NUM_WARPS=4, NUM_STAGES=3.
#
# On return, in the CALLER's scope:
#   ${RESULT_VAR}              -> the generated C sources (add to a target)
#   ${RESULT_VAR}_INCLUDE_DIR  -> the dir holding <OUT_BASE>.h (the stable header)
#   ${RESULT_VAR}_HEADER       -> "<OUT_BASE>.h"
#
# Runs at CONFIGURE time (execute_process): re-run `cmake` after editing the
# kernel or signature. Only ever called under VLLM_CPP_TRITON (guard at call site).
function(add_triton_kernel RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID)
  set(_num_warps 4)
  set(_num_stages 3)
  if(ARGC GREATER 6)
    set(_num_warps "${ARGV6}")
  endif()
  if(ARGC GREATER 7)
    set(_num_stages "${ARGV7}")
  endif()

  if(NOT EXISTS "${VLLM_CPP_TRITON_PYTHON}")
    message(FATAL_ERROR
      "VLLM_CPP_TRITON=ON but VLLM_CPP_TRITON_PYTHON does not exist:\n"
      "  ${VLLM_CPP_TRITON_PYTHON}\n"
      "Point -DVLLM_CPP_TRITON_PYTHON=<path> at a Python that has Triton.")
  endif()
  if(NOT EXISTS "${KERNEL_PY}")
    message(FATAL_ERROR "add_triton_kernel: kernel file not found: ${KERNEL_PY}")
  endif()

  set(_outdir "${CMAKE_BINARY_DIR}/triton_aot")
  file(MAKE_DIRECTORY "${_outdir}")

  # Wipe any stale artifacts for this base so a reconfigure regenerates cleanly
  # and the header glob below never picks up leftovers (per-spec OR the linked
  # <base>.{c,h}). Matches <base>.* which covers both.
  file(GLOB _stale "${_outdir}/${OUT_BASE}.*")
  if(_stale)
    file(REMOVE ${_stale})
  endif()

  set(_target_args)
  if(VLLM_CPP_TRITON_TARGET)
    list(APPEND _target_args --target "${VLLM_CPP_TRITON_TARGET}")
  endif()

  # Split SIGNATURE into one-or-more specializations on '|'.
  string(REPLACE "|" ";" _sigs "${SIGNATURE}")

  # 1) Compile each specialization -> <base>.<hash>_<suffix>.{c,h}
  foreach(_sig IN LISTS _sigs)
    string(STRIP "${_sig}" _sig)
    message(STATUS "Triton AOT: compile ${KERNEL_NAME} [${_sig}] grid(${GRID}) "
                   "warps=${_num_warps} stages=${_num_stages}")
    execute_process(
      COMMAND "${VLLM_CPP_TRITON_PYTHON}" -m triton.tools.compile
              "${KERNEL_PY}"
              --kernel-name "${KERNEL_NAME}"
              --out-name "${OUT_BASE}"
              --out-path "${_outdir}/${OUT_BASE}"
              --signature "${_sig}"
              --grid "${GRID}"
              --num-warps "${_num_warps}"
              --num-stages "${_num_stages}"
              ${_target_args}
      WORKING_DIRECTORY "${_outdir}"
      RESULT_VARIABLE _rc
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR
        "Triton AOT compile failed (${KERNEL_NAME} / '${_sig}'):\n${_out}\n${_err}")
    endif()
  endforeach()

  # 2) Link all per-spec headers -> stable <base>.{c,h}. The glob runs BEFORE the
  #    linker writes <base>.h (which lacks a middle '.' and would not match the
  #    <base>.*.h pattern anyway), so it only sees the per-spec headers.
  file(GLOB _spec_headers "${_outdir}/${OUT_BASE}.*.h")
  if(NOT _spec_headers)
    message(FATAL_ERROR "Triton AOT: no per-spec headers generated for ${OUT_BASE}")
  endif()
  execute_process(
    COMMAND "${VLLM_CPP_TRITON_PYTHON}" -m triton.tools.link
            ${_spec_headers} -o "${_outdir}/${OUT_BASE}"
    WORKING_DIRECTORY "${_outdir}"
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "Triton AOT link failed (${OUT_BASE}):\n${_out}\n${_err}")
  endif()

  # 3) Collect the C sources: every per-spec launcher + the linked dispatcher.
  file(GLOB _spec_sources "${_outdir}/${OUT_BASE}.*.c")
  set(_all_sources ${_spec_sources} "${_outdir}/${OUT_BASE}.c")

  # This is GENERATED codegen, not our code: compile with warnings suppressed
  # (-w). The launch stub deliberately has a missing-return on its empty-grid
  # branch; without -w that would trip a warning. The files end in .c, so CMake
  # compiles them as C automatically (enable_language(C) at the call site).
  set_source_files_properties(${_all_sources} PROPERTIES COMPILE_OPTIONS "-w")

  set(${RESULT_VAR} "${_all_sources}" PARENT_SCOPE)
  set(${RESULT_VAR}_INCLUDE_DIR "${_outdir}" PARENT_SCOPE)
  set(${RESULT_VAR}_HEADER "${OUT_BASE}.h" PARENT_SCOPE)
  message(STATUS
    "Triton AOT: ${OUT_BASE} -> stable symbol ${OUT_BASE}_default() ; sources: ${_all_sources}")
endfunction()
