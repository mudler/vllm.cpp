# cmake/TritonAOT.cmake — Triton AOT kernels, VENDORED per-arch (CUDA-only, gated).
#
# SANCTIONED CUDA-ONLY accelerator (user decision). @triton.jit kernels are
# compiled to cubins that are EMBEDDED (as C byte arrays) in generated C
# launchers and linked into libvllm. At runtime there is NO Triton/Python — the
# launcher uses the CUDA driver API (cuModuleLoadData + cuLaunchKernel). The hand
# C++/CUDA path is always kept as the fallback; a Triton realization is selected
# only behind both a compile gate (VLLM_CPP_TRITON) and a runtime toggle.
#
# ── Two modes ────────────────────────────────────────────────────────────────
#
# BUILDER (VLLM_CPP_TRITON=ON, the default flow): the generated .c/.h artifacts
# are VENDORED in-repo under src/vt/cuda/triton_aot_vendored/<arch>/ (like the
# Marlin / FlashAttention-2 vendored kernels). The build consumes them directly:
# NO Python, NO Triton, NO GPU needed at build time — only a C compiler. If the
# target arch has no vendored artifacts, configuration FAILS with instructions.
#
# MAINTAINER (VLLM_CPP_TRITON_REGEN=ON): runs the original Python toolchain
# (triton.tools.compile / triton.tools.link, needs Python+Triton+ptxas and a
# visible GPU unless VLLM_CPP_TRITON_TARGET is set) AND refreshes the vendored
# tree + its MANIFEST, so `scripts/regen-triton-aot.sh` + `git diff` is the
# whole regen workflow. The build then compiles the refreshed vendored files
# (exactly what a builder will compile).
#
# Mechanism of the generated code (proven on a dgx PoC):
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
# Staleness: the vendored MANIFEST records the sha256 of every
# triton_kernels/*.py plus the exact generation parameters per kernel base.
# Consuming a vendored tree whose hashes/parameters no longer match the working
# tree WARNS loudly (the kernels changed but the vendored cubins did not) but
# does not fail — regen is a maintainer task on a GPU box.
#
# VLLM_CPP_TRITON=OFF (the default) => this file only DEFINES the options + cache
# vars + the functions and does nothing else. No Python is invoked, no sources
# are added, no compile definitions change: a build without it is byte-identical
# to a tree that never included this file.

option(VLLM_CPP_TRITON
  "Link the vendored Triton AOT kernels (cubins embedded in C) into libvllm (CUDA-only; no Python needed)" OFF)

option(VLLM_CPP_TRITON_REGEN
  "MAINTAINER: regenerate the vendored Triton AOT artifacts with Python+Triton (needs a visible GPU); see scripts/regen-triton-aot.sh" OFF)

# Where the vendored artifacts live, one subdir per arch (sm_121a today;
# sm_90/sm_80/gfx* slots in later). Each arch dir holds the generated .c/.h for
# every kernel base plus a MANIFEST (generator versions + source hashes +
# generation parameters).
set(VLLM_CPP_TRITON_VENDORED_DIR "${CMAKE_SOURCE_DIR}/src/vt/cuda/triton_aot_vendored"
  CACHE PATH "Root of the vendored Triton AOT artifacts (per-arch subdirs)")

# Arch subdir name. Empty => derived from VLLM_CPP_CUDA_ARCHITECTURES as
# "sm_<arch>" (e.g. 121a -> sm_121a). Set explicitly for cross-arch layouts the
# derivation cannot guess (a multi-arch fat build, or a future gfx* backend).
set(VLLM_CPP_TRITON_VENDORED_ARCH "" CACHE STRING
  "Vendored Triton AOT arch subdir (empty = sm_<VLLM_CPP_CUDA_ARCHITECTURES>)")

# The Python interpreter that has Triton installed. Used ONLY at configure time,
# ONLY when VLLM_CPP_TRITON_REGEN=ON (the maintainer path). Normal builds never
# touch it. Defaults to the project's oracle venv on dgx.
set(VLLM_CPP_TRITON_PYTHON "$ENV{HOME}/venvs/vllm-oracle/bin/python"
  CACHE FILEPATH "Python interpreter with Triton installed (REGEN only)")

# Optional explicit Triton target "<backend>:<arch>:<warp-size>" (e.g. cuda:121:32
# for GB10 sm_121). Empty => Triton auto-detects the GPU present at configure
# time (the dgx case; the REGEN step requires a visible GPU + ptxas regardless).
set(VLLM_CPP_TRITON_TARGET "" CACHE STRING
  "Triton AOT target '<backend>:<arch>:<warpsize>' (REGEN only; empty = autodetect active GPU)")

# Fresh manifest accumulator per configure run (REGEN mode appends to it; the
# property does not persist across cmake runs, this is just belt-and-braces).
set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_MANIFEST_BASES "")

# _triton_aot_arch_dir(OUTVAR) -> absolute path of the active vendored arch dir.
function(_triton_aot_arch_dir OUTVAR)
  if(VLLM_CPP_TRITON_VENDORED_ARCH)
    set(_a "${VLLM_CPP_TRITON_VENDORED_ARCH}")
  else()
    string(REPLACE ";" "_" _a "${VLLM_CPP_CUDA_ARCHITECTURES}")
    set(_a "sm_${_a}")
  endif()
  set(${OUTVAR} "${VLLM_CPP_TRITON_VENDORED_DIR}/${_a}" PARENT_SCOPE)
endfunction()

# add_triton_kernel(RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID
#                   [NUM_WARPS] [NUM_STAGES])
#
# Provides the AOT build of KERNEL_NAME (a @triton.jit fn defined in KERNEL_PY):
# from the vendored tree (default) or freshly regenerated (REGEN). SIGNATURE is
# one Triton signature, or several separated by '|' to build multiple
# specializations of the SAME kernel (they share OUT_BASE so triton.tools.link
# groups them under one stable dispatcher that picks the right specialization at
# runtime). GRID is the "gx,gy,gz" launch-grid expression (may reference the
# signature's scalar args, e.g. "n_rows,1,1"). Defaults: NUM_WARPS=4, NUM_STAGES=3.
#
# On return, in the CALLER's scope:
#   ${RESULT_VAR}              -> the generated C sources (add to a target)
#   ${RESULT_VAR}_INCLUDE_DIR  -> the dir holding <OUT_BASE>.h (the stable header)
#   ${RESULT_VAR}_HEADER       -> "<OUT_BASE>.h"
#
# Runs at CONFIGURE time. Only ever called under VLLM_CPP_TRITON (guard at call
# site); call triton_aot_finalize() once after the last add_triton_kernel().
function(add_triton_kernel RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID)
  set(_num_warps 4)
  set(_num_stages 3)
  if(ARGC GREATER 6)
    set(_num_warps "${ARGV6}")
  endif()
  if(ARGC GREATER 7)
    set(_num_stages "${ARGV7}")
  endif()

  if(NOT EXISTS "${KERNEL_PY}")
    message(FATAL_ERROR "add_triton_kernel: kernel file not found: ${KERNEL_PY}")
  endif()
  get_filename_component(_py_name "${KERNEL_PY}" NAME)
  _triton_aot_arch_dir(_adir)

  # The MANIFEST line that describes THIS generation request. Written on regen;
  # compared against the vendored MANIFEST when consuming (a mismatch means the
  # pins/signature changed without a regen -> stale cubins).
  set(_manifest_line
    "base ${OUT_BASE} py=${_py_name} kernel=${KERNEL_NAME} warps=${_num_warps} stages=${_num_stages} grid=${GRID} signature=${SIGNATURE}")

  if(NOT VLLM_CPP_TRITON_REGEN)
    # ── BUILDER path: consume the vendored artifacts. No Python. ─────────────
    if(NOT EXISTS "${_adir}/MANIFEST")
      message(FATAL_ERROR
        "VLLM_CPP_TRITON=ON but there are no vendored Triton AOT artifacts for "
        "this arch:\n  ${_adir}\n"
        "Options:\n"
        "  * build without -DVLLM_CPP_TRITON=ON (the portable C++ CUDA kernels are\n"
        "    the always-available fallback), or\n"
        "  * set -DVLLM_CPP_TRITON_VENDORED_ARCH=<dir> if the artifacts exist under\n"
        "    another name, or\n"
        "  * regenerate for this arch (MAINTAINER task; needs Python+Triton+GPU):\n"
        "    scripts/regen-triton-aot.sh  (configure with -DVLLM_CPP_TRITON_REGEN=ON)")
    endif()
    if(NOT EXISTS "${_adir}/${OUT_BASE}.c" OR NOT EXISTS "${_adir}/${OUT_BASE}.h")
      message(FATAL_ERROR
        "Vendored Triton AOT tree ${_adir} is missing the '${OUT_BASE}' kernel "
        "(expected ${OUT_BASE}.c/.h). The vendored tree predates this kernel — "
        "regenerate it: scripts/regen-triton-aot.sh")
    endif()
    file(GLOB _spec_sources "${_adir}/${OUT_BASE}.*.c")
    if(NOT _spec_sources)
      message(FATAL_ERROR
        "Vendored Triton AOT tree ${_adir} has ${OUT_BASE}.c but no per-spec "
        "${OUT_BASE}.<hash>.c launchers — the tree is corrupt; regenerate it: "
        "scripts/regen-triton-aot.sh")
    endif()
    set(_all_sources ${_spec_sources} "${_adir}/${OUT_BASE}.c")

    # Parameter staleness: find this base's line in the MANIFEST and compare.
    set(_found_line "")
    file(STRINGS "${_adir}/MANIFEST" _mlines)
    foreach(_line IN LISTS _mlines)
      string(FIND "${_line}" "base ${OUT_BASE} " _pos)
      if(_pos EQUAL 0)
        set(_found_line "${_line}")
      endif()
    endforeach()
    if(NOT _found_line STREQUAL _manifest_line)
      message(WARNING
        "STALE VENDORED TRITON AOT ARTIFACTS: the generation parameters for "
        "'${OUT_BASE}' differ from ${_adir}/MANIFEST.\n"
        "  expected: ${_manifest_line}\n"
        "  vendored: ${_found_line}\n"
        "The build proceeds with the VENDORED cubins (they are self-consistent), "
        "but they do not reflect the current CMake pins. Regenerate: "
        "scripts/regen-triton-aot.sh")
    endif()
    message(STATUS "Triton AOT: ${OUT_BASE} <- vendored ${_adir} (no Python)")
  else()
    # ── MAINTAINER path: regenerate with the Python toolchain, then refresh ──
    # the vendored tree and compile THOSE files (what a builder will compile).
    if(NOT EXISTS "${VLLM_CPP_TRITON_PYTHON}")
      message(FATAL_ERROR
        "VLLM_CPP_TRITON_REGEN=ON but VLLM_CPP_TRITON_PYTHON does not exist:\n"
        "  ${VLLM_CPP_TRITON_PYTHON}\n"
        "Point -DVLLM_CPP_TRITON_PYTHON=<path> at a Python that has Triton.")
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

    # 3) Refresh the vendored tree: drop this base's old files (hash-named files
    #    can change name across regens), copy the fresh ones with a provenance
    #    header prepended (deterministic — no timestamps; regen with the same
    #    toolchain must be byte-identical so `git diff` is the staleness signal).
    file(MAKE_DIRECTORY "${_adir}")
    file(GLOB _old_vendored "${_adir}/${OUT_BASE}.*")
    if(_old_vendored)
      file(REMOVE ${_old_vendored})
    endif()
    string(CONCAT _hdr
      "/*\n"
      " * vllm.cpp VENDORED Triton AOT artifact — GENERATED CODE. DO NOT EDIT.\n"
      " *\n"
      " * Generated by `python -m triton.tools.compile` / `triton.tools.link` from\n"
      " * triton_kernels/${_py_name} (kernel ${KERNEL_NAME}); the embedded byte\n"
      " * array (if any) is the compiled cubin for the arch named by this\n"
      " * directory. Generator versions, kernel-source hashes and the exact\n"
      " * generation parameters are recorded in the MANIFEST next to this file.\n"
      " *\n"
      " * Regenerate (MAINTAINER task; needs Python + Triton + a visible GPU):\n"
      " *   scripts/regen-triton-aot.sh\n"
      " *\n"
      " * The generating kernel source (triton_kernels/${_py_name}) is ported\n"
      " * verbatim from flash-linear-attention via vLLM (MIT; see its header);\n"
      " * this generated artifact is distributed under the repository's\n"
      " * Apache-2.0 license (see LICENSE and NOTICE).\n"
      " */\n")
    file(GLOB _gen_files "${_outdir}/${OUT_BASE}.*")
    set(_all_sources)
    foreach(_f IN LISTS _gen_files)
      get_filename_component(_fname "${_f}" NAME)
      file(READ "${_f}" _content)
      file(WRITE "${_adir}/${_fname}" "${_hdr}${_content}")
      if(_fname MATCHES "\\.c$")
        list(APPEND _all_sources "${_adir}/${_fname}")
      endif()
    endforeach()
    set_property(GLOBAL APPEND PROPERTY VLLM_TRITON_AOT_MANIFEST_BASES
      "${_manifest_line}")
    message(STATUS "Triton AOT: ${OUT_BASE} regenerated -> ${_adir}")
  endif()

  # This is GENERATED codegen, not our code: compile with warnings suppressed
  # (-w). The launch stub deliberately has a missing-return on its empty-grid
  # branch; without -w that would trip a warning. The files end in .c, so CMake
  # compiles them as C automatically (enable_language(C) at the call site).
  set_source_files_properties(${_all_sources} PROPERTIES COMPILE_OPTIONS "-w")

  set(${RESULT_VAR} "${_all_sources}" PARENT_SCOPE)
  set(${RESULT_VAR}_INCLUDE_DIR "${_adir}" PARENT_SCOPE)
  set(${RESULT_VAR}_HEADER "${OUT_BASE}.h" PARENT_SCOPE)
  message(STATUS
    "Triton AOT: ${OUT_BASE} -> stable symbol ${OUT_BASE}_default()")
endfunction()

# triton_aot_finalize()
#
# Call ONCE after the last add_triton_kernel() (only under VLLM_CPP_TRITON).
#  * REGEN: writes the vendored MANIFEST — generator versions (triton, its
#    bundled ptxas, python, the CUDA toolkit configured here), the sha256 of
#    every triton_kernels/*.py, the generation date, and one 'base' line per
#    kernel (accumulated by the add_triton_kernel calls).
#  * builder: verifies the triton_kernels/*.py hashes against the vendored
#    MANIFEST and WARNS loudly on any drift (kernels changed, cubins did not).
function(triton_aot_finalize)
  _triton_aot_arch_dir(_adir)

  if(VLLM_CPP_TRITON_REGEN)
    # Generator versions. Triton compiles the cubin with its OWN bundled ptxas
    # (triton/backends/nvidia/bin/ptxas), not the toolkit's — record that one.
    execute_process(
      COMMAND "${VLLM_CPP_TRITON_PYTHON}" -c
"import os, sys, subprocess, triton
print(triton.__version__)
print('%d.%d.%d' % sys.version_info[:3])
p = os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin', 'ptxas')
if os.path.exists(p):
    out = subprocess.run([p, '--version'], capture_output=True, text=True).stdout
    print(out.strip().splitlines()[-1].strip())
else:
    print('unknown (no bundled ptxas)')"
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _vers ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Triton AOT: version probe failed:\n${_err}")
    endif()
    string(REGEX REPLACE "\n+$" "" _vers "${_vers}")
    string(REPLACE "\n" ";" _vlist "${_vers}")
    list(GET _vlist 0 _triton_ver)
    list(GET _vlist 1 _python_ver)
    list(GET _vlist 2 _ptxas_ver)
    string(TIMESTAMP _date "%Y-%m-%d" UTC)
    if(VLLM_CPP_TRITON_TARGET)
      set(_tgt "${VLLM_CPP_TRITON_TARGET}")
    else()
      set(_tgt "autodetect (GPU visible at regen time)")
    endif()

    set(_m "")
    string(APPEND _m
      "# vllm.cpp vendored Triton AOT artifacts — MANIFEST. GENERATED, do not edit.\n"
      "# Regenerate: scripts/regen-triton-aot.sh (maintainer task; Python+Triton+GPU).\n"
      "# 'source' lines are the sha256 of the generating triton_kernels/*.py at regen\n"
      "# time (the configure-time staleness check compares against these); 'base'\n"
      "# lines are the exact generation parameters per kernel dispatcher.\n")
    get_filename_component(_arch_name "${_adir}" NAME)
    string(APPEND _m "arch ${_arch_name}\n")
    string(APPEND _m "generated ${_date}\n")
    string(APPEND _m "triton ${_triton_ver}\n")
    string(APPEND _m "python ${_python_ver}\n")
    string(APPEND _m "ptxas ${_ptxas_ver}\n")
    string(APPEND _m "cuda_toolkit ${CMAKE_CUDA_COMPILER_VERSION}\n")
    string(APPEND _m "triton_target ${_tgt}\n")
    file(GLOB _pys "${CMAKE_SOURCE_DIR}/triton_kernels/*.py")
    list(SORT _pys)
    foreach(_py IN LISTS _pys)
      get_filename_component(_pn "${_py}" NAME)
      file(SHA256 "${_py}" _h)
      string(APPEND _m "source ${_pn} sha256=${_h}\n")
    endforeach()
    get_property(_bases GLOBAL PROPERTY VLLM_TRITON_AOT_MANIFEST_BASES)
    list(SORT _bases)
    foreach(_b IN LISTS _bases)
      string(APPEND _m "${_b}\n")
    endforeach()
    file(WRITE "${_adir}/MANIFEST" "${_m}")
    message(STATUS "Triton AOT: MANIFEST written -> ${_adir}/MANIFEST "
                   "(triton ${_triton_ver}, ptxas '${_ptxas_ver}'). "
                   "Review + commit the vendored tree: git diff ${_adir}")
  else()
    # Source staleness for the vendored tree we just consumed.
    file(STRINGS "${_adir}/MANIFEST" _mlines)
    set(_manifest_pys "")
    set(_stale "")
    foreach(_line IN LISTS _mlines)
      if(_line MATCHES "^source ([^ ]+) sha256=([0-9a-f]+)$")
        set(_pn "${CMAKE_MATCH_1}")
        set(_want "${CMAKE_MATCH_2}")
        list(APPEND _manifest_pys "${_pn}")
        set(_py "${CMAKE_SOURCE_DIR}/triton_kernels/${_pn}")
        if(NOT EXISTS "${_py}")
          list(APPEND _stale "${_pn}: in MANIFEST but missing from triton_kernels/")
        else()
          file(SHA256 "${_py}" _have)
          if(NOT _have STREQUAL _want)
            list(APPEND _stale "${_pn}: sha256 differs (kernel edited since regen)")
          endif()
        endif()
      endif()
    endforeach()
    file(GLOB _pys "${CMAKE_SOURCE_DIR}/triton_kernels/*.py")
    foreach(_py IN LISTS _pys)
      get_filename_component(_pn "${_py}" NAME)
      if(NOT _pn IN_LIST _manifest_pys)
        list(APPEND _stale "${_pn}: present in triton_kernels/ but not in MANIFEST")
      endif()
    endforeach()
    if(_stale)
      list(JOIN _stale "\n  " _stale_txt)
      message(WARNING
        "STALE VENDORED TRITON AOT ARTIFACTS (${_adir}):\n"
        "  ${_stale_txt}\n"
        "The triton_kernels/*.py sources changed but the vendored cubins were NOT "
        "regenerated — the build proceeds with the OLD kernels. Regenerate "
        "(maintainer task; Python+Triton+GPU): scripts/regen-triton-aot.sh")
    else()
      message(STATUS "Triton AOT: vendored tree ${_adir} matches triton_kernels/ "
                     "(MANIFEST hashes OK)")
    endif()
  endif()
endfunction()
