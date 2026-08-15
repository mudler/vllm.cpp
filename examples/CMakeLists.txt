add_executable(dump_container dump_container/main.cpp)
target_link_libraries(dump_container PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dump_container)

add_executable(tokenize tokenize/main.cpp)
target_link_libraries(tokenize PRIVATE vllm::vllm)
vllm_cpp_set_warnings(tokenize)

add_executable(dequant_nvfp4 dequant_nvfp4/main.cpp)
target_link_libraries(dequant_nvfp4 PRIVATE vllm::vllm)
vllm_cpp_set_warnings(dequant_nvfp4)

# deepseek-v4-gen: the single-Spark DeepSeek-V4-Flash keep-quant GGUF greedy
# generation driver (the W8-run vehicle — stateless full-recompute greedy loop).
add_executable(deepseek-v4-gen deepseek_v4_gen/main.cpp)
target_link_libraries(deepseek-v4-gen PRIVATE vllm::vllm)
vllm_cpp_set_warnings(deepseek-v4-gen)

# laguna-gen: the single-GB10 Laguna-S-2.1 keep-quant GGUF greedy generation
# driver (W5 RUNNABLE vehicle — 3-shard keep-quant load + stateless recompute).
add_executable(laguna-gen laguna_gen/main.cpp)
target_link_libraries(laguna-gen PRIVATE vllm::vllm)
vllm_cpp_set_warnings(laguna-gen)

# kimi-linear-gen: the Kimi-Linear-48B-A3B golden token battery, a THIN CLIENT
# of the public C ABI (ONE SURFACE / ROW 7 §20.3 B4): vllm_engine_load + the
# v13 vllm_complete_tokens over the pre-tokenized golden prompts. Links the
# packaged shared library like vllm-cli — no internal headers.
add_executable(kimi-linear-gen kimi_linear_gen/main.cpp)
target_link_libraries(kimi-linear-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(kimi-linear-gen)

# vllm-bench (M2.1): the throughput/latency benchmark harness — the gate-#1
# measurement tool. Links the full C++ engine (vllm::vllm) so it can build both a
# synthetic CPU engine (no --model) and a real loaded engine (--model on dgx).
add_executable(vllm-bench bench/main.cpp)
target_link_libraries(vllm-bench PRIVATE vllm::vllm)
target_include_directories(vllm-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
vllm_cpp_set_warnings(vllm-bench)

# vllm-cli: a llama.cpp-style client driven ENTIRELY through the C ABI. It links
# the packaged SHARED libvllm (vllm_shared) using only the public include/vllm.h,
# proving library-first consumption (M3.5 Task 3).
add_executable(vllm-cli cli/main.cpp)
target_link_libraries(vllm-cli PRIVATE vllm::shared)
vllm_cpp_set_warnings(vllm-cli)

# quant-gemm-bench (QUANT-GGUF-CIQ-GEMM G6 evidence tool, NOT a gate): op-level
# GFLOP/s for kMatmulBTQuant, portable vs Arm i8mm mmla tier.
add_executable(quant-gemm-bench quant_gemm_bench/main.cpp)
target_link_libraries(quant-gemm-bench PRIVATE vllm::vllm)
target_include_directories(quant-gemm-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_set_warnings(quant-gemm-bench)

add_executable(vllm-cpu-kernel-bench cpu_kernel_bench/main.cpp)
target_link_libraries(vllm-cpu-kernel-bench PRIVATE vllm::vllm)
target_include_directories(vllm-cpu-kernel-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_set_warnings(vllm-cpu-kernel-bench)
if(VLLM_CPP_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME test_cpu_kernel_bench_cli
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tests/scripts/test_cpu_kernel_bench.py
      $<TARGET_FILE:vllm-cpu-kernel-bench>)
endif()

# The OpenAI HTTP server example is gated on the vendored cpp-httplib transport.
if(VLLM_CPP_SERVER)
  # vllm-video-studio: a STANDALONE browser console for MiniMax-H3, deliberately
  # SEPARATE from the API server. examples/server is the OpenAI-compatible API
  # surface and a UI does not belong in it; this owns its own endpoints and
  # drives the public C ABI (vllm_video_*) like any FFI consumer would. It shares
  # the server's gate only because it uses the same vendored httplib transport.
  add_executable(video-studio video_studio/main.cpp)
  target_link_libraries(video-studio PRIVATE vllm::shared)
  target_include_directories(video-studio SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
  set_target_properties(video-studio PROPERTIES OUTPUT_NAME vllm-video-studio)
  vllm_cpp_set_warnings(video-studio)

  add_executable(server server/main.cpp)
  # ARCH-ONE-SURFACE: server/main.cpp is a THIN CLIENT of the public C ABI
  # (vllm_server_main, v17) -- it includes vllm.h and nothing else.  The release
  # executable links the static core below, but reaches it only through that same
  # public surface.
  # Release bundles carry one static-core server.  The executable remains a
  # thin C-ABI client; only the link form changes so an extracted archive does
  # not depend on a separately installed libvllm.
  target_link_libraries(server PRIVATE vllm::vllm)
  if(VLLM_CPP_LITERAL_STATIC)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR VLLM_CPP_CUDA OR
       VLLM_CPP_METAL OR VLLM_CPP_MLX OR VLLM_CPP_VULKAN OR VLLM_CPP_HIP)
      message(FATAL_ERROR
        "VLLM_CPP_LITERAL_STATIC is restricted to the experimental Linux CPU-only lane")
    endif()
    target_link_options(server PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
  set_target_properties(server PROPERTIES OUTPUT_NAME vllm-server)
  if(VLLM_CPP_MLX)
    set_target_properties(server PROPERTIES
      BUILD_RPATH "${MLX_ROOT}/lib"
      INSTALL_RPATH "@loader_path/../lib")
    find_file(_vllm_mlx_metallib mlx.metallib
      PATHS "${MLX_ROOT}/lib" "${MLX_ROOT}" NO_DEFAULT_PATH)
    if(NOT _vllm_mlx_metallib)
      message(FATAL_ERROR
        "MLX release packaging requires mlx.metallib under MLX_ROOT")
    endif()
    install(FILES "${MLX_LIBRARY}" "${_vllm_mlx_metallib}"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      COMPONENT vllm-server)
  endif()
  vllm_cpp_set_warnings(server)
  install(TARGETS server
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT vllm-server)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(_vllm_server_backends)
  if(VLLM_CPP_CUDA)
    list(APPEND _vllm_server_backends cuda)
  endif()
  if(VLLM_CPP_METAL)
    if(VLLM_CPP_MLX)
      list(APPEND _vllm_server_backends metal-mlx)
    else()
      list(APPEND _vllm_server_backends metal)
    endif()
  endif()
  if(VLLM_CPP_VULKAN)
    list(APPEND _vllm_server_backends vulkan)
  endif()
  if(VLLM_CPP_HIP)
    list(APPEND _vllm_server_backends rocm)
  endif()
  if(NOT _vllm_server_backends)
    list(APPEND _vllm_server_backends cpu)
  endif()
  string(JOIN "-" _vllm_server_backend ${_vllm_server_backends})
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _vllm_server_os)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _vllm_server_arch)
  set(_vllm_server_artifact_backend "${_vllm_server_backend}")
  if(VLLM_CPP_CUDA)
    set(_vllm_server_artifact_backend cuda)
  elseif(VLLM_CPP_LITERAL_STATIC)
    set(_vllm_server_artifact_backend cpu-static)
  endif()
  if(_vllm_server_os STREQUAL "linux")
    if(VLLM_CPP_LITERAL_STATIC)
      set(_vllm_server_abi musl)
    else()
      set(_vllm_server_abi glibc)
    endif()
    set(_vllm_server_artifact_id
      "linux-${_vllm_server_arch}-${_vllm_server_abi}-${_vllm_server_artifact_backend}")
  elseif(_vllm_server_os STREQUAL "darwin")
    set(_vllm_server_artifact_id
      "macos-${_vllm_server_arch}-${_vllm_server_artifact_backend}")
  else()
    set(_vllm_server_artifact_id
      "${_vllm_server_os}-${_vllm_server_arch}-${_vllm_server_artifact_backend}")
  endif()
  set(_vllm_server_release_dir "${CMAKE_BINARY_DIR}/release")
  set(_vllm_server_stage_dir "${_vllm_server_release_dir}/stage")
  set(_vllm_server_archive
    "${_vllm_server_release_dir}/vllm.cpp-${PROJECT_VERSION}-${_vllm_server_artifact_id}.tar.gz")

  add_custom_target(vllm-server-stage
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/package-server.py
      --build-dir ${CMAKE_BINARY_DIR}
      --stage-dir ${_vllm_server_stage_dir}
      --config $<CONFIG>
    DEPENDS server ${CMAKE_SOURCE_DIR}/scripts/package-server.py
    COMMENT "Installing the vllm-server component into an empty staging prefix"
    VERBATIM)

  add_custom_target(vllm-server-archive
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/package-server.py
      --build-dir ${CMAKE_BINARY_DIR}
      --stage-dir ${_vllm_server_stage_dir}
      --archive ${_vllm_server_archive}
      --archive-format tar.gz
      --config $<CONFIG>
    DEPENDS server ${CMAKE_SOURCE_DIR}/scripts/package-server.py
    BYPRODUCTS ${_vllm_server_archive}
    COMMENT "Creating deterministic installed vllm-server archive"
    VERBATIM)

  if(VLLM_CPP_BUILD_TESTS)
    add_test(NAME test_server_help COMMAND $<TARGET_FILE:server> --help)
    set_tests_properties(test_server_help PROPERTIES
      PASS_REGULAR_EXPRESSION "--max-num-seqs N.*--max-num-batched-tokens N.*--cuda-profile-graph-replays N.*--cuda-profile-graph-batch N.*--benchmark-shutdown-fifo F.*--enable-force-include-usage.*--\\[no-\\]enable-prefix-caching")

    # --video-family (row MODEL-DIFFUSION-LTX25 / #435). Black-box, because the
    # flag table lives in an anonymous namespace inside server_main.cpp and there
    # is no unit seam onto it: the binary IS the surface under test. Both cases
    # exit during ParseArgs, so neither loads a model, a checkpoint or a byte of
    # weights, and both run in milliseconds.
    #
    # 1. AN UNREGISTERED NAME IS REFUSED, and the refusal NAMES what is
    #    registered. Before the flag existed this printed "unknown argument
    #    '--video-family'" instead, which is the RED.
    add_test(NAME test_server_video_family_unregistered
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-family not-a-registered-family)
    set_tests_properties(test_server_video_family_unregistered PROPERTIES
      PASS_REGULAR_EXPRESSION "--video-family 'not-a-registered-family' is not a registered video family. Registered families:.*ltx-2\\.5.*minimax-h3")

    # 2. A REGISTERED name is ACCEPTED — and specifically, the check consults the
    #    REGISTRY rather than a list written out here, so the family LTX-2.5
    #    registers from its own file passes. Proven by getting PAST the family
    #    check to the next validation in ParseArgs (the tool-parser dialect),
    #    with the family refusal absent from the output. A hardcoded list that
    #    omitted ltx-2.5 REDs on the FAIL expression, and a check accidentally
    #    dropped altogether REDs on it too (the unknown-argument spelling).
    add_test(NAME test_server_video_family_registered
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-family ltx-2.5
              --tool-call-parser definitely-not-a-registered-parser)
    set_tests_properties(test_server_video_family_registered PROPERTIES
      PASS_REGULAR_EXPRESSION "unknown tool-call parser \"definitely-not-a-registered-parser\""
      FAIL_REGULAR_EXPRESSION "not a registered video family;unknown argument")

    # 3. --video-extra takes KEY=VALUE, and a bare KEY is REFUSED rather than
    #    read as an empty value. An extra that silently became "" is how a
    #    mistyped knob renders the default and looks like the feature not
    #    working — the same failure the families' own unknown-extra refusals
    #    exist to stop, and it would be defeated one layer earlier here.
    add_test(NAME test_server_video_extra_needs_a_value
      COMMAND $<TARGET_FILE:server> --model /nonexistent-model-dir
              --video-extra dit_config_path)
    set_tests_properties(test_server_video_extra_needs_a_value PROPERTIES
      PASS_REGULAR_EXPRESSION "--video-extra takes KEY=VALUE, got 'dit_config_path'")
  endif()
endif()

# minimax-h3-mux: muxes MiniMax-H3 PPM frames + WAV into an MP4 by invoking
# ffmpeg — a THIN client of the public C ABI (include/vllm.h + vllm::shared
# ONLY, like vllm-cli), per the ONE SURFACE directive. This is the RATIFIED
# home of that process spawn (developer decision 2026-08-03): the library
# composes the argv (vllm_video_mux_argv), examples/ runs it.
add_executable(minimax-h3-mux minimax_h3_mux/main.cpp)
target_link_libraries(minimax-h3-mux PRIVATE vllm::shared)
vllm_cpp_set_warnings(minimax-h3-mux)

# minimax-h3-gen: MiniMax-H3 video generation end to end — a THIN client of
# the public C ABI (ARCH-ONE-SURFACE ROW 2). The assembly pipeline it used to
# own privately lives in the library behind vllm_video_engine_load +
# vllm_video_generate; the fold gate holds its frames + WAV byte-identical to
# the pre-fold binary. Lives here because it invokes ffmpeg (the ratified
# boundary).
add_executable(minimax-h3-gen minimax_h3_gen/main.cpp)
target_link_libraries(minimax-h3-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(minimax-h3-gen)

# ltx2-gen: an LTX-2.5 render end to end, the SECOND family behind the generic
# video seam and a THIN client of the same C ABI. It exists rather than a flag on
# minimax-h3-gen because LTX-2.5 cannot load without the v18 family-specific
# extras (the audio-stream prompt embeds, the DiT config the shipped FP8
# checkpoint does not carry, the latent upsampler) and that example drives
# neither `family` nor the extras arrays. Row MODEL-DIFFUSION-LTX25 / #435.
add_executable(ltx2-gen ltx2_gen/main.cpp)
target_link_libraries(ltx2-gen PRIVATE vllm::shared)
vllm_cpp_set_warnings(ltx2-gen)

# vulkan-gemm-ab: the VK-C tactic A/B — cooperative matrix vs the portable scalar
# GEMM, SAME BINARY with VT_VULKAN_COOPMAT as the only variable. Built only in a
# Vulkan build, because it links the Vulkan context directly to report which
# tactic each arm actually ran (a silent fallback would post a plausible pair of
# timings that mean nothing).
if(VLLM_CPP_VULKAN)
  add_executable(vulkan-gemm-ab ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_gemm_ab.cpp)
  target_link_libraries(vulkan-gemm-ab PRIVATE vllm::vllm)
  target_include_directories(vulkan-gemm-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-gemm-ab)

  # vulkan-dispatch-floor: the VK-A2 sizing measurement. Sweeps ONE op across a
  # 65,536x range of element counts to separate per-dispatch overhead from real
  # kernel cost -- two causes that produce an identical wall clock but imply
  # opposite fixes. Same Vulkan-only condition, same reason.
  add_executable(vulkan-dispatch-floor
                 ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_dispatch_floor.cpp)
  target_link_libraries(vulkan-dispatch-floor PRIVATE vllm::vllm)
  target_include_directories(vulkan-dispatch-floor PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-dispatch-floor)

  # vulkan-gemv-ab: the BACKEND-VULKAN-GEMVROWS variant sweep. Runs the decode
  # GEMV over the (k, n) shapes Qwen3.6-27B actually dispatches, so a variant can
  # be accepted or discarded without spending a 50 GB e2e run on it. Same
  # Vulkan-only condition and the same reason as the two above: it prints the
  # pipeline cache key, so an arm proves which specialization it ran.
  add_executable(vulkan-gemv-ab ${CMAKE_SOURCE_DIR}/benchmarks/vulkan_gemv_ab.cpp)
  target_link_libraries(vulkan-gemv-ab PRIVATE vllm::vllm)
  target_include_directories(vulkan-gemv-ab PRIVATE ${CMAKE_SOURCE_DIR}/src)
  vllm_cpp_set_warnings(vulkan-gemv-ab)
endif()

# parakeet-transcribe: the ASR example — a THIN client of the public C ABI
# (include/vllm.h + vllm::shared ONLY, like vllm-cli), per the ONE SURFACE
# directive. The pipeline it used to own privately lives in the library behind
# vllm_engine_load + vllm_transcribe (ARCH-ONE-SURFACE ROW 1); the fold gate
# holds its transcript byte-identical to the pre-fold binary.
add_executable(parakeet-transcribe parakeet_transcribe/main.cpp)
target_link_libraries(parakeet-transcribe PRIVATE vllm::shared)
vllm_cpp_set_warnings(parakeet-transcribe)
