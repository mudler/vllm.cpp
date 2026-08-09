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
  add_executable(server server/main.cpp)
  # ARCH-ONE-SURFACE: server/main.cpp is a THIN CLIENT of the public C ABI
  # (vllm_server_main, v17) -- it includes vllm.h and nothing else, so it links the
  # PACKAGED shared library exactly as an out-of-tree consumer would, not the
  # internal C++ target.
  target_link_libraries(server PRIVATE vllm::shared)
  vllm_cpp_set_warnings(server)
  if(VLLM_CPP_BUILD_TESTS)
    add_test(NAME test_server_help COMMAND $<TARGET_FILE:server> --help)
    set_tests_properties(test_server_help PROPERTIES
      PASS_REGULAR_EXPRESSION "--max-num-seqs N.*--max-num-batched-tokens N.*--cuda-profile-graph-replays N.*--cuda-profile-graph-batch N.*--benchmark-shutdown-fifo F.*--enable-force-include-usage.*--\\[no-\\]enable-prefix-caching")
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
