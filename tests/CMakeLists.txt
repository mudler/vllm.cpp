add_library(vllm_test_main STATIC doctest_main.cpp)
target_include_directories(vllm_test_main SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

if(UNIX)
  add_test(NAME test_triton_aot_drift_contract
    COMMAND bash "${CMAKE_SOURCE_DIR}/tests/scripts/test_triton_aot_drift.sh")
endif()

# CPU-only benchmark-harness contract. The tools deliberately use only the
# Python standard library; image/model/GPU preflights remain dgx-only.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME test_serve_low_tools
  COMMAND ${Python3_EXECUTABLE} -m unittest discover
    -s "${CMAKE_SOURCE_DIR}/tests/tools" -p "test_*.py")
set_tests_properties(test_serve_low_tools PROPERTIES
  ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}")

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  vllm_cpp_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_version vllm/test_version.cpp)
vllm_cpp_add_test(test_sampling_params vllm/test_sampling_params.cpp)
vllm_cpp_add_test(test_hf_config vllm/test_hf_config.cpp)
vllm_cpp_add_test(test_safetensors vllm/test_safetensors.cpp)
vllm_cpp_add_test(test_gguf vllm/test_gguf.cpp)
vllm_cpp_add_test(test_nvfp4_dequant vllm/test_nvfp4_dequant.cpp)
vllm_cpp_add_test(test_ct_nvfp4_emulation vllm/test_ct_nvfp4_emulation.cpp)
vllm_cpp_add_test(test_gguf_dequant vllm/test_gguf_dequant.cpp)
vllm_cpp_add_test(test_qwen36_weights vllm/test_qwen36_weights.cpp)
vllm_cpp_add_test(test_gguf_qwen36_loader vllm/test_gguf_qwen36_loader.cpp)
vllm_cpp_add_test(test_model_loader_gguf vllm/test_model_loader_gguf.cpp)
vllm_cpp_add_test(test_model_registry vllm/models/test_model_registry.cpp)
target_include_directories(test_model_registry PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_decode_graph_sizes vllm/models/test_decode_graph_sizes.cpp)
vllm_cpp_add_test(test_qwen35_paged_forward vllm/models/test_qwen35_paged_forward.cpp)
vllm_cpp_add_test(test_qwen27_dense_forward vllm/models/test_qwen27_dense_forward.cpp)
vllm_cpp_add_test(test_qwen27_paged_forward vllm/models/test_qwen27_paged_forward.cpp)
target_include_directories(test_qwen27_paged_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_unicode_data vllm/test_unicode_data.cpp)
vllm_cpp_add_test(test_pretokenizer vllm/test_pretokenizer.cpp)
vllm_cpp_add_test(test_bpe vllm/test_bpe.cpp)
vllm_cpp_add_test(test_detokenizer vllm/test_detokenizer.cpp)
vllm_cpp_add_test(test_request vllm/test_request.cpp)
vllm_cpp_add_test(test_kv_cache_utils vllm/v1/test_kv_cache_utils.cpp)
vllm_cpp_add_test(test_kv_cache_interface vllm/v1/test_kv_cache_interface.cpp)
vllm_cpp_add_test(test_block_pool vllm/v1/test_block_pool.cpp)
vllm_cpp_add_test(test_single_type_kv_cache_manager vllm/v1/test_single_type_kv_cache_manager.cpp)
vllm_cpp_add_test(test_kv_cache_coordinator vllm/v1/test_kv_cache_coordinator.cpp)
vllm_cpp_add_test(test_kv_cache_manager vllm/v1/test_kv_cache_manager.cpp)
vllm_cpp_add_test(test_scheduler_config vllm/test_scheduler_config.cpp)
vllm_cpp_add_test(test_request_queue vllm/v1/test_request_queue.cpp)
vllm_cpp_add_test(test_sched_output vllm/v1/test_sched_output.cpp)
vllm_cpp_add_test(test_scheduler vllm/v1/test_scheduler.cpp)
vllm_cpp_add_test(test_async_scheduler vllm/v1/test_async_scheduler.cpp)
vllm_cpp_add_test(test_scheduler_wave vllm/v1/test_scheduler_wave.cpp)
vllm_cpp_add_test(test_block_table vllm/v1/worker/test_block_table.cpp)
vllm_cpp_add_test(test_input_batch vllm/v1/worker/test_input_batch.cpp)
vllm_cpp_add_test(test_sampling_metadata vllm/v1/sample/test_metadata.cpp)
vllm_cpp_add_test(test_logits_processors vllm/v1/sample/test_logits_processors.cpp)
vllm_cpp_add_test(test_sampler vllm/v1/sample/test_sampler.cpp)
vllm_cpp_add_test(test_prepare_inputs vllm/v1/worker/test_prepare_inputs.cpp)
vllm_cpp_add_test(test_combine_tokens vllm/v1/worker/test_combine_tokens.cpp)
vllm_cpp_add_test(test_async_output vllm/v1/worker/test_async_output.cpp)
vllm_cpp_add_test(test_async_runner_flag vllm/v1/worker/test_async_runner_flag.cpp)
vllm_cpp_add_test(test_runner vllm/v1/worker/test_runner.cpp)
# test_runner uses detail::ValidateGdnStateIndices from the internal header.
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_common_attn_metadata vllm/v1/attention/test_common_attn_metadata.cpp)
vllm_cpp_add_test(test_attention_window vllm/model_executor/layers/attention/test_attention.cpp)
vllm_cpp_add_test(test_chunked_local_attention
  vllm/v1/attention/test_chunked_local_attention.cpp)
vllm_cpp_add_test(test_rotary_embedding
  vllm/model_executor/layers/rotary_embedding/test_rotary_embedding.cpp)
vllm_cpp_add_test(test_gdn_metadata_builder vllm/v1/attention/test_gdn_metadata_builder.cpp)
vllm_cpp_add_test(test_mtp_speculator vllm/v1/spec_decode/test_mtp_speculator.cpp)
vllm_cpp_add_test(test_structured_output vllm/v1/structured_output/test_structured_output.cpp)
vllm_cpp_add_test(test_apply_grammar_bitmask vllm/v1/structured_output/test_apply_grammar_bitmask.cpp)
vllm_cpp_add_test(test_backend_native vllm/v1/structured_output/test_backend_native.cpp)
vllm_cpp_add_test(test_json_schema_to_gbnf vllm/v1/structured_output/test_json_schema_to_gbnf.cpp)
vllm_cpp_add_test(test_response_format_e2e vllm/v1/structured_output/test_response_format_e2e.cpp)
vllm_cpp_add_test(test_engine_types vllm/test_engine_types.cpp)
vllm_cpp_add_test(test_engine_core vllm/v1/test_engine_core.cpp)
vllm_cpp_add_test(test_engine_core_proc vllm/v1/test_engine_core_proc.cpp)
vllm_cpp_add_test(test_async_admission_timing vllm/v1/test_async_admission_timing.cpp)
vllm_cpp_add_test(test_input_processor vllm/v1/test_input_processor.cpp)
target_compile_definitions(test_input_processor PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
vllm_cpp_add_test(test_output_processor vllm/v1/test_output_processor.cpp)
vllm_cpp_add_test(test_async_llm vllm/v1/test_async_llm.cpp)
vllm_cpp_add_test(test_llm_engine vllm/v1/test_llm_engine.cpp)
vllm_cpp_add_test(test_loaded_engine_dense vllm/entrypoints/test_loaded_engine_dense.cpp)
vllm_cpp_add_test(test_openai_protocol vllm/entrypoints/openai/test_protocol.cpp)
vllm_cpp_add_test(test_openai_serving vllm/entrypoints/openai/test_serving.cpp)
vllm_cpp_add_test(test_openai_utf8 vllm/entrypoints/openai/test_utf8.cpp)
vllm_cpp_add_test(test_openai_tool_parsers
  vllm/entrypoints/openai/tool_parsers/test_tool_parsers.cpp)
vllm_cpp_add_test(test_tool_choice_grammar
  vllm/entrypoints/openai/tool_parsers/test_tool_choice_grammar.cpp)
vllm_cpp_add_test(test_chat_template vllm/entrypoints/test_chat_template.cpp)
# The HTTP api_server smoke test needs the vendored cpp-httplib (gated).
if(VLLM_CPP_SERVER)
  vllm_cpp_add_test(test_openai_api_server vllm/entrypoints/openai/test_api_server.cpp)
  # M3.6: the OpenAI server CONFORMANCE suite — the full API contract exercised
  # end to end over the REAL cpp-httplib server on an ephemeral port.
  vllm_cpp_add_test(test_openai_conformance vllm/entrypoints/openai/test_conformance.cpp)
endif()
vllm_cpp_add_test(test_outputs vllm/test_outputs.cpp)

# M2.1: the benchmark harness smoke — drives the synthetic engine through the
# full admission + step() measurement loop (shares examples/bench/bench_core.h
# with the vllm-bench CLI).
vllm_cpp_add_test(test_bench examples/test_bench.cpp)
target_include_directories(test_bench PRIVATE ${CMAKE_SOURCE_DIR}/examples/bench)

# C ABI (M3.5 Task 1): the C++ test drives the public C API over a synthetic
# engine via the internal MakeEngineHandle hook (reached under src/).
vllm_cpp_add_test(test_capi capi/test_capi.cpp)
target_include_directories(test_capi PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Pure-C compile check: include/vllm.h must compile as C11 with no C++ leaks.
enable_language(C)
add_library(vllm_capi_c_check STATIC capi/c_header_compile.c)
target_link_libraries(vllm_capi_c_check PRIVATE vllm::vllm)
target_compile_options(vllm_capi_c_check PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Werror -std=c11>)

# ── Packaging DoD (M3.5 Task 3): dlopen the built libvllm.so + verify exports ──
# These are gated on the shared-lib target existing. The dlopen smoke test links
# ONLY the doctest main + ${CMAKE_DL_LIBS} (NOT libvllm) — it resolves the whole
# C ABI at runtime via dlsym, proving a header-less FFI consumer works.
if(TARGET vllm_shared)
  add_executable(test_dlopen capi/test_dlopen.cpp)
  target_link_libraries(test_dlopen PRIVATE vllm_test_main ${CMAKE_DL_LIBS})
  # Only the public header for the ABI *types*; the .so path is injected so the
  # test dlopen()s the exact artifact this build produced.
  target_include_directories(test_dlopen PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(test_dlopen PRIVATE
    VLLM_SHARED_LIB_PATH="$<TARGET_FILE:vllm_shared>")
  vllm_cpp_set_warnings(test_dlopen)
  add_test(NAME test_dlopen COMMAND test_dlopen)

  # Export-set verification: fail if libvllm.so leaks any non-`vllm_*` dynamic
  # symbol (i.e. C++ internals). Runs nm via cmake -P.
  if(UNIX AND NOT APPLE)
    add_test(NAME capi_shared_exports_only_abi
      COMMAND ${CMAKE_COMMAND}
        "-DVLLM_SHARED_LIB=$<TARGET_FILE:vllm_shared>"
        -P "${CMAKE_SOURCE_DIR}/cmake/VerifyExports.cmake")
  endif()
endif()
vllm_cpp_add_test(test_cuda_backend vt/test_cuda_backend.cpp)
vllm_cpp_add_test(test_dropin_abi vt/test_dropin_abi.cpp)
target_include_directories(test_dropin_abi PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cuda_ops vt/test_cuda_ops.cpp)
vllm_cpp_add_test(test_dtype vt/test_dtype.cpp)
vllm_cpp_add_test(test_backend vt/test_backend.cpp)
vllm_cpp_add_test(test_tensor vt/test_tensor.cpp)
vllm_cpp_add_test(test_arena vt/test_arena.cpp)
vllm_cpp_add_test(test_ops_matmul vt/test_ops_matmul.cpp)
# CPU-tier contract for the env-gated cuBLASLt GEMM algo-selection diagnostic
# helpers. Reaches the internal header under src/ (not public API); the cuBLASLt
# emit it gates is CUDA-only and lives in src/vt/cuda/cuda_matmul.cu.
vllm_cpp_add_test(test_gemm_algo_log vt/test_gemm_algo_log.cpp)
target_include_directories(test_gemm_algo_log PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the register-resident packed-decode tiling A/B flag
# (VT_GDN_PACKED_REG_TILE). Reaches the internal header under src/ (not public
# API); the kernel it gates is CUDA-only and lives in src/vt/cuda/cuda_gdn.cu.
vllm_cpp_add_test(test_gdn_packed_reg_tile vt/test_gdn_packed_reg_tile.cpp)
target_include_directories(test_gdn_packed_reg_tile PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the vendored Triton AOT packed-decode fast-path flag
# (VT_GDN_PACKED_DECODE_TRITON, default ON). Reaches the internal header under
# src/ (not public API); the vendored cubin it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (TryTritonPackedDecode -> gdn_decode_h48).
vllm_cpp_add_test(test_gdn_packed_decode_triton vt/test_gdn_packed_decode_triton.cpp)
target_include_directories(test_gdn_packed_decode_triton PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the two NUMERICS-NEUTRAL FP4-quant vectorized-load
# fast-path flags (VT_FP4_QUANT_FAST + VT_SILU_FP4_FAST, both default OFF /
# '1'-opt-in). Reaches the internal header under src/ (not public API); the
# bit-identical vectorized kernels they gate are CUDA-only and live in
# src/vt/cuda/cuda_matmul_nvfp4.cu (ScaledFp4QuantFastKernel /
# SiluAndMulFp4QuantFastKernel).
vllm_cpp_add_test(test_fp4_quant_fast vt/test_fp4_quant_fast.cpp)
target_include_directories(test_fp4_quant_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the decode-RMSNorm kernel-efficiency A/B flag
# (VT_RMSNORM_DECODE_FAST). Reaches the internal header under src/ (not public
# API); the vectorized kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_ops.cu (RmsNormRowFastKernel).
vllm_cpp_add_test(test_rmsnorm_decode_fast vt/test_rmsnorm_decode_fast.cpp)
target_include_directories(test_rmsnorm_decode_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the GDN gated-RMSNorm decode kernel-efficiency A/B flag
# (VT_RMSNORM_GATED_FAST). Reaches the internal header under src/ (not public
# API); the bit-identical fast kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (RmsNormGatedRowFastKernel).
vllm_cpp_add_test(test_rmsnorm_gated_fast vt/test_rmsnorm_gated_fast.cpp)
target_include_directories(test_rmsnorm_gated_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the GDN decode causal_conv1d state-update kernel-efficiency
# A/B flag (VT_CONV_UPDATE_FAST). Reaches the internal header under src/ (not public
# API); the bit-identical fast kernel it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (CausalConv1dUpdateFastKernel).
vllm_cpp_add_test(test_conv_update_fast vt/test_conv_update_fast.cpp)
target_include_directories(test_conv_update_fast PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU threadpool (QUANT-GGUF-CPU-THREADPOOL): barrier/park-wake/chunk-steal
# stress + the n_threads 1/3/20 determinism battery. Reaches the internal
# header under src/ (the pool is not public API).
vllm_cpp_add_test(test_cpu_threadpool vt/test_cpu_threadpool.cpp)
target_include_directories(test_cpu_threadpool PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_nvfp4_matmul vt/test_ops_nvfp4_matmul.cpp)
vllm_cpp_add_test(test_ops_nvfp4_fp4 vt/test_ops_nvfp4_fp4.cpp)
target_include_directories(test_ops_nvfp4_fp4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_nvfp4_persistent_cache vt/test_nvfp4_persistent_cache.cpp)
target_include_directories(test_nvfp4_persistent_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_nvfp4_persistent_cache PRIVATE
  TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
if(VLLM_CPP_CUTLASS)
  foreach(_NVFP4_TACTIC RANGE 0 31)
    add_test(NAME test_ops_nvfp4_fp4_tactic_${_NVFP4_TACTIC}
      COMMAND test_ops_nvfp4_fp4
        "--test-case=matmul_nvfp4_cutlass every forced SM12 tactic matches reference and captures")
    set_tests_properties(test_ops_nvfp4_fp4_tactic_${_NVFP4_TACTIC} PROPERTIES
      ENVIRONMENT
        "VT_FP4_FORCE_TACTIC=${_NVFP4_TACTIC};VT_FP4_FULL_TACTICS=1;VT_FP4_AUTOTUNE=1")
  endforeach()
  add_test(NAME test_ops_nvfp4_persistent_runtime
    COMMAND test_ops_nvfp4_fp4
      "--test-case=NVFP4 persistent runtime loads frozen plans before warmup")
  set_tests_properties(test_ops_nvfp4_persistent_runtime PROPERTIES
    ENVIRONMENT
      "VT_FP4_TEST_PERSISTENT_RUNTIME=1;VT_FP4_PERSISTENT_CACHE=1;VT_FP4_AUTOTUNE_CACHE_READONLY=1;VT_FP4_FLASHINFER_CACHE_PATH=${CMAKE_SOURCE_DIR}/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json;VT_FP4_AUTOTUNE_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/nvfp4-runtime-readonly-unused.json;VT_FP4_AUTOTUNE_DELAY_US=5000;VT_FP4_PLAN_CACHE=1;VT_FP4_AUTOTUNE=1;VT_FP4_FULL_TACTICS=1")
  add_test(NAME test_ops_nvfp4_persistent_save_lifecycle
    COMMAND test_ops_nvfp4_fp4
      "--test-case=NVFP4 persistent runtime publishes only completed warmup")
  set_tests_properties(test_ops_nvfp4_persistent_save_lifecycle PROPERTIES
    ENVIRONMENT
      "VT_FP4_TEST_PERSISTENT_SAVE=1;VT_FP4_PERSISTENT_CACHE=1;VT_FP4_AUTOTUNE_CACHE_READONLY=0;VT_FP4_AUTOTUNE_CACHE_PATH=${CMAKE_CURRENT_BINARY_DIR}/nvfp4-runtime-save-lifecycle.json;VT_FP4_AUTOTUNE_DELAY_US=0;VT_FP4_PLAN_CACHE=1;VT_FP4_AUTOTUNE=1;VT_FP4_FULL_TACTICS=1")
endif()
vllm_cpp_add_test(test_ops_fp8_cutlass vt/test_ops_fp8_cutlass.cpp)
vllm_cpp_add_test(test_ops_moe_grouped vt/test_ops_moe_grouped.cpp)
vllm_cpp_add_test(test_ops_rmsnorm vt/test_ops_rmsnorm.cpp)
vllm_cpp_add_test(test_ops_fused_chain vt/test_ops_fused_chain.cpp)
vllm_cpp_add_test(test_ops_activation vt/test_ops_activation.cpp)
vllm_cpp_add_test(test_ops_rope vt/test_ops_rope.cpp)
vllm_cpp_add_test(test_ops_rope_cache vt/test_ops_rope_cache.cpp)
vllm_cpp_add_test(test_ops_gdn vt/test_ops_gdn.cpp)
target_include_directories(test_ops_gdn PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_TRITON)
  # A separate process guarantees this race test reaches the generated module
  # loaders before another gate-shape GDN case can warm them.
  add_test(NAME test_ops_gdn_aot_concurrent_first_load
    COMMAND test_ops_gdn
      "--test-case=CUDA gdn Triton AOT concurrent first load is safe across two queues")
endif()
vllm_cpp_add_test(test_ops_moe vt/test_ops_moe.cpp)
vllm_cpp_add_test(test_ops_glue vt/test_ops_glue.cpp)
vllm_cpp_add_test(test_ops_attn_preamble vt/test_ops_attn_preamble.cpp)
vllm_cpp_add_test(test_ops_attention vt/test_ops_attention.cpp)
vllm_cpp_add_test(test_ops_reshape_cache vt/test_ops_reshape_cache.cpp)
vllm_cpp_add_test(test_ops_paged_attn vt/test_ops_paged_attn.cpp)
vllm_cpp_add_test(test_ops_sample vt/test_ops_sample.cpp)
vllm_cpp_add_test(test_ops_penalties vt/test_ops_penalties.cpp)

vllm_cpp_add_test(test_npy parity/test_npy.cpp)
target_compile_definitions(test_npy PRIVATE
  VLLM_CPP_GOLDENS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity/goldens")

vllm_cpp_add_test(test_tokenizer_parity vllm/test_tokenizer_parity.cpp)
target_compile_definitions(test_tokenizer_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

vllm_cpp_add_test(test_op_parity parity/test_op_parity.cpp)
target_compile_definitions(test_op_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_op_parity PRIVATE ${CMAKE_SOURCE_DIR}/tests/parity)

# Paged-engine 35B greedy acceptance gate (checkpoint-gated, dgx-only; skips on
# CPU/CI when the real snapshot is absent). Counterpart to RunQwen36Logits.
vllm_cpp_add_test(test_qwen36_paged_engine parity/test_qwen36_paged_engine.cpp)
target_compile_definitions(test_qwen36_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GGUF 35B greedy acceptance gate (checkpoint-gated, dgx-only; skips when the
# APEX k-quant GGUFs are absent). Loads the REAL .gguf single-file path
# (metadata + embedded vocab + k-quant dequant) through the paged engine and
# asserts greedy token-for-token vs the pinned same-file llama.cpp oracle.
vllm_cpp_add_test(test_qwen36_gguf_engine parity/test_qwen36_gguf_engine.cpp)
target_compile_definitions(test_qwen36_gguf_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_gguf_engine PRIVATE ${CMAKE_SOURCE_DIR}/tests/parity)

# Dense 27B W4A4 greedy acceptance gate (checkpoint-gated, dgx-only; skips on
# CPU/CI when the real snapshot is absent). Counterpart to the 35B gate; asserts
# the paged dense engine reproduces the pip-vLLM oracle greedy continuation
# token-for-token via the fp4-resident W4A4 GEMM. See qwen27b-w4a4-notes.md §5.
vllm_cpp_add_test(test_qwen27_paged_engine parity/test_qwen27_paged_engine.cpp)
target_compile_definitions(test_qwen27_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
