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
  # tests/ on the include path so every suite can reach the shared test helpers
  # under tests/support/ as "support/<name>.h". No header name under tests/
  # collides with one under include/ or src/, so nothing else resolves here.
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(TARGET vllm::sanitize_test_shared)
    target_link_libraries(${name} PRIVATE vllm::sanitize_test_shared vllm_test_main)
  else()
    target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  endif()
  vllm_cpp_set_warnings(${name})
  add_test(NAME ${name} COMMAND ${name})
  # A gate that cannot run must not report success. doctest exits 0 after a
  # TEST_CASE returns early, printing "assertions: 0 | 0 passed | 0 failed" and
  # "Status: SUCCESS!" — indistinguishable, in a log or a `&&` chain, from a gate
  # that loaded a model and matched an oracle (issue #463; multimodal-speed.md
  # §17.7 hit exactly this). A test whose preconditions are absent exits 77
  # instead and CTest reports it **Skipped**. Inert for any test that never
  # returns 77, which today is all but test_voxtral_e2e.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

# The gate on the golden-comparison helper itself (issue #449): a NaN operand
# must be a FAILURE, never the 0.0 that both historical spellings returned.
vllm_cpp_add_test(test_max_abs_diff support/test_max_abs_diff.cpp)
# ...and the wrapper must RAISE, not merely return +infinity, so the
# "these must differ" callers cannot read a NaN as a difference. That case is
# skipped in the normal run because a raised failure would count as one; here it
# is forced on and REQUIRED to exit non-zero. Without the guard this entry fails.
add_test(NAME test_max_abs_diff_nan_raises
  COMMAND test_max_abs_diff --no-skip --test-case=*RAISES*)
set_tests_properties(test_max_abs_diff_nan_raises PROPERTIES WILL_FAIL TRUE)

vllm_cpp_add_test(test_version vllm/test_version.cpp)
set(_vllm_cpp_expected_build_version "${PROJECT_VERSION}")
if(DEFINED VLLM_CPP_BUILD_VERSION)
  set(_vllm_cpp_expected_build_version "${VLLM_CPP_BUILD_VERSION}")
endif()
target_compile_definitions(test_version PRIVATE
  VLLM_CPP_EXPECTED_BUILD_VERSION="${_vllm_cpp_expected_build_version}")
vllm_cpp_add_test(test_sampling_params vllm/test_sampling_params.cpp)
vllm_cpp_add_test(test_punica_cpu vllm/lora/test_punica_cpu.cpp)
vllm_cpp_add_test(test_lora_layers vllm/lora/test_lora_layers.cpp)
vllm_cpp_add_test(test_plugin_system
  vllm/plugins/test_plugin_system.cpp
  vllm/plugins/toy_model_plugin.cpp)
vllm_cpp_add_test(test_hf_config vllm/test_hf_config.cpp)
vllm_cpp_add_test(test_safetensors vllm/test_safetensors.cpp)
vllm_cpp_add_test(test_load_direct_upload vllm/test_load_direct_upload.cpp)
vllm_cpp_add_test(test_gguf vllm/test_gguf.cpp)
vllm_cpp_add_test(test_nvfp4_dequant vllm/test_nvfp4_dequant.cpp)
vllm_cpp_add_test(test_awq_gptq_dequant vllm/test_awq_gptq_dequant.cpp)
vllm_cpp_add_test(test_mxfp4_dequant vllm/test_mxfp4_dequant.cpp)
vllm_cpp_add_test(test_ct_nvfp4_emulation vllm/test_ct_nvfp4_emulation.cpp)
# MODEL-TEXT-nemotron-h W1 (#517): the ModelOpt MIXED_PRECISION per-module
# quant-algo resolver. Two arms, deliberately two BINARIES: the curated fixture
# arm is always-on, and the exhaustive real-config arm exits 77 (Skipped) when
# the 30B checkpoint is not staged. Folding them into one target would let the
# skip swallow the always-on gate as well.
#
# The resolver header is INTERNAL (src/, not include/) until a loader consumes
# it, so both targets take ${CMAKE_SOURCE_DIR}/src on the include path — the
# same arrangement test_linear_method and test_model_registry already use.
vllm_cpp_add_test(test_modelopt_mixed_precision
  vllm/model_executor/layers/quantization/test_modelopt_mixed_precision.cpp)
target_include_directories(test_modelopt_mixed_precision PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_modelopt_mixed_precision PRIVATE
  MODELOPT_MIXED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/modelopt_mixed_precision")
vllm_cpp_add_test(test_modelopt_mixed_precision_checkpoint
  vllm/model_executor/layers/quantization/test_modelopt_mixed_precision_checkpoint.cpp)
# tests/parity as well as src/: the exhaustive arm resolves its checkpoint
# through the SINGLE pinned parity::Nemotron35LightningSnapshot() rather than
# joining $CHECKPOINT_ROOT by hand, so that one resolver owns the revision pin
# for both env spellings (#517 LOW-3) -- the same arrangement test_qwen36_weights
# uses for the 35B pin.
target_include_directories(test_modelopt_mixed_precision_checkpoint PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests/parity)
vllm_cpp_add_test(test_gguf_dequant vllm/test_gguf_dequant.cpp)
vllm_cpp_add_test(test_gguf_nvfp4 vllm/test_gguf_nvfp4.cpp)
# gguf_nvfp4_goldens.inc lives next to the test source.
target_include_directories(test_gguf_nvfp4 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
vllm_cpp_add_test(test_qwen36_weights vllm/test_qwen36_weights.cpp)
# GATE-PIN-UNPINNED-SNAPSHOTS (#471): resolves the 35B checkpoint through the
# pinned parity::HfSnapshot, so it needs tests/parity on the include path.
target_include_directories(test_qwen36_weights PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
vllm_cpp_add_test(test_gguf_qwen36_loader vllm/test_gguf_qwen36_loader.cpp)
vllm_cpp_add_test(test_gguf_keep_quant vllm/test_gguf_keep_quant.cpp)
vllm_cpp_add_test(test_model_loader_gguf vllm/test_model_loader_gguf.cpp)
vllm_cpp_add_test(test_model_registry vllm/models/test_model_registry.cpp)
vllm_cpp_add_test(test_vocoder1d vllm/models/test_vocoder1d.cpp)
vllm_cpp_add_test(test_speech_engine vllm/multimodal/test_speech_engine.cpp)
vllm_cpp_add_test(test_indextts2_family vllm/models/test_indextts2_family.cpp)
vllm_cpp_add_test(test_campplus vllm/models/test_campplus.cpp)
vllm_cpp_add_test(test_w2vbert vllm/models/test_w2vbert.cpp)
vllm_cpp_add_test(test_fvq vllm/models/test_fvq.cpp)
vllm_cpp_add_test(test_wavenet vllm/models/test_wavenet.cpp)
vllm_cpp_add_test(test_dit_tail vllm/models/test_dit_tail.cpp)
vllm_cpp_add_test(test_dit_skip vllm/models/test_dit_skip.cpp)
vllm_cpp_add_test(test_indextts2_s2mel_loader vllm/models/test_indextts2_s2mel_loader.cpp)
vllm_cpp_add_test(test_vocos vllm/models/test_vocos.cpp)
vllm_cpp_add_test(test_lenreg vllm/models/test_lenreg.cpp)
vllm_cpp_add_test(test_cfm vllm/models/test_cfm.cpp)
vllm_cpp_add_test(test_adaln vllm/models/test_adaln.cpp)
vllm_cpp_add_test(test_dit vllm/models/test_dit.cpp)
vllm_cpp_add_test(test_talker vllm/models/test_talker.cpp)
vllm_cpp_add_test(test_indextts2_pipeline vllm/models/test_indextts2_pipeline.cpp)
target_include_directories(test_dit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_adaln PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_cfm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_lenreg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_vocos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_fvq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_wavenet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_tail PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_w2vbert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_campplus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gpt2 vllm/models/test_gpt2.cpp)
target_include_directories(test_gpt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_minimax_h3 vllm/models/test_minimax_h3.cpp)
# MiniMax-Music3 W1 (#672): the six-component diffusers-arm loader, gated against
# the REAL checkpoint's headers (minimax_music3_manifest.inc, 1012 tensors, no
# weight bytes) so CI never needs the 28.5 GB asset.
vllm_cpp_add_test(test_minimax_music3_loader vllm/models/test_minimax_music3_loader.cpp)
target_include_directories(test_minimax_music3_loader PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# ARCH-ONE-SURFACE ROW 2 fold gate: the video seam == the replicated pre-fold
# pipeline == the goldens the PRE-fold minimax-h3-gen binary rendered at the
# branch base (frames + WAV byte-identical, mux argv pinned).
vllm_cpp_add_test(test_minimax_h3_video_fold vllm/models/test_minimax_h3_video_fold.cpp)
target_compile_definitions(test_minimax_h3_video_fold PRIVATE
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold")
# LTX-2.5 L1: the GENERALIZED VideoEngine seam + its family registry, held to
# the SAME pre-fold goldens through the abstract path (and to the never-guess
# refusals). Needs the fold fixture builder, which lives beside the H3 test.
vllm_cpp_add_test(test_video_engine vllm/multimodal/test_video_engine.cpp)
target_include_directories(test_video_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_compile_definitions(test_video_engine PRIVATE
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold")
# LTX-2.5 L7: the family behind the VideoEngine seam and the driving loop, run
# over a reduced-dimension checkpoint set written in the SHIPPED file format.
vllm_cpp_add_test(test_ltx2_video vllm/multimodal/test_ltx2_video.cpp)
# LTX-2.5 DiT parity gate (row MODEL-DIFFUSION-LTX25, spec phase L2). The goldens
# live next to the test source, like every other generated .inc.
vllm_cpp_add_test(test_ltx2 vllm/models/test_ltx2.cpp)
# LTX-2.5 L8: the DEVICE-RESIDENT forward, held to the SAME ltx2_goldens.inc the
# L2 CPU gate above is held to. Its own target so test_ltx2's recorded 29/1615
# baseline stays exactly where it is; it runs on the CPU backend everywhere and
# adds the CUDA cases when a CUDA backend is registered.
vllm_cpp_add_test(test_ltx2_device vllm/models/test_ltx2_device.cpp)
# LTX-2.5 L4: the VAE bricks, gated against upstream ltx_core executed at reduced
# dimensions. ltx2_vae_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_vae vllm/models/test_ltx2_vae.cpp)
target_include_directories(test_ltx2_vae PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L3: the text conditioning path, gated against upstream ltx_core executed
# at reduced dimensions. ltx2_text_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_text_encoder vllm/models/test_ltx2_text_encoder.cpp)
target_include_directories(test_ltx2_text_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L5: the pipeline layer — sigma schedule, noiser, diffusion steps,
# guidance, patchifiers, the recipe table, the latent spatial upsampler, the
# duration head and the embeddings connector. ltx2_pipeline_goldens.inc lives
# next to the test source.
vllm_cpp_add_test(test_ltx2_pipeline vllm/models/test_ltx2_pipeline.cpp)
target_include_directories(test_ltx2_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L6: the quantized loaders, gated three ways — the SHIPPED checkpoints'
# own manifests, a few hundred of their own bytes against torch, and synthetic
# files for whole-model materialization. All three .inc files live beside the
# test source.
vllm_cpp_add_test(test_ltx2_loader vllm/models/test_ltx2_loader.cpp)
target_include_directories(test_ltx2_loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_video_api vllm/entrypoints/openai/test_video_api.cpp)
target_include_directories(test_model_registry PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_platform vllm/platforms/test_platform.cpp)
vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
vllm_cpp_add_test(test_linear_method vllm/model_executor/layers/test_linear_method.cpp)
target_include_directories(test_linear_method PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_decode_graph_sizes vllm/models/test_decode_graph_sizes.cpp)
vllm_cpp_add_test(test_qwen35_paged_forward vllm/models/test_qwen35_paged_forward.cpp)
# SPEC-MTP I5a: GDN layer spec routing (GdnBlockPaged spec branch) — needs the
# internal header under src/ for the GdnBlockPagedForTest entry point.
vllm_cpp_add_test(test_qwen3_5_gdn_spec_routing vllm/models/test_qwen3_5_gdn_spec_routing.cpp)
vllm_cpp_add_test(test_qwen3_5_gguf_mtp vllm/models/test_qwen3_5_gguf_mtp.cpp)
vllm_cpp_add_test(test_qwen3_dflash_gguf vllm/models/test_qwen3_dflash_gguf.cpp)
target_include_directories(test_qwen3_5_gdn_spec_routing PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen27_dense_forward vllm/models/test_qwen27_dense_forward.cpp)
# MODEL-GDN-OUTPUT-GATE-TYPE (#489): `GlueFuseEnabled()` defaults ON and is read
# ONCE per process into a function-local static, so one run of these two suites
# can only ever execute the glue-FUSED gated-RMSNorm tail. The UNFUSED tail
# carries its own `vt::RmsNormGatedArgs{eps, sigmoid_gate}` construction in each
# of the three GDN blocks, and inverting only that construction left the entire
# declared focused gate GREEN (test_hf_config 210|210, test_qwen27_dense_forward
# 583|583, test_qwen3_5_gdn_spec_routing 52|52) — live code, but ungated.
# Re-run the SAME binaries with the lever forced to 0 so the `VT_GLUE_FUSE=0`
# fallback is gated too, mirroring the precedent the fusion row itself set
# (parity-ledger.md:112 — "fused ON default AND `VT_GLUE_FUSE=0` fallback 16/16")
# and the per-lever re-run below.
foreach(_glue_fuse_test test_qwen27_dense_forward test_qwen3_5_gdn_spec_routing)
  add_test(NAME ${_glue_fuse_test}_glue_fuse_off COMMAND ${_glue_fuse_test})
  set_tests_properties(${_glue_fuse_test}_glue_fuse_off PROPERTIES
    ENVIRONMENT "VT_GLUE_FUSE=0" SKIP_RETURN_CODE 77)
endforeach()
# MODEL-GDN-OUTPUT-GATE-TYPE (#489), the FP8 half. Each GDN tail carries TWO fp8
# gate-carrying constructions: the `vt::kRmsNormGatedQuantFp8` FusedRecipe copy
# (taken at the `VT_FUSED_CHAIN_ADOPT` default) and the direct
# `vt::RmsNormGatedQuantFp8` hand-call (taken only at =0). `FusedChainAdoptEnabled()`
# is another read-once static, so a single run reaches exactly one of the two and
# the other is live-but-ungated — the same defect the `_glue_fuse_off` entries
# above were added for. Re-run the SAME binary with the lever forced to 0.
# CUDA-only in effect: the fp8 tail needs `Platform::supports_fp8()`.
add_test(NAME test_qwen3_5_gdn_spec_routing_fused_chain_off
  COMMAND test_qwen3_5_gdn_spec_routing)
set_tests_properties(test_qwen3_5_gdn_spec_routing_fused_chain_off PROPERTIES
  ENVIRONMENT "VT_FUSED_CHAIN_ADOPT=0" SKIP_RETURN_CODE 77)
# PERF-27B-DENSE-MARLIN-GATEUP (#365): the pre-kernel DECISIONS that route the
# 27B's dense W4A16 MLP through the fused Marlin gate_up pair. CPU-only by
# construction — the fused GEMM itself is CUDA + VT_MARLIN_NVFP4.
vllm_cpp_add_test(test_dense_gateup_fused_marlin
  vllm/models/test_dense_gateup_fused_marlin.cpp)
# Each of the three levers the composed guard ANDs is read ONCE per process into
# a function-local static, so a single run can only ever observe one value of
# each and a deleted / hardwired / misspelled term stays green (fresh-review
# findings F1/F2). Re-run the SAME binary with each lever forced to 0: the
# "EVERY lever term is load-bearing" case then asserts the guard reads OFF, which
# is the assertion a dropped term fails.
foreach(_gateup_lever VT_DENSE_MARLIN_GATEUP VT_NVFP4_MARLIN VT_MOE_FUSED_W13)
  string(TOLOWER "${_gateup_lever}" _gateup_lever_lc)
  add_test(NAME test_dense_gateup_fused_marlin_off_${_gateup_lever_lc}
    COMMAND test_dense_gateup_fused_marlin)
  set_tests_properties(test_dense_gateup_fused_marlin_off_${_gateup_lever_lc}
    PROPERTIES ENVIRONMENT "${_gateup_lever}=0")
endforeach()
vllm_cpp_add_test(test_qwen35_plain_weights vllm/models/test_qwen35_plain_weights.cpp)
vllm_cpp_add_test(test_qwen3_8_text_only vllm/models/test_qwen3_8_text_only.cpp)
# The PUBLISHED `Qwen/Qwen3.8-2.4T-A95B` config.json, committed VERBATIM, so the
# config-resolution case reads the real document (nested `rope_parameters`, 92
# `layer_types`, transformers-4.57.3 `dtype`) instead of a paraphrase of it.
target_compile_definitions(test_qwen3_8_text_only PRIVATE
  QWEN3_8_CONFIG_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen3_8_2_4t_a95b/config.json")
# POOL-DEVICE-KEY (#516, .agents/specs/pool-device-key.md): the shared scratch
# pool's free list must be keyed by DEVICE as well as size class, or a block
# allocated through one backend is handed to a DBuf running on another —
# host-side SIGSEGV in one direction, silent all-NaN output in the other.
# Hardware-free: two fake backends stand in for two devices, so this runs
# everywhere and in ~milliseconds rather than only where a GPU is present.
vllm_cpp_add_test(test_device_pool vllm/models/test_device_pool.cpp)
vllm_cpp_add_test(test_moe_resident_lifetime vllm/models/test_moe_resident_lifetime.cpp)
vllm_cpp_add_test(test_qwen3_load vllm/models/test_qwen3_load.cpp)
vllm_cpp_add_test(test_qwen3_forward vllm/models/test_qwen3_forward.cpp)
target_include_directories(test_qwen3_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_dflash_forward vllm/models/test_qwen3_dflash_forward.cpp)
target_include_directories(test_qwen3_dflash_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W2: the DSpark Markov transition head + draft->target vocab map.
vllm_cpp_add_test(test_qwen3_dspark_markov vllm/models/test_qwen3_dspark_markov.cpp)
target_include_directories(test_qwen3_dspark_markov PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W3: speculators-format DSpark config translation.
vllm_cpp_add_test(test_qwen3_dspark_config vllm/models/test_qwen3_dspark_config.cpp)
target_include_directories(test_qwen3_dspark_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W3: the DSpark draft weight loader (Markov head + d2t).
vllm_cpp_add_test(test_qwen3_dspark_weights vllm/models/test_qwen3_dspark_weights.cpp)
target_include_directories(test_qwen3_dspark_weights PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DSPARK W4: sequential Markov sampling + the anchor block layout.
vllm_cpp_add_test(test_cudagraph_dispatch vllm/v1/worker/gpu/test_cudagraph_dispatch.cpp)
target_include_directories(test_cudagraph_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_dspark_sample vllm/v1/spec_decode/test_dspark_sample.cpp)
target_include_directories(test_dspark_sample PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_llama_forward vllm/models/test_llama_forward.cpp)
target_include_directories(test_llama_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# FUSION-DENSE-MIGRATE: executed CPU coverage for the four DENSE model TUs whose
# gate-up MLP now routes through layers::UnquantizedMlpGateUpMethod (their only
# other gates are checkpoint-gated + dgx-only).
vllm_cpp_add_test(test_dense_gate_up_seam_forward
  vllm/models/test_dense_gate_up_seam_forward.cpp)
target_include_directories(test_dense_gate_up_seam_forward PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
# ARCH-ONE-SURFACE fold ROW 6: embeddings through the registry/runner path on
# the committed tiny LlamaModel fixture (see scripts/mm/llama_embed_fixture_gen.py).
vllm_cpp_add_test(test_llama_embedding_fold
  vllm/models/test_llama_embedding_fold.cpp)
target_compile_definitions(test_llama_embedding_fold PRIVATE
  LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
vllm_cpp_add_test(test_mistral_forward vllm/models/test_mistral_forward.cpp)
target_include_directories(test_mistral_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma3_load vllm/models/test_gemma3_load.cpp)
vllm_cpp_add_test(test_gemma3_forward vllm/models/test_gemma3_forward.cpp)
target_include_directories(test_gemma3_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma2_load vllm/models/test_gemma2_load.cpp)
vllm_cpp_add_test(test_gemma2_forward vllm/models/test_gemma2_forward.cpp)
target_include_directories(test_gemma2_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma_load vllm/models/test_gemma_load.cpp)
vllm_cpp_add_test(test_gemma_forward vllm/models/test_gemma_forward.cpp)
target_include_directories(test_gemma_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_gemma4_honesty vllm/models/test_gemma4_honesty.cpp)
vllm_cpp_add_test(test_qwen3_moe_load vllm/models/test_qwen3_moe_load.cpp)
vllm_cpp_add_test(test_qwen3_moe_forward vllm/models/test_qwen3_moe_forward.cpp)
target_include_directories(test_qwen3_moe_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_deepseek_v2_load vllm/models/test_deepseek_v2_load.cpp)
vllm_cpp_add_test(test_deepseek_v2_forward vllm/models/test_deepseek_v2_forward.cpp)
target_include_directories(test_deepseek_v2_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)

# DeepSeek-V4-Flash (DeepseekV4ForCausalLM) W1/W2 scaffolding gate: the arch
# resolves through the registry + the shipped config descends through
# ParseDeepseekV4Params. No checkpoint / GPU — the forward + strict gate are
# named W3-W8 residuals (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_scaffold vllm/models/test_deepseek_v4_scaffold.cpp)
target_include_directories(test_deepseek_v4_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)

# DeepSeek-V4-Pro shape-generality gate (#504): the Pro config descends through
# the SAME ParseDeepseekV4Params with zero new keys, and compress_ratios alone
# reproduces the compressor/indexer layer counts of BOTH real checkpoints
# (Flash 41/21, Pro 61/30). No checkpoint / GPU — Pro's 805 GiB does not fit one
# GB10, so a load gate is memory-infeasible (.agents/specs/deepseek-v4-pro.md).
vllm_cpp_add_test(test_deepseek_v4_pro_variant vllm/models/test_deepseek_v4_pro_variant.cpp)
target_include_directories(test_deepseek_v4_pro_variant PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Laguna-S-2.1 W1/W2 scaffolding gate: LagunaForCausalLM RESOLVES through the
# registry + ParseLagunaParams descends the shipped config (nested dual-rope +
# variable Q-head + ungrouped sigmoid-noaux MoE). No 73 GB checkpoint / GPU — the
# forward + loader materialization + strict dual-oracle gate are named W3/W4
# residuals (.agents/specs/laguna-s21-w1w2-2026-07-30.md).
vllm_cpp_add_test(test_laguna_scaffold vllm/models/test_laguna_scaffold.cpp)
target_include_directories(test_laguna_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_laguna_nvfp4_loader vllm/models/test_laguna_nvfp4_loader.cpp)
target_include_directories(test_laguna_nvfp4_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)

# DeepSeek-V4-Flash W3: the NEW attention primitives (DSA Lightning-Indexer
# top-k selection + 512-wide MLA attention sinks + grouped output-LoRA) unit-
# gated vs hand-derived cases + double-precision references. Full-model gate is
# multi-Spark-blocked (156.7 GiB); MHC/MoE are W5/W6 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_dsa vllm/models/test_deepseek_v4_dsa.cpp)

# DeepSeek-V4-Flash W4: the DSA COMPRESSOR forward (softmax-weighted window pool
# + save-time APE add + RMSNorm) + the fp8_ds_mla KV-cache state layout (448 fp8
# UE8M0 + 64 bf16 rope, 576B stride) unit-gated vs hand-derived cases +
# double-precision references. Full-model gate is multi-Spark-blocked (156.7 GiB);
# MHC/MoE/device-kernel are W5/W6/W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_compressor vllm/models/test_deepseek_v4_compressor.cpp)
# DeepSeek-V4-Flash W5 — Manifold/Markov Hyper-Connections (MHC): the Sinkhorn-
# normalized hc_mult-stream residual manifold + mHC pre/post mixes + hc_head
# collapse + folded RMSNorms. Ported from the vLLM eager reference
# (mhc/torch.py + triton.py, 4 upstream impls agree) + gated against a
# from-first-principles double-precision derivation. Full-model gate multi-Spark-
# blocked; MoE/device-kernel are W6/W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_mhc vllm/models/test_deepseek_v4_mhc.cpp)
# DeepSeek-V4-Flash W6 — the sqrtsoftplus + hash-routed MoE: the V4 router score
# function (sqrt∘softplus), the noaux_tc bias-for-selection top-k, the hash
# `tid2eid` route that bypasses top-k, and the clamped SwiGLU expert activation.
# Ported from the vLLM eager reference (fused_topk_bias_router.py + activation.py,
# cross-checked SGLang) + gated against hand-derived literals and double-precision
# references. Full-model gate multi-Spark-blocked (156.7 GiB); device-kernel +
# forward assembly are W7 (.agents/specs/deepseek-v4-flash.md).
vllm_cpp_add_test(test_deepseek_v4_moe vllm/models/test_deepseek_v4_moe.cpp)
# DeepSeek-V4-Flash W7 — the DeepseekV4Model::Forward ASSEMBLY: composes the four
# landed host primitives (W3 DSA/MLA seams, W4 compressor + fp8_ds_mla KV, W5 MHC
# + Sinkhorn, W6 sqrtsoftplus/hash MoE) into an end-to-end logits producer at a
# tiny synthetic config. STRUCTURAL/composition gate (interleave runs, [T,hc,H]
# stream, hash-by-tid routing, DSA select+compress, finiteness) + RED-first
# miswires — NOT a real-checkpoint token gate (that is W8, multi-Spark, 156.7 GiB).
vllm_cpp_add_test(test_deepseek_v4_forward vllm/models/test_deepseek_v4_forward.cpp)
# DeepSeek-V4-Flash W2b — the `deepseek4` GGUF keep-quant TOWER materialization:
# tiny synthetic GGUF (real blk.N.* names + Q8_0 keep-quant) -> accounting +
# keep-quant residency + load->Forward end-to-end, RED-first (missing/leftover).
vllm_cpp_add_test(test_deepseek_v4_gguf_load vllm/models/test_deepseek_v4_gguf_load.cpp)
target_include_directories(test_deepseek_v4_gguf_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm)
# DeepSeek-V4-Flash MTP W1 — the native nextn self-speculative draft head: tiny
# synthetic nextn forward (nvidia/mtp.py:128-258) + RED-first miswires + the
# LOSSLESS gate (DS4 draft + DS4 target verified by the shared RejectionSampler ==
# pure target greedy => MTP-on == MTP-off). See .agents/specs/deepseek-v4-mtp.md.
vllm_cpp_add_test(test_deepseek_v4_mtp vllm/models/test_deepseek_v4_mtp.cpp)
target_include_directories(test_deepseek_v4_mtp PRIVATE ${CMAKE_SOURCE_DIR}/src)
# DeepSeek-V4-Flash W7-DEVICE — the four NEW V4 op families' CUDA kernels
# (src/vt/cuda/cuda_deepseek_v4.cu, dispatched through the OpProvider seam) each
# unit-gated vs its landed HOST reference at small shape + the ForwardDevice
# composition gate (device forward == host forward, near-tie). The device cases
# SKIP on a CPU-only build; on the DGX GB10 they are the RUNTIME evidence the
# kernels match the references. NOT a real-checkpoint token gate (W8, multi-Spark).
vllm_cpp_add_test(test_cuda_deepseek_v4 vllm/models/test_cuda_deepseek_v4.cpp)
# KIMI-K3 W2/W5 SCAFFOLDING (KimiK3ForConditionalGeneration) — the DERIVE-AND-SHIP
# structural gate: registry-resolve + nested-config descent + text-backbone
# enumeration + REFUSE-by-name forward + MXFP4-refuse loader. No checkpoint / GPU;
# K3 is beyond the pinned oracle (.agents/specs/kimi-k3.md).
vllm_cpp_add_test(test_kimi_k3_scaffold vllm/models/test_kimi_k3_scaffold.cpp)
vllm_cpp_add_test(test_muse_glimmer_scaffold vllm/models/test_muse_glimmer_scaffold.cpp)
vllm_cpp_add_test(test_muse_glimmer_vision vllm/models/test_muse_glimmer_vision.cpp)
target_include_directories(test_muse_glimmer_vision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# Muse Glimmer W1 TEXT TOWER gate: the whole text forward vs an INDEPENDENT fp32
# reference transcribed from vllm#51655 head 075d645af, plus a property test per
# mechanism (weightless embed_norm, QK-norm, query pre-scale, the iRoPE NoPE/sliding
# split, split pre/post eps, the offset-free final norm, the attention output gate,
# the output multiplier and the soft-cap). CPU-only, no checkpoint. No speed axis
# exists for this model — the pinned oracle cannot load it (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_text vllm/models/test_muse_glimmer_text.cpp)
# The DFlash drafter's RELEASED config.json, committed verbatim. It is the only
# real artifact that OMITS qk_scale_factor / post_norm_eps / output_multiplier /
# final_logit_softcapping, so it is what pins their defaults to the architecture's
# constants rather than to neutral values (issue #412). Committed, not read off
# the NAS, so the gate is asset-free.
target_compile_definitions(test_muse_glimmer_text PRIVATE
  MUSE_GLIMMER_DRAFT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/muse_glimmer_30b_assistant")
# ...and the SAME gate on the HAND-CALL arm. `FusedChainAdoptEnabled()` defaults ON
# and is read once per process into a function-local static, so the three
# `else vt::RmsNorm(...)` fallbacks in muse_glimmer.cpp (input layernorm,
# pre-feedforward layernorm, final norm) are UNREACHABLE in a default-env run —
# mutating one of them left every gate green — and a same-process env flip cannot
# reach them either. A SECOND ctest registration of the same binary is the only way
# to gate that arm, and it is the arm that runs wherever the fusion recipes are
# unavailable, so both must be green. The `... the fusion arm the environment
# selected` case inside the binary is what proves this registration really took the
# other branch instead of silently repeating the first.
add_test(NAME test_muse_glimmer_text_fallback COMMAND test_muse_glimmer_text)
set_tests_properties(test_muse_glimmer_text_fallback PROPERTIES
  ENVIRONMENT "VT_FUSED_CHAIN_ADOPT=0")

# Muse Glimmer W4 WIRING gate: the REAL-CHECKPOINT structural contract (every name
# EnumerateMuseGlimmerTensors produces exists in the released 1436-tensor
# Muse-Glimmer-30B index at the right shape, and nothing in the checkpoint is
# unaccounted) plus the perception-encoder wiring (vision load with the q|k|v merge
# order and every bias, the soft-token projection, the masked scatter onto the
# image/video rows, and the mm forward branch). The structural half runs off a
# COMMITTED header-only fixture so CI never needs the NAS; a second case re-reads
# the live checkpoint when VLLM_MUSE_CKPT points at it. CPU-only, no GPU, no tensor
# bytes. NOT an image e2e correctness gate — the pinned oracle cannot load this
# model at all, so no speed or token-exactness claim exists (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_wiring vllm/models/test_muse_glimmer_wiring.cpp)
target_compile_definitions(test_muse_glimmer_wiring PRIVATE
  MUSE_GLIMMER_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/muse_glimmer_30b")
# Muse Glimmer REAL-WEIGHT gate. Loads the released 30B bf16 checkpoint (or a
# reduced-depth model assembled from its REAL tensors) through the real loader,
# forwards it on CPU, and compares against a standalone torch transcription of
# vllm#51655 head 075d645af produced by scripts/mm/muse_glimmer_text_ref.py.
# Every case SKIPs unless its env var names a fixture directory, so CI never
# depends on the 59.55 GB NAS asset. This is NOT a token-exact gate against the
# model's own runtime — no such runtime exists on this box (released
# transformers does not register `muse_glimmer` and the pin has no Muse code) —
# and it makes no speed claim whatsoever (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_real_weights
  vllm/models/test_muse_glimmer_real_weights.cpp)
target_compile_definitions(test_muse_glimmer_real_weights PRIVATE
  VLLM_MUSE_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models")

# Muse Glimmer GGUF K-QUANT gate (.agents/porting-a-model.md §2). The quantized
# arm the port was missing: config-from-GGUF, the canonical<->GGUF name map, the
# three convert-time transforms this loader inverts (the +1 sandwich-norm
# pre-offset, the query pre-scale folded into attn_q_norm, the iRoPE mask on
# sliding_window_pattern), and STRUCTURAL ACCOUNTING in both directions against
# COMMITTED header-only manifests of the real 731/809/58-tensor files, so CI
# needs no asset. A live case runs when VLLM_MUSE_GGUF names the 16.76 GB
# k-quant. No forward, no e2e, and NO SPEED AXIS: the pinned oracle cannot load
# `muse_glimmer` in any weight format (specs/muse-glimmer.md §0).
vllm_cpp_add_test(test_muse_glimmer_gguf vllm/models/test_muse_glimmer_gguf.cpp)
target_include_directories(test_muse_glimmer_gguf PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# Kimi-Linear-48B-A3B (KimiLinearForCausalLM) W1 scaffolding gate: registry-resolve
# + config parse (20 KDA + 7 NoPE-MLA schedule, MLA/MoE/KDA dims, reject cases) +
# checkpoint name-map faithfulness (block_sparse_moe, verified vs the HF index) +
# loader round-trip on a synthetic 2-layer checkpoint (throws BY NAME on a missing
# or mis-shaped tensor). CPU-only; the forward REFUSES-by-name (W3-W6).
# .agents/specs/kimi-linear.md.
vllm_cpp_add_test(test_kimi_linear_scaffold vllm/models/test_kimi_linear_scaffold.cpp)

# Nemotron-H (NemotronHForCausalLM) W3 STRUCTURAL gate (#517,
# .agents/specs/nemotron-h-model.md §4 W3): registry-resolve + config descent off
# the REAL released config.json (52-entry layers_block_type schedule, the legacy
# hybrid_override_pattern/mamba_* aliases, the refuse cases) + the on-disk name map
# accounted BOTH WAYS against all 18487 tensors of the released index + the
# heterogeneous KV topology (6-layer full-attention group + 23-layer Mamba2 group,
# shapes from mamba2_state_shape). The structural half runs off a COMMITTED
# headers-only fixture so CI never needs the 20.1 GiB NAS asset; a final case
# re-verifies that fixture against the LIVE checkpoint when VT_NEMOTRON35_SNAPSHOT
# names it, and SKIPs loudly otherwise. CPU-only, no forward (W4), no speed claim.
vllm_cpp_add_test(test_nemotron_h_scaffold vllm/models/test_nemotron_h_scaffold.cpp)
target_compile_definitions(test_nemotron_h_scaffold PRIVATE
  NEMOTRON_H_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/nemotron_h_35_lightning")
# GATE-PIN-UNPINNED-SNAPSHOTS (#471): the live re-verification resolves the
# checkpoint through the pinned parity::HfSnapshot, so it needs tests/parity on
# the include path.
# nemotron_h.h is a MODEL-PRIVATE header under src/, not include/vllm/: W3 ships
# nothing on the public ABI and include/vllm/ is a USER_USAGE_PREFIXES surface
# (#515). Same call as the W1 resolver header.
target_include_directories(test_nemotron_h_scaffold PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Nemotron-H W4 FORWARD gate (#517, .agents/specs/nemotron-h-model.md §4 W4): the
# hybrid layer loop over all three block kinds, every block compared ELEMENTWISE
# against a reference written independently in `double` from the upstream
# formulas — not against a second call of the code under test, and not through a
# helper both arms share. Both dtype arms (f32 and the released bf16) on every
# case; a LONG-prompt arm as well as a short one, because only 6 of 52 layers are
# attention; the SSD chunk-size invariance; the SSM state dtype carried across
# two legs, which is the only way it is observable at all; the routed scale on
# the OUTPUT rather than the router weights or logits; the f32 router; and the
# no-RoPE finding. CPU-only, no checkpoint, no speed claim.
vllm_cpp_add_test(test_nemotron_h_forward vllm/models/test_nemotron_h_forward.cpp)
# nemotron_h.h / nemotron_h_forward.h are MODEL-PRIVATE headers under src/, not
# include/vllm/: W4 ships nothing on the public ABI and include/vllm/ is a
# USER_USAGE_PREFIXES surface (#515). Same call as the W1 resolver header.
target_include_directories(test_nemotron_h_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Kimi-Linear CPU REFERENCE FORWARD gates (W2-W6): the per-op reference forwards
# (KDA layer vs the kimi_kda host-ref composition; NoPE-MLA vs a materialized-MHA
# reference; sigmoid noaux_tc router + block vs a hand-computed case) and the whole
# 2-layer forward (loader materializes host weights, finite/coherent, greedy-decodes
# N tokens with the context advancing). The DEVICE forward stays refuse-by-name
# (born-on-runner W6/W7). .agents/specs/kimi-linear.md §5.
vllm_cpp_add_test(test_kimi_linear_forward vllm/models/test_kimi_linear_forward.cpp)
vllm_cpp_add_test(test_kimi_linear_paged vllm/models/test_kimi_linear_paged.cpp)
vllm_cpp_add_test(test_kimi_linear_fold_gate vllm/models/test_kimi_linear_fold_gate.cpp)

# KDA (Kimi Delta Attention) net-new gated-linear-attention primitives — host
# reference + unit gate (KERNEL-KDA-DELTA). Shared unblocker for Kimi-Linear-48B
# and Kimi-K3 (W4). Hand-derived literal cases + double-precision references over
# the per-channel [H,D] low-rank decay, the sigmoid-gated output norm, the 3 q/k/v
# short convs and q/k L2-norm. Full-model gate is the Kimi-Linear-48B proxy
# (DGX-blocked); mirrors the DSA lane (.agents/specs/kda-kernel-delta.md).
vllm_cpp_add_test(test_kimi_kda vllm/models/test_kimi_kda.cpp)

# MLA CAMPAIGN W8 (DeepSeek-V2-Lite, DeepseekV2ForCausalLM) — the SACRED MLA
# greedy correctness gate: drives a compact prompt battery through the paged
# engine and checks STRICT token-exact agreement with the vLLM 0.25.0 oracle
# (which W0 measured DETERMINISTIC on this model at batch=1, K=5, 8/8 prompts).
# Also gates the SCHEDULER/RUNNER batch ORDER MLA requires (decodes first,
# with-context prefills leading the prefill tail) with staggered concurrent
# requests + a prefix-cache-driven with-context prefill.
# Checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot/goldens are absent.
vllm_cpp_add_test(test_deepseek_v2_paged_engine vllm/models/test_deepseek_v2_paged_engine.cpp)
target_compile_definitions(test_deepseek_v2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_deepseek_v2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GLM/DSA G1 (GLM-4.7-Flash, Glm4MoeLiteForCausalLM) — reuses the DeepSeek-V2 MLA
# stack over the SAME DeepseekV2Weights; the FIRST e2e exercise of the q_lora
# query branch AND the noaux_tc grouped router. Loader gate (host, MTP-tail skip)
# + the SACRED paged-engine greedy gate (checkpoint-gated, dgx-only).
vllm_cpp_add_test(test_glm4_moe_lite_load vllm/models/test_glm4_moe_lite_load.cpp)
vllm_cpp_add_test(test_glm4_moe_lite_paged_engine vllm/models/test_glm4_moe_lite_paged_engine.cpp)
target_compile_definitions(test_glm4_moe_lite_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_glm4_moe_lite_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen27_paged_forward vllm/models/test_qwen27_paged_forward.cpp)
target_include_directories(test_qwen27_paged_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_unicode_data vllm/test_unicode_data.cpp)
vllm_cpp_add_test(test_pretokenizer vllm/test_pretokenizer.cpp)
# Metaspace split=true + Metaspace decoder (Parakeet tokenizer shape) —
# ARCH-ONE-SURFACE ROW 1 W1(a); gates against the pre-refactor DecodeIds
# reference on the committed parakeet_e2e fixture tokenizers.
vllm_cpp_add_test(test_tokenizer_metaspace_split
  vllm/test_tokenizer_metaspace_split.cpp)
target_compile_definitions(test_tokenizer_metaspace_split PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e")
vllm_cpp_add_test(test_bpe vllm/test_bpe.cpp)
vllm_cpp_add_test(test_detokenizer vllm/test_detokenizer.cpp)
vllm_cpp_add_test(test_request vllm/test_request.cpp)
vllm_cpp_add_test(test_kv_cache_utils vllm/v1/test_kv_cache_utils.cpp)
# issue #371: the recurrent-state budget (spec widens GDN state by k+1).
vllm_cpp_add_test(test_kv_state_budget vllm/v1/core/test_kv_state_budget.cpp)
vllm_cpp_add_test(test_prefix_match_unit vllm/v1/test_prefix_match_unit.cpp)
vllm_cpp_add_test(test_kv_cache_interface vllm/v1/test_kv_cache_interface.cpp)
vllm_cpp_add_test(test_block_pool vllm/v1/test_block_pool.cpp)
vllm_cpp_add_test(test_kv_events vllm/v1/test_kv_events.cpp)
vllm_cpp_add_test(test_single_type_kv_cache_manager vllm/v1/test_single_type_kv_cache_manager.cpp)
vllm_cpp_add_test(test_kv_cache_coordinator vllm/v1/test_kv_cache_coordinator.cpp)
vllm_cpp_add_test(test_kv_cache_manager vllm/v1/test_kv_cache_manager.cpp)
vllm_cpp_add_test(test_none_hash_determinism vllm/v1/test_none_hash_determinism.cpp)
# This suite explicitly exercises configured and unconfigured hash seeding in
# child processes. Insulate its unconfigured control arm from a developer shell
# that happens to export Python's seed (the Nix dev shell exports 0).
set_tests_properties(test_none_hash_determinism PROPERTIES
  ENVIRONMENT_MODIFICATION "PYTHONHASHSEED=unset:")
vllm_cpp_add_test(test_prefix_cache_stats vllm/v1/test_prefix_cache_stats.cpp)
vllm_cpp_add_test(test_prometheus_metrics vllm/v1/test_prometheus_metrics.cpp)
vllm_cpp_add_test(test_kv_offload_cpu vllm/v1/test_kv_offload_cpu.cpp)
vllm_cpp_add_test(test_kv_offload_fs vllm/v1/test_kv_offload_fs.cpp)
vllm_cpp_add_test(test_kv_offload_tiering vllm/v1/test_kv_offload_tiering.cpp)
vllm_cpp_add_test(test_kv_offload_connector vllm/v1/test_kv_offload_connector.cpp)
# LMCache MODE-1 (lm://) wire codec — KV-EXTERNAL-CACHE W1. Byte/bit-exact
# against fixtures captured from the real Python codec (struct + the `blake3`
# PyPI package + numpy); see tests/fixtures/lmcache/lmcache_fixtures.json.
vllm_cpp_add_test(test_lmcache_codec vllm/v1/kv_offload/lmcache/test_lmcache_codec.cpp)
target_compile_definitions(test_lmcache_codec PRIVATE
  TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
# Multimodal M1 processor-parity gate: C++ Qwen3-VL image pipeline
# (pixel_values/grid/mm-hash/placeholder-expansion) BIT-identical to the vLLM
# 0.25.0 oracle fixture (tests/vllm/multimodal/fixtures/qwen3vl).
vllm_cpp_add_test(test_qwen3vl_processor vllm/multimodal/test_qwen3vl_processor.cpp)
target_compile_definitions(test_qwen3vl_processor PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
# Multimodal M2a vision-tower faithfulness gate: the C++ Qwen3_VisionTransformer
# forward vs the dumped vLLM 0.25.0 tower reference (patch-embed/block0/merger/
# DeepStack taps + full [196,10240] output). GPU + VLLM_QWEN3VL_WEIGHTS required,
# else skipped (scripts/mm/m2a_tower_ref_dump.py + m2a_tower_weight_dump.py).
vllm_cpp_add_test(test_qwen3vl_tower vllm/multimodal/test_qwen3vl_tower.cpp)
target_compile_definitions(test_qwen3vl_tower PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TOWER_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_tower")
# Gemma-4 G2-impl: NaFlex SigLIP2 vision-tower per-stage faithfulness gate
# (patch-embed -> encoder -> pooled -> projected vs transformers-eager refs).
# Weights via VLLM_GEMMA4_VISION_WEIGHTS (dumped, uncommitted); refs default to
# the committed golden vision_refs dir (VLLM_GEMMA4_VISION_REFS to override).
vllm_cpp_add_test(test_gemma4_vision_tower vllm/multimodal/test_gemma4_vision_tower.cpp)
target_include_directories(test_gemma4_vision_tower PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_vision_tower PRIVATE
  GEMMA4_VISION_REF_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/gemma4_e4b_image/vision_refs")
# CLAIM-GEMMA4-MM-E2E: the STRICT image->text token-exact gate THROUGH the ENGINE
# registered mm-forward (ModelRegistry::Forward). dgx-only (CUDA + cached E4B).
vllm_cpp_add_test(test_gemma4_registry_e2e vllm/multimodal/test_gemma4_registry_e2e.cpp)
target_include_directories(test_gemma4_registry_e2e PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_registry_e2e PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
# Gemma-4 G3: USM-Conformer audio-tower per-stage faithfulness gate (subsample ->
# rel-pos-enc -> N conformer layers -> output_proj -> projected vs the dumped
# transformers-eager refs). Host f32, no GPU. Weights via VLLM_GEMMA4_AUDIO_WEIGHTS
# (dumped, uncommitted); refs from the committed golden audio_refs dir.
vllm_cpp_add_test(test_gemma4_audio_tower vllm/multimodal/test_gemma4_audio_tower.cpp)
target_include_directories(test_gemma4_audio_tower PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(test_gemma4_audio_tower PRIVATE
  GEMMA4_AUDIO_REF_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/gemma4_e4b_audio/audio_refs")
# Multimodal M2b/M2c text-backbone unit gates: MRoPE get_rope_index, 3-section
# rope application (vt::RopeFromCache), DeepStack scatter, and embed-merge —
# BIT/near-exact vs the dumped vLLM 0.25.0 reference (CPU-only, no weights).
vllm_cpp_add_test(test_qwen3vl_text vllm/multimodal/test_qwen3vl_text.cpp)
target_compile_definitions(test_qwen3vl_text PRIVATE
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# Multimodal M2c STRICT e2e image->text token-exact gate on Qwen3-VL-4B: the full
# pipeline (processor -> tower -> merge -> forked MRoPE/DeepStack greedy decode)
# vs the committed vLLM 0.25.0 golden. GPU + Qwen3-VL-4B checkpoint required, else
# skipped (VLLM_QWEN3VL_CKPT or the default HF cache path).
vllm_cpp_add_test(test_qwen3vl_e2e vllm/multimodal/test_qwen3vl_e2e.cpp)
target_compile_definitions(test_qwen3vl_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# MM-ENGINE-FORWARD — the SAME STRICT image->text token-exact gate driven THROUGH
# the ENGINE registered forward (ModelRegistry::Forward → ForwardQwen3VL...), the
# engine-mm-forward closing gate. GPU + Qwen3-VL-4B checkpoint required, else skipped.
vllm_cpp_add_test(test_qwen3vl_registry_e2e vllm/multimodal/test_qwen3vl_registry_e2e.cpp)
target_compile_definitions(test_qwen3vl_registry_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  TEXT_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_text")
# M3-b — STRICT image->text token-exact on the Qwen3.6-27B GDN-hybrid gate model
# vs the committed vLLM 0.25.0 golden. GPU + the vision-inclusive bf16 checkpoint
# Qwen/Qwen3.6-27B required, else skipped (VLLM_QWEN36_CKPT or the HF cache path).
vllm_cpp_add_test(test_qwen3_5_vl_e2e vllm/multimodal/test_qwen3_5_vl_e2e.cpp)
target_compile_definitions(test_qwen3_5_vl_e2e PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  GOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3_5_27b")
# Vision-tower SPEED A-B (CLAIM-MULTIMODAL-SPEED-TOWER): baseline host-weights
# forward (per-call marshal+compute) vs the resident-weights fast forward (compute
# only), on the Qwen3.6-27B vision tower. Timing only; correctness is proven by
# test_qwen3_5_vl_e2e (the host overload now delegates to the resident forward).
vllm_cpp_add_test(bench_qwen3_5_vl_tower vllm/multimodal/bench_qwen3_5_vl_tower.cpp)
target_compile_definitions(bench_qwen3_5_vl_tower PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
# Multimodal M3c VIDEO-processor UNIT gate: C++ Qwen3-VL video pipeline
# (pixel_values_videos/grid/timestamps/timestamp-interleaved expansion) BIT-identical
# to the vLLM 0.25.0 oracle fixture (tests/vllm/multimodal/fixtures/qwen3vl_video).
vllm_cpp_add_test(test_qwen3vl_video_processor vllm/multimodal/test_qwen3vl_video_processor.cpp)
target_compile_definitions(test_qwen3vl_video_processor PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_video")
# Multimodal M3c STRICT e2e VIDEO->text token-exact gate on Qwen3-VL-4B: the full
# pipeline (video processor -> tower per-frame windowed attn -> merge -> forked
# video MRoPE/DeepStack greedy decode) vs the committed vLLM 0.25.0 golden. GPU +
# Qwen3-VL-4B checkpoint required, else skipped (VLLM_QWEN3VL_CKPT or HF cache).
vllm_cpp_add_test(test_qwen3vl_video_e2e vllm/multimodal/test_qwen3vl_video_e2e.cpp)
target_compile_definitions(test_qwen3vl_video_e2e PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl_video")
# Audio-track A1 audio-processor parity gate: the C++ Whisper-class audio INPUT
# pipeline (WAV decode -> log-mel input_features -> placeholder expansion -> mm-hash)
# vs the vLLM 0.25.0 / transformers 5.13.1 oracle fixture for openai/whisper-small
# (tests/vllm/multimodal/fixtures/whisper_audio; scripts/mm/a0_audio_ref.py).
# Log-mel rel-L2 within a stated tolerance (FFT vs DFT summation order); ids + mm-hash
# bit/byte-exact. RED-first: perturbed mel/hop/normalization blow the tolerance.
vllm_cpp_add_test(test_audio_processor vllm/multimodal/test_audio_processor.cpp)
target_compile_definitions(test_audio_processor PRIVATE
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio")
# P4 (MODEL-AUDIO-PARAKEET-ENCODER) — the Parakeet log-mel FRONT END. Gates the
# mel filter bank against a dump of transformers `audio_utils.mel_filter_bank`
# (the exact function vLLM's ParakeetExtractor calls) AND against librosa's
# float32 bank (what HF's own ParakeetFeatureExtractor uses), then the full
# `input_features` against a dump of ParakeetFeatureExtractor.__call__ on a fixed
# waveform. Fixture COMMITTED (228 KiB), so it runs in CI with no download
# (scripts/mm/p4_parakeet_extractor_ref.py). CPU-only.
vllm_cpp_add_test(test_parakeet_audio_processor
  vllm/multimodal/test_parakeet_audio_processor.cpp)
target_compile_definitions(test_parakeet_audio_processor PRIVATE
  MM_PARAKEET_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/parakeet_audio")
# P4 — the ParakeetEncoder/ParakeetForCTC MODULE gate: the forward vs an
# INDEPENDENT in-test scalar reference written from the transformers definition
# (and performing `_rel_shift` LITERALLY, so agreement is evidence for
# vt::AttentionRelPos's closed form). No fixture, no checkpoint, CPU-only.
vllm_cpp_add_test(test_parakeet_encoder vllm/models/test_parakeet_encoder.cpp)
# P4 — the e2e gate: the whole encoder + CTC head + greedy collapse vs a dumped
# HuggingFace `ParakeetForCTC` ORACLE (transformers 5.3.0 — the module vLLM
# itself runs). Stage-by-stage rel-L2 plus EXACT greedy and collapsed token ids.
# The oracle is a small SEEDED, RANDOMLY-INITIALISED model, not a pretrained
# checkpoint, so the gate is on logits and ids and never on a transcript; the
# pretrained-checkpoint arm is the separate VLLM_PARAKEET_CKPT case, skipped
# when unset. Fixture COMMITTED (464 KiB, scripts/mm/p4_parakeet_oracle_dump.py).
vllm_cpp_add_test(test_parakeet_ctc_engine vllm/models/test_parakeet_ctc_engine.cpp)
target_compile_definitions(test_parakeet_ctc_engine PRIVATE
  PARAKEET_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet")

# ARCH-ONE-SURFACE ROW 1 fold gate: the library transcription seam
# (ParakeetTranscriber) byte-identical to the PRE-refactor example pipeline
# AND to the committed goldens captured from the pre-refactor binary; plus the
# registry refuse-by-task contract. Fixture committed
# (scripts/mm/parakeet_e2e_fixture_gen.py).
vllm_cpp_add_test(test_parakeet_transcription_fold
  vllm/models/test_parakeet_transcription_fold.cpp)
target_compile_definitions(test_parakeet_transcription_fold PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e")
# P5: the RNN-T / TDT TRANSDUCER gate: the LSTM prediction network, the joint
# (with the TDT duration head) and the greedy transducer loop vs a dumped
# HuggingFace `ParakeetForRNNT` / `ParakeetForTDT` ORACLE from transformers
# `main`: which DOES implement them, contrary to what the P4 spike recorded off
# the locally installed 5.3.0. Decoder and joint tensors by rel-L2; the emitted
# sequence and the per-step durations EXACT. Also gates the LSTM cell against an
# independent in-test scalar reference. The oracle is a small SEEDED,
# RANDOMLY-INITIALISED model, so nothing is claimed about a transcript; the
# pretrained arm (VLLM_PARAKEET_TRANSDUCER_CKPT + _REF) gates ids against a real
# `generate()` run and is skipped when unset. Fixture COMMITTED (1.1 MiB,
# scripts/mm/p5_parakeet_transducer_oracle_dump.py). CPU-only.
vllm_cpp_add_test(test_parakeet_transducer vllm/models/test_parakeet_transducer.cpp)
target_compile_definitions(test_parakeet_transducer PRIVATE
  PARAKEET_TRANSDUCER_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_transducer")
# M3d — NEAR-TIE-ROBUST e2e VIDEO->text token gate on the Qwen3.6-27B GDN-hybrid
# gate model: the full pipeline (video processor -> tower per-frame windowed attn,
# 27B vision config -> merge -> forked GDN-hybrid video MRoPE greedy decode) vs the
# committed vLLM 0.25.0 golden. GPU + Qwen/Qwen3.6-27B checkpoint required, else
# skipped (VLLM_QWEN36_CKPT or the HF cache path).
vllm_cpp_add_test(test_qwen3_5_vl_video_e2e vllm/multimodal/test_qwen3_5_vl_video_e2e.cpp)
target_compile_definitions(test_qwen3_5_vl_video_e2e PRIVATE
  MM_VIDEO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3_5_27b_video")
# Audio-track A2 Whisper encoder-tower faithfulness gate: the C++
# WhisperAudioEncoderForward (2x Conv1d im2col+MatmulBT + sinusoidal pos + 12
# pre-norm bidirectional encoder blocks + final layer_norm) vs the dumped
# transformers-5.13.1 WhisperEncoder reference (per-stage post_conv/post_pos/
# block0/final_ln within the bf16-depth envelope). Needs a CUDA backend + the
# dumped encoder.* weights (VLLM_WHISPER_ENC_WEIGHTS); else skipped, not failed
# (scripts/mm/a2_audio_encoder_{ref,weight}_dump.py; embed_positions committed).
vllm_cpp_add_test(test_whisper_audio vllm/multimodal/test_whisper_audio.cpp)
target_compile_definitions(test_whisper_audio PRIVATE
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio")
# A3 — e2e AUDIO->TEXT on Voxtral-Mini-3B. GPU-gated: runs only with CUDA + the
# downloaded consolidated.safetensors (VLLM_VOXTRAL_SAFETENSORS); else skipped, not
# failed. Fixtures (WAV, log-mel/mel/sinusoid goldens, prompt+golden tokens) are
# committed (scripts/mm/a3_voxtral_oracle_capture.py).
vllm_cpp_add_test(test_voxtral_e2e vllm/multimodal/test_voxtral_e2e.cpp)
target_compile_definitions(test_voxtral_e2e PRIVATE
  MM_VOXTRAL_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/voxtral_audio")
# Multimodal M1 encoder-cache manager (inert-when-off engine seam).
vllm_cpp_add_test(test_encoder_cache_manager vllm/v1/core/test_encoder_cache_manager.cpp)
# LMCache MODE-1 (lm://) blocking TCP client — KV-EXTERNAL-CACHE W2. Round-trip
# vs an in-process C++ mock server (always), and vs a REAL lmcache.v1.server
# when VT_LMCACHE_LIVE_HOST/PORT are set (scripts/lmcache/run_live_roundtrip.sh).
vllm_cpp_add_test(test_lmcache_client vllm/v1/kv_offload/lmcache/test_lmcache_client.cpp)
# LMCache lm:// KVConnector subclass over the W5 seam — KV-EXTERNAL-CACHE W3.
# Connector-level round-trip (store->lookup->shortcut->load-identical) through
# the REAL scheduler vs an in-process mock server (always), foreign-key refusal,
# default-off inertness; the live gate vs a real server under VT_LMCACHE_LIVE_*.
vllm_cpp_add_test(test_lmcache_connector vllm/v1/kv_offload/lmcache/test_lmcache_connector.cpp)
# LMCache lm:// KEY AGREEMENT vs a REAL Python vLLM+LMCache peer —
# KV-EXTERNAL-CACHE W4 crux gate. Byte-identical CacheEngineKey strings at
# chunk_size 256 with vLLM's own sha256_cbor hash; fixtures dumped from the real
# lmcache ChunkedTokenDatabase by scripts/lmcache/gen_key_agreement_fixtures.py.
vllm_cpp_add_test(test_lmcache_key_agreement vllm/v1/kv_offload/lmcache/test_lmcache_key_agreement.cpp)
target_compile_definitions(test_lmcache_key_agreement PRIVATE
  TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
vllm_cpp_add_test(test_scheduler_config vllm/test_scheduler_config.cpp)
# vllm#49343 eagle-draft max_position_embeddings clamp (config/speculative.h).
vllm_cpp_add_test(test_speculative_draft_max_position_embeddings
  vllm/config/test_speculative_draft_max_position_embeddings.cpp)
# SPEC-DSPARK W1: the DSpark config slice (method acceptance, block-size floor,
# lookahead slots, draft identity).
vllm_cpp_add_test(test_speculative_dspark vllm/config/test_speculative_dspark.cpp)
# ENG-MM-INPUT-PIPELINE L1 (#607): get_limit_per_prompt precedence —
# language_model_only ⇒ 0 BEFORE the map, else the map, else the 999 default.
vllm_cpp_add_test(test_multimodal_config vllm/config/test_multimodal_config.cpp)
# ENG-MM-INPUT-PIPELINE L1 (#607): the refusal those limits carry — the min()
# fold against the model's own ceiling, upstream's exact message, and both call
# sites with the enable_mm_embeds escape.
vllm_cpp_add_test(test_processing_limits vllm/multimodal/test_processing_limits.cpp)
vllm_cpp_add_test(test_request_queue vllm/v1/test_request_queue.cpp)
vllm_cpp_add_test(test_sched_output vllm/v1/test_sched_output.cpp)
vllm_cpp_add_test(test_scheduler vllm/v1/test_scheduler.cpp)
vllm_cpp_add_test(test_scheduler_lpm vllm/v1/test_scheduler_lpm.cpp)
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
vllm_cpp_add_test(test_attn_backend_registry vllm/v1/attention/test_attn_backend_registry.cpp)
vllm_cpp_add_test(test_attention_window vllm/model_executor/layers/attention/test_attention.cpp)
vllm_cpp_add_test(test_chunked_local_attention
  vllm/v1/attention/test_chunked_local_attention.cpp)
vllm_cpp_add_test(test_rotary_embedding
  vllm/model_executor/layers/rotary_embedding/test_rotary_embedding.cpp)
vllm_cpp_add_test(test_pooler
  vllm/model_executor/layers/pooler/test_pooler.cpp)
vllm_cpp_add_test(test_pooler_heads
  vllm/model_executor/layers/pooler/test_pooler_heads.cpp)
vllm_cpp_add_test(test_pooling_runner
  vllm/v1/worker/gpu/pool/test_pooling_runner.cpp)
vllm_cpp_add_test(test_gdn_metadata_builder vllm/v1/attention/test_gdn_metadata_builder.cpp)
vllm_cpp_add_test(test_mtp_speculator vllm/v1/spec_decode/test_mtp_speculator.cpp)
vllm_cpp_add_test(test_rejection_sampler vllm/v1/spec_decode/test_rejection_sampler.cpp)
vllm_cpp_add_test(test_prepare_prefill_inputs vllm/v1/spec_decode/test_prepare_prefill_inputs.cpp)
vllm_cpp_add_test(test_dflash_kvprep vllm/v1/spec_decode/test_dflash_kvprep.cpp)
target_include_directories(test_dflash_kvprep PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_dflash_propose vllm/v1/spec_decode/test_dflash_propose.cpp)
target_include_directories(test_dflash_propose PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-NGRAM (ROAD-V1-D3) — the draft-free n-gram matcher unit gate (ports
# vllm/tests/v1/spec_decode/test_ngram.py, host-side, runs everywhere).
vllm_cpp_add_test(test_ngram_proposer vllm/v1/spec_decode/test_ngram_proposer.cpp)
target_include_directories(test_ngram_proposer PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DRAFT-MODEL (CLAIM-SPEC-DRAFT-MEDUSA) — the generic separate-draft-model
# greedy propose brick; host-side, reuses the landed RejectionSampler verify.
vllm_cpp_add_test(test_draft_model_proposer
  vllm/v1/spec_decode/test_draft_model_proposer.cpp)
target_include_directories(test_draft_model_proposer PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_structured_output vllm/v1/structured_output/test_structured_output.cpp)
vllm_cpp_add_test(test_apply_grammar_bitmask vllm/v1/structured_output/test_apply_grammar_bitmask.cpp)
vllm_cpp_add_test(test_backend_native vllm/v1/structured_output/test_backend_native.cpp)
vllm_cpp_add_test(test_backend_xgrammar vllm/v1/structured_output/test_backend_xgrammar.cpp)
vllm_cpp_add_test(test_jump_forward vllm/v1/structured_output/test_jump_forward.cpp)
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
vllm_cpp_add_test(test_beam_search vllm/entrypoints/test_beam_search.cpp)
# ROAD-V1-MM serving W1: OpenAI multimodal content-part parse + decode + route
# to the existing single-sequence mm processors (CPU, no model weights). Reuses
# the M1 image (qwen3vl) + A1 audio (whisper_audio) processor-parity fixtures.
vllm_cpp_add_test(test_chat_mm vllm/entrypoints/openai/test_chat_mm.cpp)
target_compile_definitions(test_chat_mm PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl"
  MM_AUDIO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/whisper_audio"
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
vllm_cpp_add_test(test_openai_protocol vllm/entrypoints/openai/test_protocol.cpp)
vllm_cpp_add_test(test_openai_serving vllm/entrypoints/openai/test_serving.cpp)
vllm_cpp_add_test(test_sse_keepalive vllm/entrypoints/openai/test_sse_keepalive.cpp)
vllm_cpp_add_test(test_openai_run_batch
  vllm/entrypoints/openai/test_run_batch.cpp)
vllm_cpp_add_test(test_openai_serving_chat_stream
  vllm/entrypoints/openai/test_serving_chat_stream.cpp)
vllm_cpp_add_test(test_openai_logprobs vllm/entrypoints/openai/test_logprobs.cpp)
vllm_cpp_add_test(test_openai_utf8 vllm/entrypoints/openai/test_utf8.cpp)
vllm_cpp_add_test(test_streaming_parser_engine
  vllm/parser/engine/test_streaming_parser_engine.cpp)
vllm_cpp_add_test(test_parser_engine_assembly
  vllm/parser/engine/test_parser_engine_assembly.cpp)
vllm_cpp_add_test(test_openai_tool_parsers
  vllm/entrypoints/openai/tool_parsers/test_tool_parsers.cpp)
vllm_cpp_add_test(test_tool_parser_detect
  vllm/entrypoints/openai/tool_parsers/test_detect.cpp)
vllm_cpp_add_test(test_protocol_tool_turns
  vllm/entrypoints/openai/test_protocol_tool_turns.cpp)
vllm_cpp_add_test(test_reasoning_parser_detect
  vllm/entrypoints/openai/reasoning_parsers/test_detect.cpp)
vllm_cpp_add_test(test_reasoning_think_auto
  vllm/entrypoints/openai/reasoning_parsers/test_think_auto.cpp)
vllm_cpp_add_test(test_deepseek
  vllm/entrypoints/openai/tool_parsers/test_deepseek.cpp)
vllm_cpp_add_test(test_deepseek_v32
  vllm/entrypoints/openai/tool_parsers/test_deepseek_v32.cpp)
vllm_cpp_add_test(test_deepseek_v4
  vllm/entrypoints/openai/tool_parsers/test_deepseek_v4.cpp)
vllm_cpp_add_test(test_longcat
  vllm/entrypoints/openai/tool_parsers/test_longcat.cpp)
vllm_cpp_add_test(test_kimi_k2
  vllm/entrypoints/openai/tool_parsers/test_kimi_k2.cpp)
vllm_cpp_add_test(test_tool_parser_inkling
  vllm/entrypoints/openai/tool_parsers/test_inkling.cpp)
vllm_cpp_add_test(test_granite_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite_tool_parser.cpp)
vllm_cpp_add_test(test_granite4_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite4_tool_parser.cpp)
vllm_cpp_add_test(test_granite_20b_fc_tool_parser
  vllm/entrypoints/openai/tool_parsers/test_granite_20b_fc_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parsers_llama
  vllm/entrypoints/openai/tool_parsers/test_llama.cpp)
vllm_cpp_add_test(test_tool_parsers_pythonic
  vllm/entrypoints/openai/tool_parsers/test_pythonic.cpp)
vllm_cpp_add_test(test_tool_parsers_llama4_pythonic
  vllm/entrypoints/openai/tool_parsers/test_llama4_pythonic.cpp)
vllm_cpp_add_test(test_tool_choice_grammar
  vllm/entrypoints/openai/tool_parsers/test_tool_choice_grammar.cpp)
vllm_cpp_add_test(test_structural_tags
  vllm/entrypoints/openai/tool_parsers/test_structural_tags.cpp)
vllm_cpp_add_test(test_tool_parser_mistral
  vllm/entrypoints/openai/tool_parsers/test_mistral.cpp)
vllm_cpp_add_test(test_reasoning_base_thinking
  vllm/entrypoints/openai/reasoning_parsers/test_base_thinking.cpp)
vllm_cpp_add_test(test_reasoning_deepseek_r1
  vllm/entrypoints/openai/reasoning_parsers/test_deepseek_r1.cpp)
vllm_cpp_add_test(test_reasoning_mistral
  vllm/entrypoints/openai/reasoning_parsers/test_mistral.cpp)
vllm_cpp_add_test(test_reasoning_minimax_m2
  vllm/entrypoints/openai/reasoning_parsers/test_minimax_m2.cpp)
vllm_cpp_add_test(test_reasoning_step3
  vllm/entrypoints/openai/reasoning_parsers/test_step3.cpp)
vllm_cpp_add_test(test_reasoning_olmo3
  vllm/entrypoints/openai/reasoning_parsers/test_olmo3.cpp)
vllm_cpp_add_test(test_reasoning_deepseek_v3
  vllm/entrypoints/openai/reasoning_parsers/test_deepseek_v3.cpp)
vllm_cpp_add_test(test_reasoning_qwen3
  vllm/entrypoints/openai/reasoning_parsers/test_qwen3.cpp)
vllm_cpp_add_test(test_tool_parser_xlam
  vllm/entrypoints/openai/tool_parsers/test_xlam_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_phi4mini
  vllm/entrypoints/openai/tool_parsers/test_phi4mini_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_internlm
  vllm/entrypoints/openai/tool_parsers/test_internlm2_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_jamba
  vllm/entrypoints/openai/tool_parsers/test_jamba_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_step3
  vllm/entrypoints/openai/tool_parsers/test_step3.cpp)
vllm_cpp_add_test(test_tool_parser_step3p5
  vllm/entrypoints/openai/tool_parsers/test_step3p5.cpp)
vllm_cpp_add_test(test_tool_parser_qwen3_coder
  vllm/entrypoints/openai/tool_parsers/test_qwen3_coder.cpp)
vllm_cpp_add_test(test_tool_parser_minicpm5
  vllm/entrypoints/openai/tool_parsers/test_minicpm5.cpp)
vllm_cpp_add_test(test_tool_parser_hy_v3
  vllm/entrypoints/openai/tool_parsers/test_hy_v3.cpp)
vllm_cpp_add_test(test_tool_parser_hunyuan_a13b
  vllm/entrypoints/openai/tool_parsers/test_hunyuan_a13b_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_apertus
  vllm/entrypoints/openai/tool_parsers/test_apertus_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_ernie45
  vllm/entrypoints/openai/tool_parsers/test_ernie45_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_gigachat3
  vllm/entrypoints/openai/tool_parsers/test_gigachat3_tool_parser.cpp)
vllm_cpp_add_test(test_tool_parser_olmo3
  vllm/entrypoints/openai/tool_parsers/test_olmo3.cpp)
vllm_cpp_add_test(test_tool_parser_lfm2
  vllm/entrypoints/openai/tool_parsers/test_lfm2.cpp)
vllm_cpp_add_test(test_tool_parser_poolside_v1
  vllm/entrypoints/openai/tool_parsers/test_poolside_v1.cpp)
vllm_cpp_add_test(test_tool_parser_functiongemma
  vllm/entrypoints/openai/tool_parsers/test_functiongemma.cpp)
vllm_cpp_add_test(test_glm47
  vllm/entrypoints/openai/tool_parsers/test_glm47.cpp)
vllm_cpp_add_test(test_minimax_m2_tool
  vllm/entrypoints/openai/tool_parsers/test_minimax_m2_tool.cpp)
vllm_cpp_add_test(test_tool_parser_gemma4
  vllm/entrypoints/openai/tool_parsers/test_gemma4.cpp)
vllm_cpp_add_test(test_tool_parser_seed_oss
  vllm/entrypoints/openai/tool_parsers/test_seed_oss.cpp)
vllm_cpp_add_test(test_tool_parser_muse_glimmer
  vllm/entrypoints/openai/tool_parsers/test_muse_glimmer_tool_parser.cpp)
vllm_cpp_add_test(test_reasoning_muse_glimmer
  vllm/entrypoints/openai/reasoning_parsers/test_muse_glimmer_reasoning_parser.cpp)
vllm_cpp_add_test(test_chat_template vllm/entrypoints/test_chat_template.cpp)
# The real Qwen3.5 chat template (extracted from the GGUF) lives under
# tests/fixtures/; hand its absolute path to the test so it resolves regardless
# of the working directory ctest runs it from.
target_compile_definitions(test_chat_template PRIVATE
  VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
# The HTTP api_server smoke test needs the vendored cpp-httplib (gated).
if(VLLM_CPP_SERVER)
  vllm_cpp_add_test(test_openai_api_server vllm/entrypoints/openai/test_api_server.cpp)
  # /v1/audio/transcriptions dispatch + socket smoke run against the REAL
  # library transcription seam on the committed parakeet_e2e fixture
  # (ARCH-ONE-SURFACE ROW 1).
  target_compile_definitions(test_openai_api_server PRIVATE
    PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e"
    # /v1/embeddings dispatch + socket smoke run against the REAL engine path
    # (LoadedEngine -> PoolingRunner) on the committed llama_embed_e2e fixture
    # (ARCH-ONE-SURFACE ROW 6).
    LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
  target_include_directories(test_openai_api_server PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # M3.6: the OpenAI server CONFORMANCE suite — the full API contract exercised
  # end to end over the REAL cpp-httplib server on an ephemeral port.
  vllm_cpp_add_test(test_openai_conformance vllm/entrypoints/openai/test_conformance.cpp)
  # Both spin up a REAL cpp-httplib server (background accept thread) and drive it
  # with a client on a bounded connect/read timeout. They pass reliably run alone,
  # but under a saturated `ctest -jN` (the multi-minute paged-engine/bench tests
  # peg every core) the server's accept thread starves and the client sees
  # connection failures / an abort. RUN_SERIAL reproduces the standalone
  # condition — the test runs with no other ctest test concurrent — so the
  # parallel suite is deterministic. (A shared RESOURCE_LOCK would only serialize
  # these two against each other, not against the CPU-hog tests that actually
  # cause the starvation.)
  set_tests_properties(test_openai_api_server test_openai_conformance
    PROPERTIES RUN_SERIAL ON)
  # SERVE-RECIPE-ARGS (#606): the accepted-and-inert serve-argument seam. It
  # drives the REAL VllmServerMain (owned by the library behind the same
  # VLLM_CPP_SERVER gate) by re-execing this binary, so it needs src/ on the
  # include path for server_main.h, exactly as test_openai_api_server does.
  vllm_cpp_add_test(test_serve_recipe_args
    vllm/entrypoints/openai/test_serve_recipe_args.cpp)
  target_include_directories(test_serve_recipe_args PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
target_compile_definitions(test_capi PRIVATE
  PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e"
  MINIMAX_H3_VIDEO_FOLD_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/minimax_h3_video_fold"
  LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
vllm_cpp_add_test(test_chat_prompt capi/test_chat_prompt.cpp)
target_include_directories(test_chat_prompt PRIVATE ${CMAKE_SOURCE_DIR}/src)

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
add_executable(test_vulkan_loader
  vt/test_vulkan_loader.cpp
  ${CMAKE_SOURCE_DIR}/src/vt/vulkan/vulkan_loader.cpp)
target_include_directories(test_vulkan_loader PRIVATE
  ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_vulkan_loader PRIVATE vllm_test_main ${CMAKE_DL_LIBS})
vllm_cpp_set_warnings(test_vulkan_loader)
add_test(NAME test_vulkan_loader COMMAND test_vulkan_loader)
# CPU-vs-device op-equality harness (BACKEND-METAL-MLX W0). Unconditional: it
# discovers which non-CPU backends this build registered and compares only those,
# so it is meaningful on a CUDA box, on the Metal M4, and inert (CPU-only bit-
# exactness checks) on a plain CPU build.
vllm_cpp_add_test(test_backend_cross_device vt/test_backend_cross_device.cpp)
# The ROCm GCN-arch capability parse (BACKEND-ROCM W0). UNCONDITIONAL by design:
# include/vt/rocm/rocm_arch.h is deliberately free of HIP headers so the one part
# of that skeleton carrying a DECISION is gated on a CPU-only CI runner with no
# AMD GPU. The rest of the ROCm skeleton is under VLLM_CPP_HIP below.
vllm_cpp_add_test(test_rocm_arch vt/test_rocm_arch.cpp)
vllm_cpp_add_test(test_triton_aot_arch_dispatch
  vt/test_triton_aot_arch_dispatch.cpp)
target_include_directories(test_triton_aot_arch_dispatch PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cpu_isa_x86 vt/test_cpu_isa_x86.cpp)
target_include_directories(test_cpu_isa_x86 PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_cpu_isa_arm vt/test_cpu_isa_arm.cpp)
target_include_directories(test_cpu_isa_arm PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_HIP)
  # Plain C++ (no HIP header): every assertion goes through the vt:: seam. Each
  # case no-ops when the build has HIP but the box has no AMD GPU.
  vllm_cpp_add_test(test_rocm_backend vt/test_rocm_backend.cpp)
endif()
if(VLLM_CPP_METAL)
  vllm_cpp_add_test(test_metal_backend vt/test_metal_backend.cpp)
endif()
if(VLLM_CPP_VULKAN)
  vllm_cpp_add_test(test_vulkan_backend vt/test_vulkan_backend.cpp)
  # Reaches the backend-internal vulkan_context.h / vulkan_spirv.h (both under
  # src/, both plain C++ with no Vulkan headers in their interface) to gate the
  # COMMITTED SPIR-V table and the probed float-controls. Same escape hatch
  # test_model_registry already uses above.
  target_include_directories(test_vulkan_backend PRIVATE ${CMAKE_SOURCE_DIR}/src)
endif()
if(VLLM_CPP_TENSTORRENT)
  # Plain C++ (no ttnn/tt-metal header): every assertion goes through the
  # public vt:: seam, same rationale as test_vulkan_backend.cpp's own comment.
  # Each case SKIPS (not fails) when the build has the backend compiled in but
  # the box has no Blackhole card.
  vllm_cpp_add_test(test_tenstorrent_backend vt/test_tenstorrent_backend.cpp)
endif()
vllm_cpp_add_test(test_tensor vt/test_tensor.cpp)
# BACKEND-ACCEL-PROVIDER: the acceleration-provider seam (include/vt/op_provider.h)
# — deterministic selection, capability predicates, decline-and-fall-back and the
# selection stats. Device-independent (registers onto the unused kXPU slots).
vllm_cpp_add_test(test_op_provider vt/test_op_provider.cpp)
# S5 portable reference tier (accelerator-seam-audit.md): the unified-memory
# safety gate, correctness-with-zero-native-kernels via the CPU fallback, and the
# native-wins / observability contract. Hardware-free — a fake backend on the
# unused kXPU slot stands in for a unified accelerator, so it runs on every build.
vllm_cpp_add_test(test_reference_tier vt/test_reference_tier.cpp)
# Scale-out W1 (scale-out-distributed.md §W1): the vt::Communicator collective
# abstraction proven on the CPU in-process multi-rank transport — 2/4-rank
# AllReduce-sum + AllGather correctness, Send/Recv rendezvous, and the
# world_size==1 byte-identical no-op that keeps the single-GPU path untouched.
# Hardware-free (N host threads share one barrier); needs threads.
vllm_cpp_add_test(test_communicator vt/test_communicator.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test_communicator PRIVATE Threads::Threads)
# Scale-out W2 (scale-out-distributed.md §W2): the tensor-parallel CORRECTNESS
# gate with NO GPU — sharded-matmul + RowParallel all-reduce == the unsharded
# tp=1 forward, through the ACTUAL W2 helpers (TpShard/TpAllReduceSum) over the
# W1 CPU communicator; RED-first (a dropped all-reduce is the wrong partial).
vllm_cpp_add_test(test_tp_forward vt/test_tp_forward.cpp)
target_link_libraries(test_tp_forward PRIVATE Threads::Threads)
# BACKEND-DISTRIBUTED-TP TP-W1 (tensor-parallelism-spike.md §S4): the rank-layout
# group math == vLLM's arange(world).reshape(...).view(-1, tp) (an independent
# reshape oracle in-test), + tp=1 inertness + the thread-local current-rank
# accessors. Pure integer math; uses std::thread only for the thread_local
# isolation check.
vllm_cpp_add_test(test_parallel_layout vllm/distributed/test_parallel_layout.cpp)
target_link_libraries(test_parallel_layout PRIVATE Threads::Threads)
# Scale-out W2 multi-device backend registry: Device{type,index} addressability
# and the byte-neutral device-0 proof (a second backend at index 1 leaves index 0
# and the type-level API identical). Hardware-free (a fake backend on kXPU).
vllm_cpp_add_test(test_backend_multidevice vt/test_backend_multidevice.cpp)
vllm_cpp_add_test(test_arena vt/test_arena.cpp)
vllm_cpp_add_test(test_ops_matmul vt/test_ops_matmul.cpp)
# D1 (fold plan Tier D1): the merged-QKV == separate-q/k/v A/B — one MatmulBT over
# the merged [Nq+Nk+Nv,H] owner + a contiguous QkvSplit is BYTE-IDENTICAL to three
# separate MatmulBT (bit-exact GEMM math, wider N; RED-first). CPU-only.
vllm_cpp_add_test(test_ops_qkv_merge vt/test_ops_qkv_merge.cpp)
# Fold plan Tier C2: the vision-tower merged-QKV *with bias* A/B — ONE MatmulBT +
# a fused merged-[3d] BIAS epilogue + a contiguous QkvSplit is BYTE-IDENTICAL to
# three separate {MatmulBT + bias} (the NEW merged-bias piece vs D1; RED-first
# wrong-bias-slice fails). Exercises models::FusedMergedQkvBiasSplit. CPU-only.
vllm_cpp_add_test(test_ops_qkv_merged_bias vt/test_ops_qkv_merged_bias.cpp)
# QUANT-GGUF-CIQ-GEMM G1: block dtypes, the quant traits table and its
# cross-check against the GGUF reader's GgmlTraits, and the kMatmulBTQuant
# skeleton's generic-composite fallback. CPU-only.
vllm_cpp_add_test(test_ops_quant_traits vt/test_ops_quant_traits.cpp)
# QUANT-GGUF-CIQ-GEMM G2/G3: the activation quantizers (quantize_row_q8_0/q8_K)
# and the six tier-0 generic vec_dot kernels plus their kMatmulBTQuant wiring.
# Gated against an INDEPENDENT f64 dequantize-then-dot reference, upstream's
# test-quantize-fns/test-backend-ops thresholds, block-boundary/ragged-K shapes
# and run-to-run + thread-count bit-exactness. Reaches src/ for the threadpool
# header (SwapForTesting) used by the thread-count determinism case. CPU-only.
vllm_cpp_add_test(test_ops_quant_dot vt/test_ops_quant_dot.cpp)
target_include_directories(test_ops_quant_dot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# QUANT-GGUF-CIQ-GEMM G7: repack-at-load for the q8_0 quant GEMM. Proves the
# i8mm interleave transform matches make_block_q8_0x4 and round-trips (portable),
# and that the repacked gemm/gemv is BYTE-IDENTICAL to the plain quant GEMM
# across decode/leftover/prefill shapes and thread counts (i8mm boxes; skips
# coherently elsewhere). Reaches src/ for the block/repack/threadpool headers.
vllm_cpp_add_test(test_ops_quant_repack vt/test_ops_quant_repack.cpp)
target_include_directories(test_ops_quant_repack PRIVATE ${CMAKE_SOURCE_DIR}/src)
# QUANT-GGUF-CIQ-GEMM-CUDA: the CUDA keep-quant GEMM (kCUDA provider for
# kMatmulBTQuant). Gates the GPU dequant-in-kernel dot against the landed CPU
# keep-quant oracle + an f64 dequant reference on the Q8_K family (DeepSeek-V4's
# IQ2_XXS/IQ3_XXS/Q2_K plus the other four k-quants). Skips coherently with no
# CUDA device. Reaches src/ for vt/cpu block/quant headers used by the reference.
vllm_cpp_add_test(test_cuda_quant_dot vt/test_cuda_quant_dot.cpp)
target_include_directories(test_cuda_quant_dot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-ELEM-GEMM: the specialized + SIMD elementwise (bf16/f16/f32) GEMM. The
# gate is BYTE-IDENTITY against an independent scalar reference across every
# dtype combination, both weight orientations, ragged K/N, row-strided
# activations, thread counts, and the exhaustive 16-bit f16/bf16 widening
# domain. Reaches src/ for the tier header and the threadpool test hook.
vllm_cpp_add_test(test_ops_matmul_elem vt/test_ops_matmul_elem.cpp)
target_include_directories(test_ops_matmul_elem PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the env-gated cuBLASLt GEMM algo-selection diagnostic
# helpers. Reaches the internal header under src/ (not public API); the cuBLASLt
# emit it gates is CUDA-only and lives in src/vt/cuda/cuda_matmul.cu.
vllm_cpp_add_test(test_gemm_algo_log vt/test_gemm_algo_log.cpp)
target_include_directories(test_gemm_algo_log PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the cuBLASLt FP8 plan-cache pure plumbing (the
# VT_FP8_PLAN_CACHE flag + the Fp8PlanKey equality/hash completeness). Reaches the
# internal header under src/ (not public API); the cached cuBLASLt {desc, layouts,
# algo} it keys is CUDA-only and lives in src/vt/cuda/cuda_matmul.cu, with the
# byte-exact cached-vs-fresh GEMM proof in vt/test_ops_fp8_cutlass.cpp.
vllm_cpp_add_test(test_fp8_plan_cache vt/test_fp8_plan_cache.cpp)
target_include_directories(test_fp8_plan_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the register-resident packed-decode tiling A/B flag
# (VT_GDN_PACKED_REG_TILE). Reaches the internal header under src/ (not public
# API); the kernel it gates is CUDA-only and lives in src/vt/cuda/cuda_gdn.cu.
vllm_cpp_add_test(test_gdn_packed_reg_tile vt/test_gdn_packed_reg_tile.cpp)
target_include_directories(test_gdn_packed_reg_tile PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the vendored Triton AOT packed-decode fast-path flag
# (VT_GDN_PACKED_DECODE_TRITON, default ON). Reaches the internal header under
# src/ (not public API); the vendored cubin it gates is CUDA-only and lives in
# src/vt/cuda/cuda_gdn.cu (TryTritonPackedDecode -> gdn_decode_h{48,32}).
vllm_cpp_add_test(test_gdn_packed_decode_triton vt/test_gdn_packed_decode_triton.cpp)
target_include_directories(test_gdn_packed_decode_triton PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the opt-in fused-GDN decode BV16 value tile. Reaches
# the CUDA-free selector/geometry header under src/; the executing kernel lives
# in cuda_gdn.cu and is covered by same-process byte-exact CUDA comparisons.
vllm_cpp_add_test(test_gdn_decode_fused vt/test_gdn_decode_fused.cpp)
target_include_directories(test_gdn_decode_fused PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
# CPU-tier contract for the GDN PREFILL causal_conv1d forward register-window kernel
# (VT_CONV_REG) and split fused post-conv kernel (VT_GDN_POSTCONV_SPLIT) A/B flags.
# Reaches the internal header under src/ (not public API); the bit-identical fast
# kernels they gate are CUDA-only and live in src/vt/cuda/cuda_gdn.cu
# (CausalConv1dFwdRegKernel / GdnPostConvSplitKernel).
vllm_cpp_add_test(test_gdn_prefill_conv vt/test_gdn_prefill_conv.cpp)
target_include_directories(test_gdn_prefill_conv PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU-tier contract for the CUDA-graph-safe grow-only device scratch bookkeeping
# (src/vt/cuda/graph_safe_scratch.h): a grown-past scratch block is RETIRED (kept
# resident), never freed, so a device pointer baked into a captured pure-decode CUDA
# graph stays valid — the fix for the 35B c2+ online-serving illegal-memory-access.
# The device proof is the DGX serving harness; this pins the never-free guarantee.
vllm_cpp_add_test(test_graph_safe_scratch vt/test_graph_safe_scratch.cpp)
target_include_directories(test_graph_safe_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# CPU threadpool (QUANT-GGUF-CPU-THREADPOOL): barrier/park-wake/chunk-steal
# stress + the n_threads 1/3/20 determinism battery. Reaches the internal
# header under src/ (the pool is not public API).
vllm_cpp_add_test(test_cpu_threadpool vt/test_cpu_threadpool.cpp)
target_include_directories(test_cpu_threadpool PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_nvfp4_matmul vt/test_ops_nvfp4_matmul.cpp)
vllm_cpp_add_test(test_ops_nvfp4_fp4 vt/test_ops_nvfp4_fp4.cpp)
target_include_directories(test_ops_nvfp4_fp4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT WIN32)
  vllm_cpp_add_test(test_nvfp4_persistent_cache vt/test_nvfp4_persistent_cache.cpp)
  target_include_directories(test_nvfp4_persistent_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(test_nvfp4_persistent_cache PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
endif()
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
# Opt-in arm: run the fp8 plan-cache byte-exact case with the cache ENABLED
# (VT_FP8_PLAN_CACHE=1 -> first MatmulFp8CublasLt call builds the plan fresh,
# later calls hit the cache). Proves the cached-plan GEMM is BYTE-identical to the
# freshly-built one. The DEFAULT (cache OFF, every call rebuilds — the shipped
# production path) runs inside the plain test_ops_fp8_cutlass.
add_test(NAME test_ops_fp8_cutlass_plan_cache_on
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM")
set_tests_properties(test_ops_fp8_cutlass_plan_cache_on PROPERTIES
  ENVIRONMENT "VT_FP8_PLAN_CACHE=1")
# Opt-in arm: PERF-FP8-ALPHA-FOLD's per-column alpha applied INSIDE the cuBLASLt
# epilogue (VT_FP8_ALPHA_VEC_EPILOGUE=1) versus the shipped two-launch form. This
# is the BINDING byte-exactness comparison. The plain test_ops_fp8_cutlass run has
# the toggle unset, where both sides execute the same fallback code and the case
# only proves that fallback is unchanged.
add_test(NAME test_ops_fp8_alpha_vec_epilogue_on
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt per-column alpha is BYTE-identical to the two-launch form")
set_tests_properties(test_ops_fp8_alpha_vec_epilogue_on PROPERTIES
  ENVIRONMENT "VT_FP8_ALPHA_VEC_EPILOGUE=1")
vllm_cpp_add_test(test_ops_moe_grouped vt/test_ops_moe_grouped.cpp)
vllm_cpp_add_test(test_ops_moe_grouped_bf16 vt/test_ops_moe_grouped_bf16.cpp)
vllm_cpp_add_test(test_ops_moe_grouped_bf16_gate_up_silu
                  vt/test_ops_moe_grouped_bf16_gate_up_silu.cpp)
vllm_cpp_add_test(test_ops_moe_nongated_relu2 vt/test_ops_moe_nongated_relu2.cpp)
vllm_cpp_add_test(test_ops_rmsnorm vt/test_ops_rmsnorm.cpp)
vllm_cpp_add_test(test_ops_fused_chain vt/test_ops_fused_chain.cpp)
vllm_cpp_add_test(test_ops_layernorm vt/test_ops_layernorm.cpp)
vllm_cpp_add_test(test_fused_chain_additivity vt/test_fused_chain_additivity.cpp)
vllm_cpp_add_test(test_ops_activation vt/test_ops_activation.cpp)
vllm_cpp_add_test(test_ops_rope vt/test_ops_rope.cpp)
vllm_cpp_add_test(test_ops_rope_cache vt/test_ops_rope_cache.cpp)
vllm_cpp_add_test(test_ops_fused_norm_rope vt/test_ops_fused_norm_rope.cpp)
vllm_cpp_add_test(test_ops_gdn vt/test_ops_gdn.cpp)
target_include_directories(test_ops_gdn PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(VLLM_CPP_TRITON)
  # A separate process guarantees this race test reaches the generated module
  # loaders before another gate-shape GDN case can warm them.
  add_test(NAME test_ops_gdn_aot_concurrent_first_load
    COMMAND test_ops_gdn
      "--test-case=CUDA gdn Triton AOT concurrent first load is safe across two queues")
endif()
vllm_cpp_add_test(test_ops_kda_recurrence vt/test_ops_kda_recurrence.cpp)
target_include_directories(test_ops_kda_recurrence PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_kda_chunk_prefill vt/test_ops_kda_chunk_prefill.cpp)
target_include_directories(test_ops_kda_chunk_prefill PRIVATE ${CMAKE_SOURCE_DIR}/src)
# Mamba2 / SSD host references (.agents/specs/mamba2-ssd.md W1, #496). Three
# sibling ops of the GDN/KDA lane above, gated against a sequential
# double-precision reference and across chunk sizes.
vllm_cpp_add_test(test_ops_mamba2_ssd vt/test_ops_mamba2_ssd.cpp)
target_include_directories(test_ops_mamba2_ssd PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_mamba2_state_update vt/test_ops_mamba2_state_update.cpp)
target_include_directories(test_ops_mamba2_state_update PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_mamba2_gated_norm vt/test_ops_mamba2_gated_norm.cpp)
target_include_directories(test_ops_mamba2_gated_norm PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_moe vt/test_ops_moe.cpp)
vllm_cpp_add_test(test_ops_glue vt/test_ops_glue.cpp)
vllm_cpp_add_test(test_ops_attn_preamble vt/test_ops_attn_preamble.cpp)
vllm_cpp_add_test(test_ops_attention vt/test_ops_attention.cpp)
# vt::AttentionDenseFa2 (multimodal-speed §17, #432): fast-path agreement with the
# byte-exact scalar kernel, full-key-range coverage, fall-through totality, and the
# VT_FA2_DENSE A/B knob. CUDA-only content; skips cleanly on a CPU box.
vllm_cpp_add_test(test_ops_attention_dense_fa2 vt/test_ops_attention_dense_fa2.cpp)
# vt::AttentionCross, DIRECTLY, on every backend that registers it. Its own
# target rather than more cases in test_ops_attention or test_ltx2_device, both
# of which carry recorded baselines. It exists because the CUDA kernel's whole
# flash-tiling machinery (cross-tile rescale, multi-block launch, npl>1 head
# split, GQA broadcast, ChooseTileCols' halving) is unreachable from every
# LTX-2.5 call geometry, which is tiles=1/npl=1/nblk=1/Hq==Hkv throughout.
vllm_cpp_add_test(test_ops_attention_cross vt/test_ops_attention_cross.cpp)
vllm_cpp_add_test(test_ops_dflash_block_attn vt/test_ops_dflash_block_attn.cpp)
vllm_cpp_add_test(test_ops_dflash_paged_block_attn vt/test_ops_dflash_paged_block_attn.cpp)
vllm_cpp_add_test(test_ops_reshape_cache vt/test_ops_reshape_cache.cpp)
# KV-FP8 W1: fp8 KV-cache store (ReshapeAndCacheFp8) + paged-attention read
# dequant + the ParseCacheDType config wiring.
vllm_cpp_add_test(test_ops_fp8_kv_cache vt/test_ops_fp8_kv_cache.cpp)
# MLA campaign W3: the compressed-latent cache write + the grouped-topk router.
vllm_cpp_add_test(test_ops_mla_cache vt/test_ops_mla_cache.cpp)
# MLA campaign W4: the two-stage split-KV MQA decode over the compressed latent.
vllm_cpp_add_test(test_ops_mla_attn vt/test_ops_mla_attn.cpp)
# MLA campaign W5: the FLASH_ATTN MLA prefill (QK 192 / V 128) ...
vllm_cpp_add_test(test_ops_mla_prefill vt/test_ops_mla_prefill.cpp)
# ... and the workspace-bounded chunked-context loop (gather + LSE merge).
vllm_cpp_add_test(test_ops_mla_chunked_context vt/test_ops_mla_chunked_context.cpp)
# MLA campaign W6: the two primitives weight ABSORPTION is built out of —
# vt::BatchedMatmul (torch.bmm) and vt::ConcatMlaNopeRope (concat_mla_q /
# _concat_k_nope_k_pe).
vllm_cpp_add_test(test_ops_mla_absorb vt/test_ops_mla_absorb.cpp)
# MLA campaign W6: the MLA attention BLOCK — projections, both q_lora branches,
# decoupled RoPE, the load-time kv_b_proj -> W_UK/W_UV split, and the
# prefill-MHA / decode-MQA dispatch, incl. the absorbed-vs-unabsorbed proof.
vllm_cpp_add_test(test_mla_attention_block
  vllm/model_executor/layers/attention/test_mla_attention_block.cpp)
vllm_cpp_add_test(test_ops_moe_router_grouped vt/test_ops_moe_router_grouped.cpp)
# Conformer / FastConformer audio-encoder kernels (spike
# .agents/specs/parakeet-conformer-encoder.md rows P1/P2/P3). Each gate is
# BYTE-IDENTITY against an INDEPENDENT in-test scalar reference (never against
# the kernel), across dtype x shape x thread count, matching the
# test_ops_matmul_elem.cpp discipline. The relpos-attention reference performs
# the upstream `_rel_shift` LITERALLY, so byte-identity is what proves the
# kernel's closed-form index. They reach src/ for the threadpool test hook
# (Threadpool::SwapForTesting) used by the determinism cases. CPU-only.
vllm_cpp_add_test(test_ops_conv2d vt/test_ops_conv2d.cpp)
target_include_directories(test_ops_conv2d PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_conv1d_depthwise vt/test_ops_conv1d_depthwise.cpp)
target_include_directories(test_ops_conv1d_depthwise PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_attn_relpos vt/test_ops_attn_relpos.cpp)
target_include_directories(test_ops_attn_relpos PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_paged_attn vt/test_ops_paged_attn.cpp)
# PERF-CPU-ATTN-DTYPE: the hoisted dtype dispatch vs the pre-change per-element
# switch, memcmp over every dtype the switch handled (spec §3, issue #391).
vllm_cpp_add_test(test_ops_paged_attn_dtype vt/test_ops_paged_attn_dtype.cpp)
vllm_cpp_add_test(test_ops_sample vt/test_ops_sample.cpp)
vllm_cpp_add_test(test_gemma4_rocm_fp8_seams vt/test_gemma4_rocm_fp8_seams.cpp)
vllm_cpp_add_test(test_ops_penalties vt/test_ops_penalties.cpp)

vllm_cpp_add_test(test_npy parity/test_npy.cpp)
target_compile_definitions(test_npy PRIVATE
  VLLM_CPP_GOLDENS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity/goldens")

vllm_cpp_add_test(test_tokenizer_parity vllm/test_tokenizer_parity.cpp)
target_compile_definitions(test_tokenizer_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# MLA CAMPAIGN W8 — the DeepSeek family's tokenizer parity gate. DeepSeek's
# pre_tokenizer is a seven-stage HF `Sequence` PIPELINE over enumerated
# codepoint ranges (not an alternation regex like every family before it), so
# it is MEASURED against HF `tokenizers` over a stage-stress corpus rather than
# argued. This is what unblocked the DeepSeek-V2-Lite paged-engine gate.
vllm_cpp_add_test(test_tokenizer_parity_deepseek vllm/test_tokenizer_parity_deepseek.cpp)
target_compile_definitions(test_tokenizer_parity_deepseek PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# LOAD-SENTENCEPIECE — the SentencePiece (Metaspace + byte-fallback) BPE
# tokenizer parity gate. Mistral-7B-v0.3's tokenizer.json is the first of this
# family (▁ metaspace + `<0xNN>` byte-fallback vocab + BOS-prepending
# TemplateProcessing). MEASURED byte-exact against HF `tokenizers` (= vLLM
# 0.25.0's backend) over a Metaspace/byte-fallback/special-token corpus. This
# is what unblocks the Mistral paged-engine SACRED gate.
vllm_cpp_add_test(test_tokenizer_parity_mistral vllm/test_tokenizer_parity_mistral.cpp)
target_compile_definitions(test_tokenizer_parity_mistral PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")

# TOK-LLAMA4-PRE (issue #347) — the GPT-4o / o200k tokenizer parity gate. This
# family is what a GGUF tags `tokenizer.ggml.pre = "llama4"` (Muse Glimmer 30B).
# It shares kLlama3's `\p{N}{1,3}` digit grouping but splits CamelCase and keeps
# the contraction as a word suffix, so an alias would mistokenize silently —
# only comparing IDS against HF `tokenizers` catches that, which is what this
# gate does.
vllm_cpp_add_test(test_tokenizer_parity_gpt4o vllm/test_tokenizer_parity_gpt4o.cpp)
target_compile_definitions(test_tokenizer_parity_gpt4o PRIVATE
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

# ROW-SERVE-ASYNC-LLM — the ASYNC-serving 35B greedy token-exact gate (the
# depth-2 AsyncLLM counterpart of the SYNC SACRED gate above). Reproduces the
# async batch-1 token-0 degeneration P0 on the mirror-OFF default and proves the
# device-resident mirror fixes it (checkpoint-gated, dgx-only; skips on CPU/CI).
vllm_cpp_add_test(test_qwen36_async_serving parity/test_qwen36_async_serving.cpp)
target_compile_definitions(test_qwen36_async_serving PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_async_serving PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W4 — the SACRED token-exact gate: Qwen3-0.6B (dense) greedy
# decode vs the vLLM 0.25.0 oracle, through the full paged engine
# (checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_paged_engine parity/test_qwen3_paged_engine.cpp)
target_compile_definitions(test_qwen3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ROW-SERVE-ASYNC-DENSE-MIRROR — the classic-dense async-serving greedy gate:
# Qwen3-0.6B/4B async batch-1 + concurrency reproduce the race-free SYNC engine
# continuation token-for-token. RED on VT_ASYNC_DEVICE_MIRROR=0 (dense EmbedInto
# races the combine), GREEN on the default fix (checkpoint-gated, dgx-only; skips
# on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_dense_async_serving parity/test_qwen3_dense_async_serving.cpp)
target_compile_definitions(test_qwen3_dense_async_serving PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dense_async_serving PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ROAD-V1-D4-APC W3 — the cache-ON automatic-prefix-caching e2e gate on Qwen3-4B
# (dense, full-attention, APC-eligible): APC-ON == APC-OFF token-exact (RED-first
# cache-bug catch) + non-zero hit counters + vLLM K-run membership + prefill/TTFT
# drop (checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot is absent).
vllm_cpp_add_test(test_qwen3_apc_e2e parity/test_qwen3_apc_e2e.cpp)
target_compile_definitions(test_qwen3_apc_e2e PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_apc_e2e PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GLM-4-9B-0414 (Glm4ForCausalLM) dense SACRED gate — first GLM-family model (G2).
vllm_cpp_add_test(test_glm4_paged_engine parity/test_glm4_paged_engine.cpp)
target_compile_definitions(test_glm4_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_glm4_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# OLMo-2-0425-1B (Olmo2ForCausalLM) dense SACRED gate — first OLMo-family model (W4).
vllm_cpp_add_test(test_olmo2_paged_engine parity/test_olmo2_paged_engine.cpp)
target_compile_definitions(test_olmo2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_olmo2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Gemma-4 E4B (Gemma4ForConditionalGeneration) text STRICT gate — first Gemma-4
# model; RED->GREEN proof of the G1b runner per-layer-KV head_dim change.
vllm_cpp_add_test(test_gemma4_paged_engine parity/test_gemma4_paged_engine.cpp)
target_compile_definitions(test_gemma4_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_gemma4_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_granite_paged_engine parity/test_granite_paged_engine.cpp)
target_compile_definitions(test_granite_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_granite_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_minicpm_paged_engine parity/test_minicpm_paged_engine.cpp)
target_compile_definitions(test_minicpm_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_minicpm_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_minicpm3_paged_engine parity/test_minicpm3_paged_engine.cpp)
target_compile_definitions(test_minicpm3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_minicpm3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_phi3_paged_engine parity/test_phi3_paged_engine.cpp)
target_compile_definitions(test_phi3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_phi3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_olmo3_paged_engine parity/test_olmo3_paged_engine.cpp)
target_compile_definitions(test_olmo3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_olmo3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_stablelm_paged_engine parity/test_stablelm_paged_engine.cpp)
target_compile_definitions(test_stablelm_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_stablelm_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_internlm2_paged_engine parity/test_internlm2_paged_engine.cpp)
target_compile_definitions(test_internlm2_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_internlm2_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_internlm3_paged_engine parity/test_internlm3_paged_engine.cpp)
target_compile_definitions(test_internlm3_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_internlm3_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_yi_paged_engine parity/test_yi_paged_engine.cpp)
target_compile_definitions(test_yi_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_yi_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_commandr_paged_engine parity/test_commandr_paged_engine.cpp)
target_compile_definitions(test_commandr_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_commandr_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_phi_paged_engine parity/test_phi_paged_engine.cpp)
target_compile_definitions(test_phi_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_phi_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GGUF 35B greedy acceptance gate (checkpoint-gated, dgx-only; skips when the
# APEX k-quant GGUFs are absent). Loads the REAL .gguf single-file path
# (metadata + embedded vocab + k-quant dequant) through the paged engine and
# asserts greedy token-for-token vs the pinned same-file llama.cpp oracle.
vllm_cpp_add_test(test_qwen36_gguf_engine parity/test_qwen36_gguf_engine.cpp)
target_compile_definitions(test_qwen36_gguf_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_gguf_engine PRIVATE ${CMAKE_SOURCE_DIR}/tests/parity)

# `QUANT-GGUF-NVFP4` column C: native fp4 compute out of a GGUF container.
# Asset-gated (VLLM_NVFP4_GGUF, optionally VLLM_NVFP4_ST_DIR); compiles and links
# on CPU/CI but only RUNS where the real 27B NVFP4 files exist.
vllm_cpp_add_test(test_qwen27_gguf_nvfp4_compute
  parity/test_qwen27_gguf_nvfp4_compute.cpp)

# `QUANT-GGUF-NVFP4` column C, MoE arm: the 35B A3B stacked-expert residency.
# Asset-gated (VLLM_NVFP4_MOE_GGUF, optionally VLLM_NVFP4_MOE_ST_DIR); compiles
# and links on CPU/CI but only RUNS where the real 35B NVFP4 files exist.
vllm_cpp_add_test(test_qwen36_gguf_nvfp4_compute
  parity/test_qwen36_gguf_nvfp4_compute.cpp)

# Dense 27B W4A4 greedy acceptance gate (checkpoint-gated, dgx-only; skips on
# CPU/CI when the real snapshot is absent). Counterpart to the 35B gate; asserts
# the paged dense engine reproduces the pip-vLLM oracle greedy continuation
# token-for-token via the fp4-resident W4A4 GEMM. See qwen27b-w4a4-notes.md §5.
vllm_cpp_add_test(test_qwen27_paged_engine parity/test_qwen27_paged_engine.cpp)
target_compile_definitions(test_qwen27_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# `GATE-27B-FP8-TOWER-GOLDEN` (issue #466): the SECOND dense 27B arm, pinned to
# nvidia/Qwen3.6-27B-NVFP4@0893e160 (`modelopt_mixed` — W4A16 NVFP4 MLP + FP8
# W8A8 tower + NVFP4 head). The gate above pins unsloth@890bdef7, whose GDN
# input projections are BF16, so every fp8 branch is dead code there and that
# gate can never fail for an fp8 defect. This arm asserts the fp8 GDN
# input-projection dispatch contract UNCONDITIONALLY — a zero count is a
# failure, not a skipped block. Checkpoint-gated, dgx-only; on CPU/CI it refuses
# loudly instead of reporting a vacuous pass.
vllm_cpp_add_test(test_qwen27n_fp8_tower_paged_engine
  parity/test_qwen27n_fp8_tower_paged_engine.cpp)
target_compile_definitions(test_qwen27n_fp8_tower_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27n_fp8_tower_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# `PERF-27B-LMHEAD-FP4` (issue #213): the ModelOpt NVFP4 lm_head stays PACKED.
# Synthetic (no checkpoint, no GPU) BECAUSE test_qwen27_paged_engine runs the
# unsloth @890bdef7 snapshot, whose head is BF16 — that gate cannot see this path.
vllm_cpp_add_test(test_qwen27_dense_lmhead_fp4
  parity/test_qwen27_dense_lmhead_fp4.cpp)

# SPEC-DFLASH D2 — the DFlash draft-forward parity gate vs the dumped vLLM DFlash
# draft reference (fc combine + context-free block forward per-stage hidden +
# proposed ids). GPU + z-lab-draft + NVFP4-target + fixture gated (skips on CPU/CI).
vllm_cpp_add_test(test_qwen3_dflash_draft_parity parity/test_qwen3_dflash_draft_parity.cpp)
target_compile_definitions(test_qwen3_dflash_draft_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dflash_draft_parity PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-DFLASH D3 (DF-DRAFT-KV-PREP) — the decisive GPU numeric-parity gate vs the
# REAL loaded vLLM draft: prepare_dflash_inputs INTEGER bit-exact (vs vLLM's actual
# Triton kernel), context-KV precompute per-layer rel-L2 within envelope, block
# proposal STRICT-or-near-tie. GPU + z-lab-draft + NVFP4-target + fixture gated.
vllm_cpp_add_test(test_qwen3_dflash_kvprep_parity parity/test_qwen3_dflash_kvprep_parity.cpp)
target_compile_definitions(test_qwen3_dflash_kvprep_parity PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_dflash_kvprep_parity PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-MTP I5d — the 27B k=1 MTP speculative-decode three-way identity gate
# (our spec-ON == our spec-OFF == vLLM greedy) + measured nonzero acceptance.
vllm_cpp_add_test(test_qwen35_gguf_spec_decode parity/test_qwen35_gguf_spec_decode.cpp)
vllm_cpp_add_test(test_qwen27_spec_decode parity/test_qwen27_spec_decode.cpp)
target_compile_definitions(test_qwen27_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-DFLASH D5 — the 27B DFlash block-diffusion e2e correctness gate: our
# DFlash-ON == the committed vLLM-DFlash-ON golden (STRICT mode-matched) + measured
# nonzero acceptance ~ vLLM's (the dead-drafter trap). dgx-only (needs the NVFP4
# target + the z-lab draft + the CUDA spec kernels); loud SKIP on the CPU dev box.
vllm_cpp_add_test(test_qwen27_dflash_spec_decode
  parity/test_qwen27_dflash_spec_decode.cpp)
target_compile_definitions(test_qwen27_dflash_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_dflash_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# SPEC-MTP M-mtp-2 — the 35B MoE-MTP k=1 speculative-decode three-way identity
# gate (Qwen3_5MoeMTP: our spec-ON == our spec-OFF == vLLM greedy) + measured
# nonzero acceptance. Mirror of the 27B gate at 35B (MoE MTP draft head).
vllm_cpp_add_test(test_qwen36_spec_decode parity/test_qwen36_spec_decode.cpp)
target_compile_definitions(test_qwen36_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen36_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# GATE-PIN-UNPINNED-SNAPSHOTS (#471). The resolver gate: CPU-only, needs no
# checkpoint, synthesises its own two-revision cache. It is the direct proof that
# a pinned accessor selects rather than substitutes, and unlike every gate it
# protects, it runs everywhere.
vllm_cpp_add_test(test_hf_snapshot_pinning parity/test_hf_snapshot_pinning.cpp)
target_include_directories(test_hf_snapshot_pinning PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)

# SPEC-NGRAM (ROAD-V1-D3) — the 27B draft-free ngram e2e gate: our ngram-ON ==
# the committed vLLM-ngram-ON golden (STRICT, exactness-preserving greedy) +
# measured nonzero acceptance ~ vLLM's. dgx-only; loud SKIP on the CPU dev box.
vllm_cpp_add_test(test_qwen27_ngram_spec_decode
  parity/test_qwen27_ngram_spec_decode.cpp)
target_compile_definitions(test_qwen27_ngram_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen27_ngram_spec_decode PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

vllm_cpp_add_test(test_qwen27_spec_decode_concurrent
  parity/test_qwen27_spec_decode_concurrent.cpp)
target_include_directories(test_qwen27_spec_decode_concurrent PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W4 (Qwen3-Coder-30B-A3B, Qwen3MoeForCausalLM) — the SACRED
# full-attention BF16 MoE greedy correctness gate: drives a compact prompt
# battery through the paged engine and checks per-(prompt,pos) set membership vs
# the vLLM 0.25.0 oracle's K-run greedy set (STRICT where vLLM is deterministic).
# Checkpoint-gated, dgx-only; skips on CPU/CI when the snapshot/goldens are absent.
vllm_cpp_add_test(test_moe_two_engines vllm/models/test_moe_two_engines.cpp)
target_include_directories(test_moe_two_engines PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3coder_paged_engine vllm/models/test_qwen3coder_paged_engine.cpp)
target_compile_definitions(test_qwen3coder_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3coder_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W2/W4 (OPT-125m, OPTForCausalLM) — the CROSS-FAMILY additivity
# canary: a host-only loader gate plus the SACRED greedy correctness gate, which
# is STRICT token-exact because the vLLM 0.25.0 oracle measured DETERMINISTIC on
# this model (K=5, 0 multi-valued cells). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_opt_load vllm/models/test_opt_load.cpp)
vllm_cpp_add_test(test_opt_paged_engine vllm/models/test_opt_paged_engine.cpp)
target_compile_definitions(test_opt_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_opt_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# KV-EXTERNAL-CACHE (LMCache): the connector-ON full-model OUTPUT-INVARIANCE +
# throughput gate. Real OPT-125m generation loop vs a live lmcache.v1.server;
# checkpoint- and server-gated (loud SKIP otherwise). Driven by
# scripts/lmcache/run_output_invariance.sh.
vllm_cpp_add_test(test_lmcache_output_invariance
  vllm/models/test_lmcache_output_invariance.cpp)

# ADDITIVE-MODEL W2/W4 (Llama-3.2-1B, LlamaForCausalLM) — the CROSS-FAMILY dense
# additivity bring-up: a host-only loader gate plus the SACRED near-tie-robust
# greedy correctness gate vs the vLLM 0.25.0 oracle (llama3 rope-scaling, no
# qk-norm, GQA 32/8, head_dim 64, tied lm_head). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_llama_load vllm/models/test_llama_load.cpp)
target_include_directories(test_llama_load PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_llama_paged_engine parity/test_llama_paged_engine.cpp)
target_compile_definitions(test_llama_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_llama_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-MODEL W2/W4 (Mistral-7B-v0.3, MistralForCausalLM) — the FIFTH-family
# near-additive dense bring-up: a host-only loader gate plus the SACRED
# near-tie-robust greedy correctness gate vs the vLLM 0.25.0 oracle (PLAIN rope
# theta 1e6 / no scaling, no qk-norm, GQA 32/8, head_dim 128, UNTIED lm_head,
# sliding_window null). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_mistral_load vllm/models/test_mistral_load.cpp)
target_include_directories(test_mistral_load PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_mistral_paged_engine parity/test_mistral_paged_engine.cpp)
target_compile_definitions(test_mistral_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_mistral_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# ADDITIVE-QUANT W2/W4 (Qwen3-32B-NVFP4A16, Qwen3ForCausalLM + compressed-tensors
# NVFP4A16 / W4A16) — the QUANT-SCHEME additivity row: the SAME already-token-exact
# dense forward with only the Linear storage/compute scheme changed. A host-only
# loader gate plus the SACRED greedy correctness gate, which is STRICT token-exact
# because the vLLM 0.25.0 oracle measured DETERMINISTIC on this checkpoint (K=5,
# 0 multi-valued cells). Checkpoint-gated, dgx-only.
vllm_cpp_add_test(test_qwen3_32b_nvfp4a16_load
  vllm/models/test_qwen3_32b_nvfp4a16_load.cpp)

vllm_cpp_add_test(test_qwen3_5_lm_head_dtypes
  vllm/models/test_qwen3_5_lm_head_dtypes.cpp)
vllm_cpp_add_test(test_qwen3_32b_nvfp4a16_paged_engine
  vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp)
target_compile_definitions(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
