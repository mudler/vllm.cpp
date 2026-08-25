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
# ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace cache layout. No socket,
# so it is not behind VLLM_CPP_SERVER; the fake-hub half is, below.
vllm_cpp_add_test(test_hf_cache vllm/transformers_utils/test_hf_cache.cpp)
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
# For src/vt/cpu/cpu_conv1d_block.h: the multi-block case asserts the geometry
# it claims rather than assuming it (#1664). Same reach as test_host_parallel.
target_include_directories(test_vocoder1d PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The bit-identity gate for the row-wise parallelisation of the host-reference
# kernels (#672). It carries VERBATIM copies of the pre-parallel serial loops as
# its oracle, so it reaches src/ for `host_parallel.h` (the size guard whose
# threshold every case must clear, or it would test the inline path twice) and
# for `Threadpool::SwapForTesting`, the same hook the vt determinism suites use.
# CPU-only.
vllm_cpp_add_test(test_host_parallel vllm/models/test_host_parallel.cpp)
target_include_directories(test_host_parallel PRIVATE ${CMAKE_SOURCE_DIR}/src)
# The accounting gate for the MiniMax-Music3 stage profiler (#672). It reaches
# src/ for `music3_profile.h`, which is an INTERNAL instrument and deliberately
# not on the public surface. CPU-only, no checkpoint.
vllm_cpp_add_test(test_music3_profile vllm/models/test_music3_profile.cpp)
target_include_directories(test_music3_profile PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_speech_engine vllm/multimodal/test_speech_engine.cpp)
vllm_cpp_add_test(test_indextts2_family vllm/models/test_indextts2_family.cpp)
vllm_cpp_add_test(test_campplus vllm/models/test_campplus.cpp)
vllm_cpp_add_test(test_indextts2_conditioning vllm/models/test_indextts2_conditioning.cpp)
vllm_cpp_add_test(test_emovec vllm/models/test_emovec.cpp)
vllm_cpp_add_test(test_w2vbert vllm/models/test_w2vbert.cpp)
vllm_cpp_add_test(test_fvq vllm/models/test_fvq.cpp)
vllm_cpp_add_test(test_codec_encoder vllm/models/test_codec_encoder.cpp)
vllm_cpp_add_test(test_w2v_fbank vllm/models/test_w2v_fbank.cpp)
vllm_cpp_add_test(test_wavenet vllm/models/test_wavenet.cpp)
vllm_cpp_add_test(test_dit_tail vllm/models/test_dit_tail.cpp)
vllm_cpp_add_test(test_dit_skip vllm/models/test_dit_skip.cpp)
vllm_cpp_add_test(test_dit_front vllm/models/test_dit_front.cpp)
vllm_cpp_add_test(test_dit_stack vllm/models/test_dit_stack.cpp)
vllm_cpp_add_test(test_bigvgan vllm/models/test_bigvgan.cpp)
vllm_cpp_add_test(test_indextts2_s2mel_loader vllm/models/test_indextts2_s2mel_loader.cpp)
vllm_cpp_add_test(test_indextts2_render vllm/models/test_indextts2_render.cpp)
vllm_cpp_add_test(test_indextts2_e2e vllm/models/test_indextts2_e2e.cpp)
vllm_cpp_add_test(test_indextts2_talker_loader vllm/models/test_indextts2_talker_loader.cpp)
vllm_cpp_add_test(test_vocos vllm/models/test_vocos.cpp)
vllm_cpp_add_test(test_lenreg vllm/models/test_lenreg.cpp)
vllm_cpp_add_test(test_cfm vllm/models/test_cfm.cpp)
vllm_cpp_add_test(test_adaln vllm/models/test_adaln.cpp)
vllm_cpp_add_test(test_dit vllm/models/test_dit.cpp)
vllm_cpp_add_test(test_talker vllm/models/test_talker.cpp)
vllm_cpp_add_test(test_tiktoken_bpe vllm/models/test_tiktoken_bpe.cpp)
vllm_cpp_add_test(test_indextts2_pipeline vllm/models/test_indextts2_pipeline.cpp)
target_include_directories(test_dit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_adaln PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_cfm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_lenreg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_vocos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_fvq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_codec_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_w2v_fbank PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_wavenet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_tail PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_front PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_dit_stack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# VT-CONV1D-MODEL-BLOCK (#1684): the time-block case reads `Conv1dTimeBlock` so
# it asserts the geometry it claims. Same reach as test_vocoder1d.
target_include_directories(test_bigvgan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  ${CMAKE_SOURCE_DIR}/src)
target_include_directories(test_w2vbert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
target_include_directories(test_campplus PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_gpt2 vllm/models/test_gpt2.cpp)
target_include_directories(test_gpt2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_minimax_h3 vllm/models/test_minimax_h3.cpp)
# VT-CONV1D-MODEL-BLOCK (#1684): the time-block case reads `Conv1dTimeBlock` so
# it asserts the geometry it claims. Same reach as test_vocoder1d.
target_include_directories(test_minimax_h3 PRIVATE ${CMAKE_SOURCE_DIR}/src)
# MiniMax-Music3 W1 (#672): the six-component diffusers-arm loader, gated against
# the REAL checkpoint's headers (minimax_music3_manifest.inc, 1012 tensors, no
# weight bytes) so CI never needs the 28.5 GB asset.
vllm_cpp_add_test(test_minimax_music3_loader vllm/models/test_minimax_music3_loader.cpp)
target_include_directories(test_minimax_music3_loader PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# MiniMax-Music3 W2+W3 (#672): the AUTOREGRESSIVE half — prompt assembly, the
# semantic CFG logit pipeline, the learned 8-layer condition mix and the RVQ
# depth decoder — against goldens produced by EXECUTING upstream's own classes at
# reduced dimensions in float32 (minimax_music3_ar_goldens.inc, no weight bytes).
# The full-scale bf16 companion is tests/parity/test_minimax_music3_ar_real.cpp.
vllm_cpp_add_test(test_minimax_music3_ar vllm/models/test_minimax_music3_ar.cpp)
target_include_directories(test_minimax_music3_ar PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  # ...and src/ for `music3_profile.h`, the INTERNAL instrument that carries the
  # `ar.depth_device` / `ar.depth_host` buckets this suite drives on both sides
  # (#1839). Deliberately not a public header.
  ${CMAKE_SOURCE_DIR}/src)
# The FULL-SCALE companion: the REAL bf16 checkpoint driven on the REAL inputs,
# against the committed oracle capture (frame_hiddens / condition_chunk0 /
# rvq_codes). Checkpoint-gated on MUSIC3_CHECKPOINT or CHECKPOINT_ROOT; every
# case emits a loud SKIP when the 28.5 GB asset is absent, so CI still compiles,
# links and runs it.
vllm_cpp_add_test(test_minimax_music3_ar_real parity/test_minimax_music3_ar_real.cpp)
target_compile_definitions(test_minimax_music3_ar_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_ar_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# MiniMax-Music3 W2's remainder (#672): the 8.6B `Qwen3ForCausalLM` forward
# itself, driven through the ADDITIVE `Qwen3DenseModel::ForwardEmbeds` entry and
# TEACHER-FORCED on the capture's own rvq_codes, against frame_hiddens[:, :4096].
# Checkpoint-gated like its two siblings; every case emits a loud SKIP when the
# 28.5 GB asset is absent, so CI still compiles, links and runs it. SEPARATE from
# test_minimax_music3_ar_real because it stages 18.5 GB that gate does not need.
vllm_cpp_add_test(test_minimax_music3_llm_real parity/test_minimax_music3_llm_real.cpp)
target_compile_definitions(test_minimax_music3_llm_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_llm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# MiniMax-Music3 W4+W5 (#672): the ACOUSTIC half — the flow-matching DiT, the
# FlowMatchEulerDiscreteScheduler with invert_sigmas, the CFG mix, the denoise
# loop's window bookkeeping and the DAC Flow-VAE vocoder — against goldens
# produced by EXECUTING upstream's own classes at reduced dimensions in float32
# (minimax_music3_acoustic_goldens.inc, no weight bytes). The full-scale fp32
# companion is tests/parity/test_minimax_music3_acoustic_real.cpp.
vllm_cpp_add_test(test_minimax_music3_acoustic vllm/models/test_minimax_music3_acoustic.cpp)
# Reaches src/ for `music3_profile.h` — an INTERNAL instrument, deliberately not
# on the public surface — because the intra-DiT spans (#1542) live inside
# `DitForwardDevice` and this is the only suite that drives that forward.
target_include_directories(test_minimax_music3_acoustic PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models
  ${CMAKE_SOURCE_DIR}/src)
# The FULL-SCALE companion: the REAL fp32 2.4B DiT and 0.054B vocoder driven on
# the REAL inputs, against the committed oracle capture (denoise_{first,last}_*,
# vocoder_input_chunk0, waveform). Checkpoint-gated exactly as the AR one is;
# every case emits a loud SKIP when the 28.5 GB asset is absent, so CI still
# compiles, links and runs it.
vllm_cpp_add_test(test_minimax_music3_acoustic_real
  parity/test_minimax_music3_acoustic_real.cpp)
target_compile_definitions(test_minimax_music3_acoustic_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")
target_include_directories(test_minimax_music3_acoustic_real PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity)
# W6 (#672): the SPEECH-FAMILY seam, the request contract and the chunk plan.
# Pure or synthetic-tree only, so it needs neither the checkpoint nor a golden.
# W7 of #672 -- the QUANTIZED ARMS. Nothing here loads a quantized weight:
# no quantized MiniMax-Music3 checkpoint is staged, and none is implemented.
# What it gates is that every quantized format a user can obtain today is
# DIAGNOSED BY NAME and refused, instead of surfacing as the generic
# "missing seven diffusers directories" or as a shape mismatch on an
# unrelated tensor. Needs no checkpoint and no network.
vllm_cpp_add_test(test_minimax_music3_quant vllm/models/test_minimax_music3_quant.cpp)

vllm_cpp_add_test(test_minimax_music3_speech vllm/models/test_minimax_music3_speech.cpp)
# W6's FULL-SCALE companion: the pipeline TAIL driven on the shipped weights,
# against the capture's own waveform, and the WAV the delivery path emits.
# Checkpoint-gated with a loud SKIP, exactly as the W4/W5 real gate is.
# W7 of #672 -- the ONE quantized arm that is implemented, against the REAL
# pinned artifact (audio-cpp/MiniMax-Music3-GGUF @c36aaeed,
# rvq_depth_decoder_q4_k.gguf, 406 MB). Skips loudly without it. Proves the
# QUANTIZED PATH IS TAKEN (resident ggml types + the Q4_K lattice), not merely
# that the numbers land inside a tolerance.
vllm_cpp_add_test(test_minimax_music3_quant_real parity/test_minimax_music3_quant_real.cpp)
target_compile_definitions(test_minimax_music3_quant_real PRIVATE
  MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle")

# test_minimax_music3_e2e_real is NOT registered here. It links `ApiServer`,
# which `CMakeLists.txt` compiles only inside `if(VLLM_CPP_SERVER)`, so its
# registration lives inside that same guard below with the other
# server-linking suites (#1883).

# THE ENGINE'S OWN CALL to the DiT arm selector (#1131, row MUSIC3-DIT-ARM-REACH).
# Wave 1 of that row gated the RULE from a CPU runner; what stayed unreached was
# the two-line CALL in `Music3SpeechEngine::Synthesize`, which needs an
# accelerator and the 28.5 GB checkpoint and therefore cannot run on anything CI
# owns. This entry is that gate. It drives `include/vllm.h` -- the production
# entry point -- with `device = 1` and asserts WHICH ARM RAN off the engine's own
# profile buckets, never off the audio, because the two arms agree by design.
#
# LABELLED, because a gate whose preconditions no runner has must be selectable
# and excludable by name rather than by whoever remembers it exists:
#   ctest -L gpu    selects it   (inside an `rc` lease; see docs/USAGE.md)
#   ctest -LE gpu   excludes it
# Without a device or a checkpoint it EXITS 77 and CTest reports **Skipped** --
# `vllm_cpp_add_test` registers SKIP_RETURN_CODE 77 -- so it can never be read as
# a pass. RUN_SERIAL because it stages ~10 GB of fp32 DiT onto the accelerator.
vllm_cpp_add_test(test_minimax_music3_device_arm_real
  parity/test_minimax_music3_device_arm_real.cpp)
# Reaches src/ for `music3_profile.h`, the INTERNAL instrument the engine's own
# `profile::Report` prints. Deliberately not a public header: it is how this gate
# reads which arm ran, and it is not part of the ABI.
target_include_directories(test_minimax_music3_device_arm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_minimax_music3_device_arm_real PROPERTIES
  RUN_SERIAL ON
  LABELS "gpu;checkpoint;music3")

# THE ENGINE'S OWN CALL to the DEPTH arm selector (#1839, row
# MUSIC3-DEPTH-ARM-REACH). The twin of the entry above, and a separate entry for
# a reason: #1131 named BOTH device-arm twins, the DiT one closed with it, and
# the depth call site at `minimax_music3_speech.cpp:638` stayed unreached --
# deleting it left `test_minimax_music3_ar` 37/37 and `test_minimax_music3_speech`
# 9/9 green. Two call sites on one `--speech-device 1` switch is how one of them
# drifts, so each carries its own gate rather than one gate asserting both.
#
# Same labels, same 77-on-missing-precondition contract, same lease. `ctest -L gpu`
# now selects TWO tests, and that count is pinned by name in
# `scripts/check-test-registration.py` because `ctest -L` returns 0 over an empty
# selection. RUN_SERIAL because it stages the depth decoder and the DiT onto the
# accelerator and loads 28.5 GB to do it.
vllm_cpp_add_test(test_minimax_music3_depth_arm_real
  parity/test_minimax_music3_depth_arm_real.cpp)
# Reaches src/ for `music3_profile.h`, the INTERNAL instrument the engine's own
# `profile::Report` prints. Deliberately not a public header: it is how this gate
# reads which arm ran, and it is not part of the ABI.
target_include_directories(test_minimax_music3_depth_arm_real PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES
  RUN_SERIAL ON
  LABELS "gpu;checkpoint;music3")
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
# The render phase INSTRUMENT, on its own, over synthetic timelines (row
# LTX25-PHASE-INSTRUMENT, #1668/#1569/#1571). Its own executable rather than a
# block inside `test_ltx2_video`, because two of its cases need a table of
# thousands of records that no render produces, and because an instrument
# question should not cost a model fixture build. The reachability half — that a
# render through the ABI actually emits what these cases pin — stays in
# `test_ltx2_video`, which is the only place that can prove it.
vllm_cpp_add_test(test_render_phase_log vllm/multimodal/test_render_phase_log.cpp)
# LTX-2.5 L7: the family behind the VideoEngine seam and the driving loop, run
# over a reduced-dimension checkpoint set written in the SHIPPED file format.
vllm_cpp_add_test(test_ltx2_video vllm/multimodal/test_ltx2_video.cpp)
# The reader-anchor claim this suite gates is a claim ABOUT a source file, so it
# reads that file. Recorded `file:line` anchors go stale within the same PR here
# (a merge of origin/main moved all nine of these by 25 lines mid-review), which
# is why the anchors are derived from the source and compared with the recorded
# ones rather than trusted.
#
# The second path is the conditioning header, and it is a DIFFERENT file from the
# one the refusal it checks lives in. The generated-keyframe refusal states its
# claims about this tree as a list of names, and the suite re-derives each one
# from these declarations. Reading the claim out of the file that makes it is the
# tautology issue #911 records, so the two files are deliberately not the same.
#
# The third path is the DFR canvas header (row LTX25-DFR-PIPELINE, #986), read
# for the same reason and by the same rule. The temporal-rounds refusal rules the
# canvas layout OUT as a blocker on the grounds that it is ported here, which is
# a claim about THIS tree; the suite re-derives it from these declarations rather
# than from the message that makes it.
target_compile_definitions(test_ltx2_video PRIVATE
  LTX2_VIDEO_SOURCE_PATH="${CMAKE_SOURCE_DIR}/src/vllm/multimodal/ltx2_video.cpp"
  LTX2_CONDITIONING_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_conditioning.h"
  LTX2_DFR_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_dfr.h")
# LTX-2.5 image conditioning (row LTX25-IMAGE-COND, issue #644): the CHAIN from
# PPM bytes to conditioned, noised tokens, gated against upstream ltx_core AND
# ltx_pipelines executed at reduced dimensions. ltx2_image_cond_goldens.inc
# lives next to the test source.
vllm_cpp_add_test(test_ltx2_image_cond vllm/multimodal/test_ltx2_image_cond.cpp)
target_include_directories(test_ltx2_image_cond
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/multimodal)
# #659/#660: both diffusion lanes' device resolution held against a PARTIAL
# backend that DECLINES the architecture. Its own executable for the same reason
# test_device_selection is one — it registers a fake platform and backend in the
# global registries (XPU slot, and over the CUDA slot so CurrentPlatform()
# resolves to it on a CUDA build too), and process isolation keeps that out of
# every other suite.
vllm_cpp_add_test(test_diffusion_device_seam vllm/multimodal/test_diffusion_device_seam.cpp)
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
# LTX25-DECODE-THREADS (issue #1009): the two threading cases reach
# vt::cpu::Threadpool::SwapForTesting and the pool's work-stealing cursor, which
# live under src/ like every other CPU-threading A/B in tests/vt/.
target_include_directories(test_ltx2_vae PRIVATE ${CMAKE_SOURCE_DIR}/src)
# LTX25-TILED-DECODE (issue #644): the tiling algebra, the AUTO layout and the
# STREAMING tiled decode, gated against upstream ltx_core executed at reduced
# dimensions. Its own target so the decode arms — which run the reduced decoder 18
# times per arm — do not lengthen test_ltx2_vae's recorded case count.
# ltx2_tiling_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_tiling vllm/models/test_ltx2_tiling.cpp)
target_include_directories(test_ltx2_tiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L3: the text conditioning path, gated against upstream ltx_core executed
# at reduced dimensions. ltx2_text_goldens.inc lives next to the test source.
vllm_cpp_add_test(test_ltx2_text_encoder vllm/models/test_ltx2_text_encoder.cpp)
target_include_directories(test_ltx2_text_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# LTX-2.5 L5: the pipeline layer — sigma schedule, noiser, diffusion steps,
# guidance, patchifiers, the recipe table, the latent spatial upsampler, the
# duration head and the embeddings connector. ltx2_pipeline_goldens.inc lives
# next to the test source.
# LTX-2.5 DFR canvas layout (ROW LTX25-DFR-PIPELINE, issue #986). Gated against
# EXECUTED upstream `dfr_layout` and the three pure-index `dfr_pipeline` helpers;
# ltx2_dfr_goldens.inc lives next to the test source. Every value is an INDEX, so
# the suite compares exact vectors rather than bounding a magnitude — there is no
# NaN, no shape disagreement and no scale for a looser gate to notice.
vllm_cpp_add_test(test_ltx2_dfr vllm/models/test_ltx2_dfr.cpp)
target_include_directories(test_ltx2_dfr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_ltx2_pipeline vllm/models/test_ltx2_pipeline.cpp)
target_include_directories(test_ltx2_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
# The out-of-scope ledger separates a REACHABLE refusal from a declared marker, and
# docs/FEATURES.md is where that distinction reaches a user. It had re-merged the two
# into one "Declared, not requestable" row, so the public surface contradicted the
# ledger it projects; the case below reads the doc and holds it to the split.
# The ledger's RETIREMENT note carries an absence claim about upstream, and it
# shipped false twice in the same header. The case that guards it reads the header
# itself rather than a copy of the sentence kept in the test.
# The ledger's reachable/marker split is a claim about the PRODUCT TREE — that
# something calls the function holding a reachable refusal, and that nothing calls
# the one holding a marker. It was wrong about `kBetaScheduler` for the whole of
# this row because it was maintained by hand, so the case now walks src/, include/
# and examples/ under this root and derives it, with two positive controls in the
# same walk (an absence claim about our own tree is still an absence claim).
target_compile_definitions(test_ltx2_pipeline PRIVATE
  VLLM_CPP_FEATURES_DOC_PATH="${CMAKE_SOURCE_DIR}/docs/FEATURES.md"
  LTX2_PIPELINE_HEADER_PATH="${CMAKE_SOURCE_DIR}/include/vllm/model_executor/models/ltx2_pipeline.h"
  VLLM_CPP_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
# LTX-2.5 L6: the quantized loaders, gated three ways — the SHIPPED checkpoints'
# own manifests, a few hundred of their own bytes against torch, and synthetic
# files for whole-model materialization. All three .inc files live beside the
# test source.
# LTX25-RETAKE (#924): the ported pieces at value level — the temporal region
# mask under both coordinate conventions, the truncate-or-pad conform and the
# CLI-stage validations. Its own target rather than cases inside
# test_ltx2_pipeline, which several concurrent rows of this campaign edit. The
# REACHABILITY proof for the same row lives in test_ltx2_video.
vllm_cpp_add_test(test_ltx2_retake vllm/models/test_ltx2_retake.cpp)
vllm_cpp_add_test(test_ltx2_lora vllm/models/test_ltx2_lora.cpp)
vllm_cpp_add_test(test_ltx2_loader vllm/models/test_ltx2_loader.cpp)
target_include_directories(test_ltx2_loader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)
vllm_cpp_add_test(test_video_api vllm/entrypoints/openai/test_video_api.cpp)
vllm_cpp_add_test(test_speech_api vllm/entrypoints/openai/test_speech_api.cpp)
target_include_directories(test_model_registry PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_platform vllm/platforms/test_platform.cpp)
vllm_cpp_add_test(test_cuda_arch_manifest
  vllm/platforms/test_cuda_arch_manifest.cpp)
vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
# ENG-EXPERT-STREAM #1123: the load-time GGUF device-fit refusal, in two targets
# for two different questions. The arithmetic runs everywhere; the REACHABILITY
# gate needs a fake WEIGHT-STAGING platform in the CUDA lookup slot, because
# `needs_weight_staging()` is true on exactly one platform in this tree and the
# branch is otherwise unreachable on a host with no CUDA device. Its own
# executable for the same reason test_device_selection is one.
vllm_cpp_add_test(test_gguf_device_fit
  vllm/model_executor/test_gguf_device_fit.cpp)
vllm_cpp_add_test(test_gguf_device_fit_reach
  vllm/entrypoints/test_gguf_device_fit_reach.cpp)
# LOAD-GGUF-MMPROJ (#821): the `clip` mmproj reader, and — separately — the
# proof that the production loader reaches it. The second target enters through
# `LoadedEngine::FromModelDir` with `--mmproj` set, so deleting the call site in
# `model_loader.cpp` reds it; the first would stay green, which is exactly the
# difference between gating a class and gating a capability.
vllm_cpp_add_test(test_clip_mmproj_gguf
  vllm/models/test_clip_mmproj_gguf.cpp)
vllm_cpp_add_test(test_gguf_mmproj_reach
  vllm/entrypoints/test_gguf_mmproj_reach.cpp)
# QUANT-QWEN38-27B-GGUF-ARM (#821 W2): the committed 866- and 334-name
# manifests of `unsloth/Qwen3.8-27B-GGUF` @ fe1e2a23d9, and the tensor
# accounting against them. Split the same way for the same reason — the first
# target calls the two enumerations, the second enters through
# `LoadedEngine::FromModelDir`, so deleting either refusal call site in
# `model_loader.cpp` reds the second and leaves the first green.
vllm_cpp_add_test(test_qwen38_27b_gguf_manifest
  vllm/models/test_qwen38_27b_gguf_manifest.cpp)
vllm_cpp_add_test(test_gguf_accounting_reach
  vllm/entrypoints/test_gguf_accounting_reach.cpp)
# FIX-FP8-BLOCKWISE-REFUSAL #1166: block-wise (128x128) FP8 is refused BY NAME
# at load. Every case enters through `ModelRegistry::Load` rather than through
# the predicate, so deleting that call site reds this target. That is the
# reachability proof and not a class test.
vllm_cpp_add_test(test_fp8_block_quant
  vllm/model_executor/layers/test_fp8_block_quant.cpp)
# MODEL-FP8-BLOCK-WEIGHT #1189 M3: `Fp8BlockWeight`, the `weight_scale_inv`
# loader rung, and the config reader. Every case enters through
# `ModelRegistry::Load` or `LoadQwen3_5Dense` -- the loader that call reaches --
# and never through `LoadFp8BlockRaw`, so deleting the rung reds this target.
vllm_cpp_add_test(test_fp8_block_weight_load
  vllm/model_executor/models/test_fp8_block_weight_load.cpp)

# MODEL-FP8-BLOCK-LINEAR (#1189 M4): the block-wise FP8 linear method and the
# dense forward that reaches it.
vllm_cpp_add_test(test_fp8_block_linear
  vllm/model_executor/models/test_fp8_block_linear.cpp)

# MODEL-FP8-BLOCK-MERGED (#1189 M6): the merged gate_up and QKV projections of
# the block-wise FP8 arm. Enters at `ModelRegistry::Forward`, so deleting either
# production merge call site reds this target.
vllm_cpp_add_test(test_fp8_block_merged
  vllm/model_executor/models/test_fp8_block_merged.cpp)

# GATE-FP8-NUMERIC-BOUND -- the numerical lower bound on per-projection outputs
# and the per-block scale-variance probe, #1189's gate design layers 2 and 3.
vllm_cpp_add_test(test_fp8_block_numeric_bound
  vllm/model_executor/models/test_fp8_block_numeric_bound.cpp)
vllm_cpp_add_test(test_linear_method vllm/model_executor/layers/test_linear_method.cpp)
target_include_directories(test_linear_method PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_break_point vllm/models/test_qwen3_break_point.cpp)
vllm_cpp_add_test(test_qwen3_decode_graph_seam vllm/models/test_qwen3_decode_graph_seam.cpp)
vllm_cpp_add_test(test_qwen3_moe_decode_graph_seam vllm/models/test_qwen3_moe_decode_graph_seam.cpp)
vllm_cpp_add_test(test_moe_async_device_ids vllm/models/test_moe_async_device_ids.cpp)
vllm_cpp_add_test(test_voxtral_decode_graph_seam vllm/models/test_voxtral_decode_graph_seam.cpp)
vllm_cpp_add_test(test_deepseek_v2_decode_graph_seam vllm/models/test_deepseek_v2_decode_graph_seam.cpp)
vllm_cpp_add_test(test_qwen3_5_decode_graph_seam vllm/models/test_qwen3_5_decode_graph_seam.cpp)
# W6 (#1374): the dispatch counters live under src/, beside the runner (:561).
target_include_directories(test_qwen3_5_decode_graph_seam PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W10 repair (#1865): the model-side FA-2 dtype/lane class — the
# spec-as-decode verify selects bf16 through the SPEC lane's own toggles. The
# seam lives in qwen3_5_internal.h under src/.
vllm_cpp_add_test(test_qwen3_5_fa2_class vllm/models/test_qwen3_5_fa2_class.cpp)
target_include_directories(test_qwen3_5_fa2_class PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_dflash_decode_graph_seam vllm/models/test_qwen3_dflash_decode_graph_seam.cpp)
# G1 for the row: bit-exactness against eager over THREE replays, on a real
# device. Inert without a CUDA backend (every case skips), because a CPU
# "replay" recomputes nothing.
vllm_cpp_add_test(test_decode_graph_seam_g1_cuda vllm/models/test_decode_graph_seam_g1_cuda.cpp)
vllm_cpp_add_test(test_decode_graph_sizes vllm/models/test_decode_graph_sizes.cpp)
vllm_cpp_add_test(test_qwen35_paged_forward vllm/models/test_qwen35_paged_forward.cpp)
# GDN-MOE-BF16-OUT (#1168): the MoE forward's GDN output-activation dtype probe
# lives in the not-installed qwen3_5_internal.h, beside the other detail:: seams.
target_include_directories(test_qwen35_paged_forward PRIVATE ${CMAKE_SOURCE_DIR}/src)
# GDN-MOE-BF16-OUT (#1168), fresh-review repair. `GdnOutDType()` caches its
# getenv in a function-local static, so one process observes exactly ONE value of
# VT_GDN_OUT_BF16 and a single run cannot tell a resolver that READS the lever
# from one that was hardwired to its default answer — the mutation
# `return DType::kBF16;` left this binary 5/5 13/13. Re-run the SAME binary with
# the lever forced to 0, the shape `_glue_fuse_off` and
# `test_dense_gateup_fused_marlin_off_*` above already use. This is also the
# documented f32 rollback and the denominator of the row's same-binary A/B, so
# without this entry the arm an operator is told to run is ungated.
add_test(NAME test_qwen35_paged_forward_gdn_out_f32
  COMMAND test_qwen35_paged_forward)
set_tests_properties(test_qwen35_paged_forward_gdn_out_f32 PROPERTIES
  ENVIRONMENT "VT_GDN_OUT_BF16=0")
# GDN-MOE-PACKED-BA (#1169): the MoE safetensors loader builds the ONE merged
# `in_proj_ba` owner (rows [b; a], nk) through `LoadQwen3_5MoeLayer`, the
# production per-layer entry, over an in-memory resolver. Hermetic, CPU, <1s.
vllm_cpp_add_test(test_qwen35_moe_gdn_ba_owner vllm/models/test_qwen35_moe_gdn_ba_owner.cpp)
# MoE vision tower (#891): the tower is loaded (and its ABSENCE refused by name)
# and the forked VL forward actually CONSUMES the merger rows.
vllm_cpp_add_test(test_qwen3_5_moe_vision vllm/models/test_qwen3_5_moe_vision.cpp)
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
# UNALIGNED SAFETENSORS PAYLOADS (#772, fourth recurrence after #301/#627/#674).
# The four loaders that formed a `const uint16_t*` over an mmap'd tensor, driven
# over checkpoints whose payload parity is FORCED odd by one space of JSON header
# padding. Two of the four launder every access through `memcpy`, so UBSan cannot
# see them and only these value checks can — which is why this is a suite of its
# own and not another sanitizer sweep.
vllm_cpp_add_test(test_loader_unaligned_offsets
  vllm/models/test_loader_unaligned_offsets.cpp)
# The two Voxtral loader steps are declared in an INTERNAL header (src/, not
# include/): they are not public ABI and must not become so for a test's sake.
target_include_directories(test_loader_unaligned_offsets PRIVATE
  ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_qwen3_8_text_only vllm/models/test_qwen3_8_text_only.cpp)
# The PUBLISHED `Qwen/Qwen3.8-2.4T-A95B` config.json, committed VERBATIM, so the
# config-resolution case reads the real document (nested `rope_parameters`, 92
# `layer_types`, transformers-4.57.3 `dtype`) instead of a paraphrase of it.
# Its `model.safetensors.index.json` (1609 tensors, 213 shards, 4.89 TB) is
# committed the same way, so the load-plan dry run resolves every name against
# the real document rather than a summary of it (#740).
target_compile_definitions(test_qwen3_8_text_only PRIVATE
  QWEN3_8_CONFIG_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen3_8_2_4t_a95b/config.json"
  QWEN3_8_INDEX_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen3_8_2_4t_a95b/model.safetensors.index.json")
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
# SPEC-DFLASH2 W2 (#1314): the DFlash2 draft -- the published configs' nested
# spellings (spec `## Owed` O3/O4), the conv weights, and the draft forward that
# runs the grouped convolution.
vllm_cpp_add_test(test_qwen3_dflash2_draft vllm/models/test_qwen3_dflash2_draft.cpp)
target_include_directories(test_qwen3_dflash2_draft PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W5 (#1314): the GGUF DFlash2 drafter arm. The container the other
# suite cannot reach -- a GGUF declares no `architectures` and writes the same
# `general.architecture = dflash` a DFlash1 drafter writes -- plus the LOWER
# BOUND its k-quant arms need, because this lane dequantizes to bf16 by design
# and a token gate therefore cannot see a quantized tensor that never decoded.
vllm_cpp_add_test(test_qwen3_dflash2_gguf vllm/models/test_qwen3_dflash2_gguf.cpp)
target_include_directories(test_qwen3_dflash2_gguf PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
# #1205: the Gemma-4 device-expert upload must refuse where `vt::MatmulBTAlphaBeta`
# has no arm, entered through `vllm::RunGemma4Moe`. Its own binary: the layer's
# env knobs are frozen into function-local statics on the first call, so the arm
# it measures has to be decided before main() and cannot share a process with a
# case that wants a different one.
vllm_cpp_add_test(test_gemma4_moe_device_arm_guard
                  vllm/models/test_gemma4_moe_device_arm_guard.cpp)
vllm_cpp_add_test(test_gemma4_indexed_max_t vllm/models/test_gemma4_indexed_max_t.cpp)
target_compile_definitions(test_gemma4_indexed_max_t PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
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
# MODEL-DSV4-EXL3 W1b — the rank-sliced EXL3 loader arm: a hermetic four-rank
# fixture through the PRODUCTION loader entry, coalesced TP4->TP1 and checked
# byte-for-byte against the rank inputs, plus refuse-by-name for every arm the
# wave does not implement. See .agents/specs/model-dsv4-exl3.md.
vllm_cpp_add_test(test_deepseek_v4_exl3_loader vllm/models/test_deepseek_v4_exl3_loader.cpp)
target_include_directories(test_deepseek_v4_exl3_loader PRIVATE ${CMAKE_SOURCE_DIR}/src)
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

# ENG-MM-INPUT-PIPELINE L3 (#607): the TOWER SKIP. Shares the tiny synthetic
# Muse Glimmer checkpoint with the wiring gate above (muse_glimmer_tiny_fixture.h)
# so both ask their question of the same bytes.
vllm_cpp_add_test(test_tower_skip vllm/models/test_tower_skip.cpp)
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

# dots3-note W1+W2 STRUCTURAL gate (#699, .agents/specs/dots3-note.md §4 / §7):
# registry-resolve + config descent off the REAL released config.json, with one
# RED-first assertion per §4 CONFIG TRAP (n_group/topk_group=1,
# indexer_rope_interleave=True, num_nextn_predict_layers=1, the two lora
# rescales, the sliding rope's own theta and the GPT-J layout both ropes share),
# the on-disk name map accounted BOTH WAYS over the WHOLE released shard index
# (W2: all 38006 tensors -- 35381 language, 2195 vision, 430 audio -- from a
# committed headers-only projection of the index joined to all 133 safetensors
# headers, plus W1's 1614-tensor slice kept as a cross-check), the two tower
# files as NAMED W6/W7 deferrals asserted by count, the padded 1088-wide MLA
# row, and the load/GGUF/forward refusals
# — the last driven through the REAL model the factory returns, never a
# fabricated LoadedModel (#730/#784). Spec §6.4 records that NO oracle for this
# model runs on hardware this project owns, so these unit assertions are the
# only instrument the row has. CPU-only, no checkpoint, no speed claim.
vllm_cpp_add_test(test_dots3_note_scaffold vllm/models/test_dots3_note_scaffold.cpp)
target_compile_definitions(test_dots3_note_scaffold PRIVATE
  DOTS3_NOTE_CKPT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/dots3_note_prev")
# dots3_note.h is a MODEL-PRIVATE header under src/, not include/vllm/: W1 ships
# nothing on the public ABI and include/vllm/ is a USER_USAGE_PREFIXES surface
# (#515). Same arrangement test_nemotron_h_scaffold uses.
target_include_directories(test_dots3_note_scaffold PRIVATE ${CMAKE_SOURCE_DIR}/src)

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

# Nemotron-H WEIGHT LOADER gate (#517, .agents/specs/nemotron-h-model.md §5b and
# §6b "Still owed after W4"): materializes the 18487 released tensors from the
# REAL 20.1 GiB checkpoint and runs the registered forward on them. Gated
# STRUCTURALLY, not by tokens — every tensor materialized or deferred BY NAME,
# the per-scheme composition (5935 NVFP4 W4A16 g16 triples + 46 FP8 W8A8 static
# triples + the bf16/f32 remainder) asserted against the checkpoint's own memory
# format, and the loaded dtypes asserted to be the SHIPPED ones rather than
# wider. The checkpoint resolves through the CONTENT-pinned
# parity::Nemotron35LightningSnapshot (#569) and the gate SKIPS loudly when it is
# absent -- a NAMED, COUNTED skip with a closing accounting case, because a skip
# that prints `Passed 0.00 sec` with zero assertions is indistinguishable from a
# real pass -- so CI never needs the NAS asset. Not the W6 token gate and no
# speed claim: it consumes the committed oracle.json for ONE forward per prompt
# and the argmax of the last position against that prompt's FIRST token; the
# full 32-token greedy decode stays W6's.
vllm_cpp_add_test(test_nemotron_h_loader vllm/models/test_nemotron_h_loader.cpp)
target_compile_definitions(test_nemotron_h_loader PRIVATE
  NEMOTRON_H_GOLDENS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/parity/goldens/nemotron_35_lightning_greedy")
# nemotron_h.h / nemotron_h_loader.h are MODEL-PRIVATE headers under src/, not
# include/vllm/ (#515); tests/parity carries the content-pinned snapshot
# resolver.
target_include_directories(test_nemotron_h_loader PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# Nemotron-H MIXED_PRECISION MEMORY FORMS, OFFLINE (#517,
# .agents/specs/nemotron-h-model.md §6d). The loader's design claim — every
# weight held in the format it SHIPS in, widened only TRANSIENTLY at the GEMM
# call site — lived entirely behind the 20.1 GiB checkpoint gate above, so on a
# runner with no CHECKPOINT_ROOT it was not executed at all: deleting
# NemotronHOwned::View's refusal left the whole suite green. This gate needs NO
# checkpoint and therefore runs on EVERY arm. It asserts that View REFUSES a
# non-dense weight by name, and that DenseBf16 reproduces a dequant derived
# INDEPENDENTLY from the upstream ModelOpt formula — the nibble order, the
# per-16 group scale, the multiplied weight_scale_2, and the FP8 input_scale
# that is carried and NOT applied. Those are exactly §6d's M4/M5 defects, which
# leave every structural count correct. CPU-only, no golden, no speed claim.
vllm_cpp_add_test(test_nemotron_h_quantized_forms
  vllm/models/test_nemotron_h_quantized_forms.cpp)
# nemotron_h.h / nemotron_h_forward.h are MODEL-PRIVATE headers under src/, not
# include/vllm/ (#515).
target_include_directories(test_nemotron_h_quantized_forms PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

# A2-Q2a (#810): the DEVICE MoE arm on a SYNTHETIC NVFP4 fixture — the cheap RED
# in front of the real-checkpoint gate. It exists because `BuildTiny` in
# test_nemotron_h_forward.cpp is all-`kDense` (:892 -> PackMoe :873) and so
# cannot reach the NVFP4 arena at all; without it the 21 GB checkpoint would be
# the ONLY instrument and every RED would cost a GB10 window. Its geometry
# (hidden 128, moe_intermediate 64) is the smallest that resolves on Marlin's
# thread configs, which `TinyParams` (24 / 10) does not. Skips LOUDLY with a
# reason on a GPU-less box.
vllm_cpp_add_test(test_nemotron_h_moe_device
  vllm/models/test_nemotron_h_moe_device.cpp)
target_include_directories(test_nemotron_h_moe_device PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

# A2-P (#810, .agents/specs/nemotron-h-a2p-paged-forward.md): the PAGED forward,
# gated through the PRODUCTION entry point. Every case builds a real
# NemotronHLoadedModel from a synthetic checkpoint, hands it to a real
# GPUModelRunner, and drives execute_model / sample_tokens — so the paged
# forward is reached at runner.cpp:1465 through ModelRegistry::Forward, never
# through a fabricated ModelForwardInput. That is the difference AGENTS.md
# §"Nothing lands dead" is about: before this change NemotronHDeviceForward had
# ONE non-declaration call site in the whole tree and it was a test.
#
# The multi-step arm is the only one that can see a DROPPED CARRY (with one leg
# and fresh state the recurrent half is unobservable — nemotron_h_forward.h
# says so outright), the per-block numeric arm is the only one that can see a
# dropped mechanism whose argmax is unchanged, and the page dtypes are asserted
# directly because a too-WIDE page is numerically correct and invisible to both.
# CPU-only, no checkpoint, no speed claim: the A3 end-to-end token gate against
# the released checkpoint is the on-box binding gate.
vllm_cpp_add_test(test_nemotron_h_paged_forward
  vllm/models/test_nemotron_h_paged_forward.cpp)
target_include_directories(test_nemotron_h_paged_forward PRIVATE
  ${CMAKE_SOURCE_DIR}/src)

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
# The load-time MergeTokenOutOfVocabulary refusal asserts that every
# committed HF golden still loads, so test_bpe reaches the goldens dir.
target_compile_definitions(test_bpe PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
# SPEC-BPE-QUADRATIC-MERGE (#1365) — the long-pretoken equivalence corpus and
# the absolute cost bound. The corpus ids are HF `tokenizers` 0.22.2 output over
# the two committed goldens (an oracle, not a snapshot of ours); the cost case
# is the row's ONLY timing assertion and it is RED against the O(n^2) merge.
vllm_cpp_add_test(test_bpe_equivalence vllm/test_bpe_equivalence.cpp)
target_compile_definitions(test_bpe_equivalence PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
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
# MoE vision tower (#891) — the HARDWARE gate on Qwen/Qwen3.6-35B-A3B bf16: the
# 333 model.visual.* tensors load and the tower runs; the full ~67 GiB e2e is
# opt-in behind VLLM_MOE_VISION_E2E=1. Skipped without the checkpoint.
vllm_cpp_add_test(test_qwen3_5_moe_vision_hw vllm/multimodal/test_qwen3_5_moe_vl_hw.cpp)
target_compile_definitions(test_qwen3_5_moe_vision_hw PRIVATE
  MM_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/multimodal/fixtures/qwen3vl")
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
# SPEC-DSPARK-QWEN3-ROUTING (#1193): REACHABILITY of that classification — the
# loader's own ResolveSpecConfig reads the draft's config.json and refuses the
# DeepSeek-V4 DSpark lane by name, instead of trusting the CLI method string.
vllm_cpp_add_test(test_dspark_draft_routing
  vllm/entrypoints/test_dspark_draft_routing.cpp)
# SPEC-DFLASH2 W1 (#1314): REACHABILITY of the DFlash2 refusal — the loader
# reads the draft's own config.json and refuses `DFlash2DraftModel` by name,
# instead of drafting it through the DFlash1 lane with the grouped convolution
# and the candidate selector silently absent.
vllm_cpp_add_test(test_dflash2_draft_routing
  vllm/entrypoints/test_dflash2_draft_routing.cpp)
# #1160: `--speculative-config` key ADMISSION — an unknown or misspelled key is
# refused by name, and an upstream key this engine does not implement is refused
# with the missing part named, instead of being silently dropped.
vllm_cpp_add_test(test_speculative_unknown_keys
  vllm/config/test_speculative_unknown_keys.cpp)
# SPEC-DRAFTER-CHAIN W1 (#1522): the `vllm_cpp.drafter_chain` extension of
# --speculative-config -- its parsing, its validation and its refusals. The
# nesting under `vllm_cpp` is the same convention `--offload-config` already
# uses for this engine's non-vLLM keys, and it is what bounds the collision
# surface with vLLM's own `SpeculativeConfig` to exactly one name.
vllm_cpp_add_test(test_speculative_drafter_chain
  vllm/config/test_speculative_drafter_chain.cpp)
# SPEC-MTP-K-GT-1 (#81): the MTP speculation-DEPTH config contract — the k=1
# default, the upstream divisibility rule, and that a configured k above 1 is
# carried rather than silently degraded.
vllm_cpp_add_test(test_speculative_mtp_depth vllm/config/test_speculative_mtp_depth.cpp)
# ENG-MM-INPUT-PIPELINE L1 (#607): get_limit_per_prompt precedence —
# language_model_only ⇒ 0 BEFORE the map, else the map, else the 999 default.
vllm_cpp_add_test(test_multimodal_config vllm/config/test_multimodal_config.cpp)
vllm_cpp_add_test(test_offload_config vllm/config/test_offload_config.cpp)
# ENG-RESIDENCY-CONFIG (#1110): the `vllm_cpp` extension of --offload-config.
vllm_cpp_add_test(test_weight_residency_config
  vllm/config/test_weight_residency_config.cpp)
# ...and the expert-stream knob's own binary (#1122). `ResolveExpertStreamRequested`
# caches its answer in a function-local static, so a process can observe what it
# resolved exactly ONCE. That one observation is spent here, on the field the
# resolver reads — rewiring it to the adjacent `.mmap` left every other suite green.
vllm_cpp_add_test(test_expert_stream_latch
  vllm/config/test_expert_stream_latch.cpp)
# ...and its REACHABILITY: the loader and the C ABI install it. `vllm.h` is the
# public surface, so this suite needs src/ on the include path exactly as
# test_capi does (it reaches the internal MakeEngineHandle hook there; here it is
# the internal weight_residency header beside the public one).
vllm_cpp_add_test(test_weight_residency_reach
  vllm/entrypoints/test_weight_residency_reach.cpp)
target_include_directories(test_weight_residency_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
# `vllm-cli`'s `--offload-config` (#1135). It runs the BINARY, because that is the
# only thing that proves the FLAG arrives: `test_weight_residency_reach` already
# drives `vllm_engine_load` with the same JSON string, so a test that called the
# ABI here would re-prove the ABI. `$<TARGET_FILE:vllm-cli>` resolves at generate
# time, after every directory is processed; the build ORDER dependency is declared
# in the top-level CMakeLists, where the `vllm-cli` target already exists.
# Registered UNCONDITIONALLY, and with examples off it leaves `VLLM_CLI_BINARY`
# undefined: the suite then exits 77 and CTest reports it Skipped, which is what
# `SKIP_RETURN_CODE` above is for. Registering it only under the option would hide
# the gate's absence instead of reporting it.
vllm_cpp_add_test(test_cli_offload_config
  vllm/entrypoints/test_cli_offload_config.cpp)
if(TARGET vllm-cli OR VLLM_CPP_BUILD_EXAMPLES)
  target_compile_definitions(test_cli_offload_config
    PRIVATE VLLM_CLI_BINARY="$<TARGET_FILE:vllm-cli>")
endif()
vllm_cpp_add_test(test_weight_offloader vllm/model_executor/test_weight_offloader.cpp)
vllm_cpp_add_test(test_weight_offload_policy vllm/model_executor/test_weight_offload_policy.cpp)
vllm_cpp_add_test(test_expert_slot_cache vllm/model_executor/test_expert_slot_cache.cpp)
vllm_cpp_add_test(test_gguf_expert_span vllm/model_executor/test_gguf_expert_span.cpp)
vllm_cpp_add_test(test_expert_streamer vllm/model_executor/test_expert_streamer.cpp)
vllm_cpp_add_test(test_host_expert_slot_store vllm/model_executor/test_host_expert_slot_store.cpp)
# ENG-EXPERT-STREAM-DEVICE W1 (#1124), gate G1: the DEVICE slot store and the
# fill contract that makes it fillable. It needs no separate binary -- it
# registers nothing process-global, reads no environment latch, and drives its
# own `vt::Backend` instances by hand -- but it is its own file because the
# oracle it compares against is `HostExpertSlotStore`, so a suite that IS the
# host store cannot also be the comparison.
vllm_cpp_add_test(test_device_expert_slot_store
  vllm/model_executor/test_device_expert_slot_store.cpp)
# ENG-EXPERT-STREAM F3 (#912): the REACHABILITY gate, in its own binary because
# VT_MOE_EXPERT_STREAM is read ONCE into a function-local static, so one process
# cannot run both the streaming and non-streaming arms from the environment.
vllm_cpp_add_test(test_expert_stream_wiring vllm/model_executor/test_expert_stream_wiring.cpp)
# The lane's observability seam lives in the internal header beside the model,
# the same arrangement test_model_registry uses.
target_include_directories(test_expert_stream_wiring PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM F7 (#912): its own binary too, and specifically one that does
# NOT set VT_MOE_EXPERT_STREAM_SLOT_BYTES -- that override is what would hide the
# defect, and the store is built from the FIRST model a process runs.
vllm_cpp_add_test(test_expert_stream_mixed_slot vllm/model_executor/test_expert_stream_mixed_slot.cpp)
target_include_directories(test_expert_stream_mixed_slot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM-DEVICE W0c (#1124): the PLATFORM branch of the expert-slice
# seam. Its own binary for two independent reasons -- it latches
# VT_MOE_EXPERT_STREAM like its siblings, and it registers a fake backend and a
# fake platform in the process-global kXPU slot, which must not leak into any
# other suite.
vllm_cpp_add_test(test_expert_stream_device_slot vllm/model_executor/test_expert_stream_device_slot.cpp)
target_include_directories(test_expert_stream_device_slot PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM-DEVICE W0f (#1299): which branch `ResidentWeight` takes on a
# staging platform whose kernels can dereference host storage. Its own binary for
# the same reason as its sibling above -- it registers a fake backend and a fake
# platform in the process-global kXPU slot, which must not leak into any other
# suite -- and separately because the two disagree about the backend's
# UnifiedMemory(), which is a process-global registration and not a per-case flag.
vllm_cpp_add_test(test_resident_weight_host_addressable vllm/model_executor/test_resident_weight_host_addressable.cpp)
target_include_directories(test_resident_weight_host_addressable PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-EXPERT-STREAM (#1091): the step clock at the MoE entry points ForwardLayers
# does NOT cover, and the final statistics line. Its own binary because it is the
# only one that can observe `steps == 0` -- the store is a process-lifetime
# singleton, so a case asking that question must run before anything ends a step.
vllm_cpp_add_test(test_expert_stream_steps vllm/model_executor/test_expert_stream_steps.cpp)
target_include_directories(test_expert_stream_steps PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
vllm_cpp_add_test(test_host_buffer_staging vllm/v1/sample/test_host_buffer_staging.cpp)
vllm_cpp_add_test(test_logits_processor_host_addressable
  vllm/v1/sample/test_logits_processor_host_addressable.cpp)
vllm_cpp_add_test(test_prepare_inputs vllm/v1/worker/test_prepare_inputs.cpp)
vllm_cpp_add_test(test_combine_tokens vllm/v1/worker/test_combine_tokens.cpp)
vllm_cpp_add_test(test_async_output vllm/v1/worker/test_async_output.cpp)
vllm_cpp_add_test(test_async_runner_flag vllm/v1/worker/test_async_runner_flag.cpp)
vllm_cpp_add_test(test_runner vllm/v1/worker/test_runner.cpp)
# test_runner uses detail::ValidateGdnStateIndices from the internal header.
target_include_directories(test_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_common_attn_metadata vllm/v1/attention/test_common_attn_metadata.cpp)
vllm_cpp_add_test(test_attn_backend_registry vllm/v1/attention/test_attn_backend_registry.cpp)
vllm_cpp_add_test(test_attn_validate_configuration
  vllm/v1/attention/test_attn_validate_configuration.cpp)
# SPEC-DFLASH2 W10 (#1857): the spec-as-decode reorder-threshold policy mirror.
vllm_cpp_add_test(test_spec_as_decode vllm/v1/attention/test_spec_as_decode.cpp)
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
# SPEC-MTP-K-GT-1 (#81): MTP speculation DEPTH through the production loader —
# the configured k reaches the propose path, or is refused by name.
vllm_cpp_add_test(test_mtp_depth vllm/v1/spec_decode/test_mtp_depth.cpp)
# ENG-CUDAGRAPH-BREAK W6 (#1374): the graph-eligibility predicate and its
# counters live under src/, beside the runner that computes them, exactly as
# test_cudagraph_dispatch reads them (:561).
target_include_directories(test_mtp_depth PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_rejection_sampler vllm/v1/spec_decode/test_rejection_sampler.cpp)
vllm_cpp_add_test(test_prepare_prefill_inputs vllm/v1/spec_decode/test_prepare_prefill_inputs.cpp)
# SPEC-MTP-K-GT-1 (#81): the draft-DECODE half of the autoregressive propose —
# the step-1 entry state, the between-step advance, and both max_model_len clamps.
vllm_cpp_add_test(test_prepare_decode_inputs vllm/v1/spec_decode/test_prepare_decode_inputs.cpp)
vllm_cpp_add_test(test_dflash_kvprep vllm/v1/spec_decode/test_dflash_kvprep.cpp)
target_include_directories(test_dflash_kvprep PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W1 (#1314): the DFlash causality rule — the port of vLLM's
# tests/v1/spec_decode/test_dflash_causality.py as edited by vllm#52816, plus the
# assertion that a DFlash1 checkpoint's resolution is unchanged.
vllm_cpp_add_test(test_dflash_causality vllm/v1/spec_decode/test_dflash_causality.cpp)
vllm_cpp_add_test(test_dflash_propose vllm/v1/spec_decode/test_dflash_propose.cpp)
# SPEC-DFLASH2 W4 (#1314): the INVERSE guard, once the walk lands and there is
# no missing mechanism left to refuse -- a DFlash2 block that reaches the DFlash1
# per-slot argmax throws by name, and the walk's own refusals are gated beside
# it. W1/W2/W3 read "the candidate-selector refusal" here, which was the boundary
# the architecture was left at while its selector was unported.
vllm_cpp_add_test(test_dflash2_argmax_guard
  vllm/v1/spec_decode/test_dflash2_argmax_guard.cpp)
# SPEC-DFLASH2 W4 (#1314): reachability (spec G5) -- a real LoadedEngine through
# GPUModelRunner::propose_drafts_block, with the drafts read off the production
# trace.
vllm_cpp_add_test(test_dflash2_runner_reach
  vllm/v1/spec_decode/test_dflash2_runner_reach.cpp)
# W10 (#1857): the spec-as-decode classification case reads the graph-dispatch
# counters, whose header lives in the src tree (same need as test_mtp_depth).
target_include_directories(test_dflash2_runner_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
# SPEC-DFLASH2 W9 (#1849): the VT_SPEC_TRACE=2 device-segment trace of the
# DFlash2 draft phase. Its OWN binary because the trace level is latched by a
# function-local static on the first propose, so one process observes one level
# and the reach binary above latches "1".
vllm_cpp_add_test(test_dflash2_draft_phase_trace
  vllm/v1/spec_decode/test_dflash2_draft_phase_trace.cpp)
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
# KV-FP8 W3 (#1593): half-sized KV blocks, --kv-cache-dtype threading and the
# checkpoint k/v-scale path, entered through the LoadedEngine loader.
vllm_cpp_add_test(test_kv_cache_fp8_wiring vllm/entrypoints/test_kv_cache_fp8_wiring.cpp)
vllm_cpp_add_test(test_dspark_block_size_guard vllm/entrypoints/test_dspark_block_size_guard.cpp)
# ENG-HF-MODEL-DOWNLOAD W2 (#1280): the HuggingFace cache walk entered through
# the production loader, with a repository identifier rather than a path. The
# transformers_utils suite calls the function directly and cannot see a deleted
# call site; this one can. Named `dflash` so `ctest -R "dflash|dspark"` runs it.
vllm_cpp_add_test(test_dflash_draft_hf_cache vllm/entrypoints/test_dflash_draft_hf_cache.cpp)
# SPEC-DRAFTER-CHAIN W1 (#1522): the REACHABILITY and ORDERING half. W1 stores a
# field nothing resolves yet, so the one way it could land dead is a parsed chain
# no production caller reads. `ResolveSpecConfig` is that reader and it refuses by
# name; `FromModelDir` runs the refusal ahead of the path resolution, which is what
# G5's "before any weight I/O" means. Needs src/ on the include path for the same
# reason `test_weight_residency_reach` does: it drives the public `vllm.h` ABI.
vllm_cpp_add_test(test_drafter_chain_reach
  vllm/entrypoints/test_drafter_chain_reach.cpp)
target_include_directories(test_drafter_chain_reach PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
  # ENG-HF-MODEL-DOWNLOAD W2 (#1280): the hub protocol against an in-process fake
  # hub. It starts a REAL cpp-httplib server on an ephemeral port, so it shares
  # api_server.cpp's gate and the RUN_SERIAL treatment two rows below.
  vllm_cpp_add_test(test_hf_hub vllm/transformers_utils/test_hf_hub.cpp)
  # ENG-HF-MODEL-DOWNLOAD W3/W4 (#1280): the byte transport and the `--model`
  # grammar, both against an in-process fake hub on an ephemeral port. Same
  # gate and same RUN_SERIAL treatment as the suite above.
  vllm_cpp_add_test(test_downloader vllm/transformers_utils/test_downloader.cpp)
  vllm_cpp_add_test(test_model_resolver
    vllm/transformers_utils/test_model_resolver.cpp)
  # ENG-HF-MODEL-DOWNLOAD W5 (#1280): the transport-layer-security suite. It is
  # the ONE place in this row that speaks https, and the one place that measures
  # whether the server half and the fetcher half were compiled against the same
  # cpp-httplib layout. Its TLS arm starts a real `httplib::SSLServer` on
  # loopback, so it shares the RUN_SERIAL treatment below.
  vllm_cpp_add_test(test_tls_transport
    vllm/transformers_utils/test_tls_transport.cpp)
  # THE EXPECTATION, computed on a path that does NOT read the macro under test.
  # `VLLM_CPP_TEST_EXPECT_TLS` says "the user asked for the download feature and
  # this host has the OpenSSL development files", which is a fact about the
  # CONFIGURATION. The suite then asserts that the BUILD delivered transport
  # layer security. Deriving the expectation from `CPPHTTPLIB_OPENSSL_SUPPORT`
  # instead would make the case a tautology that a silently disabled feature
  # passes, which is the exact failure `scripts/validate-container-image.py`
  # exists to catch in the container.
  if(VLLM_CPP_HF_DOWNLOAD_REQUESTED AND VLLM_CPP_OPENSSL)
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND)
      target_compile_definitions(test_tls_transport PRIVATE
        VLLM_CPP_TEST_EXPECT_TLS=1)
    endif()
  endif()
  vllm_cpp_add_test(test_openai_api_server vllm/entrypoints/openai/test_api_server.cpp)
  # /v1/audio/transcriptions dispatch + socket smoke run against the REAL
  # library transcription seam on the committed parakeet_e2e fixture
  # (ARCH-ONE-SURFACE ROW 1).
  target_compile_definitions(test_openai_api_server PRIVATE
    PARAKEET_E2E_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/parakeet_e2e"
    # /v1/embeddings dispatch + socket smoke run against the REAL engine path
    # (LoadedEngine -> PoolingRunner) on the committed llama_embed_e2e fixture
    # (ARCH-ONE-SURFACE ROW 6).
    LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e"
    # #1681: the REAL published Qwen3.8 chat template, rendered through the
    # production /v1/chat/completions dispatch. Absolute so it resolves from
    # whatever working directory ctest runs the suite in.
    VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
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
  set_tests_properties(test_openai_api_server test_openai_conformance test_hf_hub
    test_downloader test_model_resolver test_tls_transport
    PROPERTIES RUN_SERIAL ON)
  # MODEL-MUSIC3 (#672): the end-to-end suite. It CONSTRUCTS an `ApiServer`
  # and drives its speech routes over a real socket, so it links the
  # translation unit `CMakeLists.txt` compiles only inside this guard. It was
  # registered outside it until #1883, which made every
  # `-DVLLM_CPP_SERVER=OFF` configuration configure cleanly and then fail to
  # link; `scripts/check-test-registration.py` now refuses that shape.
  vllm_cpp_add_test(test_minimax_music3_e2e_real parity/test_minimax_music3_e2e_real.cpp)
  target_compile_definitions(test_minimax_music3_e2e_real PRIVATE
    MUSIC3_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens/minimax_music3_oracle"
    # Where the e2e case writes the WAV it generated, so a human can listen to it.
    # The BUILD tree, never the source tree: it is an artifact of a run and not a
    # golden, and nothing under tests/parity/goldens/ is created or replaced.
    BUILD_ARTIFACT_DIR="${CMAKE_BINARY_DIR}/music3")
  target_include_directories(test_minimax_music3_e2e_real PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/parity)
  # This suite binds a REAL cpp-httplib server on an ephemeral port (the
  # speech-only route-table cases, #672), which is the same reason
  # test_openai_api_server and test_openai_conformance carry RUN_SERIAL above:
  # under a saturated `ctest -jN` the server's accept thread starves and the
  # client sees a connection failure rather than the defect it was looking for.
  set_tests_properties(test_minimax_music3_e2e_real PROPERTIES RUN_SERIAL ON)

  # SERVE-RECIPE-ARGS (#606): the accepted-and-inert serve-argument seam. It
  # drives the REAL VllmServerMain (owned by the library behind the same
  # VLLM_CPP_SERVER gate) by re-execing this binary, so it needs src/ on the
  # include path for server_main.h, exactly as test_openai_api_server does.
  vllm_cpp_add_test(test_serve_recipe_args
    vllm/entrypoints/openai/test_serve_recipe_args.cpp)
  target_include_directories(test_serve_recipe_args PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # ENG-HF-MODEL-DOWNLOAD W4 (#1280): THE REACHABILITY GATE. It enters at the
  # REAL VllmServerMain with `--model org/repo` and an in-process fake hub, so
  # it needs src/ on the include path for server_main.h, exactly as the suite
  # above does, and the committed llama_embed_e2e checkpoint as the payload the
  # hub serves. It binds a port and forks a server, so it is RUN_SERIAL for the
  # same starvation reason as test_openai_api_server.
  vllm_cpp_add_test(test_serve_hf_model
    vllm/entrypoints/openai/test_serve_hf_model.cpp)
  target_include_directories(test_serve_hf_model PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(test_serve_hf_model PRIVATE
    VLLM_LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e")
  set_tests_properties(test_serve_hf_model PROPERTIES RUN_SERIAL ON)
  # ENG-MM-INPUT-PIPELINE wave L2 (#607): --language-model-only /
  # --limit-mm-per-prompt. Same shape and the same reason as the row above — it
  # re-execs the REAL VllmServerMain to observe the flags reaching the config and
  # a malformed limit aborting before the load.
  vllm_cpp_add_test(test_serve_mm_limits
    vllm/entrypoints/openai/test_serve_mm_limits.cpp)
  target_include_directories(test_serve_mm_limits PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # ENG-RESIDENCY-CONFIG (#1110): --offload-config's vllm_cpp key reaching the
  # loader through the REAL VllmServerMain. Same include path for server_main.h.
  vllm_cpp_add_test(test_serve_residency_config
    vllm/entrypoints/openai/test_serve_residency_config.cpp)
  target_include_directories(test_serve_residency_config PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # KV-FP8 W3 (#1593): --kv-cache-dtype reaching EngineParams through the REAL
  # VllmServerMain. Same shape and the same reason as the row above.
  vllm_cpp_add_test(test_serve_kv_cache_dtype
    vllm/entrypoints/openai/test_serve_kv_cache_dtype.cpp)
  target_include_directories(test_serve_kv_cache_dtype PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
  LLAMA_EMBED_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/vllm/models/fixtures/llama_embed_e2e"
  # #1681: the real published Qwen3.8 chat template, rendered through the C
  # ABI's own vllm_chat path. Absolute so it resolves from whatever working
  # directory ctest runs the suite in.
  VLLM_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
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
# The head_dim-128 decode arm (#382) ships default OFF, and its flag is read into
# a `static const bool` -- once per process. The default registration above
# therefore only ever gates the PagedAttnOnline fallback; the arm that runs the new
# kernel needs its own INVOCATION with the flag set. Same shape as the
# test_dense_gateup_fused_marlin_off_* lever pair above. The filter avoids commas:
# doctest reads them as filter separators and the case name contains three.
add_test(NAME test_backend_cross_device_vt_attn_decode_d128
  COMMAND test_backend_cross_device "--test-case=*head_dim 128*")
set_tests_properties(test_backend_cross_device_vt_attn_decode_d128 PROPERTIES
  ENVIRONMENT "VT_ATTN_DECODE_D128=1" SKIP_RETURN_CODE 77)
# The ROCm GCN-arch capability parse (BACKEND-ROCM W0). UNCONDITIONAL by design:
# include/vt/rocm/rocm_arch.h is deliberately free of HIP headers so the one part
# of that skeleton carrying a DECISION is gated on a CPU-only CI runner with no
# AMD GPU. The rest of the ROCm skeleton is under VLLM_CPP_HIP below.
vllm_cpp_add_test(test_rocm_arch vt/test_rocm_arch.cpp)
# #785 P1 host package (fixture/oracle/trace classifier). CPU-only.
vllm_cpp_add_test(test_ops_paged_attn_sharedk_wmma_p1
  vt/test_ops_paged_attn_sharedk_wmma_p1.cpp)
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
# #785 P1 GPU product-seam witness. Executable only — NOT add_test.
# Ordinary CTest must not see this target. Runner fail-closes on 77/nonzero.
add_executable(test_ops_paged_attn_sharedk_wmma_p1_gpu
  vt/test_ops_paged_attn_sharedk_wmma_p1_gpu.cpp)
target_include_directories(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR})
if(TARGET vllm::sanitize_test_shared)
  target_link_libraries(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
    vllm::sanitize_test_shared vllm_test_main)
else()
  target_link_libraries(test_ops_paged_attn_sharedk_wmma_p1_gpu PRIVATE
    vllm::vllm vllm_test_main)
endif()
vllm_cpp_set_warnings(test_ops_paged_attn_sharedk_wmma_p1_gpu)
endif()
if(VLLM_CPP_METAL)
  vllm_cpp_add_test(test_metal_backend vt/test_metal_backend.cpp)
  if(VLLM_CPP_MLX)
    # THE #1584 EXACTNESS CASE IN ITS OWN PROCESS (issue #1692 O2). The case
    # asserts `declines == 1` on the FIRST decline of the process, so it only
    # measures anything while `MlxFallback`'s function-local static is cold. In
    # the full-suite run it is first by line order and therefore cold, but that
    # is a property of where it sits in the file; a dedicated process makes it a
    # property of the gate. This is O3 of the same row
    # (https://github.com/mudler/vllm.cpp/issues/1812) applied to the Metal arm,
    # in the shape `test_max_abs_diff_nan_raises` above already uses.
    #
    # `--test-case=` splits its argument on COMMAS, so the case name deliberately
    # contains none. A filter that matches nothing prints "0 cases ran" and exits
    # 0, so the macos-metal-mlx CI job additionally asserts the `test cases:` line
    # rather than trusting this entry's exit status alone.
    # QUOTED, and that is not cosmetic: unquoted, CMake splits the argument on
    # its spaces into `--test-case=*EXACTLY`, `one` and `decline*`, doctest
    # matches no case, prints `0 cases ran` and `Status: SUCCESS!`, and ctest
    # reports the entry green forever. The macos-metal-mlx job asserts the
    # `test cases: 1 | 1 passed` line out of a `ctest -V` run for exactly this.
    add_test(NAME test_metal_backend_mlx_decline_exact
      COMMAND test_metal_backend "--test-case=*EXACTLY one decline*")
    set_tests_properties(test_metal_backend_mlx_decline_exact
      PROPERTIES SKIP_RETURN_CODE 77)
  endif()
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
# #839 host lifetime + source invariants (no GPU).
vllm_cpp_add_test(test_ops_gemma4_prefill_peer vt/test_ops_gemma4_prefill_peer.cpp)
target_compile_definitions(test_ops_gemma4_prefill_peer PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
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
# MODEL-DSV4-EXL3 W1a: the EXL3 (exllamav3) trellis reference dequant. Gated by
# tiles the TEST packs itself off the ENCODE side (tests/test_quant_fn.py:83-128
# + pack.cu:29-57), so the fixture never shares code with the decode under test,
# plus a dense double-precision blockwise-Hadamard reference and a real
# rank-sliced DeepSeek-V4 expert linear (skips when the shard is not mounted).
vllm_cpp_add_test(test_exl3_dequant vt/test_exl3_dequant.cpp)
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
# LTX25-TEXT-LINEAR-MEM (#1286): the seam's ALLOCATION bound, on both
# `MatmulOneChunk` instantiations. Every other GEMM gate here asserts values or
# byte equality, so a kernel that allocated a whole intermediate per worker
# would pass all of them. It counts bytes REQUESTED through replaced global
# `operator new` overloads, not peak RSS -- the peak-RSS draft read zero at
# every geometry while passing every bound, because glibc retains the arena
# (spec `## Outcome` W4).
#
# Registered only on Linux, and NOT because of /proc or the threadpool: the
# counter reads nothing under /proc and the pool is portable. It is that the
# replaced allocation functions serve the WHOLE PROGRAM, so the gate depends on
# every `delete` in the binary reaching the matching replaced `delete`, and on
# its fixed slack term being calibrated against the library doing the
# allocating. Both hold for glibc/libstdc++ and are unestablished elsewhere; a
# standard library linked as its own runtime keeps its own `operator new`, and
# the mismatch is a heap corruption rather than a red assertion. Compiling it
# elsewhere with a skip would print `Status: SUCCESS!` over `assertions: 0`,
# which reads as a pass. Extending the lanes is owed, not a one-line change.
# Reaches src/ for the threadpool test hook, which is what lets one process
# sweep the worker count.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  vllm_cpp_add_test(test_ops_matmul_mem vt/test_ops_matmul_mem.cpp)
  target_include_directories(test_ops_matmul_mem PRIVATE ${CMAKE_SOURCE_DIR}/src)
  # The C allocator is counted through the LINKER, not through a symbol
  # definition. `--wrap` redirects the calls made by the objects in this link --
  # which includes libvllm.a, and so cpu_ops.cpp -- while leaving `malloc` itself
  # defined by whoever normally defines it. Defining `malloc` in the test would
  # instead be a second strong definition beside AddressSanitizer's own
  # interceptor, which is what would break the sanitize-cpu lane. Scoped to this
  # one target, so nothing else in the suite is affected.
  #
  # These flags are LOAD-BEARING, not an optimization: the test references
  # `__real_malloc` directly, so dropping one fails the LINK rather than quietly
  # costing the gate its malloc coverage. Verified by mutation (M-W).
  target_link_options(test_ops_matmul_mem PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc
    -Wl,--wrap=aligned_alloc
    -Wl,--wrap=posix_memalign)
endif()
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
# CPU-tier contract for the cuBLASLt heuristic-result cache's pure plumbing
# (VT_GEMM_PLAN_CACHE flag, DEFAULT ON, + the GemmPlanKey equality/hash
# completeness; issue #1732: the per-call heuristic query returns status 14
# inside CUDA graph capture, so the cache is the fix and its default is ON).
# Reaches the internal header under src/ (not public API); the cached
# cublasLtMatmulHeuristicResult_t it keys is CUDA-only and lives in
# src/vt/cuda/cuda_matmul.cu; the end-to-end proof is the row's board gate.
vllm_cpp_add_test(test_gemm_plan_cache vt/test_gemm_plan_cache.cpp)
target_include_directories(test_gemm_plan_cache PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
# CPU-tier REDUCTION-ORDER contract for the single-warp MoE router top-k
# (VT_MOE_ROUTER_WARP, default ON; spec moe-router-topk-single-warp.md, #378).
# The claim is that the lane map `expert = lane + 32*slot` plus a per-lane
# halving tree performs the bit-identical float additions as the incumbent
# 256-thread block reduction in src/vt/cuda/cuda_moe.cu -- pure IEEE-754, so it
# is provable with no GPU and no nvcc. Reaches the CUDA-free map/tree/selector
# header under src/; the executing kernel is covered on-device by the
# parallel-vs-serial byte-exactness case in test_ops_moe_grouped.
vllm_cpp_add_test(test_moe_router_warp_map vt/test_moe_router_warp_map.cpp)
target_include_directories(test_moe_router_warp_map PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
vllm_cpp_add_test(test_breakable_graph vt/test_breakable_graph.cpp)
vllm_cpp_add_test(test_persistent_step_input vt/test_persistent_step_input.cpp)
vllm_cpp_add_test(test_graph_safe_scratch vt/test_graph_safe_scratch.cpp)
target_include_directories(test_graph_safe_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_graph_dedup vt/test_graph_dedup.cpp)
target_include_directories(test_graph_dedup PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-CUDAGRAPH-DEDUP / #1184: the two device-free halves of the CUDA/HIP binding --
# the error-latch guard every entry point installs, and the structural-signature walk.
# Both lived inside a header that includes <cuda_runtime.h> and were therefore reached
# by NO test on any tier; the latch defect hid in exactly that gap.
vllm_cpp_add_test(test_graph_dedup_runtime vt/test_graph_dedup_runtime.cpp)
target_include_directories(test_graph_dedup_runtime PRIVATE ${CMAKE_SOURCE_DIR}/src)
# ENG-CUDAGRAPH-DEDUP / #1226: the coarse-key knob itself. vt::GraphDedupCoarseKeyEnabled
# caches its answer in a function-local static -- deliberately, because the drivers
# capture lazily and a knob that moved between two captures would give one model a mixed
# set of handles -- so ONE PROCESS CAN OBSERVE ONE VALUE. Proving that "1" enables the
# coarse key and "10" does not therefore takes one process per value, which is what these
# entries are. Without them the accessor's only coverage was a single unset-variable
# CHECK_FALSE, which two mutations survived: accepting any non-null value, and reading a
# completely different variable name.
vllm_cpp_add_test(test_graph_dedup_coarse_env vt/test_graph_dedup_coarse_env.cpp)
target_include_directories(test_graph_dedup_coarse_env PRIVATE ${CMAKE_SOURCE_DIR}/src)
function(vllm_cpp_add_coarse_key_case suffix value expect)
  add_test(NAME test_graph_dedup_coarse_env_${suffix}
    COMMAND test_graph_dedup_coarse_env)
  set_tests_properties(test_graph_dedup_coarse_env_${suffix} PROPERTIES
    ENVIRONMENT "VT_CUDA_GRAPH_DEDUP_COARSE_KEY=${value};VT_COARSE_KEY_EXPECT=${expect}"
    SKIP_RETURN_CODE 77)
endfunction()
# The one value that turns it on. This entry is what proves the accessor reads THIS
# variable: a mutation that reads any other name leaves it unset and fails here.
vllm_cpp_add_coarse_key_case(on "1" 1)
# ...and the near misses, each of which enables the key under a `value != nullptr` test.
# "10" is a hand-edited "1" plus a stray keystroke, and it turning a default-off
# correctness-sensitive path on by accident is the failure the polarity exists to stop.
vllm_cpp_add_coarse_key_case(off_zero "0" 0)
vllm_cpp_add_coarse_key_case(off_ten "10" 0)
vllm_cpp_add_coarse_key_case(off_oh_one "01" 0)
vllm_cpp_add_coarse_key_case(off_one_x "1x" 0)
vllm_cpp_add_coarse_key_case(off_true "true" 0)
vllm_cpp_add_coarse_key_case(off_empty "" 0)
# The knob is DEFINED but the OTHER knob is the one set. The two are separate variables
# and the coarse key must not piggyback on VT_CUDA_GRAPH_DEDUP: a run that asked only for
# dedup must get the exact key the 2026-08-18 device gate ran, or the A/B is not one.
add_test(NAME test_graph_dedup_coarse_env_off_sibling_knob
  COMMAND test_graph_dedup_coarse_env)
set_tests_properties(test_graph_dedup_coarse_env_off_sibling_knob PROPERTIES
  ENVIRONMENT "VT_CUDA_GRAPH_DEDUP=1;VT_COARSE_KEY_EXPECT=0"
  SKIP_RETURN_CODE 77)
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
# VT-FP8-W8A8-CPU-ARM (#468): the CPU arm of the static fp8 W8A8 path. Runs on a
# box with NO GPU by construction — that is the point of the row — and carries a
# CUDA-gated arm (G2) for CPU-vs-CUDA byte agreement wherever a device exists.
vllm_cpp_add_test(test_ops_fp8_cpu vt/test_ops_fp8_cpu.cpp)
# VT-QUANT-FP8-GROUP (#1189 M1): the dynamic per-token per-group fp8 activation
# quant. CPU-gateable by construction; its CPU-vs-CUDA byte-identity arm (G6) is
# CUDA-gated and reports PENDING rather than skipping where no device exists.
vllm_cpp_add_test(test_ops_quant_fp8_group_cpu vt/test_ops_quant_fp8_group_cpu.cpp)
# VT-MATMUL-FP8-BLOCK-REF (#1189 M2): the 128x128 block-scaled fp8 GEMM, CPU
# reference arm — the numerical oracle every later block-fp8 kernel is measured
# against. Its G4 holds the constraint the row exists for: the scales apply in
# the MAINLOOP, once per K-block, which an epilogue-folded alpha cannot express.
vllm_cpp_add_test(test_ops_matmul_fp8_block_cpu vt/test_ops_matmul_fp8_block_cpu.cpp)
# VT-MATMUL-FP8-BLOCK-CUDA (#1189 M5): the HOST-SIDE contract of the same op's
# CUDA arm — the tile-config heuristic, the refusal predicate, the two scale
# layouts cutlass DEDUCES from the problem shape, and the dispatch counter. Runs
# on every machine including one with no GPU and no CUDA toolkit, which is the
# whole point: those four are the parts of an sm_12xa-only kernel that can be
# wrong in a way a successful compile cannot see. Reaches the internal header
# under src/ (not public API), exactly like its test_fp8_plan_cache neighbour.
vllm_cpp_add_test(test_fp8_block_scaled_dispatch vt/test_fp8_block_scaled_dispatch.cpp)
target_include_directories(test_fp8_block_scaled_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# PERF-FP8-SMALL-M-DISPATCH (#1866): the PER-TENSOR fp8 GEMM's sm120 tile ladder,
# host tier. Same arrangement and the same reason as the blockwise neighbour
# above -- a wrong tile is a SLOW answer, so no correctness suite in this tree
# can see it and it has to be gated by value on a host with no GPU.
vllm_cpp_add_test(test_fp8_per_tensor_dispatch vt/test_fp8_per_tensor_dispatch.cpp)
target_include_directories(test_fp8_per_tensor_dispatch PRIVATE ${CMAKE_SOURCE_DIR}/src)
# VT-MATMUL-FP8-BLOCK-CUDA (#1189 M5), the DEVICE tier: the CUDA arm against the
# CPU reference arm of the SAME op, carrying upstream's own
# test_w8a8_block_fp8_cutlass_matmul whole (M=32, N=576, K=7168, rel_diff <
# 0.001). IT HAS NEVER RUN: no GPU was leased for that row and it skips without a
# device. The row's spec lists the on-hardware leg under `## Owed`; a skip here
# is not a pass and must not be reported as one.
vllm_cpp_add_test(test_ops_matmul_fp8_block_cuda vt/test_ops_matmul_fp8_block_cuda.cpp)
target_include_directories(test_ops_matmul_fp8_block_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)
# fp8 plan-cache arms, re-pinned by FIX-FP8-PLAN-CAPTURE-1843: the cache is
# DEFAULT ON (on CUDA 13.3 the uncached lane queries the cuBLASLt heuristic
# inside CUDA-graph capture and dies — #1843), so the cache-ON arm of the
# byte-exact case is the DEFAULT arm (no env: the first MatmulFp8CublasLt call
# on a shape builds the plan fresh, later calls hit the cache, and the
# first-vs-later byte compare is fresh-plan == cached-plan). The rollback arm
# pins the VT_FP8_PLAN_CACHE=0 escape hatch (every call rebuilds fresh — the
# same-binary A/B arm), proving it stays byte-stable across repeats.
add_test(NAME test_ops_fp8_cutlass_plan_cache_default_on
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM")
add_test(NAME test_ops_fp8_cutlass_plan_cache_rollback_off
  COMMAND test_ops_fp8_cutlass
    "--test-case=fp8 cuBLASLt cached-plan GEMM is BYTE-identical to the fresh-plan GEMM")
set_tests_properties(test_ops_fp8_cutlass_plan_cache_rollback_off PROPERTIES
  ENVIRONMENT "VT_FP8_PLAN_CACHE=0")
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
# Reaches the CUDA-free VT_MOE_ROUTER_WARP selector header under src/, through
# the shared test-only pin tests/vt/moe_router_warp_env.h: the parallel-vs-serial
# router case PINS that toggle both ways and ASSERTS the pinned state, so it can
# no longer degenerate to block-vs-serial when the ambient environment says 0
# (spec moe-router-topk-single-warp.md §9, #378). Same include pattern as
# test_moe_router_warp_map above, which executes that pin without a GPU.
target_include_directories(test_ops_moe_grouped PRIVATE ${CMAKE_SOURCE_DIR}/src)
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
# SPEC-DFLASH2 W2 (#1314): the DFlash2 grouped dynamic depthwise convolution,
# gated against upstream's own sequential reference at block 5 (modulo arm), 8
# and 16 (the two published checkpoints) plus CUDA==CPU bit-identity.
vllm_cpp_add_test(test_ops_dflash2_grouped_conv vt/test_ops_dflash2_grouped_conv.cpp)
vllm_cpp_add_test(test_ops_topk_values_indices vt/test_ops_topk_values_indices.cpp)
vllm_cpp_add_test(test_ops_dflash2_selector_edges vt/test_ops_dflash2_selector_edges.cpp)
# SPEC-DFLASH2 W4 (#1314): the candidate selector's PATH WALK -- one program per
# request with the slot-to-slot dependency as a loop inside it, gated against
# upstream's own `_selector_walk_kernel` control flow plus CUDA==CPU bit-exactness.
vllm_cpp_add_test(test_ops_dflash2_path_walk vt/test_ops_dflash2_path_walk.cpp)
vllm_cpp_add_test(test_ops_reshape_cache vt/test_ops_reshape_cache.cpp)
# KV-FP8 W1: fp8 KV-cache store (ReshapeAndCacheFp8) + paged-attention read
# dequant + the ParseCacheDType config wiring.
vllm_cpp_add_test(test_ops_fp8_kv_cache vt/test_ops_fp8_kv_cache.cpp)
# KV-FP8 W2 (#1593): the CUDA fp8-e4m3 KV store + the fp8 dequant on the CUDA
# paged-attention read, gated for parity against the W1 CPU reference. The
# device cases skip with a MESSAGE when no CUDA backend is present; the
# provider-routing case runs on every leg.
vllm_cpp_add_test(test_cuda_fp8_kv_cache vt/test_cuda_fp8_kv_cache.cpp)
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
# BigVGAN / DAC vocoder 1-D convolutions (#672): vt::Conv1d + vt::ConvTranspose1d
# against a verbatim copy of the pre-op vocoder1d host loop, plus a CPU-vs-CUDA
# BYTE-equality arm that skips loudly on a CPU box. The four consumers' own
# end-to-end bit-identity gate is tests/vllm/models/test_host_parallel.cpp.
vllm_cpp_add_test(test_ops_conv1d_general vt/test_ops_conv1d_general.cpp)
target_include_directories(test_ops_conv1d_general PRIVATE ${CMAKE_SOURCE_DIR}/src)
# General 3-D convolution (LTX25-DEVICE-RESIDENCY W5, #1007): vt::Conv3d against
# an independent in-test scalar reference, plus the case that proves its
# per-input-channel partial is a DIFFERENT number from kConv2d's flat
# accumulator. The LTX-2.5 decode's own byte-identity gate is
# tests/vllm/models/test_ltx2_vae.cpp.
vllm_cpp_add_test(test_ops_conv3d vt/test_ops_conv3d.cpp)
target_include_directories(test_ops_conv3d PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_attn_relpos vt/test_ops_attn_relpos.cpp)
target_include_directories(test_ops_attn_relpos PRIVATE ${CMAKE_SOURCE_DIR}/src)
vllm_cpp_add_test(test_ops_paged_attn vt/test_ops_paged_attn.cpp)
# SPEC-DFLASH2 W10 (#1857): the host lane-classification seam of the CUDA
# paged-attention dispatch (include/vt/paged_attn_route.h).
vllm_cpp_add_test(test_paged_attn_route vt/test_paged_attn_route.cpp)
# PERF-CPU-ATTN-DTYPE: the hoisted dtype dispatch vs the pre-change per-element
# switch, memcmp over every dtype the switch handled (spec §3, issue #391).
vllm_cpp_add_test(test_ops_paged_attn_dtype vt/test_ops_paged_attn_dtype.cpp)
vllm_cpp_add_test(test_ops_sample vt/test_ops_sample.cpp)
vllm_cpp_add_test(test_gemma4_rocm_fp8_seams vt/test_gemma4_rocm_fp8_seams.cpp)
vllm_cpp_add_test(test_ops_getblas_dualslot vt/test_ops_getblas_dualslot.cpp)
target_compile_definitions(test_ops_getblas_dualslot PRIVATE
  VLLM_CPP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(VLLM_CPP_HIP)
  # Separate HIP target: calls production GetBlas. Missing devices → exit 77.
  # .cpp would compile as host CXX and miss __HIP_PLATFORM_AMD__.
  set_source_files_properties(vt/test_ops_getblas_product.cpp PROPERTIES LANGUAGE HIP)
  vllm_cpp_add_test(test_ops_getblas_product vt/test_ops_getblas_product.cpp)
endif()
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

# ROCm M4 — the GDN sibling of the Qwen3-dense gate: Qwen3.5-0.8B greedy decode
# vs the pinned vLLM-ROCm oracle (the ONLY capture of this model — no dgx
# capture exists; the base golden pair is ROCm-captured on gfx1100).
# Checkpoint-gated: skips loudly when the HF snapshot is absent.
vllm_cpp_add_test(test_qwen35_paged_engine parity/test_qwen35_paged_engine.cpp)
target_compile_definitions(test_qwen35_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen35_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_qwen35_paged_engine_prerequisites
  COMMAND ${CMAKE_COMMAND}
    "-DTEST_EXE=$<TARGET_FILE:test_qwen35_paged_engine>"
    "-DPROBE_ROOT=${CMAKE_CURRENT_BINARY_DIR}/qwen35_paged_engine_prerequisites"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/parity/test_qwen35_paged_engine_prerequisites.cmake")

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

# SPEC-DFLASH2 W6 — the DFlash2 e2e gate (#1314): G2 draft-token identity and
# G3 SAME-TRAJECTORY acceptance against the beyond-pin oracle
# (vllm-project/vllm#52816 @ 66e5414c6, TRITON_ATTN). dgx-only; loud SKIP
# everywhere else, naming the two checkpoints and the golden it needs.
vllm_cpp_add_test(test_qwen38_dflash2_spec_decode
  parity/test_qwen38_dflash2_spec_decode.cpp)
target_compile_definitions(test_qwen38_dflash2_spec_decode PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen38_dflash2_spec_decode PRIVATE
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

# LOAD-MODELOPT-NVFP4-BORROW (#1647, .agents/specs/load-modelopt-nvfp4-borrow.md):
# WHERE a dense-loaded weight's bytes live, asked through the production loader
# `LoadQwen3_5Dense` over a REAL safetensors shard on disk — an in-memory fake
# resolver carries no `StTensor::mapping`, so `BorrowStTensorBytes` fails closed
# on it and the defect would read as already fixed. Hermetic and CPU-only: the
# shard is written to a temporary file per case, and the one platform shape a CPU
# tier cannot otherwise reach (unified memory that nevertheless STAGES) comes
# from a fake platform in the unused kXPU slot.
vllm_cpp_add_test(test_qwen3_5_dense_load_residency
  vllm/models/test_qwen3_5_dense_load_residency.cpp)

# QUANT-QWEN38-27B-NVFP4-ARM W4 (#821, .agents/specs/qwen38-27b-quant-arms.md):
# the compressed-tensors `mixed-precision` accounting + scheme-resolution gate
# for `unsloth/Qwen3.8-27B-NVFP4`. Hermetic — it reads the two COMMITTED
# manifests and the COMMITTED `config.json`, never a weight byte; the live
# re-verification arm is env-gated on VLLM_CPP_QWEN38_27B_NVFP4_DIR and SKIPS
# loudly without it. `src` is on the include path because the resolver under
# test is a model-private header under src/, not an include/vllm/ surface.
vllm_cpp_add_test(test_qwen38_27b_nvfp4_arm
  vllm/models/test_qwen38_27b_nvfp4_arm.cpp)
target_compile_definitions(test_qwen38_27b_nvfp4_arm PRIVATE
  QWEN38_27B_NVFP4_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen38_27b_nvfp4")
target_include_directories(test_qwen38_27b_nvfp4_arm PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

# QUANT-QWEN38-27B-NVFP4-ARM W5 (#821, #1574,
# .agents/specs/qwen38-27b-quant-arms.md): the ModelOpt `MIXED_PRECISION`
# accounting + config/tensor cross-check for
# `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`. A SECOND artifact of the same model,
# and a second file rather than a widening of the gate above: that one pins the
# 1968-name compressed-tensors set and this one pins a 2001-name ModelOpt set,
# and a gate that accepted both by loosening would measure neither. Hermetic —
# it reads the four COMMITTED manifests and the two COMMITTED config documents,
# never a weight byte; the live re-verification arm is env-gated on
# VLLM_CPP_QWEN38_27B_MODELOPT_MTP_DIR and SKIPS loudly without it. `src` is on
# the include path because the resolver under test is a model-private header
# under src/, not an include/vllm/ surface.
vllm_cpp_add_test(test_qwen38_27b_modelopt_mtp_arm
  vllm/models/test_qwen38_27b_modelopt_mtp_arm.cpp)
target_compile_definitions(test_qwen38_27b_modelopt_mtp_arm PRIVATE
  QWEN38_27B_MODELOPT_MTP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vllm/models/fixtures/qwen38_27b_modelopt_mtp")
target_include_directories(test_qwen38_27b_modelopt_mtp_arm PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/vllm/models)

vllm_cpp_add_test(test_qwen3_32b_nvfp4a16_paged_engine
  vllm/models/test_qwen3_32b_nvfp4a16_paged_engine.cpp)
target_compile_definitions(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  PARITY_GOLDENS_DIR="${CMAKE_SOURCE_DIR}/tests/parity/goldens")
target_include_directories(test_qwen3_32b_nvfp4a16_paged_engine PRIVATE
  ${CMAKE_SOURCE_DIR}/tests/parity ${CMAKE_SOURCE_DIR}/src)

# The registry type-confusion CLASS gate (#847, .agents/specs/registry-downcast-sweep.md).
# One case per SHAPE of the swept `static_cast<XLoadedModel&>(model)` site — const
# and non-const binding, `prepare` and `forward`, an alias family that shares one
# factory across two registered architectures, and the one inline-call site that a
# uniform search-and-replace over the other 33 would step past. Every case enters
# through the production seam (`reg.factory->forward` / `->prepare`, what
# `ModelRegistry::Forward` calls) and refuses before any weight is touched, so it is
# CPU-only and needs no checkpoint.
vllm_cpp_add_test(test_registry_downcast_refusal
  vllm/models/test_registry_downcast_refusal.cpp)

# The five suites that reach src/vllm/support/platform_compat.h, granted per
# target rather than globally: the file already carries 123 explicit
# ${CMAKE_SOURCE_DIR}/src grants, and a blanket one in vllm_cpp_add_test would
# convert that deliberate opt-in into a repo-wide default (#503).
foreach(_pc_target
        test_safetensors
        test_minimax_h3
        test_minimax_h3_video_fold
        test_kv_offload_connector
        test_kv_offload_tiering)
  if(TARGET ${_pc_target})
    target_include_directories(${_pc_target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  endif()
endforeach()
