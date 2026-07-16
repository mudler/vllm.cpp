# Canonical Triton AOT kernel contract.
#
# This is the single declaration surface for the dispatcher base, generating
# source, kernel symbol, launch pins, grid, and signature of every vendored AOT
# kernel. The CUDA build consumes these records, while the CPU-only drift check
# evaluates the same records with `cmake -P`; neither path parses CMake source.

function(_vllm_triton_aot_declare BASE PY KERNEL WARPS STAGES GRID SIGNATURE)
  set_property(GLOBAL APPEND PROPERTY VLLM_TRITON_AOT_KERNEL_RECORDS
    "${BASE}@@${PY}@@${KERNEL}@@${WARPS}@@${STAGES}@@${GRID}@@${SIGNATURE}")
endfunction()

function(vllm_triton_aot_declare_all)
  set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_KERNEL_RECORDS "")

  # FLA launch pins were selected on GB10 at the exact gate-model shapes. Cache
  # variables remain overridable for maintainer sweeps; any override makes a
  # normal vendored build fail until the matching artifacts are regenerated.
  foreach(_H 48 32)
    set(VT_GDN_DELTAH_BV_${_H}     64 CACHE STRING "delta_h H=${_H} Triton BV")
    set(VT_GDN_DELTAH_WARPS_${_H}  4  CACHE STRING "delta_h H=${_H} Triton warps")
    set(VT_GDN_DELTAH_STAGES_${_H} 3  CACHE STRING "delta_h H=${_H} Triton stages")
    set(VT_GDN_CHUNKO_BK_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BK")
    set(VT_GDN_CHUNKO_BV_${_H}     64 CACHE STRING "chunk_o H=${_H} Triton BV")
    set(VT_GDN_CHUNKO_WARPS_${_H}  4  CACHE STRING "chunk_o H=${_H} Triton warps")
    set(VT_GDN_CHUNKO_STAGES_${_H} 3  CACHE STRING "chunk_o H=${_H} Triton stages")
  endforeach()
  set(VT_GDN_KKT_BK_48      128 CACHE STRING "kkt H=48 Triton BK")
  set(VT_GDN_KKT_WARPS_48   8   CACHE STRING "kkt H=48 Triton warps")
  set(VT_GDN_KKT_STAGES_48  3   CACHE STRING "kkt H=48 Triton stages")
  set(VT_GDN_KKT_BK_32      64  CACHE STRING "kkt H=32 Triton BK")
  set(VT_GDN_KKT_WARPS_32   4   CACHE STRING "kkt H=32 Triton warps")
  set(VT_GDN_KKT_STAGES_32  3   CACHE STRING "kkt H=32 Triton stages")
  foreach(_H 48 32)
    set(VT_GDN_TRIL_WARPS_${_H}  4 CACHE STRING "solve_tril H=${_H} Triton warps")
    set(VT_GDN_TRIL_STAGES_${_H} 3 CACHE STRING "solve_tril H=${_H} Triton stages")
    set(VT_GDN_WU_WARPS_${_H}    4 CACHE STRING "recompute_w_u H=${_H} Triton warps")
    set(VT_GDN_WU_STAGES_${_H}   3 CACHE STRING "recompute_w_u H=${_H} Triton stages")
  endforeach()

  # chunk_delta_h.py: gate shapes H=48 (27B) and H=32 (35B), Hg=16,
  # K=V=128, BT=BV=64. Flags mirror FLA's varlen path with initial/final state.
  set(_deltah_head
    "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32")
  foreach(_H 48 32)
    set(_tail
      "*bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32, ${_H}, 16, 128, 128, 64, ${VT_GDN_DELTAH_BV_${_H}}, 1, 0, 1, 1, 1, 1, 0")
    math(EXPR _gx "(128 + ${VT_GDN_DELTAH_BV_${_H}} - 1) / ${VT_GDN_DELTAH_BV_${_H}}")
    _vllm_triton_aot_declare(
      gdn_deltah_h${_H} chunk_delta_h.py
      chunk_gated_delta_rule_fwd_kernel_h_blockdim64
      ${VT_GDN_DELTAH_WARPS_${_H}} ${VT_GDN_DELTAH_STAGES_${_H}}
      "${_gx},NH,1" "${_deltah_head}, ${_tail}")
  endforeach()

  # chunk_o.py. Both output dtypes are required: f32 preserves the current
  # default, bf16 mirrors vLLM/FLA and is the path evaluated by this change.
  set(_chunko_f32
    "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
  set(_chunko_bf16
    "*bf16:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
  foreach(_H 48 32)
    set(_tail
      "${_H}, 16, 128, 128, 64, ${VT_GDN_CHUNKO_BK_${_H}}, ${VT_GDN_CHUNKO_BV_${_H}}, 1, 1")
    math(EXPR _gx "(128 + ${VT_GDN_CHUNKO_BV_${_H}} - 1) / ${VT_GDN_CHUNKO_BV_${_H}}")
    _vllm_triton_aot_declare(
      gdn_chunko_h${_H} chunk_o.py chunk_fwd_kernel_o
      ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}}
      "${_gx},NT,${_H}" "${_chunko_f32}, ${_tail}")
    _vllm_triton_aot_declare(
      gdn_chunko_bf16_h${_H} chunk_o.py chunk_fwd_kernel_o
      ${VT_GDN_CHUNKO_WARPS_${_H}} ${VT_GDN_CHUNKO_STAGES_${_H}}
      "${_gx},NT,${_H}" "${_chunko_bf16}, ${_tail}")
  endforeach()

  # fused_recurrent packed pure-decode recurrence: 27B ONLY (H=16, HV=48,
  # K=V=128, BK=128, BV=32), num_warps=1, num_stages=3, grid (cdiv(V,BV)=4, NBH).
  # Strides/dims baked to the 27B call site and guarded in TryTritonPackedDecode.
  set(VT_GDN_DECODE_WARPS_48  1 CACHE STRING "packed decode H=48 Triton warps")
  set(VT_GDN_DECODE_STAGES_48 3 CACHE STRING "packed decode H=48 Triton stages")
  _vllm_triton_aot_declare(
    gdn_decode_h48 fused_recurrent_packed_decode.py
    fused_recurrent_gated_delta_rule_packed_decode_kernel
    ${VT_GDN_DECODE_WARPS_48} ${VT_GDN_DECODE_STAGES_48}
    "4,NBH,1"
    "*bf16:16, *bf16:16, *bf16:16, *fp32:16, *fp32:16, *bf16:16, *fp32:16, *fp32:16, *i32:16, i32, 10240, 96, 96, 786432, 786432, 1, 16, 48, 128, 128, 128, 32, 20, 1")

  # WY pipeline: chunk_scaled_dot_kkt -> solve_tril -> recompute_w_u.
  set(_kkt_head
    "*bf16:16, *fp32:16, *fp32:16, *fp32:16, *i32:16, *i32:16, i32, i32")
  set(_tril_head "*fp32:16, *bf16:16, *i32:16, *i32:16, i32, i32")
  set(_wu_head
    "*bf16:16, *bf16:16, *fp32:16, *bf16:16, *bf16:16, *bf16:16, *fp32:16, *i32:16, *i32:16, i32, i32")
  foreach(_H 48 32)
    _vllm_triton_aot_declare(
      gdn_kkt_h${_H} chunk_scaled_dot_kkt.py chunk_scaled_dot_kkt_fwd_kernel
      ${VT_GDN_KKT_WARPS_${_H}} ${VT_GDN_KKT_STAGES_${_H}}
      "NT,${_H},1"
      "${_kkt_head}, ${_H}, 16, 128, 64, ${VT_GDN_KKT_BK_${_H}}, 1, 1")
    _vllm_triton_aot_declare(
      gdn_tril_h${_H} solve_tril.py merge_16x16_to_64x64_inverse_kernel
      ${VT_GDN_TRIL_WARPS_${_H}} ${VT_GDN_TRIL_STAGES_${_H}}
      "NT,${_H},1" "${_tril_head}, ${_H}, 64, 1")
    _vllm_triton_aot_declare(
      gdn_wu_h${_H} wy_fast.py recompute_w_u_fwd_kernel
      ${VT_GDN_WU_WARPS_${_H}} ${VT_GDN_WU_STAGES_${_H}}
      "NT,${_H},1"
      "${_wu_head}, ${_H}, 16, 128, 128, 64, 64, 64, 1")
  endforeach()
endfunction()

function(vllm_triton_aot_expected_lines OUT_VAR)
  get_property(_records GLOBAL PROPERTY VLLM_TRITON_AOT_KERNEL_RECORDS)
  set(_lines "")
  foreach(_record IN LISTS _records)
    string(REPLACE "@@" ";" _fields "${_record}")
    list(LENGTH _fields _field_count)
    if(NOT _field_count EQUAL 7)
      message(FATAL_ERROR "invalid Triton AOT declaration: ${_record}")
    endif()
    list(GET _fields 0 _base)
    list(GET _fields 1 _py)
    list(GET _fields 2 _kernel)
    list(GET _fields 3 _warps)
    list(GET _fields 4 _stages)
    list(GET _fields 5 _grid)
    list(GET _fields 6 _signature)
    list(APPEND _lines
      "base ${_base} py=${_py} kernel=${_kernel} warps=${_warps} stages=${_stages} grid=${_grid} signature=${_signature}")
  endforeach()
  list(SORT _lines)
  set(${OUT_VAR} "${_lines}" PARENT_SCOPE)
endfunction()
