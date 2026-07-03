# NVFP4 GGUF extension types — killgate fork prior art (M0.4 Task 5)

Mined 2026-07-03 on dgx.casa from mudler's llama.cpp forks:

- `~/llama-phase84-attn-only-source` (primary source citations below)
- `~/llama-phase93-qwen3next-gqa-bcast` (identical `ggml_type` enum — verified,
  same ids 39/40/41 at `ggml/include/ggml.h:429-431`)
- `~/killgate_series/*.patch` (patches 0015/0017/0020/0023/0025 exercise
  `GGML_TYPE_NVFP4` in CUDA MMQ/MoE paths; the type itself is defined in the
  fork source tree, not introduced by a patch)

## 1. Fork type ids (ggml/include/ggml.h)

The fork's `enum ggml_type` matches mainline llama.cpp exactly through id 35,
then appends two fork-specific ids after mainline's MXFP4:

```
GGML_TYPE_MXFP4 = 39,  // MXFP4 (1 block)            ggml.h:429  (same as mainline)
GGML_TYPE_NVFP4 = 40,  // NVFP4 (4 blocks, E4M3 scale) ggml.h:430  (FORK EXTENSION)
GGML_TYPE_Q1_0  = 41,                                 ggml.h:431  (FORK EXTENSION)
GGML_TYPE_COUNT = 42,                                 ggml.h:432
```

File-type (ftype) ids, `gguf-py/gguf/constants.py`: `MOSTLY_NVFP4 = 39`,
`MOSTLY_Q1_0 = 40` ("except 1d tensors").

## 2. NVFP4 block layout (id 40)

Source: `ggml/src/ggml-common.h:211-217`:

```c
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
```

- **block_elems = 64**, **block_bytes = 36** (4 + 32). Confirmed by the fork's
  `gguf-py/gguf/constants.py:4595`: `GGMLQuantizationType.NVFP4: (64, 4 + 32)`
  and by `ggml/src/ggml.c:741-748` type_traits (`.blck_size = QK_NVFP4`,
  `.type_size = sizeof(block_nvfp4)`).
- **Scales**: one unsigned E4M3 (**UE4M3**, no sign bit semantics — decoded by
  `ggml_ue4m3_to_fp32`, `ggml/src/ggml-impl.h:502`) scale byte per 16-element
  sub-block; 4 sub-blocks per 64-element block. This is the per-16 NVFP4
  micro-block scale layout (not per-32 like MXFP4).
- **Elements**: 4-bit **E2M1** codes, two per byte, decoded through the shared
  `kvalues_mxfp4` LUT (values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, stored as
  int8 2x-scaled entries in ggml-common.h).
- **Nibble order** (`dequantize_row_nvfp4`, `ggml/src/ggml-quants.c:531-554`):
  within sub-block `s`, byte `qs[s*8 + j]` (j in 0..7) holds element `j` in the
  low nibble and element `j + 8` in the high nibble; `y = kvalue * d`.
- **No per-tensor scale tensor.** Unlike NVIDIA's TensorRT NVFP4 recipe (per-16
  FP8-E4M3 scale x per-tensor FP32 scale), the fork's GGUF encoding is fully
  self-contained per block: quantization (`quantize_row_nvfp4_ref`,
  ggml-quants.c:346) picks `d = ue4m3(amax_sub / 6)` per sub-block and stores
  nothing outside the block. GGUF loaders need no side-channel tensors.

### Q1_0 (id 41), for completeness

`ggml-common.h:177-182`: `QK1_0 = 128`, `block_q1_0 = { ggml_half d; uint8_t
qs[QK1_0/8]; }` → **128 elems / 18 bytes** (`constants.py:4596`: `(128, 2+16)`).
Sign-bit-per-element ternary-ish 1-bit format; not observed in any APEX file.

### MXFP4 (id 39, same as mainline)

`ggml-common.h:205-210`: `QK_MXFP4 = 32`, 1 byte E8M0 scale + 16 bytes packed
e2m1 → **32 elems / 17 bytes**. Killgate patches 0015/0017 tune NVFP4 and MXFP4
together in the CUDA MMQ path.

## 3. APEX tensor-type histogram decoded

Observed histogram `{0: 301, 11: 159, 12: 178, 13: 34, 14: 1, 22: 60}` is from
`Qwen3.6-35B-A3B-APEX-I-Mini.gguf` (733 tensors,
`dgx.casa:/home/mudler/work/apex/qwen36_35b/`). Decoded against the FORK's
enum (which is identical to mainline for ids <= 35):

| id | fork enum name | block (elems, bytes) | count | used for |
|----|----------------|----------------------|-------|----------|
| 0  | F32    | (1, 4)     | 301 | norms, `ffn_gate_inp`, ssm_a/conv1d/dt/norm, 1-d tensors |
| 11 | Q3_K   | (256, 110) | 159 | token_embd, GDN attn_gate/ssm_{alpha,beta,out}, some expert ffn_{gate,up}_exps |
| 12 | Q4_K   | (256, 144) | 178 | attn_qkv, shared-expert ffn_*_shexp, attn_output, some ffn_down_exps |
| 13 | Q5_K   | (256, 176) | 34  | higher-precision shexp / ffn_down_exps in early layers |
| 14 | Q6_K   | (256, 210) | 1   | output.weight |
| 22 | IQ2_S  | (256, 82)  | 60  | MoE expert weights ffn_{down,gate,up}_exps.weight in 20 layers |
| 23 | IQ4_XS | (256, 136) | 60 (Quality variants) | MoE expert weights |
| 8  | Q8_0   | (32, 34)   | 120 (Balanced/Quality variants) | — |

**id 22 is IQ2_S, NOT NVFP4.** Verified by reading the fork enum
(`ggml.h:412`) and by dumping the file with the fork's own gguf-py: e.g.
`blk.10.ffn_down_exps.weight` shape ggml-[512, 2048, 256], type IQ2_S,
nbytes = 85_983_232 = 512*2048*256 / 256 * 82 — matches our traits math
(256-elem blocks, 82 bytes: 2 d + 64 qs + 16 qh).

All seven APEX GGUFs in `~/work/apex/qwen36_35b/` were histogrammed:

```
APEX-Balanced / I-Balanced: {0: 301, 8: 120, 13: 81, 14: 231}
APEX-Compact  / I-Compact : {0: 301, 11: 90, 12: 191, 14: 151}
APEX-I-Mini               : {0: 301, 11: 159, 12: 178, 13: 34, 14: 1, 22: 60}
APEX-Quality / I-Quality  : {0: 301, 8: 120, 13: 30, 14: 222, 23: 60}
```

**No APEX file uses NVFP4 (40) or any fork-specific id.** The APEX quant sweep
is pure mainline K-quants + i-quants. NVFP4 GGUFs do exist elsewhere in
mudler's fleet (killgate patches benchmark Qwen3-32B-NVFP4 and
Qwen3.6-35B-A3B NVFP4 dense/MoE), so the reader supports id 40 for those.

## 4. Implications for vllm.cpp

- `GgufFile` traits table (`src/vllm/model_executor/model_loader/gguf_reader.cpp`)
  now carries ids 22, 23 (needed by APEX Mini/Quality) and 39, 40, 41
  (MXFP4 + the two fork extensions). nbytes math is `numel / block_elems *
  block_bytes` with divisibility enforced — matches the fork's
  `ggml_row_size` for all these types (none has padding).
- **M2.2 kernels (NVFP4 dequant/GEMM)**: per-16 UE4M3 sub-block scale, e2m1
  LUT shared with MXFP4, low-nibble = element j / high-nibble = element j+8
  within a sub-block's 8 bytes. `QR_NVFP4 = 2`, `QI_NVFP4 = 8`
  (ggml-common.h:109-110) for the CUDA int-pack view. Killgate patch 0017
  notes dense NVFP4 decode GEMM is weight-read bandwidth-bound on GB10 with
  mmq_y=128 tiles; patch 0015 gates MoE token-tile selection for
  NVFP4/MXFP4 MoE (256 experts, top-8) — relevant when we port MMQ.
- Expert weights are the NVFP4 target in fork models (MoE `*_exps` 3-d
  tensors), same tensor family APEX-Mini puts in IQ2_S — M2.2 should plan
  for quantized 3-d expert tensors with per-expert row strides in blocks.

**M2.2 kernel-author trap (review finding):** the fork's `ggml_ue4m3_to_fp32`
returns value × 0.5 to compensate for the 2×-scaled `kvalues_mxfp4` LUT (see
ggml-impl.h comment; 0x7F decodes to 0). If you write your own UE4M3 decode
from the bias=7 spec while reusing the fork LUT, you will be 2× off.
