# Spec: QUANT-GGUF-IQ-TENSTORRENT wave 1 — IQ3_XXS on the P150

Row: `QUANT-GGUF-IQ-TENSTORRENT`
Issue: `ISSUE-LOCAL-01M2CNC22SYAKJN3YBGNVGSG56`
State: ACTIVE (2026-09-14)
Git integration: one PR for spec and implementation (developer preference,
recorded in `.agents/developer-preferences.md`).

## Scope

Serve `DType::kIQ3_XXS` matmul weights through the TENSTORRENT keep-quant
decode: an on-core IQ3_XXS×q8_K vec_dot in the int8-dot device kernel, the
packed i32 word staging, and the loader route admission. Wave 1 covers the
int8-dot arm only; the APEX-I-Nano 27B artifact becomes loadable and
generating on the P150 against the pinned oracle.

Out of scope (owed, see `## Owed`): the W4a grouped E=1 arm, IQ2_S, IQ2_XXS,
Q3_K, and every other IQ encoding.

## Upstream anchors

- CPU reference (the oracle this port mirrors bit-exactly):
  `VecDotIQ3_XXSQ8_K`, `src/vt/cpu/cpu_quant_dot.cpp:622`
  (ggml quants.c:999, `ggml_vec_dot_iq3_xxs_q8_K_generic`).
- Block layout: `BlockIQ3_XXS`, `src/vt/cpu/cpu_quant_blocks.h:163` —
  `{u16 d; u8 qs[96]}` = 98 B/block, 256 elems. `qs[0..63]` are grid-index
  bytes (two 4-byte grid entries per lane byte); `qs[64..71]` are 8
  scale+sign u32s, one per 32-element sub-block.
- Codebook tables: `kIq3xxsGrid[256]` (u32, 4 bytes per entry),
  `kKsignsIq2xs[128]`, `kKmaskIq2xs[8]` — `src/vt/cpu/cpu_quant_iq_tables.h:38-120`.
- Activation pairing: q8_K (same pairing as Q4_K/Q5_K/Q6_K), per
  `cpu_quant_traits.cpp:102`.
- Route predicate: `DeviceKeepQuantSupported` kTENSTORRENT case,
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:187`, pinned by
  `tests/vllm/test_gguf_keep_quant.cpp`.
- Device kernel: `kKeepQuantInt8DotKernelSrc`,
  `src/vt/tenstorrent/tenstorrent_ops.cpp:3277`, shared math in
  `src/vt/tenstorrent/kernels/keepquant_kernel_code.h`.
- Staging: `KeepQuantWordsPerBlock`, `src/vt/tenstorrent/tenstorrent_ops.cpp:1953`
  (64-B zero-padded rows; Q6_K's 210→256 B proves the sub-word tail path).
- Gate artifact: `mudler/Qwen3.8-27B-APEX-GGUF` @
  `98454f31de8ac2e8bc7cd359c526d9db230ca547`,
  `Qwen3.8-27B-APEX-I-Nano.gguf`, 10.7 GB,
  sha256 `47b627b7de17c2bcfa9cebb7cf2d9d81e68b4b61f8cad16cf85f839604c694cb`.
  Census (issue record, /tmp/apex-gguf-dump.txt 2026-09-13): 164 IQ3_XXS +
  89 IQ2_S + 44 IQ2_XXS + 78 Q3_K + 122 Q4_K + 8 Q8_0 + 1 Q6_K tensors.

## Design

1. Device math — `keepquant_kernel_code.h` gains `kq_vec_dot_iq3_xxs_q8_K`
   (packed-stream form of the CPU reference): per 32-sub-block read the
   scale+sign u32 (`ls = 2*(aux32>>28)+1`), four lanes of `kIq3xxsGrid`
   lookups with `kKsignsIq2xs` signs, int32 accumulation in the ported
   order, `0.25f * sumf` fold, per-block `d = f16(x.d) * y->d` scale. The
   decoder reads only the true 98 block bytes; the staging pad is never
   dereferenced. Tables move to a new
   `src/vt/tenstorrent/kernels/iq3xxs_tables.h` (device include path is the
   kernels dir only); the CPU header keeps its own copy — the op-level
   oracle test pins the two against each other, which is the drift guard.
2. Staging — `KeepQuantWordsPerBlock(kIQ3_XXS) = 32` (98 B zero-padded to
   128 B). `EnsureKeepQuantWords` needs no structural change (the Q6_K
   sub-word tail pattern covers 98 B); verify the pad is zero-filled.
3. Kernel dispatch — `enc_sel` 4 for IQ3_XXS in
   `MatmulBTQuantInt8DotKernel`; kernel `enc == 4` arm calls
   `kq_vec_dot_iq3_xxs_q8_K`; activation quantization takes the existing
   `kq_quantize_row_q8_K` path (only `enc == 3` uses q8_0).
   `qb_pad` arithmetic: IQ3_XXS falls into the `nb*292` q8_K arm unchanged.
4. Route — `MatmulBTQuantKernel` dispatches IQ3_XXS to the int8-dot arm
   REGARDLESS of `VT_TT_KEEPQUANT_INT8DOT` (the capability must be reachable
   on the default configuration); the refusal message names IQ3_XXS in the
   registered set. `DeviceKeepQuantSupported` kTENSTORRENT adds kIQ3_XXS;
   the `test_gguf_keep_quant.cpp` pin widens in the same change.
5. Docs — `docs/USAGE.md` records the artifact (file name, size, repo AND
   revision, sha256, refused arms IQ2_S/IQ2_XXS/Q3_K named beside it) in the
   change that makes it reachable.

## Risks

- Device soft-float must preserve the int32 accumulation order; the header
  comment's build-hazard discipline (-ffast-math) applies; no divisions in
  this dot, so the reciprocal hazard does not arise.
- 1 KB codebook in the device binary: well inside previous kernel sizes.
- L1 budget: `weight_row_bytes` grows to 128*nb per streamed block — the
  same streaming pattern as Q6_K (256 B/block), no residency change.
- The int8-dot Q4_K vehicle showed a one-nat band-edge flip (accepted for
  the Q4_K arm, env-gated). IQ3_XXS defaults ON through the same kernel; its
  own e2e gate decides. If the APEX gate fails on quantized-domain flips,
  fall back to env-gating IQ3_XXS too and record the owed default-path work.

## Tests

- Red-first op-level: extend the keep-quant op suite in
  `tests/vt/test_tenstorrent_backend.cpp` with an IQ3_XXS×q8_K case vs the
  CPU `VecDotIQ3_XXSQ8_K` oracle (the four-encoding pattern). RED = the
  route refuses kIQ3_XXS today (registered-set VT_CHECK). GREEN = byte-exact
  device vs CPU on the sweep shapes.
- Route pin: `tests/vllm/test_gguf_keep_quant.cpp` kIQ3_XXS admitted on
  kTENSTORRENT (red before the predicate widens).
- Full backend suite stays 73/73.

## Gates

1. Op-level oracle: device IQ3_XXS dot == CPU `VecDotIQ3_XXSQ8_K` (bit-exact
   sweep, the W4b red-first pattern).
2. Route: `test_gguf_keep_quant` green with kIQ3_XXS on kTENSTORRENT.
3. Backend suite: 73/73 (+ the new cases) green on the P150.
4. E2E: `vllm-bench` on APEX-I-Nano (the 27B recipe: `flock ~/gpu.lock`,
   luwen reset, `VT_TT_KEEPQUANT_INT8DOT` may stay UNSET — default path —
   `--num-blocks 64 --max-num-batched-tokens 64`) generating against the
   pinned llama.cpp b10451 oracle (MTP blk.64 caveat noted in the oracle
   record; check the APEX census for nextn tensors first).

## Evidence

- Census: issue record `/tmp/apex-gguf-dump.txt` (2026-09-13).
- Artifact hash recomputed at wave start (2026-09-14): sha256 above.
- Red/green logs under `/tmp/row-tt-iq-evidence/`.
- Wave-1 outcome (2026-09-14): op sweep 116/116 bit-exact vs the CPU
  oracle (`oplevel-green.log`); route pin green; review mutations 4/5
  red-detected, the dispatch-only gap repaired by a dedicated env-unset
  default-path leg (mutation red `repair-mutation-red.log`, green
  `repair-green.log`); backend suite 74/74 (524,455 assertions);
  APEX e2e recorded OOM/owed (`apex-alloc-trace.log`, see `## Owed`).

## Owed

- **APEX e2e generation gate (spec Gate 4) — recorded OOM 2026-09-14.**
  `vllm-bench` on APEX-I-Nano (27B recipe, default path, seed 0) fatals on
  the first engine step: `kq-decode/repair` asks 1,116,733,440 B inside
  `KQuantGrouped/chunk-loop` (the pre-existing Q4_K grouped decode repair
  plane, N=12288/K=5120), fatal `Out of Memory: Not enough space to
  allocate 1006632960 B` with 108 MB largest free block. The IQ3_XXS
  staging itself is NOT the trigger (`EnsureKeepQuantWords` deltas
  15-47 MB); APEX's residency profile (133 refused-arm Q3_K/IQ2_*
  tensors expanded to bf16 + keep-quant words) leaves the allocator
  fragmented. Evidence: `/tmp/row-tt-iq-evidence/apex-alloc-trace.log`.
  Owner: the keep-quant row's W4 residency redesign (chunked repair
  planes / smaller dequant row-chunks). NOT this row's wave 1.
- W4a grouped E=1 arm for IQ3_XXS (the non-env expert-tower path).
- IQ2_S (CPU dot is its prerequisite), IQ2_XXS, Q3_K waves.
- Embed-table dequantizing gather; `ssm_out` block-safe column permutation
  (row-level owed items carried from the keep-quant row).

## Stop conditions

- The device dot cannot be made bit-exact vs the CPU reference (soft-float
  drift in the grid/sign path): stop, record the gap, escalate — a
  distributional gate is a developer decision, not an implementer default.
- The APEX artifact OOMs the P150 in a way the 27B recipe does not cover:
  record the alloc trace, stop before redesigning residency (that is the
  keep-quant row's W4 territory).
