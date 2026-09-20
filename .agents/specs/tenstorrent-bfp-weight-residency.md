# Tenstorrent native BFP4/BFP8 weight residency for the 27B-class decode

Issue: `.agents/issues/BACKEND-TENSTORRENT/ISSUE-LOCAL-01M2YXN1QEMAEY5W8QCKH76HTS.md`
(row `BACKEND-TENSTORRENT`).

Status: **DRAFT, 2026-09-13.** Spec-first: no implementation is in scope until
this file is committed and the row moves `READY`.

## Now

The f32-exact decode stack measures ~0.028 tok/s on
`Qwen3.8-27B-APEX-I-Nano` (B2, token-exact vs the llama.cpp `b10451` greedy
oracle; `.agents/benchmark-record.md` 2026-09-13 entry), while Tenstorrent's
native tt-metal pipeline reports ~50 tok/s on the same model class on one
P150A with BFP4 weights / BFP8 KV / BF16 deltaNet state. The ~1800× gap is a
FORMAT gap: our decode GEMMs run GGUF block-dequant plus the SFPU
f32-exact path (ttnn matmul truncates f32 operands to tf32 on Blackhole —
measured, and `ComputeConfig` does not lift it), while the native pipeline
runs tensor-core matmul on BFP-typed weights where BFP precision IS the
hardware's precision contract. No tuning pass closes this. This spec commits
the native BFP residency path.

## Scope

In scope:

- **Load-time BFP conversion.** New residency values beside the existing
  `GgufResidency` arms (`kExpandBf16`, `kKeepQuant`, `kKeepF16`,
  `kTransformedWeight`, …,
  `include/vllm/model_executor/model_loader/gguf_keep_quant.h:65`): the
  bf16/dequantized weight operand is converted at load into a
  device-resident `BFLOAT4_B` or `BFLOAT8_B` tensor and never exists as a
  full-precision resident copy. This follows the expand-bf16 precedent —
  conversion happens once, device-side, inside the existing weight-residency
  machinery (`Route` in
  `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp`), not as a new
  checkpoint format.
- **Dense 27B forward on BFP weights.** The dense projection GEMMs
  (attention q/k/v/o, MLP gate/up/down) of the Qwen3.8-27B-class forward
  served by the NATIVE `ttnn` matmul path. Explicitly NOT the SFPU
  f32-exact floor: BFP matmul is the precision contract; running it through
  an f32-exact re-check would reintroduce the exact cost this row removes.
- **Admission and refusal by name.** `DeviceKeepQuantSupported` (or its BFP
  equivalent) admits the new arms; tensors that cannot take BFP residency
  refuse by name and fall through to the existing arms, never silently.

Out of scope, recorded as named follow-ups (each gets its own row):

- **BFP8 KV cache** — the KV blocks stay in their current residency.
- **BF16 deltaNet/GDN state** — the GDN chain keeps its current state
  residency.

Non-goals:

- **No GGUF format changes.** No new ggml type, no BFP GGUF writer, no
  converter. The input artifact set is unchanged (GGUF and HF bf16).
- **No MTP.** No speculative-decoding interaction.

## Upstream anchors

- `tt-metal` `ttnn` dtypes `BFLOAT4_B`/`BFLOAT8_B` run tensor-core matmul as
  first-class operands. The native 27B-class shape evidence is the quasar
  linear-graph census:
  `models/experimental/ops/quasar/tests/qwen3_vl_ops/test_linear.py` —
  activation bf16 (TILE, L1 width-sharded) × weight `BFLOAT4_B` or
  `BFLOAT8_B` (TILE, DRAM width-sharded), `ttnn.linear` with a bf16 output
  dtype. The same dtype split appears in the DeepSeek-V3 demo weight
  transforms (`models/demos/deepseek_v3_b1/weights/`), including the
  experts-only BFP4 arm (`tp4_attention.py`, `overlap_configs.py`).
- The reported external reference (~50 tok/s, single P150A, BFP4 weights /
  BFP8 KV / BF16 deltaNet state) is the perf target, recorded as an external
  reference and not re-measured by this row.
- No vLLM mirror exists for a TT-native BFP residency — deviation by design
  (secondary substrate, same disposition as the GDN row's TT kernels).

## Design

**Which seam owns conversion.** The load-time residency machinery owns it,
not the model file and not the kernel. `GgufResidency` gains the BFP arms;
the `Route` decision in the qwen3.5/27B GGUF weight path selects them per
tensor, exactly where `kExpandBf16`/`kKeepQuant` are chosen today
(`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:288`,
`:344`). Conversion is device-resident: upload once in bf16, convert on
device to the BFP tile layout, drop the bf16 staging (the same shape as the
existing expand path — no second resident copy of a 27B weight set). HF
bf16-safetensors sources take the same conversion arm; only the staging
read differs.

**Which GEMM path serves it.** The NATIVE ttnn matmul —
`BFLOAT4_B`/`BFLOAT8_B` weight × bf16 activation → bf16 output, TILE
layout, the shape family the quasar census pins. The f32-exact SFPU decode
floor stays exactly where it is: it serves the arms that demand token-exact
behavior (keep-quant int8-dot, expand-bf16 with `VT_TT_AFFINE_F32=1`). The
BFP arms dispatch by residency, so the two floors coexist under the same
`ModelRegistry::Forward` entry with no parallel forward path.

**Correctness gate.** The gate is a **near-tie disposition** against the
`b10451` greedy oracle on the quantized model, the ratified sacred-pair
treatment (`tenstorrent-qwen35.md` e2e sacred pair;
`tenstorrent-gsq-keepquant.md` IQ1 near-tie disposition). BFP4/BFP8
matmul is correct at BFP precision by construction — the same reasoning
that makes `b10451` the oracle for a quantized artifact. Per-op goldens
compare at BFP precision: quantize the reference operand, then compare
(`quantize-then-compare`), never compare raw f32 against a truncated
device output. The token-exact gate this tree uses for the f32-exact arms
does not apply to this row, by design and recorded here — not widened,
not silently.

**Decode stack interaction.** `VT_TT_AFFINE_F32` and the f32-exact decode
levers stay default for the non-BFP arms. The BFP arms opt in by residency
selection behind ONE general lever, `VT_TT_WEIGHT_RESIDENCY`
(`off` default | `bfp8` | `bfp4` reserved and refused by name until
implemented; parser: include/vllm/config/tt_weight_residency.h), so a new
variant is a new VALUE, not a new flag. Whether an arm lands default-on or
behind the lever follows the
`VT_TT_KEEPQUANT_INT8DOT` precedent (land default-off when the e2e anchor
band fails, decide in the Outcome, never inferred) and is recorded in the
row's Outcome.

## Risks

- **BFP4 accuracy is not uniform across roles.** The native stack's own
  split (quasar census: `BFLOAT4_B` for the wide expert-side projections,
  `BFLOAT8_B` elsewhere) suggests attention projections are more sensitive
  than experts/MLP. The row may land experts-only BFP4 with BFP8
  attention, following the native split, rather than forcing BFP4
  everywhere. The per-role choice is a measured Outcome, not an
  assumption.
- **tf32 truncation interactions.** The tf32 truncation of f32 operands is
  a matmul-input-property, not a global one; a BFP-typed weight operand
  sidesteps it for that operand. But any residual f32 activation-side
  path that falls back into ttnn matmul re-enters the truncation. The
  dispatch must never route a BFP-resident weight through an f32-operand
  matmul by accident; the residency→kernel mapping is pinned by test.
- **Capture safety.** The decode graph replays under `VT_TT_PROGRAM_CACHE`;
  BFP tensors in DRAM with different layouts than today's arms can miss
  capture. The keep-quant row already paid for seven layers of
  capture-time program-cache misses — the BFP arms run the same
  capture-replay gate before any e2e claim.

## Tests

- **Per-op golden at BFP precision (unit, red-first).** For each admitted
  (role, dtype) pair: quantize the bf16 reference weight to
  `BFLOAT4_B`/`BFLOAT8_B` on CPU, run the native matmul on device, compare
  against the quantize-then-matmul CPU reference at a tolerance derived
  from the BFP quantization step. Mutating the tolerance to raw-f32 must
  red.
- **Residency dispatch pin.** Each admitted dtype routes to the native
  matmul arm and each refused dtype refuses by name;
  `tests/vllm/test_gguf_keep_quant.cpp` gains the BFP cases.
- **Capture-replay gate.** The BFP decode graph captures and replays with
  JIT cache hit parity with the f32-exact stack.
- **E2E near-tie gate.** `Qwen3.8-27B-APEX-I-Nano` end-to-end vs the
  `b10451` greedy oracle, near-tie disposition (flip rate within the
  ratified anchor band; the sacred-pair production entry point
  `ModelRegistry::Forward`, not a hand-constructed forward).
- **Reachability.** The smallest failing test enters the BFP path through
  the production loader/forward entry point; a class-construction unit
  alone does not close the row.

## Gates

- G0-CORRECT: the e2e near-tie gate above, on the quantized artifact, via
  the production entry point.
- G0-SPEED: decode tok/s recorded against the ~50 tok/s external reference
  and the 0.028 tok/s f32-exact baseline on the identical workload
  (1×128→64, same env, same model file). The ratio is recorded even when
  the external number is not reproducible here; the gap stays open until
  either the reference is matched with a traceable delta or the next
  hypothesis is named. No ceiling is declared.
- Oracle: vLLM does not implement a TT BFP path; the pinned `b10451`
  llama.cpp greedy oracle remains the correctness denominator, per the
  near-tie precedent.

## Stop conditions

- The native ttnn matmul cannot be made to consume our converted BFP
  tensors at the 27B shapes (layout/sharding refusal that no config lifts):
  stop the arm, record the trace, keep the row open with the named next
  hypothesis.
- The near-tie flip rate exceeds the anchor band for every dtype split
  tried (including experts-only BFP4): stop with the accuracy evidence
  recorded; the row does not soften the gate to pass.
- Device access blocked mid-campaign: unit gates recorded, e2e named
  `Owed`, the row never closes on class-construction evidence alone.

## Owed

- BFP8 KV cache residency (named follow-up row).
- BF16 deltaNet/GDN state residency confirmation against the native stack
  (named follow-up row).
- The batched (M=2, cc=2) TPOT measurement on both the f32-exact and BFP
  stacks, so the speed gate compares like against like.
- The default-on vs opt-in residency decision, resolved in the Outcome.
- A reproduction of the ~50 tok/s external reference on our own harness, or
  an explicit record that the reference remains external-only.

## Now

`DRAFT` — this spec commits the load-time BFP4/BFP8 conversion and the
native-matmul dense decode path for the 27B forward. On commit the row
moves `READY`; implementation starts only after the committed spec is the
row's contract.

## Git integration

One pull request per the repository default; the spec lands first in the
same PR's commit history (spec commit precedes implementation commits).
