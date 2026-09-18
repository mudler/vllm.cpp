# Mirror the compiled primary's BF16 LM-head output boundary

Owning row: `BACKEND-ROCM-BF16-MOE`, whose lifecycle remains `ACTIVE`.
Parent row: `BACKEND-ROCM`, whose lifecycle remains `ACTIVE`.
Owner: the delegated head-boundary implementer, fresh reviewer, and coordinating operator.
Issue: [#3116](https://github.com/mudler/vllm.cpp/issues/3116), `ISSUE-GH-3116`.
Base: `691b7af30`, branch `row/BACKEND-ROCM-BF16-MOE-head-bf16`.
Integration: one pull request under the repository default. This spec is committed before implementation.
The implementation pull request closes #3116 when its required gates pass and it lands.

## Now

State: `SPIKE`. The boundary measurement is complete on `gfx1100` and the specification precedes the implementation.
The measurement supports landing the boundary as a fidelity mirror and does **not** support claiming a token repair.
The operator's 18-workload production token gate passes at `8e43d18bd` under the corrected whole-sequence-membership rule
(28598 of 28598 assertions, receipt `/home/vikash/.cache/moe-head-bf16/operator-gate.log`), and the frozen-head preflight
there exits 0 (receipt `/home/vikash/.cache/moe-head-bf16/preflight-final.log`). The fresh review's two LOW findings and
two bookkeeping corrections are repaired on this branch; a fresh scoped review of the repair is the next gate.

## Problem and scope

The pinned compiled primary stores its LM-head output in BF16 and widens it to F32 only where the sampler needs it.
The native Qwen3-MoE forward stores F32 logits directly:

- `src/vllm/model_executor/models/qwen3_moe.cpp:279-302` — the UNTIED `lm_head` block declares
  `DBuf logits(d, DType::kF32, {n_out, vocab})` and calls `vt::Matmul(d.q, logits.t(), src, lm)`.
- `src/vllm/model_executor/models/qwen3_moe.cpp:21` states the contract this row is changing: "Returns `[n_out, vocab] f32` logits".

Scope is the head-output dtype boundary only:

1. The head projection writes BF16, mirroring the primary's head output dtype.
2. The shared `vt::CastF32` widens that BF16 result into the F32 buffer the forward returns today.

Out of scope:

- Decode attention and Q/K preamble parity (#3115). Hidden-state parity at about one BF16 ulp remains required, and
  this change alone cannot move the native answer onto the primary's concurrency-2 token.
- The router, grouped MoE, quantization arms, the upstream pin, CI, and checkers.
- Any change to the sampler, the logits views, or the captured-graph logits slot.

Changed by this work, and unreplayed: the tied-embedding arm's arithmetic. `lm_head::Project` sends BOTH arms through the
same BF16 store and the same `vt::CastF32` (`include/vllm/model_executor/models/lm_head_projection.h:64-69`), so the tied
arm's `vt::MatmulBT` now narrows to BF16 where it stored F32 directly before this change. No captured artifact exercises
that arm — the fixture this spec replays records `"tie_word_embeddings": false`
(`/home/vikash/.cache/rdna3-moe-impl/preserved/fixture/config.json`), and the measurement above is the untied `vt::Matmul`
orientation only. The tied arm is therefore changed here, unreplayed, and unreachable from this measurement; its own
artifact replay is owed in `## Owed`.

## Upstream anchor

- Primary oracle: vLLM `e126687a9a828d513c01a07cd69f025f27d63280` (`.agents/upstream-sync.md:7`).
- Executing chain: `Qwen3MoeForCausalLM` computes its untied `lm_head` as a BF16 `F.linear`; the logits processor
  widens the result to F32 before sampling (`.agents/specs/rocm-residual-norm.md:320-325`).
- Captured primary evidence: `/home/vikash/.cache/rdna3-moe-impl/preserved/oracle-diagnostic-2/L33-C2-R0-head-6.json`
  records `"resolved_head_dtype": "torch.bfloat16"` with `hidden` [2,128], `weight` [128,128] and `logits` [2,128],
  all `torch.bfloat16`, for decode step 6 of the L33/C2/R0 workload — the exact-tie step of
  `.agents/specs/rocm-bf16-moe.md`.
- Local anchor: `src/vllm/model_executor/models/qwen3_moe.cpp:297-301` (the F32 store), `include/vt/ops.h:2652`
  (`vt::Matmul`), `include/vt/ops.h:5619` (`vt::CastF32`), `src/vt/rocm/rocm_matmul_hipblaslt.hip:491`
  (`MatmulKernelRocm`, `hipblasGemmEx` with an F32 compute type and an F32-or-BF16 output store),
  `src/vt/rocm/rocm_dense_basic.hip:408` (`CastF32KernelRocm`).

## Measurement that decides the change

Harness: `tests/vllm/model_executor/test_qwen3_moe_lm_head_bf16.cpp`, case
"qwen3 MoE LM-head BF16 boundary: primary artifact replay".
It loads the three `.bin` files, interprets them from the JSON metadata, uploads the weight in the `[H, vocab]`
orientation the loader builds (`src/vllm/model_executor/models/qwen3_moe_weights.cpp:144-149`,
`LoadBf16Transposed("lm_head.weight")`), and runs the operators the forward itself calls:
`vt::Matmul` into an F32 buffer (arm a), `vt::Matmul` into a BF16 buffer followed by `vt::CastF32` (arm b), and
`logits.bin` widened to F32 (arm c). Arm a′ is arm a narrowed on the host with round-to-nearest-even.

One command, no model checkpoint:

```sh
VT_MOE_HEAD_FIXTURE=/home/vikash/.cache/rdna3-moe-impl/preserved/oracle-diagnostic-2 \
  ./build-head-hip/tests/test_qwen3_moe_lm_head_bf16
```

Input hashes:

| File | sha256 |
|---|---|
| `L33-C2-R0-head-6-hidden.bin` | `213770e378ff6f2bc06635b47d46ff623b3a8b53075fcd73b23f7b99f43cf0b6` |
| `L33-C2-R0-head-6-weight.bin` | `a174ad23ec6021ec75759ea580c08385482e528c102111d485cd09022a5a1ef4` |
| `L33-C2-R0-head-6-logits.bin` | `bc92004872f6f80ae93d87a867710868987f591a5ad63308cb857858a4a0a434` |

Measured on 2026-09-10 at base `691b7af30`. The CPU and ROCm arms are identical value for value, so one table
carries both; `device=ROCM` is the production policy and was executed on `gfx1100` (RX 7900 XTX, ROCm 7.15.26333).

| Row | Arm | argmax | top-2 (value) | max abs diff vs (c) | mean abs diff vs (c) | b==c |
|---|---|---|---|---|---|---|
| 0 | (a) F32, today | 118 | 118: 0.335236, 63: 0.334908 | 9.23157e-04 | 1.58187e-04 | 128/128 differ |
| 0 | (b) BF16 + `CastF32` | 118 | 118: 0.335938, 63: 0.333984 | 0 | 0 | **EQUAL, 0/128** |
| 0 | (c) primary `logits.bin` | 118 | 118: 0.335938, 63: 0.333984 | — | — | — |
| 1 | (a) F32, today | 85 | 85: 0.408177, 121: 0.334220 | 9.72956e-04 | 1.44222e-04 | 128/128 differ |
| 1 | (b) BF16 + `CastF32` | 85 | 85: 0.408203, 121: 0.333984 | 0 | 0 | **EQUAL, 0/128** |
| 1 | (c) primary `logits.bin` | 85 | 85: 0.408203, 121: 0.333984 | — | — | — |

`max|a-b|` equals `max|a-c|` for both rows, and arm (a′) equals (c) word for word on both rows
(0/128 differing, `max|a'-c| = 0`). Two consequences:

- The device BF16 store, the host round-to-nearest-even narrowing, and the primary's own words agree, so the only
  difference between (a) and (b) is the store rounding, not the reduction.
- The F32 boundary's error is bounded by half a BF16 ulp (2^-9 = 1.953125e-03 at this magnitude): a dtype gap,
  not an arithmetic defect.

### Does the boundary alone change the step-6 argmax?

No, on both available inputs.

1. On the primary's own captured head input, (b) and (c) choose 118, and so does the F32 arm (a). The mirror is
   therefore invisible to this row's token at step 6 while the native run is fed the primary's hidden state:
   what separates native from the primary's concurrency-2 answer is the hidden state (#3115), not the head.
2. On the native production step-6 logits themselves, recorded in
   `/home/vikash/.cache/residual-norm-repair1/green-cc9d4f565/production-fusion-1.json` (identical bytes to
   `/home/vikash/.cache/moe-6fd1650c4-tmp/gpu-run-da0ce377b.json`), the F32 margin is
   `63: 0.335850269` against `118: 0.335278690` (+5.71579e-04). Round-to-nearest-even narrowing maps **both** to
   the same word `0x3eac = 0.3359375` — an exact tie — and the native lowest-index tie-break
   (`src/vt/rocm/rocm_dense_basic.hip:170`, `v > best || (v == best && j < arg)`) keeps 63, which is the token the
   native path already emits at all three repeats of concurrency 1 and concurrency 2. Request 1 is unaffected
   (85 before and after).

Reproduce the second measurement with
`PYTHONPATH=/home/vikash/.cache/rdna3-moe-impl/numpy-only-python python3 /home/vikash/.cache/moe-head-bf16/native-step6-narrowing.py /home/vikash/.cache/residual-norm-repair1/green-cc9d4f565/production-fusion-1.json`.
It is a host-side narrowing of a recorded native run, not a device execution, and is labelled as such.

### Decision

**Supported, as a correctness mirror only.** The measurement supports the boundary change for these reasons:

- Given identical head inputs, the mirrored boundary reproduces the primary's logits element for element (0/128
  differing words on both rows), while the shipped F32 boundary does not (up to 9.73e-04 = half a BF16 ulp).
  "vLLM is the reference" makes the primary's head output dtype the contract, and this removes a measured,
  quantifiable divergence rather than an aesthetic one.
- The change does not move the decode step-6 token on either measured input, so landing it cannot by itself
  explain, or repair, the 63/118 tie.

**Not supported**, and explicitly out of scope: any claim that this change fixes the tie, improves hidden-state
parity, or makes the native run adopt the primary's concurrency-2 answer. The issue's own dependency statement
holds: #3115 remains required.

## Design

Where the narrowing belongs, and what must not move:

1. `include/vllm/model_executor/models/lm_head_projection.h` (new) exposes
   `vllm::lm_head::Project(dense_attn::Dev, const vt::Tensor& src, const vt::Tensor& lm, bool tied)`, which returns
   an owning `dense_attn::DBuf` of `[n_out, vocab]` **F32**. The function is the whole head: it allocates the BF16
   projection buffer, calls `vt::Matmul` (untied, `lm` is `[H, vocab]`) or `vt::MatmulBT` (tied, `lm` is
   `[vocab, H]`), then widens with `vt::CastF32` into the F32 buffer it returns.
2. `src/vllm/model_executor/models/qwen3_moe.cpp` `ForwardLayers` calls that seam and nothing else. The returned
   buffer keeps the dtype, rank and shape it has today, so every downstream consumer stays valid: the device-logits
   view (`ViewDeviceLogits`, `:337-345`), the owning wrapper (`WrapDeviceLogits`, `:321-331`), the host download in
   `Qwen3MoeModel::Forward` (`:411-423`), the captured graph slot `SizeSlot::logits` (`:514`, `:636`, `:710`) and
   the sampler all read `DType::kF32`.
3. The narrowing is a device store and the widening is the shared op — no host round-trip, no new op, no
   per-element loop in the model, and no change to `vt::Matmul`'s contract (it already admits a BF16 output,
   `include/vt/ops.h:2649-2652`, `src/vt/ops.cpp:125-126`).
4. Why a seam rather than three inline lines: the boundary is a production behavior that no checkpoint-free test
   could otherwise execute, and the projection is the same operator sequence in both places. It is not a parallel
   path — `ForwardLayers` has exactly one head call site and it is this function. (`qwen3_5_moe_block.h` is the
   same pattern for the MoE block: an internal body exposed to a second caller over primitive `vt::` types.)

Graph and pool consequences, stated because the residual spec warns that "changing only the head buffer dtype
would leave those views and host copies invalid" (`.agents/specs/rocm-residual-norm.md:325`):

- The returned F32 buffer, its address and its dtype are unchanged, so `ViewDeviceLogits`/`WrapDeviceLogits` and
  the captured slot stay byte-compatible.
- The BF16 head buffer is one additional transient `DBuf` per forward ([S, vocab] bf16 = half the bytes of the F32
  result). The cold eager pre-warm step at a padded size runs the same `ForwardLayers` call
  (`qwen3_moe.cpp:727-731`), so its size class is already in the `DevicePool` free list when the capture at that
  size runs — the capture's no-allocation requirement is preserved by construction.

## Risks

| Risk | Handling |
|---|---|
| The emitted token becomes tie-break dependent at steps whose top-2 margin is under one BF16 ulp. | Measured: it does not move at step 6 on either input. The operator's 18-workload membership gate is rerun at the frozen head; a moved position is a finding, not a tolerated difference. |
| The change is invisible to the token gate and could therefore land unreached. | The focused artifact replay is the gate; the reviewer mutates the production call site as well as the boundary, and the operator's production run exercises the code path. |
| An extra BF16 buffer per step changes memory or capture behavior. | One transient pooled block per forward, half the F32 result's bytes, warmed by the same cold step; no persistent allocation, no new residency. |
| A future reader mistakes this for the tie repair. | The issue, this spec, the commit body and the header comment all state that hidden-state parity (#3115) is still required. |
| The replay pins the NN (`vt::Matmul`) orientation only, so a future switch to `MatmulBT` for the untied head would not be covered. | Recorded as owed below; the untied arm is `Matmul` today (`qwen3_moe.cpp:300-301`). |

## Tests

- Focused, red first: `tests/vllm/model_executor/test_qwen3_moe_lm_head_bf16.cpp`.
  - `primary artifact replay` — the three-arm measurement above, kept as a regression witness for the operators.
  - `the production head projection mirrors the primary` — calls `vllm::lm_head::Project` with the primary's exact
    hidden and weight bytes on every available device and requires the returned F32 logits to equal the primary's
    widened BF16 words element for element, to keep the F32 `[rows, vocab]` shape, and to select the primary's
    argmax. It skips itself (reported skipped by doctest) without `VT_MOE_HEAD_FIXTURE`, and the binary exits 77
    (CTest: Skipped) on that run.
  - `the production forward returns BF16 logits` — the call-site case, and the one a reverted call site must
    redden. It drives `Qwen3MoeModel::Forward` with a degenerate but legal zero-decoder-layer config (embed ->
    final RMSNorm -> lm_head) and synthetic BF16 weights, so it needs no checkpoint, no capture and no fixture
    directory, and it requires every logit the sampler would receive to be a BF16 word widened to F32. It carries no
    skip decorator, so it runs and is reported in the fixture-absent run too, where the two replay cases skip; a
    failure in it keeps doctest's non-zero exit instead of being folded into the run's 77.
- Red before implementation: with the production projection storing F32 (the pre-change behavior, extracted
  verbatim), the projection case reports 256/256 differing logits per device at `max_abs = 9.72956e-04`, and the
  forward case reports 128/128 logits that are not BF16-representable.
- Mutation (IMP-MUTATE), two independent guarantees, each restored byte for byte and re-verified by sha256:
  1. Delete the BF16 narrowing and the `vt::CastF32` inside the seam (the pre-change body) — the projection case
     reddens, 2 `CHECK` failures, one per device; the forward case and the measurement case stay green.
  2. Revert the forward's call site to the inline F32 `vt::Matmul` — the forward case reddens, 128/128 logits not
     BF16-representable; the two seam cases stay green, which is exactly the coverage split the two cases exist
     to make visible.
- Existing gates that must stay green: `test_rocm_moe_bf16` (needs the operator's GPU and fixture),
  `test_rocm_moe_reference_set`, and the frozen-head preflight.

## Gates

| Gate | Requirement |
|---|---|
| Focused green | `build-head-hip/tests/test_qwen3_moe_lm_head_bf16` exits 0 with the fixture set, on CPU and ROCm. |
| Focused, fixture absent | The same binary reports the two replay cases skipped, runs the fixture-free call-site case on CPU and ROCm, and exits 77 (CTest: Skipped); a failing case exits non-zero rather than being folded into that 77. |
| Red and mutation | The pre-change form reddens the production case; the reviewer reproduces it and restores byte-for-byte. |
| Preflight | `scripts/agent-preflight.sh --staged` at the implementation head, log under `/home/vikash/.cache/moe-head-bf16/`. |
| Production token gate | Operator only: the 18-workload membership gate in `tests/vllm/models/test_rocm_moe_bf16.cpp` at the frozen head, with the recorded concurrency-1/-2 reference set. |
| Reachability | The production forward calls the seam; the zero-layer case reddens when that call site is reverted, and the operator's run executes it in the real checkpoint path. |

## Evidence

- Measurement log: `/home/vikash/.cache/moe-head-bf16/measure-replay.log` and `measure-replay-record.txt`
  (command, head, exit status, the a/a′/b/c table for both rows on CPU and ROCm).
- Native step-6 narrowing log: `/home/vikash/.cache/moe-head-bf16/native-step6-narrowing.txt`.
- Build log: `/home/vikash/.cache/moe-head-bf16/build-lib.log`, `build-measure.log`.
- Preflight log: `/home/vikash/.cache/moe-head-bf16/preflight.log`.
- Fixture-absent run of the repaired suite: `/home/vikash/.cache/moe-head-bf16/implt-no-fixture.log` — exit 77,
  `1 passed | 0 failed | 2 skipped` with 10 assertions, both replay cases reported skipped and the call-site case
  reported per device. With the fixture: `implt-with-fixture.log` — exit 0, 3 cases, 126/126 assertions.
- Repair mutations (scratch seam revert, restored byte-for-byte at sha256
  `26682c309b39264e4ecf349023652affad27d07818ff0a25a675c2cee15cab3c`): `implt-mut1-no-fixture.log` exits 1 with 2
  failing assertions when the failure guard is present, `implt-mut2-no-fixture.log` reports a masked 77 when it is
  disabled, and `implt-mut3-no-fixture.log` exits 1 through the in-case `REQUIRE_MESSAGE` guard when the skip decorator
  is removed. CTest reports Skipped without the fixture and Passed with it: `implt-ctest.log`.

## Stop conditions

- `STOP-VERIFY` (BLOCKED): the focused case or the preflight cannot pass at the implementation head.
- `STOP-AUTHORITY` (BLOCKED): the repair requires a file outside the head-output boundary, or the token gate needs
  a device or checkpoint the operator has not provided.
- `STOP-DECISION` (NEEDS_DECISION): the operator's 18-workload gate moves a token position, which would make this
  boundary a token-visible change rather than the fidelity mirror this spec measures.

## Owed

- The 18-workload production token gate at the frozen head is owed by the operator; this row's record keeps its
  complete token gate open until that receipt exists.
- Hidden-state parity (#3115) is owed by the BF16-MoE row and is not delivered here.
- The replay covers the NN (`vt::Matmul`) untied orientation only. A future untied-head switch to `MatmulBT`, and
  the tied `MatmulBT` branch, need their own artifact replay before either can claim the boundary.
