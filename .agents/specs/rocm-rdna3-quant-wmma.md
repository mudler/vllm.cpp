# Reuse quantized WMMA prefill on gfx1100

Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3`.
Parent: `KERNEL-QUANT-CIQ-GEMM-ROCM`, which retains its RDNA4 work.
Issue: `ISSUE-LOCAL-01M2F0PQWGSCXG0N4951NF9DPZ`.
Base: `cef9f821632eb33156bb7d24a0a85eeab7f7484a`.
Integration: one pull request, selected by the developer on 13 September 2026.
Commit this specification before implementation.

## Now

`DONE` for architecture admission, 13 September 2026.
Implementation `c3fe98ba6c55ce71e75746e1b944a27640464e0f` passes G1-G3 and
fresh mutation review. Both executing tile bodies remain unchanged.
The operator independently reproduces the hardware, public, and 240 original
primary fixture gates. G5 retains its explicitly recorded baseline resource skips.
G4 executes both task-pinned oracles and six native processes with exact input
and generated token IDs. Native traces prove prefill-only WMMA dispatch.
The observed prefill ratio is 1.2891 with WMMA enabled.
Full-model floors remain failing, and accepted clock attribution remains pending
under the dedicated issue in `## Owed`.
[Evidence report and archive](../../docs/bench-evidence/rocm-rdna3-quant-wmma/README.md)
retain commands, results, mutations, and measurement limits.

## Scope

Enable the existing 16 by 16 quantized prefill tiles on gfx1100 through the
ordinary `vt::MatmulBTQuant` dispatch. Preserve Q4_K and Q6_K block decoding,
Q8_K activation quantization, output dtypes, scales, and remainder handling.
Preserve the existing `VT_ROCM_QUANT_WMMA=0` scalar control.
Keep admission bounded to gfx1100 and the already admitted gfx1200/gfx1201.
Do not enable other gfx11 devices without their own evidence.

This is an architecture-admission change, not a new model or arithmetic port.
The user selected it independently of the broader Qwen state characterization.
Do not add graph activation, attention changes, new formats, grouped expert
tiles, kernel redesign, or a general numerical-characterization dependency.

## Sources and hypothesis

The parent [kernel spec](kernel-quant-ciq-gemm-rocm.md) records the existing
implementation and its oracle hierarchy. Current local source uses generic
rocWMMA fragments in `src/vt/rocm/rocm_grouped_gemm.hip::KQuantGemmKWmmaQ6K`
and `src/vt/rocm/rocm_grouped_gemm.hip::KQuantGemmKWmmaQ4K`.
Each operation loads fragments through `load_matrix_sync`, computes with
`mma_sync`, and stores a row-major tile through `store_matrix_sync`.
The scalar epilogue reads that stored tile, not the hardware register layout.

Installed rocWMMA 2.2.1 defines the signed-int8 gfx11 specialization in
`internal/wmma_impl.hpp:602-634`. It selects
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` and an int32 accumulator.
Its `internal/config.hpp:222-226` requires wave32 and block dimension 16.
The existing kernel uses both. The installed compiler is HIP 7.15.26333,
Clang 23 at revision `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.
Record the compiler and header hashes with actual build evidence.

Pinned llama.cpp `b10451`, revision `10bf611e5`, confirms the distinct RDNA3
operand layout in `ggml/src/ggml-cuda/mma.cuh::get_input_data_layout` and its
RDNA3 integer WMMA specialization. rocWMMA owns that difference here.
The source is available at the exact revision through the upstream repository.
The developer explicitly selected current upstream main/master for this
experiment on 13 September 2026, instead of the previously installed oracles.
The fetched task snapshots are vLLM
`39545e475d3627287ff69c25465dc0bd405f67e1`, llama.cpp
`093a2f86c3e37c54fa3e1f9efb17b304f3433abd`, and GGUF plugin
`d4c1f0d082fc7cd4350da56689109a01c1f29d6c`.
The plugin's current main is unchanged from the repository pin.
Build and execute isolated runtimes at these revisions before accepting their
results. Recheck the executing source chain there. This task-specific developer
direction does not claim reconciliation of unrelated repository goldens.

## Design

1. Add an additive quant-specific architecture policy header. Its predicate
   accepts gfx1100 plus the current gfx1200/gfx1201 predicate, with valid
   feature suffixes and rejection of malformed or unmeasured names.
2. Admit `__gfx1100__` at the quantized translation unit's rocWMMA include
   and compile guard. Use the quant-specific predicate at its runtime gate.
3. Keep `GcnArchNameIsGfx12PrefillWmma` unchanged. Attention uses that predicate
   but its gfx1100 device body is still excluded. Widening it could launch
   an empty attention kernel.
4. Reuse both tile bodies. Change a body only if a concrete failing test proves
   an architecture-specific correction is required. Report that finding before
   expanding this design. Do not change the scalar control or output dtype.

## Tests and gates

G1: Add a CPU architecture-policy test before admission. It must reject
unmeasured architectures and retain the attention rejection of gfx1100.
The existing Q4_K/Q6_K physical tests must run on gfx1100 independently of
the new production predicate. Before admission, dispatch assertions must fail.
Preserve their existing fixtures and tolerances, F32/BF16 outputs, aligned
shapes, joint tails, and asymmetric bottom/right tails. Include a partial
four-wave block because the kernels contain block-wide barriers.

G2: Build the actual HIP translation unit for gfx1100 and inspect emitted
ISA for signed-int8 WMMA. Execute registered-operation tests on the local
7900 XTX under the configured GPU mutex. Kernel output, not just a host
dispatch counter, must be checked. Re-run scalar-control outputs in separate
processes because the environment selection is cached.

Port the applicable plugin `tests/test_kernels.py::test_mmq` coverage with its
original Q4_K/Q6_K sample tensors, token counts 7/83/128/2048, hidden widths
256/1024, seed zero, and F16/BF16/F32 inputs. Preserve the upstream tolerances.
Native outputs support F32/BF16 only; explicitly narrow a native F32 result
for the upstream F16-output comparison and document that harness adaptation.
Reuse the sample verification/export helpers under `tools/rocm_quant_gather`
where useful. Synthetic local correctness tests remain tighter independent
guards; they do not replace the upstream fixture comparison.

G3: Prove production reachability through the public load/completion path on
a Q4_K/Q6_K fixture or the recorded Qwen3.5-4B Q4_K_M checkpoint. Use prompt
lengths that enter prefill tiles, including a nonmultiple of 16. Record completed
tokens and both format dispatch counters or matching profiler kernel traces.
Deleting each launch or restoring gfx12-only admission must fail this gate.

G4: After correctness, compare the same binary with the environment unset and
with `VT_ROCM_QUANT_WMMA=0`. Use identical weights, prompts, token counts,
batching, and sampling. Record prefill, decode, latency, and memory on an idle
host with interleaved repeats. Run the applicable pinned primary workload and
the quant-matched llama.cpp floor before accepting a performance conclusion.
The retained checkpoint is `Qwen3.5-4B-Q4_K_M-unsloth-e87f1764.gguf`, SHA256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`, from
`unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523`.
Verify its hash before use. A missing oracle resource stays explicitly pending.

G5: Run the full declared local gate: the HIP `rocm|cross_device` CTest set,
the production reachability case, CPU architecture tests, and the full
`scripts/agent-preflight.sh --staged`. Qualify skips and pair baseline failures.
A fresh reviewer inspects the immutable head, mutates admission and both launch
sites in a scratch worktree, and restores source hashes after each mutation.
The operator independently reruns the focused hardware and production gates.

## Risks and stop conditions

The rocWMMA implementation can compile but use a wrong fragment layout.
Output comparisons and ISA inspection distinguish that failure from admission.
Partial blocks can expose barrier assumptions. Tail tests must complete.
An instruction-path win need not improve full-model prefill. Measure both.
Retain the existing behavior on every excluded device and on decode.
If a new kernel algorithm is required, report `NEEDS_DECISION` with evidence.
Do not silently broaden the row or claim an unrun gate passed.

## Evidence packaging

`ISSUE-LOCAL-01M2GNY58NHBK3D4JQQ738M6GR` owns the developer-requested
packaging correction of 14 September 2026. Keep this correction in the existing
pull request. Preserve every implementation, test, and validation-harness byte.

Archive all 236 evidence files from
`d6e40c91f634a041c873c7a04516d55c4d05772a` with their original relative paths.
Publish the archive and checksums as assets of a dedicated release on the
developer's fork. Pin that release to the source revision. Verify anonymous
download, the archive checksum, and every extracted file against the source tree
before replacing the raw files in the diff.

Keep one concise evidence report with the download link, source revision,
SHA256, extraction instructions, correctness results, model results, review
verdicts, and all material performance and gate qualifications. Update this
specification and its issue references to that report. Detailed values, failed
attempts, commands, and original manifests remain in the archive.

The packaging gate checks archive completeness and bytes, repeatable archive
creation, safe member paths, and detection of missing or altered files. It also
checks retained links, canonical records, diff classification, commit style,
and equality of product, test, and harness files with the reviewed head.
A fresh reviewer verifies the immutable packaging commit and downloaded archive.
The operator independently repeats the packaging gate. Reuse the recorded
hardware results when their executing files remain byte-identical. Stop if
publication or preservation cannot be verified. Do not weaken a checker or
reclassify pending measurements as passing.

On 14 September 2026, the published archive preserves all 236 original files.
The [retained report](../../docs/bench-evidence/rocm-rdna3-quant-wmma/README.md#retrieve-the-complete-evidence)
pins the source revision, download, checksum, and extraction procedure.
The operator verified anonymous download and every original byte before replacement.
The original detailed reports and their qualifications remain unchanged in the archive.

## Owed

`ISSUE-LOCAL-01M2F4WCD6ZK5VH5S8TF83APD6` owns the remaining full-model
performance gaps on the retained gfx1100 workload. It includes every below-floor
axis, matching oracle traces, comparable timing windows, and accepted clock
attribution. The [evidence report and archive](../../docs/bench-evidence/rocm-rdna3-quant-wmma/README.md)
retain all measured values, ratios, and limits. This debt is separate from
architecture admission and from the deferred Qwen state characterization.

Other formats, grouped expert tiles, and other RDNA3 devices keep their prior
owners. The parent RDNA4 row and its open cooperative-tile work remain separate.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA3` | Pinned llama.cpp RDNA3 integer WMMA; rocWMMA 2.2.1 gfx11 fragments | `KQuantGemmKWmmaQ4K`, `KQuantGemmKWmmaQ6K` | [G1-G5 receipts](../../docs/bench-evidence/rocm-rdna3-quant-wmma/README.md), independent review and operator verification | [This spec](rocm-rdna3-quant-wmma.md) | `DONE` | RDNA3 helper; coordinating operator | `ISSUE-LOCAL-01M2F0PQWGSCXG0N4951NF9DPZ` |

## Outcome

The existing generic rocWMMA tile bodies execute correctly on physical gfx1100.
Architecture admission is sufficient. No arithmetic or fragment-layout repair
is needed. The Q4_K and Q6_K bodies retain their original byte hashes.
The original 240 plugin cases preserve every input mode and tolerance.
The public gate matches 1024 logits and eight completion tokens against scalar.
Independent mutations detect missing admission, missing launches, scalar-control
changes, and finite corrupted output. The operator repeats the applicable gates.

The full-model workload uses eight requests with 183 and 174 input tokens in
alternating order. Exact input arrays and all 128 generated IDs match the primary
and llama.cpp captures. All six native process outputs match both oracles.
The enabled trace contains 152 WMMA calls per prefill and zero during decode.
The scalar trace contains no WMMA calls.

Enabled and disabled median prefill rates are 224.12 and 173.86 tokens/s.
Median first-token latencies are 386.51 and 617.09 ms.
Mean per-stream decode rates are 53.33 and 53.15 tokens/s.
Sampled whole-device memory peaks are 4.864 and 4.857 GB.
These dynamic-clock observations motivate retaining the enabled default.
They do not establish clock-attributed performance or full-model parity.
The dedicated performance issue retains every below-floor axis and missing
measurement obligation. No apparent performance limit is accepted.

The default architecture predicate adds only measured gfx1100 to gfx1200/gfx1201.
Other gfx11 devices remain excluded because they lack their own physical evidence.
The attention predicate remains unchanged because its gfx1100 body is not enabled.
The scalar environment override remains available for reproducible controls.
Output dtypes, quantization formats, and tile arithmetic retain their prior values
because this row proves admission without changing those contracts.
