# Enable rocWMMA attention prefill on gfx1100

Row: `BACKEND-ROCM-RDNA3-WMMA-ATTN`.
Parent: `BACKEND-ROCM`.
Issue: `ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ`.
Base: `31509d91f`.
Integration: one pull request, following the repository default.

## Now

`ACTIVE`. The developer requests architecture admission, correctness, compiled
resource inspection, performance measurement, and end-to-end validation.
The developer explicitly forbids subagents for this task. This session performs
implementation and validation. Independent human review remains due at the MR.
The existing quantized admission is separately implemented in PR #3187.

Default gfx1100 WMMA and the corrected scalar control pass the unchanged
original and expanded token gates at cache blocks 16 and 32, both with graph
replay and eager execution. The integrated matrix passes 16 configurations
and 3072 output tokens. All 20 decode fixtures are byte-exact. The 20 focused
regressions include the downloaded Gemma 1B checkpoint bodies. All four decode
templates and prefill use zero scratch on gfx1100. gfx1200 and gfx1201 compile.

DOT2 arithmetic and fully masked tiles explain the scalar failures. A dedicated
128-thread decoder, captured Gemma forward, and exact normalization/selection
changes close the measured decode deficit. Incremental staging and device RoPE
construction remove unused host storage. Three final Release comparisons clear the
primary's sampled latency and throughput floors. Block-16 decode is 66.58
versus 66.48 tokens/s, at parity. Block-32 decode is 70.01 versus 44.65 tokens/s.
The prior unoptimized performance report is superseded and retained.

The branch integrates main ea6da1f80. PR #3195 carries the final evidence.
All 657 affected host translation units compile. Final HIP compilation and
the repeated 20-test regression set pass. Independent human review remains due. The row stays ACTIVE until the work lands.
[Measured report](../../docs/bench-evidence/rocm-rdna3-attention-wmma/README.md).

### Bundle the regression fixtures (15 September 2026)

Package the 44 frozen tensors into one `cases.bin` and one JSON manifest,
following `tests/fixtures/rocm_attn_gate_split`. Preserve every tensor byte,
shape, dtype, source pin, and SHA-256. Store each tensor's offset and byte
count. The test reader must validate the bundle size and slice bounds.
Consolidate expression and RoPE export in the existing primary generator so
regeneration produces the same layout. Compare all slices against the prior
commit, then rerun the CPU and physical ROCm compiled-Gemma tests. Keep the
runtime implementation and numerical assertions unchanged.

### Pin the embedding bounds readback (15 September 2026)

The integrated block-16 medians are 66.4774 native and 66.4976 primary tokens
per second. One native outlier lowers the median below the literal floor.
Preserve all samples. The bounds check still copies its 16-byte result into
pageable stack memory before synchronization on every token. Test a fixed
pinned host record per queue through the same shared scratch bookkeeping.
Keep invalid-ID errors synchronous and clear the device record on each call.
Validate alternating invalid/valid IDs and both integer widths, then repeat
unchanged exact model gates and performance. Reject the candidate if it does
not reduce measured overhead. Do not change the summary statistic or floor.

### Captured pool ownership witness (15 September 2026)

The exact-token workload detects deleted graph dispatch and stale uploads.
Deleting scratch pinning stays token-green because these requests do not hold
another allocation that aliases the captured temporary storage. Preserve that
negative mutation result. Add a test probe through the public completion ABI:
after the first request captures decode, hold pool allocations from every
recorded decode demand class, fill them with canaries, replay later requests,
and verify their bytes remain unchanged. Return allocations through the pool.
The probe must pass on the retained implementation and fail when the production
pin call is removed. It tests an overlapping allocation lifetime without
freeing graph addresses or relying on a GPU memory fault. Exact model outputs
still use the existing primary manifest and references. No golden changes.

## Scope

Admit gfx1100 to the existing BF16 SharedK rocWMMA attention prefill kernel.
Preserve the current gfx1200 and gfx1201 behavior. Keep other gfx11 targets
excluded. Preserve head dimension 256, query-to-KV head ratio two, one request,
at least 64 query tokens, BF16 buffers, and the existing scalar override.
Enable gfx1100 by default after the unchanged expanded gate passes.
Preserve the gfx12 default. The gfx1100 single-query decode prerequisite
operates independently of the prefill override.
Do not enable the deferred dimension-512 WMMA arm or alter quantized dispatch.
The quantized implementation in PR #3187 remains a separate reviewed change.

## Sources and design

The local implementation is
`src/vt/rocm/rocm_paged_attn.hip::PagedAttnPrefillSharedKWmma`.
Its 16 by 16 by 16 BF16 fragments accumulate FP32. Loads and stores use
rocWMMA's public operations. The gfx1100 MMA adapter aligns packed inputs
with the executing primary, as specified below. Consumers read ordinary
shared-memory matrices and never index accumulator coordinates.
Installed rocWMMA 2.2.1 provides the corresponding gfx1100 operation in
`internal/wmma_impl.hpp`, using the gfx11 BF16 wave32 builtin.

Use a new attention-specific architecture predicate that accepts gfx1100 plus
the existing gfx1200/gfx1201 predicate. Preserve the old predicate because
quantized dispatch also uses it on this base. Admit gfx1100 at the device
include guard and use the new predicate at the attention runtime launch guard.
Neither guard alone supplies a working implementation.

The primary pin is vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.
Read and execute `vllm/v1/attention/ops/prefix_prefill.py::context_attention_fwd`
on identical exported Q/K/V, cache layouts, windows, lengths, and softcaps.
Use the head-256, query/KV-ratio-two parameters from
`tests/kernels/attention/test_prefix_prefill.py`. That suite uses F16 inputs.
Record the BF16 input extension needed to exercise this existing BF16 kernel,
and preserve the upstream reference and tolerances. Local fixtures cover 64 and nonmultiple-of-16 query
lengths, causal and sliding masks, reordered blocks, and softcaps.

The existing attention-parity spec records that WMMA arithmetic still owes a
comparison with the primary. An architecture guard change does not discharge
that debt. Capture any numeric difference, and repair only a demonstrated
defect needed by this admission. Do not widen the correctness tolerance.
All model-path buffers keep their existing BF16 format. FP32 accumulation
remains the matrix operation's existing accumulation type.

The real-model artifact is `unsloth/gemma-3-4b-it` at
`bf46152c47f5dd20b896357cb51abc4c03b8ee8c`. It has eight query heads, four KV
heads, and head dimension 256. Its two BF16 shards contain a multimodal wrapper.
For the text-only public entry point, export only `language_model.*` tensors,
remove that prefix, and flatten `text_config` to `Gemma3ForCausalLM`.
Preserve every retained tensor's bytes and all text configuration values.
Run the primary with the identical exported artifact. Record original shard
hashes and exported tensor hashes. This is a harness adaptation, not a loader
change or a newly claimed multimodal capability.

## Gates

### Complete the local RDNA3 gates (15 September 2026)

ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ owns this continuation. Preserve the
original and expanded workloads, both cache block sizes, and strict tokens.
The scalar override must remain a real scalar control. Identify its first
arithmetic difference against captured primary attention before changing it.
Probe hardware accumulation with fixed operands when reduction order alone
does not explain the result. Preserve failed candidates and intermediate bytes.

Trace the expanded block-16 workload on both engines with rocprofv3. Separate
initialization, prefill, and matching decode steps before ranking costs.
The current decoder launches the 512-thread prefill kernel for one query.
Investigate a dedicated decode realization through the existing attention seam,
retaining the primary's tile boundaries, accumulation, and BF16 stores.
Use new hardware-specific files for new kernels. Measure graph execution,
launch gaps, and GEMM costs before attributing the full-model deficit.

Inspect host allocation lifetimes, including unused CPU RoPE caches and source
weight mappings. Remove redundant allocations without changing resident tensor
ownership or other backend behavior. Compare equal workload and KV capacity.
Download the authorized Gemma 1B checkpoint and execute the existing regression
bodies. Keep external hardware limits separate from local implementation work.

Capture a failing focused case for each repair, then require focused green,
unchanged model tokens, compiled resources, and repeated idle performance.
Resolve the current main conflict and rerun applicable gates before publishing.
Do not accept a documented local failure as task completion.

### Gemma decode graph design (15 September 2026)

The matched Release trace now places 1054 native attention calls at 197.761 us
median versus 200.582 us in the primary. The full decode span still exceeds
the primary: 580.720 ms versus 557.097 ms. Native kernels occupy 509.813 ms
versus 501.899 ms. The remaining launch gaps require graph execution.

Extend the platform's architecture-aware graph policy without changing its
existing unqualified answer. Admit only the tested linear-RoPE Gemma3,
gfx1100, one-request, one-query decode path. Keep prefill and other models on
their current execution routes. Honor the shared graph kill switch.
Use `vt::GraphCaptureScope` in full mode and `vt::BreakableGraph` for lifetime,
replay, and errors. Reuse the existing layer forward; never duplicate its
arithmetic. Keep embedding and input refresh outside capture. Own stable
metadata and device inputs for the graph lifetime. Warm the exact region,
measure pool demand, and pin its scratch so an intervening prefill cannot
reuse a captured address. Reset before changing captured shapes or KV owners.
Propagate capture failures; unexecuted graph outputs cannot become logits.

The unchanged 96-token and expanded 256-token gates must pass with graph
replay and its eager control at cache blocks 16 and 32. The expanded sequence
alternates request lengths and crosses the sliding-window boundary. Trace
actual graph launches and kernel dispatches, and test repeated request reuse,
cache-table shape changes, graph opt-out, and captured scratch ownership.
Repeat idle performance after exact outputs. This remains part of
ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ and the same reviewable MR.

### Decode output-column partition (15 September 2026)

The retained graph, selector and bounds-record changes reach 65.50 and
65.57 tokens/s, against the current primary's 66.57. Test two workgroups per
KV head, each computing 128 of the 256 output columns. Each group retains
the complete QK reduction, both query heads, and the primary's key-tile loop.
PV accumulates the same terms in the same order for each owned coordinate.
No partial-output reduction or new rounding boundary is needed. The groups
write disjoint output columns through the existing attention entry point.

Require all twenty decoder fixtures to remain byte-identical to the primary,
including NaN cache tails, non-power-of-two blocks, and windows. The runtime
witness must require the new grid's two column groups, and static resources
must stay spill-free for every admitted tile. Retain this partition only if
repeated end-to-end measurements improve after exact model output.

The output-column partition preserved all twenty exact fixtures but regressed
block-16 decode to 64.06 and 60.86 tokens/s. Reject it. The next candidate
computes QK for two consecutive key tiles in parallel waves, then processes
the two original softmax/PV tiles in their original order. Pair tiles only
for block sizes 16 and 32; retain one tile for block 64. QK coordinates are
independent across keys. The candidate must preserve each score's D reduction,
each online softmax update, and each PV accumulator's order. Apply the same
exact-output, resource and repeated-model gates as the rejected partition.

The paired-QK candidate also regressed: 59.57 and 59.59 tokens/s at block 16.
Retain the unpartitioned zero-spill decoder. Trace differences outside attention
include one QKV split and two Q/K normalization launches per layer. For the
compiled Gemma single-token path, use contiguous views of the merged QKV
owner and keep that owner alive through attention. Multi-token and unmerged
paths retain materialized buffers. This removes a copy without changing bytes.

Combine the compiled Gemma Q and K row kernels through the existing
`AttnQkNormRope`/`FusedChain` provider. Reuse one row implementation. Preserve
its reduction order: four positions per lane, eight adjacent lane values
summed sequentially, then an eight-group 4/2/1 fold. Warp shuffles can replace
block barriers while preserving each addition. Require the frozen primary
Q/K fixtures, non-Gemma regressions, both model block sizes, and actual traces.

### Remaining decode launch costs (15 September 2026)

The final block-16 comparison still owes about 0.38 ms per output token.
The trace places the 262208-column greedy selector at 138.7 us on one
256-thread block. Test a larger block for this low-batch gfx1100 shape,
through `vt::GreedyArgmax`, retaining the existing comparison and lowest-index
tie rule. Check ragged vocabularies, ties across lanes, BF16 and FP32 inputs,
and the unchanged model continuations before accepting a speed result.

The embedding wrapper allocates and frees its bounds record synchronously
on every step. Keep its checked `vt::Embedding` contract and its position
outside the decode graph. Test stream-ordered allocation, copy and release,
with exception-safe cleanup. Preserve rejection of negative and upper-bound
IDs, repeated calls after rejection, and both index widths. Retain either
change only after its focused tests and repeated model measurements pass.
Stream-ordered allocation preserved tokens but regressed block-16 decode to
54.77 and 56.74 tokens/s. Reject that allocation candidate. Test the existing
`vt::GrowOnlyStreamScratch` seam instead: one 16-byte bounds record per queue
identity, cleared before each use, with stream synchronization before return.
The shared seam retains its allocation for process lifetime. A changed queue
identity gets a separate record; a native stream-handle reuse cannot alias it.
The 1024-thread selector and merged preamble bring block-16 TPOT to 15.07 ms,
near the current primary's 15.02 ms. Test 32 independent vocabulary partitions
for at most four gfx1100 rows with at least 65536 columns, then reduce their
(value, original-index) winners. Maximum selection adds no floating-point
rounding; preserve the lowest original index on every tie. Use the shared
scratch seam, keyed by queue identity, with a fixed 128-partial bound and
explicit prewarm refusal during capture. The existing small-row route stays.
Run the large/ragged/tie/all-negative-infinity cases and exact model gates.
These repairs belong to ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ.

### Expanded-gate repair design (15 September 2026)

The developer requests resolving the opt-in blockers and publishing a ready MR.
ISSUE-LOCAL-01M2HQEEXHD2B0BT3N71HQ0CRZ owns the remaining numerical repair.
The executing primary at the same pin exposes three materialized boundaries
that the original Gemma implementation does not mirror:

- `activation.py:451-464` selects erf GELU on ROCm. Its generated GeGLU kernel
  retains the activation in FP32 through the gate/up multiply.
- The generated Q/K preamble combines Gemma normalization and cached rotation
  before its single BF16 output store. Extend the existing gate-free
  `AttnQkNormRope` realization to honor its Gemma flag and route Gemma through
  `FusedChain` with the corresponding recipe.
- The generated sandwich norms evaluate `Npost(a) + base`, then
  `Npost(delta) + (Npost(a) + base)` in FP32. They round only the normalized
  output and the next-layer residual to BF16. Preserve BF16 operand ownership
  across the MLP and extend the typed residual-expression/FusedChain seam to
  represent normalized operands. Do not allocate a persistent FP32 residual.

The generated module `cycnctzmkc6yjxbdnjsayvhkrt7wcw2yzliovagbbvctjna42lab.py`
contains the intermediate-layer stores and returns its BF16 residual. Module
`cduy4zdp3dldo7p3yvkxoff27y2ioqz3atgb4yqnwofcobarhk5l.py` contains the final
norm, without a residual consumer. Both are retained in the task's primary
compiler cache and will be sealed with hashes in the evidence.

Intermediate capture additionally establishes the initial embedding boundary:
the first norm's variance uses the FP32 scaled embedding, while its numerator
reloads the BF16 scaled embedding. Add a typed scaled-norm operation through
`FusedChain`; its outputs remain BF16. Matching this boundary made layer zero's
QKV input and output byte-identical for the failing unique7 prefill and decode.

The affected attention paths also owe the primary's aligned key tiles and
BF16 probability conversion before PV. Update dimension-256 SharedK scalar,
SharedK WMMA, and GQA decode with the executing ROCm attention softmax
expression. Preserve dimension-512 behavior. Export actual layer-zero Q/K/V
and attention outputs to distinguish kernel arithmetic from model-front-end
differences. A scratch FP32-buffer experiment is diagnostic evidence only;
the implementation must keep BF16 buffers and use typed shared fusion calls.

The executing chain is `rocm_attn.py:459-480`, then
`chunked_prefill_paged_decode.py`. Prefill enters `prefix_prefill.py::_fwd_kernel`;
decode enters `kernel_paged_attention_2d`. Current-chunk prefill uses 64 keys,
cached-prefix prefill uses 32, and decode uses `min(block_size, 128)`.
The primary production cache block is 16. Set the native gate harness to 16
for the identical workload, and retain separate coverage of its default 32.
The pinned ROCm backend passes `sliding_window - 1` into a strict-distance
kernel mask. Adapt this at the Gemma model boundary; the shared inclusive
window contract remains unchanged.

The generated ISA uses BF16 WMMA for QK and PV. With exact layer-zero Q/K,
unique4 still differs in seven prefill outputs and eight first-decode outputs.
Test a WMMA PV accumulator against these captures. Reuse the Q tile storage
for accumulator rescaling after QK, reload Q for each next tile, and reuse
score storage for BF16 probabilities after every score reader synchronizes.
Keep FP32 accumulator fragments in registers and keep total LDS below 64 KiB.
Use public rocWMMA load/store operations for accumulator layout conversion.
If necessary, carry the same matrix arithmetic into dimension-256 GQA decode.
Record this prerequisite separately from prefill acceleration and test its
resources and performance. Do not change dimension-512 or quantized paths.

A 15 September operand capture identifies a gfx11 rocWMMA 2.2.1 difference:
`PreMmaXFormA/B` swap the two eight-element halves in lanes 16 through 31.
The executing Triton kernel duplicates lanes 0 through 15 without this swap.
Both matrices remain mathematically equivalent, but 1,936,922 of 5,696,064
captured QK scores differ by up to 1.526e-5. The diagnostic primary kernel
reproduces its original output byte-for-byte, so the comparison is gateable.

Add a gfx1100-only adapter around rocWMMA's packed MMA input seam. Keep its
input/output transforms, accumulator representation, and matrix instruction.
Broadcast the lower-half packed inputs into the upper half before MMA.
This requires rocWMMA's header implementation types; record the tested 2.2.1
version and isolate that dependency in one architecture-specific header.
Do not index accumulator coordinates or alter gfx12's public MMA operation.
Verify the operand capture, score intermediates, token gates, and resources
before accepting this adapter. Retain the AMD license for adapted glue.

The complete first-layer QKV projection is byte-identical on unique4, but
its CPU-built RoPE cache differs from the executing primary's GPU-built cache.
`rotary_embedding/base.py:89-112` explicitly builds inverse frequencies and
trigonometric values on the device. A HIP FP32 prototype matches all 33,554,432
local and 268,435,456 global BF16 cache values. Substituting those cache bytes
restores unique4's 16-token diagnostic sequence.

Extend `RopeCosSinCache` with an explicit linear factor. Positive factors use
the primary's FP32 reciprocal-power, position division, and trigonometry.
Factor one expresses the unscaled local cache. Zero preserves existing callers.
Extend resident tensor initialization with a callback for generated device data.
Gemma's compiled path uses that shared seam to build each cache once on-device,
then narrows it to BF16. Keep the CPU loader cache for materialized backends.
Test both theta values, scaled positions, invalid factors, and full-model tokens.

The completed compiler inspection also fixes reduction and contraction order.
Q/K norms use 64-element tiles and eight adjacent elements per lane. Hidden
norms use 1024-element tiles and four adjacent elements per lane, reduce each
half-wave before combining halves, and divide by width before adding epsilon.
The second rotary coordinate contracts `second*cos + first*sin` with the
first product fused. Real-model Q/K and sandwich outputs now match exactly.

Decode source spells `acc += dot(P,V)`, but the executing Triton IR folds
that addition into the dot accumulator. Preserve the scaled accumulator in
WMMA. This restores the expanded WMMA gate to 256/256 matching tokens.

The dimension-256 decode repair is a correctness prerequisite on gfx1100.
Enable it independently of the prefill-only environment switch, for a single
query and one request with cache blocks at most 64. Keep the existing gfx12
decode selection. Both prefill A/B arms then use the same repaired decoder.
Retain the existing GQA/decode switches as their broader controls. The trace gate must
separate multi-workgroup prefill launches from single-workgroup decode
launches. A shared kernel name alone no longer proves prefill reachability. Do not admit
short multi-token prefill through this prerequisite. Test that the default
prefill path launches on gfx1100 and that its explicit disable retains the
scalar prefill control. Repeat both full-model gates after this separation.

Add focused primary-generated fixtures for these expressions, including
nonuniform gamma, BF16 rounding boundaries, full rotary dimensions,
and two/three operand residual expressions. The executing Gemma partition
rotates all 256 coordinates. Generic rotary tests cover partial dimensions.
Do not describe those generic tests as compiled Gemma parity evidence. Capture red before each repair,
then green and production reachability. Preserve existing backend defaults
outside the measured ROCm Gemma3 path. Run the unchanged original 96-token
and expanded 256-token gates with scalar and WMMA controls. Investigate any
remaining differences; these source findings alone do not prove token parity.

1. Before admission, the CPU predicate case rejects the requested gfx1100
   target and a physical production-dispatch witness fails to observe WMMA.
2. Compile the actual translation unit for gfx1100. Inspect generated ISA for
   BF16 WMMA and record VGPR count, spills, private bytes, and LDS bytes for
   the production specialization. Compile gfx1200 and gfx1201 as controls.
   A spill is measured debt, not an assumed architecture incompatibility.
3. Run the existing frozen SharedK fixture through `vt::PagedAttention` and
   compare finite output against its declared oracle. Extend physical cases
   through the same entry point for tails and masks. Run enabled and disabled
   controls in separate processes because the environment is cached.
4. Execute the pinned primary on the same arrays. Record absolute errors,
   output dtypes, and tolerance verdicts. Preserve every failed attempt.
5. Enter through the public load/completion API using a deterministic model
   fixture and a pinned real Gemma 3 4B text checkpoint. Trace the call site,
   compare generated IDs, and retain logits when tokens differ. The user
   authorized downloading weights on 14 September 2026. Do not claim a
   synthetic fixture establishes full-checkpoint correctness.
6. After correctness, run same-binary scalar/WMMA comparisons on an idle local
   RX 7900 XTX under `/home/vikash/gpu.lock`. Alternate order across repeats.
   Record prefill and decode rates, latency, memory, clock samples, and boot ID.
   Trace both primary and native with rocprofv3. Below-floor axes remain gaps.
7. Run CPU architecture tests, the HIP attention and cross-device tests, and
   full preflight. Qualify baseline skips and failures. Mutate admission and
   the production launch in a scratch copy and prove the focused gate fails.
   This session's mutation checks do not claim independent review.

## Risks and stop conditions

rocWMMA's gfx11 fragments need more input registers than gfx12 fragments.
Compilation can succeed yet spill or reduce occupancy. Shared-memory and
barrier assumptions need physical execution. BF16 arithmetic can change token
selection even when a float reference tolerance passes.
Keep the gfx1100 default disabled if correctness fails or the measured path
regresses. Record the exact failure and required repair. Do not hide a failure
behind a skipped test, a CPU fallback, or a model that misses the call site.
Do not merge without independent review. Prepare a reviewable MR only after
the stated end-to-end gate passes, or report the measured blocker precisely.

## Evidence

Keep one validation summary under `docs/bench-evidence/rocm-rdna3-attention-wmma/`.
The developer requested removing detailed receipts from the implementation diff.
Preserve the complete report and receipts at commit `b962497b67c7f82d331aa657b25cc5c96533d611`,
retained by the fork tag `evidence/rdna3-wmma-2026-09-15`. The summary links there.
Retain raw arrays, traces, compiler output, and logs in the ignored build directory.

## Row inventory

| ID | Upstream source | Local anchor | Tests and evidence | Spec | State | Owner | Issue |
|---|---|---|---|---|---|---|---|
| `BACKEND-ROCM-RDNA3-WMMA-ATTN` | vLLM prefix prefill at e126687a9a; rocWMMA 2.2.1 gfx11 BF16 | `PagedAttnPrefillSharedKWmma`, `compiled_gemma`, Gemma `FusedChain`, resident cache initialization | Frozen primary expressions, device caches, public token/trace gates, resources, same-binary A/B | [This spec](rocm-rdna3-attention-wmma.md) | `ACTIVE` | Codex, single-agent user direction | `ISSUE-LOCAL-01M2HK0GFJDRXXAAA8F0014XQQ` |
