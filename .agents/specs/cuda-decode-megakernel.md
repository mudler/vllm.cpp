# KERNEL-CUDA-DECODE-MEGAKERNEL — tile-scheduled persistent decode spike

**Row:** `KERNEL-CUDA-DECODE-MEGAKERNEL` (`kernel-matrix`).
**Issues:** [#3084](https://github.com/mudler/vllm.cpp/issues/3084) owns the
spike; [#3085](https://github.com/mudler/vllm.cpp/issues/3085) owns the record
gate repair discovered while adding the row; and
[#3099](https://github.com/mudler/vllm.cpp/issues/3099) owns the claim-state
consistency repair found by the final roadmap review.
**Kind:** roadmap intake and bounded research spike. This spec ships no kernel,
model path, dependency, benchmark, or user-visible capability.

## Now

`SPIKE`. The roadmap records a hypothesis, not a performance result. The next
action is W0: trace identical low-batch decode work in the pinned vLLM oracle
and vllm.cpp after CUDA Graph replay. W1 is permitted only when W0 shows
material recoverable idle capacity across operation boundaries.

## Git integration

The spec and roadmap entry use one integration change, with this spec committed
before the matrix edit. The developer authorized the operator to merge this
roadmap-only change directly to `main` and push it on 2026-09-08. The authority
does not extend to a kernel implementation, a dependency change, or a
force-push.

## Problem

Cohere's research engine replaces the ordinary sequence of decode launches with
one persistent CUDA kernel. One block stays resident on each streaming
multiprocessor (SM), consumes host-built tile tasks, and crosses operation
boundaries when dependencies are ready. This may recover partial-wave and
barrier gaps that CUDA Graphs do not remove.

vllm.cpp already removes launch submission overhead with CUDA Graphs and exposes
operation-local composition through `vt::FusedChain` and
`vt::MergedGemmGroup`. It does not expose a cross-operation tile scheduler. The
unresolved question is whether a representative vllm.cpp decode path leaves
enough device-side idle capacity after graph replay to justify that additional
execution model.

## Scope

In scope:

- Pin and inspect the complete executing path in Cohere's release.
- Trace the same low-batch decode workload in vLLM and vllm.cpp with CUDA Graphs
  enabled in each production configuration.
- Attribute launch gaps, wave tails, achieved memory bandwidth, SM occupancy,
  and idle intervals with the same tracing tool on both sides.
- Describe a fixed-size CUDA task ABI: descriptor fields, operation codes,
  dependency counters, queue ownership, and static and dynamic scheduling.
- If W0 passes, build one synthetic persistent-kernel proof that schedules at
  least two existing kernel families through the proposed ABI.
- Preserve eager and CUDA Graph fallbacks for every unsupported architecture,
  dtype, model, batch, shape, and quantization arm.
- Repair the record checker so adding this keyed row does not require a stored
  cardinality update in another tracked file.

Out of scope:

- Model integration or a production entry-point change in this spike.
- A universal execution model for CPU, ROCm, Metal, Vulkan, or Tenstorrent.
- Vendoring ThunderKittens or adopting Cohere's task ABI without an explicit
  dependency and ABI decision.
- Treating an H100 BF16 result as evidence for another GPU, dtype, model, or a
  required GGUF k-quant arm.
- A throughput claim before token correctness and identical-workload A/B gates
  pass.
- A migration of the shared matrices to one-file-per-row storage. That larger
  record-surface change remains separate from the narrow derived-at-read-time
  repair in #3085.

## References and pins

### Primary correctness oracle

vLLM remains the only primary oracle. The active pin is
`e126687a9a828d513c01a07cd69f025f27d63280` in
[`../upstream-sync.md`](../upstream-sync.md). Any later implementation must read
and run the complete vLLM execution chain at that pin. This intake does not move
the pin and does not claim that its currently owed token revalidation has run.

### Performance reference

The Cohere source is pinned at
[`67d0b9ca22ea3652796b715d1d1863459e0e2c3c`](https://github.com/cohere-ai/cohere-megakernel/tree/67d0b9ca22ea3652796b715d1d1863459e0e2c3c).
It is a performance reference, not an oracle and not a mirror source. Its README
records:

- one H100 (`sm_90a`), CUDA 13+, BF16, and batches 1–8 as the tested envelope;
- one persistent block per SM, a host-built task list, and global dependency
  counters;
- 292 tok/s at batch one versus 185 tok/s for vLLM v0.24 on its isolated 8K
  decode workload, and 1.25–1.41x on its end-to-end suites;
- PyTorch prefill, a North Mini Code-specific schedule, and ThunderKittens tile
  primitives as present limitations.

Those are upstream claims. This project has not reproduced them. The first W0
record must cite the precise scheduler, descriptor, barrier, and kernel-dispatch
source lines that execute at this pin; README prose alone cannot pass W0.

### Local anchors

- CUDA graph capture and replay: `include/vt/backend.h` and
  `src/vt/cuda/cuda_backend.cu`.
- Declarative fusion: `include/vt/fused_recipe.h`, `include/vt/recipes.h`, and
  `src/vt/ops.cpp`.
- Mergeable projection groups: `include/vt/merged_gemm.h` and
  `src/vt/merged_gemm.cpp`.
- Required production route for a later model implementation:
  `ModelRegistry::Forward`, `dense_attn::AttnBlock`, on-device sampling,
  `vt::FusedChain`, and `vt::MergedGemmGroup` where each seam applies.
- Existing Cohere2 MoE model inventory:
  `MODEL-TEXT-cohere2-moe-cohere2-moe-for-causal-lm` in
  `../model-matrix.md`. This spike does not claim that model as its first target.

## Design and work breakdown

### W0 — attribution trace

Choose one model and artifact that both the pinned vLLM oracle and vllm.cpp can
run. Record exact revisions, hashes, build recipes, prompts, context length,
generated-token count, batch, concurrency, sampling, dtype, attention backend,
and CUDA Graph mode. Trace both engines with the same tool on the same idle
leased device.

Report one result for each axis: token correctness, time per output token,
launch gaps, partial-wave tails, achieved memory bandwidth, SM active time,
peak memory, and contention state. Separate host-launch latency from device
underfill. A graph that removes host gaps but still leaves complementary
attention and expert tails supports W1; a trace dominated by full, efficient
GEMMs does not.

### W1 — synthetic ABI proof

W1 may start only after W0 passes its admission test. Specify a fixed-width,
trivially copyable descriptor and version it. Each operation code must define
its input and output tensors, tile coordinates, required counter values,
counter updates, alignment, dtype, architecture support, and failure behavior.
The host must reject unsupported descriptors before launch.

The proof uses one persistent block per SM and combines a static task order with
a bounded dynamic queue for ready work. It must schedule tiles from at least two
existing families and compare their outputs against the unchanged eager
sequence. It is test-only and synthetic: no model forward may reach it during
this spike.

### Later production work

A production decode path requires a new issue and spec after W1. It must name a
model, checkpoint, quantized arms, entry point, fallback predicates, scheduling
policy, ABI ownership, and architecture matrix. It must extend the shared seams
when necessary instead of bypassing them.

### Gate repair — derive matrix contents at read time

The red result for the first matrix edit was `60 KERNEL rows; expected 59`.
`scripts/check-agent-record.py` stores each matrix path beside a hand-maintained
row count. That is a measurement of one file stored in another file, and it
forces every row addition to edit the same checker. Both properties violate the
record-surface rules in `AGENTS.md`. The count is also not a key-preservation
gate: deleting one row and adding another leaves it green.

Issue #3085 removes the stored cardinalities for all matrices, not only the
kernel value exposed by this row. Matrix paths and prefixes remain declared;
their rows are parsed and counted only from the files being checked. Every
structural, duplicate-ID, lifecycle, issue-ownership, spec, anchor, and
matrix-specific invariant remains active.

Tests that patch or assert the stored number must instead name the guarantee
they intend to protect. A historically significant row stays protected by an
exact key assertion and its semantic cell assertions. A new regression fixture
must show that adding one structurally valid, uniquely keyed row does not
require any checker constant to change. Negative fixtures must still prove that
a duplicate key, malformed row, invalid lifecycle transition, missing claim,
or broken issue/spec ownership fails for its own reason.

The long comments that explain retired cardinality bumps are historical
evidence. They move intact to one file under `.agents/completed/`; they are not
deleted or rewritten as current policy. No replacement current count is stored
there or elsewhere.

The final roadmap review exposed one adjacent ownership defect after the
per-row claim existed. A claim's `Row IDs` cell annotates each owned row with
its lifecycle state, and the claim has its own lifecycle column. The checker
read only the row ID. Issue #3099 must parse both values, require the annotated
row state to equal the matrix state, and require a claim used by a `SPIKE` or
`ACTIVE` row to be active. Focused mutations must change `SPIKE` to `ACTIVE` in
the annotation and `ACTIVE` to `DONE` in the claim column; each must fail for
its own mismatch.

A complete read-only inventory at `03cfbdb85` found 60 claim files and 63
recognized row references. Fifty-seven annotations already match. The six
pre-existing mismatches are a finite migration, not an allowlist:

- `CLAIM-MODEL-GLM-MOE-DSA`, `CLAIM-MODEL-MUSIC3-W0`, and
  `CLAIM-SERVE-RECIPE-ARGS` annotate their live `ACTIVE` owner rows as `SPIKE`.
- `CLAIM-ROCM-GEMMA4-GETBLAS-DUALSLOT`,
  `CLAIM-ROCM-GEMMA4-INDEXED-MAX-T`, and
  `CLAIM-ROCM-GEMMA4-PREFILL-PEER-HELPER` reference the `ACTIVE`
  `BACKEND-ROCM` row without a lifecycle annotation. They are not its selected
  owner, but the claim syntax must still be complete.

The #3099 implementation changes those six annotations to `ACTIVE` in the same
commit that turns on the general check. It does not change their separate claim
lifecycle columns. The repository currently uses `ACTIVE`, `IMPLEMENTING`, and
`SPIKE` as nonterminal claim states; each counts as live when a matrix row names
the claim as owner. `DONE` does not. An unknown claim state fails instead of
silently becoming live. Tests must cover one matching example of each existing
nonterminal state, the two review mutations above, and a missing annotation.

## Tests and gates

This roadmap intake must pass:

1. `scripts/agent-preflight.sh` on the unchanged reviewed commit.
2. The repository record checks, including issue ownership and spec anchors.
3. A fresh static review confirming that the issue, spec, and matrix use the
   same row ID, state, scope, pin, and exclusions.
4. A scratch mutation that removes or changes the matrix row and demonstrates
   that the record check or an explicit exact-row assertion fails.
5. The focused #3085 suite proving matrix contents are derived at read time and
   all non-cardinality record guarantees remain load-bearing.
6. The focused #3099 suite proving matrix state, claim row-state annotation,
   and active claim lifecycle agree.

W0 must pass before W1:

1. Token-exact greedy output against the pinned vLLM oracle on the identical
   workload. If the oracle is genuinely nondeterministic, stop for explicit
   ratification of a distributional gate.
2. Identical model artifacts, prompt tokens, token count, batch, concurrency,
   sampling, dtype, and production graph configuration.
3. Same-tool trace attribution on an idle leased device, with values and ratios
   for every declared axis.
4. A material device-side idle interval that a cross-operation schedule can
   legally fill. Host launch gaps alone are insufficient after graph replay.

W1 must pass before any production proposal:

1. Descriptor validation rejects every unsupported operation code, version,
   alignment, shape, dtype, and architecture.
2. Focused output tests are byte-exact to the unchanged operation sequence.
3. A fresh reviewer mutates each claimed dependency and barrier guarantee in a
   scratch copy and observes the focused test fail.
4. Compute Sanitizer reports no memory, race, or synchronization error.
5. Same-binary A/B keeps the eager and CUDA Graph fallbacks reachable.

## Risks and decisions

- **R1 — model-specific result.** North Mini Code has parallel attention and MoE
  branches that create unusually useful backfill. W0 must not generalize that
  topology to a serial transformer.
- **R2 — architecture lock-in.** The reference is tuned for `sm_90a`. A
  Blackwell, ROCm, or other result is a separate measurement and dispatch arm.
- **R3 — scheduler overhead.** Global counters and dynamic queues may cost more
  than the recovered tail. W1 measures net time, not scheduler activity alone.
- **R4 — numerical drift.** Reordering ready work must not change a reduction's
  defined order. Token equality follows operation-level output checks; it does
  not replace them.
- **R5 — duplicate execution seam.** A megakernel cannot become a parallel model
  implementation. Later work must enter through the existing production and
  composition seams.
- **D1 — no dependency adoption.** ThunderKittens remains external until W1
  identifies a missing primitive and a separate decision measures the cost.
- **D2 — no ceiling.** A failed first proof records the next traceable
  hypothesis; it never declares the architecture exhausted.
- **D3 — retire the count, retain the guarantees.** #3085 removes only the
  cross-file cardinality measurement. It does not turn off parsing, key
  uniqueness, row shape, lifecycle, issue, claim, spec, anchor, or semantic
  checks. Historical count commentary moves to `.agents/completed/` because it
  remains provenance even after it stops controlling the gate.

## Evidence

At intake, evidence consists of the pinned upstream source assessment and the
existing local seam inspection. No GPU lease, build, correctness run, or
benchmark was performed, so every runtime claim remains `PENDING`. Future
measurements belong in this spec or a per-benchmark evidence file with their
complete recipes and hashes.

## Stop conditions

- Stop before W1 if W0 does not show material recoverable device-side idle
  capacity after CUDA Graph replay.
- Leave H100-specific execution `PENDING` until an authorized H100 runs inside
  the required lease.
- Stop and file a separate decision if the proof requires a new dependency,
  changes a shared-seam contract, or selects a production model.
- Refuse any performance result whose correctness, model artifact, workload,
  production configuration, device identity, or contention state differs from
  its denominator.

## Owed

Issue [#3084](https://github.com/mudler/vllm.cpp/issues/3084) owns W0 and the
conditional W1. Issue [#3085](https://github.com/mudler/vllm.cpp/issues/3085)
closes when the derived-at-read-time record gate and its tests land in this
integration. Issue [#3099](https://github.com/mudler/vllm.cpp/issues/3099)
closes when claim-state consistency and both negative fixtures land. The row
remains `SPIKE` until W0 and W1 either justify a separate production row or
falsify the hypothesis.
