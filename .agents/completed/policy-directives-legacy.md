# Standing directives — the full text

AGENTS.md is the index and carries the T0 reminders; **this file is the binding
statement of every directive summarised there**. It was split out on 2026-08-04
because AGENTS.md had reached 697 lines of co-equal MUST directives, which is
more than a session reliably holds: with everything at the same weight, rules
get dropped arbitrarily rather than by priority.

Nothing was reworded in the split. Every directive below is verbatim as it stood
in AGENTS.md, and changing one here changes the binding rule.

## Public document obligations

**`README.md` is the HUMAN-READABLE, USER-FACING document, in the LocalAI house
style. It is NOT a status-tracking log.** vllm.cpp is a LocalAI-team C++ port
(same family as parakeet.cpp / depth-anything.cpp), and its README follows the
`presenting-localai-cpp-projects` house format: the
"Brought to you by the LocalAI team" header plus badges, honest measured numbers
only, NO em-dashes (use commas, periods, parentheses, or hyphens), and the
LocalAI CTA / author / citation / license footer. The README MUST document, for
a newcomer who has never seen the project: **what it is**, its **Features**
(honest current state per capability), **Supported models**, **Performance**
(real measured numbers vs the vLLM oracle, or an explicit not-measured), the
**Build** (the real cmake invocations and options from `CMakeLists.txt`), the
**CLI / usage** (the real example-CLI and OpenAI-server arguments, grounded in
`examples/` and `src/vllm/entrypoints/`), and **how to consume it** (as a
library / C ABI, as the CLI, and as the OpenAI server, with a runnable example
each). Every argument, flag, and endpoint documented MUST be grounded in actual
source, never invented.

**Keep [`docs/STATUS.md`](../../README.md#project-status) CURRENT at EVERY feature/iteration
checkpoint. THAT is the per-capability status surface, NOT `README.md`.** In the
SAME change that shifts a feature's lifecycle state, update its ONE binding
current-state line in the matching `docs/STATUS.md` section or table (typically
the **Capability status** table, which is where a new feature's status line goes)
with the exact current stage: correctness-complete / speed-pending / build-only /
hardware-blocked / `ACTIVE` / `GATING`, plus active gaps and the next gate. Do
not wait for a feature to land or a gate to pass, and do not turn progress into a
support claim. `docs/STATUS.md` must never lag reality. It keeps ONE binding
result and the current lifecycle state per feature; **the detailed status,
forensic chronology, and evidence live in the append-only `.agents/state.md`,
`.agents/parity-ledger.md`, the area matrices, and `docs/BENCHMARKS.md`, NOT in
STATUS prose.** A STATUS table cell must never grow into a wall of per-attempt
prose; at each checkpoint collapse superseded narratives and stale intermediate
numbers into a concise disposition and relocate the detail to the record (move
it, never delete evidence). Do not preserve history merely because an older
paragraph already exists.

**`README.md` changes ONLY when a USER-VISIBLE HEADLINE shifts**, never on a
routine lifecycle transition. A headline shift is: a new supported architecture
family (one compact row), a new backend or quantization format, a change to the
headline performance numbers, or a change to the honest pre-release caveat. A
capability moving `ACTIVE` -> `GATING` -> `DONE`, a new gate count, a new
per-attempt number, or any forensic detail goes to `docs/STATUS.md` ALONE and
MUST NOT be added to the README. If you are tempted to add a sentence of nuance
to the README, that sentence belongs in `docs/STATUS.md` with a link.
`scripts/check-readme-structure.py` (CI-gated, with its mutation test
`tests/scripts/test_check_readme_structure.py`) enforces that the required
user-facing sections exist, that `docs/STATUS.md` exists and carries the
capability ledger, and that the README stays inside its length, paragraph, and
table-cell budgets; do not weaken the checker to bypass the obligation.

**Keep [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) CURRENT at the SAME
checkpoint.** Every feature/iteration records its benchmark disposition there
in the same change: accepted numbers with exact workload/reference/evidence,
or an explicit `PENDING`, `NOT APPLICABLE`, `FAILED`, or `VOID` reason and the
next reproduction command. Never publish a partial/contended/stale-denominator
number as binding. `scripts/check-doc-checkpoint.py` and its CI job enforce that
every code/test/benchmark/spike/lifecycle commit updates both public checkpoint
surfaces; do not weaken the checker to bypass the obligation.

**`docs/BENCHMARKS.md` is a KEYED TABLE, not an append log.** It is in the same
class as `.agents/roadmap_v1.md` and the area matrices: **merge by key, never
union-append.** A checkpoint **UPDATES ITS ROW IN PLACE**; a genuinely new
subject adds ONE ROW. Adding an H2 section per attempt is a protocol violation,
not a style preference: that is exactly how the page reached 11,405 lines and
171 claim-titled sections and stopped being readable by users (converted
2026-08-04). Per-attempt narrative, refuted hypotheses, profiler tables and
superseded numbers go to the append-only
[`.agents/benchmark-record.md`](../benchmark-record.md) **in the same
change** (move it, never delete evidence). If sections have accumulated anyway,
`scripts/roll-benchmark-record.py --apply` moves every non-canonical section
into the record verbatim. `scripts/check-public-doc-tables.py` (CI-gated, with
its mutation test `tests/scripts/test_check_public_doc_tables.py`) enforces the
required user-facing sections, the section/prose/character/table-cell budgets,
the house no-em-dash rule, and that the page still points at the record; do not
weaken the checker to bypass the obligation. When a DENOMINATOR turns out to be
wrong, correct every ratio built on it, never keep the flattering one.

**Keep [`docs/FEATURES.md`](../../docs/FEATURES.md) CURRENT — same-change
obligation.** It is the PUBLIC feature surface: what we support, next to vLLM,
SGLang and llama.cpp. It is a KEYED TABLE under the same rules as
`docs/BENCHMARKS.md` above (one row per feature, updated in place, no appended
sections). Any change that touches `.agents/feature-matrix.md`,
`.agents/model-matrix.md`, `.agents/backend-matrix.md`,
`.agents/quantization-matrix.md`, or `src/vllm/model_executor/models/` updates
the row it moves in the SAME change, and `scripts/check-doc-checkpoint.py`
enforces it. The marks are load-bearing and must not be inflated: ✅ means
implemented AND gated (token-exact against the pinned oracle for model rows, a
named test in the tree for engine rows), ◐ means the path exists with limits
NAMED in `docs/STATUS.md`, ☐ means not yet. Never mark a row ✅ because the code
compiles, never mark a competitor ☐ to flatter a column, and keep the
"Not supported yet" table honest, it is the reason the rest of the page is
believable.


## Record obligations

**Keep the ROADMAP (`.agents/roadmap_v1.md`) and its AREA MATRICES CURRENT —
same-change obligation.** The roadmap is the single top-level portfolio table;
the linked engine/model/quantization/kernel/backend matrices are the detailed
execution-status surfaces; `feature-matrix.md` remains the broad parity coverage
view. Any change that shifts a feature's or track's state updates
the owning matrix row (status, spike/spec, implementation + test evidence) and
the roadmap portfolio row **in the SAME change**, exactly like the README rule
above. Neither is ever updated speculatively: `DONE` means merged and gated,
with non-empty code and test anchors. Applies to every sub-agent; reviewers
treat a state-shifting diff without its matrix/roadmap update as incomplete.

**Adding a CUDA architecture requires its Triton-AOT cubins — same-change
obligation (user-directed 2026-07-28).** The vendored Triton-AOT GDN kernels
(`src/vt/cuda/triton_aot_vendored/sm_XX/`) are PER-ARCH: a cubin loads only on
the SM it was compiled for, so a build whose target arch has no vendored tree
silently falls back to the spilling hand-CUDA kernel (the codegen regression the
`ROAD-V1-D1-GDN-AOT` gap corrected — Triton REG:205/0-spill vs hand REG:255 +
stack spills). Therefore any change that introduces a NEW CUDA-architecture
target — a new `VLLM_CPP_CUDA_ARCHITECTURES` value or a new
`.agents/backend-matrix.md` CUDA-arch row — MUST, in the SAME change, do ONE of:
(a) regenerate and vendor that arch's FULL AOT set via the sanctioned
configure-time pipeline (`-DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=ON
-DVLLM_CPP_TRITON_VENDORED_ARCH=sm_XX`; cross-compiled, no target board needed;
57 artifacts + MANIFEST matching the `sm_121a` fileset — see
[.agents/specs/triton-aot-per-arch.md](../specs/triton-aot-per-arch.md) and
`cmake/TritonAOT.cmake`), with `scripts/check-triton-aot-drift.sh` rc=0; or
(b) if regen is not yet possible on that arch, record the GDN-decode gap
HONESTLY in `backend-matrix.md`/`kernel-matrix.md` (that arch runs the hand
kernel, build-verified-only, NOT AOT-parity) rather than leaving a silent
fallback. The `sm_121a` cubins are SACRED (the 27B/35B GDN gate) and MUST stay
byte-identical when adding any other arch — prove it (git-untouched) in the same
change.

**Doc lifecycle — live context vs completed record (user-directed 2026-07-10).**
`.agents/` holds documents that are LIVE context for current work; era-closed
documents move to **`.agents/completed/`** (version/era-stamped name, e.g.
`completed/roadmap_mvp_v0.md`) in the same change that closes their era, with
all repo links fixed. The roadmap is VERSIONED: `roadmap_v1.md` is current;
when superseded, it moves to `completed/` and `roadmap_v2.md` takes its place.
Nothing under `completed/` may be load-bearing for live decisions — if you need
to cite it for current work, the relevant content belongs (summarized) in a
live doc. Rationale: a reader of `.agents/` should see exactly what bears on
what we are doing NOW, nothing stale mixed in.

**Periodic live-document compaction (user-directed 2026-07-13, extended
2026-07-14).** At every
feature/iteration checkpoint, and whenever a live narrative accumulates more
than one superseded attempt block, rewrite it to the current binding result,
the constraints learned, the active lifecycle state, and the next gate. Do not
copy run-by-run chronology into README, BENCHMARKS, roadmap, matrices, or live
spec introductions. Closed narrative that still has reference value moves to
`.agents/completed/`; otherwise Git plus the append-only `state.md` and
`parity-ledger.md` are the historical record. Those two files are append-only
**within the current record era**. At a roadmap-version boundary or the close
of a long benchmark campaign, freeze them under `.agents/completed/` with
era-stamped names, create new live files containing a concise carry-forward
summary plus the still-active entries, and repair every live evidence link in
the same change. Frozen records are immutable and never deleted. Do not defer
this rollover once the raw history has become burdensome cold-session context;
the point is to preserve auditability without making old attempts load-bearing.
Compaction preserves every current requirement and evidence anchor; it removes
duplicated context, not evidence.

**Spec/scoping location.** All feature-specific implementation specs, scoping
reports, semantics notes, feasibility studies, and design references live under
`.agents/specs/`, never at the `.agents/` top level. The top level is reserved
for the live project-wide protocol, roadmap, status, environment, inventory,
and ledger. Specs that cease to be live context follow the same lifecycle and
move to `.agents/completed/` with their links repaired.

## STANDING DIRECTIVE — tabular inventory, spike first, then parallel claims

The canonical record is **table-first**. Before implementation begins, enumerate
the complete upstream surface in the owning area matrix. Every row has a stable
ID and these fields: upstream source, our implementation anchor, tests/evidence,
spike/spec, lifecycle state, and owner/claim. The canonical area matrices are:

- `.agents/engine-matrix.md` — stable execution rows for engine, KV, scale-out,
  sampling, serving, and other cross-cutting behavior;
- `.agents/feature-matrix.md` — broad one-by-one parity coverage view; it rolls
  up to the engine matrix and the domain matrices below;
- `.agents/model-matrix.md` — every model architecture/family registered by the
  pinned vLLM;
- `.agents/quantization-matrix.md` — every tracked storage/quantization scheme,
  separated by loader, dequant/compute path, backend, and end-to-end gate;
- `.agents/kernel-matrix.md` — vLLM and dependency kernel families plus our
  dispatch/architecture coverage;
- `.agents/backend-matrix.md` — platform and CUDA-architecture targets plus the
  native competitor/performance gate for each backend.

**Every item is spiked before it is implemented.** Its committed
`.agents/specs/<slug>.md` spike must inventory the whole upstream/dependency
chain, dispatch rules, exact files to port, tests to port, hardware needs,
correctness/performance gates, dependencies, and a row-sized work breakdown.
No row may enter `READY`/`ACTIVE` without that spike. An implementation already
present in the tree is not allowed to claim `DONE` in a matrix until the row
links exact code and test/evidence anchors, its ledger evidence, and the closing
commit; record gaps honestly as `ANCHOR-BACKFILL` or `PARTIAL`.

Parallel work is coordinated through `.agents/coordination.md`. When the
developer preferences enable parallel agents, each sub-agent claims stable row
IDs there before editing, uses an isolated worktree/branch, and owns only the
listed files/rows. A single-agent task does not need a worktree or claim unless
it is explicitly participating in coordinated roadmap work. GPU execution uses
the `${GPU_LOCK}` policy from the developer preferences whenever concurrent
work may contend. The universal rule is an uncontended benchmark: an A/B or
profile series holds one exclusion mechanism across the whole series. When every row in
an execution block is `DONE`, move the block plan/report to
`.agents/completed/` in the closing change. Permanent support matrices retain
the completed row and its code/test anchors so current capability remains
discoverable. Full lifecycle and claim protocol: `.agents/coordination.md`.
CI runs `scripts/check-agent-record.py` plus its mutation suite. It rejects
missing semantic columns, ungrounded implemented states, `READY+` specs that do
not name the row and cover the full spike contract, `SPIKE`/`ACTIVE` owners not
present in the coordination claim table, and `DONE` rows without ledger/commit
closure. Do not weaken the checker to make a transition pass; repair the record.

`.agents/model-matrix.md` additionally opens with an **architecture-support
checklist**: a rollup of how many architecture rows sit at each lifecycle state
plus one support-marked line per *engaged* architecture (any row past
`INVENTORIED`). This checklist MUST stay in sync with the detailed row lifecycle
states, enforced by `scripts/check-model-checklist.py` (CI-gated, with its
mutation test `tests/scripts/test_check_model_checklist.py`): a `✅`/`🚧`/`📋`/`🚫`
mark can never claim more than the row's state backs, no engaged architecture may
be omitted, and the rollup counts may not drift from the real per-state counts. A
change that advances (or regresses) a model's state updates its checklist entry
AND the rollup in the SAME commit; do not weaken the checker to bypass this.

## STANDING DIRECTIVE — MIRROR vLLM across ALL features (don't ask, mirror)

Feature parity with vLLM across all features is the policy. When vLLM already
does something a certain way, **mirror it** — do not stop to ask which way to go.
If vLLM supports MULTIPLE modes (e.g. nvfp4 W4A4 has both true-fp4-activations
AND a `use_a16` bf16-activation mode, over a capability-gated kernel family:
cutlass / flashinfer / marlin / emulation), **support them all** and mirror
vLLM's selection logic, including what it selects on GB10/sm_121. Only escalate
genuine PRODUCT/scope calls vLLM can't answer (e.g. "is model X in the MVP?"),
never "how should feature X behave?" (→ mirror vLLM).

**GROUND EVERY CHECK IN THE WHOLE EXECUTION CHAIN, not just the vLLM repo.** vLLM
is an ORCHESTRATION layer — the kernels that actually run (and that make it fast)
live in its DEPENDENCIES: **flashinfer** (CuTe-DSL / cutlass fp4·fp8 GEMMs, fused
norm+quant, MoE, sm_121 "blackwell_sm12x" kernels), **cutlass**, **cuBLASLt**
(nvjet), **DeepGEMM**, and **torch/Inductor** (the fused Triton it codegens). Read
the actual pinned vLLM code (`${VLLM_SOURCE}` @ pin `555967922`, vLLM
0.26.0.dev0 — advanced 2026-07-26 from the prior `e24d1b24`/0.25.0 pin; see
[.agents/specs/pin-advance.md](../specs/pin-advance.md)) AND, as
needed, the installed dependency source (`${DEPENDENCY_SOURCE}`, for example
`flashinfer/cute_dsl/*.py` and `flashinfer/gemm/`), cite `file:line` on every
side, and mirror what you find. **NEVER declare a lever "build-specific",
"unverifiable", or "out of reach" without first inspecting the dep chain and, for
compiled/JIT/Inductor code, DUMPING the generated kernel** — `TORCH_LOGS=output_code`
/ `TORCH_COMPILE_DEBUG=1` for Inductor Triton; nsys kernel names +
`cublasLtMatmulAlgoGetHeuristic` for cuBLASLt selection; the CuTe-DSL / cutlass
source for flashinfer. A fusion "absent from vLLM's csrc" may live in flashinfer
(e.g. `add_rmsnorm_fp4quant` — the fused Add+RMSNorm+FP4-quant we initially and
WRONGLY declared nonexistent). Verify the whole chain as necessary. This applies to
every subagent and every design/parity check.

**We can VENDOR generated kernels — account for this in every architecture
decision.** A vLLM edge that lives in a codegen'd or fused kernel (Inductor Triton,
a DeepGEMM/CuTe-DSL JIT kernel, a flashinfer fusion) is NEVER out of reach on the
ground that "it's compiled/fused and we're C++": our engine reproduces any such
kernel two ways — (1) a hand-written fused kernel declared through `vt::FusedChain`
(`include/vt/fused_recipe.h`, `include/vt/recipes.h`), or (2) AOT-compiling the
kernel and shipping the cubin through
the Triton-AOT vendoring pipeline (`src/vt/cuda/triton_aot_vendored/sm_XX/`,
`cmake/TritonAOT.cmake` — the GDN decode path already does this per arch), and we can
equally vendor generated C++/CUDA. So the gap to an Inductor/DeepGEMM fusion is
COVERAGE/EFFORT (Inductor auto-generates graph-wide at JIT time and shape-specializes
per call; we port the hot-path fusions deliberately, and AOT covers the known shapes
of a fixed model/decode step), NEVER a capability ceiling. The only genuinely-hard
residual is a specific hand-tuned kernel outrunning our best CUTLASS/cuBLASLt/AOT port
on one shape — still "port a better kernel", not "impossible". Design every forward
and kernel plan on the assumption that the fused/graphed/codegen'd form is reachable.

**FOLD onto the shared fusion / merged-GEMM frameworks — do NOT hand-roll (CI-gated).**
The engine has TWO shared frameworks a new model MUST route through, not re-implement:
(1) the **glue** catalog `vt::FusedChain` (add+residual+RMSNorm, norm+rope — declare a
`constexpr FusedRecipe` once, every backend inherits it); (2) the **merged-GEMM** family
— the gate-up MLP method seam `layers::MlpGateUpMethodBase`
(`UnquantizedMlpGateUpMethod` SwiGLU / `UnquantizedMlpGateUpGeluMethod` GeGLU, with the
nvfp4 `GateUpFusedMarlinD` arm inherited free on quantized checkpoints), the declarative
`vt::MergedGemmGroup` descriptor (N GEMMs sharing operand A + a fused epilogue), and the
shared fused ops (`vt::MoeGateUpSwiGLUGrouped`, `vt::MatmulBTQuantGrouped`). A model
forward is meant to be **"born fused"**: it picks the quant arm and binds the recipes; the
tuned kernel is a shared `vt::` op, NOT per-model code. The model FORWARD is legitimately
arch-scoped (the op sequence — MLA, MHC, hash-routed MoE — is the architecture, hand-write
it like vLLM's model files); the KERNELS it calls are op-scoped and SHARED. Hand-rolling a
`{gate; up; SiluAndMul/GeluAndMul}` MLP, an add+RMSNorm chain, or a per-model merged-GEMM
instead of routing through these seams is **drift**: a new backend can't inherit it by
registration, a new vLLM fusion PR can't port as one declaration, and the tuned kernel
stays siloed (the exact regression the frameworks exist to prevent — see
`.agents/specs/arch-fusion-fold-plan-2026-07-30.md`,
`.agents/specs/keepquant-shared-ops-2026-07-30.md`,
`.agents/specs/portable-fusion-framework.md`). This is enforced by
`scripts/check-fusion-consistency.py` (two floors — glue + merged-GEMM — each with an
allowlist: `scripts/fusion-consistency-allowlist.txt`,
`scripts/merged-gemm-consistency-allowlist.txt`). A model that legitimately cannot adopt
yet is a CONSCIOUS, reviewable allowlist entry with a reason + its fold-plan tier — never a
silent landing; removing the entry after the fold (byte-exact + token-exact/near-tie gated)
is the enforcement gate closing.

**ROUTE decode through the shared RUNTIME/decode framework — do NOT hand-roll a
private decode driver (the THIRD "MUST route through" seam, alongside fusion +
merged-GEMM).** A new model's decode MUST enter the production `ModelRunner`
(`ModelRegistry::Forward`) and reuse the shared decode stack: the attention preamble
`dense_attn::AttnBlock` (paged **bf16** KV sized from the spec dtype
`v1::ResolveKvCacheDType()`, `vt::ReshapeAndCache`, the shared FA2 split-KV /
`vt::PagedAttention` kernel — NOT a private `std::vector<float>` KV + a bespoke attention
kernel); **bf16-resident activations** (`DBuf`s, not an f32 host residual stream with a
`CastBf16` before every projection); **on-GPU sampling** (`ForwardLogits.on_device()` with
a device logits carrier — NO host `logits.Download`/argmax on the default `gather_logits`
path); the shared **MoE resident builders** (`BuildMoeMarlinResident` fused-w13, not a
hand-rolled split-w13); a **device-resident RoPE cache** (not a host cos/sin rebuild per
step); and a `*DenseDecodeGraph`-style capture (in-graph KV write via `slot_mapping`, not
a host between-replay `Copy` loop). The forward's ARCH is arch-scoped (hand-write the op
sequence like vLLM's model files); the decode DRIVER is SHARED — a model **expresses its
layout** (attention config, KV spec dtype, MoE structure, quant arm) and **inherits the
parity-enablers for free**: "born on the runner", the sibling of "born fused". A model
that hand-rolls a private resident/graph decode driver off the runner is **drift**: it
inherits none of the enablers and forces per-model rediscovery — the **Laguna
anti-pattern** (its resident/graph decode went fully off-framework: private f32 host-vector
KV, bespoke `DecodeAttnGqaKernel`, per-GEMM `CastBf16`, host RoPE rebuild, host
logit download — costing a multi-cycle lever hunt to recover what `AttnBlock`/the runner
provide by default; **Qwen3VL** `VLGenerateCore` + host `ArgMax` and the GGUF
**DeepSeek-V4** resident decode are the same escape). Enforced by the **born-on-the-runner
CI guard** `scripts/check-runner-routing-consistency.py` (the sibling of
check-fusion-consistency.py, wired into the same `agent-record` CI job): a
`check-*-consistency.py`-shape invariant over every `REGISTER_VLLM_MODEL` with THREE
sub-checks: (a) the default `gather_logits=true` forward returns
`ForwardLogits.on_device()==true` — no `HostLogits`/`logits.Download` off the on-GPU sampler
on the production path (it follows the `.forward` hook through its
`SomeModel::ForwardDevice` delegate + `using`-aliases, so a HOST-stub ForwardDevice is caught
and a REFUSE-by-name stub is skipped); (b) the model does not ALSO ship a private
`*GenerateCore` host generate/argmax loop as its real decode path; **(c) bf16-resident
activations** — the decode keeps its residual/activation stream in bf16 **device** buffers
(the shared `dense_attn::AttnBlock` preamble + `DBuf` glue), NOT a hand-rolled private
`std::vector<float>` host residual stream cast to bf16 before every projection
(`CastBf16`/`GemmBf16`), so a model that declares >=`MIN_F32_RESID` private f32
residual-stream buffers AND routes through neither the shared bf16 attn preamble nor a
bf16-`DBuf` residual is flagged **F32_STREAM** (drift), while a `DBuf`-based decode is
BF16_RESIDENT (clean). Axis (c) is ORTHOGONAL to (a): `deepseek_v4` returns device-resident
logits (clean on (a)) yet computes its whole decode in an f32 host vector stream (drift on
(c)); `qwen3_vl` is HOST on (a)/(b) yet its per-token transformer decode residual is a bf16
`DBuf` (clean on (c) — its text decode is already routed onto DBufs). A model that
legitimately cannot route yet is a CONSCIOUS allowlist entry with a reason + fold-plan tier —
never a silent landing. The (a)/(b) logits/generate axis uses
`scripts/runner-routing-allowlist.txt` (`laguna`, `qwen3_vl`); the (c) bf16-activation axis
uses a SIBLING `scripts/runner-bf16-activation-allowlist.txt` (`laguna`, `deepseek_v4`) —
kept separate because the two axes have DIFFERENT membership, so a shared list would silently
cross-suppress. Removing an (a)/(b) entry after the model returns device-resident logits, or
a (c) entry after its decode residual routes onto the shared bf16 `AttnBlock`/`DBuf` glue, is
the gate closing. Decode-CUDA-graph coverage is a **tracker/report**, NOT a red-build gate —
small bf16 dense models are legitimately eager (they hit vLLM parity without a graph). The
audit that established this (only Laguna/DeepSeek-V4/Qwen3VL off-framework of 24 models) is a
named follow-up spec.

**TRACE THE EXECUTION, not just the code — nsys BOTH vLLM and ours before any perf
comparison.** Reading source finds the DISPATCH LOGIC + the AVAILABLE kernels; it does
NOT tell you what ACTUALLY RAN, because vLLM's real kernels are resolved at RUNTIME and
are invisible to the source: cuBLASLt/cutlass HEURISTICS pick the kernel per call
(nvjet_sm121 vs cutlass_80), flashinfer AUTOTUNE picks the tactic, torch.compile
CODEGENS kernels, backend selection is a capability probe, and runtime-linked deps the
source never names run (we found vLLM executes **TensorRT-LLM** `delayStreamKernel` +
`flash::flash_fwd_kernel` + runs the GDN as cutlass GEMMs — NONE of which our three
code-only scans surfaced; they found buildable levers that measured NEUTRAL, while one
`nsys` of vLLM revealed the actual structural differences). So for ANY throughput/parity
work: `nsys profile` the vLLM oracle AND our engine on the SAME workload, diff the actual
GPU-kernel lists (`nsys stats --report cuda_gpu_kern_sum`), and let THAT — what vLLM's
stack really resolved to on this GPU — drive which kernels you then read in source and
mirror. Code-comparison alone is necessary but NOT sufficient; the execution trace is
the ground truth for "what vLLM is faster at". (Caveat: a graphed-vLLM nsys is
warmup/JIT/capture-contaminated — the kernel NAMES are reliable, the %s need a
steady-state/warmup-excluded capture.) **For any graphed local engine, nsys MUST use
`--cuda-graph-trace=node`; CUDA-driver ≥11.7 defaults to whole-graph mode, which omits
child kernels. Verify the export has graph-node kernel rows whenever graph launches
occur; a whole-graph-only report is attribution-incomplete and cannot select a lever.**
This applies to every subagent and every parity
check. Full method:
[.agents/parity-lever-protocol.md](../parity-lever-protocol.md) § Verify the
whole chain.

## STANDING DIRECTIVE — port the TESTS with the code (upstream tests = the spec)

vLLM's `tests/` tree is the executable spec of every feature. **Every port
carries its matching upstream test module(s) in the same change**, re-expressed
in our suite (doctest/parity/e2e tiers), named traceably after the upstream
cases, with the upstream test file cited in the header. Specs
(`.agents/specs/<slug>.md`) must include a "Tests to port" inventory; a ported
test that can't pass yet is checked in SKIPPED with a tracked reason, never
dropped. This ground-rules our work against what vLLM actually guarantees and
turns the suite into the regression net. Full protocol:
[.agents/test-porting.md](test-porting-legacy.md).

## STANDING DIRECTIVE — always compare vs vLLM (the oracle), same workload

Every change that could affect correctness OR performance MUST be compared
apples-to-apples against vLLM and both numbers + the ratio recorded in the
ledger: **correctness** vs the pinned pip-vLLM oracle (`${VLLM_ORACLE}`),
**performance** vs `vllm bench throughput` on the *identical* workload. Never
re-base the bench config without re-running vLLM on it. This is non-negotiable
and applies to subagents. Full rule: [.agents/gates.md](mvp-gates-legacy.md)
§ PROTOCOL DIRECTIVE.

**Acceptance rule — match or beat vLLM on EVERY axis, never below.** Benchmark
against vLLM on ALL axes (total + output throughput, req/s, TTFT, TPOT/ITL, peak
memory), BOTH gate models, at their large-concurrency operating point; ours must
be ≥ vLLM (throughput) / ≤ vLLM (latency, memory) on every one, with correctness
(16/16 token-for-token) as a precondition you may never trade off. Below vLLM on
any axis = an open gap, not a done change; "near parity" is NOT met.
**Reproduction is a gate:** record the exact repro recipe (commit, full command,
seed, build, vLLM oracle cmd), re-run ≥2–3× to confirm within run-noise, use a
same-binary A/B, and run only on an idle box (contended runs are void). A number
that doesn't reproduce does not count. Full protocol:
[.agents/benchmark-protocol.md](benchmark-protocol-legacy.md).

**Additional competitor floor — equivalent SGLang binds wherever it is
faster.** vLLM remains mandatory on every workload, but an exact SGLang arm is
also a floor once model/quantization, tokens, sampling, concurrency, cache
capacity/policy and serving features are proven equivalent. Cache-neutral/off
and shared-prefix cache-on are separate gates. Never compare incidental
defaults: for Qwen hybrid cache-on, explicitly enable vLLM's
`mamba_cache_mode=align`, prove cache hits/reuse in every arm, and match or beat
the faster applicable reference on every axis. Full rules remain in the
benchmark protocol above.

## STANDING DIRECTIVE — when stuck, SCAN vLLM vs ours (never accept a "ceiling")

**Same architecture, same model, same GPU → if vLLM hits a number, we CAN too.**
An apparent perf/parity "ceiling" is NEVER real — it means there are SPECIFIC
implementation/config differences we haven't found yet. Do NOT conclude "diffuse
per-op efficiency / hardware ceiling / hand-kernel exhausted" and stop. (Proven
~4× in one session: every declared ceiling dissolved into a concrete lever once
compared against vLLM — dense bf16→fp8 +6%, prefill-attn vectorized staging
+8.1%, GDN tensor-core WY-solve +4.5%, mnbt config +2.7%; the 35B went
0.79×→0.985× *after* the floor was supposedly "hit".)

**The loop: SCAN → RE-ADAPT → FIND LEVERS — especially when a lever search
stalls.** When developer preferences and the current task authorize parallel
agents, run a **dynamic Workflow** that fans out **many** sub-agents to
deep-compare vLLM's throughput HOT PATH against ours **from a VARIETY of angles**
— by subsystem (forward dtype/casts, GDN/MoE/attention kernels, norm/rope fusion,
cudagraph/compile, quant dispatch, scheduler/batching, KV-cache, per-step host
overhead) AND by lens on the same area (kernel wall-time, memory traffic, launch
count, dtype/precision, tile/config, algorithm/fusion, occupancy), plus
adversarial "refute we-match-vLLM" + completeness ("what haven't we looked at?")
critics. Overlapping coverage is good — it finds blind spots. Without parallel
authorization, run the same lenses serially; the evidence standard is
unchanged. Each participating agent reads BOTH the pinned vLLM
(`${VLLM_SOURCE}` @ the parity pin) AND our source, cites
`file:line` on both sides, and reports what vLLM does DIFFERENTLY that makes it
faster. Then verify each diff adversarially (real? on the gate hot path?), rank
by gain÷effort, drive the top lever, re-measure vs vLLM, repeat. Full protocol:
[.agents/parity-lever-protocol.md](../parity-lever-protocol.md). Caveat: a
real per-op comparison needs a CLEAN slice of OURS (not inferred proportions).
A vLLM edge that is an Inductor/DeepGEMM/flashinfer FUSION is NOT a ceiling: only
*eager* op-by-op dispatch can't fuse, and our engine is not limited to eager — we
reproduce any such kernel via `vt::FusedChain` (hand-fused CUDA) or by
AOT-compiling and VENDORING the generated cubin / C++ (the Triton-AOT pipeline,
`src/vt/cuda/triton_aot_vendored/`, already ships the GDN decode kernels per arch).
The difference is coverage/effort (Inductor auto-generates graph-wide at JIT time;
we port hot-path fusions deliberately, and AOT covers a fixed model's known shapes),
never capability — see the vendoring principle in the MIRROR directive above.

**Same kernel, different throughput ⇒ look STRUCTURAL and scan the reference's rationale —
but the scan GENERATES hypotheses; per-shape MEASUREMENT is the arbiter.** When the
execution trace shows BOTH engines running the IDENTICAL kernel yet ours appears slower at
the SAME clock, the gap is LIKELY structural — how our engine DRIVES execution AROUND that
kernel (stream count, CUDA-graph structure, overlap/scheduling policy, allocator-pool
placement, per-step sync-vs-async) — not the kernel, and a "DRAM/hardware wall" while the
reference does better on the same op is NOT a proven ceiling (a real ceiling caps BOTH
engines equally). As a DEFAULT lane of every perf/parity investigation — box-independent, so
run it ALONGSIDE the nsys trace, not after — SCAN THE REFERENCE'S OWN performance RATIONALE,
not just its kernels: vLLM for most models, but EQUALLY any non-vLLM reference a model is
gated against (ds4/DwarfStar, SGLang, llama.cpp) — its source CODE COMMENTS + kernel
structure, ENV-VAR/design docs (`vllm/envs.py`, `docs/design/`), and (where they exist)
GitHub ISSUES/PRs. A reference's tree often documents the exact structural pitfall and its
magnitude. ★★ BUT — and this is the hard rule the two 2026-08-04 cases teach — the scan only
gives you a HYPOTHESIS; you MUST confirm it with PER-SHAPE / PER-KERNEL measurement before
implementing, and you MUST distrust any AGGREGATE/DERIVED throughput number (bytes÷time)
until the byte count is verified and cross-checked per-kernel:
- **POSITIVE (ds4 / DeepSeek):** reading ds4's OWN kernel source (`ds4_cuda.cu`) revealed it
  OFFLOADS a chunk of projections to f16 tensor-core GEMMs (`matmul_f16_pair` /
  `cutlass_80_wmma`) — a real DECOMPOSITION difference (ds4 does ~1.8× less GPU work),
  MEASUREMENT-CONFIRMED → a beat-path our per-launch-parity int8-GEMV scans had all missed.
- **CAUTIONARY (vLLM / Laguna):** a "20% in-engine bandwidth deficit" that triggered a whole
  structural hunt was a MEASUREMENT ARTIFACT — an aggregate bytes÷time whose byte total was
  UNDERCOUNTED; the clean PER-SHAPE table showed our projections at per-call PARITY with
  vLLM. And vLLM's own "single global auxiliary stream **to avoid an explosion of streams for
  every layer**" comment (`torch_utils.py`) generated a plausible fix (collapse our 41
  streams), which per-shape measurement then REFUTED — the projections were 0.41% concurrent
  (nothing to de-contend) and the more-serial variant measured ~2% SLOWER. The scan surfaced
  the idea; measurement killed it. Do NOT implement a scan-hypothesis against your own trace.
So: same-kernel-different-throughput is a real RED FLAG worth the structural scan and forbids
declaring a "ceiling" prematurely — but confirm the anomaly PER-SHAPE (never on an aggregate
derived metric), and verify the reference's rationale actually applies to YOUR measured
situation before you build the fix. Full method:
[.agents/parity-lever-protocol.md](../parity-lever-protocol.md) § The STRUCTURAL lens.

## Standing directive — ONE SURFACE: every capability ships through the C ABI

**A capability is DONE when `include/vllm.h` exposes it, not when an example can
do it.**

`examples/` and `examples/server` are THIN CLIENTS over the same library entry
points an external embedder receives. They may own argv parsing, process spawning
(the ratified ffmpeg boundary), file writing and printing. They may NOT own
generation logic that has no ABI path, because anything that lives only there is
unreachable to every embedder.

### Why this is T0

Video generation shipped end to end through `examples/minimax_h3_gen` and the
server's `/v1/videos` routes, gated to hundreds of assertions, while
`VLLM_ABI_VERSION` stayed at 10 with 18 text-only symbols and zero video entry
points. LocalAI's `vllm-cpp` backend `dlopen`s exactly that ABI. So a capability
this project had fully built and tested was, from an embedder's view, absent. The
tests were green the whole time: no gate asks "can a consumer reach this?"

llama.cpp is the model. Its library is a set of reusable pieces callable from
anywhere; `llama-cli` and `llama-server` are consumers of that library, not
privileged holders of behaviour.

### What this requires of a new capability

1. A C ABI entry point in `include/vllm.h`, appended so zero values preserve
   existing behaviour, with `VLLM_ABI_VERSION` bumped and `vllm_abi_version()`
   truthful.
2. The library doing the work, with `src/vllm/` still spawning NOTHING (the
   ffmpeg boundary stands: the library produces artifacts and argv, the caller
   invokes).
3. `examples/` REFACTORED onto that entry point rather than keeping a parallel
   implementation. Two code paths for one capability is the defect this directive
   exists to prevent.
4. `examples/server` routed through the same entry point, so HTTP and FFI cannot
   drift.

### The check to apply before calling a capability done

Ask: *could someone who only has `libvllm` and `vllm.h` do this?* If the honest
answer is "no, they would have to shell out to our example binary", it is not
done.
