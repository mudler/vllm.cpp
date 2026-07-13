# AGENTS.md — vllm.cpp canonical index

This file is the **index** to the project's canonical record. Every session:
read this first, follow the links that matter for your task, and keep the
record updated (append to the state log) — commit it with your changes.
Push directly to `main`.

**Keep `README.md` (the user-facing status) CURRENT at EVERY feature/iteration
checkpoint.** In the SAME change that advances a spike, implementation, test,
gate, benchmark attempt, or lifecycle state, update the matching README
section/table row with the exact current stage — including `ACTIVE`/`GATING`,
failed or void runs, and explicit pending hardware work. Do not wait for a
feature to land or a gate to pass. Keep its ⚠️ header, architecture /
acceleration / quantization tables, and "Status & caveats" mutually consistent.
The README must never lag reality and must not turn progress into support.
It is a **current-state snapshot, not a chronological log**: keep one binding
result, the current lifecycle state, active gaps and the next gate. At each
checkpoint, collapse superseded attempt narratives and stale intermediate
numbers into a concise disposition; detailed chronology belongs in the
append-only `.agents/state.md` / `.agents/parity-ledger.md` record and feature
specs. Do not preserve history in the README merely because an older paragraph
already exists.

**Keep [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) CURRENT at the SAME
checkpoint.** Every feature/iteration records its benchmark disposition there
in the same change: accepted numbers with exact workload/reference/evidence,
or an explicit `PENDING`, `NOT APPLICABLE`, `FAILED`, or `VOID` reason and the
next reproduction command. Never publish a partial/contended/stale-denominator
number as binding. `scripts/check-doc-checkpoint.py` and its CI job enforce that
every code/test/benchmark/spike/lifecycle commit updates both public checkpoint
surfaces; do not weaken the checker to bypass the obligation.
`docs/BENCHMARKS.md` is likewise a **scoreboard snapshot**, not an attempt log:
retain binding/current component results, explicit pending/failed/void
dispositions and the current reproduction entry point, while compacting
superseded run-by-run forensics into the append-only ledger/state record.

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

Parallel work is coordinated through `.agents/coordination.md`. A sub-agent
claims stable row IDs there before editing, uses an isolated worktree/branch,
and owns only the listed files/rows. GPU execution is serialized with
`flock /tmp/gpu` per `/home/mudler/_git/skills/sharing-a-gpu-with-flock/SKILL.md`
WHEN multiple agents may run GPU work concurrently — there is no external
contention on dgx, so a sole GPU owner (verified idle via `nvidia-smi`) may run
lock-free; benchmark series always need an uncontended GPU regardless;
an A/B or profile series holds one lock for the whole series. When every row in
an execution block is `DONE`, move the block plan/report to
`.agents/completed/` in the closing change. Permanent support matrices retain
the completed row and its code/test anchors so current capability remains
discoverable. Full lifecycle and claim protocol: `.agents/coordination.md`.
CI runs `scripts/check-agent-record.py` plus its mutation suite. It rejects
missing semantic columns, ungrounded implemented states, `READY+` specs that do
not name the row and cover the full spike contract, `SPIKE`/`ACTIVE` owners not
present in the coordination claim table, and `DONE` rows without ledger/commit
closure. Do not weaken the checker to make a transition pass; repair the record.

**Every commit MUST carry the trailer `FOLLOWING_AGENTS_PROTOCOL`** in its
message. This asserts the contributor (human or AI-assisted) has read this
AGENTS.md and follows the protocol. **CI rejects any commit lacking it**
(see `.github/workflows/ci.yml` → `commit-protocol-tag`). It is a one-line
trailer, e.g.:

```
<your commit subject>

<body…>

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-4-8 [ClaudeCode]
```

**TL;DR:** 1:1 port of vLLM to pure C++ (no Python/PyTorch; ggml as example,
not dependency), structured so every future upstream vLLM PR can be ported
mechanically. MVP gate: Qwen3.6-35B-A3B + 27B (NVFP4) at vLLM throughput
parity on `dgx.casa` (GB10), loading from safetensors **and GGUF**, shipped
llama.cpp-style as a library + example CLI/OpenAI server, with tool calling,
grammars, streaming/non-streaming, and e2e test suites.

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
the actual pinned vLLM code (`/home/mudler/_git/vllm` @ pin `e24d1b24`) AND, as
needed, the installed dep source (`~/venvs/vllm-oracle/lib/python3.12/site-packages/`
— e.g. `flashinfer/cute_dsl/*.py`, `flashinfer/gemm/`), cite `file:line` on every
side, and mirror what you find. **NEVER declare a lever "build-specific",
"unverifiable", or "out of reach" without first inspecting the dep chain and, for
compiled/JIT/Inductor code, DUMPING the generated kernel** — `TORCH_LOGS=output_code`
/ `TORCH_COMPILE_DEBUG=1` for Inductor Triton; nsys kernel names +
`cublasLtMatmulAlgoGetHeuristic` for cuBLASLt selection; the CuTe-DSL / cutlass
source for flashinfer. A fusion "absent from vLLM's csrc" may live in flashinfer
(e.g. `add_rmsnorm_fp4quant` — the fused Add+RMSNorm+FP4-quant we initially and
WRONGLY declared nonexistent). Verify the whole chain as necessary. This applies to
every subagent and every design/parity check.

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
[.agents/parity-lever-protocol.md](.agents/parity-lever-protocol.md) § Verify the
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
[.agents/test-porting.md](.agents/test-porting.md).

## STANDING DIRECTIVE — always compare vs vLLM (the oracle), same workload

Every change that could affect correctness OR performance MUST be compared
apples-to-apples against vLLM and both numbers + the ratio recorded in the
ledger: **correctness** vs the pinned pip-vLLM oracle (`~/venvs/vllm-oracle`),
**performance** vs `vllm bench throughput` on the *identical* workload. Never
re-base the bench config without re-running vLLM on it. This is non-negotiable
and applies to subagents. Full rule: [.agents/gates.md](.agents/gates.md)
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
[.agents/benchmark-protocol.md](.agents/benchmark-protocol.md).

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
stalls.** Run a **dynamic Workflow** that fans out **many** sub-agents to
deep-compare vLLM's throughput HOT PATH against ours **from a VARIETY of angles**
— by subsystem (forward dtype/casts, GDN/MoE/attention kernels, norm/rope fusion,
cudagraph/compile, quant dispatch, scheduler/batching, KV-cache, per-step host
overhead) AND by lens on the same area (kernel wall-time, memory traffic, launch
count, dtype/precision, tile/config, algorithm/fusion, occupancy), plus
adversarial "refute we-match-vLLM" + completeness ("what haven't we looked at?")
critics. Overlapping coverage is good — it finds blind spots. Each agent reads
BOTH the
pinned vLLM (`/home/mudler/_git/vllm` @ the parity pin) AND our source, cites
`file:line` on both sides, and reports what vLLM does DIFFERENTLY that makes it
faster. Then verify each diff adversarially (real? on the gate hot path?), rank
by gain÷effort, drive the top lever, re-measure vs vLLM, repeat. Full protocol:
[.agents/parity-lever-protocol.md](.agents/parity-lever-protocol.md). Caveat: a
real per-op comparison needs a CLEAN slice of OURS (not inferred proportions),
and distinguish the few vLLM edges that are build-specific (Inductor/DeepGEMM
fusions) — which eager-C++ can't 1:1 replicate — from the many we CAN.

## Policy for AI-Assisted Contributions

This project follows the Linux kernel project's [guidelines for AI coding
assistants](https://docs.kernel.org/process/coding-assistants.html). Before
submitting AI-assisted code, read
[.agents/ai-coding-assistants.md](.agents/ai-coding-assistants.md). Key rules:

- **No `Signed-off-by` from AI.** Only the human submitter may sign off on the
  Developer Certificate of Origin.
- **No `Co-Authored-By: <AI>` trailers.** The human contributor owns the change.
- **Use an `Assisted-by:` trailer** to attribute AI involvement. Format:
  `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]`.
- **The human submitter is responsible** for reviewing, testing, and
  understanding every line of generated code.

## Index

- [.agents/mission.md](.agents/mission.md) — what this project is and is not.
- [.agents/gates.md](.agents/gates.md) — the 5 MVP gates (success definition).
- [.agents/parity-lever-protocol.md](.agents/parity-lever-protocol.md) — the
  **scan → re-adapt → find levers** loop: never accept a "ceiling"; when stuck,
  dynamic-workflow-scan vLLM's hot path vs ours to find the specific diffs.
- [.agents/benchmark-protocol.md](.agents/benchmark-protocol.md) — **match or
  beat vLLM on EVERY axis (never below)**; how to benchmark vs vLLM on all axes,
  both models; **reproduction is a gate** (record recipe, re-run to confirm,
  idle box, same-binary A/B).
- [.agents/discipline.md](.agents/discipline.md) — **non-negotiable** porting
  rules: mirrored structure, port-don't-reinvent, upstream-commit file
  headers, recorded deviations, parity-first testing.
- [.agents/upstream-sync.md](.agents/upstream-sync.md) — **sync protocol**:
  the PARITY PIN (the vLLM commit we are parity-comparable against) and the
  repeatable sync cycle (enumerate → classify → report → port → re-verify →
  advance pin) that keeps porting upstream PRs a routine task.
- [.agents/environment.md](.agents/environment.md) — dev box, `dgx.casa`
  (GB10/sm_121) specs, benchmark models, gate-model architecture, prior-art
  patch series, environment TODOs.
- [.agents/vllm-v1-v2.md](.agents/vllm-v1-v2.md) — V1 engine vs "Model Runner
  V2" terminology; we port MRV2.
- [.agents/backends.md](.agents/backends.md) — backend portability strategy
  (CUDA/CPU now; ROCm, Metal, Vulkan, Intel XPU and ANE later) via vLLM's own Platform +
  attention-backend seams; MLX/ANE explorations; binding vt:: interface
  requirements for M0.2.
- [.agents/workflow.md](.agents/workflow.md) — **agent operating manual**:
  session protocol, Definition of Done, practicalities.
- [.agents/coordination.md](.agents/coordination.md) — **parallel-work control
  plane**: stable IDs, spike gate, claims/worktrees, dependency and GPU-lock
  rules, handoff, and completed-block archival.
- [.agents/porting-inventory.md](.agents/porting-inventory.md) — **living
  parity record**: full vLLM feature/architecture inventory, T0 (gate) / T1 /
  T2 / T3 tiers, upstream paths, inline status markers. Kept up to date with
  every change.
- [.agents/parity-ledger.md](.agents/parity-ledger.md) — **append-only
  ledger**: one row per change we introduce — what it does vs vLLM, upstream
  PR/commit references, how parity was verified.
- [.agents/roadmap_v1.md](.agents/roadmap_v1.md) — **THE ROADMAP** (post-MVP,
  live): one ordered portfolio table over the area matrices and current gates.
- [.agents/completed/roadmap_mvp_v0.md](.agents/completed/roadmap_mvp_v0.md) —
  ARCHIVED M0–M3 record of the completed MVP (both throughput gates passed
  2026-07-10).
- [.agents/engine-matrix.md](.agents/engine-matrix.md) — canonical stable-ID
  execution rows for cross-cutting engine/KV/sampling/serving/loading work,
  with exact code, tests, spike and owner fields.
- [.agents/feature-matrix.md](.agents/feature-matrix.md) — broad one-by-one
  cross-cutting vLLM parity coverage view; execution claims use engine-matrix.
- [.agents/model-matrix.md](.agents/model-matrix.md) — comprehensive pinned-vLLM
  model architecture/family inventory and port status.
- [.agents/quantization-matrix.md](.agents/quantization-matrix.md) — canonical
  per-scheme quantization inventory, with loader/compute/backend/e2e evidence.
- [.agents/kernel-matrix.md](.agents/kernel-matrix.md) — kernel-family and
  dispatch parity inventory across vLLM and its runtime dependency chain.
- [.agents/backend-matrix.md](.agents/backend-matrix.md) — backend/platform and
  CUDA target matrix, including native-competitor performance gates.
- [.agents/specs/](.agents/specs/) — live feature implementation specs,
  scoping reports, semantics notes, feasibility studies, and design references.
- [.agents/state.md](.agents/state.md) — **append-only state log**: progress,
  decisions, next steps. Update this every working session.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — user-facing accepted benchmark
  scoreboard plus the current pending/failed/void checkpoint and repro status.

## Canonical documents (outside .agents/)

- [docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md](docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md)
  — core architecture design (vt:: tensor runtime, engine mirroring,
  performance plan for the parity gate, milestones M0–M3).
