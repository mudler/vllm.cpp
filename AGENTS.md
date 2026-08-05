# AGENTS.md — vllm.cpp canonical index

This file is the **index** to the project's canonical record. Every session,
read this first and follow the links that matter for the task. Commits are
allowed for completed in-scope changes and must follow the commit protocol
below.

**Developer preferences.** After this file, read
`.agents/developer-preferences.md` when it
exists. It is intentionally untracked and records the current developer's Git
integration choices, usable hosts, local paths, GPU contention policy,
download/service permissions, and collaboration preferences. Start from the
tracked
[developer-preferences example](.agents/developer-preferences.example.md).
Preferences control operations, not project truth: they cannot weaken the
correctness, testing, evidence, attribution, lifecycle, or documentation rules
in this file. Do not infer preferences from a developer name, filesystem path,
Git author, or machine identity.

If the preference file is absent or silent, use the safe defaults: local edits,
tests, and commits are allowed; do not push, merge, force-update refs, use
external hosts, install/download large assets, manage services, or start
parallel agents. Ask before those actions. In the protocol, `${VLLM_SOURCE}`,
`${VLLM_ORACLE}`, `${DEPENDENCY_SOURCE}`, and `${GPU_LOCK}` mean the values
selected by that file. Exact Ettore infrastructure paths retained in the
environment registry or historical evidence are not commands for other
developers.

**Read [`.agents/NOW.md`](.agents/NOW.md) FIRST — it is the one-Read resume
surface.** The canonical record is large by design (evidence is never deleted),
which made orientation expensive: the files a cold session was told to read are
the largest in the repo. NOW.md is the fix — a ≤100-line SNAPSHOT, rewritten in
place, of the live claims, the gate being chased, and the next actions. It is
never a log; the detail it summarises stays in the append-only record.
**Refresh it in the SAME change as any `.agents/state.md` append**, because a
state append is exactly the event that moves what is live.
`scripts/check-now-current.py` (CI-gated, with its mutation test
`tests/scripts/test_check_now_current.py`) enforces both its budget and that
freshness coupling; do not weaken the checker to bypass the obligation.

## T0 — the non-negotiables

These survive any context pressure. Each links to its full statement in
[.agents/directives.md](.agents/directives.md); the linked text is the binding
version, this list is the reminder.

- **Mirror vLLM.** Feature parity across all features; when vLLM has an answer,
  mirror it including all its modes. Never ask the user how a feature should
  behave, only genuine product/scope calls.
  ([full](.agents/directives.md#standing-directive--mirror-vllm-across-all-features-dont-ask-mirror))
- **Ground every check in the whole execution chain**, not just the vLLM repo:
  flashinfer, cutlass, cuBLASLt, DeepGEMM, torch/Inductor. Cite `file:line` on
  both sides. Never declare a lever unreachable without dumping the generated
  kernel.
- **Trace the execution, not just the code.** `nsys` BOTH vLLM and ours on the
  same workload before any perf comparison; graphed local engines need
  `--cuda-graph-trace=node`. Source finds dispatch logic, not what ran. **cuBLAS/
  kernel INVOCATION parity:** any GEMM/GEMV parity claim MUST verify vLLM's ACTUAL
  call on FOUR axes — (1) output/C dtype (it SELECTS the gemvx template: an
  API-name match can still be a slower `<bf16,FLOAT>` template than vLLM's
  `<bf16,bf16>`), (2) compute+scale type, (3) entry point + algo policy
  (`cublasGemmEx` default-algo vs `cublasLtMatmul` requestedAlgoCount/heuristic),
  (4) the resolved kernel TEMPLATE dtypes read off the SAME tool's trace. HARD
  RULE: a CROSS-TOOL comparison (our nsys vs vLLM's torch-profiler) can NEVER
  establish invocation parity — a same-tool trace where entry point AND resolved
  template match is required. Op-contract gate:
  `scripts/check-gemv-invocation-consistency.py`; full lane in
  [.agents/parity-lever-protocol.md](.agents/parity-lever-protocol.md) § The
  STRUCTURAL lens.
- **Three MUST-route seams (CI-gated).** A model routes through the fusion
  catalog (`vt::FusedChain`), the merged-GEMM family
  (`layers::MlpGateUpMethodBase`, `vt::MergedGemmGroup`), and the shared decode
  runner (`ModelRegistry::Forward`, `dense_attn::AttnBlock`, on-GPU sampling).
  Hand-rolling any of them is drift: fold, or take a conscious allowlist entry.
- **Compare against the oracle, same workload.** Correctness vs the pinned
  pip-vLLM oracle; performance vs `vllm bench throughput` on the identical
  workload. Both numbers and the ratio go in the ledger.
- **Match or beat vLLM on EVERY axis**, never below, on both gate models, with
  16/16 token-exact correctness as a precondition you may never trade. Below on
  any axis is an open gap, not a done change. Reproduction is part of the gate.
- **Never accept a "ceiling".** Same architecture, same GPU: if vLLM hits a
  number we can. An apparent ceiling means specific differences not yet found.
- **Port the tests with the code.** Upstream `tests/` is the executable spec;
  every port carries its upstream test module in the same change.
- **Spike before implementing.** No row enters `READY`/`ACTIVE` without a
  committed `.agents/specs/<slug>.md` covering the full spike contract.
- **Never weaken a checker** to make a transition pass. Repair the record.
- **Evidence is moved, never deleted.** Compaction relocates detail into the
  append-only record; it never drops it.
- **Every commit carries `FOLLOWING_AGENTS_PROTOCOL`** plus `Assisted-by:`, and
  never `Signed-off-by` or `Co-Authored-By` from an AI.
- **Run `scripts/agent-preflight.sh`** at session start and before committing,
  and chain the push to it (`gate && git push`) so a red gate cannot be followed
  by a green push.
- **Know your ROLE before you work.** Operator or helper
  ([protocol](.agents/specs/operator-helper-protocol.md)). It cannot be derived
  at session start — several sessions launch from one checkout — so DECLARE it
  (`scripts/agent-role.py claim operator|helper --row <ROW-ID>`), which
  materializes it into an exclusive lock or a worktree+PR, after which it is
  re-derived rather than remembered. A helper works in an isolated worktree on
  `row/<ROW-ID>` and opens a DRAFT PR at the START: that PR **is** the claim.
  The operator merges PRs first thing, owns `main` and the GPU, and drives
  feature work through sub-agents rather than writing it.
- **Never three-way merge a keyed record.** `docs/STATUS.md`,
  `docs/BENCHMARKS.md`, `docs/FEATURES.md`, `.agents/NOW.md`, the matrices and
  `coordination.md` are merged by taking `main`'s version wholesale, re-applying
  your edit, and verifying the other side is byte-identical. A three-way merge
  silently produced a VARIANT of another session's binding numbers on
  2026-08-04 — no conflict, no marker. Union-append only the append-only logs.

**Session handoff.** Deeper cold-resume context for unfinished work lives in the
newest [`.agents/state.md`](.agents/state.md) entries plus the live claim row
in [`.agents/coordination.md`](.agents/coordination.md): active claim, exact
source/evidence roots, prohibitions, and the first resume/verification
commands. **The state tail is only trustworthy below the
`<!-- state-order:enforced-below -->` marker**, where every entry carries a
sortable `<!-- state: YYYY-MM-DD -->` anchor on the line after its heading and
`scripts/check-state-order.py` proves the order runs oldest-to-newest. That gate
exists because union-merging appends from parallel worktrees had silently
interleaved the tail, so "newest last" was false and cold resume returned a
jumble; repair an interleaved merge with
`python3 scripts/sort-state-tail.py --apply`, never by hand.
Append to the state log for a feature/lifecycle checkpoint, a
material implementation decision, or unfinished work that needs a handoff.
Routine review, Git housekeeping, and protocol discussion do not require a
state entry. Before ending a session with work in flight, record the handoff in
the same checkpoint change. (User-directed 2026-07-14: the separate
`HANDSOFF.md` replace-in-place surface is retired; do not recreate it.)

**Public document obligations (full text:
[.agents/directives.md](.agents/directives.md#public-document-obligations)).**
`README.md` is the user-facing landing page and changes ONLY when a
user-visible headline shifts. `docs/STATUS.md` is the per-capability status
surface updated at EVERY checkpoint. `docs/BENCHMARKS.md` and `docs/FEATURES.md`
are KEYED TABLES: update the row in place, never append a section. Forensic
detail goes to the append-only `.agents/` record. Each is CI-gated
(`check-readme-structure.py`, `check-public-doc-tables.py`,
`check-doc-checkpoint.py`); do not weaken a checker to bypass the obligation.

**The obligated public surfaces, declared once.** This block is the single
statement of what `scripts/check-doc-checkpoint.py` enforces.
`scripts/check-protocol-consistency.py` (CI-gated, with its mutation test
`tests/scripts/test_check_protocol_consistency.py`) asserts it equals the
checker's constants AND appears verbatim in
[`.agents/workflow.md`](.agents/workflow.md), the session operating manual.
That gate exists because the obligation was migrated off `README.md` here and in
the checker but NOT in the manual, which went on instructing agents to do the
exact thing the migration removed — prose and gate must move together, and prose
is what agents actually read. `README.md` is deliberately absent from the block.

<!-- doc-obligation-contract:begin -->
| Public surface | Owed by |
|---|---|
| `docs/STATUS.md` | every feature/iteration checkpoint |
| `docs/BENCHMARKS.md` | every feature/iteration checkpoint |
| `docs/FEATURES.md` | any change to a feature/model/backend/quantization surface |
<!-- doc-obligation-contract:end -->

**Record obligations (full text:
[.agents/directives.md](.agents/directives.md#record-obligations)).** The
roadmap portfolio row and its owning area matrix row move in the SAME change as
the state they describe, and `DONE` means merged and gated with real anchors.
Adding a CUDA architecture requires vendoring that arch's full Triton-AOT cubin
set in the same change, or recording the GDN gap honestly. `.agents/` holds live
context only: era-closed documents move to `.agents/completed/`, specs live in
`.agents/specs/`, and live narratives are compacted to the binding result at
every checkpoint.

**Tabular inventory, spike first, then parallel claims (full text:
[.agents/directives.md](.agents/directives.md#standing-directive--tabular-inventory-spike-first-then-parallel-claims)).**
The record is table-first: every row carries a stable ID, upstream source, our
anchor, tests/evidence, spike, lifecycle state and owner, across the engine,
feature, model, quantization, kernel and backend matrices. Every item is spiked
before implementation. Parallel work claims row IDs in
[.agents/coordination.md](.agents/coordination.md) and uses isolated worktrees.
`scripts/check-agent-record.py` and its mutation suite gate all of it.

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
parity on the project GB10/sm_121 release target, loading from safetensors **and GGUF**, shipped
llama.cpp-style as a library + example CLI/OpenAI server, with tool calling,
grammars, streaming/non-streaming, and e2e test suites.

**The performance and parity directives (full text:
[.agents/directives.md](.agents/directives.md#standing-directive--mirror-vllm-across-all-features-dont-ask-mirror)):**
mirror vLLM across all features; ground every check in the whole execution chain
and vendor generated kernels rather than declaring them out of reach; fold onto
the shared fusion, merged-GEMM and decode-runner frameworks; trace the execution
with `nsys` on both sides; port the upstream tests; compare against the oracle on
every axis; and never accept a ceiling. These are summarised in T0 above and
stated in full in the linked document.

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

- `.agents/developer-preferences.md` — the
  ignored, developer-owned execution profile for this workspace (copy the
  tracked example below; absence uses the safe defaults above).
- [.agents/developer-preferences.example.md](.agents/developer-preferences.example.md)
  — tracked template for Git integration, paths, hosts, compute, and
  collaboration preferences.
- [.agents/directives.md](.agents/directives.md) — **the full text of every
  standing directive** summarised in T0 above. Binding; AGENTS.md is the index.
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
- [.agents/environment.md](.agents/environment.md) — factual environment
  profile registry, benchmark models, gate-model architecture, prior-art patch
  series, and environment TODOs; availability is selected by developer
  preferences.
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
- [.agents/sglang-matrix.md](.agents/sglang-matrix.md) — the SGLang parity
  PROGRAM's whole-surface inventory: every SGLang runtime capability classified
  FUSED / SGLANG-DISTINCT / INVENTORIED / OUT-OF-SCOPE vs our vLLM-derived
  engine, with the SGLang-as-oracle gate methodology in
  [.agents/specs/sglang-parity-oracle.md](.agents/specs/sglang-parity-oracle.md).
  SGLang is a full parity target (competitor perf floor + correctness
  cross-check), not the mirror source — vLLM remains the behavior truth.
- [.agents/specs/](.agents/specs/) — live feature implementation specs,
  scoping reports, semantics notes, feasibility studies, and design references.
- [.agents/state.md](.agents/state.md) — **append-only state log**: progress,
  decisions, next steps. Update this every working session.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — user-facing accepted benchmark
  scoreboard plus the current pending/failed/void checkpoint and repro status.
  KEYED TABLE: update the row, never append a section.
- [docs/FEATURES.md](docs/FEATURES.md) — user-facing feature matrix against
  vLLM, SGLang and llama.cpp. KEYED TABLE, same rules.
- [.agents/benchmark-record.md](.agents/benchmark-record.md) — **append-only
  benchmark record**: every attempt, refuted hypothesis, profiler table and
  superseded number. Read it before re-running a lever; most entries are dead
  ends already measured and closed.

## Canonical documents (outside .agents/)

- [docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md](docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md)
  — core architecture design (vt:: tensor runtime, engine mirroring,
  performance plan for the parity gate, milestones M0–M3).
