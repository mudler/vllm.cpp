# AGENTS.md — vllm.cpp canonical index

This file is the **index** to the project's canonical record. Every session:
read this first, follow the links that matter for your task, and keep the
record updated (append to the state log) — commit it with your changes.
Push directly to `main`.

**Keep `README.md` (the user-facing status) CURRENT.** Whenever the *externally
visible* status changes — a gate passes, a model/quant/backend moves state, a
throughput number materially shifts, a feature lands — update the matching
README section/table row in the SAME change (its ⚠️ header, the architecture /
acceleration / quantization tables, and "Status & caveats"). The README must
never lag reality: no "pending"/"not yet confirmed" for something that now
works, and no overclaim for something that doesn't. It is part of the record,
kept honest and current alongside the ledger.

**Keep the ROADMAP (`.agents/roadmap_v1.md`) and FEATURE MATRIX
(`.agents/feature-matrix.md`) CURRENT — same-change obligation.** Any change
that shifts a feature's or track's state — a feature lands, a gap opens, a spec
gets written (`☐ → 🚧`), an agent starts/finishes implementing (`🚧 → ✅`,
merged + gated), a research track reports, a queue item is re-ranked — updates
the matching feature-matrix ROW (status + Spec link + one-line grounded note)
and the roadmap track line (mark DONE with a one-line outcome + pointer to the
full report/branch) **in the SAME change**, exactly like the README rule above.
The matrix is the single feature-level status surface; the roadmap is the
single track-level one — neither may lag reality, and neither is ever updated
speculatively (a row flips ✅ only when merged + gated). Applies to every
sub-agent; reviewers treat a state-shifting diff without its matrix/roadmap
update as incomplete.

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
steady-state/warmup-excluded capture.) This applies to every subagent and every parity
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
  (CUDA/CPU now; Metal, Vulkan, Intel XPU later) via vLLM's own Platform +
  attention-backend seams; MLX/ANE explorations; binding vt:: interface
  requirements for M0.2.
- [.agents/workflow.md](.agents/workflow.md) — **agent operating manual**:
  session protocol, Definition of Done, practicalities.
- [.agents/porting-inventory.md](.agents/porting-inventory.md) — **living
  parity record**: full vLLM feature/architecture inventory, T0 (gate) / T1 /
  T2 / T3 tiers, upstream paths, inline status markers. Kept up to date with
  every change.
- [.agents/parity-ledger.md](.agents/parity-ledger.md) — **append-only
  ledger**: one row per change we introduce — what it does vs vLLM, upstream
  PR/commit references, how parity was verified.
- [.agents/roadmap_v1.md](.agents/roadmap_v1.md) — **THE ROADMAP** (post-MVP,
  live): closing tracks, research tracks, T1/T2 queue, the feature-level
  breakdown (→ feature-matrix.md, each gap row delegable via its
  `.agents/specs/<slug>.md` spec), and the protocol evolution
  (mirror-as-floor, surpass-by-fusing).
- [.agents/completed/roadmap_mvp_v0.md](.agents/completed/roadmap_mvp_v0.md) —
  ARCHIVED M0–M3 record of the completed MVP (both throughput gates passed
  2026-07-10).
- [.agents/feature-matrix.md](.agents/feature-matrix.md) — **the one-by-one
  vLLM feature parity table** (what vLLM has vs what we have, every feature:
  parallelism, quant, serving, spec decode, multimodal, …). Living doc.
- [.agents/specs/](.agents/specs/) — live feature implementation specs,
  scoping reports, semantics notes, feasibility studies, and design references.
- [.agents/state.md](.agents/state.md) — **append-only state log**: progress,
  decisions, next steps. Update this every working session.

## Canonical documents (outside .agents/)

- [docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md](docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md)
  — core architecture design (vt:: tensor runtime, engine mirroring,
  performance plan for the parity gate, milestones M0–M3).
