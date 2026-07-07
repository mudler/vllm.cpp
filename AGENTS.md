# AGENTS.md — vllm.cpp canonical index

This file is the **index** to the project's canonical record. Every session:
read this first, follow the links that matter for your task, and keep the
record updated (append to the state log) — commit it with your changes.
Push directly to `main`.

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

**GROUND EVERY CHECK IN vLLM SOURCE.** Do not decide behavior from memory or
assumption — read the actual pinned vLLM code (`/home/mudler/_git/vllm` @ the
parity pin `e24d1b24`) every time, cite `file:line`, and mirror what you find.
This applies to every subagent and every design/parity check.

## STANDING DIRECTIVE — always compare vs vLLM (the oracle), same workload

Every change that could affect correctness OR performance MUST be compared
apples-to-apples against vLLM and both numbers + the ratio recorded in the
ledger: **correctness** vs the pinned pip-vLLM oracle (`~/venvs/vllm-oracle`),
**performance** vs `vllm bench throughput` on the *identical* workload. Never
re-base the bench config without re-running vLLM on it. This is non-negotiable
and applies to subagents. Full rule: [.agents/gates.md](.agents/gates.md)
§ PROTOCOL DIRECTIVE.

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
- [.agents/roadmap.md](.agents/roadmap.md) — engineer's work breakdown:
  milestones M0–M3 as pick-up-able units with DoD; status kept current.
- [.agents/state.md](.agents/state.md) — **append-only state log**: progress,
  decisions, next steps. Update this every working session.

## Canonical documents (outside .agents/)

- [docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md](docs/superpowers/specs/2026-07-02-vllm-cpp-core-design.md)
  — core architecture design (vt:: tensor runtime, engine mirroring,
  performance plan for the parity gate, milestones M0–M3).
