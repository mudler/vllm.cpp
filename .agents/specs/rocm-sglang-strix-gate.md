# ROCm SGLang gate: Qwen3-4B on Strix Halo

Row: `BACKEND-GATE-ROCM-SGLANG`.

## Scope

Bind a ROCm low-concurrency serving gate against SGLang on Strix Halo
(`gfx1151`, ROCm 7.2.4). The model is Qwen3-4B BF16
(`Qwen/Qwen3-4B` @ `1cfa9a72`). Four engines run the identical workload on one
lease: vllm.cpp, pinned production vLLM, pinned SGLang, and stock llama.cpp
`b10451` (GGUF k-quant floor). Correctness precedes throughput. No predetermined
speed winner.

## Upstream anchors

| Component | Pin | Source |
|---|---|---|
| SGLang | `v0.5.15` / `f63458b5b` | [oracle](../oracles/sglang.md), `gateable = yes` on CUDA |
| vLLM | `5559679229` (current pin) | [upstream-sync](../upstream-sync.md) |
| llama.cpp | `b10451` / `10bf611e5` | [oracle](../oracles/llama-cpp.md) |
| Model | `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c` | BF16 safetensors, native for ours/vLLM/SGLang; BF16 GGUF for llama.cpp |

The SGLang oracle is gateable on CUDA (GB10 sm_121a) but has never run on
gfx1151. The ROCm arm of this gate establishes that denominator.

## Design

A four-engine qualification harness lives in `tools/bench/strix_four_engine/`.
Each engine is a thin adapter over its public entry point — no engine internals
are modified. The harness enforces identical prompts, token counts, batching,
sampling, and serial execution under one Strix lease.

The gate has three axes:

1. **Correctness** — token-exact greedy decode against vLLM (primary) and SGLang
   (cross-check). Distributional gate only when the oracle's greedy decode is
   non-deterministic.
2. **Throughput** — decode tok/s at c1, c4, c32. Same model, same workload, same
   box. SGLang is the binding floor where it beats vLLM.
3. **Memory** — peak RSS during inference.

## Hard blocker

**#3111 — first-inference GPU hang.** vllm.cpp cannot complete even the first
decode at concurrency 1 on Strix without hanging the GPU. Until first-c1 runs
clean, no ratio, no token gate, no downstream work. The diagnostic harness
(`strix-native-first-c1-control.md`) is implemented and CPU-validated but has
not reached the GPU.

## Sub-specs

| Spec | Issue | Status |
|---|---|---|
| `strix-four-engine-qwen3-4b.md` | #3053 | on 3111 branch, not on main |
| `strix-qwen3-4b-conversion-audit.md` | #3053 | on 3111 branch |
| `strix-four-engine-qualification.md` | #3053 | on 3111 branch |
| `strix-sglang-hip-identity.md` | #3072 | on 3111 branch |
| `strix-sglang-offload-inspection.md` | #3074 | on 3111 branch |
| `strix-llama-c4-input-race.md` | #3075 | on 3111 branch |
| `strix-qwen3-4b-worker-start-profiling.md` | #3076 | PR #3247 |
| `strix-qwen3-4b-distributional-calibration.md` | #3077 | PR #3248 |
| `strix-c1-c4-c32-workload.md` | #3107 | on 3111 branch |
| `strix-adapter-child-reaping.md` | #3108 | on 3111 branch |
| `strix-lifecycle-diagnostic-publication.md` | #3110 | on 3111 branch |
| `strix-native-first-c1-control.md` | #3111 | on 3111 branch |

## Tests

CPU-validated harness suites pass: 62 tests, 34 mutations detected and restored.
No hardware gate has run. The 3111 branch carries the harness code and tests but
is 380 commits behind main and needs rebasing before it can merge.

## Gates

- **Token gate:** `PENDING` — first-c1 must run clean before any token
  comparison.
- **Throughput:** `PENDING` — no ratio until the gate runs on Strix.
- **Memory:** `PENDING`.
- **Correctness calibration:** `PENDING` (#3077 — SGLang's greedy decode is
  non-deterministic, so a distributional gate may be required).

## Evidence

- CPU suites: 62 tests pass, 34 mutations detected (on 3111 branch, not main).
- vLLM head-pin on Strix: `BUILT` but first generation failed at metadata RPC
  (#3043, BACKEND-GATE-ROCM-VLLM).
- llama.cpp denominator on Strix: measured 12.233 tok/s (Q4_K_M, 27B — not the
  4B model this gate uses). See `BACKEND-GATE-ROCM-LLAMACPP` row.

## Risks

- The first-c1 hang may be a ROCm backend defect, not a gate harness defect.
  The harness isolates the native adapter; the fault is in the engine path.
- SGLang's greedy decode non-determinism means a strict token gate may not be
  achievable; a distributional gate needs ratification.
- The 3111 branch is 380 commits stale; rebasing carries conflict risk.

## Stop conditions

1. First-c1 runs clean on Strix (the hang is diagnosed and fixed).
2. Token gate passes (or a distributional gate is ratified).
3. Throughput ratio established at c1, c4, c32.
4. Memory measured.
5. Row advances from `INVENTORIED` to `ACTIVE` (first gate result) and
   eventually `DONE` (all axes satisfied or explicitly waived).

## Now

`INVENTORIED`. No hardware gate has run. The diagnostic harness is built and
CPU-validated on the 3111 branch but not merged. Local issues filed for all
9 GitHub issues (#3053, #3072, #3074, #3075, #3076, #3077, #3107, #3108,
#3110, #3111). The hard blocker is #3111 (first-c1 GPU hang).

## Git integration

One pull request for spec + implementation, per repository default.
