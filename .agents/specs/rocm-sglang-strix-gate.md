# ROCm SGLang gate: Qwen3-4B on Strix Halo

Row: `BACKEND-GATE-ROCM-SGLANG`.

## Scope

Bind a ROCm low-concurrency serving gate against SGLang on Strix Halo
(`gfx1151`, ROCm 5.7). The model is Qwen3-4B BF16
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

## Hard blocker — CLEARED

**#3111 — first-inference GPU hang.** The hang does not reproduce on the
current main (SHA `1da8b4d`) with ROCm 5.7 on Strix. Five consecutive first-c1
runs and a 64-token throughput test all completed with exit 0 and no GPU fault.
The original hang was observed on 2026-09-09; the code that caused it may have
been fixed by subsequent BACKEND-ROCM commits, or the ROCm version difference
(5.7 on the box vs the 7.2.4 the spec was written for) prevents the fault.
The first-c1 result was obtained with the production `vllm-cli` binary, not the
diagnostic harness (`strix-native-first-c1-control.md`, still on the 3111
branch).

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

- **Token gate:** `PENDING` — first-c1 is clean; token comparison can proceed
  once a reference oracle runs on Strix.
- **Throughput:** `PARTIAL` — vllm.cpp c1 measured at ~10 tok/s (64 tokens,
  bf16, 3 repeats). Other engines not yet measured on Strix.
- **Memory:** `PENDING`.
- **Correctness calibration:** `PENDING` (#3077 — SGLang's greedy decode is
  non-deterministic, so a distributional gate may be required).

## Evidence

- CPU suites: 62 tests pass, 34 mutations detected (on 3111 branch, not main).
- vLLM head-pin on Strix: `BUILT` but first generation failed at metadata RPC
  (#3043, BACKEND-GATE-ROCM-VLLM).
- llama.cpp denominator on Strix: measured 12.233 tok/s (Q4_K_M, 27B — not the
  4B model this gate uses). See `BACKEND-GATE-ROCM-LLAMACPP` row.
- **First-c1 gate (2026-09-21):** vllm-cli built from `1da8b4d` with
  `VLLM_CPP_HIP=ON`, `VLLM_CPP_HIP_ARCHITECTURES=gfx1151`, ROCm 5.7, clang-17.
  rc job `08de4b1d` on `strix:gpu0`. Binary sha256 not recorded (26976 B
  thin client; logic in libvllm.so, 28185496 B). Model: Qwen3-4B BF16 at
  `1cfa9a72`, loaded from `/tmp/models/Qwen3-4B`.
  - First-c1 (max-tokens 1, temp 0): exit 0, output " Paris", 0.264s.
  - Stability (5x repeat): 5/5 PASS, no hangs, 0.124-0.487s per run.
  - Throughput c1 (max-tokens 64, 3 repeats): 10.334, 9.168, 9.580 tok/s.
  - Non-fatal warning every run: "The GPU node has an unrecognized id."
    (ROCm 5.7 does not fully recognize gfx1151).

## Risks

- The first-c1 hang (#3111) does not reproduce on current main with ROCm 5.7.
  The original fault may have been fixed by BACKEND-ROCM commits since 2026-09-09,
  or the ROCm version difference (5.7 vs 7.2.4) prevents it. The managed-alloc
  fix (`row/BACKEND-ROCM-MANAGED-ALLOC-FIX`, still unmerged) addresses a Q4_K
  hang (#2511) that is a separate issue.
- SGLang's greedy decode non-determinism means a strict token gate may not be
  achievable; a distributional gate needs ratification.
- The 3111 branch is 380 commits stale; rebasing carries conflict risk.

## Stop conditions

1. First-c1 runs clean on Strix — **ACHIEVED** (2026-09-21, 5/5 passes + throughput).
2. Token gate passes (or a distributional gate is ratified).
3. Throughput ratio established at c1, c4, c32.
4. Memory measured.
5. Row advances from `INVENTORIED` to `ACTIVE` (first gate result) and
   eventually `DONE` (all axes satisfied or explicitly waived).

## Now

`ACTIVE`. The first-c1 hard blocker (#3111) is cleared: vllm.cpp completes
first-c1 inference on Strix with exit 0, 5/5 stability passes, and ~10 tok/s
at c1 with 64 tokens. The ROCm version on the box is 5.7, not the 7.2.4 the
spec was written for. The hang may have been fixed by code changes since
2026-09-09 or may not reproduce on 5.7. The diagnostic harness is built and
CPU-validated on the 3111 branch but not merged; the first-c1 result was
obtained with the production binary. Next: establish the token gate against a
reference oracle, then throughput at c4 and c32.

## Git integration

One pull request for spec + implementation, per repository default.
