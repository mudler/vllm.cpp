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

- **Token gate:** `PARTIAL` — llama.cpp (Q4_K_M) vs vllm.cpp (BF16) partial
  gate completed on 2026-09-22. All 6 prompts succeeded on both engines.
  First generated token matches on 4 of 6. Not token-exact due to
  quantization difference (BF16 vs Q4_K_M). A quant-matched comparison is
  blocked because vllm.cpp does not support the `qwen3` GGUF arch.
  vLLM/SGLang oracles are blocked on ROCm 5.7 (they need ROCm 7.2 + torch
  2.13.0+rocm7.2).
- **Throughput:** `PARTIAL` — vllm.cpp c1 ~9-10 tok/s (BF16, GPU); llama.cpp
  ~21 tok/s (Q4_K_M, CPU). Different model formats, not directly comparable.
  Other engines not yet measured on Strix.
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
- **Partial token gate (2026-09-22):** llama.cpp (Q4_K_M) vs vllm.cpp (BF16),
  rc job `d9f5998d` on `strix:gpu0`. 6 prompts, max-tokens 128, temp 0,
  seed 42. vllm-cli sha256
  `f8ade18ef41bbcb06e1f221e29b04f6c456c4519c60275a470eea782382db74b`.
  llama-cli sha256
  `e66df178ef5f019ad94db768abd552466b7c10ae2846f5bbf3ecb00dfa8982bd`.
  Model: Qwen3-4B at `1cfa9a72`.
  - All 6 prompts completed on both engines with coherent, correct output.
  - First generated token matches on 4 of 6 prompts (0, 3, 4, 5).
  - Prompt 1 (primary colors): Q4_K_M gives subtractive (red/blue/yellow),
    BF16 gives additive (red/green/blue) — both valid.
  - Prompt 2 (boiling point): Q4_K_M gives 100C directly, BF16 gives 212F
    then converts — both correct.
  - Q4_K_M (llama.cpp) entered thinking mode (`[Start thinking]` blocks);
    BF16 (vllm.cpp) did not.
  - Not token-exact: quantization difference (BF16 vs Q4_K_M) explains
    divergence. A quant-matched comparison requires vllm.cpp to support the
    `qwen3` GGUF arch, which it does not (only qwen35/qwen35moe/qwen3next).
  - vLLM and SGLang oracles cannot run on Strix: both need torch
    2.13.0+rocm7.2 (ROCm 7.2), and Strix has ROCm 5.7 with no /opt/rocm.
  - Full results persisted at `/tmp/vllmcpp_bf16_results.log` on Strix.

## Risks

- The first-c1 hang (#3111) does not reproduce on current main with ROCm 5.7.
  The original fault may have been fixed by BACKEND-ROCM commits since 2026-09-09,
  or the ROCm version difference (5.7 vs 7.2.4) prevents it. The managed-alloc
  fix (`row/BACKEND-ROCM-MANAGED-ALLOC-FIX`, still unmerged) addresses a Q4_K
  hang (#2511) that is a separate issue.
- SGLang's greedy decode non-determinism means a strict token gate may not be
  achievable; a distributional gate needs ratification.
- The 3111 branch is 380 commits stale; rebasing carries conflict risk.
- vLLM and SGLang cannot run on Strix ROCm 5.7. Both require torch
  2.13.0+rocm7.2 (ROCm 7.2), which has no ROCm 5.7 wheels. The vLLM oracle
  worker hardcodes `/opt/rocm/bin/hipcc`. A previous lease installed ROCm 7.2.4
  globally at `/opt/rocm`, but it was cleaned up. This blocks the primary
  token gate and the full four-engine harness until ROCm 7.2 is restored or a
  different device is used.
- vllm.cpp does not support the `qwen3` GGUF arch (only qwen35/qwen35moe/
  qwen3next), so a quant-matched llama.cpp vs vllm.cpp comparison is impossible
  with this model. A different model or adding `qwen3` GGUF support is needed
  for a token-exact C++-only gate.

## Stop conditions

1. First-c1 runs clean on Strix — **ACHIEVED** (2026-09-21, 5/5 passes + throughput).
2. Token gate passes (or a distributional gate is ratified).
3. Throughput ratio established at c1, c4, c32.
4. Memory measured.
5. Row advances from `INVENTORIED` to `ACTIVE` (first gate result) and
   eventually `DONE` (all axes satisfied or explicitly waived).

## Now

`ACTIVE`. The first-c1 hard blocker (#3111) is cleared (2026-09-21). A
partial token gate between llama.cpp (Q4_K_M) and vllm.cpp (BF16) completed
on 2026-09-22: all 6 prompts succeeded on both engines, first token matches on
4 of 6, not token-exact due to quantization difference. The primary token
gate (against vLLM) and the full four-engine harness are blocked: vLLM and
SGLang need ROCm 7.2 + torch 2.13.0+rocm7.2, and Strix has ROCm 5.7. The ROCm
7.2.4 install a previous lease put at `/opt/rocm` was cleaned up. A quant-
matched C++-only gate is also blocked: vllm.cpp does not support the `qwen3`
GGUF arch. Next: restore ROCm 7.2 on Strix or find a device that can run
vLLM/SGLang, then run the full four-engine harness.

## Git integration

One pull request for spec + implementation, per repository default.
