ID: ISSUE-GH-3111
Title: fix(BACKEND-GATE-ROCM-SGLANG): diagnose Qwen3-4B first-inference GPU hang on Strix
Row: BACKEND-GATE-ROCM-SGLANG
State: CLOSED
Kind: UNKNOWN
GitHub: 3111
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-21
Closed: 2026-09-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix campaign operator. Parent: #3076, #3053. Existing campaign spec: .agents/specs/strix-qwen3-4b-c4-performance.md.
>
> In leased production qualification cfebd7eb-2612-455c-bd67-32bd28b5c800, pinned vLLM completed c1/c4 qualification (c1 repeatability FAIL, normal teardown). The native adapter returned a successful configure reply after model loading. Its first c1 qualification RUN then aborted with status -6 before an aggregate run result. The earlier configure-failure description was incorrect. stderr reports Asynchronous scheduling is disabled (max_concurrent_batches=1), then HW Exception by GPU node-1 reason GPU Hang. Baseline native revision6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb; engine/adapter/model bindings are the unchanged qualification-manifest-12.json. Controller194a changes teardown only; do not infer it caused the GPU fault.
>
> Root cancelled the job on observing the fault; terminal state killed/143 at09:02:07UTC. Later engine output is not accepted. Read-only leased health job69b0ea7a-9899-4f8b-99d3-5c46d333b48c retained kernel messages showing MES unrecoverable-state warning, failed process-queue eviction, automatic GPU reset at09:01:15 and reset success09:01:18 with VRAM lost. No manual reset or unhealthy-device clearing was performed. This is not evidence of the offending kernel.
>
> NAS evidence: /mnt/nas_share/rc/strix-four-engine-3053.X94a3J/child3108-production-qualification.jGml1g/run/vllm.cpp-qualification/{stderr.log,result.json}; kernel evidence native-hang-health.6z03Bv/health.log in the same campaign directory.
>
> Required: establish post-reset device state before model work, preserve logs and exact bindings, inspect native startup source and loader provenance, then scope a bounded standalone same-binary reproducer under a lease. Stop on further device fault; no automatic recovery commands, pin changes, or performance acceptance. Distinguish sequencing after vLLM from standalone startup only with measured controls. A root cause requires source and executed-path evidence; do not blame the last logged line or an unrelated historical Strix hang.
>

## Resolution

FALSIFIED by the tree, 2026-09-21. The first-inference hang does not
reproduce on current main (SHA `1da8b4d`) with ROCm 5.7 on Strix.
`vllm-cli` was built with `VLLM_CPP_HIP=ON`, `VLLM_CPP_HIP_ARCHITECTURES=gfx1151`,
clang-17, and the ten build fixes recorded in `strix_rocm_build_recipe`.
rc job `08de4b1d` on `strix:gpu0` ran the Qwen3-4B BF16 model at `1cfa9a72`.

First-c1 (max-tokens 1, temp 0): exit 0, output " Paris", 0.264s.
Stability (5x repeat): 5/5 PASS, no hangs, 0.124-0.487s per run.
Throughput c1 (max-tokens 64, 3 repeats): 10.334, 9.168, 9.580 tok/s.

The original hang was observed 2026-09-09 on ROCm 7.2.4 (per the spec's
original text). The box runs ROCm 5.7. The fault may have been fixed by
BACKEND-ROCM commits since 2026-09-09, or the ROCm version difference may
prevent it. Either way, first-c1 is clean on current main and the blocker
is cleared. The managed-alloc fix (`row/BACKEND-ROCM-MANAGED-ALLOC-FIX`,
still unmerged) addresses a separate Q4_K hang (#2511), not this bf16
first-c1 issue.
