ID: ISSUE-GH-3105
Title: fix(BACKEND-ROCM): diagnose the gfx1100 Qwen3 paged anchor drift
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 3105
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> The unchanged gfx1100 HIP build at `760c8dd7e27781430b909c3679790c52fe332182`
> fails the ordinary Qwen3-0.6B paged gate and its KV-boundary case.
> Both fail the device-specific anchor check at
> `tests/parity/test_qwen3_paged_engine.cpp:365`.
>
> Prompt 1 is `Once upon a time,`. At zero-based generated position 9,
> the engine emits token 264 and the committed ROCm anchor expects 279.
> The selected anchor is `our_ids_rocm.npy`, not the CPU/CUDA anchor.
> Both cases stop at this anchor assertion before any later near-tie check.
> The boundary case fails before its separate boundary exercise.
>
> The operator reproduced the result on the original HIP library and a clean
> baseline worktree. The baseline executable SHA256 is
> `08e3bbc5c53a09bc2e0a2fc9d710303deab86975eedec706fdbaa81e60fb23b8`.
> It reports 4 cases, 2 failed cases, and 72 passing assertions out of 74.
> The F16 implementation reports the identical errors and assertion counts.
> All 19 paired existing HIP failures match individually across that change.
> This establishes that the observed failure predates #3092. It does not
> establish whether the model execution, artifact selection, or anchor is wrong.
>
> This differs from #3102, which records CPU prompt 0, position 5, tokens
> 15344 versus 9625. It also differs from #269, which records a gfx1200
> failure with the lowercase-France prompt. #3070 owns 17 other full-HIP
> failures and their Strix baseline pairing; that measurement remains separate.
>
> The operator used the supplied local RX 7900 XTX host, serial CTest,
> `HIP_VISIBLE_DEVICES=0,1`, `ROCR_VISIBLE_DEVICES=0,1`,
> `VT_OP_PROVIDER_STATS=1`, and `VT_CUDA_GRAPH_DEDUP=0`, under the GPU mutex.
> The exact command, library and executable hashes, clean source identity,
> logs, and comparison are retained under
> `/home/vikash/vllm.cpp-rdna3-f16-impl/build-rocm-f16-evidence/full-hip-baseline-v1/`:
> `command.json`, `operator-input-audit.json`, `operator-run.log`,
> `operator-receipt.json`, and `operator-comparison.json`.
> The baseline log SHA256 is
> `abd5ec9157b4e3d553782d2251348dc23eeb70dc4374586359d21fd271a627cb`.
>
> The backend row owes an artifact-matched diagnosis against the active oracle
> and a reviewed correction. Preserve the current anchor assertion, fixtures,
> and tolerances during diagnosis. Do not refresh the golden solely to make
> this gate green. The F16 row records this unresolved baseline failure under
> its own Owed section without expanding its implementation scope.
>

## Resolution

-
