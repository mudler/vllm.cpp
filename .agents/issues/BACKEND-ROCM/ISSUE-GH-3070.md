ID: ISSUE-GH-3070
Title: test(BACKEND-ROCM): reconcile the Strix full-suite failures
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 3070
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-08
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> The operator's HIP full-suite gate for PR #3010 is unresolved. Strix lease `d7a790d5-158e-46d8-9e32-1284f4c41a6c` recorded 18 failed targets and then lost its host to a reboot at 741/745 completed tests. It supplies no completed full-suite verdict. Issues #3066 and #3068 own the diffusion fixture failures; this issue owns the remaining 17 targets and the required baseline pairing.
>
> The failed targets fall into these groups:
>
> - `test_qwen3_dflash2_draft`, `test_dflash2_runner_reach`, `test_dflash2_concurrency`, `test_dflash2_ctx_capacity`, and `test_dflash2_prefix_cache`: explicit missing ROCm `IndexSelect`; host reference fallback is refused because device memory is not host-addressable.
> - `test_dflash2_exl3_reach` and `test_dflash2_draft_phase_trace`: caught exceptions and missing draft outputs; the log does not print their exception text, so the operation cause is not established.
> - `test_mtp_depth`: explicit missing `CausalConv1dSpecUpdate`, plus an async-capability expectation that needs separate diagnosis.
> - `test_qwen4_exp_gguf_load_plan`, `test_qwen4_exp_gguf_weights`, and `test_qwen4_exp_runner`: ROCm keep-quant/gather expectations or explicit refusal. #2394 owns missing block-decoding gather arms; #1940 owns broader I-quant coverage.
> - `test_placement_reach` and `test_engine_scratch_steady_state`: fixtures appear to assume CPU placement or a CPU pool while the runtime selects ROCm. This is not yet a runtime-paired diagnosis.
> - `test_loaded_engine_dense` and `test_llm_engine`: expected async scheduling/multiple concurrent batches, but the ROCm backend does not advertise sampled-token readback support.
> - `test_cuda_quant_dot`: a CUDA-named residency fixture routes through `CurrentPlatform()` and expects formats not admitted by ROCm.
> - `test_glm_moe_dsa_forward`: explicit missing ROCm `DsaIndexerLogits`; the related W2 source/spec and historical closed issues need reconciliation.
>
> The affected tests and relevant registration, loader, and runner sources are byte-identical between main `08a34c3a74d78046f83886f242d07110a70ff45e` and sampling repair `2bd0ca67f051d7ef7d48b52ad75a8729fe30d0b3`. That is supporting evidence only. None of these 17 targets has a paired main runtime result in the supplied log.
>
> Next gate: reproduce each target on pinned main under the same leased device, build, environment, and serial scheduling. Capture omitted exception text in diagnostic scratch. Then distinguish test-fixture assumptions from actual missing backend capabilities, and assign each required implementation to its owning row/spec before changing code. A change in another row is not authorized merely to make this report green.
>
> Do not enable unsafe managed allocation to hide missing native operations. Do not expand keep-quant tables to bypass residency refusals. Do not call this a passing or narrowly waived gate. Local evidence remains `/tmp/pr3010-full-lease.log`; the separate focused sampling tests do not replace this full-suite obligation.
>

## Resolution

-
