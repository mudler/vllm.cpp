ID: ISSUE-GH-1863
Title: ROCm decode is 2.4x behind llama.cpp on a DENSE model (Ornith-1.5-9B, qwen35): the gap is not MoE-specific
Row: BACKEND-ROCM
State: OPEN
Kind: perf
GitHub: 1863
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-24
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ROCm decode is **2.4x behind llama.cpp on a DENSE model**, measured on
> `Ornith-1.5-9B` (arch `qwen35`, no MoE block, no expert offload, fits VRAM with
> 10 GiB to spare). The gap therefore is not MoE-specific, which is what
> [#1400](https://github.com/mudler/vllm.cpp/issues/1400) and
> [#1294](https://github.com/mudler/vllm.cpp/issues/1294) had measured and where
> every current ROCm decode hypothesis sits.
>
> ## Measurement
>
> `Ornith-1.5-9B-Q4_K_M.gguf`, 5.23 GiB, 8.95 B params, batch 1, 32 tokens.
> RX 9060 XT (gfx1200, 16304 MiB), ROCm 7.2.3, NixOS. Same card, same file, both
> sides inside the same 100 seconds under the `$HOME/gpu.lock` mutex, with free
> VRAM asserted above 13 GiB and zero resident model processes before each arm.
>
> | | decode t/s | vs ours |
> |---|---|---|
> | vllm.cpp `4b1154bc5` ROCm | **18.393 / 18.574** | 1.00x |
> | llama.cpp `b10451` (`10bf611e5`) HIP | **44.24 +/- 6.10** | **2.40x** |
>
> The 14B MoE was measured in the same session for context, and its gap is the
> same order rather than larger:
>
> | | decode t/s | vs ours |
> |---|---|---|
> | vllm.cpp, `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M` | 13.000 / 13.145 | 1.00x |
> | llama.cpp `b10451` HIP, same file | 38.06 +/- 8.01 | 2.90x |
>
> vllm.cpp run 1 of each arm is a cold load (6.469 and 4.452 t/s) and is excluded;
> runs 2 and 3 are warm and agree to within 1%.
>
> ## What this refutes
>
> That the ROCm decode deficit is a property of the MoE path. `Ornith-1.5-9B`
> reaches the kernels through `Qwen3_5ForCausalLM` with no MoE block, no router,
> no grouped expert GEMM and no offload, and still loses 2.4x. Any explanation
> scoped to expert routing or to `kMatmulBTQuantGrouped` cannot account for this
> measurement.
>
> ## Named hypothesis, not yet tested
>
> `QuantizeQ8KK` has two call sites in `src/vt/rocm/rocm_grouped_gemm.hip`. #1400
> profiled the grouped one at `:547`, inside `MatmulBTQuantGroupedKernelRocm`, and
> found it at 35% of MoE decode GPU time. The other is at `:479`, inside
> **`MatmulBTQuantKernelRocm`** — the non-grouped, dense path — and a dense
> k-quant GGUF now routes there, because `kMatmulBTQuant` was registered on ROCm
> by [#523](https://github.com/mudler/vllm.cpp/pull/523) (`7bcf2f5e1`, 2026-08-21).
>
> ```c
> // rocm_grouped_gemm.hip:479, in MatmulBTQuantKernelRocm
> QuantizeQ8KK<<<static_cast<unsigned>((m * nsb + 127) / 128), 128, 0, s>>>(
> ```
>
> At dense decode `m` is 1, so the launch is thinner than the MoE case #1400
> measured, where `m` was T x top_k. If the shared quantizer is the cause, one
> parallelization fix moves both paths. **This is a hypothesis with two datapoints
> and no profile.** Nothing here establishes it; a `rocprofv3 --kernel-trace`
> pass on the 9B is what would.
>
> ## Honest gaps in this measurement
>
> - **The two sides are not measured with one construction.** `vllm-cli` divides
>   completion tokens by whole-call wall time; `llama-bench` reports `tg32`. With
>   a 5-token prompt these are close, but #1400's method — differencing
>   `--max-tokens 4` against `--max-tokens 36` — is the one that isolates decode
>   on both sides, and it is what a gate on this number owes.
> - **Error bars are wide.** +/- 6.10 on 44.24 is 14%, and +/- 8.01 on 38.06 is
>   21%, at 3 repetitions.
> - **The host was not idle.** VRAM was clear and asserted, but loadavg ran
>   2.2-3.0. An earlier attempt at these numbers had to be discarded entirely
>   because a resident `llama-server` held the card while loadavg looked normal,
>   so the precondition here is free VRAM rather than load.
> - **One prompt, one model, one quantization, batch 1.** No claim is made about
>   other shapes.
> - The llama.cpp `b10451` HIP build is new: the oracle file records this pin as
>   gateable from a CPU-only run and states in terms that no timing may be
>   promoted out of that evidence. The HIP figure above is a fresh measurement on
>   this card, not a previously established one.
>
> ## Separately observed, not this issue
>
> `VT_GGUF_KEEP_QUANT=0` now fails with `hipMalloc: out of memory` on **both**
> models on this 15.92 GiB card — the bf16 fallback no longer fits. Keep-quant has
> become load-bearing rather than an optimization. That deserves its own issue and
> is not folded in here.
>
> Owned by `BACKEND-ROCM`.
>

## Resolution

-
