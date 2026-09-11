ID: ISSUE-GH-1651
Title: GFX1100-TG150: serve Qwen3.5-4B-Q4_K_M on the RX 7900 XTX at >= 150 tok/s pure autoregressive tg
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1651
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-22
Updated: 2026-08-22
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Goal
>
> Serve `/home/ghazni/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf` on the RX 7900 XTX (gfx1100, RDNA3, 24 GiB, ROCm 7.14.0 container) at **>= 150 tok/s text-generation throughput**, pure autoregressive greedy decode:
>
> - no MTP / nextn drafter, no speculative decoding of any kind;
> - an 8-bit KV cache: fp8-e4m3 via vLLM's `cache_dtype=fp8` surface (per-tensor k/v scales), extending the landed `KV-FP8` CPU brick to a ROCm store + paged-attention read;
> - every existing correctness gate held throughout: the keep-quant integer core stays bit-exact vs CPU (`test_rocm_quant_dot`, unchanged), and token identity vs pre-campaign outputs on identical inputs is asserted at acceptance.
>
> Developer-ratified 2026-08-22.
>
> ## Acceptance gate
>
> Median of >= 5 repetitions, idle host, `$GPU_LOCK`/flock held for the whole window, batch 1, one ~512-token real prompt, 256 generated tokens, greedy (`--temperature 0 --seed 0`). The run records output tok/s AND steady-state TPOT; VRAM axis recorded beside them. A number taken under co-tenancy is provisional and never satisfies this gate.
>
> ## Why it is plausible, and what could falsify it
>
> Physics ceiling: ~2.74 GB of weights streamed per token at ~960 GB/s peak = ~2.9 ms/token (~350 tok/s). 150 tok/s = 6.67 ms/token end-to-end = 43% of the ceiling — plausible, never guaranteed, and no stage may declare a ceiling; a shortfall names the next traceable hypothesis instead.
>
> Current measured position (branch `row/ROCM-QUANT-GEMM-BW`, unmerged): warm wall clock 17.8 tok/s against 4.41 ms/token GPU-busy. Wall exceeds GPU-busy by roughly an order of magnitude per step, so the step is plausibly HOST-bound (dispatch/sync/scheduler) before it is kernel-bound. That inference comes from one capture at a different prompt length and is S1's first thing to verify or refute.
>
> ## Stages
>
> | Stage | Content |
> |---|---|
> | S1 | Fresh attribution re-take at current head on the EXACT gate workload: rocprofv3 both sides, same tool, wall vs GPU-busy split, per-family shares. Ranks S3-S5. |
> | S2 | Dispatch-collapse: HIP graph capture of the steady decode step (or FusedChain recipe reduction where capture cannot reach), routing through `vt::FusedChain` / `layers::MlpGateUpMethodBase` + `vt::MergedGemmGroup` per shared-seam policy. Expected largest lever if S1 confirms host-boundness. |
> | S3 | Quant GEMM toward >= 80% peak effective weight streaming (continues #1586's ladder, which owns the kernel internals). |
> | S4 | bf16 hipBLASLt arms (26.2% of baseline busy): algo-policy A/B at decode shapes; merged-GEMM seam where applicable. |
> | S5 | GDN decode family levers (16.9% of baseline busy), ranked by S1. |
> | S6 | ROCm fp8 KV cache: store in reshape-and-cache + dequant read in paged attention + `--kv-cache-dtype` CLI threading (today `rocm_paged_attn.hip` refuses fp8 KV by name; `KV-FP8` W1 landed the codecs and the CPU brick). Correctness: distributional gate ratified against the f32-KV arm if greedy anchors move at fp8 rounding ties, per the near-tie doctrine. |
> | S7 | Acceptance gate run + landing: docs/USAGE.md weights provenance (repo @ revision, file size, sha256), BENCHMARKS row, spec `## Outcome`. |
>
> S2-S5 ordering is S1's output, not this table's.
>
> ## Non-goals
>
> - Speculative decoding / MTP in any measurement arm.
> - Other boards, other models, other quantizations.
> - Declaring a performance ceiling.
>
> ## Relations
>
> - #1586 stays scoped to the quant-GEMM bandwidth ladder (S3 consumes its results).
> - #1587 owns the W1 providers this campaign runs on.
> - `BACKEND-ROCM` is the owning matrix row; spec: `.agents/specs/gfx1100-tg150.md`.
>
> FOLLOWING_AGENTS_PROTOCOL
>

## Resolution

GitHub records issue #1651 as completed by `ghazni101` on 2026-08-22, but the local authority rejects that state: pull request #1652 closed without merge, the named spec is absent, and external pull request #6 reports 76.6 tok/s against the 150 tok/s gate. The canonical record remains OPEN.
