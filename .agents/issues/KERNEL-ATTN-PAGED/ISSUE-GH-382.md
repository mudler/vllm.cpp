ID: ISSUE-GH-382
Title: decode-opt attention kernel is head_dim-256 only; head_dim 128 (Qwen3-dense/Llama/Mistral) falls to the block kernel
Row: KERNEL-ATTN-PAGED
State: OPEN
Kind: perf
GitHub: 382
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `KERNEL-ATTN-PAGED`
>
> ## The gap
>
> `PagedAttentionDecodeOptKernel` -- the FlashInfer-style warp-split-KV decode kernel with coalesced 128-bit K/V loads, warp-shuffle online softmax and register-resident O -- is reachable only through
>
> ```cpp
> // src/vt/cuda/cuda_paged_attn.cu:1978
> if (DecodeOptEnabled() && d == 32 * kDecEpl) {   // kDecEpl == 8, so d == 256
> ```
>
> `kDecEpl` is fixed at 8 because `LoadRow8` is hard-coded to eight elements per lane (`:281-300`). So **head_dim 128 never reaches it** and falls through to the generic block kernel at the bottom of `LaunchDecode`.
>
> head_dim 128 is what Qwen3-dense, Llama and Mistral use. It is not an edge case.
>
> The d128 FA2 varlen arm (`Fa2DecodeQwen3Enabled`, `:2557`) covers part of this on the architectures that build FA2 at all. On **sm_110 it covers none of it**: `cmake/CudaArchFeatures.cmake:349` reads `fa2|8.0,8.6,8.7,8.9,12.0a,12.1a`, so `11.0` is absent, `VLLM_CPP_FLASH_ATTN` is undefined, and the entire FA2 block -- `fa2_decode_qwen3` included -- compiles to `false`. Every d128 decode on Thor therefore lands on the generic block kernel with nothing underneath it. `docs/BUILD.md:56-58` describes the general form of this: attention falls back to "the portable path, which is slower. Nothing fails and no test goes red."
>
> Whether that `fa2` cell should gain `11.0` is a **separate question** and I am not bundling it here; I mention it only because it is why the head_dim gate bites harder on sm_110 than the gate alone suggests.
>
> ## What I measured
>
> sm_110 / Jetson AGX Thor, Qwen3-1.7B-NVFP4A16, W4A16 Marlin build (the one #326 enabled). Prototype: template the decode-opt kernel on elements-per-lane, add an `EPL=4` instantiation for d128, leave `EPL=8` forwarding to the existing `LoadRow8` so d256 is byte-for-byte unchanged.
>
> Serving A/B, same harness / model / config, 128 tokens, temp 0, 3 warmups:
>
> | | c=1 | c=2 | c=4 | c=8 |
> |---|---|---|---|---|
> | block kernel | 81.630 | 162.738 | 316.255 | 477.070 |
> | EPL=4 opt kernel | 131.023 | 258.953 | 508.139 | 936.048 |
> | ratio | 1.61x | 1.59x | 1.61x | 1.96x |
>
> Kernel numerics vs the block kernel, 8 context lengths straddling the 32-token page boundary, 2048 elements each: six of eight are **100% bit-for-bit identical** (ctx 1, 31, 32, 33, 64, 685); ctx 100 is 99.95% with max_abs 3.7e-09; ctx 1000 is 99.95% with max_abs 6.1e-05 / max_rel 5.1e-03. The residual is consistent with the reduction-order difference between the two kernels, not with a correctness defect -- but see the limits below.
>
> ## Limits of that evidence
>
> - One model, one head_dim, one arch. Nothing measured off sm_110.
> - The A/B is a **single repetition** per point and was taken across two builds.
> - **No model-level token-exact gate was run.** I cannot run your SACRED gates. A token-exact run happened during the experiment but its output was never written to disk, so I will not quote it; re-running means loading models on a box serving live production traffic, which I will not do outside a maintenance window.
> - The comparator has been shown able to fail (seeding 1-ULP corruption into 3 of 2048 elements drops that row from 100.0000% to 99.8535% bitwise and moves no other row), but that is a **comparator self-test, not a kernel negative control** -- it proves the instrument works, not that the kernel comparison would have caught a bad kernel.
>
> ## Why this needs a decision, not just a patch
>
> Widening the gate is arch-independent and the two kernels reduce the KV sequence in a different **order**, so a greedy anchor can move at an exact bf16 tie -- i.e. goldens for d128 models could shift. I am **not** asking you to take that on the evidence above.
>
> My proposal is to land it **gated OFF** (`VT_ATTN_DECODE_D128`, opt in with `=1`), correctness-complete, exactly as `perf(fa2): decode GQA group-swap port, gated OFF - correctness-complete` (#48) landed before #49 flipped it ON against the full gate. Default-OFF keeps every existing golden byte-identical and leaves the flip -- near-tie razor, distributional gate, regen under the ratified-tie rule -- as a separate change on your hardware.
>
> PR follows. Two things I would rather ask than assume:
>
> 1. `scripts/check-role-discipline.py` wants a `row/<ID>` branch. As in #326 I have not claimed an internal roadmap row as an outside contributor. Tell me how you want this satisfied.
> 2. If you would rather this arrive default-ON, or not at all until a d128 GQA-fused kernel exists, say so before I spend your review time.
>

## Resolution

-
