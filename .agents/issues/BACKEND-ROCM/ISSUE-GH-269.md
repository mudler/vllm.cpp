ID: ISSUE-GH-269
Title: ROCm gfx1200: Qwen3-0.6B produces wrong greedy output despite all-native execution
Row: BACKEND-ROCM
State: OPEN
Kind: verification
GitHub: 269
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-10
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ## ROCm gfx1200 (RX 9060 XT): Qwen3-0.6B produces wrong greedy output despite all-native execution — embedding gather cleared
>
> **Board:** AMD Radeon RX 9060 XT, `gfx1200` (Navi 44, RDNA4, discrete). ROCm
> 7.2.3, hipClang/Clang 22.0.0. M0/M1 independently verified on this board (see
> my earlier comment).
>
> **Repro:**
> ```sh
> ./build-hip/examples/vllm-cli --model <Qwen3-0.6B, bf16 safetensors> \
>   --prompt 'The capital of france is' --max-tokens 8 --temperature 0
> ```
> - CPU (`--device cpu`): ` Paris. The capital of the United States` — correct.
> - ROCm (default `auto`, picks `kROCM`): ` 1000000` — garbage.
>
> No crash, no missing-kernel throw, no reference-tier fallback (discrete board,
> none available). `VT_OP_PROVIDER_STATS=1` shows every exercised op resolving
> `selected=vt-native`: `kEmbedding`, `kRopeCosSinCache`, `kCastBf16`, `kRmsNorm`,
> `kMatmulBT`, `kQkvSplit`, `kRopeFromCache`, `kReshapeAndCache`,
> `kPagedAttention`, `kSiluAndMul`, `kGreedyArgmax`. All of these have
> cross-device coverage in `test_backend_cross_device.cpp` and pass there against
> synthetic data — so this is either a real-shape/real-value edge case none of
> that synthetic coverage hits, or a bug outside the op kernels themselves.
>
> **Isolated to one token.** Capitalizing the prompt (`'...of France is'`) gets
> the *correct* answer (` Paris. The capital of Italy is Rome`). Both prompts
> tokenize to 5 tokens, identical except position 3: `47587` ("france") vs
> `9625` ("France"). Same shape, same op sequence — one input token id differs
> and the output goes from coherent to garbage.
>
> **Ruled out:**
> - The `is_cuda()`-vs-`is_cpu()` host-pointer-aliasing defect at
>   `dense_attn_block.h:181` — already fixed generically, confirmed correct for
>   `kROCM` in current source.
> - **The embedding gather itself.** Pulled `model.embed_tokens.weight` directly
>   from the safetensors file and ran `vt::Embedding` on CPU and ROCm at the
>   real row indices 47587 and 9625 (not a repacked small table — same offset
>   arithmetic as production). Both rows are **bit-exact** against the CPU
>   oracle. The table bytes and the gather kernel are correct for this row; the
>   bug is downstream, in the per-layer forward.
>
> **Not yet root-caused.** Candidates: `kRmsNorm` (cross-lane reduction),
> merged-QKV `kMatmulBT` (hipBLASLt GEMM), RoPE, `kPagedAttention` (hand-written,
> least test coverage), or `kGreedyArgmax` (its ROCm `ArgmaxK` NaN handling looks
> suspect by inspection — a NaN never satisfies `v > best`, so a fully-NaN row
> would leave `arg` at a sentinel `0x7fffffff` rather than behaving like the CPU
> path; unconfirmed as the cause here, but cheap to check first since it's the
> last op before the wrong token is picked).
>
> Full writeup, evidence, and next steps: `.agents/specs/rocm-gfx1200-m2-correctness.md`
> (this PR/branch — will link once posted).
>
> Distinct from #201 and #132 — neither reproduced on this board/toolchain.
> Does not touch the in-flight Qwen3.5 GDN kernel work (disjoint op set, disjoint
> board).

## Resolution

-
