# Qwen3-dense DECODE strict-16/16 bit-match razor — investigation + remaining scope

Row: `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (correctness follow-up to the
near-tie-robust gate). Worktree: `agent-a7e66dd02ed286cb2` (investigation),
`agent-a31b006ff4766f03b` (FA2 VARLEN d128 implementation). Status: **RESOLVED
2026-07-20 — the FA2 VARLEN d128 decode was VENDORED, ROUTED, and MEASURED. The
exact kernel BIT-MATCHES vLLM's decode attention OUTPUT (teacher-forced logit
delta 0.0000 at all-but-the-near-tie positions), but strict 16/16-vs-vLLM's-decode
is GENUINELY bf16-tie-bounded and NOT reached (the exact kernel scores one fewer
strict than the fallback). Shipped OPT-IN (default OFF); the near-tie-robust gate
stays the closure. See "RESULT (2026-07-20)" below.**

## RESULT (2026-07-20) — FA2 VARLEN d128 vendored, routed, measured; escape hatch CONFIRMED

**Implementation (this campaign, worktree `agent-a31b006ff4766f03b`, kept):**
- Vendored the d128 bf16 split-KV instantiations
  `flash_fwd_split_hdim128_bf16_{,causal_}sm80.cu` (mirror the d256 pair; CMake
  `_FA2_KERNEL_SRCS`). Note: vLLM's paged decode ALWAYS runs the split-KV kernel
  (the non-split `flash_fwd_kernel` does not read `block_table`); with a paged KV
  cache upstream `mha_varlen_fwd` forces the split kernel even at `num_splits==1`,
  so this IS `flash_fwd_splitkv` with `Split=false` — the same kernel the varlen
  prefill launcher uses, exactly what vLLM's `flash_attn_varlen_func` resolves to.
- `LaunchDecodeVarlenFA2Bf16` (`src/vt/cuda/cuda_flash_attn_fa2.cu`): PLAIN varlen
  decode — `cu_seqlens_q = query_start_loc` (1 q-row/req), `h = hq`,
  `h_h_k_ratio = groups`, `seqused_k` per-request K length, `is_causal = true`,
  **NO group swap**; `num_splits` = exact port of `num_splits_heuristic` (=1 for
  the gate seqlens). For `num_splits>1` the split combine writes O via
  `batch_idx*o_batch_stride`; with `seqlen_q==1` that equals `o_row_stride`, so
  `o_batch_stride := o_row_stride` makes the combine land the packed
  `[total_q,Hq,D]` output correctly (the packed-prefill combine restriction —
  `seqlen_q>1`, irregular spacing — does NOT apply to decode).
- Routed via `VT_FA2_DECODE_QWEN3` (`Fa2DecodeQwen3Enabled`, **default OFF**),
  gated to `d==128` bf16 paged causal pure-decode (only Qwen3-dense; the two gate
  models are d256, so the d256 arms are NEVER touched).
- Op-parity test `tests/vt/test_ops_paged_attn.cpp` — byte-exact vs the f32
  `ComposedPagedRef` at both Qwen3 ratios (16/8, 32/8), batch {1,2,4,8}, short
  (`num_splits==1`) AND long (`num_splits>1` split-combine) contexts; plus an
  opt-in-off fallback assertion. **`compute-sanitizer memcheck` 0 errors** (after
  sizing the test block_table to the kBlockN-rounded page count the real KV
  allocator over-provisions — the engine passes the block table at full
  `max_num_blocks_per_req` width, so the production varlen path is memcheck-safe).

**Does the varlen d128 kernel bit-match vLLM's attention OUTPUT? — YES.**
Teacher-forcing vLLM (`prompt_logprobs`) on OUR new-path sequence: at essentially
every divergence position vLLM's OWN argmax given our prefix IS our token, gap
**0.0000 nats** (bit-identical logprobs). The only REAL (first) forward
divergences are bf16 near-ties: `Qwen3-4B` p11 tok12 = 0.062, p12 tok11 = 0.250;
`Qwen3-0.6B` p5 tok10 = 0.125 (max over all positions 0.375). So our varlen output
reproduces vLLM's decode attention/logits within bf16 tie resolution.

**Strict counts (same-binary A/B vs the SAME `greedy_ids` oracle):**
| arm | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| CUDA-core fallback (default, `VT_FA2_DECODE_QWEN3=0`) | **12/16** | **10/16** |
| FA2 VARLEN d128 (`=1`) | **11/16** | **9/16** |

**16/16 NOT reached — and the exact kernel is ONE WORSE than the fallback.** This
is the spec's escape hatch, now proven with the EXACT vLLM kernel: strict-16/16
is bounded by bf16 near-tie resolution. The residual flips (e.g. 0.6B p0 tok5:
our=15344=vLLM's teacher-forced argmax, gap 0.0000, but vLLM's INCREMENTAL greedy
decoded 9625) are ties where vLLM ITSELF is self-inconsistent — its one-shot
prefill argmax disagrees with its incremental decode. The CUDA-core fallback
happens to round one such tie (p0 tok5) toward vLLM's decode; the "exact" kernel
rounds it toward vLLM's prefill argmax. Neither is "more correct"; there is no
fixed decode token to bit-match to.

**Disposition:** the varlen d128 kernel is KEPT as an OPT-IN faithful mirror of
vLLM's exact decode reduction (default OFF — default-ON would REGRESS strict
12→11/10→9 and flip the committed near-tie anchor, violating the parity-enabler
policy's "don't ship a regressing default"). The engine gate is NOT tightened to
strict equality (16/16 is not real). The near-tie-robust gate remains the correct
closure. **Regression: 27B `test_qwen27_paged_engine` 235/235 + 35B
`test_qwen36_paged_engine` 315/315 UNCHANGED; the default Qwen3 near-tie gate
still 16/16 (0.6B strict 12/16, 4B strict 10/16). `-Werror` 0-warn.** dgx tree
`~/work/vllm-cpp-decode-varlen-a31`.

## Goal
Tighten `test_qwen3_paged_engine` from the near-tie band to STRICT token-exact
16/16 vs vLLM 0.25.0's actual DECODE output on Qwen3-0.6B AND Qwen3-4B, by
bit-matching our decode-time attention reduction order to vLLM's.

## Baseline (main HEAD ade97ac, CUDA-core d128 decode fallback)
STRICT token-exact vs vLLM per-prompt greedy (batch=1, deterministic):
- Qwen3-0.6B: **12/16** (4 near-tie flips, max gap 0.125 nats).
- Qwen3-4B:   **10/16** (6 near-tie flips, max gap 0.25 nats).
The near-tie band gate PASSES 16/16 on both (this is the shipped, passing gate).

## What was tried (this session) — and DISPROVEN
Hypothesis (from the task): route Qwen3 d128 decode through the vendored FA2
split-KV kernel (`LaunchDecodeFA2Bf16`, the `seqlenq_ngroups_swapped`
`flash_attn_with_kvcache` path that the 27B/35B d256 arms use) to reproduce
vLLM's decode reduction order.

Implementation (fully working, then reverted): a d128 non-causal split-KV
instantiation (`flash_fwd_split_hdim128_bf16_sm80.cu`), CMake wire-up, a
head-dim-dependent `kBlockN` (128 for d128), a `head_dim==128` dispatch branch,
and a `VT_FA2_DECODE_QWEN3` gate widening `fa2_decode` to admit d128 Qwen3
ratios. Built clean (`-Werror` 0-warn on GB10, d128 kernel compiled).

**MEASURED RESULT — the group-swap split-KV route is WORSE, not a bit-match:**
- Qwen3-0.6B: **11/16** strict (was 12/16).
- Qwen3-4B:   **9/16** strict (was 10/16).
Divergences were early-token flips that then cascade (e.g. 0.6B p0 tok5
our=15344 vs vLLM=9625). So the vendored FA2 split-KV kernel computes CORRECT
attention but with a bf16 reduction order FURTHER from vLLM than the CUDA-core
fallback. Reverted to baseline — no regression, no dead default.

## Root cause (source-confirmed — AGENTS.md "trace the execution")
`~/venvs/vllm-oracle/.../vllm/v1/attention/backends/`:
1. **fa_version = 2 on GB10.** `fa_utils.py:132-260 get_flash_attn_version`:
   `device_capability.major` for GB10 is **12** (sm_121); the `==9`→FA3 and
   `==10`→FA4 branches miss, so it falls to `fa_version = 2`. vLLM runs Qwen3 on
   FA2. (Bundled `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`; no separate
   `vllm-flash-attn` pkg; flashinfer present but not the Qwen3 attn backend.)
2. **The entry is `flash_attn_varlen_func`, NOT `flash_attn_with_kvcache`.**
   `flash_attn.py:36-40,444-587` builds `scheduler_metadata` and calls the
   unified VARLEN forward for BOTH prefill and decode. The varlen path does NOT
   apply the `seqlenq_ngroups_swapped` group-swap that our `LaunchDecodeFA2Bf16`
   uses — the group-swap changes the M-tiling and thus the online-softmax
   accumulation order, which is exactly why our route diverged.
3. **Oracle is `enforce_eager=True`** (`scripts/qwen3-oracle-capture.py:102`),
   so no CUDA graphs; `flash_attn.py:470-483` leaves `max_num_splits` at the FA
   heuristic default (not the CG-fixed value). For the short gate sequences
   (prompt ~5 tok + 16 decode, block 16 ⇒ ≤2 blocks) `num_n_blocks` is 1, so
   the heuristic yields **num_splits = 1** (no split): vLLM decode is a single,
   non-split FA2 VARLEN reduction over the paged KV.

Why d256 (27B/35B) passes strict through the group-swap kvcache path but d128
does not: those are larger models with SPARSE near-ties, so the group-swap's
tiny bf16 rounding difference vs vLLM's varlen reduction never flips their 16
tokens; the small d128 models have DENSE near-ties (≤0.125–0.25 nats) that the
difference does flip. The group-swap path is not vLLM's decode reduction — it
just happens to be close enough for the large models.

## Remaining scope (the real bit-match — a scoped sub-campaign)
Reproduce vLLM's FA2 **VARLEN** d128 decode reduction (no group-swap), matching
`flash_attn_varlen_func` with seqlen_q=1, paged block_table, `seqused_k`, and
vLLM's num_splits (=1 for the gate seqlens). Concretely:
1. Vendor the d128 VARLEN (`run_mha_fwd`, non-split) instantiation — the
   `flash_fwd` (not `flash_fwd_splitkv`) kernel the varlen path uses; the
   launch template (`run_mha_fwd_hdim128`) already exists, only the explicit
   bf16 d128 instantiation `.cu` is needed (mirror `LaunchPrefillFA2Bf16`, which
   is today gated d256 — generalize it to d128, or add a decode-shaped varlen
   launcher that feeds 1 query row/request against the paged KV).
2. Route Qwen3 decode there (default ON only once it bit-matches). Note our
   PREFILL already reproduces vLLM's prefill logits bit-identically via the
   CUDA-core d128 path (teacher-forced gap 0.0000) — so the varlen FA2 decode is
   the last divergent op; do NOT touch prefill.
3. Byte-exact d128 attention parity test vs a reference; memcheck 0.
4. Then tighten `test_qwen3_paged_engine` to strict equality vs `greedy_ids`
   (the strict-form test was drafted this session and reverted; re-apply once
   16/16 lands) and re-run 27B 235/235 + 35B 315/315 (d256 must stay
   byte-identical — the varlen d128 change must not touch the d256 arms).

Risk/escape: if the varlen FA2 d128 reduction STILL does not flip our near-tie
tokens to vLLM's (i.e. even the exact varlen kernel rounds a ≤0.125-nat tie the
other way), then strict-16/16-vs-decode is genuinely bounded by bf16 tie
resolution and the near-tie-robust gate is the correct closure (as the shipped
gate already records). Measure before concluding.

## Evidence / repro
- dgx work dir this session: `~/work/vllm-cpp-decode-a7e` (fresh CUDA build,
  `-Werror` 0-warn; d128 kernel compiled).
- Baseline strict measured via a strict variant of `test_qwen3_paged_engine`
  (reverted): 0.6B 12/16, 4B 10/16. Group-swap FA2 d128: 0.6B 11/16, 4B 9/16.
- Run: `flock /tmp/gpu env PATH=$HOME/venvs/vllm-oracle/bin:$PATH
  ./tests/test_qwen3_paged_engine` under the standard build flags.
