# Qwen3-dense DECODE strict-16/16 bit-match razor — investigation + remaining scope

Row: `MODEL-TEXT-qwen3-qwen3-for-causal-lm` (correctness follow-up to the
near-tie-robust gate). Worktree: `agent-a7e66dd02ed286cb2`. Status: **investigation
done; strict bit-match NOT achieved — the confirmed remaining scope is an FA2
VARLEN d128 decode path (a scoped sub-campaign), recorded here for continuation.**

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
