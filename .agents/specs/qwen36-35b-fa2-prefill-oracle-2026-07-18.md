# 35B FA2-prefill + fused-preamble: oracle-fidelity decision (2026-07-18)

Investigation claim: `CLAIM-35B-FA2-ORACLE`. Scope: the MIRROR-vLLM oracle
question that gates enabling FA2 prefill + the fused qk-norm-rope preamble on
the **35B** full-attention layers (levers `VT_FA2_PREFILL` / `VT_FUSE_ATTN_PREAMBLE`,
model gate `qwen3_5.cpp:3250-3286`). Relates to matrix row `KERNEL-ATTN-FA2`
(owned by `CLAIM-SERVE-GATE-1`; not re-owned here). The oracle decision (no
re-baseline) is recorded below; the 35B FA2-prefill default-ON FLIP subsequently
LANDED via `CLAIM-35B-FA2-FLIP-1` (see "LANDED" section) — no oracle change.

## Question

Enabling FA2 prefill rounds the query to bf16 and runs vLLM's `flash_fwd_splitkv`
instead of our `PagedFlashWmmaGqa` (f32 q). The 35B full-attn deterministic-greedy
stream was recorded (2026-07-09) to DIVERGE within 16 tokens under
`VT_FUSE_ATTN_PREAMBLE=1`, so both levers ship DEFAULT-OFF for the 35B
(`FuseAttnPreambleOn(fp4=false)` returns false). The MIRROR-vLLM question:
**does adopting FA2 bf16-q change the 35B greedy tokens vs a production-faithful
oracle, and is our stored oracle production-faithful?**

## What vLLM PRODUCTION runs for the 35B full attention (source-grounded)

- Backend select on GB10 (sm_121, `device_capability.major == 12`) falls to the
  `else` branch → **FLASH_ATTN is top priority** (`vllm/platforms/cuda.py:145-160`).
- FA version: not major 9, not major 10 → **fa_version = 2** (FA2)
  (`vllm/attention/utils/fa_utils.py:154-162`).
- The 35B is `qwen3_5_moe`; its full-attn layer is `Qwen3NextAttention`
  (`qwen3_5.py:137` → `qwen3_next.py:225`). Its preamble dispatches to the
  **FUSED** `fused_qk_rmsnorm_rope_gate` Triton kernel whenever
  `attn_output_gate && rotary is neox_style && is_cuda && text_only`
  (`qwen3_next.py:318-357`) — all true on GB10 for a text prompt.
- **This dispatch is set at `__init__` from platform/config; it is INDEPENDENT
  of `enforce_eager`.** FA2 backend selection is likewise `enforce_eager`-independent.
- vLLM's fused kernel deliberately **round-trips the normed q/k through bf16
  before RoPE** (`.to(INPUT_DTYPE).to(tl.float32)`,
  `vllm/model_executor/layers/fused_qk_norm_rope.py:67`), and its RoPE FMA is
  `o1 = x_rot1*cos - x_rot2*sin` (`:104-105`) — i.e. vLLM's fused kernel is
  bit-designed to equal its own unfused reference.

**Conclusion:** vLLM PRODUCTION (graphed) 35B full attention = **FA2 with bf16 q +
the fused qk-norm-rope-gate preamble**. And because both the fused-preamble
dispatch and the FA2 backend are `enforce_eager`-independent, the stored oracle —
dumped with `enforce_eager=True` (`tools/parity/dump_qwen36.py:242`) — ALREADY ran
vLLM's fused-preamble + FA2 bf16-q path.

## Empirical: production == the stored oracle (the linchpin)

Ran vLLM **0.25.0 PRODUCTION** (graphed: torch.compile + CUDA graphs, NOT
enforce_eager; `~/venvs/vllm-oracle` is now 0.25.0) on the pinned 35B checkpoint,
same prompt/sampling as the golden dump, only toggling `enforce_eager` off.
Evidence `dgx:/tmp/prod_greedy_35b.log` (script `/tmp/prod_greedy_35b.py`):

```
MODE PRODUCTION(graphed)   VLLM_VERSION 0.25.0
GREEDY_IDS [6511,314,9564,369,19241,13,198,760,6511,314,9338,369,11751,11,321,279]
ORACLE_IDS [6511,314,9564,369,19241,13,198,760,6511,314,9338,369,11751,11,321,279]
MATCHES_EAGER_ORACLE True
```

All 16 tokens IDENTICAL. So graphing + torch.compile + the 0.24.0→0.25.0 version
bump change NOTHING about the 35B greedy stream. **The stored oracle IS
production-faithful.** (Contrast the 27B, whose token-7 near-tie DOES differ
between vLLM graphed and eager — the 35B is robust at these positions.)

## ORACLE DECISION: DO NOT re-baseline

The oracle already matches vLLM production greedy tokens. There is no
production-vs-oracle discrepancy to re-baseline. The `IF` (re-baseline) branch of
the task is REFUTED; this is the `ELSE` branch (oracle faithful).

## Root-cause of our historical FA2 divergence + the empirical staleness finding

- Our fused preamble kernel (`AttnQkNormRopeGateKernel`, `cuda_ops.cu:781`;
  `GemmaNormElem`, `cuda_ops.cu:769-773`) keeps the normed q/k in **f32** and only
  rounds the FINAL RoPE result to bf16 (pinned by `test_ops_attn_preamble.cpp:245-315`:
  bf16 out == RN(f32 out)). It does **NOT** round the normed value to bf16 before
  RoPE. vLLM's fused kernel (and our own UNFUSED bf16 path via the `dq3`=bf16
  RmsNorm store, `qwen3_5.cpp:3272-3285`) DO. That missing bf16 round-trip is the
  ≤1-ULP source that can flip a razor-tie — the 2026-07-09 NO-GO cause.
- **EMPIRICAL (old build `~/work/vllm.cpp-35b-fix-clean/build-production`, FLASH_ATTN=ON,
  behind main):** with `VT_FUSE_ATTN_PREAMBLE=1 VT_FA2_PREFILL=1` the 35B gate
  `test_qwen36_paged_engine` passes **315/315** on BOTH cases (single + batched-graph),
  identical to the WMMA baseline (also 315/315). nsys CONFIRMS FA2 genuinely ran:
  `flash::flash_fwd_splitkv_kernel<...256,64,64,4...>` (10 launches = the 35B
  full-attn prefill layers) + `AttnQkNormRopeGateKernel<float,__nv_bfloat16,float>`
  (the bf16 fused preamble). So the 2026-07-09 divergence is **STALE** — resolved by
  the many intervening forward-numerics changes (the 35B forward is now closer to
  vLLM, so FA2 — vLLM's exact kernel with vLLM's exact bf16 q — agrees). FA2 is
  actually the MOST vLLM-faithful attention path (more than f32 WMMA).
- Attention prefill speedup (same old build, oracle prompt, per full-attn layer):
  WMMA `PagedFlashWmmaGqaFlash2VecBMKernel` 17.8 µs/layer (+ `PagedFlashBuildTiles`
  3.2 µs) → FA2 `flash_fwd_splitkv` 9.55 µs/layer (no tile-build) ≈ **1.86× on the
  attention kernel**. (Absolute TTFT gain not representative on a 9-token prompt;
  a realistic-prefill A/B is still owed.)

## LANDED (2026-07-18, `CLAIM-35B-FA2-FLIP-1`) — flip default-ON, UNTIGHTENED

The `ELSE` recommendation was executed on current main. Outcome: **the 35B
FA2-prefill + fused-preamble default is FLIPPED ON** (`FuseAttnPreambleOn` now
returns true for all arches; `VT_FUSE_ATTN_PREAMBLE=0`/`VT_FA2_PREFILL=0` roll
back). Both sacred gates hold on the FULL current-main default set (async +
GDN cubin + all fast kernels + this flip): 35B `test_qwen36_paged_engine`
**315/315** (both single + batched-graph cases), 27B `test_qwen27_paged_engine`
**235/235**. The kernel-side `fa2_prefill` (`cuda_paged_attn.cu:2494`) admits any
GQA ratio at head_dim 256, so the 35B ratio-8 layers take the exact
`flash_fwd_splitkv` kernel the 27B already uses.

### The tighten was TRIED and REJECTED (RMSNorm-saga near-tie flip)

Step 1's "round normed q/k to bf16 before RoPE" tighten (`RoundToStore<Tqk>` in
`AttnQkNormRopeGateKernel`) was implemented and **op-level VALIDATED bit-identical
to the unfused bf16 path** (`test_ops_attn_preamble` fused-bf16 == unfused-bf16:
q 0/32768, k 0/4096, gate 0) — exactly the vLLM `fused_qk_norm_rope.py:67`
numerics. BUT with the tighten, `test_qwen27_paged_engine` **failed 233/235**: it
flipped the 27B's known **tok6 whitespace near-tie** AWAY from the pip-vLLM oracle
(`greedy_ids.npy` → `greedy_ids_emulation.npy`). This is the RMSNorm-saga lesson —
an op that is individually MORE vLLM-faithful flips a razor near-tie in
COMBINATION, because our OTHER sub-ULP op diffs compensated the un-rounded
preamble to land on the vLLM-correct token. Per the guardrail (**do NOT ship a
divergence**), the tighten is NOT shipped. The 35B passes 315/315 with OR without
it, so the preamble ships UNTIGHTENED and BOTH arches stay token-exact on their
graphed oracles. The op test keeps its original (bf16 store == RN(f32)) contract;
the finding is recorded in-code (`AttnQkNormRopeGateKernel` + `FuseAttnPreambleOn`
NOTEs) so a future agent does not re-attempt the tighten blind.

### Evidence (this landing)
- 35B 315/315 + 27B 235/235 + op 14/14, full default set — `dgx:/tmp/fa2gates_u.log`
  (`~/work/vllm.cpp-35b-fa2-flip`, clean `-Werror` build, CUTLASS+FA2+Triton).
- memcheck 35B prefill `--tool memcheck` — `dgx:/tmp/fa2_memcheck2.log`.
- Realistic-prefill TTFT A/B (input-1024, conc8, FA2 on vs `VT_FA2_PREFILL=0`,
  3 interleaved pairs) — `dgx:/tmp/fa2_ttft2.log`: **Mean TTFT 824.7 vs 874.4 ms
  = −5.7%** (median −5.7%; prefill tput +5.5%). Below the ~7-9% offline-kernel
  target — the 1.86× attention-kernel win dilutes across the full prefill; no
  decode regression (prefill-only, both arms token-exact 315/315).

## Evidence roots

- `dgx:/tmp/prod_greedy_35b.log` — vLLM 0.25.0 graphed greedy == oracle (16/16).
- `dgx:/tmp/fa2chk.*`, `/tmp/wmmachk.*` — nsys kernel sums proving FA2 ran + timings.
- Old-build 315/315 both cases both arms (task-notification background runs).
- Stored oracle: `tests/parity/goldens/qwen36_logits_35b/greedy_ids.npy`
  (`pip-vllm:0.24.0`, `enforce_eager`), unchanged.
