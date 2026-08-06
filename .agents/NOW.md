# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-06 -->

Read this FIRST, every session. A SNAPSHOT, rewritten in place: what is live,
the gate being chased, what to do next. Never a log — evidence lives in the
append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md) and the
benchmark record. Budget: 100 lines.

## Live claims

Working head: `bench/qwen35-upstream-rebenchmark-20260805`, one benchmark
checkpoint on `upstream/main` at `59674cf1d`.

| Claim / track | State | Next command or step |
|---|---|---|
| Laguna NVFP4 decode speed | **Closed: PARITY+ 1.03x** (byte-exact, default; `VT_LAGUNA_RESIDENT_BF16W` bf16-residency). Benchmark record | vLLM K-run set when convenient |
| DeepSeek-V4-Flash decode | **Closed: BEATS ds4 1.144x** (`VT_V4_RESIDENT_W`, byte-exact). Phase-2 routed-expert residency NEGATIVE (−3.4%), default-OFF | — |
| f32-out GEMV audit | Only laguna + deepseek_v4 bf16 tower affected; gate/on-framework dense unaffected | Re-verify deepseek_v4 tower same-tool |
| Invocation-parity prevention | CI guard (`check-gemv-invocation-consistency.py`) + AGENTS.md checklist landing | Review + merge; CUDA build-verify `kGemvHeuristicAlgos` on dgx |
| MiniMax-H3 lane | **W-FP4a fp4 routing CPU-landed** (`row/H3-FP4-SPEED`): NVFP4 kept packed -> Marlin W4A16, no new quant code; gate 62/62 | GB10 CUDA gate; real e2e disk-blocked |
| Kimi-Linear-48B (KDA+NoPE-MLA+MoE) | **Full-model GB10 e2e RUNS** (bf16-resident §13): CPU+CUDA 13/13·656, no OOM. **Token gate NEAR-TIE 106/128** (6/8 token-exact) | device GDN/MLA islands + bf16 stream; 1.59 tok/s; default OFF |
| 35B fresh grid | **BOUND** @`1ea26427`: tput 0.93-1.03x, c16 0.93x. INTAKE + Option A both **RESOLVED NEGATIVE** (H2D-out-of-capture tput WASH) | Real lever left: prefill glue (task #61) |
| Qwen3.5-4B revalidation | 0.9971x @`59674cf1` (#35); TTFT/PSS pass, TPOT/ITL open | `docs/bench-evidence/` |
| RPi5 A76 CPU | **R5 ASSEMBLY GREEN:** output-exact AAPCS64 beats compiler SDOT 3.66-5.08%; Qwen TTFT/E2E improve | W6: M1/T4 −2.43%, BF16 GEMM, memory/concurrency and llama.cpp floor |
| MXFP4 parity | **TERMINAL:** c1 1.020x pass; c2-c8 0.962-0.969 GPU-intrinsic; final cap/glue levers exhausted | Record; branch unmerged |
| ROW-SERVE-ASYNC-DENSE-MIRROR | **LANDED+dgx-VERIFIED** (`f9c969ae`): #31 async mirror on classic dense Qwen3; gate RED→GREEN, SACRED 184/184 | Residual: sibling scope one-liner |

In-flight branches (default-OFF, not pushed): `laguna-fp4proj-prod` (fp4),
laguna bf16/legacy/pipeline-gemv, `ds4-hc-expand-fuse`. Records:
`row/SPEC-DECODE-INVENTORY`: 13 vLLM spec-decode methods enumerated from
source, 9 gaps `INVENTORIED` ([inventory](specs/spec-decode-inventory.md)).

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).

Method rules hardened (AGENTS.md): the STRUCTURAL lens (same kernel, different
throughput ⇒ audit the context; per-shape MEASUREMENT arbitrates).

## Next actions

1. **Qwen3.5-4B serving follow-up:** the synchronous 0.9971x harness remains
   speed-pending; bind the default-ON async-serving path against the same oracle
   before attributing the remaining TPOT gap.
2. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
3. **Same-tool re-verify deepseek_v4's bf16 resident tower** (the one other
   f32-out caller) once the Laguna fix proves the mechanism.
4. **Restore `local-ai-worker`** on dgx when the GPU campaign ends
   (`docker update --restart=always` + `docker start`).
5. **Protocol substrate — partly done.** Claim triage + live-state audit DONE
   (10 unevidenced rows → `READY`, 11 claims retired, 9 amended); `STATUS.md`
   ratcheted; `AGENTS.md` tiered. REMAINING: anchor backfill
   (6 model rows need a DECISION); record-era rollover BLOCKED on `DONE` rows
   bound to `parity-ledger.md` LINE anchors (re-anchor by ROW ID).
   ★ The gate SELF-BLINDS on those same 10 (audit §➁a); its fix owes an 8-row
   adjudication. workflow.md states the `ACTIVE` precondition.

**Operator/helper protocol**
([spec](specs/operator-helper-protocol.md)): roles DECLARED then MATERIALIZED
into a lock or worktree+PR; operator merges PRs first and does features only via
sub-agents; helpers use worktrees on `row/<ROW-ID>` and open a DRAFT PR at the
START, which IS the claim. **W0-W5 LANDED**; role discipline ENFORCING,
`--require-role` still opt-in. Queue: 10 rows — 6 are audit-vacated, with LANDED gate anchors; READ before picking. Backfill: 79 rows, 30 anchored; blocker is claim FAMILIES.
**Upstream inventory** ([spec](specs/upstream-derived-inventory-2026-08-05.md),
drift-gated, arch parity BOTH ways): SM060/061/070 below vLLM's floor =
OUT-OF-SCOPE; COMP-*/DISTRIBUTED-* are REAL unported work; **all 362 archs now have rows**; llama.cpp's 11 extra devices are IN SCOPE, spike-gated
(`ROAD-V1-D6`).

## Protocol invariants that bite most often

- Every commit carries `FOLLOWING_AGENTS_PROTOCOL` + `Assisted-by:`; never
  `Co-Authored-By` or `Signed-off-by` from AI.
- Three MUST-route seams: fusion, merged-GEMM, born-on-the-runner decode.
  Not routing is drift; allowlist consciously or fold.
- Mirror vLLM; never ask how a feature should behave.
- `nsys` BOTH sides, SAME tool, before any perf claim; cross-tool comparisons
  never establish invocation parity; whole-run sums mix prefill.
- GPU: park `local-ai-worker`, flock `$HOME/gpu.lock`, single-load
  steady-state, never reload per rep, named tmux.
- Never weaken a checker to pass; repair the record.
- Feature code needs a `row/*` PR (enforced); integration paths push direct.
