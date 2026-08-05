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
| Laguna NVFP4 decode speed | **Closed: PARITY+ 1.03x** (44.46 vs 43.10, byte-exact, default). Root cause = bf16 weight residency via `VT_LAGUNA_RESIDENT_BF16W` (default-ON). Detail in benchmark record | Residual: formal vLLM K-run set when convenient |
| DeepSeek-V4-Flash decode | **Closed: BEATS ds4 1.144x** (`VT_V4_RESIDENT_W` on, byte-exact). Phase-2 routed-expert residency NEGATIVE 2026-08-05 (−3.4%), HELD default-OFF. See state | — |
| f32-out GEMV audit | Only laguna + deepseek_v4 bf16 tower affected; gate models & on-framework dense unaffected (bf16-out, e2e-verified) | Re-verify deepseek_v4 bf16 tower same-tool |
| Invocation-parity prevention | CI guard (`check-gemv-invocation-consistency.py`) + AGENTS.md checklist landing | Review + merge; CUDA build-verify `kGemvHeuristicAlgos` on dgx |
| MiniMax-H3 lane | Portable path complete; e2e prompt-conditioned video on real weights (Thor). Speed = NVFP4 FP4 device path, sm_121-gated | PR #26 rebase + supports-audit synthesis (workflow ran; integrate) |
| Kimi-Linear-48B (KDA+NoPE-MLA+MoE) | **W7 device COMPUTE landed, CPU-gated** (`CLAIM-KIMI-LINEAR-W7`, `ACTIVE`): DBuf-resident `ForwardDeviceCompute` (2 host islands: KDA recurrence, NoPE-MLA softmax); `test_kimi_linear_forward` **12/12·614**; opt-in `VT_KIMI_DEVICE_COMPUTE=1` | GPU-verify: CUDA build, token-exact vs oracle, e2e §8 |
| 35B fresh grid | **BOUND** @`1ea26427`: tput 0.93-1.03x, c16 0.93x. `VT_ASYNC_DEVICE_MIRROR` (drain MOVE) c16 NEUTRAL 0.999x — now **DEFAULT ON for CORRECTNESS** (below). Real c16 fix still needs drain-removal + double-buffer | — |
| Async-serving bug **FIXED** (`ROW-SERVE-ASYNC-LLM`) | Device-write/graph-host-read race caused batch-1 token-0 garbage. Mirror default ON; async gate RED(=0)→GREEN(default), SACRED/UAF clean, c16 neutral | Push PR #31 |
| Qwen3.5-4B revalidation | **Speed-pending:** local 0.999971x prior; pinned-oracle ratio 0.9971x. TTFT/PSS pass; TPOT/ITL/VRAM open; 18/18 legs idle | Publish branch checkpoint; evidence in `docs/bench-evidence/` |

In-flight branches (gated default-OFF, not pushed): `laguna-fp4proj-prod`
(fp4 opt-in), laguna bf16/legacy/pipeline-gemv, `ds4-hc-expand-fuse`.

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).

Method rules hardened (AGENTS.md): the STRUCTURAL lens (same kernel, different
throughput ⇒ audit the context; scan the REFERENCE's own rationale; per-shape
MEASUREMENT arbitrates; distrust aggregate bytes/time and CROSS-TOOL comparisons).

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
5. **Protocol substrate — partly done.** Claim triage DONE; `docs/STATUS.md`
   under a shrink-only ratchet; roadmap compacted; `AGENTS.md` tiered. REMAINING:
   anchor backfill (98 rows `SPIKE`/`ACTIVE`, need code/test anchors; 6 model rows
   need a DECISION, architecture unregistered); record-era rollover BLOCKED on
   `check-agent-record.py` binding `DONE` rows to `parity-ledger.md` LINE anchors
   (re-anchor by ROW ID first; `state.md`/`benchmark-record.md` can roll now).

**Operator/helper protocol**
([spec](specs/operator-helper-protocol.md)): roles DECLARED then MATERIALIZED
into a lock or worktree+PR; operator merges PRs first and does features only via
sub-agents; helpers use worktrees on `row/<ROW-ID>` and open a DRAFT PR at the
START, which IS the claim. **W0-W5 LANDED**; role discipline ENFORCING,
`--require-role` still opt-in. Queue: 4 rows. Backfill: 79 rows, 30 anchored; blocker is claim FAMILIES.
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
