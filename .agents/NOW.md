# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-06 -->

Read this FIRST, every session. A SNAPSHOT, rewritten in place: what is live,
the gate being chased, what to do next. Never a log — evidence lives in the
append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md) and the
benchmark record. Budget: 100 lines.

## Live claims

Working head: `origin/main`.

| Claim / track | State | Next command or step |
|---|---|---|
| Laguna NVFP4 decode speed | **Closed: PARITY+ 1.03x** (44.46 vs 43.10, byte-exact, default). Root cause = bf16 weight residency via `VT_LAGUNA_RESIDENT_BF16W` (default-ON). Detail in benchmark record | Residual: formal vLLM K-run set when convenient |
| DeepSeek-V4-Flash decode | **Closed: BEATS ds4 1.144x** (`VT_V4_RESIDENT_W` on, byte-exact). Phase-2 routed-expert residency NEGATIVE 2026-08-05 (−3.4%), HELD default-OFF. See state | — |
| f32-out GEMV audit | Only laguna + deepseek_v4 bf16 tower affected; gate models & on-framework dense unaffected (bf16-out, e2e-verified) | Re-verify deepseek_v4 bf16 tower same-tool |
| Invocation-parity prevention | CI guard (`check-gemv-invocation-consistency.py`) + AGENTS.md checklist landing | Review + merge; CUDA build-verify `kGemvHeuristicAlgos` on dgx |
| MiniMax-H3 lane | Portable path complete; e2e prompt-conditioned video on real weights (Thor). Speed = NVFP4 FP4 device path, sm_121-gated | PR #26 rebase + supports-audit synthesis (workflow ran; integrate) |
| Protocol substrate repair | BENCHMARKS.md converted to scoreboard (landed); STATUS.md budget + record-era roll still open | Items below |
| Kimi-Linear-48B (KDA+NoPE-MLA+MoE) | **W7 device COMPUTE landed, CPU-gated** (`CLAIM-KIMI-LINEAR-W7`, `ACTIVE`): DBuf-resident `ForwardDeviceCompute` (2 host islands: KDA recurrence, NoPE-MLA softmax); `test_kimi_linear_forward` **12/12·614**; opt-in `VT_KIMI_DEVICE_COMPUTE=1` | GPU-verify: CUDA build, token-exact vs oracle, e2e §8 |
| 35B fresh grid | **BOUND** @`1ea26427`: tput 0.93-1.03x, TTFT 0.93-0.98x, c1 closed, c16 0.93x. c16 drain-sync lever **A/B'd NEGATIVE 2026-08-06** (blocking-event −1.9%; drain KEPT). Cost = depth-2 serialization, not the drain (see state) | Real lever = GPU-resident sampled tokens (W4 `VT_ASYNC_DEVICE_MIRROR` on integrated); then TTFT |

In-flight branches (gated default-OFF, not pushed): `laguna-fp4proj-prod`
(fp4 opt-in), laguna bf16/legacy/pipeline-gemv, `ds4-hc-expand-fuse`.

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).

Method rules hardened this cycle (AGENTS.md): the STRUCTURAL lens (same kernel,
different throughput ⇒ audit the context; scan the REFERENCE's own rationale —
vLLM or ds4/SGLang/llama.cpp — as a default lane; the scan GENERATES hypotheses,
per-shape MEASUREMENT arbitrates; distrust aggregate bytes/time and CROSS-TOOL
comparisons — the Laguna "ceiling" was a cross-tool artifact, twice).

## Next actions

1. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
3. **Same-tool re-verify deepseek_v4's bf16 resident tower** (the one other
   f32-out caller) once the Laguna fix proves the mechanism.
4. **Restore `local-ai-worker`** on dgx when the GPU campaign ends
   (`docker update --restart=always` + `docker start`).
5. **Protocol substrate — partly done.** Claim triage DONE (54 rows out of
   `ACTIVE`, 29 claims retired); `docs/STATUS.md` now under a shrink-only
   ratchet; roadmap compacted; `AGENTS.md` tiered. REMAINING: (a) anchor
   backfill, 98 rows still `SPIKE`/`ACTIVE` (backend 36, engine 26, kernel 16,
   model 16, quant 4) because every honest destination needs code/test anchors
   they lack — root cause is that the lifecycle has no zero-cost parking state
   for a landed-but-unowned row; 6 model rows need a DECISION, not an anchor
   (architecture unregistered). (b) The record-era rollover is BLOCKED:
   `check-agent-record.py` binds `DONE` rows to exact LINE anchors in
   `parity-ledger.md` (43 references), so freezing it invalidates the evidence
   graph — re-anchor by ledger ROW ID first. `state.md` and
   `benchmark-record.md` have no line anchors and can roll now.

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
