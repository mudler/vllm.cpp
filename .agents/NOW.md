# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-11 -->

Snapshot, not log. History is git; evidence:
[parity ledger](parity-ledger.md), and benchmarks. Budget: 100 lines / 6,000
characters.

## Live claims

Rendered on demand: **`scripts/now.py`**. It assembles every `SPIKE`/`ACTIVE`
row, its claim and PR, and the row's own next step from the matrices,
`.agents/claims/`, and each row spec's `## Now`.

They are NOT listed here any more (ENG-NOW-DERIVED, #374). A per-row table in
this file made it a surface every row-advancing PR had to write, which is a lock
under `AGENTS.md` §Records; it conflicted in 5 of the 16 conflicting open PRs
measured at `d928e2c3`. What remains below is authored at operator cadence, so
no per-row change needs to touch this file at all.

## Current gate

Token-exact (or ratified distributional) vs pinned vLLM; ≥ throughput and ≤
latency/memory on every axis, both gate models, reproduced 2–3x idle. See
[verification](verification.md). Pin: vLLM `555967922` (0.26.0.dev0).


## Next actions

0. **35B mid-band: first lever LANDED** (+1.31% c8, +1.38% c4). The fused
   shared gate_up sink still took the MoE-marlin route (20320 launches = 5.4%
   GPU); `VT_MARLIN_DENSE_PAIR` ON. Second lever LANDED: shared down-proj emits
   bf16, **+2.05% BIT-IDENTICAL**. SiLU [spec](specs/moe-silu-vectorize.md)
   **NEGATIVE**: the 9.2x was a MEAN over a bimodal kernel (min 1.34/max 979us);
   decode SiLU already beats vLLM's. ~5% UNATTRIBUTED; needs decode-only, 1 tool.
1. **27B NVFP4 0.72x -> 0.85x** (FP8 tower native). Next: NVFP4 MLP marlin, 68%
   of roof. Dense-marlin +0.5%; Triton-AOT GDN a WASH.
2. **Spike the Parakeet encoder row** (vLLM: `nano_nemotron_vl.py`; the
   transducer half is NOT in vLLM: separate call).
3. **Qwen3.5-4B #206:** +2.83% `PENDING`; latency/VRAM open.
4. **Invocation-parity prevention:** CI guard + checklist; build-verify
   `kGemvHeuristicAlgos` on dgx.
5. **Restore `local-ai-worker`** on dgx at campaign end (`--restart=always`).
6. **Protocol substrate — partly done.** Triage/audit + `STATUS.md` ratchet +
   `AGENTS.md` tiering DONE. REMAINING: anchor backfill (6 model rows need a
   DECISION); record-era rollover BLOCKED on `DONE` rows bound to
   `parity-ledger.md` LINE anchors (re-anchor by ROW ID).

**Operator/helper protocol** ([spec](workflow.md)): roles are a coordinator
record or worktree+PR; helpers claim `row/<ROW-ID>` with a DRAFT PR. Role gates
ENFORCE `agent-start.py` → claim → preflight. Review FAIL loops to a fresh
implementer until PASS. Queue: 10 rows; backfill 79, 30 anchored.
**Upstream inventory** ([spec](specs/upstream-derived-inventory-2026-08-05.md)):
SM060/061/070 below vLLM's floor = OUT-OF-SCOPE; COMP-*/DISTRIBUTED-* are REAL
unported work; **every arch in the pinned registry has a row** (the count lives in
[model-matrix.md](model-matrix.md), not here — #622); llama.cpp's 11 extra
devices IN SCOPE (`ROAD-V1-D6`).

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
- Work happens in its own worktree on a task branch; the shared checkout stays
  clean on `main`, never a work surface. Land via `row/*` PR or authorized
  local merge; remove the worktree after.
