# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-07 -->

Read this FIRST, every session. A SNAPSHOT, rewritten in place: what is live,
the gate being chased, what to do next. Never a log — evidence lives in the
append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md) and the
benchmark record. Budget: 100 lines.

## Live claims

Working head: `row/backend-rocm-w0` (#41). Prior: benchmark checkpoint
`bench/qwen35-upstream-rebenchmark-20260805` on `upstream/main` @ `59674cf1d`.

| Claim / track | State | Next command or step |
|---|---|---|
| Laguna NVFP4 / DeepSeek-V4 decode | **Both CLOSED, byte-exact, default-ON**: 1.03x vLLM, 1.144x ds4 | Laguna vLLM K-run when convenient |
| f32-out GEMV audit | Only laguna + ds4 bf16 tower affected; gate models unaffected | Re-verify ds4 tower same-tool |
| Invocation-parity prevention | CI guard + checklist landing | Merge; build-verify `kGemvHeuristicAlgos` on dgx |
| MiniMax-H3 lane | **fl2va COHERENT; ref2va grid DIAGNOSED (#95): NO loader bug; bf16 13-shard DiT STREAMS** | residual = community-NVFP4 quant fidelity §8.12; no bf16 render yet |
| Kimi-Linear-48B (KDA/NoPE-MLA/MoE) | device-KDA **122/128, 4.24 tok/s** best (§15); MLA device NEG (§16). chunk_kda prefill AOT **SPIKED**: 5 kernels authored+pinned+recipe (§17). Bar = MEET vLLM speed | Phase-2: regen harness, wire `vt::KdaChunkPrefill`, gate STRICT + vLLM 0.82 ladder |
| 35B fresh grid | **BOUND** @`1ea26427`: 0.93-1.03x, c16 0.93x. INTAKE + Option A both NEGATIVE | Lever left: prefill glue (#61) |
| Qwen3.5-4B revalidation | 0.9971x @`59674cf1` (#35); TTFT/PSS pass, TPOT/ITL open | `docs/bench-evidence/` |
| MXFP4 parity | c1 1.020, c2-c8 0.962-0.969. **#82 CLOSED: ptxas-lineage REFUTED (A/B ties our+vLLM PTX all ptxas/JIT; +10us=engine context, not codegen)** | TERMINAL: at parity |
| ROW-SERVE-ASYNC-DENSE-MIRROR | **LANDED+dgx-VERIFIED** (`f9c969ae`): async mirror on classic dense Qwen3; SACRED 184/184 | Residual: sibling scope one-liner |
| CPU levers (`QUANT-GGUF-CIQ-GEMM`) | Profile DONE: decode **47% threadpool sync**, prefill **~39% paged attn**. **G5 not next** | Parakeet encoder; attn dtype hoist |
| Supported-models list (`row/DOCS-SUPPORTED-MODELS-MATRIX`) | **DRAFT PR**: FEATURES per-arch table CI-bound to registry (30 archs) | Reviewer merge |
| `/v1/videos` OpenAI shape | **MERGED** (#71): Sora `model`/`size`/`seconds` + `GET /{id}/content` | `row/SERVE-VIDEOS-REFS` PR open: reference conditioning |
| `BACKEND-ROCM` W0 | Skeleton in; **HIP never compiled** (no AMD HW) | #41 contributors build it; a compile error IS the deliverable |

In-flight (default-OFF, not pushed): `laguna-fp4proj-prod`, laguna
bf16/legacy/pipeline-gemv, `ds4-hc-expand-fuse`.

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See [gates.md](gates.md) and
[benchmark-protocol.md](benchmark-protocol.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).

Method rules hardened (AGENTS.md): the STRUCTURAL lens (same kernel, different
throughput ⇒ audit the context; per-shape MEASUREMENT arbitrates).

## Next actions

1. **Spike the Parakeet encoder row.** Upstream vLLM has `parakeet.py` +
   `conformer_encoder.py` as the audio encoder of `nano_nemotron_vl.py`, which we
   already carry `MODEL-MM-nano-nemotron-vl-*` rows for, so it is owed mirror work.
   The transducer decode half (RNN-T/TDT/CTC) is NOT in vLLM: separate scope call.
2. **Qwen3.5-4B serving follow-up:** bind the default-ON async-serving path
   against the same oracle before attributing the remaining TPOT gap.
2. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
3. **Same-tool re-verify deepseek_v4's bf16 resident tower** (the one other
   f32-out caller) once the Laguna fix proves the mechanism.
4. **Restore `local-ai-worker`** on dgx when the GPU campaign ends
   (`docker update --restart=always` + `docker start`).
5. **Protocol substrate — partly done.** Triage/audit, `STATUS.md` ratchet and
   the `AGENTS.md` tiering are DONE. REMAINING: anchor backfill (6 model rows
   need a DECISION); record-era rollover BLOCKED on `DONE` rows bound to
   `parity-ledger.md` LINE anchors (re-anchor by ROW ID). ★ The gate SELF-BLINDS
   on the 10 audited rows (audit §➁a); its fix owes an 8-row adjudication.

**Operator/helper protocol**
([spec](specs/operator-helper-protocol.md)): roles DECLARED then MATERIALIZED
into a lock or worktree+PR; operator merges PRs first and does features only via
sub-agents; helpers use worktrees on `row/<ROW-ID>` and open a DRAFT PR at the
START, which IS the claim. **W0-W5 LANDED**; role discipline ENFORCING,
`--require-role` is the DEFAULT. Queue: 10 rows; backfill 79 rows, 30 anchored.
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
