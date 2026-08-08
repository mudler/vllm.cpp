# NOW — the one-Read resume surface

<!-- now-updated: 2026-08-09 -->

Read this FIRST, every session. A SNAPSHOT, rewritten in place: what is live,
the gate being chased, what to do next. Never a log — evidence lives in the
append-only [state.md](state.md), [parity-ledger.md](parity-ledger.md) and the
benchmark record. Budget: 100 lines.

## Live claims

Work: exact-chunks on main `1ce0d662b`; sm_120 measured at `3d2581551`.

| Claim / track | State | Next command or step |
|---|---|---|
| Laguna NVFP4 / DeepSeek-V4 decode | **Both CLOSED, byte-exact, default-ON**: 1.03x vLLM, 1.144x ds4 | Laguna vLLM K-run when convenient |
| f32-out GEMV audit | Only laguna + ds4 bf16 tower affected; gate models unaffected | Re-verify ds4 tower same-tool |
| Invocation-parity prevention | CI guard + checklist landing | Merge; build-verify `kGemvHeuristicAlgos` on dgx |
| MiniMax-H3 lane | **bf16 shards STREAM both towers (DiT + encoder); Q4_K_M enc cond cos 0.9975, 3.5° med, DIFFUSE** | render A/B on saved embeds |
| Kimi-Linear-48B | **ROW 7 fold LANDS (#122 §21): engine==CLI 128/128; golden 122/128; SACRED green; v13 tokens ABI** | ACTIVE: 19.0 tok/s vs vLLM ~21 (~0.90×) |
| 35B fresh grid | **BOUND** @`1ea26427`: 0.93-1.03x, c16 0.93x. INTAKE + Option A both NEGATIVE | Lever left: prefill glue (#61) |
| Qwen3.5-4B sm_120 | Exact chunks ON: rebased-main reprofile 3.072x kernel / +2.272% run; sealed-vLLM throughput 1.021x PASS. Latency/VRAM OPEN | Spike residual 1.609x conv gap |
| RPi5 A76 CPU | **R5 asm GREEN; llama NOT MET**: 0.461x pf, 0.653x dec, RSS -24% | W6: BF16 GEMM |
| MXFP4 parity | c1 1.020, c2-c8 0.962-0.969. **#82 CLOSED: ptxas-lineage REFUTED (A/B ties our+vLLM PTX all ptxas/JIT; +10us=engine context, not codegen)** | TERMINAL: at parity |
| ROW-SERVE-ASYNC-DENSE-MIRROR | **LANDED+dgx-VERIFIED** (`f9c969ae`): async mirror on classic dense Qwen3; SACRED 184/184 | Residual: sibling scope one-liner |
| CPU levers (`QUANT-GGUF-CIQ-GEMM`) | Profile DONE: decode **47% threadpool sync**, prefill **~39% paged attn**. **G5 not next** | Parakeet encoder; attn dtype hoist |
| Supported-models list | **LANDED**: FEATURES arch table CI-bound (33 archs) | — |
| `/v1/videos` OpenAI shape | **MERGED** (#71): Sora `model`/`size`/`seconds` + `GET /{id}/content` | `row/SERVE-VIDEOS-REFS` PR open: reference conditioning |
| Vulkan 27B decode | **4.285 vs 4.35**. Host ~3 ms/tok not 5.3; pipelined submit -1.41 ms/tok 5/5; spin REJECTED | NOT met. Next: 900 barriers/tok |
| `BACKEND-ROCM` | **(b) fix in; #140 gfx1201 hipBLAS + Gemma-4 MoE landed (contributor, authorship-preserved); W0 green 4 archs** | compile + M2 ([spec](specs/rocm-unified-memory-b.md)) |
| TP spike #287 (PR #143) | **TP-W1 LANDED**: rank-group table + TP handle (6/6); DSR leak FIXED (unblocks #127/#154/#155) | TP-W2 (linears + loader) |
| Release | **ACTIVE; W5 19/19+10/10; contract 30/30** | #141; artifacts pending |
| Surface coverage (`ARCH-ONE-SURFACE`) | ROW 8 + #139 IN; **ROW 6 LANDED (#137): embeddings LIVE — `LlamaModel` arch, PoolingRunner in the step, `vllm_embed` v15, `/v1/embeddings`, fold gate 4/4-231, 9 kills** | Real-checkpoint oracle cosine residual |

In-flight, default-OFF, not pushed: see state.md.

## Current gate

Unchanged: token-exact (or the ratified distributional gate) against the pinned
vLLM oracle, AND ≥ vLLM on every throughput axis / ≤ on latency and memory, on
both gate models, reproduced 2–3x on an idle box. See the
[verification procedure](verification.md). Parity pin: vLLM `555967922`
(0.26.0.dev0).


## Next actions

0. **`ROAD-V1-MEM`** KV auto-sizing spike LANDED (`specs/kv-sizing.md`, `READY`).
1. **Spike the Parakeet encoder row** (vLLM carries it inside
   `nano_nemotron_vl.py`; the transducer half is NOT in vLLM: separate call).
2. **Qwen3.5-4B sm_120:** rebased branch is GREEN and reprofiled. Spike the
   residual 1.609x conv gap; latency/VRAM and gate models stay open.
2. **Merge the invocation-parity prevention** (CI guard + AGENTS.md checklist);
   CUDA build-verify the byte-exact `kGemvHeuristicAlgos` refactor on dgx.
3. **Same-tool re-verify deepseek_v4's bf16 resident tower** (the one other
   f32-out caller) once the Laguna fix proves the mechanism.
4. **Restore `local-ai-worker`** on dgx at campaign end (`--restart=always`).
5. **Protocol substrate — partly done.** Triage/audit + `STATUS.md` ratchet +
   `AGENTS.md` tiering DONE. REMAINING: anchor backfill (6 model rows need a
   DECISION); record-era rollover BLOCKED on `DONE` rows bound to
   `parity-ledger.md` LINE anchors (re-anchor by ROW ID).

**Operator/helper protocol**
([spec](workflow.md)): roles are a lock or worktree+PR; operator delegates,
helpers claim `row/<ROW-ID>` with a start-time DRAFT PR. **W0-W5 LANDED**;
role/entrypoint gates ENFORCE `agent-start.py` → claim → preflight. Review FAIL
loops through fresh implementer, both gates, and fresh review until PASS; budgets
cannot stop it. Queue: 10 rows; backfill 79 rows, 30 anchored.
**Upstream inventory** ([spec](specs/upstream-derived-inventory-2026-08-05.md),
drift-gated, arch parity BOTH ways): SM060/061/070 below vLLM's floor =
OUT-OF-SCOPE; COMP-*/DISTRIBUTED-* are REAL unported work; **all 362 archs have
rows**; llama.cpp's 11 extra devices IN SCOPE, spike-gated (`ROAD-V1-D6`).

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
