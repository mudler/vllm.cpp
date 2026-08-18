# Roadmap-v1 reality audit — code-grounded status + gaps + e2e-proof (2026-07-31)

USER-directed. An 8-lane parallel audit + adversarial DONE-verify pass (26 agents)
that VERIFIED the plan-of-record (`roadmap-v1-completion.md`, 18 rows =
DONE 4 / RI 13 / HW 1) against the actual `src/` + `tests/` + matrices, not the
ledger. Expanded to 24 sub-rows. **4 DONE claims were refuted.** `CLAIM-ROADMAP-V1-AUDIT`.

## Bottom line
The roadmap is REAL on **correctness + CPU-gated features**, but the completion
ledger over-states it in specific places. The single dominant open gate across
nearly every RI row is **every-axis SPEED** (correctness landed, speed pending).
Most GPU "DONE" proofs are **DGX-recorded, not re-verifiable on the current box** —
they rest on the ledger record of past GB10 runs.

## Genuinely DONE + e2e-proven, CPU-runnable NOW (re-verifiable)
- **C1 extensibility cornerstone** — op-fusion framework + consistency CI check; `check-fusion-consistency.py` RC=0 re-verified live. DONE-PROVEN.
- **C7 sampling / C8 tokenize-parse-metrics** — full sampling-control surface + streaming/assembly parser engine + `/metrics` Prometheus + utility endpoints, CPU-gated (`test_parser_engine_assembly` 5038/5038, `test_prometheus_metrics`, `test_openai_api_server`). Runnable now.
- **C4 GGUF byte-exactness** + **C5 RoPE/sliding op-level** — CPU-checkable, re-verified green in-audit.

## DONE but proof is DGX-RECORDED (rests on ledger, not re-runnable here)
- **MM image** (Qwen3-VL-4B + 27B) — STRICT 32/32, dgx-recorded. Genuinely strict.
- **C3 spec-decode** (MTP k=1 + DFlash) — correctness+speed gates dgx-recorded.
- **D3 ngram** — exactness-preserving (no separate speed gate owed), dgx-recorded.
- **C4 3 quant schemes** (NVFP4-W4A16/W4A4, FP8-static) — dgx-recorded.
- **C9 pin advance** (0.26.0.dev0) — re-gate 296/299 GREEN on dgx, pushed. DONE (recurring row).

## OVER-CLAIMED — the 4 adversarial refutations (ledger says DONE, code/record says NOT)
1. **MM video** — recorded as "STRICT 32/32" but the actual gate is a **near-tie distributional** gate, NOT strict. Correct the record.
2. **Gemma-4 image** — ledger implies STRICT 18/18; the dgx record is **16/18 near-tie** (2 bf16-near-tie punctuation diffs); **audio e2e is UNBUILT**. Row is RI/EXT, not DONE.
3. **ROAD-V1-A perf** — `ENG-ASYNC-SCHED` code is DONE-PROVEN, but the ROW is NOT: both SGLang-floor arms (`BACKEND-GATE-CUDA-SGLANG` BLOCKED on `SERVE-ASYNC-LLM` prod-ON; `-PREFIX` READY) have **never produced a binding run** (task #137 still pending); 27B is **114/124 "effective parity" via coin-flip totality**, not a clean 124/124; 35B c1/c2 low-conc still fail.
4. **C6 async/priority serving** — DONE marking not genuine: `SERVE-ASYNC-LLM` is **GATING** (engine-matrix.md:189), not prod-ON — the explicit blocker for the SGLang arm.
   > **SUPERSEDED 2026-08-12 ([#534](https://github.com/mudler/vllm.cpp/issues/534)) — kept as the audit said it, do not act on it.** "Not prod-ON" reads a `GATING` lifecycle state as a production default. `SERVE-ASYNC-LLM` IS the production serving path (`src/vllm/entrypoints/openai/server_main.cpp:731-734`); it is `GATING` on broader every-axis parity, and the runner-side default that the punch-list conflates with it (`runner_supports_async`) has been ON since `a0013a2` (2026-07-17). The C6 row's one genuinely open leaf is `ENG-PRIORITY-SCHED`, whose priority-vs-FCFS gate does not exist yet. Item 3's SGLang clause carries the same misreading.
   Plus **D5 LoRA** (RI, not refuted-from-DONE but flagged): the headline LoRA is only a **standalone unwired CPU float-math brick** (`src/vllm/lora/punica_cpu.cpp` + `test_punica_cpu`) — no engine/runtime/endpoint integration. Most of D5 is unimplemented.

## Concrete debt / staleness found (fixable)
- **STALE MATRIX:** `backend-matrix.md:233` (`BACKEND-GATE-CUDA-VLLM`) still records "NEW BINDING 246a23c: 49/124" while `roadmap_v1.md:494` has superseded it to **114/124 (9ecd9d0)** — the matrix lags the roadmap by two bindings. Fix.
- **C9 debt:** §2D mechanical upstream re-sync explicitly DEFERRED (rmsnorm-fusion #46998, ReplaySSM #48018, MoeWNA16→MK #44120, olmo3.py not landed); oracle provenance DEGRADED (pinned source tree `~/work/vllm-src-5559679` pruned → clean re-capture not reproducible).
- **C2:** ~20 model families correctness-complete + self-registering, but **all 20 are speed-pending** (every-axis SPEED open on each).

## Correctly BLOCKED (bounds "complete roadmap_v1")
- **D2 multi-GPU/TP** — needs ≥2 GPUs (GB10 is single). NCCL code exists, never run.
- **D1 backend runtimes** — ROCm/XPU/discrete-Vulkan/ANE (no boards); CUDA arch bodies beyond sm_120/121 build-only.
- **Frontier models >119 GiB** (DeepSeek-V3/GLM-5/Kimi-K2/MiniMax); **Gemma-4/Command-R** (HF-gated, no dgx token); **DSA models** (dep-blocked).

## Honest completion picture
- **Truly complete + e2e-proven (any hardware):** the extensibility cornerstone + the CPU-gated serving/sampling/parse/metrics surface + GGUF byte-exactness.
- **Complete-on-GB10 but proof is historical:** image mm, spec-decode, 3 quant schemes, prefix-caching code, pin.
- **The big remaining reachable work:** (1) **every-axis SPEED** on ~20 models + all mm rows + 35B; (2) **arm the SGLang floor** (needs `SERVE-ASYNC-LLM` prod-ON); (3) **real LoRA** runtime (D5); (4) re-verify the DGX-recorded gates on a clean current build.
- **Hard bound (not this-box completable):** D2 multi-GPU, non-CUDA backends, frontier >119 GiB, HF-gated/DSA models.
