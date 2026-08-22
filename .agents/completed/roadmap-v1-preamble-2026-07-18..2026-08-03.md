# Roadmap v1 preamble - archived chronology (2026-07-18 .. 2026-08-03)

The run-by-run narrative that had accumulated above the roadmap's portfolio
table, moved here verbatim on 2026-08-04. AGENTS.md is explicit that run-by-run
chronology must not live in the roadmap; 484 lines of it had grown there anyway,
pushing the actual table to line 485 so the document could not answer "what
next" without reading a superseded story first.

Nothing here is load-bearing for live decisions. The current position is
[NOW.md](../NOW.md), the binding numbers are
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md), per-capability state is
[docs/STATUS.md](../../README.md#project-status), and the ordered portfolio is
[roadmap_v1.md](../roadmap_v1.md). Kept for audit: evidence is moved, never
deleted.

---

*(user-directed 2026-07-10: this document is the live roadmap; the completed
M0–M3 record is archived at
[completed/roadmap_mvp_v0.md](../completed/roadmap_mvp_v0.md).)*

**Context (2026-07-18):** both gate models run end-to-end and retain token-exact
greedy correctness. **27B has reached effective performance PARITY-OR-BETTER with
vLLM v0.25.0** — two-grid totality (`9ecd9d0` 114/124 + `f0fb727` 111/124) gives
110 axes pass-in-both + 5 noise-band coin-flips = **115/124 effective**, and the 9
persistent residuals are all the low-concurrency-median edge of our
deterministic-forward tradeoff where we are NET-POSITIVE (we win the corresponding
tails + high-concurrency + throughput; async-forward jitter to "win" them is
net-negative). No closeable real deficit remains; details in state/ledger +
`.agents/specs/c8-p99-itl-tail-2026-07-18.md`. 35B performance closure follows the
accepted 27B verdict, then the operational + T1/T2 portfolio (orders 1–14) opens.
The historical open-gap framing below is retained for the SGLang floor and 35B.

The prior "exact performance closure is open" framing against vLLM v0.25.0; the
**new binding 27B result `9ecd9d0`** (fresh interleaved exact-grid rerun on the
full production default set — async + vendored Triton GDN decode cubin +
bit-identical fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU) reached
**115/124 effective parity** by two-grid totality (`9ecd9d0` 114 + `f0fb727` 111,
110 pass-in-both + 5 coin-flip), superseding `a875397` (52/124), `246a23c`
(49/124) and `3f256ab` (55/124), all retained.
Restore every throughput, latency, and memory axis on 27B and then 35B before
resuming the operational and T1/T2 portfolio. Detailed status lives in the area
matrices, active ownership in `coordination.md`, and chronological evidence in
the append-only state/ledger record.

**Current order-0 substage (2026-07-18):** the **two-grid parity verdict** (`9ecd9d0`
114/124 + `f0fb727` 111/124 → **115/124 effective**, see VERDICT below) supersedes `a875397` (52/124), `246a23c`
(49/124) and `3f256ab` (55/124), all retained. Evidence
`~/work/vllm.cpp-online-gate/evidence/9ecd9d0…`; ratios.json `8c81083e…`,
all-runs.json `c7e4a831…`, manifest.json `a3871da2…`. ZERO void, 12/12
binding-eligible. Per concurrency: mem **4/4**, c1 **20/20**, c2 **20/20**, c16
**19/20**, c4 18/20, c32 18/20, c8 15/20.

**+62 axes — a decode-kernel-efficiency close.** The prior binding's uniform ~1–2%
decode deficit was the batch-independent norm/quant/act kernel glue (per the
`c2-c8-attribution` measurement); closing it required a family of **bit-identical
(0-ulp) fast decode kernels** — each reproduces its shipped reference's exact
float-op order (so the fast set yields identical logits and never crosses the 27B
tok6 razor near-tie that reverted `a875397`), vectorizing only memory access:
`RmsNormRowFastKernel` (2.41× isolated, `348d12d`) closed c2 entirely (3→20);
`RmsNormGatedRowFastKernel` (2.04× at c16, `9ecd9d0`) closed the c16 floor (6→19).
The RMSNorm-alone in-situ A/B was null, but the STACKED reduction kernels rose
well above noise. All on the async (`a0013a2`, `ENG-ASYNC-SCHED` DONE) + vendored
Triton GDN cubin (`a321d7c`) defaults; full set 27B 235/235 + 35B 315/315
token-exact. Memory PASSES (windowed-load `cb2d310`); packed-decode CLOSED on
equivalence (`e47b4d6`); qkvz in the binary (`45f9e6d`).

**VERDICT (2026-07-18) — effective PARITY-OR-BETTER.** A second grid `f0fb727`
(adds bit-identical conv-update 1.92× + FP4/SiLU flips, all default ON) landed
111/124; the 111-vs-114 delta is pure noise-band coin-flip, calibrating the noise
floor. Two-grid per-axis totality: **110 pass-in-both + 5 coin-flip-splits =
115/124 effective**, 9 fail-in-both. All 9 residuals are the low-concurrency-median
edge of the deterministic-forward tradeoff and we are NET-POSITIVE on each: the c8
cluster (mean/median/p99) loses at c8 but wins c16/c32 p99 (1.055/1.078); c4
mean/median TTFT loses at c4 but wins c4 p90/p99 ttft (1.009/1.013) + c8/c16/c32
mean_ttft (1.030/1.100/1.136); c16/c32 median_itl is the median side, we win the
tails. No closeable real deficit — the async-forward jitter that would win the
low-conc medians forfeits the tail/high-conc/throughput wins (net-negative). NEXT:
35B performance closure, the SGLang floor, then the T1/T2 portfolio (orders 1–14).

*Slot-fix / equivalence / qkvz / async-discriminator / near-tie detail preserved in
the append-only state/ledger and the specs; not re-copied here per the compaction
directive.*

**★ SGLang elevated to a FULL parity target (user-directed 2026-07-27,
`CLAIM-SGLANG-PARITY-PROGRAM`).** "We need the same vLLM approach there for
reaching parity." The SGLang floor under `ROAD-V1-A` (`BACKEND-GATE-CUDA-SGLANG*`)
is now backed by the full vLLM-parity apparatus: a whole-surface tabular
inventory ([sglang-matrix.md](../sglang-matrix.md), 44 rows classified
FUSED 23 / SGLANG-DISTINCT 8 / INVENTORIED 5 / OUT-OF-SCOPE 8) and the
SGLang-as-oracle gate methodology ([specs/sglang-parity-oracle.md](../specs/sglang-parity-oracle.md)):
SGLang stood up as a correctness cross-check + a binding perf floor on dgx GB10
via the digest-pinned arm64 cu130 image (no from-source build needed). SGLang is
a COMPETITOR, not the mirror source — vLLM stays the behavior truth; a
SGLANG-DISTINCT behavior is an opt-in over our vLLM-derived design, never a fork.
The honest finding: most of SGLang's surface is FUSED (shared lineage); the value
is the precise SGLANG-DISTINCT map (LPM scheduling, in-batch prefix
de-prioritization, radix eviction strategies, jump-forward, custom logit
processors, batch-invariant determinism, PD disaggregation, two-batch EP overlap),
ranked in the oracle spec §6. This program does NOT add a new portfolio row — it
deepens `ROAD-V1-A`; the implementation rows (`KV-SGLANG-RADIX-CACHE`,
`ENG-SGLANG-BEHAVIOR-FLAG`) remain owned by `CLAIM-SGLANG-RADIX-SCOPE`, and the
residual perf-gate blocker stays `SERVE-ASYNC-LLM` (our comparable async server),
not SGLang runnability. **ENABLEMENT/DOCS side-quest 2026-07-28
(`CLAIM-SGLANG-ABI-DOCS`, reconciled to ABI v10):** the SGLANG-DISTINCT opt-ins are
now first-class, DOCUMENTED knobs exposed on the C++ API AND the C ABI (not
env-var/internal-only) — LPM scheduling via the concurrent session's v9 **string**
field `scheduling_policy="lpm"` + jump-forward (`enable_jump_forward`, ABI **v10**)
joined the already-exposed RadixAttention/APC (v7) and custom logits processors
(v8); user doc [docs/SGLANG-COMPAT.md](../../docs/SGLANG-COMPAT.md) + spec
[specs/sglang-enablement.md](../specs/sglang-enablement.md). Default-inert (all-zero
⇒ byte-identical); CPU exact-gate; `vllm_abi_version()`==10.

**★ SGLang PERF oracle STOOD UP + first floor MEASURED (2026-07-28,
`CLAIM-SGLANG-PERF-BENCH`).** The v0.5.15-cu130 arm64 image RAN the 27B-NVFP4
gate model on GB10 sm_121a (no from-source build — as the oracle spec §2
predicted). First reproduced cache-neutral SGLang-vs-ours comparison at c8/c16
(27B, 3 reps, idle box, one flock): **ours BEATS the SGLang floor on throughput
(2.21×@c16, 1.44×@c8) and TTFT (6–12× lower), but SGLang WINS per-token latency
(TPOT/ITL 1.18–1.49× below ours) — a reproduced OPEN GAP / candidate lever.**
Evidence + repro: [sglang-matrix.md](../sglang-matrix.md) § "Perf oracle results",
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md), [oracle spec §9](../specs/sglang-parity-oracle.md).
Residuals: 35B, c1/c2/c4 low-conc sweep, the shared-prefix cache-ON arm
(`BACKEND-GATE-CUDA-SGLANG-PREFIX`), and the token-exact correctness cross-check
(`SGLANG-ORACLE-CORRECT`). The existing `run_serve_low.py bench` harness drove
both arms; note its `SGLANG_IMAGE` pin is still v0.5.13 vs the oracle-spec v0.5.15.

**★ SGLang TPOT/ITL gap EXPLAINED — batch-composition, disposition set (2026-07-28,
`CLAIM-DECODE-LATENCY-EXPLORE`).** The above SGLang per-token-latency OPEN GAP is
now CHARACTERIZED (exploration, no source changed): it is **batch-composition, not
a decode-kernel deficiency.** On our engine, ITL(decode-batch=1)=**101.75 ms** is
already ≤ SGLang's op-point ITL (104–105 ms), and ITL rises monotonically with the
decode batch we pack (→158.5 ms @ B16); nsys shows every hot decode kernel
sub-linear in batch (per-token cost ↓~10×). SGLang's effective decode concurrency
is ~4 (not 16) because of its 33 s admission queue — its low ITL is the ITL of a
small batch. **Our throughput win IS the ITL cost — the same lever.** Disposition:
**NOT a bug/kernel-lever; a throughput↔latency operating-point choice.** Knob
already exists (`max_num_seqs` / `max_num_batched_tokens`); latency-oriented point
`max_num_seqs≈8` = ITL −21% at 1.38× SGLang throughput. Default stays
throughput-oriented (unchanged). Full curve + repro:
[specs/decode-latency-lever.md](../specs/decode-latency-lever.md),
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md).

**★ ACTIVE PHASE (user-directed 2026-07-20/21) — the BREADTH SWEEP.** With 27B/35B parity,
the `KERNEL-FUSION-FRAMEWORK` extensibility cornerstone (W0-W4), and the FIRST additive model
(Qwen3 dense: correctness-complete + c1 effective every-axis parity, c8 within a cross-cutting
decode-GEMM residual) all landed, the active priority is BREADTH: "more models and architectures",
recent-first. Plan + audit: [breadth-sweep-plan.md](../specs/breadth-sweep-plan.md). Key findings:
(A) CUDA-ARCH additivity — the POLICY/selection layer is clean+additive (PR-#4 debt paid; a
same-family sm_120 GPU = ~2 additive touches); the COMPUTE-KERNEL layer is a single-arch sm_121a
monolith with no runtime SM selector, so different families (sm_90/100/80) need new kernel bodies +
the named seam-gap fixes (CMake arch-guard→feature-table, a kernel SM selector, the paged-attn
d==256 hardcode). CRITICAL: only GB10 sm_121 is testable on dgx — the CUDA device sweep beyond
sm_120 is HW-BLOCKED (build/gencode-only), so it is NOT the actionable breadth. (B) The actionable
breadth is the MODEL sweep on GB10. RANKED QUEUE (recent + present + additive first): Tier 1 =
Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`, present, recent-MoE flagship, reuses the MoE path),
Qwen3-32B dense/NVFP4A16 (dense forward done + W4A16 loader), OPT-125m cross-family canary; Tier 2
(download) = Llama-3.x → Mistral → Gemma3 → GLM4/Olmo → Qwen3-Next; Tier 3 (campaigns) = MLA
(DeepSeek), Mamba/SSM hybrids. Each model held to the Qwen3-established bar: token-exact
(near-tie-robust where vLLM is self-inconsistent) AND vLLM-speed on every axis. This drives
`ROAD-V1-C2` (model half, actionable) + `ROAD-V1-D1` (device half, HW-blocked). AFTER the first
sweep: Metal/MLX (M4), Intel XPU, Vulkan, CPU-opt (benchmark vs llama.cpp, steal faster ideas).
**Sweep progress (correctness-complete, speed pending):** Qwen3-Coder-30B, Qwen3-32B-NVFP4A16,
OPT-125m, Llama-3.2-1B, Mistral-7B, Gemma-1/2/3, GLM-4-9B, GLM-4.7-Flash, DeepSeek-V2-Lite (MLA),
and now **OLMo-2 (`Olmo2ForCausalLM`/`Olmo3ForCausalLM`, SACRED 16/16, ZERO new kernels)** all
landed with passing SACRED gates; OLMo-2's `Olmo3ForCausalLM` W5 sliding-window code landed too
but its SACRED gate is oracle-BLOCKED (RUN-VERIFIED W0 2026-07-26 — see below); every landed
model's speed close remains. **DeepSeek-V4-Flash W0 SCOPE (2026-07-28, `CLAIM-DEEPSEEK-V4-SCOPE`,
[deepseek-v4-flash.md](../specs/deepseek-v4-flash.md)):** the ~167B/256-expert `DeepseekV4ForCausalLM`
is a NEW multi-brick arch — DeepSeek Sparse Attention (Lightning Indexer + compressor + fp8_ds_mla
KV + SWA) MLA, **Manifold Hyper-Connections** (`[T, hc_mult, H]` Sinkhorn streams, TileLang-only, no
eager ref), MegaMoE/NVFP4 (SM100-only ⇒ GB10 uses the FusedMoE fallback), sqrtsoftplus + hash-routed
MoE, clamped SwiGLU, grouped output LoRA, MTP + DSpark speculators. **W1/W2 IMPL SCAFFOLDING
LANDED (2026-07-28, `CLAIM-DEEPSEEK-V4-IMPL`, base `df18ca91`):** additive registry stub + config
parse + checkpoint loader name-map VERIFIED vs the real `nvidia/DeepSeek-V4-Flash-NVFP4` header
(HTTP-range, no download); clean CPU `-Werror`; forward is an honest W3-W8 `VT_CHECK` stub.
**HW-FIT REVERSAL:** the NVFP4 checkpoint is **156.7 GiB** (index total_size), NOT the spike's
~83 GiB — only the 256 experts are W4; the MLA + shared linears are FP8 + NVFP4 double-scale ⇒ it
does NOT fit ONE GB10's 119 GiB pool. **W1 (single-GB10 oracle run) is therefore MEMORY-INFEASIBLE**
(needs multi-node TP / CPU offload / smaller quant), not merely disk-contended. Forward + strict gate
= W3-W8. `DeepSeekV4MTPModel` promoted INVENTORIED→SPIKE. **GGUF BENCHMARK LOADABILITY spike
(2026-07-28, `CLAIM-DSV4-GGUF-SPIKE`):** a same-quant `UD-IQ2_XXS` GGUF benchmark of our engine vs
vLLM is **NOT viable today, blocked on BOTH sides** — vLLM 0.26 moved GGUF out-of-tree to
`vllm-gguf-plugin` (uninstalled; DOES dequant IQ2_XXS) but `DeepseekV4ForCausalLM` has no
`packed_modules_mapping`/GGUF wiring; our engine hard-rejects GGUF for DeepSeek-V4/V2
(`deepseek_v4_registry.cpp:61-64`) and lacks IQ2_XXS AND Q2_K dequant (only F32/F16/BF16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K/NVFP4).
So the C4/GGUF roadmap item to make the single-Spark GGUF vehicle real is: lift the V4 GGUF
reject + wire a V4-GGUF loader + port IQ2_XXS (llama.cpp `dequantize_row_iq2_xxs` + `iq2xxs_grid`)
and/or Q2_K dequant — gated vs llama.cpp-on-card (vLLM cannot oracle V4-from-GGUF). Apples-to-apples
DeepSeek-V4 is the NVFP4 2×-Spark vehicle; a true same-GGUF cross-engine number is only available on
a Qwen3/dense k-quant both engines already load. **QUANT-TYPE HALF LANDED (2026-07-29,
`CLAIM-DSV4-GGUF-LOADER`, [gguf-iquant-dsv4.md](../specs/gguf-iquant-dsv4.md)):** IQ2_XXS (id 16) +
Q2_K (id 10) now dequant (1:1 from llama.cpp `ggml-quants.c`; `QUANT-GGUF-IQ2_XXS`/`QUANT-GGUF-Q2_K`
→ ACTIVE; `test_gguf_dequant` 15/15). HTTP-range-verified the real GGUF header
(`general.architecture=deepseek4`, `general.file_type=19`=IQ2_XXS, `split.tensors.count=1328`, full
`deepseek4.*` config-KV schema). The V4 registry GGUF reject STAYS — the V4-GGUF `blk.N.*` name map
needs the tensor manifest (beyond the CDN range cap + uncached; 90 GB download prohibited) and the
forward is W3-W8. So the MODEL-ARCH half + the forward remain the residual to a runnable V4-GGUF vehicle. **W3 PRIMITIVES LANDED (2026-07-28,
`CLAIM-DEEPSEEK-V4-W3`, base `308c312a`):** the genuinely-NEW attention math is ported + unit-gated
as portable host references — the DSA "Lightning Indexer" sparse SELECTION (weighted-MQA logit
`Σ_h w·ReLU(q·k)` with the load-bearing per-head ReLU + causal top-k `index_topk=512` + short-context
all-select) and the two 512-wide-MLA output seams V2/V3 lack (per-head attention-sink softmax +
grouped output-LoRA `wo_a` bmm→`wo_b`); new TUs `deepseek_v4_dsa.{h,cpp}` + `test_deepseek_v4_dsa`
**13/13·38** (hand-derived literals + double-precision references rel-L2 < 1e-6), clean CPU
`-Wall -Werror -Wextra`, SACRED-inert (shared `mla_attention` untouched — extraction is a W7
follow-on); new kernel row `KERNEL-ATTN-DSA-SPARSE-INDEX` (`SPIKE`). **SGLang `v0.5.15` registers +
implements `DeepseekV4ForCausalLM`** (full DSA/MHC/o_lora stack, 2856 LoC) ⇒ a viable SECOND
benchmark/primitive-dump reference (same single-GB10 memory constraint). **W4 PRIMITIVES LANDED
(2026-07-29, `CLAIM-DEEPSEEK-V4-W4`, base `4d1be010`):** the second half of the DSA stack ported +
unit-gated as portable host references — the DSA COMPRESSOR forward (`CompressorPoolNorm` = the
softmax-weighted window POOL, `softmax(score,dim=0)` per head-dim column then RMSNorm; the fused
save-time APE add) and the **fp8_ds_mla** KV-cache state read/write layout (448-wide NoPE FP8 e4m3
with per-64 UE8M0 power-of-two block scales + 64-wide RoPE bf16, 576B token stride, 7+1 scale
region, + the dequant read); new TUs `deepseek_v4_compressor.{h,cpp}` + `test_deepseek_v4_compressor`
**12/12·164** (hand-derived literals + double-precision references + independent UE8M0 recompute;
RED-first proven), ported 1:1 from `fused_compress_quant_cache.py`/`save_partial_states.py`/
`compressor.py` and cross-checked vs SGLang `v0.5.15` `dsv4/dequant_k_cache.py`; SACRED-inert (shared
`mla_attention` empty-diff); new kernel row `KERNEL-ATTN-DSA-COMPRESSOR` (`SPIKE`). Residuals: MHC
(W5), sqrtsoftplus/hash MoE (W6), the fused device kernel + forward integration + the compressor
state-cache gather addressing (W7), strict gate (W8) = multi-Spark.
**W5 MHC HYPER-CONNECTIONS LANDED (2026-07-29, `CLAIM-DEEPSEEK-V4-W5`, base `d0bc0f41`):** the hardest
V4 brick — the Manifold/Markov Hyper-Connections residual topology ported + unit-gated as portable
host references: `MhcSinkhorn` (the 20-iteration Sinkhorn: row-softmax seed `+eps` then alternating
col/row normalization toward a doubly-stochastic matrix), `MhcPre` (folded weight-free RMSNorm
projection → pre/post/comb sigmoid+Sinkhorn gates → stream collapse → optional folded attn/ffn
RMSNorm), `MhcPost` (comb-matrix mix + post-gate residual fold), `HcHeadCollapse` (weight-free RMSNorm
→ hc_head_fn → sigmoid → weighted stream sum); new TUs `deepseek_v4_mhc.{h,cpp}` +
`test_deepseek_v4_mhc` **14/14·125** (hand-derived literals + from-first-principles double-precision
references rel-L2 < 1e-5..1e-4; RED-first proven BOTH the iteration count and a normalization axis, via
a dedicated small-iteration-count gate since the Sinkhorn converges by 20 iters). **The W0 "ZERO eager
reference upstream" premise is CORRECTED:** vLLM ships `model_executor/kernels/mhc/torch.py`
(`mhc_pre_torch`/`mhc_post_torch`) + `triton.py` (head collapse), and four upstream impls agree
byte-for-byte on the Sinkhorn — ported 1:1 AND cross-checked against an independent double-precision
derivation. SACRED-inert (no existing forward touched; prior V4 tests unchanged); new kernel row
`KERNEL-MHC-SINKHORN` (`SPIKE`). Residuals: sqrtsoftplus/hash MoE (W6), device kernel +
`DeepseekV4Model::Forward` assembly (W7), strict gate (W8) = multi-Spark.
**W6 sqrtsoftplus + HASH-ROUTED MoE LANDED (2026-07-29, `CLAIM-DEEPSEEK-V4-W6`, base `5b843be5`):**
the three genuinely-new-vs-V2/V3 MoE pieces ported + unit-gated as portable host references —
`SqrtSoftplus` (the V4 router score `sqrt(softplus(x))`, distinct from V2/V3's sigmoid/softmax
`noaux_tc`), `SqrtSoftplusRouteTopk` (score all experts → add `e_score_correction_bias` for
SELECTION ONLY → top-k, OR the `tid2eid` token-id→expert HASH lookup that BYPASSES top-k → GATHER
weights from the UNBIASED scores → renormalize → ×routed_scaling_factor), `ClampedSwiGLU`
(`SiluAndMulWithClamp`: gate clamped max-only, up clamped both-sided, `gate·σ(α·gate)·(up+β)`); new
TUs `deepseek_v4_moe.{h,cpp}` + `test_deepseek_v4_moe` **12/12·716** (hand-derived literals — the
sqrt∘softplus composition, bias-flips-selection-but-weight-stays-unbiased, the hash bypass, the
asymmetric clamp — + from-first-principles double-precision references; RED-first proven ALL THREE
load-bearing levers: drop the sqrt fails 8/493, gather weights from the biased scores fails 2/181,
symmetric-clamp the gate fails 2/6). Ported 1:1 from vLLM `fused_topk_bias_router.py:75-118`
(`_topk_softplus_sqrt_torch`) + `activation.py:197-201`, cross-checked SGLang `v0.5.15`
`moe/{topk.py, hash_topk.py}`. REUSE not re-port: the shared DeepSeek grouped-GEMM / 256-expert /
shared-expert / NVFP4 machinery is untouched — only scoring+hash+clamp are net-new; MegaMoE is
SM100-only so GB10 mirrors the FusedMoE-fallback router. SACRED-inert (no existing forward touched;
prior V4 tests 14/14·125 + 12/12·164 + 13/13·38 + 4/4·40 unchanged); new kernel row
`KERNEL-MOE-SQRTSOFTPLUS-HASH` (`SPIKE`). Residuals: device kernels (reuse the existing grouped-GEMM)
+ `DeepseekV4Model::Forward` assembly (W7), strict/near-tie gate (W8) = multi-Spark.
**W7 FORWARD ASSEMBLY LANDED (2026-07-29, `CLAIM-DEEPSEEK-V4-W7`, base `a856383c`):** the brick that
finally makes V4 structurally runnable — the `VT_CHECK(false, "W3-W8 pending")` stub is REPLACED by a
REAL `DeepseekV4Model::Forward` (`DeepseekV4ForwardHost`, in `deepseek_v4.{h,cpp}`) that COMPOSES the
four landed host primitives (W3 DSA/MLA seams, W4 compressor + fp8_ds_mla KV, W5 MHC + Sinkhorn, W6
sqrtsoftplus/hash MoE) into an end-to-end logits producer on the portable CPU path at a SMALL synthetic
config. Interleave grounded 1:1 (`nvidia/model.py:1080-1148` + `:866-957`): embed → per layer
[first-layer MHC-pre stream EXPAND `[T,H]→[T,hc,H]`, else fused MhcPost(prev-ffn)+MhcPre(attn)] →
512-wide MLA (q/kv+norms, RoPE, DSA indexer→topk→compressor→fp8_ds_mla KV, sink softmax, grouped
o-LoRA) → fused MhcPost(attn)+MhcPre(ffn) → MoE (sqrtsoftplus/hash + shared+routed clamped-SwiGLU) →
final MhcPost → hc_head collapse → norm → lm_head. Gate `test_deepseek_v4_forward` **6/6·26** —
STRUCTURAL/composition (finite logits end-to-end + deterministic + `[T,vocab]`; MHC stream `[T,hc,H]`;
hash layers route by `tid2eid` vs gated top-k; DSA SELECTS + compressor POOLS; `logits_indices` gather)
+ RED-first PROVEN 3 levers (all-gated hash, skip-final-MhcPost, no-sink each change the output). Honest
3-state: DERIVED + BUILD-VERIFIED (structural) — does NOT claim V4 "runs" a real model (documented
tiny-vs-167B divergences). SACRED-inert (only `deepseek_v4.{h,cpp}` + new test + CMake; shared MLA/MoE +
W3-W6 TUs empty-diff; prior V4 tests unchanged); NO new kernel row / no checker bump; new TU
`-Wall -Werror -Wextra`-clean. Residuals: W7-device CUDA kernels + `ForwardDevice`; W2b real-tower
materialization; W8 strict/near-tie engine gate (multi-Spark, 156.7 GiB); the single-Spark IQ2_XXS-GGUF
vehicle additionally needs the GGUF `blk.N.*` name-map (W2, download-blocked).
**W7-DEVICE LANDED + DGX-GATED (2026-07-29, `CLAIM-DEEPSEEK-V4-W7-DEVICE`, base `33016f34`):** the last
engineering brick before an actual V4 run — the CUDA kernels for the four NEW V4 op families (MHC
Sinkhorn/pre/post/head; DSA indexer weight-fold + weighted-MQA ReLU logits + causal top-k +
attention-sink softmax + grouped output-LoRA; compressor pool+norm + save-APE + fp8_ds_mla KV
encode/decode; sqrtsoftplus/hash router + clamped SwiGLU), each a 1:1 device port of the landed host
reference, registered through the vt OpProvider seam (`kDeepseekV4{Mhc,Dsa,Compressor,Moe}`) and
dispatched by a real `DeepseekV4Model::ForwardDevice` (the ONE composition now runs on host refs OR the
device kernels via a `V4Backend` policy). The 512-wide MLA attn + expert grouped-GEMM REUSE the existing
NVFP4/FP8 kernels (NOT re-ported). New TUs `src/vt/cuda/cuda_deepseek_v4.cu` + `deepseek_v4_device.{h,cpp}`
+ `test_cuda_deepseek_v4.cpp`; new kernel row `KERNEL-DSV4-W7-DEVICE` (`SPIKE`, count 42→43). **DGX GB10
(sm_121a) `test_cuda_deepseek_v4` 11/11 cases · 153 assertions GREEN** vs the host-ref oracle at small
shape (BIT-EXACT top-k/router ids, near-tie rel-L2 < 1e-4 for the fp reductions, fp8_ds_mla within e4m3
granularity) + the ForwardDevice composition gate (device == host rel-L2 < 2e-3), **compute-sanitizer
memcheck 0 errors**, RED-first proven. CUDA + CPU `-Werror` clean (the pre-existing GCC-13 voxtral #155
array-bounds/stringop false positive neutralized locally). SACRED-inert (shared MLA/MoE CUDA + W3-W6 host
TUs empty-diff; host oracle 6/6·26 unchanged). Honest 3-state: kernels RUNTIME-VERIFIED at small shape on
real GB10; the real-checkpoint e2e stays W8 (156.7 GiB does not fit ONE GB10) + W2b tower materialization
+ the GGUF `blk.N.*` name-map.
**MEASURED PARITY LEVERS INTEGRATED TO main (2026-08-03, GB10 build+gate):** four verified branches landed
together. (1) DeepSeek MHC-pre FP64→FP32 fold (`CLAIM-DSV4-MHC-FOLD`, `VT_V4_MHC_FUSED` default-ON) — the
float MHC-pre mirroring ds4 is BIT-EXACT (decode ids token-identical `=1`/`=0`; `test_cuda_deepseek_v4`
Brick-B PASS on GB10) and MEASURED **85.4%→87.5% of ds4 ~16.5** (recomputed onto the corrected ds4 bar). (2) GGUF UD-IQ2_M CPU bring-up — multi-shard
`gguf-split` stitching (`CLAIM-GGUF-SPLIT-SHARDS`, `test_gguf` 33/33) + IQ2_S/MXFP4 keep-quant `vec_dot`
(`CLAIM-DSV4-UDIQ2M-QUANT`, CPU-green); the IQ2_S device `DotSuperblock<kIQ2_S>` is now **CUDA-BUILT + LINKED
on GB10** (sm_121a, `-Werror`; one integration fix: `DotMXFP4` marked `[[maybe_unused]]` — the file had never
been through nvcc). (3) Laguna long-context levers (`CLAIM-LAGUNA-LONGCTX-LEVERS`) — window-bounded SWA reads
(`VT_LAGUNA_SWA_WINDOW` default-ON, BYTE-EXACT: GB10 A/B token-IDENTICAL at 520-token context) MEASURED
**−0.30 ms/step@2k**; bf16 paged KV (`VT_LAGUNA_KV_BF16` default-OFF opt-in) a near-tie left UNRATIFIED.
**SACRED gates UNMOVED on the merged GB10 build: `test_qwen27_paged_engine` 235/235 + `test_qwen36_paged_engine`
315/315.** See docs/BENCHMARKS.md + docs/STATUS.md.
**DEEPSEEK-V4 DECODE GLUE-FUSION LEVERS 2+3 INTEGRATED TO main (2026-08-03, GB10 sm_121a clean build+gate, base `6576814b`):**
two measured byte-exact decode-glue folds landed on top of the MHC-pre fold. (3) norm+RoPE FP64→FP32 fold
(`CLAIM-DSV4-ROPE-FLOAT`, `VT_V4_ROPE_FLOAT` default-ON) — the fused `NormRopeRows` kernel dropped **4.58→0.46 ms/step
(~10×)**, decode +6.1%. (2) MHC-pre finish occupancy widen (256→1024) + sqrsum-fold (`CLAIM-DSV4-MHC-LEAN`,
`VT_V4_MHC_LEAN` default-ON=1024) — +0.7%; HONEST BOUND: the finish is floored by 86 sequential single-block
launches/step (NOT warp-latency/memory), the owed residual = a layer-chain-spanning MHC restructure (high-risk).
Net decode **14.02→14.96 tok/s (+6.7%) → 90.7% of ds4 ~16.5** (ds4 bar corrected from the unreproduced 17.13 anchor to the fair same-session ~16.5; a later MHC-SINK4 lever reached ~15.87 tok/s = ~96%). BYTE-EXACT: decode ids token-IDENTICAL `=1`/`=0`
for BOTH levers on ds4flash IQ2XXS via the resident-decode path (`--gpu --kv-cache`, device kernels genuinely
exercised) + `test_cuda_deepseek_v4` 20/20·67073 (Brick-7 + Brick-B). **SACRED UNMOVED on the merged GB10 build:
`test_qwen36_paged_engine` 315/315** (27B checkpoint absent on box ⇒ that gate SKIPs). See docs/BENCHMARKS.md + docs/STATUS.md.
**W8-FINAL — ENTRYPOINT WIRING LANDED + GATED; THE RUN RE-SCOPED TO A CODE BLOCKER (2026-07-29,
`CLAIM-DEEPSEEK-V4-W8`, base `376e186b`):** with W8 keep-quant + W2b tower materialization landed, this
lane wired the top-level GGUF dispatch arm (the "one small code piece W2b named") and attempted the real
single-Spark run. **Wiring (LANDED):** `DeepseekV4HfConfigFromGguf` (`deepseek_v4_weights.cpp`) maps
`general.architecture=deepseek4`→the registered `DeepseekV4ForCausalLM` (republishing geometry into
`config.raw` for `ParseDeepseekV4Config`); `LoadedEngine::FromModelDir` routes via
`HfConfigFromGgufDispatch` (was qwen-only). Gate `test_deepseek_v4_gguf_load` **6/6·168** (CPU
full-library `-Werror`-clean; +1 case: config maps + `ModelRegistry::Resolve`→V4 factory), qwen path
byte-neutral (`test_model_registry` 24/24). **The RUN did NOT execute — BLOCKED on a CODE residual, not
download/box, provable from source; NOT attempted (would OOM-reboot the box):** the forward
(`ForwardComposeImpl`) reads the FULLY-DEQUANTIZED f32 `weights.host` tower, and `LoadDeepseekV4FromGguf`
builds it unconditionally (every routed expert `HostVec`→f32) ≈ ~24 GiB/layer × 43 ≈ **~1.0 TiB** f32,
past the 119 GiB unified pool by ~layer 5; the keep-quant `weights.gguf` tower (~91 GiB) is built but
never read by the forward. **Named residual W2c:** rewire the forward onto the CIQ `kMatmulBTQuant`
keep-quant blocks + gate off the host-f32 dequant, THEN the download + GB10 greedy gen +
self-consistency/coherence gate + benchmark. No tokens generated (not faked); DGX left as found.
**W2c LANDED — the OOM-infeasibility is FIXED (2026-07-29, `CLAIM-DEEPSEEK-V4-W2C`, base `328e6a50`):**
`LoadDeepseekV4FromGguf` no longer f32-expands the big MLA/MoE/lm_head weights (only the small
norms/embed/MHC/DSA/hash tensors dequant to `host`); a new `DeepseekV4ForwardGguf` runs the SAME
composition with the 512-wide MLA linears + 256 routed/shared expert GEMMs + lm_head CONSUMING the
COMPRESSED `weights.gguf` blocks in place via `vt::MatmulBT`→the CPU `kMatmulBTQuant` CIQ GEMM (a `Gemm`
/ `GemmRowSlice` / `GroupedOutputLoraGguf` helper trio), and `DeepseekV4Model::Forward` gates on
`has_gguf_weights` (the safetensors/NVFP4 + tiny-synthetic host path stays byte-identical). Gate
`test_deepseek_v4_gguf_load` **7/7·185** (CPU Release `-Werror`-clean): keep-quant forward RUNS
finite+deterministic; keep-quant(Q8_0)==dequant(bf16) RelL2 **0.0116** (< 0.05 near-tie); RED-first
(no-sink miswire diverges 0.122; a load that rebuilds the f32 tower fails a load-time `VT_CHECK` + the
host<gguf-bytes assertion). MEMORY-BOUND asserted: host 23,980 B < keep-quant 141,676 B at tiny shape;
projected full-scale the 256 routed experts alone are **~1032 GiB** f32 (OOMs the 119 GiB pool) vs the
keep-quant `UD-IQ2_XXS` **~91 GiB** + small host **< 3 GiB** = **memory-FEASIBLE** on ONE GB10. REUSES the
landed `kMatmulBTQuant` (no new kernel row / no checker bump). SACRED-inert (W3-W6 primitive tests
unchanged; shared MLA/MoE + CUDA W7-device untouched). The real 91 GB run stays the operational W8-run
(download + GB10 generate + benchmark), now memory-feasible.
**CUDA KEEP-QUANT GEMM LANDED — the experts move OFF the ARM cores (2026-07-29,
`CLAIM-CUDA-KEEPQUANT-GEMM`, `KERNEL-QUANT-CIQ-GEMM-CUDA`, base `2191f771`):** W2c reused the CPU
`kMatmulBTQuant`, so on a CUDA runner the keep-quant experts fell to the unified-memory CPU reference
tier (the 20 ARM cores). This lane adds the FIRST CUDA keep-quant GGUF k-quant GEMM — a native kCUDA
`kMatmulBTQuant` provider (`src/vt/cuda/cuda_quant_dot.cu`, MMVQ-style: Q8_K activation quant + integer
dot against the compressed Q8_K-family blocks, dequant-in-kernel, weights kept COMPRESSED in the unified
pool). Registering it flips `GgufQuantComputeAvailable` TRUE on `kCUDA`, so `vt::MatmulBT` dispatches
these GEMMs to the GPU (the biggest DeepSeek-V4 speed lever). RUNTIME-VERIFIED on the DGX GB10:
`test_cuda_quant_dot` 2/2·92401 vs the CPU oracle (NMSE ≤1e-6, int core bit-exact) + f64 dequant
(≤5e-4), compute-sanitizer memcheck 0, RED-first proven. A 1:1 numeric port of the landed CPU oracle;
additive (CPU reference untouched; `.cu` CUDA-only so the CPU build is byte-identical). KERNEL 44→45.
The experts-on-GPU tok/s (part of the W8-run) is the follow-on benchmark.
**MXFP4 QUANT PATH W0/W1 LANDED (2026-07-28, `CLAIM-QUANT-MXFP4`, base `42c56b51`,
[`QUANT-CT-MXFP4`](../quantization-matrix.md) `INVENTORIED`→`ACTIVE`):** the shared unblocker BOTH
DeepSeek-V4 (W6 MegaMoE MXFP4 experts) and Kimi-K3 (real checkpoint is `mxfp4-pack-quantized`)
need to load — CPU weight unpack + E8M0 dequant (`2^(byte-127)`, group 32, NO global scale — the
distinctions vs our NVFP4 group-16 fp8-scale path), new TUs `mxfp4_dequant.{h,cpp}` +
`test_mxfp4_dequant` **5/5·1142** vs a double-precision `dq_mxfp4_torch` port (RED-first: a
bias-128 mutation fails 446 assertions), clean CPU `-Werror`, ports FROM
`compressed_tensors_w4a4_mxfp4.py` + `mxfp8_utils.py` + `reference_mxfp4.py`. Residual (NAMED
W2-W5): scheme wiring, GPU W4A4 fp4 GEMM + Marlin W4A16 fallback, MoE expert path, e2e; the two
model loaders keep their MXFP4 refusal until wired.
**NEXT-TIER BATCH TRIAGE (2026-07-24,
`sweep-recent-dense-batch.md`, `CLAIM-SWEEP-RECENT-DENSE`):** 8 recent dense/small-MoE families
advanced `INVENTORIED` -> `SPIKE` with a ranked one-agent-each queue — **4 ZERO-NEW-KERNEL
near-additive** (Phi-3/Phi-4 `Phi3ForCausalLM` a Llama subclass, Granite-3 4-scalar-multipliers,
StableLM LayerNorm+partial-rope, MiniCPM scalars) + 4 small-new-op (InternLM2 wqkv-split,
Command-R logit_scale+parallel-residual, Phi-1/2 NewGELU-unary, MiniCPM3 MLA-rewire). **TOP 3
IMPLEMENTED (2026-07-24, batch3, `sweep-recent-dense-batch3`):** **Granite-3 (`GraniteForCausalLM`)
`ACTIVE` — SACRED 16/16** vs vLLM 0.25.0 (K=5 deterministic → STRICT; 4 default-1 scalars +
attention scale proven, ZERO new kernel); **Phi-3/Phi-4 (`Phi3ForCausalLM`) `ACTIVE` — gate now a CLEAN PASS (2026-07-26)** via the ratified near-tie ROOT-divergence methodology: Phi-4-mini 16/16 (7 strict + 9 near-tie; p12's >0.5 positions are CASCADE downstream of an exact-tie root, not a forward error) + BIGGER-DENSE STRICT anchor `microsoft/phi-4` (14B) 16/16 with STRICT token-exact 14/16, RED-first-verified, TEST+goldens only;
**OLMo-3 (`Olmo3ForCausalLM`) IMPLEMENTED but oracle-BLOCKED** — the pinned vLLM 0.25.0
cannot run `allenai/OLMo-3-1025-7B` (`KeyError: 'rope_theta'`, per-layer-type rope schema newer than
the oracle's transformers), so no SACRED bar (our engine loads+runs it; W5 gate pending a pin/oracle
advance). **PIN-ADVANCE EXECUTED + FLIPPED 2026-07-26 (`CLAIM-PIN-ADVANCE-W5`,
[`specs/pin-advance.md`](../specs/pin-advance.md)): pin is now `555967922` / vLLM 0.26.0.dev0 +
transformers 5.14.1** (from `e24d1b24`/0.25.0). Zero real golden drift (W3-W4: 27B-W4A4 +
32B-NVFP4A16 bit-identical; the W0-W2 "27B drift" was a capture-config near-tie), re-gate
296/299 GREEN, rollback preserved (`~/venvs/vllm-oracle-v0.25.0-stage`). UNBLOCKS OLMo-3 W5 →
DFlash D1-D6 (mixed-attn constructs+accepts under MRV2) → Gemma-4 mm+audio → frontier arches.
(Scoping was `CLAIM-PIN-ADVANCE-SCOPE`.) the three oracle-pin
blockers (DFlash vllm#40898, Gemma-4 mm, OLMo-3 nested-rope) ALL clear at ONE
coherent target — vLLM `origin/main` `55596792` + transformers 5.14.1 / torch 2.13.0
/ flashinfer 0.6.15 / cutlass-dsl 4.6.0 (no release tag carries the fixes; DFlash
mixed drafts need `VLLM_USE_V2_MODEL_RUNNER=1` = our MRV2). HONEST cost: advancing
likely DRIFTS the 2 NVFP4 hybrid gates (27B/35B) + 32B-NVFP4A16 + Coder (~4 core
golden re-captures) + a diff pass over ~30 rows (Torch/CUTLASS/FlashInfer +
NVFP4-MoE + rmsnorm-quant-fusion move the oracle). Migration = staged
`~/venvs/vllm-oracle-next` → 3-unblock smoke test → golden re-capture → mechanical
re-sync (qwen3_next/layernorm/fused_moe/mamba/sampler/dflash + new olmo3) → pin flip
(W0-W5, spec §4). Post-unblock priority: OLMo-3 W5 → DFlash D1-D6 → Gemma-4 mm/audio.
**rank-4 StableLM (`StableLmForCausalLM`, `stablelm-2-1_6b`) `ACTIVE` — SACRED 16/16**
vs vLLM 0.25.0 (2026-07-26; per-prompt K=5 deterministic → STRICT; 14/16 exact + 2/16 near-tie,
max gap 0.438 nats, 0 divergent; ZERO new kernel = nn.LayerNorm + partial NeoX rope 16/64 + merged
qkv bias all reuse; one inert shared tokenizer canon for the Arcade100k CR/LF split-regex variant).
**rank-6 InternLM2 (`InternLM2ForCausalLM`, `internlm2-chat-1_8b`) `ACTIVE` — SACRED 16/16** vs
vLLM 0.25.0 (2026-07-26; per-prompt K=5 deterministic → STRICT; 12/16 exact + 4/16 near-tie, **max
gap 0.0 nats**, 0 divergent; ZERO new kernel = reuses the Llama/Qwen3-dense forward VERBATIM, the
ONLY delta a loader-side de-interleave of the fused `wqkv` — packed q/k/v interleaved by KV-group,
`internlm2.py:158-176`; RED wrong-split CAUGHT at prompt0 tok0). Gated via the additive
`TokensPrompt` engine path (InternLM2's non-standard fast BPE is not a supported tokenizer family —
a full port is orthogonal). **Command-R / Cohere (`CohereForCausalLM`) IMPLEMENTED but `BLOCKED`**
(2026-07-26, `sweep-commandr`): faithful ZERO-NEW-KERNEL port grounded in `commandr.py` (weight-only
Cohere LayerNorm + GPT-J full-width rope + PARALLEL residual + `logit_scale` + tied embeds), compiles
+ links + self-registers CPU `-Werror` clean; W0 oracle RUN-VERIFIED (tiny-random builds+runs on vLLM
0.25.0, arch confirmed `CohereForCausalLM`≠`Cohere2`). NO SACRED gate — every real small
`CohereForCausalLM` is HF-gated (no dgx token), the only ungated vehicles are tiny-random (head_dim
8/2, outside validated attn), and dgx is disk-full. **rank-8 Phi-1/Phi-2 (`PhiForCausalLM`,
`microsoft/phi-2`) `ACTIVE` — SACRED 16/16** vs vLLM 0.25.0 (2026-07-26, worktree `phi12-bringup`,
dgx `~/vllmcpp-phi12`; per-prompt K=5 deterministic → STRICT; 9/16 exact + 7/16 near-tie, max gap
0.25 nats, 0 divergent). The OLDER Microsoft Phi arch, DISTINCT from `Phi3ForCausalLM`. **ZERO new
kernel** (the spike's predicted `kGelu` unary was unnecessary — `gelu_new` == the landed
`vt::GeluTanh`): GPT-J parallel residual (Command-R wiring) + nn.LayerNorm+bias + biased qkv/dense
+ partial NeoX rope 32/80 + non-gated NewGELU MLP + untied biased lm_head, all reuse; F16→BF16
dtype-aware loader (LOCAL, shared header untouched). RED-first: dropped qkv bias 1.25 nats +
sequential residual 21.19 nats gate-CAUGHT, wrong rotary fraction hard-aborts. **rank-5 MiniCPM
(`MiniCPMForCausalLM`, `openbmb/MiniCPM-2B-sft-bf16`) `ACTIVE` — SACRED 16/16** vs vLLM 0.25.0
(2026-07-26, worktree `minicpm-bringup`, dgx `~/vllmcpp-minicpm`; per-prompt K=5 deterministic →
STRICT; 10/16 exact + 6/16 near-tie, **max gap 0.0 nats**, 0 divergent). The first OpenBMB MiniCPM
model. **ZERO new kernel** = the landed Llama/Granite dense forward + three scalars (scale_emb 12
embedding scale; scale_depth/sqrt(40)=0.2214 scaled residual add per sublayer; hidden/scale_width=9.0
before lm_head, all `vt::MulScalar`/`vt::Add`), tied lm_head. W0 `.bin`-only risk resolved WITHOUT a
C++ pickle loader: converted the official openbmb `.bin`→safetensors via trusted torch (same weights,
both oracle+engine); vLLM builds+runs it with `trust_remote_code=True`. RED-first: dropping
scale_depth → 256/256 divergent, max gap 29.375 nats gate-CAUGHT. Gated via the additive
`TokensPrompt` path (MiniCPM's SentencePiece normalizer is a follow-up). **rank-9 MiniCPM3
(`MiniCPM3ForCausalLM`, `openbmb/MiniCPM3-4B`) `ACTIVE` — SACRED 16/16** vs vLLM 0.25.0
(2026-07-26, worktree `sweep-minicpm3`, dgx `~/vllmcpp-minicpm3`; per-prompt K=5 deterministic →
STRICT; 13/16 exact + 3/16 near-tie, **max gap 0.0 nats**, 0 divergent). The first MLA-attention
MiniCPM and the model that **CLOSES the non-trivial recent-dense tier**. **ZERO new compute kernel**
= the landed MiniCPM 3 scalars with attention swapped GQA→**MLA**, REUSING the landed DeepSeek-V2 MLA
block; THREE deltas handled faithfully — `is_neox_style=True` (via a new default-false shared
`MlaBlockDims` field ⇒ DeepSeek byte-identical), LongRoPE-not-YaRN (phi3_long_rope short cache,
mscale 1.0), q_lora-always. ONE reuse-not-new shared change: the FA-2 MLA prefill zero-pads
qk_head_dim 96→128 to reuse the compiled hdim128 kernel (identity for DeepSeek d=192/GLM d=256).
RED-first (MLA-specific): wrong rope style → first-token divergence gate-CAUGHT. **DeepSeek-V2-Lite
re-gated 8/8** (shared-MLA non-regression). `.bin`-only vehicle resolved via trusted torch
`.bin`→safetensors. **TRIVIAL-TAIL DONE (2026-07-26, `sweep-yi-internlm3`) — the recent-dense TEXT
tier is now CLOSED (all 8 families accounted for).** Both remaining rows brought up as Llama
aliases on the ACTIVE `MODEL-TEXT-llama-llama-for-causal-lm` row (near-zero-work), each W0
RUN-VERIFIED on the pinned vLLM 0.25.0 oracle (the OLMo-3 lesson): **Yi (`01-ai/Yi-Coder-1.5B-Chat`)
— VERDICT modern Yi IS the Llama arch (`architectures:["LlamaForCausalLM"]`), ZERO code delta, no
`YiForCausalLM` alias (vLLM registers none); SACRED 16/16** vs vLLM 0.25.0 (K=5 deterministic ⇒
STRICT; 13 strict + 3 near-tie, max gap 0.125 nats, 0 divergent). **InternLM3
(`internlm3-8b-instruct`) — VERDICT a PLAIN Llama alias (`registry.py:134`→llama), NOT
InternLM2+sliding-window (no `sliding_window` in config; dynamic-NTK rope factor 6.0); ONE additive
registry-alias line; SACRED 16/16** vs vLLM 0.25.0 (K=5 deterministic ⇒ STRICT; 14 strict + 2
near-tie, max gap 0.0 nats, 0 divergent). Additive-only ⇒ `Llama-3.2-1B` re-gated 16/16 UNCHANGED.
**Recent-dense TEXT tier CLOSED:** Granite-3/StableLM/InternLM2/Phi-3-4/Phi-1-2/MiniCPM/MiniCPM3
SACRED-landed, Command-R HF-gate-blocked, OLMo-3 oracle-pin-blocked, + Yi/InternLM3 landed here.
Falcon / Falcon-H1(SSM) / GraniteMoe* / Cohere2Moe / PhiMoE stay `INVENTORIED` as MoE/SSM
campaigns; the pin-removed names (Phi3Small/Phi4Flash/Phi4Multimodal/InternLM2VE) have no 0.25.0
oracle.

**FRONTIER SWEEP SCOPED (2026-07-25, [sweep-kimi-minimax-glm-latest.md](../specs/sweep-kimi-minimax-glm-latest.md),
`CLAIM-SWEEP-FRONTIER-KMG`):** the three user-named frontier families dispositioned as ACTIONABLE
MECHANICAL PORTS (honesty-pass gates substitute for HW-blocked e2e). User-named versions vs the pin:
**Kimi K3 ABSENT** (no arch class; big Kimi MoE loads as `DeepseekV3ForCausalLM`); **MiniMax "M2.7"
not an arch** (loads as `MiniMaxM2ForCausalLM`; newest registered = M3 `MiniMaxM3Sparse`); **GLM latest**
= `Glm4Moe`/`GlmMoeDsa` (owned by the GLM/DSA claim). Ranked: (1) **Kimi-Linear-48B FITS GB10
(91.5 GiB, 0.77× pool)** = the ONLY frontier model with a REAL e2e SACRED gate — reuses MLA + sigmoid
router + bf16 grouped-MoE + GDN base, ONE genuinely-new kernel (the **KDA gated-delta gate** — first
buildable brick landed 2026-07-28, `KERNEL-KDA-DELTA`: the per-channel `[H,D]` low-rank decay
(`f_a`/`f_b`), the sigmoid-gated output norm, the 3 q/k/v convs + q/k L2-norm as portable host
references, `test_kimi_kda` 14/14 CPU; KDA subclasses GDN so the recurrence is reused and the brick is
GDN-inert; device kernel + Kimi-Linear-48B proxy e2e are named residuals), needs
~10 GiB dgx-disk reclaim; (2) **MiniMax-M2** HW-blocked (**214.3 GiB fp8-native, 1.80× over** — corrects
the matrix's wrong "428 GiB bf16/4×"), ZERO-new-kernel port → honesty-pass; (3) **`Glm4Moe`** 0-new-kernel,
honesty-pass at bf16, with **GLM-4.5-Air-FP8 (104.8 GiB) the fitting variant** that jumps the queue to a
real e2e gate if fp8-checkpoint loading lands. Kimi-K2 (958.5 GiB) / MiniMax-M3 (795.5 GiB + sm100 sparse
+ multimodal) / GLM-5 (1404 GiB + DSA DEP-blocked) = registry/config-resolution only.

**KIMI K3 W0 SCOPE — DERIVE-AND-SHIP (2026-07-28, [kimi-k3.md](../specs/kimi-k3.md),
`CLAIM-KIMI-K3-SCOPE`):** Kimi K3 RELEASED 2026-07-27 (post-pin) — this SUPERSEDES the sweep's
"Kimi K3 ABSENT / loads as `DeepseekV3ForCausalLM`" line. From the HF `config.json`:
`KimiK3ForConditionalGeneration` wraps a `text_config.architectures:["KimiLinearForCausalLM"]`
backbone (the pinned Kimi-Linear KDA+MLA+MoE hybrid) massively scaled — **H=7168, L=93 (69 KDA +
24 MLA), 896 experts/top-16/2 shared**, DeepSeek-V3 MLA geometry, **MXFP4 (group-32/e8m0) + MXFP8
acts**, **MoonViT-V2** vision. **HEAVY REUSE** (GDN=KDA parent, our landed DeepSeek-MLA, DeepSeek
MoE, the Qwen3.6-35B GDN-hybrid-MoE skeleton); NET-NEW = the KDA kernel delta (shared with the
Kimi-Linear row), MXFP4, AttnRes (report-only/UNCONFIRMED), MoonViT-V2. **HW: does NOT fit GB10 —
~1.56 TB MXFP4 ≈ ~12× the 119 GiB pool**; no small K3 exists. **DERIVE-AND-SHIP** (no on-box
golden, like the beyond-vLLM CUDA bricks): REAL proxy gate of KDA+MLA+MoE on the FITTING
`Kimi-Linear-48B-A3B` (~89–91 GiB) vs the pin, + build-verify/structural review for the scale-up.
The pin has no `kimi_k3` ⇒ oracle-gating K3 itself needs a pin advance. New SPIKE row
`MODEL-MM-kimi-k3-*`. **W2/W5 CPU SCAFFOLDING LANDED (2026-07-28, `CLAIM-KIMI-K3-W2-W5`,
DERIVED+BUILD-VERIFIED):** additive registry stub (`KimiK3ForConditionalGeneration`, hybrid+mm),
nested `text_config`/`vision_config`/`quantization_config` descent, the 93-layer KDA/MLA +
896-expert MoE text-backbone structural name-map (grounded 1:1 in `kimi_linear.py` +
`kimi_gdn_linear_attn.py`), REFUSE-by-name forward, MXFP4-refuse loader; clean CPU build + scaffold
gate 6/6 (`kimi_k3{,_registry,_weights}.cpp`, `test_kimi_k3_scaffold.cpp`). MXFP4 / KDA delta /
MoonViT-V2 correctly left NOT-YET-BUILDABLE (shared DeepSeek-V4 MXFP4 row, Kimi-Linear KDA row, W7).
Row stays SPIKE. NEXT: W1 proxy primitive gate (shares the KDA kernel campaign).
