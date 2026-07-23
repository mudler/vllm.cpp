# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. For the user-facing overview (what the project is,
how to build it, the CLI, the OpenAI server, and how to consume it), see the
[README](../README.md); this page is the detailed numbers behind its
[Performance](../README.md#performance) section. Attempt chronology and failure
forensics live in the [parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git. Those raw records are
append-only within the current era and are frozen under `.agents/completed/`
when the era is rolled up; this page never accumulates their run-by-run history.
House style: honest measured numbers only, and no em-dashes (use commas,
periods, parentheses, or hyphens), matching the README.

**Roadmap note (2026-07-20):** the first additive-model bring-up (Qwen3 dense on
`Qwen3-0.6B`, [spike](../.agents/specs/first-additive-model-qwen3-dense.md)) is a
CORRECTNESS deliverable - its gate is greedy vs the vLLM 0.25.0 oracle, not a perf
benchmark; no binding number is claimed for it. **W4 CORRECTNESS COMPLETE
2026-07-20 - the near-tie-robust gate PASSES on both 0.6B and a bigger 4B.** The
2026-07-20 razor's "vLLM greedy non-deterministic" premise was a BATCHING artifact:
per-prompt (batch=1 - the gate's single-request regime) vLLM 0.25.0 greedy is
DETERMINISTIC (0.6B 0 multi-member cells over K=10, 4B 0 over K=5; only when all 16
prompts are batched in one call does it flip). Forward correctness is PROVEN by
teacher-forcing vLLM on OUR exact prefix (`scripts/qwen3-neartie-gap.py`): at
all-but-2 positions vLLM's own argmax given our prefix IS our token with gap 0.0000
(bit-identical logprobs) - our forward matches vLLM's prefill logits; the residual
flips are bf16 near-ties (0.6B ≤0.125 nats, 4B ≤0.25) where vLLM's OWN one-shot
prefill argmax disagrees with its incremental decode (vLLM contradicts itself - no
forward bug, no single 16/16 decode target). Gate PASS = our token within 0.5 nats
of vLLM's teacher-forced argmax (strict where equal). With the RoPE cos/sin cache now
DEFAULT-ON (below) and goldens regenerated on the canonical `$HOME/cutlass-4.5.0`
build: **Qwen3-0.6B 16/16** (strict 10/16 + near-tie 6/16, max 0.0 nats) and the
**bigger-model complete-correctness proof Qwen3-4B** (BF16, 36 layers, GQA 32/8, hidden
2560 - a different config, same forward code) **16/16** (strict 11/16 + near-tie 5/16,
max 0.25 nats), both deterministic run-to-run (K=3 identical). Correctness is COMPLETE.
Its regression bar HOLDS on the canonical build: the two gate models stay token-exact
(27B `test_qwen27_paged_engine` **235/235** + 35B `test_qwen36_paged_engine` **315/315**),
unchanged by construction (the RoPE flip lives only in the Qwen3-dense TU).

**Mistral (`MistralForCausalLM`, Mistral-7B-v0.3) - CORRECTNESS COMPLETE, no speed
number (2026-07-23, `CLAIM-MODEL-MISTRAL` + `CLAIM-LOAD-SENTENCEPIECE`,
[spike](../.agents/specs/sweep-mistral.md), [tokenizer](../.agents/specs/sentencepiece.md)).**
Disposition: **NO throughput measured or claimed - SPEED PENDING** (a separate
increment). The **full paged-engine SACRED gate now RUNS + PASSES** - unblocked by the
new SentencePiece (Metaspace + byte-fallback) tokenizer (`LOAD-SENTENCEPIECE`, encode/
decode byte-exact vs vLLM 0.25.0). `test_mistral_paged_engine` drives the 16-prompt
battery through the paged `LLMEngine` and matches the vLLM 0.25.0 per-prompt greedy
oracle **16/16** (vLLM ALL-DETERMINISTIC over K=5 ⇒ STRICT bar): 15/16 strict
token-exact + 1/16 near-tie band (prompt[5]@tok10 our 1535 vs vLLM greedy 1281, but
vLLM's OWN teacher-forced argmax given our prefix == our token, gap **0.0000 nats**
≪ 0.5), 0 forward-divergent. Prompt tokenization byte-exact vs vLLM's own tokenizer
(BOS=1; `p{i}_prompt.i32` == our `EncodeWithSpecialTokens`). (Earlier W0-W3 also proved
the forward tokenizer-free at 30/30.) Mistral = the shared Qwen3/Llama dense forward
with NO new primitive (plain rope θ1e6, no qk-norm, untied lm_head, null sliding_window
all pre-existing); 3 new model files, ZERO shared-code edit. W2 loader validated on the
real 3-shard checkpoint (1541 assertions). CUDA `-Werror` 0 warnings; DSR 32 unchanged.
Row `ACTIVE`, not `DONE` (SPEED pending). Repro on dgx: capture the oracle with
`qwen3-oracle-capture.py --per-prompt --runs 5 --gpu-mem 0.40 --model mistralai/Mistral-7B-v0.3
--out-dir tests/parity/goldens/mistral_greedy_7b`, bootstrap our ids with
`VT_DUMP_IDS=1 ./build-cuda/tests/test_mistral_paged_engine` + `qwen3-neartie-gap.py`,
then `flock $HOME/gpu.lock ./build-cuda/tests/test_mistral_paged_engine`.

**SentencePiece tokenizer (`LOAD-SENTENCEPIECE`) - correctness gate, NOT a benchmark.**
NOT APPLICABLE for throughput (tokenization is host-side and negligible vs the model
forward). Correctness: `test_tokenizer_parity_mistral` **6/6, 421 assertions** byte-exact
vs HF `tokenizers` 0.22.2 (= vLLM 0.25.0 backend) over a 45-entry Metaspace/byte-fallback/
special-token corpus; SACRED cross-check vs vLLM's own `AutoTokenizer` 0/45 mismatch;
byte-level suites byte-identical (test_bpe 852, test_detokenizer 221, test_tokenizer_parity
1175, test_tokenizer_parity_deepseek 2461); ASan+UBSan clean on the tokenizer path.

**Qwen3-dense SPEED binding vs vLLM 0.25.0 production - same-session, matching-recipe
(2026-07-21, `Qwen3-4B` BF16).** SUPERSEDES the 2026-07-20 table below, whose vLLM
c1 TTFT (61.6 ms ⇒ "2.27×") and c8 ITL P99 ("4.3×") were **denominator / num-prompts
artifacts**: a fresh same-session vLLM capture gives c1 TTFT **~152 ms** and c8 ITL P99
**~130 ms**, and OURS BEATS vLLM on both TTFT axes and ties c1 ITL. Workload: random
in1024/out128 (range-ratio 0), `--ignore-eos`, closed-loop (`--request-rate inf
--max-concurrency C`), num-prompts 16 (c1) / 96 (c8). Denominator = vLLM PRODUCTION
(graphed/async default), `vllm serve` + `vllm bench serve`; ours = `examples/vllm-bench`
(RoPE-cache + FA2 prefill/decode default-ON, qkv-merge default-OFF), flashinfer-cutlass
build. Mean of 2 reps (<1% noise). Ratio = ours/vLLM (throughput ≥1 wins, latency ≤1 wins).

| axis | c1 vLLM | c1 ours | ratio | c8 vLLM | c8 ours | ratio |
|---|---|---|---|---|---|---|
| total tput tok/s | 192.1 | 189.0 | **0.98×** | 1472.3 | 1372.0 | **0.93×** |
| TTFT median ms | 152.0 | 136.5 | **0.90× ✓ WIN** | 382.7 | 144.0 | **0.38× ✓ WIN** |
| TPOT median ms | 45.95 | 46.6 | **1.01×** | 46.4 | 51.0 | **1.10×** |
| ITL P99 ms | 48.3 | 48.1 | **0.996× ✓** | 130.1 | 145.1 | **1.12×** |

**Verdict: c1 effective every-axis parity; c8 decode residual → `ACTIVE`.** At **c1**
all four axes are within ~1.5% with TWO wins (TTFT 0.90×, ITL 0.996×) and two near-ties
(tput 0.98×, TPOT 1.01×) - effective parity. At **c8** TTFT is a big WIN (0.38×) but the
DECODE residual persists: total tput **0.93×**, TPOT **1.10×**, ITL P99 **1.12×** (~7-10%).
**Grounded cause (nsys, this build):** the c8 decode step is **93% GPU-busy = compute-bound**
(not host-bound), dominated (~72%) by the small-M=8 `cutlass_80_wmma` projection GEMMs;
the gap is decode-GEMM efficiency, not launch overhead or a slow kernel. The **qkv-merge**
lever (single QKVParallelLinear GEMM + `QkvSplit`, mirroring vLLM) was **implemented and
measured NEUTRAL** here (c1 +0.8% / c8 +0.4% tput, within noise) - it does not cut decode
FLOPs, so it ships **DEFAULT-OFF** (`VT_QWEN3_QKV_MERGE=1` opts in; byte-affecting, flips
one 0.6B near-tie, 4B stays 16/16). Closing c8 needs a decode-GEMM/kernel-fusion
sub-campaign, not the qkv merge. Prefill/TTFT is **RESOLVED** - the earlier 2.27×/5.85×
TTFT "gap" was the bad denominator; ours now WINS TTFT at both concurrencies.
**cutlass verification (open question resolved):** 27B `test_qwen27_paged_engine` on the
flashinfer-cutlass build = **235/235** - the prior "flashinfer ⇒ 234/235" claim was a
build artifact. **Memory** (unified GB10): ours peak RSS ~8.5 GB; a strict peak-device
ratio is not measurable on unified memory. Evidence:
[bench-evidence/qwen3-4b-binding-20260721.log](bench-evidence/qwen3-4b-binding-20260721.log);
full recipe in the [parity ledger](../.agents/parity-ledger.md) 2026-07-21 binding row.

**Qwen3-dense TTFT levers - DevicePool (neutral) + RoPE cos/sin cache (opt-in), 2026-07-20
(`Qwen3-4B` BF16, base `812a57a`).** Two profile-#81 levers, measure-first. Recipe here:
`examples/vllm-bench --input-len 1024 --output-len 128 --concurrency {1,8}` (ours) vs
`vllm serve` + `vllm bench serve --dataset-name random --random-input-len 1024
--random-output-len 128 --random-range-ratio 0 --max-concurrency {1,8}` (graphed vLLM
0.25.0); idle box, 2 reps steady-state median. NOTE: this is NOT the `--ignore-eos`
closed-loop recipe of the FA2-prefill row above, so ABSOLUTE vLLM ratios differ (that
row's TTFT gap is larger); the recipe-robust signal is the **relative same-binary A/B**.

- **LEVER 1 - DevicePool extraction: byte-identical, MEASURED PERF-NEUTRAL.** Same-binary
  A/B (pooled vs baseline naive Alloc/Free), Qwen3-4B: c1 median TTFT 208 vs 209 ms, c8
  316 vs 317 ms, TPOT/tput within noise - the "~44% GPU-idle from cudaMalloc/cudaFree"
  premise is DISPROVEN (async scheduler overlaps the host-side alloc syncs). Kept as
  byte-safe hygiene + code sharing, not a TTFT lever.
- **LEVER 2 - RoPE cos/sin cache (`VT_QWEN3_ROPE_CACHE`, opt-in): the real TTFT lever.**
  Same-binary A/B (cache vs RopeNeox): **c1 median TTFT 209 → 135 ms (−35%); c8 316 → 144
  ms (−54%)**; TPOT/tput flat (prefill-only win). Under this recipe that is 0.87× (c1) /
  0.38× (c8) vs vLLM's median TTFT (135 vs 156 ms; 144 vs 383 ms) - at/beyond median
  parity - while total tput (0.92-0.97×) and c8 TPOT still lag. **Shipped DEFAULT-ON
  2026-07-20** (`VT_QWEN3_ROPE_CACHE=0` opts back to RopeNeox). The prior opt-in blocker -
  "the 1-ULP RopeFromCache FMA shift lands on the engine's FA2-split-KV near-tie
  NONDETERMINISM → flaky strict-anchor gate (RopeNeox 4/4, RopeFromCache flips)" - was
  GROUNDED AND DISPROVEN: the paged engine is byte-DETERMINISTIC run-to-run (RoPE-off gate
  4/4 + RoPE-on dumps md5-identical 3/3 + flipped-default gate 16/16 identical K=5), and for
  the short gate contexts `num_splits==1` so the split-KV combine kernel is never launched.
  The 1-ULP FMA shift is real but deterministic; it moves two genuine bf16 near-tie tokens,
  handled by regenerating the near-tie goldens on the canonical `$HOME/cutlass-4.5.0` build
  (0.6B max gap 0.0 nats, 4B 0.25 nats, 0 forward-divergent; gate re-passes 16/16 both).
  Repro + full recipe: [parity ledger](../.agents/parity-ledger.md) 2026-07-20
  RoPE-default-ON SPEED row.

**Strict-decode razor - investigation, no perf claim (2026-07-20).** An attempt to
tighten the Qwen3 near-tie band to strict token-exact 16/16 by routing d128 decode
through the vendored FA2 group-swap split-KV kernel (`LaunchDecodeFA2Bf16`) built
clean on dgx (`-Werror` 0-warn, d128 kernel compiled) but measured WORSE than the
CUDA-core baseline - STRICT 0.6B 12→**11/16**, 4B 10→**9/16** (both vs the same
`greedy_ids` oracle) - and was reverted. Source-confirmed root cause: GB10 is sm_121
so vLLM selects fa_version 2 and runs the unified FA2 **varlen** path
(`flash_attn_varlen_func`), NOT the group-swap `flash_attn_with_kvcache` kernel; the
group-swap changes the reduction order. Remaining bit-match target = an FA2 varlen
d128 decode ([spec](../.agents/specs/qwen3-decode-strict-bitmatch.md)).
`benchmark_binding=false` (investigation only, binary byte-identical to HEAD).

**FA2 VARLEN d128 decode - vendored + measured, correctness only, no perf claim
(2026-07-20).** The remaining bit-match target above is now implemented
(`LaunchDecodeVarlenFA2Bf16`, opt-in `VT_FA2_DECODE_QWEN3`, default OFF): the EXACT
vLLM varlen decode (plain varlen, no group swap, num_splits=exact heuristic). It
BIT-MATCHES vLLM's decode attention OUTPUT - teacher-forcing vLLM on our new-path
sequence, our token IS vLLM's argmax at gap **0.0000 nats** at all-but-the-near-tie
positions - but STRICT 16/16 is **bf16-tie-bounded and NOT reached**: FA2-varlen
0.6B **11/16** & 4B **9/16**, ONE WORSE than the CUDA-core fallback (12/16 & 10/16),
because the residual ≤0.375-nat ties are ones vLLM itself resolves differently
between its prefill argmax and its incremental decode. Op-parity byte-exact vs the
f32 reference (`test_ops_paged_attn` 23/23 cases, 454433 assertions, incl.
num_splits>1 split-combine); `compute-sanitizer memcheck` **0 errors**; `-Werror`
0-warn. **Regression 27B 235/235 + 35B 315/315 + default Qwen3 gate 16/16 UNCHANGED**
(d256 arms byte-identical, opt-in default OFF). Shipped opt-in; the near-tie-robust
gate stays the closure; the engine gate is NOT tightened to strict equality.
`benchmark_binding=false` (correctness campaign; the vLLM-throughput SPEED benchmark
remains the open Qwen3-dense deliverable).

Last updated: **2026-07-19**. **Both gate models bound at HEAD `786aa0e`** (fresh
fully-interleaved 3-rep grid, ZERO void, 12/12 binding-eligible both models): **27B
117/124** (parity holds - the 7 fails are c4 mean/median TTFT 0.911/0.950 + 5 ITL/TPOT
tail axes at 0.93-1.00, all noise-band; matches the regression-confirmed 118) and
**35B 70/124**. The 35B result is now DECISIVELY characterized: **decode is at-or-beyond
vLLM everywhere** - memory 4/4, and at c4/c8/c16/c32 the ONLY failures are the 4 TTFT
axes (mean/median/p90/p99 ttft 0.877-0.971) with every decode axis (tpot/itl/e2el/
throughput) PASSING; c1 (2/20) + c2 (0/20) are near-miss across the board (0.935-0.975)
but TTFT-led (at low concurrency TTFT dominates e2e with no queuing to amortize). **The
entire remaining 35B gap - and 27B's last residual - is PREFILL TTFT.** The GDN kernel-glue
wins (post-conv + gated-RMSNorm, both default-ON, byte-exact) held 70/124 without flipping
the TTFT axes → the TTFT gap is bigger than kernel-glue can close; the concurrency-dependence
(worse at c2-c8/c32 than c1) points at batched-prefill SCHEDULING, not single-stream kernel
latency. Active lever: prefill-TTFT full-step attribution (host-side / scheduling vs kernel).
Evidence `dgx:~/work/vllm.cpp-online-gate/evidence/786aa0e…/summary-{27,35}/`.

**Async robustness fix (2026-07-20, not a perf change).** A pre-existing
`async_scheduler` `num_output_placeholders >= 0` assertion could crash the engine at
concurrency-8 with short output lengths when `VT_ASYNC_RUNNER` is ON (the binding grid
runs async-ON). Root cause was a missing `discard_request_mask` in the model runner: it
emitted a sampled token for prefill-CHUNK requests, so under chunked-prefill + preemption
the async placeholder count underflowed. Fixed by mirroring vLLM (runner returns empty
tokens for still-prefilling requests; scheduler unchanged). Byte-identical on the
non-chunked gates (27B 235/235 + 35B 315/315); `vllm-bench` c8 + short-output + chunked +
KV-pressure with async ON now completes, `compute-sanitizer memcheck` 0 errors.

**Prefill-TTFT attribution CLOSED (2026-07-19, tasks #61/#62) - no further perf lever
banked.** Full-step nsys (both sides): 35B prefill is **99.4% GPU-BUSY** (not host/
scheduling - the concurrency-scheduling hypothesis was measured false; mixed steps are
98.5-99.6% busy, so mixed-step piecewise cudagraph was REJECTED - our C++ async runner
already keeps the GPU fed). ~71% of prefill is real compute at parity (Marlin MoE / FLA
GDN / fp8 dense / flash-attn); the residual is ~20% norm/quant/act "glue" we run as
separate kernels vs vLLM Inductor's fused chains, worth a **~3.5%/step ceiling @ c1**
(35B c1 190 vs 183 ms; 27B c1 is ours-faster). A one-fold fusion experiment
(`RmsNormGatedQuantFp8`, `a83d93a`, byte-exact default-ON) confirmed each clean fold is
~1% TTFT - real but bounded, and the big remaining glue kernels already match vLLM. So
prefill-TTFT is a compute-bound residual with no large closer; 35B is banked as
decode-at/beyond-parity + memory/throughput wins. The fusion work now proceeds as the
roadmap_v1 ORDER-1 **portable op-fusion framework** (`KERNEL-FUSION-FRAMEWORK`, spike
[portable-fusion-framework.md](../.agents/specs/portable-fusion-framework.md)) -
justified by EXTENSIBILITY + mechanical upstream-sync, NOT perf (the ceiling above is
the honest cap; benchmark disposition = the framework must be perf NEUTRAL-OR-BETTER,
not a speedup claim). **W0 landed (2026-07-19):** the first production adoption of the
seam - the 35B post-attention layernorm routes its plain add+residual+RMSNorm through
`vt::FusedChain(kFusedAddRmsNorm)` (`VT_FUSED_CHAIN_ADOPT`, default-ON, `=0` rollback),
behaviour-preserving and byte-identical to the prior hand-call (35B 315/315 + 27B
235/235 token-exact BOTH arms, byte-exact composite==interp==golden incl. the production
H=2048 shape, memcheck 0 errors). **Benchmark disposition: perf-NEUTRAL by construction**
(the default Tier-0 composite dispatches to the same primitive the hand-call used) ⇒ no
re-grid; this is structural plumbing, not a speedup.

**W1 landed (2026-07-20):** the `FusedRecipe` POD is generalized (full
activation/norm/quant/rope opcodes + an indexed operand table) so all five quant-fused
W2 target chains are expressible as `constexpr` recipes whose Tier-0 composite is
byte-exact to the standalone-op sequence the model hand-calls today. Infrastructure
only - NO model call site changed. Byte-exact `test_ops_fused_chain` CPU 196 + CUDA 361
assertions, memcheck 0; token-exact regression (the W0-adopted site) 35B 315/315 + 27B
235/235 on both `VT_FUSED_CHAIN_ADOPT` arms. **Benchmark disposition: perf-NEUTRAL**
(no call site changed) ⇒ no re-grid. Build-hygiene follow-up: a pre-existing
CPU-only `-Werror=unused-function` (`FuseSigmoidGateQuantEnabled`, used only under
`#ifdef VT_CUTLASS_NVFP4`) fixed with `[[maybe_unused]]` - no behavior change,
keeps the CPU CI build green.

**W2 landed (2026-07-20):** the bespoke hand-fused ops are MIGRATED to
`vt::FusedChain(recipe)` at six production call sites (`kSiluMulFp4Quant` MoE
down-proj, `kSigmoidGateFp4Quant` full-attn o-proj, `kRmsNormGatedQuantFp8` ×2 GDN
out-proj, `kRmsNormQuantFp8` input-layernorm, `kAttnQkNormRopeGate` ×2 attn preamble),
behind `VT_FUSED_CHAIN_ADOPT` (`=0` restores the exact prior hand-calls). Each recipe
binds (via a new backend-agnostic `FusedRecipe.fast_op`) to its EXISTING single-launch
fused kernel, so `FusedChain(recipe)` dispatches to the SAME fast kernel the model
called directly before migration. **Benchmark disposition: perf-NEUTRAL by construction**
- the same fast `OpId` dispatches (no extra kernel, no per-forward getenv/alloc), so
this is NOT a speedup and needs no re-grid. Gates (dgx `~/work/vllm.cpp-fusion-w2`,
prod flags, clean CUDA `-Werror` 0-warn): byte-exact `test_ops_fused_chain` CPU 228 +
CUDA 420 assertions (fast == Tier-0 composite == unfused-op-sequence golden per recipe),
memcheck 0; token-exact 27B 235/235 + 35B 315/315 on BOTH `VT_FUSED_CHAIN_ADOPT` arms.

**W3 landed (2026-07-20) - the mechanical-upstream-sync PROOF (NOT a perf change).**
A NEW, previously-unported vLLM fusion pass - `SiluMulFp8StaticQuantPattern`, the
static-per-tensor-FP8 activation variant of `ActivationQuantFusionPass`
(`act_quant_fusion.py:81` → `_C.silu_and_mul_quant`) - was ported as ONE `constexpr
FusedRecipe kSiluMulQuantFp8` + its byte-exact test, touching EXACTLY TWO files
(`include/vt/recipes.h` + `tests/vt/test_ops_fused_chain.cpp`): no kernel, no dispatch,
no model-site edit, no new primitive (its Tier-0 composite realizes through the existing
`vt::MoeSiluMul` + `vt::QuantFp8Static` ops). **Benchmark disposition: NOT APPLICABLE /
perf-neutral** - the recipe is DECLARED, not wired into any model, so the engine is
byte-for-byte unchanged; W3 is the extensibility/mechanical-sync deliverable (a new
upstream fusion PR = one declaration), explicitly not a perf lever (§ the ceiling above).
No re-grid. Gates (dgx `~/w3_fusion_sync/build-w3`, prod flags, clean CUDA `-Werror`
0-warn): byte-exact `test_ops_fused_chain` CUDA 432 assertions (the new `kSiluMulQuantFp8`
FusedChain==composite==golden arm; CPU 228 unchanged), memcheck 0; no token regression
27B 235/235 + 35B 315/315.

**W4 landed (2026-07-20) - the backend-additivity PROOF, closing the W-series (NOT a perf
change).** The additivity claim ("a new backend registers `kFusedChain` once and inherits
the ENTIRE catalog correct, zero per-recipe work") is made EXECUTABLE. Approach (spec §10
W4): treat the existing CPU backend AS the 'second backend' relative to CUDA (no mock
`DeviceType` - that would edit the core enum + every switch, ironically non-additive). New
test `tests/vt/test_fused_chain_additivity.cpp` enumerates the whole catalog (all 7 recipes)
and in ONE generic loop asserts each runs byte-exact on the CPU backend via the Tier-0
composite: 4 CPU-full end-to-end + 3 CPU-prefix (static-fp8 terminal `vt::QuantFp8Static`
CUDA-only per §3b/§6 - byte-exact prefix, plus the full composite asserted to THROW on CPU,
documenting the backend-negotiated tail). Additivity evidence: `recipes.h` grew 1→6→7
recipes while `FusedChainCompositeImpl` stayed ONE per-opcode function (12 `FOp::` cases) and
the CPU/CUDA `kFusedChain` registration stayed ONE line each; `cpu_ops.cpp` never `#include`s
`recipes.h`; W3's whole new `kSiluMulQuantFp8` appears in zero backend TUs - inherited free.
**Benchmark disposition: NOT APPLICABLE / perf-neutral** - test-only + record change, the
engine binary is byte-identical; no re-grid. Gates (dev box `build-w4-cpu`, clean CPU
`-Werror` 0-warn - no `.cu`/header/CUDA TU touched, so CPU is the valid green): `test_fused_chain_additivity`
17 assertions 0 failed, `test_ops_fused_chain` 228/228 unchanged; engine byte-identical ⇒
27B 235/235 + 35B 315/315 structurally unchanged; memcheck N/A. **KERNEL-FUSION-FRAMEWORK
ORDER-1 proof milestone DONE (W0+W1+W2+W3+W4).** Named future/HW-blocked (does not gate the
milestone): the Tier-1 single-pass perf interpreter for the quant chains (composite-only
today) and per-recipe fast kernels. The Metal (2026-07-22) and Vulkan (2026-07-22)
realizations are DONE at skeleton level - both register one `kFusedChain` interpreter and
inherit the whole catalog, both tiers checked against the CPU oracle.

### Accelerator-seam `S4` - LinearMethod / QuantizationConfig seam (2026-07-23, `CLAIM-BACKEND-SEAM-S4-1`) - byte-identical, NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - structural refactor, no performance claim.**
`S4` of the accelerator-seam audit lands the `LinearMethod`/`QuantizationConfig` seam
(`include/vllm/model_executor/layers/{linear.h,quantization/…}`, mirroring vLLM
`linear.py:141-230` + `base_config.py:20-229`): scheme selection is centralized into a
policy object chosen ONCE from the checkpoint (the dense model's projections route through
`method.Apply()`), and 16 shared-layer `device == kCUDA` availability gates in
`qwen3_5.cpp` + 2 in `dense_nvfp4_gemm.h` + 1 in `qwen3.cpp` became
`vt::OpRegistered(op, device)` op-table queries. Those ops are registered ONLY for
`kCUDA`, so on the production build the query is bit-for-bit the old `== kCUDA` test, and
`OpRegistered` is a memoized relaxed-atomic load after first resolution (the audit's
"resolve once, cache" pattern) so the decode hot path is unaffected. The engine is
byte-identical by construction; **no re-grid, no perf number is owed or claimed.**
**DSR 86 → 67** (`kcuda` 44→25); baseline lowered in the same commit; ratchet + 24-case
mutation suite green. Gates (dgx `~/s4-build-src/build-cuda`, prod flags
`-DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a -DVLLM_CPP_TRITON=ON`, clean CUDA
`-Werror` **0 warnings**; standalone under `flock $HOME/gpu.lock`): **byte-identical
27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6 · Qwen3-dense 16/16 · OPT 6/6 (96/96 tokens) ·
DeepSeek-V2 8/8**, goldens content-hash unchanged before/after; new
`test_linear_method` (scheme×device selection + bf16 apply) + quant suites (nvfp4,
ct-nvfp4-emulation, gguf-dequant, gguf-keep-quant) green; full CPU ctest green
(`test_openai_conformance` a `-j4` contention flake, passes standalone). The delicate
27B-W4A4 / fp8-recipe device gates (ops dual-registered on CPU or realized via a recipe)
were deliberately LEFT for `S6`, which needs `S5`'s reference-tier fallback first -
converting them would not have been byte-identical.

### Accelerator-seam `S5` - portable reference tier (2026-07-23, `CLAIM-BACKEND-SEAM-S5-1`) - infrastructure, NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - vt-runtime infrastructure, no performance claim
is owed, and a fired reference tier is BY DEFINITION a missing native kernel that MUST be 0
in any performance arm.** `S5` of the accelerator-seam audit lands the portable reference
tier that mirrors vLLM `custom_op.py:138 forward_native`: on a UNIFIED-MEMORY device the CPU
kernel installs LAZILY (first `GetOp` miss) as a negative-priority (`-1000`) `vt-cpu-ref`
`OpProvider`, so a backend lacking a native kernel for an op runs the portable CPU reference
instead of throwing. A native kernel (priority ≥ 0) always wins, and the install is
lazy-on-miss, so a backend that never misses (CUDA on the gate models) is byte-identical -
no provider added, no table changed. **DSR unchanged at 67** (nothing under `src/vllm`/
`include/vllm` - the DSR scope - changed); ratchet + 24-case mutation suite green. Gates
(dgx, prod flags `-DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a -DVLLM_CPP_TRITON=ON`,
clean CUDA `-Werror` **0 warnings** + clean CPU `-Werror`; standalone under
`flock $HOME/gpu.lock`): **byte-identical 27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6 ·
Qwen3-dense 16/16 · OPT 6/6 · DeepSeek-V2 8/8**, goldens content-hash unchanged. Hardware-free
proof: new `tests/vt/test_reference_tier.cpp` (5 cases) runs `vt::Relu` on a fake UNIFIED
device with ZERO native kernels and gets the bit-identical CPU answer, and proves a fake
DISCRETE device is refused (`GetOp` still throws - no CPU kernel on true device memory). The
S4-deferred fp4/fp8 gates were reassessed and stay deferred (their ops are dual `kCUDA`+`kCPU`
/ CPU-only / recipe-realized, so `OpRegistered` ≠ `== kCUDA` on a CPU device - not
byte-identical); S5 is `S6`'s safety net, not the conversion, so **DSR does not drop here
and that is expected.** The one number this row will ever own is a CORRECTNESS gate, not a
speed gate - OPT token-exact on Metal via the tier - and observability (`GetReferenceTierHits`)
enforces that any performance arm reads 0 reference-tier hits.

### Accelerator-seam `S6` - convert the S4-deferred fp4/fp8 device gates (2026-07-23, `CLAIM-BACKEND-SEAM-S6-1`) - ASSESSED NO-OP, NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - assessment/records-only, no code changed, no
performance claim owed.** `S6` was to convert the S4-deferred fp4/fp8 `device==kCUDA`
gates behind `S5`'s reference tier, driving DSR down. It lands **no byte-identical
conversion**: every remaining deferred fp4/fp8 gate bottoms out at a **dual-registered**
(CPU+CUDA) bespoke op (`kMatmulNvfp4Fp4`/`kScaledFp4Quant`/`kSiluMulFp4Quant`/
`kSigmoidGateFp4Quant`/`kRmsNormQuantFp8`/`kRmsNormGatedQuantFp8` - none CUDA-only, unlike
S4's convertible gates), so `vt::OpRegistered(op,dev)` is TRUE on `kCPU` and the class-A
swap flips the CPU reference/emulation path to a different numeric path - the task's
"irreducible residual" rule. S5's reference tier does not change this (those CPU kernels
are present natively, never a `GetOp` miss). **No `src/`/`include/`/test byte changed, DSR
stays 67, no baseline moved.** Since the binary is untouched, every SACRED gate is
byte-identical by construction (27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6 · Qwen3-dense
16/16 · OPT 6/6 · DeepSeek-V2 8/8 · Llama 16/16, no golden touched); no dgx build or GPU
run is owed. Genuine byte-identical unlock re-scoped to `S3` (Platform capability fields
mirroring `supports_fp8`/`cutlass_fp4_supported`) + `S7` (layer extraction). Bare-RC
checkers on the dev box: `check-device-leakage.py` RC=0 (DSR 67 == baseline 67), 24-case
mutation suite 24/24, `check-agent-record.py` + `check-doc-checkpoint.py` RC=0.

### Accelerator-seam `S3` - Platform capability fields, the byte-identical fp4/fp8 unlock (2026-07-23, `CLAIM-BACKEND-SEAM-S3-1`) - byte-identical, NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - structural/decoupling refactor, no
performance claim.** `S3` mirrors vLLM's `Platform` capability surface
(`interface.py:914,933,977,1058` + `nvfp4_utils.py:56 cutlass_fp4_supported` +
`is_device_capability_family:441`) as base-`false` virtuals with CudaPlatform
answers, then replaces **12** deferred `device==kCUDA`/graph gates in `qwen3_5.cpp`
with the capability predicate (7 fp4-activation → `cutlass_fp4_supported()`, 3
fp8-fused → `supports_fp8()`, 2 decode-graph → `support_static_graph_mode()`). This
is the byte-identical move `S6` §11 re-scoped the fp4/fp8 gates onto: a capability
answers the base `false` off CUDA - exactly what `device==kCUDA` returned - where
S6's `OpRegistered` answered TRUE on `kCPU` for the dual-registered ops. The
selected kernels are unchanged (proven on GB10: `cutlass_fp4_supported()`/
`supports_fp8()`/`support_static_graph_mode()` all `true`), so **no re-grid, no
perf number is owed or claimed.** **DSR 67 → 55** (`kcuda` 25→13); baseline lowered
in the same commit; ratchet + 24-case mutation suite green. Gates (dgx CUDA + dev
CPU, clean `-Werror` **0 warnings**; standalone under `flock $HOME/gpu.lock`, one
big-model at a time): **byte-identical 27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6
· Qwen3-dense-32B 16/16 · OPT 6/6 · DeepSeek-V2 8/8 · Llama 16/16**, no golden
regenerated. `test_platform` CUDA-leg + family cases green; seam/quant nets green;
`compute-sanitizer memcheck` **0 errors** on `test_platform` + `test_ops_nvfp4_fp4`.
The residual class-D residency/stream/FA2-dtype/merged-layout sites and the
`#ifdef VT_*` build gates in `dense_nvfp4_gemm.h` were LEFT (no clean byte-identical
capability mirror - that is `S7`), which is why the as-built DSR is **55**, not lower.

### Accelerator-seam `S7` - extract the last shared-layer device coupling; DSR to its floor (2026-07-23, `CLAIM-BACKEND-SEAM-S7-1`) - byte-identical, NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - structural/decoupling refactor, no
performance claim; each converted predicate resolves to exactly the former
`device==kCUDA` value on GB10, so no re-grid is owed.** `S7` hoists **all 23**
remaining runtime `kCUDA`/`is_cuda()` leakage sites in the shared model layer onto
platform/backend capabilities: new `Platform::needs_weight_staging()` (the CUDA
device-resident staging policy - residency 704/725/2500/5298, merged/packed GDN
2656/2752/2780/3151, dense-weights direct-load; NOT `is_unified_memory()`, which
would flip GB10) and `Platform::supports_fa2_attention()` (FA2 dtype 3647/3653), new
`Backend::SupportsAuxStream()` (MoE aux-stream 4292), plus reuse of S3's
`supports_fp8()` (merged-fp8-QKV 1272), `cutlass_fp4_supported()` (fp4 warmup +
dense/moe fp4 decode-graph gates), `support_static_graph_mode()` (DeepSeek/Qwen3-MoE
decode-graph gates), `is_integrated_gpu()` (runner device combine/scatter 702/1080),
and `vt::OpRegistered(kMoeGroupedGemmBf16)` (a CUDA-only op, S4 pattern - DeepSeek
GroupedMoeEligible). Each is byte-identical because GB10 is the only registered CUDA
platform and every predicate returns `true` there / base `false` elsewhere - exactly
what the device test answered. **DSR 55 → 32** (`kcuda` 13→0, `is_cuda` 10→0,
`cuda_inc` 0, `vt_ifdef` 32); baseline lowered in the same commit; ratchet + 24-case
mutation suite green. **32 is the honest floor:** every remaining site is a
`#ifdef VT_*` compile-time gate for a kernel that only builds on one GPU family
(legitimately irreducible); the shared model layer now holds **zero** runtime device
tests. The audit's "<10" target is thus not reachable - build gates are the floor.
Gates (dgx CUDA + dev CPU, clean `-Werror` **0 warnings**; standalone under
`flock $HOME/gpu.lock`, one big-model at a time): **byte-identical 27B 235/235 · 35B
315/315 · Qwen3-Coder 6/6 · Qwen3-dense-32B 16/16 · OPT 6/6 · DeepSeek-V2 8/8 ·
Llama 16/16**, no golden regenerated (goldens content-hash unchanged before/after).
New `test_platform` cases (needs_weight_staging / supports_fa2_attention, per leg) +
Backend `SupportsAuxStream` cases (`test_backend`, `test_cuda_backend`) green;
seam/quant nets green; `compute-sanitizer memcheck` **0 errors** on the touched
device path. Bare-RC checkers: `check-device-leakage.py` RC=0 (DSR 32 == baseline
32), 24-case mutation suite 24/24, `check-agent-record.py` + `check-doc-checkpoint.py`
RC=0. This completes the accelerator-seam campaign's runtime-decoupling arc
(86 → 67 → 55 → **32**, its irreducible build-gate floor).

**27B has reached effective performance
PARITY-OR-BETTER with vLLM v0.25.0.** Two independent fully-interleaved exact-grid
reruns on the full production default set (async + vendored Triton GDN decode cubin
+ bit-identical fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU) - `9ecd9d0`
(114/124) and `f0fb727` (111/124; adds the bit-identical conv-update + FP4/SiLU
flips - the 111-vs-114 delta is pure noise-band coin-flip, which calibrates the
noise floor) - establish by **two-grid per-axis totality: 110 axes pass in BOTH
grids, 5 are noise-band coin-flips that flip between grids (at-parity by totality)
→ 115/124 effective parity (REGRESSION-CONFIRMED 2026-07-19: a fresh 27B grid at `fcfde41`, after all 35B-era changes - Platform seam, GDN conv, FA2 preamble - binds 118/124 with throughput winning every concurrency, NO regression), and 9 fail in both.** They SUPERSEDE `a875397`
(52/124), `246a23c` (49/124) and `3f256ab` (55/124), all retained immutable;
`benchmark_binding` refers to the `9ecd9d0`+`f0fb727` two-grid totality. Evidence
`dgx:~/work/vllm.cpp-online-gate/evidence/{9ecd9d0…,f0fb727…}` (immutable; g1
ratios.json `8c81083e…`, g2 `38296763…`), ZERO void, 12/12 binding-eligible; our
arm ran pure defaults, vLLM its production async default; correctness holds
throughout (full default set 27B 235/235 + 35B 315/315).

**The 9 persistent residuals are all the low-concurrency-median edge of one
determinism tradeoff, and we are NET-POSITIVE on every one.** Our synchronous
deterministic forward keeps co-admitted requests in lockstep where vLLM's
async-runtime jitter de-phases them: this costs slightly on low-concurrency
*median* decode/TTFT (c8 mean/median/p99 itl+tpot, c4 mean/median ttft, c16/c32
median_itl) but WINS the corresponding *tail* and the same metric at higher
concurrency - c8 p99_itl 0.86 at c8 but **1.055 at c16 / 1.078 at c32**; c4
median_ttft 0.95 but c4 **p90/p99 ttft 1.009/1.013** and c8/c16/c32 mean_ttft
**1.030/1.100/1.136**. No axis is meaningfully or closeably slower; the only "fix"
(async-forward jitter) forfeits the tail + high-conc + throughput wins with no
throughput basis (vLLM's own async is −0.7%) - net-negative. A literal per-run
124/124 is gated by ~5 noise-band coin-flips + this favorable tradeoff, not any
real deficit. Root-cause: `.agents/specs/c8-p99-itl-tail-2026-07-18.md`.

**+62 axes - a decode-kernel-efficiency close.** Per concurrency: mem **4/4**, c1
**20/20**, c2 **20/20**, c16 **19/20**, c4 18/20, c32 18/20, c8 15/20. The lever:
a family of **bit-identical (0-ulp) fast decode kernels** - each reproduces its
shipped reference's exact float-op order (so the fast set yields identical logits
and can never cross the 27B tok6 razor near-tie) while vectorizing memory access.
`RmsNormRowFastKernel` (2.41× isolated, `348d12d`) closed the c2 lanes entirely
(3→20); `RmsNormGatedRowFastKernel` (2.04× at c16, `9ecd9d0`) closed the uniform
c16 floor (6→19); it was templated `<Tin,Tout>` (2026-07-19) so the 35B MoE **f32**
gated norm - previously excluded to the slow shipped kernel (77.6 ms / 3.3% of 35B
prefill) - now also takes the bit-identical fast path (1.55× isolated, token-exact 315/315).
This CONFIRMS the measured attribution: the uniform decode
deficit was the norm/quant/act kernel glue, and vectorizing it bit-identically
closed it. Correctness holds - the full default set is 27B 235/235 + 35B 315/315.

**The 10 remaining axes** are near-parity or low-concurrency noise: seven are the
batch-independent decode floor at **0.987-0.999** (c8 mean/median itl+tpot, c16
median_itl 0.9983, c32 median_itl 0.9879 + median_tpot 0.9989 - a whisker under the
strict ≥1.0 bar); two are **c4 TTFT** (mean 0.906 / median 0.943, the bimodal
low-concurrency arrival lottery); one is the **c8 p99_itl 0.86** wave-boundary
prefill-co-schedule tail (ours 557 vs vLLM 479 ms), now ROOT-CAUSED (2026-07-18,
[spec](../.agents/specs/c8-p99-itl-tail-2026-07-18.md)) as the **honest residual -
irreducible-as-mirrored**: at c8, ours' deterministic synchronous forward keeps
co-admitted requests in byte-identical lockstep (22 uniform 840 ms two-full-prefill
stalls/rep, lockstep 0.77) where vLLM's async-future runtime jitter de-phases them
(11 + graded 440/660, lockstep 0.51); but ours BEATS vLLM on this same tail at c16
(p99 1.055) and c32 (1.078) - vLLM's jitter spawns extreme 900-band outliers ours
lacks - so it is the trailing edge of the per-step determinism that wins higher
concurrency + throughput, not a capability gap. Scheduler + async placeholder are
byte-identical (no vLLM policy to mirror); the only structural fix (true
async-forward executor) risks the c16/c32 wins with no throughput basis, and forced
de-phase is a fake fix - so NO code change. Closing path for the seven decode-floor
axes: the batch-independent glue is now all landed BIT-IDENTICAL and DEFAULT ON
(2026-07-18, `CLAIM-CONV-UPDATE-FAST-1`) - the two FP4-quant/SiLU fast kernels
flipped ON (parity-enabler policy, `861b518` bodies unchanged, `=0` rollback) and
the new GDN decode conv-update fast kernel (`CausalConv1dUpdateFastKernel`,
`VT_CONV_UPDATE_FAST` default ON, 0-ulp bit-identical, **isolated 1.92×** at the 27B
c16 shape via a 2D grid removing two int64 div/mod per thread + a register-cached
state row; DGX byte-exact 330/330, full default set 27B 235/235 + 35B 315/315). The
next binding grid runs the full bit-identical fast-decode stack by default and
re-measures the in-situ effect; characterize the c4 TTFT lottery vs run-noise.
`benchmark_binding=false` for these levers - no isolated speed credit; the grid owns
the in-situ number.

### 27B sigmoid-gate → o_proj fold (2026-07-19, `CLAIM-SIGMOID-GATE-FOLD-1`) - byte-exact, perf-NEUTRAL, opt-in

Component (NOT binding). Folds the standalone full-attention sigmoid output gate
(`attn*sigmoid(gate)`) into the o_proj NVFP4 activation quant on the 27B
true-W4A4 path - one fused `vt::SigmoidGateFp4Quant` kernel replacing
`SigmoidGateBf16` (bf16 `gated` intermediate) + `ScaledFp4Quant`, mirroring vLLM's
Inductor `triton_poi_fused_mul_scaled_fp4_quant_sigmoid` and reusing the
`SiluMulFp4Quant` precedent. Only the 27B (W4A4) o_proj quantizes its activation;
the 35B (W4A16-Marlin / fp8 o_proj) reads bf16 activations, so the fusion is inert
there. `VT_FUSE_SIGMOID_QUANT=1` opt-in (default OFF).

Disposition: **byte-exact + perf-NEUTRAL ⇒ landed OPT-IN (default OFF).** DGX GREEN
(`~/work/vllm.cpp-sigmoid-fold`, production flags CUTLASS sm120a + Marlin + FA2
sm_121a + Triton AOT, clean CUDA `-Werror` **0 warnings**, one flock): byte-exact
op test `sigmoid_gate_fp4_quant` **14/14** (CPU f32/bf16 + CUDA f32, 3 shapes,
fused == `SigmoidGateBf16`+`ScaledFp4Quant`); 27B `test_qwen27_paged_engine`
**235/235 token-exact BOTH arms** (fused default-ON build + `VT_FUSE_SIGMOID_QUANT=0`
fallback); 35B `test_qwen36_paged_engine` **315/315** (fusion inert). In-situ 27B
same-binary interleaved TTFT A/B (input-1024, output-8, greedy, 3 reps/arm): **c1
med OFF 419.6 vs ON 419.0 ms (−0.15 %), c2 792.9 vs 792.7 ms (−0.03 %)** - both
within ±0.5 % rep-to-rep noise, prefill tok/s indistinguishable (~914 c1 / ~1345
c2). The full-attn o_proj is a small slice of 27B prefill (dominated by GDN + MoE +
the QKV/gate/up GEMMs), so the one-launch + bf16-round-trip saving sits below the
noise floor. Kept opt-in per "token-exact but not measurably faster". Repro: build
branch with the flags above, `flock /tmp/gpu`, `test_ops_nvfp4_fp4 -tc="*sigmoid_gate_fp4_quant*"`
+ 27B/35B engine gates + `env VT_FUSE_SIGMOID_QUANT={0,1} vllm-bench --input-len 1024`.

### 35B GDN out_proj gated-RMSNorm → fp8-quant fold (2026-07-19, `CLAIM-GDN-OUT-FP8-FUSE-1`) - byte-exact, measured-faster (small), DEFAULT ON

Component (NOT binding). Folds the 35B GDN out_proj's static W8A8 fp8 activation
quant INTO the gated-RMSNorm output store - a new `vt::RmsNormGatedQuantFp8`
(CPU + CUDA) emits the fp8 activation directly (fed to `MatmulFp8CutlassPreQuantD`),
removing the standalone `QuantFp8Static` pass AND the bf16 gated-norm output the
split path writes then re-reads. It is the gated sibling of the existing
`RmsNormQuantFp8Row` fusion, mirroring vLLM's Inductor fusion of the fla
`layernorm_guard` RMSNormGated epilogue with the following RowParallelLinear's
static-fp8 activation quant. Only the 35B GDN out_proj is fp8 (27B's out_proj is
W4A4 fp4), so the fold is **35B-only** - no 27B greedy-razor exposure.
`VT_GDN_OUT_FP8_FUSE=0` rolls back to the split path.

Disposition: **byte-exact + measured-faster (small) ⇒ DEFAULT ON** (like
`GdnPostConvFast`; the parity-enabler policy). **DGX GREEN** (`~/work/vllm.cpp-gdn-fp8-fuse`,
production flags CUTLASS sm120a + Marlin + FA2 sm_121a + Triton AOT, clean CUDA
`-Werror` **0 warnings**, one flock):

- **Byte-exact.** CPU op test `rmsnorm_gated_quant_fp8` (silu + sigmoid gate) -
  fused == `RmsNormGated(bf16)`+static fp8 quant, bit-for-bit; full `test_ops_rmsnorm`
  36/36. The fused CUDA kernel reproduces `RmsNormGatedRowFastKernel`'s exact 0-ulp
  variance reduction, then the same bf16-round + `__nv_cvt_float_to_fp8` the split
  `QuantFp8Static` applies. `compute-sanitizer memcheck` **0 errors** (fused ON).
- **Token-exact.** 35B `test_qwen36_paged_engine` **315/315** on the default-ON
  build, on `VT_GDN_OUT_FP8_FUSE=0` rollback, AND on the pre-flip `=1` arm; 27B
  `test_qwen27_paged_engine` **235/235** (inert - 27B out_proj is fp4).
- **Isolated per-kernel** (nsys `--cuda-graph-trace=node`, 35B prefill c1 input-1024,
  30 GDN layers): OFF = `RmsNormGatedRowFast` 4.989 ms (30) + the 30 out_proj
  `QuantFp8Static` launches ≈ 1.45 ms = **6.43 ms**; ON = `RmsNormGatedQuantFp8RowFast`
  **4.584 ms** (30) - the fused fp8 1-byte store is even cheaper than the unfused
  gated norm's bf16 store, and absorbs the quant. **−28.7 % on the chain (≈ −1.85 ms /
  1024-tok prefill)**, but only **~0.28 %** of the ~655 ms total prefill GPU.
- **In-situ 35B TTFT A/B** (input-1024, output-8, greedy, 3 reps/arm interleaved,
  same binary): **c1 median OFF 175.0 → ON 172.6 ms (−1.4 %, 3/3 reps favor ON),
  c2 333.2 → 329.0 ms (−1.3 %, 3/3), c8 854.9 → 851.2 ms (−0.4 %, 2/3; one ON rep an
  outlier)**; prefill throughput +1.0 % consistently; mean TTFT c1 −1.2 % / c2 −1.4 %.
  c8 was measured with `VT_ASYNC_RUNNER=0` because a **pre-existing** async-scheduler
  assertion (`async_scheduler.cpp:65 num_output_placeholders >= 0`) crashes c8 +
  short-output - unrelated to this fold (which only touches a GDN kernel); a separate
  bug to track.

**Honest verdict:** a REAL, byte-exact prefill-TTFT win, but **SMALL** (~1 % of TTFT,
~0.28 % of prefill GPU). It does NOT flip the 35B binding TTFT axes alone (they are
3-14 % off). Fusion IS covering genuine prefill-TTFT, but per-fold gains are ~1 %, so
closing the 35B prefill gap requires STACKING many such folds - this is the first of
that push. Repro: build branch with the flags above, `flock /tmp/gpu`,
`test_ops_rmsnorm -tc="*gated_quant_fp8*"` + 27B/35B engine gates +
`env VT_GDN_OUT_FP8_FUSE={0,1} vllm-bench --input-len 1024 --output-len 8`.

### Lever-3 eager Marlin repack (2026-07-19) - VERIFIED already-at-load-time, skipped

Scoping premise: the first-touch Marlin repack was 42 % of a COLD (no-warmup) 35B
prefill trace; does it leak into the binding grid's measured c1/c2 TTFT? **Verified
NO - already moved to load-time.** `Qwen3_5Model::PrepareMarlinResident` builds all
Marlin residents (routed experts `BuildMoeMarlinResident`, shared experts, lm_head)
at engine init, called from the `GPUModelRunner` constructor (`runner.cpp:302/321`)
BEFORE any warmup/serving - mirroring vLLM's `process_weights_after_loading`. The
forward first-touch path (`if (!mr.ready) BuildMoeMarlinResident`) is a dead
fallback in production. Empirically an un-warmed 35B c1 bench (6 reqs, input-1024)
shows an elevated first-request TTFT (mean 247.9 / median 178.1 / p99 575.5 ms) -
but that ~400 ms first-request delta is general warmup (FP4 autotune / plan-cache /
CUDA-graph capture / page-in), NOT the multi-second expert repack (a repack leak
would add seconds, not ~0.4 s, and is provably at construction anyway). The binding
grid warms with a 1×1024-token request before timing, excluding even that general
first-request cost, so the repack is OUTSIDE the measured c1/c2 window ⇒ lever
skipped (no implementation needed).

### 35B FP8 merged-QKV projection (2026-07-19, `CLAIM-FP8-MERGED-QKV-1`) - token-exact, perf-NEUTRAL, opt-in

Component (NOT binding). Extends the fp4-only merged-QKV fusion to the 35B FP8
W8A8 full-attn path: ONE fp8 GEMM over the N-concatenated Q/K/V operand + per-
column dequant (`vt::MulColVecF32`) replacing 3 separate per-shard GEMMs (10
attn layers, 30→10 attn-QKV GEMMs/step). `VT_FP8_MERGED_QKV` opt-in.

Disposition: **token-exact + perf-NEUTRAL ⇒ landed OPT-IN (default OFF).** CPU
gates GREEN (`test_ops_glue` 10/10 incl. 2 new byte-exact MulColVecF32,
`test_ops_fp8_cutlass` 6/6, `test_ops_matmul` 7/7, clean `-Werror`). DGX GREEN
(`~/work/vllm.cpp-fp8-merged-qkv` @ `e9ce593`, production flags, CUTLASS_OK,
clean CUDA `-Werror` 0 warnings, one flock): 35B `test_qwen36_paged_engine`
**315/315 token-exact both arms** (default OFF + `VT_FP8_MERGED_QKV=1`) + 27B
`test_qwen27_paged_engine` **235/235 both arms** (inert); `compute-sanitizer
memcheck` (35B ON) **0 errors**. The merged path is
proven to fire (its strided value view triggered - and was fixed for - a
downstream `cast_bf16` contiguity check, which an inert merge could not produce).
In-situ same-binary interleaved TPOT A/B (input-1024/output-128, greedy, 3
reps/arm): c1 ~0% (ON 13.94 vs OFF 13.93 ms), c2 −0.5% (17.75 vs 17.84), c4
−0.5% (21.40 vs 21.50), c8 ~0% (31.23 vs 31.21) - all ≤0.9% and within the
~0.3-0.4 ms rep-to-rep noise (no clear win; the ~20-GEMM-launch saving over 10
attn layers sits below the decode-step noise floor - the step is dominated by
30 GDN layers + MoE). Kept opt-in per "token-exact but not measurably faster".
Repro: build branch `kernel-gemm-fp8-merged-qkv` with
`-DVLLM_CPP_CUTLASS_DIR=$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DVLLM_CPP_TRITON=ON`,
`/tmp/dgx_gate.sh` + `/tmp/dgx_ab2.sh`, one `flock /tmp/gpu`. Follow-up: GDN
qkvz/BA fp8 merge (30 layers - larger launch slice - may clear the noise floor).

### 35B PREFILL GDN conv-fwd + fused post-conv kernel efficiency (2026-07-18, `CLAIM-GDN-PREFILL-CONV-1`) - bit-exact, conv modest win / post-conv opt-in

Component (NOT binding): the prefill `causal_conv1d` forward
(`CausalConv1dFwdRegKernel`, `VT_CONV_REG` DEFAULT ON) and the fused post-conv split
(`GdnPostConvSplitKernel`, `VT_GDN_POSTCONV_SPLIT` OPT-IN) mirror vLLM's FLA
register-resident sliding-window conv (`causal_conv1d.py:397-452`) and per-V-head
post-conv grid (`fused_gdn_prefill_post_conv.py:57-149`). Both **BIT-IDENTICAL
(0-ulp)** to the shipped tiled/megablock kernels (DGX byte-exact reg==tiled +
split==megablock **268** GPU assertions + full GDN 3081/3081; memcheck **0**; token
27B **235/235** + 35B **315/315** on the final defaults). nsys per-call A/B on
`nvidia/Qwen3.6-35B-A3B-NVFP4` (`--cuda-graph-trace=node`, evidence
`dgx:~/work/prefill-attr-conv-35b`): **conv c1 337.1→321.1 µs (−4.7%), c6
1036.4→960.3 µs (−7.3%)** - consistently faster ⇒ DEFAULT ON; post-conv split c1
−3.8% / c6 **+4.7%** - near-neutral (GdnPostConv time is q/k-L2-norm-dominated, not
the V-megablock) ⇒ OPT-IN. TTFT c1 A/B (3 reps): reg-ON 186.23 vs reg-OFF 186.96 ms
(**−0.39%, within run-noise** - the conv kernel is ~2.5% of GPU time). **Finding:**
the kernels are BANDWIDTH-bound (conv ~215 GB/s of GB10's ~273 peak, f32); the
structural mirror is a real-but-modest win, and the larger vLLM conv gap is TRAFFIC
(bf16 post-conv activations, `VT_GDN_IN_BF16`, task #40 sibling), not structure.
`benchmark_binding=false`; the 35B binding grid re-measures the in-situ c1-c4 effect.

### GDN prefill fused post-conv fast kernel (2026-07-19, `CLAIM-GDN-POSTCONV-FAST-1`) - bit-exact, DEFAULT ON, measured win BOTH models

Follow-on to the split's near-neutral disposition above. A fresh production-path
nsys profile (`--cuda-graph-trace=node`, VLLM_CPP_TRITON=ON, input-1024) confirmed
the GDN chunk **compute** (delta_h / chunk_o / kkt / recompute_w_u) runs the vendored
FLA Triton AOT cubins by default (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
`chunk_fwd_kernel_o`, `recompute_w_u_fwd_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`
in the trace) - i.e. **at FLA parity by construction**. The #1 remaining NON-AOT
GDN-specific kernel on BOTH models is the fused post-conv prep, and the shipped
default is the single-megablock `GdnPostConvKernel` (the split `VT_GDN_POSTCONV_SPLIT`
measured neutral/slower - 27B 175.6→201.4 µs SLOWER, 35B ~0%).

`GdnPostConvFastKernel` (`VT_GDN_POSTCONV_FAST`, DEFAULT ON, `=0` rollback) keeps the
shipped megablock grid `(T, Hk+1)` (the low-launch layout the split failed to beat)
and makes two provably **BYTE-IDENTICAL** changes for the Dk==Dv==128 gate dims:
(a) 128 threads/block instead of 256 - every lane owns one q/k element, so the
128-wide L2-norm tree is the 256-wide tree minus a leading `+partial[t+128]` over the
always-zero upper half (a no-op), same summation over the same squared values, +better
reduction occupancy; (b) the V copy (the largest memory pass) is staged in 128-bit
transactions (raw `int4` when conv/out dtypes match; the SAME per-element
`__bfloat162float`/`__float2bfloat16` converts otherwise) instead of one per element.
No arithmetic is reordered. Grounded in FLA `_fused_post_conv_kernel`
(`fused_gdn_prefill_post_conv.py:57-149`, grid `(cdiv(L,BLOCK_T), H+HV)` per-head).

**Isolated nsys per-call (`--cuda-graph-trace=node`, input-1024, evidence
`dgx:~/work/vllm.cpp-gdn-chunk/prof`):** 27B `unsloth/Qwen3.6-27B-NVFP4`
**175.6 → 133.0 ms (−24.3%)**; 35B `nvidia/Qwen3.6-35B-A3B-NVFP4`
**93.5 → 70.3 ms (−24.8%)** (kernel totals over 10-prompt runs). **In-situ TTFT A/B
(input-1024, output-8, greedy, 3 reps/arm, interleaved default-vs-`=0`):** 27B c1
418.9→414.1 ms (**−1.14%**), c2 805.6→795.1 ms (**−1.31%**); 35B c1 232.0→230.3 ms
(**−0.72%**), c2 387.7→383.9 ms (**−0.99%**) - every FAST rep below its paired BASE.
**Bit-exact:** 27B `test_qwen27_paged_engine` **235/235** + 35B
`test_qwen36_paged_engine` **315/315**, token-exact on BOTH the default (fast) and
`=0` (megablock) arms; CPU predicate `test_gdn_prefill_conv` 28/28; clean CUDA
`-Werror` 0 warnings. Per the parity-enabler policy (byte-exact ⇒ never-slower +
token-safe, measured faster on BOTH models) it ships **DEFAULT ON**. `benchmark_binding=false`;
the binding grid re-measures the in-situ effect.

### Prior binding narrative (`246a23c`, superseded 2026-07-17, retained)

The nominal 49 < 55 vs `3f256ab` was a
STRUCTURAL RECOMPOSITION, not a plain regression: memory (**4/4**), c1 (**20/20**),
and **every** TTFT axis swept clean for the first time, and the entire
remaining failure mass was the decode-coupled family at c2-c32 (TPOT/ITL/E2EL means
**2.2-6.5%** slower, throughput inversely coupled) plus two ITL tail anomalies
(c8 p99, c32 p90) - the latter now resolved by the async default at `a875397`.

The `246a23c` binary is **substantially different from `3f256ab`**, not a
re-measurement of the same code: it carries the correctness slot-fix (`c172336`),
the windowed-load release (`cb2d310`, which flips both memory axes to PASS), merged
qkvz (`45f9e6d`), and packed GDN decode as the default. The order-0
[packed-decode leaf](../.agents/specs/gdn-packed-decode.md)
(`KERNEL-GDN-PACKED-DECODE`) is **CLOSED on EQUIVALENCE** (`DONE`, owner
`e47b4d6`). Correctness is immutable at `f344dec` (default + rollback each
**235/235 + 16/16**); structure is accepted from `7ff713e`/`24cea4f` (packed
**915** nodes vs rollback's **963**, the exact 48-for-96 GDN substitution, all
remaining topology invariant); the earlier c16 HTTP-500 slot defect was fixed
test-first (request-identity keying) and proven at `c172336`.

**HONEST throughput regression vs `3f256ab`.** Ours lost **−2.67% total throughput
at c16** (790.63 vs 812.30 tok/s) and **−3.64% at c32** (1081.10 vs 1121.95) while
vLLM held (+0.52% / +0.31%), so the old c16/c32 total-throughput WINS
(1.0279/1.0394) are GONE. **HYPOTHESIS (clearly labeled, unproven):** the `3f256ab`
binary silently carried the GDN slot-sharing defect - two concurrent long requests
could share ONE recurrent-state slot at high concurrency, reducing effective state
bandwidth - which the `c172336` correctness fix removed; the fix may have traded
that inflated high-concurrency throughput away, alongside every other change
between the SHAs (qkvz, windowed-load, packed-default). An **era A/B** (`3f256ab`
binary vs `246a23c` binary, interleaved c16 legs, same corpus/box) is **RUNNING now
on dgx** as the diagnostic to isolate it; it is in-flight and this record does not
wait on it.

**Source+arithmetic grounding (2026-07-16, `CLAIM-GDN-BA-ROUNDING-1`; DGX A/B is the
final arbiter).** A two-round `9ad8fb7`→`c172336` bisect (`825→790` tok/s, TPOT
`159→167 ms`, both `c172336` arms equal incl. `VT_GDN_PACKED_DECODE=0`) now grounds
the hypothesis and **refutes a "per-step host-machinery" reading**: every new host
cost is per-STEP over n≤32 (O(n²) uniqueness ≤1024 compares, string-keyed remap ≤32
ids ⇒ **<~15 µs/step**, ~3 orders below the 8 ms), and on CUDA `IndexedGdnStateIoEnabled`
defaults ON so there is **no graph→eager flip / no row-copy fallback** - the decode
device path is byte-identical to `9ad8fb7` except the state-index VALUES. `9ad8fb7`
keyed the compact pool on the mamba block-id, which collapses to the shared null
block-0 for long sequences → every long c16 sequence shared ONE slot (silent
corruption). De-collapsing to distinct per-sequence slots restores the recurrent-state
DRAM traffic every correct GDN decode must pay (real 27B ssm row 48·128·128·F32 =
**3 MB/slot/layer**; ~16 distinct rows × ~36 GDN layers ≈ **~3.4 GB/step** ÷ ~273 GB/s
≈ **8-12 ms** - the measured regression). vLLM pays the same traffic (its 159 ms TPOT
includes it), so `825` is unrecoverable (bug artifact) and **~790 is the CORRECT floor,
already 0.995× vLLM at c16**; the residual 790→794 is decode-**kernel** efficiency
(a separate lever), NOT a host cost. A perf-neutral validation/allocation cleanup was
landed this change (`ValidateGdnStateIndices` O(n²)→O(n) + `VT_GDN_VALIDATE`; reused
remap scratch); it recovers ~0 of the 8 ms by design. DGX to confirm: the same c16 A/B
(expect ≈790, NOT 825) + a decode state-I/O byte-count nsys.

**W1D3 / G3 closed on evidence-totality (compacted).** The packed-decode
component campaign proved **EQUIVALENCE - no stable regression on any axis**
across **eight sealed components** + the **8-pair locked c16 A/B** (`00bf484`:
paired mean **−0.205% ± 0.30, <1σ**; cuBLASLt algo selection process-deterministic
→ algo-lottery REFUTED) + the **24-window trace** (**packed is GPU-cheaper**:
kernel compute −1.30..−1.58%/step, GDN+BA −296 µs/window, no attributable
packed-side cost). Packed stays the default (`VT_GDN_PACKED_DECODE=0` rollback);
**no `complete-pass` marker exists and no packed speed credit is claimed**. Merged
**qkvz** (`KERNEL-GEMM-BF16` W2) landed test-first (one BF16 `in_proj_qkvz` GEMM
per GDN layer, `VT_GDN_MERGED_QKVZ=0` rollback) and its DGX gates closed green at
`45f9e6d` (structural −48 BF16 GEMMs/window). All three - packed-default, qkvz, and
the windowed-load release - are in the `246a23c` binding binary, so the
**authorized exact-grid rerun has now RUN** (this scoreboard). The equivalence-audit
pin `--mamba-ssm-cache-dtype float32` is wired into the vLLM arm and recorded in
this run's evidence. Detailed per-seal chronology and evidence SHAs live in the
append-only [ledger](../.agents/parity-ledger.md) and
[state log](../.agents/state.md). 35B stays blocked until 27B reaches 124/124.

**Decode recurrence perf lever - MEASURED codegen-bound → vendored Triton cubin,
now DEFAULT ON (2026-07-16).** The named +2.06 ms/step recurrence-tiling gap (correct-state c16
traces: ours 21.31 vs vLLM 19.24 ms/step; ~83% vs ~92% of ~273 GB/s) was first
ATTEMPTED via a register-resident hand port of vLLM FLA's single-warp
`num_stages=3` packed-decode kernel (`fused_recurrent.py:256-336`) - the DGX
proof FAILED (oracle boundary FAIL; c16 A/B reg-tile 700.5-701.4 vs legacy
793.6-794.5 tok/s, TPOT 190.5 vs 166.5). **Phase-1 `cuobjdump -res-usage` on the
compiled cubins at the c16 shape named the cause** (root
`~/work/vllm.cpp-gdn-recurrence/phase1`): vLLM's FLA decode cubin holds the
register-resident `[BV=32,BK=128]` fp32 state tile at **REG:205, STACK:0
(0 spills)**; the byte-for-byte hand port hits **REG:255 (hard cap) + STACK:48
(SPILLS)** - register allocation / codegen, not a config we mis-set (legacy hand
kernel is REG:56/8-warp/0-spill at ~83% BW). DECISION: the sanctioned vendored
Triton cubin (`gdn_decode_h48`, 27B-only) behind `VT_GDN_PACKED_DECODE_TRITON`
(**default ON since the 2026-07-16 flip** - MIRROR policy: it is vLLM's exact
token-identical FLA kernel and vLLM runs it by default; `=0` rolls back to the
hand `GdnPackedDecodeKernel` in the same binary, which also remains the fallback
for any non-27B shape).
DGX landing gates (GB10 sm_121a, root `~/work/vllm.cpp-gdn-recurrence`, one flock/series):
AOT-vs-legacy-vs-CPU op test 28/28, full `test_ops_gdn` 49/49, oracle boundary
12/12 (legacy path bit-exact preserved), **27B model gate 235/235 token-exact
with the Triton decode path ON**, compute-sanitizer 0 errors/0 leaks, default-off
gate 235/235. **c16 A/B (`VT_GDN_PACKED_DECODE_TRITON=1` vs the-then-default, interleaved
3 pairs + w0 cold discard, one flock, root `ab-decode-triton`): triton [817.51, 821.06, 822.55] vs legacy [813.77, 815.62, 815.30] tok/s - paired mean **+5.48 tok/s (+0.67%)**, monotone (+3.74/+5.44/+7.25), 3/3 pairs positive; mean TPOT triton [161.04, 160.49, 160.35] vs legacy [162.09, 161.65, 161.93] = **-1.26 ms (-0.78%)** (median TPOT -1.13 ms); w0 cold-discard (triton 821.48/160.44) excluded.**
ACCEPTANCE MET (oracle PASS + consistent c16 TPOT improvement + no throughput regression).

**DEFAULT FLIP ON (2026-07-16, `CLAIM-GDN-DECODE-TRITON-FLIP`).** Per MIRROR
policy the default flipped OFF→ON, joining the sibling GDN Triton kernels
(`VT_GDN_DELTAH/CHUNKO/WU_TRITON`, all default ON). Test-first: new default-ON
pure-header predicate `src/vt/cuda/gdn_packed_decode_triton.h` + CPU flag test
(RED→GREEN 10/10). **35B: no specialization added** - it never selects packed
decode (MoE, excluded by the dense-only `ShouldUsePackedGdnDecode`) and the
launcher guard rejects its `Hv=32` shape anyway (clean fallback, so a 35B cubin
would be dead code). **Flip gates - ALL EIGHT PASS, exit 0 each** (root
`~/work/vllm.cpp-gdn-decode-triton-flip` `gates.verdict`/`gates.out`,
build `-DVLLM_CPP_TRITON=ON` + CUTLASS-4.5.0/nvcc-13.0, configure-log CUTLASS/FA2
lines verified, one flock): 27B DEFAULT (now Triton) **235/235** + `=0` rollback
**235/235**; 35B DEFAULT **315/315** + `=0` rollback **315/315** (inert); AOT op
test **28/28** (default fires cubin, `=0` fires legacy); full GDN **49/49
(2,343/2,343)**; oracle boundary **12/12**; memcheck **28/28, 0 errors**. No new
A/B (9dd7d3f's stands). **The next binding grid runs the Triton decode path by
default** (production-default set: async scheduling ON + Triton decode cubin ON +
RMSNorm-fast ON since the 2026-07-17 c2 preflight win); no separate flip speed
credit is claimed. CPU gates GREEN
(`test_ops_gdn` 45/45, `test_gdn_packed_decode_triton` 10/10, clean -Werror).
Build repair (2026-07-16): the vendored-cubin landing left the launch-counter
helper defined unconditionally while its only caller is Triton-gated, breaking
the default Triton-less CUDA build under `-Werror` (`-Wunused-function`); the
helper is now guarded behind `VLLM_CPP_TRITON` (no runtime change in any
configuration; found by the block-table cleanup agent's DGX gate build). Repro
commands in
[the packed-decode spec](../.agents/specs/gdn-packed-decode.md#decode-recurrence-perf-lever--measured-codegen-bound-vendored-triton-cubin-2026-07-16).

**Blast-radius caveat (correctness).** The c16 slot defect predated its
validator and was independent of `VT_GDN_PACKED_DECODE` (both arms hit the same
remap). The compact slot pool was introduced at `66715e1` (2026-07-05); the
uniqueness validator only at `f344dec` (2026-07-14). Binaries in between -
including the `3f256ab` binding grid and earlier c16/c32 campaigns - would have
**silently** run two or more long concurrent sequences on ONE GDN recurrent
state slot (cross-request state corruption) rather than crashing. Low-concurrency
16/16 correctness gates do not surface it; high-concurrency runs measure
throughput, not token correctness, so any such prior c16/c32 per-token output
correctness is **suspect**. This same slot-sharing defect grounds the labeled
throughput-regression hypothesis above: `3f256ab`'s c16/c32 throughput may have
been inflated by two long sequences sharing state bandwidth, which the `246a23c`
correctness fix removed. The `246a23c` binding binary carries the fix (its
correctness is sound); the era A/B is the diagnostic.

## Binding 27B online gate

Workload equivalence of the two arms is AUDITED and accepted
([audit report](../.agents/specs/benchmark-equivalence-audit-2026-07-15.md)):
batch cap (max_num_seqs=32, zero preemption), token budget (2048 + chunked
prefill), context, greedy sampling, corpus bytes, KV/conv/SSM dtypes (SSM is
FP32 on BOTH arms - the vLLM arm now carries the audited explicit
`--mamba-ssm-cache-dtype float32`, surfaced in its `non-default args:` startup
log), FP4 kernel family, FA2, and graphed decode all match; the client commands differ
in exactly one token (result directory). The one material engine difference
is vLLM's Inductor prefill fusion + piecewise cudagraphs - its production
config, i.e. the protocol-correct denominator.

| Item | Binding value |
|---|---|
| Model | Qwen3.6-27B NVFP4 |
| Hardware | NVIDIA GB10 / DGX Spark, sm_121a |
| vllm.cpp source | `9ecd9d0` (representative grid) + `f0fb727` (confirming grid); full bit-identical fast-decode default set: async + vendored Triton GDN cubin + fast RMSNorm + gated-RMSNorm + conv-update + FP4/SiLU |
| Reference | vLLM v0.25.0, tag `702f4814fe54fabff350d43cb753ae3e47c0c276`, FlashInfer 0.6.13; vLLM arm carries `--mamba-ssm-cache-dtype float32` |
| Workload | Cache off, input 1,024 → output 128, greedy, closed loop, c1/c2/c4/c8/c16/c32 |
| Repetitions | Three interleaved repetitions per point, per grid (rep1 ours/vLLM … rep3; one whole-model flock); TWO independent grids for the totality verdict |
| Evidence completeness | 12/12 performance groups, 2/2 memory groups, 124/124 axes eligible (0 void), each grid |
| Evidence root | `dgx:~/work/vllm.cpp-online-gate/evidence/{9ecd9d0…,f0fb727…}` |
| Artifact SHA-256 | g1 `summary-27/ratios.json` `8c81083e…`, `all-runs.json` `c7e4a831…`, `manifest.json` `a3871da2…`; g2 ratios.json `38296763…` |
| Disposition | **EFFECTIVE PARITY-OR-BETTER: two-grid totality 115/124** (110 pass-in-both + 5 noise-band coin-flips; g1 114/124, g2 111/124). Supersedes `a875397` (52), `246a23c` (49), `3f256ab` (55) |

Ratios are direction-normalized: throughput is ours/vLLM, latency is
vLLM/ours, and **1.0 or higher passes**. Values are medians of three
interleaved repetitions (representative grid `9ecd9d0`).
Output/request/input throughput share the total ratio.

| Concurrency | Axes passing | Total tok/s ours / vLLM (ratio) | Mean TTFT (v/o) | Mean TPOT (v/o) | Mean E2EL (v/o) |
|---:|---:|---:|---:|---:|---:|
| 1 | **20/20** | 86.05 / 82.32 (**1.0453×**) | 1.0671× | 1.0444× | 1.0453× |
| 2 | **20/20** | 159.68 / 158.03 (**1.0105×**) | 1.0460× | 1.0079× | 1.0104× |
| 4 | 18/20 | 292.34 / 290.31 (**1.0070×**) | 0.9058× | 1.0111× | 1.0048× |
| 8 | 15/20 | 508.77 / 505.46 (**1.0066×**) | 1.0519× | 0.9980× | 1.0056× |
| 16 | **19/20** | 801.76 / 789.16 (**1.0160×**) | 1.0886× | 1.0054× | 1.0150× |
| 32 | 18/20 | 1095.01 / 1076.25 (**1.0174×**) | 1.1339× | 1.0031× | 1.0171× |

**We now beat vLLM on total throughput at EVERY concurrency** (1.007-1.045×;
the prior `246a23c` binding lost at 0.98×) and on mean TTFT/TPOT/E2EL at all but
the low-concurrency-median points noted below. The bit-identical fast-decode
kernel stack is what moved throughput from 0.98× to 1.007-1.045×.

| Memory axis | Ours | vLLM | Normalized ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 24,879,201 KiB | 28,184,400 KiB | 1.132850× | **PASS** |
| Peak RSS | 24,881,800 KiB | 28,563,020 KiB | 1.147948× | **PASS** |
| Peak GPU memory | 40,996 MiB | 70,531 MiB | 1.720436× | **PASS** |
| Peak `MemAvailable` drop | 68,346,844 KiB | 80,660,556 KiB | 1.180165× | **PASS** |

**Failing-axis composition (current `9ecd9d0`+`f0fb727` binding).** All four
**memory** axes PASS (windowed-load, ours peak PSS 24.88 GB vs vLLM 28.18 GB);
**c1 20/20, c2 20/20**, and we beat vLLM on **total throughput at every
concurrency** (1.007-1.045×). The 9 persistent fail-in-both-grids axes are all the
low-concurrency-median edge of one determinism tradeoff - our synchronous
deterministic forward loses slightly on low-conc *median* decode/TTFT (c8
mean/median/p99 itl+tpot, c4 mean/median ttft, c16/c32 median_itl) but WINS the
corresponding *tail* and the same metric at higher concurrency: c8 p99_itl 0.86 at
c8 but **1.055 at c16 / 1.078 at c32**; c4 median_ttft 0.95 but c4 **p90/p99 ttft
1.009/1.013** and c8/c16/c32 mean_ttft **1.030/1.100/1.136**. No axis is meaningfully
or closeably slower (root-cause `.agents/specs/c8-p99-itl-tail-2026-07-18.md`). The
narrative below this line is the SUPERSEDED `246a23c` (49/124) analysis, retained
for the era A/B and W3-discriminator history; its numbers do not describe the
current binding.

<details><summary>Superseded `246a23c` failing-axis analysis (retained history)</summary>

All four **memory** axes PASS for the first time
(the windowed-load release, now binding - ours peak PSS 24.88 GB vs vLLM 28.18 GB).
**c1** sweeps 20/20 and **every TTFT axis** (mean/median/p90/p99) passes at every
concurrency (zero failing TTFT). The 75 failing axes are the decode-coupled family
at c2-c32 only: throughput (total/output/request/input), TPOT, ITL, E2EL (means and
most tails), driven by decode being **2.2-6.5% slower** - mean TPOT ratios
0.9779 / 0.9694 / 0.9445 / 0.9532 / 0.9597 (ours 109.85 / 116.35 / 131.41 / 167.46 /
245.58 ms vs vLLM 107.42 / 112.79 / 124.12 / 159.63 / 235.68), worst median TPOT
0.9348 at c8. Two ITL tails stand out as anomalies beyond that band: **c8 p99_itl
0.5599** (853.34 vs 477.81 ms) and **c32 p90_itl 0.7925** (706.80 vs 560.15) -
both DISCRIMINATED (2026-07-16, CPU-only wave-boundary diff;
[spec](../.agents/specs/tail-stall-analysis-2026-07-16.md)) to async-runtime
output-timing phasing (W3), **NOT** a scheduler divergence: driving the REAL
vLLM sync `Scheduler` and async `AsyncScheduler` through the identical
wave-boundary script (`tools/bench/scheduler_wave_diff.py`, golden
`tests/fixtures/scheduler_wave/wave_script_oracle.json`) yields BYTE-IDENTICAL
per-step composition - both pack the 2048 budget as 1024 + chunk at the stall
step (~860 ms) - and our C++ schedulers reproduce it exactly
(`tests/vllm/v1/test_scheduler_wave.cpp`, 3 cases / 44 asserts). H-B/H-C are
dead, H-A already dead (`89b329e`). The ~860 vs ~500 ms magnitude gap is the
async depth-2 output-timing/overlap regime (which CPU simulation of the async
driver does NOT reproduce - it also budget-packs to 2048), so the axes are
expected to close only under W3-on (implemented + now DEFAULT-ON since the
2026-07-17 flip, `CLAIM-ASYNC-SCHED-W3`); per MIRROR policy the fix IS
async-on, no invented single-prefill cap. The empirical confirmation **RAN and
CONFIRMED the mechanism** (2026-07-17, the W3 async TTFT-premium discriminator,
`CLAIM-W3-ASYNC-DISC`, [spec + full tables](../.agents/specs/w3-async-ttft-discriminator-2026-07-16.md);
one flock, 42 legs, token gates 6/6 @ `6ea7856`, evidence
`dgx:~/work/vllm.cpp-w3-discriminator/6ea7856…`; `benchmark_binding=false`, no
speed credit): under W3-on (`VT_ASYNC_RUNNER=1`) **both anomalies flip** - c8
p99_itl 856.8→**527.4 ms** (ratio 0.552→**0.897**, in the 15 % band) and c32
p90_itl 698.7→**534.4 ms** (0.791→**1.048**, ours now beats vLLM) - and the
~500 ms single-prefill band appears (c8: 54 events vs ZERO sync). The
campaign's vLLM v0.25.0 self-A/B (async ON vs `--no-async-scheduling`,
arm log-confirmed) also settles the TTFT question: **vLLM's own async pays
+26/+31/+28 % mean TTFT at −0.7 to −0.9 % throughput** (−2.6 to −4.3 ms TPOT)
vs its own sync at c8/c16/c32 - the premium is inherent to depth-2 scheduling
and upstream ships it ON anyway; ours-W3on reproduces vLLM-async's deltas and
spike structure within noise (no engine divergence; the mis-cited "vLLM async
~2005 ms" was corrected - binding vLLM async c16 TTFT is ~2848 ms vs ours sync
~1990). Axis arithmetic vs the production bar (18 axes × c8/c16/c32):
strict-PASS 14→15/54, flips up c8/c16 p99_tpot + c32 p90_itl, flips down only
the noise-scale c8 mean/p99 TTFT (0.9951/0.9940) ⇒ **W3-ON nets positive
as-is; the W3 default is now FLIPPED ON** (2026-07-17, `CLAIM-ASYNC-SCHED-W3`,
`ENG-ASYNC-SCHED` DONE, owner `6ea7856`; `VT_ASYNC_RUNNER=0` / `VT_ASYNC_SCHED=0`
roll back). DGX-re-confirmed TOKEN-NEUTRAL (evidence `dgx:~/work/vllm.cpp-async-flip`,
CUTLASS+FA2 hard-verified, one flock): all three async arms bit-identical; the
shipping default (async ON + RMSNorm-fast OFF) is **27B 235/235 + 35B 315/315** with
the "Asynchronous scheduling is enabled (mcb=2)" log, both rollback arms 235/235 +
315/315 "disabled". **TTFT means will RISE into vLLM's async envelope BY DESIGN**
(+26-31 %, the mirrored depth-2 trade) - the next binding grid runs async by default
and its TTFT must not be misread as a regression. The same DGX gate incidentally
caught a pre-existing RMSNorm-fast 27B token divergence (`VT_RMSNORM_DECODE_FAST`
default-ON `696a991`, gated only on `paged_FORWARD`): fast-ON 234/235 vs fast-OFF
235/235 vs the pip-vLLM oracle, async-independently - the `VT_RMSNORM_DECODE_FAST`
default was rolled back to OFF, then the 2026-07-17 numerics rework FIXED the
token-exactness (real `cub::BlockReduce` = the eager oracle's exact reduction;
235/235 + 315/315 fast-ON) and the 2026-07-17 c2 preflight A/B delivered the awaited
in-situ win, so the default is now ON - see the RMSNorm decode-fast section below. The
decode body (tokens 16-111) is at parity with ZERO mid-sequence stalls;
capping the per-event stall at one prefill was measured-sufficient to flip both
axes to PASS (counterfactual ratios 0.87-0.96 / 0.93-1.11). The
old c16/c32 total-throughput WINS are GONE (see the honest regression above);
c16/c32 total throughput now barely misses (0.9953 / 0.9985). No 35B performance
command is authorized until the 27B result reaches 124/124.

**Next levers (roadmap order-0).** The era A/B + four-round bisect is COMPLETE
and VALIDATED (probe root `~/work/vllm.cpp-c16-era-ab/20260716`; attribution
`6dd24df`, validation 2026-07-16): the c16/c32 "regression" vs `3f256ab` was
corruption-subsidized state bandwidth in pre-fix binaries (collapsed slots =
1/16th the GDN state traffic); the honest correct-state c16 floor is
**~790-799 vs vLLM 794**, both model gates 235/235, and no recoverable host
cost exists (new machinery ≤15 µs/step; O(n) validation landed). Pre-fix GDN
kernel evidence (H1d ranking, B=2 traces) is contamination-suspect for the
state path. Active levers now: (a) **GDN decode kernel efficiency** - MEASURED
2026-07-16 by fresh correct-state c16 kernel traces (ours nsys node trace of
`build-fix2`/`6dd24df`; vLLM torch-profiler at `--mamba-ssm-cache-dtype float32`;
root `dgx:~/work/vllm.cpp-gdn-stateio-trace/20260716`, `SUMMARY.json`). The
~8 ms/step wall gap (c16 TPOT 159.6 vs 167.5) decomposes as **~4.65 ms GPU-busy
+ ~3.25 ms host/idle** (ours ~79.2% vs vLLM ~80.2% busy). Per-c16-decode-step
GPU time (ms, ours/vLLM/Δ): GDN packed recurrence 21.31/19.24/**+2.06**;
RMSNorm(129) 2.006/0.391/+1.62; RMSNorm-gated 0.403/fused/+0.40; FP4-quant
0.641/0.342/+0.30; SiLU-mul 0.630/0.369/+0.26; GDN conv-update 0.584/0.432/+0.15;
GEMM+MoE+attention 106.79/106.71/**+0.08 (parity)**; TOTAL busy 132.70/128.05.
**Verdict:** state-I/O is only ~26% of the gap - decode state r/w is FUSED into
the recurrence on BOTH sides (ours runs NO separate decode gather/scatter; those
kernels are prefill-only), layout is identical `[slots,HV,V,K]` fp32; the busy
gap is ~2.06 ms recurrence TILING + ~2.6 µs·k norm/quant/act glue. Named lever:
port vLLM's register-resident single-warp `num_stages=3` FLA packed-decode tiling
(`fused_recurrent.py:282-336`) into `GdnPackedDecodeKernel` (ours reaches ~83%
of ~273 GB/s peak vs vLLM ~92%) ⇒ ~+2 ms/step. The **norm/quant glue is a
kernel-EFFICIENCY gap, NOT a fusion gap** ([reconciliation
2026-07-16](../.agents/specs/decode-norm-quant-fusion-reconcile-2026-07-16.md)):
vLLM does NOT fuse rmsnorm+fp4quant in this config - the `3f256ab` dumped Inductor
body stores bf16 after add+RMSNorm and calls `scaled_fp4_quant.out` separately,
`fuse_norm_quant=False`, and the fresh trace shows count parity (vLLM
`cvt_fp16_to_fp4` 144/win == ours 144 `ScaledFp4Quant`/step, both 129 rmsnorm);
ours already matches that structure and already mirrors the ONE real fusion
(silu+fp4quant, `VT_FUSE_SILU_QUANT`). The residual (rmsnorm 391 vs 2006 µs) is
cross-profiler-confounded (nsys node vs torch CUPTI; the 2026-07-14 rescan already
called the +1.81 ms residual a cross-profiler artifact). The `KERNEL-EW-NORM-ACT`
**Phase-1 same-profiler adjudication** (2026-07-16, nsys pure-kernel BOTH sides,
isolated, at the real 27B decode shape M×H=**5120**; evidence
`dgx:~/work/vllm.cpp-ewnorm-phase1`, [spec](../.agents/specs/rmsnorm-decode-fast-2026-07-16.md))
removes that confound and CONFIRMS the lever: ours `RmsNormRowKernel`
**8.44-8.53 µs/launch** vs vLLM's `triton_red_fused…rms_norm` **2.37-2.68 µs**
= **3.18-3.56×** (honest Δ ≈ **0.77 ms/step** at both c2 and c16; c2's is ~33% of
the ~2.4 ms c2 decode gap). The confound was only in-situ (ours in-situ nsys 15.5
µs is ~1.84× contention-inflated over the 8.46 µs isolated); the isolated gap is
real. PORTED test-first as `RmsNormRowFastKernel` (`VT_RMSNORM_DECODE_FAST`,
**DEFAULT ON / `=0`-rollback**). **BIT-SAFETY rework (2026-07-17,
`CLAIM-EW-NORM-ACT-3`, evidence `dgx:~/work/vllm.cpp-ewnorm-bitsafe`):** the kernel's
output is now BIT-IDENTICAL (0-ulp) to the shipped `RmsNormRowKernel` - the 235/235
through-stack bit-reference - by reproducing its exact float op sequence: residual
add `bf16(f32(x)+f32(res))`, variance in shipped's exact kBlock=256 strided-partial
+ shared-memory-tree ORDER (a 1024-thread vectorized Pass 1 stages each element's
f32 square to shared memory, then 256 threads run the shipped tree), and
`inv=1.0f/sqrtf`; only the element-independent normalize pass is vectorized. So
`fast+cubin ≡ shipped+cubin ≡ 198` BY CONSTRUCTION - removing the ≤1-ulp near-tie
flip that had forced two prior reverts (a0013a2 234/235; a875397 combination
233/235 vs `want_emu`). DGX proof (clean `-Werror` build, CUTLASS sm120a NVFP4 +
FA2 sm_121a hard-verified, one flock): `test_cuda_ops` decode-fast fast==shipped
**0-ulp BIT-EXACT** (assertion tightened from ≤1-ulp; 132/132), full 432/432;
production default (unset = fast+cubin+async ON) **`test_qwen27_paged_engine`
235/235** (token 6 = 198) + **`test_qwen36_paged_engine` 315/315**, both `=0`
rollback arms 235/235 + 315/315; flag test inverted RED→GREEN default-ON 10/10.
**Perf - the win SURVIVES bit-identity** (the 1024-thread memory passes keep the
decode parallelism): isolated nsys **3.55 µs vs shipped 8.58 µs (2.41×)** at
M=16×H=5120, and in the real 27B engine forward RmsNorm median **4.38 vs 16.13 µs
(3.68×)**, total RmsNorm GPU time 48.3 vs 55.4 ms - strictly less GPU work with
identical bits, so it cannot regress in-situ. The perf case was already accepted
(the predecessor's c2 preflight measured **+1.446% tput / −0.887% pooled-median
TPOT**, `CLAIM-SERVE-GATE-2`); the revert's sole cause was the combination token
failure, now fixed, so per the parity-enabler policy the default is flipped ON
(gated) and the next binding grid re-measures it. `benchmark_binding=false` for this
flip. The completed **lost-lanes rescan**
([spec](../.agents/specs/rescan-lost-lanes-2026-07-16.md)) adds the c2-c8
angle: RMSNorm's ~129 launches/step are batch-INDEPENDENT, so the per-launch
gap is a larger fraction of the small c2 mean (total gap ~2.4 ms/step) than of
c16's - the lever is being pursued as a 1:1 port of vLLM's own CUDA kernel,
microbench-first. The rescan also found the **block-table host cluster**
(full-width host re-materialization 4-5×/step, 8192 cols from the
max_model_len default; mechanical-mirror fix dispatched), sampler per-step
cudaMalloc/cudaFree (handed to the W3-throughput owner), and a missing 24
CUDA-graph bucket; its c2-c8 UNATTRIBUTED downgrade is now **RESOLVED** by the
**correct-state c2/c8 full-step split (2026-07-16,
[spec](../.agents/specs/c2-c8-attribution-2026-07-16.md); evidence
`dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`, SUMMARY.json `5fa07663…e231`)**,
which mirrors the c16 method at c2/c8 (fresh `beb8497` build, gate 235/235;
binding corpus/client; ours nsys node-trace 127/126-step pure-decode spans; vLLM
torch-profiler at `--mamba-ssm-cache-dtype float32`, digests equal, 1524/1508
clean windows; wall = `246a23c` binding TPOT, capture corroborates −0.6/−1.3%).
Verdict per step (ours/vLLM/Δ): **c2** busy 107.310/104.151/+3.159, idle
2.540/3.269/**−0.729**, wall +2.43 ⇒ the c2 gap is **ENTIRELY GPU-busy** (130%
busy / −30% idle): glue +2.40 (RMSNorm(129×) +1.74 batch-independent, FP4-quant
+0.30, SiLU +0.23, gated +0.14) + GDN recurrence +0.93, GEMM parity (−0.24).
**c8** busy 114.706/111.890/+2.816, idle 16.704/12.230/**+4.474**, wall +7.29 ⇒
**38.6% GPU-busy** (glue +2.45, recurrence +1.53, GEMM −1.28 ours-faster) **+
61.4% wave-boundary stall time**: inside pure-decode waves both engines are ≥99%
busy at parity (ours in-span idle 0.92-0.94 ms/step, post-sampler boundary hole
0.116/0.186; vLLM ~0.84-0.88 incl. 48-68 µs inter-window host gap) - the idle
mass accrues at wave boundaries, i.e. the prefill-event stall mechanism moves the
c8 MEAN, not just tails. Regime change, not interpolation (busy Δ 3.16/2.82/4.65,
idle Δ −0.73/+4.47/+3.25 at c2/c8/c16). Lever routing: c2-c4 → kernel glue
(`KERNEL-EW-NORM-ACT`) + recurrence tiling; c8+ extra mass → the W3 overlap
family (see (b): composition is byte-identical, the magnitude gap is async
depth-2 overlap); host plumbing bounded ≤~0.2 ms/step (hygiene only);
(b) the **c8 p99 / c32 p90 ITL tail mechanism - DISCRIMINATED**
(CPU wave-boundary diff, see above): NOT a scheduler divergence (our sync
`Scheduler` == vLLM's async `AsyncScheduler` composition, byte-identical) but
async-runtime output-timing phasing = our W3 (default-ON since 2026-07-17); no
sync-scheduler fix exists, per MIRROR policy the fix IS async-on. Confirmation =
the pending c8+c32 W3-on/off ITL-tail A/B on an idle GPU. Diagnostic
(`benchmark_binding=false`, no speed credit); binding stays 49/124.

**FP4-quant decode-glue vectorization sub-lever (2026-07-17, `CLAIM-FP4-QUANT-FAST-1`).**
Attacks the batch-independent FP4-quant (+0.30 ms/step) + SiLU (+0.23) glue floor
above with two NUMERICS-NEUTRAL fast kernels behind default-OFF opt-in flags
`VT_FP4_QUANT_FAST` (`ScaledFp4QuantFastKernel`) / `VT_SILU_FP4_FAST`
(`SiluAndMulFp4QuantFastKernel`): each thread does ONE 16-byte `uint4` vectorized
load (vs 16 scalar) + ONE 64-bit packed store (vs eight 1-byte), memory-access
pattern ONLY - the exact `CastToFp4NibbleDev`/`F32ToFp8Dev`/`fmaxf`-amax/bf16-SiLU
math is unchanged (grounded 1:1 in vLLM `nvfp4_quant_kernels.cu:56-80,98`). BIT-IDENTITY
PROVEN on DGX (byte-exact new-vs-old nibbles + fp8 scales, 60/60 adversarial parity
asserts; full FP4 suite 24/24 = 26,976 asserts; flag test 20/20). Isolated nsys
per-launch (swizzled bf16, median, scalar→fast): ScaledFp4Quant K=5120 **1.12-1.18×** /
K=17408 (down_proj) **1.44-1.62×**; SiluAndMul I=17408 **1.14×** (c2/M=2) / **1.38×**
(c16-c32/M≥16). **Verdict vs the ≥1.3× flip bar = PARTIAL**: both clear at the larger
shapes but MISS at their dominant production shapes (ScaledFp4Quant K=5120; SiluAndMul
c2) - swizzled small-M is padding-thread-dominated (127/128 threads at M=2 are identical
scale-zero padding early-outs), masking the real-thread win; the residual to vLLM's
~1.7× is the numerics-CHANGING hardware `cvt.rn.satfinite.e2m1x2` + `__hmax2` bf16
reduction (out of scope by the hard bit-identity requirement - that is the non-bit-exact
`VT_FP4_FUSED_VEC` native kernel). Both ship **default OFF** - NOT ready for an
unconditional flip; bit-identical + engine-safe so the orchestrator may fold them into
the combined in-situ A/B. `benchmark_binding=false`, no speed credit; binding 52/124.
DGX evidence `dgx:~/work/vllm.cpp-fp4-quant-fast`.

</details>

## Binding 35B online gate (first, 2026-07-18)

First-ever 35B (Qwen3.6-35B-A3B-NVFP4, MoE, `modelopt_mixed`) online-serving
binding on `69f2717` (35B-IMA-fixed; Platform seam `54d6569` behavior-identical).
Both arms 18/18 legs, 12/12 binding-eligible; evidence
`dgx:~/work/vllm.cpp-online-gate/evidence/69f27178…/summary-35`, ratios.json
sha256 `e7576e09…`. 35B correctness holds (315/315 token-exact throughout). The
vLLM oracle arm required a disk reclaim (flashinfer sm120 GEMM JIT).

**Disposition: 70/124** (19→70; the MoE shared-expert aux-stream overlap flipped c4 to winning). c4-c32 ALL WIN vLLM (16/20 each, decode+throughput, TPOT 1.05-1.18×); memory 4/4 beats vLLM; only c1/c2 residual (~0.96-0.98, within 2-4%). Next: merged-projection fp8 glue-fusion + remaining aux-stream slices for c1/c2.

| Concurrency | Axes | Total tok/s ours / vLLM (ratio) | Mean TTFT (v/o) | Mean TPOT (v/o) |
|---:|---:|---:|---:|---:|
| 1 | 0/20 | 491.5 / 601.9 (0.8167×) | 0.8778× | 0.8132× |
| 2 | 0/20 | 767.7 / 904.7 (0.8486×) | 0.8266× | 0.8516× |
| 4 | 0/20 | 1236.4 / 1366.8 (0.9046×) | 0.7988× | 0.9286× |
| 8 | 6/20 | 1855.1 / 1923.2 (0.9646×) | 0.8019× | **1.0215×** |
| 16 | 16/20 | 2489.1 / 2464.9 (**1.0098×**) | 0.8029× | **1.1025×** |
| 32 | 15/20 | 3030.5 / 2993.0 (**1.0125×**) | 0.7906× | **1.0971×** |

Three gaps: (1) **memory** - the MoE host double-store is FIXED (2026-07-18,
`ENG-MOE-HOSTFREE`): freeing the routed-expert fp4 host mirror after the device
Marlin resident is built (`ReleaseHost` = `madvise(MADV_DONTNEED)`+swap, gated on
`MarlinMoeEnabled()`, `VT_MOE_HOST_FREE=0` rollback) drops 35B **STEADY serving
PSS 20.17→3.53 GiB** (clean same-binary A/B on DGX, `sample_process_memory.py`;
beats vLLM's 13.3 GB steady), token-neutral (315/315+235/235), memcheck clean.
The grid's whole-window `peak_pss` (~20.2 GiB) load-phase coexistence - all
routed-expert host copies coexisting during `LoadQwen3_5Moe` before
`PrepareMarlinResident` freed them - is now addressed by the **load-time streaming
interleave (2026-07-18, `ENG-MOE-LOADSTREAM`, DGX-PROVEN):** the loader DEFERS the
routed experts and `PrepareMarlinResident` builds+frees them one layer at a time,
so at most one layer's ~256 experts coexist on the host (device Marlin residents
byte-identical - only the host-copy lifetime changes). DGX A/B (`~/work/vllm.cpp-mem35-loadstream`
new vs `-parent` 7a1a6d6 eager, production flags, one flock): 35B load-to-ready
**peak RSS (VmHWM) 21.43→4.19 GiB (−17.24 GiB / −80%, below vLLM 13.3)**; token
byte-identical both binaries (315/315 + 235/235); 27B unaffected (24.8 GiB
baseline, dense loader); `compute-sanitizer memcheck` on the deferred load path
0 errors / 315. `benchmark_binding=false` - the orchestrator re-grids the binding
`peak_pss`/`peak_rss` axes to confirm the FAIL→PASS flip. (2)
**low-batch decode** - the
Marlin MoE grouped-GEMM is inefficient at batch=1 (c1 TPOT 0.734×) but scales to
WINNING at c16/c32 (TPOT 1.05×); (3) **TTFT** 0.80-0.86× at all concurrencies
(prefill). Attack: attribute+close low-batch MoE decode, the prefill gap, and MoE
residency memory - a distinct campaign from the 27B decode-kernel close.


**Platform seam (extensibility item 1) DGX-confirmed 2026-07-18** - behavior-preserving refactor, both model gates token-exact (27B 235/235 + 35B 315/315), CUDA -Werror-clean; NOT APPLICABLE to perf (no numeric change). See the parity ledger.

**Platform seam per-tensor correctness regression FIXED 2026-07-18** - the 7 migrated sites were keyed on the process-global `CurrentPlatform().is_cuda()` (accelerator-first), which mis-routed a CPU queue/tensor on a GPU box into the CUDA branch (red DGX CPU tests). Fixed to per-object `GetPlatform(<obj>.device.type).is_cuda()`; also guarded a separate `6a8c5cf` CPU-build breakage (`test_ops_moe_grouped` unguarded CUDA symbol). Dual-config gate: CPU 122/122 CTest + tools 164/164 + checkers green; DGX CPU-tier tests on the GPU box + 27B 235/235 + 35B 315/315. NOT APPLICABLE to perf (correctness fix, no numeric change). See the parity ledger.

The local Qwen3.5-4B checkpoint corpus is exactly
`/tmp/qwen35-4b-sharegpt-1024.json`, SHA-256
`9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`.

## Ours-vs-MLX on Apple M4 - Qwen3-1.7B (2026-07-23, M3b) - INDICATIVE / `BLOCKED-ON-SUDO`

**The first ours-vs-MLX comparison. NOT binding.** Qwen3-dense now runs on Metal
(`BACKEND-METAL-MLX` work row M3b), so both arms exist for the SAME model. Both =
**Qwen3-1.7B bf16**, SAME box, SAME session, p=512 g=128, b∈{1,2,4,8,16}. Ours =
`examples/vllm-bench` on our Metal engine (`Qwen/Qwen3-1.7B` HF safetensors;
device=2/Metal confirmed via `VT_OP_PROVIDER_STATS` - every op `vt-native`); MLX =
`python -m mlx_lm.benchmark` (`mlx-community/Qwen3-1.7B-bf16`), re-run the same
session (it reproduced the committed baseline below).

**Correctness basis (2026-07-23, updated - speed numbers below UNCHANGED).** The
Metal Qwen3-dense forward is CONFIRMED CORRECT by the vLLM oracle teacher-forced on
the Metal prefix (`scripts/qwen3-neartie-gap.py`, vLLM 0.25.0, batch=1,
`gpu_memory_utilization=0.40`): all 60 Metal-vs-CUDA divergent positions on
Qwen3-0.6B are within 0.5 nats of vLLM's own argmax given the Metal prefix, **max
0.125 nats**, none outside top-20 (the p0 tok5 France→Italy flip is gap 0.0000 -
vLLM's teacher-forced argmax on the identical prefix is itself Italy). The e2e gate
is therefore near-tie-robust, NOT strict token-exact (0.6B is a near-tie model;
strict wants a bigger deterministic dense model, Qwen3-4B, not on this Mac -
deferred). This SUPERSEDES the earlier "strict 16/16 token-exact on Metal" claim.
This is a correctness disposition; no Metal speed number is affected.

| B | ours decode tok/s/stream | MLX gen tok/s (agg) | ours TTFT | MLX TTFT | ours peak | MLX peak |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.29 | 27.77 | 4.86 s | 0.47 s | 7.36 GB | 3.78 GB |
| 2 | 4.32 | 50.34 | 9.08 s | 0.90 s | 7.36 GB | 3.97 GB |
| 4 | 4.16 | 93.11 | 17.66 s | 1.74 s | 7.37 GB | 4.18 GB |
| 8 | 3.25 | 160.24 | 28.52 s | 3.40 s | 7.64 GB | 4.47 GB |
| 16 | 2.14 | 211.55 | 47.17 s | 7.16 s | 8.78 GB | 5.28 GB |

**Ours loses on every axis: ~6-11× slower decode, ~7-10× slower TTFT, ~2× peak
memory.** This is a **knowingly-unoptimised FLOOR, not our best** - one command
buffer per op (commit+wait), a plain threadgroup-tiled GEMM with no
simdgroup-matrix, no batched encoders (work row `M3c`) - vs MLX's `steel` tiled
kernels. It is the first ours-vs-MLX number and sets the optimisation target;
the named levers are `M3c` (batched encoders) and a simdgroup-matrix GEMM.

**Status: `BLOCKED-ON-SUDO` - INDICATIVE, NOT BINDING.** The Mac could not be
quieted: `sudo -n true` → "a password is required"; the root `com.localai.worker`
LaunchDaemon and the desktop aerial wallpaper (`WallpaperAerialsExtension`,
~9.8% CPU) were both up. Per the standing contended-run rule this may not be cited
as a bound comparison - though memory pressure read 83% free and MLX's trial
spread stayed sub-1%. **To make it binding**, on the M4: `sudo launchctl bootout
system/com.localai.worker`, disable the aerial wallpaper (System Settings →
Wallpaper, or log the console user out), re-run both arms, then restore with
`sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist`.
**Repro (M4, `~/work/m3b-qwen3/build`):** ours `env VT_OP_PROVIDER_STATS=1
./examples/vllm-bench --model <Qwen/Qwen3-1.7B snapshot> --num-prompts <B>
--input-len 512 --output-len 128 --concurrency <B> --temperature 0` (peak RSS via
`/usr/bin/time -l`); MLX `HF_HOME=$HOME/hf-cache ~/mlx-venv/bin/python -m
mlx_lm.benchmark --model mlx-community/Qwen3-1.7B-bf16 -p 512 -g 128 -b <B> -n 3`.

---

## MLX competitor baseline on Apple M4 - UNOPPOSED FLOOR (2026-07-22)

**READ THIS FIRST. This is NOT a parity result and NOT a binding floor.** (Now
superseded as the live comparison by the ours-vs-MLX table above; retained as the
original MLX-only baseline.) Every number below is **MLX's own**, produced by
**MLX's own harness**.

**Status: `BLOCKED-ON-SUDO` - INDICATIVE, NOT BINDING.** The `com.localai.worker`
root LaunchDaemon on the M4 could not be stopped (`sudo -n true` -> "a password is
required"; no passwordless sudo). Per the standing rule that a contended run is
void, these figures may **not** be cited as a bound competitor floor for
`BACKEND-GATE-METAL-MLXLM`. They are a design target plus an exact re-run recipe.
The worker WAS measured genuinely idle (0.0% CPU; `ioreg IOAccelerator`
Device/Renderer/Tiler Utilization all **0**), which is consistent with the
sub-1% trial spread - but "measured idle" is not "stopped".

### Repro recipe (reproduction is a gate)

```sh
# Box: Apple M4, 16 GiB unified, macOS 26.5.2 (25F84), ssh 192.168.68.103
#   GPU: applegpu_g16g, max_recommended_working_set_size 12,713,115,648 B (11.84 GiB)
# vllm.cpp commit: 1cb5f64 (NOTHING OF OURS WAS BUILT OR RUN - MLX-only measurement)
#
# PRECONDITIONS FOR A BINDING RE-RUN (neither was satisfiable by an agent):
#   sudo launchctl bootout system/com.localai.worker      # then RESTORE after:
#   sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist
#   ...and disable the desktop AERIAL VIDEO WALLPAPER (WallpaperAerialsExtension,
#      measured 8.2% CPU + VTDecoderXPCService 2.2%, touches the GPU continuously)
#
# install - VENV ROUTE, brew deliberately NOT used (brew mlx pulls python@3.14 into
# /opt/homebrew/bin, first on the PATH our macOS builds use, changing find_package(Python3))
/usr/bin/python3 -m venv ~/mlx-venv && ~/mlx-venv/bin/pip install -U pip mlx-lm
# resolved: mlx 0.29.3, mlx-metal 0.29.3, mlx-lm 0.29.1
#   (CLT python 3.9.6 caps the resolve BELOW brew's 0.32.0 - record it; an unpinned
#    competitor arm is not a floor)

# model: mlx-community/Qwen3-1.7B-bf16 @ rev 9cd6692855d3e06772228e9a962b2606359b2d24
#   1.7B and not 4B/8B: the desktop session already holds ~13 of 17.2 GB, and the
#   b=16 arm must not page. Gate-scale (27B/35B) is PERMANENTLY out of reach here.
export HF_HOME=$HOME/hf-cache
~/mlx-venv/bin/hf download mlx-community/Qwen3-1.7B-bf16

# measurement - MLX-LM's OWN harness (their tool, their tps definitions):
#   mlx_lm/benchmark.py; 512 random prompt tokens, 128 generated, EOS disabled,
#   mx.random.seed(0), 1 warmup + 3 timed trials
for B in 1 2 4 8 16; do
  ~/mlx-venv/bin/python -m mlx_lm.benchmark \
      --model mlx-community/Qwen3-1.7B-bf16 -p 512 -g 128 -b $B -n 3
done
```

### Measured - MLX 0.29.3, Qwen3-1.7B-bf16, p=512 / g=128

`prompt_tps` and `generation_tps` are **aggregate over the batch**
(`mlx_lm/generate.py:1657,1660`); the derived columns follow from that.

| batch | prompt_tps | out tok/s (aggregate) | out tok/s (per req) | TTFT | ITL | peak mem | trial spread |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1089.92 | **27.57** | 27.57 | **470 ms** | **36.3 ms** | 3.78 GB | 0.12% |
| 2 | 1137.45 | 48.91 | 24.45 | 900 ms | 40.9 ms | 3.97 GB | 0.25% |
| 4 | 1178.67 | 90.15 | 22.54 | 1,738 ms | 44.4 ms | 4.18 GB | 0.63% |
| 8 | 1199.06 | 156.95 | 19.62 | 3,416 ms | 51.0 ms | 4.47 GB | 0.36% |
| 16 | 1194.64 | **213.39** | 13.34 | 6,857 ms | 75.0 ms | 5.28 GB | 0.14% |

**Reading.** Prefill saturates at **~1,200 tok/s** by b=8 and does not improve
(b=16 is marginally lower) - a compute roof. Decode scales **7.74x from b=1 to
b=16**, the signature of a bandwidth-bound decode amortizing weight traffic
across the batch. Peak memory grows only **1.40x over a 16x batch increase** -
the ~3.4 GB of bf16 weights dominate and the KV cache is small at 512+128.
Trial spread is **0.12%-0.63%** across all five arms, comfortably inside
run-noise.

**What this is sufficient for today:** it prices the §5 decision in the
[reuse study](../.agents/specs/metal-mlx-reuse-study.md) - whether delegating
GEMM to MLX is worth its ~105 MB `mlx.metallib` + `libmlx.dylib` dependency is a
question our own MSL GEMM must answer *by measurement against this line*.
That is why the `vt::OpProvider` seam carries mandatory
`VT_OP_PROVIDER_STATS` instrumentation.

### MLX-as-a-PROVIDER (2026-07-22) - **NO TIMING, and why**

The seam and the MLX GEMM provider are now landed and correctness-gated
(`BACKEND-ACCEL-PROVIDER`; `-DVLLM_CPP_MLX=ON`, default OFF). **No MLX-vs-MSL
speed number is published here, and none was taken as binding**, for the reason
already recorded in §8 of the study and unchanged since: the M4 could not be
quieted. `sudo -n true` still answers *"a password is required"*, so the
`com.localai.worker` root LaunchDaemon is still up, and the desktop **aerial
video wallpaper** (`WallpaperAerialsExtension`, ~8.2% CPU, plus
`VTDecoderXPCService`) still decodes video onto the same GPU. A GEMM A/B is
exactly the kind of measurement those perturb, so publishing one would be
laundering a contended number.

**What the user must run before any MLX-vs-MSL timing can bind:**

```sh
ssh 192.168.68.103
sudo launchctl bootout system/com.localai.worker      # stop the worker
# System Settings -> Wallpaper: replace the AERIAL wallpaper with a static one
#   (or log the console user out entirely)
# ... then the A/B below ...
sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist   # RESTORE
```

**The A/B itself needs no rebuild** - that is what the seam bought. One binary,
one lever, and the run *proves which provider executed* rather than assuming it:

```sh
cd ~/vllmcpp-accel/build-mlx
export DYLD_LIBRARY_PATH=$HOME/mlx-venv/lib/python3.9/site-packages/mlx/lib
VT_OP_PROVIDER_STATS=1 ./tests/test_metal_backend        # arm A: MLX selected
VT_OP_PROVIDER_DISABLE=mlx VT_OP_PROVIDER_STATS=1 \
  ./tests/test_metal_backend                             # arm B: native MSL
```

Build recipe (Metal + the optional MLX provider, CLT-only, no Xcode, brew's
`python@3.14` kept off the build PATH):

```sh
PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin cmake -S . -B build-mlx \
  -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_METAL=ON \
  -DVLLM_CPP_MLX=ON -DMLX_ROOT=$HOME/mlx-venv/lib/python3.9/site-packages/mlx
```

**Correctness IS recorded (it does not need a quiet box).** Per op, at real
projection widths, against our own CPU backend as the oracle; bar
**NMSE <= 5e-4**; bit-exactness across providers is explicitly *not* promised:

| GEMM | shape / dtype | native MSL vs CPU | MLX vs CPU | MLX vs MSL |
|---|---|---:|---:|---:|
| `kMatmul` | 1x2048x2048 bf16 (decode) | 2.80e-06 | 2.80e-06 | 0 |
| `kMatmulBT` | 1x2048x2048 bf16 (decode) | 2.76e-06 | 2.76e-06 | 0 |
| `kMatmul` | 32x2048x6144 bf16 (prefill) | 2.75e-06 | 2.75e-06 | 0 |
| `kMatmulBT` | 32x2048x6144 bf16 (prefill) | 2.74e-06 | 2.74e-06 | 0 |
| `kMatmul` | 128x512x512 f32 | 3.81e-14 | 3.81e-14 | 0 |
| `kMatmulBT` | 128x512x512 f32 | 3.74e-14 | 3.74e-14 | 0 |

The zeros are an **observation on these shapes, not a promise** - both providers
run IEEE (MLX pins `setFastMathEnabled(false)`, we pin `MTLMathModeSafe`) and
both accumulate in f32 ascending k, which is enough to coincide here; nothing in
the design relies on it and the gate is the NMSE column. **Both paths are proven
to have run**: the test asserts `last_selected` names the provider AND that its
`declines` counter is zero, so a silent fall-back to MSL cannot masquerade as an
MLX pass.

## DeepSeek-V2 MLA prefix-cache-hit assert fix - correctness, no new benchmark (2026-07-23)

`CLAIM-MLA-PREFIX-CACHE-ASSERT` (base `6abe09c`). A correctness/robustness fix only - **no throughput
number is produced or changed**. The DeepSeek-V2 SACRED gate `test_deepseek_v2_paged_engine` aborted at
`single_type_kv_cache_manager.cpp:293` under asserts-enabled builds (a too-strict
`FullAttentionManager::find_longest_cache_hit` precondition rejecting the MLA cache group); it was inert
under the canonical `-DCMAKE_BUILD_TYPE=Release` gate build (NDEBUG). Relaxed to upstream's admissible set.
Evidence (dgx, asserts-on, `-Werror` 0 warnings): `test_deepseek_v2_paged_engine` **8/8, 3/3 runs**
(incl. Phase-3 prefix-cache reuse byte-identity); manager unit suite 28 cases / 77,643 asserts; prefix-cache
units interface 21/21, coordinator 16/16, block_pool 14/14, kv_cache_manager 9/9. Full-attention models are
byte-identical (the changed TU compiles to identical object code under NDEBUG). The binding W9 every-axis
throughput grid below is UNCHANGED (the fix does not touch any compute/decode path). Repro: `git archive`
`6abe09c` + fix → canonical dgx flags WITHOUT `-DCMAKE_BUILD_TYPE` (asserts live) →
`ctest`/`test_deepseek_v2_paged_engine`.

## Binding DeepSeek-V2-Lite (MLA) every-axis grid - MLA campaign W9 (2026-07-22)

**Verdict: NOT every-axis parity. The row stays `ACTIVE`.** Two optimizations
landed and are measured below; they took decode throughput from **0.50x to 0.87x**
of vLLM at c1, and TTFT now BEATS vLLM at c4/c8 - but output/total/request
throughput is short at every concurrency and TPOT/ITL is short at c1/c4/c8, so
this is an **attributed miss**, not a pass.

### The denominator, and why it is `--moe-backend triton`

vLLM AUTO-selects the **FlashInfer CUTLASS Unquantized MoE** backend for this
model (`unquantized.py:262`, out of
`['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON']`).
**It cannot run on GB10.** It has now HARD-REBOOTED dgx **five times** - three at
W8, and twice more at W9 with the Qwen3-Coder mitigations deliberately applied:

| attempt | config | outcome |
|---|---|---|
| W8 x3 | `gpu_memory_utilization=0.40`, `max_num_batched_tokens` capped to 2048; third attempt on a freshly-rebooted box with an empty page cache | box reboot |
| W9 #1 | `gmu=0.40`, `max-model-len 2048`, `max-num-batched-tokens 2048`, `max-num-seqs 8` | box reboot at 07:59 (log ends 07:58:49 right after `torch.compile took 9.23 s`; `uptime` then read `up 0 min`) |
| W9 #2 | PRISTINE freshly-rebooted box, page cache **0 GiB**, 116 GiB free, `gmu=0.40`, `max-model-len 2048`, `max-num-batched-tokens 1024`, `max-num-seqs 4` | box reboot at ~08:08 (log ends 08:05:24, same phase) |

Both W9 deaths land at the IDENTICAL phase - immediately after `torch.compile`,
in the memory-profiling dummy run / FlashInfer autotune where the CUTLASS MoE
expert workspace is first allocated and exercised. GB10's 119 GiB is UNIFIED, so
that workspace competes with the weights, the reservation and the page cache in
one pool. `sudo` is password-gated on this box, so `drop_caches` was never
actually available; W9 #2 substitutes for it exactly by starting from a 0 GiB
page cache. Evidence: `~/w9mla/logs/cutlass_serve.log`,
`cutlass_serve2.log`, `cutlass_probe.stamp`, `cutlass_probe2.stamp`.

**Therefore `--moe-backend triton` IS vLLM's best STABLE, GRAPHED, production
configuration on this hardware, and it is the legitimate bar.** This is stated
rather than quietly assumed, because picking the slower backend to flatter our
numbers would be exactly the wrong move - and note that the substitution does not
flatter us: we LOSE against it. The oracle logs confirm the arms are otherwise
matched - both sides resolve `TRITON_MLA` decode + `FLASH_ATTN` MLA prefill, and
vLLM runs GRAPHED (`cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes
`[1,2,4,8,16]`), never `--enforce-eager`.

### Methodology

Fresh `vllm serve` **per concurrency** (driving several concurrencies at one
server replays identical `RandomDataset` prompts into the prefix cache).
**Prefix-cache hit rate VERIFIED in every serve log: 0.0% at c1/c2/c4**; c8
drifted to **1.1%** across its 3 reps (96 prompts, a small `RandomDataset`
collision) - reported, not hidden; it slightly favours vLLM at the one
concurrency where our TTFT already wins. Medians of 3 reps per cell, idle box,
same-binary A/B for every lever.

### The grid (medians of 3 reps; 1024-in / 128-out; ratio = ours ÷ vLLM)

Throughput wants ratio >= 1.00; latency wants <= 1.00. **Bold = FAILS the axis.**

| c | metric | vLLM 0.25.0 | ours (W9 prod) | ratio |
|---|---|---|---|---|
| 1 | output tok/s | 38.17 | 33.18 | **0.87** |
| 1 | request req/s | 0.30 | 0.26 | **0.87** |
| 1 | median TTFT ms | 214.85 | 227.14 | **1.06** |
| 1 | median TPOT ms | 24.71 | 27.51 | **1.11** |
| 1 | median ITL ms | 24.69 | 27.46 | **1.11** |
| 2 | output tok/s | 55.27 | 52.63 | **0.95** |
| 2 | request req/s | 0.43 | 0.41 | **0.95** |
| 2 | median TTFT ms | 329.80 | 374.67 | **1.14** |
| 2 | median TPOT ms | 33.94 | 33.03 | 0.97 |
| 2 | median ITL ms | 33.32 | 33.27 | 1.00 |
| 4 | output tok/s | 81.51 | 70.36 | **0.86** |
| 4 | request req/s | 0.64 | 0.55 | **0.86** |
| 4 | median TTFT ms | 549.05 | 528.49 | 0.96 |
| 4 | median TPOT ms | 45.19 | 52.38 | **1.16** |
| 4 | median ITL ms | 43.35 | 50.38 | **1.16** |
| 8 | output tok/s | 116.35 | 102.37 | **0.88** |
| 8 | request req/s | 0.91 | 0.80 | **0.88** |
| 8 | median TTFT ms | 605.46 | 533.48 | 0.88 |
| 8 | median TPOT ms | 63.62 | 74.20 | **1.17** |
| 8 | median ITL ms | 59.34 | 67.10 | **1.13** |

Axes PASSED: **TTFT at c4 and c8** (we are 4% / 12% faster), **TPOT and ITL at
c2**. Everything else is short. Total-token throughput tracks request throughput
exactly (identical 1152-token request shape), so it fails with the same ratios.

**Peak memory - the one axis we win decisively.** Ours: **31.38 GB** peak RSS
(`/usr/bin/time -v`, c8, 1024-in/128-out). vLLM: **68.5 GiB** (29.34 GiB weights +
39.15 GiB KV, from its own startup log at `gpu_memory_utilization=0.58`), i.e.
ratio **0.46**. Stated with its caveat rather than banked uncritically: vLLM
PRE-RESERVES a fixed fraction of memory up front and then serves KV out of it,
whereas we allocate the KV blocks the configured concurrency actually needs, so
this is a real difference in operating footprint at this workload but NOT evidence
that our per-token KV cost is lower - the MLA page geometry is identical on both
sides (36864 B = block 32 x 576 x 2B, no factor 2).

### The two optimizations, with their measured deltas

Same binary, same workload, medians of 3, rollback envs recorded.

| lever | rollback | c1 | c2 | c4 | c8 |
|---|---|---|---|---|---|
| decode CUDA graph | `VT_DEEPSEEK_CUDAGRAPH=0` | +2.5% | +2.5% | +1.8% | +2.0% |
| MLA split-KV occupancy fill | `VT_MLA_SPLIT_FILL=0` | **+69.5%** | **+53.3%** | **+32.0%** | **+19.5%** |
| both vs the W8 baseline | - | **+73.6%** | **+57.2%** | **+34.5%** | **+21.8%** |

(output token throughput; W8 baseline was 19.11 / 33.49 / 52.33 / 84.02 tok/s.)

The decode CUDA graph is a MUCH smaller lever here than its Qwen3-Coder analogue
(which was worth the whole c1 deficit): DeepSeek-V2-Lite's decode step is
GPU-bound, not host-bound, so there is only ~1.2 ms/step of launch tax to remove.

### nsys kernel-list diff (the evidence that selected the lever)

`nsys profile --cuda-graph-trace=node -t cuda`, both sides, 1024-in / 64-out,
batch 1. Ours: `~/w9mla/logs/ours_c1.nsys-rep` (before),
`ours_prod_c1.nsys-rep` (after). vLLM: `vllm_c1.nsys-rep`, captured through the
**LLM API** (nsys breaks vLLM's server EngineCore).

BEFORE, ours: `MlaDecodeStage1` was **44.7% of ALL GPU time at 837 us/instance**
(27 layers x 837 us = ~22.6 ms of a ~47.7 ms TPOT). The KV latent a batch-1 step
reads is ~1.25 MiB/layer, i.e. ~5 us at GB10's memory rate - the kernel was
running about **two orders of magnitude off its own memory-bound floor**. Cause:
`_compute_num_kv_splits` derives the split count from `max_seq_len` alone and
returned **2**; MLA has one KV head so the head dimension collapses to a single
tile; at batch 1 the grid was **2 CTAs on the whole GPU**.

AFTER the occupancy fill: **837 us -> 45.8 us, an 18.3x kernel speedup**, and
`MlaDecodeStage1` fell from 44.7% to **3.9%** of GPU time (`MlaDecodeStage2` rose
1.7% -> 1.9% as there are now more partials to combine - net hugely positive).

The residual, from the same diff:
1. **MoE grouped GEMM is now our top kernel at 40.5%**
   (`MoeGroupedGemmBf16NaiveSplitK`, 158 us median x 3 launches/layer). vLLM runs
   ONE `fused_moe_kernel` family instead. Our per-step MoE time is ~1.2x its own
   bandwidth floor, so this is close but not free.
2. **Dense projections at batch 1 fall to cuBLAS `gemvx` (GEMV) for 31.8% of our
   GPU time; vLLM splits the same work between `gemvx` (12.7%) and
   tensor-core `nvjet_sm121_tst_mma_*` (6.6%).** This is the largest single
   attributable difference and is the next lever.
3. **vLLM runs Inductor-codegenned FUSED glue we do not have** -
   `triton_red_fused_add_fused_add_rms_norm_1`,
   `triton_red_fused_fused_add_rms_norm_moe_forward_shared_0`,
   `triton_poi_fused_mul_silu_slice_0`, `triton_poi_fused_2` - roughly 7-8% of
   its profile, replacing what for us are separate `RmsNorm` / `SiluAndMul` /
   `MoeCombine` launches. Same theme the 35B campaign recorded.
4. Our MLA decode attention is still ~2.3x vLLM's per call (ours 45.8 + 22.6 us
   vs its `_fwd_grouped_kernel_stage1` at 29.6 us) - small in absolute terms now,
   but real.

**Next lever, by gain / effort: (2), routing the batch-1 dense projections to a
tensor-core GEMM path instead of `gemvx`.** It is the biggest single line in the
diff and it is a dispatch change, not a new kernel.

### Repro recipe (reproduction is a gate)

```
# commit: see the W9 closing SHA in the ledger.  Box: dgx.casa (GB10, sm_121a), IDLE.
# build:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DVLLM_CPP_TRITON=ON -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build -j 16
# model:
M=$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0
# vLLM denominator (FRESH server per concurrency; VERIFY 0.0% prefix-cache hit):
~/w9mla/w9_vllm.sh                      # gmu 0.58, --moe-backend triton, 3 reps
# ours, all three A/B arms:
~/w9mla/w9_ab.sh                        # base / graph / prod, 3 reps
# correctness (must stay 8/8 with the optimizations DEFAULT-ON):
./build/tests/test_deepseek_v2_paged_engine
```

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| LMCache C++ CLIENT feasibility - how vLLM connects to a running LMCache and whether a from-scratch C++ client can speak the wire (`KV-EXTERNAL-CACHE`, `CLAIM-LMCACHE-CPP-CLIENT`, base `f6be46e`) | **NOT APPLICABLE - ANALYSIS/SCOPING ONLY; NO NUMBER MEASURED OR CLAIMED** (2026-07-23; [spike](../.agents/specs/lmcache-cpp-client-connector.md)). Records-only change: no code, no build, no GPU, nothing installed (read a public read-only `LMCache/LMCache@8570aad` clone). A speed number is meaningless here - nothing is implemented; the deliverable is a wire-protocol map + a feasibility verdict, not a measurement. | **Source-grounded findings, not measurements.** vLLM connects to a RUNNING LMCache over two portable wires, so the parent spike's "no specified wire protocol" verdict is REFUTED: (1) `lm://` remote-store - plain TCP + a fixed `struct.pack("iiiiiiiii150s")` 186-byte header + raw KV bytes (`lmcache/v1/protocol.py:214-321`), and (2) the MP server - ZMQ DEALER↔ROUTER + `msgspec.msgpack` + CUDA-IPC (`multiprocess/mq.py:270-353`). Both need ZERO in-process `lmcache` and both key on LMCache's own blake3 token hash, not vLLM block hashes, so the R1 hash blocker does not bind. Pickle is confined to the MP one-time registration; CUDA-IPC to the MP data plane (co-located, `cudaIpcGetMemHandle`-speakable). **Verdict: a C++ client is FEASIBLE; recommend the `lm://` mode first.** | **`PENDING` - the first number this track will owe is a MODE (1) interop round-trip against a REAL `lmcache.v1.server` (CPU, no GPU): a C++ PUT then GET returns byte-identical KV, then a key-agreement test that a chunk stored by a Python vLLM+LMCache peer is found and decoded by our client.** Blocked only on standing an `lmcache` server up on a test box (it is on no project box today) and on the W5 abstract-`KVConnector` seam landing. Repro for what IS recorded here: none - nothing was executed. |
| LMCache C++ CLIENT **W1** - the MODE-1 (`lm://`) wire codec (`KV-EXTERNAL-CACHE` `SPIKE`→`ACTIVE`, `CLAIM-LMCACHE-CPP-CLIENT`, base `96c5cbb`) | **NOT APPLICABLE - a pure CPU codec moves no tokens; NO throughput/latency number measured or claimed** (2026-07-23; [spec](../.agents/specs/lmcache-cpp-client-connector.md)). W1 is a wire codec only (framing + key + blake3 + KV_2LTD layout); nothing connects to a server yet (W2) and NO call site routes to it (INERT), so there is no engine path whose latency could change and nothing to A/B. A file-IO or hash micro-rate here would be a number without a workload. | **Correctness, not speed: our wire bytes == the REAL Python codec's, byte/bit-exact.** Fixtures captured from stdlib `struct` (LMCache's own framing codec), the `blake3` PyPI package (the exact package `token_hasher.py` imports) and numpy (KV_2LTD); `tests/vllm/v1/kv_offload/lmcache/test_lmcache_codec.cpp` asserts equality - **6/6 cases, 2074/2074 assertions.** blake3 digest VERIFIED byte-identical on the dev box x86-64 AND the `dgx.casa` aarch64 box, both matching Python. Vendored official BLAKE3 C impl (1.5.5, portable backend, separate static lib). Clean CPU `-Werror` 0 warnings; CUDA build untouched. INERT by construction (grep: no external reference). | **`PENDING` (unchanged track owed) - the MODE (1) interop round-trip against a REAL `lmcache.v1.server` (W2, CPU): C++ PUT then GET returns byte-identical KV; then key-agreement vs a Python vLLM+LMCache peer.** Still blocked on standing an `lmcache` server up on a test box. Repro for what IS recorded here: `cd build && ctest -R test_lmcache_codec`. |
| LMCache C++ CLIENT **W2** - blocking `lm://` TCP client + KV_2LTD repack, round-trips a REAL server (`KV-EXTERNAL-CACHE` `ACTIVE`, `CLAIM-LMCACHE-CPP-CLIENT`, base `9d2b8ba`) | **NOT APPLICABLE - a transport client with no engine call site moves no tokens; no throughput/latency number** (2026-07-23; [spec](../.agents/specs/lmcache-cpp-client-connector.md)). W2 adds the POSIX-socket client; nothing in the engine routes to it yet (W3 wires the KVConnector), so there is no path whose latency could change. | **Correctness/interop, not speed: our client and a REAL `lmcache.v1.server` agree byte-for-byte.** `test_lmcache_client` real-server case 36/36 (PUT then GET byte-identical at 1 B / 4 KiB / 64 KiB, EXIST true/absent, KV_2LTD repack byte-identical), bidirectional Python-codec interop both directions, plus an always-on in-binary mock server 45/45 for CI without Python. Clean CPU `-Werror` 0 warnings; INERT (grep: no engine reference). | **`PENDING` - W3 (KVConnector wiring) owes the first live prefix-hit-shortcut number against a running LMCache.** Repro: `cd build && ctest -R test_lmcache_client`; live: `scripts/lmcache/run_live_roundtrip.sh`. |
| KV offload - first demonstrable speedup (`KV-OFFLOAD` W4, `CLAIM-KV-PERSISTENCE-LMCACHE`, base `93b55c0`) | **ACCEPTED - CACHE-HIT / PREFILL-SAVED demonstration (behavioural, CPU).** The W1-W3 tiers now have a scheduler consumer: the tiering manager + the `OffloadingConnector` scheduler half, wired opt-in + default-OFF. This is the FIRST time offload becomes a measured speedup rather than storage. **Independently re-verified 2026-07-23 on the stable dgx (rebased onto `main` `5c2a9a4`, head `e8a5d53`): the 32/48 prefill-saved demonstration reproduces; clean CUDA `-Werror` build 0 warnings; all six SACRED gates UNCHANGED standalone (27B 235/235, 35B 315/315, Qwen3-Coder 138/138, Qwen3-dense 184/184, OPT 63/63, DeepSeek-V2 223/223); tiering/connector memcheck 0 errors/0 leaks - the default-OFF connector is inert on every model.** | **Restarted-prefix workload through the REAL scheduler** (`tests/vllm/v1/test_kv_offload_connector.cpp`, e2e case): a 48-token prompt = 3 prefix blocks, all persisted to the DISK tier, then the CPU tier evicted (a genuine restart). Offloading OFF baseline schedules the full 48-token prefill; offloading ON, the connector reports a disk HIT and **prefill is shortcut by 32 of 48 tokens (2 of 3 blocks promoted from disk; the 3rd recomputed per the num_tokens-1 rule), a measured hit rate of 2/3 blocks.** The promoted bytes are proven **byte-identical** to the cold store, so the continuation is token-identical (a wrong promoted block would be a silent wrong-output bug; identity refusal holds through the promotion). Gate-scale: this is a correctness+shortcut demonstration on CPU, not a throughput grid - the engine-level TTFT/throughput number is owed once W5 wires the worker-side GPU load and a CLI switch | **W5** - the worker-side load into live GPU memory + a user-facing switch; THEN a DGX every-axis TTFT/throughput grid on a shared-prefix corpus vs vLLM's `--kv-transfer-config`, with hits proven in every arm |
| Accelerator-seam audit (`BACKEND-SEAM-AUDIT`) | **NOT APPLICABLE - AUDIT + PLAN ONLY; NO NUMBER MEASURED, PRODUCED OR IMPLIED** (2026-07-22, `CLAIM-BACKEND-SEAM-AUDIT-1`; [audit](../.agents/specs/accelerator-seam-audit.md)). Records-only change: no source file, no CMake, no kernel, no test, no build, no GPU work, nothing downloaded. One new row enters at `SPIKE`; no implementation row moves | **Static analysis of the tree at `72f5db2` plus `file:line` reading of pinned vLLM `e24d1b24` - no execution.** Verdict on the user's question (*do MLX/Vulkan port vLLM's CUDA-path strategy?*): **PARTIAL** - MIRROR at the `Platform` and attention-registry seams, **ABSENT** at `model_executor/layers/` (no `QuantizationConfig`/`LinearMethod`). Shared-layer device references **DSR = 94**, **67 (71%) in `qwen3_5.cpp`** alone, versus **1** in its 802-line upstream twin `qwen3_next.py:321`; upstream keeps 199 of 544 device predicates in `layers/` and 14 across 287 model files. Counts had already REGROWN since the reuse study (`kCUDA` 54->63, `is_cuda()` 13->16) with no bad commit involved. The one behavioural fact reused - Vulkan V1 edited 0 pre-existing `src/`/`include/` files - is carried from the fan-out spike, where it was measured | **`NOT APPLICABLE`. No performance claim is owed by this change and none may be made for Metal or Vulkan until a model runs on either - neither runs one today.** The first number this track ever owes is work row `S5`'s correctness gate, not a speed gate: **OPT token-exact on Metal against our own CPU backend on the same M4, with only today's 10 registered Metal kernels**, proving the portable reference tier. Work rows `S3`-`S7` are refactors on the 27B/35B hot path and each additionally owes a same-binary throughput-neutrality A/B on the serving grid plus the unchanged standalone dgx regression set (27B 235/235, 35B 315/315, Qwen3-Coder 6/6, Qwen3-dense 16/16, OPT 6/6, DeepSeek-V2 8/8). Repro: none - nothing was executed |
| DSR device-leakage ratchet (`BACKEND-SEAM-AUDIT` work row `S1`) | **NOT APPLICABLE - NO NUMBER MEASURED, PRODUCED OR IMPLIED** (2026-07-22, `CLAIM-BACKEND-SEAM-S1-1`; [audit §8](../.agents/specs/accelerator-seam-audit.md)). Tooling + records only: a Python checker, a committed baseline, a CI job and a mutation suite. **NOT ONE FILE UNDER `src/` OR `include/` WAS TOUCHED**, so there is no behaviour change to benchmark and no model gate is owed. No build, no GPU, nothing downloaded | **Static analysis only - the checker runs standalone with no CUDA toolkit and no GPU, like `cmake/CudaArchFeaturesTest.cmake`.** The one number this row produces is a STRUCTURAL metric, not a performance one: shared-layer **DSR = 86** at `18094ee` (44 `kCUDA` + 10 `is_cuda()` + 0 unconditional CUDA includes + 32 `#ifdef VT_*`), **66 of it (77%) in `qwen3_5.cpp`**. Re-derived from the current tree, not carried: the audit's 94 was 6 too high on composition (comments/strings now stripped uniformly; its 4 "unconditional" CUDA includes were already `#ifdef`-guarded) and 2 too high because main reduced real leakage since. Evidence: `python3 scripts/check-device-leakage.py --report` and `python3 tests/scripts/test_device_leakage.py` (24/24) | **`NOT APPLICABLE`.** DSR is a leakage budget that may only ever go DOWN; it is not a quality score and it is meaningless without the checker's allowlist. Repro: `python3 scripts/check-device-leakage.py --report --list`. Performance claims remain owed by `S3`-`S7` (throughput-neutral A/B on the serving grid) if and when they are claimed; none is made here |
| Portable reference tier (`BACKEND-SEAM-AUDIT` work row `S5`) | **NOT APPLICABLE - INFRASTRUCTURE; NO PERFORMANCE NUMBER OWED, AND A FIRED FALLBACK MUST READ 0 IN ANY PERF ARM** (2026-07-23, `CLAIM-BACKEND-SEAM-S5-1`; [audit §10](../.agents/specs/accelerator-seam-audit.md)). Mirrors vLLM `custom_op.py:138 forward_native`: on a UNIFIED-MEMORY device the CPU kernel installs LAZILY as a negative-priority `vt-cpu-ref` `OpProvider` on the first `GetOp` miss, so a partial backend runs an op it lacks natively instead of throwing. Entirely in the `vt` runtime (`src/vt/op_provider.cpp`, `src/vt/backend.cpp`) - **DSR unchanged at 67** | **Byte-identical on real hardware (dgx, standalone under `flock`): 27B 235/235 · 35B 315/315 · Qwen3-Coder 6/6 · Qwen3-dense 16/16 · OPT 6/6 · DeepSeek-V2 8/8**, goldens unchanged - native always wins by priority and CUDA never misses on the gate models, so no provider is added and no table changes. Hardware-free proof: `tests/vt/test_reference_tier.cpp` (5 cases) runs `vt::Relu` on a fake UNIFIED device with ZERO native kernels for the bit-identical CPU answer, and proves a fake DISCRETE device is REFUSED (`GetOp` still throws - no CPU kernel on true device memory). DSR checker RC=0 (67 == baseline), 24-case mutation suite green, full CPU ctest green (standalone), clean CPU + CUDA `-Werror` 0 warn | **`NOT APPLICABLE`.** The reference tier is a CORRECTNESS device, not a speed one; the only number it will ever own is a correctness gate (OPT token-exact on Metal via the tier). `GetReferenceTierHits()` MUST read 0 in any performance arm - a non-zero value means a missing native kernel and the portable CPU path ran, which voids the arm. The S4-deferred fp4/fp8 gates stay deferred (verified not byte-identically convertible by S5); DSR does not drop here and that is expected. Repro: `./tests/test_reference_tier` |
| KV persistence to disk + external KV providers / LMCache (`KV-OFFLOAD`, `KV-EXTERNAL-CACHE`, `KV-CONNECTORS`) | **NOT APPLICABLE - SPIKE ONLY; NO NUMBER MEASURED OR CLAIMED** (2026-07-22, `CLAIM-KV-PERSISTENCE-LMCACHE`; [spike](../.agents/specs/kv-persistence-lmcache.md)). Records-only change: no code, no kernels, no build, no GPU work, nothing downloaded. Three rows `INVENTORIED` -> `SPIKE`. The deliverable is a 60-feature parity verdict over pinned vLLM's `kv_offload` tiers and connector ABI, plus a design for our disk format - not a speed result | **TWO independent reasons no offload benchmark can be run correctly today, both verified in source.** (1) INHERITED from the caching spike: we have **no prefix-cache hit-rate statistics at any level** (`src/vllm/v1/core/kv_cache_manager.cpp:139` is a comment), so no arm can satisfy the protocol's requirement to PROVE cache hits, and an arm that cannot prove a hit is void. (2) NEW, and our own: **our block hashes are not stable across processes** - `NONE_HASH` is filled from `std::random_device` when unseeded (`src/vllm/v1/core/kv_cache_utils.cpp:307-319`) and the sole production caller passes no seed (`src/vllm/entrypoints/model_loader.cpp:144`), so a content-addressed disk tier would score **0% hits on restart**. This also FALSIFIES the caching spike's section-B2 claim that we are deterministic by construction; we are in fact WORSE than vLLM, which exposes `PYTHONHASHSEED` as an escape hatch and warns when it is unset. Separately recorded so it is not re-derived: vLLM's own `fs` tier writes a `config.json` it NEVER READS, and its only identity check is a path digest omitting checkpoint content, weight quantization, rope config and `sliding_window` - we will not copy that, and a mismatched identity must REFUSE rather than warn | **`PENDING` - the first number this track owes is the W3 disk round-trip and identity-refusal gate (CPU, no GPU), then the W4 every-axis grid vs vLLM launched with the equivalent `--kv-transfer-config`.** Order is fixed by dependency, not preference: W1 deterministic hashes -> W2 CPU primary tier (upstream's disk tier is reachable only through it) -> W3 disk tier -> W4 tiering + connector scheduler half. **W3/W4 are additionally GATED behind the caching spike's W1 hit counters**; until those exist every offload arm is void regardless of what this row implements. Constraints that cannot be traded: an offload-ON arm may never be scored against an offload-OFF vLLM denominator; a hit-rate gain that regresses ANY binding axis is a failed change; and disk-tier tests must bound and clean their `root_dir`, since upstream's tier has no eviction and a 16 GiB dev box or a 95%-full dgx turns ENOSPC into bogus unrelated test failures. Repro: none - nothing was executed |
| Prompt / prefix caching parity (`KV-PREFIX-CACHE`, `KV-BLOCK-POOL`, `KV-HYBRID-COORD`, `KV-MAMBA-ALIGN`, `KV-EVENTS`, `ENG-CASCADE-ATTN`) | **NOT APPLICABLE - SPIKE ONLY; NO NUMBER MEASURED OR CLAIMED** (2026-07-22, `CLAIM-PREFIX-PROMPT-CACHING`; [spike](../.agents/specs/prefix-prompt-caching-parity.md)). Records-only change: no code, no kernels, no build, no GPU work, nothing downloaded. Three rows `INVENTORIED` -> `SPIKE`, three already-evidenced rows corrected in place. The deliverable is a per-feature parity verdict over the complete pinned-vLLM caching surface, not a speed result | **The audit's benchmark-relevant finding is that NO caching benchmark has ever been run, and one cannot currently be run correctly.** Two independent reasons, both verified in source: (1) we have **no prefix-cache hit-rate statistics at any level** - `PrefixCacheStats` is unported and the record call is a comment at `src/vllm/v1/core/kv_cache_manager.cpp:139` - so no arm can satisfy the protocol's requirement to PROVE cache hits, and logical input-token counters do not reveal prefix hits; (2) every existing binding grid was run **cache-OFF** on the hybrid gate models, which is the correct mirror of vLLM's hybrid default and therefore says nothing about APC. Separately, the previously recorded blocker "model-positive APC is blocked on a supported non-hybrid model family" is now **STALE**: Qwen3-dense, Qwen3-32B-NVFP4A16, Qwen3-Coder-30B, OPT and DeepSeek-V2-Lite have all landed and dense models default prefix caching **ON**, so a cache-ON gate is runnable today and has simply never been run | **`PENDING` - the first caching number is owed at work row W3**, the first-ever cache-ON model gate on a dense model, and is GATED behind W1 (hit-rate counters) because an arm that cannot prove hits is void. Order: W1 counters (CPU) -> W3 dense cache-ON token-exact + every-axis grid vs vLLM (DGX) -> W6 Mamba `align` (DGX), which is what [`BACKEND-GATE-CUDA-SGLANG-PREFIX`](../.agents/backend-matrix.md) needs for its hybrid cache-on arm. Constraint that cannot be traded: a cache-ON arm may never be scored against a cache-OFF vLLM denominator, and a hit-rate gain that regresses ANY binding axis is a failed change, not a win. Repro: none - nothing was executed |
| KV persistence to disk + external KV providers / LMCache (`KV-OFFLOAD`, `KV-EXTERNAL-CACHE`, `KV-CONNECTORS`) | **W1-W3 IMPLEMENTED; NO THROUGHPUT/LATENCY NUMBER MEASURED OR CLAIMED - and none is owed yet** (2026-07-22, `CLAIM-KV-PERSISTENCE-LMCACHE`; [spike](../.agents/specs/kv-persistence-lmcache.md)). CPU-only work: deterministic block hashes (W1), the CPU primary tier (W2) and the disk `fs` tier (W3) all landed with unit gates. **A speed number would be meaningless at this stage and is deliberately NOT published:** nothing in the scheduler or the runner consumes these tiers yet (W4 tiering manager + the `OffloadingConnector` scheduler half, W5 the connector seam), so there is no engine path whose latency could change and no configuration to A/B. Publishing a synthetic file-IO rate here would be a number without a workload. | **Two REAL measurements, both correctness evidence rather than speed.** (1) **Cross-PROCESS hash determinism, the precondition for any persisted cache:** `tests/vllm/v1/test_none_hash_determinism.cpp` re-execs the test binary via `/proc/self/exe` and compares chained block-hash lists produced by SEPARATELY LAUNCHED processes - identical with zero configuration (7/7 cases). The negative control matters: with the opt-in `VLLM_PREFIX_CACHING_HASH_SEED=random` the two processes DISAGREE, which is exactly the state the tree was in before this change and which would have scored **0% hits on every restart**. `$PYTHONHASHSEED` is mirrored and honoured. (2) **Disk-tier hit rate across a restart: 6/6 blocks (100%)** - real chained block hashes written by one `FileSystemTier` and read back byte-exactly by a freshly constructed one (`tests/vllm/v1/test_kv_offload_fs.cpp`). Safety evidence, which is the point of this feature: a **27-field identity-refusal matrix** with a positive control - a block staged where a differently-configured run would look for it is REFUSED, per field (model, config digest, weight quantization, KV dtype and quant mode, rope config, sliding window, spec kind, page size, block sizes, hash algo, the NONE_HASH chain seed, parallelism, format version). Truncated, foreign-magic and misfiled blocks are refused and self-healed. The byte budget is honoured including across a restart. **Upstream would silently serve every one of those cases**: its `config.json` is written and never read, and its path digest omits checkpoint content, weight quantization, rope config and `sliding_window`. Gates: clean CPU `-Werror` build 0 warnings, full CPU ctest green, CUDA build unbroken, all six dgx model regressions UNCHANGED. **INDEPENDENTLY VERIFIED 2026-07-23 at SHA `a2112b2` (not carried over from the authoring commit, whose gates were never run): clean dgx CUDA `-Werror` build 0 warnings; the 4 new suites green on CUDA (determinism 7/7, stats 12/12 @0.75, cpu 21/21, fs 22/22+3skip @6/6); regressions each STANDALONE 27B 235/235, 35B 315/315, Coder 6/6, dense 16/16, OPT 6/6, DeepSeek-V2 8/8 (223/223) - DeepSeek needed a standalone re-run to clear the documented "95/223 under contention" transient, 223/223 ×3 idle; memcheck 0 errors/0 leaks on both offload tests (CAVEAT: they run the kCPU backend, so the new `cudaEventQuery` `QueryEvent` override compiles under `-Werror` but is exercised by no test); goldens md5 `2965ef5772b556d3f3f86fedf4221b2f` identical before/after.** | **The first number this track owes is a W4 end-to-end offload arm**, and it is correctly blocked until W4/W5 wire the tiers into the scheduler. When it runs: token-exact with offloading ON, OFF and against the pinned vLLM oracle (an offload hit may change timing, never a token), a non-zero cache-hit rate proven in BOTH engines' logs - now possible, since the prefix-cache counters landed in the same change - and every-axis parity vs vLLM launched with an equivalent `--kv-transfer-config`. Constraint that cannot be traded: offloading is opt-in and must be provably INERT when off (a same-binary A/B reproducing the prior binding numbers within run noise). Repro for what IS measured here: `cd build && ctest -R 'test_none_hash_determinism\|test_kv_offload_cpu\|test_kv_offload_fs'` |
| Prompt / prefix caching parity (`KV-PREFIX-CACHE`, `KV-BLOCK-POOL`, `KV-HYBRID-COORD`, `KV-MAMBA-ALIGN`, `KV-EVENTS`, `ENG-CASCADE-ATTN`) | **W1 STATISTICS LANDED - the first cache-hit rate this project has ever been able to report** (2026-07-22, `CLAIM-PREFIX-PROMPT-CACHING`; [spike](../.agents/specs/prefix-prompt-caching-parity.md)). The prior disposition recorded that no caching benchmark could be run CORRECTLY because we had no hit-rate statistics at any level and an arm that cannot prove hits is void. **That blocker is now closed.** `PrefixCacheStats` and `CachingMetrics` are ported 1:1 from `vllm/v1/metrics/stats.py:35-142`, `log_stats` DEFAULTS ON mirroring upstream's `disable_log_stats=False`, and the rolling rate is readable from `Scheduler`/`EngineCore`/`LLMEngine::prefix_cache_metrics()`. Still NOT claimed: any speed or model-level correctness result, because no end-to-end gate has yet been run cache-ON (that is W3). | **MEASURED prefix-cache hit rate: 0.75 - 1920 of 2560 queried TOKENS served from cache over 16 requests** (`tests/vllm/v1/test_prefix_cache_stats.cpp`, CPU, deterministic). Workload: 16 requests each sharing an 8-block / 128-token system prompt plus 2 blocks (32 tokens) unique to the request, admitted one per scheduler step. The number is a PREDICTION that was then confirmed, not just an observation: request 0 populates the cache and hits nothing, requests 1-15 hit all 8 shared blocks, and queries count `request.num_tokens`, so 15 x 128 / 16 x 160 = 0.75 exactly. **Negative control: the identical workload with caching OFF reports 0.0**, so the counter is measuring the cache rather than itself. Semantics gated against upstream: queries/hits count TOKENS (not requests or blocks); previously-preempted requests go ONLY into the mutually-exclusive `preempted_*` triple, so their guaranteed re-hit cannot inflate the headline rate; empty observations never slide useful history out of the 1000-request window; the newest observation is never dropped; and `reset_prefix_cache` clears the aggregate BEFORE the next fold rather than blending pre- and post-reset rates. 12/12 cases. | **Unchanged and still owed: W3, the first-ever cache-ON model gate on a dense model** (DGX) - token-exact ON/OFF/oracle plus an every-axis grid vs vLLM, then W6 Mamba `align`, which is what [`BACKEND-GATE-CUDA-SGLANG-PREFIX`](../.agents/backend-matrix.md) needs for its hybrid cache-on arm. That gate is **no longer blocked on counters** - the remaining precondition is simply GPU time. Constraint that cannot be traded: a cache-ON arm may never be scored against a cache-OFF vLLM denominator, and a hit-rate gain that regresses ANY binding axis is a failed change, not a win. Repro for the number above: `cd build && ctest -R test_prefix_cache_stats` (or `./tests/test_prefix_cache_stats -s` to see the printed rate). |
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | **NEW BINDING `246a23c`: 49/124** (fresh interleaved exact-grid rerun; supersedes `3f256ab`'s 55/124, retained immutable). Per concurrency: c1 **20/20**, c2 4, c4 5, c8 4, c16 6, c32 6, memory **4/4**. Memory + c1 + every TTFT axis sweep clean for the first time (windowed-load `cb2d310` binding; ours peak PSS 24.88 GB vs vLLM 28.18 GB). Failure mass is the decode-coupled family at c2-c32 (mean TPOT 2.2-6.5% slower) + two ITL tail anomalies (c8 p99_itl 0.5599, c32 p90_itl 0.7925). HONEST regression: ours c16/c32 total throughput dropped −2.67%/−3.64% since `3f256ab` (790.63/1081.10 vs 812.30/1121.95) while vLLM held; the old c16/c32 wins are GONE. Evidence root `~/work/vllm.cpp-online-gate/evidence/246a23c…`; ratios.json `f784ba01…e046`, all-runs.json `b7ef3442…3240`, manifest.json `7f25c614…83e8` | HYPOTHESIS (labeled, unproven): `3f256ab`'s c16/c32 throughput was inflated by the silent GDN slot-sharing defect (2 long requests / 1 state slot), removed by the `c172336` correctness fix - plus every other change between the SHAs. The **era A/B** completed and VALIDATED the hypothesis (see "Next levers" above). The **c2/c8 full-step attribution is now DONE** (2026-07-16, [spec](../.agents/specs/c2-c8-attribution-2026-07-16.md)): c2 entirely GPU-busy (kernel glue + recurrence; idle Δ negative), c8 = 38.6% busy + 61.4% wave-boundary scheduling; per-step host window not exposed (≤0.2 ms/step). Next levers (order-0): kernel glue (`KERNEL-EW-NORM-ACT`) + GDN recurrence tiling for c2-c4; admission grading/async-sched family for the c8+ wave-boundary mass (same mechanism as the attributed c8 p99 / c32 p90 tails). 35B blocked until 27B 124/124
| `ENG-MOE-SHARED-AUX` | **DONE - DEFAULT FLIPPED ON 2026-07-19 (mirror vLLM's decode overlap), token-exact; `CLAIM-MOE-SHARED-AUX-1`.** MoE **shared-expert MLP** on a 2nd persistent CUDA stream concurrent with the **routed-expert** router/align/grouped-Marlin-GEMMs (main stream) inside `MoeBlockFusedMarlinCuda`, joined before the combine - 1:1 mirror of `fused_moe/runner/shared_experts.py:99-104,125-142` + `maybe_execute_in_parallel` (`multi_stream_utils.py:20-58`, TRT-LLM port). Byte-identical by construction (independent shared/routed paths both complete before combine). The scratch `DevicePool` single-stream reuse invariant is kept by an isolated `AuxPool` for the aux stream (the cross-stream race vLLM avoids via `record_stream`). `VT_MOE_SHARED_AUX_STREAM` default ON (`=0` rollback), `VT_MOE_SHARED_AUX_THRESHOLD` 128 (GB10 48-SM calibration) | **DGX (production flags CUTLASS sm120a + Marlin + FA2 sm_121a + Triton AOT, one flock, `~/work/vllm.cpp-moe-shared-aux`):** overlap **ON == OFF BYTE-IDENTICAL** - `VT_MOE_SHARED_AUX_STREAM`∈{0,1} both 35B `test_qwen36_paged_engine` **315/315** + 27B `test_qwen27_paged_engine` **235/235**; captured-vs-eager (`VLLM_CPP_CUDAGRAPH=0`, ON) 315/315; shipping default (no env) 315/315 + 235/235, rollback `=0` 315/315 + 235/235; `compute-sanitizer memcheck` (default ON, captured graph) 0 errors. **In-situ same-binary interleaved TPOT A/B** (35B, input-1024/output-128, greedy, `=1` vs `=0`, 3 reps/arm, cold rep1 dropped): **c1 13.60 vs 14.40 ms −5.6%** (tput +5.1%), **c2 17.45/17.94 −2.7%**, **c4 21.04/21.84 −3.7%**, **c8 30.47/31.55 −3.4%**, **c16 47.36/48.11 −1.6%**, **c32 72.30/73.43 −1.5%** - WINS at every concurrency, zero regression | Token-exact + faster c1-c4 + non-regressing (winning) c8+ ⇒ default flipped ON. `benchmark_binding=false` - the orchestrator re-grids the 35B binding c1/c2/c4 TPOT (this is the first slice of the multi-stream OVERLAP named the largest c1/c2 lever; remaining routed/attention/prefill slices + portable glue-fusion follow). Evidence `dgx:/tmp/moeaux_{correct,ab,gate,mc_full}.log` |
| `ENG-ASYNC-SCHED` (W3) | **DONE - DEFAULT FLIPPED ON 2026-07-17 (mirror vLLM), DGX-re-confirmed TOKEN-NEUTRAL; owner `6ea7856`.** `VT_ASYNC_RUNNER` defaults ON, so depth-2 `AsyncScheduler` + `step_with_batch_queue` + runner async input-combine/copy-stream D2H + `LoadedEngine` mcb=2 is the production default; `VT_ASYNC_RUNNER=0` / `VT_ASYNC_SCHED=0` roll back. DGX re-confirm (`dgx:~/work/vllm.cpp-async-flip`, CUTLASS+FA2 hard-verified, one flock): all three async arms bit-identical; shipping default (async ON + RMSNorm-fast OFF) **27B 235/235 + 35B 315/315** log "enabled (mcb=2)", rollbacks 235/235+315/315 "disabled". TTFT rises into vLLM's async envelope BY DESIGN (the mirrored trade); no new A/B (the discriminator's `f086b64`/`6ea7856` stands) | **DGX proof `f086b64` (root `dgx:~/work/vllm.cpp-w3-proof/f086b64…`):** 5/5 correctness gates PASS (27B+35B default, both W3-on, rollback - token-exact, arm-log correct). Interleaved c16 A/B (same binary, `VT_ASYNC_RUNNER=1` vs `+VT_ASYNC_SCHED=0`): **W3-on** tput 790.9-792.7, meanTPOT **160.9-161.2** (**−5.4 ms/step, WIN**), meanTTFT **2757.9-2778.1**; **W3-off** tput 793.5-794.1, meanTPOT 166.2-166.4, meanTTFT **2028.4-2032.0**. So overlap wins decode (−5.4 ms/step) but c16 meanTTFT **+36 % (+730 ms)** and throughput is **neutral (−0.3 %)**. **DIAGNOSIS (CPU-verified):** the +730 ms is NOT an admission delay - `test_async_admission_timing.cpp` proves depth-2 schedules a new prefill the SAME step as sync/vLLM (`UniProcExecutor` is synchronous, `uniproc_executor.py:91-106`); it is the closed-loop Little's-law consequence of neutral throughput + faster decode (`127×5.4≈686`). `benchmark_binding=false`, no speed credit | **NOT a CPU-fixable regression.** **2026-07-16 throughput-lever attempt RAN and is REFUTED:** the per-step sampler alloc/free serialization (rescan item #2 - `cudaMalloc`/`cudaHostAlloc`/events per step + the overlap-killing `cudaFree` inside `get_output`) was removed with persistent pooled buffers (`AsyncOutputPool` + `Sampler` greedy scratch, mirror of `gpu_model_runner.py:873-878` + `async_utils.py:12-70`). Clean re-proof (dgx `~/work/vllm.cpp-w3-tput/ab-fix2`, CUTLASS+FA2 hard-verified build): token-exactness **6/6 PASS**; interleaved c16 (w0+3 pairs): W3-on tput 788.14-790.58 / meanTPOT 161.60-162.04 / TTFT ~2732 vs W3-off 790.32-792.44 / 166.69-167.00 / ~2027 ⇒ **tput −0.32 % - gate ≥+1.5 % FAILS**; TPOT −4.95 ms retained; identical to pre-fix `f086b64`. Ceiling analysis: the removed syncs are O(10-100 µs)/step ≤0.1 % of a ~165 ms c16 step - two orders below the gate. (First attempt VOID: configure omitted `-DVLLM_CPP_CUTLASS_DIR` ⇒ WMMA fallback ~50 tok/s both arms + FA2 dropped ⇒ gates failed on `kv_cache_backend_resident()`; recorded, no verdict drawn.) The pooled-buffer structure LANDS (token-exact, no speed credit). W3 **stays default-OFF**; the depth-2 tput lever moves to the c8+ wave/overlap family (`62d4762`). **Re-proof gate unchanged:** ACCEPT = TPOT win retained AND TTFT within ~2 % of sync (achievable only with a throughput gain) |
| `KERNEL-GEMM-BF16` | **GATING - W2A qkvz implemented, DGX gates PENDING** | W1 BA: `0091cd1` closes structure, `f925294` closes projection/inertness, clean `f344dec` closes W1D2/G2 at **235/235**. **W2A qkvz (2026-07-15, test-first):** one raw-NK `[q,k,v,z]` owner (loader), ONE BF16 GEMM + strided mixed/z views on the CUDA default, split rollback from the same owner (`VT_GDN_MERGED_QKVZ=0`/`VT_GDN_MERGED_PROJ=0`), CPU merged-owner == split bit-exact, 35B/GGUF inert; CPU gates green (full CTest 107/107, tools 162/162, clean -Werror rebuild). `benchmark_binding=false`, no speed credit. **DGX gates at `baea3ec`: jobs 1 (default suites 8/8), 2a (`VT_GDN_MERGED_QKVZ=0`), 3 (35B native + legacy) PASS; 2b (`VT_GDN_MERGED_PROJ=0`) failed in the TEST contract only** - engine correct (master-off deselects packed decode by the designed BA coupling; 229/229 token asserts green); fixed test-first (shared `PackedGdnDecodeEnvSelected` truth table, CPU 16/16; serial CTest 107/107 re-green). First memcheck run was PATH-only (`compute-sanitizer: command not found`). **DGX gates then closed GREEN at `45f9e6d`**: default suites 8/8, both rollback arms, 35B inert, memcheck 0 errors/0 leaks, structural trace confirmed **−48 BF16 GEMMs/window** (145→97; wmma-BF16/packed-window 1.959≈2.0 merged vs 2.980≈3.0 split). qkvz is in the `246a23c` binding binary | **W2A closed green.** qkvz rides the `246a23c` binding exact-grid rerun (49/124; no isolated qkvz speed credit). Remaining `KERNEL-GEMM-BF16` work: none for attribution - the c2/c8 full-step split (2026-07-16) closed it (GEMM bundle at parity c2, ours-faster c8); open GEMM-adjacent lever is the GDN recurrence tiling |
| `KERNEL-GDN-PACKED-DECODE` | **`DONE` - W1D3 CLOSED on EQUIVALENCE** (owner `e47b4d6`) | Slot lifecycle fix `c172336` (identity-keyed pool; RED→GREEN `test_runner` 8/8) proven on DGX (`--diagnostic-c16` 3/3, both model gates 235/235); scheduler co-schedule parity with vLLM verified (`test_scheduler` 31/31). W1D3/G3 closed on evidence-totality: **eight sealed components** (harness calibrated test-first; focused suite 79/79, tools 162/162) + the 8-pair locked c16 A/B (`00bf484`: paired mean **−0.205% ± 0.30, <1σ**; cuBLASLt algo selection process-deterministic → algo-lottery REFUTED) + the 24-window trace (**packed is GPU-cheaper**: kernel compute −1.30..−1.58%/step, GDN+BA −296 µs/window, no attributable packed-side cost). The eighth (22-leg) seal `complete-failed` at `e47b4d6`: **38/40 + 8/8 memory**, stability clean, paired-consistency PASS at both concurrencies; the 2 fails are sign-flipping band-edge statistics of a true-zero effect. Full chronology and per-seal SHAs in state/ledger | **Disposition: EQUIVALENCE PROVEN - no stable regression on any axis.** Packed is the default (`VT_GDN_PACKED_DECODE=0` rollback); **no `complete-pass` marker exists and no speed credit is claimed**. qkvz (`KERNEL-GEMM-BF16` W2) closed green (`45f9e6d`); the authorized exact grid has now RUN (new binding `246a23c`, 49/124)
| RMSNorm/generated partitions ("norm/quant fusion") | **CLOSED / DISPROVEN as a parity gap** (RECONFIRMED 2026-07-16) | The 2026-07-14 [parity rescan](../.agents/specs/parity-rescan-2026-07-14.md) verified vLLM's `RmsNormQuantFusionPass` is FP8-only (no nvfp4 keys); the nvfp4 path runs standalone `scaled_fp4_quant` exactly like ours, and the +1.81 ms residual was a cross-profiler artifact. **2026-07-16 [reconciliation](../.agents/specs/decode-norm-quant-fusion-reconcile-2026-07-16.md)** re-verified this against the fresh correct-state c16 trace (count parity `cvt_fp16_to_fp4` 144/win == ours 144/step; `3f256ab` body-dump stores bf16 then quantizes separately) + a decode-shape microbench (isolated RMSNorm 6-9 µs), and CORRECTED the 2026-07-16 `SUMMARY.json` "vLLM fuses add+rmsnorm+fp4quant" note. Fusion-attributable headroom ~0 ms; the modest non-bit-exact efficiency residual is reassigned to `KERNEL-EW-NORM-ACT` | None - fusion lever stays CLOSED; efficiency redirect is spike + in-situ-A/B-gated, not implemented |
| fp8 cuBLASLt plan cache (`KERNEL-GEMM-FP8`) | **DONE / MEASURED NEUTRAL - bit-exact mirror, OPT-IN (`VT_FP8_PLAN_CACHE=1`), premise disproven** | The fp8 dense GEMM (`cuda_matmul.cu`) rebuilt the cuBLASLt descriptor + 3 TN layouts + heuristic every call; vLLM reuses an in-graph plan. Added a per-device `{desc,layouts,algo}` cache keyed on the full shape/config (`fp8_plan_cache.h`); BIT-EXACT (algo process-deterministic per shape - byte-exact cached==fresh `test_ops_fp8_cutlass` + CPU key/flag 4/4; 27B 235/235 + 35B 315/315 both flags). **Same-binary 35B A/B (one flock) is wall-clock NEUTRAL**: prefill TTFT in1024/c8 async-on ON 1491.5/OFF 1496.8, async-off ~1496.7/~1503.2; decode TPOT c1 15.16/15.14, c4 21.79/21.83. **nsys (async-off eager prefill): the pre-fp8-GEMM GPU gap is UNCHANGED - median 210 µs (off) vs 204 µs (on)** ⇒ the grounded "~0.8 ms removable gap" premise is NOT reproduced; the heuristic host cost is negligible/hidden (prefill GPU-bound; decode graph-captured so the heuristic runs at capture, not per replay-step). Kept opt-in for eager/non-graph regimes; default NOT flipped (faster unmet). Evidence `dgx:/tmp/fp8pc-*` + `~/work/prefill-attr-35b` | None - the vLLM-mirror lands opt-in; no default-flip and no speed credit |
| Serving transport (TCP_NODELAY) | **DONE / MEASURED NEUTRAL on the gate workload** | Mirror landed (`SERVE-HTTP-TRANSPORT`): `set_tcp_nodelay(true)` matches vLLM's uvicorn/asyncio default; behavioral accepted-socket test RED **0** → GREEN **1**, 22/22 cases. The non-binding one-lock localhost A/B (`~/work/vllm.cpp-tcpnodelay-sizing/ff915e8…`, 4a450f9 Nagle-ON vs ff915e8 Nagle-OFF, c1/c2 ×2 reps, identical pinned-client workload; raw-set SHA `f5b52900…2128`) is **neutral within noise** on every ITL/TPOT/throughput metric (c1 mean ITL ~102.7 both arms; c2 ~108-109; first cold-start leg excluded). Mechanism: ~100 ms per-token write cadence vs µs loopback ACKs means Nagle never coalesces - the rescan's rank-1 gain hypothesis is REFUTED for the loopback gate; the mirror stays for real-network parity | None for the gate - the c2/c8 full-step attribution is complete (transport ruled out; c2 gap is GPU-busy kernel glue) |
| Block-table host-cluster cleanup ([rescan](../.agents/specs/rescan-lost-lanes-2026-07-16.md) §1,§5,§6) | **CPU-SIDE, BIT-IDENTICAL; `benchmark_binding=false`, no speed credit** - the payoff is measured by the dispatched correct-state c2/c8 full-step probe and the next authorized exact grid, not by these mirrors. **(c) LANDED** - `block_table.compute_slot_mapping` drops the dead tail-pad (`~2×(max_num_batched_tokens−total)` int64 writes/step; the decode graph re-pads via `BuildPaddedDecode`, the only other consumer slices `[0,total)`); test_block_table 11/11, test_prepare_inputs 6/6. **(d) LANDED** - the decode-graph capture-size set is DERIVED from `max_num_seqs` (`include/vllm/model_executor/models/decode_graph_sizes.h`, `DecodeGraphSizes`/`PadToCaptureSize`; mirrors vLLM `_set_cudagraph_sizes` reduced to the full-decode-cudagraph regime): `max_num_seqs=32` → `{1,2,4,8,16,24,32}` (adds the missing 24 bucket, drops the never-reachable 64; batches 17-24 stop over-padding to 32, +1 captured graph). CUDA-only; padding rows inert → token-exact. RED→GREEN `test_decode_graph_sizes` 5 cases/478 asserts (RED = 24 bucket absent under the old fixed set). **(e) LANDED** - `InputBatch::make_sampling_metadata` caches + rebuilds only on batch change (add/remove/condense/swap set a dirty flag), mirroring vLLM `refresh_metadata` (gpu_input_batch.py:812-830); deviation: the penalty-active path rebuilds every step (our port copies output_token_ids where vLLM holds a live ref), so the greedy/no-penalties gate gets the full win bit-identically. `scheduler.cpp:371,377` `std::move`s the `num_scheduled_tokens` map + `finished_req_ids` set (container plumbing, zero policy change; move-then-clear keeps observable behavior identical). RED→GREEN `test_input_batch` (+2 cases; RED = stale cache after a 2nd add returns size 1 != 2); `test_scheduler` 31/31, `test_runner` 228/228, `test_sampling_metadata` 6/6 unchanged. Items (a)-runner + (b) (GDN col-0) are in `runner.cpp` (async/GDN-claim owned) - reported, not touched. Landed `8a717b2`/`81afc36`/`0c4b41c` (merged `e027ad5`); clean full `-Werror` rebuild 0 warnings, CPU battery + tools 164/164 green. **DGX token-exactness gate PASSED** on the `e027ad5` build (GB10, one `flock /tmp/gpu` series, root `dgx:~/work/vllm.cpp-blocktable-gate`): 27B default **235/235**, 27B `VT_GDN_PACKED_DECODE=0` rollback **235/235**, 35B **315/315**, all exit 0 | **CLOSED - claim released.** No A/B by design (`benchmark_binding=false`, no speed credit); payoff measured by the c2/c8 attribution probe + the next authorized exact grid |
| Host-weight ownership | **Established 27B binding memory axes remain PASS at `246a23c`**. **Local plain-BF16 4B checkpoint (`LOAD-SAFETENSORS-DIRECT-DENSE`, `GATING`):** corrected AOT root `/tmp/qwen35-transplant-4b-aot-557ab41d` used the exact 128-request / 131,784-input / 16,384-output / c32 corpus and completed 18/18 guarded legs. Direct ON/OFF/local-vLLM-0.24 means: total **6155.10/6064.06/6730.46 tok/s**, output **680.61/670.54/744.24**, request **5.317/5.240/5.814 req/s**, mean TTFT **722.56/815.05/903.20 ms**, mean TPOT/ITL **41.44/41.43/33.56 ms**, mean E2EL **5985.90/6076.37/5164.87 ms**. Peak PSS **2.405/8.571/7.569 GiB**, stable PSS **0.733/8.571/4.066 GiB**, peak VRAM **12892/12884/12942.7 MiB**. ON=OFF outputs equal 128/128 in all three pairs; ON is +1.50% total and −71.9%/−91.4% peak/stable PSS. | **Corrected AOT diagnostic complete; performance gap remains.** Previous AOT ON was total/output/request **6607.04/730.59/5.710**, mean TTFT/TPOT/E2EL **658.92/38.74/5578.69 ms**, peak/stable PSS **1.768/0.757 GiB**, VRAM **11692 MiB**. Current changes are therefore **−6.84% total/output, −6.89% req/s, +9.66% TTFT, +6.99% TPOT, +7.30% E2EL, +36.0% peak PSS, −3.20% stable PSS and +1200 MiB VRAM**; current ON is **0.9145x** local vLLM. Current local vLLM is 1.002x the previous reference, so the reproducible residual is project/build-side. Build proof: `VLLM_CPP_TRITON=ON`, regeneration ON, generated arch `sm_120`, FlashInfer CUTLASS; current sources require CMake CUDA arch `120a` whereas the preserved previous cache used `120`. nsys exact-workload traces show AOT `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`, `recompute_w_u_fwd_kernel` and `chunk_scaled_dot_kkt_fwd_kernel`; the 27B-only `gdn_decode_h48` cubin is ineligible, so the 4B decode trace correctly contains the hand `GdnPackedDecodeKernel`. Aggregate SHA-256 `6ff00982…8ed`; project/vLLM trace SHA-256 `f2e73f4b…681` / `cbe2afbc…9a9`. Reproduction: `REQUIRE_TRITON_AOT=1 CPP_BENCH=$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench CMAKE_CACHE=$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh /tmp/qwen35-transplant-4b-aot-<commit>`; the driver puts the live NVIDIA driver before Nix's link-only `libcuda` stub and fails closed if AOT is not enabled. **27B/35B/Coder REGRESSION + SANITIZER NOW RUN AND GREEN (2026-07-21, PR #5 rebased onto `a63c497`, worktree `worktree-agent-a11d47f6329f70c95`):** clean full dgx rebuild, CUDA `-Werror` 0/0, goldens md5-verified, FINAL binary - 27B `test_qwen27_paged_engine` **235/235**, 35B `test_qwen36_paged_engine` **315/315**, Qwen3-Coder `test_qwen3coder_paged_engine` **6/6** (max teacher-forced gap 0.0000 nats), `test_ops_moe_grouped_bf16` 7/7, NVFP4 `test_ops_moe_grouped` 9/9, `test_qwen3_moe_forward` 3/3, PR tests `test_model_registry` 18/18 + `test_cuda_backend` 5/5 (`pageable=1 integrated=1 UnifiedMemory=true` - the classification fix is BEHAVIOR-NEUTRAL on GB10) + `test_bench` 4/4; `compute-sanitizer memcheck` on the 27B engine path **0 errors** (235/235 under the sanitizer); `test_qwen35_plain_weights` 3/3 but its REAL-CHECKPOINT leg **SKIPS** (`Qwen/Qwen3.5-4B` absent from dgx), so the headline direct-device gate is NOT exercised on this hardware. **THE −6.84% IS NOT A CODE REGRESSION AGAINST MAIN:** same-arch `121a` interleaved same-box A/B of `origin/main` vs the rebased head on 27B NVFP4 1024/128 c8/32-prompts (4 pairs, one flock, cold pair dropped, `dgx:/tmp/ab_c8.log`) is neutral on every axis - PR/main output throughput 58.27/58.30 tok/s **0.9995×**, median TPOT 128.51/128.47 ms **+0.03%**, median TTFT 825.04/826.18 ms **−0.14%**, all inside ±0.15% against a ~0.3% per-arm spread. The arch flag is independently confirmed a first-order lever: `CMakeLists.txt:71,88,97,137,612` gate `VT_FP4_MMA_SM120A`, the CUTLASS NVFP4 GEMM, the vendored Marlin NVFP4 MoE GEMM and (via `VLLM_CPP_CUTLASS`) the vendored FlashAttention-2 on `MATCHES "12[01]a"`, and `cmake/TritonAOT.cmake:98-101` derives the vendored AOT subdir as `sm_<arch>` - measured same-binary on the 27B (c1, `dgx:/tmp/fa2ab.log`), FA2 ON/OFF alone is output throughput **9.64/9.28 tok/s (+3.9%)**, median TTFT **403.7/634.5 ms (−36.4%)**, median TPOT **101.29/103.52 ms (−2.2%)**. STILL `PENDING` (richiejp's sm_120 box + the 4B checkpoint only): the current-v0.25 4B correctness oracle, strict ON<=OFF VRAM (+8 MiB), and the same-source `120`-vs-`120a` A/B that would actually close the −6.84% attribution - the PR asserts that attribution but has not measured it, and its stated direction is counter-intuitive because `120a` compiles IN more fast kernels than `120`. No support extrapolation; no 4B result implies 27B/35B support. |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN (grid crash FIXED)** | Correctness passes; no current v0.25.0 performance denominator exists. The 35B online-serving **c2+ crash that blocked the grid is root-caused + FIXED** (2026-07-18, `CLAIM-35B-GRAPH-SCRATCH-1`, [spec](../.agents/specs/decode-graph-scratch-uaf-2026-07-18.md)): at concurrency > 1 the engine died with `cudaEventSynchronize: an illegal memory access`. cuda-gdb pinned the faulting kernel to `marlin_moe_wna16::Marlin<…><<<(144,1,1),(128,1,1)>>>` - its fp32-reduce scratch `c_tmp` (`EnsureCtmp`, `cuda_moe_marlin.cu`) is a grow-on-free per-stream buffer whose pointer is baked into the captured pure-decode CUDA graph; a bigger later prefill/decode `cudaFreeAsync`s it → the next graph replay reads freed memory (single-stream c1 never grows it → never crashed). Differential isolation: needs graphs ON + concurrency > 1 + long context; async/WMMA ruled OUT (async-off + wmma-off still 5/5 crash; graphs-off CLEAN); memcheck masks it. Fix = retire-on-grow (`RetireGraphScratch`, `src/vt/cuda/graph_safe_scratch.h`) across the four decode-graph-reached scratch allocators. Sweep c1-c16 pre-fix 5/5 crash → post-fix 0; 315/315 token-exact preserved. `benchmark_binding=false` (correctness fix, no speed credit) | Run the v0.25.0 performance grid (now unblocked) after 27B reaches 124/124 |
| 35B FA2-prefill + fused-preamble lever | **LANDED / default-ON (2026-07-18, `CLAIM-35B-FA2-FLIP-1`); `benchmark_binding=false` (offline op A/B, no grid speed credit - the 35B grid re-measures in-situ)** ([spec](../.agents/specs/qwen36-35b-fa2-prefill-oracle-2026-07-18.md)) | `FuseAttnPreambleOn` flipped default-ON all arches ⇒ the 35B ratio-8 full-attn layers take the exact `flash_fwd_splitkv` kernel (kernel-side `fa2_prefill` admits any GQA ratio at head_dim 256, `cuda_paged_attn.cu:2494`). FULL current-main default-set gate (async + GDN cubin + all fast kernels + flip): **35B `test_qwen36_paged_engine` 315/315 + 27B `test_qwen27_paged_engine` 235/235** (`dgx:/tmp/fa2gates_u.log`); **memcheck 35B prefill 0 errors** (`dgx:/tmp/fa2_memcheck2.log`); `test_ops_attn_preamble` 14/14. Realistic input-1024 TTFT A/B (same binary, FA2 default vs `VT_FA2_PREFILL=0`, conc8/num-prompts32, 3 interleaved on/off pairs + dropped warmup, `dgx:/tmp/fa2_ttft2.log`): **FA2-on Mean TTFT 824.7 ms vs off 874.4 ms = −5.7%** (median 638.6 vs 676.9 = −5.7%; prefill token-throughput 5170.5 vs 4902.5 tok/s = +5.5%; per-arm spread <7 ms, so the ~50 ms gap is well-separated). Below the ~7-9% offline-kernel target because the 1.86× attention-kernel win dilutes across the whole prefill (GEMM/MoE/GDN dominate); no decode/TPOT regression (prefill-only lever, both arms 315/315 token-exact). The "round normed q/k→bf16 before RoPE" tighten (`fused_qk_norm_rope.py:67`) was op-level bit-identical (fused-bf16 == unfused-bf16, q/k 0-mismatch) but flipped the 27B tok6 whitespace near-tie away from the pip-vLLM oracle (233/235) in COMBINATION (RMSNorm-saga) ⇒ NOT shipped; preamble ships UNTIGHTENED, both arches token-exact. CPU gate: clean `-Werror` rebuild 0 errors/0 warnings, full DGX ctest **156/157** (both engine gates green; the sole failure `test_capi` is a KNOWN pre-existing nondeterministic dgx-box detokenizer UTF-8 flake - 3 runs of the same binary give 3 different results - not a regression, unrelated to the attention path), tools unittest 164/164, checkers green | The 35B v0.25.0 perf grid (orchestrator-owned) re-measures the in-situ prefill/TTFT gain from this lever |
| 35B FA2-decode lever (split-KV, ratio-8) | **LANDED / default-ON (2026-07-19, `CLAIM-35B-FA2-DECODE-1`); `benchmark_binding=false` (focused same-binary in-situ A/B - the 35B grid re-measures)** ([spec §35B ratio-8 extension](../.agents/specs/fa2-gqa-split-kv-decode.md)) | Extended the ratio-6-only FA2 split-KV DECODE to the 35B ratio-8 (Hq/Hkv=16/2) hd-256 full-attn layers (new env `VT_FA2_DECODE_35B`, default ON). The old ratio-8 decode ran `PagedAttentionDecodeGqaKernel` at **grid=(num_reqs,num_kv_heads) = 2 blocks** at single-request decode (near-zero GB10 occupancy); the vendored `flash_fwd_splitkv` main+combine splits the KV dimension so the grid fills the machine. nsys `--cuda-graph-trace=node` (`test_qwen36_paged_engine`): clean 1:1 decode-kernel swap - OFF `PagedAttentionDecodeGqaKernel<...(int)8...>` ×300 (grid (1,2,1)=2 / (8,2,1)=16, no combine) → ON `flash_fwd_splitkv_kernel` ×300 + `flash_fwd_splitkv_combine_kernel` ×300 (split axis GridZ up to 16), old kernel absent. FULL current-main default-set token gate: 35B `test_qwen36_paged_engine` **315/315** + 27B `test_qwen27_paged_engine` **235/235**; operator `test_ops_paged_attn` **21 cases / 454,358 assertions** (adds ratio-8 parity ladder B∈{1,2,4,8,16} + ratio-4/window/toggle fallback); memcheck 35B decode **0 illegal-access errors** (315/315; `--leak-check full` leaks are engine exit-time model residency). IN-SITU A/B (35B NVFP4, input-1024/output-128, greedy, same binary `VT_FA2_DECODE_35B=1` vs `=0`, one flock, 4 interleaved ON/OFF pairs, first dropped, pooled pairs 2-4): **c1 Mean TPOT 14.96 vs 16.72 ms = −10.5%** (total tput 540.1 vs 489.3 = +10.4%); **c8 Mean TPOT 33.02 vs 34.12 ms = −3.2%** (tput 1835.1 vs 1785.7 = +2.8%); Mean TTFT neutral (c1 ~233, c8 ~826 ms both arms - decode-only lever). Per-arm spread <0.15 ms TPOT, well-separated. Token-exact + faster ⇒ default-ON. Evidence `dgx:~/work/vllm.cpp-35b-fa2-decode/{gpu_series,gates_engine,gates_default,memcheck35,nsys_ON,nsys_OFF}.log`. CPU gate: clean `-Werror` CUDA build 0/0, tools 164/164, checkers green; full DGX ctest `-j8` **157/160** - 3 non-numerics misses: `test_async_llm` (parallel-port flake, passes 1/1 isolated), `test_capi` (documented nondeterministic dgx detokenizer UTF-8 flake, fails even isolated, unrelated to attention), `test_qwen36_gguf_engine` (MEMORY-capacity artifact: loads TWO full 35B GGUF models sequentially; with FA2 decode ON the first runs 16/16 correct through the FA2 decode path then the second model load OOMs; `VT_FA2_DECODE_35B=0` rerun passes 2 cases / 28/28, so the added decode scratch tips an already-marginal ~119 GiB unified-memory box - NOT a correctness regression; the production safetensors NVFP4 35B is 315/315 ON - flagged for a decode-scratch-pool follow-up) | The 35B v0.25.0 perf grid (orchestrator-owned) re-measures the in-situ c1-c4 decode-TPOT gain; this directly targets the 35B c1 decode 0.810× low-batch residual |
| Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) BF16 MoE - W7 | **ACCEPTED NUMBERS / EVERY-AXIS PARITY NOT YET MET - `ACTIVE`** (2026-07-21, `CLAIM-MODEL-QWEN3-CODER`; [spike §9](../.agents/specs/sweep-qwen3-coder-30b.md)) | Breadth-sweep model #1's SPEED row. W5 shipped the fast BF16 grouped-MoE GEMM `vt::MoeGroupedGemmBf16` (**default ON**) plus a host-mirror release that closed the memory axis (ours ~62 GiB vs vLLM's 56.93 GiB weights + 11.38 GiB KV); it missed speed parity on every axis, `nsys`-attributed to an untuned prefill WMMA tile (**56.0% of GPU time at ~4.3 TFLOP/s ~= 1.7% of GB10 bf16 peak**) and a block-starved decode kernel (**21.3% at 151 GB/s = 55% of peak**). **W6 closes that kernel deficit.** Two levers, both `src/vt/cuda/cuda_matmul_nvfp4.cu`, both DEFAULT-ON: (1) **`MoeGroupedGemmBf16WmmaPipe`** - the W5 tile's real defect was an UNCOALESCED `[K,N]` weight stage (consecutive lanes `n_cols` elements apart => 32 sectors per warp, 2 useful bytes each) on a GEMM that is WEIGHT-BANDWIDTH bound (~1.2 GB of expert weight per prefill layer, ~64 flop/byte vs the machine's ~900); fixed with a k-major `[BK][BN]` shared tile read as `wmma::matrix_b` ROW_MAJOR (global read now runs along contiguous N), 16-byte vectorized stages, a **3-deep `cp.async` multi-stage pipeline**, BN 64->128 and +8-half padded shared rows. Tile/pipeline shape ported from vLLM `fused_moe.py:294` `fused_moe_kernel` + `:1238` `get_default_config`; multistage structure from CUTLASS `mma_multistage.h`. Shape-guarded (`Bf16PipeShapeOk`) with the W5 tile retained for ragged pitches; `VT_MOE_BF16_PIPE=0` rolls back. (2) **DETERMINISTIC split-K** for the small-P decode GEMM (`MoeGroupedGemmBf16NaiveSplitK` + a fixed ascending-order `...SplitKReduce`, partials in the graph-safe `EnsureMoePartials`) - mirrors vLLM's `SPLIT_K` (`fused_moe.py:338`) but explicitly NOT `atomicAdd`, whose nondeterministic reduction order would break greedy reproducibility and the SACRED gate; `VT_MOE_SPLIT_K=0` rolls back. **KERNEL A/B (same binary, E=128/K=2048/top-8):** prefill T=1024 N=768 **4.50 -> 12.15 TFLOP/s (2.70x)**, N=2048 **4.49 -> 12.48 TFLOP/s (2.78x)**, decode-WMMA P=64 **2.84x**, split-K P=8 **2.63x** - against vLLM's Triton `fused_moe` **~10.0 TFLOP/s**, so the bf16 MoE GEMM now runs at **~1.2x vLLM's rate** (19.9 TFLOP/s in situ on the c2 2048-token prefill step). **CORRECTNESS UNCHANGED, ZERO TOKEN MOVEMENT** - the pipelined tile preserves the per-output K-reduction ORDER exactly and is BIT-IDENTICAL to W5: `test_qwen3coder_paged_engine` **6/6**, STRICT token-exact **5/6**, max teacher-forced gap **0.0000 nats**, 0 forward-divergent, goldens md5-verified unchanged. `test_ops_moe_grouped_bf16` extended **4 -> 7 cases / 19 assertions** (aligned-pitch pipelined prefill/decode/down shapes with ragged K/N tails exercising the `cp.async` zfill + a run-to-run BIT-REPRODUCIBILITY assertion on the split-K reduction); the new decode case caught and fixed a genuine pipeline bug (tail iterations skipped `__pipeline_commit()`, under-counting `__pipeline_wait_prior`). **DENOMINATOR CORRECTION (retroactive to W5).** W5 drove all four concurrencies against ONE `vllm serve` with `vllm bench serve --seed 0` each time; vLLM's `RandomDataset` builds prompts as `(offset + index + arange(input_len)) % vocab_size` (`vllm/benchmarks/datasets/datasets.py:557-566`), so runs 2-4 REPLAYED run 1's prompts into vLLM's default-on prefix cache - the serve log shows **43.6%-63.5% hit rate**, and vLLM's c2 TTFT came out FASTER than its c1. Our arm is a fresh process per concurrency and never had that carry-over, so the **W5 c2/c4/c8 denominators were inflated and its ratios there PESSIMISTIC**. W6 re-measures with a **FRESH `vllm serve` per concurrency**, confirmed **0.0% prefix-cache hit rate** in all four runs (c1 was always clean: 324.04 -> 321.88 ms, which also bounds run-to-run noise at ~0.7%). Our prefix caching mirrors vLLM and is likewise default-ON for this model (`model_loader.cpp:121-131`). **BINDING GRID** vs vLLM 0.25.0 PRODUCTION GRAPHED (`enforce_eager=False`, `CUDAGraphMode.FULL_AND_PIECEWISE`, Inductor `VLLM_COMPILE`, `moe_backend='triton'`), random 1024/128 `--ignore-eos`, num-prompts 16/24/48/96, idle box under one `flock /tmp/gpu`, checkpoint page-cache evicted - ours / vLLM -> ratio (>1 = we win): **median TTFT** c1 313.9/321.9 **1.03x WIN**, c2 533.6/435.5 **0.82x**, c4 754.1/789.6 **1.05x WIN**, c8 762.3/904.7 **1.19x WIN**; **median TPOT** c1 36.22/31.75 **0.88x**, c2 44.12/44.67 **1.01x WIN**, c4 60.74/74.26 **1.22x WIN**, c8 81.41/85.29 **1.05x WIN**; **median ITL** c1 36.14/31.79 **0.88x**, c2 42.94/44.06 **1.03x WIN**, c4 58.40/71.41 **1.22x WIN**, c8 70.97/77.56 **1.09x WIN**; **output throughput** c1 22.62/29.45 **0.77x**, c2 37.24/41.44 **0.90x**, c4 55.03/50.45 **1.09x WIN**, c8 86.02/87.27 **0.99x**. **12 of 16 cells at/above vLLM; c4 passes EVERY axis; c8 passes 3 of 4** - but every-axis-at-every-concurrency is NOT met, so the row stays `ACTIVE`. Ours vs OUR OWN W5 (identical methodology): median TTFT **904.5 -> 313.9 / 1684.6 -> 533.6 / 2489.6 -> 754.1 / 2501.4 -> 762.3 ms** (2.9x-3.3x); median TPOT **40.40 -> 36.22 / 43.81 -> 44.12 / 98.98 -> 60.74 / 148.85 -> 81.41 ms**; output throughput **18.55 -> 22.62 / 31.85 -> 37.24 / 32.20 -> 55.03 / 46.55 -> 86.02 tok/s**; TPOT no longer degrades pathologically with concurrency (W5 3.4x c2->c8, now 1.8x vs vLLM's 1.9x). **`nsys --cuda-graph-trace=node` + `cuda_gpu_kern_sum` ATTRIBUTION of the 4 residual cells (ours, c1 and c2):** (a) **c1 decode = the MISSING bf16 DECODE CUDA GRAPH, not kernel speed** - summed GPU kernel time per decode step is **~31 ms against a measured 36.22 ms TPOT => ~86% GPU-busy, ~5 ms/step of host/launch gap**, which is essentially the whole 4.5 ms c1 deficit vs graphed vLLM; the kernels themselves are near the memory roof (fused decode MoE GEMM **37.9%** at **116.4 us** avg, down from 166.7 us for gate/up alone; dense qkv/o cuBLAS `gemvx` **24.5%** at **190-211 GB/s = 70-77% of GB10's ~273 GB/s**). This also drives the c1 output-throughput cell. (b) **c2 TTFT is NO LONGER the MoE GEMM** - the prefill tile fell from **56.0% to 16.6%** of GPU time at c1 (21.1% at c2) and costs 48 x (2 x 2.487 + 2.812) = **374 ms of a 533.6 ms TTFT at 19.9 TFLOP/s**, so the ~100 ms deficit vs vLLM's 435.5 ms sits in the NON-MoE prefill path (attention + dense projections + per-step glue). Memory axis remains CLOSED. Gates: `-Werror` **0-warn** clean full rebuild; **6/6** + `test_ops_moe_grouped_bf16` **7/7** + NVFP4 `test_ops_moe_grouped` **9/9** + `test_qwen3_moe_forward` **3/3** on the FINAL binary; **27B 235/235 + 35B 315/315 UNCHANGED**; memcheck **0 memory errors** on the MoE GEMM + engine path (`test_llm_engine` 5/5, `test_runner` 14/14; the only reported allocations are the by-design graph-safe `EnsureMoeScratch`/`EnsureMoePartials` retained blocks) **W7 (2026-07-21) - the bf16 DECODE CUDA GRAPH, W6's largest named lever.** NEW `Qwen3MoeDecodeGraph` (`include/vllm/model_executor/models/qwen3_moe.h`, `src/vllm/model_executor/models/qwen3_moe.cpp`), dispatched from `qwen3_moe_registry.cpp` on `pure_decode && is_cuda()` - the third sibling of the SAME driver already gated in-tree for the 35B (`Qwen3_5DecodeGraph`) and 27B (`Qwen3_5DenseDecodeGraph`), whose graph was fp4-gated so a bf16 model never got one. Ported from `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner` + `vllm/compilation/cuda_graph.py::CUDAGraphWrapper.__call__` @ `e24d1b24`. Coder-specific: attention-only padded-input builder (no GDN state to pad or gate, so EVERY captured size is usable) and no defensive copy of the persistent hidden. `ForwardBody` split into `EmbedInto` (kept OUTSIDE the graph - `vt::Embedding` allocates a device flag and syncs the stream) + the capturable `ForwardLayers`; the split alone is behavior-preserving (`VT_QWEN3MOE_CUDAGRAPH=0` -> 6/6). **One real latent bug found and fixed:** `BuildStepInputs` uploaded the RoPE identity row-index from a STACK-LOCAL vector, which under capture bakes a dead stack address that every replay re-reads - it produced a wrong token (`prompt[0] tok=2`, 3555 vs 576) and is now served from a process-persistent per-T table (contents byte-identical; the host-side analog of the `EnsureMoeScratch` retire-don't-free contract). **MEASURED (nsys `--cuda-graph-trace=node`, c1):** GPU-busy **~86% -> ~92%**, host tax **~5 -> ~2.7 ms/step**, median TPOT **36.22 -> 34.13 ms**; 60 `cudaGraphLaunch` (one per decode step) vs 3,569 total `cudaLaunchKernel` in the window. **CORRECTNESS: ZERO TOKEN MOVEMENT** - 6/6 (138 assertions), STRICT 5/6, max gap 0.0000 nats, goldens md5-verified unchanged. **BINDING GRID** vs GRAPHED vLLM 0.25.0, **FRESH server per concurrency with 0.0% prefix-cache hit VERIFIED in all four serve logs**, ours 2 reps with the cold leg discarded (>1 = we win): median TTFT c1 315.3/320.4 **1.02x**, c2 534.1/485.7 **0.91x**, c4 753.4/792.3 **1.05x**, c8 766.3/900.3 **1.17x**; median TPOT c1 34.13/31.75 **0.93x**, c2 39.68/44.56 **1.12x**, c4 57.97/73.80 **1.27x**, c8 78.22/84.59 **1.08x**; median ITL c1 34.02/31.76 **0.93x**, c2 40.10/43.96 **1.10x**, c4 55.54/71.21 **1.28x**, c8 68.05/77.50 **1.14x**; output throughput c1 24.60/29.46 **0.84x**, c2 39.85/41.64 **0.96x**, c4 56.34/50.63 **1.11x**, c8 90.10/87.27 **1.03x**. **11 of 16 cells at/above vLLM; c4 AND c8 now pass EVERY axis** (c8 output throughput crossed 0.99 -> 1.03x). Ours vs our own W6: TPOT **36.22->34.13 / 44.12->39.68 / 60.74->57.97 / 81.41->78.22 ms**, output throughput **22.62->24.60 / 37.24->39.85 / 55.03->56.34 / 86.02->90.10 tok/s**, TTFT unchanged within noise - a decode-only lever, and it lifted latency AND throughput together (no TPOT-for-throughput trade). **Denominator reproducibility:** 3 of 4 vLLM cells land within 0.5% of W6 (c1 TPOT IDENTICAL at 31.75 ms); the exception is **c2 TTFT, which swung 435.5 -> 485.7 ms (+11.5%)**, so that cell is honestly a band (0.82-0.91x), not a point. **RESIDUAL (5 cells, all c1/c2), attributed:** c1 TPOT/ITL/throughput = the remaining **~2.7 ms/step HOST tax** - vLLM's whole 31.75 ms decode step costs about what our KERNELS ALONE cost (~31.5 ms), so c1 needs no kernel work. The obvious CUDA-API explanation was profiled and **REFUTED**: `cuda_api_sum` puts `cudaStreamSynchronize` at a **median of 4.5 us** and the 19,184 `cudaMalloc` calls are one-time `ResidentWeight` uploads, with `cudaGraphLaunch` at 438 us/step - under 1 ms of the 2.7 ms is CUDA API, so the balance is engine-side per-step host bookkeeping needing a **CPU-side** profile. c2 TTFT/throughput are prefill-bound, untouched by a decode graph (our c2 TTFT moved 533.6 -> 534.1 ms) and remain the non-MoE prefill glue. GATES: `-Werror` **0-warn** clean full rebuild; 6/6 graph-ON + 6/6 graph-OFF + `test_ops_moe_grouped_bf16` 7/7 + NVFP4 9/9 + forward 3/3 + Qwen3-dense green; **27B 235/235 + 35B 315/315 UNCHANGED**; memcheck **0 errors** on the runner decode path, the engine path, AND the full Coder decode-graph gate. | **Named levers to close the row to `DONE`:** (i) a **bf16 decode CUDA graph** - now the single largest item (the 35B's graph is fp4-gated, `qwen3_5_moe.cpp:76-82`); it closes the c1 TPOT/ITL/output-throughput cells, which are ~86%-GPU-busy host tax rather than kernel time. (ii) A **non-MoE prefill-glue profile at c2** (attention + dense projections + per-step glue now dominate our TTFT there), mirroring the Qwen3-dense TTFT campaign. (iii) Optionally the **w13 gate+up fusion** in its cheap DUAL-OPERAND form (one launch, weight-pointer array and output pointer selected by N-tile half) - W6 skipped it because its two benefits (N-grid doubling, launch halving) are subsumed by split-K and the decode graph, and a true concatenated `[K,2I]` operand would need the loader to materialise fused expert buffers against a 57 GiB / 119 GiB unified-pool budget. Repro: `examples/vllm-bench --model <coder-snapshot> --input-len 1024 --output-len 128 --concurrency {1,2,4,8} --num-prompts {16,24,48,96}` vs a **FRESH** `vllm serve <snapshot> --moe-backend triton --gpu-memory-utilization 0.58 --max-model-len 2048 --max-num-seqs 32` **per concurrency** + `vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 --ignore-eos --max-concurrency C` (a shared server across concurrencies replays prompts into the prefix cache and voids the denominator). |
| OPT-125m (`OPTForCausalLM`) BF16 dense - the CROSS-FAMILY canary | **SPEED PENDING - NO NUMBER MEASURED OR CLAIMED** (2026-07-21, `CLAIM-MODEL-OPT-125M`; [spike §7](../.agents/specs/sweep-opt-125m.md)) | Breadth-sweep model #4 (rank 4, the cross-family additivity canary). **CORRECTNESS IS COMPLETE AND IS THE ONLY THING CLAIMED HERE:** the SACRED gate `test_opt_paged_engine` passes **STRICT token-exact 6/6 prompts / 96/96 tokens** against the pinned vLLM 0.25.0 oracle, with the strict bar SELECTED BY MEASUREMENT per [[near-tie-distributional-gate]] - `scripts/opt-oracle-capture.py --runs 5` found vLLM's own greedy DETERMINISTIC on all 6 prompts (**0 multi-valued (prompt,pos) cells**), so unlike the near-tie-gated Qwen rows there is no tolerance band at all. Our prompt tokenization also matches vLLM's `prompt_token_ids` exactly (the cross-check that caught the missing post-processor BOS, which had silently scored 0/6 with fluent output). Both arms run **bf16** via vLLM's first-class `--dtype bfloat16`, because the checkpoint's native fp16 has no compute path on our side (`kF16` is unimplemented outside `cuda_gdn.cu`) - recorded as spec decision D1, not as a benchmark shortcut. **NO THROUGHPUT/LATENCY/MEMORY NUMBER HAS BEEN TAKEN.** Publishing one now would be premature rather than merely unflattering: the model is structurally un-optimized by construction - head_dim 64 falls to the GENERIC `LaunchPaged` path because the FA2 fast kernels are gated to d128/d256, there is no decode CUDA graph for OPT (the Qwen3-Coder W7 pattern is the template), and the three new `vt::{LayerNorm,Relu,Add}` kernels are plain correctness-grade grid-stride launches with no fusion into the surrounding GEMMs. Supporting correctness evidence on the final dgx binary: `test_opt_load` 958 assertions (all 197 tensors mapped, no leftover), `test_ops_layernorm` 5 cases / 9230 assertions vs the torch `nn.LayerNorm` reference (CPU + CUDA), `test_pretokenizer` 18 cases / 97926 assertions with the new GPT-2 split goldens generated by HF tokenizers itself. Regressions UNCHANGED on the final dgx binary: 27B `test_qwen27_paged_engine` **235/235**, 35B `test_qwen36_paged_engine` **315/315**, Qwen3-Coder `test_qwen3coder_paged_engine` **6/6** (138 assertions), Qwen3-dense `test_qwen3_paged_engine` **664 assertions**; clean full CUDA rebuild `-Werror` **0 warnings / 0 errors**; `compute-sanitizer memcheck` on the OPT engine path gives **0 invalid reads / 0 invalid writes / 0 invalid `__global__` accesses** with the gate still 6/6 under the sanitizer (its 66 `--leak-check=full` reports are the shared `DevicePool`'s by-design retained scratch - `Put` never frees when uncapped - and are FEWER than the 125 the pre-existing Qwen3-0.6B path reports on the same binary). Goldens md5-verified identical before AND after the series; tree transferred by `git archive`, never rsync. | **NEXT (to make a speed number meaningful, in order):** (i) extend the FA2 varlen prefill/decode gates to **head_dim 64** - today OPT gets the generic CUDA-core paged path while vLLM runs FlashAttention, so any grid measured before this is a kernel-coverage artifact, not an engine comparison; (ii) an **OPT decode CUDA graph**, the third application of the driver already shared by the 27B/35B/Coder, which on Coder was worth ~2.3 ms/step of host tax; (iii) fuse the new LayerNorm/bias-Add/ReLU glue into the neighbouring projections (the pre-LN residual chain is 4 tiny launches per layer). Only then is a binding grid worth running. Repro when it is: `examples/vllm-bench --model ~/models/opt-125m-bf16-st --input-len 1024 --output-len 128 --concurrency {1,2,4,8}` against a **FRESH** `vllm serve ~/models/opt-125m-bf16-st --dtype bfloat16` per concurrency + `vllm bench serve --dataset-name random --random-input-len 1024 --random-output-len 128 --random-range-ratio 0 --ignore-eos --max-concurrency C`, idle box under one `flock /tmp/gpu`, prefix-cache hit rate verified 0.0% in every serve log (a shared server across concurrencies replays prompts into the prefix cache and voids the denominator). |
| Llama-3.2-1B (`LlamaForCausalLM`) BF16 dense - the CROSS-FAMILY DENSE additivity proof | **SPEED PENDING - NO NUMBER MEASURED OR CLAIMED** (2026-07-23, `CLAIM-MODEL-LLAMA-3.2`; [spike](../.agents/specs/sweep-llama-3.2.md)) | Breadth-sweep Tier-2 rank 5 - the roadmap's explicit "Llama-first" increment and the first mainstream non-Qwen/non-OPT dense arch. **CORRECTNESS IS COMPLETE AND IS THE ONLY THING CLAIMED HERE:** the SACRED gate `test_llama_paged_engine` passes **16/16 prompts (STRICT token-exact 12/16 + near-tie band 4/16, max teacher-forced gap 0.0000 nats, 0 forward-divergent)** against the pinned vLLM 0.25.0 oracle, with the bar SELECTED BY MEASUREMENT per [[near-tie-distributional-gate]] - `qwen3-oracle-capture.py --per-prompt --runs 6` found vLLM's own greedy DETERMINISTIC on all 16 prompts (**0 multi-valued cells**) ⇒ STRICT token-exact; at all 13 divergent positions vLLM's OWN teacher-forced argmax given our prefix IS our token (gap 0.0000), i.e. our forward reproduces vLLM's logits bit-identically. Llama = the Qwen3-dense forward with exactly two ADDITIVE deltas - NO qk-norm (shared AttnBlock skips it, byte-preserving) + llama3 RoPE scaling (4 default-0 `RopeArgs` fields + a `Llama3ScaleFreq` helper in the RopeNeox/RopeCosSinCache kernels, no-op elsewhere; formula verified 2e-7 rel vs vLLM). **ONE correctness-fatal tokenizer bug found+fixed:** Llama's `Sequence` post_processor wrapping a `TemplateProcessing` was not descended into ⇒ BOS 128000 never prepended, silently 1/16 with fluent output; a CUDA prefill-argmax diagnostic proved the FORWARD correct (4/4 first-token argmax given vLLM's exact tokens), isolating the fault to the tokenizer; the fix is byte-preserving for every existing model (Qwen/OPT/DeepSeek all ByteLevel or top-level-TemplateProcessing). **NO THROUGHPUT/LATENCY/MEMORY NUMBER HAS BEEN TAKEN** - head_dim 64 falls to the GENERIC `LaunchPaged` path (FA2 fast kernels are d128/d256-only) and there is no Llama decode CUDA graph, so any grid now would measure a kernel-coverage artifact. Supporting correctness: `test_llama_load` 792 assertions (all tensors mapped, q_norm/k_norm empty, tied lm_head empty, no leftover, llama3 config parsed) + a tokenizer diagnostic (5/5 prompts incl. BOS match vLLM); `test_llama_forward` 510 assertions (qk-norm-absent finite+deterministic, llama3-scaling applied, fusion ADOPT byte-exact, CUDA prefill 4/4). Checkpoint `unsloth/Llama-3.2-1B` (ungated mirror; 401 on gated meta-llama weights). Regressions UNCHANGED, each STANDALONE: 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6** (138 assertions), Qwen3-dense **16/16** (184 assertions), OPT **6/6** (63 assertions), DeepSeek-V2 **8/8** (223 assertions); clean full CUDA rebuild `-Werror` **0 warnings / 0 errors**; `compute-sanitizer memcheck` on the Llama engine path **0 invalid reads / 0 invalid writes / 0 invalid `__global__` accesses**; device-reference ratchet **DSR 67 unchanged**. | **NEXT (to make a speed number meaningful, in order):** (i) extend the FA2 varlen prefill/decode gates to **head_dim 64** (or bring up Llama-3.2-3B, head_dim 128, which reuses the already-validated d128 FA2 path directly) - today Llama gets the generic CUDA-core paged path while vLLM runs FlashAttention; (ii) a Llama **decode CUDA graph** (the fourth application of the 27B/35B/Coder driver). Only then is a binding grid worth running. Repro when it is: `examples/vllm-bench --model <llama-3.2 snapshot> --input-len 1024 --output-len 128 --concurrency {1,2,4,8}` against a FRESH `vllm serve <snapshot> --gpu-memory-utilization 0.40` per concurrency + `vllm bench serve --random-input-len 1024 --random-output-len 128 --ignore-eos --max-concurrency C`, idle box under one `flock $HOME/gpu.lock`, prefix-cache hit rate verified 0.0%. |
| GLM family + DSA + latest DeepSeek (V3.2, V4) | **NOT APPLICABLE - SPIKE ONLY; NO NUMBER MEASURED OR CLAIMED** (2026-07-21, `CLAIM-GLM-DSA-LATEST-DEEPSEEK`; [spike](../.agents/specs/glm-dsa-latest-deepseek.md)) | Design-only change: no code, no kernels, no build, no GPU work, nothing downloaded. Seven model-matrix rows `INVENTORIED` -> `SPIKE`. The benchmark-relevant output is a **hardware-feasibility verdict per variant**, measured rather than assumed (HF metadata fetched 2026-07-21; dgx checked directly): `Glm4MoeLiteForCausalLM` / `zai-org/GLM-4.7-Flash` 31.2B **58.2 GiB bf16 - FITS** GB10's ~119 GiB and is a second MLA gate vehicle that closes both coverage gaps the MLA campaign named as unit-gated-only; `Glm4ForCausalLM` / `GLM-4-9B-0414` and `GlmForCausalLM` / `glm-4-9b-chat-hf` **17.5 GiB - FIT**; `ChatGLMModel` / `chatglm3-6b` 23.3 GiB - FITS; `Glm4MoeForCausalLM` / `GLM-4.5-Air` **205.8 GiB - HW-BLOCKED** (the 104.8 GiB FP8 variant depends on an fp8-loading row we do not own); `GlmMoeDsaForCausalLM` / `GLM-5` **1404.2 GiB - HW-BLOCKED** and additionally DEP-BLOCKED because DSA has no working sm_121 backend; `DeepseekV4ForCausalLM` / `DeepSeek-V4-Flash` **148.7 GiB - HW-BLOCKED** (fits the 184 GiB disk, not the 119 GiB memory); DeepSeek-V3.2-Exp 642.1 GiB - HW-BLOCKED. Also corrected: **dgx free disk is 184 GiB, not 238 GiB** (95% full), and **no GLM checkpoint is present** - verified directly per the OPT lesson that the sweep plan's "present" column was wrong | **No benchmark is possible or owed until an implementation exists.** The first speed number this track can produce is the `Glm4MoeLiteForCausalLM` every-axis grid on GLM-4.7-Flash, which is gated behind (a) `CLAIM-MLA-DEEPSEEK` reaching its W6 MLA attention block, (b) the shared `vt::MoeRouterTopK` extension, and (c) a staged 58.2 GiB download that must free disk first - 93.2 GiB of gate checkpoints against 184 GiB free at 95% utilization, where an ENOSPC mid-download presents as bogus test failures. Correctness precedes speed unconditionally: no row reaches `DONE` on a token-exact gate alone. Repro: none - nothing was executed. Re-derive the feasibility table with `python3 /tmp/claude-1000/-home-mudler--git-vllm-cpp/4053e545-397a-4134-ad21-1c1c397025fa/scratchpad/hfsize.py` (HF metadata only, downloads nothing) and `ssh dgx.casa 'df -h /; ls ~/.cache/huggingface/hub/'` |
| Metal backend - FIRST MODEL RUNS (`BACKEND-METAL-MLX`, work row `M3a`) | **NOT APPLICABLE - NO SPEED NUMBER MEASURED, CLAIMED OR OWED, AND THAT IS NOW A DELIBERATE REFUSAL RATHER THAN AN ABSENCE OF SOMETHING TO MEASURE** (2026-07-22, `CLAIM-BACKEND-METAL-M3A-1`; [study §12](../.agents/specs/metal-mlx-reuse-study.md)). **A MODEL NOW RUNS ON APPLE GPU:** OPT-125m generates end to end and is STRICT token-exact 6/6 prompts / 96/96 tokens vs the committed vLLM 0.25.0 goldens - so unlike every previous row here, there IS something that could be timed. **It was not timed, and no number may be quoted.** The M4 could not be quieted: the root `com.localai.worker` LaunchDaemon needs interactive sudo (`sudo -n true` -> "a password is required"), and the desktop aerial video wallpaper (`WallpaperAerialsExtension`, 8.2% CPU, plus `VTDecoderXPCService`) is the LARGER contender and also touches the GPU. Under the standing contended-run rule any figure from that box is VOID, so none was produced - correctness and speed are separate bars and only correctness is met, which is exactly why `BACKEND-METAL-MLX` stays `ACTIVE` and not `DONE`. **To make a binding run possible the user must run BOTH, on `ssh 192.168.68.103`:** `sudo launchctl bootout system/com.localai.worker` (restore afterwards with `sudo launchctl bootstrap system /Library/LaunchDaemons/com.localai.worker.plist`), AND disable the aerial wallpaper (System Settings -> Wallpaper, or log the console user out). Also note dispatch is still ONE COMMAND BUFFER PER OP with commit+wait and the GEMM is a plain threadgroup tile loop with no simdgroup-matrix use, so a number taken today would measure a known-unoptimized placeholder (work row `M3c`). Metal registered ops are now **15 of 75** - the five added here being `kEmbedding`, `kQkvSplit`, `kReshapeAndCache`, `kPagedAttention`, `kGreedyArgmax`. Earlier state, superseded: W0 landed a Metal `vt::Backend`/`Platform` SKELETON: 8 ops (`kAdd`, `kRelu`, `kSiluAndMul`, `kCastBf16`, `kCastF32`, `kLayerNorm`, `kRmsNorm`) plus ONE `kFusedChain` interpreter. at that point GEMM, attention, KV cache, quant and sampling were all unregistered and no model ran. Vulkan followed the same day with its own skeleton (next row); Intel XPU remains SPIKE-ONLY and unchanged | The deliverable is CORRECTNESS evidence, not speed. Oracle = **our own CPU backend on the same M4** (vLLM cannot run there, so no vLLM comparison is made or implied). Bit-exactness across CPU/GPU is explicitly NOT promised for reductions (our CPU determinism comes from a fixed sequential order); the bar is the already-ported **NMSE <= 5e-4**, with bit-exactness retained for pure copy/layout paths. **M3a per-op evidence (2026-07-22), oracle = our own CPU backend through the SAME public `vt::` entry points on the SAME M4:** `kEmbedding`, `kQkvSplit`, `kReshapeAndCache` and `kGreedyArgmax` are **BIT-EXACT** - and that is principled, not lucky: they perform no floating-point reduction, so a GPU implementation has no reordering freedom, and the argmax reduces over an order-INDEPENDENT (value, lowest-index) max reproducing `torch.argmax`'s first-maximum rule (pinned by a deliberate two-position tie and an all-equal row). `kPagedAttention` is **NMSE 4.99e-13** against the 5e-4 bar, and **bit-exactness is explicitly NOT claimed for it**: it is the algebraically identical ONLINE (flash) softmax where the CPU reference is a materialized 3-pass, i.e. a different reduction order BY CONSTRUCTION. **Every arm also PROVES the Metal path executed** - `runner().device().type == kMETAL`, plus `selections > 0` AND `declines == 0` for all nine ops OPT dispatches (`kPagedAttention` selections 1152), plus NaN-poisoned output buffers so an un-executed kernel cannot pass numerically; `last_selected` alone would not be proof, since a provider can decline inside its kernel and forward down. Earlier W0 op evidence over widths {128, 100, 17}: `kAdd`/`kAdd`-broadcast/`kRelu` **0 (exact)**; `kSiluAndMul` 2.50e-15; `kRmsNorm` 5.30e-15..1.90e-14; `kRmsNorm` residual stream **0 (exact)**; `kLayerNorm` 3.86e-15..9.95e-15; `kFusedChain` (both realization tiers) 9.05e-15 - **worst case 1.9e-14, eleven orders of magnitude inside the bar**. BIT-EXACT (`memcmp`) for `Backend::Copy`/`Memset` and the bf16<->f32 codec incl. NaN, +-inf, +-0 and 16 exact rounding ties. M4 (M3a): clean `-Werror` **0 warnings** on a CLEAN FULL rebuild (AppleClang 21, CLT-only - MSL compiled at RUNTIME, there is no offline `metal` compiler on this box); `test_metal_backend` **12 cases / 18,535 assertions**; full macOS ctest **154/156**, both misses the documented pre-existing platform gaps, with `test_capi`'s standalone macOS failure PROVEN pre-existing by building unmodified `72f5db2` on the same box and reproducing the identical `CHECK(1 == 2)`. Earlier W0: `test_backend` **7/7**, `test_metal_backend` **6/6 (59/59)**, `test_backend_cross_device` **5/5 (73/73)**. The unrelated-but-required regression proof on dgx (shared CMake + a shared destructor were touched): clean CUDA `-Werror` 0 warn and all six model gates UNCHANGED, each STANDALONE - 27B **235/235**, 35B **315/315**, Qwen3-Coder **6/6**, Qwen3-32B-NVFP4A16 dense **6/6**, OPT **6/6 (96/96 tokens)**, DeepSeek-V2 **8/8**, plus the FULL dgx ctest suite **190/190** | **`PENDING`, AND NOW BLOCKED ONLY ON A QUIET BOX - the first Apple speed number is owed at work row `M3b`.** `M3a` removed the technical blocker (a model runs, and runs token-exact); what remains is (a) the two host-quieting steps above, which need interactive sudo and a desktop change, and (b) `M3b`, because the competitor arm must be Qwen3-dense - MLX-LM runs that model and does not run OPT, so an OPT timing would have no comparable MLX arm and would not constitute a floor. **The competitor floor is now NAMED: MLX** (`BACKEND-GATE-METAL-MLXLM`, user directive 2026-07-22) - same model, same workload, same box, ours must match or beat it on EVERY axis (total + output throughput, req/s, TTFT, TPOT/ITL, peak memory), correctness a precondition never traded off. llama.cpp Metal (`BACKEND-GATE-METAL-LLAMACPP`) remains an additional floor. **This fixes the first model: Qwen3-dense** - MLX-LM runs it, and it satisfies the spike's "OPT or Qwen3-dense first, never Qwen3.5-Next" constraint. **Precondition (1) is now DONE and precondition (2) is NOT:** MLX **is installed** on the M4 as of 2026-07-22 via the venv route (brew NOT used, so `python@3.14` never touched the build PATH), resolving **`mlx` 0.29.3 / `mlx-metal` 0.29.3 / `mlx-lm` 0.29.1**, and **an UNOPPOSED MLX BASELINE HAS BEEN MEASURED** - see the dedicated section "MLX competitor baseline" below. The `com.localai.worker` daemon **could NOT be stopped** (`sudo -n true` -> "a password is required"), so that baseline is recorded **`BLOCKED-ON-SUDO` / INDICATIVE, NOT BINDING**, and it is an MLX-only floor with **no "ours" column - none exists and none was manufactured, because no model runs on Metal.** **Apple gate-scale is permanently out of reach on this host** (16 GB / 11.84 GiB working set cannot hold the 27B/35B gate models), so no Apple result may ever be extrapolated to them. **The first-model cost is now QUANTIFIED** ([reuse study](../.agents/specs/metal-mlx-reuse-study.md)): OPT needs 9 ops (6 new) and its TUs are CUDA-free; Qwen3-dense needs 10 (7 new). Repro: nothing of OURS was executed as a benchmark
| Vulkan backend W0/V1 skeleton (`BACKEND-VULKAN`) | **NOT APPLICABLE - NO SPEED NUMBER MEASURED, CLAIMED OR OWED** (2026-07-22, `CLAIM-BACKEND-FANOUT-1`; [spike](../.agents/specs/backend-fanout-metal-vulkan-xpu.md)). V1 landed a Vulkan `vt::Backend`/`Platform` SKELETON with the SAME eight ops as the Metal skeleton (`kAdd`, `kRelu`, `kSiluAndMul`, `kCastBf16`, `kCastF32`, `kLayerNorm`, `kRmsNorm`) plus ONE `kFusedChain` interpreter. **GEMM, attention, KV cache, quant and sampling are UNREGISTERED, so NO MODEL RUNS on Vulkan** and there is nothing to benchmark. Dispatch is deliberately SYNCHRONOUS and mutex-serialized (record + submit + fence wait per op) - correct, not fast - so even an op-level microbenchmark would measure a placeholder, and none was taken. The backend is also OFF by default (`VLLM_CPP_VULKAN` AUTO -> OFF), so no shipped configuration is affected. `BACKEND-VULKAN` is `ACTIVE`, which here means a GATED SKELETON, NOT supported. Intel XPU remains SPIKE-ONLY and unchanged | The deliverable is CORRECTNESS evidence, not speed - and here the oracle is unusually strong: GB10 runs our CPU backend, our CUDA backend AND Vulkan **in one binary**, so `test_backend_cross_device` reports **144 assertions on GB10 versus 73 on a Vulkan-only box** because it compares the CPU reference against BOTH device backends. Bit-exactness across CPU and GPU is explicitly NOT promised for reductions; the bar is the already-ported **NMSE <= 5e-4**, with bit-exactness retained for pure copy/layout paths. MEASURED on GB10 over widths {128, 100, 17}, Vulkan quoted against CUDA on the same box: `kAdd`/`kAdd`-broadcast/`kRelu` **0 (exact)** on both; `kSiluAndMul` 3.23e-15 (CUDA 4.17e-16); `kRmsNorm` 2.92e-14/1.23e-14/3.77e-15 (CUDA 2.37e-14/1.61e-14/2.59e-15); `kRmsNorm`+residual output 5.20e-15/6.21e-15/7.55e-15 (CUDA 0/8.73e-15/1.34e-15); residual STREAM **0 (exact)** on both; `kLayerNorm` 8.31e-15/8.02e-15/1.75e-15 (CUDA 8.82e-15/7.48e-15/3.64e-15); `kFusedChain` both realization tiers 1.05e-14 (CUDA 1.08e-14). **Worst case Vulkan 2.92e-14 - ten orders of magnitude inside the bar, and within 1.3x of CUDA's own worst case.** BIT-EXACT (`memcmp`) for `Backend::Copy`/`Memset` and the bf16<->f32 codec incl. +-inf, +-0 and 16 exact rounding ties. Relaxed-precision knobs: `1.0/sqrt` replaces llama.cpp's `inversesqrt` (the Vulkan analogue of Metal's `MTLMathModeSafe` pin), no `RelaxedPrecision` is emitted, fp32 float-controls are PROBED and printed rather than assumed, and FMA contraction is deliberately left undecorated and RECORDED (it is why the same committed SPIR-V gives 2.37e-14 on llvmpipe and 2.92e-14 on GB10). GB10: clean CUDA `-Werror` 0 warn on BOTH the Vulkan-OFF and Vulkan-ON builds, `test_vulkan_backend` **8/8 (82/82)**, `test_backend_cross_device` **5/5 (144/144)**. Dev box on `llvmpipe` with NO GPU: clean `-Werror`, `test_vulkan_backend` **8/8 (82/82)**, `test_backend_cross_device` **5/5 (73/73)**, full ctest 155/156 (one known `-j` flake, passes on rerun) - **a GPU-free CI path is PROVEN**. Required regression proof on dgx, run STANDALONE on the production (Vulkan-OFF) build under one `flock`: 27B **235/235**, 35B **315/315**, Qwen3-Coder **138/138**, Qwen3-dense **664/664**, OPT **36/36**, DeepSeek-V2 **223/223**; goldens md5 identical before and after | **`PENDING` - the first Vulkan speed number is owed at work rows `V3`/`V4`**, when GEMM (scalar `mul_mm` first, then the `coopmat`/`coopmat2` tactics GB10 exposes) and paged attention make a model runnable. **The competitor floor is llama.cpp's own Vulkan backend** (`BACKEND-GATE-VULKAN-LLAMACPP`, still `INVENTORIED`) - the same role MLX now plays for Metal: same model, same workload, same box, match or beat on every axis. Unlike Apple, gate-scale IS reachable here - GB10's single 89.72 GiB unified heap holds the 27B/35B gate models - so a Vulkan number will eventually be directly comparable to the CUDA numbers above, on identical hardware. Repro: nothing was executed as a benchmark
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

### Active packed-decode implementation checkpoint

The semantic and operator gates are retained at clean `f18ca23` and
`9ad8fb7`; their complete hashes and attempt chronology live in the ledger and
state log. W1D3 is now **CLOSED on EQUIVALENCE** (`KERNEL-GDN-PACKED-DECODE`
`DONE`, owner `e47b4d6`; no `complete-pass` marker, no speed credit):

| Surface | Current evidence | Disposition |
|---|---|---|
| Exact vLLM boundary | Direct packed output/state BF16 differences **0/1** | PASS; upstream explicit reference has the same one-element state delta |
| Cache ABI | `MambaSpec` conv then temporal; gate allocation BF16 conv + FP32 SSM | PASS on registry/runner production-like tests |
| 27B default | **235/235 + 16/16**; prefill 0 packed calls, first decode exactly 48 | Immutable G2 PASS at `f344dec` |
| 27B rollback | `VT_GDN_PACKED_DECODE=0`, **235/235 + 16/16**, 0 packed calls | Immutable G2 PASS at `f344dec` |
| 35B native/batched | **315/315**, 0 packed calls | Immutable inertness PASS |
| 35B GGUF | Compact **14/14**, Balanced **14/14**, loader **98/98** | Immutable isolated-process PASS |
| Safety | CUDA GDN **43/43, 1,707/1,707**; packed/corner/FP16-SSM memcheck zero errors/leaks | Immutable PASS |
| W1D3 trace harness | Packed **915** nodes with 48 packed recurrence calls; rollback **963** with 48 decomposed + 48 post-conv calls; both retain 145 BF16 GEMMs, isolate exactly 48 mode-coupled BA projections, and require every remaining signature to match | Raw capture PASS at `7ff713e`: **12/12 + 12/12** exact ranges; both oracle traces structurally exact |
| Component runner | Production profile-control-off build; exact source/vLLM corpus manifests and partition hashes; full oracle/toolchain/artifact inventory; exact raw-sample throughput/TTFT/ITL recomputation plus bounded pinned-clock E2E/TPOT validation and duration-span consistency; frozen 64-plan fixture; fresh isolated server per leg; c2=6 requests and c16=96 requests per rep; a discarded cold-start warmup pair then 5 timed reps AB/BA/AB/BA/AB (22 legs); one lock across exact-snapshot correctness and all legs | The `d82d282` c16 HTTP-500 (`duplicate live GDN state index`) was fixed test-first (request-identity slot keying) and proven on DGX at `c172336`. **Eight full components have since sealed** a marker-last terminal status (3× `complete-void` on the TTFT-family arrival lottery, then 5× `complete-failed` as the harness calibrated to true-zero noise); the eighth (first 22-leg) seal is `complete-failed` with 38/40 + 8/8 memory PASS, stability clean, paired-consistency PASS at BOTH concurrencies - **W1D3 CLOSES on equivalence** |
| Performance | Marker-last finalization requires all **40 timing + 8 memory** median axes and the gated paired run axes, per-run stability of ≤4% for non-tail timing and all memory axes and ≤15% for the tail axes (p90/p99 of ttft/tpot/itl/e2el - revised test-first after two `c172336` tail voids), fixed 1-GiB memory-return tolerance, parsed throttle counters, successful pinned GPU-idle probes, exact server/client/preflight commands and lifecycle markers. The c2 TTFT-family (mean/median/p90/p99 of ttft) is instead compared on the arm-pooled 30-per-request distribution, stability-gated on a 50% pooled sanity bound, and excluded from the gated per-rep paired axes (bimodal arrival lottery; c16/non-TTFT keep 4%/15%; E2EL unchanged). **Acceptance uses a NOISE BAND (`contract.acceptance`): a comparison axis (median, gated paired, c2 mode means, memory) fails only when the packed deficit exceeds run noise - 0.5% for non-tail timing and PSS/RSS, 15% for tail axes, per-mode c2 TTFT, and the recalibrated GPU-memory bands; packed≥rollback always passes.** The gated per-rep **paired** axes use a **MAJORITY-CONSISTENCY** rule (`contract.paired_gate`): a paired axis fails only when ≥3 of its 5 rep-pairs breach the band in the same (packed-worse) direction; single-pair breaches are recorded as diagnostics. A stable regression is `complete-failed`; a sealable unstable/malformed run is `complete-void` | **Eight sealed → W1D3 CLOSES on EQUIVALENCE.** VOID×3 (all TTFT-family: a bimodal prefill arrival lottery, no scheduler divergence), then `complete-failed`×5 as the harness calibrated to true-zero noise (acceptance noise band, mode-conditional c2 TTFT + GPU-memory recalibration, majority-consistency pairing, 5-rep + cold-discard precision upgrade). The eighth (first 22-leg: cold-discard pair + 5 reps) seal `complete-failed` at `e47b4d6`: **38/40 axes, 8/8 memory, stability clean, `validation_error=None`, paired-consistency PASS at BOTH c2/c16**; c16 at equivalence (packed med 801.97 vs rollback 802.95, −0.12%, passes); the 2 fails are sign-flipping band-edge statistics of a true-zero effect (c2 `median_tpot_ms` 0.9899, c2 pooled `p99_ttft_ms` 0.8464). Combined with the 8-pair locked c16 A/B (−0.205% ± 0.30, <1σ) and the 24-window trace (packed GPU-cheaper), **no stable regression exists on any axis - EQUIVALENCE PROVEN. No `complete-pass` marker exists; NO SPEED CREDIT** |

Failure evidence is immutable. Order/run/execution/c16 raw/client/server SHA-256
values are `a2de5b07…6de0` / `297e3c62…a6fb` / `ff71c9f0…0684` /
`f8571d48…945c` / `8b9526f4…63a9` / `7b05e066…7dfe`. The C++ API wraps the
engine exception text into its JSON error body, but pinned vLLM's OpenAI
benchmark stores only `response.reason`; therefore the retained client evidence
says only `Internal Server Error`. Exact exception capture is the next
diagnostic, not a guessed runtime fix.

The current raw root is
`~/work/vllm.cpp-gdn-packed-trace/7ff713e377457130db4ed15929133d1b463aff96`.
Execution / configure / model-gate / run-log SHA-256 values are
`253ff089…a4bb` / `903d10a1…5c12` / `3ebdd9f9…bfca` /
`6c0c2cae…2235`; packed/rollback oracle trace hashes are
`4448c549…c293` / `9d918979…a420`. The first finalizer log
(`a71ac6e4…7e67`) records the expected fail-closed unknown-signature error.
Accepted summary / manifest / marker / pushed-finalizer-log hashes are
`bf5c04b7…702f` / `2e92b3a2…55c0` / `e1314019…8c1` /
`626c6844…8211`; artifact-set SHA is `ea286db4…c3dc`. Source, GPU and lock
returned clean/idle/free; the complete artifacts are retained.

Immutable evidence root:
`~/work/vllm.cpp-gdn-packed-decode/f344decf457a4d50c3bcae78a2903d7fe176a511/evidence-g2`.
Status is `complete-g2`; the complete one-lock order is frozen in
`run-plan.txt`, and its core entry points are:

```sh
flock /tmp/gpu build-cuda/tests/test_ops_gdn
flock /tmp/gpu build-cuda/tests/test_op_parity \
  -tc='qwen27 GDN packed decode boundary*'
flock /tmp/gpu build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu env VT_GDN_PACKED_DECODE=0 \
  build-cuda/tests/test_qwen27_paged_engine
flock /tmp/gpu build-cuda/tests/test_qwen36_paged_engine
```

G2 and W1D3 structural evidence are closed. The repaired source has been pushed
and launched exactly once at `d82d282`; that incomplete root must never be
reused or appended. The failure inspection and next diagnostic entry point are
in the newest [state record](../.agents/state.md) entry and the
[spike](../.agents/specs/gdn-packed-decode.md). The failed launch provenance is:

```sh
set -euo pipefail
SOURCE_REPO="$HOME/work/vllm.cpp"
git -C "$SOURCE_REPO" fetch origin main
SHA=d82d282f9efd1a5b97e7c6f1ac7a55b949849d09
ROOT="$HOME/work/vllm.cpp-gdn-packed-component/$SHA"
BINDING="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
test ! -e "$ROOT"
mkdir -p "$(dirname "$ROOT")"
git -C "$SOURCE_REPO" worktree add --detach "$ROOT/source" "$SHA"
mkdir -p "$ROOT/evidence/corpus"
cp -a "$BINDING/corpus/27" "$ROOT/evidence/corpus/27"
cmake -S "$ROOT/source" -B "$ROOT/build-production" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$HOME/venvs/vllm-oracle/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_FLASH_ATTN=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=OFF \
  -DVLLM_CPP_BENCH_PROFILE_CONTROL=OFF \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$ROOT/configure.log"
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --execute \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence/corpus/27" \
  --evidence "$ROOT/evidence" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

`$ROOT/source` is the clean detached worktree at `$SHA`; the recipe above is the
`--execute` reproduction pattern (the eighth run used the 22-leg 5-rep plan). The
request corpus is copied byte-for-byte from the binding `3f256ab` evidence (the
5-rep corpus is a byte-verified deterministic extension); the separate 64-plan
fixture is committed under `tests/fixtures/` and the runner requires it in
read-only mode. The packed component claims no throughput number, so the binding
27B grid stays `benchmark_binding=false` until the authorized fresh exact-grid
rerun.

**W1D3 is CLOSED on equivalence** (see the summary at the top). All eight
component roots are immutable and must never be reused
(`c172336`/`-r2`, `d19e091`, `2dbe892`, `da05444`, `495ba780`, run-6, and the
eighth/closing seal `e47b4d6`). The active tracks are now **qkvz**
(`KERNEL-GEMM-BF16` W2) and the authorized fresh binding/exact-grid rerun; no
further packed-decode component run is required (further single-run seals of the
true-zero effect are band-edge coin flips that would not change the recorded
conclusion). The `--diagnostic-c16` recipe below is retained as the focused
regression check: it reruns the packed c16 boundary three times under one GPU
lock, captures any `engine-fatal:` root cause on stderr (server log), and
replays the corpus row into a persisted HTTP body - writing only
`component-diagnostic.json` and a `diagnostic/` subtree, never a sealable
component. Provision `SNAPSHOT`/corpus/build exactly as above, then, with a new
root whose basename contains `diagnostic-c16`:

```sh
SHA=<pushed diagnostic SHA>
ROOT="$HOME/work/vllm.cpp-gdn-packed-diagnostic-c16/$SHA"
# ... provision $ROOT/source, $ROOT/evidence-diagnostic-c16/corpus/27,
#     $ROOT/build-production, $ROOT/configure.log exactly as the --execute recipe
"$ROOT/source/scripts/dgx-gdn-packed-component.sh" --diagnostic-c16 \
  --snapshot "$SNAPSHOT" \
  --source-corpus "$ROOT/evidence-diagnostic-c16/corpus/27" \
  --evidence "$ROOT/evidence-diagnostic-c16" \
  --build-dir "$ROOT/build-production" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

The mode asserts the evidence basename contains `diagnostic-c16` and holds no
`component-*.json`; it runs no model gates, no 2/16 sweep, and never calls
`finalize`. `benchmark_binding` stays `false` throughout.

This reproduction has been executed once at `4a450f9` (root
`~/work/vllm.cpp-gdn-packed-diagnostic-c16/4a450f9…`, build **154/154**,
marker-last `diagnostic_complete`, GPU/lock returned idle/free). All three
fresh-server reps failed identically with
`engine-fatal: … duplicate live GDN state index at qwen3_5.cpp:73`.
Core SHA-256: `component-diagnostic.json` `42de1323…13ea`; r1/r2/r3 server
logs `f26f0030…8bc0` / `8ecba873…02a3` / `e68411cf…6f3f`; r1 error body
`c5aa0933…7fc3`. The root is diagnostic evidence only and is preserved
unchanged; the repair checkpoint and a fresh SHA/root full component rerun
are next.

Closed controls do not remain active leaves: async scheduling measured neutral
for speed, and the prior fused-producer candidate remains default-off after its
strict component failure. Their exact results and reproduction history remain
in the append-only record rather than this live scoreboard.

The completed core correctness/safety root is
`~/work/vllm.cpp-gdn-ba/immutable-581d335fec2e5a96d9ccbb38c1ec001c39ac1789`.
Status / artifact-list SHA-256 values are `3895e658…4cf6` / `ed2bf8d8…895b`.
Focused CTest, merged/split 27B, 35B, and memcheck log hashes are
`4cf699ad…759b`, `c2a6f93f…cf96` (both arms), `b926716e…9875`, and
`a3d61cb9…fb87`; fixture `e81e9181…7edd` loaded 64 plans and the forbidden
native cache stayed absent. The rejected BF16-output preflight log remains
`09078b76…b050`. This closes immutable core correctness/safety only and earns no
speed ratio.

The completed structural root is
`~/work/vllm.cpp-gdn-ba-trace/0091cd192d9a6baa2197a4f3bdb0561bd859baf5`.
All 24 local ranges pass their exact contracts. The merged/split oracle traces
have SHA-256 `b8d26d4c…fc59` / `cef841ce…ede5`; they contain 1,522 / 1,521
internally invariant steady B=2 windows at 1,160 kernels and ordered-name SHA
`858915dd…fad0`. Their full launch signatures are `17e1037e…14ed` /
`f7a3ca1f…cadf`. Pushed finalizer commit `8a1f923` wrote the marker last with
status `complete-structural`. Summary / manifest / marker / artifact-set /
finalizer SHA-256 values are `03601168…54d5` / `b203f0d2…5412` /
`72328c48…63e` / `b93fd633…70a2` / `57395e99…b146`. The summary proves merged
963/145 versus split 1,011/193, exact 48/48 deltas, unchanged non-BF16 family
counts and `benchmark_binding=false`; it grants no speed ratio.

The current W1C hardware root is
`~/work/vllm.cpp-gdn-ba-rounding/f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3/evidence/w1c-correctness-inertness`.
The source is clean at exact `f925294`, and the binary SHA-256 values are frozen
in `provenance.txt`. Projection / loader / native-35B / real-GGUF / default-27B /
BF16-27B log SHA-256 values are
`a791c567…37d1` / `d455b8fc…05f6` / `72caeca9…06c` /
`87833f22…af8d` / `da5dd836…091e` / `148d743f…86a`.
The final BF16 assertion failure intentionally prevents `status.txt`,
`sha256sum.txt`, and the terminal event from being written. GPU and lock were
verified idle afterward. This is a valid **FAILED** correctness checkpoint,
not a partial performance number.

## Evidence selecting merged GDN projections

The immutable completed root is
`~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`.
Its status is `complete-diagnostic`.

| Artifact | SHA-256 |
|---|---|
| c2 summary | `0ef6a1240d33c16410cd4e43b30ca8667a6d92e6eee8506d7bd03388fe010273` |
| c2 manifest | `2556cfd032fae2201d9f8deb818343731b7dc99d9f8e6329da9b793262712f21` |
| status | `9e0143fa1b9c74e218e486fedd0606850708619a0e859dafe94957e24a507b57` |
| artifact set | `cc248ad2b5bf08f85b0d6b178de70682a104917e16c59c9adf34d661217f823a` |
| fresh oracle trace | `2b3bf41269fd19ef65c5c3e06f067af73d7d997de3b6be17a2af785b6a86785c` |

All **12/12** local B=2 graph ranges are invariant at 1,011 kernels + 7
memcpy + 1 memset. The oracle contains **1,522** invariant steady B=2 windows
at 1,160 kernels plus two bounded B=1 drains. Both engines execute the same
**128 Stream-K + 80 static-persistent** FP4 tactic split.

| BF16 projection structure per B=2 window | vllm.cpp | vLLM v0.25.0 |
|---|---:|---:|
| qkv / packed qkvz | 48 | 48 |
| z | 48 | included in qkvz |
| b+a / packed ba | 96 | 48 |
| lm_head | 1 | 1 |
| Total | **193** | **97** |

The source arithmetic independently gives `(4-2) × 48 = 96`; this is not a
profiler-name classification artifact. Diagnostic family medians are
**51.662672 vs 48.798042 ms** (+2.864630 ms), with a shape-level decomposition
ranking BA at about 1.882 ms and qkvz at about 0.476 ms. These durations cross
Nsight/Torch-profiler domains and are not a speed ratio. The accepted spike
therefore gates BA and qkvz separately and forbids duplicate weight owners or
split-copy kernels.

## Verify or reproduce the current checkpoint

Verify the durable diagnostic without GPU work:

```sh
RAW_SHA=179a0fc2afc1c33b63d14de8e50d3fde976c7356
ROOT="$HOME/work/vllm.cpp-executed-path-c2/$RAW_SHA/evidence/$RAW_SHA/trace/27"
sha256sum "$ROOT"/{c2-summary.json,c2-manifest.json,status-c2.json}
# Expected prefixes: 0ef6a124…0273 / 2556cfd0…2f21 / 9e0143fa…7b57
```

Verify the pushed-SHA core correctness/safety checkpoint without rerunning the GPU:

```sh
SHA=581d335fec2e5a96d9ccbb38c1ec001c39ac1789
ROOT="$HOME/work/vllm.cpp-gdn-ba/immutable-$SHA"
test "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/src" status --porcelain)"
sha256sum "$ROOT/evidence"/{status.txt,sha256sums.txt}
# Expected: 3895e658…4cf6 / ed2bf8d8…895b
```

Verify the durable structural checkpoint without GPU work:

```sh
RAW_SHA=0091cd192d9a6baa2197a4f3bdb0561bd859baf5
ROOT="$HOME/work/vllm.cpp-gdn-ba-trace/$RAW_SHA"
TRACE="$ROOT/evidence/$RAW_SHA/trace/27"
sha256sum "$TRACE"/{gdn-ba-summary.json,gdn-ba-manifest.json,status-gdn-ba.json}
# Expected: 03601168…54d5 / b203f0d2…5412 / 72328c48…63e
```

Verify the current fail-closed W1C checkpoint without rerunning the GPU:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
E="$ROOT/evidence/w1c-correctness-inertness"
test "$(git -C "$ROOT/source" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/source" status --porcelain)"
test ! -e "$E/status.txt"
sha256sum "$E"/{01-projection.log,02-qwen36-weights.log,03-qwen36-native.log,04-qwen36-gguf.log,05-qwen27-default.log,06-qwen27-bf16.log}
```

The exact single-lock command below reproduces the accepted subgates and final
233/235 failure. Any partial or unlocked execution is void:

```sh
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA"
SOURCE="$ROOT/source"
BUILD="$ROOT/build-cuda"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$SOURCE" status --porcelain)"
flock /tmp/gpu bash -lc "
  set -euo pipefail
  '$BUILD/tests/test_op_parity' \
    -tc='qwen27 GDN BA BF16 projection matches vLLM 0.25 oracle*'
  '$BUILD/tests/test_qwen36_weights'
  '$BUILD/tests/test_qwen36_paged_engine'
  '$BUILD/tests/test_qwen36_gguf_engine'
  '$BUILD/tests/test_qwen27_paged_engine'
  VT_GDN_BA_OUT_BF16=1 '$BUILD/tests/test_qwen27_paged_engine'
"
```

The prior end-to-end failure remains reproducible and is the model-level RED
precondition for the packed production dispatch:

```sh
SHA=f9252943d1e96dbfa43e3b8f2d06dec1aa5f20d3
BUILD="$HOME/work/vllm.cpp-gdn-ba-rounding/$SHA/build-cuda"
flock /tmp/gpu env VT_GDN_BA_OUT_BF16=1 \
  "$BUILD/tests/test_qwen27_paged_engine"
# Current disposition: 233/235; got == greedy_ids_emulation.npy
```

The accepted W1D3 finalization is reproducible without recapturing the GPU
series. Its marker records both the finalizer and imported launch-validator
hashes:

```sh
set -o pipefail
RAW_SHA=7ff713e377457130db4ed15929133d1b463aff96
FINALIZER_SHA=24cea4f1fe28c89968cad1ed845fbfbd64514b0c
ROOT="$HOME/work/vllm.cpp-gdn-packed-trace/$RAW_SHA"
EVIDENCE="$ROOT/evidence/$RAW_SHA"
SOURCE="$HOME/work/vllm.cpp-gdn-packed-finalizer/$FINALIZER_SHA/source"
VERIFY="$ROOT/finalizer-replay-$FINALIZER_SHA"
test -z "$(git status --porcelain)"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$FINALIZER_SHA"
test ! -e "$VERIFY"
cp -a --reflink=auto "$EVIDENCE" "$VERIFY"
rm "$VERIFY/trace/27/gdn-packed-summary.json" \
  "$VERIFY/trace/27/gdn-packed-manifest.json" \
  "$VERIFY/trace/27/status-gdn-packed.json"
PYTHONPATH="$SOURCE" python3 \
  "$SOURCE/tools/bench/finalize_gdn_packed_trace.py" \
  --evidence "$VERIFY" --source-commit "$RAW_SHA" \
  --run-log "$ROOT/gdn-packed-run.log" \
  2>&1 | tee "$ROOT/gdn-packed-finalizer-replay.log"
```

Acceptance requires packed **915** versus rollback **963** total nodes, 145
BF16 GEMMs in both arms, 48 packed recurrence calls replacing 48 decomposed +
48 post-conv calls, exactly 48 BA projection nodes at the accepted `(8,1,1)`
geometry in each arm (hashed separately because BF16-vs-F32 output may change
the cuBLASLt tactic), and an identical normalized signature multiset for every
remaining kernel/memcpy/memset node.
Exact requirements are in the
[packed-decode spike](../.agents/specs/gdn-packed-decode.md).

Re-aggregate the **new binding** result without GPU work; exit **1** is expected
for a not-all-axes-pass gate, while exit 2 means malformed evidence. The prior
binding `3f256ab` re-aggregates identically from its own immutable root:

```sh
# New binding (49/124). Swap the SHA for 3f256ab… to re-aggregate the prior.
SOURCE="$HOME/work/vllm.cpp-online-gate/evidence/246a23cfa423e8e50c65b0ff067be55f3a3c7bf9"
CHECK="/tmp/vllm-cpp-246a23c-summary-$USER"
cp -a --reflink=auto "$SOURCE" "$CHECK"
rm -rf "$CHECK/summary-27"
set +e
PYTHONPATH="$PWD" python3 tools/bench/online_gate_summary.py \
  --evidence "$CHECK" --model 27
rc=$?
set -e
test "$rc" -eq 1
# Expected: report.md "Every-axis ratios failing or void: 75/124" (49 pass).
# Verify artifact hashes: ratios.json f784ba01…e046, all-runs.json b7ef3442…3240,
# manifest.json 7f25c614…83e8.
```

Relaunch the authorized timed grid from the pushed SHA. `--execute` is now a
PURE timed production grid (model gate + interleaved ours/vLLM legs +
memory/thermal/cache-drop capture + summary) from the profile-control-OFF
build; the H1d paired trace stays a separate `--trace-only` run that requires
the instrumented build:

```sh
SHA=$(git rev-parse HEAD)
CLAIM="$HOME/work/vllm.cpp-online-gate"
EVIDENCE="$CLAIM/evidence/$SHA"
# 1. Dry-run plan (no GPU lock).
scripts/dgx-online-serving.sh --dry-run --claim-root "$CLAIM" --vllm-cpp-sha "$SHA"
# 2. Byte-identical binding corpus copy into evidence/corpus/27.
scripts/dgx-online-serving.sh --prepare-corpus --model 27 \
  --source-corpus "$EVIDENCE/corpus/27" --evidence "$EVIDENCE"
# 3. Timed production grid (one /tmp/gpu lock for the whole model).
scripts/dgx-online-serving.sh --execute --model 27 \
  --snapshot "$SNAPSHOT" --source-corpus "$EVIDENCE/corpus/27" \
  --evidence "$EVIDENCE" --build-dir "$BUILD" --configure-log "$CONFIGURE_LOG"
```

## Correctness-only changes (benchmark disposition NOT APPLICABLE)

- **sm_120a (consumer Blackwell) as a BUILD-supported CUDA target - `NOT
  APPLICABLE` (2026-07-22, `CLAIM-CUDA-SM120-BRINGUP`,
  [spec §W8](../.agents/specs/cuda-arch-additivity.md)).**
  `benchmark_binding=false`. **No sm_120 hardware exists here, so there is
  nothing to benchmark and no number is owed, pending, or claimed.** The change
  is build-configuration plus records: a configure-tier feature-table test and
  its CI job, a precise Triton-AOT multi-arch diagnostic, and a documented cache
  entry. It adds no kernel, changes no dispatch and touches no numeric path.
  The GB10 relevance is purely negative and was verified, not assumed: the
  same-family fat binary `"120a;121a"` changes CODEGEN for the architecture we
  do run, so the full regression set was re-run ON the fat binary and is
  unchanged (see the ledger row). No existing binding number is created,
  re-based or invalidated. The first sm_120 speed number this project could ever
  publish requires an actual RTX 50-series card; the validation steps for an
  owner are in the spec.

- **sm_90a (Hopper) as a SINGLE-ARCH, PORTABLE-KERNELS-ONLY build-supported CUDA
  target - `NOT APPLICABLE` (2026-07-23, `CLAIM-CUDA-SM090-BRINGUP`,
  [spec §W9](../.agents/specs/cuda-arch-additivity.md)).**
  `benchmark_binding=false`. **No sm_90 (Hopper H100/H200) hardware exists here,
  so there is nothing to benchmark and no number is owed, pending, or claimed -
  and even a Hopper owner would get only the PORTABLE-KERNEL path, since every
  accelerated feature (FP4 MMA, CUTLASS NVFP4/FP8, Marlin, FlashAttention-2)
  resolves DISABLED for `90a` (we have no Hopper wgmma/TMA kernel body).** The
  change is build-enablement plus records: three native-fp4 helper functions in
  `cuda_matmul_nvfp4.cu` are gated on `VT_FP4_MMA_SM120A` so the TU compiles when
  the native path is off (which unblocks the single-arch `90a` build), plus
  feature-table test assertions (`90a`→all-EMPTY) and a cross-family registry
  test. The GB10 relevance is purely negative and was verified, not assumed: the
  guard change alters what `cuda_matmul_nvfp4.cu` compiles, so a full production
  `121a` build (CUTLASS+Marlin+FA2+Triton) was rebuilt `-Werror` 0-warn and the
  regression battery re-run - UNCHANGED (see the ledger row) - and the guards are
  no-ops on `121a` where `VT_FP4_MMA_SM120A` is defined. No existing binding
  number is created, re-based or invalidated. A vLLM-competitive Hopper speed
  number would first require the Hopper fast-path kernel bodies (a separate
  campaign) AND an actual Hopper card; the validation steps for an owner are in
  the spec.

- **Qwen3-32B NVFP4A16 (compressed-tensors W4A16) - CORRECTNESS MET,
  SPEED PENDING (2026-07-21, `CLAIM-QUANT-NVFP4-CT-W4A16`,
  [spike](../.agents/specs/sweep-qwen3-32b-nvfp4a16.md)).**
  `benchmark_binding=false`; **PENDING - deliberately not measured.** This change
  adds a new quantization scheme (compressed-tensors NVFP4A16 / W4A16) to the
  already-done dense `Qwen3ForCausalLM` forward. It creates, re-bases and
  invalidates **no** existing binding number: the BF16 path is selected by a
  per-Linear `.weight_packed` probe that is false on every previously benchmarked
  checkpoint, and the 27B/35B hot path is byte-untouched by construction
  (verified: 27B **235/235**, 35B **315/315**, Qwen3-Coder **138/138**,
  Qwen3-dense **664/664**, OPT **36/36**, all UNCHANGED; plus
  `test_ops_nvfp4_fp4` 27002/27002, `test_ops_moe_grouped` 146/146,
  `test_ops_moe_grouped_bf16` 19/19, and `compute-sanitizer memcheck` 0 errors).
  **Correctness is now MET (W4b, 2026-07-21) and speed is STILL not measured** -
  the two are separate gates and this entry stays `PENDING` until the throughput
  run below is executed. The strict token-exact bar scored **4/6 prompts (67/96
  tokens)** against the deterministic vLLM 0.25.0 oracle and was NOT loosened on
  that evidence; the ratified TEACHER-FORCING isolation
  (`scripts/qwen3-32b-nvfp4a16-neartie-gap.py`) was then run and measured **all
  29 divergent positions within 0.0625 nats of vLLM's own argmax given OUR
  prefix, with 28/29 EXACTLY 0.0** - one root flip being an exact bf16 tie at
  which vLLM's teacher-forced argmax is OUR token while its incremental greedy
  chose the other (vLLM contradicts itself). The residual is therefore the
  PRE-EXISTING dense-forward bf16 near-tie drift, recorded against the dense
  `Qwen3ForCausalLM` row, not the quantization - which is separately exonerated
  by the bit-exact CPU proof, `fallback_gemms=0`, and invariance across both of
  our quantized GEMMs. The gate now closes **6/6 (strict 4/6 + near-tie band 2/6,
  max gap 0.062 nats vs the 0.5-nat bar shared with the BF16 Qwen3 gates, 0
  forward-divergent)** with the per-position nats evidence committed as goldens;
  see the spike §6c.
  The future benchmark disposition is fixed now so it cannot be quietly loosened
  later: the vehicle is **`RedHatAI/Qwen3-32B-NVFP4A16`** measured against
  **graphed** vLLM 0.25.0 (`enforce_eager=False`,
  `CUDAGraphMode.FULL_AND_PIECEWISE`, `linear_backend='auto'` - which resolves to
  `MarlinNvFp4LinearKernel`, confirmed from the oracle's own startup log), via
  `vllm serve` + `vllm bench serve --dataset-name random --random-input-len 1024
  --random-output-len 128 --random-range-ratio 0 --ignore-eos --max-concurrency C`
  against our `examples/vllm-bench --input-len 1024 --output-len 128
  --concurrency C` at production defaults, on an idle box under `flock /tmp/gpu`
  with a fresh server per concurrency at a verified 0% prefix-cache hit rate.
  The bar is the standing one: **match or beat vLLM on EVERY axis** (total and
  output throughput, req/s, TTFT, TPOT/ITL, peak memory) at c1/c2/c4/c8, with
  token-exact correctness as a precondition that may never be traded off.

- **MLA + DeepSeek/Kimi/MiniMax campaign - W0-W8 (2026-07-22, `CLAIM-MLA-DEEPSEEK`,
  [spike](../.agents/specs/mla-deepseek-campaign.md)).**
  `benchmark_binding=false`; **NOT APPLICABLE - SUPERSEDED 2026-07-22 by W9**, which
  gave this track its binding number (see "Binding DeepSeek-V2-Lite (MLA) every-axis
  grid - MLA campaign W9" above; that entry is the current scoreboard for the
  track). W0-W7 were correctness-only and **unreachable from any existing hot
  path** - every new `vt::` op had no caller in any model TU until W7, the W1
  allocator refactor is byte-for-byte identical for every existing model by
  construction, and W2 changes only which backend NAME a `use_mla=true` request
  resolves to. W8 was the SACRED correctness gate (**8/8**: STRICT 5/8, near-tie
  3/8, max teacher-forced gap 0.25 nats, 0 forward-divergent), which is what moved
  the row to `ACTIVE` and explicitly NOT to `DONE`. The no-movement claim was
  proved, not assumed, at every step: 27B **235/235**, 35B **315/315**,
  Qwen3-Coder, Qwen3-dense and OPT all token-exact UNCHANGED, memcheck/racecheck/
  synccheck 0. One W9 input recorded here: the oracle must run
  `moe_backend='triton'`, because vLLM's auto-selected FlashInfer CUTLASS
  unquantized MoE backend REBOOTS dgx. Per-W chronology and evidence anchors live
  in the append-only [state log](../.agents/state.md) and
  [parity ledger](../.agents/parity-ledger.md), not here.

- **MLA campaign W10 - the blocked-row honesty pass (2026-07-22,
  `CLAIM-MLA-DEEPSEEK`, [spike](../.agents/specs/mla-deepseek-campaign.md)).**
  `benchmark_binding=false`; **NOT APPLICABLE - RECORDS ONLY. No code, no build, no
  GPU work, nothing downloaded, and NO NUMBER MEASURED, CREATED OR INVALIDATED.**
  Because no source file is touched, every binding result stands unchanged - the
  W9 DeepSeek-V2-Lite grid above remains the current scoreboard for this track
  (tok/s 0.87/0.95/0.86/0.88, TTFT 1.06/1.14/**0.96**/**0.88**, TPOT
  1.11/**0.97**/1.16/1.17, peak memory **0.46** at c1/c2/c4/c8 vs graphed vLLM
  0.25.0 `--moe-backend triton`), and the row stays `ACTIVE`, not `DONE`.
  **The benchmark-relevant output of this step is a permanent NO-BENCHMARK
  verdict per family, so no future session re-opens it as "pending":**
  DeepSeek-V3/V3.2 (671B, ~642 GiB fp8 / ~1250 GiB bf16), Kimi-K2/K2.5
  (~2000 GiB), MiniMax-M2 (~428 GiB), MiniMax-M3 and GLM-5 (1404 GiB) **do not fit
  GB10's 119 GiB of unified memory** (several also miss dgx's ~184 GiB of free
  disk - corrected down from the 238 GiB measured before the V2-Lite download), so
  **no benchmark will ever be run here for them**; V3.2 and GLM-5 are additionally
  DEP-BLOCKED because DSA's sole sm_121 backend candidate dispatches to
  flashinfer's dense-only XQA kernel, which discards `sparse_mla_top_k`.
  Kimi-Linear-48B (~89.4 GiB) stays HW-MARGINAL and would need a memory-pressure
  check before any number from it counted. **The one number this track can still
  produce beyond the `gemvx` lever is a GLM-4.7-Flash grid**
  (`Glm4MoeLiteForCausalLM`, 31.2B / **58.2 GiB - FITS**), which is also the only
  reachable vehicle that converts the campaign's two permanent unit-only coverage
  gaps (`noaux_tc` router, `q_lora` query branch) into e2e coverage; it is gated
  behind a staged download that must free disk first. **Next binding run for this
  track (unchanged):** land the named `gemvx` -> tensor-core dispatch lever
  DEFAULT-ON, then re-run the identical c1..c8 recipe recorded in the W9 section
  above (`~/w9mla/w9_ab.sh` + `~/w9mla/w9_vllm.sh`, idle box, one `flock /tmp/gpu`,
  fresh server per concurrency, 0.0% prefix-cache hit rate, cold leg discarded).
  Repro for THIS entry: none - nothing was executed; re-verify with
  `python3 scripts/check-agent-record.py` and
  `python3 scripts/check-doc-checkpoint.py`.

- **CUDA-arch additivity seams - per-arch build feature table, capability
  threading, runtime SM-dispatch registry, queried smem ceiling (2026-07-21,
  `BACKEND-CUDA-ARCH-ADDITIVITY`, `CLAIM-CUDA-ARCH-ADDITIVITY`).**
  `benchmark_binding=false`; **NOT APPLICABLE, and deliberately so.** This is a
  structural/mechanical change with no new kernel, no numerics change and no
  dispatch change on GB10: exactly ONE tactic is registered (the pre-existing
  `sm_12x` native fp4 path, still behind its default-OFF `VT_NVFP4_FP4_NATIVE`
  toggle), so on the gate models the new selector chooses nothing and the same
  portable kernels run as before, bit-identically. The only per-launch cost added
  is one cached-struct read plus a predicate over an 8-entry table, on a path the
  gate models do not take. **No speed credit is taken and no binding number is
  re-based.** The gate is behavior PRESERVATION, verified on the final binary:
  27B `test_qwen27_paged_engine` 235/235, 35B `test_qwen36_paged_engine` 315/315,
  Qwen3-Coder `test_qwen3coder_paged_engine` 6/6 - all UNCHANGED - plus the
  configure-level multi-arch evidence and the tactic-registry counters that prove
  the new seam actually executed. Reproduce:
  `cmake -S . -B build -DVLLM_CPP_CUDA_ARCHITECTURES="90a;121a"` (configure only)
  for the feature report, and `VT_ARCH_TACTIC_STATS=1 ./tests/test_ops_nvfp4_fp4`
  for the selection signal. Detail: [arch additivity
  spike](../.agents/specs/cuda-arch-additivity.md).

- **Sweep model #1 W0-W3 - Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`) registry
  stub + reusable-piece refactors + BF16 loader + forward (2026-07-21,
  `CLAIM-MODEL-QWEN3-CODER`).** Infrastructure + behaviour-preserving refactors +
  a host-only loader + a correctness forward; the SPEED deliverable is W5, so
  there is still nothing to benchmark and no perf claim. W0 = a new registry TU
  (`qwen3_moe_registry.cpp` + `qwen3_moe.h`). W1 = (#1) the dense self-attention
  block EXTRACTED verbatim to `dense_attn_block.h`, (#2) the bf16 `MoeBlock`
  EXPOSED cross-TU via `RunMoeBlock` (`qwen3_5_moe_block.h`), (#3) a
  no-shared-expert guard in `MoeBlock`. W2 = the BF16 loader
  (`qwen3_moe_weights.cpp`): merged qkv/o + per-head q/k norm + a NEW bf16
  per-expert loader (router + 128 experts × gate/up/down in Matmul-B layout, no
  shared expert) + UNTIED `lm_head` loaded separately. W3 = the forward
  (`qwen3_moe.cpp` `Qwen3MoeModel::Forward/ForwardDevice`) = the dense forward
  body with the per-layer MLP replaced by the MoE block, composing the reused
  `AttnBlock` + `RunMoeBlock`, wired into the factory. **Regression UNCHANGED**:
  Qwen3-dense 0.6B+4B near-tie 16/16, 27B 235/235, 35B 315/315, dgx CUDA
  `-Werror` 0-warn, memcheck 0. Load gate: all 18867 tensors mapped (131746
  assertions). Forward doctest: real-checkpoint prefill argmax = 12095 (" Paris")
  for "The capital of France is" - correct, a strong W3 sanity (the token-exact
  vs-oracle bar is W4). The SPEED deliverable (a fast BF16 grouped-MoE GEMM - only
  NVFP4-Marlin exists; the bf16 path is the slow per-expert reference loop) is W5
  and will be benchmarked then.
- **First additive-model bring-up W0+W1+W2 - Qwen3 dense (`Qwen3ForCausalLM`) +
  the runner generalization + the weight loader (2026-07-20,
  `CLAIM-MODEL-QWEN3-DENSE`, `ENG-RUNNER-MODELSHAPE`).** Infrastructure + a
  behaviour-preserving runner generalization + a host-only weight loader; NO
  forward yet (the Qwen3 dense forward lands W3), so there is nothing to
  benchmark and no perf claim. W0 adds a new registry TU (`qwen3_dense.cpp` +
  `qwen3.h`, one `REGISTER_VLLM_MODEL`, a full-attention-only KV spec, a
  clear-throwing forward stub). W1 makes `GPUModelRunner` **model-shape-agnostic**:
  a full-attention-only (empty `layer_types`, no GDN group) KV config allocates +
  steps without the hybrid GDN metadata path, gated on the model-agnostic
  `has_mamba_group`/`gdn_group_id_ >= 0` predicate. **The hybrid path is
  byte-identical.** W2 adds the **weight loader** (`qwen3_weights.cpp`): Qwen3-0.6B
  safetensors → `Qwen3DenseWeights` (merged qkv/gate_up per vLLM
  `packed_modules_mapping`, per-head q/k norm, tied `lm_head` aliasing
  `embed_tokens` with the checkpoint's `lm_head.weight` skipped - vLLM
  `skip_prefixes`); shared BF16 loader helpers extracted to
  `dense_weight_loaders.h` (27B load byte-identical). The behaviour-preservation
  gate is DGX 27B `test_qwen27_paged_engine` **235/235** + 35B
  `test_qwen36_paged_engine` **315/315** token-exact UNCHANGED (one flock, prod
  flags CUTLASS sm120a + FA2 sm_121a + Triton AOT). Clean CUDA `-Werror` **0
  warnings**; the W2 load gate `test_qwen3_load` loads all 311 Qwen3-0.6B tensors
  (**1567/1567** asserts, no leftover, tied lm_head resolves) on dgx,
  `compute-sanitizer memcheck` **0 errors / 0 leaks** on the load path; full CPU
  ctest **125/125**. ⇒ **NOT APPLICABLE** to the throughput / latency / memory
  scoreboard; `benchmark_binding=false`. The SACRED token-exact `Qwen3-0.6B` vs
  vLLM 0.25.0 oracle gate lands at W4.

- **Residency as a consumed `Platform::residency_policy()` capability (2026-07-19,
  extensibility item 2, `BACKEND-PLATFORM` / `CLAIM-BACKEND-PLATFORM-2`).** A
  behavior-preserving refactor: the MoE host-weight-release, per-layer load-stream
  interleave, and device-scratch-pool cap decisions in `qwen3_5.cpp` now READ
  `GetPlatform(<obj>.device.type).residency_policy()` (per-device) instead of an
  inline `device.type`/env gate - host-free via `ShouldReleaseHostWeights`,
  load-stream via `ShouldInterleaveLoadStream` (both `platforms/interface.h`),
  DevicePool soft cap via `residency_policy().device_pool_cap_bytes`. The residency
  POLICY comes from the platform; whether Marlin is the committed compute path
  stays the orthogonal *kernel* gate `MarlinMoeEnabled()`.
  `CudaPlatform::residency_policy().release_host_weights_after_upload` is flipped
  false→true (now CONSUMED ⇒ reproduces today's GB10 host-free-after-Marlin-build
  and the 35B ~4 GiB load-to-ready peak EXACTLY; `VT_MOE_HOST_FREE`/
  `VT_MOE_LOADSTREAM` stay as rollback overrides). No numeric/kernel/memory-behavior
  change on GB10 - so **NOT APPLICABLE** to the throughput / latency / memory
  scoreboard. Verified by the `test_platform` residency-consumption cases (7 cases /
  43 assertions), clean CPU `-Werror` build, full CPU CTest (126/126 in isolation;
  8 HTTP/engine/threadpool tests were `-j nproc` port/resource timeouts, all pass
  isolated), tools 164/164. **DGX behavior-preserving gate PASSED** (`~/work/vllm.cpp-residency-policy`
  @ `62fc0e0`, production flags CUTLASS sm120a + FA2 sm_121a + Triton AOT verified,
  one flock): clean CUDA `-Werror` 0 warn; 27B **235/235** + 35B **315/315**
  token-exact; **35B load-to-ready peak RSS (VmHWM) 4,201,632 KB ≈ 4.0 GiB - the
  ENG-MOE-LOADSTREAM ~4.19 GiB load-stream win PRESERVED**; `compute-sanitizer
  memcheck` 35B **0 errors / 315**. Additive win: a new (discrete) GPU sets
  `residency_policy()` field values with zero model-code edits.

- **Attention-backend registry + Platform-driven priority (2026-07-19,
  extensibility item 4, `BACKEND-ATTN-REGISTRY` / `CLAIM-ATTN-REGISTRY-1`).** A
  behavior-preserving, engine-level SELECTION refactor: *which* `AttentionBackend`
  a platform selects becomes data (a registered backend + a capability-ordered
  platform priority), not an inline code edit. Added
  `include/vllm/v1/attention/registry.{h,cpp}` (a `(DeviceType, name)` registry +
  `SelectAttentionBackendName`/`SelectAttentionBackend` selector, mirror of
  `vllm/v1/attention/backends/registry.py` self-registration +
  `cuda.py::get_attn_backend_cls`/`_get_backend_priorities`), filled
  `Platform::get_attn_backend_priority()` (the item-1 stub) with the vLLM-mirrored
  capability-ordered lists on `CudaPlatform`/`CpuPlatform`, and self-registered
  FLASH_ATTN + GDN_ATTN. No numeric/kernel/dispatch change - the concrete
  attention KERNEL stays selected at the device-agnostic vt:: op table
  (`GetOp(kPagedAttention)`, UNTOUCHED), and the walk resolves to the same
  FlashAttention-2 path on both gate models - so **NOT APPLICABLE** to the
  throughput / latency / memory scoreboard. Verified by the new
  `test_attn_backend_registry` (8 cases / 25 assertions) + updated `test_platform`,
  clean CPU `-Werror` build, full CPU CTest (126/126 in isolation;
  test_openai_conformance was a `-j nproc` HTTP-port flake, passes isolated),
  tools 164/164 + scripts 18/18. Behavior-preserving ⇒ both DGX model gates are
  unchanged (**DGX-confirmed 2026-07-19 @ `2c732e7`, production flags CUTLASS
  sm120a + FA2 sm_121a + Triton**: clean CUDA `-Werror`, 27B **235/235** + 35B
  **315/315** token-exact with FA2 selected, `compute-sanitizer memcheck` 35B **0
  errors**). MLA-branch priorities were deferred here and are now COMPLETE - see
  the MLA campaign W2 entry above (the whole of `_get_backend_priorities`, both
  branches, ported as a per-architecture data table; dense selection unchanged).

- **Model self-registration + per-arch entry-point TU split (2026-07-19,
  extensibility item 5, `MODEL-FACTORY-registry` / `CLAIM-MODEL-SELFREG-1`).** A
  behavior-preserving, code-organization-only refactor: the fixed
  `constexpr std::array<ModelRegistration,2> kRegistrations` is replaced by a
  `REGISTER_VLLM_MODEL(...)` static-`Registrar` self-registration idiom (copying
  the `RegisterOp`/`RegisterBackend`/`RegisterPlatform` pattern), and the Qwen
  dense/MoE arch-specific registry entry points move out of the
  `model_registry.cpp` monolith into NEW per-variant TUs `qwen3_5_dense.cpp` +
  `qwen3_5_moe.cpp` over a NEW shared `qwen3_5_common.{h,cpp}`. No
  numeric/kernel/dispatch change (the same factory functions run, byte-for-byte;
  `qwen3_5.cpp` is UNTOUCHED), so **NOT APPLICABLE** to the throughput / latency /
  memory scoreboard. Verified by the extended `test_model_registry` (138
  assertions + a new `self_registration` case), clean CPU `-Werror` build, full
  CPU CTest (all 125 pass in isolation; 5 HTTP/bench/capi tests are parallel
  port/resource-contention flakes - each PASSES isolated), tools 164/164, and
  (behavior-preserving ⇒ must be unchanged) both DGX model gates
  `test_qwen27_paged_engine` **235/235** + `test_qwen36_paged_engine` **315/315**
  token-exact + `compute-sanitizer memcheck` 35B **0 errors** (DGX-confirmed
  2026-07-19 @ `669679a`, production flags CUTLASS sm120a + FA2 sm_121a + Triton
  AOT). The deeper `qwen3_5.cpp` DevicePool/matmul/GDN shared-machinery factoring
  is a deferred follow-up.

- **Platform seam extraction (2026-07-18, `BACKEND-PLATFORM`).** A
  behavior-preserving refactor: added the C++ `Platform` capability seam
  (`include/vllm/platforms/interface.h` + `src/vllm/platforms/{platform,cpu,cuda}.cpp`,
  faithful mirror of `vllm/platforms/interface.py:134-229`) and migrated the 7
  memory-model / residency conditionals (KV-cache device residency + async
  device combine/scatter in `runner.cpp`, the decode-graph CUDA gate in
  `model_registry.cpp`, the host→device weight-residency branches in
  `qwen3_5.cpp`) from inline `device.type == kCUDA` tests to
  `CurrentPlatform()` capability calls. No kernel/numeric/dispatch change (the
  same branch is taken on CUDA/CPU as before), so **NOT APPLICABLE** to the
  throughput / latency / memory scoreboard. Verified by the new `test_platform`
  unit test, the full CPU CTest suite, and (behavior-preserving ⇒ must be
  unchanged) both DGX model gates `test_qwen27_paged_engine` 235/235 +
  `test_qwen36_paged_engine` 315/315.

- **CPU `kCastF32` kernel registration (2026-07-18).** Registered the missing
  CPU `bf16→f32` cast kernel so the CPU backend supports both cast directions
  (mirrors the already-present `f32→bf16` `kCastBf16`), closing a CPU
  op-registration asymmetry. CPU-only op registration + unit test; no CUDA path
  change and no gate-workload impact, so **NOT APPLICABLE** to the throughput /
  latency / memory scoreboard. Verified by `test_ops_glue` (`cast_f32` RED→GREEN)
  and the full CPU CTest suite.

- **M=1 decode MoE routing/align parallelization (2026-07-18,
  `CLAIM-MOE-DECODE-PARALLEL-1`).** The two MoE decode kernels that at M=1 (c1)
  launched a single block and ran serially - MoeAlign (29.3 µs vs vLLM 3.6 µs,
  8.2×) and MoeRouterTopK (20.2 µs vs vLLM 6.7 µs, 3.0×), together ~1.7 ms/tok ≈
  63% of the grounded 2.7 ms 35B c1 TPOT gap - are parallelized (router →
  per-thread argmax + tree reduction; align → one-thread-per-expert
  `cub::BlockScan` prefix sum), mirroring vLLM `topk_softmax_kernels.cu` /
  `moe_align_sum_kernels.cu`, plus L3 block_size_m 16→8 at low M and L4 removal of
  4 redundant Marlin workspace memsets. **BYTE-EXACT** to the retained serial
  reference (router 72/72 + align 60/60 parity), **27B 235/235 + 35B 315/315
  token-exact**, memcheck-clean 35B decode; efficiency-only ⇒ shipped ON by
  default. Per-kernel nsys same-box A/B (c1/M=1, `--cuda-graph-trace=node`,
  baseline = serial-kernel build a7d08d7 on-box): **MoeAlign 29.5 µs → 3.0 µs
  (9.8×, now below vLLM's 3.6 µs); MoeRouterTopK 20.2 µs → 12.3 µs (1.64×,** vs
  vLLM's 6.7 µs - byte-exactness ties the softmax reduction to our tree order,
  so vLLM's register-fused `topkGating` is off-limits). Combined excess vs vLLM
  39.2 µs → 5.0 µs per invocation (~87% eliminated). memcheck 35B graph decode
  0 errors. The in-situ c1/c8 TPOT recovery is `benchmark_binding=false` here -
  the 35B v0.25.0 performance grid re-measures it (the ~1.7 ms/tok c1 target; no
  regression at c16/c32, where the serial cost already amortized). Evidence
  `dgx:~/work/vllm.cpp-moe-decode-par`.

## Benchmark policy

- Correctness is a precondition and cannot be traded for speed.
- Ours and every reference use the same model, requests, sampling, token
  budget, cache policy, concurrency, and hardware.
- A benchmark series runs on an idle GPU under one ownership window and is
  repeated enough to distinguish signal from run noise.
- Partial, contended, stale-denominator, or diagnostically incomplete results
  are `VOID`; they never contribute to an accepted ratio.
- Every 27B throughput, latency, and memory axis must pass before 35B
  performance or broader roadmap execution.

The complete contract is in the
[benchmark protocol](../.agents/benchmark-protocol.md) and
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md).

**Breadth sweep note (2026-07-21):** the active phase is model-architecture breadth (recent-first), each held to token-exact + vLLM-speed on every axis. Ranked queue + CUDA-arch additivity audit: [.agents/specs/breadth-sweep-plan.md](../.agents/specs/breadth-sweep-plan.md). CUDA archs beyond same-family sm_120 are HW-blocked (only GB10 testable).

**Sweep model #1 - W0-W4 LANDED / CORRECTNESS-DONE (2026-07-21): Qwen3-Coder-30B-A3B** (`Qwen3MoeForCausalLM`, BF16 full-attention MoE) - [spike](../.agents/specs/sweep-qwen3-coder-30b.md). Composes the done Qwen3-dense attention + the done 35B MoE experts (zero runner change); 4 model-layer seams (extract/expose/guard + a new bf16-expert loader); the SPEED gap is a fast bf16 grouped-MoE GEMM (W5). No quant (correctness covered by the reference MoE path). W0-W3 (registry stub + extract/expose/guard + bf16 loader + forward) landed behaviour-preserving; load gate all 18867 tensors mapped; forward doctest real-ckpt prefill argmax=12095 (" Paris"). **W4 SACRED correctness gate PASSES 6/6** (`test_qwen3coder_paged_engine.cpp`, paged-engine greedy vs vLLM 0.25.0): vLLM's own greedy is DETERMINISTIC (K=5, 0 multi-valued cells) → STRICT-where-well-posed near-tie-robust gate (dense methodology) - STRICT token-exact 4/6 + near-tie-band 2/6, max teacher-forced gap 0.125 nats, 0 forward-divergent; regression UNCHANGED (27B 235/235, 35B 315/315), `-Werror` 0-warn, memcheck 0. The bf16-MoE reference-loop SPEED benchmark stays NOT-APPLICABLE/PENDING until W5 (the fast grouped-MoE GEMM). Oracle used vLLM's TRITON MoE backend (FlashInfer-CUTLASS OOMs the 57 GiB model on GB10's 119 GiB unified pool). W5 fast-MoE + every-axis speed remain.

**Qwen3-Coder-30B W0-W4 landed / CORRECTNESS-DONE (2026-07-21):** W0-W3 (registry stub + extract/expose/guard + bf16 loader + forward) behavior-preserving (27B 235/235 + 35B 315/315 + Qwen3-dense 0.6B/4B 16/16 UNCHANGED). W4 SACRED gate `test_qwen3coder_paged_engine.cpp` **6/6 PASS** (vLLM deterministic K=5 → near-tie-robust strict-where-well-posed gate: 4/6 strict + 2/6 near-tie, max gap 0.125 nats, 0 divergent). W5 bf16-fast-MoE (SPEED) next.

## CPU vs llama.cpp - RSS gap CLOSED to parity (L7: repack-source release + prefault + keep-f16 default-on), `ACCEPTED` / binding (2026-07-23)

`BENCH-CPU-LLAMA` / `QUANT-GGUF-KEEPQ-LOADER` **L7** (`CLAIM-QUANT-GGUF-RSS-L7-1`;
[keep-quant loader leaf](../.agents/specs/gguf-keep-quant-loader.md) § L7 result).
**This is the current binding CPU result and it REVERSES the L6 refutation below:
the CPU peak-RSS gap is CLOSED - CPU is now at or ahead of llama.cpp on EVERY axis.**

**The profile disproved the L6 "engine workspace" attribution.** With per-pool
instrumentation the DevicePool (process-wide retain-don't-free scratch) measured
**19.98 MiB** and the whole KV cache (6 full-attn layers × 256 blocks + GDN state)
**114.6 MiB** - both already ≤ llama.cpp. The residual was WEIGHT residency: on Arm
the G7 repack COPIES each q8_0 weight into an anonymous buffer, and because keep-f16
holds the mmap open, the DEAD source q8_0 pages stayed file-backed - a ~1 GiB
double-count. Releasing them (`OwnGgufQuantBlocks` → `DropSpanResidency`, port of
llama.cpp `unmap_fragment` `llama-mmap.cpp:490`) removes the duplicate; a load-time
`PrefaultBorrowedSpan` (port of llama.cpp's mmap prefetch `llama-mmap.cpp:451`)
removes L6's prefill regression; keep-f16 flips **default ON** (`VT_GGUF_KEEP_F16=0`
opt-out).

**Binding arm = `dgx.casa` (GB10, aarch64, 20 cores), idle** (loadavg-gated < 0.8),
whole series one `flock $HOME/gpu.lock`, `git archive` transfer, base-vs-L7
same-binary discipline, `Qwen3.5-2B-UD-Q8_K_XL.gguf`, 3 reps. llama.cpp `237ad9b96`
fresh on the same host: **pp128 173.2±2.7, tg32 25.09±0.33, peak RSS 2,934,152 KB =
2.798 GiB**.

| arm | peak RSS | ratio vs llama | TTFT med (ms) | prefill t/s | TPOT (ms) | decode t/s | file-backed | anon | token md5 |
|---|---|---|---|---|---|---|---|---|---|
| base default (keep-f16 off) | 3.885 GiB | 1.389× | 570/571/574 | 224 | 40.4 | 24.7 | 0.959 | 2.923 | `809f2d0…` |
| **L7 default (keep-f16 + release + prefault)** | **2.832 GiB** | **1.012× (PARITY)** | 628/625/625 | **204** | 40.95 | 24.4 | 1.629 | 1.200 | `809f2d0…` |
| oracle (`VT_CPU_REF=1`) | - | - | - | - | - | - | - | - | `809f2d0…` |

**Peak RSS 3.885 → 2.832 GiB = 1.39× → 1.01× llama.cpp - the CPU RSS gap is CLOSED.**
The clean base-vs-L7 isolation shows the win is File 2.632 → 1.629 (−1.0 GiB, the
released q8_0 source) at unchanged anon 1.200. **Prefill 204 t/s = 1.18× AHEAD** of
pp128 173.2 (the prefault erased L6's 0.72× regression; ~9% under the keep-f16-off
default's 224 but comfortably above the competitor floor). **Decode 24.4 t/s ≈
parity** (llama 25.09; native-f16 compute costs ~1.4% vs the bf16 default, within the
known decode-scaling item). **Output token md5 `809f2d0d6aac93a11faecd68df8a131f`
byte-identical across base, L7-default and `VT_CPU_REF=1`.** Anon 1.200 GiB is
IRREDUCIBLE (repacked q8_0 1.06 + KV 0.115 + pool 0.02); the repacked q8_0 is what
makes Arm prefill fast (G7) and llama.cpp carries the same - there is no further
loader RSS lever. **This is the last open CPU gap; with it closed, CPU is at or
ahead of llama.cpp on peak RSS, prefill and decode simultaneously.**

**Repro:** the L4/G4 recipe; the new default IS keep-f16+release+prefault (no env);
`VT_GGUF_KEEP_F16=0` reproduces the 3.885 GiB base; `VT_GGUF_PREFAULT=0` isolates the
prefill-fault effect; `VT_GGUF_RELEASE_PAGES=0` reverts the source-page release. The
per-pool figures use a temporary `VT_POOL_STATS`/`VT_KV_ALLOC_LOG` instrumentation
(reverted before the binding binary was built). Regressions (CUDA build, standalone,
keep-f16 inert on CUDA - expand_nk false): 27B 235/235, 35B 315/315, Qwen3-Coder 6/6,
Qwen3-dense 16/16, OPT 6/6, DeepSeek-V2 8/8, Llama 16/16, `test_qwen36_gguf_engine`
28/28 (16/16 both APEX files); `test_gguf_keep_quant` 36/36 (x86 + aarch64).

## CPU vs llama.cpp - G7 q8_0 repack-at-load (prefill crosses parity), `ACCEPTED` / binding (2026-07-23)

`QUANT-GGUF-CIQ-GEMM` G7 / `BACKEND-GATE-CPU-LLAMACPP` (`CLAIM-QUANT-GGUF-CIQ-G7-1`;
[spec](../.agents/specs/gguf-compute-in-quant-gemm.md) § G7 result). Base
`99b7443`, binding idle-gated `dgx.casa` (GB10, aarch64, 20 cores), one
`flock $HOME/gpu.lock`, `git archive` transfer, `Qwen3.5-2B-UD-Q8_K_XL.gguf`,
same-binary A/B. llama.cpp `237ad9b96` refreshed same session (`-t20 -r5 -ngl0`):
**pp128 177.32±2.50**, tg32 25.40±0.80.

**The lever (profile-ranked #1):** the fresh op-dispatch profile put
`kMatmulBTQuant` (the q8_0 quant GEMM) at **55 %** of prefill. G7 ports
llama.cpp's repack-at-load `q8_0_4x8` tier (`ggml_repack_get_optimal_repack_type`
→ `q8_0_4x8_q8_0`, `repack.cpp:4683 @ 237ad9b96`): the loader repacks each q8_0
weight ONCE into the `block_q8_0x4` i8mm interleave, and `kMatmulBTQuant`
dispatches a pre-shuffled gemm/gemv that reads contiguous aligned blocks with no
in-register row shuffles.

**BIT-IDENTICAL** (target met, not a near-tie): byte-permute weight + non-fused
`vmlaq_f32` in tier-0 block order under `-ffp-contract=off`. Output-token md5
`d235db12f2cd304007530286a1755c95` byte-identical across repack-ON (default),
repack-OFF (`VT_CPU_QUANT_REPACK=0`) and `VT_CPU_REF=1`.
[test_ops_quant_repack](../.agents/../tests/vt/test_ops_quant_repack.cpp) 305
assertions on dgx: repacked gemm/gemv `memcmp`-equal to the plain path across
decode/leftover/prefill, f32+bf16 out, strided activations, threads 1/2/4/20.
`test_qwen36_gguf_engine` STANDALONE 2/2 · 16/16 on APEX Compact+Balanced with
repack live.

**Op-level q8_0 GFLOP/s (cache-warm, mmla → repacked):** prefill qkv 518.75 →
**2400.92** (4.63×), gate_up 583.37 → **3455.86** (5.92×), down 514.13 →
**1901.66** (3.70×), decode qkv (M=1) 408.75 → 418.54 (1.02×, memory-bound).

**E2E prefill A/B** (`--input-len 128 --output-len 32`, 20 threads, 6 interleaved
reps, cold discarded, median 2-6):

| arm | TTFT median (ms) | prefill t/s | TPOT (ms) |
|---|---|---|---|
| BEFORE `VT_CPU_QUANT_REPACK=0` (mmla) | 1096.38 | 116.7 | ~40.5 |
| **AFTER - repack default** | **571.88** | **223.8** | ~40.5 |

**Same-binary prefill 1.92×.** Prefill 223.8 t/s vs llama.cpp pp128 177.32 =
**1.26× - AT/BEYOND llama.cpp parity** (from 0.66× / 1.52× behind on the
repack-OFF arm; the same-binary 1.92× on the recorded 1.32×-behind idle baseline
likewise lands beyond llama). Decode TPOT ~40.5 ms unchanged (24.7 t/s ≈ llama
tg32 25.4, at parity - repack neutral at M=1). Peak RSS 3.884 GiB unchanged.

**A wiring bug this gate caught:** the op path was bit-identical from the start,
but the first E2E run emitted ALL-ZERO tokens - `ResidentWeight`/`MakeTensor`
(qwen3_5.cpp) dropped the `OwnedTensor.repacked` marker, so the kernel read
repacked bytes as a plain q8_0 weight. Fixed at that single host→kernel
weight-tensor chokepoint (covers every weight incl. experts); `Tensor::Slice`
now throws on a repacked weight. Post-fix md5 is byte-identical (above).

**Fresh post-change profile + what remains:** the q8_0 GEMM's ~4.6× drops its
prefill share from 55 % (~603 ms) to ~21 % (~131 ms); the now-dominant prefill
cost is the elementwise f16 GEMM (`kMatmulBT`, the tied lm_head/embed + f16 ffn,
~48 %, UNCHANGED - not a block encoding). **But CPU prefill now matches/beats
llama.cpp on this file**, so the CPU prefill-lever search is CLOSED: what remains
vs llama.cpp is **peak RSS (1.39×, loader/f16-expansion bound, not compute)**.

Gates: clean CPU + CUDA `-Werror` 0 warnings; CPU quant/gguf units green; CUDA
regression set UNCHANGED, each STANDALONE (27B 235/235, 35B 315/315, Coder 6/6,
Qwen3-dense 16/16, OPT 6/6, DeepSeek-V2 8/8, Llama 16/16); goldens content-hash
identical before/after; checkers green.

## CPU vs llama.cpp - GDN split-projection orientation (last slow-orientation GEMM), `ACCEPTED` / binding (2026-07-23)

`BACKEND-GATE-CPU-LLAMACPP` / `QUANT-GGUF-KEEPQ-LOADER`
(`CLAIM-CPU-GDN-ORIENT-1`; [spec](../.agents/specs/cpu-gdn-proj-orientation-2026-07-23.md)).
Base `bead27d`, binding host idle-gated `dgx.casa` (GB10, aarch64, 20 cores),
one `flock $HOME/gpu.lock` (loadavg 0.43 at series start, gated < 1.2),
`git archive` transfer, `Qwen3.5-2B-UD-Q8_K_XL.gguf`, same-binary A/B.

**Ranked by a FRESH op-dispatch profile of the current binary** (temporary
`vt::GetOp` wall-time hook + a per-GEMM shape histogram, reverted before the
binding run). Warm prefill (`--input-len 128 --output-len 1`, 20 threads, total
1070 ms = TTFT): kMatmulBTQuant **50.7 %**, **kMatmul (slow [K,N]) 17.9 % / 72
calls**, kMatmulBT ([N,K]) 14.9 % / 20, kGdnPrefill 5.2 %, kPagedAttention 3.2 %.
The histogram pinned the 72 `kMatmul` calls EXACTLY: `M=128 N=2048 K=2048 ×36`
+ `M=128 N=16 K=2048 ×36` = the four GDN split projections (`in_proj_qkv/z/b/a`)
across all 18 GDN layers, the ONE weight family the loader still transposed into
Matmul-B [K,N] (nk=false) after G4 gave every other expanded weight the file's
own [N,K] order. **Gain ÷ effort #1**: LOW effort, bit-identical, moves 18 % of
prefill to the M-blocked kernel (G7 repack of the 50.7 % quant GEMM is larger but
MEDIUM-HIGH effort - the NEXT lever, not this one).

**The fix:** `GgufLoadPolicy::gdn_expand_nk` (rides `expand_nk`; `VT_GGUF_GDN_NK=0`
the A/B opt-out, `VT_CPU_REF=1` the historical transpose) keeps those projections
in [N,K] nk=true so they reach `vt::MatmulBT`. Bit-identical, not near-tie:
`MatmulBT` and `Matmul` are the same sequential f32 K-reduction differing only in
the weight memory offset. Grounded in llama.cpp keeping every GGUF weight in its
native `[out,in]` disk order (`ggml/src/ggml-cpu/ggml-cpu.c:1245-1443 @ 237ad9b96`).

**Correctness - NO token movement:** output-token md5
`d235db12f2cd304007530286a1755c95` byte-identical across AFTER (default), BEFORE
(`VT_GGUF_GDN_NK=0`) and ORACLE (`VT_CPU_REF=1`), and at thread counts 1/4/20.
`test_qwen36_gguf_engine` STANDALONE **2/2 cases, 28/28 assertions, 16/16 tokens**
on APEX-Compact AND APEX-Balanced vs the same-file llama.cpp oracle.

**Prefill e2e A/B** (`--output-len 32`, `VLLM_CPP_CPU_THREADS=20`, 5 reps, cold
rep discarded, median 2-5). llama.cpp refreshed same session (`-t20 -r5 -ngl0`):
pp512 211.25, **pp128 176.01±2.04**, tg32 24.93.

| arm | TTFT median (ms) | prefill t/s | TPOT (ms) |
|---|---|---|---|
| BEFORE `VT_GGUF_GDN_NK=0` | 1125.05 | 113.8 | 44.1 |
| **AFTER (default)** | **1031.94** | **124.0** | **40.4** |

**Same-binary prefill 1.090×** (box-state-robust; a second gated run corroborates
1.091×). Decode 1.09× (TPOT 44.1→40.4 ms → 24.75 t/s = **1.01× llama tg32, at
parity**). RSS 3.884 GiB both arms (transpose removed, resident bytes identical).
This session's absolute t/s ran ~7 % below the idle G6 baseline (residual
same-session load), so the vs-llama absolute uses that recorded idle baseline:
122.2 t/s (1.44×) × 1.090 = **133 t/s ≈ 1.32× behind**.

**Fresh post-change profile:** `kMatmul` **17.9 % → 0 % (72 → 0 calls, ELIMINATED)**;
`kMatmulBT` **14.9 % → 27.7 % (20 → 92 calls)** - it absorbed the 72. Next CPU
prefill lever = the quant GEMM (kMatmulBTQuant, now 55 %): G7 repack-at-load.

Gates: clean CPU `-Werror` 0 warnings; CPU unit ctest 136/136; CUDA clean
`-Werror` 0 warnings (change is CUDA-inert: `gdn_expand_nk` is false off the CPU
quant path, safetensors never reaches `LoadGdnGguf`); regression set UNCHANGED,
each STANDALONE (27B 235/235, 35B 315/315, Coder 6/6, Qwen3-dense 16/16, OPT 6/6,
DeepSeek-V2 8/8, Llama 16/16).

## CPU vs llama.cpp - Arm i8mm (mmla) quant-GEMM tier (CIQ G6), `ACCEPTED` / binding (2026-07-23)

`CLAIM-QUANT-GGUF-CIQ-G6-1`, base `93b55c0`, binding host idle-gated `dgx.casa`
(GB10, aarch64, 20 cores), one `flock $HOME/gpu.lock`, `git archive` transfer.
The i8mm `nrc==2` `vmmlaq_s32` `vec_dot` tier for q8_0/q4_0/q4_K/q6_K, 2x2-tiled
into `kMatmulBTQuant` (`src/vt/cpu/cpu_quant_dot_arm.cpp`); q3_K/q5_K have no
upstream mmla and stay portable. Runtime `HWCAP2_I8MM` probe; `VT_CPU_QUANT_MMLA=0`
is the same-binary A/B defeat.

**Op-level GFLOP/s (`kMatmulBTQuant`, `quant-gemm-bench`, best-of-N, portable→i8mm):**

| shape (M×N×K) | q8_0 | q4_K | q6_K |
|---|---|---|---|
| prefill qkv 128×3072×2048 | 450-492→**518** (~1.2×) | 148→**1109** (**7.5×**) | 240→**930** (**3.9×**) |
| prefill gate_up 128×12288×2048 | 492→**580** (1.18×) | 152→**1270** (**8.4×**) | 248→**1070** (**4.3×**) |
| prefill down 128×2048×6144 | 454→**514** (1.13×) | 148→**1058** (**7.1×**) | 238→**905** (**3.8×**) |
| decode qkv 1×3072×2048 | 412→412 (mmla off at M=1) | 139→143 | 227→227 |

q8_0's portable kernel was already near-optimal (a simple int8 dot the compiler
vectorizes); the K-quants gain most (mmla does the block unpack in-register vs
the portable scalar `aux8` decode).

**E2e prefill A/B - same binary, `Qwen3.5-2B-UD-Q8_K_XL`, 3 reps, medians:**

| arm | TTFT (ms) | prefill t/s |
|---|---|---|
| mmla OFF (portable) | 1135.68 | 112.7 |
| **mmla ON (default)** | **1047.70** | **122.2** |

Same-binary **1.084× prefill**; vs llama.cpp pp128 (refreshed same session,
175.41±1.62): **1.56× → 1.44× behind**. Decode TPOT ~44 ms unchanged (mmla off at
M=1). This file is the PESSIMAL case for G6 - q8_0 (portable already fast) + 60 %
`f16` (elementwise GEMM, not G6's domain) - so the e2e win is Amdahl-bounded; the
4-8× K-quant kernel win lands end-to-end on the k-quant-heavy APEX 35B files.

**Correctness:** q8_0/q4_0 mmla BIT-IDENTICAL to the portable/scalar tier
(`vmlaq_f32` non-fused under `-ffp-contract=off`); q4_K/q6_K within NMSE ≤ 5e-4;
mmla GEMM bit-identical across threads 1/2/4/20. `test_ops_quant_dot` 19 cases /
78,162 assertions; `test_qwen36_gguf_engine` 2/2 · 16/16 on APEX-Compact and
-Balanced vs the same-file llama.cpp oracle; bench-file token md5
`d235db12f2cd304007530286a1755c95` byte-identical across mmla-OFF/ON/`VT_CPU_REF=1`.
CUDA `-Werror` 0-warn; regression set UNCHANGED. **Fresh bottleneck:** mmla drops
the q8_0 `kMatmulBTQuant` share from 50 % to ~44 % of prefill (the 8.4 % e2e is
exactly `0.50·(1−1/1.2)`), so the elementwise f16/f32 GEMM (~30 %, unchanged - the
60 % f16 mass incl. the tied lm_head/embed) is now co-dominant and the next CPU
prefill lever on this file; on a k-quant file `kMatmulBTQuant` collapses (7-8× on
q4_K) and the elementwise GEMM + attention dominate outright.

**Reproduction:**

```sh
# dgx.casa, aarch64, idle, ONE flock; loadavg-gate inside the lock
R=$HOME/work/bench-cpu-llama; M=$R/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=$R/llamacpp/build/bin; W=$HOME/work/vllm.cpp-ciq-g6/build-cpu/examples
cmake -S vllmcpp -B build-cpu -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF
cmake --build build-cpu -j 20 --target vllm-bench quant-gemm-bench
flock $HOME/gpu.lock sh -c '
  VT_CPU_QUANT_MMLA=0 VLLM_CPP_CPU_THREADS=20 '"$W"'/quant-gemm-bench   # portable
  VLLM_CPP_CPU_THREADS=20 '"$W"'/quant-gemm-bench                        # i8mm
  LD_LIBRARY_PATH='"$LB"' '"$LB"'/llama-bench -m '"$M"' -p 128 -n 32 -t 20 -r 5 -ngl 0
  for arm in "VT_CPU_QUANT_MMLA=0" ""; do for rep in 1 2 3; do
    env $arm VLLM_CPP_CPU_THREADS=20 '"$W"'/vllm-bench --model '"$M"' --num-prompts 1 \
      --input-len 128 --output-len 32 --concurrency 1 --seed 0 --temperature 0 \
      --output-token-ids /tmp/ids.$arm.$rep.json
  done; done'  # all token-id files MUST md5 d235db12f2cd304007530286a1755c95
```

## CPU vs llama.cpp - thread kGdnPrefill + kPagedAttention, `ACCEPTED` / binding (2026-07-23)

`BENCH-CPU-LLAMA` / `BACKEND-GATE-CPU-LLAMACPP` / `QUANT-GGUF-CPU-THREADPOOL`
(`CLAIM-CPU-THREAD-GDN-PAGED-1`; [spec](../.agents/specs/cpu-thread-gdn-paged-2026-07-23.md)).
Binding host **dgx.casa (GB10, aarch64, 20 cores), idle** (loadavg 0.25 inside
one `flock $HOME/gpu.lock`, gated < 1.2, foreground; a first attempt at loadavg
2.87 was ABORTED by the gate and is VOID). `git archive` transfer,
`Qwen3.5-2B-UD-Q8_K_XL.gguf`, same-binary A/B, 4 reps, cold rep discarded.

**Prefill 1.38× faster same-binary (73.0 → 100.9 t/s); vs llama.cpp 2.43× → 1.76×
behind. Decode stays at parity, output tokens byte-identical.** The two serial
non-GEMM kernels on the prefill hot path are now threaded via the existing
`ParallelForRows`: kGdnPrefill over the (sequence, value-head) axis, kPagedAttention
over query-token rows. Both **bit-identical** to single-thread (disjoint
output/state partition, no reduction reassociation): benchmark output-token md5
`d235db12f2cd304007530286a1755c95` unchanged at threads 1/4/20 and `VT_CPU_REF=1`;
`test_cpu_threadpool` determinism battery (extended: single-seq `Hv=20` GdnPrefill
+ 37-token causal PagedAttention) byte-identical at 1/3/20; full CPU ctest 158/158.

**llama.cpp floor refreshed same session** (`llama-bench -t20 -r5 -ngl0`): pp512
**213.33±1.60**, pp128 **177.54±2.16**, tg128 25.51±0.30, tg32 25.30±0.89.

**Prefill e2e A/B** (`--input-len 128 --output-len 32 --concurrency 1
--seed 0 --temperature 0`, `VLLM_CPP_CPU_THREADS=20`, TTFT-derived = 128000/TTFT_ms,
reps 2-4 after cold discard):

| arm | TTFT median (ms) | prefill t/s | vs llama pp128 177.54 |
|---|---|---|---|
| BEFORE `4884d03` | 1753.46 | **73.0** | 2.43× behind |
| AFTER `e53ae4c` | 1269.08 | **100.9** | **1.76× behind** |

Same-binary **prefill speed-up 1.382×** (TTFT 1753→1269). Decode (settled,
`--output-len 96`): BEFORE 40.3 ms / AFTER 41.0 ms TPOT → **at parity** (1000/41.0
= 24.4 t/s = 1.04× vs llama tg32 25.30, inside run-to-run spread). The
`--output-len 32` arm shows a larger early-decode warmup (41.5→44.6) that
amortizes away by 96 tokens - a timing artifact of the faster TTFT, not a kernel
change (GdnDecode/decode-paged code paths are unchanged: paged attention at c1
decode has total_q=1 ⇒ inline, GDN decode uses the same recurrence).

**Op-level thread-scaling 1→20** (standalone harness, model c1-prefill shapes:
GDN n=1 T=128 Hv=16 Dk=Dv=128; Paged T=128 Hq=8 KVh=2 D=256; median of 60):

| nth | kGdnPrefill ms | speed-up | kPagedAttention ms | speed-up |
|---|---|---|---|---|
| 1 | 21.36 | 1.00× | 51.64 | 1.00× |
| 2 | 10.92 | 1.96× | 28.67 | 1.80× |
| 4 | 5.47 | 3.91× | 15.55 | 3.32× |
| 8 | 2.75 | 7.76× | 8.04 | 6.42× |
| 20 | 3.02 | 7.08× | 5.77 | **8.96×** |

kGdnPrefill saturates ~7.8× (peaks at 8 threads): it partitions over `Hv=16`
value-heads on this model, so 20 threads leaves 4 idle plus kick overhead;
kPagedAttention (128 query rows) keeps climbing to 8.96×. GDN's finer per-channel
axis would lift the head-count cap but was not needed to move prefill.

**Fresh post-change op-dispatch profile** (uncommitted `VT_OP_PROFILE` GetOp hook,
100 % of wall time attributed, reverted before the binding runs;
`--input-len 128 --output-len 4`, prefill-dominant ~90 %). Prefill is now
**GEMM-bound again** - the two threaded kernels dropped from 35 % combined to
**8.6 %**:

| op | share (was) |
|---|---|
| kMatmulBTQuant | **50.2 %** (37 %) |
| kMatmul | 16.1 % (12 %) |
| kMatmulBT | 14.2 % (10 %) |
| kAttnQkNormRopeGate | 2.0 % |
| kRmsNorm | 1.9 % |
| kCastBf16 | 1.7 % |
| **kPagedAttention** | **4.6 %** (10 %) |
| **kGdnPrefill** | **4.1 %** (25 %) |
| everything else | < 1.3 % each |

**New bottleneck named: the GEMMs (kMatmulBTQuant 50 % + kMatmul 16 % + kMatmulBT
14 % = 80 % of prefill).** These already run at 211-417 GFLOP/s, so the next CPU
prefill lever is the SIMD/repack GEMM tiers (`QUANT-GGUF-CIQ` G5 x86 AVX2/AVX512,
G6 Arm NEON/i8mm mmla, G7 repack-at-load), exactly as the floor-remeasurement's
deferred plan predicted - NOT another non-GEMM kernel. RSS (loader) is the other
remaining single deficit.

## CPU vs llama.cpp - keep-quant loader RSS (L5), `ACCEPTED` / binding (2026-07-23)

`BENCH-CPU-LLAMA` / `BACKEND-GATE-CPU-LLAMACPP` / `QUANT-GGUF-KEEPQ-LOADER`
**L5** (`CLAIM-QUANT-GGUF-KEEPQ-L5-1`). Full result:
[keep-quant loader leaf](../.agents/specs/gguf-keep-quant-loader.md) § L5 result.

**Peak RSS 6.401 → 3.884 GiB - the 2.29× → 1.39× llama.cpp move. Decode stays at
parity and output tokens stay byte-identical; prefill TTFT +4% (deferred paging).**

The loader now (1) BORROWS kept q8_0 blocks in place out of the GGUF's read-only
mmap instead of copying them into owned buffers (refcounted `GgufMapping`;
mirrors llama.cpp `load_data_for` under `use_mmap`), (2) SHARES one bf16 vocab
matrix between a tied `token_embd`/`lm_head` instead of materialising it twice
(llama.cpp `TENSOR_DUPLICATED`), and (3) DROPS the read-once file pages of
tensors it has finished expanding (`MADV_DONTNEED`, a per-tensor port of
llama.cpp `unmap_fragment`). All three are CPU-only (they ride keep-quant) and
all three are OFF under `VT_CPU_REF`.

**Binding arm = `dgx.casa` (GB10, aarch64, 20 cores), idle**, whole series under
one `flock $HOME/gpu.lock`, `git archive` transfer, goldens md5-verified
(`2965ef5772b556d3f3f86fedf4221b2f`) before/after. Model
`Qwen3.5-2B-UD-Q8_K_XL.gguf`. Same binary, 3 reps, medians. llama.cpp
`237ad9b96` fresh on the same idle host: pp128 180.14±2.78, tg32 25.37±0.81,
peak RSS 2,934,060 KB = 2.798 GiB.

| arm | TTFT (ms) | TPOT (ms) | peak RSS (KB) |
|---|---|---|---|
| BEFORE - L4 copy (`VT_GGUF_MMAP=0 VT_GGUF_SHARE_TIED_HEAD=0`) | 1,590.4 | 41.77 | 6,712,464 |
| **AFTER - L5 production default** | **1,656.1** | **41.79** | **4,072,720** |
| oracle - `VT_CPU_REF=1` | 2,952.6 | 92.26 | 7,788,300 |

Ratios AFTER: prefill 180.14 / (128/1.656) = 2.33× behind; decode 25.37 /
(1000/41.79) = 1.06× behind (≈ parity); **peak RSS 3.884 / 2.798 = 1.388× worse**
(was 2.29×). Per-lever RSS isolation on the same host: copy-only 6.401 GiB → +
tied-share 5.455 → + mmap borrow 4.457 → + page-release **3.884**. Output token
md5 `d235db12f2cd304007530286a1755c95` identical across BEFORE / AFTER / ORACLE
- the same golden the two prior CPU checkpoints recorded, unchanged by mmap or
head-sharing. **Contention caveat:** the first attempt showed the AFTER arm 40×
slower with loadavg 13.5 (a co-tenant agent's CPU build; `flock` guards GPU
only); re-run on a genuinely idle box (loadavg gate in the recipe) all arms are
~42 ms and these numbers reproduce. CUDA regression set standalone (production
CUTLASS+FA2 build): 35B 315/315, Qwen3-Coder 6/6, Qwen3-dense 16/16, OPT 6/6,
DeepSeek-V2 8/8, GGUF engine 28/28 all PASS; the 27B gate selects the NVFP4
EMULATION path on this box's CUDA-13.0/cutlass-4.5.0 runtime (`got == want_emu`)
BYTE-IDENTICALLY with and without L5 (base `18094ee` reproduces the same tokens),
a pre-existing environment condition, not an L5 regression.

## CPU vs llama.cpp - keep-f16 residency (L6): premise REFUTED, `ACCEPTED` / binding (2026-07-23)

**SUPERSEDED by the L7 entry above (2026-07-23):** L6's "the residual is engine
workspace" conclusion was itself wrong. L7 profiled the pools (DevicePool 20 MiB,
KV 115 MiB - both ≤ llama.cpp) and found the residual was a q8_0 repack-source
double-count; releasing it + prefaulting flips keep-f16 default-ON and closes the
RSS gap to 1.01×. This entry is retained for the L6 measurement of record.

`BENCH-CPU-LLAMA` / `QUANT-GGUF-KEEPQ-LOADER` **L6** (`CLAIM-QUANT-GGUF-KEEPF16-L6-1`).
Full result: [keep-quant loader leaf](../.agents/specs/gguf-keep-quant-loader.md)
§ L6 result. **This entry records a NEGATIVE / refutation (why keep-f16 shipped
DEFAULT OFF at L6, later reversed by L7).**

**keep-f16 (keep the file's F16 weights resident as F16 and compute on them, like
llama.cpp) is RSS-NEUTRAL (3.884 → 3.832 GiB, −52 MB) and regresses prefill. It
does NOT close the CPU RSS gap.** The hypothesis (L4/L5) that the residual gap was
the f16→bf16 expansion is DISPROVEN: L5's read-once page-release had already
dropped the f16 FILE pages, so keeping f16 resident merely swaps an anonymous
bf16 buffer for an equal-size file-backed f16 page set.

**Binding arm = `dgx.casa` (GB10, aarch64, 20 cores), idle**, whole series one
`flock $HOME/gpu.lock`, same binary, 3 reps, medians, `Qwen3.5-2B-UD-Q8_K_XL.gguf`
(176 F32 + 56 F16 + 103 Q8_0). llama.cpp `237ad9b96` fresh on the same host:
pp128 177.5±1.5, tg32 24.3-25.2, peak RSS 2,934,316 KB = 2.798 GiB.

| arm | TTFT (ms) | TPOT (ms) | peak RSS (KB) | token md5 |
|---|---|---|---|---|
| BEFORE - L5 state (`VT_GGUF_KEEP_F16=0`) | 577 | 40.7 | 4,072,976 | `d235db1…` |
| keep-f16 (`VT_GGUF_KEEP_F16=1`) | ~1000 | 41.0 | 4,017,668 | `d235db1…` |
| oracle (`VT_CPU_REF=1`) | 2490 | 95 | 7,788,404 | `d235db1…` |

**Peak-RSS attribution (`/proc/PID/smaps_rollup` at peak, output-len 400):**

| arm | peak RSS | file-backed | anonymous |
|---|---|---|---|
| keep-f16 | 3.834 GiB | **2.634 GiB** (≈ llama.cpp's 2.68 file) | 1.200 GiB |
| bf16 (L5) | 3.885 GiB | 0.961 GiB | 2.923 GiB |

keep-f16 moves ~1.8 GiB from anonymous bf16 buffers to file-backed f16 mmap pages,
bringing WEIGHT residency to llama.cpp parity (file-backed 2.63 ≈ 2.68 GiB) - but
total RSS is unchanged because the f16 mass is 2 bytes/elem either way. **The
remaining ~1.08 GiB gap vs llama.cpp is the engine's ANONYMOUS activation/KV
workspace** (keep-f16 anon 1.20 vs llama-bench's ~0.12), which no loader residency
change can touch - the real, separate CPU RSS lever. keep-f16 also REGRESSES
prefill: borrowing f16 out of the mmap moves its first-touch faults into the timed
prefill, TTFT 577 → ~1000 ms (prefill 222 → 128 tok/s, from **1.25× AHEAD** of
llama.cpp to **0.72× BEHIND**); decode TPOT stays at parity (~41 ms). **Output
token md5 `d235db12f2cd304007530286a1755c95` is byte-identical across all three
arms** - the same golden the prior CPU checkpoints recorded; keep-f16 computes on
native f16 (closer to the file's values than the lossy bf16 re-round) with zero
greedy movement. Ships DEFAULT OFF; retained as an opt-in llama.cpp-faithful
residency (multi-process weight sharing is its only real, unmeasured benefit).

**Repro:** the L4/G4 recipe with a third A/B axis `VT_GGUF_KEEP_F16 ∈ {unset,1}`;
attribution polls `/proc/PID/smaps_rollup` `Anonymous:`/`Rss:` at peak.
Regressions (CUDA build, standalone, default keep-f16 off = inert): 27B **235/235**,
35B **315/315**, GGUF engine **28/28** (16/16 both APEX files) - UNCHANGED.
`test_gguf_keep_quant` **35/35** (7 new L6 cases) on x86 AND aarch64.

## CPU vs llama.cpp - elementwise GEMM vectorized, `ACCEPTED` / binding (2026-07-22)

`BENCH-CPU-LLAMA` / `BACKEND-GATE-CPU-LLAMACPP` / `KERNEL-GEMM-CPU-ELEM`
**E1-E4** (this checkpoint) on top of `QUANT-GGUF-CIQ-GEMM` **G4**.
Full result: [elementwise CPU GEMM](../.agents/specs/cpu-elementwise-gemm.md);
the quant-routing step under it: [CIQ G4](../.agents/specs/gguf-compute-in-quant-gemm.md);
the attribution that selected both levers:
[floor re-measurement](../.agents/specs/cpu-llamacpp-floor-remeasure-2026-07-22.md).

**Decode is now AT PARITY with llama.cpp (1.03×, inside its own run spread).
Prefill is 2.34× behind. Peak RSS was 2.29× worse at THIS checkpoint; the
keep-quant loader L5 (section above, 2026-07-23) has since taken it to 1.39×.**

| Axis | llama.cpp `237ad9b96` | ours, before | **ours, now** | **ratio now** | ratio before |
|---|---|---|---|---|---|
| Prefill (pp128 vs 128/TTFT) | 173.28 ± 1.75 t/s | 21.67 t/s | **73.97 t/s** | **2.34× behind** | 8.00× |
| Decode (tg32 vs 1000/TPOT) | 24.52 ± 0.45 t/s (isolated 24.54 ± 0.79) | 7.649 t/s | **23.79 t/s** | **1.03× behind** (3.1 %; 1.032× vs isolated) | 3.21× |
| Peak RSS | 2,934,136 KB = **2.798 GiB** | 6.401 GiB | **6.401 GiB** (→ 3.884 at L5, above) | **2.29× worse** (→ 1.39× at L5) | 2.29× |

**Same-binary A/B gains: prefill 3.41×, decode 3.11×, peak RSS unchanged - with
output tokens BYTE-IDENTICAL** across the BEFORE arm, the production default and
the `VT_CPU_REF=1` oracle (one md5, `d235db12f2cd304007530286a1755c95` - the
same md5 the previous CPU checkpoint recorded, so the CPU token stream has been
stable across two consecutive kernel rewrites).

**Binding arm = `dgx.casa` (GB10, aarch64, 20 cores), idle** (no co-tenant CPU
process, `nvidia-smi` reporting no compute apps), whole build+measure series
under one `flock $HOME/gpu.lock`, source transferred by `git archive`, goldens
md5-verified (475 files, `2965ef5772b556d3f3f86fedf4221b2f`) before and after.
Model `Qwen3.5-2B-UD-Q8_K_XL.gguf` (`qwen35` dense, 1.94 B, 2.68 GiB; 103
`q8_0` + 56 `f16` + 176 `f32` tensors). Keep-quant is ON in every arm, so this
A/B isolates the ELEMENTWISE kernel. Three arms of the SAME binary, 3 reps,
medians:

| arm | TTFT (ms) | TPOT (ms) | peak RSS (KB) |
|---|---|---|---|
| BEFORE - `VT_CPU_MATMUL_TIER=ref` (the historical scalar kernel) | 5,907.66 | 130.74 | 6,712,412 |
| **AFTER - production default** | **1,730.38** | **42.03** | **6,712,340** |
| oracle - `VT_CPU_REF=1` | 2,977.07 | 92.18 | 7,788,252 |

AFTER rep spread: TTFT 0.70 %, TPOT 2.3 %. The BEFORE arm reproduces the
previous checkpoint's published AFTER arm (5,970.21 ms / 130.72 ms /
6,712,368 KB) to 1.1 % / 0.02 % / 28 KB, which is what makes the A/B valid.
The llama.cpp denominator reproduced across three series on the box (pp128
173.28 ± 1.75, 171.27 ± 3.40, 173.68 ± 2.22); two `tg32` legs that returned
17.40 ± 9.70 and 3.95 ± 1.83 under page-cache pressure from our own 6.4 GiB
arms are DISCARDED as contaminated rather than averaged in.

### What changed, and the bit-exactness claim

The elementwise CPU GEMM had two defects, both in the K loop: a per-ELEMENT
dtype switch (`LoadF32`) for both operands, and a SINGLE serial f32
accumulator, i.e. one multiply-accumulate per FP-add latency. The fix
specializes the dtypes out of the loop and runs 16 independent accumulator
chains, vectorized per architecture (AArch64 NEON; x86-64 SSE2 with F16C
probed), with M-blocking so several activation rows share one weight load.

The SIMD is ported from llama.cpp's `ggml_vec_dot_bf16`/`_f16`
(`ggml/src/ggml-cpu/vec.cpp:139,264`) with one deliberate deviation: ggml
vectorizes ALONG K and horizontally reduces, which reassociates the sum, while
we vectorize ACROSS OUTPUT COLUMNS so each output keeps its sequential
reduction. The result is **byte-identical to the historical kernel** - gated by
`memcmp` (not NMSE) over every dtype pair, both orientations, ragged shapes,
thread counts 1/2/4/8, and the exhaustive 65,536-pattern f16/bf16 widening
domain including every inf/NaN. No golden was regenerated and no tolerance was
widened.

### Op-level GFLOP/s (bf16 × bf16, 20 threads, best of 3, aarch64)

| shape (M×N×K) | before | portable tier | **NEON tier** |
|---|---|---|---|
| decode qkv 1×3072×2048 | 18.39 | 35.17 | **81.44** |
| decode down 1×2048×6144 | 18.20 | 34.75 | **68.80** |
| decode lm_head 1×248320×2048 | 24.40 | 52.20 | **133.99** |
| prefill qkv 128×3072×2048 | 24.29 | 51.46 | **351.11** |
| prefill gate_up 128×12288×2048 | 24.28 | 51.57 | **349.98** |

The portable tier alone is ~2.1× - that is the multi-accumulator fix, with no
intrinsics, and every backend inherits it. At prefill shapes the elementwise
GEMM now reaches 347-351 GFLOP/s against the quant tier-0's 388-417, so a mixed
GGUF no longer has a slow half.

### Next benchmark checkpoint - a FRESH profile, not a lever

The most useful result here is a negative one: M-blocking raised prefill
op-level throughput 1.63× (216 → 351 GFLOP/s) and moved end-to-end TTFT by
**0.0 %**. Prefill is therefore no longer bound by this kernel, and the
`kMatmul = 95.37 % of wall time` attribution that ranked the whole CPU plan is
STALE. The next checkpoint is an op-dispatch profile of the CURRENT binary at
the prefill operating point; no further CPU lever should be started before it.
Standing facts for that ranking: RSS (loader row **L5** - mmap-in-place
residency + tied-head sharing, ≈1.9 GiB of the 3.6 GiB gap) is now the largest
single deficit; decode needs no further kernel work; the quant SIMD/repack tiers
(**G5**-**G7**) are candidates for the prefill residual but must be confirmed
against the new profile, not against the old one.

`BACKEND-GATE-CPU-LLAMACPP` closes only when decode t/s, prefill t/s AND peak
RSS all match or beat llama.cpp on the same file. Decode now does; the other two
do not.

### Superseded, retained as disposition

- **B4 (2026-07-10)**: 54-75× decode / ≈1,480× prefill / 2.7× RSS. Validated as
  sound (the current binary at 1 thread on B4's own box reproduces its
  TTFT/TPOT to 4.8 %/5.5 %), then superseded three times.
- **Floor re-measurement (2026-07-22, morning)**: 11.6× / 33.5× / 2.65×. Its
  movement over B4 was entirely the **threadpool** (same-binary 1-vs-20-thread
  A/B: prefill 12.47×, decode 8.05×, RSS 1.000×) - which remains
  `QUANT-GGUF-CPU-THREADPOOL` **W4**'s owed reproduction and still **MISSES**
  W4's ≥10× decode bar at 8.05×, so W4 stays open.
- **CIQ G4 (2026-07-22, afternoon)**: routed the block-quantized weights onto
  the quant GEMM - 3.38× decode / 8.20× prefill / 2.29× RSS behind. Its
  projected 9-17× did not hold because only 1.062 GiB of that file's weight
  bytes are `q8_0` while 1.615 GiB are `f16` (including the 970 MiB tied
  `token_embd`/`lm_head`), so ~60 % of the mass could not take the quant path.
  This checkpoint is what fixed that 60 %.
- The **x86 dev-box arm remains `VOID`** for binding (co-tenant load: llama.cpp
  ±11.8 %/±24 %, a 1.9× outlier inside our own series). It shipped an SSE2
  4-wide tier here, so real x86 headroom exists but cannot be gated until that
  box is exclusively idle or another x86 host appears.
- `QUANT-GGUF-KEEPQ-LOADER` gate 3 stays **MEASURED / NOT MET** (work row
  **L4**): 6.401 GiB against a ≈3.1 GiB bar.

**Repro (binding arm).**

```sh
R=$HOME/work/bench-cpu-llama; M=$R/models/Qwen3.5-2B-UD-Q8_K_XL.gguf
LB=$R/llamacpp/build/bin       # llama.cpp 237ad9b96, Release, GGML_CUDA=OFF, GGML_NATIVE=ON
# ours: Release, -DVLLM_CPP_CUDA=OFF   (transferred with `git archive`, never rsync)
B=./build-cpu/examples/vllm-bench
flock $HOME/gpu.lock sh -c '
  uptime; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  LD_LIBRARY_PATH='"$LB"' '"$LB"'/llama-bench -m '"$M"' -p 512,128 -n 128,32 -t 20 -r 5 -ngl 0
  LD_LIBRARY_PATH='"$LB"' /usr/bin/time -v '"$LB"'/llama-bench -m '"$M"' -p 0 -n 32 -t 20 -r 3 -ngl 0
  # three arms of the SAME binary; arm 2 is the production default
  for arm in "VT_CPU_MATMUL_TIER=ref" "" "VT_CPU_REF=1"; do
    for rep in 1 2 3; do
      env $arm VLLM_CPP_CPU_THREADS=20 /usr/bin/time -v '"$B"' --model '"$M"' \
        --num-prompts 1 --input-len 128 --output-len 32 --concurrency 1 \
        --seed 0 --temperature 0 --output-token-ids /tmp/ids.$arm.$rep.json
    done
  done'
# All nine token files MUST be md5 d235db12f2cd304007530286a1755c95.
```

### Architecture-support checklist + its CI checker (2026-07-23, `CLAIM-DOCS-MODEL-CHECKLIST`) - NOT APPLICABLE

**Benchmark disposition: NOT APPLICABLE - docs/policy/CI only, no `src/`/`include/`
touched, no kernel or engine behavior changed.** Added an at-a-glance
architecture-support checklist to the top of `.agents/model-matrix.md` plus
`scripts/check-model-checklist.py` (and its mutation test) that keeps the
checklist marks and rollup counts in lockstep with the detailed row lifecycle
states. All model gates are inert by construction; regression set unaffected.
