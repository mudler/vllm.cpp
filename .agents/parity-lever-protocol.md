# Parity-lever protocol — scan → re-adapt → find levers (never accept a "ceiling")

**Core axiom:** vllm.cpp is a 1:1 port — SAME architecture, SAME model, SAME GPU
as vLLM. Therefore, **if vLLM achieves a throughput/quality number, we can too.**
Any apparent "ceiling", "diffuse per-op inefficiency", or "hand-kernel exhausted"
conclusion is a signal that there are SPECIFIC, findable implementation/config
differences — not a real limit. Do not stop at "near parity". Ground every
target, denominator, and A/B in vLLM (the oracle).

## The loop

1. **SCAN.** When a lever search stalls (or you're tempted to declare a ceiling),
   run a **dynamic Workflow** that fans out one agent per throughput-critical
   subsystem to deep-compare vLLM's HOT PATH against ours. Suggested subsystems
   (the hot path is what runs per-token/per-step at the gate):
   - forward dtype + cast flow (bf16 vs f32; where vLLM stays bf16 and we cast)
   - GDN / linear-attention kernels (FLA Triton chunk/recurrent vs our WMMA)
   - MoE fused kernel + routing (fused_moe, Marlin tile/config, align/combine)
   - attention kernels (FA-2 config, decode attn) + KV-cache layout/dtype
   - norm / RoPE / elementwise **fusion** (vLLM fuses; do we launch separately?)
   - CUDA graph coverage + torch.compile/Inductor + DeepGEMM/PDL
   - quant dispatch (the exact kernel vLLM selects per op on this GPU)
   - scheduler / batching / chunked-prefill / prefix-cache effectiveness
   - per-step engine/host overhead (input prep, sampling, async scheduling)

   Each agent reads BOTH the pinned vLLM (`/home/mudler/_git/vllm` @ the parity
   pin) AND our source, and reports, with `file:line` cited on BOTH sides, **what
   vLLM does differently that makes it faster** and the concrete fix in our code.

   **Use MANY sub-agents and attack from a VARIETY of angles — do not rely on
   one agent or one lens.** Fan out broadly and diversely, e.g.:
   - **By subsystem** (the list above) — one agent per hot-path area.
   - **By lens on the same area** — separate agents examining the same op through
     different perspectives: kernel wall-time, memory traffic / bandwidth,
     launch count, dtype/precision, tile/config values, algorithm/fusion, and
     occupancy. Each lens catches gaps the others miss.
   - **By evidence source** — some agents diff SOURCE (vLLM vs ours), others read
     a CLEAN nsys/ncu profile, others run a direct vLLM-vs-ours A/B microbench.
   - **Adversarial angle** — an agent whose job is to REFUTE "we already match
     vLLM here" and one asking "what is vLLM doing that we haven't even looked
     at?" (a completeness critic).
   Redundant/overlapping coverage is good — it's how blind spots get found.
   Cross-check the agents' findings against each other before ranking.

2. **VERIFY.** Adversarially confirm each candidate diff: is it real in both
   sources? Is it actually on the gate hot path (runs per token/step)? Does the
   estimated gain survive our measured constraints (e.g. if we're already 96%+
   GPU-busy, an "idle-hiding" lever is capped by the idle window)?

3. **RE-ADAPT + FIND LEVERS.** Rank surviving diffs by (est. gain ÷ effort).
   Drive the top lever (implement behind an A/B toggle, keep correctness — 16/16
   token-for-token + chunked `max|diff|=0`), then **re-measure vs vLLM on the
   identical workload**. Merge if it's a measured win (or faithful+neutral);
   record in the [parity-ledger](parity-ledger.md). Repeat until we match vLLM.

## What this has found (proof the loop works)

In one session the loop turned every "ceiling" into a concrete lever and took the
35B gate from ~0.33× to ~0.985× vs vLLM:

- logits-gather before lm_head (+54%), on-GPU sampling, bf16-GDN inputs + state,
  FlashInfer-style decode attention (+18.7%), GQA-fused decode (+9.1%).
- **After I declared the floor at 0.79×**, comparing against vLLM found: dense
  bf16→**fp8 cuBLASLt** (+6%), **prefill-attn vectorized staging** (+8.1%), GDN
  **tensor-core WY solve** + vectorized DeltaH/O staging (+4.5%), concurrency-
  aware **mnbt=8192** (+2.7%), **bf16 residual** matching vLLM's model dtype
  (+1.2%).
- The hot-path-diff workflow then found the biggest *structural* gap: the **27B
  dense decode path runs eager** (no CUDA graph) while the 35B is graphed — the
  reason the 27B lagged.

## Honest guardrails

- A real per-op vLLM-vs-ours comparison needs a **CLEAN nsys slice of OURS**, not
  `proportions × TPOT` (that inferred-numbers shortcut misled twice — it put our
  dense at 72ms when it was 15ms, and named two neutral levers).
- vLLM's baselines **drift** — always re-measure the vLLM denominator on the
  IDENTICAL workload (e.g. 35B 2768→3145, 27B 397→452 across re-measures).
- Compare at the **right operating point**: "large concurrency" is the gate; a
  single-wave (np=conc) measurement hides sustained-load effects, and a
  memory-tight model may need a different concurrency than the 35B.

## Verify the WHOLE chain (vLLM is orchestration; the kernels live in its deps)

**Do not scope "what vLLM does" to the vLLM repo.** vLLM dispatches to kernels in
its dependencies; a lever "absent from vLLM's `csrc/`" often lives one layer down.
Before calling anything "build-specific / unverifiable / out of reach", inspect the
chain and, for compiled/JIT code, DUMP the generated kernel:

- **flashinfer** (`~/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/`):
  the real fp4/fp8 GEMMs + fused kernels, many as **readable CuTe-DSL Python** — e.g.
  `cute_dsl/add_rmsnorm_fp4quant.py` (fused Add+RMSNorm+FP4-quant), `cute_dsl/
  rmsnorm_fp4quant.py`, `gemm/` (the fp4 GEMM + its tactic selection),
  `fused_moe/cute_dsl/blackwell_sm12x/` (sm_121-specific). CuTe-DSL is portable — read
  it and hand-write the equivalent CUDA. (This is how we found the rms+fp4 fusion we
  had WRONGLY declared nonexistent from vLLM's csrc alone — 2026-07-07.)
- **cuBLASLt kernel selection** (closed, but OBSERVABLE): `nsys` shows the exact
  kernel name (e.g. `cutlass_80_tensorop_*` sm_80 vs `nvjet_sm121_*`); probe
  `cublasLtMatmulAlgoGetHeuristic` to see what it picks for a given shape/dtype
  (e.g. bf16→f32 vs bf16→bf16 output) before assuming a reselect.
- **torch/Inductor**: run the real oracle with `TORCH_LOGS=output_code` /
  `TORCH_COMPILE_DEBUG=1` and READ the generated Triton — the fusions are a specific,
  finite set for these specific ops, not "arbitrary". Most are hand-portable once read.
- **cutlass / DeepGEMM**: vendored + in flashinfer; read the actual tile/config/
  scheduler the dep instantiates for our shapes and mirror it.

Only after reading the dep source AND (for JIT/compiled) dumping the generated kernel
may you conclude a lever is genuinely irreducible for eager-C++ — and say exactly why
(cite the dep file:line / the dumped kernel). The default assumption is PORTABLE.

- Distinguish the few genuinely irreducible edges (proven so by the above) from the
  many that are plain code/config/dtype/algorithm diffs — but prove it, don't assume
  it. The CLEAN nsys slice of OURS (above) tells you WHERE to look; the dep chain
  tells you WHAT vLLM actually runs there.

## Trace the EXECUTION, not just the code (nsys BOTH sides FIRST)

**Code-comparison finds buildable levers; execution-tracing finds what vLLM actually
runs. They DIVERGE, and the trace wins.** (Proven 2026-07-07: three exhaustive
code-only scans produced bf16/fusion levers that measured NEUTRAL, and concluded a
"ceiling" — then ONE `nsys` of the vLLM oracle revealed vLLM runs `flash::flash_fwd_kernel`
for attention, executes the GDN as **cutlass GEMMs** (no dedicated chunk kernel like our
WMMA), links **TensorRT-LLM** kernels, and gets `nvjet_sm121` where we get the sm_80
`cutlass_80` kernel — structural differences NONE of the code scans surfaced.)

WHY they diverge: vLLM's real kernels are resolved at RUNTIME, invisible to source —
cuBLASLt/cutlass heuristics, flashinfer autotune, torch.compile codegen, capability-probed
backend selection, and runtime-linked deps (TRT-LLM) the source never names.

THE METHOD — do this BEFORE and DURING any parity/throughput work:
1. **nsys BOTH** the vLLM oracle (`~/venvs/vllm-oracle`, graphed = production) AND our
   engine on the IDENTICAL workload: `nsys profile -t cuda -o OUT --stats=false <cmd>`
   then `nsys stats --force-export=true --report cuda_gpu_kern_sum OUT.nsys-rep`.
2. **Diff the actual GPU-kernel lists.** For each of OUR hot kernels, find what vLLM runs
   for the same op (by kernel name + count + time). The DIFFERENCES (vLLM runs kernel X,
   we run kernel Y; vLLM has N launches, we have 3N; vLLM's op is a fused Triton, ours is
   3 kernels) are the parity gaps — concrete, not inferred.
3. **THEN read source** for the divergent kernels — use the trace to know WHICH dep kernel
   to open (flash_fwd traits, the cutlass GEMM config nvjet picks, the TRT-LLM kernel) and
   mirror it. The trace tells you WHAT is faster; the source tells you HOW to port it.
4. Caveat: a graphed-vLLM nsys is warmup/JIT/capture-contaminated — kernel NAMES are
   reliable; for reliable %/time, capture STEADY-STATE (long run so warmup amortizes, or
   an nsys capture-range that excludes warmup).

Code-only comparison is necessary but NOT sufficient. If you have not nsys'd vLLM on the
workload, you do not yet know where the gap is — you are guessing.
