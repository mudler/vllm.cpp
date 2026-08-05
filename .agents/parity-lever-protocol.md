# Parity-lever protocol — scan → re-adapt → find levers (never accept a "ceiling")

Source, oracle, dependency and profiler paths come from the untracked
`developer-preferences.md`. The examples recorded for Ettore's infrastructure
are not implied capabilities or authority in another workspace.

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

   Each agent reads BOTH the pinned vLLM (`${VLLM_SOURCE}` @ the parity
   pin) AND our source, and reports, with `file:line` cited on BOTH sides, **what
   vLLM does differently that makes it faster** and the concrete fix in our code.

   **When the developer preferences and current task permit parallel agents,
   use MANY sub-agents and attack from a VARIETY of angles.** Otherwise run the
   same lenses serially; do not drop coverage merely because the execution is
   single-agent. Fan out broadly and diversely when authorized, e.g.:
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

- **flashinfer** (`${DEPENDENCY_SOURCE}/flashinfer/`):
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
1. **nsys BOTH** the vLLM oracle (`${VLLM_ORACLE}`, graphed = production) AND our
   engine on the IDENTICAL workload. For our graphed engine the command is
   `nsys profile -t cuda --cuda-graph-trace=node -o OUT --stats=false <cmd>`
   then `nsys stats --force-export=true --report cuda_gpu_kern_sum OUT.nsys-rep`.
   CUDA-driver ≥11.7 otherwise defaults to whole-graph activity and omits child
   kernels. Query the export and require graph-node kernel rows whenever graph
   launches exist; graph-level timing alone is not an actual kernel list.
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

Code-only comparison is necessary but NOT sufficient. If you have not PROFILED vLLM on the
workload, you do not yet know where the gap is — you are guessing.

## The STRUCTURAL lens — same kernel, different throughput (scan → then VERIFY per-shape)

The trace can show BOTH engines running the IDENTICAL kernel yet ours *appearing* slower at
the same clock. A real bandwidth/compute wall caps BOTH engines EQUALLY, so an apparent
same-kernel throughput split is a RED FLAG worth investigating and forbids declaring a
"ceiling" — but it is a HYPOTHESIS to confirm, NOT a proven structural deficit. If the gap
is real it usually lives in how the engine DRIVES execution AROUND the kernel; audit the
context, not the kernel:

- **Stream count + graph structure** — how many CUDA streams does decode run on; one full
  cudagraph on one stream, or a per-layer stream explosion?
- **Overlap / scheduling policy** — is DRAM-heavy work co-scheduled with a bandwidth-bound
  kernel on a saturated bus (a possible net loss), or is overlap gated to compute-slack work
  when the primary leaves idle SMs? (vLLM gates overlap OFF above token thresholds bs=1 never
  trips.)
- **Allocator-pool placement** — per-stream pools can strand scratch and shift KV/weight
  physical placement.
- **Per-step sync vs async** — a per-step device sync that idles the GPU vs async
  step-N/step-(N+1) overlap.
- **Kernel DECOMPOSITION** — does the reference split the SAME logical op into a DIFFERENT
  set of kernels (e.g. offloading part of a quantized GEMV to tensor-core GEMMs)? Same op ≠
  same kernels.

**MANDATORY lane #1 — read THE REFERENCE'S OWN rationale (vLLM AND non-vLLM).** Box-
independent, so run it ALONGSIDE the trace, not after. Scan the reference's own performance
REASONING — vLLM for most models, EQUALLY any non-vLLM reference the model is gated against
(ds4/DwarfStar, SGLang, llama.cpp): source CODE COMMENTS + kernel structure, ENV-VAR/design
docs (`vllm/envs.py`, `docs/design/`), and (where they exist) issues/PRs. It often documents
the exact structural pitfall and its magnitude.

**MANDATORY lane #2 — the scan is a HYPOTHESIS GENERATOR; per-shape MEASUREMENT is the
arbiter.** This is the hard rule the two 2026-08-04 cases teach. Confirm both the ANOMALY and
the proposed FIX with PER-SHAPE / PER-KERNEL timing before implementing, and DISTRUST any
aggregate/derived throughput number (bytes÷time) until the byte total is verified and
cross-checked per-kernel:

- **POSITIVE (ds4 / DeepSeek):** reading ds4's OWN kernel source (`ds4_cuda.cu`) revealed it
  OFFLOADS a chunk of projections to f16 tensor-core GEMMs (`matmul_f16_pair` /
  `cutlass_80_wmma`) — a real DECOMPOSITION difference, MEASUREMENT-CONFIRMED (ds4 does ~1.8×
  less GPU work) → a beat-path our per-launch-parity int8-GEMV scans had all missed.
- **CAUTIONARY (vLLM / Laguna):** a "20% in-engine bandwidth deficit" (~131 vs 167 GB/s) that
  triggered a whole structural hunt was a MEASUREMENT ARTIFACT — an aggregate bytes÷time
  whose byte total was UNDERCOUNTED (it missed the 64-head sliding layers' bigger 10240-qkv /
  K=8192 o_proj); the clean PER-SHAPE table showed our projections at per-call PARITY with
  vLLM (~172 GB/s ≈ the isolated microbench). And vLLM's own "avoid an explosion of streams
  for every layer" comment generated a plausible fix (collapse our 41 streams) that per-shape
  measurement then REFUTED: the projections were 0.41% concurrent (nothing to de-contend) and
  the more-serial variant measured ~2% SLOWER. The scan surfaced the idea; measurement killed
  it — do NOT implement a scan-hypothesis against your own trace.

RED FLAGS: declaring "DRAM/hardware wall", "weight-stream floor", "diffuse inefficiency", or
"irreducible" (a) from an AGGREGATE metric you have not confirmed per-shape, or (b) without
scanning the reference's rationale — that is a structural miss you have not found yet.
EQUALLY a red flag: implementing a scan-derived structural fix WITHOUT per-shape confirmation
that its premise holds in your measured trace (the Laguna 41-stream fix would have shipped a
~2%-slower regression).

RED FLAGS that you skipped this lane: concluding "DRAM/hardware wall", "Q8_0/weight-stream
floor", "diffuse in-engine inefficiency", or "irreducible" while your OWN trace shows the
reference achieving a higher number (or doing LESS work) on the same op — that is a
structural miss you have not found yet, never a ceiling.

**MANDATORY lane #3 — INVOCATION parity (same kernel NAME ≠ same resolved kernel).** The
trace can name the IDENTICAL kernel on both engines yet run a DIFFERENT resolved TEMPLATE —
the cost that hid on Laguna for weeks: our bf16 M=1 decode GEMVs ran cuBLAS
`gemvx<bf16,FLOAT>` (f32 output, 204 us o_proj) where vLLM runs `gemvx<bf16,bf16>` (bf16
output, 139 us) — the OUTPUT dtype SELECTS the gemvx template, and `requestedAlgoCount=1`
skips the algo search. Before claiming any GEMM/GEMV at-parity, verify vLLM's ACTUAL call on
FOUR axes: (1) **output/C dtype** — it SELECTS the template, so an API-name match can still
be a slower `<bf16,FLOAT>` template; (2) **compute + scale type**; (3) **entry point + algo
policy** — `cublasGemmEx` default-algo vs `cublasLtMatmul` requestedAlgoCount/heuristic; (4)
**the resolved template dtypes** read off the SAME tool's trace. HARD RULE: a CROSS-TOOL
comparison (our nsys vs vLLM's torch-profiler) can NEVER establish invocation parity — it
cannot compare in-graph stall% or template dtypes across two different profilers; a SAME-tool
trace where entry point AND resolved template BOTH match is required. This is CI-gated on both
sides so the regression cannot reland silently: the op-contract side (the C/D layout stays
dtype-faithful to the caller via `out_type`, requestedAlgoCount is the named
`kGemvHeuristicAlgos`) by `scripts/check-gemv-invocation-consistency.py` over
`src/vt/cuda/cuda_matmul.cu`, and the CALLER side (an f32-resident decode stream that buys the
slow template model-wide) by `check-runner-routing-consistency.py` invariant (c).

### MANDATORY during autonomous porting: profile vLLM's ACTUAL kernels, port 1:1 what it runs

**"At-parity" inferred from SOURCE is NOT parity — you must MEASURE it.** (Proven
2026-07-08, per user: *"this kind of profiling is what we have to put as a rule in our
protocol to follow during porting autonomously"*; *"if vllm has these numbers, I don't see
why we can't by porting 1:1 every component to the vllm execution path."*) A whole session
was spent concluding our fp4/fp8 GEMM + attention were "at-parity" by reading that both use
cutlass/nvjet — while the measured TOTAL stayed 0.82×. **The tell: a total-vs-parts
CONTRADICTION.** If every component reads "at-parity" from source but the end-to-end number
is not, then a component you LABELED at-parity is NOT — and only vLLM's actual per-kernel
profile finds it. When it landed, the 0.82× dissolved into a concrete finite port list:
vLLM runs GDN `chunk_..._h_blockdim64` (V-split) + `merge_16x16_to_64x64_inverse` (blocked-
inverse) + heavily-fused `triton_..._add_rms_norm[_scaled_fp4_quant]` chains + a specific
`flash_fwd_splitkv<256,64,64,4>` + the `nvjet...TST...TNNN` fp8 variant — none of which the
source scan surfaced as gaps. **The gap is never a ceiling; it is the specific kernels vLLM
runs that we haven't matched 1:1.**

**THE WORKING RECIPE (nsys BREAKS vLLM's V1 EngineCore on GB10 — use vLLM's own torch
profiler):**
1. OURS: `nsys profile -t cuda ... ; nsys stats --report cuda_gpu_kern_sum` (works for us).
2. vLLM: nsys fails (EngineCore init). Use the LLM-API torch profiler instead — a Python
   script (run with the `${VLLM_ORACLE}` Python) with these NON-obvious requirements:
   - Wrap the body in `if __name__ == '__main__':` — vLLM uses `spawn` (CUDA pre-init) which
     re-imports the module; an unguarded top-level `LLM(...)` double-inits and crashes.
   - `LLM(model=..., profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir="/tmp/vprof3"))`
     (`from vllm.config.profiler import ProfilerConfig`) — v0.24.0's API. The old
     `VLLM_TORCH_PROFILER_DIR` env is UNRECOGNIZED (warns "Unknown env var") and `start_profile`
     then no-ops → fails.
   - `warmup generate → llm.start_profile() → generate ×3 (graphed steady state) → llm.stop_profile()`.
   - Fit GPU mem: a crashed run can leak a ~65 GiB CUDA context with NO owning process (only a
     reset/reboot clears it); drop `gpu_memory_utilization` (e.g. 0.35) so `request_memory`
     doesn't ValueError on startup.
   - Parse the `*.pt.trace.json.gz` chrome trace: aggregate `traceEvents` with `cat=='kernel'`
     by name, sum `dur`, rank — that IS vLLM's per-kernel GPU breakdown.
3. DIFF vLLM's kernel list vs OURS → each divergence (vLLM runs X, we run Y / vLLM fuses N→1)
   is a port target. Port 1:1 from the dep source the trace names (FLA/flashinfer/cutlass),
   cite it, realize PORTABLY (see discipline.md: DSL source is a reference, never a compile-
   target). Re-measure. Repeat until the kernel lists match.

**Converge on vLLM's execution path — CLEAN UP superseded/dormant kernels.** When the
trace shows we REINVENTED a kernel the code-reading hid (e.g. hand-WMMA GDN chunk vs
vLLM's GDN-as-cutlass-GEMM; hand-WMMA attention vs vLLM's flash_fwd; the sm_80 in_proj vs
nvjet), mirror vLLM's structure, make it the DEFAULT (A/B behind a toggle → flip default
once it wins), then clean up the superseded kernel. Do NOT leave two competing default
CUDA paths — that fights the "structured so every upstream vLLM PR ports mechanically"
premise and rots. This is `port-don't-reinvent` (discipline.md) enforced by the trace.

**DEFAULT TO DEMOTE, NOT DELETE — check cross-backend value FIRST.** Before removing any
superseded kernel, verify it has NO ongoing value for: the multi-backend roadmap
(**CPU now; Vulkan / hipBLAS/ROCm / Metal / Intel XPU later** — `backends.md`), the
correctness oracle/reference, or a capability fallback (non-cutlass builds, GPUs without
the fast path). The vt:: OP INTERFACE is the portable contract and ALWAYS stays; the
CPU kernel and any GENERIC/vendor-neutral implementation are the backbone other backends
port FROM — KEEP them. Only DELETE a kernel that is a PURE CUDA-vendor-specific
reinvention with no reference/fallback/cross-backend use (e.g. a hand-WMMA path fully
superseded by the vLLM-mirrored cutlass/cuBLASLt kernel AND not needed by any other
backend). When in doubt, GATE it as a documented fallback rather than delete — a demoted,
labeled fallback costs little; a deleted portability asset is expensive to rebuild. (We
reimplement the same algorithm/STRUCTURE vLLM runs — we cannot byte-copy Triton/CuTe-DSL
into C++ — so "mirror the execution" = match what kernel/structure runs, then demote/
remove our now-redundant CUDA alternative, keeping every reference/portable path.)
