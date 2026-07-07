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
- Distinguish the few vLLM edges that are **build-specific** (Inductor epilogue
  fusion, DeepGEMM, flashinfer-cutlass exact accumulation) — which eager-C++
  cannot replicate 1:1 — from the many that are plain code/config diffs we CAN.
- Compare at the **right operating point**: "large concurrency" is the gate; a
  single-wave (np=conc) measurement hides sustained-load effects, and a
  memory-tight model may need a different concurrency than the 35B.
