# ROCm Qwen3.5-0.8B forward-divergence fix — the RED gate's target

## The problem (oracle-measured)

The M4 gate (`test_qwen35_paged_engine`, #559) FAILS 13/16 prompts against the
pinned vLLM-ROCm oracle on gfx1100, with **first-token divergences on 6 prompts**
(gaps 0.375–1.188 nats, over the 0.5-nat band) and max gap 14.125 nats deep in a
diverged continuation. This is a REAL forward divergence, not a bf16 near-tie.

## The evidence (all measurements on gfx1100, GPU-locked)

1. **The split is ROCm-kernel-specific, not model math.** For prompt
   `The capital of France is` (p0):
   - pinned oracle (vLLM `555967922`, enforce_eager): argmax **11751 " Paris"** (-1.9957), token 25 ":" at -2.8082 (#4)
   - **our CPU backend: matches the oracle exactly** (` Paris.\nThe capital of France is`)
   - our ROCm backend: argmax **25 ":"** (-1.6535), 11751 at -3.0910 (#9)
   The candidate SETS are the same with similar spread; the rankings shift ~1.1
   nats in opposite directions — a systematic numerical offset in some block
   output, not a distribution reshape.
2. **Backend proof is green**: all 15 dispatched GDN/full-attn ops run natively
   on device 5 with 0 declines (kPagedAttention 1,536, kGdnDecode 4,320) — the
   divergence is numerical, not dispatch.
3. **Not the ROCm paged-attention kernel**: `VT_ROCM_ATTN_CPU_REF=1` (the CPU-ref
   attention path) produces the SAME divergent tokens.
4. **GDN ops were cleared in isolation**: the merged cross-device cases
   (tests/vt/test_backend_cross_device.cpp) pass at real dims vs the CPU oracle.
5. **The 0.6B dense gate passes 16/16** with the same ROCm attention/GEMM/RMSNorm
   kernels — so the diverging op is exercised by 0.8B's GDN hybrid path but not
   (or differently) by dense 0.6B.
6. Prior characterization (issue #41, op-level): attention-block localized
   (layer-3 attn block 20% gated / 4.4% raw mismatch); GDN ops + op 67 + op 66 +
   paged-attention + GEMM all cleared at real dims; conclusion then was
   "bf16-softmax amplification" — but (3) refutes the paged-attention softmax as
   the sole source, so the amplification must be fed by an earlier drift.

## The plan (W-plan)

- **W1 — instrument.** Add `VT_DUMP_ACT`-style per-layer hidden-state dumps to
  `qwen3_5.cpp` (the deepseek_v4.cpp precedent), plus a scratch host comparator
  (CPU vs ROCm, per-layer max-abs/rel delta at the residual stream) in
  `/tmp/vllm-diag` (never committed).
- **W2 — localize.** Find the first layer+op where the ROCm hidden state drifts
  past ~1e-3 rel vs CPU on the p0 prefill. Suspects in order: the GDN
  prefill/decode recurrence IN CONTEXT (state carry across the chunk boundary),
  the fused preamble `kAttnQkNormRopeGate` at 0.8B geometry, the conv1d state
  handoff, the GDN output gated RMSNorm.
- **W3 — fix.** The minimal kernel fix at the localized op (portable-math
  corrections first; e.g. the reduction-order or accumulation-dtype issue the
  in-isolation tests couldn't see).
- **W4 — gate GREEN.** `test_qwen35_paged_engine` 16/16 (strict + band), the
  0.6B gate re-run green, focused + full gates, fresh review, operator gate.

## RESULT (W1–W4 complete — fix verified; the gate lands GREEN stacked on top)

**Stack shape (post-review):** this branch carries ONLY the kernel fix + its
teeth + the debug instrumentation. The M4 0.8B gate with the GREEN (fixed-engine,
oracle-re-derived) goldens lands in the stacked sibling commit on
`row/ROCM-M4-GDN-GATE` — so every commit in the stack is individually green on
the gate host, and the gate never exists in a state whose goldens its own tree
cannot reproduce.

**Root cause:** `AttnQkNormRopeGateKernelRocm` (src/vt/rocm/rocm_gdn_fused.hip)
dispatched its template on the SOURCE dtype (`qgate.dtype`) instead of the
OUTPUT dtype like the CUDA lane (`LaunchAttnPreambleOut` switches on
`q_out.dtype`). The 0.8B bf16 model runs bf16 QKV-projection output + f32
q/k/gate outs (the f32-attention path — no FA-2 on ROCm), which the src-keyed
dispatch silently mis-launched as all-bf16: the kernel wrote bf16 bits through
the f32 out pointers. The cross-device test covered only f32-src, so the bug
class was invisible in isolation; the in-context logits comparison (W2) showed
`fa0_q` rms-rel 1.196 with `fa0_qkv`/`fa0_gate` clean, localizing it to the
preamble's q/k path.

**Fix:** dispatch on the output dtype (bf16 src + f32 out is now a first-class
combo); VT_CHECK bf16-out requires bf16-src.

**Teeth:** the cross-device AttnQkNormRopeGate case gains the bf16-src/f32-out
combo at the real 0.8B dims (Hq=8, Hkv=2, Dh=256, rot=64) — FAILS with the fix
reverted (mutation-proven), passes with it.

**Gates (gfx1100, flock):**
- `test_qwen35_paged_engine` (the RED anchor): **16/16 PASS — 15/16 strict
  token-exact vs the pinned oracle, 1/16 near-tie band, max gap 0.125 nats,
  0 forward-divergent** (was 3/16, 13 divergent, max 14.125 nats)
- `test_qwen3_paged_engine` (0.6B, regression): 16/16 PASS unchanged
- `test_backend_cross_device`: 19/19, 346 assertions
- e2e: `The capital of France is` → ` Paris.\nThe capital of France is`,
  matching the pinned oracle AND our CPU backend token-for-token

**Also fixed (pre-existing main bug, blocked the full HIP build):**
same-line double `CAPTURE` in test_qwen3_5_gdn_spec_routing.cpp:357 (doctest
redefinition that clang/hipcc rejects).

## Boundaries

- The RED gate stays RED until W4; no gate-weakening (no band-widening, no
  goldens recapture against a "fixed" engine without the oracle re-derivation).
- ROCm-only files + the additive dump hook; no shared-path behavior change.
- Rollback: the dump hook is env-gated (off by default).
