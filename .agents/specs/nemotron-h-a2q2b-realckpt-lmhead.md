# A2-Q2b — the REAL-CHECKPOINT per-block gate, and `lm_head` on NVFP4

**Issue:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Sibling / predecessor:** [`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md)
(A2-Q2a — the MoE arm, landed on the synthetic gate).
**State:** spec only. No product code.

---

## 0. Why this exists as its own unit

A2-Q2 was split twice, and this file owns the second split.

The first split (A2-Q2a / A2-Q2b) was taken because keeping `lm_head` on the
host **preserves A2-R's attributability**: both arms end in the identical host
projection, so a token difference stays attributable to the MoE arm.

The second split — deferring the **real-checkpoint per-block numeric gate** out
of A2-Q2a — was a scheduling decision, and it is recorded here rather than left
implicit. A2-Q2a's device MoE arm is **unreached** (G-SAFE refuses before it,
see that spec's `## Owed`), and end-to-end NemotronH is blocked on **A2-P**, not
on the quantized arms. Holding A2-Q2a's branch for a 21 GB checkpoint load on a
contended, reboot-prone box would have held the critical path behind a gate that
unlocks nothing. So A2-Q2a landed on the synthetic NVFP4 fixture, and the
expensive gate came here.

**This is a deferral, not a cancellation.** The synthetic result is explicitly
bounded (A2-Q2a §13.6.1) and cannot stand in for what this unit measures.

---

## 1. Scope

| In A2-Q2b | Out |
|---|---|
| the per-block numeric gate on the REAL checkpoint: every one of the 23 MoE layers against `trace.mixer[l]` | anything A2-P owns (paging, carried state, batching, the G-SAFE narrowing) |
| `lm_head` through the NVFP4 dense route, with the §4.3 residency decision applied as A2-Q2a applied it | the FP8 mamba arm — A2-Q1 |
| hybrid-vs-host token identity, and the disclosure that A2-R's attributability property ENDS when `lm_head` moves | fixing [#984](https://github.com/mudler/vllm.cpp/issues/984) or [#962](https://github.com/mudler/vllm.cpp/issues/962) |
| the §5.3 mutations A2-Q2a left owed (Q2-M3 … Q2-M7) | any throughput number, on any axis |

---

## 2. What it must measure, and why the synthetic arm does not

A2-Q2a's fixture reported **bit-exact** device-vs-host agreement (`0` over 512
elements, against a routed-scale separation of `0.6`). That is a real result and
a NARROW one. Its output is bf16, its contraction is K=128, its E2M1 codes are
exactly representable and its group scales are powers of two — precisely the
conditions under which a bf16 store absorbs genuine reduction-order differences.

So this unit must measure what the synthetic arm structurally cannot:

- **the real geometry** — H=2688, I=1856, E=128, top_k=6. The contraction is 21x
  longer, so the reduction order genuinely differs.
- **the real scale VALUES** — `weight_scale_2` and the fp8 group scales the
  checkpoint ships are neither powers of two nor uniform, unlike the fixture's.
- **the other Marlin thread configs.** The fixture resolves on `{128,64,128}`
  (up) and `{64,128,128}` (down) only; the real shapes may select others.

**A red here after a bit-exact synthetic is an EXPECTED outcome, not a
contradiction.** It is what A2-Q2a §13.6.1 predicts. Report it as a result; never
widen a band to absorb it.

---

## 3. The gate

Per-block numeric equivalence against the host reference via `NemotronHTrace`,
on the real checkpoint, at **every** one of the 23 MoE layers, plus `lm_head`
against the host projection on the same gathered rows, plus hybrid-vs-host token
identity.

**Bands are MEASURED, and the guard is a PROPERTY** — the shape A2-Q2a arrived
at after its first attempt failed on its own instrument:

- Every comparison reports **how many elements it examined**, and the caller
  asserts that count **against the geometry**, never against the buffer's own
  size (which would agree with itself if the buffer were short). A maximum over
  zero elements is `0.0`, and so is a bit-exact comparison; without the count
  the two are indistinguishable.
- Agreement, separation and the property guard must report the **same** count,
  or the band between them is fiction.
- The band must **admit exact agreement**. A2-Q2a's first band was
  `sqrt(agreed * separation)`, which collapses to 0 when the arms agree exactly
  and failed on the best possible outcome. `separation / 2` with an explicit
  `REQUIRE(separation > 0)` has the intended property and keeps `<` strict.
  Never relax `<` to `<=`: that accepts a band of 0.

---

## 4. ★ Operational — the host working set is what takes the box down

`gpu_memory_utilization` does **not** bound host RAM on GB10, and `nvidia-smi`
attributes only the device side. **Both instruments are blind to the transient
host working set**, which is what actually reboots the machine
(`vm.overcommit_memory=1`, zero swap: the kernel grants memory it cannot back and
the box reboots rather than OOM-killing a process).

So:

1. Take `$GPU_LOCK` with a **blocking** `flock` and wait. Never race.
2. **Check `free -g` headroom INSIDE the locked region**, not before acquiring
   it. A blocking flock says the previous holder released; it never says the box
   recovered. Abort loudly below a stated floor (A2-Q2a used 60 GB).
3. **Sample `free -g` on a loop for the whole load and record the PEAK**, not the
   final value. The figure that matters is transient.
4. Build `-j 4`. One log per run. Never hang holding the lock.
5. Verify the configure log reads `ENABLED for [121a]` — a `DISABLED` line or a
   `[121]` **voids** the result rather than failing it.

Budget to size against: ~17.6 GiB host weights + 16.5 GB device arena + page
cache for a 21 GB checkpoint. Checkpoint at
`${CHECKPOINT_ROOT}/nemotron-3.5-lightning-30b-nvfp4`, with `CHECKPOINT_ROOT` =
`/usr/local/nas_share/checkpoints` — `/usr/local` is COS_PERSISTENT and survives
a reboot; `/mnt` is the ephemeral root overlay of the immutable OS and does not.

**This row has lost four GB10 windows to environment rather than to code** —
`121` instead of `121a`, an unconstrained build, a CUTLASS fetch with no egress,
and an 8h19m outage ended by a human power cycle. Size the plan for that.

---

## 5. Owed

- [#962](https://github.com/mudler/vllm.cpp/issues/962) — NVFP4 Marlin disagrees
  with itself on sm_110 (`bitdiff=15/32768`). The **Thor leg stays PENDING** on
  it; do not quote a number from a kernel that contradicts itself.
- [#984](https://github.com/mudler/vllm.cpp/issues/984) — the address-keyed
  Marlin repack cache. A2-Q2a routed around it by never calling either
  `MarlinDenseResidentFor`; `lm_head` must do the same or say why not.
- The §5.3 mutations A2-Q2a left owed: **Q2-M3** (expert stride off by one),
  **Q2-M4** (`routed_scaling_factor` folded into the logits), **Q2-M5** (shared
  expert added before the routed scale), **Q2-M6** (`MoeRelu2` → plain relu),
  **Q2-M7** (device call site deleted). Q2-M1 and Q2-M2 were run in A2-Q2a.
  Q2-M3 matters most: `src/vt/ops.cpp:874-895` validates **no extent of
  `b_q_weight` and nothing at all about `b_scales`**, so a stride defect is
  silent at the op boundary and only the numeric gate can see it.

## 6. Now

Claimable once A2-Q2a lands. Its blocker is a GPU window, not a design question.
