# SPEC-DFLASH2 — the selector's edge kernel reads every successor codebook row K times

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding).
**Issue:** [#2155](https://github.com/mudler/vllm.cpp/issues/2155).
**Related:** [#2154](https://github.com/mudler/vllm.cpp/issues/2154) (acceptance
collapse — the other half of the c=8 deficit),
[#2152](https://github.com/mudler/vllm.cpp/issues/2152) (no reading on this rung
is currently admissible), [#545](https://github.com/mudler/vllm.cpp/issues/545)
(the measurement host reboots roughly hourly).
**Kind:** performance. Spec first; the implementation lands separately because
it needs a lease and this does not.

## Now

`ACTIVE` — spec only.

## The observation

`VT_SPEC_TRACE=1` on a c=8 step, stable across every run of 2026-08-28:

```
[spec-phase] pre=0.26ms backbone=0.30ms sample=19.36ms logits=2234880
```

`sample` is the candidate selector plus the path walk, not a vocabulary softmax.
It costs **65x the draft forward pass** and, at the observed step rate, roughly a
quarter of the decode step — spent producing k=8 drafts.

## The mechanism

`ComputeCandidatesDevice` is not the cost. It is a `TopKValuesIndices` over
`[rows, vocab]` with `rows = P*L = 64` and `vocab = 2234880 / 72 = 31040` — about
8 MB — followed by a `MulScalar` and a `SoftCap` over `[64, K]`.

`Dflash2SelectorEdgesKernel` (`src/vt/cuda/cuda_ops.cu:3735`) is. Its grid is one
block per `(b, l, predecessor)` slot. Each block stages the gated predecessor row
into shared memory ONCE — correct, and the same idea this spec applies to the
other axis — then loops over `K` successors, reading a full successor codebook
row of length `R` from global memory on every iteration:

```
for (int64_t c = 0; c < K; ++c)
  for (int64_t r = threadIdx.x; r < R; r += blockDim.x)
    acc += gated[r] * Load(succ_codebook, cid * R + r);
```

The `K` successor rows for a given `(b, l)` are the SAME across all `K`
predecessor blocks of that `(b, l)`. Successor traffic is therefore

    B * L * K * K * R    elements

where `B * L * K * R` suffices: **a factor-`K` blowup**, `K` being
`selector_top_k`.

## Design

Give each block more than one predecessor and stage the successor rows — or
tiles of them — in shared memory beside the gated predecessor. Each successor
row is then read once per `(b, l)` rather than `K` times.

The tile size is bounded by shared memory: staging `T` successor rows costs
`T * R` elements, so `T` is chosen from `R` and the block's shared budget, and
the loop becomes a two-level walk over successor tiles. When `T >= K` the whole
successor set is staged once and the factor of `K` disappears; when `T < K` the
reduction is `T`.

## What this does NOT change

The arithmetic. Every accumulation is the same product in the same order per
`(pred, succ)` pair, so the CUDA arm stays bit-identical to the CPU reference
`Dflash2SelectorEdgesKernel` in `cpu_ops.cpp` that it mirrors.

## Tests

The existing CPU-versus-CUDA parity case for this kernel is the correctness
gate, and it must be run BEFORE and AFTER unchanged — an optimisation that moves
a score is a defect, not a speedup. Add a case at the production `(B, L, K, R)`
if none covers it, for the reason #2171 records one axis over: this file's
kernels have twice shipped a path no test executed.

## Gates

- CPU/CUDA parity unchanged, at the production shape.
- A measured before/after on one lease, both arms in ONE session, interleaved,
  with a terminal control — and taken through
  `tools/bench/dflash2_speed_harness.py`, not around it (#2152).
- `[spec-phase] sample=` is the axis. The step-level `out tok/s` is NOT, until
  #2154 is settled: on this rung it varies 0.0-79.6% in zero-draft-block rate
  across runs of one binary, which swamps anything this change can produce.

## Owed

- **`selector_top_k` and `selector_rank` are unread.** They come from the draft
  checkpoint's `config.json` (`qwen3_dflash_weights.cpp:432-438`), on the NAS
  behind a host that was down when this spec was written. Without them the
  factor-`K` argument is a scaling claim and the size of the prize is unknown.
- **No select-versus-walk split exists for this build.** `VT_SPEC_TRACE=2`
  prints `[spec-phase-dev] pre= fwd= select= walk=` and would attribute the
  19.4 ms between the two halves. Until it runs, the walk is not excluded.

## Stop conditions

Return `NEEDS_DECISION` if the level-2 trace attributes most of `sample` to the
WALK rather than the selector. The design above then addresses the smaller half
and the row should be rescoped before any kernel is written.

## Outcome

Filled in when the row reaches `DONE`.
