# Laguna — the only forward-runnable fixture is constant, so its run-gates hold trivially

**Row:** `MODEL-LAGUNA-FIXTURE-2834` — new row, `ACTIVE`.
**Issue:** [#2834](https://github.com/mudler/vllm.cpp/issues/2834).
**Date:** 2026-09-04. **Base:** `7b8b480b1`.
**Predecessor row:** `MODEL-LAGUNA-REGISTRY-FORWARD-2618`, whose `## Owed` O3
names this defect and whose mutation M5 (zeroing `positions`) SURVIVED because
of it.

## Scope

`BuildFiniteTensors()` in `tests/vllm/models/test_laguna_nvfp4_loader.cpp` is
the only forward-runnable Laguna fixture in the tree. Four shipped cases run a
forward on it, two through `LagunaForwardGguf` directly and two through
`ModelRegistry::Forward`. The forward on this fixture returns the same number
for every logit, so none of those cases can detect a defect in the embedding
gather, in RoPE, in the causal mask, or in the position plumbing.

This row makes the fixture discriminate, and gates the discrimination itself so
that a later fixture edit cannot silently flatten it again.

Out of scope: a real-checkpoint Laguna gate (no `poolside/Laguna-S-2.1-NVFP4`
snapshot exists on any host this row can reach), a Laguna token golden vs vLLM
([#2841](https://github.com/mudler/vllm.cpp/issues/2841)), the paged arm, and
any change to `LagunaForwardGguf` or to `BuildTensors()` — the byte-identity
round-trip fixture, which is a different builder and is not touched.

## The mechanism, measured rather than assumed

`Bf16Finite(name, shape, seed)` fills element `i` with

```
((i * 7 + seed) % 16 - 8) * 0.01F
```

The period of that sequence is 16. `model.embed_tokens.weight` and
`lm_head.weight` are both `[V, H] = [8, 32]`, and a row stride of 32 advances
the sequence by `32 * 7 = 224`, which is `0 (mod 16)`. **Every row of both
tensors is therefore byte-identical to every other row.**

Three consequences follow, and they are exactly the three invariances #2834
measured:

- Every embedding row is the same vector, so the token id cannot move anything.
- Every hidden state entering attention is the same vector, so a convex
  combination of the values is that vector whatever the attention weights are,
  and RoPE — hence the position — cannot move anything either.
- Every `lm_head` row is the same vector, so all `V` logits of a row are equal,
  and all rows are equal.

The value `0.0444346` reported in #2834 is that single number.

## Design

One line changes in the fixture: the period-16 generator becomes period 13.

```
((i * 7 + seed) % 13 - 6) * 0.01F
```

`32 * 7 = 224` is `3 (mod 13)`, so consecutive rows advance by 3 and the eight
vocabulary rows take eight distinct phases. The range narrows from
`[-0.08, 0.07]` to `[-0.06, 0.06]`, so the values stay small, bounded and
finite, which is what `std::isfinite` on the output needs to remain meaningful.
Nothing about the dtype, the shapes, the tensor names or the safetensors layout
moves, so the loader sees the same file structure it saw before.

`BuildTensors()` — the builder the byte-identity round-trip case at
`test_laguna_nvfp4_loader.cpp:272` compares against — uses `Fill()`, not
`Bf16Finite()`. #2834's stated reason for deferring the repair ("the same
tensors serve the byte-for-byte load checks") does not hold: the two builders are
disjoint, and the round-trip assertions do not move.

## Tests

The repair is worth nothing without an assertion that holds it, because the
next fixture edit can re-flatten the generator by accident exactly as this one
did. A new case asserts the three discriminations directly, on the production
entry point:

- **T1** two different token vectors through `ModelRegistry::Forward` on the
  same weights produce different logits.
- **T2** the same tokens at two different position vectors produce different
  logits.
- **T3** within one step the per-row logit vectors are not all equal, and within
  one row the `V` logits are not all equal.

T1 to T3 are the RED. On the fixture as committed each of the three maxima is
exactly `0`, so all three fail while the four pre-existing forward cases stay
green — which is the defect stated precisely: the shipped gates cannot see what
T1 to T3 measure.

## Gates

```sh
cmake --build build -j 3 --target test_laguna_nvfp4_loader
./build/tests/test_laguna_nvfp4_loader
scripts/agent-preflight.sh
```

## Evidence

Recorded in the pull request body: RED with both doctest counts, GREEN with both
counts, every mutation with its rebuild rc and the restore sha256, and the
reachability mutation.

## Stop conditions

Stop and report `NEEDS_DECISION` if making the fixture discriminate turns any
pre-existing assertion in this file red, because that would mean a real forward
defect the constant fixture was hiding, and a forward repair is a different row.

## Owed

- O1. The routed experts are still uniform. `ExpertProjFinite` writes the same
  packed byte `0x11` and the same scale byte `0x38` into every expert, so the
  router's choice cannot change the output and no gate on this fixture can see a
  routing defect. T1 to T3 do not close this, and it needs its own red-first
  argument about what the loader round-trip on the fp4 fields still compares.
- O2. A real-checkpoint Laguna gate, inherited unchanged from
  `MODEL-LAGUNA-REGISTRY-FORWARD-2618` O2 and
  [#2841](https://github.com/mudler/vllm.cpp/issues/2841). A synthetic fixture
  that discriminates is still not an oracle.
