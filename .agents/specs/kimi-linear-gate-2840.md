# Kimi-Linear — 122/128 is a failing strict gate, and the shipped test asserts a floor

**Row:** `MODEL-TEXT-KIMI-LINEAR-GATE-2840` — new row, `ACTIVE`.
**Issue:** [#2840](https://github.com/mudler/vllm.cpp/issues/2840).
**Date:** 2026-09-04. **Base:** `bbabd84fa`.
**Predecessor row:** `MODEL-TEXT-kimi-linear-kimi-linear-for-causal-lm`, whose
spec [`kimi-linear.md`](kimi-linear.md) already calls 122/128 a failure of the
required bar at §589-591, §1295-1296 and §1354.

## Scope

`tests/vllm/models/test_kimi_linear_fold_gate.cpp` is the CLI-incremental
reference leg for `KimiLinearForCausalLM`. This row makes it assert the bar
`CLAUDE.md` §Gates licenses, and makes the bar SELECTION executable rather than
prose.

Out of scope: the forward, the six divergent tokens, the speed axis, and any
capture. All of those need the 91.5 GiB `Kimi-Linear-48B-A3B` snapshot, which is
not on this host and not on the NAS.

## What is true at `bbabd84fa`

**D1 — the oracle is deterministic, so STRICT is the only licensed bar.** The
committed capture `tests/parity/goldens/kimi_linear_greedy/greedy_dist.npy` is
`(8,16,3) int32`: three independent greedy runs of the pinned oracle. Every one
of the 128 `(prompt,position)` cells holds a single id, so the oracle reproduces
itself exactly, and `greedy_dist[:,:,0]` equals `greedy_ids.npy` element for
element. `CLAUDE.md` §Gates: "Use an explicitly ratified distributional gate only
when the oracle's greedy decode is non-deterministic." It is deterministic here.

**D2 — the shipped assertion is a floor, not exactness.**

```cpp
CHECK(static_cast<int64_t>(matched) * 128 >= static_cast<int64_t>(total) * 122);
```

That admits six wrong tokens in a 128-token battery on a bar that admits none.
`docs/FEATURES.md` called the resulting 122/128 "the intrinsic near-tie profile"
until [#2825](https://github.com/mudler/vllm.cpp/issues/2825) corrected it; the
row's own spec had called it a failure all along.

**D3 — nothing in the tree measures what licenses the bar.** The determinism
verdict lives in prose in `kimi-linear.md`. The capture that supports it is
committed and machine-readable, and no test reads it.

The case's `doctest::skip(std::getenv("VT_KIMI_MODEL_DIR") == nullptr)` is
CORRECT and is kept. It is a named skip, so it does not report a pass — this row
does not repeat #2839's D3 here, because this file does not have it.

## Design

**A — the floor becomes exactness.** `matched == total`, with the message naming
the two admissible exits rather than a number to tune toward. A subset run stays
honest, because exactness scales to any subset without an inequality.

**B — the bar selection becomes a test.** A new case reads the committed capture
with no checkpoint and no GPU: zero multi-valued cells over K=3, and
`greedy_dist[:,:,0] == greedy_ids`. That second assertion is what makes the first
one about the ORACLE rather than about an array: a capture whose first run did
not produce the committed ids would be a capture of something else.

`test_kimi_linear_fold_gate` gains `PARITY_GOLDENS_DIR` so it can read the
committed golden. The env-selected `VT_KIMI_GOLDEN_DIR` is unchanged, because the
91.5 GiB leg is run against a directory the operator names.

## Tests and expected verdicts

- **T1** (checkpoint-free) zero multi-valued cells over K=3, and the capture's
  first run equals the committed ids. Expected GREEN, and it is what licenses T2.
- **T2** (checkpoint-gated) `matched == total`. NOT RUNNABLE HERE. The last
  recorded measurement is 122/128, so this is expected to FAIL on the next host
  that has the snapshot, and that failure is the row's point.

## Gates

```sh
cmake --build build -j 3 --target test_kimi_linear_fold_gate
./build/tests/test_kimi_linear_fold_gate
scripts/agent-preflight.sh
```

## Stop conditions

Stop and report `NEEDS_DECISION` before relaxing T2. The two admissible exits are
128/128 against the deterministic golden, or an explicit ratification of a
distributional bar over an oracle that reproduces itself — which `CLAUDE.md`
§Gates does not offer.

## Owed

- O1. T2 is unrun. There is no `Kimi-Linear-48B-A3B` snapshot on this host and
  none on the NAS, so `VT_KIMI_MODEL_DIR` cannot be set and the case skips. It
  compiles; nothing here can execute a 91.5 GiB load.
- O2. No `our_ids` artifact is committed for this row, so the 122/128 cannot be
  reproduced from the tree the way #2839's 69/128 can. Committing one alongside
  the golden would make the failure checkable without the checkpoint, and it is
  owed to whoever next runs the leg.
- O3. `docs/USAGE.md` carries no checkpoint row for `KimiLinearForCausalLM` — no
  revision and no sha256 — which `CLAUDE.md` §"Say which weights, and from where"
  requires. Not repaired here because the values must be read off an artifact
  this host does not have.
