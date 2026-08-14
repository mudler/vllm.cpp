# Tenstorrent tt-forge — the only reference for Tenstorrent execution

vLLM has no Tenstorrent platform anywhere
([`../porting-inventory.md`](../porting-inventory.md) §15), so `BACKEND-TENSTORRENT`
is an extension with no upstream analog to mirror. That is the strongest form of
the fallback case: not "vLLM disagrees", but "vLLM has nothing to say".

**It qualifies as an oracle because it executes, not merely compiles.** tt-forge
is an end-to-end stack over TT-Metalium with three frontends (TT-XLA for
PyTorch/JAX, TT-Forge-ONNX, TT-Lang), running inference and training and testing
800+ model variants in CI. A stack that only lowered IR would be a source to read,
not an oracle to run.

**`tt-forge-models` is not a second oracle.** It is the shared model-definition
repository — a `ForgeModel` interface with `load_model()` / `load_inputs()` — and
it publishes no reference outputs and no PCC or accuracy comparison against a
reference framework. It supplies *subjects*, so it is named in the scope below
rather than given a record of its own.

**ttnn is not an oracle either.** Our backend is a thin `vt::` adapter over
ttnn's C++ op library, which makes ttnn an implementation dependency. An oracle
is something we compare against; ttnn is something we call.

**Not gateable, and hardware is the reason.** tt-forge requires Wormhole or
Blackhole silicon, and nothing in this repository has executed it — the ACTIVE
Tenstorrent row gates its ops against our own CPU f32 path instead
([`../backend-matrix.md`](../backend-matrix.md) `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`).
The Blackhole P150 that the row's real-hardware gates run on is where a tt-forge
arm would first become measurable. There are also no release tags, so its pin will
be a commit, not a version.

```oracle-pin
id = tt-forge
role = secondary
upstream = https://github.com/tenstorrent/tt-forge
scope = Tenstorrent Wormhole/Blackhole execution, whose subjects come from tenstorrent/tt-forge-models; vLLM has no Tenstorrent platform at all
pin = UNPINNED
pin_label = none
pinned_on = 2026-08-13
gateable = no
evidence = #647
```
