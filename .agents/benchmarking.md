# Task guide — measuring performance

How to produce a number worth believing. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## The denominator

vLLM is the bar, quant-matched, in its **production** configuration. Never
benchmark against `--enforce-eager` and call it parity. llama.cpp may appear
only as an explicitly labelled secondary comparison.

Both sides run the pinned oracle on identical model artifacts, prompts, token
counts, batching, concurrency, and sampling. If the two sides differ in any of
those, the ratio means nothing.

Prove the oracle actually *runs* the model before trusting it as a
denominator — constructing a config proves nothing.

## Getting a clean measurement

One GPU job at a time. Take the box lock before any measurement, stop competing
services, and never run two large models at once — unified-memory boxes reboot
rather than swap.

Calibrate the noise band from repeated identical legs *before* interpreting a
delta. Discard cold legs for a named cause, never because they are
inconvenient. Use paired, order-alternated A/B legs and a majority rule; a
single pair is an anecdote.

Prefer an instrument that is immune to page-cache effects (GPU-active time per
step) over wall clock when the host is doing heavy I/O.

## The clock is part of the measurement

**The SM clock differs between boots and does not announce it.** On `dgx.casa`
one boot ran the timed window at a median 2470 MHz and the next at a flat
2190 — a 12.79% delta, with `clocks_throttle_reasons.active = 0x0` and
persistence `Enabled` throughout, so nothing looked wrong. It repriced a
byte-identical `marlin::Marlin` with no source change by **+9.65%**, which is
larger than either deficit that comparison was being used to rank (#543). Two
probes eight minutes apart *inside one boot* disagreed by ~6% uniformly.

So a number is quotable only with the clock it was taken at.
`tools/bench/gpu_clock_state.py` is the one helper that samples, folds and
asserts it: the SM clock across the measured window (min/median/max, the
retained sample count and the idle count), `clocks.max.sm`,
`clocks.applications.graphics`, the active throttle reasons, persistence mode,
and the **boot id**.

**Two harnesses call it today.** `scripts/dgx-online-serving.sh` records a
clock window per leg, and `tools/bench/online_gate_summary.py` asserts it.
`scripts/dflash2-speed-gate.sh` opens ONE WINDOW PER ARM and
`tools/bench/dflash2_speed_harness.py` delegates the whole judgement to this
helper, including the cross-arm pairing — a single window spanning both arms
cannot see the offset, and the offset is the term that transfers into the
ratio. The
trace and per-kernel harnesses — `finalize_*_trace.py`,
`summarize_torch_kernels.py`, `gdn_packed_component.py` — are **not wired**, so
a `us/call` or per-kernel figure from those paths carries **no clock
attribution** and cannot be quoted as one. That is not a footnote: it is the
path both retracted #543 findings came from. Wiring them is owed work tracked in
[`specs/bench-assert-clock-state.md`](specs/bench-assert-clock-state.md); until
it lands, import the helper and record a window yourself before ranking anything
from a trace, or say plainly that the figure is unattributed. Any new harness
imports this helper rather than rolling its own.

Two arms on **different boots are not comparable**. The summary refuses that
pair outright; `--allow-cross-boot` waives the **boot id and nothing else**, and
stamps a recorded caveat rather than passing silently — the GPU, driver, maximum
SM clock, applications clock and persistence mode are compared across the arms
unconditionally, because a waived boot is not a waived machine. Within a run the
SM-clock spread must stay at or below **5%**, and the two arms' medians within
**1%** of each other — and their **means** within 1% too, which is a separate
rule and not a restatement: on the three 2026-08-19 Qwen3.8-27B c1 pairings the
median offset reads exactly **0.00%** on all three while the arms' mean clocks
are 0.10 to 0.25 points apart, because the excursion population sits below the
median and is the part that does not cancel between the arms (#1546). Throughput
is an integral over the window, so the mean is what transfers. A record captured
before that term carries no mean and is refused rather than skipped, so an
archived evidence tree has to be re-recorded rather than re-summarized. A window must also have been **observed**: at least **30
retained busy samples** and a **majority** of the window busy, because the
spread over one sample is definitionally 0.00% — the best score the gate can
award — so without a floor the window nobody watched outscores the one that was.
The argument for all four numbers, including why the spread ceiling is
deliberately *not* held to the criterion the offset was chosen by, is in
[`specs/bench-assert-clock-state.md`](specs/bench-assert-clock-state.md).

**Pin the clocks before measuring, under the lock.** Passwordless `sudo` is
available on `dgx`, and `-lgc` is supported:

```sh
sudo nvidia-smi -lgc 2100        # pin, before the first leg
sudo nvidia-smi -rgc             # release, after the last one
```

**This works on the host path only.** Inside an `rc` lease `nvidia-smi -lgc`
returns `LGC_RC=4`, "The current user does not have permission to change
clocks", even as root, measured 2026-08-19 on `dgx:gpu0` in three jobs. Fleet
devices are reachable by lease only, so for them the SM clock can be SAMPLED and
not pinned, and a pairing may be refused on within-run spread with no lever to
fix it. **The missing capability is named and the ask is a one-line manifest
change**: the worker's `CapBnd` is the default OCI set and holds no
`CAP_SYS_ADMIN`, measured inside leases on two fleet devices
([`specs/lease-gpu-capability.md`](specs/lease-gpu-capability.md), #1354). Read
[`environment.md`](environment.md) before you plan a paired series.

Pinning is a **shared-host mutation**. Never run `-lgc` or `-rgc` while another
session holds `$HOME/gpu.lock` — it silently reprices their in-flight
measurement, which is the very defect this section exists for. Take the lock,
pin, measure, reset, release. It is a **pre-measurement step**, not a standing
configuration: leaving the box pinned makes every later run inherit a state
nobody recorded, which is where this started.

Figures recorded before 2026-08-12 predate clock assertion. They are not
withdrawn and are not restated — they simply carry no clock attribution, so a
delta smaller than ~10% between two of them is not established by them alone.

Budget the disk before the run. A production RelWithDebInfo CUDA build tree is
about **169 GiB** — the build contract claimed ~3 GiB until 2026-08-10, a 56x
underestimate on the one number that decides whether a grid fits. A full disk
does not fail loudly: it voids the binding through memory-return tolerance while
still emitting plausible ratios. Leave real headroom, and delete the tree once
the evidence directory is captured (evidence is tens of MiB).

Two ratio sets that disagree may be two different HARNESSES rather than a
regression. Compare their absolute numbers before believing either; ratios are
scale-invariant and hide an order-of-magnitude mismatch completely. If the
change between the readings is provably inert (a byte-identical refactor),
suspect the measurement, not the code.

## Two arms have to BE two arms

A pair that measured one artifact twice already produced a "no speedup" result in
this tree and was nearly reported as a refutation
([#672](https://github.com/mudler/vllm.cpp/issues/672),
[`specs/minimax-music3.md`](specs/minimax-music3.md) §16.6a). The tell was not
the times, which were 0.26 % apart and read as noise. It was the **identical call
count**: equal times are noise, equal counts are identity.

**A different `sha256` does not establish it**
([#1516](https://github.com/mudler/vllm.cpp/issues/1516)). Measured on a minimal
project of the shape this repository's examples have — one `SHARED` library
carrying the change, one thin client linking it — two byte-identical source trees
built into two build directories give two clients of equal size with different
hashes, and making the library change for real leaves the client byte-for-byte
the hash it already had. CMake writes the build-tree RPATH into a client and no
`CMAKE_SKIP_BUILD_RPATH` is set here. Hashing the library instead only moves the
problem: it is stable across two build directories and differs across two SOURCE
directories, because `VT_CHECK` embeds `__FILE__` and nothing sets
`-ffile-prefix-map`. In the two-clone shape a two-tree A/B is required to use,
**no artifact hash in this tree is falsifiable**.

**Prefer a same-binary A/B.** Where a runtime switch turns the change off inside
one binary there is no second artifact and no hash to be vacuous:
`VT_OP_PROVIDER_DISABLE=<provider>` is that lever for anything behind the op
provider seam (`src/vt/cuda/cuda_attention_cross.cu:636`), and
`GetOpProviderStats` says which kernel actually ran.

**Otherwise render the verdict with the control, not the hash.**

```sh
scripts/ab-arms-differ.py --artifact-a A --artifact-b B \
    --root-a /tmp/b-old --root-b /tmp/b-new \
    --control ar.depth_forward 1414 808
```

Equal hashes stay `FATAL`. A hash-only verdict is refused by name, and an
artifact that embeds its own build or source root is reported with the offset, so
the hash leg's worth is stated instead of assumed. At least one control must have
moved. Two kinds catch different failures and neither subsumes the other: a
**behavioural** control is a value the arms computed and is the only leg that
catches a stale binary; a **source** control is the hash of the file the change
lives in and catches two arms that are one source.
`scripts/music3-vocoder-conv-ab.sh` is the worked example.

## Reading a profile

**A whole-run kernel ranking is a trap.** It sums prefill and decode, so the
top-percentage kernel is frequently one-time prefill work that no decode step
touches. Use a decode-only window or diff two sequence lengths. A `Max` far
above the `Median` means you are looking at a mixture, not a hot loop.

Profile the entire step, not only the kernels. Several of the largest wins here
were host-side waste, not slow math.

Before accepting a gap as "GPU-bound", trace both implementations with the same
tool on the same workload and compare what actually ran.

## Recording it

Record the exact build and run recipe, revisions, model hashes, environment,
clock and contention state, raw output, and the same-binary A/B. Reproduce on an
idle box before acceptance. "Clock state" is the concrete list in §The clock is
part of the measurement, boot id included, not a prose adjective.

Record every required axis — throughput, latency, memory — as both values and
ratios. An axis below floor is an open gap, not a rounding error.

Never record a ceiling. An apparent same-architecture limit is an unresolved
implementation difference; name the next traceable hypothesis instead.

Accepted and pending results go in [`benchmark-record.md`](benchmark-record.md)
and `docs/BENCHMARKS.md`. Method specific to one lever stays in
[`parity-lever-protocol.md`](parity-lever-protocol.md).
