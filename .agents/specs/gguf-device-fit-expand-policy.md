# GGUF-DEVICE-FIT-EXPAND-POLICY — the device-fit bound stops assuming the loader picks the cheaper residency

Issue: [#1870](https://github.com/mudler/vllm.cpp/issues/1870). **This row does
NOT close #1870** — see the boxed note at the end of Scope. It fixes one
necessary, verified defect in the refusal's arithmetic; a second, deeper gap
(#1934) keeps the refusal itself unreachable on ROCm today.
Owed, filed separately: [#1928](https://github.com/mudler/vllm.cpp/issues/1928)
(ROCm has no `kMoeGroupedGemmBf16` provider) and
[#1934](https://github.com/mudler/vllm.cpp/issues/1934) (`RocmPlatform::needs_weight_staging()`
is stale-false, so `CheckDeviceWeightFit` — this row's fix included — never
runs on ROCm; the ACTUAL blocker for #1870's reproduced crash).
Base: `3574065e7eb6a968fc57928a28cf5fb59b748778` (`upstream/main` at the claim;
this checkout's `origin` is a personal fork that lags `upstream` by ~200
commits — see `.agents/developer-preferences.md`).

## Scope

`CheckDeviceWeightFit` (`gguf_device_fit.h`, ENG-EXPERT-STREAM issue #1123) is
the load-time refusal that stops a GGUF whose weights cannot be staged onto the
target device from reaching the first forward and dying there instead. Its
per-tensor term, `StagedBytes`, is `min(gguf_bytes, elems * model_dtype_bytes)`
— the smaller of "kept quantized" and "expanded to the model dtype". The header
defends this as a lower bound: whichever the loader actually does, this is not
larger than it.

That defence has an unstated precondition: it assumes the loader is FREE to pick
the cheaper of the two for a given tensor. `RouteGgufTensor`
(`gguf_keep_quant.h`) says otherwise — it is a total decision over
`{keep_quant, keep_f16, nvfp4_fp4, cpu_ref}` and a tensor's role, and states
its own totality plainly: "Anything else is kExpandBf16 — the decision is
total and never throws." When an operator sets `VT_GGUF_KEEP_QUANT=0` on a
device where `keep_f16` and `nvfp4_fp4` are also off — every ROCm device
today, since `nvfp4_fp4` requires `kMatmulNvfp4`, CUDA-only, and `keep_f16`
rides `expand_nk`, which rides `keep_quant` — `RouteGgufTensor` returns
`kExpandBf16` for EVERY tensor, unconditionally. The bound still takes
`min(gguf_bytes, elems * 2)`, which for a compressed GGUF picks the on-disk
term nearly every time, so it reports a footprint roughly a quarter of what the
load will actually stage. On a 16 GiB ROCm card the refusal that exists
specifically to replace an allocator crash with a named message never fires,
and the load reaches `hipMalloc: out of memory` instead — reproduced on
`RX 9060 XT` (gfx1200, 16304 MiB) in #1870 on two checkpoints, one dense, one
MoE.

IN SCOPE:

- Make the footprint exact, not merely a safe lower bound, in the one case
  where `RouteGgufTensor`'s totality guarantee already makes the per-tensor
  answer known rather than assumed: every residency-shrinking flag off.
- Wire the one production call site (`model_loader.cpp`) to tell the bound
  when that condition holds, from the SAME resolved `GgufLoadPolicy` the
  loader already computes for the expert-stream lane check beside it.
- Document the precondition `docs/ENVIRONMENT.md:94` omits: the toggle needs
  roughly 4x the file size in device memory, and what refuses when it does not
  fit.

OUT OF SCOPE:

- Any per-tensor role classification inside `gguf_device_fit.cpp` for the
  MIXED case (some flags on, some tensors keep, some expand). That still needs
  role information this file does not have, is still a genuine approximation
  problem, and is unchanged by this row — see Risks.
- The `kMoeGroupedGemmBf16` ROCm provider gap #1870 names as a "related gap,
  same area". Filed as its own issue, #1928, and recorded under `## Owed`
  below rather than attempted here: the CUDA implementation is a single TU
  (`cuda_matmul_nvfp4.cu:1478-2732`) with a WMMA prefill path, a split-K
  decode path, persistent CUDA-graph-safe scratch, and a fused
  reduce+SwiGLU epilogue — a from-scratch HIP kernel port needing its own
  spec and hardware-gated evidence, not a slice of this one.
- Moving any residency DEFAULT. `keep_quant`, `keep_f16`, `nvfp4_fp4` availability
  rules are untouched; this row only makes the bound agree with what they
  already decide.
- `RocmPlatform::needs_weight_staging()`. Filed as #1934, and recorded under
  `## Owed` below, NOT attempted here: it is a platform-capability graduation
  with at least six other downstream consumers (load-path optimization, GDN
  kernel-dispatch defaults) that each need their own correctness check, not a
  one-line flag flip.

> **THIS ROW DOES NOT CLOSE #1870.** Discovered mid-implementation, on real
> `gfx1200` hardware: `CheckDeviceWeightFit`'s ONE call site
> (`model_loader.cpp:2273`, moved from `:2258` by this row's own hoisted-policy
> edit) is gated on `target.needs_weight_staging()`, and
> `RocmPlatform` hardcodes that `false` (`rocm.cpp:88`, a stale W0-era
> placeholder — ROCm has landed grouped-GEMM, MoE combine/gate, ROCM_ATTN and
> hipGraph capture since it was written and was never revisited). Measured
> directly: `VT_DEVICE_WEIGHT_BUDGET_BYTES=1` against a real checkpoint on
> `--device auto` produced **no refusal**, because `CheckDeviceWeightFit`
> returns before computing anything when `needs_weight_staging()` is false.
> The real device allocation #1870's crash comes from (`d.b.Alloc(nb)` in
> `qwen3_5.cpp`'s `ResidentWeight`) is NOT gated by that flag, so the crash
> stays fully reachable on ROCm after this row lands. This row's fix is
> necessary — once the guard runs, the arithmetic must be right — but #1934 is
> what makes the guard run at all on this platform, and #1870 stays open
> until it does. See `## Owed`.

## Upstream chain

No upstream vLLM mirror. Pinned vLLM (`.agents/upstream-sync.md`, `555967922`)
has no GGUF load format, so there is no `DeviceConfig`/`memory_profiling`
counterpart that answers "will this fit before I pay for it" — the same header
comment `gguf_device_fit.h` opens with. This row edits a vllm.cpp-original
predicate against its own prior design, not an upstream port.

## Design

`RouteGgufTensor`'s switch is total: `kNvfp4Fp4` needs `nvfp4_fp4`, `kKeepQuant`
needs `keep_quant`, `kKeepF16` needs `keep_f16`, and every tensor that clears
none of those three gates is `kExpandBf16` — regardless of role, dtype or
shape. So the conjunction `!(keep_quant || keep_f16 || nvfp4_fp4)` is not a
heuristic; it is the exact condition under which the footprint can stop
guessing and charge the expanded size unconditionally, because that is the
ONLY residency `RouteGgufTensor` can produce.

`StagedBytes` and `GgufStagedWeightFootprint` gain one new parameter,
`policy_forces_full_expand` (default `false`), and `CheckDeviceWeightFit`
forwards it through. Default `false` is a byte-for-byte no-op for every
existing caller and test: unset, the function takes the same
`min(gguf_bytes, elems * model_dtype_bytes)` it always has. Set `true`,
`StagedBytes` returns the expanded term outright — no `min`, because there is
nothing left to be uncertain about.

The one production call site, `model_loader.cpp`'s GGUF branch, already
resolves `GgufLoadPolicy::FromEnv()` once for the expert-stream lane check
beside this one (`GgufExpertTowersReachSlotLane`). That resolution is hoisted
into a local, reused by both calls (previously two separate `FromEnv()` calls
computed the identical policy twice), and
`!(policy.keep_quant || policy.keep_f16 || policy.nvfp4_fp4)` is passed as the
new argument. `cpu_ref` needs no term of its own: it forces `kExpandBf16`
unconditionally too, but every `cpu_ref` load is a CPU load, and
`CheckDeviceWeightFit` already returns before computing anything when
`needs_weight_staging` is false — the oracle switch never reaches a device
that stages weights in the first place (`RouteGgufTensor`'s comment: "the
oracle switch wins over everything").

**Why not thread a `GgufLoadPolicy` all the way into the footprint and drop
`min` argument entirely?** Considered and rejected. The MIXED case — some
residency flag on, but a given tensor's role/dtype/shape makes it ineligible
(a ragged K, an unsupported encoding, `DeviceKeepQuantSupported` false) — still
needs the per-tensor role this file deliberately does not carry (see the
header's own note on `GgufExpertTowersReachSlotLane` needing the loader to ask
the SAME routing question the model's own loader asks, tensor by tensor,
because only the loader knows roles). `min()` stays the defensible bound for
that case, unchanged. The new parameter narrows the exact claim to precisely
the sub-case where role does not matter, and says so.

## Risks and decisions

- **The MIXED case is still approximate, and stays so.** A load with
  `keep_quant=1` on a checkpoint carrying one dtype ROCm's keep-quant list
  does not cover (`DeviceKeepQuantSupported`) still under-counts that one
  tensor's contribution exactly as it did before this row. Not a regression:
  `policy_forces_full_expand` is false whenever any flag is on, so that path is
  byte-for-byte unchanged. Recording it here rather than silently accepting is
  what the header already does for the file/load-scope over-count (#1136); this
  is the same shape, named rather than fixed, because fixing it needs role
  information this predicate does not have.
- **`cpu_ref` gets no explicit term.** Argued above: every `cpu_ref` load is a
  CPU load in this tree (the flag exists to reproduce the historical dequant
  path for the parity oracle, and nothing stages weights on the CPU platform),
  so `CheckDeviceWeightFit`'s existing `!needs_weight_staging` early return
  already keeps it out of scope. If a future device both stages weights and
  wants `cpu_ref` this term would need revisiting; nothing in this tree does
  that today, and `test_gguf_device_fit.cpp` already has fixtures using
  `PolicyWith(..., /*cpu_ref=*/true)` this row's tests build on.
- **A caller that has NOT resolved a `GgufLoadPolicy` still gets the safe
  (approximate, never over-tight) answer.** Default `false` is the
  conservative choice — under-counting is what #1870 is about, so the new
  parameter's default must never accidentally introduce it for a caller this
  row does not touch. Tests pin the default.

## Tests

**`tests/vllm/model_executor/test_gguf_device_fit.cpp`**, red first:

- A fixture combining the existing two-tensor file's Q8_0 and F32 tensors
  (`BuildTwoTensorGguf`, already in the file) with
  `policy_forces_full_expand=true`: the expected footprint is the SUM OF
  EXPANDED SIZES (`64 + 16 = 80`), not the existing `min`-based `50`. Asserted
  against `GgufStagedWeightFootprint` directly, and against
  `CheckDeviceWeightFit` at a budget between 50 and 80 (refuses only with the
  new parameter, passes without it) — the case that is RED before the fix,
  because today's `StagedBytes` has no such parameter and cannot distinguish
  the two loads.
- A default-parameter case pinning `policy_forces_full_expand=false` reproduces
  the EXISTING `kExpectedLowerBound=50` result byte for byte, so the new
  parameter's default is proven to be a no-op rather than merely documented as
  one.
- A production-shaped case: a policy built with
  `PolicyWith(/*keep_quant=*/false, /*keep_f16=*/false, /*nvfp4_fp4=*/false,
  /*cpu_ref=*/false)` derives `policy_forces_full_expand` the same way
  `model_loader.cpp` does (`!(keep_quant || keep_f16 || nvfp4_fp4)`), so the
  test exercises the exact boolean expression the call site uses rather than a
  hand-picked `true`/`false` literal that could drift from it.

**`tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp`** — reachability half,
already registers a fake staging platform; extended with one case setting
`VT_GGUF_KEEP_QUANT=0` (and no other override) against a fixture sized to fit
the MIN-based bound but not the full-expand bound, proving the production call
site now refuses where it previously loaded past the point of no return.

## Gates

- `./build-hip/tests/test_gguf_device_fit` (ROCm build, this box, gfx1200) — GREEN, 20/20 cases, 148/148 assertions
- `./build-hip/tests/test_gguf_device_fit_reach` — GREEN, 15/15 cases, 73/73 assertions
- `./build-nix-cpu/tests/test_gguf_device_fit` / `test_gguf_device_fit_reach` (CPU build, same box) — GREEN, both suites
- `python3 scripts/check-env-doc.py` — OK, 373 production env vars documented/classified
- `python3 scripts/check-commit-trailers.py` / `check-commit-style.py` against the true fork point (`git merge-base upstream/main HEAD`, NOT `origin/main` — see `.agents/developer-preferences.md`) — OK
- A real hardware repro attempted on this box and **BLOCKED**, not by this
  row's fix, by #1934: with `needs_weight_staging()` false on `RocmPlatform`,
  no local checkpoint — real or budget-starved via
  `VT_DEVICE_WEIGHT_BUDGET_BYTES=1` — can be made to hit the refusal through
  `vllm-cli` at all today, regardless of this fix. What DID run on real
  hardware: both unit suites above, built and executed on `gfx1200`, and
  `tests/vt/test_rocm_backend` / `test_rocm_arch` / `test_backend_cross_device`
  confirming the platform itself (device probe, real kernel execution) is
  healthy on this card. See `## Owed`.

## Evidence

Recorded in the pull request body: the red output (compile failure against the
unfixed three/six-argument signatures, captured by stashing the implementation
and rebuilding) before the fix, green after on both CPU and ROCm builds on this
box, and the real-hardware attempt that surfaced #1934.

## Owed

- **#1928** — the ROCm `kMoeGroupedGemmBf16` provider gap #1870's "related gap,
  same area" section names. Not fixed in flow; scoped out above.
- **#1934** — `RocmPlatform::needs_weight_staging()` is stale-false, so this
  row's fix (and the refusal it corrects) never runs on ROCm. THE ACTUAL
  remaining blocker for #1870's reproduced crash. Not fixed in flow: it is a
  platform-capability graduation with its own downstream correctness burden
  (GDN kernel-dispatch defaults move), scoped out above.
- **The MIXED-case approximation** stays exactly as conservative (and exactly
  as capable of under-counting on a role this predicate cannot see) as it was
  before this row. No issue filed for it beyond the standing `#1136` note the
  header already carries, because this row changes nothing about that case.

## Stop conditions

Stop and report rather than widening scope if the reachability fixture cannot
be sized to separate the two bounds without also tripping the existing
`nextn`-block over-count case (`kExpectedLowerBound + kNextnStaged`) — that
would mean the two rows' fixtures interact and need reconciling together
rather than in this row alone.

Stop rather than touching any residency DEFAULT (`keep_quant`,
`GgufQuantComputeAvailable`, `DeviceKeepQuantSupported`, ...). This row moves no
default and no availability rule.

## Now

`DONE` for this row's own scope (the arithmetic fix), `PARTIAL` against
#1870's user-facing symptom. Implementation, tests, docs and CPU+ROCm
hardware evidence are in; the real end-to-end crash-to-refusal repro is
blocked on #1934 (recorded above), which is why #1870 stays open rather than
closing with this PR. Awaiting fresh review and the operator's gate.

## Outcome

**What was measured.** `CheckDeviceWeightFit`'s per-tensor bound
(`min(gguf_bytes, elems*model_dtype_bytes)`) silently assumed the loader could
always pick the cheaper residency. `RouteGgufTensor`'s totality guarantee
makes that assumption false exactly when `keep_quant`, `keep_f16` and
`nvfp4_fp4` are all off — the case `VT_GGUF_KEEP_QUANT=0` produces on any
device without a keep-f16 or native-fp4 arm, ROCm included. Fixed by adding
`policy_forces_full_expand`, a parameter derived once from the resolved
`GgufLoadPolicy` and threaded through `GgufStagedWeightFootprint` and
`CheckDeviceWeightFit`, defaulting to `false` (byte-identical for every
existing caller).

**What was rejected.** Threading a full `GgufLoadPolicy` into the footprint to
also close the MIXED-case approximation (some flags on, one tensor's
dtype/role ineligible) — rejected because that needs per-tensor ROLE
information this file deliberately does not carry, and closing it is a
different, larger change (see Design's "why not" paragraph). Attempting the
`kMoeGroupedGemmBf16` ROCm port and the `needs_weight_staging()` platform-flag
fix in this same row — both rejected as separate-issue-sized platform work
(#1928, #1934) discovered/scoped during this row rather than folded in under
time pressure.

**Why the real-hardware repro stayed a compile+unit-level result instead of
the planned crash-to-refusal transcript.** Every locally available checkpoint
either uses a GGUF architecture or tensor encoding this build's reader does
not support (nonstandard high-numbered `ggml_type` ids from a community
requant pipeline, or standard IQ variants this tree has not ported), or —
once a loadable file was found — the discovery above: `needs_weight_staging()`
false on ROCm means no checkpoint, however sized, can reach the refusal
through the production loader on this platform today. The fix's correctness
is established by the CPU+ROCm unit and reachability suites instead, which
construct the exact byte-level scenario directly rather than depending on a
compatible real checkpoint.
