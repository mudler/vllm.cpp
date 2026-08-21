# ENG-CUDAGRAPH-DEDUP: one graph executable per topology, not one per bucket

**Row:** `ENG-CUDAGRAPH-DEDUP` in [engine-matrix.md](../engine-matrix.md).
**Issue:** [#1162](https://github.com/mudler/vllm.cpp/issues/1162).
**Analysis that derived this row:** [sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md), `## Owed`.
**Kind:** memory and capture-time optimization behind the `vt` graph seam. **Not a
throughput row.** A deduped replay launches exactly the same nodes with exactly the
same parameters; if it launched anything else it would be a correctness defect, which
is why the load-bearing gate below is byte-identity rather than a speed ratio.

## Now

`ACTIVE`, and the state is argued rather than inherited. The shared registry, its CPU
contract suite, the CUDA wiring, the
[#1184](https://github.com/mudler/vllm.cpp/issues/1184) repair and the coarse key of
[#1226](https://github.com/mudler/vllm.cpp/issues/1226) have all landed — the last of
them as `2c8f53d93`, so the "PR #1232 is a draft" caveat every earlier record carried is
retired and the fold is reachable on `main` today, behind two default-OFF flags.

**The row's benefit question is now CLOSED, and the answer is NEGATIVE.** W6 ran the
device-byte measurement W4 and W5 both named as owed, on the SHIPPED bucket sets rather
than the 2-3 buckets earlier runs reached. The fold is real there: 7 captures fold to 3
executables at `max_num_seqs=32` and 11 fold to 5 at 64. The saving is not: 13.83 MiB
nominal at 7 buckets, which is 0.42% of a 3.25 GiB process and SMALLER than the
disagreement of its own null control, and nothing at all at 11 buckets. The driver
refuses 43% of probes at 7 buckets and 73% at 11, every one of them
`cudaGraphExecUpdateErrorTopologyChanged`, because the decode graphs are two topologies
of 376 and 404 nodes rather than one. **That inverts the premise the coarse key was
built on**: a key that cannot see a topology difference produces more false candidates,
not more folds. Both flags therefore stay OFF, on measurement rather than on silence.
Full result: [`## Outcome`](#outcome), W6.

**Why the row is still `ACTIVE` and not `DONE`.** The measurement obligations are
discharged and the decision is delivered, which is the DONE case and it is a real one.
Three things stop the flip here, and none of them is a checker technicality. The
feature is unreachable on the DEFAULT serving path, because the async runner captures no
decode graph — that reach is owned by `ENG-CUDAGRAPH-BREAK`
([#1179](https://github.com/mudler/vllm.cpp/issues/1179)) and is the "nothing lands
dead" half of this row. Two items still sit under this row's own issue: the device-tier
signature stability and discrimination tests, and probing `group.current_raw` instead of
`raws.front()` to retire the transitivity assumption in `Replay`. And the `DONE` record
surface itself owes a `.agents/parity-ledger.md` entry, a closing-commit owner in place
of `CLAIM-ENG-CUDAGRAPH-DEDUP`, an exact test anchor, and the release of that claim —
files this record-only branch does not own. It is not `PARTIAL`, because nothing
upstream is omitted; the coarse key is our own extension past SGLang. It is not
`BLOCKED`, because nothing external stops the next step.

## Scope

In scope:

1. A device-agnostic graph-executable dedup registry behind the existing `vt`
   `Backend::{EndCaptureGraph,ReplayGraph,DestroyGraph}` seam, keyed on a structural
   signature of the captured graph, folding compatible captures onto one executable
   with `cudaGraphExecUpdate` / `hipGraphExecUpdate`.
2. Retention of the raw captured graph, which the current seam destroys immediately
   after instantiate. Both the signature and the update need it.
3. The CUDA ops table and the CUDA backend wiring, off by default behind
   `VT_CUDA_GRAPH_DEDUP`.
4. An observable capture-count / exec-count log line and the programmatic
   `CapturedCount()` / `ExecCount()` accessors behind it, so the ratio is a measurement
   rather than a claim.

Out of scope, each owned elsewhere:

- Break points and piecewise capture: `ENG-CUDAGRAPH-BREAK`,
  [#1163](https://github.com/mudler/vllm.cpp/issues/1163).
- The diffusion / denoise-loop capture: `ENG-CUDAGRAPH-DIFFUSION`,
  [#1164](https://github.com/mudler/vllm.cpp/issues/1164), blocked.
- Retiring the nine hand-rolled decode drivers. This row does not touch their
  internals; it changes only what the seam they already call hands back.
- Any change to bucket selection, padding, or the `pure_decode` predicate.

## Upstream chain

vLLM has **no analogue**. Its graph executables come from `torch.compile`
(`vllm/config/compilation.py:60-63,517,615,630` @ `555967922`), which owns
instantiation, so there is no vLLM file to mirror. Per AGENTS.md `## When vLLM has no
implementation`, the secondary oracle is SGLang, pinned in
[oracles/sglang.md](../oracles/sglang.md) at `v0.5.15` / `f63458b5beaceabbd9d749b9fc956370e1b649e6`.
Paths below are relative to `python/sglang/srt/`.

| Mechanism | Upstream anchor | Ported here |
|---|---|---|
| `cudaGraphExecUpdate` wrapper returning (ok, detail) | `model_executor/runner_backend/cuda_graph_dedup_mixin.py:27-37` | `GraphDedupOps::update` |
| Structural signature over node payloads in a deterministic topological order plus the edge set | `:139-179` | `GraphSignature()` in `src/vt/graph_dedup_runtime.h` |
| Per-node payload by node type (kernel / memcpy / memset / child graph / empty) | `:105-136` | same function, same five cases |
| Group holding one live exec plus the raw graph it currently reflects | `:182-188` (`GraphExecGroup`) | `GraphDedupRegistry::Group` |
| Register: signature hit re-points, signature miss instantiates a new group | `:219-242` | `GraphDedupRegistry::Register` |
| Replay: update only when the group's current raw graph differs, then launch | `:258-275` | `GraphDedupRegistry::Replay` |
| Compatibility probe on a separate executable so a failed probe cannot leave the live one in an undefined state | `:236-238` (`compat_exec`), destroyed by `seal()` at `:244-251` | `GraphDedupRegistry::Register`, **transient** probe — see the divergence below |
| `"captured %d CUDA graphs, deduped to %d execs"` | `:353-358` | `GraphDedupRegistry::CapturedCount()` / `ExecCount()` plus the log line |
| Retained raw graph (`torch.cuda.CUDAGraph(keep_graph=True)`) | `:326` | `EndCaptureGraph` stops calling `cudaGraphDestroy` when dedup is on |

Three deliberate divergences from the upstream construction, each forced by a
difference in our seam rather than chosen:

- **The probe executable is transient, not persistent.** SGLang holds a second
  `compat_exec` per group for the whole capture phase and frees it in `seal()`,
  which its backend calls at `end_cuda_graph_capture()` (`:352-358`). **Our seam has
  no capture-phase end.** The nine drivers capture lazily, the first time a padded
  bucket is seen, interleaved with replays of buckets already captured
  (`src/vllm/model_executor/models/qwen3_5.cpp:9918`,
  `src/vllm/model_executor/models/qwen3.cpp:888`). A persistent probe would therefore
  never be sealed, would live for the process, and would hold **two** executables per
  group — which is worse than the one-per-bucket baseline this row exists to reduce.
  So the probe is instantiated from the group's first raw graph, updated to the
  candidate, and destroyed inside `Register`. Peak executables during a registration
  is 2, the same as today's steady state per capture, and steady state afterwards is
  1 per signature group.
- **A failed probe falls back; it does not abort.** SGLang asserts (`:229`). Here a
  signature hit whose update the driver rejects gives that graph its own group, so the
  worst case degrades exactly to today's behaviour instead of killing the process. This
  is what makes an under-specified signature safe rather than wrong: the signature is
  only a candidate filter, and `cudaGraphExecUpdate` is the authority.
- **Kernel identity is the host function pointer, not the kernel name.** SGLang reads
  names through the CUDA driver API (`cuKernelGetName` / `cuFuncGetName`, `:44-55`)
  because it holds driver handles. The runtime API hands us
  `cudaKernelNodeParams::func` directly, and pointer identity is strictly stronger than
  name identity for the same reason two `__global__` templates can share a demangled
  prefix. SGLang's launch-attribute probing (`:58-102`) is driver-API-only and is
  dropped for the same reason; dropping it can only produce a false signature *hit*,
  which the probe then rejects, and never a false miss.

## Our baseline

`grep -rn "cudaGraphExecUpdate" src include` returns nothing at the base of this
branch (`9bc4d7f44`). Today every capture instantiates its own executable and throws
the raw graph away:

- `src/vt/cuda/cuda_backend.cu:222-232` — `EndCaptureGraph` does
  `cudaStreamEndCapture` → `cudaGraphInstantiate` → `cudaGraphDestroy(graph)` and
  returns the exec as the opaque `void*` handle.
- `src/vt/rocm/rocm_backend.hip:271-283` — the same shape on `hipGraph`.
- `src/vt/backend.cpp:36-38` — the default implementations `VT_CHECK(false)`;
  Vulkan, Metal and CPU do not support capture. Tenstorrent implements the trio over
  `MeshTraceId` (`src/vt/tenstorrent/tenstorrent_backend.cpp:79-81`) and is not in
  scope.

The executables multiply along two axes. `include/vllm/model_executor/models/decode_graph_sizes.h:32-41`
yields 7 padded decode buckets at `max_num_seqs=32` and 11 at 64, and nine drivers
each build their own set: `Qwen3_5DecodeGraph`, `Qwen3_5DenseDecodeGraph`,
`Qwen3MoeDecodeGraph`, `Qwen3DenseDecodeGraph`, `DeepseekV2DecodeGraph`,
`VoxtralDecodeGraph`, plus the graph code in
`src/vllm/model_executor/models/deepseek_v4.cpp:1900` and
`src/vllm/model_executor/models/laguna.cpp:2691`, and the DFlash draft graph in
`src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106`. This
spec said **eight** until 2026-08-18; the ninth was found and the count corrected by
[#1179](https://github.com/mudler/vllm.cpp/issues/1179) on `ENG-CUDAGRAPH-BREAK`, and
carried here so the two rows describe the same tree.

Why it matters here specifically, restated from the issue so this spec stands alone:
on GB10 unified memory an out-of-memory event reboots the box, capture time is startup
latency and startup latency is a recorded gate axis, and bucket count is exactly the
quantity that `ENG-CUDAGRAPH-BREAK` would raise if it widened coverage.

The standing hazard on this path is recorded in
[decode-graph-scratch-uaf-2026-07-18.md](decode-graph-scratch-uaf-2026-07-18.md):
capture bakes the pointers live at capture time, so a later free of a captured buffer
is a use-after-free on replay, and a clean `compute-sanitizer` run is **not** evidence
that a capture path is safe, because its serialization hides exactly this bug. Dedup
interacts with that hazard in one direction only and it is the safe one: retaining the
raw graph extends a host-side object's lifetime and frees nothing earlier than before.
Nothing in this change frees a device buffer, and the executables it does not
instantiate were never a source of baked pointers.

## Port map

| New / changed | What |
|---|---|
| `src/vt/graph_dedup.h` (new) | `GraphDedupOps` (six function pointers: signature, instantiate, update, destroy exec, destroy graph, launch) and `GraphDedupRegistry`. Header-only, no device dependency, CPU-unit-testable — the same shape `src/vt/cuda/graph_safe_scratch.h` used for the graph-safe scratch bookkeeping. Also `GraphDedupEnabled()`, the `VT_CUDA_GRAPH_DEDUP` read, and the `CapturedCount()` / `ExecCount()` accessors. There is no single `Stats()` method: two accessors read at the call site beat one struct nobody stores. |
| `src/vt/graph_dedup_runtime.h` (new) | The CUDA / HIP ops table, written **once** and bound to either runtime by a macro alias block. This is the file that must not become two hand-written copies. After [#1184](https://github.com/mudler/vllm.cpp/issues/1184) it holds only what is device-shaped: the `GetEdges` version binding, the five node-payload cases, the six raw operations, and the `Runtime` policy that carries `ClearLatchedError`. `Ops()` is its only export. |
| `src/vt/graph_dedup_latch.h` (new, #1184) | `ScopedLatchClear<Rt>` and `MakeLatchGuardedOps`. Device-free. Every `GraphDedupOps` field is a wrapper address, so a runtime error latched anywhere inside the binding is consumed before control leaves it — on a plain return, on a degradation escape, and on a `VT_CHECK` unwinding. |
| `src/vt/graph_dedup_signature.h` (new, #1184) | The device-free half of the structural signature: Kahn ordering, the topological re-index, the sorted edge emission, the child-graph depth bound and the four degradation escapes, templated on the `Rt` policy above. Split out so a CPU test can drive it; it was previously reachable by no test on any tier. |
| `src/vt/cuda/cuda_backend.cu` | `EndCaptureGraph` routes through the registry when enabled and retains the raw graph; `ReplayGraph` and `DestroyGraph` dispatch on `registry->Owns(handle)`. |
| `src/vt/rocm/rocm_backend.hip` | The same three edits against the same shared header. |
| `docs/ENVIRONMENT.md` | `VT_CUDA_GRAPH_DEDUP`. Required: `scripts/check-env-doc.py` fails a `VT_*` name that is neither documented nor allowlisted, and this is a user-facing behaviour knob rather than kernel-internal tuning. |
| `.agents/engine-matrix.md` | The row moves `INVENTORIED` → `ACTIVE`; the summary table's "Engine and scheduling" line moves one row between the two state columns. No row is added or removed, so `ENGINE_ROWS` stays 160. |

The `void*` handle contract does not change. With dedup off, `EndCaptureGraph` returns
the raw executable exactly as today and the path is byte-identical. With dedup on it
returns a registry-owned handle, and `Owns()` — an O(1) set membership test on handles
the registry itself minted — is what lets one backend serve both kinds without
guessing at a pointer's provenance.

## Tests to port

SGLang ships no unit test for the mixin, so there is nothing to port parameter-for-
parameter; what is portable is the guarantee set, and each case below names the
upstream line it pins. New suite `tests/vt/test_graph_dedup.cpp`, driven by a fake ops
table over a toy graph model so every guarantee is gated on **every** platform rather
than only on a CUDA box. The fake's `launch` appends the id of the graph the executable
currently reflects, which makes "the right nodes ran" an observable sequence instead of
an assertion about intent.

| Case | Guarantee | Upstream |
|---|---|---|
| `folds compatible captures onto one executable` | K same-signature captures leave exactly 1 live exec; `(CapturedCount(), ExecCount())` is `(K, 1)` | `:219-242`, `:253-256` |
| `keeps incompatible captures apart` | different signatures never share | `:225-241` |
| `replays the graph the caller asked for` | over an interleaved multi-replay sequence, the deduped launch log equals the non-deduped control log, element for element | `:258-275` |
| `does not re-update for a repeated shape` | a second consecutive replay of one handle issues no update | `:267` |
| `falls back to a private executable when the driver rejects the update` | a probe failure yields its own exec, the existing group's live exec is untouched, and replay stays correct | diverges from `:229`, stated above |
| `frees the shared executable only with its last graph` | destroying one sibling keeps the exec; destroying the last frees the exec and every retained raw graph, with all counters balanced | `:277-292` |
| `does not claim a handle it did not mint` | `Owns()` is false for a foreign pointer | ours; the seam has no upstream twin |
| `is off unless the environment asks for it` | `GraphDedupEnabledFor()` polarity, including the values that only the terminator check rejects (`"10"`, `"11"`, `"1 "`, `"1x"`) | `:321` |
| `a capture the driver cannot instantiate fails at the capture site` | a null `instantiate` throws inside `Register`, mints no handle, counts nothing, and releases the capture it took ownership of | ours; added by the fresh review of #1178 |
| `a probe the driver cannot instantiate degrades instead of driving a null exec` | a failed probe answers "cannot fold" rather than passing a null executable to `cudaGraphExecUpdate` | ours; added by the fresh review of #1178 |
| `a replay update the driver refuses fails loudly rather than launching stale nodes` | a refusal on the fold `Register` never probed throws, and the executable's previous contents are NOT launched under the asking handle | ours; pins the safety claim that makes the transitivity assumption below survivable |
| `a two-member group replays a pair the probe never tested` | the exact `(reflected -> target)` sequence a two-member group issues: one probe `(1 -> 2)`, no update for a replay of the graph already reflected, `(1 -> 2)` on the first alternation and `(2 -> 1)` on the second. The reverse pair is the one nothing probed, and it arrives at the SMALLEST group the registry can form | ours; added by the fresh review of #1232, which traced the pairs and found the disclosure one member too late |
| `the unprobed reverse pair of a two-member group fails loudly` | the same shape with the driver refusing `(2 -> 1)`: the probe still accepts, the captures still fold, and the refusal surfaces on the ALTERNATION rather than at capture time — loudly, never as a launch of the executable's previous contents | ours; the coarse key is what makes two-member groups real, so this is its boundary and not a theoretical one |

A second suite, `tests/vt/test_graph_dedup_runtime.cpp`, was added by
[#1184](https://github.com/mudler/vllm.cpp/issues/1184) over the two device-free halves
of the runtime binding. It has no upstream twin: SGLang runs on a Python runtime that
raises rather than latching, so the whole error-latch class does not exist there.

| Case | Guarantee |
|---|---|
| `every guarded operation consumes the latch its own failure set` | five arms, one per fallible operation; after each the latch is clean and a stand-in for the next unrelated kernel launch does not throw. The refusal still carries the driver's `detail` string, so clearing does not swallow the reason the caller acts on |
| `the clear survives an operation that throws` | `Launch` reports a failed `cudaGraphLaunch` by throwing. This case is what decides between a clear placed after the call and a clear placed in a destructor: only the destructor runs on the unwinding path |
| `a refused fold leaves nothing latched for the next kernel` | #1184 in the shape a CPU tier can hold: a registry whose probe the driver REFUSES — the feature's normal operation — leaves nothing for the next caller to misread |
| `no table field holds an address that skips the guard` | each of the six `GraphDedupOps` fields differs from the raw function's address |
| `an operation the guarded builder does not wire cannot reach the registry` | a `GraphDedupOps` member that `MakeLatchGuardedOps` does not set stays null, and the registry's constructor refuses an incomplete table |
| `the signature is independent of the order the runtime reports nodes in` | the load-bearing property of the walk: `cudaGraphGetNodes` promises no order, so without the Kahn re-index two captures of one topology key differently and fold nothing, observable only in the device-side log line |
| `the signature is independent of the order the runtime reports edges in` | a diamond with its edges scrambled keys identically |
| `the signature separates topologies that differ only in their edges` / `only in a node payload` | discrimination in both directions |
| `a child graph contributes its own signature, bounded at depth four` | the bound is exact on both sides: level 4's payload is emitted and its child is not walked |
| `a child graph is walked, not merely noted` | the nested signature appears inside the parent's, in the exact byte form |
| `each runtime failure degrades the key instead of aborting inside a capture` | five escapes, five exact strings: `nodes?;`, `edges?;`, `edge?;`, `cycle?;`, `[A,;node?;` |
| `an empty graph still produces a signature` | `[]` rather than an escape |
| `the exact key discriminates every launch dimension and every copy extent` | the SHIPPED default's own coverage: `grid.y`, `grid.z`, `block.y`, `block.z` and the memcpy `height` and `depth` each separate two payloads. Every helper here passed 1 for all six, so they were executed and never discriminated, and dropping `grid_z` from the exact key left the suite green |
| `the coarse key drops the y and z components too, not only x` | the same six fields, in the other polarity |
| `the coarse key groups two decode buckets that differ only in launch dimensions` / `still separates two different kernel functions` / `keeps sharedMemBytes` / `drops the memcpy extent and keeps the kind` / `drops the memset width and keeps the element size and height` / `does not weaken the topology half of the signature` | what `VT_CUDA_GRAPH_DEDUP_COARSE_KEY` drops and what it keeps, field by field, against the production field-selection functions |
| `two captures the coarse key groups share one executable when the driver accepts` / `the probe counters tell a key that never grouped from a driver that refused` | the end of the chain, and the instrument: `probes=0` and `refused=probes` are the two opposite causes of `N == M`, asserted on both shapes and printed on the registry's own line |

A third suite, `tests/vt/test_graph_dedup_coarse_env.cpp`, gates the coarse-key KNOB.
It is a separate binary because `GraphDedupCoarseKeyEnabled()` caches its answer in a
function-local static — deliberately, so a knob cannot move between two lazy captures of
one model — and one process can therefore observe exactly one value. `tests/CMakeLists.txt`
registers it once per `(value, expectation)` pair: `1` is ON, and `0`, `10`, `01`, `1x`,
`true`, empty, unset and "only the sibling knob `VT_CUDA_GRAPH_DEDUP` is set" are each
OFF. Without it the accessor's only coverage was one unset-variable `CHECK_FALSE`, which
survived both accepting any non-null value and reading a completely different variable
name — so nothing proved WHICH variable the device A/B's coarse arm was asking for.

| Case | Guarantee |
|---|---|
| `the coarse key follows VT_CUDA_GRAPH_DEDUP_COARSE_KEY, and only "1" enables it` | one process per value; the ON entry is what pins the variable NAME, and the near-miss entries are what pin the terminator check |
| `the coarse key is read once, so it cannot change between two captures` | flipping the variable after the first read does not move the answer |

## Gates

1. **Red first.** Every case above is written and run against the un-implemented
   registry, and the red output is captured, before the implementation exists.
2. **Focused green.** `ctest -R test_graph_dedup`, which now selects both
   `test_graph_dedup` and `test_graph_dedup_runtime`.
3. **Mutation.** Each guarantee is deleted or inverted in a scratch copy, the focused
   suite is proven to fail, and the tree is restored byte-for-byte. `git diff --stat`
   and the compile status are printed for every mutation, because a mutation that fails
   to build and a mutation that never applied both read as a passing test.
4. **Full gate.** `./scripts/agent-preflight.sh`.
5. **CUDA compile.** The `cuda-fat-build` CI job (`.github/workflows/ci.yml:669-712`,
   container `nvidia/cuda:13.3.0-devel-ubuntu24.04`) compiles the CUDA leg on the PR.
   This is the gate. The #1184 session had no local toolkit and used a scratch
   header-shape stub as a proxy — `graph_dedup_runtime.h` instantiated through `Ops()`
   at `CUDART_VERSION` 12090 and 13030 and on the HIP arm, `-Werror`, all three clean,
   with a negative control proving the instrument can fail. A proxy is not the gate: the
   CUDA 13 `cudaGraphGetEdges` break this file already took was invisible to exactly
   this kind of local check and only `cuda-fat-build` reported it.
6. **Device byte-identity A/B — RAN 2026-08-18, PASS; see [`## Outcome`](#outcome).** Same binary, `VT_CUDA_GRAPH_DEDUP`
   off then on, identical prompts and sampling, token-exact equality, with **more than
   one replay per padded bucket** — a first replay can be correct by accident, and the
   whole hazard of a shared executable is that the *second* visit to a shape is the one
   that has to re-point it. **Stated limit on how that clause was satisfied:** the driver
   prints a replay TOTAL and no per-shape breakdown, so "more than one replay per bucket"
   rests on the total over the captured shapes (workload B, 60 replays over 2 shapes,
   ~30 each) and is arithmetic rather than a direct measurement.
7. **Not gated, deliberately:** throughput. This row must not be sold as a speed
   change. The reportable numbers are executable count and capture wall time.
8. **The coarse-key device experiment (W5, #1226).** The same harness, the same three
   multi-bucket workloads, the same OFF/OFF controls first, one binary and three arms:
   `VT_CUDA_GRAPH_DEDUP` unset, `=1` with the EXACT key, and `=1` with
   `VT_CUDA_GRAPH_DEDUP_COARSE_KEY=1`. A cell is VOID unless its log carries the
   `key mode = ` line for the arm it claims and a `captured N graphs, deduped to M execs`
   line, because an A/B where nothing engaged is not a pass. Three outcomes, all of them
   a complete result:
   - **`M < N` and the arms are byte-identical** — the row delivers. Record the ratio and
     the probe-refusal rate; the default flip becomes arguable on that evidence rather
     than on a rerun.
   - **`probes > 0` and `refused == probes`** — the driver will not re-point across
     launch dimensions on this driver. A REAL negative: it closes #1226 with a
     driver-level reason and prints the exact `cudaGraphExecUpdateResult`, which is the
     evidence the W4 record could not produce because the fold was never attempted.
     `AGENTS.md` forbids declaring a ceiling, so a negative names the next traceable
     hypothesis rather than closing the question.
   - **`M < N` and the arms DIFFER** — stop and report, do not repair. That is a
     correctness finding about `cudaGraphExecUpdate`, and it outranks the row.
   A fourth reading is a defect in the run rather than a result: `probes == 0` with more
   than one distinct captured bucket in a coarse cell means the key still did not group
   and the hypothesis was never tested.
9. **Reachability for W5 is the DEVICE run, and the CPU suite cannot supply it.** Deleting
   the `AppendKernelFields` call in `src/vt/graph_dedup_runtime.h` leaves
   `tests/vt/test_graph_dedup_runtime.cpp` green, because that suite calls the
   field-selection functions directly. It measures the functions, not the path. What
   proves the path is the `key mode = ` line and the fold counts in the cell logs: both
   are emitted from `Runtime::AppendNodePayload` through `graph_dedup_rt::Ops().signature`,
   `GraphDedupRegistry::Register` and `CudaBackend::EndCaptureGraph`, which is a
   production entry point. Stated here rather than left for the reviewer to find, because
   this is exactly the shape `.agents/reachability.md` warns about.

## Dependencies

- The `vt` graph seam `include/vt/backend.h:196-209` and its two accelerator
  implementations. No other row is editing them: `ENG-CUDAGRAPH-BREAK` is a parallel
  agent on [#1163](https://github.com/mudler/vllm.cpp/issues/1163) and works above the
  seam, on the drivers and the runner routing.
- `scripts/check-env-doc.py` for the new environment name.
- No dependency on the diffusion path, which captures nothing today.

## Work breakdown

- **W1 — the shared registry.** `src/vt/graph_dedup.h`, `tests/vt/test_graph_dedup.cpp`,
  red first, then green, then the mutations. Device-free; gated on every platform.
- **W2 — the runtime ops table.** `src/vt/graph_dedup_runtime.h`, one source bound to
  both runtimes, plus the CUDA backend wiring and the environment documentation.
  Compile-gated by `cuda-fat-build`.
- **W3 — the ROCm wiring.** The same three edits in `rocm_backend.hip`.
- **W4 — the device A/B.** A leased GPU, both arms of the same binary, token-exact
  comparison and the executable-count ratio. **DONE, and it split**: see
  [`## Outcome`](#outcome).
- **W5 — the coarse key** ([#1226](https://github.com/mudler/vllm.cpp/issues/1226)).
  W4 refuted the row's premise by measuring `N == M` with `cudaGraphExecUpdate` never
  attempted, because the key carried the padded batch dimension. W5 tests the one
  hypothesis that can still deliver the benefit: **the key is stricter than the operation
  it guards.** `cudaGraphExecUpdate` requires the TOPOLOGY to match and exists to permit
  PARAMETER changes; a kernel node's launch configuration is a parameter, its function is
  not. `VT_CUDA_GRAPH_DEDUP_COARSE_KEY` drops the launch dimensions, the memcpy extent
  and the memset width so two padded buckets group at all, and the probe then decides.
  Default OFF, so the arms are a same-binary A/B. Its three legitimate outcomes and what
  each one closes are in [`## Gates`](#gates); the design argument for what is dropped
  and what is kept lives beside the code in `src/vt/graph_dedup_signature.h`.

## Risks/decisions

- **A shared executable is shared state, and that is the whole risk.** Two handles that
  resolve to one executable cannot be replayed concurrently on two streams, because the
  second replay's update would re-point the executable out from under the first. Today's
  drivers replay decode graphs from one runner thread on one queue
  (`src/vllm/model_executor/models/qwen3_5.cpp:9840,10270`), so the constraint holds at
  the base, but it is a constraint the seam did not previously carry. It is documented at
  the registry, and `ENG-CUDAGRAPH-BREAK` must honour it if it widens who replays.
- **Alternating shapes cost one update per switch.** The registry re-points only when
  the group's current raw graph differs, mirroring `:267`, so a run that stays in one
  bucket pays nothing. A workload that alternates buckets every step pays one
  `cudaGraphExecUpdate` per step. This is the reason the row is `T2` and the reason the
  default is off until W4 measures it: shipping it on by default without that number
  would be trading an unmeasured latency for an unmeasured memory saving.
- **A COARSER key cannot make a replay wrong, and can make it LOUD.** The safety
  argument is unchanged and unweakened: `Register` probes every candidate with the real
  driver update on a throwaway executable before it folds, so a key that groups two
  graphs the driver rejects costs one wasted probe and a private executable. What a
  coarser key DOES raise is exposure to the transitivity assumption recorded below. The
  probe tests exactly one pair per fold, `(raws.front(), member)`, because the group's
  executable is instantiated from its first capture. **Every replay whose
  `(current_raw, target)` pair is not that exact probed pair is untested — including the
  REVERSE direction inside a two-member group, which is the ordinary alternating-decode
  case.** Once such a group has re-pointed onto its second member, going back to the
  first asks the driver for `(second -> first)`, and nothing probed that. A refusal there
  lands on `Replay`'s `VT_CHECK`: a loud failure rather than a wrong answer, which is the
  polarity the row wants, but an availability failure all the same, and one the smallest
  group the registry can form already reaches. This bullet said "three or more members"
  until the fresh review of [#1232](https://github.com/mudler/vllm.cpp/pull/1232) traced
  the pairs; the statement was one member too late, and it was vacuous while the exact
  key formed no groups at all. The coarse key is what makes two-member groups real —
  `b_coarse_a` and `c_coarse_a` in the W5 run are exactly that shape. Both boundaries are
  now gated in `tests/vt/test_graph_dedup.cpp`, at size two by
  `a two-member group replays a pair the probe never tested` and
  `the unprobed reverse pair of a two-member group fails loudly`, and at size three by
  `a replay update the driver refuses fails loudly rather than launching stale nodes`.
  The device run records the probe-refusal rate rather than only the executable count for
  the same reason, and the counters are in the product for it.
- **The W5 device run did NOT exercise an alternation, and its `refused=0` therefore says
  nothing about the reverse pair.** Asked directly of the logs rather than assumed. In
  every cell of `/mnt/nas_share/rc/dedup-key/logs-ab/`, the padded decode buckets are
  captured in STRICTLY DESCENDING order — `sizes=[24 16 8]` for workload A, `[16 8]` for
  B, `[32 24]` for C — which is the signature of a decode batch that only ever shrinks:
  the harness admits a fixed prompt pool at a fixed concurrency, so once the pool is
  exhausted the batch falls to the remainder and never climbs back. A bucket is therefore
  replayed only before the next, smaller one is captured. The one pair each coarse group
  ever issued is `(raws.front() -> second member)`, which is precisely the pair the probe
  tested. **This is a derivation, and it is stated as one.** No log records the replay
  SEQUENCE: the driver prints a replay total and a slot count, the registry prints
  capture and executable counts, and neither says which handle was replayed when. The
  capture order and the harness's fixed prompt pool are the whole of the evidence. Two further limits follow from the instrument rather than from the workload:
  the registry counts PROBE refusals only, so a replay-time re-point is not counted at
  all, and a replay-time refusal would have aborted the process on `Replay`'s `VT_CHECK`
  rather than incrementing anything. Every cell completed, so no replay-time refusal
  occurred — but that is not evidence that the risky pair was ever asked. **Owed:** a
  workload whose batch size returns to a previously captured bucket, and a counter for
  replay-time re-points so the ask is observable rather than inferred.
- **`N == M` had two indistinguishable causes, and now has one each.** W4 could not tell
  "the key never grouped, so the driver was never asked" from "it grouped and the driver
  refused" without reading the signature's source afterwards. `probes=` and `refused=`
  on the registry's own line separate them, and the driver's `cudaError_t` and
  `cudaGraphExecUpdateResult` pair is printed verbatim on each refusal. A device run that
  cannot make that distinction measures nothing, which is why the instrument landed with
  the flag rather than after it.
- **`sharedMemBytes` is KEPT under the coarse key, and this was decided rather than
  swept.** It is a kernel-node parameter like the dimensions are, so the same reasoning
  would drop it. Two facts say do not. It is not where the batch dimension lives — decode
  dynamic shared memory is sized by head dimension and block geometry — so dropping it
  buys the hypothesis nothing. And a dynamic size above the 48 KiB static limit is legal
  only for a function that opted in through `cudaFuncAttributeMaxDynamicSharedMemorySize`,
  an attribute of the FUNCTION rather than of the node, which makes a changed size the
  field most likely to force a re-instantiate instead of a re-point. Keeping it preserves
  a real discriminator at no cost to what is being measured and holds the experiment to
  one variable. The memcpy `kind` and the memset `elementSize` and `height` are kept for
  the harder version of the same reason: the update contract itself refuses a changed
  memcpy memory type and refuses to change a memset that is not 1-D, so a key that
  dropped them would manufacture refusals rather than folds.
- **Retaining the raw graph costs host memory.** A `cudaGraph_t` holds the node
  descriptions, not the device-side executable image; it is retained only while dedup is
  on, and it is released with its handle. The direction is safe with respect to
  [decode-graph-scratch-uaf-2026-07-18.md](decode-graph-scratch-uaf-2026-07-18.md),
  which is a premature-**free** hazard.
- **The signature can only be too coarse, never too fine, and coarse is safe.** Every
  hit is probed with the real driver call before it is honoured. This is deliberate:
  a signature that had to be exhaustive would be a correctness surface, and it is
  instead a lookup key.
- **The fold that is probed is not always the fold that is replayed, and the gap is an
  unasserted transitivity assumption.** `Register` probes the pair
  `(group.raws.front(), raw_graph)`; `Replay` issues `(group.current_raw, entry.raw)`.
  They coincide only for the FIRST replay that re-points a member, and they diverge from
  group size **two** onwards: the reverse direction `(second -> first)` inside a
  two-member group is a pair no probe ever made, and the third member adds
  `(second -> third)` on top of it. Honouring the probe therefore treats
  `cudaGraphExecUpdate` compatibility as **transitive and symmetric** across a group's
  members — if the driver re-points A onto B and A onto C, then it re-points B onto C and
  B back onto A. Neither the CUDA nor the HIP documentation states this,
  and nothing here asserts it. The fresh review of [#1178](https://github.com/mudler/vllm.cpp/pull/1178)
  demonstrated it with a driver refusal keyed on the `(current, target)` pair: every
  `Register` probe succeeded and the **second** `Replay` threw. The failure polarity is
  what makes it survivable — a refusal lands on the `VT_CHECK` in `Replay`, loudly, and
  never on a silent launch of the executable's previous contents — which two cases now
  gate, each by refusing exactly a pair `Register` never asks about:
  `a replay update the driver refuses fails loudly rather than launching stale nodes` at
  size three, and `the unprobed reverse pair of a two-member group fails loudly` at size
  two, the shape the coarse key makes ordinary. The stronger fix is to
  probe `group.current_raw` instead of `raws.front()`, which removes the assumption
  entirely; it is **not** taken here because it changes probe behaviour while the device
  A/B (below) is measuring this exact commit. It is owed.
- **One registry serves every model on a device, and it is unsynchronised.**
  `CudaBackend::dedup_` is a member of a process-singleton per-device backend
  (`src/vt/cuda/cuda_backend.cu:355-365`), so the registry — and its two plain
  `unordered_map`s — is shared by every model loaded onto that device. The constraint is
  therefore wider than the "one runner thread" note above implies: a second model
  capturing or destroying a graph concurrently with the first races the containers
  themselves, before any question about a shared executable arises. Correct today
  because the engine drives one device from one thread. A lock belongs at the registry
  the moment that stops being true, and `ENG-CUDAGRAPH-BREAK` must not widen who
  captures without adding one.
- **A file allowed to see the runtime fail must consume the runtime's latched error,
  and this one did not.** [#1184](https://github.com/mudler/vllm.cpp/issues/1184). The
  whole safety argument above rests on a refused `cudaGraphExecUpdate` being NORMAL —
  the signature is a lookup key, the driver is the authority, and a refusal means "do
  not fold". A CUDA call that fails also LATCHES its code in the runtime's sticky
  per-thread slot, and none of the twelve fallible calls in `graph_dedup_runtime.h`
  consumed it. The rest of this tree launches kernels with the ordinary
  `kernel<<<>>>(...); Check(cudaGetLastError())` pattern, so the next unrelated kernel
  read our routine refusal and reported it as its own failure. On GB10 that was 6/6
  deterministic:

      vt graph dedup: captured 1 graphs, deduped to 1 execs
      [Qwen3DenseDecodeGraph] captured dense decode graph for padded size S=8 (real B=8)
      engine-fatal: EngineCore busy loop threw:
        vt cuda: greedy_argmax launch: invalid device function

  from a `greedy_argmax` launch that had SUCCEEDED. Every symptom follows from the
  mechanism and none of them from `greedy_argmax`: it needs both `dedup=1` and a real
  capture (no capture, no probe, no latch); `CUDA_LAUNCH_BLOCKING=1` does not move it,
  because the latch is host-side and synchronous rather than a deferred async error; and
  `cudaGraphLaunch` itself returns success, because reading a return value does not
  consume the latch. The two sites that mattered most were the ones whose failure is by
  design — the `cudaGraphExecUpdate` probe, and the `cudaGraphInstantiate` the earlier
  review made fatal — which is the general shape: the more deliberate a "this may fail"
  path is, the more reliably it poisons the next caller.

  **The repair is a scope guard, not twelve clears.** Twelve hand-placed
  `cudaGetLastError()` calls are a fix the thirteenth fallible call silently misses, and
  a file whose design is "these calls are allowed to fail" will grow a thirteenth. The
  clear therefore lives in `ScopedLatchClear`'s destructor
  (`src/vt/graph_dedup_latch.h`), installed at the binding's entry points, which are
  exactly the six `GraphDedupOps` members. Every exit runs it: a plain return, a
  degradation escape, and the `VT_CHECK` unwinding out of `Launch`. `MakeLatchGuardedOps`
  is the only constructor of the table and takes the raw functions as template
  arguments, so no raw address reaches a field; a seventh operation wired anywhere else
  leaves its field null and `GraphDedupRegistry`'s constructor refuses the table. One
  line covers both arms because there is one source: `VTGD_FN(GetLastError)` resolves to
  `cudaGetLastError` or `hipGetLastError`.

  **What the CPU suite proves and what it does not.** A CPU test drives a fake runtime,
  so it cannot observe the CUDA runtime's real latched-error state and it cannot prove
  #1184 is gone on a device. It proves the structure the fix rests on, red-first: with
  the destructor emptied to the pre-fix state the suite reports 22 failed assertions
  including `CHECK_NOTHROW(NextUnrelatedKernelLaunch()) THREW exception: "greedy_argmax
  launch: invalid device function"`, which is the production message reproduced from the
  mechanism alone. The device half stays owed.

- **The signature builder had no executable coverage, and now has most of one.**
  Previously the 13 cases in `tests/vt/test_graph_dedup.cpp` gated `GraphDedupRegistry`
  and nothing else, and all of `src/vt/graph_dedup_runtime.h` was reached by no test on
  any tier — the gap that hid #1184 for a whole review cycle. The device-free half is now
  `src/vt/graph_dedup_signature.h` and is driven by `tests/vt/test_graph_dedup_runtime.cpp`
  over a fake `Rt`: the Kahn ordering, the topological re-index, the sorted edge
  emission, the child-graph depth bound and the four graph-level degradation escapes.
  Seven negative mutations were detected, each with its `git diff --stat`, compile status
  and exit status recorded; one first attempt failed to build under `-Werror` and was
  re-run in a compiling form, because a mutation that fails to build reads as a passing
  test. **Still uncovered on any tier:** the five node-payload cases (kernel, memcpy,
  memset, child graph, default) and their four query escapes, which read CUDA parameter
  structs and stay behind the policy. And the silent mode the original note named is
  unchanged in kind: a signature that is unstable for some real topology folds nothing,
  every CPU test stays green, and the only observable is the device-side
  `"captured N graphs, deduped to N execs"` line. What would close it is still
  device-tier — capture one trivial graph twice and assert the signatures are byte-equal,
  capture two different topologies and assert they differ — so both remain owed with the
  A/B.

- **The shipped async serving path never reaches this feature, which bounds the row's
  value.** Measured during the #1184 device gate: `vllm-bench` on the async path captures
  NO decode graph at all, because `DenseDecodeGraphForward` returns `nullopt` whenever
  `input.device_token_ids != nullptr` (`src/vllm/model_executor/models/qwen3.cpp`, the
  [#323](https://github.com/mudler/vllm.cpp/issues/323) mitigation that
  [#1179](https://github.com/mudler/vllm.cpp/issues/1179) re-derived). No capture means
  no registration, so dedup engages only under `VT_ASYNC_RUNNER=0` — which is not the
  default and not what a user serves with. This does not change the row's correctness
  argument and it does change its worth: the executables this row folds are, on the
  default path, executables that are never instantiated in the first place. The saving is
  real only for the drivers and configurations that still capture. Recorded here rather
  than filed separately because the fix is `StepDevInputs`-shaped and already owned by
  `ENG-CUDAGRAPH-BREAK` / #1179; this row must not be sold on a number the default path
  cannot produce.
- **Two runtime-API calls changed shape across CUDA major versions, and only one was
  foreseen.** `cudaGraphExecUpdate` took the 3-argument `cudaGraphExecUpdateResultInfo`
  form at CUDA 12, so both it and the legacy 4-argument form are bound; HIP has only
  the 4-argument form. `cudaGraphGetEdges` gained a fifth `cudaGraphEdgeData*`
  parameter at **CUDA 13**, which was NOT foreseen: the local syntax check ran against
  CUDA 12.9 headers and passed, and `cuda-fat-build` on `nvidia/cuda:13.3.0` is what
  reported it (4 errors, `graph_dedup_runtime.h:161,168`). Both shapes are now bound in
  one `GetEdges` helper so the topology walk stays one piece of code, and the 5-argument
  shape is verified locally against CUDA 12.9's identical `cudaGraphGetEdges_v2`. The
  general lesson for this file: a header-shape check is only as current as the toolkit
  it ran against, and the CI job is the gate, not the local proxy.

## Outcome

The device gate ran on 2026-08-18 on `dgx:gpu0` through an `rc` lease (job `f88d484b`,
pod `rc-worker-4b8lj`, boot_id `1cf6179f-0150-4052-b507-506fd6751953`), GB10, driver
`580.173.02`, nvcc `13.0.88`, runtime cuBLASLt the staged cu130
`/tmp/tsite/nvidia/cu13/lib/libcublasLt.so.13`. Twelve cells of one binary
(sha256 `e166ed8d…7666fb`), `VT_ASYNC_RUNNER=0` and `VT_DECODE_GRAPH_STATS=1`
throughout, the only variable `VT_CUDA_GRAPH_DEDUP`. Recipe, hashes, per-cell table and
caveats: [`.agents/benchmark-record.md`](../benchmark-record.md), entry
`ENG-CUDAGRAPH-DEDUP W4`.

**The commit gated is `72de552c8`, not the merge `2a976eb9f`.** The row squashed, so the
gated tree is not an ancestor of what landed. What carries the claim instead is a
content equality that was checked rather than assumed: all four dedup sources —
`src/vt/graph_dedup.h`, `graph_dedup_runtime.h`, `graph_dedup_latch.h` and
`graph_dedup_signature.h` — are byte-identical at the two commits.

### What was measured

1. **[#1184](https://github.com/mudler/vllm.cpp/issues/1184) is fixed on the device.**
   All 12 cells exit 0. `grep -c "invalid device function"` and `grep -c "engine-fatal"`
   return zero in every cell log. On the pre-fix head `e4ce5571a` every `dedup=1` cell
   died after exactly one replay. The ON arms now replay as often as the OFF arms:
   60 = 60 on workload B, 33 = 33 on A, 43 = 43 on C.
2. **Byte-identity holds, 10/10.** The OFF/OFF controls passed first — `a_off_a ==
   a_off_b`, `b_off_a == b_off_b`, `c_off_a == c_off_b` — and then every OFF-vs-ON pair
   was identical. The artifact is `--output-token-ids`, real generated identifiers, 0
   empty rows, 960 / 672 / 1176 tokens. The three workloads hash to three DIFFERENT
   values (`d3b7028b…`, `02a1add6…`, `4f8714db…`), so the identity is not vacuous.
3. **The fold never happens.** `N == M` in every ON cell, now with two and three
   *distinct* padded buckets per process, where the first attempt managed only one and
   could therefore prove nothing:

```text
a_on_a / a_on_b: captured 3 graphs, deduped to 3 execs   sizes=[24 16 8]
b_on_a / b_on_b: captured 2 graphs, deduped to 2 execs   sizes=[16 8]
c_on_a:          captured 2 graphs, deduped to 2 execs   sizes=[32 24]
```

The registry's own line climbs `1 -> 1`, `2 -> 2`, `3 -> 3`, which is the proof that
more than one capture reached it.

### What was refuted, and why the row's premise was wrong

The cause is structural rather than a tuning miss. `AppendKernelPayload` hashes
`(func, gridDim.{x,y,z}, blockDim.{x,y,z}, sharedMemBytes)`
(`src/vt/graph_dedup_runtime.h:121-128`) and the memcpy payload hashes the copy extent.
The padded batch dimension is in both, so two decode buckets never share a key, no
candidate group ever forms, and `cudaGraphExecUpdate` is **never attempted**.

This refutes the row's own premise. `graph_dedup.h`'s header comment says the fold
exists because "the decode graphs of two padded batch sizes are usually the same node
topology with different parameters", and the signature as written cannot group exactly
those. SGLang hashes the same fields (`cuda_graph_dedup_mixin.py:105-114`), so whatever
folds upstream is not decode buckets either. The machinery is correct and does nothing
on the workload it was built for.

### Why the defaults have their values

`VT_CUDA_GRAPH_DEDUP` stays **OFF**. The flip was gated on this A/B, and the A/B says
the ON arm allocates exactly as many executables as the OFF arm on the only driver
exercised. A default is a measurement, and this measurement does not support one.

**No throughput or memory number is recorded, on any axis.** Two independent reasons,
and either alone is sufficient: the clocks were not pinned (208 MHz idle, 3003 MHz max,
2418 MHz applications), and dedup allocated the same number of executables as OFF, so
there is no memory delta to claim. The `replay branch avg` figures in the logs
(0.033-0.120 ms/step) are diagnostics, not a measurement.

### Honest gaps in this run

- **Per-shape replay counts are unavailable.** The driver prints a total only.
  Workload B's ~30 replays per shape is arithmetic over that total (60 replays, 2
  shapes), stated as arithmetic and not as a measurement.
- **The driver's "N captured size(s)" line counts SLOTS, not captures.** Workload A
  reports 6 sizes and emits only 3 `captured … padded size S=` lines.
- **The container's own cuBLASLt was never re-tested at CUDA 13.0.** The smoke probe
  tried the staged cu130 prefix first and it worked, so the original reason for the shim
  — a 13.6.0.2 cuBLASLt that could not capture — may no longer apply.
- **Only the Qwen3 dense decode driver was exercised.** Whether any other capture site,
  or two models sharing the process-singleton registry, can produce a fold is untested.
- **The in-pod toolkit changed under us**, from 13.3.73 at the first attempt to 13.0.88
  here, on the same pod and the same boot, because a neighbouring session held the box
  first. A job that needs a known toolkit must assert it rather than assume it.
- **The supporting `orin:gpu0` lane is BLOCKED**, cleanly: the Jetson `540.4.0` driver
  cannot run a CUDA 13 runtime (`cudaGetDeviceCount err=35`). A CUDA 12.x toolkit is the
  untried route. It is recorded, not pursued, because the dgx gate answered the question
  orin was there to support.

### W5, 2026-08-18: the coarse key, and the hypothesis W4 left open

The W4 result above named one traceable hypothesis: `cudaGraphExecUpdate` requires
*topology* to match and is documented to permit *parameter* changes, and a kernel's
launch configuration is a parameter, so a key that drops the launch dimensions might
group two decode buckets the exact key cannot. The probe-before-fold design makes that a
COST question rather than a safety question — a key that groups two graphs the driver
then refuses costs one wasted probe and a private executable, never a wrong replay.

**The hypothesis is CONFIRMED.** The experiment ran on `dgx:gpu0` through an `rc` lease,
pod `rc-worker-4b8lj`, boot_id `3fd9745a-d25a-426c-ba3c-97c958a85515` at both ends, GB10,
driver `580.173.02`, `### DONE_AB_KEY 2026-08-18T20:58:46Z`. Thirteen cells of one binary
(sha256 `ca114abb…c772ad`) built from `b48b51df1` (tar sha256 `7aa50f3b…2ad`, asserted
before extraction), `VT_ASYNC_RUNNER=0` and `VT_DECODE_GRAPH_STATS=1` throughout, the only
variables `VT_CUDA_GRAPH_DEDUP` and `VT_CUDA_GRAPH_DEDUP_COARSE_KEY`. Recipe, cell table
and caveats: [`.agents/benchmark-record.md`](../benchmark-record.md), entry
`ENG-CUDAGRAPH-DEDUP W5`.

```text
a_exact:    captured 3 graphs, deduped to 3 execs  (probes=0 refused=0)
a_coarse_a: captured 3 graphs, deduped to 2 execs  (probes=1 refused=0)
a_coarse_b: captured 3 graphs, deduped to 2 execs  (probes=1 refused=0)
b_exact:    captured 2 graphs, deduped to 2 execs  (probes=0 refused=0)
b_coarse_a: captured 2 graphs, deduped to 1 execs  (probes=1 refused=0)
c_exact:    captured 2 graphs, deduped to 2 execs  (probes=0 refused=0)
c_coarse_a: captured 2 graphs, deduped to 1 execs  (probes=1 refused=0)
```

**`probes=0` on every exact-key cell is the direct proof of W4's diagnosis, not a
restatement of it.** W4 concluded from source that with the launch dimensions in the key
no candidate group ever forms and `cudaGraphExecUpdate` is never asked. The probe counter
says exactly that, from the running process. Drop those fields and the driver is asked
once per fold and accepts every time. So on this hardware the driver does tolerate a
grid-dimension change across an exec update, and the benefit W4 recorded as unreachable is
reachable via the key.

**`refused=0` is one driver on one hardware and toolkit combination.** AGENTS.md forbids
declaring a ceiling; it equally forbids declaring a floor from one box. Nothing here says
`cudaGraphExecUpdate` accepts a grid change on another driver, another architecture, or
another CUDA release, and a refusal elsewhere is a cost, not a correctness failure.

**Byte-identity holds on A and C.** Workload A (`--concurrency 24 --num-prompts 24
--output-len 40 --seed 4242 --max-num-batched-tokens 128`, captured sizes 24/16/8): all
five cells `ids_sha256=59ebff4a22362086b9fcf426feba847b15b0dc7bf31b37892b1bfc7221b435dc`,
3655 bytes, 960 tokens, 0 empty rows. Workload C (`--concurrency 32 --num-prompts 49
--output-len 24 --seed 31337`, captured sizes 32/24): all four cells
`ids_sha256=ff205260c1fa41a8cc10ad1b8fd0c29f40bd6a2324aa8a64f5192f62cedbd76b`, 4608 bytes,
1176 tokens.

**Workload B is VOID, and the reason is a NEW DEFECT that is not this row's.** The two
OFF/OFF control cells — `VT_CUDA_GRAPH_DEDUP` unset in both — disagreed:

```text
*** DIFFERS *** b_off_a vs b_off_b
b_off_a  = 5973c5a10a6210085417fb25a29edbd0dc15fe61d7d4f774dd8ff3883dae1d64  bytes=2638
b_off_b  = 4cf7923080db6aa29759537f2192f3d9500db11c0e8d72bbb2b4ac6e4614af7c  bytes=2650
```

Same binary, same workload (`--concurrency 16 --num-prompts 21 --output-len 32 --seed
777`, greedy `--temperature 0`), 23 seconds apart, different token ids. Both runs emit 672
tokens with 0 empty rows, so the 2638-vs-2650 byte delta is JSON decimal width and **not**
a length difference; exactly two of 21 rows differ, rows 17 and 18, and both diverge
mid-decode (at token 11 and token 5) rather than at the first token. Those two rows are in
the ragged tail that `21 % 16` produces. B's `b_off_a == b_exact` and `b_off_a ==
b_coarse_a` lines therefore compare against a baseline that does not reproduce itself and
are worth nothing: **B is VOID, not a pass.** Only the OFF/OFF control makes that visible;
without it B would have read as three more confirmations. Filed as
[#1283](https://github.com/mudler/vllm.cpp/issues/1283), which is a decode-path defect
unrelated to dedup — the feature is off in both control cells — and is NOT diagnosed here
beyond what the evidence shows.

**The toolkit is not the one W4 ran on.** W5 built under nvcc **`13.3.73`**; the completed
baseline gate in `fe24a3029` ran under **`13.0.88`**, which is the version the recorded dgx
gate stack names. The pod's toolkit is a property of the pod, not of the job. The OFF-vs-ON
and EXACT-vs-COARSE comparisons **within this one binary** are valid, and **this run and
that baseline are not directly comparable to each other.**

**Nothing on `main` folds today.** The coarse key lives behind `VT_CUDA_GRAPH_DEDUP_COARSE_KEY`,
default OFF, inside `VT_CUDA_GRAPH_DEDUP`, itself default OFF, on **PR #1232, still a
draft**. This record measures a configuration that has not landed.

**No throughput and no memory number is claimed or implied, on any axis.** The clocks were
not pinned (2405 MHz current, 3003 max, 2418 applications, 53 °C) and nothing in this run
measured bytes saved — the fold count is a count of executables, not of memory. The
`Benchmark duration` figures (2.52-2.72 s) and the `replay branch avg` figures
(0.038-0.124 ms/step) in the logs are diagnostics.

**Only the Qwen3 dense decode driver was exercised**, as in W4. Whether any other capture
site folds under either key is untested.

### W6, 2026-08-19: the DEVICE-BYTE measurement, and the topology finding that inverts the premise

This is the measurement W4 and W5 both recorded as owed. Every number before it counted
executables; the row's stated saving is a MEMORY saving. It also removes W5's largest
caveat: **PR #1232 landed as `2c8f53d93`**, so this run measures a configuration that
ships rather than a draft.

Ran on `dgx:gpu0` through an `rc` lease, job `93f783de-228f-47d5-806d-c5b56aa72c3a`, pod
`rc-worker-4b8lj`, boot_id `3fd9745a-d25a-426c-ba3c-97c958a85515` at BOTH ends, GB10,
driver `580.173.02`, nvcc **13.0.88**, `### DONE_BYTES 2026-08-19T04:57:19Z`. Twelve
cells of one binary (sha256 `be697268…0ce657a7`) built from `origin/main` `2c8f53d93`
(tar sha256 asserted before extraction), `VT_ASYNC_RUNNER=0` and
`VT_DECODE_GRAPH_STATS=1` throughout, the only variables `VT_CUDA_GRAPH_DEDUP` and
`VT_CUDA_GRAPH_DEDUP_COARSE_KEY`. All 12 cells exit 0 with zero VOID markers. Recipe,
instruments, cell tables and every caveat:
[`.agents/benchmark-record.md`](../benchmark-record.md), entry `ENG-CUDAGRAPH-DEDUP W6`.

**The bucket churn is the shipped one.** `vllm-bench` sets
`max_num_seqs = concurrency`, so W32 captured `[1 2 4 8 16 24 32]`, **7 of 7**, and W64
captured `[1 2 4 8 16 24 32 40 48 56 64]`, **11 of 11** — exactly
`include/vllm/model_executor/models/decode_graph_sizes.h:32-41`. W5 reached 2-3 buckets
and every earlier conclusion about probe behaviour came from a process that presented
the driver with one pair.

**The fold engages.**

```text
w32_coarse_a/b  captured 7 graphs,  deduped to 3 execs   (probes=7  refused=3)
w32_exact_a     captured 7 graphs,  deduped to 7 execs   (probes=0  refused=0)
w64_coarse_a/b  captured 11 graphs, deduped to 5 execs   (probes=22 refused=16)
w64_exact_a     captured 11 graphs, deduped to 11 execs  (probes=0  refused=0)
```

Token ids are byte-identical across every cell of a workload, including both OFF/OFF
controls (W32 `ff0db6c6…be9d`, 11720 B; W64 `e1cbf5fc…e5d0`, 57620 B). Neither workload
has [#1283](https://github.com/mudler/vllm.cpp/issues/1283)'s ragged-tail shape, and
neither hit it.

**The saving does not survive its own null control.** Two instruments, and they agree.
`nvidia-smi --query-compute-apps` per-process tail median shows W64 identical to the
megabyte in all five cells (9737 MiB) and W32's coarse arm reading 10-23 MiB *higher*
than OFF (3252/3262 OFF against 3262/3275 COARSE). A `cudaMemGetInfo` shim summed over
every `cudaGraphInstantiate` gives a nominal **13.83 MiB** at 7 buckets — **0.42%** of
the 3.25 GiB process — and **−0.75 MiB**, i.e. nothing, at 11.

That nominal effect is **NOT ESTABLISHED**, on four grounds that are independent of each
other. `EXACT` is a true null control — it retains exactly as many executables as OFF (7
and 11) at `probes=0` — and it disagrees with OFF by 10.6-13.1 MiB, against a candidate
effect of 13.83 MiB. The W64 OFF/OFF pair disagrees with ITSELF by 18.2 MiB. One
instantiate recorded a NEGATIVE delta (`-5,165,056` B in `w32_exact_a`), which is only
possible if something else released memory inside the same window. And
`cudaGraphExecDestroy` returned `reclaimed_bytes=0` in every cell. Per-instantiate
deltas for byte-identical 404-node graphs range from 0 to 10,514,432 B, and 17 of 27
instantiates in `w64_coarse_a` read exactly zero: these are POOL-GRANULAR readings, and
the coarse arm's throwaway probe executables grow that pool exactly like retained ones
do. What can be priced is one ~390-node executable at **2.08-4.35 MiB**, or 10.0-10.6 KB
per node.

**THE MECHANISM, and why it inverts this row's own hypothesis.** The driver refuses
**43% of probes at 7 buckets and 73% at 11**, and every single refusal is
`vt graph dedup: probe refused a fold (err=910 result=2)` —
`cudaErrorGraphExecUpdateFailure` with `cudaGraphExecUpdateErrorTopologyChanged`. The
shim's `cudaGraphGetNodes` reading says why false candidates form at all: the decode
graphs are **two topologies, 376 and 404 nodes**, mixed across the bucket set
(`w32_off_a` captured `404 404 376 376 404 404 404`).

Every refusal is about TOPOLOGY, never about a parameter. **So a coarser key produces
more false hits, not more folds** — the opposite of what W5's 2-bucket A/B suggested.
W5's "the driver accepts every time, `refused=0`" was an artefact of workloads whose
buckets only ever shrank, so exactly one pair was ever presented. This is the correction
W5's own `refused=0` caveat asked for, and it arrived from the same box.

**The probe cost, exact and clock-independent.** W32: OFF 7 instantiates / 0 updates
against COARSE **10** (3 retained + 7 probes) / 11 updates. W64: OFF 11 / 0 against
COARSE **27** (5 retained + 22 probes) / 28 updates — **2.45x the instantiate calls** to
retain 6 fewer executables. **Peak transient did NOT double**: in every ON cell the
live-bytes peak equals its end value, because `Register` destroys the probe executable
before it returns. The feared "double the peak to save the steady state" trade did not
occur.

**A replay-time re-point occurred, as ARITHMETIC over two printed totals.** `probes=7`
against 11 update calls and `probes=22` against 28 leaves 4 and 6 updates that are not
probes, over 88 and 244 replays — the reverse re-point W5 recorded as untested on a
device. Those cells exit 0 with byte-identical ids, so `Replay`'s transitivity
assumption neither aborted nor changed a token here. It is stated as arithmetic and NOT
as a measurement, because the registry counts probe refusals only and a replay-time
refusal would abort on `VT_CHECK` rather than increment anything.

**Times: NONE is attributable.** The clock pin was REFUSED inside the lease — `The
current user does not have permission to change clocks for GPU 0000000F:01:00.0`, and
the log records `clocks_pinned=0`. The instantiate-wall and update-wall figures in
`bytes.log` are diagnostics and are deliberately quoted nowhere as a result.

### W6: why the defaults keep their values

1. **`VT_CUDA_GRAPH_DEDUP` stays OFF — supported by numbers, not merely ungated.** Best
   nominal saving 13.83 MiB (0.42% of process) at the 7-bucket set, smaller than its own
   null control's disagreement; nothing at 11 buckets; cost is 16 extra
   instantiate/destroy pairs and 28 update calls on the capture path.
2. **`VT_CUDA_GRAPH_DEDUP_COARSE_KEY` alone is a NO-OP, not merely unsupported.**
   `GraphDedupCoarseKeyEnabled()` (`src/vt/graph_dedup.h:114`) is read only by the
   signature builder (`src/vt/graph_dedup_runtime.h:177`), which runs only from
   `Register`, which `src/vt/cuda/cuda_backend.cu:237` calls only under
   `GraphDedupEnabled()`. With dedup off, its sole observable is one stderr line.
3. **Both on — not supported.** The same numbers plus a 73% probe-refusal rate.

### W6: what this run does NOT establish

- **No time-based figure is attributable**, because the clock pin was refused.
- **nvcc was 13.0.88 here and 13.3.73 for the W5 fold run.** Comparisons WITHIN this one
  binary are valid; this run and W5 are NOT directly comparable to each other.
- **`result=2` is one driver (`580.173.02`), one GB10, one toolkit.**
- **Only the Qwen3 dense decode driver was exercised**, as in W4 and W5.
- **`VT_ASYNC_RUNNER=0` throughout**, so the feature remains unreachable on the DEFAULT
  serving path ([#1179](https://github.com/mudler/vllm.cpp/issues/1179)).
- **`cudaMemGetInfo` cannot separate an executable's own cost from the pool chunk that
  satisfied it.** Every byte figure inherits that limit.
- `run.sh`'s per-cell `shim_inst … sum_bytes` reducer sums only POSITIVE deltas and
  reads higher in the two cells holding a negative one. The tables use the SIGNED sum,
  which matches the shim's in-process `SUMMARY` counter in every cell.

### W6: no ceiling — what would change the answer

- **Where the 376/404 node split comes from.** The FA-2 split-KV grid is the first
  suspect. A capture that fixes the node set across buckets removes every refusal in this
  run and turns the fold ratio into the whole bucket set. Until then, coarsening the key
  is the wrong direction.
- **An instrument that resolves a single 2-4 MiB executable** against driver pool
  granularity — `cuMemGetAllocationGranularity`, a pool-statistics query, or a build with
  pool instrumentation.
- **A 60-80 layer checkpoint.** Bytes scale with node count and this graph is 376-404
  nodes on a 28-layer 0.6B model. 10.0-10.6 KB per node is the figure to re-run against
  before the flip is refused permanently for large models.

## Owed

**DELIVERED by W4, 2026-08-18 — see [`## Outcome`](#outcome).** The device
byte-identity A/B: **PASS**, 10/10, OFF/OFF controls first. The `#1184` re-run
confirming that `VT_CUDA_GRAPH_DEDUP=1` completes a decode step on a device: **PASS**,
12/12 cells exit 0, zero `invalid device function`, zero `engine-fatal`. The
executable-count ratio: **DELIVERED AND NEGATIVE**, `N == M` in every ON cell over two
and three distinct padded buckets, so the row's headline saving is measured NOT to
occur with the current signature. What remains owed is below.

**DELIVERED by W6, 2026-08-19 — THE DECISION, and it is NEGATIVE.** The default flip
this row has owed since W4 is no longer owed: it is DECIDED, against, on measurement.
`VT_CUDA_GRAPH_DEDUP` stays OFF, `VT_CUDA_GRAPH_DEDUP_COARSE_KEY` stays OFF and is a
no-op on its own, and both together are unsupported. The two numbers W5 named as
missing were both taken. **Bytes:** 13.83 MiB nominal at the shipped 7-bucket set, 0.42%
of a 3.25 GiB process, smaller than the disagreement of its own null control, and
nothing at 11 buckets. **The probe cost at real bucket churn:** 43% of probes refused at
7 buckets and 73% at 11, 2.45x the instantiate calls, every refusal
`cudaGraphExecUpdateErrorTopologyChanged`. Two W5 items are closed as a side effect: PR
#1232 LANDED as `2c8f53d93`, and the replay-time re-point did occur (4 and 6 of them,
arithmetic over the update and probe totals) without an abort or a token change. W6 also
INVERTED this row's premise — the decode graphs are two topologies of 376 and 404 nodes,
so a coarser key produces more false candidates rather than more folds. See
[`## Outcome`](#outcome), W6. **What remains owed is below, and it is now three
measurements plus the residual engineering — never another attempt at the same flip.**

**[#1226](https://github.com/mudler/vllm.cpp/issues/1226) is CLOSED.** It owned the
coarse-key hypothesis, and W5 confirmed it while W6 priced it and inverted it. The rows
in the older tables below that still name it are history and are kept as such; the live
residue moved on 2026-08-19 to
[#1303](https://github.com/mudler/vllm.cpp/issues/1303), which owns the 376/404 topology
split, and to [#1162](https://github.com/mudler/vllm.cpp/issues/1162), which owns this
row and everything else still open on it.

| Item | Issue | Why not here |
|---|---|---|
| **Where the 376/404 node split comes from.** Every refusal in W6 was `cudaGraphExecUpdateErrorTopologyChanged`, and the node counts say the decode buckets genuinely differ in STRUCTURE, not only in parameters. The FA-2 split-KV grid is the first suspect. A capture that fixes the node set across buckets would remove every refusal and turn the fold ratio into the whole bucket set; until then, coarsening the key is measured to be the wrong direction | [#1303](https://github.com/mudler/vllm.cpp/issues/1303) | it is a NEW hypothesis about the capture path rather than about the key, needs its own spec and its own red-first evidence, and W6's job was to price the key that shipped |
| **An instrument that resolves a single 2-4 MiB executable against driver pool granularity.** Neither W6 instrument can: `cudaMemGetInfo` measures pool growth, 17 of 27 instantiates in one cell read exactly zero, one read NEGATIVE, and `cudaGraphExecDestroy` reclaimed nothing anywhere. `cuMemGetAllocationGranularity`, a driver-pool statistics query, or a build with pool instrumentation would price the fold directly instead of through a total whose noise exceeds the effect | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | it is an instrument-building task, and W6 deliberately reported the noisy total WITH its null control rather than presenting a 13.83 MiB number the control cannot support |
| **A 60-80 layer checkpoint.** Bytes scale with node count and W6's graph is 376-404 nodes on a 28-layer 0.6B model, measured at 10.0-10.6 KB per node. The flip is refused for THIS model class on THIS evidence; it is not refused permanently for large models until the same measurement runs on a deep one | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | it needs another lease and a large checkpoint, and refusing the flip for the measured case does not depend on it |

**DELIVERED by W5, 2026-08-18 — the [#1226](https://github.com/mudler/vllm.cpp/issues/1226)
hypothesis: CONFIRMED.** A coarse key folds every decode bucket the exact key left alone
(3→2, 2→1, 2→1) and the driver accepted every probe, `refused=0`. That closes the
*question*. It does not close the *decision*, and the two are now the whole of what this
row owes: **whether the coarse key becomes the default, which needs a MEMORY measurement
nobody has taken.** Every number so far counts executables. Nothing has measured bytes.

| Item | Issue | Why not here |
|---|---|---|
| ~~Flipping `VT_CUDA_GRAPH_DEDUP` on by default~~ **DECIDED AGAINST by W6, 2026-08-19.** It is no longer owed, and no longer merely unjustified: 13.83 MiB nominal (0.42% of process) at 7 buckets, nothing at 11, smaller than its own null control's disagreement, against 16 extra instantiate/destroy pairs and 28 update calls. History below | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | **STILL NOT JUSTIFIED, and the reason CHANGED.** W4's reason was that the ON arm allocated as many executables as OFF. W5 removed that reason: with the coarse key the ON arm allocates FEWER. The reason now is that nobody has priced the saving. A default is a measurement, and "2 execs instead of 3" is a count, not bytes. See the row below |
| ~~**THE DECISION, and the measurement it needs: does the COARSE key become the default?**~~ **DELIVERED AND NEGATIVE by W6, 2026-08-19.** Both numbers were taken at the SHIPPED bucket sets: bytes (0.42% at 7 buckets, none at 11, inside the null control's own noise) and the probe cost at real churn (43% and 73% refused, 2.45x the instantiate calls). The original wording follows.** Two numbers are missing and neither is optional. (1) **Bytes.** What does one `cudaGraphExec_t` cost on this model at this bucket set, and what does the fold actually return — device allocation at steady state, ON versus OFF, on a pinned-clock idle box. Every W4 and W5 figure counts executables. (2) **The probe cost at real bucket churn.** W5 probed once per fold on a 2-3 bucket process. A serving grid with 7 or 11 buckets across nine capture drivers churns differently, and the probe runs on the capture path. Only then is the flip a measurement rather than a preference | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | it needs a leased GPU, pinned clocks, and PR #1232 landed first — the key it would default to is still a draft. Deliberately NOT decided by this record, exactly as W5's own hypothesis was not decided by W4's |
| ~~Landing PR [#1232](https://github.com/mudler/vllm.cpp/pull/1232)~~ **LANDED as `2c8f53d93`, and W6 measured that exact commit.** The original wording follows. It is a DRAFT, so nothing on `main` folds today and this row's saving is unreachable on every configuration that ships | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | the experiment measured whether the key works before asking whether it should land, which is the correct order. It needs its own fresh review, and its red-first discrimination test — that the coarse key still SEPARATES two genuinely different topologies — is the load-bearing one |
| **PARTLY ANSWERED by W6, 2026-08-19:** 4 and 6 replay-time re-points occurred over 88 and 244 replays — arithmetic over the update and probe totals, not a counter — with every cell exiting 0 and byte-identical, so `Replay`'s transitivity assumption neither aborted nor changed a token. The counter is STILL owed, and so is a workload that alternates by design rather than by accident. Original wording: **An ALTERNATING workload, and a counter for replay-time re-points.** W5's three workloads captured their padded buckets in strictly descending order, so no group was ever replayed back to an earlier member and the only pair any coarse group issued was the one the probe tested. The reverse re-point — the pair that is untested from group size two onwards — is therefore unmeasured on a device, and `refused=0` does not cover it. The registry counts PROBE refusals only, so the ask is not even observable today: a replay-time refusal aborts on `Replay`'s `VT_CHECK` instead of incrementing anything | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | it needs a workload whose decode batch returns to a bucket it already captured, which the fixed-pool bench harness does not produce, plus a replay-side counter. Both ride with the lease the memory measurement above already needs |
| A second driver, a second architecture, a second CUDA release. `refused=0` is one driver (`580.173.02`) on one GB10 under nvcc `13.3.73`. A refusal elsewhere costs a wasted probe rather than a wrong replay, but the rate is unknown and the default decision depends on it | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | no second CUDA box with a different driver is reachable from this session, and `orin:gpu0` stays BLOCKED on the CUDA 13 runtime |
| **[#1283](https://github.com/mudler/vllm.cpp/issues/1283): greedy decode is not reproducible at concurrency 16.** W5's OFF/OFF control caught it and it VOIDED workload B. NOT this row's defect — `VT_CUDA_GRAPH_DEDUP` is unset in both cells that disagree — but this row's future A/Bs cannot use that workload shape until it is understood | [#1283](https://github.com/mudler/vllm.cpp/issues/1283) | it is a decode-path defect with its own issue and its own isolation plan. Folding it into this row would hide it behind a graph feature it has nothing to do with |
| Probing `group.current_raw` rather than `raws.front()`, retiring the transitivity assumption above | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | it changes probe behaviour, and the device A/B is measuring the current one. Land it with the A/B rerun, not before |
| Executable coverage for the signature builder's DEVICE half — stability and discrimination on a real `cudaGraph_t`, plus the five node-payload cases and their query escapes | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | the device-free half is now covered by `tests/vt/test_graph_dedup_runtime.cpp`; what remains needs a real `cudaGraph_t`, so it rides with the leased box the A/B already needs |
| Reaching the feature from the DEFAULT serving path. Dedup engages only under `VT_ASYNC_RUNNER=0`, because the async path captures no decode graph at all ([#323](https://github.com/mudler/vllm.cpp/issues/323) mitigation) | [#1179](https://github.com/mudler/vllm.cpp/issues/1179) | the repair is the `StepDevInputs`-shaped one `ENG-CUDAGRAPH-BREAK` already owns; until it lands, this row's saving is unreachable on the configuration users serve with |
| The ROCm leg's compile and run verification | [#41](https://github.com/mudler/vllm.cpp/issues/41) | no ROCm hardware or `hipcc` is reachable from this session and CI has no ROCm job, so the HIP wiring is written against the same shared header but is compile-unverified |
| ~~**THE NEXT TRACEABLE HYPOTHESIS, not decided here.**~~ **ANSWERED by W5, 2026-08-18.** A coarser signature key — function addresses and topology kept, launch dimensions and memcpy extents dropped — does let two padded decode buckets form a candidate group, and the driver accepts the resulting `cudaGraphExecUpdate` every time on this hardware (`probes=1 refused=0` per fold, 3→2 / 2→1 / 2→1). It was a cost question and it remains one; what moved is that the cost is now the only open half. See [`## Outcome`](#outcome), W5 | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | DELIVERED. The red-first discrimination test and the landing of PR #1232 are tracked in the rows above |
| Whether ANY capture site can fold. W4 exercised only the Qwen3 dense decode driver. The other hand-rolled drivers, and two models sharing the process-singleton registry, are untested | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | the same lease that answers the key question answers this one, and answering it before the key is settled measures the wrong thing |
| A supporting device leg on `orin:gpu0` | [#1226](https://github.com/mudler/vllm.cpp/issues/1226) | BLOCKED, cleanly: the Jetson `540.4.0` driver cannot run a CUDA 13 runtime (`cudaGetDeviceCount err=35`). A CUDA 12.x toolkit is the untried route. Not needed now that the dgx gate answered |

## Stop conditions

- Stop and report if the device A/B shows any token divergence between the arms. Byte
  identity is the premise of the whole row, not a nice property of it.
- Stop if a measurement shows the per-switch update cost is material at the bucket
  churn a real serving grid produces. The row is then a memory-versus-latency trade to
  decide, not an optimization to enable.
- Do not extend this row to break points or coverage. That is
  [#1163](https://github.com/mudler/vllm.cpp/issues/1163) and it has an owner.
