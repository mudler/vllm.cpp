# ENG-CUDAGRAPH-DEDUP: one graph executable per topology, not one per bucket

**Row:** `ENG-CUDAGRAPH-DEDUP` in [engine-matrix.md](../engine-matrix.md).
**Issue:** [#1162](https://github.com/mudler/vllm.cpp/issues/1162).
**Analysis that derived this row:** [sglang-breakable-cuda-graph.md](sglang-breakable-cuda-graph.md), `## Owed`.
**Kind:** memory and capture-time optimization behind the `vt` graph seam. **Not a
throughput row.** A deduped replay launches exactly the same nodes with exactly the
same parameters; if it launched anything else it would be a correctness defect, which
is why the load-bearing gate below is byte-identity rather than a speed ratio.

## Now

`ACTIVE`. The shared registry, its CPU contract suite and the CUDA wiring land here.
The device-level byte-identity A/B and the ROCm leg are named under
[`## Owed`](#owed) with the issue that owns each.

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
4. An observable capture-count / exec-count log line and a programmatic `Stats()`
   accessor, so the ratio is a measurement rather than a claim.

Out of scope, each owned elsewhere:

- Break points and piecewise capture: `ENG-CUDAGRAPH-BREAK`,
  [#1163](https://github.com/mudler/vllm.cpp/issues/1163).
- The diffusion / denoise-loop capture: `ENG-CUDAGRAPH-DIFFUSION`,
  [#1164](https://github.com/mudler/vllm.cpp/issues/1164), blocked.
- Retiring the eight hand-rolled decode drivers. This row does not touch their
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
| `"captured %d CUDA graphs, deduped to %d execs"` | `:353-358` | `GraphDedupRegistry::Stats()` plus the log line |
| Retained raw graph (`torch.cuda.CUDAGraph(keep_graph=True)`) | `:326` | `EndCaptureGraph` stops calling `cudaGraphDestroy` when dedup is on |

Three deliberate divergences from the upstream construction, each forced by a
difference in our seam rather than chosen:

- **The probe executable is transient, not persistent.** SGLang holds a second
  `compat_exec` per group for the whole capture phase and frees it in `seal()`,
  which its backend calls at `end_cuda_graph_capture()` (`:352-358`). **Our seam has
  no capture-phase end.** The eight drivers capture lazily, the first time a padded
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
yields 7 padded decode buckets at `max_num_seqs=32` and 11 at 64, and eight drivers
each build their own set: `Qwen3_5DecodeGraph`, `Qwen3_5DenseDecodeGraph`,
`Qwen3MoeDecodeGraph`, `Qwen3DenseDecodeGraph`, `DeepseekV2DecodeGraph`,
`VoxtralDecodeGraph`, plus the graph code in
`src/vllm/model_executor/models/deepseek_v4.cpp:1900` and
`src/vllm/model_executor/models/laguna.cpp:2691`.

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
| `src/vt/graph_dedup.h` (new) | `GraphDedupOps` (six function pointers: signature, instantiate, update, destroy exec, destroy graph, launch) and `GraphDedupRegistry`. Header-only, no device dependency, CPU-unit-testable — the same shape `src/vt/cuda/graph_safe_scratch.h` used for the graph-safe scratch bookkeeping. Also `GraphDedupEnabled()`, the `VT_CUDA_GRAPH_DEDUP` read. |
| `src/vt/graph_dedup_runtime.h` (new) | The CUDA / HIP ops table, written **once** and bound to either runtime by a macro alias block. `GraphSignature()` lives here. This is the file that must not become two hand-written copies. |
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
| `folds compatible captures onto one executable` | K same-signature captures leave exactly 1 live exec; `Stats()` is `(K, 1)` | `:219-242`, `:253-256` |
| `keeps incompatible captures apart` | different signatures never share | `:225-241` |
| `replays the graph the caller asked for` | over an interleaved multi-replay sequence, the deduped launch log equals the non-deduped control log, element for element | `:258-275` |
| `does not re-update for a repeated shape` | a second consecutive replay of one handle issues no update | `:267` |
| `falls back to a private executable when the driver rejects the update` | a probe failure yields its own exec, the existing group's live exec is untouched, and replay stays correct | diverges from `:229`, stated above |
| `frees the shared executable only with its last graph` | destroying one sibling keeps the exec; destroying the last frees the exec and every retained raw graph, with all counters balanced | `:277-292` |
| `does not claim a handle it did not mint` | `Owns()` is false for a foreign pointer | ours; the seam has no upstream twin |
| `is off unless the environment asks for it` | `GraphDedupEnabled()` polarity | `:321` |

## Gates

1. **Red first.** Every case above is written and run against the un-implemented
   registry, and the red output is captured, before the implementation exists.
2. **Focused green.** `ctest -R test_graph_dedup`.
3. **Mutation.** Each guarantee is deleted or inverted in a scratch copy, the focused
   suite is proven to fail, and the tree is restored byte-for-byte. `git diff --stat`
   and the compile status are printed for every mutation, because a mutation that fails
   to build and a mutation that never applied both read as a passing test.
4. **Full gate.** `./scripts/agent-preflight.sh`.
5. **CUDA compile.** The `cuda-fat-build` CI job (`.github/workflows/ci.yml:669-712`,
   container `nvidia/cuda:13.3.0-devel-ubuntu24.04`) compiles the CUDA leg on the PR.
6. **Device byte-identity A/B (owed, see below).** Same binary, `VT_CUDA_GRAPH_DEDUP`
   off then on, identical prompts and sampling, token-exact equality, with **more than
   one replay per padded bucket** — a first replay can be correct by accident, and the
   whole hazard of a shared executable is that the *second* visit to a shape is the one
   that has to re-point it.
7. **Not gated, deliberately:** throughput. This row must not be sold as a speed
   change. The reportable numbers are executable count and capture wall time.

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
  comparison and the executable-count ratio.

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
- **Retaining the raw graph costs host memory.** A `cudaGraph_t` holds the node
  descriptions, not the device-side executable image; it is retained only while dedup is
  on, and it is released with its handle. The direction is safe with respect to
  [decode-graph-scratch-uaf-2026-07-18.md](decode-graph-scratch-uaf-2026-07-18.md),
  which is a premature-**free** hazard.
- **The signature can only be too coarse, never too fine, and coarse is safe.** Every
  hit is probed with the real driver call before it is honoured. This is deliberate:
  a signature that had to be exhaustive would be a correctness surface, and it is
  instead a lookup key.
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

## Owed

| Item | Issue | Why not here |
|---|---|---|
| The device byte-identity A/B and the executable-count measurement (W4) | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | needs a leased CUDA box; the CPU tier proves the registry's launch-sequence identity, not the device's |
| Flipping `VT_CUDA_GRAPH_DEDUP` on by default | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | gated on W4. A default is a measurement, not a preference |
| The ROCm leg's compile and run verification | [#41](https://github.com/mudler/vllm.cpp/issues/41) | no ROCm hardware or `hipcc` is reachable from this session and CI has no ROCm job, so the HIP wiring is written against the same shared header but is compile-unverified |

## Stop conditions

- Stop and report if the device A/B shows any token divergence between the arms. Byte
  identity is the premise of the whole row, not a nice property of it.
- Stop if a measurement shows the per-switch update cost is material at the bucket
  churn a real serving grid produces. The row is then a memory-versus-latency trade to
  decide, not an optimization to enable.
- Do not extend this row to break points or coverage. That is
  [#1163](https://github.com/mudler/vllm.cpp/issues/1163) and it has an owner.
