# Startup profiling for the KV memory budget

Row: `KV-WARMUP-PROFILE`.
Owner: [#3046](https://github.com/mudler/vllm.cpp/issues/3046).
Design base: `e1948ecebde085907171abce41a7b2e7541fd7d8`.
Primary oracle: vLLM `e126687a9a828d513c01a07cd69f025f27d63280`.

GPU means graphics processing unit. KV means key and value. GDN means Gated
Delta Net. BF16 names the bfloat16 storage dtype. All memory quantities below
are integer bytes unless a displayed unit is named.

## Now

`INVENTORIED`. This commit supplies the missing spec and changes no runtime.
It does not establish helper readiness, a profiling implementation, or device
acceptance. The raw upstream observation gate below precedes implementation
selection. The operator owns device execution and final verification.

One pull request carries this spec before the eventual implementation, using
the repository default. This spec-only commit does not close #3046. Promote
the row only after its committed readiness proof and dependencies are real.
Update this spec's `## Now` and the owning inventory row together when that
state changes. No other row changes in this specification.

## Scope and history

Replace the 256-block fraction fallback with actual startup profiling through
the normal loader. Preserve explicit block-count precedence, absolute-byte
precedence, sampling, dtypes, provider selection, and existing shared seams.
Measure the resources used by the actual serving implementation. Do not use
checkpoint file size or a guessed headroom percentage as a profile.

Commit `31384c8b7841530c23956ebb032ddefb00bbe58f` added an explicit-fraction
warning and deliberately retained the fallback. Its
[warning spec](gpu-mem-util-inert.md) does not claim profiling. Commit
`72a8fa8abeac2ec8f79e59cb337dc15870e9919d` preserved the inventory placeholder.
Both were checked with `git log -S`. The old [sizing spec](kv-sizing.md)
describes the earlier pin and a free-memory formula. The active source below
supersedes that formula for this work. The inaccessible legacy #83 link is
provenance, not verified current ownership; #3046 owns this implementation.

The first admitted cell is a discrete gfx1100 device running the existing
Qwen3.5-0.8B BF16 safetensors text arm. Gate `auto`, `bfloat16`, and
`fp8_e4m3` KV modes separately. Automatic sizing must be reachable with the
fraction unset, through the production default configuration of that cell.
Do not ship an opt-in-only implementation as the completed default path.

Admission uses the actual execution device, allocator capability, loaded
weight representation, model profile capability, and resolved serving envelope.
An architecture string, host label, or device type alone is insufficient.
Keep unadmitted paths unchanged, with truthful existing fallback/refusal
reporting and explicit owed work. Do not widen other backend defaults.

## Source anchors

Local anchors refer to the design base. Upstream anchors refer to the full
primary revision above. Bind source bytes again before implementation and
before each gate; line numbers identify the audited versions, not moving main.

| Contract | Executing source |
|---|---|
| Public load | `include/vllm.h:819::vllm_engine_load`; `src/capi/vllm_c.cpp:575`; `LoadedEngine::FromModelDir` in `src/vllm/entrypoints/model_loader.cpp:2448` |
| Current sizing | `model_loader.cpp:1727::ResolveNumBlocks`; `:1792::MakeKVCacheResolved`; `:1878::ResolveMaxNumSeqs`; `:1931::ResolveMaxModelLen` |
| Construction order | `model_loader.cpp:2154` resolves capacity before runner creation; `src/vllm/v1/worker/gpu/runner.cpp:528` and `:633` allocate KV before model preparation |
| Shared execution | `include/vllm/model_executor/models/model_registry.h:613::ModelForwardInput`; `runner.cpp:3155::ModelRegistry::Forward`; `include/vllm/v1/worker/gpu/model_runner_base.h:32` defers profile methods |
| Actual cache cost | `src/vllm/v1/kv_cache_interface.cpp:241::KVBytesPerBlock`; `src/vllm/v1/core/kv_cache_utils.cpp:965::recurrent_state_bytes`; `src/vllm/v1/core/hybrid_kv_budget.cpp:15,91`; runner cache allocation/accessors at `:655–735` |
| Native memory observation | `include/vt/backend.h:123::DeviceMemoryInfo`; `src/vt/rocm/rocm_backend.hip:486`; `include/vllm/model_executor/models/device_pool.h:303–350`; `include/vt/arena.h::HighWater` |
| Allocations outside that seam | `src/vt/rocm/rocm_matmul_hipblaslt.hip:325–332,362–366`; `src/vt/rocm/rocm_grouped_gemm.hip:1086–1093`; `src/vt/grow_only_stream_scratch.h` |
| Upstream device and request | `vllm/v1/worker/gpu_worker.py:405–437`; `vllm/v1/worker/utils.py:516::request_memory`; `vllm/config/cache.py:111` |
| Upstream memory accounting | `vllm/utils/mem_utils.py:86::DeviceMemoryProfiler`, `:109::MemorySnapshot`, `:234::memory_profiling`; `gpu_worker.py:110::maybe_rocm_profiling_fallback`, `:510::determine_available_memory` |
| Upstream model profile | `vllm/v1/worker/gpu/model_runner.py:374::load_model`, `:695::_dummy_run`, `:851::profile_run`, `:896::profile_cudagraph_memory` |
| GDN profile warmup | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1054::_warmup_prefill_kernels`, `:1207`, `:1781` |
| Graph estimate and teardown | `vllm/v1/worker/gpu/cudagraph_utils.py:718::profile_cudagraph_memory`, `:825::_extrapolate_full_graph_memory`, `:834::_init_minimal_kv_cache_for_profiling`, `:863::_teardown_profiling_state` |
| Final upstream capacity | `vllm/v1/core/kv_cache_utils.py:1336::_get_kv_cache_bytes_per_block`, `:1415`, `:1458`, `:2282–2335`; `vllm/v1/engine/core.py:254–345` |
| Frontend reservation | `vllm/multimodal/gpu_ipc_memory.py:155::reserve_mm_ipc_gpu_memory`; called by `gpu_worker.py:542,666` |

Read the complete dependencies these functions execute. Include the actual
runner, allocator, graph mode, and model preparation path in the source binding.
The completed local oracle logs select V2. A later run selecting another runner
requires its own source binding before the gate proceeds.

## Gate 0: observe the unchanged upstream profile

Build a private observer before implementing native profiling. Keep upstream
source, installed runtime, model, input, defaults, and allocator configuration
unchanged. The observer needs fresh implementation, independent CPU mutation
review, and an operator CPU rerun before any device use.

Observe the actual objects consumed by the original calls. Transparent wrappers
and source-bound, code-filtered CPython frame observation are allowed where
operands exist only as locals. Preserve argument objects, return objects,
exceptions, generator yield/unwind behavior and call counts. Do not substitute
an engine, precompute an answer, invoke a second profile, or add device memory
queries inside the observed phase. Copy snapshot fields after their original
measurement; preserve raw and applied values where the ROCm fallback later
updates the profile result. Retain plain metadata, not live device objects.

Install a dedicated startup observer before the spawned worker's `init_device`.
The existing state-observer RPC installation follows LLM construction and is too
late to observe this startup. Bind the bootstrap and observed code objects to
the pinned source. Detect missing/failed installation at launcher finalization;
Python startup can catch an initialization exception and continue. Record and
restore prior tracing state without silently replacing another observer.

The raw record must contain:

1. Source/wheel/package, runtime image, device and model identities. Record the
   worker's actual visible index and its physical-device binding, never assumed
   device zero. Record PID, worker/rank, and initialization ordinal.
2. Requested and resolved model/weight/KV dtypes; block geometry; maximum tokens,
   requests and length; sampler/EOS; scheduler, multimodal and graph settings.
   Record resolved utilization, absolute bytes and block override, including
   unset values. Production uses utilization 0.92 and normal graph/default
   scheduling. Do not introduce `enforce_eager` or shorter budgets.
3. Every original initial, before-profile and after-profile snapshot: device,
   total/free memory, current/peak allocated, reserved, non-framework memory,
   and timestamp. Include actual model-load consumption and its boundaries.
4. Peak reset, model profile, sampler/pooler, applicable encoder and GDN warmup,
   synchronization, graph profiling, final cache initialization and real graph
   capture call order. Record the actual profile shapes and request partition.
5. Raw persistent consumption, transient headroom, negative-delta branch,
   reservation fallback, graph samples/estimate/applied value, frontend
   reservation, and the exact returned available bytes. Bind the values to the
   objects used by the executing calculation, not parsed rounded log messages.
6. Actual final cache allocation/configuration and successful generation through
   the same engine. Preserve diagnostics and exceptions before refusing an
   incomplete record. Publish no COMPLETE marker before all required files
   close successfully and an atomic non-overwriting publication succeeds.

Also record `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` and
`VLLM_ENABLE_STARTUP_PLAN`. At this pin they default to true and false,
respectively (`envs.py:2104,1882`). A startup plan can route sizing through an
absolute value (`worker/startup_plan.py:134`); record its before/after absolute
setting and detect that branch explicitly. Observe the original initial-memory
admission even when the later sizing takes the absolute branch. Do not mistake
that branch for a new measured fraction profile.

CPU tests use the real bound source definitions with fake external boundaries.
Prove native argument/return/exception identity and exact snapshot consumption.
Delete each real observer installation and final call site; substitute cached
snapshots, rounded values, another device, reordered resets, and omitted graph
or fallback fields. Exercise bootstrap failure, generator exceptions and tracing
interference/restoration. Each mutation must cause the intended refusal. Add a
detector-removal control and restore all source bytes after each case.

The operator then runs enabled, repeat, and disabled controls in fresh processes
under the device mutex, using the existing pinned environment and owned caches.
Keep source/configuration/output preservation distinct from expected allocator
variation. Reconstruct all arithmetic from the raw operands. Retain differing
operands and their causes; do not require identical free bytes or hide changes.
No native implementation selection proceeds until this gate is independently
reviewed and the operator verifies the raw record.

Existing completed captures prove execution, but expose rounded memory logs.
They do not supply Gate 0's exact operands. Their source audit is retained in
the operator's private `profile-gap-ownership/source-scope-audit` archive:
manifest SHA256 `e2e9ee6b3a8d487798df7e3860c298e4be5ea1c33e051435ee547a6a47adb7a9`.
No new memory measurement is claimed by this spec.

## Budget arithmetic and defaults

Mirror the active pin's calculation. Let `T` be initial total device memory,
`F0` initial free memory, and `u` the resolved fraction:

```text
requested = ceil(T * u)
require F0 >= requested
C_raw = F0 - F_after_profile
H = peak_allocated_after_profile - current_allocated_after_profile
C = C_raw
if ROCm and C_raw < 0:
    C = max(reserved_after_profile - reserved_initial, 0)
    emit the reference-shaped warning and record the fallback
else:
    require C_raw >= 0
non_KV = C + H
available_KV = requested - non_KV - graph_estimate_applied - frontend_reservation
```

All operations require checked ranges and units. Reject nonfinite explicit
fractions and values outside `(0,1]`. Preserve the C ABI's documented unset
sentinel and resolve an unset fraction to 0.92. Never coerce failed telemetry
to zero, wrap a negative budget, or convert an invalid count through a narrowing
cast. A failed query or impossible minimum allocation refuses before final KV.

Weight bytes and non-framework growth are diagnostics, not extra summands in
`non_KV`. The driver delta already includes persistent weight and external
allocations. The transient term adds peak demand that no longer remains live.
Keep allocated and reserved bytes separate. Framework-free memory is
`(total-free)-reserved` in the upstream snapshot. A native equivalent must name
its actual tracking basis rather than label backend allocations as PyTorch.

The baseline follows required device/context/distributed initialization and
precedes device weights. Resources already in that baseline are not newly
consumed bytes. The initial free-memory admission accounts for their occupancy.
This is a per-instance allocation budget, not a claim that total device usage,
including preexisting allocations, equals `T*u`.

Preserve final override precedence:

| Selected knob | Required behavior on an admitted path |
|---|---|
| Positive explicit block count | Keep that count literally. Validate its actual allocation/admission; never silently lower it to fit. The count overrides both other sizing values. |
| Positive absolute KV bytes | Ignore the fraction when selecting KV capacity. Use the complete cache-cost planner below. Run the original-shaped preparation/profile work even though it does not derive this budget. Apply applicable frontend reservation. |
| Neither override | Derive capacity from the measured fraction formula. The unset and explicitly selected default fraction reach the same calculation. |

Upstream's earlier initial-free admission still runs before the absolute-budget
branch. An absolute override does not bypass every startup check or the model
warmup. A block override alone does not suppress upstream profiling either.
Preserve those distinctions in native calls, diagnostics and tests.

For a block override, capacity admission uses the override's checked actual
cost, as upstream replaces its effective planning memory with the override cost.
Do not compare that count against an ignored absolute or profiled KV budget.
Initial device admission and real allocation failure still apply.

Absolute-byte sizing changes deliberately on the admitted hybrid arm. Today
the native divisor budgets paged attention alone. The new planner includes
recurrent cache storage, so the same absolute value can select fewer blocks.
This corrects the budget's meaning; it does not change block size, scheduler
policy, or the explicit-count override. Do not apply this change to unadmitted
backends by accident. State the correction in the eventual usage/PR description.

The ROCm negative-delta fallback is a measured lower bound, not a proof that
untracked consumption vanished. Preserve its warning and return semantics.
A hardware acceptance run that sees external memory release or contention is
invalid for idle acceptance and must be repeated without altering thresholds.

## Exact recurrent capacity plan

Use one immutable allocation plan for sizing and final storage. Bind each plan
entry to its group/layer, count, dtype, scale storage, shape and byte extent.
The active upstream `_get_kv_cache_bytes_per_block` and `KVCacheTensor` builder
budget the shared cache's actual storage. The native runner instead allocates
attention by block and recurrent state by resolved sequence slots. Preserve that
physical representation here and adapt its cost explicitly.

For a candidate block count `N`, define:

```text
A      = sum of actual paged attention bytes for one block across all layers
S(N)   = existing ResolveMaxNumSeqs for the model/config with N blocks
R(N)   = exact recurrent allocation bytes for S(N), from every MambaSpec/layer
X(N)   = any other cache-owned device storage not already counted in A or R
P(N)   = checked(N * A + R(N) + X(N))
```

`S(N)` retains the existing hybrid seat calculation, configured cap and model
limit. `R(N)` uses `recurrent_state_bytes` and the allocation's actual slot
width, including the speculative multiplier when that scope is later admitted.
No hardcoded conv/SSM pair or one-layer divisor is allowed. Do not include final
recurrent state in both `C` and `R`. Profile placeholders are disposable and
must be absent from the final-cost observation.

For fraction or absolute sizing, select the largest representable admissible
`N` for which `P(N) <= available_KV`. Do not reserve an unexplained maximum
recurrent envelope and call the resulting smaller cache exact.

The first admitted topology has `A>0`; `S(N)`, `R(N)` and `X(N)` must be
nondecreasing. Thus `P(N)` is strictly increasing. Prove those properties over
the declared shape/domain, including every seat transition. Use checked binary
search bounded by `floor(available_KV/A)` and the actual count type's maximum.
Verify both `P(N)<=available_KV` and `P(N+1)>available_KV`, unless `N` is the
maximum representable count. Reject overflow and any topology lacking the
required cost/monotonicity contract. Do not guess a result for attention-free
or unadmitted layouts.

The admissible domain includes the shared planner's usable-block and request
admission requirements. A reserved null block cannot serve tokens. The separate
[#2719](https://github.com/mudler/vllm.cpp/issues/2719) / `KV-SIZING` correction
remains a required compatible dependency wherever its admission bug applies.
Do not claim this profile fixes its block planning or bypass that gate.

Resolve final `S(N)`, length and batch capacity from the selected plan and
propagate them to runner inputs, recurrent slots and scheduler together. Keep
the current configured/default policies; changing capacity can change their
derived outputs, which must be logged. Explicit length still refuses when it
does not fit; automatic length reports its reduction. Never retain a length or
slot count calculated from the old 256-block probe.

Before allocating, validate the complete plan. After allocation, independently
sum actual allocator extents and inspect final cache storage. Require agreement
with the plan and the selected effective budget. The existing paged getter alone
is insufficient.
Include recurrent buffers and any cache scale/auxiliary allocation. A failure
unwinds owned storage and prevents serving-ready publication.

## Shared initialization and telemetry

Refactor the ordinary loader/runner into these phases, preserving the shared
`ModelRegistry::Load`, `Prepare`, `Forward`, attention and sampler paths:

1. Resolve the actual queue/device and profile capability. Initialize required
   device context, take the initial snapshot and validate the request.
2. Load and prepare the actual model. Carry a device-bound profile session
   through `FromModelDir`; do not start observation after lazy weight staging.
3. Resolve an initial serving envelope before capacity clamps. It must cover
   the largest tokens, requests, length-dependent buffers and applicable model
   work the final configuration can execute. Allocate non-cache persistent
   input/sampler resources and measure the real maximum-shape profile.
4. Estimate applicable graph resources using temporary cache/state ownership.
   Dispose of every profiling graph/cache safely, retaining only resources that
   the accounting explicitly treats as persistent serving resources.
5. Compute final capacity, length and concurrency. If any final shape or path
   exceeds the measured envelope, discard the plan and reprofile before final
   cache allocation. Bound convergence; refuse an unstable configuration instead
   of looping or publishing an unmeasured plan.
6. Allocate final storage, verify its exact cost, and initialize the real
   scheduler/cache/graph state. Restore sampler, RNG and request state before
   the first user request. Complete outstanding work before releasing resources.

The current runner interface defers dummy/profile execution. Extend that shared
interface and `ModelForwardInput` where needed. The profile must call the real
registry forward and sampler. Initial attention metadata suppression mirrors
the upstream profile; it must not skip model preparation or GDN warmup. Do not
implement a second model forward by hand.

Extend `vt::Backend` with a device-specific profile capability, snapshot and
peak-reset lifecycle. Unsupported is explicit. The ROCm free/total seam selects
its backend's device; do not query process-default device zero. Counter scopes
must identify device, stream ownership and load session. Serialize conflicting
profiles without resetting another engine's counters or releasing its storage.

Define native tracked current, peak and reservation bytes. Track owned
allocations across real allocator routes, including a temporary spike that is
freed before the final snapshot. Drain only releasable scratch after required
synchronization, mirroring upstream cache release. Count retained scratch,
including old graph-safe allocations, as still resident. Direct HIP and vendor
allocations outside the tracker remain visible in the driver delta. Do not
claim their transient peak was observed when it was not; retain the upstream
accounting limitation and test its persistent external-allocation case.

One arena's high-water mark does not cover the runner. Pool hit counts are not
allocated bytes. Requested tensor bytes are not necessarily allocator extents.
Record alignment/reservation overhead and verify the actual allocation basis.
A telemetry implementation that cannot establish its claimed coverage cannot
admit automatic profiling.

Direct in-memory construction cannot invent a pre-load baseline for caller-owned
device weights. Require a valid supplied load/accounting context for profiling
there, or keep that unadmitted construction's documented explicit sizing path.
Its fixture does not replace the required on-disk production loader test.

## Graph, model and configuration boundaries

At the design base, ROCm's shared static-graph policy is disabled. Native graph
reservation is zero only when the actual serving policy proves no capture.
Do not force capture merely to mimic the oracle's allocation size.

[#3041](https://github.com/mudler/vllm.cpp/issues/3041) separately activates
native graphs. Reconcile its actual landed/implementation seam before coding
and before default activation here. If the serving cell captures graphs, this
profile must measure that cell's graph resources and pass lifetime gates first.
An eager-only profiler cannot become dead code behind a newly graphed default.
Do not change #3041's admission or borrow its BF16 proof for quantized weights.

Mirror the upstream graph estimate's ownership: disposable minimum KV, separate
pool, actual configured captures, supported extrapolation, and full teardown.
Do not double-count shared pools or replay graphs referencing temporary caches.
If native graph capture populations differ, report and measure the actual native
population. Do not copy an upstream GiB estimate as a native constant.

The initial model binding is `Qwen/Qwen3.5-0.8B` at
`2fc06364715b967f1860aea9cf38778875588b17`. Resolve its path from the supplied
`CHECKPOINT_ROOT`. The BF16 shard is
`model.safetensors-00001-of-00001.safetensors`, 1,746,942,600 bytes, SHA256
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.
Its config SHA256 is
`b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
Rehash all consumed model, tokenizer and input files before device gates.

Native profiling follows the actually registered multimodal capability and
resolved limits. Do not infer that a text prompt removes upstream encoder or
frontend reservations. Record skipped native towers and unavailable model paths
truthfully. Adding a multimodal port belongs to its existing model row.

The completed upstream production controls use maximum tokens 8192, requests
256, length 262144, prefix caching and async scheduling enabled. Native dense
batch tokens default to 2048 at this base; other resolved controls also differ.
Do not change those defaults here or call equal requested fractions equivalent
configurations. [#2773](https://github.com/mudler/vllm.cpp/issues/2773) retains
the separate block-size, scheduler, format, state and numerical requirements.
No native/upstream tensor metric can bypass its strict control checks.

## Tests, mutations and gates

Every implementation starts red for its intended defect, then passes focused
tests and the full declared gate. A fresh reviewer independently mutates every
claimed guarantee, including real production call sites. Preserve each failed
attempt, compile status, test count, exact command and byte restoration.

| Gate | Required test and effective mutation |
|---|---|
| Public reachability | Load a real on-disk fixture through `vllm_engine_load`, then generate through the same engine. Observe baseline before weights, actual forward/sampler profile, and final allocation after sizing. Delete each production call; constant 256 or a no-op profile must fail. |
| Accounting | Port `tests/utils_/test_mem_utils.py:14–91`: 512 MiB baseline external bytes, 512 MiB weights, 1 GiB transient spike and 256 MiB new external bytes. Preserve exact peak and the source's 5% device-accounting tolerances. Mutate each summand, baseline boundary, allocation/reservation distinction and reset. |
| Snapshot modes | Port the source's discrete/UMA snapshot cases and `tests/v1/worker/test_gpu_worker.py:97–154` ROCm/non-ROCm fallback cases unchanged. Mutate device selection, negative delta, reservation shrink, warning and off-ROCm assertion. |
| Capacity | Check `P(N)` against an exhaustive small-domain oracle and actual final runner storage. Include a case where paged-only division fits but paged plus recurrent storage exceeds the budget. Mutate each layer/state/scale term, seat transition, maximality check, overflow guard and real allocation verification. |
| Overrides | Test all precedence combinations, unset/default fraction, invalid values, tiny and oversized budgets, explicit length and explicit count. Prove the admitted hybrid absolute-byte correction. Mutate the selected knob, silently clamp explicit N, or restore paged-only division. |
| Envelope | Test maximum tokens/requests, changed final length/slots, smaller final shapes, and reprofile when a final path exceeds the envelope. Mutate recorded shapes, reprofile/call ordering and convergence refusal. |
| Lifetime | Profile then serve a fresh request, repeat loading, and fail at each allocation/profile/capture/cleanup boundary. Detect dirty dummy requests, sampler/RNG state, cached snapshots and foreign-device counters. No failure publishes a complete profile or serving-ready engine. |
| Graphs | For every admitted graph mode, vary capture shapes/state/metadata and force recapture. Detect omitted graph cost, shared-pool double counting, temporary-graph replay, early scratch frees and teardown without completing outstanding work. |
| Physical formats | Verify actual model/KV/recurrent dtypes, bytes, layouts and scales for each admitted cache mode. Delete dtype propagation or widen a buffer; a token-only detector is insufficient. |

Each new detector needs a removal control proving that its assertion detects
the mutation. An unchanged source assertion or a planner checking its own
formula does not replace independent final storage/reachability evidence.

The CPU tier covers arithmetic, allocation-plan properties, actual native
loader/runner wiring using bounded fixtures, and private observer integrity.
It cannot prove HIP memory or provider execution. Preserve all existing knob,
group/layer, recurrent and default tests; update expectations only for the
explicitly specified admitted absolute-byte correction.

The operator's local gfx1100 gate requires Gate 0, reviewed native changes and
CPU gates first. Use a fresh process, the device mutex, an idle device, exact
source/binary/runtime/model hashes and the actual physical device identity.
Run every admitted cache mode, default and override paths, maximum serving
shapes, low-budget refusal, repeated cold/warm loads and final allocation checks.
Do not use another agent's build or run outside the operator's device scope.

Compare automatic sizing with the same native binary and explicit observed N.
Match all other resolved controls. Require token-exact output and unchanged
final state bytes where the layout/control join is valid; the first real
request must not inherit profile state. Preserve the consuming row's operation
and dtype gates. Do not change upstream tests, tolerances, goldens or sampling.

Retain initial/free/reserved/peak/final device memory, actual cache bytes,
startup time and steady throughput/latency. Reproduce any accepted comparison
at least three times on an idle host using the same native binary. Keep vLLM's
production configuration as the denominator. Inherited throughput, latency and
memory floors remain binding; an unmet axis stays an open gap. Profiling alone
is not performance acceptance or proof of equal upstream/native capacities.

For each implementation and review, run focused gates plus one full unchanged
candidate preflight with isolated Git/Python environment, external temporary
space, and explicit `scripts/check-tree-compiles.py --jobs 4`. Record every
skipped argument/resource check as PENDING. The operator reruns the applicable
gates independently. A spec-only gate does not run or satisfy product mutations.

## Owed and stop conditions

- #3046 owns the private raw observer gate, native telemetry, phased profile,
  exact cache planner, CPU/mutation gates and admitted gfx1100 validation.
  It also retains unadmitted CPU/CUDA/UMA/other-backend and model/quantized-arm
  profiling work. A bounded first slice leaves the whole row PARTIAL.
- Quantized weights require their own profile, lifetime, provider, byte and
  quant-matched oracle evidence. The existing 4B Q4_K_M arm remains required
  campaign work; BF16 evidence cannot admit it. Use the recorded oracle registry
  and its measured gateability, without creating a new oracle or denominator.
- #3041 owns graph activation; this row owns accounting for whichever graph
  scope serving actually admits. Reconcile both before automatic activation.
- #2719 / `KV-SIZING` owns the separate usable/null-block admission correction.
  #2773 owns the consuming state/configuration and numerical prerequisites.
  This spec does not close, weaken, or edit those rows.

Stop with `NEEDS_CONTEXT` for missing source/runtime/model bindings or unavailable
raw operands that affect the implementation. Stop with `NEEDS_DECISION` when a
required change exceeds this admission/accounting scope. Do not replace a
missing measurement with a constant or widen scope to make a gate convenient.

Refuse default activation on unknown allocation coverage, failed profile or
cleanup, nonmonotonic/unbound cost, arithmetic overflow, invalid final capacity,
unmeasured graph resources, or a serving envelope outside the measured one.
Preserve failure evidence before cleanup. Missing hardware remains PENDING.
Only the reviewed implementation and operator gates can change that result.
