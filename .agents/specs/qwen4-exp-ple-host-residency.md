# `qwen4_exp`: a per-tensor host-residency policy for the PLE n-gram table

Row `MODEL-MM-QWEN4-EXP`. Issue
[`ISSUE-LOCAL-01M2DAFPJN13SKQDF3NAS849H2`](../issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2DAFPJN13SKQDF3NAS849H2.md).
Evidence
[`qwen4exp-ple-placement-gfx1151-20260913.md`](../../docs/bench-evidence/qwen4exp-ple-placement-gfx1151-20260913.md).

**This spec proposes. It implements nothing, and it predicts a gain of ZERO on
the axis the investigation was opened to improve.** That is the finding, and the
spec exists so the next wave does not re-derive it.

## The question this spec answers, and the answer

The investigation asked why llama.cpp's `-ngl 99` hybrid split decodes 13x
faster than forcing full `ROCm0` residency, so that vllm.cpp could port the
split deliberately.

**There is no split to port.** The 27,465.95 MiB llama.cpp leaves on the host is
one tensor, `per_layer_token_embd.weight`, and it is there because llama.cpp's
HIP `GET_ROWS` refuses an `IQ4_NL` source whose row length is not a multiple of
`QK_K` (`ggml/src/ggml-cuda/ggml-cuda.cu:5012-5016` at pin
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`; this row is 160, `QK_K` is 256).
**vllm.cpp already has the kernel llama.cpp lacks**
(`src/vt/rocm/rocm_embedding_quant.hip:51,142`, registered at
`src/vt/rocm/rocm_ops.hip:207`, #3093), so we are device-resident by capability
where llama.cpp is host-resident by refusal.

## Scope

**In scope.** A per-tensor residency override that can keep a named GGUF tensor
in host memory while the rest of the model loads onto the accelerator, and its
one first consumer, `per_layer_token_embd.weight` on `qwen4_exp`.

**Out of scope.** Any throughput claim. Any change to
`EmbeddingQuantKernelRocm`. Any change to the MoE placement seam. Any cross-engine
comparison, which the row may not make until it has a token-exact gate
([`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`](../issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG.md)).

## What the fix buys, stated so it can be falsified

| Axis | Prediction | How to falsify it |
|---|---|---|
| Device memory for this artifact on `gfx1151` | **-26.822 GiB**, from 71.96 GiB to about 45.1 GiB | Read the allocator total on both arms of a same-binary A/B |
| Decode throughput | **0%, within the +/-10% this board's instrument can resolve** | A same-binary A/B on `strix:gpu0` at `n=64`; a change outside that band falsifies the prediction in either direction |
| Load time | **Improves**, by whatever the 26.822 GiB chunked pinned H2D costs (`include/vt/rocm/rocm_pinned_h2d.h:121-139`) | `VT_LOAD_STATS` on both arms |
| Steps per second at large context or batch | **Improves or is unchanged**, because 26.822 GiB of freed carve is KV and activation headroom | Not measurable at `num_reqs = 1`, which is all this architecture serves today |

**The honest headline is the memory axis.** On a 96 GiB carve, 26.822 GiB is 28%
of the board, and this architecture's declared context is 262,144. The
throughput prediction is zero and is written down precisely so a wave that
measures a speedup knows it has found something this spec did not anticipate.

## Why it is not free, and why that matters

Today the gather is one ROCm kernel launch reading a device-resident table. Under
this proposal it becomes a host gather plus an H2D of the gathered rows. The
moved bytes per step are `T * heads * 160` elements, kilobytes, against 26.822
GiB of residency saved. But it adds a **host round trip on the decode critical
path**, and this architecture already serves one sequence per step
(`docs/FEATURES.md:163`), so the latency is not hidden behind anything.
**That is why the policy must be opt-in and must default OFF on a device that
registers `kEmbeddingQuant`.** A device that does not register it already
refuses the load by name (`qwen4_exp_weights.cpp:666-681`); this policy is what
would let such a device load instead of refuse, which is the second, larger
reason to build it.

## Design

### The seam, and why each candidate is or is not it

| Candidate | file:line | Verdict |
|---|---|---|
| `dense_attn::ResidentWeight` | `include/vllm/model_executor/models/dense_attn_block.h:181`, device arm `:231-245`, predicate `:206-220` | **This is the hook.** The predicate is a binary `plat.is_cpu()` on the *Dev handed in* and never reads the tensor name. One consultation of a name-keyed policy here is the whole placement half of the change. |
| `GgufLoadPolicy` | `include/vllm/model_executor/model_loader/gguf_keep_quant.h:431,449-450` | **This is where the policy belongs.** It already routes per tensor and per role, and already has one location-ish member, `ComputeDeviceFor(name, role)`, whose own comment says it differs for exactly one role. `kEmbeddingTable` would become the second. |
| `DevicePlacement::DeviceFor` + `PlacementOverride` | `include/vllm/config/weight_residency.h:119-127,141-147`; `include/vllm/model_executor/device_placement.h:65-68` | **Reuse it. Do not write a second regex matcher.** It already mirrors `-ot`: a pattern, a device, `regex_search`, first-match-wins, unsorted. What it lacks is a non-MoE consumer. |
| `MoePlacementPlan` | `device_placement.h:185,222` | **Not it.** Keyed on `blk.<N>.` and the three `ffn_{gate,up,down}_exps.weight` names. The PLE table has no `blk.` prefix and the gather is not an MoE block. |
| `KqGrouped`, `KqResidentSlice` | `src/vllm/model_executor/models/qwen3_5.cpp:6155,6018` | **Not it.** Compute helpers, same CPU-vs-device binary. |
| `include/vllm/config/offload.h` | | **Not it.** Upstream vLLM's device-to-host KV and weight offloader; `weight_residency.h:5-14` already records that it has nothing to mirror for per-tensor weight placement. |

### The three blockers, and what each costs

1. **The op contract refuses a split-device call.** `src/vt/ops.cpp:1766`:
   `VT_CHECK(table.device == out.device && ids.device == table.device && table.device == q.device, ...)`.
   A host table on a ROCm queue throws before dispatch. **Fix: do not break the
   invariant.** Wrap the call site instead — gather on a CPU queue, then H2D the
   `[T*heads, 160]` result. `PlacementQueue` (`device_placement.h:240`) already
   provides a process-lifetime CPU queue for exactly this shape of round trip.
   Call site to wrap: `src/vllm/model_executor/models/qwen4_exp_ple_block.cpp:494-497`.
2. **The portable reference tier cannot rescue it.** `ReferenceTierEligible`
   (`src/vt/op_provider.cpp:916`) requires
   `Backend::DeviceMemoryIsHostAddressable()`, and the refusal message
   (`:243-261`) names ROCm on an XNACK-less integrated part
   ([#2511](https://github.com/mudler/vllm.cpp/issues/2511)) as exactly the case
   where it is false. **So a CPU function bound to the ROCm slot is not a
   shortcut here**, and the explicit round trip above is the only correct shape.
3. **Device selection is global and resolved once.**
   `ResolveModelDeviceType` / `SelectQueueForModel`
   (`src/vllm/entrypoints/model_loader.cpp:136-163,346-372`), with a comment at
   `:2699-2717` recording that resolving twice in one load is itself a bug
   class. **Do not add a second resolution.** The policy must ride on the single
   resolved device as an override consulted by the loader, never as a second
   device.

### Mirror `-ot` where sensible

llama.cpp's `-ot` at the pin: `,` separates variant groups and `;` separates
overrides inside a group (`tools/llama-bench/llama-bench.cpp:925-947`); a
pattern is matched with `std::regex_search`, and the first match wins.
`PlacementOverride` already matches that semantics exactly. **Mirror the
semantics, not the spelling** — the surface here is `--offload-config`'s
`vllm_cpp.placement` object (`docs/FEATURES.md:100`), which already maps `-ot`,
`-cmoe`, `-ncmoe` and `--fit`, so the new key belongs in that object and not in
a new flag.

## Risks

- **A host round trip on the decode critical path.** Mitigated by defaulting OFF
  wherever `kEmbeddingQuant` is registered, and by the gate below.
- **Widening `ResidentWeight` widens the blast radius of every model.** It is on
  the shared dense path. The change must be a consultation that returns "device"
  for every tensor no policy names, and a mutation test must show that deleting
  the consultation reds the new case and nothing else.
- **A value regression the token gate cannot see.** The CPU and ROCm gathers are
  separately gated bit-exact against `vt::cpu::BlockToFloat`, so the round trip
  should be value-identical rather than merely close. **Assert bitwise equality,
  not NMSE** — anything looser would hide a codec divergence.
- **The reachability trap.** A unit test that constructs the policy proves the
  class works. The gate must enter through `ModelRegistry::Forward` on a real
  `qwen4exp` GGUF, and deleting the production call site must red it.

## Tests

1. **Red first.** A case that loads a `qwen4exp` fixture with the PLE table named
   in a host-residency override and asserts the table's `Tensor::device` is CPU
   while a layer weight's is the engine device. Red today because
   `ResidentWeight` has no such arm.
2. **Bitwise value equality** of the PLE block output with the policy on and off,
   on the same fixture. Not NMSE.
3. **Reachability by mutation.** Delete the `ResidentWeight` consultation; case 1
   must red. Delete the round-trip wrapper at `qwen4_exp_ple_block.cpp:494-497`;
   case 2 must red.
4. **Default-off.** A case asserting that with no override the ROCm arm still
   routes `kKeepQuant` to the device, so #3093's behaviour is unchanged.
5. **The refusing-device arm.** On a device with no `kEmbeddingQuant`, the load
   currently refuses by name (`qwen4_exp_weights.cpp:666-681`). A case that the
   override converts that refusal into a successful load.

## Gates

- The focused suite above, then the full gate.
- A same-binary A/B on `strix:gpu0` inside an `rc` lease, reporting device
  memory (the falsifiable axis) and decode `tok/s` (predicted unchanged), each
  with its leg spread, **and no ratio against any other engine.**

## Stop conditions

- **Stop if the memory saving is not measured.** The throughput prediction is
  zero, so the memory axis is the entire case for the change. If an A/B does not
  show about -26.8 GiB of device memory, the change has no reason to exist.
- **Stop and return `NEEDS_DECISION` if the round trip costs more than 10% of
  decode.** That is the board's resolution floor; a cost below it is not
  established and a cost above it changes the trade.
- **Do not widen this into a general offload subsystem.** One policy consultation,
  one consumer.

## Owed

- **The ranked kernel table for the vllm.cpp ROCm arm on `gfx1151`**, which this
  investigation could not produce. We own no per-kernel timing instrument on
  ROCm (`src/vt/rocm/rocm_backend.hip:460` records the `rocprofiler` equivalent
  of `VT_CUDA_PROFILE` as owed; `VT_OP_PROVIDER_STATS` counts selections, not
  time), and the out-of-process `rocprofv3` route produced invalid timestamps on
  this board ([#3040](https://github.com/mudler/vllm.cpp/issues/3040)). **Where
  our decode step time goes on this model is UNVERIFIED**, and this spec does not
  guess. It is owned by the issue this spec files.
- **Two stale prose sites.** `docs/FEATURES.md:163` and the code comments at
  `src/vllm/model_executor/models/qwen4_exp_weights.cpp:635-641` and
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:232-240` still say
  ROCm does not register `kEmbeddingQuant` and is refused. #3093 falsified that.
  Repairing one copy creates the next contradiction, so all three move together
  or none does.

## Now

`PROPOSED`. Nothing is implemented. The evidence that motivates it is landed;
the throughput premise that motivated the investigation is refuted.
