# BACKEND-ROCM-IQ-EXPERT-RESIDENCY — route a placed routed-expert tower's residency through the device that will RUN it

Issue: [#2516](https://github.com/mudler/vllm.cpp/issues/2516) (the routing),
[#2517](https://github.com/mudler/vllm.cpp/issues/2517) (the fit check that
ignores the plan it just installed).
Related, and deliberately NOT repaired here:
[#2515](https://github.com/mudler/vllm.cpp/issues/2515) (the host-slot lane
cannot serve on `gfx1151`; this row's answer does not need it),
[#2518](https://github.com/mudler/vllm.cpp/issues/2518) (the managed ceiling;
measured below to be unable to decide this load),
[#1940](https://github.com/mudler/vllm.cpp/issues/1940) (ROCm i-quant
`vec_dot`; measured below to be unnecessary for this load),
[#2505](https://github.com/mudler/vllm.cpp/issues/2505) (no `--device rocm`).

Base SHA: `35116605bb294e78f1be103cd040c378c6d00e20`.

## Scope

Make GLM-5.3 (`GlmMoeDsaForCausalLM`) `UD-IQ1_S` load and generate on
`strix:gpu0` (`gfx1151`, ROCm 7.2.4) with its routed experts placed on the
host, by fixing the two predicates that stop a HYBRID-PLACED load from
reaching a forward. It does not port a kernel, does not move a platform
predicate, and claims no performance number.

IN SCOPE:

- `GgufLoadPolicy::Route` / `PeekRoute`: for a `kStackedExpertWeight`, resolve
  the residency against the device the installed `MoePlacementPlan` says will
  COMPUTE that tower, instead of always against the engine device.
- `GgufStagedWeightFootprint` / `CheckDeviceWeightFit`: exclude the
  routed-expert tensors of layers the installed plan places off the engine
  device, and say so in the refusal.
- `tests/vllm/model_executor/test_gguf_device_fit.cpp`'s IQ case, which is red
  on every ROCm build for a reason the case does not state (#2516).

OUT OF SCOPE:

- ROCm i-quant `vec_dot` kernels (#1940). Measured unnecessary: a tower placed
  on the CPU is executed by `vt::cpu`'s `vec_dot`, and
  `vt::cpu::HasQuantDotKernel` already answers true for all six encodings this
  checkpoint's experts use (`cpu_quant_dot.cpp:995,996,1000,1001,1004,1005`).
  The ROCm gap is what makes an UNPLACED tower expand, and this checkpoint has
  no unplaced-tower arm on this board for an unrelated reason (see W2).
- `RocmPlatform::residency_policy().device_memory_total_bytes` (#2518). Left at
  the `hipMemGetInfo` probe. See W4 for why it cannot decide this load.
- The host-slot expert-streaming lane (#2515). Not repaired and not needed: see
  W2.
- `--fit`'s own resolver. It sizes a placement from bytes and cannot know that
  an UNPLACED tower expands on this device, so its default answer on this board
  is a placement that still refuses. Recorded under `## Owed`.

## W1. The chain, measured rather than assumed

Five facts, each read off the tree at the base SHA or off the board:

1. **GLM-5.3's expert towers are never uploaded to any device.**
   `LoadStackedExperts` (`glm_moe_dsa_loader.cpp:381-421`) takes no `Dev` and
   no queue; it returns an mmap-borrowed, unprefaulted `OwnedTensor`. The
   registry's `prepare` hook is a no-op (`glm_moe_dsa_registry.cpp:88-93`), and
   the forward reaches a tower only through `GlmExpertSlice`
   (`glm_moe_dsa_forward.cpp:293,294,297-300`), never through
   `dense_attn::ResidentWeight` — which is the one place `d.b.Alloc` +
   `d.b.Copy` live (`dense_attn_block.h:250-268`). The file says so itself at
   `glm_moe_dsa_forward.cpp:94-113`.

2. **On ROCm the unplaced MoE forward cannot run at all.**
   `GlmResidentExpertSlice` throws unless the platform is CPU or host-addressable
   (`glm_moe_dsa_forward.cpp:112-122`), and `expert_stream::ExpertSlice` admits
   the streaming lane on the same predicate (`expert_stream_seam.cpp:431`).
   `host_memory_is_device_addressable()` is FALSE on `gfx1151`
   (#2515, re-measured in W4), so BOTH arms are closed. GLM-5.3's routed
   experts on this board can only run on a CPU `Dev` handed to `MoeBlock` by
   `RunMoePlaced` — which the model already composes, unconditionally placeable
   (`glm_moe_dsa_forward.cpp:471-475`).

3. **The loader asks the wrong device about a placed tower.** The one policy is
   `GgufLoadPolicy::FromEnv(source.device)` (`glm_moe_dsa_registry.cpp:80`), and
   `Route` hands `policy.device` straight to `RouteGgufTensor`
   (`gguf_keep_quant.cpp:401-408`). `DeviceKeepQuantSupported` serves exactly
   {Q8_0, Q4_K, Q5_K, Q6_K} on ROCm (`gguf_keep_quant.cpp:135-147`), so every
   IQ1_S / IQ2_XXS / IQ3_XXS / IQ4_XS / Q2_K / Q3_K tower routes to
   `kExpandBf16` and `LoadStackedExperts` refuses by name
   (`glm_moe_dsa_loader.cpp:392-400`) — for a tower the plan has already
   decided will execute on the CPU and whose bytes will never touch the GPU.

4. **The fit check ignores the plan installed 220 lines above it.**
   `InstallMoePlacementPlan` runs at `model_loader.cpp:2672`,
   `CheckDeviceWeightFit` at `:2895`. Neither `GgufStagedWeightFootprint` nor
   `CheckDeviceWeightFit` takes any representation of a resolved placement, so
   the 187.3125 GiB of towers is charged to the device pool whatever the plan
   says. This is #2517, and on this board it is not cosmetic: it is the line
   between the placed load and a forward.

5. **The CPU can execute every encoding this checkpoint's experts use.**
   `vt::cpu::HasQuantDotKernel` (`cpu_quant_traits.cpp:165-170`) accepts 17
   block dtypes including all six here. So a tower routed against `kCPU` keeps
   its blocks and is read in place out of the mapping.

The chain therefore closes with two edits and no new kernel.

## W2. Design

**One question, asked of the right device.** `GgufLoadPolicy::device` is
documented as "the device the ENGINE resolved for this load", and the field's
own comment states that EVERY device-dependent decision reads it. Hybrid
placement (#2023/#2314) introduced a second, narrower answer for exactly one
class of tensor: `ActiveMoePlacementPlan().DeviceForLayer(l)` is the device a
layer's routed experts are COMPUTED on. A residency decision for a
`kStackedExpertWeight` is a question about that device, not about the engine.
This is the same defect as #1136 (the bound and the policy naming different
devices) and #2406 (a host-ISA repack applied to a weight staged elsewhere),
one seam further along.

`GgufLoadPolicy::Route` and `PeekRoute` resolve through ONE new private helper
so the two entry points cannot answer differently — the property
`GgufExpertTowersReachSlotLane` depends on, since it peeks the same tensors the
loader routes.

**Inert by construction, not by a flag.** The override applies only when

```
role == kStackedExpertWeight
  && ActiveMoePlacementPlan().PlacesAnything()
  && the tensor names a block index
  && that block's device != the plan's engine device
```

`PlacesAnything()` is false on every load that configured no placement, which is
every CUDA and CPU load in the tree's gates today, so those loads route
byte-for-byte as they did. The last term is what stops a default-constructed
global (whose `engine_device_` is `kCPU`) from silently answering `kCPU` on a
CUDA load: the answer is only ever ADOPTED when the plan actively moves that
tensor somewhere else.

**The block index comes from the name**, because that is the only key the loader
has. `MoePlacementPlan::DeviceForRoutedExpertTensor(name)` parses the
`blk.<N>.` prefix llama.cpp's GGUF export writes — the same spelling
`RoutedExpertTensorNames` and `LlmFfnExpsBlockRegex` already compose — and
answers the engine device for a name that carries none.

**The fit check credits the same plan.** `GgufStagedWeightFootprint` gains a
`const MoePlacementPlan*` (null = today's behaviour, byte for byte) and moves a
placed routed-expert tensor into two new fields, `placed_tensor_count` /
`placed_bytes`, exactly as `StreamedExpertLane` already moves a streamed one.
The justification is the placement seam's own: `ResidentWeight` aliases host
bytes on a CPU `Dev`, so a layer whose block only ever runs there is never
staged to the accelerator (`moe_placement_seam.h`, the "ACROSS" comment). The
refusal appends a NOTE naming the placed count and bytes, so an operator can
never again read two contradictory numbers in consecutive lines.

**Why not make the lane serve instead.** #2515 is right and this row does not
contest it: `HostMemoryIsDeviceAddressable` is `integrated &&
pageable_memory_access`, `pageableMemoryAccess` is 0 on `gfx1151`, and the slot
arena is a pageable `std::vector<uint8_t>`. Forcing the lane on would delete a
correct refusal. The device-side slot store that would fix it is
`ENG-EXPERT-STREAM-DEVICE` W2 and a different row.

**Why not widen ROCm keep-quant.** Writing IQ `vec_dot` in
`rocm_grouped_gemm.hip` (#1940) would keep an UNPLACED tower compressed — and
W1 fact 2 says an unplaced tower cannot be executed on this board at all,
because `GlmResidentExpertSlice` refuses a non-host-addressable device before
any kernel is asked for. So #1940 would not produce a token here. It stays owed
for the discrete-ROCm arm and for performance.

## W3. Tests

Red-first, and each named test fails for its intended reason before the change.

**`tests/vllm/test_gguf_keep_quant.cpp`** (new cases):

- An IQ1_S stacked-expert tower under a policy with `device = kROCM` and an
  installed plan that places its layer on `kCPU` routes `kKeepQuant`. RED
  before (`kExpandBf16`).
- The SAME tensor with NO plan installed still routes `kExpandBf16` on
  `kROCM` — the inertness pin, green before and after.
- A tensor of a layer the plan does NOT place still routes against the engine
  device while a placed sibling does not, in one assertion pair, so the
  override is proved per-layer rather than per-load.
- `Route` and `PeekRoute` agree on every case above.
- `kMatmulWeight` and `kEmbeddingTable` are UNMOVED by an installed plan: the
  override is scoped to the one role whose compute the plan actually moves.

**`tests/vllm/model_executor/test_device_placement.cpp`**:
`DeviceForRoutedExpertTensor` on `blk.7.ffn_gate_exps.weight`, on a name with
no block index, and on an out-of-range block.

**`tests/vllm/model_executor/test_gguf_device_fit.cpp`**:

- A footprint with a plan that places the trailing N layers reports
  `placed_bytes` equal to those towers and drops them from
  `lower_bound_bytes`; with a null plan it is byte-identical to today.
- `CheckDeviceWeightFit` REFUSES without the plan and PASSES with it, on one
  synthetic file whose numbers straddle the budget. RED before (the parameter
  does not exist).
- The #2516 case is re-pointed from `CurrentPlatform().device_type()` to the
  two devices it is actually about: `kCPU` routes IQ towers `kKeepQuant`, and
  `kROCM` routes them `kExpandBf16`. Platform-independent, and it PINS the
  ROCm gap #1940 owns instead of failing over it.

**Reachability (`.agents/reachability.md`).** The production entry point is
`vllm_engine_load` → `ModelLoader::FromModelDir`'s GGUF branch. The mutation
deletes the placement term at BOTH production call sites (the `Route` helper
and the `CheckDeviceWeightFit` argument), one at a time, and requires the
focused suites to go red; a by-hand construction of the policy is not the
proof and is not offered as one.

## W4. Gates

- `scripts/agent-preflight.sh --fail-on-skip` on the branch.
- Hand-run, with case AND assertion counts read together:
  `test_gguf_keep_quant`, `test_gguf_device_fit`, `test_gguf_device_fit_reach`,
  `test_device_placement`, `test_glm_moe_dsa_gguf_load`, `test_expert_stream_wiring`.
- On `strix:gpu0` inside an `rc` lease: the same suites on a `gfx1151` HIP
  build, plus the product legs below.
- CUDA/CPU inertness: every suite above on a non-HIP build, and the assertion
  that a load with no placement installed routes and prices byte-identically.

**The product bar.** `vllm-cli --model <derived shard 1> --device auto` with
`VT_CPU_MOE=1` on `strix:gpu0`, against
`/workspace/ckpt/GLM-5.3-UD-IQ1_S` (six shards, first shard, never the
directory; shard 1 derived by `scripts/glm-dsa-write-indexer-types.py`).
Reported with it: the emitted text verbatim, the resident footprint, whether
the towers stayed compressed, and `GetReferenceTierHits()` / the
`[vt reference-tier]` distinct count — because #2505 means no `--device` value
names ROCm and configuration alone cannot separate a GPU run from a host
fallback.

**No speed number, either way.** `docs/ROCM.md:60-61` disqualifies a
performance result from a run with a non-zero reference-tier count, and this
model's MLA/DSA arm is eight ops short on ROCm
(`.agents/specs/rocm-glm53-dsa.md` W1.3), every one of which lands on that
tier.

## W5. Risks

- **A fourth predicate.** Three predicates on this row have each looked like
  the last one (#2507, #2515, #2516). W1 traces the whole chain from load to
  `MoeBlock` rather than the next hop, but the MLA/DSA arm is unexercised on
  this board and may refuse for its own reasons. The stop condition below says
  what happens then.
- **A global read inside a routing function.** `Route` becomes dependent on
  process state. Argued for rather than against: the plan IS the resolved
  answer to "which device runs this tensor", it is installed before any weight
  I/O by construction (`model_loader.cpp:2668-2672`), and the alternative —
  threading a plan pointer through twelve `FromEnv` call sites — widens the
  change without making any of them able to answer differently. The inertness
  term keeps a never-installed global from ever being consulted.
- **`quant_repack` is still resolved from the ENGINE device**, so a tower
  placed on the CPU is not i8mm-repacked. That is a performance gap, not a
  correctness one (the CPU quant GEMM reads unrepacked blocks), and it is
  listed as owed rather than fixed, because moving it means moving a flag
  `FromEnv` resolves once for the whole file.

## W6. Stop conditions

Stop and report `NEEDS_DECISION` rather than widening scope if the placed load
reaches a forward and then refuses inside the MLA/DSA arm: porting those eight
ops is `.agents/specs/rocm-glm53-dsa.md`'s recorded campaign, not this row.

Stop rather than forcing the streaming lane on, or lowering
`host_memory_is_device_addressable()`'s bar, if the placed route turns out to
need it. #2515 records why that deletes a correct refusal.

## Owed

- **ROCm i-quant `vec_dot`** (#1940), for a discrete ROCm board and for the
  performance arm. Unnecessary for this row's result, for the reason in W2.
- **The ROCm MLA/DSA arm**, eight ops (`.agents/specs/rocm-glm53-dsa.md`).
  Every one of them runs on the reference tier in this row's result, which is
  why no speed number is quoted.
- **`--fit`'s resolver is not residency-aware**, filed as
  [#2565](https://github.com/mudler/vllm.cpp/issues/2565) and CONFIRMED on
  hardware: with no placement configured it places the trailing 56 of 78, the
  fit check then passes (this row's own repair), and the load refuses on
  `blk.3`, one of the 22 it left on the device. On this board the operator has
  to say `cpu_moe`.
- **The MLA `fused_nr` availability probe**,
  [#2564](https://github.com/mudler/vllm.cpp/issues/2564) — the blocker that
  now stands between this checkpoint and a token on `gfx1151`. Owner
  `BACKEND-ROCM`.
- **`quant_repack` for a placed tower** (W5).
- **The device budget** (#2518) and **the host-slot lane** (#2515), both
  untouched and both re-measured here.

## Now

`ACTIVE` — implemented, gated, and measured on `strix:gpu0`; open as #2562 for a
fresh review. The row's own scope is complete and the model's remaining blocker
is named and filed (#2564), not left to be discovered.

## Outcome

**What was measured, on the real artifact.** Both predicates this row names are
real, and they are SEQUENTIAL rather than alternative: on the base build with
`VT_CPU_MOE=1` the fit check refuses quoting the un-reduced 216433205760 B with
all 78 layers already placed (#2517), and with that refusal suppressed the load
gets one stage further and refuses on
`blk.3.ffn_gate_exps.weight routed to an EXPAND residency` (#2516). With both
fixed, the same command LOADS: 1809 tensors, `[vt load] weights 1372.153 s`,
`gguf prefault spans=589 paged_in=11.620 GiB in 1037.930 s (11.5 MiB/s)`, the
engine auto-fits `max_model_len` from 1048576 to 8192 against 256 blocks of 32
tokens, and the scheduler starts. **All 228 routed-expert towers stayed
compressed**, which is not an inference: `LoadStackedExperts` throws by name on
any tower that expands, so a load that completes is the proof.

**NO TOKEN CAME OUT, and none is claimed.** The first forward throws inside the
MLA block. `fused_nr` (`mla_attention.cpp:550-551`) has a
`vt::OpRegistered(kFusedNormRope, ...)` term, ROCm registers no
`kFusedNormRope`, and `OpRegistered` is deliberately a native-only probe that
cannot see the reference tier — so the split A-projection path is taken and
refuses this checkpoint's block-quantized `kv_a_proj_with_mqa`. The comment at
`:628-631` states that path is reachable "only with VT_MLA_FUSED_NORM_ROPE=0";
this run had it unset and is the counterexample. Filed as
[#2564](https://github.com/mudler/vllm.cpp/issues/2564). That is exactly this
spec's W6 stop condition, and the row stops there rather than widening into the
MLA arm.

**What was rejected, with the reason measured rather than assumed.** Repairing
the host-slot lane: `pageableMemoryAccess = 0` was read off this board, so
`HostMemoryIsDeviceAddressable` is false and the lane cannot serve here at all
(#2515 confirmed). Porting ROCm i-quant `vec_dot` (#1940): a placed tower is
executed by the CPU, and an UNPLACED tower on this board cannot be executed at
all because `GlmResidentExpertSlice` refuses a non-host-addressable device before
any kernel is asked for — so those kernels would not have produced a token here.
Correcting the device budget (#2518): the credited footprint is 11.620 GiB paged
in against a 58.000 GiB managed ceiling and a 64.00 GiB reported pool, so the
6.00 GiB of optimism between them cannot decide this load in either direction.

**One measurement about the instrument, not the code.** `vllm-cli`'s sha256 is
BYTE-IDENTICAL across the base and patched builds
(`ac37fb6d11ba3c1ae7d3d931777ac32ef6fe6bee00231ba18470330740192529`) while
`libvllm.so.0.0.3` moves — it is a thin ABI client and nothing in it changed. A
binary-identity guard that digests only the executable reports two different
builds as one, and this row measured that rather than inheriting it.
