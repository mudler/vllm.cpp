# GLM-5.3 (`GlmMoeDsaForCausalLM`) on ROCm / `strix:gpu0`

Row: `BACKEND-ROCM` (the landed fix) and `MODEL-TEXT-GLM-MOE-DSA` (the survey).
Issue: [#2498](https://github.com/mudler/vllm.cpp/issues/2498).
Base SHA: `11fed3ba56b8f823c07032416982a44a8c0967b5`.

## Scope

Establish whether GLM-5.3 non-flash can run on `strix:gpu0`, map the ROCm arms
arm by arm against what the model actually calls, and land the smallest complete
and gated slice that the map identifies. It does **not** port the MLA/DSA
attention arm; §W1.5 records why that is a campaign and not a wave.

## W1. The box, and every number attributed to it

Measured on `strix:gpu0`, rc jobs `0fca4182` and `8293713b`, 2026-09-01.
Numbers from `gfx1100`, `gfx1200` or `gfx1201` in other ROCm specs do not
transfer to this board and are not reused here.

| Fact | Value |
|---|---|
| Marketing name | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |
| `gfx` target | `gfx1151` (also advertises `gfx11-generic`) |
| ROCm / HIP | 7.2.4 / 7.2.53211-97f5574fe2 |
| Compiler | AMD clang 22.0.0git, `roc-7.2.4` |
| CUs | 40 (`rocminfo`); `hipDeviceProp.multiProcessorCount` reports 20 WGPs |
| `warpSize` | 32 |
| `integrated` / `managedMemory` / `concurrentManagedAccess` | 1 / 1 / 1 |
| `pageableMemoryAccess` | 0 |
| System RAM | 62 GiB total, 58 GiB available |
| `hipMemGetInfo` total | 64.000 GiB |

`amdgpu-arch` and `rocm_agent_enumerator` both answer `gfx1151`; `rocm-smi` is
not installed in the leased container.

### Memory ceiling

Walked in 1 GiB steps, every block retained and both end pages touched with a
`hipDeviceSynchronize` after each:

| Allocator | Ceiling | Samples |
|---|---|---|
| `hipMallocManaged` | **58.000 GiB** (62,277,025,792 B) | 3 of 3 identical |
| `hipMalloc` | 63.000 GiB (67,645,734,912 B) | 1 |

`hipMallocManaged` is the operative one: on this board the registrar sets
`managed_alloc`, so **every** `Backend::Alloc` is a managed allocation
(`src/vt/rocm/rocm_backend.hip:163-181`). This independently reproduces the
prior wave's 58.000 GiB figure on the same board.

## W1.1 The residency arithmetic, re-derived

Parsed from the six staged shards' own GGUF headers, not inherited
(`/workspace/glm53-rocm/census.py`, rc job `0fca4182`). Byte counts are computed
from each tensor's ggml type block size, so they are the on-disk footprint.

| Class | Tensors | Bytes | GiB |
|---|---|---|---|
| Expert (`*_exps.weight`) | 228 | 201,125,265,408 | **187.3125** |
| Resident (everything else) | 1581 | 15,580,554,240 | **14.5105** |
| Total | 1809 | 216,705,819,648 | 201.8230 |

**The arithmetic holds; the load still refuses.** 14.5105 GiB of resident class
against a 58.000 GiB managed ceiling is **25.0%**, leaving 43.49 GiB of headroom
-- and W6 records that the device path never gets to use that split, because one
predicate stops the streaming lane being built on ROCm. The arithmetic below is
correct and was worth deriving; it is not on its own a statement that the model
runs here, and this spec does not make that statement. For contrast the Flash
sibling's 101.2535 GiB is 1.75x over the same ceiling with no streaming path,
which is why that row stopped and this one does not.

The experts are 92.8% of the bytes and stream through the slot lane, which is
device-agnostic and host-backed: `ExpertStreamLane` always constructs a
`HostExpertSlotStore` (`src/vllm/model_executor/expert_stream_seam.cpp:343`) and
`ExpertStreamer::Fetch` writes through `SlotForWrite`/`CommitSlot` with no device
predicate (`src/vllm/model_executor/expert_streamer.cpp:60-108`).

## W1.2 Which encodings the resident class needs

This is the finding that removes the presumed keep-quant blocker.

| Class | Encodings present |
|---|---|
| **Resident** | Q5_K (312, 7.1543 GiB), Q8_0 (476, 4.8517 GiB), Q6_K (82, 0.9998 GiB), Q4_K (2, 0.9970 GiB), F32 (709, 0.5078 GiB) |
| **Expert only** | IQ3_XXS (71, 81.5391 GiB), IQ1_S (106, 62.1094 GiB), IQ2_XXS (44, 34.0312 GiB), IQ4_XS (4, 6.3750 GiB), Q2_K (2, 1.9688 GiB), Q3_K (1, 1.2891 GiB) |

ROCm keep-quant serves exactly {Q8_0, Q4_K, Q5_K, Q6_K}
(`.agents/specs/rocm-gg-keep-quant.md`, `src/vt/rocm/rocm_grouped_gemm.hip`).
That set **covers the resident class exactly**; F32 needs no keep-quant path.
Every i-quant in this checkpoint is expert-side, and the experts reach the GEMM
through the streaming lane rather than the resident path.

So `.agents/specs/rocm-gg-keep-quant.md`'s owed formats and #1940's "ROCm ports
zero I-quant formats" are **not blockers for this model's resident arm**. They
remain owed for the expert arm.

## W1.3 Arm-by-arm map

Ops the model actually calls, resolved from
`src/vllm/model_executor/models/glm_moe_dsa_forward.cpp` and
`src/vllm/model_executor/layers/attention/mla_attention.cpp`, against
`RegisterOp(OpId::...)` sites under `src/vt/rocm/`:

| Op | ROCm | CUDA | CPU | Used by |
|---|---|---|---|---|
| `kEmbedding` | YES | yes | yes | forward |
| `kMatmulBT` | YES | yes | yes | forward, MLA |
| `kMoeRouterTopK` | YES | yes | yes | forward |
| `kMoeSiluMul` | YES | yes | yes | forward |
| `kMoeCombine` | YES | yes | yes | forward |
| `kCastF32` | YES | yes | yes | forward, MLA |
| `kRmsNorm` | YES | yes | yes | forward, MLA |
| `kLayerNorm` | YES | yes | yes | MLA |
| `kMulScalar` | YES | yes | yes | MLA |
| `kRopeFromCache` | YES | - | yes | MLA |
| `kSharedExpertGate` | YES | - | yes | MLA |
| `kFusedChain` | MISSING | yes | yes | forward |
| `kBatchedMatmul` | **MISSING** | yes | yes | MLA |
| `kConcatAndCacheMla` | **MISSING** | - | - | MLA |
| `kConcatMlaNopeRope` | **MISSING** | yes | - | MLA |
| `kDsaIndexerLogits` | **MISSING** | yes | yes | MLA |
| `kDsaTopkSelect` | **MISSING** | yes | yes | MLA |
| `kFusedNormRope` | **MISSING** | - | - | MLA |
| `kGatherMlaCache` | **MISSING** | yes | yes | MLA |
| `kMlaDecodeAttention` | **MISSING** | - | - | MLA |

ROCm registers 50 distinct `OpId` values; CUDA registers 118.

### Two briefing premises this map corrects

1. **`kMoeGateUpSwiGLUGrouped` is not on this model's path at all.** Its only
   model call site is the Flash sibling, `glm5_next_forward.cpp:308`. The
   non-flash forward runs experts one at a time as `vt::MatmulBT` over
   `GlmExpertSlice` with `vt::MoeSiluMul` and `vt::MoeCombine`
   (`glm_moe_dsa_forward.cpp:293-298, 422-424`). Porting that provider would
   have been dead code for this model.
2. **The MoE arm is already complete for this model.** Every op the routed and
   shared expert path calls is registered on ROCm today.

The single genuinely missing arm is **MLA/DSA**: eight ops, including the
attention kernel itself and the sparse indexer's logits and top-k select.

## W1.4 The reference tier, and the defect it exposes

`gfx1151` reports `managedMemory=1` and `integrated=1`, so the registrar sets
`unified_memory_`, `DeviceMemoryIsHostAddressable()` returns true
(`src/vt/rocm/rocm_backend.hip:371`), and the portable CPU reference tier is
eligible. `docs/ROCM.md:53-58` documents that tier as the intended bring-up path
for Strix Halo. So the eight missing MLA ops do not refuse; they install CPU host
kernels and run.

**The briefing's O42 claim that this happens silently is stale.** The tier is
announced: `GetOp` warns once per `(op, device)` on stderr as
`[vt reference-tier] op=... device=... has NO native kernel`
(`src/vt/op_provider.cpp:605-616`), counts every hit in `GetReferenceTierHits()`
(`include/vt/op_provider.h:269`), and `OpRegistered` deliberately excludes the
tier so it stays a native-only probe (`src/vt/op_provider.cpp:788-806`).
`docs/ROCM.md:60-61` already forbids a performance result with a non-zero hit
count. Device-vs-host is therefore assertable today with no new mechanism, and
#2433 shows the lines being read off this exact board.

**What is not sound is the tier's memory ordering on ROCm.** `FlushPending()`
exists so a host reference kernel does not read device memory an unfinished
submission has not written (`include/vt/backend.h:53-59`). `GetOp` calls it at
`src/vt/op_provider.cpp:708-711` and the decline path at `762-765`. Metal
overrides it (`src/vt/metal/metal_backend.mm:101`) and Vulkan overrides it
(`src/vt/vulkan/vulkan_backend.cpp:118`). **`RocmBackend` does not**, so it takes
the default no-op — while `CreateQueue` builds a genuine asynchronous stream with
`hipStreamCreate` (`src/vt/rocm/rocm_backend.hip:214-217`).

The default's contract says it "suits every backend that submits eagerly". HIP
submits eagerly and **completes** asynchronously, so the contract does not hold.
ROCm is the only host-addressable backend in the tree that inherits the no-op.

## W1.5 Why the MLA/DSA arm is a campaign — NEEDS_DECISION

Eight ops, two of which (`kMlaDecodeAttention`, `kConcatAndCacheMla`) have no
CUDA registration to mirror either and two more (`kDsaIndexerLogits`,
`kDsaTopkSelect`) are the sparse-attention indexer. That is not one wave's work,
and a partial arm is worse than none: with the tier eligible, a half-ported arm
still runs, still emits tokens, and hides which half executed on the device.
This wave does not open it. It is recorded as owed below.

## W2. The slice this wave lands

`RocmBackend::FlushPending()`, issue #2498.

It is the smallest change that is complete on its own, and it is on this model's
path eight times over. It is a precondition for every one of the eight MLA ops
being *correct* on this board, and it stays load-bearing after any of them is
ported natively, because the remainder still run on the tier.

- **Red first.** A HIP device kernel that spins long enough to still be running
  when the host regains control writes a managed buffer; the host reads it after
  `FlushPending()`. With the default no-op the host observes the pre-kernel
  value. The test asserts the post-kernel value.
- **Mutation.** Delete the override; the focused case must fail. Restore and
  re-run.
- **Inertness.** A run reaching zero reference-tier ops calls `FlushPending()`
  never — the call site is guarded by `slot.ref_selected`
  (`src/vt/op_provider.cpp:707`) — so the CUDA, CPU and discrete-ROCm paths are
  untouched by construction.

## W3. Measured result (rc job `0d94bc20`, `strix:gpu0`, 2026-09-01)

Build: hipBLASLt installed to clear #2499, `cmake rc=0`,
`ROCm backend: ENABLED for arch(es) [gfx1151]` asserted fatally at configure time.

### The mutation ladder

| Step | Tree | Case | Result |
|---|---|---|---|
| A | override **deleted** (= pre-fix) | by-hand | **RED** `exit=1`, 1 case / 6 assertions, 1 failed |
| B | restored byte-identical | by-hand | GREEN `exit=0`, 6/6 |
| C | production call site **deleted** | by-hand | GREEN 6/6 — *it does not go through the call site* |
| C | production call site **deleted** | GetOp-entry | **RED** `exit=1`, 10 assertions, 1 failed |
| D | call site restored byte-identical | GetOp-entry | GREEN 10/10 |

Both restores verified by sha256 (`ae6eea5b...` for `rocm_backend.hip`,
`7a449104...` for `op_provider.cpp`). Every mutated build was `rc=0` first, so no
RED here is a build error wearing a test result, and every run reports a non-zero
selected-case and assertion count, so none is an empty filter wearing a pass.

**The failure signature is the race itself.** In steps A and C the failing
assertion is the LAST element of the buffer:

```text
test_rocm_backend.cpp:282: ERROR: CHECK( dout[n - 1] == doctest::Approx(2.0f) )
  values: CHECK( -1 == Approx( 2 ) )
```

`dout[0]` and `dout[n/2]` pass in the same run. The host observed the front of a
2048x1024 output the device had already written and the tail it had not, which is
what an undrained stream looks like and not what a wrong kernel looks like.

Steps C and D are the reachability proof §"Nothing lands dead" asks for: deleting
the production call site reds the GetOp-entry case while the by-hand case stays
green, so the two cases demonstrably measure different things.

### Suites

- `test_rocm_backend` on the restored tree: **11 cases / 11 passed / 1087
  assertions / 0 failed**.
- `test_backend_cross_device`: 26 cases / 25 passed / **1 failed**; 80253
  assertions / 1 failed. The failure is `MoeSiluMul matches the CPU oracle within
  NMSE <= 5e-4`, logged `bf16 := true`, `DeviceName(dt) := ROCM`, differing by
  +/-1 ULP in bf16. That is the **pre-existing** #1954 / #1513 defect, and this
  wave's diff touches neither that file nor any arithmetic path -- the three
  changed files are this spec, `rocm_backend.hip` and `test_rocm_backend.cpp`.
  Recorded here because #1954 and #1513 name `gfx1200`: it reproduces on
  `gfx1151` too.

## W4. `--device rocm` does not exist, and that is a seam gap

The model run did not happen, and the reason is not the model:

```text
vllm-cli: unknown --device 'rocm' (expected auto, cpu, or cuda)
```

`examples/cli/main.cpp:143-150` maps only `auto`/`cpu`/`cuda`, mirroring the C ABI
it is a thin client of: `vllm_model_params.device` is documented at
`include/vllm.h:113-120` as `0=auto, 1=cpu, 2=cuda`, where `2` *requires* the CUDA
platform and fails the load when it is absent rather than substituting.

So on an AMD box `auto` is the only route to the GPU, and ROCm cannot be pinned
explicitly. A user cannot distinguish "ran on the Radeon" from "auto fell back to
the CPU queue" by configuration; they have to read the log. §"Shared seams" says
every shipped capability is exposed through `include/vllm.h`, and ROCm, Metal,
Vulkan, XPU and Tenstorrent are all shipped backends with no ABI selector.

Filed as #2505. Not fixed here: extending the ABI enum is a public-surface change
that owes its own row, spec and version note, and it is not a model bring-up.

## W5. A ccache-enabled HIP build does not link (observed, cause not yet isolated)

Two runs on the same box, same source archive, same ROCm, differing only in
whether `ccache` was installed and `CCACHE_DIR` exported:

| Run | ccache | `vllm-cli` link |
|---|---|---|
| probe4 (`0d94bc20`) | no | **rc=0**, binary ran |
| probe5b (`529b9edc`) | yes | rc=1 |
| probe5c (`7894a2bc`) | yes | rc=1, identical failure |

Both failures are the same set of undefined references while linking `vllm-cli`
against `libvllm.so.0.0.3` -- the HIP runtime and hipBLASLt entry points
together:

```text
/usr/bin/ld: libvllm.so.0.0.3: undefined reference to `hipSetDevice@hip_4.2'
/usr/bin/ld: libvllm.so.0.0.3: undefined reference to `hipblasLtMatrixLayoutCreate'
/usr/bin/ld: libvllm.so.0.0.3: undefined reference to `__hipUnregisterFatBinary@hip_4.2'
```

The configure output is byte-identical between the runs -- both resolve
`ROCm hipBLAS: /opt/rocm/lib/libhipblas.so` and
`ROCm hipBLASLt: /opt/rocm/lib/libhipblaslt.so`, and both print
`ROCm backend: ENABLED for arch(es) [gfx1151]` -- and both used a freshly deleted
build directory, so this is not a stale cache. The `target_link_libraries(vllm
PUBLIC ...)` calls for `amdhip64` and the math libraries are unconditional inside
`if(VLLM_CPP_HIP)` (`CMakeLists.txt:1747-1775`), so nothing in the project's own
conditionals explains it.

`-DVLLM_CPP_BUILD_TESTS=ON` was tested as the candidate discriminator and
**refuted**: the option already defaults to `ON` (`CMakeLists.txt:58`), so passing
it is a no-op, and probe5c failed exactly as probe5b did.

**This is recorded as an observation, not a diagnosis.** The correlation is
clean across three runs but the mechanism is not established, and this wave does
not claim one. It matters because `/workspace/ccache` is shared fleet
infrastructure that other ROCm rows may reach for. Filed as #2506.

## W6. THE ACTUAL BLOCKER, and it is none of the things this wave expected

The model was run on `strix:gpu0` through the production CLI on `--device auto`
(rc job `79fb8bab`, probe6). It **refuses at load**:

```text
vllm_engine_load: device 'rocm' cannot serve this GGUF: staging its weights needs
at least 216433205760 bytes (201.56 GiB) of device memory across 1809 tensors ...
and this device's memory pool is 68719476736 bytes (64.00 GiB).
```

The whole expert set is charged to the device, although this model declares
`streams_routed_experts = true`
(`glm_moe_dsa_registry.cpp:133`) precisely so it is not.

**The cause is one predicate.** The streamed-expert lane is guarded at
`model_loader.cpp:2761-2768`, and its FIRST condition is
`target.needs_weight_staging()`. `RocmPlatform` returns **false** for it
(`platforms/rocm.cpp:98`) -- deliberately, because that flag selects the
fully-optimized GDN forward whose consumers lack per-op fallbacks, a decision
#1934 made and this wave does not reopen. The other five conditions hold on this
board. So the lane is never built, `lane` stays empty, and `CheckDeviceWeightFit`
charges the 187.312 GiB of towers.

**It is the same defect #1934 fixed, one call site short.** #1934 split the wide
predicate because it was answering a narrower question, and added
`allocates_bounded_device_memory()` -- which `RocmPlatform` overrides to **true**
(`platforms/rocm.cpp:122`). The refusal was moved onto the narrow predicate. The
lane guard was not. ROCm therefore gets the worst pairing available: the refusal
fires, and the exemption that would satisfy it does not. On CUDA both are true,
the lane engages, and the same checkpoint generates on GB10.

Filed as #2507. Not fixed here: it changes load-path behaviour for every ROCm MoE
model, which is a surprising fix owing its own row, spec and fresh review rather
than an in-flow edit.

### Two further observations from the same load

The hybrid MoE placement resolved and installed a plan, and the refusal ignored
it. `216433205760 - 147798884352 = 68634321408` B = 63.92 GiB, **under** the
64.00 GiB budget by 81 MiB, yet the next line refuses quoting the un-reduced
201.56 GiB. Two lines of one load contradict each other.

And the budget itself is optimistic here: 64.00 GiB is `hipMemGetInfo` total,
while every `Backend::Alloc` on this board is `hipMallocManaged`, whose measured
ceiling is 58.000 GiB. Even the reduced 63.92 GiB is 5.92 GiB above what this
allocator reaches, so honouring the plan alone would have produced a late OOM.

### What this changes about the rest of this spec

The arm-by-arm map in W1.3 stands: the MLA/DSA arm really is eight ops short. But
it is **not** what stops GLM-5.3 on ROCm today, and neither is keep-quant, and
neither is memory. The load never reaches a forward, so no MLA op is ever asked
for. **The ordering is: #2507 first, then the MLA arm.** Porting MLA kernels
before #2507 is settled would produce eight kernels nothing can reach.

**No token is claimed.** `reference-tier distinct=0` and no expert-stream counter
was emitted, because the process refused before any forward -- consistent with a
load-time refusal and not evidence about any kernel.

## Owed

Unreached or unported after this wave, none of it claimed here:

- **#2507 first**: the streamed-expert lane guard reads `needs_weight_staging()`,
  so GLM-5.3 is refused at load on ROCm before any forward runs. Nothing in the
  MLA arm is reachable until this is settled.
- The ROCm MLA/DSA attention arm, AFTER #2507: `kBatchedMatmul`,
  `kConcatAndCacheMla`, `kConcatMlaNopeRope`, `kDsaIndexerLogits`,
  `kDsaTopkSelect`, `kFusedNormRope`, `kGatherMlaCache`, `kMlaDecodeAttention`.
  Owner `BACKEND-ROCM`; needs its own issue and spec. Porting these before #2507
  would produce eight kernels nothing can reach.
- Whether the device-fit budget should track the managed ceiling (58.000 GiB
  measured) rather than the reported pool (64.00 GiB), and whether the installed
  MoE placement plan should be credited by the fit check. Both in #2507.
- `kFusedChain` on ROCm. The forward falls back to a standalone `vt::RmsNorm`
  when the recipe is unavailable (`glm_moe_dsa_forward.cpp:85-89`), so this is a
  performance gap, not a correctness one.
- ROCm i-quant keep-quant for the expert class (IQ1_S, IQ3_XXS, IQ2_XXS, IQ4_XS,
  Q2_K, Q3_K). Existing gap, #1940.
- The hipBLASLt build asymmetry, #2499: `VLLM_CPP_HIP=ON` configures on a ROCm
  install without hipBLASLt and then fails at TU 539/570, because CMake treats
  the library as optional while `rocm_matmul_hipblaslt.hip` includes its header
  unconditionally. Found while building this wave on `strix:gpu0`; not fixed here
  because the repair is a deliberate choice between making the library required
  and compile-guarding the TU, and that choice belongs to `BACKEND-ROCM` rather
  than to a model bring-up. Worked around in the lease by installing the package.
- No explicit ROCm device selector in the C ABI or the CLI, #2505. Extending
  `vllm_model_params.device` past `0=auto/1=cpu/2=cuda` is a public-surface
  change owing its own row, spec and ABI version note, so this wave records it
  rather than making it. `--device auto` is the working route meanwhile.
- The ccache/HIP link correlation in W5, #2506. Observed three times, mechanism
  not isolated; this wave routes around it rather than diagnosing it.
- `MoeSiluMul` bf16 is +/-1 ULP off the CPU oracle on `gfx1151` as well as the
  `gfx1200` boards #1954 and #1513 name. Pre-existing and untouched here.
- No speed number is claimed or owed for this board by this wave. Any run of this
  model on ROCm reaches eight reference-tier ops, which `docs/ROCM.md:60-61`
  disqualifies as a performance result.
