# LTX25-VAE-DEVICE-RESIDENCY — the volume stays on the device between the convolutions

Row: `LTX25-VAE-DEVICE-RESIDENCY`, a stage of `LTX25-DEVICE-RESIDENCY`
([`ltx25-device-residency.md`](ltx25-device-residency.md)), under the
`ROAD-V1-LTX25` campaign.
Issue: [#1451](https://github.com/mudler/vllm.cpp/issues/1451), already indexed
at `.agents/issue-index.md` under owner `LTX25-DEVICE-RESIDENCY`. **This row does
not append a second index row for it**, for the reason
[`ltx25-device-residency.md`](ltx25-device-residency.md) records at its own
`## Records` note: the index is keyed by issue, and a second `#1451` row
DUPLICATES rather than merges.
Base: `ced0ab639dc4d9bac16029c8da17e97fd1bd7f66`.

## Now

`ACTIVE`. Wave A is the change in this pull request. `AttnBlock3d` is deferred
and declared under `## Owed`.

## Scope

**In scope.** The LTX-2.5 **conv** video VAE decode keeps its volume in device
memory from `conv_in` to `unpatchify`, and its weights are staged onto the device
ONCE rather than re-uploaded per convolution.

**Out of scope, deliberately.**

* The diffusion decoder arm (`NADiffusionDecoder`). `Ltx2VideoDecode` refuses
  `kDiffusion` by name (`include/vllm/model_executor/models/ltx2_video_vae.h`)
  and that refusal stands unchanged.
* The video **encoder** half of the same translation unit. It shares
  `CausalConv3d`, `PixelNorm`, `ApplyNorm` and `AttnBlock3d` with the decoder, so
  it inherits every kernel this row adds, but no encoder call site is moved onto
  a resident volume and no encoder residency claim is made.
* Any speed number, ratio or throughput claim. **No GPU lease was taken for this
  row.** See `## What a CPU-only run can and cannot establish`.
* [#1011](https://github.com/mudler/vllm.cpp/issues/1011), the memory-format
  rider. `[C, T, H, W]` stays the layout on both arms; this row makes the volume
  resident and does not re-choose its format.

## Our baseline

**The defect, verified against this tree rather than taken from the issue.**

Read against `src/vllm/model_executor/models/ltx2_video_vae.cpp` at
`ced0ab639`, not taken from the issue:

* `Conv3dThroughSeam` (`:184-236`) is the only device dispatch. On a non-CPU
  queue it allocates FOUR `DevBuf`s per call — input, weight, output and bias —
  copies three of them up, dispatches `vt::Conv3d`, and **downloads the output**
  (`:235`). Every operand is freed when the call returns.
* Therefore the weight and the bias are re-uploaded on **every** convolution, and
  the volume is on the host **between** every pair of convolutions.
* `Silu` (`:371`), `PixelNorm` (`:381`), the GroupNorm arm through `ApplyNorm`
  (`:437-446`), `FeedSpatialNoise` (`:507`), `ApplyAdaLn` (`:529`),
  `Linear3d` (`:339`), `DepthToSpaceUpsample` (`:603`), `AttnBlock3d` (`:691`)
  and the `unpatchify` tail (`:945-970`) are all `std::vector<float>` host loops
  over `Volume::data`.
* **One thing the issue does not say, and it is this row's finding:** the causal
  and spatial PAD inside `CausalConv3d` (`:274`, `vt::cpu::ParallelForRows`) is
  also a host loop, so even the convolution stage is not fully resident — the
  padded volume is BUILT on the host and then uploaded.

## Upstream chain

**Why this is a DIVERGENCE and not only a cost, verified at the oracle pin.**

Lightricks/LTX-2 @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`, asserted by
`git rev-parse HEAD` in a clean checkout before any anchor below was read.

* `packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py:273` —
  `device = torch.device("cuda") if device is None else device`. The decoder is
  built onto a device once, at build time.
* `packages/ltx-core/src/ltx_core/model/video_vae/conv_video_decoder.py:283-284`
  — `weights_dtype = next(self.parameters()).dtype; sample = sample.to(weights_dtype)`.
  The latent follows the weights.
* `grep -rn '\.cpu()\|to("cpu")' packages/ltx-core/src/ltx_core/model/video_vae/`
  returns only the checkpoint loader's `safetensors.safe_open(..., device="cpu")`
  (`model_configurator.py:230,334,497`) and the DIFFUSION decoder's timestep
  schedule (`diffusion_video_decoder.py:106`). **The conv decoder's forward
  contains no host round-trip at all.**

vLLM registers nothing LTX and is not the reference here; vLLM-Omni @ `a4ea67a21`
states the contract as *"VAE(s) (always on GPU)"*
(`vllm_omni/diffusion/models/interface.py:92`).

So the shape in this tree — upload, compute one convolution, download, compute
the next stage on the host — is not the structure upstream has. It is the
`parakeet_encoder.cpp` marshalling shape, which is correct for a single op and
wrong for a decoder.

## Port map

**One code path, and the device is a property of the queue.** Every stage
dispatches through a `vt::` op registered for BOTH `kCPU` and `kCUDA`. This is
the `ltx2_kernels.h` / `Ltx2DitForwardDevice` precedent stated at
`include/vllm/model_executor/models/ltx2_kernels.h:30-34` — *"Registered for BOTH
kCPU … and kCUDA … so the device forward's STRUCTURE is covered by CPU CI and a
GPU is needed to gate the KERNELS, not the port."* A
`queue != nullptr ? device : host` ternary is REFUSED for the reason
`Conv3dThroughSeam` already gives at `:172-178`: it puts the interesting branch
where nothing in a CPU build can execute it.

**The volume becomes a resident buffer: ONE op-dispatch path, two allocation
strategies.** A `VaeVolume` exposes a `vt::Tensor` on the queue's device and every
stage dispatches a `vt::` op over it. How that tensor is BACKED depends on the
device, and this is the shape W5 already argued for and a reviewer already
accepted at `Conv3dThroughSeam:192-206`: on `kCPU` a tensor is already a view
over host memory, so the CPU arm allocates nothing and *"moves no byte it did not
move before this seam existed"*; on a real device the volume owns a backend
allocation that lives from `conv_in` to `unpatchify`. The compute path does not
branch — only the allocator does — so the CPU arm keeps its no-regression
property while the device arm becomes resident.

The latent is uploaded once at `conv_in` and the frames are downloaded once after
`unpatchify`.

**The weights are staged once.** `Ltx2VaeDeviceWeights` uploads each named tensor
of an `Ltx2VaeWeights` to the queue's device and holds it for the life of the
decode, keyed by the same upstream parameter name the host map uses. Two
in-tree precedents, and this row mirrors the second because it is the same
shape of object: `Ltx2StageDitWeightsToDevice`
(`include/vllm/model_executor/models/ltx2_device.h:110-112`), and
`StageMiniMaxH3VideoVaeWeights`
(`src/vllm/model_executor/models/minimax_h3_video_vae_device.cpp:138`) — an
existing **video VAE decoder staged once onto a queue's device**, with a
`storage` vector of `shared_ptr<void>` owning the allocations and plain
`vt::Tensor` views beside it (`include/vllm/model_executor/models/minimax_h3.h:907`).

**The kernels.** A new `vt::OpId::kLtx2Vae` op whose payload is a struct of typed
launchers, inventoried as `KERNEL-LTX2-VAE` in
[`kernel-matrix.md`](../kernel-matrix.md) and claimed by
`CLAIM-LTX25-VAE-DEVICE-RESIDENCY`, appended before `kCount` so no existing op id shifts, in a thin
vt-only header for the reason `ltx2_kernels.h:6-10` gives (nvcc must not parse
`nlohmann/json.hpp`). Each entry names the host helper it replaces and the
upstream line that helper came from:

| Kernel | Replaces | Upstream, at `fd4ded7f2` |
|---|---|---|
| `pixel_norm` | `PixelNorm` (`:381`) | `model/common/normalization.py:32-40` (`forward`); the default `eps = 1e-8` at `:22` |
| `group_norm3d` | the `MiniMaxH3GroupNorm3d` arm of `ApplyNorm` (`:443`) | `model/video_vae/normalization.py:1` re-exports `build_normalization_layer` from `model/common/normalization.py` |
| `ada_ln` | `ApplyAdaLn` (`:529`) | `model/video_vae/resnet.py:135-148` — `scale_shift_table + timestep`, then `x * (1 + scale) + shift` |
| `spatial_noise` | `FeedSpatialNoise` (`:507`) | `model/video_vae/resnet.py:104-119` — ONE `[H, W]` draw, broadcast over channels and time, scaled per channel |
| `depth_to_space` | `DepthToSpaceUpsample::expand` (`:605`) | `model/video_vae/sampling.py:112-118` — `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` |
| `frame_slice` | `DepthToSpaceUpsample::drop_first_frame` (`:632`) | `model/video_vae/sampling.py:119-120` — `x[:, :, 1:]` when `stride[0] == 2` (`:121-122` is the residual ADD, not the slice) |
| `channel_repeat` | the residual repeat (`:660-670`) | `model/video_vae/sampling.py:108` — `x_in.repeat(1, num_repeat, 1, 1, 1)` |
| `linear_cn` | `Linear3d` (`:339`) | `model/video_vae/convolution.py:84-85` — `make_linear_nd` for `dims == 3`, a 1x1x1 `Conv3d` |
| `unpatchify` | the decode tail (`:945-970`) | `model/video_vae/ops.py:35-60` — `b (c p r q) f h w -> b c (f p) (h q) (w r)` |
| `causal_pad` | the pad loop inside `CausalConv3d` (`:274`) | `model/video_vae/convolution.py:305-311` — `k_t - 1` copies of frame 0, then `k // 2` spatial padding in `spatial_padding_mode` |

`Silu` reuses the EXISTING `vt::OpId::kLtx2` entry `Ltx2Silu`
(`src/vt/cpu/cpu_ltx2.cpp:188`) rather than growing a second one, and the residual
adds reuse `vt::Add` (`include/vt/ops.h:2440`).

**Every CPU arm is the existing host loop moved, in the same association order.**
That is what makes the committed goldens (`tests/vllm/models/ltx2_vae_goldens.inc`)
the red-before/green-after instrument rather than a new tolerance: if an arm's
arithmetic drifts, the goldens say so.

## Risks

1. **A moved loop that changes association order changes the pixels.** Mitigated
   by moving the loops verbatim and by the goldens.
2. **A `DevBuf` freed between launch and synchronize** is a use-after-free that
   no CPU box can see. `Conv3dThroughSeam:222-230` already records this and
   [`ltx2-device-staged-view-uaf.md`](ltx2-device-staged-view-uaf.md) is the
   row that found it before. Every resident buffer here outlives the decode by
   construction, which removes the class rather than avoiding an instance.
3. **The CUDA arms cannot be compiled or executed by this row.** That is
   [#1452](https://github.com/mudler/vllm.cpp/issues/1452)'s standing limit for
   `cuda_conv3d.cu` and it applies identically to every kernel added here. See
   `## What a CPU-only run can and cannot establish`.

## Gates

The residency claim is falsifiable WITHOUT a GPU, and this is the whole reason
the row is gateable at all. `tests/vllm/multimodal/test_diffusion_device_seam.cpp`
already carries a `FakeXpuBackend` (`:63`) and W5 already uses
`vt::OpProviderStats` to prove a dispatch went to the queue's device and not to
`kCPU` (`:285-305`). This row extends that harness:

1. **Bit-identity.** The tiny decode on the `kXPU` queue is `memcmp`-identical to
   the same decode on the CPU queue. This is W5's existing assertion, kept.
2. **Exclusive dispatch.** Every stage op selects on `kXPU` and ZERO on `kCPU`.
   Asserting only that the xpu counter moved would pass an implementation that
   ran both arms — W5's own note at `:300-302`.
3. **RESIDENCY, as a COUNT.** `FakeXpuBackend` counts `Copy` calls. The decode
   performs a number of host-device transfers that is a function of the WEIGHT
   COUNT and not of the STAGE COUNT. The gate asserts an exact number and the
   test states the arithmetic that produces it. **This is the assertion that
   fails today**: at this base the count is proportional to the convolution
   count, because each convolution copies its input, weight and bias up and its
   output down.
4. **Every kernel is dispatched on a non-CPU queue by SOME case.** One fixture
   cannot do it: the stages are mutually exclusive by config. `MakeStagedDecoder`
   carries GroupNorm, timestep conditioning and noise injection; its complement
   `MakeShortcutDecoder` carries pixel-norm, a channel-halving `res_x_y` (so the
   shortcut's `Linear3d` runs) and a residual `compress_all` at temporal stride 2
   (so the channel repeat and the frame slice run). Measured by instrumenting
   each CPU arm with its queue's device type: **10 of 10 distinct kernels reach
   `kXPU`**, where the first draft of this row reached only four.

**What assertion 3 does NOT do, and what was claimed for it.** An earlier draft
of this spec said the seam test "runs every one of them on a non-CPU queue and
requires the pixels to match", presenting the `memcmp` as a correctness gate on
the kernels. It is not. The test registers the device op as the SAME FUNCTION
POINTER as the CPU op (`RegisterOp(kLtx2Vae, kXPU, GetOp(kLtx2Vae, kCPU))`), so
both arms run identical code and the comparison can only catch MARSHALLING
defects -- a wrong extent, a missing upload, a dispatch that fell back to the
host. A fresh review confirmed this by mutating all twelve kernel arms: every one
left the seam test green while reding the goldens. The kernels' arithmetic is
gated by `test_ltx2_vae`, and only there.

**A gate this row does NOT have.** "Decoding twice performs no further weight
upload" was listed here as an assertion and was never implemented, and it cannot
be while `VaeWeightCache`'s lifetime is one decode -- a second decode restages by
construction. Within one decode the cache does deduplicate, because
`scale_shift_table` is fetched twice per resnet block, and deleting the lookup
reds assertion 3. Across decodes there is nothing to assert until the cache is
hoisted to load time, which `## Owed` records.

Focused gate:

```sh
cmake --build build --target test_diffusion_device_seam test_ltx2_vae
ctest --test-dir build -R 'diffusion_device_seam|ltx2_vae' --output-on-failure
```

There is deliberately **no separate `test_ops_ltx2_vae`**. A unit test that fed
each kernel a hand-built buffer would gate the class and not the capability: it
would stay green with the decode's call site deleted. The kernels are gated
where they are REACHED -- `test_ltx2_vae`'s committed goldens run every one of
them on the CPU arm, and `test_diffusion_device_seam`'s two fixtures between them
dispatch all ten on a non-CPU queue. That is the `## Nothing lands dead` rule
applied to a kernel table rather than to a model -- with the limit stated above:
the seam proves the marshalling, the goldens prove the arithmetic.

Full gate: `scripts/agent-preflight.sh`.

## What a CPU-only run can and cannot establish

Stated plainly, because a device arm that only exists cannot be gated by a token
test and this row took no lease.

**A CPU-only run ESTABLISHES:**

* that the resident structure is entered — the volume is allocated on the
  queue's backend, every stage dispatches on the queue's device, and nothing
  falls back to `kCPU`;
* that the resident arm's pixels are bit-identical to the host arm's;
* that the host-device transfer count is bounded by the weight count rather than
  the stage count, and that a second decode re-uploads no weight;
* that each CPU kernel arm reproduces the committed goldens.

**A CPU-only run CANNOT establish**, and each of these is `PENDING` on a lease:

* that any CUDA kernel added here compiles. No `nvcc` is in this row's reach and
  CI has no GPU runner — [#1452](https://github.com/mudler/vllm.cpp/issues/1452)
  records that `src/vt/cuda/cuda_conv3d.cu` has never been compiled anywhere in
  this project's reach, and every `.cu` this row adds inherits that limit;
* that the CUDA arms are numerically equal to the CPU arms. Bit-identity is a
  design argument (`__fmul_rn` / `__fadd_rn` against `-ffp-contract=off`), not a
  measurement;
* **any speed or memory number whatsoever.** The round-trip cost this row removes
  is UNMEASURED at this base and remains unmeasured after it. `FakeXpuBackend` is
  a `memcpy` over `malloc` on a unified-memory box, so its transfer count is a
  STRUCTURAL fact and its cost is zero. Nothing here may be quoted as a speedup.

## Tests to port

**There is nothing to port, and that is a finding rather than an omission.**
Lightricks/LTX-2 @ `fd4ded7f2` ships no unit test for any of these ten stages:
`packages/ltx-core` carries no test for `PixelNorm`, `AttnBlock3D`,
`DepthToSpaceUpsample`, `CausalConv3d`'s padding or `unpatchify`, and vLLM
registers nothing LTX at all, so `AGENTS.md`'s "port the upstream tests in the
same change" has no upstream artifact to take.

What stands in for them, and why it is stronger here than a ported unit test
would be: every kernel is a host loop this repository already had, and those
loops are gated by `tests/vllm/models/ltx2_vae_goldens.inc` — goldens generated
from the reference implementation by `scripts/gen-ltx2-vae-goldens.py`. Because
the transcription changes no arithmetic, those committed goldens are a
regression gate on the MOVE, at the oracle's own numbers. A fresh review mutated
all ten arms and each one red 2-8 golden cases, which is the evidence a ported
test would have been asked to produce.

The two things the goldens do NOT bound are recorded under `## Owed`: the f64
accumulator width in `group_norm`, and anything about the CUDA arm.

## Dependencies

* **`vt::OpId::kConv3d`** — W5 (#1007), `KERNEL-CONV3D`. This row is the other
  half of that one and is meaningless without it: `kConv3d` put the decode's
  convolution on a device, and this row stops the volume returning to the host
  between two of them. Its accumulation-order contract is frozen here.
* **`vt::OpId::kLtx2`'s `silu`** (`src/vt/cpu/cpu_ltx2.cpp:188`) — reused rather
  than duplicated; a second ungated SiLU would be the parallel path
  `## Shared seams` forbids.
* **`vt::Add`** (`include/vt/ops.h:2440`) — both residual accumulates. Chosen
  over an eleventh table entry for the same reason.
* **The oracle pin.** Lightricks/LTX-2 @ `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`,
  the secondary-oracle branch of `AGENTS.md` (vLLM implements nothing here).
  Every anchor in `## Port map` was read at that revision in a clean checkout whose
  `git rev-parse HEAD` was asserted first.
* **A GPU lease** — for the CUDA arm only, and NOT taken. Everything this row
  claims is reachable without one; everything it cannot claim is listed in
  `## What a CPU-only run can and cannot establish`.

## Work breakdown

| Wave | Content | State |
|---|---|---|
| **A** | The ten-kernel `kLtx2Vae` table with CPU and CUDA arms; `VaeStore`, `VaeWeightCache` and `VaeScratch`; the decode resident from `conv_in` to `unpatchify`; the residency gate as a transfer COUNT; the `KERNEL-LTX2-VAE` inventory row | **LANDED** in this pull request |
| **B** | `AttnBlock3d` on the device, which needs an attention rung selected (`vt::Attention` op 18 against `AttentionDenseFast` / `DenseFlash` / `DenseFa2`) and changes the numbers, so it needs its own red-first re-gate | owed, #1451 stays open |
| **C** | Hoist `VaeWeightCache` to `Ltx2VideoEngine::Load` so a tiled render stages the decoder once per RENDER rather than once per tile, which is where upstream stages it | owed |
| **D** | A tiled-decode case on a fake accelerator queue, so the production chain's queue is gated at the entry point rather than at the model function | owed; a fresh review proved nothing catches it today |
| **E** | The CUDA arm compiled and executed, and a CPU-vs-CUDA byte comparison | `PENDING` a lease, inheriting #1452 |
| **F** | The video ENCODER made resident | owed; not reachable from a device queue today |


## Owed

| Item | State |
|---|---|
| [#1451](https://github.com/mudler/vllm.cpp/issues/1451) | Wave A closes the elementwise, norm, sampling and tail stages plus the weight staging. It does NOT close `AttnBlock3d`; see the row below |
| **`AttnBlock3d` stays on the host** | **Owed, declared, and it is this row's staged remainder.** The volume is downloaded before an `attn` decoder block and re-uploaded after it. It is held back deliberately rather than for time: it is the only stage that needs an ATTENTION RUNG selected (`vt::Attention` op=18 naive against `vt::AttentionDenseFast` / `DenseFlash` op=21 / `DenseFa2` op=22, which are separate ops with no selector and no fallback notice — the cause of [#1549](https://github.com/mudler/vllm.cpp/issues/1549) and [#1794](https://github.com/mudler/vllm.cpp/issues/1794)), and it is the only stage whose port CHANGES THE NUMBERS: its softmax and its two 1x1 convolutions have an accumulation order that a shared attention op does not reproduce, so it cannot ride the goldens the other ten kernels ride. It owns its own red-first re-gate and its own fresh review |
| **The weight cache's lifetime is ONE DECODE, and a tiled render decodes per tile** | **Owed, and it is the honest limit of the staging half.** `VaeWeightCache` is constructed inside `Ltx2ConvVideoDecode`, so each weight is uploaded exactly once PER CALL -- which is what the gate measures and what closes the per-convolution re-upload. But `AccumulateTemporalGroup` (`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp:123`) calls that function once per TILE, so a tiled render still stages the decoder once per tile. Upstream stages at BUILD time and never restages (`single_gpu_model_builder.py:273`). Hoisting the cache to `Ltx2VideoEngine::Load` needs an owner on the engine and a lifetime that spans the render, which is a change to the engine rather than to the VAE, and it is not made here |
| **The video ENCODER is still entirely host** | Owed. It shares `CausalConv3d`, `ApplyNorm` and `AttnBlock3d` with the decoder and therefore already reaches every kernel this row added, but no encoder volume is resident. It is not reachable from a device queue today: `SpecOf(config)` is called without one at the single site that builds the encoder's spec, so the host arm is what runs and no silent wrong-memory path exists |
| CUDA compile + CPU-vs-CUDA equality for every kernel added here | `PENDING` on a lease, inheriting [#1452](https://github.com/mudler/vllm.cpp/issues/1452). `src/vt/cuda/cuda_ltx2_vae.cu` has never been compiled or executed anywhere in this project's reach, and its bit-identity with the CPU arm is a design argument (`__fmul_rn` / `__fadd_rn` / `__fdiv_rn` / `__fsqrt_rn` against `-ffp-contract=off`), not a measurement |
| A speed or memory number for the removed round-trips | `PENDING` on a lease. Not claimed, not estimated |
| **No gate can see a stage that leaves the `vt` table altogether** | **Owed, and it is the sharpest limit of this row's instrument.** The exclusive-dispatch pair catches a stage that falls back to the CPU QUEUE, and a fresh review confirmed it catches ALL TEN arms that way. It cannot catch a stage that stops dispatching at all: replacing a kernel call with an arithmetically identical host loop over the volume's resident `.ptr()` leaves the seam suite green 9 of 9 and the goldens green too. `FakeXpuBackend::Alloc` is `std::malloc`, so its "device" memory is host-addressable and the loop simply runs; the arithmetic is unchanged, so no golden moves; `d2h` stays 1 and `cpu.selections` stays 0, because nothing went through the CPU table. **That is precisely the regression this row exists to prevent** -- a stage quietly returning to a host loop -- and nothing here detects it. A discrete-memory fake backend whose pointers are NOT host-addressable would, and that is a change to the shared test backend rather than to this row |
| **NOTHING gates the queue on the PRODUCTION chain** | **Owed, and it is the most important thing a fresh review found.** `AccumulateTemporalGroup` (`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp:123-124`) is the only path from `include/vllm.h` to this decode, and replacing its `queue` argument with `nullptr` -- so a render silently runs the WHOLE decode on the host -- leaves `test_ltx2_vae`, `test_diffusion_device_seam`, `test_ltx2_tiling` and `test_ltx2_video` ALL GREEN. The review applied it as `((void)queue, nullptr)`; the naive `nullptr` fails to compile only because of `-Werror=unused-parameter`, which is an accident rather than a gate. The cause is that no test drives `Ltx2VideoDecodeStreaming` or `Ltx2ConvVideoDecodeTiled` with a non-null queue: production reaches it at `src/vllm/multimodal/ltx2_video.cpp:5325` through `im.on_device ? &*im.queue : nullptr`, which is always `nullptr` on a CPU box, and the seam test enters one level below at `Ltx2ConvVideoDecode`. So this row's residency is gated at the model function and NOT at the entry point `## Nothing lands dead` names. Closing it needs a tiled-decode case on a fake accelerator queue, which is a new fixture rather than an assertion |
| **The f64 accumulator width in `group_norm` is ungated** | **Owed.** Mutating `double mean` to `float mean` in `src/vt/cpu/cpu_ltx2_vae.cpp` builds clean and leaves `test_ltx2_vae` at 45/45 and 3152/3152 green, so nothing protects the width the goldens were actually taken through. The kernel comment now says so instead of asserting a safety that does not exist. A gate would need a reduction-order-sensitive fixture, which is its own row |
| **The pad buffer is zero-filled TWICE on the CPU arm** | **Owed, found by this row's own author while the fresh review was still out, and NOT fixed here.** `VaeStore::Alloc` value-initialises its host `std::vector<float>`, and the `pad` kernel's CPU arm then `std::fill`s the same buffer, so the CPU path makes two O(n) zeroing passes per convolution where the base made one. That contradicts, by a small margin, this row's own claim that the host arm moves no byte it did not move before. The fill CANNOT simply be deleted: on a device the allocation is uninitialised and the zero padding MODE skips its taps rather than writing them, so something has to zero the buffer. The fix is a `VaeStore::AllocZeroed()` used only by `padded` -- host `assign(0)` as now, device `Alloc` plus `Backend::Memset` -- with the fill dropped from both kernel arms and the "output must arrive zeroed" precondition written into `ltx2_video_vae_kernels.h`. It is deferred rather than squeezed in so that one repair pass, with one gate run and one fresh review, handles it together with whatever the fresh review returns |
| [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | still owed by `LTX25-DEVICE-RESIDENCY`; unchanged by this row |
| [#1904](https://github.com/mudler/vllm.cpp/issues/1904) — `DevBuf` is a hand-rolled copy of the shared `DBuf` seam | **Owed, filed in flow by this row, NOT fixed here, and owned by this row.** `DevBuf` (`src/vllm/model_executor/models/ltx2_video_vae.cpp:145-170`) duplicates `vllm::dense_attn::DBuf` (`include/vllm/model_executor/models/dense_device_glue.h:109`), which is this tree's move-only owning device buffer and is routed through the shared `DevicePool` (`device_pool.h:71`) so a per-op `Alloc`/`Free` round does not serialise on the driver. That is a parallel path in the sense `AGENTS.md` `## Shared seams` names. It is NOT fixed in flow because `DBuf` resolves `platforms::GetPlatform(device.type)` through `ResolveDevicePoolPolicy` (`dense_device_glue.h:88`) and THROWS for a device type whose platform was never registered, so switching the video VAE onto it makes a registered platform a new precondition of a decode that does not have one today. That is a behaviour change with its own red-first case and its own review, not a rename |

## Outcome — Wave A

**Measured on this branch, on an x86 CPU box, with no GPU lease.** Every number
below is a COUNT taken by `FakeXpuBackend`, which is a `memcpy` over `malloc`.
None of them is a time.

### The defect, before

`tests/vllm/multimodal/test_diffusion_device_seam.cpp`'s new case, run against
the tree before the residency change:

```
ERROR: CHECK( d2h == 1u ) is NOT correct!
  values: CHECK( 7 == 1 )
  logged: host<-device transfers: 7, host->device transfers: 21
```

Seven host-device downloads for a decode with seven convolutions — one per
`nn.Conv3d`, exactly as W5 left it — and twenty-one uploads.

### After

```
CHECK( d2h == 1u ) is correct!   values: CHECK( 1 == 1 )
CHECK( h2d == 41u ) is correct!  values: CHECK( 41 == 41 )
```

* **The volume is downloaded exactly ONCE**, after `unpatchify`, on a fixture
  that carries GroupNorm, timestep conditioning, ada-LN, noise injection, two
  resnet blocks and a depth-to-space upsample. It is never on the host between
  two convolutions.
* **41 uploads is the DISTINCT TENSOR count**, not the convolution count: 1
  latent + 31 weight tensors staged once each + 9 per-call scratch buffers (five
  timestep embeddings computed on the host, four spatial-noise planes drawn on
  the host). The test writes that arithmetic out rather than recording the
  number.
* **Every stage dispatches on the queue's device and ZERO on `kCPU`.** The
  `kCPU` half is what makes it exclusive: asserting only that the device counter
  moved would pass an implementation that ran both arms.

### Mutations

Each anchor was asserted UNIQUE before it was applied, each mutation was built
before it was run, and the tree was restored and its sha256 re-checked after
every one.

| # | Mutation | Gate | Verdict |
|---|---|---|---|
| M1 | restore W5's per-convolution download and re-upload | `test_diffusion_device_seam` | **RED** on both `d2h == 1` and `h2d == 41` |
| M2 | delete the weight cache's lookup, so every fetch re-uploads | `test_diffusion_device_seam` | **GREEN — the gate did not detect it.** See finding 1 below |
| M2b | the same mutation, against the repaired fixture | `test_diffusion_device_seam` | **RED** on `h2d == 41` |
| M3 | delete the production call site that hands the decode its cache | `test_diffusion_device_seam` | **RED**, refused by name at the first convolution |
| M4 | `pixel_norm` divides per channel instead of forming one reciprocal | `test_ltx2_vae` | **GREEN — the goldens do not discriminate.** See finding 2 |
| M5 | `unpatchify` swaps `q` and `r`, transposing every patch | `test_ltx2_vae` | **RED**, three golden cases |

### The two mutations that stayed green, and what they found

**Finding 1 — M2 exposed a real defect in the change, not only in the gate.**
The first fixture used `kPixelNorm`, no timestep conditioning and no noise
injection, so it never entered `group_norm`, `ada_ln` or `spatial_noise` at all,
and no tensor was ever fetched twice, so a cache and no cache were
indistinguishable. Chasing that green found the actual bug: **eight sites were
passing HOST pointers into kernels dispatched on the queue's device** — the
GroupNorm weight and bias, the ada-LN table and embedding, the noise plane and
its per-channel scale, and both `conv_norm_out` tails. `FakeXpuBackend` is a
`memcpy` over `malloc`, so its "device" memory IS host memory and it executed all
of them without complaint; a discrete GPU would have read unmapped memory. The
repair routes every weight through `VaeWeightCache` and every per-call host
buffer through a new `VaeScratch`, and the fixture now turns those three paths
on. **This is the exact failure the instrument was built to catch and it took a
second mutation to make it catchable — a fixture that does not enter a path
cannot gate it.**

**Finding 2 — M4 is a limit of the goldens, and the comment that claimed
otherwise was corrected.** `pixel_norm` forms one reciprocal per pixel and
multiplies, which is what the host loop did. The kernel header originally said
the committed goldens were "held to the first" form. They are not: the
per-channel divide differs only in the last ulp and the golden tolerance absorbs
it, leaving 45/45 and 3152/3152 green. The comment now says so. The form is kept
because it is the one the replaced loop had, which is what makes the move a
no-op, not because a gate would catch a change.

### Correctness

`tests/vllm/models/test_ltx2_vae.cpp`: **45 cases, 3152 assertions, all green**,
unchanged from the base. That is the whole point of transcribing each host loop
into its kernel rather than rewriting it: the committed goldens are a regression
gate on the MOVE. The resident arm is also `memcmp`-identical to the host arm on
the same fixture.

### What landed, stage by stage

| Stage | Where it runs now | Upstream, at `fd4ded7f2` |
|---|---|---|
| the causal + spatial pad | device (`pad`) | `convolution.py:305-311` |
| `nn.Conv3d` | device (`vt::Conv3d`, W5) | `convolution.py:312` |
| `PixelNorm` | device (`pixel_norm`) | `common/normalization.py:37-40` |
| GroupNorm arm | device (`group_norm`) | `common/normalization.py`, via `video_vae/normalization.py:1` |
| `Silu` | device (`vt::OpId::kLtx2`'s `silu`, reused) | `resnet.py:150` |
| `ApplyAdaLn` | device (`ada_ln`) | `resnet.py:135-148` |
| `FeedSpatialNoise` | device (`spatial_noise`); the DRAW stays host | `resnet.py:104-119` |
| `Linear3d` | device (`linear_cn`) | `convolution.py:84-85` |
| `DepthToSpaceUpsample` | device (`depth_to_space`, `frame_slice`, `channel_repeat`, `vt::Add`) | `sampling.py:108-122` |
| residual accumulates | device (`vt::Add`, reused) | `resnet.py:186`, `sampling.py:122` |
| `unpatchify` | device (`unpatchify`) | `ops.py:35-60` |
| **`AttnBlock3d`** | **HOST — downloads and re-uploads** | `attention.py:58-69` |
| the decode prologue (noise blend, de-normalize) | host, BEFORE the one upload | `conv_video_decoder.py:286-301` |
| the whole ENCODER | host | `video_vae.py:39-336` |

The prologue is host on purpose and costs no round trip: both steps touch the
latent before `conv_in`, so running them there and uploading once afterwards is
the same single transfer. The noise draw has to be host in any case, because
`Ltx2NoiseStream` is this project's reproducibility seam.

### What was rejected, and why

* **A `vt::Silu` of our own.** `vt::OpId::kLtx2` already carries an ungated SiLU
  (`cpu_ltx2.cpp:188`) computing the same `x / (1 + exp(-x))`. A second copy
  would be the parallel path `AGENTS.md` `## Shared seams` forbids.
* **An eleventh VAE kernel for the residual add.** `vt::Add` (`ops.h:2440`) is
  elementwise, may alias in place, and has both arms. Both residual sites use it.
* **`vt::LayerNorm` for the `num_groups == 1` norm.** It normalizes over the LAST
  axis of a row-major `[rows, width]`; this volume is channel-major, so the two
  reduce over different elements and agree only when `spatial == 1`.
* **`vt::Matmul` + `vt::Add` for `Linear3d`.** `vt::Add` broadcasts a rank-1
  operand over the last axis; this bias is per CHANNEL, the FIRST axis of a
  channel-major `[Cout, N]`. The shared pair would need a materialized `[Cout, N]`
  bias — more bytes than the whole operation — or a transpose op this tree does
  not have.
* **Narrowing GroupNorm's f64 accumulators to f32.** They are f64 in
  `MiniMaxH3GroupNorm3d` and every committed golden on four call sites was taken
  through them. Whether f64 is the right mirror of torch is a real question and
  it is not this row's to reopen.
* **A parallel tree reduction in the CUDA `group_norm`.** It would be a different
  sum. The arm is one thread per group; the fast version is owed with the
  measurement.

### Two things this row found that the issue did not say

1. **The pad was a host loop too.** `CausalConv3d` materialized its causal and
   spatial padding with `vt::cpu::ParallelForRows` and then handed the padded
   volume to the device, so even the CONVOLUTION stage was not fully resident.
2. **`DevBuf` was a hand-rolled copy of `dense_attn::DBuf`.** It is filed as
   [#1904](https://github.com/mudler/vllm.cpp/issues/1904) and closed here by
   deletion rather than migration: nothing in this file allocates per call any
   more, so the duplicate is gone. #1904 stays open for the audit it also asks
   for.

## Stop conditions

* Stop and report `NEEDS_DECISION` rather than narrowing the port further than
  the declared `AttnBlock3d` remainder.
* Stop rather than quoting any wall-clock or memory figure. No lease was taken.
* Stop rather than changing a golden to accommodate a moved loop. A moved loop
  that changes a golden is a defect in the move.
