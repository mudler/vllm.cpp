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

## The defect, verified at this base

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

## Why this is a divergence, verified at the oracle pin

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

## Design

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
launchers, appended before `kCount` so no existing op id shifts, in a thin
vt-only header for the reason `ltx2_kernels.h:6-10` gives (nvcc must not parse
`nlohmann/json.hpp`). Each entry names the host helper it replaces and the
upstream line that helper came from:

| Kernel | Replaces | Upstream, at `fd4ded7f2` |
|---|---|---|
| `pixel_norm` | `PixelNorm` (`:381`) | `model/common/normalization.py:32-40` (`forward`); the default `eps = 1e-8` at `:22` |
| `group_norm3d` | the `MiniMaxH3GroupNorm3d` arm of `ApplyNorm` (`:443`) | `model/video_vae/normalization.py:1` re-exports `build_normalization_layer` from `model/common/normalization.py` |
| `ada_ln` | `ApplyAdaLn` (`:529`) | `model/video_vae/resnet.py:135-148` — `scale_shift_table + timestep`, then `x * (1 + scale) + shift` |
| `spatial_noise` | `FeedSpatialNoise` (`:507`) | `model/video_vae/resnet.py:104-119` — ONE `[H, W]` draw, broadcast over channels and time, scaled per channel |
| `depth_to_space` | `DepthToSpaceUpsample::expand` (`:605`) | `model/video_vae/sampling.py:114-120` — `b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)` |
| `frame_slice` | `DepthToSpaceUpsample::drop_first_frame` (`:632`) | `model/video_vae/sampling.py:121-122` — `x[:, :, 1:]` when `stride[0] == 2` |
| `channel_repeat` | the residual repeat (`:660-670`) | `model/video_vae/sampling.py:109` — `x_in.repeat(1, num_repeat, 1, 1, 1)` |
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

## Tests and gates

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
4. **Weights staged once.** Decoding TWICE with the same
   `Ltx2VaeDeviceWeights` performs no further weight upload. A per-call upload
   passes assertion 3 on a single decode and fails this one.

Focused gate:

```sh
cmake --build build --target test_diffusion_device_seam test_ltx2_vae test_ops_ltx2_vae
ctest --test-dir build -R 'diffusion_device_seam|ltx2_vae' --output-on-failure
```

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

## Owed

| Item | State |
|---|---|
| [#1451](https://github.com/mudler/vllm.cpp/issues/1451) | Wave A closes the elementwise, norm, sampling and tail stages plus the weight staging. It does NOT close `AttnBlock3d`; see the row below |
| **`AttnBlock3d` stays on the host** | **Owed, declared, and it is this row's staged remainder.** The volume is downloaded before an `attn` decoder block and re-uploaded after it. It is held back deliberately rather than for time: it is the only stage that needs an ATTENTION RUNG selected (`vt::Attention` op=18 naive against `vt::AttentionDenseFast` / `DenseFlash` op=21 / `DenseFa2` op=22, which are separate ops with no selector and no fallback notice — the cause of [#1549](https://github.com/mudler/vllm.cpp/issues/1549) and [#1794](https://github.com/mudler/vllm.cpp/issues/1794)), and it is the only stage whose port CHANGES THE NUMBERS: its softmax and its two 1x1 convolutions have an accumulation order that a shared attention op does not reproduce, so it cannot ride the goldens the other ten kernels ride. It owns its own red-first re-gate and its own fresh review |
| CUDA compile + CPU-vs-CUDA equality for every kernel added here | `PENDING` on a lease, inheriting [#1452](https://github.com/mudler/vllm.cpp/issues/1452) |
| A speed or memory number for the removed round-trips | `PENDING` on a lease. Not claimed, not estimated |
| [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | still owed by `LTX25-DEVICE-RESIDENCY`; unchanged by this row |
| [#1904](https://github.com/mudler/vllm.cpp/issues/1904) — `DevBuf` is a hand-rolled copy of the shared `DBuf` seam | **Owed, filed in flow by this row, NOT fixed here, and owned by this row.** `DevBuf` (`src/vllm/model_executor/models/ltx2_video_vae.cpp:145-170`) duplicates `vllm::dense_attn::DBuf` (`include/vllm/model_executor/models/dense_device_glue.h:109`), which is this tree's move-only owning device buffer and is routed through the shared `DevicePool` (`device_pool.h:71`) so a per-op `Alloc`/`Free` round does not serialise on the driver. That is a parallel path in the sense `AGENTS.md` `## Shared seams` names. It is NOT fixed in flow because `DBuf` resolves `platforms::GetPlatform(device.type)` through `ResolveDevicePoolPolicy` (`dense_device_glue.h:88`) and THROWS for a device type whose platform was never registered, so switching the video VAE onto it makes a registered platform a new precondition of a decode that does not have one today. That is a behaviour change with its own red-first case and its own review, not a rename |

## Stop conditions

* Stop and report `NEEDS_DECISION` rather than narrowing the port further than
  the declared `AttnBlock3d` remainder.
* Stop rather than quoting any wall-clock or memory figure. No lease was taken.
* Stop rather than changing a golden to accommodate a moved loop. A moved loop
  that changes a golden is a defect in the move.
