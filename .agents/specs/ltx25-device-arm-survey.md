# SPEC — `LTX25-DEVICE-ARM-SURVEY`: which LTX-2.5 device arm is worth taking

Issue: [#2478](https://github.com/mudler/vllm.cpp/issues/2478).
Owner row: `LTX25-DEVICE-ARM-SURVEY`.

## Scope

`LTX25-ORACLE-ABSOLUTE` closed correctness at `fa9903b86` (`rc` job
`4b0666ee-248c-45fc-9de6-372b6d0c1fab`, `VERDICT PASS`), and
`LTX25-RENDER-CONFIRM` closed the first speed reading (`rc` job `93a60151`,
**302.954 s at n = 3**, 3.230x the oracle).

**The speed reading was NOT taken on the correctness tree, and this row names
the tree it was taken on.** `PROVENANCE` records
`source_sha=790c582bbba45ab0f7b74aafee361e4557a84bf2`
(`feat(LTX25-RENDER-CONFIRM): one harness that verifies the binary it then
times`), from which it built `binary_sha256=600cf798c48ebabe...` in the lease.
`fa9903b86` is an ancestor of that commit by **421 commits**, and the commit
itself is not an ancestor of `origin/main` at `855905f59` — it is a
`LTX25-RENDER-CONFIRM` working commit. Every number below is therefore a
reading on `790c582bb`, and the correctness verdict is a reading on
`fa9903b86`. AGENTS.md asks an evidence table to name its own tree; the two
trees are different and saying otherwise would be false.

**The box was not quiet, and every duration below inherits that.** The same
`PROVENANCE` records pre-render `loadavg` of **1.97 / 10.10 / 12.51** for r1 /
r2 / r3, and the walls track it: **289.002 / 313.335 / 306.525 s**. The tower
arm's share of its own run's wall moves with them — 28.38% / 26.20% / 27.05%.
The phase log's `notice` says to quote a duration only with the host, the
checkpoint and the contention state beside it: the host is `dgx:gpu0`, the
checkpoints are the four `checkpoint_sha256` lines in `PROVENANCE`, and the
contention state is those three load averages. This is why the ranking below is
by **share**, and why the ordering of its top two arms is asserted per run in
`## Gates` rather than off a mean.

What none of this closed is that most of that render runs on the host even
though it was taken with `--device cuda`.

The owed device work is spread over the `## Owed` sections of the `ltx25-*`
specs with no shared denominator. This row supplies one and stops.

IN scope:

- Every owed LTX-2.5 device arm, with the spec that owns it and the exact
  host-pinned site.
- A shape for each: **queue swap**, **PORT**, or **refusal that should stay**.
- A ranking by **measured** cost on the default production render.
- At most one arm taken. **This row takes none**, and `## The decision this row
  returns` is why rather than an omission.

OUT of scope, declared rather than approximated:

- **Every port.** This row measures and classifies. Taking the top arm is a
  decision this row returns rather than makes; see `## The decision this row
  returns`.
- **A new GPU lease.** Every number below is read off phase logs already on the
  NAS from `rc` job `93a60151`. No new measurement was taken and none is
  claimed.
- **A published benchmark ID.** One request at one geometry is a reading.

## The denominator, and why it is the only honest one

Every ranking below is against **the default production render**: the `ltx2-gen`
binary, `ti2vid_one_stage`, 320x192 / 25 frames / 8 steps, **`--device cuda`**,
GB10 (`dgx:gpu0`), the request `ltx2_oracle_manifest.json` pins. That is the
entry point `## Nothing lands dead` names, and it is the one the correctness
verdict and the speed reading were both taken on.

The source is the three `phase-log.json` files at
`/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z/{r1,r2,r3}/`.
`"device": "cuda"` in each, and `peak_device_bytes` reaches 127.87 GB on `load`,
so the DiT really was staged and really did denoise on the accelerator. Leaf
durations are summed per name per run and then averaged over the three runs.

| leaf | mean s, n = 3 | % of 302.954 | runs on |
|---|---:|---:|---|
| `load` (span) | 94.550 | 31.21% | host file I/O + dequant + DiT staging |
| — `load.text_encoder` | 55.160 | 18.21% | host |
| — `load.dit` | 36.110 | 11.92% | host read, **device stage** |
| `generate.guiders` | 95.506 | 31.52% | **host** |
| — `guiders.tower` | 55.322 | 18.26% | **host** |
| — `guiders.connector.compute` | 40.104 | 13.24% | **host** |
| `generate.conditioning` (span) | 76.977 | 25.41% | **host** |
| — `conditioning.tower` | 27.014 | 8.92% | **host** |
| — `conditioning.connector` | 49.960 | 16.49% | **host** |
| —— `conditioning.connector.compute` | 41.234 | 13.61% | **host** |
| —— `conditioning.connector.weights` | 8.726 | 2.88% | host I/O, not an arm |
| `denoise` | 15.122 | 4.99% | **device** |
| `decode.video` | 10.927 | 3.61% | convolutions device, everything else host |
| — `decode.video.vae` | 10.916 | 3.60% | as above |
| `decode.audio` | 8.773 | 2.90% | **host** |
| — `decode.audio.mel` | 4.926 | 1.63% | **host** |
| — `decode.audio.vocoder` | 3.846 | 1.27% | **host** |
| `artifacts.frames` | 0.895 | 0.30% | host |

**The one sentence this row exists to write: the DiT is the only MODEL on the
accelerator and `denoise` is the only PHASE whose compute is all there, at 4.99%
of the render.** The qualifier is load-bearing and the table above is what
forces it: `load.dit` stages to the device, and `decode.video` runs its
`kConv3d` convolutions on a real device queue between host loops (§3). Neither
is a model that lives on the accelerator, and neither makes `denoise` less than
the whole of what does. 56.93% is text conditioning on a host queue: two tower
passes and two connector passes, which are almost exactly the same size as each
other and are **not the same kind of work**.

The log carries no `guiders.connector` leaf, only `guiders.connector.compute`,
and `ConnectorWeightSet` is why: it caches the materialized weights across the
two passes, so the negative pass has no weight cost to bracket. The two named
leaves account for 95.426 s of `generate.guiders`' 95.506 s.

## The inventory

Read off the `## Owed` sections of the `.agents/specs/ltx25-*.md` files and then
checked against `origin/main` at `ed5ecea2e`. **The count is stamped with a SHA
because it is otherwise unauditable:** the glob returns **50** at `ed5ecea2e`,
the base this branch was cut from and the tree every `file:line` below is
resolved against, and **51** at this branch's head, the one addition being this
spec. An auditor reruns the glob at either SHA and reconciles it; a bare count
with no tree beside it cannot be reconciled at all.

Cost is the leaf sum above, so it is a **ceiling** on what moving the arm can
win, never a projection of the win.

| # | arm | owning spec | measured cost on the default render | shape | reached by the default render? |
|---|---|---|---:|---|---|
| 1 | **The Gemma-4 text tower, both passes** | `ltx25-text-linear-seam`, `ltx25-decode-speed` | `conditioning.tower` + `guiders.tower` = **82.336 s / 27.18%** | **QUEUE SWAP + one allocation + 49 downloads per pass** | yes |
| 2 | **The connector, both passes** | `ltx25-connector-repair`, `ltx25-connector-gemm`, `ltx25-text-cond-device` | `conditioning.connector.compute` + `guiders.connector.compute` = **81.338 s / 26.85%** | **PORT** | yes |
| 3 | The video VAE's non-convolution volume | `ltx25-device-residency`, `ltx25-vae-device-residency` | `decode.video.vae` = **10.916 s / 3.60%** | PORT (partial: the convolutions are already device) | yes |
| 4 | The audio VAE decode through `vt::Conv2d` | `ltx25-audio-decode-cost` | `decode.audio` = **8.773 s / 2.90%** | queue swap, but COUPLED to a golden-changing accumulator change | yes |
| 5 | The video VAE **encoder**'s queue | `ltx25-vae-device-residency`, `ltx25-device-residency` | **0 s** — the encoder does not run on this recipe | queue swap | no |
| 6 | The T2A one-stage device forward | `ltx25-t2a-one-stage` | **0 s** — different entry point | PORT | no |
| 7 | STG on the T2A device forward | `ltx25-t2a-one-stage` | 0 s — rides #6 | rides #6 | no |
| 8 | `CudaBackend::DeviceMemoryInfo` | `ltx25-device-residency` | 0 s — instrument, not compute | **STALE — the override landed** | yes, and it answers |
| 9 | Narrow dtypes (f16/bf16) in the CUDA `conv3d` | `ltx25-vae-device-residency` | 0 s directly | **REFUSAL THAT SHOULD STAY** until a kernel exists | yes, as a refusal |
| 10 | The CUDA `pad` memset, and `[SKIP]`-as-`PASS` on CPU lanes | `ltx25-vae-device-residency` | 0 s | gate gap, not an arm | no CI lane owns a GPU |
| 11 | The HQ preset's isolated-modality pass on the device forward | `ltx25-res2s-loop` | 0 s | **STALE — already done** | — |

### Arms 1 and 2 are 0.997 s apart, so the ordering is shown per run

The lead is **0.33% of the wall**, and the phase log's own `notice` warns that
"the RANK of its two largest phases has reversed between two such runs" on a
contended box. This box was contended (`## Scope`). A mean is therefore not
enough to put arm 1 first, so the per-run column is here and `## Gates` asserts
the ordering run by run.

| arm | r1 | r2 | r3 | mean | spread | % of mean wall |
|---|---:|---:|---:|---:|---:|---:|
| 1, tower = `conditioning.tower` + `guiders.tower` | 82.014 | 82.090 | 82.903 | **82.336** | 1.08% | 27.18% |
| 2, connector compute = `conditioning.connector.compute` + `guiders.connector.compute` | 81.165 | 80.961 | 81.888 | **81.338** | 1.14% | 26.85% |
| **arm 1 − arm 2** | **+0.849** | **+1.129** | **+1.014** | **+0.997** | 28.09% | 0.33% |

**The lead holds 3 for 3 and is never smaller than 0.849 s**, so the ordering is
a measurement rather than a rounding. It is still 0.33% of the wall against a
28.09% spread of its own, and **the decision below picks arm 1 on shape, not on
this second**: were the two to swap, arm 1 would still be the queue swap and arm
2 would still be the 386-line port.

### 1. The text tower — the top of the ranking, and the cheapest shape

**There are THREE CPU `text_queue` sites in this file and only two of them are
on the default render.** Each builds

```cpp
vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
```

and the arm is the two that `Ltx2VideoEngine::Generate` (opens at `:2130`)
contains:

| site | enclosing function | what it feeds | measured |
|---|---|---|---:|
| `ltx2_video.cpp:2499` | `Ltx2VideoEngine::Generate` | the positive pass, bracketed by `phase::Scope tower_phase("conditioning.tower")` at `:2500` | **27.014 s** |
| `ltx2_video.cpp:3229` | `Ltx2VideoEngine::Generate`, the negative-prompt branch | the negative pass, bracketed by `const phase::Scope guiders_tower("guiders.tower")` at `:3240` | **55.322 s** |
| `ltx2_video.cpp:5797` | `Ltx2VideoEngine::GenerateAudioOnly` | the T2A entry point, arms 6/7 below | **0 s here** |

**`:3229` is the larger half by more than 2x**, and `"guiders.tower"` occurs
exactly once in `src/` (`ltx2_video.cpp:3240`), so the attribution is not
ambiguous. An implementer who swapped `:2499` and `:5797` would move 27.014 s of
the 82.336 s, leave the largest leaf on the host, and find §2 of `## The
decision this row returns` waiting to explain the shortfall as the known
unmeasured split. It is not that: it is a missed call site, and naming all three
here is what stops the bound from absorbing it.

`:2473-2479` states the host queue as a limit: "the text path has no device arm
to run on". **That sentence is now false about the tower**, and correcting it is
this row's main finding.

`Ltx2EncodePromptToConditioning` reaches the tower through
`Gemma4Model::ForwardHiddenStates(..., queue)`
(`src/vllm/model_executor/models/ltx2_text_encoder.cpp:1166`). That function is
`src/vllm/model_executor/models/gemma4.cpp:787`, it builds
`Dev d{vt::GetBackend(queue.device.type), queue}`, and its whole body is
`dense_attn`'s `Dev`/`DBuf`/`ResidentWeight` glue. `ResidentWeight`
(`include/vllm/model_executor/models/dense_attn_block.h:181-206`) uploads and
caches a host `OwnedTensor` into `w.d_dev` for any non-CPU device, so **the
tower's weights stage themselves on first use** and no load-time change is
needed. The compute is a shipped model forward with a device arm, not host
loops.

**That staging is free on GB10 and is not free anywhere else.**
`ltx2_video.cpp:857-864` records the resident tower as "~24 GB of host bf16" and
keeps it resident for the whole engine lifetime because a prompt arrives per
request. GB10's pool is unified, so making those bytes device-resident moves no
byte and costs no second copy; on a discrete accelerator it is 24 GB of VRAM
held beside a DiT that already stages. Any implementer takes that measurement
before claiming the arm is portable off this box.

**What is genuinely missing is one allocation.** `ltx2_text_encoder.cpp:1130-1146`
builds the tower's paged KV cache as

```cpp
std::vector<std::vector<uint16_t>> kv_storage;
...
kv.data = kv_storage.back().data();
```

— host memory — and `KvSlice`
(`dense_attn_block.h:364-380`) then LABELS that pointer with the forward's
device. On GB10 the pool is unified so the label is accidentally true; on a
discrete accelerator it hands a host pointer to a device kernel. That second
half is a correctness obligation rather than a speed one.

**And there is a third item, which is neither: 49 device-to-host transfers per
tower pass.** `ForwardHiddenStates` captures every hidden state through the
`capture` lambda at `gemma4.cpp:604-611`, whose body is
`buf.Download(d, capture_row.data())` (`:607`) followed by a per-element
`vt::BF16ToF32` into a host `std::vector<float>` (`:609`). The LTX call site
requires all of them — `ltx2_text_encoder.cpp:1168` refuses unless
`run.hidden_states.size() == feature_config.num_layers`, and that is
`num_hidden_layers + 1 = 49`
(`include/vllm/model_executor/models/ltx2_text_encoder.h:211`). On the CPU queue
`Download` is a copy inside one address space. On a device queue it is 49
transfers plus 49 host conversions per pass, 98 per render, before the host
feature extractor even starts. **So the shape is three items, not two:** the
queue swap, the device-resident KV allocation, and a download volume the swap
creates rather than removes. Whether the third one has to be paid — the
extractor consumes f32 on the host either way — is part of the same unmeasured
split §2 of `## The decision this row returns` names, and an implementer sizes
it before claiming the arm is cheap.

**The 82.336 s is a ceiling and this row will not pretend otherwise.**
`conditioning.tower` brackets the Gemma-4 forward AND the host feature extractor
that follows it — `Ltx2StackHiddenStates`, `Ltx2NormAndConcatPaddedBatch`, the
`_rescale_norm` pass and the two caption projections, whose `Linear` runs on its
own hard-coded CPU queue at `ltx2_text_encoder.cpp:446` and whose
`Ltx2TextFeatureExtractorForward` refuses any compute dtype but f32 by name
(`:54-56`). **Nothing splits those two halves**, so how much of the 82.336 s the
queue swap can reach is unmeasured. Naming that is the difference between a
sizing and a guess.

### 2. The connector — the same size, a much larger change

`Ltx2ConnectorForward` (`src/vllm/model_executor/models/ltx2_connector.cpp`) is
386 lines of host `std::vector<float>`: `RmsNormRows` at `:48` mutates a
`std::vector<float>&` in place, `weights.Get(name)` returns
`const std::vector<float>&` at `:72` and `:144`, and the attention block at
`:242-273` copies `state.hidden_states` into a host vector between each GEMM.
`LTX25-CONNECTOR-REPAIR` already established that this is a port and this row
does not contradict it. It is the same 27% of the render as the tower for
several times the work.

### 3. The video VAE, and what "the decode runs on the device" does not mean

`vt::OpId::kConv3d` has a CUDA arm and the streaming decode does get a real
queue on this render — `ltx2_video.cpp:5516` passes
`im.on_device ? &*im.queue : nullptr`, which is non-null here. So the **10.916 s
already includes** the device convolutions plus an upload and a download around
each of them, because the norms, SiLU, AdaLN, noise injection,
`DepthToSpaceUpsample`, `AttnBlock3d`, `Linear3d` and `unpatchify` between them
are still host loops. `ltx25-device-residency` records that as owed and this row
attaches the first number to it: **the whole leaf is 3.60% of the render**, so
the arm's ceiling is 3.60% and its realistic win is less.

### 4. The audio decode — the CPU repair changed the answer

`ltx25-audio-decode-cost` owes "the device arm through `vt::Conv2d`", and that
bullet was written when `decode.audio.mel` was **47.175 s**. `4fef1f413`
parallelised the CPU `Conv2d` over output lines and the leaf is now **4.926 s**,
a 9.58x. `decode.audio` entire is **8.773 s, 2.90% of the render**.

**So the device arm here can win at most 2.90% of the wall, and it cannot be
taken on its own**: that spec couples it to replacing the `double` accumulator,
which is a golden-changing correctness change. A 2.90% ceiling behind a
golden-changing prerequisite is not the next thing to do. Recorded as **not
worth porting at the current geometry**, which is a result and not a deferral.

### 5, 6, 7. Arms nothing on this render reaches

The **video VAE encoder** is host-pinned because `SpecOf(config)` is called
without a queue at `ltx2_video_vae.cpp:1595` while the decoder's site at
`:1165` passes one. It is a queue swap plus a queue parameter on
`Ltx2ConvVideoEncode`. It costs **0 s on this render** because `ti2vid` decodes
and never encodes; it is reached by the image- and audio-conditioned ingestion
recipes instead.

The **T2A device forward** refuses by name at `ltx2_video.cpp:5844-5847`
("text-to-audio is not served on the accelerator"), because
`Ltx2DitForwardDevice` takes both streams by reference —
`ltx2_device.cpp:1381` and `:1390` pass `*video` and `*audio` into
`PrepareStreamDev`, and `:1397-1402` read `video->batch` and `video->tokens` per
block. That is a rewrite of the function, so it is a PORT. It is on a different
entry point (`GenerateAudioOnly`), so it moves nothing on the default render.

**The third CPU `text_queue`, `ltx2_video.cpp:5797`, is here and not in arm 1.**
It sits inside `Ltx2VideoEngine::GenerateAudioOnly` (opens at `:5675`), so it is
reached only by the T2A entry point, and the same function then refuses a device
at `:5844`. Swapping that queue would move 0 s of the default render and would
hand a device queue to a path that has already refused the device. It rides this
arm, and it is not part of the tower arm's shape.

### 9, 10. Refusals that should stay, and gate gaps that are not arms

The CUDA `conv3d`'s refusal of f16/bf16 by name should stay until a kernel
exists: a refusal that names the missing part is what this project asks for, and
widening it silently is the defect. `#1939`'s finding that a `[SKIP]`ped doctest
case scores as `passed` under `ctest --output-on-failure`, and that no CI lane
here owns a GPU, is a gate gap that applies to every arm in this table and is
the reason each one needs a lease rather than a CI verdict.

### 11. One owed item is STALE, and the tree says so

`ltx25-res2s-loop` `## Owed` says the HQ preset is host-only because
"`Ltx2DitForwardDevice` takes no `perturbations`". At `ed5ecea2e` it does:
`ltx2_device.cpp:1326` declares
`const Ltx2DitPerturbation* perturbations`, `:1360-1366` validates it, and
`:1427-1433` sets all four flags per block. `ltx25-guided-video` `## Owed`
records the same thing as CLOSED by its §12 on 2026-08-19. The `res2s` bullet
was not updated. It owes a record correction, not a port.

### What the sweep found that is NOT an owed device arm

Recorded so the enumeration is auditable rather than a claim. The sweep read the
`## Owed` section of every `.agents/specs/ltx25-*.md` file — **50 of them at
`ed5ecea2e`, 51 at this branch's head with this spec added** — and grepped each
for `device|GPU|CUDA|queue|host-only|CPU-only`. The count carries the SHA
because that is the only form an auditor can rerun; a bare number goes stale the
next time the campaign files a spec, and a stale one cannot say which files it
missed. What matched but is not an arm:

- **A GPU lease, not a device arm.** `ltx25-a2v-audio-input`, `ltx25-bf16-dit`,
  `ltx25-ic-lora`, `ltx25-phase-lora`, `ltx25-prompt-adherence`,
  `ltx25-resolution-envelope`, `ltx25-retake`, `ltx25-ti2vid-recipe`,
  `ltx25-dit-attn-flash` and `ltx25-dit-attn-fa2-hd128` each owe a real-checkpoint
  render or a lease-taken measurement. They are scheduling, not porting.
- **`load.dit` staging.** `ltx25-decode-speed` owes "DiT staging is 7.5 min at
  70.5 MiB/s, GPU idle". On the reference render `load.dit` is **36.110 s** and it
  is the one leaf that got *slower* between the two runs (+1.067 s). The staging
  path already writes to the device; what is owed there is throughput, not an arm.
- **`load.text_encoder`, and it is bigger than `load.dit`.** At **55.160 s /
  18.21%** it is the second-largest single leaf in the table — larger than every
  arm except the top two — so leaving it unclassified beside an explained
  `load.dit` would be the omission `load.dit` is not. It is **not a device arm**:
  it is `safetensors` I/O and dequantisation into the ~24 GB host bf16 tower that
  `ltx2_video.cpp:857-864` then keeps resident for the engine's lifetime. There is
  no compute in it to move. It is also the leaf that arm 1 makes *worse* on a
  discrete accelerator and free on GB10, for the unified-pool reason §1 gives.
- **`load` is paid once per process, so the shares above understate a server.**
  `load` is **94.550 s / 31.21%** of this denominator and a server pays it at
  startup, not per request. Re-based on the per-request remainder, every arm's
  share is **~1.45x** what the table states: the tower is ~39.5% of a warm
  request rather than 27.18%. **The ranking is unaffected**, because the factor
  is common to every arm. The ceilings are not, and a serving row that quotes
  27.18% as its target has understated it. Stated here rather than restated in
  the table, because the table's denominator is the render this row measured and
  changing it silently would break every other number in the document.
- **Instruments and checkers.** `ltx25-device-seam-sibling`'s `dev_cast` leakage
  check is a gate on device work rather than device work.
- **A SECOND stale owed item.** `ltx25-device-residency` records, at length, that
  "**`CudaBackend` does not override `DeviceMemoryInfo` at all**", that
  `grep -rn 'DeviceMemoryInfo' src/vt include/vt` returns the base declaration and
  one ROCm override, and that a CUDA render would therefore print `-1` in every
  row of the device column. **At `ed5ecea2e` that grep returns four hits and one of
  them is `src/vt/cuda/cuda_backend.cu:93`**, landed by `a22030924`
  (`PERF-QWEN35-STAGE-WEIGHTS`, #2327/#2335) for an unrelated reason. The reference
  render confirms it from the other side: `device_bytes_source` reads
  `vt::Backend::DeviceMemoryInfo` and `peak_device_bytes` reaches **127.87 GB**
  rather than the `-1` sentinel. That row owes a record correction, and the
  `Gemma4MoE` device-expert LRU it warned the override would wake is a live
  behaviour change nobody measured when the override landed.
- **A dead arm, not an owed one.** `ltx25-retire-dead-arms` records
  `kMultiGpuParallelism` as having no call site. Retiring it is the opposite of
  this row's question.

## What the CPU repairs took off the table

The parent question was whether the recent host-side repairs make some of these
not worth porting. **They do, and the numbers are these:**

| arm | cost at the #2296 baseline (518.398 s) | cost now (302.954 s) | verdict |
|---|---:|---:|---|
| audio VAE decode | `decode.audio` 50.745 s / 9.79% | **8.773 s / 2.90%** | **not worth porting** at this geometry, and coupled to a golden-changing accumulator change |
| video VAE decode | `decode.video` 15.970 s / 3.08% | **10.927 s / 3.61%** | marginal; a partial port for a 3.60% ceiling |
| the whole text-conditioning block | `conditioning.tower` + `conditioning.connector` + `generate.guiders` = 340.747 s / **65.73%** | **172.483 s / 56.93%** | still more than half the render, and still entirely on the host |
| — of which the tower | `conditioning.tower` 28.343 s / 5.47%; **the negative pass's tower was not a leaf** and is unmeasurable from that table | **82.336 s / 27.18%** | **the largest single share on the `.compute` accounting**, and the cheapest shape |
| — of which the connector | not separable at that baseline; `#2354` measured the compute at 224.882 s in its own run | **81.338 s / 26.85%** compute | joint-largest, still a PORT |

**"The largest single share" depends on costing the connector at `.compute`
only, and this row says so rather than letting the headline carry the
exclusion.** `conditioning.connector.weights` (8.726 s) is charged to host I/O
rather than to arm 2 because it is a re-read of the connector tensors out of the
DiT file, not connector compute — `ConnectorWeightSet` caches it across the
two passes, which is why the negative pass has no weights leaf at all. That is
disclosed in the inventory and it is defensible. It is also decisive for the
word "largest": add it back and `conditioning.connector` +
`guiders.connector.compute` is **89.854 / 89.528 / 90.811 s**, mean **90.064 s /
29.73%**, which exceeds the tower in **all three runs**. The ranking's shape
does not move — the tower stays the queue swap and the connector stays the
port — but the superlative is accounting-dependent, and a reader who charges
the weights to the arm should read arm 2 as the larger one.

**The two "of which" rows are not a before/after on the same instrument.** The
#2296 binary predates the `.compute` and `guiders.tower` splits, so its tower and
connector costs are folded together inside `generate.guiders`. What is comparable
is the block total, and it fell 1.98x while the wall fell 1.71x — the text path
got *better* than the render and is still the majority of it.

The tower is the whole shape of the campaign after the repairs. `conditioning.tower`
barely moved in seconds -- 28.343 to 27.014 -- while everything around it
collapsed, so its share nearly doubled.
`component-speedup-is-not-system-speedup-fixed-serial-term` running in the other
direction: the term that did not move became the term that decides the render.

## A finding in flow: the tower computes a 262144-wide logit vector nobody reads

Found while establishing arm 1's shape, and recorded here with its sizing
because AGENTS.md requires an in-flow finding to name an owner rather than wait
for a sweep.

`Gemma4Model::ForwardHiddenStates` returns both `hidden_states` and `logits`,
and `ForwardBody` (`gemma4.cpp:733-756`) computes the tied lm_head
unconditionally over **every** row unless `logits_indices` is non-empty and
shorter than `T` (`:738-739`). `ltx2_text_encoder.cpp:1166` calls it **without**
`logits_indices`, so `do_gather` is false, and the call site then reads only
`run.hidden_states` (`:1167-1187`). The `[T, 262144]` f32 buffer, the
`MatmulBT` against the `[262144, 3840]` tied embedding table, the `SoftCap` and
the full `Download` into `out.logits` are all discarded.

Sizing, stated as arithmetic and **explicitly not as a measurement**: at the
shipped `hidden_size = 3840`, `vocab_size = 262144`, `num_hidden_layers = 48`,
`intermediate_size = 15360` (`tests/vllm/models/ltx2_gemma4_text_config.json`),
the lm_head is `2 * 3840 * 262144 = 2.01 GFLOP` per token against roughly
`2.16e10` FLOP per token for the 48 layers — **about 8.5% of the tower's GEMM
work**, on both passes, on the host, twice per render.

**Applied to the 82.336 s of measured tower that is 7.0 s, and 7.0 s is an UPPER
BOUND, not a midpoint.** Two independent reasons, and both push the same way.
First, the 82.336 s brackets more than the Gemma-4 forward: `conditioning.tower`
also contains `Ltx2StackHiddenStates`, `Ltx2NormAndConcatPaddedBatch`, the
`_rescale_norm` pass and the two caption projections (§1), and the lm_head is
8.5% of the *forward*, not of the leaf. Second, the ratio assumes **equal
achieved FLOP/s** between one `[T, 3840] x [262144, 3840]` `MatmulBT` and 48
layers of far smaller GEMMs, and a single very large well-shaped GEMM is the one
most likely to beat the small ones on a host BLAS path — which makes the true
share of *time* smaller than the share of *FLOP*. So read the range as **"up to
~7 s, ~2.3% of the wall"** rather than "6 to 7 s". It is arithmetic, it is
labelled as arithmetic, and the number a lease would return is at or below its
top end. The change also removes a `T * 262144 * 4`-byte allocation and download
per pass, which no FLOP count sees at all.

Passing `logits_indices = {0}` at the one call site removes it and cannot change
the conditioning, because `capture(dnorm)` (`gemma4.cpp:731`) records the final
hidden state **before** the lm_head. **The gate is the open question**, not the
change: a test can hold the seam (`out.logits.size() == vocab` rather than
`T * vocab`), but holding the LTX call site itself needs an instrument this tree
does not expose, and a mutation that moves nothing is not evidence. That gate
design is why this is filed rather than squeezed in here.

Owner: **owed, unowned.** Issue: [#2479](https://github.com/mudler/vllm.cpp/issues/2479).

## The decision this row returns

**`NEEDS_DECISION`.** The ranking justifies exactly one arm — the text tower at
27.18% — and it is the cheapest shape in the table, but it is not a contained
change and this row will not start it silently:

1. **It needs a device-resident KV cache**, which is a correctness change on a
   discrete accelerator, invisible on the GB10 the campaign measures on. That is
   the `on-disk ≠ runtime` class of defect, in memory space.
2. **Its win is bounded by an unmeasured split** inside `conditioning.tower`
   between the Gemma-4 forward and the f32 host feature extractor that follows
   it. The extractor refuses non-f32 by name and would stay on the host, so the
   ceiling above is loose by an unknown amount.
3. **Its correctness bar needs a lease.** Byte equality is unavailable across a
   CPU/device boundary; the bar is the [#1864](https://github.com/mudler/vllm.cpp/issues/1864)
   blockiness gate plus the same-arm pixel comparison, on a full render on
   `dgx:gpu0`, and that gate's ceiling is one-sided.

The cheap step that makes (2) a number rather than a bound is one instrument
change in the same file: split `conditioning.tower` and `guiders.tower` into a
`.gemma` leaf and an `.extract` leaf. It rides the phase log that already ships
on the default path and needs the same lease, so it should ride whichever render
the tower arm's implementer takes.

## Gates

```sh
python3 - <<'PY'
import json
from collections import defaultdict
agg = defaultdict(list)
for r in ('r1', 'r2', 'r3'):
    d = json.load(open(f'/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z/{r}/phase-log.json'))
    assert d['device'] == 'cuda', d['device']
    tot = defaultdict(float)
    for p in d['phases']:
        tot[p['name']] += p['duration_seconds']
    for k, v in tot.items():
        agg[k].append(v)
mean = {k: sum(v) / len(v) for k, v in agg.items()}
tower = mean['conditioning.tower'] + mean['guiders.tower']
conn = mean['conditioning.connector.compute'] + mean['guiders.connector.compute']
assert abs(tower - 82.336) < 0.01, tower
assert abs(conn - 81.338) < 0.01, conn
assert abs(mean['denoise'] - 15.122) < 0.01, mean['denoise']
print('tower %.3f  connector %.3f  denoise %.3f' % (tower, conn, mean['denoise']))

# THE ORDERING, PER RUN. The three means above cannot see a rank flip, and the
# phase log's own `notice` says the rank of a render's two largest phases has
# reversed between two runs on a contended box. Arm 1 leads arm 2 by 0.33% of
# the wall, so assert the lead in EVERY run and print the margin.
leads = []
for r in ('r1', 'r2', 'r3'):
    d = json.load(open(f'/mnt/nas_share/rc/ltx25-render-confirm/run/20260901T075837Z/{r}/phase-log.json'))
    tot = defaultdict(float)
    for p in d['phases']:
        tot[p['name']] += p['duration_seconds']
    t = tot['conditioning.tower'] + tot['guiders.tower']
    c = tot['conditioning.connector.compute'] + tot['guiders.connector.compute']
    assert t > c, ('ARM 1 NO LONGER LEADS ARM 2', r, t, c)
    leads.append(t - c)
    print('%s tower %.3f  connector %.3f  lead %+.3f  wall %.3f' %
          (r, t, c, t - c, d['wall_seconds']))
assert min(leads) > 0.5, min(leads)
print('min lead %.3f  mean lead %.3f' % (min(leads), sum(leads) / len(leads)))
PY
```

**That last block is the ordering gate and it is the reason arm 1 is first.** A
rank flip fails it by name rather than leaving a paragraph to be re-read, and
the printed margin is what a reader checks when the box changes: a lead that
shrinks toward zero is the signal that this ranking has stopped discriminating,
which a tolerance on the two means alone would never show.

The tree-side claims are held by `git grep` at `ed5ecea2e` and each is cited by
`file:line` in the sections above. They are anchors, not a gate, and this row
says so rather than dressing a transcription as one.

## Owed

- **The text tower's device arm, 82.336 s / 27.18%, is the top of the ranking
  and is NOT taken here.** Queue swap at **`ltx2_video.cpp:2499` and `:3229`** —
  both inside `Ltx2VideoEngine::Generate`, feeding `conditioning.tower` (27.014 s)
  and `guiders.tower` (55.322 s) respectively; **`:5797` is NOT this arm**, it is
  `GenerateAudioOnly`'s and belongs to arms 6/7. Plus a device-resident KV cache
  at `ltx2_text_encoder.cpp:1130-1146`, plus sizing the 49 per-pass downloads at
  `gemma4.cpp:604-611`, plus the `#1864` verdict and a pixel comparison on a
  `dgx:gpu0` lease. Owner: unowned; sizing is here. Issue:
  [#2478](https://github.com/mudler/vllm.cpp/issues/2478).
- **The split of `conditioning.tower` into `.gemma` and `.extract`.** Without it
  the 82.336 s stays a ceiling rather than a target. One phase scope each in
  `ltx2_text_encoder.cpp`; it needs a render to produce a number. Owner:
  unowned.
- **The discarded lm_head at `ltx2_text_encoder.cpp:1166`**, sized above at ~8.5%
  of the tower's GEMM work. The change is one argument; the gate that holds the
  call site is the open part. Owner: unowned. Issue:
  [#2479](https://github.com/mudler/vllm.cpp/issues/2479).
- **`ltx25-res2s-loop`'s stale `## Owed` bullet.** The device forward does take
  `perturbations`. A record correction, owned by that row.
- **`ltx25-device-residency`'s stale `## Owed` bullet.** `CudaBackend` DOES
  override `DeviceMemoryInfo` at `src/vt/cuda/cuda_backend.cu:93` since
  `a22030924`. A record correction, owned by that row -- and it owes the
  `Gemma4MoE` device-expert LRU measurement that row said the override would
  require.
- **No new measurement.** Every number here is read from `rc` job `93a60151`'s
  artifacts. This row took no lease and rendered nothing.
- **Four issue numbers this row cites could not be resolved through `gh`** —
  1005, 1269, 1451 and 2354 each returned "Could not resolve to an issue" while
  2405 and 2457 resolved in the same command. That is `REMOTE_UNVERIFIED` and
  never absence, so this row cites them from the specs that name them and
  asserts nothing about their state.

## Outcome

**What was measured:** the leaf table in `## The denominator`, at n = 3, from
`rc` job `93a60151`'s three phase logs. Nothing new was rendered.

**What was rejected, and why.** *Taking the audio VAE's `vt::Conv2d` arm* — its
ceiling fell from 9.79% to 2.90% of the wall when `4fef1f413` parallelised the
CPU kernel, and its own spec couples it to a golden-changing accumulator change.
A 2.90% ceiling behind a correctness prerequisite is not the next thing to do.
*Taking the video VAE's remaining volume* — 3.60% ceiling for a partial port
across eight kernel families. *Taking the connector* — the same 27% as the tower
for several times the work, and `LTX25-CONNECTOR-REPAIR` has already taken
2.765x out of it on the host. *Taking the tower here* — see `## The decision
this row returns`.

**Why each default has its value.** The denominator is the **default production
render** rather than a microbenchmark because an arm that nothing routes to is
worth less than one on the default path, and only the phase log can tell those
apart. Costs are stated as **ceilings** rather than projections because a leaf
brackets both the arm and whatever host work shares its scope, and
`conditioning.tower` is the case that proves it. The ranking is by **share of
wall**, not by seconds, because the repairs moved the wall and a seconds ranking
taken before them ranks the audio decode above the video VAE, which is now
backwards.

**The thing this row expected to find and did not.** The campaign's owed bullets
read as though the device arms are all ports. Two of the top four are not: the
tower is a queue swap over a shipped model forward that already has a device
arm, and the VAE encoder is one missing queue argument. The expensive one is the
connector, and it is the one already half-repaired on the host.

## Now

`LTX25-DEVICE-ARM-SURVEY` is `DONE` as a survey and returns `NEEDS_DECISION` on
which arm to take. The ranking and the shapes are in `## The inventory`; the two
arms the CPU repairs demoted and the two owed bullets the tree falsified are in
`## What the CPU repairs took off the table` and `### What the sweep found that
is NOT an owed device arm`.
