# LTX25-DEVICE-RESIDENCY — staging the ranked levers into work that lands, and why the instrument is stage zero

Row: `LTX25-DEVICE-RESIDENCY`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)).
Issue: [#1264](https://github.com/mudler/vllm.cpp/issues/1264).
Base: `27d5432f9`.

**This row exists because [`ltx25-decode-speed.md`](ltx25-decode-speed.md) says,
in its own stop conditions, that it has none of what is needed to act:** *"Do not
implement any lever from this row. It has no implementation authority and no
fresh review."* That spike ranked the levers, read every oracle, recorded
provenance per number and filed thirteen issues it owns. It is cited throughout
and **deliberately not restated**. Where a fact belongs to it, this row links
rather than re-derives; where this row disagrees with it, §*Our baseline* says so
with a `file:line` at this base.

## Now

`READY`. The spec is committed, no stage has started, and the row is
**deliberately unclaimed** — each stage below is dispatched to its own fresh
implementer and its own fresh reviewer, so a claim file naming this session would
be false the moment this pull request merges.

The stage order is **W0 → W1 → W2 → W3 → W4 → W5 → W6 → W7**, with the oracle
lane **O1** running beside it and blocking only ratios. **W1 is a gate on the
order itself.** The ranking every later stage inherits was measured on a host
that stopped answering ([#1040](https://github.com/mudler/vllm.cpp/issues/1040)),
and two of the levers it ranked have since landed without re-measurement, so the
profile the ranking describes is not the profile in this tree. W1 either
reproduces the order or replaces it, and **this section is amended with the
re-derived order before W2 starts, unless W1 reports `BLOCKED`, in which case W2
and W3 proceed on their byte-compare and RSS gates and W4/W5 do not start**. A
lever whose rank does not survive W1 is re-placed or dropped here, in writing,
rather than carried.

## Scope

**In scope.** Implementation ownership of the device-residency levers of an
LTX-2.5 render on GB10: the phase instrument, the retrievable-evidence contract,
the load-time memory defects, the DiT staging path, a device arm for the text
conditioning stage, a device arm for the video VAE decode, and the attribution of
the ~59 GiB. Fourteen issues, staged in §*Work breakdown*: the thirteen the
spike owns ([#1007](https://github.com/mudler/vllm.cpp/issues/1007),
[#1009](https://github.com/mudler/vllm.cpp/issues/1009),
[#1010](https://github.com/mudler/vllm.cpp/issues/1010),
[#1011](https://github.com/mudler/vllm.cpp/issues/1011),
[#1012](https://github.com/mudler/vllm.cpp/issues/1012),
[#1014](https://github.com/mudler/vllm.cpp/issues/1014),
[#1015](https://github.com/mudler/vllm.cpp/issues/1015),
[#1016](https://github.com/mudler/vllm.cpp/issues/1016),
[#1021](https://github.com/mudler/vllm.cpp/issues/1021),
[#1024](https://github.com/mudler/vllm.cpp/issues/1024),
[#1040](https://github.com/mudler/vllm.cpp/issues/1040),
[#1202](https://github.com/mudler/vllm.cpp/issues/1202),
[#1210](https://github.com/mudler/vllm.cpp/issues/1210)) plus
[#1269](https://github.com/mudler/vllm.cpp/issues/1269), which this row filed
because the defect had an owner in prose and no issue anywhere (§*Our baseline*
finding 2).

**Out of scope, deliberately.**

* **Any engine code in this pull request.** This change is the spec, the issue
  index rows and the one matrix cell the campaign invalidates. Each stage below
  opens its own `row/LTX25-DEVICE-RESIDENCY-W<n>` branch.
* **[#655](https://github.com/mudler/vllm.cpp/issues/655)** — registering
  Lightricks `ltx_core` as an admitted oracle. It is a policy row that predates
  this campaign, it is not this campaign's to close, and O1 records what this
  campaign needs from it rather than taking it.
* **Any parity ratio against any oracle.** The spike §7 establishes that no
  admitted oracle is installed on the gate host and that a same-tool both-sides
  LTX-2.5 profile is not possible today. Every stage below measures our engine
  against our engine, which is what a defect against our own structure needs, and
  says so.
* **[#1164](https://github.com/mudler/vllm.cpp/issues/1164)**, denoise-loop
  graph capture. W7 is its *decision point*, not its implementation.
* **The diffusion VAE decode arm** (`NADiffusionDecoder`). `Ltx2VideoDecode`
  refuses `kDiffusion` by name today
  (`include/vllm/model_executor/models/ltx2_video_vae.h:236-244`) and that
  refusal is correct; W5 gives the **conv** decoder a device arm and leaves the
  refusal standing.

## Upstream chain

**The oracle question is settled and this row does not re-open it.** The spike
§2 read Lightricks `ltx_core` (the reference implementation), `diffusers` (which
implements LTX-2.5, both decode arms), SGLang (2.0 and 2.3, not 2.5), vLLM-Omni
(2.3, and its 2.5 route disqualified as a reference-degraded configuration) and
vLLM (implements nothing here, so AGENTS.md's *"implements nothing"* branch
applies). §7 records the denominator verdict. **None of that is re-derived here.**

What this row adds is the mapping from *stage* to *what upstream binds it*, and
the honest statement that **half the stages have no upstream to mirror at all**:

| Stage | Upstream that binds it | Where it was read |
|---|---|---|
| W0 instrument | none — no upstream emits our phase names | local seam |
| W1 re-measure | none — a measurement of our engine | local |
| W2 load memory | none for #1015/#1016; upstream offloads transformer weights and never the VAE (`installation.md:90`) | spike §6 lever 5 |
| W3 staging | none — `Alloc`/`Copy`/`Synchronize` is our seam | local |
| W4 text tower device arm | `feature_extractor.py:85-129`, `encoder_configurator.py:187,206-208` for the extractor contract this port already mirrors | already ported; see `include/vllm/model_executor/models/ltx2_text_encoder.h:282-293` |
| W5 VAE decode device arm | `blocks.py:1139` + `single_gpu_model_builder.py:273`; `decoding_av.py:71`; `interface.py:92`; `conv_video_decoder.py:282-284`; `normalization.py:32-40` | spike §6 levers 1 and 2, which read them |
| W5 rider (#1011) | `memory_efficient_decode.py:1-20`, `:91-105`, `:122-204`, `:234-248`, `:541-609`, `:617-627`; default `True` at `blocks.py:1059` | spike §6 lever 4, which read them |
| W6 residency | upstream offloads the transformer, never the VAE (`installation.md:90`) | spike §6 lever 5 |
| O1 oracle record | AGENTS.md oracle table; [`.agents/oracles/diffusers.md`](../oracles/diffusers.md) | — |

**Every upstream anchor in that table is cited from the spike, which read those
files. This row read none of them.** Naming a line I did not open would be the
same defect as quoting a number I did not measure. The stage that takes each row
re-derives its own anchors against the pinned oracle before it writes code, per
[`.agents/porting.md`](../porting.md), because line anchors go stale and these
are already one campaign old.

**Where there is no upstream, the stage says so and gates against our own prior
arm.** That is the same footing `LTX25-DECODE-THREADS`
([`ltx25-decode-threads.md`](ltx25-decode-threads.md) §1) stood on, and it is
legitimate precisely because a single-threaded host stage on a `--device cuda`
render is a defect against *our* structure, not a divergence from upstream's.

## Our baseline

The state of the tree at `27d5432f9`, re-derived rather than inherited. Four of
these are corrections to the framing this campaign was handed, and each names the
line that settles it.

**1. The denoise IS device-routed, so #1024 is a symptom with more than one
candidate producer, and the code has already eliminated one of them.**
[#1024](https://github.com/mudler/vllm.cpp/issues/1024) is titled *"stages 35.5
GiB onto the GPU and then never uses it"*, which reads as an unrouted forward.
It is not: `src/vllm/multimodal/ltx2_video.cpp:847` selects
`Ltx2StreamDitToDevice` on `im.on_device`, and `:3864-3866` selects
`Ltx2DitForwardDevice` over `Ltx2DitForward` on the same flag.
`Ltx2DitForwardDevice` (`src/vllm/model_executor/models/ltx2_device.cpp:1111`)
runs a real device forward — it takes `vt::Queue&`
(declared `include/vllm/model_executor/models/ltx2_device.h:136`) and obtains the
backend inside it via `vt::GetBackend(queue.device.type)` at
`ltx2_device.cpp:1146`, calls `CheckWeightsResident(weights, queue.device)` at
`:1144`, and drives 48 blocks of device ops. The spike's own §5 records that the
sampled window **had written no frame**, which excludes the VAE decode as well.
So the 1.00-core, GPU-zero window is *neither* the routed denoise *nor* the
decode, and the remaining candidates
are the conditioning stage (finding 2), the LoRA fuse
([#1202](https://github.com/mudler/vllm.cpp/issues/1202),
[#1210](https://github.com/mudler/vllm.cpp/issues/1210)) and the load path
itself. **Nothing in the tree picks between them, and that is exactly what
[#1010](https://github.com/mudler/vllm.cpp/issues/1010) is.** This is the single
strongest argument for putting the instrument first: the campaign's headline
measurement does not identify its own subject.

**2. The conditioning stage is host-only by TYPE, and it was in nobody's lever
table.** `src/vllm/multimodal/ltx2_video.cpp:2085`, `:2799` and `:4638` each
build `vt::Queue text_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}`;
`src/vllm/model_executor/models/ltx2_text_encoder.cpp:446` builds a **second**
hard-coded CPU queue inside the extractor, so changing the call sites alone would
not move the projection. The comment at `:443-445` states the reason and it is
the real blocker: the tower's weights are `Gemma4Weights`
(`include/vllm/model_executor/models/ltx2_text_encoder.h:542-548`) of
`OwnedTensor` (`include/vllm/model_executor/models/gemma4.h:114-125`) over
`OwnedBytes` (held at `include/vllm/model_executor/models/qwen3_5_weights.h:47-56`,
defined at `include/vllm/model_executor/models/owned_bytes.h:42`), which has no
device field — `grep -icE 'device|cuda|vt::Queue' owned_bytes.h` returns **0**.
`grep -c 'Device\|kCUDA\|Queue'
src/vllm/model_executor/models/gemma4_weights.cpp` returns **0** against a
positive control of **16** for the identical pattern in `ltx2_device.cpp` and
**4** in `ltx2_text_encoder.cpp` itself, so the zero is a finding and not a
mistyped pattern. `LTX25-TEXT-LINEAR-SEAM`
([`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md) `## Owed`) found this
and wrote *"it needs its own issue"*; none existed, so this row filed
[#1269](https://github.com/mudler/vllm.cpp/issues/1269) and stages it as W4. The
spike's §6 lever table does not carry it at all — it post-dates the table.

**3. [#1015](https://github.com/mudler/vllm.cpp/issues/1015) does NOT gate device
residency; [#1016](https://github.com/mudler/vllm.cpp/issues/1016) does.** #1264
groups them as *"the two memory defects that must land before device residency
has headroom"*. The device path **refuses the widen by name**:
`src/vllm/model_executor/models/ltx2_loader.cpp:759-765` fails
`Ltx2StreamDitToDevice` on `widen_to_f32`, and `:780-800` keeps exactly one host
buffer live per tensor. So #1015's double-hold — `Ltx2WidenDitToF32` at
`:739-754` pushes the f32 copy onto `checkpoint.storage` at `:752` while the bf16
buffer pushed at `:729` is never dropped — is a **host-arm** defect. It is still
in W2 and still early, because the host f32 arm is the *reference* every W4 and
W5 correctness gate compares against, and an arm that cannot load is not a
reference. #1016 is the one that gates device headroom: on GB10's unified memory
the device copy and the dead file cache come out of the same 119 GiB, and the
LTX loaders call `MaybeReleaseSourcePages` nowhere while 14 other files under
`src/vllm` do — `grep -rl` returns 15 files, one of which is the defining
`safetensors_reader.cpp` and one of the remaining 14 a header
(`safetensors_reader.cpp:339` is the definition;
`qwen3_5_dense_weights.cpp:87` and `gemma4_weights.cpp:149-150` are two of the
callers). Note the fix is armed by default —
`LoadWindowedReleaseEnabled()` (`safetensors_reader.cpp:303-312`) reads
`VT_LOAD_WINDOWED_RELEASE` and defaults ON — so adding the call changes behaviour
on the shipped default, which is why W2 gates peak RSS on both arms rather than
assuming an improvement.

**4. Two of the ranked levers have LANDED since the ranking, and neither was
re-measured, so the ranking is stale by construction and not merely
unverified.** [#1009](https://github.com/mudler/vllm.cpp/issues/1009) (lever 3)
closed via [#1041](https://github.com/mudler/vllm.cpp/issues/1041): three
`vt::cpu::ParallelForRows` dispatch sites are live in
`src/vllm/model_executor/models/ltx2_video_vae.cpp:170`, `:218` and `:276`.
[#1208](https://github.com/mudler/vllm.cpp/issues/1208) closed via
`LTX25-TEXT-LINEAR-SEAM`, which measured one conditioning pass from **671.777 s
of one core to 78.421 s (8.57x)** on x86 and records the projection as
**39-100%** of the ~1731 s resolution-constant phase
([#1087](https://github.com/mudler/vllm.cpp/issues/1087)) — a bound, because the
GB10-to-x86 per-core ratio was never measured. `docs/BENCHMARKS.md:494` carries
that bound. **Neither landing has been re-measured end to end**, and
`LTX25-DECODE-DTYPE`'s speed magnitude was never measured either
([`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) `## Owed`). The spike's own
re-ordering note (*"the remaining order is 6, 3, then 1"*) predates #1208
entirely.

**5. The video VAE decode arm does not exist, and the signature says so.**
`Ltx2VideoDecode` and `Ltx2ConvVideoDecode`
(`include/vllm/model_executor/models/ltx2_video_vae.h:230-244`) take
`const std::vector<float>& latent` and no queue; `grep -c 'Queue'
src/vllm/model_executor/models/ltx2_video_vae.cpp` returns **0** against a
positive control of **6** `vt::` hits in the same file and 2 `Queue` hits each in
the sibling `ltx2_dit.cpp` and `ltx2_device.cpp`. The spike's reading stands: the
arm does not exist, so W5 is a new op family and not wiring.

**6. The positive control [#1024](https://github.com/mudler/vllm.cpp/issues/1024)
owes is, as far as this tree records, still owed.** The spike states plainly that
it has no control proving `utilization.gpu` reads high for a real kernel on GB10,
and that `--query-gpu=memory.used` returns `[N/A]` on that box (§4.3).
`grep -rn '11\.25\|power\.draw\|board power' .agents/ docs/` returns only
unrelated matches, so no such control is written down anywhere. **W1 takes it and
records it**, using the two instruments that do work on GB10 —
`--query-gpu=power.draw,temperature.gpu` and
`--query-compute-apps=used_memory` — beside the host-side thread states and
`utime`/`stime` rates, so the two sides corroborate rather than one carrying the
claim alone.

## Port map

Anchors at `27d5432f9`. Each stage re-derives them on its own branch before it
edits, because several already drifted between the spike's base and this one
(the spike's `ltx2_video.cpp:2946`/`:2948` is now `:3864-3866`; its
`ltx2_loader.cpp:694-710` is now `:739-754`; the seam row's `ltx2_video.cpp:4479`
is now `:4638`).

**That re-derivation is the ONLY check these anchors get, and the gap is
specific.** No checker in this tree reads a bare `path.cpp:123` written in prose.
`check_links` (`scripts/check-agent-record.py:862-886`) parses markdown link
targets only, and validates a `#L<n>` fragment against the target's line count —
so a dangling link and an out-of-range `#L999999` are both caught, and a bare
`ltx2_video.cpp:99999` in a sentence is not read at all. `check_spec`
(`scripts/check-agent-record.py:1169`) takes a `ClaimRow`, so its
structured-section requirement reaches only a spec linked from a matrix row, and
this row has none (§*Decisions taken here*, "No matrix row, and no claim file").
Every `file:line` below and in §*Our baseline* is therefore checked by a person
reading it and by nothing else — four of them were off by one or two lines, in
both directions, when this spec was first reviewed. Each stage re-derives with
`grep -n` against its own base and pastes the output into its report, rather
than counting by eye.

| Stage | Our anchor at this base | What changes | Production entry point it must be reached from |
|---|---|---|---|
| W0 | `src/vllm/multimodal/ltx2_video.cpp` render driver; `include/vllm.h` | phase-boundary timestamps + peak host/device bytes emitted per phase | `examples/ltx2_gen` through the `vllm.h` video ABI, on the shipped default |
| W1 | none — measurement only | none | the same binary W0 shipped |
| W2a #1016 | `src/vllm/model_executor/models/ltx2_loader.cpp:722-735`, `:780-800` | `MaybeReleaseSourcePages` after each materialization, both arms | `Ltx2LoadDitFromSafetensors` / `Ltx2StreamDitToDevice` via `ltx2_video.cpp:847` |
| W2b #1015 | `src/vllm/model_executor/models/ltx2_loader.cpp:739-754` | drop the bf16 source buffer as each view is repointed | the host f32 arm at `ltx2_video.cpp:847` |
| W2c #1210 | `src/vllm/model_executor/models/ltx2_loader.cpp:808-912` (`Ltx2RebindDitLoras`) | delete the load-time fuse that phase 0 provably undoes | `Ltx2RebindDitLoras` on a LoRA render |
| W3 #1021 | `src/vllm/model_executor/models/ltx2_loader.cpp:780-800`; `src/vt/cuda/cuda_backend.cu:77-95` | batch/overlap the per-tensor `Alloc`+`Copy`+`Synchronize` | `Ltx2StreamDitToDevice` |
| W4 #1269 | `src/vllm/multimodal/ltx2_video.cpp:2085`, `:2799`, `:4638`; `src/vllm/model_executor/models/ltx2_text_encoder.cpp:446`; `include/vllm/model_executor/models/gemma4.h:114-125` | a device weight arm for the tower + queue threading | `Ltx2EncodePromptToConditioning` from the render driver |
| W5 #1007 | `src/vllm/model_executor/models/ltx2_video_vae.cpp:161-290`; `include/vllm/model_executor/models/ltx2_video_vae.h:230-244`; `include/vllm/model_executor/models/ltx2_tiling.h:308` | a `vt::` conv3d op with CUDA and CPU arms; a device `Ltx2VaeWeights` | `Ltx2VideoDecodeStreaming` from the render driver |
| W5 rider #1011 | same, plus the NDHWC half owed to [`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) `## Owed` | workspace reuse, in-place norm/SiLU, free-before-conv, temporal chunking, memory format | same |
| W6 #1014 | W0's peak-memory emission; `--query-compute-apps=used_memory` | attribution, then a release if one is found | the render |
| W7 | `src/vt/cuda/cuda_backend.cu:206-222` (`BeginCapture`, `EndCapture`, `Replay` — the capture primitive) | nothing unless the measurement says capture | — |
| O1 #1012 | [`.agents/oracles/diffusers.md`](../oracles/diffusers.md) | record that `diffusers` implements LTX-2.5 and measure its gateability | — |

**Nothing lands dead.** Each stage's smallest failing test enters through the
entry point in the last column, at that stage's own merge commit, and each
stage's fresh reviewer deletes that production call site in a scratch copy and
reruns the focused gate — a gate that stays green without the call site measured
a class, not a capability
([`.agents/reachability.md`](../reachability.md)). W1, W6 and W7 land no code
and therefore have nothing to reach; they land evidence and a record.

## Tests to port

**There is no upstream test to port for W0, W1, W2, W3, W6 or W7.** Saying so is
the honest answer: none of them mirrors an upstream behaviour. Their tests are
local red-first cases, and each names the guarantee it pins:

| Stage | The smallest failing test, and what makes it red first |
|---|---|
| W0 | a render at a small fixed geometry through the video ABI emits a phase table whose entries carry a monotone timestamp and a byte count; red before, because the path emits one line per render ([#1010](https://github.com/mudler/vllm.cpp/issues/1010)) |
| W2a | peak `Rss_File` across a load, asserted below the pre-change value by a stated margin; red before, because no release call exists |
| W2b | peak RSS across a **host** load, asserted below the pre-change value; red before, because both buffers are retained (`ltx2_loader.cpp:729` and `:752`) |
| W2c | a LoRA load followed by a phase-0 rebind performs the fuse **once**; red before, because it fuses, un-fuses and re-fuses |
| W3 | staged bytes byte-compare equal to the serial path AND the round-trip count falls; the byte-compare is the correctness half and must be written first |
| W4 | conditioning tensors from the device arm against the host arm within a stated tolerance, entering through `Ltx2EncodePromptToConditioning`; red before, because there is no device arm to select |
| W5 | decoded **pixels** from the device arm against the current host f32 arm at a fixed seed and geometry, entering through `Ltx2VideoDecodeStreaming`; red before, because `kind`-to-device does not resolve |

**W5 and W2b do have an upstream reference and must use it.** The conv decoder's
goldens are generated by `scripts/gen-ltx2-vae-goldens.py`, and the spike §8
records the trap: that generator casts every upstream parameter to f32 at
`:223`, so the oracle itself runs f32 and a width comparison against it is
**vacuous by construction** (`ltx2_video_vae.cpp:41-44`). W5 therefore carries a
separable-reduction width case of its own, in the shape `LTX25-DECODE-DTYPE`
shipped — a `[+1e8, 0.1 x 25, -1e8]` tap set where an f32 accumulator lands on
exactly 0 in any summation order and an f64 one does not — and **must not let its
expectation coincide with the zero value of an absent computation**, because a
reachability mutation that replaced the decode with a zero-filled buffer *passed*
for that row until the expectation was biased away from zero.

Three rules bind every mutation in every stage, each earned by a failure this
campaign already had:

1. Report `git diff --stat` with the mutation, because **a mutation that never
   applied reads as a passing test**.
2. Report the compile status, because **a mutation that fails to build reads as a
   passing test** and a stale binary prints a plausible verdict.
3. Report the exit code and a non-zero case count, because `assertions: 0` is a
   skip wearing a pass and `doctest -tc` splits its filter on commas.

## Gates

**Correctness before speed, at every stage, without exception.** A decode or a
conditioning pass that is faster and different is not a win.

**The completion gate is pixels, never an exit code.**
[#1149](https://github.com/mudler/vllm.cpp/issues/1149) is the precedent: a
completed render exited 127 from a missing `ffmpeg` and read as a failure. The
shipped verification recipe is `LTX25-RESOLUTION-ENVELOPE`'s, and every stage
that completes a render reuses it verbatim: distinct per-frame md5s for every
frame, zero near-uniform and zero near-black frames, adjacent-frame mean absolute
difference against a uniform-noise reference on the same shape, zero zero-motion
pairs, and the audio track's rate, duration, dBFS and above-threshold window
count.

Per stage, with the refutation shape stated **before** the run, because AGENTS.md
forbids declaring a ceiling and a refuted lever owes the next traceable
hypothesis:

| Stage | PASS | REFUTED, and the next hypothesis |
|---|---|---|
| **W0** | one completed render emits a phase table whose durations **sum to wall within a stated tolerance**, plus per-phase peak host bytes and device bytes; the file is retrievable from this repo's evidence path | the phases do not sum. Then the instrument is incomplete and **W1 cannot start**: the missing time is a phase nobody named, and W0 iterates until it is named. A partial phase table is worse than none, because it invites the same interpolation the sampler CSVs already invited |
| **W1** | a per-phase table at **two geometries** on an idle leased box, with board power and `--query-compute-apps=used_memory` sampled beside it, that either reproduces the spike's order or replaces it; `## Now` is amended with the result; #1040's artifacts land in the repo | the phase table contradicts the ranking. That is a **success of this stage**, not a failure: the order in `## Now` is rewritten, and any stage whose lever is now small is dropped from the campaign here in writing |
| **W2a** | peak `Rss_File` across a load falls, measured on the same binary A/B, both arms; the DiT byte-compares identical after the release | RSS does not fall. Then the pages were not resident to begin with and the +10.83 GiB was a different allocation; next hypothesis is the dequantize scratch inside `MaterializeDitTensor` |
| **W2b** | peak RSS across a host f32 load falls by approximately the bf16 half of the contract; the f32 views still byte-compare identical | RSS does not fall. Then the bf16 buffers were already being freed by some path this reading missed, and the ~37.9 GiB figure is wrong; next hypothesis is `checkpoint.storage`'s own lifetime |
| **W2c** | the fuse count on a LoRA load-then-phase-0 sequence drops from two to one; fused weights byte-compare identical to the two-pass result | the weights differ. Then the load-time pass is **not** provably wasted and #1210's premise is wrong; the row stops and reports `NEEDS_DECISION` |
| **W3** | staged bytes byte-compare identical to the serial path, **then** staging wall and MiB/s improve on an idle box | wall does not move. Then the cost is not the round trips; next hypothesis is the host-side dequantize inside `MaterializeDitTensor`, which W3 must then time separately from the transfer |
| **W4** | conditioning tensors match the host arm within a stated tolerance, **then** the conditioning-pass wall falls at the shipped geometry, measured against the same binary's host arm | the conditioning pass is a small fraction of W1's phase table on GB10. Then #1269 is refuted **as a lever** while remaining a correct capability, W4 drops to the tail, and the next hypothesis is whichever phase W1's table actually shows |
| **W5** | pixels against the current host f32 arm at fixed seed and geometry, `max\|diff\|` recorded and inside a stated tolerance argued at the site; **then** decode wall; **then** a completed render verified on pixels | pixels differ beyond tolerance → the arm is wrong and does not land. Wall does not move with pixels correct → the decode was not the term W1's table said it was, and the next hypothesis is the memory format (#1011's NDHWC half), measured before any further kernel work |
| **W6** | the ~59 GiB either reproduces under W0's per-phase byte counters and is attributed to a named allocation, or it **does not reproduce**, which closes #1014 as not-reproducible-with-this-instrument | neither: it reproduces and W0's counters do not see it. Then the allocation is invisible to the counters W0 emits, which is the spike §4.3 device-class hypothesis, and the next instrument is `--query-compute-apps=used_memory` sampled per phase |
| **W7** | GPU-busy against wall on a device-resident loop. Host-dispatch-bound → capture #1164. GPU-bound → **#1164 closes as a refutation**, the way [#1161](https://github.com/mudler/vllm.cpp/issues/1161) closed prefill capture | the loop is neither, because a host phase still dominates. Then W7 does not run and the campaign returns to W1's table |
| **O1** | `diffusers` recorded as covering LTX-2.5 in its oracle file with a gateability measurement on the gate host | it does not build or run 2.5 there → `gateable = no` with the issue that owes the measurement named, per AGENTS.md |

**No stage may quote a ratio against any oracle** until O1 lands a gateability
measurement. Every number above is our engine against our engine on the same
binary, and each stage says so in its own `## Outcome`.

**The full gate before every push**, per stage, and the operator reruns it
itself: an implementer or reviewer report is an input, never a gate result. Known
reds excluded by name and by issue: `test_cpu_x86_llamacpp_floor`
([#618](https://github.com/mudler/vllm.cpp/issues/618)) and the `windows-msvc-*`
lanes ([#584](https://github.com/mudler/vllm.cpp/issues/584)).

## Dependencies

```
W0  instrument ─────────────► W1 re-derive the ranking ──┬──► W2 ──► W3 ──► W4 ──► W5 ──► W6 ──► W7 (#1164)
                                                          │
O1  oracle record (#1012) ────────────────────────────────┘  (blocks RATIOS only, never a defect fix)
```

* **W1 depends on W0** and on nothing else. Without a phase log the re-derivation
  would be another sampler interpolation, which is the exact evidence class
  [#1040](https://github.com/mudler/vllm.cpp/issues/1040) files against.
* **W2, W3, W4, W5 depend on W1** for their *order*, not for their *correctness*.
  Each is a defect against our own structure and each could be implemented today;
  what W1 decides is which is worth doing first and whether any is worth dropping.
  A stage may start early only if W1's table has already named its phase.
* **W3 depends on W2a.** Batching the staging raises transient device residency,
  and doing that before the dead file cache is released is how a 119 GiB box
  OOM-reboots.
* **W4 and W5 depend on W2b**, because the host f32 arm is the reference their
  correctness gates compare against, and #1015 makes that arm's peak
  approximately 105.9 GiB on a 119 GiB box.
* **W5 depends on W3** only for cost: without it, every W5 iteration pays 7.5
  minutes of staging before it reaches the decode. That is a schedule dependency
  and it is stated as one.
* **W6 depends on W0** for its instrument and may dissolve without any
  implementation at all.
* **W7 depends on W5** — a graph collapses host launch dispatch, and this render
  has almost none to collapse until the loop is device-resident. Its decision
  point is a measurement, not an implementation.
* **O1 blocks no defect fix.** The spike §7 splits this by axis and the split is
  correct: a ratio needs a denominator, an 89%-of-wall single-threaded host phase
  does not.
* **[#655](https://github.com/mudler/vllm.cpp/issues/655) blocks every ratio and
  is not this row's to close.** Recorded, not taken.

## Work breakdown

Each stage lands on its own branch, is reachable from a production entry point at
its own merge commit, is revertible without touching the next, and **owes an open
issue before it starts**. Every issue this campaign stages is named below exactly
once.

### The order, argued

**The instrument is not first because instruments are virtuous. It is first
because the ranking has three independent defects and all three are measurement
defects.** (a) The evidence is unretrievable — the sampler CSVs, rung 2's
`run.log` and `render8-console.log` exist only on a host that stopped answering
(#1040), and the spike marks even its *passing* numbers with that reason. (b) The
subject is unnamed — #1087 measures a 1731 s phase and says in its own text *"Do
not guess it from the duration"*, and #1024's GPU-zero window is now known not to
be the denoise (§*Our baseline* finding 1) and known not to be the decode. (c)
The ranking is **stale**, not merely unverified: #1009 and #1208 both landed
after it and neither was re-measured (finding 4). A campaign staged on that
ranking would be spending months against a profile that no longer exists.

**After W1, the order is not by magnitude. It is by what each stage does to the
campaign's own measurement loop.** W2 and W3 are small, cheap and revertible, and
they remove the two things that make every later measurement expensive or
dangerous: the memory headroom that decides whether a render completes on a
119 GiB box at all, and the 7.5 minutes of staging paid at the front of every
render *and every gate run*. Buying those first makes W4 and W5 cheaper to
iterate, and W5 is the stage that will need the most iterations. That is the
whole argument, and it is a scheduling argument, stated as one.

**W5 is last of the implementation stages despite being ranked first on
magnitude, and that is a deliberate disagreement with the spike's table.** Lever
1 is ranked first on FLOPs — 7.25 TFLOP of dense 3x3x3 convolution — and that
arithmetic is sound. But the evidence #1024 produced cannot support it: the
sampled GPU-zero window had written no frame, so it was not in the decode. The
spike itself reaches the same conclusion from the other side, recommending *"6,
2, 3, then 1"* and then *"6, 3, then 1"*. W5 stays in the campaign, ranked on its
arithmetic, and starts after the stage that tells us what the render is actually
doing. **If W1's phase table puts the decode back on top, W5 moves up and this
section is amended** — that is what W1 is for.

### The stages

| Stage | Issues | Lands | Size |
|---|---|---|---|
| **W0** the instrument | [#1010](https://github.com/mudler/vllm.cpp/issues/1010), [#1040](https://github.com/mudler/vllm.cpp/issues/1040) (artifact contract half) | phase-boundary timestamps and per-phase peak host/device bytes, emitted from the render path on the shipped default; the evidence-path contract in §*Risks/decisions* H2 | small |
| **W1** re-derive the ranking | [#1040](https://github.com/mudler/vllm.cpp/issues/1040) (closes), [#1024](https://github.com/mudler/vllm.cpp/issues/1024) (re-measured, with the control it owes) | no code. A phase table at two geometries on an idle leased box, artifacts in the repo, `## Now` amended with the re-derived order | small, one lease |
| **W2** the load path | [#1016](https://github.com/mudler/vllm.cpp/issues/1016), [#1015](https://github.com/mudler/vllm.cpp/issues/1015), [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | three independent commits on one branch, each with its own red-first RSS or fuse-count case | small each |
| **W3** the staging path | [#1021](https://github.com/mudler/vllm.cpp/issues/1021) | batched/overlapped device staging behind a byte-compare | medium |
| **W4** the conditioning stage | [#1269](https://github.com/mudler/vllm.cpp/issues/1269) | a device weight arm for the Gemma-4 tower and the caption projection, and the queue threading that reaches it | medium-large |
| **W5** the video VAE decode | [#1007](https://github.com/mudler/vllm.cpp/issues/1007), [#1011](https://github.com/mudler/vllm.cpp/issues/1011) as a rider | a `vt::` conv3d op with CUDA and CPU arms, a device `Ltx2VaeWeights`, and the memory format the arm consumes | large |
| **W6** the residency question | [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | attribution or a documented non-reproduction | medium, conditional |
| **W7** the graph decision | — ([#1164](https://github.com/mudler/vllm.cpp/issues/1164) is decided, not implemented, here) | a measurement and a verdict | small |
| **O1** the oracle record | [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | `diffusers` recorded as covering LTX-2.5, with a gateability measurement | small-medium |
| — | [#1009](https://github.com/mudler/vllm.cpp/issues/1009) | **CLOSED.** Landed as [#1041](https://github.com/mudler/vllm.cpp/issues/1041); `ltx2_video_vae.cpp:170`, `:218`, `:276`. Listed so the count closes, not staged | — |
| — | [#1202](https://github.com/mudler/vllm.cpp/issues/1202) | **DEFERRED behind W1**, argued below | — |

### Two levers this campaign does not carry as stages, and why

**[#1011](https://github.com/mudler/vllm.cpp/issues/1011) does not get a stage. It
becomes a rider on W5.** The spike §4 already re-ranked it from *"the 60 GiB"* to
byte traffic and memory format. Once W5 puts the decode on the device, upstream's
memory-efficient **host** decode is an optimization of a path production no longer
takes, and its one genuinely portable part — the memory format — is the thing that
decides which kernel family W5's device arm can reach. Carrying it as a separate
stage would schedule a second independent rewrite of `CausalConv3d` against a host
arm we are trying to stop using. **If W5 is refuted** — no device arm is
achievable, for a reason W5 names — #1011 comes back as its own stage against the
host fallback, and this paragraph is the record of that condition.

**[#1202](https://github.com/mudler/vllm.cpp/issues/1202) is deferred behind W1,
not dropped.** `Ltx2FuseLoraIntoTensor` is a scalar single-threaded loop at
~0.53 GFLOP/s and it should route through the `vt::` GEMM seam like everything
else. But the spike's own `## Owed` measures it at **2.3% of one pass in
10.4 min** on the full model, and it is paid only on a LoRA arm. Against W4's
39-100% bound that is small enough that ranking it before W1's table exists would
be guessing. It is deferred with a number attached rather than carried as debt
with no number, and W1's table decides it.

## Risks/decisions

### The hardware hazards, as gates

Each is a condition a stage's run script asserts, not a paragraph an operator is
expected to remember. A stage that cannot satisfy its applicable gate **does not
run**.

**H1 — a GB10 unified-memory OOM reboots the whole box, and the spike records
that it needs a physical power cycle.** Every stage that starts a render arms a
memory watchdog with a **named floor** before the engine starts, in the shape
`LTX25-RESOLUTION-ENVELOPE` used (a 2 s sampler with a stated `MemAvailable`
floor, which ended a run at 13.77 GiB against an 18 GiB floor rather than letting
the box fall over). **The watchdog's own arming is positive-controlled** before
the real run, because an absent hook looks exactly like an armed one and a broken
instrument fails toward a verdict about the code. Gate: no render starts without
a floor in its command line and a control that proves the watchdog fires.

**H2 — `/tmp` does not survive that reboot, and evidence has already been lost to
this.** `dgx.casa` runs an immutable Kairos OS: `/mnt` and the root overlay are
ephemeral and do not survive a reboot, while `/usr/local` is `COS_PERSISTENT` and
does ([`.agents/environment.md:390-398`](../environment.md),
[`nas-mount-path.md:29-40`](nas-mount-path.md)). Two rows already lost an `ncu`
comparison to a shared-box OOM-reboot
([`mxfp4-flash-audit-2026-08-06.md:33`](mxfp4-flash-audit-2026-08-06.md),
[`qwen3-mxfp4-flash-occupancy-2026-08-06.md:16`](qwen3-mxfp4-flash-occupancy-2026-08-06.md)).
Gate, three parts: (a) the run script **asserts its artifact path is not under
`/tmp`, `/var/tmp` or `/mnt`** and fails loudly if it is; (b) artifacts are copied
off the box **before the lease ends**, not after the run; (c) the copy is verified
by reading the file back, because `REMOTE_UNVERIFIED` is what
[#1040](https://github.com/mudler/vllm.cpp/issues/1040) is made of and unknown is
not success.

**H3 — the fleet is leased through `rc`, never through `ssh`.** `rc run` or
`rc hold` claims a device; raw `ssh` makes the fleet report the box free while
someone is on it, which is the contention the lease exists to remove
([`.agents/environment.md:10-31`](../environment.md)). Gate: every measurement
stage records its lease id, and a stage whose evidence carries no lease id is not
accepted. Run `rc devices` before calling any box unhealthy — `dgx.casa`
being unreachable to one session is not the same fact as the device being down.

**H4 — a lease carries bytes, not executables, and the submitting client's death
kills the job.** `/tmp`, `/var/tmp` and `$HOME` (`/home/rc`) inside a lease are
writable and take a real exec bit, so a runtime staged on `/workspace` runs after
a copy ([`.agents/environment.md:148-183`](../environment.md)); but `rc run`
submissions die with the detaching client, and this harness's own 10-minute
command cap will kill a queued job. Gate: a render stage stages its script where
the worker can read it and submits with a `--max-runtime` that covers the render,
and **never** holds a foreground client for the duration of a 2.5-hour run.

**H5 — a long A/B must not run a binary from a worktree.** Merging the branch
reaps the worktree mid-run and the measurement dies with it. Gate: every W1, W3,
W4 and W5 wall-clock run executes a binary copied outside any worktree, and
records the sha256 of what it ran.

### Decisions taken here

**Row state is `READY`, not `SPIKE`.** A `SPIKE` produces a finding and ships no
code, which is precisely what
[`ltx25-decode-speed.md`](ltx25-decode-speed.md) already is and precisely what
[#1264](https://github.com/mudler/vllm.cpp/issues/1264) says is *not* missing. A
second spike over the same ground would be the duplication that issue's own
*"this issue does NOT duplicate it"* section forbids. This row has the nine
sections, an ordered staged plan, an issue per stage, and implementation
authority — which is the definition of `READY`. It is not `ACTIVE`, because no
stage has started and no gate has run; claiming `ACTIVE` on a spec-only commit
would be a lifecycle state with nothing behind it.

**No matrix row, and no claim file.** The four sibling `LTX25-*` rows
(`LTX25-DECODE-SPEED`, `LTX25-DECODE-THREADS`, `LTX25-TEXT-LINEAR-SEAM`,
`LTX25-RESOLUTION-ENVELOPE`) appear in **no** matrix, and the reason is
structural rather than an oversight: `scripts/check-agent-record.py` builds its
row-ID pattern from `MODEL`, `QUANT`, `KERNEL`, `BACKEND`, `ENG`, `KV`, `PAR`,
`SAMPLE`, `TOOLS`, `SPEC`, `SERVE`, `LORA`, `ATTN` and `LOAD` (`:327-338`,
`:599-603`), so an `LTX25-` id in a matrix is a row **no checker reads**, sitting
in a shared file that every future edit must then write. AGENTS.md names that
shape a lock. The record surfaces are therefore this spec, the append-only issue
index, and the one matrix cell this campaign genuinely invalidates —
`ENG-CUDAGRAPH-DIFFUSION` in [`engine-matrix.md`](../engine-matrix.md), whose
blocker list names #1024, #1007, #1087 and #1010 and now names the row that owns
their unblock order. No claim file is written, because the row is unclaimed by
design: each stage goes to its own fresh implementer, and a claim naming this
session would be stale at merge.

**No `docs/STATUS.md` or `docs/BENCHMARKS.md` edit.** This change ships a spec
and issue rows. No capability changes lifecycle state — LTX-2.5's line at
`docs/STATUS.md:146` describes L1-L9c as landed and this row lands nothing — and
no measurement is taken, accepted, or made pending. The spike §8 argued the same
for the same reason and the projection table in AGENTS.md agrees: editing
`.agents/` on its own owes neither. **W1 owes a `docs/BENCHMARKS.md` edit** when
its phase table lands, and W5 owes one when its wall is accepted.

### Risks to the plan itself

* **W1 needs a lease on a box that has been unreachable.** If `dgx.casa` does not
  answer, W1 reports `BLOCKED` with `REMOTE_UNVERIFIED` and the campaign runs W2
  **and W3** and then stops — the two load-path defects and the staging path are
  provable on any box with the checkpoint, because their gates are RSS and
  byte-compares, not wall clock. W4 and W5 do **not** start on an unreproduced
  ranking. That is the stop condition, and it is deliberate: proceeding would
  repeat the exact failure #1040 records.
* **W5 may be large enough to need its own campaign.** If its port map comes back
  with more than one new `vt::` op family, the stage splits and this section is
  amended rather than the stage being pushed through as one review.
* **The tolerance for W4 and W5 is not yet argued.** Both compare against a host
  arm rather than an upstream oracle, and a tolerance chosen after seeing the
  numbers is not a gate. Each stage argues its tolerance **at the site, before
  the run**, in the shape [`ltx25-decode-dtype.md`](ltx25-decode-dtype.md) §3
  used.
* **This campaign's own numbers can go the way the spike's did.** H2 is the
  answer, and it is written as an assertion in a run script rather than as an
  instruction to a person, because the instruction form has already failed once.

## Owed

| Issue | Stage | State |
|---|---|---|
| [#1264](https://github.com/mudler/vllm.cpp/issues/1264) | this row: the staged campaign spec | closed by this row landing |
| [#1010](https://github.com/mudler/vllm.cpp/issues/1010) | W0 | owed |
| [#1040](https://github.com/mudler/vllm.cpp/issues/1040) | W0 (contract) + W1 (closes) | owed |
| [#1024](https://github.com/mudler/vllm.cpp/issues/1024) | W1 | owed; its `utilization.gpu` positive control is still unrecorded in this tree |
| [#1016](https://github.com/mudler/vllm.cpp/issues/1016) | W2a | owed |
| [#1015](https://github.com/mudler/vllm.cpp/issues/1015) | W2b | owed; host arm only, see `## Our baseline` finding 3 |
| [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | W2c | owed |
| [#1021](https://github.com/mudler/vllm.cpp/issues/1021) | W3 | owed |
| [#1269](https://github.com/mudler/vllm.cpp/issues/1269) | W4 | owed; filed by this row |
| [#1007](https://github.com/mudler/vllm.cpp/issues/1007) | W5 | owed |
| [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | W5 rider | owed |
| [#1014](https://github.com/mudler/vllm.cpp/issues/1014) | W6 | owed, conditional |
| [#1012](https://github.com/mudler/vllm.cpp/issues/1012) | O1 | owed |
| [#1202](https://github.com/mudler/vllm.cpp/issues/1202) | deferred behind W1 | owed, with a number: 2.3% of one pass |
| [#1164](https://github.com/mudler/vllm.cpp/issues/1164) | W7 decision point | not owed here — owned by `ENG-CUDAGRAPH-DIFFUSION`; this row owns its unblock order |

**[#1009](https://github.com/mudler/vllm.cpp/issues/1009) is not owed. It
landed**, as [#1041](https://github.com/mudler/vllm.cpp/issues/1041), and listing
a closed issue as owed by a row that did not close it misattributes the work and
leaves a fixed defect reading as open debt — the same argument the spike makes
for #1008 and #1208.

Also owed, and not attached to a stage:

* **[#655](https://github.com/mudler/vllm.cpp/issues/655)**, the `ltx_core`
  oracle registration. It blocks every ratio and it is not this row's to close.
* **The GB10-to-x86 per-core ratio** that bounds #1269 to 39-100% rather than a
  number. Owed to [`ltx25-text-linear-seam.md`](ltx25-text-linear-seam.md)
  `## Owed`, and W1's lease could close it in the same run.
* **The third single-core stretch** of the #1208 trace (2589 s+, RSS flat at
  31 GiB), unattributed. W1's phase table should name it; if it does not, that is
  a W0 gap and W0 iterates.
