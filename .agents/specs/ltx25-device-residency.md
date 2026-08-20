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

`ACTIVE`. **W0 has landed and W1 is next**, which is a lease on a box with the
checkpoints — `dgx.casa`, per `H3`. Every other stage is unstarted, and each is
dispatched to its own fresh implementer and its own fresh reviewer.

**What W0 answered, and what it did not.** A completed render now writes
`<output_dir>/phase-log.json` on the shipped default: a flat, non-overlapping
timeline of named phases, each carrying a monotone timestamp, a duration, a peak
host byte count and a peak device byte count, with `unaccounted_seconds` emitted
beside the sum rather than smeared over the phases. On a completed 64x64/9-frame
render through the `vllm.h` video ABI the named leaves account for **99.94% of
wall** (4.459892 s of 4.462572 s, residue 2.68 ms over 33 entries), so **the
phases sum and W1 may start**. That is the artifact's own run and the figure
`## Outcome — W0` records. **The RATIO is what the gate reads. No wall here is a
benchmark** — the same binary at the same geometry has measured 0.147 s, 0.158 s,
1.676 s, 4.463 s, 6.138 s and 12.030 s on a contended box, and the first two of
those are one minute apart. The artifact is
[`benchmarks/demo/ltx25_phase_log_fixture_cpu.json`](../../benchmarks/demo/ltx25_phase_log_fixture_cpu.json)
and its provenance is in `## Outcome — W0` below.

**What W0 did NOT measure is the device column, and a lease will not close it.**
Every `peak_device_bytes` in the artifact reads the `-1` "no probe answered"
sentinel. Two reasons are stacked. The render W0 could take was the CPU arm,
where `-1` is the correct answer. The one that matters is that **`CudaBackend`
does not override `vt::Backend::DeviceMemoryInfo` at all** —
[#1126](https://github.com/mudler/vllm.cpp/issues/1126), and `grep -rn
'DeviceMemoryInfo' src/vt include/vt` returns the base declaration
(`include/vt/backend.h:94`) and exactly one override, ROCm's
(`src/vt/rocm/rocm_backend.hip:358`). A CUDA render on `dgx:gpu0` would print
`-1` in that column too.

W0 did not wire it, and the reason is written down where the seam is:
`include/vllm/platforms/interface.h:68-72` records that CUDA's absence from that
seam is load-bearing, because `Gemma4MoE`'s device-expert LRU is its only
consumer. **Which arm of that LRU the override would wake is narrower than this
row first claimed, and the narrower statement is the one to carry.** The bf16
arm, `EnsureGemma4Fp8ExpertOnDevice` (`src/vllm/model_executor/models/gemma4_moe.cpp:548`),
is dead on CUDA for a SECOND and independent reason: it refuses at `:571` on
`vt::HasMatmulBTAlphaBeta`, whose only implementation in this tree is ROCm's
([#1205](https://github.com/mudler/vllm.cpp/issues/1205)), so wiring
`DeviceMemoryInfo` does not wake it at all. What the override WOULD wake is the
**FP8-native arm**, `EnsureGemma4Fp8NativeOnDevice` (`:611-628`), which reaches
`DevExpertLru::MakeRoom` behind no device gate other than the probe itself and
whose budget defaults to a 2048 MiB fill-only device cache when
`VT_GEMMA4_EXPERT_VRAM_MB` is unset (`:416-432`). So the override is still a
behaviour change with its own measurement, on one named arm rather than on
"every CUDA model". That is #1126's change, not an instrument's. **Until it lands, W1 samples
`--query-compute-apps=used_memory` per phase beside the table** — which is the
fallback this spec's W6 row already names, and the only device-memory instrument
GB10 answers. `## Owed` states what the column cannot report and why.

The stage order is **W0 (landed) → W1 → W2 → W3 → W4 → W5 → W6 → W7**, with the
oracle lane **O1** running beside it and blocking only ratios. **W1 is a gate on the
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
| [#1010](https://github.com/mudler/vllm.cpp/issues/1010) | W0 | **closed, for a run that FINISHES.** The render writes a phase table on the shipped default and the ABI names it through `vllm_video_last_phase_log` (v23). Read the row below it before quoting that as "the render is instrumented" |
| [#1413](https://github.com/mudler/vllm.cpp/issues/1413) | W0-live | **closed by `## W0-live` below.** W0's table is written by the success path only, and nothing at all is emitted while a render runs, so an aborted render and a working one both report nothing and the ~162 s DiT forward of [#1375](https://github.com/mudler/vllm.cpp/issues/1375) has no in-process counter |
| **a table on a run that does NOT finish** | [#1413](https://github.com/mudler/vllm.cpp/issues/1413), stage W0-live | **the LIVE half is closed by this change; a PARTIAL TABLE on abort is still owed and has no issue.** `WritePhaseLog` has exactly two call sites (`ltx2_video.cpp:2259` audio-only, `:4677` video) and both sit immediately before a successful `return`, three lines after `im.trace.completed = true`. Nineteen `VT_CHECK` sites throw out of `Ltx2VideoEngine::Generate` above them and that body contains no `try` and no `catch`; `vllm_video_generate`'s own two catches set an error and return, and `engine->last_phase_log` is assigned on the success path only. **So a render that is killed, aborted by a lease governor, refused by a guard, or still running leaves no `phase-log.json` at all** — not a truncated one, not an empty one, nothing. The mutation is the demonstration: deleting the video call site removes the file entirely and the W0 gate goes red on `REQUIRE(probe.good())`. That matters here more than anywhere, because the runs this campaign has are the ones that died: [#1375](https://github.com/mudler/vllm.cpp/issues/1375) is `child exit=-15` at 0 frames, and [`ltx25-decode-speed.md`](ltx25-decode-speed.md)'s two rungs are `EXIT=137` and `EXIT=1` at 0 frames. A reader who takes "the render writes a phase table" at face value will expect a 2.5 h render that is killed at 2.4 h to leave a table naming where it was. It leaves none. #1413 CLOSED the live half, in `## W0-live` below — a line per phase boundary and per DiT forward, so a killed run is legible from its stderr; a signal handler that flushed a PARTIAL table on abort is a separate change with its own re-entrancy argument and is not owed by either |
| the phase table's DEVICE column | W1, and it needs [#1126](https://github.com/mudler/vllm.cpp/issues/1126) first | **owed, and a LEASE WILL NOT CLOSE IT.** The column is defined as the driver's live in-use bytes, read per phase through `vt::Backend::DeviceMemoryInfo`. It reports the `-1` no-probe sentinel in W0's artifact, and there are TWO reasons stacked, only one of which is a scheduling problem. **(1)** The render W0 could take was the CPU arm, where the sentinel is correct — `dgx:gpu0` was busy with two queued jobs and `orin:gpu0`, the one free device, holds no LTX-2.5 checkpoints. **(2)** The one that matters: **`CudaBackend` does not override `DeviceMemoryInfo` at all**, which is [#1126](https://github.com/mudler/vllm.cpp/issues/1126), and `grep -rn 'DeviceMemoryInfo' src/vt include/vt` returns exactly the base declaration at `include/vt/backend.h:94` and one override, `src/vt/rocm/rocm_backend.hip:358`. So a CUDA render on `dgx:gpu0` would print `-1` in every row of this column too, and it would print it for a reason no lease can fix. **What the column cannot report today, stated as three things:** how many device bytes the DiT staging leaves resident; whether the denoise grows device residency across steps; and whether the ~59 GiB #1014 asks about is device-class at all. **Why W0 did not just wire it:** `include/vllm/platforms/interface.h:68-72` records that CUDA's absence from that seam is load-bearing — `Gemma4MoE`'s device-expert LRU is the seam's only consumer and is DEAD on CUDA. Narrowed, because the broad form of this sentence is wrong: the bf16 arm `EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:548`) is dead for a SECOND, independent reason — it refuses at `:571` on `vt::HasMatmulBTAlphaBeta`, implemented only by ROCm ([#1205](https://github.com/mudler/vllm.cpp/issues/1205)) — so the override alone would not wake it. The arm the override WOULD wake is the FP8-native one, `EnsureGemma4Fp8NativeOnDevice` (`:611-628`), which reaches `MakeRoom` behind no device gate but the probe, with a 2048 MiB fill-only budget by default (`:416-432`). One named arm, not every CUDA model — and still a behaviour change with its own measurement. That is #1126's change to make, not an instrument's. **What W1 does instead, until #1126 lands:** sample `nvidia-smi --query-compute-apps=used_memory` per phase beside the table — the fallback this spec's W6 row already names, and the one instrument GB10 answers, since `--query-gpu=memory.used` returns `[N/A]` there. On GB10 the peak HOST column is not a poor substitute either: the pool is unified, so host resident bytes and device bytes are the same 119 GiB arena, and that column does report |
| [#1040](https://github.com/mudler/vllm.cpp/issues/1040) | W0 (contract) + W1 (closes) | contract half **met** — the table is a file beside the frames rather than a console line, and it is retrievable from this repo at `benchmarks/demo/`. The closing half is W1's |
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
| [#1439](https://github.com/mudler/vllm.cpp/issues/1439) | W0 (its own gate) | **owed, and RED on `main` rather than on any branch.** `CHECK(leaves >= 0.95 * wall)` in `ltx2 video: a render through the ABI emits a phase table that SUMS to wall` is a RATIO, and the un-named residue is 4.80% to 6.32% of `wall` across twelve runs on one x86 box - so the 95% floor sits INSIDE the measurement's own range at the 64x64 / 9-frame fixture scale, and the case decides by coin flip, mostly red. With this lane's four files reverted so the binary is main at `89261c955`, six runs read 94.32%, 95.20%, 93.74%, 94.20%, 94.69%, 94.19%; the W0-live merge reads 93.82%, 93.68%, 94.34%, 94.62%, 94.39%, and 94.12% with `VLLM_RENDER_PROGRESS=0`, which exonerates the live emitter. Box contention is NOT the cause and main's one green disproves it: that run had `wall=0.579684s`, more than double the others, because the box was loaded - a SLOWER render passes. The tolerance was argued for the 21.004 B render, where the residue is a far smaller fraction of the wall. Naming the un-named time, or bounding `unaccounted_seconds` beside the ratio so the assertion says the same thing at both scales, is gate semantics and owes its own row |
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
* **A VAE-SIDE sub-scope for `decode.video`.** The anchor W0 lands sits in the
  driver: `decode.video.chunk` runs from the leaf's own open to the moment
  `Ltx2VideoDecodeStreaming` hands a chunk back. That END is a production event
  and it is what catches M10, but the WORK it wraps is the whole call rather than
  the tile accumulation inside it. The honest sub-scope is around
  `AccumulateTemporalGroup`
  (`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp`), which is outside
  the authority W0 was dispatched with. Owed to W5, whose lever this phase is,
  and it is a refinement of a gated phase rather than an ungated one.

## Outcome — W0, the instrument

Landed on `row/LTX25-RESIDENCY-W0`, issue
[#1010](https://github.com/mudler/vllm.cpp/issues/1010), base `5f68e60df`.

### The gate, and the number

**PASS.** One completed render through the `vllm.h` video ABI emits a phase
table whose named leaves sum to **99.94% of wall** — 4.459892 s of leaves
against 4.462572 s of wall, residue **2.68 ms** over 33 entries and 118 samples.
The tolerance was fixed at **>= 95%** in the test's own comment and in the
red-first commit message *before* the instrumented run, and the sum is checked
in the same case that checks the named boundaries, because one leaf called
`render` would sum to wall exactly and measure nothing.

The same case has produced **98.33% at a 0.158 s wall, 98.74% at 0.147 s, 99.88%
at 1.676 s, 99.94% at 4.463 s, 99.96% at 6.138 s and 99.84% at 12.030 s** on six
runs of the identical binary at the identical geometry. The RATIO is what the
gate reads and it never came near the 95% floor; the WALL moved by a factor of 82
across those runs because the box was building other sessions' trees — and the
0.147 s and the 4.463 s are ONE MINUTE apart, which is the sharpest form this
finding has taken. That is why no wall figure in this section is a benchmark and
why W1 is written as a lease on an idle box.

The evidence file is
[`benchmarks/demo/ltx25_phase_log_fixture_cpu.json`](../../benchmarks/demo/ltx25_phase_log_fixture_cpu.json),
sha256 `59a860163e6ab2569c789209be709858915ba58393e8eb6900559f192ea7b950`.
**Re-taken twice.** First for the first review's F4, which found the original
artifact carrying no provenance at all in a directory whose every sibling carries
`_source`, `hardware`, `workload` and `footnotes`; then for the second review's
finding above, because an artifact taken before the `denoise.step` and
`decode.video.chunk` anchors existed cannot show the containment the gate now
requires. Every phase record in it is the render's
own output; the four keys above them — `_source`, `_caveat`, `_headline`,
`_footnotes` — are written by hand at commit time, because a render cannot know
its host, its checkpoint or what else the box was doing, and those are exactly
the facts whose absence made #1040 and #1087 unreadable. The library now writes
the part it CAN know into every phase log it produces: `notice`, `sum_rule` and
`sampler_enabled`. Provenance:

| | |
|---|---|
| producer | `examples/ltx2_gen` (`build/examples/ltx2-gen`), through `vllm_video_engine_load` + `vllm_video_generate` |
| tree | binary built from `3dc2ae98b` on `row/LTX25-RESIDENCY-W0` |
| build | `cmake -DCMAKE_BUILD_TYPE=Release -G Ninja`, gcc 13.3.0 |
| host | `mudler-ubuntu-box`, Linux 6.8.0-136-generic x86_64, **20 cores and CONTENDED** — 1-minute load average between **31 and 55** across the render and the one a minute before it, other sessions building throughout |
| checkpoint | the reduced-dimension fixture `tests/vllm/multimodal/ltx2_video_fixture.h` writes, in the shipped file format |
| geometry | `--frames 9 --width 64 --height 64 --seed 7 --max-phase 0 --device cpu` |
| completion | 9 frames written, **9 distinct per-frame md5s**, plus a 48 kHz WAV. The exit code is not the completion gate here ([#1149](https://github.com/mudler/vllm.cpp/issues/1149)); the distinct md5s are |
| device column | **every row reads `-1`.** See `## Owed` — this is the CPU arm, and `CudaBackend` would read `-1` as well |

**The wall figures are NOT a benchmark and nothing may quote them as one.** The
host was contended, the checkpoint is a reduced fixture, and the same case has
measured 0.15 s, 0.16 s, 1.68 s, 4.46 s, 6.14 s and 12.03 s of wall on six runs
of the identical binary. What the artifact supports is the SHAPE of the table and the fact that
its parts add up. `docs/BENCHMARKS.md` is therefore untouched, which is what
this spec's `### Decisions taken here` already said W0 owes: W1 owes the
benchmark edit, on a leased idle box, at two geometries.

### What the table says about this render, which is not what anybody expected

Two leaves are the whole render on this fixture, and **which of them is larger
changed between two runs of the identical binary**. On the artifact this row
first shipped, `denoise` was 8.127 s (67.7% of the named leaves) against
`decode.audio` 3.062 s (25.5%). On an earlier run of the same case at the same
geometry the order was reversed: `decode.audio` 0.0889 s against `denoise`
0.0539 s of a 0.158 s wall. The first re-take recorded a third split again —
`denoise` 48.3% against `decode.audio` 42.6% — and **the artifact this row now
ships records a fourth**: `denoise` 75.79% against `decode.audio` 18.07%. The
gap between the two has been 2.7x, 0.6x, 1.1x and 4.2x on one binary at one
geometry.

The re-take also names something the first artifact could not, because F1 split
it: **`decode.audio.vocoder` alone is 0.8006 s, 17.95% of the named leaves, and
99.998% of `decode.audio`.** Almost the whole of `decode.audio` is the vocoder,
and until this row that phase had no name of its own anywhere in the tree. The
same is now true one level up: `denoise.step` covers **99.993%** of `denoise`
over 8 denoiser evaluations, and `decode.video.chunk` **99.982%** of
`decode.video`.

That reversal is worth more than either number. The host was building other
sessions' trees throughout, and the two phases do not have the same threading —
so a contended box does not merely add noise to a phase table, it can change the
RANKING the table reports. This campaign exists because a stale ranking was
taken as a profile, and W1's own gate is an idle leased box for exactly this
reason. Nothing here says anything about the 21.00B checkpoint, whose DiT load
alone is minutes; what it says is that the table is now capable of showing an
ordering, and that reading one off a contended run is how the last ranking went
wrong. That ordering belongs to a two-block reduced DiT and says
nothing about the 21.00B checkpoint — it is recorded because it is the first
time any LTX-2.5 render in this tree has said where its time went at all, and
because it is exactly the kind of term the spike's lever table could not have
ranked.

### What was rejected, and why

**A shared phase table with the process as its scope was rejected**, after it
shipped and failed. `Begin` was first written idempotent, so a process that
loaded several engines accumulated one timeline and reported the time between
unrelated renders as this render's residue: the 90-case suite read 8.33 s of
residue against a 5.20 s budget on a render that had accounted for 99.96% of its
own wall. A load now discards the earlier timeline. The origin still sits at the
LOAD and not at the generation, because the spike measures ~7.5 minutes of DiT
staging paid at the front of every render.

**Chaining the phases into a partition was rejected.** A `Mark(name)` that
closed the open phase and opened the next would make the sum hold by
construction, and the missing time would be silently charged to whichever phase
happened to be open. The scopes therefore leave real gaps, and the residue is a
real quantity — which is what makes the gate refutable at all, and what mutation
M2 below demonstrates.

**Appending a field to `vllm_video_result` was rejected** in favour of the
`vllm_video_last_phase_log(engine)` query. Growing an OUTPUT struct is the one
append a caller cannot absorb by zero-initializing, because the library writes
the new field using its own `sizeof`.

**Instrumenting `vt`'s allocator for the device column was rejected** as a
different row's change: it is a hot path taken thousands of times per step. The
column reads `vt::Backend::DeviceMemoryInfo` instead, which is backend-neutral
and is the shape a device-byte question already has in this tree.

**Overriding `CudaBackend::DeviceMemoryInfo` so the column would answer was also
rejected, and this is the one rejection a reader should not skip.** It would
have made W0's own gate look complete on a CUDA box, and it is
[#1126](https://github.com/mudler/vllm.cpp/issues/1126)'s change:
`include/vllm/platforms/interface.h:68-72` records that the seam's only consumer,
`Gemma4MoE`'s device-expert LRU, is DEAD on CUDA. **This row first wrote that as
"wakes a landed residency policy on every CUDA model", which claims more than the
code supports and is corrected here.** The bf16 arm
`EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:548`) is separately dead behind
`vt::HasMatmulBTAlphaBeta` at `:571` — ROCm holds the only implementation
([#1205](https://github.com/mudler/vllm.cpp/issues/1205)) — so the override
would not wake that one. It would wake `EnsureGemma4Fp8NativeOnDevice`
(`:611-628`), which reaches `MakeRoom` with the probe as its only device gate and
a 2048 MiB fill-only default budget (`:416-432`). One arm, named, and still a
behaviour change with its own measurement — which is the reason, and the reason
did not need the overstatement. An instrument that changes model behaviour to
make its own column non-empty is not an instrument. The column reports `-1` and
`## Owed` says why.

### Mutations, each with its diff, its compile status and its exit code

| # | Mutation | `git diff --stat` | compile | focused gate |
|---|---|---|---|---|
| M1 | delete the `WritePhaseLog` call in `Generate` — the **reachability** mutation | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 4 assertions, 1 failed |
| M2 | mark the `denoise` scope a span so its time is named by nobody | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 273 assertions, 1 failed, on `leaves >= 0.95 * wall` (0.078 s of 0.117 s) |
| M3 | make `HostResidentBytes` return `-1` | 1 file, +4/-1 | rc 0, 0 errors | **RED**, exit 1, 273 assertions, **21** failed, one per phase entry |

M1 is the mutation `.agents/reachability.md` asks for: the production call site
is deleted in a scratch copy and the focused gate goes red, so the gate measures
a capability rather than a class.

### What a fresh review found, and the mutations that closed it

A fresh reviewer returned `PASS_WITH_FINDINGS`. It reproduced M1, M2 and M3
exactly, confirmed the instrument is reachable on the shipped default, and found
no correctness defect. Its central finding is the one that mattered, because W0
gates a campaign: **this gate could not tell a correct table from a useless one,
and the reviewer built the useless one.**

**F2 — existence plus a sum is not attribution.** Mutation M4 leaves the
`decode.video` leaf open across the audio decode and gives `decode.audio` its
name with no work beneath it. All six required names are present, no leaf nests,
and the leaves account for 99.9% of wall — so the gate passed, over a table that
reported the video decode as 32% of the render (it is 2.4%) and the audio decode
as free (it is a quarter of it). A W1 reader ranking levers off that sends W5,
this campaign's largest stage, at the wrong phase and drops the audio decode
entirely. Seven of the 21 leaves also read 0.0000 s on the gate's own render, so
nothing about their placement was proven either.

**The suggested repair was a differential over two frame counts, and this box
refuted it.** Render at 9 frames and at 33, and require the phases whose work
scales with the clip to grow. It was written, built and run here. The 9-frame
render measured **8.03 s** of named leaves and the 33-frame render **3.80 s**,
minutes apart on the same binary — the 2.5x longer clip cost HALF the time — and
an earlier run of the same 9-frame render measured **4.80 s**. Wall noise here is
a factor of two in both directions against a 2.5x signal, so a
seconds-differential is a coin flip wearing a gate. The form is dropped and the
measurement is kept, because it is the same finding this section already records
at 76x and it is the concrete reason W1 is written as a lease on an idle box.

**What closed F2 instead needs no clock.** `decode.audio` declares that it covers
the audio decode, and the audio decode is exactly two calls —
`Ltx2AudioDecoderForward` and `Ltx2VocoderWithBweForward` — each of which F1 gives
a scope of its own. The new case asserts **containment**: each sub-scope's
interval lies inside the interval of the leaf that claims to cover it, the two
together cover at least 90% of it, and the decode and writer leaves never
overlap. Every number comes from one clock in one run and no threshold is
crossed, so contention cannot move the verdict. A **share floor** of 0.05% of the
leaf sum covers `denoise`, `decode.audio` and `decode.video`, the three phases
that carry this render: a name detached from its work measures two function
calls, five orders of magnitude below the floor, while the smallest of the three
holds 3.7%. **That repair covered ONE of those three, and the next section is a
second fresh review demonstrating what the other two could still do.**

**F1 — #1010 named six phases and two of them were folded away.** `decode.audio`
carried `Ltx2AudioDecoderForward` and `Ltx2VocoderWithBweForward` in one leaf,
which on the first artifact was 3.062 s, 25.5% of wall, the second-largest phase
in the table and un-decomposed. The two-stage recipe's latent spatial upsampler
ran inside `phase.prepare`, a leaf whose name does not mention it. Both are now
split as NESTED leaves — `decode.audio.mel`, `decode.audio.vocoder`,
`phase.upsample_latent` — which decomposes them without moving what the table
adds up to, since nested records are excluded from the sum. The vocoder split is
what makes F2's containment invariant expressible at all; the upsampler is gated
by the two-phase DFR case, the only render in `test_ltx2_video` that reaches it.

**F3 — the gate reddened on the next refinement it needed.** The reconciliation in
the SUMS case accumulated every record that was not a `span`, while the emitter's
own `Sum` skips `span || nested`. So the case silently asserted "no nested leaf
has a non-trivial duration" although the header advertises nesting as supported,
and splitting the mel decode out failed it on `CHECK( 0.0118791 < 1e-06 )` — a
message that says nothing about nesting and reads as "the emitter does not
reconcile". Repaired to skip both, which is what made F1's split possible at all.
M6 below re-runs the reviewer's demonstration against the landed split.

**F4 — the artifact had no provenance and sits under `benchmarks/`.** Every
sibling in that directory carries `_source`, `hardware`, `workload`, `headline`
and `footnotes`; the first phase-log artifact carried none of them, so a reader
opening `denoise 8.13 s (67.6%)` beside `decode.audio 3.06 s (25.5%)` in a
directory called `benchmarks/` had nothing telling them the host was contended,
the checkpoint was a two-block fixture, or that the rank of those two phases had
reversed between two runs. That is #1040 and #1087's failure in miniature, inside
the row that exists to stop it. Two changes: **the emitter now writes the caveat
into every phase log it produces** — `notice`, `sum_rule` and `sampler_enabled`,
so the warning travels with the file rather than living in a document a later
reader would have to know to look for — and the committed artifact carries a
`_source` provenance header in the shape its siblings use.

**F5 — `## Now` quoted the superseded run.** It carried 98.33% of a 0.1575 s wall,
which is the run whose phase ranking this section says may not be quoted. It now
carries the artifact's own 99.84%, with both walls named as the non-benchmarks
they are.

**F9 and F10 — the sampler's lifetime.** `StopSampler` hands the thread object out
under `mu` and joins it outside; between those two points an `Open` on another
thread could run `StartSamplerLocked`, which cleared the single `stop` member the
old worker was still reading, so the old worker never exited and the join blocked
forever with two samplers live. The flag is now owned by the worker that reads
it, so nobody else's start can clear it. Separately, nothing but `Reset()` and the
destructor ever stopped the sampler, so a server that rendered one clip kept a
100 ms `/proc/self/statm` read under the process-wide mutex for the rest of its
life and accumulated that idle time into the next table's sample count. The last
`Close` now stops it and `Begin` stops whatever the previous timeline left
running. On this driver that is two starts per process, because the `load` and
`generate` spans each stay open across everything beneath them.

**F6 and F8 — two one-line record defects.** `docs/FEATURES.md`'s ABI capability
table listed the video entry points and omitted `vllm_video_last_phase_log`. And
the comment above the monotone-`start` assertion said the sequence holds *because*
the records are appended in completion order, which is backwards: completion order
is what would break it (`load` closes at start 0.0001 and is appended after
`load.prompt_embeds`, which starts at 0.0625), and what the line actually pins is
`ByStart`'s `stable_sort`.

**A claim this row made about #1126 was too broad, and is narrowed.** See `## Now`
and `## Owed`: the override would wake one named arm, not "a landed residency
policy on every CUDA model".

| # | Mutation | `git diff --stat` | compile | focused gate |
|---|---|---|---|---|
| M4 | leave `decode.video` open across the audio decode and give `decode.audio` its name with no work beneath it — the **attribution** mutation | 1 file, +4/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **26 assertions, 4 failed** — two containment failures (`start >= audio_start`), the mel/vocoder ordering, and the floor at 8.85e-5% against 0.05% |
| M6 | revert F3: reconcile over `span` only, so a nested leaf is summed twice | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, 302 assertions, 1 failed, on `fabs((wall - leaves) - unaccounted) < 1e-6` |

**M4 is the acceptance test for F2, and the number that matters is the one
beside it.** Under M4 the pre-existing SUMS case still reports **exit 0, 1 case
passed, 302 assertions, 302 passed**, at 8.00802 s of leaves against 8.01027 s
of wall — 99.97% accounted over 23 entries, every required name present. So the
gate that shipped is blind to M4 exactly as the reviewer said, and the
containment case is what sees it. On the unmutated tree that same case reports
exit 0, 26 assertions, 26 passed, with mel+vocoder covering **99.9994%** of
`decode.audio` and the three floors clearing by 40x (`decode.video`, 1.99%),
286x (`decode.audio`, 14.32%) and 1562x (`denoise`, 78.10%).

**Read those three shares beside the artifact's own and the point makes
itself:** the artifact recorded `denoise` at 67.66%, `decode.audio` at 25.50%
and `decode.video` at 3.73%, and the run above recorded 78.10%, 14.32% and
1.99% on the same binary at the same geometry. The SHARES move by a factor of
two on this box, which is why F2 asserts a floor three orders of magnitude below
them rather than a value, and why no ordering in any table this row produced may
be quoted.

### What a SECOND fresh review found, and what closed it

The repair above asserted containment for `decode.audio`, because `decode.audio`
was the only leaf with sub-scopes. `denoise` and `decode.video` were held by the
0.05% share floor alone, and the non-overlap loop did not mention `denoise` at
all. A second fresh reviewer ran M4's shape one level over, on the phase that
carries this render.

**M7 — the transfer, on the largest phase in the table.** Close `denoise` after
the first sampler step and open `phase.finish` there. No overlap, no nesting, the
sum untouched: 1 file, **+7/-0**, compile rc 0, 0 errors. The containment case
reported **exit 0, 1/1 cases, 26/26 assertions**. The SUMS case reported **exit
0, 1/1 cases, 314/314 assertions**, 99.94% accounted over 24 entries. The table
it emitted put `phase.finish` at **2.129 s, 55.0%** of a 3.873 s leaf sum and
`denoise` at **0.232 s, 6.0%**, on a binary whose honest run measured `denoise`
at **73.4%**. **82% of the denoise was re-labelled and both gates passed.**

That is the same defect M4 named, on the phase #1024 and #1087 are about, in the
table W1 ranks this campaign's levers from. The reviewer recommended a prose
correction. The operator overrode it, and the right call: a prose correction
would have been honest and would still have shipped a gate that cannot see an
82% misattribution.

**What closed it: every carrying phase now has an anchor.** `denoise.step` wraps
the denoiser evaluation inside `Evaluate`, which every sampler arm reaches — the
first-order loop calls it directly and the res_2s loop reaches it through
`hooks.denoise` — so one nested leaf per evaluation says where the sampler
actually spent its time. `decode.video.chunk` runs from the leaf's own open to
the moment the streaming decoder hands a chunk BACK, so its end is a production
event rather than an instrument statement. Both are nested, so
`sum_leaf_seconds` does not move.

**And a third assertion, which sees a transfer directly rather than through a
sum: EXCLUSIVITY.** No other leaf may overlap the window a phase's sub-scopes
span. Containment says "the work is not inside the name"; exclusivity says "a
second name is inside the work". M7 fails both — seven of its eight
`denoise.step` records fall outside the shortened `denoise` leaf, and
`phase.finish` opens in the middle of the denoise window while the steps keep
running around it.

**Why the sub-scope for `decode.video` sits in the driver and not in the VAE.**
The per-chunk decode itself is `AccumulateTemporalGroup` in
`src/vllm/model_executor/models/ltx2_video_vae_tiled.cpp`, one directory outside
this stage's authority. The driver-side anchor is weaker and it is not nothing:
its END is the production callback firing, so a `decode.video` leaf that closes
before its chunk arrives, or that is re-labelled after one, stops containing the
chunk it produced. M10 below is that case. A VAE-side sub-scope naming the tile
accumulation is **owed** and is listed under `## Owed`.

| # | Mutation | `git diff --stat` | compile | containment case | SUMS case |
|---|---|---|---|---|---|
| **M7** | close `denoise` after the first sampler step and open `phase.finish` there — the **transfer** mutation | 1 file, +7/-0 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **124 assertions, 8 failed** — 7 containment, 1 exclusivity | exit 0, 434/434 |
| M4 | re-anchored onto the repaired tree: leave `decode.video` open across the audio decode and give `decode.audio` a name with no work | 1 file, +2/-2 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **124 assertions, 5 failed** | exit 0, 422/422 |
| M10 | the M7 shape on the VIDEO side: after the first chunk, re-label the decode as the writer | 1 file, +1/-1 | rc 0, 0 errors | **RED**, exit 1, 1 case failed, **122 assertions, 1 failed** | exit 0, 422/422 |
| M9 | revert F10: the last `Close` no longer stops the sampler | 1 file, +0/-1 | rc 0, 0 errors | n/a | **RED**, exit 1, 1 case failed, 2 assertions, 1 failed, at 10 samples against 4 |

The SUMS column is not a defect. It is the measurement: **a sum cannot see a
transfer**, in three independent mutations, which is exactly why the containment
case exists and why it is the one that must be run.

On the unmutated tree the containment case reports **exit 0, 1/1 cases, 124/124
assertions**, with `denoise.step` covering **99.667%** of `denoise` over 8
evaluations, `decode.video.chunk` **99.439%** of `decode.video` over 2 leaf
records, and mel+vocoder **99.995%** of `decode.audio`.

**Two more findings from the same review, repaired here.**

*The coverage threshold and the share floor were both mute switches.* At 0.90,
`decode.audio` could have opened 11% early and swallowed 0.13 s — 87% of this
render's entire `decode.video` — while passing everything in the file. Coverage
is a ratio of two intervals measured inside the same contention window, so it is
the one number here that a loaded box does not move, and the audio threshold is
now **0.99**. The three thresholds are deliberately not one number:
`decode.video` is 1.6 ms on this fixture, where a single preempted
`/proc/self/statm` read at a scope boundary is percent-scale, so its threshold is
the loosest at 0.90 and the reason is written beside it. **The 0.05% share floor
is kept and its margin is NAMED rather than tightened**, because the measured
share is precisely the quantity this box destroys: the same binary at the same
geometry has reported `denoise` at 38.1% and at 73.4%, `decode.audio` at 50.9%
and at 16.7%, and `decode.video` at 5.27% and at 2.13%. A floor set at a fraction
of the measured value would be a flake, and under M7 `denoise` sat at 6.0% and
would have cleared any floor a quiet box could justify. The floor now reads "this
name is not detached" and nothing more.

*The emitter baked measured wall times into library source.* The `notice` string
named "0.158 s, 6.138 s and 12.030 s" from one contended box, in `src/`, where no
gate reads it and the next run that refutes it would have to edit a source file
to say so. That is a live number in a place nobody looks — the failure this row
exists to stop, one directory over. The notice is qualitative now and the
artifact's `_caveat` keeps the numbers beside the run they came from.

### No claim file, and the checker says why

A `CLAIM-LTX25-RESIDENCY-W0.md` was written and then removed.
`scripts/check-agent-record.py:1712-1714` refuses an active claim whose `Row IDs`
cell holds no ID its `ID_RE` recognises, and that pattern is built from the same
fixed prefix list (`:327-338`) that keeps this campaign out of every matrix —
`LTX25-` is not in it. So a claim file for this stage is a record no checker can
validate, sitting beside a row no matrix carries. The stage's ownership,
evidence and exclusions are recorded here and in the pull-request body instead,
which is where this spec's `### Decisions taken here` already said they would be.

### Owed out of W0

* **The device column has never executed.** See `## Owed`. Every
  `peak_device_bytes` in the artifact is the `-1` no-probe sentinel, and it would
  read `-1` on CUDA too, for the reason #1126 names.
* **A render on the SHIPPED 21.00B checkpoint has not been instrumented**, which
  is W1's whole job. Nothing here says what the table looks like when the DiT
  load is minutes rather than milliseconds; every number in this section is the
  reduced two-block fixture.
* **`GenerateAudioOnly` writes the table but has no gate on it.** The t2a arm
  flushes through the same helper and the case that would check it was not
  written; the video arm is now gated at three levels — the sum, the named
  boundaries, and the containment case.
* **No gate reads a phase table on a host that is not contended,** and this is
  the one W1 must close first. The seconds-differential written for F2 was
  refuted on this box, and the refutation is a property of the box: the same
  binary at the same geometry has moved a factor of 76 in wall and has reversed
  the rank of its two dominant phases. Until W1 takes a lease on an idle box, no
  DURATION in any phase table this row produced may be compared with any other
  duration, including its own from the run before.
* **The sub-millisecond leaves are named, not measured.** `generate.setup`,
  `generate.geometry`, `generate.guiders`, `generate.image_cond`,
  `generate.audio_input`, `generate.retake` and `phase.finish` are each under
  0.001% of the leaf sum on the fixture; F2's containment and floor cover the
  three phases that carry the render and say nothing about these. That is a fact
  about the driver — they really are bookkeeping — but it means their PLACEMENT
  is unproven, and a shipped-checkpoint render is where it would show.
* **F10 IS NOW GATED, and the sentence that said it could not be was wrong.**
  This entry used to read "F10 is an absence — a thread that keeps running —
  which no assertion in this suite is positioned to observe", and asserted that a
  gate would need an injected scheduler. A second fresh review refuted it in
  twelve lines against the PUBLIC surface: `PhaseLog::Samples()` is a counter the
  worker increments every 100 ms, so a worker that outlived its timeline moves a
  number a test can read. Those twelve lines are now
  `ltx2 phase log: the last Close stops the sampler` — the first `PhaseLog`-level
  unit case this instrument has, and the first gate in the file that does not pay
  for a complete render. Reverting F10 (dropping the `TakeSamplerLocked` in the
  last `Close`) makes it fail: **exit 1, 1 case failed, 2 assertions, 1 failed**,
  at 10 samples against 4 over 600 ms of idle. The lesson is the one the row
  keeps re-learning: "no assertion is positioned to observe it" was a claim about
  what had been tried, written as a claim about what is possible.
* **F9 is FIXED AND UNGATED, and that justification stands.** F9 is a race
  between `Reset()` and `Open` whose window is a few instructions wide. It is a
  logic race and not a data race — every access is already under `mu` — so a
  sanitizer would not flag it, and a test that loses it reliably would have to
  instrument the emitter's internals. What the suite proves for F9 is that the
  repair regressed nothing, not that it is detected.
* **`phase.upsample_latent` is gated only on the reduced two-phase fixture.** It
  is the only leaf whose sole reader is the DFR case, because every other render
  in `test_ltx2_video` pins `max_phase = 0`, where the input transform is never
  the spatial upsample.

## W0-live — the lane that runs while the render is alive (#1413)

W0 above landed the table. This section is the half of #1010 that table does not
reach, filed separately as
[#1413](https://github.com/mudler/vllm.cpp/issues/1413) so the two are
attributable to the two changes that made them.

### The gap, in one sentence each

**The table is written by the success path only.** `WritePhaseLog` is reached
after `im.trace.completed = true`, so a render that is killed, aborted by a lease
governor, or still running writes nothing. That is not a corner: #1375 is
`ABORT[92] PROJECTED OVERRUN`, `child exit=-15`, **0 frames**; the spike's rung 1
is `EXIT=137`, 0 frames; rung 2 is `EXIT=1`, 0 frames. The instrument reports on
the runs that finish, and the runs the campaign has are the ones that do not.

**Nothing at all is emitted while the render runs.** `PhaseLog::Open` and
`PhaseLog::Close` record and print nothing, and `VLLM_RENDER_PHASE_LOG_STDERR`
fires inside `WriteJson`, i.e. on the success path again. Between
`ltx2-gen: family=...` and `wrote N frames` a 2.5-hour render is silent, so
**working and hung are byte-identical from outside**, and hours of GPU lease have
been spent telling them apart.

**The unit that costs the wall has no counter.** `denoise` is one leaf. #1375
measures **~162 s per DiT forward** at 21.004 B with **60 forwards structural**
(30 steps x 2 CFG legs — `cfg_scale != 1.0` forces the unconditional branch at
`ltx2_pipeline.cpp:521-523`). ~2.7 h inside one leaf that reports one number, at
the end, if the run survives to write it.

### Why an external sampler is not the fallback

It was tried and it does not work. #1375 records `phase=OTHER` throughout,
because **`eu-stack` unwinds zero frames inside the `rc` worker container**;
`dit_runs=0` is that blind counter and is not evidence of absence. The 162 s is a
wall-clock interval between GPU busy/idle edges, not a profile, and #1375's own
"What is NOT established" names an in-process phase marker as the way to
attribute it. This lane is that marker.

### The shape, and why it is this shape

Two emitters, both on stderr, both on the shipped default.

1. **A boundary line on open and on close.** `+name` when a phase opens,
   `-name` when it closes with its duration and its peak host bytes. The open
   line is the load-bearing one: it means **the last line printed names the phase
   that is currently running**, which is the whole difference between a hang and
   work. A close-only emitter would have printed nothing for the 3002 s
   conditioning stretch, because that stretch never closed.

2. **A tick per DiT forward**, carrying the recipe phase, the sampler step
   `k/N`, the cumulative forward index, elapsed, and `last=` — the seconds since
   the previous tick of the same unit. That `last=` is the per-forward cost
   #1375 could only obtain as an interval between GPU edges, and it is now a
   number the process itself reports, per forward, on every run including the
   ones that die.

   **The tick fires BEFORE the forward, not after.** After is the obvious
   placement and it is wrong here: the line a reader most needs is the one naming
   the forward that is in flight when the run stops. The cost of that choice is
   stated rather than hidden — the last forward of a completed render has no
   `last=` line of its own, and its cost is inside the `-denoise` boundary line.

**Cadence.** 60 tick lines and ~50 boundary lines over 2.7 h. Not chatty enough
to drown a log; not quiet enough that a 162 s forward reads as a hang. That 60
is #1375's config and it moved while this lane was in review:
[#1092](https://github.com/mudler/vllm.cpp/issues/1092) gave
`Ltx2DitForwardDevice` its `perturbations` argument, so the device-resident arm
runs up to four forwards per step rather than two and the same geometry emits
up to 120 ticks. Neither number is printed and neither had to be re-derived:
the forward counter carries no denominator by the paragraph below, and the step
fraction reads this recipe phase's own `sigmas`.

**No denominator on the forward counter, deliberately.** 60 = 30 x 2 is true of
the config #1375 measured and it is not structural in this code: the sampler
decides how many denoiser calls happen and `Ltx2GuidedDenoise` decides how many
forwards each call is (one to four — cond, uncond, ptb, mod). The **step**
counter carries the honest fraction `k/N`, because `sigmas.size() - 1` is exact
and in scope. Printing `forward 24/60` would be an instrument guessing, and this
row exists because guessing is what the campaign has been doing.

### Not behind a flag, and the reason is measured

`VT_H3_PROGRESS` is the existing shape for this in the tree
(`minimax_h3.cpp:776-793`): per-step forward seconds, on stderr, for the
MiniMax-H3 denoise loop. It is **opt-in**, and that is exactly why no LTX-2.5 run
has one — the runs whose profile is needed are the long ones, on a leased box, by
somebody who did not know they would need the number until afterwards. The
failure this closes happened on default settings.

`VLLM_RENDER_PROGRESS=0` silences it, in the same measurement-lane shape
`VLLM_RENDER_PHASE_SAMPLER` and `VLLM_LTX2_POOL_DRAIN` already take here: it
exists so an A/B over what the emitter itself costs runs on ONE binary. It is
never a configuration and nothing in the tree sets it.

### Cost

One `std::fprintf` to stderr per phase boundary and per DiT forward, flushed,
under the process-wide phase mutex the 100 ms sampler also takes — so a tick is a
held global lock plus a flushed write rather than a bare `fprintf`, which at ~110
lines per render is a footnote and is written down rather than left in a profile.
On the 21.004 B geometry that is ~110 writes against 2.7 h of wall. There is **no
per-token and no per-tile emission**: the VAE decode's tile loop and the frame
writer's inner loop are untouched, and the decode's per-chunk boundary lines are
bounded by the chunk count, which is bounded by the frame count.

**`VLLM_RENDER_PROGRESS=0` costs one `getenv` per process, and that took a second
edit to become true.** `ProgressEnabled()` caches the read and `Tick` returns on
its first line, but the `detail` argument is built by the CALLER and was built
unconditionally: a `std::string` from a literal, three `std::to_string`s and four
concatenations, per DiT forward. Against a 162 s forward the magnitude is
nothing; the sentence was still false as written, and a cost claim that is false
in the small is the shape that gets quoted in the large. The call site now asks
`phase::ProgressEnabled()` before it formats anything, which is why that function
is declared in the header instead of living in an anonymous namespace.

### Gate

`ltx2 video: a render through the ABI PRINTS its progress while it runs`, in
`tests/vllm/multimodal/test_ltx2_video.cpp`. It captures **real stderr** with
`dup`/`dup2` across `vllm_video_engine_load` and `vllm_video_generate` — not a
test-only sink — so what it asserts is exactly what a user sees on default
settings, and there is no seam that could be armed in the test and absent in
production. It asserts:

* the boundary lines for `load.dit`, `denoise`, `decode.video` and
  `artifacts.frames` appear, each with a matching `+` and `-`;
* `+denoise` precedes every `dit forward` tick and `-denoise` follows every one
  of them, which is what says the ticks are inside the phase they claim;
* at least two ticks, with **strictly increasing** forward indices — the step
  count moves;
* every tick from the second onward carries `last=`;
* `t=` is non-decreasing over the whole capture.

The strictly-increasing assertion is the non-vacuous half. An emitter that
printed a constant `forward 1` would satisfy "the marker appears" and would
report nothing, which is the failure mode the spike's sampler already had.

### Owed out of W0-live

* **A capture from a real 21.004 B render.** Every number in this section about
  what the lane prints at that scale is derived from #1375's measured 162 s, not
  observed. W1's lease closes it, and the format is fixed here so that run does
  not have to negotiate one.
* **The last forward of a completed render has no `last=` line**, by the
  before-the-forward placement argued above. Its cost is inside `-denoise` and a
  reader who needs it per-forward has n-1 samples, not n.
* **`GenerateAudioOnly` emits boundary lines and no ticks**, because the t2a arm
  reaches the same `Evaluate`. Unverified: no t2a case captures stderr.
* **Nothing writes a partial table on abort.** This lane makes the *phases*
  visible on a killed run; it does not make `phase-log.json` appear for one. A
  signal handler that flushed the table is a separate change with its own
  re-entrancy argument, and it is not this one.
