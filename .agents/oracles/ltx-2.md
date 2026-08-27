# Lightricks `LTX-2` — the LTX-2.5 model author's own runtime

A **third** repository, neither `vllm-project/vllm` nor `vllm-project/vllm-omni`,
and the one every `file:line` anchor in the LTX-2.5 lane actually resolves in.
Two packages carry it: `packages/ltx-core/src/ltx_core/` is the architecture —
the joint video+audio DiT, the Conv video VAE, the audio VAE and vocoder, the
Gemma text-encoder plumbing, the guiders and schedulers — and
`packages/ltx-pipelines/src/ltx_pipelines/` is the recipe layer: sigma schedules,
samplers, generation-version parameter inheritance and the shipped defaults.

**Prefer vLLM-Omni wherever it implements the pipeline.** It registers `ltx2`
(`vllm_omni/diffusion/registry.py:69-87` at `a4ea67a2`) and its structure is the
one this project mirrors. Reach here where it does not: its `_PIPELINE_RECIPES`
stop at generation 2.3 (`vllm_omni/diffusion/models/ltx2/ltx2_recipes.py:161-166`),
2.5 is open upstream at
[vllm-omni#6066](https://github.com/vllm-project/vllm-omni/issues/6066), and the
2.4/2.5 rows this port adds take their values from here. Record which of the two a
stage was gated against, because they do not always agree — the default negative
prompt differs, Lightricks' carrying five leading tags vLLM-Omni's lacks, and
this port keeps both strings and gates the disagreement as a value
(`kLtx2NegativePromptsAgree`) rather than resolving it by preference.

`diffusers` overlaps here too, and the tie-break is written down rather than left
to whoever reaches for one first. [#1012](https://github.com/mudler/vllm.cpp/issues/1012)
records that `diffusers` at its recorded pin `c6da9936` implements LTX-2.5 in both
decode arms, and that record is `gateable = yes` while this one is not. Reach for
`diffusers` for a *diffusers-shaped* question — a scheduler or VAE as that library
composes it. Reach here for what the model author's own runtime defines, which is
what every LTX-2.5 correctness gate in this tree already runs against. Where they
disagree, this file is the architecture reference and `diffusers` is not; where a
stage was gated, name which one it was gated against.

vLLM proper registers nothing LTX at the parity pin `5559679229`. Searched in
that checkout at that revision: `git grep -inE "\bltx" -- '*.py' '*.md' '*.yaml'
'*.yml' '*.txt' '*.json'` returns no line, `git grep -ilE "lightricks"` returns
no file, and a whole-tree case-insensitive `ltx` search matches eight paths:
seven PNG/SVG documentation assets and the vendored minified
`vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js`, which git
treats as text. All eight are incidental byte matches rather than
registrations. `vllm/model_executor/models/registry.py` exists at that
revision and is inside the searched set. So this is a secondary oracle under the
rule that admits one, not a preference.

**Identity, asserted rather than assumed.** The revision below was read from a
clone whose `origin` is `https://github.com/Lightricks/LTX-2.git`, whose worktree
is clean, and whose `HEAD` is `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca` — the
merge of `Lightricks/LTX-2#273`, authored 2026-08-11. The repository carries **no
tags**, so `pin_label` uses the package version both `ltx-core` and
`ltx-pipelines` declare at that revision. Nothing here is a claim about what
upstream `main` holds today: this record, like the checker that reads it, is
network-free.

The pin is not new to the tree, only to the registry. `git grep -l fd4ded7f`,
measured at this head rather than quoted, returns **30** files matching
`.agents/specs/ltx25-*.md`, **5** under `scripts/` — four `gen-ltx2-*.py`
generators and `probe_ltx2_tiling_layout.py` — and **15** under `src/`, twelve of
them named `ltx2_*`. Those are three named subsets, not a partition: **97** files
in the tree carry the revision. `.agents/porting-inventory.md` (5 hits) and
`.agents/kernel-matrix.md` (1) carry it as well, from `.agents/` — one directory
above the `.agents/specs/` glob, and outside all three.

Six C++ suites carry the 40-hex literal and FIVE of them assert it:
`test_ltx2_tiling.cpp:88,392-393`,
`test_ltx2_pipeline.cpp:63,199`, `test_ltx2_dfr.cpp:105-106`,
`test_ltx2_vae.cpp:1757,1767` and `test_ltx2_image_cond.cpp:79,343-344`
each compare a revision constant and fail when it is any other revision. The
sixth,
`test_ltx2_lora.cpp:9`, carries the literal in a prose header comment only: it
includes no goldens `.inc`, declares no revision constant, and none of its
`CHECK`s touches one, so it cannot fail on a revision advance. Carrying the
string and asserting it are different things, and a `git grep -l` file list
cannot tell them apart.

**None of those five reaches the `pin` field below, and nothing else does
either.** Each compares a constant emitted into its own goldens against one
hardcoded in the test file; none reads `.agents/oracles/`, and all need a C++
build. Flipping one hex digit of `pin` here leaves `check-oracle-pins.py`,
`check-agent-record.py`, `check-gate-commands.py` and the 185 cases of
`test_check_oracle_pins.py`, `test_agent_record.py` and
`test_check_gate_commands.py` green, because the checker is deliberately
network-free and validates the shape of a 40-hex string rather than its value.

Python does carry the revision, and no gate reads it there either. The five
`scripts/` files above hold it, TWO as live module constants rather than prose —
`gen-ltx2-tiling-goldens.py:59` (`_PIN`, the full 40 hex) and
`gen-ltx2-res2s-goldens.py:60` (`PIN`, the 7-char prefix) — and no gate
executes any of the five: neither `scripts/agent-preflight.sh`, nor anything
under `tests/scripts/`, nor any workflow under `.github/workflows/` names
`gen-ltx2-*` or `probe_ltx2_*`. They are generators a human runs by hand.

The defence is the identity block above — a named clone, a clean
worktree, a stated `HEAD` — and re-derivation by a reader, not a gate.

**Not gateable until 2026-08-27, because nothing had run the model.** The
paragraphs that follow are the measurement that kept `gateable = no` for three
days, and every sentence in them is still true of the thirteen scripts they
describe. What changed is at the end of this file: upstream now renders on real
weights, from a committed script, and `gateable` is `yes`. Thirteen tracked scripts
import and execute upstream code, and every one of them runs individual modules
at reduced dimensions on synthetic PRNG weights, or reads constants and
safetensors headers — the generated goldens say so in their own headers
(`tests/vllm/models/ltx2_vae_goldens.inc:3-6`: "Weights and inputs come from the
shared deterministic stream, so no weight byte is checked in").
`ltx2_pipeline_goldens.inc` repeats that sentence verbatim.
`ltx2_goldens.inc` and `ltx2_text_goldens.inc` carry the operative clause inside
a different sentence — "the MATH is gated exactly here and no weight byte is
checked in" — and `ltx2_tiling_goldens.inc:6-8` and
`tests/vllm/multimodal/ltx2_image_cond_goldens.inc:4-6` state it in their own
words. All six say no weight byte is checked in; only two say it in the same
words, and other LTX-2 artifacts carry the clause as well, so these six are a
sample rather than the population.

Of those thirteen, the two that touch real checkpoint bytes run one
`AdaLayerNormSingle` module
(`scripts/measure-ltx2-prompt-adaln.py:126-140`, the forward itself at :140)
and a meta-device loader probe
(`scripts/measure-ltx2-keyframes-meta.py:156,203`); neither writes a committed
artifact. There was no `tools/oracle/` LTX script and no
`tests/parity/goldens/ltx*` manifest either.
[#1864](https://github.com/mudler/vllm.cpp/issues/1864) owed the measurement, and
`.agents/specs/oracle-ltx-2-pin.md` §"Owed" listed it. Both gaps closed on
2026-08-27: `tools/oracle/ltx2_oracle.py` and
`tests/parity/goldens/ltx2_oracle/` exist, and the section at the end of this
file is what they record.

One consequence worth stating, because it is the reason the run matters rather
than a formality: where the revision is asserted no weight is loaded, and where
weights are loaded only the interpreter *path* is asserted, never the revision.

**A lease was taken on 2026-08-25, and the lease is NOT what blocks the
measurement.** `dgx:gpu0` was held for this work (`rc` jobs
`60ea39f4-386b-4b83-bb76-30c2148ad35d` and
`674521fc-49ba-4944-90b6-2e1a6eccf58d`). That falsifies the reason
`.agents/specs/oracle-ltx-2-pin.md` gave — "W2 is unclaimed and needs a GPU
lease" — and it matters, because a wrong blocker sends the next reader to book
hardware that will sit idle. What follows is measured unless labelled otherwise,
and NO render was run: `gateable` stays `no`.

**The wall is one absent file, and it is not an access problem.** A real-weights
upstream render needs the bf16 Gemma-4 text tower,
`Lightricks/LTX-2.5` path `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
(upstream `README.md:77`, `packages/ltx-pipelines/docs/hdr.md:45`). It is
**26,263,858,182 bytes (24.46 GiB)**, sha256
`ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1`, both read
from the `x-linked-size` and `x-linked-etag` response headers. The repository is
gated — an anonymous `HEAD` on `/resolve/main/` returns **401** with
`x-error-code: GatedRepo` — and **this box has been granted access**: the same
request authenticated returns **302** to the CDN. So the file is reachable and
the download is a bandwidth and authority question, not a permissions one.
`.agents/developer-preferences.md` closes with "Any other large download,
package installation, or service management: ask first", and no download
authority is recorded for this row, so the measurement stays `PENDING` on that
and not on hardware.

**The four other slots `ti2vid_one_stage` needs are already local**, under
`${CHECKPOINT_ROOT}/ltx-2.5/`: the 42,018,190,584-byte dev bf16 transformer, the
video VAE, the audio VAE and the duration head. The text encoder is the only
empty slot.

**The Gemma we DO hold cannot stand in for it.** The local
`gemma4-12b-with-proj-nvfp4-torchao.safetensors` is a torchao-format artifact,
and upstream at this pin reads no such tensor: `torchao` appears **zero** times
in `packages/` and in `pyproject.toml`. Upstream's own NVFP4 implementation
(`packages/ltx-core/src/ltx_core/quantization/nvfp4/`) is unrelated to it.

**Nor can precomputed embeddings bypass the tower**, which was the one route
that would have removed the download. `TI2VidOneStagePipeline.__call__`
(`ti2vid_one_stage.py:132-134`) and `DistilledPipeline.__call__`
(`distilled.py:187-188`) both take `prompt: str` with no tensor override, and a
search for `prompt_embeds`, `encoder_hidden_states` and `text_embeds` across
`packages/ltx-pipelines/src/` returns no hit outside two comments. Exactly one
pipeline takes embeddings — `HDRICLoraPipeline` (`hdr_ic_lora.py:230`, loaded at
`:275-281`) — and it is the wrong one three times over: it is a different
pipeline, its shipped embeddings are LTX-**2.3** assets this tree does not hold,
and it consumes POST-connector context while our own `--prompt-embeds` files are
consumed PRE-connector (`src/vllm/multimodal/ltx2_video.cpp:1516-1578`). Feeding
it from our encoder would also make our text tower the source of the oracle's
conditioning, which is close to circular.
`--text-encoder-path` being argparse-optional (`utils/args.py:554-558` passes no
`required=`) is not a hole either: `PromptEncoder.__init__` calls
`model_paths.text_encoder()` (`utils/blocks.py:648`) and `ModelPaths._require`
raises (`utils/model_paths.py:114-119`).

**One more cost the next reader should not rediscover:** there is no `uv.lock`
in the upstream checkout, so `uv sync` must resolve from the network, and
`pyproject.toml` pins `nvidia-cudnn-cu13` and a `cu132` torch index. Per
`.agents/environment.md` the container side of a lease has egress while
`dgx.casa` itself does not, so that resolve belongs inside the lease. The
resolve and render wall-clock are UNMEASURED — nothing here ran either.
`.agents/specs/ltx-2-5.md` §7.0(b) records why that is not hypothetical — a decoy
`ltx_core` once produced byte-identical goldens and exited 0.

**2026-08-25, second attempt: the download landed and was verified, and the model
still did not run.** Both blockers recorded above are now cleared. `gateable`
stays `no` for a third reason that is neither of them.

**The artefact is here, and its hash is MEASURED rather than advertised.**
Large-download authority was granted for this one file
(`.agents/developer-preferences.md`, 2026-08-25) and the bf16 Gemma-4 tower was
fetched to the shared NAS at `/workspace/ckpt/ltx-2.5/` path
`text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, complete by
`2026-08-25T20:37Z`. Read back over CIFS from the coordinator in **3m49s**
(229 s, about 110 MiB/s):

```text
26263858182 bytes
ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1
```

Both agree with the `x-linked-size` and `x-linked-etag` values recorded above.
Those were the repository's claim about the file; these are this project's own
measurement of the bytes on disk. The distinction is the whole of
[#1723](https://github.com/mudler/vllm.cpp/issues/1723), where a checkpoint was
re-quantized in place under an unchanged name and every measurement on that lane
had loaded the earlier file. The four other slots `ti2vid_one_stage` needs were
already staged, so the checkpoint set is now complete for the first time.

**The run was attempted and produced nothing.** `rc` job
`378a892a-ae9b-4f14-a223-544704bf3a4d` took `dgx:gpu0` at `20:06:51Z` and had
executed no step of the work **2h37m** later, at `22:43Z`. That is a
lower bound and not a total: the job was still holding the device, still at zero
bytes, when this was written. Its log
`/workspace/ltx2-oracle/logs/run-20260825T200651.setup.log` is **zero bytes**
with an mtime of `20:06:51.911`, the instant `tee` created it and never written
since.

The cause is a shell bug in the job harness, not in upstream and not on the box.
`setup.sh:5` reads `HB=$(heartbeat setup)`, and `heartbeat` backgrounds a
subshell that inherits the command substitution's stdout pipe:

```sh
heartbeat() {
  ( while true; do echo "[hb $(date -Is)] $1"; sleep 30; done ) &
  echo $!
}
```

A command substitution closes when every writer on its pipe closes, and the
backgrounded loop never closes, so the assignment blocks forever. Reproduced in
isolation: those five lines under `timeout 20` exit **124**. The script never
reached its own line 7, which is why the log holds not even the heartbeat lines
that were added to prove liveness. The fix is one redirect, `( ... ) >&2 &`,
which takes the subshell off the capture pipe and leaves the heartbeat visible,
because `run.sh` merges each script's stderr into `tee` with `2>&1`. Verified in
isolation: past line 5 immediately, heartbeat still printing, process exits.

This tree's own long-running job scripts do not carry the defect —
`scripts/ltx25-dit-attn-flash-pixel-ab.sh:159-160` backgrounds the loop and takes
`HEARTBEAT=$!` directly, with no command substitution — so the bug entered with
the throwaway job harness rather than from anything committed here. Its neighbour
comment is the lesson stated from the other side: a line emitted on a fixed
cadence cannot distinguish work from a hang. Here the **absence** of that line
was the whole diagnosis, and it took upwards of 2h37m of a shared device to read.

**What this cost, and what stays UNMEASURED.** Measured: the 24.46 GiB fetch and
its verification, and at least 2h37m of a fleet lease spent on nothing. Still
unmeasured,
because the job never reached them: the upstream clone, the venv and torch
install, the NAS-to-local staging of the four checkpoints, model load, host-RAM
high-water and render wall-clock. None of those is estimated here.

**`gateable` stayed `no` that day, and the remainder was then exactly one
thing.** No weight had yet been loaded by upstream code in this tree and no
reference frame existed.
[#1864](https://github.com/mudler/vllm.cpp/issues/1864) stayed open and owed the
render itself. Three attempts followed, and the section below is what they cost
and what finally ran.

**2026-08-26, attempts three, four and five: three more defects, and not one of
them was in the model, the hardware or the checkpoint.** Each was found by
running, each cost part of a lease, and each is written down here because the
next reader's cheapest path is not to rediscover them.

*Attempt three* died ten seconds into the render, after the four checkpoints had
staged. `torchvision` was absent, and `transformers` imports it at module scope
in `models/gemma4_unified/image_processing_gemma4_unified.py:23`. The error the
log carried was not that: `utils/import_utils.py:2430` wraps any failure inside
a lazy module as `Could not import module 'Gemma4UnifiedProcessor'. Are this
object's requirements defined correctly?`, which names the CLASS and hides the
dependency. That message was first read as a missing `numpy`, by inferring the
module from the class name rather than from the traceback. It was not numpy.
**A wrapped import error names the wrapper, so read the traceback and never the
summary line.**

*Attempt four* installed `torchvision` and died on `torchvision` anyway. The
install had been added inside an `if ! python -c 'import torch'` block, and the
worker's `/tmp/ltx2` had survived from the previous job, so torch was present,
the branch never ran, and the line that would have fixed the defect was never
executed. **An install guarded by a DIFFERENT package's absence guarantees
nothing.** Worker `/tmp` is indeterminate between jobs in both directions here:
it persisted between attempts three and four and did not between two and three.
Every install in this harness is now unconditional, and `pip` and `apt` being
idempotent is what makes that cheap.

*Attempt five* got furthest and is the reason this file gained a toolchain
section. The processor imported, the four checkpoints staged, the 22 B
transformer and the 24.46 GiB Gemma tower loaded, and the render entered the
text tower and died **20 seconds in**, at
`modeling_gemma4_unified.py:278` — `inv_freq_expanded.float() @
position_ids_expanded.float()`, the RoPE outer product. `torch._native`'s
`eager_router` dispatches that `[B,D,1] @ [B,1,S]` shape to
`bmm_outer_product`'s **triton** implementation, triton's runtime JIT-builds a
CPython extension before it can launch anything, and `/usr/bin/gcc` exited 1:

```text
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpsgb2k21a/cuda_utils.c',
 '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o',
 '/tmp/tmpsgb2k21a/cuda_utils.cpython-312-aarch64-linux-gnu.so',
 '-l:libcuda.so.1', '-L.../triton/backends/nvidia/lib', '-L/lib/aarch64-linux-gnu',
 '-I.../triton/backends/nvidia/include', '-I/tmp/tmpsgb2k21a',
 '-I/usr/include/python3.12']' returned non-zero exit status 1.
```

**The compiler's reason was not in the log, and that is a property of the
caller.** `triton/runtime/build.py:48` is
`subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)` — no `stderr=` at
all — and `CalledProcessError` carries the argv and the return code and nothing
else. Five attempts had produced return codes; none had produced a diagnostic.

**2026-08-27: upstream RAN, on real weights, and this record is gateable.** `rc`
job `44159e4f-f810-4c16-a4ab-67a0b3019f0c` took `dgx:gpu0` at `00:14:27Z` and
released it at `00:19:58Z`. **Five minutes thirty-one seconds of device time**,
against the 2h37m the heartbeat bug burned on nothing, because everything the
render needed was proved in the setup step where a failure costs seconds.

The wall was CPython headers, and this is the compiler line five attempts had not
produced:

```text
/tmp/tmp6zhzohw0/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
    9 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
```

`apt-get install -y python3-dev` is the whole fix. Measured on the worker before
it: `gcc 13.3.0` present, `/usr/include/python3.12` **absent**, `python3-config`
**absent**. Measured after: `Python.h` present, the same `cuda_utils.c` compiles,
`bmm_outer_product` dispatches, and an ordinary Triton kernel compiles and
launches. **The link half was never the problem on this box** —
`-l:libcuda.so.1 -L/lib/aarch64-linux-gnu` linked with `rc=0` before any install,
because `dgx` keeps `libcuda.so.1` there. It does not link on the `thor:gpu0`
worker of the same Ubuntu 24.04 image. That is measured too, in `rc` job
`2006eb26-6737-4c2e-b9f5-7e7f84c18251` on 2026-08-26, which compiled the two
halves of the same command separately and read:

```text
/usr/bin/ld: cannot find -l:libcuda.so.1: No such file or directory
libcuda.so.1 (libc6,AArch64) => /opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1
```

`thor` is Tegra and keeps libcuda outside the search path Triton derives, so that
reading is one box's and not the image's. It is also where `Python.h` was first
found absent, about ninety minutes before `dgx` confirmed it on its own worker —
a lead taken on an idle device while `dgx` was held by another job, which is why
the fix and the render fitted in one lease instead of two.

**The route around it exists and was deliberately not taken.**
`torch._native` exposes `TORCH_DISABLE_NATIVE_JIT`, read out of the installed
package rather than guessed. Setting it would move the RoPE outer product onto a
different implementation, and a reference render whose kernels were chosen to
avoid provisioning a toolchain is not a reference. A `TRITON_LIBCUDA_PATH`
override is admissible on the same reasoning and was left unused here: it changes
where the LINKER looks and nothing about what computes.

**What ran, and what it produced.** `tools/oracle/ltx2_oracle.py` — the script
this row commits, not an inline command that exists only inside a job — asserted
`git rev-parse HEAD == fd4ded7f...` and that both `ltx_core` and `ltx_pipelines`
resolve inside that clone, **before opening a weight file**, then sha256'd all
four checkpoints, ran `ltx_pipelines.ti2vid_one_stage`, decoded the result and
wrote the manifest. Render 93.8 s; whole run 243.7 s, of which 149 s was hashing the four
checkpoints, 70,099,185,228 bytes (65.3 GiB).

```text
prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
geometry 320x192, 25 frames, 8 steps, seed 42, --offload cpu
device   NVIDIA GB10, capability (12,1), torch 2.13.0+cu130
output   upstream-render.mp4, 225,151 bytes; 25 PPM frames; audio.wav 1.02 s stereo 48 kHz
```

**The four checkpoint digests are this project's own, measured off the worker's
local disk.** None of them is new: all four were already on `main`, the two VAEs
in `docs/USAGE.md` and the transformer's at `docs/models/ltx-2-5.md:62` and
`.agents/specs/ltx25-bf16-dit.md:248`. What the run adds is a **second
independent derivation** of each, on another host from another copy, and all four
agree to 64 hex characters. The standing gap is unchanged and is not this run's
to close: `docs/models/ltx-2-5.md` records that `792a2bad...` "has never been
compared against the published artifact", and hashing our own copy twice does not
compare it to Lightricks'.

```text
792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584  22b-dev-transformer-bf16   42,018,190,584
ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1  gemma4-12b-with-proj-bf16  26,263,858,182
685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d  video-vae-conv-bf16         1,452,269,922
c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5  audio-vae-bf16                364,866,540
```

**Frames were counted and then looked at, because a count is not a picture.**
25 of 25 frame digests are distinct, the geometry is 320x192x3, pixels span the
full 0..255 with a mean of 121.70 over all 25 frames, 87.1% of bytes differ
between the first two frames and 98.2% between the first and the last, and the
audio peaks at 13010 of 32767 over 1.024 s.
A blank render, a frozen render and a silent track each fail at least one of
those. Measured on the coordinator from the NAS copies.

`tests/parity/goldens/ltx2_oracle/SHA256SUMS` records all 28 digests, and
`tests/scripts/test_ltx2_oracle_goldens.py` **recomputes** the two that are
committed. That distinction is the finding a fresh review produced: a digest
nothing recomputes is a comment, and `check-oracle-pins.py` asserts only that the
`evidence` PATH exists, so falsifying the manifest's contents and truncating the
mp4 to one byte both left it green. Both mutations now red on several assertions
each. The 26 digests of the uncommitted frames and audio stay a record for
whoever fetches them from the NAS, and the suite says so rather than skipping
them silently.

**What this does NOT establish.** One geometry, one prompt, one seed, one
pipeline (`ti2vid_one_stage`), one offload mode, bf16 only. No comparison against
this project's own render was made, and none is claimed:
[#1854](https://github.com/mudler/vllm.cpp/issues/1854) owns that, and it now has
an absolute reference to ask for. The thirteen synthetic-weight golden scripts are
unchanged, so every LTX-2.5 golden in the tree still comes from upstream modules
on PRNG weights — what changed is that the oracle those goldens name can now be
run against real ones.

```oracle-pin
id = ltx-2
role = secondary
upstream = https://github.com/Lightricks/LTX-2
scope = the LTX-2.5 architecture and pipeline recipes from the model author's own runtime, for the generations and defaults vLLM-Omni's ltx2 registration does not reach
pin = fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
pin_label = ltx-core / ltx-pipelines 1.2.0 (main, merge of LTX-2#273)
pinned_on = 2026-08-24
gateable = yes
evidence = tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json
```
