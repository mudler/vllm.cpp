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

**Not gateable, because nothing has run the model.** Thirteen tracked scripts
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
artifact. There is no `tools/oracle/` LTX script and no `tests/parity/goldens/ltx*`
manifest. [#1864](https://github.com/mudler/vllm.cpp/issues/1864) owes the
measurement, and `.agents/specs/oracle-ltx-2-pin.md` §"Owed" lists it.

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

**`gateable` stays `no`, and the remainder is now exactly one thing.** No weight
has yet been loaded by upstream code in this tree and no reference frame exists,
so the finding this file opens with is unchanged: every LTX-2.5 golden still
comes from upstream modules on synthetic weights.
[#1864](https://github.com/mudler/vllm.cpp/issues/1864) stays open and owes the
render itself.

```oracle-pin
id = ltx-2
role = secondary
upstream = https://github.com/Lightricks/LTX-2
scope = the LTX-2.5 architecture and pipeline recipes from the model author's own runtime, for the generations and defaults vLLM-Omni's ltx2 registration does not reach
pin = fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
pin_label = ltx-core / ltx-pipelines 1.2.0 (main, merge of LTX-2#273)
pinned_on = 2026-08-24
gateable = no
evidence = #1864
```
