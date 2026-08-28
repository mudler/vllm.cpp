# Oracle: `ggml-org/llama.cpp` PR #27752, the only llama.cpp that knows `glm5next`

A scoped, third llama.cpp record, admitted for one narrow reason. No llama.cpp
RELEASE converts or loads the `glm5next` architecture, and
`MODEL-MM-GLM53-FLASH` needs a llama.cpp denominator on the identical GGUF
artifact. It does not replace the [`llama-cpp`](llama-cpp.md) oracle, it does not
outrank vLLM, and it is never a mirror source.

## Why the `llama-cpp` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, so one file holds one revision. That is not an accident of the checker.
The `llama-cpp` pin is deliberately **stock upstream release** `b10451`, because
the floor that oracle supplies is *"the CPU and GGUF k-quant speed and memory
numbers a user can actually get today"*. An open pull request is not something a
user can get today. Folding a PR head into that record would quietly change what
the floor means for every measurement already taken against it.

Three llama.cpp records therefore say three different true things. `llama-cpp`
says what a release does. [`llama-cpp-qwen4exp`](llama-cpp-qwen4exp.md) says what
one unmerged branch does for `qwen4exp`. This file says what one unmerged branch
does for `glm5next`, and nothing else.

## When a scoped PR-oracle is warranted, and when it is not

This record and the `glm_moe_dsa` row landed on the same day and reached opposite
conclusions, which is the cheapest available lesson on the shape of this rule.

- **Warranted here.** `git grep -il 'glm5next\|glm5_next' b10451` is **rc=1**,
  nothing tree-wide. The released tag this project pins as its llama.cpp floor
  cannot name the architecture, so the floor oracle has nothing to say about this
  model and a second scoped record is the only honest way to have a llama.cpp
  denominator for it.
- **NOT warranted for `glm_moe_dsa`.** For GLM-5.3, the non-Flash model,
  [#2194](https://github.com/mudler/vllm.cpp/issues/2194) measured that stock
  `b10451` already carries the architecture: `LLM_ARCH_GLM_DSA -> "glm-dsa"` at
  `src/llama-arch.cpp:85` with its case at `:1051` and its enumerator at
  `src/llama-arch.h:90`, its graph at `src/models/glm-dsa.cpp`, and its converter
  at `conversion/glm.py:274-276`. Re-measured in the same bare clone that
  produced the table below: that grep returns rc=0 with those three
  `src/llama-arch.*` lines. So `MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm`
  needed **no scoped file at all**, and its `gateable = no` is a MEMORY blocker
  on an architecture llama.cpp already supports.

The test is therefore not "is this model exotic" and not "is this row blocked".
It is one question with a command behind it: **does the pinned release name the
architecture?** When it does, use `llama-cpp` and record any blocker as a
blocker. When it does not, and only then, a scoped PR pin is admissible.

## Scope, and what this oracle may not do

Use it ONLY as a llama.cpp denominator and reference for the `glm5next`
architecture: the GGUF conversion in `conversion/glm5next.py`, the graph in
`src/models/glm5next.cpp`, the architecture and hyper-parameter registration in
`src/llama-arch.{h,cpp}`, the hybrid index memory in
`src/llama-memory-hybrid-idx.{h,cpp}`, and the CPU and GGUF k-quant speed and
memory numbers those produce on a `glm5next` checkpoint.

It covers the TEXT backbone and nothing else. The vision tower is not in this
revision at all, and §"W6 has no vision denominator" below records that debt
rather than letting a reader assume the scope stretches.

For every other path, including CPU and GGUF k-quant floors generally, the oracle
is [`llama-cpp`](llama-cpp.md) at its own stock pin. Where vLLM or vLLM-Omni
implements the behavior, that is the reference and this is not, exactly as
`AGENTS.md` section "When vLLM has no implementation" requires. vLLM implements
`glm5next` at no revision, which
[#1998](https://github.com/mudler/vllm.cpp/issues/1998) measured and
[#2067](https://github.com/mudler/vllm.cpp/issues/2067) re-confirmed.

## The pin, verified rather than relayed

Measured on 28 August 2026 from refs and objects, never from a working tree.
Every command below ran against a fresh bare repository in a scratch directory
whose only remote is `https://github.com/ggml-org/llama.cpp`
(`git config --get remote.origin.url` returns exactly that).

| Claim | Command | Result |
|---|---|---|
| the PR's live head | `git ls-remote origin 'refs/pull/27752/*'` | `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc` at `refs/pull/27752/head` |
| the forge agrees | `gh api repos/ggml-org/llama.cpp/pulls/27752 --jq .head.sha` | the same object, `open`, `draft: false`, `merged: false` |
| the pinned object is servable | `git fetch --depth 1 origin 8a8d0bcc...` | rc=0, then `cat-file -t` is `commit` |
| the PR is unmerged | `git merge-base --is-ancestor 8a8d0bcc... refs/heads/master` | **rc=1** |
| that ancestry test is not itself broken | the same command for `b10451`'s commit | rc=0 |
| no released llama.cpp has it | `git grep -il 'glm5next\|glm5_next' b10451` | **rc=1**, nothing tree-wide |
| that grep is not itself broken | the same grep for `glm4_moe` at `b10451` | rc=0, nine files |
| `master` does not have it either | the same grep at `refs/heads/master` | **rc=1** |
| the converter is present at the pin | `git cat-file -s 8a8d0bcc...:conversion/glm5next.py` | 4714 bytes |
| the graph is present at the pin | `git cat-file -s 8a8d0bcc...:src/models/glm5next.cpp` | 55716 bytes |
| the hybrid index memory is present | `git cat-file -s 8a8d0bcc...:src/llama-memory-hybrid-idx.{h,cpp}` | 7034 and 22502 bytes |
| the architecture registration is present | `git cat-file -s 8a8d0bcc...:src/llama-arch.{h,cpp}` | 23367 and 81843 bytes |
| that size probe is not itself broken | the same probe for `conversion/no-such-file.py` | rc=128, "path does not exist" |

`refs/heads/master` was `50f068ffffc3e0e4c9c2e4139281c6075224f429` when those ran,
and `b10451` resolved to `10bf611e533d81f739128304991c5e133c6aebd8`, which is the
`llama-cpp` pin.

The pin is a 40-character object id and not the string `#27752`, for the reason
[`llama-cpp-qwen4exp.md`](llama-cpp-qwen4exp.md) measured on its own PR:
[#2060](https://github.com/mudler/vllm.cpp/issues/2060) named a head that had
stopped being the head almost four hours before the issue was written, and every
measurement taken against the NAME inherits that error with no signal.
[#2178](https://github.com/mudler/vllm.cpp/issues/2178) named the same two heads
this record measured, and this time they had not moved — which is a re-measured
agreement, not a transcription.

**Quote the pin with `${SHA}:path`, quoted.** In `zsh` an unquoted
`$SHA:conversion/...` loses the `:c` to the `:c` history modifier and the command
reports a mangled object name that reads like a missing file. That happened once
while this table was being measured; the quoted form is what the table records.

## Two competing implementations, and why this one

#27752 and [#27773](https://github.com/ggml-org/llama.cpp/pull/27773) are not the
text half and the vision half of one stack. They are separate authors, separate
branches, and **two competing implementations of the same model that disagree on
the GGUF architecture string.** Measured at the two heads:

| | #27752 (`eauchs`) | #27773 (`timkhronos`) |
|---|---|---|
| state on 2026-08-28 | open, **not** a draft, +2267/-46 over 16 files | open, **DRAFT**, +1694/-197 over 28 files |
| head | `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc` | `9370c82dbd1774941f9d8a05c9eafdac1ecb2e2c` |
| architecture string | `LLM_ARCH_GLM5NEXT -> "glm5next"` (`src/llama-arch.cpp:87`) | `LLM_ARCH_GLM5_NEXT -> "glm5-next"` (`src/llama-arch.cpp:152`) |
| text graph | `src/models/glm5next.cpp`, 55716 bytes | `src/models/glm5-next.cpp`, 35072 bytes |
| converter | `conversion/glm5next.py`, 4714 bytes | `conversion/glm.py:407-409` for text, AND `conversion/qwen3vl.py:254-260` for vision, both `@ModelBase.register("Glm5NextForConditionalGeneration")` |
| vision | none: `git grep -il glm5 <head> -- tools/` is **rc=1** | `PROJECTOR_TYPE_GLM5V -> "glm5v"` (`tools/mtmd/clip-impl.h:551`), `tools/mtmd/models/glm4v.cpp`, `mtmd_image_preprocessor_glm5v` |

That the vision row's rc=1 is absence and not a broken invocation is proved by
the same command at the other head, which returns rc=0 and six files.

**The published artifact settles the choice, and it is a measurement, not a
preference.** The one GGUF of this model that exists,
`unsloth/GLM-5.3-Flash-GGUF` at revision `d425e572fb9686125831f476129e51cea34bc5b4`,
declares `general.architecture = glm5next` in its first shard — read directly out
of the staged file's GGUF header, 72 KV pairs, version 3. That is #27752's
spelling exactly, and it is also what this project's own converter emits
(`scripts/convert-glm5-next-gguf.py`, [#2011](https://github.com/mudler/vllm.cpp/issues/2011),
wired to `kGgufArchArms` by [#2067](https://github.com/mudler/vllm.cpp/issues/2067)).

So pinning #27773 would produce a denominator that **refuses both artifacts by
name**: the published one and ours. A denominator that cannot open the file under
test is not a denominator. Pinning #27752 keeps the oracle, the published
artifact and our own converter on one architecture string.

**One file, not two, and the reason is that two would both claim the text
backbone.** The recommendation in #2178 assumed #27773 was the vision half; it is
not. A second `.agents/oracles/llama-cpp-glm5next-vision.md` would register a
second oracle whose scope necessarily includes a *competing* `glm5next` text
graph at an incompatible spelling, and the registry would then admit two llama.cpp
references for one architecture with no rule saying which wins. The debt below is
a smaller lie than that record would be.

## W6 has no vision denominator against the PUBLISHED mmproj

**Scope this claim carefully, because an earlier draft of this file overstated
it.** What is true is that #27773 cannot open the *published* `mmproj-BF16.gguf`:
that file declares `clip.projector_type = glm5next`, read out of the staged
bytes, and neither head's projector table defines that string. What is NOT true
is that #27773 could never supply a vision denominator at all. Measured at
`9370c82dbd1774941f9d8a05c9eafdac1ecb2e2c`:

- `conversion/qwen3vl.py:254-260` carries
  `@ModelBase.register("Glm5NextForConditionalGeneration")` on a
  `Glm5NextVisionModel(Glm4VVisionModel)` whose `projector_type` is
  `gguf.VisionProjectorType.GLM5V`;
- `gguf-py/gguf/constants.py:5723` defines `GLM5V = "glm5v"`;
- `tools/mtmd/clip-impl.h:551` accepts `{ PROJECTOR_TYPE_GLM5V, "glm5v" }`.

That is a closed, self-consistent vision path from the `zai-org/GLM-5.3-Flash`
safetensors to a tower that head can load. So the honest statement is that a
vision denominator is obtainable from #27773 by CONVERTING the checkpoint
ourselves, and is not obtainable by pointing that head at the published mmproj.
Whoever pays this debt should not exclude #27773 on the strength of the earlier
absolute.

Recorded as owed rather than waived, under **O4** in
[`../specs/glm5-next-flash.md`](../specs/glm5-next-flash.md).

The reason is stronger than "the vision PR is a draft", and the draft flag is not
what blocks it. The staged `mmproj-BF16.gguf` declares:

```
general.architecture  = clip
general.type          = mmproj
clip.projector_type   = glm5next
```

read from its header, 26 KV pairs, 348 tensors. Neither head defines that
projector string: #27773's table has `glm4v` and `glm5v`
(`tools/mtmd/clip-impl.h:550-551`) and #27752 has `glm4v` and no glm5 vision at
all (`:548`); `git grep -c '"glm5next"' <head> -- tools/` is **rc=1 at both**.

So on 2026-08-28 there is no revision of llama.cpp, released or proposed, that
can load the published mmproj of this model. Advancing #27773 out of draft does
not on its own discharge O4. What discharges it is a head whose projector table
accepts the string the published artifact carries, or a re-quantized mmproj that
carries a string llama.cpp accepts. Whoever pays this debt must re-measure both
sides before assuming the mismatch has gone away.

## Gateability

`gateable = no`, and [#2178](https://github.com/mudler/vllm.cpp/issues/2178) owes
the measurement.

`AGENTS.md` admits `gateable = yes` only once an oracle demonstrably BUILDS and
RUNS the model. **Neither half is measured here.** This change is records and
verification only: nothing was compiled and nothing was loaded. A build is not a
run, and a pin is neither.

**What has changed is that the run half is REACHABLE for the first time.** When
the row's spec was written no GGUF of this model existed anywhere — all four
repositories named `*-GGUF` held zero `.gguf` files. That is no longer true.
Under a developer grant recorded on 2026-08-28,
`unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL is being staged to
`/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/`. Sizes from the HuggingFace
revision listing at `d425e572fb9686125831f476129e51cea34bc5b4`, staging state
`stat`-ed at 2026-08-28T21:06:23Z:

| File | Upstream size | Staged |
|---|---|---|
| `...-00001-of-00004.gguf` | 9,429,859 B | **complete**, byte-for-byte the upstream size; it is a metadata shard, `n_tensors = 0`, `n_kv = 72` |
| `...-00002-of-00004.gguf` | 49,294,975,936 B (45.910 GiB) | 35,593,197,112 B, **72.2%, still growing** |
| `...-00003-of-00004.gguf` | 49,949,266,048 B (46.519 GiB) | **absent** |
| `...-00004-of-00004.gguf` | 9,466,399,584 B (8.816 GiB) | **absent** |
| `mmproj-BF16.gguf` | 1,164,010,080 B (1.084 GiB) | **complete and verified**, sha256 `513c9bfc55898998186543caefc01626fb28e378b92f391018e1c3dd6655b113` computed locally over the staged bytes |

The four model shards total 108,720,071,427 bytes, **101.2535 GiB**. An
earlier revision of this file said 108,729,501,347 bytes / 101.262 GiB. That
figure was 9,429,920 bytes high because it was summed from a naive substring
match on `UD-Q2_K_XL`, which returns FIVE entries at this revision: the four
shards plus `Shard_Rewrite/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf_file`,
a 9,429,920-byte sibling that is not a shard and is not staged. Match on the
`UD-Q2_K_XL/` prefix and the `.gguf` suffix, not on the substring. 33.16 GiB of
them are on the share. **The download is NOT complete, and no run may be claimed
against a partial file** — loading a truncated shard measures a truncated file,
not an oracle, which is the same trap
[`llama-cpp-qwen4exp.md`](llama-cpp-qwen4exp.md) recorded when its own artifact
was mid-flight.

The flag says `no`, and it keeps saying `no` until somebody records a build at
this object and a generation from a complete artifact.

## The fidelity facts a comparison against this oracle must carry

Two, both measured above rather than assumed.

1. **This oracle is text-only.** It has no vision tower, so an end-to-end
   comparison on a multimodal prompt is not matched work. State which side ran the
   vision path, exactly as [`llama-cpp.md`](llama-cpp.md) already requires for the
   `blk.64` tensors its own pin silently ignores. Otherwise the ratio measures a
   configuration difference and reads as a performance one.
2. **The artifact is `UD-Q2_K_XL`, an unsloth dynamic mixed-precision quant, not
   a uniform Q2_K.** Our own W7a arm is a uniform Q2_K of 100.35 GiB by the
   converter's own type resolver. The two are close in size and are NOT the same
   bytes. A quant-matched claim needs the same file on both sides, or it needs the
   difference stated.

## Pin

```oracle-pin
id = llama-cpp-glm5next
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the glm5next TEXT architecture, its GGUF conversion, its graph, and the CPU and GGUF k-quant floors on a glm5next checkpoint, which no released llama.cpp defines; the vision tower is excluded and owed
pin = 8a8d0bcc4d5fdf024c457526245bec4bc3a12adc
pin_label = pr-27752
pinned_on = 2026-08-28
gateable = no
evidence = #2178
```
