ID: ISSUE-LOCAL-01M29AV2VX9RBRF409F6YVX3MK
Title: `gguf_reader.cpp:194` sources the traits for ggml ids 39-41 from mudler's killgate llama.cpp fork and attributes ids 40 and 41 to it as fork appends (naming 39 as mainline's), and mainline has defined 40 and 41 for months: `GGML_TYPE_Q1_0 = 41` landed in stock ggml-org/llama.cpp by merged PR #21273 (2026-04-06) and is present at our own `llama-cpp` pin `b10451` (`ggml/include/ggml.h:431`), at the same 128 elems / 18 bytes the reader already uses, with `GGML_FTYPE_MOSTLY_Q1_0 = 27` and `LLAMA_FTYPE_MOSTLY_Q1_0 = 40`. The traits are therefore RIGHT for a stated reason that is now wrong, at `:194` and `:378-392`. Mainline at the same pin also defines `GGML_TYPE_Q2_0 = 42` (64 elems / 18 bytes, `ggml-common.h:187-192`), which `FindGgmlTraits` does not handle and falls through to `nullptr` -- a real gap now that files carrying ids past 39 are being published. Found while scoping DeepSeek-V4.1-Flash, whose published Q1_0 GGUF measures exactly 18/128 bytes per element and so confirms the geometry against a real file for the first time. Product code, so it takes its own change rather than riding a records commit
Row: QUANT-GGUF-Q1_0
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-12
Closed: -

## Problem

A provenance comment the tree falsifies, beside an unhandled mainline type.

### Id 40 is mainline too, and two OTHER records still say it is not

**The stale attribution is exactly two ids, 40 and 41, and it must not be
re-widened to three.** The comment's own words append "GGML_TYPE_NVFP4 = 40 and
GGML_TYPE_Q1_0 = 41 after mainline's GGML_TYPE_MXFP4 = 39", so it already names
39 as mainline's and only the geometry citation reaches across 39-41. An earlier
revision of `.agents/specs/deepseek-v4-1-flash.md`'s `## Owed` bullet said **two
ids**; that reading was the accurate one and a later revision widened it to
three in error. The title and this section are worded to keep the narrow
reading. This issue therefore owes id **40** as well as id 41, and owes nothing
about id 39 beyond the shared geometry sentence. Verified 2026-09-12 against the
pinned stock `llama-cpp` oracle `b10451`
= `10bf611e533d81f739128304991c5e133c6aebd8`, in a checkout whose `origin` is
`https://github.com/ggml-org/llama.cpp` and where
`git merge-base --is-ancestor 10bf611e origin/master` exits 0, so the tag is
stock upstream and not the fork:

- `git show b10451:ggml/include/ggml.h | grep -n 'GGML_TYPE_'` prints
  `429: GGML_TYPE_MXFP4 = 39`, **`430: GGML_TYPE_NVFP4 = 40, // NVFP4 (4 blocks,
  E4M3 scale)`**, `431: GGML_TYPE_Q1_0 = 41` and `432: GGML_TYPE_Q2_0 = 42`.
- `git show b10451:gguf-py/gguf/constants.py | grep -n 'NVFP4'` gives
  `GGMLQuantizationType.NVFP4 = 40` and, in `GGML_QUANT_SIZES`,
  **`NVFP4: (64, 4 + 32)`** = 64 elements in 36 bytes, which is byte-for-byte
  the `GgmlTypeTraits t{64, 36, "NVFP4"}` our reader already tabulates at
  `src/vllm/model_executor/model_loader/gguf_reader.cpp:384`. It also gives
  `LLAMA_FTYPE_MOSTLY_NVFP4 = 39`.

So the id, the name and the block geometry for type 40 are all mainline at our
own pin, exactly as they are for type 41. Two records outside this issue's
current fix still carry the falsified provenance, and **this issue owes both**;
they are named here rather than edited, because each belongs to its own row and
to a change that is not this one:

- `.agents/quantization-matrix.md:84`, the `QUANT-GGUF-NVFP4` row, whose
  description reads "40 / NVIDIA NVFP4 in a GGUF container (fork/toolchain type
  id; ...)".
- `.agents/specs/gguf-nvfp4-notes.md:20`, `## Upstream chain`, which reads "No
  mainline ggml type 40: it is a fork/toolchain extension", and `:47`, whose
  enum listing annotates `GGML_TYPE_NVFP4 = 40` and `GGML_TYPE_Q1_0 = 41` as
  `(FORK EXTENSION)` and states `GGML_TYPE_COUNT = 42` -- which the same read of
  `b10451` also falsifies, since mainline there defines `GGML_TYPE_Q2_0 = 42`.

None of those statements changes any NUMBER those records depend on: the traits,
the block layout and the dequant math were and remain correct. What is wrong in
all three places is only the stated REASON, which is why the repair is one
provenance correction and not a behavior change.

## Resolution

-
