# Send multimodal input

Use the OpenAI-compatible server for image, video, and audio input.

The OpenAI API accepts multimodal input. `vllm-cli` accepts only text:
`--model --prompt --max-tokens --temperature --top-k --top-p --seed --stream
--speculative-config --tokenizer-config`.

Start the server with a multimodal model, then send content parts on
`/v1/chat/completions`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.chat.completions.create(model="Qwen3.6-27B", messages=[{"role": "user", "content": [
    {"type": "text",      "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
]}])
```

Accepted part types (`src/vllm/entrypoints/openai/chat_mm.cpp`):

| part type | modality |
|---|---|
| `image_url` | image |
| `video_url` | video |
| `input_audio` / `audio_url` | audio |

## Add a `clip` multimodal projector to GGUF

A GGUF multimodal model has two files: the language `.gguf` and a
`clip`-architecture `mmproj-*.gguf` carrying the vision tower. Name the second
one with `--mmproj` (`vllm-server`) or `vllm_model_params.mmproj_path` (C ABI
v22); it is never auto-discovered from a sibling filename, because a directory
holding two unrelated models must not silently fuse them.

```console
./build/examples/vllm-server \
  --model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf
```

The loader opens the projector and reads its `clip.*` metadata and `v.*` and
`mm.*` tensors. The engine then holds the same vision tower that the
safetensors path builds. No forward pass consumes this tower yet. Neither the
server nor the C ABI has a multimodal GGUF request path, so the flag validates
and loads the tower but does not produce an image answer
([#821](https://github.com/mudler/vllm.cpp/issues/821)).

The loader refuses four conditions by name before it reads the tokenizer or
any language-model weight bytes:

- `--model` is not a `.gguf`. A safetensors checkpoint carries its tower in its
  own shards and needs no projector file.
- the file's `general.architecture` is not `clip` (this is what you get for
  passing the language file twice).
- its `clip.projector_type` is not `qwen3vl_merger`. A `muse-glimmer` projector
  is routed to MuseGlimmer's own recorded refusal instead, which names the
  missing axis.
- it carries `v.patch_embd.weight` without `v.patch_embd.weight.1`. llama.cpp
  writes the temporal patch embedding as two halves; with one of them absent,
  loading would mean inventing the other, and the result would be a fluent,
  wrong model rather than an error.


The [checkpoint registry](../USAGE.md#checkpoint-registry) owns the exact file
sizes, repository revision, and checksums. The
[Qwen3.8 27B model recipe](../models/qwen3-8-27b.md) owns current arm support and
limitations. Loader and gate evidence remains in
[the Qwen3.8 quantized-arm spec](../../.agents/specs/qwen38-27b-quant-arms.md).

## Per-prompt input limits

vLLM caps how many items of each modality one prompt may carry
(`--limit-mm-per-prompt`), and `--language-model-only` is sugar for setting every
one of those limits to 0. Both flags are accepted (#607, waves L1+L2) and both
are enforced **on this server's chat path**, which is the one place that installs
the multimodal chat seam the check runs behind.

Both are also C ABI fields (`vllm_model_params.language_model_only` /
`.limit_mm_per_prompt`, ABI v19), and there they configure the engine, including
a server built on it, but they do not change what a `vllm_chat` call returns:
the C ABI has no multimodal request path yet, so an `image_url` content part sent
through it is dropped and answered as text. The refusals below are the server's.
`vllm_model_params.mmproj_path` (ABI v22) is in the same position: it loads and
validates the projector, and no C-ABI call can feed the tower an image yet.

The flag sets all modality limits to zero. It does not skip the encoder. The
server refuses multimodal requests.

```console
$ curl -s localhost:8000/v1/chat/completions -d '{... three image_url parts ...}'
{"error":{"type":"BadRequestError",
          "message":"At most 1 image(s) may be provided in one prompt."}}   # HTTP 400

$ vllm-server --model … --language-model-only     # then any image request:
{"error":{"type":"BadRequestError",
          "message":"At most 0 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit."}}
```

Two things follow from how the limit is computed
(`min(user limit, what the model/seam supports)`):

- A user limit can only **lower** the ceiling. `--limit-mm-per-prompt
  '{"image": 99}'` on this server still refuses a second image, because the
  OpenAI chat seam handles exactly one image today (video and audio parts are
  not routed at all, so their limit is 0 and they are refused by name rather
  than dropped, this is what closed
  [#686](https://github.com/mudler/vllm.cpp/issues/686)).
- The ``Set `--limit-mm-per-prompt` to increase this limit.`` hint appears only
  when raising the limit would actually help, that is, when the seam could take
  the items and the configuration is what refused them. Its absence is currently
  the only way to tell an unimplemented arm from a configured limit; the
  refusal message itself does not say which
  ([#758](https://github.com/mudler/vllm.cpp/issues/758)).

**What the zero limits now free.** A tower whose every modality is at limit 0 is
constructed but never loaded: its geometry is still parsed from `vision_config`,
so a refusal can still name what is missing, and its checkpoint tensors are never
read. This mirrors vLLM's `_mark_tower_model`
(`vllm/model_executor/models/interfaces.py:288-293`), and it follows from the
LIMITS rather than from the flag: `--limit-mm-per-prompt '{"image":0,"video":0}'`
skips the same tower, and one non-zero modality keeps it.

Three production tower loads exist and all three are gated: the two
architectures that read a tower out of their own checkpoint,
`MuseGlimmerForConditionalGeneration` and `Qwen3VLForConditionalGeneration`, and
the `--mmproj` projector, which is the Qwen3-VL tower read out of a second
`clip` GGUF beside a `.gguf` language file. On the `--mmproj` path the file is
still opened and still partly validated at zero limits, and only its tensors go
unread.

Which validation survives the zero limits is worth being exact about, because
the two classes behave differently:

- **Refused whatever the limits are** — the checks that run *before* the read:
  a projector whose architecture or `clip.projector_type` this build cannot use
  (`RefuseUnsupportedClipMmproj`), and one carrying tensors the reader would not
  consume (`RefuseUnaccountedClipMmproj`). Both sit above the skip, alongside
  `ClipMmprojVisionConfig`, so the geometry still resolves and the file is still
  named in the error.
- **NOT reached at zero limits** — the checks that live *inside*
  `LoadQwen3VLVisionFromClipMmproj`, which is the call the skip removes. A
  projector missing a tensor the tower needs, such as the temporal half
  `v.patch_embd.weight.1`, is refused at default limits and walks straight past
  the loader with `--language-model-only`. That is the construct half of
  construct-without-initialise doing what it says: what stops is the storage,
  and the reader's own missing-tensor refusals stop with it.

The server prints one line naming what was skipped, read back off the loaded
model rather than off the flag:

```console
$ vllm-server --model /path/to/muse-glimmer-30b --language-model-only
server: multimodal limits language-model-only=ON audio=0 image=0 video=0
server: multimodal towers NOT loaded (every modality they serve is at limit 0): vision_tower
```

Nothing is printed when nothing was skipped, so a text model and a multimodal
model at their default limits both look exactly as they did before.

**How many bytes that saves, measured on one model.** On
**Qwen3-VL-4B-Instruct** the flag freed **1.542 GiB of host RSS at load**
(1,655,791,616 bytes: peak 10,209,501,184 B without it against 8,553,709,568 B
with it), and the swapped repeat agreed to within 200,704 B. Measured 2026-08-24
on `thor:gpu0` under an `rc` lease, `--device cpu`, at `41ab550b9`, against a
threshold of 1,495,251,763 B that was declared before the run.

Three things that figure is not, all of which matter before you quote it:

- **It is one model's tower, not a general saving.** How much a skip frees is
  how big that model's tower is, and nothing else. `muse-glimmer-30b`'s tower is
  4.6x larger and is still unmeasured, so the number above says nothing about
  it.
- **About half of it is a defect of ours.** Qwen3-VL's tower is 0.774 GiB on
  disk in bf16, and our loader widens it to host f32
  ([#1359](https://github.com/mudler/vllm.cpp/issues/1359)). When that is fixed
  this saving should roughly halve, and the smaller number will be the honest
  one.
- **It is load-time residency, and it is host RAM.** The measured window ends at
  server readiness, and the build was CPU-only, so this is not a steady-state
  serving figure and not a VRAM claim.

The procedure and its pre-declared thresholds are `scripts/mm/tower_skip_rss.sh`
and `.agents/specs/multimodal-track.md` §1.5 L3 — one threshold per model kind,
`muse-glimmer` and `qwen3-vl`, each derived from that checkpoint's own
safetensors headers, because one model's tower size does not describe another's.
`muse-glimmer` has not run: it needs about 56 G staged to local disk on a leased
device ([#607](https://github.com/mudler/vllm.cpp/issues/607),
[#1358](https://github.com/mudler/vllm.cpp/issues/1358)). See
[Memory benchmarks](../benchmarks/memory.md).

