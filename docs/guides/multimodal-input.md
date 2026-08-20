# Send multimodal input

Use the OpenAI-compatible server for image, video, and audio input.

Multimodal input is served over the **OpenAI API**, not the CLI. `vllm-cli` is text-only:
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

## The second GGUF file: a `clip` multimodal projector

A GGUF multimodal model ships as **two** files: the language `.gguf` and a
`clip`-architecture `mmproj-*.gguf` carrying the vision tower. Name the second
one with `--mmproj` (`vllm-server`) or `vllm_model_params.mmproj_path` (C ABI
v22); it is never auto-discovered from a sibling filename, because a directory
holding two unrelated models must not silently fuse them.

```console
./build/examples/vllm-server \
  --model /models/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf
```

What this does today, exactly: the projector is opened, its `clip.*` metadata
and its `v.*` / `mm.*` tensors are read into the same vision tower the
safetensors path builds, and the result is held on the engine. **No forward
consumes it yet**, there is no multimodal request path for a GGUF model on
either the server or the C ABI, so the flag buys validation and a loaded tower,
not an image answer ([#821](https://github.com/mudler/vllm.cpp/issues/821)).

Four things are refused **by name**, all of them before the tokenizer and before
any language-model weight byte is read:

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

### The exact files this was gated against

`--mmproj` was built and gated against the two files below. Both are
**third-party quantizations by Unsloth**, not first-party releases from the model
authors, and a repo id alone is not a pin, because a checkpoint gets re-quantized
in place under an unchanged name.

| Arm | Repo and revision | File | Bytes | sha256 |
|---|---|---|---|---|
| `clip` projector (`--mmproj`) | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `mmproj-BF16.gguf` | 931 146 432 | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| Q4_K_M language file (`--model`) | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` | `Qwen3.8-27B-Q4_K_M.gguf` | 17 106 775 008 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |

Both sha256 values were computed locally on this project's mirrored copy, not
read back from the hub.

The projector is GGUF v3 with 334 tensors (110 BF16 + 224 F32) and 35 metadata
keys, `general.architecture = clip`, `general.type = mmproj`,
`clip.projector_type = qwen3vl_merger`. Its tower is 27 blocks of hidden 1152,
16 heads, feed-forward 4304, patch 16, spatial merge 2, projected to 5120, with
2304 position embeddings and no DeepStack tap, its
`clip.vision.is_deepstack_layers` is 27 `false` values, and it ships no
`v.deepstack.*` tensor.

To re-run the mapping against those bytes rather than against the synthetic
fixture CI uses, name the file and run the reader's own gate:

```console
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_clip_mmproj_gguf
```

That gate reads the projector alone. To confirm the other half, that a load
which COMPLETES leaves the tower on the engine, reachable through
`LoadedEngine::vision_tower()`, name both files and run the loader's gate:

```console
VLLM_CPP_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_gguf_mmproj_reach
```

This one loads the whole 17 GB language file. Measured on an x86 CPU-only build
reading both files over CIFS: 5 min 37 s and 6 min 22 s in two runs, the wall
time is bound by the share, not by the build, at 33.06 GB peak resident both
times. Do not start it on a box with less than about 40 GB of available memory.

Unset, both cases skip loudly and the gates stay hermetic; CI never reads the
file.

#### The tensor accounting, in CI and on the bytes

Both files now have a **committed manifest**, their tensor names, ggml dims and
type ids and their scalar metadata, no weight bytes, generated by
`scripts/gen-qwen38-27b-gguf-manifest.py` and frozen at
`tests/vllm/models/qwen38_27b_q4km_gguf_manifest.inc` (866 tensors, 51 keys) and
`tests/vllm/models/qwen38_27b_mmproj_gguf_manifest.inc` (334 tensors, 35 keys).
CI accounts both against the loaders' own enumerations with no asset:

```console
./build/tests/test_qwen38_27b_gguf_manifest
```

The load itself now **refuses a file carrying tensors nothing reads**, naming
them, before the tokenizer and before any weight byte. That is the direction
that was silent: a tensor the loader asks for and the file lacks already refuses
by name, and one the file ships and no loader reads was simply dropped. On this
artifact that matters concretely, `Qwen3.8-27B-Q4_K_M.gguf` declares
`qwen35.block_count = 65` with `qwen35.nextn_predict_layers = 1`, so it holds 64
decoder blocks plus an MTP drafter at `blk.64`, and a reader spending the whole
65 on the trunk would load, decode fluently, and be the wrong graph.

To account the shipped bytes instead of the frozen manifest, name either file:

```console
VLLM_CPP_QWEN38_27B_GGUF=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
VLLM_CPP_QWEN38_27B_MMPROJ=/path/to/mmproj-BF16.gguf \
  ./build/tests/test_qwen38_27b_gguf_manifest
```

That reads only the two headers, no weight data and no 17 GB map, so it costs
seconds rather than the minutes the loader gate above costs. Unset, both live
cases skip loudly.

**What is still owed on these artifacts** is the Q4_K_M arm's token gate against
the pinned llama.cpp, which is `PENDING` on
[#857](https://github.com/mudler/vllm.cpp/issues/857) because that oracle is
recorded `gateable = no`, and any image or video answer at all,
`QUANT-QWEN38-27B-GGUF-ARM`,
[#821](https://github.com/mudler/vllm.cpp/issues/821).

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

The limits are the mechanism and the flag is the sugar, so it is worth stating
what the flag actually does: it does not "skip the encoder", it makes the server
**refuse** multimodal requests.

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

**Not yet:** `--language-model-only` frees no memory. Nothing gates vision-tower
construction on the limits, so the flag today changes what the server accepts,
not what it allocates ([#607](https://github.com/mudler/vllm.cpp/issues/607)
