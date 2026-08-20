# MiniMax-Music3

Use this page for MiniMax-Music3 checkpoints, commands, supported arms, and current limitations.

## MiniMax-Music3: the exact weights (so a song is reproducible)

**The repository is 57.4 GB and the arm we load is 28.5 GB**, because
`MiniMaxAI/MiniMax-Music3` ships the same weights **twice**: a native
`AbabForCausalLM` + `.pth` layout that SGLang-Omni serves, and a `diffusers`
six-component layout. They are the same numbers in a different arrangement —
diffusers' own `scripts/convert_minimax_music3_to_diffusers.py` renames tensors
and does nothing else — and this port loads the diffusers one. So the download
is 57.4 GB unless you filter, and what has to fit is 28.5 GB.

## The arm that loads: `diffusers`, bf16 + fp32

Repository [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
revision **`fbdf52fbaaca799592917417eb05f1899f1255ec`**. First-party. A repo id
alone is not a pin — checkpoints do get re-quantized in place under an unchanged
name — so the revision is recorded, and it was verified rather than copied:
`condition_encoder/diffusion_pytorch_model.safetensors` on disk here hashes to
`83179c5eaa9a68a370affe0c1b96c2179f659ea4175666b31071490a202c2a4d`, which is
that revision's own LFS record for the file.

| component | file(s) | size | dtype on disk |
|---|---|---|---|
| `language_model/` | `model-0000{1,2,3,4}-of-00004.safetensors` + index | **17.17 GB** | BF16 |
| `transformer/` | `diffusion_pytorch_model-0000{1,2}-of-00002.safetensors` + index | **9.73 GB** | **F32** |
| `rvq_depth_decoder/` | `diffusion_pytorch_model.safetensors` | **1.29 GB** | BF16 |
| `vocoder/` | `diffusion_pytorch_model.safetensors` | **217 MB** | F32 |
| `condition_encoder/` | `diffusion_pytorch_model.safetensors` | **101 MB** | F32 |
| `tokenizer/` | `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja` | **11 MB** | — |
| `scheduler/` | `scheduler_config.json` | 483 B | — |
| the root itself | `modular_model_index.json`, `config.json`, `README.md` | 14 KB | — |
| | **resident total** | **28.5 GB** (28 517 617 303 B) | |

The transformer being 9.73 GB for a 2.4B model is **fp32 storage, not a 4.9B
model** — that is upstream's own choice for the acoustic half and we mirror it.
The download:

    hf download MiniMaxAI/MiniMax-Music3 --revision fbdf52fb \
      --local-dir "$CHECKPOINT_ROOT/minimax-music3" \
      --exclude 'qwen_7B/*' '*.pth'

Two components are BF16 and three are F32, and **that set is not runnable as
stored**. Upstream casts in exactly two places, so the language model, the RVQ
depth decoder and the condition encoder must share one dtype; the gated
configuration is bf16 for those three and fp32 for the transformer and vocoder.
The loader enforces it and refuses a violation by name. The section below has
the detail.

## The arm that is REFUSED: the native `.pth` layout

The same repository's other 28.9 GB. **We refuse it by name** — a tree in this
shape is diagnosed as the native arm, told which diffusers components it lacks,
and pointed at the conversion script. It is never silently mis-loaded.

| file | size | what it holds |
|---|---|---|
| `qwen_7B/qwen_7B/` | ~17 GB | `AbabForCausalLM` shards; the RVQ depth decoder and the audio embedding live *inside* them as `model.audio_decoder.*` / `model.audio_extra_embedding` |
| `flowmatching_vae.pth` | ~9.7 GB | the DiT plus the condition projection |
| `dav.pth` | ~0.2 GB | the DAC Flow-VAE decoder |

**SGLang-Omni serves this arm exclusively.** If you are comparing against
`sgl-omni serve`, that is the layout it reads — same weights, so the comparison
is valid, but not the same files.

## The quantized arm that IS implemented: GGUF Q4_K, one component

| field | value |
|---|---|
| repo | [audio-cpp/MiniMax-Music3-GGUF](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) — **third party**, not MiniMaxAI |
| revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| file | `rvq_depth_decoder_q4_k.gguf` |
| size | 405 752 480 bytes (406 MB, against 1.29 GB bf16) |
| sha256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |
| contents | 47 tensors: 36 Q4_K projections, 9 BF16 norms, 2 F16 embedding tables |

It is the **only** quantized arm implemented, and one component is not a
quantized model. The remaining four are refused by name and owed; the section
"MiniMax-Music3: the quantized arms" below records what each refusal says.

## The quantized arms that are REFUSED — and they are all third-party

**MiniMaxAI ships bf16/fp32 only.** A HuggingFace survey on 2026-08-14 found
**fourteen community repositories in five formats**, published within days of the
release, and none of them is from the model's authors. Every one carries
different provenance from a first-party release, and every one except the single
Q4_K file above is refused by name.

| format | repositories | coverage | state |
|---|---|---|---|
| GGUF, `audiocpp` lineage | [audio-cpp/MiniMax-Music3-GGUF](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) | all five components, bf16 and Q4_K arms | `rvq_depth_decoder_q4_k` **LOADS**; `transformer_q4_k` (1 396 MB), `language_model_q4_k` (7 184 MB), `vocoder` (217 MB) and `condition_encoder` (101 MB) are **OWED**. Note the last two are bf16 GGUF, not k-quant — same size as the safetensors, so they buy nothing |
| GGUF, `mm3` lineage | [scragnog/MiniMax-Music3-GGUF](https://huggingface.co/scragnog/MiniMax-Music3-GGUF) | 2-file split (`mm3-lm-*` / `mm3-synth-*`), 13 tiers incl. MXFP4 and NVFP4 as GGML tensor types | **REFUSED**: needs a rename table *plus* fused QKV to split and folded weight-norm to invert. Its NVFP4 tier uses GGML type id 40, which is not a standard llama.cpp id |
| GGUF, ComfyUI lineage | [Abiray](https://huggingface.co/Abiray), [realrebelai/MiniMax-Music-3_GGUFs](https://huggingface.co/realrebelai/MiniMax-Music-3_GGUFs), [molbal](https://huggingface.co/molbal), [ChrisColeTech](https://huggingface.co/ChrisColeTech) | the 2.46B **DiT alone**, Q2_K…Q8_0, 0.9-2.7 GB | **REFUSED, and it can never be a complete arm**: these files carry the DiT and condition encoder only — no language model, no depth decoder, no vocoder — so even a finished GGUF arm would not make them generate audio |
| int8 / w4a8 | [Comfy-Org/MiniMax-Music-3](https://huggingface.co/Comfy-Org/MiniMax-Music-3) (`_int8_convrot`), [NidAll/MiniMax-Music3-W4A8](https://huggingface.co/NidAll/MiniMax-Music3-W4A8), [dummy9996/…-w4a8-bf16-comfyui](https://huggingface.co/dummy9996) | DiT | **REFUSED** by name |
| MLX 4/6/8-bit | [ddalcu](https://huggingface.co/ddalcu), [vanch007](https://huggingface.co/vanch007), [elishabjm](https://huggingface.co/elishabjm) | | **REFUSED**: MLX is a shared seam this project implements for no model, so it is not a per-model addition |
| proprietary | [infosave/MiniMax-Music-3-cmf](https://huggingface.co/infosave/MiniMax-Music-3-cmf) (Cortiq 4-bit) | | **not implementable**, recorded rather than owed |

**"The GGUF arm" is three mutually incompatible lineages, and
`general.architecture` cannot separate them** — it reads `audiocpp`, `mm3`,
`qwen3` and `wan` across files of the same model, and `wan` collides with genuine
Wan video GGUFs. That is why the detector keys on
`audiocpp.model_spec.family` instead, and why pointing a `.gguf` at this loader
gets a refusal naming the lineage rather than a shape error.

**NOT found** by those queries on that date: AWQ, GPTQ, compressed-tensors, fp8 /
`fp8_e4m3fn` / `fp8_scaled`, bitsandbytes. That is "not found by these queries on
this date", never "does not exist".


## MiniMax-Music3: the checkpoint loader

**It loads, it does not generate.** `include/vllm/model_executor/models/`
`minimax_music3_loader.h` is phase W1 of #672 — it resolves the shipped
`diffusers` layout, parses the six component configs, and accounts every tensor
in the files against what those configs owe. No forward, no scheduler step and
no audio; those are W2-W7, and nothing below produces a song.

Point it at the **diffusers arm**, the six-component tree:

```
minimax-music3/
  modular_model_index.json
  transformer/           config.json + 2 shards + index   441 tensors  F32
  condition_encoder/     config.json + 1 file               4 tensors  F32
  rvq_depth_decoder/     config.json + 1 file              47 tensors  BF16
  vocoder/               config.json + 1 file             121 tensors  F32
  language_model/        config.json + 4 shards + index   399 tensors  BF16
  scheduler/scheduler_config.json
  tokenizer/
```

`MiniMaxMusic3ResolveCheckpoint` refuses anything else **by name**, and the
refusal you are most likely to hit is the useful one. The same repository also
ships a **native** arm — `qwen_7B/qwen_7B/`, `flowmatching_vae.pth`, `dav.pth` —
which SGLang-Omni serves and which holds every weight this port needs in a layout
nothing here reads. Pointed at that tree the loader names it as the native arm,
lists the diffusers components it lacks, and tells you to convert it with
diffusers' `scripts/convert_minimax_music3_to_diffusers.py`. It is never
silently mis-loaded.

Two things the loader enforces that a correctness gate later could not catch:

**On-disk dtype and runtime dtype are different things, and the loader keeps
them apart.** The files store F32 for the transformer, condition encoder and
vocoder and BF16 for the RVQ depth decoder and language model, and
`MiniMaxMusic3AccountTensors` refuses a file that disagrees. That set is *not* a
runnable configuration. Upstream casts in exactly two places, `denoise.py:83`
(condition encoder output into the transformer) and `decoders.py:84` (latents
into the vocoder), and never on the way in: `denoise.py:82` hands the language
model's hidden states to the condition encoder with a device move and no dtype
move. So the autoregressive half must share one dtype, and loading the on-disk
set raises `Input type (c10::BFloat16) and bias type (float) should be the same`
from `condition_embedder_minimax_music3.py:64`.

`MiniMaxMusic3ResolveRuntimeDtypes` answers the runtime question.
`kBf16ArFp32Acoustic` is the gated configuration: language model, depth decoder
and condition encoder in bf16, transformer and vocoder in fp32.
`MiniMaxMusic3CheckRuntimeDtypes` refuses a violation by name, listing all three
autoregressive components with their dtypes, because upstream's own error names
a bias dtype and never says which component disagreed with which.
`kAsStored` is kept selectable so that failure stays reproducible; it is
reported as not runnable rather than quietly repaired.

**The vocoder's weight norm is folded at load.** Its 30 weight-normed
convolutions ship as torch's legacy `weight_g`/`weight_v` pairs;
`MiniMaxMusic3LoadVocoderWeights` collapses each to a single `<module>.weight`
through `vocoder1d::MaterializeWeightNorm`, so no `_g`/`_v` name survives and
nothing downstream can read the direction `v` as if it were the weight. Four of
the thirty are `ConvTranspose1d`, whose weight is `[C_in, C_out, K]` — torch
reduces over dimension 0 either way, which for those four is the *input* channel.

## MiniMax-Music3: the quantized arms

**One quantized arm loads: the RVQ depth decoder from a GGUF Q4_K file.**
Everything else is the bf16/fp32 diffusers checkpoint — bf16 `language_model` +
`rvq_depth_decoder` + `condition_encoder`, fp32 `transformer` + `vocoder`,
~28.5 GB resident.

The implemented arm is pinned to a specific artifact, because an unpinned
quantized checkpoint is not reproducible:

| Field | Value |
|---|---|
| repo | `audio-cpp/MiniMax-Music3-GGUF` |
| revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| file | `rvq_depth_decoder_q4_k.gguf` (405 752 480 bytes) |
| sha256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |

`MiniMaxMusic3LoadRvqDepthDecoderFromGguf` reads it: 47 tensors as 36 Q4_K
projections, 9 BF16 norms and 2 F16 embedding tables, dequantized to bf16
through the shared `gguf_dequant.h` seam. Only the **audio-cpp lineage** is
read, keyed on `audiocpp.model_spec.family == "minimax_music3"` — not on
`general.architecture`, which reads `audiocpp`, `mm3`, `qwen3` *and* `wan` across
GGUFs of this one model and collides with genuine Wan video checkpoints. The
other two published lineages are refused by name.

**The other quantized formats still refuse**, and quantized MiniMax-Music3
checkpoints do exist in five formats — a survey on 2026-08-14 found fourteen
community repositories. Rather than mis-loading one or failing with a confusing
shape error, `MiniMaxMusic3ResolveCheckpoint`, `MiniMaxMusic3AccountTensors` and
`MiniMaxMusic3LoadConfig` each refuse **by name**:

```
minimax_music3: this checkpoint is QUANTIZED -- GGUF (evidence:
condition_encoder.gguf, language_model_q4_k.gguf, ...; 5 of 5 entries examined
carry the marker). NO quantized arm is implemented for MiniMax-Music3, so this
is REFUSED rather than mis-loaded: a GGUF arm needs a name map, the
GGUF-vs-torch dim reversal, a geometry source, and k-quant dequantization routed
through vllm/model_executor/model_loader/gguf_dequant.h ...
The supported arm is the bf16/fp32 diffusers arm ... The quantized arms are owed
rather than forgotten: phase W7 of .agents/specs/minimax-music3.md, issue #672.
```

Eight formats are diagnosed — GGUF, NVFP4, MXFP4, FP8, INT8, AWQ/GPTQ,
bitsandbytes and MLX — plus an `UNIDENTIFIED` case. Each message names the
evidence found in *your* file, how many entries carried it, what a working arm
would need, and the arm that does load. Detection happens in three places,
because a quantized checkpoint announces itself in three different ways:

| You point us at | Caught by | Because |
|---|---|---|
| a directory of `.gguf` files | the tree walk (depth 2, so `diffusion_models/` and `text_encoders/` count) | there is no component directory and no config to inspect |
| a diffusers-shaped tree whose tensors are quantized | the manifest scan, from safetensors headers only | the sidecars (`weight_scale_2`, `weight_packed`, `qweight`, `absmax`) and the dtype-only formats (fp8, int8) are invisible to a shape check |
| a checkpoint that *declares* it | the config parse | `quantization_config.quant_method`, or MLX's bare `quantization` |

A bare `weight_scale` with no `weight_scale_2` and no `weight_packed` is
reported as unidentified and the message names all three candidate schemes. It
never picks one: guessing yields a finite, correctly shaped, correctly scaled,
**wrong** result that no shape gate can see.

Note if you hold a ComfyUI-format Music3 GGUF: those ship the DiT and condition
encoder only — no language model, no depth decoder, no vocoder — so they cannot
generate audio even once a GGUF arm lands.

The refusal gate needs no checkpoint and no network:

```sh
cmake --build build -j 8 --target test_minimax_music3_quant
./build/tests/test_minimax_music3_quant
```

The Q4_K arm's own gate needs the pinned GGUF and the bf16 checkpoint, and skips
loudly without them:

```sh
CHECKPOINT_ROOT=... \
  ./build/tests/test_minimax_music3_quant_real
```

It does not merely check that the numbers land inside a tolerance. It asserts
the **resident ggml type** of all 47 tensors, checks the dequantized values lie
on the **Q4_K lattice** (at most 16 distinct values per 32-element sub-block —
a structure a bf16 read cannot produce), and bounds the output **two-sidedly**.
The lower bound is the important one: a silent dequant fallback to the bf16
weights lands *closer* to the golden (mean|d| 0.00182) than the genuine
quantized arm (0.0324), so upper bounds alone cannot tell them apart.

## Reproduce the real-checkpoint gates

Set `VLLM_CPP_MUSIC3_CHECKPOINT` to the 27 GB MiniMax-Music3 checkpoint tree.
Each test skips with a message when the variable is unset.

```sh
VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_loader

VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_ar_real

VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_acoustic_real

VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 VLLM_CPP_MUSIC3_DIT=1 \
  ./build/tests/test_minimax_music3_acoustic_real

VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  ./build/tests/test_minimax_music3_llm_real

VLLM_CPP_MUSIC3_CHECKPOINT=/path/to/minimax-music3 \
  VLLM_CPP_MUSIC3_DIT=1 \
  ./build/tests/test_minimax_music3_e2e_real
```

The DiT cases need `VLLM_CPP_MUSIC3_DIT=1` because they load 9.1 GB of fp32
weights. The acoustic DiT gate runs four 2.4B forwards on the host.


## MiniMax-Music3: the autoregressive half

Phases W2 and W3 of #672.
`include/vllm/model_executor/models/minimax_music3_ar.h` is what consumes three
of W1's six components: the prompt the `language_model` is driven with, the
semantic stage's classifier-free-guidance logit pipeline, the learned 8-layer
condition mix, and the 4-layer RVQ depth decoder. **It still does not generate a
song** — the DiT, the scheduler and the vocoder are W4–W5, and the 8.6B
`Qwen3ForCausalLM` forward itself is the remainder of W2.

## MiniMax-Music3: the acoustic half

Phases W4 and W5 of #672.
`include/vllm/model_executor/models/minimax_music3_acoustic.h` is the rest of the
pipeline: the 2.4B fp32 flow-matching DiT, the `FlowMatchEulerDiscreteScheduler`
with `invert_sigmas`, the classifier-free-guidance mix, the denoise loop's
overlapping-window bookkeeping, and the DAC Flow-VAE vocoder that turns latents
into a **44100 Hz stereo** waveform. Joining the two halves through
`SpeechRegistry`, the `vllm_speech_*` ABI and the example server is W6, and the
8.6B `Qwen3ForCausalLM` forward at the front of the pipeline is the rest of W2 —
see [the language model](#minimax-music3-the-language-model-and-the-end-to-end-path).

Configs are W1's (`MiniMaxMusic3TransformerConfig`,
`MiniMaxMusic3VocoderConfig`, `MiniMaxMusic3SchedulerConfig`) rather than new
ones, and every convolution, transposed convolution, pad and activation is a
call into the shared `vllm::vocoder1d` primitives. Nothing in `vocoder1d` is
modified, so MiniMax-H3 and IndexTTS-2.5 are byte-identical.

## MiniMax-Music3: the language model, and the end-to-end path

The rest of phase W2 of #672, and the piece that made the pipeline whole.
`include/vllm/model_executor/models/minimax_music3_llm.h` carries the
autoregressive loop itself (`encoders.py:299-353`) and the 8.6B
`Qwen3ForCausalLM` at its centre. With it, a request generates a song.

## The `inputs_embeds` entry the dense path did not have

Upstream calls `language_model.model(inputs_embeds=...)` twice and
`input_ids` never (`encoders.py:311`, `:353`), because the frame feedback
`_embed_audio_frame` is a *sum* of one language-model embedding row and seven
depth-decoder rows scaled by `num_codebooks^-0.5` — a continuous vector that
corresponds to no vocabulary entry and that no token id can spell.

`Qwen3DenseModel::ForwardEmbeds` is that door. The Qwen3 family already had it
on its multimodal siblings — `qwen3_vl.h` takes `inputs_embeds_bf16` after
scattering the vision tower's rows into it, and Gemma-4 and Muse-Glimmer do the
same — because upstream's own `Qwen3Model.forward` accepts either input. Only the
**dense** registration had never wired it.

It is additive, and that is asserted rather than argued: feeding the embedding
*of the same token ids* through the new entry reproduces `Forward` **bit for
bit**, in the logits and in the paged KV it wrote, and
`tests/vllm/models/test_qwen3_forward.cpp` checks both. `Qwen3ForCausalLM`,
`LlamaForCausalLM`, `MistralForCausalLM`, `InternLM2ForCausalLM` and
`InternLM3ForCausalLM` all ride that one forward, so nothing less than
bit-identity would do.

`out_hidden` is the second half of the same entry: the post-final-norm rows,
returned from the forward that produced the logits. Music3 reads
`last_hidden_state[:, -1]` and then applies `lm_head` to that very row, so
fetching the two halves with two 8.6B passes would be pure waste.

## `num_condition_layers: 8` does not mean eight transformer layers

Worth stating because it is the reading a fresh implementer reaches for. The
eight rows of a `frame_hiddens` entry are `cat(last_hidden, depth_hidden_1..7)`
(`encoders.py:343`) — **one** language-model hidden state and the **seven**
per-depth-step states of the RVQ decoder. Nothing captures per-layer outputs
from the Qwen3 stack, and nothing needs to.

## Speech and music generation

A **music-only server**, which is what you almost certainly want:

    vllm-server --speech-model /path/to/minimax-music3 \
      [--speech-family minimax-music3] [--speech-device 0|1] [--port 8000]

**`--model` is not required here**, and that is deliberate. Upstream's own recipe
is `sgl-omni serve --model MiniMaxAI/MiniMax-Music3` and nothing else: a music
model is not an accessory to a text model. With `--speech-model` alone this
server loads the music checkpoint, registers `/v1/audio/speech`, and registers
**nothing else** — no `/v1/completions`, no `/v1/chat/completions`. That is the
same task-conditional shape a pooling checkpoint (`/v1/embeddings` only) and a
Parakeet checkpoint (`/v1/audio/transcriptions` only) already take here, and the
same one vLLM's `api_server.py:255-265` uses.

Attach it to a text server instead, and one process serves both surfaces:

    vllm-server --model /path/to/text-model \
      --speech-model /path/to/minimax-music3

`--speech-model` names the checkpoint **set** — MiniMax-Music3 ships six
component directories beside a `modular_model_index.json`, so this is not a
single model directory. `--speech-family` is optional: omitted, the family is
**detected** by inspecting the artifact, and a directory no registered family
claims is refused at startup naming every family that was tried. A name that is
not registered is refused too; it is never treated as a hint, because the wrong
family would not fail — it would render noise. `--speech-family` without
`--speech-model` is still an error: there is nothing to load it from.

`--speech-device` says **where** the family runs. `0` is the default and the CPU
arm; `1` is the accelerator this build resolves. It is refused rather than
substituted: `--speech-device 1` on a build with no accelerator backend, or on a
partial backend that has not registered this family's kernels, fails at startup
naming the piece that is missing. `--speech-device` without `--speech-model` is
an error for the same reason `--speech-family` is — a knob that applies to
nothing reads as one that was honoured. What device 1 currently moves for
MiniMax-Music3 is documented under
[What runs on the device](#what-runs-on-the-device-and-what-does-not), and it is
**not the whole model**.

In the speech-only form the served model name defaults to the **family**
(`minimax-music3`) rather than to a directory basename, because there is no
`config.json` to take one from. `--served-model-name` still wins.

A successful music-only start prints what it resolved, so you can tell a working
server from a listening one without sending a request:

    server: speech/music-only model (family=minimax-music3, 44100 Hz,
            text-only synthesis, family DETECTED, device cpu);
            serving /v1/audio/speech
    server: listening on http://0.0.0.0:8000 (model 'minimax-music3')

`family DETECTED` means the artifact was inspected; `family DECLARED` means you
passed `--speech-family`. `text-only synthesis` is the answer to
`requires_reference_audio()` — a family that needs a reference clip says
`reference clip REQUIRED` there instead, and refuses a clipless request before
anything stages. `device` is what the load **granted**, not what you asked for:
a build that cannot serve `--speech-device 1` refuses at startup rather than
printing `cuda` and running on the CPU.

Or skip HTTP entirely. `minimax-music3-gen` drives the same seam through the C
ABI and writes the WAV itself:

    minimax-music3-gen --model /path/to/minimax-music3 --out song.wav \
      --lyrics @lyrics.txt --description "Genre: acoustic pop. BPM: 96." \
      --duration 8 --steps 8 --seed 7 [--device 0|1]

`--lyrics` and `--description` take literal text or `@path` to read a file,
because lyrics are multi-line and a `[Verse]` tag inside an argv is easy to
mangle. It prints the delivered length, rate, channels, RMS, peak and wall clock
to stderr — the *delivered* length, not the requested one, because a duration
resolves to a whole number of 25 Hz frames and is therefore quantized. It also
prints the device the handle **resolved to** rather than the one `--device`
asked for, which is the difference between timing two arms and timing one arm
twice.

The route is OpenAI's `createSpeech` shape, with the two **music** inputs as
additional named fields:

    curl http://localhost:8000/v1/audio/speech \
      -H 'Content-Type: application/json' \
      -d '{"model": "minimax-music3",
           "lyrics": "[Verse]\nMorning light filtering through the pine\n",
           "description": "Genre: acoustic pop. BPM: 96. Key: C major.",
           "audio_duration": 30, "num_inference_steps": 30, "seed": 7}' \
      --output song.wav

The response body is RIFF/WAVE 16-bit PCM at the family's **native** rate
(44100 Hz stereo for MiniMax-Music3, never resampled), with content type
`audio/wav`.

**Every field, and what it does.** Anything not in this table is refused by name
rather than ignored — see below the table for why that polarity matters here.

| field | type | default | what it does |
|---|---|---|---|
| `lyrics` | string | **required** for MiniMax-Music3 | the sung text, with `[Verse]` / `[Chorus]` section tags. An empty lyric normalizes to a bare `[start]` prompt, so it is a 400 rather than an instrumental |
| `description` (alias `prompt`) | string | **required** for MiniMax-Music3 | genre, BPM, key, instrumentation, mood. NOT a voice or speaker description. Supplying both spellings with different values is a 400, never a silent winner |
| `audio_duration` (alias `duration`) | number, seconds | `60` | resolved to `int(seconds x 25)` autoregressive frames, then **clamped** to the 9000-frame ceiling — the same silent clamp upstream applies (`encoders.py:287`). Shorter than one frame (0.04 s) is a 400. **`0` means "take the family's default"**, the same as omitting it; only a NEGATIVE value is a 400 (#1338) |
| `num_inference_steps` | integer | `30` | flow-matching Euler steps in the acoustic half. Must be > 0 |
| `guidance_scale` | number | `1.7` | classifier-free guidance on the DiT. **0 is legal** and selects the unconditional branch, so omitting the field is how you ask for the default — not sending 0 |
| `seed` | integer | `0` | seeds the autoregressive top-k draw *and* the initial denoise latents. A fixed seed, not a random one: 0 is as deterministic as any other value |
| `model` | string | — | echoed; the route does not check it |
| `response_format` | string | `"wav"` | `"wav"` is the only accepted value |

`lyrics` and `description` are separate fields rather than one `input` behind a
separator because upstream runs a different normalizer over each — `_clean_caption`
on the description, `_normalize_lyrics` on the lyrics (`encoders.py:54-91`). A
one-utterance family keeps using OpenAI's `input`; MiniMax-Music3 refuses it, so
a request cannot half-arrive.

**We expose `guidance_scale` where neither upstream arm does.** In diffusers it
is frozen into the guider component at 1.7 (`denoise.py:180`); in SGLang-Omni it
is a serve-time knob (`dit_cfg_scale`) and not a request field. It is a genuine
per-request control here, and its default is upstream's 1.7.

**Every refusal, and the one rule behind them.** A knob the server will not
honour must not come back behind a 200. Silently dropping one returns audio the
caller did not ask for with nothing to say so — and this project has already paid
for that once (#925), which is why the list is long rather than convenient.

| refused | why |
|---|---|
| `audio_duration_s` | the name of the *field* the key fills, not a key. It is the misspelling you reach for by reading the struct instead of the docs, and dropping it silently returned the 60 s default: 0.1 s became 60 s, 2 autoregressive frames became 1500, and this project's own e2e gate spent four multi-hour runs inside a 750x job it read as a hung weight load (#852, #925) |
| `voice` | no registered family exposes named voices, and there is no enumeration endpoint to pick one from. Upstream refuses it too (`request_builders.py:90-92`) |
| `speed` | no family implements a rate control. Upstream accepts only `1.0` (`request_builders.py:83-89`) |
| `stream`, `stream_format` | MiniMax-Music3 generates the whole song before the first sample exists, so buffering it into chunks would be a stream in name only. **Upstream has no streaming either** — SGLang-Omni declares `supports_streaming_vocoder=False` and rejects `stream=true` by name (`request_builders.py:115-116`) |
| `response_format` other than `"wav"` | no mp3/opus/aac/flac encoder is vendored, and relabelling RIFF bytes is worse than refusing |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | this model's autoregressive stage has ONE sampler — a fixed top-50 draw (`encoders.py:48,94-103`). There is no temperature to set and no nucleus branch to widen, so the knob can be neither honoured nor honestly ignored. Upstream refuses all four (`request_builders.py:14-19,109-114`). Use `seed` to control the draw |
| `max_new_tokens` | SGLang-Omni's spelling of the length, counted in 25 Hz **frames** rather than seconds (`request_builders.py:56-68`). This route takes `audio_duration` in seconds — divide by 25. Two spellings of one meaning on one route is exactly what #925 was |
| `token_count`, `duration_tokens` | SGLang-Omni's length in **duration tokens** (`protocol.py:355-356`). Same meaning as `audio_duration`, different unit; upstream refuses both by name for this model (`request_builders.py:20-30,71-81`). A length key nobody reads is how a short request becomes a 60 s song (#1315) |
| `instructions` | it names two things and this route honours neither. **OpenAI's** createSpeech uses it for voice **style** and emotion, and no registered family exposes a style control. **SGLang-Omni** uses it for the music **caption**, the string it assembles into `<\|caption_start\|>` and requires non-empty (`request_builders.py:104-106`); this route calls that `description`, so send `description` if the caption is what you meant. Refused rather than aliased because a secondary oracle never becomes the mirror source, and because a global alias would bake one family's meaning into a shared route (#1315) |
| `speaker` | SGLang-Omni's declared **alias** for `voice` (`protocol.py:337-339`). Refusing `voice` and dropping `speaker` refused one spelling of one field and returned a 200 for the other (#1315) |
| `ref_audio`, `ref_text` | upstream's reference-clip spellings, where `ref_audio` is a path or URL. This route takes `reference_audio` as a `data:` URL, because the server and the client need not share a filesystem; no family conditions on a reference **transcript** at all. Upstream refuses both by name (`request_builders.py:20-30,71-81`) |
| `task_type`, `x_vector_only_mode`, `initial_codec_chunk_frames` | there is one synthesis mode per loaded family and no task selector, no speaker-embedding-only path, and the chunk schedule is the family's own (MiniMax-Music3 fixes it at 200 frames with a 100 hop). Upstream refuses all three by name (`request_builders.py:20-30,71-81`) |

**Where you put a key does not change whether it is read.** EVERY key on this
route is accepted at the **top level** and nested under **`extra_params`**,
because vLLM-Omni nests them and OpenAI does not, and a client should not have to
know which surface it is talking to. That covers the generation knobs
(`audio_duration`, `duration`, `num_inference_steps`, `guidance_scale`, `seed`),
the content fields (`model`, `input`, `text`, `language`, `lyrics`,
`description`, `prompt`, `reference_audio`) and every refusal in the table above.
`extra_params` wins where both carry the same key, the same precedence the video
route uses.

This used to be an either/or: an `extra_params` object as empty as `{}` stopped
the top level being read at all, so a body that nested `seed` and put
`audio_duration` where OpenAI puts it lost the duration and got the 60 s default,
which is #925's cost behind a 200 with #925's own guard unable to fire (#1315).
The **content fields** were the second half of the same hole and were fixed one
change later (#1336): a nested `text`, `language` or `reference_audio` was
dropped, which defeated the family refusals that catch those keys at the top
level, and a nested reference clip made a family that requires one answer
"`reference_audio` is required" to a caller who had just supplied it.

A family with no text-only synthesis — IndexTTS-2.5 is one — is refused
**before** anything stages: the route asks the loaded engine's
`requires_reference_audio()` and answers 400 naming the family and the missing
`reference_audio`, which is supplied as a `data:` URL carrying a 16-bit PCM mono
WAV.

**Every stage of MiniMax-Music3 is implemented and gated**, and a request
reaches all of them: the 8.6B `Qwen3ForCausalLM` autoregressive stage, the RVQ
depth decoder, the learned condition mix, the flow-matching DiT and the DAC
Flow-VAE vocoder. **A composed request has been observed to completion** — an
HTTP POST returns a real 44100 Hz stereo WAV (#852) — and the end-to-end gate
now runs it over a real socket against a music-only server. There is no by-name
refusal left: nothing here is unimplemented. IndexTTS-2.5 still refuses naming
its own missing pieces.

**What no gate compares is the music itself.** The autoregressive codes are a
seeded `torch.multinomial` draw and the denoise loop's initial latents are a
seeded normal draw, so a request's waveform can never equal the oracle's golden
— twice over, and structurally rather than by omission. Every *stage* is gated
against the capture on the capture's own recorded inputs; a **generated** song
is evidence that the pipeline runs, not that the notes are right. Believe the
stage gates, and listen with that in mind.

**Ask for less than 8 seconds while you are exploring.** `Music3ChunkPlan` only
splits past 200 autoregressive frames, which is 8 s of audio, and the
multi-window composition — the overlap blend, the carry span, the waveform crop
across windows — is this row's one named coverage gap: each primitive is gated
individually, the composition across windows is not, because the oracle capture
is a single 25-frame window.

**It runs on CPU and it is slow.** Every gate this row has was taken on CPU
(`dgx.casa` was down throughout), and the acoustic half is upstream's own fp32.
A 0.1 s request takes tens of minutes; no speed number exists and none is
claimed. Ask for a short duration and few `num_inference_steps` while you are
checking that it works.

The part that dominates is *not* the one you would guess. The 8.6B language
model goes through `vt` and uses the CPU threadpool; the RVQ depth decoder does
not — it is a scalar host loop with a double accumulator, written that way in
W2/W3 so its reduction order is reproducible against torch. In one 0.1 s request
the depth decoder alone is the majority of the wall clock.

At a *real* duration the DiT is the whole story instead, which is why it is the
stage that moved first: a 45 s clip at the default 30 inference steps runs the
DiT 660 times (30 steps x 2 CFG branches x 11 windows) for roughly 634 TFLOP
against about 29 TFLOP for the entire autoregressive half. On the host loops
that is measured in hours. `--speech-device 1` puts it on the accelerator.

### What runs on the device, and what does not

`--speech-device 1` (or `minimax-music3-gen --device 1`, or
`vllm_speech_model_params.device = 1`) is **a partial arm, and this table is the
whole of it**. Reading it as "the model runs on the GPU" would be wrong in the
direction that matters.

| stage | where `--speech-device 1` runs it |
|---|---|
| 8.6B `Qwen3ForCausalLM` (prefill + every decode step, its paged KV) | **device** |
| guided-logit pipeline, top-k draw, frame feedback embedding | host (two 200 000-wide rows per step; not the cost) |
| **2.4B fp32 flow-matching DiT** (every denoise step, both CFG branches) | **device**, weights staged ONCE |
| **0.646B RVQ depth decoder** (8 appends per frame) | **device**, weights staged ONCE at **bf16** |
| condition mix (once per window), scheduler, CFG mix, Euler step | **host** |
| DAC Flow-VAE vocoder (`Conv1d` / `ConvTranspose1d`) | **host**, scalar loops |

The language model reaches the device because it is already routed through the
shared `Qwen3DenseModel` forward that five text registrations ride — nothing was
forked for it, and the only thing this option changes is which queue that
forward is handed and where its KV cache is allocated.

The DiT reaches it the same way: through shared `vt` ops only
(`MatmulBT`, `LayerNorm`, `AttentionCross`, `RopeFromCache`, `SiluAndMul`,
`Add`), with **no new kernel**. Its 9.7 GB of fp32 weights are uploaded once per
request, before the window loop, and the host copy is released as each tensor
lands — a 45 s clip runs that forward 660 times, so a per-step or even
per-window upload would cost more than the compute it enables. `fp32 stays
fp32`: the acoustic half is float32 because upstream chose float32 for it, and
this arm mirrors that rather than buying speed with a narrower dtype.

The remaining stages do not move, for two different reasons, and both are owed
rather than hidden:

* the condition mix is a host `std::vector<float>` reference loop under
  `-ffp-contract=off` running at `ArCompute::kBFloat16`. It also runs **once per
  window** rather than once per step, so it is outside the per-step loop
  entirely;
* the vocoder needs `ConvTranspose1d`, and **`vt` has no such op at all** — the
  1-D convolutions it does have (`vt::DepthwiseConv1d`, `vt::CausalConv1dFwd`)
  are depthwise or causal-with-state, and `vt::Conv2d` and `vt::DepthwiseConv1d`
  are registered for the **CPU only**. There is no CUDA kernel behind the op
  this stage would need, so it is named here rather than hand-rolled outside the
  seam.

The **depth decoder** reaches the device the same way, and its dtype is the one
thing about it worth knowing. Upstream's `MiniMaxMusic3RVQDepthDecoder` declares
**no dtype at all** — no `dtype` parameter, no `torch.float32` literal, no
`.float()` call — so it inherits whatever `load_components(dtype=...)` resolves,
and that is **bf16** for this checkpoint. The arm therefore stages its weights at
bf16 and keeps every activation at bf16 with f32 accumulation, which is
`vt::MatmulBT`'s own contract. The narrowing is **lossless**: the loader already
rounds every AR-half tensor through bf16 into an f32 carrier, so the device copy
holds exactly the values the host arm holds, in half the bytes.

It runs on five shared ops with **no new kernel** (`MatmulBT`, `RmsNorm`,
`AttentionCross`, `SiluAndMul`, `Add`), with the MLP's gate/up pair routed
through the shared merged-GEMM seam. What does **not** move with it, and is owed
rather than hidden: the audio heads, the CFG mix, the top-k draw and the fed-back
projection row, which together are ~1.6 % of the stage.

**Its numbers are not identical to the host arm's, and the difference is
measured rather than assumed.** Against the host reference, over **six seeds** of
the gate's reduced geometry, the device arm reads a median of exactly **1 bf16
ULP** at every seed, means of 2.095 to 9.904 and worst-case values of 110 to
7340. The gate bounds the median at 2 and the mean at 15; it does **not** bound
the worst case, because the worst case cannot tell a correct arm from a broken
one — a correct arm reads 7340 on the seed where a swapped gate/up half reads
6641. An earlier revision of this document quoted one seed's 110 and 2.095 as
though they were the arm's deviation; they are one draw of it.

Almost all of that deviation — the composed stage goes to **zero,
bit-identical**, once the two are aligned — is one rounding per element, from two
places, and the two are **not** the same kind of thing:

- `vt::SiluAndMul` computes the whole gated expression in f32 where a bf16 torch
  module narrows `silu(gate)` before multiplying by `up`. That one is a genuine
  gap in the shared seam and is tracked as its own issue.
- `vt::RmsNorm` keeps f32 across the weight multiply, and **that is correct**.
  vLLM's own RMSNorm does exactly the same on both its CPU and its CUDA path, and
  upstream reverted the change that would have made it multiply in the weight's
  dtype. The Music3 *host* arm rounds twice because it mirrors the `diffusers`
  module this model is; the device arm rounds once because it mirrors vLLM. Two
  references disagree here and neither side is defective, so this term will not
  go away. An earlier revision of this document called it a shortcoming of the
  shared op.

Until the difference is settled against the oracle, **no throughput number is
quoted for this arm** and the full-scale gate against the committed oracle
goldens has not been run with it.

If the build has no provider for the device you asked for, the depth arm
**refuses by name at staging time**, naming the op and the device, rather than
falling back to the host loop — a silent fallback would be a large slowdown
wearing a correct answer. `--speech-device 0` keeps the host reference arm and
stages nothing.

Because the host stages are unchanged — and because `--speech-device 0` takes
the same `DitForward` it always did, source byte for source byte — the CPU arm
is **bit-identical** to the one every Music3 correctness gate was taken on. The
device arm's output differs from it exactly where the language model's and the
DiT's own arithmetic differ: two stages, not six, and neither difference is a
shape or an ordering defect. The DiT's device forward is gated against the same
upstream goldens at the same tolerance as the host one; nothing was widened for
it, and `VLLM_CPP_MUSIC3_DEVICE=1` runs that comparison on either arm
(`tests/parity/test_minimax_music3_acoustic_real.cpp`, with
`VLLM_CPP_MUSIC3_DIT=1`).

**The two arms do not produce the same song, and that is structural.** The
autoregressive stage has no greedy path upstream: it ends every draw in a seeded
multinomial, so a different logit changes the drawn code and everything after it.
Do not compare the two WAVs sample by sample. What *is* comparable is the
language model's own hidden state against the oracle capture, and
`tests/parity/test_minimax_music3_llm_real.cpp` runs that comparison on either
arm — `VLLM_CPP_MUSIC3_DEVICE=1` selects the device one, unset is the CPU one —
at the same bounds, with the same negative control. Numbers for both are in
[BENCHMARKS](../BENCHMARKS.md).

### Where the time actually goes: `VLLM_CPP_MUSIC3_PROFILE`

The table above says which stage runs where. It does not say what each one
*costs*, and at a real duration that is the only question anyone asks. Set

```sh
VLLM_CPP_MUSIC3_PROFILE=1 minimax-music3-gen --model ... --duration 20 --steps 30 --device 1
```

and the engine prints a `MUSIC3_PROFILE` table to **stderr** when the request
finishes: one row per stage with seconds, a call count, and its share of the
request, then the resident-set size at each stage boundary.

Read it as follows.

* `leaf` rows partition the request and are the ones that add up. `span` rows
  enclose leaves — `ar.TOTAL_loop`, `denoise.TOTAL` — and are printed for
  context but never summed, so the table cannot claim more work than the run
  contained.
* `cnt` rows carry no time at all. They are the counts a split has to state to
  be readable: frames, windows, requested steps.
* `unattributed` is the glue between the leaves — chunk slicing, the overlap
  blend, the Euler step, the WAV assembly. It is printed rather than spread
  silently over the measured stages, so a bracket in the wrong place shows up as
  a number instead of as a plausible share somewhere else.
* the `calls` column on `denoise.dit_device` counts *steps*, not forwards: one
  bracket covers both classifier-free-guidance branches, so the forward count is
  twice it.
* the `load.ar.*` and `load.ac.*` rows break the two weight loads down further.
  They are spans *inside* the `load.ar_weights` / `load.acoustic_weights` leaves,
  so they are printed and never summed. They exist because the safetensors
  reader is an **mmap** whose tensors are copied out, which interleaves
  page-fault I/O with the copy inside a single call — so the load total on its
  own cannot say whether a slow load is storage or CPU, and `load.ac.dit_build`
  in particular touches no file at all.
* the `calls` column on `ar.depth_forward` and `ar.depth_projection` changed
  meaning in #672, so **two profile tables from different builds are not directly
  comparable on those two rows**. The depth decoder used to re-run the whole
  growing depth sequence at each of the seven codebook steps, separately per
  guidance branch — 14 calls per frame, 70 rows of work to read 14 of them. It
  now appends one position at a time against a K/V cache, both branches in one
  batch-2 call: **8 calls per frame, 16 rows**. The seconds fell with the rows;
  the call count fell for a different reason, and reading the drop in `calls` as
  the speedup would double-count it. The output is bit-identical either way.

It is **off unless the variable is set to `1`, `true`, `on` or `yes`**. Any
other value, including a near miss like `y`, leaves it off — an operator who
mistypes gets a run with no table rather than a run whose meaning quietly
changed. With it off, no clock is read and no `/proc` file is opened.

This is an attribution instrument, not a benchmark harness: it takes no GPU
clock window, so its rows are a within-run **split** and must not be quoted as
per-kernel or cross-box figures.

If you want to price the depth stage on your own box without generating a song,
`tools/bench/music3_depth_stage_ab.cpp` drives one frame of it directly. Nothing
RUNS it — it allocates 2.5 GB and is a two-build A/B, which one target could not
express — but both arms are COMPILED, as the never-linked OBJECT libraries
`vllm_music3_depth_stage_ab_{before,after}`, so it cannot rot behind a signature
change while still being the artifact the §16.6 measurement is reproducible from
(#1246). Its header carries the exact `g++` and run lines. Alternate the arms and
take the minimum; it prints one fingerprint per process, after its round loop, so
a "speedup" that changed the answer cannot be mistaken for one that did not.

To price the **vocoder** the same way, `scripts/music3-vocoder-conv-ab.sh` runs
the whole A/B for you:

```sh
scripts/music3-vocoder-conv-ab.sh https://github.com/mudler/vllm.cpp <after-ref> <before-ref>
# LENGTHS=20,40,86,172,344  REPEATS=3  ROUNDS=3  JOBS=8  are the knobs
```

It clones two trees that differ in `src/vt/cpu/cpu_conv1d_general.cpp` and in
nothing else, builds each in its own directory, and **refuses to time anything
when the two binaries hash the same** — that is the failure that voided this
model's first depth A/B, and equal times are noise where equal binaries are
identity. It then runs the correctness gates on the after arm before reading any
speed number, alternates the arms across a sweep of latent window lengths, and
prints `uptime` on both sides of the sweep.

The executable it builds, `vllm_music3_vocoder_conv_ab`, can also be run alone
(`--lengths=`, `--repeats=`). It drives `VocoderDecode` — the same call
`vocoder.decode_window` brackets — at the shipped vocoder geometry with
synthetic weights, so it prices that stage without a checkpoint and makes no
claim about audio. It prints one waveform fingerprint per length, which is how
two arms are shown to agree BIT FOR BIT rather than closely. `ctest` never runs
it (#1334).

**What it times is the WINDOW, not the convolution.** The ratio it prints covers
everything `VocoderDecode` does — `vt::Conv1d`, `vt::ConvTranspose1d`, the
alias-free activations, the strided downsamples, and the threadpool and
allocation around all of them. A kernel-level figure for `vt::Conv1d` alone is
several times larger than the window figure at the same build and thread count,
so the two are not interchangeable and this tool only ever reports the second.

**Measured, so expectations are calibrated rather than hoped for.** On a Jetson
Thor (sm_110, 14 cores) the device arm was *slower* on a two-frame request
(846.6 s vs 835.1 s) and 5.4 % faster on a ten-frame one (1430.4 s vs 1512.1 s).
The difference between the arms works out to about 11.7 s saved per
autoregressive frame against a fixed cost of about 35 s, so it breaks even
around three frames — roughly 0.12 s of audio. If you are generating an actual
song the device arm helps; if you are smoke-testing the shortest request that
enters every stage, it does not.

**A first sample, measured.** Two seconds of stereo music, generated by this
engine through `minimax-music3-gen` on an idle-to-busy 20-core x86 CPU box:

| property | value |
|---|---|
| duration | 1.9969 s (88 064 frames per channel) |
| rate / channels | 44 100 Hz, 2 channels, 16-bit PCM |
| RMS | 0.03169 full-scale |
| peak | 0.97437 full-scale, **0 clipped samples** |
| L != R | 84 073 of 88 064 positions, so the stereo fold is real rather than a duplicated channel |
| wall clock | **3286 s** (54.8 min) for 2.0 s of audio, at `--steps 2`, load average 40-150 throughout |

**Its samples are compared to nothing.** The token gate this row once promised
was withdrawn — upstream's autoregressive stage has no greedy path — and a
generated waveform can never equal the oracle's golden anyway, because both the
codes and the initial latents are seeded random draws. So the numbers above
demonstrate that the pipeline runs end to end and produces a well-formed,
non-silent, non-clipped, genuinely stereo signal. They say nothing about whether
the music is right. The per-stage gates are what say that.

The clip is **not committed**: `scripts/check-pr-size.py` classifies every
repository path, and no classified path accepts a `.wav` outside `tests/`, where
a file compared to nothing would sit beside the goldens and imply it was one.
Regenerate it instead — the command above is the whole recipe.

The same seam is reachable from the C ABI at v21 — `vllm_speech_engine_load`,
`vllm_speech_engine_family` / `_sample_rate` / `_requires_reference_audio` /
`_device`, `vllm_synthesize` and `vllm_speech_result_free` — so HTTP and FFI
drive one implementation. `vllm_speech_model_params.device` is the same 0 = CPU
/ 1 = accelerator selector `--speech-device` sets, and
`vllm_speech_engine_device` reports what the load granted. `vllm_speech_result` carries both the float waveform and the
RIFF/WAVE bytes, so an embedder writes a playable file without a second encoder.
