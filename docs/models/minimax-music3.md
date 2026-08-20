# MiniMax-Music3

MiniMax-Music3 generates 44.1 kHz stereo WAV files. Use this page for its
checkpoint, commands, supported arms, and current limits.

## Download the checkpoint

Use `MiniMaxAI/MiniMax-Music3` at revision
`fbdf52fbaaca799592917417eb05f1899f1255ec`.

```sh
hf download MiniMaxAI/MiniMax-Music3 \
  --revision fbdf52fbaaca799592917417eb05f1899f1255ec \
  --local-dir "$CHECKPOINT_ROOT/minimax-music3" \
  --exclude 'qwen_7B/*' '*.pth'
```

The complete repository is 57.4 GB because it also contains a native `.pth`
layout. The supported `diffusers` layout uses 28,517,617,303 bytes.

| Component | Size | Stored dtype |
|---|---:|---|
| `language_model/` | 17.17 GB | BF16 |
| `transformer/` | 9.73 GB | F32 |
| `rvq_depth_decoder/` | 1.29 GB | BF16 |
| `vocoder/` | 217 MB | F32 |
| `condition_encoder/` | 101 MB | F32 |
| `tokenizer/` | 11 MB | n/a |

The loader runs the language model, RVQ depth decoder, and condition encoder in
BF16. It runs the transformer and vocoder in F32. It refuses incompatible dtype
sets by component name.

The native `qwen_7B/`, `flowmatching_vae.pth`, and `dav.pth` layout is not
supported. Convert it with Diffusers
`scripts/convert_minimax_music3_to_diffusers.py`.

## Quantized support

One quantized component is supported:

| Field | Value |
|---|---|
| Repository | `audio-cpp/MiniMax-Music3-GGUF` |
| Revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| File | `rvq_depth_decoder_q4_k.gguf` |
| Size | 405,752,480 bytes |
| SHA-256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |

The file contains 36 Q4_K projections, nine BF16 norms, and two F16 embedding
tables. The other components still use the `diffusers` checkpoint.

Other Music3 GGUF, NVFP4, MXFP4, FP8, INT8, AWQ, GPTQ, bitsandbytes, and MLX
arms are unsupported. ComfyUI GGUF files contain only the DiT and condition
encoder, so they cannot generate audio by themselves.

The loader refuses a quantized component that it cannot identify. It also
refuses a quantized Diffusers tree that declares an unsupported sidecar or
quantization method. The refusal gate does not need a checkpoint:

```sh
cmake --build build -j 8 --target test_minimax_music3_quant
./build/tests/test_minimax_music3_quant
```

The Q4_K gate needs both the Diffusers checkpoint and the pinned GGUF:

```sh
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
VLLM_CPP_MUSIC3_GGUF="$CHECKPOINT_ROOT/rvq_depth_decoder_q4_k.gguf" \
  ./build/tests/test_minimax_music3_quant_real
```

## Generate a sample

Use a duration below eight seconds and a small step count for an initial test.
CPU generation is slow. A 0.1-second request can take tens of minutes.

```sh
minimax-music3-gen --model "$CHECKPOINT_ROOT/minimax-music3" \
  --out song.wav --lyrics @lyrics.txt \
  --description "Genre: acoustic pop. BPM: 96." \
  --duration 4 --steps 8 --seed 7 --device 0
```

Use `--device 1` to place the language model, DiT, and RVQ depth decoder on the
configured accelerator. The condition mix, scheduler, sampling, and vocoder
remain on the host. Durations above eight seconds use overlapping windows. The
complete multi-window composition has no oracle comparison.

## Start the server

```sh
build/examples/vllm-server \
  --speech-model "$CHECKPOINT_ROOT/minimax-music3" \
  --speech-family minimax-music3 --speech-device 0 --port 8000
```

You do not need `--model` for a speech-only server. The server detects the
standard checkpoint layout, so `--speech-family` is optional.

`--speech-device 0` runs the family on the CPU. `--speech-device 1` requests the
configured accelerator for the language model, DiT, and RVQ depth decoder. The
server refuses accelerator selection when the build or family cannot provide
the required device path. It also refuses `--speech-device` without
`--speech-model`.

## Request a song

```sh
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-music3", \
       "lyrics":"[Verse]\\nMorning light filtering through the pine\\n", \
       "description":"Genre: acoustic pop. BPM: 96. Key: C major.", \
       "audio_duration":4, "num_inference_steps":8, "seed":7}' \
  --output song.wav
```

The route accepts fields at the top level or inside `extra_params`. Values in
`extra_params` take precedence.

- `audio_duration` or `duration` sets the length in seconds. The default is 60.
- `num_inference_steps` sets the denoising step count. The default is 30.
- `guidance_scale` controls classifier-free guidance on the DiT. The default is
  `1.7`; `0` is valid and selects the unconditional branch.
- `seed` controls the autoregressive and denoising draws.
- `lyrics` supplies the song text.
- `description` supplies the music caption.

The route clamps the autoregressive length to 9,000 frames. A positive duration
below `0.04` seconds is invalid. A negative duration is invalid. A duration of
`0` or an omitted duration selects the family default of 60 seconds.

The route refuses these request fields by their exact names:

| Fields | Reason |
|---|---|
| `voice`, `speaker`, `speed` | The family has no named voice or rate control. |
| `stream`, `stream_format` | The engine produces the complete song before it returns samples. |
| non-`wav` `response_format` | The server has no encoder for another format. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | The autoregressive stage uses its fixed top-50 sampler. |
| `audio_duration_s`, `max_new_tokens`, `token_count`, `duration_tokens` | Use `audio_duration` or `duration` in seconds. |
| `instructions` | Use `description` for the music caption. |
| `ref_audio`, `ref_text` | This family does not condition on a reference clip or transcript. |
| `task_type`, `x_vector_only_mode`, `initial_codec_chunk_frames` | The family has one synthesis mode and a fixed chunk schedule. |

The route also refuses `lyrics` or `description` when either field is empty. It
refuses conflicting values for `description` and its `prompt` alias. Top-level
fields and fields under `extra_params` use the same rules; `extra_params` wins
when both locations contain the same field.

## Inspect stage timing

Set `VLLM_CPP_MUSIC3_PROFILE=1` to print a stage timing table to standard error:

```sh
VLLM_CPP_MUSIC3_PROFILE=1 minimax-music3-gen \
  --model "$CHECKPOINT_ROOT/minimax-music3" \
  --duration 4 --steps 8 --device 1
```

The table separates leaf times, containing spans, call counts, and unattributed
time. It is a within-run attribution tool, not a benchmark harness. See
[benchmarks](../BENCHMARKS.md) for accepted measurements.

## Vocoder precision and device selection

MiniMax-Music3 shares its one-dimensional vocoder convolutions with MiniMax-H3,
LTX-2.5, and IndexTTS-2.5. These operations now accumulate in F32, matching
torch's float-convolution behavior. This changes waveform bytes compared with
builds that used the earlier F64 accumulator, but every committed model golden
still passes at its existing tolerance; no tolerance was widened.

Set `VLLM_CPP_VOCODER_DEVICE=cuda` to request the CUDA providers for
`vt::Conv1d` and `vt::ConvTranspose1d`. The default remains `cpu`. The setting
accepts any device name known to `vt`, but the build must contain providers for
that device or startup refuses the request instead of falling back silently.

The CPU and CUDA implementations use the same F32 operation order and are
intended to be byte-identical. Their earlier F64 implementations passed a
`memcmp` gate on Jetson Thor, but that CPU-versus-CUDA gate has not been rerun
since the accumulator changed to F32. Treat the current CUDA vocoder arm as
unverified until that device gate is repeated. See the
[F32 accumulator specification](../../.agents/specs/vt-conv1d-f32-accumulator.md)
for the precision evidence and outstanding device measurement.

## Run the checkpoint gates

```sh
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
  ./build/tests/test_minimax_music3_loader
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
  ./build/tests/test_minimax_music3_ar_real
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
  ./build/tests/test_minimax_music3_acoustic_real
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
  ./build/tests/test_minimax_music3_llm_real
VLLM_CPP_MUSIC3_CHECKPOINT="$CHECKPOINT_ROOT/minimax-music3" \
  VLLM_CPP_MUSIC3_DIT=1 ./build/tests/test_minimax_music3_e2e_real
```

Set `VLLM_CPP_MUSIC3_DEVICE=1` on the language-model and DiT gates to select the
accelerator arm. The stage gates compare fixed inputs with oracle captures. A
generated song proves that the pipeline runs, but it does not establish musical
parity because generation includes seeded random draws.

See the [MiniMax-Music3 specification](../../.agents/specs/minimax-music3.md)
for implementation chronology, mutation evidence, rejected approaches, and
open performance work.
