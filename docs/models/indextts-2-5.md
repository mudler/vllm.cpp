# IndexTTS 2.5

IndexTTS 2.5 loads through `--speech-model`. The server returns a 22.05 kHz
mono WAV from `/v1/audio/speech`.

## Prepare the checkpoint

Start with the `IndexTeam/IndexTTS-2.5` checkpoint directory. The directory
must contain `config.yaml` and
`multilingual_zh_ja_yue_char_del.tiktoken`.

The engine does not read the upstream `.pth` files. Convert them offline:

```sh
python3 scripts/convert-indextts2-checkpoint.py \
  --checkpoint "$CHECKPOINT_ROOT/IndexTTS-2.5" \
  --out "$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors"
```

Before you start the server, the converted directory must contain these files:

- `gpt.safetensors`
- `s2mel.safetensors`
- `bigvgan.safetensors`

The BigVGAN weights come from `nvidia/bigvgan_v2_22khz_80band_256x`. The
conversion script does not create `bigvgan.safetensors` from that separate
checkpoint.

## Start a speech-only server

```sh
build/examples/vllm-server \
  --speech-model "$CHECKPOINT_ROOT/IndexTTS-2.5" \
  --speech-family indextts2 \
  --port 8000
```

You do not need `--model` for a speech-only server. The server registers
`/v1/audio/speech` without loading a text model. `--speech-family` is optional
when the checkpoint has the expected `config.yaml` keys.

## Synthesize speech

The request must include `reference_audio` as a data URL. The data URL must
contain a 16-bit PCM mono WAV.

```sh
REFERENCE_AUDIO="$(base64 -w0 reference.wav)"
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"indextts2\",\"input\":\"Hello world\",\
\"reference_audio\":\"data:audio/wav;base64,$REFERENCE_AUDIO\"}" \
  --output speech.wav
```

## Current limitations

The engine requires a reference clip but does not use the reference clip for
conditioning. The clip does not select the output voice. Reference-audio
conditioning remains incomplete.

The vLLM-Omni quality and parity comparison is pending. Current structural
gates show that the pipeline renders audio. They do not establish output
quality or parity with the oracle.

The inferred emotion path is not implemented. The request can provide text,
but the server exposes no named voice or speaking-rate control.

See the [server reference](../reference/server.md) for the shared endpoint
contract. See the
[IndexTTS 2.5 specification](../../.agents/specs/indextts-2-5.md) for internal
implementation and verification evidence.
