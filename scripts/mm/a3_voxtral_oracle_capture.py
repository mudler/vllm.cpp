#!/usr/bin/env python3
"""A3 oracle reference capture: e2e AUDIO->TEXT on Voxtral-Mini-3B.

Vehicle `mistralai/Voxtral-Mini-3B-2507` (Whisper-large-v3-class audio encoder +
AudioLanguageAdapter projector + Mistral/Llama text decoder), the pinned oracle
= vLLM 0.25.0 + transformers 5.13.1 + mistral_common 1.11.5, load_format=mistral.

Two phases:
  PHASE 1 (CPU, no GPU): build a DETERMINISTIC 30 s audio clip, run the
    MistralTokenizer audio chat encode, and dump the goldens the C++ A3 pipeline
    reproduces:
      - the canonical PCM16 mono 16 kHz WAV of the (padded) waveform (committed)
      - the encoder input_features log-mel [num_mel_bins=128, n_frames] (the REAL
        vLLM VoxtralEncoderModel.compute_whisper_melspec = torch.stft path)
      - the mel filterbank [1+window_size//2, 128] (mistral_common.audio
        .mel_filter_bank, a config constant, dumped as a golden)
      - the full placeholder-expanded prompt token ids + audio_token_id +
        num_audio_tokens + downsample_factor
  PHASE 2 (GPU under flock): construct the vLLM engine (load_format=mistral) and
    run GREEDY generation K>=3 to (a) capture the golden output token ids + decoded
    text and (b) MEASURE self-determinism to fix the gate FORM (STRICT token-exact
    if all K identical, else the ratified near-tie <=0.5-nat teacher-forced gate).

Provenance (cite file:line):
  vllm/model_executor/models/voxtral.py @ e24d1b24:
    VoxtralForConditionalGeneration.embed_multimodal:382-412 (downsample reshape +
      adapter + split), AudioLanguageAdapter:660-668 (w_in -> gelu -> w_out),
      VoxtralEncoderModel.compute_whisper_melspec:754-786 (torch.stft log-mel),
      forward:819-839, load_weights:502-568, audio_token_id (config) = 24.
  vllm/model_executor/models/whisper.py:458-535 WhisperEncoder (encoder tower).

Run (CPU phase):  ~/venvs/vllm-oracle/bin/python a3_voxtral_oracle_capture.py --phase 1
Run (GPU phase):  flock "${GPU_LOCK:-$HOME/gpu.lock}" ~/venvs/vllm-oracle/bin/python \
                    a3_voxtral_oracle_capture.py --phase 2
Fixtures land in ~/a3_fixture, then are copied into
tests/vllm/multimodal/fixtures/voxtral_audio/.
"""
import argparse
import hashlib
import json
import os
import struct
import wave

import numpy as np

MODEL = "mistralai/Voxtral-Mini-3B-2507"
OUT = os.path.expanduser("~/a3_fixture")
os.makedirs(OUT, exist_ok=True)

SR = 16000
DURATION_S = 30.0
N_SAMPLES = int(round(SR * DURATION_S))  # 480000 (exactly one 3000-frame chunk)


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_deterministic_clip() -> np.ndarray:
    """A fixed multi-tone + seeded-noise 30 s waveform (float32 in [-1,1]).

    Non-trivial spectral content across the mel bands (so a wrong encoder /
    projector / merge flips output tokens) yet fully deterministic. Mirrors the
    A0/A1 synthetic-clip methodology, extended to the 30 s Voxtral chunk."""
    t = np.arange(N_SAMPLES, dtype=np.float64) / SR
    # A few speech-band tones with slow amplitude modulation.
    sig = (
        0.30 * np.sin(2 * np.pi * 220.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
        + 0.20 * np.sin(2 * np.pi * 440.0 * t)
        + 0.15 * np.sin(2 * np.pi * 880.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
        + 0.10 * np.sin(2 * np.pi * 1500.0 * t)
    )
    rng = np.random.default_rng(20260725)
    sig = sig + 0.02 * rng.standard_normal(N_SAMPLES)
    sig = np.clip(sig, -1.0, 1.0)
    return sig.astype(np.float32)


def write_wav_pcm16(path: str, samples: np.ndarray) -> bytes:
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    with open(path, "rb") as f:
        return f.read()


def dump_bin(path: str, arr: np.ndarray) -> str:
    b = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(b)
    return sha256_hex(b)


def phase1():
    from mistral_common.audio import mel_filter_bank
    from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.audio import Audio
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_hf_hub(MODEL)
    audio_encoder = tokenizer.instruct_tokenizer.audio_encoder
    acfg = audio_encoder.audio_config
    ec = acfg.encoding_config  # num_mel_bins, hop_length, window_size
    # model-side encoder dims (params.json multimodal.whisper_model_args.encoder_args)
    D_MODEL, MAX_SRC_POS, DOWNSAMPLE = 1280, 1500, 4
    print("audio_config:", acfg)

    # Canonical decode of the committed WAV: PCM16 -> f32 (== C++ int16/32768.0).
    clip = make_deterministic_clip()
    wav_path = os.path.join(OUT, "voxtral_input_16k_mono.wav")
    wav_bytes = write_wav_pcm16(wav_path, clip)
    # Re-decode exactly like C++ does, so the golden waveform is the quantized one.
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
    wav_f32 = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
    print("wav samples:", wav_f32.shape, "sha", sha256_hex(wav_bytes)[:16])

    question = "Describe what you hear in this audio."
    audio = Audio(audio_array=wav_f32.astype(np.float32), sampling_rate=SR, format="wav")
    chunk = AudioChunk.from_audio(audio)
    req = ChatCompletionRequest(
        messages=[UserMessage(content=[chunk, TextChunk(text=question)])],
        model=MODEL,
    )
    enc = tokenizer.encode_chat_completion(req)
    prompt_ids = list(enc.tokens)
    proc_audio = enc.audios[0].audio_array.astype(np.float32)
    print("prompt len:", len(prompt_ids), "processed audio len:", proc_audio.shape)

    audio_token_id = int(audio_encoder.audio_token)
    num_audio_tokens = int(sum(1 for x in prompt_ids if x == audio_token_id))
    print("audio_token_id:", audio_token_id, "num_audio_tokens:", num_audio_tokens)

    # mel filterbank (the exact vLLM VoxtralEncoderModel constructs).
    mel = mel_filter_bank(
        num_frequency_bins=1 + ec.window_size // 2,
        num_mel_bins=ec.num_mel_bins,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=acfg.sampling_rate,
    ).astype(np.float32)  # [1+window//2, num_mel_bins]
    print("mel_filters:", mel.shape)

    # log-mel input_features via the REAL compute_whisper_melspec (torch.stft).
    import torch

    waveforms = torch.tensor(proc_audio, dtype=torch.float32)
    window = torch.hann_window(ec.window_size)
    stft = torch.stft(
        waveforms, ec.window_size, ec.hop_length, window=window, return_complex=True
    )
    magnitudes = stft[..., :-1].abs() ** 2
    mel_t = torch.tensor(mel, dtype=torch.float32)
    mel_spec = mel_t.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    glmm = getattr(acfg, "global_log_mel_max", None)
    if glmm:
        log_spec_max = torch.tensor(float(glmm))
    else:
        log_spec_max = log_spec.max()
    log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    input_features = log_spec.numpy().astype(np.float32)  # [num_mel_bins, n_frames]
    print("input_features:", input_features.shape, "global_log_mel_max:", glmm)

    downsample_factor = DOWNSAMPLE

    # Fixed sinusoidal encoder position embedding (NOT in the checkpoint — vLLM
    # computes it via transformers.sinusoids in fp32; dumped as a golden constant
    # like the A2 tower's enc_embed_positions, so the C++ encoder loads the identical
    # [max_source_positions, d_model] matrix and the block math is the parity var).
    from transformers.models.whisper.modeling_whisper import sinusoids

    embed_positions = sinusoids(MAX_SRC_POS, D_MODEL).numpy().astype(np.float32)
    print("embed_positions:", embed_positions.shape)

    sha = {}
    sha["embed_positions"] = dump_bin(
        os.path.join(OUT, "voxtral_embed_positions_f32.bin"), embed_positions
    )
    sha["wav"] = sha256_hex(wav_bytes)
    sha["waveform"] = dump_bin(os.path.join(OUT, "voxtral_waveform_f32.bin"), proc_audio)
    sha["input_features"] = dump_bin(
        os.path.join(OUT, "voxtral_input_features_f32.bin"), input_features
    )
    sha["mel_filters"] = dump_bin(os.path.join(OUT, "voxtral_mel_filters_f32.bin"), mel)

    manifest = {
        "model": MODEL,
        "sampling_rate": SR,
        "duration_s": DURATION_S,
        "n_samples": int(proc_audio.shape[0]),
        "window_size": int(ec.window_size),
        "hop_length": int(ec.hop_length),
        "num_mel_bins": int(ec.num_mel_bins),
        "max_source_positions": MAX_SRC_POS,
        "d_model": D_MODEL,
        "n_frames": int(input_features.shape[1]),
        "num_freq_bins": int(mel.shape[0]),
        "downsample_factor": downsample_factor,
        "audio_token_id": audio_token_id,
        "num_audio_tokens": num_audio_tokens,
        "global_log_mel_max": (float(glmm) if glmm else None),
        "prompt_ids": prompt_ids,
        "question": question,
        "sha256": sha,
    }
    with open(os.path.join(OUT, "voxtral_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("PHASE1 OK ->", OUT)
    print(json.dumps({k: v for k, v in manifest.items() if k != "prompt_ids"}, indent=1))


def phase2():
    import torch
    from mistral_common.protocol.instruct.chunk import AudioChunk, TextChunk
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.audio import Audio
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from vllm import LLM, SamplingParams

    man = json.load(open(os.path.join(OUT, "voxtral_manifest.json")))
    wav_path = os.path.join(OUT, "voxtral_input_16k_mono.wav")
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
    wav_f32 = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    tokenizer = MistralTokenizer.from_hf_hub(MODEL)
    audio = Audio(audio_array=wav_f32.astype(np.float32), sampling_rate=SR, format="wav")
    chunk = AudioChunk.from_audio(audio)
    req = ChatCompletionRequest(
        messages=[UserMessage(content=[chunk, TextChunk(text=man["question"])])],
        model=MODEL,
    )
    enc = tokenizer.encode_chat_completion(req)
    prompt_ids = list(enc.tokens)
    audios_and_sr = [(a.audio_array, a.sampling_rate) for a in enc.audios]

    llm = LLM(
        model=MODEL,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_model_len=8192,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.30,
        limit_mm_per_prompt={"audio": 1},
    )
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    runs = []
    for k in range(5):
        out = llm.generate(
            {"prompt_token_ids": prompt_ids, "multi_modal_data": {"audio": audios_and_sr}},
            sp,
        )
        toks = list(out[0].outputs[0].token_ids)
        txt = out[0].outputs[0].text
        runs.append(toks)
        print(f"run {k}: {len(toks)} toks :: {txt!r}")

    deterministic = all(r == runs[0] for r in runs)
    gate_form = "STRICT" if deterministic else "NEAR_TIE"
    golden_text = None
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer as MT

    golden = {
        "gate_form": gate_form,
        "deterministic_K5": deterministic,
        "output_token_ids": runs[0],
        "runs": runs,
    }
    # decode text via the tokenizer
    try:
        golden_text = tokenizer.decode(runs[0])
    except Exception as e:
        golden_text = f"<decode-err {e}>"
    golden["output_text"] = golden_text
    with open(os.path.join(OUT, "voxtral_golden.json"), "w") as f:
        json.dump(golden, f, indent=1)
    print("PHASE2 gate_form:", gate_form, "deterministic:", deterministic)
    print("golden text:", golden_text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True)
    a = ap.parse_args()
    if a.phase == 1:
        phase1()
    else:
        phase2()
