# MiniMax-H3 — omni-modal video+audio diffusion transformer

**Rows:** `MODEL-DIFFUSION-minimax-h3-mini-max-h3-dit` (model-matrix),
`ROAD-V1-H3` (roadmap portfolio).
**Claim:** `CLAIM-MINIMAX-H3-W0-W2`, `CLAIM-MINIMAX-H3-W6A-W9`.
**Upstream:** vLLM-Omni (`vllm-project/vllm-omni`), `vllm_omni/diffusion/models/minimax_h3/`.
**Checkpoint:** `MiniMaxAI/MiniMax-H3` (gated), ~354 GB, BF16 safetensors.
**Quantized checkpoints:** `realrebelai/MiniMax-H3_GGUFs` (ComfyUI GGUF),
`lilcheaty/MiniMax-H3-NVFP4` — these FIT one GB10; see section 0.
**Status:** W0 spike + W1/W2 (layout, scheduler, DiT forward incl. the bf16
production stream), W6a (request planning) and W9 shape/geometry (GGUF arm) landed,
all parity-gated. End-to-end is NOT hardware-blocked on the quantized arms; it is
gated on the remaining bricks (encoder, VAEs, pipeline).

---

## 0. Honesty statement — what is and is not claimed

MiniMax-H3 is **not an autoregressive LLM**. It is a CFG-distilled joint
video+audio **diffusion transformer**: one request runs a fixed 50-step
flow-matching denoise loop in which a 33.1B DiT is forwarded ONCE PER STEP over
the whole packed sequence, and the resulting latents are decoded to 24 FPS frames
plus a 32 kHz stereo waveform by two VAEs. There is no KV cache, no sampler, no
logits, and no token-exact gate — the SACRED near-tie methodology this project
uses for decoders does not apply to it.

**Hardware — CORRECTED 2026-08-03 (user-directed).** The BF16 release validates on
**4x NVIDIA B300** at ~133 GB peak per rank (~103 GB with text-encoder TP), and at
~354 GB of storage it does not fit one GB10 (119 GiB UNIFIED). An earlier revision
of this spec concluded from that alone that H3 e2e was "impossible on this
project's hardware". **That was wrong**: it reasoned from the BF16 release only.
QUANTIZED H3 checkpoints exist and they DO fit:

| Arm | Components | Size |
|---|---|---|
| GGUF (ComfyUI format) | DiT `MiniMax-H3-FL2VA-Q3_K_M.gguf` 15.6 GB + Qwen3-VL encoder `qwen3vl-32B-...-Q4_K_M.gguf` 14.6 GB + the two VAEs (fp16 video ~10 GB, fp32 audio ~0.6 GB) | **~41 GB** |
| NVFP4 (safetensors) | `minimax_h3_ref2va_nvfp4_{full,mixed}.safetensors` + `text_encoders/qwen3vl_32b_..._nvfp4_awq.safetensors` + `vae/minimax_h3_{video_vae_fp16,audio_vae_fp32}.safetensors` | fits, repo 77.2 GB across 3 DiT variants |

Sources: `realrebelai/MiniMax-H3_GGUFs` and `lilcheaty/MiniMax-H3-NVFP4`. Both land
well inside the 119 GiB pool, so **end-to-end H3 IS reachable here, and therefore
so is a speed comparison.** NVFP4 is the more interesting arm for this project:
GB10/sm_121 has native FP4 tensor cores and our NVFP4 stack (cutlass FP4 GEMM,
Marlin W4A16 grouped MoE, the Laguna arm's tuning) is the most optimized path we
own. What remains blocked is a like-for-like comparison against vLLM-Omni's own
published numbers, which were measured on 4x B300 — a different machine class.

**Therefore:** the CORRECTNESS gate is upstream itself executed at REDUCED
DIMENSIONS on CPU (section 4) — that is available today and is exact. The
END-TO-END gate is now a matter of finishing the remaining bricks (encoder, VAEs,
pipeline) and downloading a quantized checkpoint, NOT a hardware wall. Nothing in
THIS change claims a generated video or a speed figure; what changed is that both
are now on the critical path rather than out of reach.

## 1. Architecture

From `minimax_h3_transformer.py:47-78` (`MiniMaxH3DiTArchConfig`) — the shipped
geometry:

| Field | Value | Note |
|---|---|---|
| `num_layers` | 50 | AdaLN DiT blocks |
| `token_refiner_num_layers` | 2 | plain pre-norm blocks over text rows |
| `hidden_size` | 5376 | |
| `num_attention_heads` | 56 | **MHA** — `total_num_kv_heads == total_num_heads` |
| `attention_head_dim` | 128 | |
| `ffn_hidden_size` | 14336 | SwiGLU, fused `[gate; up]` fc1 |
| `latents_dim` | 24 | video VAE latent channels |
| `audio_latents_dim` | 32 | audio VAE latent channels |
| `patch_size` | (1, 2, 2) | video row width = 24*1*2*2 = **96** |
| `text_dim` | 5120 | H3-Encoder hidden width |
| `timestep_input_dim` | 256 | sinusoidal, **cosine before sine** |
| `time_embed_dim` | 2688 | AdaLN input |
| `adaln_out_features` | 18*5376 | 6 vectors x 3 modalities x H |
| `rope_inv_freq_len` | 16 | 3D RoPE rotates 6*16 = **96 of 128** head dims |

Per block: `norm1 -> AdaLN scale/shift -> attention -> AdaLN gated residual ->
norm2 -> AdaLN scale/shift -> SwiGLU MLP -> AdaLN gated residual`. AdaLN
parameters are produced per (unique timestep, modality) pair and selected per row
by `combined_indices = inverse_indices * 3 + token_tags.clamp(min=0)`
(`minimax_h3_transformer.py:1057`).

Attention is **packed varlen NON-CAUSAL**: rows of all modalities live in one
sequence, `cu_seqlens = {0, used, seq_len}` gives two documents (content and
64-alignment padding), and attention never crosses that boundary. This maps
exactly onto our shared `vt::DFlashBlockAttention(causal=false)` — no new kernel.

12 parameters plus the RoPE buffer stay **FP32** after load
(`minimax_h3_transformer.py:85-101`, re-asserted by `post_load_weights` at
`:898-904`): both patch projections, both time-embedder projections, and both
final output heads. Everything else is BF16.

## 2. Component inventory (the whole chain, not just the DiT)

| Component | Upstream | Size | Our status |
|---|---|---|---|
| Omni DiT | `minimax_h3_transformer.py` (1112 L) | 66.3 GB | **W2 LANDED** (CPU reference forward, parity-gated) |
| Packed layout | `packed_sequence.py` (572 L) | — | **W1 LANDED** (fl2va + ref2va, fp64 grid bit-exact) |
| Latent packing | `packed_tokens.py` (114 L) | — | **W1 LANDED** (+ round-trip) |
| Scheduler | `scheduling_..._euler_ancestral.py` (179 L) | — | **W1 LANDED** |
| Denoise loop | `denoise_loop.py` (249 L) | — | **W2 LANDED** (driver ported, not e2e-gated) |
| H3-Encoder | `encoder.py` (1214 L) | 51.5 GB | **W3 COMPLETE** — text tower at 1.2e-7 (truncation + UNNORMALIZED output + DeepStack), the FULL vision tower at 6.0e-8, and the MM processor as REUSE of our existing Qwen3-VL front end, gated on H3's own processor config |
| Video VAE | `vae.py` adapter + checkpoint REMOTE CODE (`FL2VA/video_vae/*.py`) | ~10 GB | **W4 DECODER DONE** — the FULL ViT3D decoder (pack, x_embedder, register/cls tokens, 3D RoPE, 36-block stack, norm_out, proj_out, unpatchify) is ported and gated at **8.9e-8**. Tiling and the 3D-CNN encoder (conditioning only) remain. See 5.1 |
| Audio VAE | `vae.py` adapter + checkpoint REMOTE CODE (`FL2VA/audio_vae/*.py`) | ~0.6 GB | **W5 LANDED** — DAC/BigVGAN decoder REIMPLEMENTED, gated vs the checkpoint's own modules at 4.2e-9 |
| Pipeline / tasks | `pipeline_minimax_h3.py` (1196 L) | — | **W6 t2va ASSEMBLED** — the whole path composes and runs (structural e2e gate); fl2va/ref2va conditioning and the torch-RNG noise seed remain |
| Conditioning | `condition_noise.py`, `reference_video.py`, `presentation.py`, `time_request.py` | — | **PARTIAL** — condition-noise augmentation DONE and reference-video GEOMETRY + FRAME SCHEDULE DONE; the rest of `reference_video.py` is ffmpeg plumbing (see 5.2) and presentation TOKEN TAGS are done (its tokenization stays with the caller) |
| Serving | vllm-omni `/v1/videos`, `/v1/videos/sync` | — | **W7 DONE (CPU)** — both routes plus `GET /v1/videos/{id}` registered on `ApiServer`, additive and opt-in via `set_video_runner` |
| GGUF arm (ComfyUI format) | `realrebelai/MiniMax-H3_GGUFs` | 15.6 GB (DiT Q3_K_M) | **W9 DONE** — identity name map (gated on the real 535-tensor manifest) PLUS `LoadMiniMaxH3DitFromGguf`: dequantize through the shared GGUF path, recover the geometry from shapes, bind the forward's views |
| NVFP4 arm | `lilcheaty/MiniMax-H3-NVFP4` | fits | **W10 LOADER DONE** — `LoadMiniMaxH3DitFromNvfp4` dequantizes the compressed-tensors triple through the project's existing NVFP4 path into a runnable DiT. Previously GROUNDED: — the real 1051-tensor manifest is textbook compressed-tensors NVFP4 (U8 packed + E4M3 group-16 `weight_scale` + F32 `weight_scale_2`), i.e. EXACTLY our existing layout; 258 quantized projections, islands unquantized |

Tasks: `t2va` (text), `fl2va` (first/last-frame), `ref2va` (reference). Duration
4-15 s snapped to `17n+5` frames at 24 FPS; 50 inference steps; flow shift 12
(video) / 3 (audio); resolution 1440p or 768p short edge, multiples of 32.

## 3. Dispatch and reuse — what we already have

* **Packed non-causal attention** -> `vt::DFlashBlockAttention(causal=false)`
  (already CPU + CUDA). Its per-document bidirectional contract IS upstream's
  varlen FA call; no new attention kernel is needed.
* **SwiGLU merged gate/up** -> the `layers::MlpGateUpMethodBase` merged-GEMM seam
  (AGENTS.md "born fused"). The W2 reference forward calls the projections
  directly; folding onto the seam is part of W2b.
* **Add+RMSNorm glue** -> `vt::FusedChain` recipes, W2b.
* **H3-Encoder** -> our existing `qwen3_vl_{text,vision}.cpp` +
  `multimodal/qwen3vl_processor.cpp`. The deltas are: keep only the first 50
  decoder layers and consume the UNNORMALIZED hidden state after layer 49; all-ones
  attention mask; DeepStack injection at the first `len(deepstack_visual_indexes)`
  layers. This is the single largest reuse in the port.
* **Multi-GPU** — upstream uses Ulysses sequence parallelism (`--usp 4`) plus
  optional DiT TP. We have `vt::communicator` / NCCL but no USP; this is W8 and is
  only reachable on multi-GPU hardware we do not have.

## 4. Gates

**What can be gated on any CPU (and IS, as of W2):** upstream's pure-Python
modules are imported by file path and executed at reduced dimensions;
`scripts/gen-minimax-h3-goldens.py` freezes their outputs into
`tests/vllm/models/minimax_h3_goldens.inc`, and `tests/vllm/models/test_minimax_h3.cpp`
reproduces them. Weights and inputs are rebuilt on both sides from an identical
FNV-1a + splitmix64 stream, so not one weight byte is checked in.

Landed results (`build-cpu`, Release, 10/10 test cases, 2539 assertions):

| Gate | Result |
|---|---|
| fl2va packed layout (ids, tags, positions, masks, cu_seqlens, doc ids) | **exact** |
| fl2va fp64 position grid | **bit-exact** (all 192 doubles) |
| ref2va block layout (image + video_audio reference blocks) | **exact**, incl. fp64 grid |
| patchify / unpatchify / audio pack / unpack | **exact** + round-trip identity |
| euler-ancestral eta0 scheduler + `rf_v_to_x0` | **exact** (<= 1e-6) |
| **DiT forward, reduced dims, f32** | **max abs diff 1.6e-7 (video), 1.5e-7 (audio)** |
| denoise-loop INVARIANTS (pinned rows reset every step, targets advance, finite) | pass |
| bf16 PRODUCTION stream vs upstream's dtype policy | max abs diff 2.4e-3 (bf16 scale) |
| request planning (frames, latent shapes, sigma schedules, canvas, task dispatch) | **exact** |
| **REAL GGUF manifest** (535 tensors of `MiniMax-H3-FL2VA-Q3_K_M.gguf`) | **exact** — every name and logical shape matches our contract, geometry derived from shapes alone equals the shipped H3 config |
| **AUDIO VAE decoder** vs the checkpoint's OWN remote code | **max abs diff 4.2e-9** (kaiser-sinc filter 3.0e-8) |
| **AUDIO VAE ENCODER** vs the checkpoint's OWN remote code, STAGE BY STAGE | **conv stack 2.98e-8, `pre_block` AttnProjection 1.64e-7, whole encode-to-latent 1.86e-8** — `DacAudioVAE` exposes only `decode`, so the encode is composed the way vLLM-Omni composes it (vae.py:317-325): preprocess right-pad -> `Encoder` -> `pre_block` -> `mean_proj`. Generator: `scripts/gen-minimax-h3-audio-vae-encoder-goldens.py`. `mean_proj` and never `logs_proj` — a sampled reference would condition differently every run |
| **REAL NVFP4 manifest** (1051 tensors) | **exact** — compressed-tensors triple, group 16, islands unquantized, names identical to our contract |
| **REAL video-VAE manifest** (560 tensors) | **exact** — decoder confirmed a 36-block ViT, encoder the 3D CNN |
| **VIDEO VAE decoder TransformerBlock** vs the checkpoint's OWN remote code | **max abs diff 6.0e-8** |
| **VIDEO VAE FULL ViT3D decoder** vs the checkpoint's OWN remote code | **max abs diff 8.9e-8** |
| **ENCODER text tower** (truncation + unnormalized output + DeepStack) | **max abs diff 1.2e-7** |
| **ENCODER vision block** (LayerNorm, fp32 rotary, varlen non-causal, tanh-GELU) | **max abs diff 6.0e-8** + boundary isolation proven |
| **ENCODER FULL vision tower** (patch embed -> pos interp -> 2D rotary -> blocks -> mergers), ragged 2-image batch | **max abs diff <= 1e-4**, DeepStack + merged both |
| **CONDITION-NOISE augmentation** (fl2va/ref2va anchors, visual + audio) | **exact** (<= 1e-6), noise supplied so the gate isolates row accounting from torch RNG |
| **REFERENCE-VIDEO geometry + frame schedule** | **exact** (canvas, sampled indices, block timestamps) |
| **VIDEO VAE tiling plan + seam blend** | **exact** (round-robin slack distribution, cross-fade) |
| **PRESENTATION token tags** (the fl2va vision-span override) | **exact**; VIDEO runs proven to be whole vision blocks |
| **VAE encoder ResnetBlock3D** (causal Conv3d + GroupNorm3D) | **exact**; causality PROVEN on the bare convolution |
| **VAE encoder Downsample3D** (asymmetric pre-pad + strided causal conv) | **exact** |
| **WHOLE VAE 3D-CNN encoder** (conv_in -> levels -> norm -> conv_out) | **exact** |
| **MM PROCESSOR reuse** (H3's own processor config through our Qwen3-VL front end) | **pass** — image + video bounds, 0.5 normalization, 32-grid |
| **WAV serialization** of the decoded waveform | **pass** — header fields, channel-major -> interleaved, clamping |
| **VIDEO OUTPUT: PPM frames + MP4 mux argv** | **pass**; and the built argv was RUN through real ffmpeg 6.1.1, producing a valid h264/yuv420p + AAC-32kHz MP4 (ffprobe-verified) |
| **`/v1/videos` request contract + job store** | **pass** — defaults, validation, lifecycle transitions, status JSON, thread safety |
| **`/v1/videos` route dispatch on `ApiServer`** | **pass** — no-runner 500, unknown-id 404, sync success returns the runner's path, a throwing runner fails the job (async worker never terminates the process), malformed body 400 without reaching the runner |
| **DEVICE-RESIDENT DiT forward** (CPU backend) | **pass** — same goldens, same 2e-5 tolerance as the CPU reference |
| **DEVICE-RESIDENT DiT forward on a REAL GPU** (Thor, sm_110) | **pass — video 1.49e-7 / audio 8.94e-8** vs upstream; 36/36 cases, and the CUDA case is proven to have RUN (220 assertions execute, not skip) |
| **AUDIO-VAE CHECKPOINT LOADER** (real 1087-tensor manifest) | **pass** — and it caught TWO silent-failure mismatches: the shipped file uses torch's LEGACY `weight_g`/`weight_v`, not the `parametrizations.weight.original0/1` the decoder reads, and BigVGAN sits under `decoder.` while `dec_in_proj.*` is top level. Mapping asserted INJECTIVE over the real manifest (2770 assertions) + an end-to-end load-and-DECODE over a synthetic file written in the shipped spellings |
| **Audio-VAE loader accepts ALL THREE weight-norm spellings** | **pass** — (1) LEGACY `weight_g`/`weight_v` (official checkpoint), (2) MODERN `parametrizations.weight.originalN` (what the decoder reads), (3) MATERIALIZED plain `weight` (repackaged community bundles). The third is reconstructed exactly, round-trip **1.49e-08** |
| **AUDIO-VAE ENCODER LOADER** (same real 1087-tensor manifest) | **pass** — takes the half the decoder loader skips: strip `encoder.`, keep top-level `pre_block.*`/`mean_proj.*`, drop `logs_proj.*`, and accept all three weight-norm spellings (the materialized one reconstructed to <=1e-6 round-trip). The manifest also confirms the SHIPPED encoder geometry from shapes alone: `encoder_rates` [2,4,4,5,5], `latent_dim` 2048, `attn_proj_dim` 32, qkv 3x the INPUT width (the narrowing AttnProjection branch). A plain Linear's `.weight` must NOT be mistaken for a materialized weight-norm, which is asserted |
| **ref2va AUDIO + VIDEO+AUDIO references WIRED** | **pass** — the last two unwired conditioning modes. Gated on conditioning CHANGING the result: an audio reference moves the AUDIO rows by **0.51**, a video+audio reference by **0.71** against the SILENT same-clip control, a DIFFERENT waveform still by **7.1e-4**, and a DIFFERENT clip with the same audio moves the VIDEO rows by **3.7e-2**. An audio-bearing block with no encoded rows behind it THROWS. Driver `--ref-audio f.wav` on a library `MiniMaxH3ReadWav`, gated against the writer it inverts and REFUSING a non-32 kHz file |
| **VIDEO-VAE CHECKPOINT LOADER** (real 560-tensor manifest) | **pass** — mapping is just the `decoder.` prefix (no weight-norm spelling change), asserted INJECTIVE. ★ Surfaced a MISSING STEP: `post_quant_conv` (Conv3d 24->24, kernel 1x1x1) sits OUTSIDE `ViT3DDecoder`, so the 8.9e-8 decoder gate never covered it and NOTHING in this port applied it — a decode that runs, looks plausible and is wrong. Now implemented, gated against a hand-computed contraction, AND wired into `MiniMaxH3GenerateT2va` — the pipeline test re-runs t2va with it present and requires the frames to move (0.056) while the waveform stays bit-identical |
| **ENCODER CHECKPOINT LOADER** (FL2VA/text_encoder, 14 shards / 1058 tensors) | **pass** — the only loader that TRANSFORMS rather than renames: HF ships `self_attn.{q,k,v}_proj` and `mlp.{gate,up}_proj` SEPARATE, the port (like vLLM) consumes them FUSED, so they are row-concatenated as `[q\|k\|v]` and `[gate\|up]`. Gated byte-exact ACROSS SHARDS (one layer deliberately split between two files), plus layer truncation, plus the H3 deltas: `norm.weight` and `lm_head` are NOT loaded, because H3 reads the UNNORMALIZED truncated output. The VISION tower needs no fusion — HF already ships `attn.qkv` fused |
| **ASSEMBLY driver** (`examples/minimax-h3-gen`) | **pass (LOAD + PLAN)** — composes the DiT + both VAEs + both shipped configs, over real file formats, on both the dequant and keep-quant GGUF paths. Shape planning verified: 768x1344 / 16 = 48x84 latent. ★ A full generation on a REAL checkpoint is still UNRUN (needs the multi-GB download) |
| **DEVICE-RESIDENT bf16 PRODUCTION stream** (CPU + Thor GPU) | **pass — video 2.41e-3 / audio 2.05e-3** vs the bf16 goldens (tol 5e-3), essentially the CPU reference's own 2.4e-3 / 2.1e-3, so the CAST POINTS agree; the test also asserts the bf16 result DIFFERS from f32 by >1e-5, without which a no-op dtype policy would pass |
| **TRUE bf16 STORAGE** (activations AND weights) | **pass — video 6.16e-4 / audio 5.21e-4**, a ~4x IMPROVEMENT on the round-in-place figures above. Cause: the bf16 golden was generated with bf16 WEIGHTS (the generator's `to_bf16_weights`), so staging the bf16-stored modules as bf16 — while keeping upstream's fp32 ISLANDS f32 — matches the golden's model, not just its activation cast points |
| **WHOLE t2va PATH composes** (layout -> sigmas -> denoise loop -> unpack -> denormalize -> both VAEs) | frames + stereo waveform, correctly shaped, finite, in [-1, 1] |
| **GGUF LOAD -> runnable DiT** (synthetic ComfyUI-format file) | geometry recovered from shapes; a real forward runs off the loaded weights |
| **NVFP4 LOAD -> runnable DiT** (synthetic compressed-tensors file) | packed [out, in/2] recovered as logical [out, in]; sidecars excluded; a real forward runs |
| config-parse invariants + weight contract + grouped-qkv reorder | pass |

The fp64 position grid is gated bit-exact deliberately: it feeds RoPE, and a
last-ulp drift would silently rotate every video token. The port therefore
reproduces upstream's arithmetic ORDER — `numpy.linspace(endpoint=False)`
evaluates `i*step + start`; `_temporal_position_span` uses numpy PAIRWISE
summation while `_video_t_span` uses Python's SEQUENTIAL `sum()`, which upstream
keeps separate on purpose (`packed_sequence.py:101-113`).

**What cannot be gated here:** any end-to-end video/audio result, any speed
number, the encoder/VAE numerics (no checkpoint), and the multi-GPU USP path. All
are recorded PENDING in `docs/BENCHMARKS.md`, not as passes.

**Reference audio, still ungated:** no real-checkpoint render with `--ref-audio` has been run. The encoder numerics, the loader mapping and the wiring are all gated; what is not is a full generation conditioned on a real waveform, which needs the multi-GB download and a GPU.

**Oracle note.** The parity pin (`555967922`, vLLM 0.26.0.dev0) does NOT contain
MiniMax-H3 — H3 was released after it, and it lives in the separate `vllm-omni`
repository, which the pin protocol does not currently cover. Advancing the pin
does not by itself make H3 gateable; a vllm-omni pin is a prerequisite for W3+ and
is tracked as an open item in 7.

## 5. Known hard parts

### 5.1 The VAEs are REMOTE CODE, not upstream Python

`vae.py:41-53` loads both VAEs with
`get_class_from_dynamic_module(config["auto_map"]["AutoModel"], component_path)` —
i.e. the actual VAE implementations ship INSIDE the HF checkpoint and run under
`--trust-remote-code`. vLLM-Omni only adapts them. A pure-C++ engine cannot do
that: W4/W5 must **reimplement both VAEs in C++ from the checkpoint's Python
source**, which must be fetched separately (the VAE modules and their `config.json`
are small; the 354 GB of weights are not needed to READ the architecture).

**Status 2026-08-05: BOTH VAEs are DONE IN BOTH DIRECTIONS.** Decoders: audio
(DAC/BigVGAN, 4.2e-9) and the video ViT3D (8.9e-8). Encoders: the video 3D CNN
(image/video conditioning) and now the AUDIO encoder — the DAC analysis stack
plus `pre_block` and `mean_proj`, gated stage by stage at 2.98e-8 / 1.64e-7 /
1.86e-8. That was the last thing standing between ref2va and its audio-bearing
reference blocks, which are now wired and gated on moving the result.

**Original note: the remote code is IN HAND** (fetched from the checkpoint's
`FL2VA/{audio,video}_vae/`, ~130 KB of Python, NOT vendored here — it ships under
the MiniMax H3 Community License). The **audio VAE is DONE** (W5): a DAC-lineage
BigVGAN vocoder, reimplemented and gated against the checkpoint's own modules at
4.2e-9 by `scripts/gen-minimax-h3-audio-vae-goldens.py`. The **video VAE (W4)** is the largest remaining brick, but the real
checkpoint manifest (560 tensors, `FL2VA/video_vae/source/model.safetensors`,
captured by range request) makes it materially smaller than `klvae.py`'s 48 KB
suggested: the **ENCODER** is the 3D CNN (116 tensors, rank-5 Conv3d down blocks)
while the **DECODER** — the half generation actually needs — is a plain **36-block
TRANSFORMER** (440 tensors: `attn.to_qkv`/`attn.to_out`, `ff.w1`/`ff.w2`, two
norms and two learned residual scales per block, plus `x_embedder`, `mask_token`,
`register_tokens`, `norm_out`, `proj_out`). We have every primitive for that. The
whole checkpoint is fp32.

Contracts already pinned down from the adapter:
* Video VAE weights stay **FP32**; keyframe encode is seeded
  (`MINIMAX_H3_KEYFRAME_ENCODE_SEED = 42`) and its normalize+patchify runs on
  **CPU in FP32** on purpose (`vae.py:185-202`) — doing it on CUDA measurably
  changes the conditioned video.
* Latents are normalized by per-channel `latents_mean`/`latents_std` from the
  component `config.json`, then patchified with (1,2,2).
* Audio VAE is FP32 for both encode and decode, 32 kHz, 2 channels, and encode
  runs under a determinism context that disables TF32, cuDNN, and the fused SDP
  backends (`vae.py:56-94`) — the C++ port must match that numerically, not just
  structurally.

### 5.2 Output is a container, not tokens

`/v1/videos` returns MP4 (H.264 video + stereo audio). We have no muxer and no
video/audio ENCODER anywhere in the tree (`third_party/` has blake3, doctest,
httplib, minja, nlohmann, vulkan). W7 must choose: vendor a minimal MP4 muxer plus
an encoder, or take a dependency. This is a genuine new dependency decision and is
called out rather than assumed.

### 5.3 Speed

Upstream reports the DiT at **88% of request latency** and FL2VA at ~87 s E2E for
an 8.7 s 1248x768 clip on 4x B300, with regional `torch.compile`, cache-dit block
caching, and USP-4. Matching that needs the device-resident forward (W2b), the
fusion folds, and multi-GPU. No speed claim is possible before W2b lands and
hardware exists to measure on.

## 6. Files ported in this change

| Ours | Upstream |
|---|---|
| `include/vllm/model_executor/models/minimax_h3.h` | the module's public contracts |
| `src/vllm/model_executor/models/minimax_h3_packing.cpp` | `packed_tokens.py`, `packed_sequence.py`, `scheduling_..._euler_ancestral.py` |
| `src/vllm/model_executor/models/minimax_h3.cpp` | `minimax_h3_transformer.py`, `denoise_loop.py` |
| `scripts/gen-minimax-h3-goldens.py` | executes the above upstream modules as the oracle |
| `tests/vllm/models/test_minimax_h3.cpp` | `tests/diffusion/models/minimax_h3/test_minimax_h3_{packing,contract}.py` |

**Tests to port (upstream `tests/diffusion/models/minimax_h3/`):**
`test_minimax_h3_packing.py` (DONE — layout + patchify goldens),
`test_minimax_h3_contract.py` (PARTIAL — config/weight contract done, pipeline
contract pending W6), `test_minimax_h3_e2e.py` (BLOCKED — needs the checkpoint),
`test_minimax_h3_parallel.py` (BLOCKED — needs multi-GPU).

## 7. Work breakdown

| Brick | Scope | Blocked by |
|---|---|---|
| **W0** | Spike, component inventory, hardware verdict | — (DONE) |
| **W1** | Packed layout + latent packing + scheduler, parity-gated | — (DONE) |
| **W2** | DiT forward + denoise driver, parity-gated on CPU at reduced dims | — (DONE) |
| **W2b** | Device-resident forward. **LANDED (f32) and VERIFIED ON A REAL GPU** — `MiniMaxH3DitForwardDevice` keeps every activation in device memory across the whole block stack; gated against the SAME upstream goldens on the CPU backend AND on a Thor sm_110 GPU at **video 1.49e-7 / audio 8.94e-8** (tolerance 2e-5). Only 3 H3 kernels were needed (`kMiniMaxH3` table: two indexed AdaLN modulates + ungated SiLU) because the port reuses the tuned shared ops — H3's 3-axis RoPE is plain NeoX rotate_half, so a per-row cos/sin cache feeds `vt::RopeFromCache` with no bespoke kernel. **bf16 PRODUCTION stream LANDED, and then upgraded to TRUE bf16 STORAGE**: activations are bf16 buffers and the bf16-stored modules are staged as bf16 weights (fp32 islands preserved), so the tuned shared ops run their native bf16 paths and activation bytes halve. It is also MORE accurate — 6.16e-4 vs 2.41e-3 — because the golden itself used bf16 weights. That also unlocked the refiner's add+RMSNorm fold onto `vt::kFusedAddRmsNormStd`, previously declined because it would have dropped a cast point; with a bf16 residual the add rounds on store, so the fold is byte-identical. REMAINS: `vt::FusedChain` glue folds, merged gate/up seam, and the FP4 path (which needs sm_121a — PROBED 2026-08-03: the warp-level `mma.sync kind::mxf4nvf4` is CONSUMER-Blackwell only, rejected by ptxas on both sm_110a and sm_100a; sm_110 does support the datacenter `tcgen05` family, but our sm_100 body is CUTLASS ArchTag=Sm100 guarded by `__CUDA_ARCH__ == 1000` — so retargeting compiles to a DEAD STUB — and CUTLASS has **zero** sm110 kernels even at v4.6.1, only capability macros. There is no upstream body to port, so Thor can never be the FP4 venue) | — |
| **W3** | H3-Encoder. **TEXT TOWER DONE** (1.2e-7): the three H3 deltas — layer truncation `min(num_hidden_layers, 50)`, the UNNORMALIZED layer-49 output (no final RMSNorm), and DeepStack injection into the first N layers — plus interleaved M-RoPE, fused QKV, per-head q/k RMSNorm, causal GQA and the gated-SiLU MLP. **VISION BLOCK also DONE** (6.0e-8): LayerNorm-with-bias, the [q_all, k_all, v_all] qkv layout, fp32 rotary, cu_seqlens-segmented NON-CAUSAL attention (boundary isolation asserted), and the TANH-approximate GELU. REMAINS: the vision surround (Conv3d patch embed, learned pos-embed interpolation, 2D rotary table, patch mergers + DeepStack mergers) and the MM processor | — |
| **W4** | Video VAE. **DECODER DONE** — the full ViT3D decoder gated at 8.9e-8 (block 6.0e-8), real hyperparameters 36 layers / 32 heads x 64 / rope_theta 100 / rope_dim_ratio 0.75 from the checkpoint's `vit_decoder_kwargs`. **TILING also DONE** (plan + seam blend, exact). **3D-CNN ENCODER primitives also DONE** (causal Conv3d with reflect spatial padding, GroupNorm3D, ResnetBlock3D). **DONE — encoder AND decoder both complete.** The 3D-CNN encoder (conv_in, per-level ResnetBlock3D + Downsample3D, norm_out, conv_out) is gated exact; it serves image/video CONDITIONING, which a t2va path does not need | — |
| **W5** | Audio VAE reimplementation | **DONE** — DAC-lineage BigVGAN decoder (weight-norm materialization, anti-aliased SnakeBeta with kaiser-sinc up/down resampling, replicate padding, final clamp). Encode-side determinism context is still open |
| **W6** | Pipeline. **t2va ASSEMBLED** — `MiniMaxH3GenerateT2va` wires layout -> sigma schedules -> denoise loop -> unpatchify/audio-unpack -> denormalize -> both VAE decoders, gated by a structural end-to-end test. REMAINS: fl2va/ref2va conditioning (condition noise, reference video, presentation) and bit-exact torch-RNG noise seeding | — |
| **W7** | Serving. **DONE (CPU)**: PPM frames + WAV + the MP4 mux argv (validated end-to-end against real ffmpeg); the `/v1/videos` request contract and job store (lifecycle, status JSON, thread-safe); and the routes themselves — `POST /v1/videos` (async, joinable worker drained in `~ApiServer`), `POST /v1/videos/sync`, `GET /v1/videos/{id}` — registered ONLY when `set_video_runner` has been called, so a server without video support is byte-identical to before. The runner is a caller-supplied callback precisely because the ffmpeg invocation lives in `examples/` per the developer's ratified decision: `src/vllm/` never spawns a process | W6 |
| **W8** | Speed: USP sequence parallelism, block caching, DiT TP | W2b + multi-GPU HW |
| **W9** | **GGUF arm — DONE.** Identity name map, `ne` reversal, the `comfy.gguf.orig_shape` reshape rule, and `LoadMiniMaxH3DitFromGguf` (shared K-quant dequant -> owned f32 -> bound views), gated by the real 535-tensor manifest plus a synthetic-file load-and-run test | — |
| **W10** | **NVFP4 arm** — `lilcheaty/MiniMax-H3-NVFP4` onto our existing NVFP4 stack (cutlass FP4 GEMM on sm_121). **LOADER DONE** — `LoadMiniMaxH3DitFromNvfp4` reuses the project's existing NVFP4 dequant, so no new quant code. REMAINS: the DEVICE path that keeps FP4 packed and routes projections through the cutlass FP4 GEMM (that is where the speed is), plus a run on the real file | W9 |

**Open items.** (0) Run the assembled t2va path on a REAL quantized checkpoint — the
pipeline now composes end to end at reduced dimensions, so what remains is loader
wiring (W9 dequant / W10 NVFP4), the encoder's vision tower, and a GPU. This
supersedes the old "hardware-blocked" framing. (0b) Noise seeding is currently an
INPUT: upstream seeds a torch CPU generator, and matching it bit-exactly decides
WHICH sample you get, not whether the pipeline is correct.
(a) A vllm-omni parity pin — the upstream-sync protocol currently
covers only the vLLM repo; H3 lives outside it. (b) The MP4 dependency decision.
(c) Hardware: nothing past W2b/W3 can be END-TO-END gated on this project's boxes,
so W4-W8 should be reviewed as structural ports with unit gates, and the honest
lifecycle cap for this row is "correctness-complete, hardware-blocked".

## 8. W-FP4 — the fp4 SPEED path (row `row/H3-FP4-SPEED`, 2026-08-06)

Until this change the NVFP4 arm ran the DiT projections in **bf16**: both the
reference loader (`LoadMiniMaxH3DitFromNvfp4`) and the streaming stager
(`StreamMiniMaxH3Nvfp4ToDeviceBf16`) DEQUANTIZE every packed FP4 weight to bf16 and
the device forward calls `vt::MatmulBT`. The sm_121a FP4 tensor-core route had
never actually run for H3. W-FP4a wires it.

### 8.1 W-FP4a — per-shape routing table (grounded in `dense_nvfp4_gemm.h`)

The `lilcheaty/MiniMax-H3-NVFP4` checkpoint is **weight-only NVFP4 (W4A16)**: every
quantized projection carries only `weight` (U8 E2M1) + `weight_scale` (E4M3, group
16) + `weight_scale_2` (F32) and **no `input_activations`** (confirmed
`minimax_h3_nvfp4.cpp:63-76` and the real manifest, spec §4). Per the dispatcher's
own contract (`dense_nvfp4_gemm.h:12-22`, mirroring vLLM
`kernels/linear/__init__.py:879-881` — *"Force a16 (Marlin) when running
weight-only quantization"*), a W4A16 weight (`Nvfp4Weight::IsTrueW4A4()==false`,
alpha==0) is **forced to the Marlin W4A16 grouped GEMM**, bypassing the
capability-based kernel registry. So on sm_121a **every quantized H3 projection
takes the SAME kernel** — `dense_nvfp4::MatmulNvfp4MarlinD` (single-expert
`vt::MoeGroupedGemmNvfp4Marlin`), the exact path the Laguna routed-experts
(`laguna.cpp`) and the dense Qwen3-32B NVFP4 arm (`qwen3_5.cpp`) use.
**MEASURED CORRECTION (GB10, 2026-08-06):** the PRODUCTION default is one level up —
`VT_MARLIN_DENSE` is default-ON, so `MatmulNvfp4W4A16D` routes each projection through
vLLM's OWN **dense** Marlin GEMM (`vt::MarlinDenseGemm`, counter `dense_gemms`), and
the grouped `MoeGroupedGemmNvfp4Marlin` (`marlin_gemms`) is taken only under
`VT_MARLIN_DENSE=0`. Both are Marlin W4A16 and both are byte-exact to the bf16 arm;
dense is marginally faster. See §8.4 + the benchmark record. The
cutlass-FP4 / true-W4A4 route (`MatmulNvfp4Fp4D`) is NOT taken here: it needs fp4
ACTIVATIONS this checkpoint does not carry, and is deliberately private to
`qwen3_5.cpp` (`dense_nvfp4_gemm.h:12-18`).

Real geometry: H=5376, ffn=14336, heads=56×128 (inner=7168), time_embed_dim=2688,
text_dim=5120, adaln_out=18·H=96768, final_adaln=2·H=10752.

| Projection (per layer unless noted) | `[N, K]` | Route | Why (file:line) |
|---|---|---|---|
| `attn.qkv_proj` | `[21504, 5376]` | **Marlin W4A16** | W4A16 forced-Marlin `dense_nvfp4_gemm.h:512-525` |
| `attn.out_proj` | `[5376, 7168]` | **Marlin W4A16** | same |
| `mlp.fc1` (merged `[gate;up]`) | `[28672, 5376]` | **Marlin W4A16** → `SiluAndMul` | fc1 is ALREADY merged, so ONE GEMM to `[M,2·ffn]` then `vt::SiluAndMul`; the fused-pair `GateUpFusedMarlinD` does NOT apply (no separate gate/up shards) — `minimax_h3_device.cpp` `MlpDev` |
| `mlp.fc2` | `[5376, 14336]` | **Marlin W4A16** | same |
| `adaln_proj.linear` (block) | `[96768, 2688]` | **Marlin W4A16** (+ bias `vt::Add`) | skinny-M (M=num_unique_timesteps): Marlin `block=8` dense tile at M≤8, `dense_nvfp4_gemm.h:273-286` |
| `condition_proj` | `[5376, 5120]` | **Marlin W4A16** (+ bias) | embed, once |
| `final_layer.adaln_proj.linear` | `[10752, 2688]` | **Marlin W4A16** (+ bias) | final, once |
| refiner `qkv/out/fc1/fc2` (×2) | same as block | **Marlin W4A16** | refiner has no adaln |
| **islands** (`video/audio_patch_proj`, `time_embedder.*`, `final_layer.{video,audio}_out`) + all norms/biases | — | **`vt::MatmulBT` bf16 / f32 (unchanged)** | fp32 island policy (`minimax_h3_transformer.py:85-101`); never quantized in the checkpoint |

All N and K are multiples of 16 (group) and of 128 (Marlin tile) — no shape blocker.
The activation MUST be bf16 for the Marlin path; in the bf16 production stream it is,
so the fp4 arm pairs with the bf16 stream. In the f32 parity stream (or any backend
without the Marlin op — e.g. CPU) the SAME dispatcher falls back to a
redundant-dequant GEMM, so the arm is correct either way, fast only where Marlin is
realized (kCUDA sm_121a).

**Implementation (this change, NO new quant code):** `Nvfp4Weight` fp4 carriers on
`MiniMaxH3DitBlockWeights`/`MiniMaxH3DitWeights`; a new fp4-resident streamer
`StreamMiniMaxH3Nvfp4ToDeviceFp4` (keeps packed FP4 host-resident, dispatcher
uploads + repacks lazily on first forward, then frees the fp4 originals — peak
device memory ~1/4 of the bf16 arm: ~16 GB packed vs ~66 GB bf16); `LinearDev`
routes a non-Empty fp4 weight through `dense_nvfp4::MatmulNvfp4W4A16D`.

**Gate.** CPU: `test_minimax_h3` "an NVFP4 checkpoint loads into a runnable DiT" now
also streams the fp4 twin, asserts the loader kept the projections PACKED (fp4 slot
set, bf16 slot Empty), runs the fp4 and bf16 device forwards on the SAME synthetic
NVFP4 file, asserts the W4A16 dispatcher executed all 11 quantized GEMMs (the
"this-path-ran" counter), and bounds the fp4-vs-bf16 delta. On CPU the dispatcher
has no Marlin op so it falls to the bf16 arm's own dequant+matmul — this is a
**wiring** gate here. The Marlin kernel's real numeric behaviour is CUDA-gated
independently by `test_ops_nvfp4_matmul` / `test_linear_method` (2e-3/8e-3 vs a
bf16 reference). **GB10 leg (fp4-vs-bf16 numeric delta + per-step timing at real
geometry): PENDING** — see §8.3.

### 8.2 Supports-audit vs vLLM-Omni (source-pinned to `a4ea67a2`, v0.26.0)

vLLM-Omni H3 modules at `vllm_omni/diffusion/models/minimax_h3/`; serving in
`vllm_omni/entrypoints/openai/`.

| Capability | vLLM-Omni (file:line) | Ours (file:line) | Verdict |
|---|---|---|---|
| Async video route `POST /v1/videos` | `api_server.py:3146` | `ApiServer` `/v1/videos` (W7, `minimax_h3` serving) | **DONE** |
| Sync route `POST /v1/videos/sync` | `api_server.py:3189` | `/v1/videos/sync` | **DONE** |
| Status `GET /v1/videos/{id}` | `api_server.py:3305` | registered | **DONE** |
| List / DELETE / `/content` download | `api_server.py:3268,3333,3385` | — | **MISSING** (list/delete/content-GET) |
| WebSocket `/v1/video/chat/stream`, `/v1/realtime/video` | `api_server.py:1593,1610` | — | **MISSING** (streaming/realtime) |
| Request schema (prompt, size/w/h, num_frames, fps, seed, steps, refs) | `protocol/videos.py:97-249` | request contract (W7) | **PARTIAL** (core fields; frame-interp/lora/generate_sound absent) |
| H3 knobs via `extra_params.{task,duration,flow_shift,audio_flow_shift}` | `pipeline:1034,403,1157-1158` | planner reads task/duration/shift | **DONE** |
| Modalities in: text/image/video/audio | `pipeline:1036-1104` | t2va (text) done; vision tower LOADS real `visual.*` + runs; merged→prompt_embeds scatter + DeepStack→device text tower WIRED 1:1 + gated (§8.9); fl2va COHERENT via BOTH the VAE-keyframe AND the encoder vision path; ref2va reference-row assembly FIXED + gated (§8.10, the block-dim double-division) — but ref2va still grids, now RE-ATTRIBUTED to the ref2va NVFP4 CHECKPOINT/loader (t2va-zero-assembly grids too), NOT the assembly | **PARTIAL** (vision→conditioning scatter + ref2va assembly DONE; residual = the NVFP4 DiT loader for the ref2va checkpoint, §8.10) |
| Output: joint video+audio, 24 fps, 32 kHz stereo | `pipeline:106-111,1187` | frames + WAV + MP4 mux (W7) | **DONE** |
| Scheduler: euler-ancestral rectified flow (single) | `scheduling_...euler_ancestral.py`; `time_request.py:34-61` | `MiniMaxH3EulerEta0Step` / `MiniMaxH3TimeShiftSigmas` | **DONE** |
| CFG: distilled, no CFG (guidance params accepted+ignored; `cfg_parallel_size==1`) | `pipeline:250,275-276` | no CFG branch | **DONE** (matches) |
| Res/frame bounds: mult-32, aspect 1:4–4:1, 17n+5 frames, 24 fps, canvas 768×1344 | `pipeline:399-430`, `time_request.py:5-31` | request planner (17n+5, canvas, sigma) EXACT | **DONE** |
| Task dispatch t2va/fl2va/ref2va | `pipeline:374-391` | planner dispatch (W6a) | **DONE** |
| Single-GPU serving | `--num-gpus 1 --enable-cpu-offload` (`recipe:53-74`) | single GB10, quantized-resident | **DONE (ours needs no offload — quantized fits)** |
| USP / DiT-TP / VAE patch-parallel | `pipeline` collectives (throughput) | `vt::communicator`/NCCL present, USP not ported (W8) | **MISSING** (multi-GPU only) |

### 8.3 Speed statement + comparability verdict (mission #3)

**vLLM-Omni CANNOT serve a quantized H3 on one GPU** (source-pinned, spec-audited):
it is **BF16-only in practice**. The generic diffusion framework has ModelOpt
FP8/NVFP4 plumbing and the H3 DiT forwards a `quant_config` to vLLM quant-capable
linears, but (i) **no quantized H3 checkpoint exists or is referenced** anywhere in
the repo; (ii) the fp32-island guard `post_load_weights()` (`minimax_h3_transformer.py:898-904`)
**raises** if the patch/time/output layers are not fp32, so a naive blanket quant
aborts; (iii) the **text encoder is hard-coded bf16** (`encoder.py:930`, no
quant_config) and the **VAEs load unquantized**; (iv) GGUF is **not wired into H3's
bespoke `load_weights`** at all. Single-GPU IS supported — but as **BF16 +
`--enable-cpu-offload`** (`recipe:53-74`).

**Therefore the comparison is HW/loader-FORCED-INDIRECT** (the DeepSeek-GGUF
precedent): a like-for-like quant-matched vllm-omni run on one GB10 is impossible
because vllm-omni has no quantized H3 arm. The honest baselines are our own bf16
arm (`StreamMiniMaxH3Nvfp4ToDeviceBf16`) and the portable path; vLLM-Omni's own best
published numbers are **4× B300 BF16**.

**Honesty correction on the "88%":** the *"DiT ≈ 88% of request latency on 4× B300"*
figure is **NOT documented anywhere in the vllm-omni checkout** (exhaustive grep).
The real documented anchor is the recipe's *"Validated four-GPU evidence"*
(`recipes/MiniMaxAI/MiniMax-H3.md:298-311`): FL2VA 209-frame 1248×768 = **86.964 s**
mean client latency on 4× B300; two-video Ref2VA 362-frame = **784.394 s**. The DiT
`diffuse` stage share is measurable per-request (`pipeline:255-262`) but no fixed
percentage is written down. The upstream **reference config** is 50 steps, 24 fps,
video flow_shift 12 / audio 3, no CFG; default canvas **768×1344**, default frames
**209** (t2va/fl2va) / **124** (ref2va) — NOT the "864×480 / 124" in the task brief.

### 8.4 Status (this row)

- **W-FP4a: CPU-LANDED + gated** — fp4-resident loader + Marlin-W4A16 routing + the
  fp4-vs-bf16 wiring gate. No new quant code.
- **W-FP4a GB10 leg: LANDED (2026-08-06, `row/H3-FP4-GPU-E2E`).** A dedicated CUDA
  case `minimax_h3: the NVFP4 fp4 forward runs Marlin W4A16 on CUDA (speed)` (the
  existing "loads into a runnable DiT" case runs the forwards on a CPU queue, so it
  could never bump the GPU counter) builds the synthetic NVFP4 file at REAL geometry
  and runs both arms on a CUDA queue. **Marlin RAN:** default `dense_gemms==11`
  (VT_MARLIN_DENSE is default-ON → vLLM's OWN dense Marlin GEMM, NOT the grouped
  route §8.1 assumed), `marlin_gemms==11` under VT_MARLIN_DENSE=0, `fallback_gemms==0`
  in both. **fp4-vs-bf16 delta = 0 (byte-exact).** **Timing crossover** (median/12,
  cold discarded): per-forward ratio bf16/fp4 = 3.47× @seq64 (fp4 faster,
  memory-bound), 0.825× @seq4224, 0.788× @seq7040 (fp4 slower, compute-bound). So
  fp4 W4A16 is a WEIGHT-BANDWIDTH win (decode-like small M) and a ~1.2× LOSS in H3's
  large-M diffusion forward; its H3 value is MEMORY (~16 GB vs ~66 GB bf16). Benchmark
  record has the full per-GEMM tables.
- **W-FP4b real-checkpoint t2va e2e: RUNS on real weights; frame COHERENCE is an open
  bug.** dgx now has room; the real NVFP4 DiT (`minimax_h3_ref2va_nvfp4_full`,
  18.75 GB, unpruned) + both VAEs + the GGUF Qwen3-VL-32B encoder were downloaded and
  the WHOLE t2va chain runs with `--fp4-resident` (new driver flag → the fp4-resident
  streamer, ~16 GB device vs ~66 GB bf16): encoder → [16,5120] text conditioning →
  fp4-resident DiT → both VAEs → ffmpeg, producing a valid `h264 256×256 + AAC 32 kHz`
  mp4 + wav. **But the decoded frame is a structured multicolour patch-grid at the
  latent-cell scale, NOT a coherent scene — identically at 12/20/50 steps, conditioned
  or not.**
  - **ROOT-CAUSED (2026-08-06, `row/H3-RENDER-COHERENCE` PR #70) by latent
    bisection:** the VAE decoder is **CORRECT** — a real image encode→post_quant_conv→
    decode round-trip (`--roundtrip`) returns a coherent frame — and the denoise loop
    moves the latent step-dependently (byte-different finals at 3/12/50 steps). The bug
    is the **DiT forward emitting a spatially-WHITE latent** at real geometry: adjacent
    latent-cell cosine is **0.06** vs **0.789** for a real encoded latent, so every VAE
    token decodes an independent patch = the grid. NOT fp4 (bf16 equally white), NOT the
    attention kernel (MMA≡chunk, VAE chunk≡warp≡keylane), NOT the init noise. The DiT
    gate runs spatial 2×3 (matches upstream 1.6e-7); the divergence is real-geometry
    only (2×3→8×8). Secondary: driver used uniform init noise, not Gaussian
    (`VT_H3_GAUSSIAN_NOISE`). Exact DiT line pends an upstream-oracle diff at real
    geometry. See the benchmark record + state entry.
  - So the composed path is proven to RUN e2e on the real checkpoint, but a
  coherent render is an OPEN bug (device video-VAE decode and/or denoise convergence
  at real geometry), independent of the fp4 speed work. **DiT s/step (full 50-layer
  fp4-resident, per forward):** 5.45 s @512×512/22f, 20.03 s @768×768/61f, 209.09 s
  @768×1344/209f (the vllm-omni REF canvas). The REF canvas fits in the pool but a full
  50-step render is ~2.85 h, so it was not run (largest-fitting-config honesty).
- **Comparability (mission #3):** HW/loader-forced-INDIRECT. 4× B300 BF16 renders a
  whole 50-step FL2VA 209f in 86.964 s (~1.8 s/forward-equiv); one GB10 fp4-resident
  is 209 s for ONE forward at the comparable canvas (~116× per-forward) — 4 datacenter
  GPUs + BF16 + USP-4 + torch.compile + block-caching vs one GB10 + fp4 + none, and
  vLLM-Omni cannot serve a quantized H3 on one GPU at all. The honest same-box number
  is the fp4-vs-bf16 ratio (0.79–0.83× per forward, 4× less weight memory).
### 8.5 DiT-forward GEOMETRY LADDER — the #70 spatial-mixing hypothesis REFUTED (2026-08-06, `row/H3-DIT-SCALE-GATE` PR #74, CPU-only)

The §8.4 render bug (#70) was root-caused to the DiT emitting a spatially-WHITE latent
at real token geometry, with the DiT parity gate only ever run at spatial 2×3. This row
tested the leading hypothesis — a spatial-MIXING bug in the position/packing/modulation
MATH, reproducible with random weights at real TOKEN geometry — by extending the
reduced-dim DiT gate into a GEOMETRY LADDER.

- **Built:** `emit_dit_ladder` in `scripts/gen-minimax-h3-goldens.py` (7 rungs: 2×3, 4×4,
  6×6, **8×8**, a 4×8 rectangle, an 8×8×3-frame temporal 3D grid, and a 6×10×5-frame
  video+audio packed mix), and the permanent gate case `test_minimax_h3.cpp :: "DiT-forward
  geometry ladder matches upstream (host+device, mixing)"`. Each rung gates the upstream
  packed-sequence layout (cu_seqlens / fp64 position grid / masks), the HOST forward, the
  DEVICE-resident forward (the pipeline's own path), and a spatial-MIXING probe.
- **Result: ours == the RefDiT oracle at EVERY rung**, host AND device, max|diff| ≤ 3e-7
  vs the 2e-5 gate. The mixing probe: perturbing one video-target token changes EVERY
  other target token (fraction 1.0 at all rungs) — the packed bidirectional attention
  (`cu_seqlens=[0,used,seq_len]`, one document) couples all video tokens at real geometry.
- **Hidden-dim-scale leg:** a second case reruns the geometries at the REAL head_dim=128 /
  rope_inv_freq_len=16 (rot_dim=96) ratio and requires the DEVICE forward to track the
  trusted HOST loops — device-vs-host ≤ 1.2e-6 across all rungs (no head_dim/rope-scale
  device-op assumption).
- **Why the ladder cannot SHOW #70's symptom:** measured the #70 adjacent-cell COSINE on
  the CORRECT oracle at reduced dims — adj_cos ≈ random-pair ≈ 0 at every geometry. With
  RANDOM weights the correct reference is ALREADY white by the cosine metric; spatial
  coherence is a TRAINED-WEIGHTS property. The harness's valid discriminators are oracle-
  logit equality and information flow (both green), not the cosine.
- **VERDICT:** the "spatial-mixing bug in the DiT-forward MATH" hypothesis is **REFUTED**.
  The #70 white latent is NOT a reduced-dim-reproducible DiT-forward bug — it is a
  trained-weights / real-scale phenomenon. A GPU re-render is NOT expected to be coherent
  from this work; nothing in the render path was changed.
- **Residuals (both beyond the CPU box):** (1) a bug shared identically by our port AND the
  RefDiT restatement vs TRUE upstream `minimax_h3_transformer.py` (not importable here —
  no `vllm`/`cache_dit`/`aenum`) is invisible to this ladder; close it on the dgx oracle
  venv where vllm is installed. (2) the real-scale DiT INPUT wiring (Qwen3-VL encoder
  embeddings, real fp64 position grid at full canvas, real per-token timesteps) is fed with
  RANDOM data here; a real-weights activation diff of the DiT inputs is the untested surface.
  Full tables: benchmark record (`row/H3-DIT-SCALE-GATE`).

## 8.6 RENDER BUG CLOSED — wrong checkpoint PARTITION, not a code bug (2026-08-06, `row/H3-RENDER-CLOSE` PR #77)

The #70/#74 white render was **using the wrong checkpoint partition for the task.**
MiniMax-H3 ships two independently-served DiT partitions and the task MUST match
(`recipes/MiniMaxAI/MiniMax-H3.md:50,289`; `pipeline._resolve_task` raises otherwise):

| Partition | Serves | Available quantized DiT |
|---|---|---|
| **FL2VA** | **t2va + fl2va** | `MiniMax-H3-FL2VA-Q3_K_M.gguf` (GGUF), FL2VA NVFP4 (not downloaded) |
| **Ref2VA** | ref2va (image/video + audio references) | `minimax_h3_ref2va_nvfp4_full` (the NVFP4 we had), REF2VA GGUF |

Every render up to #74 ran **t2va on `minimax_h3_ref2va_nvfp4_full` (the Ref2VA
partition)** — an out-of-distribution task/partition combination upstream rejects. That
is the white latent, invariant to prompt/steps.

**Verified before switching partitions (all NEW, real 512x512/22f scale, dgx):** the
t2va DiT INPUTS diff EXACTLY vs upstream `pipeline_minimax_h3.py` (`VT_H3_DUMP_INPUTS`:
packed layout / fp64 grid / token_tags / inverse+combined AdaLN indices / sigmas all
byte-equal; tokenization byte-equal); the encoder conditioning is correctly shaped and
carries the expected Qwen massive-activation; `DequantNvfp4ToBf16` is byte-exact
(Laguna/Qwen3 + independent torch dequant); and the CUDA device forward == the CPU host
forward at the REAL render seq (1920) at head_dim=128 (new permanent gate
`test_minimax_h3 :: "CUDA device forward tracks the host at the REAL render seq (1920)"`,
28/28) — closing the "CUDA kernel at scale" hole #74's CPU-backend device-vs-host left open.

**Proof:** t2va on `MiniMax-H3-FL2VA-Q3_K_M.gguf` (`--dequant-bf16`, 512x512/22f, prompt
"an orange cat sitting on a wooden table") renders a **COHERENT photorealistic orange cat
on a wooden table** — VAE-input latent adj-cell cosine **0.9467** (white was 0.06), frame
seam16/interior **1.00** (no patch grid), velocity stable ~1.37, final latent rms **1.00**.
Valid h264 512x512 + AAC 32kHz mp4.

**Fixed in this row:** `MiniMaxH3GenerateT2va` now strips the PREPENDED pinned reference
rows (ref2va) before unpatchify/unpack — they are zeroed in the DiT output and only the
trailing target rows are the clip; the old code fed unpatchify the full buffer and hit
"rows not divisible by t*h*w" (no-op for t2va/fl2va). **Open:** a partition/supported_tasks
guard mirroring upstream (community files strip the release config); the encoder vision
tower (W3) is still unported, so image/video-conditioned ref2va/fl2va renders are not yet
clean (ref2va with a synthetic reference + text-only encoder still grids).

## 8.7 TASK/PARTITION GUARD — mirror `_resolve_task`'s raise (2026-08-06, `row/H3-TASK-PARTITION-GUARD` PR #84)

The #70/#74 white grid cost three campaigns because our driver silently accepted
`task=t2va` on the Ref2VA-partition checkpoint. Upstream `pipeline._resolve_task`
RAISES on the mismatch (`pipeline_minimax_h3.py:374-391`, esp. 387-390); the recipe
documents the split (`recipes/MiniMaxAI/MiniMax-H3.md:50-51,289`: "One server loads one
checkpoint partition … must match the served partition"). This row mirrors the raise 1:1.

**Partition detection — two paths, and the definitive no-discriminator finding.**
Upstream reads the served-task set from the release config
(`pipeline_minimax_h3.py:279-282`):

```
release = model_index.get("_minimax_h3") or {}
self.partition       = str(release.get("partition", ""))       # "fl2va" | "ref2va"
self.supported_tasks = frozenset(release.get("tasks") or ())
```

`MiniMaxH3PartitionFromModelIndex(model_index)` mirrors those exact keys. But community
GGUF/NVFP4 redistributions STRIP that block, and — measured on the two real manifests
this spec already captured — there is **NO structural fallback**: the Ref2VA NVFP4
(1051 tensors) and FL2VA GGUF (535 tensors) carry the **IDENTICAL DiT**. Normalizing the
NVFP4 `{weight, weight_scale, weight_scale_2}` split, both files reduce to the **SAME 535
base tensor names AND the SAME shapes** (video_patch_proj `[5376,96]`, audio_patch_proj
`[5376,32]`, condition_proj `[5376,5120]`, time_embedder.proj_in `[5376,256]` on both;
`comm -23`/`-13` of the normalized name sets is empty both ways). Ref2VA conditioning is
achieved by PREPENDING reference rows through the SAME `video/audio_patch_proj` weights,
so it introduces no reference-specific tensor to key on. A name/shape auto-detector is
therefore impossible in principle. When the config is stripped the partition must be
**DECLARED** (`--partition fl2va|ref2va`), never guessed; `MiniMaxH3PartitionFromFlag`
maps it to the recipe's served-task set (fl2va→{t2va,fl2va}, ref2va→{ref2va}).

**The refuse.** `MiniMaxH3CheckTaskPartition(task, info)` is the raise half of
`_resolve_task`. The task is what the request ENCODES (`MiniMaxH3TaskOfRequest`:
`ref_blocks`→ref2va, `keyframe_frame_indices`→fl2va, else t2va), and
`MiniMaxH3GenerateT2va` calls the pair before denoising. A declared partition refuses a
task it does not serve; an UNKNOWN partition (stripped file, no `--partition`) refuses
EVERY task as ambiguous and names the recipe lines. A default-constructed
`MiniMaxH3PartitionInfo` (`declared=false`) leaves the guard inactive, so the pure
pipeline-math unit tests are unaffected. Wired at both checkpoint-loading entry points:
the driver (`--partition`) and the server (`--video-partition`).

**Guard behavior table (task × partition → pass/refuse):**

| task \ partition | FL2VA {t2va,fl2va} | Ref2VA {ref2va} | unknown/stripped |
|---|---|---|---|
| **t2va**  | pass | **REFUSE (the #77 mismatch)** | REFUSE (declare `--partition`) |
| **fl2va** | pass | REFUSE | REFUSE |
| **ref2va**| REFUSE | pass | REFUSE |

**RED-first proof.** New case `test_minimax_h3 :: "the task/partition guard refuses the
#77 mismatch"` (38 assertions): the #77 combo `MiniMaxH3CheckTaskPartition("t2va",
ref2va)` throws; the correct pairings pass; the stripped case refuses every task and
`--partition` recovers it; `MiniMaxH3TaskOfRequest` maps the three request shapes; and it
asserts the two real manifests reduce to the identical 535-name set (proving the
no-discriminator premise in the harness). Neutralizing the guard body (reviewer mutation)
turned the case RED at 10 assertions, restoring it turned it GREEN — the test has teeth.
Suite: 67/67 (66 prior + this), 46549 assertions. `test_video_api` 4/4 (server wiring).

## 8.8 ENCODER VISION TOWER — record reconciliation + real-weights wiring (2026-08-06, `row/H3-CONDITIONED-E2E`)

**The contradictory record, reconciled (file:line).** Two prior lanes disagreed. The
#26/W3 lane recorded the vision tower as **"W3 COMPLETE … the FULL vision tower at
6.0e-8 … only the MM processor remains"** (this spec lines 101, 162-163); the #77
residual recorded **"the encoder vision tower (W3) is still unported"** (lines 565-566).
Reading the actual code resolves it — **both describe different halves and both are
literally true of what they describe**:

- The vision-tower **MATH exists** as a CPU scalar f32 reference in
  `minimax_h3_encoder.cpp`: `MiniMaxH3VisionBlockForward` (:311), the surround
  `MiniMaxH3VisionPosEmbedInterpolate` (:430) / `MiniMaxH3VisionRotary` (:500) /
  `PatchMerger` (:545) / `MiniMaxH3VisionTowerForward` (:572). It is gated ONLY in
  `tests/vllm/models/test_minimax_h3.cpp` (:3942, :4041) at **reduced dims with SYNTHETIC
  weights** (`MakeParam`), block 6.0e-8 / tower ≤1e-4 vs a self-restated oracle.
- It is **NEVER wired to real weights.** `LoadMiniMaxH3EncoderFromGguf`
  (`minimax_h3_encoder_gguf.cpp:47`) loads the **TEXT tower only** — it iterates
  `model.layers.N.*` + `model.embed_tokens.weight` and **skips every `visual.*` tensor**
  (the comment at :51-52 even names `visual.*` as present-but-unloaded). The device
  encoder `MiniMaxH3EncoderTextForwardDevice` (`minimax_h3_encoder_device.cpp:103`) runs
  text only and takes **no deepstack / no visual-mask** argument (the HOST reference
  `MiniMaxH3EncoderTextForward` does, :113-118). The driver
  (`examples/minimax_h3_gen/main.cpp:476-547`) and server
  (`examples/server/main.cpp:659-716`) call only the text path.
- **So the reconciled truth:** the tower math is CPU-gated at reduced dims with synthetic
  weights; there is **zero real-weights wiring** — no GGUF `visual.*` loader, no image→patch
  MM processor on the H3 path, no device vision forward, no merge/DeepStack injection into
  the encoded prompt. The #26 "only the MM processor remains" understated the gap (loader,
  real-weights forward, and the merge/inject were ALSO absent); the #77 "still unported" was
  right in the sense that matters (nothing real ran through it).

**The encoder ARM already carries the vision weights (no download).** The on-box encoder
`~/h3fp4/ckpt/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` (14 GiB, the ComfyUI-format text-tower
GGUF that already serves text conditioning) **DOES carry the full vision tower**: measured
`visual.blocks.{0..26}` (27, Q4_K/Q5_K), `visual.patch_embed.proj` (F16 `[16,16,6,1152]` =
Conv3d as a linear over `patch_elems`=1536), `visual.pos_embed.weight` (F16 `[2304,1152]` =
48² grid), `visual.merger.*`, and **`visual.deepstack_merger_list.{0,1,2}`** (3 DeepStack
mergers). Names map 1:1 to `MiniMaxH3VisionTowerForward` / the reuse target
`multimodal::Qwen3VLVisionWeights`. So the encoder-arm decision is settled: **reuse the
in-place encoder GGUF; no new download** (disk floor 15 GiB / ~23 GiB free honoured).

**Vision geometry (from the checkpoint + state.md :23310-23318, the Qwen3.6-27B vision
config which shares this tower):** hidden **1152**, **16 heads** (head_dim 72), depth **27**,
intermediate **4304**, out_hidden **5120** (== encoder text dim), patch **16**, temporal **2**,
merge **2**, num_position_embeddings **2304**, gelu-tanh blocks / exact-erf merger. H3 differs
from the 27B only by having **3 real DeepStack mergers** (the 27B's are empty). The one
config value NOT recoverable from the ComfyUI GGUF (weights-only, no arch metadata) is
`deepstack_visual_indexes` — the WHICH-layers taps — needed for a bit-correct DeepStack
inject; it is inferred + flagged as the residual for a fully-correct conditioned render.

**The reuse path (mission: "stock Qwen3VLProcessor + our existing front end").** The image
MM processor already exists and is gated: `multimodal::Qwen3VLImageProcessor::ProcessImage`
(`qwen3vl_processor.h`, patch 16 / temporal 2 / merge 2 / 0.5 normalize → pixel_values +
grid_thw), `ExpandImagePlaceholders`, and the device tower
`multimodal::Qwen3VLVisionForward` (`qwen3_vl_vision.cpp`) with `PrepareVisionDeviceWeights`.
The only genuinely-new code is the **GGUF `visual.*` → `Qwen3VLVisionWeights` loader**
(`LoadQwen3VLVisionFromGguf`), mirroring the safetensors `LoadQwen3VLVisionWeights`
(`qwen3_vl.cpp:417`) but dequantizing the Q4_K/Q5_K blocks (ComfyUI reshapes non-256-aligned
rows to ne0=256; dequant preserves the flat row-major order the tower reads as `[out,in]`)
and converting the F16 patch/pos tensors.

**This row's status (honest):** loader + real-image processor reuse + the real-weights
vision-tower forward gate LAND here (see §8.4-style status in STATUS/BENCHMARKS). The full
vision-ENRICHED DiT render (DeepStack scatter into the DEVICE text tower changing the frames)
depends additionally on the exact `deepstack_visual_indexes` and a device-text DeepStack/merge
extension; its e2e render verdict is recorded honestly in the benchmark record.

**GB10 VERIFIED (2026-08-07, dgx sm_121a).**
- **Vision-tower probe RAN on real weights:** `--prompt-image` loaded the real `visual.*`
  tower (27 blocks / 3 DeepStack mergers), processed a 512×512 image → grid [1,32,32], and
  `Qwen3VLVisionForward` returned [256, 20480] all FINITE + non-degenerate (merged rms 1.45 /
  maxabs 29.1 — the expected Qwen massive-activation). Deliverable-1 core DONE.
- **fl2va e2e COHERENT:** FL2VA GGUF (`--dequant-bf16`) + a real first-frame (VAE-keyframe) +
  `--partition fl2va`, 512×512/22f/12steps → all 22 frames a coherent photorealistic orange
  cat on a wooden table matching the conditioning frame (no grid). Frame-sanity PASS.
- **ref2va STILL GRIDS (honest):** Ref2VA NVFP4 (`--fp4-resident`) + a real `--ref-image` +
  `--partition ref2va` → every frame a multicolour patch grid. Landing the tower LOADER does
  NOT fix it: the tower is a loader+probe, NOT yet scattered into the DiT render-conditioning,
  so this render never used it. fl2va (same session, VAE-keyframe) is coherent ⇒ DiT/VAE/
  partition are sound; the ref2va grid is specific to the ref2va conditioning assembly. The
  render-conditioning scatter (merge features into prompt_embeds + DeepStack inject into the
  DEVICE text tower) is the tracked residual that would let the vision-enriched-prompt
  hypothesis be tested. The `--ref-video` VAE encode is a slow single-thread CPU 3D-CNN path
  (separate perf limit).

## 8.9 ENCODER VISION SCATTER — merged→prompt_embeds + DeepStack→device text tower (2026-08-07, `row/H3-VISION-SCATTER` PR #90)

Closes the §8.8 residual at the FRAMEWORK level and RE-ATTRIBUTES the ref2va grid with a
render A/B. Three deliverables.

**`deepstack_visual_indexes` CONFIRMED (was #86-inferred).** The value is `[8, 16, 24]`,
grounded in the release config: MiniMax-H3's `text_encoder/` IS **Qwen3-VL-32B-Instruct**
(HF `.../MiniMax-H3/.../Qwen3-VL-32B-Instruct/config.json`), whose
`vision_config.deepstack_visual_indexes = [8, 16, 24]`, depth 27, text `num_hidden_layers = 64`
(truncated to 50) — identical to vllm-omni's `Qwen3VLMoeVisionConfig` default and the public
`Qwen/Qwen3-VL-30B-A3B-Instruct` config. The #86 inference was correct; comment updated in
`minimax_h3_vision_gguf.cpp:46-52`.

**Deliverable 1 — the DEVICE scatter+inject is WIRED 1:1 + GATED.** `MiniMaxH3EncoderTextForwardDevice`
(`minimax_h3_encoder_device.cpp:103,216-243`) now takes the optional `visual_pos_mask` + per-tap
`deepstack` blocks and, after each of the first `len(deepstack)` decoder layers, ADDS each block
into the masked visual-token rows — the device mirror of the gated host reference and of upstream
`MiniMaxH3Qwen3VLTextModel._deepstack_process` (`encoder.py:770-800`,
`hidden_states[visual_pos_masks] += visual_embeds`). The MERGED-feature masked_scatter into
`inputs_embeds` stays the caller's job (upstream `_encode` scatters it BEFORE the tower runs;
`encoder.py:1071`), exactly like the host reference. Text-only prompts pass the defaults and are
byte-identical. **Gate** (`test_minimax_h3.cpp :: "the DEVICE keep-quant encoder matches the host
f32 reference"`): the device forward now also runs WITH a visual mask + two DeepStack blocks and
checks device==host-reference (max|diff| **3.8e-4** ≤ 2e-3) AND that DeepStack MOVES the
conditioning (scale 1.006→1.062) — the surface #86 could not cover. All encoder/vision gates green
(host text tower + full vision tower + GGUF `visual.*` loader + MM processor).

**Driver wiring — `--cond-image` routes a reference image through the ENCODER vision path**
(`examples/minimax_h3_gen/main.cpp`, mirroring `_encode`). Reuse-only: `Qwen3VLImageProcessor` →
`Qwen3VLVisionForward` (real `visual.*` tower) → merged `[nm,5120]` + 3 DeepStack blocks;
`ExpandImagePlaceholders` inserts `nm` image-pad tokens; merged masked_scatter into the embeds at
those rows; M-RoPE positions from `Qwen3VLGetRopeIndex` (byte-equivalent to H3's own
`_get_rope_index` for a single-frame image, t==1 — position math verified: text sequential, image
block the 3D grid, next-text advances by `max(llm_h,llm_w)`). Additive: without `--cond-image` the
text-only path is byte-identical.

**GB10 RENDER A/B (2026-08-07, dgx sm_121a, 256×256/22f/12steps).**
- **Deliverable 3 — fl2va WITH the encoder vision path = COHERENT + matching (PASS).** FL2VA GGUF
  (`--dequant-bf16`) + `--first-frame` (VAE-keyframe) + **`--cond-image`** (encoder vision) +
  `--partition fl2va`, prompt "a fluffy orange cat sitting on a windowsill in warm sunlight".
  Conditioning `[82,5120]` = 16 prompt + a 66-token vision block (64 merged image-pad rows + 2
  markers). Frame 0 = a coherent photorealistic ORANGE CAT matching the keyframe; frame 21 = the
  same cat on a **WINDOWSILL in warm sunlight** — the clip EVOLVED toward the text prompt. No grid.
  The vision-enriched conditioning is SOUND and load-bearing. Artifact `~/h3fp4/out_vs_fl2va.mp4`.
- **Deliverable 2 — ref2va WITH the vision-enriched prompt STILL GRIDS (honest FAIL).** Ref2VA NVFP4
  (`--fp4-resident`) + `--ref-image` (VAE reference rows) + **`--cond-image`** (encoder vision) +
  `--partition ref2va`, same prompt. Conditioning `[82,5120]` (64 merged + 3 DeepStack), 1 reference
  image, latent 7×16×16. Every frame (0/10/21) is the same multicolour PATCH GRID as #86's
  text-only ref2va. Artifact `~/h3fp4/out_vs_ref2va.mp4`.

**RE-ATTRIBUTION (with evidence).** The mission's "vision-enriched conditioning fixes the grid"
hypothesis is **REFUTED**. The ref2va grid is NOT the encoder conditioning: (a) the DiT forward MATH
is byte-exact vs upstream (§8.5 geometry ladder green every rung; §8.6 device==host at real seq
1920); (b) the vision scatter+DeepStack is proven sound by the COHERENT fl2va-with-`--cond-image`
render — the SAME conditioning path; (c) the ref2va grid is INVARIANT to text-only (#86) vs
vision-enriched (this row) prompts. The ONLY thing that differs between the coherent fl2va and the
gridding ref2va is that **fl2va PINS output rows (keyframe cond rows) each denoise step** while
**ref2va PREPENDS free-running reference rows** — so the residual is the **ref2va-specific
reference-row conditioning ASSEMBLY** (`MiniMaxH3EncodeReferenceImages` VAE-reference rows +
`minimax_h3_packed_sequence_ref2va_blocks` noised-anchor layout + how the denoise loop conditions
the un-pinned target rows on them), NOT the prompt_embeds and NOT the DiT forward. Next diagnostic:
dump the ref2va target-row VAE-input latent adjacency-cosine (like #77 did for the coherent fl2va,
0.95) to confirm the target rows are white, and A/B the reference-row condition-noise vs a clean
anchor.

## 9. W-OAI — the `/v1/videos` OpenAI (Sora) WIRE SHAPE, 2026-08-06

Row `SERVE-VIDEOS-OAI` (engine matrix, Serving surface), claim
`CLAIM-SERVE-VIDEOS-OAI`, branch `row/SERVE-VIDEOS-OAI`.

Developer-directed: an unmodified OpenAI client must work against `/v1/videos`.
ADDITIVE — the vLLM-Omni-derived fields keep working, and every body that parsed
before means exactly what it meant before.

SPLIT, deliberately: this row is the REQUEST/RESPONSE SHAPE only (`model`,
`size`, `seconds`, and the MP4 download route). It touches no generation code and
loads no VAE. The REFERENCE CONDITIONING half (`input_reference` -> fl2va, plus
the two `metadata` reference modalities -> ref2va) is §10, row
`SERVE-VIDEOS-REFS`, because it is a separate capability that pulls in the VAE
encoder halves and the runner. Each half is independently reviewable and gated.

### 9.0 Spike contract (`SERVE-VIDEOS-OAI`)

| Section | Content |
|---|---|
| Scope | IN: the OpenAI (Sora) REQUEST SPELLINGS `model`, `size`, `seconds` on `/v1/videos`, their precedence against the native fields, the `model`-mismatch warning on the job, and `GET /v1/videos/{id}/content`. OUT: reference conditioning of any modality (§10); OpenAI's status vocabulary / id shape / `progress` / multipart upload; any change to generation, the DiT, the VAEs or the muxer. |
| Upstream chain | OpenAI's published video API (`POST /v1/videos`, `GET /v1/videos/{video_id}/content`, `size` "WxH", `seconds` string enum) is the request CONTRACT; vLLM-Omni's `/v1/videos` async+sync job pair is the endpoint shape we already mirror. |
| Our baseline | `ParseVideoRequest` took only the native spellings (`duration`, `height`/`width`, `num_frames`, `num_inference_steps`, `flow_shift`, `audio_flow_shift`, `seed`, plus `extra_params`); an OpenAI client's body parsed to DEFAULT geometry and duration. `VideoJobStore` had no `model`/`warning`. The routes stopped at status: the produced .mp4 was reachable only through the filesystem. |
| Port map | Request contract -> `include/vllm/entrypoints/openai/video_api.h` (`VideoRequest::model` + `ParseVideoSize`) and `src/vllm/entrypoints/openai/video_api.cpp` (`ParseVideoRequest`, `ReadDuration`, `ParseWholeNumber`). Job record -> `VideoJobStore::Create(model, warning)` + `VideoJobStatusJson`. Download route -> `ApiServer::handle_video_content` + `video_model_warning` + their registration in `src/vllm/entrypoints/openai/api_server.cpp`. |
| Tests to port | No upstream test module exists for this surface (OpenAI publishes an API, not tests; vLLM-Omni's video endpoint has no ported test). The contract is gated in-tree instead, extending the existing files: `tests/vllm/entrypoints/openai/test_video_api.cpp` (parsing, precedence, the job record) and `tests/vllm/entrypoints/openai/test_api_server.cpp` (routes, content behaviour, additivity over a real socket). Every assertion uses values that DIFFER from the field default. |
| Gates | CPU, foreground: `test_video_api` 11/11 (125 assertions), `test_openai_api_server` 40/40 (509), `server` builds clean. Content route: 404 unknown / 409 unfinished (no bytes leaked) / 500 failed / 500 vanished / 200 byte-exact `video/mp4`. Additivity: with no `VideoRunner`, `POST /v1/videos` is 404 over a real socket with no `ErrorResponse` envelope; with one, it is 200 and the unknown-id 404 IS ours. Commands: `cmake --build build --target test_video_api test_openai_api_server server -j12`. Real-weights e2e rides §8's GB10/disk window. |
| Dependencies | Row IDs: the MiniMax-H3 model rows and `row/H3-FP4-SPEED` (UNTOUCHED - no generation code changed); `SERVE-VIDEOS-REFS` (§10) stacks on this row. No new download, no GPU, no toolchain change for the CPU gate. |
| Work breakdown | (1) alias parsing + precedence + `ParseVideoSize`; (2) `model` recording + the job `warning`; (3) `handle_video_content` + its route; (4) both test files; (5) docs + record. |
| Risks/decisions | NATIVE-wins precedence: the only direction that leaves every previously-parsing body meaning what it meant. `model` mismatch WARNS rather than 404s: a Sora client cannot know the local model name, so a rejection would defeat the compatibility; silence would hide it. A 409 (never bytes) on an unfinished job: a partially muxed file would reach the client as a valid-looking, truncated MP4. No vLLM-defined behaviour is reopened. |

### 9.1 The aliases and their precedence

| OpenAI | Lands on | Notes |
|---|---|---|
| `model` | `VideoRequest::model` | Recorded + echoed; an unserved name is a job `warning`, never a rejection (a Sora client cannot know the local model's name) |
| `size` | `width`, `height` | `"<w>x<h>"`, whole positive pixels, one `x`/`X` |
| `seconds` | `duration_seconds` | Number OR numeric string — OpenAI types it as a string enum ("4"/"8"/"12") |

PRECEDENCE: the NATIVE field WINS (`width`/`height` over `size`, `duration` over
`seconds`). Both spellings are VALIDATED whichever wins, so a malformed `size` is
a 400 even when explicit `width`/`height` override it. Precedence is PER-AXIS: an
explicit `width` alone still lets `size` supply the height it did not specify.

### 9.2 `GET /v1/videos/{id}/content`

Returns the finished MP4 as `video/mp4`. Without it a caller could start and poll
a job but never fetch the result over HTTP. Unknown id -> 404; queued/running ->
409 naming the status (never a truncated file); failed -> 500 carrying the
failure; a vanished output -> 500, not a 200 with zero bytes.

### 9.3 Status

- **CPU-LANDED + gated.** `test_video_api` 11/11 (125 assertions),
  `test_openai_api_server` 40/40 (509), `server` builds clean. Additivity is
  gated over a REAL socket: without a `VideoRunner` all four routes are absent
  (a 404 with no `ErrorResponse` envelope), with one they serve.
- **Residuals, named.** OpenAI's status vocabulary is not mirrored (ours stays
  queued/running/succeeded/failed, ids `vid_N`, no `object`/`progress`/
  `created_at`); reference conditioning is §10 (`SERVE-VIDEOS-REFS`), not this row.
- **Real-weights leg** rides the same GB10/disk window as §8.

## 10. W-REFS — reference conditioning over `/v1/videos`, 2026-08-06

Row `SERVE-VIDEOS-REFS` (engine matrix, Serving surface), claim
`CLAIM-SERVE-VIDEOS-REFS`, branch `row/SERVE-VIDEOS-REFS`, stacked on §9.

§9 made an OpenAI client's request PARSE. This row makes its REFERENCES do
something: an image the video starts from, a clip it continues, a voice it
carries. Before it, no reference modality was reachable over HTTP at all.

### 10.0 Spike contract (`SERVE-VIDEOS-REFS`)

| Section | Content |
|---|---|
| Scope | IN: OpenAI's `input_reference` mapped to fl2va first-frame conditioning; the two reference modalities OpenAI has no slot for carried in `metadata` (`input_reference_video`, `input_reference_audio`) mapped to ref2va blocks; the fl2va/ref2va combination rule enforced at the request boundary; the `examples/server` runner wiring (PPM decode, frame-directory clip, WAV, lazily-loaded VAE encoder halves). OUT: the ref2va IMAGE modality (reachable via the native `task` + the CLI, deliberately not bound to `input_reference`); any change to generation, the DiT, the VAEs or the muxer; OpenAI's multipart upload. |
| Upstream chain | OpenAI documents `input_reference` as the image the generated video STARTS FROM. The conditioning entry points are ours and already gated: `MiniMaxH3EncodeKeyframeCondRows` (fl2va), `MiniMaxH3EncodeReferenceVideo` / `MiniMaxH3EncodeReferenceAudio` (ref2va), `MiniMaxH3ReadWav`. The exclusivity rule is `src/vllm/model_executor/models/minimax_h3_pipeline.cpp:251`. |
| Our baseline | After §9 the OpenAI wire shape parses, but every reference field is absent: an image-to-video request silently generated from the prompt alone. |
| Port map | Request contract -> `include/vllm/entrypoints/openai/video_api.h` (`input_reference*`, `metadata`, the `has_*` predicates) and `src/vllm/entrypoints/openai/video_api.cpp` (`ReadReferenceSource`, `ReadMetadata`, the combination `VT_CHECK`); the `data:` decode REUSES `entrypoints::openai::DecodeDataUri` (chat_mm) rather than a second decoder. Reference wiring (the process boundary keeps it out of the library) -> `examples/server/main.cpp`: `DecodePpmChw`, `ReadReferenceClipChw` (the CLI's `DIR/frame_%06d.ppm` convention), `ReadReferenceBytes`, and the fl2va / ref2va branches with lazily-loaded VAE encoder halves. |
| Tests to port | No upstream test module exists for this surface. The contract is gated in-tree, extending the same two files: `test_video_api.cpp` (reference parsing, `metadata` passthrough, combination legality) and `test_api_server.cpp` (each modality ARRIVES at the runner, and an illegal pair is a 400 that generates nothing). |
| Gates | CPU, foreground: `test_video_api` 14/14 (167 assertions), `test_openai_api_server` 41/41 (525), `server` builds clean. Commands: `cmake --build build --target test_video_api test_openai_api_server server -j12`. Real-weights e2e rides §8's GB10/disk window. |
| Dependencies | Row `SERVE-VIDEOS-OAI` (§9), stacked. Code: `MiniMaxH3Encode{KeyframeCondRows,ReferenceVideo,ReferenceAudio}`, `MiniMaxH3ReadWav`, `DecodeDataUri`. Runtime: `--video-vae` for an image or video reference, `--audio-vae` for an audio reference (both encoder halves, loaded lazily and once). No new download, no GPU. |
| Work breakdown | (1) `input_reference` parsing (path or `data:` URL) -> fl2va, with the geometry refusal; (2) the `metadata` map + the video/audio reference keys; (3) the combination rule in the parser; (4) the `examples/server` runner branches; (5) both test files; (6) docs + record. |
| Risks/decisions | `input_reference` -> fl2va, NOT ref2va: OpenAI documents it as the frame the video starts from; ref2va would silently change what the API promises. The two extra modalities go in `metadata` rather than new top-level fields, so a strict client's schema validation still passes. Combination legality is enforced in the PARSER, not left to the pipeline, so a supplied reference is never silently dropped. |

### 10.1 `input_reference` is fl2va, not ref2va

OpenAI documents it as the image the video STARTS FROM, which is what
`MiniMaxH3EncodeKeyframeCondRows` expresses (frame 0 of the output pinned to the
image, `imgvid_noise_aug = 1.0`). `MiniMaxH3EncodeReferenceImages` prepends whole
reference images as their own blocks — guidance that never becomes a frame — so
mapping it there would have changed what the API promises. With a reference image
and no explicit `task`, the task IS `fl2va`, and the image's aspect drives the
default resolution through `MiniMaxH3ResolveShape`.

Two limits, both refused up front rather than deep in the denoise: the image must
be a binary PPM (P6), because no PNG/JPEG codec is vendored (the same NAMED
residual the chat multimodal path carries), and it must already be at the
resolved output geometry, because no image resampler is vendored. The refusal
names both geometries.

### 10.2 The two reference modalities OpenAI has no slot for

H3 has three (image, silent video, audio); the Sora schema carries one. The other
two enter through `metadata`, the standard OpenAI free-form string map that
strict clients tolerate, rather than invented top-level fields that would fail a
client's schema validation. The whole map is kept verbatim.

- `metadata.input_reference_video` — a DIRECTORY of `frame_%06d.ppm`, the exact
  layout `minimax-h3-gen` and the server WRITE, so clips chain. No demuxer is
  vendored, hence a frame directory rather than a container; a `data:` URL cannot
  name a directory and is refused by name.
  `MiniMaxH3EncodeReferenceVideo` emits `ref_audio_t == 0`: the clip is SILENT.
- `metadata.input_reference_audio` — a 16-bit PCM WAV path or `data:` URL, read
  by the existing `MiniMaxH3ReadWav` and encoded by
  `MiniMaxH3EncodeReferenceAudio`. Supplied with a video reference it ATTACHES to
  that block (one `kVideoAudio` block carrying both, the layout
  `packed_sequence.py` builds); alone it is its own block.

LEGALITY is the pipeline's own rule (`minimax_h3_pipeline.cpp:251`: fl2va
keyframes and ref2va blocks are exclusive), enforced in the PARSER so it is a 400
naming the pair rather than a failed job — and never a silently dropped
reference, which is the failure that looks like it worked. Legal: none / image /
video / audio / video+audio. Illegal: `input_reference` with either metadata
reference.

### 10.3 Status

- **CPU-LANDED + gated.** `test_video_api` 14/14 (167 assertions),
  `test_openai_api_server` 41/41 (525), `server` builds clean. Each modality is
  gated as ARRIVING at the runner, and an illegal pair is a 400 that generates
  nothing (`calls == 0`).
- **Residuals, named.** Reference images are binary PPM at the output resolution;
  a video reference is a frame DIRECTORY; OpenAI's real `input_reference` upload
  is multipart, ours is the JSON spelling.
- **Real-weights leg** rides the same GB10/disk window as §8.
