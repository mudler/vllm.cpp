# MiniMax-Music3 — text-to-music, and our first music-generating model

**Rows:** `MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation`
(model-matrix).
**Issue:** [#672](https://github.com/mudler/vllm.cpp/issues/672).
**Claim:** `CLAIM-MODEL-MUSIC3-W0`.
**Checkpoint:** `MiniMaxAI/MiniMax-Music3`, 57.4 GB total — but the arm we port is
**~28.5 GB** (see §2).
**Upstream:** `diffusers` PR
[#14456](https://github.com/huggingface/diffusers/pull/14456), head
`c6da9936e4bda83107943a16eb8682e9a37d8527` — **OPEN, not merged**.
**Cross-check:** SGLang-Omni `748a0b437e4a8faad44d7bbfd5a0ae55d1fef830`.
**Status:** **W0-W7 DONE, W2 INCLUDED; a PARTIAL DEVICE ARM landed 2026-08-16 (§11).** Every stage is implemented and gated; a COMPOSED request is not yet observed to completion on CPU (see `## Now`). Spec committed, both oracles pinned, §1.1 resolved and confirmed at runtime, the diffusers oracle gateable against committed goldens, the modular loader in the tree, the autoregressive half's compute gated at reduced dimensions and against the real bf16 checkpoint, and the ACOUSTIC half — flow-matching DiT, scheduler, CFG, window bookkeeping, DAC Flow-VAE vocoder — gated at both scales against the committed capture. §5's token-exact gate is WITHDRAWN: upstream's AR stage has no greedy path, and the acoustic half never had one to withdraw. W6 has landed: the model is a registered `SpeechRegistry` family reachable through the new `vllm_speech_*` C ABI (v20) and `POST /v1/audio/speech`, and the denoise+decode composition reproduces the capture's waveform. W7 has landed (§9): quantized checkpoints for this model exist in five formats; the RVQ depth decoder's GGUF Q4_K arm IS implemented and value-gated against a pinned artifact, and every other format and lineage is refused by name with the missing piece rather than surfacing as a confusing shape error. The 8.6B language-model forward and the other four components' GGUF arms are owed.
**Developer directive (2026-08-13):** "land minimax music 3 support complete, to
vllm.cpp, wired to the ABI and to the example http server, merge to main, tested
e2e." That fixes W6's shape (the ABI surface and the example server are in scope,
not optional follow-ups) and records merge authority for this campaign. Merge is
still gated on PROVED: fresh review PASS, the operator's own gate rerun, and no
red bought by weakening a detector.

---

## 0. Honesty statement — what is and is not claimed

Nothing has been ported. Nothing has been measured on hardware. This spec records
what was **read from the checkpoint and from upstream source**, and separates that
from what is still assumed.

**Measured** (safetensors headers by HTTP range request, and each component's
`config.json`): every geometry and dtype in §1. **Read** (upstream source at the
pinned SHAs): the component decomposition, the native↔diffusers relationship, and
the dtype policy. **Established 2026-08-14, after this section was
written:** the oracle runs here — `tools/oracle/music3_oracle.py` loaded all seven
components and generated audio, so `.agents/oracles/diffusers.md` records
`gateable = yes` against a golden path. That measurement was taken on CPU;
nothing about speed is established.

**This model has no token-exact gate on its generative half.** Like MiniMax-H3, the
acoustic path is a flow-matching denoise loop with no logits and no sampler, so the
SACRED near-tie methodology does not apply to it. What *is* token-exact is the
global LLM half, which emits discrete RVQ codes. §5 states which gate binds where;
conflating the two is the failure mode this section exists to prevent.

---

## 1. What the model is, measured

Lyrics (with `[Verse]` / `[Chorus]` section tags) plus a structured music
description in; a multi-minute stereo song out. Hierarchically: a global LLM
predicts a semantic frame sequence, a small depth decoder expands each frame into
eight RVQ codebooks, a flow-matching DiT synthesises continuous latents, and a
DAC-style Flow-VAE decodes them to a waveform.

| Component | Class | Geometry | Params | dtype on disk |
|---|---|---|---|---|
| `language_model` | `Qwen3ForCausalLM` (transformers) | 36L, hidden 4096, 32 heads / 8 KV, head_dim 128, ffn 12288, **vocab 200000**, rope_theta 1e6, max_pos 10240, `tie_word_embeddings: false` | ~8.6B | BF16 |
| `condition_encoder` | `MiniMaxMusic3ConditionEncoder` | **4 tensors**: `layer_scale`, `layer_weight_logits`, `proj.{weight,bias}`; `num_condition_layers: 8`, `condition_hidden_dim: 4096`, `out_dim: 2048` | 0.025B | F32 |
| `rvq_depth_decoder` | `MiniMaxMusic3RVQDepthDecoder` | 4L, hidden 4096, 16 heads, ffn 6144, `num_codebooks: 8`, `audio_vocab_size: 1024`, `max_position_embeddings: 16` | 0.646B | BF16 |
| `transformer` | `MiniMaxMusic3Transformer1DModel` | 36L, 32 heads × `attention_head_dim: 64` (hidden 2048), `ff_inner_dim: 8192`, `in_channels: 128`, `condition_dim: 2048`, `fourier_embedding_dim: 256`, **`rotary_dim: 32`** | 2.4B | **F32** |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | `invert_sigmas: true`, `num_train_timesteps: 1`, `shift: 1.0`, `time_shift_type: exponential`, no dynamic shifting | — | — |
| `vocoder` | `MiniMaxMusic3Vocoder` | DAC-style, `latent_channels: 128`, `upsampling_ratios: [8,8,4,2]` (hop 512), decoder hidden 1536 / in 1024, snake activations, `weight_g`/`weight_v` weight-norm | 0.054B | F32 |

Header measurement: `transformer` shard 1 of 2 carries 231 tensors / 1.240B
params, all `F32` — so the model card's "2.4B" is correct and the 9.73 GB on disk
is **fp32 storage, not a 4.9B bf16 model**. `vocoder` is 121 tensors / 0.054B with
`weight_g`+`weight_v` pairs, so weight-norm must be folded at load or reproduced.
`condition_encoder` has only four tensors, which is the finding that corrects the
obvious reading of its name: it is a **learned weighted mix over 8 LLM hidden
layers**, not an encoder tower.

**`language_model` is our existing `Qwen3ForCausalLM` architecture exactly**,
retrained on a 200 000-entry music vocabulary. Vocabulary size is a config value,
not an architecture change, and `MODEL-TEXT-qwen3-qwen3-for-causal-lm` is ✅
(token-exact 16/16). This is the single largest brick and it is already built.

### 1.1 Sample rate — RESOLVED 2026-08-13: a stage boundary, not a contradiction

The model card and SGLang-Omni's README say **32 kHz** stereo; every config says
**44100**. Both are right, about different points in the pipeline. Read from
source at the pinned SHAs:

**The vocoder natively emits 44100 Hz, 2 channels**, and that is derivable rather
than merely declared. The condition encoder's `output_sampling_rate: 44100` /
`output_hop_length: 512` set a latent frame rate of 44100/512 = **86.133 Hz**
(`condition_embedder_minimax_music3.py:40-41`; `modular_pipeline.py:48-53`
documents `latent_hop_length` as "waveform samples per Flow-VAE latent frame").
The decoder applies one `ConvTranspose1d` per `upsampling_ratios` entry
(`minimax_music3_vocoder.py:84,92-95`), so 8·8·4·2 = **512×**, and
86.133 × 512 = 44100. The declared `sampling_rate: 44100`
(`minimax_music3_vocoder.py:85`) matches the convolution stack rather than being a
stale annotation. SGLang-Omni's independent implementation agrees exactly
(`dav.py:94,115`). Stereo comes from folding the 128 latent channels into two
64-channel streams (`minimax_music3_vocoder.py:110,115`; `dav.py:140-142`).

**diffusers returns 44.1 kHz with no resample** (`modular_pipeline.py:32-36`;
`decoders.py:84-92`, whose block description says it "stitches the windows into
the final stereo waveform at 44.1 kHz"). **SGLang-Omni's server resamples
44100 → 32000 on the way out** (`constants.py:18-19` `DAV_SAMPLE_RATE` /
`OUTPUT_SAMPLE_RATE`; `acoustic.py:55-58,422-431`). diffusers' own docs state the
split: the pipeline "returns the vocoder's native 44.1 kHz stereo output. The
reference server additionally resamples to 32 kHz."

The 24000 / 960 pair in `condition_encoder/config.json` is the AR stage's 25 Hz
frame rate and is unrelated to output.

**Decision: goldens are captured at 44100 stereo** — the model's native generative
rate, resample-free, and what the primary oracle hands the caller. The 32 kHz form
is a **downstream delivery transform**, gated separately if and when SGLang-Omni
byte parity is wanted. That is not a free conversion: `acoustic.py:58` passes no
`lowpass_filter_width`, `rolloff` or `resampling_method`, so reproducing its bytes
means reproducing torchaudio's default sinc filter, not merely converting
44.1 → 32 by any correct method. **A latent-tensor parity check sits entirely
upstream of that call and cannot see the difference** — which is why the rate is
fixed here, before the first waveform golden, rather than discovered later.

---

## 2. Two packagings, one set of weights

The repository ships the model twice, which is why it is 57.4 GB:

**Native arm** — `qwen_7B/qwen_7B/` (`AbabForCausalLM`, `model_type: mixtral`,
`num_local_experts: 1`, `auto_map` → remote `modeling_abab.py`),
`flowmatching_vae.pth` (the DiT plus the condition projection), `dav.pth` (the DAC
Flow-VAE decoder). The RVQ depth decoder and the audio embedding live *inside* the
Qwen shards, as `model.audio_decoder.*` and `model.audio_extra_embedding`.

**Diffusers arm** — the six components of §1, safetensors only.

**SGLang-Omni serves the native arm**, exclusively:
`sglang_omni/models/minimax_music3/checkpoint.py:35-56` resolves exactly
`qwen_7B/qwen_7B`, `flowmatching_vae.pth` and `dav.pth`, and `load_audio_state`
(`:84-102`) selects the `model.audio_decoder.` / `model.audio_extra_embedding`
prefixes out of the Qwen shard index.

**The two are the same weights.**
`scripts/convert_minimax_music3_to_diffusers.py` at the pinned diffusers SHA loads
those three native artefacts (`:29-38`) and renames tensors into the diffusers
modules — `convert_transformer` `:47`, `convert_condition_encoder` `:86`,
`convert_vocoder` `:99`, `convert_rvq_depth_decoder` `:131`,
`convert_language_model` `:170`, which is the Qwen state *minus* the two audio
prefixes (`:189`). No retraining, no fusion, no numerical step: a re-layout.

**Decision: port the diffusers arm.** ~28.5 GB resident (17.17 + 9.73 + 1.29 +
0.22 + 0.10), safetensors only, no `torch.load` pickle path, no
`trust_remote_code`, and every component has an upstream class to gate against
one at a time. Because the conversion is a re-layout, SGLang-Omni remains a valid
**e2e and speed** cross-check rather than an incomparable second model — but that
claim is verified in W1 by comparing converted tensors against native ones, not
assumed from reading the script.

**The native arm is explicitly out of scope for loading**, and a checkpoint in
that layout is **refused by name** with a message saying so. It is not silently
mis-loaded, and it is recorded as owed rather than discovered later.

### 2.1 dtype — ON DISK IS NOT RUNNABLE (corrected 2026-08-14 by the oracle)

**An earlier revision of this section was wrong, and the correction is the point.**
It read the converter — `convert_minimax_music3_to_diffusers.py:267` defaults
`--dtype float32`, transformer/condition_encoder/vocoder take it (`:208-211`), the
RVQ depth decoder is forced to bf16 (`:214`) — saw that it matched the measured
headers exactly, and concluded that the on-disk set *was* upstream's resolved
runtime policy, to be mirrored as-is. Standing the oracle up refuted that.

**Loading the on-disk dtypes and running upstream's own pipeline raises**
`RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the
same` at `condition_embedder_minimax_music3.py:64`. The reason is that upstream
casts in exactly **two** places and nowhere else — `denoise.py:83` (condition →
`transformer.dtype`) and `decoders.py:84` (latents → `vocoder.dtype`) — so the
condition encoder and the depth decoder consume the language model's hidden states
**uncast**.

**The invariant every runnable configuration satisfies:**

```
dtype(language_model) == dtype(rvq_depth_decoder) == dtype(condition_encoder)
```

**The gated configuration is bf16 AR half / fp32 acoustic half**: language model,
depth decoder and condition encoder in bf16; transformer and vocoder in fp32. That
is the converter's default for the DiT and vocoder, and what SGLang-Omni states it
runs ("both layouts run the acoustic stage in FP32").

Two things follow, and they are the reason this correction is worth its space.
**On-disk dtype and runtime dtype are different facts about this checkpoint**, and
a per-tensor header read answers only the first — the measurement in §1 is still
correct, the inference drawn from it was not. And **fp32 on the acoustic half is
still upstream's choice rather than a too-wide accident**, so the original
conclusion survives for the DiT and vocoder even though its reasoning did not;
each fp32 buffer carries the one-line reason AGENTS.md requires, naming this
section.

**W1 therefore enforces the equality above at load time and refuses a violating
configuration BY NAME**, naming the three components and their dtypes, rather than
letting it surface as a type error deep inside a forward pass. The oracle keeps
`--dtype-policy on-disk` selectable so the failure stays reproducible.

---

## 3. Oracles

Per AGENTS.md §"When vLLM has no implementation": **`minimax_music3` is absent
from the pinned vLLM**. There are no source files under `vllm/` and the registry
carries only the MiniMax M2/M3 *text* architectures. This is the first row to
exercise the fallback rule.

| Role | Oracle | Pin | Answers |
|---|---|---|---|
| primary | `diffusers` | PR #14456 head `c6da9936` | per-component correctness, the scheduler, the conversion mapping |
| cross-check | `sglang-omni` | `748a0b43` | e2e output and the speed axis |
| supporting | `transformers` | 5.14.1 | the `Qwen3ForCausalLM` half |

**The primary oracle is an unmerged PR branch**, so the pin is the exact head SHA
and not a branch name: `huggingface:minimax-music3-integration` can be rebased or
force-pushed under us, and a comparison against "whatever the branch was that day"
is not reproducible. If the PR merges, advancing to the merge commit is a pin
advance with its own reconciliation, not a silent follow.

**Both pins are recorded** — [`../oracles/diffusers.md`](../oracles/diffusers.md)
and [`../oracles/sglang-omni.md`](../oracles/sglang-omni.md), landed in #679 and
advanced in #708. SGLang-Omni has its **own record** rather than riding on
`sglang.md`: it is a third repository with its own cadence, and this row binds to
it directly. (An earlier revision said the pins "go into `.agents/oracles/` in
W0", which read as future work and misled two implementers into reporting the
SGLang-Omni record as owed after it existed. Present tense, because the record
is a fact and not a plan.)

---

## 4. What we already own

| Need | Have | Anchor |
|---|---|---|
| Qwen3 dense forward, paged KV, sampling | ✅ token-exact 16/16 | `MODEL-TEXT-qwen3-qwen3-for-causal-lm` |
| flow-matching denoise loop | H3: fixed-step loop, DiT forwarded once per step | [minimax-h3.md](minimax-h3.md) |
| audio VAE decode + WAV writing | H3 audio VAE, LTX-2 audio VAE, `minimax_h3_wav.cpp` | `src/vllm/model_executor/models/` |
| diffusion request planning / pipeline shape | H3 planner + pipeline, LTX-2.5 pipeline | `minimax_h3_planner.cpp`, `ltx2_pipeline.cpp` |
| quantized arms on a diffusion model | H3 GGUF + NVFP4 arms | [minimax-h3.md](minimax-h3.md) §0 |
| **an audio-GENERATION engine seam** | `multimodal::SpeechEngine` + `SpeechRegistry`, landed 2026-08-13 by the IndexTTS-2.5 lane | `include/vllm/multimodal/speech_engine.h` |
| **a 1D vocoder** | `Vocoder1D`, same lane | `src/vllm/model_executor/models/vocoder1d.cpp` |

### 4.1 Music3 routes through `SpeechEngine`, which needs ONE additive extension

`multimodal::SpeechEngine` did not exist when this spec was first written. It does
now, and AGENTS.md is explicit that a capability not reachable through the shared
surface is not done, and that a seam is extended rather than forked. Music3 is a
speech-family registration, not a new engine.

It fits better than it might look. `SpeechResult` already carries `channels` and
already documents `sample_rate` as "the family's native rate ... rather than a
resampled one, so the caller decides whether to resample" — which is exactly
§1.1's 44100 stereo, and exactly why SGLang-Omni's 32 kHz stays a caller-side
concern. `requires_reference_audio()` exists so a server can refuse before
staging; Music3 returns `false`, where IndexTTS-2 returns `true`.

**The one genuine gap is `SpeechGenParams`.** It carries a single `text` field,
because IndexTTS-2 synthesises one utterance. Music3 takes **two** distinct
inputs — lyrics (with `[Verse]` / `[Chorus]` section tags) and a structured music
description — plus generation controls (duration or frame count, denoise steps,
CFG). Squeezing both into `text` with a separator would be a private protocol
inside a shared struct, which is the fork this rule exists to prevent.

W6 therefore **extends `SpeechGenParams` additively** and leaves IndexTTS-2.5's
behaviour byte-identical. A field an existing family ignores costs it nothing; a
second parallel params struct costs every future family a choice. If the
extension cannot be made additive, that is a `NEEDS_DECISION`, not a fork.

**`SpeechEngine` is not yet on the ABI.** `include/vllm.h` (v18) exposes the
video engine but no `vllm_speech_*` surface, and no open PR adds one. W6 owns
that: the ABI surface, the version bump, and the example HTTP server as a thin
client of it — never including internal headers.

Genuinely new: the eight-codebook RVQ frame path, the depth decoder, the learned
8-layer condition mix, snake-activated DAC decoding with weight-norm, and the
LLM→diffusion handoff on *continuous hidden states* rather than discrete tokens.

---

## 4G. The row's structured record

| Field | Value |
|---|---|
| Scope | IN: the diffusers-arm six-component checkpoint, all five modules, load through waveform; lyrics + structured description in, 44100 Hz stereo out; registration as a `SpeechRegistry` family with the `vllm_speech_*` ABI and the example HTTP server as a thin client (§4.1); quantized arms incl. GGUF k-quants (W7). OUT: the native `AbabForCausalLM` + `.pth` arm, refused by name (§2); streaming, which upstream does not support and which is refused rather than faked; a 32 kHz delivery arm, which is a downstream resample gated separately (§1.1); any change to `SpeechEngine` behaviour for IndexTTS-2.5. |
| Upstream chain | `minimax_music3` is ABSENT from the pinned vLLM, from vLLM `main` and from `vllm-omni` — this row is why AGENTS.md §"When vLLM has no implementation" exists. Primary oracle `diffusers` PR [#14456](https://github.com/huggingface/diffusers/pull/14456) head `c6da9936` (OPEN), [`../oracles/diffusers.md`](../oracles/diffusers.md) `gateable = yes`. Cross-check SGLang-Omni `748a0b43`, [`../oracles/sglang-omni.md`](../oracles/sglang-omni.md) `gateable = no`, which serves the NATIVE layout (§2). `transformers` 5.14.1 for the `Qwen3ForCausalLM` half. Checkpoint `MiniMaxAI/MiniMax-Music3` diffusers arm, 27 GB, at `/mnt/nas_share/checkpoints/minimax-music3`. |
| Our baseline | LANDED for this row: the modular loader `minimax_music3_loader.{h,cpp}` (#714, 1413/1413 assertions against the real tree, all 1012 tensors accounted, native arm refused by name) and the gateable oracle `tools/oracle/music3_oracle.py` with 13 per-stage goldens (#708). REUSED rather than rebuilt: the token-exact Qwen3 dense forward and paged KV, the `vocoder1d` primitives, `multimodal::SpeechEngine`, and the H3 / LTX-2.5 flow-matching and audio-VAE precedent (§4, §4.1). Before this row there was no music generation and no text-to-audio path of any kind. |
| Port map | loader -> `src/vllm/model_executor/models/minimax_music3_loader.cpp` (LANDED, from `scripts/convert_minimax_music3_to_diffusers.py`). `language_model` -> the landed Qwen3 dense path (W2). `condition_embedder_minimax_music3.py` -> W3. `minimax_music3_rvq_depth_decoder.py` -> W3. `transformer_minimax_music3.py` + `FlowMatchEulerDiscreteScheduler` -> W4. `minimax_music3_vocoder.py` -> W5, over the shared `vocoder1d` primitives. `modular_pipelines/minimax_music3/{encoders,before_denoise,denoise,decoders}.py` -> W6. |
| Tests to port | Upstream ships NO unit tests for this model at the pinned SHA — the PR carries docs, a conversion script and the modules, and nothing test-shaped was found in the seven PR files fetched. So the references are CAPTURED, not ported, and this spec says so rather than implying a port that never happened: `tests/parity/goldens/minimax_music3_oracle/` holds per-stage tensors with a manifest recording shape, dtype, sha256 and min/max/mean per entry. Each phase gates against its own stage's entry. If upstream later adds tests, they are ported in the same change that touches the corresponding module. |
| Gates | Split by half, and conflating them is the failure mode §0 warns about. LLM half: TOKEN-EXACT against `rvq_codes.npy` `[26,8]` int32, where row 0 is the priming decode that emits no frame so `rows[1:]` align with the 25 frames. Acoustic half: per-stage tensor parity at fixed seed and reduced dimensions against `condition_chunk0`, `denoise_{first,last}_*`, `vocoder_input_chunk0`, `waveform` — no logits exist, so no token gate does either. A correlation coefficient is NOT a gate here: Pearson is scale-invariant and cannot see a uniformly scaled latent. Speed is measured against SGLang-Omni in its production configuration (both CUDA graphs, compiled DIT and DAV, batched seeded sampling), never with those disabled. |
| Dependencies | `multimodal::SpeechEngine` + `SpeechRegistry` for W6, extended additively per §4.1 with IndexTTS-2.5 left byte-identical. The landed Qwen3 dense forward and paged KV for W2. The `vocoder1d` primitives for W5. The diffusers oracle staying gateable at its pin, for every phase. NO dependency on vLLM-Omni, and none on `dgx.casa`, which was down throughout W0 — the correctness gate runs on CPU by design. |
| Work breakdown | §6. W0 spec + both oracle pins + §1.1 + oracle stand-up (DONE). W1 modular loader, weight-norm folding, dtype invariant, native-arm refusal (DONE). W2 global LLM. W3 condition mix + RVQ depth decoder. W4 flow-matching DiT + scheduler. W5 vocoder over `vocoder1d`. W6 speech-family registration + `vllm_speech_*` ABI + example HTTP server. W7 quantized arms, anything unimplemented refused by name. |
| Risks/decisions | The primary oracle is an OPEN PR: it may be rebased or refactored in review, so the pin is the head SHA and the W1 tensor mapping is re-checked at merge. The on-disk dtype set is NOT runnable (§2.1) — an early revision of this spec asserted the opposite, and the correction is why the loader enforces `dtype(LM) == dtype(rvq) == dtype(cond)` and refuses violations by name. fp32 on the acoustic half is upstream's choice, mirrored, and sets a speed baseline in a regime this project has not optimised for — W7 is where that becomes interesting. The 5000-token prompt and 9000-frame ceilings are enforced, not discovered. Non-streaming is refused by name rather than buffered and called streaming. |

## 5. Gates

**LLM half — token-exact. WITHDRAWN 2026-08-14 by W2/W3; the artifact refuted
it.** What this paragraph said was: "The global LLM and the depth decoder emit
discrete RVQ codes. Greedy decode of the code sequence is compared against the
oracle token-for-token on a fixed prompt. This is a real token gate and it
binds." It is kept in full, because a withdrawn claim that leaves no trace is how
the same wrong gate gets re-specified.

**There is no greedy decode of this model to compare against.** `_sample_top_k`
(`encoders.py:94-103`) is the only sampler either stage uses; `_AR_SAMPLING_TOP_K`
is a module constant of 50, there is no temperature and no argmax branch, and the
last line is `torch.multinomial(probs, 1, generator=generator)`. The committed
`rvq_codes.npy` is therefore a **seeded sample**, and reproducing it
token-for-token means reproducing torch's CPU Mersenne-Twister and its
multinomial — a claim about torch's RNG, not about this model.

A second, independent reason the same conclusion holds, and the one that would
survive even a bit-exact RNG: **both** stages sample from a CFG mix of a
conditional and an unconditional row (`encoders.py:327-328`, `:134-135`), and the
goldens store the **conditional row only** (`encoders.py:132,343`, both slice
`[:1]`). The unconditional branch is not in the golden set, so the guided
distribution the codes were drawn from cannot be reconstructed from what is
committed.

**What replaces it.** The codes are consumed as INPUTS and the AR half is gated
on TENSORS, at two scales:

* reduced dimensions, float32, against goldens produced by *executing* upstream's
  own `MiniMaxMusic3ConditionEncoder` and `MiniMaxMusic3RVQDepthDecoder`
  (`scripts/gen-minimax-music3-ar-goldens.py`). This separates an algebra defect
  from rounding, and it runs in CI with no checkpoint;
* full scale, bf16, real weights: the condition mix against
  `condition_chunk0.npy` (176 128 values) and the depth decoder against
  `frame_hiddens[:, 4096:]` (716 800 values), driven by the golden codes and the
  golden `last_hidden`.

The full-scale bound is calibrated against a **matched control** rather than
guessed. torch's own `sdpa_kernel(MATH)` arm, running upstream's own module on
the identical inputs, reproduces the goldens to 46.34% bit-identical at mean
absolute error 1.659e-03 — its CPU attention kernel runs a blocked online softmax
that no closed-form rounding model reproduced. Ours is 43.61% and 1.824e-03,
inside that spread. Chasing a particular kernel's rounding below the control is
not "more correct" (AGENTS.md's near-tie discipline).

**The LLM half is now gated too, and the same way.** `frame_hiddens[:, :4096]`
is the language model's own hidden state; it is reproduced by running the model
TEACHER-FORCED on the golden codes through the landed Qwen3 dense path, reached
by the additive `Qwen3DenseModel::ForwardEmbeds` entry (see `## Now`). The bound
is in bf16 ULPs against a matched control, and it is joined by a RANK statistic
that no scaling can pass: the oracle's own sampled semantic codes rank 2.48 on
average under our reproduced guided logits, where chance over the 16384-entry
semantic window is 8191.5.

**Acoustic half — per-stage tensor parity.** No logits, no sampler, so no token
gate exists to have. Each stage is compared against the oracle's own output for
the same input at a fixed seed: condition mix, DiT output per step, VAE latents,
waveform. Following H3, the exact correctness gate runs upstream at **reduced
dimensions on CPU**, which is available today and does not depend on the 57 GB
checkpoint fitting anywhere.

**A correlation coefficient is not a gate on this path.** Pearson is
scale-invariant, so a uniformly scaled latent passes it while sounding wrong;
bounds are on absolute and relative error with a stated tolerance, per component.

**Speed** is measured against SGLang-Omni in its production configuration —
its documented defaults are backbone decode CUDA graph, RVQ depth CUDA graph,
compiled DIT blocks, compiled DAV decoder and batched seeded sampling. Comparing
against it with those off would be a dishonest denominator.

---

## 6. Phases (work breakdown)

Each phase is dispatched to a **fresh implementer** from this spec, reviewed by a
**fresh reviewer** who mutates the claimed guarantees, and its gate is rerun by
the operator. Phases are separately claimable except where noted.

| Phase | Scope | Done when |
|---|---|---|
| **W0** | This spec; both oracle records pinned; §1.1 sample rate settled from source (**DONE**) and confirmed at runtime (**DONE**); stand the diffusers oracle up and prove it builds and runs (**DONE**, `tools/oracle/music3_oracle.py`) | oracle executes the model and `diffusers.md` flips to `gateable = yes` with a path as evidence |
| **W1** (**DONE**) | Modular loader: the six-component layout, weight-norm folding, the fp32/bf16 policy of §2.1, native-arm refusal by name | every component loads with shapes asserted against §1; converted-vs-native tensor equality checked, not assumed |
| **W2** (**DONE**) | Global LLM on our landed Qwen3 path at vocab 200 000 | hidden-state parity against the oracle capture — DONE: an additive `inputs_embeds` entry on the dense path (bit-identical to the token-id forward), 25 teacher-forced steps inside a measured torch-vs-torch control, and the oracle's own codes ranking 2.48 under our guided logits. The token-exact RVQ code parity this row once promised is WITHDRAWN (§5): upstream has no greedy path |
| **W3** (**DONE**) | Condition mix (8-layer weighted) + RVQ depth decoder, 8 codebooks | per-stage tensor parity; the depth decoder's 16-position window exercised at its boundary |
| **W4** (**DONE**) | Flow-matching DiT + `FlowMatchEulerDiscreteScheduler` with `invert_sigmas` | per-step latent parity against the oracle at a fixed seed — DONE: scheduler BIT-EXACT on both recorded steps, DiT guided velocity inside the measured torch-vs-torch control |
| **W5** (**DONE**) | Vocoder **through the shared `vocoder1d` primitives** (§4.1): snake activations, weight-norm, `[8,8,4,2]` upsampling, the 128→2×64 stereo fold, at **44100 stereo** (§1.1) | waveform parity within a stated absolute tolerance, and H3/IndexTTS-2.5 behaviour byte-identical — DONE: 88 064 samples, 0 outside tolerance, `vocoder1d` unmodified. WAV WRITING itself is W6's, with the rest of the delivery surface |
| **W6** (**DONE**) | Register as a `SpeechRegistry` family; extend `SpeechGenParams` ADDITIVELY for lyrics + description + controls (§4.1); NEW `vllm_speech_*` **`include/vllm.h`** surface with the ABI version bump; **the example HTTP server as a thin ABI client** | a song generates end to end from an HTTP request; IndexTTS-2.5 unchanged; SGLang-Omni cross-check; speed axis recorded with values and ratios |
| **W7** (**DONE**) | Quantized arms — GGUF k-quants are a standing requirement, not a per-model choice | each arm gated, or refused by name and recorded as owed — BOTH branches exercised. Quantized checkpoints for this model DO exist (14 repos, 5 formats, surveyed with counts in §9.1). The RVQ depth decoder's GGUF Q4_K arm is IMPLEMENTED and value-gated against a pinned artifact at a DERIVED bound (§9.6); every other component, lineage and format is diagnosed and refused BY NAME with the missing piece. 29/125 without a checkpoint plus 6/319 against the artifact; 18 of 18 mutations RED. §9.6 records the gate-design finding: an upper-bound-only tolerance cannot separate a real quantized arm from a dequant fallback, because the fallback is CLOSER |

**W0 blocks everything.** Until the oracle demonstrably runs, no phase can produce
evidence, and an implementer told to "gate against diffusers" would have nothing
to gate against.

---

## 7. Risks

**The oracle is an open PR.** It may be rebased, refactored in review, or renamed
before merge — the H3 integration had exactly that follow-up (#14371 refactoring
#14355). Pinning the head SHA makes us reproducible but not immune: a merged
version that renames tensors invalidates the W1 mapping. Re-check at merge.

**fp32 on the acoustic path is 2.4B + 0.054B of fp32 weights and fp32 compute.**
That is upstream's choice and we mirror it, but it sets the speed baseline in a
regime this project has mostly not optimised for, and the quantized arms in W7 are
where that becomes interesting rather than a footnote.

**The 5 000-token prompt and 9 000-frame ceilings** are documented model limits.
They are context limits on our side too and must be enforced, not discovered.

**Non-streaming only, upstream.** Refuse a streaming request by name rather than
buffering silently and calling it streaming.

---

## 8. Stop conditions

Stop and report `NEEDS_DECISION` rather than proceeding if: the diffusers PR is
closed unmerged or force-pushed to an incompatible tree; or the converted-vs-native
tensor check in W1 finds the two packagings are *not* the same weights, which
invalidates the SGLang-Omni cross-check and this spec's §2 decision.

Stop and report `NEEDS_CONTEXT` if the checkpoint cannot be fetched to the box the
gate runs on, or if a component's upstream class has no readable definition at the
pinned SHA.

---

## 9. W7 — the quantized arms: the survey, and what is owed

### 9.1 The survey, and why it is written down rather than summarized

W7 opened with the question AGENTS.md forces: *which* quantized arms does this
checkpoint family actually ship? The H3 precedent (§0 of
[minimax-h3.md](minimax-h3.md)) is why the question is asked before any code is
written — there, third-party GGUF and NVFP4 arms changed a row's verdict from
"hardware-blocked" to "reachable", and reasoning from the first-party release
alone had produced the wrong conclusion.

**Every query is recorded with its result count**, because an absence claimed
from a search nobody can re-run is not evidence. Queries were run against the
HuggingFace HTTP API (authoritative; `search=` is substring-over-repo-id) on
2026-08-14, with web search used only as a labelled cross-check and every repo id
it produced re-verified against the API.

| Query | Endpoint | Results | Notable |
|---|---|---|---|
| `?author=MiniMaxAI&limit=200` | models | 30 | `MiniMaxAI/MiniMax-Music3` (the base repo); **no quantized repo under the org** |
| `?author=MiniMax&limit=200` / `?author=MiniMax-AI&limit=200` | models | 0 / 0 | the org id is `MiniMaxAI` |
| `?search=music3&limit=100` | models | 68 | 14 are MiniMax-Music3 derivatives |
| `?search=minimax-music&limit=100` | models | 42 | |
| `?search=MiniMax-Music3&limit=100` | models | 27 | |
| `?filter=gguf&search=minimax` | models | 11 | 6 are Music3 |
| `?filter=gguf&search=music3` | models | 6 | |
| `?search=music3-gguf` / `-nvfp4` / `-awq` / `-fp8` / `-int8` / `-w4a8` | models | 6 / 0 / 0 / 0 / 1 / 2 | |
| `?author=<quantizer>` for QuantStack, city96, calcuis, Kijai, mradermacher, bartowski, unsloth, nvidia, RedHatAI | models | 9 authors swept, **0 Music3 repos** | the usual quantizers had not touched it |

**The finding is that first-party ships bf16/fp32 ONLY, and the community had
already published fourteen quantized repositories in five formats within days of
the release.** The relevant ones:

| Format | Repos | Coverage |
|---|---|---|
| GGUF | `audio-cpp/MiniMax-Music3-GGUF` | **all five components**, one GGUF each, bf16 and Q4_K arms |
| GGUF | `scragnog/MiniMax-Music3-GGUF` | 2-file split (`mm3-lm-*` / `mm3-synth-*`), 13 tiers incl. MXFP4 and NVFP4 as GGML tensor types |
| GGUF | `Abiray/…`, `realrebelai/MiniMax-Music-3_GGUFs`, `molbal/…`, `ChrisColeTech/…` | the 2.46B **DiT alone**, ComfyUI-style, Q2_K…Q8_0 (0.9–2.7 GB) |
| int8 / w4a8 | `Comfy-Org/MiniMax-Music-3` (`_int8_convrot`), `NidAll/MiniMax-Music3-W4A8`, `dummy9996/…-w4a8-bf16-comfyui` | DiT |
| MLX 4/6/8-bit | `ddalcu/…`, `vanch007/…`, `elishabjm/…` | |
| proprietary | `infosave/MiniMax-Music-3-cmf` (Cortiq 4-bit) | not implementable; recorded only |

**NOT found by the queries above**: AWQ, GPTQ, compressed-tensors, fp8 /
`fp8_e4m3fn` / `fp8_scaled`, or bitsandbytes. That is "not found by these
queries on this date", never "does not exist".

### 9.2 The GGUF headers, MEASURED — and the finding that matters

Ten published GGUFs had their **headers read by HTTP range request** (56 MiB
total; the metadata and tensor-info table sit at the start of the file, so no
weight byte was fetched). This is the same instrument §1 used for the bf16 arm's
geometry, and it is what turns §9.1's repo list into a contract.

**"The GGUF arm" is THREE MUTUALLY INCOMPATIBLE LINEAGES, and
`general.architecture` cannot separate them.** It reads `audiocpp`, `mm3`,
`qwen3` and `wan` across files of the same model — and `wan` collides with
genuine Wan video GGUFs, so keying on it would bind another model's checkpoint.
The usable discriminators are:

* `audiocpp.model_spec.family == "minimax_music3"` — **EXACT diffusers tensor
  names, no rename table**, but the geometry lives only in the sibling
  `config.json`, so the file is not self-describing;
* `mm3.model == "MiniMax-Music3"` — **fully self-describing metadata**, but it
  needs a rename table *plus* fused QKV to split and folded weight-norm to
  invert;
* co-occurring `diffusion_transformer.` + `latent_conditioners.` prefixes — the
  ComfyUI lineage, which ships **the DiT and condition encoder only**: no
  language model, no depth decoder, no vocoder, so it **cannot generate audio by
  itself** whatever we implement.

Two further measurements: `comfy.gguf.orig_shape.*` — the H3 arm's shape-override
key — is **absent from all ten files** (0 occurrences), so H3's reshape handling
does not carry over; and scragnog's NVFP4 tier uses GGML tensor type id **40**,
which is not a standard llama.cpp id and would need its own resolution.

### 9.3 What W7 implemented, and what it deliberately did not

**ONE arm is implemented and value-gated — the RVQ depth decoder at GGUF Q4_K
(§9.6). The other four are refused by name and owed (§9.5).** That split is not
a compromise: §9.6 records why this component is the only one whose bound can be
*derived* today rather than asserted, and adding a second arm to look thorough
would have meant claiming tolerances nothing measured.

**W7 also implemented the refusal**, which AGENTS.md makes non-optional whether
or not a checkpoint exists: *"an arm that is not implemented is refused with a
message naming the missing piece and recorded as owed, never left to be
discovered later."*

`minimax_music3_quant.{h,cpp}` is a **separate translation unit**, per
[porting-a-model.md](../porting-a-model.md) ("GGUF is its own translation unit,
not an afterthought bolted onto the safetensors loader") and per the H3 layout
(`minimax_h3_gguf.cpp`, `minimax_h3_nvfp4.cpp`). When an arm lands it lands
beside this file and the detector routes to it; the detection is not embedded in
W1's loader, so the first real arm is not a rewrite of W1.

**Three detectors, because a quantized checkpoint announces itself in three
places and no single detector sees all three:**

| Level | Sees | What it caught that W1 could not |
|---|---|---|
| TREE | `.gguf` files (nested to depth 2), and any `config.json` declaring a quantization | a GGUF tree has **none** of the seven diffusers directories, so W1 told it "missing transformer, condition_encoder, …" — seven directories the user does not have and never will, with no mention of GGUF |
| MANIFEST | the NVFP4 triple, MXFP4 packs, AWQ `qweight`, bitsandbytes `absmax`, and **dtype-only** formats (fp8, int8) | a real NVFP4 `condition_encoder` refused on **`layer_scale`** — a tensor that is not quantized, is not wrong, and has nothing to do with the problem; it is simply the name that sorts first |
| CONFIG | `quantization_config.quant_method`, and MLX's bare `quantization` | nothing: W1 ignored both, so an MLX or compressed-tensors tree fell through to a shape mismatch |

**What the detector refuses to guess.** A bare `weight_scale` with no
`weight_scale_2` and no `weight_packed` is consistent with NVFP4 missing its
global scale, with a compressed-tensors block scheme, and with a per-channel int8
scale. It resolves to `kUnknownScheme` and the refusal **names all three
candidates rather than picking one** — `ltx2_loader.h:232-268` records what
picking one costs: a finite, correctly shaped, correctly scaled, WRONG result
that no shape gate can see.

### 9.4 Evidence

`tests/vllm/models/test_minimax_music3_quant.cpp` — **29 cases / 125 assertions**,
no checkpoint and no network. `tests/parity/test_minimax_music3_quant_real.cpp` —
**6 cases / 319 assertions** against the pinned artifact (§9.6), skipping loudly
without it.

RED first, against the tree as it stood: a probe asserting that a GGUF tree, an
NVFP4 component and an fp8 component are each diagnosed by format failed **8 of
8** checks, and printed the three misleading messages quoted in §9.3's table. The
same probe passes 8 of 8 after the change.

**Eighteen mutations across both layers, all eighteen RED** — each hook removed,
each detection rule neutered, each clause of the refusal text deleted, the
dequant given the wrong ggml type, the resident-type report falsified, the GGUF
dim order reversed a second time, and each lineage guard defeated; sources
restored and verified `sha256`-identical, final rebuild green. Three are worth
recording rather than counting:

* **One mutation initially STAYED GREEN** and it was a genuine coverage hole:
  hardcoding `matched = 1` passed every case in the file, because each happened
  to carry exactly one marker. The count was reported but never *discriminated*,
  so a refusal reading "1 of 400" on a fully quantized checkpoint would have read
  as one stray tensor. A 36-marker case was added; the mutation then fires.
* **One mutation was INVALID as first written** — reverting the config hook by
  deleting its only caller tripped `-Werror` unused-function, so the *compiler*
  refused it and the gate never got to speak. A build failure is not a red gate.
  Re-run as neutering the call while keeping the function used: RED.

The negatives are gated too, because a detector that fired on the shipped
checkpoint would refuse every real load: the bf16/fp32 dtypes, the real
transformer config, a `null` quantization_config, and — specifically — the
vocoder's 30 legacy `weight_g`/`weight_v` weight-norm pairs, which are a
*parameterization* and not a quantization. The existing suites are unchanged:
loader 21/1393, AR 25/338, acoustic 27/265, speech 9/222.

### 9.5 What is OWED

**The remaining GGUF k-quant arms are the highest-priority debt on this row.**
The bf16/fp32 arm is ~28.5 GB; the same weights at Q4_K are ~9 GB. That is the
difference between a model most users cannot run and one they can, and it is
what a quant-matched llama.cpp comparison needs.

| Candidate | Size | State |
|---|---|---|
| `audio-cpp/MiniMax-Music3-GGUF` → `rvq_depth_decoder_q4_k` | 406 MB | **DONE — §9.6** |
| `audio-cpp/…` → `transformer_q4_k` | 1 396 MB | OWED. Gateable against W4's per-step latents, whose fp32 control is measured, so its bound is derivable the same way |
| `audio-cpp/…` → `language_model_q4_k` | 7 184 MB | OWED, and blocked behind W2's LM forward regardless |
| `audio-cpp/…` → `vocoder`, `condition_encoder` | 217 / 101 MB | OWED, and note these are **bf16 GGUF, not k-quant** — same size as the safetensors, so they buy nothing |

Also owed, and recorded rather than discovered later: **the ComfyUI-lineage
GGUFs can never be a complete arm** (§9.2 — DiT and condition encoder only), so
a user pointing one at us must be told that even a finished GGUF arm would not
make their file generate audio; **the `mm3` (scragnog) lineage** needs a rename
table plus fused QKV to split and folded weight-norm to invert, and is refused
by name; the Cortiq `.cmf` format is proprietary and is recorded as not
implementable rather than owed; and MLX and bitsandbytes are **new shared seams**
this project implements for no model, so they are not per-model additions.

### 9.6 The one arm that IS implemented — and the gate-design finding

**Artifact, pinned. An unpinned quantized checkpoint is not reproducible**, and
this project has already been bitten by a repo re-quantized in place under an
unchanged id:

| Field | Value |
|---|---|
| repo | `audio-cpp/MiniMax-Music3-GGUF` |
| revision | `c36aaeed683f33b05796788e4204f4eeba8fa547` |
| file | `rvq_depth_decoder_q4_k.gguf` |
| size | 405 752 480 bytes |
| sha256 | `4c5d41b27418d9c1046345f649cb61d7cde0e3bbda4af7f7cb142df2c70cbdd0` |
| staged at | `$CHECKPOINT_ROOT/minimax-music3-gguf/` |

The digest was verified against the repository's own LFS record at fetch time.
47 tensors — exactly the count the safetensors contract owes — as **36 Q4_K
projections, 9 BF16 norms and 2 F16 embedding tables**, with
`audiocpp.tensor_name_format = native`, so every name binds to
`EnumerateMiniMaxMusic3RvqDepthDecoderTensors` with **no rename table**.

**THE FINDING, and it generalizes past this row: for a quantized arm, an
upper-bound-only tolerance cannot distinguish a real quantized path from a
silent dequant fallback — because the fallback is CLOSER to the golden.**

Measured, on the identical forward over identical inputs against the W3 golden's
716 800 values:

| Arm | bit-identical | mean\|d\| | max\|d\| |
|---|---|---|---|
| Q4_K, the real quantized path | 2.84 % | **0.0324** | 0.3125 |
| bf16 weights — a simulated dequant fallback | 43.61 % | **0.00182** | 0.125 |
| (bf16 control from §5, for reference) | 46.34 % | 0.00166 | 0.125 |

The fallback is **17.8x closer** to the golden than the genuine arm. So every
upper bound the Q4_K arm could plausibly be given — mean, max, identical
fraction — *passes on the fallback*, and passes comfortably. A gate built only
from upper bounds would have reported the quantized arm correct while the
quantized bytes were never read. That is the dequant-fallback class this project
has been bitten by before, and the reason it survives review is that the failing
configuration produces *better-looking* numbers than the passing one.

**What catches it is a LOWER bound**: the deviation must EXCEED the unquantized
arm's. `kQ4KMeanAbsFloor = 5e-3` sits between the two measurements (0.00182 <
0.005 < 0.0324). A result that is "too good" on a quantized arm is not a better
port; it is a different set of weights.

**The lower bound and the resident-dtype proof are complementary, not
redundant**, and W7 keeps both. The resident-type report is *bookkeeping the
loader does about itself* — it proves which bytes were selected, and it is what
localizes a fault to a named tensor — but a loader that lied about its own
tallies would still pass it (mutation QM2 exists precisely to show that
assertion has teeth). The lower bound is *a property of the output* and needs no
cooperation from the loader at all. And the Q4_K **lattice** check is a third,
independent leg: within any 32-element sub-block the dequantized values take at
most 16 distinct values, which is a structural signature of 4-bit block
quantization that a bf16 read cannot produce — measured 0 of 524 288 windows
over 16, against a bf16 control from the same file at 127 of 128 windows over 16.
Three legs, keyed on three different things: what the loader selected, what the
data structurally is, and how the output behaves.

**Tolerances, each derived and each printed on every run:**

| Bound | Measured | Set to | Why it still discriminates |
|---|---|---|---|
| worst per-tensor relative L2, Q4_K | 0.0742 | < 0.10 | a mis-decoded k-quant block yields ~1.0 — an order of magnitude of margin |
| per-tensor relative L2, BF16 islands | exactly 0 | == 0 | stored unquantized; anything else is a mangled read |
| per-tensor relative L2, F16 islands | 3.23e-08 / 2.37e-08 | < 1e-6 | see the F16 finding below |
| full-scale mean\|d\| | 0.0324 | 5e-3 … 0.045 | two-sided; the floor is the fallback detector above |
| full-scale max\|d\| | 0.3125 | < 0.50 | |
| full-scale bit-identical | 2.84 % | > 2 % | |

**A second finding, small but exactly the kind that is otherwise discovered
later: the two F16 islands do NOT round-trip exactly, and it is the artifact's
doing, not ours.** The quantizer re-encoded `audio_embeddings.weight` and
`pos_embedding.weight` from BF16 to F16. F16's exponent range is *narrower* than
BF16's, so weights below ~6e-08 flush to zero and cannot come back. The gate
therefore splits the islands: BF16-stored tensors must be bit-exact, F16-stored
tensors are bounded. Asserting "unquantized means exact" for all eleven — which
is what the first draft did — reds on a correct reader.

**No speed number is claimed.** Nothing here was measured for throughput, the
box is CPU-only for this row, and a 4-bit arm's speed on a path with no
quantized GEMM would be a number about dequantization, not about the model.

---

## Now

**W0 + W1 DONE, W3 DONE, W2 PARTIAL; row `ACTIVE`.** The diffusers oracle
generates audio and is `gateable = yes` against 13 committed per-stage goldens;
both oracles are pinned; §1.1 is resolved and confirmed at runtime; the modular
loader is in the tree with the dtype invariant §2.1 enforced and the native arm
refused by name. W2/W3 add the autoregressive half's compute —
[`minimax_music3_ar.h`](../../include/vllm/model_executor/models/minimax_music3_ar.h)
and its two gates. Nothing generates a song yet.

**W3 is complete and gated at both scales.** The learned 8-layer condition mix
reproduces `condition_chunk0.npy` to 175 989 of 176 128 values **bit-identical**
(mean absolute error 1.99e-07, no value beyond one bf16 ULP-or-2^-7), and the
4-layer RVQ depth decoder reproduces `frame_hiddens[:, 4096:]` — 716 800 values
over 25 frames × 7 depth steps — inside the matched control's spread (§5). The
16-position window is exercised at its boundary and one past it. The reduced
dimension gate is 25 cases / 338 assertions and needs no checkpoint.

**W2 is partial, and the split is exact.** Everything the autoregressive loop
does *around* the language model has landed and is gated: the prompt the
checkpoint contract fixes (both upstream rewrite passes, string for string, on
the oracle capture's own prompt), the unconditional CFG row, the frame budget and
its two refusals, the semantic vocabulary mask, the guided-logit pipeline
including the re-mask that keeps a NaN from becoming a candidate, `_sample_top_k`
up to its draw, and the frame feedback embedding. What has NOT landed is the
8.6B `Qwen3ForCausalLM` forward itself — see §5's "still owed".

**§5's token-exact claim is withdrawn**, and that is this phase's most important
finding rather than a footnote: upstream's AR stage has no greedy path at all, so
`rvq_codes.npy` is a seeded sample and is consumed as an input by these gates.
§5 now records the reasoning and the tensor gates that replace it.

**W4 + W5 are complete and gated at both scales.**
[`minimax_music3_acoustic.h`](../../include/vllm/model_executor/models/minimax_music3_acoustic.h)
carries the flow-matching DiT, the `FlowMatchEulerDiscreteScheduler`, the CFG
mix, the denoise loop's window bookkeeping and the DAC Flow-VAE vocoder. Latents
now become a waveform.

* **Reduced dimensions, float32, no checkpoint:** 27 cases / 265 assertions
  against goldens produced by *executing* upstream's own
  `MiniMaxMusic3Transformer1DModel`, `MiniMaxMusic3Vocoder`,
  `FlowMatchEulerDiscreteScheduler` and `ClassifierFreeGuidance`
  (`scripts/gen-minimax-music3-acoustic-goldens.py`).
* **Full scale, float32, real weights:** the scheduler step reproduces the
  capture's own trajectory **bit-exactly** at both recorded steps (22 016 of
  22 016 values), `denoise_last_latents_out` is bit-identical to
  `vocoder_input_chunk0` so the stage handoff is proved rather than assumed, and
  the 0.054B vocoder reproduces `waveform.npy` over **88 064 samples** with zero
  values outside tolerance (mean |d| 3.19e-08, max |d| 3.18e-07). The 2.4B DiT
  arm reproduces the guided velocity at both recorded steps and is opt-in behind
  `VLLM_CPP_MUSIC3_DIT` because it is four 2.4B fp32 host forwards.

**The full-scale bounds are calibrated against a measured control**, not chosen:
upstream's own modules on the identical inputs under `torch.set_num_threads(1)`
— the capture ran at the box's default 20 — reproduce the goldens to 1.911 %
bit-identical (vocoder, mean |d| 3.015e-08, max |d| 3.576e-07) and 15.416 % /
5.596 % (DiT first / last step, mean |d| 7.526e-07 / 1.424e-06). Two correct
float32 implementations differ by that much on these tensors, so no bit-exact
claim is made where none is available — and the **absolute** floor is what binds,
because the control's own max *relative* deviation is 7.4e-02, attained on
near-zero samples.

**Three findings from this phase, recorded because each was nearly missed.**
(1) `minimax_music3_loader.h` documented `folded == 20` for the shipped vocoder
while its own weight-norm paragraph counts 30 and the checkpoint yields 30; the
comment is corrected in the same change. (2) A relative tolerance of 1e-5 is
loose enough to hide upstream's `(1 - 1e-6)` overlap-blend factor, which moves
values by only 3.3e-07 relative — the mutation stayed **green** until the blend
assertion became bit-exact, which it can be because the blend has no reduction.
(3) CFG at scale 1 does **not** recover the conditional row bit-for-bit in
float32 (10 of 12 values here), so the gate asserts scale **0** against the
unconditional row instead, which is exact and is what actually discriminates the
two formulations.

Next: W2's remaining 8.6B language-model forward, W6 the speech-family
registration plus the `vllm_speech_*` ABI and the example HTTP server, W7 the
quantized arms. Nothing generates a song end to end yet — W6 is what joins the
two halves.

**W6 is complete: the model reaches the SHARED SURFACE.**
[`minimax_music3_speech.h`](../../include/vllm/model_executor/models/minimax_music3_speech.h)
registers `minimax-music3` as a `SpeechRegistry` family — detection INSPECTS
`modular_model_index.json` for the pipeline CLASS plus all seven component
directories, never the path spelling — declares 44100 Hz stereo and
`requires_reference_audio() == false`, and composes the four modular-pipeline
blocks nothing had composed: `before_denoise.py` -> `Music3ChunkPlan`,
`denoise.py` -> `Music3DenoiseChunks`, `decoders.py` -> `Music3DecodeChunks`.

**§4.1's additive extension held.** `multimodal::SpeechGenParams` grew by
`lyrics`, `description`, `audio_duration_s`, `num_inference_steps` and
`guidance_scale`; every default means "the family decides", and `guidance_scale`
uses a NEGATIVE sentinel because **0 is a legal guidance scale** and a
0-means-default would make the unconditional branch unreachable. IndexTTS-2.5 is
byte-identical: `indextts2.{h,cpp}` and `speech_engine.cpp` have zero lines
changed, and its gates still read 4 cases / 8 assertions and 7 cases / 20.

**The ABI is v20, not v19.** `origin/main` took v19 for the multimodal input
limits (#607 L2) while this phase was in flight. That is a renumber, not a
conflict: the speech surface is appended and no v19 field moved.

**The route is `POST /v1/audio/speech`**, OpenAI's createSpeech spelling with the
two music inputs as ADDITIONAL named fields, registered only when a synthesizer
is attached. `voice`, `speed`, streaming and any non-`wav` `response_format` are
refused BY NAME. The `requires_reference_audio()` refusal fires BEFORE the runner
is called, which is the reason that method exists on the seam.

**What W6 gates, and the tolerance that binds.** The delivery path reproduces
`waveform.npy` over 88 064 values with 0 outside W5's own measured bound, and the
WAV payload is BIT-EXACT against the quantization of that golden (88 064 int16
samples, 0 mismatched) — there is no reduction there, so a tolerance would be
slack for no reason. The WHOLE TAIL, driven from `frame_hiddens.npy` and the
capture's own `denoise_first_sample_in.npy` through the condition mix, four
guided 2.4B DiT steps and the vocoder, lands at max|d| 4.523e-06 / mean|d|
1.225e-07 on the waveform and max|d| 2.396e-05 on the latents — exactly where the
DiT's own measured per-step error carried over four Euler steps says it should,
so the composition introduced no error of its own. The first bounds written for
this file were 5e-4/5e-5; the measurement showed them to be ~100x slack and they
were tightened to under an order of magnitude of headroom, because a bound nobody
measured is not a bound.

**A request's waveform can never equal the golden, and that is structural.**
The AR codes are a seeded `torch.multinomial` draw (§5) and the denoise loop's
initial latents are a seeded `randn_tensor` (`denoise.py:117-121`). So
`Music3NoiseSource` is a PARAMETER of the loop rather than a private detail: the
engine supplies a seeded normal draw and the gate supplies the capture's own
noise. That is the ONLY entry at which this pipeline is comparable to the oracle,
and hiding it would have made the e2e gate impossible rather than inconvenient.

**W2's remainder was REFUSED BY NAME by W6, and is now CLOSED.** What W6 shipped
was a `Synthesize` that resolved the whole request and then named the missing
8.6B forward. The final section below is what closed it.

**One coverage gap, named rather than discovered.** The MULTI-WINDOW arm of
`Music3DenoiseChunks` and `Music3DecodeChunks` — the overlap blend, the carry
span, the post-loop restore and the waveform crop across windows — is not gated
end to end, because the oracle capture is a single 25-frame window and no
multi-window golden exists. Each primitive is gated individually by W4 at reduced
dimensions, and `Music3ChunkPlan` is gated at two and four windows; the
COMPOSITION across windows is not. Closing it needs a longer capture.

Two things are owed and neither is this phase's to close: **no speed number
exists** — every capture so far ran on CPU because `dgx.casa` was down, so
nothing here touches the speed axis — and SGLang-Omni remains `gateable = no`,
read but never executed.

**W7 is complete: ONE arm implemented and value-gated, the rest refused.**
§9 records it in full. **Quantized MiniMax-Music3 checkpoints DO exist** — the
survey found 14 community repositories in 5 formats within days of the release,
with counts per query in §9.1 — and the **RVQ depth decoder's GGUF Q4_K arm now
loads and is gated** against a pinned artifact (`audio-cpp/MiniMax-Music3-GGUF`
@`c36aaeed`, sha256 `4c5d41b2…c70cbdd0`), routed through the shared
`gguf_dequant.h` seam. The other four components, the other two GGUF lineages,
and every non-GGUF format are refused BY NAME and owed (§9.5).

`minimax_music3_quant.{h,cpp}` is a SEPARATE translation unit per
[porting-a-model.md](../porting-a-model.md). It diagnoses every format at the
three places a quantized checkpoint announces itself — the TREE (`.gguf` files),
the MANIFEST (sidecar tensors, and the dtype-only formats no name carries), and
the CONFIG (`quantization_config.quant_method`, and MLX's bare `quantization`) —
each refused with the missing piece, the supported arm, the phase and the issue.
**29 cases / 125 assertions** with no checkpoint, **6 cases / 319 assertions**
against the artifact; RED first at 8 of 8 probe checks; **18 of 18 mutations
fire**.

**The finding worth carrying off this row (§9.6): for a quantized arm, an
upper-bound-only tolerance CANNOT distinguish a real quantized path from a silent
dequant fallback, because the fallback is CLOSER to the golden.** Measured on the
identical forward: the genuine Q4_K arm lands at mean|d| **0.0324**, a bf16
fallback at **0.00182** — 17.8x nearer. Every plausible upper bound passes the
failure. What catches it is a **lower** bound, plus two independent legs that need
no cooperation from the loader's own bookkeeping: the resident ggml type of all
47 tensors, and the Q4_K lattice (0 of 524 288 sub-blocks exceed 16 distinct
values; a bf16 control from the same file gives 127 of 128).

**Three things W7 measured that a later phase would otherwise re-derive.**
(1) "The GGUF arm" is THREE MUTUALLY INCOMPATIBLE LINEAGES and
`general.architecture` cannot separate them — it reads `audiocpp`, `mm3`,
`qwen3` and `wan` for the same model, and `wan` collides with genuine Wan video
GGUFs. (2) The ComfyUI lineage ships **the DiT and condition encoder only**, so it
can never generate audio however complete our arm becomes.
(3) `comfy.gguf.orig_shape.*`, which the H3 GGUF arm depends on, is absent from
all ten files measured.

**No speed number is claimed** and none was measured; the box is CPU-only for
this row. (SUPERSEDED IN PART by §11: a device arm and a two-arm wall clock on
Jetson Thor exist as of 2026-08-16. Still no comparison to a reference.)

---

**W2 IS COMPLETE: the pipeline is WHOLE. What is NOT claimed is a request observed to completion.**
[`minimax_music3_llm.h`](../../include/vllm/model_executor/models/minimax_music3_llm.h)
carries `MiniMaxMusic3SemanticGenerationStep.__call__` (`encoders.py:299-353`)
and `_generate_depth_codes` (`:117-142`), driven through the landed Qwen3 dense
path. `Synthesize` no longer refuses.

**The door the dense path did not have.** Upstream calls
`language_model.model(inputs_embeds=...)` twice and `input_ids` never
(`encoders.py:311`, `:353`), because `_embed_audio_frame` (`:106-115`) is a SUM
of one language-model embedding row and seven depth-decoder rows scaled by
`num_codebooks^-0.5` — a continuous vector corresponding to no vocabulary entry,
which no token id can spell. `Qwen3DenseModel::ForwardEmbeds` is that entry. The
Qwen3 family already had it on its multimodal siblings
([`qwen3_vl.h:145,159`](../../include/vllm/model_executor/models/qwen3_vl.h),
`gemma4.h:210-218`, `muse_glimmer.h:369-380`); only the DENSE registration had
never wired it, and upstream's own `Qwen3Model.forward` has always accepted
either input. `out_hidden` is the second half of the same entry: the
post-final-norm rows, from the forward that produced the logits, because upstream
reads `last_hidden_state[:, -1]` and applies `lm_head` to that very row.

**Additive, and proved rather than argued.** Five registrations —
`Qwen3ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`,
`InternLM2ForCausalLM`, `InternLM3ForCausalLM` — ride that one forward, and one
of them is a ✅ token-exact row, so nothing less than bit-identity was
acceptable. `tests/vllm/models/test_qwen3_forward.cpp` feeds the embedding OF
THE SAME TOKEN IDS through the new entry and asserts it reproduces `Forward` bit
for bit in the logits (500 of 500 identical) AND in the paged KV it wrote — the
second half matters because a forward that agreed on this step's logits while
writing a different cache would decode differently on the NEXT one, which no
logits comparison can see. Three mutations were run against it and all three
fire: an extra scale on the embedded rows (500 differing logits), the hidden
tail ignoring the `logits_indices` gather (throws), and the shape refusal
deleted (a different, less specific exception). A FOURTH mutation — scaling the
rows by 1.0009 — did NOT move, and it is a BAD MUTATION rather than a coverage
hole: bf16's relative spacing is 1/256, so a 9e-4 perturbation rounds back to
the same bf16 value before anything can see it.

**§1's "learned weighted mix over 8 LLM hidden layers" needs one correction, and
it is the reading that would have cost a phase.** `num_condition_layers: 8` does
NOT mean eight of the language model's 36 transformer layers. The eight rows of
a `frame_hiddens` entry are `cat(last_hidden, depth_hidden_1..7)`
(`encoders.py:343`) — ONE language-model hidden state and the SEVEN per-depth-step
states of the RVQ decoder. So the Qwen3 stack needs no per-layer output capture
at all, and `ForwardEmbeds` returning only the post-final-norm rows is
sufficient. §1's phrase is about the condition encoder's own input width, not
about where those rows come from; this paragraph is the disambiguation.

**The gate: teacher-forced, and calibrated against a MATCHED CONTROL.** §5
withdrew the token gate, so the codes are consumed as INPUTS here exactly as
W2/W3 consumes them: `rvq_codes.npy` is fed back frame by frame and what is
COMPARED is the model's hidden state at each step, against
`frame_hiddens[:, :4096]`. That is 25 steps deep through 36 decoder layers over a
61-token KV history, so an error at step 1 is carried by 24 further steps of
attention over the cache it poisoned.

| | bit-identical | mean\|d\| | max\|d\| | outside 2 ULP-or-2^-6 |
|---|---|---|---|---|
| **CONTROL** (upstream's own model, `sdpa_kernel(MATH)`, vs default-backend goldens) | 12 036 / 102 400 (11.75%) | 1.475e-02 | 5.000e-01 | 29 968 (29.27%) |
| **OURS** | 9 337 / 102 400 (9.12%) | 1.763e-02 | 1.000 | 37 572 (36.69%) |
| **NEGATIVE CONTROL** (ours, one-step-shifted alignment) | 201 / 98 304 (0.20%) | 8.025e-01 | 34.44 | 96 537 (98.20%) |

Ours is 1.20x the control's mean absolute error — a near tie, and chasing one
kernel's rounding below the control is not "more correct" (AGENTS.md). Two things
make that a real result rather than a loose bound. **The per-step error does not
grow**: step 0 mean|d| 1.794e-02, step 24 1.645e-02, which is the signature of a
rounding floor and not of a compounding divergence — a wrong KV write would
diverge, not plateau. And **the gate carries its own negative control**: the same
bounds applied to a one-step-shifted alignment (our step k against the golden's
k+1, every value a real hidden state of this very model on this very prompt, and
exactly the off-by-one that `rvq_codes` having one more row than `frame_hiddens`
invites) read 46x outside. A bf16 model this deep is noisy enough that a loose
bound would be indistinguishable from no bound, so the gate demonstrates that its
bound discriminates instead of asserting it.

**The strongest statement is the RANK, and it is scale-free.** The oracle's own
sampled semantic codes rank **2.48 on average** under our reproduced guided
logits — worst 15, 10 of 25 at rank 0 — where chance over the 16384-entry
semantic window is 8191.5 and `_AR_SAMPLING_TOP_K` is 50. Every code the oracle
actually drew is inside our top 50, which is a distributional claim a uniformly
scaled or slightly-wrong hidden state cannot satisfy. It is the same instrument
W2/W3 used on the depth heads (mean rank 8.99 where chance is 511.5), at the
stage above.

**The prompt is not a variable.** The assembled prompt tokenizes to the SAME 61
ids HF `transformers` 5.14.1's `Qwen2Tokenizer` produces on the identical string,
verified both ways before the first hidden state was compared, so a tokenizer
divergence could never present as a model divergence 25 steps later.

**One recorded dtype deviation, mirrored rather than forked.** The shared Qwen3
dense forward emits f32 logits from an f32-accumulating lm_head GEMM; upstream's
`language_model.lm_head` is a bf16 `nn.Linear` whose output is bf16 and is only
then widened by `.float()` (`encoders.py:312`). The shared seam is not forked for
that — five registrations ride it — so the rounding is restored in the Music3 TU,
on the way out, at the one place it is observable. It is a NARROWING, which is
the direction `.agents/porting.md` cares about.

**End to end, over HTTP — OBSERVED TO PASS 2026-08-15 (#852).** The gate is
`tests/parity/test_minimax_music3_e2e_real.cpp`, "an HTTP request generates a
real 44100 Hz stereo WAV": it posts a body at `ApiServer::handle_audio_speech`,
and it asserts the RIFF header, the length the request's duration implies, and
four properties that each rule out a different way of returning a well-formed
non-song — non-zero, unclipped (a scale error would otherwise hide behind the
decode's own clamp), non-constant, and two channels that DIFFER. The request
mapping was hoisted out of `server_main.cpp`'s lambda into
`vllm::openai::SynthesizeSpeechRequest` so the gate calls the code HTTP runs
rather than a test-only copy of it; the server is a one-line client of it now.

MEASURED, this box (20-core x86_64, CPU only; `dgx.casa` down), watched in the
foreground:

```
POST /v1/audio/speech -> 200 audio/wav, 12332 bytes
request shape: 2 AR frames -> 6 latent frames -> 3072 samples/channel (0.06966 s)
waveform:      6144 int16 samples, 6144 non-zero, 0 clipped, range [42, 156],
               RMS 0.00267716, 2818 of 3072 positions differ between L and R
```

The HTTP case alone: **1 case | 21 assertions | 0 failed**, 7:54 wall
(205.63 s user + 20.75 s sys), peak RSS 18.17 GiB, loadavg 47 -> 28. The whole
file with both env vars set: **5 cases | 535 assertions | 0 failed**, 31:14 wall
(1424.17 s user + 38.91 s sys), peak RSS 18.17 GiB, loadavg 29 -> 47. The WAV is
re-read independently of the gate at
`build/music3/minimax_music3_e2e_http.wav`: 44100 Hz, 2 channels, 16-bit,
3072 frames, 0.069660 s, peak 156, 67 distinct left-channel values.

Two honest qualifications. This is **0.07 s of audio, not a song** — it is the
shortest request that still enters every stage, and its length is what makes the
gate affordable on CPU. And its samples are **not compared to anything**: §5
withdrew the token gate because the AR codes are a seeded `torch.multinomial`
draw and the initial latents a seeded `randn_tensor`, so no request's waveform
can equal the capture. What this case proves is that the composition runs, that
every stage hands the next one a shape it accepts, and that the bytes leaving
`/v1/audio/speech` are a well-formed, non-constant, unclipped, genuinely stereo
44100 Hz PCM WAV of the length the request implies. The per-stage NUMERIC gates
are the other four cases in the same file, which compare against the capture and
pass in the same run.

**Why it had never been seen, and it was not the weight load.** The body posted
`"audio_duration_s": 0.1`. That is the name of the FIELD
(`speech_api.h:55`), not the wire key: `ParseSpeechRequest` reads
`audio_duration` (or `duration`), documented at `docs/USAGE.md:1235` and used by
every other test. The key was silently dropped, `audio_duration_s` stayed at its
`0.0` sentinel, and `Music3ResolveRequest` substituted
`kMusic3DefaultDurationSeconds = 60.0`. So the gate asked for a **60-second
song**:

| | asked for | actually ran |
|---|---:|---:|
| duration | 0.1 s | 60 s |
| AR frames | 2 | 1500 |
| denoise windows | 1 | 8 |
| vocoder latents | 6 | 5167 |
| samples/channel | 3072 | 2645504 |

A ~750x job — and one that could never have passed anyway, because the case
asserts the payload length 0.1 s implies. `audio_duration_s` is now REFUSED and
named by `ParseSpeechRequest` (#925), red-first in `test_speech_api.cpp`.

**The recorded diagnosis was wrong, and the way it was wrong is the lesson.**
Earlier runs were placed "inside `LoadQwen3ForCausalLMWeights`" because the four
`language_model/*.safetensors` fds were still open and the depth decoder's file
was not. Neither fact says that. `LoadBf16Direct` **borrows** the mapping when it
can (`BorrowStTensorBytes`), and a borrowed `OwnedTensor` holds its own
`shared_ptr` to it, so those fds stay open for the whole request long after the
loader returned; the depth decoder's tensors are COPIED, so its `SafetensorsFile`
dies and its fd closes on return. The fd pattern is a copy-vs-borrow artifact,
not a program counter.

What actually resolves it is a symbol-resolved profile, and the phases are
unambiguous (`perf record -g` on the live process, 173K samples):

| phase | span | signature |
|---|---|---|
| LM weight load | 0 -> **180 s** | 1 thread, state `D`, ~92 MB/s off the NAS, 3.7 s of user CPU |
| AR generate | 180 -> 19154 s | 20 threads, `LinearNoBias` 42-57%, `Threadpool::Barrier` 25%, `Bt16Avx512` 12-14% |
| acoustic | 19154 s -> killed at 15 h 19 m | `vocoder1d::ConvTranspose1d` 88.5%, `Conv1d` 7.7% |

The load is **3 minutes in both binaries** — the same ~3 minutes
`test_minimax_music3_llm_real` takes — and it is I/O bound, not spinning. The
`__sched_yield` frames in the earlier profile are `Threadpool::PollForWork`'s
bounded hybrid poll (`cpu_threadpool.h`, ggml's `poll = 50`) around a genuinely
compute-bound run, not a stall. Two supporting measurements, both taken rather
than argued: the `lm_head` 200000 x 4096 scalar transpose that was suspected
costs **1.388 s**, and the vocoder at this request's real shape costs **5.4 s**
against 84 s at the capture's L=86.

**Still true, and not this phase's to change.** The CPU run is slow for the
reason recorded below — `LinearNoBias` is a scalar triple loop under
`-ffp-contract=off` by construction, and it is 42-57% of the AR half's profile
while the `vt` threadpool sits at a barrier. Parallelising it over OUTPUT ROWS
would not change its reduction order and so would not move a gated number, but
it is a separate, measured change with its own evidence. There is **still no
speed number** and no oracle comparison: every gate for this row was taken on
CPU because `dgx.casa` was down throughout, and a CPU number against
SGLang-Omni's CUDA-graphed production configuration would be a dishonest
denominator. `PENDING` on the hardware, not on the work. W7's quantized arms are
owed, and W6's multi-window coverage gap is unchanged (the capture is a single
25-frame window).

**Why the CPU run is slow, named rather than left to be rediscovered.** The
autoregressive half's host GEMM (`LinearNoBias`, `minimax_music3_ar.cpp`) is a
scalar triple loop with a DOUBLE accumulator under `-ffp-contract=off` — it does
not vectorize, by construction, because W2/W3 needed a reproducible reduction
order to gate rounding against torch. That is fine for the W2/W3 gate, which
makes 25 calls at sequence length 8. The GENERATION loop makes calls at sequence
lengths 2..8, and the depth decoder streams its whole 2.3 GB of weights per call,
so the short sequences get roughly 4x less arithmetic per streamed byte than the
gate's single seq-8 call does. The acoustic half's `vocoder1d::ConvTranspose1d`
and `Conv1d` are the same shape of code and dominate that half.

MEASURED on this box (`perf record -g`, 173K samples, symbols resolved),
per AR frame of the accidental 60 s run: **12.65 s** of wall per frame, with
`LinearNoBias` 42-57% of samples and `vt::cpu::Threadpool::Barrier` a further
25% — the `vt` threadpool spends a quarter of the AR half waiting, because the
depth decoder and the DiT do not go through `vt` at all and the LM steps it does
run are sequence length 2..8, too small to amortise a barrier.

An earlier revision of this paragraph recorded "~1500 s of single-threaded CPU
for the AR half of one 0.1 s request". That number is WITHDRAWN: it was taken
against the request that had silently become 60 s (#925), so it describes 1500
frames, not 2. The whole 0.1 s request is **7:54 wall / 226 s CPU** including
both weight loads off the NAS.

This is a correctness-first implementation doing exactly what it was written to
do; it is recorded here because the obvious first read of a slow run is "the
language model is slow", and the language model is not the part that is slow —
the LM's own weight load is 180 s of I/O and its forward is 12-14% of the AR
profile.

**One sentence above is now out of date on the device arm, and is corrected here
rather than left to mislead.** *"the depth decoder and the DiT do not go through
`vt` at all"* was true when it was written; it is still true of the DEPTH DECODER
on both arms and of the DiT under `--speech-device 0`. It is NOT true of the DiT
under `--speech-device 1`, which §13 routes through `vt::MatmulBT`,
`vt::LayerNorm`, `vt::AttentionCross`, `vt::RopeFromCache`, `vt::SiluAndMul` and
`vt::Add` with device-resident weights. Every profile number quoted above is the
CPU arm's and still describes it exactly.

**Where the vocoder stands after §18, so this section and `docs/STATUS.md` agree
rather than one of them being newer.** The row does NOT move lifecycle state —
MiniMax-Music3 stays `ACTIVE` — so by AGENTS.md's trigger table neither this line
nor the STATUS row was owed: both are tied to a lifecycle move, and only
`docs/BENCHMARKS.md` is owed by a new accepted measurement. They are written
anyway, and written TOGETHER, because the STATUS row's notes cell carries this
model's measured wins and would otherwise name the depth decoder's and not the
vocoder's. That mismatched pair was a review finding on this row (2026-08-19),
and the resolution recorded here is the pair, not the trigger.

`vocoder.decode_window` was the term §16.7 named as the largest one no row owned.
It is owned now, and measured: **1.36-1.44x** on Thor's 14 threads and **2.16x**
per core (§18.8a, §18.8b), bit-identical output on every length. Next on this
model, in the order §18.9 argues for: the vocoder's PARALLEL DECOMPOSITION, which
§18.8b measured as the cap rather than assuming one; the CUDA-vs-CPU `memcmp`
arm, which needs a box carrying `nvcc`; the e2e pair on the real checkpoint; and
the f64-throughput explanation of the device arm, which §18.2 now carries as an
inference rather than a finding.

---

## 10. The parity sweep, the music-only server, and the weights record (#672)

**Developer directive (2026-08-15):** parity on what upstream supports —
"we want to be a good reference" — usage docs for MiniMax-Music3, and in those
docs the models and weights used and supported, the way MiniMax-H3 already does
it. Then, mid-flight: **"we should allow to load only the music model"** and
**"we need to have an e2e test working"**. The first two lines are the scope;
the last two fixed two of its answers as requirements rather than judgements.

### 10.1 The upstream surface, enumerated

SGLang-Omni `748a0b43` at `sglang_omni/models/minimax_music3/` and the diffusers
PR at `c6da9936` were read field by field. What a user can set upstream, and
where each lands here:

| upstream field | upstream default and anchor | here |
|---|---|---|
| `prompt` / `instructions` (the description) | required, `encoders.py:194-198`; SGLang `request_builders.py:104-106` | `description` (alias `prompt`) — **PARITY** |
| `lyrics` / `input` | required, `encoders.py:199-200`; `request_builders.py:103` | `lyrics` — **PARITY** |
| `audio_duration` | 60.0 s, `encoders.py:251-259` | `audio_duration` (alias `duration`) — **PARITY**, same default |
| `num_inference_steps` | 30, `denoise.py:141-148` | `num_inference_steps` — **PARITY**, same default |
| CFG scale | **not a request field** — frozen at 1.7 into the guider component, `denoise.py:180`; a serve-time knob `dit_cfg_scale` in SGLang, `stages.py:76-95` | `guidance_scale`, a real per-request control defaulting to 1.7. **AHEAD of both arms** |
| `generator` / `seed` | a `torch.Generator` in diffusers (`encoders.py:260`, `denoise.py:111`); an integer defaulting to 0 in SGLang (`payload_types.py:25`) | `seed`, integer, default 0 — **PARITY** with the SGLang spelling |
| `max_new_tokens` (frames) | 9000 cap, `request_builders.py:56-68` | **REFUSED BY NAME**, pointing at `audio_duration` and the /25 conversion |
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | **refused** by upstream, `request_builders.py:14-19,109-114` | **REFUSED BY NAME** — was SILENT, and that silence was the #925 class |
| `voice`, `speed` | refused, `request_builders.py:83-92` | refused — **PARITY** |
| `stream` | refused, `request_builders.py:115-116`; `supports_streaming_vocoder=False` | refused — **PARITY**. Upstream has no streaming in either arm |
| `response_format` | wav/mp3/flac/pcm/aac/opus, `protocol.py:291` | `"wav"` only — **OWED**, no encoder is vendored. Note upstream **downmixes to mono** for any non-wav format (`client/audio.py:328-334`) |
| prompt ceiling 5000 tokens | `encoders.py:42,212-215` | enforced, `minimax_music3_ar.cpp:226` — **PARITY** |
| frame ceiling 9000 | diffusers **CLAMPS** silently (`encoders.py:287`); SGLang **REJECTS** (`request_builders.py:64-67`) | we CLAMP, mirroring the primary oracle. Gated at 360 s and 3600 s |
| output rate | diffusers 44100, no resample; SGLang resamples to 32000 (`acoustic.py:55-58,423`) | 44100 native — the §1.1 decision. The 32 kHz delivery transform stays **OWED** |
| N samples per request | **neither arm supports it** (`denoise.py:117-122` is batch 1; no `n` field on `protocol.py:334-368`) | one waveform per request — **PARITY** |
| N concurrent requests batched | SGLang only: continuous batching at 16, **two engine rows per request** for the CFG twin (`engine_builder.py:74-77`), plus `POST /v1/audio/speech/batch` (`openai_api.py:1277`) | we serialize per engine handle — **OWED** |
| `sgl-omni serve --model <music-model>` and nothing else | the norm: the pipeline is three stages with no chat LLM, `models/minimax_music3/config.py:29-63` | **CLOSED** — see §10.2 |

**Closed by this change:** the music-only server, the missing example, the four
sampling refusals, the `max_new_tokens` refusal.
**Refused by name and recorded as owed:** the non-wav response formats, request
batching and the `/batch` route, the 32 kHz delivery resample, the native `.pth`
arm, streaming (which upstream does not have either, so it is a permanent
refusal rather than a debt).

### 10.2 `--model` is optional when `--speech-model` is given

Serving a 28.5 GB music model also forced loading an unrelated text model,
because `--model <dir>` was unconditionally required. On this box the smallest
available text checkpoint is 35B, so **the recipe this project documented was
effectively unrunnable**, and upstream's own is `sgl-omni serve --model
MiniMaxAI/MiniMax-Music3` with no text tower anywhere.

`--speech-model` alone now loads only the speech engine and registers only
`/v1/audio/speech`. It is the third instance of a shape already in
`server_main.cpp` — a pooling checkpoint serves `/v1/embeddings` alone, a
Parakeet checkpoint serves `/v1/audio/transcriptions` alone — and it mirrors
vLLM's task-conditional registration (`api_server.py:255-265`).

**It is ADDITIVE and that is proved, not argued.** The only case whose verdict
changes is `--model` absent *and* `--speech-model` absent, which was an error
and remains one, with a message that now names both ways to satisfy it.
`--model` alone and `--model` + `--speech-model` take byte-identical paths.

The route table is gated **in both directions over a real socket**, because a
handler-dispatch test cannot see route registration at all: with no synthesizer
`/v1/audio/speech` is a 404 from the route table with no envelope leaked, and on
a speech-only server `/v1/completions` and `/v1/chat/completions` are 404 while
`/v1/audio/speech` returns `audio/wav`.

### 10.3 The e2e gate: what it examined, reported rather than implied

The gate reported `test cases: 5 | 5 passed` and **`assertions: 0`** whenever the
checkpoint was absent. Five green case names over an empty run — the same shape
that fooled this project on `test_qwen3_paged_engine`, which "passes 2/2" while
asserting nothing because its snapshots are dgx-only.

The file is now split. **The checkpoint-free half runs unconditionally in CI**:
the request contract on the exact body the real case posts, the near-miss
refusals, the duration arithmetic including both ceilings, and the speech-only
route table over a real socket with a stub synthesizer. `assertions: 0` is
therefore structurally impossible. **The checkpoint half** keeps its env gate,
and the real case now runs over a real socket against the music-only server
shape rather than calling `handle_audio_speech` directly.

A **coverage-report case** prints, every run, which arms ran and why any did not.
Its assertion deliberately is **not** a cross-case counter: `-tc="…COVERAGE…"`
runs it alone, the counter is legitimately zero, and a gate that reds for the way
it was invoked is a gate somebody deletes. It asserts a cheap fact about the
checkpoint itself instead — 44100 Hz, hop 512, vocab 200000, 8 codebooks, read
from the component `config.json` files in milliseconds — which holds under any
invocation.

**All three arms, measured on this box 2026-08-15**, so the difference between
them is visible rather than asserted:

| arm | cases | assertions | what ran |
|---|---|---|---|
| no env vars | 9 | **37** | the checkpoint-free half only. Was 5 / **0** |
| `VLLM_CPP_MUSIC3_CHECKPOINT` | 9 | **86** | + decode, WAV and condition-mix; `checkpoint_arms_run=3` |
| + `VLLM_CPP_MUSIC3_DIT=1` | 9 | **582** | + the full tail and the music-only server over a real socket; `checkpoint_arms_run=5` |

The full arm's own numbers: `POST /v1/audio/speech -> 200 audio/wav, 12332 bytes
in 518.0 s wall`; 2 AR frames -> 6 latent frames -> 3072 samples per channel
(0.0697 s); 6144 int16 samples, all non-zero, 0 clipped, 2818 of 3072 positions
differing between left and right; and `/v1/completions` and
`/v1/chat/completions` both 404 from the route table, which is the music-only
claim made over the wire against the real 28.5 GB engine rather than a stub.

### 10.4 The weights are documented (porting-a-model.md §2.1)

`docs/USAGE.md` carries the tables the H3 sections already carried, one row per
artifact, with the repo **and revision**: the diffusers arm at
`MiniMaxAI/MiniMax-Music3` @ `fbdf52fbaaca799592917417eb05f1899f1255ec`,
component by component, **28.5 GB resident** (28 517 617 303 B, measured) out of
a 57.4 GB repository and why the two differ; the native `.pth` arm we refuse and
that SGLang-Omni serves; the one implemented GGUF Q4_K artifact with its sha256;
and the fourteen third-party quantized repositories in five formats, each marked
refused and each marked third-party.

The revision is **verified rather than copied**:
`condition_encoder/diffusion_pytorch_model.safetensors` on disk hashes to
`83179c5eaa9a68a370affe0c1b96c2179f659ea4175666b31071490a202c2a4d`, which is that
revision's own LFS record for the file.

### 10.5 The first sample a human can hear, and where it is not

**2.0 s of 44100 Hz stereo, from this engine, in 3286 s of wall clock.** The
e2e gate's own artifact is 0.07 s — the shortest request that still enters every
stage — which nobody can listen to. `minimax-music3-gen` at `--duration 2.0
--steps 2 --seed 7` produced 88 064 frames per channel: RMS 0.03169, peak
0.97437 full-scale with **0 clipped samples**, 175 858 of 176 128 int16 samples
non-zero, and 84 073 of 88 064 positions differing between left and right, so
the 128 latent channels are folded into two streams of 64 rather than
interleaved. Verified independently of the generator, by re-reading the RIFF
file.

x86 20-core CPU, load average swinging 7 to 150 across the run (several other
sessions on the box), 17.8 GB resident. No speed claim is made or implied: the
acoustic half is upstream's own fp32 and the depth decoder and DiT are scalar
host loops by construction (see `## Now`).

**Its samples are compared to nothing, and that is structural rather than an
omission.** §5 withdrew the token gate; §6/W6 records that a request's waveform
can never equal `waveform.npy` because both the codes and the initial latents
are seeded random draws. The clip demonstrates the pipeline runs and emits a
well-formed, non-silent, non-clipped, genuinely stereo signal. The per-stage
gates are what speak to correctness.

**It is NOT committed, and the reason is a checker rather than a preference.**
`scripts/check-pr-size.py` classifies every repository path; `ASSET` accepts
`assets/*.{png,svg}`, `BENCH_EVIDENCE` accepts
`benchmarks/{demo,media}/*.{json,png,gif,mp4,log}`, and neither takes a `.wav`.
The only classified home for one is under `tests/`, where a file compared to
nothing would sit beside the oracle goldens and imply it was one — which
`test_minimax_music3_e2e_real.cpp` explicitly refuses for its own artifact
("under the build tree, never under tests/ — no golden is created, replaced or
implied by this"). Widening either pattern would be widening a checker's scope
to make a change pass, which AGENTS.md forbids without its own spec and
red-before evidence, and this clip does not justify one. Regenerating it is one
command.

### 10.6 A red that belonged to nobody, found by checking a matched arm (#965)

`windows-msvc-cpu` and `windows-msvc-vulkan` failed on this row's pull request.
Both are habitually red and both are habitually attributed to
[#645](https://github.com/mudler/vllm.cpp/issues/645). **They were not #645.**
#645 is the `M_PI` portability regression in three LTX2 sources; this was:

```
server_main.cpp(1315,55): error C2220: the following warning is treated as an error
server_main.cpp(1315,55): warning C4456: declaration of 'loaded' hides previous local declaration
```

— W6's own speech-attach block declaring `loaded` inside the scope of the text
engine's `loaded` at `:1025`. The only warning in the job, and on `main` since
W6 landed.

**What found it was the matched-arm check, not the label.** Three unrelated open
pull requests — #956, #950, #939, none touching the speech surface — fail with
the identical `C4456`. That is what separates "pre-existing" from "mine", and it
is the step that a known-red list invites you to skip. Because `windows-msvc-*`
are PR-only ([#584](https://github.com/mudler/vllm.cpp/issues/584)), `main`
carries no baseline, so the failure presents to every author in turn as a red
their own diff caused — and a second cause sitting behind a known one is
invisible for exactly as long as nobody reads the log.

Fixed in flow: the inner declaration is renamed, with a comment saying why the
name is not `loaded`. No detector weakened, no warning suppressed, no behaviour
changed.

**And behind it, a second one (#968).** With the `C4456` gone the same two jobs
failed again, now on `C4244: conversion from 'const double' to 'float'` raised
inside MSVC's own `<vector>` from `ltx2_video.cpp:203,214` — two narrowing
`positions.assign` calls that `c7cb59fbb` (#964) landed on `main` while this row
was in flight. **Not this row's**, and not fixed here: #964's own comment
reasons that the round trip reproduces the bits, so the narrowing is deliberate
and a silencing cast is a claim about that reasoning. The matched arm splits
exactly on the merge base — #966 and #951 (on `c7cb59fbb`) fail, #967/#956/#950/
#939/#938 (before it) do not.

Two independent causes were stacked behind one habitually-red job name and the
first hid the second, which is the finding worth carrying: **a known-red list
tells you a job is often red, never that today's red is the same one.** Only
reading the log does.

### 10.7 `CleanCaption`'s italic unwrap ([#1083](https://github.com/mudler/vllm.cpp/issues/1083))

The #672 sweep found `CleanCaption` diverging from `_clean_caption`
(`encoders.py:72` @ `c6da9936`) on markdown italics, and this is the change that
closes it.

**A zero-width assertion is not a captured group.** `(^|[^*])\*([^*\n]+)\*($|[^*])`
consumed the character after the closing `*`, so `regex_replace` resumed scanning
*past* it and a span opening within one character of the previous close was never
examined — the surviving asterisks then re-paired **across** the intended spans.
`*a* *b* *c*` came out `a *b c*`, and
`Warm *lo-fi* *jazzy* keys with a *soft* *brushed* snare` came out
`Warm lo-fi *jazzy keys with a soft brushed* snare`: not a leftover marker but a
string upstream would never emit, handed to the tokenizer as the caption. Since
`encoders.py`'s own header states that whitespace-level prompt changes change the
generated audio, that is a contract break rather than cosmetics.

**The two sides are spelled differently on purpose.** std::regex's ECMAScript
grammar has negative *lookahead* but no lookbehind, so the trailing `(?!\*)` is
ported literally and only the leading `(?<!\*)` stays emulated as `(^|[^*])`.
Consuming on the leading side is harmless because that character sits *before*
the span, never between this span and the next one — which is the whole
mechanism, and the reason a single fix could not be applied symmetrically.

Gated in `tests/vllm/models/test_minimax_music3_ar.cpp`: two new prompt goldens
(`adjacent_emphasis`, `unbalanced_emphasis`) produced by *executing* the pinned
`_clean_caption`, plus an eight-row caption case carrying the issue's
counter-examples and the negative side. RED first — 2 cases / 6 assertions
failing on exactly the divergences the issue measured — then 26 cases / 352
assertions green. Three mutations, all RED: consuming the trailing neighbour
again (6), dropping the leading guard (3), and deleting the call site (12).

**Differentially, on 16 012 inputs** (12 realistic markdown descriptions plus
16 000 from a markdown-flavoured alphabet, our built `CleanCaption` against the
pinned `_clean_caption`): **147 mismatches before, 85 after, 62 fixed and 0 newly
broken**, with the after-set a strict subset of the before-set.

**Owed, and deliberately not chased here.** All 85 residual mismatches are one
degenerate class: a caption that is entirely a horizontal rule, where Python's
`re.MULTILINE` `^\s*[-*_]{3,}\s*$` lets `\s` span a newline and collapse several
lines at once, while we apply the rule per line (73 of the 85 match that regex
directly; the other 12 are the same mechanic after tag rewriting). Upstream
returns `''` where we return `'\n'`. Two smaller residues are owed with it and
were already recorded: `std::tolower` is per byte, so non-ASCII uppercase
survives where Python's `.lower()` would fold it, and we split lines on `\n`
only where `splitlines()` also covers `\v`, `\f`, bare `\r`, U+2028 and U+2029.

---

## 11. The device arm (#672) — what a queue bought, and what it did not

**Developer directive (2026-08-16):** give MiniMax-Music3 a device arm so it runs
on the GPU, "use the thor device, with a docker container". That fixes the
hardware and the shape of the evidence; it does not enlarge the scope past what
a queue can reach.

### 11.1 The finding this closes, and the one it does not

`minimax_music3_speech.cpp:492` carried a `vt::Queue` built from a **compile-time
constant CPU device**, under a comment that already named the seam: *"CPU is what
W2 ships and what every gate for this row has been taken on; a device arm is a
queue, not a fork."* That was true and it was also unreachable — nothing could
supply a different queue, so a 28.5 GB music model was a host-only model whatever
hardware the box had.

It is now literally a queue. `multimodal::SpeechModelParams` grew `device`, the
engine builds its queue once in the constructor, and `Music3GenerateFrameHiddens`
receives it. **No model file was forked, and no numeric path was rewritten.**

**What the queue does NOT reach is the majority of the profile, and pretending
otherwise would be the whole failure mode.** §"Now" records it: `LinearNoBias`
42-57 % of the AR half, `vocoder1d::ConvTranspose1d` 88.5 % of the acoustic half.
Those are host `std::vector<float>` scalar loops that take no queue at all.

| stage | device 1 runs it | why |
|---|---|---|
| 8.6B `Qwen3ForCausalLM`, prefill + every decode step + its paged KV | **device** | already on the shared `Qwen3DenseModel::ForwardEmbeds` that five text registrations ride |
| guided logits, top-k draw, frame feedback | host | two 200 000-wide rows per step; not the cost |
| 0.646B RVQ depth decoder | **host** | scalar loop; OWED (§11.4) |
| condition mix + 2.4B fp32 DiT + scheduler | **host** | scalar loops; OWED |
| DAC Flow-VAE vocoder | **host** | **BLOCKED on a missing op, not on effort** (§11.4) |

### 11.2 Three things the change had to get right

**The device selector is a MAPPING, not a cast.** `static_cast<vt::DeviceType>`
is the defect `minimax_h3_video.cpp:230-237` records: it reads the ABI selector
as an enum value, correct only while `kCUDA` stays 1. The tree already carried
two copies of the correct three-question mapping (`minimax_h3_video.cpp:255`,
`ltx2_video.cpp:706`), so a **third** copy was the point at which they start to
disagree. It went on the seam instead, as `multimodal::SpeechEngineDeviceType`,
keyed on the family string because that is this lane's stable registry name.

**Zero is CPU, and that is not the ABI's other spelling.**
`vllm_model_params.device` is 0=auto / 1=cpu / 2=cuda, mirroring vLLM's
`DeviceConfig`. Reusing it here would have made an accelerator build's DEFAULT an
ungated path for every caller that zero-fills the struct — which is every caller
written before this. `VideoModelParams::device`'s 0=cpu / 1=accelerator is the
polarity a generative engine seam already chose, for this reason.

**The paged KV had to move with the forward, and this is the piece that would
have failed silently.** `Music3LmSession` allocated 36 layers of
`std::vector<uint16_t>`. `dense_attn::KvSlice` (`dense_attn_block.h:233`) builds
its tensor view with `d.q.device`, so on a CUDA queue that host pointer becomes a
**host pointer wearing a device tensor's label** — the exact shape of defect
[[keepquant-device-slice-needs-residentweight]] records. The cache is now
allocated on the queue's device and zeroed there, because `KvSlice` hands the
whole `[num_blocks, block_size, Hkv, Dh]` view to the kernel and the unwritten
tail of the last block is real memory.

Two things did NOT need doing, and both were checked rather than assumed:
`ResidentWeight` stores its device copy **on the `OwnedTensor` itself** (`w.d_dev`,
`dense_attn_block.h:190-199`), so the weights upload once and are freed with the
AR scope rather than cached against a host address that can be reused; and
`ForwardEmbeds` already owns its own H2D/D2H for the embeds and the logits.

### 11.3 What the CPU arm is, after

**Bit-identical, and structurally so rather than by measurement luck.** Device 0
takes the same `std::vector` KV allocation, the same host loops and the same
`vt::Queue{kCPU, 0}` the constant used to build. Every existing Music3 gate is
unchanged (§11.5).

### 11.4 What is OWED, named here rather than discovered later

| owed | what it needs | blocked on |
|---|---|---|
| RVQ depth decoder on device | route `LinearNoBias` through `vt::MatmulBT` with device-resident weights, keeping the host loop as the CPU arm so W2/W3's reduction order survives | nothing but the work; 42-57 % of the AR half |
| flow-matching DiT on device | same, plus its attention | nothing but the work |
| DAC Flow-VAE vocoder on device | ~~a `vt` transposed 1-D convolution, which does not exist~~ — **the op now EXISTS with a CPU AND a CUDA provider (§13)**. What is still owed is the DEFAULT: the device arm ships opt-in behind `VLLM_CPP_VOCODER_DEVICE=cuda` | a wiring row that re-gates the four consumers on device, named in §13 |

**The vocoder is the one that is not merely unfinished.** `vt` has no
`ConvTranspose1d` op of any kind — the 1-D convolutions it does carry are
`vt::CausalConv1dFwd` (causal, stateful, SiLU-folded) and `vt::DepthwiseConv1d`
(centre-padded, depthwise), and neither expresses a transposed convolution.
`vt::Conv2d` and `vt::DepthwiseConv1d` are moreover registered for the **CPU
only** (`src/vt/cpu/cpu_conv2d.cpp:111`,
`src/vt/cpu/cpu_conv1d_depthwise.cpp:95`; no CUDA registration exists for
either). So the stage that is 88.5 % of the acoustic half has **no CUDA kernel
behind any op it could route through**, and hand-rolling one outside the shared
seam is what AGENTS.md forbids. Adding `vt::ConvTranspose1d` with a CUDA provider
is its own row: `vocoder1d` is shared with MiniMax-H3 and the LTX-2 audio VAE, so
that op has three consumers rather than one.

**A cheaper, arm-independent win is also owed and is recorded so it is not
re-derived**: `LinearNoBias` parallelised over OUTPUT ROWS is bit-identical —
each output's dot product stays a single sequential double accumulator, so no
reduction order moves — and the same restructure is available for
`vocoder1d::ConvTranspose1d`, whose scatter is indexed by `(out_channel,
position)` and therefore partitions by output channel with its `(ic, t, k)` visit
order intact. Neither is in this change, because a bit-identity claim needs its
own measurement and this change's evidence budget went to the device seam.

**That last paragraph is now DONE — §12.** It landed with its own measurement and
its own gate, and it took `vocoder1d::Conv1d` with it.

**TWO of the three device rows are now DONE.** The vocoder row closed in §13:
`vt::Conv1d` and `vt::ConvTranspose1d` exist with a CPU provider and a CUDA
provider, and every `vocoder1d` consumer routes through them. The DiT row closed
in §14.

**A coverage hole in this row, found by fresh review and filed as #1131.** The
DiT device arm's kernels and staging are gated — eight mutations against them go
RED — but its **production switch is not**. Setting `on_device = false` in
`Music3DenoiseChunks`, or disabling the half-set refusal, leaves every suite
GREEN. A change that silently stopped the DiT reaching the device would be
invisible, and the arm would run on the host with every number still looking
right. The gate that closes it must assert the device path was TAKEN (invocation
count or resident dtype), not merely that outputs agree — the two arms agree
numerically by design.

**Only the depth decoder remains, and its "blocked on" entry above is now known
to be WRONG in one respect**, corrected in §14.5: it is not "nothing but the
work". The shipped depth decoder runs `ArCompute::kBFloat16`, which rounds the
RESULT of every op to bf16 (`minimax_music3_ar.cpp:36-40`), so routing it through
an f32 `vt::MatmulBT` would silently drop that rounding. Mirroring it needs bf16
STORAGE — a dtype decision with its own numeric evidence, not a transcription.

### 11.5 Evidence — Jetson Thor, sm_110, in the container

Host `kairos-4db2` (`192.168.68.23`), Jetson Thor, **sm_110**, aarch64, 14 cores,
~122 GB UNIFIED, driver 595.78. Image `vllmcpp-thor:cuda13.0.1`, nvcc 13.0.88,
`--runtime=nvidia -e NVIDIA_DISABLE_REQUIRE=1`, configured
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
-DVLLM_CPP_SERVER=ON`, no cutlass. Checkpoint mounted read-only from
`/usr/local/nas_share/checkpoints/minimax-music3`. Every run held
`flock $HOME/gpu.lock`, so no two arms ever overlapped, and `uptime` is recorded
on both sides of each.

#### The numeric gate, both arms, SAME BOX AND SAME BINARY

`tests/parity/test_minimax_music3_llm_real.cpp` now takes
`VLLM_CPP_MUSIC3_DEVICE` (default 0 = CPU, so an unset environment reproduces
every number this file ever printed) and resolves it through the SAME
`multimodal::SpeechEngineDeviceType` the engine calls. **This is the device
arm's only numeric gate, and it exists because a generated waveform cannot be
compared to anything** — §5 withdrew the token gate, and both the codes and the
initial latents are seeded draws, so the two arms produce DIFFERENT SONGS by
construction and a sample-wise comparison of them would be meaningless.

25 teacher-forced steps against `frame_hiddens[:, :4096]`, 102 400 values:

| arm | bit-identical | mean\|d\| | outside 2 bf16 ULP | golden-code mean rank | negative control mean\|d\| |
|---|---|---|---|---|---|
| **Thor CPU** (`device 0`) | 9337 (9.118 %) | **1.76348e-02** | 37 572 (36.69 %) | 2.48, worst 15 | 0.802531, 98.20 % outside |
| **Thor CUDA** (`device 1`) | 9324 (9.105 %) | **1.71668e-02** | 36 509 (35.65 %) | 2.44, worst 15 | 0.802583, 98.18 % outside |
| CONTROL (§"Now": upstream's own model under `sdpa_kernel(MATH)`) | 12 036 (11.75 %) | 1.475e-02 | 29 968 (29.27 %) | — | — |

Both arms **4 cases / 220 assertions / 0 failed**, at the bounds that were
already there — no tolerance was widened for the device arm, which is the claim
that matters. The CUDA arm is marginally CLOSER to the golden than the CPU arm
(1.717e-02 vs 1.763e-02) and both sit inside the measured torch-vs-torch control,
which is the near-tie regime AGENTS.md says not to chase below.

**The condition mix, downstream of those rows:** CPU 16 319 of 176 128
bit-identical, mean\|d\| 3.91549e-03, max\|d\| 0.046875; CUDA 16 736,
3.78351e-03, 0.0498047.

**Two things this table proves that a single arm could not.** The negative
control fires identically on both (98.2 % outside, mean\|d\| 0.80), so the bound
still discriminates on the device arm rather than having become slack. And the
**Thor CPU arm reproduces the x86-64 numbers this spec already recorded — 9337
bit-identical, mean\|d\| 1.763e-02, 37 572 outside, rank 2.48 — value for value**,
so the CPU path is unchanged by this row across two architectures, not merely
unchanged on the box that measured it.

#### Wall clock, both arms, same request

`minimax-music3-gen`, identical lyrics/description/seed, `--steps 2`,
alternating arms, one checkpoint resident at a time:

| request | AR frames | delivered | `--device 0` (CPU) | `--device 1` (CUDA) | ratio |
|---|---|---|---|---|---|
| `--duration 0.1` | 2 | 0.070 s / 3072 samples per channel | **835.1 s** | **846.6 s** | **1.014x — the device arm is SLOWER** |
| `--duration 0.4` | 10 | 0.395 s / 17 408 samples per channel | **1512.1 s** | **1430.4 s** | **0.946x — 5.4 % faster** |

Load averages, before -> after: 3.96 -> 6.78 and 4.29 -> 4.87 for the first pair,
3.31 -> 4.72 and 4.00 -> 5.29 for the second; the box was otherwise idle
(`vmstat` 99 % idle before the series) and no other container ran.

**The honest reading, and it is not "the GPU made it faster".** The DIFFERENCE
between the arms is what isolates the language model, because every other stage
is the same host code on both and cancels:

    D(2 frames)  = +11.5 s      D(10 frames) = -81.7 s
    slope     = -11.65 s per AR frame        intercept = +34.8 s

So the device arm **saves ~11.7 s of wall clock per autoregressive frame and pays
a fixed ~34.8 s**, breaking even at about **three AR frames (~0.12 s of audio)**.
Two points determine a line exactly, so this is an attribution with no residual
and no error bar — it is stated as a fit, not as a bound, and a third duration
would be needed to claim more.

The fixed cost is consistent with the one-time host->device upload of the 8.6B
model (`ResidentWeight`, 17.2 GB) plus context creation, but **that attribution
was NOT measured separately** and is offered as the plausible reading rather than
as a result.

**Neither number is a speed claim against a reference.** There is still no
SGLang-Omni comparison: it is `gateable = no`, it serves the native layout, and
its production configuration is CUDA-graphed and compiled while five of our six
stages are scalar host loops. The denominator in §5 is unchanged and every axis
in `docs/BENCHMARKS.md` stays `PENDING`.

#### Gate counts

Local x86-64 CPU build, this tree: `test_speech_engine` 11/38,
`test_capi` 65/653, `test_minimax_music3_speech` 9/223,
`test_minimax_music3_loader` 21/1413, `test_minimax_music3_ar` 25/338,
`test_minimax_music3_acoustic` 27/265, `test_minimax_music3_quant` 29/125,
`test_minimax_music3_ar_real` 4/894, `test_minimax_music3_acoustic_real` 6/76,
`test_minimax_music3_quant_real` 6/319, `test_minimax_music3_llm_real` 4/220,
`test_minimax_h3` 79/57395, `test_indextts2_family` 7/22,
`test_openai_api_server` 62/733, `test_speech_api` 6/67. All green, **and every
one of them reports a non-zero assertion count** — `assertions: 0` is a skip
wearing a pass.

Thor CUDA build: `test_speech_engine` 11/37 (one fewer assertion than the CPU
build BY DESIGN — the device-1 case takes its GRANTED branch there and its
REFUSED branch on a CPU-only build, and it prints which), `test_minimax_music3_speech`
9/223, `test_minimax_music3_llm_real` 4/220 on both arms.

`test_capi` on the Thor CUDA build is **RED, and it is red on pristine `main`
too**. A matched-arm control was built in the same container from a `git archive`
of `origin/main` `c07526aa1`: identical SIGSEGV at `:487` (the ABI v8 logits
processor case already recorded in `docs/STATUS.md` and in the Thor baseline
[#955](https://github.com/mudler/vllm.cpp/issues/955)) and, once that case is
excluded by name, the identical two `structured_choice` failures at `:910`/`:932`
— main 62 cases / 60 passed / 587 assertions / 2 failed, this tree 64 / 62 / 622
/ 2, the delta being exactly the two cases and 35 assertions this row adds. The
`structured_choice` pair was **not** in any baseline because the SIGSEGV aborts
the run before it, so it was filed as
[#994](https://github.com/mudler/vllm.cpp/issues/994) rather than left behind a
known-red name (§10.6's lesson, applied).

#### One instrument defect, found and fixed inside this change

The first version of the gate's arm banner read
`MESSAGE("... ran on '" << vt::DeviceTypeName(...) << "' (VLLM_CPP_MUSIC3_DEVICE=" << ... << ")")`
and printed **`ran on '1' (VLLM_CPP_MUSIC3_DEVICE=1)` on a CPU-only build with
the variable unset** — both fields collapsed to `1` inside doctest's `MESSAGE`
chain. Had it not been read, the CPU arm's numbers would have been recorded as
the device arm's. Rebuilt as one `std::string` and it prints
`ran on 'cpu' (VLLM_CPP_MUSIC3_DEVICE=unset)`. An instrument reports on the state
it was GIVEN, and a banner nobody reads is not a report.
---

## 12. The CPU arm gets its cores back (#672) — row-wise parallelisation, bit-identical

§11.4 named this as owed and said why it was not in the device arm's change: *"a
bit-identity claim needs its own measurement"*. This is that change and that
measurement. It is **arm-independent** — it makes the CPU path faster on every
box, with or without an accelerator, and it is the only one of §11.4's three
items that helps a user who has no GPU at all.

### 12.1 What moved, and the argument that no number can

Three host-reference kernels now partition their **output elements** across the
one threadpool `vt::cpu` already owns (`src/vt/cpu/cpu_threadpool.h`, the 1:1
ggml port). Nothing else changed: no op was rerouted, no dtype narrowed, no
tolerance touched.

| kernel | profile share | how it partitions |
|---|---|---|
| `vocoder1d::ConvTranspose1d` | **88.5 % of the acoustic half** | by OUTPUT CHANNEL |
| `vocoder1d::Conv1d` | 7.7 % of the acoustic half | by OUTPUT CHANNEL |
| `music3::LinearNoBias` | **42-57 % of the AR half** | by flat (row, out) OUTPUT ELEMENT |

**Why no gated number can move, stated as a property rather than a hope.**
`LinearNoBias` and `Conv1d` were already indexed by their output: each output
element owned one sequential `double` accumulator, and it still does, walked in
the same ascending order. Partitioning those loops cannot reassociate a sum it
does not touch.

`ConvTranspose1d` is the one that needed an argument, because it was a SCATTER:
the old loop ran `ic` outermost and wrote into every destination channel. The
pivot to `(dst_c, ic, t, k)` is safe because **a destination accumulator is only
ever reached from its own group's inputs**, so for any one accumulator the
sequence of additions is unchanged — `ic` ascending, then `t` ascending, then
the single `k` with `t*stride + k == p`. The `value == 0.0` skip is a property of
`(ic, t)` and moves with them.

`LinearNoBias` is deliberately compiled `-ffp-contract=off` so W2/W3 could gate
its reduction order against torch. That pinning is untouched, and so is the
order.

**A size guard, and it is a scheduling decision only.** `ParallelForRows` kicks
the pool for any `nr > 1`, and this row's AR half already spends ~25 % of its
wall clock inside `Threadpool::Barrier`; handing it more sub-microsecond
dispatches would make it slower. Below `host_parallel::kMinParallelWork` (2^16
scalar multiply-accumulates) the body runs inline on the caller — the same body
over the same range.

### 12.2 The gate, and the finding that a first draft of it would have missed

`tests/vllm/models/test_host_parallel.cpp` compares each shipped kernel against
a **VERBATIM copy of its own pre-parallel loop**, carried in the test file, at
five thread counts (1, 2, 3, 7, 13), with **bitwise** equality. The oracle is
the old code rather than the new code at another thread count: comparing the
shipped function to itself would prove determinism, and a consistently
reassociated sum is still consistent.

**THE FINDING: for these kernels a `double` accumulator stored through a `float`
CANNOT SEE a reduction-order change at all, so the obvious version of this gate
is green under the exact defect it exists to catch.** Mutating `LinearNoBias`
into two interleaved accumulators — the textbook reassociation — left every
assertion of the ordinary shapes GREEN, and so did reversing `Conv1d`'s input-
channel walk. The reason is arithmetic, not luck: a reassociated sum of
well-scaled terms differs by ~2^-53 relative while the `float` store rounds at
2^-24, so the narrowing swallows it. (This is the same class as the recorded
`bf16 store absorbs reduction-order defects` finding, one dtype up.)

What restores the teeth is engineering the cancellation the wide accumulator
otherwise hides. Two cases do it, and both were added because a mutation stayed
green:

* `LinearNoBias`: taps 0 and 1 carry `+2^30` and `-2^30`, so the serial order
  cancels them immediately and accumulates the remainder exactly, while any
  split carries `2^30` through the remainder and quantises it.
* `Conv1d`: the bias is `-2^40` and input channel 0 is all ones against a
  `+2^40` tap, so the serial `(ic, k)` walk cancels on its FIRST tap.

**And a second leg, because bit-identity alone is satisfied by never
parallelising at all.** The guard case asserts that above the threshold the body
actually ran on more than one thread — deterministic, not a race that usually
wins, because `ParallelForRows` seeds worker `ith` with chunk `ith` and the grid
is 4x-oversubscribed. Hard-wiring the helper to run inline leaves every
bit-identity assertion green and reds exactly that case.

**Mutations: 8 applied, 7 RED, 1 unmoved and explained.**

| # | mutation | result |
|---|---|---|
| M1 | `LinearNoBias` dot split into two interleaved accumulators | **RED** (5) — only after the cancellation case existed; see above |
| M1b | `LinearNoBias` drops the first term of every dot | **RED** (15) |
| M2 | `ConvTranspose1d` walks its group's `ic` descending | **RED** (5) |
| M3 | `ConvTranspose1d` walks `k` descending | **GREEN, correctly** — those taps land in DIFFERENT accumulators, so the order between them is not a reduction order. Recorded rather than counted, because it says what the gate does not claim |
| M4 | the size guard hard-wired to run inline | **RED** (4) on the thread-distinctness leg only, which is why that leg exists |
| M5 | the guard drops the last row of every range | **RED** (114) |
| M6 | `ConvTranspose1d`'s reused per-thread scratch not cleared between channels | **RED** (24) |
| M7 | `Conv1d` walks `ic` descending | **RED** (5) — again only after its cancellation case |

M4 was **invalid as first written**: neutering the guard by deleting its use of
`work_per_row` tripped `-Werror=unused-parameter`, so the compiler refused it and
the gate never got to speak. A build failure is not a red gate; it was re-run in a
form that keeps the parameter used. That is the same trap §9.4 recorded, hit
again.

Sources restored and verified `sha256`-identical after every mutation.

### 12.3 The CPU path is BIT-IDENTICAL, proved end to end on the real checkpoint

The unit gate above proves each kernel against its own pre-parallel loop. What
proves the *composition* — five stages, three touched kernels, a 28.5 GB
checkpoint and 3072 stereo samples of actual music — is that the two binaries
write **the same file**.

`minimax-music3-gen`, x86-64, 20 cores, `--duration 0.1 --steps 2 --seed 7
--device 0`, identical lyrics and description, checkpoint
`/mnt/nas_share/checkpoints/minimax-music3`:

| binary | output | sha256 |
|---|---|---|
| `origin/main` `d9441ef3` | `base-0.1.wav`, 12 332 bytes | `12452152876072b280a7a2551dd182731a8475decc625758de28c345f194de9d` |
| this branch | `new-0.1.wav`, 12 332 bytes | `12452152876072b280a7a2551dd182731a8475decc625758de28c345f194de9d` |

`cmp` reports no difference. Both runs report `0.070 s, 44100 Hz, 2 channel(s),
3072 samples/channel, RMS 0.00265, peak 0.00474`.

**That is the claim this change owes, and it is the strong form of it.** Not "the
tolerances still pass" and not "the RMS agrees to five digits" — the same bytes.
Every gated Music3 number was taken on this path, so a path that emits identical
bytes cannot have moved one.

### 12.4 Speed — a KERNEL A/B, because the e2e pair was spoiled twice

**The whole vocoder convolution chain runs 10.7x faster on a 20-core box, and it
emits the same bytes.** That number is a KERNEL measurement, said so plainly,
and it is not offered as an end-to-end speedup.

**Why not the e2e pair.** It was attempted first and both attempts are VOID, and
naming which runs were spoiled is what makes the replacement honest. The 27 GB
checkpoint is mmap'd from a CIFS mount, so the FIRST run of a series pays a
fault-in no later run pays: `--duration 0.1` gave `d9441ef3` 369.5 s COLD
against this tree 311.8 s warm, which is a statement about the page cache as
much as about the kernels. The `--duration 0.4` pair (786.2 s against 524.0 s)
was taken while another session's full `ctest` sat on the same 20 cores at a
1-minute load average of **76.6**. A contention-guarded re-run is queued.

**What replaces it, and why the statistic is defensible.** A kernel loop is
short enough to repeat, so the MINIMUM over repetitions is available — and a
minimum is the least-disturbed sample rather than an average of somebody else's
contention. Five interleaved rounds (base, new, base, new, ...), the same driver
source compiled twice against the two `libvllm.a` builds, at the vocoder's REAL
geometry (`decoder_hidden_dim` 1536, ratios `[8,8,4,2]`, `kernel = 2*stride`,
`padding = ceil(stride/2)`, exactly as `minimax_music3_acoustic.cpp:738-744`
builds them) and the depth decoder's real 4096 -> 6144 projection.

| kernel | shape | `d9441ef3` | this branch | speedup |
|---|---|---|---|---|
| `ConvTranspose1d` stage 0 | 1536->768, L=128, stride 8 | 0.3812 s | 0.1935 s | 1.97x |
| `ConvTranspose1d` stage 1 | 768->384, L=1024, stride 8 | 0.7707 s | 0.4023 s | 1.92x |
| `ConvTranspose1d` stage 2 | 384->192, L=8192, stride 4 | 8.4197 s | 0.4239 s | **19.86x** |
| `ConvTranspose1d` stage 3 | 192->96, L=32768, stride 2 | 3.7413 s | 0.2342 s | **15.98x** |
| `Conv1d` k=7 | 1536->1536, L=134 | 1.0334 s | 0.0859 s | **12.03x** |
| `LinearNoBias` | 4096->6144, 16 rows, bf16 | 0.2045 s | 0.0188 s | **10.88x** |
| **the convolution chain** | the five rows above it | **13.36 s** | **1.25 s** | **10.7x** |

`uptime` 3.36 before the series and 12.64 after; the noisy rounds are visibly
higher on BOTH arms, which is what the minimum exists to discard.

**And the bit-identity holds AT THESE SHAPES**, which is a third leg under the
correctness claim and the one taken where the vocoder actually calls. Each
kernel printed an FNV-1a fingerprint of its raw output bytes and all six matched
between the arms in every round: `8117c200e328c320`, `f85b530c211840c8`,
`7ec0b57567ae1d1b`, `aebd8d61c6c7539e`, `9e23c0016f1b1cf3`, `be2376b0ebe5177e`.
§12.2 gates small shapes against the serial loop, §12.3 gates the composition,
and this gates production geometry.

#### The two stages that are only ~2x, recorded because it is a finding

Stages 0 and 1 gain 1.9x on 20 cores while stages 2 and 3 gain 16-20x, and the
parallelism is identical in all four. What differs is which array each version
streams. The old scatter's accumulator is `out_channels * full` doubles — **50 MB**
at stage 2 — written in an order that touches every destination channel per
input; the pivot gives each worker a scratch ONE channel wide (262 KB at stage
2, L2-resident), so stages 2 and 3 collect a locality win on top of the thread
win. Stages 0 and 1 do not: their accumulator was already small (6.4 MB) and
their WEIGHTS are large (75 MB at stage 0) and are now read with a stride of
`out_per_group * kernel` floats instead of contiguously.

**The pivot trades weight locality for accumulator locality.** Named rather than
left implicit: a weight pre-transpose, or blocking the `ic` loop, would recover
stage 0/1's contiguity without touching a reduction order, and it is worth its
own measurement. It is not in this change.

### 12.5 The e2e pair — still PENDING, and said so rather than fudged

The wall-clock pair this change owes is **not reported yet**, because the two
runs that exist are not comparable and pretending otherwise would be worse than
waiting.

| run | wall (`gen_s`) | page cache |
|---|---|---|
| `d9441ef3`, `--duration 0.1` | 369.5 s | **COLD** — first touch of a 27 GB checkpoint over CIFS |
| this branch, `--duration 0.1` | 311.8 s | warm |

The checkpoint is mmap'd from a CIFS mount, so the first run of a series pays a
fault-in that no later run pays. The ratio those two numbers form is therefore
about the page cache as much as about the kernels, and it is recorded here as a
confound rather than as a result. A second series then had another session's
full `ctest` land on the box mid-run (1-minute load average 76.6 on 20 cores),
which voided the `--duration 0.4` pair as well.

The re-measurement is guarded: it waits for two consecutive quiet samples with
no foreign compiler or test binary running before each arm, alternates the arms,
takes two samples of each, and records `uptime` on both sides. Until it lands,
the honest statement is that **the e2e axis is PENDING, the KERNEL axis is
MEASURED at 10.7x on the convolution chain (§12.4), and the correctness axis is
CLOSED (§12.3)**.

**And the e2e axis will not be a large number even when it lands**, which is
worth saying in advance so the result is not read as a disappointment. Five of
six stages are host loops and only three of their kernels moved; the 8.6B
language model's decode is elsewhere, the 2.4B fp32 DiT is untouched, and a
short request is dominated by faulting in 27 GB of weights. The kernel A/B is
the number that isolates what this change did; the e2e pair will be the number
that says how much of a whole request that was, and the two answer different
questions.

What is *not* pending: the parallelism is real and asserted, not hoped for. The
gate's thread-distinctness leg fails if the body runs on one thread, and
`VLLM_CPP_CPU_THREADS` now governs these three kernels.

## 13. The vocoder gets a DEVICE op (#672) — `vt::ConvTranspose1d`, `vt::Conv1d`

§12 gave the convolution chain the box's cores. This gives it a GPU — or rather,
it gives it the first `vt` op that a GPU *could* run, because there was none.

### 13.1 The gap, stated exactly

`vt` had **no transposed 1-D convolution of any kind, on any device.** The two
1-D convolutions it carried are `vt::CausalConv1dFwd` (causal, stateful,
SiLU-folded — the Mamba/GDN conv) and `vt::DepthwiseConv1d` (centre-padded,
depthwise — the conformer conv), and neither can express a scatter that GROWS
the time axis. `vt::Conv2d` and `vt::DepthwiseConv1d` are moreover registered
for the CPU only. So the stage that is **88.5 % of the acoustic half's profile**
had nothing to route to, and hand-rolling a kernel outside the shared seam is
what `AGENTS.md` forbids.

This adds `vt::Conv1d` and `vt::ConvTranspose1d` — torch's general grouped
`nn.Conv1d` and `nn.ConvTranspose1d` — with a CPU provider
(`src/vt/cpu/cpu_conv1d_general.cpp`) and a CUDA provider
(`src/vt/cuda/cuda_conv1d_general.cu`), and routes `vllm::vocoder1d` through
them.

### 13.2 Why these are new ids and not modes of `vt::DepthwiseConv1d`

Two reasons, and the second is the one that matters.

The first is expressiveness: a transposed convolution is not a parameterisation
of a forward one.

The second is that **the accumulator width is part of the contract, not an
implementation detail.** `vt::DepthwiseConv1d` accumulates in **f32** and its
byte-exactness gate pins that. These two accumulated in **f64** when this row
landed, because f64 is what the `vocoder1d` host loops used. Widening the
depthwise op would move the conformer encoders; narrowing these re-gated four
audio models. So they are SIBLINGS, and `vt::DepthwiseConv1d` is untouched —
the same call that op itself made against `vt::CausalConv1dFwd`.

**CORRECTED by `VT-CONV1D-F32-ACC`
([#1474](https://github.com/mudler/vllm.cpp/issues/1474),
[`vt-conv1d-f32-accumulator.md`](vt-conv1d-f32-accumulator.md)): these two
accumulate in f32 now, and the sentence this paragraph used to carry was
false.** It said f64 was "what every committed golden for all four consumers was
taken with". It was not. All three generators run torch in f32 —
`scripts/gen-bigvgan-goldens.py:48` builds f64 and then calls `.float()`,
`gen-ltx2-vae-goldens.py:223,234` and `gen-minimax-music3-acoustic-goldens.py:81,134`
cast every parameter and input with `astype(np.float32)` — so the goldens were
the output of an **f32-accumulating** reference and this op was wider than the
oracle its own goldens came from. torch accumulates a float convolution in f32,
measured on a probe that separates the two widths, and vLLM owns neither op at
the parity pin. Narrowing moved the port toward its goldens: over 194 arms, 182
unchanged, 10 improved, 2 one unit-in-the-last-place worse and three or more
decimal orders inside their bounds. The width remains part of the contract; what
changed is which width the contract names.

### 13.3 The CPU path did not move, and it is PROVED

The two CPU kernels are the `vocoder1d` host loops as they stood at `8fa405bb7`,
carried into the op statement for statement — same f64 accumulator, same visit
order, same bias seeding, same `value == 0.0` skip, same output-channel
partition over the same threadpool, and the tensors are VIEWS over the caller's
own `std::vector` rather than copies.

The instrument is the one §12.2 built: `tests/vllm/models/test_host_parallel.cpp`
compares the shipped function against a VERBATIM copy of the pre-change loop at
five thread counts, bitwise. It stays green through the move, which is the whole
claim. This change adds the transposed op's **missing cancellation case** — §12
had one for `LinearNoBias` and one for `Conv1d` but none for `ConvTranspose1d`,
and the gather transcription in the CUDA provider is precisely a rearrangement
of that op's input-channel sweep.

### 13.4 The CUDA provider is BYTE-IDENTICAL, not "within tolerance"

This is the result worth reading twice, because the row was scoped expecting a
tolerance and to have to justify it against a measured control.

Both providers are one f64 accumulator per output element. For `Conv1d` that is
trivial: the host loop is already a gather, so (ic ascending, k ascending) with
the bias seeded first transcribes directly.

For `ConvTranspose1d` it is the whole design. The host loop is a SCATTER: for
each input channel `ic` ascending, each input position `t` ascending, it adds
`x[ic,t] * w[ic,oc,k]` into destination cell `t*stride + k*dilation`. Fix a
destination cell `p` and ask which additions land in it and in what order — `ic`
ascending, then `t` ascending, and for each `t` at most ONE tap `k`, the one with
`t*stride + k*dilation == p`. A thread that owns `p` and sweeps `ic` then `t`
performs the **identical sequence of f64 additions into the identical
accumulator**.

Two details are load-bearing rather than cosmetic. The `value == 0.0` skip is
reproduced exactly, because dropping it changes the SIGN of a zero output cell:
`(-0.0) + (+0.0) == +0.0` while `-0.0` left alone stays `-0.0`. And the bias is
added LAST for the transposed op and FIRST for the forward one, matching each
host loop respectively.

That leaves exactly one way the arms could still disagree: FMA contraction. Both
sides are pinned. The host has been pinned project-wide since `CMakeLists.txt`
gained `-ffp-contract=off` (:40-56) for exactly this class of bug; the device
kernel pins itself, locally and visibly, with `__dmul_rn` / `__dadd_rn`, because
nvcc's flags are separate and its `-fmad` default is on.

So every arithmetic operation on both arms is an IEEE-754 double multiply or add
with round-to-nearest-even, on the same values in the same order. **The gate
asserts `memcmp` equality and no tolerance is claimed, because none is needed.**

**Measured on Jetson Thor, sm_110** (`kairos-4db2`, aarch64, driver 595.78, in
`vllmcpp-thor:cuda13.0.1`, nvcc 13.0.88, built
`-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`):
`test_ops_conv1d_general` **8 cases / 385 assertions, 0 failed**. The same binary
source on the x86-64 CPU box reports **8 / 347** — the 38-assertion difference IS
the CUDA-vs-CPU `memcmp` arm, which is how the run proves it executed rather than
skipped. Both `[SKIP]` lines are absent from the Thor output.

**And the stronger leg, which was not planned and is the one that answers the
default question.** The consumer gates were run TWICE on that box, once on each
arm, and they are identical:

| suite | `VLLM_CPP_VOCODER_DEVICE=cpu` | `=cuda` |
|---|---|---|
| `test_host_parallel` | 8 / 877 | **8 / 877** |
| `test_vocoder1d` | 10 / 58 | **10 / 58** |
| `test_bigvgan` | 6 / 65 | **6 / 65** |

`test_host_parallel` is not an ordinary suite to pass on a device arm. Its oracle
is a VERBATIM in-test copy of the pre-op host loop and its comparison is
bitwise, so a green there with the device selected says the CUDA kernel is
byte-identical to the pre-change scalar host loop **end to end through the
consumers' own entry point**, at every shape it carries including the engineered
catastrophic-cancellation cases — not merely at the op boundary.

Two things it does NOT say, stated so the leg is not over-read. Its thread-count
sweep is redundant on the device arm (the host pool is not used there), so that
axis tests one thing five times rather than five things. And it is these three
suites, not the four consumers' full golden sets — `test_minimax_h3` (79 /
57,395) and `test_ltx2_vae` (42 / 3,120) were run on the CPU arm only, and
re-gating them with the device selected is exactly the work the default flip is
waiting on (§13.6).

The measurements above were taken on `b25a7ebf6`. `git diff` against the landed
HEAD is **empty** for `cpu_conv1d_general.cpp`, `cuda_conv1d_general.cu`,
`ops.cpp`, `include/vt/ops.h` and both gate files — the only later change was how
`ResolveConvDevice` turns a device NAME into an enum, which is not in the numeric
path. A same-SHA re-run is queued behind another session's `~/gpu.lock` holder.

### 13.5 The gate had to earn its teeth, twice

**An f64 accumulator stored through an f32 cannot see a reduction-order change.**
That is measured, not supposed — §12.2 recorded that mutating the dot product
into two interleaved accumulators left every ordinary-data assertion GREEN. So
every equality claim here is also exercised on engineered catastrophic
cancellation: input channels 0 and 1 carry `+2^40` and `-2^40` through a shared
weight row, so the sequential order cancels them immediately and keeps the small
remainder exactly, while any other order carries `2^40` through it and quantises
at ~1.2e-4.

And the cancellation case asserts its OWN teeth rather than assuming them:
reversing the input-channel sweep must change the answer, and the check reports
how many cells move. **A weaker mutation was tried first and correctly read 0** —
swapping WHICH channel carries the positive tap leaves the partial sums at the
same magnitude at the same step, so it is not an order change at all. That is
recorded in the test file so it is not re-derived, and it is the reason the
teeth-check is there: without it, the whole cancellation apparatus could have
been vacuous and still green.

### 13.6 What is REACHED, and what is staged

`vocoder1d` is the shared 1-D BigVGAN core, so routing it routes everything that
decodes through it. Verified by call-site survey, not by assumption:

| file | `Conv1d` call sites | `ConvTranspose1d` | reaches the op |
|---|---|---|---|
| `minimax_music3_acoustic.cpp` | :154, :710, :720, :785, :794, :812 | :741 | yes |
| `minimax_h3_audio_vae.cpp` | :111, :133, :181, :196, :232, :305, :671 | :146 | yes |
| `ltx2_audio_vae.cpp` | :582, :643, :702, :794 | :654, :838 | yes |
| `bigvgan.cpp` | :25 | :60 | yes |
| `minimax_music3_ar.cpp` | :502 | — | yes |
| `indextts2_pipeline.cpp` | :164 | — | yes |
| `bigvgan_loader.cpp` | — | — | n/a, load-time weight-norm folding only |
| `minimax_music3_loader.cpp` | — | — | n/a, load-time only |

A whole-`src/` sweep found no other caller. **One gap is named rather than left
to be discovered**: `ltx2_audio_vae.cpp:75` carries its OWN 2-D host convolution
loop that goes through no op at all. It is out of scope here (this row adds 1-D
ops) and is filed with the rest of the unrouted convolution surface as #1114.

**What is NOT reached is the DEFAULT.** The device arm ships opt-in behind
`VLLM_CPP_VOCODER_DEVICE=cuda`; `cpu` remains the default, so every consumer
above is byte-for-byte where it was. Turning it on by default would move the
numerics of four shipped audio models at once, and that is not a default the row
that ADDED the arm is entitled to set — it needs its own re-gate against each
consumer's goldens with the device arm selected. Per `.agents/reachability.md`
"Landing a slice that is not reached yet", the three things it asks for are named
here: what is not reached is the default resolution in
`vocoder1d.cpp ResolveConvDevice()`; the row that owns the wiring is
`MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation`; the
issue is #672.

**And one more cost is owed rather than hidden.** The device arm allocates,
uploads, downloads and frees PER CALL, and creates a queue per call with it.
That is deliberately literal for a first landing — `cuda` means cuda, with no
size threshold quietly sending small shapes back to the host, because a
threshold would make the consumer gates report on a state they were not given,
which is the exact failure this project keeps re-learning. Three things are left
on the table by it, and all three are ordinary work rather than open questions:
**device-resident weights** (they are loop-invariant and re-uploaded on every
call), **one persistent queue**, and a chain that **stays on the device between
stages** instead of round-tripping through the host at every one of the thirty
convolutions.

### 13.7 The `Conv2d` / `DepthwiseConv1d` device arms — ASSESSED, and DECLINED here

The obvious follow-on was to extend the same machinery to the two existing
CPU-only conv ops in the same change. The survey says do not, and the reason is
not kernel difficulty: **a CUDA provider for those two would be dead on
arrival.** Of the seven models named as stuck behind them, exactly ONE
(`parakeet_encoder.cpp` :166, :194) calls either op; the other six run their own
host loops and would gain nothing until routed, and three of those are 3-D
convolutions the 2-D op cannot express. No caller passes device tensors either —
the single production entry point hands the encoder the CPU backend
(`parakeet_transcription.cpp:100`) and its forward is host-marshalled by design.
The full survey, including why the dtype matrix (27 gated combinations) and the
f32 accumulator make the kernel bodies non-shared, is **#1114**.



### 13.9 One checker had to change, and it got STRONGER

`tests/scripts/test_vocoder1d_single_home.py` went red on this change with
`Conv1d has 2 definitions; exactly one is allowed`. It was a real report of a
real fact and it deserved reading rather than silencing.

That file guards a failure no numeric test can see: a FORK of the vocoder core,
which "passes every tensor comparison on both sides on the day it is made, and
only drifts later". Its instrument is a line-anchored TEXT match for
`^(std::vector<float>|void|double)\s+Name\s*\(` over every tracked `.cpp`. A text
match cannot see a namespace, so it read `vt::Conv1d`'s definition in
`src/vt/ops.cpp` as a second copy of `vllm::vocoder1d::Conv1d`.

It is the opposite of a second copy. It is the op the core now DELEGATES to —
the thing that removed the duplicated arithmetic. But "the checker is wrong here"
is a claim that has to be paid for, because excluding a tree from a guard is
exactly how guards die.

**So the exclusion was priced.** `src/vt/` is skipped from the count — it is the
kernel seam and not a candidate home for this core — and two assertions were
ADDED alongside it:

1. **The core must still call `vt::Conv1d` and `vt::ConvTranspose1d`.** Without
   this, the count would read a perfectly happy `1` while `vocoder1d.cpp`
   quietly re-grew its own loops and all six consumers left the shared seam —
   and no numeric gate anywhere would notice, because a re-grown loop computes
   the same thing. That is the same class of failure the file was written for,
   and nothing else in the tree asserted it.
2. **The walk must report how many files it examined.** A scan narrowed to
   `vocoder1d.cpp` alone would report every count as exactly `1` and pass while
   seeing none of the tree — a green no count-of-1 assertion can detect.

**Evidence, mutated rather than argued** (scratch copy, restored byte-for-byte):

| | result |
|---|---|
| RED-BEFORE: unrepaired checker, this tree | `AssertionError: 2 != 1 : Conv1d has 2 definitions` — the exact CI failure |
| GREEN-AFTER: repaired checker, this tree | 6 tests, OK |
| M1: delegation removed from `vocoder1d.cpp` | **FAILS** — "no longer calls `vt::Conv1d(`" (the new assertion; the count still read 1) |
| M2: a genuine fork added to `bigvgan.cpp` | **FAILS** — "ConvTranspose1d has 2 definitions" (the original invariant survives) |
| M3: the file walk narrowed to one file | **FAILS** — "only 1 .cpp files scanned; the walk is broken" (the other new assertion) |

The first mutation attempt was itself broken and is recorded so it is not
repeated: the scratch copy had no `.git`, `git ls-files` failed, and the suite
reported 2 ERRORS and 5 tests instead of a failure — an infra fault presenting as
a code verdict. The control run above (unmutated scratch copy) exists because of
it.

### 13.10 Speed — VOID, and the reason is a lease I did not take

**Every timing below was taken OUTSIDE the fleet lease, and that invalidates all
of it.** It is recorded rather than deleted because the failure is more
instructive than the numbers were.

The GPU fleet is scheduled by `rc` (`rc devices` / `rc run` / `rc hold`). These
runs went in by `ssh` + `docker run` directly on the box, serialised by
`flock ~/gpu.lock` — the OLD mutex. The concurrent MiniMax-Music3 DiT session was
holding the same box through `rc` at the same time. So the two sessions took
**different mutexes and neither excluded the other**, which is verbatim the
failure `.agents/environment.md` already records for a `GPU_LOCK` naming the
wrong path: "`flock` succeeds on it, so the run is unserialised and only looks
like someone else misbehaving. That cost a whole Marlin series (#777)."

That is almost certainly the 3x swing below. It is not a hypothesis about the
kernel; it is a known defect in how the samples were taken.

**Why it was not simply re-run under a lease.** `rc run` executes inside the
worker's container, and thor's worker has no compiler and no toolchain at all
(`no gcc / g++ / cmake / ninja / nvcc / make`, probed 2026-08-17). Its
`/workspace` is the shared NAS over CIFS; the build tree used here lives in the
box's `$HOME`, which the worker does not mount. So the binary cannot be built
through the lease, and it cannot be reached from inside it. **What a valid
re-measurement needs is named rather than left vague: either a worker image
carrying the CUDA devel toolchain, or this build placed on `/workspace` by
something that already has one.** Until then the speed axis has no instrument,
and that is an OPEN GAP.

**The numbers, retained as VOID.** Two things are true and they must not be
collapsed into one sentence.

**The device arm did not beat the host arm at any size measured.** Jetson Thor,
sm_110, in the container, on an otherwise idle box (`uptime` 4.54 before, 4.57
after; 0 other users), same binary, `VLLM_CPP_VOCODER_DEVICE` the only variable,
best-of-3 per stage, three interleaved repetitions:

| stage | shape | CPU (14 cores) | CUDA | |
|---|---|---|---|---|
| up0 | 1536->768, L=96, stride 8, K=16 | 0.0596 s | 0.1538 s | 0.39x |
| up1 | 768->384, L=96 | 0.0142 s | 0.0388 s | 0.37x |
| up2 | 384->192, L=96 | 0.0022 s | 0.0056 s | 0.39x |
| up3 | 192->96, L=96 | 0.0004 s | 0.0010 s | 0.40x |
| **chain** | | **0.0765 s** | **0.2000 s** | **0.38x** |

**And the A/B is not accepted, because a second run disagreed with it by 3x on
the SAME arm at the SAME size.** A follow-up sweep, taken minutes later on the
same box and binary, put the CPU chain at frames=96 at **0.2280 s** against the
table's 0.0765 s, while CUDA read 0.2000 s in BOTH runs:

| frames | CPU | CUDA |
|---|---|---|
| 96 | 0.2280 s (0.0765 s in the run above) | 0.2000 s |
| 384 | 0.7950 s | 0.7757 s |
| 1536 | 2.5908 s | 3.0677 s |

The device arm is stable to four digits across runs; the HOST arm moved 3x for
an identical workload. So the instrument that is not trustworthy here is the CPU
side, and no ratio from either run is accepted. What survives is the weaker, and
therefore defensible, claim: **at no measured size did the device arm win**, and
at the largest and most compute-dominated point it was 1.18x slower.

**A hypothesis, labelled as one.** The per-stage ratios in the first run are
flat — 0.37x to 0.40x across a 150x span of work — which is the signature of a
COMPUTE-RATE difference rather than of per-call staging overhead, since fixed
overhead would punish the smallest stage far more than the largest. The obvious
candidate is the f64 accumulator: consumer/Jetson Blackwell runs fp64 at a small
fraction of its fp32 rate, and f64 is not optional here — it is what makes the
arms byte-identical and what four models' goldens were taken with. That is a
hypothesis and not a measurement: `nsys` in this image is 2024.2.3 and cannot
trace CUDA on this box, so nothing here has read a counter.

**This is an open gap, not a ceiling.** The next traceable steps, in order:

0. **Take the lease.** Nothing above is admissible until the arms are measured
   under `rc`, which needs a worker image with a toolchain or a build on
   `/workspace`. This is step zero, not a caveat.
1. **Get an instrument.** A newer `nsys`, or `ncu`, on Thor. Everything below is
   a guess until a counter is read; the flat-ratio argument above is inference
   from wall clock — and from wall clock that was contended.
2. **Remove the staging** (§13.6's owed list) — device-resident weights, one
   persistent queue, a chain that stays on the device. The flat ratio argues
   this is NOT the dominant term, which is exactly why it should be measured
   rather than assumed.
3. **An f32-accumulate device variant.** If the fp64 hypothesis holds, this is
   the lever, and it is expensive in the right way: it is NOT byte-identical, so
   it needs its own gate against each of the four consumers' goldens, and it
   cannot inherit this row's `memcmp`.
4. **A GPU whose fp64 is not 1/64.** Thor's fp64 rate may not be representative.
   **`dgx:gpu0` is UP** — a GB10 with unified memory, visible and schedulable in
   `rc devices`; the "dgx.casa is down" note this row was briefed with was stale.
   `orin:gpu0` (AGX Orin) is also free but reports no GPU labels. With three
   materially different boxes on the fleet, no number is meaningful without the
   device it ran on.

**What this does NOT change.** The correctness result stands on its own and is
what this row turns on: the op exists, both providers exist, the four consumers
route through them, and the arms are byte-identical — confirmed at these very
shapes by the per-stage checksums, which matched to every digit printed across
all six runs (`8250.57898`, `-633.539342`, `903.742105`, `657.314583`). The
device arm shipping OFF by default was already the right call for numerics
reasons (§13.6); this measurement says it would also have been the right call for
speed.

---

## 14. The 2.4B fp32 DiT reaches the device (#672) — §11.4's second owed row

§11.4 recorded three device rows as owed. §12 closed the arm-independent one.
This closes the **DiT**, which is the one that mattered most, and it says up
front which of the other two it does not close and why.

### 14.1 Why this row and not another

The DiT is not one stage among six; at a real duration it is the request.

A 45 s clip at the shipped defaults (`num_inference_steps` 30) runs `DitForward`
**660 times** — 30 steps x 2 CFG branches x 11 windows — and each call is 36
blocks over `length + 1` tokens at inner dim 2048, ff 8192. That is on the order
of **634 TFLOP in the DiT against ~29 TFLOP for the entire autoregressive half**:
the DiT is roughly **20x everything else in the model put together**. On the
scalar host loops it is measured in hours; one run was killed at 8 h 11 m having
averaged 4.3 of 20 cores.

That is also why §11.5's device arm reached only 0.946x. It moved the 8.6B
language model, which is real work, and left the stage that is twenty times
larger on the host. A device arm that does not include the DiT is a device arm
for the minority of the profile.

### 14.2 What moved, onto which shared op, and what did NOT

**No new kernel.** Every op below already existed with a CUDA provider; this row
adds a forward that composes them, not a kernel that competes with them.

| reference helper (`minimax_music3_acoustic.cpp`) | shared op |
|---|---|
| `Linear` | `vt::MatmulBT` (+ `vt::Add` for the rank-1 bias) |
| `LayerNorm` | `vt::LayerNorm` |
| `ApplyPartialRotary` | `vt::RopeFromCache` over a `[seq, rotary_dim]` cache |
| `Attention` (NON-causal) | `vt::AttentionCross`, `bias = nullptr` |
| `value * silu(gate)` | `vt::SiluAndMul`, over a stage-time half swap |
| residual adds | `vt::Add` |
| `PointwiseConv` (both 1x1 convolutions) | `vt::MatmulBT` on the transposed activation |

The stage table, after:

| stage | `--speech-device 1` runs it |
|---|---|
| 8.6B `Qwen3ForCausalLM`, prefill + decode + paged KV | **device** (§11) |
| guided logits, top-k draw, frame feedback | host |
| 0.646B RVQ depth decoder | **host** — OWED, and §14.5 corrects why |
| condition mix (once per WINDOW, not per step) | **host** — OWED |
| **2.4B fp32 DiT, every step, both CFG branches** | **device — THIS ROW** |
| scheduler, CFG mix, Euler step, overlap blend, carry | host (elementwise on `[128, length]`; not the cost) |
| DAC Flow-VAE vocoder | **host — BLOCKED on a missing op** (§11.4) |

### 14.3 Four things this had to get right

**The 1x1 convolutions are GEMMs, and that is what unblocked the row.** `vt` has
no CUDA 1-D convolution provider at all — the finding §11.4 recorded against the
vocoder applies here too, because the DiT's `preprocess_conv` and
`postprocess_conv` are `nn.Conv1d(kernel=1)`. But a kernel-1 convolution over
`[C, L]` is a GEMM once the activation is transposed:

    conv(x)[co][t] = SUM_ci W[co][ci] * x[ci][t]
    transposed:     conv(x)^T[t][co] = SUM_ci x^T[t][ci] * W[co][ci] = MatmulBT(x^T, W)

So the forward works FRAME-MAJOR `[length, channels]` throughout and transposes
once on the host at each end, where the tensors are `[128, length]`. No
convolution op is needed, nothing is hand-rolled outside the seam, and the
vocoder's blocker does not transfer.

**The half swap is an identity applied exactly once.** Upstream computes
`ff_out(gate_states * silu(gate))` where `gate_states, gate = ff_in(x).chunk(2,
-1)` — the FIRST half is the value, the SECOND is what SiLU runs on
(`transformer_minimax_music3.py:142-143`). `vt::SiluAndMul` computes
`silu(x[:, :D]) * x[:, D:]`: the opposite assignment. Exchanging the two ROW
BLOCKS of the projection and the two halves of its bias — **once, at stage
time** — makes the shared op compute upstream's expression exactly, with no
per-step permutation. 660 forwards x 36 layers would otherwise permute a
`[seq, 16384]` tensor 23 760 times per clip. The gate for this is a mutation, not
an assertion: §14.4.

**The rotary is the LEADING slice, and `vt::RopeFromCache` already rotates
exactly that.** Music3 ships `rotary_dim` 32 of `head_dim` 64 and rotates only
the leading window, leaving the tail copied through
(`minimax_music3_acoustic.cpp:500-514`). `RopeFromCacheKernel` indexes
`row + pair` and `row + pair + half` within each head and computes
`x*c - y*s, x*s + y*c` — the same rotation over the same slice. `BuildDitRotaryTables`
returns cos/sin already duplicated across both halves of the window, so the cache
this forward builds is the FIRST half of each, packed `cos | sin`.

**The attention is NON-causal and `vt::Attention` is not it.** Upstream
dispatches with no mask (`:97-103`), so every token attends to every token
INCLUDING the prepended timestep one. `vt::Attention` is the causal op; using it
would have silently masked the future and still produced a finite, plausible
tensor. `vt::AttentionCross` with a null bias is the op that means this.

### 14.4 Correctness — same goldens, same bounds, nothing widened

**The CPU arm is bit-identical, and structurally rather than by measurement.**
`minimax_music3_acoustic.cpp`, `minimax_music3_ar.cpp`, `minimax_music3_llm.cpp`
and `vocoder1d.cpp` have a **zero diff** in this change. `--speech-device 0`
takes the same `DitForward`, source byte for source byte, so there is no number
to move. The device forward is an ADDITIONAL entry point in a new file
(`minimax_music3_device.cpp`), which is the shape `minimax_h3_device.cpp` and
`ltx2_device.cpp` already use.

**Reduced dimensions, against upstream's own goldens, at the EXISTING bound.**
`DitForwardDevice` is checked through the SAME `ExpectClose` at the SAME
`kRelTol` 1e-5 / `kAbsFloor` 1e-6 as `DitForward`, and each case reports BOTH
arms' distance to the golden — because the question is not whether the two arms
agree with each other (a shared-helper comparison proves consistency, not
correctness) but whether the device arm is as close to UPSTREAM as the host arm
already is:

| arm | worst \|arm - upstream\| |
|---|---|
| host `DitForward` (the accepted control) | 1.565e-07 |
| device forward, CPU backend | 1.192e-07 |
| device forward, **CUDA sm_110** | 2.980e-07 |

All three are inside the 1e-6 absolute floor with room to spare, and **no
tolerance was relaxed**. The CPU-backend arm is closer to upstream than the host
loops are; the CUDA arm is about 1.9x the host arm's distance and about a fifth
of the bound.

**Two mutations, because a bound that nothing violates has not been shown to
discriminate.**

* **The half swap.** Pre-swapping the host weights makes the stage-time swap undo
  the test's, so the forward computes `silu(value) * gate` — the wrong network,
  same shapes, same finiteness. **20 of 20 values outside the bound, worst
  \|diff\| 1.538e-03**, four orders above the noise. The pair pins the DIRECTION,
  not just the magnitude: routing it the other way round would fail the right
  case and pass this one.
* **The condition.** Conditional and unconditional forwards must be different
  tensors — a DiT that dropped its conditioning would match both goldens
  identically. 20 of 20 differ, on both backends.

Two more cases guard the staging contract itself: every mis-sized weight is
refused **at stage time** naming the tensor (before 9.7 GB moves at real
dimensions), and `release_host` is asserted to leave the source vectors empty
AND at zero capacity while the staged copy still reproduces the golden — which is
also the check that would catch a released host buffer uploaded without a
synchronize.

**FULL SCALE — the real 2.4B fp32 checkpoint against the oracle capture, on
sm_110.** `tests/parity/test_minimax_music3_acoustic_real.cpp` now takes
`VLLM_CPP_MUSIC3_DEVICE` (default 0 = CPU, so an unset environment reproduces
every number this file ever printed) resolved through the SAME
`multimodal::SpeechEngineDeviceType` the engine calls. Both arms, same box, same
binary, same goldens, **same bounds** — `kDitRelTol` 1e-4 / `kDitAbsFloor` 5e-5 /
`kDitMeanAbsTol` 5e-6, all unchanged. 11 008 values per step:

| arm | step | bit-identical | mean\|d\| | max\|d\| | outside |
|---|---|---|---|---|---|
| **Thor CPU** (`device 0`) | first | 423 (3.843 %) | 1.71434e-06 | 2.38419e-05 | **0** |
| **Thor CPU** | last | 235 (2.135 %) | 2.22396e-06 | 2.83718e-05 | **0** |
| **Thor CUDA** (`device 1`) | first | 473 (4.297 %) | **1.64344e-06** | 2.47955e-05 | **0** |
| **Thor CUDA** | last | 222 (2.017 %) | 2.44677e-06 | **2.59876e-05** | **0** |
| CONTROL (torch vs torch, `set_num_threads(1)`) | first | 15.416 % | 7.526e-07 | 7.153e-06 | — |
| CONTROL | last | 5.596 % | 1.424e-06 | 1.335e-05 | — |

**Three things this table shows that one arm could not.** The device arm sits
ON TOP of the host arm rather than beside it — better on two of the four figures
(more bit-identical and a lower mean at the first step, a lower max at the last)
and marginally worse on the other two, which is what two correct float32
implementations of the same graph look like. Both arms sit at the same
multiple of the recorded torch-vs-torch control (about 1.2-1.7x its mean, 2-3.5x
its max), so the device arm did not move the row's relationship to the control.
And **the Thor CPU arm reproduces the x86-64 numbers this spec already recorded —
3.843 %, 1.714e-06, 2.384e-05; 2.135 %, 2.224e-06, 2.837e-05 — VALUE FOR VALUE**,
so the CPU path is unchanged across two architectures, not merely unchanged on
the box that measured it.

The four ARM=1 cases that print those numbers are 464 assertions against the
CPU arm's 461; the three extra are this row's staging `CHECK` and the two
`REQUIRE`s that refuse a device arm with no staged weights.

### 14.6 Speed — MEASURED, on one named device, per DiT forward

**Device: `thor:gpu0` — NVIDIA Thor, sm_110, aarch64, 14 cores, ~122 GB UNIFIED,
driver 595.78.** Every number below is from that one box. No number here is
compared to one from `dgx:gpu0` (GB10) or `orin:gpu0`, because those are
different machines and a ratio across them would mean nothing.

Image `vllmcpp-thor:cuda13.0.1`, nvcc 13.0.88, configured `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=ON`,
no cutlass. Checkpoint read-only from the NAS. Same binary, same weights, same
committed inputs on both arms; the arms never overlapped.

**What is timed is the DiT and only the DiT.** `VLLM_CPP_MUSIC3_DIT_REPEAT=R`
makes the gate run its guided velocity R times per timestep instead of once, and
the timer brackets that loop — the 9.7 GB checkpoint load and the weight staging
are outside it, and the staging is timed separately.

**Corrected in fresh review:** an earlier revision of this paragraph also claimed
the GOLDEN READS were outside the bracket. They are not — four `LoadF32Npy`
calls, two `Compare` and two `ReportInto` sit INSIDE the `t0`/`loop_s` bracket
(`test_minimax_music3_acoustic_real.cpp:592-606`). That inflates the intercept and
makes the per-forward number SLOWER than the pure forward, so the headline ratio
is conservative rather than inflated — but the sentence was wrong as written, and
a reader checking the intercept against the fit would have been misled.

| arm | repeats | forwards | loop | per forward | staging | box load |
|---|---|---|---|---|---|---|
| CPU (`device 0`) | 1 | 4 | **819.818584 s** | **204.954646 s** | 0 (host, no-op) | 3.42 |
| CPU (`device 0`) | 1 | 4 | **819.992 s** | **204.998 s** | 0 (host, no-op) | 10.37 |
| CUDA (`device 1`) | 1 | 4 | **0.749077 s** | **0.187269 s** | 0.603561 s | 4.79 |
| CUDA (`device 1`) | 3 | 12 | **2.110301 s** | **0.175858 s** | 0.660600 s | 5.32 |
| CUDA (`device 1`) | 1 | 4 | **0.743367 s** | **0.185842 s** | 0.609463 s | 5.1 |
| CUDA (`device 1`) | 1 | 4 | **0.743881 s** | **0.185970 s** | 0.612787 s | 4.44 |

**Per DiT forward at the capture's geometry (latent length 86, seq 87):
204.955 s on the host, 0.1706-0.1873 s on the device — between 1094x and
1201x.** (An earlier revision wrote the low end as 1102x. That is the ratio at
the FASTEST R=1 point, 0.185970 s; the range's slow end is 0.187269 s, and
204.954646 / 0.187269 = 1094x. Caught in fresh review.) The two-point fit over the device arm's 4- and 12-forward runs gives

    slope = 0.170607 s per forward     intercept = 0.063012 s

so the ratio is 1102x taken on the matched R=1 pair and 1201x taken on the
fitted per-forward slope. The device R=1 point was taken THREE times across two
sessions, bracketing the R=3 point, at 0.749077 / 0.743367 / 0.743881 s — a
0.77 % spread.

**The contention asymmetry was checked rather than assumed, and it is nil.** The
first CPU point was taken at box load 10.37 while the device points sat at
4.4-5.3, which would have inflated the ratio if it mattered. It was re-taken on
an idle box (load 3.42) with the fixed instrument: **204.954646 s vs 204.998 s,
agreeing to 0.021 %**. The host DiT forward is single-threaded and this box has
14 cores, so a load of 10 still leaves it a core. Both CPU points are reported
above rather than the convenient one.

**The weights are staged ONCE, and this is the measurement that says so rather
than the code comment.** One staging costs 0.60-0.66 s. The ENTIRE four-forward
loop costs 0.745 s and the twelve-forward loop 2.110 s; twelve stagings would be
7.35 s on their own. The loop's fitted intercept is 0.063 s — a tenth of one
staging. A per-forward upload is arithmetically excluded by the numbers, not
argued away.

Extrapolated to a full clip — and it is an EXTRAPOLATION, labelled as one,
because the only geometry measured is the capture's single 86-frame window — the
660 forwards of a 45 s clip at the shipped defaults are **~37.6 h of DiT on the
host against ~113 s on the device, with the one-time staging 0.54 % of the
latter**.

**The whole-process ratios, which are lower and are the honest ceiling on what a
user sees today.** The same gate binary end to end, including the identical
9.7 GB NAS load on both arms, ran 1054-1071 s (CPU) vs 238-298 s (CUDA) —
**3.5-4.5x**, the spread being NAS cache state rather than compute;
and the earlier full two-arm correctness series, identical scripts throughout,
ran 49 min 17 s vs 15 min 49 s — **3.12x**. The gap between 1100x on the DiT and
4x on the process is the point of §14.5: the load, the host vocoder and the depth
decoder are unchanged, and they now dominate.

**No end-to-end song pair is offered.** At the shipped 30 steps the host arm's
DiT alone is ~37.6 h, so an e2e pair at a realistic setting is not runnable on
the CPU arm; at a setting short enough to run, the DiT is a small enough share
that the pair would measure the vocoder. The per-forward A/B above is the
measurement that isolates what this row changed, and no clip-level speed claim is
made from it.

**No parity claim.** SGLang-Omni is still `gateable = no`; every reference axis
in `docs/BENCHMARKS.md` stays `PENDING`.

### 14.7 One instrument defect, found inside this change

The first revision of the timing line printed **`DIT_TIMING arm=1`** on the CPU
run. `vt::DeviceTypeName` returns `const char*`, and a `const char*` fed to
doctest's `MESSAGE` chain takes the **bool** conversion and prints `1`. The
staging line had it too, printing `(1)` where it meant `(host, no-op)`.

**This is #672's own §11.5 defect reappearing in a new line**, which is the
reason it is recorded here rather than quietly fixed: the lesson from the first
occurrence was written down, and a fresh `<<` chain reintroduced it anyway. Both
lines are now assembled as one `std::string` and printed, which is what the arm
banner beside them already did — and the banner is why the numbers survived,
because it said `ran on 'cpu' (VLLM_CPP_MUSIC3_DEVICE=0)` correctly while the
line below it said `arm=1`.

**Every number in §14.6 was re-taken with the fixed instrument**, and the CPU
arm's pre-fix point is kept beside its post-fix twin rather than replaced by it,
because the pair is what proves the label defect never touched the values:
819.992 s (pre-fix, `arm=1` printed, load 10.37) against 819.818584 s (post-fix,
`arm=cpu` printed, load 3.42). The correct banner sat above both, the process
wall clocks corroborate both, and the correctness numbers both runs printed match
the x86-64 values this spec already recorded value for value.

### 14.5 What is still OWED, and a correction to §11.4

§11.4 said the depth decoder was blocked on "nothing but the work". **That is
wrong, and this row is where it was found out.** The shipped depth decoder and
condition mix run at `ArCompute::kBFloat16`, which rounds the RESULT of every op
to bf16 (`minimax_music3_ar.cpp:36-40`) because that is what torch stores.
Routing them through an f32 `vt::MatmulBT` would silently drop that rounding — a
change to the numbers wearing a refactor's clothes. Mirroring them needs bf16
STORAGE so the shared op rounds where the reference's `Store` rounds, which is a
dtype decision with its own numeric evidence, not a transcription. It is also
worth ~15 TFLOP against the DiT's 634, so it is second in size as well as second
in order.

The condition mix has a further reason to be second: it runs **once per window**,
not once per step, so it is outside the 660-forward loop entirely.

The vocoder row is unchanged: `vt` still has no `ConvTranspose1d` with any
provider, that op has three consumers, and it is its own row.

---

## 15. Where the time ACTUALLY goes (#672) — the developer's question, measured

**The question, verbatim (2026-08-18):** *"I find it weird that takes so much
time for generating 20s of audio"*, and *"can you check the upstream oracle's
implementation vs ours?"*

The honest answer needed three things this row did not have: a per-STAGE split of
our own wall clock, the same split for the pinned oracle, and a structural
comparison that survives the two arms running on different hardware. Every
earlier attempt reconstructed the split from FLOP counts, which needs a guessed
host scalar rate — and the guesses were an order of magnitude apart.

### 15.1 The instrument, and what it may not be quoted as

`src/vllm/model_executor/models/music3_profile.h`, gated on
`VLLM_CPP_MUSIC3_PROFILE` and OFF by default. Buckets are LEAVES (they partition
the request and are summed) or SPANS (they enclose leaves, are printed, and are
never added), plus pure counters; `unattributed` is the glue between the leaves
and is printed rather than spread over the measured stages, so a misplaced
bracket shows up as a number instead of as somebody else's share.

**It takes no GPU clock window** (`.agents/benchmarking.md`). Its rows are a
within-run SPLIT and are not quotable as per-kernel or cross-box figures. The
oracle side used a scratch driver rather than `tools/oracle/music3_oracle.py`,
because that script's job is goldens and its `StageRecorder` clones every step's
latents — which is not what a timing run should be paying for.

### 15.2 OUR split — Jetson Thor, sm_110, `--device 1`, `VLLM_CPP_VOCODER_DEVICE=cuda`

4 s of audio, 4 steps, 100 AR frames, 1 window, **1481.524 s total**. Box load
4.2-17.6 across the run, `VLLM_CPP_CPU_THREADS` unset, 14 cores. Built from
`769fa55b2` (tree `32aa40e2f`, identical to the pushed commit).

| kind | stage | seconds | % | calls |
|---|---|---|---|---|
| leaf | **`load.ar_weights`** | **780.015** | **52.65** | 1 |
| leaf | **`ar.depth_forward`** | **347.276** | **23.44** | 1414 |
| leaf | **`load.acoustic_weights`** | **249.018** | **16.81** | 1 |
| leaf | `vocoder.decode_window` | 53.563 | 3.62 | 1 |
| leaf | `ar.lm_decode_step` | 14.782 | 1.00 | 100 |
| leaf | `ar.lm_prefill` | 11.643 | 0.79 | 1 |
| leaf | `ar.depth_projection` | 10.133 | 0.68 | 1414 |
| leaf | **`denoise.dit_device`** | **5.061** | **0.34** | 4 (= 8 forwards) |
| leaf | `acoustic.dit_staging` | 1.219 | 0.08 | 1 |
| leaf | `ar.depth_head_and_draw` | 0.757 | 0.05 | 707 |
| leaf | `ar.semantic_guide_and_draw` | 0.349 | 0.02 | 101 |
| leaf | `denoise.condition_mix` | 0.257 | 0.02 | 1 |
| span | `ar.depth_stage` | 365.390 | 24.66 | 101 |
| span | `ar.TOTAL_loop` | 392.181 | 26.47 | 1 |
| | `sum(leaf)` | 1474.072 | 99.50 | |
| | `unattributed` | 7.452 | 0.50 | |

**Three facts, and they reorder every assumption this row was operating on.**

**One: the checkpoint LOAD is the single largest cost — 1029.0 s, 69.5 % of the
run.** It is I/O off the CIFS-mounted NAS, it is a FIXED cost that does not scale
with duration, and it is not compute at all. A 4 s clip pays it in full.

**Two: of the autoregressive loop, 88.5 % is `ar.depth_forward`.** The **0.646B
RVQ depth decoder, on the host, costs 23.5x the 8.6B language model on the
GPU** — 347.276 s against 14.782 s. That is the inversion §11.1's table
gestured at ("42-57 % of the AR half") and it is now a number: at a real
duration the depth decoder IS the autoregressive stage.

**Three: the DiT on the device is 0.34 % of the run.** §14 moved the stage that
is 20x everything else in FLOP terms, and it worked so well that the stage is now
a rounding error: 0.633 s per forward at latent length 344. **The GPU is not the
problem, and no further DiT work will move this number.**

> **CORRECTED 2026-08-20 by §20.5, and the correction is the point.** That last
> sentence is FALSE on current main at the shipped step count. It was true of a
> 4 s clip whose wall was 69.5 % CIFS load and 23.4 % a host depth decoder. Stage
> the checkpoint (§15.6), move the depth decoder (#1330) and ask for the 20 s at
> 30 steps the developer actually asks for, and `denoise.dit_device` is
> **62.24 % of the run** at 370.556 s over 120 calls. The bucket did not move —
> §15.7 read 370.746 s over the same 120 calls — everything around it did. A
> share is not a property of a stage.

### 15.2a The AR loop is LINEAR in frames — the O(n^2) suspicion, refuted

The obvious hypothesis for "20 s costs far more than 4x 5 s" is that per-frame
work grows with the sequence. It does not. A second run at **8 s / 4 steps /
200 frames**, same box, same binary, same prompt, TOTAL **1560.679 s**:

| bucket | 100 frames | 200 frames | ratio (2.00x expected) |
|---|---|---|---|
| `ar.TOTAL_loop` | 392.181 | **754.772** | **1.925** |
| `ar.depth_forward` | 347.276 | **684.625** | **1.971** |
| `ar.depth_projection` | 10.133 | 19.844 | 1.958 |
| `ar.lm_decode_step` | 14.782 | 30.507 | 2.064 |
| `vocoder.decode_window` | 53.563 | 105.254 | 1.965 |
| `denoise.dit_device` | 5.061 | 12.436 | 2.457 |

**Per frame the AR loop gets slightly CHEAPER, not dearer: 3.9218 s -> 3.7739 s
(-3.8 %).** `ar.depth_forward` per frame is 3.4728 -> 3.4231 s (-1.4 %), and
`ar.depth_forward` is **14.00 calls per depth stage at both sizes** — the loop
does a fixed amount of work per frame, exactly as upstream does.

The only bucket that grows super-linearly with frames is `ar.lm_decode_step`, at
**+3.2 % per frame** over a 2x sequence — that is the language model's attention
over a growing KV cache, it is upstream's behaviour too, and at 0.15 s per step
it is 2 % of the loop. `denoise.dit_device` grows 2.457x for a 2x latent length
because its attention is O(seq^2); at 0.80 % of the run that is not yet
interesting.

**So nothing re-prefills, nothing accumulates, and duration buys time linearly.**

### 15.2b The load is CACHE STATE, and every figure carries its state in its name

There is no single "load time" for this checkpoint, and averaging the readings
would destroy the only information they carry. Three states are measured, and
each is quoted with its state attached, always:

| figure | `load.ar_weights` | `load.acoustic_weights` |
|---|---|---|
| **cold-CIFS** — first read in the container, off the NAS mount | **780.015 s** | 249.018 s |
| **warm-CIFS** — same mount, host page cache warm from the previous run | **428.452 s** | 257.876 s |
| **local-disk** — checkpoint staged to the worker's own disk | **7.539 s / 5.689 s** | **3.406 s / 2.139 s** |

The reads are byte for byte identical; only the storage state differs. The
cold-CIFS to warm-CIFS delta of 351.5 s is host page cache and needs no further
explanation. `load.acoustic_weights` barely moves between the two because 9.7 GB
of fp32 does not stay cached beside 17.6 GB of bf16 on a 122 GB box that is also
holding the model.

**The warm-CIFS residual is the number that needed its own measurement**, and it
is why §15.6 exists. The reasoning that motivated it was: 28.5 GB in 428.452 s is
~67 MB/s, far too slow for a page-cached read, so a large part of that residual
is probably not I/O at all.

**That reasoning was WRONG, and §15.6 is the measurement that says so.** The
residual is I/O, essentially all of it. It is recorded here rather than quietly
deleted because it was the best available inference from the numbers to hand,
and the only thing that beat it was a run.

### 15.3 RSS — explained, and it is NOT a leak

The developer saw RSS climb 7.3 -> 14.7 GB during one 20 s generation. The
markers say exactly what that is:

| marker | t (s) | RSS (MiB) |
|---|---|---|
| `synthesize.enter` | 0.000 | 224.0 |
| `ar.weights_loaded` | 780.016 | **17584.3** |
| `ar.loop_done` | 1172.198 | 17964.6 |
| `ar.weights_released` | 1172.403 | **673.6** |
| `acoustic.weights_loaded` | 1421.422 | **10246.5** |
| `acoustic.dit_staged` | 1422.640 | **1030.3** |
| `denoise.done` | 1427.961 | 1130.8 |
| `vocoder.done` | 1481.524 | 1131.0 |

Two weight sets, each loaded into host memory and each released at its scope
boundary: the 8.6B bf16 language model plus the 0.646B depth decoder (17.6 GB,
freed when the AR scope exits, exactly as `minimax_music3_speech.cpp` intends and
as upstream does by hand at `encoders.py:302-309`), then the 9.7 GB fp32 DiT,
which drops to 1.0 GB the moment `StageMusic3DitWeights(..., release_host=true)`
hands it to the device. **The observed climb was the FIRST load in progress,
sampled before it finished.** Peak is 17.96 GB, and it is bounded, not growing.

### 15.4 The structural comparison — counts, because they survive different hardware

Measured on BOTH sides. The oracle is diffusers PR #14456 at `c6da9936`, x86
20-core CPU, 6 AR frames / 2 steps / 1 window; ours is the run above. Seven depth
stages ran upstream (`max_frames + 1` iterations), 101 in ours.

| module | upstream calls / depth stage | ours | rows per frame, upstream | ours | verdict |
|---|---|---|---|---|---|
| depth decoder forward | **7** (batch 2, seq 2..8) | **14** (batch 1, x2 CFG rows) | 70 | 70 | **SAME work**, ours unbatched |
| depth `projection` | **8** (batch 2) | **14** | **16** | **70** | **OURS 4.375x** |
| depth `audio_heads` | **7** | 14 (batch 1) | 14 | 14 | SAME |
| language model decode | 1 (batch 2, 1 token) | 1 | 2 | 2 | SAME — incremental KV both sides |
| `lm_head` | 1 (batch 2) | 1 | 2 | 2 | SAME |
| condition mix | once per WINDOW | once per window | — | — | SAME |
| DiT forward | 2 per step per window | 2 per step per window | — | — | SAME |
| vocoder | once per window, uncropped | once per window, uncropped | — | — | SAME |

Anchors — upstream: `encoders.py:118-142` (depth loop), `:125,:127,:141` (the
`sequence` list that caches projected rows), `:131` (the whole-prefix forward),
`:347` (`past_key_values`), `before_denoise.py:28-29,:67-70` (windowing),
`denoise.py:82` (condition mix), `:219-227` (two CFG passes),
`decoders.py:83-87` (vocoder). Ours: `minimax_music3_llm.cpp:423-456` (depth
loop), `:281-347` (paged incremental KV), `minimax_music3_ar.cpp:747-771`
(`DepthSequenceEmbeds`), `minimax_music3_acoustic.h:190-191` +
`minimax_music3_acoustic.cpp:300-309` (windowing),
`minimax_music3_speech.cpp:244` (condition mix), `:280-296` (two CFG passes),
`:361` (vocoder).

**ONE structural difference, and it is ours.** `DepthSequenceEmbeds` projects the
WHOLE depth sequence on every codebook step; upstream projects only the newly
appended element and keeps the earlier ones in its `sequence` list. Per frame
that is 70 rows of a 4096x4096 GEMV against upstream's 16 — 54 redundant rows,
and the recomputed rows are bit-for-bit the rows already computed (same inputs,
same weights, same op), so caching them is numerically INERT.

**And it is NOT the answer to the developer's question, which is why it is priced
rather than announced.** `ar.depth_projection` is 10.133 s of 1481.524 s; the
redundant 54/70 of it is **~7.8 s per 100 frames, 0.53 % of the run**. Filed as
[#1235](https://github.com/mudler/vllm.cpp/issues/1235) rather than fixed inside
a measurement change — it needs a bit-identity gate on the real checkpoint, and
that is its own row's evidence budget, not this one's.

**Everything else matches upstream call for call.** In particular the depth
decoder re-runs its whole prefix every codebook step on BOTH sides — neither
implementation caches depth KV — so our 14 batch-1 forwards do exactly the work
of upstream's 7 batch-2 forwards. We are not doing more work there; the host
scalar loop is simply what that work costs without a GEMM behind it.

### 15.5 The verdict on the developer's question

**Nothing is structurally wrong, and the model is not "simply this expensive"
either.** The time is in two places that are both fixable and neither of which is
the GPU:

1. **the 27 GB checkpoint load** — 69.5 % of a 4 s clip, a fixed cost, I/O off a
   CIFS mount;
2. **the 0.646B RVQ depth decoder as a host scalar loop** — 88.5 % of the
   autoregressive loop, and 23.5x the cost of the 8.6B language model that runs
   on the GPU beside it.

The DiT device arm §14 landed did its job so completely that the DiT is now
0.34 % of the run. The next move for this lane is the depth decoder — which
§14.5 already records as owed, and correctly records as needing bf16 STORAGE
rather than a transcription onto an f32 `vt::MatmulBT`.

**Both of those costs were then measured further, and one of them is already
solved.** §15.6 shows the load is I/O and that staging the checkpoint to local
disk removes 137x of it, so of the two items above only the depth decoder
survives as work. §15.7 measures the developer's actual request — 20 s at 30
steps — where the load amortises to 18.6 % and the depth decoder rises to
91.7 % of the autoregressive loop. **The single remaining answer to "why does
this take so long" is the RVQ depth decoder running as a host scalar loop.**

### 15.6 Staging the checkpoint to local disk — the load cost is I/O, and staging REMOVES it

**Device: `thor:gpu0` — NVIDIA Thor, sm_110, aarch64, 14 cores, ~122 GB unified,
driver 595.78.** One job, one binary, three storage states, minutes apart, box
load 4.8-10.6 throughout. Job `72db2b49`, built from `6925548`.

**The instrument checked itself first, and the checks are the reason the figures
are quotable.**

| check | result |
|---|---|
| staging copy completed | `SRC_BYTES=28517617303` = `DST_BYTES=28517617303`, asserted, hard-fail on mismatch |
| which filesystem the staged bytes came off | `findmnt /tmp/ckpt` -> `overlay overlay` |
| which filesystem the CIFS arm came off | `findmnt /workspace/music3/ckpt` -> `//192.168.68.102/Data[/rc] cifs` |
| the eight breakdown rows exist in the built source | `breakdown_rows_in_source=8`, all named |
| the eight rows were actually EMITTED by the binary | `breakdown_rows_emitted=8` on every probe |
| local-disk sequential read ceiling | `cat` all 28.518 GB to `/dev/null`: **7 s = 4074 MB/s** |
| staging throughput off CIFS | 679 s for 28.518 GB = **41 MB/s** |

**The three figures, each labelled, never averaged** (`load.ar_weights`):

| state | `load.ar_weights` | `load.acoustic_weights` | aggregate for the 28.518 GB |
|---|---|---|---|
| **cold-CIFS** | **780.015 s** | 249.018 s | — |
| **warm-CIFS** | **428.452 s** | 257.876 s | — |
| **CIFS, same job as the local-disk probes** | **368.390 s** | 215.358 s | **49 MB/s** |
| **local-disk, first probe** | **7.539 s** | 3.406 s | **2606 MB/s** |
| **local-disk, second probe** | **5.689 s** | 2.139 s | **3643 MB/s** |

The third row exists so the comparison does not have to cross jobs: it is the
same box, the same binary and the same 0.24 s request as the two local-disk
probes, taken minutes apart, differing only in which path `--model` names.

**THE VERDICT: staging works. The residual was I/O, not loader CPU.**

**780.015 s cold-CIFS -> 5.689 s local-disk is 137x. Against the same-job CIFS
control it is 65x.** The developer's staging directive does exactly what it was
meant to do, and the ~7-minute wait it was aimed at becomes ~6 seconds.

### 15.6a The breakdown, which is why the verdict needs no argument

Same job, same binary, the CIFS control against the first local-disk probe:

| row | CIFS | local-disk | ratio | reads a file? |
|---|---|---|---|---|
| `load.ar.open_shards` | 0.823 | 0.001 | 823x | header only |
| **`load.ar.lm_weights`** | **338.572** | **6.121** | **55x** | yes, ~17.6 GB |
| `load.ar.depth_weights` | 27.998 | 0.874 | 32x | yes, ~1.3 GB |
| `load.ar.tokenizer` | 0.794 | 0.439 | **2x** | small JSON, then CPU parse |
| `load.ac.condition` | 2.232 | 0.056 | 40x | yes, tiny |
| `load.ac.vocoder` | 4.443 | 0.135 | 33x | yes, 0.2 GB |
| **`load.ac.dit_read`** | **208.671** | **3.190** | **65x** | yes, 9.7 GB |
| **`load.ac.dit_build`** | **0.000** | **0.000** | — | **NO** |

Three shapes, and together they close the question:

* every row that **reads bytes** collapses by 32-65x when the bytes move to
  local disk;
* **`load.ac.dit_build` is 0.000 s on every arm.** It is
  `DitWeightsFromTensors` rebuilding the weight struct from the map
  `load.ac.dit_read` already filled, so it touches no file — and it costs
  nothing. The host copy that the "loader CPU" hypothesis pointed at **does not
  exist as a cost**;
* `load.ar.tokenizer` is the control in the other direction: the one row that is
  mostly CPU (parsing `tokenizer.json`) is also the one row that barely moves,
  2x against its neighbours' 32-65x.

A hypothesis that the residual was loader CPU predicts the opposite of all three.

### 15.6b One figure sits above the stated ceiling, and the ceiling is what is wrong

The `cat`-to-`/dev/null` control gives **4074 MB/s**, and any load implying more
than that would be void. On the aggregate — the only quantity whose byte count is
known exactly — neither local-disk probe exceeds it: **2606 MB/s** and
**3643 MB/s**.

Split by half, the second probe's acoustic arm implies ~4650 MB/s, which is
above the control. **That does not void the figure; it bounds what the control
bounds.** The control measures `read(2)` through `cat` into a pipe; the loader
mmaps and faults pages that the first probe has already made resident. Those are
different access paths and the second is legitimately faster. The aggregate is
reported as the ceiling test because its byte count is measured
(`SRC_BYTES`) rather than apportioned, and the per-half split is not.

Recorded rather than dropped, because a self-check that fires and is then
silently ignored is worse than no self-check.

### 15.7 The developer's exact configuration, measured: 20 s of audio at 30 steps

`thor:gpu0`, `--device 1`, `VLLM_CPP_VOCODER_DEVICE=cuda`, 500 AR frames, 4
windows, checkpoint on **CIFS** (this run predates the staging probe). Box load
3.2-17.9. **TOTAL 3269.789 s = 54.5 min**, delivering 20.016 s of 44.1 kHz
stereo (RMS 0.15282, peak 0.98182, no clipping).

| kind | stage | seconds | % | calls |
|---|---|---|---|---|
| leaf | **`ar.depth_forward`** | **1710.456** | **52.31** | 7014 |
| leaf | `vocoder.decode_window` | 421.670 | 12.90 | 4 |
| leaf | **`load.ar_weights`** | 397.909 | 12.17 | 1 |
| leaf | `denoise.dit_device` | 370.746 | 11.34 | 120 (= 240 forwards) |
| leaf | `load.acoustic_weights` | 210.482 | 6.44 | 1 |
| leaf | `ar.lm_decode_step` | 83.629 | 2.56 | 500 |
| leaf | `ar.depth_projection` | 49.580 | 1.52 | 7014 |
| leaf | `ar.lm_prefill` | 11.649 | 0.36 | 1 |
| leaf | `ar.depth_head_and_draw` | 3.300 | 0.10 | 3507 |
| leaf | `denoise.condition_mix` | 1.899 | 0.06 | 4 |
| leaf | `ar.semantic_guide_and_draw` | 1.725 | 0.05 | 501 |
| leaf | `acoustic.dit_staging` | 1.202 | 0.04 | 1 |
| span | `ar.depth_stage` | 1768.593 | 54.09 | 501 |
| span | `ar.TOTAL_loop` | 1865.650 | 57.06 | 1 |
| | `sum(leaf)` | 3264.249 | 99.83 | |
| | `unattributed` | 5.540 | 0.17 | |

**At the duration the developer actually asked for, the shape changes and the
answer sharpens.** The load is no longer the headline — it is 18.6 % (608.4 s of
the two loads) because it is a FIXED cost being amortised over 5x more audio,
and §15.6 now removes almost all of it anyway. What is left is the
autoregressive loop at **57.06 %**, of which **91.7 % is `ar.depth_forward`**.

**The 0.646B RVQ depth decoder on the host costs 20.5x the 8.6B language model on
the GPU** (1710.456 s against 83.629 s), and it is **14.00 calls per depth stage**
(7014 / 501) — exactly upstream's work, done unbatched.

The DiT is now visible but still not the problem: **1.545 s per forward** over
240 forwards at latent length ~689, 11.34 % of the run. The vocoder at 12.90 %
has overtaken it.

**With the checkpoint staged (§15.6), the same run would be ~2670 s and the depth
decoder would be 64 % of it.** That is the next row, and §14.5 already records
what it needs: bf16 STORAGE, not a transcription onto an f32 `vt::MatmulBT`.

### 15.8 The oracle comparison is a NORMALISATION, not a matched-hardware pair

**This is not a benchmark and must never be quoted as one.** The two arms ran on
different machines:

* **ours** — `thor:gpu0`, NVIDIA Thor sm_110, 14 aarch64 cores, box load 3.3,
  the DiT and the language model on the GPU;
* **the oracle** — diffusers PR #14456 at `c6da9936`, **x86-64, 20 cores, CPU
  only, at box load 193** (recorded at the end of its own run).

**The oracle arm's CALL COUNTS are load-bearing. Its SECONDS are not**, and no
ratio between the two hosts' wall clocks appears below or anywhere else in this
spec.

What *is* matched is the request: `--duration 0.24 --steps 2 --seed 7`, the same
description, and the same lyrics file, giving **223 prompt tokens on our side
against the oracle's prefill shape `(2, 224, 4096)`** — a one-token difference,
which is the evidence that the inputs really are the same rather than merely
described the same way.

| | ours (Thor sm_110, load 3.3) | oracle (x86 20-core CPU, load 193) |
|---|---|---|
| audio delivered | 0.232 s | 0.232 s |
| AR frames / windows | 6 / 1 | 6 / 1 |
| depth-decoder forwards per stage | **14** (batch 1, seq 2..8, x2 CFG) | **7** (batch 2) — same rows |
| depth `projection` rows per frame | **70** | **16** |
| `lm` decode calls | 1 per frame, incremental KV | 1 per frame, incremental KV |
| DiT forwards | 4 | 4 |
| vocoder calls | 1 | 1 |
| wall clock | 712.697 s (**93.8 % of it CIFS checkpoint load**) | 827.434 s generate + 212.366 s load |

The wall clocks are printed only so the record is complete. Ours is dominated by
a checkpoint load off CIFS that §15.6 shows is removable, and the oracle's was
taken on a box carrying a load average of 193 — **neither number measures the
model, and dividing one by the other would measure nothing at all.**

### 15.9 The vocoder device arm, priced at a small size — and it LOSES there

Same job, same request, the only difference being `VLLM_CPP_VOCODER_DEVICE`:

| arm | `vocoder.decode_window` (latent length 20) |
|---|---|
| `cuda` | 3.552 s |
| host (unset) | **2.983 s** |

At 20 latent frames the device arm is **19 % SLOWER**, which is what a
per-convolution host/device round trip costs when the convolution is tiny. That
is consistent with §13.6 keeping the arm opt-in, and it is a reason to keep it
opt-in that is now measured rather than argued. At the 20 s clip's ~689 latent
frames the device arm ran 105.4 s per window; no host arm was taken at that size,
so no crossover point is claimed.

---

## 16. The depth decoder stops recomputing itself (#672) — §11.4's last owed row, and it did not need the dtype

> **This section was §15 until #1237 landed `## 15. Where the time ACTUALLY goes`
> on `main` at the same end-of-file position.** It is §16 from that merge onward.
> The bodies of issues #1246 and #1247 were written before the merge and cite
> `§15.2`, `§15.3`, `§15.5` and `§15.6` meaning the subsections that are now
> §16.2, §16.3, §16.5 and §16.6. Those two rows are corrected here rather than in
> `.agents/issue-index.md`, because that index is append-only and an edited row
> is duplicated rather than merged.

§14.5 left the depth decoder as the one owed device row and corrected §11.4's
reason for it: routing it through an f32 `vt::MatmulBT` would drop the bf16
`Store` that every gated number was taken with, so a device arm needs **bf16
storage** and its own numeric evidence.

**That is still true, and this row did not need it.** The measured bottleneck
turned out not to be how fast the depth decoder's arithmetic runs but **how much
of it was arithmetic nobody read**.

### 16.1 The measurement that defined the job

`row/MUSIC3-PERF-VS-ORACLE`'s per-stage profiler (#1231), `thor:gpu0`, `--device 1`,
4 s of audio / 4 steps / 100 frames / 1 window, 1481.5 s total:

| stage | seconds | share | calls |
|---|---|---|---|
| `load.ar_weights` | 780.0 | 52.7 % | — (cold CIFS) |
| **`ar.depth_forward`** | **347.3** | **23.4 %** | **1414** |
| `load.acoustic_weights` | 249.0 | 16.8 % | — (cold CIFS) |
| `vocoder.decode_window` | 53.6 | 3.6 % | — |
| `ar.lm_decode_step` | 14.8 | 1.0 % | 100 |
| `ar.depth_projection` | 10.1 | 0.7 % | 1414 |
| `denoise.dit_device` | 5.06 | 0.34 % | 8 forwards |

**The 0.646B host depth decoder cost 23.5x the 8.6B language model on the GPU**,
and once the cold load is set aside it was ~88.5 % of the autoregressive loop.
§14 having put the DiT on the device is why: at 0.34 % there is nothing left to
win there, and the stage that was second is now first.

### 16.2 What it was spending it on: rows it already had

`_generate_depth_codes` (encoders.py:117-142) walks seven codebook steps over a
sequence that grows by one row per step, and reads `hidden[:, -1]` each time. Our
`Music3DepthStage` implemented that literally — `DepthSequenceEmbeds` then
`DepthDecoderForward` over the WHOLE sequence, once per CFG branch:

    rows through the decoder, per frame:   2 x (2+3+4+5+6+7+8) = 70 FORWARDED
                                           14 READ (2 CFG branches x 7 steps)
    rows through the 4096x4096 projection: the same 70 PROJECTED
                                           16 DISTINCT (branch, position) pairs

**Upstream re-runs the sequence too** — its profile shows `depth.forward` at
7 calls per frame with shapes `(2, 2..8, 4096)` — so nothing was mis-ported.
It is an algebraic identity that neither side had taken, and upstream's own
`projection` line (56 calls, 8 of them batch-2, per frame) shows it HAS taken the
smaller half of it: `sequence` keeps its projected rows and only the newly
appended element is projected (encoders.py:125,127,141).

### 16.3 The identity, and why it is exact rather than close

The decoder is causal and carries position in a **learned table**, not in RoPE
(`minimax_music3_rvq_depth_decoder.py:138-139`). So the input to position `t` is
fixed the moment it is appended, and by induction over the layers every
intermediate of positions `0..t-1` is bit-for-bit what a whole-sequence forward
recomputes. `DepthDecoderAppend` computes each position ONCE against a K/V cache
and returns the row the schedule actually reads.

    rows through the decoder, per frame:  16   (2 branches x 8 positions)
    rows through the projection:           9   (3 prefix + 6 appended)

**16 and 9 count different things, and §16.2's "16" is the first of them.** 16 is
the number of distinct **(branch, position) pairs** a frame has: 2 x 8. 9 is the
number of distinct **row values** that have to be projected, because the two CFG
branches share every row except position 0 — 2 last-hidden rows, the shared
semantic row, and 6 fed-back codes. The decoder must see all 16, one per branch
per position; the projection sees each value once. Neither number is 14, which is
what the OLD arm READ (2 branches x 7 steps) out of the 70 rows it forwarded.

Two further things fall out, and both are upstream's own shape rather than
inventions of ours. The two CFG rows differ ONLY at position 0, so they go
through as **one batch-2 call** — `(2, 2..8, 4096)` is exactly what upstream's
profile shows. And the projection is applied once per row: three rows in one
sweep for the prefix, one per appended code.

**`DepthDecoderForward` is untouched.** It is the mirrored reference, it is still
what the committed upstream goldens gate, and it is what `DepthDecoderAppend` is
asserted equal to. A fast path that replaced its reference would have nothing to
be checked against.

### 16.4 The other half: the kernel was weight-streaming bound, not FLOP bound

`LinearNoBias` partitioned its output as a flat `(row, out)` index, so for each
input row it swept the WHOLE weight matrix. At the depth decoder's geometry one
row-forward reads

    4 layers x (4 x 4096x4096 + 2 x 4096x6144 + 6144x4096) x 4 bytes = 2.28 GB

of f32 weights for 570 MMAC, which is ~4 bytes per multiply-accumulate: the loop
is bound by streaming weights, not by arithmetic. 347.3 s over 70 row-forwards
per frame x 100 frames is ~46 GB/s of weight traffic, which is the right order
for this box's memory rather than for its cores.

So the row loop moved INSIDE one output column. Each output element still owns
one sequential `double` accumulator walked in ascending `i` — the order W2/W3
pinned against torch, under the project-wide `-ffp-contract=off` — but the weight
row is now read once for every row in the call instead of once per row.

| per frame | before | after |
|---|---|---|
| decoder row-forwards | 70 | **16** |
| decoder weight sweeps | 70 | **8** |
| projection rows | 70 | **9** |
| projection weight sweeps | 70 | **7** |

### 16.5 Correctness — BITWISE, and the gate had to earn it

`tests/vllm/models/test_minimax_music3_ar.cpp` gains four cases that compare
`DepthDecoderAppend` against `DepthDecoderForward` by `memcmp`, not by tolerance:

* every prefix length 1..`max_position_embeddings`, at BOTH `ArCompute` widths
  (96 values at the goldens' geometry);
* a **wider** geometry — 8 heads of 8, three layers, the real 8-position
  schedule, pseudo-random weights — for coverage BREADTH: 1024 values compared
  against the goldens' 96, at a head count and a head_dim that differ from each
  other and from the goldens'. **Not because a head stride is invisible at the
  goldens' geometry.** That reason was recorded and it was false (#1247):
  `minimax_music3_ar_goldens.inc` sets `kMusic3DepthHeads` = 2 over
  `kMusic3DepthHidden` = 8, so the goldens are 2 heads of 4 and
  `heads * head_dim` (8) is NOT `head_dim` (4). Dropping the head stride from the
  cached KEY index reds 4 cases / 32 assertions, two of them at the goldens' own
  geometry;
* the CFG pair as a batch of 2, each row against its OWN whole-sequence forward,
  plus a check that the two rows have not collapsed into each other;
* the cache's refusals: a null cache, batch 0, a mis-sized row, a batch change
  mid-cache, and the position ceiling.

**An f64 accumulator stored through an f32 cannot see a reassociation.** That is
measured here rather than supposed — reversing the attention's value sweep leaves
every ordinary-data assertion in the file GREEN. So the fourth case engineers the
cancellation: `to_q` and `to_k` zero make every softmax weight exactly `1/seq`,
and an `input_layernorm` whose first component is `2^60` makes positions 0 and 1
carry `+2^60` and `-2^60` into `v` while every later position carries 0 there and
O(1) elsewhere. Ascending `j` cancels the pair immediately and keeps the
remainder exactly; any order that carries `2^60` through the remainder
annihilates it across a 57-bit gap. The case also asserts its OWN teeth — that
the remainder is finite and non-zero — because a zero remainder would make the
whole apparatus vacuous and still print green.

**Mutations: 7 applied, 7 RED.** Sources restored and `sha256`-verified after
each.

| # | mutation | result |
|---|---|---|
| M1 | attention value sweep walks `j` DESCENDING | **RED** 1 case / 4 — the cancellation case ALONE; every other case green |
| M2 | `pos_embedding` read at row 0 instead of `position` | **RED** 3 cases / 30 |
| M3 | the appended position excluded from its own attention | **RED** 4 cases / 35 |
| M4 | every batch row served row 0's history | **RED** 2 cases, THROWN — 0 failed assertions, which is why cases are reported beside them |
| M5 | `LinearNoBias` drops the last input row | **RED** in both suites |
| M6 | `Store` dropped from the incremental position add | **RED** 3 cases / 30 |
| M7 | `LinearNoBias` dot split into two interleaved accumulators | **RED** `test_host_parallel` 1 case / 5 |

M7 is the one that gates the kernel restructure, and it reds through
`test_host_parallel`'s own cancellation case (§12.2) rather than through anything
added here — which is the check that the row-loop pivot did not quietly become a
reassociation.

M1 and M4 are each recorded with their shape rather than only their count: M1
because it is the whole justification for the engineered case, and M4 because it
fails by THROWING, so a reader grepping `assertions:` would see a suite that
"passed" 408 of 408.

**The COMPOSITION leg — a fifth case, because the fingerprint was not one
(#1246).** As first landed, nothing in the tree entered `Music3DepthStage`:
`test_minimax_music3_ar_real` exercises `DepthDecoderForward`,
`test_minimax_music3_llm_real` checks `frame_hiddens[:, :4096]`, and the four
cases above gate `DepthDecoderAppend` one row at a time. The schedule composed on
top of it — the 3-row prefix projection, position 0 fed for its K/V alone, the
batch-2 sequencing, the fed-back `(index-1) * audio_vocab_size + drawn`
projection row — had no gate at all. Deleting the prefix append left all five
music3 suites GREEN, and so did silently dropping the feedback row, which changes
the generated song.

**§16.6's fingerprint could not close that**, and that is the correction which
matters most here: `tools/bench/music3_depth_stage_ab.cpp` is a hand
TRANSCRIPTION of the schedule, not a call to `Music3DepthStage`. A transcription
cannot detect divergence between itself and the function it transcribes, and
nothing compiled it either. Both halves are repaired. The driver is now compiled
by CI as `vllm_music3_depth_stage_ab_{before,after}` (§16.6), and
`test_minimax_music3_ar` gains a fifth case that drives the PRODUCTION
`Music3DepthStage` against a transcription of the whole-sequence schedule it
replaced — same sampler, same weights, same order of draws — at 8 heads of 8,
2 layers, 8 codebooks and a 32-entry audio vocabulary, wider than any committed
geometry and needing no checkpoint. It compares 448 values bitwise, asserts that
the drawn codes and the draw count agree, and asserts its own teeth (448 of 448
reference values non-zero).

| # | mutation | result |
|---|---|---|
| N8 | delete the 3-row prefix's position-0 K/V append | **RED** 1 case / 2 assertions, the new case ALONE |
| N9 | silently drop the fed-back projection row | **RED** 1 case / 2 assertions, the new case ALONE |

**What the new case still does NOT reach**, said rather than implied: it enters
at `Music3DepthStage`, one hop below the top. The remaining hop is the single
unconditional call in `Music3GenerateFrameHiddens`, which needs the 8.6B language
model and so has no checkpoint-free driver; the registered speech family reaches
it through `MiniMaxMusic3SpeechEngine`. Everything downstream of the returned
block — `AudioHeadLogits`, the CFG mix, the top-k draw, the feedback embedding,
the acoustic half — is unchanged code, and `LinearNoBias`, the one kernel they
share with this row, is gated bitwise by `test_host_parallel`. §16.6's
fingerprint remains what proves the identity at 4096-wide PRODUCTION geometry
across two separately compiled libraries, which is a leg the unit gate does not
have; it is no longer asked to be the composition gate as well.

**What is NOT claimed is the WAV pair, and it was attempted.** §12.3's strong
form — two binaries writing byte-identical audio through five stages and a
28.5 GB checkpoint — is the leg this row does not have. The run was launched and
died at `rc=127` on a missing `libvllm.so.0`: the BEFORE arm's binary was
dynamically linked into a measurement worktree that had already been removed. It
is recorded as attempted and not taken rather than quietly omitted, and rebuilding
the pair is owed work whenever a quiet box is available; §16.7 carries it.

### 16.6 Speed — a STAGE A/B, on one named box, and why not the e2e pair

**One frame of the depth stage, at the REAL geometry, runs 3.5x faster and emits
the same bytes.** That is a STAGE measurement, said so plainly, and it is not
offered as an end-to-end speedup.

**Why not the e2e pair, again.** §12.5 recorded the same refusal for the same
reason and it recurred: this 20-core x86-64 box was carrying three other
sessions' `test_ltx2_video` runs and two full `ctest` builds throughout, at a
1-minute load average between **39 and 52** (5-minute 71-92). A wall-clock e2e
pair taken there measures somebody else's scheduler. The Thor per-stage pair —
the one that would price this against §16.1's own profile — has since been
taken twice. The first result is **VOID** and §16.6a records why the number
cannot be read; §16.6b is the corrected pair, and it measures **4.45x on the
depth forward and 2.74x on wall** against the real checkpoint.

**What replaces it.** The depth stage's inner loop is short enough to repeat, so
the MINIMUM over rounds is available, and a minimum is the least-disturbed sample
rather than an average of someone else's contention (§12.4's argument, reused
because the situation is the same). ONE driver source —
`tools/bench/music3_depth_stage_ab.cpp`, in the tree, and NOT the
`depthbench.cpp` an earlier revision of this paragraph named — is compiled twice
and linked against the two `libvllm.a` builds. The BEFORE arm is **`fc163f62b`**,
the `row/MUSIC3-PERF-VS-ORACLE` head; the AFTER arm is that commit plus this
change, and they differ in exactly the four files this row touches. The benchmark
record carried `origin/main` `727163997` in its heading beside `fc163f62b` in its
body; `727163997` is only the `origin/main` commit `fc163f62b` had merged, the
delta between them is #1231's profiler alone, and the driver never enters it
(#1247).

**CI now COMPILES that driver, both arms, and runs neither.** It reaches internal
headers and `LinearNoBias`, it is the only artifact a reader can reproduce this
section from, and nothing built it — so the next signature change would have
rotted it silently. `CMakeLists.txt` builds it as the OBJECT libraries
`vllm_music3_depth_stage_ab_{before,after}`, linked nowhere, so no weight is ever
allocated and CI's runtime is unchanged. The guard paid for itself on its first
run: the file did not compile under the project's own `-Werror=comment`. It drives ONE frame of the
depth stage as `Music3DepthStage` drives it: seven codebook steps, two CFG rows,
the projection and the decoder, with a fixed code per step so the two arms
traverse identical rows. `DepthDecoderConfig`'s defaults are the real 4096 / 4 /
16 / 6144 / 8 geometry, and the weights are 2.5 GB of seeded pseudo-random floats
drawn identically on both arms.

Eight alternating pairs across two series (base, new, base, new, ...). Series A
is 5 pairs x 1 round and series B is 3 pairs x 4 rounds, so **17 timed rounds per
arm** across **16 processes**:

| series | BEFORE rounds (s) | AFTER rounds (s) | pair ratio |
|---|---|---|---|
| A, 1 round/process | 6.7769 / 6.6714 / 6.9562 / 7.5485 / 6.4413 | 1.7795 / 2.0487 / 2.5047 / 2.4196 / 2.1171 | 3.81, 3.26, 2.78, 3.12, 3.04 |
| B, 4 rounds/process | 6.5467 5.8667 6.2444 6.0943 / 6.3505 5.9991 5.8783 6.5534 / 6.3186 6.3508 8.6671 7.6575 | 1.9943 2.0208 1.8607 1.6766 / 2.0193 2.1022 1.6981 1.8891 / 3.3631 3.0259 3.4770 3.7275 | 3.50, 3.46, 2.09 |

    minimum over all rounds:  BEFORE 5.8667 s   AFTER 1.6766 s   3.50x
    median of the 8 pair ratios:                                 3.19x

`uptime` 39.30 before the series and 51.98 after, 20 cores; the noisy rounds are
visibly higher on BOTH arms, which is what the minimum exists to discard. Series
B's third pair is the loudest on both arms and is kept rather than dropped.

**And the bit-identity holds AT THIS GEOMETRY, which is a leg the unit gate does
not have.** The driver computes an FNV-1a fingerprint of the frame's 28 672 depth
hidden values every round and prints ONE line per process, after the round loop.
The table is 8 alternating pairs x 2 arms, so **16 processes printed one
fingerprint each and all 16 read `f0cfeed6eee4f55d`**. An earlier revision said
"all 20 runs", which is neither the 17 rounds per arm nor the 16 processes
(#1247). §16.5 gates reduced dimensions against the reference forward; this gates
4096-wide production geometry across two separately compiled libraries.

#### 3.5x, not 8.75x — recorded because the gap is the finding

§16.4's byte accounting says the weight traffic falls 8.75x and the arithmetic
4.375x. The measured 3.5x sits just BELOW the arithmetic ratio, not near the byte
one, which says this stage is closer to compute-bound on THIS box than the
byte-per-MAC arithmetic suggested — a 20-core x86-64 with a large L3 is not the
regime the ~46 GB/s figure in §16.4 was inferred from, and that figure came from
a 14-core Jetson Thor with unified LPDDR5X. The two boxes do land on
different sides of it: §16.6b measures **4.45x on Thor, just above the 4.375x
arithmetic ratio**, against this 3.50x just below it. That is the direction the
paragraph above predicted, which is why §16.6b labels the explanation a
hypothesis rather than treating the agreement as evidence.

Two costs are also real and are named rather than absorbed: the incremental arm
makes **8 pooled calls per frame where the old one made 14 larger ones**, so it
pays proportionally more `Threadpool::Barrier` — the row's AR half already spends
~25 % of its wall clock there (§11.4) — and its per-step attention is a scalar
loop over one query row that no longer rides the output-row partition. Both are
visible in the after arm's higher round-to-round spread (1.68-3.73 against
5.87-8.67, a wider RELATIVE band). The Thor number in §16.6b now says this stage
sits nearer the arithmetic bound than the byte one on that box too, so neither
is worth a lever ahead of the vocoder, which §16.7 names as the larger term.

### 16.6a The Thor pair was taken, and the first one is VOID — both arms were the same binary

**The first Thor per-stage pair (job `56848b2e`, `thor:gpu0`) is VOID. It did not
measure this change, because the two arms were byte-identical binaries.** Its own
log is what says so:

```text
gen_before: 72744 bytes sha=b98a5dbba37a67f1
gen_after:  72744 bytes sha=b98a5dbba37a67f1
```

One `sha256` for both arms. The job reused a single source tree and a single
build dir across the two arms, so the second configure-and-build found the tree
already up to date and produced no distinct binary. `gen_before` and `gen_after`
are the same program, run twice.

**What it reported is therefore not a result and is not recorded as one.** It
showed `ar.depth_forward` at 77.930 s on the "before" arm against 77.726 s on the
"after" arm, with an identical **808 calls on both**. Those two numbers are a
0.26 % difference between two runs of one binary, which is run-to-run noise on a
shared box; the identical call count is the same fact stated a second way, since
§16.2's whole claim is that this change takes the depth stage from 14 calls a
frame to 8. **A pair that cannot move the call count cannot have contained the
change.** Read at face value the pair would have refuted §16.6's 3.50x, and it is
recorded here precisely so that it cannot be quoted later as a refutation.

**This is the class of defect §16.5's own gate work is about, arriving from the
other side.** A mutation that fails to rebuild reads as a passing test; an A/B
that fails to rebuild reads as a levelled speedup. In both the instrument
silently measures the previous artifact and returns a verdict about the code.
The rule both cases want is the same one: **whenever two artifacts are required
to differ, assert that they differ before believing anything downstream of them.**

**The corrected pair (job `7b22b5b0`) has since run, and §16.6b is its result.**
It took the two arms from **separate clones with separate build dirs**, behind a
hard `FATAL_ARMS_IDENTICAL` guard on the two binaries' `sha256`, so this failure
would abort the job rather than produce a plausible table. The guard reported
`ARMS_DIFFER=yes` on two distinct hashes.

**CORRECTED 2026-08-22 ([#1516](https://github.com/mudler/vllm.cpp/issues/1516),
row `BENCH-AB-ARMS-CONTROL`, spec
[`ab-arms-control.md`](ab-arms-control.md)): that `ARMS_DIFFER=yes` is NOT the
precondition this section exists to insist on, and this paragraph used to say it
was.** `minimax-music3-gen` is a 72 744-byte client of `libvllm_shared.so` and
nothing sets `CMAKE_SKIP_BUILD_RPATH`, so CMake writes the build-tree RPATH into
it and two build directories hash apart from identical source. Reproduced on a
minimal project of the same shape: two byte-identical source trees give two
equal-sized clients with different hashes, and making the library change for
real leaves the client byte-for-byte the hash it already had. The guard would
have reported `yes` on two clones of one commit. **The rule the paragraph above
draws is right and stands; the leg that carries it is the CALL COUNT, not the
hash** — which is the same tell that voided this pair, read in the other
direction.

### 16.6b The corrected Thor pair — the real-checkpoint number, and the WAV identity leg it closes

**On Thor, against the real 28.5 GB checkpoint, the depth forward runs 4.45x
faster and the whole run finishes 2.74x sooner, writing byte-identical audio.**
Job `7b22b5b0`, `thor:gpu0` sm_110, `--device 1 --duration 4 --steps 4 --seed 7`,
100 frames. Arms **`c802dba8d`** — this row's merge-base, so the delta is exactly
the 10 files this row touches — and **`4568c6e71`**. Three alternating pairs.

**The instrument's own preconditions were asserted before any timing was read**,
which is the whole reason this pair exists:

    gen_before sha=33f5c5fb18a7e91a5fe7b2fe26f7f5c1   ARMS_DIFFER=yes
    gen_after  sha=d2fdac95a34ce177bbdd9e766e87078a
    STAGE_SECONDS=815   SRC_BYTES=DST_BYTES=28517617303

**Which of those lines is load-bearing, corrected 2026-08-22 (#1516, spec
[`ab-arms-control.md`](ab-arms-control.md)).** The `ARMS_DIFFER=yes` is
vacuous for this binary: it is an ABI client whose hash tracks the build
directory rather than the change. **The result is not in doubt, because the leg
that does separate these arms is in the table below** — `ar.depth_forward` moves
1414 -> 808 calls, and a pair that could not contain the change cannot move a
call count. The `STAGE_SECONDS`/`SRC_BYTES`/`DST_BYTES` assertion is unaffected
and stays load-bearing.

The checkpoint was staged to local disk first, with source and destination byte
counts asserted equal, so §15.6's cold-CIFS load is not inside these figures.

| bucket | BEFORE, mean of 3 | AFTER, mean of 3 | ratio | calls |
|---|---|---|---|---|
| `ar.depth_forward` | 348.273 s | 78.316 s | **4.45x** | 1414 -> 808 |
| `ar.depth_projection` | 10.102 s | 1.307 s | **7.73x** | 1414 -> 707 |
| `ar.depth_stage` | 359.090 s | 80.221 s | **4.48x** | 101 -> 101 |
| `ar.TOTAL_loop` | 375.687 s | 94.084 s | **3.99x** | 1 -> 1 |
| **wall clock** | **446.33 s** | **163.00 s** | **2.74x** | — |
| `vocoder.decode_window` | 53.648 s | 53.605 s | 1.00x | 1 -> 1 |
| `ar.lm_decode_step` | 14.989 s | 12.339 s | 1.21x | 100 -> 100 |

The spread is tight enough that the means are not hiding anything.
`depth_forward` reads 348.076 / 349.588 / 347.154 before and 78.190 / 78.627 /
78.131 after — the two bands are three orders of magnitude apart in separation
relative to their own width. Wall is 450 / 446 / 443 s before and 164 / 163 /
162 s after.

**The call count is the control that the void pair failed.** 1414 depth forwards
before against 808 after is 101 frames x 14 and 101 frames x 8, exactly the
schedule change §16.2 describes, and `depth_projection` at 1414 -> 707 is
101 x 14 and 101 x 7. The void pair's 808-on-both-arms is now positively
explained: it ran the AFTER binary twice.

**Where the 4.45x comes from, decomposed.** Per call, `depth_forward` costs
246.30 ms before and 96.93 ms after. So the stage is **1.75x fewer calls times
2.54x cheaper per call**, and the two factors multiply to the measured 4.45x.
The per-call figures are not like-for-like by design — a BEFORE call re-forwards
the whole growing depth sequence while an AFTER call appends one position against
a cache — and that is the change, not a confound.

#### 2.74x is the number a user feels, and it is smaller for a reason worth naming

**Lead with 2.74x, not 4.45x.** The depth stage is no longer the whole run. It
was **80.5 % of wall before and is 49.2 % after**, and the run is now dominated
by a term this change does not touch: `vocoder.decode_window` costs **53.6 s on
both arms**, going from 12.0 % of wall to **32.9 %**. A fixed serial phase does
not shrink when you speed up the parallel one, so the stage ratio and the wall
ratio are different claims and each is quoted where it applies. The depth stage
saved 278.9 s and the wall fell 283.3 s, so essentially all of the wall saving is
this change and there is no unexplained gain hiding in the total.

**One delta is NOT explained and is recorded rather than smoothed.**
`ar.lm_decode_step` fell 14.989 s -> 12.339 s (1.21x) across an unchanged 100
calls, and this row does not touch the LM decode. The plausible reading is that
the AR loop's working set is smaller once the growing depth sequence is gone, so
the LM's own memory traffic gets cheaper — but that is a **hypothesis**, it was
not measured, and 2.65 s of the 283.3 s wall saving therefore has no established
cause. It is named here so that a later reader does not attribute it to the depth
schedule by default.

#### The e2e WAV identity leg is TAKEN — not owed, and not at unit scale

**All three pairs wrote `sha 5e81fc133d653560` on BOTH arms.** Six runs, one
hash. The two arms also report identical `RMS 0.01350` and `peak 0.25180` on
every run, and identical geometry (3.994 s, 44100 Hz, 2 channels, 176 128
samples per channel).

This is the leg §12.3 had and §16.5 could not reach, and it is a strictly
stronger statement than the unit gate or than §16.6's fingerprint. It runs the
real checkpoint through **every** stage — AR loop, depth decoder, DiT denoise,
and the vocoder — and compares the finished audio file, so it proves the change
is inaudible in the product rather than bit-exact at a layer boundary. §16.7 no
longer carries it as owed.

#### 4.45x on Thor against 3.50x on x86 — two measurements, not one confirming the other

These are **different machines, different weight sources and different arms**,
and neither validates the other. §16.6 is a synthetic in-process bench on a
20-core x86-64 under load 39-52, driving `DepthDecoderConfig` defaults against
2.5 GB of seeded pseudo-random weights, with arms `fc163f62b` and that commit
plus this change. §16.6b is the shipped binary on a 14-core Jetson Thor against
the real checkpoint, with arms `c802dba8d` and `4568c6e71`. Each is quoted with
its box and its weight source at the point of use, and the row claims neither as
a reproduction of the other.

**Why Thor is higher is a HYPOTHESIS, and it is labelled one because it was
predicted rather than measured.** §16.4's accounting gives a weight-traffic ratio
of 8.75x and an arithmetic ratio of 4.375x, and a stage that is bandwidth-bound
should land nearer the first. Thor's **4.45x sits just above the arithmetic
ratio; x86's 3.50x sits just below it**, which is the direction a unified-LPDDR5X
box versus a large-L3 x86 box would predict. That the prediction matches is
suggestive and nothing more — the ~46 GB/s figure §16.4 reasons from was itself
inferred from Thor, so this is the same argument returning, not independent
evidence for it. Settling it needs a bandwidth measurement on both boxes, which
no axis here has.

### 16.7 What is still OWED after this row

**The dtype decision is untouched and still owed.** §14.5 stands: the depth
decoder runs `ArCompute::kBFloat16`, an f32 `vt::MatmulBT` would drop that
rounding, and a device arm for this stage needs **bf16 storage** with its own
numeric evidence. This row deliberately did not take that on — it removed
arithmetic nobody read and made the arithmetic that remains stream half the
bytes, both bit-identically, and neither claim needs a dtype argument. **A device
arm for the depth decoder is worth strictly less after this row than before it**,
because it is now 4.4x less arithmetic to move.

**The next traceable hypothesis, named rather than left as a ceiling.** The
kernel is weight-streaming bound at ~4 bytes per multiply-accumulate. Three
levers are visible and none is taken here:

1. **bf16 weight STORAGE for the host arm.** The checkpoint is bf16 and the
   loader widens it to f32, so every weight value is already exactly
   representable in bf16 — expanding on the fly would halve the bytes streamed
   BIT-IDENTICALLY on this path. It is not free to claim: the GGUF Q4_K arm
   dequantises to values that are NOT bf16-representable, so the identity holds
   for the safetensors lineage only and the two would have to be distinguished.
2. **A larger batch.** The two CFG rows are the only rows this schedule has, so
   the weight sweep is already amortised twice. Nothing above 2 exists to batch.
3. **The device arm**, which is lever 1 plus `vt::MatmulBT`, and is the row §14.5
   describes.

**The e2e WAV pair is NO LONGER OWED — it was taken and it passed (§16.6b).**
`minimax-music3-gen` on both arms with the same seed wrote `sha
5e81fc133d653560` on all three pairs, six runs and one hash, with identical RMS
and peak. The first attempt died on a linkage artefact; the corrected job ran it
on Thor against the real checkpoint. This closes the one leg §12.3 had that this
row lacked, and it closes it at full scale rather than at unit scale.

**`DepthSequenceEmbeds` is now gate-only.** It is still the documented mirror of
`_generate_depth_codes`'s sequence assembly and is still gated against the
committed goldens and exercised by `test_minimax_music3_quant_real`, but no
production path calls it any more. That is recorded here rather than deleted,
because it is what the incremental schedule is checked to agree with.

**The Thor number is TAKEN (§16.6b): 4.45x on the depth forward and 2.74x on
wall, against the real checkpoint.** §16.6a keeps the voided first attempt and
its cause on the record, because a void with a named cause is evidence and the
number that replaced it does not erase it. What remains open is narrower than
before and is stated as such: **why Thor's 4.45x exceeds x86's 3.50x is a
hypothesis, not a result.** Settling it needs a measured memory bandwidth on
both boxes, and no axis in this row has one. A second open item is smaller and
concrete: 2.65 s of the wall saving sits in `ar.lm_decode_step`, a bucket this
change does not touch, and its cause is unestablished.

**The vocoder is now the largest single term in a run, and no row owns it.** At
53.6 s it is 32.9 % of wall after this change, against 12.0 % before, and it did
not move because nothing here touches it. That is the next thing worth
attacking on this model, and it is named here rather than left for the next
reader to rediscover from a profile.

**No parity claim, again.** SGLang-Omni is still `gateable = no`, and every
reference axis in `docs/BENCHMARKS.md` stays `PENDING`. Everything above is an
internal two-arm number on named hardware.


## 17. The request-key contract, finished (#1315) — a guard that stopped looking

[#925](https://github.com/mudler/vllm.cpp/issues/925) refused `audio_duration_s`
and [#953](https://github.com/mudler/vllm.cpp/issues/953) refused the five keys
SGLang-Omni names. Both landed and both are correct. This section is what they
did not cover, and the first half of it is the same defect they fixed, still
live at `678fc672c` and reachable by an ordinary body.

### 17.1 Scope

[#1315](https://github.com/mudler/vllm.cpp/issues/1315). One function,
`ParseSpeechRequest`
([`src/vllm/entrypoints/openai/speech_api.cpp`](../../src/vllm/entrypoints/openai/speech_api.cpp)),
plus the tests that enter it through the registered route. No engine, no
kernel, no model code, no lifecycle change: the row stays `ACTIVE`, so this
change owes `docs/USAGE.md` (a request-key surface) and nothing in
`docs/STATUS.md`, `docs/BENCHMARKS.md` or `## Now`.

### 17.2 The oracle, and what it says

vLLM registers no `/v1/audio/speech` and no MiniMax-Music3, so the primary is
silent here and the secondary applies: SGLang-Omni, pinned at
`748a0b437e4a8faad44d7bbfd5a0ae55d1fef830`
([`.agents/oracles/sglang-omni.md`](../oracles/sglang-omni.md)). The pin was
asserted against the local checkout's `HEAD` before a line of it was read,
because an oracle whose identity is assumed measures whatever happens to be
checked out.

`_build_tts_params` (`sglang_omni/serve/speech_service.py:737-779`) forwards a
named set of wire keys into the model builder.
`_UNSUPPORTED_TTS_PARAMS` (`sglang_omni/models/minimax_music3/request_builders.py:20-30`)
lists what MiniMax-Music3 cannot honour, and `_validate_tts_contract` (`:71-81`)
raises `"MiniMax Music 3 does not support speech parameters: <names>"` for each.
Two further keys come from the schema itself: `voice` carries
`AliasChoices("voice", "speaker")` (`sglang_omni/serve/protocol.py:337-339`),
and `instructions` is the music CAPTION, which the builder requires non-empty
(`request_builders.py:104-106`).

The oracle is `gateable = no` — cloned and read, never executed here — so this
is a SOURCE reading and it is recorded as one. Nothing in this section claims a
measured comparison against a running SGLang-Omni.

### 17.3 The defect, in two parts

**Part 1 — `extra_params` was a REPLACEMENT, not a second place to look.** The
`const nlohmann::json& extra` binding in `ParseSpeechRequest` selects one of the
two objects. It is at `speech_api.cpp:159-161` in `f06b9e93d`, the merge base of
the change that repairs it. An earlier draft of this section and the `#1315` row
in [`issue-index.md`](../issue-index.md) both cite `:160-163`, which is off by
one line and, at the repaired head, lands on the `voice` and `speed` refusals
instead. This copy is corrected; the index row is append-only and keeps the
slip, so the anchor to trust is this one and the symbol name is what survives
either way.

```cpp
const nlohmann::json& extra =
    (json.contains("extra_params") && json.at("extra_params").is_object())
        ? json.at("extra_params") : json;
```

So a body carrying an `extra_params` object AT ALL — an empty `{}` suffices —
stops the top level being read for every knob and every refusal that resolves
through `extra`. `{"lyrics": …, "extra_params": {"seed": 7}, "audio_duration":
0.1}` drops the duration, leaves `audio_duration_s` at its `0.0` sentinel, and
§4.1's resolver substitutes the family's 60 s
(`minimax_music3_speech.cpp:478-479`). That is #852's ~750x job, behind a 200,
with #925's guard in the tree and unable to fire. The five #953 refusals go
quiet on the same body. The mirror hole is the other side of it: `voice`,
`speed`, `stream`, `stream_format` and `response_format` were read from the top
level only, so nesting any of them dropped it.

**Part 2 — nine keys the oracle names, dropped silently.** The seven of
`_UNSUPPORTED_TTS_PARAMS` that this route can see (`task_type`, `ref_audio`,
`ref_text`, `x_vector_only_mode`, `initial_codec_chunk_frames`, `token_count`,
`duration_tokens`), plus `speaker` and `instructions`. `token_count` and
`duration_tokens` are LENGTH keys, so they repeat #925's cost exactly.

### 17.4 Design

`extra_params` becomes a second place to look, with `extra_params` winning —
the precedence the video route already documents
(`src/vllm/entrypoints/openai/video_api.cpp:216-225`), so the two routes resolve
a knob the same way instead of two ways. Every request key goes through one
`Owner(key)` resolver.

**That sentence was false when this section was first written, and §17.7 records
what the false version cost.** The first implementation routed the KNOBS and the
REFUSALS through `Owner` and left the eight CONTENT keys (`model`, `input`,
`text`, `language`, `lyrics`, `description`, `prompt`, `reference_audio`)
reading the bare `json` handle, while claiming in the code that "there is no
longer a handle on one placement only". There was: `Owner` needs `json` as its
fallback, so the handle cannot go away. What is guaranteed is a CONVENTION that
the code states and the tests pin, not a structural impossibility, and it is
written that way now because the impossibility claim is exactly what let eight
direct reads sit underneath it through an implementation and a first review.

The nine keys are REFUSED by name, each naming what to send instead. `speaker`
points at `voice`'s refusal, `instructions` at `description`, `ref_audio` at
`reference_audio`, and both length keys at `audio_duration` in seconds.

**`instructions` is refused rather than aliased, and that is a decision.**
Upstream HONOURS it as the caption, so an alias would be the closer mirror of
behaviour. TWO reasons hold. AGENTS.md holds that a secondary oracle "never
becomes the mirror source", and SGLang-Omni is the secondary here only because
vLLM registers no `/v1/audio/speech` at all. And `instructions` means
style-and-emotion for a TTS family (`protocol.py:348`) and the caption for this
music one, so a global alias on a SHARED route would bake one family's meaning
into it.

A third reason that the first draft of this section gave, *"one meaning keeps
one name on this route"*, does NOT hold, and the counter-example is four
refusals up in the same function: `prompt` is ACCEPTED as a second spelling of
`description`, with a comment stating the opposite policy. The rule the cases
actually follow is narrower, and it is stated in the code beside the refusal:
an ALIAS is accepted when both spellings carry the same value in the same UNITS
and mean the same thing for every family this route can load.
`prompt`/`description` qualify. `instructions` fails on MEANING. And the
`max_new_tokens` precedent from #953 — which is real, and landed in `c90e3fc02`
— fails on UNITS, 25 Hz frames against seconds, so aliasing it would need a
silent conversion of the one quantity this route has already shipped a 750x
error in.

**The refusal has to name BOTH readings of the key, and now does.**
`instructions` is OpenAI's OWN createSpeech field, for voice style and emotion,
and this route describes itself as "OpenAI's createSpeech, extended with the two
MUSIC inputs" (`api_server.cpp:496-497`). Framing the refusal purely as
SGLang-Omni's caption spelling and redirecting to `description` is right for the
caller who ported an SGLang recipe and wrong for the caller who read OpenAI's:
it moves a VOICE-STYLE string into the music caption, which is the exact
conflation this refusal exists to prevent. The message states the OpenAI reading
first (no registered family exposes a style control, so there is nothing to
send), then upstream's, then the `description` redirect for that second reading
only.

**`language` is deliberately absent from the refused set.** Upstream lists it,
but it is already refused BY NAME one layer down
(`minimax_music3_speech.cpp:456-460`), which is the layer this tree puts
family-specific refusals at, and where a family that HAS a language can still
take it. A family-specific refusal belongs at the family layer, and a future
speech family that takes a language must not have to unpick a parser-level
refusal to get one.

**The first draft justified this with "Moving it up would break IndexTTS-2.5",
and that is not true today.** No registered family reads
`SpeechGenParams::language` at all: `grep -rn '\.language'
src/vllm/model_executor/models/` returns exactly one hit, the MiniMax-Music3
refusal above, and `grep language src/vllm/model_executor/models/indextts2.cpp`
returns nothing. Moving the refusal up would therefore break nothing; it would
convert a SILENT DROP into a refusal. That drop is this row's own defect class
in the family this row did not touch, and it is filed as
[#1337](https://github.com/mudler/vllm.cpp/issues/1337) against
`MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation` rather than
fixed here, because either fix is IndexTTS-2 model code with its own oracle
reading. The argument above is the forward-looking form and is the one that
stands.

**The boundary from #925 is unchanged.** An unknown key is still accepted, so
`extra_params` stays forward-compatible. Only keys the pinned oracle names are
refused, and the negative control in the test file pins that boundary so a later
sweep cannot quietly turn it into "refuse everything".

### 17.5 Gate

Red first, and the red is the point: the parser suite fails 24 assertions and
the server suite 54 before the change, on assertions that name the dropped key.
The content-key repair in §17.7 adds a second red on top of that one, measured
against the merge base `f06b9e93d`.
The failing tests enter through the production entry point —
`ApiServer::handle_audio_speech`, which is what
`server.Post("/v1/audio/speech", …)` calls (`api_server.cpp:1116-1120`) — and one
of them over a real socket, because #925 was a claim about an HTTP request and a
parser test cannot make one. The parser-level cases stay as well; they localize
a failure, which is worth having, and they are not the proof.

Reachability is proven by deleting the parse call site
(`api_server.cpp:504`) in a scratch copy and confirming the focused gate reds.

### 17.6 Risks

A guard that fires on an ordinary request is worse than the drop it replaces, so
every refusal carries a negative control: a body without the key parses, an
unknown key parses, and the honoured knobs keep their values across both
placements. The precedence choice is the one behaviour change a body could
notice, and it only becomes observable when a caller sends the SAME key twice in
two places, which today resolves to whichever object `extra` happened to bind.

### 17.7 The invariant was false, and three refusals were defeatable by nesting

A fresh review of the implementation returned FAIL, and it was right. The repair
itself reproduced exactly: the 750x defect, the fact that `"extra_params":{}`
alone was enough, and every number claimed. What failed was a STRUCTURAL claim
the change rested its durability on, and the live defect that claim was hiding.

**What was false.** `speech_api.cpp` said "Every read and every refusal below
resolves through `Owner` ... a guard that looks at one placement only can no
longer be written, because there is no longer a handle on one placement only".
The handle `json` was still in scope and still used about thirteen times below
that sentence, for `model`, `input`, `text`, `language`, `lyrics`,
`description`, `prompt` and `reference_audio`. None of the eight resolved
through `Owner`.

**What that cost**, measured against the real downstream chain rather than
argued. Three refusals stayed defeatable by moving the key one level down
([#1336](https://github.com/mudler/vllm.cpp/issues/1336)):

| body | before | after |
|---|---|---|
| `{…,"text":"hello"}` | REFUSED (`minimax_music3_speech.cpp:440-446`) | unchanged |
| `{…,"extra_params":{"text":"hello"}}` | 200, `text` silently DROPPED | reaches the family, which refuses it |
| `{…,"language":"en"}` | REFUSED (`:456-460`) | unchanged |
| `{…,"extra_params":{"language":"en"}}` | 200, `language` silently DROPPED | reaches the family, which refuses it |
| `{…,"reference_audio":"data:…"}` | decoded | unchanged |
| `{…,"extra_params":{"reference_audio":"data:…"}}` | 200, clip silently DROPPED | decoded |

The `text` case is the sharpest, because the refusal it bypassed ends with the
words *"rather than having it silently dropped"* and nesting produced precisely
that drop. The `reference_audio` case has a second-order cost a caller could not
have diagnosed: for a family whose `requires_reference_audio()` is true, the
nested clip was dropped and `api_server.cpp:511-516` then answered
`400 "reference_audio … is required"` naming the field the caller had just sent.

**None of this was a regression.** The pre-#1315 code read those keys from the
top level only as well. It was in scope, untested and unfiled, and the false
invariant is what let it survive an implementation and a first review: a claim
that a defect *cannot be written* reads as a reason not to look for it.

**The repair.** The eight content keys route through `Owner`, so the invariant is
true in substance, and the comment now claims a convention the tests pin instead
of an impossibility the code cannot provide. Two smaller findings ride with it:
the `instructions` argument loses the leg `prompt` falsifies (§17.4), and the
`audio_duration` refusal stops promising `> 0` while implementing `>= 0.0`
([#1338](https://github.com/mudler/vllm.cpp/issues/1338)), which had made an
explicit `"audio_duration": 0` resolve to the family's 60 s default with the
message that would have explained it never printed. `docs/FEATURES.md` picks up
the placement change its own trigger owed.

**The lesson worth keeping**, because neither the code nor Git will say it: the
review did not find the drops by reading the diff. It found them by testing the
INVARIANT the diff asserted, and the invariant was the only part of the change
that had no test. A claim about what can no longer be written is a claim, and it
gets mutated like any other.

---

## 18. The vocoder's convolution stops running one dependent add chain per cell (#672, #1334)

§16.7 named this row before it existed: *"The vocoder is now the largest single
term in a run, and no row owns it. At 53.6 s it is 32.9 % of wall after this
change, against 12.0 % before, and it did not move because nothing here touches
it."* This is that row. The issue is
[#1334](https://github.com/mudler/vllm.cpp/issues/1334).

### 18.1 The gap, and the two facts that shape it

`vocoder.decode_window` is **53.6 s**, and #1238's depth A/B measured it at
53.6 s on BOTH legs. It is therefore a FIXED term: it does not scale with the
depth work, it was 12.0 % of wall before that row and 33.2 % after, and it grows
as a share of every future improvement to anything else.

And the device arm is not the answer waiting to be switched on. §15.9 prices it
at **3.552 s CUDA against 2.983 s host** at latent length 20, and §13.10's
per-stage ratios are flat at 0.37-0.40x across a 150x span of work. The arm is
real, reachable, and slower.

### 18.2 Why both arms are slow, and why it is ONE cause

`vt::cpu::Conv1dKernel` (`src/vt/cpu/cpu_conv1d_general.cpp::Conv1dKernel`)
computes each output cell with a SINGLE f64 accumulator swept over
`(ic ascending, k ascending)`:

```c++
for (int64_t t = 0; t < length; ++t) {
  double acc = bp != nullptr ? bp[oc] : 0.0;
  for (int64_t ic = 0; ic < in_per_group; ++ic)
    for (int64_t k = 0; k < kernel; ++k) { ...; acc += ...; }
  on[t] = static_cast<float>(acc);
}
```

For a MiniMax-Music3 residual unit that is `in_per_group * kernel = 384 * 7 =
2688` **strictly dependent** f64 additions per output element. The loop has no
instruction-level parallelism and cannot be vectorised at any width, because
every add waits on the previous one. Its measured rate on one core of the 20-core
Zen 5 is **1.76-1.86 GMAC/s**: 1.86 / 1.82 / 1.84 at ranks 8 / 16 / 32 and 1.7622
at rank 1024 for the depth decoder's loop of the same shape
(`.agents/benchmark-record.md`, the MUSIC3-DEPTH entry), and **1.827 GMAC/s** for
THIS loop at `in_per_group = 384`, `kernel = 7` under GCC 13.3 `-O2
-ffp-contract=off`, pinned, measured by the fresh review of this row on
2026-08-19.

**A correction, because the sentence this section first carried was not
self-consistent** (review finding, 2026-08-19). It read "1.7-2.0 GMAC/s on one
core, ~2.8-3.0 cycles per multiply-accumulate on a 5.0 GHz Zen 5 whose `fadd`
latency is 3", and three of those numbers cannot all be true at once: 2.0 GMAC/s
at 5.0 GHz is **2.5** cycles per MAC rather than 2.8-3.0, and a strictly
dependent chain of latency-3 `fadd`s cannot exceed **5.0 / 3 = 1.67 GMAC/s** at
all, so the top of the quoted band sat above its own stated ceiling.

**The cycles-per-MAC figure is WITHDRAWN rather than corrected, because this box
cannot supply the clock it converts through.** The measurement host is a KVM
guest on a Ryzen 9 9950X3D; `/proc/cpuinfo` reports a nominal 4291.948 MHz,
`/sys/devices/system/cpu/cpu0/cpufreq/` does not exist in the guest, and
`perf_event_paranoid` is 4, so the boost clock during a timed loop is not
observable from inside it and neither is a cycle count. Every cycles-per-MAC
number this section quoted was therefore a conversion through an ASSUMED clock,
which is where the inconsistency came from. What the conversion would give across
the plausible range, stated so the next reader does not redo it: 1.827 GMAC/s is
2.35 cycles per MAC at the reported 4.29 GHz, 2.74 at 5.0, and 3.12 at the "5.7
GHz-class" this record uses elsewhere for the same box.

**What the ceiling implies, which is the part worth keeping.** A dependent chain
of latency-`L` adds cannot beat `clock / L` MACs per second, so the measured
1.76-1.86 GMAC/s bounds `L` at or below ~2.4 cycles at the clock the guest
reports, and clears a latency-3 ceiling at any clock below 5.3 GHz. Either the
add latency on this core is under 3 or the core was boosting above 5.3 GHz, and
this box can distinguish neither. **The mechanism does not rest on that
constant.** What supports it is that breaking the chain — and changing nothing
else about the arithmetic, its order, or its width — is worth **2.16x per core**
(§18.8b). A dependency is the only property that intervention removed.

The CUDA provider loses for a DIFFERENT mechanism, and **the mechanism is an
INFERENCE rather than a finding** — stated that way here because the first
version of this paragraph asserted it flat, and a fresh review was right to
refuse it (2026-08-19).

What is CHECKED, by reading the source: `src/vt/cuda/cuda_conv1d_general.cu`'s
`Conv1dKernelCudaImpl` gives each output cell its own thread and one f64
accumulator, and pins every operation with `__dadd_rn` / `__dmul_rn`. So the
device arm is not latency-bound the way the host loop is — there are millions of
independent chains — and its arithmetic is f64 throughout.

What is MEASURED: §13.10's per-stage device-vs-host ratios are flat at 0.37-0.40x
across a 150x span of work. That rules out a FIXED overhead, which would punish
the smallest stage far more than the largest, and it rules out very little else.

What is INFERRED, and owed: that the residual is f64 THROUGHPUT — that Thor's
consumer Blackwell runs fp64 at a small fraction of its fp32 rate and the kernel
is sitting on that limit. Nothing here measured it. No fp64:fp32 ratio was taken
on the box, no occupancy or pipe-utilisation counter was read, and no CUDA A/B
was run against an f32-accumulate arm. It is the explanation this row offers, not
a result it establishes, and §18.9 carries it as owed.

The two arms' mechanisms are different either way. The property they both turn on
— and the only one this row changes on the host — is that the arithmetic is f64
and arranged as one accumulator per output cell.

### 18.3 The f64 does not have to move for THIS row, and the reason it was kept was wrong

**Superseded in part by `VT-CONV1D-F32-ACC`
([#1474](https://github.com/mudler/vllm.cpp/issues/1474),
[`vt-conv1d-f32-accumulator.md`](vt-conv1d-f32-accumulator.md)), which narrowed
the accumulator to f32.** The paragraph this section used to open with said the
f64 was "what every committed golden for all FOUR consumers was taken with".
That was false — every one of those generators runs torch in f32, so the
goldens came from an f32-accumulating reference and this op was wider than its
own oracle. §13.2 carries the correction and the evidence.

What survives, and is the point of this section, is the part that never
depended on the width: **this row did not need to touch it.** Narrowing was a
separate lever, it re-gated four shipped models, it could not inherit §13.4's
`memcmp` against the pre-op host loop, and it was worth strictly less after this
row than before it — which is exactly why it was taken as its own row with its
own goldens measurement rather than folded in here.

**Because the chain can be broken without touching the width.** Hold one f64
accumulator per output cell over a TILE of output positions, and hoist the
`(ic, k)` sweep OUTSIDE the position loop:

```c++
for (t0 = 0; t0 < length; t0 += TILE) {
  for (i = 0; i < tn; ++i) acc[i] = seed;
  for (ic ascending)
    for (k ascending)
      for (i in the in-range part of the tile) acc[i] += x[...] * w[ic][k];
  for (i = 0; i < tn; ++i) on[t0 + i] = (float)acc[i];
}
```

Fix any single output cell `t = t0 + i` and read the additions it receives, in
order: the bias, then `(ic=0,k=0)`, `(ic=0,k=1)`, ... — the identical sequence of
IEEE-754 double additions of the identical double products, in the identical
order, as the shipped loop. Nothing is reassociated. The additions are
INTERLEAVED across independent cells rather than serialised into one, which is
a scheduling change and not an arithmetic one. **`memcmp` equality survives by
construction, and no tolerance is introduced or widened.**

The same argument is why the zero-padding skip has to be handled by CLAMPING the
tile's position range rather than by testing each position: for a fixed `k` the
in-range `i` form one contiguous interval, so the skipped `(t, ic, k)` triples
are exactly the ones the shipped loop skips.

### 18.4 The compiler has to be given a constant trip count, and that is measured

The restructure alone is not the win. GCC's `-O2` vector cost model is
`very-cheap`, which vectorises only a loop whose trip count is a known multiple
of the vector width, and the accumulate loop's bounds are runtime values. So the
kernel takes a fast path with a CONSTANT trip count when the whole tile is
in range and the stride is 1, and a fixed-width chunked path plus a scalar tail
otherwise.

`Release` is `-O3` and is what CI, `scripts/build-cpu-release.sh` and the
accelerator release build use; `scripts/dgx-bringup.sh` uses `RelWithDebInfo`,
which is `-O2`. The kernel is therefore written to be fast at BOTH, rather than
inheriting whichever one the next measurement happens to use. Measured
difference on the same source: with a runtime-bounded accumulate loop the
speedup is 1.1-1.8x at `-O2` and 5.1-5.4x at `-O3`; with the constant-trip fast
path it is 5.2-5.8x at `-O2` and 5.3-6.5x at `-O3`.

**Every one of those six figures is the KERNEL alone**, taken on `Conv1dKernel`
at the vocoder's stage geometries, and none of them is what the decode window
does. The window is not only this convolution: it also runs
`vt::ConvTranspose1d` (§18.5 prices it far lower), the alias-free activations,
the strided downsamples that keep the shipped gather, and the threadpool and
allocation around all of them. The gap between the two quantities is measured
rather than asserted — the same review that asked for this clause built the
project at `-O2` and timed the window single-threaded, and got **2.56x and
2.67x** where the kernel gives 5.2-5.8x. §18.8a's **1.36-1.44x** is the same
quantity again at `Release` on 14 threads, and §18.8b's **2.16x** is that window
on ONE thread. A kernel number applied to the window overstates the win by
roughly a factor of two, so the two are never quoted interchangeably here.

`stride > 1` keeps the shipped arithmetic path. The MiniMax-Music3 vocoder's
`Conv1d` calls are all stride 1; the strided caller is the alias-free downsample
in `vocoder1d::AliasFreeActivation1d::Apply`, which is depthwise
(`in_per_group == 1`) and whose chain is one tap deep, so it is not the shape
this row is about. That is a decision with a reason, not an omission.

### 18.5 `vt::ConvTranspose1d` needs much less, and the number says why

The scatter already writes `kernel` INDEPENDENT cells per input value, so it has
ILP by construction and the only thing missing is a constant trip count on the
tap loop. At `-O3` the compiler finds it and the op is **~6 % of the chain's
wall**; at `-O2` it does not, and a fixed-width tap chunk recovers 2.7-2.9x on
the three kernels of 8 or more taps.

Both are done, because the four-line chunk is what makes a `RelWithDebInfo`
measurement of this row mean the same thing as a `Release` one. Neither changes
the visit order: the taps of one input value land in `kernel` DISTINCT cells, so
chunking them reorders nothing.

### 18.6 Upstream anchors

vLLM does not implement a DAC Flow-VAE vocoder, and no secondary oracle is
consulted for a NUMERIC question here, because this row changes no number. The
semantics being preserved are `torch.nn.functional.conv1d` /
`conv_transpose1d` as §13's port already mirrors them
(`minimax_music3_vocoder.py:42,44,55,89,98`), and the arithmetic contract being
preserved is this repository's own, stated in `include/vt/ops.h` at
`vt::Conv1d` / `vt::ConvTranspose1d` and gated by
`tests/vt/test_ops_conv1d_general.cpp` and
`tests/vllm/models/test_host_parallel.cpp`. **The oracle for this row is the
shipped kernel itself, carried verbatim into the gate**, which is the same
instrument §12.2 and §13.3 used and for the same reason.

### 18.7 Tests and gates

| gate | what it holds |
|---|---|
| `test_ops_conv1d_general` | the op contract, including the CPU-vs-CUDA `memcmp` arm on a CUDA build |
| `test_host_parallel` | the shipped `vocoder1d` entry points against a VERBATIM copy of the pre-op host loop, bitwise, at five thread counts |
| `test_vocoder1d`, `test_bigvgan` | the two smallest consumers |
| `test_minimax_h3`, `test_ltx2_vae`, `test_minimax_music3_*` | the four consumers' committed goldens |

The shapes the first two do NOT reach today are what this row's new cases add,
because the restructure's whole risk surface is the tile boundary and the
position clamp:

- `padding != 0` — `cpu_conv1d_general.cpp:74` states in its own comment that
  every existing caller passes `padding == 0` and that the skip "is therefore
  unreachable on those shapes and exists for torch parity". A clamp that is
  wrong at the left or right edge is invisible without it.
- a `length` that is not a multiple of the tile, and a `length` BELOW one tile.
- `dilation > 1` combined with a tile boundary.
- the catastrophic-cancellation case §13.5 built, at a length that spans several
  tiles, so an f64 accumulator stored through an f32 cannot hide a reordering —
  **once per SWEEP AXIS, because one case covers one axis and not the other.**

Every one of them is asserted against a verbatim in-test copy of the pre-change
kernel, bitwise.

**The two axes, and why the count is two rather than one** (review finding,
2026-08-19). The sweep this row hoists out of its accumulators is
`(ic ascending, k ascending)`, and reversing EITHER loop is a genuine
reassociation. The cancellation case as first written had teeth along `ic` ONLY,
by construction: it pairs input CHANNELS
(`w[(oc*cin+1)*kernel+k] = w[(oc*cin+0)*kernel+k]` with `x[0] = +2^40` and
`x[1] = -2^40`), and its own teeth self-check reverses `ic` and nothing else.
Measured on this tree: with the kernel's `k` loop reversed and nothing else
changed, `test_ops_conv1d_general` reported 9/375, `test_host_parallel` 8/877,
`test_vocoder1d` 10/58 and `test_bigvgan` 6/65, all `SUCCESS!` and all rc 0 —
binary sha256 `c4bb0e76...` against the baseline `760061c5...`, so the mutation
was in the binary that ran and the green was not a stale artifact.

The hole is OLDER than this row — the pre-change kernel with its `k` sweep
reversed is green too — so it is not a regression this row introduced. It is
still this row's to close, because this row is what turns that order into a
load-bearing guarantee. `tests/vt/test_ops_conv1d_general.cpp` therefore carries
a SECOND cancellation case, "holds its TAP order", which pairs +2^40 against
-2^40 across `k` inside one input channel held constant along its length, and
whose teeth check reverses `k` alone. With that case present the same mutation
fails 1552 of 1576 cells in exactly that case and nowhere else (10 cases / 379
assertions, 1 failed, binary `eb25d992...`), and the restored tree hashes back to
`f84d3a87...`. `SerialConv1d` grew a `reverse_k` parameter beside `reverse_ic`
for it.

**So state the guarantee with its axes attached.** What is gated is that the
order each individual cell sees is unchanged along `ic` AND along `k`. Neither
"nothing held the forward sweep's order" nor "a length at which an f64
accumulator stored through an f32 cannot hide a reordering" was true of both
axes when it was first written here; both were true of `ic` and false of `k`.

### 18.8 Speed evidence — what it must be, before it is taken

`.agents/benchmarking.md` and the recorded defects of §13.10 and §16.6a bind
this row to four things it cannot report without:

1. **An `rc` lease on `thor:gpu0`.** §13.10 is VOID because its arms were taken
   over `ssh` under `$HOME/gpu.lock` while another session held the same box
   through `rc`. Not a caveat — step zero.
2. **Two separately built binaries, in separate source trees and separate build
   directories, whose `sha256` DIFFER.** §16.6a is void because both arms were
   the same binary, and the tell was identical call counts rather than equal
   times.
3. **The checkpoint staged to local disk**, with `SRC_BYTES == DST_BYTES`
   asserted and `findmnt` printed for the path actually read. §15.6 measured
   780 s cold CIFS / 428 s warm against 7.5 s local.
4. **A RANGE of latent lengths.** One point is what made the device arm look
   simply "slower" in §15.9.

### 18.8a The measurement, TAKEN — Jetson Thor, under a lease, two binaries

**`rc` job `da3a2f94-90e3-4e97-b519-9456310673b7` on `thor:gpu0`**, worker
`rc-worker-hqfj4`, `Linux 6.8.12-1021-tegra` aarch64, 14 cores,
`--max-runtime 150m`, 2026-08-19. No `ssh`, no `rc hold`, and no file mutex — the
lease is the whole of the serialisation, which is the thing §13.10 did not have.

Both arms built INSIDE the lease from two clones in `/tmp` (local overlay, not
the CIFS `/workspace`), `CMAKE_BUILD_TYPE=Release` (`-O3`), CPU-only, `ninja -j 8`:

| | before | after |
|---|---|---|
| `src/vt/cpu/cpu_conv1d_general.cpp` sha256 | `6fb15174c1533b93` | `a0e429ace3536479` |
| `vllm_music3_vocoder_conv_ab` sha256 | `d90e3912cd636666` | `41ba78d2b7a8b99e` |

`diff -rq` over `src/` reports that ONE file as the only difference between the
trees, and the harness refuses to time anything when the two binaries hash the
same. That guard exists because §16.6a's first Thor pair was VOID for exactly
this: both arms were the same binary, and the tell was identical call counts,
because equal times are noise where equal binaries are identity.

**Correctness first, on the after arm, before any speed number was read**
(`test cases` / `assertions` / `Status` quoted in full, because `assertions: 0`
is a skip wearing a pass):

| suite | result | rc |
|---|---|---|
| `test_ops_conv1d_general` | 9 cases, 375 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_host_parallel` | 8 cases, 877 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_vocoder1d` | 10 cases, 58 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_bigvgan` | 6 cases, 65 assertions, 0 failed, `SUCCESS!` | 0 |

Three `[SKIP]` lines are printed and are read rather than ignored: this worker
has no CUDA toolkit (`nvcc` MISSING), so the build is CPU-only and every
CPU-vs-CUDA arm in that file, including this row's new one, did NOT run here.
The device provider is unchanged by this row, and §13.4's `memcmp` arm was
measured on a CUDA build; **it has not been re-measured against the tiled host
kernel, and that is named as owed in §18.9 rather than implied by these greens.**

**The sweep: arms ALTERNATED, three rounds, best-of-3 per point, the default
arm** (`VLLM_CPP_VOCODER_DEVICE` unset, so this is the host kernel — see below
for why that matters). `uptime` load average 3.29 before the build, 9.07 before
the sweep, 8.48 after it; 0 other users; the box was NOT idle and both arms ate
the same contention, which is what alternating them is for.

Medians of the three rounds, in seconds:

| latent frames | before | after | ratio |
|---|---|---|---|
| 20 | 5.5688 | 4.0831 | **1.364x** |
| 40 | 11.0535 | 7.8186 | **1.414x** |
| 86 | 23.5149 | 16.6614 | **1.411x** |
| 172 | 47.9201 | 33.6498 | **1.424x** |
| 344 | 97.4463 | 67.7083 | **1.439x** |

The loudest pair is KEPT rather than dropped: 20 frames is the weakest ratio in
the set and it is the one §15.9 priced the device arm at. The spread across a
**17x span of work is 1.364x to 1.439x**, which is the signature of a RATE change
and not of a fixed overhead — the same test §13.10 applied to the device arm, run
in the other direction.

**And the arms are BIT-IDENTICAL at full scale.** The harness prints an FNV-1a
fingerprint of the whole stereo waveform, and across all six arm-rounds every
length produced one value on both arms: `0x7c31c2ea73418503` (20),
`0x35a02aad4c9cb983` (40), `0xc2d5eaf095d1c483` (86), `0x2dc69976150a5903` (172),
`0x95e771d5f0051283` (344). Six processes, two binaries, one answer per length.

**What this number is NOT.** It is the decode window's COMPUTATION at the shipped
geometry with synthetic weights, driven through `VocoderDecode` — the call
`Music3DecodeChunks` brackets as `vocoder.decode_window`
(`minimax_music3_speech.cpp:397-398`). It is not an end-to-end synthesis, no
checkpoint was read, and therefore §18.8's staging assertion
(`SRC_BYTES == DST_BYTES`, `findmnt` on the path read) is NOT APPLICABLE here
rather than skipped: there is no path to assert. The e2e pair on the real
checkpoint is owed (§18.9).

**And 53.6 s is the CUDA arm, not this one.** §15.2's profile ran
`VLLM_CPP_VOCODER_DEVICE=cuda`, so the 53.6 s / 33.2 % that defined this row is
the DEVICE arm's decode window. The default is `cpu` (§13.6) and that is the arm
this row moves. Applying 1.42x to the 53.6 s figure would be comparing two
different arms, so no such projection is made here.

### 18.8b The KERNEL is 2.16x and the THREADPOOL gives back a third of it

The obvious reading of §18.8a is that aarch64 simply vectorises worse than
x86 and 1.42x is what the kernel is worth here. **That is measured, and it is
wrong.**

Second lease, **`rc` job `5b98f95e-a37b-4fa2-8ee9-81959caa828f` on `thor:gpu0`,
`--max-runtime 45m`**, a fresh container, both arms rebuilt from the same two
refs into two new binaries (`a3b14f2995...` before, `5b894d6b67...` after — both
different from the first run's pair, so this is an independent build as well as
an independent run). One latent length, 20 frames, because that is where §18.8a's
ratio is weakest and therefore where the claim is hardest:

| threads | before | after | ratio |
|---|---|---|---|
| **1** (`VLLM_CPP_CPU_THREADS=1`) | 37.0508 s | 17.1751 s | **2.157x** |
| 14 (default), same container, same binaries | 5.4845 s | 4.0182 s | **1.365x** |

The 14-thread control reproduces §18.8a's 1.364x to three digits, on a different
container and a different pair of binaries, which is what makes the single-thread
figure comparable to it rather than merely adjacent.

**Read the scaling instead of the ratio and the cause is plain:**

| arm | 1 -> 14 threads | of a possible 14x |
|---|---|---|
| before | 6.76x | 48 % |
| after | **4.27x** | **31 %** |

Neither arm scales, and the FASTER one scales WORSE. That is the signature of a
shared resource: the tiled kernel needs the same bytes in less time, so it
saturates whatever is shared sooner and gives back a third of its per-core win.
The kernel is worth **2.16x on this box**; the threadpool returns 1.37x of it.

Stated with its limits. This does not identify the resource — no bandwidth
counter was read, and none is available on this worker — so "memory bandwidth"
remains the leading candidate and not a finding. What it does establish is where
the next lever is, and it is not the kernel: **it is the vocoder's parallel
decomposition**, which partitions OUTPUT CHANNELS and therefore has every thread
sweep the whole input tensor. §18.9 carries it as owed rather than as a ceiling.

And it removes one hypothesis §18.8a left open: the aarch64 codegen is NOT the
whole story, because per core the same source is worth 2.16x here against ~5x on
AVX-512 x86 — a gap the 4x narrower f64 vector explains without needing anything
else.

### 18.9 What is OWED after this row, named rather than left to a profile

- **The f32-accumulate variant** (§13.10 step 3) is **DISCHARGED**, by
  `VT-CONV1D-F32-ACC` ([#1474](https://github.com/mudler/vllm.cpp/issues/1474),
  [`vt-conv1d-f32-accumulator.md`](vt-conv1d-f32-accumulator.md)). It did what
  this bullet asked: its own gate against each of the four consumers' goldens,
  measured per arm before and after, and it did not inherit §13.4's `memcmp`
  against the pre-op host loop — it replaced that standing with a width gate
  against torch's own answer and said so.
- **The device arm's staging** (§13.6's owed list: device-resident weights, one
  persistent queue, a chain that stays on the device between stages) is
  untouched. This row does not make the device arm win, and after it the host
  bar the device arm has to clear is several times higher.
- **`stride > 1`** keeps the shipped arithmetic path (§18.4).
- **`ltx2_audio_vae.cpp:75`'s own 2-D host convolution loop** still routes
  through no op at all (#1114), so it is not reached by anything here.
- **The CUDA arm's f64-THROUGHPUT explanation is unmeasured** (§18.2). The flat
  0.37-0.40x ratio rules out a fixed overhead; it does not establish what the
  residual IS. Three instruments would, none of them run here: an fp64:fp32 rate
  probe on `thor:gpu0`, a pipe-utilisation or occupancy counter over
  `Conv1dKernelCudaImpl`, and a CUDA A/B against an f32-accumulate arm — which
  the f32 variant in the first bullet would supply for free. Until one of them
  exists the sentence in §18.2 is an inference, and this row does not act on it.
  Deliberately NOT settled in flow: it needs a `thor:gpu0` lease and this row's
  measurement budget was spent on the host arm it actually moves.
- **The CUDA-vs-CPU `memcmp` arm is RE-MEASURED and it holds — §20.5.** An e2e
  pair on `thor:gpu0` under a lease, on the real checkpoint, wrote ONE WAV
  sha256 across both vocoder arms at 344 latents (six runs per code arm) and one
  across both at ~689 x 4, so the device provider is byte-identical to the tiled
  host kernel in the finished audio. The paragraph below is retained because it
  states why the leg was open and what it needed; the need is met by a CUDA build
  inside the lease (`apt` installs `cuda-nvcc-13-0`, §20.2).
- **The original statement, kept for provenance.** The CUDA-vs-CPU `memcmp` arm
  had NOT been re-measured against the tiled kernel. The Thor worker carries `gcc`, `g++`, `cmake`, `ninja` and `python3`
  but **no `nvcc`** (probed 2026-08-19, which also CORRECTS §13.10's "no compiler
  and no toolchain at all"), so the arms above are CPU-only and every
  CPU-vs-CUDA case printed `[SKIP]`. The argument that it must still hold is
  §18.3's — the host arm's per-cell order is unchanged, and §13.4's device kernel
  was written against that order — but an argument is not a measurement. It needs
  a build on a box with a CUDA toolkit.
- **The e2e pair on the real checkpoint is TAKEN — §20.4, §20.5.** It carries the
  staging assertion and the WAV identity check in the shape §16.6b took, and it
  reports something §18.8a could not have: the untiled host arm costs **54.091 s**
  at 344 latents through the real request against this bench's **97.4463 s** at
  the same length, a **1.80x disagreement between the two instruments that is
  unexplained**. The consequence binds this row's headline figure: **1.364x-1.439x
  must not be multiplied onto an e2e bucket.**
- **The vocoder's PARALLEL DECOMPOSITION is now the lever, and §18.8b measured
  it rather than guessing.** The kernel is worth **2.16x per core** on Thor and
  the threadpool returns **1.37x**, because both arms scale badly (6.76x and
  4.27x of a possible 14x) and the faster arm scales worse. `ForOutputRows`
  partitions OUTPUT CHANNELS, so every one of the 14 threads sweeps the whole
  input tensor for its own slice of channels; a decomposition that also splits
  the TIME axis would give the threads disjoint input windows. Not attempted
  here, and not a ceiling: what it needs first is a bandwidth counter, and this
  worker has none.

### 18.10 Stop conditions

Stop and report rather than widen scope if: the byte-exactness gates cannot be
made green without a tolerance; `thor:gpu0` is unhealthy or the lease cannot be
taken; or the measured host win does not survive the threadpool, which would
mean the parallel arm is bound by something this row did not measure.
---

## 19. The depth decoder reaches the device (#672, [#1309](https://github.com/mudler/vllm.cpp/issues/1309)) — §11.4's LAST owed device row, and the dtype it was blocked on

§11.4 named three device rows. The vocoder closed in §13, the DiT in §14, and
§14.5 left this one blocked on a dtype rather than on the work. §16 then removed
the arithmetic nobody read, which made this row **worth 4.4x less than when it
was written** and did not make it unnecessary: on `thor:gpu0` the depth forward
is still **48.4 % of a run**.

This section settles the dtype and describes the arm. The row is
`MUSIC3-DEPTH-DEVICE`.

### 19.1 The gap, measured, on the head this row starts from

`main` at `678fc672c`, `thor:gpu0` sm_110, `--device 1 --duration 4 --steps 4`,
100 frames, checkpoint staged to local disk (so §15.6's cold-CIFS load is not
inside these figures):

| bucket | seconds | share of wall |
|---|---|---|
| `ar.depth_forward` | 78.1 | **48.4 %** |
| `vocoder.decode_window` | 53.6 | 33.2 % |
| `ar.lm_decode_step` | 12.3 | 7.6 % |
| `denoise.dit_device` | 5.1 | 3.1 % |

A **0.646 B** decoder costs **6.3x** the **8.6 B** language model beside it. The
sizes are not the reason; which processor each runs on is. §16.4 measured the
host kernel at ~4 bytes of f32 weight traffic per multiply-accumulate — 2.28 GB
per `DepthDecoderAppend` call, 8 calls a frame — and §16.6b measured 96.93 ms
per call, i.e. **~23.5 GB/s achieved** across Thor's 14 cores.

### 19.2 THE DTYPE — settled, and the oracle settles it by declaring nothing

§14.5 blocked this row because routing an `ArCompute::kBFloat16` decoder through
an **f32** `vt::MatmulBT` would silently drop the rounding every gated number was
taken with. That reasoning is correct and its conclusion has expired, because
the row does not need an f32 `vt::MatmulBT`. It needs a **bf16** one, and
`vt::MatmulBT` has been one all along:

> `a/b bf16 (or f32), out f32 or bf16, contiguous, same device` — same f32
> accumulation and dtype contract as `Matmul`
> ([`include/vt/ops.h:1281-1283`](../../include/vt/ops.h))

The CUDA provider states the same contract as a refusal by name
(`src/vt/cuda/cuda_matmul.cu:296-300`): `supported: (bf16,bf16)->f32|bf16,
(f32,f32)->f32|bf16`. **Mixed f32-activation × bf16-weight is rejected**, so the
arm is bf16 on both operands or it is not bf16 at all.

**What the oracle says.** `MiniMaxMusic3RVQDepthDecoder` declares **no dtype**.
Its `@register_to_config __init__` takes no `dtype` parameter and constructs every
submodule at torch's ambient default
(`diffusers` @ `c6da9936`,
`models/transformers/minimax_music3_rvq_depth_decoder.py:101-125`). There is **no
`torch.float32` literal and no `.float()` call anywhere in the file**. The single
cast in the module is a **down**-cast that re-aligns the attention output to the
query dtype:

```python
51        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
```

So the dtype is imposed from outside, by `load_components(dtype=...)`, and
[`tools/oracle/music3_oracle.py:91-95,103-108`](../../tools/oracle/music3_oracle.py)
resolves `rvq_depth_decoder: torch.bfloat16` under **both** its policies —
`ON_DISK_DTYPES` and `REFERENCE_DTYPES` differ only in `condition_encoder`. That
is the invariant §2.1 already records at :176 and :179-182:
`dtype(language_model) == dtype(rvq_depth_decoder) == dtype(condition_encoder)`,
bf16 AR half over fp32 acoustic half.

**Where fp32 legitimately appears, and why it is not this row's.** The depth loop
promotes to fp32 exactly twice, both **after** the decoder and both on logits:
the CFG mix at `encoders.py:134` (`logits[:1].float()`) and the sampler at
`encoders.py:95` (`torch.nan_to_num(logits.float(), ...)`). Both already run on
the host in `Music3DepthStage`, and this row does not move them.

**The decision.** Weights bf16, activations bf16, every op boundary storing bf16,
accumulation f32. That is `vt::MatmulBT`'s contract unmodified, and it is what a
bf16 torch module on a GPU does.

#### 17.2a The finding this makes visible, which no gate in this tree can see

The host arm holds the depth decoder's weights in `std::vector<float>`. The
checkpoint is bf16 and the loader widens it, so every stored value is
bf16-exact: the numbers are right, and the goldens, the token gates and §16.6b's
WAV hash all pass **while the path moves twice the bytes the oracle moves**.

That is `AGENTS.md`'s "a token gate cannot detect a dtype that is too wide", in
this tree, on the stage that is half the run. §16.7 records the same fact from
the other side as lever 1. It is written here as a **finding**, not as a defect
introduced by this row: the host arm's f32 containers are what W2/W3's
reduction-order gates were taken against, and this row does not narrow them. The
device arm is the narrow one.

Per [`.agents/porting.md`](../porting.md) "Mirror the memory format, not just the
math", the four questions and their answers for this path:

| ask | answer, and where upstream answers it |
|---|---|
| what dtype does the linear OUTPUT? | bf16 — `nn.Linear` at module dtype, no cast (`minimax_music3_rvq_depth_decoder.py:63-66,114,124`) |
| what dtype do the intermediate activation buffers carry? | bf16 — read the consumer: `:51` casts the attention output back **to** `query.dtype` |
| is a projection one physical GEMM or several? | several upstream (`to_q`/`to_k`/`to_v`, `gate_proj`/`up_proj`); **merged here**, see §19.3 |
| what `kv_cache_dtype` is resolved? | not applicable — the depth cache is this decoder's own 16-position table, not a paged KV cache; it is bf16 with its activations |

### 19.3 The design — no new kernel, and two merges the seam asks for

Every op already exists with a **CPU and a CUDA provider**, so this row composes
a forward rather than adding a kernel. That is §14.2's shape and it is deliberate.

| host reference (`minimax_music3_ar.cpp`) | shared op |
|---|---|
| `RmsNorm` | `vt::RmsNorm` |
| `LinearNoBias` | `vt::MatmulBT` |
| `CausalAttentionStep` | `vt::AttentionCross`, `bias = nullptr` |
| `silu(gate) * up` | `vt::SiluAndMul` |
| residual adds | `vt::Add` |

**The attention is causal upstream and `vt::AttentionCross` is the right op
anyway, for a reason that is an identity rather than a shortcut.**
`DepthDecoderAppend` presents **one** query row against a cache of `seq`
positions that are all at or before it. A causal mask over a single query at the
last position masks nothing. So the non-causal op computes upstream's causal
result exactly, and `vt::Attention` — which would apply a mask keyed on a query
index this call does not have — is the wrong one. Upstream's own
`dispatch_attention_fn(..., is_causal=True)` (`:43-50`) runs whole sequences;
ours runs one row, and that is §16.3's identity, already gated bitwise.

**Two merges, because AGENTS.md routes mergeable projections through the merged
seam and because they cost nothing numerically.** Each output element remains its
own dot product, so merging changes no reduction:

* `to_q | to_k | to_v` stage as one `[3H, H]` weight; one `vt::MatmulBT` produces
  `[batch, 3H]`, split by re-view (`vt::QkvSplit` is available but a free
  `MakeTensor` at an offset is what the DiT arm uses and needs no copy).
* `gate_proj | up_proj` stage as one `[2I, H]` weight. This is the merge that
  `vt::SiluAndMul` **requires**: it computes `silu(x[:, :D]) * x[:, D:]` over one
  `[T, 2D]` buffer, and upstream's `silu(gate_proj(x)) * up_proj(x)` puts gate
  first. **No half swap** — unlike §14.3's DiT, where the assignment was the
  opposite way round and needed a stage-time exchange.

The weight sweeps per frame fall from 8 (decoder) + 7 (projection) of f32 to the
same counts at **half the bytes**, and the sweep itself moves from 14 host cores
to the device.

**What does NOT move.** The audio heads, the CFG mix, the top-k draw, the
feedback embedding and the projection of the fed-back row stay on the host in
this row, exactly as `Music3DepthStage` has them. The heads are seven
`[1024, 4096]` GEMVs a frame against the decoder's 8 sweeps of 570 M
multiply-accumulates; §15's profile puts `ar.depth_projection` at 1.307 s and
the forward at 78.316 s. Moving them is owed, not skipped silently: §19.7.

### 19.4 Correctness — bitwise is NOT achievable, said before the code, with the tolerance and its reason

**This arm cannot be bit-identical to the host arm, and no gate here will claim
it.** §16 could claim it because a causal identity is exact. This row changes the
machine. Three independent reasons, each sufficient on its own:

1. **The accumulator narrows.** The host keeps a sequential `double` per output
   element (`minimax_music3_ar.cpp::LinearNoBias`); `vt::MatmulBT` accumulates in
   **f32** (`CUBLAS_COMPUTE_32F`). Over `in_dim` 4096 and 6144 that is a real
   difference, not a formality.
2. **The reduction re-associates.** cuBLASLt splits K by an algorithm this row
   does not choose, and `vt::AttentionCross`'s CUDA kernel uses an **online
   softmax recurrence** where the host and the CPU provider use a three-pass
   max/sum/weighted-sum. The two are not bit-identical to each other either.
3. **The two arms normalize against different references, and both are right.**
   The Music3 **host** arm's own RmsNorm (`minimax_music3_ar.cpp`) mirrors the
   diffusers module this model *is*: `normalization.py::RMSNorm.forward` casts
   back to the weight dtype at `:560` and then multiplies at `:561`, so it rounds
   **twice**, and the decoder constructs exactly that class — `class RMSNorm` at
   `:510`, built at `minimax_music3_rvq_depth_decoder.py:78,80,122`. (`:600-606`
   is `GlobalResponseNorm` and `:590-597` is `MochiRMSNorm`; neither is on this
   path. The earlier `:600-606` citation in this spec was wrong by ~46 lines.)
   The **device** arm routes through the shared `vt::RmsNorm`, which mirrors
   vLLM, and vLLM keeps f32 across the weight multiply and rounds **once**.

   **`vt::RmsNorm` does not diverge from its reference, and an earlier revision of
   this section said it did.** Verified directly at the parity pin `555967922`:
   `csrc/cpu/layernorm.cpp` computes `fp32_out = fp32_x * fp32_s_variance *
   fp32_w` and narrows once at `scalar_vec_t out(fp32_out)`;
   `csrc/libtorch_stable/layernorm_kernels.cu:93` computes
   `static_cast<scalar_t>(x * s_variance * w)`. Upstream landed the
   weight-dtype multiply as vllm#42379 and **reverted** it as vllm#46070, which is
   an ancestor of the pin. `AGENTS.md` makes vLLM the only reference wherever it
   implements the behaviour and it implements RMSNorm, so diffusers was never the
   mirror source for the shared op — only for the model.

   What follows is that this term of the band will **not** go away, because
   neither side is defective. Which rounding is right *for this model* is a
   question the diffusers oracle settles through `test_minimax_music3_ar_real`,
   and §19.6 records that as owed.

**The gate is therefore a tolerance, and the tolerance is derived rather than
chosen.** The proposal, to be replaced by the measured value before the row is
`DONE`:

* the reference is the **host arm at `ArCompute::kBFloat16`**, not an f32 forward,
  because the host arm is what every committed number was taken with;
* the bound is stated in **bf16 ULPs of the reference value**, not in absolute
  units, because the activations span orders of magnitude across four layers;
* the proposed acceptance is **max 2 bf16 ULP** with a **mean below 0.5 ULP**
  over every compared value, which is the band §5's own matched control already
  established for this decoder (the header records ~1.3 bf16 ULP average between
  an f32 and a bf16 forward of the same weights);
* the gate asserts its own **teeth**: that the compared reference values are
  non-zero and that a deliberately wrong arm exceeds the bound. A tolerance
  nothing can fail is not a gate.

**A tolerance gate cannot see a dropped stage, so it is not the only gate.** The
composition is held by the same instrument §16.5 had to add: the production
`Music3DepthStage` is driven end to end and its **drawn codes and draw count**
are compared, which a numeric tolerance would not notice.

### 19.4a The tolerance was MEASURED, and it refuted §19.4's own ordering

§19.4 named three sources and led with the accumulator. **The measurement puts
that last and it is corrected here rather than quietly left standing.**

`test_minimax_music3_ar`, 8 heads of 8, three layers, the full 8-position
schedule, bf16-exact pseudo-random weights, on a **CPU `vt::Queue`**:

| arm | worst | mean | composed stage, worst |
|---|---|---|---|
| as shipped | **110 bf16 ULP** | **2.095** | **255 bf16 ULP** |
| host's two intermediate roundings COLLAPSED to the seam's shape | 8 ULP | 0.0596 | **0 — bit-identical** |

The second row is an attribution mutation, applied to the **host** reference and
restored `sha256`-verified. It removes exactly two roundings:

1. The **host** arm's cast back to the weight dtype **before** the affine
   multiply — `Store(Store(normed) * w)` becomes `Store(normed * w)`. That is
   `normalization.py::RMSNorm.forward:559-561` — the `.to(self.weight.dtype)` at
   `:560` and the `* self.weight` at `:561` — which the host arm mirrors on
   purpose because it mirrors the diffusers module, and which `vt::RmsNorm` does
   not do because it mirrors vLLM, where the multiply is f32 and the narrowing is
   single (§19.4 reason 3, verified at the pin). **Neither side is defective**,
   so this term does not close.
2. SiLU's own store before the `* up` multiply — `Store(Store(silu) * up)`
   becomes `Store(silu * up)`. torch computes `F.silu(gate)` into a **bf16
   tensor** and then multiplies, so upstream rounds there too; `vt::SiluAndMul`
   computes the whole expression in f32 and rounds once.

**So ~97 % of the mean deviation is rounding POLARITY inside two shared ops, not
the accumulator, not the reduction order and not the dtype.** The accumulator and
reduction-order terms §19.4 led with are the 0.0596 ULP remainder, which is the
order of magnitude first principles predict for f32-vs-f64 accumulation over 64
terms. §19.4's reason (3) was right and was ranked third; reasons (1) and (2) are
real and are nearly invisible beside it.

**The counterfactual is the strongest statement this row has: with the roundings
aligned, the composed device stage is BIT-IDENTICAL to the host arm over 448
values.** The port's algebra — the merged `[3H, H]` and `[2I, H]` layouts, the
cache indexing, the batch-2 sequencing, the attention identity — is therefore
exactly right, and the entire remaining difference is rounding polarity in two
shared ops — one of which (`vt::SiluAndMul`) is a genuine seam gap and one of
which (`vt::RmsNorm`) is two references legitimately disagreeing. See §19.4
reason 3, which corrects an earlier claim in this spec that both were seam gaps.

#### What this changes about the row, said plainly

**The blocker was never the dtype, and it is not the dtype now.** §14.5 named the
dtype; §19.2 settled it and the settlement stands. What the measurement exposes
is a different obstacle, and **it is half the size this section first claimed**.

**The `vt::RmsNorm` half is not a seam gap at all.** §19.4 reason 3 carries the
verification: vLLM's own RMSNorm multiplies in f32 and narrows once, on both the
CPU and the CUDA path at the parity pin, and upstream reverted the weight-dtype
variant. `vt::RmsNorm` therefore mirrors its reference exactly. The two arms
differ because the *host* arm mirrors the diffusers module and the *device* arm
mirrors vLLM, and both are correct against the reference each answers to. That
term stays in the band permanently, and no seam extension removes it.

**The `vt::SiluAndMul` half is real.** `F.silu(gate)` produces a **bf16 tensor**
that is then multiplied, where `vt::SiluAndMul` computes the whole expression in
f32 and rounds once. It cannot be worked around at the call site, because `vt`
has no elementwise or row-broadcast multiply — `MulScalar` takes a scalar and
`MulColVecF32` is an f32 in-place column scale — so expressing
`Store(silu) * up` needs a seam extension with CPU and CUDA providers and its own
gates. That is [#1322](https://github.com/mudler/vllm.cpp/issues/1322)'s
surviving half, it is its own row, and it is NOT taken here.

**What is consequently NOT claimed.** No parity claim, and no speed number. The
row's numeric gate bounds an attributed divergence; it does not prove agreement
with the reference. The decisive gate is `test_minimax_music3_ar_real` — the
full-scale bf16 companion against the committed oracle goldens — run with the
device arm, and it needs the 28.5 GB checkpoint. It has NOT been run, and until
it has, `AGENTS.md`'s "establish the declared token-exact gate before you accept
a performance result" forbids quoting a Thor A/B for this arm. The A/B is
therefore **blocked on correctness, not on the box**, and that is why this row
lands without one.

The stake is measurable rather than speculative: the header of
`minimax_music3_ar.h` records that an **fp32** depth forward left 448 450 of
716 800 committed golden values beyond one bf16 ULP, at mean absolute error
2.65e-03. A seam that keeps f32 through two roundings per layer is a step in that
direction, and whether it lands inside the full-scale gate's tolerance is exactly
what has not been measured.

#### One instrument defect, found inside this row, and it printed a clean green

The mutation runner restored the tree with `tar x`, which sets mtimes **from the
archive** — i.e. from before the mutation. The restored source was therefore
OLDER than the object ninja had built from the mutated one, **ninja skipped the
rebuild, and the next run reported 35/35 cases and 508/508 assertions SUCCESS
while executing the MUTATED binary**. `git diff --name-only` could not see it
either, because after restoration the file matches `HEAD` byte for byte and a
`sha256sum` of the source says — correctly, and uselessly — that the source is
pristine.

That is `.agents/verification.md`'s "a copied build directory rebuilds the
original sources" arriving from the other side, and it is the same shape as
§16.6a: **whenever an artifact is required to have changed, assert that it
changed rather than that its inputs did.** The runner now restores with `tar xm`,
touches every source, and rebuilds before declaring the mutation over. The
numbers in the table above were re-taken on a forced rebuild.

### 19.4b The tolerance was fitted to ONE seed, and a review falsified it

§19.4a placed the bound from a single draw of the reference weights. A fresh
review changed **nothing but the RNG seed** — same distribution, same geometry,
no defect — and the shipped `mean <= 4.0` **reds on three of six equally valid
draws**. A tolerance a redraw can fail is measuring the draw, not the arm.

The correct arm, `test_minimax_music3_ar`, 8 heads of 8, three layers, the full
8-position schedule, batch 2, on a CPU `vt::Queue`:

| seed | worst | mean | median | under the old bound |
|---|---:|---:|---:|---|
| `0x9E3779B9` | 110 | 2.095 | 1 | pass (the seed it was fitted to) |
| `0x2468ACE0` | 435 | 3.804 | 1 | pass |
| `0x00000001` | 1110 | 5.755 | 1 | **FAIL** |
| `0x13579BDF` | 3663 | 7.154 | 1 | **FAIL** |
| `0xDEADBEEF` | 7340 | 9.904 | 1 | **FAIL** |
| `0x51ED2701` | 939 | 3.103 | 1 | pass |

**`worst` cannot discriminate and is no longer gated.** A *correct* arm reads
worst 7340 at `0xDEADBEEF`; the gate/up half swap reads 6641 and the wrong
attention scale 6865. Any `worst` bound loose enough to admit a correct
implementation admits two structural defects, so a correct arm is *worse* on that
axis than two defects. It is reported, with a canary at 1e6 for a non-finite
value.

**The metric itself had a defect, found by the sixth seed.** A reference value of
exactly zero has no ULP — bf16's spacing at zero is the denormal floor — so
`|got| / Bf16Ulp(0)` reads ~1e35 for an absolute difference of 1e-5.
`0x51ED2701` draws one such value in 1024 and read mean **8.8e32 with the arm
correct**. Zero references are now measured **absolutely** in their own bucket
(worst observed 8.30e-05 against a 1e-2 bound), and the two counts are asserted
to **sum**, so a defect cannot hide by growing the un-gated bucket.

#### The battery, and why the median is the primary gate

Five structural mutations, each run over all six seeds, on a clean tree, with the
compiler exit status, `git diff --stat` and the **binary** `sha256` printed for
every run and the source restored and verified after each:

| mutation | mean range | median range | verdict |
|---|---:|---:|---|
| wrong attention scale (`1/sqrt(hidden)`) | 19.6 – 71.3 | 4 – 4 | RED 2 cases / 13 assertions |
| gate/up half swap | 52.5 – 158.8 | 12 – 15 | RED 2 / 14 |
| K/V cache row collision (`l*batch`) | 166.4 – 376.1 | 3 – 4 | RED 2 / 14 |
| dropped position embedding | 357.8 – 1487 | 76 – 90 | RED 2 / 15 |
| `q\|k\|v` merge order swapped | 400.0 – 1731 | 104 – 131 | RED 2 / 15 |

**The median is exactly 1 bf16 ULP at every seed for the correct arm** — a
constant, not a distribution — because the deviation this arm carries is one
rounding-polarity tick per element and the seed only changes the tail. Every
defect is at least 3. The bound is **2**, and it cannot be moved by a redraw,
which is precisely the property the single-seed bound lacked.

**The mean stays gated at 15, and its margin is honestly thin.** It is
tail-sensitive, so it catches a *sparse* defect that leaves the middle of the
distribution alone and the median would miss. Its window is
(9.904, 19.6): the correct arm's worst draw against the tightest defect, which is
the wrong attention scale at `0xDEADBEEF` — the same seed that produces the
correct arm's own worst mean. 15 is 1.51x above every correct draw measured and
1.31x below every defect draw measured. A seventh seed drawing a correct mean
above 15 is possible in a way that one drawing a median above 2 is not, and if
that happens the median is the gate that still holds.

**This bound does not become obsolete.** §19.4 reason 3 records the verification:
the `vt::RmsNorm` term is two references legitimately disagreeing, not a defect
awaiting repair, so it does not close. #1322's surviving `vt::SiluAndMul` half
may narrow the band later; it is not landed, this row does not wait for it, and
nothing here is deferred against it.

### 19.5 Reachability — the #1131 trap, named before it is fallen into

[#1131](https://github.com/mudler/vllm.cpp/issues/1131) is this exact failure for
the DiT arm: its kernels and staging are gated, its **production switch** is not,
and setting `on_device = false` leaves every suite green. The reason it happened
is structural rather than careless — the CUDA arm needs a GPU that CI does not
have, so the natural gate is a unit test of the forward, and a unit test of the
forward is exactly what cannot see the switch.

**The way out is that every op this arm uses has a CPU provider.** So the device
forward runs on a `vt::Queue` whose device is `kCPU`, with no GPU and no
checkpoint, and CI can drive it **through the production call site**:

1. `Music3DepthStage` takes a `Music3DepthDeviceArm` (queue + staged weights).
   Non-null selects `DepthDecoderAppendDevice`; null keeps the host loop.
2. `MiniMaxMusic3SpeechEngine` stages the arm through
   `Music3SelectDepthArm`, on the **default** `--speech-device 1` — the same
   switch `StageMusic3DitWeights` already rides.

   **This was written as an `if` in the engine, and a review proved that was the
   trap it claims to avoid.** Deleting the whole
   `if (queue_.device.type != kCPU) { … }` block — the only thing
   `--speech-device 1` reaches — left `test_minimax_music3_ar` 35/35 · 508/508
   and `test_minimax_music3_speech` 9/9 · 223/223 SUCCESS. That is #1131's shape
   reproduced by the change that names #1131 as its reason for existing, and it
   is structural: on a CPU-only runner that condition can never be true, so no
   branch written at that line is reachable by any gate CI owns.

   The rule now lives in `Music3SelectDepthArm`, which executes on **both** sides
   of the condition and is therefore drivable from a CPU gate. Three outcomes,
   and the third is the defect: a CPU queue stages nothing and returns a
   disengaged arm; any other device stages and returns an engaged one, or
   `StageMusic3DepthWeights` refuses by name because this build has no provider
   for it; a non-CPU queue **quietly** taking the host loop is what must never
   happen. Gutting the selector reds 1 case / 6 assertions.
3. The CI gate constructs the arm on a **CPU queue** and drives
   `Music3DepthStage`, asserting (a) agreement with the host arm inside §19.4's
   band, (b) identical drawn codes, and (c) that the device path was **taken** —
   a counter, not an inference from the numbers, because the two arms agree
   numerically by design. That last assertion is the one #1131 says is missing,
   quoted in its own words: the gate "must assert the device path was TAKEN
   (invocation count or resident dtype), not merely that outputs agree".
4. The reachability mutation deletes the production selection in a scratch copy
   and the focused gate must go **RED**. It does: gutting `Music3SelectDepthArm`
   reds 1 case / 6 assertions.

**What is still NOT gated, stated rather than implied.** Deleting the engine's
two-line *call* to `Music3SelectDepthArm` leaves both suites green
(`test_minimax_music3_ar` 37/37 · 640/640, `test_minimax_music3_speech`
9/9 · 223/223), with the mutated `test_minimax_music3_speech` binary `sha256`
`77334986…` against the baseline `6bd42129…` so the run is not a stale-binary
artefact. Nothing here changes that, and nothing can on a CPU-only runner: the
engine needs the 28.5 GB checkpoint and a real device. The rule it calls is now
gated, its two components are gated, and the residual is the call itself. It is
owned by `MUSIC3-DEPTH-DEVICE` and tracked by
[#1131](https://github.com/mudler/vllm.cpp/issues/1131), listed under §19.7, and
it is the same residual the DiT block one screen below still carries in full.

### 19.6 Gates and evidence this row owes

| leg | where | what it proves |
|---|---|---|
| numeric agreement, CPU queue | `tests/vllm/models/test_minimax_music3_ar.cpp` | the composition, at reduced geometry, in CI, no GPU |
| composition + drawn codes | same | a dropped stage a tolerance cannot see |
| device path TAKEN | same | the #1131 hole |
| numeric agreement, CUDA | `thor:gpu0` under `rc` | the kernels the CPU provider does not exercise |
| stage A/B + wall | `thor:gpu0` under `rc` | the reason the row exists |
| WAV identity | `thor:gpu0` under `rc` | that it is inaudible in the product — **within** §19.4's band, so a HASH pair is not available here and an RMS/peak/max-abs comparison replaces it |

**The A/B's own preconditions, which §16.6a paid for.** Two source trees, two
build dirs, both binaries' `sha256` printed and a hard failure when they are
**equal**; the checkpoint staged to local disk with `SRC_BYTES == DST_BYTES`
asserted; `uptime` on both sides; alternating pairs; the loudest pair kept. The
**call count is the control**: an arm that cannot move the counter cannot contain
the change.

### 19.7 Owed, named here rather than discovered later

Every item below is owned by row `MUSIC3-DEPTH-DEVICE` and names the issue that
tracks it, per `.agents/reachability.md` and `AGENTS.md` `## Nothing lands dead`.

* **The engine's call to `Music3SelectDepthArm` is reachable but not gated**
  ([#1131](https://github.com/mudler/vllm.cpp/issues/1131), row
  `MUSIC3-DEPTH-DEVICE`). `--speech-device 1` reaches it and no CI gate can:
  deleting the two-line call leaves `test_minimax_music3_ar` 37/37 · 640/640 and
  `test_minimax_music3_speech` 9/9 · 223/223 green, because the engine needs the
  28.5 GB checkpoint and a real device. §19.5 carries the mutation and the binary
  hashes. The *rule* it calls is gated on both sides of its condition, so what is
  owed is the call, not the logic. It closes with the `thor:gpu0` legs in §19.6.
* **`scripts/check-fusion-consistency.py` is satisfied by a COMMENT**
  ([#1351](https://github.com/mudler/vllm.cpp/issues/1351), row
  `MUSIC3-DEPTH-DEVICE`). Replacing the `layers::UnquantizedMlpGateUpMethod` call
  in `minimax_music3_depth_device.cpp` with an inline hand-rolled
  `vt::MatmulBT` + `vt::SiluAndMul` path, while leaving the two comment mentions
  in place — the `#include`'s trailing `// layers::UnquantizedMlpGateUpMethod`
  and the `gate_up` field note — leaves the checker **green**, and the numbers
  bit-identical so nothing else sees it either. Verified two ways: a fresh
  reviewer ran the checker end to end at `rc=0`, and calling the checker's own
  functions gives `uses_merged_gemm_seam = True` for the hand-rolled variant with
  comments kept and `False` only once comments are stripped. `Check 2` runs
  `_MERGED_GEMM_SEAM` over `path.read_text()`; the file's only comment-aware
  helper, `allowlisted_names`, strips `#` comments from the **allowlist file**
  rather than from the scanned source. So the checker can no longer detect a
  regression to a hand-rolled path in this TU. **Reported rather than fixed
  here:** a checker change needs its own spec, a red-before test and green-after
  evidence per `AGENTS.md` `## Changing the rules or a checker`, and widening the
  regex is exactly the move that section forbids. Renaming `MlpGateUp` to
  `MlpGateUpXX` is *not* a detection gap — `MlpGateUp[A-Za-z]*Method` matches it
  by design, to cover `UnquantizedMlpGateUpGeluMethod`.
* **The depth K/V cache's pooling is unmeasured**
  ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), row
  `MUSIC3-DEPTH-DEVICE`). It now draws from the device pool through `DBuf`
  instead of `backend.Alloc`/`Free`, which at the shipped 4-layer batch-2
  geometry removes 16 `cudaMalloc` and 16 synchronizing `cudaFree` per frame. The
  change is correct by construction — `ReleaseShared` returns the block to the
  pool it came from — but **no speed number is quoted for it**, because this box
  has no GPU and the A/B is blocked on correctness. Measure it with the §19.6
  `thor:gpu0` legs, not before.
* The **audio heads, the CFG mix, the top-k draw and the fed-back projection**
  stay on the host. They are ~1.6 % of the stage today; they become the stage's
  remainder once the forward moves, and that is the next thing to measure rather
  than to assume.
* The **host arm's f32 weight containers** are unchanged (§19.2a). §16.7's lever 1
  — bf16 host storage, bit-identical on the safetensors lineage and **not** on
  the GGUF Q4_K one — is still open and is still not this row.
* The **condition mix** is still host-side (§14.5). It runs once per window, not
  once per step.
* **Why Thor's stage ratios exceed x86's** is still a hypothesis (§16.6b) and
  this row does not settle it.

### 19.8 Stop conditions

Stop and report rather than widening scope if: the CUDA arm's disagreement with
the host arm exceeds §19.4's band and the cause is not one of that section's
three; `vt::AttentionCross` refuses this geometry; the staged bf16 weights do not
fit beside the 8.6 B language model on the measurement box; or
`scripts/ab-arms-differ.py` returns FATAL for the pair (#1516: the bare
`ARMS_DIFFER` hash comparison this used to name cannot fire for an ABI client).

---

## 20. Where the time goes on CURRENT MAIN (#672, [#1512](https://github.com/mudler/vllm.cpp/issues/1512)) — five merges, priced end to end, and the conclusion §15.2 drew is now inverted

Five changes landed on this model's two largest buckets in two days and not one
had an end-to-end number on the shipped binary: the depth incremental schedule
(#1238), the depth **device arm** ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), PR #1330),
the vocoder `Conv1d` tiling ([#1334](https://github.com/mudler/vllm.cpp/issues/1334), PR #1356),
`vt::SiluAndMul`'s rounding polarity ([#1322](https://github.com/mudler/vllm.cpp/issues/1322), PR #1347)
and the vocoder conv **f32 accumulator** ([#1474](https://github.com/mudler/vllm.cpp/issues/1474), PR #1484).
This section is that number, on one box, in one job, under one lease.

**The headline: at 4 s / 4 steps the run is 3.44x faster and at the developer's
20 s / 30 steps it is 5.49x faster — and the flow-matching DiT, which §15.2
measured at 0.34 % of a run and declared closed, is now 62.24 % of it.**

### 20.1 The run, and the mutex that is the whole of the serialisation

`rc` job **`c206ec87-65eb-4d0d-93ad-05538325e66e`** on **`thor:gpu0`**,
`--max-runtime 480m`, worker `rc-worker-m4d7t`, `Linux 6.8.12-1021-tegra`
aarch64, 14 cores, NVIDIA Thor sm_110 (capability 11.0), driver 595.78,
`overlay` root with 169 GB free. No `ssh`, no `rc hold`, no `$GPU_LOCK`. §13.10
retains a whole speed axis as VOID because its arms went in over `ssh` under the
file mutex while another session held the same box through `rc`; this job has
exactly one mutex and it is the lease.

**Contention, recorded rather than assumed.** `uptime` was **3.27** at job start
with **0 logins**, which is this box's idle floor (§18.8a read 3.29 on the same
worker). It sits at 4.8-16 across the timed runs, and the high readings are our
own 14-thread host vocoder, not a foreign job: the load average rises inside a
`vocoder.decode_window` and falls between runs. Arms are alternated so both eat
the same contention, and the three rounds agree to better than 1.5 % on every
bucket, which is the evidence that nothing else was on the box.

The instrument is `VLLM_CPP_MUSIC3_PROFILE=1` (§15.1). It takes **no GPU clock
window**, so every figure here is a within-run SPLIT or a same-box A/B and none
is quotable as a per-kernel or cross-box number.

### 20.2 The arms, and why the sha256 pair is NOT what separates them

| | OLD | NEW |
|---|---|---|
| commit | **`d0598a255`** — `main` immediately before PR #1330 | **`a50c57d69`** — `origin/main` |
| `git rev-parse HEAD`, asserted equal to the expected sha | yes, `FATAL_WRONG_SHA` guard | yes |
| clone | separate `git clone`, `FATAL_CLONE` guard | separate clone |
| build dir | `/tmp/b-old` | `/tmp/b-new` |
| `minimax-music3-gen` sha256 | `91387d74e27cdf64492dd2632ad1162febd606bd4d3eef5f5f87df4ffca4bfa2` | `e2742d4ab471159feec407cc72ddfd430baada0ad931fff8599b35daa192d9bb` |

Both built inside the lease, `CMAKE_BUILD_TYPE=Release`, `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`, `ninja -j 6`, after
`apt` installed `cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0`
(nvcc 13.0.88). 291 s and 303 s. `diff -rq` reports **73 files differing under
`src/` and 49 under `include/`**.

**The hash guard passed and it proves nothing, which is [#1516](https://github.com/mudler/vllm.cpp/issues/1516).**
`examples/CMakeLists.txt:425-426` links `minimax-music3-gen` against the SHARED
`vllm::shared` (`CMakeLists.txt:2633`), so the timed program is a **72 744-byte
ABI client** — both arms are that size to the byte — and every line of the change
lives in `libvllm_shared.so`, which no arm hashed. Nothing sets
`CMAKE_SKIP_BUILD_RPATH`, so two build directories write two RPATH strings into
the client and the hashes differ whatever the source says. §16.6b reads exactly
this construction as "the precondition this section exists to insist on"; that
inference does not hold, there or here.

**What DOES separate the arms is behavioural, and it is stronger than a hash.**
`Music3DepthDeviceForwardCount()` (`minimax_music3_depth_device.cpp:154`) is not
reachable from any production run — it is read only by
`test_minimax_music3_ar` — so the e2e observable is the bucket SET.
`Music3SelectDepthArm` (`minimax_music3_llm.cpp:569-586`) brackets its staging as
`ar.depth_staging` and returns an un-engaged arm without it, and the loop takes
the device path **iff** the arm is engaged (`minimax_music3_llm.cpp:469-476`). So
the bucket is present exactly when the device arm ran. Three controls at
`--duration 0.24 --steps 2`, 56 depth forwards each:

| control | `ar.depth_staging` | `ar.depth_forward` | denoise bucket |
|---|---|---|---|
| **OLD, `--device 1`** | **absent** | 5.982 s | `denoise.dit_device` 0.472 |
| **NEW, `--device 1`** | **0.917 s** | **0.352 s** | `denoise.dit_device` 0.478 |
| **NEW, `--device 0`** | absent | 5.814 s | `denoise.dit_host` **196.691** |

One binary cannot emit two different bucket sets. The middle row is the arm
engaging; the third is the same NEW binary refusing to, on a CPU queue, and
landing on the host cost the OLD binary pays. **The depth device arm rides
`--device` / `--speech-device` and nothing else** — there is no separate flag and
no environment variable — so "depth host" and "depth device" are not independently
selectable from a production entry point, and the `--device 0` row also moves the
8.6 B language model and the 2.4 B DiT to the host. That is why the matrix below
varies the vocoder rather than the depth arm.

### 20.3 The checkpoint was staged, and the assertion is the reason the figures are quotable

| check | result |
|---|---|
| what `--model` would have read | `findmnt /workspace` -> `//192.168.68.102/Data[/rc] cifs` |
| what it actually read | `findmnt -T /tmp/ckpt` -> `overlay overlay /` |
| staging copy completed | `SRC_BYTES=28517617303` = `DST_BYTES=28517617303`, hard fail on mismatch |
| against the recorded byte count | `EXPECT_BYTES=28517617303`, equal |
| staging cost | `STAGE_SECONDS=999` = **28.5 MB/s off CIFS** |
| local sequential read ceiling | `cat` all 28.518 GB: **6 s = 4752 MB/s** |
| what the loader then paid | `load.ar_weights` **5.37-6.21 s**, `load.acoustic_weights` **1.93-2.13 s** |

§15.6 measured 780.015 s cold-CIFS and 5.689 s local for `load.ar_weights`; this
job reads **5.37 s** on the same box, so the staging result reproduces. No load
figure implies more than the 4752 MB/s control.

### 20.4 The matrix — 4 s / 4 steps / 100 frames / seed 7, three alternated rounds

Twelve runs, `old·host, old·cuda, new·host, new·cuda` in that order, three
times. `VLLM_CPP_VOCODER_DEVICE` unset is the shipped default and resolves to
`kCPU` (`vocoder1d.cpp:96-98`); `=cuda` is the opt-in device arm. Every run is
`--device 1`. Medians of three, with the raw triples beside them because the
spread is what says the box was quiet:

| bucket | OLD·host | OLD·cuda | NEW·host | NEW·cuda |
|---|---|---|---|---|
| **wall clock** | **166.038** | **165.449** | **48.070** | **52.684** |
| | 166.038 / 165.610 / 167.042 | 166.039 / 165.011 / 165.449 | 48.193 / 48.070 / 47.918 | 52.435 / 52.684 / 52.795 |
| `ar.depth_forward` (808 calls) | **80.728** | 81.070 | **4.272** | 4.305 |
| | 80.118 / 80.728 / 81.694 | 81.513 / 80.734 / 81.070 | 4.247 / 4.289 / 4.272 | 4.305 / 4.312 / 4.294 |
| `vocoder.decode_window` (1 call, 344 latents) | **54.091** | **53.580** | **15.078** | **19.288** |
| | 54.091 / 54.042 / 54.646 | 53.580 / 53.593 / 53.553 | 15.107 / 15.078 / 14.897 | 19.218 / 19.288 / 19.306 |
| `ar.lm_decode_step` (100 calls) | 12.338 | 12.349 | 9.517 | 9.516 |
| `denoise.dit_device` (4 calls = 8 forwards) | 5.060 | 5.060 | 5.048 | 5.050 |
| `ar.depth_staging` | — | — | 0.908 | 0.915 |
| WAV sha256 (first 32 hex) | `8a997f193b589adac37abe0a77ad029e` | same | `9b7d0a2a92aa3388010349fd70742fb6` | same |
| RMS / peak | 0.01943 / 0.23248 | same | 0.02940 / 0.38700 | same |

**The instrument reproduces the record.** §16.6b's AFTER arm (`4568c6e71`, three
pairs, same box, same request, staged) reads wall **163.00 s**,
`ar.depth_forward` **78.316 s**, `vocoder.decode_window` **53.605 s**,
`ar.lm_decode_step` **12.339 s**. OLD·cuda here reads **165.449 / 81.070 /
53.580 / 12.349** — within 1.5 %, 3.5 %, 0.05 % and 0.08 %. Two different
commits, two different jobs and two different prompts, agreeing on three buckets
to better than the difference this section is about.

### 20.5 The four answers

**1. What today's work bought, end to end.** Both sides named, one job, one box,
one staged checkpoint, medians of three:

| comparison | before | after | ratio |
|---|---|---|---|
| wall, host vocoder on both sides | 166.038 s | 48.070 s | **3.45x** |
| wall, CUDA vocoder on both sides | 165.449 s | 52.684 s | **3.14x** |
| wall, each side's own best arm | 165.449 s | 48.070 s | **3.44x** |
| `ar.depth_forward`, 808 calls both sides | 80.728 s | 4.272 s | **18.90x** |
| `vocoder.decode_window`, host arm | 54.091 s | 15.078 s | **3.59x** |
| `vocoder.decode_window`, CUDA arm | 53.580 s | 19.288 s | **2.78x** |

Against the RECORDED 163.00 s (§16.6b, arm `4568c6e71`, CIFS-free), current main
delivers **48.070 s**, which is **3.39x** — quoted second because it crosses two
jobs, where the 3.44x does not.

At the developer's configuration, **20 s / 30 steps / 500 frames / 4 windows**,
one run per vocoder arm on NEW, checkpoint staged:

| | recorded §15.7 | NEW·cuda | NEW·host |
|---|---|---|---|
| wall | **3269.789 s** | 624.127 s | **595.899 s** |
| `ar.depth_forward` | 1710.456 s / 7014 | 21.055 s / 4008 | 21.099 s / 4008 |
| `vocoder.decode_window` | 421.670 s / 4 | 150.060 s / 4 | **122.169 s / 4** |
| `denoise.dit_device` | 370.746 s / 120 | **370.634 s / 120** | **370.556 s / 120** |
| `ar.lm_decode_step` | 83.629 s / 500 | 56.082 s / 500 | 56.174 s / 500 |
| `load.ar_weights` | 397.909 s | 5.369 s | 5.442 s |
| `load.acoustic_weights` | 210.482 s | 2.035 s | 1.979 s |

**3269.789 s -> 595.899 s is 5.49x**, and the two sides differ in TWO things, not
one: the code, and a checkpoint that §15.7 read off CIFS and this run read off
local disk. Excluding both load buckets from each side gives **2661.398 s ->
587.951 s = 4.53x**, which is the part the code bought. Both are stated because
neither alone is the honest answer.

`ar.depth_forward` at **81.2x** on that row mixes the incremental schedule
(7014 -> 4008 calls) with the device arm; per call it is **243.9 ms -> 5.25 ms =
46.4x**. `denoise.dit_device` at **1.000x over an identical 120 calls** is the
control that says the two runs are the same workload.

**2. Does the vocoder CUDA arm win now that the accumulator is f32? NO — and the
f64 hypothesis was still right.**

| latent frames | arm | host | CUDA | CUDA/host |
|---|---|---|---|---|
| 344 (4 s clip) | OLD, f64 accumulator, untiled host kernel | 54.091 s | 53.580 s | **1.010x** |
| 344 (4 s clip) | NEW, f32 accumulator, tiled host kernel | 15.078 s | 19.288 s | **0.782x** |
| ~689 x 4 windows (20 s clip) | NEW | 122.169 s | 150.060 s | **0.814x** |

The f32 accumulator is worth **2.78x on the device arm** (53.580 -> 19.288 at
344 latents), so §13.10's leading suspect was the right one and Thor's fp64 rate
was a large term. The arm still loses, because #1356 and #1474 together moved the
HOST kernel further — **3.59x** against the device arm's 2.78x. The answer holds
at both sizes measured and it is taken on an idle box under a real lease, which
is what §13.10 lacked.

**This also closes a leg §18.9 carries as owed.** §13.4's `memcmp` between the
providers had not been re-measured against the tiled host kernel. It is measured
here end to end and it holds: **six OLD runs wrote one WAV hash and six NEW runs
wrote one WAV hash**, across both vocoder arms, and the 20 s pair wrote
`55856deb3b5b727a4ca4fcc473e01a56` on both arms. The device vocoder is
byte-identical to the host vocoder in the finished audio, at f64 and at f32, at
344 and at ~689 latents.

**3. Where the time goes now.** Current main, `--device 1`, default (host)
vocoder, checkpoint staged, at the duration the developer actually asks for:

| rank | bucket | seconds | % of wall | calls |
|---|---|---|---|---|
| 1 | **`denoise.dit_device`** | **370.556** | **62.24** | 120 (= 240 forwards) |
| 2 | `vocoder.decode_window` | 122.169 | 20.52 | 4 |
| 3 | `ar.lm_decode_step` | 56.174 | 9.44 | 500 |
| 4 | `ar.depth_forward` | 21.099 | 3.54 | 4008 |
| 5 | `ar.depth_projection` | 6.966 | 1.17 | 3507 |
| 6 | `load.ar_weights` | 5.442 | 0.91 | 1 |
| 7 | `ar.semantic_guide_and_draw` | 3.701 | 0.62 | 501 |
| 8 | `ar.depth_head_and_draw` | 3.489 | 0.59 | 3507 |
| | `unattributed` | 0.331 | 0.06 | |

**§15.2's conclusion is inverted, and that is the finding this row exists to
deliver.** It read "the DiT on the device is 0.34 % of the run ... **the GPU is
not the problem, and no further DiT work will move this number**". That sentence
was true of a 4 s clip whose wall was 69.5 % CIFS load and 23.4 % a host depth
decoder. Remove the load (§15.6), remove the depth decoder (#1330) and quadruple
the step count to the shipped 30, and the same bucket is **62.24 %**. The DiT did
not get slower — 370.6 s against §15.7's 370.7 s over an identical 120 calls —
everything around it got faster. **The next row is the DiT**, and after it the
vocoder's parallel decomposition, which §18.8b already named.

**4. What did not behave as the record predicts.**

* **The untiled host vocoder is 54.091 s here and 97.4463 s in the kernel
  bench, at the same 344 latents, through the same call.** §18.8a drove
  `VocoderDecode` — the call `Music3DecodeChunks` brackets as
  `vocoder.decode_window` — with SYNTHETIC weights and measured 97.4463 s
  before the tiling and 67.7083 s after. This job measures the real checkpoint
  through the real request and reads **54.091 s** on the same untiled kernel,
  which is below even that bench's tiled arm. **The two instruments disagree by
  1.80x and neither is withdrawn**: this one is e2e on real weights, that one is
  a two-binary A/B whose ratio is internally consistent across a 17x span of
  work. A hypothesis, labelled as one: synthetic pseudo-random weights can drive
  an f64 dependent accumulate chain into subnormals in a way a trained
  checkpoint does not, and no counter was read on either side. **The consequence
  is concrete: the 1.364x-1.439x tiling figure cannot be multiplied onto any
  e2e bucket**, and this section's 3.59x is tiling AND f32 together and is not
  decomposable from these data.
* **`ar.lm_decode_step` fell 12.338 s -> 9.517 s (1.30x) across an unchanged 100
  calls**, and at 20 s it is 83.629 -> 56.174 (1.49x) across an unchanged 500.
  Nothing in the five merges touches the language model's decode. §16.6b
  recorded the same shape at 1.21x and offered a smaller working set as a
  hypothesis; it recurs here, larger, and is still unmeasured.
* **Two host remainders got SLOWER**, which §19.7 predicted would happen and did
  not predict the sign of: `ar.semantic_guide_and_draw` 1.725 -> 3.701 s over an
  unchanged 501 calls, and `ar.depth_head_and_draw` 3.300 -> 3.489 over an
  unchanged 3507. They are 1.2 % of the run and are named rather than absorbed.
* **`Music3DepthDeviceForwardCount()` is unreachable from a production run.** The
  counter exists for exactly the question this row had to answer and only
  `test_minimax_music3_ar` can read it, so the arm had to be established from
  `ar.depth_staging` instead. That works — the bucket is present iff the arm is
  engaged — but a run cannot report how many device forwards it made, and a
  reader checking 4008 against 501 x 8 has no instrument for it.
* **The two 166-second OLD runs of round 1 agree to 1 ms** (166.038 and 166.039).
  Every bucket inside them differs, so they are two runs and not one; the
  coincidence is recorded because an equal wall clock is the tell §16.6a taught
  us to distrust, and here the bucket table is what refutes it.

### 20.6 What this closes, and what it does not

Closed: §19.6's `stage A/B + wall` and `WAV identity` legs on `thor:gpu0` under
`rc`; §18.9's owed re-measurement of the CUDA vocoder against the tiled host
kernel; and the e2e number every one of the five merges was missing.

Not closed, and not turned into a ceiling: the DiT at 62 % of the run has no
row; the vocoder's `ForOutputRows` decomposition still has every thread sweeping
the whole input tensor (§18.8b); `ar.lm_decode_step`'s repeated unexplained gain;
the 1.80x instrument disagreement in 20.5; [#1516](https://github.com/mudler/vllm.cpp/issues/1516)'s
guard; and the unreachable forward counter. The `--device 0` arm is 286.569 s for
a 0.232 s clip against `--device 1`'s 19.440 s, so nothing on the host path is a
supported configuration at any real duration and no host baseline is offered
above 0.24 s.

**Evidence:** `.agents/benchmark-record.md`, section `MUSIC3-E2E-ON-MAIN`. Raw
log `/workspace/music3-e2e/log-20260820T221735Z.txt` on the shared NAS, 1080
lines, every bucket table for all seventeen runs.

---

## 21. The DiT gets a profile below the stage boundary (#672, [#1542](https://github.com/mudler/vllm.cpp/issues/1542)) — §20.6's "the DiT at 62 % of the run has no row"

§20 measured `denoise.dit_device` at **370.556 s, 62.24 % of the developer's
20 s / 30 steps run**, over 120 calls = 240 forwards, and closed with "the next
row is the DiT". This is that row.

### 21.1 The gap, stated exactly

The instrument reports **one bucket** for the whole forward
(`minimax_music3_speech.cpp:321`) and nothing inside it. Everything known about
where those 370.556 s go is arithmetic performed on the outside of a black box.

**The one number that frames the row.** 370.556 s over 240 forwards is
**~1.544 s per forward**. At the shipped geometry — 36 blocks, `inner_dim` 2048,
`ff_inner_dim` 8192, `num_attention_heads` 32 x `attention_head_dim` 64, and a
window of ~689 latent frames so `seq` ~690 — one forward is

| term | per forward at seq 690 |
|---|---:|
| block-stack GEMM (qkv, out, ff_in, ff_out) | 4.83 GFLOP/token x 690 = **3.33 TFLOP** |
| attention scores + values, 36 layers | 0.14 TFLOP |
| fp32 weight bytes read | 9.66 GB |

so the forward runs at **~2.2 TFLOP/s** and reads its weights at ~6.3 GB/s. The
weight traffic alone is ~35 ms at this box's bandwidth, so the forward is **44x
above its memory floor** and the question is entirely what the compute is doing.
That fraction of the device is not known, and this row's first job is to stop
guessing it.

### 21.2 THE DTYPE — settled against the oracle before any lever is proposed

AGENTS.md is explicit that a token gate cannot detect a dtype that is too wide,
so the DiT's fp32 was re-derived from the pinned oracle rather than taken from
§2.1, and it is **upstream's resolved choice, not a too-wide accident**:

| question | oracle answer, at pin `c6da9936` |
|---|---|
| what dtype does the converter give the transformer? | `scripts/convert_minimax_music3_to_diffusers.py:267` — `--dtype` defaults to `float32`; `:208` applies it as `convert_transformer(...).to(args.dtype)` |
| is that overridden anywhere for this component? | no. `:214` forces the RVQ depth decoder to `torch.bfloat16` and **only** that one, which is what makes the transformer's fp32 a deliberate default rather than an unset one |
| what does the pipeline cast into it? | `src/diffusers/modular_pipelines/minimax_music3/denoise.py:83` — `condition = condition.to(components.transformer.dtype)` |
| what does the released artifact carry? | the checkpoint's `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` header reports **`F32` for all 231 tensors** |

**So a bf16 or TF32 DiT would be a divergence from the oracle, not a repair of
one, and this row does not propose one.** The one precision-adjacent lever that
is admissible is a mechanism that keeps the declared operand and accumulate
dtype at fp32 — CUDA 13's cuBLASLt fp32 emulation
(`CUBLASLT_MATMUL_DESC_EMULATION_STRATEGY`) is the candidate, and it is
admissible only if it holds the EXISTING `kDitRelTol` 1e-4 / `kDitAbsFloor` 5e-5
/ `kDitMeanAbsTol` 5e-6 bounds against the upstream capture with the margin
§14.4 already records. It is measured before it is proposed, and rejected if the
margin moves.

**vLLM owns none of this.** vLLM and vLLM-Omni do not register this
architecture, so `diffusers` is the primary oracle for the DiT under AGENTS.md
`## When vLLM has no implementation`. The op beneath it is a different question:
`vt::MatmulBT`'s CUDA provider is shared with every vLLM-mirrored path in the
tree, so a change to its numerics or its plan handling is a **vLLM-owned**
surface and is out of this row's scope unless it is bit-identical by
construction.

### 21.3 The instrument this row adds

`music3_profile.h` already separates a LEAF from a SPAN: leaves partition the
run and are summed, spans enclose leaves, are printed for context and are
**never added**. The intra-DiT buckets are therefore SPANS, so
`denoise.dit_device` stays the leaf, `sum(leaf)` is unchanged, `unattributed`
stays a real quantity, and §15.7's and §20's tables stay comparable value for
value.

**They are behind a SECOND opt-in, and that is a correctness property of the
measurement rather than caution.** Attributing time inside the forward needs a
`Backend::Synchronize` at every span boundary, because the ops are asynchronous
on one stream and an un-synchronized bracket measures the launch, not the
kernel. Those syncs perturb the total. So `VLLM_CPP_MUSIC3_DIT_SPANS=1` is
separate from `VLLM_CPP_MUSIC3_PROFILE=1`: with only the latter set the forward
is byte-for-byte the path §20 timed, and the perturbation is MEASURED by running
both arms in the same job rather than argued to be small.

**The bucket set is the engagement control, which is the pattern §20.2
established.** `Music3DepthDeviceForwardCount()` is unreachable from any
production run, so a counter proves nothing about a shipped binary; a span that
is present *iff* the code ran is the observable. `dit.*` spans appear only from
`DitForwardDevice`, so their presence is the assertion that the device arm — not
the host `DitForward` — produced the numbers beside them.

### 21.4 Gates

| id | gate |
|---|---|
| G1 | the spans partition the forward: `sum(dit.*)` is within the sync overhead of `denoise.dit_device` on the same run, and a deleted bracket shows up as a gap |
| G2 | spans OFF is byte-for-byte the §20 path: no `Synchronize` and no clock read on the default configuration |
| G3 | correctness unmoved — `test_minimax_music3_acoustic_real` at the SAME `kDitRelTol` / `kDitAbsFloor` / `kDitMeanAbsTol`, both arms, nothing widened |
| G4 | reachability — deleting the production call site reddens the focused gate |
| G5 | the A/B for whatever lever the profile names, alternated, on `thor:gpu0` under an `rc` lease, checkpoint staged with `SRC_BYTES == DST_BYTES`, `uptime` on both sides, and a BEHAVIOURAL control rather than a binary hash (#1516) |

### 21.5 Risks

* **The sync-per-span perturbation could exceed the split it reports.** Mitigated
  by measuring both arms in one job and quoting the split as a within-arm ratio
  only.
* **A cuBLASLt lever moves a shared vLLM-mirrored op.** Any change to
  `cuda_matmul.cu` is scoped to be bit-identical by construction or to be
  refused; a numerics change there needs its own row and its own oracle.
* **The window geometry is inferred.** ~689 latent frames per window is derived
  from §20's vocoder call and the condition encoder's 25 Hz -> 86.13 Hz
  resample; this row PRINTS `seq` from the running forward rather than carrying
  the inference.

### 21.6 Stop conditions

Stop and report `NEEDS_DECISION` if the profile says the forward is at the
device's fp32 ceiling, because then the only remaining lever is a precision
change that §21.2 has already ruled a divergence from the oracle, and that is
the developer's call and not this row's.

### 21.7 Owed

* **[#1555](https://github.com/mudler/vllm.cpp/issues/1555) — the attention
  kernel.** §21.9 measures `vt::AttentionCross` at 43.9 % of the DiT forward for
  4.0 % of its arithmetic. This row measures it and does not fix it: the kernel
  has TWO consumers (this DiT and LTX-2.5, which reaches it six times per layer),
  every candidate reorders the head-dim summation so none is bit-identical, and
  admitting one needs both consumers to hold their EXISTING tolerances. That is
  its own row with its own numerics gate, not a change taken in passing here.
* **[#1131](https://github.com/mudler/vllm.cpp/issues/1131) is NOT closed by this
  row, and the shape of what is missing is now sharper.** That issue owes a gate
  that drives the engine through `Music3DenoiseDeviceArm` and asserts the device
  path was taken. The `dit.*` span set is exactly the "present iff the arm
  engaged" observable it asks for, and §21.9's runs show a production
  `minimax-music3-gen --device 1` emitting it — but the focused gate here drives
  `DitForwardDevice` directly, so deleting the `on_device ? DitForwardDevice(...)`
  call site at `minimax_music3_speech.cpp:323` leaves it green. A unit gate would
  need a reduced-dimension `Music3AcousticWeights` and `frame_hiddens` to drive
  `Music3DenoiseChunks`, which is a fixture this row did not build.
* **The synchronize is not gated.** §21.8's M3 establishes that the focused suite
  runs on the CPU backend, where `Backend::Synchronize` is a no-op, so no case in
  it can distinguish a drained bracket from an undrained one. The drain's
  necessity rests on §21.9's device pair.
* **The GEMM half's headroom.** The four `vt::MatmulBT` calls are 53.3 % of the
  forward at 3.98 TFLOP/s, and no measurement says what cuBLASLt could deliver at
  those shapes on this device. That is the row after
  [#1555](https://github.com/mudler/vllm.cpp/issues/1555), not this one.

### 21.8 The gate earns its teeth — four mutations, and one of them produced NO verdict

Every mutation below reports the COMPILER return code, `git diff --stat` and the
sha256 of the BINARY, because a mutation that fails to build and a mutation that
never applied both read as a passing test. `x86_64`, `Release`, at `0e18f8afd`.
Baseline binary `6d356119bd4c7e61...`, 36 cases / 345 assertions / `SUCCESS!`.

| mutation | compile | binary moved | result |
|---|---|---|---|
| **M1** the `dit.qkv` bracket deleted | rc 0 | `8038c347...` | **36 cases, 2 failed, 311 assertions, `FAILURE!`, rc 1** — restored to `6d356119...` byte for byte |
| **M2** `AddSince(..., span=false)`, so the sixteen land as LEAVES | rc 0 | `c5445572...` | **36 cases, 1 failed, 16 assertions failed, `FAILURE!`, rc 1** |
| **M3** `backend->Synchronize(queue)` deleted | **rc 1** | — | **NO VERDICT** — did not compile. Re-run as `(void)queue;`: rc 0, `a5a1271a...`, **36 cases / 345 assertions / `SUCCESS!` / rc 0** — GREEN by design; see below |
| **M4** `span_backend` forced null | rc 0 | `80a9a22b...` | **36 cases, 3 failed, 294 assertions, `FAILURE!`, rc 1** |

**M3 is the finding inside the mutation pass.** Deleting the drain orphaned the
`queue` parameter into `-Werror=unused-parameter`
(`minimax_music3_device.cpp:124`), the object failed to compile, and the run
produced no test result at all. Had the runner not printed `COMPILE_RC` the
STALE binary from the previous mutation would have printed `SUCCESS!` and the
line would have read as "the gate does not detect a missing synchronize" — a
conclusion about the code drawn from a broken instrument, which is the shape
[`.agents/verification.md`](../verification.md) names. It was re-run in a form
that keeps the parameter used (`(void)queue;`).

**The re-run's answer is a SCOPE STATEMENT rather than a pass, and the reason is
a code fact rather than an inference.** `Backend::Synchronize` is declared
`virtual void Synchronize(Queue&) {}` at `include/vt/backend.h:42` — an EMPTY
body — and no CPU backend overrides it. So on the CPU queue this suite runs on,
M3 removes a call to a function that does nothing, and no case here can
distinguish a drained bracket from an undrained one however it is written. The
necessity of the drain is established on the DEVICE, by §21.9's spans-on /
spans-off pair, and not by this suite.

**The re-run is GREEN, and the mutation reached the binary**, which is the pair
that makes it evidence rather than noise:

| | value |
|---|---|
| `COMPILE_RC` | **0** |
| binary | `6d356119...` -> **`a5a1271a...`** |
| result | **36 cases / 345 assertions / 0 failed / `SUCCESS!` / `TEST_RC=0`** |

So the drain can be deleted and this suite does not notice, on a compiled and
executed binary rather than by inference from `backend.h:42`. The inference and
the measurement agree.

**AND THE FIRST REPORT OF THIS LINE WAS WRONG, WHICH IS ITSELF THE THIRD
INSTRUMENT DEFECT IN THIS ROW.** An earlier revision recorded the re-run as
pending "because the authoring host sat at load 68-101 and the rebuild had not
finished". It had not been dispatched at all. The launcher waited on
`until ! pgrep -f mutate.sh` and the wait loop's OWN command line contains the
string `mutate.sh`, so the pattern matched the watcher; the loop could never
exit and never reached the line below it. Every later check then ran
`pgrep -f mutate3.sh`, which matched ITS own watcher for the same reason, so
"still running" was a process watching itself for the better part of an hour.
The tells were all present and all read the wrong way: `/tmp/mut3-build.log` did
not exist, no `ninja` or `cc1plus` was alive, and the test binary still hashed to
the untouched baseline. **A self-matching `pgrep` reports a job that was never
started as a job still in progress**, and the load average supplied a plausible
cause for a state that had a different one. Recorded here beside M3's compile
failure and the probe's two build failures because it is the same class as both:
an instrument failing toward a confident answer.

**M2 is the one that matters most and it is the least obvious.** Sixteen spans
landing as leaves changes no call count, no bucket name and no number inside the
DiT — it changes only whether `Report` adds them to `sum(leaf)`. Every §15.7 and
§20 table would then double-count the DiT and `unattributed` would go negative,
and the run would still print a plausible-looking split. Sixteen assertions fire.

### 21.9 The split, MEASURED — and §21.1's premise is wrong in the useful direction

`rc` job **`0f95377f-70dd-4bf8-93b5-8e44fd762713`** on **`thor:gpu0`**,
`--max-runtime 180m`, worker `rc-worker-m4d7t`, `Linux 6.8.12-1021-tegra`
aarch64, 14 cores, NVIDIA Thor sm_110, driver 595.78, boot id
`c99b7805-6e26-47a7-bc9d-93d592d676a6`. No `ssh`, no `rc hold`, no `$GPU_LOCK` —
the lease is the whole of the serialisation. `uptime` **3.46 with 0 logins** at
job start, which is this box's idle floor (§20.1 read 3.27, §18.8a 3.29); it
sits at 4.8-13.6 across the runs and the peaks are our own 14-thread host
vocoder, not a foreign job.

Tree `0e18f8afd`, `Release`, `-DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`, nvcc 13.0.88.

**The checkpoint was staged and the assertion is why the figures are quotable.**

| check | result |
|---|---|
| what `--model` would have read | `findmnt /workspace` -> `//192.168.68.102/Data[/rc] cifs` |
| what it actually read | `findmnt -T /tmp/ckpt` -> `overlay overlay /` |
| copy completed | `SRC_BYTES=28517617303` = `DST_BYTES=28517617303`, hard fail on mismatch |
| against the recorded count | `EXPECT_BYTES=28517617303`, equal |
| staging cost | `STAGE_SECONDS=708` = 40.3 MB/s off CIFS |

**`nvidia-smi` reports `clocks.sm` as `[N/A]` on this device**, so — as in §20 —
every figure here is a within-run SPLIT or a same-box A/B, and none is quotable
as a per-kernel or cross-box number.

#### The geometry, measured rather than inferred

`dit.seq_sum / 16 = 690` and `dit.length_sum / 16 = 689`. §21.1 derived `seq`
~690 from §20's vocoder latent count and the condition encoder's 25 Hz -> 86.13
Hz resample; the forward now prints it, and the inference was right.

#### The split — `--duration 20 --steps 2 --device 1`, 8 calls = 16 forwards

`denoise.dit_device` **25.104 s**; the sixteen spans sum to **25.100 s**, a
**99.98 % partition**. `sum(leaf)` 251.910 and `unattributed` 0.304 (0.12 %),
so the spans were printed and never summed and no §15.7 or §20 table moved.

| span | seconds | per forward | % of the DiT |
|---|---:|---:|---:|
| **`dit.attn`** | **11.010** | **0.6881** | **43.9 %** |
| `dit.ff_in` | 6.635 | 0.4147 | 26.4 % |
| `dit.ff_out` | 3.238 | 0.2024 | 12.9 % |
| `dit.qkv` | 2.591 | 0.1619 | 10.3 % |
| `dit.attn_out` | 0.926 | 0.0579 | 3.7 % |
| `dit.silu` | 0.225 | 0.0141 | 0.9 % |
| `dit.temb` | 0.167 | 0.0104 | 0.7 % |
| `dit.pre` | 0.122 | 0.0076 | 0.5 % |
| `dit.rope` | 0.062 | 0.0039 | 0.2 % |
| `dit.norm1` | 0.046 | 0.0029 | 0.2 % |
| `dit.norm2` | 0.045 | 0.0028 | 0.2 % |
| `dit.pack`, `dit.rope_build`, `dit.post`, `dit.untranspose`, `dit.readback` | 0.033 total | — | 0.1 % |

#### THE FINDING: the DiT is attention-bound, not GEMM-bound, and not at any fp32 ceiling

| group | seconds | % of the DiT | TFLOP per forward | TFLOP/s |
|---|---:|---:|---:|---:|
| **`vt::AttentionCross`** | **11.010** | **43.9 %** | **0.140** | **0.204** |
| the four `vt::MatmulBT` GEMMs | 13.390 | 53.3 % | 3.334 | **3.98** |
| norms, rope, SiLU, packing, readback | 0.700 | 2.8 % | — | — |

**The attention kernel does 4.0 % of the forward's arithmetic in 43.9 % of its
time — 19.5x slower per flop than the GEMMs beside it, on the same tensors, in
the same forward, on the same device.**

**§21.1's own premise is refuted, and in the useful direction.** That section
divided 370.556 s by 240 forwards and 3.33 TFLOP and reported "~2.2 TFLOP/s",
attributing the whole forward to the GEMM. The GEMMs are **3.98 TFLOP/s**; the
2.2 figure was an average over a forward that is nearly half something else.
Every sentence in §21.1 that rests on the 2.2 number is therefore superseded by
this section rather than merely refined.

**§21.6's stop condition does NOT fire.** It said to report `NEEDS_DECISION` if
the forward were at the device's fp32 ceiling, because the only remaining lever
would then be a precision change §21.2 has already shown to be a divergence from
the oracle. The forward is not at that ceiling and the lever is not a precision
change, so the oracle-divergence question never arises.

**At the developer's configuration** `dit.attn` is 0.6881 s x 240 forwards =
**165.2 s of the 370.556 s `denoise.dit_device` bucket, and 27.7 % of the whole
595.9 s run** (§20.5).

#### The perturbation is MEASURED, which is the whole reason the spans are a second opt-in

Arms alternated in one job, on one staged checkpoint:

| arm | `denoise.dit_device`, 8 calls | wall |
|---|---:|---:|
| spans ON | 25.104 s | 252.902 s |
| spans OFF | **24.737 s** | 253.577 s |

**The 331 synchronizes per forward cost 1.48 % of the DiT bucket**, and the two
wall clocks differ by 0.27 % in the OPPOSITE direction, which is inside this
harness's noise. So the split above is trustworthy at the 1.5 % level, and
`denoise.dit_device` with the flag unset stays comparable to §15.7's 370.746 s
and §20's 370.556 s value for value.

#### The engagement controls, per §20.2

`ar.depth_staging` 0.757 s is present in every run, so the depth device arm
engaged; the `dit.*` span set is present exactly in the spans-on runs, so those
numbers came from `DitForwardDevice` and not from the host `DitForward`. A
bucket set is the observable because `Music3DepthDeviceForwardCount()` is
unreachable from any production run.

#### The 30-step point — the §20 configuration, spans OFF, and it reproduces to 0.012 %

`--duration 20 --steps 30 --device 1`, checkpoint staged, box `uptime` 14.14
before and 12.80 after (both inside our own host vocoder):

| bucket | §20 (`a50c57d69`) | here (`0e18f8afd`) | calls | delta |
|---|---:|---:|---:|---:|
| **`denoise.dit_device`** | **370.556** | **370.510** | 120 | **-0.012 %** |
| `vocoder.decode_window` | 122.169 | 124.429 | 4 | +1.85 % |
| `ar.lm_decode_step` | 56.174 | 56.395 | 500 | +0.39 % |
| `ar.depth_forward` | 21.099 | 21.158 | 4008 | +0.28 % |
| `ar.depth_projection` | 6.966 | 6.880 | 3507 | -1.23 % |
| `ar.semantic_guide_and_draw` | 3.701 | 3.757 | 501 | +1.51 % |
| `ar.depth_head_and_draw` | 3.489 | 3.400 | 3507 | -2.55 % |
| wall | 595.899 | 598.207 | — | +0.39 % |

`sum(leaf)` 597.247 and `unattributed` 0.313 (**0.05 %**), so the table still
adds up with the sixteen spans compiled in.

**`denoise.dit_device` at 370.510 s against §20's 370.556 s over an identical
120 calls is 0.012 % — two commits, two jobs, two days.** That is G2 measured
end to end rather than argued: with `VLLM_CPP_MUSIC3_DIT_SPANS` unset this tree
produces the number §20 recorded, so the row's instrument did not move the
quantity the row exists to explain.

**And the audio is BYTE-IDENTICAL to the record.** This run wrote
`55856deb3b5b727a4ca4fcc473e01a56`, 3 530 796 bytes — the same hash §20.5
recorded for its 20 s pair at `a50c57d69`. The four short runs likewise wrote one
hash (`61a8989763bba749edab8ddc3e597d7a`) across both spans arms. So the
instrument is a pure timing arm at both durations, proved on the real checkpoint
rather than inferred from the diff.

**The finding therefore lands on the shipped configuration directly.** The
per-forward geometry is identical at 2 and 30 steps (`seq` 690 both times), so
`dit.attn` at 0.6881 s per forward x 240 forwards is **165.1 s of this run's
370.510 s DiT bucket — 44.6 % of the DiT and 27.6 % of the whole 598.207 s
run**.

#### One lever named in §21.2 is MEASURED UNAVAILABLE

§21.2 named CUDA's cuBLASLt fp32 emulation
(`CUBLASLT_MATMUL_DESC_EMULATION_STRATEGY`) as the single precision-adjacent
candidate that would keep the declared operand and accumulate dtype at fp32.
**It does not exist in this toolkit.** A probe transcribing
`MatmulBTKernelCuda`'s descriptor construction fails to compile on `thor:gpu0`
under nvcc/cuBLASLt **13.0.88**: `identifier
"CUBLASLT_MATMUL_DESC_EMULATION_STRATEGY" is undefined`, `PROBE_BUILD_RC=2`. It
is an enumerator rather than a macro, so no preprocessor test can guard it and
the arm is refused at run time instead.

That is a negative result and it costs nothing, because §21.9's split says the
GEMMs are 53.3 % of a forward that is 43.9 % attention. **The lever is the
attention kernel, and it is not a precision question at all.**

#### What is NOT established

* **The mechanism of the attention deficit.**
  [#1555](https://github.com/mudler/vllm.cpp/issues/1555) names the per-key
  five-step `__shfl_xor_sync` butterfly in `AttentionCrossFlashKernel` and an
  occupancy near 8 warps per scheduler as the leading hypothesis, with the
  instruction-count arithmetic beside it (~1.8 ms per layer predicted against
  19.1 ms measured, so roughly a 10x latency-hiding deficit; and 17.9 GB of K/V
  re-reads per forward, ~65 ms, about 10 % of the 688 ms). **No `ncu` counter
  was read on either side and no occupancy figure was measured.** The split is
  measured; its attribution is a hypothesis.
* **Any speed claim.** Nothing was made faster by this row.
* **A per-kernel or cross-box figure.** `nvidia-smi` reports `clocks.sm` as
  `[N/A]` on this device, so no clock window exists.

### 21.10 The GEMM half is at the device's fp32 ceiling — measured, so the attention lever is the ONLY one left in-oracle

A standalone probe transcribing `MatmulBTKernelCuda`'s descriptor, three
layouts, `TRANSA=T`/`TRANSB=N`, `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale type,
32 MB workspace and `requestedAlgoCount=1` — so what it prices is OUR
invocation and not a generic SGEMM. `rc` job on `thor:gpu0`, nvcc/cuBLASLt
**13.0.88 / 13.1**, driver 13020, binary
`9960f5e1e1fb41b90c1d0669c507927830bb486572a43d3030910bdfeeda44b0`, three
rounds.

**The device.** `NVIDIA Thor cc=11.0`, **20 SMs at 1.049 GHz**, so the fp32
CUDA-core peak is `2 x 128 x 20 x 1.049e9` = **5.369 TFLOP/s**, and the memory
bandwidth is 273.0 GB/s. (The lane count is an assumption, stated because the
percentages rest on it.)

**At the DiT's own M = 690**, medians of three rounds that agree to 0.3 %:

| shape | f32 `COMPUTE_32F` | % of fp32 peak | f32 `FAST_TF32` | bf16 | plan rebuild |
|---|---:|---:|---:|---:|---:|
| `qkv` `[690,2048]x[2048,2048]` | **3.84** | **71.5 %** | 52.84 | 134.66 | 0.9 us |
| `attn_out` same | **3.84** | **71.5 %** | 52.97 | 134.45 | 0.9 us |
| `ff_in` `[690,2048]x[16384,2048]` | **4.17** | **77.7 %** | 59.04 | 94.40 | 1.1 us |
| `ff_out` `[690,8192]x[2048,8192]` | **4.24** | **79.0 %** | 29.51 | 48.03 | 0.9 us |

TFLOP/s. The TF32 and bf16 columns run on TENSOR cores, so the fp32-CUDA-core
denominator does not apply to them and no percentage is quoted; they are here
only to price what precision would be worth, and §21.2 has already shown
precision to be a divergence from the oracle.

**Three things this settles.**

**1. The GEMM half is essentially at the ceiling.** cuBLASLt's true-fp32 GEMMs
reach **71.5-79.0 %** of this device's fp32 CUDA-core peak at the DiT's shapes,
which is what a well-served SGEMM looks like. §21.9 measured the in-situ GEMM
half at **3.98 TFLOP/s**, inside that 3.84-4.24 band.

**2. And the in-situ calls ARE those calls, within 2.9 %.** Summing the probe's
isolated per-call times over one block (3 x `qkv` + `attn_out` + `ff_in` +
`ff_out` = 22.59 ms) and over 36 blocks predicts **813.5 ms** of GEMM per
forward; §21.9 measures **836.9 ms**. So there is no dispatch overhead, no
launch-gap term and no untuned-shape term hiding in the 53.3 %: the GEMM half
costs what the library costs.

**3. The per-call plan rebuild is REFUTED as a lever.** `MatmulBTKernelCuda`
builds a descriptor, three layouts, a preference and a heuristic on EVERY call,
and the DiT makes 252 of them per forward. Measured at **0.9-1.1 us** each, that
is **~0.25 ms of a 1569 ms forward — 0.016 %**. A plan cache for this op would
buy nothing here. Named because it was this row's second-ranked hypothesis and
it is now closed rather than left open.

**So the only lever left inside the oracle is the attention kernel.** fp32 is
upstream's resolved dtype (§21.2, with anchors); cuBLASLt's fp32 emulation is
measured unavailable in this toolkit (§21.9); the GEMMs are at the fp32 ceiling;
and the plan rebuild is 0.016 %. Everything else in the forward is 2.8 %.

**The size of the prize, as a BOUND rather than a promise.** If
`vt::AttentionCross` merely matched the GEMMs' measured per-flop rate of
3.98 TFLOP/s, its 0.140 TFLOP would cost **35.2 ms instead of 688.1 ms**, the
forward would fall from 1569 to 916 ms (**1.71x on the DiT**), the bucket from
370.510 to 216.3 s, and the run from 598.207 to 444.0 s (**1.35x end to end**).
That is an upper bound derived from a flop ratio on a kernel nobody has written,
not a projection of any design, and it is stated so that the next row knows what
it is playing for.

### 21.11 What this closes, and what it does not

Closed: §20.6's "the DiT at 62 % of the run has no row"; the DiT's dtype
question, settled against the pinned oracle with anchors rather than carried
from §2.1; and the attribution of the 370.556 s, which is now a measured split
rather than an arithmetic guess.

Not closed, and deliberately not turned into a ceiling: the attention kernel
itself ([#1555](https://github.com/mudler/vllm.cpp/issues/1555)), which is the
next row and has two consumers; the 53.3 % GEMM half, whose headroom against
cuBLASLt at the DiT's own shapes is still unmeasured; and
[#1131](https://github.com/mudler/vllm.cpp/issues/1131), which this row does not
close — see `## Owed`.
