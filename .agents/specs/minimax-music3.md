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
**Status:** **W0-W7 DONE, W2 INCLUDED.** Every stage is implemented and gated; a COMPOSED request is not yet observed to completion on CPU (see `## Now`). Spec committed, both oracles pinned, §1.1 resolved and confirmed at runtime, the diffusers oracle gateable against committed goldens, the modular loader in the tree, the autoregressive half's compute gated at reduced dimensions and against the real bf16 checkpoint, and the ACOUSTIC half — flow-matching DiT, scheduler, CFG, window bookkeeping, DAC Flow-VAE vocoder — gated at both scales against the committed capture. §5's token-exact gate is WITHDRAWN: upstream's AR stage has no greedy path, and the acoustic half never had one to withdraw. W6 has landed: the model is a registered `SpeechRegistry` family reachable through the new `vllm_speech_*` C ABI (v20) and `POST /v1/audio/speech`, and the denoise+decode composition reproduces the capture's waveform. W7 has landed (§9): quantized checkpoints for this model exist in five formats; the RVQ depth decoder's GGUF Q4_K arm IS implemented and value-gated against a pinned artifact, and every other format and lineage is refused by name with the missing piece rather than surfacing as a confusing shape error. The 8.6B language-model forward and the other four components' GGUF arms are owed.
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
this row.

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
