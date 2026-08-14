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
**Status:** **W0 + W1 DONE, W3 DONE, W2 PARTIAL.** Spec committed, both oracles pinned, §1.1 resolved and confirmed at runtime, the diffusers oracle gateable against committed goldens, the modular loader in the tree, and the autoregressive half's compute gated at reduced dimensions and against the real bf16 checkpoint. §5's token-exact gate is WITHDRAWN: upstream's AR stage has no greedy path. The 8.6B language-model forward and W4-W7 are owed.
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

**Still owed on the LLM half:** the 8.6B `Qwen3ForCausalLM` forward itself.
`frame_hiddens[:, :4096]` is the language model's own hidden state, and
reproducing it means running that model teacher-forced on the golden codes
through our landed Qwen3 path, which needs an `inputs_embeds` entry it does not
have. That is the remainder of W2 and it is recorded here rather than discovered
later.

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
| **W1** | Modular loader: the six-component layout, weight-norm folding, the fp32/bf16 policy of §2.1, native-arm refusal by name | every component loads with shapes asserted against §1; converted-vs-native tensor equality checked, not assumed |
| **W2** | Global LLM on our landed Qwen3 path at vocab 200 000 | hidden-state parity vs `transformers`, then token-exact RVQ code parity vs the oracle |
| **W3** | Condition mix (8-layer weighted) + RVQ depth decoder, 8 codebooks | per-stage tensor parity; the depth decoder's 16-position window exercised at its boundary |
| **W4** | Flow-matching DiT + `FlowMatchEulerDiscreteScheduler` with `invert_sigmas` | per-step latent parity against the oracle at a fixed seed |
| **W5** | Vocoder **through the shared `Vocoder1D`** (§4.1): snake activations, weight-norm, `[8,8,4,2]` upsampling, the 128→2×64 stereo fold, WAV at **44100 stereo** (§1.1) | waveform parity within a stated absolute tolerance, and H3/IndexTTS-2.5 behaviour byte-identical |
| **W6** | Register as a `SpeechRegistry` family; extend `SpeechGenParams` ADDITIVELY for lyrics + description + controls (§4.1); NEW `vllm_speech_*` **`include/vllm.h`** surface with the ABI version bump; **the example HTTP server as a thin ABI client** | a song generates end to end from an HTTP request; IndexTTS-2.5 unchanged; SGLang-Omni cross-check; speed axis recorded with values and ratios |
| **W7** | Quantized arms — GGUF k-quants are a standing requirement, not a per-model choice | each arm gated, or refused by name and recorded as owed |

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

Next: W4/W5 acoustic, W6 the speech-family registration plus the `vllm_speech_*`
ABI and the example HTTP server, W7 the quantized arms. W4 is unblocked and does
not depend on W2's remainder.

Two things are owed and neither is this phase's to close: **no speed number
exists** — every capture so far ran on CPU because `dgx.casa` was down, so
nothing here touches the speed axis — and SGLang-Omni remains `gateable = no`,
read but never executed.
