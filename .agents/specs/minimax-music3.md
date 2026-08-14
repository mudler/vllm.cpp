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
**Status:** W0 — spec committed, no engine code.
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
the dtype policy. **Not established:** that the oracle runs here at all — no
diffusers execution of this model has happened on this project's hardware, which
is why `.agents/oracles/diffusers.md` still records `gateable = no`. W0 is not
complete until it does.

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

### 2.1 The dtype policy is upstream's, and it is deliberate

`convert_minimax_music3_to_diffusers.py:267` defaults `--dtype float32`; the
transformer, condition encoder and vocoder take that dtype (`:208-211`) while the
RVQ depth decoder is forced to bf16 regardless (`:214`). That matches the measured
headers exactly, and SGLang-Omni's README states both of its placements "run the
acoustic stage in FP32".

So fp32 on the DiT and the vocoder is **upstream's resolved choice, not a
too-wide accident** — the case AGENTS.md warns about (a token gate cannot see a
dtype that is too wide) does not apply here, because the oracle is fp32 too. Each
fp32 buffer on this path still carries the one-line reason the policy requires,
naming this section. Any future narrowing to bf16 is a measured change with its
own evidence, never a silent default.

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

Both pins go into `.agents/oracles/` in W0. SGLang-Omni gets its **own record**
rather than riding on `sglang.md`: it is a third repository with its own cadence,
and this row binds to it directly.

---

## 4. What we already own

| Need | Have | Anchor |
|---|---|---|
| Qwen3 dense forward, paged KV, sampling | ✅ token-exact 16/16 | `MODEL-TEXT-qwen3-qwen3-for-causal-lm` |
| flow-matching denoise loop | H3: fixed-step loop, DiT forwarded once per step | [minimax-h3.md](minimax-h3.md) |
| audio VAE decode + WAV writing | H3 audio VAE, LTX-2 audio VAE, `minimax_h3_wav.cpp` | `src/vllm/model_executor/models/` |
| diffusion request planning / pipeline shape | H3 planner + pipeline, LTX-2.5 pipeline | `minimax_h3_planner.cpp`, `ltx2_pipeline.cpp` |
| quantized arms on a diffusion model | H3 GGUF + NVFP4 arms | [minimax-h3.md](minimax-h3.md) §0 |

Genuinely new: the eight-codebook RVQ frame path, the depth decoder, the learned
8-layer condition mix, snake-activated DAC decoding with weight-norm, and the
LLM→diffusion handoff on *continuous hidden states* rather than discrete tokens.

---

## 5. Gates

**LLM half — token-exact.** The global LLM and the depth decoder emit discrete RVQ
codes. Greedy decode of the code sequence is compared against the oracle
token-for-token on a fixed prompt. This is a real token gate and it binds.

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

## 6. Phases

Each phase is dispatched to a **fresh implementer** from this spec, reviewed by a
**fresh reviewer** who mutates the claimed guarantees, and its gate is rerun by
the operator. Phases are separately claimable except where noted.

| Phase | Scope | Done when |
|---|---|---|
| **W0** | This spec; both oracle records pinned; §1.1 sample rate settled from source (**DONE**); **stand the diffusers oracle up and prove it builds and runs** | oracle executes the model and `diffusers.md` flips to `gateable = yes` with a path as evidence |
| **W1** | Modular loader: the six-component layout, weight-norm folding, the fp32/bf16 policy of §2.1, native-arm refusal by name | every component loads with shapes asserted against §1; converted-vs-native tensor equality checked, not assumed |
| **W2** | Global LLM on our landed Qwen3 path at vocab 200 000 | hidden-state parity vs `transformers`, then token-exact RVQ code parity vs the oracle |
| **W3** | Condition mix (8-layer weighted) + RVQ depth decoder, 8 codebooks | per-stage tensor parity; the depth decoder's 16-position window exercised at its boundary |
| **W4** | Flow-matching DiT + `FlowMatchEulerDiscreteScheduler` with `invert_sigmas` | per-step latent parity against the oracle at a fixed seed |
| **W5** | Vocoder: snake activations, weight-norm, `[8,8,4,2]` upsampling, WAV at **44100 stereo** (§1.1) | waveform parity within a stated absolute tolerance |
| **W6** | e2e through the shared seams; **`include/vllm.h` surface**; **the example HTTP server as a thin ABI client**, never including internal headers | a song generates end to end from an HTTP request; SGLang-Omni cross-check; speed axis recorded with values and ratios |
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

W0 — spec committed, no engine code. Next: pin both oracles in `.agents/oracles/`,
stand the diffusers oracle up far enough to prove it builds and runs this model,
and resolve the sample-rate contradiction in §1.1 from source. W1 is not
dispatched until W0's oracle is gateable.
