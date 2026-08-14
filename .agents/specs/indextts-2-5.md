# SPEC — IndexTTS-2.5, the first audio-GENERATING lane

**Rows:** `MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation`,
`MODEL-MM-indextts2-index-tts2-s2-mel-decoder`
**Issue:** [#634](https://github.com/mudler/vllm.cpp/issues/634)
**State:** `INVENTORIED` — scoped in this spec, unclaimed, and blocked on
[#633](https://github.com/mudler/vllm.cpp/issues/633). `SPIKE` would owe a
`CLAIM-*` owner these rows do not have.

## Scope

Port IndexTTS-2.5 — upstream-supported at
`https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5` and served by vLLM-Omni — so
that text plus a reference clip renders 22.05 kHz speech through our own engine
and the OpenAI-compatible speech surface.

This is the project's **first audio-generating model**. Every audio path we ship
today consumes audio (Parakeet, Voxtral, `audio_processor.cpp`); nothing
synthesizes it, and `/v1/audio/speech` does not exist.

In scope: both registered architectures, the mandatory reference-audio
conditioning path, bf16 inference, **the `SpeechEngine` seam, the ABI entry
points and the two OpenAI routes** (none of which exist today — see Shared seams
and ABI), and the gates below. Out of scope: quantized arms (see Risks/decisions), streaming
(`async_chunk=false` upstream, disabled for correctness), and any language whose
support cannot be confirmed against the shipped config.

## Upstream chain

`vllm-project/vllm-omni` registers **two** architectures for this model, at
`vllm_omni/model_executor/models/registry.py` @
`bbe6ccc512a404a2df8c977ea29003002f2683e8` (the same commit the Moss-TTS rows
anchor to):

| Registry key | Module | Class |
|---|---|---|
| `IndexTTS2TalkerForConditionalGeneration` | `indextts2/indextts2_talker.py` | `IndexTTS2TalkerForConditionalGeneration` |
| `IndexTTS2S2MelDecoder` | `indextts2/indextts2_s2mel_decoder.py` | `IndexTTS2S2MelDecoder` |

Deploy config `vllm_omni/deploy/indextts2_5.yaml`, selected by `model_type`.
Offline entry point `vllm_omni.model_executor.models.indextts2.end2end`. Serving
is `vllm-omni serve IndexTeam/IndexTTS-2.5 --omni --trust-remote-code`, exposing
`/v1/audio/speech` and `/v1/audio/voices`.

**Pipeline.** Stage 0 is a ~0.8B GPT-2 AR talker turning text + reference audio
into mel codes. Stage 1 is EnhancedCodec (2.5 replaces IndexTTS-2's RepCodec, and
sets `use_gpt_latent=false`), then an S2Mel CFM/DiT flow-matching decoder, then
BigVGAN, emitting 22.05 kHz mono. About 6 GB VRAM — it fits GB10 with enormous
headroom.

**Reference audio is mandatory.** Upstream states IndexTTS-2 does not support
text-only synthesis, so the voice-cloning encoders (w2v-bert-2.0, MaskGCT
semantic codec, CAMPPlus speaker embedding) are required port surface, not an
optional extra. They fetch into `checkpoints/hf_cache/` on first run and need
revisions pinned under the NAS checkpoint policy.

## Our baseline

Nothing named IndexTTS exists in the tree. What exists and is reusable:

| Piece | Where | Note |
|---|---|---|
| BigVGAN 1-D core (Conv1d, ConvTranspose1d, replicate/zero pad, Snake/SnakeBeta, alias-free `Activation1d`) | published from `include/vllm/model_executor/models/minimax_h3.h`, gated by BOTH the H3 and LTX-2.5 suites | **already shared across two consumers.** LTX-2.5 (#435, merged) deliberately did NOT copy it: `ltx2_audio_vae.cpp:223-230` records that a second copy of the alias-free trim geometry "goes wrong quietly, because each copy keeps its own green gate while the two audio VAEs drift apart". IndexTTS-2.5 would be the THIRD consumer |
| WAV serialization | `minimax_h3_wav.cpp` | channel-major → interleaved, clamped |
| Flow-matching denoise loop, AdaLN/timestep machinery | the H3 lane | S2Mel is the same shape of computation |
| Conformer encoder | `parakeet_encoder.cpp` | w2v-bert-2.0 is a Conformer |
| Mel front end | `whisper_audio.cpp`, plus LTX-2.5's own (`ltx2_audio_vae_encoder.h`, #641) | two exist; which one w2v-bert-2.0 needs is a W3 question, not an assumption |
| Generation serving seam | `vllm::multimodal::VideoEngine` (abstract, checkpoint-detected; H3 and LTX-2.5 both behind it since #641) | the template for a speech engine seam, and the precedent that a second generative lane extends the seam rather than forking it |
| GPT-2-family backbone | `opt.cpp` | learned absolute positions + LayerNorm; the talker is an additive delta, not a fresh transformer |

New from scratch: EnhancedCodec, S2Mel, and the three reference encoders.

## Port map

| Stage | Upstream | Ours | Kind |
|---|---|---|---|
| AR talker | `indextts2_talker.py` | additive GPT-2 arch routed through `ModelRegistry::Forward` + `dense_attn::AttnBlock` + on-device sampling | new, small |
| Reference encoders | aux checkpoints under `hf_cache/` | partial reuse of `parakeet_encoder.cpp` / `whisper_audio.cpp` | new, largest |
| EnhancedCodec | `indextts2/` | — | new |
| S2Mel CFM/DiT | `indextts2_s2mel_decoder.py` | H3 denoise loop | reuse |
| BigVGAN | — | the shared 1-D core published from `minimax_h3.h`, relocated to a neutral home (W1) | reuse, third consumer |
| WAV 22.05 kHz | — | `minimax_h3_wav.cpp` | reuse |
| `/v1/audio/speech`, `/v1/audio/voices` | `vllm_omni/entrypoints/openai/` | a `SpeechEngine` seam + ABI v19 + two additive routes; see Shared seams and ABI | new |

Examples and servers stay ABI clients; no internal headers.

## Shared seams and ABI

A capability that is not reachable through the shared surface is not done, so
this is scope, not follow-up.

**Verified state today: nothing is wired.** `include/vllm.h` is at
`VLLM_ABI_VERSION 18` and contains no speech or TTS entry point of any kind.
`ApiServer` registers `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/models`, `/v1/audio/transcriptions`, `/v1/videos` and
`/v1/videos/sync`. `audio/speech` and `audio/voices` have zero hits across
`src/`, `include/` and `examples/`. Every audio route we serve today consumes
audio; none produces it.

**The precedent to mirror is the video lane**, which solved the same problem for
a generative modality: an abstract `vllm::multimodal::VideoEngine`
(`include/vllm/multimodal/video_engine.h:135`) with checkpoint detection, the
whole assembly library-owned behind ABI entry points (`vllm_video_engine_load`,
`vllm_video_generate`, `vllm_video_result_free`, `vllm_video_params_default`,
`vllm_video_engine_family`, and the mux argv pair), `/v1/videos` routing through
that same seam, and `minimax_h3_gen` reduced to a thin `vllm.h` client. H3 and
LTX-2.5 both sit behind it, which is the proof the shape holds for a second lane.

**What this lane adds**, mirroring that structure rather than inventing one:

| Piece | Shape |
|---|---|
| `vllm::multimodal::SpeechEngine` | abstract seam, checkpoint-detected, in `include/vllm/multimodal/speech_engine.h`; IndexTTS-2.5 is its first implementation and must not be its only possible one, since the omni TTS family is ~10 more architectures |
| ABI v19 | `vllm_speech_engine_load` / `_free` / `_family`, `vllm_synthesize` (naming symmetric with the existing `vllm_transcribe`), `vllm_speech_params` / `_default`, `vllm_speech_result` / `_free`, and a voice-enumeration pair for the voices route |
| `/v1/audio/speech` | additive route on `ApiServer`, routing through the SAME seam, in OpenAI's wire shape; the reference clip arrives per-request because upstream has no text-only synthesis |
| `/v1/audio/voices` | enumerates registered reference voices; upstream exposes it alongside speech |
| `examples/` | a thin `vllm.h` client only. No internal headers, per the ABI-clients rule |

The ABI bump is a real version increment with the `test_capi` section that goes
with it, not a header edit: the video lane's v12 bump is the template.

**Refusal is part of the surface.** Asking a non-TTS checkpoint to synthesize, or
asking for a quant arm that is not implemented, refuses at load naming the
missing piece. An arm that is silently absent is the failure this project has
already recorded; an arm that refuses by name is owed debt.

## Tests to port

Upstream's modules are Python and the checkpoint carries its own remote code
under `--trust-remote-code`, which a pure-C++ engine cannot execute. So the
oracle is upstream's modules executed offline and frozen, exactly the
`scripts/gen-minimax-h3-goldens.py` pattern — a `scripts/gen-indextts2-5-goldens.py`
that imports them by file path, with both sides rebuilding weights and inputs
from an identical deterministic stream so no checkpoint byte is committed.

| Golden | Stage |
|---|---|
| talker logits + emitted mel codes at fixed seed | Stage 0 |
| reference-encoder embeddings (w2v-bert-2.0, MaskGCT, CAMPPlus) | conditioning |
| EnhancedCodec round trip | Stage 1 |
| S2Mel mel output at supplied noise | Stage 1 |
| BigVGAN waveform | vocoder |
| full render, fixed seed / c1 / fixed batch composition | e2e |

Noise is **supplied**, never sampled, wherever a stage is stochastic — the
comparison must isolate the computation from the RNG, as the H3
condition-noise gate does.

## Gates

**Binding: per-stage numerics vs the checkpoint's own remote code**, at the
tolerances the H3 lane already achieves (that port landed 1.6e-7 worst case on
the DiT forward and 4.2e-9 on BigVGAN), plus **token-exact mel codes** out of the
AR talker at fixed seed.

Gate conditions are pinned by upstream's own statement that a seed controls both
AR sampling and per-request CFM noise, and that differing concurrent batch
composition does not guarantee a bit-identical waveform: **fixed seed, c1, fixed
batch composition**. A token-exact e2e waveform gate is therefore not available
and is not claimed.

**Additional ratchet: e2e perceptual.** ASR round-trip through our own Parakeet
(WER against the input text) plus a speaker-similarity band against the reference
clip. This bounds **max relative error from both sides with a measured band**. It
is explicitly NOT a correlation gate — Pearson is scale-invariant and cannot see
a scale error — and NOT a count-based tolerance, which bounds nothing.

**Memory format is checked against the oracle explicitly.** A token gate cannot
see a dtype that is too wide: it stays numerically correct while moving twice the
bytes. Every f32 on this path owes a one-line reason.

**Speed axes:** RTF (audio seconds per wall second), TTFB, peak VRAM/RSS,
startup. Denominator is vLLM-Omni's production configuration, bf16 both sides,
never `--enforce-eager`.

## Dependencies

- **[#633](https://github.com/mudler/vllm.cpp/issues/633) — hard blocker.** The
  oracle is registered at [`.agents/oracles/vllm-omni.md`](../oracles/vllm-omni.md)
  (#650) and that file states the position exactly: `pin = UNPINNED`,
  `gateable = no`, `evidence = #633`. vllm-omni additionally requires vLLM 0.27.0+
  against our 0.26.0.dev0 parity pin. Until it is pinned there is no oracle this
  row can be gated against, and it does not advance past `INVENTORIED`.
- Checkpoint access for `IndexTeam/IndexTTS-2.5` plus the three auxiliary
  encoders, at pinned revisions on the NAS.
- W1 touches a header two shipped lanes already depend on, so it coordinates with both H3 and LTX-2.5 (#435, merged).

## Measured component inventory

Taken from the reference implementation itself (`github.com/index-tts/index-tts`,
cloned 2026-08-13), not estimated. This is what W3-W5 actually contain, and it is
the reason this lane is a campaign rather than a change.

| Reference component | Python LOC | Ours |
|---|---:|---|
| `indextts/gpt` (UnifiedVoice talker) | 17,171 | backbone DONE (W2); the talker head, conditioning and generate loop remain |
| `indextts/s2mel` (CFM/DiT + length regulator + CAMPPlus) | 15,011 | not started |
| `indextts/utils` | 18,265 | not started; much is training-only and will not be ported |
| `indextts/codec` (EnhancedCodec) | 1,930 | not started |
| `indextts/BigVGAN` | 3,740 | 1-D core DONE (W1), shared with H3 and LTX-2.5 |
| `indextts/vqvae` | 395 | not started |
| `Wav2Vec2BertModel` (HF transformers) | external | not started; a Conformer, so `parakeet_encoder.cpp` is partial reuse |

**The pipeline order, read from `infer_v2_5.py`** (`infer_generator`, lines
569-660), which supersedes the recipe page's prose:

1. `SeamlessM4TFeatureExtractor` -> features from the 16 kHz reference clip
2. `Wav2Vec2BertModel` (`semantic_model`) -> `vq_emb`
3. `EnhancedCodec.quantize` -> `semantic_code`, `feat`
4. `CAMPPlus` -> a 192-d global style vector from `feat`
5. `s2mel.models['length_regulator']` -> `prompt_condition`
6. `UnifiedVoice` (GPT-2 talker, `spk_cond_mode="campplus"`) -> mel codes
7. `s2mel` CFM/DiT -> mel, then BigVGAN -> 22.05 kHz waveform

Note step 6: the talker is conditioned on the CAMPPlus style vector, so **CAMPPlus
is upstream of the talker, not a post-hoc speaker check**. Sequencing W3 after W2
but before W5 is therefore forced, not a preference.

**CAMPPlus is not the small brick its file size suggests.** 436 lines define a
2-D conv front end (`FCM`: Conv2d + BatchNorm2d + `BasicResBlock` x4) and an
`xvector` stack of `TDNNLayer` + three `CAMDenseTDNNBlock`s of 12, 24 and 16
layers, each carrying a `CAMLayer` attention, separated by `TransitLayer`s, then
`StatsPool` and a `DenseLayer`. That is 52 dense layers plus batch norms; the
line count is small because the blocks are looped.

## What the SHIPPED checkpoint actually contains

Read from the repository manifest and `config.yaml` itself (not the recipe page,
not the paper). Four findings change the plan, and two of them settle questions
this spec previously left open.

**The model runs at TWO sample rates.** The talker's mel front end is 24 kHz with
100 mel bins; S2Mel and the vocoder work at 22.05 kHz with 80. The OUTPUT is
22.05 kHz. Conflating them yields audio at the wrong speed rather than an error,
so both are pinned in `indextts2_config.h` and gated.

**The language question is SETTLED, and the recipe page was wrong.** The shipped
tokenizer is `multilingual_zh_ja_yue_char_del.tiktoken` — zh, ja, yue. That
matches the vLLM-Omni docs (zh/en/zhen/ja/yue) and contradicts the recipe page's
zh/en/ja/es/ar. Nothing may claim Spanish or Arabic.

**The text tokenizer is TIKTOKEN, not a HuggingFace `tokenizer.json`.** This lane
therefore inherits the constraint already recorded for Kimi-Linear: a tiktoken-only
checkpoint has no `tokenizer.json`, so any path that assumes one is unavailable.

**A Qwen-0.6B EMOTION MODEL ships inside the checkpoint** (`qwen0.6bemo4-merge/`,
with its own `model.safetensors`, tokenizer and config), alongside `feat1.pt` /
`feat2.pt` speaker and emotion matrices and an `emo_condition_module`.

*Corrected from the first reading.* It was recorded here as unscoped work, on the
assumption that a second language model inside a TTS lane implied a second port.
Its safetensors header says otherwise. Read by HTTP range request — 2 MB, no
weights — it is **310 BF16 tensors of stock `Qwen3ForCausalLM`**: hidden 1024, 28
layers, GQA 16 query heads over 8 KV heads at `head_dim` 128, `intermediate_size`
3072, vocab 151936, `tie_word_embeddings: true`. `config.json` names that
architecture literally, this tree registers it in
`src/vllm/model_executor/models/qwen3_dense.cpp`, and
`src/vllm/model_executor/models/qwen3_weights.cpp:168` already has the tying
branch that explains the absent `lm_head.weight`.

So the emotion *language model* needs **no port**. That reduction is pinned by
`tests/scripts/test_indextts2_emotion_arch_covered.py` against a committed
manifest, so if it is re-exported under another name, or our loader is renamed,
or tying is dropped, the claim fails there rather than rotting in this paragraph.

**But the emotion PATH is much larger than that model, and the same day's first
correction understated it.** It was written here as "surrounding wiring:
`feat1.pt` / `feat2.pt` and `emo_condition_module`". Reading `gpt.pth`'s own
pickle header (below) shows the emotion path is two unported networks living
inside the talker checkpoint:

| Group | What it is | Evidence |
|---|---|---|
| `emo_conditioning_encoder` | A **Conformer** encoder at width 512: relative-position MHA carrying `pos_bias_u` / `pos_bias_v` `[4, 128]`, macaron feed-forwards, a conv module with depthwise kernel 15, and a Conv2d-subsampling front end whose `embed.out` is `[512, 261632]` | 38 name patterns in `gpt.pth` |
| `emo_perceiver_encoder` | A **Perceiver resampler**: learned `latents [1, 1024]`, `to_q` / `to_kv` / `to_out`, a GEGLU feed-forward at 2730, `proj_context [1024, 512]` | 9 patterns |
| `emo_layer`, `emovec_layer` | The two projections into the talker: `[1280, 1280]` and `[1280, 1024]` | 4 patterns |

Neither network is ported, and neither is one this tree already has. The lesson
is the one this campaign keeps re-learning: an architecture name settles what a
*model* costs, and settles nothing about what a *checkpoint* contains.

### Loading them: convert offline, never read pickle in the engine

Upstream ships `.pth`, which is a ZIP around a Python pickle. This tree has no
torch-pickle reader, and deliberately does not grow one. Pickle executes
arbitrary code by construction, so a reader in the engine would run a
attacker-controllable program inside the process that serves users, and every
other lane here already loads safetensors or GGUF.

So the conversion is OFFLINE and once:
`scripts/convert-indextts2-checkpoint.py` flattens the nested state dicts with
'.' -- which is exactly the naming the manifest above records, so the converted
names ARE the manifest's names and the manifest checks the conversion -- and
writes safetensors the existing reader can open. Measured on the shipped
checkpoint:

| Source | Tensors kept | Dropped | .pth | .safetensors |
|---|---|---|---|---|
| `gpt.pth` | 456 | 0 | 3108.60 MiB | 3108.45 MiB |
| `codec.pth` | 243 | **729** | 579.16 MiB | **192.99 MiB** |
| `s2mel.pth` | 284 | 0 | 395.69 MiB | 395.63 MiB |

`codec.pth` is **75% optimizer state**: 729 of its 972 tensors are training
residue, and dropping them takes the file from 579 MiB to 193 MiB. That is
dropped loudly, with a count, and the drop prefix is gated from both sides --
every optimizer key must match it, and no weight in `gpt.pth` or `s2mel.pth` may.

The conversion needs torch and 4 GiB of weights, so CI cannot run it.
`tests/scripts/test_indextts2_convert.py` holds the part where a silent mistake
would be unrecoverable -- which tensors survive, under which names -- with fakes
and no torch, because a dropped weight looks exactly like a weight that was
never there.

### What the three .pth checkpoints actually hold

`gpt.pth`, `codec.pth` and `s2mel.pth` are torch ZIPs: one small pickle names
every tensor, and the gigabytes are separate blobs. `scripts/read-torch-manifest.py`
fetches the central directory and that pickle by range request and unpickles it
with a stub Unpickler, so the full manifest of **1712 tensors across 4.0 GiB**
costs a few hundred KB and needs no torch. The record is committed at
`tests/vllm/models/indextts2_pth_manifest.json`.

It confirms four constants in `indextts2_config.h` from a source independent of
`config.yaml`: `kTalkerDim` 1280 is `emo_layer.weight`'s square, `kStyleDim` 192
is `spk_emb_proj.weight`'s input, `kVocosDim` 384 and `kVocosIntermediateDim`
2048 are the codec decoder's ConvNeXt widths, and `kCodecHiddenSize` 1024 is that
decoder's input. `test_indextts2_config_contract.py` compares the header to the
config, which shares its source; `test_indextts2_pth_manifest.py` compares it to
the weights, which does not.

It also names what our ports do NOT model. The reduced-dim gates all pass, and
they pass over a smaller network than the checkpoint holds:

| Where | Unported | Note |
|---|---|---|
| ~~`s2mel.pth` `net.cfm.estimator`~~ | ~~`wavenet.*`~~ | **PORTED** in `wavenet.cpp`, gated against upstream `WN` at reduced dims (3 cases / 133 assertions, 6 mutations caught). Not a conditioning stack but the DiT's FINAL LAYER: the config sets `final_layer_type: wavenet`, which is also what `t_embedder2`, `conv1` and `conv2` belong to |
| ~~same~~ | ~~`skip_linear`, `layers.N.skip_in_linear`~~ | BOTH **PORTED**: the long skip in `dit_tail.cpp`, the per-layer U-Net skip in `dit_skip.cpp`. The routing was RECORDED from upstream's own Transformer rather than read off the formula (`scripts/gen-dit-skip-schedule.py`): at the shipped depth 13, layers 0-5 emit, 7-12 receive LIFO so layer 7 takes layer 5's output, and layer 6 does neither. At EVEN depth there is one more emitter than receiver and the earliest skip is never consumed; we report that rather than correct it |
| ~~same~~ | ~~`t_embedder2`, `conv1`, `conv2`~~ | **PORTED** in `dit_tail.cpp` together with `skip_linear`, `res_projection` and `final_layer`, gated against upstream's own DiT modules end to end (4 cases, 6 mutations caught). Note the coupling upstream hides by setting both to 512: `final_layer` is sized at the WAVENET width but conditioned on `t1` at the DiT width, so the two must be equal. We refuse unequal widths by name |
| ~~same~~ | ~~`cond_projection`, `cond_x_merge_linear`~~; `content_mask_embedder` | **PORTED** in `dit_front.cpp`, both the conditional and the CFG unconditional branch. **`cond_embedder` is DEAD in 2.5**: upstream forces `cond_in_module = cond_projection` and the `content_type` switch that would have selected it is commented out, so the tensor ships and is never read. A port that restored the switch would read a tensor this model does not use |
| `s2mel.pth` `net.length_regulator` | `mask_token`, `embedding`, `content_in_proj` | Our `lenreg` port has the interpolate/GroupNorm/Mish stack and none of these |
| `s2mel.pth` | `net.gpt_layer` | Three weight/bias pairs; unmodeled and unexplained |
| `codec.pth` | `model.encoder.*`, `model.down`, `model.up` | Only the quantizer (`fvq`) and the Vocos-shaped decoder are ported |

`codec.pth` also ships `optimizer.state`, so part of its 0.57 GiB is training
residue rather than weights.

**Two components are NOT in this repository at all**: BigVGAN
(`bigvgan_generator.pt`, fetched into `hf_cache/bigvgan`) and w2v-bert-2.0. They
download separately at first run, so a byte count of this repo understates what a
render needs.

The full manifest is 22 files: `gpt.pth`, `codec.pth`, `s2mel.pth`,
`wav2vec2bert_stats.pt`, `feat1.pt`, `feat2.pt`, the tiktoken vocabulary, the
Qwen emotion directory, and `config.yaml`.

### The emotion vector has TWO paths, and the cheap one is not the ported-looking one

`infer_v2_5.py:668-677` against `:723-726`. This matters because it reorders
what a first render needs.

**Path A, inferred from audio.** `get_emo_conditioning` runs the
`emo_conditioning_encoder` Conformer and the `emo_perceiver_encoder` Perceiver
resampler over an emotion reference clip, then `emovec_layer` and `emo_layer`
project into the talker. That is the ~50 name patterns in `gpt.pth` this spec
records as unported, and it is the path taken when no vector is supplied.

**Path B, SUPPLIED.** When `emo_vector` is given -- eight weights, one per
emotion, either passed in or produced from text by the bundled Qwen model -- the
Conformer and the Perceiver are NOT RUN AT ALL. Instead:

```
weight_vector = tensor(emo_vector)                      # 8 emotions
random_index  = [find_most_similar_cosine(style, m) for m in spk_matrix]
emo_matrix    = [m[i] for i, m in zip(random_index, emo_matrix)]
emovec_mat    = weight_vector.unsqueeze(1) * emo_matrix
```

`spk_matrix` and `emo_matrix` are `feat1.pt` and `feat2.pt`, both shipped in the
checkpoint and both small. So Path B is a cosine similarity, a row selection and
a weighted sum -- no new network at all.

**Consequence for sequencing.** The emotion Conformer and Perceiver are NOT on
the critical path to a first render; they are what a caller needs when the
emotion should be *inferred from a clip* rather than stated. A render can supply
the vector directly. That moves both networks behind the render rather than in
front of it, and moves `feat1.pt` / `feat2.pt` in front.

### The reference-audio path, read from the running code

`infer_v2_5.py:280-295` and `:630`. Three facts here each produce a model that
runs and sounds wrong, and none is visible from the architecture:

**The features come from HIDDEN STATE 17, not the final layer.**
`get_emb` takes `vq_emb.hidden_states[17]` out of `Wav2Vec2BertModel`. A port
that used the encoder's output would get features of the right shape from the
right model, conditioned on the wrong representation. Our `w2vbert::EncoderStack`
returns the final state, so consuming it for this path needs an intermediate tap.

**They are then normalized by STORED statistics**, not per-utterance ones:
`feat = (feat - semantic_mean) / semantic_std`, where both come from
`wav2vec2bert_stats.pt` in the checkpoint (`w2v_stat` in `config.yaml`). Using
per-utterance statistics is the natural assumption and is wrong.

**The feature extractor is KALDI-style, and this tree's existing one is not.**
`SeamlessM4TFeatureExtractor` is fully specified by
`transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py`:

| Parameter | Value |
|---|---|
| pre-scale | waveform x 2^15 (Kaldi expects 16-bit integers) |
| window | `povey`, non-periodic, length 400 |
| frame / hop / FFT | 400 / 160 / 512 |
| preemphasis | 0.97, with `remove_dc_offset` |
| power, log | 2.0, `log`, `mel_floor` 1.192092955078125e-07 |
| mel bins | 80, then a **stride-2 stack** giving 160 columns |

Measured: 4000 samples at 16 kHz produce 12 frames of 160.

That differs from `Ltx2WaveformToLogMel`, which is the Slaney/torchaudio kind
with no preemphasis and no DC removal. Reusing it would be the mistake this spec
elsewhere warns about -- a gate that passes because both arms call the same
helper proves consistency, not correctness -- so the extractor is a NEW unit. It is now **ported** in
`w2v_fbank.cpp`; the `hidden_states[17]` tap and the
stored-statistics normalization are ported too (`w2vbert::EncoderHiddenState`
and `w2vbert::NormalizeWithStats`), so this path is complete from waveform to
semantic codes.

**What IS ported on this path**: the w2v-bert Conformer itself
(`w2vbert::FeatureProjection` and `w2vbert::EncoderStack`, including the
relative-key attention, the causal left-only conv pad and the absent final
norm), the semantic codec encoder and quantizer (`codec_encoder`, `fvq`), and
CAMPPlus. What is missing between a WAV file and those is exactly the feature
extractor above.

## Work breakdown

| W | Work | Depends on |
|---|---|---|
| W1 | Relocate the ALREADY-SHARED vocoder core out of the `minimax_h3.h` header into a neutral home, plus WAV. Smaller than it first looked: the sharing exists and is gated by two suites, so this is a rename/relocate with a live precedent, not a generalization | — |
| W2 | GPT-2 talker backbone, additive, on the existing decode framework | — |
| W3 | Reference-encoder path (w2v-bert-2.0, MaskGCT, CAMPPlus) | — |
| W4 | EnhancedCodec + S2Mel CFM/DiT (13 blocks, hidden 512, 8 heads, in_channels 80) | W1 |
| W5 | Compose the render; goldens per stage | W2-W4, #633 |
| W6a | `SpeechEngine` seam + ABI v19 entry points + `test_capi` section | W5 |
| W6b | `/v1/audio/speech` + `/v1/audio/voices` on `ApiServer`, routed through the seam; example as a thin ABI client | W6a |
| W7 | Speed axes vs the omni oracle | W5, #633 |

W1-W4 are gateable against frozen upstream goldens without a pin, since they
compare against upstream modules executed offline. W5's e2e claim, W7 entirely,
and any parity statement need #633. W6a/W6b need no oracle at all: a seam and an
ABI are gateable against their own contract, which is why they are not deferred
behind the pin.

## Risks/decisions

- **Quantization: bf16 for v1, arms refused and owed.** vLLM-Omni ships no
  quantized IndexTTS arm, the same situation recorded for H3, and the model fits
  GB10 unquantized. Every unimplemented quant arm refuses at load naming the
  missing piece and is recorded as owed — never left to be discovered.
- **We mirror vLLM-Omni, which itself deviates from IndexTeam.** Upstream's
  Stage 0 uses plain vLLM sampling and deliberately does not reproduce the
  official `num_beams=3` beam search. Mirroring vLLM-Omni is the rule; the
  divergence from the reference implementation is recorded, not silently
  inherited.
- **Language claims disagree between upstream surfaces.** The recipe page says
  zh/en/ja/es/ar; the vllm-omni docs say zh/en/zhen/ja/yue. Resolve against
  `indextts2_5.yaml` before anything reaches `docs/FEATURES.md`. Shipping the
  wrong list is a user-visible false claim.
- **Licensing.** The checkpoint is under a custom bilibili-model-license, not
  Apache-2.0. Check it before any fixture or golden derived from those weights
  lands in-tree.
- **The shared vocoder core is gated at H3's and LTX-2.5's hyperparameters, not
  this model's.** Reuse is right, and the two existing consumers prove the seam
  holds, but it must be re-gated at IndexTTS-2.5's own configuration against the
  checkpoint's own remote code. A gate that passes because both arms call the
  same helper proves consistency, not correctness.
- **Stop** if the reference-encoder path cannot be reproduced without executing
  checkpoint Python: that would make the mandatory conditioning path
  un-portable, which is a scope question, not an implementation detail.

## Now

**Text renders to audio on the real checkpoints.** `test_indextts2_e2e`
tokenizes with the shipped vocabulary, runs the 24-layer talker to mel codes,
and drives those through the length regulator, a CFG'd CFM Euler loop over the
13-layer S2Mel estimator with a real rotary table, and BigVGAN.
`vllm_synthesize` does the same from C and returns 8192 samples for
"hello world".

**NOTHING HERE IS A CORRECTNESS CLAIM.** Every gate asserts STRUCTURE -- finite,
bounded, correctly-shaped, not silence, not a rail. vLLM-Omni is unpinned
(#633), so no one can say whether any of it resembles what upstream produces.
That is the single largest open item and no code in this repository moves it.

### What is DONE

The feature extractor, the w2v-bert Conformer with the `hidden_states[17]` tap
and stored-statistics normalization, the semantic codec encoder and FVQ,
CAMPPlus, the stated-emotion selector and its banks, the tiktoken reader, the
talker prompt and greedy loop, the S2Mel DiT (front end, stack with U-Net
skips, wavenet tail), the length regulator, BigVGAN, the offline converter,
loaders for every artifact, the `SpeechEngine` implementation, and
`indextts2::Render`. The ABI entry points and `/v1/audio/speech` came from
MUSIC3's W6 (#799) and this family reaches both through
`GlobalSpeechRegistry`.

### What is NOT

1. **The reference clip does not condition anything.** `Synthesize` validates
   it and refuses without it, but the encoders that would turn it into speaker
   and semantic conditioning are ported and NOT WIRED -- the conditioning rows
   are zeros. This is the first thing to fix.

   The two checkpoints it needs are now STAGED, so nobody has to find them
   again:

   | Artifact | Where | Note |
   |---|---|---|
   | `facebook/w2v-bert-2.0` | `$CHECKPOINT_ROOT/w2v-bert-2.0` | ships `model.safetensors` already; NO conversion needed |
   | `funasr/campplus` | `$CHECKPOINT_ROOT/IndexTTS-2.5-safetensors/campplus.safetensors` | converted from `campplus_cn_common.bin`, 937 tensors, `head.*` / `xvector.*` naming that matches `campplus.h` |

   What remains is loaders binding those onto the ported `w2vbert` and
   `campplus` structs, and then feeding the result into the three conditioning
   rows `talker::PrepareInputs` already accepts.
2. **The INFERRED emotion path** (`emo_conditioning_encoder` Conformer,
   `emo_perceiver_encoder` Perceiver) is unported. A caller can STATE the
   emotion instead, which is why this sits behind a render rather than in front.
3. **`/v1/audio/speech` is unproven by request.** The wiring was read, not
   exercised: `server_main.cpp` loads from the same registry this family
   registers into. A live check needs a TEXT model too -- `vllm-server` refuses
   `--speech-model` without `--model`, so the speech route cannot be served
   standalone. Worth knowing before someone tries.
4. **The `exact` flag on `tiktoken::Pretokenize` is not proven load-bearing**: a
   mutation disabling every range check still passes. Recorded in the test.
5. **W7 speed**, which additionally needs #633.

### A process note for whoever continues

This lane was built while several other agents merged in parallel, and twice a
gap named here had already been closed by someone else's work -- once after a
whole duplicate ABI slice had been written, which only a merge conflict caught.
Re-verify each item against `origin/main` before implementing it. The protocol
already says so; on this repository it is not optional.
