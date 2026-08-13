# SPEC — IndexTTS-2.5, the first audio-GENERATING lane

**Rows:** `MODEL-MM-indextts2-index-tts2-talker-for-conditional-generation`,
`MODEL-MM-indextts2-index-tts2-s2-mel-decoder`
**Issue:** [#634](https://github.com/mudler/vllm.cpp/issues/634)
**State:** `SPIKE` — scoped, not implemented, and blocked on
[#633](https://github.com/mudler/vllm.cpp/issues/633).

## Scope

Port IndexTTS-2.5 — upstream-supported at
`https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5` and served by vLLM-Omni — so
that text plus a reference clip renders 22.05 kHz speech through our own engine
and the OpenAI-compatible speech surface.

This is the project's **first audio-generating model**. Every audio path we ship
today consumes audio (Parakeet, Voxtral, `audio_processor.cpp`); nothing
synthesizes it, and `/v1/audio/speech` does not exist.

In scope: both registered architectures, the mandatory reference-audio
conditioning path, bf16 inference, the speech serving surface and its ABI, and
the gates below. Out of scope: quantized arms (see Risks/decisions), streaming
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
| BigVGAN | `src/vllm/model_executor/models/minimax_h3_audio_vae.cpp` | already gated at 4.2e-9 vs the checkpoint's own remote code; needs generalizing out of the `minimax_h3_*` namespace, which converges with [#435](https://github.com/mudler/vllm.cpp/issues/435) |
| WAV serialization | `minimax_h3_wav.cpp` | channel-major → interleaved, clamped |
| Flow-matching denoise loop, AdaLN/timestep machinery | the H3 lane | S2Mel is the same shape of computation |
| Conformer encoder | `parakeet_encoder.cpp` | w2v-bert-2.0 is a Conformer |
| Mel front end | `whisper_audio.cpp` | — |
| Generation serving seam | `/v1/videos`, `/v1/videos/sync` | the template for `/v1/audio/speech` |
| GPT-2-family backbone | `opt.cpp` | learned absolute positions + LayerNorm; the talker is an additive delta, not a fresh transformer |

New from scratch: EnhancedCodec, S2Mel, and the three reference encoders.

## Port map

| Stage | Upstream | Ours | Kind |
|---|---|---|---|
| AR talker | `indextts2_talker.py` | additive GPT-2 arch routed through `ModelRegistry::Forward` + `dense_attn::AttnBlock` + on-device sampling | new, small |
| Reference encoders | aux checkpoints under `hf_cache/` | partial reuse of `parakeet_encoder.cpp` / `whisper_audio.cpp` | new, largest |
| EnhancedCodec | `indextts2/` | — | new |
| S2Mel CFM/DiT | `indextts2_s2mel_decoder.py` | H3 denoise loop | reuse |
| BigVGAN | — | `minimax_h3_audio_vae.cpp`, generalized | reuse |
| WAV 22.05 kHz | — | `minimax_h3_wav.cpp` | reuse |
| `/v1/audio/speech`, `/v1/audio/voices` | `vllm_omni/entrypoints/openai/` | additive routes on `ApiServer` + `include/vllm.h` ABI bump | new |

Examples and servers stay ABI clients; no internal headers.

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

- **[#633](https://github.com/mudler/vllm.cpp/issues/633) — hard blocker.** There
  is no vllm-omni parity pin, and vllm-omni additionally requires vLLM 0.27.0+
  against our 0.26.0.dev0 core pin. Until that lands there is no oracle this row
  can legally be gated against. This row does not proceed past `SPIKE` without it.
- Checkpoint access for `IndexTeam/IndexTTS-2.5` plus the three auxiliary
  encoders, at pinned revisions on the NAS.
- The BigVGAN generalization touches the H3 lane, so it coordinates with #435.

## Work breakdown

| W | Work | Depends on |
|---|---|---|
| W1 | Generalize BigVGAN + WAV out of the `minimax_h3_*` namespace behind a shared seam | — |
| W2 | GPT-2 talker backbone, additive, on the existing decode framework | — |
| W3 | Reference-encoder path (w2v-bert-2.0, MaskGCT, CAMPPlus) | — |
| W4 | EnhancedCodec + S2Mel CFM/DiT on the H3 denoise loop | W1 |
| W5 | Compose the render; goldens per stage | W2-W4, #633 |
| W6 | `/v1/audio/speech` + `/v1/audio/voices` + ABI bump | W5 |
| W7 | Speed axes vs the omni oracle | W5, #633 |

W1-W4 are gateable against frozen upstream goldens without a pin, since they
compare against upstream modules executed offline. W5's e2e claim, W7 entirely,
and any parity statement need #633.

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
- **The reused BigVGAN is gated for H3's configuration, not this one.** Reuse is
  the plan, but the shared seam must be re-gated at IndexTTS-2.5's own
  hyperparameters. A gate that passes because both arms call the same helper
  proves consistency, not correctness.
- **Stop** if the reference-encoder path cannot be reproduced without executing
  checkpoint Python: that would make the mandatory conditioning path
  un-portable, which is a scope question, not an implementation detail.

## Now

`SPIKE`, unclaimed, blocked on
[#633](https://github.com/mudler/vllm.cpp/issues/633). Nothing implemented. W1-W4
can start against frozen goldens once the row is claimed; no parity or speed
claim is possible until the omni pin exists.
