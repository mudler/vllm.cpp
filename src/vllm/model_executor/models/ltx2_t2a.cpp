// `T2AOneStagePipeline` (ltx-pipelines t2a_one_stage.py:43, `__call__` at :109)
// at Lightricks/LTX-2 @ fd4ded7f. See ltx2_t2a.h for the port map and for the
// three details that fail silently if guessed.
//
// Row LTX25-T2A-ONE-STAGE, issue #1005.

#include "vllm/model_executor/models/ltx2_t2a.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) { throw std::runtime_error("ltx2 t2a: " + why); }

uint64_t DigestF32(const std::vector<float>& values) {
  uint64_t h = 1469598103934665603ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  const size_t n = values.size() * sizeof(float);
  for (size_t i = 0; i < n; ++i) {
    h ^= bytes[i];
    h *= 1099511628211ULL;
  }
  return h;
}

double AbsMax(const std::vector<float>& values) {
  double m = 0.0;
  for (const float v : values) m = std::max(m, std::abs(static_cast<double>(v)));
  return m;
}

// `to_denoised` (ltx-core utils.py:39-52) as `X0Model.forward` applies it
// (model.py:590-604): the DiT emits a VELOCITY and everything downstream — the
// guider AND the sampler — wants the x0 prediction. It is therefore applied to
// EVERY PASS, on the way out of the forward, and not once to the guider's
// output: see ltx2_t2a.h item 4 (#1039).
//
// Identical arithmetic to the joint driver's own `ToDenoised`; kept here rather
// than shared because the joint one is a static in `ltx2_video.cpp`'s anonymous
// namespace and hoisting it would move lines above that file's gated READER
// ANCHORS list for no behavioural reason.
std::vector<float> ToDenoised(const std::vector<float>& sample, const std::vector<float>& velocity,
                              const std::vector<float>& timesteps, int64_t tokens, int64_t width) {
  VT_CHECK(velocity.size() == sample.size(), "ltx2 t2a: the velocity is the wrong size");
  std::vector<float> out(sample.size());
  for (int64_t t = 0; t < tokens; ++t) {
    const float sigma = timesteps[static_cast<size_t>(t)];
    for (int64_t c = 0; c < width; ++c) {
      const size_t i = static_cast<size_t>(t * width + c);
      out[i] = sample[i] - sigma * velocity[i];
    }
  }
  return out;
}

}  // namespace

std::vector<uint8_t> Ltx2StgBlockMask(const std::vector<int64_t>& stg_blocks, int64_t num_layers) {
  VT_CHECK(num_layers > 0, "ltx2 t2a: num_layers must be positive");
  std::vector<uint8_t> mask(static_cast<size_t>(num_layers), 0);
  // `blocks is None` upstream means EVERY block (perturbations.py:19-33). An
  // EMPTY list is not that: the CLI's `--audio-stg-blocks` with no values yields
  // an empty list (utils/args.py:1107-1113, `nargs="*"`), and an empty list
  // perturbs nothing. Conflating the two would turn "STG off" into "STG
  // everywhere" for a caller who typed the flag and no numbers.
  for (const int64_t b : stg_blocks) {
    if (b < 0 || b >= num_layers) {
      Fail("STG block index " + std::to_string(b) + " is outside [0, " +
           std::to_string(num_layers) +
           "). Upstream indexes `self.transformer_blocks` by it and raises; ignoring it here "
           "would run an UNPERTURBED pass and report it as STG, which is a different guidance "
           "delta on a render that still finishes");
    }
    mask[static_cast<size_t>(b)] = 1;
  }
  return mask;
}

Ltx2T2aResult Ltx2T2aGenerate(const Ltx2T2aRequest& req) {
  VT_CHECK(req.dit_params != nullptr && req.dit_weights != nullptr && req.audio_cfg != nullptr &&
               req.audio_weights != nullptr && req.vocoder_cfg != nullptr &&
               req.vocoder_weights != nullptr && req.noise != nullptr,
           "ltx2 t2a: the request is missing a required borrowed pointer");
  const Ltx2DitParams& params = *req.dit_params;

  if (req.num_frames < 1) {
    // `require_num_frames_source` (utils/blocks.py:894-905) — upstream's own
    // fast refusal, raised at the TOP of `__call__` before prompt encoding, so
    // an unsatisfiable auto-duration costs no work.
    //
    // WHAT IS *NOT* THE REASON: not the duration head's ARITHMETIC.
    // `Ltx2DurationHeadForward` is ported and gated, including the audio-only
    // case this pipeline would use — `test_ltx2_pipeline` runs it against
    // `kLtx2DurAudioOnlyGolden`, generated from executed upstream. What is
    // missing is a CONSTRUCTED head: nothing in this engine builds one, and
    // `duration_head_path` is refused by name at load (#611). So the arithmetic
    // exists and the object does not.
    Fail("this request carries no frame count, and audio-only generation derives its DURATION "
         "from one: `AudioLatentShape.from_video_pixel_shape` reads `frames` and `fps` off the "
         "pixel shape (ltx-core types.py:184-200), which is why upstream passes a 512x512 "
         "PLACEHOLDER resolution and a real frame count (t2a_one_stage.py:37-40, :163-166). "
         "Auto duration needs a DurationHead this engine does not construct. Pass num_frames or "
         "duration_seconds.");
  }
  if (req.frame_rate <= 0.0) Fail("frame_rate must be positive");
  if (req.context == nullptr || req.context_tokens < 1) {
    Fail("no audio conditioning was supplied; `ctx_p.audio_encoding` is what this pipeline "
         "cross-attends over (t2a_one_stage.py:134)");
  }

  // ── the audio latent shape (types.py:164-200) ─────────────────────────────
  //
  // `AudioLatentShape.from_video_pixel_shape` takes `frames` and `fps` from the
  // pixel shape and NOTHING else — height and width are unused, which is exactly
  // why upstream can pass a placeholder for them.
  const Ltx2AudioPatchifierParams ap;
  const double latents_per_second = static_cast<double>(ap.sample_rate) /
                                    static_cast<double>(ap.hop_length) /
                                    static_cast<double>(ap.audio_latent_downsample_factor);
  Ltx2AudioLatentShape ashape;
  ashape.batch = 1;
  ashape.channels = 8;   // types.py:184-200 defaults, asserted against the DiT below
  ashape.mel_bins = 16;
  ashape.frames = static_cast<int64_t>(
      std::llround(static_cast<double>(req.num_frames) / req.frame_rate * latents_per_second));
  if (ashape.frames < 1) {
    Fail("the audio latent resolved to zero frames for " + std::to_string(req.num_frames) +
         " frames at " + std::to_string(req.frame_rate) + " fps");
  }
  // The latent's channels x mel_bins IS the DiT's audio stream width, and a
  // mismatch reinterprets the spectrogram rather than failing. Checked against
  // BOTH factors and not only their product: a (16, 8) latent is the same width
  // as an (8, 16) one and unpatchifies into a different tensor.
  if (ashape.channels * ashape.mel_bins != params.audio_in_channels) {
    Fail("the audio latent is " + std::to_string(ashape.channels) + " x " +
         std::to_string(ashape.mel_bins) + " = " +
         std::to_string(ashape.channels * ashape.mel_bins) +
         " wide (types.py:184-200) but this DiT's audio stream takes " +
         std::to_string(params.audio_in_channels));
  }
  if (params.audio_cross_attention_dim < 1) Fail("this DiT declares no audio context width");

  Ltx2T2aResult result;
  result.latent_frames = ashape.frames;

  const int64_t width = ashape.channels * ashape.mel_bins;
  const int64_t tokens = ashape.frames;  // AudioPatchifier(patch_size=1)
  result.audio_tokens = tokens;

  // ── the schedule (t2a_one_stage.py:141-143) ───────────────────────────────
  //
  // `LTX2Scheduler()` is HARD-CODED at `:67`, so there is no scheduler-kind
  // question here and no distilled sigma table: the recipe carries none and this
  // computes them.
  //
  // THE TOKEN COUNT IS THE SCHEDULER'S OWN DEFAULT, NOT THE AUDIO LATENT'S, and
  // this is the detail a re-derivation gets wrong. `execute` takes an OPTIONAL
  // `latent` and falls back to `default_number_of_tokens = MAX_SHIFT_ANCHOR`
  // when it is absent (schedulers.py:29, :32) — and `t2a_one_stage.py:141` calls
  // `self._scheduler.execute(steps=num_inference_steps)` with no latent at all.
  // The joint video driver in `ltx2_video.cpp` passes its own `target_tokens`
  // and is right to, because the pipelines it mirrors pass a shape; copying that
  // here would move `sigma_shift` by `(tokens - MAX_SHIFT_ANCHOR) * mm`
  // (schedulers.py:36-38) and bend every sigma in the schedule. Nothing about
  // the render's length, its channel count or its finiteness could see it.
  //
  // Passing 0 is `Ltx2SigmaSchedule`'s own spelling for "take the default"
  // (ltx2_pipeline.h), so this is the fallback rather than a substitute for it.
  const int64_t steps = req.steps;
  if (steps < 1) Fail("num_inference_steps resolved to " + std::to_string(steps));
  const std::vector<float> sigmas = Ltx2SigmaSchedule(steps, /*tokens=*/0);
  const int64_t sigma_count = static_cast<int64_t>(sigmas.size());
  VT_CHECK(sigma_count >= 2, "ltx2 t2a: the schedule needs at least one step");

  // ── the guider (t2a_one_stage.py:149-152) ─────────────────────────────────
  const Ltx2MultiModalGuiderParams& g = req.guidance;
  const bool want_uncond = g.DoUnconditionalGeneration();
  const bool want_perturbed = g.DoPerturbedGeneration();
  if (g.DoIsolatedModalityGeneration()) {
    Fail("isolated-modality guidance (`modality_scale` = " + std::to_string(g.modality_scale) +
         ") asks for a fourth forward over the OTHER modality, and this pipeline has no other "
         "modality to run it over. Upstream pins `modality_scale` to 1.0 for exactly this "
         "reason and says so: \"Audio-only generation has no video modality, so the "
         "video->audio (v2a) cross-modal guidance is meaningless here. 1.0 disables it\" "
         "(t2a_one_stage.py:200-202). Use 1.0.");
  }
  if (want_uncond && req.negative_context == nullptr) {
    Fail("the guider asks for an unconditional pass (`cfg_scale` = " +
         std::to_string(g.cfg_scale) +
         ") and no negative conditioning was supplied. Upstream's guider carries a "
         "`negative_context` and the CLI always fills it (t2a_one_stage.py:151, :193). "
         "Substituting a zero tensor would make the CFG delta `cfg_scale * cond`, which is a "
         "different render and not a missing one. Supply a negative prompt, or set the scale "
         "to 1.0.");
  }
  std::vector<uint8_t> stg_mask;
  if (want_perturbed) {
    stg_mask = Ltx2StgBlockMask(g.stg_blocks, params.num_layers);
    for (int64_t b = 0; b < params.num_layers; ++b) {
      if (stg_mask[static_cast<size_t>(b)] != 0) result.perturbed_blocks.push_back(b);
    }
    if (result.perturbed_blocks.empty()) {
      Fail("the guider asks for a perturbed pass (`stg_scale` = " + std::to_string(g.stg_scale) +
           ") and `stg_blocks` names no block, so the perturbed forward would be identical to "
           "the conditional one and the STG delta would be exactly zero — a full extra forward "
           "per step that changes nothing. Name the blocks, or set the scale to 0.");
    }
  }
  Ltx2DitPerturbation perturbation;
  perturbation.audio_self_attn = stg_mask;

  // ── the state (helpers.py:428-447; ModalitySpec(context=...) alone at :168) ─
  //
  // No initial latent and no freeze: T2A's audio `ModalitySpec` carries a
  // context and nothing else (t2a_one_stage.py:168), so the denoise mask is all
  // ones and there is no `clean` to blend back.
  //
  // THE NOISE IS UNIT VARIANCE, NOT SCALED BY `sigmas[0]`, and this draft scaled
  // it until the chain was read. `ModalitySpec.noise_scale` defaults to 1.0
  // (utils/types.py:110), `create_noised_state` forwards it (helpers.py:434,
  // :443) and `GaussianNoiser.__call__` is `torch.lerp(latent, noise,
  // noise_scale)` (noisers.py:31) — at 1.0 the state IS the noise. A reader who
  // knows other flow-matching samplers will expect the scaling, so the absence
  // is written down.
  //
  // AND THE TWO FORMS AGREE HERE, WHICH IS WHY NO TEST SEPARATES THEM.
  // MEASURED: a mutation adding `for (float& v : latent) v *= sigmas[0];` left
  // the focused gate at 6 cases / 484 assertions / exit 0. That is not a blind
  // instrument — it is an identity. `LTX2Scheduler` starts at `linspace(1, 0,
  // steps + 1)[0] == 1`; the shift map sends 1 to `exp(s)/(exp(s) + (1/1 - 1))`
  // which is exactly 1 (schedulers.py:41-45); and the stretch sends it to
  // `1 - (1 - 1)/scale_factor`, again exactly 1 (`:47-55`). So `sigmas[0]` is
  // 1.0 for EVERY step count, and the multiply is a no-op.
  //
  // That identity is GATED rather than left as this comment's word, in
  // `test_ltx2_video`'s "the schedule starts at exactly 1.0". If upstream ever
  // moves the first sigma off 1, that gate fires and this line becomes a real
  // difference — which is the point of pinning it rather than pinning the
  // mutation's survival.
  std::vector<float> latent = req.noise->Draw(tokens * width);
  const std::vector<float> positions_f = Ltx2AudioPatchTimings(ashape, ap);
  const std::vector<double> positions(positions_f.begin(), positions_f.end());

  // ── the denoise loop (samplers.py:39-79) ──────────────────────────────────
  //
  // `last_denoised_audio` (utils/denoisers.py:85-91): a step the guider SKIPS
  // reuses the previous step's denoised prediction instead of running a forward.
  std::vector<float> last_denoised;
  for (int64_t step = 0; step + 1 < sigma_count; ++step) {
    const float sigma = sigmas[static_cast<size_t>(step)];
    // Every token carries the schedule's own sigma: the mask is all ones, so
    // `timesteps_from_mask` is a constant fill (helpers.py:466-503).
    const std::vector<float> timesteps(static_cast<size_t>(tokens), sigma);

    Ltx2ModalityInput ain;
    ain.batch = 1;
    ain.tokens = tokens;
    ain.context_tokens = req.context_tokens;
    ain.latent = latent.data();
    ain.timesteps = timesteps.data();
    ain.sigma = &sigma;
    ain.positions = positions.data();
    ain.context = req.context;

    // `should_skip_step` (guiders.py:287-291). `skip_step` defaults to 0, which
    // never skips, so this is reachable only from an explicit request.
    //
    // A SKIPPED STEP RUNS NO FORWARD AT ALL AND REUSES THE PREVIOUS STEP'S
    // DENOISED PREDICTION. This draft ran the CONDITIONAL forward and used it,
    // which is a plausible reading of "skip the guidance" and is not what
    // upstream does: `_guided_denoise` returns
    // `DenoisedLatentResult.result_or_none(denoised=last_denoised_audio)` when
    // every guider skips (`utils/denoisers.py:85-91`), before it assembles a
    // single pass. The difference is a whole DiT forward per skipped step and a
    // different trajectory, on a render that finishes either way.
    //
    // `step == 0` can never skip — `0 % (skip_step + 1)` is 0 — so
    // `last_denoised` is always populated by the time this branch is taken. The
    // guard is kept anyway, because "the arithmetic makes it impossible" is
    // exactly the reasoning that a later change to `ShouldSkipStep` would
    // silently invalidate, and the failure would be a read of an empty vector.
    const bool skip = g.ShouldSkipStep(step);
    if (skip && last_denoised.empty()) {
      Fail("step " + std::to_string(step) +
           " is a skipped step and no earlier step produced a denoised prediction to reuse. "
           "`should_skip_step` is `step % (skip_step + 1) != 0` (guiders.py:287-291), which is "
           "false at step 0, so this is unreachable through the request surface and is a defect "
           "rather than a bad request");
    }
    if (skip) {
      latent = Ltx2EulerStep(latent.data(), last_denoised.data(), sigmas.data(), sigma_count, step,
                             static_cast<int64_t>(latent.size()));
      continue;
    }

    // THE VIDEO STREAM IS `nullptr`, NOT A DISABLED ONE. See ltx2_t2a.h item 1:
    // upstream's `run_v2a` tests PRESENCE (transformer.py:269), so a
    // present-but-disabled stream still feeds video->audio cross attention.
    //
    // AND EVERY PASS IS CONVERTED TO X0 *HERE*, BEFORE THE GUIDER SEES IT. This
    // lambda is `X0Model` (model.py:590-604): upstream never hands the denoiser
    // the raw velocity model, it hands `X0Model(builder.build(...))`
    // (utils/blocks.py:480-482), so `_guided_denoise`'s
    // `all_v, all_a = transformer(...)` at utils/denoisers.py:188 already
    // carries DENOISED tensors and `audio_guider.calculate(...)` at `:203`
    // combines those. See ltx2_t2a.h item 4 for why converting once after the
    // guider instead is a different function on the DEFAULT arm (#1039).
    //
    // EVERY FORWARD GOES THROUGH THIS ONE LAMBDA, and that is what makes
    // `video_stream_present` an OBSERVATION rather than a restatement. Written
    // as `result.video_stream_present = false` beside a `nullptr` literal it
    // would be a comment that compiles: a build that started passing a stream
    // would report `false` and stay green. Derived at the call, a mutation that
    // hands any forward a video stream flips it.
    const auto x0_model = [&](const Ltx2ModalityInput* video, const Ltx2ModalityInput* audio,
                              const Ltx2DitPerturbation* p,
                              std::vector<float>* velocity_out = nullptr) {
      if (video != nullptr) result.video_stream_present = true;
      const Ltx2DitOutputs out = Ltx2DitForward(req.device, params, *req.dit_weights, video, audio,
                                                req.compute_dtype, /*cache=*/nullptr, p);
      // The RAW velocity, before the conversion, recorded only where a caller
      // asked for it. It is the other half of the pair that makes "which space
      // did the guider combine" an arithmetic question — see the header.
      if (velocity_out != nullptr) *velocity_out = out.audio;
      // `to_denoised(audio.latent, ax, audio.timesteps)` (model.py:603).
      return ToDenoised(latent, out.audio, timesteps, tokens, width);
    };

    const std::vector<float> cond = x0_model(
        /*video=*/nullptr, &ain, /*p=*/nullptr,
        step == 0 ? &result.first_step_velocity : nullptr);
    ++result.cond_forwards;

    std::vector<float> denoised = cond;
    if (want_uncond || want_perturbed) {
      std::vector<float> uncond_text;
      std::vector<float> uncond_perturbed;
      if (want_uncond) {
        Ltx2ModalityInput nin = ain;
        nin.context = req.negative_context;
        // The velocity of THIS arm, recorded beside its x0 exactly as the
        // conditional pass's is. Recorded per arm rather than once, because the
        // conversion is per pass and a claim made about "every pass" from one
        // recorded pass is a claim about a third of them (see the header).
        uncond_text = x0_model(/*video=*/nullptr, &nin, /*p=*/nullptr,
                               step == 0 ? &result.first_step_uncond_velocity : nullptr);
        if (step == 0) result.first_step_uncond = uncond_text;
        ++result.uncond_forwards;
      }
      if (want_perturbed) {
        // The POSITIVE context with the self-attention perturbed — upstream
        // perturbs the MODEL, never the conditioning (guiders.py:244-273 takes
        // `uncond_perturbed` from a forward whose `perturbations` differ and
        // whose context does not).
        uncond_perturbed = x0_model(/*video=*/nullptr, &ain, &perturbation,
                                    step == 0 ? &result.first_step_perturbed_velocity : nullptr);
        if (step == 0) result.first_step_perturbed = uncond_perturbed;
        ++result.perturbed_forwards;
      }
      denoised = Ltx2MultiModalGuidance(g, cond.data(),
                                        want_uncond ? uncond_text.data() : nullptr,
                                        want_perturbed ? uncond_perturbed.data() : nullptr,
                                        /*uncond_modality=*/nullptr,
                                        static_cast<int64_t>(cond.size()));
    }

    // Kept for the next step's `should_skip_step` branch, which reuses it rather
    // than recomputing (utils/denoisers.py:85-91).
    if (step == 0) {
      result.first_step_latent = latent;
      result.first_step_cond = cond;
      result.first_step_denoised = denoised;
      result.first_step_sigma = static_cast<double>(sigma);
    }
    last_denoised = std::move(denoised);
    // `EulerDiffusionStep()` — `DiffusionStage.__call__`'s own default
    // (utils/blocks.py:524-527), which T2A does not override (it passes no
    // `stepper`, t2a_one_stage.py:154-170). The ancestral sampler that
    // `distilled.py` selects for generation 2.5 reaches this pipeline through
    // nothing.
    latent = Ltx2EulerStep(latent.data(), last_denoised.data(), sigmas.data(), sigma_count, step,
                           static_cast<int64_t>(latent.size()));
    // What the sampler WROTE, recorded after the step rather than derived from
    // what was recorded before it. It is the only observable that says which
    // tensor `Ltx2EulerStep` was actually handed: a second `ToDenoised` applied
    // to `denoised` on the way in leaves every other field here untouched.
    if (step == 0) result.first_step_next_latent = latent;
  }

  result.latent_digest = DigestF32(latent);
  result.latent_absmax = AbsMax(latent);

  // `clear_conditioning` + `unpatchify` (blocks.py:575-580). There is no
  // conditioning item on this path and nothing appended, so the clear is an
  // identity here — stated rather than called, because calling a no-op would
  // suggest an append this pipeline cannot make.
  const std::vector<float> volume = Ltx2AudioUnpatchify(latent.data(), ashape);

  // ── the decode (t2a_one_stage.py:172) ─────────────────────────────────────
  const Ltx2AudioSpectrogram mel = Ltx2AudioDecoderForward(
      *req.audio_cfg, *req.audio_weights, volume, ashape.channels, ashape.frames, ashape.mel_bins);
  result.waveform = Ltx2VocoderWithBweForward(*req.vocoder_cfg, *req.vocoder_weights, mel.data,
                                              mel.channels, mel.frames, mel.mel_bins,
                                              &result.samples_per_channel);
  result.channels = mel.channels;
  result.sample_rate = req.vocoder_cfg->output_sampling_rate;
  return result;
}

}  // namespace vllm
