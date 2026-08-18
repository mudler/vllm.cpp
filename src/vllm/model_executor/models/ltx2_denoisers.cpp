// `_guided_denoise` (ltx-pipelines utils/denoisers.py:61-211) at
// Lightricks/LTX-2 @ fd4ded7f. See ltx2_denoisers.h for the four things that
// fail silently if guessed.
//
// Row LTX25-GUIDED-VIDEO, issue #1092.

#include "vllm/model_executor/models/ltx2_denoisers.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("ltx2 guided denoise: " + why);
}

// `_POSITIVE_ONLY_GUIDER` (denoisers.py:25-28) is
// `MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)`,
// which is `Ltx2MultiModalGuiderParams`'s own default construction. Stated as a
// function rather than inlined so the identity is checkable by eye against
// `_ensure_guider` (`:31-33`).
Ltx2MultiModalGuiderParams PositiveOnlyGuider() { return Ltx2MultiModalGuiderParams{}; }

// `perturbations.mask(type, block)` collapsed to this port's one sample. The
// KEEP polarity is upstream's — 1 keeps, 0 perturbs (perturbations.py:53-56) —
// so the DiT flag, which is `all_perturbed`, is the negation.
bool PerturbedAt(const Ltx2BatchedPerturbationConfig& config, Ltx2PerturbationType type,
                 int64_t block, int64_t sample) {
  const std::vector<int32_t> mask = config.Mask(type, block);
  return mask[static_cast<size_t>(sample)] == 0;
}

}  // namespace

Ltx2GuidedDenoiseResult Ltx2GuidedDenoise(const Ltx2X0Model& transformer,
                                          const Ltx2GuidedDenoiseInputs& in) {
  if (in.video == nullptr && in.audio == nullptr) {
    // `transformer.py:259-260` refuses it upstream, one level lower.
    Fail("both modalities are null; at least one of `video` or `audio` must be provided");
  }
  if (in.num_blocks < 1) {
    Fail("`num_blocks` is " + std::to_string(in.num_blocks) +
         "; the perturbation masks are sized by it (denoisers.py:180) and a wrong count "
         "perturbs a prefix of the blocks and renders");
  }

  // `_ensure_guider` (denoisers.py:31-33): an ABSENT modality takes the
  // positive-only guider, so its `calculate` returns `cond` unchanged and it asks
  // for no extra pass. A modality that is PRESENT keeps its caller's guider even
  // when every scale is at its no-op value.
  const Ltx2MultiModalGuiderParams video_guider =
      in.video != nullptr ? in.video_guider : PositiveOnlyGuider();
  const Ltx2MultiModalGuiderParams audio_guider =
      in.audio != nullptr ? in.audio_guider : PositiveOnlyGuider();

  Ltx2GuidedDenoiseResult result;

  // `should_skip_step` (denoisers.py:84-85).
  const bool v_skip = video_guider.ShouldSkipStep(in.step_index);
  const bool a_skip = audio_guider.ShouldSkipStep(in.step_index);
  result.video_skipped = v_skip;
  result.audio_skipped = a_skip;

  // `if v_skip and a_skip` (`:87-90`) — NO FORWARD AT ALL. Running the
  // conditional pass and using it is the plausible reading of "skip the
  // guidance" and is a whole DiT forward per skipped step and a different
  // trajectory, on a render that finishes either way.
  auto reuse = [&](const std::vector<float>* last, const char* which) {
    if (last == nullptr || last->empty()) {
      Fail(std::string("step ") + std::to_string(in.step_index) +
           " skips the " + which +
           " guider and no earlier step produced a denoised prediction to reuse. "
           "`should_skip_step` is `step % (skip_step + 1) != 0` (guiders.py:287-291), which is "
           "false at step 0, so this is unreachable through the request surface and is a defect "
           "rather than a bad request");
    }
    return *last;
  };
  if (v_skip && a_skip) {
    if (in.video != nullptr) result.video_denoised = reuse(in.last_denoised_video, "video");
    if (in.audio != nullptr) result.audio_denoised = reuse(in.last_denoised_audio, "audio");
    return result;
  }

  // ── the pass list (denoisers.py:97-137) ───────────────────────────────────
  //
  // ONE list for BOTH modalities, and the union of what the two guiders want.
  // See ltx2_denoisers.h item 2 for the render a per-modality list produces.
  struct Pass {
    Ltx2DenoisePass kind;
    const float* video_context;
    const float* audio_context;
    Ltx2PerturbationConfig perturbation;
  };
  std::vector<Pass> passes;

  const float* v_context = in.video != nullptr ? in.video->context : nullptr;
  const float* a_context = in.audio != nullptr ? in.audio->context : nullptr;
  if (in.video != nullptr && v_context == nullptr) {
    Fail("v_context is required when video_state is provided (denoisers.py:92-93)");
  }
  if (in.audio != nullptr && a_context == nullptr) {
    Fail("a_context is required when audio_state is provided (denoisers.py:94-95)");
  }
  passes.push_back({Ltx2DenoisePass::kCond, v_context, a_context, Ltx2PerturbationConfig{}});

  // `:102-109`. `force_uncond_pass` adds the pass for a modality that is PRESENT
  // even when its own guider does not ask (retake.py:305-311 is the one upstream
  // caller that sets it).
  const bool v_needs_neg = video_guider.DoUnconditionalGeneration() ||
                           (in.force_uncond_pass && in.video != nullptr);
  const bool a_needs_neg = audio_guider.DoUnconditionalGeneration() ||
                           (in.force_uncond_pass && in.audio != nullptr);
  if (v_needs_neg || a_needs_neg) {
    if (v_needs_neg && in.video_negative_context == nullptr) {
      Fail("negative context is required for unconditioned denoising on the VIDEO stream "
           "(denoisers.py:104-105). `do_unconditional_generation` is "
           "`not isclose(cfg_scale, 1.0)` (guiders.py:275-277), so either supply the negative "
           "conditioning or set the video cfg scale to 1.0");
    }
    if (a_needs_neg && in.audio_negative_context == nullptr) {
      Fail("negative context is required for unconditioned denoising on the AUDIO stream "
           "(denoisers.py:106-107). `do_unconditional_generation` is "
           "`not isclose(cfg_scale, 1.0)` (guiders.py:275-277), so either supply the negative "
           "conditioning or set the audio cfg scale to 1.0");
    }
    // `:108-109` — a modality with no negative context falls back to its POSITIVE
    // one rather than being dropped from the pass. That is not a defensive
    // default: it is how a pass forced for the OTHER modality still carries a
    // legal context for this one.
    passes.push_back({Ltx2DenoisePass::kUncond,
                      in.video_negative_context != nullptr ? in.video_negative_context : v_context,
                      in.audio_negative_context != nullptr ? in.audio_negative_context : a_context,
                      Ltx2PerturbationConfig{}});
  }

  // `:111-119`. ONE perturbed pass carrying BOTH modalities' blocks.
  {
    Ltx2PerturbationConfig stg;
    if (video_guider.DoPerturbedGeneration()) {
      Ltx2Perturbation p;
      p.type = Ltx2PerturbationType::kSkipVideoSelfAttn;
      p.blocks = video_guider.stg_blocks;
      stg.perturbations.push_back(std::move(p));
    }
    if (audio_guider.DoPerturbedGeneration()) {
      Ltx2Perturbation p;
      p.type = Ltx2PerturbationType::kSkipAudioSelfAttn;
      p.blocks = audio_guider.stg_blocks;
      stg.perturbations.push_back(std::move(p));
    }
    // A BLOCK LIST THAT MISSES EVERY BLOCK IS A WASTED FORWARD AND A ZERO TERM.
    // `Perturbation.is_perturbed` is `block in self.blocks`
    // (perturbations.py:26-33), so `stg_blocks = [28]` on a model with fewer
    // blocks perturbs nothing: the perturbed pass returns the conditional pass's
    // own tensor and `stg_scale * (cond - perturbed)` is exactly zero. The render
    // is finite, the right size, and carries no spatio-temporal guidance at all.
    // Upstream never meets this because it only ever runs 48-block checkpoints;
    // this port runs reduced ones, and a smaller checkpoint is a legal thing to
    // hand it.
    //
    // AN EMPTY LIST IS EXEMPT, and it was not until 2026-08-17. `blocks=[]` is
    // upstream's documented spelling for "perturb no block", distinct from
    // `blocks=None`'s "perturb every block" (perturbations.py:26-33), named as
    // the way to disable STG at `ltx-pipelines/docs/multimodal-guidance.md:13`,
    // shipped in `LTX_2_3_HQ_PARAMS` (constants.py:105, :113), and reachable
    // through `nargs="*"` (args.py:979-985). Upstream runs the pass and takes
    // the zero term; so does this. What is refused is a list that NAMES blocks
    // and reaches none of them, which is a request that disagrees with the
    // CHECKPOINT rather than a caller who asked for nothing.
    const auto check_reaches_a_block = [&](const Ltx2MultiModalGuiderParams& guider,
                                           const char* which) {
      if (!guider.DoPerturbedGeneration()) return;
      if (guider.stg_blocks.empty()) return;
      for (const int64_t block : guider.stg_blocks) {
        if (block >= 0 && block < in.num_blocks) return;
      }
      Fail(std::string("the ") + which + " STG scale is " + std::to_string(guider.stg_scale) +
           " and none of its " + std::to_string(guider.stg_blocks.size()) +
           " stg_blocks is in range for this DiT's " + std::to_string(in.num_blocks) +
           " blocks, so the perturbed forward would be identical to the conditional one and "
           "`stg_scale * (cond - perturbed)` would be exactly zero (guiders.py:264). Name blocks "
           "this checkpoint has, or set the STG scale to 0.0");
    };
    if (!stg.perturbations.empty()) {
      check_reaches_a_block(video_guider, "video");
      check_reaches_a_block(audio_guider, "audio");
      passes.push_back({Ltx2DenoisePass::kPerturbed, v_context, a_context, std::move(stg)});
    }
  }

  // `:121-137`. The isolated-modality pass: BOTH cross directions, ALL blocks
  // (`blocks=None`), when EITHER guider isolates. `modality_scale` is 3.0 on
  // every video row of the params table (utils/constants.py:54, :64), so this is
  // the default arm rather than a corner.
  if (video_guider.DoIsolatedModalityGeneration() ||
      audio_guider.DoIsolatedModalityGeneration()) {
    Ltx2PerturbationConfig mod;
    Ltx2Perturbation a2v;
    a2v.type = Ltx2PerturbationType::kSkipA2vCrossAttn;
    a2v.all_blocks = true;
    Ltx2Perturbation v2a;
    v2a.type = Ltx2PerturbationType::kSkipV2aCrossAttn;
    v2a.all_blocks = true;
    mod.perturbations.push_back(std::move(a2v));
    mod.perturbations.push_back(std::move(v2a));
    passes.push_back({Ltx2DenoisePass::kModality, v_context, a_context, std::move(mod)});
  }

  // ── the perturbation config (denoisers.py:182-187) ───────────────────────
  //
  // ONE batched config over the whole pass list, then one sample slice per pass,
  // which is upstream's `batched_ptb_configs` followed by the per-sample mask the
  // block reads. Building a fresh single-sample config per pass would be
  // arithmetically identical and would leave `Ltx2BatchedPerturbationConfig` —
  // the shared seam that mirrors `BatchedPerturbationConfig` — with no product
  // caller, which is the defect #1049 records.
  const int64_t pass_count = static_cast<int64_t>(passes.size());
  std::vector<Ltx2PerturbationConfig> configs;
  configs.reserve(passes.size());
  for (const Pass& p : passes) configs.push_back(p.perturbation);
  const Ltx2BatchedPerturbationConfig batched(configs, in.num_blocks);

  // ── run the passes (`:186`, one call there, `pass_count` calls here) ───────
  for (int64_t index = 0; index < pass_count; ++index) {
    const Pass& pass = passes[static_cast<size_t>(index)];

    const Ltx2BatchedPerturbationConfig slice = batched.BatchSlice(index, index + 1);
    Ltx2DitPerturbation perturbation;
    bool any = false;
    for (int64_t block = 0; block < in.num_blocks; ++block) {
      const bool v =
          PerturbedAt(slice, Ltx2PerturbationType::kSkipVideoSelfAttn, block, /*sample=*/0);
      const bool a =
          PerturbedAt(slice, Ltx2PerturbationType::kSkipAudioSelfAttn, block, /*sample=*/0);
      if (v || a) any = true;
      // Both vectors are sized whenever either is, because `Ltx2DitForward`
      // refuses a vector that is neither empty nor one entry per block and an
      // empty one means "nothing perturbed" for that stream.
      perturbation.video_self_attn.push_back(v ? 1 : 0);
      perturbation.audio_self_attn.push_back(a ? 1 : 0);
    }
    // The cross flags are not per block, because `Ltx2DitPerturbation` has no
    // per-block cross vector and upstream's reader is the per-block scalar
    // `cross_attn_skip_all` (transformer.py:335,367) rather than a mask
    // multiply. That flattening is only sound while the config says the same
    // thing on every block, which is what `blocks=None` produces
    // (denoisers.py:132-135) — so it is CHECKED here rather than assumed. A
    // block-list cross perturbation would otherwise be silently widened to all
    // blocks, which renders.
    const auto flatten_cross = [&](Ltx2PerturbationType type, const char* name) {
      const bool first = PerturbedAt(slice, type, /*block=*/0, /*sample=*/0);
      for (int64_t block = 1; block < in.num_blocks; ++block) {
        if (PerturbedAt(slice, type, block, /*sample=*/0) == first) continue;
        Fail(std::string("the ") + name +
             " cross-attention perturbation differs between block 0 and block " +
             std::to_string(block) + ". `Ltx2DitPerturbation` carries one boolean per direction "
             "because the only thing upstream builds these with is `blocks=None` "
             "(denoisers.py:132-135); a per-block cross perturbation cannot be represented and "
             "would be widened to every block rather than refused");
      }
      return first;
    };
    perturbation.video_cross_attn_skip_all =
        flatten_cross(Ltx2PerturbationType::kSkipA2vCrossAttn, "audio-to-video");
    perturbation.audio_cross_attn_skip_all =
        flatten_cross(Ltx2PerturbationType::kSkipV2aCrossAttn, "video-to-audio");
    if (perturbation.video_cross_attn_skip_all || perturbation.audio_cross_attn_skip_all) {
      any = true;
    }
    if (!any) {
      // `PerturbationConfig.empty()` reaches the forward as upstream's
      // `perturbations=None` (model.py:509-511), not as an all-ones mask, so the
      // conditional and unconditional passes take the same path an unguided
      // render takes.
      perturbation = Ltx2DitPerturbation{};
    }

    Ltx2ModalityInput video_in;
    Ltx2ModalityInput audio_in;
    if (in.video != nullptr) {
      video_in = *in.video;
      video_in.context = pass.video_context;
      // `enabled=not v_skip` (`:158`). A skipped modality stays PRESENT, so the
      // other stream's cross attention still reads its latent
      // (transformer.py:269 tests presence, not `enabled`).
      video_in.enabled = !v_skip;
    }
    if (in.audio != nullptr) {
      audio_in = *in.audio;
      audio_in.context = pass.audio_context;
      audio_in.enabled = !a_skip;
    }

    const bool perturbed = any;
    Ltx2X0Outputs out = transformer(in.video != nullptr ? &video_in : nullptr,
                                    in.audio != nullptr ? &audio_in : nullptr,
                                    perturbed ? &perturbation : nullptr);

    const size_t slot = static_cast<size_t>(pass.kind);
    result.pass_ran[slot] = true;
    result.video_pass[slot] = std::move(out.video);
    result.audio_pass[slot] = std::move(out.audio);
    result.video_pass_velocity[slot] = std::move(out.video_velocity);
    result.audio_pass_velocity[slot] = std::move(out.audio_velocity);

    // Observed at the call rather than restated from the guider params: a
    // perturbation that is BUILT and not HANDED OVER leaves the params untouched
    // and the render finite. Derived here, so a mutation that drops the argument
    // moves this record.
    if (pass.kind == Ltx2DenoisePass::kPerturbed && perturbed) {
      for (int64_t block = 0; block < in.num_blocks; ++block) {
        if (perturbation.video_self_attn[static_cast<size_t>(block)] != 0) {
          result.perturbed_video_blocks.push_back(block);
        }
        if (perturbation.audio_self_attn[static_cast<size_t>(block)] != 0) {
          result.perturbed_audio_blocks.push_back(block);
        }
      }
    }
    if (pass.kind == Ltx2DenoisePass::kModality && perturbed) {
      result.modality_pass_skipped_a2v = perturbation.video_cross_attn_skip_all;
      result.modality_pass_skipped_v2a = perturbation.audio_cross_attn_skip_all;
    }
  }

  // ── the combination (`:192-204`) ──────────────────────────────────────────
  //
  // EACH MODALITY WITH ITS OWN GUIDER, over the SAME splits. `r.get("uncond",
  // (0.0, 0.0))` is upstream's absent pass and is the float 0.0 its
  // `calculate` signature admits; a null here is the same thing, and
  // `Ltx2MultiModalGuidance` reads it as 0.0 (guiders.py:247-249).
  const auto at = [&result](const std::vector<float>* passes_array, Ltx2DenoisePass kind) {
    const size_t slot = static_cast<size_t>(kind);
    return result.pass_ran[slot] ? passes_array[slot].data() : nullptr;
  };
  const size_t cond_slot = static_cast<size_t>(Ltx2DenoisePass::kCond);

  if (in.video != nullptr) {
    if (v_skip) {
      result.video_denoised = reuse(in.last_denoised_video, "video");
    } else {
      const std::vector<float>& cond = result.video_pass[cond_slot];
      result.video_denoised = Ltx2MultiModalGuidance(
          video_guider, cond.data(), at(result.video_pass, Ltx2DenoisePass::kUncond),
          at(result.video_pass, Ltx2DenoisePass::kPerturbed),
          at(result.video_pass, Ltx2DenoisePass::kModality), static_cast<int64_t>(cond.size()));
    }
  }
  if (in.audio != nullptr) {
    if (a_skip) {
      result.audio_denoised = reuse(in.last_denoised_audio, "audio");
    } else {
      const std::vector<float>& cond = result.audio_pass[cond_slot];
      result.audio_denoised = Ltx2MultiModalGuidance(
          audio_guider, cond.data(), at(result.audio_pass, Ltx2DenoisePass::kUncond),
          at(result.audio_pass, Ltx2DenoisePass::kPerturbed),
          at(result.audio_pass, Ltx2DenoisePass::kModality), static_cast<int64_t>(cond.size()));
    }
  }
  return result;
}

}  // namespace vllm
