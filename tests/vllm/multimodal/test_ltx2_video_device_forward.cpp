// row LTX25-GUIDED-VIDEO (#1426) — LINK B: the x0 model reaches
// `Ltx2DitForwardDevice(..., p)`, and the perturbations it carries survive that
// hop.
//
// ─── THE DEFECT THIS FILE EXISTS FOR ─────────────────────────────────────────
//
// `src/vllm/multimodal/ltx2_video.cpp` picks the DiT forward with ONE ternary on
// `im.on_device`:
//
//     im.on_device ? Ltx2DitForwardDevice(*im.queue, ..., p)
//                  : Ltx2DitForward(im.device, ..., p);
//
// Every gate in this tree ran the SECOND branch, because `Ltx2VideoEngine::Load`
// refuses `device != 0` unless the platform seam resolves a REGISTERED
// accelerator backend (ltx2_video.cpp:829-838, :859-868), and no CI job in
// `.github/workflows/ci.yml` has a GPU runner. So the row measured two
// mutations and recorded the pair as a RESULT (spec §12.8):
//
//   D11  drop `p` from BOTH branches   -> RED  (test_ltx2_video, exit 1)
//   D10  drop `p` from the DEVICE arm  -> GREEN, for lack of a runner
//
// D10 being green is the whole finding. The device branch of that ternary was
// code nothing in this tree had ever executed, so its argument list was
// unguarded: the perturbations could be dropped from it and every gate stayed
// green. This file supplies the missing RUNNER and turns D10 RED.
//
// ─── WHAT IT PROVES, AND WHAT IT DOES NOT ────────────────────────────────────
//
// PROVES: on a build where an accelerator backend IS registered, the production
// entry point `vllm_video_engine_load` -> `LoadVideoEngine` ->
// `Ltx2VideoEngine::Load` -> `Generate` -> the phase loop -> `Ltx2GuidedDenoise`
// -> the x0 model lambda reaches the DEVICE arm of that ternary, on a queue whose
// device is NOT `kCPU`, and hands it the `Ltx2DitPerturbation` the guidance seam
// built. Nothing below constructs a DiT, a queue, a guider or a perturbation by
// hand; the only thing this file builds that a user would not is the accelerator.
//
// DOES NOT PROVE: that a GPU runs it. That is the HARDWARE half of #1426, it is
// one leased `--device cuda` run, and it is owed. `ltx2_video.cpp:829-838`
// refuses to substitute the one measurement for the other, and so does this
// comment: a synthetic accelerator whose memory is host memory answers "does the
// device arm execute and honour its arguments", and it answers nothing about
// residency, bandwidth or a real kernel.
//
// ─── WHY A SEPARATE EXECUTABLE ───────────────────────────────────────────────
//
// `vt::DeviceType` is a closed 7-value enum (include/vt/device.h:16-28), so a
// fake accelerator has to impersonate a real vendor, and registering a platform
// is PROCESS-GLOBAL: `CurrentPlatform()` would answer `xpu` for every case in
// the binary. `tests/vllm/multimodal/test_diffusion_device_seam.cpp` is a
// separate executable for exactly that reason and for this same family, and this
// file follows it. Keeping it out of `test_ltx2_video` also leaves that suite's
// own baseline where it is: 105 CASES, unmoved by this addition, which is what
// the spec's deferral was protecting. The case count is the comparable figure
// and the ASSERTION count is not — seven runs of one `test_ltx2_video` binary
// over an unchanged tree read 4721 once and 4719 six times while the case count
// stayed at 105 (#1885, and §12.8.1 of the row's spec records the readings).
//
// ─── HOW THE FAKE ACCELERATOR GETS KERNELS, AND WHY IT TAKES TWO CALLS ───────
//
// The two registrations are NOT interchangeable and neither alone is enough:
//
//   1. `vt::RegisterOp(OpId::kLtx2, kXPU, vt::GetOp(OpId::kLtx2, kCPU))` — the
//      LTX glue table, registered NATIVELY. It has to be this call and not the
//      reference tier, because `Ltx2DeviceKernelsAvailable`
//      (ltx2_device_resolve.cpp:14) asks `vt::OpRegistered`, which EXCLUDES the
//      reference tier deliberately and by name
//      (`src/vt/op_provider.cpp::OpRegistered`, whose contract comment says
//      "a FALLBACK, not a native kernel" at :756-757). `vt::RegisterOp` is the
//      public registration API (include/vt/op_provider.h:127) and is exactly
//      the call `src/vt/cpu/cpu_ltx2.cpp:203-204` makes to install the same
//      table for `kCPU`; reading the CPU pointer back and registering it for a
//      second device type is a marshalling instrument, not a re-implementation.
//   2. `vt::RegisterReferenceTier(kXPU)` — every OTHER op the staging and the
//      forward need (`MatmulBT`, `Add`, `RmsNorm`, `LayerNorm`, `GeluTanh`,
//      `AttentionDenseFa2`, `AttentionCross`, the VAE's `Conv3d`, ...). It loops
//      every OpId (`src/vt/op_provider.cpp::RegisterReferenceTier`, the loop at
//      :895-902) and copies the CPU `fn` verbatim
//      (`src/vt/op_provider.cpp::MaybeInstallReferenceTier`, :204-225). It runs
//      AFTER call 1 so that it SKIPS `kLtx2` rather than adding a redundant
//      fallback under the native registration.
//
// The tier's eligibility is `DeviceMemoryIsHostAddressable()`, which is STRICTLY
// NARROWER than `UnifiedMemory()` (backend.h:59) — GB10 reports the second and
// not the first. `FakeUnifiedBackend` below returns true for both and it is
// honest about it: its "device memory" is `std::malloc`, which a host kernel may
// dereference.
//
// ─── WHAT THIS FILE DOES NOT ASSERT, AND WHY ─────────────────────────────────
//
// Not numerics. `Ltx2VideoEngine::Load` resolves `compute_dtype = kBF16` on the
// device arm (ltx2_video.cpp:919) while the host arm runs f32, so the two arms
// are not comparable element by element and this file makes no claim that they
// are. `tests/vllm/models/test_ltx2_device.cpp` gates the device forward's
// arithmetic against the same upstream goldens as the CPU forward. What is
// asserted here is REACH and ARGUMENT DELIVERY, which is what link B is.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "ltx2_video_fixture.h"
#include "vllm/model_executor/models/ltx2_kernels.h"
#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/multimodal/ltx2_video.h"
#include "vllm/multimodal/video_engine.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

// A UNIFIED-MEMORY accelerator whose memory is host memory. Shaped like
// `FakeXpuBackend` in test_diffusion_device_seam.cpp, plus the ONE property that
// file does not need and this one cannot work without:
// `DeviceMemoryIsHostAddressable`, which is what `ReferenceTierEligible`
// actually tests (`src/vt/op_provider.cpp::ReferenceTierEligible`, whose return
// is :888-889).
class FakeUnifiedBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override { return std::malloc(bytes == 0 ? 1 : bytes); }
  void Free(void* p) override { std::free(p); }
  void Memset(vt::Queue&, void* p, int value, size_t bytes) override {
    std::memset(p, value, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
  // TRUE, and honestly so: `Alloc` is `std::malloc`. This is the property the
  // portable reference tier gates on, and it is NOT the same claim as
  // `UnifiedMemory()` — backend.h:59 records the difference, and GB10 is the box
  // where the two answers differ.
  bool DeviceMemoryIsHostAddressable() const override { return true; }
};

// A COMPLETE platform: it accepts every architecture, because the thing under
// test here is the forward and not the capability guard.
// test_diffusion_device_seam.cpp already gates the DECLINING arm of that guard
// for this family, and duplicating it here would gate it twice and this once.
class FakeAcceleratorPlatform final : public vllm::platforms::Platform {
 public:
  explicit FakeAcceleratorPlatform(FakeUnifiedBackend& backend) : backend_(backend) {}

  vt::DeviceType device_type() const override { return vt::DeviceType::kXPU; }
  vt::Backend& backend() const override { return backend_; }
  vllm::platforms::DeviceCapability get_device_capability() const override { return {}; }
  std::vector<vt::DType> supported_dtypes() const override { return {vt::DType::kBF16}; }
  vllm::platforms::ResidencyPolicy residency_policy() const override { return {}; }
  bool supports_model_architecture(std::string_view) const override { return true; }

 private:
  FakeUnifiedBackend& backend_;
};

FakeUnifiedBackend& Backend() {
  static FakeUnifiedBackend backend;
  return backend;
}

FakeAcceleratorPlatform& Platform() {
  static FakeAcceleratorPlatform platform(Backend());
  return platform;
}

// Registered into the CUDA slot as well as its own, for the reason
// test_diffusion_device_seam.cpp gives: `CurrentPlatform()` walks
// {kCUDA, kROCM, kXPU, ...} and returns the first REGISTERED entry
// (src/vllm/platforms/platform.cpp), so on a CUDA build the engine would resolve
// the real platform and this fake would never be asked.
//
// Idempotent, so every case can call it without ordering itself against the
// others.
void RegisterFakeAccelerator() {
  vt::RegisterBackend(vt::DeviceType::kXPU, &Backend());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kXPU, &Platform());
  vllm::platforms::RegisterPlatform(vt::DeviceType::kCUDA, &Platform());

  // ORDER IS A REDUNDANCY QUESTION, NOT A CORRECTNESS ONE — and that is
  // MEASURED, because the reverse was tried. The native glue table first, then
  // the tier: the tier SKIPS an op the target already serves natively
  // (`src/vt/op_provider.cpp::RegisterReferenceTier`, the `OpRegistered`
  // `continue` at :900), so this order leaves exactly ONE provider under
  // `kLtx2` — the native entry `Ltx2DeviceKernelsAvailable` can see.
  //
  // SWAPPING THE TWO CALLS REFUSES NOTHING. The tier-first build compiles
  // clean, relinks the executable, and exits 0 at 1 case / 30 assertions. It is
  // not a no-op mutation either: `RegisterReferenceTier` returns 112 tier-first
  // against 111 native-first, and `vt::OpRegistered(kLtx2, kXPU)` reads 0
  // immediately after the tier-first call, so `kLtx2` genuinely IS covered by
  // the tier at that point. That reading is also the measurement behind
  // "neither alone is enough" above: with the tier only, `OpRegistered` answers
  // 0 and `Ltx2DeviceKernelsAvailable` refuses. The `vt::RegisterOp` that
  // follows then installs the priority-0 native provider, `OpRegistered` reads
  // 1, and `GetOp(kLtx2, kXPU)` is the CPU pointer again — so
  // `Ltx2DeviceKernelsAvailable` answers true and the forward runs either way.
  //
  // What the reverse order actually costs is ONE redundant negative-priority
  // fallback left registered under `kLtx2`, outranked by the native entry. That
  // is precisely the waste the tier's own skip comment says it exists to avoid
  // (:897-899). This order avoids it. It does not rescue the forward, and an
  // earlier draft of this comment claimed that it did.
  void* cpu_glue = vt::GetOp(vt::OpId::kLtx2, vt::DeviceType::kCPU);
  REQUIRE(cpu_glue != nullptr);
  vt::RegisterOp(vt::OpId::kLtx2, vt::DeviceType::kXPU, cpu_glue);
  (void)vt::RegisterReferenceTier(vt::DeviceType::kXPU);
}

struct Workspace {
  std::string root, fixture;
  ltx2_fixture::Paths paths;
  Workspace() {
    static int counter = 0;
    root = "/tmp/vllm_ltx2_dev_fwd_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++);
    ::mkdir(root.c_str(), 0755);
    fixture = root + "/fixture";
    paths = ltx2_fixture::WriteFixture(fixture);
  }
  ~Workspace() {
    const char* keep = std::getenv("VLLM_KEEP_TEST_ARTIFACTS");
    if (keep != nullptr && keep[0] == '1') {
      std::printf("[ltx2] kept workspace: %s\n", root.c_str());
      return;
    }
    const int rc = std::system(("rm -rf '" + root + "'").c_str());
    (void)rc;
  }
};

// Resolved from the recipe table rather than listed, the way
// test_ltx2_video.cpp's own helper is and for the same reason: a hard-coded
// class here would be a second copy of `ResolveLtx2PipelineRecipe`'s table that
// could drift from it without anything saying so.
std::string FixtureCheckpointClass(const std::string& kind) {
  const vllm::Ltx2PipelineRecipe recipe = vllm::ResolveLtx2PipelineRecipe(kind, "2.5");
  switch (recipe.checkpoint_class) {
    case vllm::Ltx2RequiredCheckpointClass::kFull:
      return "full";
    case vllm::Ltx2RequiredCheckpointClass::kDistilled:
      return "distilled";
    case vllm::Ltx2RequiredCheckpointClass::kKeyframeSlotSft:
      return "keyframe_slot_sft";
    case vllm::Ltx2RequiredCheckpointClass::kFullOrDistilled:
      return "distilled";
    case vllm::Ltx2RequiredCheckpointClass::kUnstated:
      return "";
  }
  return "";
}

// `one_stage` on the shipped fixture, on the DEVICE. Everything except `device`
// mirrors test_ltx2_video.cpp's `OneStageParams`, because the arm this case sits
// on has to be the one the CPU gate sits on: `one_stage`'s own recipe resolves
// `cfg_scale = 3.0`, `stg_scale = 1.0` and `modality_scale = 3.0`, so `p` is
// non-null on the perturbed and isolated-modality passes with no request extra
// at all. A configuration invented here could have made `p` null and the case
// would have passed by measuring nothing.
vllm::multimodal::VideoModelParams DeviceOneStageParams(const ltx2_fixture::Paths& paths) {
  vllm::multimodal::VideoModelParams mp;
  mp.dit_path = paths.dit;
  mp.video_vae_path = paths.video_vae;
  mp.audio_vae_path = paths.audio_vae;
  mp.prompt_embeds_path = paths.video_embeds;
  mp.extras[vllm::multimodal::kLtx2AudioPromptEmbedsExtra] = paths.audio_embeds;
  mp.extras[vllm::multimodal::kLtx2NegativePromptEmbedsExtra] = paths.negative_video_embeds;
  mp.extras[vllm::multimodal::kLtx2NegativeAudioPromptEmbedsExtra] =
      paths.negative_audio_embeds;
  mp.extras[vllm::multimodal::kLtx2PipelineKindExtra] = "one_stage";
  mp.extras[vllm::multimodal::kLtx2CheckpointClassExtra] = FixtureCheckpointClass("one_stage");
  // THE DEVICE. 0 is the CPU and 1 is accelerator index 0, which is the mapping
  // `Ltx2VideoEngine::Load` documents. This one field is the difference between
  // the branch every existing gate runs and the branch this file exists for.
  mp.device = 1;
  return mp;
}

vllm::multimodal::VideoGenParams DeviceOneStageGen(const std::string& out_dir) {
  vllm::multimodal::VideoGenParams gen;
  gen.num_frames = 9;
  gen.height = 64;
  gen.width = 64;
  gen.has_seed = true;
  gen.seed = 7;
  gen.output_dir = out_dir;
  gen.steps = 2;
  // The ONE guider field the fixture forces, for the reason test_ltx2_video.cpp
  // records: the reduced DiT has TWO blocks, so the params table's own
  // `stg_blocks = [28]` names a block this checkpoint does not have, and the
  // engine refuses that by name rather than perturbing nothing. Setting the STG
  // scale to zero instead would turn the perturbed pass OFF, which vacates every
  // assertion below.
  gen.extras[vllm::multimodal::kLtx2VideoStgBlocksExtra] = "1";
  gen.extras[vllm::multimodal::kLtx2AudioStgBlocksExtra] = "1";
  return gen;
}

double MaxAbsOf(const std::vector<float>& v) {
  double m = 0.0;
  for (const float x : v) m = std::max(m, std::abs(static_cast<double>(x)));
  return m;
}

double MaxAbsDiffOf(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

}  // namespace

TEST_CASE("ltx2 one_stage on an accelerator: the x0 model reaches the DEVICE forward (#1426)") {
  RegisterFakeAccelerator();

  // The three questions `Ltx2VideoEngine::Load` asks before it will bind a
  // queue, asserted here as PRECONDITIONS of the case rather than left to be
  // inferred from a green render. A build where one of these answered
  // differently would refuse the load, and the refusal message — not this case's
  // own assertions — would be what a reader had to decode.
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kXPU);
  REQUIRE(vt::TryGetBackend(vt::Device{vt::DeviceType::kXPU, 0}) != nullptr);
  REQUIRE(vllm::platforms::CurrentPlatform().supports_model_architecture(
      vllm::multimodal::kLtx2VideoFamily));
  // And the one `Glue()` asks, which is the one the reference tier CANNOT
  // answer: a NATIVE `kLtx2` registration for this device.
  REQUIRE(vllm::ltx2::Ltx2DeviceKernelsAvailable(vt::DeviceType::kXPU));
  REQUIRE(vllm::ltx2::Ltx2Device(vt::DeviceType::kXPU) ==
          vllm::ltx2::Ltx2Device(vt::DeviceType::kCPU));

  Workspace ws;

  // The guidance the RECIPE resolves, read before the render so that the
  // assertions below rest on a configuration this case did not choose. A
  // `stg_scale` or `modality_scale` of zero would make the perturbed and
  // isolated-modality passes vacuous and every difference below would be zero
  // for a reason that has nothing to do with the ternary.
  const vllm::Ltx2PipelineRecipe recipe = vllm::ResolveLtx2PipelineRecipe("one_stage", "2.5");
  REQUIRE(recipe.phases.size() == 1);
  CHECK(recipe.phases[0].video_guidance.stg_scale == 1.0);
  CHECK(recipe.phases[0].video_guidance.modality_scale == 3.0);

  // Counted from here, so the numbers below are this render's. Stats are a
  // process-global latch, which is why they are reset rather than assumed zero.
  vt::EnableOpProviderCallStats(true);
  vt::ResetOpProviderStats(vt::OpId::kLtx2, vt::DeviceType::kXPU);
  vt::ResetOpProviderStats(vt::OpId::kLtx2, vt::DeviceType::kCPU);

  // THE PRODUCTION ENTRY POINT. `vllm_video_engine_load` (include/vllm.h) calls
  // `LoadVideoEngine` through src/capi/vllm_c.cpp, which dispatches on the
  // declared family to this engine's own `Load`. Entering at
  // `Ltx2VideoEngine::Load` would skip the registry hop, and entering at
  // `Ltx2DitForwardDevice` would prove that a function works rather than that
  // anything reaches it.
  const std::unique_ptr<vllm::multimodal::VideoEngine> engine =
      vllm::multimodal::LoadVideoEngine(DeviceOneStageParams(ws.paths));
  REQUIRE(engine != nullptr);
  (void)engine->Generate(DeviceOneStageGen(ws.root + "/device_guided"));

  const auto* ltx = dynamic_cast<const vllm::multimodal::Ltx2VideoEngine*>(engine.get());
  REQUIRE(ltx != nullptr);
  const vllm::multimodal::Ltx2ConditioningTrace t = ltx->last_conditioning();
  REQUIRE(t.completed);

  vt::EnableOpProviderCallStats(false);

  // ── the DEVICE branch of the ternary is the one that ran ──────────────────
  //
  // `Ltx2DitForwardDevice` reaches its glue table through `Glue(d)` ->
  // `Ltx2Device(d.q.device.type)` -> `vt::GetOp(kLtx2, <queue device>)`, and
  // `Ltx2DitForward` never touches that table at all. So a non-zero XPU
  // selection count is the DEVICE arm executing on a queue whose device is not
  // `kCPU`, and it is read off the op seam's own counter rather than inferred
  // from an output that both arms can produce.
  //
  // The CPU counter staying at zero is what makes it EXCLUSIVE: asserting only
  // that the XPU counter moved would pass on a build that ran both arms, or that
  // ran the device forward on a CPU queue relabelled as an accelerator.
  const vt::OpProviderStats xpu = vt::GetOpProviderStats(vt::OpId::kLtx2, vt::DeviceType::kXPU);
  const vt::OpProviderStats cpu = vt::GetOpProviderStats(vt::OpId::kLtx2, vt::DeviceType::kCPU);
  INFO("kLtx2 glue dispatches: xpu=" << xpu.selections << " cpu=" << cpu.selections);
  CHECK_MESSAGE(xpu.selections > 0u,
                "no LTX glue table was resolved for the accelerator, so the x0 model took the "
                "HOST branch of the ternary and this case is measuring the arm every other "
                "gate already runs");
  CHECK_MESSAGE(cpu.selections == 0u,
                "the LTX glue table was resolved for the CPU device during a device render, so "
                "some part of this forward ran on a CPU queue");

  // ── the render went through the GUIDED seam, with all four passes ─────────
  //
  // Counts, not tensors: an arm silently skipped changes a counter and changes
  // no shape, no frame count and no sample rate. Without these, the three
  // difference assertions below could be satisfied by a render that never issued
  // a perturbed pass at all.
  REQUIRE_MESSAGE(t.video_guided, "the video denoise did not go through the guided seam at all");
  CHECK(t.video_cond_forwards == 1);
  CHECK(t.video_uncond_forwards == 1);
  CHECK(t.video_perturbed_forwards == 1);
  CHECK(t.video_modality_forwards == 1);
  // The perturbation REACHED the DiT, read off the mask that was handed over
  // rather than off the guider params.
  CHECK(t.video_perturbed_blocks == std::vector<int64_t>{1});
  CHECK(t.video_modality_skipped_a2v);
  CHECK(t.video_modality_skipped_v2a);

  // ── LINK B: `p` survived the hop into the device forward ──────────────────
  //
  // THE MUTATION THE PERTURBED AND MODALITY LINES EXIST FOR is D10: drop `p`
  // from the DEVICE arm of the ternary alone. Under it, all four passes still run and still
  // count, the render still completes, every artifact is still written — and the
  // perturbed and isolated-modality passes return the CONDITIONAL pass's own
  // tensor, because the only thing that distinguished them was the argument that
  // was dropped. The guidance terms `stg_scale * (cond - perturbed)` and
  // `(modality_scale - 1) * (cond - mod)` then become identically zero and the
  // render silently loses its spatiotemporal guidance.
  //
  // The uncond arm is included because it is the CONTROL: its difference comes
  // from the negative context and not from `p`, so it stays non-zero under D10.
  // A case where all three went to zero together would be measuring something
  // else — a dead denoise loop, say — and the pair says which happened.
  const double latent_span = MaxAbsOf(t.video_first_latent);
  REQUIRE_MESSAGE(latent_span > 1e-3,
                  "the step-0 sample is zero, so nothing below discriminates");

  const double d_uncond = MaxAbsDiffOf(t.video_first_uncond, t.video_first_cond);
  const double d_perturbed = MaxAbsDiffOf(t.video_first_perturbed, t.video_first_cond);
  const double d_modality = MaxAbsDiffOf(t.video_first_modality, t.video_first_cond);
  INFO("max|latent| = " << latent_span << "  |uncond-cond| = " << d_uncond
                        << "  |perturbed-cond| = " << d_perturbed << "  |modality-cond| = "
                        << d_modality);

  CHECK_MESSAGE(d_uncond > 1e-6 * latent_span,
                "the unconditional pass returned the conditional pass's own tensor on the "
                "DEVICE arm, so the negative context did not reach the device forward");
  CHECK_MESSAGE(d_perturbed > 1e-6 * latent_span,
                "the perturbed pass returned the conditional pass's own tensor on the DEVICE "
                "arm, so `Ltx2DitForwardDevice` was called WITHOUT the perturbations the x0 "
                "model was handed (#1426, mutation D10)");
  CHECK_MESSAGE(d_modality > 1e-6 * latent_span,
                "the isolated-modality pass returned the conditional pass's own tensor on the "
                "DEVICE arm, so the cross-attention perturbation did not reach "
                "`Ltx2DitForwardDevice` (#1426, mutation D10)");

  // And the guider MOVED what it was handed, so the arms above are inputs to a
  // combination rather than four values recorded beside one.
  CHECK(t.video_first_denoised != t.video_first_cond);
}
