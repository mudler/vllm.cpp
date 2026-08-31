// MODEL-MM-QWEN4-EXP W5b — the assembled `Qwen4ExpTextModel` layer loop, and
// the loader/op compositions nothing in this tree had ever run together.
//
// Issue #2031, campaign issue #1978, spec `.agents/specs/qwen4-exp-flash-next.md`.
//
// WHY THIS SUITE EXISTS SEPARATELY FROM THE ELEVEN THAT PRECEDE IT. Every
// qwen4_exp suite before this one gates ONE side of a seam: the loader suite
// asserts what `LoadQwen4ExpFromGguf` produced, and the four device-op suites
// assert what `vt::Qwen4Exp*` computes from operands the TEST built. Neither can
// see a disagreement about what those operands MEAN, and this row has carried
// exactly one such disagreement since W5b-2 (#2218): the loader stores every
// gamma in the RAW HuggingFace parameterization and `vt::Qwen4ExpGatedResidual`
// documented the folded one. A gate that composes the two is the only
// instrument that can see it, because both halves are individually correct.
//
// ORACLE. vLLM registers `qwen4_exp` at no revision, so the algorithm oracle is
// transformers **5.16.0**, this row's accepted lane pin, reached here through
// the W2/W3 HOST references (`qwen4_exp_hc.cpp`, `qwen4_exp_ple.cpp`) that were
// themselves gated against it golden-for-golden.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "support/qwen4_exp_gguf_fixture.h"

#include "vllm/model_executor/models/dense_attn_block.h"  // dense_attn::ResidentWeight
#include "vllm/model_executor/models/qwen4_exp_hc.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

namespace {

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

vt::Tensor MakeT(void* p, vt::DType dt, const std::vector<int64_t>& shape) {
  vt::Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = vt::Device{vt::DeviceType::kCPU, 0};
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= t.shape[i];
  }
  return t;
}

// A loaded weight's VALUES, widened to f32 without rounding. bf16 -> f32 is
// exact, so the host oracle below is fed the same numbers the device op reads
// and any difference between them is the op's arithmetic and never the operand.
std::vector<float> AsF32(const vllm::OwnedTensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == vt::DType::kBF16) {
    const auto* p = reinterpret_cast<const uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  } else if (t.dtype == vt::DType::kF32) {
    const auto* p = reinterpret_cast<const float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = p[i];
  } else {
    FAIL("unexpected weight dtype");
  }
  return out;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

// --- the loader/op polarity composition (#2218) ------------------------------

TEST_CASE(
    "qwen4_exp forward: a LOADED hc_norm gamma drives the gated-residual op to "
    "the value the FILE asked for") {
  // WHY THIS CANNOT BE DONE WITH A HAND-BUILT GAMMA, which is what
  // `test_qwen4_exp_hc_device.cpp` does and why the defect survived eleven
  // waves. The polarity question is "what does the number in
  // `Qwen4ExpGatedResidualWeights::hc_norm` MEAN", and only the loader can
  // answer it. A test that builds the operand itself has already chosen the
  // answer and is then asserting its own choice — the tautological-fixture
  // shape this campaign produced five of.
  //
  // The chain under test, end to end:
  //   ggml-org/llama.cpp#27742 writes `hc_norm.weight` WITH the `+1` fold, so
  //   the file carries `1 + w_hf`   ->   `LoadNormBf16(..., unshift=true)`
  //   inverts it, so the model holds the RAW `w_hf`   ->   the op must put the
  //   1 back, because upstream's `Qwen4ExpTextRMSNorm.forward` is
  //   `output * (1.0 + self.weight.float())` (modeling_qwen4_exp.py:173-178).
  //
  // The oracle is therefore driven with the FILE's gamma and the op with the
  // MODEL's, and the two must agree. Hand the op a raw gamma under the old
  // "this op never adds 1" contract and every hyper-connection norm scales by
  // `w_hf`, which the fixture puts in [0, 1) and the released checkpoint puts
  // within an ulp or two of ZERO — a plausible tensor, never a crash, and no
  // token gate on any hardware this project owns could see it.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  const vllm::Qwen4ExpWeights& w =
      vllm::ModelAs<vllm::Qwen4ExpLoadedModel>(
          *model, "Qwen4ExpForConditionalGeneration")
          .weights();
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));

  const int64_t T = 3;
  const int64_t flat = kStream;
  const float eps = 1e-6f;

  // Both hyper-connection sites of layer 0 AND the model-level `use_combine`
  // mixer, because the mixer is the one call with a null `block_inject` and a
  // polarity defect there lands directly on `lm_head`.
  struct Site {
    std::string name;
    const vllm::Qwen4ExpGatedResidualWeights* w;
    int64_t tag;
  };
  const std::vector<Site> sites{
      {"layer0.attn_hc", &w.layers[0].attn_hc, HcNormTag(0, "attn")},
      {"layer0.mlp_hc", &w.layers[0].mlp_hc, HcNormTag(0, "ffn")},
      {"model.mixer", &w.mixer, kMixerNormTag},
  };

  for (const Site& s : sites) {
    INFO("site ", s.name);
    const vllm::Qwen4ExpGatedResidualWeights& gw = *s.w;
    REQUIRE(gw.hc_norm.rank == 1);
    REQUIRE(gw.hc_norm.shape[0] == flat);

    // THE MODEL'S gamma: what the loader left in the weight, raw.
    const std::vector<float> model_gamma = AsF32(gw.hc_norm);
    // THE FILE'S gamma: `1 + w_hf`, the value ggml-org/llama.cpp#27742 wrote.
    // Reconstructed from the fixture's own generator rather than re-read from
    // the file, so this expectation is independent of the loader under test.
    std::vector<float> file_gamma(static_cast<size_t>(flat));
    for (int64_t i = 0; i < flat; ++i)
      file_gamma[static_cast<size_t>(i)] = NormValue(i, s.tag);

    // The precondition this whole case rests on, asserted rather than assumed:
    // the loader really did remove the fold, so the two gammas differ by
    // exactly one everywhere. If this ever fails the loader changed polarity
    // and the rest of the case is measuring something else.
    for (int64_t i = 0; i < flat; ++i) {
      REQUIRE(model_gamma[static_cast<size_t>(i)] + 1.0f ==
              doctest::Approx(file_gamma[static_cast<size_t>(i)]));
    }
    // ... and they are far enough apart that no tolerance can absorb the
    // difference. `w_hf` is in [0, 1) and `1 + w_hf` in [1, 2), so the smallest
    // ratio between them is 1.0 / 1.996 and the largest is unbounded.
    REQUIRE(MaxAbsDiff(model_gamma, file_gamma) == doctest::Approx(1.0f));

    const std::vector<float> down = AsF32(gw.down);
    const std::vector<float> up = AsF32(gw.up);
    const std::vector<float> inject = gw.has_inject ? AsF32(gw.inject) : std::vector<float>{};

    // A hyper stream with per-branch scale separation, so a norm that collapses
    // toward zero cannot be mistaken for a small numerical difference.
    std::vector<float> hyper(static_cast<size_t>(T * flat));
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t p = 0; p < flat; ++p) {
        hyper[static_cast<size_t>(t * flat + p)] =
            0.37f * std::sin(0.11f * static_cast<float>(p + 7 * t)) +
            0.05f * static_cast<float>((p / kH) + 1);
      }
    }

    // ── the op ──────────────────────────────────────────────────────────────
    std::vector<float> mixed(static_cast<size_t>(T * kH), 0.0f);
    std::vector<float> injection(static_cast<size_t>(T * kHcCount), 0.0f);
    vt::Tensor t_hyper = MakeT(hyper.data(), vt::DType::kF32, {T, flat});
    vt::Tensor t_mixed = MakeT(mixed.data(), vt::DType::kF32, {T, kH});
    vt::Tensor t_inj = MakeT(injection.data(), vt::DType::kF32, {T, kHcCount});
    vt::Tensor t_w = gw.hc_norm.View();
    vt::Tensor t_down = gw.down.View();
    vt::Tensor t_up = gw.up.View();
    vt::Tensor t_inject = gw.has_inject ? gw.inject.View() : vt::Tensor{};

    vt::Qwen4ExpGatedResidualArgs args;
    args.hc_count = kHcCount;
    args.hidden_size = kH;
    args.lowrank = kHcLowrank;
    args.eps = eps;
    vt::Queue q = CpuQ();
    vt::Qwen4ExpGatedResidual(q, t_mixed, gw.has_inject ? &t_inj : nullptr, t_hyper, t_w,
                              t_down, t_up, gw.has_inject ? &t_inject : nullptr, args);

    // ── the oracle, driven with the FILE's gamma ────────────────────────────
    vllm::qwen4_exp::GatedResidualWeights ow;
    ow.hc_norm_weight = file_gamma;  // vLLM form == what the file carried
    ow.mix_down = down;
    ow.mix_up = up;
    ow.block_inject = inject;
    std::vector<float> want_mixed(static_cast<size_t>(T * kH));
    std::vector<float> want_inj(static_cast<size_t>(T * kHcCount));
    for (int64_t t = 0; t < T; ++t) {
      const std::vector<float> row(hyper.begin() + static_cast<size_t>(t * flat),
                                   hyper.begin() + static_cast<size_t>((t + 1) * flat));
      const vllm::qwen4_exp::GatedResidualResult r =
          vllm::qwen4_exp::GatedResidualForward(row, ow, kHcCount, kH, eps);
      for (int64_t h = 0; h < kH; ++h)
        want_mixed[static_cast<size_t>(t * kH + h)] = r.mixed_input[static_cast<size_t>(h)];
      if (gw.has_inject) {
        for (int64_t j = 0; j < kHcCount; ++j)
          want_inj[static_cast<size_t>(t * kHcCount + j)] =
              r.injection_weights[static_cast<size_t>(j)];
      }
    }

    // `mixed` IS THE DISCRIMINATOR AND `injection` IS NOT — said out loud,
    // because a reader counting green assertions would count both. The
    // injection logit is `inject . normed / hc` over a 128-wide row of the
    // fixture's `inject` ramp, which reaches ~10^4 whichever gamma is used, so
    // `2 * sigmoid(.)` saturates at 2.0 under BOTH polarities and the check
    // below passes either way. It is kept as an equality check on the arm's
    // OTHER properties (shape, per-token layout, the `has_inject` split), and
    // it is asserted saturated so the day it stops being saturated is loud.
    CHECK(MaxAbsDiff(mixed, want_mixed) < 1e-5f);
    if (gw.has_inject) {
      CHECK(MaxAbsDiff(injection, want_inj) < 1e-5f);
      for (float v : injection) {
        REQUIRE_MESSAGE(v == doctest::Approx(2.0f),
                        "the injection sigmoid is expected SATURATED at this "
                        "fixture; if it is not, this arm now discriminates "
                        "polarity and the comment above is stale");
      }
    }
  }
}

// --- W5r (#2031): the shared ResidentWeight carries the load-time layout -----

TEST_CASE(
    "dense_attn::ResidentWeight carries the repack markers the qwen4_exp "
    "forward's mix weights depend on") {
  // THE DEFECT, AND WHY IT IS THIS ROW'S. `vt::Tensor::repacked` says the
  // block-quant bytes were rewritten at load into the CPU i8mm interleave
  // (`q8_0 -> block_q8_0x4`), and `kMatmulBTQuant` keys on it to pick the
  // repacked GEMM. `MakeTensor` drops it, and the SHARED
  // `dense_attn::ResidentWeight` — the helper 25 models use — did not put it
  // back, while `qwen3_5.cpp`'s private copy of the same helper always has
  // (:1055, :1060). So a repacked weight taken through the shared helper
  // reached the kernel flagged as plain q8_0 and was decoded as garbage, with
  // no crash and no refusal anywhere.
  //
  // It is this row's because the qwen4_exp forward is the caller that makes it
  // reachable AND visibly inconsistent: it hands `hc_*_down`/`hc_*_up` to
  // `vt::Qwen4ExpGatedResidual` through `ResidentWeight`
  // (`qwen4_exp_forward.cpp:421-422, :479-480, :538-539`) and `hc_*_inject`
  // through `OwnedTensor::View()` (:417, :475), which has carried the marker
  // all along — two operands of one op disagreeing about the same flag. W5p
  // (#2031) is what made that matter: before it those two tensors were float
  // weights on `LinearNoBias`, and now they are block-quant operands routed to
  // `vt::MatmulBTQuant`, which is the consumer that reads `repacked`.
  //
  // THIS CASE IS THE ONLY GATE THAT CAN RUN ON THIS HOST, and that is stated
  // rather than glossed. `vt::cpu::QuantRepackActive()` is true only on aarch64
  // with i8mm, so on x86 the loader never sets the marker and no end-to-end
  // path can exercise it. The flag is therefore set by hand here, on the same
  // `OwnedTensor` type the loader produces, and the assertion is that the
  // helper propagates what it is given. `thor` — the box the released
  // Qwen3.8-Flash-Next checkpoint loads on — is an aarch64 i8mm host.
  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  vllm::OwnedTensor w;
  w.dtype = vt::DType::kQ8_0;
  w.rank = 2;
  w.shape[0] = 4;
  w.shape[1] = 32;
  w.nk = true;
  // One Q8_0 block per row: 4 rows x 34 bytes. The bytes are never read; only
  // the metadata the helper copies is under test, and `ResidentWeight` refuses
  // an empty buffer by name (#1953), so the buffer has to be real.
  w.bytes.assign(4 * 34, 0);

  SUBCASE("a repacked weight stays repacked") {
    w.repacked = true;
    const vt::Tensor t = vllm::dense_attn::ResidentWeight(d, w);
    CHECK(t.repacked);
    CHECK(t.dtype == vt::DType::kQ8_0);
    CHECK(t.data == static_cast<const void*>(w.bytes.data()));
  }
  SUBCASE("and an unrepacked one stays unrepacked") {
    // The other polarity, so the fix cannot be "always true", which would send
    // a plain q8_0 weight to the repacked GEMM and break the far larger set of
    // callers that never repack.
    w.repacked = false;
    const vt::Tensor t = vllm::dense_attn::ResidentWeight(d, w);
    CHECK_FALSE(t.repacked);
  }
  SUBCASE("the elementwise [K,N] transpose marker rides across too") {
    vllm::OwnedTensor e;
    e.dtype = vt::DType::kBF16;
    e.rank = 2;
    e.shape[0] = 4;
    e.shape[1] = 8;
    e.nk = true;
    e.bytes.assign(4 * 8 * 2, 0);
    e.elem_kn_repacked = true;
    const vt::Tensor t = vllm::dense_attn::ResidentWeight(d, e);
    CHECK(t.elem_kn_repacked);
  }
  SUBCASE("and the mix weights the forward routes keep it through the real shape") {
    // THE ROW'S OWN OPERAND SHAPE, not a generic one: `hc_*_down` is
    // `[hc_lowrank, hc_count * hidden_size]` and the forward passes that shape
    // explicitly at `qwen4_exp_forward.cpp:421`. A helper that propagated the
    // marker only on the default-shape arm would pass the subcases above and
    // still drop it on every call this row makes.
    w.repacked = true;
    w.shape[0] = 2;
    w.shape[1] = 64;
    w.bytes.assign(2 * 2 * 34, 0);
    const vt::Tensor t = vllm::dense_attn::ResidentWeight(d, w, {2, 64});
    CHECK(t.repacked);
    CHECK(t.shape[0] == 2);
    CHECK(t.shape[1] == 64);
  }
}
