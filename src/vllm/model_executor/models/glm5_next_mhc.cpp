// GLM-5.3-Flash W4 — the mHC wiring and the unweighted head collapse.
// See glm5_next_mhc.h for the port anchors on both sides, the oracle, and why
// the head is NOT `deepseek_v4::HcHeadCollapse`.
#include "vllm/model_executor/models/glm5_next_mhc.h"

#include <stdexcept>
#include <string>

namespace vllm::glm5_next {

namespace {

void RequireShape(bool ok, const char* what) {
  if (!ok) {
    throw std::invalid_argument(std::string("glm5_next mHC: ") + what);
  }
}

}  // namespace

deepseek_v4::MhcPreResult MhcPre(const std::vector<float>& residual, const HcSite& site,
                                 const Glm5NextMhcParams& mhc, int64_t hidden,
                                 float rms_norm_eps) {
  const int64_t hc = mhc.mult;
  const int64_t mix = (2 + hc) * hc;
  RequireShape(hc > 1 && hidden > 0, "`hc_mult` must be > 1 and `hidden_size` > 0");
  RequireShape(residual.size() == static_cast<size_t>(hc * hidden),
               "`residual` must be [hc_mult, hidden_size] for one token");
  RequireShape(site.fn.size() == static_cast<size_t>(mix * hc * hidden),
               "`fn` must be [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]");
  RequireShape(site.base.size() == static_cast<size_t>(mix),
               "`base` must be [(2 + hc_mult) * hc_mult]");
  RequireShape(site.scale.size() == 3, "`scale` must be [3]");

  // The five bindings this file exists to hold in one place:
  //   rms_eps            = config.rms_norm_eps  (1e-5) -- the folded weight-free
  //                        RMSNorm's epsilon, NOT hc_eps.
  //   hc_pre_eps         = config.hc_eps (1e-6), added to the `pre` gate.
  //   hc_sinkhorn_eps    = config.hc_eps, the SAME constant, added to every
  //                        Sinkhorn denominator.
  //   hc_post_mult_value = 2.0, a literal in the reference, not a config key.
  //   sinkhorn_iters     = config.hc_sinkhorn_iters (20).
  // The empty `norm_weight` is deliberate: the reference applies
  // `input_layernorm` / `post_attention_layernorm` as a separate module after
  // the collapse has been cast back to the activation dtype, so folding it here
  // would be exact in f32 and wrong in bf16.
  return deepseek_v4::MhcPre(residual, site.fn, site.scale, site.base, hc, hidden,
                             rms_norm_eps, static_cast<float>(mhc.eps),
                             static_cast<float>(mhc.eps), kHcPostAlpha, mhc.sinkhorn_iters,
                             /*norm_weight=*/{}, /*norm_eps=*/0.0f);
}

std::vector<float> MhcPost(const std::vector<float>& sublayer_out,
                           const std::vector<float>& residual,
                           const deepseek_v4::MhcPreResult& pre, int64_t hc, int64_t hidden) {
  RequireShape(hc > 1 && hidden > 0, "`hc_mult` must be > 1 and `hidden_size` > 0");
  RequireShape(sublayer_out.size() == static_cast<size_t>(hidden),
               "`sublayer_out` must be [hidden_size] for one token");
  RequireShape(residual.size() == static_cast<size_t>(hc * hidden),
               "`residual` must be [hc_mult, hidden_size] for one token");
  RequireShape(pre.post_mix.size() == static_cast<size_t>(hc),
               "`post_mix` must be [hc_mult]");
  RequireShape(pre.comb_mix.size() == static_cast<size_t>(hc * hc),
               "`comb_mix` must be [hc_mult, hc_mult]");
  return deepseek_v4::MhcPost(sublayer_out, residual, pre.post_mix, pre.comb_mix, hc, hidden);
}

std::vector<float> HcHeadCollapseMean(const std::vector<float>& hidden_streams, int64_t hc,
                                      int64_t hidden) {
  RequireShape(hc > 1 && hidden > 0, "`hc_mult` must be > 1 and `hidden_size` > 0");
  RequireShape(hidden_streams.size() == static_cast<size_t>(hc * hidden),
               "`hidden_streams` must be [hc_mult, hidden_size] for one token");
  // `hidden_streams.mean(dim=2)`, and nothing else: no RMSNorm, no projection,
  // no gate, no epsilon. The accumulator is `float`, not `double`, because the
  // reference reduces a float32 tensor in float32 and this reference exists to
  // reproduce it; over `hc_mult == 4` terms the two agree to well inside the
  // gate's tolerance either way.
  std::vector<float> out(static_cast<size_t>(hidden), 0.0f);
  for (int64_t m = 0; m < hc; ++m) {
    for (int64_t h = 0; h < hidden; ++h) {
      out[h] += hidden_streams[m * hidden + h];
    }
  }
  const float inv = 1.0f / static_cast<float>(hc);
  for (int64_t h = 0; h < hidden; ++h) out[h] *= inv;
  return out;
}

}  // namespace vllm::glm5_next
