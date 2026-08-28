// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W4: the manifold hyper-connection
// (mHC) residual topology, and the one place it is NOT DeepSeek-V4.
//
// Model-private header, deliberately not under `include/`: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. Same arrangement as `glm5_next.h` (W1) and
// `qwen4_exp.h`.
//
// ORACLE. vLLM registers no `glm5_next` at our parity pin `555967922` nor at
// its `main`, and neither do vllm-omni, SGLang or llama.cpp. Under AGENTS.md
// "When vLLM has no implementation" the reference for this surface is
// `transformers` **v5.16.1**, the commit `refs/tags/v5.16.1` resolves to,
// `93c8b7b485963a10800c91f55304db6be211c2bd`. (The annotated TAG OBJECT is
// `fb405cdf1bb6fa7b85ac8871b5d8a8b1376f5a3c`; an earlier revision of this
// comment mislabelled the commit as the tag object. The revision was always
// right, only the word for it was wrong.)
// W0 (#2096) owns recording that lane revision in
// `.agents/oracles/transformers.md`; this file cites it, it does not record it.
//
// ─── WHAT IS REUSED, AND WHAT IS NOT ─────────────────────────────────────────
//
// Three of the four mHC pieces are DeepSeek-V4's, unchanged.
// `Glm5NextTextHyperConnection` is a bare `pass` over `DeepseekV4HyperConnection`
// (`modular_glm5_next.py:364-365` @ v5.16.1), so `MhcSinkhorn`
// (`deepseek_v4_mhc.cpp:23`), `MhcPre` (`:72`) and `MhcPost` (`:149`) are called
// here as they stand. This file adds no numerics to them; what it adds is the
// BINDING of GLM-5.3-Flash's constants — `hc_mult` 4, `hc_sinkhorn_iters` 20,
// `hc_eps` 1e-6 used as BOTH the pre epsilon and the Sinkhorn epsilon,
// `hc_post_alpha` 2.0, and `rms_norm_eps` (1e-5) rather than `hc_eps` for the
// folded weight-free RMSNorm. Those five bindings are exactly where a port of
// this topology goes wrong silently, so they live in one place and are gated.
//
// The FOURTH piece is different, and reusing V4's is the trap this wave exists
// to close. `Glm5NextTextHyperHead.forward` is `hidden_streams.mean(dim=2)`
// (`modular_glm5_next.py:368-372`; flattened `modeling_glm5_next.py:298-302`),
// and its own docstring says "Unlike DeepSeek-V4, this is an unweighted mean."
// `HcHeadCollapse` (`deepseek_v4_mhc.cpp:168`) is V4's weighted collapse —
// weight-free RMSNorm, `hc_head_fn` projection, sigmoid gate, weighted sum.
// Substituting it yields a model that runs and produces fluent text through a
// WRONG final projection: at `fn == 0` and `base == 0` its gate is
// `sigmoid(0) + hc_eps` on every stream, so it returns `(2 + 4e-6)x` the mean at
// `hc_mult == 4` — plausible, never obviously wrong, and off by a factor of two.
// The published checkpoint settles it independently: `hc_{attn,ffn}_{fn,base,
// scale}` are flat on each layer and there is NO `hc_head.*` tensor at any
// layer, so there are no weights for a gated collapse to read.
//
// ─── PORT ANCHORS (file:line on BOTH sides) ──────────────────────────────────
//   OURS                       <-  transformers v5.16.1, models/glm5_next/
//   glm5_next::MhcPre          <-  modular_glm5_next.py:364-365 (`pass`) ->
//                                  modeling_glm5_next.py:267-295; ours wraps
//                                  deepseek_v4_mhc.cpp:72 (`MhcPre`) + :23
//                                  (`MhcSinkhorn`)
//   glm5_next::MhcPost         <-  modeling_glm5_next.py:1316-1318 and the
//                                  identical FFN-site line :1325-1327; ours
//                                  wraps deepseek_v4_mhc.cpp:149 (`MhcPost`)
//   glm5_next::HcHeadCollapseMean
//                              <-  modular_glm5_next.py:368-372 (flattened
//                                  modeling_glm5_next.py:298-302). NET-NEW.
//                                  NOT deepseek_v4_mhc.cpp:168.
//
// DTYPE. Every buffer here is `float`, matching `deepseek_v4_mhc.*`, because the
// reference computes the whole mHC mapping in fp32 and casts `post`/`comb` down
// to the activation dtype only at the mix (`post.to(dtype)`,
// `modeling_glm5_next.py:1316`). This is a host reference, so it carries no
// model-path storage: the bf16 rounding of the residual manifold between steps
// stays the named device seam it already is for V4 (`deepseek_v4_mhc.h`).
//
// NOT REACHED YET. `Glm5NextForConditionalGeneration` refuses at weight
// materialization and at `Forward`, by name (W1, #2067). Wiring these three
// entry points into `Glm5NextTextModel::Forward` is W5's, on the row
// MODEL-MM-glm5-next-glm5-next-for-conditional-generation, tracked by campaign
// issue #1998; `.agents/specs/glm5-next-flash.md` lists it under `## Owed`.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MHC_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MHC_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_mhc.h"
#include "vllm/model_executor/models/glm5_next.h"

namespace vllm::glm5_next {

// `2 * torch.sigmoid(...)` at `modeling_glm5_next.py:284`. It is a literal in
// the reference, not a config key, and it is the same 2.0 DeepSeek-V4 resolves
// as `hc_post_alpha` (`deepseek_v4_mhc.h`).
inline constexpr float kHcPostAlpha = 2.0f;

// One site's learned hyper-connection parameters. The checkpoint stores these
// FLAT on the layer as `hc_attn_{fn,base,scale}` and `hc_ffn_{fn,base,scale}`,
// not under the reference's `attn_hc.*` / `ffn_hc.*` module paths, and
// `hyper_connection` is in the FP8 `modules_to_not_convert` list so they ship
// unquantised. A decoder layer owns two of these.
struct HcSite {
  std::vector<float> fn;     // [(2 + hc_mult) * hc_mult, hc_mult * hidden_size]
  std::vector<float> base;   // [(2 + hc_mult) * hc_mult]
  std::vector<float> scale;  // [3] — the pre, post and comb gains, in that order
};

// The mHC "pre" block for one token: collapse the `hc_mult` residual streams
// into the single vector the sublayer consumes, and produce the `post` gate and
// the Sinkhorn `comb` matrix the matching `MhcPost` folds back.
//
//   residual : [hc_mult, hidden] row-major, the stream manifold for one token
//   hidden   : `config.hidden_size`
//   rms_norm_eps : `config.rms_norm_eps` (1e-5), NOT `hc_eps`. The reference
//                  builds `input_norm` with `eps=config.rms_norm_eps`
//                  (`modeling_glm5_next.py:257`); `hc_eps` is a different
//                  constant and is added to the gates and the Sinkhorn
//                  denominators.
//
// `MhcPre`'s optional folded attn/ffn RMSNorm is deliberately NOT taken here.
// The reference applies `input_layernorm` / `post_attention_layernorm` as a
// separate module AFTER the collapse has been cast back to the activation dtype
// (`modeling_glm5_next.py:1296`, `:1323`), so folding it would be exact in f32
// and wrong in bf16. The caller applies that norm.
deepseek_v4::MhcPreResult MhcPre(const std::vector<float>& residual, const HcSite& site,
                                 const Glm5NextMhcParams& mhc, int64_t hidden,
                                 float rms_norm_eps);

// The mHC "post" block for one token: fold the sublayer output back into the
// manifold (`modeling_glm5_next.py:1316-1318`).
//
//   new[j, h] = sum_i comb[i, j] * residual[i, h] + post[j] * sublayer_out[h]
//
// `comb` is consumed TRANSPOSED — the sum runs over the FIRST hc axis. The
// Sinkhorn result is doubly stochastic but ASYMMETRIC, so transposing it wrongly
// degrades quality silently instead of crashing.
//
// Returns [hc_mult, hidden] row-major.
std::vector<float> MhcPost(const std::vector<float>& sublayer_out,
                           const std::vector<float>& residual,
                           const deepseek_v4::MhcPreResult& pre, int64_t hc, int64_t hidden);

// The FINAL hyper-connection head collapse, for one token: an UNWEIGHTED MEAN
// over the stream axis (`modular_glm5_next.py:371-372`).
//
//   out[h] = (1 / hc) * sum_m hidden_streams[m, h]
//
// There is no projection, no gate and no epsilon, and there are no weights: the
// reference's `Glm5NextTextHyperHead` declares no parameters at all
// (`modeling_glm5_next.py:298-302`) and the checkpoint carries no `hc_head.*`
// tensor. `deepseek_v4::HcHeadCollapse` is NOT a substitute; see this header's
// preamble for what substituting it costs.
//
// The model's final RMSNorm(weight) is applied to the result afterward as a
// separate module (`modeling_glm5_next.py:1493`,
// `self.norm(self.hc_head(hidden_states))`), and is not folded here.
//
//   hidden_streams : [hc, hidden] row-major
// Returns [hidden].
std::vector<float> HcHeadCollapseMean(const std::vector<float>& hidden_streams, int64_t hc,
                                      int64_t hidden);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_MHC_H_
