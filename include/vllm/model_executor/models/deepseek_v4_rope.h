#pragma once

// DeepSeek-V4's per-layer RoPE, as a SHARED seam.
//
// It lived in `deepseek_v4.cpp`'s anonymous namespace until
// `DSV4-DSPARK-DRAFTER` W-3 needed it: the DSpark drafter derives its own KV rows
// from the projected trunk taps and applies the SAME partial rotation the trunk
// applies to its KV (`exllamav3/modules/arch_specific/dspark.py:134-155`).
//
// It is exported rather than re-implemented because DeepSeek-V4's RoPE is DUAL and
// a second copy is a second place for that rule to drift: dense layers
// (`compress_ratio == 0`) rotate with `base = rope_theta`, `freq_scale = 1`,
// `ext_factor = 0`, while the 41 compressed layers rotate with
// `base = compress_rope_theta` (160000) and a YaRN interpolation plus the
// beta_fast/beta_slow correction-dim ramp. Getting that split wrong scrambles the
// rope half of q.k on every compressed layer, and the model loses positional
// structure rather than failing.
//
// The DSpark blocks are the `"sliding"` kind, so they take the DENSE arm.

#include <cstdint>

namespace vllm::deepseek_v4 {

// YaRN correction dimension (`ds4.c:rope_yarn_corr_dim`).
double YarnCorrDim(int64_t n_dims, int64_t n_ctx_orig, double beta, double base);

// Decoupled NeoX-free pairwise RoPE over an `r`-wide (r even) rope subvector,
// applied IN PLACE to `v`. GPT-J adjacent pairs. `inverse` un-rotates, mirroring
// ds4's `sin_sign`. 1:1 port of `ds4.c:rope_tail_ext_inplace`.
void RopeInplaceLayer(float* v, int64_t r, int64_t pos, double base, double freq_scale,
                      double ext_factor, int64_t n_ctx_orig, double beta_fast,
                      double beta_slow, bool inverse = false);

}  // namespace vllm::deepseek_v4
