// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5b-2: the decoder layer, the mHC
// stream threading and the assembled `Glm5NextTextModel::Forward`.
//
// Issue [#2241](https://github.com/mudler/vllm.cpp/issues/2241), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5b-2.
//
// Model-private header, deliberately not under `include/`: nothing outside this
// model needs these types, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. Same arrangement as `glm5_next.h` (W1),
// `glm5_next_mhc.h` (W4), `glm5_next_dsa.h` (W3), `glm5_next_moe.h` (W5) and
// `glm5_next_attn.h` (W5b-1).
//
// ORACLE. vLLM registers no `glm5_next` at our parity pin `555967922` nor at its
// `main`, and neither do vllm-omni, SGLang or llama.cpp. Under AGENTS.md "When
// vLLM has no implementation" the reference for this surface is `transformers`
// **v5.16.1**, this row's lane pin (`.agents/oracles/transformers.md`), whose
// `models/glm5_next/modeling_glm5_next.py` sha256 is
// `2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b` — asserted
// by the golden generator against the file the oracle actually ran, not assumed
// from a version string.
//
// ─── PORT ANCHORS (file:line on BOTH sides) ──────────────────────────────────
//   OURS                            <-  transformers v5.16.1, models/glm5_next/
//   glm5_next::DecoderLayerForward  <-  modeling_glm5_next.py:1279-1329
//                                       (`Glm5NextTextDecoderLayer.forward`),
//                                       the arm selection at :1259-1272
//   glm5_next::TextModelForward     <-  :1431-1494
//                                       (`Glm5NextTextModel.forward`)
//   the manifold expansion          <-  :1477
//   the final collapse              <-  :1493
//
// ═════════════════════════════════════════════════════════════════════════════
//  THE ONE THING THIS FILE EXISTS TO GET RIGHT: THE `[T, hc_mult, hidden]`
//  MANIFOLD
// ═════════════════════════════════════════════════════════════════════════════
//
// The whole stack carries FOUR residual streams, not one. `:1477` expands the
// embedding as `inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)` and
// NOTHING collapses it again until `self.hc_head(hidden_states)` at `:1493`,
// immediately before the final norm. Every decoder layer takes `[B, S, hc, H]`
// and returns `[B, S, hc, H]`; the collapse to `[B, S, H]` happens TWICE inside
// each layer (`:1294` and `:1321`, the `pre` weighted sum) and is UNDONE both
// times (`:1316-1318` and `:1325-1327`, the `post`/`comb` fold).
//
// **A port that threads `[T, hidden]` and collapses early RUNS.** It produces
// finite activations of the right shape, it emits fluent tokens, and no shape
// check anywhere in this tree catches it — the collapsed stream is exactly what
// the sublayers consume, so every sublayer gate still passes. What it loses is
// the four-way residual mixing the model was trained with, and the only symptom
// is that the text is subtly worse. `.agents/specs/glm5-next-flash.md` §Gates
// records that no end-to-end token gate for this model exists on this fleet, so
// there is nothing downstream that would catch it either.
//
// So the manifold is in the TYPES: `DecoderLayerForward` takes and returns
// `[batch, seq_len, hc_mult, hidden_size]`, `Glm5NextParams::residual_stream_width()`
// is the flat width (16384 on the published checkpoint), and the focused gate
// runs a deliberately EARLY-COLLAPSING reference beside the real one and asserts
// they differ by a printed margin. An assertion that only checks the final
// logits would pass a port that collapsed early and then re-expanded by
// broadcast, because the two agree on the FIRST layer and diverge only after the
// second — which is why the gate is at least two layers deep and asserts the
// per-stream values, not only the collapse.
//
// ─── WHAT `prev_topk_indices` THREADS, AND WHERE IT IS WIPED ─────────────────
//
// `:1479-1491` seeds `topk_indices = None` and feeds each layer the PREVIOUS
// layer's return. Three facts follow, and all three are upstream's arithmetic:
//
//   * a KDA layer returns `None` (`:1297` sets it and the linear arm at
//     `:1299-1304` never reassigns it), so a KDA layer WIPES the thread;
//   * a `full` DSA layer returns its own selection only when `next_skip_topk`
//     (`:1216`), and `None` otherwise;
//   * a `shared` DSA layer therefore RAISES if the layer before it was KDA or
//     was a `full` layer that does not propagate — upstream's own
//     `ValueError("Shared DSA layers require top-k indices from a previous full
//     indexer layer.")` (`:1189-1190`), mirrored verbatim by `Attention`.
//
// A port that carried the last DSA selection across an intervening KDA layer
// would silently feed a stale key set to a shared layer instead of refusing.
// The published `GLM-5.3-Flash` config selects `shared` on ZERO of its 45
// layers, so this arm is exercised by fixture alone; the spec's O25 discloses
// that and this file does not pretend otherwise.
//
// ─── `blk.45` IS THE MTP BLOCK AND IS NOT A LAYER ────────────────────────────
//
// `num_hidden_layers` is 45 and the published artifact carries 46 layer
// directories. `:1480` iterates `self.layers[: self.config.num_hidden_layers]`
// and `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]` drops the
// 46th, which is a DeepSeek-V3-style multi-token-prediction block — DSA/MLA, a
// full 288-expert MoE, and NO `hc_*` tensors at all, so it does not even run on
// the four-stream manifold. `LoadGlm5NextFromGguf` already skips its 29 tensors
// and counts them in `Glm5NextWeights::mtp_block_tensors_dropped`; this file
// refuses a weight tower whose layer count disagrees with `num_hidden_layers`
// rather than iterating whatever it was handed, so "45 layers were built from
// 0..44" is checked and not assumed.
//
// ─── NOT REACHED YET, AND THIS WAVE'S OWN SCOPE SAID IT WOULD BE ─────────────
//
// `ForwardGlm5NextForConditionalGeneration` (`glm5_next_registry.cpp`) still
// refuses by name, so at this merge commit the only call site of anything in
// this file is `tests/vllm/models/test_glm5_next_layer.cpp`. W5b-2's plan was to
// discharge O15, O16, O17, O23 and O25 — the five reachability debts the KDA
// arm, the mHC bricks, the DSA indexer, the MoE block and W5b-1's two files each
// carry — and it discharges NONE of them, because
// `.agents/reachability.md` is explicit that "an intermediate hop that is itself
// unreached does not carry".
//
// What changed is that the five had five separate dead ends and now have ONE
// assembly point, gated against the reference; what is still owed is the weight
// bridge for the four arms `BridgeDsaLayer` does not cover and the engine
// binding from `ModelForwardInput`. The MoE half of that bridge is a design
// problem and not more of the same: the 42 sparse layers' routed experts are
// ~1,150 GiB in f32 against a ~119.63 GiB box, and
// `kBridgeTensorF32ByteCeiling` correctly REFUSES a 9 GiB expert bank today, so
// what fits is an on-demand decode of the 8 experts a token actually selects.
//
// This is the staged-slice disclosure AGENTS.md "Nothing lands dead" requires,
// declared rather than claimed by silence. **W5b-2b owns the wiring**, on row
// `MODEL-MM-glm5-next-glm5-next-for-conditional-generation`, tracked by
// [#2241](https://github.com/mudler/vllm.cpp/issues/2241) under campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998); the spec's `## Owed`
// O26 carries it.
//
// ─── HOST REFERENCE, f32 ─────────────────────────────────────────────────────
//
// Every buffer here is `float`, exactly as `glm5_next_kda.cpp`,
// `glm5_next_dsa.cpp`, `glm5_next_mhc.cpp`, `glm5_next_moe.cpp` and
// `glm5_next_attn.cpp` are. Upstream computes the mHC mapping in fp32 (`:278`)
// and both RMSNorms in fp32 (`:77`), so those are its arithmetic and not a
// widening; the projections it runs in the model dtype are widened here, which
// IS a deviation and is the one this whole row already carries. The device arm
// is owed, not implied — see the spec's `## Owed`.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LAYER_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LAYER_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_attn.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_kda.h"
#include "vllm/model_executor/models/glm5_next_mhc.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vt/ops.h"  // vt::Queue

namespace vllm::glm5_next {

// One decoder layer's host-f32 weights. `Glm5NextTextDecoderLayer.__init__`
// (`:1260-1277`) with the two arm selections made explicit.
//
// The two arms are UNION-SHAPED and not both populated: exactly one of
// `kda`/`dsa` carries values, selected by `attn_kind`, and exactly one of
// `dense_mlp`/`moe`, selected by `mlp_kind`. `DecoderLayerForward` refuses a
// layer whose populated arm disagrees with its kind rather than reading an
// empty vector as a zero weight — which is the shape of every "fluent wrong
// model" failure this row's headers keep naming.
struct DecoderLayerWeights {
  // `config.layer_types[layer_idx]` (`:1261`). Selects `Glm5NextTextLinearAttention`
  // or `Glm5NextTextAttention` (`:1263-1268`).
  Glm5NextLayerKind attn_kind = Glm5NextLayerKind::kLinearAttention;
  // `config.mlp_layer_types[layer_idx]` (`:1270-1272`). Selects
  // `Glm5NextTextMoE` or `Glm5NextTextMLP`. NOT `first_k_dense_replace`, which
  // the runtime config class does not declare at all — see `glm5_next.h`.
  Glm5NextMlpKind mlp_kind = Glm5NextMlpKind::kDense;

  std::vector<float> input_layernorm;           // [hidden] — `:1274`
  std::vector<float> post_attention_layernorm;  // [hidden] — `:1275`

  HcSite attn_hc;  // `:1277` — applied at `:1294` and `:1316-1318`
  HcSite ffn_hc;   // `:1278` — applied at `:1321` and `:1325-1327`

  glm5_next_kda::Glm5NextKdaLayerWeights kda;  // iff kLinearAttention
  // Carries the MLA projections AND the indexer's own storage, with
  // `IndexerView()` rebuilding the pointer set on every call so nothing dangles
  // across a move. Reused rather than re-declared because it is exactly what
  // `BridgeDsaLayer` produces.
  BridgedDsaLayer dsa;                          // iff kDeepseekSparseAttention

  DenseMlpWeights dense_mlp;  // iff kDense
  MoeLayerWeights moe;        // iff kSparse
};

// One layer's persistent state — the host mirror of the KV-cache groups
// `MakeGlm5NextKVCache` publishes, bound per layer.
//
// A KDA layer uses `kda` (group 1, the `MambaSpec`: one `[conv_dim,
// conv_kernel_dim]` conv state and one f32 `[heads, head_dim, head_dim]`
// recurrent state) and a DSA layer uses `dsa` (group 0's 512-wide MLA latent and
// group 2's 257-wide indexer side cache). The unused member stays empty.
//
// **`kda` is a VECTOR, one entry per batch row, and that is not a convenience.**
// `Glm5NextKdaLayerForward` is a SINGLE-SEQUENCE reference — its recurrence
// carries one `[heads, head_dim, head_dim]` state — so a shared cache across a
// batch would mix one request's history into another's. The runner's own group
// is `[num_state_blocks, ...]` for the same reason.
struct LayerCache {
  std::vector<glm5_next_kda::Glm5NextKdaCache> kda;  // size() == batch, or empty
  DsaCache dsa;
};

// What one decoder layer returns (`:1329`).
struct DecoderLayerResult {
  // [batch, seq_len, hc_mult, hidden_size] row-major — the manifold, NOT a
  // collapsed `[batch, seq_len, hidden_size]`.
  std::vector<float> hidden_streams;
  // `topk_indices` at `:1329`. EMPTY when the layer propagates nothing, which is
  // every KDA layer (`:1297`) and every `full` DSA layer whose successor is not
  // `shared` (`:1216`). A caller must pass an empty selection on to the next
  // layer as "none", never carry the last non-empty one forward.
  std::vector<int32_t> topk_indices;
  int64_t topk_width = 0;
};

// `Glm5NextTextDecoderLayer.forward` (`:1279-1329`).
//
//   hidden_streams : [batch, seq_len, hc_mult, hidden_size] row-major
//   mask           : [batch, seq_len] uint8, 0 for a padding slot. ONE mask for
//                    both arms: `:1472-1475` maps `deepseek_sparse_attention`
//                    and `linear_attention` to the same tensor.
//   prev_topk_indices : the PREVIOUS layer's `topk_indices`, or null. Required
//                    by a `shared` DSA layer and ignored by every other kind.
//   cache          : may be null for a one-shot forward with no carry. When it
//                    is not null the layer reads and writes it, so a two-step
//                    run agrees with the one-shot run over the same tokens.
//   queue          : must be a CPU queue; the KDA recurrence and the MoE router
//                    dispatch on it.
DecoderLayerResult DecoderLayerForward(
    const Glm5NextParams& p, int64_t layer_idx, const DecoderLayerWeights& w,
    const std::vector<float>& hidden_streams,
    const std::vector<uint8_t>& mask,
    const std::vector<int32_t>* prev_topk_indices, int64_t prev_topk_width,
    int64_t batch, int64_t seq_len, LayerCache* cache, vt::Queue& queue);

// The assembled text stack's weights (`Glm5NextTextModel.__init__`, `:1412-1426`)
// minus the embedding table, which the caller gathers.
//
// `layers.size()` MUST equal `params.num_hidden_layers`. `TextModelForward`
// refuses any other count by name, which is what makes the `blk.45` exclusion a
// checked fact rather than a comment: a tower built from blocks 1..45 has the
// right count and the wrong blocks, but a tower that built 46 layers does not,
// and that is the failure a loader change would actually produce.
struct TextModelWeights {
  Glm5NextParams params;
  std::vector<DecoderLayerWeights> layers;
  std::vector<float> norm;  // [hidden] — `Glm5NextTextModel.norm`, `:1421`
};

// A source of ONE decoder layer's host-f32 weights, consulted per layer.
//
// **This exists for the same reason `glm5_next_bridge.h` has no `BridgeTower`.**
// A `TextModelWeights` holds all 45 layers at once, and at the published
// geometry that is the whole float tower: 426.72 GiB against a ~119.63 GiB box.
// A production forward therefore cannot build one. It walks the layers, and it
// must be able to DROP each one before it reaches the next.
//
// So the streaming overload of `TextModelForward` takes this instead of a
// tower, and the resident overload wraps a `TextModelWeights` in the trivial
// implementation. Both run the SAME loop, the same manifold and the same
// collapse; there is deliberately no second copy of that code, because a
// parallel forward is the shape AGENTS.md `## Shared seams` forbids.
class LayerWeightSource {
 public:
  virtual ~LayerWeightSource() = default;

  // How many decoder layers this source can serve. `TextModelForward` checks it
  // against `num_hidden_layers` and refuses a disagreement by name, which is
  // what keeps the `blk.45` exclusion a checked fact on the streaming path too.
  virtual int64_t size() const = 0;

  // The layer's weights. **The reference is valid only until the NEXT call.**
  // An implementation that materializes overwrites ONE slot; it must not keep a
  // map keyed by layer index, which is the tower again with a slower ramp.
  virtual const DecoderLayerWeights& Layer(int64_t layer_idx) = 0;
};

// `Glm5NextTextModel.forward` (`:1431-1494`) over a LAYER SOURCE — the shape a
// production forward takes, because it never holds two layers at once.
//
//   p     : the resolved config; `p.num_hidden_layers` must equal `layers.size()`
//   norm  : [hidden] — `Glm5NextTextModel.norm`, `:1421`
// Every other parameter is the resident overload's, unchanged.
std::vector<float> TextModelForward(const Glm5NextParams& p,
                                    const std::vector<float>& norm,
                                    LayerWeightSource& layers,
                                    const std::vector<float>& inputs_embeds,
                                    const std::vector<uint8_t>& mask,
                                    int64_t batch, int64_t seq_len,
                                    std::vector<LayerCache>* caches,
                                    vt::Queue& queue);

// `Glm5NextTextModel.forward` (`:1431-1494`), from `inputs_embeds`.
//
// It takes the EMBEDDINGS and not the token ids, mirroring upstream's own
// `inputs_embeds` parameter (`:1437`, `:1447-1448`) — and for a residency reason
// as well: the published checkpoint's embedding table is
// `[154880, 4096]`, 2.54 GiB in f32, and a caller that already holds it (or that
// gathers `seq_len` rows straight out of the quantized `OwnedTensor`) must not be
// forced to materialize the whole table to call this.
//
//   inputs_embeds : [batch, seq_len, hidden_size] row-major
//   mask          : [batch, seq_len] uint8. `:1463-1470` guarantees the mask
//                   exists for the indexer, so this is required rather than
//                   optional.
//   caches        : null for a one-shot forward, or a vector of exactly
//                   `num_hidden_layers` entries carried across steps.
// Returns         : [batch, seq_len, hidden_size] — `self.norm(self.hc_head(...))`
//                   (`:1493`), the LAST HIDDEN STATE and not logits.
std::vector<float> TextModelForward(const TextModelWeights& w,
                                    const std::vector<float>& inputs_embeds,
                                    const std::vector<uint8_t>& mask,
                                    int64_t batch, int64_t seq_len,
                                    std::vector<LayerCache>* caches,
                                    vt::Queue& queue);

// `Glm5NextTextKdaDims` from the resolved config. Every field comes from
// `p.kda` and none is defaulted here: `linear_head_dim`, `linear_num_heads` and
// `linear_conv_kernel_dim` all arrive through the `linear_attn_config`
// sub-object under different spellings, and `gate_lower_bound`'s PRESENCE
// selects the forget-gate formula (`glm5_next.h`) — its absence is the SOFTPLUS
// branch, which is Kimi-Linear's and not this model's.
//
// Public rather than file-local because the weight bridge needs the same dims
// the forward runs on, and two builders of one geometry is how a port ends up
// bridging at one shape and computing at another.
glm5_next_kda::Glm5NextKdaDims KdaDimsFrom(const Glm5NextParams& p);

// The manifold expansion at `:1477`, isolated so the one line a wrong port omits
// has a name and a gate of its own.
//
//   inputs_embeds : [batch, seq_len, hidden_size]
// Returns         : [batch, seq_len, hc_mult, hidden_size], every stream a copy
std::vector<float> ExpandToHiddenStreams(const std::vector<float>& inputs_embeds,
                                         int64_t batch, int64_t seq_len,
                                         int64_t hc_mult, int64_t hidden);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LAYER_H_
