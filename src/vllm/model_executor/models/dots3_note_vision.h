// dots3-note VISION tower — the DENSE arm (W6a, #2512) and the PYRAMID MoE arm
// (W6b, #2613).
//
// Ported from vLLM read in the local clone `~/_git/vllm` at
// **`9035151d6`**, the merge of vllm#51255 that added the architecture. That
// SHA is written beside every anchor in this file on purpose: `dots3_note` does
// not exist at our parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, and
// upstream has ALREADY moved — `nvidia/vision_attention.py` is 477 lines at
// `9035151d6` and 494 lines at vLLM `main` `7a100bb61`. An anchor with no
// revision beside it is a line number read in the wrong tree.
//
//   vllm/models/dots3_note/nvidia/vision.py @ 9035151d6 (677 lines)
//     DotsMoEVitConfig            :27    -> Dots3NoteVisionParams
//     RMSNorm                     :108   -> vt::RmsNorm (one deliberate
//                                          rounding difference, see below)
//     DotsSwiGLUFFN               :128   -> layers::UnquantizedMlpGateUpMethod
//     DotsPatchEmbed              :321   -> the patch GEMM + RMSNorm
//     MoEVisionBlock              :352   -> Dots3NoteVisionBlockWeights
//     PatchMergerAdapter          :464   -> the adapter
//     DotsMoEVitModel             :508   -> Dots3NoteVisionForward
//       get_pos_ids_by_grid       :565   -> Dots3NoteVisionPosIds
//       rot_pos_emb               :601   -> Dots3NoteVisionRopeCache
//       forward                   :631
//   vllm/models/dots3_note/nvidia/vision_attention.py @ 9035151d6 (477 lines)
//     rotate_half                 :33
//     apply_rotary_pos_emb_vision :39    -> vt::RopeFromCache (NeoX, rotary_dim
//                                          == head_dim, [L, head_dim] cache)
//     VisionRotaryEmbedding       :52    -> Dots3NoteVisionRopeCache
//     _RMSNorm                    :97    -> the per-head q_norm/k_norm
//     _VisionAttentionBase        :134   -> the qkv/proj pair
//       _qkv_with_rope            :149   -> qk-norm BEFORE rope; the order is
//                                          load-bearing and silent when wrong
//     VisionAttentionV2           :207   -> vt::AttentionDenseFlash, causal=false
//     apply_vision_attention_residual :436
//
// THE PYRAMID ARM, ADDED BY W6b (#2613).
//   vllm/models/dots3_note/nvidia/vision.py @ 9035151d6
//     MoESwiGLUFFN               :139  -> Dots3NoteVisionMoeWeights + the MoE
//                                         branch of Dots3NoteVisionForward
//       __init__                 :142
//       router_bias  (F32)       :152-154
//       the expert list          :156-163
//       gate_weight  (F32 init)  :165-168
//       forward                  :170-218
//       the sigmoid branch       :182-183
//     PixelShuffleAdapter        :419  -> the `pixel_shuffle_mlp` adapter arm
//       _pixel_shuffle           :401
//
// THE FP8 ARM, ADDED BY W9d (#2881). This paragraph used to say the arm was
// "W9's, not W6b's" and that is now false, which is why it is rewritten rather
// than left standing.
//   vllm/models/dots3_note/nvidia/vision.py @ 9035151d6
//     enable_fp8_moe (default TRUE)  :69   -> Dots3NoteVisionParams
//     mlp_cls selection              :369  -> ResolveDots3NoteVisionMoeArm
//     _per_block_cast_to_fp8_padded  :225-239 -> Dots3NoteVisionBlockCastFp8
//     MoESwiGLUFFNFP8                :242
//       process_weights_after_loading :245-283 -> MaterializeDots3NoteVision's
//                                         cast, including `del self.experts`
//       forward                      :285-315 -> the fp8 branch of VisionMoeFfn
//   vllm/models/dots3_note/nvidia/vision_moe.py @ 9035151d6 (149 lines)
//     note_vision_fused_moe_fp8      :25   -> layers::Fp8BlockMlpGateUpMethod
//                                         + layers::Fp8BlockLinearMethod, one
//                                         expert at a time (no `vt` grouped
//                                         block-FP8 MoE op exists)
//
// AND IT IS THE ARM THE RELEASED CHECKPOINT TAKES. This paragraph said the
// opposite until PR #2947 and that claim was FALSE, so it is corrected here
// rather than deleted. It read `note_vision_fused_moe_fp8` as quantizing an
// activation of width `moe_intermediate_size`, the released 2112 as failing
// `per_token_group_quant_fp8`'s divisibility assertion, and upstream's own
// default class as raising there. `_per_block_cast_to_fp8_padded` pads each
// expert shard up to a multiple of 128 and NEVER SLICES THE PAD BACK, so the
// stored shard is 2176, `w13` is 4352, the width that assertion reads is 2176,
// and `2176 % 128 == 0`. `ResolveDots3NoteVisionMoeArm` carries the arithmetic
// and the retraction in full; it keys only on `embed_dim`, which is 1536 on the
// released checkpoint, so that checkpoint runs the FP8 class here.
//
// WHAT IS STILL OUT OF REACH, and it is two unrelated things.
//   (a) A CUDA BUILD OUTSIDE THE `cutlass-fp8` ARCH CELL `12.0a,12.1a`
//       (`cmake/CudaArchFeatures.cmake:290`) registers no
//       `vt::MatmulFp8BlockScaled` at all, so an image request against the
//       released `vision_config` is REFUSED BY NAME there. It used to be
//       ANSWERED, on the bf16 class, and the released checkpoint changing arm
//       is what changed that. `thor:gpu0` (sm_110) and `orin:gpu0` (sm_87) are
//       both outside the cell and both are this row's own e2e hosts.
//       `VisionMoeFfn`'s `Dots3NoteVisionFp8UnrunnableHere` guard is where this
//       is decided and said; the spec's `## Owed` carries the unblocking
//       condition, which is a block-scaled FP8 GEMM registered for those
//       arches. W9d cited #1189 milestone M5 as its owner; that issue is
//       DELETED and 404s, so the condition is stated with no number and #3007
//       owns re-establishing the live one.
//   (b) A blockwise-QUANTIZED CHECKPOINT, the `-fp8` sibling that ships
//       `weight_scale_inv` on disk instead of the bf16 experts this cast
//       converts at load. `Dots3NoteVisionRefusal` still names W9 for it, it is
//       a different thing from this arm, and it is still owed.
// See `.agents/specs/dots3-note.md` §4.11, §4.12 and §4.20.
//
// WHY IT SHARES NO CODE WITH `qwen3_vl_vision`. The two towers agree on the
// block OUTLINE and on almost nothing below it: RMSNorm vs LayerNorm, no bias
// anywhere vs bias everywhere, a per-head qk-norm applied BEFORE rope, a
// three-tensor SwiGLU vs a two-tensor GELU MLP, a patch-merger adapter vs a
// pixel-shuffle merger, no DeepStack, no interpolated position-embedding table.
// Extending `Qwen3VLVisionConfig` to carry all of that would be a parallel path
// wearing one struct's name. What IS shared is every seam underneath: the
// `vt::` ops, `dense_attn`'s device glue, `dense_loaders`' weight readers and
// `layers::MlpGateUpMethodBase` for the mergeable projections.
//
// THE ONE DELIBERATE FORMULA DIFFERENCE, named rather than left to be found.
// Upstream's `RMSNorm.forward` (`vision.py:114-116`) is
// `self._norm(x.float()).type_as(x) * self.weight` — it rounds BACK to the
// activation dtype before multiplying by the weight. `vt::RmsNorm` keeps f32
// through the weight multiply and rounds once on the store (`vt/ops.h`'s own
// note on it). Using the shared op is the seam rule; the difference is one bf16
// rounding step, it is what the gate's tolerance carries, and the in-test
// double-precision reference mirrors UPSTREAM rather than this file so that the
// difference is measured rather than defined away.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_
#define VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Fp8BlockWeight
#include "vt/backend.h"
#include "vt/device.h"  // vt::DeviceType

namespace vllm {

class SafetensorsFile;
struct HfConfig;
struct Dots3NoteTensor;

// `DotsMoEVitConfig.__init__` (`vision.py:30-105` @ `9035151d6`), reduced to
// the fields this arm reads plus the ones it REFUSES on. Every default here is
// upstream's own default, so a config that omits a key gets what upstream's
// constructor would have given it — never a value chosen locally.
struct Dots3NoteVisionParams {
  // False when `config.json` carries no `vision_config` at all. Upstream builds
  // no `DotsMoEVitModel` in that case (`multimodal.py:113-118` @ `9035151d6`,
  // the `vision_config_dict is not None` guard), and neither does this port.
  bool present = false;

  int64_t embed_dim = 1536;
  // The TEXT tower's `hidden_size`, copied from the LANGUAGE config at parse.
  // It is NOT a `vision_config` key, and it replaces one that was: upstream's
  // `vision_config.hidden_size` is a SECOND copy of the same number, nothing
  // read it, and a refusal that read it would still not be the predicate the
  // encoder applies. `EncodeMmDots3NoteForCausalLM` compares `adapter_out_dim`
  // against `config.hidden_size`, so that is the value that has to be here for
  // `Dots3NoteVisionRefusal` to answer the same question the route asks.
  int64_t text_hidden_size = 0;
  int64_t intermediate_size = 4224;  // the DENSE SwiGLU width
  int64_t moe_intermediate_size = 2112;  // the PYRAMID expert width (W6b); read only to report it
  int64_t num_hidden_layers = 42;
  int64_t num_attention_heads = 24;
  int64_t num_channels = 3;
  int64_t patch_size = 14;
  int64_t spatial_merge_size = 2;
  int64_t temporal_patch_size = 1;
  double rms_norm_eps = 1e-5;
  bool use_bias = false;
  bool use_qk_norm = true;
  bool is_causal = false;
  bool post_norm = true;
  // `pre_pixel_shuffle` (`vision.py:61-64`): when TRUE the PREPROCESSOR already
  // emits patch rows in 2x2-grouped order (`common/processor.py:185-197`), so
  // the RoPE positions are regrouped to match (`get_pos_ids_by_grid:565-574`).
  // It is NOT the same switch as `adapter_type`, and conflating the two is the
  // reading #2512's prose invites — see `.agents/specs/dots3-note.md` §4.11.1.
  //
  // FALSE is `DotsMoEVitConfig`'s own default (`vision.py:64`), which is why it
  // is the default here; the RELEASED checkpoint sets it TRUE in its
  // `vision_config` and the parser reads that. The struct default matters
  // because this field selects between two INCOMPATIBLE token orders, so a
  // default-constructed params must not silently claim the regrouped one.
  bool pre_pixel_shuffle = false;

  // The per-block routed-expert counts (`vision.py:91`). `is_moe` is
  // `pyramid_num_routed[i] > 0` (`vision.py:363-366`), so the released
  // checkpoint's leading 25 entries of `-1` are DENSE and the trailing 17
  // (4, 8, ... 60, 64, 64) are the pyramid W6b computes.
  std::vector<int64_t> pyramid_num_routed;
  double capacity_factor = 2.0;
  std::string router_scoring_func = "sigmoid";
  double router_scale = 1.0;

  // `enable_fp8_moe` (`vision.py:69` @ `9035151d6`), the key that chooses
  // between upstream's TWO pyramid-MoE classes:
  //
  //   mlp_cls = MoESwiGLUFFNFP8 if config.enable_fp8_moe else MoESwiGLUFFN
  //                                                        (`vision.py:369`)
  //
  // TRUE is `DotsMoEVitConfig.__init__`'s own default, and it is the default
  // here for that reason and no other. The RELEASED `vision_config` does not
  // carry the key -- verified on `dots-studio/dots3-note-prev`, on the `-fp8`
  // sibling, and in this repo's committed fixture -- so the default applies and
  // upstream selects the FP8 class on the checkpoint this tree already serves.
  // W6b ported the OTHER class and nothing here read the key at all; #2881 is
  // that finding.
  //
  // Reading it IS taking that branch on the released checkpoint. This comment
  // used to say the opposite -- that upstream's FP8 class cannot execute on the
  // released geometry -- and that was FALSE: the weight cast pads 2112 to 2176
  // and does not slice the pad back, so nothing raises. See
  // `ResolveDots3NoteVisionMoeArm`, which keys only on `embed_dim` and carries
  // the retracted arithmetic in full.
  bool enable_fp8_moe = true;

  // The adapter (`vision.py:419-496`). `patch_merger` is the arm the released
  // checkpoint selects; `pixel_shuffle_mlp` is a DIFFERENT token order from the
  // same pixels AND a different state dict, and W6b implements it too.
  std::string adapter_type = "pixel_shuffle_mlp";
  int64_t adapter_in_dim = 1536;
  int64_t adapter_out_dim = 2048;
  int64_t adapter_merge_size = 2;

  int64_t head_dim() const { return embed_dim / num_attention_heads; }
  // `merged_dim = in_dim * merge_size**2` (`vision.py:431` for
  // `pixel_shuffle_mlp`, `:478` for `patch_merger`).
  int64_t merged_dim() const {
    return adapter_in_dim * adapter_merge_size * adapter_merge_size;
  }
  // One patch row as the processor ships it:
  // `channel * temporal_patch_size * patch_size * patch_size`
  // (`common/processor.py:213` @ `9035151d6`).
  int64_t patch_row() const {
    return num_channels * temporal_patch_size * patch_size * patch_size;
  }
  // `topk = min(int(self.capacity_factor), self.num_routed)`
  // (`vision.py:190` @ `9035151d6`). The truncation to `int` is upstream's own,
  // and it is why `capacity_factor` 2.0 selects TWO experts rather than a
  // fraction of a capacity. Returns 0 for a dense block, which has no router.
  int64_t routed_top_k(int64_t layer) const {
    if (!is_moe_block(layer)) return 0;
    const int64_t ne = pyramid_num_routed[static_cast<size_t>(layer)];
    const int64_t cf = static_cast<int64_t>(capacity_factor);  // int(), :180
    return cf < ne ? cf : ne;
  }
  bool is_moe_block(int64_t layer) const {
    return layer >= 0 &&
           layer < static_cast<int64_t>(pyramid_num_routed.size()) &&
           pyramid_num_routed[static_cast<size_t>(layer)] > 0;
  }
  // How many DENSE blocks this config has, counted over ALL blocks rather than
  // as a leading run: a config that interleaved them would report the truth
  // instead of the length of its first run.
  int64_t num_dense_blocks() const {
    int64_t n = 0;
    for (int64_t i = 0; i < num_hidden_layers; ++i)
      if (!is_moe_block(i)) ++n;
    return n;
  }
  int64_t num_moe_blocks() const { return num_hidden_layers - num_dense_blocks(); }
};

// WHICH pyramid-MoE class this tower runs, and WHY -- `vision.py:363-374` @
// `9035151d6` asked here rather than at every use site.
//
// Upstream's selection is one line (`mlp_cls = MoESwiGLUFFNFP8 if
// config.enable_fp8_moe else MoESwiGLUFFN`, `:369`), and on the released
// checkpoint it selects the FP8 class AND THAT CLASS RUNS. W9d shipped the
// opposite claim and it was FALSE; the correction is the substance of this
// block, so it is written out rather than quietly deleted.
//
// WHAT W9d GOT WRONG. It read `_per_block_cast_to_fp8_padded` as a cast that
// pads for tiling and hands back the ORIGINAL extent, concluded that upstream's
// `w13` is `[2 * moe_intermediate_size, embed_dim]` = `[4224, 1536]`, that
// `activated_size = intermediate_size // 2` is therefore 2112, and that
// `per_token_group_quant_fp8`'s `assert x.shape[-1] % group_size == 0`
// (`fp8_utils.py:563-566`) fires on `2112 % 128 == 64`. The pad is not
// droppable and the extent is not the original one:
//
//   `_per_block_cast_to_fp8_padded` (`vision.py:225-239`) builds
//   `weight.new_zeros(ceil(rows,128), ceil(cols,128))`, copies the weight into
//   its corner, and calls `per_block_cast_to_fp8(padded, use_ue8m0=False,
//   gran_k=128)` on THAT. It never slices the result back.
//
//   `per_block_cast_to_fp8` (`deep_gemm/utils/math.py:51-61` @ DeepGEMM
//   `e21c821f39a2056d68067a466c64ddc942200106`, the revision
//   `cmake/external_projects/deepgemm.cmake:33` pins and `:168-176` vendors
//   into `vllm.third_party.deep_gemm`, which is the module `vision.py:14`
//   imports) DOES slice, to `[:m, :n]` -- but `m, n` is the shape of ITS OWN
//   input, which is the ALREADY-PADDED matrix. On the released expert that is
//   `[:2176, :1536]`, i.e. the identity.
//
// So fc1 casts to `(2176, 1536)`, `torch.cat((w1, w3), dim=0)`
// (`vision.py:258`) is `(4352, 1536)`, `intermediate_size` read at
// `vision_moe.py:47` is 4352, `activated_size` at `:70` is 2176, and
// `2176 % 128 == 0`. Upstream does not raise.
//
// THE PADDING IS LOAD-BEARING, WHICH IS WHY UPSTREAM WRITES IT. It is what
// makes the `w13` concatenation well-defined at all. A block scale is indexed
// by `n / 128`, so the concatenated scale grid has to have as many rows as the
// concatenated operand has block rows:
//
//   unpadded: `cdiv(2112,128)` = 17 per shard, 34 stacked, against
//             `cdiv(4224,128)` = 33 for the merged operand -- a MISMATCH, and
//             the geometry `dense_fp8_block::CheckFp8BlockMergeable` refuses.
//   padded:   `cdiv(2176,128)` = 17 per shard, 34 stacked, against
//             `cdiv(4352,128)` = 34 -- exact, and every shard but the last is
//             a whole multiple of `block_n`, which is that checker's rule.
//
// The pad is numerically inert and structurally required, and W9d's "the
// padding cannot move a scale and is ported anyway" was true of the scale
// VALUES and false about the SHAPE. `fp8_utils.py:563` reads the shape.
//
// TWO FIELDS, THREE OUTCOMES, AND THE MIDDLE ONE IS NARROW. The struct has
// two members and this predicate has three answers; they are not the same
// count, and the list below is the three:
//   `enable_fp8_moe` false                  -> bf16, `upstream_raises` empty.
//       Upstream's own other branch, and W6b's arm, unchanged.
//   true, `embed_dim` not a multiple of 128 -> bf16, `upstream_raises` set.
//       A RECORDED DIVERGENCE and not a silent fallback: upstream raises here
//       and this tree answers. `MaterializeDots3NoteVision` writes the notice
//       to stderr once and the gate asserts its text.
//   true, `embed_dim` 128-aligned           -> fp8. THE RELEASED CONFIG IS
//       HERE, because `embed_dim` is 1536.
//
// WHY ONLY `embed_dim` SURVIVES AS A REFUSAL. Padding fixes every width the
// EXPERTS own and does nothing for the first quantization, which is over the
// activation the tower hands in: `per_token_group_quant_fp8(hidden_states,
// 128, use_ue8m0=False)` at `vision_moe.py:77-81`, whose last dimension is
// `embed_dim` and which no pad reaches. `moe_intermediate_size` no longer
// appears in this predicate at any value, because `activated_size` is derived
// from the PADDED `w13` and is 128-aligned by construction. The second
// quantization (`:119-123`) therefore cannot fail once the first one passed.
struct Dots3NoteVisionMoeArm {
  // `MoESwiGLUFFNFP8` is the class this CONFIG selects, and this GEOMETRY can
  // express it. That is the ONLY question this field answers:
  // `ResolveDots3NoteVisionMoeArm` takes no device, so `true` does not say the
  // arm executes here. `Dots3NoteVisionFp8UnrunnableHere` below answers the
  // DEVICE question, and it refuses a `true` read on `thor:gpu0` (sm_110) and
  // `orin:gpu0` (sm_87).
  bool fp8 = false;
  // Non-empty exactly when `enable_fp8_moe` selected the FP8 class and this
  // tower runs the bf16 one anyway. Names the width, the assertion and the
  // anchor.
  std::string upstream_raises;
};

Dots3NoteVisionMoeArm ResolveDots3NoteVisionMoeArm(
    const Dots3NoteVisionParams& v);

// WOULD `VisionMoeFfn` REFUSE ON THIS DEVICE? The forward's arch guard, named
// rather than spelled inline, so a gate can ask the production predicate itself
// instead of re-deriving the conjunction beside it.
//
// It is exactly `arm.fp8 && !dense_fp8_block::BlockFp8Runnable(device)`. Both
// halves matter and neither is a config property: the FIRST is decided by
// `ResolveDots3NoteVisionMoeArm` and is TRUE on the released `vision_config`
// since PR #2947 corrected the resolver, and the SECOND is decided by which
// arch cell this build compiled -- `vt::MatmulFp8BlockScaled` is registered
// only for `cutlass-fp8` `12.0a,12.1a`. Their conjunction is the behaviour
// change that arm correction brings with it: on a CUDA build outside that cell
// the released checkpoint is now refused BY NAME where the bf16 class used to
// answer. `dense_fp8_block::BlockFp8Runnable` is declared in
// `dense_fp8_block_gemm.h`, which this header deliberately does not pull in, so
// this is a declaration and the definition sits beside the call site.
bool Dots3NoteVisionFp8UnrunnableHere(const Dots3NoteVisionMoeArm& arm,
                                      vt::DeviceType device);

// `_ceil_to_multiple(value, 128)` (`vision.py:222-223` @ `9035151d6`), which is
// DeepGEMM's own `align` (`deep_gemm/utils/math.py:9-10` @ `e21c821f`). Exposed
// because every consumer of a cast expert shard needs the PADDED extent and
// deriving it twice is how the two would drift apart.
int64_t Dots3NoteVisionFp8PadTo128(int64_t extent);

// `_per_block_cast_to_fp8_padded` (`vision.py:225-239` @ `9035151d6`) over
// `per_block_cast_to_fp8` (`deep_gemm/utils/math.py:51-61` @ DeepGEMM
// `e21c821f39a2056d68067a466c64ddc942200106`), as a WEIGHT-side caster.
//
// THE MODULE, because the obvious citation is the wrong one. `vision.py:14`
// imports `per_block_cast_to_fp8` from `vllm.third_party.deep_gemm`, which is
// DeepGEMM VENDORED at build time -- `cmake/external_projects/deepgemm.cmake:33`
// pins the revision and `:168-176` installs `deep_gemm/utils/*.py` under
// `vllm/third_party/deep_gemm/utils`. It is NOT `vllm/utils/deep_gemm.py`, whose
// same-named function has the signature `(x, block_size: list[int], use_ue8m0)`
// and no `gran_k` at all (`:662-664` @ `9035151d6`); `vision.py:238` passes
// `gran_k=block_size`, so a call into that module would `TypeError`. The two
// bodies also differ: the vendored one hard-codes `448.0` where vLLM's calls
// `get_fp8_min_max()`.
//
// `w` is a raw-NK `[N, K]` bf16 torch Linear weight. The result is upstream's
// pair AT THE PADDED EXTENT: `packed` the e4m3fn bytes at
// `[align(N,128), align(K,128)]` and `scale` the f32
// `[cdiv(N,128), cdiv(K,128)]` grid, `block_n`/`block_k` both 128. It is an
// `Fp8BlockWeight` because that is what `layers::Fp8BlockLinearMethod` and
// `dense_fp8_block::MatmulFp8BlockScaledD` consume, so the FP8 arm rides the
// shared block-FP8 seams rather than a caster-specific path.
//
// THE PAD IS EMITTED, NOT DROPPED, AND THAT IS THE WHOLE POINT. Upstream pads
// and never slices back (the two-function chain is spelled out on
// `Dots3NoteVisionMoeArm` above), so the shard this returns is the shard
// upstream stacks. Dropping it -- W9d's defect -- makes the concatenated `w13`
// scale grid disagree with the concatenated operand's block rows by one row on
// the released geometry, which is the case
// `dense_fp8_block::CheckFp8BlockMergeable` refuses by name.
//
// THE PAD IS NUMERICALLY INERT, and the two independent reasons are both
// structural rather than tolerated: `x_padded` is zero-filled and amax is a max
// of ABSOLUTE values, so a pad lane never raises a block's amax and no scale
// moves; and `(0 * (1/sf))` encodes to the e4m3 zero byte, so a pad ROW of the
// gate half and of the up half give `SiLU(0) * 0 = 0`, against pad COLUMNS of
// `w2` that are zero as well.
//
// THE THREE CONSTANTS THAT ARE NOT `vt::QuantFp8Group`'s, because a reader who
// assumes the two quantizers agree will get all three wrong:
//   * the amax floor is `clamp(1e-4)` (`math.py:57`), NOT the `1e-10` the
//     ACTIVATION quantizer seeds its reduction with (`per_token_group_quant.cu
//     :47`). Two different upstream kernels, two different constants.
//   * the scaling is a RECIPROCAL MULTIPLY, `x * (1.0 / sf)` (`math.py:60`),
//     where `QuantFp8Group` ships a DIVIDE and `include/vt/ops.h` says at
//     length not to "correct" that divide into a multiply. The polarity is
//     reversed here, for the same reason: mirror what upstream ships at THIS
//     site.
//   * `use_ue8m0` is FALSE (`vision.py:237`), so the scales are FP32 and are
//     never rounded onto the e8m0 lattice.
Fp8BlockWeight Dots3NoteVisionBlockCastFp8(const OwnedTensor& w, int64_t n,
                                           int64_t k);

// Resolve + validate `config.json`'s `vision_config`. Returns `present=false`
// when the key is absent. Throws (VT_CHECK) naming the key on a value this arm
// cannot represent AND that no later brick owns — a shape that IS owed to a
// later brick is reported by `Dots3NoteVisionRefusal` instead, because a
// checkpoint whose tower is owed must still LOAD its language half.
Dots3NoteVisionParams ParseDots3NoteVisionParams(const HfConfig& config);

// Why the vision tower cannot be materialized, or "" when it can. Names ONE
// thing — the first unrepresentable feature in brick order — and the brick that
// owes it. A non-empty answer leaves the 2195 `vision_encoder.*` tensors in the
// accounting's existing `vision` deferral bucket, exactly as before W6a, so
// every W2 count assertion is unchanged.
std::string Dots3NoteVisionRefusal(const Dots3NoteVisionParams& v,
                                   const std::string& quant_method,
                                   const std::vector<int64_t>& weight_block_size);

// The same answer from a CONFIG alone, for a caller that holds a checkpoint
// directory and no loaded model. The multimodal CHAT seam is that caller, and
// it is the reason this overload exists rather than a convenience.
//
// A refusal raised from `encode_mm` is FATAL: it is thrown inside the engine's
// busy loop, which stops `AsyncLLM` and turns every later request — including
// TEXT ones — into a 500. Measured on this row's served-request gate before
// this function existed. The entrypoint is where a "this server cannot serve
// images for this checkpoint" answer belongs, and `InstallMultiModalChatSeam`
// already has the shape for it: a factory that throws installs a REFUSING seam,
// which answers an image request with HTTP 400 naming the architecture and the
// reason while the text path keeps working. The `encode_mm` check stays as
// defence in depth, on the same polarity as Qwen3-VL's ("reaching this point is
// a defect").
std::string Dots3NoteVisionRefusalFor(const HfConfig& config);

// One DENSE block's weights, by the names the checkpoint ships
// (`vision_encoder.blocks.{B}.*`). Every tensor is BF16 on disk and BF16 here:
// the released index carries 37944 BF16 + 62 F32 and every F32 of those is a
// `router_bias` or an `e_score_correction_bias`, neither of which this arm
// reads. Widening any of these would be invisible to a token gate and is what
// `porting.md`'s memory-format rule is about.
// One PYRAMID block's `mlp` (`MoESwiGLUFFN`, `vision.py:139-218` @ `9035151d6`).
//
// THE SPELLING IS NOT THE LANGUAGE TOWER'S, and conflating the two is the
// obvious way to get this brick wrong. The vision router is `mlp.gate_weight` +
// `mlp.router_bias` (`vision.py:152-168`); `Dots3NoteMoeWeights` in
// `dots3_note.h` reads `mlp.gate.weight` + `mlp.gate.e_score_correction_bias`
// (`deepseek_v2.py:313-318`). The two are different tensors with different
// ranks in the same checkpoint.
struct Dots3NoteVisionMoeWeights {
  int64_t num_routed = 0;  // `pyramid_num_routed[layer]` (`vision.py:147`)
  int64_t top_k = 0;       // `min(int(capacity_factor), num_routed)` (`:190`)

  // THE ROUTER. `gate_weight` is BF16 on disk — upstream registers the
  // parameter f32 (`vision.py:165-168`) and the published bf16 checkpoint
  // stores it bf16, then `forward` re-widens it with `.float()` (`:180`). The
  // OPERANDS therefore stay bf16 here and only the GEMM's OUTPUT is f32: a
  // bf16 x bf16 product is exact in f32, so a bf16-operand GEMM with an f32
  // accumulator IS `F.linear(x.float(), w.float())`, and widening the stored
  // operand would double the resident bytes for no information at all
  // (porting.md's memory-format rule, pointed the other way).
  OwnedTensor gate_weight;  // [num_routed, E] BF16

  // ...and this one really is F32, which is UPSTREAM'S OWN CHOICE:
  // `register_buffer("router_bias", torch.zeros(num_routed,
  // dtype=torch.float32))` (`vision.py:152-154`). These 17 buffers are exactly
  // the 17 F32 tensors the released vision tower carries against 2178 BF16
  // ones, measured in the committed shard index. A token gate cannot see a
  // dtype that is too WIDE, so the load asserts this one by name in both
  // directions rather than accepting whatever the file holds.
  OwnedTensor router_bias;  // [num_routed] F32

  // THE EXPERTS, as TWO operands rather than one merged [2Im, E].
  // `layers::UnquantizedMlpGateUpSplitMethod` is the `MlpGateUpMethodBase`
  // member for exactly this case, and its own prose says why: merging costs a
  // COPY out of the mmap, and on the released checkpoint that copy is 608
  // experts x 2 x 2112 x 1536 x 2 B = **7.9 GiB** of resident bytes bought for
  // one fewer kernel launch. The DENSE blocks keep W6a's merged operand, where
  // the same copy is 649 MiB across 25 blocks and the merge is already paid.
  std::vector<OwnedTensor> expert_gate;  // num_routed x fc1 [Im, E]
  std::vector<OwnedTensor> expert_up;    // num_routed x fc3 [Im, E]
  std::vector<OwnedTensor> expert_down;  // num_routed x fc2 [E, Im]

  // ── the FP8 ARM's operands (W9d, #2881) ──────────────────────────────────
  // `MoESwiGLUFFNFP8.process_weights_after_loading` (`vision.py:245-283` @
  // `9035151d6`) casts the bf16 experts to block FP8 AT LOAD and then
  // `del self.experts`. Both halves are mirrored: these are populated only when
  // `Dots3NoteVisionMoeArm::fp8`, and the three bf16 vectors above are CLEARED
  // in the same step, so the FP8 tower does not carry two copies of 608
  // experts. On the bf16 arm these stay empty and nothing above moves.
  //
  // UPSTREAM'S `w13` IS A MERGE, AND IT RIDES THE MERGED SEAM RATHER THAN A
  // HAND-BUILT OPERAND. `process_weights_after_loading` casts fc1 and fc3
  // SEPARATELY and then `torch.cat((w1, w3), dim=0)` (`vision.py:255-259`), so
  // the arm runs ONE `[2Im, E]` GEMM and a SwiGLU tail. That is exactly
  // `layers::Fp8BlockMlpGateUpMethod` over `vt::kFp8BlockGateUpSwiGLU`, which
  // is `MlpGateUpMethodBase` and `MergedGemmGroup` -- the two seams AGENTS.md
  // names for a mergeable MLP pair. The shards are therefore kept SPLIT here
  // and `dense_fp8_block::ResidentFp8BlockMerged` concatenates them once,
  // lazily, on the device; nothing concatenates fp8 bytes by hand.
  //
  // The order matters and is upstream's: cast each half, then merge. A merged
  // bf16 operand cast as one piece is a DIFFERENT scale grid whenever `Im` is
  // not a multiple of `block_n`, because a block scale is indexed by
  // `n / block_n`. Casting first is also what makes the merge REPRESENTABLE:
  // each shard leaves the caster at `align(Im, 128)` rows, so every shard but
  // the last is a whole multiple of `block_n` and the stacked scale grid has
  // exactly `cdiv(2 * align(Im, 128), 128)` rows. That is the rule
  // `dense_fp8_block::CheckFp8BlockMergeable` enforces, and the released
  // geometry satisfies it only BECAUSE of the pad: 17 + 17 == 34 ==
  // cdiv(4352, 128), where the unpadded reading gives 34 against
  // cdiv(4224, 128) == 33.
  std::vector<Fp8BlockWeight> expert_gate_fp8;  // num_routed x fc1 [Imp, E]
  std::vector<Fp8BlockWeight> expert_up_fp8;    // num_routed x fc3 [Imp, E]
  std::vector<Fp8BlockWeight> expert_down_fp8;  // num_routed x fc2 [E, Imp]
  // The lazily-built merged `gate_up` device operand, one per expert. Mutable
  // state on a const weight, exactly as `Fp8BlockWeight::d_packed` is.
  std::vector<Fp8BlockMergedResident> expert_gateup_merged;
};

struct Dots3NoteVisionBlockWeights {
  OwnedTensor norm_1;   // [E]
  OwnedTensor norm_2;   // [E]
  OwnedTensor qkv;      // [3E, E]   (NO bias: use_bias == false)
  OwnedTensor proj;     // [E, E]
  // Empty when `use_qk_norm` is false: upstream then builds no `q_norm`/`k_norm`
  // at all (`vision_attention.py:145-147` @ `9035151d6`) and the checkpoint
  // ships neither.
  OwnedTensor q_norm;   // [head_dim]
  OwnedTensor k_norm;   // [head_dim]
  // ── the DENSE arm's mlp (`DotsSwiGLUFFN`, `vision.py:128-136`) ────────────
  // fc1 (the SwiGLU gate) and fc3 (the up projection) MERGED into one [2I, E]
  // raw-NK operand, so the pair rides `layers::MlpGateUpMethodBase` rather than
  // a hand-written parallel path (AGENTS.md, "Shared seams"). The order is
  // gate-then-up because `vt::SiluAndMul` reads `silu(x[:, :I]) * x[:, I:]` and
  // upstream is `fc2(F.silu(fc1(x)) * fc3(x))` (`vision.py:136`).
  // EMPTY on a routed block, where `moe` carries the mlp instead.
  OwnedTensor gate_up;  // [2I, E] = concat(fc1, fc3)
  OwnedTensor down;     // fc2 [E, I]

  // ── the PYRAMID arm's mlp, when `is_moe` ─────────────────────────────────
  // `MoEVisionBlock.__init__` picks one or the other on
  // `pyramid_num_routed[layer_number] > 0` (`vision.py:363-374`), which is the
  // same predicate `Dots3NoteVisionParams::is_moe_block` spells.
  bool is_moe = false;
  Dots3NoteVisionMoeWeights moe;
};

struct Dots3NoteVisionWeights {
  bool present = false;
  // Which MoE class this tower runs (`vision.py:369`), resolved ONCE at
  // materialization from the params so the forward cannot ask a different
  // question than the loader answered.
  Dots3NoteVisionMoeArm moe_arm;
  OwnedTensor patch_proj_w;  // [E, C*tp*p*p]  (on disk [E, C, p, p])
  OwnedTensor patch_proj_b;  // [E]
  OwnedTensor patch_norm;    // [E]
  // ALL 42 blocks, in order, dense and routed alike. W6a held only the dense
  // ones because it refused a routed tower; W6b computes both, so the index
  // into this vector IS the layer number and no re-mapping stands between the
  // config's `pyramid_num_routed[i]` and the weights it describes.
  std::vector<Dots3NoteVisionBlockWeights> blocks;
  // Empty when `post_norm` is false. Upstream builds NO substitute module
  // there — there is no `nn.Identity` anywhere in the file: `__init__` creates
  // the `post_trunk_norm` attribute only under `if config.post_norm:`
  // (`vision.py:525-526` @ `9035151d6`) and `forward` guards the call with the
  // same flag (`:673-674`), so on a false flag the step is absent rather than
  // an identity. The checkpoint ships no tensor for it.
  OwnedTensor post_trunk_norm;  // [E]
  // The adapter, under EITHER spelling. `patch_merger` fills `ln_q` + `mlp.0` +
  // `mlp.2`; `pixel_shuffle_mlp` fills `proj.0` + `proj.1` + `proj.3` into the
  // same four slots, and the two disagree on more than the names — see
  // `MaterializeDots3NoteVision` for the shape table.
  OwnedTensor adapter_ln_w;     // ln_q [in_dim]      | proj.0 [merged_dim]
  OwnedTensor adapter_ln_b;     // ln_q [in_dim]      | proj.0 [merged_dim]
  OwnedTensor adapter_mlp0_w;   // mlp.0 [M, M]       | proj.1 [out_dim, M]
  OwnedTensor adapter_mlp0_b;   // mlp.0 [M]          | proj.1 [out_dim]
  OwnedTensor adapter_mlp2_w;   // mlp.2 [out_dim, M] | proj.3 [out_dim, out_dim]
  OwnedTensor adapter_mlp2_b;   // mlp.2 [out_dim]    | proj.3 [out_dim]
};

// Every `vision_encoder.*` name the tower claims, with its named consumer.
// Over the released checkpoint this is all 2195 vision tensors: W6a's 235 dense
// ones plus the 1960 the pyramid adds (17 blocks x 8 block tensors + 608 routed
// experts x 3).
std::vector<Dots3NoteTensor> EnumerateDots3NoteVisionTensors(
    const Dots3NoteVisionParams& v);

// Read the tower out of `shards`, dense and routed blocks alike. REFUSES BY
// NAME on the first tensor
// whose shape disagrees with the config. Only called when
// `Dots3NoteVisionRefusal` is empty.
Dots3NoteVisionWeights MaterializeDots3NoteVision(
    const std::vector<SafetensorsFile>& shards, const Dots3NoteVisionParams& v);

// `get_pos_ids_by_grid` (`vision.py:565-599` @ `9035151d6`) for ONE grid.
// Returns L = t*h*w pairs {h_pos, w_pos} in the tower's token order.
std::vector<std::array<int64_t, 2>> Dots3NoteVisionPosIds(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v);

// `VisionRotaryEmbedding` + `rot_pos_emb` (`vision_attention.py:52-89`,
// `vision.py:601-607`) folded into the [L, head_dim] = [cos(hd/2) | sin(hd/2)]
// cache `vt::RopeFromCache` consumes. f32 host precompute, deterministic.
std::vector<float> Dots3NoteVisionRopeCache(
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionParams& v);

// One routed block's ROUTING DECISION, captured for the gate.
//
// WHY THE GATE NEEDS THIS AND A TOLERANCE DOES NOT SUFFICE. Top-k expert
// selection is a DISCRETE choice: its error is bimodal, not continuous. A
// selection either flips or it does not, and when it does not, no output
// tolerance is measuring it at all. A relative bound on the tower's output
// therefore cannot see a selection defect that happens not to have flipped on
// the fixture, and cannot report how close it came to flipping. The gate
// asserts SET equality against the reference's own selection and prints the
// minimum decision MARGIN — the gap between the last selected and the first
// rejected biased score — so the reader knows how much room the assertion had.
struct Dots3NoteVisionMoeRoute {
  int64_t block = 0;
  int64_t num_routed = 0;
  int64_t top_k = 0;
  std::vector<float> logits;    // [L, num_routed] f32, straight off the router GEMM
  std::vector<int32_t> ids;     // [L, top_k] the SELECTED expert ids
  std::vector<float> weights;   // [L, top_k] f32, post-renormalize, post-scale
  // THE COMBINE'S DENOMINATOR, per token, and the reason this field exists:
  // the two upstream classes divide by DIFFERENT things and no output
  // tolerance separates them on a fixture where the two happen to agree.
  // `MoESwiGLUFFN` accumulates `aggregated_gate` in the ACTIVATION dtype
  // (`vision.py:188`, bf16) over addends already rounded to bf16 (`:200`) and
  // adds `1e-9` (`:216`); `MoESwiGLUFFNFP8` takes `topk_weights.sum(-1)` in
  // F32 and `clamp_min(1e-9)` (`:314`). Captured so the gate can assert WHICH
  // one this arm took and print the margin between them.
  std::vector<float> denominator;  // [L] f32
  // True when this block ran `MoESwiGLUFFNFP8`.
  bool fp8 = false;
};

// Optional intermediate capture, for the unit gate only. Production passes
// nullptr and pays nothing.
struct Dots3NoteVisionCapture {
  std::vector<float> rope_cache;      // [L, head_dim]
  std::vector<float> patch_embed_out; // [L, E]
  std::vector<float> block0_out;      // [L, E]
  std::vector<float> trunk_out;       // [L, E], after post_trunk_norm
  // One entry per ROUTED block, in block order. Empty on an all-dense tower.
  std::vector<Dots3NoteVisionMoeRoute> moe_routes;
};

// THE TOWER. `pixel_values_bf16` is [L, patch_row()] raw bf16 bits as the
// processor ships them; `grid_thw` is {t, h, w}. Returns
// [L / merge^2, adapter_out_dim] host f32 — the rows the placeholder span is
// expanded to (`multimodal.py:151-155` @ `9035151d6`:
// `grid.prod(-1) // merge_size**2`).
std::vector<float> Dots3NoteVisionForward(
    const std::vector<uint16_t>& pixel_values_bf16,
    const std::array<int64_t, 3>& grid_thw, const Dots3NoteVisionWeights& w,
    const Dots3NoteVisionParams& v, vt::Backend& backend,
    Dots3NoteVisionCapture* capture = nullptr);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_DOTS3_NOTE_VISION_H_
