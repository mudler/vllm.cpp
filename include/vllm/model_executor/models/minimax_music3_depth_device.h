// MiniMax-Music3 — the DEVICE-RESIDENT RVQ depth decoder (#672, #1309, spec §19).
//
// `DepthDecoderAppend` (minimax_music3_ar.cpp) is the portable reference: host
// `std::vector<float>` throughout, one sequential `double` accumulator per
// output element. It is what every Music3 gate was taken on, it is NOT changed
// by this header, and it stays the CPU arm. This is an ADDITIONAL entry point.
//
// ─── WHY THIS EXISTS ─────────────────────────────────────────────────────────
//
// Spec §19.1, measured on `thor:gpu0` at `678fc672c`: `ar.depth_forward` is
// 78.1 s of a 161 s run — 48.4 %, the largest single term. A 0.646B decoder
// therefore costs 6.3x the 8.6B language model beside it, and the reason is
// which processor each runs on. §16.4 measured the host kernel at ~4 bytes of
// f32 weight traffic per multiply-accumulate — 2.28 GB per call, 8 calls a
// frame — and §16.6b measured 96.93 ms per call, i.e. ~23.5 GB/s across Thor's
// 14 cores. This arm moves that sweep to the device and halves it.
//
// ─── WHAT IS PORTED, AND ONTO WHICH SHARED OP ────────────────────────────────
// NO new kernel. Every op below already carries a CPU AND a CUDA provider.
//
//   RmsNorm              -> vt::RmsNorm
//   LinearNoBias         -> vt::MatmulBT
//   CausalAttentionStep  -> vt::AttentionCross (bias = nullptr)
//   silu(gate) * up      -> vt::SiluAndMul
//   residual adds        -> vt::Add
//
// THE ATTENTION IS CAUSAL UPSTREAM AND `vt::AttentionCross` IS STILL THE RIGHT
// OP, by an identity rather than a shortcut. This call presents ONE query row
// against a cache of `seq` positions that are all at or before it, so a causal
// mask masks nothing. `vt::Attention` would apply a mask keyed on a query index
// this call does not have. Upstream's `dispatch_attention_fn(..., is_causal=True)`
// (minimax_music3_rvq_depth_decoder.py:43-50) runs whole sequences; §16.3's
// identity is why ours runs one row, and that identity is gated bitwise.
//
// ─── bf16 STORAGE, f32 ACCUMULATION — THE DECISION §14.5 LEFT OWED ───────────
//
// Spec §19.2. The oracle settles this by declaring NOTHING:
// `MiniMaxMusic3RVQDepthDecoder` takes no `dtype` parameter and contains no
// `torch.float32` literal and no `.float()` call
// (minimax_music3_rvq_depth_decoder.py:101-125); its single cast is the
// DOWN-cast `:51` `hidden_states.flatten(2, 3).to(query.dtype)`. The dtype is
// imposed by `load_components(dtype=...)`, and tools/oracle/music3_oracle.py
// resolves `rvq_depth_decoder: torch.bfloat16` under BOTH its policies. That is
// the §2.1 invariant `dtype(language_model) == dtype(rvq_depth_decoder) ==
// dtype(condition_encoder)`.
//
// `vt::MatmulBT` has carried exactly that contract all along — "a/b bf16 (or
// f32), out f32 or bf16, f32 accumulation" (vt/ops.h) — so §14.5's objection,
// that an *f32* `vt::MatmulBT` would drop the bf16 rounding every gated number
// was taken with, is answered by using the *bf16* one. The CUDA provider refuses
// a mixed f32-activation x bf16-weight combo by name, so this arm is bf16 on
// both operands or it is not bf16 at all.
//
// THE NARROWING IS LOSSLESS, WHICH IS WHY IT IS SAFE TO CALL A MIRROR RATHER
// THAN A CHANGE. `AtRuntimeDtype` (minimax_music3_llm.cpp) already rounds every
// AR-half tensor through bf16 into an f32 carrier, and `Store(..., kBFloat16)`
// does the same to every activation the host arm hands across this boundary. So
// every value staged or uploaded here is ALREADY exactly bf16-representable and
// `vt::F32ToBF16` on it is exact. The host arm's f32 CONTAINERS are what move
// twice the oracle's bytes (spec §19.2a); this arm's do not.
//
// ─── NUMERICS: NOT BIT-IDENTICAL, AND SAID BEFORE THE CODE ───────────────────
//
// Spec §19.4. §16 could claim bitwise identity because a causal identity is
// exact; this row changes the machine, and it does NOT claim it. Three
// independent reasons, each sufficient alone:
//
//   1. THE ACCUMULATOR NARROWS. The host keeps a sequential `double` per output
//      element; `vt::MatmulBT` accumulates in f32 (CUBLAS_COMPUTE_32F). Over
//      in_dim 4096 and 6144 that is real. Note the DIRECTION: torch's own bf16
//      matmul accumulates in f32, so on this axis the device arm is the CLOSER
//      mirror of upstream, not the looser one.
//   2. THE REDUCTION RE-ASSOCIATES. cuBLASLt splits K by an algorithm this file
//      does not choose, and `vt::AttentionCross`'s CUDA kernel uses an online
//      softmax recurrence where the host and the CPU provider use an explicit
//      three-pass max/exp/normalize. Those two are not bit-identical to each
//      other either.
//   3. THE TWO ARMS NORMALIZE AGAINST DIFFERENT REFERENCES, AND BOTH ARE RIGHT.
//      The Music3 HOST arm's own RmsNorm (minimax_music3_ar.cpp) mirrors the
//      diffusers module this model IS: `normalization.py::RMSNorm.forward` casts
//      back to the weight dtype at `:560` before the affine multiply at `:561`,
//      so it rounds TWICE, and the decoder constructs exactly that class
//      (minimax_music3_rvq_depth_decoder.py:78,80,122). The device arm routes
//      through the SHARED `vt::RmsNorm`, which mirrors vLLM — and vLLM keeps f32
//      across the weight multiply and rounds ONCE. Verified at the parity pin
//      `555967922`: `csrc/cpu/layernorm.cpp` computes
//      `fp32_out = fp32_x * fp32_s_variance * fp32_w`, and
//      `csrc/libtorch_stable/layernorm_kernels.cu:93` computes
//      `static_cast<scalar_t>(x * s_variance * w)`. Upstream tried the
//      weight-dtype multiply (vllm#42379) and REVERTED it (vllm#46070, an
//      ancestor of the pin).
//
//      So this is NOT the shared op diverging from its reference, and an earlier
//      revision of this comment said it was. AGENTS.md makes vLLM the only
//      reference wherever it implements the behaviour, and it implements
//      RMSNorm, so `vt::RmsNorm` is correct against it. What differs is which
//      reference each arm answers to. WHICH ONE IS RIGHT FOR THIS MODEL is a
//      question the diffusers oracle settles through
//      `test_minimax_music3_ar_real`, and §19.6 records that as owed.
//
// The gate is therefore a tolerance in bf16 ULPs of the reference value, and it
// asserts its own teeth. A tolerance cannot see a DROPPED STAGE, so it is not
// the only gate: the composed schedule's drawn codes are compared too.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_ar.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, for the merged gate_up
#include "vt/device.h"
#include "vt/ops.h"

namespace vllm {
namespace models {
namespace music3 {

// One decoder layer's weights, resident on the queue's device, bf16.
//
// TWO MERGES, because AGENTS.md routes mergeable projections through the merged
// seam and because they cost nothing numerically — each output element remains
// its own dot product, so no reduction moves:
//
//   * `qkv` is `to_q | to_k | to_v` stacked as one [3H, H]. One `vt::MatmulBT`
//     sweeps the weight once for all three where the host arm sweeps three
//     times, and the three results are read back by free re-view.
//   * `gate_up` is `gate_proj | up_proj` stacked as one [2I, H], and it is
//     consumed through `layers::UnquantizedMlpGateUpMethod` — the shared
//     merged-GEMM seam, not a hand-rolled equivalent. The merge is REQUIRED
//     rather than merely useful: `vt::SiluAndMul` consumes ONE [T, 2D] buffer as
//     `silu(x[:, :D]) * x[:, D:]`, and upstream's
//     `silu(gate_proj(x)) * up_proj(x)` puts gate first. So NO half swap —
//     unlike §14.3's DiT, where the assignment was the other way round and
//     needed a stage-time exchange.
struct Music3DepthDeviceLayer {
  vt::Tensor input_layernorm;           // [H]
  vt::Tensor post_attention_layernorm;  // [H]
  vt::Tensor qkv;                       // [3H, H]  MERGED
  vt::Tensor to_out;                    // [H, H]
  // [2I, H] MERGED, and an `OwnedTensor` rather than a staged `vt::Tensor`
  // because it is consumed through `layers::UnquantizedMlpGateUpMethod`, which
  // is the shared merged-GEMM seam AGENTS.md routes mergeable MLP projections
  // through. That method IS this arm's MLP byte for byte — one `vt::MatmulBT`
  // over the merged [2I, H] operand then `vt::SiluAndMul` — so routing through
  // it costs nothing and puts the fused-kernel scheme choice (a quantized arm
  // may fuse the pair into one Marlin GEMM) in its one home. `ResidentWeight`
  // owns the residency: it ALIASES on a CPU queue and uploads once on a device.
  OwnedTensor gate_up;
  vt::Tensor down_proj;                 // [H, I]
};

// The depth decoder staged ONCE onto a device, with the storage that owns it.
//
// STAGED ONCE IS THE POINT, as it is for the DiT: a 4 s clip runs this decoder
// 808 times, so a per-call upload of the weights would cost more than the
// compute it enables.
//
// `projection`, `audio_embeddings` and `audio_heads` are NOT here. They stay on
// the host in this row, deliberately: §15's profile puts `ar.depth_projection`
// at 1.307 s against the forward's 78.316 s, so they are ~1.6 % of the stage.
// Spec §19.7 carries them as owed rather than leaving them to be discovered.
struct Music3DepthDeviceWeights {
  vt::Tensor pos_embedding;  // [max_position_embeddings, H]
  vt::Tensor norm;           // [H]
  std::vector<Music3DepthDeviceLayer> layers;

  std::vector<std::shared_ptr<void>> storage;

  bool staged() const { return !layers.empty(); }
};

// Upload the depth decoder to `queue`'s device, once, narrowing to bf16.
//
// `release_host` empties each source vector as it is uploaded, for the same
// unified-memory reason `StageMusic3DitWeights` has it: on Jetson Thor host and
// device draw on one pool, so holding both copies is a real peak. Pass true from
// a serving path and false from a gate that compares the two arms.
//
// Throws, naming the tensor, if a weight is mis-sized for `config`; and throws
// naming the op and the device if the device has no provider for one of the ops
// the forward needs — a refusal at stage time rather than four layers into the
// first of 808 calls.
Music3DepthDeviceWeights StageMusic3DepthWeights(vt::Queue& queue,
                                                 const DepthDecoderConfig& config,
                                                 DepthDecoderWeights& weights,
                                                 bool release_host);

// The device twin of `DepthDecoderCache`: the K and V of every position already
// fed, for `batch` INDEPENDENT rows sharing one weight set.
//
// ONE CONTIGUOUS BLOCK PER (layer, row) of [max_position_embeddings, H], sized
// once at first use. `vt::AttentionCross` requires contiguous operands, so a
// row's history must be contiguous ACROSS positions — which fixes this layout
// and is why the batched K/V projection is copied into the cache rather than
// written there directly by the GEMM. The copy is H bf16 values per row per
// tensor (8 KB at the shipped geometry), against a 2.28 GB weight sweep.
struct Music3DepthDeviceCache {
  int64_t batch = 0;
  int64_t hidden = 0;
  int64_t layers = 0;
  int64_t positions = 0;
  int64_t capacity = 0;
  // [layer * batch + row] -> [capacity, hidden] bf16, device-resident.
  std::vector<vt::Tensor> keys;
  std::vector<vt::Tensor> values;
  std::vector<std::shared_ptr<void>> storage;
};

// How many times `DepthDecoderAppendDevice` has run in this process.
//
// #1131 IS WHY THIS EXISTS. The DiT device arm's kernels and staging are gated
// and its production SWITCH is not: setting `on_device = false` leaves every
// suite green, because the two arms agree numerically BY DESIGN and no gate ever
// asked which one ran. `.agents/verification.md` requires the gate to assert the
// device path was TAKEN — "invocation count or resident dtype", not an inference
// from the numbers. This counter is that assertion's instrument, and it is the
// reason a reachability mutation on the production selection goes RED here.
uint64_t Music3DepthDeviceForwardCount();

// #1131's OTHER instrument, and this row took only the first until a review said
// so. The words are #1131's own: a gate "must assert the device path was TAKEN
// (invocation count OR RESIDENT DTYPE), not merely that outputs agree".
//
// A bit per `vt::DType`, OR-ed over every buffer `DepthDecoderAppendDevice`
// makes resident — the activations, the merged QKV block, the MLP result, the
// downloaded output and the K/V cache — accumulated across every call in this
// process. It reads back what the forward ACTUALLY allocated rather than what
// this header says it should, which is the whole point: AGENTS.md's "a token
// gate cannot detect a dtype that is too WIDE" applies with full force to the
// tolerance gate above, and a review proved it — widening one activation buffer
// to `kF32` left the ULP band, the drawn codes and all 35 cases green while the
// path moved twice the bytes. That is §19.2a's own finding about the HOST arm,
// arriving inside the arm that exists to fix it.
//
// The mask is monotone and never reset, so a single widened buffer anywhere in
// the process is visible for the rest of it.
uint64_t Music3DepthDeviceResidentDtypes();

// ONE depth position, appended to `cache`, for `batch` rows at once — the device
// twin of `DepthDecoderAppend`, with the same inputs, the same outputs and the
// same refusals.
//
// `inputs_embeds` is [batch, hidden_size] on the HOST carrying bf16-exact
// values; the return is [batch, hidden_size] on the HOST, the post-`norm` hidden
// state of the appended position for each row — the only row the generation
// schedule reads (encoders.py:131-132).
//
// The host<->device boundary is the ARGUMENTS ONLY: one upload of [batch, H] in
// and one download of [batch, H] out per call. The weights and the K/V history
// never cross it.
std::vector<float> DepthDecoderAppendDevice(vt::Queue& queue, const DepthDecoderConfig& config,
                                            const Music3DepthDeviceWeights& weights,
                                            const std::vector<float>& inputs_embeds,
                                            int64_t batch, Music3DepthDeviceCache* cache);

}  // namespace music3
}  // namespace models
}  // namespace vllm
