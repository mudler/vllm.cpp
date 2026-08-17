// MiniMax-Music3 — the DEVICE-RESIDENT acoustic forward (#672, spec §11.4).
//
// `DitForward` (minimax_music3_acoustic.cpp) is the portable reference: host
// `std::vector<float>` throughout, one scalar loop per op. It is what every
// Music3 gate was taken on and it is NOT changed by this header — the device arm
// is an additional entry point, not a rewrite of the existing one.
//
// WHY THIS EXISTS. The 2.4B fp32 DiT is ~20x the whole autoregressive half: a
// 45 s clip at the default 30 inference steps runs `DitForward` 660 times (30
// steps x 2 CFG branches x 11 windows) at ~0.96 TFLOP each, and on the scalar
// host loops that is measured in hours. `d9441ef3` put the 8.6B language model
// on the device and reached 0.946x precisely because this half did not move.
//
// ─── WHAT IS PORTED, AND ONTO WHICH SHARED OP ────────────────────────────────
// Every step below names the reference helper it replaces. NO new kernel is
// added: each one is a shared `vt::` op that already carries a CUDA provider.
//
//   Linear                  -> vt::MatmulBT (+ vt::Add for the rank-1 bias)
//   LayerNorm               -> vt::LayerNorm
//   ApplyPartialRotary      -> vt::RopeFromCache over a [seq, rotary_dim] cache
//   Attention (NON-causal)  -> vt::AttentionCross (bias = nullptr)
//   `x * silu(gate)`        -> vt::SiluAndMul, over a STAGE-TIME half swap
//   residual adds           -> vt::Add
//   PointwiseConv (1x1)     -> vt::MatmulBT on the transposed activation
//
// ─── fp32 STAYS fp32 ─────────────────────────────────────────────────────────
// Spec §2.1: the acoustic half is float32 because upstream chose float32 for it,
// and this arm mirrors that. Every staged weight and every device activation
// below is `vt::DType::kF32`. Narrowing to bf16 would be a different change with
// its own evidence, not a free speedup taken in passing.
//
// ─── NUMERICS: CLOSE, NOT BIT-IDENTICAL, AND SAID SO ─────────────────────────
// This arm does NOT reproduce the host reference bit for bit, and does not claim
// to. Three named differences, none of them a shape or an order-of-operations
// defect:
//
//   1. The reference accumulates every reduction in `double` and stores float32
//      (see the dtype note at the top of minimax_music3_acoustic.cpp). The
//      shared ops accumulate in float32 — which is what torch itself does, so on
//      this axis the device arm is the CLOSER mirror of upstream, not the looser
//      one.
//   2. A `nn.Linear` bias enters the reference INSIDE the accumulator
//      (`double acc = bias`), and here it is a separate `vt::Add` afterwards.
//   3. `vt::AttentionCross`'s CUDA kernel uses the online-softmax recurrence
//      where the reference uses an explicit three-pass max/exp/normalize.
//
// It is gated against the SAME upstream goldens at the SAME tolerance as the
// host forward (tests/vllm/models/test_minimax_music3_acoustic.cpp), and no
// tolerance was widened to admit it.
#pragma once

#include <memory>
#include <vector>

#include "vllm/model_executor/models/minimax_music3_acoustic.h"
#include "vt/device.h"
#include "vt/ops.h"

namespace vllm {
namespace models {
namespace music3 {

// One transformer block's weights, resident on the queue's device.
//
// `ff_in_weight` / `ff_in_bias` are the ONLY tensors whose CONTENT differs from
// the host struct, and the reason is mechanical rather than a choice.
// `transformer_minimax_music3.py:142-143` computes `ff_out(gate_states *
// silu(gate))` where `gate_states, gate = ff_in(x).chunk(2, -1)` — the FIRST
// half is the value and the SECOND is what SiLU runs on. `vt::SiluAndMul`
// computes `silu(x[:, :D]) * x[:, D:]`, i.e. the opposite assignment. Swapping
// the two ROW BLOCKS of the projection once, at stage time, makes the shared op
// compute exactly upstream's expression with no per-step permutation and no
// bespoke kernel. The multiply is commutative, so this is an identity, not an
// approximation.
struct Music3DitDeviceLayer {
  vt::Tensor norm1_weight;   // [inner]
  vt::Tensor norm1_bias;     // [inner]
  vt::Tensor to_q;           // [attn_inner, inner]
  vt::Tensor to_k;           // [attn_inner, inner]
  vt::Tensor to_v;           // [attn_inner, inner]
  vt::Tensor to_out;         // [inner, attn_inner]
  vt::Tensor norm2_weight;   // [inner]
  vt::Tensor norm2_bias;     // [inner]
  vt::Tensor ff_in_weight;   // [2 * ff, inner], HALVES SWAPPED (see above)
  vt::Tensor ff_in_bias;     // [2 * ff],        HALVES SWAPPED
  vt::Tensor ff_out_weight;  // [inner, ff]
  vt::Tensor ff_out_bias;    // [inner]
};

// The DiT staged ONCE onto a device, with the storage that owns it.
//
// STAGED ONCE IS THE WHOLE POINT. 660 forwards per 45 s clip means a per-step
// upload of 9.7 GB would cost more than the compute it enables; the fixed
// +34.8 s `d9441ef3` measured for the language model is what a one-time upload
// looks like, and this arm keeps that shape. `storage` holds one
// `vt::Backend::Alloc` block per tensor, freed with this object.
struct Music3DitDeviceWeights {
  vt::Tensor preprocess_conv_weight;  // [concat, concat] (the 1x1 kernel axis is dropped)
  vt::Tensor proj_in_weight;          // [inner, concat]
  std::vector<Music3DitDeviceLayer> layers;
  vt::Tensor proj_out_weight;          // [in_channels, inner]
  vt::Tensor postprocess_conv_weight;  // [in_channels, in_channels]

  // The timestep embedder stays on the HOST, deliberately and cheaply.
  // `time_proj` + `time_embed` is ONE row through a [inner, fourier_dim] and an
  // [inner, inner] projection — 4.7M MACs against the 2.4G MACs PER TOKEN the
  // block stack runs, i.e. under a millionth of the forward at any real window
  // length. Running it through the existing `DitTimestepEmbedding` keeps that
  // piece BIT-IDENTICAL to the CPU arm for free; moving it would have needed an
  // ungated SiLU op that `vt` does not carry. Only these five tensors are
  // populated in this struct.
  DitWeights host_time_embed;

  std::vector<std::shared_ptr<void>> storage;
};

// Upload the DiT to `queue`'s device, once.
//
// `release_host` EMPTIES each source vector as it is uploaded. The shipped DiT
// is 9.7 GB of fp32 and Jetson Thor's ~122 GB is UNIFIED — host and device draw
// on one pool — so holding both copies is a real 19.4 GB peak on the box this
// arm was written for, which `vm.overcommit_memory=1` and zero swap turn into a
// REBOOT rather than an OOM kill (.agents/environment.md). Pass true from a
// serving path, false from a gate that compares the two arms.
//
// Throws (naming the tensor) if a weight is mis-sized for `config`, or if the
// device has no provider for one of the ops the forward needs — a refusal at
// stage time rather than 36 layers into the first step.
Music3DitDeviceWeights StageMusic3DitWeights(vt::Queue& queue,
                                             const MiniMaxMusic3TransformerConfig& config,
                                             DitWeights& weights, bool release_host);

// `DitForward`'s device twin: same inputs, same outputs, same layouts.
//
//   `latents`   [in_channels, length], CHANNEL-major (host)
//   `condition` [length, condition_dim], FRAME-major (host)
//   returns     [in_channels, length], the flow-matching VELOCITY (host)
//
// The host<->device boundary is the ARGUMENTS ONLY: one upload of
// [length, concat_channels] on the way in and one download of
// [length, in_channels] on the way out, per call. Everything between — all 36
// blocks, both 1x1 convolutions, both projections — stays in device memory.
std::vector<float> DitForwardDevice(vt::Queue& queue, const std::vector<float>& latents,
                                    int64_t length, const std::vector<float>& condition,
                                    double timestep,
                                    const MiniMaxMusic3TransformerConfig& config,
                                    const Music3DitDeviceWeights& weights);

}  // namespace music3
}  // namespace models
}  // namespace vllm
