// LTX-2.5 DiT — the DEVICE-RESIDENT forward, and its kernel seam (phase L8).
//
// Row: MODEL-DIFFUSION-LTX25. Spec: .agents/specs/ltx-2-5.md phase L8. Issue #435.
//
// ─── WHAT THIS CLOSES ────────────────────────────────────────────────────────
//
// Phase L7 wired LTX-2.5 through `vllm::multimodal::VideoEngine` and then had to
// REFUSE `device = 1` by name, because the two halves of the device story did not
// meet: `Ltx2DitForward` (ltx2.h:433-434) accepts only `vt::DType::kF32` and
// computes into host `std::vector<float>`, while `Ltx2StreamDitToDevice`
// (ltx2_loader.h:266-272) stages bf16 onto the device and refuses `widen_to_f32`
// by design. This header is the half that was missing: the SAME graph with every
// activation in device memory and the stream in the checkpoint's own dtype.
//
// ─── DTYPE POLARITY, STATED FIRST ────────────────────────────────────────────
//
// Upstream resolves ONE model dtype and every layer inherits it — `model.py` has
// no per-layer dtype at all. So the device stream is bf16, which is what
// `Ltx2StreamDitToDevice` already puts on the device and what the shipped
// checkpoints dequantize to. `kF32` is accepted too, and ONLY because it is the
// PARITY dtype of the L2 gate: the f32 arm is what lets this forward be compared
// against `tests/vllm/models/ltx2_goldens.inc` — the same goldens the CPU forward
// is held to — at f32 round-off rather than at a bf16 band. It is a gate arm, not
// a production one, and nothing widens a bf16 load to reach it.
//
// The `scale_shift_table` family stays F32 on BOTH arms. That is not a widening:
// the CHECKPOINT stores those tensors F32 (ltx2_loader.h:64-66), and narrowing a
// tensor the file itself widened would be the dtype rule applied backwards. They
// are [2, dim] / [9, dim] / [5, dim] tables — a few kilobytes against a 21 GB
// model — so the f32 read costs nothing per token.
//
// The RoPE cos/sin tables are also f32 on both arms, for the reason
// `Ltx2FreqGrid` already documents (ltx2.h:253-260): the two frequency ladders
// differ only in the last f32 ulps of every angle, and the audio ladder
// multiplies those up by four orders of magnitude before RoPE takes their cosine.
// They are [batch, heads, tokens, head_dim/2] and are built ONCE per forward on
// the host, exactly as MiniMax-H3 builds its own rope cache
// (minimax_h3_device.cpp:193-210).
//
// ─── THE KERNEL SEAM IS NEXT DOOR ────────────────────────────────────────────
//
// `vt::OpId::kLtx2` and the seven-entry table behind it live in
// `ltx2_kernels.h`, which includes nothing but `vt/`. That split is not tidiness:
// `src/vt/cuda/cuda_ltx2.cu` includes the table, and this header pulls in
// `ltx2.h` and with it `nlohmann/json.hpp`, which nvcc has no business parsing.
// The table's contents, and why it is only seven ops, are documented there.
//
// ─── NUMERICS: CLOSE, NOT BIT-IDENTICAL ──────────────────────────────────────
//
// The f32 device arm is NOT bit-identical to `Ltx2DitForward` and does not claim
// to be. The divergence is in the SHARED ops, not in these kernels: `vt::RmsNorm`
// and `vt::LayerNorm` reduce in f32 where `RmsNormRows` / `LayerNormRows`
// accumulate in double, and MatmulBT and the attention ops use their own
// accumulation orders. f32 is what upstream torch does, so the device path is
// arguably the closer mirror. It is held to the SAME upstream goldens.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2.h"
#include "vllm/model_executor/models/ltx2_kernels.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

// A DiT whose every weight lives in device memory, plus the allocations that own
// it. `weights` views into `storage`; dropping this struct frees the model.
//
// This is the TEST/second path onto the device. The production path for a real
// checkpoint is `Ltx2StreamDitToDevice` (ltx2_loader.h:266-272), which never
// materializes the whole model on the host at all; this one starts from an
// already-host-resident weight set, which is exactly what the golden fixtures
// build. Both produce `Ltx2DitWeights` whose views are device-resident, and
// `Ltx2DitForwardDevice` cannot tell them apart.
struct Ltx2DitDeviceWeights {
  std::vector<std::shared_ptr<void>> storage;
  std::map<std::string, vt::Tensor> views;
  Ltx2DitWeights weights;
};

// Upload every tensor of `host` (keyed by its upstream parameter name, exactly
// what `Ltx2LoadDitFromSafetensors` and the golden fixtures produce) to `queue`'s
// device, converting to `stream_dtype` where the dtype policy says so.
//
// THE POLICY, and it is the whole reason this takes a dtype at all:
//   * `stream_dtype == kBF16` — every projection weight and bias becomes bf16,
//     which is what upstream's one-model-dtype resolution gives and what
//     `Ltx2StreamDitToDevice` already puts on the device.
//   * `stream_dtype == kF32` — everything stays f32. The PARITY arm; see this
//     header's DTYPE note for why it exists and what it is not.
// In BOTH cases the `scale_shift_table` family stays F32, because the checkpoint
// stores it F32. `Ltx2DitTensorIsTable` is the predicate, exposed so a caller can
// assert the split rather than re-derive it.
bool Ltx2DitTensorIsTable(const std::string& name);

Ltx2DitDeviceWeights Ltx2StageDitWeightsToDevice(vt::Queue& queue, const Ltx2DitParams& params,
                                                 const std::map<std::string, vt::Tensor>& host,
                                                 vt::DType stream_dtype = vt::DType::kBF16);

// ---------------------------------------------------------------------------
// The forward
// ---------------------------------------------------------------------------

// `Ltx2DitForward` (ltx2.h:443-446) with every activation in device memory.
//
// `weights` must be DEVICE-RESIDENT on `queue`'s device — from
// `Ltx2StreamDitToDevice` or `Ltx2StageDitWeightsToDevice`. Handing it host
// views on a CUDA queue is the failure mode that reads as all-zeros rather than
// as an error, so the forward checks the device of every weight it binds.
//
// Inputs stay HOST pointers and outputs stay host `std::vector<float>`, exactly
// as `MiniMaxH3DitForwardDevice` does (minimax_h3.h:1794-1798): the forward
// uploads what it needs and downloads the two output heads. That is what makes
// the device arm comparable against the SAME goldens as the CPU arm rather than
// against a second, differently-shaped harness.
//
// `compute_dtype` is the STREAM dtype and must be kBF16 (production) or kF32
// (parity). It must agree with how the weights were staged: a bf16 stream over
// f32-staged weights compares a different model, not a different dtype policy.
//
// The prompt-K/V cache (ltx2.h:335-350) is NOT ported to this path yet and a
// non-null `cache` is REFUSED BY NAME rather than silently ignored — an ignored
// cache would recompute correctly and quietly lose the optimization, which is
// exactly the kind of divergence that is discovered a phase later.
Ltx2DitOutputs Ltx2DitForwardDevice(vt::Queue& queue, const Ltx2DitParams& params,
                                    const Ltx2DitWeights& weights,
                                    const Ltx2ModalityInput* video,
                                    const Ltx2ModalityInput* audio, vt::DType compute_dtype,
                                    Ltx2PromptKvCache* cache = nullptr);

}  // namespace vllm
