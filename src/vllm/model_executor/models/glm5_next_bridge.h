// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5b-1: the `OwnedTensor` -> host
// f32 bridge, and the RESIDENCY DECISION it implements.
//
// Issue [#2241](https://github.com/mudler/vllm.cpp/issues/2241), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5b and `## Owed` O22.
//
// Model-private, deliberately not under `include/`, for the reason
// `glm5_next_loader.h` gives: this wave ships no capability above the block
// layer, and `include/vllm.h` is the seam a SHIPPED capability is exposed
// through.
//
// ─── THE QUESTION O22 LEFT OPEN, AND THE ANSWER THIS FILE TAKES ──────────────
//
// W5c landed the weight tower. Its 1383 backbone tensors are `OwnedTensor`s,
// and 774 of them KEEP THEIR GGML BLOCKS. W2/W3/W4/W5's host references consume
// `std::vector<float>`. O22 states the gap and refuses to close it by fiat:
// "Whoever writes the forward decides whether to decode per layer or to go
// device-native; nothing here forecloses either."
//
// **THE DECISION: decode ONE LAYER AT A TIME, ON DEMAND, AND NEVER RETAIN THE
// TOWER IN FLOAT.** The tower stays block-resident exactly as loaded; this
// bridge produces a bounded, caller-owned f32 mirror of ONE DSA layer's
// attention weights and nothing else.
//
// **The arithmetic that forces it**, all of it measured on this row and
// recorded in the spec's `### The measured residency`:
//
//   | what | GiB |
//   |---|---:|
//   | the published `UD-Q2_K_XL` artifact, block-resident as loaded | **101.14** |
//   | the same tower with every tensor expanded (`### The measured residency`) | **426.72** |
//   | all-bf16 | 597.46 |
//   | usable on `dgx:gpu0`, the largest device this project reaches | **~119.63** |
//   | ONE bridged DSA layer, f32 (`BridgedDsaLayerF32Bytes`, published dims) | **0.4654** |
//   | all ELEVEN DSA layers held at once | 5.12 |
//
// A bridge that decoded the tower would cost 426.72 GiB against a 119.63 GiB
// box — 3.57x over, and that is the *expanded* figure this campaign spent
// #2245 and #2247 removing. A float tower is not "expensive"; it does not
// exist on any hardware this project can reach. Per layer it is 0.4654 GiB,
// 0.39% of the box, and the caller's peak is one layer because the mirror is a
// value it can drop.
//
// **What this rules out, said positively.** There is no `BridgeTower`, no
// cache, and no lazily-populated map keyed by layer index — each of those turns
// "one layer" into "every layer visited so far", which is the tower again with
// a slower ramp. A caller that wants the whole model resident in float has to
// write that loop itself, and the ceiling below will refuse it one tensor
// before it gets there.
//
// ─── O19 / #2260: THIS BRIDGE CANNOT MAKE THE MOE THROW REACHABLE ────────────
//
// O19 records that the moment this row routes the experts through
// `layers::MlpGateUpMethodBase` / `vt::MergedGemmGroup` on CUDA,
// `MoeGateUpSwiGLUGroupedCuda` throws: neither IQ2_XS nor IQ4_XS is in
// `IsCudaKeepQuantSupported`, and 85 of this artifact's tensors are those two
// types. W5's MoE deliberately reaches only `vt::MoeRouterTopK` /
// `vt::MoeCombine` with host GEMM loops for that reason.
//
// This file cannot make that throw reachable, and it is gated rather than
// argued:
//
//   * **Structurally.** There is no overload taking `Glm5NextMoeWeights`,
//     `Glm5NextMlpWeights` or any expert bank. The bridge's whole surface is
//     `Glm5NextMlaWeights` and `Glm5NextIndexerWeights` — the DSA attention
//     tower, which carries no IQ2_XS or IQ4_XS tensor at all.
//   * **Numerically.** `kBridgeTensorF32ByteCeiling` is 1 GiB. The LARGEST
//     tensor the bridge legitimately touches is `o_proj` at
//     4096 x 16384 x 4 B = 0.25 GiB, 4x under. The SMALLEST expert bank is
//     `up_exps` at 288 x 2048 x 4096 x 4 B = 9.0 GiB, 9x over. The ceiling sits
//     between them by a factor of four in both directions and refuses BY NAME,
//     so an expert bank handed to `DecodeOwnedTensorToF32` is a named error and
//     not a 9 GiB allocation.
//
// A ceiling no legitimate input can reach would be a mute switch
// (`.agents/verification.md`); this one is placed where the two populations
// actually separate, and the test pins both sides of the gap.
//
// ─── WHY NOT DEVICE-NATIVE, SINCE O22 ALLOWED IT ────────────────────────────
//
// Because there is nothing to be device-native AGAINST yet. Every glm5_next
// primitive on this row — `glm5_next_kda.cpp`, `glm5_next_dsa.cpp`,
// `glm5_next_mhc.cpp`, `glm5_next_moe.cpp` and `glm5_next_attn.cpp` — is a host
// f32 reference; W3's CUDA arm is committed and UNMEASURED for want of a
// `dgx:gpu0` lease. A device bridge would have to land beside a device forward
// that does not exist, and it would be the "unpassed parameter" shape. The
// choice is recorded as a decision with a reason rather than a preference, and
// the spec's `## Owed` names the wave that revisits it.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_BRIDGE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_BRIDGE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/glm5_next_attn.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vllm/model_executor/models/glm5_next_loader.h"

namespace vllm::glm5_next {

// One gibibyte. See the O19 section above for why the value is here and not
// higher or lower: 4x above the largest legitimate tensor and 9x below the
// smallest expert bank.
inline constexpr int64_t kBridgeTensorF32ByteCeiling = int64_t{1} << 30;

// The f32 host mirror of `t`, in BYTES, computed from the shape ALONE — no
// decode, no allocation. This is the number a caller budgets with, and it is
// what the ceiling is checked against BEFORE anything is allocated.
int64_t HostF32Bytes(const OwnedTensor& t);

// The f32 host mirror of one bridged DSA layer, in BYTES, computed from the
// DIMS alone. 499,657,728 (0.4654 GiB) at the published checkpoint's geometry.
// Exists so the residency claim in the header is a value a gate can read rather
// than a sentence a reader has to trust.
int64_t BridgedDsaLayerF32Bytes(const MlaDims& d, const IndexerDims& id);

// Decode ONE `OwnedTensor` into host f32.
//
// Handles every residency the loader produces: `kF32` (copied), `kF16`/`kBF16`
// (widened) and the block-quantized encodings (through
// `vt::cpu::BlockToFloat`, the same decoder `RouteGgufTensor` chose to keep the
// blocks for). `what` names the tensor in every refusal, because a bridge that
// says "shape mismatch" without saying WHICH weight is a bridge whose failures
// cost a bisect.
//
// REFUSES BY NAME, rather than serving a wrong or enormous buffer:
//   * a tensor whose host bytes were released (`host_released`) — its `bytes`
//     are empty and an empty result would read as a zero weight;
//   * a tensor whose f32 mirror exceeds `byte_ceiling`;
//   * a block-quantized dtype with no `BlockToFloat` decoder in this build;
//   * a byte span that is not the size its shape and dtype require.
std::vector<float> DecodeOwnedTensorToF32(
    const OwnedTensor& t, const std::string& what,
    int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

// One DSA layer's attention weights, mirrored into host f32.
//
// **The indexer view is a METHOD and not a member, deliberately.**
// `glm5_next_dsa::IndexerWeights` is a struct of `const float*`. A member of
// that type pointing into this object's own vectors would dangle the moment the
// object is moved or copied — silently, into freed-but-plausible memory, which
// is the failure mode a value type must not have. `IndexerView()` builds the
// pointer set from the CURRENT storage on every call.
struct BridgedDsaLayer {
  MlaWeights mla;

  // The indexer's own storage, field for field with
  // `glm5_next_dsa::IndexerWeights`.
  std::vector<float> idx_wq_b;
  std::vector<float> idx_wk;
  std::vector<float> idx_k_norm_weight;
  std::vector<float> idx_k_norm_bias;
  std::vector<float> idx_weights_proj;
  std::vector<float> idx_kpool_ape;
  std::vector<float> idx_kpool_gate;

  // What this mirror actually cost, MEASURED from the decoded buffers rather
  // than predicted. A caller budgeting a stack reads this, and a gate can check
  // it against `BridgedDsaLayerF32Bytes` for the same dims.
  int64_t host_f32_bytes = 0;

  IndexerWeights IndexerView() const;
};

// Bridge ONE DSA layer. `src` is the layer's `Glm5NextMlaWeights` as the loader
// produced it, INCLUDING its nested `Glm5NextIndexerWeights`.
//
// Every tensor's shape is checked against `d` and `id` and refused by name on a
// disagreement, so a geometry that drifted between the config and the file is a
// named error here rather than a wrong number in the attention block.
//
// This is the ONLY entry point. There is no tower-wide form; see the header.
BridgedDsaLayer BridgeDsaLayer(
    const Glm5NextMlaWeights& src, const MlaDims& d, const IndexerDims& id,
    int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_BRIDGE_H_
