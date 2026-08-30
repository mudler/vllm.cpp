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
#include "vllm/model_executor/models/glm5_next_kda.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vllm/model_executor/models/glm5_next_mhc.h"
#include "vllm/model_executor/models/glm5_next_moe.h"

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

// The f32 host mirror of ONE leading-axis row of `t`, in BYTES, from the shape
// alone. `HostF32Bytes(t) / t.shape[0]`, stated as its own function because it
// is the number the per-expert and per-token paths budget with and the two
// populations the 1 GiB ceiling separates are a WHOLE tensor and a ROW of one.
int64_t HostF32RowBytes(const OwnedTensor& t);

// Decode a contiguous RANGE of `t`'s leading-axis rows into host f32.
//
// ─── WHY A ROW RANGE AND NOT THE TENSOR ──────────────────────────────────────
//
// Three tensors of this model cannot be decoded whole on any device this
// project reaches, and all three are consumed a few rows at a time:
//
//   | tensor, published geometry | whole, f32 | one row, f32 |
//   |---|---:|---:|
//   | `token_embd.weight` [154880, 4096] | 2.36 GiB | 16 KiB |
//   | `output.weight` [154880, 4096] | 2.36 GiB | 16 KiB |
//   | `ffn_up_exps.weight` [288, 2048, 4096] | 9.0 GiB | 32 MiB |
//
// A prompt gathers `seq_len` embedding rows of 154,880, a step reads the
// `lm_head` in chunks, and a token selects 8 experts of 288. Whole-tensor
// decoding is not "expensive" for any of them; `DecodeOwnedTensorToF32` refuses
// all three by name at the 1 GiB ceiling, which is that ceiling working.
//
// **The ceiling is UNCHANGED and this function is checked against the same
// one.** It is the RANGE that is checked, so the caller cannot ask for the
// whole bank through the back door: `first_row = 0, num_rows = 288` on
// `ffn_up_exps` is 9.0 GiB and is refused by exactly the same arithmetic that
// refuses the whole tensor.
//
// REFUSES BY NAME, in addition to every refusal `DecodeOwnedTensorToF32` has:
//   * a rank-0 tensor, which has no leading axis to slice;
//   * a row range outside `[0, shape[0])`;
//   * a BLOCK-quantized tensor whose row is not a whole number of blocks — the
//     slice would start mid-block, and a decoder handed a misaligned base
//     returns plausible values from the wrong scales rather than failing.
std::vector<float> DecodeOwnedTensorRowsToF32(
    const OwnedTensor& t, const std::string& what, int64_t first_row,
    int64_t num_rows, int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

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
// This is the ONLY entry point for the DSA arm. There is no tower-wide form;
// see the header.
BridgedDsaLayer BridgeDsaLayer(
    const Glm5NextMlaWeights& src, const MlaDims& d, const IndexerDims& id,
    int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

// ─── THE OTHER FOUR ARMS (W5b-2b) ───────────────────────────────────────────
//
// `BridgeDsaLayer` covers the 11 DSA layers' attention. A forward needs four
// more, and three of them are mechanical: their tensors are the same size class
// as the DSA layer's and each one is a shape-checked decode. The MoE is the one
// that is not, and `BridgeMoeLayer` below is where that shows.

// One KDA layer's 15 tensors (`Glm5NextKdaWeights` -> the host reference's
// `Glm5NextKdaLayerWeights`). Field for field, shape-checked against `d`.
//
// 18.5 GiB in f32 for all 34 KDA layers at the published geometry, 0.545 GiB
// for one — the same size class as a DSA layer's 0.4654 and the same rule
// applies: bridge one, use it, drop it.
glm5_next_kda::Glm5NextKdaLayerWeights BridgeKdaLayer(
    const Glm5NextKdaWeights& src, const glm5_next_kda::Glm5NextKdaDims& d,
    int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

// A gated MLP: the dense layers' feed-forward and every sparse layer's SHARED
// expert. `what` prefixes each refusal, because the two are the same struct at
// different widths and a message that said only `gate_proj` would not say which.
DenseMlpWeights BridgeMlp(const Glm5NextMlpWeights& src, int64_t hidden,
                          int64_t intermediate, const std::string& what,
                          int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

// One `Glm5NextTextHyperConnection`'s three tensors. Two per decoder layer.
HcSite BridgeMhcSite(const Glm5NextMhcWeights& src, const Glm5NextMhcParams& mhc,
                     int64_t hidden, const std::string& what,
                     int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

// The per-expert source `MoeForward` consults, backed by the loader's
// block-resident banks. See `glm5_next_moe.h` for the arithmetic that makes
// this the only shape that fits; this class is the implementation of it.
//
// It decodes THREE rows per expert — one of `gate_exps`, one of `up_exps`, one
// of `down_exps` — fuses the first two into the `[2I, H]` gate-first row the
// seam declares, and holds NOTHING between calls. `decoded()` counts the
// experts it was asked for, in order, so a gate can assert that only the
// SELECTED ones were decoded rather than trusting the loop.
class GgufExpertSource final : public ExpertSource {
 public:
  GgufExpertSource(const Glm5NextMoeWeights& src, const MoeDims& d,
                   const std::string& what,
                   int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

  void Expert(int64_t e, std::vector<float>& gate_up,
              std::vector<float>& down) override;

  // The expert ids this source was asked to decode, in call order. An
  // instrument, and a cheap one: `num_experts_per_tok` is 8 and a step's
  // distinct hits are bounded by `n_routed_experts`.
  const std::vector<int64_t>& decoded() const { return decoded_; }

 private:
  const Glm5NextMoeWeights* src_;
  MoeDims d_;
  std::string what_;
  int64_t byte_ceiling_;
  std::vector<int64_t> decoded_;
};

// One sparse layer's ROUTER, correction bias and shared expert, bridged; the
// three expert BANKS are deliberately left EMPTY and the caller sets
// `expert_source`. Bridging them here is what `kBridgeTensorF32ByteCeiling`
// refuses by name, and rightly: 27.0 GiB per layer against a ~119.63 GiB box.
MoeLayerWeights BridgeMoeLayer(const Glm5NextMoeWeights& src, const MoeDims& d,
                               const std::string& what,
                               int64_t byte_ceiling = kBridgeTensorF32ByteCeiling);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_BRIDGE_H_
