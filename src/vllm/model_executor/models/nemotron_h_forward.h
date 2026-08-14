// Nemotron-H (`NemotronHForCausalLM`) — W4, the FORWARD
// ([spec](../../../../.agents/specs/nemotron-h-model.md) §4 W4, issue #517).
//
// W3 made the architecture KNOWN, PARSED, ENUMERATED and KV-SHAPED
// (nemotron_h.h). This header is where it first COMPUTES: the hybrid 52-layer
// decoder loop over the three block kinds `layers_block_type` names — 23 Mamba2,
// 6 GQA attention at indices {5,12,19,26,33,42}, 23 non-gated relu² MoE.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   NemotronHModel::Forward   <-  nemotron_h.py:604-660 — the single-branch
//                                 pre-norm residual stream. NOTE: unlike Llama,
//                                 a NemotronH decoder layer has exactly ONE norm
//                                 and ONE mixer (:301-313, :344-355, :390-401,
//                                 :525-528), not an attention branch plus an MLP
//                                 branch. 52 layers == 52 (norm, mixer) pairs.
//   Mamba2Mixer               <-  layers/mamba/mamba_mixer2.py:548-586 (in_proj →
//                                 conv+SSM → gated group norm → out_proj), the
//                                 split points at :532-543 and :692-696, the
//                                 prefill scan call at :866-891.
//   Attention                 <-  nemotron_h.py:473-486. Ported deliberately
//                                 WITHOUT any positional embedding — see
//                                 `kNemotronHAttentionHasNoRope`.
//   MoE                       <-  nemotron_h.py:126-256 + the W2 arm recorded in
//                                 spec §6a (up → relu² → down → scaled combine).
//
// ─── SCOPE, exactly ─────────────────────────────────────────────────────────
// This is the HOST (CPU) reference forward, the same cadence Kimi-Linear's W2-W6
// took (kimi_linear_forward.cpp:1-26) and DeepSeek-V4's `DeepseekV4ForwardHost`
// before it: a real forward composed out of the ALREADY-LANDED `vt::` ops, run
// on a CPU queue, so the mechanism is gateable before any GPU is involved. What
// it is NOT:
//   * NOT a weight loader. Nothing here reads a checkpoint. The safetensors
//     materialization of the 18487 tensors `EnumerateNemotronHTensors` names —
//     including the NVFP4 W4A16 g16 experts and the FP8 W8A8 mamba projections —
//     is still owed, and `NemotronHForward` REFUSES BY NAME when handed
//     unmaterialized weights rather than returning zeros.
//   * NOT the MTP head (W5), NOT the GGUF arm (W7), NOT a speed claim.
//   * NOT the paged/device runner path (W6). The recurrent state here is FRESH
//     per call and single-sequence; `MakeNemotronHKVCache` already declares the
//     het-KV topology the runner will drive, and wiring it is W6's.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/nemotron_h.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// NemotronH's attention carries NO positional embedding of any kind. This is not
// an omission to be repaired later: `models/nemotron_h.py` @ 555967922 contains
// ZERO occurrences of `rope`, `rotary` or `Rotary` — `NemotronHAttention.__init__`
// (:415-471) builds `qkv_proj`, `o_proj` and `Attention` and nothing else, and
// `.forward` (:473-486) is qkv → split → attn → o_proj with no rotation step.
// Position information reaches the residual stream through the 23 Mamba2 layers'
// recurrence instead, which is what makes a 1M context cheap here.
//
// The released config.json DOES ship `rope_theta: 10000` and
// `partial_rotary_factor: 1.0`, and `NemotronHParams` parses both (nemotron_h.h)
// — they are INERT for this architecture. Applying them would be numerically
// plausible, would not change a single tensor SHAPE, and on a short prompt might
// not move a token; the forward gate mutates exactly this to prove it is caught.
inline constexpr bool kNemotronHAttentionHasNoRope = true;

// ─── host weights ───────────────────────────────────────────────────────────
//
// One owned, contiguous host tensor in a DECLARED dtype. Deliberately a lean
// byte buffer rather than the safetensors-backed `OwnedTensor`: nothing in W4
// loads a checkpoint, so pulling the loader's includes into this header would
// buy nothing, and the byte-buffer form is what a later loader writes into
// directly.
//
// The DTYPE IS THE POINT, not an implementation detail. vLLM resolves ONE model
// dtype and every layer inherits it (AGENTS.md); this struct carries it
// explicitly so a gate can assert the memory format instead of inferring it from
// matching tokens — a dtype that is too WIDE is numerically correct and
// therefore invisible to a token comparison.
struct NemotronHOwned {
  std::vector<uint8_t> bytes;
  vt::DType dtype = vt::DType::kF32;
  std::vector<int64_t> shape;

  bool Empty() const { return bytes.empty(); }
  int64_t Numel() const;
  // A non-owning contiguous view over the current buffer. Rebuilt on each call
  // so it survives moves/reallocations of the owning struct.
  vt::Tensor View(vt::Device device) const;

  // Pack canonical f32 values into `dtype`. This is the seam a real loader
  // replaces: it hands over checkpoint bytes that are ALREADY in the model
  // dtype, and no packing happens per forward.
  static NemotronHOwned FromF32(const std::vector<float>& values, vt::DType dtype,
                                std::vector<int64_t> shape);
};

// Mamba2 mixer weights, by the names the checkpoint ships (nemotron_h.h's
// enumeration): `mixer.{in_proj,conv1d,A_log,D,dt_bias,norm,out_proj}`.
struct NemotronHMambaWeights {
  // torch Linear orientation [out, in] — vt::MatmulBT's `b [N,K]`.
  NemotronHOwned in_proj;   // [in_proj_out_features, hidden_size]
  NemotronHOwned out_proj;  // [hidden_size, mamba_intermediate_size]
  // The disk tensor is [conv_dim, 1, conv_kernel]; vt::CausalConv1dFwd takes the
  // squeezed [conv_dim, conv_kernel] view, exactly as upstream's
  // `self.conv_weights = self.conv1d.weight.view(conv_dim, K)`
  // (mamba_mixer2.py, `conv_weights`).
  NemotronHOwned conv1d_weight;  // [conv_dim, conv_kernel]
  NemotronHOwned conv1d_bias;    // [conv_dim] — use_conv_bias=true here
  // f32 BY CONTRACT, not by choice: vt::Mamba2ChunkScan validates A/D/dt_bias as
  // f32 (ops.cpp CheckMamba2*), mirroring upstream, which keeps `A` in f32
  // (`self.A = -torch.exp(self.A_log.float())`) regardless of the model dtype.
  NemotronHOwned A_log;    // [mamba_num_heads] f32
  NemotronHOwned D;        // [mamba_num_heads] f32
  NemotronHOwned dt_bias;  // [mamba_num_heads] f32
  // Mixer2RMSNormGated weight over the SSM intermediate width.
  NemotronHOwned norm_weight;  // [mamba_intermediate_size]
};

// GQA attention weights. q/k/v ship SEPARATE on disk (upstream fuses them into
// `qkv_proj` at load through its stacked-params mapping), so they are separate
// here too — that is the shape `EnumerateNemotronHTensors` already claims.
struct NemotronHAttentionWeights {
  NemotronHOwned q_proj;  // [num_attention_heads*head_dim, hidden_size]
  NemotronHOwned k_proj;  // [num_key_value_heads*head_dim, hidden_size]
  NemotronHOwned v_proj;  // [num_key_value_heads*head_dim, hidden_size]
  NemotronHOwned o_proj;  // [hidden_size, num_attention_heads*head_dim]
};

// One NON-GATED expert: `ckpt_names=("up_proj","down_proj","")`
// (nemotron_h.py:220). There is no gate_proj tensor anywhere in the checkpoint.
struct NemotronHExpertWeights {
  NemotronHOwned up_proj;    // [moe_intermediate_size, hidden_size]
  NemotronHOwned down_proj;  // [hidden_size, moe_intermediate_size]
};

struct NemotronHMoeWeights {
  // The router. Its OUTPUT DTYPE IS F32 AND THAT IS MIRRORED, NOT INHERITED:
  // `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)`
  // (nemotron_h.py:150-156). This is the annotated f32 escape AGENTS.md allows,
  // and it is upstream's own polarity rather than a local precision choice.
  NemotronHOwned gate;  // [n_routed_experts, hidden_size]
  // `self.gate.e_score_correction_bias` (nemotron_h.py:158-160), f32 upstream.
  NemotronHOwned e_score_correction_bias;  // [n_routed_experts] f32
  std::vector<NemotronHExpertWeights> experts;
  // The shared expert is the SAME non-gated shape at
  // moe_shared_expert_intermediate_size * n_shared_experts (nemotron_h.py:176-190).
  NemotronHExpertWeights shared;
  bool has_shared = false;
};

// The dense `-` block. No released in-scope NemotronH checkpoint ships one, so
// this arm exists because the schedule can name it and a silent zero would be
// worse than a computed answer; it is exercised only by the unit gate.
struct NemotronHMlpWeights {
  NemotronHOwned up_proj;    // [intermediate_size, hidden_size]
  NemotronHOwned down_proj;  // [hidden_size, intermediate_size]
};

struct NemotronHLayerWeights {
  NemotronHBlock block = NemotronHBlock::kMamba;
  // The layer's SINGLE norm (`self.norm`, one per decoder layer).
  NemotronHOwned norm;  // [hidden_size]
  NemotronHMambaWeights mamba;
  NemotronHAttentionWeights attn;
  NemotronHMoeWeights moe;
  NemotronHMlpWeights mlp;
};

struct NemotronHHostWeights {
  // The ONE model dtype every activation and every weight above inherits
  // (AGENTS.md: "vLLM resolves ONE model dtype and every layer inherits it").
  // bf16 is the released checkpoint's; the gate also sweeps f32, because a bf16
  // store absorbs reduction-order defects.
  vt::DType act_dtype = vt::DType::kBF16;
  NemotronHOwned embeddings;  // [vocab_size, hidden_size]
  std::vector<NemotronHLayerWeights> layers;
  NemotronHOwned norm_f;   // [hidden_size]
  NemotronHOwned lm_head;  // [vocab_size, hidden_size]
  // False until a loader materializes the enumerated tensors. The forward
  // refuses by name on false rather than computing on zeros.
  bool materialized = false;
};

// ─── per-block entry points (each independently gateable) ───────────────────
//
// Every one takes the ALREADY-NORMED hidden `[T, hidden_size]` in `act_dtype`
// and returns the mixer output `[T, hidden_size]` in the same dtype, which is
// exactly the contract of an upstream `layer.mixer(hidden_states)` call. They
// are public so the gate can compare a BLOCK's activations against an
// independent reference — a mechanism can be missing while the argmax is
// unchanged, so a tokens-only comparison cannot see a dropped one
// (porting-a-model.md §3).

// The Mamba2 mixer's two recurrent states, carried ACROSS calls so a sequence
// can be fed in more than one leg (upstream's chunked prefill: leg 2 reads leg
// 1's `final_states` as its `initial_states`, ssd_combined.py:79,:194).
//
// The two dtypes are resolved SEPARATELY and that is the whole point:
//   * `conv` is f32 because `vt::CausalConv1dFwd` validates the conv state as
//     f32 (ops.cpp CheckConvCommon; bf16 only where the backend advertises
//     SupportsCompressedConvState). The persistent page `MakeNemotronHKVCache`
//     declares is the cache dtype and W6 owns reconciling the two.
//   * `ssm` carries `NemotronHSsmCacheDType(params, act_dtype)` — "float32" on
//     this checkpoint whatever the model dtype. A shared helper keyed on
//     Qwen3.5's `mamba_ssm_dtype` spelling returns the CONVOLUTION dtype here
//     and silently halves the recurrent state; nemotron_h.h records why.
//
// With FRESH state and a single leg the ssm dtype is UNOBSERVABLE in the output
// — the scan computes in f32 and only STORES `final_states` at this dtype. It
// becomes observable exactly when a state is carried in, which is what the
// two-leg gate exercises and what the W6 paged decode does on every step.
struct NemotronHMambaState {
  std::vector<float> conv;  // [conv_dim, conv_kernel-1] f32
  NemotronHOwned ssm;       // [mamba_num_heads, mamba_head_dim, ssm_state_size]
  // False on the first leg (both states read as zero); set by the mixer on
  // return so the next leg carries them in.
  bool has_initial = false;
};

// Mamba2 mixer over ONE sequence of T tokens. `state` nullptr means FRESH state
// discarded on return (the single-leg case); non-null carries it in and out.
std::vector<float> NemotronHMamba2Mixer(const NemotronHMambaWeights& w,
                                        const NemotronHParams& params,
                                        const std::vector<float>& hidden_normed,
                                        int64_t num_tokens, vt::DType act_dtype,
                                        vt::Queue& queue,
                                        NemotronHMambaState* state = nullptr);

std::vector<float> NemotronHAttentionMixer(const NemotronHAttentionWeights& w,
                                           const NemotronHParams& params,
                                           const std::vector<float>& hidden_normed,
                                           int64_t num_tokens, vt::DType act_dtype,
                                           vt::Queue& queue);

std::vector<float> NemotronHMoeMixer(const NemotronHMoeWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue);

std::vector<float> NemotronHMlpMixer(const NemotronHMlpWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue);

// ─── the whole decoder ──────────────────────────────────────────────────────

// Optional per-layer capture. `hidden` holds the residual stream AFTER each
// layer's mixer add, `normed` the norm input each mixer saw; both [T,H] and in
// f32 for comparison. Populated only when `capture` is true. This is the
// instrument porting-a-model.md §3 asks for: per-layer activations, not tokens.
struct NemotronHTrace {
  bool capture = false;
  std::vector<std::vector<float>> normed;  // [L][T*H]
  std::vector<std::vector<float>> mixer;   // [L][T*H]
  std::vector<std::vector<float>> hidden;  // [L][T*H] (residual after the layer)
  std::vector<float> final_normed;         // [T*H]
};

// Logits `[num_requested, vocab_size]` in f32 for the requested positions (all
// positions when `logits_indices` is empty).
std::vector<float> NemotronHForward(const NemotronHHostWeights& host,
                                    const NemotronHParams& params,
                                    const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& logits_indices,
                                    vt::Queue& queue, NemotronHTrace* trace = nullptr);

// Greedy argmax decode over a fresh forward per step. Single sequence; the
// incremental paged/recurrent decode is W6's.
std::vector<int32_t> NemotronHGreedyDecode(const NemotronHHostWeights& host,
                                           const NemotronHParams& params,
                                           const std::vector<int32_t>& prompt,
                                           int num_new, vt::Queue& queue);

}  // namespace vllm
