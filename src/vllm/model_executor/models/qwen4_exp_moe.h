// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5d-4 — the MoE weight adapter, and the
// one production composition that runs `Qwen4ExpTextSparseMoeBlock` through the
// SHARED sparse-MoE seam instead of a second MoE path.
//
// Issue [#2249](https://github.com/mudler/vllm.cpp/issues/2249) item 4, wave
// issue [#2031](https://github.com/mudler/vllm.cpp/issues/2031), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── THE GAP THIS FILE CLOSES ────────────────────────────────────────────────
// `LoadMoe` (`qwen4_exp_weights.cpp`) produces `Qwen4ExpMoeWeights`, which the
// tree constructs and NOTHING reads. `RunMoeBlock`
// (`qwen3_5_moe_block.h`) consumes `MoeBlockWeights`. #2249 records the two as
// nearly the same object — "the `_kq` arm's shapes match and `KqExpertSlice` is
// dtype-generic, so this looks small". Three of the four differences below are
// invisible to a shape comparison, and one of them is a hard refusal on the
// path the shipped checkpoints actually take:
//
//   1. RANK. `LoadStackedExperts` records the tower as RANK 3 `[E, N, K]`
//      (`qwen4_exp_weights.cpp:160-164`); `MoeBlockWeights::expert_*_kq` is
//      RANK 2 `[E*N, K]`, and the default keep-quant route
//      (`Qwen35GroupedMoeEnabled`, ON) hands that tensor straight to
//      `vt::MatmulBTQuantGrouped`, whose FIRST check is
//      "matmul_bt_quant_grouped: rank-2 out/act/weight required"
//      (`src/vt/ops.cpp:223`). ON THE DEFAULT GROUPED ROUTE a rank-3 tower does
//      not "match"; it throws. That qualifier is load-bearing and is not a
//      hedge: with `VT_QWEN35_GROUPED_MOE=0` the seam takes the per-expert
//      `ExpertMlpKq` path, which reaches `KqResidentSlice`
//      (`qwen3_5.cpp:5665-5678`) — and that helper rebuilds a rank-2 view from
//      its `N`/`K` ARGUMENTS by pointer arithmetic, sets `wt.rank = 2` itself
//      and never reads the tower's declared rank. A rank-3 tower does not throw
//      there, and because the tower is contiguous `[E, N, K]` it even answers
//      correctly. #2249 item 4's "the shapes match" is literally true of that
//      route and false of the default one, which is the one every shipped
//      checkpoint takes. The suite carries both behaviours.
//   2. DTYPE. `LoadMoe` keeps the router and the shared gate in **f32**
//      deliberately (`qwen4_exp_weights.cpp:437-447`). The seam consumes both
//      through `MatmulBf16` / `MatmulF32D` against a **bf16** activation, and
//      the CUDA GEMM accepts only (bf16,bf16) or (f32,f32) —
//      "vt cuda: matmul_bt: unsupported dtype combo" (`cuda_matmul.cu:397-403`).
//      Passing the f32 tensors through works on CPU and dies on every GPU, which
//      is the shape of defect this row has produced four times. They are
//      converted here, which is also what the oracle has: upstream's router
//      `weight` and `shared_expert_gate` are ordinary model-dtype parameters
//      (`modeling_qwen4_exp.py:905`, `:925`), and `F.linear` there returns bf16
//      logits that `softmax(..., dtype=torch.float)` then upcasts (`:909-910`) —
//      exactly the seam's bf16-logits / f32-softmax split.
//   3. ARM SELECTION. `MoeBlock` decides the whole expert path from
//      `w.expert_gate_kq.Empty()` ALONE (`qwen3_5.cpp:7257`, `:7296`) — it never
//      looks at up or down. `GgufLoadPolicy::Route` is per tensor, so a policy
//      that keeps `gate` quantized and expands `down` produces a set the seam
//      reads as keep-quant and then dereferences an EMPTY down tower. Refused by
//      name below rather than left to be discovered as a wrong answer.
//   4. THE BF16 ARM CANNOT USE THE STACKED FIELDS AT ALL. A bf16 tower in
//      `expert_*_kq` takes the same grouped route as (1) and hits
//      "matmul_bt_quant_grouped: weight must be a block-quantized dtype"
//      (`ops.cpp:231`) — route-conditional in exactly the way (1) is, since
//      `MatmulF32Slice`'s generic `vt::MatmulBT` would accept the bf16 slice on
//      the `ExpertMlpKq` path. The bf16 arm therefore fills the PER-EXPERT vectors, and
//      it fills them with zero-copy BORROWED views of the stacked buffer: the
//      released geometry is 512 experts x 640 x 2560, so materialising three
//      per-expert copies per layer is 240 GB across 48 layers. Zero-copy is not
//      an optimisation here, it is the only representable arm.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// vLLM registers `qwen4_exp` at NO revision, so under AGENTS.md "When vLLM has
// no implementation" the ALGORITHM oracle is transformers **5.16.0**, this row's
// accepted lane pin (`.agents/oracles/transformers.md`), at
// `models/qwen4_exp/modeling_qwen4_exp.py`:
//
//   * `Qwen4ExpTextSparseMoeBlock.forward` (:927-938) — shared expert, router,
//     routed experts, `sigmoid(gate) * shared`, then `routed + shared`.
//   * `Qwen4ExpTextTopKRouter.forward` (:907-916) — `F.linear(x, weight)` with
//     `weight [E, H]`, `softmax(dtype=float)`, `topk`, and `norm_topk_prob`.
//   * `Qwen4ExpTextExperts.forward` (:869-894) — `gate_up_proj [E, 2I, H]` split
//     by `chunk(2, dim=-1)`, `down_proj [E, H, I]`, `act_fn(gate) * up`.
//   * `Qwen4ExpTextMLP` (:842-855) — the shared expert, at
//     `shared_expert_intermediate_size`.
//
// The split `gate_exps` / `up_exps` this loader holds is the CONTAINER's doing,
// not a divergence: llama.cpp #27742's converter splits upstream's fused
// `gate_up_proj` at the chunk point, and `ffn_gate_exps` is the FIRST half.
// Feeding the halves to the seam in the other order is mutation M2 and is red.
//
// ─── WHAT IS NOT HERE ────────────────────────────────────────────────────────
// A caller. `ForwardQwen4ExpForConditionalGeneration` still refuses by name, so
// this composition is reached only by its own gate at its merge commit; the
// spec's `## Owed` records that with the row and the issues that own the wiring
// (`MODEL-MM-QWEN4-EXP`, #2031, #2249). Also not here: `norm_topk_prob`. The
// seam hardcodes `renormalize = true` (`qwen3_5.cpp:7242`) and `HfConfig` has no
// field for it, so a config that turned it off could not be represented; the
// upstream default is `True` (`configuration_qwen4_exp.py:163`) and
// `Qwen4ExpParams` does not carry the field, which the spec already owes.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_MOE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_MOE_H_

#include <cstdint>

#include "vllm/model_executor/models/qwen3_5_moe_block.h"   // RunMoeBlock
#include "vllm/model_executor/models/qwen3_5_weights.h"     // MoeBlockWeights
#include "vllm/model_executor/models/qwen4_exp.h"           // Qwen4ExpParams
#include "vllm/model_executor/models/qwen4_exp_weights.h"   // Qwen4ExpMoeWeights
#include "vllm/transformers_utils/hf_config.h"
#include "vt/tensor.h"

namespace vllm {

// The five `HfConfig` fields `MoeBlock` reads, projected from `Qwen4ExpParams`.
// Everything else on the returned config is left at its default: `MoeBlock`
// reads `hidden_size`, `num_experts`, `num_experts_per_tok`,
// `moe_intermediate_size` and `shared_expert_intermediate_size` and nothing
// else, and populating fields it does not read would invite a reader to believe
// they were honoured.
HfConfig Qwen4ExpMoeHfConfig(const Qwen4ExpParams& p);

// Adapt one layer's loaded MoE weights onto the shared `MoeBlockWeights` seam.
//
// `moe` is a NON-CONST reference because the per-expert views borrow its bytes:
// `OwnedBytes::KeepAlive()` converts an owned buffer into a shared read-only one
// in place (it moves the vector into a refcounted holder, so the byte ADDRESS is
// unchanged) and hands back the keep-alive every view holds. The returned
// `MoeBlockWeights` therefore does not own the expert bytes and must not outlive
// `moe`'s owner — which is the same lifetime rule the GGUF mmap residency
// already imposes on every kept weight.
//
// Refuses by name, never silently, on: a shape that does not match `p`, a router
// or shared gate that is not f32, an expert set whose three towers do not agree
// on one residency, and a stacked dtype the seam has no arm for.
MoeBlockWeights Qwen4ExpMoeBlockWeights(Qwen4ExpMoeWeights& moe,
                                        const Qwen4ExpParams& p);

// Run `Qwen4ExpTextSparseMoeBlock` over a device-resident `dh` [T, hidden_size]
// bf16 and return the combined [T, hidden_size] bf16 block output.
//
// This is the whole block, not a helper: it is `RunMoeBlock` with the config
// this model resolves, and it exists so the layer loop enters the shared seam
// through ONE named call rather than reassembling the config at each site.
MoeBlockOutput RunQwen4ExpMoeBlock(vt::Queue& queue, const MoeBlockWeights& weights,
                                   const Qwen4ExpParams& p, const vt::Tensor& dh,
                                   int64_t T);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_MOE_H_
