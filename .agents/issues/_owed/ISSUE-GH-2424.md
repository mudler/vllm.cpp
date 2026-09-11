ID: ISSUE-GH-2424
Title: qwen4_exp is directly wireable to the MoE placement seam: one call swap, no refactor
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2424
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## What
>
> `docs/FEATURES.md` records that qwen4_exp "is NOT on the PLACEMENT seam: the loop calls `RunMoeBlock` and not `RunMoePlaced`". That is accurate about the current code, but it reads as an interface gap and there is none — qwen4_exp's MoE step already has exactly the seam's shape.
>
> ## Evidence
>
> `RunQwen4ExpMoeBlock` (`src/vllm/model_executor/models/qwen4_exp_moe.h`) takes a `[T, hidden_size]` bf16 tensor and returns `MoeBlockOutput { vt::Tensor tensor; std::shared_ptr<void> storage; }`, which is **structurally identical** to `MoePlacedOutput` (`include/vllm/model_executor/moe_placement_seam.h`). Its call site in `qwen4_exp_forward.cpp` passes `mlp_in.t()`, a device `[T,H]` bf16 buffer, and the enclosing forward already trades in the seam's `dense_attn::Dev` / `DBuf` glue.
>
> It is not blocked by any of the three reasons that block the others: not Gemma4's accumulate-into-caller returning `void`, not Laguna's per-token host-float FFN boundary (#2050), not DeepSeek-V4's host-resident experts.
>
> ## The wiring
>
> Include `moe_placement_seam.h` and replace the call with `RunMoePlacedPair(d, il, mlp_in.t(), T, H, body)`, where the body adapts `MoeBlockOutput` to `MoePlacedOutput` — byte-for-byte the adapter `qwen3_moe.cpp` already uses. Three mechanical notes:
>
> - keep `Qwen4ExpMoeBlockWeights` built OUTSIDE the lambda: it takes a non-const reference and mutates it via `OwnedBytes::KeepAlive()`;
> - the lambda's `Dev` parameter must not be named `p`, which is already the `Qwen4ExpParams` in scope (`kimi_linear_device.cpp` names its `pd`);
> - `il` is the correct `layer_index` — every layer of this model has a MoE block, so plan indices line up with `0..L-1`.
>
> ## No refusal is owed
>
> `Qwen4ExpMoeBlockWeights` fills only the keep-quant stacked arm or the per-expert bf16 borrowed-view arm, and hard-refuses a third dtype. It never touches `expert_gate_fp4` / `expert_up_fp4` / `expert_down_fp4`, so **this architecture has no fp4 arm** and the `placeable=` refusal the Qwen3 sites carry has no analogue to trip here.
>
> Stronger: the adapter is rebuilt per layer per step, so `ResidentWeight::d_dev` is never reused and the device arm re-uploads the tower each step (the forward's own comment says so). A CPU placement is therefore strictly *cheaper* than the unplaced device arm for this architecture today — the opposite of the fp4 defect the refusal exists for.
>
> ## Verification owed before this is claimed done
>
> The placed branch is unreachable on a CPU-only build (#2383), so wiring this cannot be verified by any test CI can run. It needs the same treatment the install got: an NMSE comparison on a GPU box via `VT_PLACEMENT_DUMP_MOE`, against the 5e-4 cross-device bar.
>
> Found while gating #2382. Not implemented here.
>
> Row: `ENG-HYBRID-PLACEMENT`

## Resolution

Commit `1444c4fb83b5c3a09d23156723a1426883562a6e` routes qwen4_exp through the placement seam described by this issue; GitHub closed it as COMPLETED on 2026-08-31.
