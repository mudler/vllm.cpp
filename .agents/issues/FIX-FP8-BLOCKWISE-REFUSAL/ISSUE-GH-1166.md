ID: ISSUE-GH-1166
Title: `Qwen/Qwen3.8-27B-FP8` is block-wise (fine-grained 128x128) FP8 and this tree implements per-tensor FP8 only, so the load stops on a message that names the wrong thing. Measured live at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` on 2026-08-17: the config declares `quant_method` `fp8`, `weight_block_size` `[128, 128]` and `activation_scheme` `dynamic`, and the safetensors header of `layers-3.safetensors`, read by RANGE REQUEST rather than downloaded, gives `self_attn.q_proj.weight` `F8_E4M3` `[12288, 5120]` beside `self_attn.q_proj.weight_scale_inv` `BF16` `[96, 40]`, which is exactly `[12288/128, 5120/128]`, with ZERO `input_scale` tensors in the shard. `LoadAttnDense` branches on the weight dtype alone (`qwen3_5_dense_weights.cpp:479`) so the block-wise projection enters the per-tensor arm at `:480`, and `LoadFp8Raw` (`qwen3_5_weights.cpp:449`) asks for `<proj>.weight_scale` at `:458`, which this checkpoint spells `weight_scale_inv`, so the resolver at `qwen3_5_dense_weights.cpp:682` raises `tensor not found: ...q_proj.weight_scale`. Nothing is missing from the checkpoint. The reader is sent after a tensor upstream never writes in this mode instead of being told the fine-grained arm is absent. NOT the silently-wrong-numerics case, and the check that rules it out is recorded rather than assumed: `ReadF32Scalar` (`qwen3_5_weights.cpp:312`) bounds its input with `t.nbytes >= sizeof(float)`, a LOWER bound, so a `[96, 40]` scale would pass and read as block `(0,0)`, but the NAME miss stops the load before that scalar read, and upstream makes the spelling strictly conditional on block quant (`weight_scale_inv if self.block_quant else weight_scale`, `fp8.py:511` at pin `555967922`), so no upstream block-wise checkpoint reaches it. FIXED IN FLOW as a named refusal at `ModelRegistry::Load`; the block-wise arm itself stays owed
Row: FIX-FP8-BLOCKWISE-REFUSAL
State: UNKNOWN
Kind: bug
GitHub: 1166
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:360`

### Frozen archive evidence

> | [#1166](https://github.com/mudler/vllm.cpp/issues/1166) | `FIX-FP8-BLOCKWISE-REFUSAL` | `Qwen/Qwen3.8-27B-FP8` is block-wise (fine-grained 128x128) FP8 and this tree implements per-tensor FP8 only, so the load stops on a message that names the wrong thing. Measured live at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` on 2026-08-17: the config declares `quant_method` `fp8`, `weight_block_size` `[128, 128]` and `activation_scheme` `dynamic`, and the safetensors header of `layers-3.safetensors`, read by RANGE REQUEST rather than downloaded, gives `self_attn.q_proj.weight` `F8_E4M3` `[12288, 5120]` beside `self_attn.q_proj.weight_scale_inv` `BF16` `[96, 40]`, which is exactly `[12288/128, 5120/128]`, with ZERO `input_scale` tensors in the shard. `LoadAttnDense` branches on the weight dtype alone (`qwen3_5_dense_weights.cpp:479`) so the block-wise projection enters the per-tensor arm at `:480`, and `LoadFp8Raw` (`qwen3_5_weights.cpp:449`) asks for `<proj>.weight_scale` at `:458`, which this checkpoint spells `weight_scale_inv`, so the resolver at `qwen3_5_dense_weights.cpp:682` raises `tensor not found: ...q_proj.weight_scale`. Nothing is missing from the checkpoint. The reader is sent after a tensor upstream never writes in this mode instead of being told the fine-grained arm is absent. NOT the silently-wrong-numerics case, and the check that rules it out is recorded rather than assumed: `ReadF32Scalar` (`qwen3_5_weights.cpp:312`) bounds its input with `t.nbytes >= sizeof(float)`, a LOWER bound, so a `[96, 40]` scale would pass and read as block `(0,0)`, but the NAME miss stops the load before that scalar read, and upstream makes the spelling strictly conditional on block quant (`weight_scale_inv if self.block_quant else weight_scale`, `fp8.py:511` at pin `555967922`), so no upstream block-wise checkpoint reaches it. FIXED IN FLOW as a named refusal at `ModelRegistry::Load`; the block-wise arm itself stays owed | bug |

## Resolution

-
