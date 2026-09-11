ID: ISSUE-GH-1613
Title: **The `Qwen/Qwen3.8-27B-FP8` block-wise token gate cannot be taken, because the 28.75 GiB checkpoint is not on the share.** `/mnt/nas_share/rc/ckpt/` holds `qwen3.8-27b-hf`, which is the **bf16** artifact -- no `quantization_config` key, `text_config.dtype = bfloat16` -- and `qwen3.8-q1_0`. Neither is this subject. The share has 3.4 TiB free, so the cost is AUTHORITY: `.agents/developer-preferences.md` authorizes large downloads for the `SPEC-DFLASH2` assets only. Nothing else blocks the gate, and that was not known before: a range-request audit of all 66 shard headers at revision `017b9c7a` shows every one of the 407 `F8_E4M3` tensors has `N % 128 == 0` and `K % 128 == 0`, so the sm120 complete-scale-block refusal (#1453) that makes DSV3's `kv_a_proj_with_mqa` unservable blocks NOTHING here; the ragged GDN `in_proj_a`/`in_proj_b` `[48, 5120]` are `BF16` and named in `modules_to_not_convert`; `weight_scale_inv` ships `BF16` (byte-checked via `data_offsets`, not the label) which `LoadFp8BlockRaw` already widens by value; and the per-layer `layers-<i>.safetensors` naming already resolves through `SelectWeightFiles`. Spec `.agents/specs/gate-qwen38-27b-fp8-block.md`, parent #1189
Row: GATE-QWEN38-27B-FP8-BLOCK
State: UNKNOWN
Kind: gap
GitHub: 1613
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:578`

### Frozen archive evidence

> | [#1613](https://github.com/mudler/vllm.cpp/issues/1613) | `GATE-QWEN38-27B-FP8-BLOCK` | **The `Qwen/Qwen3.8-27B-FP8` block-wise token gate cannot be taken, because the 28.75 GiB checkpoint is not on the share.** `/mnt/nas_share/rc/ckpt/` holds `qwen3.8-27b-hf`, which is the **bf16** artifact -- no `quantization_config` key, `text_config.dtype = bfloat16` -- and `qwen3.8-q1_0`. Neither is this subject. The share has 3.4 TiB free, so the cost is AUTHORITY: `.agents/developer-preferences.md` authorizes large downloads for the `SPEC-DFLASH2` assets only. Nothing else blocks the gate, and that was not known before: a range-request audit of all 66 shard headers at revision `017b9c7a` shows every one of the 407 `F8_E4M3` tensors has `N % 128 == 0` and `K % 128 == 0`, so the sm120 complete-scale-block refusal (#1453) that makes DSV3's `kv_a_proj_with_mqa` unservable blocks NOTHING here; the ragged GDN `in_proj_a`/`in_proj_b` `[48, 5120]` are `BF16` and named in `modules_to_not_convert`; `weight_scale_inv` ships `BF16` (byte-checked via `data_offsets`, not the label) which `LoadFp8BlockRaw` already widens by value; and the per-layer `layers-<i>.safetensors` naming already resolves through `SelectWeightFiles`. Spec `.agents/specs/gate-qwen38-27b-fp8-block.md`, parent #1189 | gap |

## Resolution

-
