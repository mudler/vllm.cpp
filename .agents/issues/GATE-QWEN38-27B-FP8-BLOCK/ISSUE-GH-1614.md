ID: ISSUE-GH-1614
Title: **Three sites said `Qwen/Qwen3.8-27B-FP8` ships "~400" `modules_to_not_convert` entries, and at revision `017b9c7a` it ships 882** (882 unique, 636 outside the vision tower). The number is the evidence for an ARGUMENT -- it is why `IsFp8BlockProjection` reads the config AND the tensors instead of probing dtypes -- so being wrong by more than 2.2x invites the next reader to re-derive it. No reading of the list produces ~400: the visual entries are duplicated under two naming conventions, so distinct modules are about 759, and half of 882 is 441. Sites: the comment above `IsFp8BlockProjection`, the comment above `Fp8BlockQuantConfig::modules_to_not_convert`, and `.agents/specs/model-fp8-block-weight.md`. The routing itself is correct and no defect in it is asserted; two other claims in the same comment were checked against the checkpoint headers and hold (zero `input_scale` tensors, and the `[96, 40]` block-grid hazard is real). Found while auditing the checkpoint for #1613, fixed in the same flow
Row: GATE-QWEN38-27B-FP8-BLOCK
State: UNKNOWN
Kind: bug
GitHub: 1614
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:579`

### Frozen archive evidence

> | [#1614](https://github.com/mudler/vllm.cpp/issues/1614) | `GATE-QWEN38-27B-FP8-BLOCK` | **Three sites said `Qwen/Qwen3.8-27B-FP8` ships "~400" `modules_to_not_convert` entries, and at revision `017b9c7a` it ships 882** (882 unique, 636 outside the vision tower). The number is the evidence for an ARGUMENT -- it is why `IsFp8BlockProjection` reads the config AND the tensors instead of probing dtypes -- so being wrong by more than 2.2x invites the next reader to re-derive it. No reading of the list produces ~400: the visual entries are duplicated under two naming conventions, so distinct modules are about 759, and half of 882 is 441. Sites: the comment above `IsFp8BlockProjection`, the comment above `Fp8BlockQuantConfig::modules_to_not_convert`, and `.agents/specs/model-fp8-block-weight.md`. The routing itself is correct and no defect in it is asserted; two other claims in the same comment were checked against the checkpoint headers and hold (zero `input_scale` tensors, and the `[96, 40]` block-grid hazard is real). Found while auditing the checkpoint for #1613, fixed in the same flow | bug |

## Resolution

-
