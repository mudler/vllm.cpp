ID: ISSUE-GH-2166
Title: **Muse Glimmer's tower could not ride the #1359 bf16 storage change, because its `compute_dtype = kF32` per-stage gate computes on the stored weight values.** The spec's §4.4 ruling that widening back is bit-identical holds for the production loader reading an all-BF16 checkpoint; it does not hold for `test_muse_glimmer_vision`, whose weights are a synthetic f32 LCG that `scripts/mm/muse_glimmer_vision_ref.py:52-61` builds as `torch.float32` and never rounds. MEASURED on a scratch tree with the bf16 store applied: the five f32-arm stages move from rel_l2 1.0-3.0e-07 to 2.164e-03 / 2.193e-03 / 2.220e-03 / 2.892e-03 / 3.462e-03 against a 1e-6 bound — five assertions red, three orders out. The PRODUCTION path is unaffected and that is measured too: the bf16 arm read `rel_l2=5.951e-03 max_abs=3.675e-02` in the same tree, byte-for-byte what it reads today, so the 3.580 GiB -> 7.161 GiB widening is genuinely removable and only the gate stands in the way. Owed: round the LCG through bf16 on BOTH sides and regenerate `muse_glimmer_vision_goldens.inc` (a reference change that needs its own red/green argument and must not ride in the change it gates), then narrow the four structs, grow `Upload` into the `UploadWeight` shape Qwen3-VL now has, and restore `TOWER_RESIDENT_BYTES` for `muse-glimmer` plus the `WIDEN` mirror to 1x. Threshold unchanged from `specs/vision-tower-dtype-polarity.md` §6.1: >= 3,459,322,368 B on the default arm
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 2166
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:841`

### Frozen archive evidence

> | [#2166](https://github.com/mudler/vllm.cpp/issues/2166) | `ENG-MM-INPUT-PIPELINE` | **Muse Glimmer's tower could not ride the #1359 bf16 storage change, because its `compute_dtype = kF32` per-stage gate computes on the stored weight values.** The spec's §4.4 ruling that widening back is bit-identical holds for the production loader reading an all-BF16 checkpoint; it does not hold for `test_muse_glimmer_vision`, whose weights are a synthetic f32 LCG that `scripts/mm/muse_glimmer_vision_ref.py:52-61` builds as `torch.float32` and never rounds. MEASURED on a scratch tree with the bf16 store applied: the five f32-arm stages move from rel_l2 1.0-3.0e-07 to 2.164e-03 / 2.193e-03 / 2.220e-03 / 2.892e-03 / 3.462e-03 against a 1e-6 bound — five assertions red, three orders out. The PRODUCTION path is unaffected and that is measured too: the bf16 arm read `rel_l2=5.951e-03 max_abs=3.675e-02` in the same tree, byte-for-byte what it reads today, so the 3.580 GiB -> 7.161 GiB widening is genuinely removable and only the gate stands in the way. Owed: round the LCG through bf16 on BOTH sides and regenerate `muse_glimmer_vision_goldens.inc` (a reference change that needs its own red/green argument and must not ride in the change it gates), then narrow the four structs, grow `Upload` into the `UploadWeight` shape Qwen3-VL now has, and restore `TOWER_RESIDENT_BYTES` for `muse-glimmer` plus the `WIDEN` mirror to 1x. Threshold unchanged from `specs/vision-tower-dtype-polarity.md` §6.1: >= 3,459,322,368 B on the default arm | bug |

## Resolution

-
