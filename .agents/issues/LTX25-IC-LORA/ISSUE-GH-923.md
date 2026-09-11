ID: ISSUE-GH-923
Title: Port upstream `ICLoraPipeline` (`ltx-pipelines/ic_lora.py` @ `fd4ded7f`): read an IC-LoRA adapter and its `__metadata__`, fuse the delta into the DiT at load, and expose it through the `ltx-2.5` load extras and `ltx2-gen --lora`. The conditioning half already landed — `Ltx2ConditionVideoByReference` (`ltx2_conditioning.cpp:221`) and `Ltx2ConvVideoEncode` are ported and gated, and `ref_video_dir` reaches the engine as a dir of `frame_%06d.ppm` — so the gap is the adapter path the reference refusal names at `ltx2_video.cpp:1341-1343`. The tree's `include/vllm/lora/` is NOT this mechanism: it is vLLM's runtime punica brick (f32, slot-indexed, `LinearMethodBase`), where LTX fuses at LOAD (`loader/fuse_loras.py:119-150`), so this row does not route through it. One hook after `MaterializeDitTensor` (`ltx2_loader.cpp:424-499`) serves the F32, BF16, FP8 and NVFP4 arms at once, because both quantized branches already `return vt::DType::kBF16`. Retires the `kLoraFusion` marker, which [#691](https://github.com/mudler/vllm.cpp/issues/691) predicted would go false exactly here
Row: LTX25-IC-LORA
State: UNKNOWN
Kind: feature
GitHub: 923
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:244`

### Frozen archive evidence

> | [#923](https://github.com/mudler/vllm.cpp/issues/923) | `LTX25-IC-LORA` | Port upstream `ICLoraPipeline` (`ltx-pipelines/ic_lora.py` @ `fd4ded7f`): read an IC-LoRA adapter and its `__metadata__`, fuse the delta into the DiT at load, and expose it through the `ltx-2.5` load extras and `ltx2-gen --lora`. The conditioning half already landed — `Ltx2ConditionVideoByReference` (`ltx2_conditioning.cpp:221`) and `Ltx2ConvVideoEncode` are ported and gated, and `ref_video_dir` reaches the engine as a dir of `frame_%06d.ppm` — so the gap is the adapter path the reference refusal names at `ltx2_video.cpp:1341-1343`. The tree's `include/vllm/lora/` is NOT this mechanism: it is vLLM's runtime punica brick (f32, slot-indexed, `LinearMethodBase`), where LTX fuses at LOAD (`loader/fuse_loras.py:119-150`), so this row does not route through it. One hook after `MaterializeDitTensor` (`ltx2_loader.cpp:424-499`) serves the F32, BF16, FP8 and NVFP4 arms at once, because both quantized branches already `return vt::DType::kBF16`. Retires the `kLoraFusion` marker, which [#691](https://github.com/mudler/vllm.cpp/issues/691) predicted would go false exactly here | feature |

## Resolution

-
