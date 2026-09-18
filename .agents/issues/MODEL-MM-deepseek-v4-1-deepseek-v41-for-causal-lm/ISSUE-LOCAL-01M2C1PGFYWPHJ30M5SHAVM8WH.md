ID: ISSUE-LOCAL-01M2C1PGFYWPHJ30M5SHAVM8WH
Title: `E8M0ToF32` implements ONE of the two UE8M0 decodes upstream uses, and they differ at byte 0. Verified at vLLM `e77daef89e`: Engram decodes a ue8m0 scale by BITCAST -- `scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)` (`common/engram.py:613-614`, whose own comment says 'ue8m0 is a power of two, so its byte *is* the fp32 exponent field'), so byte 0 yields the bit pattern 0x00000000 = **+0.0**. MXFP8 decodes ARITHMETICALLY -- `descale = torch.exp2(scales.to(float32) - 127.0)` (`quantization/utils/mxfp8_utils.py:66,237`), so byte 0 yields **2^-127**; `:129-130` states upstream is aware of exactly this, commenting that 'sb == 0 (an all-zero / denormal block) makes that divisor 2**-127' and multiplying by the reciprocal to dodge it. Our `src/vllm/model_executor/model_loader/mxfp4_dequant.cpp:15-22` implements only the arithmetic form (`std::ldexp(1.0F, byte - 127)`) and SIX call sites share it: `gguf_dequant.cpp`, `nvfp4_dequant.cpp`, `mxfp4_dequant.cpp`, `src/vt/rocm/rocm_quant_dequant.h`, `src/vt/cuda/cuda_quant_dot.cu`, `src/vt/cuda/cuda_quant_dequant.cuh`. The MXFP8 arm this is correct for; any arm mirroring the bitcast semantics needs byte 0 to be +0.0 and gets 2^-127 instead. The Engram host reference (V4.1 W3a) REQUIRES the bitcast form, so this must be a SECOND entry point rather than a change to the existing one -- the four existing callers depend on current semantics and nothing upstream pins the two forms against each other, so no ported test would catch a wrong choice. UNVERIFIED and owed by whoever takes it: whether our existing V4 fp8 128x128 block path mirrors a bitcast upstream, which would make this a live numerical defect in shipped code rather than only a constraint on new work
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

One decode where upstream has two; they disagree at byte 0 and no test pins them.

## Resolution

-
