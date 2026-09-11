ID: ISSUE-GH-1322
Title: `vt` cannot express torch's **per-op bf16 rounding**, and no caller can build it. A bf16 torch module rounds at every op boundary; `vt::RmsNorm` keeps full f32 across the weight multiply (`include/vt/ops.h` says so) where `normalization.py:600-606` casts back to the weight dtype first, and `vt::SiluAndMul` computes `silu(g)*u` wholly in f32 where `F.silu(gate)` produces a **bf16 tensor** that is then multiplied. MEASURED on MiniMax-Music3's depth decoder ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), spec §19.4a): the device arm reads worst **110 bf16 ULP** / mean **2.095** against the host reference, and collapsing those two host roundings to the seam's shape takes it to worst 8 / mean 0.0596 and the composed stage to **ZERO — bit-identical**. So ~97 % of the deviation is rounding POLARITY inside two shared ops, not dtype, accumulator or reduction order. A call site cannot split them, because `vt` has no elementwise or row-broadcast multiply — `kMulScalar` takes a scalar and `kMulColVecF32` is an f32 in-place column scale. Closing it needs either a `vt::Mul` with `vt::Add`'s two operand shapes or rounding-faithful modes on the two ops, with CPU and CUDA providers; #1309 records the measurement and stops. A token gate cannot see any of this
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1322
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:453`

### Frozen archive evidence

> | [#1322](https://github.com/mudler/vllm.cpp/issues/1322) | `MUSIC3-DEPTH-DEVICE` | `vt` cannot express torch's **per-op bf16 rounding**, and no caller can build it. A bf16 torch module rounds at every op boundary; `vt::RmsNorm` keeps full f32 across the weight multiply (`include/vt/ops.h` says so) where `normalization.py:600-606` casts back to the weight dtype first, and `vt::SiluAndMul` computes `silu(g)*u` wholly in f32 where `F.silu(gate)` produces a **bf16 tensor** that is then multiplied. MEASURED on MiniMax-Music3's depth decoder ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), spec §19.4a): the device arm reads worst **110 bf16 ULP** / mean **2.095** against the host reference, and collapsing those two host roundings to the seam's shape takes it to worst 8 / mean 0.0596 and the composed stage to **ZERO — bit-identical**. So ~97 % of the deviation is rounding POLARITY inside two shared ops, not dtype, accumulator or reduction order. A call site cannot split them, because `vt` has no elementwise or row-broadcast multiply — `kMulScalar` takes a scalar and `kMulColVecF32` is an f32 in-place column scale. Closing it needs either a `vt::Mul` with `vt::Add`'s two operand shapes or rounding-faithful modes on the two ops, with CPU and CUDA providers; #1309 records the measurement and stops. A token gate cannot see any of this | bug |

## Resolution

-
