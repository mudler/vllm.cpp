ID: ISSUE-LOCAL-01M2C2QSFZWCXNQBJFBWYFNV2P
Title: DeepSeek-V4.1 W3b: MXFP8 32x32 UE8M0 host reference
Row: MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

DeepSeek-V4.1-Flash routes a LinearBase whose weight_block_size == [32, 32] and is_scale_e8m0 to ModelOptLinearMethod(kMxfp8Static / kMxfp8Dynamic) (vllm/models/deepseek_v4_1/quant_config.py:186-201 @ e77daef89e). This tree has no MXFP8 reader at all: src/vllm/model_executor/layers/quantization/modelopt_mixed_precision.h:1087-1091 refuses the algo by name with 'this build has no MXFP8 loader'. W3b owes the portable host reference for the family, per the spec's wave table: the checkpoint-to-runtime scale row expansion (repeat_interleave(32, dim=0) BEFORE TP slicing), the runtime per-32-column dequant, the dynamic activation quantizer, the emulation linear arm, and the weight_scale / weight_scale_inv checkpoint name split.

## Resolution

2026-09-13, PARTIAL — the host reference lands; the issue stays OPEN until a
production entry point reaches it.

Landed `include/vllm/model_executor/models/deepseek_v4_1_mxfp8.h`,
`src/vllm/model_executor/models/deepseek_v4_1_mxfp8.cpp` and
`tests/vllm/models/test_deepseek_v4_1_mxfp8.cpp`. The five pieces are
`ExpandMxfp8CheckpointScale`, `DequantMxfp8ToF32` / `DequantMxfp8ToBf16`,
`QuantizeMxfp8E4m3`, `Mxfp8LinearEmulationBf16` (the model-path arm) with
`Mxfp8LinearEmulation` (its f32 reference), and `Mxfp8LinearScaleParamName`.

The unit gate is 16 cases / 7940 assertions green. It ports
`tests/quantization/test_fp8.py:67-238` @ vLLM `e77daef89e`, including that
test's bit-exact (`rtol=0, atol=0`) equivalence between the checkpoint-layout
32x32 dequant and the expanded per-32-column runtime dequant, over both
`tp_rank` values and the six checkpoint shapes it names.

## Mutations

Every mutation below was re-run against the final tree on 2026-09-13 after a
review found one entry of the earlier table falsified. Digests are absolute, so
the recipe that produced them is stated once here and nowhere else:

    host      x86_64, Ubuntu 24.04, g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
    config    CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG -std=c++20 -ffp-contract=off
              -Wall -Wextra -Werror), VLLM_CPP_CUDA=OFF, VLLM_CPP_TRITON=OFF
    tree      /home/mudler/_git/vllm.cpp/.wt/dsv41-w3b
    build     ninja test_deepseek_v4_1_mxfp8 -j 4
    binary    build-cpu/tests/test_deepseek_v4_1_mxfp8
    BASELINE  md5 90e498e6f3325cfb5f60729a3f289342 — 16/16, 7940 assertions

The `tree` line is load-bearing and not decoration. `VT_CHECK` and doctest both
expand `__FILE__`, and CMake feeds the compiler absolute paths, so the linked
binary carries **257** copies of that worktree prefix. A build of identical
source in a differently named directory therefore hashes differently. A
different compiler, build type or flag set does the same. A digest alone proves
nothing without the five lines above, and no reader should expect to reproduce
these exact digits outside that path.

**What the table claims is the DIFFERENCE, not the digits.** Each mutant digest
below differs from the baseline taken in the same tree by the same compiler, so
every row is a mutation the compiler actually applied and not a no-op the
optimiser erased. That comparison is reproducible anywhere; only its absolute
values are tree-relative.

| # | mutation | mutant md5 | result |
|---|---|---|---|
| M1 | `repeat_interleave` stride becomes a tile stride | `9ae27063` | KILLED (3 cases, 86 assertions) |
| M2 | runtime scale row index `n` becomes `n / 32` | `1978c7cb` | KILLED (4 cases, 2610 assertions) |
| M3 | upper clamp 254 becomes 255 | `d9d0fd90` | KILLED (1 case, 2 assertions) |
| M4 | lower clamp 0 removed | `c2440e00` | KILLED (2 cases, 67 assertions) |
| M5 | `block_cols != 32` refusal removed | `87a8fb23` | KILLED (1 case, 2 assertions) |
| M6a | linear-arm weight narrowing truncates instead of rounding | `c83976ce` | KILLED (1 case, 1 assertion) |
| M6b | linear-arm weight narrowing deleted (f32 weight into the dot) | `beec96b9` | KILLED (1 case, 2 assertions) |
| M7 | `expert_dtype` half of the name split dropped | `8e47abeb` | KILLED (1 case, 1 assertion) |
| M8 | bias term dropped | `ca1f11c7` | KILLED (1 case, 16 assertions) |
| M9 | multiply by `exp2(127 - sb)` becomes a true divide | `ebff7c42` | **SURVIVED** |
| M10 | `amax` TINY floor becomes 0 | `8539b56d` | **SURVIVED** |
| M11 | `E8M0ToF32` arithmetic decode becomes the engram bitcast decode | `fe1a90a3` | KILLED (2 cases, 67 assertions) |
| M12 | `DequantMxfp8ToBf16` narrowing truncates instead of rounding | `955bc633` | KILLED (1 case, 3 assertions) |
| M13 | `ceil` becomes `floor` in the exponent byte | `31079adc` | KILLED (2 cases, 130 assertions) |
| M14 | emulation `K > 0` pre-check removed | `30925401` | KILLED (1 case, 1 assertion) |
| M15 | bf16 arm's store is not the bf16 round | `2a7046bd` | KILLED (1 case, 30 assertions) |

Fourteen killed, two survive. The two survivors are recorded rather than papered
over: the MULTIPLY-versus-divide form (M9) and the `amax` TINY floor (M10) are
both indistinguishable on a host WITHOUT flush-to-zero, which is every host this
gate runs on, and **W5's CDNA arm owes both measurements**.

The earlier version of this record said "Five mutations are killed" and then
listed eight, and it counted the bf16 narrowing (M6) as killed when nothing
gated it. That was true: an MXFP8 dequant product is an integer in [1, 15] times
a power of two, so it carries at most 4 significant bits where bf16 holds 8, and
`bf16(v) == v` over the whole product space. The narrowing is separable only at
the EXPONENT floor, where bf16's smallest subnormal is 2^-133 and f32's is
2^-149: the smallest product this format can build, e4m3 `0x01` (2^-9) times
scale byte 0 (2^-127), is 2^-136 — exact in f32, zero in bf16. The gate now
asserts that value, plus the round-to-nearest-even tie 3 * 2^-134, which is what
M6a, M6b, M12 and M15 kill.

## Owed

Four obligations leave W3b for the wiring waves on this row. None of them is a
property of the five functions W3b lands, which is why the unit gate cannot
carry them and why they are named here rather than left to be rediscovered:

1. **W8 — `test_fp8.py:186-188`, the other half of the expansion guarantee.**
   Upstream asserts the MoE expert scale and the engram scale arrive EQUAL to the
   raw checkpoint bytes: the row expansion must NOT be applied to them. W3b has
   no loader, so nothing here can assert where the expansion is and is not wired.
2. **W8 — `test_fp8.py:191`, the routing assertion.**
   `isinstance(linear.quant_method, ModelOptLinearMethod)`. W3b ports the five
   leaves this family routes to and NO routing predicate; `quant_config.py:186-201`
   (`DeepseekV4FP8Config.get_quant_method`) is still unported.
3. **W8 — `test_fp8.py:226-237`, the mapper-rename half of the name split.**
   `Mxfp8LinearScaleParamName` resolves the NAME; nothing holds that
   `_make_deepseek_v4_vl_weights_mapper` then renames `...attn.wq_a.scale` onto
   it while leaving `...attn.wkv.weight` alone.
4. **W8 — the ORDER of the expansion against the TP shard.** The expansion must
   run BEFORE the shard, because a 96-row shared-expert projection has a 3-row
   checkpoint scale that does not divide by a TP world of 2. W3b's gate holds
   this as a property of a test that shards by hand; no shipped code encodes it,
   because no shipped code shards. W8's loader is where it becomes real.

WHAT IS STILL OPEN: nothing calls any of it. W1 registered `deepseek_v41`, but
its forward and its loader refuse by name, so no production entry point reaches
this file; the wiring is owed by W8 (loader) and W4 (host forward assembly) on
this row. W4 wires `Mxfp8LinearEmulationBf16`, not the f32 reference beside it.
`.agents/specs/deepseek-v4-1-flash.md` `## Owed` carries the same record.
