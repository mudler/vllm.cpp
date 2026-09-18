ID: ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM
Title: The GDN V-head reorder dequantizes 36 layers of projections to bf16 at load, 61% of decode weight traffic, and permuting a vector instead of the weight would avoid it entirely
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: -

## Problem

MEASURED by `ISSUE-LOCAL-01M2EK69SESGH6ST1ESMFZC808`'s attribution and confirmed
by independent arithmetic here: the decode step moves **6.792 GB** of weights,
and **4.152 GB of it -- 61% -- is a load-time bf16 expansion of Gated DeltaNet
projections that are 1.508 GB in the file.** A 2.75x inflation of the largest
single term in the step.

## The tree already states the cause, against itself

`qwen4_exp_weights.cpp:292-302`:

> A reordered projection is LAYOUT-REWRITTEN at load, so it is
> `kTransformedWeight` and can never keep its blocks -- a k-quant superblock
> spans elements the permutation moves. [...] **THE COST IS REAL AND IT IS NOT
> HIDDEN: on the released 16-vs-48 config the reorder is always on, so every
> Gated DeltaNet projection of all 36 linear layers expands to bf16 at load
> rather than staying Q5_K/Q6_K.**

`linear_num_key_heads` 16 against `linear_num_value_heads` 48 makes `reorder`
permanently true on the released artifact, so `DequantAll` + `Bf16From` runs for
`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha` and `ssm_out` on all 36 layers.

Per GDN layer, derived independently and matching the attribution's 4.152 GB to
three digits:

| tensor | shape | params | reorder kind |
|---|---|---|---|
| `attn_qkv` | `[conv_dim 10240, 2560]` | 26.2 M | ROW (trailing V rows only) |
| `attn_gate` | `[value_dim 6144, 2560]` | 15.7 M | ROW (all rows) |
| `ssm_out` | `[2560, value_dim 6144]` | 15.7 M | **COLUMN** |
| `ssm_beta`, `ssm_alpha` | `[48, 2560]` each | 0.25 M | ROW |
| | | **57.7 M x 36 x 2 B = 4.152 GB** | |

## THE FIX THE COMMENT DOES NOT CONSIDER: permute a VECTOR, not the weight

The comment offers one alternative and rejects it -- "permuting whole rows inside
the block stream [...] is only available for the ROW reorders, never for
`out_proj`'s COLUMN one" -- which is true and covers 73% of the expansion.

**But a permutation of a GEMV operand does not have to be applied to the weight
at all.** For `out = W x`:

- a ROW permutation `P` of `W` satisfies `(P W) x = P (W x)` -- **permute the
  OUTPUT vector** after the GEMV;
- a COLUMN permutation satisfies `(W P) x = W (P x)` -- **permute the INPUT
  vector** before it.

Both are exact re-indexings with no arithmetic, so both are BIT-IDENTICAL to
today's result, and both are O(rows) or O(cols) on a vector of at most 10,240
elements against 115 MB of weight traffic per layer.

> **BOTH HALVES OF THAT SENTENCE WERE CORRECTED AFTER IMPLEMENTATION. See
> "Two corrections to this issue's own gate" and "The cost is O(T * N)" in
> `## Resolution` below before quoting it.** **This covers `ssm_out`'s
column reorder too**, which is the case the comment calls unavailable -- because
it stops at permuting the weight.

The weights then stay verbatim k-quant operands: `LoadMatmul` on every branch,
`reorder` no longer forcing `kTransformedWeight`.

## What it is worth

| | bytes/step | floor at 273 GB/s | |
|---|---|---|---|
| today, as loaded | 6.792 GB | 24.88 ms | ~40 tok/s |
| ROW reorders only (73%) | ~4.81 GB | 17.6 ms | ~57 tok/s |
| **all five, including `ssm_out`** | **4.148 GB** | **15.19 ms** | **~66 tok/s** |

The reference is 66.17 tok/s. **The fixed floor and the reference coincide**,
which is not a coincidence worth over-reading -- sojufx runs NVFP4, a different
width -- but it does mean this defect alone accounts for the order of the gap.

## What is NOT established, and the risks a scope must answer

1. **Consumer layout.** The reorder exists so the rest of the GDN path sees HF
   V-head order. Permuting vectors instead moves that obligation to every
   consumer of these projections -- the causal conv, the delta rule, the gate.
   Each must be shown to see the same order it sees today. This is where the work
   is, and it is a correctness question before it is a speed one.
2. **Which vector, and how often.** Permuting the output of `attn_qkv` once per
   layer per step is cheap; permuting inside a loop would not be. The scope must
   state the site.
3. **`ssm_beta`/`ssm_alpha` are `[48, 2560]`** -- 0.25 M params, negligible bytes.
   Include them for uniformity of the layout argument, not for the traffic.
4. **The bandwidth is not promised.** The floors above assume the freed traffic
   converts at today's achieved 88.3 GB/s or better. Our own `QuantDotGemm*`
   family measures 93.4 GB/s (34.2% of peak) against cuBLAS's 162.3 GB/s
   (59.5%), so moving these operands from cuBLAS bf16 GEMV to our quant kernels
   may LOWER the achieved rate even as it lowers the bytes. **That trade must be
   measured, not assumed** -- it is the one way this fix could disappoint.
5. Upstream parity: transformers 5.16.0 is the algorithm pin and defines the
   V-head order. A vector permutation must be shown equivalent to its layout, not
   merely self-consistent.

## Gate

Bit identity against today's output on the released artifact, per the row's
standing bar for a re-layout: this change must not move a single float. The
`ReorderVRows`/`ReorderVCols` helpers and their fixtures already exist
(`qwen4_exp_weights.cpp:231`), so a red-first is available by permuting one
vector wrongly and showing the goldens catch it.


## Resolution

IMPLEMENTED 2026-09-14, branch `row/MODEL-MM-QWEN4-EXP-gdn-impl` off
`cef9f821632eb33156bb7d24a0a85eeab7f7484a`. The five projections load verbatim;
`vt::VHeadPermute` (new op, CPU + CUDA + ROCm) re-indexes the vectors at four
sites, all at the projection boundary. The consumer audit is in
`.agents/specs/qwen4-exp-flash-next.md`, "THE CONSUMER AUDIT FOR THE DEFERRED
V-HEAD PERMUTATION".

### Two corrections to this issue's own gate, and both matter

**1. "Both are exact re-indexings, so both are BIT-IDENTICAL" is true of the
re-indexing and not of the reduction that consumes it.** A ROW permutation moves
whole dot products, so `attn_qkv`, `attn_gate`, `ssm_beta` and `ssm_alpha` are
bit-identical by construction. A COLUMN permutation permutes the summation order
INSIDE every dot product, and floating-point addition is not associative, so
`ssm_out` is bit-identical only under exact arithmetic. MEASURED on `thor:gpu0`
(CPU arm, K = 2 against R = 3, T = 3 decode rows, all five deferred at once):
the two arms agree BIT FOR BIT, `bad == 0` over the whole `[T, H]` output.
So it holds here -- but it holds as a measurement, not as an identity, and the
measurement is narrow on TWO axes, not one:

1. `value_dim` **96**, not the released **6144** -- the reordered dot product is
   64x shorter than the one that ships, and a column permutation's
   reassociation error grows with the reduction length;
2. the fixture's `out_proj` is `[value_dim, H]` with `nk` UNSET through the
   **bf16 CPU `Matmul`** (`test_gdn_v_head_permute.cpp`), while the released
   path is `[H, value_dim]` with `nk = true`, a **Q6_K** operand through
   **`QuantDotGemm*`**. Orientation, encoding and kernel each pick their own
   summation order and none of the three is held fixed by this reading.

The released orientation is covered on the VALUE axis by the deferred subcase of
`test_qwen4_exp_layer_loop.cpp`, but at a tolerance rather than at equality, so
the bit-identity claim stays a 96-wide bf16-CPU reading. The two shipped
comments that asserted it unqualified (`qwen4_exp_weights.cpp:280-288`,
`include/vt/ops.h:229`) were corrected on 2026-09-15; the commit body of
`708a96589` is immutable and still carries the unqualified wording.

### 3. "O(n) on a vector of at most 10,240 elements" IS A DECODE READING WITH T DROPPED

The op runs on `mixed [T, 10240]`, `z [T, 6144]`, the out-projection input
`[T, 6144]` and `a`/`b` `[T, 48]` each, once per layer per forward -- **O(T * N),
not O(N)**. Per token per layer that is 22,624 elements gathered read-and-written
at bf16 = 90.5 kB, so 3.26 MB per token over 36 layers, against a saving of
2.649 GB per STEP whatever T is.

| | permutation cost | saving | net |
|---|---|---|---|
| breakeven | 2.649 GB | 2.649 GB | T ~= **817** |
| one T = 2048 prefill step | 6.67 GB | 2.649 GB | **+4.0 GB, net-POSITIVE** |
| one decode step | 3.26 MB | 2.649 GB | **-2.646 GB** |

**THE PREFILL ROW ALONE READS AS A REGRESSION AT THE DEFAULT TOKEN BUDGET AND IT
IS NOT ONE FOR ANY GENERATION LONGER THAN TWO TOKENS.** The 4.0 GB a T = 2048
prefill overpays is repaid after **1.5 decode steps**; 2048 prompt + 16 decodes
is already **-38 GB**, and this row's reference workload (35-token prompt, 400
decodes) is **-1060 GB**. The prefill loss is real and worth stating; it does not
justify folding the re-indexing into consumer addressing, which is a different
and much larger change.

**NEITHER HALF IS DEVICE-MEASURED.** Both are byte arithmetic, and the
measurement is owed under `## Owed` in `.agents/specs/qwen4-exp-flash-next.md`.

**2. "Bit identity against today's output on the released artifact" is not
reachable by ANY implementation of this fix.** The fix's purpose is to stop
dequantizing: today the five projections are bf16 values rounded from an f32
dequant and consumed by cuBLAS `gemvx`; afterwards they are native Q5_K/Q6_K
operands consumed by `QuantDotGemm*`. Different operand encodings through a
different kernel cannot produce identical floats, and a gate demanding it would
be demanding the defect back. What the committed tests gate at EQUALITY is the
permutation mechanism held apart from the residency change -- same values, same
dtype, permuted at load versus permuted on the vector.

### The bytes, re-derived from the committed manifest

`ssm_beta` and `ssm_alpha` are **F32 in the released file**, not quantized, so
they contribute nothing; the saving is the three k-quant towers.

| tensor | ggml type | bf16/layer | file/layer |
|---|---|---|---|
| `attn_qkv` | Q5_K | 52.429 MB | 18.022 MB |
| `attn_gate` | Q5_K | 31.457 MB | 10.813 MB |
| `ssm_out` | Q6_K | 31.457 MB | 12.902 MB |
| x 36 layers | | **4.152 GB** | **1.503 GB** (2.763x) |

Decode step **6.792 -> 4.142 GB, -39.0%**, which reproduces this issue's own
arithmetic to three digits from an independent path.

### THE MEASURED STEP CHANGE IS NOT YET TAKEN, and the byte figure must not be read as one

This issue's risk 4 is the thing to measure and it is still owed. At the two
rates this row has already measured -- cuBLAS bf16 GEMV at 162.3 GB/s against
`QuantDotGemm*` at 93.4 GB/s -- the PREDICTION is 25.58 ms of kernel time
becoming 16.09 ms, a **9.5 ms** saving against a 76.9 ms step. That is roughly a
third of what the 2.763x byte ratio alone suggests, and it is a prediction from
two previously measured rates, not a measurement of this change. Taking it needs
a CUDA build on the released 67.564 GiB artifact, which this implementation
session did not reach.

### Gate evidence, `thor:gpu0`, CPU arm

`rc` job `d83ff516-160e-4444-857b-60cd5d17ebce`, Release, `-j 8`:

```
test_gdn_v_head_permute        3/3   cases,  204/204 assertions   (md5 075694c39fd8ab6dcf9f95db83873e45)
test_qwen3_5_gdn_spec_routing  7/7   cases,   82/82
test_qwen4_exp_layer_loop     14/14  cases,  470/470
test_qwen4_exp_qsa            14/14  cases, 7263/7263
test_qwen4_exp_forward         2/2   cases,  429/429
test_qwen4_exp_gguf_load_plan 10/10  cases, 7462/7462
test_qwen4_exp_matmul_bt_dtype 3/3   cases,    2/2
```

### The device arms COMPILE, and that is all that is claimed of them

`vt::VHeadPermute`'s CUDA and ROCm kernels are transcriptions of the CPU
reference and were compiled inside leases, each as a single translation unit:

| arm | box | toolchain | result |
|---|---|---|---|
| CUDA | `thor:gpu0`, `rc` job `e2cf10da-ce6f-4e5a-a10a-3f91b6d16ff6` | nvcc 13.0.88, `-arch=sm_110` | `NVCC_RC=0`, `cuda_gdn.o` 4,427,912 B |
| ROCm | `strix:gpu0`, `rc` job `9fa1753e-09be-4c0b-8527-eb265823ad16` | HIP 7.2.53211 / ROCm 7.2.4, `--offload-arch=gfx1151` | `HIPCC_RC=0`, `rocm_gdn.o` 133,912 B |

**NEITHER WAS EXECUTED.** A compile says the kernel is well-formed for that
architecture and says nothing about its values. The equality and red-first cases
above ran on the CPU arm only, so the device kernels are OWED a run -- which is
the same shape as saying a CUDA forward for this model is owed, since no token
has come out of a CUDA device for `qwen4_exp` at all.

The red-first is inside `test_gdn_v_head_permute`: each of the four deferred
tensor groups is flipped to the WRONG permutation direction in turn and the
output must diverge. At K = 2 against R = 3 the backwards map is a genuinely
different map -- at K = 1 it is the identity and at K == R its own inverse, and
neither of those fixtures could tell a correct permutation from none.


## Repair after the fresh review (2026-09-15, `aea9b99f0` off `708a96589`)

The fresh review returned FAIL. The permutation MECHANISM survived seven
mutations and no defect was found in it; four findings around it are repaired.

**The largest was that the END-TO-END SUITE WAS BLIND TO V-HEAD ORDER.**
`FillGdn` in `test_qwen4_exp_layer_loop.cpp` builds `Qwen4ExpGdnWeights` by hand
and never set `v_head_perm_key_heads`, so `vt::VHeadPermute` was INERT in the one
transformers-oracle case in the tree, and inverting its two index maps at every
production site left that suite 14/14 green. The oracle case now runs TWO ARMS
against the same golden — the second with the five projections re-indexed into
the converter's TILED order and the flag set, which is also the first coverage of
the RELEASED `out_proj` orientation `[H, value_dim]` with `nk = true`. Gated on
`thor:gpu0` sm_110, `rc` job `9cae72c4-b2bd-4fe3-b388-28639a6e105a`: both arms
read `max|diff| = 0.0118021` against a 0.03 bound, and under the inverted map the
tiled arm reads **1.83194** while the grouped arm in the same binary is unmoved.
Full table, md5s and the restore proof are in
`.agents/specs/qwen4-exp-flash-next.md`, "Mutation record — the deferred V-head
permutation repair".

The other three: the O(n) cost claim gained its T dependence AND the repayment
arithmetic (section 3 above); the `ssm_out` bit-identity claim was narrowed on
both axes in the two shipped comments that still asserted it flat
(`qwen4_exp_weights.cpp`, `include/vt/ops.h`); and three test defects were fixed
— `in_proj_b` and `in_proj_a` flipped together so neither was convicted alone,
`RunLayer` dropped both persistent caches so their write side was unasserted, and
the two untested permute sites (`GdnBlock`, `GdnBlockPagedMixedSpec`) are now
recorded as UNREACHED rather than left to read as coverage.

**Still owed and now named under the spec's `## Owed`:** the prefill cost and its
repayment are byte arithmetic and have never been measured on a device.
