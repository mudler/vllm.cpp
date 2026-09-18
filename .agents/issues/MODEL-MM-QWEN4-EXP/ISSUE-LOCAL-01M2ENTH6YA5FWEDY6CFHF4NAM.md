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
elements against 115 MB of weight traffic per layer. **This covers `ssm_out`'s
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

-
