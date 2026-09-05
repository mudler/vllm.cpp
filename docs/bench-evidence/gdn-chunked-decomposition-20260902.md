# How much of vLLM's GDN prefill error is bf16, and how much is chunking, 2 September 2026

Wave GDNDECOMP of [`KERNEL-GDN-CHUNKED-MIRROR`](../../.agents/specs/gdn-chunked-mirror.md),
[#2612](https://github.com/mudler/vllm.cpp/issues/2612).

**The one-line result.** vLLM's chunked gated delta rule sits `2.286426e-04`
(out) and `2.248661e-03` (state) from the exact recurrence. **Of that,
`2.4e-17` is the chunked reassociation and the whole remainder is the bf16
intermediates.** The reassociation is exact: recomputing the same WY
decomposition in `float64` reproduces the sequential recurrence to
`2.43e-17`, which is `1e-13` of the gap it was supposed to explain. Modelling
only where FLA rounds to bf16, and changing nothing else, reproduces vLLM's
distance from exact to seven significant figures.

**Therefore a chunked CPU port that keeps f32 intermediates does not mirror
vLLM. It reproduces our current CPU output.** At bf16 output granularity it
differs from vLLM's real kernel in **4157 of 8192** elements, exactly as the
sequential arm does, and from the sequential arm in **2 of 8192**. Chunking
without bf16 buys the algorithm's name and none of its numbers.

## 1. What was measured, and against what

**The oracle is a committed dump of the real kernel**, not a re-derivation:
`tests/parity/goldens/gdn_prefill_bf16_realdims/`. Its `manifest.json` records
`oracle.callable = "fla/ops/chunk.py::chunk_gated_delta_rule"`, `torch 2.11.0`,
device `NVIDIA GB10`, source `pinned:e24d1b24`. `out.npy` and `state_out.npy`
are that Triton kernel's own outputs. Dims are the real gate dims: `Hk=1`,
`Hv=2`, `Dk=Dv=128`, `scale=0.08838834764831845`, two sequences of 20 and 12
tokens, the first with an initial state and the second without
(`state_in[1]` is all zero, verified: `nonzero=0`).

**The reference is the exact recurrence in `float64`**, run on the golden's own
inputs. `q`, `k`, `v` are stored as raw bf16 bit patterns and are bf16-exact
(asserted); `g`, `beta`, `state_in` are f32. Every arm below consumes those
same values, so input rounding is common mode and cancels. What is measured is
the algorithm and the placement of its rounding, nothing else.

**The golden was dumped at the PREVIOUS pin, and the two revisions were
DIFFED rather than argued about.** Its manifest records
`oracle.source = "pinned:e24d1b24"`; the current parity pin is `5559679229`,
and the FLA sources this study's replica models were read at `5559679229`.
Between those two revisions the vendored tree also MOVED, from
`vllm/model_executor/layers/fla/ops/` to
`vllm/third_party/flash_linear_attention/ops/`, so a path-wise `git diff` reads
as a wholesale addition and says nothing. Diffed across the move, in a checkout
of the vLLM oracle:

| FLA op | `e24d1b24` -> `5559679229` |
|---|---|
| `chunk.py`, `chunk_o.py`, `wy_fast.py`, `solve_tril.py`, `cumsum.py` | **byte-identical** |
| `chunk_delta_h.py` | 4 lines: a `torch.version.hip` guard that drops `num_stages=4` from the autotune list on AMD |
| `chunk_scaled_dot_kkt.py` | 20 lines: a `current_platform.is_rocm()` / `on_gfx1x()` branch, whose non-RDNA arm is the old `b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))` verbatim |

**The golden was dumped on an NVIDIA GB10.** On that device
`current_platform.is_rocm()` is false and `torch.version.hip` is `None`, so both
deltas take the arm that is the old code, character for character. The kernel
that produced `out.npy` is numerically identical at the two revisions, and this
is a diff rather than an inference. Every line number this file cites is read at
`5559679229`; `chunk_o.py` is byte-identical, so its anchors hold at both.

A pin advance that DOES change FLA's rounding placement would invalidate §3, and
the check that catches it is this same diff, not a separation of measured
values.

**The metric is `max|diff|`**, the metric the golden's own manifest uses for its
recorded `chunk_vs_sequential_max_abs_out = 2.441406e-04` and
`chunk_vs_sequential_max_abs_state = 2.248675e-03`.

**The instrument is a numpy replica of FLA's chunked algorithm**, read at the
current pin `5559679229bc961848b121ccdeaa8fa5d79bec98` in the vendored tree
`vllm/third_party/flash_linear_attention/ops/`:
`chunk.py:37-82` (the five-stage pipeline and `solve_tril(output_dtype=k.dtype)`),
`cumsum.py:160-280` (f32 cumsum),
`chunk_scaled_dot_kkt.py:86-116,161` (A in f32),
`solve_tril.py:227-505` (the blocked 16x16 inverse, f32 `ieee` dots),
`wy_fast.py:70-116` (u, w),
`chunk_delta_h.py:132-302,352-357` (the cross-chunk recurrence),
`chunk_o.py:88-138` (the output).
`FLA_CHUNK_SIZE = 64` (`utils.py:31`), matching our `kChunk`
(`src/vt/cuda/cuda_gdn.cu:169`).

**A replica is required, because upstream's own kernel cannot be run at higher
precision.** `chunk.py:213-215` asserts `q.dtype != torch.float32`, which is
why `tests/parity/test_op_parity.cpp:595` already forces `VT_GDN_CHUNKED=0` for
f32 on CUDA. The replica is the only way to hold the algorithm fixed and move
the dtypes.

**The replica is validated three ways, not asserted, and each check is a
committed script.** (a) Its bf16 helper is bit-identical to `torch.bfloat16`
over **15531 of 15531** values — a 12-decade magnitude sweep, 5000 EXACT ties
(low half `0x8000`, the only inputs where round-half-to-even differs from
round-half-away), and the structural corners including subnormals, the bf16
max normal, infinities and NaN. `check_bf16_helper.py` in this directory
performs it, and it is the one script here that imports `torch`. (b) At
`float64` with every rounding site disabled the replica reproduces the
sequential recurrence to `2.43e-17`, so it is the same function and not an
approximation of one. (c) Run with upstream's dtype placement it reproduces the
published `1.15e-08` and `2.29e-04` / `2.25e-03` figures that #2612 reports,
from an independent implementation.

## 2. The decomposition

`max|diff|` against the exact `float64` sequential recurrence, on the golden's
own inputs. Four chunks processed (asserted: 2 sequences x 2 heads x 1 chunk).

| arm | max\|d\| out | max\|d\| state | share of the gap |
|---|---|---|---|
| exact f64 sequential (the reference) | 0 | 0 | — |
| **chunked, f64 intermediates** | **2.428613e-17** | **1.110223e-16** | **1.1e-11 %** |
| chunked, f32 intermediates | 3.855455e-08 | 1.144919e-07 | 0.017 % |
| sequential, f32 (our CPU arm) | 1.150589e-08 | 5.079557e-08 | — |
| **chunked, upstream's bf16 placement** | **2.286426e-04** | **2.175109e-03** | **~100 %** |
| vLLM's real kernel (`out.npy`) | 2.286426e-04 | 2.248661e-03 | — |

**Read the second row first.** The chunked WY decomposition is an algebraic
identity, and at f64 it behaves like one. It is not an approximation that
chunking introduces; it is the same recurrence evaluated in a different order.

**Read the last two rows together.** The replica, given nothing but upstream's
bf16 placement, lands at `2.286426e-04` — the same seven figures as the real
Triton kernel's own distance from exact. Nothing else was fitted.

**Carried at f32 working precision the reassociation costs 3.4x**, `3.86e-08`
against the sequential arm's `1.15e-08`. Both are four orders of magnitude below
the bf16 term. The same ratio holds on synthetic inputs at `T = 5, 64, 256,
1024` and `Dk=Dv=128`: the bf16 term is 5,500x to 37,000x the reassociation term
(`run_final.py`, in this directory).

## 3. Which bf16 sites carry it

Each site turned off alone, and each site turned on alone, against the exact
reference. `max|diff|`.

| site (upstream anchor) | this site OFF | ONLY this site | on `out` |
|---|---|---|---|
| `solve_tril` out -> bf16 (`chunk.py:50`) | 2.2864e-04 | 2.6174e-06 | moves |
| `(beta*v)`, `(beta*eG*k)` -> bf16 (`wy_fast.py:92,114`) | 2.0735e-04 | 7.4423e-05 | moves |
| `u`, `w` stores -> bf16 (`wy_fast.py:94,116`) | 2.2864e-04 | 7.8918e-05 | moves |
| `h` chunk-start snapshot -> bf16 (`chunk_delta_h.py:352`) | 2.2864e-04 | 1.3397e-04 | moves |
| `h` operand of `w @ h^T` -> bf16 (`chunk_delta_h.py:178`) | 2.2864e-04 | 1.3131e-05 | moves |
| `v_new` store -> bf16 (`chunk_delta_h.py:206`) | 2.2864e-04 | 7.8918e-05 | moves |
| decayed `v_new` -> bf16 (`chunk_delta_h.py:274`) | 2.2864e-04 | 3.8555e-08 | state only |
| intra-chunk `A` -> bf16 (`chunk_o.py:137`) | 3.2218e-04 | 8.8539e-05 | moves |
| `o` store -> bf16 (`chunk_o.py:138`) | 2.2783e-04 | 2.0735e-04 | moves |

**No single site is the mechanism.** Removing any one leaves the gap essentially
intact, and every site alone is worth `1e-6` to `2e-4`. Two facts follow for a
port. First, **`chunk_delta_h.py:178` is a rounding site the existing record
does not name**: the running state is held in f32 registers and then cast down
to bf16 *as an operand* of `w @ h^T`. Second, **`chunk.py:50` is another**:
`solve_tril` returns `A^-1` in `k.dtype`, so the triangular inverse itself is
stored bf16 even though `A` is f32. Both are absent from the dtype table in
[`qwen4exp-cuda-prefill-divergence-20260902.md`](qwen4exp-cuda-prefill-divergence-20260902.md) §4.

The two sites that move only the state (`chunk_delta_h.py:274`) or mostly the
state (`wy_fast.py`, `v_new`) are why the state gap (`2.25e-03`) is an order
above the output gap: the state accumulates the WY rounding across the chunk and
never passes through a compensating output projection.

## 4. Where an f32-intermediate chunked port would sit

The decision this evidence exists to inform. `out` compared at the production
output dtype (bf16), 8192 elements.

| arm | max\|d\| vs vLLM's kernel | bf16 elements differing |
|---|---|---|
| vLLM's kernel (`out.npy`) | 0 | 0 / 8192 |
| chunked, upstream bf16 placement (replica) | 6.1035e-05 | **62 / 8192** |
| chunked, f32 intermediates | 2.4414e-04 | **4157 / 8192** |
| sequential f32 (our CPU arm today) | 2.2865e-04 | **4157 / 8192** |
| exact f64 sequential | 2.2864e-04 | 4157 / 8192 |

and pointwise, f32-intermediate chunked against our own sequential arm:
`max|d| = 2.0735e-04`, **2 of 8192** bf16 elements differing.

**A chunked CPU kernel with f32 intermediates is our current answer wearing
vLLM's algorithm.** It does not become a third answer; it fails to become the
second one. Modelling the bf16 placement takes disagreement with the real kernel
from 4157 elements to 62, a **67x** reduction, and nothing else in this study
moves that count at all.

The residual 62 elements are the part of the real kernel this replica does not
model: Triton's tile-level reduction order, its blocked `solve_tril` merge
against a dense inverse, and the exact `tl.dot` operand precision for `K K^T`.
A TF32 variant of that one dot was measured and is worse (`1.2207e-04` vs
`6.1035e-05`), which is consistent with the ieee path being the one that runs.

## 5. What this is NOT

n = 1 golden, 32 tokens, 2 heads, one chunk per sequence, one dim pair
(`Dk=Dv=128`). The multi-chunk regime is covered only by synthetic inputs
(`run_final.py`, `T` up to 1024), where the conclusion is the same but the
oracle is my own f64 reference rather than a dumped kernel. **No token gate**:
nothing here decodes anything, and §4 of
[`qwen4exp-cuda-prefill-divergence-20260902.md`](qwen4exp-cuda-prefill-divergence-20260902.md)
already measured that token agreement is not monotone in this distance. **No
speed number.** The replica is numpy, not a candidate implementation.

The residual `6.1e-05` between the replica and the real kernel is named in §4,
not diagnosed. It bounds how much of vLLM's output a bf16-faithful C++ port
could still miss for reasons that are not dtype placement.

## 6. Reproducing

CPU only, no GPU, ~50 s. From the repository root:

```sh
python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/check_bf16_helper.py
python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/run_golden.py
python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/run_golden_ablate.py
python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/run_prodview.py
python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/run_final.py
```

**`check_bf16_helper.py` is the only one that needs `torch`**; the other four
are numpy-only, which is why the torch-dependent check is a separate file rather
than a self-check inside `run_golden.py`. An earlier draft of this section said
`run_golden.py` needed `torch` for that check. It does not, it never imported
`torch`, and the bit-identity claim was consequently not reproducible from the
committed artifact. It now is.

The captured stdout of all five is `OUTPUT.txt` in that directory, measured on
`torch 2.11.0+cu130` / `numpy 2.3.5`. `run_final.py` also writes two benign
`RuntimeWarning: overflow encountered in exp` lines to stderr at `T = 1024`, from
`exp(G_i - G_j)` evaluated on the full matrix before `np.tril` discards the
upper triangle; stderr is not captured. `gdn_decomp.py` carries the algorithm,
with each rounding site keyed by the upstream line it models.
