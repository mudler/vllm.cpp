# `BACKEND-TENSTORRENT-GDN` — the GDN linear-attention op chain on Tenstorrent

Issue [#1715](https://github.com/mudler/vllm.cpp/issues/1715). Child of
`BACKEND-TENSTORRENT` ([backend-matrix.md](../backend-matrix.md)); sibling
pattern `BACKEND-TENSTORRENT-MISTRAL`. This row is the hard prerequisite for
Qwen3.8 on Tenstorrent, chosen at the 2026-08-22 planning pass.

## Now

`ACTIVE` (claimed 2026-08-22, helper, worktree `vllmcpp-tt-gdn`). W1 (`4d165d130`)
landed the prefill set (kL2Norm, kRmsNormGated, kCausalConv1dFwd, kGdnPrefill —
chunk_gated_delta_rule adapter, fresh-review PASS) and W2 (`337f6e07a`) landed
the decode set (kCausalConv1dUpdate, kGdnDecode both state_idx forms,
kGdnStateGather, kGdnStateScatter; device shadows; composed decode chosen over
chunked by 1.139 vs 3.164 ms/step measurement; steady-state traffic
h2d=0 d2h=0; NaN-hardened comparator), both fresh-review PASS; all eight ops
registered for kTENSTORRENT and op-gated vs the CPU f32 oracle (ambient
33/33 cases 2124/2124; leg0 32/33 with only the pre-existing #1696 red at
test:83); production-unreached pending the wiring row (see ## Owed, #1715 open).

## Scope

**In.** Native Tenstorrent kernels + `RegisterOp` wiring + op-level gates vs
the CPU f32 oracle, for the GDN op set the Qwen3.5-family forward issues in
its default bf16 configuration:

- **W1, prefill:** `kL2Norm` (`L2NormArgs`), `kRmsNormGated`
  (`RmsNormGatedArgs`), `kCausalConv1dFwd` (`CausalConv1dArgs`), and
  `kGdnPrefill` (`GdnArgs`).
- **W2, decode:** `kCausalConv1dUpdate`, `kGdnDecode` (both `state_idx`
  forms), `kGdnStateGather`, `kGdnStateScatter`.

Op contracts live at `src/vt/ops.cpp` (`CausalConv1dFwd:1823`,
`CausalConv1dUpdate:1844`, `L2Norm:1929`, `RmsNormGated:1941`, `GdnDecode:2291`,
`GdnStateGather:2466`, `GdnStateScatter:2484`, `GdnPrefill:1986`); the CPU f32
reference implementations in `src/vt/cpu/cpu_ops.cpp`
(`GdnPrefillKernel:1591`, `GdnDecodeKernel:1628` and their step helpers) are
the correctness oracle, per the `BACKEND-TENSTORRENT-RESIDUAL-GOLDEN`
precedent.

**Out.**

- Architecture allow-list registration for `Qwen3_5*` — the wiring row, owed
  below; it also owes the capacity decision (bf16 27B ≈ 53.8 GB does not fit
  the P150's GDDR; the largest proven fit is Mistral-7B bf16 ≈ 14.5 GB).
- `kGdnSpecDecode`, `kGdnPackedDecode` — spec decode on TT owes async sampled
  token readback first ([#1627](https://github.com/mudler/vllm.cpp/issues/1627)).
- MoE, quant arms, the VL half (MRoPE/DeepStack/vision tower).
- Hand-written Tensix kernels. This row composes ttnn ops, mirroring the E1
  decision (`BACKEND-TENSTORRENT`), except where `chunk_gated_delta_rule`
  already IS the kernel.

## Upstream chain

**There is no vLLM mirror for any TT kernel** (vLLM has no Tenstorrent
platform). The references, in order:

1. **Our own CPU f32 arm** is the correctness oracle for every op (the
   residual-golden pattern; the Mistral gate used the same doctrine with a
   transformers alternative-oracle at the e2e tier, POL-ORACLE deviation
   recorded).
2. **tt-metal at our pinned checkout** is the implementation substrate. The
   load-bearing anchor: `ttnn::transformer::chunk_gated_delta_rule`
   (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.hpp`)
   — standalone FLA chunked GDN forward, one Tensix core per (B·HV) head,
   recurrent state held on-core, `initial_state`/`final_state` optional,
   `use_qk_l2norm` flag, documented to match FLA `naive_chunk_gated_delta_rule`
   numerics (fp32/HiFi4). It maps onto `kGdnPrefill` with the adapter duties
   below. Everything else composes from the ttnn ops `tenstorrent_ops.cpp`
   already uses (matmul, eltwise, rms_norm, slice/concat/permute).
3. The **CUDA arm** (`src/vt/cuda/`) is the behavior mirror for op semantics
   (dtypes, layout, the fused variants' numerics), never the TT design.

## Our baseline

What exists today, measured against what this row needs:

- **TT op set** (`src/vt/tenstorrent/tenstorrent_ops.cpp`, registration block
  `:3165-3199`): 20 ops — matmul/matmulBT, add, relu, embedding, layernorm,
  rmsnorm (incl. fused chain), silu-and-mul, casts, three rope forms,
  qkv-split, reshape-and-cache, paged attention (decode+prefill), greedy
  argmax. ZERO GDN-family ops. Three archs served
  (`src/vllm/platforms/tenstorrent.cpp:55`), all dense standard-attention.
- **The CPU f32 reference** (`src/vt/cpu/cpu_ops.cpp:1537-1740`):
  `GdnPrefillKernel` / `GdnDecodeKernel` and their step helpers — the
  correctness oracle for every kernel this row adds. Pre-normalized q/k,
  `scale` on q, state `[N,Hv,Dv,Dk]`, varlen via `query_start_loc`.
- **The tt-metal substrate at our pin**: `ttnn::transformer::
  chunk_gated_delta_rule` — standalone FLA chunked GDN forward, one Tensix
  core per (B·HV) head, recurrent state on-core, `initial_state` /
  `final_state` optional, `use_qk_l2norm` flag, documented to match FLA
  `naive_chunk_gated_delta_rule` numerics (fp32/HiFi4). Not yet called from
  our tree.
- **Dispatch reality**: the P150 is discrete (`UnifiedMemory() == false`),
  so an op miss refuses BY NAME (`src/vt/op_provider.cpp` `Resolve`; SAFETY
  at `include/vt/op_provider.h:206`). No free CPU fallback — the ops must
  land before any `Qwen3_5*` registration could work.

## Port map

Op-by-op mapping onto ttnn. vLLM defines none of these for TT (no
  Tenstorrent platform upstream); the CPU arm is the semantic contract and
  the behavior mirror is the CUDA arm where one exists.

| Op | Contract anchor | TT mapping |
|---|---|---|
| `kGdnPrefill` | `ops.cpp:1986`, CPU `:1591` | `chunk_gated_delta_rule` per sequence (or padded dense batch — measured choice): `use_qk_l2norm=false` (caller pre-normalizes), `scale` passed through, state permute `[N,Hv,Dv,Dk]` ↔ `[B,HV,K,V]` inside the adapter |
| `kGdnDecode` | `ops.cpp:2291`, CPU `:1628` | rank-1 step composed from `ttnn::matmul` + eltwise over `[B·Hv,Dv,Dk]`; state in a device shadow keyed by host pointer (`PagedKvShadow` pattern, `tenstorrent_ops.cpp:395`); both `state_idx` forms |
| `kCausalConv1dFwd` | `ops.cpp:1823` | slice/concat windows + eltwise MAC; conv state device shadow; host-staged fallback allowed in W1 if composition regresses |
| `kCausalConv1dUpdate` | `ops.cpp:1844` | T=1 shift+MAC eltwise on the conv shadow |
| `kL2Norm` | `ops.cpp:1929` | per-row L2 from existing TT norm machinery |
| `kRmsNormGated` | `ops.cpp:1941` | TT `RmsNormKernel` machinery + silu(gate) eltwise pass (sigmoid behind the arg) |
| `kGdnStateGather` | `ops.cpp:2466` | device row-gather over the state shadow |
| `kGdnStateScatter` | `ops.cpp:2484` | device row-scatter into the state shadow |

Design notes that cut across the table: the decode state shadow is the
  whole point on a discrete card (no per-token PCIe round-trip of the
  state); registering the decode set makes `IndexedGdnOpsNative`
  (`src/vllm/model_executor/models/qwen3_5.cpp:3322`) answer TRUE for TT,
  which is intended and must be gated in the indexed form explicitly;
  `kRmsNormGatedQuantFp8` is CUDA-only and out of scope (TT
  `supported_dtypes()` = {BF16, F32}).

## Design (adapter decisions the port map does not carry)

- **`kGdnPrefill` adapter.** Our contract is varlen: `q/k [T,Hk,Dk]`,
  `v/out [T,Hv,Dv]`, `g [T,Hv]`, `beta [T,Hv]`, `state [N,Hv,Dv,Dk]`,
  `query_start_loc [N+1]`; q/k arrive PRE-NORMALIZED and `GdnArgs::scale`
  applies to q (CPU `GdnHeadTokenStep` — scale multiplies q at load; there is
  no l2norm inside the op), so the tt-metal `use_qk_l2norm` stays FALSE and
  `scale` is passed through (FLA `scale` semantics match: applied to q in the
  attention). The tt-metal op takes dense `[B,T,...]` batches: the adapter
  either calls per sequence (T_b from qsl; sequential over B, state slices
  carried as `initial_state`/`final_state`) or pads to a dense batch — pick by
  measurement, record the choice. State layout differs in the last two dims:
  ours `[N,Hv,Dv,Dk]`, tt-metal `[B,HV,K,V]` — one permute on the upload and
  one on the download, inside the adapter, invisible to the caller.
- **`kGdnDecode`.** The step is rank-1: decay state by `exp(g)`, `dot =
  S·k`, `v' = (v − dot)·beta`, `S += outer(v',k)`, `o = S·(q·scale)`. At T=1
  compose from `ttnn::matmul` + eltwise ops over `[B·Hv, Dv, Dk]` slices —
  measured first against one T=1 `chunk_gated_delta_rule` call; take the
  faster, keep the other behind an env diagnostic if the gap is interesting.
  State must live in a **device shadow** keyed by the host pointer, exactly
  the `PagedKvShadow` pattern (`tenstorrent_ops.cpp:395`), so decode does not
  round-trip the state each token — on a discrete PCIe card that is the whole
  point of a native kernel. The `state_idx` form indexes into the FULL cache
  rows; mirror `NotePagedKvRacWrites`-style bookkeeping so host mirrors stay
  valid for `EnsureHost` consumers.
- **`kCausalConv1dFwd`/`kCausalConv1dUpdate`.** Depthwise causal conv over
  time with a rolling `conv_state` (pad+shift windows). Compose from
  slice/concat + eltwise multiply-accumulate, or host-stage in W1 if the
  composition regresses load-time; the conv state also wants a device shadow
  with the same keyed-pointer discipline.
- **`kL2Norm`, `kRmsNormGated`.** `kL2Norm` is a per-row L2 with eps;
  `kRmsNormGated` is RMS × `silu(gate)` (sigmoid variant behind the arg).
  Compose from the existing TT `RmsNormKernel` machinery plus an eltwise gate
  pass. The `kRmsNormGatedQuantFp8` fused variant is CUDA-only and NOT in
  scope (TT `supported_dtypes()` = {BF16, F32}).
- **Registration.** The `RegisterOp` block at
  `src/vt/tenstorrent/tenstorrent_ops.cpp:3165-3199` gains the eight ops.
  Nothing else in the backend changes.
- **Host-free decode interplay.** Registering the four decode/index ops makes
  `IndexedGdnOpsNative` (`src/vllm/model_executor/models/qwen3_5.cpp:3322`)
  answer TRUE for TT — on the day the arch row lands, the indexed state-I/O
  path activates. That is the intended behavior, and the op-level gates must
  cover the indexed (`state_idx != nullptr`) form explicitly so the activation
  is not a leap of faith.

- **Registration.** The `RegisterOp` block at
  `src/vt/tenstorrent/tenstorrent_ops.cpp:3165-3199` gains the eight ops.
  Nothing else in the backend changes.
- **Host-free decode interplay.** See the port-map note on
  `IndexedGdnOpsNative`; the indexed (`state_idx != nullptr`) form is gated
  explicitly so the future activation is not a leap of faith.

## Tests to port

There is no upstream TT test to port (no vLLM mirror). The gates are
op-level doctest cases in `tests/vt/test_tenstorrent_backend.cpp`, on the
existing pattern (random f32 inputs, CPU f32 kernel and TT kernel run the
identical op, compare), extended for GDN:

- Per op: shape sweeps including the Qwen3.8 geometry (Hv:Hk 4:1 GQA form,
  Dk 128, Dv 128), chunked T incl. non-multiples of the tt-metal chunk
  size, N>1 ragged `qsl`, empty sequences.
- `kGdnPrefill`↔`kGdnDecode` state round-trip: prefill's final state must
  equal decode's state after replaying the same tokens one at a time (both
  arms), pinning cross-kernel recurrence consistency.
- `kGdnDecode` indexed form vs the row-copy reference; gather/scatter
  inverse property on live slots.
- Numerics doctrine: f32-in-f32-out tight tolerance where the TT path
  computes in f32; bf16-tile tolerance (relative, absolute floor) where the
  device path runs bf16 tiles — calibrated on the `RESIDUAL-GOLDEN`
  measurement (0.0459 abs at rows≥32), stated per T, not global, because a
  recurrence amplifies rounding.
- Red-first mutations per asserted guarantee (drop the decay term; drop
  beta; swap the state dims in the permute) — each must turn the suite RED.
- Inertness: the pre-existing 835-assertion suite unchanged; with the new
  ops unregistered, nothing else moves.

## Gates

1. Focused: the new doctest cases green on the P150 (`thalia`), both ambient
   legs (`VT_TT_HOST_FREE_DECODE` unset and `=0`).
2. Full TT suite green; CPU gate green on the shared checkout tier.
3. Mutation evidence recorded per asserted guarantee (the fresh reviewer
   re-runs them).
4. `scripts/agent-preflight.sh` all-green on the row branch.

## Dependencies

- The pinned tt-metal checkout must build and expose
  `ttnn::transformer::chunk_gated_delta_rule` (it does at the current
  `TT_METAL_HOME`; the row records the exact revision in `## Evidence` when
  W1 lands).
- The CPU f32 reference stays UNTOUCHED — it is the oracle; any need to edit
  `src/vt/cpu/cpu_ops.cpp` is out of scope and a stop condition.
- Real Blackhole P150 (`thalia`) under the file mutex
  `${GPU_LOCK:-$HOME/gpu.lock}` (local board, not an `rc` fleet device).
- Independent of, but ordered before: the wiring row (`Qwen3_5*` arch
  registration + capacity decision), which cannot gate until these ops exist.
- Independent of #1625 (capture hang) and #1627 (async readback); this row
  neither requires nor fixes either.

## Work breakdown

Two non-overlapping waves, one branch, spec-first:

- **W1 — prefill set.** `kL2Norm`, `kRmsNormGated`, `kCausalConv1dFwd`,
  `kGdnPrefill` (+ the `chunk_gated_delta_rule` adapter and the state
  permute). Gates: the prefill op cases, the T-sweep tolerance table, the
  adapter's composition measurement (per-sequence vs padded).
- **W2 — decode set.** `kCausalConv1dUpdate`, `kGdnDecode` (both `state_idx`
  forms), `kGdnStateGather`, `kGdnStateScatter`, the device state/conv
  shadows, the prefill↔decode round-trip gate, the indexed-path gate.

Each wave lands its own focused gate green before the next starts; the full
TT suite + CPU gate + preflight run at the end of the row, and the fresh
reviewer re-runs the mutations.

## Evidence

### W1 — the prefill op set (recorded 2026-08-22, P150)

**Substrate.** Built and gated against the pinned tt-metal checkout
`a3d330289752192754277638fe5c09eb2fb49763` (2026-05-20, "dispatch: prefetch
src 2KB ahead in memcpy_to_device non-temporal path"), which exposes
`ttnn::transformer::chunk_gated_delta_rule`.

**Tolerance table** (deterministic LCG inputs, CPU f32 oracle, log
`/tmp/w1_baseline.log`: 4 new cases, 102 assertions, all green). The gate is
an absolute envelope (`rel=0.0`); the `max_rel` column is diagnostic and is
inflated by near-zero oracle entries.

| Op | Path | Envelope | Worst `max_abs` over the sweep | Worst `max_rel` |
|---|---|---|---|---|
| `kL2Norm` (T 3–200, H 2/8, D 128) | device, bf16 tiles | 0.02 | 1.43e-3 | 0.0110 |
| `kRmsNormGated` (T 3–200, Hv 2/8, silu+sigmoid, padded gate stride) | device, bf16 tiles | 0.035 | 2.63e-2 | 0.0267 |
| `kCausalConv1dFwd` (T 3–200 + ragged N=3, silu on/off) | host-staged f32 | 1e-4 rel + 1e-5 abs | **0.0** (bit-exact) | 0.0 |
| `kGdnPrefill` out (T 3–200 + ragged N=3, GQA 4:1 and 1:1) | device, bf16 q/k/v + fp32 state | 0.05 (T≤64), 0.08 (T>64) | 1.53e-3 (T=200) | 0.405 |
| `kGdnPrefill` final_state | same | 0.05 | 1.44e-2 (T=200) | 3.10 |

Per-T `kGdnPrefill` out `max_abs`: 3.99e-5 → 5.73e-5 (T=3), 3.38e-4 (T=64),
4.22e-4 (T=65), 1.53e-3 (T=200). The error grows **sub-linearly** in T
(≈27× error for 67× tokens), so the recurrence-amplification risk did not
trigger; no stop condition fired.

**Mutation evidence** (each: one named mutation, focused case RED, restored
byte-identical — `sha256` of `tenstorrent_ops.cpp` after each restoration
`798db19956b1fa52ff342a6a3a70c4e62d3d559c5dbe5a7e13e599ea6b12870a`, equal to
the pre-mutation hash; `git diff` vs `21e27c3d8` still the W1 +387/−14 diff):

| # | Kernel | Mutation | Result | Log |
|---|---|---|---|---|
| 1 | `kL2Norm` | omit the `rsqrt` (`inv = denom`) | RED, `test_tenstorrent_backend.cpp:1680` (9 shapes), exit 1 | `/tmp/w1_mut1.log` |
| 2 | `kRmsNormGated` | drop the silu/sigmoid activation (raw gate multiplies) | RED, `:1759` (16 shapes), exit 1 | `/tmp/w1_mut2.log` |
| 3 | `kCausalConv1dFwd` | delete the `conv_state` tail-writeback loop | RED, `:1852` **only** on the `conv_state` check; the out check (`:1848`) stayed green — out matched, state diverged | `/tmp/w1_mut3.log` |
| 4 | `kGdnPrefill` | drop `scale` (`chunk_gated_delta_rule` called with `1.0f`) | RED, `:1965` **only** on the out check; `final_state` (`:1971`) stayed green — the state update is scale-free, as the algebra predicts | `/tmp/w1_mut4.log` |

Mutation 4 first failed to compile (`-Werror=unused-parameter`: `scale` was
`args`' only use); it was re-applied with `(void)args;` and then ran RED. The
first-attempt build failure is not a mutation result.

**Both legs on the final restored tree** (exit 139 arrives after each green/red
summary — the known #1486 teardown, counted as the summary it follows):

- Ambient (`VT_TT_HOST_FREE_DECODE` unset): 27/27 cases, 936/936 assertions,
  SUCCESS (`/tmp/w1_leg_ambient.log`).
- `VT_TT_HOST_FREE_DECODE=0`: 26/27 cases, 935/936 assertions
  (`/tmp/w1_leg0.log`). The single red is `test_tenstorrent_backend.cpp:83`
  `CHECK(support_static_graph_mode())` — the **pre-existing** #1696 latent red,
  fixed by the unmerged PR #1699 (commit `9d61dc436`); base `1db7e59cf` carries
  the old 3-arm test and the W1 diff does not touch lines before 1599. Not a
  regression of this row.

**W1 composition choices** (as implemented, `tenstorrent_ops.cpp:3182-3575`):

- `kL2Norm` — device-composed: square → row sum → +eps → `rsqrt` → broadcast
  multiply, bf16 tiles, on the `DeviceRows` residency path (a rank-3
  `[T,H,D]` view reuses a resident same-numel device shadow).
- `kRmsNormGated` — device-composed: `ttnn::rms_norm` (the `kRmsNorm`
  machinery, `[1,D]` affine upload) multiplied by an uploaded silu-or-sigmoid
  gate pass; the padded-row rank-3 gate view is gathered on the host honoring
  the token stride before upload.
- `kCausalConv1dFwd` — **host-staged in W1**, on the port map's explicit
  allowance. The backend's `Alloc` is host memory, so the scalar port runs the
  oracle's exact instruction order on the host bytes, and the measurement above
  is bit-exact (`max_abs = 0`). A composed slice/concat+MAC path pays a full
  `[T*K,C]` window materialization to build what this loop streams; the
  conv-state device shadow that would flip that trade is W2's decode work.
- `kGdnPrefill` — the fused kernel `ttnn::transformer::chunk_gated_delta_rule`
  itself, **not** a primitive composition (the spec's "except where
  `chunk_gated_delta_rule` already IS the kernel"). Adapter: varlen → ONE dense
  padded `[N,L,…]` batch, not per-sequence calls — per-sequence initial states
  ride the batch `[B,HV,K,V]` `initial_state`, so N sequences cost one op call
  (the spec's per-sequence-loop risk is void by construction); L is padded to
  the 64-token chunk multiple so the op's own time-pad path stays idle, and
  zero q/k/v/g/beta padding rows are an identity state update (`exp(0)=1`,
  `v'=0`), so empty sequences and short tails leave the state exactly where
  the oracle leaves it. `use_qk_l2norm=false` (the caller pre-normalizes; the
  op fatal-errors on true), `scale` passed through (FLA semantics match the
  CPU `GdnHeadTokenStep`), state trailing-dims transpose
  `[N,Hv,Dv,Dk] ↔ [B,HV,K,V]` inside the adapter, Dk/Dv multiple-of-32 tile
  constraint refused by name.
- **Adapter composition note.** The padded-vs-per-sequence choice was made on
  structure (one kernel launch for the whole varlen batch; the padded form
  strictly subsumes the N=1 per-sequence form), not on a B>1 timing A/B; a
  batched timing comparison stays open for W2, where the decode shadow makes
  prefill↔decode state residency the measured quantity.
- The state-shadow traffic claim is **not asserted in W1**: prefill is one op
  call per batch, so there is no per-token traffic to count. The claim belongs
  to W2's decode shadows, per Gates.

Still owed as waves land: the W2 decode-set equivalents of every table above,
and the state-shadow traffic measurement.

### W2 — the decode op set (recorded 2026-08-23, P150)

**Substrate.** Same pinned tt-metal checkout and build/run recipe as W1.
Focused gate `-tc="*CausalConv1dUpdate*,*GdnDecode*,*round-trip*,*StateGather*,*edge shapes*"`:
6 cases, 1254 assertions, all green (`/tmp/w2_focus8.log`; the exit 139 after
the green summary is the known #1486 teardown).

**Red first.** `/tmp/w2_red.log`: before the kernels registered, every W2 op
refused by name (`kCausalConv1dUpdate` / `kGdnDecode` / `kGdnStateGather` /
`kGdnStateScatter` → "no provider for op").

**Movement exactness substrate** (a one-shot micro-probe, host truth in
double, removed before commit; logs `/tmp/w2_micro*.log`). At this pin ttnn
is byte-exact for: row (dim-0) slice/concat at any row count, dim-1
slice/concat at TILE-aligned widths, 2D transpose, reshape across TILE
boundaries, dim-0 gather at any index value, `indexed_fill`, and exact 0/1
mask multiplies. It is f32-rounding (~1 ulp) for multiply/add/silu, and
BROKEN — wrong data, not rounding — for reduce-sum (tf32 partials, 7.6e-3),
batched f32 matmul (3.5e-3), and ANY last-dim slice/concat at SUB-TILE widths
(~2.0 error) or in 3D. The conv kernel therefore keeps a transposed
time-major shadow and accumulates the width taps as sequential single-row
adds, which reproduces the oracle's MAC order bit-exactly
(`cpu_ops.cpp:1370-1372`).

**Tolerance table** (CPU f32 oracle, absolute envelopes):

| Op | Path | Envelope | Worst `max_abs` |
|---|---|---|---|
| `kCausalConv1dUpdate` (B 1/3 × silu × bias; indexed, NULL, widened, fwd-continuation) | device, transposed shadow, exact moves | 1e-4 rel + 1e-5 abs | out 1.49e-8 (silu rounding only); **state 0.0 bit-exact in every arm** |
| `kGdnDecode` (B 1/3 × GQA 2:8 / 2:2, both `state_idx` forms, NULL slot) | device, composed matmul+eltwise | 0.02 | out 1.85e-5, state 1.77e-4 |
| round-trip `kGdnPrefill`↔`kGdnDecode` (T 3/64/65/200, plus T=65 Hv=2) | both device arms | 0.05 | decode-vs-CPU 1.82e-3 (T=64); cross-kernel 5.55e-3; the CPU arm is bit-exact 0 |
| `kGdnStateGather`/`kGdnStateScatter` (6 slot/NULL configs) | device, exact gather + `indexed_fill` | **0 (bit-exact)** | 0.0 for working, cache, and untouched rows |

**State-shadow traffic** (the counted gate; persistent device state buffer,
`GdnShadowTraffic` counters in `tenstorrent_device.h`): `kGdnDecode` 3 chained
steps → `h2d == 1× state bytes, d2h == 0`; `kCausalConv1dUpdate` 3 chained
steps → `h2d == 1× cache bytes, d2h == 0`; round-trip replay T=3..200 →
`steps == T, h2d == 1× state bytes, d2h == 0`; the decode microbench's timed
window (50 steps after 5 warmup) → `h2d == 0, d2h == 0` — at steady state not
one state byte crosses PCIe per token.

**Composition measurement** (decode-shaped B=8, GQA Hk=2→Hv=8, Dk=Dv=128,
opt-in `TT_GDN_BENCH=1` microbench, 50 timed steps): composed rank-1 step
**1.139 ms/step** vs one T=1 `chunk_gated_delta_rule` call **3.164 ms/step**
— 2.78× — because the fused op pads T=1 to its 64-token chunk internally.
Composed stays the default; the fused call remains the
`VT_TT_GDN_DECODE=chunked` diagnostic. Logs `/tmp/w2_bench_composed2.log`,
`/tmp/w2_bench_chunked2.log`.

**Mutation evidence** (each: one named mutation, focused case RED, then the
ops file restored byte-identical — sha256
`3af6aa477021dfa6fa67f8050071c981d4f5044a30afeacf6e219fc38b8c1183` before and
after the battery):

| # | Kernel | Mutation | Result | Log |
|---|---|---|---|---|
| 1 | `kCausalConv1dUpdate` | gather-index cache key `kind` 1→0 (the roll reuses the mac index) | RED, 11/46 assertions: `conv_state` across every B/silu/bias shape + a TT_THROW | `/tmp/w2_m1_conv_roll_kindtag.log` |
| 2 | `kGdnDecode` | `args.scale`→`1.0f` in the composed step | RED, 6 errors ALL on out (`d.within`×5, `dn.within`); every state check GREEN — the state update is scale-free, as the algebra predicts | `/tmp/w2_m2_decode_scale.log` |
| 3 | `kGdnStateGather` | keep-mask polarity inverted (`live?0:1`) | RED, working+cache (`:2898`/`:2900`) | `/tmp/w2_m3_gather_keepmask.log` |
| 4 | `kGdnStateScatter` | slot map `idx+1 mod slots` | RED, 6× cache `dc.within` — working rows land in wrong cache slots | `/tmp/w2_m4_scatter_slotmap.log` |
| 5 | comparator | NaN multiplied into `qs` (out path only) | RED, `std::isfinite(d.max_abs)` + `d.within` — the NaN-safe comparator catches NaN, where a plain `abs(got-want) > tol` would pass it | `/tmp/w2_m5_nan_comparator.log` |

Mutation 4 first failed to compile (`idxv` is const; the mutation wrote
through `auto&`) — a first-attempt build failure, not a mutation result (the
same class as W1's mutation 4); it was re-applied through a mutable copy and
then ran RED.

**Both legs on the final tree** (exit 139 after each summary is the known
#1486 teardown):

- Ambient (`VT_TT_HOST_FREE_DECODE` unset): 33/33 cases, 2124/2124
  assertions, SUCCESS (`/tmp/w2_leg_ambient.log`).
- `VT_TT_HOST_FREE_DECODE=0`: 32/33 cases, 2123/2124 assertions
  (`/tmp/w2_leg0.log`). The single red is `test_tenstorrent_backend.cpp:83`
  `support_static_graph_mode()` — the pre-existing #1696 latent red (fix
  #1699 unmerged), identical to the W1 baseline. Not a regression of this row.

## Risks

- **Recurrence amplification.** A GDN state carried over hundreds of tokens
  compounds rounding differences; a per-op tolerance that passes at T=64 may
  fail at T=4096. Mitigation: the T-sweep in the gates; if TT-vs-CPU grows
  super-linearly in T, that is a finding, not a tolerance to loosen — record
  and stop for adjudication.
- **bf16 vs f32 tiles.** The TT matmul path casts to bf16 tiles; GDN's decay
  (`exp(g)`) in bf16 loses precision fast. Where numerics demand it, force
  the f32 compute path for the state update even at a speed cost, and say so.
- **Per-sequence loop cost.** If the adapter calls `chunk_gated_delta_rule`
  once per sequence, a large-B prefill pays B kernel launches; measure before
  accepting, or pad.
- **tt-metal op constraints.** The op's own shape/alignment rules (chunk
  size, head counts, tile divisibility) may reject Qwen3.8 geometries; the
  adapter must refuse by name with the failing constraint, never silently
  reshape into a wrong answer.

## Stop conditions

- `chunk_gated_delta_rule` numerics diverge from the CPU f32 oracle beyond
  the per-T tolerance on inputs the adapter cannot influence (the op's own
  numerics, not our adapter's) — stop, record the measurement, return
  `NEEDS_DECISION` on the oracle tier (tt-metal as substrate vs composing
  the scan from primitives ourselves).
- The state-shadow traffic cannot be eliminated without a layout the caller
  cannot honor — stop and report rather than land a kernel that round-trips
  state per token (that would be a slower CPU).
- Anything that requires registering the `Qwen3_5*` archs to prove reach —
  that is the wiring row's job, not this row's.

## Owed

- **The wiring row:** allow-list registration for `Qwen3_5*` + the capacity
  decision (quant arm or nothing fits the P150) + the first e2e gate. This
  row's ops land production-unreached until it exists — named here, in the
  landing commit body, and in the PR body per `## Nothing lands dead`.
  Tracked by #1715 (stays open until the family runs).
- GDN ops under host-free capture (`VT_TT_DECODE_CAPTURE`): the shadows must
  be capture-compatible (fixed device buffers) — measured when the capture
  hang [#1625](https://github.com/mudler/vllm.cpp/issues/1625) is resolved,
  not before.

## Git integration

One pull request for spec and implementation (developer answer recorded
2026-08-22 in `.agents/developer-preferences.md`). Spec commits first; the
commit order proves spec-before-code.
