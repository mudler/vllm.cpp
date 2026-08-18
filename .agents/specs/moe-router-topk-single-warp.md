# MoE router top-k: one warp per token, zero barriers, byte-exact

**Issue:** [#378 — MoE router top-k is 1.9x slower than vLLM's `topkGating`
(12.98 vs 6.85 us/call) — and the "byte-exactness forbids it" conclusion was too
strong](https://github.com/mudler/vllm.cpp/issues/378)

**Row:** `KERNEL-MOE-ROUTING` — the existing kernel-matrix row that owns router
top-k, align, permute/unpermute and combine, whose current claim is
`CLAIM-MOE-DECODE-PARALLEL-1` (landed by `6a8c5cf9`). This work is a lever
*inside* that family, not a new kernel family, so it registers no new matrix
row. Work branch and role claim: `KERNEL-MOE-ROUTER-WARP`.

**Parity pin:** vLLM `555967922` (0.26.0.dev0), per
[upstream-sync.md](../upstream-sync.md)

**Lifecycle:** `READY`. This spec authorizes an implementation and a focused
correctness gate. It authorizes **no** performance claim, no default-flip
credit, and no benchmark record; the operator owns the A/B, the `ncu`
attribution and the token gate.

## 1. Scope

Add a second CUDA realization of the **ungrouped** MoE router top-k
(`vt::MoeRouterTopK`, `MoeScoringFunc` softmax, `num_expert_group == 0`) that
runs **one warp per token with the whole logit row in registers**: no shared
memory, no `__syncthreads()`, and one global read of the row instead of two.
It must be **bit-identical** to the incumbent `MoeRouterTopKKernel<Tin,false>`
on every input, including NaN, Inf, exact ties and `k > E`.

In scope:

- `MoeRouterTopKWarpKernel<Tin, VPT>` in `src/vt/cuda/cuda_moe.cu` for
  `VPT ∈ {1,2,4,8}`, i.e. `E ∈ {32,64,128,256}`.
- A portable, host-compilable header carrying the lane map, the two register
  reduction trees and the selector, so the numerics-critical part is testable
  without a GPU.
- Dispatch from `RouterDispatch`, default ON, with a same-binary rollback.
- Tests: the reduction-order equivalence proof (host, runnable anywhere) and
  extra CUDA parallel-vs-serial cases at the real gate shape.

Explicitly out of scope:

- The grouped-topk (`noaux_tc`) router. It is a separate kernel and is not
  touched, so DeepSeek-class routing stays byte-identical by construction.
- The `Serial` template path. It stays exactly as it is: it is the byte-exact
  oracle the parity test compares against, and a change there would invalidate
  the oracle rather than test the candidate.
- The CPU reference (`cpu_ops.cpp MoeRouterTopKKernel`).
- The two structurally larger prizes named in #378 and **not attempted here**:
  folding the router into the preceding gate-GEMM epilogue, and batching the
  40 per-step launches. Each needs its own row.
- Any arithmetic change. See §4.

## 2. The measured defect

From the 35B-A3B both-arms decode trace in #378
(`nvidia/Qwen3.6-35B-A3B-NVFP4`@`491c2f1e`, GB10, batch-1):

| kernel | ms/step | calls | us/call |
|---|---:|---:|---:|
| ours `vt::cuda::MoeRouterTopKKernel` | 0.5193 | 40 | 12.98 |
| vLLM `vllm::moe::topkGating<8,256,4,16,32,int,__nv_bfloat16,…>` | 0.2740 | 40 | 6.85 |

1.9x, ~0.245 ms/step, **~1.5% of GPU-busy**. Both numbers are **PROVISIONAL**:
the vLLM arm ran against the 0.25.0 rollback, not the recorded pin (#375). They
are the motivation for this row, not evidence for it.

The cost is structural, and re-verified against the current tree at
`6dbedf9f`. `src/vt/cuda/cuda_moe.cu:61-201`, launched at `:420-432` with
**one block of 256 threads per token** and `E * sizeof(float)` dynamic shared
memory. At `E=256`, one token costs:

| phase | anchor | `__syncthreads()` |
|---|---|---:|
| block-tree max over 256 lanes | `:73-79` | 1 + 8 + 1 |
| second global read of the logit row + `expf` | `:83-87` | — |
| block-tree sum over 256 lanes | `:89-95` | 1 + 8 + 1 |
| normalize | `:103` | 1 |
| k=8 rounds x (warp argmax + leader scan) | `:139-195` | 8 x 2 |
| **total** | | **37** |

plus 3 KiB of dynamic shared memory (`sp[E]`), 1 KiB `red[256]`, 1 KiB
`redi[256]`, and **two** full global reads of the row (`:71` and `:83`).

`topkGating` has **zero** barriers and **zero** shared memory: the row lives in
`float row_chunk[VPT]` registers
(`csrc/libtorch_stable/moe/topk_softmax_kernels.cu:352` @ `555967922`), the
reductions are `VLLM_SHFL_XOR_SYNC_WIDTH` butterflies (`:417-421`, `:433-437`),
and the top-k masks the winner in the owning lane's own register (`:566-577`).

## 3. Why the previous conclusion was too strong

`6a8c5cf9` parallelized the argmax and recorded:

> vLLM's 6.7us register-fused topkGating reorders the softmax reduction so it is
> off-limits under byte-exactness — only the serial argmax was removable

and `cuda_moe.cu:161-163` states the softmax trees are deliberately untouched.
That conclusion is correct **for vLLM's own lane partition** and false in
general, because it conflates "register-resident" with "reassociated".

vLLM's partition really does reassociate. At the traced instantiation
`topkGating<8,256,4,16,32,…>`: `ELTS_PER_LDG = 16/sizeof(bf16) = 8`,
`THREADS_PER_ROW = 256/8 = 32`, `LDG_PER_THREAD = 1`, and
`first_elt_read_by_thread = thread_group_idx * 8` (`:344-346`), so **lane `L`
owns the contiguous experts `[8L, 8L+8)`**. Summing eight contiguous experts in
a lane and then butterflying is a different association than our block tree, so
it lands on a different last ulp. That is a property of *that map*, not of
register residency.

**A different lane map is bit-exact.** In the block tree

```c
for (int s = kBlock / 2; s > 0; s /= 2) {
  if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
  __syncthreads();
}
```

levels `s = 128, 64, 32` all have `s ≡ 0 (mod 32)`, so `t` and `t + s` always
have the **same lane id** `L = t mod 32`. Those three levels therefore never
cross a lane: they combine exactly the entries `red[L + 32q]`, `q ∈ [0,8)`, and
they combine them by the standard halving recursion on `q`. Levels
`s = 16 … 1` operate entirely inside warp 0 and are what
`__shfl_down_sync(0xffffffffu, v, s)` reproduces.

So: **give lane `L` slot `q` the expert `L + 32q` and reduce the slots with the
same halving tree, then shuffle.** Identical float operations, identical
association, identical operands — bit-identical by structural congruence, not by
an appeal to associativity. Zero barriers and zero shared memory fall out.

Written in the leaf order of the tree, the offsets are
`32 * {0,4,2,6,1,5,3,7}`, the bit-reversal of `{0..7}` (#378's phrasing). The
stride form above is the same statement and is the one implemented, because it
is checkable by inspection against the incumbent loop.

The derivation is worked per `VPT` in §5. It is **not** assumed for any `VPT`
that is not shown there, and a `VPT` that cannot be shown exact is not
dispatched.

The top-k argmax needs none of this. It is a reduction over the total order
"higher value, then lower expert index", which is associative and commutative,
so any grouping yields the same `(value, index)` — the argument already recorded
at `cuda_moe.cu:156-163` and unchanged here.

## 4. Arithmetic held fixed — this is a SHAPE port, not a MATH port

The candidate keeps every one of the incumbent's arithmetic decisions. vLLM
does four of them differently, and porting any of them changes tokens:

| decision | ours (`cuda_moe.cu`) | vLLM @ `555967922` | verdict |
|---|---|---|---|
| normalize | `pj = sum > 0.f ? sp[j] / sum : 0.f` — a **divide** (`:99`) | `reciprocal_row_sum = 1.f/row_sum; row_chunk *= reciprocal` (`:445-451`) | KEEP OURS |
| `sum <= 0` guard | present (`:99`) | absent | KEEP OURS |
| renormalize | `weights[j] /= denom` (`:198`) | `scale = routed_scaling_factor; scale /= denom; output *= scale` (`:582-590`) | KEEP OURS |
| `denom <= 0` | `if (!(denom > 0.f)) denom = 1.f` (`:197`) | `denom = selected_sum > 0.f ? selected_sum : 1.f` (`:584`) | equivalent; keep our spelling |
| mask value | `-INFINITY` (`:189`) | `-10000.f`, and only `if (k_idx+1 < k)` (`:570-576`) | KEEP OURS |
| max seed | `m = -INFINITY` then fold (`:71`) — erases NaN | `thread_max = row_chunk[0]` then fold (`:410-414`) | KEEP OURS |
| NaN/Inf clamp | `if (!isfinite(pj)) pj = 0.f`, **after** normalize (`:100-101`) | `if (isnan||isinf) 0.f`, after normalize (`:465-471`) | already agrees |
| `best < 0` | `indices = -1`, `weights = -INFINITY` (`:189-191`) | `indices = NUM_EXPERTS` / `-1` for pad rows (`:559-561`) | KEEP OURS |

The renormalize row is independently load-bearing beyond this row: the pin at
`555967922` folds `routed_scaling_factor / denom` into one multiply where the
prior `e24d1b24` pin divided. Ours divides. **Advancing this op past the
0.25.0 arithmetic is a token-affecting event regardless of this row** — relevant
to #375 and to roadmap C10, and out of scope here.

## 5. The lane map, derived per dispatched VPT

Notation: `l[·]` are the logits of one token; `E_j = expf(l[j] - mx)`;
`R_j = fmaxf(-INFINITY, l[j])`; `⊔` is `fmaxf`; the block kernel's per-thread
seeds are `e[t]` (sum) and `m[t]` (max), `t ∈ [0,256)`.

For every dispatched `E ≤ 256 = blockDim`, the strided seed loops at `:71` and
`:83` give each thread **at most one** expert, so

```
m[t] = R_t  and  e[t] = (+0.0f) + E_t    for t < E
m[t] = -INFINITY, e[t] = +0.0f           for t >= E   (loop body never runs)
```

**The sum seed is `(+0.0f) + E_t`, not `E_t`** (corrected 2026-08-11; the
original text wrote `e[t] = E_t`). The incumbent is
`float acc = 0.0f; ... acc += ex` (`cuda_moe.cu:84-89`), so every occupied
thread's seed is an addition against `+0.0f`, and the candidate's per-lane tree
starts from the loaded value instead. `(+0.0f) + x` differs from `x` for exactly
two classes of `x`: `-0.0f`, which `expf` never produces (it never sets the sign
bit), and a **signalling** NaN, which the addition quiets and a bare copy does
not — and `expf` returns only quiet NaNs. So the two spellings coincide on every
value these seeds can hold. That is a **third instance of the same contained
class** as the two zero-leaf arguments below and in §8.1: an operation that is
formally distinguishable but whose distinguishing inputs are unreachable here.
It is called out rather than glossed because "unreachable" is the load-bearing
word, and a future change to what feeds the seed would have to re-establish it.

The three cross-warp levels give, for `t = L < 32`:

```
C(L) = ((e[L] + e[L+128]) + (e[L+64] + e[L+192]))
     + ((e[L+32] + e[L+160]) + (e[L+96] + e[L+224]))
```

**VPT = 8 (E = 256).** No padding; `e[L+32q] = E_{L+32q}` for all `q ∈ [0,8)`.
Halving tree on `v[q] = E_{L+32q}`:

```
s=4:  v0 = E_L      + E_L+128     v1 = E_L+32  + E_L+160
      v2 = E_L+64   + E_L+192     v3 = E_L+96  + E_L+224
s=2:  v0 = (E_L + E_L+128) + (E_L+64 + E_L+192)
      v1 = (E_L+32 + E_L+160) + (E_L+96 + E_L+224)
s=1:  v0 = v0 + v1
```

which is `C(L)` term for term, parenthesis for parenthesis. **EXACT.**

**VPT = 4 (E = 128).** `e[t] = +0.0f` for `t ∈ [128,256)`, so

```
C(L) = ((E_L + 0) + (E_L+64 + 0)) + ((E_L+32 + 0) + (E_L+96 + 0))
```

Dropping the zero leaves is bit-exact here because `x + (+0.0f) == x` for every
`x` these seeds can hold: `expf` returns `+0.0f` or a positive value and never
`-0.0f` (the one float for which `x + 0.0f` flips the sign bit), and `±INFINITY`
and NaN are absorbed unchanged. A NaN sum cannot leak either: it is clamped at
`:100`. So `C(L) = (E_L + E_{L+64}) + (E_{L+32} + E_{L+96})`, which is the
halving tree on `v[q] = E_{L+32q}`, `q ∈ [0,4)`. **EXACT.**

**VPT = 2 (E = 64).** `C(L) = ((E_L + 0) + (0 + 0)) + ((E_{L+32} + 0) + (0 + 0))
= E_L + E_{L+32}`, the halving tree on two slots. **EXACT.**

**VPT = 1 (E = 32).** `C(L) = E_L`, a zero-level tree. **EXACT.**

**Max.** Identical structure with `⊔` for `+`. The per-element seed
`R_j = fmaxf(-INFINITY, l[j])` is applied **before** the tree because that is
what the incumbent does (`float m = -INFINITY; ... m = fmaxf(m, Load(...))`,
`cuda_moe.cu:70-71`), and reproducing the incumbent verbatim is the whole method
of this row. **It stays.** Dropping the `-INFINITY` pad leaves for `VPT < 8` is
bit-exact because `x ⊔ -INFINITY == x` for every non-NaN `x` including `-0.0f`
(verified exhaustively over all 2^32 non-NaN `x` during review).

**Corrected 2026-08-11 — what the seed does, and does not, buy.** The original
text said the seed "is what erases NaN" and that reproducing it "is what makes
an all-NaN row behave identically". Review falsified the second half by
mutation: deleting the seed from `MoeRouterWarpTreeMax` fails **only** on the
`mx` intermediate; the weights and the indices stay byte-identical on every
case, `all-nan` included. `fmaxf` already returns the non-NaN operand, so the
tree erases a NaN whether or not the leaves were seeded; the seed changes the
result only when a lane holds *nothing but* NaN, and then only by making `mx`
`-INFINITY` instead of NaN — which `expf(l - mx)` washes out, because
`expf(NaN - anything)` is NaN either way, every prob is then clamped to `0.0f`
at `:100`, and the tie-break hands back `0,1,…,k-1` regardless.

So the seed's **only** guard in the gate is the `mx` intermediate comparison in
`tests/vt/test_moe_router_warp_map.cpp`, itself under the `SameIntermediate`
exemption of §8.1. That is a thin guard, and it is deliberately named as thin:
the reason the seed must not be removed is fidelity to the incumbent, not an
output the test can see.

**Warp levels.** `s = 16,8,4,2,1` combine `t` and `t+s` with `t < s < 32`, all
inside warp 0, and are reproduced by `v = op(v, __shfl_down_sync(0xffffffffu, v,
s))`. Lane 0's dependency cone stays within lanes `< 16` after the first step,
so the lanes whose `__shfl_down_sync` returns their own value never enter it —
exactly as the block tree leaves `red[t >= s]` stale and reads only `red[0]`.

**VPT > 8 is not dispatched.** At `E > 256` the seed loop at `:83` accumulates
several experts per thread in ascending order, so `e[t]` is itself an
association we would have to reproduce. Not derived, not dispatched.

## 6. Design

New header `src/vt/cuda/moe_router_warp.h`, host-compilable (the
`#if defined(__CUDACC__) __host__ __device__ #endif` idiom already used by
`src/vt/cuda/gdn_decode_fused.h:57`), carrying:

- `MoeRouterWarpExpert(lane, slot) -> lane + 32 * slot` — **the lane map**.
- `MoeRouterWarpTreeSum<VPT>` / `MoeRouterWarpTreeMax<VPT>` — the halving trees
  of §5, `MoeRouterWarpTreeMax` applying the `fmaxf(-INFINITY, ·)` seed first.
- `MoeRouterWarpValuesPerThread(e) -> VPT` for `e ∈ {32,64,128,256}`, else `0`.
- `MoeRouterWarpFlagIsOn(const char*)` — the selector, default ON, `"0"` off,
  mirroring `Fa2PrefillEnabled()` (`cuda_paged_attn.cu:2504-2507`).

Putting the map and the trees in a header is deliberate: it is the part that can
be wrong in a way no compiler catches, and it is the part a host test can
execute without a GPU.

`MoeRouterTopKWarpKernel<Tin, VPT>` in `cuda_moe.cu`:

- `dim3(32, kRouterWarpsPerCta)` with `kRouterWarpsPerCta = 4`, one token per
  warp, `grid = ceil(T / 4)` — the same geometry as
  `topkGating<…,WARPS_PER_CTA=4,…>` (`ROWS_PER_WARP = 1`, `ROWS_PER_CTA = 4`,
  `topk_softmax_kernels.cu:311-317`). The out-of-range exit is warp-uniform, so
  a launched warp always has all 32 lanes active and `0xffffffffu` masks are
  valid.
- One coalesced pass loads `p[q] = Load(lrow, MoeRouterWarpExpert(lane, q))`.
  Slot `q` reads the 32 consecutive experts `[32q, 32q+32)`, so every load is
  fully coalesced. This trades vLLM's single 16-byte vectorized `LDG` for `VPT`
  coalesced scalar loads; the bytes moved are unchanged, and it is still **one**
  read of the row against the incumbent's two.
- Max, `expf`, sum, normalize and the k rounds all run out of `p[]`.
- Winner masking is `if (best >= 0 && lane == (best & 31))` followed by an
  unrolled `for (q) if (q == slot) p[q] = -INFINITY;`. The unrolled compare is
  required: a runtime index into a per-thread array forces it to local memory and
  would spill the whole row, which is the entire point of the kernel.
- Lane 0 writes `weights`/`indices` and accumulates `denom` in `k` order, and
  applies the `denom <= 0 -> 1` guard and the final divide — the same statements
  as `:189-199`.

Dispatch in `LaunchRouter`: when `!serial`, the selector is on, and
`MoeRouterWarpValuesPerThread(e) != 0`, launch the warp kernel; otherwise fall
through to the **unchanged** block kernel. `RouterDispatch`'s grouped branch is
untouched, so `num_expert_group > 0` never reaches this code.

The selector is read fresh per launch (a `getenv` on a host path that runs once
per MoE layer per step, ~40 per 16 ms step) so an in-process test can flip it,
matching `Fa2PrefillEnabled()`. Under CUDA-graph capture the selector is read at
capture time and the graph bakes the chosen kernel, which is how every other
lever in this file behaves.

## 7. Risks

| risk | why it is contained |
|---|---|
| The lane map is wrong for some `VPT` | Derived per `VPT` in §5 and executed by the host equivalence test in §8; an underived `VPT` is not dispatched |
| A register array spills to local memory, erasing the win | Every index into `p[]` is a compile-time constant under `#pragma unroll`; the masking step is an unrolled compare, not a dynamic index |
| `__shfl_*_sync` on a partly-exited warp | The only early exit is `row >= T`, which is warp-uniform |
| The lever changes the grouped router | It is a different kernel behind a different branch of `RouterDispatch`; not edited |
| The lever changes the oracle | The `Serial` path is not edited; the parity test still compares against it |
| A NaN/Inf row behaves differently | The clamp, the `sum > 0` guard and the `-INFINITY` max seed are all reproduced verbatim; covered by a dedicated test case |
| It is fast but not byte-exact, and someone widens a tolerance | The parity test is `memcmp` on f32 weights and `==` on i32 indices. Widening it is forbidden; see §10 |
| It is byte-exact but not faster | Then it is a neutral change and takes no credit. §9 |

## 8. Tests

**RED-first, two vehicles, because the box that implements this has no `nvcc`
and no GPU.** That limitation is stated in the evidence, not worked around.

**(a) Host equivalence test** — `tests/vt/test_moe_router_warp_map.cpp`, runs
anywhere, registered in `tests/CMakeLists.txt` with
`${CMAKE_SOURCE_DIR}/src` on the include path (the pattern of
`test_gdn_decode_fused` at `tests/CMakeLists.txt:966-967`). It contains:

- a literal transcription of the incumbent block kernel (`cuda_moe.cu:61-201`)
  over a 256-entry array, including the pad seeds and the `t < s` update rule;
- a warp model that calls the **shipped** `MoeRouterWarpExpert` and
  `MoeRouterWarpTree{Sum,Max}` from the header, with `__shfl_down_sync` emulated
  by its defined semantics (lane `l` reads lane `l+off`, or itself when
  `l+off >= 32`);
- for `E ∈ {32,64,128,256}`, `k ∈ {1,8}` and `k > E`, `renormalize ∈ {0,1}`, and
  **both f32 and bf16-rounded logit arms**: `mx`, `sum`, the full weights vector
  by `memcmp`, and the indices vector must be identical;
- adversarial rows: exact-tie storms, all-NaN, all-Inf, mixed NaN/Inf, and
  `-0.0f`;
- a **permanent discrimination assertion**: the same warp model instantiated
  with the contiguous map `lane * VPT + slot` (vLLM's own partition) must
  produce a *different* weight bit pattern on at least one seeded random row.
  A test that cannot see that difference cannot see a reduction-order defect,
  and this pins the discriminating power in the gate instead of in a one-off
  transcript;
- the selector truth table.

The bf16 arm is not optional. A bf16 store has hidden a defect in this tree
before; f32 weights are compared bitwise here, but the *logits* arm still
matters because bf16 rounding manufactures the exact ties the tie-break must
resolve.

**(b) CUDA parallel-vs-serial test** — extending the existing case at
`tests/vt/test_ops_moe_grouped.cpp:503`. That case's comment at `:556` says
`// 35B routing shape: E=128, top-8`, which is **wrong**: the gate model is
`E=256` (`num_experts=256`, `num_experts_per_tok=8`; 40 MoE layers is the
observed 40 calls/step). E=256 was covered only by a hand-built near-tie
pattern. Fix the comment and add:

- random-logit `E=256` top-8, and an exact-tie storm at `E=256`;
- `E ∈ {32,64}` — every dispatched `VPT` gets device coverage;
- `k=1` and `k > E` (the `-1` sentinel with `weights = -INFINITY`) — **the
  `k > E` half is retracted for the device case by §8.2; the op rejects it**;
- an all-NaN row and an all-Inf row.

Land these against **current** code first; they only add coverage and must be
GREEN before the kernel exists. Then get the device RED by mutation: build the
warp kernel with the contiguous map and confirm the weights `memcmp` fails.

**Ordering.** (a) and the extra (b) cases go in green against current code.
Then the mutation red. Then the correct map, green.

### 8.1 Amendment from implementation: two IEEE-unspecified intermediates

Implementing (a) surfaced two facts the spec did not anticipate. Both are
recorded here because they change what the test may assert, and a reviewer must
be able to check the reasoning rather than take the relaxation on trust.

The **weights and indices comparison is unaffected** — it stays a strict, total
`memcmp` / `==`, and it holds at `-O0`, `-O1`, `-O2` and `-O3`. What moved is
only the two internal intermediates `mx` and `sum`, which the test pins purely
because they localize a defect faster than the output does.

1. **The NaN payload of `sum` is not an algorithm property.** IEEE-754 leaves
   the payload of `NaN + NaN` unspecified and a compiler may commute the
   operands of a float add freely. Measured here: `red[0] = red[0] + red[1]` and
   `n[0] = v[0] + v[1]`, over the identical operands `0x7fc00000` and
   `0xffc00000` and in the identical source-level operand order, produce
   `0x7fc00000` and `0xffc00000` — one loop got vectorized, the other did not.
   Unobservable in the output: `sum` is consumed only through the predicate
   `sum > 0.0f` (`cuda_moe.cu:99`), which is false for **every** NaN payload, so
   every probability takes the `: 0.0f` arm identically.
2. **The sign of a zero `mx` is a HOST CODEGEN artifact** (reworded 2026-08-11;
   the original text framed it as an unspecified IEEE tie that "the algorithm
   may resolve either way", which is wrong). The algorithm has no such freedom.
   Review checked exhaustively that `fmaxf(x, -INFINITY) == x` **bitwise** for
   all 2^32 non-NaN `x`, so the seed reproduces each leaf exactly; past the seed
   both models apply `fmaxf` to *identical pairs in identical order*. Whatever a
   conforming `fmaxf` does with the `(+0.0f, -0.0f)` tie — and C does leave that
   unspecified — it does the same thing on both sides, so the two transcriptions
   cannot disagree by any property of the algorithm.

   What actually disagrees is the **host compiler**: gcc vectorises one
   transcription and not the other, and the vector max instruction does not
   implement `fmaxf`'s zero handling. Hence the `-O`-dependence — the same
   source passes at `-O1` and fails at `-O0` and `-O2`. It is a fact about this
   test's two host loops, not about the kernels, and it says nothing about what
   the device will compute. Unobservable in the output either way: `mx` is
   consumed only as `expf(logit - mx)` (`cuda_moe.cu:84`); `x - (+0.0f)` and
   `x - (-0.0f)` differ for exactly one input, `x == -0.0f`, and `expf` maps
   both zeros to the same `1.0f`, so `sp[]` is bit-identical.

Neither is a tolerance and neither may be widened further. Both proofs are
asserted executably by the test case *"the intermediate exemptions cannot reach
the output"*; if either stops holding, the exemption is invalid and the
intermediate comparisons go back to a strict `memcmp` — they do **not** get
loosened again to accommodate a failure.

This also sharpens what the device gate in (b) must and must not compare: the
existing parallel-vs-serial case compares only weights and indices, which is
exactly right, and it must stay that way.

### 8.2 Amendment: `k > E` is not an input this op has

§8(b) above asked the device case to add "`k=1` and `k > E` (the `-1` sentinel
with `weights = -INFINITY`)". The `k > E` half of that was wrong, and it made
the device case RED: `vt::MoeRouterTopK` validates

```c
VT_CHECK(args.top_k >= 1 && args.top_k <= e,
         "moe_router_topk: top_k must be in [1, num_experts]");
```

(`src/vt/ops.cpp`) **before** dispatch, so the arm threw and aborted the case
partway — with every assertion that had already run still reporting *passed*,
exactly the `grep 'assertions:'` trap §9 gate 2 warns about. The reachable domain of
this op is `1 <= top_k <= num_experts`, on rank-2 contiguous same-device
tensors with `weights`/`indices` shaped `[T, top_k]`.

**The sentinel is dead through the public op, and that is a fact about the op,
not a gap in the test.** Within `1 <= k <= E` it cannot fire: after
`sum > 0 ? sp[j]/sum : 0` and the `isfinite` clamp every prob is finite and
`>= 0` (`cuda_moe.cu`), the argmax seeds `best_v = -INFINITY`, so each round
finds *some* unmasked expert; only masking all `E` of them leaves `best == -1`,
which needs a `(k+1)`-th round. `best < 0` is therefore a defensive guard.

Driving it from the device case was also unsafe. That case calls the oracle
`vt::cuda::MoeRouterTopKSerialCuda` **directly**, bypassing the validator, and
the `Serial` branch's `sp[best] = -INFINITY` carries no `best >= 0` guard — a
`k > E` round writes shared memory at `sp[-1]`, which gate 5's
`compute-sanitizer memcheck` would rightly flag. The oracle is not editable
under §1, so the fix is to stop feeding it an input the op forbids.

Resolution: the device case drops the `k > E` arm and instead pins the contract
with one `CHECK_THROWS_AS`, so the reason the sentinel is unreachable is
asserted rather than assumed. The **host** case keeps its `k > E` arm: it models
the two kernels directly, it is the only vehicle that can compare their
`best >= 0` guards, and its block transcription carries the guard so it never
indexes `-1`. `k=1`, `E ∈ {32,64,128,256}` and the all-NaN / all-Inf / mixed
rows are all inside the reachable domain and stay on device unchanged — the
validator constrains shapes, dtypes and `top_k`, never logit *values*.

### 8.3 Amendment from review: three gate-integrity repairs (2026-08-11)

An adversarial review of `be6a1f57` found no correctness defect in the kernel or
the lane map, and three defects in what the gate could *see*. All three are
fixed; none touches kernel arithmetic.

1. **The device case did not pin `VT_MOE_ROUTER_WARP`, so it could not say which
   kernel it had tested.** `MoeRouterWarpEnabled()` (`cuda_moe.cu`) is a fresh
   `getenv` per launch, and the parallel-vs-serial sweep neither set, cleared nor
   asserted the variable. With `0` in the ambient environment — which §9 gates
   6/7 *instruct* the operator to export, in the shell they then run `ctest` in —
   every `run()` exercised the block kernel, i.e. block-vs-serial, green since
   `6a8c5cf9`, reporting the identical `14 cases / 1137 assertions / SUCCESS`.
   Green could not distinguish "the warp kernel is byte-exact" from "the warp
   kernel never ran". The sweep is now hoisted into a lambda and run **twice**
   under `vt_test::ScopedMoeRouterWarp` (`tests/vt/moe_router_warp_env.h`),
   pinned `"1"` and `"0"`, each arm asserting the pinned state before it runs, so
   the candidate **and** the rollback are both gated and the ambient environment
   is irrelevant. **This doubles the case's assertion count by design** — gate 2
   reads a changed count as a red flag, and this is the change that explains it.
   The pin is a shared header rather than a per-file copy precisely so the
   portable companion, which runs with no GPU, executes the *same* pin the device
   gate uses.

2. **Both host models shared one argmax comparator**, so a comparator divergence
   between `cuda_moe.cu:169`/`:186` (block) and `:302` (warp) was invisible by
   construction: the models would have agreed because they ran the same code.
   Measured: dropping the `cur_i < 0 ||` sentinel clause from the shared helper
   left the suite at `7 cases / 3806 assertions / SUCCESS`. Each model now
   carries its own transcription (`BlockArgmaxTakes` / `WarpArgmaxTakes`), and a
   dedicated case pins them equal over every `(value, index)` pair that matters
   (`±INFINITY`, signed zeros, NaN, and the `-1` sentinel); the same mutation
   applied to the warp side alone is now RED.

3. **The discrimination case asserted `sum`-or-`weights`, not `weights`.** Its
   stated purpose is to prove the *weights* `memcmp` has discriminating power —
   the comparison the device gate makes — but the `||` was satisfiable by a
   contiguous map that moved the denominator without ever reaching the output.
   It now asserts on `weights` alone. The property holds: of 16 seeded rows,
   `2 / 2 / 6` differ in the weight bits at `VPT = 2 / 4 / 8`.

## 9. Gates

Correctness first. Nothing below authorizes a performance claim on its own.

1. `test_moe_router_warp_map` green, with the mutation red captured on the same
   binary.
2. `test_ops_moe_grouped` green on a real GPU, with its assertion count recorded
   — a *changed* count is a red flag, not a pass, and `grep 'assertions:'` alone
   has hidden thrown cases here before, so `Status` and the test-case counts are
   recorded too.
3. `test_ops_moe`, `test_ops_moe_router_grouped`, `test_ops_moe_grouped_bf16`
   unchanged.
4. Clean CUDA `-Werror` rebuild — not incremental; a header changes here.
5. `compute-sanitizer memcheck` clean on the router fixture.
6. The 35B-A3B token gate: greedy output byte-identical with the lever ON and
   with `VT_MOE_ROUTER_WARP=0`, **same binary**. Byte-exactness is the claim, so
   any divergence rejects the candidate outright rather than opening a
   tolerance discussion.
7. Only then: same-binary A/B on an idle GB10 with the lever the only
   difference, plus a same-tool `nsys` slice showing the router kernel's
   us/call, and an `ncu` attribution of where the remaining gap sits. Report the
   value and the ratio; a kernel-level win with no step-level movement is
   diagnostic evidence, not a default-flip credit.

The vLLM denominator must be **re-measured against the rebuilt pin** before any
number from #378 is recorded as established.

## 10. Stop conditions

- If the map cannot be shown byte-exact for some `E`, **narrow the dispatch** to
  the `VPT` that can and record which and why. Never re-ratify a golden to
  accommodate a reduction-order change, and never widen the `memcmp` — that is
  the developer's decision, and it is a `NEEDS_DECISION`, not an implementation
  choice.
- If a test outside the ones this spec adds goes red, stop and report.
- If the kernel is byte-exact but measures neutral or slower, record it as a
  falsified hypothesis in `## Outcome` with the regime it was measured in, keep
  the block kernel as the default, and do **not** declare a ceiling: the next
  named hypotheses are the gate-GEMM epilogue fold and batching the 40 launches
  (§1), and #378 sizes the whole prize at ~1.5% of GPU-busy, which does not
  reach parity alone.
- Anything requiring a public API change, an arithmetic change from §4, a pin
  advance, or a keyed-record edit beyond this row returns `NEEDS_DECISION`.

## 11. Evidence obligations

Record: the immutable SHA; the exact commands; the build type (a Release/NDEBUG
green over an assert-firing bug is a latent failure); the host, driver and
contention state; the assertion and test-case counts for every suite run; the
mutation transcript; and, separately, everything that could **not** be verified
on the implementing host. An implementer's report is an input; the operator
reruns the gate.

**Which golden gate is evidence for this row, and which is not.**
`test_qwen36_paged_engine` (35B-A3B, `num_experts=256`, `num_experts_per_tok=8`)
is **the** relevant model-level gate: E=256 is a dispatched width, the router is
ungrouped, so the warp kernel is on the path a greedy golden actually walks.
`test_qwen27_paged_engine` is **NEUTRAL** — 27B is dense and has no ungrouped
MoE router at all, so it exercises none of this code and cannot be cited as
evidence for this row in either direction. A green there is not a pass for
`KERNEL-MOE-ROUTER-WARP`, and a red there is somebody else's bug.

**Counts to expect after the §8.3 repairs.** `test_moe_router_warp_map` moves
from `7 cases / 3806 assertions` to `9 cases / 4597 assertions` (4598 when
`VT_MOE_ROUTER_WARP` is set in the ambient environment — the restore check
asserts one extra time when there is a prior value to restore).
`test_ops_moe_grouped`'s router case runs its sweep twice, so its assertion count
roughly doubles; the previously recorded `14 cases / 1137 assertions` is
superseded and a device re-run owes the new numbers.

## 12. First DEVICE verification (2026-08-12, GB10 `dgx.casa`)

Everything below is the first time this kernel has been **compiled or executed at
all**. `cce81c7e` shipped it unbuilt and unrun ("no nvcc, no GPU"), and the branch
never landed, so until now the row's only evidence was host arithmetic.

**Build.** Source `6c3be5c3` (this branch on main `bbc482a2`), clean tree.
`RelWithDebInfo`, pinned `/usr/local/cuda-13.0/bin/nvcc`,
`VLLM_CPP_CUDA_ARCHITECTURES=121a`, `VLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
`VLLM_CPP_TRITON=ON`, `VLLM_CPP_BENCH_PROFILE_CONTROL=OFF`. Configure log verified
to print `FlashAttention-2 prefill/decode: ENABLED for arch(es) [121a]` and the
`sm_121a` vendored Triton AOT lines. **The `.cu` compiles**; that was not known.

**Oracle identity, asserted by commit.** `~/work/vllm-src-5559679` HEAD =
`5559679229bc961848b121ccdeaa8fa5d79bec98` = the pin. The venv carrying it is
`~/venvs/vllm-oracle-next` (`0.23.1rc1.dev1511+g555967922` — a setuptools_scm
nearest-tag artifact, not a mismatch). **The shared `~/venvs/vllm-oracle` symlink
currently points at `vllm-oracle-v0.25.0-stage`, i.e. NOT the pin.** Left
untouched, since other sessions depend on it; recorded because any denominator
taken through that symlink is the 0.25.0 rollback — exactly the caveat #378 carries.

**Gates 1-3, 6 — all GREEN, one `flock $HOME/gpu.lock` held for the group.**

| gate | suite | cases | assertions | Status |
|---|---|---|---|---|
| 1 | `test_moe_router_warp_map` | 9 | 4597 | SUCCESS |
| 2 | `test_ops_moe_grouped` (self-pins both arms) | 14 | 1907 | SUCCESS |
| 2 | same, ambient `VT_MOE_ROUTER_WARP=0` | 14 | 1907 | SUCCESS |
| 3 | `test_ops_moe` | 9 | 33451 | SUCCESS |
| 3 | `test_ops_moe_router_grouped` | 14 | 3880 | SUCCESS |
| 3 | `test_ops_moe_grouped_bf16` | 7 | 19 | SUCCESS |
| 6 | `test_qwen36_paged_engine`, warp ON | 2 | 315 | SUCCESS |
| 6 | `test_qwen36_paged_engine`, `=0` rollback | 2 | 315 | SUCCESS |

Gate 1's predicted `9 / 4597` and gate 6's `315 / 315` land exactly on the counts
§11 predicted. Gate 2's 1907 is the post-§8.3 doubled count replacing the
superseded `14 / 1137`. Gate 1's mutation RED was re-proven on the same source:
replacing the halving tree with a left-linear fold fails **76 assertions across 2
cases** at `-O2` (§8.3 recorded 32 against the smaller pre-repair suite). The 35B
gate is confirmed to have actually RUN, not skipped — it loaded
`nvidia/Qwen3.6-35B-A3B-NVFP4@491c2f1e` and emitted its real continuation. Gate 4
(clean `-Werror` CUDA build) is satisfied by construction: a from-scratch tree.
**Gate 5 (`compute-sanitizer memcheck`) was NOT run and is owed.**

**Which kernel actually ran — proven by name, not inferred.** `nsys
--cuda-graph-trace=node`, same binary, both arms:

- warp arm: `MoeRouterTopKWarpKernel<__nv_bfloat16, (int)8>` — VPT=8, the E=256
  dispatch the gate model uses — 1280 calls, and **no** block kernel present.
- `=0` arm: `MoeRouterTopKKernel<__nv_bfloat16, (bool)0>`, 1280 calls, and **no**
  warp kernel present.

That closes the F1 hazard at the model level: the 315/315 is not a green that
could have come from either kernel.

**Kernel-level A/B — ESTABLISHED.** 3 reps/arm, order-alternated `on off off on
on off`, one lock, 1280 router calls in every leg:

| arm | mean | min | max | spread |
|---|---|---|---|---|
| warp | **10.422 ms** | 10.307 | 10.529 | 0.222 |
| block | **14.201 ms** | 14.141 | 14.275 | 0.134 |

**-3.779 ms, -26.6%, ratio 0.7339 (1.363x on the kernel)**, bands
non-overlapping. All six legs also passed the token gate, so correctness holds
across six further independent runs.

**Step-level movement — NOT SEPARABLE, and the window is why.** Total GPU-busy
read warp 1601.890 ms vs block 1619.395 ms (-17.505 ms, ratio 0.9892). **That
number is not attributable and is not claimed.** The kernels this change does not
touch — identical code, identical 155,310 launches every leg — differ by 13.726 ms
between arms, and their full spread across the six legs is 24.140 ms, i.e. **6.4x
the 3.779 ms effect**. An earlier single pair showed it more crudely still: an
unchanged `gemvx::kernel` moved 15.7 ms at zero call-count change. This
instrument's systematic error swamps the effect, so the apparent 1.011x is an
artifact of which legs landed where.

The deeper reason is the window: this is the whole paged-engine test, prefill and
decode aggregated, so the router's 40 calls/step are diluted across ~1600 ms.
Sized against the decode step instead — 40 calls/step, and #378's measured
16.1863 ms/step decode GPU-busy — the saving is **0.083-0.118 ms/step, i.e.
~0.5-0.75% of decode GPU-busy** (medians and means respectively). That is
**smaller than #378's ~1.5% estimate**, which assumed the whole 6.13 us/call gap
to `topkGating` was recoverable; the warp kernel closes roughly 79% of it
(12.98 -> 8.14 us/call mean against vLLM's 6.85), not all of it.

**Consequences, stated plainly.**

1. Per gate 7 this is a kernel-level win **without** demonstrated step-level
   movement, and therefore **not a default-flip credit**. The default is left as
   `cce81c7e` authored it (ON) because that is the reviewed author's decision and
   the standing parity-enabler policy, *not* because this measurement earned it.
   Whether default-ON is justified for a byte-identical, strictly-cheaper kernel
   (no shared memory, no barriers, one global read instead of two) whose
   step-level effect is real but under 1% is a **`NEEDS_DECISION` for the
   operator**, not an implementer's call.
2. #378 cannot close `ROAD-V1-A` alone and never could; at ~0.5-0.75% of decode
   GPU-busy it does not reach a 3-8% grid gap. §10 already said so.

**Next traceable hypotheses** (no ceiling is declared): a decode-only profiler
window to convert the 0.5-0.75% estimate into a measurement; then the two levers
§1 names and does not attempt — folding the router into the preceding gate-GEMM
epilogue, and batching the 40 per-step launches, where the remaining ~1.3 us/call
against `topkGating` and most of the launch/tail cost live.

**Owed:** gate 5 `compute-sanitizer memcheck`; the decode-only window; and a vLLM
denominator re-measured against the pin rather than the 0.25.0 rollback.

## Outcome

Pending. Filled in when the row reaches `DONE`, with what was measured, what was
rejected and why, and why the default is set the way it is. §12 supplies the
correctness half and the kernel-level measurement; the default-flip credit and
the decode-only attribution remain open.
