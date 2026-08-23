# LTX25-DIT-ATTN-FA2-HD128 — the tensor-core rung LTX-2.5 could not reach, because one template was never instantiated

Row: `LTX25-DIT-ATTN-FA2-HD128`, under the `ROAD-V1-LTX25` campaign
([`roadmap_v1.md`](../roadmap_v1.md), [`ltx-2-5.md`](ltx-2-5.md)), against the
model-matrix row `MODEL-DIFFUSION-ltx-2-5-ltx2-video-transformer-3d-model`.
Issue: [#1551](https://github.com/mudler/vllm.cpp/issues/1551), filed by
[#1549](https://github.com/mudler/vllm.cpp/issues/1549) and carried under `##
Owed` in [`ltx25-dit-attn-flash.md`](ltx25-dit-attn-flash.md) until this row
existed.

## Now

`DONE`. The change is landed as three product edits plus one vendored
translation unit, its correctness gates are green on `dgx:gpu0` (GB10,
sm_121a), and the speed axis is **measured rather than pending**: one binary,
one lease, `768x448/49f`, **flash 6.236 s against FA-2 2.276 s per DiT forward,
a ratio of 2.74x**. Section 8 carries the run. The A/B needed `dgx:gpu0` — the
box both prior LTX numbers were measured on — and it ran there.

Two things this row does NOT close, named here rather than left to a reader.
The naive-against-flash A/B inherited from #1549 stays `PENDING`, because the
naive arm did not run in this lease. The pixel comparison of a full render
across attention rungs stays owned by #1612.

Every bare `file:line` anchor below is read at this row's base,
**`73ada0df8de72853a20f9f0e5b00e33b1ab02a6c`**, which is `origin/main` at the
claim. Anchors written `file::Symbol` survive the moves this change makes and
are what `scripts/check-symbol-anchors.py` gates.

## 0. The one-sentence finding

`vt::AttentionDenseFa2` is the only op in this tree that reaches a tensor core
for dense non-causal attention, and it refused every head dim but 64
(`src/vt/cuda/cuda_flash_attn_fa2.cu` `LaunchDenseFA2Bf16`, and the dispatch
test in `cuda_ops.cu::AttentionDenseFa2KernelCuda`), so LTX-2.5's head_dim-128
DiT could not enter it — not because the kernel is missing, but because one
explicit template instantiation was never compiled.

## 1. Where this row starts

[#1549](https://github.com/mudler/vllm.cpp/issues/1549) moved the LTX-2.5 DiT
self-attention off `vt::Attention` and onto `vt::AttentionDenseFlash`, and one
DiT forward at `768x448/49f` went from **47.84 s** (n=119, median 47.91 s) to
**7.680 s** (n=19, median). That row's own `## Owed` names what it left:

> **True tensor cores at head_dim 128.** `vt::AttentionDenseFa2` refuses
> anything but head_dim 64. Reaching the vendored FA-2 `mma.sync` path for LTX's
> head_dim 128 needs an extra `run_mha_fwd_<bfloat16_t, 128, false>`
> instantiation. That is explicitly out of scope for this row, and it is the
> difference between this fix and a materially larger one: everything below is
> still a scalar warp-per-query recurrence.

That is this row. `AttentionDenseFlash` removed the redundant global K/V traffic
— the naive kernel re-read K and V once per (query, head) — but it did not
change the arithmetic unit. It is still one warp per query walking the keys
through a dependent online-softmax chain, in scalar FMAs. No `mma.sync`
anywhere.

## 2. Why the gap was a build-list entry and not a kernel

This is the part worth writing down, because it makes the size of the change
much smaller than the issue's framing suggests and the reader should be able to
check that claim rather than take it.

The vendored FlashAttention-2 tree at `src/vt/cuda/flash_attn/` already carries
the complete head_dim-128 machinery:

- `src/vt/cuda/flash_attn/src/flash_fwd_launch_template.h::run_mha_fwd_hdim128`
  is upstream's own generic launcher, unmodified, and it has been in this tree
  since the vendoring.
- `flash_fwd_split_hdim128_bf16_sm80.cu` and its causal sibling already
  instantiate `run_mha_fwd_splitkv_dispatch<bfloat16_t, 128, {true,false}>` and
  ship on every build. That is the kernel Qwen3-dense decode runs. The kernel
  traits, the `mma.sync` bodies, the softmax and the epilogue at head_dim 128
  are therefore already compiled, already exercised, and already gated.

What was missing is the **non-split entry point** at that head dim:
`run_mha_fwd_<bfloat16_t, 128, false>`, upstream's plain batch `mha_fwd`. The
split-KV dispatch is what a PAGED caller needs; a dense, non-paged,
single-request attention wants the batch forward, which is also what vLLM
dispatches for this shape (`vllm/model_executor/models/whisper.py`
`WhisperEncoderAttention:255` -> `forward:298-317` ->
`flash_attn_varlen_func`, `causal=False`).

`flash_fwd_hdim64_bf16_sm80.cu` — added for the Whisper encoder — is the
precedent and says so in its own header: it is a three-line file asking the
existing generic launcher for one more `(Headdim, entry-point)` pair. This row
adds the same file at 128.

**One thing genuinely differs at 128, and it is the reason the head dim is worth
stating rather than assuming.** On a non-`sm8x` target `run_mha_fwd_hdim128`
selects `Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, T>`, whose
`kSmemSize` is over CUDA's default 48 KiB launch cap. That is **not** the #1544
problem. `run_flash_fwd` raises the cap itself with
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
immediately before the launch. The vendored path carries its own opt-in; the
scalar `LaunchAttentionDenseFlash` is the one that does not, which is exactly
what [#1578](https://github.com/mudler/vllm.cpp/pull/1578) narrowed its
advertised bound to. **The two head_dim domains are therefore not nested**, and
§6 records the one consequence.

## 3. Scope

**In scope.**

1. `src/vt/cuda/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu` — the
   `run_mha_fwd_<bfloat16_t, 128, false>` instantiation, and its entry in
   `_FA2_KERNEL_SRCS` in `CMakeLists.txt`.
2. `cuda_flash_attn_fa2.cu::LaunchDenseFA2Bf16` — admit head_dim 128, dispatch
   to the matching instantiation, and keep refusing everything else **by name**.
3. `cuda_ops.cu::AttentionDenseFa2KernelCuda` — widen the shape gate to
   `{64, 128}`; every other term (bf16, non-causal, MHA) is unchanged.
4. `ltx2_device.cpp` — the DiT self-attention calls `vt::AttentionDenseFa2`, and
   the A/B knob becomes three-way so each rung is separately runnable from one
   binary.
5. `include/vt/ops.h` — the op's advertised head_dim domain.
6. Tests: head_dim-128 parity, key-range, causal refusal, fall-through and A/B
   knob cases in `tests/vt/test_ops_attention_dense_fa2.cpp`; the ported
   upstream tolerance rule (§5); the reachability case in
   `tests/vllm/models/test_ltx2_device.cpp` retargeted and widened.

**Out of scope, and named rather than silently absent.**

- **The causal hd-128 non-split instantiation.** LTX's DiT self-attention is
  bidirectional and no causal non-split hd-128 caller exists. Adding it is one
  line; adding it now would be compiling a kernel nothing reaches, which is the
  failure `AGENTS.md` §"Nothing lands dead" names. `LaunchDenseFA2Bf16` refuses
  causal by name until one appears.
- **f32.** FA-2 has no f32 arm at all. f32 callers fall through, as they always
  have.
- **The other head dims.** 96, 160, 192 and 256 have upstream launchers and no
  caller here. Same reason.
- **[#1552](https://github.com/mudler/vllm.cpp/issues/1552)**, the sweep of the
  remaining `vt::Attention` call sites. Different defect shape, its own row.

## 4. The dispatch, and why the model asks for the op unconditionally

`ltx2_device.cpp` calls `vt::AttentionDenseFa2` with **no shape test of its
own**. That is deliberate and is the op's own contract: it is TOTAL. Its fast
path is bf16 / head_dim in {64,128} / non-causal / MHA with the vendored kernels
compiled, and every other shape falls through to `AttentionDenseFlash`
bit-exactly. Four fall-through cases in
`tests/vt/test_ops_attention_dense_fa2.cpp` hold that, and this row adds two
more at head_dim 128.

Writing a shape test at the call site would be a **second copy of the op's
domain**, and the copy is what goes stale when the instantiation set moves. The
model states what it wants — the fastest correct dense non-causal attention —
and the op resolves it.

Both LTX streams are inside the fast path at the production dtype: video is 32
heads x 128, audio is 32 heads x 64, both bf16, both non-causal, both `h_k ==
h`. The f32 parity arm and the whole CPU backend take the fall-through and are
unchanged.

**The A/B knob is three-way**, because a two-arm knob cannot measure a
three-rung ladder, and because each arm must be able to state which rung it ran
from its own log rather than from its timing — the timing being the quantity
under measurement:

| `VLLM_LTX2_DIT_FLASH_ATTN` | op called | `VT_OP_PROVIDER_STATS` id |
|---|---|---|
| unset (default) | `vt::AttentionDenseFa2` | `kAttentionDenseFa2` |
| `flash` | `vt::AttentionDenseFlash` | `kAttentionDenseFlash` |
| `0` | `vt::Attention` | `kAttention` |

`=0` keeps exactly the meaning #1549 gave it and `docs/ENVIRONMENT.md` records,
so the 47.84 s denominator stays reachable from this binary. `flash` is the arm
this row's ratio is taken against.

The alternative — reusing the global `VT_FA2_DENSE=0` for the flash arm — was
rejected. It works, but both arms then resolve `kAttentionDenseFa2` and the
op-provider log cannot tell them apart, so the only evidence of which kernel ran
would be the wall clock. Three ops give three ids.

## 5. Numerics: the gate is upstream's rule, not a bound fitted to the result

FA-2 is **not** bit-identical to `AttentionDenseFlash`, and this row does not
pretend otherwise. `mma.sync` reassociates both the QK^T and the P.V reductions,
and the vendored kernel exponentiates with `exp2f` on a log2-scaled score where
the scalar kernels use `expf` on a linearly-scaled one. A diffusion model has no
token gate to fall back on, so the deviation has to be bounded directly.

**Two comparisons, and only one of them is a gate.**

**(a) The arm-to-arm rel-L2, against the ALREADY-COMMITTED bound.**
`kRelL2Bound = 1.0e-2` and `kMaxAbsVsRmsBound = 0.15` are in
`tests/vt/test_ops_attention_dense_fa2.cpp` at this row's base and were derived
for the hd-64 case from the **output dtype**: bf16's relative resolution is
`2^-8 = 3.9e-3`, so 1e-2 is about 2.5 bf16 ulps. That is a property of the store
width, not of a head dim, a sequence length or a measurement. This row reuses
those constants unchanged at head_dim 128, at LTX's own geometry (T=2352, H=32,
D=128) and at two ragged shapes. **Widening either constant for this head dim
would be fitting the bound to the result and is a stop condition, not a repair.**

**(b) The PORTED upstream tolerance rule, which is the actual gate.** (a) tells
you how far the two arms diverge and nothing about which is right; both could be
wrong together. Upstream FA-2 answers that question with a rule stated against a
more accurate reference — `tests/test_flash_attn.py::test_flash_attn_output`
(`vllm-project/flash-attention @ 2c839c33`):

```python
assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
```

`out_ref` is the higher-precision reference, `out_pt` the reference kernel at
the tested dtype, `out` the FA-2 result. **Only the harness is adapted**:
`out_ref` is a `double` host reference computed in the test file, because
porting torch's fp32 SDPA is not possible here, and `out_pt` is our own
`AttentionDenseFlash` rather than torch's bf16 SDPA. **The factor of 2 and the
max-abs statistic are upstream's and are not re-derived.**

The rule is run at LTX's production **sequence length** (T=2352) with a reduced
head count (H=2), and that reduction is stated rather than hidden: the reference
is `O(T^2 * D)` in scalar double per head, heads are independent in this op, and
the sequence length is the axis the reduction order actually accumulates over.
Reducing T instead would reduce the thing being measured.

The case also `REQUIRE`s that the incumbent's own error against the reference is
non-zero, because otherwise the rule reads `x <= 0` and is a mute switch wearing
a strict gate.

**What neither comparison bounds, and this row inherits the gap rather than
closing it by assertion.** Both are per-op comparisons at one attention. Neither
bounds the accumulated deviation over 48 blocks x 120 forwards of a denoise
trajectory, and no pixel comparison of an LTX render against an alternative
attention rung exists anywhere in this tree —
[#1612](https://github.com/mudler/vllm.cpp/issues/1612) already owes that for
#1549's swap and this row does not discharge it. §8 records what was attempted.

## 6. Risks

- **The advertised-domain asymmetry.** The FA-2 arm has no 48 KiB bound (the
  vendored launcher opts in for itself); the fall-through arm does. bf16
  head_dim 128 is inside both. **f32 head_dim 128 is inside neither** — f32 is
  not an FA-2 shape and its `AttentionDenseFlash` tile is 65,536 B — so it
  reaches that op's named refusal. That is the LTX f32 parity arm at production
  geometry, which #1549 already disclosed and #1612 already owns. This row does
  not change it and does not silently paper over it.
- **Compile time and binary size.** One more `run_flash_fwd` instantiation,
  with the same `BOOL_SWITCH` fan-out every other vendored TU already pays.
  Measured in §8.
- **A partial revert.** One stream moved back to flash while the other stays on
  FA-2 would leave the video stream — the whole point of this row — without
  tensor cores. The reachability case's negative half now names
  `kAttentionDenseFlash` as well as `kAttention` for exactly this.

## 7. Tests and gates

| Gate | What it holds |
|---|---|
| `test_ops_attention_dense_fa2` hd-128 parity | FA-2 vs flash at T=2352/H=32/D=128 and two ragged shapes, against the base's own committed bounds |
| `test_ops_attention_dense_fa2` upstream rule | §5(b): `max\|fa2-ref\| <= 2 * max\|flash-ref\|` against a `double` host reference |
| `test_ops_attention_dense_fa2` hd-128 key range | perturbing V's tail must move the output; a clamped `seqlen_k` or a mis-stride goes red |
| `test_ops_attention_dense_fa2` hd-128 causal | a causal request must LEAVE the op; kills a gate written `(d==64 && !causal) \|\| d==128` |
| `test_ops_attention_dense_fa2` fall-through | head_dim 80 AND 192 must fall through bit-exactly; {64,128} is a set, not an interval |
| `test_ops_attention_dense_fa2` `VT_FA2_DENSE=0` at hd-128 | the same-binary rollback really routes back, and the ON arm is NOT that same answer |
| `test_ltx2_device` reachability | through `Ltx2DitForwardDevice`: `kAttentionDenseFa2` selected exactly `2 * layers * batch`, `kAttentionDenseFlash` 0, `kAttention` 0 |
| A/B on `dgx:gpu0` | per-forward median at `768x448/49f`, fa2 arm against flash arm, one binary, from the engine's own `last=` lines |

**Reachability mutation.** Replace `vt::AttentionDenseFa2` with
`vt::AttentionDenseFlash` at `ltx2_device.cpp`'s self-attention branch in a
scratch copy and rerun `test_ltx2_device`. The case must go red on the positive
half. A gate that stays green without the call site measures a class, not a
capability.

## 8. Evidence

Measured on `dgx:gpu0` (GB10, sm_121a) on 2026-08-22, inside ONE `rc` lease.
Nothing is written here that was not run. The full entry, including the
reproduce recipe, is in [`.agents/benchmark-record.md`](../benchmark-record.md).

### 8.1 Provenance

| | |
|---|---|
| rc job | `91e0b5d9-b7f7-4b69-bf3f-d593aa25f871`, device `dgx:gpu0`, one lease, no other job on the box |
| harness | `scripts/ltx25-dit-attn-fa2-hd128-ab.sh`, sha256 `981265f3340f966c1e72b8dd1a3c251cee959298b346f02ef21968f0218d32bb`, verified byte-identical to the committed file on both sides |
| binary | `/root/abbin/ltx2-gen`, sha256 `f8738c39d1bb6cb7ca0bf95e77fa815bd1594796a0babcefd01269750a328342` — **ONE** binary, both arms |
| `built_from` | `6b37934b8ebf140b13057aed0e33411ab1626d6d` |
| toolchain | nvcc 13.3, arch `121a`, CUTLASS at `/root/cutlass`, FlashAttention-2 `ENABLED for [121a]`, `BUILD_RC=0`, `compile_errors=0` |
| geometry | `768x448`, 49 frames = **2352** video tokens, seed 20260820, `one_stage`, bf16 dev DiT, full 21.00B checkpoint |
| statistic | per-forward `last=` from the engine's own `[render] dit forward` lines, never the governor |
| stop | both arms `stopped_by=sample-cap`, never the memory floor; `exit=130` is the expected SIGINT |
| artifacts | `/mnt/nas_share/rc/ltx25-fa2hd128/out/20260822T203535Z/` |

**The measured tree is not the landed tree, and the difference is stated rather
than assumed.** The binary was built from `6b37934b8`. The head this row lands
on is `bb1acedd5`, and the drift between them comes from two places rather than
one, so it is listed rather than summarised. Run
`git diff --name-only 6b37934b8..bb1acedd5` to reproduce this paragraph.

**This row's own later commits touch three records and nothing else.**
`3c0d020c0` and `dde8dc4f9` change `.agents/issue-index.md`,
`.agents/benchmark-record.md` and this spec.

**The merge `bb1acedd5` then brings seven unrelated commits from `origin/main`
`6354755ba`, and this is what the earlier reading of this paragraph missed.**
Outside `.agents/` and `docs/`, that merge moves exactly seven files:
`src/vllm/model_executor/models/campplus.cpp`,
`include/vllm/v1/attention/backend.h`, `scripts/check-windows-portability.py`,
`tests/vllm/models/test_campplus.cpp`,
`tests/scripts/test_check_windows_portability.py`,
`tests/vllm/v1/attention/test_attn_validate_configuration.cpp` and
`tests/vllm/v1/worker/test_runner.cpp`. None of them is on the LTX-2.5 or the
FA-2 path: `campplus.cpp` is the CAMPPlus speaker model, `backend.h` declares
ROCM_ATTN's KV block sizes, `check-windows-portability.py` is a checker, and the
four tests are the tests of those three.

**Of the eighteen files this row touches, four differ between `6b37934b8` and
`bb1acedd5`, and all four are records.** They are the three `.agents/` files
above, plus `docs/USAGE.md`, which main's `ab1de9b95` moved rather than this
row. No product source, no test, no script and no CMake entry that this row
touches differs between the binary that produced these numbers and the landed
head. All eleven of them are byte-identical on both sides: `ltx2_device.cpp`,
`cuda_flash_attn_fa2.cu`, `cuda_ops.cu`, `flash_fwd_hdim128_bf16_sm80.cu`,
`include/vt/ops.h`, `CMakeLists.txt`,
`scripts/ltx25-dit-attn-fa2-hd128-ab.sh`,
`scripts/check-attention-rung-consistency.py`,
`tests/scripts/test_check_attention_rung_consistency.py`,
`test_ltx2_device.cpp` and `test_ops_attention_dense_fa2.cpp`. Nobody has to
assume the measured tree is the landed tree.

### 8.2 Which rung each arm ran — READ, not inferred

| arm | knob | op-provider announce on `device=1` | wanted |
|---|---|---|---|
| flash (denominator) | `VLLM_LTX2_DIT_FLASH_ATTN=flash` | `op=21`, `kAttentionDenseFlash` | exactly `[21]` |
| fa2 (numerator) | unset (the default) | `op=22`, `kAttentionDenseFa2` | exactly `[22]` |

`assert_arm_op` exits 47 on a mismatch, so the rung is asserted and not merely
printed. Neither arm resolved the other arm's op, and neither resolved `op=18`,
`kAttention`. This is what the three-way knob of §4 buys: the wall clock is the
quantity under measurement, so it cannot also be the evidence of which kernel
ran.

### 8.3 Per-forward wall time

The `last=` value printed on the line that announces forward N is the interval
from the announcement of forward N-1, so it covers forward N-1 and the
bookkeeping between the two. Both arms use the same convention and the same
forward indices, so the paired reduction of §8.4 compares equal intervals.

| arm | n | median | mean | min | max | IQR |
|---|---|---|---|---|---|---|
| flash | 13 | **6.236 s** | 6.167 s | 5.842 s | 6.394 s | [6.010, 6.322] |
| fa2 | 14 | **2.276 s** | 2.268 s | 1.915 s | 3.162 s | [2.184, 2.289] |

Quartiles use the Weibull definition, position `p * (n + 1)`, named because two
conventions give different hinges at these sample sizes.

flash raw, sorted:

```
5.842 5.984 6.009 6.011 6.019 6.183 6.236 6.239 6.259 6.307 6.336 6.355 6.394
```

fa2 raw, sorted:

```
1.915 1.920 1.926 2.270 2.273 2.275 2.275 2.277 2.280 2.288 2.288 2.291 2.309 3.162
```

Both arms asked for 13 samples. The FA-2 arm produced 14, because the watchdog
polls every 5 s and an FA-2 forward costs 2.3 s, so the arm passed its cap
between two polls. The extra sample is kept.

**THE CLAIM: flash median divided by FA-2 median = 6.236 / 2.276 = 2.74x**, one
binary, one lease, one geometry, one seed, one prompt.

### 8.4 Corroboration — PAIRED by forward index

| line | step | flash | fa2 | ratio |
|---|---|---|---|---|
| 2 | 1/30 | 6.336 | 3.162 | 2.004x (FA-2's one-time warm-up) |
| 3 | 1/30 | 5.984 | 2.273 | 2.633x |
| 4 | 1/30 | 6.019 | 2.270 | 2.652x |
| 5 | 2/30 | 5.842 | 1.915 | 3.051x |
| 6 | 2/30 | 6.307 | 2.275 | 2.772x |
| 7 | 2/30 | 6.239 | 2.280 | 2.736x |
| 8 | 2/30 | 6.183 | 2.275 | 2.718x |
| 9 | 3/30 | 6.009 | 1.920 | 3.130x |
| 10 | 3/30 | 6.355 | 2.277 | 2.791x |
| 11 | 3/30 | 6.394 | 2.288 | 2.795x |
| 12 | 3/30 | 6.259 | 2.288 | 2.736x |
| 13 | 4/30 | 6.011 | 1.926 | 3.121x |
| 14 | 4/30 | 6.236 | 2.291 | 2.722x |

Paired median **2.736x**, paired mean 2.758x, n=13. Every paired interval is at
least 2.00x and twelve of the thirteen are at least 2.63x. The paired median
2.7364x and the median-of-medians 2.7399x agree to **0.13%**, which is what
makes the headline robust to the choice of reduction instead of dependent on it.

### 8.5 Two distribution facts, stated rather than smoothed away

**FA-2's slowest sample is its FIRST measured forward.** 3.162 s against a
steady 2.28 s. It is a one-time warm-up and it is **INCLUDED** in the median and
in the claim. Excluding it raises the paired median from 2.736x to 2.754x. It is
not excluded.

**FA-2's samples are bimodal.** Three of the fourteen sit at 1.915 s, 1.920 s
and 1.926 s against ten between 2.270 s and 2.309 s. The three cheap samples are
the `last=` values on the lines announcing forwards 5, 9 and 13, the first
forward of denoise steps 2/30, 3/30 and 4/30. Under the convention of §8.3,
those intervals cover forwards 4, 8 and 12 — the LAST forward of the preceding
denoise step — together with that step's teardown.

**Flash carries the same effect, and the first reading of these samples recorded
it as absent.** Flash's three step-boundary intervals are 5.842 s, 6.009 s and
6.011 s, which are ranks 1, 3 and 4 of its thirteen samples. Its boundary mean
is 5.954 s against 6.231 s for the other ten, a saving of 0.277 s. FA-2's is
1.920 s against 2.283 s, a saving of 0.362 s. The two savings are close in
absolute terms, so the effect reads as a fixed per-step term and not as a
property of either kernel. It looks like a split in the FA-2 arm alone because
0.362 s is 15.9% of 2.28 s while 0.277 s is 4.4% of 6.23 s.

**The two interiors above are NOT built by the same rule, and the asymmetry is
named here rather than left for a reader to find.** Flash's interior is every
one of its ten non-boundary samples, its first included. FA-2's is ten of its
eleven, because the 3.162 s warm-up of the paragraph above is also removed. The
rule is "make both interiors n=10", which is the defensible one — equal counts
are what let two means be set against each other at all — but it is a choice,
so the other two are stated beside it:

| interior rule | flash saving | FA-2 saving |
|---|---|---|
| both n=10; the warm-up leaves FA-2's set only (**used above**) | 0.277 s | 0.362 s |
| both drop their first sample; n=9 and n=10 | 0.266 s | 0.362 s |
| both keep every non-boundary sample; n=10 and n=11 | 0.277 s | 0.442 s |

**The reading survives all three.** Under every rule the two savings sit within
0.17 s of each other while the two interiors themselves differ by 2.73x, which
is the fixed-per-step-term reading and not a marginal call. Under every rule the
FA-2 saving is the larger of the two in absolute terms, so no rule turns the
effect into a property of flash. The rule used is nonetheless the one whose two
savings are CLOSEST — 0.085 s apart, against 0.097 s and 0.165 s — which is
worth saying out loud, because it is the rule that flatters that reading most.

Neither fact moves the comparison. Both arms cover identical forward indices, so
the paired table of §8.4 compares a boundary interval against a boundary
interval and an interior interval against an interior interval.

### 8.6 What is NOT claimed

**The harness prints `47.84 / 2.276 = 21.02x` and that line is a CROSS-RUN
comparison, not this A/B.** The 47.84 s naive figure came from another binary in
another lease (#1549). It is not claimed here, and it is not claimed in the
benchmark record.

**This run's own flash arm measured 6.236 s where #1549 recorded 7.680 s for the
same rung.** The denominator moved by 18.8% between the two runs, on the same
box, at the same geometry. That movement is why this row's claim is same-binary
and same-lease, and why the cross-run number is not a claim. The 7.680 s figure
is not withdrawn: it is a correct measurement of a different binary in a
different lease, and #1549's own entry already names its confounds.

**The naive arm did not run in this lease.** It is opt-in in the harness and
costs about 1500 s, and its value already exists at n=119. The
naive-against-flash A/B inherited from #1549 therefore stays `PENDING` and this
row does not discharge it.

**No pixel comparison of a full render across rungs.** Owed under #1612, not by
this row. This row's numeric evidence is per-op and does not bound a 120-forward
denoise trajectory.

### 8.7 Correctness, established BEFORE the speed number

Phase `[E]`, same binary, same lease.

| suite | cases | assertions | result |
|---|---|---|---|
| `test_ops_attention` | 11 | 37,259 | SUCCESS |
| `test_ltx2_device` | 22 | 757 | SUCCESS |
| `test_ops_attention_dense_fa2` | 12 | **29** | SUCCESS |

The 29 is load-bearing. Every case in `test_ops_attention_dense_fa2` is
CUDA-gated and returns early on a CPU build, where the suite reports `12 cases |
0 assertions`. A non-zero assertion count is therefore the proof that the
numeric cases RAN instead of skipping.

### 8.8 The numeric gate — upstream FA-2's own rule, ported at pin 2c839c33

At `T=2352` (the production sequence length), `H=2`, `D=128`, against a `double`
host reference:

| quantity | value |
|---|---|
| `max\|fa2 - ref\|` | 1.79339e-4 |
| `max\|flash - ref\|` | 1.22079e-4 |
| ratio | 1.46904 |
| rule `max\|fa2 - ref\| <= 2 * max\|flash - ref\|` | 1.79339e-4 <= 2.44158e-4, **PASS**, 26.5% margin |
| rel-L2 against the f64 reference | fa2 2.3466e-3, flash 1.65505e-3 |

`max|flash - ref|` is non-zero, so the rule is a real inequality and not
`x <= 0`.

### 8.9 The arm-to-arm bound is DERIVED from the store width, not fitted

`kRelL2Bound = 1.0e-2` and `kMaxAbsVsRmsBound = 0.15` are BYTE-UNCHANGED from
before this row. bf16's relative resolution is `2^-8` = 3.90625e-3, so 1e-2 is
2.56 bf16 units in the last place (ulp).

At LTX's real geometry, `T=2352 H=32 D=128` non-causal:

| quantity | measured | bound | margin |
|---|---|---|---|
| rel-L2 | **2.30865e-3**, which is 0.59 bf16 ulp | 1.0e-2 | 4.33x |
| `max\|diff\|` | 9.76562e-4 | `0.15 * rms(ref)` = 1.82540e-3 | 1.87x |

Supporting cases, all of which fired:

| case | measured |
|---|---|
| hd-64 parity, three shapes | rel-L2 2.55259e-3, 2.50784e-3, 2.36637e-3 |
| hd-128 parity, two further shapes | rel-L2 2.45971e-3 at `T=577 H=3`, 2.25325e-3 at `T=17 H=2` |
| hd-64 key range | perturbing V rows `[750,1500)` moved the output by rel-L2 254.113 against a 0.01 envelope; the scalar reference moved by 254.114 |
| hd-128 key range | perturbing V rows `[1176,2352)` moved the output by rel-L2 311.431 |
| causal refusal | causal against non-causal rel-L2 2.17173, which must be large |
| fall-through | head_dim 80 AND head_dim 192 bit-exact, so `{64, 128}` is a set and not an interval |
| `VT_FA2_DENSE=0` rollback at hd-128 | the ON arm differs from flash in **56,025** elements, so it is a different kernel and not the same answer twice |

### 8.10 Reachability, through the production entry point, counted on the CPU backend

**These counts are a CPU-device measurement, and nothing here is a CUDA
measurement.** The case builds `vt::Queue q{Cpu(), nullptr}` and every
`GetOpProviderStats` call in it reads `vt::DeviceType::kCPU`, which is why the
table below names no device. The binary that ran it is the CUDA-enabled build of
§8.1, but a binary that CAN reach CUDA is not evidence that CUDA ran.

That is not a weakening, for the reason
`tests/vllm/models/test_ltx2_device.cpp::Ltx2DitForwardDevice`'s own comment
gives: `GetOp` is the dispatch point on every backend, so the ROUTING is
observable without a GPU, and what a GPU gates is the kernel BEHIND the routing
rather than the selection of it. On CPU all four ops resolve to the same
registered function in `src/vt/cpu/cpu_ops.cpp`, which is exactly why the golden
cases stay byte-unchanged by this swap and why this case has to ask the provider
rather than the numbers. The CUDA kernel behind `kAttentionDenseFa2` is gated by
§8.8 and §8.9 instead, and the wall clock it buys by §8.3.

Counted through `Ltx2DitForwardDevice`:

| arm | `kAttentionDenseFa2` | `kAttentionDenseFlash` | `kAttention` |
|---|---|---|---|
| default | **8** (want 8) | 0 | 0 |
| `VLLM_LTX2_DIT_FLASH_ATTN=flash` | 0 | **8** (want 8) | 0 |

Both rows are two-sided. The arm under test counts exactly `2 * layers * batch`
and each of the other two ops counts zero, so a partial revert of one stream
goes red instead of passing quietly.

### 8.11 One harness defect this run found — [#1734](https://github.com/mudler/vllm.cpp/issues/1734)

`memavail low-water:` printed EMPTY for both arms. The cause is the writer and
not the reducer that prints it. At
`scripts/ltx25-dit-attn-fa2-hd128-ab.sh:367`,
`n=$(grep -c 'last=' "$log" 2>/dev/null || echo 0)` emits two lines when the
count is zero, because `grep -c` prints `0` and also exits 1, so `|| echo 0`
fires as well. The tab-separated record then lands split across two lines, and
`$4` is empty on 170 of `watch-flash.tsv`'s 186 lines (85 with `NF=3`, 85 with
`NF=2`, 16 with `NF=4`; `watch-fa2.tsv` reads 85 / 85 / 6). The 85 pairs are the
polls taken during model load, before any `last=` line existed. An empty string
sorts first under `sort -n`, so the reducer at line 390 returns it.

**It touches no number in this section.** Both arms report
`stopped_by=sample-cap`, which is the direct evidence that neither was stopped
by memory pressure. Re-derived from the same files with a prefix-stripping match
instead of a positional one, the low-water is **40.3 GiB on both arms** against
`MEM_FLOOR_GIB=12.0`, so the run stayed 3.36x above its own floor throughout.

Not repaired in the commit that recorded it, because that commit is
`.agents/`-only by scope and a `scripts/` edit owes a red-first case over a
fixture TSV. Carried under `## Owed`.

## 9. Stop conditions

- **The numeric gate cannot be met at the committed bounds.** Report the
  measured deviation and stop. Do not widen `kRelL2Bound`, do not widen
  upstream's factor of 2, and do not reduce the geometry until it passes.
- **The A/B cannot run on `dgx:gpu0`.** A different box is not a valid
  denominator for the 7.680 s flash number. Record the speed axis `PENDING` with
  the reason and land the correctness half, or hold the row.
- **The instantiation does not build for `sm_121a`.** Report the compiler
  output. Do not disable a guard to get past it.

## Owed

- **The pixel comparison of a full render across attention rungs.** Inherited
  from #1549 and owned by
  [#1612](https://github.com/mudler/vllm.cpp/issues/1612), not by this row. This
  row's numeric evidence is per-op and does not bound a 120-forward denoise
  trajectory.

- **The distilled NVFP4 DiT's recorded revision AND its recorded size both
  disagree with the local artefact.** Two fields, not one, and the review of this
  row found the second. `docs/USAGE.md` pins
  `Lightricks/LTX-2.5 @ 6c7e5e573ac1667efc83407806fe9b0b93730e60` for
  `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`, while
  that file's `huggingface_hub` `.metadata` sidecar on the shared checkout
  records `8a4ff96f581e72bedc1b44367581c49d544a05f1`. The same row records
  **18,721,548,408 bytes**, while `stat -c %s` on
  `/mnt/nas_share/checkpoints/ltx-2.5/lightricks-ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
  gives **18,721,432,024 bytes**, re-derived on 2026-08-22 rather than
  transcribed — a difference of 116,384 bytes. A SIZE disagreement is the
  stronger of the two, because a size is what this registry uses to identify an
  artefact when no content hash is available, and 116,384 bytes is far too small
  to be a different model and far too large to be rounding. Both fields still
  admit the same benign explanation — a later re-quantization published under an
  unchanged name, re-downloaded after the row was written — and the two bf16 DiT
  rows have no sidecar at all, so nothing local contradicts them. Deliberately
  NOT folded in: replacing a possibly-stale revision and size with values whose
  provenance is only "what happens to be on the share today" swaps a possibly
  stale pin for a definitely unverified one, which is worse. Settling it needs an
  authenticated fetch at a named revision, which this row has no authority to
  make and no way to gate. This row does not run that model arm. Owner: this row.
  Issue: [#1723](https://github.com/mudler/vllm.cpp/issues/1723).

  **Why #1723 and not #1702, which this bullet named first.** The discrepancy was
  found while fixing [#1702](https://github.com/mudler/vllm.cpp/issues/1702) and
  was first recorded against it. #1702's subject is a different bug — three of
  the four LTX-2.5 artefacts every render is fed having no row in the
  `docs/USAGE.md` checkpoint registry — and this pull request fixes that
  completely and carries `Closes #1702`, so the merge closes it. The discrepancy
  above is about a FOURTH row that already existed and that the fix does not
  touch. Tracking it on #1702 would therefore have made it invisible at the exact
  moment this change landed, because AGENTS.md relies on GitHub holding the open
  and closed state and a `## Owed` bullet pointing at a closed issue tracks
  nothing. It is split onto #1723 so the closed half and the open half each have
  their own record.

- **The #1702 index row's sidecar count is stale and cannot be repaired in
  place.** That row states that "all four `Lightricks/LTX-2.5` sidecars carry the
  SAME `commit_hash`". Re-derived on 2026-08-22: there are **six**, not four —
  `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`,
  `model_patches/ltx-2.5-duration-head-bf16.safetensors`,
  `vae/ltx-2.5-video-vae-bf16.safetensors`,
  `vae/ltx-2.5-audio-vae-bf16.safetensors`,
  `vae/ltx-2.5-video-vae-conv-bf16.safetensors` and
  `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`. The
  SUBSTANCE of the claim survives the correction: all six record
  `8a4ff96f581e72bedc1b44367581c49d544a05f1`, which is what makes that value a
  snapshot revision rather than a blob id, and it is the two sidecars the row did
  not count that carry the extra evidence. `.agents/issue-index.md` is
  append-only and carries `merge=union`, so the row itself is not editable and is
  not edited; this bullet is where the corrected count lives. It is also recorded
  on [#1723](https://github.com/mudler/vllm.cpp/issues/1723), because the
  six-sidecar census is the evidence that makes `8a4ff96f…` a snapshot revision,
  and #1702 — the issue whose row carries the stale count — closes with this pull
  request and cannot hold a correction after that. Owner: this row.
  Issue: [#1723](https://github.com/mudler/vllm.cpp/issues/1723).

- **The FA-2 dense head-dim fall-through `else throw` has no executable
  coverage, and no mutation in this tree can turn it red.** The launcher's
  head-dim dispatch now ends `if (d == 64) { … } else if (d == 128) { … } else
  { throw … }` (`cuda_flash_attn_fa2.cu::LaunchDenseFA2Bf16`), and the fresh
  review of this row asked for that shape because it is right: the shape it
  replaced put the 128 call in a bare `else`, so widening only the
  `d != 64 && d != 128` admissibility guard earlier in the same function — the
  exact first edit a head_dim-192 rung makes — would have routed 192 into the 128
  kernel, which reads 128 of its 192 channels and returns a silently truncated
  answer. But the two guards are ORDERED, so while the admissibility guard stands
  the `else` cannot be entered: no input reaches it, no test can enter it without
  first making the very edit it guards, and deleting the arm or inverting its
  condition leaves every case green. It is a guarantee no mutation can red. This
  is NOT a regression and NOT a capability that landed dead — the same input
  previously produced a silently truncated answer with no diagnostic, so the arm
  strictly replaces silence with a named refusal. The owning suite,
  `test_ops_attention_dense_fa2`, is CUDA-gated in all 12 of its cases and
  reports `12 cases | 0 assertions` on a CPU build, so it gives the arm no
  coverage on the authoring host either. OWED: the first commit that widens the
  `d != 64 && d != 128` guard — for head_dim 192, for f32, or for any new rung —
  owes a RED-FIRST case proving this throw fires for a head dim the widened guard
  admits and the launcher has no instantiation for, taken BEFORE the
  instantiation that makes the throw unreachable again is added. That widening is
  the only moment at which the guarantee is both reachable and provable. Owner:
  this row. Issue: [#1724](https://github.com/mudler/vllm.cpp/issues/1724).

- **`scripts/ltx25-dit-attn-fa2-hd128-ab.sh` prints an empty
  `memavail low-water:` for every arm.** Found by this row's own run and
  diagnosed in §8.11: `n=$(grep -c 'last=' "$log" 2>/dev/null || echo 0)` at
  line 367 emits two lines when the count is zero, so the tab-separated record
  lands split across two lines and the positional reducer at line 390 reads an
  empty `$4`. It touches no number here — both arms report
  `stopped_by=sample-cap` and the true low-water is 40.3 GiB against a 12.0 GiB
  floor — but the harness is one this repository reuses, and a memory report
  that cannot fail loudly is worse than none. NOT fixed in the record commit
  that found it, because that commit is `.agents/`-only and a `scripts/` edit
  owes a red-first case over a fixture TSV. Owner: this row. Issue:
  [#1734](https://github.com/mudler/vllm.cpp/issues/1734).

- **`VLLM_LTX2_DIT_FLASH_ATTN` matches the naive arm on a PREFIX and falls
  through to the FA-2 default for every value it does not recognise, so a typo
  cannot refuse.** In `ltx2_device.cpp` the naive arm tests `arm[0] == '0'`
  while the flash arm uses `strcmp(arm, "flash")`, so the two arms of one knob
  are read by two rules: `0x`, `07` and `0flash` all select `vt::Attention`,
  and `falsh`, `FLASH`, `naive` or an empty string all land in the bare `else`
  and run the FA-2 default silently. The `flash` value is the DENOMINATOR of
  §8's ratio, so a mistyped denominator does not fail — it runs the numerator's
  kernel twice and yields ~1.00x, which is also what "no speedup" looks like, so
  the number cannot report its own failure. That is the shape `847e22f80`
  already repaired once on this knob. The measurements here are NOT exposed:
  `assert_arm_op` reads the op-provider announcement and exits 47 on a mismatch,
  so every arm in §8.2 stated its rung. An operator setting the variable by hand
  gets no such check. The fix is an exact three-way parse with a refusal that
  names the variable and lists its values, and it is NOT folded in here because
  a dispatch edit owes a red-first case asserting the refusal, taken before the
  parse changes — a separate rung with its own row, and this change is
  `.agents/`-only by scope. Owner: this row. Issue:
  [#1751](https://github.com/mudler/vllm.cpp/issues/1751).

## Outcome

### What was measured

One binary, one lease, one geometry. The LTX-2.5 DiT forward at `768x448/49f`
costs **6.236 s** on `vt::AttentionDenseFlash` and **2.276 s** on
`vt::AttentionDenseFa2`, a ratio of **2.74x** (§8.3). A paired reduction over
the same thirteen forward indices gives 2.736x (§8.4). The two reductions agree
to 0.13%, so the headline does not depend on which one a reader prefers. The
whole change was one template instantiation, its build-list entry, two widened
shape gates and one call site.

### What was rejected, and why

**Claiming the ratio against #1549's recorded 7.680 s.** That number is a
correct measurement of another binary in another lease. This run's own flash arm
measured 6.236 s for the same rung, on the same box, at the same geometry —
18.8% faster. The denominator moved between runs by more than any effect a
cross-run comparison could hope to resolve. Quoting the `47.84 / 2.276 = 21.02x`
line the harness prints would have inherited that movement twice over, once in
each term. The claim is same-binary and same-lease, and it is 2.74x.

**Reusing the global `VT_FA2_DENSE=0` for the flash arm.** It works and it costs
no new knob. Both arms then resolve `kAttentionDenseFa2`, so the op-provider log
cannot tell them apart, and the only remaining evidence of which kernel ran is
the wall clock — the quantity under measurement. The three-way
`VLLM_LTX2_DIT_FLASH_ATTN` knob gives three ops and three announce ids, and
`assert_arm_op` exits 47 on a mismatch rather than printing them for a reader to
check (§4, §8.2).

**Excluding FA-2's 3.162 s first forward.** It is a real one-time warm-up, and
dropping it would raise the paired median from 2.736x to 2.754x. A claim
improved by deleting its own worst sample is a claim about a shorter run. It is
in.

**Recording that flash shows no step-boundary effect.** The first reading of
these samples said FA-2 is bimodal and flash is flat. Flash's three
step-boundary intervals are ranks 1, 3 and 4 of its thirteen samples, and its
boundary saving is 0.277 s against FA-2's 0.362 s. The effect is a fixed
per-step term present in both arms, and §8.5 records it that way instead.

**Repairing the harness defect in the same commit.**
[#1734](https://github.com/mudler/vllm.cpp/issues/1734) is real and its fix is
three lines, but it lives in `scripts/` and owes a red-first case over a fixture
TSV. The commit that found it is `.agents/`-only by scope.

### Why each default has its value

**The head_dim domain is the SET `{64, 128}` and not an interval.**
`LaunchDenseFA2Bf16` dispatches on a template parameter, so each admitted head
dim is a compiled instantiation and not an argument to one call. head_dim 96,
160, 192 and 256 have upstream launchers and no caller in this tree, and
compiling them would land a kernel nothing reaches. Two fall-through cases hold
the set as a set: head_dim 80 and head_dim 192 both fall through bit-exactly,
and 192 is the one that goes red if the guard is ever rewritten as a range.

**The causal hd-128 instantiation is deliberately absent.** LTX's DiT
self-attention is bidirectional and no causal non-split hd-128 caller exists
anywhere in this tree. Adding the instantiation is one line, and it would
compile a kernel nothing reaches, which is the failure `AGENTS.md` names under
"Nothing lands dead". `LaunchDenseFA2Bf16` refuses causal by name until a caller
appears, and the causal case holds that refusal at rel-L2 2.17173 against the
non-causal answer. #1724 owns the one guarantee this ordering leaves untestable.

**The numeric bounds were not touched.** `kRelL2Bound = 1.0e-2` and
`kMaxAbsVsRmsBound = 0.15` are byte-unchanged from before this row, and the
factor of 2 is upstream's own rule. The measured rel-L2 at the production
geometry is 0.59 bf16 ulp, a 4.33x margin, so no bound needed widening and none
was widened.

**The A/B ran on `dgx:gpu0` and on no other box.** Both prior LTX numbers were
measured there, and §9 named a different box as a stop condition rather than as
a fallback.
