# ROCM-FUSED-NORM-ROPE — `vt::FusedNormRope` on ROCm, so GLM-5.3 reaches a token on `gfx1151`

Row: `BACKEND-ROCM`
Issue: [#2564](https://github.com/mudler/vllm.cpp/issues/2564)

## Now

`ACTIVE`, measured. Base `4fe3852b119e40cda05c0fbcf64d3e2a4796ada2`.

## Scope

Register a native `kFusedNormRope` kernel for `DeviceType::kROCM`, and correct
the MLA block's stale comment and refusal message so both causes of the split
A-projection branch are named.

Out of scope: the other seven missing MLA/DSA ops
(`.agents/specs/rocm-glm53-dsa.md` W1.3). They keep serving from the portable
reference tier and are recorded under `## Owed`.

## The defect

`src/vllm/model_executor/layers/attention/mla_attention.cpp:550-551`:

```cpp
const bool fused_nr = R > 0 && !has_k_rope_norm && MlaFusedNormRopeEnabled() &&
                      vt::OpRegistered(vt::OpId::kFusedNormRope, d.q.device.type);
```

`vt::OpRegistered` is a native-only probe by design
(`src/vt/op_provider.cpp:779-803`) and ROCm registers no `kFusedNormRope`. With
every environment variable unset, `fused_nr` is therefore false on `gfx1151`,
the split path row-slices `kv_a_proj_with_mqa`, and a `q8_0` weight has no row
slice — so GLM-5.3's first forward throws. The throw's own comment
(`:628-631`) says the only way to reach it is `VT_MLA_FUSED_NORM_ROPE=0`. That
enumerates the backends that HAVE the op and forgets the ones that do not; the
measured run in #2564 is the counterexample.

## Why repair 1, and not 2 or 3

#2564 prices three repairs. This spec takes the first.

1. **Port `kFusedNormRope` to ROCm** — taken. Both halves of the composite are
   already native ROCm kernels: the latent RMS reduction is
   `rocm_rmsnorm.hip`'s `RmsNormRowKernel` and the decoupled-pe rotation is
   `rocm_dense_basic.hip`'s `RopeFromCacheK`. The port is the same
   composition CUDA already makes, it makes the predicate true honestly, it
   runs the work on the device rather than the host, and it decrements the
   reference-tier hit count `docs/ROCM.md:60-61` gates on.
2. **Let `fused_nr` consider the reference tier** — rejected. It changes what
   "available" means at a shared seam for every backend and every op, and
   `src/vt/op_provider.cpp:779-786` states the contract it would break: a
   unified accelerator would report every op registered the moment its fallback
   installed, and the fused-recipe ladder would stop choosing its portable
   composite path. It also keys naturally on host-addressability, and the two
   host-addressability predicates on this board answer differently:
   `DeviceMemoryIsHostAddressable()` is true (`rocm_backend.hip:371`, which is
   what makes the reference tier eligible) while
   `HostMemoryIsDeviceAddressable()` is false, because `gfx1151` reports
   `pageableMemoryAccess=0` (#2515, measured twice on hardware). A predicate
   written against the wrong one of those two reads plausible and answers
   backwards on the only board that can test it.
3. **Teach the split path to slice a block-quantized merged row** — rejected
   for this wave. It is the largest of the three and it repairs a fallback that
   nobody wants taken: the fused arm is bit-identical and one launch cheaper.
   It stays owed, because a backend that registers neither op still needs it.

## Upstream anchors

Read on the pinned oracle, `~/_git/vllm` @ `5559679229`
(`.agents/upstream-sync.md`). vLLM composes the same two steps, unfused, in the
MLA A-projection:

- `vllm/model_executor/layers/mla.py:164-165` — `kv_c, k_pe =
  kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)` and then
  `kv_c_normed = self.kv_a_layernorm(kv_c)`. That is the LATENT half of the
  fused kernel, and the split it performs is exactly the row split the local
  block cannot make on a block-quantized weight.
- `vllm/model_executor/layers/mla.py:175-177` — `self.rotary_emb(positions,
  q[..., self.qk_nope_head_dim:], k_pe)`. That is the ROPE half, applied to the
  trailing slice of the SAME merged row and to nothing the latent half touches,
  which is why fusing the two is arithmetically inert.
- `vllm/model_executor/models/deepseek_v2.py:512-518` — the merged weight is
  `ReplicatedLinear(hidden_size, kv_lora_rank + qk_rope_head_dim)`, which fixes
  the `[L + R, H]` row shape both arms of the local branch assume.
- `vllm/model_executor/models/deepseek_v2.py:1930` — `class
  GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`, which
  `vllm/model_executor/models/registry.py:117` maps the checkpoint's
  architecture string to. **Not `deepseek_v32.py`:** at this pin that file does
  not carry the registration, and a reader sent there finds nothing.

vLLM has no fused kernel for the pair, so upstream is the BEHAVIOURAL reference
and the in-tree CUDA sibling is the STRUCTURAL one, exactly as
`vt::FusedNormRope`'s own contract already records
(`include/vt/ops.h:3388-3410`).

Ported `file:line`, verbatim composition:

- `src/vt/cuda/cuda_ops.cu:1163-1245` — `FusedNormRopeKernel`,
  `LaunchFusedNormRope`, `FusedNormRopeKernelCuda`.
- `src/vt/rocm/rocm_rmsnorm.hip:67-98` — the `__syncthreads()` shared-memory
  tree reduce and the scale loop, reproduced element for element. It is
  wavefront-width agnostic, which is the property that makes it portable to a
  64-lane wavefront at all.
- `src/vt/rocm/rocm_dense_basic.hip:607-635` — the cache read and the
  neox/gpt-j pairing of `RopeFromCacheK`, reproduced element for element.

## Design

One new translation unit, `src/vt/rocm/rocm_mla_fused_norm_rope.hip`, holding
one kernel and its `FusedNormRopeFn` entry point, registered in
`rocm_ops.hip`'s `Registrar` and listed twice in `CMakeLists.txt` (the source
list and the `HIP_ARCHITECTURES` property list).

Block width stays 256, as `rocm_rmsnorm.hip:41` fixes it and for the reason
stated there: it is four whole wavefronts AND it keeps the reduction order
identical to the CUDA and CPU siblings, which is what keeps the NMSE bar
meaningful. Grid is one block per token, as CUDA's is.

The two halves address disjoint dims, so the fused output is the composite of
`RmsNorm(x[:, :off])` and `RopeFromCache(x[:, off:])` by construction, not by
inspection.

## Risks

- **A wavefront assumption.** The donor reduction uses no warp-level primitive,
  so 64-lane wavefronts are safe; the gate below measures it rather than
  asserting it.
- **A partial MLA arm hides which half ran.** With the reference tier eligible
  a half-ported arm still emits tokens. Mitigated by requiring
  `VT_OP_PROVIDER_STATS=1` on every leg and reporting the reference-tier hit
  count beside any token (#2505's silent-fallback failure).
- **No speed claim is admissible** from this board for this model while the
  hit count is non-zero (`docs/ROCM.md:60-61`). None is made.

## Tests

- `tests/vt/test_backend_cross_device.cpp` — a new `FusedNormRope` case,
  against the CPU oracle at NMSE <= 5e-4, in both rope styles and both
  dtypes, on every backend that registers the op. It is SKIPPED on a build
  where no device registers `kFusedNormRope`, which is stated plainly rather
  than counted as a pass.
- The e2e leg on `strix:gpu0` is the reachability gate, through the production
  entry point (`vllm-cli` -> `vllm_engine_load` -> `ModelRegistry::Forward`),
  never a by-hand construction.

## Gates

1. `ctest` for the focused unit target on `strix:gpu0`, case AND assertion
   counts both read.
2. GLM-5.3 `UD-IQ1_S` through `vllm-cli --device auto` with `VT_CPU_MOE=1`,
   greedy, on `strix:gpu0`: generated text printed verbatim, with the
   reference-tier hit count beside it.
3. Reachability mutation: remove the `RegisterOp(OpId::kFusedNormRope,
   DeviceType::kROCM, ...)` line, REBUILD, rerun the e2e leg. It must throw
   #2564's message again. Restore, verify by sha256, rebuild, rerun.

## Evidence

All on `strix:gpu0` (`gfx1151`, Radeon 8060S, ROCm `7.2.53211-97f5574fe2`),
under `rc` leases, from tree `b413e323be50822dcbecb30bd61dc90333a416b5`. The
tarball's sha256 was read on both the host and the worker and the two agree
(`cf3841a4983b54b8b972ef06ba43f84034477a60358b984ad6a189ba4bf8210b`).

**Build.** `ninja rc=0`. `[555/577] Building HIP object
CMakeFiles/vllm.dir/src/vt/rocm/rocm_mla_fused_norm_rope.hip.o` -- the first
compile this TU has ever had.

**Gate 1, the focused numeric case** (job `6b35b8d3`): `1 test case | 1 passed |
0 failed`, `20 assertions | 20 passed | 0 failed`.

*The assertion count is the discriminator, not a grep of the trace.* `CAPTURE`
prints only on failure, so grepping the passing run for `ROCM` returns 0 and
proves nothing -- that instrument was useless and is recorded as such. The same
binary's case on a CPU-only build with no accelerator registered runs **2**
assertions. 20 vs 2 is the ROCm arm executing: five `Upload` REQUIREs, two
`CHECK`s and two REQUIREs inside `Nmse`, per rope style.

**Gate 2, GLM-5.3 e2e through the production entry point** (job `6b35b8d3`):
`VT_CPU_MOE=1 VT_OP_PROVIDER_STATS=1 vllm-cli --model <derived shard 1>
--device auto --prompt "The capital of France is" --max-tokens 4
--temperature 0`, `LEG rc=0`,
`prompt_tokens=5 completion_tokens=4 finish_reason=length`, stdout:

```text
 Paris, which is
```

`op=114 device=5 selected=vt-native` -- op 114 is `kFusedNormRope`, device 5 is
`kROCM`. The mapping is cross-checked three ways against the run's own named
lines: 29/`ConcatAndCacheMla`, 99/`ConcatMlaNopeRope`, 33/`MlaPrefillAttention`.

**Five distinct ops served from the portable reference tier** in that run:
`ConcatAndCacheMla`, `ConcatMlaNopeRope`, `MlaPrefillAttention`,
`BatchedMatmul`, `MlaDecodeAttention`. `kFusedNormRope` is not among them.
**No speed number is admissible and none is offered** (`docs/ROCM.md`).

**Gate 3, the mutation ladder** (job `8b508c69`), each rebuilt and each restored
before the next:

| Mutation | Build | Result |
|---|---|---|
| M1: delete the `RegisterOp(kFusedNormRope, kROCM)` line | rc=0 | `LEG rc=1`, the #2564 throw reproduced verbatim. KILLED |
| M2: drop the sin term from the rope half | rc=0 | `GATE1 rc=1`, `1 case failed`, `20 assertions | 18 passed | 2 failed`. KILLED |
| restored control | rc=0 | `pre.sha == post.sha` byte-for-byte on both files; `GATE1 rc=0`, 1 case / 20 assertions |

M1 is simultaneously the RED and the reachability proof. The mutated tree is
behaviourally the base tree at the branch this change repairs, and deleting the
production call site reds the production gate -- which is what
`.agents/reachability.md` asks for and what a by-hand construction cannot show.
Its throw also reads back the corrected message: *"The fused path was not taken
because this backend (rocm) registers NO NATIVE vt::FusedNormRope kernel, and
vt::OpRegistered is a native-only probe that cannot see the portable reference
tier"*.

**Full cross-device suite:** `29 test cases | 28 passed | 1 failed`,
`80296 assertions | 80295 passed | 1 failed`. The one failure is
`MoeSiluMul matches the CPU oracle`'s bit-exact bf16 `CHECK(got == ref_b)`,
which is the standing red #1954 already tracks on `gfx1200`, now recorded on
`gfx1151` too. This change adds one TU and one registration and touches no MoE
path. It is **not** measured at the base commit on this board, so "pre-existing"
is argued from the absence of code-path overlap rather than from an A/B.

**Host-side control:** on a CPU-only build of the same tree,
`test_mla_attention_block` runs 21 cases / 2,282,067 assertions green, and the
whole cross-device suite runs 28 cases / 13 assertions green.

## Stop conditions

- The board faults or resets in a way that is a property of the board rather
  than of this change (#2546 measured 12/12 GPU resets for a gate-sized native
  run). Report it as such; do not paper over it.
- A second MLA op turns out to block GENERATION rather than merely make it
  slow. Return `NEEDS_DECISION` naming which ops are in which class.

## Owed

- The seven remaining MLA/DSA ops on ROCm — `kFusedChain`, `kBatchedMatmul`,
  `kConcatAndCacheMla`, `kConcatMlaNopeRope`, `kDsaIndexerLogits`,
  `kDsaTopkSelect`, `kGatherMlaCache`, `kMlaDecodeAttention`. Each has a CPU
  registration, so each serves from the reference tier on this host-addressable
  board and none of them refuses. They are what makes a speed result
  inadmissible here. Owned by `BACKEND-ROCM`, recorded in
  `.agents/specs/rocm-glm53-dsa.md` W1.5 as a campaign this wave does not open.
- Repair 3 of #2564 — a block-quantized row slice for the split path — for a
  backend that registers neither `kFusedNormRope` nor a native alternative.

## RETRACTION — the tier is WITHDRAWN on gfx1151, so the remaining ops REFUSE

**Measured on `strix:gpu0`, 2026-09-03**, by a case that asks the predicate rather
than assuming an answer (`tests/vt/test_backend_cross_device.cpp`, landed with
#2843):

```
ROCm reference tier: UnifiedMemory=0 DeviceMemoryIsHostAddressable=0
                     ReferenceTierEligible=0
VERDICT: the tier is WITHDRAWN here — the missing MLA ops are a REFUSAL,
         so W2/W3 block GENERATION and not only measurement
```

**This falsifies the framing of this spec and of the `FEATURES.md` row landed at
`fbdbe663d`.** Both said the remaining MLA/DSA ops make GLM-5.3 *slow* on this
board. They do not. They **refuse**, so GLM-5.3 non-flash does **not generate text
on `gfx1151` at current `main`.**

**The earlier result was not wrong; it was true of a different tree.** The run that
emitted ` Paris, which is` (n=2, reproduced on a separately rebuilt binary) is real
and is recorded. It ran on a tree that did not contain
[`6b97a6800`](https://github.com/mudler/vllm.cpp/commit/6b97a6800) (#2511), which
narrowed managed allocation to `PageableMemoryAccess == 1`.
`git merge-base --is-ancestor 6b97a6800 9f3e6e223` returns **false**, and
`6b97a6800` **is** an ancestor of `main`. `gfx1151` reports 0, and
`docs/ROCM.md:83-85` already states that such a part gets plain `hipMalloc` and no
reference tier.

**What still holds, and what does not.** The `kFusedNormRope` port is unaffected —
it is a real native ROCm kernel and remains registered. Four more ops landed at
`928d89a9b` (#2715). What is falsified is the claim that the REMAINING ops are a
performance concern.

**CORRECTED, and the first version of this entry got it wrong twice.** It is
**FOUR** ops, not three — `kMlaPrefillAttention`, `kMlaDecodeAttention`,
`kDsaIndexerLogits` and `kDsaTopkSelect` each have **zero** registrations under
`src/vt/rocm/` at `main`, verified by count. The narrower set of **three**
that `rocm_ops.hip:283` calls "gating" is the *consulted-before-the-call* set, and
that distinction only means anything on a board where the tier exists — which this
one is not. And `ReferenceTierEligible` gates on `DeviceMemoryIsHostAddressable()`,
not on `UnifiedMemory()` (`op_provider.cpp:203-205`); the probe prints both, which
is what made the mis-attribution easy.

**One more precision, because the retraction overstated what it was retracting.**
The prior record did not say these ops make GLM-5.3 *slow*. It said they disqualify
a speed *number*, which was correct for a board whose tier was eligible. What
changed is that the premise is gone — a stronger correction than the one first
written here, and stated so rather than left flattering.

**The lesson worth encoding, because this is the second instance in one day.** A
commit that lands AFTER a measurement can invalidate that measurement's premise,
and no gate in this tree notices. The first instance was a merge falsifying a PR's
own prose; this one is a correctness fix silently withdrawing a capability another
row had just measured. Filed as [#2848](https://github.com/mudler/vllm.cpp/issues/2848).
**A recorded result is only true of the tree it ran on, and the tree must be named
beside it.**

**Owed:** the three remaining ops, which are now generation-blocking rather than
speed-blocking on `gfx1151` — #2715's successor waves W2 and W3.
