# ROCM-MLA-ATTENTION — the two attention ops GLM-5.3 non-flash refuses on `gfx1151`

Row: `BACKEND-ROCM`
Issue: [#2926](https://github.com/mudler/vllm.cpp/issues/2926) (the two
kernels; CLOSED by `967c1906f`) and
[#2965](https://github.com/mudler/vllm.cpp/issues/2965) (generation, which
they did not restore; OPEN, and it owns everything under `## Owed`)
Base SHA: `d023e3357b907927fb6d459f83d21b4729b78d84`

## Now

`ACTIVE`. The kernels are registered, native, gated and REACHED on the model,
and all FIVE numeric guarantees are killed by the gate rather than by the
compiler. **GENERATION IS RE-ESTABLISHED**: rc job `499b30ae` emitted
` Paris, which is` at exit 0 with `reference-tier hits: 0` and all eighteen
reached ops native -- byte-identical to the pre-#2511 output. Read the
configuration with the result: that leg needed
`VT_ROCM_MANAGED_ALLOC=1 AMD_SERIALIZE_KERNEL=3 HSA_ENABLE_SDMA=0`, which is
DIAGNOSTIC and not the shipping default, and the default-allocator arm still
times out. NO speed number, and the `tok_s=0.001` the leg prints must never be
quoted. See `## Generation is re-established` and `## Owed`.

## Scope

Register native `DeviceType::kROCM` kernels for `vt::MlaPrefillAttention` and
`vt::MlaDecodeAttention`, so GLM-5.3 non-flash generates text again on
`strix:gpu0`.

Out of scope, and each recorded under `## Owed`: `kDsaIndexerLogits`,
`kDsaTopkSelect`, `kMergeAttnStates` and `kFusedChain`. §Reached set says why
none of them blocks generation on this workload.

## The reached set, and how it was established

**Generation is all-or-nothing here.** With the reference tier withdrawn, any
ONE unregistered op the forward reaches refuses the step. So the set that
blocks generation is what has to be priced, not the set that is unregistered.

Four ops have **zero** registrations under `src/vt/rocm/` at the base SHA:
`kMlaPrefillAttention`, `kMlaDecodeAttention`, `kDsaIndexerLogits`,
`kDsaTopkSelect`. Only the first two are reached.

**The primary evidence is a measurement, not a reading.** `9f3e6e223`'s GLM-5.3
run on this board, recorded in `.agents/specs/rocm-fused-norm-rope.md`
§Evidence, names the FIVE ops it observed on the then-eligible reference tier:
`ConcatAndCacheMla`, `ConcatMlaNopeRope`, `MlaPrefillAttention`,
`BatchedMatmul`, `MlaDecodeAttention`. That is the model's own forward,
enumerated by `VT_OP_PROVIDER_STATS=1` on the production entry point, on this
part, on this prompt. Three of the five landed native at `928d89a9b` (#2715
W1). **The two that remain are `kMlaPrefillAttention` and
`kMlaDecodeAttention`.**

The code agrees, and says why the others are absent:

| Op | Reached? | Why |
|---|---|---|
| `kMlaPrefillAttention` | **YES, first** | `mla_attention.cpp` §5a `ForwardMlaPrefillMha`, on every prefill step |
| `kMlaDecodeAttention` | **YES, second** | `src/vllm/v1/attention/backend.cpp:343`, from §5b, on every decode step |
| `kDsaIndexerLogits` / `kDsaTopkSelect` | no | guarded by `run_indexer = dims.has_indexer() && sparse_step`, and `sparse_step = !meta.indexer_cu_seqlens_q.empty()` (`mla_attention.cpp:493-494`). That mirrors upstream's `use_dense_mha = prefill_max_seq_len <= self.topk_tokens` (`sparse_mla_attention.py:296-299`): below the threshold the top-k selects every causal candidate and DENSE attention is upstream's own answer. A short prompt is a dense step. |
| `kGatherMlaCache` | no (and already native) | chunked context only (`has_context = !meta.chunks.empty()`); a fresh prompt has none. Landed at `928d89a9b`. |
| `kMergeAttnStates` | no | same chunked-context branch |
| `kFusedChain` | no | Tier-1 interpreter, reached only at `VT_FUSED_TIER=1` (`src/vt/ops.cpp:1287-1298`, `include/vt/fused_recipe.h:176-179`), which is a rejected performance arm on every backend that measured it |

**ORDER.** `kMlaPrefillAttention` refuses first: the first forward of a fresh
prompt is prefill-only (`decode_toks == 0`), so §5b never runs. Landing prefill
alone therefore moves the failure from step 1 to step 2, which is visible
progress and is what the gate ladder below measures.

## Why it refuses rather than running slowly

`6b97a6800` (#2511) narrowed managed allocation to `PageableMemoryAccess == 1`
and made the unified claim FOLLOW the allocator
(`include/vt/rocm/rocm_arch.h:180-195`). `gfx1151` reports 0.
`docs/ROCM.md:83-85` states that such a part gets plain `hipMalloc` and no
reference tier. Measured by the probe case landed with #2843:

```text
ROCm reference tier: UnifiedMemory=0 DeviceMemoryIsHostAddressable=0
                     ReferenceTierEligible=0
VERDICT: the tier is WITHDRAWN here — the missing MLA ops are a REFUSAL,
         so W2/W3 block GENERATION and not only measurement
```

`9f3e6e223`'s ` Paris, which is` is not falsified — it is true of a tree that
does not contain `6b97a6800`. `git merge-base --is-ancestor 6b97a6800
9f3e6e223` answers no, and `6b97a6800` IS an ancestor of `main`.

## Pricing — and the correction it makes to the campaign spec

`.agents/specs/rocm-mla-dsa-ops.md` prices `kMlaDecodeAttention` at ~600 HIP
lines, **high risk**, from a 560-line CUDA donor. That price is for porting the
CUDA arm, which is the two-stage split-KV flash decode
(`cuda_mla_attn.cu:140-750`), and the four risks it names are all properties of
the SPLIT: a >48 KiB dynamic shared-memory request with no HIP opt-in, a
`multiprocessor_count` probe `DeviceCaps` does not carry, a grow-only
capture-safe `[B, Hq, splits, Dv+1]` workspace, and a sink that stage 2 must
count once per row rather than once per split.

**None of those is a property of the OP.** vLLM's behaviour is the MQA decode
over the compressed latent (`triton_mla.py:189-260 forward_mqa`), and this tree
already carries a second, independent port of it that has none of the four
risks: `src/vt/cpu/cpu_mla_attn.cpp`, itself a port of upstream's OWN CPU
kernel `vllm/csrc/cpu/mla_decode.cpp` (`mla_decode_kvcache_cpu_impl`). A ROCm
arm written to that shape — one block per `(request, head)`, one streaming
online-softmax pass over the keys — is a native kernel that computes the same
function, and the split disappears along with every one of its four risks.

| Op | Shape | HIP est. | Test est. | Risk |
|---|---|---|---|---|
| `kMlaPrefillAttention` | one block per `(query row, head)`, streaming online softmax over the visible keys; causal bottom-right bound + optional sliding window | ~150 | ~120 | low |
| `kMlaDecodeAttention` | one block per `(request, head)`, streaming online softmax over the block-table walk; window + DSA selection + attention-sink arms | ~180 | ~150 | low-medium — three optional arms, each its own gate |
| **Total** | | **~330** | **~270** | |

The two kernels are the SAME kernel with two different key enumerations, which
is why they share one translation unit.

**What this costs.** The split-KV schedule exists to fill a GPU when
`batch * heads` is small; a single-pass kernel leaves the tail of a small decode
under-occupied. That is a SPEED property, and no speed number is admissible
from this board for this model anyway (`docs/ROCM.md:60-61` — and see §Owed:
the DSA pair is still unregistered, so a sparse run would still refuse). The
split stays owed as a performance wave, named rather than silently skipped.

**This spec does not retract the campaign spec's estimate.** It measured the
CUDA port, and it is right about the CUDA port. What it did not price is the
CPU port, which is the cheaper of the two mirrors and reaches the same
behaviour.

## Upstream anchors

vLLM at the pinned oracle `5559679229` (`.agents/upstream-sync.md`).
`GlmMoeDsaForCausalLM` is `vllm/model_executor/models/deepseek_v2.py:1930`
(**not** `deepseek_v32.py`; at this pin that file carries no registration).

| Op | Upstream behaviour | Structural donor in this tree |
|---|---|---|
| `kMlaPrefillAttention` | `vllm/v1/attention/backends/mla/prefill/flash_attn.py:153-248` — `_flash_attn_varlen_diff_headdims`, `run_prefill_new_tokens` (causal) and `run_prefill_context_chunk` (non-causal). The V zero-pad to the QK width and the output slice-back (`:164-168`, `:196-197`) are a launcher detail; computing at the true widths is the same number. | `src/vt/cpu/cpu_mla_prefill.cpp` (the numerics), `src/vt/rocm/rocm_mla_fused_norm_rope.hip:98-116` (the shared-memory reduction tree) |
| `kMlaDecodeAttention` | `vllm/v1/attention/backends/mla/triton_mla.py:189-260` `TritonMLAImpl.forward_mqa`, which passes ONE buffer as both K and V with `is_mla=True` (`:236-244`); NUMERICS from `vllm/csrc/cpu/mla_decode.cpp` `mla_decode_kvcache_cpu_impl` | `src/vt/cpu/cpu_mla_attn.cpp` (the numerics), `src/vt/cuda/cuda_mla_attn.cu:255-266,296-320,398-407` (the three optional arms' exact rules) |

**There is no vendored FlashAttention to mirror on ROCm**, and that is the one
place this port is not a transcription. `cuda_mla_prefill.cu`'s
`kMlaPrefillAttention` arm is inside `#ifdef VLLM_CPP_FLASH_ATTN` and is the
FA-2 launcher, so the ROCm arm is written against the CPU reference — exactly as
`.agents/specs/rocm-mla-dsa-ops.md` §Owed already records it must be.

## Design

One new translation unit, `src/vt/rocm/rocm_mla_attn.hip`, two kernels, two
entry points, two `RegisterOp` lines in `rocm_ops.hip`, two `CMakeLists.txt`
entries (the source list and the `HIP_ARCHITECTURES` property list).

Both kernels share one body shape:

- **Grid** is one block per output row. Prefill: `(total_q, num_heads)`.
  Decode: `(batch, num_heads)`. **Block** is 256 threads — four whole
  wavefronts, the width `rocm_rmsnorm.hip:41` fixes and for the reason stated
  there.
- **Dynamic shared memory** holds `q[head_dim]`, `acc[v_head_dim]` and a
  `float[256]` reduction scratch. At GLM-5.3 geometry that is
  `(576 + 512 + 256) * 4 = 5,376 B`, an order of magnitude under the 48 KiB
  every architecture guarantees. This is the whole reason the CUDA arm's
  `cudaFuncSetAttribute` risk does not appear: the split kernel tiles
  `(kBlockH + kNTile)` whole rows into shared memory and this one holds one.
- **The dot product** is the `__syncthreads()` shared-memory tree reduce
  `rocm_rmsnorm.hip` and `rocm_mla_fused_norm_rope.hip` already use. It uses no
  warp-level primitive, so it is wavefront-width agnostic — the property that
  makes it correct on a 64-lane wavefront without a `warpSize` assumption.
- **The softmax is streaming online**, key by key, in the CPU reference's own
  order: `m_new = max(m, qk)`, `rescale = isinf(m) ? 0 : exp(m - m_new)`,
  `acc = acc * rescale + p * v`, `l = l * rescale + p`. Since the key loop is
  sequential within the block, the reduction order over KEYS is identical to
  the CPU oracle's, and only the order over the head dimension differs. That is
  what makes an NMSE bound the right gate rather than a hope.

Three arms carry over verbatim, each cited to the line that defines it:

1. **Sliding window.** Prefill: `first = clamp(iq + causal_shift - win_left)`
   against the same bottom-right position the causal bound uses
   (`cpu_mla_prefill.cpp`, the `first` computation). Decode:
   `j_start = max(0, seq_len - 1 - win_left)` (`cuda_mla_attn.cu:196`).
   `win_left < 0` is the absent state and restores the byte-identical full
   range.
2. **DSA selection** (decode only). `sel`/`sel_cnt` present maps visit index
   `i` to token position `sel[b * sel_s0 + i]`, count clamped
   `min(topk, max(0, sel_cnt[b]))` — the kernel contains what `ops.cpp` could
   not validate on device memory (`ops.cpp`'s own note). A `-1` or
   out-of-range position scores nothing. **This mirrors the CUDA arm, not the
   CPU one**, and the divergence is upstream of this change: on an
   out-of-range position `cpu_mla_attn.cpp` REFUSES by name where
   `cuda_mla_attn.cu:281-289` returns a number. A device kernel cannot throw,
   so the CUDA rule is the only one this arm can take, and the divergence stays
   recorded in `.agents/specs/dots3-note.md` `## Owed` where it already lives.
3. **Attention sink** (decode only). Seeded ONCE per row: `m = sink[h]`,
   `l = 1`, `acc = 0` (`cpu_mla_attn.cpp`, and `cuda_mla_attn.cu:398-407` for
   why the split arm has to put it in stage 2 instead). A single-pass kernel has
   no per-split double-count to avoid, which is the second of the four W3 risks
   dissolving rather than being solved.

`MlaDecodeAttentionArgs::num_kv_splits` is **ignored** by this arm and that is
stated rather than implied: it is a scheduling hint for a split kernel, and
`ops.cpp` validates only `>= 0`. The CUDA arm's own heuristic overrides it too
when it is 0.

Dtypes are `f32` and `bf16`, the two `rocm_mla_fused_norm_rope.hip` carries and
the two a GLM-5.3 GGUF arm produces. `f16` refuses BY NAME rather than
silently, so a caller meets a message and not a wrong answer.

## Reachability

The production entry points are
`src/vllm/model_executor/layers/attention/mla_attention.cpp` §5a
(`ForwardMlaPrefillMha` -> `vt::MlaPrefillAttention`) and
`src/vllm/v1/attention/backend.cpp:343` (`vt::MlaDecodeAttention`), both reached
from `ModelRegistry::Forward` through `vllm_engine_load` and `vllm-cli`.
Nothing is added to the model layer: this change alters which implementation
`GetOp` returns for a call the model already makes on every MLA step.

The mutation that proves it is deleting each `RegisterOp` line, REBUILDING, and
rerunning the e2e leg: it must refuse again, by name, exactly as `main` does.

## Tests and gates

New arms in `tests/vt/test_backend_cross_device.cpp`, in the file's existing
style. Each asserts THREE things, and the middle one is load-bearing:

1. The device result matches the CPU oracle at NMSE <= 5e-4.
2. `vt::OpRegistered(op, device)` is **true**. `OpRegistered` is a native-only
   probe (`src/vt/op_provider.cpp:788-806`), so it is the only one of the three
   that can tell a native kernel from the reference tier.
3. `vt::GetReferenceTierHits()` does not increase across the call — the same
   quantity `docs/ROCM.md:60-61` disqualifies a performance result on.

**The #2715 wave's own trap is why assertion 2 exists.** Four of its RED cases
PASSED on their `REQUIRE` guards alone, because the device loop bodies never
ran and the oracle-equality assertion is green with no kernel at all. On this
board the tier is withdrawn, so the shape is different — but the RED must still
be read as BOTH counts, and a non-zero assertion count is what proves the cases
asserted rather than threw.

Arms:

- Prefill, causal, multi-request varlen with UNEQUAL query and key lengths, so
  the bottom-right `causal_shift` is non-zero and a top-left implementation
  fails.
- Prefill, non-causal (the context-chunk call, `flash_attn.py:246`).
- Prefill, sliding window.
- Prefill LSE against the oracle's `[num_heads, total_q]` layout.
- Decode, dense, block table with SHUFFLED blocks so the walk is exercised.
- Decode, sliding window.
- Decode, DSA selection, including the FULL-selection identity case: a
  selection naming every causal key must reproduce the dense answer.
- Decode, attention sink.
- Decode LSE.

Gate ladder on `strix:gpu0`, in this order, because each step moves the failure
further along:

1. Focused unit target, case AND assertion counts both read.
2. GLM-5.3 through `vllm-cli` on the production entry point at the tree with
   ONLY the prefill kernel: the refusal must move from `MlaPrefillAttention` to
   `MlaDecodeAttention`.
3. The same with both kernels: **generated text printed verbatim**, with
   `GetReferenceTierHits()` beside it.
4. Mutation ladder, each REBUILT and each restored by sha256.

## Risks

- **A long prefill is slow.** One block per `(query row, head)` with a
  whole-block reduction per key is `O(len_k)` block reductions per output row.
  It is correct and it is not the split schedule. No speed number is admissible
  here, and the split stays owed.
- **`MoeSiluMul` bf16 is a standing red on this board** (±1 ULP vs the CPU
  oracle, #1954/#1513). It is pre-existing, this change touches no MoE
  arithmetic, and it is REPORTED rather than tolerated into green.
- **`-Werror` reaches HIP translation units** since `6f6caa725`, so a
  diagnostic in the new file is a build failure. A diagnostic in code this
  change did not write is `main` red under its own flag and is reported as
  such.

## Evidence (`strix:gpu0`, gfx1151, ROCm 7.2.4, rc jobs `943ca573` and `a77aa104`, tree `56fd248e9`)

`-DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151`, `ninja -j 4`, no
ccache. Three earlier jobs bought nothing and each failure was in the harness,
not the tree: `/opt/rocm/bin` off `PATH`; CMake 3.28 refusing the `hipcc`
wrapper by name and wanting `/opt/rocm/llvm/bin/clang++`; and `/opt/rocm/lib`
absent from the library search path, which made `libvllm.so` link and only
`vllm-cli` fail with 71 `undefined reference to ...@hip_4.2`. Recorded because
the next reader on a fresh worker will meet all three.

**Build.** `NINJA rc=0`, zero warnings with `-Werror` reaching HIP translation
units (`6f6caa725`). `[566/583] Building HIP object
CMakeFiles/vllm.dir/src/vt/rocm/rocm_mla_attn.hip.o` -- this TU's first compile.

**The reference tier, MEASURED at the seam rather than read off `rocm_arch.h`:**

```text
PROBE_TREE_BASE_SHA=56fd248e9909e0d68381b92fd0c29d64ac0618a5
ROCm reference tier: UnifiedMemory=0 DeviceMemoryIsHostAddressable=0 ReferenceTierEligible=0
CHECK_THROWS( (void)vt::GetOp(vt::OpId::kDsaIndexerLogits, DeviceType::kROCM) ) threw as expected!
CHECK( vt::GetReferenceTierHits() == before ) is correct!  values: CHECK( 0 == 0 )
VERDICT: the tier is WITHDRAWN here
```

An older `strix` log describing the opposite carries no base SHA and cannot be
dated. This one can, which is the whole point of printing it.

**Gate 1**, before and after the whole mutation ladder:
`4 test cases | 4 passed | 0 failed | 37 skipped`,
`119 assertions | 119 passed | 0 failed`. The probe-only case runs **2**
assertions; 119 against 2 is the device loop bodies executing, which is the
#2715 trap measured rather than argued.

**Reachability, on the real model through the production entry point.** In the
unmutated leg, `op=33 device=5 selected=vt-native priority=0 registered=1` --
id 33 is `kMlaPrefillAttention` and device 5 is `kROCM`, both read from
`include/vt/ops.h` and `include/vt/device.h` rather than from prose. Deleting
both `RegisterOp` lines (MUT-R1, which IS `main`'s behaviour) reproduces the
refusal:

```text
engine-fatal: EngineCore busy loop threw: vt: no kernel for op MlaPrefillAttention
(id 33) on device rocm (type 5), and the portable CPU reference tier is NOT eligible
```

**NO TOKEN WAS OBSERVED, and the timeout is the reason to suspect first.**
Heartbeat bracketing of the job's own log:

| leg | wall | rc |
|---|---|---|
| green (both kernels) | ~50 min | 124 (`timeout 3000`) |
| MUT-R1 (both deleted) | **~41 min** | 1 (refused) |
| MUT-R2 (decode deleted) | ~49 min | 124 |

MUT-R1 refuses at the FIRST unregistered MLA op, so its ~41 min is the LOAD
alone -- against a managed-era reference of 1372 s. The green leg's 3000 s
budget therefore left roughly **9 minutes of forward**, which is not evidence
about the kernels. The load itself COMPLETES under plain `hipMalloc`
(`device placement INSTALLED`, the KV auto-fit line, then `op-provider` lines
that only issue from inside the forward), so the `## Owed` "GLM-5.3 loads on
gfx1151" claim survives; what is unproven is generation.

**Numeric mutations, and three of five prove NOTHING.** A mutation the compiler
rejects is not a killed mutation, and the job's own summary line flattened the
two cases into one number:

| # | mutation | build | gate | verdict |
|---|---|---|---|---|
| N1 | prefill drops the bottom-right causal shift | rc=1 -> **rc=0** | rc=1, `117 assertions | 113 passed | 4 failed` | **KILLED** (job `a77aa104`) |
| N2 | decode drops the attention-sink seed | rc=0 | rc=1, `119 assertions | 115 passed | 4 failed` | **KILLED** |
| N3 | decode ignores the sliding-window start | rc=1 -> **rc=0** | rc=1, `117 assertions | 115 passed | 2 failed` | **KILLED** (job `a77aa104`) |
| N4 | decode ignores the DSA selection list | rc=1 -> **rc=0** | rc=1, `117 assertions | 113 passed | 4 failed` | **KILLED** (job `a77aa104`) |
| N5 | drop the online-softmax rescale, BOTH kernels | rc=0 | rc=1, `119 assertions | 103 passed | 16 failed` | **KILLED** |

**All five guarantees are now proven by the test.** In job `943ca573` three of
the five proved nothing, because each deleted the last use of an operand and
`-Werror,-Wunused-*` killed the build; a mutation the COMPILER rejects says
nothing about the TEST. Job `a77aa104` re-expressed N1/N3/N4 to multiply the
operand by zero instead of deleting the reference, so every operand stays used,
the translation unit builds (`NINJA rc=0` on each), and the GATE is what has to
catch them. It did, in all three cases. The base and the restored tree both
read `3 test cases | 3 passed`, `117 assertions | 117 passed`, with
`ALL FILES RESTORED BYTE-FOR-BYTE` by sha256 between them
([job7 log](rocm-mla-attention-nonflash-job7.log)).

**Restore.** `ALL FILES RESTORED BYTE-FOR-BYTE` by sha256 against a manifest
taken before any mutation, and `GATE1[final] rc=0` on the restored tree.

**Full cross-device suite:** `41 test cases | 40 passed | 1 failed`,
`83970 assertions | 83969 passed | 1 failed`. The one failure is
`test_backend_cross_device.cpp:2278`, `CHECK( got == ref_b )` in
"MoeSiluMul matches the CPU oracle" -- the standing bf16 +/-1 ULP red
#1954/#1513. This wave touches no MoE arithmetic. It is reported, never widened
into green.

**No speed number is admissible and none is offered.** The DSA indexer pair is
unregistered, so a sparse step refuses, and `docs/ROCM.md:60-61` applies. The
ROCm GLM-5.3 speed axis stays **VOID**.

### The managed-allocator leg CRASHED, and its instrument reported success

Job `a77aa104` ran the model leg under `VT_ROCM_MANAGED_ALLOC=1` — the
pre-`6b97a6800` behaviour, which the refusal message itself names — at a 3 h
timeout. **It did not generate a token. It dumped core.**

```
LEG[managed] rc=0 wall=4118s
[  4118s] vllm-cli: loading model from .../GLM-5.3-UD-IQ1_S-00001-of-00006.gguf
[  4118s] engine: device placement INSTALLED: 78 layers ... experts on cpu ...
[  4118s] vllm.cpp: Asynchronous scheduling is disabled (max_concurrent_batches=1)
[  4118s] timeout: the monitored command dumped core
```

Read the two lines together, because they disagree. `timeout` prints
`the monitored command dumped core` when its child dies on a core-producing
signal, and `wall=4118s` is well under the 10800 s budget, so the timeout did
not fire — **the process crashed on its own**. `LEG[managed] rc=0` is not that
process's status: the runner captured `$?` after `( … | awk … ) > log`, and in
a pipeline `$?` is the LAST command's status, which is `awk`'s. The instrument
therefore printed a green rc over a core dump. This is the repository's own
`$?`-after-a-pipe failure, and it is the reason the verdict here is read off the
`timeout` line rather than off the rc.

**The load/forward split this job was built to measure was NOT measured.** Every
captured line carries the same `[  4118s]` stamp, including the first line of
the load. `vllm-cli` block-buffers its stdout when that stdout is a pipe, so the
whole stream reached `awk` at flush time and `awk` stamped it all at once. The
next attempt has to defeat the buffering (`stdbuf -oL`, or a pty) rather than
stamp downstream of it. Until then the "~41 min load" in `## Owed` stays an
observation.

**What the crash does establish** is narrow and it is a NEGATIVE: the managed
allocator is not the escape hatch. The working hypothesis that #2511's allocator
change alone explains the missing token predicted that a managed leg would
generate. It crashed instead, which is consistent with #2511's own measurement
of 17 GPU faults in 21 managed legs, and which is why that flag is a diagnostic
lever and never a shipping default. The `first op-provider line` is present
(`op=4 device=5 selected=vt-native`), so the load completed and the forward had
begun; the crash is inside the forward, not in the loader.

**LEG B, the default-allocator control, never ran.** The CIFS share dropped
after leg A (`job7.sh: error reading input file: Host is down` — `bash` reads a
script lazily by byte offset and the script lives on `/workspace`), and the job
idled to its 8 h `max_runtime` kill. So there is **no A/B**, and the allocator
is still not isolated from the kernels. A future job stages its own script into
`/tmp` before running it, exactly as it already stages the source tree.

## Stop conditions

- A numeric arm that needs a widened tolerance to pass stops the wave. The
  bound is the one every other GEMM/attention arm in this file uses.
- The board faults or resets in a way that is a property of the board rather
  than of this change (#2546 measured 12/12 GPU resets for a gate-sized native
  run). Report it as such.
- A THIRD op turns out to block generation. Return the op's name and stop;
  §Reached set would then be wrong and has to be corrected before more code.

## The allocator A/B, run at last (rc job `8c458a73`, `strix:gpu0`, 2026-09-05)

job8 ran the control leg job7 never reached, and then repeated the managed leg
with kernel serialisation. **Both exit statuses are trustworthy this time** --
the run is a plain redirect with no pipe, so `$?` is the run's own status and
not `awk`'s, which is the defect that made job7 report a core dump as `rc=0`.

| Leg | Configuration | Result | Wall |
|---|---|---|---|
| job7 | `VT_ROCM_MANAGED_ALLOC=1` | **core dump** | 4118 s |
| job8-1 | default `hipMalloc` (what we ship) | **rc=124, TIMED OUT** -- it did not crash | 9001 s |
| job8-2 | managed + `AMD_SERIALIZE_KERNEL=3` + `HSA_ENABLE_SDMA=0` | **rc=0, EXITED CLEANLY** | 6942 s |

**`vllm-cli` exits 1 on a refusal and 0 on a completed generation**, so job8-2's
`rc=0` is consistent with GENERATION, and it is the first clean exit this board
has produced on this model since `6b97a6800`. **No token is claimed, because no
token was observed** -- see the capture failure below.

### What the three legs suggest, and what they do not

The original hypothesis was that `#2511`'s allocator change alone explains the
missing token, and it predicted the managed leg would generate. job7 refuted
that: managed alone dumps core. job8-2 shows managed **plus** serialised kernels
**plus** SDMA disabled exits 0.

So the next hypothesis is that the fault is an **asynchrony or DMA race rather
than the allocator or the kernel arithmetic**: `AMD_SERIALIZE_KERNEL=3` makes
every kernel launch synchronous and `HSA_ENABLE_SDMA=0` takes the SDMA copy
engine out of the path, and between them the crash stops happening.

**That is a direction, not an attribution.** job8-2 moved TWO variables at once
against job7, so neither is isolated, and the two runs are on a contended shared
box on different days. Splitting them is one leg each and is owed.

The default-allocator leg is the one the shipping configuration uses, and it
neither crashed nor finished in 2.5 h. Whether it was making progress or wedged
is not established, for the same reason as below.

### The capture failed, and it lost exactly the result the job was run to get

Both legs' captured output is **empty**: `leg-default.stamped` and
`leg-serial.stamped` are zero bytes on the share. So "it generated" is an
inference from an exit code and not an observation, and "the default leg printed
nothing" cannot be distinguished from "the default leg's output was lost".

The defect is in the job, not in the tree. The runner copied only the STAMPED
file to `/workspace` and left the raw file on the worker's `/tmp`, so when the
lease ended the evidence went with the worker. `stdbuf -oL` was supposed to make
the stamper's timestamps real; it appears not to have reached the binary's
output at all, which is consistent with `stdbuf` setting `LD_PRELOAD` for C
stdio and not reaching a C++ `streambuf` detached by `sync_with_stdio(false)`.

**This is the third instrument failure in this row's history and the second that
destroyed a result rather than merely misreporting one** -- after `$?` across a
pipe, and the block-buffered timestamper. The pattern is the same each time: the
instrument's failure mode is indistinguishable from a real observation.

job10 repeats job8-2's exact configuration with four independent capture paths:
a PTY via `script -q -e -c` (which forces line buffering on C++ iostreams, not
only C stdio), an unconditional copy of the raw file to the share, a `cat` of
the raw file into the job's own log, and a re-echo of its tail. It also prints
`RAW BYTES CAPTURED`, so a zero reads as a CAPTURE failure rather than as
silence from the program.

## Generation is re-established on `gfx1151` (rc job `499b30ae`, 2026-09-06)

**GLM-5.3 non-flash emits a token on this board again.** This is the first one
since `6b97a6800` (#2511) withdrew the reference tier, and it closes the
question this row and #2965 were opened around.

```text
LEG[serial2] rc=0 wall=3949s
RAW BYTES CAPTURED: 3129
 Paris, which is
vllm-cli: run=1/1 finish_reason=length prompt_tokens=5 completion_tokens=4
Script done on 2026-09-06 05:01:24+00:00 [COMMAND_EXIT_CODE="0"]
--- reference-tier hits (must be ZERO on this board) ---
0
```

**The output is byte-identical to the pre-#2511 run** that `docs/FEATURES.md`
records for this artifact: ` Paris, which is`. So the model is not merely
running, it is producing the same continuation it produced before the allocator
change, on a tree carrying the native MLA kernels.

**Every op the forward reached resolved NATIVE.** Eighteen unique
`[vt op-provider]` lines, all `selected=vt-native registered=1`, and
`reference-tier hits: 0`. Among them `op=33` (`kMlaPrefillAttention`), `op=29`
(`ConcatAndCacheMla`) and `op=99` (`ConcatMlaNopeRope`) -- the ops #2954 and
#2715 W1 landed. Nothing was served from the portable CPU tier, which on this
board could not have happened anyway since the tier is withdrawn, and the zero
is the assertion that proves it rather than the assumption.

### The configuration, and why it is not a shipping claim

The leg ran under `VT_ROCM_MANAGED_ALLOC=1 AMD_SERIALIZE_KERNEL=3
HSA_ENABLE_SDMA=0`, with `VT_CPU_MOE=1` placing all 78 layers' routed experts on
the host. **That is a DIAGNOSTIC combination and not the shipping default.**
#2511 measured 17 GPU faults in 21 managed legs against 0 in 21 without, so
`VT_ROCM_MANAGED_ALLOC=1` remains a lever for debugging and never a default.

**The default-allocator arm still does not finish.** job8 leg 1 ran the shipping
configuration for the full 9000 s and timed out (`rc=124`), neither crashing nor
completing. So what is established is that the MODEL and the KERNELS can produce
a correct token on this board; what is NOT established is that the shipping
configuration can.

### NO SPEED NUMBER, and the reason is stronger than usual

The leg reports `secs=3476.252 tok_s=0.001`. **That figure is meaningless as
performance and must never be quoted.** `AMD_SERIALIZE_KERNEL=3` makes every
kernel launch synchronous and `HSA_ENABLE_SDMA=0` removes the copy engine; the
run was configured to make a fault attributable, not to go fast. The ROCm
GLM-5.3 speed axis stays **VOID**, and it would stay void even with a plausible
number, because the DSA indexer pair is still unregistered and a benchmark
prompt drives sparse steps.

### What this narrows

The crash and the timeout are now bracketed by a success:

| Leg | Allocator | Serialised | SDMA | Result |
|---|---|---|---|---|
| job7 | managed | no | on | **core dump** at 4118 s |
| job8-1 | default | no | on | **timeout** at 9001 s |
| job8-2 / job10 | managed | **yes** | **off** | **generates**, 6942 s / 3949 s |

Two variables still move together between the crashing and the generating leg,
so neither `AMD_SERIALIZE_KERNEL=3` nor `HSA_ENABLE_SDMA=0` is isolated. One leg
each is owed and is recorded below. But the direction is now measured rather
than hypothesised: the fault is in ASYNCHRONY -- serialise the launches and
remove the copy engine and a crash becomes a correct token -- and not in the
allocator alone, which was the original hypothesis and is refuted, nor in the
kernels, which produce the right answer when they are allowed to run in order.

## Owed

- **Whether GLM-5.3 generates on `gfx1151` at all after #2511.** No token has
  been observed on a tree carrying `6b97a6800`, and the leading hypothesis is
  now REFUTED rather than untested. Job `a77aa104` ran the managed-allocator
  leg the refusal message names, at a 3 h timeout; the load completed, the
  forward began (`op=4 device=5 selected=vt-native`) and the process **dumped
  core** at 4118 s. So `VT_ROCM_MANAGED_ALLOC=1` is not the way back to a token,
  which is consistent with #2511's own 17 GPU faults in 21 managed legs and is
  why that flag stays a DIAGNOSTIC LEVER and never a shipping default. The next
  hypothesis is the crash itself: it is inside the FORWARD, on a board #2546
  measured resetting 12/12 under a gate-sized native run, so separating a board
  fault from a kernel fault is the next traceable step. Owner `BACKEND-ROCM`,
  issue [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- **The default-allocator control leg RAN** (job `8c458a73`, leg 1) and timed
  out at 9000 s without crashing, `rc=124`. What is still owed is what it was
  DOING: its captured output is zero bytes, so "wedged" and "loading slowly"
  are not separated. Owner `BACKEND-ROCM`, issue
  [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- **A token was OBSERVED, and this entry is discharged.** Job `499b30ae`
  (job10) repeated job8's leg 2 with a working capture and emitted
  ` Paris, which is` at `COMMAND_EXIT_CODE=0`. See `## Generation is
  re-established` below. What remains owed from it is the DEFAULT-allocator
  arm, which still times out. Owner `BACKEND-ROCM`, issue
  [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- **Serialisation and SDMA are confounded.** job8 leg 2 moved
  `AMD_SERIALIZE_KERNEL=3` and `HSA_ENABLE_SDMA=0` together against job7's
  crashing managed leg, so neither is isolated. One leg each is owed. Owner
  `BACKEND-ROCM`, issue
  [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- **The ~41 min load under plain `hipMalloc`** is STILL an observation, not a
  measurement. Job `a77aa104` carried a per-line timestamper meant to settle it
  and the timestamper did not work: `vllm-cli` block-buffers behind a pipe, so
  every line — the first line of the load included — reached `awk` at flush and
  was stamped `[  4118s]`. Defeat the buffering at the source (`stdbuf -oL`, or
  a pty) rather than stamping downstream of it. Owner `BACKEND-ROCM`, issue
  [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- `kDsaIndexerLogits` + `kDsaTopkSelect` on ROCm — #2715's W2, still owed. Not
  generation-blocking on a dense step; a SPARSE step (a prompt longer than
  `index_topk`) still refuses. Owner `BACKEND-ROCM`, issue
  [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- `kMergeAttnStates` on ROCm — chunked-context prefill only. Owner
  `BACKEND-ROCM`, issue [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- `kFusedChain` on ROCm — non-gating (`VT_FUSED_TIER=1` only). Owner
  `BACKEND-ROCM`, issue [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- The **split-KV decode schedule** on ROCm, as a performance wave. Owner
  `BACKEND-ROCM`, issue [#2965](https://github.com/mudler/vllm.cpp/issues/2965).
- The ROCm GLM-5.3 speed axis stays **VOID**.
