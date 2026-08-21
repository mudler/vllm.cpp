# Qwen3.8-27B-FP8 (block-wise): the token gate

**Row:** `GATE-QWEN38-27B-FP8-BLOCK`. Like the other rows of this milestone
chain -- `VT-QUANT-FP8-GROUP`, `VT-MATMUL-FP8-BLOCK-REF`,
`VT-MATMUL-FP8-BLOCK-CUDA`, `MODEL-FP8-BLOCK-WEIGHT`, `MODEL-FP8-BLOCK-LINEAR`,
`MODEL-FP8-BLOCK-MERGED` -- it is a spec-owned row and carries no matrix cell of
its own. The capability it gates is recorded in `QUANT-FP8-GENERIC`
([`quantization-matrix.md`](../quantization-matrix.md)) and in the block-wise
FP8 row of [`docs/FEATURES.md`](../../docs/FEATURES.md).
**Issue:** [#1189](https://github.com/mudler/vllm.cpp/issues/1189), the
milestone after M6 -- the `## Gate design` section of that issue is this row's
brief.
**Related:** [#1437](https://github.com/mudler/vllm.cpp/issues/1437) established
that the CUDA arm computes the CPU reference's values on seven shapes, and
closed; [#1435](https://github.com/mudler/vllm.cpp/issues/1435) owns the
no-CUTLASS segfault a gate run must not trip over;
[#1166](https://github.com/mudler/vllm.cpp/issues/1166) landed the named refusal
this arm replaced; [#915](https://github.com/mudler/vllm.cpp/issues/915) gated
the **bf16** arm of the same model and is the harness this row reuses;
[#1354](https://github.com/mudler/vllm.cpp/issues/1354) records that clock
pinning is unavailable inside an `rc` lease, which is why this row makes no
speed claim.
**Lifecycle:** `BLOCKED` -- on the checkpoint, see [`## Owed`](#owed).
**Owner:** unassigned

## Scope

One thing: run `Qwen/Qwen3.8-27B-FP8` on the block-wise FP8 arm against the
pinned vLLM oracle on identical inputs, and adjudicate the tokens.

Out of scope, and each named so that nothing here is read wider than it is:

- **Every speed axis.** This row takes a lease for CORRECTNESS only. #1354
  records that `sudo nvidia-smi -lgc` is unavailable to a lease's worker pod, so
  a lease taken by this row has no clock control, and per
  [`.agents/benchmarking.md`](../benchmarking.md) no ratio derived from it would
  be defensible. **No throughput, latency or memory number is produced by this
  row, and none may be quoted from it.** The FP8 speed axes belong to a
  benchmark row that can take clock control -- today that is the
  `rc hold` plus host-shell authorization scoped to `BENCH-QWEN38-27B-SOTA`.
- The vision tower. `modules_to_not_convert` excludes every `visual.*`
  projection, so the tower is bf16 and this arm never touches it. The gate is a
  text-path gate.
- The NVFP4 and Q4_K_M arms of the same model
  ([#821](https://github.com/mudler/vllm.cpp/issues/821),
  `.agents/specs/qwen38-27b-quant-arms.md`).
- Advancing the parity pin.
- Any repair. If the gate diverges, this row reports the divergence and stops;
  a fix is a different row with a fresh implementer.

## What is already established, so that nobody re-derives it

Read `.agents/specs/vt-matmul-fp8-block-cuda.md` `## Owed` before this section;
it is the authority and this is a pointer, not a copy.

- The CUDA arm has **executed on a GB10** and matches the CPU reference on
  **seven shapes** spanning all three tile configs and both the swapped and the
  unswapped path (`5870cb2bf`, `#1437`). 136 assertions, 0 failed, 95 of them on
  the board.
- It has **no token gate**. That sentence is what this row exists to remove.
- The sm120 collective refuses a shape without complete scale blocks:
  `N % 128 != 0` or `K % 128 != 0` is refused **by name**, which is correct
  behaviour and not a defect (`#1453`).

## The checkpoint, and the pre-lease audit that says the arm can serve it

`Qwen/Qwen3.8-27B-FP8` @ `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` -- the
revision [#1189](https://github.com/mudler/vllm.cpp/issues/1189) named, asserted
again here by reading the redirect target of the unauthenticated
`resolve/main/config.json`.

Everything below was derived **without downloading the weights**, by HTTP RANGE
request over each shard's safetensors header. The method is in
[`## Evidence`](#evidence) and is reproducible in about nine megabytes of
transfer. It matters because it answers, before any lease is taken, the one
question that could have made this gate impossible.

| Fact | Value |
|---|---|
| `quant_method` / `weight_block_size` / `activation_scheme` / `fmt` | `fp8` / `[128, 128]` / `dynamic` / `e4m3` |
| shard layout | **66 shards, `layers-<i>.safetensors` + `mtp.safetensors` + `outside.safetensors`** -- NOT the bf16 repo's `model-000NN-of-00018` |
| tensors | 1606 (1199 `BF16`, 407 `F8_E4M3`) |
| bytes | 28.75 GiB total, 27.89 GiB outside the vision tower |
| `modules_to_not_convert` | 882 entries, 636 of them non-visual |

**Every one of the 407 FP8 tensors has `N % 128 == 0` AND `K % 128 == 0`. The
sm120 complete-scale-block constraint refuses NOTHING in this checkpoint.**

| projection | count | `[N, K]` |
|---|---|---|
| `linear_attn.in_proj_qkv` | 48 | `[10240, 5120]` |
| `linear_attn.in_proj_z` | 48 | `[6144, 5120]` |
| `linear_attn.out_proj` | 48 | `[5120, 6144]` |
| `mlp.gate_proj` / `mlp.up_proj` | 64 + 64 | `[17408, 5120]` |
| `mlp.down_proj` | 64 | `[5120, 17408]` |
| `self_attn.q_proj` | 16 | `[12288, 5120]` |
| `self_attn.k_proj` / `self_attn.v_proj` | 16 + 16 | `[1024, 5120]` |
| `self_attn.o_proj` | 16 | `[5120, 6144]` |
| the seven `mtp.layers.0.*` siblings | 7 | same shapes |

The merged forms `MODEL-FP8-BLOCK-MERGED` builds are servable for the same
reason, because each part is already a multiple of 128 and concatenation along
N cannot introduce a ragged block: QKV is `12288 + 1024 + 1024 = 14336`
(112 blocks) and `gate_up` is `17408 + 17408 = 34816` (272 blocks).

**The ragged tensors exist and are excluded by the checkpoint itself.** The GDN
small tensors `linear_attn.in_proj_a` and `linear_attn.in_proj_b` are
`[48, 5120]`, and `48 % 128 = 48`. They are BF16 and they are named in
`modules_to_not_convert`, so they never reach this arm. This is the
`kv_a_proj_with_mqa` class of shape -- the one that is unservable on sm120 --
and this checkpoint keeps it out of FP8. **That is the difference between this
model and DSV3, and it is why this gate is takeable at all.**

**`weight_scale_inv` ships `BF16`, not `F32`**, in all 407 cases, at shape
exactly `[ceil(N/128), ceil(K/128)]`. Confirmed at byte level rather than from
the dtype label: `layers-0`'s `mlp.gate_proj.weight_scale_inv` declares
`[136, 40]` with `data_offsets [1100608, 1111488]`, a span of 10880 bytes, which
is `136 * 40 * 2`. This is not a new defect. `LoadFp8BlockRaw`
(`include/vllm/model_executor/models/dense_weight_loaders.h`) already widens
BF16 by VALUE and refuses any third dtype, its comment already names this
checkpoint as the BF16 case, and upstream reaches the same f32 through
`parameter.py`'s converting `copy_()`. It is recorded here because a gate run
that hit it unprepared would read as a model defect.

**The per-layer shard naming is handled.** `SelectWeightFiles`
(`src/vllm/transformers_utils/model_resolver.cpp`) takes the `weight_map` values
as the exact shard names when an index is present, so `layers-0.safetensors`
needs no pattern change. This is asserted from the source, not from a run.

## Upstream anchors

Read at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, asserted as
the HEAD of the local checkout before the first read.

| What | Where |
|---|---|
| `weight_block_size` read from the config | `vllm/model_executor/layers/quantization/fp8.py:161`, validated `:115-131` |
| the block scale registers as `weight_scale_inv`, not `weight_scale` | `fp8.py:378-379`, restated `:511`, `:641` |
| `Qwen3_5ForConditionalGeneration` is registered | `vllm/model_executor/models/registry.py:572` |
| the scales apply in the GEMM MAINLOOP, once per K-block, into an f32 accumulator | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:826-838` |
| DeepGEMM auto-disabled for `qwen3_5_text` on device-capability family 120, so CUTLASS is what upstream runs here | `vllm/utils/deep_gemm.py:27-46` |

## Design of the gate

**Greedy, token-exact, one axis.** Both arms decode the same prompts to the same
token counts at the same batching and concurrency with the same sampling, on the
same artifact bytes, with the pinned oracle on the vLLM side.

**Ask whether a greedy path is deterministic BEFORE promising a token gate.**
The oracle's greedy decode is captured three times and compared with itself.
A token gate is well-posed only if that returns `deterministic: true` with zero
multi-member cells, which is the precondition #915 recorded for the bf16 arm of
this same model. If it does not, this row reports `NEEDS_DECISION` and asks for
a distributional gate to be ratified explicitly. **No distributional gate is
pre-ratified here**, and none may be substituted silently.

**Shape**, mirroring #915 so the two arms of one model are comparable: 7 prompts
x 16 tokens, greedy, batch 1, concurrency 1, on the artifact staged at the path
recorded by the run. Comparison is strict and per prompt.

**Only the first divergence per prompt is adjudicable.** After it the arms are
conditioned on different prefixes, so a later position compares two conditionings
rather than one disagreement. Seven prompts give at most seven adjudicable
numbers, never 112.

**The band, and why this arm needs one.** Our CUTLASS collective associates the
two scale multiplies left to right where the reference forms their product first,
so the arms differ by up to one f32 ULP per K-block even when both are correct
(`.agents/specs/vt-matmul-fp8-block-cuda.md`, which cites upstream's own
`rel_diff < 0.001` criterion for exactly this pair). A divergence at a near-tie
is therefore expected physics, not a defect. Each first divergence is adjudicated
against the ratified `kNearTieMnats = 500` band: the oracle's top-2 gap in
millinats and our token's rank in the oracle's top-20. In band or an exact tie is
a PASS for that position; out of band is a real divergence and the row FAILS.

**A token gate cannot see a silent dequant fallback, and this arm has one.**
Issue #1189's own gate design says so and gives the measurement: a x1.02 and a
x1.10 scale perturbation were demonstrably reached and still produced 16/16
identical tokens
(the RED-mutation sweep recorded in the header of
`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp`, where only x2.00 fails). So the token
comparison is necessary and not sufficient, and the run carries three
instruments beside it, each of which fails LOUDLY rather than degrading:

1. **`REFERENCE_TIER_LINES=0`.** A non-zero count means a device tensor reached
   the portable host kernel, which is the #1435 failure mode, and VOIDS the run.
2. **The configure step aborts on `CUTLASS headers NOT found`.** Without it the
   TU is silently absent, the op is unregistered, and the process either
   segfaults (#1435) or measures a different kernel.
3. **A bytes-moved / dispatch assertion.** The block-scaled dispatch counter must
   be non-zero and the resident weight must be one byte per element. A model that
   dequantized at load time produces correct tokens and moves twice the bytes,
   and only this instrument separates the two.

## Risks

- **R1. The gate is unrunnable because the artifact is absent.** This is the
  live state, not a hypothesis. See [`## Owed`](#owed).
- **R2. A pass that measured the wrong kernel.** Mitigated by the three
  instruments above; each of them, not the token count, is what makes the pass
  mean something.
- **R3. The oracle cannot load this checkpoint on GB10.** The FLASHINFER wheel
  runs `Qwen/Qwen3.8-27B` bf16 there; it has **never** been run on the FP8 arm of
  it. FlashAttention has no sm_12x code and cannot be built for it, which is
  upstream and unfixable, so `attention_backend="FLASHINFER"` is the only
  denominator available and `VLLM_ATTENTION_BACKEND` does not exist at the pin.
  If the oracle refuses the checkpoint, that is a finding and the row reports it
  rather than substituting a weaker reference.
- **R4. Memory.** 27.89 GiB of non-visual weights against the bf16 arm's ~54 GiB,
  which already ran on this box. Fit is expected, not established. Note that
  `gpu_memory_utilization` does not bound host RAM on GB10.
- **R5. A dtype too wide, which no token gate can detect.** Read the resident
  dtype of the block scale and of the GEMM output against the oracle's, as
  [`.agents/porting.md`](../porting.md) requires. `f32` is correct for the scale
  and is upstream's own choice; it is not correct for the model-path buffers.
- **R6. Contention.** `dgx:gpu0` is the only device in the fleet at cc 12.1 and
  is the subject of at least three concurrent campaigns. Record the contention
  state actually observed, and never overlap a lease with a file mutex -- the
  #777 double-mutex failure cost `.agents/specs/minimax-music3.md` a whole axis.

## Tests

This row adds no product code, so it ports no upstream test. What it produces is
a gate run and its evidence. The tests that already bind the arm are
`tests/vt/test_ops_matmul_fp8_block_cuda.cpp` (the seven-shape device comparison
and the named refusals) and the block-wise FP8 loader and merge tests landed by
M3, M4 and M6; they are preconditions of this run, not substitutes for it.

## Gates

- **Token gate:** every first-divergence position is an exact tie or is within
  `kNearTieMnats = 500` of the oracle's teacher-forced argmax, on the pinned
  oracle, or the row FAILS. This is the row's only gate.
- **The run is VOID, not passing,** if `REFERENCE_TIER_LINES` is non-zero, if the
  configure step did not abort on missing CUTLASS headers, or if the
  block-scaled dispatch counter is zero.
- **No speed gate exists on this row**, and no number produced under a lease
  without clock control may be quoted as one (#1354).

## Evidence

**The checkpoint audit above, reproducible without the weights.** Both files are
public and unauthenticated. The `lfs.oid` values in the HuggingFace tree API are
fabricated for some repos and are never used here: what is read below is the
FILE BYTES of each shard's own safetensors header, and the `data_offsets`
arithmetic checks the dtype independently of the declared label.

```sh
B=https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main
curl -sSL -o config.json "$B/config.json"
curl -sSL -o index.json  "$B/model.safetensors.index.json"
# then, per shard name in index.json's weight_map values:
curl -sSL -H 'Range: bytes=0-262143' -o head.bin "$B/<shard>"
# u64 LE at offset 0 is the header length; the JSON that follows carries
# every tensor's dtype, shape and data_offsets.
```

**The audit also found one thing wrong in this tree, and it is repaired in the
same flow rather than filed and deferred.** Three sites -- the comment above
`IsFp8BlockProjection`, the comment above
`Fp8BlockQuantConfig::modules_to_not_convert`, and
`.agents/specs/model-fp8-block-weight.md` -- said this checkpoint ships "~400"
`modules_to_not_convert` entries. It ships **882**, all unique, 636 of them
outside the vision tower. The number is the evidence for an argument, namely why
`IsFp8BlockProjection` reads the config AND the tensors instead of probing
dtypes, so a figure wrong by more than 2.2x invites the next reader to re-derive
what the records exist to settle. Tracked by
[#1614](https://github.com/mudler/vllm.cpp/issues/1614).

**What a gate run must record, and none of it is optional:** the oracle identity
asserted with an ABORT on mismatch (`vllm.__version__`, `flashinfer`, and
`vllm.__file__`, because both memory instruments have been blind here before);
the exact build and run recipe; our tree SHA and the binary's provenance read
off the ARTIFACT (`cuobjdump --list-elf` naming an `sm_121a` cubin) rather than
off a log, because this tree's build is not byte-reproducible and a binary hash
identifies a build, not a tree; the checkpoint revision and per-shard sizes; the
driver, CUDA version and compute capability; `REFERENCE_TIER_LINES`; the
contention actually observed from `rc devices`; and, per divergence, the oracle
top-2 gap in millinats, our token's rank in the oracle top-20, and the verdict
against the band.

**The recipe is not novel and must not be re-invented.** `/workspace/q38bf16/`
carries the working two-phase job for the bf16 arm of this same model --
`job.sh`, `build.sh`, `bench.sh`, `reap-orphans.sh` -- and
`/workspace/oracle-vllm/README-WHEELS.md` carries the wheel, the PEP 427 rename
that `pip install` needs, and the `attention_backend="FLASHINFER"` argument that
replaces the environment variable that does not exist at this pin. Stage scripts
on the share, record their sha256, build in `/tmp` because `/workspace` is CIFS
and carries no exec bit or symlink, copy out with `cp -rL`, use `-j 4` because
unconstrained parallelism has OOM-rebooted this box, and install `nvcc`
unconditionally because the worker container is recreated when the box reboots.
Configure with `-DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=121a
-DVLLM_CPP_CUTLASS_FETCH=ON` and ABORT on `CUTLASS headers NOT found`.

## Stop conditions

Report `NEEDS_DECISION` and stop, rather than widening the gate, if any of these
holds.

- The checkpoint is not staged and no download authority has been given. **This
  is the current state.**
- The oracle's greedy decode is not deterministic across three repeats. A
  distributional gate is not pre-ratified and may not be substituted.
- The oracle refuses `Qwen/Qwen3.8-27B-FP8` on GB10.
- The pinned oracle's checkout is not at `5559679229bc961848b121ccdeaa8fa5d79bec98`.
- Any first divergence falls outside the band. Report the divergence, run no
  benchmark on this checkpoint, and do not attempt a fix.
- A lease would have to overlap with the file mutex, or with another session's
  hold on the same device.

## Now

`BLOCKED`. The gate is designed, its preconditions are audited, and it has not
run. `Qwen/Qwen3.8-27B-FP8` has still never been executed against the pinned
oracle on this arm, on any device, and no sentence in this tree may say
otherwise.

**No lease was taken by this row.** `rc devices` read `dgx:gpu0 busy`, held by
`claude/BENCH-QWEN38-27B-SOTA` under `hold` for 30m55s, at the time this spec was
written. Queuing behind it would have bought nothing, because the artifact the
job needs is not on the share.

## Owed

- **The gate itself. `Qwen/Qwen3.8-27B-FP8` has not been run against the pinned
  oracle on this arm, on any device.** This is the debt
  `.agents/specs/vt-matmul-fp8-block-cuda.md` `## Owed` names, and it is
  unchanged by this row. What this row removes is every reason to think the run
  would be pointless: the checkpoint's 407 FP8 tensors are all servable, the
  ragged GDN tensors are excluded by the checkpoint itself, the BF16
  `weight_scale_inv` is already handled by value, and the per-layer shard naming
  already resolves.
- **The checkpoint is not staged, and that is the whole blocker.**
  `/mnt/nas_share/rc/ckpt/` holds `qwen3.8-27b-hf` -- which is the **bf16**
  artifact, `config.json` carrying no `quantization_config` and `dtype:
  bfloat16` -- and `qwen3.8-q1_0`. Neither is this row's subject. Staging costs
  **28.75 GiB** and the share has 3.4 TiB free, so the cost is authority and not
  space: `.agents/developer-preferences.md` authorizes large downloads for the
  `SPEC-DFLASH2` assets only and says "Any other large download, package
  installation, or service management: ask first". **Asking is the next action
  on this row.** Tracked by
  [#1613](https://github.com/mudler/vllm.cpp/issues/1613).
- **The oracle's gateability on this arm.** The FLASHINFER wheel is measured to
  run `Qwen/Qwen3.8-27B` bf16 on GB10. It has never been asked to load the FP8
  artifact, so `gateable` for THIS arm is unmeasured rather than yes. The first
  lease this row takes measures it before anything else, and a refusal is a
  finding.
- **Every speed axis, by construction.** #1354 puts clock control outside a
  lease, so this row cannot produce a defensible ratio and does not try. The FP8
  speed cells stay open gaps and are owed by a benchmark row with host-shell
  authority.
