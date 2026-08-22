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
**Lifecycle:** `BLOCKED` -- on the gate RUN, which is the only thing this row
still owes. It is no longer blocked on the checkpoint, which was staged and
verified on 2026-08-21; the arm is measured REACHED on that checkpoint; and the
pinned oracle is measured on 2026-08-22 to LOAD that checkpoint on GB10 and
generate from it, so the stop condition that would have ended this row before it
started does not hold and the token gate is RUNNABLE. See [`## Now`](#now) and
[`## Owed`](#owed).
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

**It is staged.** The developer granted download authority on **2026-08-21**,
and the artifact is on the share at `/mnt/nas_share/rc/ckpt/qwen38-27b-fp8`,
which a leased worker reads as `/workspace/ckpt/qwen38-27b-fp8`. The
HuggingFace repository is confirmed to need no access grant. Read what staging
discharges and what it does not in
[`## Owed`](#owed): the checkpoint is STAGED, the arm is REACHED on those bytes
-- measured on 2026-08-22 by `rc` job `15e0bfa4-53bc-4f1e-93ad-b9e939e22235` on
`dgx:gpu0` -- and it is NOT GATED. Those are three different states and this
spec keeps them apart.

**How the staged bytes were verified, in two independent legs.** The
HuggingFace API returned **zero LFS oids** for this repository, so there was no
upstream hash to compare against, and neither leg uses one.

1. **File sizes against a manifest captured from the API BEFORE the download
   began.** **81 of 81 files present, 0 missing, 30,890,049,597 B =
   28.769 GiB.**
2. **Safetensors header self-consistency**, which validates the bytes against
   themselves and needs no HuggingFace metadata at all: per shard,
   `8 + header_len + max(data_offsets.end) == file size`, and every tensor's
   span equals `numel * dtype_width`. **66 safetensors files, 0
   header-inconsistent.** Leg 2 is what would catch a truncated CIFS write. A
   file count would not.

The staged headers reproduce the pre-staging audit below exactly: **1606
tensors = 1199 BF16 + 407 F8_E4M3**; **all 407 FP8 tensors satisfy
`N % 128 == 0` AND `K % 128 == 0`, violations 0**; **`weight_scale_inv` present
407 times, 0 FP8 weights missing one**.

**The two byte totals count different things and both are correct.** The
28.75 GiB in the table below is the SAFETENSORS sum, derived from the shard
headers before staging; the staged shards measure 28.747 GiB, which is the same
figure. **28.769 GiB is the whole 81-file tree**, which adds 22.11 MiB of
config, index, tokenizer and chat-template files that carry no tensor. Neither
number supersedes the other, so this spec now says which one it means at each
site.

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
| bytes | 28.75 GiB of safetensors, 27.89 GiB of them outside the vision tower (the staged 81-file tree is 28.769 GiB) |
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

- **R1. The gate is unrunnable because the artifact is absent. RETIRED
  2026-08-21**, when download authority was granted and the checkpoint was
  staged and verified at `/mnt/nas_share/rc/ckpt/qwen38-27b-fp8`. It is written
  here rather than deleted, because what replaced it is narrower and still
  live: a run asserts that path at revision `017b9c7a` and STOPS if it cannot,
  which belongs under [`## Stop conditions`](#stop-conditions) rather than
  here. See [`## Owed`](#owed).
- **R2. A pass that measured the wrong kernel.** Mitigated by the three
  instruments above; each of them, not the token count, is what makes the pass
  mean something.
- **R3. The oracle cannot load this checkpoint on GB10. MEASURED 2026-08-22,
  and it does not hold.** The pinned oracle loaded
  `/workspace/ckpt/qwen38-27b-fp8` and generated from it on `dgx:gpu0`; the
  numbers and the log paths are in [`## Evidence`](#evidence). Before that probe
  the FLASHINFER wheel had been run on `Qwen/Qwen3.8-27B` bf16 there and
  **never** on the FP8 arm of it, which is what made this risk live. The risk
  stays written because what replaced it is narrower and still open: the run
  selected `FLASH_ATTN` rather than the backend it was asked for
  ([#1679](https://github.com/mudler/vllm.cpp/issues/1679)), and
  `VLLM_ATTENTION_BACKEND` does not exist at the pin, so the backend each side
  executes is not yet pinned and `attention_backend="FLASHINFER"` is the lever
  the pin honours. A refusal at gate time is still a finding this row reports
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
fabricated for some repos and are never used here -- for THIS repository the API
returned zero of them, so there was no upstream hash to use even had one been
wanted, which is why the staged-bytes verification above rests on two legs that
need none. What is read below is the
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

**The reachability probe, and every number it produced.** `rc` job
`15e0bfa4-53bc-4f1e-93ad-b9e939e22235` ran on `dgx:gpu0` -- NVIDIA GB10, driver
580.173.02, compute capability 12.1, aarch64, CUDA 13.0 V13.0.88 -- at tree
`c020347a73c7b28117a40cf00991a6f2e4fc260b`. It decoded
`CKPT=/workspace/ckpt/qwen38-27b-fp8`, which is the FP8 artifact and not the
`qwen3.8-27b-hf` bf16 artifact or the `qwen3.8-q1_0` artifact that share that
directory. Its logs are on the share at `/mnt/nas_share/rc/fp8-reach/`:
`job.log`, `decode.txt`, `cuobjdump.log`, `configure.log`, and `build.log`.

Each condition [`## Gates`](#gates) calls VOID was measured, and none of them
fired.

| VOID condition | What the job measured |
|---|---|
| CUTLASS is present | `CFG_RC=0`, with the configure step logging `CUTLASS found at /tmp/b-fp8reach/_deps/cutlass-src`, then `BUILD_RC=0` |
| the translation unit is in the binary | `cuobjdump --list-elf` on `cuda_matmul_fp8_block_cutlass.cu.o` reported `ELF file 1: cuda_matmul_fp8_block_cutlass.cu.1.sm_121a.cubin`, and the run logged `[fp8-reach] TU_LINKED cuda_matmul_fp8_block_cutlass registrar ran` |
| the portable host kernel did not serve | `REFERENCE_TIER_LINES=0` |
| the block-scaled dispatch counter | `FP8_BLOCK_DISPATCH swap_ab=912 pingpong=0 default=0 refused=0 dispatched=912`, against `FP8_BLOCK_GEMMS_ASKED=912` -- asked equals dispatched, and refused is zero |

**Read the CUTLASS evidence off the dependency and the cubin, never off the
feature line.** `CUDA feature cutlass-fp8: ENABLED for [121a]` is not evidence
that CUTLASS is present, because it reports the architecture intersection alone,
before any header detection. The `_deps/cutlass-src` line and the `sm_121a`
cubin are what carry the claim.

One decode ran: `prompt_tokens=12 completion_tokens=24 finish_reason=length`.
The output is coherent, and it continued a list of capitals correctly -- "Rome.
The capital of Spain is Madrid. The capital of Germany is Berlin. The capital of
the United Kingdom is London". The verdict lines are `RESULT=REACHED` and
`DONE_MARKER rc=0`. **Coherent output is not token-exactness.** No oracle ran
beside it, and no token was compared with anything. The job's `tok_s` line is
not quoted here or anywhere else, because the run had no clock control (#1354),
no contention record and no denominator, and because it was a cold first load.
**No timing figure from that job is admissible as a performance number.**

**The oracle-feasibility probe, and every number it produced.** `rc` job
`0d5dfa6a-195f-4475-8527-538ad91102c8` ran on `dgx:gpu0` -- NVIDIA GB10, driver
580.173.02, compute capability 12.1, aarch64 -- on 2026-08-22. Its logs are on
the share at `/mnt/nas_share/rc/fp8-oracle/`: `job.log`, `probe.out` and
`identity.log`. **Only the ORACLE side ran in that job.** Our engine was not
started, no output was compared with anything, no teacher forcing was used, and
[`## Gates`](#gates) is untouched by it.

The oracle identity was asserted before anything else, from `/` so that the
source tree could not be imported by accident.

| What | Value |
|---|---|
| `vllm.__file__` | `/tmp/oracle-venv/lib/python3.12/site-packages/vllm/__init__.py` |
| `vllm.__version__` | `0.1.dev1+g555967922`, which carries the pin `5559679229bc961848b121ccdeaa8fa5d79bec98` |
| torch | `2.13.0+cu130`, with `cuda_available True` |
| identity verdict | `IDENTITY_RC=0` |

The wheel is the prebuilt one already on the share, and no oracle was rebuilt.

**The oracle accepted the model.** It logged
`Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod`, then
`Starting to load model /workspace/ckpt/qwen38-27b-fp8`, and reported
`ORACLE_LOADED=1`. **It then generated**, greedy, at `temperature=0.0`,
`max_tokens=24`, `max_model_len=512`, `gpu_memory_utilization=0.55`,
`enforce_eager=True` and `trust_remote_code=True`, on the prompt
`"The capital of Italy is"`:

- text: `' Rome.\nThe capital of Italy is Rome.\nThe capital of Italy is Rome.\nThe capital of Italy is'`
- token ids: `[21047, 13, 198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760, 6511, 314, 14898, 369]`

The verdict lines are `RESULT=ORACLE_FEASIBLE` and `DONE_MARKER rc=0`.

**This is not a gate result and it is not a speed result.** It establishes
exactly one thing: the stop condition "the oracle refuses
`Qwen/Qwen3.8-27B-FP8` on GB10" does not hold, so the gate can be attempted. No
timing figure from this job is recorded anywhere or is admissible as one -- the
run had no clock control (#1354), no denominator and no contention record, it
ran `enforce_eager=True`, and it was a cold load.

**Do NOT read this decode against the reachability probe's decode.** The
reachability job continued a list of capitals and this one repeats one sentence,
and that difference is NOT evidence of divergence. The two runs used different
harnesses and different sampling configuration and neither was teacher forced,
so they are not comparable. Only a controlled two-sided run -- identical
prompts, token counts, batching and sampling, both arms on the same artifact
bytes, with teacher forcing at the adjudicated position -- can adjudicate a
token, and this row still owes it.

**The probe also produced one OPEN DISCREPANCY, which is filed rather than
resolved.** The run exported `VLLM_ATTENTION_BACKEND=FLASHINFER`, and the engine
logged `Using FLASH_ATTN attention backend out of potential backends:
['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`, then loaded and
generated on `FLASH_ATTN` with no attention-backend failure.
`/mnt/nas_share/rc/oracle-vllm/README-WHEELS.md` records as a MEASURED claim
that this wheel's FlashAttention carries no sm_12x code and that FA2 on GB10
fails with `cudaErrorUnsupportedPtxVersion`, and
[#1456](https://github.com/mudler/vllm.cpp/issues/1456) reaches the same
conclusion from a from-source build at another revision. It did not fail here.
**Neither side is asserted:** the README is not declared wrong, and `FLASH_ATTN`
is not declared safe on GB10 because one decode completed. The reason is
unknown. It matters because a gate pins the executed backend on both sides, and
the oracle's selection here did not do what the environment asked.
[#1679](https://github.com/mudler/vllm.cpp/issues/1679) owns it. One part of the
run is already known to be wrong and explains nothing about the discrepancy: the
same README records that `VLLM_ATTENTION_BACKEND` does not exist at this pin and
that the backend is chosen by the `LLM(..., attention_backend=...)` argument, so
the probe may never have asked for FLASHINFER through a lever the pin honours.
That is a recipe defect the gate run must not repeat.

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

- The checkpoint is not staged at
  `/mnt/nas_share/rc/ckpt/qwen38-27b-fp8` (`/workspace/ckpt/qwen38-27b-fp8`
  from a leased worker), at revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. **This condition is discharged:**
  authority was granted on 2026-08-21 and the checkpoint is staged and
  verified. It stays written because a run that cannot see those bytes at that
  revision must still stop rather than substitute another artifact, and because
  the share also holds two Qwen3.8 artifacts that are not this row's subject.
- The oracle's greedy decode is not deterministic across three repeats. A
  distributional gate is not pre-ratified and may not be substituted.
- The oracle refuses `Qwen/Qwen3.8-27B-FP8` on GB10. **This condition was
  TESTED on 2026-08-22 and it did not fire.** `rc` job
  `0d5dfa6a-195f-4475-8527-538ad91102c8` measured the pinned oracle loading the
  staged checkpoint on that device and generating from it
  (`RESULT=ORACLE_FEASIBLE`), so the gate is runnable. The condition stays
  written, because a later run that meets a refusal must still stop, and because
  one load on one device on one date is not a promise about the next one.
- The pinned oracle's checkout is not at `5559679229bc961848b121ccdeaa8fa5d79bec98`.
- Any first divergence falls outside the band. Report the divergence, run no
  benchmark on this checkpoint, and do not attempt a fix.
- A lease would have to overlap with the file mutex, or with another session's
  hold on the same device.

## Now

`BLOCKED`. The gate is designed, its preconditions are audited, the checkpoint
is staged, the arm is REACHED on that checkpoint, the pinned oracle is measured
to load that checkpoint and generate from it, and the gate has not run. **No
token of ours has been adjudicated against a token of the oracle's on this arm,
on any device**, and no sentence in this tree may say otherwise.

**The oracle-refusal stop condition is now MEASURED, and it does not hold.**
`rc` job `0d5dfa6a-195f-4475-8527-538ad91102c8` on `dgx:gpu0`, 2026-08-22, ran
the pinned oracle -- identity asserted first, `vllm.__version__` reading
`0.1.dev1+g555967922`, `IDENTITY_RC=0` -- against
`/workspace/ckpt/qwen38-27b-fp8`. It logged
`Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod`, loaded the model,
and generated 24 greedy tokens. `RESULT=ORACLE_FEASIBLE`, `DONE_MARKER rc=0`.
That moves this row in exactly one way: the reference side is available on this
device, so the token gate is RUNNABLE rather than hypothetical. The numbers and
the log paths are in [`## Evidence`](#evidence).

**What that probe did NOT do, written out because that distance is the point.**
It ran the ORACLE side alone. Our engine was not started in that job, nothing
was compared, no teacher forcing was used, and the token gate in
[`## Gates`](#gates) is untouched and entirely owed. It produced no speed result
and none may be derived from it: no clock control (#1354), no denominator, no
contention record, `enforce_eager=True`, and a cold load. **And its decode may
not be set against the reachability probe's decode.** Different harness,
different sampling configuration, no teacher forcing on either side; the two
outputs are not comparable, and the difference between them is not evidence of
divergence. Only a controlled two-sided run under identical prompts, token
counts, batching and sampling can adjudicate a token.

**One open discrepancy came out of it.** The run asked for FLASHINFER through
`VLLM_ATTENTION_BACKEND` and the oracle selected and ran `FLASH_ATTN`, which a
recorded measurement says carries no sm_12x code on this device. Neither side is
asserted here and the reason is unknown.
[#1679](https://github.com/mudler/vllm.cpp/issues/1679) owns it, and the gate
run has to pin the executed backend on both sides.

**Staged is not reached, and reached is not gated. This row took the first of
those two steps and not the second.** The REACHABILITY probe has RUN and
returned `RESULT=REACHED`: `rc` job `15e0bfa4-53bc-4f1e-93ad-b9e939e22235` on
`dgx:gpu0`, at tree `c020347a73c7b28117a40cf00991a6f2e4fc260b`, decoding
`/workspace/ckpt/qwen38-27b-fp8`. It measured every condition
[`## Gates`](#gates) calls VOID, and none of them fired. The numbers, the log
paths and the decoded text are in [`## Evidence`](#evidence).

**What the probe did NOT do, written out because that distance is the point.**
It adjudicated no token against any oracle. The pinned vLLM never ran, no output
of ours was compared with any reference output, and the token gate in
[`## Gates`](#gates) is untouched and entirely owed. The probe produced no speed
result either, and none may be derived from it: the run had no clock control
(#1354), no contention record and no denominator, and it was a cold first load.
**No timing figure from that job may be quoted as a performance number.**
Coherent output is evidence that the kernel computes sane values on real
weights. It is not token-exactness, and it is not a ratio.

**The probe exercised the swap_ab configuration ONLY, which bounds what it
covers.** All 912 dispatches took swap_ab, and pingpong and default each took
zero. That follows from a decode at `M = 1` against the
`swap_ab = (M <= 64) || (M % 4 != 0)` heuristic. The pingpong and default
configurations are therefore UNEXERCISED on the model path by this run.
`tests/vt/test_ops_matmul_fp8_block_cuda.cpp` covers all three on seven shapes
(#1437), and it stays the only coverage the other two have.

**This row has taken two leases, both for probes and neither for the gate.** The
first ran our side alone and no oracle; the second ran the oracle alone and no
engine of ours. Neither compared a token, neither produced a performance number,
and the row's gate is unchanged by both.

## Owed

- **The gate itself, and it is the only debt this row still carries.
  `Qwen/Qwen3.8-27B-FP8` has not been run against the pinned oracle on this arm,
  on any device.** This is the debt
  `.agents/specs/vt-matmul-fp8-block-cuda.md` `## Owed` names. Neither the
  staging nor the reachability probe touches it. What those two removed is every
  reason to think the run would be pointless: the checkpoint's 407 FP8 tensors
  are all servable, the ragged GDN tensors are excluded by the checkpoint
  itself, the BF16 `weight_scale_inv` is already handled by value, the
  per-layer shard naming already resolves, and the CUTLASS kernel served all 912
  GEMMs of a real decode on the real weights. What is left is the comparison,
  and only a lease that runs the oracle beside us can produce it.
- **The checkpoint is STAGED, so the blocker this bullet used to name is
  discharged. What is owed here is the RUN, not the staging.** The developer
  granted download authority on **2026-08-21**, and `Qwen/Qwen3.8-27B-FP8` @
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` -- a HuggingFace repository
  confirmed to need no access grant -- is at
  `/mnt/nas_share/rc/ckpt/qwen38-27b-fp8`, which a leased worker
  reads as `/workspace/ckpt/qwen38-27b-fp8`. Verified as **81 of 81 files, 0
  missing, 30,890,049,597 B = 28.769 GiB**; **66 safetensors files, 0
  header-inconsistent**; **1606 tensors = 1199 BF16 + 407 F8_E4M3**; **all 407
  FP8 tensors `N % 128 == 0` AND `K % 128 == 0`, violations 0**; and
  **`weight_scale_inv` present 407 times, with 0 FP8 weights missing one**. The
  method is the two-leg one recorded in the checkpoint section above, and it
  uses no HuggingFace hash because the API returned zero LFS oids for this
  repository. The share still holds `qwen3.8-27b-hf` -- the **bf16** artifact,
  `config.json` carrying no `quantization_config` and `dtype: bfloat16` -- and
  `qwen3.8-q1_0`; neither was this row's subject before staging and neither is
  now. [#1613](https://github.com/mudler/vllm.cpp/issues/1613) tracked the
  staging and the ask for authority, and staging discharges it. **Staging
  discharges nothing else.** The run this bullet used to block is owed by the
  bullet above, and no sentence anywhere may read the staged bytes as a gate
  result.
- **The arm is REACHED on those bytes, and reached is still not gated.** The
  probe recorded in [`## Evidence`](#evidence) returned `RESULT=REACHED`, so the
  block-wise FP8 CUDA kernel is measured to serve `Qwen/Qwen3.8-27B-FP8` from a
  decode rather than only from a unit test. Two limits bound that result. It
  compared **no token against any oracle**, so it discharges no part of
  [`## Gates`](#gates) and it is not a gate result. And it exercised **one of
  the three tile configurations**: 912 of 912 GEMMs took swap_ab, because decode
  runs at `M = 1` against `swap_ab = (M <= 64) || (M % 4 != 0)`, so pingpong and
  default stay unexercised on the model path.
- **The oracle's gateability on this arm, which is now PARTLY measured.** The
  FLASHINFER wheel was already measured to run `Qwen/Qwen3.8-27B` bf16 on GB10.
  On **2026-08-22** it was asked for the first time to load the FP8 artifact,
  and it loaded it and generated from it: `rc` job
  `0d5dfa6a-195f-4475-8527-538ad91102c8`, `RESULT=ORACLE_FEASIBLE`,
  `DONE_MARKER rc=0`, with the identity asserted first. **The refusal this
  bullet used to fear is measured and absent, and that is the whole of what it
  discharges.** Three parts of gateability stay unmeasured, and the run
  discharges none of them. The oracle's greedy decode has NOT been captured
  three times and compared with itself, which
  [`## Design of the gate`](#design-of-the-gate) makes the precondition of a
  well-posed token gate. The executed attention backend is NOT pinned: the run
  selected `FLASH_ATTN` after being asked for FLASHINFER through a variable that
  does not exist at the pin
  ([#1679](https://github.com/mudler/vllm.cpp/issues/1679)). And it ran under
  `enforce_eager=True`, a feasibility setting the gate run must choose
  deliberately rather than inherit, since [`## Gates`](#gates) forbids it as a
  denominator. A refusal at gate time is still a finding.
- **Every speed axis, by construction.** #1354 puts clock control outside a
  lease, so this row cannot produce a defensible ratio and does not try. The
  reachability probe took a lease under exactly that constraint, so its `tok_s`
  line is not a result and is recorded nowhere. The FP8 speed cells stay open
  gaps and are owed by a benchmark row with host-shell authority.
