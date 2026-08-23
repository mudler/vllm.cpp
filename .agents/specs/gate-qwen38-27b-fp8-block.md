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
**Lifecycle:** `DONE` -- the token gate RAN on 2026-08-23 and it PASSED. `rc`
job `2911ed39-413f-4624-ae2e-87953b6bd9fd` on `dgx:gpu0` decoded the same seven
prompts on both arms and adjudicated every first divergence:
`TOKEN_VERDICT=PASS`, `SUMMARY strict=6 in_band=1 out_of_band=0
integrity_fail=0 length_mismatch=0`, `RESULT=PASS`, `DONE_MARKER rc=0`. Six of
the seven prompts are token-identical at all 16 positions; the seventh is an
adjudicated near-tie whose teacher-forced verdict is `IN_BAND` at rank 1 with a
logprob difference of exactly 0. That is this row's only gate and it is MET.
**No speed number was produced by the run and none may be derived from it**
(#1354). See [`## Outcome`](#outcome), [`## Now`](#now) and [`## Owed`](#owed).
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
- It **HAS a token gate since 2026-08-23**, and this row is what removed the
  sentence that said it had none. `Qwen/Qwen3.8-27B-FP8` was decoded beside the
  pinned oracle on `dgx:gpu0` and every first divergence is an exact tie or in
  band: `TOKEN_VERDICT=PASS`. See [`## Now`](#now) and
  [`## Evidence`](#evidence).
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
  the pin honours. **That lever is now MEASURED to work**: on 2026-08-22 the
  argument was accepted and the language-model path logged FLASHINFER, while the
  vision tower and the multimodal encoder logged `FLASH_ATTN` in the same
  process, because vLLM selects per path. What stays open in #1679 is why a
  `FLASH_ATTN` path runs on this device at all. A refusal at gate time is still
  a finding this row reports rather than substituting a weaker reference.
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

**The precondition probe, and every number it produced.** `rc` job
`ebc8214b-9813-4a53-8711-9bacd4261d6f` ran on `dgx:gpu0` -- NVIDIA GB10, driver
580.173.02, compute capability 12.1 -- on 2026-08-22. Its logs are on the share
at `/mnt/nas_share/rc/fp8-precond/`: `job.log`, `precond.out` and `result.json`.
**Only the ORACLE side ran**, on the same staged checkpoint
`/workspace/ckpt/qwen38-27b-fp8`. The identity was asserted before anything
else, from `/` so that no source tree could be imported by accident:
`vllm.__version__` read `0.1.dev1+g555967922`, which carries the pin
`5559679229bc961848b121ccdeaa8fa5d79bec98`, and `IDENTITY_RC=0`. The engine
logged `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` again. The
verdict lines are `RESULT=PRECONDITIONS_MET` and `DONE_MARKER rc=0`.

**Greedy determinism is MET on this workload.** Three identical greedy calls --
`temperature=0.0`, `max_tokens=24`, prompt `"The capital of Italy is"`,
`max_model_len=512`, `gpu_memory_utilization=0.55`, `enforce_eager=True` --
returned byte-identical token ids all three times, compared on TOKEN IDS and not
on text:

```
[21047, 13, 198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760,
 6511, 314, 14898, 369, 21047, 13, 198, 760, 6511, 314, 14898, 369]
```

`DETERMINISTIC=true`. That is the precondition
[`## Design of the gate`](#design-of-the-gate) sets, so the token gate is
well-posed and no distributional gate has to be ratified. **It is one prompt of
24 tokens, on one device, on one date.** Determinism in general is not claimed
here, and the gate run re-establishes it over its own seven-prompt set.

**The backend is pinnable, and two backends run in one process.** The run passed
`LLM(..., attention_backend="FLASHINFER")`, the lever `README-WHEELS.md` records
this pin as honouring in place of the `VLLM_ATTENTION_BACKEND` environment
variable that does not exist at the pin. The argument was accepted
(`BACKEND_OK=FLASHINFER`) and the engine logged, in one process:

| Engine log line | The path it names |
|---|---|
| `[cuda.py:422] Using AttentionBackendEnum.FLASHINFER backend.` | the language-model path |
| `[flashinfer.py:822] FlashInfer resolved query dtypes: prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend=flashinfer-native` | the same path, resolved |
| `[cuda.py:541] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention` | the vision tower |
| `[mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.` | the multimodal encoder |

**vLLM selects the attention backend PER PATH**, so the language-model path ran
FLASHINFER while the tower and the encoder ran `FLASH_ATTN` beside it. Only the
FLASHINFER configuration was attempted in this job and no fallback was reached.
Read the four lines above as the measurement and not the job's own summary
field, which reads `EXECUTED='unknown'` because it greps for the earlier log
shape.

**What this does to [#1679](https://github.com/mudler/vllm.cpp/issues/1679), and
what it does not.** #1679 records that the feasibility run logged `FLASH_ATTN`
against a recorded measurement that this wheel's FlashAttention carries no
sm_12x code and should fail on GB10 with `cudaErrorUnsupportedPtxVersion`. This
probe supplies a PARTIAL explanation: because the selection is per path, the
earlier bare `Using FLASH_ATTN` line **may** have named the vision path rather
than the decode path. **That is a hypothesis consistent with this evidence and
it is not a demonstrated fact.** Nobody has re-read the earlier log to attribute
its line to a path, and it does not explain why a `FLASH_ATTN` path ran at all
on this device without the recorded PTX failure. #1679 stays OPEN on that
second part.

**This is not a gate result and it is not a speed result.** Our engine was not
started in this job, no output was compared with anything, no teacher forcing
was used, and [`## Gates`](#gates) is byte-for-byte what it was before the
probe. No timing figure from this job is recorded anywhere or is admissible as
one: no clock control (#1354), no denominator, no contention record,
`enforce_eager=True`, and a cold load.

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

**THE GATE RUN, and every number it produced.** `rc` job
`2911ed39-413f-4624-ae2e-87953b6bd9fd` ran on `dgx:gpu0` -- NVIDIA GB10, driver
580.173.02, compute capability 12.1, aarch64, CUDA 13.0 V13.0.88 -- on
**2026-08-23**, starting at `00:04:26Z`. Its logs are on the share at
`/mnt/nas_share/rc/fp8-gate/`: `job.log`, `oracle.txt`, `ours.txt`,
`gate_result.json`, `ckpt-assert.log`, `cuobjdump.log`, `identity.log`,
`prompt_ids.json` and `our_ids.json`. **This is the two-sided run the row was
waiting for.** Both arms decoded the same seven prompts to 16 tokens, greedy,
batch 1, concurrency 1, on the same staged checkpoint bytes, and every first
divergence was adjudicated by teacher forcing on OUR prefix.

| Verdict line | Value |
|---|---|
| `TOKEN_VERDICT` | `PASS  every first divergence is an exact tie or in band` |
| `SUMMARY` | `strict=6 in_band=1 out_of_band=0 integrity_fail=0 length_mismatch=0` |
| `RESULT` | `PASS  three instruments clean: REFERENCE_TIER_LINES=0, dispatched=2736, bytes_per_element=1.000000.` |
| `DONE_MARKER` | `rc=0` |
| `TREE_SHA` | `4cbf94c9fa5fc8310cde7967c72e551288fca5f0` |
| `ORACLE_PIN` | `5559679229bc961848b121ccdeaa8fa5d79bec98` |
| `ORACLE_SHAPE` | `graphed` |

**What ran on our side, and through which entry point.** `TREE_SHA` is
`4cbf94c9fa5fc8310cde7967c72e551288fca5f0`, which is this row's own
precondition-probe commit and an ancestor of `main`. The harness is a SCRATCH
client of the STABLE C ABI (`include/vllm.h`) and includes no internal header:
it calls `vllm_complete_tokens` (ABI v13), which takes the prompt as ids and
returns the generated ids, so **no tokenizer sits inside the comparison**. It is
recorded as scratch, with its sha256 `bf366d87f...` in
`/mnt/nas_share/rc/fp8-gate/STAGED-SHA256.txt`. A SCRATCH patch supplied part of
instrument 3: `instrument.patch` (sha256 `0c81051595...`, `PATCH_RC=0`, applied
cleanly to three files). **Read what it does and does not add.** The dispatch
counters are the TREE's own -- `Fp8BlockScaledStats`, incremented only after
CUTLASS reports success, and `dense_fp8_block::BlockGemmCount` -- and the patch
adds only a way to PRINT them from a binary whose sole interface is the C ABI.
What it genuinely adds is the RESIDENT-WEIGHT counter, three atomics on the
packing path, plus the `TU_LINKED` constructor line that proves the CUTLASS
translation unit survived into the linked image. **Neither the harness nor the
patch is committed**, and this spec records them as scratch rather than as tree
code.

**THE SEVEN ADJUDICATIONS.** Six prompts are token-identical at all 16
positions. One diverges, and it is adjudicated `IN_BAND`.

| # | prompt | first divergence | verdict |
|---:|---|---|---|
| 0 | `The capital of France is` | none, `d=None`, 16 = 16 | `STRICT` |
| 1 | `def fibonacci(n):` | `d=6` | `IN_BAND` |
| 2 | `Once upon a time,` | none, `d=None`, 16 = 16 | `STRICT` |
| 3 | `The largest planet in our solar system is` | none, `d=None`, 16 = 16 | `STRICT` |
| 4 | `The chemical symbol for gold is` | none, `d=None`, 16 = 16 | `STRICT` |
| 5 | `import numpy as np` | none, `d=None`, 16 = 16 | `STRICT` |
| 6 | `In the beginning God created` | none, `d=None`, 16 = 16 | `STRICT` |

Every row logged `length_mismatch=False` and every arm returned
`completion_tokens=16 finish_reason=length`, so no verdict rests on a truncated
sequence.

**The one adjudicated position, read exactly as the instrument printed it.**

```text
ADJUDICATION prompt=1 pos=6 ours=16 oracle_argmax=16 oracle_greedy=15
  top2_gap_mnats=125.000000 oracle_minus_ours_mnats=0.000000 rank=1
  verdict=IN_BAND
```

**Read that line in the right order, because it is the whole reason the band and
the teacher forcing exist.** The two arms' FREE continuations chose different
tokens at position 6: the oracle's own greedy decode took token `15` and ours
took token `16`. Under TEACHER FORCING on OUR prefix, the oracle's argmax at the
same position **is our token**: `oracle_argmax=16`, at `rank=1`, with
`oracle_minus_ours_mnats=0.000000` -- the oracle ranks our token first and
assigns it exactly the logprob of its own argmax (`-0.681679904460907` for both,
`gate_result.json`). The oracle's own top two were separated by
`top2_gap_mnats=125.000000` against the ratified `kNearTieMnats = 500` band.
**That is a near-tie, which `.agents/specs/vt-matmul-fp8-block-cuda.md` predicts
as expected physics**: our CUTLASS collective associates the two scale multiplies
left to right where the reference forms their product first, so the arms can
differ by up to one f32 ULP per K-block while both are correct. **It is not a
defect**, and it is exactly the case [`## Design of the gate`](#design-of-the-gate)
declared adjudicable in advance.

**The index that would invert that verdict was PROVED, not assumed.**
`gate_oracle.py` aborts with `INTEGRITY_FAIL` rather than printing a number
unless the engine echoes the supplied ids unchanged, the teacher-forced argmax
at every position BEFORE the divergence reproduces the oracle's own greedy
token, and our token is present in the cell that must contain it.
`integrity_fail=0` and `"integrity_ok": true`.

**Greedy determinism was re-established over THIS prompt set**, rather than
inherited from the one-prompt precondition probe: three greedy repeats over all
seven prompts returned byte-identical id lists, `ORACLE_REPEAT0`,
`ORACLE_REPEAT1` and `ORACLE_REPEAT2` compared by token id, and
`ORACLE_DETERMINISTIC=true`. The token gate is therefore well-posed on its own
workload and no distributional gate was substituted.

**THE THREE INSTRUMENTS, and all three are clean.**

| Instrument | What the run measured |
|---|---|
| 1. the portable host kernel did not serve | `REFERENCE_TIER_LINES=0` |
| 2. CUTLASS is present, read off the ARTIFACT | `CFG_RC=0` and `CUTLASS_NOT_FOUND_HITS=0`; `cuobjdump --list-elf` on `cuda_matmul_fp8_block_cutlass.cu.o` reports `ELF file 1: cuda_matmul_fp8_block_cutlass.cu.1.sm_121a.cubin`, `CUOBJDUMP_RC=0`, and the linked `libvllm.so` carries `LIB_SM121A_CUBINS=54` |
| 3. the block-scaled dispatch and the resident bytes | `DISPATCHED=2736 SWAP_AB=2736 PINGPONG=0 DEFAULT=0 REFUSED=0` against `FP8_BLOCK_GEMMS_ASKED=2736`, and `FP8_BLOCK_RESIDENT tensors=400 bytes=24326963200 elements=24326963200 bytes_per_element=1.000000` |

Instrument 3 is the one a token gate cannot replace. `bytes_per_element` is
`1.000000` across 400 resident tensors and 24,326,963,200 bytes, so **the weights
stayed FP8 in device memory and nothing dequantized at load time**. A dequantizing
load would have produced the same tokens and moved twice the bytes.
`refused=0` with `asked == dispatched` says every block-scaled GEMM the model
asked for was served by the CUTLASS arm, and `TU_LINKED_LINES=1` says the
translation unit's registrar ran.

**The run exercised swap_ab ONLY, and that bounds what it covers.** All 2736
dispatches took swap_ab; pingpong and default each took zero, because decode runs
at `M = 1` against the `swap_ab = (M <= 64) || (M % 4 != 0)` heuristic. This is
the same bound the reachability probe carried. `tests/vt/test_ops_matmul_fp8_block_cuda.cpp`
covers all three configurations on seven shapes (#1437), and it stays the only
coverage the other two have.

**THE ORACLE RAN IN ITS PRODUCTION CONFIGURATION.** `ORACLE_ENFORCE_EAGER=False`
and `ORACLE_SHAPE=graphed`. **That settles the third precondition
[`## Owed`](#owed) listed as unmeasured**, rather than dodging it: the earlier
probes ran `enforce_eager=True` as a feasibility shortcut and settled nothing,
[`## Gates`](#gates) forbids eager as a denominator, and this run chose the
graphed shape deliberately. The engine was constructed with
`LLM(model=/workspace/ckpt/qwen38-27b-fp8, max_model_len=1024,
gpu_memory_utilization=0.55, enforce_eager=False, trust_remote_code=True,
attention_backend="FLASHINFER")`, which `oracle.txt` echoes as its non-default
args, and it executed:

| Engine log line | The path it names |
|---|---|
| `[cuda.py:422] Using AttentionBackendEnum.FLASHINFER backend.` | the language-model path |
| `[__init__.py:604] Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` | the block-wise FP8 GEMM upstream dispatches here |
| `[cuda.py:541] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention` | the vision tower |
| `[mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.` | the multimodal encoder |

The decode path therefore ran FLASHINFER on the oracle side, through the
`attention_backend` argument this pin honours rather than the environment
variable that does not exist at it.
**[#1679](https://github.com/mudler/vllm.cpp/issues/1679) STAYS OPEN.** This run
touches only the first half of it. Why a `FLASH_ATTN` path runs on GB10 at all,
against the recorded measurement that this wheel's FlashAttention carries no
sm_12x code and should fail with `cudaErrorUnsupportedPtxVersion`, is untouched
here and nothing in this run explains it.

**The oracle identity was asserted three ways, from `/`, with an ABORT on
mismatch** (`identity.log`, `IDENTITY_RC=0`), because both memory instruments
have been blind here before:

| What | Value |
|---|---|
| `vllm.__file__` | `/tmp/oracle-venv-gate/lib/python3.12/site-packages/vllm/__init__.py`, under the venv |
| `vllm.__version__` | `0.1.dev1+g555967922`, which carries the pin `5559679229bc961848b121ccdeaa8fa5d79bec98` |
| `flashinfer.__version__` | `0.6.15.post1` |
| torch | `2.13.0+cu130`, `cuda_available True` |

**The checkpoint was asserted before either arm loaded it** (`ckpt-assert.log`,
`CKPT_ASSERT=OK`): `ARCH Qwen3_5ForConditionalGeneration`,
`QUANT fp8 [128, 128] dynamic e4m3`, `MODULES_TO_NOT_CONVERT 882`,
`FILES 82 TOTAL_BYTES 30890049597`,
`SAFETENSORS_SHARDS 66 SAFETENSORS_BYTES 30866866928`, and
`CONFIG_SHA256 74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc`,
with all 66 per-shard sizes recorded in that file. `TOTAL_BYTES` matches the
staging audit's 30,890,049,597 B exactly and the safetensors shard count matches
its 66. The file count reads 82 where the staging audit read 81, and because the
byte totals are identical the extra entry carries zero bytes; no cause is
asserted for it here, and nothing tensor-bearing differs.

**The prompt ids came from the CHECKPOINT's own tokenizer**, on CPU with no
model loaded (`TOKENIZE_OK 7`, `TOKENIZE_RC=0`), and both arms were fed those ids
rather than text. `PROMPT_IDS` for all seven are in `job.log` and
`prompt_ids.json`.

**Contention, and the arms never overlapped.** The job saw no compute apps at
start, and the GPU was returned between the arms:
`COMPUTE_APPS_BEFORE_ORACLE=0` before the oracle loaded and
`COMPUTE_APPS_AFTER=0` at teardown. The run held an `rc` lease on `dgx:gpu0` for
its whole duration and took no file mutex beside it, so the #777 double-mutex
failure was not repeated.

**NO SPEED NUMBER.** The run produced none, records none, and none may be
derived from it: no clock control inside a lease (#1354), no denominator, and no
speed axis on this row. **The run's duration is not a result.** The FP8 speed
cells stay open gaps owed by a benchmark row with host-shell authority.

**THE PROMPT SET, and its provenance stated honestly.** #915's exact seven could
NOT be recovered, and this spec does not pretend otherwise. The bf16 arm's
harness at `/workspace/q38bf16` is the SPEED harness, which drives
`vllm bench serve --dataset-name random` and carries no fixed prompt list at
all; #915's body and comments carry none; and that token-gate harness was never
committed. What was used instead, recorded in `prompts.py` on the share and
reproduced here:

- the SIX committed at `scripts/qwen3coder-oracle-capture.py::PROMPTS`, verbatim
  and in order. That list is a recoverable strict subset of the same family: it
  contains all three prompts `.agents/specs/qwen38-27b-bf16-gate.md` names as
  the bf16 arm's divergent ones -- `Once upon a time,`, `The largest planet in
  our solar system is` and `import numpy as np`.
- a SEVENTH chosen by a stated rule rather than by taste: the first entry of
  this tree's standard 16-prompt battery
  (`scripts/qwen3-oracle-capture.py::PROMPTS`) that the six do not already
  contain, which is `In the beginning God created`.

**This set is NOT asserted to be #915's set.** Six of the seven are provably
from the same committed family and carry #915's three adjudicated prompts; the
seventh is this row's own choice.

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
  distributional gate is not pre-ratified and may not be substituted. **This
  condition was MEASURED on 2026-08-22 and it does not fire.** `rc` job
  `ebc8214b-9813-4a53-8711-9bacd4261d6f` captured three greedy decodes of one
  prompt and compared them by token id: three byte-identical sequences,
  `DETERMINISTIC=true`. The condition stays written, because one prompt of 24
  tokens on one device on one date is not determinism in general, and a later
  run that meets a non-deterministic decode must still stop and ask.
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

## Outcome

**What was measured.** One two-sided run, `rc` job
`2911ed39-413f-4624-ae2e-87953b6bd9fd` on `dgx:gpu0`, 2026-08-23: seven prompts
x 16 tokens, greedy, batch 1, concurrency 1, both arms on the same staged
`Qwen/Qwen3.8-27B-FP8` bytes, our arm through the stable C ABI and the oracle at
the parity pin in its graphed production configuration. Six prompts are
token-identical at all 16 positions. The seventh diverges at position 6 and is
adjudicated `IN_BAND` at rank 1 with `oracle_minus_ours_mnats=0.000000` against
the oracle's own 125-millinat top-2 gap. Beside the tokens, three instruments
read clean: `REFERENCE_TIER_LINES=0`, an `sm_121a` cubin read off the artifact
with `CUTLASS_NOT_FOUND_HITS=0`, and `dispatched=2736 == asked` with
`bytes_per_element=1.000000` over 400 resident tensors. `TOKEN_VERDICT=PASS`,
`RESULT=PASS`, `DONE_MARKER rc=0`.

**Why each default has the value it has.**

- **7 prompts x 16 tokens, greedy, batch 1, concurrency 1.** Mirrors
  [#915](https://github.com/mudler/vllm.cpp/issues/915), so the bf16 and the
  block-wise FP8 arms of one model are comparable rather than each measured on
  its own shape.
- **`kNearTieMnats = 500`.** The band ratified before this run, not chosen after
  seeing the divergence. `.agents/specs/vt-matmul-fp8-block-cuda.md` derives it
  from the scale-multiply association difference, and upstream applies its own
  `rel_diff < 0.001` criterion to exactly this pair. The one adjudicated gap came
  in at 125 millinats, a quarter of the band, and our token's own deficit was 0.
- **Teacher forcing at the FIRST divergence only.** After the first disagreement
  the arms carry different prefixes, so a later position compares two
  conditionings rather than one disagreement. Seven prompts give at most seven
  adjudicable numbers, never 112.
- **Top-20 logprobs.** A bounded window is enough to place our token's rank, and
  the instrument prints `>20` rather than a number if it falls outside. It
  returned `rank=1`.
- **Token ids on both sides, from the checkpoint's own tokenizer.**
  `vllm_complete_tokens` takes ids and returns ids, and the oracle side uses
  `TokensPrompt` and asserts the engine echoes them, so no tokenizer sits inside
  the comparison and no text-level normalisation can hide or invent a
  difference.
- **`enforce_eager=False`, the graphed shape.** Chosen deliberately.
  [`## Gates`](#gates) forbids eager as a denominator and vLLM's production
  configuration is graphed, so the reference side had to be the production one.
  This is the third precondition that was still owed, and it is now measured
  rather than inherited.
- **`max_model_len=1024`, `gpu_memory_utilization=0.55`.** The longest sequence
  the run builds is 8 prompt ids plus 16 generated tokens, far under 1024, so the
  length bound cannot truncate anything. The memory fraction is the value the
  earlier probes fit in, and the two arms never overlapped on the device, so no
  larger fraction was needed.

**What was rejected, and why.**

- **A distributional gate.** None was pre-ratified and none was substituted. The
  run re-established `ORACLE_DETERMINISTIC=true` over its own seven-prompt set
  rather than inheriting the one-prompt precondition result, so the token gate is
  well-posed on the workload it actually ran.
- **Calling prompt 1 a divergence of ours.** Rejected on the teacher-forced
  evidence: conditioned on our prefix the oracle's argmax IS our token, at rank 1,
  with a logprob difference of exactly 0. The free continuations differ because
  the oracle's own top two are 125 millinats apart, which is the near-tie the
  design predicted.
- **Reading CUTLASS presence off the configure feature line.** `CUDA feature
  cutlass-fp8: ENABLED for [121a]` reports an architecture intersection before any
  header detection. The claim rests on the `_deps/cutlass-src` line and on
  `cuobjdump --list-elf` naming an `sm_121a` cubin in the object.
- **Trusting the token comparison alone.** #1189's own gate design measures that a
  x1.02 and a x1.10 scale perturbation still produce 16/16 identical tokens, so
  the tokens cannot see a silent dequant. Instrument 3's
  `bytes_per_element=1.000000` is what separates a quantized run from a
  dequantizing one.
- **Claiming #915's exact prompt set.** It could not be recovered, and the
  provenance is stated as what it is: six of seven from a committed family that
  carries #915's three adjudicated prompts, and a seventh chosen by a stated
  rule.
- **Every speed number.** By construction. No clock control inside a lease
  (#1354), no denominator, no speed axis on this row. The run's duration is not a
  result.
- **Committing the instruments.** The harness and the counter patch are scratch,
  recorded as scratch with their sha256 on the share. Landing a probe counter as
  product code was not this row's scope, and no row asked for it.

**What the pass does NOT establish.** It exercised swap_ab only, because decode
runs at `M = 1`; pingpong and default stay unexercised on the model path and
`tests/vt/test_ops_matmul_fp8_block_cuda.cpp` remains their only coverage. It
produced no speed number. It does not close
[#1679](https://github.com/mudler/vllm.cpp/issues/1679), whose unexplained half
-- why a `FLASH_ATTN` path runs on GB10 at all against the recorded PTX
measurement -- this run does not touch. And it is one gate on one device on one
date, not a promise about the next run.

## Now

`DONE`. **The token gate has RUN and it PASSED.** `rc` job
`2911ed39-413f-4624-ae2e-87953b6bd9fd` on `dgx:gpu0` -- NVIDIA GB10, driver
580.173.02, compute capability 12.1, aarch64, CUDA 13.0 V13.0.88 -- ran both
arms on 2026-08-23 at tree `4cbf94c9fa5fc8310cde7967c72e551288fca5f0` against the
pinned oracle `5559679229bc961848b121ccdeaa8fa5d79bec98`. Seven prompts, 16
tokens each, greedy, batch 1, concurrency 1, the same staged checkpoint bytes on
both sides, prompt ids from the checkpoint's own tokenizer so no tokenizer sits
inside the comparison. `TOKEN_VERDICT=PASS`, `SUMMARY strict=6 in_band=1
out_of_band=0 integrity_fail=0 length_mismatch=0`, `RESULT=PASS`,
`DONE_MARKER rc=0`.

**The sentence this row existed to remove is now removable.** A token of ours
HAS been adjudicated against a token of the oracle's on this arm, on this
device, on this date. Six of the seven prompts are token-identical at all 16
positions. The seventh, `def fibonacci(n):`, first differs at position 6, and
under teacher forcing on OUR prefix the oracle's argmax is OUR token at rank 1
with `oracle_minus_ours_mnats=0.000000`, against its own 125-millinat top-2 gap
and the ratified 500-millinat band: `verdict=IN_BAND`. The free continuations
differ because the oracle's own top two are that close, which is the near-tie
`.agents/specs/vt-matmul-fp8-block-cuda.md` predicts from the scale-multiply
association difference. **It is not a defect.** The numbers, the integrity
proof that makes the teacher-forcing index trustworthy, and the whole log
inventory are in [`## Evidence`](#evidence).

**The three instruments are clean, and they are what makes the pass mean
something.** `REFERENCE_TIER_LINES=0`, so the portable host kernel never served.
`CUTLASS_NOT_FOUND_HITS=0` with `cuobjdump --list-elf` naming
`cuda_matmul_fp8_block_cutlass.cu.1.sm_121a.cubin` in the object and 54 `sm_121a`
cubins in the linked library, read off the ARTIFACT rather than off a feature
line. And `DISPATCHED=2736 SWAP_AB=2736 PINGPONG=0 DEFAULT=0 REFUSED=0` against
`FP8_BLOCK_GEMMS_ASKED=2736`, with `FP8_BLOCK_RESIDENT tensors=400
bytes=24326963200 elements=24326963200 bytes_per_element=1.000000`. **The weights
stayed one byte per element in device memory**, which is the only instrument that
separates this run from a dequantizing load that would have produced the same
tokens and moved twice the bytes.

**The oracle ran GRAPHED, in its production configuration.**
`ORACLE_ENFORCE_EAGER=False`, `ORACLE_SHAPE=graphed`. **That settles the third
precondition this row carried as unmeasured**, rather than dodging it: the two
earlier probes ran `enforce_eager=True` as a feasibility shortcut,
[`## Gates`](#gates) forbids eager as a denominator, and this run chose the
graphed shape deliberately. The engine was constructed with
`attention_backend="FLASHINFER"`, the lever this pin honours, and the
language-model path logged `Using AttentionBackendEnum.FLASHINFER backend` while
the vision tower and the multimodal encoder logged `FLASH_ATTN` beside it,
because vLLM selects per path. It logged `Selected CutlassFp8BlockScaledMMKernel
for Fp8LinearMethod` on the block-wise FP8 projections.

**[#1679](https://github.com/mudler/vllm.cpp/issues/1679) STAYS OPEN.** The
oracle ran FLASHINFER on the language-model path here, so the gate's own
requirement to pin the executed backend is met. The unexplained half of #1679 is
untouched by this run: why a `FLASH_ATTN` path runs on GB10 at all against the
recorded measurement that this wheel's FlashAttention carries no sm_12x code.
Nothing here explains it and nothing here may be read as closing it.

**NO SPEED NUMBER EXISTS ON THIS ROW.** The run produced none and records none,
and none may be derived from it: no clock control inside a lease (#1354), no
denominator, and no speed axis in this row's scope. **The run's duration is not
a result.** The FP8 speed cells stay open gaps owed by a benchmark row with
host-shell authority.

**What the pass does not reach.** All 2736 dispatches took swap_ab, because
decode runs at `M = 1` against the `swap_ab = (M <= 64) || (M % 4 != 0)`
heuristic, so pingpong and default stay UNEXERCISED on the model path and
`tests/vt/test_ops_matmul_fp8_block_cuda.cpp` remains their only coverage
(#1437). The harness and the resident-weight counter were SCRATCH -- recorded as
scratch, with their sha256 on the share, and not committed; the dispatch counters
they print are the tree's own. And one gate on
one device on one date is not a promise about the next run.

**This row took four leases. Three were probes and the fourth was the gate.** The
first ran our side alone with no oracle; the second and the third ran the oracle
alone with no engine of ours; the fourth ran both, compared tokens, and returned
`RESULT=PASS`. Only the fourth adjudicated anything, and none of the four
produced a performance number.

## Owed

- **The gate itself is DISCHARGED. It was this row's only gate and it is MET.**
  `Qwen/Qwen3.8-27B-FP8` HAS been run against the pinned oracle on this arm, on
  `dgx:gpu0`, on 2026-08-23, and every first divergence is an exact tie or in
  band: `TOKEN_VERDICT=PASS`, `SUMMARY strict=6 in_band=1 out_of_band=0
  integrity_fail=0 length_mismatch=0`. That is the debt
  `.agents/specs/vt-matmul-fp8-block-cuda.md` `## Owed` named, and this run
  removes it. What is NOT discharged by it is written in the bullets below,
  because a pass is a measurement of what ran and not of what did not.
- **The checkpoint is STAGED and the arm is REACHED and now GATED on those
  bytes.** All three states hold, and this spec kept them apart while only the
  first two did. The staging verification is unchanged and is recorded above: 81
  of 81 files, 30,890,049,597 B, 66 safetensors files with 0
  header-inconsistent, 1606 tensors = 1199 BF16 + 407 F8_E4M3, all 407 FP8
  tensors `N % 128 == 0` AND `K % 128 == 0`, and `weight_scale_inv` present 407
  times. [#1613](https://github.com/mudler/vllm.cpp/issues/1613) tracked the
  staging and staging discharged it.
- **Two of the three tile configurations are STILL UNEXERCISED on the model
  path.** The gate run dispatched 2736 GEMMs and all 2736 took swap_ab, because
  decode runs at `M = 1`. Pingpong and default were not reached by any decode on
  this arm, and `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` stays their only
  coverage (#1437). A prefill-shaped or batched workload would reach them, and
  no row here owns that measurement.
- **R5's dtype read is only PARTLY discharged.** Instrument 3 measured the
  RESIDENT WEIGHT at one byte per element over 400 tensors, which is what rules
  out a dequantizing load. It did not read the block scale's resident dtype or
  the GEMM output dtype against the oracle's, which is the wider comparison
  [`.agents/porting.md`](../porting.md) asks for and which no token gate can see.
  The committed loader and kernel tests bind those sites, and this run adds
  nothing to them.
- **The oracle's gateability on this arm is now FULLY measured for this row's
  purpose, and the third part is settled.** The oracle loads the FP8 artifact and
  generates from it (2026-08-22); its greedy decode is deterministic, now
  re-established over this row's own seven-prompt set rather than one prompt
  (`ORACLE_DETERMINISTIC=true`); the backend is pinnable and was pinned
  (`attention_backend="FLASHINFER"`, the language-model path logging FLASHINFER);
  and **the eager question is closed by the run choosing the GRAPHED shape**,
  `ORACLE_ENFORCE_EAGER=False`, `ORACLE_SHAPE=graphed`. That was the part this
  bullet carried as UNCHANGED and still owed, and it is owed no longer.
  **[#1679](https://github.com/mudler/vllm.cpp/issues/1679) STILL STAYS OPEN**:
  why a `FLASH_ATTN` path runs on GB10 at all, against the recorded sm_12x
  measurement, is untouched by this run.
- **Every speed axis, by construction, and this row never owed one.** #1354 puts
  clock control outside a lease, so none of this row's four leases could produce
  a defensible ratio and none tried. No timing figure from any of them is
  recorded anywhere or is admissible as one. The FP8 speed cells stay open gaps
  owed by a benchmark row with host-shell authority.
- **Part of the instrumentation is SCRATCH and no row owns landing it.** The ABI
  harness (`vllm_complete_tokens`) and the patch that prints `FP8_BLOCK_DISPATCH`
  and adds the `FP8_BLOCK_RESIDENT` counter live on the share with their sha256
  recorded, and neither is committed. The dispatch counters themselves are the
  tree's own and stay committed; only the resident-weight counter and the printing
  are scratch. A future run of this gate stages
  them again from that record. Turning either into tree code is not this row's
  scope and no issue asks for it.
- **[#1189](https://github.com/mudler/vllm.cpp/issues/1189) is NOT closed by
  this row.** Its `## Gate design` asks for three layers, and this run
  establishes the first and part of the third: token-exactness against the pinned
  oracle, plus the dispatch counter and the bytes-moved assertion. Its second
  layer -- a numerical lower bound on per-projection outputs tight enough to fail
  the x1.10 scale perturbation -- and the per-block scale-variance probe are not
  established here, and no `Fp8BlockStats` type exists in the tree. Whoever
  closes #1189 owes those, and this row does not.
