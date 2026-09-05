# Qwen3.8-27B on ONE quantization: our engine on pangoleen's own checkpoint

| Field | Value |
|---|---|
| Issue | [#2761](https://github.com/mudler/vllm.cpp/issues/2761) |
| Owning row | `BENCH-QWEN38-27B-SOTA` ([backend-matrix](../backend-matrix.md)), whose parent spec is [bench-qwen38-27b-sota.md](bench-qwen38-27b-sota.md) |
| Subject | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9b5f1b79dc11246341960360e6ef05ef4`, draft `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` @ `bd7a934213c47a9e7ef69eef36bb3325f47fd1f1` |
| Comparator | [`pangoleen/qwen3.8-27b-dgx-spark-dflash2`](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2), SGLang + DFlash2 on one DGX Spark, `RESULTS.md` and `data/*.csv` read at clone on 2026-09-03 |
| Host | `dgx.casa`, GB10 `sm_121a`, under one `rc` lease |
| Status | `SPIKE`. **No number in the results section is measured yet.** Section 2 is arithmetic over two checkpoints' safetensors headers; sections 3 and 4 are hermetic reads of this tree's own loaders. The device job is queued |

## 1. Why this leg, and what it replaces

[`docs/benchmarks/qwen38-27b-exl3-gb10.md`](../../docs/benchmarks/qwen38-27b-exl3-gb10.md)
records 16.74 tok/s target-only and 59.5 with the DFlash2 draft, on
`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`. pangoleen records a 12.71 tok/s no-drafter
median and about 70 tok/s with a drafter, on `RadixArk/Qwen3.8-27B-NVFP4` under
SGLang. Placing 16.74 beside 12.71 reads as "our target forward is 1.3x theirs",
and section 2 shows that reading is an artifact of the checkpoint rather than a
result about either engine.

The leg therefore does one thing: it runs OUR engine on THEIR checkpoint, so the
quantization stops being a free variable.

## 2. The byte arithmetic, which reframes the question before any GPU time

Both checkpoints' tensor names, dtypes and shapes were read from their own
safetensors headers by HTTP range request over the first 4 MiB of each shard, on
the revisions in the header table. Decode-resident means every tensor a text
decode step reads once per token: the whole model minus the vision tower, minus
`embed_tokens` (gathered by row, never swept) and minus the MTP head the engine
does not execute. `lm_head` is included, because a decode step reads all of it.

| | `Mia-AiLab` EXL3 3.5bpw | `RadixArk` NVFP4 |
|---|---:|---:|
| total on disk | 15.338 GB | 21.921 GB |
| MLP | 7.039 | 9.626 |
| GDN `linear_attn` | 2.823 | 5.588 |
| self-attention | 0.844 | 1.678 |
| `lm_head` | 0.954 | 0.715 |
| everything else | 0.001 | 0.001 |
| `embed_tokens` (excluded) | 2.543 | 2.543 |
| vision tower (excluded) | 0.921 | 0.921 |
| MTP (excluded) | 0.213 | 0.849 |
| **decode-resident** | **11.661 GB** | **17.608 GB** |

Both artifacts ship an MTP head this engine does not execute, and both are
excluded: 39 trellis tensors here, 15 bf16 ones there.

Their box measures about 231 GB/s effective against a 273 GB/s spec figure
(`RESULTS.md` header; their section 3 re-derives 235 GB/s from a `ms/pass` fit
over 15 rungs, R² 0.86). Ours is the same silicon class and this tree has used
273 GB/s as the spec roof.

| | tok/s, no drafter | x decode-resident GB | achieved GB/s | of 231 GB/s |
|---|---:|---:|---:|---:|
| SGLang, `RadixArk` NVFP4 | 12.71 | 17.608 | **224** | **97%** |
| ours, `Mia-AiLab` EXL3 | 16.74 | 11.661 | **195** | **85%** |

Three things follow, and they are the reason this spec exists.

**Their target-only decode is at the bus.** 97% of the bandwidth their own
measurement says the box delivers. No engine can be much more than 3% faster on
that checkpoint, ours included. A "we beat them target-only" outcome on matched
quantization is therefore not available to be won, and any such claim would be a
measurement error.

**Byte-normalized, the 1.3x reverses sign.** 195 against 224 GB/s is 0.872x. The
EXL3 checkpoint is 1.51x smaller than theirs, which is more than the whole 1.32x
throughput difference. This is an indication and not a verdict: EXL3's trellis
decode does real work per byte, so our 85% may be compute-bound rather than a
scheduling loss, and the two figures are on different formats, different engines
and different harnesses. The measurement that settles it is leg G.

**The contest is therefore the speculation multiplier, not the target forward.**
Their own summary says the drafter buys 5.5-5.9x, and their recommended recipe's
shortest rung reads 72.7 against a 12.61 no-drafter figure on the same box, which
is 5.77x. Per prompt in `data/lossless.csv` the multiplier runs 1.9x to 9.0x,
median 4.85 over the nine prompts that emit at least 32 tokens, so a single
figure for it is not meaningful. Ours is 2.91x on the favourable prompt and 3.56x
on real HumanEval at T = 0.6. **These multipliers are on different workloads and
different stopping rules and do not compare directly** — see axis (3) below. What
does compare is the mechanism metric: their accepted tokens per pass at budget 16
is 6.8-8.1; ours saturates at 6.24 from budget 12.
Both drafters are the SAME architecture — the two `dflash_config` blocks are
byte-identical (`block_size` 8, taps `[5,19,33,47,61]`, `selector_rank` 256,
`selector_top_k` 16, `mask_token_id` 248070) and the two configs differ in
`quantization_config` and nothing else. The drafters are also nearly the same
size: EXL3 5.0bpw is 1.471 GB, the ModelOpt NVFP4 one 1.550 GB. So their
published "+5 to +10% from the NVFP4 drafter, 3.85 GB to 1.45 GB per pass" is a
lever against a **bf16** drafter, and our EXL3 drafter already has it.

### What this tree has already measured on a ModelOpt NVFP4 27B on GB10

The prediction for leg G is not a guess. `docs/benchmarks/vllm-online-serving.md`
records this engine against vLLM on `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160`,
which is the same shape of artifact — ModelOpt `MIXED_PRECISION`, NVFP4 MLP next
to an FP8 tower — on this same box:

| | c1 |
|---|---|
| ours / vLLM, decode TPOT, at the pin, graphed, `--language-model-only`, clocks pinned 2184 MHz | **0.976x** |
| ours / vLLM, decode throughput, same series | 0.973x |
| ours / vLLM tok/s, canonical 2026-08-11 (superseded, optimistic) | 10.756 / 11.250 = 0.956x |

That page also names where the loss is, in bytes rather than in adjectives.
`lm_head` ships U8 NVFP4 at 0.666 GiB and we read 2.368 GiB of BF16 unless
`VT_LMHEAD_FP4` keeps it packed: **+1.702 GiB per step, 11.183 ms**. The GDN
`in_proj` moves an identical 6.7188 GiB per step on both arms and we spend 96
GEMMs at 165.9 GiB/s against vLLM's 48 merged `qkvz` at 204.3, with
`in_proj_qkv` itself at **129.3 against 213.6 GiB/s**. Two levers exist for that
second one, `VT_GDN_PACKED_DECODE_FP8_TOWER` and `VT_GDN_FP8_IN_BF16`, both
**default off**, worth +0.7% to +2.7%.

So the honest prior for leg G is that we land at or slightly below their 12.71,
not above it, and the named mechanism is already written down. Leg G either
reproduces that position on a second checkpoint or falsifies it, and both are
worth the lease.

## 3. Can we load their target? The config gate says yes

`RadixArk/Qwen3.8-27B-NVFP4` declares `quant_method: "modelopt"`,
`quant_algo: "MIXED_PRECISION"`, and a `quantized_layers` map of 401 exact module
names: 208 `FP8` and 193 `NVFP4` `group_size` 16. Its `config_groups.group_1`
sets `input_activations` `{dynamic: false, num_bits: 4, group_size: 16}`.

Both whole-checkpoint refusal gates in the Qwen3.5 dense loader were run against
the real `config.json` and the real 2194 shipped tensor names from
`model.safetensors.index.json`. The probe was compiled and run twice, at
`3047871581bc55a0ab1a44006421bbe02698d5b8` and again at
`309dfaa19a0265287a0879b5851ce07737d954ef` after this branch merged 32 commits
of `origin/main`, and both runs answer identically:

| gate | answer |
|---|---|
| `layers::compressed_tensors::RefusalForHfConfigRaw` | `""` (not compressed-tensors) |
| `layers::modelopt::MixedPrecisionConfig::IsMixedPrecision` | `true` |
| `layers::modelopt::RefusalForQuantizationConfig` | `""` |

The same probe run against `unsloth/Qwen3.8-27B-NVFP4`, which sits on the NAS and
is the subject of the queued `/workspace/dflash2-staged/nvfp4.sh` job, answers
with a 233-module refusal naming per-output-channel `weight_scale` and dynamic
per-token activations. That artifact is not loadable by this tree, and the
difference between the two is the whole reason this row's subject moved.

**`RadixArk` and `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` are the same weights.**
Shard 1's 970-tensor committed fixture
(`tests/vllm/models/qwen38_27b_modelopt_mtp_s1_manifest.inc`) is a subset of
`RadixArk`'s 2194 tensors with **zero** dtype or shape disagreements, and
`r0b0tlab`'s shard 3 plus shard 4 sum to `RadixArk`'s shard 3 to within a
safetensors header. So the artifact this row has been scoped against since
2026-08-21 is a re-split, re-labelled copy of the one pangoleen actually serves.

## 4. Three axes that do NOT match, named before they are measured

**(1) They run W4A4; we will run W4A16.** `r0b0tlab` relabels all 193 modules
`W4A16_NVFP4` and ships no `input_scale` on them. `RadixArk` declares `NVFP4`
and ships an `input_scale` on all 192 MLP modules and on `lm_head`. This tree
routes a projection by which tensor NAMES are present rather than by the
declaration (`qwen3_5_dense_weights.cpp`; the tree records this as #1597 in
`.agents/completed/issue-index.md`, and that number does not resolve on the
forge from this session, so its state is `REMOTE_UNVERIFIED` rather than closed),
and
`VT_MODELOPT_W4A4` defaults to `0` because consuming `input_scale` produced
incoherent text on `nvidia/Qwen3.6-27B-NVFP4` (`docs/ENVIRONMENT.md`). So we will
execute the weight-only arm against a checkpoint whose producer declared static
fp4 activations, and nothing in the output says so. The weight bytes per step are
the same, so section 2's arithmetic is unaffected; the activation path and the
numerics are not. [#2760](https://github.com/mudler/vllm.cpp/issues/2760).

**(2) We cannot load their drafter at all.** The DFlash2 draft loader asks
`dense_loaders::IsExl3Projection(has, "fc")` once and otherwise takes a BF16
reader; there is no ModelOpt or compressed-tensors arm for a draft's own
weights, and the draft's `quantization_config` is never read. The NVFP4 drafter
will pass `fc` (its own `exclude_modules` leaves `fc` unquantized) and then die
inside layer 0 on `ConcatRawNK` with `qwen3_dflash: expected BF16 for
layers.0.self_attn.q_proj.weight` — a message that never says NVFP4. That is
both a missing arm and a refusal that does not name the missing part, which
`AGENTS.md` §"Shared seams" requires. Job leg F captures the verbatim string;
leg H substitutes the EXL3 drafter of the same architecture.
[#2758](https://github.com/mudler/vllm.cpp/issues/2758).

**(3) Their harness honours EOS through a chat template; ours cannot.**
`bench/ctxsweep.py` posts to `/v1/chat/completions` with
`chat_template_kwargs {enable_thinking: false, preserve_thinking: false}`, a 512
`max_tokens` cap and no `ignore_eos`. Across the 25 rows of
`data/ctxsweep-recommended.csv` `out_tokens` is min 170, median 269, max 333, so
**not one of 25 reaches the cap**: their generations terminate naturally.
`examples/bench/bench_core.h:558` sets `sp.ignore_eos = true` unconditionally,
with no flag and no environment variable, and sends raw prompts with no chat
template. `vllm-cli` honours EOS but applies no chat template and reports no
acceptance counters. `vllm-server` has the chat template and natural EOS but
exposes no speculative-decoding metrics over HTTP — the counters exist at
`runner.h:269-270` and their only non-test readers are the two lines in
`bench_core.h`. **No harness in this tree can reproduce their protocol today.**
[#2759](https://github.com/mudler/vllm.cpp/issues/2759).

Two consequences the report must carry rather than round away. Suppressing EOS
decodes past where the model would have stopped, into lower-entropy
continuation, which can raise draft acceptance; that would flatter us, so any
figure taken with `ignore_eos` prints its observed output lengths beside it.
And their median generation is 269 tokens against our 128, which amortizes
per-request overhead over more steady-state decode and mildly flatters them.

Their counting convention is `completion_tokens / (wall - ttft)`. Ours is
`1 / mean_tpot` with `tpot = (latency - ttft) / (output_len - 1)`. The two differ
by a factor of `(n-1)/n`, so at their median 269 tokens ours reads about 0.4%
high. That is inside every other uncertainty here and is recorded, not corrected.

## 5. The job

`/workspace/nvfp4-sota/job1.sh`, source tarball sha256
`66f19bbeadc8467851ec22c15798ea8c7d5bca7288d13a20c778d069e15cdcd5` cut from
`309dfaa19a0265287a0879b5851ce07737d954ef`, both asserted inside the job.

That pin was moved once, deliberately. The branch was cut at
`3047871581bc`, and `origin/main` gained 32 commits while the spec was written,
three of them in `dense_attn_block.h`, `dense_device_glue.h` and `qwen3_5.cpp` —
the decode path of this very model family. Measuring the older tree would have
produced a number for a tree nobody runs. Legs, in the order a crash truncates
them least usefully:

All four staged files were hashed on the download host and **all four match the
publisher's LFS object hash**: `fbcdb5ba...`, `db6146a5...` and `d3cfb927...`
for the target's three shards, `2228b9b2...` for the drafter. Shard 2 needed five
disjoint HTTP ranges reassembled by hand after a single stream stalled, so its
hash is the check that the reassembly is the file.

| leg | question |
|---|---|
| B | sha256 of every shard, recomputed **on the device** against those four |
| E | does `RadixArk` load and decode coherently? The device answer nothing in this tree has ever taken for a Qwen3.8-27B NVFP4 artifact |
| F | the verbatim failure of the NVFP4 DFlash2 drafter |
| I | **the saturation question**, on the EXL3 pair alone, so a target that refuses does not cost it |
| G | target-only decode, NVFP4 and EXL3, **interleaved on one binary and one boot** |
| H | NVFP4 target with the EXL3 DFlash2 drafter at budgets 7, 12 and 16 |

Every timed leg is resumable from its own log, because this box has rebooted
three times in one session under a ladder of this length.

**Leg I is the cheapest leg with the largest fork in it, and it needs neither
their target nor their drafter.** Our accepted tokens per pass saturates at 6.24
from budget 12 onward on free-form work, so budgets of 16 and 24 buy strictly
more draft work for no more accepted tokens. pangoleen's `bench/accfix.py`
measures the same drafter architecture on a task where the model reproduces a
fixed block verbatim, and reads **14.45-14.53 accepted of a budget of 16** — 90%
saturation, stable to 0.2% from a 1k prefix to a 130k one — against 7-9 on their
free-form sweep. On their engine the ceiling is therefore the TASK.

Leg I runs the same shape on our EXL3 pair: 32 copies of one
reproduce-this-class-exactly prompt, budgets 7 and 16. If we reach about 14, the
free-form 6.24-against-7.5 gap is drafter quality and the answer is a better
drafter. If we stall near 6 on text a drafter can predict almost perfectly, the
ceiling is in our propose-and-verify loop rather than in the checkpoint, and that
is a defect in this engine that no amount of quantization matching would have
found. The two outcomes are more than 2x apart, so one sample per rung separates
them.

## 6. Gates

- Leg E is a load gate, not a speed gate: `rc=0` plus a coherent continuation.
  A wrong codebook on this format yields a correctly distributed and entirely
  wrong weight, so coherent factual text is evidence.
- No speed figure is quoted until leg G's two NVFP4 legs, separated by an EXL3
  leg, agree inside the 0.75% this pair has previously held.
- Their own caution travels in reverse: boot-to-boot median absolute difference
  4.3%, worst 8.1%. A difference smaller than 4.3% against a published rung of
  theirs is not a difference.
- Nothing from this row is published in `docs/benchmarks/` until a leg produces
  a number. Section 2 is arithmetic and says so.

## 7. Stop conditions

- Leg E refuses: that is a complete answer. Record the verbatim refusal, stop.
- The paged draft route faults (#2274): legs H run `VT_DFLASH_PAGED=0`, as every
  published DFlash2 figure in this tree already does. Whether the fault also
  fires on this target is itself recorded.
- `max_num_seqs` or `max_model_len` is clamped on the KV budget: the leg is
  still valid at concurrency 1, and the clamp is read off stderr rather than
  from the requested value, which is a tautology.

## Now

`SPIKE`. Sections 2, 3 and 4 are established. The device job is queued on
`dgx:gpu0` at position 10 behind a 10-hour job, so no measurement exists.
Next action: read `/workspace/nvfp4-sota/out1/results.txt` when the job lands,
starting with leg E.

## Owed

- Every number in section 2's second table, re-derived on our own box rather
  than taken from pangoleen's bandwidth figure.
- A harness that can honour EOS and apply a chat template, without which their
  protocol cannot be matched: [#2759](https://github.com/mudler/vllm.cpp/issues/2759).
- A ModelOpt NVFP4 arm for the DFlash2 draft weights, without which the drafter
  half of the comparison cannot be matched at all:
  [#2758](https://github.com/mudler/vllm.cpp/issues/2758).
- The declared-W4A4-runs-as-W4A16 divergence on this target, which no output
  names: [#2760](https://github.com/mudler/vllm.cpp/issues/2760).
- The `## Outcome` section this spec owes at `DONE`.
