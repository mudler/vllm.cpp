# `qwen38-27b-exl3-gb10` — Qwen3.8-27B EXL3 3.5bpw, with and without its DFlash2 draft

The published EXL3 pair loads and generates on GB10. This file has the numbers,
the conditions they were taken under, and the parts of the upstream recipe that
are not matched here.

## Disposition

Measured. On the task and temperature the [Mia-AiLab card][card] quotes, real
HumanEval at T = 0.6, the pair decodes at 59.5 tok/s where they report 47.5. Our
acceptance is 4.06 tokens per step against their 4.43, so the speed is not
coming from an easier drafting task.

Three things about that comparison are still open and are listed under
[Limitations](#limitations). Their recipe uses a longer context and an NVFP4 KV
cache, we run with the paged draft route disabled, and their card does not say
whether its figure counts the prefill. Counted with the prefill in, the same run
of ours reads 45.1 tok/s.

There are two measurements below. The first uses a short greedy prompt and is
kept because the correctness evidence lives there. The HumanEval numbers are the
ones to compare against anything.

[card]: https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw

## Subject

| | |
|---|---|
| target | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac874c4018295da848e250f23511361cda4` |
| draft | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f0436269bca761b071f05319e8e04a87cc633f9` |
| device | NVIDIA GB10, compute capability 12.1, driver 580.173.02, nvcc 13.0, `sm_121a` |
| tree | `origin/main` `5649e07d2120df4c5d33fd1d245336490c790e2b`, pinned inside the job by tarball sha256 |
| binary | one `vllm-cli`, md5 `3bc87f47b5325468ce575d30114d7928`, serving **both** arms |

Both shard sha256 values were **recomputed on the device** and matched the
download-host pins, so the bytes measured are provably the pinned artifact:
`7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6` and
`411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2`.

## Method

Both arms ran interleaved in one process, on one boot, from one binary: target,
draft, target, draft. A sequential A/B would measure drift along with the arm,
and this repository has a recorded case of one unchanged binary reading 36.8 and
78.9 tok/s in the same session. Run 1 of every leg is cold and discarded, which
is what the harness does on every arm anyway.

```sh
VT_DFLASH_PAGED=0 vllm-cli --model <target> --device cuda \
  --prompt 'The capital of France is' --max-tokens 64 --temperature 0 \
  --seed 0 --repeat 5 --max-num-seqs 1 \
  [--speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":7}']
```

## First measurement: a short greedy prompt

| arm | warm tok/s, runs 2-5 of two interleaved legs | spread |
|---|---|---|
| target only | 16.706 16.758 16.796 16.729 / 16.737 16.769 16.670 16.701 | 0.75% |
| + DFlash2 draft, k=7 | 48.970 49.079 48.944 48.469 / 48.751 48.672 48.677 48.446 | 1.3% |

The draft arm runs 2.91 times the target-only rate. The two target legs sit
either side of a draft leg and agree to 0.75%, so drift does not explain the
gap.

## The MATCHED workload: real HumanEval at T = 0.6

The section above is a favourable prompt and says so. This section removes two of the
three differences from the upstream README's 47.5 tok/s: the real 164-problem
HumanEval set (`openai/human-eval` `data/HumanEval.jsonl.gz`, sha256
`1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`, ShareGPT-shaped
for the harness) and **T = 0.6**, their temperature. 128 output tokens, concurrency
1, seed 0, two interleaved legs on one binary (md5 `6ae2949042e256b707c2e75d0a547d9b`).

Counted **decode only**: `Mean per-stream decode rate` is `1 / mean_tpot` where
`tpot = (latency - ttft) / (output_len - 1)`, so prefill is excluded.

| arm | leg A | leg B |
|---|---|---|
| target only | 16.74 | 16.64 |
| + DFlash2 draft, k = 7 | **59.59** | **59.48** |

**Acceptance, measured rather than inferred**: 30,863 draft tokens proposed, 17,841
accepted, rate 0.58 -> **4.06 accepted per step** (5.06 emitted). The upstream README
states 4.43. So this is not a case of an easier drafting task: our acceptance is at or
below theirs and the throughput is higher anyway.

Every request produced exactly 128 tokens (`duration_s` 1355.86 x `output_throughput`
15.48 = 20,992 = 164 x 128), so no leg was skewed by early EOS.

**The counting convention is the open question, and it is theirs, not ours.** Their
README says "decode tok/s" and defines it nowhere. The same run of ours reads **59.5**
counted decode-only and **45.1** counted over whole-run wall time. Against the first
convention we are 1.25x; against the second, 0.95x. Their page pins no engine revision
either, so this cannot be settled from published material. Both numbers are given here
for that reason.

## 16.7 is a QUANTIZATION result, not an engine result

This page's 16.74 tok/s invites one comparison in particular, and that comparison
does not hold. `pangoleen/qwen3.8-27b-dgx-spark-dflash2` publishes a **12.71
tok/s** no-drafter median for the same model on the same class of box, under
SGLang, on `RadixArk/Qwen3.8-27B-NVFP4`. Read side by side that is 1.32x in our
favour. It is not.

Read from both checkpoints' own safetensors headers, the weights a text decode
step sweeps once per token — everything but the vision tower, `embed_tokens`,
which is gathered by row, and the MTP head nothing executes — are **11.661 GB**
here and **17.608 GB** there. Their checkpoint is 1.51x the bytes, which is more
than the whole throughput difference:

| | tok/s | x decode-resident GB | achieved GB/s |
|---|---:|---:|---:|
| this page, EXL3 3.5bpw | 16.74 | 11.661 | **195** |
| pangoleen, SGLang, NVFP4 | 12.71 | 17.608 | **224** |

Their box measures about 231 GB/s effective, so their target forward is at 97% of
its bandwidth limit and this one is at 85%. **Byte-normalized the comparison
reverses sign**, 0.872x. Treat that as an indication and not a verdict: EXL3's
trellis decode does real work per byte, so that 85% may be compute-bound rather
than a scheduling loss, and the two numbers are on different formats, different
engines and different harnesses. The measurement that settles it is our engine on their checkpoint, which
is [`bench-qwen38-27b-nvfp4-matched`](../../.agents/specs/bench-qwen38-27b-nvfp4-matched.md).

What follows from the table, and is not stated elsewhere on this page: the
target forward is close to the bus on both engines, so it is not where either
engine has room left. What is contested is the speculation multiplier, and the
section below is where this page measures ours.

## The draft budget is a real lever, and our knee is not theirs

`pangoleen/qwen3.8-27b-dgx-spark-dflash2` serves the SAME DFlash2 drafter architecture
-- its `dflash_config` is byte-identical to ours, `block_size` 8, taps
`[5,19,33,47,61]`, `selector_rank` 256 -- at a verify budget of 16, and states: "the
drafter's block_size of 8 is not a cap on the verify budget: 8 -> 16 is +69.3%
edit-heavy and +10.5% fresh. 24 is past the knee."

Swept on our engine, real HumanEval at T = 0.6, **32 prompts** (a hotter subset than
the 164 above: acceptance 0.69 against 0.58 at the same k = 7), one sample per rung:

| k | decode tok/s | acceptance rate | accepted per pass | proposed |
|---|---|---|---|---|
| 0 (no draft) | 16.65 | - | - | - |
| 7 | 68.67 | 0.69 | 4.83 | 5,243 |
| **12** | **76.28** | 0.52 | 6.24 | 7,536 |
| 16 | 45.21 | 0.39 | 6.24 | 10,048 |
| 24 | 40.70 | 0.25 | 6.00 | 15,504 |

Each rung's engine-reported `block` equalled `k + 1`, so every budget genuinely took
rather than being clamped back to 8.

**Accepted tokens per pass saturates at ~6.0-6.24 from k = 12 onward**, so past 12 the
arm buys strictly more draft work for no more accepted tokens and throughput falls
monotonically. **Our knee is between 12 and 16; theirs is between 16 and 24.** The
reason is visible in the acceptance: at budget 16 they report 6.8-8.1 accepted per
pass where we saturate at 6.24. Their drafter extracts more from the same budget --
theirs is NVFP4 served by SGLang, ours the EXL3 quant, and our drafter's `block_size`
of 8 is being run at a block of 17.

**These rungs are one sample each and are indicative, not gated.** This repository has
a recorded case of one unchanged binary reading 36.82 and 78.86 tok/s in the same
session. The 41% fall at k = 16 is far too large to be noise, so the knee is real; the
+11.1% from k = 7 to k = 12 is exactly the size where a single sample is weak evidence.
A repeated k = 7 vs k = 12 comparison on the full 164-problem set is owed and is what
should be quoted.

## Correctness, which is the result that gates the other one

**The two arms emit token-identical output.** Both continue `The capital of
France is` into the same list of European capitals, ending mid-phrase at the
same token. Speculative decoding must not change greedy output, and here it
does not. The speed figure is only admissible because this passed.

The target arm alone was also checked from cold in a separate job: greedy, 16
tokens, `rc=0`, ` Paris. The capital of Germany is Berlin. The capital of Italy
is`. A wrong codebook on this format yields a correctly distributed and entirely
wrong weight, so coherent, factually correct continuation is meaningful evidence
and not merely a smoke test.

## Reproduce this run

Everything here runs on one GB10 board with CUDA 13.0 and about 20 GB of disk
free. Set aside ninety minutes or so; the CUDA build eats most of it, and each
measurement leg is another twenty.

If you are going to quote a number from this, read the
[limitations](#limitations) as well. Three axes of the upstream comparison are
not matched here, and one of them is a bug we have open.

### 1. Get the weights

Both checkpoints are pinned by revision below, and it is worth pinning yours
too. Publishers do requantize in place under an unchanged repository name, so a
bare repo id can quietly get you different weights than the ones measured here.

```sh
hf download Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw \
    --revision 19441ac874c4018295da848e250f23511361cda4 \
    --local-dir ./target-3.5bpw

hf download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw \
    --revision 4f0436269bca761b071f05319e8e04a87cc633f9 \
    --local-dir ./draft-dflash2-5.0bpw
```

Then check what you got against the hashes in [subject](#subject).

```sh
sha256sum ./target-3.5bpw/*.safetensors ./draft-dflash2-5.0bpw/*.safetensors
```

If those directories live on a NAS or any other network mount, copy them to
local disk before you run anything. Otherwise the run spends a good part of its
time waiting on the filesystem instead of the GPU.

### 2. Build

```sh
cmake -S . -B build -G Ninja \
    -DVLLM_CPP_CUDA=ON \
    -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4 --target vllm-cli vllm-bench
```

`121a` is GB10. Change `-DVLLM_CPP_CUDA_ARCHITECTURES` if you are on something else.

### 3. Check that it generates

Do this before you time anything.

EXL3 stores weights as a trellis, and if the codebook resolves wrong you get
weights with the right distribution and no relationship to the real ones. Every
shape check still passes. The model still writes fluent English. It writes the
wrong fluent English, and any throughput number you take in that state is
worthless.

```sh
build/examples/vllm-cli --model ./target-3.5bpw --device cuda \
    --prompt 'The capital of France is' \
    --max-tokens 16 --temperature 0 --seed 0
```

You should get ` Paris. The capital of Germany is Berlin. The capital of Italy
is`, and an exit code of 0. It stops there because it hit the 16-token cap, which
the `finish_reason=length` line tells you. This is a plain completion prompt with
no chat template, so the model has no reason to emit an end-of-sequence token —
it just keeps listing capitals.

### 4. Build the prompt set

The harness wants the 164 HumanEval problems in the ShareGPT shape, so convert
them once.

```sh
curl -L -O https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
gunzip HumanEval.jsonl.gz

python3 -c 'import json
rows = [json.loads(l) for l in open("HumanEval.jsonl")]
json.dump([{"conversations": [{"from": "human", "value": r["prompt"]},
                              {"from": "gpt", "value": ""}],
            "id": r["task_id"]} for r in rows],
          open("humaneval-sharegpt.json", "w"))'
```

For reference, `HumanEval.jsonl` should hash to
`1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2`.

### 5. Measure both arms

`VT_DFLASH_PAGED=0` is not optional right now. With the paged draft route
enabled, the first run finishes and the second one dies inside the same process
with an illegal memory access. That is
[#2274](https://github.com/mudler/vllm.cpp/issues/2274); it predates this
checkpoint and it is still open.

Target alone first.

```sh
VT_DFLASH_PAGED=0 build/examples/vllm-bench --model ./target-3.5bpw \
    --dataset-path humaneval-sharegpt.json --num-prompts 164 \
    --output-len 128 --temperature 0.6 --seed 0 --concurrency 1
```

Then the same binary again, with the draft:

```sh
VT_DFLASH_PAGED=0 build/examples/vllm-bench --model ./target-3.5bpw \
    --dataset-path humaneval-sharegpt.json --num-prompts 164 \
    --output-len 128 --temperature 0.6 --seed 0 --concurrency 1 \
    --speculative-config '{"method": "dflash",
                           "model": "./draft-dflash2-5.0bpw",
                           "num_speculative_tokens": 7}'
```

Now run both legs again, in the same order. A board drifts over minutes, and
running target-draft-target-draft lets you see whether a difference belongs to
the arm or to the hour.

### 6. Read the result

The report gives you two throughput lines, and they answer different questions.
This page quotes the first one.

```text
Mean per-stream decode rate (tok/s):       59.59
Output (decode) token throughput (tok/s):  45.12
Draft tokens proposed:                     30863
Draft tokens accepted:                     17841
Acceptance rate (accepted/proposed):       0.58
```

`Mean per-stream decode rate` is `1 / mean_tpot`, with `tpot` being
`(latency - ttft) / (output_len - 1)`, so the prefill is out of it.
`Output (decode) token throughput` divides the same tokens by the whole wall
time, prefill included. Which one you want depends on what you are comparing
against, and plenty of published figures do not say which they used.

The acceptance counters are read straight out of the engine. Do not try to back
them out of the throughput.

### What to expect

| Arm | Mean per-stream decode rate |
|---|---|
| Target only | 16.6 to 16.8 tok/s |
| Target and draft, `num_speculative_tokens: 7` | 59.4 to 59.6 tok/s |

Your two target legs should land within about 1% of each other. If they differ
by less than that, you are looking at drift rather than at anything you changed.

### Change the draft budget

`num_speculative_tokens` moves this number more than anything else you can set
from the command line. The drafter's config declares `block_size: 8`, but that
does not cap the verify budget — the loader takes the budget from your flag and
prints what it resolved.

Try 12. Look for `block=13` in the startup output, which is the budget plus one
and tells you the flag actually took.

The [draft budget section](#the-draft-budget-is-a-real-lever-and-our-knee-is-not-theirs)
above has what each value produced on this box, including where it stops
helping.

## Limitations

Two of these apply to every number on this page. The first applies only to the
greedy measurement, and the matched HumanEval run above exists to remove it.

1. **Workload — closed for the matched run, open for the first one.** The
   published figure is HumanEval-style at **T = 0.6 with acceptance 4.43**. The
   *first* measurement here is **greedy, T = 0**, on `The capital of France is`,
   which continues into a list of capitals — close to the easiest possible text
   for a drafter to predict — and the engine reported no acceptance rate for it,
   so its gap cannot be quantified at all. The **matched HumanEval run** uses
   their task and their temperature and reports acceptance from the engine's own
   counters, so this difference does not apply to it.
2. **Context and KV cache.** The published recipe is `-cs 262144` with an NVFP4
   KV cache. Here auto-fit reduced `max_model_len` from 262144 to 8192 and
   `max_num_seqs` from 32 to 1 on the KV budget, and no NVFP4 KV cache was used.
3. **The paged draft route was disabled.** `VT_DFLASH_PAGED=0` was required:
   with the paged route on, run 1 of 5 completes and the engine dies on run 2
   with `cudaMemcpyAsync: an illegal memory access`, resurfacing at `cudaFree`.
   That is [#2274](https://github.com/mudler/vllm.cpp/issues/2274), a
   pre-existing fault, here reproduced on a checkpoint it had never been seen on
   — which shows it is not specific to the bf16 drafter it was found with. Both
   arms above ran with it off, so the comparison is internally consistent, but
   neither is the shipped default configuration.

**One prompt, one length, one device, one boot.** 64 tokens, five runs per leg,
two legs per arm. No multi-request batching, no long context, no second box.

## Owed

- ~~A HumanEval-style prompt set at T = 0.6 with the acceptance rate reported~~ —
  **done**, see the matched-workload section above: 59.5 tok/s at acceptance 4.06
  per step against their 47.5 at 4.43.
- A repeated `k = 7` vs `k = 12` comparison on the **full 164-problem set**. The
  budget sweep above ran on a 32-prompt subset that drafts hotter than the full
  set, so its 76.28 is not transferable and must not be scaled onto the 59.5.
- The remaining unmatched axes against the upstream recipe: context (8192 here
  against their `-cs 262144`) and KV dtype (bf16 here against their `-cq nvfp4`,
  which this engine refuses by name — [#2620](https://github.com/mudler/vllm.cpp/issues/2620),
  whose named owner row did not exist). `vllm-bench` also cannot select a KV
  dtype at all ([#2619](https://github.com/mudler/vllm.cpp/issues/2619)), so no
  number on this page states the KV dtype it was measured on.
- [#2274](https://github.com/mudler/vllm.cpp/issues/2274), so the shipped paged
  route can be measured rather than routed around.
- [#2570](https://github.com/mudler/vllm.cpp/issues/2570): the `m <= 8` EXL3
  GEMV. When this page was first written it instantiated `(3,1)` only, and this
  checkpoint has no `(3,1)` tensor at all, so the arm was dead on it. `(3,2)`
  has since landed, which covers the 137 bits-3 modules; `(4,2)` and its 270 is
  in flight. Upstream's GEMV takes 407 of the 409.

  Instantiating an arm is not the same as reaching it. Admission runs through an
  occupancy test, `size_n / 32 <= narrow_coresident`, and this checkpoint's
  bits-3 shapes need that to clear 544 and 160. Measured on Thor it is 60. So
  whether any of those 137 modules take the fast path on GB10 is **unmeasured**,
  and the queued job reads `SM_COUNT` and `MAX_THREADS_PER_SM` before its A/B so
  that a decline stays distinguishable from a null result. No throughput effect
  is claimed here in either direction.
