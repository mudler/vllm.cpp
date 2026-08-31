# The RELEASED `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact through the REPAIRED path, 31 August 2026

W5q of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2031](https://github.com/mudler/vllm.cpp/issues/2031).

**The one-line result. The W5n refusal is GONE, and the checkpoint still produces
no usable token.** `POST /v1/completions` now returns **HTTP 200** with eight
completion tokens instead of the 500 W5n measured, the forward throws nothing,
and the engine survives. Every one of those eight tokens is **id 0**, and the
answer is **byte-identical for two different prompts**. What W5p removed was the
refusal, not the defect behind it.

**Nothing here is a benchmark.** Two prompts, greedy, one repetition, no clock
window, and a shared box carrying four other sessions' processes throughout. No
number in this file may be quoted as a speed.

## 1. What was run, and on what

The composed tree `701606e51`, which merges two reviewed heads that each hold
half of what this run needs:

| branch | head | what it contributes |
|---|---|---|
| `row/MODEL-MM-QWEN4-EXP-W5P` | `ba378d461` | `vt::Qwen4ExpGatedResidual` accepts a block-quantized `mix_down`, `mix_up` and `block_inject` — the operand contract W5n's prefill died on |
| `row/MODEL-MM-QWEN4-EXP-LOADIO` | `ee3b32165` | `VT_LOAD_STATS` reports on the GGUF branch at all, and the finding that the 74-minute load is the CIFS mount |

`thor:gpu0`, pod `rc-worker-n8smh`, aarch64, kernel `6.8.12-1021-tegra`, 14 cpus,
Mem total 122 GiB, `/` an overlay with 508 G free. Device choice is the
developer's standing instruction that CPU work goes to `thor`; this row has no
production CUDA call site for any `qwen4_exp` op.

Built in the lease, CPU-only, `-j 4`: `cmake configure rc=0`, `ninja rc=0` in
**510 s**, 540 steps, `vllm-server` 20,999,688 bytes. No CUDA entered the build.

**CONTENTION WAS PRESENT THROUGHOUT AND IS NOT ESTIMATED AWAY.** `ps` before the
first timing recorded another session's `vllm-cli`, a `vllm-server` 49 h old,
three `nsys` processes, two `python3`, and four `cc1plus` processes 54 h old;
loadavg over the run ran **4.6 to 7.3** on 14 cores. Every wall time below is an
upper bound taken under that load, which is the reason none of them is offered
as a speed.

## 2. The artifact, and that it is the pinned one

`/workspace/q4exp-bench/UD-IQ1_S/`, three shards,
10,946,624 + 49,990,818,368 + 22,544,696,352 = **72,546,461,344 bytes = 67.564 GiB**.

Identity was proved on the STAGED copy, not on the share: after the copy,
`sha256sum` of shard 1 read
`88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, equal to the
pinned value, and the staged byte count equalled the artifact byte count exactly.
The job refuses and exits on either mismatch, so the run could not have proceeded
against a different file.

## 3. Staging and load are reported as TWO numbers, never one

| step | wall | rate |
|---|---:|---|
| staging copy, 67.564 GiB CIFS -> worker-local `/tmp` | **1860 s** | 37.2 MiB/s |
| load from local disk, page cache evicted first | **61 s** | — |

**LOADIO's prediction is confirmed on an independent run.** It measured 4446 s
from CIFS against 60 s from local disk; this run, on the same box with a
different binary, reads **61 s**. The staging copy was FASTER here than LOADIO's
2916 s (37.2 vs 23.7 MiB/s), which is a share-contention difference between two
windows and not an improvement anyone made.

`VT_LOAD_STATS=1` phase lines, verbatim from `out/server.log`:

```text
[vt load] mmap+header       0.055 s
[vt load] weights          48.788 s
[vt load] gguf prefault spans=290 paged_in=64.748 GiB in 30.775 s (2154.4 MiB/s)
```

The prefault is 30.8 s of the 61 s, matching LOADIO's split within the spread it
already recorded.

**Peak RSS: `VmHWM` 77,525,952 kB = 73.935 GiB** against a 67.564 GiB artifact
(`VmPeak` 148,393,304 kB is address space, not residency). W5n's CIFS run read
69.206 GiB. **This wave does not explain the 4.7 GiB difference and does not
claim one**: the two runs differ in filesystem, binary and page-cache history at
once, and nothing here isolated them.

## 4. The gate, run on the composed tree BEFORE the artifact

Whole binaries, no filter — a doctest filter that matches nothing prints
`assertions: 0 ... SUCCESS!` at rc 0, so nothing here is filtered.

| binary | cases | assertions | failed | skipped |
|---|---:|---:|---:|---:|
| `test_qwen4_exp_hc_device` | 11 | 516 | 0 | **0** |
| `test_qwen4_exp_gguf_weights` | 12 | 3074 | 0 | **0** |
| `test_qwen4_exp_layer_loop` | 6 | 309 | 0 | **0** |
| `test_gguf_keep_quant` | 43 | 6481 | 0 | **0** |
| **total** | **72** | **10,380** | **0** | **0** |

Read from `out/gate-<binary>.txt`, one file per binary. The oracle golden is
**UNMOVED**, read from
`out/gate-test_qwen4_exp_layer_loop.txt:7`:

```text
test_qwen4_exp_layer_loop.cpp:571: MESSAGE: layer loop vs transformers 5.16.0:
max|diff| = 0.00982457 against a bound of 0.03
```

The three suites W5p's own record names agree with it exactly (11/11 516,
12/12 3074, 6/6 309); the fourth is LOADIO's.

## 5. What happened: it SERVES, it RETURNS 200, and the answer is a CONSTANT

The server reached `/health` in 61 s, `/v1/models` answered, the tokenizer and
the 9993-character chat template came out of the GGUF's own metadata, and the
engine clamped `max_num_seqs` from 32 to 1 exactly as W5L's recurrent-state
budget says it should.

**Request 1**, verbatim body sent:

```json
{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":8,"temperature":0}
```

`curl rc=0`, **http=200**, wall 12 s. Response body, verbatim:

```json
{"choices":[{"finish_reason":"length","index":0,"logprobs":null,"prompt_logprobs":null,"text":"!!!!!!!!"}],"created":1788146729,"id":"cmpl-0","model":"qwen4exp","object":"text_completion","usage":{"completion_tokens":8,"prompt_tokens":5,"total_tokens":13}}
```

**Request 2**, a DIFFERENT prompt, because a constant answer is not a token and
one request cannot tell the two apart:

```json
{"model":"qwen4exp","prompt":"Water boils at","max_tokens":8,"temperature":0}
```

**http=200**. Response body, verbatim:

```json
{"choices":[{"finish_reason":"length","index":0,"logprobs":null,"prompt_logprobs":null,"text":"!!!!!!!!"}],"created":1788146741,"id":"cmpl-1","model":"qwen4exp","object":"text_completion","usage":{"completion_tokens":8,"prompt_tokens":3,"total_tokens":11}}
```

The two `text` fields are byte-identical across prompts of different lengths
(`prompt_tokens` 5 and 3).

**`!` IS TOKEN ID 0 IN THIS CHECKPOINT'S OWN VOCABULARY, and that was read off
the file rather than assumed.** Parsing shard 1's GGUF key-value block directly:
`general.architecture = qwen4exp`, `tokenizer.ggml.tokens` has 248,320 entries,
and ids 0-5 are `'!'`, `'"'`, `'#'`, `'$'`, `'%'`, `'&'`. So the eight-character
answer is **token id 0, eight times** — the value `argmax` returns over a logit
row that has no maximum to find.

The engine's own step log shows the work was really done rather than skipped:

```text
INFO prefill id=cmpl-0-0 status=begin prompt_tokens=5 already_computed=0 remaining=5 scheduling=5 chunked=0
INFO prefill id=cmpl-0-0 computed=5/5 (100%) status=done elapsed_s=3.04162 prefill_tok_s=1.64386
INFO core-step end model_executed=1 n_out=1 elapsed_s=1.593
... seven more decode steps, model_executed=1 each
```

A full 5-token prefill and eight decode steps ran, each executing the model. No
`engine-fatal`, no `vt:` refusal, and no warning of any kind appears anywhere in
`out/server.log`.

## 6. What this DOES and DOES NOT establish

**Established.**

1. The W5n blocker is gone at its source. The released file's 194 Q8_0
   hyper-connection mix weights now pass `check_projection`, the fused mixer
   consumes them, and the first prefill completes instead of throwing
   `input_mix_weight_down must be float`.
2. The whole production path runs on the released artifact: load, cache sizing,
   HTTP, prefill, eight decode steps, sampling, detokenization, and a 200.
3. Local staging removes the load as an obstacle, independently reproduced at
   61 s against 4446 s.
4. The forward's output on the REAL weights is degenerate and
   prompt-independent.

**NOT established, and no sentence above should be read as any of it.**

- **Why the logits are degenerate.** Nothing read them. `logprobs` was not
  requested and the server returned `null` for it, so this run CANNOT
  distinguish an all-`NaN` row, an all-zero row, and a constant non-zero row.
  All three sample id 0. Naming the cause needs an instrumented run, not another
  request.
- **Which op is responsible.** No mutation, no bisection and no per-layer probe
  was run against the released weights. W5p's own `## Owed` entry predicted this
  class — every `vt::` op in this loop has a dtype contract the synthetic fixture
  exercises on exactly one side — but predicting a class is not identifying a
  site, and this file names none.
- **That the mixer repair is CORRECT rather than merely accepted.** The gate
  proves the quantized arm is reached and agrees with a double reference on a
  miniature. It does not prove the released file's `[10240, 320]` and
  `[320, 10240]` Q8_0 projections produce right values at the released geometry.
- **Any speed.** Prefill read 1.64 tok/s and decode steps 0.8-1.6 s under
  loadavg 4.6-7.3 with four foreign process groups on the box. These are wall
  times recorded for liveness, not measurements.

## 7. Two operational facts this run adds

**THE `tee`-PLUS-EXIT-TRAP HANG IS REAL AND HAPPENED AGAIN, IN ITS THIRD
RECORDED INSTANCE.** The job printed `=== DONE ===` and `cleanup done` at
03:25:55 and `rc devices` still reported `thor:gpu0` **busy** with it minutes
later, at 42m6s elapsed. This job carries LOADIO's own fix — no bare `wait` in
the trap — and still held the device, so the workaround is not sufficient and the
trap is not the whole cause. `rc kill <full job id>` released it immediately and
thor picked up the next queued job within 8 s. **Verifying `rc devices` after
DONE is what caught it**; the job's own log says it finished.

**THE STAGED COPY'S RECLAMATION IS UNPROVEN.** The cleanup's `du` measured 69 G
present, `rm -rf /tmp/w5q` ran, and the `df` on the very next line — same
timestamp, 03:25:55 — read 431 G used against 364 G before the job. The server
process was still listed in `ps` one second earlier at 316% CPU. So the `df` was
taken before the unlink could have settled and possibly while the file was still
open, and this wave CANNOT say whether the 67 GB was reclaimed on
`rc-worker-n8smh`. Recorded under `## Owed` rather than asserted either way.

## 8. What may NOT be quoted from this file

No throughput, latency or memory-efficiency figure. No comparison against
llama.cpp or any oracle: no same-box control was run, and the `llama-cpp-qwen4exp`
pin aborts on this box before loading a byte (LOADIO). No claim that the model
works. The correct one-sentence summary is the first paragraph.
