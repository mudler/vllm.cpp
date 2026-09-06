# `variadic-load-methodology` — how the mixed-load serving benchmark works

This page describes a method, not a result. It says what the load is, how the
statistics are computed, what is discarded, how two engines are made
comparable, and what the harness cannot see. Read it before you quote a number
from a page that cites it, and read it before you rerun the harness on something
else.

The harness is [`benchmarks/variadic/`](https://github.com/mudler/vllm.cpp/tree/main/benchmarks/variadic).
It is parameterised. Nothing in it is specific to one model.

| | |
|---|---|
| Harness | `benchmarks/variadic/{build_corpus.py, client.py, report.py, job.sh}` |
| Controls | `tests/scripts/test_variadic_harness.py`, in preflight and in CI |
| Upstream anchor | vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`, `vllm/benchmarks/serve.py` |
| First run | [`qwen38-27b-exl3-variadic-gb10`](qwen38-27b-exl3-variadic-gb10.md) |
| Predecessor | [`qwen38-27b-exl3-gb10`](qwen38-27b-exl3-gb10.md), one prompt band at concurrency 1 |

## The load model

### Closed loop at a fixed concurrency

`C` client workers each take the next prompt from one queue and send it as soon
as their previous request completes. The offered load is therefore `C`
outstanding requests, whatever the server does with them.

This is vLLM's own `--request-rate inf --max-concurrency C`
(`vllm/benchmarks/serve.py:787`).

vLLM also offers a gamma arrival process with a burstiness factor
(`serve.py:463`; burstiness 1 gives Poisson). This harness does not use it, and
the reason is not a preference. Against a server that serializes generation, a
fixed arrival rate above its service rate makes the queue grow without bound.
The measured latency then reports how long the leg ran rather than a property of
the engine, and two engines with different service rates cannot be put on one
arrival rate at all. A closed loop offers the same bounded load to both sides at
every rung.

**What that costs.** A closed loop cannot show you a queue collapsing, and it
cannot answer "what request rate can this box hold". If you need those, add an
open-loop mode; the client's worker pool is the only part that changes.

### The prompt-length distribution

`build_corpus.py` builds the prompt set. It is a pure function of the source
bytes, the seed, the band weights and the count, so the same command on another
machine produces the same file. The manifest it writes carries the sha256 of
every source and of the corpus itself.

Four bands, one composition rule each:

| Band | Default share | Source | Rule |
|---|---|---|---|
| `S` short question | 35% | GSM8K `test.jsonl` | one `question` field, verbatim |
| `M` code completion | 40% | HumanEval `HumanEval.jsonl` | one `prompt` field, verbatim |
| `L` prose summary | 15% | vLLM `benchmarks/sonnet.txt` | a contiguous line block, under a fixed instruction |
| `XL` long code review | 10% | HumanEval `HumanEval.jsonl` | `k` problems concatenated, under a review instruction |

The `L` block length and the `XL` problem count are drawn per prompt, so each
band has its own internal spread instead of one length repeated.

**Why these sources.** HumanEval keeps continuity with the predecessor run and
is the only band the two pages share. GSM8K supplies real short natural
language, which HumanEval has none of. `sonnet.txt` is vLLM's own benchmark
corpus, so it is pinned by the vLLM parity pin and the job needs no download.

**Why the `L` band takes a contiguous block and not a shared prefix.** vLLM's
`SonnetDataset` builds every prompt on one fixed prefix. That is useful when you
want to measure a prefix cache and wrong when you do not, because it turns a
length distribution into a cache-hit distribution. Here the blocks start at
independent offsets.

**What the mix cannot claim.** It is not a trace of production traffic. No such
trace is pinned in this repository, and a benchmark job may not download one.
Three of the four bands are code or arithmetic, so the register is narrower than
a real chat deployment. What it is: variadic in length, which is the axis this
method measures, and reproducible byte for byte, which a trace would not be.

### The realised histogram is read back, never assumed

`build_corpus.py` has no tokenizer, so its targets are in characters. The
published length histogram comes from `usage.prompt_tokens`, which each server
reports for each request. Both engines' histograms are published, because two
chat templates can render the same text to different token counts.

## Warmup, and why both views are published

The client sends `W = max(C, 4)` warmup requests at the head of each leg and
runs them **to completion** before the measured window opens. Overlapping them
would put a cold request inside the measured wall clock, which is the thing the
discard exists to remove. vLLM does the same discard (`serve.py:875`,
`--num-warmups`).

The warmup requests are recorded, not thrown away. The report prints every table
twice:

- **warm only** — warmup excluded. This is the headline.
- **with warmup** — every request.

The predecessor's mean time to first token was dominated by request `i=0`, which
cost 27.5 s and 25.2 s on our side against 7.9 s and 1.9 s on theirs. A silent
discard hides that and a silent include distorts everything else. Publishing
both from one set of records lets a reader see the size of the choice.

The report also prints a cold-start table: the first warmup request's time to
first token, beside the warm median.

## The statistics

### Percentiles

Linear interpolation between order statistics, which is `numpy.percentile`'s
default and what vLLM uses (`serve.py:739`). `report.py` implements it directly
so the report runs on a container with no numpy:

```text
idx = (n - 1) * p / 100
value = x[floor(idx)] + (x[ceil(idx)] - x[floor(idx)]) * (idx - floor(idx))
```

A nearest-rank implementation gives a different answer, and
`tests/scripts/test_variadic_harness.py` pins the difference.

### The axes

Mirrored from `vllm/benchmarks/serve.py:321` `BenchmarkMetrics`:

| Axis | Definition | Upstream |
|---|---|---|
| `ttft` | time to the first content-carrying chunk | `serve.py:615` |
| `tpot` | `(latency - ttft) / (output_tokens - 1)` | `serve.py:610` |
| `itl` | gap between consecutive streamed chunks | `serve.py:614` |
| `e2el` | client-side request latency | `serve.py:616` |

`ttft` counts the first chunk that carries **content**. A role-only opening
delta would otherwise read as a token and shorten the figure on whichever engine
happens to send one. Both `content` and `reasoning_content` count, because one
of these two servers routes the model's `<think>` block into the second field,
and reading only the first measures its time to first token to the end of the
reasoning block.

**`tpot` is the primary inter-token axis. `itl` is secondary.** `tpot` is
normalised by the token count the server itself reported, so it is one quantity
on both engines. `itl` is a gap between chunks, and a chunk is one token only on
an engine that streams one token per chunk. The `itl` table always carries the
tokens-per-chunk figure beside it.

### Reported percentiles, and what `n` allows

p50, p90, p95, p99 and the maximum. A leg's own table stops at p95: p99 over 128
requests is the second-worst request, which is a maximum wearing a percentile's
name. The table's column headings are generated from the same
tuple the values are computed from, so a percentile cannot be printed under
another one's heading. p99 and the maximum are published only from the pool of every round of a
cell, and the pooled `n` is printed in the table.

### Throughput

```text
output tok/s     = sum(completion_tokens) / measured wall clock
decode-only tok/s = 1000 / mean(tpot)
```

The first has prefill in the denominator and the second does not. Neither is
privileged; both are printed for both engines, computed from the same records.
Published figures elsewhere often do not say which convention they used, which
is what made the predecessor run necessary.

### Spread, and how many samples a number needs

Every cell runs at least twice, in different rounds, on different server boots.
The report prints both values and their relative spread, `max/min - 1`.

This is not caution for its own sake. This repository has one unchanged binary
reading 36.82 and 78.86 tok/s in the same session at concurrency 8, and it has
speculative legs spreading 24% while target-only legs held 0.35%. A single
sample at a high rung is not a measurement here. A cell whose rounds disagree by
more than 10% is quoted as a spread and not as a value.

Round 2 reverses both the arm order and the rung order. A monotone drift over
the session then biases each rung in opposite directions in the two rounds, so
it appears as a rung-to-rung disagreement between rounds rather than as a shape
in the sweep.

## Making two engines comparable

| Axis | How |
|---|---|
| Client | One process, one timing path, both engines. It does not know which engine answers. |
| Chat template | Each engine applies its own, server side, as both published protocols do. |
| Sampling | Identical on both sides, and stated on the results page. |
| Token counts | From each engine's `usage`, never from a count of streamed chunks. A leg is publishable only when EVERY successful request carried a server-side count; one that did not reads `publishable = NO (G-USAGE)` in the per-leg table. A mixed leg is the dangerous case, because its throughput would divide the tokens it could count by a wall clock containing the requests it could not. |
| `ignore_eos` | Not sent. Both published protocols let the model stop. |
| Acceptance | From `/metrics` where the engine exports it, from `usage` where it does not, and **absent** where neither. Absent is never rendered as zero, because zero reads as a draft that never fired. |
| Recipe | Each engine at its own published configuration, with every difference listed on the results page. |

### Streaming granularity, and what it does to time to first token

Two engines can stream the same tokens at different chunk sizes. The
predecessor measured 4.46 completion tokens per chunk on one side and 1.08 on
the other. A time to first token compared across those is not one quantity: the
coarser engine's first chunk carries several tokens, so its figure includes
generating them.

The client records, per request: the wall time to the first content chunk, the
character length of that chunk, the total character length, the chunk count, and
`usage.completion_tokens`. The report publishes:

| Figure | How |
|---|---|
| `tokens per chunk` | `completion_tokens / n_chunks`. Measured. |
| `chars per token` | `total_chars / completion_tokens`. Measured. |
| `first chunk tokens` | `first_chunk_chars / chars_per_token`. **An estimate.** |
| `ttft` | time to the first content chunk. Measured. |
| `ttft_corrected` | `ttft - (first_chunk_tokens - 1) * tpot`, **refused when it goes to zero or below**. **An estimate**, and derived from one. |

An over-correction is refused rather than floored. `first_chunk_tokens` is an
estimate, so at 8 tokens per chunk and a 30 ms time per output token the
subtraction is 210 ms, which exceeds a 200 ms measured figure. A floor would
publish `0.0` as though it were a measurement of an instant first token. The
report drops those requests from the corrected column and prints how many it
dropped beside it.

Both figures are published for both engines and neither is called the correct
one. The raw figure is what a streaming client observes. The corrected figure is
closer to what the engine did. The estimate exists because the OpenAI streaming
protocol carries no per-chunk token count; if you need an exact number, the
server has to emit one.

**The correction removes a chunk size. It does not remove a hold-back, and it
must not be read as doing so.** One of the two engines here buffers 16 characters
before emitting (`tools/serve_openai.py:74`) and then emits
`pending[:len(pending) - HOLD_BACK]`, so its first chunk contains the text minus
the characters that caused the delay. Its `first_chunk_chars` therefore excludes
exactly the held-back text, its estimated first-chunk token count comes out near
1, and its correction comes out near zero. The formula is applied to both engines
because it is one formula, not because it equalises them.

Two smaller asymmetries in the same direction, recorded rather than corrected.
That engine's wrapper strips `<think>` tags and left-strips the reasoning head,
so its `total_chars` is short of what its `completion_tokens` counted; its
characters-per-token is therefore understated, its estimated first-chunk tokens
overstated, and its corrected figure slightly flattered. Ours leaves the tags
inline and loses nothing. The effect is about 2%.

## What the harness cannot see

- **Anything inside either engine.** It is an HTTP client. A kernel-level
  difference reaches it only as latency.
- **Output quality.** Nothing scores what either engine writes. At any
  temperature above zero the two token streams are not expected to match, so no
  token-exact gate can run on a leg. A results page that carries throughput
  ratios states this where the numbers are.
- **Queue time separately from prefill.** Time to first token at `C > 1`
  contains both. A server that serializes will show its queue as time to first
  token, and the harness cannot split the two from outside.
- **Per-chunk token counts.** Estimated, as above.
- **Boot-to-boot variation beyond the rounds it ran.** Two rounds means two
  boots per cell. It does not mean the cell is stable across days.
- **A production traffic trace.** See the load model.

## Running it

```sh
python3 benchmarks/variadic/build_corpus.py \
    --gsm8k test.jsonl --humaneval HumanEval.jsonl --sonnet sonnet.txt \
    --count 144 --seed 0 --out corpus.json --manifest corpus-manifest.json

python3 benchmarks/variadic/client.py --url http://127.0.0.1:8811 --model ours \
    --dataset corpus.json --num-prompts 128 --concurrency 8 \
    --max-tokens 192 --temperature 0.6 --top-p 0.95 --top-k 20 --seed 0 \
    --label OURS-r1-c8 --arm OURS --round 1 --out OURS-r1-c8.json

python3 benchmarks/variadic/report.py --dir . --glob '*-r*-c*.json'
```

`job.sh` runs the whole sweep inside an `rc` lease: it provisions the container,
stages the weights off the share, checks their sha256 on the device, builds both
engines, probes the draft route, runs every leg, and writes the report. Every
phase drops a marker and every leg writes its JSON as it finishes, so a
resubmission after a crash resumes. Each knob is an environment variable:

```sh
RUNGS="1 4 8" ROUNDS=2 NPROMPTS=128 OUTLEN=192 \
    setsid nohup rc run -d dgx:gpu0 --max-runtime 6h \
    -- bash /workspace/exl3-variadic/job.sh
```

## Extending it

- **A different model or engine.** `job.sh` holds the two server command lines
  and the checkpoint pins. Nothing else in the harness knows what it is talking
  to.
- **A different load shape.** `build_corpus.py` holds the bands. Add one, or
  change the weights on the command line.
- **An open-loop arrival process.** `run_phase` in `client.py` is the only place
  that decides when a request is sent.
- **More axes.** `report.py` reads the per-request records, so a new statistic
  needs no new run.
