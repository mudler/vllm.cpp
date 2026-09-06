# Variadic serving load on Qwen3.8-27B EXL3: percentiles, concurrency, and both engines

| Field | Value |
|---|---|
| Issue | [#2970](https://github.com/mudler/vllm.cpp/issues/2970) |
| Owning row | `BENCH-QWEN38-EXL3-VARIADIC` |
| Published page | [`docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md`](../../docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md) |
| Methodology | [`docs/benchmarks/variadic-load-methodology.md`](../../docs/benchmarks/variadic-load-methodology.md) |
| Harness | [`benchmarks/variadic/`](../../benchmarks/variadic/) |
| Subject | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac8`, draft `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f043626` |
| Comparator | `MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed` |
| Upstream anchor | vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`, `vllm/benchmarks/serve.py` |
| Host | `dgx:gpu0`, GB10 `sm_121a`, inside an `rc` lease |
| Predecessor | [`bench-qwen38-exl3-headtohead.md`](bench-qwen38-exl3-headtohead.md) |
| Status | `ACTIVE` |

## 1. Scope

| In scope | Out of scope |
|---|---|
| A reusable, parameterised harness for a variadic serving load | A one-shot script for this checkpoint |
| A mixed prompt-length distribution built by a committed script from pinned corpora | A distribution asserted in prose |
| A concurrency sweep with interleaved arms and repeated rounds | A single sample at any rung |
| p50, p90, p95, p99 and max for TTFT, ITL, TPOT and end-to-end latency | Means as the headline |
| An explicit warmup discard, with both views published | A discard rule applied silently |
| Tokens per streamed chunk, measured on both engines | A TTFT comparison that ignores chunking |
| Acceptance from both engines | An acceptance ratio, if either side stops exporting one |
| A methodology document | A rewrite of the predecessor page |
| A correctness gate | Nothing: see §4 |

## 2. The question

The predecessor put both engines behind one client and settled the counting
convention. It measured one prompt-length band at concurrency 1 and published
means. An operator sizing a deployment reads none of the numbers they need from
it: the tail, the behaviour under queueing, and the cost of the first request.

This row measures those. It does not replace the predecessor's ratios, which
remain the concurrency-1 HumanEval result.

## 3. Design

### 3.1 The load model

Closed loop at a fixed concurrency `C`. `C` client workers each take the next
prompt from one deterministic queue and send it as soon as their previous
request completes. This mirrors vLLM's own serving benchmark at
`--request-rate inf --max-concurrency C`
(`vllm/benchmarks/serve.py:393` `get_request`, and the `max_concurrency`
semaphore at `vllm/benchmarks/serve.py:787`).

**Why closed loop and not Poisson arrivals.** vLLM also offers a gamma arrival
process (`serve.py:463`, `burstiness=1` giving Poisson). Against a server that
serializes generation, a fixed arrival rate above its service rate makes the
queue grow without bound, so the measured latency becomes a function of how long
the leg ran rather than a property of the engine. The comparator serializes by
construction (§3.5), so an open-loop rung would not be a comparison. A closed
loop offers the same bounded load to both sides at every rung.

### 3.2 The prompt-length distribution

Four bands, built by `benchmarks/variadic/build_corpus.py` from two corpora that
are pinned by sha256, with one composition rule each. The script is seeded, so
the corpus is a function of `(sources, seed, weights, count)` and nothing else.

| Band | Share | Source | Rule | Target prompt tokens |
|---|---|---|---|---|
| `S` short question | 35% | GSM8K `test.jsonl` | one `question` field, verbatim | 30 to 250 |
| `M` code completion | 40% | HumanEval `HumanEval.jsonl` | one `prompt` field, verbatim | 93 to 457 |
| `L` prose summary | 15% | vLLM `benchmarks/sonnet.txt` | a contiguous line block under a fixed instruction | about 800 |
| `XL` long code review | 10% | HumanEval `HumanEval.jsonl` | `k` problems concatenated under a review instruction | about 3000 |

**Why these sources.** HumanEval keeps continuity with the predecessor and is
already staged and pinned on the share. GSM8K supplies real short natural
language, which HumanEval has none of. `sonnet.txt` is vLLM's own benchmark
corpus, so it arrives already pinned by the vLLM parity pin and needs no
download.

**What the mix cannot claim.** It is not a trace of production traffic. No such
trace is pinned in this repository, and the run may not download one. Three of
the four bands are code or arithmetic, so the mix is narrower in register than a
real chat deployment. It is variadic in length, which is the axis this row
measures, and it is reproducible byte for byte, which a trace would not be.

**The realised histogram is read back, never assumed.** Each engine reports
`usage.prompt_tokens` per request. The report bins those, and the published
histogram is the one the servers counted.

### 3.3 The sweep, and how drift is separated from the rung

Rungs `C = 1, 4, 8`. Two rounds. Within a round each arm serves all three rungs
from one server boot, and the round order is:

```text
round 1:  THEIRS c=1, c=4, c=8   then  OURS c=1, c=4, c=8
round 2:  OURS   c=8, c=4, c=1   then  THEIRS c=8, c=4, c=1
```

Round 2 reverses both the arm order and the rung order. A monotone drift over
the session therefore biases each rung in opposite directions in the two rounds,
so it shows up as a rung-to-rung disagreement between rounds rather than as a
shape in the sweep.

**Two samples per cell is the floor, not the target.** This repository has one
unchanged binary reading 36.82 and 78.86 tok/s in the same session at `c=8`, and
speculative legs spreading 24% while target-only legs held 0.35%. Every
published number therefore carries the two rounds' values and their spread. A
cell whose two rounds disagree by more than 10% is published as a spread and not
as a value.

### 3.4 Warmup

The client sends `W = max(C, 4)` warmup requests at the head of each leg, drawn
from the same queue, and tags them. It reports both views from the same records:

- `warm` — the measured window, warmup requests excluded. This is the headline.
- `all` — every request, warmup included.

vLLM's benchmark does the same discard (`serve.py:875`, `--num-warmups`) and
does not publish the discarded view. Publishing both is this row's addition,
because the predecessor's mean TTFT is dominated by request `i=0` and a reader
cannot see that from a warm-only number alone.

### 3.5 Making the two engines comparable

| Axis | Handling |
|---|---|
| Client | One process, one timing path, both engines. It does not know which engine answers. |
| Chat template | Each engine applies its own, server side, as both published protocols do. |
| Sampling | `temperature 0.6`, `top_p 0.95`, `top_k 20`, `seed 0`, thinking enabled, on both. |
| `max_tokens` | 256 on both. Their generator stops at `max_tokens - 1 - num_draft_tokens`, so expect 248 from them and 256 from us. Each side is divided by the tokens it produced. |
| `ignore_eos` | Not sent. Both published protocols let the model stop. |
| Token counts | From each engine's own `usage`, never from a count of streamed chunks. |
| Concurrency | Offered identically. **Their server holds one lock for the whole generation** (`tools/serve_openai.py:45` `gen_lock`, taken at line 315 around the entire `generator.iterate()` loop, and their own docstring at line 18 says "requests are serialized"). At `C > 1` their numbers therefore measure queueing and ours measure batching. That is a property of the software as shipped and it is published on the face of the table, not in a footnote. |
| Prefix caching | Disabled explicitly on our side, because the comparator's paged cache reuse is not configurable from its server wrapper and an unmatched cache is not a comparison. |
| Streaming granularity | Measured per request on both sides: chunk count, first-chunk characters, total characters, and `usage.completion_tokens`. See §3.6. |

### 3.6 Time to first token, across two chunkings

The predecessor measured 4.46 completion tokens per streamed chunk on our side
and 1.08 on theirs. A TTFT compared across those two chunkings is not one
quantity. Their wrapper also holds back 16 characters
(`tools/serve_openai.py:74` `HOLD_BACK`), which delays their first chunk by
whatever it takes to generate 16 characters.

The client records, per request: the wall time to the first content-carrying
chunk, the character length of that chunk, the total character length, the
chunk count, and `usage.completion_tokens`. The report publishes:

- `ttft_raw` — time to the first content chunk. Directly measured.
- `chars_per_token` — total characters over `completion_tokens`, per request.
- `first_chunk_tokens` — first-chunk characters over `chars_per_token`. **An
  estimate**, and labelled as one everywhere it appears.
- `ttft_first_token` — `ttft_raw - (first_chunk_tokens - 1) * tpot`. The
  estimated time to the first token rather than to the first chunk.

Both TTFT figures are published for both engines. Neither is presented as the
correct one; the raw figure is what a streaming client observes, and the
corrected figure is what the engine did.

### 3.7 Percentiles

`numpy.percentile` with the default linear interpolation, which is what
`vllm/benchmarks/serve.py:739` uses. Reported at 50, 90, 95, 99, plus the
maximum, for `ttft`, `itl`, `tpot` and `e2el`, matching upstream's metric names
and definitions (`serve.py:321` `BenchmarkMetrics`):

```text
ttft = time to the first content-carrying chunk    serve.py:615
tpot = (latency - ttft) / (output_tokens - 1)      serve.py:610
itl  = gap between consecutive streamed chunks     serve.py:614
e2el = client-side request latency                 serve.py:616
```

**`tpot` is the primary inter-token axis and `itl` is secondary**, because
`tpot` is normalised by the token count the server reported and is therefore
immune to the chunking difference, while a raw `itl` is a per-chunk gap. The
`itl` table carries the tokens-per-chunk figure beside it.

**n and what a percentile means at that n.** Each leg sends 128 measured
requests. p99 over 128 samples is the second-worst request. The per-leg tables
therefore stop at p95, and p99 and the maximum are published only from the
pooled 256 requests per (arm, rung) across both rounds. The pooling is stated
wherever a p99 appears.

### 3.8 Resumability

`dgx:gpu0` has crashed roughly hourly under load through this campaign. Every
phase drops a marker under `$STATE`, every leg writes its JSON as it finishes,
and a resubmission skips what is already recorded. A crash costs one leg.

### 3.9 The paged draft route

`42b309508` fixed the eager fault that made `VT_DFLASH_PAGED=0` mandatory for
the predecessor, and closed [#2274](https://github.com/mudler/vllm.cpp/issues/2274).
No published number has been measured on a tree that contains that fix.

This row runs the **shipped default**, with the paged route on. Before the legs,
the job runs a smoke probe of 8 requests at the highest rung. If it faults, the
job sets `VT_DFLASH_PAGED=0` for every leg and records that it did. The
published page states which route every number was measured on.

## 4. Gates

**No correctness gate covers this run, and it cannot.** Both engines sample at
`T = 0.6`, so the two token streams are not expected to match and a token-exact
comparison cannot run on them. This is the predecessor's position and it is
unchanged. The page says so where the numbers are, not in a footnote.

What this run does gate on itself:

| Gate | Condition |
|---|---|
| `G-BYTES` | Both checkpoint shard sha256 values recomputed on the device match the pins. A mismatch aborts. |
| `G-RESOLVED` | Our server's resolved `max_num_seqs` is at least the rung's concurrency. A leg that ran with a lower resolved value is discarded, not published. |
| `G-COMPLETE` | Every leg completed every request. A leg with a failure publishes the failure count on its row. |
| `G-SPREAD` | Every published cell carries both rounds' values and their spread. |
| `G-USAGE` | Token counts came from `usage` on both sides, never from a chunk count. The client records which and the report refuses to publish a leg counted by chunks. |

## 5. Risks, and what is done about each

| Risk | What is done |
|---|---|
| A single high-concurrency sample is noise | Two rounds, reversed rung order, spread published beside every value |
| The comparator serializes, so a `C > 1` ratio reads as an engine result | The serialization is stated on the face of every table that carries a `C > 1` number, with the source line |
| The chunking difference silently moves TTFT | Chunk statistics measured on both sides; both raw and corrected TTFT published |
| Cold start hidden by a discard rule | Both views published from the same records |
| The KV budget auto-fits `max_num_seqs` below the rung | `--num-blocks` and `--max-model-len` set explicitly; `G-RESOLVED` reads the server's own resolved value back |
| The box crashes mid-run | Per-leg checkpointing and a resumable job |
| The paged draft route faults | A smoke probe decides, and the decision is recorded and published |
| The corpus is not a production trace | Stated in §3.2 and on the published page |

## 6. Evidence

`docs/bench-evidence/qwen38-27b-exl3-variadic-<date>/` carries the job as
submitted, the harness as run, the corpus manifest with every source sha256, the
per-leg summaries, and `results.txt`. Per-request records stay on the share.

## 7. Stop conditions

- The comparator fails to build or serve: record it verbatim and publish that.
- Our server refuses a rung: record the refusal and publish the rung as refused.
- Two rounds of a cell disagree by more than 25%: publish the spread and open a
  variance row rather than quoting a value.
- The lease cannot be obtained: report the fleet state; never `ssh` to the box.

## Now

`DONE`, with three legs owed. The run completed nine of twelve legs on
5-6 September 2026 before `dgx:gpu0` was lost 2h33m in; the resume is `rc` job
`493d6c72-1ff0-4362-bde4-1a1887edaf9a`, queued. Results are published on
[`qwen38-27b-exl3-variadic-gb10`](../../docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md).

## Outcome

**What was measured.** `rc` job `7b5084ab-f214-4d8d-b1fe-1eca86efb1e8`, tree
`3351ec54f`, binary md5 `23be01c338457038fe8354b01d92c7aa`. `G-BYTES` and
`G-RESOLVED` both passed; every leg reported `publishable = yes` under `G-USAGE`;
every cell of ours carries two rounds agreeing to 0.4% or better.

**The paged draft route ran at its shipped default**, which the smoke probe
cleared on the first attempt at the top rung. Every previously published number
on this pair ran `VT_DFLASH_PAGED=0`, so this is the first measurement of the
default configuration and no number here is comparable to one on the predecessor
page.

**Three results.** Their aggregate throughput does not move with concurrency
(33.10, 33.33, 33.52 output tok/s at c = 1, 4, 8) and ours does (35.81, 53.17,
53.93), which is their `gen_lock` measured rather than argued. We prefill long
prompts about half as fast as they do, and the gap widens with prompt length —
298 against 585 tok/s implied on the `XL` band — which the predecessor's
93-to-457-token workload could not have produced. And both engines now report
acceptance, 0.796-0.799 against 0.774-0.778 per output token.

**What was rejected.** An open-loop arrival process, because a fixed rate above a
serialized server's service rate makes the measured latency a function of how
long the leg ran. A shared-prefix long band, because that measures a prefix cache
rather than a length distribution. A `p99` computed from a single leg, because
128 requests make it the second-worst request.

**One design assumption was falsified by the run and is worth recording.** The
predecessor inferred that our first streamed chunk carries a whole accepted
speculative block, and this row built a correction for it. Measured, our first
content chunk is exactly two characters on all 768 requests, so the estimated
first-chunk token count is 0.58 and the correction is nil on both engines. The
correction stays in the harness because the property is real on other
configurations; it did nothing here, and the page says so rather than quoting a
corrected column that equals the raw one without explanation.

**Why `--num-blocks` is pinned rather than left to auto-fit.** The loader clamps
`max_num_seqs` against the pool on this hybrid, and the draft speculative context
adds 160.312 MiB per concurrent request that `gpu_memory_utilization` does not
bound ([#2993](https://github.com/mudler/vllm.cpp/issues/2993)). An auto-fitted
pool would have made the concurrency ladder measure its own configuration.

## Owed

- **[#2993](https://github.com/mudler/vllm.cpp/issues/2993): `gpu_memory_utilization`
  does not account for the DFlash2 draft speculative context.** Found by this
  run, which recorded 1282.5 MiB of it at `max_num_seqs 8`
  (`src/vllm/v1/worker/gpu/runner.cpp:4090-4098` prints the number and says the
  flag does not bound it). The term scales with concurrency and with
  `max_model_len`, so an operator raising either exceeds the fraction they set.
  This row does not fix it; it pinned `--num-blocks` explicitly so the ladder
  would not measure it as noise. The issue carries a `-` row and is owned here.
- **`THEIRS` round 2 at every rung**, lost with the box 2h33m into the run.
  Their column is one sample per rung. Resume: `rc` job
  `493d6c72-1ff0-4362-bde4-1a1887edaf9a`.
- A third round, if any cell's two rounds disagree by more than 10%. None does:
  the widest is 0.4%.
- A matched-configuration leg. Each engine still runs its own published recipe,
  and this engine still refuses an NVFP4 KV cache by name
  ([#2620](https://github.com/mudler/vllm.cpp/issues/2620)).
- A second boot for each rung. Rounds 1 and 2 restart both servers, so this row
  has two boots per cell, which the predecessor did not.
