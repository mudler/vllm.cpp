# Head-to-head on Qwen3.8-27B EXL3: their engine and ours behind one client

| Field | Value |
|---|---|
| Issue | [#2495](https://github.com/mudler/vllm.cpp/issues/2495) |
| Owning row | `BENCH-QWEN38-EXL3-HEADTOHEAD` |
| Published page | [`docs/benchmarks/qwen38-27b-exl3-gb10.md`](../../docs/benchmarks/qwen38-27b-exl3-gb10.md) |
| Evidence | [`docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/`](../../docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/README.md) |
| Subject | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac8`, draft `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f043626` |
| Comparator | `MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`, a pin chosen here because their card publishes none |
| Host | `dgx:gpu0`, GB10 `sm_121a`, `rc` lease `d32255f7-2004-432a-b656-dcaef50037a9` |
| Role | branch `row/BENCH-QWEN38-EXL3-HEADTOHEAD` |
| Status | `DONE`. The run finished on 4 September 2026 and its result is published. |

**This spec was written after the harness landed and after the run.** The row
carried a committed harness and a committed evidence README, and no spec. That
is the debt this file pays, and it is recorded here rather than presented as
spec-first work. Nothing in the design below was decided after reading the
numbers; every item of it is readable in `job-as-run.sh` at commit `f0c24345d`,
which predates the run.

## 1. Scope

| In scope | Out of scope |
|---|---|
| One throughput comparison between `MiaAI-Lab/exllamav3` and this engine, on one box, behind one client | Any comparison against a published figure rather than against a running engine |
| Both counting conventions, computed from one set of timings | A single headline number |
| The EXL3 checkpoint pair both engines can serve | The NVFP4 checkpoint and the SGLang comparator, which are [#2761](https://github.com/mudler/vllm.cpp/issues/2761) |
| Each engine at its own published recipe | A matched context length and KV dtype, which this engine cannot run yet |
| Acceptance from whatever each engine exports | An acceptance ratio, which needs [#2770](https://github.com/mudler/vllm.cpp/issues/2770) |

## 2. The question

The published page led with 1.25x on this pair. That ratio divided our 59.5
tok/s, counted decode only, by the 47.5 "decode tok/s" the Mia-AiLab card
quotes. The card defines the term nowhere. Counted over whole-run wall time the
same run of ours reads 45.1, which against the same 47.5 is 0.95x. Two
conventions, opposite verdicts, and no way to choose between them from published
material, because their card also pins no engine revision.

Reading their prose harder cannot settle it. Running both engines behind one
client can, because then one definition applies to both sides and the convention
stops being a free variable.

## 3. Design

One client, two engines, four legs in the order `THEIRS-A`, `OURS-A`,
`THEIRS-B`, `OURS-B`. Each server starts and stops around its own leg. The
client speaks OpenAI `/v1/chat/completions` with `stream: true` and does not
know which engine answers.

Every default and why it has its value:

| Default | Value | Why |
|---|---|---|
| Prompt set | the full 164-problem HumanEval set at T = 0.6 | The card's own task class and temperature. A subset drafts hotter than the full set, which this page has already measured. |
| `max_tokens` | 128 | The output length the earlier published run used, so the two workloads differ in shape and not in length. |
| `top_p`, `top_k` | 0.95, 20 | Their server's defaults. Ours were set to match them rather than the reverse. |
| `seed` | 0 | Fixed on both sides. It does not make the two engines agree; it makes each leg repeatable against itself. |
| Concurrency | 1 | Their server serializes generation behind one lock by construction, so a higher concurrency would queue on their side and batch on ours. |
| `ignore_eos` | not sent | Both published protocols let the model stop. Our bench sends it, which is one reason the bench numbers are not comparable to these. |
| `num_speculative_tokens` | 7 | Their DFlash2 default, which is `block_size - 1`. Ours is set to the same budget. |
| `VT_DFLASH_PAGED` | 0 | The shipped paged draft route faulted eagerly on the tree this job built. `42b309508` fixed that later and closed [#2274](https://github.com/mudler/vllm.cpp/issues/2274); the measured tree `5649e07d` does not contain it. |
| Recipes | each engine at its own | A matched configuration is not available: this engine refuses an NVFP4 KV cache by name ([#2620](https://github.com/mudler/vllm.cpp/issues/2620)). Making their arm run our configuration would degrade the comparator to produce a matched number, which is not a comparison worth publishing. |
| Interleaving | A, B per arm, alternating | A sequential A/B measures the hour along with the arm. |
| Execution surface | `/tmp`, never `/workspace` | `/workspace` is CIFS. A network filesystem in the load path is a confound. |

The one adaptation to their code is `serve_openai-usage.patch`, confined to
`tools/serve_openai.py`. Their streaming path already computes the token counts
and the `Job`'s `accepted_draft_tokens` and then drops them. The patch forwards
them as the standard OpenAI `include_usage` terminal chunk. No file under
`exllamav3/` changes, so no engine, kernel or sampling behaviour changes.

## 4. Gates

| Gate | Result |
|---|---|
| Both engines serve the same bytes | `PASS`. Both target shard sha256 values and the draft sha256 were recomputed on the device and matched the pins. |
| Every leg completes every request | `PASS`. 164 of 164 on all four legs, `failures: []`. |
| Both conventions computed from one set of timings | `PASS`. Decode only and whole run are both in every leg record. |
| Token counts come from each engine's own `usage` | `PASS`. `completion_tokens_counted_by: usage` in all four records. |
| A token-exact or correctness gate over the compared runs | `NOT AVAILABLE`. Both engines sample at T = 0.6. See the null results below. |
| An acceptance ratio | `PENDING` on [#2770](https://github.com/mudler/vllm.cpp/issues/2770). |

## 5. Risks, and what was done about each

| Risk | What was done |
|---|---|
| Counting SSE chunks as tokens | Rejected before the run. Their wrapper buffers with `HOLD_BACK = 16` and emits arbitrary string slices, so a chunk is roughly a fifth of a token count. Token counts come from `usage` on both sides. |
| Measuring their time to first token to the end of the reasoning block | Rejected before the run. Their server routes `<think>` into `reasoning_content`; the client counts that field and `content` alike. Against a mock shaped like their stream this reads 229 ms rather than about 830 ms on a 200 ms prefill. |
| Their engine failing to build on aarch64 | The job reports a build or wheel failure verbatim rather than as a slow result. It did not fire; see the null results. |
| Drift inside the session | Two legs per arm, interleaved. |
| A crash mid-sequence | The job is resumable: a marker per phase, a JSON per leg. |

## 6. Evidence

`results.txt` and the four `*.clientlog` files are committed under
`docs/bench-evidence/qwen38-27b-exl3-headtohead-20260903/`. The per-request JSON
records and the server logs stay at `/mnt/nas_share/rc/exl3-headtohead/out/`.
`job-as-run.sh`, `client.py` and `serve_openai-usage.patch` in that directory are
byte-identical to the files that ran on the box, checked by `diff` against the
share on 5 September 2026.

## 7. Stop conditions

A build failure on either engine, a leg that does not complete all 164 requests,
or a checkpoint sha256 that does not match its pin. Each one is a publishable
answer and none of them is a slow result.

## Now

`DONE`. The page leads with the measured comparison, and the README-derived
ratio pair is marked superseded on the page rather than removed.

## Outcome

**What was measured.** One client, two engines, one GB10 board, four interleaved
legs of 164 HumanEval problems at T = 0.6, on 4 September 2026.

| leg | decode-only tok/s | whole-run tok/s | mean TTFT ms | output tokens | wall s |
|---|---|---|---|---|---|
| `THEIRS-A` | 44.63 | 32.54 | 1021.7 | 19,680 | 604.9 |
| `OURS-A` | 53.68 | 37.33 | 1062.8 | 20,992 | 562.3 |
| `THEIRS-B` | 45.01 | 33.99 | 886.6 | 19,680 | 579.1 |
| `OURS-B` | 53.57 | 37.36 | 1055.1 | 20,992 | 561.8 |

Means per arm: decode only ours 53.63 against theirs 44.82, a ratio of 1.197x;
whole run ours 37.35 against theirs 33.26, a ratio of 1.123x. Paired at their
least favourable ends, our slower leg against their faster one, 1.190x and
1.098x. Our legs agree to 0.21% decode only and 0.08% whole run; theirs to 0.85%
and 4.46%. The pooled rate gives the same verdict as the per-request mean,
1.1960x against 1.1966x, so the choice between them does not move the result.

**The convention question is retired, not answered.** Nothing on the page now
divides our number by a number read off their card. Both conventions were
computed from the same timings on both engines and both put us ahead.

**Null results, each of which is a result.**

1. **No correctness gate covers these legs.** Both engines sampled at T = 0.6
   with `top_p` 0.95 and `top_k` 20, so their token streams are not expected to
   match and a token-exact comparison cannot be run on them. Nothing scored the
   code either engine wrote. The page says so beside the numbers rather than
   leaving it out.
2. **Our acceptance rate is absent, not zero.** `spec_drafts_proposed()` and
   `spec_drafts_accepted()` are `GpuRunner` accessors that no HTTP route reads.
   Theirs reports 15,238 accepted draft tokens over 19,680 output tokens in leg
   A and 15,253 over 19,680 in leg B, which is 0.774 and 0.775 per output token.
   No ratio is available until [#2770](https://github.com/mudler/vllm.cpp/issues/2770)
   lands.
3. **Time to first token goes to them.** Mean TTFT 954.2 ms theirs against
   1059.0 ms ours, median 762.7 ms against 863.3 ms, and both of their legs beat
   both of ours on both statistics. The size is not established: their own two
   legs differ by 15.2% on mean TTFT, against 0.73% between ours.
4. **No text-quality comparison is available.** Their streamed text drops spaces
   at chunk boundaries, which is their wrapper's `HOLD_BACK` buffering, so the
   assembled strings cannot be compared as text. Both first completions do open
   on the same reasoning about the same problem, which is a coherence check and
   not a score.
5. **The expected aarch64 blocker did not fire.** The oracle file records that
   stock `turboderp-org/exllamav3` fails 7 of 129 translation units on aarch64.
   Their fork carries a real port and built here: `THEIRS engine build+install
   rc=0`. That reading was made from source before the run and the run confirmed
   it.
6. **Their package has no version attribute.** `import exllamav3;
   exllamav3.__version__` raises `AttributeError`, so the engine identity in
   this record is the tarball sha256 and the fork revision, never a version
   string.

**One finding about their engine, read from their source rather than inferred.**
Every one of their 164 requests in both legs returned exactly 120 completion
tokens against a requested 128, where ours returned exactly 128. At their pin,
`exllamav3/generator/job.py:202` sets `self.max_new_tokens = max_new_tokens - 1
or 1` and line 742 stops the job at `self.new_tokens >= self.max_new_tokens -
self.generator.num_draft_tokens`, so 128 - 1 - 7 = 120. Each side is divided by
the tokens it produced, and their shorter sequences carry slightly less
attention work per token, so the asymmetry does not run in our favour.

**One cross-check that raises confidence in their arm.** If a pass emits one
target token plus its accepted drafts, their counters give
19,680 / (19,680 - 15,238) = 4.430 tokens per pass in leg A and 4.445 in leg B.
Their card claims 4.43. Their engine ran at the acceptance they publish, so this
is not a comparison against a misconfigured competitor. The conversion assumes
that accounting; only the counters are read from a file.

**What was rejected.** Chunk counting, reading only `content`, and any further
attempt to settle the convention question from their published material. Also
rejected: forcing their arm onto our configuration to obtain a matched number,
because degrading the comparator to make the axes line up produces a number
nobody should quote.

## Owed

- [#2770](https://github.com/mudler/vllm.cpp/issues/2770): our OpenAI server
  exports no speculative-decoding acceptance metric, so this row could measure
  one side of the mechanism only.
- A matched-configuration leg at one context length and one KV dtype, which
  needs [#2620](https://github.com/mudler/vllm.cpp/issues/2620).
- A rerun on a tree that carries `42b309508`, so our arm runs the shipped paged
  draft route rather than `VT_DFLASH_PAGED=0`.
- A repeat on a second boot. Four legs in one lease control drift inside that
  window and say nothing about boot-to-boot movement.
