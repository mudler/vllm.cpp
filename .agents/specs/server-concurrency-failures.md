# Dropped requests under concurrency, and the metric that hid them

**Rows:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`
**Issue:** [#931](https://github.com/mudler/vllm.cpp/issues/931)
**Lifecycle:** `READY`
**Owner:** unassigned

## Scope

Two defects, in the order they must be fixed:

1. **The measurement defect** — no harness asserts `failed == 0`, so a leg with
   dead requests still emits a throughput number, and that number is wrong in an
   unpredictable direction. Fix this first: it is cheap, it is independent of the
   root cause, and until it lands every throughput figure we take is unsound.
2. **The server defect** — our OpenAI server fails requests under concurrency
   where vLLM does not, on the identical workload from the identical client.

Out of scope: any performance work, the vision path, and re-running the #915
benchmark grid (that follows once this closes).

## The evidence

`vllm bench serve` against `Qwen/Qwen3.8-27B` bf16 on GB10, interleaved under one
GPU lock, `vllm-server` md5 `bda95d34a7e2587c6e2195e365f77bc0` from `11a42dc4c`:

| leg | requested | completed | failed |
|---|---:|---:|---:|
| ours c1 | 6 | 5 | **1** |
| ours c4 | 24 | 24 | 0 |
| ours c8 | 48 | 36 | **12** |
| vLLM c1/c4/c8 | — | all | **0** across nine legs |

Reproducible: **1/6 at c1 in all three reps**, and 12/11/12 of 48 at c8.

**Why it corrupts the axis:** `output_throughput` divides tokens by the leg's
wall duration, and that duration still contains the time the dead request spent
before failing. Our c1 leg reads 2.36 tok/s against vLLM's 3.50 (**0.675x**)
while median TPOT in the same file is **220.3 ms against 224.1 ms** — parity or
better per output token. Read without the completion counts, that leg claims we
are 33% slower where the per-token evidence says we are 1.7% faster.

## What the shape of the failure already rules out

**Not HTTP worker-pool starvation.** That race is real and documented at
`api_server.cpp:60-65` — cpp-httplib's default pool grows only when
`idle_thread_count_` is exactly zero at enqueue, so a burst can queue accepted
sockets while long-lived SSE jobs hold every worker. But the mitigation is
already the **production default**: `HttpWorkerPoolMode::kCapacityFixed` sizes a
fixed pool at `max_num_seqs + kControlWorkerHeadroom (4)`
(`server_main.cpp:1196-1201`). More decisively, **a pool of that size cannot
starve at concurrency 1**, and c1 fails deterministically.

**A deterministic 1-of-6 at c1 is not a race at all.** It points at a specific
request — a particular prompt, length, or sampling parameter — rather than
contention. The c8 rate (~25%) may or may not share that cause; do not assume one
explanation covers both.

## Design

**Reproduce before fixing.** `failed` in `vllm bench serve` is a bare boolean with
no reason attached, and our server logged nothing — the captured log was 27
startup lines, taken at readiness rather than after the leg, so the post-leg state
was lost. A fix written against a guess would be indistinguishable from a fix that
works by accident.

The reproduction must record, per failed request: **HTTP status, exception class,
error body, request index, and the exact request payload**. Run it at c1 first —
the deterministic case is the cheap one — and only then at c8.

Candidate causes to separate, not to assume:

- request rejection under concurrency;
- a connection or keep-alive difference the client tolerates differently between
  the two servers;
- an interaction with `--max-num-batched-tokens 8192`, given c8 × 1024-token
  prompts lands exactly on that budget;
- something specific to one prompt in the six.

## Risks

- **Fixing the symptom.** Making the harness assert `failed == 0` turns a silent
  wrong number into a loud failure — necessary, but it does not fix the server,
  and the row must not close on it alone.
- **A fix that hides the failure** rather than removing it: a retry, a longer
  timeout or a swallowed error would make the counter green while the defect
  survives. Any change must be justified by the recorded cause.
- **Assuming one cause.** c1 and c8 may be two defects. Report them separately.

## Tests

1. **RED-first: a test that reproduces a dropped request** and asserts on the
   recorded status/exception, not merely on a count.
2. **Harness: `failed == 0` asserted** wherever a throughput number is derived —
   red before, green after, with a synthetic failed request proving the assert
   fires.
3. Inertness: existing serving suites unchanged.

## Gates

- Focused suites plus the full serial gate.
- **The reproduction must go from RED to green on real hardware**, at c1 and at
  c8, with completion counts recorded.
- A re-run of the #915 c1/c8 legs, whose throughput numbers are currently
  withheld, is the acceptance evidence that the axis is sound again.

## Stop conditions

- If the cause cannot be established from the recorded evidence, stop and report
  rather than fixing a candidate. A guessed fix here is worse than none, because
  it would close the issue that is currently protecting the numbers.
- If the only available fix is a retry or a timeout increase, stop and say so —
  that is a decision about what we are willing to claim, not an implementation
  detail.

## Now

Both phases implemented on `row/FIX-SERVER-CONCURRENCY-931`. The measurement
guard landed first and stands on its own; the server fix follows the recorded
cause below.

## Outcome

### The cause, and it is ONE cause for c1 and for c8

**Our server emits an SSE comment frame that vLLM never emits, by default, and
vLLM's own benchmark client cannot survive one.**

`serving_utils.h:42` defines `kSsePingFrame = ":\n\n"`. `AssignSseWaitResult`
(`serving_utils.cpp:242-251`) emits it whenever the request's output collector
produces nothing for `VT_SERVER_SSE_PING_S` seconds — **default 15**
(`serving_utils.cpp:253-264`), reached from both `CompletionSseStream::WaitOutput`
(`serving_completion.cpp:39-49`) and `ChatSseStream::WaitOutput`
(`serving_chat.cpp:332-342`). It is the only non-`data:` frame either stream can
produce, and it was added by [#316](https://github.com/mudler/vllm.cpp/issues/316)
for proxies with inactivity timeouts.

The pinned oracle emits nothing of the kind: grepping `0.23.1rc1.dev1511+g555967922`
for a yielded comment across all of `vllm/entrypoints/` returns no hit. So this
is not a mirrored behaviour, it is an invention — and `vllm bench serve` cannot
parse it. The client strips every network chunk before parsing
(`benchmarks/lib/endpoint_request_func.py:207`), which destroys the `\n\n`
separator at chunk boundaries, and `StreamedResponseHandler`'s only
resynchronisation path requires a `data: ` prefix (`:48`). A comment frame
therefore lands in the buffer as a bare `:` that no later frame can clear:

- **before the first data frame** → `first_chunk_received` stays false, the
  client reports `Never received a valid chunk to calculate TTFT` and marks the
  request **failed** (`:249-257`) — with an HTTP **200**, a normal server-side
  completion, and nothing logged;
- **mid-decode** → the request still "succeeds" but every later frame is
  swallowed, so tokens and the usage frame are silently lost.

### Proven end to end, over a socket, with the real client

A socket server replaying our exact frame bytes, driven by the pinned
`vllm bench serve` at `--max-concurrency 1 --num-prompts 6`, one variable:

| arm | completed | failed | error |
|---|---:|---:|---|
| no comment frame | 6 | **0** | — |
| one comment frame before request 1's first data frame | 5 | **1** | `Never received a valid chunk to calculate TTFT` |

`output_lens[1]` is `0` and `ttfts[1]` is `0.0` in the failing arm, and the
`total_input_tokens` of 5025 for five completions is the same shape the #915 c1
legs show. Nothing but the comment frame differs between the two arms.

### Why exactly those requests, in exactly those legs

`vllm bench serve` records no TTFT for a failed request, but at a fixed
concurrency the leg's wall duration is a budget: slot-seconds not spent on a
successful request were spent on a failed one. Subtracting the decode time the
same file measures (`median_tpot_ms x 127`) leaves the failed request's TTFT.
From the committed #915 records:

| leg | completed | failed | imputed TTFT per failed request | p99 TTFT among successes |
|---|---:|---:|---:|---:|
| c1 r1/r2/r3 | 5 of 6 | 1 | **93.9 / 92.2 / 92.2 s** | 4.24 s |
| c4 r1/r2/r3 | 24 of 24 | 0 | — | 5.62 s |
| c8 r1/r2/r3 | 36/37/36 of 48 | 12/11/12 | **47.0 / 47.5 / 46.7 s** | 7.23 s |

The 15 s threshold separates the two populations exactly, in every leg, in all
three repeats. At c1 the arithmetic is exact — the semaphore serialises the
requests — and it reproduces to within 1.7 s across reps, which is why the c1
failure is deterministic rather than a race: the runner starts one server per
rep and runs c1 first, so the first timed request is the first inference that
process ever does and pays the whole warm-up. By c4 the server is warm and
nothing crosses 15 s. At c8, queueing behind 8-way concurrency puts a dozen
requests back over the line. The c8 figure assumes perfectly packed slots and so
overstates the span, but not by the factor of three that would be needed to drop
it under 15 s.

**This is a recurrence, not a discovery.** [#577](https://github.com/mudler/vllm.cpp/issues/577)
already recorded the same mechanism — the keepalive made 27B c16 VOID at 93/96
and *"the three missing are the SLOWEST"* — and its recorded remedy was to
disable the keepalive **in the recipe**. `run_bench_main.sh` for #915 did not,
so it came back. A recipe is not a fix.

### What was refuted

**HTTP worker-pool starvation, at every concurrency.** The #915 server log
records `HTTP worker pool 36 fixed` (`--max-num-seqs 32` + `kControlWorkerHeadroom`),
against an offered concurrency of 8. A pool of 36 cannot starve at 8, and
certainly not at 1. The related mirror gap — `--max-num-seqs` doubling as our
HTTP concurrency ceiling where upstream bounds only the scheduler batch — is
real, is not this bug, and is filed as
[#952](https://github.com/mudler/vllm.cpp/issues/952).

**A per-prompt cause.** The failing request is not a particular prompt: at c1 it
is whichever request runs first, and the c1 prompt set is a prefix of the c4 set
that failed nothing.

### The fix

`SsePingIntervalSec()` defaults to **0**. Both streams then take the blocking
`get_output()` path, which #316 itself records as byte-identical to the
behaviour before it, so our default wire format is `data: ` frames only — the
frame set vLLM has. The capability stays reachable at `VT_SERVER_SSE_PING_S=<n>`
for a deployment behind a proxy that needs it, with the client-compatibility
cost now stated in `docs/USAGE.md` and `docs/ENVIRONMENT.md`.

This removes the cause rather than the symptom. It is not a retry, not a longer
timeout and not a swallowed error, and it is not flattering: the requests that
used to vanish now complete and their 47-to-94-second TTFTs enter the latency
distribution, where they belong.

### The measurement guard, which is what stops the next one

`serve_low_common.require_complete_request_set` refuses to derive any rate from
a record that cannot prove its request set is whole, and is called at the
derivation sites rather than only at whichever validator runs first. Had it
existed, #915's c1 leg would have aborted instead of publishing 0.675x. It also
converts any future re-enabling of the keepalive from a silent 25% loss into a
loud failure, which is the protection #577 asked for and did not get.

## Owed

- The #915 c1 and c8 output-throughput cells stay WITHHELD until re-run on a
  binary carrying this fix; that re-run is the acceptance evidence and belongs
  to [#915](https://github.com/mudler/vllm.cpp/issues/915).
- A ~47 s TTFT for a quarter of the c8 requests is a real scheduling tail that
  the keepalive was hiding. It is not a failure and is not this row's scope, but
  it is now visible and unexplained.
- [#952](https://github.com/mudler/vllm.cpp/issues/952) — `--max-num-seqs` also
  bounds our HTTP concurrency, where upstream bounds only the scheduler batch.
  Found here while refuting worker-pool starvation as this bug's cause, and
  refuted as that cause: the pool was 36 against an offered concurrency of 8. It
  is a real mirror gap and needs its own change, because the next person raising
  `--max-num-seqs` for throughput silently raises the HTTP ceiling with it.
