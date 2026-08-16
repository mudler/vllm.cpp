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
produces nothing for `VT_SERVER_SSE_PING_S` seconds — **which defaulted to 15**
(`serving_utils.cpp:253-264`; this row changes that default to 0), reached from
both `CompletionSseStream::WaitOutput`
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
  swallowed, so tokens and the usage frame would be silently lost. This mode is
  read from the client's source; it is **not what happened here**, and the
  measured negative is recorded below.

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

### The per-request evidence, measured

A fresh reproduction on the **byte-identical #915 binary** (`vllm-server` md5
`bda95d34a7e2587c6e2195e365f77bc0`), keepalive at the old default, captured what
#915 never recorded: the client's own per-request error strings. **All twelve
failures across both legs carry ONE distinct string**, byte-identical to the
single-comment-frame signature the socket probe produced:

```text
Never received a valid chunk to calculate TTFT.This response will be marked as failed!
```

| leg | completed | failed | failing indices | evidence |
|---|---:|---:|---|---|
| c1 | 5 of 6 | 1 | 0 | **MEASURED**, causation closed by the A/B below |
| c8 | 37 of 48 | 11 | 2, 12, 18, 26, 27, 32, 33, 38, 39, 44, 45 | **MEASURED**, causation closed by the A/B below |

At c1 the failure is index **0**, the first timed request: `ttfts[0] = 0.0`,
`output_lens[0] = 0`, `generated_texts[0]` empty, and a `start_times`-derived
TTFT of ≈91 s against 1.08–4.29 s for every other request. (The no-ping arm then
measured that same request directly at **91.613 s**, so the derivation was 2.5%
high.) The server logged
nothing: `diff server-at-ready.log server-after-c1.log` is empty, 27 lines
before and after, and 27 again after c8 — reproducing #915's silent-server
observation deliberately rather than inferring it.

c8 reproduces #915 quantitatively: 37/11 here against 36/37/36 and 12/11/12
there, output throughput 16.42 against 15.93–16.0.

**No mid-decode truncation occurred.** No success in either leg had
`output_len != 128`. The swallow-everything-after mode is real in the client's
parser, and it did not happen here — every request either completed whole or
failed before its first data frame. This closes an open question the earlier
draft of this Outcome asserted rather than checked.

**Where each number comes from.** The committed #915 record
(`.agents/benchmark-record.md`, Q38-27B-BF16) carries completion counts,
aggregate throughput and median TPOT/ITL/TTFT **ratios** only. It does not carry
p99 TTFT, and it does not carry any per-request array — no `errors`, `ttfts`,
`output_lens` or `start_times`. Every per-request figure above therefore comes
from the fresh reproduction named here, not from the tree and not from the #915
summary.

### The A/B, which is what closes causation

One variable, same binary, same workload, same box, clocks flat at 2184 MHz.
Evidence at `dgx:~/fix931/out_noping_run1/`:

| leg | keepalive ON (old default) | `VT_SERVER_SSE_PING_S=0` |
|---|---|---|
| c1 | 5 of 6, failed 1 (index 0) | **6 of 6, failed 0** |
| c8 | 37 of 48, failed 11 | **48 of 48, failed 0** |

Server log 27 lines in all four legs, `diff` empty throughout. **The keepalive is
causal at BOTH concurrencies.** c8 is not a second defect, and every request that
used to vanish completes once the frame is gone.

### The 15 s relationship, corrected — both earlier positions were wrong

This row has now stated the TTFT relationship twice and got it wrong in opposite
directions. Both are recorded, because the shape of the error is the lesson.

The committed claim **OVERCLAIMED**: an earlier draft of this Outcome, and the
message of commit `9e23bd074` on this branch, imputed a per-request TTFT from
the leg's wall duration and concluded that *"the 15 s threshold separates the two
populations exactly, in every leg, in all three repeats."* An intermediate
retraction then **UNDERCLAIMED**, saying the threshold "neither predicts failure
nor follows from it". (The commit message cannot be edited and does not need to
be: this branch lands by squash, and the squash message is the pull request
body.)

Measured, from the no-ping arm's real TTFTs cross-mapped onto the default arm's
failures:

- **failure ⇒ TTFT > 15 s: 11 of 11.** No failed request was fast. The
  implication the mechanism requires holds without exception.
- **TTFT > 15 s ⇒ failure: 11 of 12.** Index **1 SURVIVED** at 38.58 s in the
  no-ping arm (43.88 s in the default arm). That survivor is **unexplained** and
  is recorded under `## Owed` rather than explained away.
- Failing requests' real TTFTs at c8: **33.8–40.8 s**.

Three earlier per-request figures for c8 are withdrawn outright: the ≈3.7 / 5.9 /
6.4 s spans once derived for indices 2 / 12 / 18 were artefacts of a crude `+8`
slot mapping, flagged as approximate when it was made. Measured, those three are
**40.8 / 37.6 / 38.6 s** — all well above the interval, not below it.

**Imputation against measurement, kept visible because it calibrates the
method:**

| leg | imputed from wall duration | measured | error, `(imputed - measured) / measured` |
|---|---|---|---|
| c1 | 93.9 s | **91.613 s** | **+2.50%**, from `(93.9 - 91.613) / 91.613` |
| c8 | 47.0 / 47.5 / 46.7 s | **33.8 to 40.8 s** | **+14.5% to +40.5%**, from `(46.7 - 40.8) / 40.8` at the low end and `(47.5 - 33.8) / 33.8` at the high end |

The denominator is the MEASURED TTFT in both rows, which is the convention the
exact c1 row already uses. The two rows are not the same shape, and the band
depends on that. At c1 one imputed number faces one measurement. At c8 the
imputation produced one number PER REP, three of them, and the measurement is a
RANGE over the eleven failed requests, so the band is the two extremes of that
pairing: smallest imputed against largest measured, and largest imputed against
smallest measured. It is not a per-request error. All six pairings fall inside
it, and none reaches +25%: +15.2, +39.1, +16.4, +40.5, +14.5 and +38.2 percent.

**An earlier `+25% to +40%` is withdrawn.** Its high end was right. Its low end
follows from no pairing of the columns printed beside it, and this table exists
so the next person can price the imputation technique before relying on it. A
band that does not follow from its own adjacent columns cannot serve that
purpose. The correction is carried in `.agents/benchmark-record.md`, which is
append-only, as an appended entry that cites the superseded lines.

At c1 the semaphore serialises and the arithmetic was nearly exact. At c8 the
imputation assumed perfectly packed slots and ran 14.5% to 40.5% high, exactly
the way it was flagged it might. The imputed row is not deleted. It is left beside
the measurement so the next person can price the technique before relying on it.

**Why the threshold was never the mechanism, even though it correlates.**
`SsePingIntervalSec()` bounds a wait on **that request's collector**, and the
stream loops on empty-but-unfinished outputs (`serving_completion.cpp:73-89`,
`serving_chat.cpp:351`, `:402`), so each `WaitOutput` restarts its own timeout: a
long prefill that keeps yielding intermediate collector entries never pings,
however large its TTFT. The condition is 15 s of silence on one request's
collector. A long TTFT makes that silence likely, which is why the correlation is
11 of 12 rather than 12 of 12 — and index 1 is the case that shows the two are
not the same condition.

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
used to vanish were completing server-side all along, so they now reach the
client and their latencies — 91.613 s for the c1 failure, 33.8–40.8 s for the
eleven at c8 — enter the distribution instead of being deleted from it.

**Causation is closed at both concurrencies.** The A/B recorded above takes c1
from 5 of 6 to 6 of 6 and c8 from 37 of 48 to 48 of 48 on one variable. c8 is not
a second defect.

**What the fix does to the numbers, INDICATIVE ONLY:**

| leg | default | `VT_SERVER_SSE_PING_S=0` | ratio then → now |
|---|---:|---:|---|
| c1 output tok/s | 2.3954 | 2.8553 | 0.675x → **0.816x** |
| c8 output tok/s | 16.4185 | 21.0875 | 0.572x → **0.757x** |

These are **single unpaired legs, and the vLLM arm was NOT re-run**: the
denominators are #915's. This is not the paired, interleaved, 3-rep protocol, so
it is not #915's re-run and must never be quoted as one. #915's c1 and c8 cells
stay WITHHELD.

Both cells also remain **real gaps**. 0.816x and 0.757x are not parity. The fix
makes the numbers honest, not good — it stops us deleting our own slowest
requests from a measurement taken with vLLM's client, and what is left is the
deficit that was there underneath.

### The c1 warm-up, and the cold-start cell it contradicts

The c1 story needs ~91 s of first-inference work **after** our server answered
`/health`, while the same #915 record publishes cold start to first `/health` as
53 s against vLLM's 780 s — 14.7x. Those two statements cannot both be innocent.

**It is not the keepalive.** `ttfts[0] = 91.613 s` appears in **both** arms of
the A/B, so the first request's cost is a genuine first-inference cost and not
an artefact of pinging. The reconciliation is that the two servers answer
`/health` at different points in their startup:

- **Ours answers on process liveness only.** `ApiServer::handle_health`
  (`api_server.cpp:286-294`) returns a bare 200 unconditionally; its own comment
  records that upstream calls `engine_client.check_health()` first and that this
  server "currently exposes process liveness only".
- **Ours defers the warm-up past the listen.** `server_main.cpp` constructs
  `LoadedEngine::FromModelDir` before `server.listen` (`:1465`), so the weight
  load is inside the 53 s — but there is no dummy run and no kernel warm-up
  anywhere in that path, and the decode CUDA graph "captures once per padded
  batch size" on the first pure-decode step (`runner.cpp:1329-1334`). That
  capture, and every kernel's first-call initialisation, land on the first real
  request.
- **vLLM does that work before it serves.** `run_server_worker` enters
  `build_async_engine_client` and only then calls `build_and_serve`
  (`api_server.py:780-785`); the engine client's construction runs
  `initialize_from_config` → `compile_or_warm_up_model`
  (`executor/abstract.py:118-125`), which is `_dummy_run` per warm-up size,
  `kernel_warmup`, then `capture_model` (`gpu_worker.py:697-708`). Its `/health`
  additionally calls `check_health()`
  (`serve/instrumentator/health.py:22-33`).

So the 14.7x cell compares "process up, weights loaded" against "process up,
weights loaded, warmed and graph-captured". It is **not wrong**, and it is not
the like-for-like readiness comparison a reader will take it for. The caveat is
recorded in `docs/BENCHMARKS.md` beside the cell.

The same applies to the c8 tail. Eleven requests at **33.8–40.8 s** TTFT are a
real scheduling tail; the keepalive was not causing them, it was deleting them
from the measurement. They are now reported.

**The page-cache hypothesis is UNVERIFIED and is not asserted.** The recipe drops
the page cache before every leg, which raises the possibility that the first
request faults in ~55 GB of weights. Reading the code does not settle it: the
loader runs before `listen`, so the checkpoint read is nominally inside the 53 s,
and whether unified memory repays part of that cost on first touch is not
measured here. Nothing in this row depends on the answer; it is named so the next
person does not re-derive it as fact.

### The measurement guard, which is what stops the next one

`serve_low_common.require_complete_request_set` refuses to derive any rate from
a record that cannot prove its request set is whole, and is called at the
derivation sites rather than only at whichever validator runs first. Had it
existed, #915's c1 leg would have aborted instead of publishing 0.675x. It also
converts any future re-enabling of the keepalive from a silent 25% loss into a
loud failure, which is the protection #577 asked for and did not get.

## Owed

- **Index 1 at c8 is unexplained.** It carried a TTFT of 38.58 s in the no-ping
  arm (43.88 s in the default arm) — well past the 15 s interval — and
  SURVIVED, while eleven requests with TTFTs of 33.8–40.8 s failed. It is the
  one case in twelve where a long TTFT did not produce a ping, and it is the
  direct evidence that TTFT and collector-silence are not the same condition.
  What made that request's collector keep delivering while its neighbours' went
  quiet is not known. It must not be answered by fitting another threshold to
  aggregates.
- **The c8 latency tail is real and now visible.** Eleven requests at 33.8–40.8 s
  TTFT are a scheduling tail the keepalive was deleting rather than causing. Not
  a failure, not this row's scope, and no longer hidden.
- The #915 c1 and c8 output-throughput cells stay WITHHELD until re-run on a
  binary carrying this fix; that re-run is the acceptance evidence and belongs
  to [#915](https://github.com/mudler/vllm.cpp/issues/915). The 0.816x / 0.757x
  figures recorded above do **not** discharge it: single unpaired legs against
  #915's denominators are not the paired interleaved 3-rep protocol.
- Both throughput cells remain real gaps once honest. 0.816x at c1 and 0.757x at
  c8 need their own investigation, and it is not this row's.
- **The cold-start 14.7x cell needs its caveat carried, not just recorded.**
  ~90 s of first-inference initialisation sits behind our `/health`, which
  answers on process liveness only. The caveat is written into
  `docs/BENCHMARKS.md`; a like-for-like readiness comparison — first `/health`
  against first *completed* request, or a warm-up before `/health` returns —
  has not been measured and is not this row's scope.
- Whether dropping the page cache before every leg makes the first request fault
  in the weights is UNVERIFIED. Stated as a hypothesis; not used by any claim
  above.
- [#952](https://github.com/mudler/vllm.cpp/issues/952) — `--max-num-seqs` also
  bounds our HTTP concurrency, where upstream bounds only the scheduler batch.
  Found here while refuting worker-pool starvation as this bug's cause, and
  refuted as that cause: the pool was 36 against an offered concurrency of 8. It
  is a real mirror gap and needs its own change, because the next person raising
  `--max-num-seqs` for throughput silently raises the HTTP ceiling with it.
