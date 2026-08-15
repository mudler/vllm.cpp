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

Row is `READY`. Spec committed; implementation not started.
