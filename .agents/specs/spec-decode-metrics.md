# Speculative-decoding metrics on the OpenAI server (`SERVE-METRICS`)

Issue: [#2770](https://github.com/mudler/vllm.cpp/issues/2770).
Row: `SERVE-METRICS` (engine-matrix §Serving). That row's own text closes with
`RESIDUAL: config-gated families (spec-decode/kv-connector/mm/LoRA)`, so this
work discharges the first named item of its own residual rather than opening a
new lane. #2770 named two candidate owners, `SPEC-DFLASH2` and "the
OpenAI-server row"; the second is this one, and the deciding fact is that
nothing here touches the speculator. The change is entirely in the scheduler's
stat snapshot, the Prometheus catalog and the server's construction of it.

Companion specs: [prometheus-metrics.md](prometheus-metrics.md) (the registry
and the always-on catalog), [async-metrics.md](async-metrics.md) (the
`AsyncLLM` step-site fold that carries these values on the serving path, and
whose `## Scope` explicitly deferred "spec-decode" to a later residual).

## Scope

**In.** The four speculative-decoding metric families vLLM registers when a
speculative config is present, exported through the same `GET /metrics` route
the server already serves, and fed from the same per-step `SchedulerStats`
channel every other family is fed from:

| Family | Exposition name |
|---|---|
| `vllm:spec_decode_num_drafts` | `vllm:spec_decode_num_drafts_total` |
| `vllm:spec_decode_num_draft_tokens` | `vllm:spec_decode_num_draft_tokens_total` |
| `vllm:spec_decode_num_accepted_tokens` | `vllm:spec_decode_num_accepted_tokens_total` |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="d"}` |

Concretely:

1. `SpecDecodingStats` and `SpecDecodingProm` ported into
   `vllm/v1/spec_decode/metrics.{h,cpp}`, mirroring the upstream module of the
   same name 1:1 in field names, counter names, help strings, label schema and
   gating.
2. `SchedulerStats` gains the `spec_decoding_stats` optional
   (`stats.py:206`), set only on a step that verified drafts.
3. `Scheduler::make_spec_decoding_stats` (`scheduler.py:2714-2731`) called from
   the exact site in `update_from_output` where our port already computes
   `num_draft_tokens` and `num_accepted` and left the comment
   `(make_spec_decoding_stats telemetry is deferred — no SpecDecodingStats.)`.
4. `PrometheusStatLogger` takes the resolved speculative-token count, registers
   the four families when it is non-zero, and folds `spec_decoding_stats` into
   them in `Record` (`loggers.py:477-481`, `:1140-1143`).
5. `vllm-server` passes that count from the engine it just loaded, so the
   families are live on the shipped server rather than on a hand-built logger.

**Out.** The three sibling residuals on this row (kv-connector, multimodal
cache, LoRA). `SpecDecodingLogging` (`metrics.py:52-174`), the human-readable
`LoggingStatLogger` line, which this port has never registered for any family.
The diffusion arm of `SpecDecodingProm` (`metrics.py:215-226`,
`vllm:diffusion_num_*`), because `is_diffusion` has no analogue in this port's
text engine and inventing one would be a guess rather than a mirror. DP /
multi-engine aggregation (`engine_idx`), already out on this row.

## What the tree said, against what the issue said

#2770 states that `spec_drafts_proposed()` and `spec_drafts_accepted()` have
"their only non-test readers in `examples/bench/bench_core.h`". **That was true
when the issue was filed and is false at `c796fea41`.** Two further production
readers exist:

- `src/capi/spec_acceptance.h:54-56`, feeding `vllm_engine_spec_acceptance`
  (ABI v25) — a C-ABI surface, so acceptance is reachable from `include/vllm.h`;
- `examples/cli/main.cpp:372-375`, which prints the per-leg delta.

The issue's **ask** is unaffected and remains open: neither reader is the HTTP
server, and `GET /metrics` still exports no spec-decode family. The correction
matters for one design decision below, so it is recorded here rather than
carried as an unstated assumption.

## Upstream chain

Read at the parity pin `e126687a9a828d513c01a07cd69f025f27d63280`
(vLLM `0.28.1rc1.dev132+ge126687a9`), which is what
[upstream-sync.md](../upstream-sync.md) records on `main`.

**THE PIN THIS ROW ALMOST USED WAS THE WRONG ONE, and the correction is
recorded rather than quietly applied.** #2770's body, and the first draft of
this spec, both name `5559679229bc961848b121ccdeaa8fa5d79bec98`. That value is
real but stale: it is what the shared checkout's current branch carries, not
what `origin/main` does, and AGENTS.md says in as many words that a resolved
value read out of a document is another reader's answer. Re-read at
`e126687a9`, `vllm/v1/spec_decode/metrics.py` and `vllm/v1/metrics/loggers.py`
are **byte-identical** to the older revision (`git diff 5559679229 e126687a9a`
on both paths is empty), and `make_spec_decoding_stats` is byte-identical too,
so the ported behaviour is unaffected. Only line anchors moved, and the six
`test_schedule_spec_decoding_stats` cases carry the same `expected` tuples. Every
anchor below is the `e126687a9` one.

| Upstream | What it defines |
|---|---|
| `vllm/v1/spec_decode/metrics.py:17-49` | `SpecDecodingStats`: the per-step aggregate, `new()` and `observe_draft()` |
| `vllm/v1/spec_decode/metrics.py:177-196` | `SpecDecodingProm` and the PromQL the names are chosen to serve |
| `vllm/v1/spec_decode/metrics.py:200-213` | the gate: `spec_decoding_enabled = speculative_config is not None`; nothing is registered when it is off |
| `vllm/v1/spec_decode/metrics.py:228-245` | the three flat counters, their names and help strings |
| `vllm/v1/spec_decode/metrics.py:247-264` | the per-position counter, its extra `position` label and its `num_speculative_tokens` cardinality |
| `vllm/v1/spec_decode/metrics.py:266-281` | `observe()` |
| `vllm/v1/metrics/stats.py:12,206` | `SchedulerStats.spec_decoding_stats`, `None` by default |
| `vllm/v1/core/sched/scheduler.py:1828,1918-1924,2211` | the per-step local, its update site inside the acceptance block, and its hand-off to `make_stats` |
| `vllm/v1/core/sched/scheduler.py:2676-2712` | `make_stats`, which threads `spec_decoding_stats` onto the snapshot unchanged |
| `vllm/v1/core/sched/scheduler.py:2714-2731` | `make_spec_decoding_stats`: the `log_stats`/`num_draft_tokens` gate, lazy construction, and the `num_invalid_spec_tokens` subtraction |
| `vllm/v1/metrics/loggers.py:447,477-481` | `PrometheusStatLogger` constructing `SpecDecodingProm` from `vllm_config.speculative_config` |
| `vllm/v1/metrics/loggers.py:1140-1143` | the `Record` fold |
| `tests/v1/core/test_scheduler.py:1307-1439` | `test_schedule_spec_decoding_stats`, the six-case parameterisation this port takes verbatim |
| `tests/v1/kv_connector/nixl_integration/test_spec_decode_acceptance.py:129,172` | the `_total`-suffixed names as they appear in a real scrape |

## Our baseline

State at `c796fea41`, the base this row branched from, and re-verified at
`885ed633f` after merging 46 commits of `main` into it -- both headers still
say this, and the scheduler still carries the deferral comment. Both headers
named the gap themselves, so nothing here had to be discovered:

- `include/vllm/v1/metrics/loggers.h` listed "spec-decoding metrics" as DEFERRED
  in the same sentence as the other three config-gated families.
- `include/vllm/v1/metrics/stats.h` said the advanced `SchedulerStats` fields
  "(DP wave, cudagraph, spec-decode, kv-connector, lora, eviction events) are
  deferred with their config-gated metric families".
- `src/vllm/v1/core/sched/scheduler.cpp` computes `num_draft_tokens`,
  `num_accepted` and `num_rejected` inside `update_from_output` for the
  num-computed rollback, and closes that block with
  `// (make_spec_decoding_stats telemetry is deferred — no SpecDecodingStats.)`.
  Every input the upstream telemetry needs is therefore already in hand at the
  call site; only the aggregate is missing.
- `Scheduler::make_stats()` already stashes and republishes the per-step
  prefix-cache delta (`last_prefix_cache_stats_`), which is the exact shape the
  spec delta needs.
- `src/vllm/entrypoints/openai/server_main.cpp` constructs the one
  `PrometheusStatLogger` and attaches it to both frontends and to the server.

## Port map

| Upstream | Here |
|---|---|
| `vllm/v1/spec_decode/metrics.py` `SpecDecodingStats` | `include/vllm/v1/spec_decode/metrics.h` (struct, `New`, `ObserveDraft`) |
| `vllm/v1/spec_decode/metrics.py` `SpecDecodingProm` | `include/vllm/v1/spec_decode/metrics.h` + `src/vllm/v1/spec_decode/metrics.cpp`, registering into the caller's `PromRegistry` |
| `stats.py:206` | `SchedulerStats::spec_decoding_stats` in `include/vllm/v1/metrics/stats.h` |
| `scheduler.py:2714-2731` | `Scheduler::make_spec_decoding_stats` |
| `scheduler.py:1918-1924` | the call at the existing acceptance block in `Scheduler::update_from_output` |
| `scheduler.py:2696,2708` | `Scheduler::make_stats` reading `last_spec_decoding_stats_` |
| `loggers.py:477-481` | `PrometheusStatLogger`'s new `num_speculative_tokens` constructor argument |
| `loggers.py:1140-1143` | the `Record` fold |
| `loggers.py:452` (`vllm_config`) | `LoadedEngine::speculative_config()` + the `server_main.cpp` construction |

### Deviations, each with its reason

- **`make_stats()` reads a stash rather than taking an argument.** Upstream
  passes `spec_decoding_stats` down from `update_from_output` into `make_stats`
  because both live in the same call. Our `make_stats()` is called by
  `EngineCore` *after* `update_from_output` returns, so it cannot receive the
  value; it already solves the identical problem for the prefix-cache delta with
  `last_prefix_cache_stats_`, and this follows that precedent exactly. The
  stash is overwritten (with an empty optional when the step verified no draft)
  on every `update_from_output`, so no step can re-report the previous one's.
- **`SpecDecodingProm` writes into a shared `PromRegistry`** instead of owning
  `prometheus_client.Counter` objects, because that is what
  `prometheus-metrics.md` already established for every other family here.
- **The source is the scheduler, not `GpuRunner`'s cumulative counters.**
  `spec_drafts_accepted()` measures the same quantity and is already exported
  over the C ABI, so reading it here would be shorter. It is the wrong source:
  upstream has exactly one, it is the scheduler, and the runner's is
  GPU-runner-specific while `SchedulerStats` is arch-agnostic and already
  reaches the logger. Mirroring vLLM decides this.
- **No `is_diffusion` arm** — see `## Scope`, Out.

## Tests to port

| Upstream | Here |
|---|---|
| `tests/v1/core/test_scheduler.py:1307-1439` `test_schedule_spec_decoding_stats` | all six parameterised cases, over the production `Scheduler::update_from_output` + `make_stats()`, asserting `num_drafts`, `num_draft_tokens`, `num_accepted_tokens` and `num_accepted_tokens_per_pos` and the `None`-on-a-draft-free-step polarity |
| `tests/entrypoints/serve/instrumentator/test_metrics.py` (substring scrape shape) | the four family names asserted PRESENT in the exposition with a speculative config and ABSENT without one |

Plus one test upstream has no analogue for, because it has no C++ server: the
reachability gate below.

## Reachability

The production entry point is `GET /metrics` on the shipped server —
`ApiServer::handle_metrics()` — reached from `server_main.cpp`, which is the
only place a `PrometheusStatLogger` is constructed in production.

The gate drives a speculative engine end to end and scrapes it through that
handler: a spec-configured `Scheduler`, a drafting `ModelRunnerBase`, `AsyncLLM`
with `check_for_draft_tokens`, a completion issued through
`OpenAIServingCompletion`, then `ApiServer::handle_metrics()`, asserting the
accepted-token counter has moved. Deleting any of the three new production call
sites (the `make_spec_decoding_stats` call, the `make_stats` republish, the
`Record` fold) must red it.

A unit test over `SpecDecodingStats::ObserveDraft` is kept because it localises
a failure, and it is not the proof.

**One link is proven by inspection and not by a test, and this says so.** No
test executes `server_main.cpp`, because reaching its logger construction needs
a real speculative checkpoint and the CPU tier has none; the existing `/metrics`
gate has always had the same property. What the gate does prove is that the
argument is load-bearing — constructing the logger without k reds the same case
— and that `server_main.cpp` is the ONLY production construction of a
`PrometheusStatLogger` in the tree, which is checkable with one grep. Recorded
here rather than left for a reader to discover, because an unpassed parameter is
the first shape `.agents/reachability.md` names.

## Gates

CPU `ctest` is the gate; this row is CPU-gated already (`SERVE-METRICS` landed
CPU-gated) and nothing here is device code. Focused: the new suite plus
`test_prometheus_metrics`, `test_scheduler`, `test_openai_api_server`,
`test_engine_core_proc`. Full: `ctest` on the CPU build.

No performance axis. The families are inert with no speculative config, and
with one the per-step work is four counter increments on a path that already
takes a mutex.

## Risks/decisions

1. **A metric that reads zero is worse than an absent one** — the reason #277
   existed on this same row. The reachability gate is written so that a wiring
   that registers the families and never moves them is RED, not green.
2. **Per-position cardinality is `num_speculative_tokens`**, fixed at
   construction. A step that somehow reports more accepted tokens than that
   would index out of range; upstream asserts against it in `observe_draft`
   (`metrics.py:45`) and this port keeps the assertion rather than clamping,
   because clamping would hide a scheduler defect in a metric.
3. **The `_total` suffix comes from the exposition, not the family name.** The
   registered name is `vllm:spec_decode_num_accepted_tokens`; a reader looking
   for the issue's `..._total` finds it only in the scrape. Both spellings are
   asserted so neither can drift.
4. **Adding a constructor argument to `PrometheusStatLogger`** touches every
   construction site. It is defaulted to 0 — "no speculative config" — so every
   existing site keeps today's behaviour and the families stay unregistered.

## Owed

**The per-request spec-decode accumulator, which exists at this pin and did not
at the previous one.** `e126687a9` adds
`ObservabilityConfig.per_request_spec_decode_metrics`
(`scheduler.py:97-98,1925-1938,2408`) and a `Request.spec_decode_metrics`
histogram bucketed by accepted-draft count, asserted by the fourth parameter
`expected_per_req` that `test_schedule_spec_decoding_stats` gained. It is a
DIFFERENT surface from the four Prometheus families — per request, not per step,
and not exported through `/metrics` — so it is out of scope here rather than
half-ported. Nothing in this tree implements it; it is named so the next reader
of the upstream test does not mistake the missing fourth parameter for an
omission in this port.

## Stop conditions

- Return `NEEDS_DECISION` if mirroring the upstream gate (families absent
  without a speculative config) is judged unacceptable for the head-to-head
  #2761 needs, because always-registering them is a divergence from vLLM and
  not a mirror.
- Return `NEEDS_CONTEXT` if the reachability gate cannot be made to draft on a
  CPU build, since a green that never accepted a token measures nothing.

## Now

`SERVE-METRICS` stays landed. This spec discharges the `spec-decode` item of
its recorded residual; the other three (kv-connector, multimodal cache, LoRA)
remain owed on the row.
