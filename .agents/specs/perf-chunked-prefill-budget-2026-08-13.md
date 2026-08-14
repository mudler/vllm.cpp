# Chunked prefill: the budget IS applied — the invariant TTFT is arithmetic

**Row:** `SERVE-GATE-ONLINE` · **Issue:** [#669](https://github.com/mudler/vllm.cpp/issues/669)
· **Pin:** `5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0)

This is a grounding/verdict spike with one test attached, not an implementation
plan. It resolves whether our budget-invariant prefill time is **chunking that is
not happening** (a bug) or **chunking that costs the same either way** (physics),
and it records why the upstream mechanism the sweep appeared to reveal is not a
mechanism we lack.

## The measurement that raised it

35B `nvidia/Qwen3.6-35B-A3B-NVFP4`@`491c2f1e`, TTFT (`max_tokens=1`, 1024-token
prompts), clock pinned 2190/2184 MHz, single `boot_id`, oracle by explicit path,
GRAPHED, `--language-model-only` (**PIN ARM ONLY** -- our server has no such
flag and its log shows `multimodal image seam wired`; it is not a shared
condition), `VT_SERVER_SSE_PING_S=0` on our arm, arms interleaved,
n=3 except where marked (operator's measurement, 2026-08-13).

**Two reductions to disclose.** The `ours@8192/c1` cell is a mean of **2** legs:
its rep-1 leg recorded **6135.86 s** and was discarded. That leg is not data --
it is an artifact of a wedged harness whose resume ledger re-imported it, and
the harness's own reducer correctly flagged that cell NOT ESTABLISHED on a
34902x spread. Figures here are **means**; the harness prints **medians** with a
>5%-spread flag, so the two do not correspond cell-for-cell.

| `--max-num-batched-tokens` | conc | ours (s) | pin (s) | pin/ours |
|---|---|---|---|---|
| 8192 | 1 | 0.1766 [n=2] | 0.1605 | 0.91 |
| 8192 | 4 | 0.6814 | 0.8222 | 1.21 |
| 2048 | 1 | 0.1789 | 0.1599 | 0.89 |
| 2048 | 4 | 0.6828 | 0.5784 | 0.85 |

## Verdict

**Chunked prefill IS applied, and our composition is identical to upstream's.
There is no scheduler divergence and no product change to make.** Confirmed
twice: statically in the scheduler (§Evidence, §Upstream) and at runtime on the
GPU with the real 35B (§Runtime).

Our c4 time is invariant because one 4096-token forward and two 2048-token
forwards are the same total prefill work. That is the expected result, not the
absence of chunking. The two hypotheses produce *the same number*, which is
exactly why the sweep could not separate them — and why the composition needed
pinning (§Tests).

One thing the sweep *did* hide, and it is a measurement defect rather than a
product one: with no `--num-blocks`, our arm ran on 8192 tokens of KV against
the pin's 1,819,368, and under that starvation our arm could not form a wave
larger than 2048 tokens **at either budget** — so the two settings were
operationally identical on our side for a reason that has nothing to do with
chunked prefill (§Runtime, [#682](https://github.com/mudler/vllm.cpp/issues/682)).

## Evidence — the budget reaches the scheduler

Unbroken chain, every hop read:

| Hop | Anchor |
|---|---|
| CLI flag parsed | `src/vllm/entrypoints/openai/server_main.cpp:400` |
| into `EngineParams` | `server_main.cpp:806` |
| resolved (explicit override wins) | `src/vllm/entrypoints/model_loader.cpp:626-641` |
| into `SchedulerConfig` | `model_loader.cpp:704-717` (`MakeSchedulerConfig`) |
| into the `Scheduler` | `model_loader.cpp:1051-1058` |
| into `max_num_scheduled_tokens` | `src/vllm/v1/core/sched/scheduler.cpp:233-234` |
| into the per-step budget | `scheduler.cpp:465` |

`enable_chunked_prefill` is hard-`true` for this path (`model_loader.cpp:711`).
The encoder-decoder disable (`src/vllm/config/scheduler.cpp:52-57`) is not on it.
`AsyncScheduler` — the production default — does not override `schedule()`
(`include/vllm/v1/core/sched/async_scheduler.h:46-52`), so the budget path is the
same under async scheduling.

## Evidence — the scheduler composition (CPU)

Scheduler-composition probe, 4 x 1024-token prompts, `max_num_seqs=256`, CPU, no
model (`tests/vllm/v1/test_scheduler.cpp` helpers):

```
budget=8192   step0 total=4096   a=1024 b=1024 c=1024 d=1024
              step1 total=4      a=1 b=1 c=1 d=1
budget=4096   step0 total=4096   a=1024 b=1024 c=1024 d=1024      (identical)
budget=2048   step0 total=2048   a=1024 b=1024
              step1 total=2048   a=1 b=1 c=1024 d=1022            (clamp fires)
              step2 total=5      a=1 b=1 c=1 d=2                  (d's tail)
budget=1024   step0 total=1024   a=1024
              step1 total=1024   a=1 b=1023
              step2 total=1024   a=1 b=1 c=1022
              step3 total=1024   a=1 b=1 c=2 d=1020
```

Note that `8192` and `4096` are the *same* composition: 4096 <= both budgets, so
the operator's "8192 vs 2048" comparison is really **1x4096 vs 2x2048**.

Note also which line ends the 2048 step 0: the loop guard, not the clamp. Two
1024-token prompts take the budget to exactly 0, so the third request is never
peeked and there is no partial third chunk. In step 1 the *running* loop is
served first, so a/b's two decode tokens come off the budget before the waiting
loop runs and "d" is genuinely split 1022 + 2 — the clamp firing mid-prompt.

## Runtime — the budget binds on the GPU, and what the sweep hid

dgx.casa (GB10), 35B NVFP4 @`491c2f1e`, `~/work/mnbt-src` server, clock pinned
2190 MHz (verified at each leg start/end), `VT_SERVER_SSE_PING_S=0`,
`--max-num-seqs 32 --no-enable-prefix-caching`, boot_id recorded. The sweep
launched its concurrent requests as `curl &` in a bash loop, which skews arrivals
by a process spawn each; these fire all four off one barrier on **pre-connected**
sockets, and record each request's OWN latency rather than only the burst's wall
clock. Raw: `dgx:~/work/chunkprobe/`.

**Starved KV — the sweep's own provisioning (no `--num-blocks`; the server logs
`auto-fit max_model_len: reduced from 262144 to 8192 to fit the KV cache (256
blocks x 32 tokens)` = 8192 tokens TOTAL):**

| mnbt | per-request latency (s) | spread | burst wall |
|---|---|---|---|
| 8192 | 0.2598 0.5359 0.5364 0.6564 | 2.53 | 0.6573 |
| 8192 | 0.2620 0.5365 0.5366 0.6579 | 2.51 | 0.6587 |
| 2048 | 0.2601 0.5388 0.5391 0.6611 | 2.54 | 0.6623 |
| 2048 | 0.2608 0.5374 0.5376 0.6594 | 2.53 | 0.6598 |

Identical to three decimals at both budgets, and the server's own prefill log
pairs at most **two** requests per step (equal `elapsed_s`: 0.224935 / 0.224910).
The scheduler never saw more than 2048 tokens of waiting work, so the budget
could not bind either way. Nothing here is about chunked prefill.

**Provisioned KV (`--num-blocks 2048 --max-model-len 8192`), same everything
else:**

| mnbt | per-request latency (s) | spread | co-scheduled per step (equal `elapsed_s`) |
|---|---|---|---|
| 8192 | 0.6650 0.6650 0.6654 0.6657 | **1.001** | 3 x 0.27737 / 0.27739 / 0.27735 |
| 8192 | 0.3062 0.6452 0.6453 0.6460 | 2.11 | 3 x 0.25551 / 0.25544 / 0.25548 |
| 2048 | 0.5090 0.5533 0.5541 0.6751 | 1.33 | **2** x 0.227394 / 0.227343 |
| 2048 | 0.2588 0.5330 0.5333 0.6556 | 2.53 | **2** x 0.210219 / 0.210237 |

**That is the budget binding, measured on the GPU.** At 8192 the wave collapses
into a single step — all four requests return within 0.7 ms of each other — and
three share one forward. At 2048, with the identical client and identical KV, the
co-scheduled group is capped at exactly two, which is 2048 tokens. This is the
CPU composition table above, reproduced by the real engine.

And the burst wall clock is *still* flat: 0.6659 at 8192 vs 0.6755 / 0.6564 at
2048, against c1 legs of 0.1721 / 0.1709. Same total work, same time. The
invariance was never evidence about chunking.

Two consequences worth separating:

* **Product:** none. The budget is applied, at both tiers, exactly as upstream
  applies it.
* **Measurement:** the sweep compared an arm holding 8192 tokens of KV against
  one holding 1,819,368 (`pin-mnbt8192-c4-r2.log`, `kv_cache_utils.py:2214`, at
  `--gpu-memory-utilization 0.6`). That is a 222x asymmetry in a paired
  comparison, and on our side it suppressed the very batching the sweep was
  varying. Filed as [#682](https://github.com/mudler/vllm.cpp/issues/682).

## Upstream — line for line

Our waiting loop against the pin. Verified by reading, not inherited:

| Element | Ours | Pinned vLLM |
|---|---|---|
| budget init | `scheduler.cpp:465` | `scheduler.py:447` |
| running loop drains budget first | `scheduler.cpp:478,512,592` | `scheduler.py:473,511,620-621` |
| waiting loop guard `token_budget > 0` | `scheduler.cpp:642` | `scheduler.py:671` |
| `max_num_seqs` break | `scheduler.cpp:643-645` | `scheduler.py:673-676` |
| `num_new = num_tokens - num_computed` | `scheduler.cpp:688` | `scheduler.py:825` |
| `long_prefill_token_threshold` cap (0 -> inert) | `scheduler.cpp:689-692` | `scheduler.py:845-847` |
| chunked-disabled break (inert, chunked ON) | `scheduler.cpp:694-696` | `scheduler.py:850-857` |
| `min(num_new, token_budget)` | `scheduler.cpp:697` | `scheduler.py:859-860` |
| `token_budget -= num_new` | `scheduler.cpp:742` | `scheduler.py:1018-1019` |
| post-conditions | `scheduler.cpp:763-764` | `scheduler.py:1054-1058` |

`max_num_scheduled_tokens` falls back to `max_num_batched_tokens` on both sides
(`include/vllm/config/scheduler.h:156-158` <- `scheduler.py:109-113`); it is only
smaller under speculative decoding (`vllm/config/vllm.py:1699-1712`), which is
off here.

Three things upstream does **not** have, checked because their absence would have
been the divergence: `max_num_partial_prefills` / `max_long_partial_prefills`
(removed upstream — zero hits at the pin; `vllm/v1/core/sched/utils.py` holds only
repetition helpers), a decode reservation or prefill/decode budget split (one flat
`token_budget`), and a structured-output budget carve-out (`scheduler.py:1248-1249`
never touches the budget). The encoder budget is separate but mnbt-derived on both
sides (`vllm/config/scheduler.py:238-239` <- `src/vllm/config/scheduler.cpp:59-60`)
and inert for text-only.

## Why vLLM swings and we do not

`max_num_batched_tokens` changes almost nothing else upstream:

* **CUDA-graph capture sizes are identical.** `vllm/config/vllm.py:1795-1802`
  caps at `min(max_num_seqs * decode_query_len * 2, 512)` then
  `min(max_num_tokens, that)` -- which on this config resolves to **64**, not
  512. Both 2048 and 8192 give the same cap. Both 2048- and 4096-token batches
  exceed it, so
  `vllm/v1/cudagraph_dispatcher.py:272-281` returns `CUDAGraphMode.NONE` with
  zero padding for both. Ruled out.
* **Runner buffers** (`gpu_model_runner.py:505,764-857`) and the **FlashInfer
  prefill workspace** (`flashinfer.py:944-958`) scale with mnbt but are
  allocate-once. Attention plan/split-kv/`reorder_batch` are not mnbt-dependent
  (`flashinfer.py:643-681`, `backends/utils.py:665`).
* **One real difference:** mnbt is in the compilation hash
  (`vllm/config/scheduler.py:206-216`, citing vllm#29585 — Inductor picks 32- vs
  64-bit indexing from the size hint) and bounds `compile_ranges`
  (`vllm/config/vllm.py:1918-1920`). Different Inductor artifacts per budget.
  Small, and it would also show at c1 — where both arms are flat, so it is not
  the c4 effect.

So vLLM's 30% is not a chunked-prefill fast path. The harness (`dgx:~/mnbt.sh`)
times `t1-t0` around the whole burst — **wall clock to drain it, not mean
TTFT** — so the mean-TTFT staggering artifact that a 1x4096-vs-2x2048 split
would otherwise produce is not in these numbers at all. Reading the columns as
prefill throughput (4096 tokens per c4 burst; ours from the provisioned-KV legs
above, which are the ones that actually run the stated forward shape):

| tokens per forward | ours | pin |
|---|---|---|
| 1024 (c1) | 5971 tok/s | 6380 |
| 2 x 2048 | 6150 | **7081** |
| 1 x 4096 | 6151 | **4982** |

**The two columns are NOT PAIRED and must not be differenced.** `ours` comes
from the chunkprobe barrier-client with `--num-blocks 2048`; `pin` from
`mnbt.sh`'s curl spawn-loop with `--gpu-memory-utilization 0.6` — different
client, different KV provisioning, and the two were never interleaved WITH EACH
OTHER. Each column is internally sound; across columns, `6151 vs 4982` and
`6150 vs 7081` are unpaired readings.

**ESTABLISHED (intra-pin, paired):** vLLM's **4096-token forward is the outlier** — 30% less efficient per token than
its own 2048-token forward, while its 2048 number sits on the same near-linear
trend as its c1. Ours is flat across forward sizes: no collapse at 4096, and no
gain at 2048 either. The 1.21x at 8192/c4 is us beating an upstream pathology,
not a win to bank; the 0.85x at 2048/c4 is the honest number, and it is the same
~10-15% prefill deficit already visible at c1 (0.89-0.91x) — the separate problem
the operator correctly refused to conflate.

The flat row is the finding to carry forward, **as a HYPOTHESIS rather than a
measurement**, because it is a cross-column comparison and the columns are
unpaired (above): we appear to extract no benefit from a larger prefill forward
(5971 -> 6150 -> 6151, +3.0% total) where the pin appears to extract +11% going
from 1024 to 2048. Sizing it needs one interleaved same-client run. Whatever it does with a
2048-token forward that it cannot do with a 1024-token one, we do not do — and
that, not chunking, is where the c4 prefill gap lives.

## Tests

`tests/vllm/v1/test_scheduler.cpp` — `Scheduler.schedule: the token budget splits
a 4x1024 prefill wave`. Two subcases pinning the composition above: 8192 admits
the wave in one 4096-token step with nobody chunked; 2048 splits it into two
2048-token steps and pins the **1022**-token remainder chunk and d's 2-token
tail. Mirrors `scheduler.py:671,825,859-860,1018-1019`.

Assertion counts: `test_scheduler` 36 cases / 423 assertions -> **37 / 448**.

**Mutation proof** (both in-place, tree restored byte-for-byte, `md5sum`
`6b88fd41533c097eaefabc7f8f2936d4` before and after, `git status` clean):

| Mutation | Result |
|---|---|
| `token_budget = 1000000` (budget never reaches the scheduler — the hypothesis) | **RED**, 5 failed |
| drop `min(num_new_tokens, token_budget)` in the waiting loop | **RED**, 5 failed |

## Gates

CPU, clean `-Werror` build (`build_exit=0`), Release:

| Suite | Result |
|---|---|
| `test_scheduler` | 37/37 cases, 448/448 assertions, SUCCESS |
| `test_scheduler_config` | 13/13, 32/32, SUCCESS |
| `test_sched_output` | 8/8, 52/52, SUCCESS |
| `test_request_queue` | 26/26, 1839/1839, SUCCESS |
| `test_scheduler_wave` | 3/3, 44/44, SUCCESS |

`scripts/agent-preflight.sh` is green except `test_cpu_x86_llamacpp_floor`,
which is [#618](https://github.com/mudler/vllm.cpp/issues/618) — the known
load-dependent exit-4-instead-of-2 case, reproduced here at loadavg 201 on a
shared box. It touches no path in this change.

The three GPU SACRED gates (`test_qwen36_paged_engine` 315/315,
`test_qwen27_paged_engine` 235/235, `test_qwen27n_fp8_tower_paged_engine`
236/236) are **PENDING**. `dgx.casa` was unreachable for most of the session and
came back with 60 GiB free on a 99%-full disk; the GPU window it did give was
spent on the runtime probes above, which is where the open question actually
was. This change touches only `tests/` and records — no product path — so it
cannot move a token, but the gates are owed before merge regardless.

## Risks / decisions

No vLLM-defined behavior is reopened. Our `ResolveMaxNumBatchedTokens` default
(`model_loader.cpp:626-641`: MoE 8192 at `max_num_seqs >= 32` else 4096; dense
2048) does NOT clearly diverge from upstream. The `2048` at
`vllm/config/scheduler.py:49` is a **pydantic testing default**;
`EngineArgs._set_default_args` overrides it to **8192/16384 on a >=70 GiB GPU**,
which GB10 is. So our MoE-8192 may BE upstream's effective default here, and the
earlier framing -- that choosing 8192 flattered us against a "true" 2048 bar --
is withdrawn.
That deviation is unchanged here and is justified by its own recorded conc-64
measurements — and this sweep is consistent with it being harmless for us, since
our throughput does not move with forward size. It is worth re-testing at conc-64
now that we know the pin's own 4096-token forward is pathological, because the
`+2.7% at 8192` that motivated the default was measured against ratios that
include that pathology.

## Stop conditions / open

Two things this spike deliberately did not close, each with its instrument named
so neither becomes a declared ceiling.

1. **Why a bigger prefill forward buys us nothing** (5971 -> 6150 -> 6151 tok/s
   across 1024 / 2048 / 4096 tokens per forward, where the pin gains 11% over the
   same first step). Next traceable hypothesis: profile BOTH arms on the
   *identical forward shape* — a single 2048-token prefill step — and pair the
   kernels by call count, per
   [`.agents/benchmarking.md`](../benchmarking.md). Our flatness across forward
   sizes is the signature of a per-token cost that does not amortize, which is
   what a profile pairs off directly. Do not read the 1.21x at mnbt=8192 as
   headroom: it is the pin's own 4096-token pathology.
2. **What the pin's 4096-token forward hits.** It is 30% less efficient per token
   than its own 2048-token forward on this box. Worth knowing before we inherit
   anything from its batching behaviour, and worth checking whether it is a
   GB10/unified-memory effect rather than a vLLM one.

Both are blocked only on GPU time, not on evidence.
