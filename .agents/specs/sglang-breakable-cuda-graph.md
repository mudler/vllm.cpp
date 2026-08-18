# SGLang Breakable CUDA Graph: what it is, and what it is worth here

**Owning rows:** `SGLANG-BCG` in [sglang-matrix.md](../sglang-matrix.md) for the
inventory row; `ENG-CUDAGRAPH` in [engine-matrix.md](../engine-matrix.md) for the
prefill verdict this spec records against it.
**Issue:** [#1161](https://github.com/mudler/vllm.cpp/issues/1161).
**Recommendation:** record the surface, record the prefill refutation, and derive
three rows. One of the three is blocked on work that is already open.

This spec exists because a public claim reached the operator and needed grounding
before it could change any plan. The claim: *"SGLang is the first engine to develop
and land Breakable CUDA Graph (BCG), the full CUDA Graph, and graph memory reuse.
BCG drops torch.compile for faster setup and broader compatibility. Full graph capture
brings prefill latency down on dynamic workloads. Memory reuse keeps the graph
footprint fixed as coverage grows."*

Every claim below was read in the pinned tree. None was taken from the claim text.

## Now

`ANALYSIS`. No engine code is in scope here. The derived work is under
[`## Owed`](#owed) and each item has its own issue.

## Pins

| Component | Pin | Where |
|---|---|---|
| SGLang | tag `v0.5.15`, commit `f63458b5beaceabbd9d749b9fc956370e1b649e6` | [oracles/sglang.md](../oracles/sglang.md); paths below are relative to `python/sglang/srt/` |
| vLLM | `555967922` / 0.26.0.dev0 | [upstream-sync.md](../upstream-sync.md); the mirror source |
| Ours | `affc2a7fd` | the base of this branch |

Two SGLang commits cited under [§5](#5-diffusion-is-the-one-shape-where-the-lever-is-real-and-it-is-blocked)
post-date the pin. They are cited as **dated upstream events**, never as pinned
evidence, and no gate here depends on them.

## 1. What BCG is

BCG captures one forward as a **sequence** of `torch.cuda.CUDAGraph` segments split at
eager break points, not as one graph.

| Mechanism | Anchor |
|---|---|
| `eager_on_graph(True)` ends the current segment, runs the marked call eagerly once, records a `replay_fn` closure over weak-ref'd args, and begins a new segment | `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:204-241` |
| Replay is `for seg: seg.replay(); break_fns[i]()` | `:244-260` |
| Every segment shares one mempool; intermediates are weak-ref'd, so the pool reclaims across segments and the activation footprint does not grow with segment count | `:14-23`, `:156-169` |
| "segment-captured graphs with eager break markers. No torch.compile." | `runner_backend/breakable_cuda_graph_backend.py:14-17` |

The break points are the attention and recurrent calls: `layers/radix_attention.py:256`,
`models/deepseek_common/attention_forward_methods/forward_mla.py:1092`,
`layers/radix_linear_attention.py:159`, `models/nemotron_h.py:1240`,
`layers/attention/dsa/dsa_indexer.py:2414`.

"Graph memory reuse" is **two** mechanisms, and the claim text merges them. One is the
shared mempool above. The other is executable dedup at
`runner_backend/cuda_graph_dedup_mixin.py:219-242`: hash each captured graph's
topology, and on a signature hit re-point one shared `cudaGraphExec` with
`cudaGraphExecUpdate` instead of instantiating a second one. It logs
`"captured %d CUDA graphs, deduped to %d execs"` at `:358`.

Backend choice is per phase. `cuda_graph_config.py:38-45` offers
`FULL | BREAKABLE | TC_PIECEWISE | DISABLED`; decode defaults to `FULL`, prefill
defaults to `BREAKABLE` on CUDA (`:95-112`).

## 2. The priority claim, stated accurately

vLLM's v1 default is already `FULL_AND_PIECEWISE`: full graph for decode, piecewise
graph split at `splitting_ops` for prefill and mixed
(`vllm/config/compilation.py:60-63,517,615,630`). The **capability** has been in the
mirror source, and therefore in our benchmark denominator, for a long time.

What SGLang landed first is the **mechanism**: piecewise coverage from runtime stream
capture plus a decorator, with no Dynamo, no Inductor and no FX splitting. That is a
real result and it removes a compiler dependency. It is not new engine behavior. A
reader of the claim text would conclude otherwise, which is why this section exists.

## 3. Claim by claim, against our measurements

| Claim | Verdict here | Basis |
|---|---|---|
| BCG drops `torch.compile` for faster setup and broader compatibility | **No lever.** We never had a compiler. `src/vt/cuda/cuda_backend.cu:203-232` is already raw `cudaStreamBeginCapture` / `cudaGraphInstantiate`. BCG is SGLang arriving where a C++ engine starts. | our own capture path |
| Full graph capture brings prefill latency down on dynamic workloads | **Refuted for our CUDA gate models.** | see below |
| Memory reuse keeps the graph footprint fixed as coverage grows | **Applies, and is unported.** | `grep -rn "cudaGraphExecUpdate" src include` is empty |

### The prefill refutation

Measured on GB10, 2026-07-09, both arms, same instrument, oracle graphed and
identity-asserted:

- steady-state prefill **GPU-idle between launches is 3.8%**; GPU-busy is above 96% on
  both arms. The one large raw idle gap is a single one-time JIT and capture window.
- the 27B prefill gap is **92.5% non-GEMM glue GPU work**. The dominant GEMM is the
  same symbol at the same call count, +0.17%. Attention is where **we are ahead**.

There are no launch bubbles in our prefill for a graph to collapse. vLLM's piecewise
prefill graph is already inside the denominator we lose to, so it does not explain the
gap either. Decode is a separate case and is already captured: batched decode at
concurrency 64 was about 24% launch overhead and is already one `cudaGraphLaunch` per
step.

**This is a negative result and it is recorded so nobody re-derives it.** It does not
generalize to a path that is currently eager and currently host-bound. Any future
prefill-graph proposal must name such a path and measure it first.

## 4. The structural finding, which is not a speed finding

Our capture is all-or-nothing. `src/vllm/v1/worker/gpu/runner.cpp:1338-1341` routes
only `pure_decode` batches to a graph, so one host-dependent op forces the whole step
eager. Coverage is a cliff, not a slope, and
[#1020](https://github.com/mudler/vllm.cpp/issues/1020) is one instance of it.

**NINE call sites hand-roll capture**, corrected 2026-08-18
([#1179](https://github.com/mudler/vllm.cpp/issues/1179)); this section recorded
eight when it landed in `9bc4d7f44`. They are `Qwen3_5DecodeGraph`,
`Qwen3_5DenseDecodeGraph`, `Qwen3MoeDecodeGraph`, `Qwen3DenseDecodeGraph`,
`DeepseekV2DecodeGraph`, `VoxtralDecodeGraph`, plus file-local graph code in
`deepseek_v4.cpp`, `laguna.cpp` and — the one this section missed — the DFlash draft
graph at `src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106`.
The full table with a declaration and implementation anchor per driver is
[eng-cudagraph-break.md](eng-cudagraph-break.md) `## Our baseline`. Each re-derives
capture, bucket padding, persistent-input threading and the pure-decode predicate.
AGENTS.md names this shape: a parallel path written by hand instead of a shared seam.

**The duplication has already cost a shipped model its decode graph, so this is a
correctness finding and not only a coverage one.** The persistent DEVICE input path
`StepDevInputs` (`src/vllm/model_executor/models/qwen3_5.cpp:3894`) exists in exactly
one driver — `grep -c StepDevInputs` returns 41 lines in `qwen3_5.cpp` and 0 in each
of `qwen3_moe.cpp`, `qwen3.cpp`, `deepseek_v2.cpp` and `voxtral.cpp`. Because
`Qwen3DenseDecodeGraph` lacks it, `src/vllm/model_executor/models/qwen3.cpp:961-986`
DECLINES the graph outright whenever the asynchronous device-token mirror is live,
with its own measured comment: `depth-1, graph ON PASS 78/78`, `depth-2, graph OFF
PASS 82/82`, `depth-2, graph ON FAIL, slots 1-3 degenerate`. The comment names the
real fix as reading the identifiers at replay time from a stable device buffer, which
is precisely what the sibling driver already does. That is one capability, written
once, unavailable to four models, and a live mitigation standing in its place.

BCG is the right construction for the fix, because vLLM gets its split from
`torch.compile` and we have no compiler. The value is **coverage, one seam instead of
nine, and the correctness capability that seam makes shared**. It is not throughput
on its own, and it must not be sold as throughput. §3's refutation stands unchanged:
3.8% host idle in prefill, GPU-busy above 96%, and 92.5% non-GEMM glue.

## 5. Diffusion is the one shape where the lever is real, and it is blocked

Our diffusion path captures no graphs at all: `grep` for capture across
`src/vllm/model_executor/models/ltx2*.cpp` returns nothing. A denoise loop is the ideal
capture target, and SGLang measured large wins on exactly it after our pin: LTX-2
two-stage H200 e2e 10.75 s to 6.90 s (`d4be483efb`), SANA 1024px e2e -26%
(`6c7498113f`), SANA denoise 0.73 s to 0.457 s (`56ef810cad`).

**Discount the magnitude hard**, because their win is mostly PyTorch host tax per
denoise step, which our C++ loop does not pay.

**The premise fails anyway.** A denoise-loop graph needs a denoise loop that runs on the
device, and ours does not:

- [#1024](https://github.com/mudler/vllm.cpp/issues/1024) measured GPU utilization
  **exactly 0 in 321 of 347 samples** on a `--device cuda` render, with all 26 non-zero
  samples inside the staging window, and the process holding exactly 1.00 core of 20
  for over 17 minutes after staging.
- [#1007](https://github.com/mudler/vllm.cpp/issues/1007): the video VAE decode has no
  device arm.
- [#1087](https://github.com/mudler/vllm.cpp/issues/1087): **57-66% of render wall** is
  one single-threaded phase measuring 1731 s and 1732 s across two rungs whose voxel
  counts differ 2.75x. [#1010](https://github.com/mudler/vllm.cpp/issues/1010) is open
  because nothing timestamps a phase boundary, so the phase is not yet named.

A graph collapses host launch dispatch. There is close to no device dispatch here to
collapse. **Capturing now would measure nothing.**

## 6. What was deliberately not done

- No engine code, no capture change, no measurement taken.
- No claim that BCG makes us faster. Three of the four candidate levers are refuted,
  structural, or blocked.
- No oracle run. SGLang remains `gateable = no` on this project's hardware for
  correctness; the perf oracle stood up separately under `SGLANG-ORACLE-PERF`.
- No advance of the SGLang pin. The two post-pin diffusion commits are dated events,
  not evidence.

## Owed

| Item | Issue | Row |
|---|---|---|
| Graph executable dedup via `cudaGraphExecUpdate` | [#1162](https://github.com/mudler/vllm.cpp/issues/1162) | `ENG-CUDAGRAPH-DEDUP` |
| A shared capture seam with break points, retiring the NINE hand-rolled drivers | [#1163](https://github.com/mudler/vllm.cpp/issues/1163) | `ENG-CUDAGRAPH-BREAK` |
| Denoise-loop capture, blocked on #1010, #1087, #1024, #1007 | [#1164](https://github.com/mudler/vllm.cpp/issues/1164) | `ENG-CUDAGRAPH-DIFFUSION` |

## Stop conditions

- Stop if a measurement contradicts §3. The refutation is dated and hardware-specific,
  and a negative result is regime-dependent.
- Stop `ENG-CUDAGRAPH-DIFFUSION` at step 4 of its issue if the denoise loop turns out
  GPU-bound once device-resident. Close it the way this spec closed prefill.
- Stop any row that starts claiming throughput from §4 without naming a measured
  host-bound path first.

## Outcome

Recorded 2026-08-17. The claim is accurate about the mechanism and overstated about
priority. Its three parts land here as one no-lever, one refutation, and one unported
memory optimization, plus one structural row and one blocked row that the analysis
surfaced rather than the claim.

What was rejected and why: porting BCG as a prefill throughput lever, because our
prefill has 3.8% host idle and is 92.5% glue-GPU-bound; and scoping the diffusion
capture now, because the render does no device compute to capture.

Corrected 2026-08-18 ([#1179](https://github.com/mudler/vllm.cpp/issues/1179)). Two
repairs to what this spec recorded on 2026-08-17. First, the hand-rolled driver count
is NINE, not eight: the `ENG-CUDAGRAPH-BREAK` scoping spike found the DFlash draft
graph (`src/vllm/model_executor/models/qwen3_dflash.cpp:771,870,1038,1091,1095,1106`)
that §4 had missed. Second, `ENG-CUDAGRAPH-BREAK` was recorded here as a COVERAGE row
only, and it is also a CORRECTNESS row: `qwen3.cpp:961-986` declines its decode graph
outright while the async device-token mirror is live, on a measured degeneration
(`depth-2, graph ON FAIL, slots 1-3 degenerate`), because the fix the comment names
lives in one sibling driver as `StepDevInputs` and nowhere else. A shipped model has
already lost its decode graph to this duplication. Neither repair touches the framing
rule: the row still makes no throughput claim, and §3's refutation is unchanged.

Why the defaults are what they are: `ENG-CUDAGRAPH` stays `PARTIAL` rather than moving,
because nothing shipped. `SGLANG-BCG` is classified `INVENTORIED` rather than `OWED`,
because the capability it names is already in the mirror source and only its
construction is distinct.
