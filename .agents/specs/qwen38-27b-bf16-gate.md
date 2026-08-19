# Qwen3.8-27B (bf16): the token gate and the speed axes

**Row:** `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation`
(`.agents/model-matrix.md`) — `Qwen/Qwen3.8-27B` declares
`Qwen3_5ForConditionalGeneration`, the architecture that row owns.
**Issue:** [#915](https://github.com/mudler/vllm.cpp/issues/915)
**Related:** [#821](https://github.com/mudler/vllm.cpp/issues/821) owns the
NVFP4 / Q4_K_M arms of the same checkpoint; [#910](https://github.com/mudler/vllm.cpp/issues/910)
owns the tie-break divergence this gate ran into three times.
**Lifecycle:** `PARTIAL`
**Owner:** unassigned

## Scope

`Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, bf16, text
path. Two axes, in order: a greedy token gate against the pinned oracle, then —
only if that gate is clean — throughput, latency and memory against vLLM's
production configuration.

Out of scope: the NVFP4 and Q4_K_M arms (#821), the vision path, any fix for
#910, and advancing the parity pin.

## Why this row needs no port

[`porting-inventory.md`](../porting-inventory.md) deviation 17 records the
finding this spec measures against: `Qwen/Qwen3.8-27B` "is the already-gated
Qwen3.6-27B shape retrained — `config.json` differs in exactly one key
(`transformers_version`) and the safetensors tensor-name set is identical (1199
names, zero difference either direction)". So no loader, forward or registry
change is owed, and none was made. What was owed is evidence, and the same
entry says so: "its own token-exact gate is OWED and unrun".

## Design of the adjudication

Greedy, 7 prompts x 16 tokens, both arms on the same prompts and token counts.
The oracle capture is deterministic across 3 repeats (`deterministic: true`,
`multi_member_cells: 0`), so a strict per-prompt comparison is well-posed.

**Only the first divergence per prompt is adjudicable.** After it the two arms
are conditioned on different prefixes, so any later position compares two
different conditionings rather than one disagreement. Three prompts diverge, so
there are three numbers — not thirty-four. A position count over the whole grid
is not a quality score and is not recorded as one.

At each first divergence the two arms agree on every preceding token, so both
are conditioned on a BYTE-IDENTICAL prefix. Feeding that prefix to the pinned
oracle and reading its **fp32** next-token distribution measures exactly one
thing: how far apart the oracle itself holds its own choice and ours.

## Risks

- **A bf16 instrument cannot resolve a bf16 tie.** A `transformers` CPU probe
  reported all three as exact ties, and every runner-up gap it printed was an
  exact multiple of 0.125 — one bf16 ULP in that exponent range. It could not
  have reported anything else, so it is a secondary cross-check with that
  limitation stated, never the answer. vLLM computes logprobs from fp32 and can
  separate pairs bf16 collapses.
- **A re-decode probe assumes the prefix it needs.** Reading the distribution
  at step `d` of a fresh greedy decode is only valid if that decode reproduced
  the captured prefix. Mitigated by a second, teacher-forced probe that feeds
  the prefix as token IDs and asserts both the echoed prefix and that the
  oracle's top-1 equals the captured token.
- **A stale binary measures a tree that does not exist.** Mitigated by
  rebuilding at `origin/main` and re-running both axes on that binary.
- **A degraded build voids every number.** Mitigated by asserting the
  configure-log fast-path lines (CUTLASS, FA2, Triton AOT `sm_121a`) and
  aborting the build otherwise.

## Tests

This row changes no `src/`, `include/` or `tests/` file, so it ports no test.
`git diff origin/main..HEAD` over those three paths is empty and that emptiness
is the claim. What it adds is evidence, listed under Evidence required.

## Gates

- Token gate: every first-divergence position within `kNearTieMnats = 500` of
  the oracle's teacher-forced argmax, on the pinned oracle, or the row fails.
- Speed: recorded only if the token gate is clean. vLLM's **production**
  configuration is the denominator — never `--enforce-eager`. Same client
  (`vllm bench serve`) drives both arms.
- Both under `$HOME/gpu.lock`, clocks pinned, contention recorded per leg.

## Evidence required

- Oracle identity asserted (`vllm.__version__`, `flashinfer`) with ABORT on
  mismatch, plus a `Python.h` precondition so a missing header aborts loudly
  rather than dying inside Triton's JIT.
- Per divergence: the oracle top-2 gap in millinats, our token's rank in the
  oracle top-20, and a verdict against the band.
- The build recipe, revisions, checkpoint size, binary md5, boot id, SM clock
  and the contention actually observed.

## Stop conditions

- If any divergence is out of band, STOP: report it as a real divergence, run
  no benchmark on that checkpoint, and let no record imply it is
  baseline-ready. Do not attempt a fix.
- If the box is not quiet at a leg boundary, wait or drop the leg. A number
  taken under load is worse than no number.

## Now

`PARTIAL`, and it stays `PARTIAL` after the 2026-08-19 re-measure. The token
axis is closed and passing. The speed axis is closed only at **c4** (0.963x
throughput, 1.008x ITL).

**Our half of the c1/c8 debt is DISCHARGED.**
[#931](https://github.com/mudler/vllm.cpp/issues/931) is closed and the fix
holds on hardware: with `VT_SERVER_SSE_PING_S=0`, three reps at each
concurrency on an idle leased box completed **162 of 162** requests with
`failed=0` and zero non-empty `errors` entries on every leg. c1 output
throughput **4.4040 tok/s** (CV 0.039%), median TPOT 218.11 ms, median ITL
216.56 ms, median TTFT 883.78 ms. c8 output throughput **22.6402 tok/s**
(CV 0.205%), total token throughput **196.10 tok/s, CORRUPTED — do not quote it**
([#1355](https://github.com/mudler/vllm.cpp/issues/1355), `## Owed`), median TPOT
250.57 ms, median ITL 232.83 ms, median TTFT 3623.5 ms. Both teardowns
`TEARDOWN_VERDICT=CLEAN`.

**And read the two output-throughput figures with the same caveat, bounded rather
than dismissed.** `output_lens` is `[128]xN` on both arms, so TPOT and ITL stand.
`output_throughput` divides output tokens by a WALL, and a genuinely truncated
prompt shortens that wall, so it is biased UP: at the c1 marginal prefill cost the
missing 202 tokens are ~0.22 s of a 174.39 s wall (~0.13%), and the missing 2,080
at c8 are ~1.1-1.6 s of 271.0 s (~0.4-0.6%). Both exceed the 0.039% and 0.205% CVs
published beside them, so "unaffected" is wrong and the bias sits outside our own
stated precision. Shorter context also cheapens decode, so these are lower bounds.
The median TTFTs (883.78 ms ours, 876.4 ms vLLM) ARE comparable: our two short
prompts are the two LOWEST TTFTs in every rep, so on both arms the median falls on
a 1024-token request.

**Neither cell became a ratio, and the two halves are blocked differently.** At
c1 the pinned oracle `0.1.dev1+g555967922` also completed every request on its
production graphed configuration (`enforce_eager=False`,
`cudagraph_mode: FULL_AND_PIECEWISE`, capture sizes `[1..64]`, `FLASH_ATTN`,
read back from its own startup line): **4.2835 tok/s** (CV 0.033%), median TPOT
228.36 ms, median ITL 226.86 ms, median TTFT 876.4 ms, cold start 426 s. Both
absolutes stand as facts with their own clock blocks, and
`gpu_clock_state compare` returned `PAIRING_VERDICT=DISCARD` on all three
pairings, so **no ratio is derived — the c1 ratio is OWED, not withheld for
being unflattering**.

The refusal is about spread, not disagreement. Cross-arm: same boot
`3fd9745a-d25a-426c-ba3c-97c958a85515`, both arms at a 2489 MHz median, offset
**0.0%**. Within-run: ours 13.58/26.36/14.34%, vLLM 10.16/17.48/18.52%, against
the 5% ceiling, with `SwThermalSlowdown` in every window and
`HwSlowdown+HwThermal` in one of ours. **All six of our legs and all three of
vLLM's breached that ceiling**, c8's included (12.92-14.99%).

**At c8 the vLLM denominator is NOT MEASURABLE on this box at the recorded
configuration.** That is the c8 answer, not a gap in it, and it is a statement
about headroom and guard granularity here rather than a claim that vLLM is
defective. Detail, trajectory and arithmetic in
[`../benchmark-record.md`](../benchmark-record.md).

**What advances this row next** is a c1 pairing taken in a window the clock gate
will accept, which needs either clock pinning (unavailable in a lease, below) or
a thermally quiet window, and a c8 denominator taken somewhere with more than
6-7 GB of headroom at the recorded configuration. Neither is reachable from
`dgx:gpu0` today.

## Outcome

**Measured, token axis.** `Qwen/Qwen3.8-27B` @`1d4bf0f2`, bf16, 55,586,114,863
bytes over 18 shards, against the pinned oracle `0.23.1rc1.dev1511+g555967922`
with FlashInfer `0.6.15.post1`, greedy, 7 prompts x 16 tokens. The oracle
capture is deterministic over 3 repeats (`multi_member_cells: 0`). **4/7 prompts
STRICT 16/16**, and the three first-divergence positions adjudicate as **exact
fp32 ties**:

| Prompt | Pos | Oracle | Ours | Top-2 gap | Oracle - ours | Our rank |
|---|---:|---|---|---:|---:|---:|
| `Once upon a time,` | 2 | 1814 `" world"` | 22960 `" magical"` | 0.000 mnats | 0.000 mnats | 3 |
| `The largest planet in our solar system is` | 1 | 11 `","` | 13 `"."` | 0.000 mnats | 0.000 mnats | 2 |
| `import numpy as np` | 8 | 16309 `" matplotlib"` | 27180 `" scipy"` | 0.000 mnats | 0.000 mnats | 2 |

The teacher-forced logprobs are `-1.5257947444915771`, `-0.74363690614700317` and
`-1.4836434125900269`; the greedy re-decode reads `-1.524564266204834`,
`-0.7444034814834595` and `-1.4876692295074463`. The two conditioning paths
differ in the last few thousandths of a nat, which is the point of reading it
twice — and the GAP is exactly `0.0` under both, on both members of each tied
pair. `INTEGRITY_OK=True`: for all three, the engine echoed the supplied prefix
back unchanged and its teacher-forced top-1 equals the captured greedy token, so
prefill and incremental decode do not disagree here.

`ALL_TIES_OR_IN_BAND` against `kNearTieMnats = 500`. Every one is
[#910](https://github.com/mudler/vllm.cpp/issues/910) and nothing else: the
oracle's pick carries the LOWER token id and ours the HIGHER at a bit-identical
logprob. `Once upon a time,` is a THREE-way tie — `" world"`, `" land"` and
`" magical"` all sit at `-1.524564266204834`.

**The shared prefix is proven, not assumed.** The adjudication is only valid if
both arms are conditioned on the same bytes. Two things establish it. Our
tokenizer reproduces the oracle's prompt ids **7/7 exactly**, checked directly
through `examples/tokenize` against `capture.json`'s `prompt_ids` — which the
token gate itself could not show, because our server's `echo` does not return
prompt tokens. And the generated ids agree up to the divergence by construction
of "first divergence".

**Rejected as the answer, kept as a cross-check.** A `transformers` CPU probe
called all three exact ties, and its own output refutes it as evidence: every
runner-up gap it printed was a multiple of **0.125**, one bf16 ULP in that
exponent range. An instrument that cannot resolve below one ULP cannot report
anything but a tie, so agreement with it is not confirmation. It is recorded
under the existing [transformers](../oracles/transformers.md) pin with that
limitation attached.

**Read twice, on the oracle's fp32 logprobs.** A greedy re-decode reads the
distribution at step `d` of the oracle's own deterministic decode. An
independent teacher-forced probe feeds the prefix as token IDs, then asserts
both that the engine echoed that prefix back and that the oracle's top-1 equals
the captured token — so a decode that had silently wandered before `d` cannot
pass as a valid conditioning. Both abort on an oracle-identity mismatch
(`vllm.__version__`, `flashinfer`) and on a missing `Python.h`, which otherwise
surfaces as an opaque failure inside Triton's JIT.

**The result does not depend on a stale tree.** The first run used a binary
built at `4a183b731`, 13 commits behind `main`, and `qwen3_5.cpp`,
`qwen3_5_dense.cpp` and `qwen3_5_weights.cpp` had all changed in between — so it
bound nothing. Rebuilt at `11a42dc4c` with the fast path asserted (CUTLASS, FA2,
Triton AOT `sm_121a`, `sm_121a` baked into the library) and re-run: **the same
4/7, the same three positions, the same tokens**. The intervening changes are
empirically inert on this path rather than assumed to be.

**Measured, speed axis, and one of three cells is all it supports.** Against
vLLM's production graphed configuration (no `--enforce-eager`,
`--language-model-only` on both arms, same `vllm bench serve` client, one
`flock`, clocks pinned to a flat 2184 MHz, one boot id), random 1024-in /
128-out, 6 prompts per concurrency unit, paired reps interleaved ours/vLLM:

| Axis | c1 | c4 | c8 |
|---|---:|---:|---:|
| ours / vLLM completed | 5,5,5 / 6,6,6 of 6 | 24x3 / 24x3 of 24 | 36,37,36 / 48x3 of 48 |
| Output throughput | WITHHELD | **0.963x** | WITHHELD |
| Median ITL | 1.013x | 1.008x | 1.021x |
| Median TPOT | 1.014x | 0.980x | 0.925x |
| Median TTFT | 0.733x | 0.881x | 1.268x |

Plus cold start **53 s vs 780 s = 14.7x** and host memory after warmup
**42.5 vs 110.1 GiB = 2.59x**, the latter caveated because vLLM's figure is set
by `--gpu-memory-utilization 0.85` pre-reserving KV rather than by the model.

**Two cells are WITHHELD, and that is the finding.** Our server failed 1 of 6
requests at c1 in all three reps and 12/11/12 of 48 at c8, where vLLM failed none
in nine legs on the identical workload from the identical client
([#931](https://github.com/mudler/vllm.cpp/issues/931)). `output_throughput`
divides tokens by a wall duration that still contains the dead request, so the
c1 cell reads 0.677x while median TPOT in the SAME file reads 1.014x in our
favour. Recording 0.677x would have published a 32% deficit contradicted by the
evidence beside it. No harness here asserted `failed == 0` before summarising a
ratio, which is the gap that let this become quotable in the first place.

The failures are silent server-side: a read-only sidecar sampled our server's
log live during a failing leg and it is 27 lines, all startup, no error, no
request line, no rejection. So the cause is not named yet, and a controlled
reproduction recording HTTP status and exception class is the next step.

**What was NOT established.** Nothing about the vision path on this checkpoint,
nothing about its NVFP4 or Q4_K_M arms (#821), no claim that #910 is fixed (this
is the second checkpoint to be costed by it), and no throughput number at c1 or
c8 until #931 closes. Concurrencies above 8 were not run.

## Owed

- [#915](https://github.com/mudler/vllm.cpp/issues/915) stays OPEN. Our arm's
  c1/c8 debt is discharged; the vLLM half is not. The c1 ratio is refused by the
  clock gate and the c8 denominator does not exist, so the row's own question —
  what our speed is against vLLM's production configuration at c1 and c8 — is
  still unanswered.
- Whether the HOST rebooted or only the k3s pod was lost when the c8 vLLM worker
  died on 2026-08-19. The artifacts cannot distinguish them. Settle it by
  reading `/proc/sys/kernel/random/boot_id` inside any later `dgx:gpu0` job and
  comparing it against `3fd9745a-d25a-426c-ba3c-97c958a85515`.
- [#1355](https://github.com/mudler/vllm.cpp/issues/1355), the prompt-token
  divergence found in this campaign's raw result files: our server reported
  5,942 prompt tokens where vLLM reported 6,144 for the identical
  client-generated prompts, 19 of 48 short at c8. Whether we under-report
  `usage.prompt_tokens` or actually truncate the prompt is not decidable from
  these artifacts. **It corrupts `total_token_throughput` on both legs** — c1
  38.4776 tok/s and c8 196.10 tok/s are computed over 5,942 and 47,072 input
  tokens where the workload intends 6,144 and 49,152 — and it biases
  `output_throughput` up by more than that figure's own CV; `## Now` carries the
  bound. Quoting either total-token figure, or setting our 38.4776 beside vLLM's
  38.5516, compares two different workloads.
- [#1365](https://github.com/mudler/vllm.cpp/issues/1365), a reproducible ~4 s
  TTFT outlier on request 3 of every c1 leg of ours, which the oracle does not
  have: index 2 reads 3.981 / 3.924 / 4.006 / 3.955 s across the warmup leg and
  all three reps against 0.73-0.93 s for every other request, four legs of four,
  on a 1024-token prompt like requests 4, 5 and 6. Nothing published is wrong,
  because the median of six averages ranks three and four and the outlier never
  occupies either; it moves the MEAN (ours 1347.6-1372.6 ms against the oracle's
  873.3-900.2 ms) and costs ~3.1 s of the 174.39 s c1 wall. The cause is not
  chased here.
- The checkpoint size disagrees between records: this spec's `## Outcome` says
  55,586,114,863 bytes and the campaign's `NOTES.txt` says 55,586,040,114, a
  difference of 74,749 bytes. The 2026-08-19 run DID re-derive it, and it agrees
  with `NOTES.txt`: `out/bench-20260819T035148Z/job.log:47,49` print
  `CKPT_SRC_BYTES=55586040114` and `CKPT_DST_BYTES=55586040114` over the staged
  tree that then served every leg. What is unresolved is WHY the `## Outcome`
  figure differs, which the artifacts cannot settle.
