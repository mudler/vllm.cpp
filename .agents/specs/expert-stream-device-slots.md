# `ENG-EXPERT-STREAM-DEVICE` — a device-reachable destination for streamed expert slices

Issue [#1124](https://github.com/mudler/vllm.cpp/issues/1124). Owning row
`ENG-EXPERT-STREAM-DEVICE` ([engine-matrix.md](../engine-matrix.md), KV cache and
memory). Parent row `ENG-EXPERT-STREAM`, spec
[expert-streaming.md](expert-streaming.md), which owns the streaming MECHANISM —
the cache, the streamer, the `pread` filler and the host store — and lists this
capability under its `## Owed`. This row owns only WHERE a slice lives and WHICH
platform may read it.

## Now

`ACTIVE`. **`--device cuda` DECODES this checkpoint on a GB10, and the
correctness gate that would let us publish a number does not pass.** W0e ran on
2026-08-19 inside one `rc hold` on `dgx:gpu0` at source `9c783a8be`.

* **G0-LIVE: PASS.** 32/32 steps where seven previous attempts produced ZERO;
  decode-phase `exhausted` delta **0** (6077 at step 1 and at step 32; the total
  is the structural prefill number this spec predicted); `W0E_DOCKER_RC=0`, no
  guard trip, peak RSS **97.75 GiB** with swap untouched.
* **G0-CORRECT: FAIL as declared. This bullet read the failure as a near-tie,
  and W0g has since shown that reading to be the SYMPTOM. Read the W0g bullet
  below before the three findings under this one.** The 32 ids match the CPU
  arm for six tokens and diverge at the seventh — `...,264,3177,7172,...` on
  CPU against `...,264,3177,303,...` on CUDA. Both continuations are coherent. Three things
  were then established rather than assumed:
  1. **Is it W0f? The two grounds first offered here could not answer that, and
     they are withdrawn.** The CPU arm was re-run on the SAME binary and the
     SAME lease and reproduced the recorded answer byte for byte, and the
     instrument counted `w0f-alias` calls **0** on that arm. Neither
     discriminates. `ResidentWeight` takes an `is_cpu()` early return
     (the `is_cpu()` branch of `qwen3_5.cpp`'s `ResidentWeight`) roughly ninety lines above the alias branch (the `host_memory_is_device_addressable()` branch of the same function; no line number, because this change moves it),
     so a zero count on the CPU arm is true by construction for every possible
     state of W0f, correct or corrupt; and "the CPU arm reproduces its own
     reference" constrains only the arm W0f cannot reach. Both show the branch
     is platform-gated. Neither separates "the arms' GEMM arithmetic differs"
     from "W0f moved a logit".
  1b. **The discriminating experiment, RUN, and it clears W0f.** The only thing
     a consumer can notice about the substitution is the pointer, so the
     question is whether cuBLASLt answers differently for a 256-aligned HOST
     block than for a `cudaMalloc` one. Measured on `thor:gpu0` (NVIDIA Thor
     `sm_110`, driver 13020, cuBLASLt 130101), which answers this branch's own
     predicate TRUE (`PageableMemoryAccess = 1`, `Integrated = 1`) and is
     therefore in the population W0f serves. Six checkpoint shapes
     (`embedding_length = 8192`, M = 1, 5, 32) crossed with BOTH formulations
     the dense path issues (row-major NN, weight as B; column-major TN, weight
     as A) — **12 measurements, `PROBE_FAILURES=0`**. The heuristic returned an
     **identical selection 12/12** on a repeated call, 12/12 with the 256
     promise stated explicitly, and 12/12 with it weakened to 16; and
     `cublasLtMatmul` over the same bytes gave **bit-exact output 12/12**, zero
     differing elements, every status `SUCCESS`. Five DIFFERENT algo
     configurations appear across the six shapes, so the instrument
     discriminates. The structural reason needs no lease:
     `cublasLtMatmulAlgoGetHeuristic` takes no operand pointers, so alignment
     reaches it only through a preference this tree never sets. **Identical algo
     and bit-exact output, so W0f cannot move a logit.**
  1c. **The GB10 leg RAN too, on the target silicon.** `rc` job
     `7c7a05e9-be87-48f4-94ae-1bbe0340f063` on `dgx:gpu0`, 2026-08-19 17:47 UTC,
     `NVIDIA GB10 sm_121`, driver 580.173.02, cuBLASLt 130101, the predicate
     re-derived in the job's own output (`pageableMemoryAccess=1 integrated=1`).
     Same six shapes, same two formulations, **12 measurements,
     `PROBE_EXIT=0`, `PROBE_FAILURES=0`**: repeated heuristic call identical
     12/12, unset preference equal to the 256 default 12/12, the promise weakened
     to 16 moving nothing 12/12, and `cublasLtMatmul` output **bit-exact 12/12**
     (`differing=0`) between a `cudaMalloc` operand and a 256-aligned host block.
     At least five distinct algo configurations appear across the twelve, and
     they differ from Thor's, so the heuristic was re-resolved rather than
     replayed. **The attribution is therefore measured on the silicon the token
     gate ran on: the alias does not cause the divergence.** It does NOT say what
     does; naming the first operation that differs is carried under `## Owed`.
  2. **The two arms rank the same two candidates.** An instrumented CPU run
     printing the top-2 logits per step shows that at the divergent step
     (`lp_call=7`) the CPU arm's top-1 is `7172` at 18.779411 and its **top-2 is
     `303` at 18.514702** — `303` being exactly the token CUDA emitted. The
     **margin is 0.264709 logits**, 1.4 % of the winning logit.
  3. **This decode is full of ties that narrow.** `lp_call=9` has a margin of
     **0.022802**, about 0.1 %. A greedy path this finely balanced flips on any
     arithmetic difference, and the two arms run genuinely different GEMM
     kernels.
  So the declared gate fails and the wave stops, which is correct. What the
  failure MEANS is a different question. This bullet used to answer it with
  "the arms agree about the distribution and disagree about a coin flip", and
  W0g falsifies that answer: the two arms already select different EXPERTS in
  the first MoE block of the first forward, so the sampler is not comparing the
  same distribution and the margins above measure an input that already
  differs. The three findings above stay true as measurements. Only the
  conclusion drawn from them is withdrawn.
  Whether a token-exact cross-arm gate is the right instrument for a path with
  no oracle is still a decision for the operator, not something this row may
  assume.
* **W0g: the divergence is expert ROUTING from the FIRST MoE block, and its
  cause is still not identified.** Two runs on `dgx:gpu0` at source `cffe59b`,
  on 2026-08-20 and 2026-08-21, with the page cache dropped on the HOST before
  each arm and the weights on local NVMe. Every number is in
  [`../benchmark-record.md`](../benchmark-record.md) under
  `ENG-EXPERT-STREAM-DEVICE W0g` and is not repeated here.
  * **What is now EXCLUDED, measured rather than argued.** The router GATE
    weights: an FNV fingerprint of the gate is identical on all 184 dump
    records of both arms, so the arms do not load different router weights.
    The W0f host alias was already excluded on this silicon by the
    algo-identity probe. **The top-k implementations are a WEAKER exclusion and
    are written with their denominator**: the selected set was re-derived
    offline from each arm's OWN logits by a plain lowest-index-wins rank, and 0
    of 5 token-rows deviate on either arm out of the 552 each dump holds, which
    is 0.91 % and is consistent with record 0 alone. That is a sample result. It
    does not license "both top-k implementations are correct", and re-deriving
    over all 552 rows is carried under `## Owed`.
  * **What the dump shows instead.** The arms differ at record 0, the first MoE
    block, in the router GEMM INPUT rather than in anything the router does
    with it, and the difference compounds smoothly up the stack. The observed
    flip is an exact bf16 tie that one arm sees and the other does not. Run A's
    step-1 miss counters corroborate this from a counter neither dump touches:
    at step 1 the slot cache is empty, so the miss count IS the number of
    distinct expert slices requested, and the arms are 27 slices apart eight
    tokens before any emitted token differs.
  * **The EMBEDDING OUTPUT is bit-identical, so the weights side is closed at
    both ends.** Run C dumped the hidden state before any GEMM, norm or
    attention touched it, one prefill per arm. Both dumps are 81,940 bytes with
    the same sha256 prefix `3f81114a87a0774e84086fe4`, and an element-wise
    comparison of all 40,960 bf16 values reports **0 differences**, with
    non-emptiness and equal element counts asserted first. The two arms read the
    same embedding table and dequantize it identically. With Run B's matching
    router gate fingerprints that closes the weights at the bottom and at the
    top of the stack, and it puts the divergence in the COMPUTE after the
    embedding, inside block 0's attention and dense path.
  * **What is NOT excluded, and this row may not present the case as closed.**
    No fingerprint was taken of the expert projections, the attention weights,
    or the norms, and the exoneration does NOT widen past the embedding table.
    The evidence is CONSISTENT with bf16 reduction-order accumulation across two
    genuinely different GEMM kernels, and consistency is not attribution. Run C
    removes one candidate. It does not identify a cause and it does not make the
    divergence benign: the 13.4 % accumulated divergence by block 91 and the
    degenerate CUDA continuation both stand.
  * **The CUDA continuation degenerates, and that is a reason not to ratify.**
    The two arms agree for 8 tokens, then the CUDA text falls into a mechanical
    recursion in which each sentence re-uses the previous object. A coin flip
    between two equally good tokens does not do that, so this reads as a wrong
    distribution rather than as admissible non-determinism.
  * **No speed result follows, and none is claimed.** Run A's decode medians
    are recorded in the benchmark record with their three qualifications: one
    stalled step inside each arm's steady window, an application clock pin of
    2418 MHz against a 3003 MHz maximum, and G0-CORRECT still failing. No ratio
    is written anywhere and neither median may reach `docs/BENCHMARKS.md` as a
    speed claim.
* **W0h: the experiment that decides whether the CUDA arm is WORSE or only
  DIFFERENT. SPEC ONLY; nothing has run.** W0g excluded three causes and named
  none, and it left the question G0-CORRECT actually turns on unanswered: every
  comparison so far is arm-against-arm with no oracle, so "they differ" cannot
  say which arm is wrong. W0h feeds BOTH arms the identical token sequence
  (teacher forcing, through the ABI logits processor) and measures the negative
  log likelihood each arm assigns to held-out text, which is a quality statement
  and not a difference statement. Its decision rule is pre-registered, its
  oracle arm is `llama-cpp-unsloth` (`gateable = no`, #933), and it produces no
  speed claim. Design, the rule, the corpus and the stop conditions:
  [cuda-arm-degradation-experiment.md](cuda-arm-degradation-experiment.md),
  issue [#1736](https://github.com/mudler/vllm.cpp/issues/1736). **That file is
  the binding copy of the rule and this bullet does not restate it**, because a
  threshold written twice can be moved in one place after a run.
* **G0-SPEED: VOID, by this row's own stop condition.** It was measured over
  the 31 DECODE steps of each arm (step 1 is prefill and is excluded),
  interleaved on one lease: CUDA median **4.598 s/token** (min 3.012, max
  126.456), CPU median **9.055 s/token** (min 7.857, max 23.174). Both maxima
  are the first decode step, with the slot cache cold. It is NOT claimed,
  because a speed number behind a failing correctness gate is exactly the shape
  #912 F1 was. **No ratio of the two medians is written here or in
  `.agents/benchmark-record.md`, deliberately**: it would rest on a token
  comparison that FAILED, and a disowned figure written out in digits is how a
  number this repository never measured becomes one it is quoted as having
  measured. Both medians are above for anyone entitled to the quotient. Second
  caution: this CPU arm is faster than the 11.05 s/token previously recorded at
  4000 slots, so the two are not the same measurement and must not be mixed.

**What W0f did, measured rather than inferred, and read at a stated point.** The
instrument added for this run counts **60.793 GiB** of dense weight aliased
instead of duplicated into device memory, against **~9.2 GiB** that declined
(misaligned GGUF borrows) and still stages. **The qualifier is part of the
number**: those are the FIRST-FORWARD totals, taken at the point re-homing
plateaus, call **1361**. The counters are per CALL and there is no memo on the
alias branch, so they keep growing at roughly 70 GiB per decode step; quoted
without the qualifier the same figure is a traffic count and not a residency
measurement. The residency claim it supports is corroborated independently by
peak RSS **97.75 GiB** against a load that previously reached 61.20 GiB and then
exhausted the box. That is the whole difference between zero decode steps and 32.

W0a HAS RUN as a standalone probe on `dgx:gpu0` and answered
`W0A_VERDICT=PAGEABLE_OK`, so the stop condition that would have returned this
row `NEEDS_DECISION` never fired. The completed decode above corroborates it from
the other end: the load succeeds only when
`host_memory_is_device_addressable()` answers true, because W0d's conditional
refusal is keyed on it.

The three gates, reported one result each. Every row states what it read at
`95883dcae` where W0f moved it, so the two runs in `## Evidence` are not confused
for one:

| Gate | Result |
|---|---|
| **G0-CORRECT** | **FAIL, and it now has a CUDA side to fail on.** At source `95883dcae` the entry read "NO CUDA SIDE", because that arm emitted zero tokens. With W0f it emits 32, and they diverge from the CPU arm at step 7 on a MEASURED near-tie: the CPU arm's own runner-up is the token CUDA emitted, 1.4% behind, and one step later the margin is 0.1%. The CPU side remains byte-identical across four runs and two slot counts (32 ids, listed in `## Evidence`). The alias is measured ON GB10 not to be the cause. **W0g then moved the failure upstream of the sampler**: at source `cffe59b` the two arms already select different EXPERTS in the FIRST MoE block of the FIRST forward, eight tokens before any emitted token differs, so the near-tie is the symptom. The router gate weights and the embedding table are now excluded as well, the second because Run C found the two arms' embedding output bit-identical. The top-k implementations are excluded only on 5 of the 552 token-rows the dumps hold, 0.91 %, and that sample may not be quoted as a property of the implementations. That closes the WEIGHTS at both ends of the stack and puts the divergence in the compute inside block 0. WHAT the cause is remains open under `## Owed`. |
| **G0-LIVE** | **PASS on both arms.** CPU: `steps=32`, `forced=0`, decode-phase `exhausted` delta **0** at both 4000 and 8000 slots. CUDA with W0f: `steps=32`, decode-phase `exhausted` delta **0**, peak RSS 97.75 GiB of a 119.631 GiB box, swap untouched, container exit 0. At `95883dcae` this read "NOT REACHED on CUDA", because no step boundary was ever crossed. |
| **G0-SPEED** | **VOID, and no ratio is published.** The CPU denominator is measured: steady decode **11.05 s/token**, rep 2's median over 29 samples (min 9.43, max 13.25) at 4000 slots, rep 1's median 11.22, the two reps agreeing within 1.5%. A CUDA number exists now and is recorded in `../benchmark-record.md` for the record only, because this row's own stop condition VOIDS a speed result behind a failing correctness gate. No ratio may be inferred from the two. |

What that means precisely, because both "W0 landed" and "W0 failed" would
misstate it:

* **W0b — the predicate.** `Platform::host_memory_is_device_addressable()`,
  base false, CUDA from `cudaDevAttrPageableMemoryAccess AND
  cudaDevAttrIntegrated` probed at registration, ROCm from the pair its backend
  already probes. Gated in `test_platform` (14 cases / 114 assertions), which
  pins the DEFAULT and the INDEPENDENCE from all four neighbouring predicates.
  **The CONJUNCTION itself was gated by nothing until #1378**, and that is a
  measurement rather than a suspicion: both attributes read 1 on the only CUDA
  box this project can reach, so `test_platform`'s implication assertion and its
  board value both survive the deletion of either term, and the fresh review of
  #1377 ran both deletions and got GREEN twice. The decision is now
  `HostMemoryIsDeviceAddressableFromAttrs(pageable, integrated)`, a pure function
  in the always-compiled `platforms/platform.cpp` that `cuda.cpp`'s registrar
  calls, gated over all four attribute pairs on a tier with no CUDA device.
  Both deletions were re-run against the extracted function and both are now
  RED: 14 cases with 1 failed and 114 assertions with 2 failed, exit status 1,
  compile status 0, no ENOSPC in the build log, `git diff --stat` showing the
  edit applied, and `platform.cpp` restored byte-identical by sha256 afterwards.
  What is STILL owed is the CUDA registrar's own probe assembly — the
  `cudaDeviceGetAttribute` calls and the failure defaults around them — which
  compiles only in a CUDA build; that row is under `## Owed`, and this bullet no
  longer claims more than it has.
* **W0c — the slot arm without the tower.** The seam takes the slot arm on
  `is_cpu() || host_memory_is_device_addressable()`, builds the slice tensor
  through a new `KqHostSliceView` instead of `ResidentWeight` (whose staging
  branch is the 1.1875 GiB allocation #1123 died on), reads the tower in place
  when the cache cannot serve a slice on a non-CPU platform, and refuses by
  name if a claimed tower ever reaches device staging. Gated in the new
  `test_expert_stream_device_slot` (5 cases / **45** assertions after W0f
  re-stated its "served normally" case; the count is read off the binary's own
  last `test cases:` line, not off what this row expected) over a fake staging,
  host-addressable platform, because no CPU tier can register a real one.
* **W0d — the conditional refusal.** The fit bound gained a
  `StreamedExpertLane` input; the loader fills it when the platform stages, can
  read host slots, the resolved model's factory declares
  `streams_routed_experts`, the config says streaming is on, and **this file's
  expert towers actually take a keep residency**. Gated in
  `test_gguf_device_fit` (17 cases / 130) for the arithmetic and the residency
  predicate, and `test_gguf_device_fit_reach` (14 cases / 66) for the production
  reach. **The lane-off bound is byte-identical**, asserted three ways against
  literal values.
  **The architecture term was a review finding, not a design choice made up
  front.** The first draft keyed the lane on the tensor-name suffix alone, and
  `_exps.weight` is what a llama.cpp MoE export writes for every MoE family it
  converts — `deepseek_v4_weights.cpp` and `laguna_weights.cpp` write the
  identical name and neither model composes `RunMoeBlock`. That dropped their
  towers from the bound and charged an arena nothing allocates, which deletes a
  CORRECT refusal, unlike the two over-counts above, which only over-refuse. The
  term is a declared capability on `ModelFactory` rather than a name list in the
  loader, so it lives beside the forward that implements it and a new
  architecture inherits false.
  **The RESIDENCY term is the same finding one level down, and it was found the
  same way** ([#1378](https://github.com/mudler/vllm.cpp/issues/1378), fresh
  review of #1377). All four terms above are properties of the DEVICE and the
  ARCHITECTURE, and a `qwen35moe` GGUF satisfies every one of them while still
  staging every tower: `LoadExpertsOrNvfp4` routes the same `_exps.weight`
  tensors to `expert_*_fp4` (`kNvfp4Fp4`) or to `expert_*` (`kExpandBf16`)
  whenever the residency is not a keep residency, and only the `expert_*_kq` arm
  reaches `KqExpertSlice`. So `VT_GGUF_KEEP_QUANT=0` -- a documented, supported
  opt-out -- and an NVFP4 GGUF each turned the lane on, dropped 335.62 GiB of
  towers from the bound and DELETED the #1123 refusal, which is the unsafe
  direction and puts back the 26-minute load and the `cudaMalloc: out of memory`
  first forward. The W0c `expert_streamed` tripwire does not catch it either,
  because a tower on those arms is never claimed. The fifth term is
  `GgufExpertTowersReachSlotLane`, which asks the model loader's OWN routing
  function (`PeekRoute`, promoted to `gguf_keep_quant.h` so one decision has one
  description) about this file under this process's policy. It is all-or-nothing
  in the safe direction: one tower that would be staged keeps the WHOLE bound, so
  it can over-refuse -- which `VT_DEVICE_WEIGHT_BUDGET_BYTES` releases -- and can
  never under-refuse.
  **The accepted set has TWO routes, and the second one was gated by nothing
  until the second review of #1377.** `GgufExpertTowersReachSlotLane` accepts
  `kKeepQuant` OR `kKeepF16`, which is read off `LoadExpertsOrNvfp4`: everything
  that is neither fp4 nor expand goes to `LoadExpertsStackedKq`, whose `VT_CHECK`
  names both, and `KqExpertSlice` sizes a row with `vt::RowSizeBytes` and never
  looks at the dtype. But every fixture in the tree stored its towers in Q8_0,
  F32 or NVFP4, so narrowing the accept to `kKeepQuant` alone left
  `test_gguf_device_fit` at 16/16 and `test_gguf_device_fit_reach` at 14/14 --
  the same "a load-bearing term no reachable input falsifies" shape the residency
  term itself was filed for. `BuildGgufWithF16ExpertTower` closes it with an F16
  (ggml type 1) tower, which `KeepQuantDType` can never accept and which
  `RouteGgufTensor` step 2 routes `kKeepF16`. Mutation M4 (narrow the accept to
  `kKeepQuant`) was re-run on both sides of that fixture: compile status 0 both
  times, no ENOSPC in either build log, `git diff --stat` showing 1 file / 1
  insertion / 1 deletion, GREEN against the pre-fixture file at 16 cases / 117
  assertions and exit status 0, RED against the current file at 17 cases with 1
  failed and 130 assertions with 1 failed and exit status 1, and both files
  restored byte-identical by sha256.
* **W0f — the dense half, and the reason W0 still produced no token.** With
  W0b-W0d in the tree the checkpoint LOADS on `--device cuda` and then exhausts
  the box inside the first forward, zero decode steps over seven attempts
  (issue [#1299](https://github.com/mudler/vllm.cpp/issues/1299)). The lane was
  doing its job; the DENSE weights were resident twice, once as the host
  `OwnedTensor` and once as `ResidentWeight`'s device staging copy, and on a part
  where device memory IS host memory that doubling is what runs it out. W0f gives
  `ResidentWeight` the same branch W0c gave `KqExpertSlice`, on the same probed
  predicate. Gated in the new `test_resident_weight_host_addressable`
  (**12 cases / 71 assertions**) over the same fake staging, host-addressable
  platform, plus one defect the work uncovered and fixed in flow
  ([#1320](https://github.com/mudler/vllm.cpp/issues/1320)) and three a fresh
  review found: the `d_dev_f32` disjunct in the dense host-mirror release, the
  refusal that fired above the device-copy memo, and the source-page release that
  repeated once per forward step. The count is quoted from the binary's own last
  `test cases:` line, not from the number this row expected.
* **W0a — the probe. RUN, and it answered the question W0b rests on.** On
  `dgx:gpu0` inside an `rc` lease: `cudaDevAttrPageableMemoryAccess = 1` and
  `cudaDevAttrIntegrated = 1`, which is exactly the pair `CudaPlatform`
  conjoins, so the predicate answers TRUE on a GB10. The probe did not stop at
  the attribute, because an attribute is a claim and this row needed the
  behaviour: a kernel READ AND WROTE a 2,490,368-byte slot — one real expert
  slice — living in plain `std::vector<uint8_t>` storage, correctly, which is
  the exact access `Qwen35ExpertStream`'s arena will serve. Measured bandwidth
  over that access was 2.06-2.28x, recorded as the range across the reps rather
  than as a single figure. Verdict token `W0A_VERDICT=PAGEABLE_OK`.
  **The stop condition therefore did not fire.** `PageableMemoryAccess == 0`
  would have made W0b wrong as designed and returned the row `NEEDS_DECISION`
  in favour of a `cudaHostAlloc` arena; W0b-W0d were written against that risk
  knowingly, and the risk has now resolved in their favour. Two of the four
  attributes the port map names — `...UsesHostPageTables` and
  `ConcurrentManagedAccess` — are not carried here, because the verdict turns on
  the two that are and inventing the other two would be worse than omitting
  them.
* **W0e — the measurement. RAN TWICE, on two trees, and the pair is the
  result.** The first run, on one `rc hold` on `dgx:gpu0` at source `95883dcae`
  (W0f's parent), produced a reproduced CPU figure of **11.05 s/token at 4000
  slots**, a CUDA load that works, and a CUDA arm that generates nothing. The
  second, at source `9c783a8be` with W0f in the tree, produced **32/32 CUDA
  decode steps**: G0-LIVE **PASS**, G0-CORRECT **FAIL**, read at the time as a
  measured near-tie and since reframed by W0g as the SYMPTOM of a divergence
  upstream of the sampler,
  G0-SPEED **VOID** by this row's own stop condition. `## Evidence` and
  [`../benchmark-record.md`](../benchmark-record.md) carry both, in two sections
  that must not be mixed: 11.05 s/token is the standing CPU number and the second
  run's CPU column is a same-lease control rather than a second attempt at it.

Today `--device cuda` on `Qwen3.8-2.4T-A95B UD-Q1_0` **loads and decodes**. The
load is what W0b-W0d were for; at `95883dcae` it then died in the first forward,
and the cause was the DENSE half of the model rather than the expert lane -- the
non-expert weights were resident twice on a unified part, and a 0.15 GiB slot
arena failed in exactly the place an 18.55 GiB one did. That was
[#1299](https://github.com/mudler/vllm.cpp/issues/1299), and W0f is the fix:
32/32 steps at peak RSS 97.75 GiB.

**The developer's target is a GPU FIGURE, and this row still cannot publish
one.** G0-CORRECT fails on a divergence W0g locates in expert ROUTING at the
first MoE block, so G0-SPEED is VOID by this row's own stop condition. A decode
number exists in `../benchmark-record.md` for the record; it is not a result and
no ratio may be inferred from it.

**The public pages now agree with this row**
([#1442](https://github.com/mudler/vllm.cpp/issues/1442)). Both carried the
`ENG-EXPERT-STREAM` row's VOID after W0e replaced it, and they carried it
differently, so the repair is stated per file rather than as one claim about
two.

* `docs/BENCHMARKS.md:8` read "Streaming-ON decode **VOID** (#912 F1);
  re-measure owed". It named no cause, and line 9, the row directly under it,
  already recorded the replacement. It now reads "Streaming-ON decode **VOID**
  (#912 F1); re-measured LIVE by `ENG-EXPERT-STREAM-DEVICE`", which names the
  row instead of its position, because a positional pointer inside a 253-row
  table breaks the moment a row is inserted above it and no checker validates
  one.
* `docs/STATUS.md:129` read "Streaming lands but its decode figure is VOID: the
  step clock had no caller, so the cache died in token 3". It said nothing about
  a re-measure being owed, and its replacement is not adjacent: that figure sits
  at `docs/STATUS.md:167`, 38 rows further down the SAME table, which spans
  lines 126 to 190 unbroken. It keeps the cause verbatim and gains the same
  named pointer.

Neither cell repeats 11.05 s/token. `docs/BENCHMARKS.md:9` and
`docs/STATUS.md:167` already carry it, and one number stated in two keyed rows
is what lets the two drift apart. W0e's landing owed that write to the row it
superseded and never made it. No gate can catch the class: the rows are keyed on
different IDs, so `check-public-doc-tables.py` sees two well-formed rows and
`check-agent-record.py` sees two individually consistent lifecycle states, and
no checker here compares a claim in one keyed row against a claim in another.

## Scope

**In scope.**

* **W0 — the integrated/unified path.** Let a platform whose host memory is
  device-addressable read expert slices out of the existing host slot store, so
  `--device cuda` serves this checkpoint on a GB10 instead of refusing. Keyed on
  a PROBED property, never on a device name, an architecture string or a
  compute capability.
* **W1 — the device slot store.** A second production `ExpertSlotStore` whose
  slots are device allocations, with the fill contract that makes filling one
  possible at all.
* **W2 — the device-capable read.** A virtual read on `ExpertSlotStore`, so the
  seam `Qwen35ExpertStream::Slice` reads through stops being the concrete
  `HostExpertSlotStore`.

**Out of scope, and owed elsewhere.** Asynchronous prefetch and miss/compute
overlap (`ENG-EXPERT-STREAM` W6, conditional on a measurement). Composing
streaming with the grouped keep-quant MoE path. A Windows filler. A zero-copy
device DMA filler (GPUDirect Storage / `cuFile`) — see the W1 design note, which
records why a staging bounce is chosen instead and what would justify replacing
it. Families other than Qwen3.5 MoE. The KV/activation term of the load-time fit
bound, which `KV-WARMUP-PROFILE` owns.

**Verdict on issue #1124's piece 3.** #1124 lists "the filler is `pread`-into-host"
as a fourth independent piece. It is not independent and it is not a wave: a
device store cannot be filled AT ALL through today's contract, because
`ExpertSlotStore::SlotForWrite` returns a pointer that `ExpertStreamer::EnsureFile`
hands straight to `::pread`, and `vt::Backend::DeviceMemoryIsHostAddressable`
is false for CUDA, so that pointer is not host-writable. W1 therefore carries a
fill-contract change or W1 cannot land. Planning it as a third wave after W1
would plan a wave that deadlocks its predecessor. It is folded into W1 below.

## Upstream chain

**There is no oracle for this row, and that is a measured statement rather than
a search that came up empty.** Pinned vLLM `555967922` has inference-time expert
paging nowhere: `vllm/model_executor/offloader/uva.py:21` is a CPU-blanket UVA
offloader over whole parameters, and `vllm/model_executor/offloader/prefetch.py:557-560`
is cpu-only. The secondary-oracle table does not rescue it either — llama.cpp's
`-ot` / `-ncmoe` (`common/arg.cpp:2451-2478` @ `237ad9b96`) moves expert COMPUTE
to the host, which is `ENG-HYBRID-PLACEMENT`'s inverse design and not a slot
store. `docs/BENCHMARKS.md` already records the parent row as "correct answer;
no oracle runs this".

So the reference for correctness is **our own CPU arm on the same checkpoint and
the same binary**, which is the shape `ENG-EXPERT-STREAM` already gates on, and
every gate below is stated against it. Nothing here is mirrored from vLLM, and
nothing here may claim to be.

The one upstream surface this row DOES mirror is the platform seam it extends:
`vllm/platforms/interface.py:914` + `vllm/platforms/cuda.py:675`
`is_integrated_gpu`, already ported as `platforms::Platform::is_integrated_gpu`
and recorded in that header as "unwired today". W0 is what wires that family of
predicates, and it adds one sibling to it rather than a bespoke test at the call
site — the same discipline `needs_weight_staging` and `supports_fa2_attention`
were hoisted under (accelerator-seam audit §12.3).

## Our baseline

Every number below is re-verified against the tree at `fe24a3029` and against
the census recorded in [expert-streaming.md](expert-streaming.md) (both GGUF
tensor tables at revision `567d3e6ac26c5474b18311e619c04350fb9a5556`, 1702
records parsed against 1702 declared).

| Quantity | Value |
|---|---|
| one IQ1_XXXS expert tower `[E*N,K]`, E=512 | 1,275,068,416 B = 1.1875 GiB |
| all 279 `*_exps` towers | 360,374,599,680 B = **335.62 GiB** |
| whole tensor table (the fit bound's scope) | 397,245,341,184 B = 369.96 GiB |
| non-expert remainder | 36,870,741,504 B = 34.34 GiB |
| ... minus the MTP / `nextn` block a default load never stages | 27,930,252,800 B = **26.01 GiB** |
| one expert slice = one slot | 2,490,368 B = 2.375 MiB |
| slices per decoded token (93 layers x 10 experts x 3 matrices) | 2790, i.e. 6,948,126,720 B = **6.95 GB (6.47 GiB) per token** |
| device pool, `cudaMemGetInfo` total on `dgx:gpu0` | 128,452,956,160 B = 119.631 GiB, exactly `/proc/meminfo MemTotal` x 1024 |

### The four pieces, re-derived at `fe24a3029`

1. **No device store.** `include/vllm/model_executor/host_expert_slot_store.h::HostExpertSlotStore`
   is the only production `ExpertSlotStore`; the only other subclass is a test
   double. `include/vllm/model_executor/expert_streamer.h:8-9,30-31` says "the
   production destination is a contiguous device-side slot array" and "production
   writes to device memory". **Both sentences are false today**, and W1 makes them
   true rather than adding a second claim beside them.
2. **No device-capable read.** `Qwen35ExpertStream` holds
   `std::unique_ptr<HostExpertSlotStore> store_` (`qwen3_5.cpp:5621`) and reads
   `store_->Slot(r.slot)` at `:5381` and `:5437` — the CONCRETE class. There is no
   `SlotForRead()` on the interface, so the seam cannot be swapped even once a
   device store exists.
3. **The filler is `pread`-into-host.** `ExpertStreamer::EnsureFile` takes
   `store_.SlotForWrite(acq.slot)` and passes it to `::pread`
   (`src/vllm/model_executor/expert_streamer.cpp:76-94`). Not host-writable on a
   device allocation; see the scope verdict above.
4. **The consumer is CPU-gated.** `qwen3_5.cpp::KqExpertSlice` takes the slot arm
   only under `GetPlatform(d.q.device.type).is_cpu()` (`:5714`); every other
   platform falls through to `KqResidentSlice` (`:5218`) and stages the whole
   tower.

### Two facts #1124 does not name, and W0 fails without both

**(a) The slot arm itself stages the tower.** Inside `KqExpertSlice` the slot
branch calls `ResidentWeight(d, w)` — solely to inherit dtype, device and repack
markers — and then overwrites `.data` with the slot pointer. On CPU
`ResidentWeight` aliases and costs nothing. On a staging platform it takes the
upload branch, `void* p = d.b.Alloc(nb)` at `qwen3_5.cpp:1095` with
`nb = w.bytes.size()`, which is **the whole 1.1875 GiB stacked tower** — and
memoizes it in `w.d_dev`, so the very first streamed slice allocates what
streaming exists to avoid. That is the identical allocation #1123 died on
(48 towers, partway through layer 16 of 93). Lifting the `is_cpu()` guard alone
therefore reproduces #1123 rather than fixing it. W0 must build the slot tensor
WITHOUT `ResidentWeight`, and must refuse loudly if any `*_exps` tower reaches
`ResidentWeight` while streaming is on.

**(b) The load-time refusal fires first.** `CheckDeviceWeightFit` runs in
`src/vllm/entrypoints/model_loader.cpp` before the tokenizer and before any
weight I/O, gated on `target.needs_weight_staging()`, which is `true` on CUDA
including GB10 — the header at `include/vllm/platforms/interface.h:248-258` says
so explicitly and says why (`is_unified_memory()` / `is_integrated_gpu()` are the
WRONG predicate for staging, because GB10 is physically unified and the CUDA path
still binds distinct device pointers). The bound sums the WHOLE tensor table,
including all 335.62 GiB of `*_exps`, so it refuses this checkpoint before any
forward exists to take the slot arm. **W0 must make the refusal conditional on
the streaming lane serving those towers, or W0 produces no run.** This is the
single reason "lift one guard" is not one guard, and it is scoped here rather
than discovered during implementation.

The arithmetic that makes the conditional refusal EXACT rather than a fudge
factor: with the lane on, the staged set is the non-expert remainder plus the
slot arena, both known at load. No headroom fraction is invented, and the
over-count direction the existing bound documents (issue #1136) is unchanged.

**Two arena figures exist, they differ by 2.2105x, and the CHARGED one is the
one that decides the load.** This was written as one number in the first draft
and the fresh review caught it. At 8000 slots:

| Quantity | Value |
|---|---|
| what `Qwen35ExpertStream::Reserve` ALLOCATES: 8000 x 2,490,368 B, the IQ1_XXXS slice a default load streams | 19,922,944,000 B = **18.55 GiB** |
| what the fit bound CHARGES: 8000 x 5,505,024 B, the largest `*_exps` slice in the FILE, which is an MTP `nextn`-block Q2_K tower | 44,040,192,000 B = **41.02 GiB** |
| charged total, against the 34.34 GiB the bound counts as the remainder | 80,910,933,504 B = **75.35 GiB** |
| resident total, against the 26.01 GiB a default load actually stages | 47,853,196,800 B = **44.56 GiB** |

Both totals fit the 119.631 GiB pool, so the verdict on this checkpoint is
unchanged. The gap is the SAME over-count the bound already documents — it scans
the file, a load streams a subset of it — applied twice, once to the remainder
and once to the slot size, and it can only over-refuse. `GgufLargestExpertSliceBytes`'s
own header records the 2.2105x factor; what was missing was saying that the
published total carried it too.

### The CUDA kernel exists

`src/vt/cuda/cuda_quant_dot.cu` carries `WType::kIQ1_XXXS` through
`LaunchGemm`, `LaunchGroupedGemm` and `LaunchGroupedFusedSwiGLU`
(`:1874,1967,2182`), transcribed from the pinned `llama-cpp-unsloth` fork
oracle. So the encoding holding 96.92 % of this checkpoint has a CUDA GEMM, and
W0 is not blocked on a missing kernel. This is stated because it is the first
thing that would have killed the wave.

### The prefill exhaustion fact any W0 harness must handle

The last two-arm run recorded `exhausted=7813` on an 8000-slot cache, identical
at steps 2, 3 and 4 — all of it in prefill. That is not a defect and no slot
count fixes it. One step is one forward, prefill processes the whole prompt in
one forward, and slices acquired within a step are protected from eviction, so
the peak protected set for a T-token prompt is

    93 layers x 3 matrices x min(512, 10*T) experts

which saturates at 142,848 slices = **331 GiB** for any T >= 52 — the whole
model. A decode step touches ~2790 and exhausts nothing. A gate that reads TOTAL
`exhausted` therefore reports a red for a healthy lane. See `## Gates` G0-LIVE
for the two admissible harness shapes and which one is primary.

## Port map

Nothing is ported; there is no upstream. This is the local change map.

| Wave | Surface | Change |
|---|---|---|
| W0a | scratch probe, no tree change | print `cudaDevAttrPageableMemoryAccess`, `cudaDevAttrPageableMemoryAccessUsesHostPageTables`, `cudaDevAttrIntegrated`, `cudaDevAttrConcurrentManagedAccess` on `dgx:gpu0` |
| W0b | `include/vllm/platforms/interface.h`, `src/vllm/platforms/cuda.cpp`, `src/vllm/platforms/rocm.cpp` | new `virtual bool host_memory_is_device_addressable() const { return false; }` beside `is_integrated_gpu`; CUDA overrides from a probe taken once at registration next to the existing `cudaDevAttrIntegrated` probe; ROCm overrides from the `pageable_memory_access` capability it ALREADY probes (`rocm_backend.hip:96-103`) |
| W0c | `src/vllm/model_executor/models/qwen3_5.cpp` | `KqExpertSlice` takes the slot arm under `is_cpu()` OR `host_memory_is_device_addressable()`; the slot branch builds its tensor without `ResidentWeight`; a named `VT_CHECK` in `ResidentWeight` refuses a streamed `*_exps` tower reaching device staging |
| W0d | `include/vllm/model_executor/model_loader/gguf_device_fit.h`, `src/.../gguf_device_fit.cpp`, `src/vllm/entrypoints/model_loader.cpp` | the fit bound gains an explicit "these tensors are served by the slot lane, and the arena costs this instead" input; the loader passes it when the resolved config says streaming is on and the platform can read host slots |
| W0f | `src/vllm/model_executor/models/qwen3_5.cpp`, `include/vllm/model_executor/models/qwen3_5_weights.h`, `src/vllm/model_executor/models/qwen3_5_weights.cpp` | `ResidentWeight` returns a tensor over `w.bytes.data()` where `host_memory_is_device_addressable()`, instead of `Alloc` + `Copy` into `w.d_dev`; `MakeHostBytesDeviceAliasable` + `kDeviceAliasAlignment` make that pointer indistinguishable from the `cudaMalloc` one it replaces; a named `VT_CHECK` refuses an i8mm-repacked weight reaching device residency on EITHER branch (#1320) |
| W1 | `include/vllm/model_executor/expert_streamer.h`, new `include/vllm/model_executor/device_expert_slot_store.h` | `CommitSlot(int32_t, size_t)` on `ExpertSlotStore` (no-op on the host store); `DeviceExpertSlotStore` allocating slots through `vt::Backend::Alloc` with one pinned host staging slot, `SlotForWrite` returning staging and `CommitSlot` doing the H2D; correct the two false sentences in `expert_streamer.h` |
| W2 | `expert_streamer.h`, `host_expert_slot_store.h`, `device_expert_slot_store.h`, `qwen3_5.cpp` | `virtual uint8_t* SlotForRead(int32_t)`; `Qwen35ExpertStream::store_` becomes `std::unique_ptr<ExpertSlotStore>`; `:5381` and `:5437` read through the virtual; the store is selected from the platform |

## Tests to port

**None.** There is no upstream implementation, so there is no upstream test
suite, no fixture, no tolerance and no revision anchor to preserve. Recording
that explicitly is the requirement; inventing a "ported" label for locally
written tests would falsify the porting inventory. Scratch-written tests are
recorded as such in `porting-inventory.md` §9.

The tests this row WRITES, red-first, with the defect each is proven against by
mutation:

| Test | Proves | Mutation that must red it |
|---|---|---|
| `tests/vllm/platforms/test_platform.cpp` (extend) | the new predicate defaults false and the CUDA/ROCm assembly threads the probed value | flip the default to true; drop the assignment |
| `tests/vllm/model_executor/test_expert_slot_store.cpp` (new) | a device-flavoured store filled via `EnsureFile` yields byte-identical slot content to the host store | delete `CommitSlot`'s copy; return staging from `SlotForRead` |
| `tests/vllm/model_executor/test_gguf_device_fit.cpp` (extend) | with the lane on, the bound excludes `*_exps` and adds the arena; with it off, the bound is byte-identical to today | make the exclusion unconditional |
| `tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp` (extend) | the loader reaches the conditional refusal from the production entry point | delete the production call site |
| a `qwen3_5` slot-arm unit gate | the slot branch never calls `ResidentWeight`, and a streamed tower reaching device staging throws by name | remove the `VT_CHECK`; restore the `ResidentWeight` call |
| `tests/vllm/model_executor/test_resident_weight_host_addressable.cpp` (new, W0f) | `ResidentWeight` aliases the host bytes on a host-addressable staging platform and allocates NOTHING; the aliased pointer meets `kDeviceAliasAlignment`; a discrete platform stages byte-identically to today; a MISALIGNED BORROW declines and stages rather than being copied into anonymous memory; the three refusals fire on the aliasing branch too | delete the aliasing branch; make the predicate unconditional; delete each `VT_CHECK`; drop the `borrowed()` guard; claim alignment without providing it; re-home without copying the bytes |

## Gates

Correctness first, always. No speed number from this row is admissible until
G0-CORRECT and G0-LIVE both pass in the SAME run that produced it — that is the
whole content of #912 F1, where a published decode figure measured a dead cache.

**G0-CORRECT (W0, blocking).** `Qwen3.8-2.4T-A95B UD-Q1_0`, `--device cuda`,
streaming ON, greedy, one fixed prompt, 32 decoded tokens. The token IDs must be
**byte-identical** to `--device cpu`, streaming ON, same binary, same checkpoint,
same prompt. "It ran" is not a result and neither is prose that looks coherent.
A mismatch stops the wave and no speed number is reported.

**G0-LIVE (W0, blocking).** From the same process:

* `steps > 0` — the step clock advanced. `steps == 0` with a printed decode
  figure is exactly the #912 F1 shape.
* `forced == 0` — no slice was served by the forced-fallback switch.
* **decode-phase `exhausted` delta == 0.** Not the total. The primary harness
  snapshots `detail::ExpertStreamSnapshot()` immediately after the prefill step
  and again at the end, and gates the DIFFERENCE. That is the shape a real
  benchmark needs, because it works at any prompt length, and prefill exhaustion
  is structural (see `## Our baseline`).
* Secondary control only, not the benchmark: a prompt of T <= 4 tokens with
  >= 11,160 slots makes prefill itself fit — 11,160 x 2,490,368 B = 25.88 GiB of
  arena plus 26.01 GiB of weights = 51.89 GiB, inside the 119.631 GiB pool — so
  TOTAL `exhausted` is legitimately 0. It is a useful cross-check on the
  snapshot arithmetic and it is not a workload, so it never replaces the diff.

**G0-SPEED (W0, reporting).** Decode s/token, CUDA arm against the CPU arm,
same box, same `rc` lease, same prompt, same token count, three interleaved
reps, min and median both recorded, plus resident bytes and the fill/hit/byte
counters. **No floor is set and none may be invented.** This row's first number
is a measurement of an unknown, and a CUDA arm SLOWER than the CPU arm is a
real, publishable result that closes the unified shortcut — recorded in
`docs/BENCHMARKS.md` as a measured negative, not as a failure to be tuned away.

**G1 (W1).** `DeviceExpertSlotStore` driven through `ExpertStreamer::EnsureFile`
produces byte-identical slot contents to `HostExpertSlotStore` on the same
input, on a CPU `vt::Backend`, red-first and mutation-proven per the table above.

**G2 (W2, reachability).** Per `## Nothing lands dead`: delete the production
selection of the device store in a scratch copy and rerun the focused gate. A
gate that stays green without it measured a class, not a capability.

**G-DISCRETE (owed, cannot run here).** See `## Owed`.

## Evidence

W0a and W0e, `dgx:gpu0` (GB10, sm_121a, driver 580.173.02, CUDA 13.0.88, 20
cores, 122,502 MiB RAM, 30,625 MiB swap), one `rc hold`
`edb4b3d0-5d6e-422f-ade6-bff5339e3396`, 2026-08-18T22:13:12Z to
2026-08-19T00:58Z, released by interrupting its client. Source `95883dcae`,
the head of PR [#1242](https://github.com/mudler/vllm.cpp/pull/1242).

### The build, because a degraded one would have voided every number

`cmake -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON
-DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_CUTLASS_DIR=/cutlass
-DVLLM_CPP_TRITON=ON`, CUTLASS 4.5.0 staged host-side. The configure log
reports `fp4-mma`, `cutlass-nvfp4`, `cutlass-fp8`, `marlin-nvfp4` and `fa2` all
`ENABLED for [121a]`, `CUTLASS found at /cutlass; enabling sm120a NVFP4 cutlass
GEMM`, `FlashAttention-2 prefill/decode: ENABLED for arch(es) [121a]`, and
`Triton AOT W2: embedded trees [sm_80;sm_86;sm_89;sm_90a;sm_100a;sm_121a]`.
Both arms are the SAME binary, and every arm ran with the page cache dropped
first (`echo 3 > /proc/sys/vm/drop_caches`, ~90.9 GB available after each drop).

### The harness, and why it is not `vllm-cli`

`benchmarks/expert_stream_device_w0e.cpp`, a thin client of `include/vllm.h`
linked against the packaged shared library, which exports the C ABI and nothing
else. **The project builds it**, as the `expert-stream-device-w0e` target. It
was first written unwired, beside `marlin_moe_standalone.cpp`, on the reading
that a gate instrument is not a shipped capability. Wiring it found that the
file did not compile under the project's own flags at all — three backslash
continuations inside `//` comments, which `-Werror=comment` rejects — so the
recorded recipe was the only thing that had ever built it. An instrument
nothing compiles rots against the very ABI it measures, and a measurement whose
harness no longer builds cannot be reproduced, so the target is the correct
shape even though the file is not a capability. It needed to exist because the gate wants three things from ONE
generation, and no shipped command produces all three: the generated TOKEN IDS
(`vllm_complete_tokens`, ABI v13, which writes them into a caller buffer), a
PER-STEP arrival timestamp (a custom logits processor, invoked once per decode
step), and the expert-stream counters at both ends of the run
(`VT_MOE_EXPERT_STREAM_STATS_EVERY=1`, so the periodic line prints at step 1 and
every step after; at the default 16 a 32-token run prints two lines and a
5-token one prints none, which is indistinguishable from a dead lane).

The logits processor is a PURE OBSERVER: it records and returns without touching
`logits`, so the argmax the sampler takes is the one it would have taken without
it. Its own `token_ids` view is recorded but NOT used for the gate, because
`tests/capi/test_capi.cpp` states that under the async scheduler that view can
lag the emitted tokens.

The prompt is fixed as TOKEN IDS rather than as text, `760,6511,314,9338,369`
("The capital of France is", produced by `build/examples/tokenize` from the
checkpoint's own GGUF metadata), so both arms are fed byte-identical input and
no tokenizer step sits between the two.

### The CPU arm: measured, reproduced, and it replaces the VOID #912 F1 figure

`--device cpu`, `VT_GGUF_PREFAULT=0 VT_MOE_EXPERT_STREAM=1`, `--max-num-seqs 1`,
greedy, 32 tokens. Four runs, two slot counts, two reps each:

| slots | arena | load | TTFT (step 1) | steady decode s/token, steps 4-32 | total generate | peak RSS | min avail | peak swap |
|---|---|---|---|---|---|---|---|---|
| 4000 | 9.28 GiB | 271.1 s | 85.90 s | median **11.22**, min 9.62, max 12.51 | 502.3 s | 86.5 GiB | 16,840 MiB | not sampled |
| 4000 | 9.28 GiB | 255.7 s | 79.09 s | median **11.05**, min 9.43, max 13.25 | 460.7 s | 86.5 GiB | 16,347 MiB | 6,883 MiB |
| 8000 | 18.55 GiB | 261.6 s | 94.25 s | median 45.40, min 20.73, max 82.68 | 1643.2 s | 86.6 GiB | 6,985 MiB | not sampled |
| 8000 | 18.55 GiB | 266.5 s | 132.74 s | median 39.98, min 16.06, max 94.30 | 1581.6 s | 86.6 GiB | 6,941 MiB | **30,625 MiB (all of it)** |

**Read the 8000-slot rows as a memory result, not a cache result.** A bigger
cache came out slower, and the four available median pairings span 3.56x to
4.11x: 3.62x same-rep on the pair that produced the publishable figure (39.98
against 11.05) and 4.05x pairing rep 1 with rep 1 (45.40 against 11.22). It also
came out far less steady: the steady window's max/min ratio is 3.99x in rep 1
(20.73 to 82.68 s) and 5.87x in rep 2 (16.06 to 94.30 s), against 1.30x and
1.40x at 4000 slots. Whichever pairing is quoted, the direction is wrong.

**The cause is NOT that the arena does not fit, and the arithmetic says so.**
18.55 GiB of arena beside this model's 62 GiB of dense weights is 80.55 GiB on a
119.63 GiB box, which fits with room to spare, and the columns agree: peak RSS
moves 86.5 -> 86.6 GiB for a 9.27 GiB arena delta. The two columns that DO move
are `min avail`, 16,347 -> 6,941 MiB, a 9,406 MiB fall that is about the arena
delta, and peak swap, 6,883 -> 30,625 MiB, which is all of it. The
best-supported reading of that pattern is **page-cache displacement**: the
borrowed 370 GiB expert mapping is served out of whatever memory is free, the
arena takes that memory, and the reclaim pressure it creates pushes anonymous
pages to swap. That is a reading of the columns and not a proven mechanism, and
this run cannot separate it from plain reclaim pressure, because it sampled no
page-cache size and no major-fault counter. The operational conclusion does not
depend on which of the two it is, and it is unchanged: **more slots is not a
free knob.**

Both 8000-slot runs reproduce each other, so this is the box's behaviour and not
a fluke. **The publishable CPU figure is the 4000-slot one, 11.05 s/token**,
which is rep 2's median; rep 1's is 11.22, so the two reproduce within 1.5%. It
is the first live-cache streaming-ON decode number this project has;
`docs/BENCHMARKS.md:8` recorded streaming-ON as VOID (#912 F1, the step clock
dead from token 3) with a re-measure owed.

Steps 1-3 are excluded from the steady figure and reported separately because
they are not steady state: step 1 is prefill, and steps 2-3 are still filling a
cold cache (85.90, 80.38, 16.06 s, then 11.7 and below).

**G0-LIVE, gated on the DIFFERENCE and never the total.** At 4000 slots the
after-prefill snapshot is `steps=1 hits=0 misses=10074 evictions=0 fills=4000
bytes=9961472000 exhausted=6074` and the final one is `steps=32 hits=37096
misses=58538 evictions=48464 fills=52464 bytes=130654666752 exhausted=6074`, so
the decode-phase `exhausted` delta is **0** over 31 decode steps. At 8000 slots
the same pair reads 2074 and 2074, delta **0**. Both reps of each slot count
produce byte-identical counters, so the lane is deterministic. `steps=32 > 0`.
`forced` is 0 by construction and is deliberately absent from the stderr line:
its only setter is `detail::ExpertStreamSetForceFallback`, which
`src/vllm/model_executor/models/qwen3_5_internal.h:488` records as having no
production caller, and `qwen3_5.cpp:5521` (`++forced_` in
`Qwen35ExpertStream::Slice`) counts it separately from `exhausted` for exactly
that reason, in a comment that says so.

**The tokens.** All four CPU runs, across both slot counts, produced the same 32
ids:

```
11751,13,11751,369,264,3177,7172,303,279,17631,919,314,9338,11,383,279,
181474,10629,13,1049,369,279,7526,3177,303,9338,321,369,3750,364,1141,25438
```

which detokenize to " Paris. Paris is a city located in the northern part of
France, on the Seine River. It is the largest city in France and is known for
its iconic", `finish_reason=length`, `completion_tokens=32`.

### The CUDA arm at `95883dcae`: it loads, and it does not generate

**Everything in this subsection was read on W0f's PARENT tree and is kept as
measured.** It is the diagnosis W0f was built from, not a claim about the tree
this spec describes; the subsection after it is the same harness re-run with W0f
and it reaches 32 decode steps.

**The load is the new thing and it works.** `--device cuda` on this checkpoint
used to refuse ([#1123](https://github.com/mudler/vllm.cpp/issues/1123)); W0d's
conditional bound removes that refusal when the lane serves the towers, and the
load now completes in 255-272 s with 61.20 GiB resident. The lane then engages:
the `[expert-stream] ON slots=... resident=... GiB` banner prints on the device
arm, which is W0c's whole point.

That banner is also the first production evidence that **W0b's CUDA leg is
reached and answers true on real hardware**. Neither the banner nor the
successful load is reachable unless `host_memory_is_device_addressable()`
returned true from the CUDA platform, so the leg no longer rests on the fake
platform in `test_expert_stream_device_slot` alone. The negative — a mutation
that makes it answer false — is still owed and is in `## Owed`.

**Then the first forward exhausts the machine. Seven attempts, zero decode
steps, every one of them.**

| slots | arena | prompt tokens | load | RSS after load | decode steps | peak system `used` | peak swap used |
|---|---|---|---|---|---|---|---|
| 8000 | 18.55 GiB | 5 | 267.2 s | 61.20 GiB | 0 | 100,215 MiB | not sampled |
| 4000 | 9.28 GiB | 5 | 271.6 s | 61.20 GiB | 0 | 94,737 MiB | not sampled |
| 4000 | 9.28 GiB | 5 | 255.3 s | 61.20 GiB | 0 | 120,351 MiB | 30,569 MiB |
| 3500 | 8.12 GiB | 1 | 262.8 s | 61.20 GiB | 0 | 120,296 MiB | 28,138 MiB |
| 3500 | 8.12 GiB | 1 | 265.3 s | 61.20 GiB | 0 | 120,347 MiB | 30,461 MiB |
| 3500 | 8.12 GiB | 1 | 272.3 s | 61.20 GiB | 0 | 120,306 MiB | 30,172 MiB |
| **64** | **0.15 GiB** | 1 | 268.9 s | 61.20 GiB | 0 | 118,257 MiB | 30,211 MiB |

Each run was stopped by a guard that kills the container when MemAvailable plus
SwapFree falls under a floor, because an out-of-memory kill on GB10 takes the
whole machine down rather than the process. The guard is not what makes these
runs fail: the last two rows above were still climbing at roughly 10 GB of swap
per minute with under 600 MiB of swap left, so the kernel was seconds from the
kill the guard exists to prevent.

**The expert lane is not the cost, and that is measured rather than argued.**
The 64-slot row is the control: a **0.15 GiB** arena dies in the same place an
**18.55 GiB** one does, 124x smaller and no further along. The in-place tower
fallback is not the cost either: a 1-token prompt has a protected set of
`93 x 3 x 10 = 2,790` slices, which fits 3500 slots with no fallback taken at
all, and it behaves exactly like the 5-token prompt whose 13,950-slice set
cannot fit any arena on this box.

**The growth is ANONYMOUS, not file-backed**, so the GPU is not pinning the
mapping's pages through its address translation — which was the first
hypothesis, and it is wrong. Sampling the container process every 5 s through
the load and into the forward:

```
RssAnon:   8.1 -> 13.9 -> 25.9 -> 38.3 -> 49.4 -> 61.4 GB     (through the load)
RssFile:   0.1-0.2 GB throughout, 4.6 GB at the very end
VmSwap:    0 -> 5.4 -> 11.4 -> 16.5 -> 27.1 -> 31.0 GB        (inside the forward)
```

Host `RssAnon` plus `VmSwap` reaches about 65 GB while the system reports
~119 GiB `used`, and the ~42 GiB difference is device memory that this unified
part does not charge to the process RSS. **So the non-expert weights are
resident twice** — once as the host-side `OwnedTensor` and once as the
`ResidentWeight` device staging copy — and on a box where device memory IS host
memory, that doubling is what runs it out.

[`expert-streaming.md`](expert-streaming.md) already measured the host half of
this: the GDN V-head reorder makes `attn_qkv` and `ssm_out`
`kTransformedWeight`, which expands them from about 5.5 bits to bf16, roughly
**50 GiB** of the 61.20. The CPU arm pays that once and serves. The CUDA arm
pays it twice and cannot. That spec's own sentence — "Whoever takes this needs
BOTH: the streaming lane for the ~330 GiB of experts, and a transformed-weight
path that does not expand" — is exactly this result, and W0 delivered the first
half. Filed as [#1299](https://github.com/mudler/vllm.cpp/issues/1299), and
**FIXED as W0f**, which is the next subsection. The prediction that the fix would
need a transformed-weight path turned out to be one option rather than the only
one: not expanding, and not paying for the expansion twice, are different repairs
and W0f is the second.

### The CUDA arm with W0f at `9c783a8be`: it decodes

Same harness, same lease shape, same box, same prompt ids `760,6511,314,9338,369`,
streaming ON at 4000 slots, greedy, 32 tokens, page cache dropped between arms.
`../benchmark-record.md` carries the full entry including the two VOID attempts
that preceded it, which were void because the build target relinked nothing.

| Observable | CUDA | CPU (same-lease control) |
|---|---|---|
| load | 266.330 s | 253.504 s |
| RSS after load | 61.20 GiB | 62.45 GiB |
| decode steps | **32** | 32 |
| decode-phase `exhausted` delta | **0** | **0** |
| peak RSS | **97.75 GiB** | 92.19 GiB |
| swap used at peak | 0 | 0 |
| container exit | 0 | 0 |

**What W0f moved, counted rather than inferred.** An RSS curve cannot separate
"the branch declined and staged", "the branch re-homed and the pages did not come
back" and "something else allocated", so `MakeHostBytesDeviceAliasable` reports
its outcome per weight. Read at the point re-homing plateaus, call **1361** of
the first forward: **60.793 GiB** re-homed into an aligned host block and then
aliased, **~9.2 GiB** declined as misaligned GGUF borrows and still staged, and
0.02 GiB aliased in place. The qualifier is part of the number, because the
counters are per CALL with no memo on the alias branch and keep growing at
roughly 70 GiB per decode step. On the CPU arm the same counter reads **0
calls**, which is a live control that the branch is platform-gated and not an
argument that it is.

**And the ids diverge at step 7.** `...,264,3177,7172,...` on CPU against
`...,264,3177,303,...` on CUDA. An instrumented CPU run printing the top-2 logits
per step shows the CPU arm's own runner-up at that step is `303` — exactly what
CUDA emitted — 0.264709 logits behind, 1.4 % of the winner; two steps later the
margin is 0.022802, about 0.1 %. The declared gate fails and the wave stops.

**The alias is excluded as the cause, on the target silicon.** `rc` job
`7c7a05e9-be87-48f4-94ae-1bbe0340f063` on `dgx:gpu0` (`NVIDIA GB10 sm_121`,
driver 580.173.02, cuBLASLt 130101, the predicate re-derived in the job's own
output as `pageableMemoryAccess=1 integrated=1`) ran six checkpoint shapes
crossed with both cuBLASLt formulations the dense path issues: 12 measurements,
`PROBE_EXIT=0`, `PROBE_FAILURES=0`. A repeated heuristic call is identical 12/12;
the tree's unset preference equals the documented 256 default 12/12; weakening
the promise to 16 moves nothing 12/12; and `cublasLtMatmul` output is bit-exact
between a `cudaMalloc` operand and a 256-aligned host block 12/12,
`differing=0`, every status `SUCCESS`. At least five distinct algorithm
configurations appear across the twelve and they differ from the earlier
`thor:gpu0` leg's, so the heuristic was re-resolved rather than replayed and the
instrument discriminates. The structural reason needs no lease:
`cublasLtMatmulAlgoGetHeuristic` takes no operand pointers, so alignment reaches
it only through a preference this tree never sets.

**Excluding one cause is not identifying another.** What the divergence IS
remains unmeasured and is carried under `## Owed` with its next traceable step
named.

### What was running beside the measurement

The lease excluded every other `rc` job for its duration. Two things were not
excluded and are recorded rather than assumed away. An orphaned
`VLLM::EngineCore` process from an earlier session held 3.32 GiB of host RSS for
the entire window; it is not this row's process and it was left alone. And the
FIRST arm started at 22:13, seconds after the previous lease holder's four-hour
render released the device, with the one-minute load average still at 17.5 —
which is one of the two reasons the first 8000-slot run is the noisiest of the
four, the other being the swap it drove. Every later arm started from a quiet
box.

## Dependencies

| Dependency | Shape |
|---|---|
| **CPU decode re-measure on a live cache** | The DENOMINATOR for G0-SPEED, not a gate that blocks starting. `docs/BENCHMARKS.md:8` records streaming-ON decode as VOID (#912 F1) with a re-measure owed, and the operator is arranging it separately. G0-CORRECT and G0-LIVE need no denominator and can run first. **If the box frees up for only one run, take the CUDA arm** — it is the one that does not exist at all today, it produces the correctness verdict as well as a number, and the CPU denominator can follow. |
| `dgx:gpu0` | The only GB10. Leased through `rc`, never bare ssh; the hold sits behind a trap that releases on every exit path. W0a HAS taken its lease and returned `PAGEABLE_OK`; W0e's measurement still QUEUES behind other work on the box. W0b/W0c/W0d are all buildable and unit-gatable without it. |
| The 370 GiB checkpoint | Already staged for the parent row; recipe in [expert-streaming.md](expert-streaming.md) "Loading a 370 GiB split GGUF". |
| PRs #1200 and #1216 | Both edit `.agents/specs/expert-streaming.md`. This row does not, deliberately — see `## Risks/decisions`. |
| #1126 (`Backend::DeviceMemoryInfo` has no CUDA override) | NOT a blocker. W0 and W1 read the budget from `residency_policy().device_memory_total_bytes`, which CUDA already probes at registration; they never touch the live free/total seam #1126 owns. |
| PR #1203 | Introduces `path::Symbol` citations and `scripts/check-symbol-anchors.py`. This spec uses symbol anchors where a symbol exists and lines only where the citation is to a comment or a specific statement. |

## Work breakdown

Each wave states its own gate and its own stop condition. A stop condition is
where the wave ENDS, not where it degrades quietly into the next one.

### W0 — the integrated/unified path (critical path for the GPU number)

* **W0a — the probe.** ~30 lines, no tree change, inside an `rc` lease: print
  `cudaDevAttrPageableMemoryAccess`, `...UsesHostPageTables`, `cudaDevAttrIntegrated`
  and `cudaDevAttrConcurrentManagedAccess` on `dgx:gpu0`. This is the cheapest
  decisive experiment in the row and it runs first, because the whole W0
  mechanism rests on a GB10 CUDA kernel being able to dereference a pointer from
  the host slot arena, which is plain `std::vector<uint8_t>` storage.
  **Stop condition:** if `PageableMemoryAccess == 0`, W0b as designed is wrong.
  The alternative is a `cudaHostAlloc`/`cudaHostRegister` arena, which works on
  ANY CUDA device including discrete but changes the store's allocator and its
  ownership story. That is a different design and it comes back as
  `NEEDS_DECISION` rather than being substituted silently.
  **RESULT (`dgx:gpu0`, `W0A_VERDICT=PAGEABLE_OK`):** `PageableMemoryAccess = 1`,
  `Integrated = 1`, a kernel read and wrote a 2,490,368-byte slot of plain
  `std::vector` storage correctly, bandwidth ratio 2.06-2.28x over the reps. The
  stop condition did not fire and the `cudaHostAlloc` fallback is not taken.
* **W0b — the predicate.** `host_memory_is_device_addressable()`, base false,
  CUDA from the W0a attribute, ROCm from its already-probed
  `pageable_memory_access`. **Why not an existing predicate:** `is_cpu()` is what
  is being lifted; `needs_weight_staging()` is true on CUDA everywhere and would
  gate nothing; `is_unified_memory()` answers the opposite question (backend.h:58-66
  records that CUDA on GB10 reports unified while a `cudaMalloc` pointer is still
  not host-dereferenceable); and `is_integrated_gpu()` alone is insufficient
  because `rocm.cpp:52-58` already documents integrated-without-pageable-access as
  a real device class. A discrete CUDA device answers false, keeps falling through
  to `KqResidentSlice`, and therefore keeps hitting the #1123 refusal — which is
  correct, and which is what W1/W2 exist to remove.
  **Gate:** `tests/vllm/platforms/test_platform.cpp`, mutation-proven — and read
  that literally only since #1378. The first shape of this gate asserted the
  implication and the board value on hardware where both attributes read 1, so
  deleting either term of the conjunction left it green. The rule now lives in
  `HostMemoryIsDeviceAddressableFromAttrs` and is gated over all four attribute
  pairs on the CPU tier, where both deletions are red.
  **Stop condition:** none; this wave is small and self-contained.
* **W0c — the slot arm without the tower.** Lift the guard, build the slot
  tensor directly instead of through `ResidentWeight`, and add the named refusal
  for a streamed `*_exps` tower reaching device staging.
  **Gate:** the `qwen3_5` slot-arm unit gate above.
  **Stop condition:** if the marker set `ResidentWeight` carries cannot be
  reproduced without staging (a repack marker that only the upload path can
  set), stop — that is a different change and it is not this wave.
* **W0d — the conditional refusal.** Teach the fit bound the streamed-tensor
  exclusion and the arena, and pass it from the loader.
  **Gate:** `test_gguf_device_fit` for the arithmetic and the residency-route
  predicate, and `test_gguf_device_fit_reach` for the production reach; the
  lane-off bound must be byte-identical to today. The reach suite pins
  `VT_GGUF_KEEP_QUANT=1` for its lane-on cases, deliberately: the production
  default for that policy is `GgufQuantComputeAvailable()`, which is true on a
  real GB10 and false in a CPU-only build carrying a fake CUDA platform, so
  inheriting it would make every lane-on case measure the absence of a CUDA
  kernel instead of the lane.
  **Stop condition:** if the exclusion cannot be expressed without the bound
  taking a general per-tensor staging POLICY (the shape #1136 explicitly refuses
  to invent), stop and return `NEEDS_DECISION`. The lane's tensor set is
  `*_exps` and is knowable; a general policy input is not.
* **W0f — the dense half.** Discovered by W0e's first seven attempts and scoped
  by them, not by reading: the checkpoint loads and then exhausts the box with
  zero decode steps, and the four measurements in #1299 rule out the arena, the
  prefill fallback, and a pinned mapping in turn. Give `ResidentWeight` the same
  branch W0c gave `KqExpertSlice`.
  **Why an alignment contract and not a kernel survey.** The staging branch is a
  verbatim byte copy, so the ONLY thing a consumer can notice about the
  substitution is the pointer's alignment. `cudaMalloc` returns 256; a
  `std::vector<uint8_t>` returns 16, because a large glibc block is an mmap chunk
  landing at page+16. Matching the allocator therefore settles every consumer at
  once, and the alternative — deriving a floor from the widest load any kernel
  performs — does not close: the widest hand-written one is a 16-byte `cp.async`
  granule whose gate checks the SHAPE and assumes the base, and cuBLASLt is
  separately PROMISED 256 by a preference default this tree never sets.
  **Why a borrow is not re-homed.** It owns no anonymous pages. Copying a clean,
  file-backed GGUF mapping into an aligned anonymous block would create exactly
  the residency this row exists to remove, and would break a tied
  `token_embd`/`lm_head` pair's single keep-alive.
  **Gate:** `test_resident_weight_host_addressable`, mutation-proven.
  **Stop condition:** if any weight on this path needed a device layout DIFFERENT
  from its host bytes, that weight could not skip the copy and W0f would need a
  per-tensor answer instead of a branch. It does not: `ResidentWeight` copies
  bytes verbatim and returns the same dtype, shape and (dropped) marker set on
  both arms, so there is no device layout to preserve. The layout-bearing
  markers are handled instead — `elem_kn_repacked` and `repacked` are refused by
  name, and `q8_0_aligned` is a load-time rewrite of the HOST bytes that no
  Qwen3.5 path sets.
* **W0e — the measurement.** G0-CORRECT, G0-LIVE, G0-SPEED, on one lease.
  **Stop condition:** a token mismatch, `steps == 0`, or a non-zero decode-phase
  `exhausted` delta stops the wave and voids the number.
* **W0g — the two-arm dump.** RAN. See `## Now` and
  [`../benchmark-record.md`](../benchmark-record.md).
* **W0h — is the CUDA arm WORSE, or only different?** A teacher-forced negative
  log likelihood comparison over a fixed corpus, with a pre-registered decision
  rule and `llama-cpp-unsloth` as the oracle arm. Scope, design, rule, tests and
  stop conditions:
  [cuda-arm-degradation-experiment.md](cuda-arm-degradation-experiment.md),
  issue [#1736](https://github.com/mudler/vllm.cpp/issues/1736). It produces no
  speed claim, and G0-SPEED stays VOID under every outcome.

### W1 — the device slot store (with its fill contract)

`DeviceExpertSlotStore` allocates its slot array through `vt::Backend::Alloc`.
It cannot be filled by `pread` (scope verdict above), so W1 adds
`CommitSlot(int32_t slot, size_t bytes)` to `ExpertSlotStore`: `SlotForWrite`
returns a host-writable staging buffer, `pread` fills it as it does today, and
`CommitSlot` performs the single contiguous H2D. `HostExpertSlotStore::CommitSlot`
is a no-op, so the host path stays byte-identical and keeps its direct
`pread`-into-slot. `ExpertStreamer` calls `CommitSlot` only on the success path;
the existing `catch` that invalidates the cache entry already covers the
partial-fill case the parent spec documents at length.

**Design answer: staging bounce, not a bespoke device filler.** A true zero-copy
filler (GPUDirect Storage / `cuFile`, or an `O_DIRECT` DMA into a device BAR
mapping) moves fewer bytes, needs a driver capability probe, a mount-level
check, an aligned-I/O path and a fallback for every case that fails those. The
bounce costs one extra host-to-device copy of 2.375 MiB per miss on top of a
disk read of the same size, lands in one wave, and keeps the zero-copy filler
genuinely optional rather than load-bearing. It is chosen for that reason and
not because it is faster; the measurement that would justify replacing it is a
device-arm decode where the H2D leg is a measurable fraction of fill time, and
that measurement does not exist yet. Recorded in `## Owed`.

**Gate:** G1. **Reachability:** W1 alone lands UNREACHED — nothing selects the
store and nothing can read it, because the read is still the concrete
`HostExpertSlotStore::Slot()`. Per `## Nothing lands dead` that is admissible
only if the commit body, the PR body and this spec's `## Owed` all name it and
name W2 as the owning wiring. **The cheaper and more honest shape is to land W1
and W2 as one pull request**, and that is the recommendation here; splitting
them is a scheduling choice that costs an explicitly-declared unreached slice.

**Stop condition:** if `CommitSlot` cannot be added without changing every
existing `ExpertSlotStore` caller's contract in a way that alters host-path
behaviour, stop — the host path must stay byte-identical, and a change that
cannot preserve that is a different design.

### W2 — the device-capable read

`virtual uint8_t* SlotForRead(int32_t)` on `ExpertSlotStore`;
`Qwen35ExpertStream::store_` becomes `std::unique_ptr<ExpertSlotStore>`; the two
`store_->Slot(...)` reads go through the virtual; the concrete store is selected
from the platform. **Gate:** G2, the call-site-deletion mutation.
**Stop condition:** if the returned pointer's device-ness cannot be expressed in
the `vt::Tensor` the GEMM binds without a second change to the tensor
construction, stop and re-scope — that is W0c's territory and it must not be
re-derived here.

## Risks/decisions

| Risk / decision | Call |
|---|---|
| **The GB10 ATS penalty could erase W0's win entirely.** Device access to host-resident weights on GB10 is recorded as carrying a real penalty, and this lane reads 6.95 GB per token that way. | Accepted as the thing being measured, not assumed away. G0-SPEED is what settles it, and the settling measurement is named: the CUDA arm's decode s/token against the CPU arm's, same box, same lease, three interleaved reps. A CUDA arm at or above the CPU arm's time is a genuine negative result — it closes the unified shortcut for this box, leaves W1/W2 standing for the discrete case, and is recorded in `docs/BENCHMARKS.md` as measured. It is a few hours, not a campaign, and that asymmetry is why W0 runs first. |
| W0 is FOUR edits, not one guard. | Stated rather than discovered. The two forcing facts are in `## Our baseline` (a): the slot arm's own `ResidentWeight` call stages the tower, and (b): the load-time refusal fires before any forward. Neither is in #1124's four pieces. W0d touches the fit bound, which is the one place this wave reaches into another row's surface; it is additive and exact (an explicit tensor exclusion plus the arena), it does not touch the KV/activation term `KV-WARMUP-PROFILE` owns, and if it cannot stay that way the W0d stop condition returns `NEEDS_DECISION`. |
| W1/W2 can be built here but only VALIDATED on hardware nobody here has. | Recorded as owed with its exact measurement rather than dressed as a gate. This host has no discrete NVIDIA GPU, and on `dgx:gpu0` device memory IS host memory, so a device store there proves the plumbing and not the capability. See `## Owed`. |
| A device store is the wrong shape if the answer is "put the slots in pinned host memory". | CLOSED for this box by W0a. The fallback was conditional on `PageableMemoryAccess == 0` on GB10, and the probe measured 1 with a kernel correctly reading and writing a 2,490,368-byte slot of plain `std::vector` storage, so the pageable arena W0c builds on is the measured answer and not a hope. The `cudaHostAlloc` arena stays written down here for the DISCRETE case, where it is still device-readable over PCIe and still almost certainly too slow to serve 6.95 GB/token — that case is W1/W2's, not W0's. |
| This row does not edit `.agents/specs/expert-streaming.md`. | Deliberate. PRs #1200 and #1216 both edit that file today, and its `## Owed` entry for #1124 remains TRUE as written — it names the capability and the issue, and it does not name a row ID that this row's existence falsifies. Cross-linking is one-way, from here to there. Re-pointing that entry at this row is a one-line follow-up once both PRs land, and it is not worth a conflict now. |
| `ENGINE_ROWS` in `scripts/check-agent-record.py` is a shared counter, exactly the "measurement of one file stored in another" coupling AGENTS.md warns about. | Bumped 162 -> 163 for a real new row, with its justification paragraph, per the constant's own comment history. Checked against every open PR: none bumps it (#851 carries a stale `156` as diff context, #361 does not touch it), so this addition takes the lock cleanly. |
| Splitting the device capability out of `ENG-EXPERT-STREAM` rather than adding a W7-W9 to it. | The parent row's spec is 1600 lines and has two open PRs editing it. A separate row gives this capability an independent lifecycle state and a per-row spec surface, which is the shape AGENTS.md prefers (one file per row, read with a glob). The parent keeps the MECHANISM; this row owns the DESTINATION. |
| Streaming and the grouped keep-quant MoE path remain mutually exclusive. | Unchanged by this row, and unchanged by W0. `VT_MOE_EXPERT_STREAM=1` still disables grouping and says so once on stderr. Making them compose needs a slot-aware grouped GEMM and is its own row. |

## Owed

| Owed | Why it is open |
|---|---|
| **`kQwen3MoeFactory.streams_routed_experts = true` is a CORRECT declaration that nothing READS today.** The flag's only reader is the loader's lane block, which is on the GGUF path, and `kGgufArchArms` (`model_loader.cpp`) maps no `general.architecture` onto `Qwen3MoeForCausalLM` (Qwen3-Coder), so no GGUF load can resolve to that factory. | It is set anyway because it is TRUE: `qwen3_moe.cpp` composes the same `RunMoeBlock` the Qwen3.5 MoE forward does, which is why it holds an `EndStepGuard` at all, so its experts do reach `KqExpertSlice`. Declaring it false to make every setting reachable would put a false statement in the registry, and the safe-direction default would then hide it. Named here per `## Nothing lands dead` rather than left for the next reader to find: `ENG-EXPERT-STREAM-DEVICE` owns the wiring under [#1124](https://github.com/mudler/vllm.cpp/issues/1124), and the flag becomes read the moment a `qwen3moe` GGUF arch arm exists. The `Qwen3_5Moe*` setting beside it IS read, and its gate is now EVIDENCED rather than asserted: mutation M-A3 (`kQwen3_5MoeFactory.streams_routed_experts = false`, `qwen3_5_moe.cpp`) was listed as NOT RUN in #1377's pull request body, was then run by that pull request's fresh review, and was re-run during the #1378 repair with the result recorded -- compile status 0, `git diff --stat` 1 file / 1 insertion / 1 deletion, `test_gguf_device_fit_reach` 14 cases with 2 failed and 66 assertions with 6 failed, exit status 1, tree restored byte-identical by sha256. **The laguna half of the same claim is VACUOUS and is not evidence for anything.** Mutation M-A3c (`kLagunaFactory.streams_routed_experts = true`) is GREEN, and correctly so: `laguna` has no entry in `kGgufArchArms` (`model_loader.cpp`), so a Laguna GGUF is refused as an unsupported architecture before the fit check runs and no setting on that factory can reach the lane. Nothing is owed to make it gateable -- manufacturing a gate for an unreachable flag would be worse than saying this -- and `DeepseekV4ForCausalLM`, which DOES have an arch arm, is the case that carries the architecture term's weight. |
| ~~**`scripts/check-doc-checkpoint.py` stays RED on this branch for commit `939755f99` and cannot be made green here** ([#1387](https://github.com/mudler/vllm.cpp/issues/1387)). That commit appended a measurement to `.agents/benchmark-record.md` without writing `docs/FEATURES.md`, whose streaming row then said "CPU keep-quant towers only" after W0c had made a host-readable staging device take the slot arm.~~ **CLOSED 2026-08-20** ([#1442](https://github.com/mudler/vllm.cpp/issues/1442)'s flow). The page half landed twice over: `5f4eb356e` (#1377) wrote the row and `e67b2a4ba` (#1427) refined it for W0f, so `docs/FEATURES.md:64` now names the staging device and both accepted residencies, keep-quant and keep-f16. Measured rather than asserted: `git grep 'CPU keep-quant towers only' origin/main -- docs/FEATURES.md` is rc 1 with no output, and the same grep at `5f4eb356e^` is rc 0 with one hit, which is the positive control that makes the empty result absence rather than a wrong pattern. | Kept as a line rather than deleted, because the SURVIVING half is a different question with a different owner and deleting the entry would lose the pointer to it. Whether a per-commit record gate should be satisfiable after its commit is published is [#573](https://github.com/mudler/vllm.cpp/issues/573), owned by `ENG-RECORD-CONFLICT-SURFACES`, and it is open. Changing the walk is checker semantics and needs its own row, spec and red-first evidence, so it was not folded into this row. Nothing in this branch touched `scripts/`. |
| **The CUDA registrar's own probe assembly is still unmutated.** `src/vllm/platforms/cuda.cpp`'s `Registrar` reads `cudaDevAttrPageableMemoryAccess` and `cudaDevAttrIntegrated`, defaults each to 0 on a query failure, and hands the pair to `HostMemoryIsDeviceAddressableFromAttrs`. That call and those defaults compile only in a CUDA build, so nothing on the CPU tier can mutate them. | The RULE they feed is no longer part of this debt: #1378 extracted it and gated it over all four attribute pairs in `test_platform`, and both term-deletion mutations are RED there. What remains is narrower and honest -- the probe calls, the failure defaults, and the registration itself -- and it needs the same `dgx:gpu0` lease as W0e. Named here rather than folded into the W0b bullet, which used to claim more than it had. |
| **G-DISCRETE: validate W1/W2 on a discrete NVIDIA GPU.** The measurement: on a device with VRAM V and `host_memory_is_device_addressable() == false`, load a GGUF whose `*_exps` towers exceed V, with the lane on, and gate (i) token-exactness against the CPU arm on the same checkpoint, (ii) decode-phase `exhausted` delta 0, (iii) peak device allocation <= non-expert remainder + arena. | No discrete NVIDIA GPU is reachable from this project. `dgx:gpu0` is a GB10 where device memory IS host memory, so a device store there exercises the plumbing and not the thing W1 exists for. Recorded rather than implied, because a gate nobody can run is not a gate. |
| **A mutation of W0b's CUDA leg.** `CudaPlatform::host_memory_is_device_addressable` compiles only in a CUDA build, so no CPU-tier gate can invert it. The bullet in `## Now` promised this line and the table did not carry it, which is fixed here. | **Half discharged by W0e and stated as half.** The lane engaged on a real `--device cuda` run — the `[expert-stream] ON` banner printed and the #1123 refusal did not fire — and neither happens unless the probed predicate returned true on the actual CUDA platform, so the leg is now proven REACHED and proven to answer true on a GB10. What is still owed is the negative: a mutation that makes it answer false and shows a gate go red. That needs a CUDA build with a test target, and W0e built with `-DVLLM_CPP_BUILD_TESTS=OFF` because the lease was for the measurement. |
| **A zero-copy device filler (GPUDirect Storage / `cuFile`).** | W1 ships the staging bounce by choice, for the reasons in its design note. The measurement that would justify replacing it — a device-arm decode where the H2D leg is a measurable fraction of fill time — does not exist until W1 has run somewhere. |
| ~~**The CUDA arm loads and then exhausts the box in its first forward, so this row still has no GPU number.**~~ **CLOSED by W0f**, 2026-08-19 ([#1299](https://github.com/mudler/vllm.cpp/issues/1299)): the non-expert weights were resident twice on a unified part, and `ResidentWeight` now aliases the host bytes where `host_memory_is_device_addressable()`. The same checkpoint reaches **32/32 decode steps** at peak RSS 97.75 GiB. | Kept as a line rather than deleted, because the entry recorded a diagnosis as well as a debt and the diagnosis held: a 0.15 GiB arena failed where an 18.55 GiB one did, and the growth was `RssAnon` while `RssFile` stayed flat, which is what pointed at the dense remainder rather than at the lane. What it got wrong was the scope call -- "not fixable inside this row's scope" -- and W0f fixing it in one branch is the correction. What is NOT closed is the GPU NUMBER: G0-CORRECT fails, so G0-SPEED stays VOID and no rate is published. |
| ~~**The CPU arm's streaming decode figure is still VOID.**~~ **CLOSED by W0e**, 2026-08-18: streaming-ON decode on a live cache is **11.05 s/token** steady at 4000 slots, rep 2's median with rep 1 at 11.22, and the decode-phase `exhausted` delta is 0 in the same run. See `## Evidence`. | Kept as a line rather than deleted because `docs/BENCHMARKS.md:8` still carries the parent row's VOID (#912 F1) text for `ENG-EXPERT-STREAM`, which owns that row's own re-measure. This row measured its own denominator and is no longer waiting on one. |
| **A ratified gate for a two-arm comparison whose greedy path is a coin flip.** The measurement that would settle it: over N prompts, the distribution of top-2 margins at each step, and the fraction of steps whose margin is below the arms' measured arithmetic spread. **This entry's own premise is now in doubt, and it is recorded as such rather than deleted.** | W0e MEASURED the margin at the divergent step (0.264709 logits, 1.4 %) and one step later (0.022802, 0.1 %), which was read as the token-exact gate failing on ties rather than on a defect. **W0g weakens that reading twice.** The arms select different experts from the first MoE block, so the two sampler inputs are not the same distribution and a margin measured on one arm does not bound the disagreement. And the CUDA continuation degenerates into a mechanical recursion after the 8 tokens the arms share, which a coin flip between two equally good tokens does not produce. Ratifying a distributional gate on this evidence would ratify a possible defect, and ratifying one at all is exactly the decision `AGENTS.md` reserves for an explicit act — "use an explicitly ratified distributional gate only when the oracle's greedy decode is non-deterministic" — and it is the operator's, not this row's. Until it is taken, G0-CORRECT stays FAILING and G0-SPEED stays VOID, which is the conservative reading and the one that cannot publish a wrong number. **The measurement that would inform the decision is now scoped as W0h**, [cuda-arm-degradation-experiment.md](cuda-arm-degradation-experiment.md) / [#1736](https://github.com/mudler/vllm.cpp/issues/1736), which reports DEGRADED, NOT-DISTINGUISHED or UNDETERMINED against a rule written before the run. W0h does not ratify anything and does not recommend ratifying anything. |
| **WHAT the divergence IS. It is expert ROUTING and not sampling, and its CAUSE is still unnamed.** W0g ran the two-arm dump this entry asked for and moved the question upstream. At source `cffe59b` the arms already select different experts in the FIRST MoE block of the FIRST forward, eight tokens before any emitted token differs, and they differ there in the router GEMM INPUT rather than in anything the router does with it. THREE causes are now EXCLUDED by measurement, and a fourth check is a SAMPLE rather than an exclusion. The three: the W0f host alias, on the algo-identity probe that ran on `dgx:gpu0` as well as on `thor:gpu0` (`rc` job `7c7a05e9-be87-48f4-94ae-1bbe0340f063`, `NVIDIA GB10 sm_121`, cuBLASLt 130101, 12/12 identical selection, 12/12 bit-exact output, `PROBE_FAILURES=0`); the router GATE weights, whose FNV fingerprint is identical on all 184 dump records of both arms; and the EMBEDDING TABLE, whose output is bit-identical on the two arms. **The fourth check is the top-k implementations, and it is NOT one of the three**: 0 deviations from a plain lowest-index-wins rank over 5 of the 552 token-rows each dump holds, 0.91 %, which is a sample and not the population. Numbers in [`../benchmark-record.md`](../benchmark-record.md) under `ENG-EXPERT-STREAM-DEVICE W0g`. | Excluding three causes is not identifying a fourth, and nothing here may present it as one. The top-k check is not counted among the three, because 5 of 552 token-rows is a sample. **The expert projections, the attention weights and the norms are NOT exonerated: none was fingerprinted.** The evidence is CONSISTENT with bf16 reduction-order accumulation across two genuinely different GEMM kernels, and consistency is not attribution. **The fourth exclusion is the EMBEDDING TABLE, and it is a probe this entry used to carry as queued and unrun.** Run C, `dgx:gpu0` under an `rc hold` that released cleanly, branch `task/1299-embed-dump` at `0544b6224`, `VT_EMBED_DUMP` over four call sites and self-bounded at 8 records: both arms' embedding output is 81,940 bytes with the same sha256 prefix `3f81114a87a0774e84086fe4` and **0 of 40,960 bf16 values differ**, with non-emptiness and equal element counts asserted before the comparison. With Run B's matching router gate fingerprints the WEIGHTS side is now closed at both ends of the stack. **The exoneration stops at the embedding table**: the expert projections, the attention weights and the norms are still unfingerprinted, this removes one candidate rather than naming a cause, and the divergence is not benign. **The next traceable step is now bounded** and is carried as its own entry below: localize where inside block 0's attention and dense path the two arms first differ. Until it runs, G0-CORRECT stays FAILING and G0-SPEED stays VOID. **A second, independent step is scoped as W0h**: whether the CUDA arm is WORSE rather than only different, which localization does not answer and which no arm-against-arm comparison can answer at all. See [cuda-arm-degradation-experiment.md](cuda-arm-degradation-experiment.md) and [#1736](https://github.com/mudler/vllm.cpp/issues/1736). The two are ordered by neither: localization names a cause, W0h says whether the effect matters. |
| **WHERE inside block 0 the two arms first differ.** The measurement: with `--device cpu` and `--device cuda` on `Qwen3.8-2.4T-A95B UD-Q1_0` and prompt ids `760,6511,314,9338,369`, dump the block-0 intermediates between the embedding output and the first MoE router input, one prefill per arm, and name the FIRST tensor whose values differ and the operation that produced it. | **This is a BOUNDED interval, and that is what Run C bought.** Embedding-out is bit-identical and router-in is not, so the first differing operation lies between them, inside block 0's attention and dense path. The instrument is the same shape as the two that already exist: an env-gated, observer-only dump that writes after each value is computed and reads none back, so an unset variable leaves the forward instruction-identical. It needs the same `dgx:gpu0` lease as W0e. This entry replaces the embedding probe as the next traceable step, and it is tracked by [#1299](https://github.com/mudler/vllm.cpp/issues/1299) under the owning row. |
| **The stalled decode step in each of Run A's steady windows was never investigated.** The measurement: re-run the two arms and capture, for the step that stalls, whether the time is in the slot-cache fill, in host paging, or in the GEMM, so the step is attributed rather than dropped. | Run A's steady window holds one step at 26.84 s on CPU and one at 87.32 s on CUDA against medians of 9.09 and 4.72. The record states the medians as the honest figure and forbids quoting either maximum, which is the correct conservative reading and is NOT an explanation. The steady window is steps 4 to 32, which is **29 samples**, and the two arms are not alike: the CUDA stall is **18.50x** its median and the CPU stall is **2.95x** its own. Either is a real periodic cost the median hides or an artifact of the box, and this row does not know which. Named here because the change that measured it declared the debt in prose and no `## Owed` entry carried it. It needs the same `dgx:gpu0` lease as W0e. |
| **Run A was taken at a pinned application clock and needs a repeat at the full clock, using the instrument this tree already ships.** The measurement: re-run both arms with `tools/bench/gpu_clock_state.py` asserting the clock state on both sides, and record whether the medians move. | The run sat at 2418 MHz against a 3003 MHz maximum, discovered after the fact, so no figure from it is clock-controlled. **The instrument that exists to prevent exactly this went unused and unrecorded**: `tools/bench/gpu_clock_state.py` and `.agents/specs/bench-assert-clock-state.md` (`BENCH-ASSERT-CLOCK-STATE`, [#543](https://github.com/mudler/vllm.cpp/issues/543)) were written for the rule that a ratio may not be quoted without the clock it was measured at, and this run quoted neither the clock nor the tool. That is the finding, not the clock value. Nothing published rests on the medians today because G0-SPEED is VOID, so this is owed before any comparison uses them and not before the record stands. |
| **The Qwen3.8 model guide carries a MECHANISM expectation that this row's own unpublishable Run A points against, and the guide MAY NOT be edited toward that number.** `docs/models/qwen3-8-2-4t.md` tells an operator that "a CUDA arm slower than the CPU arm remains a real possible outcome", on the ATS penalty for device access to host-resident weights and the ~6.95 GB per token this lane reads that way. Run A's CUDA median is lower than its CPU median. | **The guide stays as written, and this entry exists so the next reader does not "discover" the contradiction and repair it in the wrong direction.** The guide's claim is about a MECHANISM, and the mechanism is unchanged. The only thing pointing the other way is a median carrying three disqualifications this row wrote itself: G0-SPEED is VOID behind a failing G0-CORRECT, the run sat at a 2418 MHz pin against a 3003 MHz maximum, and an uninvestigated 87.32 s stall sits inside the CUDA arm's own 29-sample steady window. A number in that state does not overturn a mechanism claim privately, let alone publicly. The guide has been narrowed to "No published figure bounds this either way", which stops it reading as a prediction and leaks nothing. **What would settle it is the full-clock repeat named directly above plus a G0-CORRECT pass, and until BOTH land no edit to that sentence may cite Run A.** |
| **The top-k exclusion rests on 5 of 552 token-rows and must be re-derived over all of them.** The measurement: re-rank every one of the 552 token-rows in each Run B dump from that arm's own logits by a plain lowest-index-wins rank, and report the deviating count with its denominator. | The dumps hold 92 blocks x 2 calls, a 5-token prefill and a 1-token decode step, which is 552 token-rows per arm. The check that was run covers 5 of them, 0.91 %, consistent with record 0 alone, and three surfaces then wrote the universal "both top-k implementations are correct" from it. Those surfaces now carry the denominator. **This needs no lease and no new instrument**, only the two dumps that already exist, which is why it is cheap debt rather than a blocked one. Until it runs, the top-k exclusion is a sample result and may not be quoted as a property of the implementations. |
| **The CUDA arm's own top-2 margin at the divergent step.** | The scratch instrument that reads `logits` in the completion callback SIGSEGVs on the CUDA arm (`SCRIPT_EXIT=139`). **WHY IT FAULTS IS UNMEASURED.** An earlier draft of this row wrote "almost certainly because the pointer it is handed there is not host memory on that arm", and that is a hypothesis, not a reading: nothing printed the pointer, nothing asked `cudaPointerGetAttributes` about it, and no fault address was recorded. In a change whose central risk is handing device kernels host pointers, a segfault whose cause was guessed at is exactly the finding that must not be dismissed — so it is recorded as unmeasured rather than as explained. **This entry's justification rests on the premise W0g withdrew, and the entry survives it.** It used to read "the CPU arm's margin is enough to establish the near-tie (`303` is its own runner-up)", which is why the CUDA side was left unmeasured. W0g shows the arms are not sampling the same distribution, so one arm's margin establishes nothing about the pair and the CUDA margin is now MORE wanted rather than less. The next lease should print `cudaPointerGetAttributes(logits)` in that callback before anything else. |
| **No CI gate reaches the alias branch through a production entry point.** `test_expert_stream_wiring` enters `Qwen3_5Model::Forward` and the reachability mutation reds it, but it runs on the **CPU** device, where `ResidentWeight` returns at the `is_cpu()` early return roughly ninety lines above the alias branch. In CI the branch is reached only through `detail::StageWeightForTest`, a test-only seam. | Deliberate, and this is the entry `## Nothing lands dead` requires for it. The branch is selected by `needs_weight_staging() && host_memory_is_device_addressable()`, and no CPU tier can register a platform that answers both — a real one exists on exactly one machine this project can reach. The device evidence is real and is the stronger of the two (the W0e run entered the branch **43,501 times** through `Qwen3_5Model::Forward` on `dgx:gpu0`); it is simply not repeatable in CI. Closing this means either a GPU CI lane on a probed-capable part, or a production entry point that a fake staging platform can drive end to end. It is owned by `ENG-EXPERT-STREAM-DEVICE` and tracked by [#1299](https://github.com/mudler/vllm.cpp/issues/1299) until either lands, and that pair is named in the landing commit body and the pull request body as well as here, because `## Nothing lands dead` requires all three and the spec alone is not the disclosure. |
| **The family-wide copy of this change: `include/vllm/model_executor/models/dense_attn_block.h`'s `ResidentWeight` still stages unconditionally.** The measurement: on a host-addressable staging platform, load any of the ~50 models that include that header and show peak resident bytes falling by the model's weight size, with tokens unchanged. | W0f deliberately changes only `qwen3_5.cpp`'s PRIVATE copy, which is the one that governs `Qwen3.8-2.4T-A95B UD-Q1_0` (that file kept its own helper; the header's copy is not on the Qwen3.5 path). The header's version is reached from `ModelRegistry::Forward` for every model that includes it, so extending it is not dead code — but nothing on a CPU tier can drive one of those forwards on a staging platform, so the extension would land with its reachability argued rather than gated, across ~50 architectures at once. That is a scope and a review question, not a line of code, and it gets its own row. |
| **The missing CPU-platform gate on `p.quant_repack` itself ([#1320](https://github.com/mudler/vllm.cpp/issues/1320)).** The measurement: `elem_kn_repack` is resolved with `CurrentPlatform().device_type() == kCPU` and `quant_repack` is not, so a device load can still perform a CPU-only transform and be caught afterwards instead of never doing it. | W0f fixes the CONSEQUENCE in flow — a named refusal on both arms of `ResidentWeight`, red-first and mutation-proven — because that is the small and clear part. Moving the gate into the loader policy changes what a GGUF load DOES on a device rather than what it refuses, which is `QUANT-GGUF-KEEPQ-LOADER`'s semantics and needs its own red-first evidence. |
| **`.agents/specs/expert-streaming.md`'s `## Owed` entry for #1124 still names no owning row ID.** | Not edited here on purpose; PRs #1200 and #1216 both edit that file. One-line follow-up once both land. |
| **W1 may land UNREACHED if it is split from W2.** | The recommendation is one pull request. If a split is chosen, the commit body and the PR body must name what is unreached and name W2 as the owning wiring, per `## Nothing lands dead`. |
