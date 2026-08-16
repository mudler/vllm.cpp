# Four-way GB10 benchmark on Qwen3.8-27B

| Field | Value |
|---|---|
| Issue | [#979](https://github.com/mudler/vllm.cpp/issues/979) |
| Owning rows | [`BACKEND-GATE-CUDA-VLLM`](../backend-matrix.md), [`BACKEND-GATE-CUDA-SGLANG`](../backend-matrix.md), [`BACKEND-GATE-CUDA-LLAMACPP`](../backend-matrix.md) |
| Roadmap | `ROAD-V1-A` (the perf and SGLang floor lane) |
| Umbrella | [competitive-benchmarks.md](competitive-benchmarks.md) fixes the workload vocabulary and the per-backend leaf-spike contract |
| Sibling leaf | [cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) owns the three-arm cache-neutral SGLang gate on the Qwen3.6 snapshots and stays authoritative for it |
| Subject | `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, `unsloth/Qwen3.8-27B-NVFP4` @ `a767244d`, `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23` |
| Host | `dgx.casa`, GB10 sm_121a, one `flock $GPU_LOCK` for the whole series |
| Role | helper, branch `row/BENCH-QWEN38-27B-FOUR-WAY`, base `598226e962ddd4a83292e3d9264bbea9f41603d2` |
| Status | `SPIKE`. Scoping and record reconciliation only. No number is measured by this spec. |

## 0. Why this is a new leaf and not an edit to the SGLang spike

[cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) is a three-arm
spike over SGLang, vLLM and ours, on the Qwen3.6 27B and 35B NVFP4 snapshots,
pinned to SGLang v0.5.13 for its P1 harness. Three things here fall outside it.

1. **The subject changes.** Qwen3.8-27B is the checkpoint [#915](https://github.com/mudler/vllm.cpp/issues/915)
   gated. Every checkpoint-scoped artifact in the sibling spike is bound to the
   Qwen3.6 snapshots by revision.
2. **A fourth engine joins.** llama.cpp on GB10 CUDA appears in no matrix row and
   in no leaf spike. The sibling spike names three arms and excludes GGUF.
3. **The deliverable is different.** The sibling spike produces a binding
   every-axis gate on one shared checkpoint. This one produces a matrix of pairs,
   each with its own stated common denominator, because at these pins **no single
   quantization is common to all four engines**.

The sibling spike is not superseded and its Qwen3.6 evidence is not reinterpreted
here.

## 1. Scope

| In scope | Out of scope |
|---|---|
| CUDA on `dgx.casa` at the four engines' recorded pins | Any other host, and any comparison across different checkpoints |
| Text-only greedy serving, `POST /v1/completions` | Vision and video inputs, chat templates, tool calling |
| A per-pair common-denominator matrix with an explicit not-comparable verdict | One headline four-way ratio |
| Raw-decode and drafted arms, each declared, never mixed | Any cell that compares a drafted arm against a raw one |
| Record reconciliation of the two stale entries in section 5 | Advancing the SGLang pin, which is separate deliberate work |

## 2. There is no single quantization all four run

Established from each engine's own source at its recorded pin. Nothing in this
table is inferred from a model card or a release note.

| | bf16 | NVFP4 | Q4_K_M GGUF |
|---|---|---|---|
| ours | GATED, [#915](https://github.com/mudler/vllm.cpp/issues/915) | owed, [#821](https://github.com/mudler/vllm.cpp/issues/821) | text-only loader only, no `clip` projector, [#821](https://github.com/mudler/vllm.cpp/issues/821) |
| vLLM `555967922` | yes | yes | **absent from the tree** |
| SGLang `f63458b5` v0.5.15 | yes | yes | stack present, this architecture unreachable |
| llama.cpp `237ad9b96` | convertible, not its representative arm | no | its native arm |

### 2.1 vLLM at our pin has no GGUF path at all

`6635279d8` (vllm#39612, 2026-06-13) removed the whole surface and moved it out
of tree. At `555967922`:

- `vllm/model_executor/model_loader/__init__.py:33-48` lists every load format and
  `gguf` is not among them, and `_LOAD_FORMAT_TO_MODEL_LOADER` at `:49-66` agrees.
- `vllm/model_executor/layers/quantization/__init__.py:12-46` has no `gguf` entry
  in `QuantizationMethods`.
- `vllm/config/load.py:30-58` documents the same set and ends with "Other custom
  values can be supported via plugins".
- The replacement is declared at `setup.py:1300` as
  `"extra-quant": ["vllm-gguf-plugin>=0.0.2"]`, and documented at
  `docs/features/quantization/gguf.md:6-13`.
- The only two case-insensitive `gguf` hits left in `vllm/` are comments, at
  `model_executor/models/qwen2_moe.py:499` and `lora/layers/utils.py:69`.

That plugin is not pinned by this project and has no `.agents/oracles/` record.
Consequence: **the vLLM-versus-llama.cpp pair has no common quantization and is
recorded not-comparable.** Naming the plugin as a pinned oracle would be a
separate decision with its own gateability measurement, not a step inside this
campaign.

### 2.2 SGLang's GGUF blocker is deeper than the alias table

The alias table is real and is exactly two entries wide. `python/sglang/srt/model_loader/loader.py:2129-2142`:

```python
        if model_type == "cohere":
            model_type = "command-r"
        elif model_type == "qwen3_moe":
            model_type = "qwen3moe"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
```

`qwen3_5` misses because `gguf.MODEL_ARCH_NAMES` spells the family `qwen35`,
which is the same missing underscore the `qwen3_moe` entry exists to paper over.
**Adding the alias would not make it load.** Three further blockers, each
independent:

1. `GGUFConfig.get_quant_method` (`layers/quantization/gguf.py:106-124`) handles
   `LinearBase`, `VocabParallelEmbedding` and `FusedMoE`, and returns `None` for
   everything else. There is no gated-delta-net state path.
2. `Qwen3_5GatedDeltaNet` holds `A_log` and `dt_bias` as bare parameters with
   custom sharded loaders (`models/qwen3_5.py:250,253,257-258`). In
   `_get_gguf_weights_map` (`loader.py:2149-2153`) an unresolvable name yields
   `None`, producing colliding `None.weight` keys, and there is no guard.
3. `conv1d` is declared a `ColumnParallelLinear` and reshaped to three dimensions
   (`models/qwen3_5.py:195-204`). Being a `LinearBase` subclass it would receive a
   `GGUFLinearMethod` and be **silently wrong** rather than refused, which is the
   worst of the three failure modes.

There is also an earlier gate at
`utils/hf_transformers/config.py:237-239`, which with a newer `transformers`
passes and lets the failure land on `loader.py:2142`.

Consequence: **the ours-versus-SGLang GGUF pair is recorded not-comparable**, and
the SGLang-versus-llama.cpp pair with it.

### 2.3 llama.cpp is the only comparator that runs the GGUF arm

`LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` are registered at
`src/llama-arch.cpp:41-42`, with per-architecture handling at `:890-891,912-913`.
The projector side has `PROJECTOR_TYPE_QWEN3VL` at `tools/mtmd/clip-impl.h:330`.

Our side loads the dense `qwen35` language file and maps it onto the registered
wrapper at `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:845,856`, and
has no `clip` projector path. Since this campaign is text-only that is sufficient
for the language arm, and the projector stays owed under
[#821](https://github.com/mudler/vllm.cpp/issues/821).

### 2.4 The resulting pair matrix

Each cell states the common denominator, or says not-comparable and why.

| Pair | Common denominator | Verdict |
|---|---|---|
| ours vs vLLM | bf16 `1d4bf0f2` | **comparable now.** c4 already reads 0.963x output throughput and 1.008x median ITL. c1 and c8 withheld under [#931](https://github.com/mudler/vllm.cpp/issues/931) until re-run. |
| ours vs vLLM | NVFP4 `a767244d` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821).** vLLM's ModelOpt and compressed-tensors NVFP4 paths are present. Ours is owed. |
| ours vs SGLang | bf16 `1d4bf0f2` | **comparable after a load preflight.** SGLang registers `Qwen3_5ForConditionalGeneration` at `models/qwen3_5.py:1633,2179`, and Qwen3.8 declares that architecture, but no SGLang load of this snapshot has been recorded. |
| ours vs SGLang | NVFP4 `a767244d` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821) and a load preflight.** |
| ours vs llama.cpp | Q4_K_M `fe1e2a23` | **comparable after [#821](https://github.com/mudler/vllm.cpp/issues/821).** The one pair where GGUF is the shared native arm. |
| vLLM vs SGLang | bf16 and NVFP4 | comparable in principle, and informational here. Neither is our engine, so no gate turns on it. |
| vLLM vs llama.cpp | none | **NOT COMPARABLE.** vLLM has no in-tree GGUF and llama.cpp has no NVFP4. bf16 GGUF would be a conversion on one side and is non-binding under the umbrella spike's fallback rule. |
| SGLang vs llama.cpp | none | **NOT COMPARABLE.** Same reason, plus section 2.2. |

## 3. Drafted or raw, declared per arm

SGLang published **38.28 tok/s decode on DGX Spark** for Qwen3.8-27B. Confirmed
as a single-source project claim on X, whose own wording is "our NVFP4 plus
DSpark", so it is a **speculative-decoding** result. An NVIDIA developer-forum
thread reports 34 to 38 tok/s for the same SGLang plus NVFP4 plus DSpark recipe.

**UNVERIFIED and not to be repeated as fact until sourced:** the drafter's
parameter count, whether TTFT is excluded from that figure, the batch size, and
the reported 16.6 to 46.7 tok/s spread of independent reproductions. None of those
appear in any artifact this project holds. This project's own record contains
**zero** occurrences of the string `38.28`, which is exactly why it is written
down here with its provenance attached rather than carried forward as a target.

Our binding quantized-27B cell is 10.756 against vLLM's 11.250 at c1
(`docs/BENCHMARKS.md:96-97`, Qwen3.6-27B NVFP4). That is a **raw** decode number.
Dividing 38.28 by it compares a drafted arm against a raw one and is refused by
this spec.

**Rule.** Every arm declares `drafted` or `raw` in its manifest before it runs. A
cell whose two arms disagree on that field is void, not a ratio. We hold the
technique on our side: `SPEC-DFLASH` is `DONE`, and `SPEC-DSPARK` is `ACTIVE` with
W1 through W8 landed and GPU-gated
([dspark-spec-decode.md](dspark-spec-decode.md)).

**The 38.28 configuration is not reachable at our SGLang pin.** v0.5.15
`f63458b5` ships DFlash, EAGLE, ngram and frozen-KV MTP under
`python/sglang/srt/speculative/`, and a repo-wide search for `dspark` returns
nothing. The announcement says Day-0 support, so that code postdates the pin.
A drafted SGLang arm therefore requires **advancing the SGLang oracle pin first**,
under `.agents/oracles/sglang.md`, with every affected row reconciled. It is not
a substitution made inside a measurement run.

Until that pin moves, the SGLang arms in this campaign are `raw`, and the only
honest statement about 38.28 is that it belongs to a configuration we have not
pinned.

## 4. What changing the subject costs

| Carries over unchanged | Must be re-established for Qwen3.8-27B |
|---|---|
| The `serve-low` workload shape: concurrency 1, 2, 4, 8, 16, 1024 in and 128 out, ignore EOS, seed 0 | Every checkpoint file manifest and `sha256sum` set, for all three artifact families |
| The whole P1 harness under `tools/bench/*serve_low*` and `scripts/dgx-sglang-low-concurrency.sh` | The deterministic token-ID corpus, because the tokenizer snapshot changes with the checkpoint |
| The vLLM parity pin `555967922`, unchanged by this campaign | The four-way tokenizer agreement preflight over every prompt |
| The one-lock, sequential-arm, teardown-verify execution discipline | Any checkpoint-equivalence proof, which was written against the Qwen3.6 snapshots |
| The `require_complete_request_set` refusal and the `failed == 0` precondition | Every SGLang and llama.cpp load and quantization-path classification |
| The clock-pinning and idle-host protocol | The vLLM production-config denominator for this checkpoint |

The token gate itself does **not** need redoing: [#915](https://github.com/mudler/vllm.cpp/issues/915)
closed it for bf16, 4 of 7 prompts strict and all three first divergences exact
fp32 ties against the pinned oracle. The NVFP4 and Q4_K_M token gates are owed
under [#821](https://github.com/mudler/vllm.cpp/issues/821) and are preconditions,
not products, of this campaign.

## 5. Record reconciliation, with evidence

### 5.1 `BACKEND-GATE-CUDA-SGLANG`, `BLOCKED` on a dependency that closed

`backend-matrix.md:260` recorded `BLOCKED on SERVE-ASYNC-LLM` with the evidence
cell "no binding run; HTTP TTFT/ITL cannot be measured honestly yet". Four
independent lines of evidence say that reason no longer holds.

1. **The production path streams incrementally.** `serving_completion.h:9-11` says
   the live pull-based `SseStream` over `AsyncLLM` is the production path and the
   buffered `LLMEngine` constructor is a test seam. `api_server.cpp:971-981` takes
   `result.sse_stream` and drives `set_chunked_content_provider` off
   `SseStream::next`, one chunk at a time.
2. **The harness enforces it rather than assuming it.**
   `tools/bench/run_serve_low.py:296-310` refuses a probe that produced no
   token-bearing event, refuses a chunk count other than the requested completion
   length, refuses `first_chunk_s >= total_s`, and refuses a spread below a
   configured floor.
3. **It was demonstrated on hardware.** The 2026-07-28 floor run
   (`sglang-matrix.md`, `CLAIM-SGLANG-PERF-BENCH`) measured our c16 mean TTFT at
   2980 ms against a mean ITL of 154.4 ms over 128 tokens. First byte therefore
   preceded completion by roughly twenty seconds. A buffered server cannot produce
   that shape.
4. **The record says so elsewhere.** `engine-matrix.md:207` carries
   `SERVE-ASYNC-LLM` as `GATING` with live completion and chat SSE, disconnect
   abort and deterministic c32 capacity, and `async-metrics.md:196` records the
   `SERVE-ASYNC-LLM` frontend plus `ENG-CORE-BUSY-LOOP` as `DONE`.

[#931](https://github.com/mudler/vllm.cpp/issues/931), landed as `638eba27f`,
strengthens this rather than establishing it: it defaults `VT_SERVER_SSE_PING_S`
to 0 (`serving_utils.h:40`, `serving_utils.cpp:254,278`) so vLLM's own bench
client can parse our stream, and it adds
`require_complete_request_set` (`tools/bench/serve_low_common.py:234`) so no rate
is derived from an incomplete request set.

**New state: `PARTIAL`, not `DONE` and not still `BLOCKED`.** The named blocker is
discharged and partial evidence exists. What is still missing, named:

- The c1, c2 and c4 points. Only c8 and c16 ran, and SGLang c1 measured about
  13.3 seconds per iteration, so three-repetition reproduction there is a real
  scheduling problem and not an oversight.
- The vLLM arm in the same series. The 2026-07-28 run was ours against SGLang
  only.
- The 35B-A3B arm.
- The SGLang token-ID cross-check. `SGLANG-ORACLE-CORRECT` is still `INVENTORIED`,
  so SGLang binds as a floor only for a model whose correctness gate passed.
- Paired warmup-excluded nsys traces at c1 and c16.
- Every point on Qwen3.8-27B, which is this campaign's subject and has no SGLang
  evidence at all.

### 5.2 The SGLang oracle recorded `gateable = no` after it had already run here

`.agents/oracles/sglang.md` said "no SGLang run has been recorded on this
project's hardware" and "Source has been read; nothing has been executed".
`.agents/sglang-matrix.md:162` records the opposite: on 2026-07-28 the
`lmsysorg/sglang:v0.5.15-cu130@sha256:d0a667e` arm64 image was pulled and ran
`unsloth/Qwen3.6-27B-NVFP4` @ `890bdef7` on GB10 sm_121a with CUDA graphs
captured, three repetitions at c8 and c16, both arms emitting exactly 80 by 128
output tokens with zero errors. `docs/STATUS.md:180` carries the same
measurement. `docs/BENCHMARKS.md:485` still said "SGLang floor arms | Never ran".

AGENTS.md sets the bar at "demonstrably builds and runs the model". The image
needed no from-source build and it ran the model. **`gateable` moves to `yes`**,
with `.agents/sglang-matrix.md` as the evidence path. This discharges the SGLang
half of the three gateability debts [#647](https://github.com/mudler/vllm.cpp/issues/647)
holds open.

**The 2026-07-28 run predates the #931 fix and is not voided by it.** The
keepalive fires only after 15 seconds with no output on a request
(`serving_utils.h:42` and the `AssignSseWaitResult` call sites). That run's worst
observed p99 TTFT was 7220 ms at c16 and 3589 ms at c8, with ITL near 154 ms, so
no frame could have been emitted, and the harness independently recorded zero
errors on every leg. This is stated because a silent assumption in the other
direction is precisely the failure #931 documents.

### 5.3 The llama.cpp-on-CUDA comparator had no owning row

`BACKEND-GATE-CPU-LLAMACPP` is the CPU floor. `BACKEND-GATE-CUDA-LLAMACPP-LEGACY`
is scoped to Pascal, Volta and Turing, which is where vLLM has no entry at all.
Neither covers llama.cpp CUDA on a current card, which is what this campaign's
fourth arm is. `bench-27b-five-way.md` already hit this and listed the arm as
"building" with no row behind it. `BACKEND-GATE-CUDA-LLAMACPP` is added
`INVENTORIED`, with no run and no number.

## 6. Protocol

Everything here is inherited from [competitive-benchmarks.md](competitive-benchmarks.md)
and [cuda-sglang-low-concurrency.md](cuda-sglang-low-concurrency.md) unless it is
stated as a change.

- One `flock $GPU_LOCK` for the whole multi-arm series. Engines run strictly
  sequentially with a teardown-verify between legs, because a 27B bf16 resident
  set is about 55 GB against a 119 GB unified pool and two engines cannot coexist.
- vLLM's **production graphed** configuration is the denominator. `--enforce-eager`
  never appears in a measured arm.
- `gpu_memory_utilization` reserves host RAM on GB10 and has hard-rebooted this
  box. Keep it low and never run a second GPU consumer alongside.
- Clocks pinned and recorded per leg, with an always-fires reset trap.
- Three repetitions per point, interleaved, with spread and coefficient of
  variation reported alongside every median.
- `failed == 0` asserted on every leg, and every rate derived through
  `require_complete_request_set`.
- Each cell names the quantization on **both** sides and the `drafted` or `raw`
  field on both sides, or it is not a cell.
- Raw per-repetition JSON, server logs, manifests, commands, engine revisions and
  contention state retained under the claim's evidence root.

## 7. Gates

| Gate | Pass condition |
|---|---|
| Pins | Every engine resolves to its recorded pin, and each pin has an `.agents/oracles/` record. A pin advance is a separate reconciled change. |
| Quantization | Every cell states one common denominator on both sides, or carries an explicit not-comparable verdict with the source reason. |
| Correctness | Ours is token-exact against the pinned vLLM oracle on the arm being timed, or falls under the ratified near-tie protocol. A comparator binds as a floor only after its own correctness cross-check passes. |
| Spec-decode declaration | Both arms of every cell declare `drafted` or `raw` and agree. |
| Completeness | `failed == 0` on every leg and no rate derived from an incomplete request set. |
| Streaming validity | First chunk precedes completion and the spread floor holds, per the harness preconditions. |
| Reproduction | Three repetitions, spread reported, unexplained outliers re-run on an idle host. |
| Record | Every accepted or explicitly pending cell reaches `docs/BENCHMARKS.md` and the owning matrix row in the state-changing commit. |

## 8. Risks

| Risk | Handling |
|---|---|
| The 38.28 figure is treated as a target and drives work | Its provenance is written down in section 3 with the unverified parts named. It is never a denominator. |
| Someone adds the two-line SGLang GGUF alias and reports a load | Section 2.2 names three further blockers, one of which fails silently. A load is not evidence. |
| Someone installs `vllm-gguf-plugin` to create a vLLM GGUF cell | That plugin is unpinned and has no oracle record. Pinning it is a separate decision. |
| The SGLang pin is advanced mid-campaign to reach DSpark | Forbidden here. Advancing it reconciles every affected row first. |
| bf16 GGUF is substituted to manufacture a vLLM-versus-llama.cpp cell | A converted checkpoint is `converted-nonbinding` under the umbrella spike's fallback rule and cannot produce a ratio. |
| A prior Qwen3.6 artifact is reused for the Qwen3.8 subject | Section 4 lists exactly what carries over. Manifests, corpora and equivalence proofs do not. |
| Two engines resident at once on the unified pool | Sequential arms with teardown-verify, one lock, memory returned to a recorded baseline between legs. |

## 9. Tests

This spec adds no product code and therefore no product test. The executable
obligations it depends on already exist and are named so a reviewer can mutate
them:

- `tests/tools/test_request_set_completeness.py` pins the incomplete-request-set
  refusal.
- `tests/vllm/entrypoints/openai/test_sse_keepalive.cpp` pins the #931 frame
  behaviour.
- `tools/bench/run_serve_low.py:296-310` carries the incremental-streaming
  precondition, and `tests/tools/test_serve_low_client.py` covers the client
  contract.
- `tests/scripts/test_check_oracle_pins.py` pins the `gateable` semantics that
  section 5.2 exercises.

## 10. Evidence required before any number is quoted

1. Sorted `sha256sum` manifest for every resolved file of all three Qwen3.8-27B
   artifact families.
2. Startup log, resolved architecture, resolved quantization and weight-loader
   warnings for every engine and every arm.
3. Four-way tokenizer agreement on every corpus prompt, with stored token IDs and
   hashes.
4. Native output IDs per engine, never a detokenize-and-retokenize round trip.
5. Per-leg clock state, thermal state, idle proof and contention state.
6. Raw per-request arrays for all three repetitions, plus the recomputed
   percentiles and the formula version.
7. The `drafted` or `raw` declaration and, when drafted, the drafter identity,
   revision and k.

## 11. Stop conditions

- Stop and report if any pair's common denominator cannot be established from
  source. Record not-comparable. Do not substitute a nearby quantization.
- Stop if an arm's `failed` count is non-zero. That leg is void.
- Stop if reproducing a point needs a pin advance. Reconcile the pin in its own
  change.
- Stop if the host is not idle, or if a co-tenant appears mid-series. Discard the
  series rather than annotate it.

## Now

`SPIKE`. This spec, the issue, and the three record corrections in section 5 are
the whole deliverable. Nothing is measured. `BACKEND-GATE-CUDA-SGLANG` moves
`BLOCKED` to `PARTIAL` because its named blocker is discharged and partial
evidence exists. `BACKEND-GATE-CUDA-LLAMACPP` is added `INVENTORIED` with no run.
The SGLang oracle moves to `gateable = yes`.

## Owed

- [#979](https://github.com/mudler/vllm.cpp/issues/979) owns this campaign and is
  listed here so the index row and this spec agree.
- [#821](https://github.com/mudler/vllm.cpp/issues/821) owes our NVFP4 and Q4_K_M
  arms for this checkpoint, including the separate `clip` projector. Three cells
  in section 2.4 are blocked on it.
- [#915](https://github.com/mudler/vllm.cpp/issues/915) owes the c1 and c8
  re-measure on a binary carrying the #931 fix.
- The SGLang token-ID correctness cross-check, `SGLANG-ORACLE-CORRECT`, is
  `INVENTORIED` in [sglang-matrix.md](../sglang-matrix.md).
- Advancing the SGLang pin past v0.5.15 to reach DSpark, if a drafted SGLang arm
  is ever wanted. Unowned today and deliberately not started here.
