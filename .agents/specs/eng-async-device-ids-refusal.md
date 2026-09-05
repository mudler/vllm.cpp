# ENG-ASYNC-DEVICE-IDS-REFUSAL — make `device_token_ids` refusable

Row: `ENG-ASYNC-DEVICE-IDS-REFUSAL`
Issue: [#2710](https://github.com/mudler/vllm.cpp/issues/2710)
Precedents: [#2544](https://github.com/mudler/vllm.cpp/issues/2544),
[#2323](https://github.com/mudler/vllm.cpp/issues/2323),
[#1305](https://github.com/mudler/vllm.cpp/issues/1305),
[#2618](https://github.com/mudler/vllm.cpp/issues/2618)

## Now

ACTIVE.

## Scope

This row is `## Owed` O1 of [`eng-async-device-ids-2544.md`](eng-async-device-ids-2544.md):
"the ENFORCEABLE contract #2544's own prose asks for: a forward that ignores a
live `device_token_ids` refused BY NAME rather than silently generating from the
previous step's identifiers." #2544 owned it and is closed, so #2710 takes it.

`ModelForwardInput::device_token_ids` is ADVISORY. When it is non-null the step's
input ids live in that device buffer and `token_ids` is stale for decode rows,
because the async runner's combine splices each decode row's sampled token on the
main queue and deliberately never writes it back. `token_ids_cpu` is
zero-initialised, so a forward that ignores the field decodes from **token id 0**
at every step after the first, at `rc=0`, with plausible-looking output.
`async_device_mirror()` is DEFAULT-ON for CUDA, integrated parts included.

Five architectures were caught ignoring it; the last four landed in #2544
(`8401a435e`). Nothing stops the sixth.

IN SCOPE:

1. A capability bit `ModelFactory::consumes_device_token_ids`, default FALSE,
   the polarity `consumes_multi_kv` and `stage_on_load` already use.
2. A pure refusal predicate `DeviceTokenIdsRefusalApplies`, and its application
   in `ModelRegistry::Forward` with a message that names the arriving
   architecture.
3. The step-granular staleness fact the refusal needs, computed by the runner
   from the combine's OWN row predicate.
4. Setting the bit true on every registration whose forward demonstrably reads
   the field.
5. Repairing the false sentence in the field's own contract comment.

OUT OF SCOPE: `## Owed` O2 and O5-O7 of the sibling spec (moving the device arm
of the seam, the DeepSeek-V4 device embed, and hardware legs for the three
forwards #2544 wired against a fixture). None of them is a precondition for a
guard, and each is its own unit of work.

## Upstream anchors

There is no vLLM anchor for the FIELD. `device_token_ids` is this project's own
seam for a scheduling shape vLLM expresses with a torch tensor that is always
device-resident: upstream's `input_ids` has no stale host twin, so upstream has
nothing to refuse.

The row predicate this guard is built on IS anchored. It is
`_combine_sampled_and_draft_tokens_kernel`
(`vllm/v1/worker/gpu/input_batch.py:304-406` @ `e24d1b24`), whose
`seq_len <= prefill_len` early-out this tree ports twice, once on each side:

* `src/vt/cuda/cuda_combine_tokens.cu:63-67` (device)
* `src/vllm/v1/worker/gpu/prepare_inputs.cpp:345-351` (host)

The guard reuses that predicate rather than deriving a second one, which is the
whole design constraint below.

## Design

### The shape, mirrored from the sibling channel

`multi_kv` is not advisory: `ModelFactory::consumes_multi_kv` declares the
capability and `MultiKvRefusalApplies` decides, so `ModelRegistry::Forward`
refuses a step a model may not read. This row gives `device_token_ids` the same
two parts.

```cpp
bool DeviceTokenIdsRefusalApplies(const int32_t* device_token_ids,
                                  bool host_token_ids_stale,
                                  bool consumes_device_token_ids);
```

It is a free function for the reason W5 made its sibling one: the decision is
gateable without a model, a runner or a registry, so both polarities can be
pinned directly. An inline copy in `Forward` would be a second derivation, and a
second derivation is the thing that can disagree with the one a test pins.

### Why nullness alone is the WRONG predicate

The obvious mirror is `device_token_ids != nullptr && !consumes`. It is wrong,
and the counterexample is in the tree today.

`ForwardLlamaModelEmbedding` (`llama_embedding_registry.cpp`) is a POOLING
forward. It embeds from host `input.token_ids` and never reads
`device_token_ids`, so a nullness guard would refuse it. But every request it
serves is prefill-only: `seq_len == prefill_len` for every row, the combine
splices nothing, and the host vector is never stale. It works today, on the
default arm, and a nullness guard would take that working path away.

A refusal that fires on a model which actually works is worse than the bug it
prevents. The same argument covers the FIRST step of every ordinary text model,
where all rows are prefill.

So the refusal fires on the condition that a step GENUINELY hands stale ids.

### The staleness fact, and why it is per-STEP

The combine splices row `i` exactly when `seq_lens[i] > prefill_len[idx(i)]`.
Both ports state it in those words. Lift it to the step:

> the host token ids are stale for this step ⟺ ANY row of the step is one the
> combine splices.

`ModelForwardInput` gains `bool host_token_ids_stale`, default FALSE (so every
existing call site is byte-identical), set by the runner beside
`device_token_ids` from `v1::AnyRowSplicedByCombine`.

**The granularity is the load-bearing part of this row.** A per-REQUEST reading
of the same rule — "this request is a prefill row, so it is fine" — lets a mixed
step through for its prefill rows while its decode row is served from stale ids.
That is a refusal whose predicate disagrees with its route predicate, which this
repository has already shipped once: a per-request refusal paired with a per-step
route veto, where every test used `num_reqs == 1` so the two agreed on every test
input and 27 mutations missed it.

The two readings are only distinguishable on a step with more than one request
and a MIXED disposition, so `## Tests` requires exactly that case.

### Not a second derivation

`AnyRowSplicedByCombine` does not re-state `seq_len > prefill_len`. The row
predicate is extracted once, as `v1::CombineSplicesRow`, and BOTH the host
combine (`combine_sampled_and_draft_tokens`, which today writes the condition
inline) and the step-level ANY call it. The host combine's behaviour is unchanged
and its existing test (`tests/vllm/v1/worker/test_combine_tokens.cpp`) keeps
pinning it, which is what makes the extraction safe rather than a rewrite.

The CUDA kernel keeps its own copy, because device code cannot call a host
inline. That copy is pre-existing and this row does not touch it; it is named
here so the reader knows the guard mirrors it deliberately.

### Why the runner computes it and the guard does not

`ModelForwardInput` carries neither `prefill_len` nor a per-request `seq_lens`
vector, and adding them to reach every registered forward would widen the seam
far past what a guard needs. The runner holds both at the site where it already
sets `device_token_ids`, so the fact is computed once, at the one place that
knows it, and travels as one bool.

### The default, and what it costs

`consumes_device_token_ids` defaults FALSE. That is the mechanism: a model added
tomorrow that does not read the field is refused rather than served stale ids. A
model that reads it sets the bit in its own registration, beside its own forward,
so no shared file has to be edited per model — the shared-file lock `## Records`
forbids.

Accepted cost, stated rather than discovered: a REFUSE-stub registration
(`ForwardParakeetRefused`, `ForwardKimiK3ForConditionalGeneration`) leaves the
bit false and will now hit this guard's message instead of its own, on a step
where the mirror is live and a row is spliced. Both refuse; only the wording
changes, and reaching that point at all requires a checkpoint whose own forward
is a `VT_CHECK(false, ...)`.

### The comment repair

The field's contract comment asserts that "a model that ignores it is simply
never given one (the runner only sets it on the discrete-CUDA async path, which
the Qwen3.5 gate vehicle owns)". That sentence is FALSE and is the sentence the
defect hid behind: `execute_model` assigns the field unconditionally for every
registered forward whenever the mirror is engaged, and the mirror is integrated
OR discrete. It is replaced with what the code does and with a pointer to the
guard.

## Classification of every registered forward

Recorded here because the next reader needs it and no other surface holds it.
Each registered forward is one of:

* **(a) CONSUMES** — reads `device_token_ids`. Sets the bit.
* **(b) IGNORES** — embeds from host `token_ids`. Leaves the bit false, and is
  refused on a step that splices a row.
* **(c) INERT / UNREACHABLE** — cannot decode at all. Leaves the bit false;
  neither arm applies.

### How to classify, and why a grep cannot do it

Recorded because it will bite the next reader, and because it has already made
one list wrong. #2544 built its candidate list from a grep and was wrong in BOTH
directions: `kimi_k3` was never affected, and `laguna` cannot be gated at all
(#2618).

**CONSUMPTION IS SPREAD ACROSS THREE DISTINCT SEAMS.** A grep for any one of them
sees roughly a third of the truth:

1. `detail::DeviceTokenIdsScope` — the DEVICE arm, for a forward that already
   uploads its ids to a device buffer and can have the pointer spliced over it.
2. `ResolveHostTokenIds` (`include/vllm/model_executor/models/host_token_ids.h`)
   — the HOST arm, added by #2544 for forwards that gather embedding rows on the
   host and so have no `dst` to splice.
3. A direct `input.device_token_ids != nullptr` read, sometimes in the
   registration's own translation unit (`qwen4_exp_registry`) and sometimes in a
   device file the registration only reaches through a call chain
   (`nemotron_h_device.cpp`, `kimi_linear_device.cpp`) — so the registry file
   itself contains no match at all.

**AND THE MATCHES LIE IN BOTH DIRECTIONS.** `kimi_k3_registry.cpp` matches a grep
for `ResolveHostTokenIds` — inside a COMMENT whose text says there is no such call
in that file. A false positive of exactly the shape that made #2544's list wrong.
Meanwhile `nemotron_h_registry.cpp` and `kimi_linear_registry.cpp` are true
consumers whose registry files match nothing.

So each registration is classified by reading its forward and following the chain
it reaches. The count below was derived that way, in a worktree at the head being
changed — NOT in the shared checkout, which is routinely behind and which lacked
#2544 entirely while this row was being written.

36 registrations. **16 (a)**, **16 (b)**, **4 (c)**.

### (a) CONSUMES — the bit is set, 16

Three distinct consumption seams, so a grep for one of them misses two thirds of
the set:

* `detail::DeviceTokenIdsScope` (the DEVICE arm) — `deepseek_v2_registry`,
  `glm4_moe_lite_registry`, `glm_moe_dsa_registry`, `internlm2_registry`,
  `llama_registry`, `mistral_registry`, `qwen3_dense`, `qwen3_moe_registry`,
  `qwen3_5_dense`, `qwen3_5_moe`.
* `ResolveHostTokenIds` (the HOST arm, added by #2544) —
  `deepseek_v4_registry`, `glm5_next_registry`, `laguna_registry`.
* A direct `input.device_token_ids != nullptr` read in the model's own
  translation unit — `qwen4_exp_registry` (in-file), `nemotron_h_registry` (via
  `nemotron_h_device.cpp`), `kimi_linear_registry` (via
  `kimi_linear_device.cpp`).

### (b) IGNORES — the bit stays false, and these are now REFUSED on a stale step, 16

`commandr`, `dots3_note`, `gemma`, `gemma2`, `gemma3`, `gemma4`, `glm4`,
`granite`, `minicpm`, `minicpm3`, `muse_glimmer`, `olmo2`, `opt`, `phi`, `phi3`,
`stablelm`.

Each reaches `<Model>::ForwardDevice(input.token_ids, ...)` and never reads the
device pointer. **This is the finding, not a side effect**: sixteen registered
architectures decode from token id 0 on the default CUDA arm today, and none of
them is convicted by any gate in this tree because a token gate has nothing to
compare fluent wrong output against. The guard converts all sixteen from a silent
wrong answer into a refusal that names the architecture and the missing
capability. It does NOT port them; each is owed by the row that ports it.

### (c) INERT or UNREACHABLE — the bit stays false and neither arm applies, 4

* `kimi_k3_registry` — both callees are `VT_CHECK(false, kPending)` skeletons
  that `(void)` their token ids (`kimi_k3.cpp`). Its registry file MENTIONS
  `ResolveHostTokenIds`, in a comment saying there is no such call, which is
  exactly the shape a grep-derived list gets wrong.
* `parakeet_registry` — `ForwardParakeetRefused` `(void)`s its input and
  `VT_CHECK(false, ...)`s: transcription only.
* `llama_embedding_registry` — POOLING. It reads host `token_ids` and never the
  pointer, so it is class (b) by code shape, but every request it serves is
  prefill-only, the combine splices nothing, and it is CORRECT. It is the
  counterexample that makes the `host_token_ids_stale` term mandatory, and it is
  a required test.
* `qwen3_vl_registry` — reads no token identifiers at all; it embeds
  `mm.inputs_embeds`. But see O3: the identifiers it is BUILT from are stale, so
  the guard refuses it, correctly, for a defect that lives one layer up.

`laguna_registry` is class (a) by CODE and unreachable by #2618 — see D5.

## Risks and decisions

* **D1 — a refusal that breaks a working path.** Mitigated by the staleness term,
  not by judgement: the guard cannot fire on a step the combine does not touch.
  `ForwardLlamaModelEmbedding` is the concrete case and is a required test.
* **D2 — the granularity trap.** Mitigated by making the predicate step-granular
  by construction and by a required test whose two readings DISAGREE.
* **D3 — extracting `CombineSplicesRow` changes the host combine.** It is a pure
  extraction of an existing condition; the existing combine test is the control.
* **D4 — a model whose forward reads the field only on SOME paths.** The bit says
  "this forward reads the field", not "on every path". A forward with a path that
  does not is a defect in that forward, not something this guard can see. Named
  so it is not mistaken for a guarantee.
* **D5 — `laguna` (#2618) is unreachable.** Its forward was wired to read the
  field by #2544, so it is class (a) by CODE and class (c) by REACHABILITY. The
  bit records what the code does. It is set, and #2618 stays the tracker for the
  fact that no step can reach it.

## Tests

RED-first, both counts captured (cases AND assertions), because
`N failed / 0 assertions failed` means the cases THREW.

1. `tests/vllm/model_executor/test_device_token_ids_refusal.cpp` — the pure
   predicate, every polarity:
   * arrived + stale + not claimed ⇒ REFUSE (the defect's shape).
   * arrived + stale + claimed ⇒ PASS (the side that, if wrong, makes the
     capability unreachable while every other gate stays green).
   * arrived + NOT stale + not claimed ⇒ PASS (the pooling case, D1).
   * NOT arrived ⇒ PASS in both claim polarities.
2. `tests/vllm/v1/worker/test_combine_row_predicate.cpp` — the step-granular ANY,
   including **the disagreement case**: `num_reqs == 3`, `seq_lens = {5, 5, 9}`,
   `prefill_len = {8, 8, 8}`, so rows 0 and 1 are prefill and row 2 is a decode
   row. A per-request reading of row 0 answers "not stale"; the step answers
   "stale". Also the `idx_mapping` indirection, and the boundary
   `seq_len == prefill_len` (the chunk that exactly completes prefill is NOT
   spliced).
3. Reachability: the refusal is entered through `ModelRegistry::Forward` with a
   registered factory, never by calling the predicate alone. A test that only
   calls the free function measures the function, not the guard.

## Gates

* `scripts/agent-preflight.sh`, read by grepping the ANSI-stripped output for
  `gate(s) failed` and `NOT a green`. The exit code is not the result.
* `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, run by hand
  because it is CI-only and preflight does not run it.
* Focused ctest on the new suites plus `test_combine_tokens` (the control for
  D3).
* Mutation, REBUILDING each time and restoring byte-for-byte under sha256,
  because a mutation that does not rebuild reads as a passing test. One mutation
  DELETES the refusal from `ModelRegistry::Forward` and must turn a gate red.

## Evidence

Measured in `/home/mudler/_git/vllm.cpp-devidrefusal` on `bb2da6f97` + this
branch, CPU, Release, `-j 3`. Both counts are recorded everywhere, because
`N failed / 0 assertions failed` means the cases THREW and is a different result.

### GREEN

| suite | cases | assertions |
|---|---|---|
| `test_device_token_ids_refusal` | 3 / 3 passed | 16 / 16 passed |
| `test_combine_row_predicate` | 6 / 6 passed | 13 / 13 passed |
| `test_combine_tokens` (the D3 control) | 7 / 7 passed | 14 / 14 passed |
| `test_multi_kv_refusal` (the sibling) | 2 / 2 passed | 7 / 7 passed |

### RED, and the mutations

Eight mutations, each APPLIED (sha256 verified changed), REBUILT (a mutation that
does not rebuild reads as a passing test), run, then restored and verified
byte-for-byte by sha256. All eight were detected; none survived.

| # | mutation | rebuild rc | detected by | cases / assertions failed |
|---|---|---|---|---|
| M1 | DELETE the refusal from `ModelRegistry::Forward` (`if (false && ...)`) | 0 | `test_device_token_ids_refusal` | 1 case / 1 assertion |
| M2b | drop the `host_token_ids_stale` term (nullness-only, the sibling's shape) | 0 | `test_device_token_ids_refusal` | 2 cases / 1 assertion |
| M3b | drop the `device_token_ids != nullptr` term | 0 | `test_device_token_ids_refusal` | 1 case / 1 assertion |
| M4 | invert the claim term (`!consumes` -> `consumes`) | 0 | `test_device_token_ids_refusal` | 3 cases / 3 assertions |
| M5 | `CombineSplicesRow` boundary `>` -> `>=` | 0 | `test_combine_row_predicate` AND `test_combine_tokens` | 2+1 cases / 2+1 assertions |
| M6 | `AnyRowSplicedByCombine` returns the FIRST row instead of the OR | 0 | `test_combine_row_predicate` | 3 cases / 3 assertions |

M1 is the reachability mutation AGENTS.md `## Nothing lands dead` asks for: it
deletes the production call site and a gate goes red, so the suite measures the
guard and not merely the free function.

M5 firing in `test_combine_tokens` as well as in the new suite is the evidence
that the extraction is REAL: the host combine and the guard are running the same
expression, so breaking it breaks both. Had `test_combine_tokens` stayed green,
the "one rule, one expression" claim would have been false.

M2 and M3 were first run WITHOUT a `(void)` cast and were caught by
`-Werror=unused-parameter` at `model_registry.cpp:502` rather than by any test.
That is a compiler detection, not a gate detection, and it is a WEAKER result, so
both were re-run as M2b/M3b with the parameter voided. The recorded verdicts are
the re-runs. The first pair is kept in this record rather than dropped, because
"the build failed" reads like a detection and is not one.

### What the gate does NOT reach

The runner's own assignment (`forward_input.host_token_ids_stale = ...`,
`runner.cpp`) has NO CPU coverage: the mirror requires CUDA, so no test in this
tree executes that line. Deleting it would not turn any gate red. That is `## Owed`
O2 and it is stated rather than implied.

## Stop conditions

* STOP and report `NEEDS_DECISION` if the staleness term cannot be computed
  without a second derivation of the combine's row predicate.
* STOP if the disagreement case cannot be constructed. Report that plainly
  rather than shipping a gate that cannot tell a per-step reading from a
  per-request one.
* STOP before setting the bit on any forward whose consumption cannot be read in
  the source. A guess in either direction is the failure #2544 recorded.

## Owed

* **O1** — the CUDA combine kernel keeps its own copy of the row predicate,
  because device code cannot call the host inline. Nothing in this tree pins the
  two copies against each other on a device. Owned by
  [#2710](https://github.com/mudler/vllm.cpp/issues/2710).
* **O2** — no hardware leg. Every gate here is CPU. The guard's PURPOSE is a
  CUDA-default path, so the arm that proves it fires in situ is owed. Owned by
  [#2710](https://github.com/mudler/vllm.cpp/issues/2710).
* **O3** — the MULTIMODAL embed path has the same defect one layer up, and this
  row only makes it loud. `execute_model` passes the stale host vector as
  `MmEmbedInputs::token_ids`, and that struct carries no device channel at all,
  so `ModelRegistry::EmbedMm` builds `inputs_embeds` from token id 0 for every
  decode row. `batch_carries_mm()` returns true on the decode steps of an image
  request by design, so the path is reached. Tracked by
  [#2730](https://github.com/mudler/vllm.cpp/issues/2730).
* **O4** — the SIXTEEN class-(b) architectures are refused, not ported. Each is
  owed by the row that ports it; this row deliberately does not wire sixteen
  forwards it cannot gate. Tracked by
  [#2732](https://github.com/mudler/vllm.cpp/issues/2732).
* **O5** — `ModelRegistry::Forward` is not the only way into a registered
  forward. The in-model drivers (`qwen3_vl.cpp`, `gemma4_mm.cpp`,
  `muse_glimmer_mm.cpp`) call model forwards directly and bypass this guard.
  They also set no `device_token_ids`, so nothing is stale there today, but the
  guard's coverage is the registry seam and not the whole tree. Owned by
  [#2710](https://github.com/mudler/vllm.cpp/issues/2710).
