# ENG-MM-EMBED-DEVICE-IDS — give the multimodal embed the device identifiers

Row: `ENG-MM-EMBED-DEVICE-IDS`
Issue: [#2730](https://github.com/mudler/vllm.cpp/issues/2730)
Precedents: [#2710](https://github.com/mudler/vllm.cpp/issues/2710),
[#2544](https://github.com/mudler/vllm.cpp/issues/2544),
[#1305](https://github.com/mudler/vllm.cpp/issues/1305),
[#2379](https://github.com/mudler/vllm.cpp/issues/2379)

## Now

ACTIVE.

## Scope

This row is `## Owed` O3 of
[`eng-async-device-ids-refusal.md`](eng-async-device-ids-refusal.md): the
`device_token_ids` defect class has a member that is not one of the 36 registered
forwards, and it is the multimodal embed hook itself.

`GPUModelRunner::execute_model` builds the multimodal step inputs as

```cpp
MmEmbedInputs embed_inputs;
embed_inputs.token_ids = &token_ids;   // the HOST step vector
```

and `MmEmbedInputs` carries no device channel at all, so `ModelRegistry::EmbedMm`
cannot see the spliced identifiers even in principle. The asynchronous runner
splices each decode row's sampled token into the DEVICE buffer on the main queue
and deliberately never writes it back; `token_ids_cpu` is zero-initialised, so
the merged `inputs_embeds` handed to a registered multimodal forward embeds
**token id 0** on every decode row. `async_device_mirror()` is DEFAULT-ON for
CUDA, integrated parts included.

`batch_carries_mm()` returns true on the decode steps of an image request BY
DESIGN and says so in its own comment (`runner.cpp`), so the path is reached
rather than theoretical.

IN SCOPE:

1. The device channel on `MmEmbedInputs`, and the runner assignment that fills
   it.
2. Making both registered `embed_mm` hooks read it, through the EXISTING splice
   expression rather than a second copy of it.
3. A capability bit for the hook, and a refusal in `ModelRegistry::EmbedMm` that
   reuses the predicate `ModelRegistry::Forward` already refuses on.
4. Setting `ModelFactory::consumes_device_token_ids` on the ONE registration
   whose whole reachable path is then correct, so the fix is not computed and
   then discarded.

OUT OF SCOPE: the sixteen class-(b) text forwards
([#2732](https://github.com/mudler/vllm.cpp/issues/2732), O4 of the sibling
row), `dots3_note`'s TEXT arm among them; the in-model multimodal drivers that
bypass the registry seam entirely (O5 of the sibling row); a hardware leg.

## Upstream anchors

There is no vLLM anchor for the FIELD, for the same reason the sibling row
records: `device_token_ids` is this project's own seam for a scheduling shape
vLLM expresses with a torch tensor that is always device-resident. Upstream's
`input_ids` has no stale host twin.

The HOOK is anchored. `embed_mm` is upstream's
`SupportsMultiModal.embed_input_ids` (`vllm/model_executor/models/interfaces.py`,
the `multimodal_embeddings: MultiModalEmbeddings | None = None,` line whose
`grep -c` in that file is 1), which upstream calls with the same `input_ids`
tensor the forward gets. So "the embed and the forward resolve their identifiers
from the same place" is upstream's shape, and this row restores it.

## Design

### Where the multimodal embed actually reads its identifiers

Read in the source before choosing an arm, because the answer selects the arm.

Both registered hooks have the SAME shape. `EmbedMmQwen3VLForConditionalGeneration`
(`qwen3_vl_registry.cpp`) and `EmbedMmDots3NoteForCausalLM`
(`dots3_note_registry.cpp`) each do:

```cpp
const std::vector<int32_t>& token_ids = *inputs.token_ids;   // HOST
dense_attn::DBuf ids(d, vt::DType::kI32, {T}, token_ids.data());  // uploaded
vt::Embedding(d.q, emb.t(), table, ids.t());                 // gathered ON DEVICE
```

The identifiers come off the HOST vector, but the GATHER is on the device out of
a device buffer a host upload has just filled. That is exactly the destination
`detail::ApplyDeviceTokenIds` is documented for: "`dst` holds `dst_count` int32
identifiers that a host upload has already filled; the override replaces its
first `ov.count` rows".

So the arm is **the device splice**, arm 1. The host-resolve arm
(`ResolveHostTokenIds`) is for a forward that gathers embedding rows ON THE HOST
and therefore has no `dst`; it costs a `Copy` plus a `Synchronize` per forward and
buys nothing here, because there IS a `dst`. A refusal alone is not the answer
either: #2710 already refuses this path, and a refusal is a disposition, not a
fix.

### Extending the splice rather than copying it

`detail::ApplyDeviceTokenIds` takes its source from the thread-local
`DeviceTokenIdsOverride()`, published by `detail::DeviceTokenIdsScope` inside a
registry FORWARD. `ModelRegistry::EmbedMm` runs BEFORE the forward, from the
runner, so no override is live at that point and the existing entry point cannot
serve this caller.

The seam is EXTENDED, not duplicated: `ApplyDeviceTokenIds` gains an overload
that takes the `DeviceTokenIds` explicitly, and the existing five-argument entry
point becomes one line that passes `TakeDeviceTokenIds()` into it. One splice
expression, two publishers. That is the condition AGENTS.md §"Shared seams" gives
for extending a seam, and it is what makes a mutation of the bounds check turn
BOTH the text gate and the new multimodal gate red rather than one of them.

The runner could instead have opened a `DeviceTokenIdsScope` around the
`EmbedMm` call. Rejected: it would make a runner→model hand-off travel through
thread-local state that is documented as "set ONLY from the registry entry points
that receive the ModelForwardInput", and `MmEmbedInputs` exists precisely to be
the runner's per-step channel to this hook.

### The bit, and why it is a SECOND bit

`ModelFactory::consumes_device_token_ids` is a statement about the FORWARD.
`dots3_note` is the counterexample that makes one bit impossible: its `embed_mm`
consumes the channel after this row, and its forward does not — it reaches
`Dots3NoteModel::ForwardDevice(input.token_ids, ...)` and is class (b) of the
sibling row's classification, owed by #2732. One bit cannot express "the embed
reads it, the forward does not", and collapsing the two facts would either
un-refuse a stale text step or refuse a correct image step.

So `ModelFactory::embed_mm_consumes_device_token_ids` is added, default FALSE,
the polarity `consumes_device_token_ids`, `consumes_multi_kv` and
`stage_on_load` already use.

### The guard, and why it is not a second predicate

`ModelRegistry::EmbedMm` refuses through `DeviceTokenIdsRefusalApplies` — the
SAME free function `ModelRegistry::Forward` calls, with the hook's bit in place
of the forward's. Not a second derivation: a refusal whose predicate is a copy of
its sibling's is a copy that can drift, and the three-term shape is exactly what
has to be preserved.

The third term is the one that spares working paths, and here it spares the
COMMON one. A VL request's PREFILL step is all-prefill: the combine splices no
row, `host_token_ids_stale` is false, and the guard cannot fire. Every image
request in this tree reaches its first token through that step. A nullness-only
guard would refuse it, which is the `ForwardLlamaModelEmbedding` mistake the
sibling row's D1 records, in the one place multimodal serving cannot survive it.

### Why a channel needs a guard at all

Adding an advisory field without a refusal would reintroduce, at a new seam, the
exact defect this family exists to remove: #2710's whole argument is that an
advisory `device_token_ids` is a silent-wrong-answer generator, and a new
`embed_mm` written after this row would ignore `MmEmbedInputs::device_token_ids`
with no gate anywhere able to see it. The field and its guard land together.

### Setting `consumes_device_token_ids` on Qwen3-VL

`ForwardQwen3VLForConditionalGeneration` reads NO token identifiers: it refuses a
step without `input.mm` by name and then embeds `mm.inputs_embeds`. Through
`ModelRegistry::Forward` the only way it can be reached is from the runner branch
that has ALREADY called `EmbedMm` with the same `device_input_ids` on the same
step — `if (mm_buffers.has_value()) forward_input.mm = mm_buffers->mm;` is inside
the same `if (supports_mm_inputs() && batch_carries_mm())` window — and without
`mm` the forward throws. So after this row the registration's whole reachable
path resolves its identifiers from the device buffer, and the bit is a true
statement about the registration rather than about the function named `forward`.
The field's contract comment is amended to say that in those words, because
leaving it saying "forward" while a registration sets it on the strength of its
embed is how a bit becomes a lie.

Without this, the row would compute a correct `inputs_embeds` and then have
`ModelRegistry::Forward` throw it away on the same step, which is a fix nothing
can reach.

`dots3_note` does NOT get it, and stays refused. That is the honest reading: the
runner takes the mm branch when ANY request in the batch carries multimodal
items, so a dots3-note batch of TEXT-ONLY requests takes the text arm, where
`ForwardDevice` embeds the stale host vector. Un-refusing it would need its text
arm ported, which is #2732's.

### Where the runner computes staleness

`forward_input.host_token_ids_stale` is computed today AFTER the `EmbedMm` call
site. It is hoisted into one local above the multimodal branch and read twice —
once into `MmEmbedInputs`, once into `ModelForwardInput` — so the embed's guard
and the forward's guard cannot disagree about the same step. It is not a second
call to `AnyRowSplicedByCombine`.

### Ordering

The splice `Copy` is enqueued on the runner's MAIN queue, which is where
`LaunchCombineSampledAndDraftTokens` wrote the source earlier in the same
`execute_model`, so it is ordered AFTER the combine rather than racing it. Both
hooks then run `vt::Embedding` and a `Download`/`Synchronize` on that same queue.
This is the same ordering argument the text device arm makes, and it is why a
host read of `MmEmbedInputs::token_ids` cannot substitute.

## Classification of every `embed_mm` implementation

The seam has exactly TWO implementations at `f31364bbc`, and this was confirmed
by reading each rather than by trusting a grep — the sibling row records a
`grep` that was wrong in both directions.

| hook | reads ids from | class | disposition |
|---|---|---|---|
| `EmbedMmQwen3VLForConditionalGeneration` (`qwen3_vl_registry.cpp`) | host `*inputs.token_ids`, uploaded to a device `DBuf`, gathered by `vt::Embedding` | (a) after this row: CONSUMES | splices; both bits set |
| `EmbedMmDots3NoteForCausalLM` (`dots3_note_registry.cpp`) | identical shape | (a) after this row: CONSUMES | splices; the HOOK bit set, the FORWARD bit NOT (its text arm is #2732's) |

**The issue's blast-radius line is wrong about two of the three names it gives.**
`#2730` says "Every registration that routes through `EmbedMm`:
`Qwen3VLForConditionalGeneration`, and the Gemma-4 and Muse-Glimmer multimodal
arms." Gemma-4 and Muse-Glimmer set NEITHER `encode_mm` NOR `embed_mm`
(`grep -n 'encode_mm\|embed_mm' gemma4_registry.cpp muse_glimmer_registry.cpp`
returns nothing), so `ModelRegistry::SupportsMmInputs` is false for both and the
runner's multimodal arm is never entered for them at all. Their multimodal paths
are the IN-MODEL drivers `gemma4_mm.cpp` and `muse_glimmer_mm.cpp`, which is O5
of the sibling row — a different seam, which sets no `device_token_ids` and is
not reached by this row. The name the issue omits is `dots3_note`, which DOES
register the hooks (#2512, after #2730 was filed).

`tests/vllm/plugins/toy_model_plugin.cpp` registers a factory with `embed_mm`
null and is therefore not a member of this set; it is named because it is the
one other `ModelFactory` initializer outside `src/`.

## Risks and decisions

* **D1 — a refusal that breaks a working path.** The prefill step of every image
  request is all-prefill and MUST NOT be refused. Mitigated by reusing the
  three-term predicate, and by a required test whose subject is exactly that
  step.
* **D2 — the granularity trap.** `host_token_ids_stale` is per-STEP by
  construction, hoisted from the one computation the forward's guard also reads,
  so the two guards cannot disagree. A required test uses `num_reqs == 2` with a
  MIXED disposition, where a per-request reading and a per-step reading give
  different answers.
* **D3 — extending `ApplyDeviceTokenIds` changes the text path.** It is a pure
  extraction: the existing entry point becomes a one-line delegation. The
  existing text suites are the control, and a mutation of the shared bounds check
  must red BOTH.
* **D4 — `consumes_device_token_ids` on a forward that does not read the
  field.** Accepted and documented in the field's own comment: the bit is a
  statement about the REGISTRATION's reachable path. It is set for exactly one
  registration, on the argument written above, and only because that
  registration's forward refuses every step that did not come through the embed.
* **D5 — dots3-note's embed is fixed and its step is still refused.** Named
  under `## Owed` rather than hidden. The hook runs (the runner calls
  `ModelRegistry::EmbedMm` before `ModelRegistry::Forward`), so the code is
  reached; the value it computes is then discarded by the forward's refusal until
  #2732 ports its text arm.

## Tests

RED-first, BOTH counts captured (cases AND assertions), because
`N failed / 0 assertions failed` means the cases THREW and is a different result.

`tests/vllm/models/test_mm_embed_device_token_ids.cpp`:

1. **The defect, through `ModelRegistry::EmbedMm` on the REAL Qwen3-VL
   registration.** A three-token step, two requests: rows 0-1 a prefill, row 2 an
   image request's decode row. The host vector holds `{5, 6, 0}` — id 0 is what
   `token_ids_cpu` zero-initialisation leaves there — and the device buffer holds
   `{5, 6, 9}`. The merged `inputs_embeds` row 2 must equal embedding table row
   **9**. Before the fix it equals row 0, which is the defect stated as a number.
   Rows 0 and 1 must be unchanged, which is what catches a splice that writes the
   wrong extent.
2. **The guard, every polarity, through `ModelRegistry::EmbedMm`.** stale +
   unclaimed REFUSES and names the architecture; stale + claimed RUNS (the side
   that, if wrong, makes the capability unreachable while every other gate stays
   green); FRESH + unclaimed RUNS, which is D1's image-prefill step; a null
   pointer RUNS in both claim polarities.
3. **The registrations declare what they do.** Qwen3-VL sets both bits;
   dots3-note sets the hook bit and NOT the forward bit. A test that asserts the
   asymmetry, because the asymmetry is the design.

4. **The SECOND hook, on a really loaded model.** `EmbedMmDots3NoteForCausalLM`
   carries the same splice and its deletion must red a gate too. Its weights
   cannot be borrowed -- `Dots3NoteLoadedModel` lives in the anonymous namespace
   of its registry translation unit -- so this case goes through the REAL loader
   over the tiny on-disk checkpoint. The assertion needs no knowledge of the
   embedding table: the hook runs three times on one model, and the
   stale-host + live-device run must equal byte-for-byte the run whose host
   vector already held the true identifiers, and must DIFFER from the run that
   embedded the stale vector. The second comparison is the control that stops a
   degenerate table from making the first true for the wrong reason.

Reachability: every case enters through `ModelRegistry::EmbedMm`. A test that
called `EmbedMmQwen3VL...` directly would measure a function, not the seam.

## Gates

* `scripts/agent-preflight.sh`, read by stripping ANSI and grepping for
  `gate(s) failed` and `NOT a green`. The exit code is not the result, and a log
  without a `gate(s) SKIPPED` line is incomplete rather than clean.
* `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, by hand,
  because it is CI-only.
* Focused ctest on the new suite plus the controls: `test_device_token_ids_refusal`
  (the shared predicate), `test_api_server_mm_forward` and
  `test_api_server_dots3_mm_forward` (the two hooks' existing served-request
  gates), `test_model_registry`.
* Mutation, REBUILDING each time, restored byte-for-byte under sha256. One
  mutation DELETES the production splice call and one DELETES the guard; both
  must turn a gate red.

## Evidence

Measured in `/home/mudler/_git/vllm.cpp/.wt/mm-embed-device-ids` on `f31364bbc`
plus this branch, CPU, Release, `-j 3`, `nice -n 15` (several sessions build on
this box concurrently). BOTH counts are recorded everywhere, because
`N failed / 0 assertions failed` means the cases THREW and is a different result.

### RED, before the splice, the guard and the bits existed

The channel fields and the capability bit landed first so the suite COMPILES and
fails on its assertions rather than on the compiler, which is a different result
again.

| suite | cases | assertions |
|---|---|---|
| `test_mm_embed_device_token_ids` | 4 \| 1 passed \| **3 failed** | 76 \| 54 passed \| **22 failed** |

The decode row read `0, 0.015625, 0.03125, ...` where the device buffer said
row 9 -- table row **0**, exactly the value `token_ids_cpu` zero-initialisation
leaves in the host vector. The null-channel control passed in the same run, which
is what says the RED is the channel and not the fixture.

### GREEN

| suite | cases | assertions |
|---|---|---|
| `test_mm_embed_device_token_ids` (new) | 5 / 5 passed | 83 / 83 passed |
| `test_device_token_ids_refusal` (the shared predicate) | 3 / 3 passed | 16 / 16 passed |
| `test_combine_row_predicate` | 6 / 6 passed | 13 / 13 passed |
| `test_combine_tokens` | 7 / 7 passed | 14 / 14 passed |
| `test_multi_kv_refusal` | 2 / 2 passed | 7 / 7 passed |
| `test_model_registry` | 24 / 24 passed (1 skipped) | 993 / 993 passed |
| `test_openai_api_server_mm_forward` (Qwen3-VL served image request) | 9 / 9 passed | 73 / 73 passed |
| `test_openai_api_server_dots3_mm_forward` (dots3-note served image request) | 16 / 16 passed | 240 / 240 passed |
| `test_dots3_note_scaffold` | 26 / 26 passed | 110835 / 110835 passed |
| `test_moe_async_device_ids` (the text arm of the shared splice) | 6 / 6 passed | 191 / 191 passed |
| `test_glm_moe_dsa_forward` | 8 / 8 passed | 5339 / 5339 passed |
| `test_glm5_next_forward` | 32 / 32 passed | 5943 / 5943 passed |
| `test_persistent_step_input` | 10 / 10 passed | 66 / 66 passed |
| `test_qwen3_decode_graph_seam` | 4 / 4 passed | 231 / 231 passed |

`scripts/agent-preflight.sh --staged`: 0 gates failed, `tree-compiles` ok
(386 of 386 translation units, 142.9s), `commit-trailers` ok, `commit-style` ok.
Five gates SKIPPED for environment reasons only -- `check-arm-isa-build.py`,
`check-cpu-isa-build.py`, `check-cuda-fat-gencode.py`, `check-pr-size.py` (CI
only, run by hand below) and `check-triton-aot-multiarch.py` (needs
`--vendored-root`) -- so the run prints `NOT a green preflight` on the skips and
on nothing else.

The full 661-binary `ctest` was NOT run: it needs about 17.6 GB and this box had
21-26 GB free with other sessions building concurrently. The suites above were
selected as the ones this change reaches, and the omission is stated rather than
implied.

### The mutations

Each was APPLIED (sha256 verified changed), REBUILT (a mutation that does not
rebuild reads as a passing test), run, then restored and verified byte-for-byte
by sha256. Every rebuild returned 0, so no verdict below is a compiler catching
what a test should have.

| # | mutation | rebuild rc | detected by | cases / assertions failed |
|---|---|---|---|---|
| M1 | DELETE the production splice call in `EmbedMmQwen3VLForConditionalGeneration` | 0 | `test_mm_embed_device_token_ids` | 1 / 16 |
| M2 | DELETE the guard from `ModelRegistry::EmbedMm` (`if (false && ...)`) | 0 | same | 1 / 3 |
| M3 | drop the `host_token_ids_stale` term (nullness-only, the wrong predicate) | 0 | same | 1 / 1 |
| M4 | invert the claim term (`consumes` -> `!consumes`) | 0 | same | 2 / 4 |
| M5 | the SHARED splice body copies ONE identifier instead of `ov.count` | 0 | `test_mm_embed_device_token_ids` **AND** `test_moe_async_device_ids` | 1 / 16 **and** 3 / 3 |
| M6 | clear `embed_mm_consumes_device_token_ids` on the Qwen3-VL factory | 0 | `test_mm_embed_device_token_ids` | 2 / 1 |
| M7 | clear `consumes_device_token_ids` on the Qwen3-VL factory | 0 | same | 1 / 1 |
| M9 | DELETE the production splice call in `EmbedMmDots3NoteForCausalLM` | 0 | same | 1 / 2 |
| M10 | clear `embed_mm_consumes_device_token_ids` on the dots3-note factory | 0 | same | 2 / 1 |

M1 and M9 are the reachability mutations AGENTS.md `## Nothing lands dead` asks
for: each deletes a production call site and a gate goes red, so the suite
measures the capability and not a class.

M5 firing in `test_moe_async_device_ids` as well as in the new suite is the
evidence that the seam extension is REAL rather than nominal: the text arm and
the multimodal arm are running the same expression, so breaking it breaks both.
Had the text suite stayed green, the "one splice, two publishers" claim in
`## Design` would have been false. `test_glm_moe_dsa_forward` stayed green under
M5 and is recorded as not covering the copy extent.

M6 and M10 report `2 cases / 1 assertion` because clearing a hook bit makes the
guard fire inside the correctness case as well, which THROWS rather than failing
an assertion. Both counts are given for exactly that reason.

### The mutation that SURVIVED, and what it means

| # | mutation | rebuild rc | detected by | result |
|---|---|---|---|---|
| M8 | DELETE the runner's `embed_inputs.device_token_ids` / `host_token_ids_stale` assignments | 0 | NOTHING | `test_mm_embed_device_token_ids` 5/5, `test_openai_api_server_mm_forward` 9/9, `test_openai_api_server_dots3_mm_forward` 16/16 -- all still GREEN |

Predicted, not discovered: the mirror requires CUDA, `device_input_ids` is
non-null only under `#ifdef VLLM_CPP_CUDA`, and every gate in this tree is CPU.
No test executes those two lines, so deleting them reds nothing. This is `## Owed`
O1 and it is the sibling row's O2 in the same shape. It is recorded here rather
than left for a reader to infer from a table of successes.

## Stop conditions

* STOP and report `NEEDS_DECISION` if the splice cannot be expressed through the
  existing seam without a second copy of it.
* STOP if the multi-request disagreement case cannot be constructed, and say so
  rather than shipping a gate that cannot tell a per-step reading from a
  per-request one.
* STOP before setting a bit on any registration whose consumption cannot be read
  in the source.

## Owed

* **O1** — no hardware leg. Every gate here is CPU; the mirror requires CUDA, so
  the runner's own two assignments (`embed_inputs.device_token_ids`,
  the hoisted `host_token_ids_stale`) execute in no test in this tree and
  deleting them would red nothing. This is the sibling row's O2 in the same
  shape, and it is stated rather than implied. Owned by
  [#2730](https://github.com/mudler/vllm.cpp/issues/2730).
* **O2** — `dots3_note`'s embed consumes the channel and its step is still
  refused at `ModelRegistry::Forward`, because its TEXT arm embeds the stale host
  vector. Owned by row `ENG-ASYNC-DEVICE-IDS-REFUSAL` O4 and tracked by
  [#2732](https://github.com/mudler/vllm.cpp/issues/2732).
* **O3** — the in-model multimodal drivers (`qwen3_vl.cpp`, `gemma4_mm.cpp`,
  `muse_glimmer_mm.cpp`) bypass `ModelRegistry::EmbedMm` and this guard entirely.
  They set no `device_token_ids` today, so nothing is stale there; the coverage
  is the registry seam and not the whole tree. This is the sibling row's O5.
  Owned by [#2710](https://github.com/mudler/vllm.cpp/issues/2710).
