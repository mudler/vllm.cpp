# The Nemotron oracle golden cannot be regenerated, and now it says so

**Issue:** [#926](https://github.com/mudler/vllm.cpp/issues/926)
**Row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm` ([model matrix](../model-matrix.md))
**Owed out of:** [#517](https://github.com/mudler/vllm.cpp/issues/517), which
committed the golden
**Blocks:** [#1289](https://github.com/mudler/vllm.cpp/pull/1289) is being scored
against this golden at 95/96

## 0. Scope (headline verdict)

`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json` is the reference
for every Nemotron-3.5-Lightning token claim in this tree. **Its capture
configuration is unrecoverable.** The recovery was attempted and failed on
evidence, not on effort, and §2 records the six independent places it was looked
for.

What this row does instead: it commits a generator, extends the golden's format
to carry the engine configuration, and makes the artifact **state which of two
things it is** — attributed, or unattributed-and-saying-so. The third state,
silence, was what #926 filed, and it is what this removes. The lost
configuration is not invented, and no gate is weakened to accommodate its
absence.

**In scope**

- `scripts/nemotron-h-oracle-capture.py` — capture, verify and validate.
- The `capture` provenance block, written truthfully into the shipped golden.
- The contract, gated in three places that cannot drift apart: the generator's
  `--check`, `tests/scripts/test_nemotron_h_oracle_capture.py`, and the C++
  consumer in `tests/vllm/models/test_nemotron_h_loader.cpp`.
- The A3 driver printing what it is being held to.

**Out of scope**

- Running the oracle. [#1431](https://github.com/mudler/vllm.cpp/issues/1431)
  owns that and five attempts have already been killed by the host-memory
  watchdog. This unit needs no GPU, which is why it could be done now.
- The index-29 top-2 margin ([#1388](https://github.com/mudler/vllm.cpp/issues/1388)).
- Deciding whether #1289's moved token is a defect.

## 1. What `oracle.json` recorded, and what it omitted

Seven top-level keys, as `af8170154` wrote them:

| Key | Value |
|---|---|
| `vllm` | `0.23.1rc1.dev1511+g555967922` |
| `transformers` | `5.14.1` |
| `flashinfer` | `0.6.15.post1` |
| `model` | `/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4` |
| `revision` | `29f2d1746d8f41e316523194b19018707749b1b1` |
| `sampling` | `{temperature: 0.0, max_tokens: 32}` |
| `golden` | 3 entries: `prompt`, `prompt_token_ids`, `token_ids`, `text` |

**Not one engine knob.** `enforce_eager`, `max_model_len`, `max_num_seqs`,
`max_num_batched_tokens`, `gpu_memory_utilization`, `num_gpu_blocks_override`,
`block_size`, the batch shape, `seed`, `ignore_eos`, the compile mode, the
cudagraph capture sizes, the attention backend and the MoE backend are all
absent. Every one of them can move a greedy argmax at a near-tie, because each
one changes batching, prefill chunking, paging or the kernel that runs — that
is, the reduction order.

For contrast, this tree already knows how to do this: the sibling golden
`tests/parity/goldens/gdn_ba_projection_bf16_sm121/oracle.json` records
`device`, `compute_capability`, `cuda_runtime`, `torch_version`, `dispatch`,
`implementation`, `repetitions` and `vllm_target_commit`. The Nemotron golden is
the outlier, not the norm.

## 2. The generator was never committed, and the configuration is gone

Six checks, each of which could have found it:

1. `git show --stat af8170154` adds exactly **three** files —
   `.agents/specs/nemotron-h-model.md`, this golden, and
   `tests/parity/hf_snapshot.h`. None is a generator.
2. `git log --all --oneline -- tests/parity/goldens/nemotron_35_lightning_greedy/`
   returns **exactly one commit** in the whole history. The golden has never
   been revised, re-attributed or regenerated.
3. `grep -rn nemotron_35_lightning_greedy scripts/` returns nothing, and a
   per-branch `git ls-tree` sweep for a Nemotron capture script finds none. The
   Nemotron scripts that do exist (`nemotron-h-a2q1-neartie-gap.py`,
   `nemotron-h-a2q1-dgx-gate.sh`, `nemotron-h-a2q2b-gpu-gate.sh`) are all later
   and none captures a golden. `git log --diff-filter=D -- scripts/` shows none
   was deleted either.
4. The capture ran from `$HOME/venvs/vllm-oracle-next` on `dgx.casa`
   (`af8170154`'s message and `nemotron-h-model.md` §5a). **`dgx.casa` was
   reimaged on 2026-08-14**, two days later.
5. Nothing of the run reached shared storage: a recursive grep for
   `MODEL_LOADED_OK` and `ORACLE_IDENTITY_OK` across `/mnt/nas_share/{rc,
   experiments,staging,...}` returns nothing, and the oldest `rc/` job directory
   is **2026-08-17** — the share post-dates the capture.
6. `af8170154`'s own message is the fullest surviving description of the run,
   and it names the venv, the identity assertions, the arch, the layer count and
   the block pattern. It names **no engine knob**.

The only timestamp that exists is the commit's author date,
`2026-08-12T22:37:57Z`, and the golden now records it as that rather than as a
run time.

**Verdict: unrecoverable.** Not "not found yet".

## 3. The lead that looked implicit and is a COMMON TERM

The logs of the rebuilt oracle show `kv_cache_dtype=fp8_e4m3` selected without
anyone asking for it, with `Checkpoint does not provide a q scaling factor.
Setting it to k_scale` (`kv_cache.py:134`) beside it. That is exactly the shape
of an unrecorded implicit choice that moves tokens — so it was checked rather
than assumed, and it is **not** a candidate difference.

The **checkpoint** carries it:

- `config.json` → `quantization_config.kv_cache_scheme` =
  `{"dynamic": false, "num_bits": 8, "type": "float"}`
- `hf_quant_config.json` → `quantization.kv_cache_quant_algo` = `"FP8"`

and at the pin, the default `kv_cache_dtype="auto"` is resolved from that
before `CacheConfig` is built:

- `vllm/engine/arg_utils.py:1916` — `resolved_cache_dtype =
  resolve_kv_cache_dtype_string(self.kv_cache_dtype, model_config)`, whose
  result is passed as `cache_dtype=` at `:1928`. `:1916` is the **unique call
  site of `resolve_kv_cache_dtype_string`** — the symbol has exactly four
  occurrences at the pin: this call, the import (`:115`), a defensive comment
  (`attention.py:282`) and the definition (`torch_utils.py:374`). It is *not*
  the unique place the string `cache_dtype` appears, which is 29 occurrences on
  21 lines of `arg_utils.py`; the resolver is what is unique, and `:1928` is
  where its result reaches `CacheConfig`.
- `vllm/utils/torch_utils.py:374-392` — returns early unless `kv_cache_dtype ==
  "auto"`, then reads `hf_config.quantization_config`.
- `vllm/utils/torch_utils.py:310-362` — for `quant_method` starting `modelopt`,
  maps a `kv_cache_scheme` dict of exactly that shape to `"fp8"`.

So **every** unoverridden run of this checkpoint at this pin gets fp8_e4m3,
including the 2026-08-12 capture. Same on both sides is a measurement, and the
golden now records it under `capture.forced_by_checkpoint_or_device` together
with the MoE backend (MARLIN — a device without native FP4 takes the first
supported backend and at this pin no environment knob selects another), the
dtype and the quantization method.

**This narrows the unrecoverable set to the knobs a driver passes**, which is
also the set §1 lists. It does not recover any of them.

## 4. Two configurations, two answers, both repeatable

| Run | Configuration | Prompt 0 | Prompt 1 | Prompt 2 |
|---|---|---|---|---|
| 2026-08-18 `oracle_only.sh` attempt `a` | `max_model_len=512`, `max_num_seqs=8`, `gpu_memory_utilization=0.30`, `max_num_batched_tokens=512`, `enforce_eager=False`, `TokensPrompt`, one prompt per `generate()` | 32/32 | 32/32 | **26/32** |
| the #926 rebuild | `enforce_eager=True`, `gpu_memory_utilization=0.25`, `max_model_len=4096` | 32/32 | 32/32 | **29/32** |

The first ran its configuration **twice in one process** (`ORACLE_LEG 1`,
`ORACLE_LEG 2`) with identical results, `ORACLE TOKEN MATCH: 180/192`, log at
`/mnt/nas_share/rc/nhspeed/oracle.a.out` (the worker's `/workspace/nhspeed`).

That is **configuration sensitivity, not non-determinism**. The distinction
decides the gate form, and it decides it against weakening: AGENTS.md admits a
ratified distributional gate **only** where the oracle's own greedy decode is
non-deterministic, and here it is not. **A distributional gate is inadmissible
on this evidence.** What is licensed is re-deriving the golden under a named
configuration.

Note what the table also says: **prompt 2 has never been reproduced by anything
this repository can name.** Prompts 0 and 1 have been, twice.

## 5. Design

### 5.1 The contract

A Nemotron oracle golden is in exactly one of two states, and the file says
which:

- `capture.engine_config_recorded = true` — then `capture.engine.resolved`
  carries **every** key in `REQUIRED_ENGINE_KEYS`, `capture.batch.shape` says how
  the prompts were submitted, `capture.legs >= 2` and `capture.legs_agree` is
  true.
- `capture.engine_config_recorded = false` — then
  `capture.unrecoverable_reason` says why and `capture.issue` names the issue
  that owes the re-derivation, and `capture.engine` is null. "Unrecorded" and
  "here is the record" cannot both be true.

A null inside `resolved` is refused for every key but
`num_gpu_blocks_override`: **a value that could not be read is not a value that
was default.** That is the same rule AGENTS.md states for `.env` — a missing
value never becomes an assumption.

The unattributed arm needs a **second** half, and the first round of this row
shipped without it. Everything above is satisfied by a file that says
"unrecorded", names an issue and argues **nothing**. The attributed arm is gated
by structure — twenty keys are in `resolved` or they are not — but the
unattributed arm's whole record is prose, and a contract that asks only whether
the prose is *truthy* gates the shape and not the substance.

The fresh review demonstrated it rather than predicting it: gut `evidence`,
`forced_by_checkpoint_or_device` and `captured_utc_is`, put the single word
`"dunno"` in `unrecoverable_reason`, and **all four gates stayed green** while
`--check` kept printing `engine_config_recorded=False`. Every argument this
artifact rests on — that nothing has ever reproduced prompt 2, that
`kv_cache_dtype` is a common term rather than a candidate difference, that a
distributional gate is inadmissible — lived in ungated prose inside a data file.
That is silence wearing the shape of a record, which is the state #926 filed,
reached from the other side.

So an unattributed golden also carries, and the contract checks:

| Requirement | Why it is named |
|---|---|
| `captured_utc_is`, non-empty | a commit's author date read as a capture time is a **fabricated** provenance |
| `forced_by_checkpoint_or_device`, an object with `kv_cache_dtype`, `moe_backend`, `dtype`, `quantization`, each non-empty | these are the terms COMMON to every unoverridden run of this checkpoint. They are what **narrows** "unrecoverable" to the knobs a driver passes (§3), so deleting one widens the unrecoverable set without saying so |
| `evidence`, an object with `never_reproduced` and `gate_form`, each non-empty | whether anything ever reproduced this golden, and which gate form its behaviour licenses. `gate_form` is the field that refuses a distributional gate; losing it loses the reason the refusal was on evidence rather than on taste |

and four of those fields carry a **length floor of 80 characters**, because their
content is an *argument* rather than a value: `unrecoverable_reason`,
`forced_by_checkpoint_or_device.kv_cache_dtype`, `evidence.never_reproduced` and
`evidence.gate_form`.

**What the floor is, and what it is not.** It cannot prove the prose is *true* —
nothing in a checker can. A keyword grep would be worse: it proves only the
checker's own vocabulary and it reds an honest rewording. What a floor detects is
**removal**, which is the threat that was actually demonstrated, one word in
place of a paragraph. 80 is set from measurement rather than taste. In the
shipped golden those four fields are 814, 448, 702 and 293 characters, so the
tightest margin is **3.7x** and no honest rewrite of an argument approaches it,
while `dunno`, `unknown`, `TBD` and `see the spec` are all under it. AGENTS.md's
rule that a gate firing on ordinary work is the defect is why the floor is taken
from the *shortest* real field and not from the longest.

Scoping the floor to the four *argument* fields is load-bearing rather than
tidy: `forced.dtype` and `forced.quantization` are **35 characters each**, so a
floor applied to every forced term would make the shipped golden red itself.

`captured_utc_is` is required but deliberately **not** floored. Its job is to say
what the timestamp is, and that can honestly be said in a clause —
"af8170154's author date, not a run time" is 44 characters and is not a
hand-wave. Flooring it would be a gate that fires on ordinary work.

The requirement is scoped to the **unattributed arm only**. An attributed golden
records `kv_cache_dtype`, `dtype`, `quantization` and `moe_backend` in
`engine.resolved` as *values*, so it owes no prose about them, and a requirement
that fired on both arms would red every future capture this generator writes.
`test_the_attributed_arm_is_not_burdened_by_these_keys` holds that scoping.

The shipped golden is in the second state. It is **kept**, because deleting
evidence to make a gate green is never the repair.

### 5.2 The generator

`scripts/nemotron-h-oracle-capture.py`, three modes:

| Mode | Needs | Does |
|---|---|---|
| `--check <golden>` | nothing | validates the contract. This is the CI gate. |
| `--verify <golden>` | the oracle | runs and compares, reporting a **configuration** difference before a token difference |
| `--capture --out <p>` | the oracle | runs and writes a golden that records its own configuration |

Four properties are deliberate:

1. **Identity is asserted, never assumed** — the pin substring `555967922` must
   be in `vllm.__version__` and the run aborts otherwise. `$HOME/venvs/
   vllm-oracle` on dgx has resolved to a 0.25.0 rollback that predates
   `NemotronHMoEDecoderLayer`, and a run through it fails in a way that reads as
   "the model is unsupported".
2. **The configuration is read BACK OUT of the built engine**, not echoed from
   the kwargs. `kv_cache_dtype`, the block size, the block count and the
   backends are chosen by vLLM, so what a driver passed is not what it ran.
3. **`--capture` refuses to write** a golden that fails its own contract, or
   whose legs disagree. A golden written from disagreeing legs records a coin
   flip.
4. **There is no builder for the unattributed block, deliberately.** One
   existed and had exactly one caller: a fixture in the test suite. `--capture`
   cannot reach it — a capture that runs records its configuration, which is the
   mode's whole point — so no production path produced the shape it described,
   and the one unattributed golden this repository has was hand-written and
   carried three keys the builder could not emit. Under "Nothing lands dead"
   that is a helper documenting a production shape nothing in production
   produces, and it is gone. Deleting it also repaired the fixture: the suite's
   own rule is that a fixture must never be derived from the module under test,
   because setup and expectation then move together and a key dropped from the
   checker drops from the fixture too. That fixture was the one place the suite
   broke its own rule. It is a test-owned literal now.
5. **The body is under `if __name__ == "__main__":`** — vLLM v1 spawns
   EngineCore, the module re-imports, and an unguarded driver fails as a
   `multiprocessing` traceback naming neither vLLM nor the caller. The tell is
   the banner printing twice.

### 5.3 The named profile

`--profile nhspeed-a` is the 2026-08-18 configuration of §4: the only oracle
configuration on this checkpoint for which this repository has determinism
evidence, with its full resolved config readable at
`/mnt/nas_share/rc/nhspeed/oracle.a.out`, and with CUDA graphs ON because
`--enforce-eager` is never the denominator. **It is a name for a run that
happened. It is not a reconstruction of the lost one**, and nothing in this row
claims it is.

It is a token-golden configuration and **not** a speed denominator:
`max_num_batched_tokens=512` against the denominator's 8192 is a regime you
cannot tune down and keep a ratio through.

## 6. Gates

```sh
python3 scripts/nemotron-h-oracle-capture.py --check \
  tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
python3 tests/scripts/test_nemotron_h_oracle_capture.py
ctest --test-dir build -R test_nemotron_h_loader --output-on-failure
```

All three run with no vLLM, no GPU and no checkpoint. That is the point: the
provenance defect is a records defect, and a gate for it must not need the
hardware whose absence caused it.

The contract exists in three copies — the generator's `REQUIRED_ENGINE_KEYS`,
the C++ consumer's `required` list, and this suite's `EXPECTED_ENGINE_KEYS`. The
suite asserts all three agree and **owns the expectation itself**, so a key
deleted from both production copies is still red.

## 7. Evidence

**Measured on** `mudler-ubuntu-box` (x86_64, 20 cores), in the worktree
`.claude/worktrees/row-926-golden-provenance` at base `5d548d003`, CPU-only
Release build (`-DVLLM_CPP_CUDA=OFF`), nothing overlaid. `libvllm 0.0.3 (ABI 23,
header 23)`. vLLM anchors read at `/home/mudler/_git/vllm` HEAD
`5559679229bc961848b121ccdeaa8fa5d79bec98`, remote `vllm-project/vllm`,
verified before citing.

### RED first

With the golden as `af8170154` left it, the suite collected 25 cases and
**3 were red**: `test_the_shipped_golden_satisfies_the_contract`,
`test_check_reads_the_shipped_golden` and
`test_the_shipped_golden_is_not_silently_attributed`, each reporting
`oracle.json: missing 'capture'`. After the provenance block: **26/26 OK**.

### Mutation proof — the golden (data)

Each mutation applied alone, `git diff --stat` printed to prove it APPLIED, the
target rebuilt to prove `compile_rc=0`, the case rerun, then the tree restored
and its **sha256** re-asserted against the pre-mutation baseline
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`. A mutation
that never applied and a mutation that failed to build both read as a passing
test, so neither is inferred here.

| # | Mutation | diff | Result |
|---|---|---|---|
| — | baseline | — | 33 assertions, 0 failed, SUCCESS |
| M1 | delete the `capture` block (the `af8170154` shape) | -24 | 20 assertions, **1 failed** |
| M2 | claim `engine_config_recorded: true` while `engine` is null | 1/1 | 28, **1 failed** |
| M3 | blank `unrecoverable_reason` | 1/1 | 33, **1 failed** |
| M4 | point `issue` at a non-vllm.cpp string | 1/1 | 33, **1 failed** |
| M5 | truncate one golden row below `max_tokens` | 1/-28 | 33, **1 failed** |
| M6 | empty the `golden` array | 1/-148 | 7, **1 failed** |
| M7 | drop `engine_config_recorded` so the file says NOTHING | -1 | 21, **1 failed** |

M5 and M6 are the anti-vacuity arms: a comparison over zero elements reports a
perfect score, so the width is asserted rather than trusted. M7 is the third
state — not "unrecorded", but silent — which is the state #926 filed.

### Mutation proof — the case is the SOLE holder (source)

`compile_rc` is load-bearing here, and it is printed. **M8**: delete the
`capture` requirement from the C++ case itself (`compile_rc=0`, 0 compile
errors, `git diff --stat` 1 insertion), then re-apply M1. The **whole binary**
then reads `3 passed | 0 failed`, **26 assertions** against the restored 40, and
`Status: SUCCESS!` over a golden with no provenance at all.

Nothing else in this tree holds the guarantee. Restored, the binary is 40
assertions and both sha256s match.

### Mutation proof — the checker (source)

Each applied alone, parsed (`parse_rc=0` — a mutation that does not parse reads
as a passing test), the suite rerun, restored by sha256
`a477bcefbcc90da30d8e0cae016fc474e281b74889ed5107553a6f851ce54fce`.

| # | Mutation | Result |
|---|---|---|
| M9 | stop requiring a reason on an unattributed golden | **1 failure** |
| M10 | stop requiring every engine key (`for key in ()`) | **2 failures** |
| M11 | drop the anti-vacuity width check | **1 failure** |

`test_each_required_engine_key_is_load_bearing` additionally drops each of the
20 engine keys in turn and asserts the contract names the one it dropped, so
M10 is a floor and not the whole proof.

### The driver states all three cases

`nemotron-h-gen --golden-info` on the shipped golden, on a copy with
`engine_config_recorded: true`, and on a copy with the block removed:

```
capture: engine configuration UNRECORDED — a token difference below is
         UNATTRIBUTABLE, not yet a defect; owed by .../issues/926
capture: engine configuration RECORDED
capture: the golden does not SAY whether its engine configuration was recorded
         (no capture block)
```

The same line prints beside `DIVERGENCE`, because a reader who sees `DIVERGENCE`
and stops reading is the reader it is for.

### A trap this hit

`REQUIRE_MESSAGE(..., "capture is missing '" << key << "'")` over a
`const char*` printed `capture is missing '1'`: doctest stringifies a bare
`char*` as a **bool**. The loop iterates `std::string` now, and the messages read
`capture is missing 'schema'`.

### The repair round — closing the SHAPE-not-SUBSTANCE gap

The fresh review refuted nothing and returned findings. The substantive one is
above: the contract gated the shape of the unattributed block and not its
content. The re-run below is the proof that it no longer does.

**Mutation E, re-applied verbatim** to the shipped golden: `unrecoverable_reason`
→ `"dunno"`, `captured_utc_is` → `""`, `forced_by_checkpoint_or_device` → `{}`,
`evidence` → `{}`. Applied alone, `json_parse_rc=0` (a mutation that does not
parse reads as a passing test), `git diff --stat` **5 insertions, 13 deletions**
so it demonstrably APPLIED, then restored and its sha256 re-asserted against the
pre-mutation baseline
`659c26bd2301317d4a6999df0b7afc3243dcff129de89abcb66b46817dd6f9e9`.

| Gate | Before the repair | After |
|---|---|---|
| `--check` | 0 problems, **rc=0** | **8 problems, rc=1**, naming all four gutted fields |
| the Python suite | 26/26 OK | **2 failures**, rc=1 |
| the C++ consumer | `SUCCESS!` | **2 failed, `Status: FAILURE!`** |

The C++ arm is the one that matters for the cross-gate: the guarantee is not
held by one file. `Status:` is read as well as `assertions:`, because an
`assertions:` line can say `0 failed` while cases threw.

**The assertion COUNT under this mutation is shape-dependent, so it is not the
evidence.** A `REQUIRE` aborts its case, so the total depends on which one trips
first, which depends on the mutation's exact diff shape and on whether the count
is read for the single case or for the whole binary. This round measured **38**
for the provenance case alone at diff shape `+5/-13`; the fresh reviewer
measured **45** at `+4/-12`. Both are correct measurements of different shapes,
and an evidence table carrying a number that does not reproduce is worse than
one that omits it. What DOES reproduce, and what the gate rests on, is
**`2 failed`, `Status: FAILURE!`, rc=1**.

Green again on the restored tree: `--check` 0 problems, the suite **41 cases OK**
(26 before this round, 13 new substance cases, 2 new cross-gate cases), and the
C++ case **63 assertions** against the previous 40, `Status: SUCCESS!`.

The new contract holds the **shipped golden byte-for-byte unchanged**. Its
sha256 is the same value before this round and after it, and its diff against
`af8170154` is still 24 insertions and **zero** deletions.

### The three copies still cannot drift, including the new ones

`test_the_cpp_consumer_names_every_unattributed_key` parses `kForcedTermKeys` and
`kEvidenceKeys` out of the C++ source as initializer lists and compares them as
sets, and `test_the_cpp_consumer_carries_the_same_argument_floor` reads
`kMinArgumentChars` out of it. Parsed, not grepped, and the distinction is
load-bearing: `"dtype"` and `"quantization"` already appear in the C++
*engine-key* list, so `assertIn('"dtype"', source)` would have passed without the
unattributed arm naming either of them. The expectation is the suite's own
literal, so a key deleted from both production copies is still red.

`test_each_forced_term_is_load_bearing` and `test_each_evidence_key_is_load_bearing`
drop each key in turn and assert the contract names the one dropped, and
`test_each_argument_field_refuses_a_one_word_answer` puts `"dunno"` in each
floored field in turn. `test_an_argument_at_the_floor_is_accepted` is the other
side of the floor, so the cases cannot pass by refusing everything.

### Mutation proof — the NEW gate, and the new cross-gate

Four more, each applied alone, `parse_rc=0` printed, `git diff --stat` printed to
prove it applied, restored by sha256 (`5b81281f7fa76824253d4b119459c1dad0b54ed2e35c6d4265beb53db9967411`
for the checker, `812808faf1e149378fb13c84be6ccb0a661d2b45c62f5d652361d14d9a60b93e`
for the C++ consumer).

| # | Mutation | Result |
|---|---|---|
| M12 | lower the checker's floor, `MIN_ARGUMENT_CHARS = 80 → 5`, to hide a shrinking record | **3 failures** |
| M13 | stop requiring the evidence keys, `REQUIRED_EVIDENCE_KEYS = ()` | **4 failures** |
| M14 | drop `quantization` from the **C++** `kForcedTermKeys` alone, leaving both other copies intact | **1 failure**, `test_the_cpp_consumer_names_every_unattributed_key` |
| M15 | change the **C++** `kMinArgumentChars` to 5 alone | **1 failure**, `test_the_cpp_consumer_carries_the_same_argument_floor` |

M14 and M15 are the ones worth reading. Nothing in Python imports the C++ file,
so a mutation confined to it could only be caught by a case that reads that
source and holds it to a suite-owned expectation. Both fire, and they fire on the
key that was actually removed. The floor and the key lists cannot be quietly
relaxed in either direction: raising the floor to hide a shrinking record is red,
and lowering it to admit one is red too.

### One more disagreement between the copies, found while proving the fix

The checker tested `unrecoverable_reason` for **truthiness** while the C++ copy
had always spelled it `is_string()`. So `"unrecoverable_reason": 123` returned
**zero problems** from `--check` and was refused by the C++ gate: two copies of
one contract disagreeing about what satisfies it, which is exactly the drift
this three-copy design promises cannot happen. The weaker copy moved to
`_is_prose`, and `test_a_non_string_reason_is_refused` holds it there. Suite
41 → 42 cases.

### The copies disagreed a SECOND time, on whitespace

Found by the fresh review, and it is the `123` divergence again with the weaker
copy on the other side. Python has always spelled these `value.strip()`; the C++
consumer spelled them `.empty()` and `.size()` on the raw string. So a golden
carrying **200 spaces** in `unrecoverable_reason`, `evidence.never_reproduced`
and `evidence.gate_form` read:

| Copy | Before | After |
|---|---|---|
| `--check` | **3 problems, rc=1** | 3 problems, rc=1 |
| the C++ consumer | **70 assertions, 0 failed, `SUCCESS!`, rc=0** | **79 assertions, 6 failed, `Status: FAILURE!`, rc=1** |

A blank paragraph is the record going missing exactly as surely as a deleted
one. `TrimmedProse()` now matches `_is_prose`/`len(value.strip())` exactly.

**The scope is deliberate and is not "trim everything".** Python trims in
`_is_prose` and in the argument floor, and it does **not** trim
`capture.batch.shape`, which it tests for truthiness. Trimming that one in C++
would have repaired this divergence by opening its mirror image, so it is left
alone.

**The C++ arm can now hold its half alone.** The provenance case reads the one
committed golden, so it can only ever exercise the shape that golden happens to
have — which is why this arm could not notice. `NemotronH golden: a blank
paragraph is not prose` tests `TrimmedProse` directly and is the counterpart of
`test_the_floor_is_not_met_by_whitespace` on the Python side. Proven: reverting
`TrimmedProse` to the untrimmed behaviour reds it **5 of 9 assertions with the
golden untouched** (`compile_rc=0`, 0 compile errors, `git diff --stat`
printed), restored by sha256
`00cd7e05a32fa4be3cc852156633dbdc4b3069bc183bbeb4889183c7613f80cb`. It asserts
both sides of the rule, so it cannot pass by refusing everything.

### The driver's third state named a cause it could not know

`nemotron-h-gen` printed `(no capture block)` for `capture_recorded == -1`. The
tri-state was right and the parenthetical was not: two different files reach that
arm — a golden with no `capture` block at all, and a golden that HAS one whose
`engine_config_recorded` flag is missing or unreadable, which is the file the
reviewer actually fed it. The message now names both and asserts neither.
Verified on all four inputs: `recorded=false` → UNRECORDED, `recorded=true` →
RECORDED, no block → the tri-state line, block-with-flag-deleted → the same
tri-state line.

### One red that is not this row's

`scripts/agent-preflight.sh` reports `test_cpu_x86_llamacpp_floor` failing on
`test_a_contended_leg_is_discarded_and_never_summarised`. That is
[#618](https://github.com/mudler/vllm.cpp/issues/618) by its exact case name —
the harness is load-dependent and this box had just finished a build. The suite
and its script are **byte-identical to `origin/main`** in this diff
(`git diff --stat origin/main...HEAD -- tests/scripts/test_cpu_x86_llamacpp_floor.py
scripts/cpu-x86-llamacpp-floor.sh` is empty, against a positive control on this
row's own files that is not). Every other preflight gate is green.

## 8. Risks

- **A named profile could be mistaken for the recovered one.** Mitigated by
  saying so in the generator, in the golden and in §5.3, and by the fact that
  `nhspeed-a` reproduces prompt 2 at 26/32 — it is visibly not the lost
  configuration.
- **Three copies of the key list can drift.** The suite asserts they agree and
  owns the expectation.
- **The re-derivation needs the oracle**, which #1431 blocks. Nothing here
  depends on it; the contract admits the unattributed state precisely so this
  work did not have to wait.

## 9. Stop conditions

- Do **not** relax the contract to admit a silent golden.
- Do **not** ratify a distributional gate for this row. §4 shows the oracle is
  deterministic at a fixed configuration, which is the condition AGENTS.md
  requires be **absent**.
- Do **not** write a reconstructed configuration into the golden. An invented
  provenance is worse than a stated absence.

## 10. Now

Landed: the generator, the contract in **both** its halves — the attributed
arm's twenty keys and the unattributed arm's required, floored record — its
three gates, the truthful provenance block, and the driver line. The golden's
tokens are **byte-for-byte unchanged** — the diff against `af8170154` is 24
inserted lines and zero deletions, and its sha256 is the same before and after
the repair round.

## 11. Owed

- **The re-derivation, and it needs a decision.** Re-deriving the golden under
  `--profile nhspeed-a` would replace an unattributable reference with an
  attributable one and would change the reference #1289 is scored against — from
  32/32, 32/32, 32/32 to 32/32, 32/32, 26/32 on the oracle side. That is a
  change to a gate's reference and is the developer's call, not an implementer's.
  It is blocked on #1431 either way.
- **The index-29 top-2 margin** (#1388), also blocked on #1431. Note the
  ordering this row establishes: the margin measures how close the two
  candidates are; it does not tell you which configuration produced the
  reference. Both are needed and this one was cheaper.
- **The other goldens.** This contract is Nemotron-only by design; whether the
  rest of `tests/parity/goldens/` can name their capture configurations is
  unmeasured and is not claimed either way here.

## 12. Outcome

Recorded because the code does not say it:

- **The recovery was attempted and failed.** §2 is the negative result, and it
  is worth more than a plausible reconstruction would have been.
- **The `kv_cache_dtype` lead was closed by reading the checkpoint and the
  pinned source**, not by running anything. It looked like the best candidate
  and it is a common term.
- **The contract does not demand the configuration back.** Demanding it would
  have made this row wait for #1431 and would have made `main` red on an
  artifact nobody can currently fix. Demanding that the file *say which state it
  is in* costs nothing, is checkable today, and is the property that was
  actually missing.
- **A distributional gate was available and was rejected**, on the evidence in
  §4 rather than on preference.
- **The first round gated the SHAPE and not the SUBSTANCE, and only a mutation
  found it.** Both gate arms read as symmetric — one requires twenty keys, the
  other requires a reason and an issue — and both were green on the shipped
  artifact, so nothing in the diff looked wrong. What made the asymmetry visible
  was gutting the block and watching four gates stay green. The rule this leaves
  behind: **where a record is prose rather than structure, a truthiness check is
  not a gate.** It admits the exact artifact the row exists to refuse, and it
  admits it silently.
- **A length floor was chosen over a keyword grep, and the choice is a
  limitation, not a feature.** The floor cannot see whether the prose is true; it
  sees only that somebody removed it. That is a genuinely weaker guarantee than
  the attributed arm's, and it is recorded as weaker rather than described as
  equivalent. The alternative — asserting the text contains "distributional" or
  "reimaged" — would gate the checker's own vocabulary and would red an honest
  rewrite, which is the failure mode AGENTS.md names when it says a gate that
  fires on ordinary work is the defect.
- **A helper with one caller, and that caller a test, is dead code even in a
  script.** `unattributed_capture()` looked like production structure. It could
  not emit three of the keys the artifact it described actually carries, and
  `--capture` could never reach it. Deleting it removed a fourth copy of the
  contract that could drift and repaired a fixture that was deriving its setup
  from the module under test.
