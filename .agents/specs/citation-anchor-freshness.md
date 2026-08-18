# Citation anchors that do not rot

Issues: [#1143](https://github.com/mudler/vllm.cpp/issues/1143) (in-repo scale),
[#1139](https://github.com/mudler/vllm.cpp/issues/1139) (upstream scale),
[#1198](https://github.com/mudler/vllm.cpp/issues/1198),
[#1199](https://github.com/mudler/vllm.cpp/issues/1199).
Row: none. This is a repository-wide convention plus its gate, so it has no
owning capability row and does not belong in a capability matrix.

## Now

`GATE-SYMBOL-ANCHORS` ships green. 85 in-repo symbol anchors are checked on
every run, 26 of them converted here; 96 `model_loader.cpp` line citations
remain unconverted and are listed under `## Owed`.

## Scope

One defect, two scales.

**In-repo (#1143).** `src/vllm/entrypoints/model_loader.cpp` is cited by
absolute line number from 109 sites across 45 files. It is ~1740 lines and is
edited by almost every engine and model row, so an edit near its top silently
retargets every citation below it, in files the editing change never opens. One
45-line insertion moved 203 references at once. Nobody could then tell a
citation that change broke from one that was already wrong.

**Upstream (#1139).** All three `vllm/v1/worker/**` line anchors on the
`KV-WARMUP-PROFILE` row point at unrelated code at the parity pin
`555967922` — a `DraftModelSpeculator.set_attn(...)` call, a `torch.zeros(...)`
argument, and a comment about `max_split_size_mb`. Two of them had already been
copied into `gguf_device_fit.h` and `expert-streaming.md`, so staleness was
propagating by copy-paste.

Out of scope: splitting `model_loader.cpp`, and any change to
`.agents/engine-matrix.md` or `scripts/check-agent-record.py` (see `## Owed`).

## The options, and why option 4

1. **Repair all 109 citations.** Fixes today, guarantees tomorrow's recurrence.
   The next edit near the top of that file breaks them all again, and the repair
   itself would launder pre-existing debt: rewriting a citation from the current
   tree makes a wrong one and a broken one indistinguishable, which is precisely
   what #1143 says nobody can do today.
2. **Repair plus a checker over line anchors.** A line number carries no claim,
   so the only thing such a checker can compare against is a stored expectation
   table — a record surface every pull request would have to edit, which
   AGENTS.md `## Records` calls a lock and a defective gate. Reading the
   expectation out of the cited file instead is the #911 tautology: that shape
   reported 27/27 FRESH while five anchors pointed at unrelated code.
3. **Change the convention to symbols.** Removes the class rather than
   re-baselining it, and is the largest.
4. **Scoped combination — chosen.** Adopt the symbol convention, gate it, and
   convert a verified subset now, recording the rest as visible debt.

Option 4 wins on one observation the other three miss: **the convention already
exists in this tree.** 539 `` `path::Symbol` `` citations are already written,
`.agents/model-matrix.md` uses the form for every upstream model anchor, and
they had never been checked. This change is not an invention, it is finishing
something half-adopted — which is why the gate lands green over 85 anchors
instead of needing a flag day.

It also resolves the tautology hazard structurally rather than by care. Under
the symbol convention the two sides of the comparison come from different
places: the EXPECTATION is the symbol name, written by the citing author, living
in the CITING file; the EVIDENCE is the cited file's text. Nothing is derived
from the cited file, and nothing is stored centrally, so the surface is not a
lock — a citation's expectation rides in the file that owns the claim.

## Design

`scripts/check-symbol-anchors.py`.

A citation is a single backtick span holding a path with a source extension,
`::`, and a possibly-qualified identifier. The extension requirement keeps
ordinary C++ prose out: `Qwen3_5MTPKind::kMoe` has no path in front of it.

| Bucket | Rule | Verdict |
|---|---|---|
| in-repo | path is a tracked file, or a bare basename with our own source extension matching exactly one tracked file | the FULL cited symbol must appear in that file with word boundaries on both sides |
| missing local | the path's directory is at least two components deep AND exists here, but the file does not | FAIL |
| upstream / unknown | anything else | counted, skipped |
| ambiguous basename | basename matches more than one tracked file | counted, skipped |

Every bucket is printed. A gate that cannot say how many things it examined has
not reported, so a run that checks ZERO in-repo citations FAILS: the tree
carries symbol anchors, and a zero means the grammar or the walk stopped
matching, not that everything is fresh.

`--upstream-root <vllm-checkout>` additionally resolves `vllm/...` citations
against the parity pin, read with `git show <pin>:<path>` after asserting the
checkout CONTAINS that commit. Opt-in and never a CI gate, because CI has no
oracle checkout — it is the instrument that would have found #1139 before a
reader did.

### What it deliberately does not do

- **Not a definition check.** "`ModelRegistry::Load`, which `model_loader.cpp`
  calls" is a legitimate citation, and a definition-only rule would reject it. A
  rename still reds, because a rename removes the token from the file.
- **Not a line-anchor check.** A bare line number carries no claim, so there is
  nothing honest to check it against.
- **Depth two, not a root list.** `tests/` was the first cut and it was wrong:
  vLLM has a `tests/` too, and twenty correct upstream anchors went red as
  broken local paths on the first run. Directory existence at depth two
  separates the cases without a list to maintain. The cost is that a typo in a
  one-component directory falls through to the upstream bucket — a miss, never a
  false accusation.
- **No network.**

## Tests

`tests/scripts/test_check_symbol_anchors.py`, 13 cases. The load-bearing one is
`test_verdict_depends_on_the_citing_text`: one cited file, unchanged, cited
twice with different symbols, asserting the two verdicts DIFFER. A #911-shaped
checker cannot pass it, because nothing it reads varies between the two runs.

`test_this_tree_is_green_over_a_non_zero_population` parses the checked count
out of the run and asserts it is above zero, so a green that examined nothing
cannot be reported as a pass.

## Gates

```sh
python3 scripts/check-symbol-anchors.py
python3 scripts/check-symbol-anchors.py --self-test
python3 tests/scripts/test_check_symbol_anchors.py
python3 scripts/check-symbol-anchors.py --upstream-root /path/to/vllm   # opt-in
```

Registered in `scripts/agent-preflight.sh` (`CHECKERS` and `SUITES`) and in
`.github/workflows/ci.yml`.

## Evidence

**Tree, after conversion.** 565 citations in 2750 scanned files, 180 frozen
files skipped; in-repo checked 85 (fresh 85, stale 0); upstream/unknown 475;
ambiguous basename 5; missing local path 0. `rc=0`.

**Upstream mode, at the pin.** 354 upstream anchors checked against
`5559679229bc961848b121ccdeaa8fa5d79bec98`: 343 fresh, 0 stale, 11 naming a file
absent at the pin (#1199). Not one symbol anchor was stale across the same pin
advance that broke every line anchor #1139 examined. That is the measurement
that decides the convention.

**Mutation, six mutations, each applied, compiling, over 13 executed cases:**

| Mutation | compile_rc | cases_ran | rc | caught by |
|---|---|---|---|---|
| M1 every symbol declared present | 0 | 13 | 1 | 4 cases |
| M2 the #911 tautology (expectation from the CITED file) | 0 | 13 | 1 | 4 cases incl. `test_verdict_depends_on_the_citing_text` |
| M3 vacuity guard removed | 0 | 13 | 1 | `test_zero_checked_citations_is_a_failure` |
| M4 missing local path downgraded to a skip | 0 | 13 | 1 | `test_a_local_path_that_does_not_exist_is_reported` |
| M5 frozen archive no longer skipped | 0 | 13 | 1 | `test_the_frozen_archive_is_skipped_and_counted` |
| M6 word boundaries dropped | 0 | 13 | 1 | 2 cases |

Every mutation restored byte-for-byte, verified by sha256. The
`DISABLED_CREATION_CHECKER` stub registered in `scripts/check-pr-size.py` fails
12 of the 13 cases, so the creation contract is rejected rather than satisfied.

## Converted here, and verified how

26 anchors across 16 files. Each was checked by locating the named symbol in
`src/vllm/entrypoints/model_loader.cpp` and confirming the surrounding sentence
is still TRUE of that symbol. Most were stale before this change; the real line
is given where it differs from the cited one.

| Site | Was | Now | Real line |
|---|---|---|---|
| `include/vllm/v1/core/sched/scheduler.h` | `model_loader.cpp:176 MakeScheduler` | `::MakeScheduler` | 812 |
| `include/vllm/v1/core/sched/scheduler.h` | `model_loader.h:206 scheduler_` | `::scheduler_` | `model_loader.h:556` |
| `src/vllm/multimodal/ltx2_video.cpp` | `:75-104` | `::SelectQueueForModel` | 148 |
| `src/vllm/multimodal/minimax_h3_video.cpp` | `:76-104` | `::SelectQueueForModel` | 148 |
| `tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp` | `:100-115` | `::ResolveAutoDevice` | 100 (was correct) |
| `tests/vllm/entrypoints/test_loaded_engine_dense.cpp` | `:1081-1083,972` | `::MakeKVCacheResolved`, `::ResolveNumBlocks` | 993, 928 |
| `tests/vllm/model_executor/layers/test_fp8_block_quant.cpp` | `:1613` | `::FromModelDir` | `ModelRegistry::Load` at 1541/1706/1724 |
| `tests/vllm/model_executor/layers/test_fp8_block_quant.cpp` | `:1536` | `::RefuseUnsupportedWeightOffload` | 1629 |
| `tests/vllm/model_executor/test_gguf_device_fit.cpp` | `:1452-1453` | `::FromModelDir` | condition at 1533/1546 |
| `tests/vllm/v1/worker/test_runner.cpp` | `:1007-1023` | `::runner_` | 1135 |
| `tests/vllm/config/test_speculative_mtp_depth.cpp` | `:831` | `::ResolveMtp` | 911 |
| `tests/parity/test_qwen36_spec_decode.cpp` | `:582` | `::Qwen3_5MTPKind` | 1550, 1663 |
| `.agents/specs/cli-serve-bench.md` ×2 | `:800-811` | `::async_engine` | 1342 |
| `.agents/specs/vt-fp8-shared-seam.md` | `:133` | `::PrintLoadBytes`, `::LoadStatsEnabled` | 206, 188 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` ×2 | `:626-641` | `::ResolveMaxNumBatchedTokens` | 701 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:704-717` | `::MakeSchedulerConfig` | 779 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:1051-1058` | `::max_num_batched_tokens_` | 1166/1173/1188 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:711` | `::MakeSchedulerConfig` | 786 |
| `.agents/specs/gpu-mem-util-inert.md` | `:954-959` | `::ResolveNumBlocks` | 928 |
| `.agents/specs/gpu-mem-util-inert.md` | `:718-728` | `::ResolveEnablePrefixCaching` | 718 (was correct) |
| `.agents/specs/gpu-mem-util-inert.md` | `:1081-1083,972` | `::MakeKVCacheResolved`, `::ResolveNumBlocks` | 993, 928 |

Two were already correct and are converted anyway, because a correct line
anchor is one edit away from a wrong one.

## Risks

- **A citation can be converted to a symbol that is present but wrong.** The
  gate cannot see that; only review can. Mitigated by converting nothing whose
  surrounding claim could not be verified true — which is why #1198 exists
  instead of eleven more conversions.
- **The `--upstream-root` mode has a known false positive** on fixture text
  inside `tests/scripts/test_agent_record.py`. It is opt-in and never gates, so
  the cost is one noisy line rather than a red lane.
- **A bare basename can become ambiguous** when a second file of that name
  lands, silently dropping the citation into the skipped bucket. The counts are
  printed, and full paths are the recommended form.

## Owed

- [#1143](https://github.com/mudler/vllm.cpp/issues/1143) stays OPEN and owns
  the residue: 96 of the 120 touchable `model_loader.cpp:NNN` line citations are
  unconverted. They were left because converting one requires deciding what the
  author meant, and 62 of the 102 measured citations name no symbol at all in
  their sentence, so there is nothing to convert them TO without re-deriving the
  claim. Deliberately excluded surfaces: `.agents/completed/**` (frozen
  archive), `.agents/issue-index.md` (append-only, an edit is forbidden),
  `.agents/benchmark-record.md` and `.agents/parity-ledger.md` (append-only
  records).
- [#1139](https://github.com/mudler/vllm.cpp/issues/1139) stays OPEN. Its
  remaining fix is one cell, `.agents/engine-matrix.md:115`, which this change
  was instructed not to touch because concurrent sessions hold that file
  alongside the hardcoded `ENGINE` count in `scripts/check-agent-record.py`. The
  replacement text is verified and ready to paste: the anchors are
  `vllm/v1/worker/gpu_worker.py::determine_available_memory`,
  `vllm/v1/worker/gpu/model_runner.py::profile_run`, and
  `vllm/v1/worker/gpu/model_runner.py::model_memory_usage`. All three resolve at
  the pin under `scripts/check-symbol-anchors.py --upstream-root`; the three
  line anchors they replace do not.
- [#1198](https://github.com/mudler/vllm.cpp/issues/1198) — three specs assert
  loader behaviour the loader no longer has, found while verifying anchors for
  conversion.
- [#1199](https://github.com/mudler/vllm.cpp/issues/1199) — 10 upstream symbol
  citations name vLLM files absent at the pin, six of them `.agents/model-matrix.md`
  rows whose repair is a claim about vLLM, not a path edit.

## Stop conditions

- Stop if a conversion cannot be verified true of the named symbol. Record it
  under `## Owed` instead.
- Stop rather than storing expectations in a shared table. That surface is a
  lock, and a lock is a worse defect than the staleness it would catch.
