# Citation anchors that do not rot

Issues: [#1143](https://github.com/mudler/vllm.cpp/issues/1143) (in-repo scale),
[#1139](https://github.com/mudler/vllm.cpp/issues/1139) (upstream scale),
[#1198](https://github.com/mudler/vllm.cpp/issues/1198),
[#1199](https://github.com/mudler/vllm.cpp/issues/1199).
Row: none. This is a repository-wide convention plus its gate, so it has no
owning capability row and does not belong in a capability matrix.

## Now

`GATE-SYMBOL-ANCHORS` ships green. 93 in-repo symbol anchors are checked on
every run against a recorded floor of 85, 26 of them converted here.
`model_loader.cpp` line references in tracked, non-frozen, non-locked files fell
from 112 in 47 files to 92 in 35 on this branch; the residue is listed under
`## Owed`.

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
`scripts/check-agent-record.py`. The `.agents/engine-matrix.md` exclusion was
lifted after review: the fix is one CELL inside an existing row, which adds no
row and moves no counter, so it cannot collide with a concurrent session that
appends one.

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
something half-adopted — which is why the gate lands green over 93 anchors
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
| in-repo | path is a tracked file, or a bare basename with our own source extension matching a tracked file | the FULL cited symbol must appear in that file with word boundaries on both sides |
| missing local | the path's directory is at least two components deep AND exists here, but the file does not | FAIL |
| upstream / unknown | anything else | counted, skipped |

Three counts ride alongside: how many of the in-repo checks resolved an
AMBIGUOUS basename, how many FROZEN files were skipped by prefix, and how many
UNTRACKED files carry a citation this run cannot see.

Every bucket is printed, and the three buckets must SUM to the citation count,
so a citation that stops being counted anywhere is arithmetic rather than
judgement. A run that checks ZERO in-repo citations FAILS, and so does one below
the recorded floor `MIN_IN_REPO_CHECKED` — a zero-guard over a population of
ninety is a mute switch, because one added `FROZEN_PREFIXES` entry or one
narrowed `CITATION_RE` takes the count from 91 to 1 and stays green the whole
way down. The floor carries headroom on purpose: no change has to edit it, so it
is a ratchet and not a per-PR record lock. `--min-checked` overrides it, and a
`--root` fixture tree defaults to 1.

`--upstream-root <vllm-checkout>` additionally resolves `vllm/...` citations
against the parity pin, read with `git show <pin>:<path>` after asserting the
checkout CONTAINS that commit. Opt-in and never a CI gate, because CI has no
oracle checkout — it is the instrument that would have found #1139 before a
reader did.

### A citation span is not evidence

The tautology immunity is "expectation from the CITING file, evidence from the
CITED file". Those two sides COLLAPSE when they are the same file: the
expectation is then read from the file under test, and the citation text itself
contains the symbol, so `` `a.cpp::GhostSymbol` `` written inside `a.cpp` read as
fresh over a symbol that exists nowhere else. That is #911 again, reached by a
different door, and `.agents/porting.md` now tells authors to write the full
path — which makes a file documenting its own symbols the most natural next
thing anyone writes.

Every citation span is therefore stripped from the cited file before the search.
Stripping all of them, not only the self-citing one, also closes the cross-file
form, where `a.cpp` names `Ghost` only inside its own citation of `b.cpp`. A
citation is a claim about some file; it is never a definition or a call, so it
may not stand as evidence for one.

### An ambiguous basename is checked, not skipped

A basename matching more than one tracked file used to be counted and dropped,
which silently skipped five live citations. It is checked against EVERY
candidate instead, and reported only when NONE of them contains the symbol: one
candidate containing it is exactly what the citation claims, and reporting on
the ambiguity itself would be a false accusation.

### The untracked blind spot is counted, not closed

An untracked file is absent from `git ls-files`, so neither its citations nor
its existence as a cited PATH reaches the walk. This change found that out about
itself: the first version of the test file wrote its fixtures as real local
paths, was untracked, and ran green. CI is sound because everything is tracked
there. A local pre-commit run is not, and a green run before `git add` is not a
green run.

Scanning the working tree instead would change what the gate's SUBJECT is, so
the skip stays. It is COUNTED and PRINTED, the same discipline `frozen_files`
already gets.

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
- **A dot-leading path is a citation.** It was not: the grammar's first
  character class excluded `.`, so a citation of anything under `.agents/` or
  `.github/` matched zero times and said nothing about it. Latent while nobody
  wrote one, and live the moment somebody cites a spec by symbol — which
  `.agents/porting.md` now invites.

## Tests

`tests/scripts/test_check_symbol_anchors.py`, 21 cases. Candidate lists are
sorted, so a citation's verdict and the path a stale one names do not depend on
the iteration order of a hash set; `test_an_ambiguous_basename_is_checked_against_every_candidate`
puts the symbol in the LAST candidate so that a first-candidate-only checker
reds deterministically. The load-bearing one is
`test_verdict_depends_on_the_citing_text`: one cited file, unchanged, cited
twice with different symbols, asserting the two verdicts DIFFER. A #911-shaped
checker cannot pass it, because nothing it reads varies between the two runs.

That case cannot see the self-citation door, because it uses two DIFFERENT
files. `test_a_citation_span_is_not_its_own_evidence` and
`test_a_citation_in_the_cited_file_is_not_evidence_either` close it from both
sides.

`test_this_tree_meets_the_recorded_floor` asserts the checked count against a
LITERAL 85, not against the checker's own constant. Reading the constant would
make the case a tautology: lowering the floor would lower the expectation with
it. `test_the_floor_reds_when_the_population_collapses` drives `--min-checked`
over a one-citation tree so the comparison itself is executable, and
`test_the_buckets_sum_to_the_citation_count` runs a fixture tree that populates
all three buckets at once — this tree has zero missing local paths, so asserting
the sum only here would leave that bucket droppable without a red.

`test_untracked_files_carrying_citations_are_counted` builds a real repository,
because outside one `git ls-files` fails, the checker falls back to walking the
filesystem, and the blind spot cannot exist to be measured.

## Gates

```sh
python3 scripts/check-symbol-anchors.py
python3 scripts/check-symbol-anchors.py --self-test
python3 tests/scripts/test_check_symbol_anchors.py
python3 scripts/check-symbol-anchors.py --upstream-root /path/to/vllm   # opt-in
python3 scripts/check-symbol-anchors.py --min-checked N                # override the floor
```

Registered in `scripts/agent-preflight.sh` (`CHECKERS` and `SUITES`) and in
`.github/workflows/ci.yml`.

## Evidence

Every absolute below was re-measured at the final head of this branch. The first
recorded set was not reproducible: the upstream run was recorded as 354/343 and
reproduces as 364/353, the residue as 106 in 46 to 86 in 34 and reproduces as
112 in 47 to 92 in 35, and `check-agent-record.py` was recorded as ENGINE=161
and reports ENGINE=162. Every DELTA in that set was right and every ABSOLUTE was
not, which is the shape a number quoted from an earlier run takes.

**Tree.** 618 citations in 2758 scanned files, 180 frozen files skipped, 0
untracked files carrying citations; in-repo checked 93 (fresh 93, stale 0), of
which 5 resolved an ambiguous basename; upstream/unknown 525; missing local path
0; buckets sum 618 vs 618; floor 85. `rc=0`.

**Upstream mode, at the pin.** 364 upstream anchors checked against
`5559679229bc961848b121ccdeaa8fa5d79bec98`: 353 fresh, **0 stale**, 11 naming a
file absent at the pin, which dedupe to the 10 reported lines (#1199). Not one
symbol anchor was stale across the same pin advance that broke every line anchor
#1139 examined. That is the measurement that decides the convention.

**The perturbation control.** "0 stale" says nothing until the corpus is shown
capable of going stale. Appending `Zq` to every cited symbol name, applied and
compiling, reports `in-repo checked 93 (fresh 0, stale 93)` and
`upstream checked 364 (fresh 0, stale 353, file absent 11)`: not one of the 446
fresh anchors survives the rename. Freshness is therefore a property this corpus
has to earn, not one the check hands it.

**`scripts/check-agent-record.py`.** `rc=0`, ENGINE=162 MODEL=377 QUANT=82
KERNEL=52 BACKEND=85.

**The residue instrument, spelled so somebody else gets the number.** Population
is `git ls-tree -r --name-only <rev>` minus four surfaces: `.agents/completed/**`
(frozen archive), `.agents/issue-index.md` (append-only, an edit is forbidden),
`.agents/benchmark-record.md` and `.agents/parity-ledger.md` (append-only
records). Counted is every occurrence of `model_loader\.cpp:[0-9]`, plus the
number of distinct files carrying one:

```sh
excl=(':!.agents/completed' ':!.agents/issue-index.md'
      ':!.agents/benchmark-record.md' ':!.agents/parity-ledger.md')
git grep -I -o -E 'model_loader\.cpp:[0-9]' <rev> -- "${excl[@]}" | wc -l
git grep -I -l -E 'model_loader\.cpp:[0-9]' <rev> -- "${excl[@]}" | wc -l
```

At the branch base `1f4878fdc`: 112 references in 47 files. On this branch
before merging: 92 in 35, so 20 converted. At `origin/main` `5af6e7631`: 112 in
47 again, and at this merged head 93 in 35, because `origin/main` independently
added one back in a file that already carried some. The 20 are the twenty rows
of the conversion table; the extra one is not this change's to explain away.

**Mutation, thirteen mutations, each applied and compiling, over 21 executed
cases.** `applied` is a moved sha256 plus a non-empty `git diff --stat`;
`cases` is the LAST `Ran N tests` line, because a mutation that never applied
and one that fails to build both read as a passing test otherwise.

| Mutation | applied | compile_rc | cases | rc | caught by |
|---|---|---|---|---|---|
| M1 every symbol declared present | yes | 0 | 21 | 1 | 7 cases |
| M2 the #911 tautology (expectation read from the CITED file) | yes | 0 | 21 | 1 | 7 cases incl. `test_verdict_depends_on_the_citing_text` |
| M3 vacuity guard removed | yes | 0 | 21 | 1 | `test_zero_checked_citations_is_a_failure` |
| M4 missing local path downgraded to a skip | yes | 0 | 21 | 1 | 3 cases |
| M5 frozen archive no longer skipped | yes | 0 | 21 | 1 | `test_the_frozen_archive_is_skipped_and_counted` |
| M6 word boundaries dropped | yes | 0 | 21 | 1 | 2 cases |
| M7 a citation span is evidence again | yes | 0 | 21 | 1 | 3 cases incl. both self-citation cases |
| M8 untracked files no longer counted | yes | 0 | 21 | 1 | `test_untracked_files_carrying_citations_are_counted` |
| M9 the floor comparison removed | yes | 0 | 21 | 1 | `test_the_floor_reds_when_the_population_collapses` |
| M10 one `FROZEN_PREFIXES` entry collapses the population | yes | 0 | 21 | 1 | 3 cases incl. `test_this_tree_meets_the_recorded_floor` |
| M11 one bucket dropped from the sum | yes | 0 | 21 | 1 | `test_the_buckets_sum_to_the_citation_count` |
| M12 only the first candidate of an ambiguous basename is consulted | yes | 0 | 21 | 1 | 4 cases, over three repeat runs |
| M13 the leading dot removed from the grammar | yes | 0 | 21 | 1 | `test_a_dot_leading_path_is_a_citation` |

Zero invalid mutations. Every one restored byte-for-byte against the baseline
sha256. Two of the thirteen did not report honestly on the first pass, and both
are recorded because a mutation that reads green for the wrong reason is the
failure this table exists to prevent. M12's catch depended on the iteration
order of a hash set, so `by_base` is now built over a sorted walk and the case
puts the symbol in the last candidate; it was then re-run three times. M11 was
not caught at all: this tree has zero missing local
paths, so dropping that bucket left the arithmetic intact — the case now runs a
fixture tree that populates all three buckets, and the mutation reds.

The `DISABLED_CREATION_CHECKER` stub registered in `scripts/check-pr-size.py`
fails 20 of the 21 cases, so the creation contract is rejected rather than
satisfied.

### The gate caught this change

`tests/scripts/test_check_symbol_anchors.py` first wrote its fixtures as
`src/vllm/a.cpp`. Untracked, it was invisible to `git ls-files` and the run was
green. The first commit made it tracked, and the checker immediately reported
twelve citations of a file that does not exist — in its own test file. That is
the failure this change exists to stop, arriving inside the pull request that
stops it, and it is why the fixtures are now `alpha/beta/...`: a directory that
exists in no tree falls into the skipped bucket in the real repository while
still resolving inside each temporary one. It also says something about the
instrument: an untracked file is not scanned, so a green run before `git add` is
not a green run. That lesson was first applied only to the
fixtures. It is now applied to the tool as well, which COUNTS and PRINTS the
untracked files carrying citations rather than passing over them in silence.

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
| `tests/parity/test_qwen36_spec_decode.cpp` | `:582` | `::is_dense_model` | 1548, 1661 |
| `.agents/specs/cli-serve-bench.md` ×2 | `:800-811` | `::async_engine` | 1342 |
| `.agents/specs/vt-fp8-shared-seam.md` | `:133` | `::PrintLoadBytes`, `::LoadStatsEnabled` | 206, 188 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` ×2 | `:626-641` | `::ResolveMaxNumBatchedTokens` | 701 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:704-717` | `::MakeSchedulerConfig` | 779 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:1051-1058` | `::MakeScheduler` | 812, called at 1184 |
| `.agents/specs/perf-chunked-prefill-budget-2026-08-13.md` | `:711` | `::MakeSchedulerConfig` | 786 |
| `.agents/specs/gpu-mem-util-inert.md` | `:954-959` | `::ResolveNumBlocks` | 928 |
| `.agents/specs/gpu-mem-util-inert.md` | `:718-728` | `::ResolveEnablePrefixCaching` | 718 (was correct) |
| `.agents/specs/gpu-mem-util-inert.md` | `:1081-1083,972` | `::MakeKVCacheResolved`, `::ResolveNumBlocks` | 993, 928 |

Two were already correct and are converted anyway, because a correct line
anchor is one edit away from a wrong one.

Two more were WEAKER than what they replaced and were repaired after review. A
symbol that occurs eight times localises nothing: `::max_num_batched_tokens_`
named the value being traced rather than the hop the row describes, and is now
`::MakeScheduler`, which is the hop. `::Qwen3_5MTPKind` named the enum where the
selection is a ternary on the factory flag, and is now `::is_dense_model`, which
is the thing that decides. A conversion that reds on a rename is the minimum; a
conversion that also tells the reader where to look is the point.

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
  the residue: 93 `model_loader.cpp:NNN` references in 35 files at this merged
  head, down from 112 in 47, over the population and command recorded under
  `## Evidence`. They were
  left because converting one requires deciding what the author meant: a
  heuristic pass that scanned a two-line context window around each of 92 such
  citations found 62 naming no `model_loader.cpp` symbol at all, so there is
  nothing to convert them TO without re-deriving the claim. The other 30 could
  not be converted mechanically either, because the window pulls in symbols from
  adjacent table rows; every conversion here was made by hand and verified
  individually.
- [#1139](https://github.com/mudler/vllm.cpp/issues/1139) is CLOSED by this
  change. Its last cell, the `KV-WARMUP-PROFILE` row in
  `.agents/engine-matrix.md`, now carries
  `vllm/v1/worker/gpu_worker.py::determine_available_memory`,
  `vllm/v1/worker/gpu/model_runner.py::profile_run` and
  `vllm/v1/worker/gpu/model_runner.py::model_memory_usage` in place of the three
  line anchors that pointed at unrelated code at the pin. All three resolve
  under `--upstream-root`; the upstream count moved 360 to 364 checked and 349
  to 353 fresh, with stale still zero. The edit is a CELL inside an existing
  row: it adds no row and moves no counter, so `check-agent-record.py` reports
  the same ENGINE=162 before and after.
- [#1198](https://github.com/mudler/vllm.cpp/issues/1198) - now FOUR specs
  assert loader behaviour the loader no longer has, found while verifying
  anchors for conversion. The fourth is
  `.agents/specs/gguf-mtp-spec-decode.md`, which says in the present tense that
  `LoadedEngine::FromModelDir` refuses a GGUF speculative target at
  `model_loader.cpp:717-723` - the same removed refusal and the same stale range
  as the `gguf-dflash-draft.md` row already on that issue. `717-723` is
  `ResolveEnablePrefixCaching`.
- [#1199](https://github.com/mudler/vllm.cpp/issues/1199) - 11 upstream symbol
  citations, deduping to 10 reported lines, name vLLM files absent at the pin;
  six of them are `.agents/model-matrix.md` rows whose repair is a claim about
  vLLM, not a path edit.
- **One-component directories still fall through to the upstream bucket.** A
  typo in `tests/foo.cpp` is not reported, because vLLM has real files sitting
  directly under top-level names we share and nothing in the path separates
  them. This is a deliberate miss and never a false accusation; it is recorded
  here rather than fixed, because narrowing it needs a rule that can tell the
  two trees apart, which is what `--upstream-root` already does for the paths it
  covers.

## Stop conditions

- Stop if a conversion cannot be verified true of the named symbol. Record it
  under `## Owed` instead.
- Stop rather than storing expectations in a shared table. That surface is a
  lock, and a lock is a worse defect than the staleness it would catch.
