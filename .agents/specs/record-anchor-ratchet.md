# ENG-RECORD-ANCHOR-RATCHET — the anchor checker range-checks its own citations and reports nothing

Issue: [#632](https://github.com/mudler/vllm.cpp/issues/632)
Row: `ENG-RECORD-ANCHOR-RATCHET` ([engine-matrix.md](../engine-matrix.md))

## The defect

Records cite code as `` `server_main.cpp:505` ``. Code moves; the citation does
not. The checker that appears to catch that reports nothing.

**This section was wrong when it was written, and the correction is the point of
this row.** It claimed the checker "walks **only** `LINK_RE` matches", so the
bare `` `file.cpp:123` `` form "is never parsed". That is false.
`RAW_LOCAL_ANCHOR_RE` has parsed the bare form since `ee511ca8a` (2026-07-10),
under the prefixes `src`, `include`, `tests`, `examples`, `cmake`, `scripts`,
`tools` and `.github/workflows`, plus `CMakeLists.txt`, and it range-checks each
one. Every one of the 38 offenders this row records sits under those prefixes,
so every one was already parsed. The false premise reached six surfaces before a
fresh review caught it. It is corrected here rather than appended to, so nobody
reads the wrong version first.

**What the checker actually did.** `local_line_anchors` reads both forms and
range-checks both. What it does not do is REPORT. On a missing file and on an
out-of-range line the loop runs `continue`, so the failing anchor never enters
the returned list. `is_code_anchor` then answers with `any()`, so one good
sibling in the same cell satisfies the row. A bad anchor is therefore invisible
twice: dropped by the parser, then covered by a neighbour.

Three gaps compound:

1. **No symbol test.** The check asks whether the line EXISTS, never whether it
   holds what the prose says. **32 of the 38** offenders recorded here are IN
   RANGE, so a range check could not have found any of them.
2. **`any`, not `all`.** `is_code_anchor` returns true if any anchor in the cell
   qualifies, so one good link covers arbitrarily many rotted citations beside
   it. That is correct for the STATE gate and the `any` stays; it is why nothing
   counted the others.
3. **State.** `EVIDENCED_STATES` omits `ACTIVE` and `READY`, so 92 live rows got
   no anchor check at all.

And a fourth, which is what kept the first three invisible: **there was no
report.** A dropped anchor produced no output at any verbosity.

### Two populations, two ratios

Quoting one population's ratio as if it were the other's is what produced the
false claim above, so both are stated with their denominator. Measured at
`8daa67b39`, the head before this branch merged `main` a second time, counting a
citation only where it is the WHOLE of a backtick span, which is what the parser
requires. An earlier count of 2134 used a looser method that
matched a `path:line` token anywhere inside a span, which is why it is larger.

**Population A — every citation form a reader sees in the five matrices, ours
and upstream:**

| Matrix | link anchors | bare `path:line` | bare under a `RAW_LOCAL_ANCHOR_RE` prefix |
|---|---:|---:|---:|
| `engine-matrix.md` | 24 | 733 | 366 |
| `model-matrix.md` | 14 | 629 | 82 |
| `quantization-matrix.md` | 182 | 47 | 4 |
| `kernel-matrix.md` | 144 | 155 | 30 |
| `backend-matrix.md` | 128 | 144 | 43 |
| **total** | **492** | **1708** | **525** |

**1017 of 2200 forms (46.2%) were already parsed.** Most of the rest are
upstream paths (`vllm/model_executor/...py:123`, `csrc/...cu:44`) that no local
checker can validate, which is why this ratio answers no question about coverage
on its own.

**Population B — the citations this ratchet classifies**, meaning the `code` and
`tests` cells of rows in `RECORD_ANCHOR_STATES` that resolve against this tree.
This is the population that matters:

| | count | share |
|---|---:|---:|
| in scope | 867 | 100% |
| already parsed AND range-checked before this row | 832 | **96.0%** |
| genuinely new to parsing (`.agents/`, `docs/`, `website/`) | 35 | 4.0% |
| yield no inferable symbol, `OK` by construction | 801 | 92.4% |

**Every offender recorded then sat in the 96%.** What this row adds is the symbol test and
the report, not the parser. The parser extension is real but small, and claiming
it as the defect was the error.

## Why a range check is not enough

Every stale anchor found in the 2026-08-13/14 campaign was **in range**:

- `docs/USAGE.md:902` → the parser-count line had moved to `:1126`; `:902` was a
  valid line (a container-tag table row).
- `multimodal.py:17-43` → the dataclass block ends at `:45`; `:43` is a valid
  line (a docstring).
- `server_main.cpp:308` → after a 4-line comment landed above it, `:308` was a
  valid line (a different table entry).

A range check passes all three. Only "does this line contain the symbol named
beside it" separates them. Each was caught by a human reading, and each cost a
round-trip.

Sharpest instance: a repair round whose **stated purpose was anchor precision**
rewrote the very line carrying `multimodal.py:17-43` and left the anchor stale.
Every gate stayed green.

## Design — ratchet, not cleanup

Enforcing the symbol test over the whole backlog in one landing would surface an
unknown amount of unrelated rot. Mirror the
`device-leakage` shape instead (`scripts/device-leakage-baseline.json`), which
this repo already trusts:

- Extend the parser to recognise **bare `` `path:line` `` and `path:line-line`**
  in addition to markdown links.
- For each resolvable citation, classify: **OK** (line exists and contains the
  symbol named adjacent to it), **STALE** (line exists, symbol not found),
  **BROKEN** (out of range or missing file).
- Record today's STALE+BROKEN count in `scripts/record-anchor-baseline.json`.
- **Fail if the count rises.** Also fail if it falls without the baseline being
  lowered in the same commit — the device-leakage rule, so improvements are
  banked rather than silently absorbed.
- `--report` prints every offender with file, line, and the symbol expected.

The symbol test is deliberately conservative: only citations where a symbol is
inferable from the adjacent backticked token are classified STALE; anything
ambiguous stays OK. A checker that cries wolf gets disabled.

## Scope

- `ACTIVE`/`READY` join the anchor COUNT as part of this row, since gap 3 is
  cheap and the ratchet absorbs whatever it surfaces.
- `EVIDENCED_STATES` itself is deliberately NOT widened, and that is a scope
  decision rather than an oversight. Making `ACTIVE` and `READY` *require* an
  anchor raises 85 errors across 53 rows that carry prose evidence today, which
  is the bulk cleanup this row exists to avoid. (The unit is errors, not rows:
  the contract check emits one per missing anchor field, so 32 of the 53 raise
  two and 21 raise one.) The 53 are 37 `ACTIVE` and 16 `READY`, and by matrix
  26 engine, 11 backend, 7 model, 6 kernel and 3 quantization.
  `RECORD_ANCHOR_STATES` is therefore a separate, wider set.
- `is_code_anchor`'s `any` semantics stay for the STATE gate (a row is still
  evidenced if it has one good anchor); the ratchet counts **every** citation
  independently, which is where `any` was hiding rot.

## Upstream chain

**None, and that is a finding rather than an omission.** vLLM has no analogue:
it keeps no stable-ID inventory of its own source, so it has nothing to cite and
nothing to rot. The nearest thing in this tree is the DSR ratchet
(`scripts/check-device-leakage.py`, work row `S1` of
[accelerator-seam-audit.md](accelerator-seam-audit.md)), which is likewise
local-only and is the shape this row mirrors deliberately: same baseline file,
same `--report` / `--write-baseline` idiom, same fails-in-both-directions rule.
The discipline being enforced is AGENTS.md §"vLLM is the reference" — "cite the
`file:line` you ported from" — so this row is what makes that rule checkable
rather than aspirational.

## Our baseline

Re-derived at this head after merging 161 commits of `main`, not carried from
any earlier number. The population tables are in `## The defect` above and are
not repeated here.

**Rot: 38 — 32 `STALE`, 6 `BROKEN`, against 844 `OK`.**

The set is the 40 the fresh review verified by hand, minus one.
`KERNEL-ATTN-MLA-SPARSE` cites `include/vllm/v1/attention/backend.h:271` for
`get_kv_cache_shape` and now reads `OK`. That is not a repair. The real
declaration is at `:341`; drift on `main` moved a ROCm comment naming the symbol
onto `:271`, and the symbol test asks only whether the cited lines contain the
name. It is a measured limit of the conservative rule, recorded rather than
worked around, because tightening it would need a parser per language.
[#1287](https://github.com/mudler/vllm.cpp/issues/1287) tracks that false
negative, and the second-order cost it carries: because the ratchet fails when a
bucket FALLS without the baseline being lowered in the same commit, unrelated
drift that parks a comment on a cited line forces the next contributor to bank
an improvement that never happened.

Each merge also turns this row's own anchors `STALE`, by moving the lines they
cite in `check-agent-record.py`. Seven moved on the first merge and five on the
second; both times they were repaired, not banked, because a row arguing that
stale anchors matter may not carry them. **No new offender was banked.**

The `BROKEN` bucket fell 7 to 6 on the second merge, and `72bd06a5a` is why:
that record reconciliation replaced the `SERVE-ASYNC-LLM` citation
`` `examples/server/main.cpp:230-247` `` with a bare `` `examples/server/main.cpp` ``,
so the anchor stopped being counted rather than being repaired. Banking it was
mandatory under the two-way rule, and it is the worked example behind #1287.

By top-level directory the 38 are: `src` 24, `include` 5, `examples` 4,
`scripts` 2, `tests` 2, `cmake` 1. Every one is under a prefix the old parser
already read.

**Specs are not policed by this ratchet.** `RECORD_ANCHOR_FIELDS` reads matrix
row cells, and no spec body is in `MATRIX_PATHS`, so the `file:line` citations
in THIS file are checked by nothing. #911 tracks that gap over 315 spec files.
The anchors here are therefore pinned by hand at the final head and will rot the
same way. Read them as of the commit that lands them.

## Port map

Nothing is ported; everything here is written from scratch against the local
record surface, and is recorded as such.

| Piece | Where |
|---|---|
| bare `` `path:line` `` / `path:line-line` parser | `BARE_CITATION_RE`, `cell_citations` in `scripts/check-agent-record.py` |
| markdown-link parser (retained) | `LINK_RE` + `LINE_FRAGMENT_RE`, read by the same `cell_citations` |
| adjacent-symbol inference | `looks_like_symbol` + the neighbour walk in `cell_citations` |
| OK / STALE / BROKEN classifier | `classify_citation` |
| row and column scope, incl. gap 3 | `RECORD_ANCHOR_STATES`, `RECORD_ANCHOR_FIELDS` |
| the ratchet itself | `check_record_anchors`, `write_record_anchor_baseline`, `scripts/record-anchor-baseline.json` |
| the offender report | `record_anchor_report`, reached by `--report` |

## Tests to port

None to port — vLLM has no such checker. The seven cases below are written
RED-first against the shape of the defect, in the file
`scripts/check-pr-size.py` already names as this checker's companion evidence
(`tests/scripts/test_agent_record.py`).

## Dependencies

None. No GPU, no build, no network, no new package: the checker is standard
library and the suite runs under `python3 -m unittest`. It depends only on the
row parser already in `scripts/check-agent-record.py`, which is why the ratchet
lives in that file rather than in a new script — a separate checker would have
had to re-derive `parse_claim_rows` and `field_index`, and would have dragged a
`CREATION_MUTATIONS` entry through `scripts/check-pr-size.py` for no gain.

## Work breakdown

| Work | Item |
|---|---|
| W1 | The seven RED-first cases, run and captured red before any implementation |
| W2 | Parser: bare citations beside the existing link form |
| W3 | Classifier: OK / STALE / BROKEN, with the conservative symbol rule |
| W4 | Baseline file, the two-way gate, `--report` and `--write-baseline` |
| W5 | Gap 3: `ACTIVE`/`READY` join the counted states |
| W6 | Wiring (`agent-preflight.sh`, the `agent-record` CI job) and the record |

## Tests

RED-first, in `tests/scripts/`:

| Case | Asserts |
|---|---|
| a bare `path:line` pointing at the wrong line | counted STALE (fails today: invisible) |
| a bare `path:line` out of range | counted BROKEN |
| a correct bare citation | counted OK |
| a cell with one good link and one rotted bare citation | the rotted one is still counted (kills `any`) |
| an `ACTIVE` row with a rotted anchor | counted (fails today: state excluded) |
| baseline raised without a reduction | REFUSED |
| `--write-baseline` on a tree with other record errors | REFUSED, file unchanged (#1270) |

The fourth is the load-bearing one — it is the exact shape that let rot hide.
Mutate the ratchet into a report-only pass and prove the count case reds.

## Gates

No GPU, no build, no network — this is a Python checker and its tests. The
exact invocations, each of which genuinely fails when the row regresses:

- `python3 scripts/check-agent-record.py --report` — the gate itself; prints
  every offender and reds on either direction of the ratchet.
- `python3 tests/scripts/test_agent_record.py` — the mutation suite, including
  the ten `RecordAnchorRatchet` cases.
- `scripts/agent-preflight.sh --staged` — the whole record gate over the staged
  change.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD` — proves the
  checker change red-before / green-after against its companion suite.
- `python3 scripts/check-commit-trailers.py --range "$(git merge-base origin/main HEAD)..HEAD"`
  — run EXPLICITLY over the merge-base range, because `agent-preflight.sh`
  silently skips it when the branch is behind `main` (#653).
- `python3 scripts/check-commit-style.py --range "$(git merge-base origin/main HEAD)..HEAD"`
  — the same skip applies to the style gate that landed with
  `POLICY-SINGLE-PR-AND-STYLE`, so it is run explicitly for the same reason.
- `python3 scripts/check-public-doc-tables.py` — this row writes `docs/STATUS.md`
  and `docs/BENCHMARKS.md`, which carry a size ratchet and a per-cell limit.

## Risks / decisions

- **Risk**: the baseline becomes a parking lot. Mitigated by the
  fails-if-it-falls rule, which forces every improvement to be banked, and by
  `--report` naming offenders so the backlog is legible rather than a number.
- **Decision**: bare citations are parsed but the **symbol** test is advisory
  where no symbol is inferable. Precision over recall — a false STALE on a
  correct citation would train people to ignore the gate.
- **Decision**: no historical record (`benchmark-record.md`, `parity-ledger.md`,
  `state-events/`) is in scope. Those cite code as it was and are evidence, not
  instructions.
- **Out of scope**: rewriting existing citations. The ratchet lets them be fixed
  by whoever next touches the row.

## Now

`ACTIVE` — the parser, the classifier, the baseline and the cases are
implemented and green; the gate is wired into `scripts/agent-preflight.sh` (via
`check-agent-record`) and into the `agent-record` CI job as `--report`.

**Where the conservative line was drawn, and what it cost.** Precision was
chosen over recall at five points, each measured rather than guessed:

1. **Columns, not paths.** Only `code` and `tests` cells are read. This is what
   keeps the upstream citations out of the count structurally; a path-prefix
   heuristic would have had to guess, and `tests/`, `cmake/`, `docs/`, `src/`
   and `tools/` all collide with upstream references in this record.
2. **Whole-span citations only.** A bare citation must be the entire content of
   a backtick span, so prose is never parsed as a path.
3. **Missing files need a parent we own.** A three-component path whose parent
   directory exists here is `BROKEN` (`registry.cpp` after the rename to
   `model_registry.cpp`); a shallower one is skipped, because llama.cpp's
   `src/llama-model.cpp` and vLLM's `cmake/utils.cmake` would otherwise be
   blamed on us.
4. **Symbols only from an immediately adjacent backtick span**, with whitespace
   or one `(` between. Never from prose.
5. **The span must look like a symbol**: an identifier, 4+ characters, carrying
   `_`, `::`, `()` or an uppercase letter, and not starting with `_`. `bf16` and
   `nvfp4` sit beside citations constantly and are not symbols. The leading-`_`
   exclusion was added after a measured false positive: `` the text-only
   `_ModelInfo` `qwen3_5_common.h:42` `` names the `ModelInfo` on that line, and
   the anchor was right.

801 of the 867 in-scope citations yield no inferable symbol and are `OK` by
construction, which is 92.4%, or about 13 in 14. That polarity is the point: a
gate that fires is believed.

The 38 recorded offenders are deliberately NOT repaired here. They are the
backlog the ratchet exists to hand to whoever next touches each row.

The first fresh review returned FAIL on the recorded justification and PASS on
the design, so nothing here was redesigned. What changed is the defect
statement, corrected on six surfaces, every ratio restated against its
denominator, and the numbers re-derived after 161 commits of `main`. #1270 was
found and fixed in the same flow: `--write-baseline` banked a baseline from a
tree that had failed other record checks.

The second fresh review returned FAIL on the record text alone, and the four
findings were record corrections rather than design changes. Two counts were
stale by exactly the case #1270 added, so `## Gates`, the claim and the commit
said ten cases while the row said nine and `## Work breakdown` said six.
`46 rows` was wrong when it was written: `22e6294c6` presented it as re-derived,
and the figure was 50 at that commit and at `8daa67b39`. It is re-derived again
here after the second merge of `main`, which added three `READY` rows and made
it 53. The fourth finding is #1287, filed for the comment-satisfies-symbol false
negative that `## Our baseline` had recorded honestly but without naming its
downstream cost.
