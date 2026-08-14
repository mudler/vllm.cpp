# ENG-RECORD-ANCHOR-RATCHET — the anchor checker sees 18% of its own citations

Issue: [#632](https://github.com/mudler/vllm.cpp/issues/632)
Row: `ENG-RECORD-ANCHOR-RATCHET` ([engine-matrix.md](../engine-matrix.md))

## The defect

Records cite code as `` `server_main.cpp:505` ``. Code moves; the citation does
not. Nothing catches it — and the checker that appears to is looking at a
different thing.

`scripts/check-agent-record.py:545`:

```python
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
```

`local_line_anchors` (`:941`) walks **only** `LINK_RE` matches, and only those
carrying an `#L<n>` fragment (`LINE_FRAGMENT_RE`, `:553`). So the only citation
form it can see is the markdown link `[label](path#L505)`.

Measured across the five matrices:

| Matrix | link anchors (seen) | bare `path:line` (invisible) |
|---|---:|---:|
| `engine-matrix.md` | 19 | **1064** |
| `model-matrix.md` | 14 | **662** |
| `kernel-matrix.md` | 144 | 215 |
| `quantization-matrix.md` | 182 | 44 |
| `backend-matrix.md` | 121 | 152 |
| **total** | **480** | **2137** |

**18.3%** of citations are examined. The dominant form in this repo is the one
the checker never looks at.

Three gaps compound:

1. **Form.** Bare `` `path:line` `` is not parsed at all (`:941-950`).
2. **`any`, not `all`.** `is_code_anchor` (`:979-985`) returns true if **any**
   anchor in the cell qualifies, so one good link covers arbitrarily many rotted
   citations beside it.
3. **State.** `EVIDENCED_STATES` (`:530`) omits `ACTIVE` and `READY`, so those
   rows get no anchor check at all.

And even inside the 18%, `local_line_anchors:957` validates only
`start >= 1 and end <= line_count` — that the line **exists**, never that it is
what the prose says it is.

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

Enforcing correctness over 2137 previously-unchecked citations would surface an
unknown backlog in one landing, unrelated to the change itself. Mirror the
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

- `ACTIVE`/`READY` join the anchor check as part of this row, since gap 3 is
  cheap and the ratchet absorbs whatever it surfaces.
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

Re-derived at `0e8b15d56`, not carried from the numbers this spec was written
with. Across the five matrices, counting `path:line` tokens inside backtick
spans:

| Matrix | link anchors (seen) | bare `path:line` (invisible) |
|---|---:|---:|
| `engine-matrix.md` | 19 | 1100 |
| `model-matrix.md` | 14 | 658 |
| `kernel-matrix.md` | 143 | 179 |
| `quantization-matrix.md` | 182 | 52 |
| `backend-matrix.md` | 121 | 145 |
| **total** | **479** | **2134** |

**17.2%** of citation forms were examined. Of the 2134 bare citations, only 688
resolve to a file in this tree; the other 1446 are upstream paths
(`vllm/model_executor/...py:123`, `csrc/...cu:44`) that no local checker can
validate. That is what fixes the classifier's scope: the ratchet reads the
`code` and `tests` cells only, never the `upstream` column.

Measured rot at the same commit: **40** — 33 `STALE`, 7 `BROKEN`, against 800
`OK`. Every one of the 40 was verified by hand against the cited file before the
baseline was written.

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

None to port — vLLM has no such checker. The six cases below are written
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
| W1 | The six RED-first cases, run and captured red before any implementation |
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

The fourth is the load-bearing one — it is the exact shape that let rot hide.
Mutate the ratchet into a report-only pass and prove the count case reds.

## Gates

No GPU, no build, no network — this is a Python checker and its tests. The
exact invocations, each of which genuinely fails when the row regresses:

- `python3 scripts/check-agent-record.py --report` — the gate itself; prints
  every offender and reds on either direction of the ratchet.
- `python3 tests/scripts/test_agent_record.py` — the mutation suite, including
  the nine `RecordAnchorRatchet` cases.
- `scripts/agent-preflight.sh --staged` — the whole record gate over the staged
  change.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD` — proves the
  checker change red-before / green-after against its companion suite.
- `python3 scripts/check-commit-trailers.py --range "$(git merge-base origin/main HEAD)..HEAD"`
  — run EXPLICITLY over the merge-base range, because `agent-preflight.sh`
  silently skips it when the branch is behind `main` (#653).

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
   keeps 1446 upstream citations out of the count structurally; a path-prefix
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

About six citations in seven yield no inferable symbol and are `OK` by
construction. That polarity is the point: a gate that fires is believed.

Next: fresh scoped review of the immutable head, then merge. The 40 recorded
offenders are deliberately NOT repaired here — they are the backlog the ratchet
exists to hand to whoever next touches each row.
