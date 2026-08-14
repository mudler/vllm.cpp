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

`scripts/agent-preflight.sh --staged`, plus `python3 -m unittest` on the new
suite. No GPU, no build — this is a Python checker and its tests.

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

`SPIKE` — spec committed, implementation not started. Next: a fresh implementer
takes the parser, the classifier, the baseline and the six cases.
