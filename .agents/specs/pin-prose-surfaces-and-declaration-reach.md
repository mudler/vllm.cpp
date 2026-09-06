# PIN-PROSE-SURFACES-AND-DECLARATION-REACH — the pin's prose surfaces get a delimiter, and the declaration check gets a call site nobody can delete

Issues: [#2883](https://github.com/mudler/vllm.cpp/issues/2883) (the pin is
restated in three prose surfaces and nothing checks them) and
[#2898](https://github.com/mudler/vllm.cpp/issues/2898) (nothing proves `main`
runs `check_declarations`).
Row: `-` (no row owns either; both are listed under `## Owed` in
[`oracle-pin-parity-reconcile.md`](oracle-pin-parity-reconcile.md), which is the
spec that owed them, and this change is the one that pays them off)
Prior art: [#2829](https://github.com/mudler/vllm.cpp/issues/2829) and PR
[#2880](https://github.com/mudler/vllm.cpp/pull/2880), which made
`.agents/oracles/vllm.md` reconcile against the ` ```parity-pin ` block in
`.agents/upstream-sync.md` and introduced `ParityThroughMainTests`, the
through-`main` shape both halves of this change reuse.

## Scope

**In scope.** Two rules in `scripts/check-oracle-pins.py`, and the marker edits
one of them needs.

1. **#2883.** Five *named* paths — `.agents/NOW.md`, `docs/FEATURES.md`,
   `docs/benchmarks/how-we-measure.md`,
   `docs/benchmarks/speculative-decoding.md` and
   `docs/benchmarks/vllm-online-serving.md` — must each carry at least one
   ` <!--pin:commit--> ` span, and every ` commit ` / ` label ` span in them must
   agree with the authority in `.agents/upstream-sync.md`. All five surfaces
   gain the markers. **The rule reads those paths and nothing else**; see
   §"A glob was built first and rejected" for the measurement that decided it.
   A sixth rule gates the marker's POSITION; see §"A marker is only invisible
   where it is inline".
2. **#2898.** `ParityThroughMainTests` gains a case that drives `main` against a
   synthetic registry holding a declaration for an unregistered oracle, and
   asserts `main([]) == 1`. A companion case declares a *registered* oracle and
   asserts `0`, so the red is attributable to the rule and not to the file.

**Out of scope, deliberately.**

- **`.agents/upstream-sync.md` itself.** It is the authority; this change reads
  it and never writes it. Another wave owns its content.
- **`.agents/oracles/vllm.md`.** Already reconciled by #2880.
- **A regex over prose.** Ruled out by #2880's 286-file measurement and not
  revisited. The design below replaces prose parsing entirely. That figure is
  #2880's and is restated here as its measurement, not as one taken again: at
  `aa7056b46` a `git grep -l 555967922` over this tree returns **846** files.
  Either count rules out the regex, which is why the conclusion survived a
  figure that had gone stale by a factor of three.
- **Removing the restatements.** #2883 admits that as a third outcome. It is a
  public-document editorial decision, not a checker decision, and the pin is
  genuinely useful in all three places.

## The two questions #2883 leaves open, and the answers

### 1. The abbreviation length: a prefix rule with a declared floor, not a fixed length

#2883 records that `e126687a9a` is ten characters and nothing declares that, and
that "accepting any prefix accepts `e1`".

**Measured, at `4e748e4a7`:** this tree has no ten-character convention. It
writes the *prior* pin as eight characters in `docs/benchmarks/how-we-measure.md`
(`55596792`), nine in `docs/FEATURES.md` (`555967922`), and forty in
`.agents/oracles/vllm.md`. Pinning the length at ten would therefore *invent* a
convention and go red on the shape the repository already uses. AGENTS.md: a
gate that fires on correct work is the defect, not the discipline.

So correctness is defined as **"is a prefix of `vllm_commit`"**, with a floor of
`MIN_ABBREV = 8` hex characters. The floor is not a style rule; it exists for
one reason, and the checker says so: below eight characters an abbreviation can
collide with a *different* commit by accident, and the rule would then pass a
stale surface. Eight hex characters is 4.3e9 values and is the shortest length
this tree actually writes, so the floor rejects the degenerate `e1` without
rejecting anything the repository does today. A value that is not hexadecimal,
is shorter than the floor, or is not a prefix is three distinct errors with three
distinct messages, because "does not match" is not a diagnosis.

### 2. Structure first, with an invisible delimiter

#2883 asks whether to structure the surfaces or parse the prose. **Structure
first**, and the extraction is then not a parse at all.

Prose parsing was prototyped far enough to reject it. A token-based rule over an
unmarked sentence has to decide what a version token is and what a revision
token is, and it gets both wrong on real content: `` `b10451` `` — the llama.cpp
pin, one line below the vLLM pin in `docs/FEATURES.md` — is six characters drawn
entirely from `[0-9a-f]`, so any "backticked hex word" heuristic reads a second
project's pin as a malformed vLLM revision. Containment is worse still: a rule
that asks whether `0.28.1rc1.dev132` appears in the sentence passes a surface
reading `0.28.1rc1.dev1320`.

The markers are inline HTML comments:

```text
Pin: vLLM <!--pin:commit-->`e126687a9a`<!--/pin--> (<!--pin:label-->0.28.1rc1.dev132<!--/pin-->) since
```

This shape was chosen over the alternatives for five reasons:

- **The extraction is exact.** The span content *is* the value. There is no
  token regex, no positional assumption, and no sentence shape. The `b10451`
  and `0.28.1rc1.dev1320` failures above are both unreachable.
- **It has no false-positive surface.** Nothing outside a marked span is read,
  so the 846 files that name a prior pin on purpose are untouched, and so is
  every unmarked sentence in the five named files. (#2883 says each surface has
  "one occurrence". That is false: `docs/FEATURES.md` restates the CURRENT pin
  twice, at line 14 and again in the *Results.* paragraph. **Both are marked.**
  An earlier draft of this spec offered the second one as an example of prose
  the rule deliberately leaves alone, and the implementation marked it — the
  document and the tree disagreed, and the tree was right. An unmarked
  restatement of the current pin inside a named surface is precisely the
  unreconciled pin #2883 filed.)
- **It fails closed.** A missing or unclosed marker in a named path is an error
  naming that path, so the rule cannot be silenced by deleting it. An
  unbalanced marker is detected by counting openers against complete spans, not
  by trusting the closing tag to exist.
- **It is invisible when rendered, WHERE IT IS INLINE — and that is a gated
  condition, not a property of HTML comments.** See §"A marker is only invisible
  where it is inline" for the measurement and the rule. Given that rule,
  `.agents/NOW.md` is authored at operator cadence; a delimiter that changes no
  rendered word is not an authored edit, and it costs no lines against that
  file's 100-line budget.
- **The repository already reads this shape.** `AGENTS.md` carries
  `<!-- oracle-registry:begin -->` and `.agents/NOW.md` carries
  `<!-- now-updated: -->`, both parsed by existing checkers.

## Design

`scripts/check-oracle-pins.py` gains one rule and `main` gains one call.

- `PIN_SPAN` matches `<!--pin:(commit|label)-->…<!--/pin-->`; `PIN_OPEN` matches
  a `<!--pin:…-->` opener whose kind is any `[A-Za-z0-9_-]*` identifier. A count
  mismatch between the two is the unclosed/misspelled-marker error. **The opener
  class has to be wider than the kind set or the rule cannot see its own
  defect.** It was `[a-z]*`, and `pin:LABEL`, `pin:Commit`, `pin:commit2` and
  `pin:pin-commit` then matched neither regex: beside one valid span the counts
  agreed, and `pin:LABEL` and `pin:Commit` were reported by nothing, through
  `main`, at rc 0.
- `PIN_AT_LINE_START` reports a marker that is first on its line. See §"A marker
  is only invisible where it is inline".
- `check_pin_surfaces(surfaces, required, parity, errors)` takes the authority
  as a **parameter** and opens no file, exactly as
  `check_parity_reconciliation` does. **The expectation is read from
  `.agents/upstream-sync.md` and never from the file under test**; a checker
  that reads its expectation out of what it checks is a tautology, and this
  repository has shipped that shape.
- `commit` spans: hexadecimal, at least `MIN_ABBREV`, and a prefix of
  `vllm_commit`. `label` spans: equal to `public_version(vllm_runtime_version)`,
  by equality after that decomposition and not by prefix — the same reasoning
  #2880 recorded for `pin_label`.
- Each path in `PIN_SURFACES` must exist and carry at least one
  `commit` span. `label` is not required per file, because the release label is
  derived from the commit and requiring it would red on an editor who tightens a
  sentence to name only the revision.
- `main` reads each path in `PIN_SURFACES` and calls the rule, swallowing the
  `OSError` so the rule reports an absent surface with a reason instead of
  skipping it. `PIN_SURFACES` is a module constant, patchable like `ROOT`,
  `ORACLES`, `AGENTS_MD`, `UPSTREAM_SYNC` and `DECLARATION_ROOTS`, so the
  through-`main` tests point the whole rule at a synthetic tree.

## A glob was built first and rejected, on evidence

The first implementation validated every marked span under `.agents/` and
`docs/`, so that marking a fourth surface would need no checker edit. **It went
red on this spec.** The `### 2` section above quotes the markers in a fenced
block to explain them, and the checker read that as three openers, two complete
spans, and a `commit` value half a document long:

```text
oracle-pins: .agents/specs/pin-prose-surfaces-and-declaration-reach.md: 3
  `<!--pin:...-->` opener(s) but 2 complete span(s)
```

**That number is a self-measurement and it moves.** It was written here first as
four; re-measured with the checker's own `PIN_SPAN` and `PIN_OPEN` it is three
at `aa7056b46`, and it is **four again** now that §"A marker is only invisible
where it is inline" quotes one more marker in an error message. The drift is the
argument rather than a footnote to it: every time somebody explains the marker,
a glob rule gains a false positive. `scripts/check-oracle-pins.py` therefore
stores no count at all, because a measurement of one file kept inside another is
a line every future edit has to remember.

Documentation of a marker is not a surface carrying one, and no cheap rule
separates them — not a code-fence exclusion, which is one more prose parser, and
not a directory exemption, which is arbitrary. Reading only the named paths is
also exactly what #2883 proposed, so the glob bought extensibility this rule was
never asked for and paid for it with a false positive on its own spec.

For #2898 no product code changes: `main` already calls `check_declarations`.
The defect is that nothing measured the call.

## A marker is only invisible where it is inline

This section exists because the first version of this change asserted the
opposite in three places and measured it in none — this spec ("It is invisible
when rendered…"), the checker docstring ("HTML comments are stripped when
Markdown renders…") and the pull request body ("HTML comments render as
nothing…") — and shipped a broken paragraph in a **public** document.

`docs/FEATURES.md:46` carried its opener at **column 1**, in the middle of the
*Results.* paragraph. Under CommonMark an HTML comment indented fewer than four
spaces begins a **type-2 HTML block**, and type 2 is one of the seven start
conditions that may **interrupt a paragraph**. The rest of that line is then
emitted as raw HTML.

**Measured on GitHub's own renderer**, `gh api /markdown` over the whole file,
before and after:

| | before | after |
|---|---|---|
| `<p>` elements | 21 | 20 |
| lines carrying a literal backtick | 1 | 0 |
| `<code>e126687a9a</code>` | 1 | 2 |

One paragraph rendered as three blocks and a reader saw `` `e126687a9a` `` with
its backticks. The repair is a reflow, and the rendered *text* of the paragraph
is unchanged by it.

**One probe per indentation, same renderer**, because the boundary itself was
then reasoned about and got written down wrong once already:

| indentation | context | result |
|---|---|---|
| 0–3 spaces | mid-paragraph or after a blank line | HTML block; paragraph splits; value renders as literal backticks |
| ≥ 4 spaces | after a blank line | indented code block; **the marker itself becomes visible text** |
| ≥ 4 spaces, or a tab | inside a paragraph | lazy continuation; inline and invisible |

So the rule is `PIN_AT_LINE_START`: **a pin marker is never first on its line.**
It refuses exactly one shape that would have rendered correctly — a marker
behind four spaces of a paragraph *continuation* line — which nobody writes and
which is one blank line away from the worst of the three outcomes.

**No renderer is imported.** `scripts/check-oracle-pins.py` is deliberately
dependency-free and network-free, and AGENTS.md is explicit that a gate failing
because GitHub is unreachable fails on the wrong thing. A Markdown library would
also be a new test dependency for one rule. The rule is therefore the **position
that produces the break**, which is exactly checkable from the bytes, and the
renderer is used to *establish* the position rather than to evaluate it.

**Red-before, green-after, on the tree.** Against `HEAD:docs/FEATURES.md` the
rule reports one error, `docs/FEATURES.md:46: pin marker <!--pin:commit-->
starts its line…`; against the repaired file it reports none. Disabling only the
`PIN_AT_LINE_START` loop reds five cases — three direct, one through-`main`, and
the self-test corpus — and nothing else.

## Risks

- **The rule fires on correct work.** Mitigated by construction (nothing outside
  a span is read) and tested explicitly: a benign rewrap of the paragraph around
  a marked span must leave both rules green.
- **A pin advance now has to edit three more files.** That is the point, and it
  is the #2829 shape the advance previously got wrong. The error message names
  the path and the expected value, so the edit is mechanical.
- **The markers get lost in a future rewrite of a public document.** That is the
  failure the required-path rule exists for: losing them is an error, not a
  silent pass.
- **A shared-file lock.** These are three *public projections* that a pin
  advance already had to edit; the rule adds no new surface that every pull
  request must write. AGENTS.md §Records is satisfied.

## Tests

`tests/scripts/test_check_oracle_pins.py`:

- Direct-call cases for `check_pin_surfaces`: agreeing surfaces pass; a stale
  commit, a stale label, a truncated label, a non-hex value, a below-floor
  abbreviation (`e1`), an unclosed marker, an unknown marker kind, a missing
  required path, and a required path with only a `label` span each red, and each
  is asserted on the *message*, not merely on the count.
- Through-`main` cases in `ParityThroughMainTests`: `main` reds on a stale
  marked surface and passes on an agreeing one, so deleting the `main` call site
  reds a case whose fixtures are otherwise clean.
- Through-`main` cases for #2898: a declaration naming an unregistered oracle
  reds; the same file naming a *registered* oracle passes.
- A benign-edit case: prose added and rewrapped around an untouched span keeps
  both rules green.
- `--self-test` gains a pin-surface corpus swept in both directions.

## Gates

- `python3 scripts/check-oracle-pins.py` and `--self-test`.
- `python3 -m unittest tests.scripts.test_check_oracle_pins`.
- `python3 scripts/check-now-current.py` (the NOW.md marker edit).
- `scripts/agent-preflight.sh`, skips named.

## Evidence

Red-before and green-after for each rule, captured in the pull request body:
the exact divergence, the checker's stderr, and the return code read directly.
Every mutation is line-anchored and the mutated line is printed, and every
restore is verified with `sha256sum -c`. `__pycache__` is purged before each
re-run, because a restored `.py` can still execute mutant bytecode.

## Stop conditions

- Stop if either rule cannot be made to go red by a mutation. A checker that
  cannot fail is a mute switch and must not be reported as a gate.
- Stop if the #2898 case reds for a reason other than the deleted call site.
  An incidental red is not a pin.
- Stop if a benign prose edit reds either rule; that would make the rule the
  defect.
- Do not edit `.agents/upstream-sync.md`, `.agents/sync/`, or the specs another
  wave owns.

## Owed

- The two `## Owed` bullets in
  [`oracle-pin-parity-reconcile.md`](oracle-pin-parity-reconcile.md) that name
  #2883 and #2898 go stale when those issues close. That file is owned by
  another wave and is not edited here.
- The same file restates #2880's **286-file** measurement as the reason a regex
  over prose is ruled out. The figure is stale — `git grep -l 555967922` returns
  **846** at `aa7056b46` — and it is attributed and re-measured in the two
  surfaces this change owns (`scripts/check-oracle-pins.py` and this spec) but
  not in that one, for the same reason. The conclusion it supports is unaffected
  by the correction, so this is an accuracy debt and not a live defect.

## Now

No row. This spec exists because a semantic checker change needs one, and it is
committed before the implementation.
