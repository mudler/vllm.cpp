# FIX-GATE-COMMANDS-PROSE-PIN: pin the device leg by structure, not by its sentence

**Issue:** [#1229](https://github.com/mudler/vllm.cpp/issues/1229).
**Affected row:** `ENG-CUDAGRAPH-DEDUP` in [engine-matrix.md](../engine-matrix.md),
whose credit test carries the defect. The row itself is correct and is not edited here.
**Kind:** checker repair. No product source is touched and no forward pass is reached.

## Now

`origin/main` at `fe24a3029` is RED.
`tests/scripts/test_check_gate_commands.py:670` asserts the literal string
`Device byte-identity A/B (owed` inside `.agents/specs/eng-cudagraph-dedup.md`.
`fe24a3029` rewrote that sentence from recording the device A/B as owed to
recording that it ran and what it found. The record edit is correct. The
assertion described a state that the record legitimately left.

Every branch that merges `origin/main` inherits the failure.

## Scope

In scope:

- Replace the prose pin with an assertion keyed on the structure of the spec's
  `Gates` section.
- Add the two pure helpers that assertion needs to
  `scripts/check-gate-commands.py`.
- Add direct coverage for those helpers, including a fixture that proves the new
  assertion holds on **both** the pre-edit and the post-edit wording.
- Append the [#1229](https://github.com/mudler/vllm.cpp/issues/1229) row to
  [`issue-index.md`](../issue-index.md).

Out of scope:

- `.agents/specs/eng-cudagraph-dedup.md`. The record is right.
- `RUNNABLE_BASELINE`. No row enters or leaves the gated population here.
- The `--check` verdict of `scripts/check-gate-commands.py`. The new helpers are
  read by the test suite and are not wired into `ratchet_errors`.
- The stderr the negative-path fixture prints. See `## Owed`.

## What the assertion was protecting

The comment above the failing line states it:

> The row's DEVICE leg is OWED, not skipped, and the spec has to say so.
> Without this the credit could rest on the CPU tier while the record
> stayed silent about the arm nobody ran, which reads as coverage.

`ENG-CUDAGRAPH-DEDUP` is credited in `RUNNABLE_BASELINE` for
`ctest -R test_graph_dedup` and `./scripts/agent-preflight.sh`. Both are CPU-tier
commands. Gate item 6 of the spec, the device byte-identity A/B, contributes no
runnable command at all. The credit therefore covers the CPU tier and says
nothing about the device tier. The assertion existed so that the record could not
go silent about the leg the credit does not cover.

Two readings were offered in [#1229](https://github.com/mudler/vllm.cpp/issues/1229),
and the code supports one half of each.

The first reading says the assertion keeps a device-gate obligation visible while
one is outstanding, so the A/B having run discharges its premise. Its **diagnosis**
is right. The premise did move.

Its **remedy** is wrong. Asserting the new wording instead of the old one only
moves the pin from `(owed` to `RAN 2026-08-18, PASS`. The row already owes a rerun
under [#1226](https://github.com/mudler/vllm.cpp/issues/1226) against a coarser
signature key, and the next honest record edit rewrites that sentence too and reds
the gate again. The same defect, a second time.

The second reading says the assertion should key on something durable. Its
**remedy** is right. Its **diagnosis** is incomplete, because the four assertions
that precede it already cover "credited for real commands" in full: baseline
membership, the `runnable` verdict from `gates.audit()`, and the two commands
pulled out of the section by `gates.runnable_commands`. The last assertion adds
what those cannot see, which is a constraint on the leg the credit excludes.

**The chosen reading is the synthesis.** The protected property is about the
device leg, and it does not depend on the leg's state. A credited row must not go
silent about a leg its credit does not cover. `owed` was that leg's value on the
day the assertion was written, not the property. Silence is the defect in both
directions, whether the leg was never run or the line was deleted after it ran.

## Design

The `Gates` section is a numbered list. Each item opens with a bold lead that
titles it. Every disposition this row has ever recorded lives in that lead:

| Revision | Item 6 lead |
|---|---|
| `2a976eb9f` | `Device byte-identity A/B (owed, see below).` |
| `fe24a3029` | `Device byte-identity A/B — RAN 2026-08-18, PASS; see [...](#outcome).` |

A status word in the lead is a declaration about the gate. The same word in the
body is ordinary prose describing what the gate does. That distinction is what
makes the lead the anchor, and it is measured rather than assumed.

**The first measurement refuted the first hypothesis, and the record keeps both.**
The drafted reason was that item 5 of this row's spec picks up the word `fail`
from the body phrase "proving the instrument can fail". It does not, because the
shipped vocabulary carries `failed` and not `fail`. All 7 gate items of
`eng-cudagraph-dedup.md` agree under both scopes, so that spec cannot pin
lead-scoping and a mutation widening the scope stayed green against it.

Surveyed instead over every `Gates` section in `.agents/specs/` on 2026-08-18:
**32 of 323 gate items gain a disposition when the search widens from the lead to
the whole item.** Among them are `**No regression:**` in `cpu-elementwise-gemm.md`
and `**Correctness gate:**` in `dropin-kernel-abi.md`, which both pick up `pass`
out of body prose. Lead-scoping is load-bearing across the corpus and is pinned by
a written-out fixture of that shape, not by another row's live sentence. Pinning a
sentence in a file this row does not own is the defect this row exists to remove.

Three functions land in `scripts/check-gate-commands.py`:

- `gate_items(section)` splits a `Gates` section into its numbered items.
- `item_lead(item)` returns the bold lead of one item, or `None`.
- `gate_disposition(item)` returns the status word declared in that lead, or
  `None`. The vocabulary is closed and small: `owed`, `waived`, `blocked`,
  `deferred`, `superseded`, `not gated`, `ran`, `pass`, `passed`, `failed`.

The test then asserts three things about `ENG-CUDAGRAPH-DEDUP`:

1. Exactly one gate item names the device byte-identity A/B. The **subject** of
   the gate is the anchor. The subject is what a record edit keeps and the status
   is what a record edit moves, so the subject is the durable half.
2. That item yields no runnable command. This is what proves, from the checker's
   own extraction rather than from a reader's belief, that the row's credit rests
   entirely on the CPU tier.
3. That item declares a disposition. This is the anti-silence rule, and it is
   satisfied by `owed` and by `RAN` alike.

**The rule is not swept over every gate item, and the reason is measured.** Items
1 and 5, `Red first.` and `CUDA compile.`, yield no runnable command and declare
no disposition. Both are real gates that this spec describes in prose instead of
quoting. A sweep flags both. Widening the vocabulary until they pass would make it
match ordinary English and detect nothing, which is the failure this repair
exists to remove. The targeted form is therefore the honest one, and the general
rule is recorded as owed below rather than shipped loose.

## Risks

- **The vocabulary is still words.** It is bounded by being lead-scoped and
  closed, and the mutation below proves it goes red when the lead loses its
  status. It is not proof against a record edit that renames the gate's subject.
  That is accepted: a renamed subject is a different gate and should be re-read.
- **`item_lead` returns `None` when a lead does not close on its own line.** That
  reads as no disposition, so the gate goes red rather than green. The safe
  direction.
- **The helpers are unused by `--check`.** They add no new verdict over the other
  gated specs, so no row can go red on arrival. This mirrors the file's own stated
  policy that a gate red on arrival must be relaxed to pass, and a relaxed gate is
  worse than no gate.

## Tests

- `tests/scripts/test_check_gate_commands.py::test_cudagraph_dedup_is_credited_for_real_commands`,
  rewritten onto the structural anchor.
- `tests/scripts/test_check_gate_commands.py::test_gate_disposition_reads_the_lead_not_the_body`,
  new. It pins lead-scoping against a written-out fixture of the shape the survey
  found, and it pins both historical wordings of item 6 as fixtures, which is what
  proves the assertion is state-independent rather than re-pinned to today's
  sentence.

## Gates

1. **Red first.** The failure is reproduced on a pristine detached worktree at
   `fe24a3029` before anything is edited, and the output is captured.
2. **Focused green.** `python3 tests/scripts/test_check_gate_commands.py`.
3. **Mutation.** Each new assertion is broken in a scratch copy and the focused
   suite is proven to fail. `git diff --stat` and the exit status are printed for
   every mutation, because a mutation that never applied and a mutation that fails
   to build both read as a passing test. The tree is restored and the restoration
   is verified by `sha256sum`.
4. **Full gate.** `scripts/agent-preflight.sh --fail-on-skip`.
5. **Not gated, deliberately:** every CUDA, GPU, SACRED, oracle and throughput
   gate. This change edits one checker and one test suite. It reaches no forward
   pass, no kernel, no dtype and no token, so no device gate is claimed.

## Stop conditions

- Stop and report if the new assertion cannot be made to fail under mutation. An
  assertion that survives its own guarantee being removed is testing nothing, and
  shipping it would be the deletion this row exists to refuse.

## Owed

| Owed | Issue | Why not here |
|---|---|---|
| A general rule that every gate item yielding no runnable command must declare a disposition, applied across all gated specs | [#1229](https://github.com/mudler/vllm.cpp/issues/1229) | measured red on arrival for items 1 and 5 of this spec alone, and the checker's own header records that a gate red on arrival has to be relaxed to pass. It needs the survey the 2026-08-06 audit did for commands, then its own spec |
| Containing the stderr that `test_check_mode_is_never_silently_swallowed_by_json` prints on every green run | [#1229](https://github.com/mudler/vllm.cpp/issues/1229) | the line `ERROR: these baseline rows left the gated population: ROW-THAT-IS-NOT-THERE` is that fixture's expected negative-path output and not a defect, established by running the fixture alone. It is still an error message on a green run, which is the shape that lets a real one hide. Changing a passing test needs its own red-before evidence, so it is filed rather than folded in |

## Outcome

**Refuted hypothesis, recorded rather than dropped.** The design first justified
lead-scoping with item 5 of `eng-cudagraph-dedup.md`, on the reading that its body
phrase "proving the instrument can fail" would credit it under a wider search. The
mutation that widens `gate_disposition` from the lead to the whole item stayed
GREEN, which is how the claim was caught. The shipped vocabulary carries `failed`
and not `fail`, and all 7 gate items of that spec agree under both scopes. The
survey over all 323 gate items replaced the anecdote and found the real figure,
32, along with two examples that do behave the way the anecdote predicted. The
mutation goes red now because the test pins a fixture of that measured shape.

The lesson repeats the row's own: the first rationale was a plausible sentence,
and a sentence is not a measurement. The mutation is what told them apart.

The assertion was not wrong when it was written. It named the right leg and the
right hazard. It encoded the hazard as the sentence that happened to express it,
and a sentence is the part of a record that a correct edit is expected to change.

The general lesson holds and is the reason for the structural form: a check that
matches prose measures the wording, and the wording is the surface the protocol
asks people to keep current. Key on what a correct edit preserves. Here that is
the numbered item, its subject, and whether the checker's own extractor finds a
command in it.
