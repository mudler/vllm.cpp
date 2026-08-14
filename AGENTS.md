# AGENTS.md: the rules

This file contains the complete policy for `vllm.cpp`. It is the only file that
every agent loads automatically, so every rule lives here. Files under
`.agents/` are task guides. They explain how to do a specific job. They cannot
add or weaken a rule in this file.

The project mirrors vLLM in C++ without PyTorch or a ggml dependency. vLLM
defines the reference behavior and the performance target.

## Start here

1. Run `scripts/agent-start.py`. Pass `--intent operator|helper|read-only` and
   `--row <ID>` when you know them. Otherwise, relay its welcome and ask what
   work is intended. Follow the printed action, then run the command again.
2. Declare a role. Use `scripts/agent-role.py claim operator` for a multi-step
   integration campaign. Use `claim helper --row <ID>` for one scoped task.
   Use `claim read-only` for inspection. The operator claim records the current
   worktree as a coordinator. Another coordinator does not block the claim.
   Add `--headless` only when the developer explicitly says the run is
   unattended. Never infer this setting.
3. Run `scripts/now.py` to get the live position. Read `.agents/NOW.md` for the
   operator's current gate and next actions. The command output is derived.
   The file is authored and fits on one screen.
4. Read only the claimed row, its spec, its evidence, and the task guide for the
   current job.
5. Run `scripts/agent-preflight.sh` before you edit a file.

Never infer a role, host, permission, or developer preference. Resolve `.env`
and `.agents/developer-preferences.md` from the shared checkout. Ask only for
the one value that the current gate needs. If a value is unavailable, leave its
gate `PENDING`. Never convert a missing value into an assumption. Preferences
control operations only. They cannot reduce a correctness, evidence,
attribution, or testing obligation.

## History is git

The project has no state log. Git is the history, and the history must agree
with the tree.

| Question | Command |
|---|---|
| Did this row already land? | `git log --oneline --grep '<ROW-ID>'` |
| When did this symbol change? | `git log -S'<symbol>' --oneline -- <path>` |
| What happened to this file? | `git log --follow --oneline -- <path>` |
| What is on main that I lack? | `git log --oneline HEAD..origin/main` |
| Why is this line like this? | `git log -L '<start>,<end>:<path>'` |
| What did that commit change? | `git show --stat <sha>` |

The roadmap row states the current position of a row. Git and the row's spec
state how it got there. Before you conclude anything about past work, read the
spec and run `git log -S`. Do not derive the history again.

## Every change starts from an issue

**Do not start work without an open GitHub issue.** Before you claim a row or
write code, make sure that an issue tracks the work. Open one if none exists.
Link the issue in three places that must agree: the index in
[`.agents/issue-index.md`](.agents/issue-index.md), the row's spec, and the pull
request body.

The index is append-only and carries `merge=union`, so two branches that each
append a row merge without a conflict. Append a row at the end. Never edit a row
and never delete one, because GitHub holds the open and closed state and an
edited row is duplicated rather than merged.

A bug that you find during other work still needs an issue. Filing the issue
does not defer the fix. File it, fix it in the same flow, reference it in the
commit, and close it. The person who found the bug has the context to fix it.
Traceability is the goal, not another round trip.

**An issue you do not fix in the same flow has to say who owns it.** Every index
row names an owning row ID, or names a spec that lists the issue under `##
Owed`. `scripts/check-agent-record.py` counts the rows that do neither and
refuses a count above the recorded mark. Filing without fixing is therefore a
gate failure rather than a habit.

This in-flow rule applies to small and clear fixes. Use the normal row, spec,
and fresh-review path for a surprising fix. Use that path when a fix needs its
own spec or changes checker semantics. The in-flow rule is not a bypass.

## Spec before code

A row cannot become `READY` or `ACTIVE` without a committed
`.agents/specs/<slug>.md`. Commit the spec before implementation. Never write
the spec after the implementation. The spec contains scope, upstream anchors,
design, risks, tests, gates, evidence, and stop conditions.

At row claim, before you write the spec, ask the developer whether the spec and
implementation use one pull request or separate pull requests. Recommend one
pull request. Record the answer under `## Git integration` in
`.agents/developer-preferences.md`, and do not ask again for that row. When no
answer is recorded and no split case applies, use one pull request. This default
is repository policy, not an inferred preference.

The single pull request is a recommendation, not an enforcement rule. Use
separate pull requests whenever the developer selects that shape. A split is
often useful for these cases:

- A helper dispatch needs a base-reachable committed spec before the helper can
  start.
- A large campaign benefits from agreement on the scope before implementation
  waves start.
- A change deliberately adds roadmap items, issues, or specs without product
  code.

Commit order still proves that the spec came first when one pull request carries
the spec and implementation. Do not add a checker that selects the pull request
shape.

Before you claim a row, verify the gap against the current code, tests, issues,
pull requests, `NOW.md`, and the owning row. If the work already landed, already
has an owner, or no longer matches its record, reconcile the record first and do
not implement.

When a row reaches `DONE`, add an `## Outcome` section to its spec. Record what
you measured, what you rejected and why, and why each default has its value.
The code and Git history do not record these decisions.

## How we write

Use [the commit and pull request guide](.agents/style/commits.md) for commit
subjects, commit bodies, pull request titles, pull request descriptions, branch
names, changelog entries, and release-note lines. Use [the technical-English
guide](.agents/style/prose.md) for repository documents and all session prose.
Session prose includes progress updates, decisions, questions, and final
reports.

These guides control language and document structure. They cannot add unrelated
project policy or weaken this file. This file takes precedence when a guide
conflicts with a repository convention or rule.

Two of those rules are gated, and `scripts/check-commit-style.py` enforces them.
A commit subject does not end in a period. A commit has an authored body,
because the reader of a commit already has the diff and lacks the reason. One
sentence satisfies the body rule.

Subject length and body wrapping stay guidance in the guide, and no gate
enforces them. Measured over the last 200 non-merge commits, a 72-character
subject limit fails 164 commits and a 72-column body wrap fails 3658 lines. A
gate that fires on ordinary work is the defect, not the discipline.

The guides bind new prose. Rewriting an existing file to satisfy a style rule is
out of scope unless a row asks for that rewrite.

## How work gets done

Work is delegated. Another person or agent reviews every implementation, and the
operator verifies the result. This sequence is the method, not a suggestion:

1. A **fresh implementer** works from the committed spec. The implementer ports
   or writes the smallest test that fails for the intended reason. They capture
   the red result, make the minimum complete change, and get focused green.
   Then they run the full gate.
2. A **fresh reviewer** reviews the immutable head. The reviewer cannot be the
   agent that wrote the code. They inspect the change statically and mutate each
   claimed guarantee in a scratch copy. This mutation proves that the tests
   detect the defect. Mutate the guarantee. Do not only read it. The reviewer
   restores the tree byte-for-byte after each mutation.
3. A **fresh implementer** repairs each finding. The coordinating session never
   repairs a finding. Repeat the focused gate, full gate, and fresh scoped
   review until the result is `PASS`. Attempt budgets control scheduling. They
   never stop a correctable finding. Only explicit developer direction or a
   precise external blocker stops the loop.
4. The **operator reruns the row's gate itself**. An implementer or reviewer
   report is an input, never a gate result.

Every delegated task states the goal, context, exact scope, exclusions,
constraints, completion condition, required evidence, authority, output
contract, and stop conditions. Return `NEEDS_CONTEXT` when binding context is
missing. Do not guess. Return `NEEDS_DECISION` for a material disagreement
rather than changing the scope silently. Use the versioned contracts in
[`.agents/prompts/`](.agents/prompts/).

The operator is a **coordinator**. It holds the plan and the graphics processing
unit (GPU). It merges reviewed pull requests and dispatches agents into separate
worktrees. It does not write an implementation that needs independent review.

**More than one operator can run at the same time.**
`scripts/agent-role.py claim operator` records the coordinator and never refuses
because another operator exists. The `show` command lists the other operators.

**Never force-push `main`.** Nobody can use `--force` or `--force-with-lease` on
`main`. A plain `git push` rejects a non-fast-forward update. Git provides the
coordination lock. After a rejected push, fetch, merge again, rerun the gate,
and push again. Never force the update.

## vLLM is the reference

**Mirror vLLM.** When vLLM defines behavior, mirror every applicable mode,
default, error, and edge case. Escalate only a genuine product decision. Never ask how a
mirrored feature must behave.

**Pin vLLM.** Compare against the pinned oracle in
[`.agents/upstream-sync.md`](.agents/upstream-sync.md). Advance the pin only
after you reconcile every affected row and gate. An oracle is only gateable once
it demonstrably builds and runs the model. Constructing a config proves
nothing.

**Run the oracle and read its source.** Check every change in two ways. Run the
pinned vLLM on the identical workload. Read the matching upstream path. Ground
the conclusion in the complete executing chain. This chain can include
FlashInfer, CUTLASS, cuBLASLt, DeepGEMM, torch or Inductor, generated code, and
local dispatch. Cite the `file:line` that you ported. Dump the generated kernel
before you call a lever unreachable. Record scratch implementations as such in
the porting inventory.

**Inherit vLLM defaults.** vLLM resolves one model dtype, and every layer
inherits it. An `f32` value is a rare, annotated exception. Mirror this
polarity. Add a one-line reason beside a model-path buffer or GEMM output that
names `f32`. **A token gate cannot detect a dtype that is too wide.** The tokens
still match, and the goldens still pass, although the path moves twice the
bytes. Compare the memory format with the oracle as
[`.agents/porting.md`](.agents/porting.md) requires.

**Port the upstream tests in the same change.** Preserve parameters, modes,
fixtures, tolerances, failure cases, and the upstream revision anchor. Document
only an unavoidable adaptation of the harness.

**Trace both sides with the same tool.** Use an identical workload before any
throughput comparison. Source inspection identifies candidates. Matching traces
identify the executed path. A GEMM or GEMV invocation-parity claim proves the
output dtype, compute type, scale type, entry point, algorithm policy, and
resolved template dtypes in the same tool.

## When vLLM has no implementation

vLLM, including `vllm-project/vllm-omni`, is the primary reference. It is the
only reference wherever it implements the behavior. Where it implements nothing, the work is
still not ungated. Run it against a **secondary oracle**. A secondary oracle is valid
only when it appears in this table and has a recorded pin:

<!-- oracle-registry:begin -->

| Oracle | Registry id | Reach for it when |
|---|---|---|
| vLLM | `vllm` | always, as the primary wherever it implements the behavior |
| vLLM-Omni | `vllm-omni` | diffusion, TTS, and omni-only architectures that vLLM does not register |
| HuggingFace `transformers` | `transformers` | a model, processor, or tokenizer reference implementation that vLLM mirrors |
| `diffusers` | `diffusers` | schedulers, VAEs, and diffusion pipelines |
| SGLang | `sglang` | a model or serving path that SGLang implements and vLLM does not |
| SGLang-Omni | `sglang-omni` | omni, speech, TTS, and music models served by SGLang's pipeline runtime, in a third repository that is not SGLang |
| llama.cpp | `llama-cpp` | CPU and GGUF k-quant floors |
| Tenstorrent tt-forge | `tt-forge` | Tenstorrent hardware, for which vLLM has no backend |

<!-- oracle-registry:end -->

**Pin every oracle.** Each oracle has its own file under
[`.agents/oracles/<id>.md`](.agents/oracles/). The file records the revision,
measurement date, gateability, and evidence. An unpinned upstream is a moving
target, not an oracle. Its measurements are not reproducible. The vLLM parity
pin remains in [`.agents/upstream-sync.md`](.agents/upstream-sync.md). The vLLM
oracle file points there instead of copying the value. Use one file per oracle
and read the files with a glob. Never require every change to write one shared
table.

**A secondary oracle answers one question:** what correct output looks like on
a path that vLLM cannot run. It never outranks vLLM, and it never becomes the mirror
source. Mirror vLLM behavior, defaults, structure, and naming wherever vLLM
defines them. "SGLang does it differently" is never on its own a reason to diverge.
When vLLM implements the path, reconcile the row onto vLLM and record the
change in the spec.

**Measure gateability.** The primary rule applies to every oracle. The oracle
must demonstrably build and run the model. Until then, its file records
`gateable = no` and names the issue that owes the measurement. This record makes
the ungateable lane visible debt.

## Gates

Correctness always comes first. Establish the declared token-exact gate before
you accept a performance result. Use an explicitly ratified distributional gate
only when the oracle's greedy decode is non-deterministic. Never trade
correctness for throughput.

Use the pinned oracle on each side. Use identical model artifacts, prompts,
token counts, batching, concurrency, and sampling. Use vLLM's production
configuration as the denominator. Never use `--enforce-eager` as the
denominator.

Record values and ratios for every required throughput, latency, and memory
axis. An axis below its floor remains an open gap. Record the exact build and
run recipe, revisions, model hashes, environment, and contention state.
Reproduce the result on an idle host with a same-binary A/B test before you
accept it.

**Never declare a ceiling.** An apparent same-architecture performance limit is
an unresolved implementation difference. Keep the gap open and name the next
traceable hypothesis.

Report exactly one result for each applicable rule. The result is satisfied, narrowly
waived, pending a named external authority or resource, or failing. A permanent
report-only state is not a result.

## Shared seams

A capability is not done until the shared surface can reach it.

- Route model fusion through `vt::FusedChain`.
- Route mergeable multilayer perceptron (MLP) projections through
  `layers::MlpGateUpMethodBase` and `vt::MergedGemmGroup`.
- Route decode through `ModelRegistry::Forward`, `dense_attn::AttnBlock`, and
  on-device sampling.
- Expose every shipped capability through `include/vllm.h`. Examples and
  servers are thin clients of this application binary interface (ABI). They
  never include internal headers.

Extend a shared seam when it cannot represent the upstream behavior. Otherwise,
record one exact tracked exception. Never write a parallel path by hand.

Add new files for new hardware, architectures, and models. Mirror the vLLM file
structure.

A model port includes the **quantized arms, not only bf16**. GGUF k-quants are a
standing requirement. They are not a choice for each model. Most users run the
quantized arms, and a quant-matched llama.cpp comparison needs them. Refuse an
unimplemented arm with a message that names the missing part. Record the arm as
owed. Never leave the missing path to be discovered later. Use
[`.agents/porting-a-model.md`](.agents/porting-a-model.md) as the checklist.

## Records

Every inventory item has a stable ID. It records the upstream source, local
anchor, tests and evidence, spec, lifecycle state, owner, and issue in the
correct matrix. When a lifecycle state changes, update the roadmap row and its
owning matrix row in the same change.

For a concurrent edit to a keyed record, take the complete target-branch
version. Apply the scoped edit again. Verify that unrelated keys remain
byte-for-byte equal. Union-append only a genuinely append-only log. Never accept
an automatic three-way merge of a keyed record.

**Do not create a surface that every pull request must write.** If N concurrent
pull requests edit file F, that file is a lock. A record surface can have only
one of these shapes:

- One file per row, read with a glob.
- A genuinely append-only file that can union-merge.
- A value that is derived at read time and is not stored.

Rewrite every other record surface into one of these shapes.

Two rules follow. **Limit an entry, not a shared file.** A shared-file budget
forces each addition to remove another entry. Merging two such edits cleanly is worse than
conflicting, because it applies both removals. **Never store a measurement of one file inside another file.** A
number that changes after each edit couples every pull request to lines that it
does not own.

A gate often creates the lock. If a checker requires every change to edit one
shared file, the checker is defective. Move the obligation to a per-row surface.
Do not delete the obligation.

Move superseded detail into `.agents/completed/`. Keep its links and provenance.
Never delete evidence to reduce context.

## Public documents

Each public document has one purpose and one trigger. These documents are
projections, not narratives. Each fact lives in one document.

| Surface | Changes when |
|---|---|
| `docs/STATUS.md` | a row changes lifecycle state |
| `docs/BENCHMARKS.md` | a row gains an accepted or explicitly pending/failed/void measurement |
| `docs/FEATURES.md` | a feature, model, backend, or quantization surface changes |
| `docs/USAGE.md` | a command, C API, config key, install step, or workflow changes |
| `README.md` | a user-visible headline, positioning, or quick start changes |
| the moved row spec's `## Now` | a row changes lifecycle state |

Editing `src/`, `include/`, or `tests/` on its own owes none of these. A
lifecycle change owes `STATUS`, `BENCHMARKS`, and the moved row spec's `## Now`.
`.agents/NOW.md` is authored only at operator cadence and is never a per-row
lifecycle write.

## Work happens in a worktree

**Do every unit of work in its own linked worktree and task branch.** A unit of
work is one issue's change together with the records that change invalidates. A
feature, a fix, a policy change, a document, and a one-line gate repair are each
a unit. A claimed row uses `row/<ID>`. Pin the base SHA when you create the
worktree.

**A record edit rides in the pull request whose change made the record stale.**
It is not a unit of work by itself. A record-only pull request is still correct
when the record is the work: a stale row, a corrected pin, a newly filed gap. It
is wrong when it only restates what just landed, because that costs a branch, a
gate run, and a fresh review to say something the landing change already knew.
This narrows what counts as a unit. It never licenses bundling unrelated work
into one branch.

Keep the shared checkout on a clean `main`. **Never use it as a work surface.**
Other worktrees branch from it, so it must remain current and safe. Never edit,
commit, or stash in the shared checkout.

Remove the worktree and delete its branch when the work merges, closes, or is
abandoned. A worktree is a complete checkout. Stale worktrees fill the disk and
can make gates fail when temporary space runs out.

## Landing work

Move work to `main` from its task branch, never from the shared checkout. A
helper opens a reviewed `row/<ID>` pull request. An operator with recorded merge
authority can merge the branch locally with a commit that names it. This local
merge keeps an in-flow repair to one step. The task branch still makes the
change visible, reversible, and attributable.

Run the applicable gate before every push. Chain the successful gate directly
to the exact-SHA push. Never force-push or add a force option to a script. A
rejected push protects another merge. Fetch, merge again, rerun the gate, and
push again. Hooks are bypassable convenience, not evidence. If you cannot query
the remote, report `REMOTE_UNVERIFIED`. Unknown is not absence or success, and
it does not authorize cleanup.

Merge each verified pull request in the current session. Close an obsolete pull
request and record the reason. Never end a session with a verified, unmerged
pull request.

Every commit contains a bare `FOLLOWING_AGENTS_PROTOCOL` paragraph and these
trailers:

```text
Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:MODEL [TOOL]
```

AI tools never add `Signed-off-by` or `Co-Authored-By`. The human submitter owns
and reviews the change.

This prohibition stops an AI from claiming authorship. It does not stop the
forge from recording the submitter. GitHub can add a `Co-authored-by` trailer
for the account that opens a pull request. The accepted form uses an
`@users.noreply.github.com` address. This trailer records the submitter, even
when the account is a bot. `AI-Assisted` and `Assisted-by` record AI involvement
and are always required. `Signed-off-by` has no exception because it is a legal
assertion, not attribution.

Classify policy, checker, document, script, test, continuous integration (CI),
generated, and product paths explicitly. Never hide mutable files behind a
general directory exemption. The project has no line budget. It retired the
per-class limits on 10 August 2026. Nine of the previous 22 merged pull requests
exceeded the product limit. Tests were one-third to one-half of each large
change. The gate therefore failed normal work, and it charged red-first
mutation tests against the same allowance as kernel code. Size is a review decision. Split a change when parts
help the reviewer, not when a counter requires it.

## Changing the rules or a checker

A checker's message defines what it enforces. This file states the rule in
prose. No gate checks whether the two descriptions agree, and that is deliberate.
Keeping two descriptions in sync is the failure mode that this protocol was
built to remove.

A semantic checker change needs a spec, a red-before test or mutation, and
green-after evidence. Never make a red gate green by deleting an assertion or
widening its scope.

The project has no waiver registry. An exception registry is a state log, and
this protocol has no state log. A commit that needs an exception argues for it
in its own message. The reason then stays with the diff, author, and date. It
cannot drift from the tree because it is part of Git history. Use
`git log --grep` to find it. An exception is visible debt, not success. A
reviewer who rejects the reason does not merge the change.

## Task guides

Read the guide for the current job.

| Doing this | Read |
|---|---|
| Porting a MODEL (the coverage checklist) | [`.agents/porting-a-model.md`](.agents/porting-a-model.md) |
| Porting a model, kernel, or feature from vLLM | [`.agents/porting.md`](.agents/porting.md) |
| Running gates, proving correctness, reviewing | [`.agents/verification.md`](.agents/verification.md) |
| Measuring performance | [`.agents/benchmarking.md`](.agents/benchmarking.md) |
| Fixing a bug | [`.agents/bugfixing.md`](.agents/bugfixing.md) |
| Working on a specific host or GPU | [`.agents/environment.md`](.agents/environment.md) |
| Coordinating parallel work | [`.agents/workflow.md`](.agents/workflow.md) |
| Chasing a parity lever | [`.agents/parity-lever-protocol.md`](.agents/parity-lever-protocol.md) |
| Syncing the vLLM pin | [`.agents/upstream-sync.md`](.agents/upstream-sync.md) |

## Commands

```sh
scripts/agent-start.py                          # always first
python3 scripts/agent-role.py show
scripts/agent-preflight.sh                      # before edits
scripts/agent-preflight.sh --staged             # before commit
python3 scripts/agent-ready.py                  # before remote handoff
python3 scripts/agent-integration.py --base origin/main
```

Never push, merge, manage services, use external compute, or download large
assets without authority recorded in developer preferences or given for the current
task.
