# AGENTS.md — the rules

This file is the complete policy for `vllm.cpp`. It is the only file every agent
loads automatically, so every rule lives here. Files under `.agents/` are task
guides — how to do a specific job — and they can never add or weaken a rule
here.

The project mirrors vLLM in C++ with no PyTorch and no ggml dependency. vLLM is
the reference for behavior and the bar for speed.

## Start here

1. Run `scripts/agent-start.py`. Pass `--intent operator|helper|read-only` and
   `--row <ID>` when you already know them; otherwise relay its welcome and ask
   what work is intended. Follow its printed action, then rerun it.
2. Declare a role: `scripts/agent-role.py claim operator` for a multi-step
   integration campaign, `claim helper --row <ID>` for one scoped task, or
   `claim read-only` for inspection. The operator claim records this worktree
   as a coordinator; it is never refused because someone else is coordinating.
   Add `--headless` only when the developer explicitly says the run is
   unattended. Never infer it.
3. Run `scripts/now.py` for the live position, and read `.agents/NOW.md`
   for the operator's current gate and next actions. The first is derived;
   the second is authored and fits on one screen.
4. Read only the claimed row, its spec, its evidence, and the task guide for
   what you are about to do.
5. Run `scripts/agent-preflight.sh` before you edit anything.

Never infer a role, host, permission, or developer preference. Resolve `.env`
and `.agents/developer-preferences.md` from the shared checkout, asking only for
the single value the current gate needs. An unavailable value leaves its gate
`PENDING`; it never becomes an assumption. Preferences control operations only —
they can never reduce a correctness, evidence, attribution, or testing
obligation.

## History is git

There is no state log. Git is the history, and it cannot disagree with the tree.

| Question | Command |
|---|---|
| Did this row already land? | `git log --oneline --grep '<ROW-ID>'` |
| When did this symbol change? | `git log -S'<symbol>' --oneline -- <path>` |
| What happened to this file? | `git log --follow --oneline -- <path>` |
| What is on main that I lack? | `git log --oneline HEAD..origin/main` |
| Why is this line like this? | `git log -L '<start>,<end>:<path>'` |
| What did that commit change? | `git show --stat <sha>` |

The roadmap row states where a row is *now*; git and the row's spec state how it
got there. Before concluding anything about past work, check the spec and
`git log -S` — do not re-derive it.

## Every change starts from an issue

**No work without an open GitHub issue.** Before claiming a row or writing code,
confirm an issue tracks the work; if none exists, open one. Link it in three
places that must agree: the issue table in
[`.agents/roadmap_v1.md`](.agents/roadmap_v1.md), the row's spec, and the PR
body.

A bug you find while doing something else still gets an issue — but filing it
does not mean deferring it. File it, fix it in the same flow, reference it in
the commit, and close it. The traceability is what matters, not the round trip:
the person who just found the bug has the context to fix it, and making them
hand it off loses that.

This covers the small and obvious. A fix that needs its own spec, changes a
checker's semantics, or would surprise a reviewer still takes the normal
row / spec / fresh-review path — "fix it in-flow" is not a bypass for those.

## Spec before code

No row becomes `READY` or `ACTIVE` without a committed
`.agents/specs/<slug>.md`. The spec is committed *before* implementation, never
written up afterwards. It carries scope, upstream anchors, design, risks, tests,
gates, evidence, and stop conditions.

Before claiming, re-verify the gap against current code, tests, issues, PRs,
`NOW.md`, and the owning row. If it already landed, is already claimed, or no
longer matches its record, reconcile the record first and do not implement.

When a row reaches `DONE`, its spec carries an `## Outcome` section: what was
measured, what was rejected and why, and why any default is set the way it is.
That is the one thing neither the code nor git records.

## How work gets done

Work is delegated, reviewed by someone else, and verified by the operator. This
sequence is the method, not a suggestion:

1. A **fresh implementer** works from the committed spec. It ports or writes the
   smallest test that fails for the intended reason, captures the red result,
   makes the minimum complete change, gets focused green, then runs the full
   gate.
2. A **fresh reviewer** — never the agent that wrote the code — reviews the
   immutable head. It inspects statically *and* mutates the claimed guarantees
   in a scratch copy to prove the tests catch their defect. Mutate, don't just
   read. Restore the tree byte-for-byte afterwards.
3. **Findings return to a fresh implementer.** Never repair a finding in the
   coordinating session. Repeat focused gate, full gate, and fresh scoped review
   until PASS. Attempt budgets are scheduling controls and never terminate a
   correctable finding; only explicit developer direction or a precise external
   blocker stops the loop.
4. The **operator reruns the row's gate itself**. An implementer or reviewer
   report is an input, never a gate result.

Every delegated task states goal, context, exact scope and exclusions,
constraints, done-when, required evidence, authority, output contract, and stop
conditions. Missing binding context returns `NEEDS_CONTEXT` rather than a guess;
a material disagreement returns `NEEDS_DECISION` rather than silent scope
change. Use the versioned contracts in [`.agents/prompts/`](.agents/prompts/).

The operator is a **coordinator**. It holds the plan and the GPU, merges
reviewed PRs, dispatches sub-agents into separate worktrees, and does not write
implementations that should be independently reviewed. **Several operators may
run at once** — `scripts/agent-role.py claim operator` records who is
coordinating where and never refuses; `show` lists the others.

**`main` is never force-pushed.** No `--force`, no `--force-with-lease`, by
anyone, ever. That is what makes concurrent coordinators safe: a plain
`git push` refuses any non-fast-forward, so git itself is the interlock. A
rejected push means fetch, re-merge, re-run the gate, and push again — never
force.

## vLLM is the reference

**Mirror it.** When vLLM defines behavior, mirror every applicable mode,
default, error, and edge case. Escalate only a genuine product decision; never
ask how a mirrored feature should behave.

**Pin it.** Comparisons run against the pinned oracle recorded in
[`.agents/upstream-sync.md`](.agents/upstream-sync.md). Advance the pin only
after every affected row and gate is reconciled. An oracle is only gateable once
it demonstrably *builds and runs* the model — constructing a config proves
nothing.

**Verify against both the running oracle and its source.** Every change is
checked two ways: execute the pinned vLLM on the identical workload, and read
the upstream path it corresponds to. Ground conclusions in the whole executing
chain — FlashInfer, CUTLASS, cuBLASLt, DeepGEMM, torch/Inductor, generated
code, and local dispatch — and cite the `file:line` you ported from. Dump the
generated kernel before calling a lever unreachable. Anything written from
scratch is recorded as such in the porting inventory.

**Inherit its defaults, do not re-invent them.** vLLM resolves ONE model dtype
and every layer inherits it; `f32` is a rare, annotated escape. Mirror that
polarity. A buffer or GEMM output that names `f32` on a model path owes a
one-line reason next to it. **A token gate cannot catch a dtype that is too
WIDE** — it is still numerically correct, so tokens match, goldens pass, and we
move twice the bytes. Check the memory format against the oracle explicitly,
per [`.agents/porting.md`](.agents/porting.md).

**Port its tests in the same change**, preserving parameters, modes, fixtures,
tolerances, failure cases, and the upstream revision anchor. Document only
unavoidable harness adaptation.

**Trace both sides with the same tool** on the identical workload before any
throughput comparison. Source inspection establishes candidates; matching traces
establish what actually ran. A GEMM/GEMV invocation-parity claim proves output
dtype, compute and scale type, entry point, algorithm policy, and resolved
template dtypes *in the same tool*.

## When vLLM has no implementation

vLLM — including `vllm-project/vllm-omni` — is the primary reference, and
wherever it implements the behavior it is the only one. Where it implements
nothing, the work is still not ungated. It runs against a **secondary oracle**,
which is admissible only if it is one of these upstreams and only at a recorded
pin:

<!-- oracle-registry:begin -->

| Oracle | Registry id | Reach for it when |
|---|---|---|
| vLLM | `vllm` | always, wherever it implements the behavior — the primary |
| vLLM-Omni | `vllm-omni` | diffusion, TTS and the omni-only architectures vLLM proper never registers |
| HuggingFace `transformers` | `transformers` | a model, processor or tokenizer's own reference implementation — the source vLLM itself mirrors |
| `diffusers` | `diffusers` | schedulers, VAEs and diffusion pipelines |
| SGLang | `sglang` | a model or serving path SGLang implements and vLLM does not |
| SGLang-Omni | `sglang-omni` | omni, speech, TTS and music models served by SGLang's pipeline runtime — a third repository, not SGLang |
| llama.cpp | `llama-cpp` | CPU and GGUF k-quant floors |
| Tenstorrent tt-forge | `tt-forge` | Tenstorrent hardware, which vLLM has no backend for at all |

<!-- oracle-registry:end -->

**Pin every one of them.** Each has a file of its own,
[`.agents/oracles/<id>.md`](.agents/oracles/), carrying the revision, the date it
was measured, whether it is gateable, and the evidence. An upstream with no pin is
not an oracle, it is a moving target, and a number measured against it cannot be
reproduced. The vLLM parity pin keeps its home in
[`.agents/upstream-sync.md`](.agents/upstream-sync.md); its oracle file points
there rather than restating it. One file per oracle, read by glob — never a
shared table every change has to write.

**A secondary oracle answers one question and no others:** what correct output
looks like on a path vLLM cannot produce at all. It never outranks vLLM, and it
never becomes the mirror source — behavior, defaults, structure and naming still
mirror vLLM wherever vLLM speaks, which is why "SGLang does it differently" is
never on its own a reason to diverge. When vLLM later implements the path, the
row reconciles onto vLLM and says so in its spec.

**Gateability is measured, not assumed.** The primary's rule holds for all of
them: an oracle is gateable once it demonstrably *builds and runs* the model.
Until it does, its file records `gateable = no` and names the issue that owes the
measurement, so an ungateable lane is visible debt rather than a discovery
someone makes mid-campaign.

## Gates

Correctness first, always. Establish the declared token-exact gate — or an
explicitly ratified distributional gate where the oracle's own greedy decode is
non-deterministic — before accepting any performance result. Never trade
correctness for throughput.

Both sides use the pinned oracle with identical model artifacts, prompts, token
counts, batching, concurrency, and sampling. The honest denominator is vLLM's
production configuration, never `--enforce-eager`.

Record values and ratios for every required throughput, latency, and memory
axis; any axis below floor is an open gap. Record the exact build and run
recipe, revisions, model hashes, environment, and contention state, and
reproduce the result on an idle box with same-binary A/B before accepting it.

**Never declare a ceiling.** An apparent same-architecture performance limit is
an unresolved implementation difference. Keep the gap open and name the next
traceable hypothesis.

Report exactly one result per applicable rule: satisfied, narrowly waived,
pending a named external authority or resource, or failing. Permanent
report-only is not a result.

## Shared seams

A capability that is not reachable through the shared surface is not done.

- Route model fusion through `vt::FusedChain`.
- Route mergeable MLP projections through `layers::MlpGateUpMethodBase` and
  `vt::MergedGemmGroup`.
- Route decode through `ModelRegistry::Forward`, `dense_attn::AttnBlock`, and
  on-device sampling.
- Expose every shipped capability through `include/vllm.h`. Examples and servers
  are thin clients of that ABI and never include internal headers.

If a shared seam cannot represent the upstream behavior, extend it or record one
exact tracked exception. Never hand-roll a parallel path.

New hardware, architectures, and models are **additive** files that mirror
vLLM's structure.

A model port covers the **quantized arms, not just bf16**. GGUF k-quants in
particular are a standing requirement, not a per-model choice: they are what most
users can actually run, and they are what a quant-matched llama.cpp comparison
needs. An arm that is not implemented is refused with a message naming the
missing piece and recorded as owed — never left to be discovered later.
[`.agents/porting-a-model.md`](.agents/porting-a-model.md) is the checklist.

## Records

Every inventory item has a stable ID and records upstream source, local anchor,
tests and evidence, its spec, lifecycle state, owner, and issue in the correct
matrix. When lifecycle state changes, update the roadmap row and its owning
matrix row in the same change.

Resolve concurrent edits to a keyed record by taking the target branch version
wholesale and reapplying your scoped edit; verify unrelated keys byte-for-byte.
Union-append only genuinely append-only logs. Never accept an automatic
three-way merge of a keyed record.

**No surface that every PR must write.** If N concurrent PRs all edit file F,
then F is a lock. A record surface is admissible in one of three shapes only:
**one file per row**, globbed for reading; **genuinely append-only**, so it
union-merges; or **derived at read time**, so nobody writes it. Rewrite anything
else into one of the three.

Two corollaries. **Cap the entry, never the file** — a budget on a shared file
turns every addition into evicting someone else's content, and merging two such
edits cleanly is worse than conflicting, because it applies both evictions.
**Never store a measurement of one file inside another** — a number that moves on
every edit couples every PR to lines it does not own.

A gate is what usually creates the lock: if a checker *requires* every change to
touch a shared file, that is the defect, not the discipline of the people
touching it. Relocate the obligation to a per-row surface rather than deleting
it.

Compact by *moving* superseded detail into `.agents/completed/` with links and
provenance intact. Never delete evidence to save context.

## Public documents

Each has one purpose and one trigger. They are projections, not narratives —
each fact lives in exactly one of them.

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

**Every unit of work — feature, fix, policy, docs, records, a one-line gate
repair — happens in its own linked worktree on its own task branch.** A claimed
row uses `row/<ID>`. Pin the base SHA when you create the worktree.

The shared checkout stays on `main`, clean, and is **never a work surface**. It
is what everything else branches from, so it has to be current and safe at all
times. Never edit, commit, or stash in it.

Remove the worktree and delete its branch once the work merges, closes, or is
abandoned. A worktree is a full checkout; leaving them behind fills the disk
until gates start failing for want of a temp file.

## Landing work

Work reaches `main` from its task branch, never from the shared checkout: a
helper opens a reviewed `row/<ID>` PR, and an operator holding recorded merge
authority may instead merge that branch locally with a commit naming it. That
keeps the repair-without-a-round-trip case one step, while still leaving every
change on a branch that git can show, revert, and attribute.

Run the applicable gate before every push and chain that success directly to the
exact-SHA push. Never force-push, and never add a force variant to a script; a
rejected push is git protecting someone else's merge, so fetch, re-merge,
re-gate and push again. Hooks are bypassable convenience, never proof. If the
remote cannot be queried, report `REMOTE_UNVERIFIED` — unknown is neither
absence nor success, and it authorizes no cleanup.

Verified PRs are merged in-session; obsolete ones are closed with the reason
recorded. Never end a session with a verified, unmerged PR.

Every commit carries a bare `FOLLOWING_AGENTS_PROTOCOL` line and these trailers:

```text
Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:MODEL [TOOL]
```

AI tools never add `Signed-off-by` or `Co-Authored-By`. The human submitter owns
and reviews the change.

That forbids an AI *claiming authorship*. It does not forbid the forge from
recording who submitted: a `Co-authored-by` that GitHub generates for the account
opening a pull request — the `@users.noreply.github.com` form — is attribution of
a submitter and is accepted, even when that account is a bot. The claim about AI
involvement is made by `AI-Assisted` and `Assisted-by`, which are the trailers
that carry it and are never relaxed. `Signed-off-by` gets no such exemption: a
sign-off is a legal assertion, not attribution.

Classify policy, checker, doc, script, test, CI, generated, and product paths
explicitly, and never hide mutable files behind a blanket directory exemption.
There is no line budget: the per-class limits were retired 2026-08-10 because
9 of the last 22 merged PRs exceeded the product one and tests were a third to
a half of every large diff, so the gate fired on ordinary work and charged
RED-first mutation tests against the same allowance as kernel code. Size is a
review judgement. Split a change when a reviewer would be better served by
parts, not when a counter says so.

## Changing the rules or a checker

A checker's own message is the authority on what it enforces; this file states
the rule in prose. Nothing verifies that the two agree, deliberately — keeping
two descriptions in sync is the failure mode this protocol was built to remove.

Changing a checker's semantics requires a spec, a red-before test or mutation,
and green-after evidence. You may never turn a red gate green by deleting an
assertion or widening a scope.

There is no waiver registry. A registry of exceptions is a state log, and this
protocol has none — a change that needs an exception argues for it in its own
commit message, where the reason is attached to the diff it excuses, carries an
author and a date, and cannot drift from the tree because it *is* the tree.
`git log --grep` is the record. An exception is visible debt, not success, and a
reviewer who does not accept the argument does not merge it.

## Task guides

Read the one for the job in front of you.

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
assets without authority recorded in developer preferences or given for the
task.
