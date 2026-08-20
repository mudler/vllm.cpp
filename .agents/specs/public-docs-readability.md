# Public documentation readability campaign

Work ID: `ENG-DOCS-PUBLIC-READABILITY`.
Roadmap owner: [`A6`, User-facing surface closure](../roadmap_v1.md).
Issue: [#1463](https://github.com/mudler/vllm.cpp/issues/1463).
Base: `c8d926ea82bd6d8f5d6312693572c84234a6a7f3`.

## Now

The developer approved one campaign in one pull request. The spec and
implementation use that pull request. The operator can merge the verified
documentation-only pull request without waiting for continuous integration.

The next action is W1, the content inventory and destination map. No public
document changes in this spec commit.

This work ID is a claim and branch identifier. It is not a new lifecycle row in
the engine matrix. Roadmap campaign `A6` owns this documentation unit. Issue
#1463 uses the accepted spec-owned form in the append-only issue index.

## Problem

The public documentation does not give a user a short path from installation to
a working command. `docs/USAGE.md` has 6,303 lines and more than 55,000 words at
the campaign base. It mixes user procedures with implementation phases, issue
history, gate analysis, benchmark interpretation, and contributor instructions.

`docs/STATUS.md` has 2,906 lines. It mixes current capability state with dated
narratives and superseded results. Other public pages repeat build options,
environment variables, feature claims, and use-case instructions.

The result is a navigation defect. A user must read project history to find a
current command. It also creates drift because one fact can appear in several
public pages.

Issues [#342](https://github.com/mudler/vllm.cpp/issues/342),
[#1281](https://github.com/mudler/vllm.cpp/issues/1281),
[#1275](https://github.com/mudler/vllm.cpp/issues/1275), and
[#704](https://github.com/mudler/vllm.cpp/issues/704) identify concrete examples.
They remain open at the campaign base. This campaign reconciles their facts when
the current tree still confirms them.

## Scope

This campaign changes the public documentation as one reviewable unit:

- Keep `docs/USAGE.md` as the generic usage guide and navigation index.
- Keep the required compact checkpoint registry in `docs/USAGE.md`.
- Extract model recipes to `docs/models/`.
- Extract cross-model workflows to `docs/guides/`.
- Extract dense lookup material to `docs/reference/`.
- Reduce `docs/STATUS.md` to current lifecycle and capability state.
- Remove overlap across `BUILD.md`, `ENVIRONMENT.md`, `FEATURES.md`, and the
  specialized guides.
- Preserve the current purpose of `BENCHMARKS.md` and `RELEASES.md`.
- Update `README.md` only where its public navigation or quick start becomes
  stale because of this campaign.
- Reconcile applicable facts from issues #342, #1281, #1275, and #704.
- Move unique internal history to an owning spec, a completed record, or
  `docs/bench-evidence/` before removing it from a public page.

## Exclusions

- No product code or runtime behavior changes.
- No build-system or continuous-integration behavior changes.
- No checker semantic changes or weaker assertions.
- No new benchmark, changed benchmark result, or changed performance claim.
- No lifecycle promotion without evidence that predates this campaign.
- No checkpoint, supported arm, refused arm, limitation, or warning deletion.
- No generated documentation site or new publishing toolchain.
- No rewrite of agent policy or contributor workflow.

If a public statement exposes a product defect, file or identify its issue. Do
not repair product code in this campaign.

## Upstream chain

This campaign does not mirror runtime behavior. vLLM does not define how this
repository presents its local public documentation. The pinned vLLM oracle and
secondary oracles have no executable role in this campaign.

Current code, public interfaces, canonical matrices, accepted evidence, and Git
history define the facts that the campaign must preserve. The campaign does not
use another documentation site as a content or structure oracle.

## Our baseline

The base has 11 top-level public Markdown files and 12,124 lines in those files.

| Surface | Lines | Current role | Campaign decision |
|---|---:|---|---|
| `docs/USAGE.md` | 6,303 | Usage, models, checkpoints, history, and evidence | Keep generic workflows, indexes, and checkpoint registry |
| `docs/STATUS.md` | 2,906 | Current state plus chronology | Keep current state only |
| `docs/BENCHMARKS.md` | 576 | Accepted and pending measurements | Keep measurement projection |
| `docs/FEATURES.md` | 392 | Feature summary | Keep shipped capability projection |
| `docs/ROCM.md` | 381 | ROCm build and use | Move or retain as one focused user guide |
| `docs/SPECULATIVE-DECODING.md` | 316 | Speculative decoding | Move or retain as one focused user guide |
| `docs/ENVIRONMENT.md` | 316 | Environment variables plus explanation | Keep environment-variable reference |
| `docs/BUILD.md` | 280 | Build procedures and options | Keep build guide |
| `docs/SGLANG-COMPAT.md` | 242 | SGLang compatibility | Keep focused compatibility guide |
| `docs/KV-OFFLOAD.md` | 179 | KV offload workflow | Move or retain as one focused user guide |
| `docs/WEIGHT-OFFLOAD.md` | 165 | Weight offload workflow | Move or retain as one focused user guide |
| `docs/RELEASES.md` | 68 | Release artifacts | Keep release reference |

`docs/USAGE.md` already contains duplicate model sections. MiniMax-H3 and
MiniMax-Music3 each appear in more than one location. It also contains headings
such as `Running the gates`, implementation wave names, and later-phase warnings.
Those are evidence or plans, not user procedures.

## Port map

Nothing is ported from vLLM. This table maps each local source class to its
planned local destination.

| Current source | Planned destination | Change kind |
|---|---|---|
| Generic sections in `docs/USAGE.md` | shorter `docs/USAGE.md` | Rewrite and deduplicate |
| Model sections in `docs/USAGE.md` | `docs/models/*.md` | Extract and rewrite |
| Cross-model sections in public pages | `docs/guides/*.md` | Extract and rewrite |
| Dense flag and interface sections | `docs/reference/*.md` | Extract and rewrite |
| Historical prose in `docs/STATUS.md` | owning internal evidence | Archive before removal |
| Overlapping build and environment prose | `BUILD.md` or `ENVIRONMENT.md` | Deduplicate |

## Public document contract

Each public surface answers one user question:

| Surface | User question |
|---|---|
| `README.md` | What is this project, and how do I reach the first request? |
| `docs/USAGE.md` | How do I run the common CLI, server, C API, and C++ workflows? |
| `docs/models/*.md` | How do I run this model, with which exact weights and limits? |
| `docs/guides/*.md` | How do I complete this task across model families? |
| `docs/reference/*.md` | What are the exact values, fields, flags, and interfaces? |
| `docs/BUILD.md` | How do I build the project for my target? |
| `docs/ENVIRONMENT.md` | Which environment variables exist, and what do they do? |
| `docs/FEATURES.md` | Which capabilities ship now? |
| `docs/STATUS.md` | What is the current lifecycle state? |
| `docs/BENCHMARKS.md` | Which measurements are accepted, pending, failed, or void? |
| `docs/RELEASES.md` | Which release artifacts exist, and how do I verify them? |

A fact has one primary public home. Other pages link to that home instead of
restating the fact. A required checkpoint record remains in `docs/USAGE.md`
because `AGENTS.md` names that literal surface.

## `USAGE.md` design

`docs/USAGE.md` uses this order:

1. Prerequisites and the shortest generic local workflow.
2. Common command-line inference.
3. OpenAI-compatible server use.
4. C application binary interface use.
5. C++ library use.
6. Shared configuration and first-line troubleshooting.
7. A task-oriented guide index.
8. A model index.
9. The compact checkpoint registry.

The generic sections contain runnable commands with placeholders that the page
defines. They do not depend on a model-specific narrative. Each specialized
procedure links to one model page or guide.

The checkpoint registry records each applicable arm with these fields:

- model or component;
- file name and size;
- exact Hugging Face repository and revision;
- SHA-256 for each quantized artifact;
- supported arms;
- refused arms and the named missing part.

Model pages can explain a checkpoint, but they do not replace this registry.

## Directory design

The implementation can add these directories as content requires:

```text
docs/
  models/       model-specific checkpoints, commands, and limitations
  guides/       tasks that apply to more than one model family
  reference/    dense lookup tables and interface details
```

Use lowercase kebab-case file names. Do not create one-line routing pages. Keep
a topic in an existing focused top-level guide when moving it would add no
navigation value. Every extracted page must appear in an index before the
campaign reaches review.

## Content classification and migration rules

Classify each source section before editing it:

| Class | Destination | Rule |
|---|---|---|
| Generic user procedure | `docs/USAGE.md` | Keep one runnable common path |
| Model recipe | `docs/models/<model>.md` | Keep commands, weights, limits, and refusals |
| Cross-model workflow | `docs/guides/<task>.md` | Keep task steps and shared behavior |
| Dense lookup material | `docs/reference/<topic>.md` | Keep exact names and values |
| Current feature state | `docs/FEATURES.md` or `docs/STATUS.md` | Keep one current keyed statement |
| Measurement | `docs/BENCHMARKS.md` | Keep the accepted value and reproduction pointer |
| Reproduction artifact | `docs/bench-evidence/` | Keep the exact command, environment, and raw result |
| Design or implementation history | owning spec or `.agents/completed/` | Move unique facts before removing public prose |
| Contributor procedure | `CONTRIBUTING.md` or `.agents/` | Remove from public usage only after a valid owner exists |

Do not move prose mechanically. Rewrite each destination around the user's task.
Keep literal commands, flags, identifiers, revisions, hashes, and measured
values unchanged unless the current source of truth proves they are stale.

## Unique evidence preservation

Before deleting or compressing a source section, the implementer records a
migration manifest in the campaign worktree. The manifest maps every source
heading to its destination and one disposition:

- `kept`: the fact stays on the same public page;
- `moved`: the fact has one named destination;
- `deduplicated`: another named public page already owns the same fact;
- `archived`: a named spec, completed record, or evidence file receives it;
- `stale`: current code or a current record disproves it, with an issue or Git
  anchor that explains the correction.

The manifest can live in this spec during implementation. It must not become a
second public record. A reviewer samples every class and checks every checkpoint
and measurement row against its source.

Never delete the only copy of:

- a checkpoint file, size, repository, revision, or checksum;
- a supported or refused execution arm;
- a user-visible default, error, limitation, or safety warning;
- an accepted, failed, pending, or void measurement;
- a reproduction command or environment condition;
- the reason for a narrow project divergence.

## Writing rules

Repository prose follows `.agents/style/prose.md`. Public pages use neutral,
plain technical language. Apply these passes in order:

1. Match established names in code and nearby documentation.
2. Apply the technical-English rules to procedures and reference prose.
3. Remove diff-anchored narration, rhetorical framing, repeated conclusions,
   and other patterns identified by the `no-ai-slop` skill.
4. Apply the `humanizer` skill in its neutral technical mode. Do not add
   personality, facts, claims, or citations.

The editing passes must preserve technical facts. A shorter sentence is not an
improvement if it loses a condition.

## Work breakdown

| Wave | Work | Completion condition |
|---|---|---|
| W0 | Commit this spec and traceability records | Git history places this commit before all public-doc edits |
| W1 | Build the heading inventory and migration manifest | Every top-level public section has a class and destination |
| W2 | Establish directories, indexes, and the generic `USAGE.md` path | CLI, server, C API, and C++ paths are reachable from `USAGE.md` |
| W3 | Extract model pages and compact the checkpoint registry | Every model recipe is indexed and every required checkpoint field remains |
| W4 | Extract cross-model guides and dense references | Each extracted page is indexed and has one public purpose |
| W5 | Reduce `STATUS.md` and reconcile `FEATURES.md`, `BUILD.md`, and `ENVIRONMENT.md` | Current projections remain and historical narrative has a named owner |
| W6 | Reconcile README navigation and issues #342, #1281, #1275, and #704 | Applicable facts match the current tree and stale claims are removed |
| W7 | Apply technical-English, no-AI-slop, and humanizer passes | The prose is plain, current, and free of added claims |
| W8 | Run the final full-document gate and prepare the outcome | All declared gates pass on one immutable head |

Each wave lands as a scoped commit when that split helps review. All commits
remain in the single campaign pull request.

## Review waves

A fresh reviewer reviews each immutable wave head. The reviewer must not be the
agent that wrote that wave.

| Review | Scope | Required checks |
|---|---|---|
| R1 | W2 generic usage and navigation | Run each generic command shape through static validation; break one index link in scratch and require the link gate to fail |
| R2 | W3 model pages and checkpoint registry | Compare every registry field with the pre-change source; remove one required field in scratch and require its gate to fail |
| R3 | W4 guides and references | Check one-owner boundaries, link reachability, and duplicated instructions |
| R4 | W5 status and core projections | Compare each retained state with the owning matrix or evidence; restore any unique history missing a destination |
| R5 | W6 and W7 navigation and language | Check issue facts, terminology, commands, and accidental claim changes |
| R6 | Full immutable head | Review the complete diff, migration manifest, all links, all public tables, and all declared gates |

Findings return to a fresh implementer. The operator reruns the campaign gate
after the final reviewer reports `PASS`.

## Tests to port

This documentation campaign has no vLLM behavior or upstream test to port. Its
tests verify the public information architecture and existing repository
contracts.

Required focused checks:

- validate all relative Markdown links in `README.md` and `docs/`;
- check that each extracted page has one `#` title and correct heading nesting;
- check that every page under `docs/models/`, `docs/guides/`, and
  `docs/reference/` appears in an index;
- check that indexed destinations exist and no duplicate index target appears;
- check the checkpoint registry for every required field;
- run `scripts/check-public-doc-tables.py`;
- run `scripts/check-supported-models.py`;
- run `scripts/check-env-doc.py`;
- run `scripts/check-readme-structure.py`;
- run `scripts/check-surface-coverage.py`;
- run `scripts/check-doc-checkpoint.py` over the campaign range;
- run `git diff --check`.

Use an existing checker when it already proves a guarantee. If no existing
checker covers index completeness or link reachability, add a test without
weakening checker semantics. A new checker requires its own red-before and
negative-mutation evidence under repository policy.

For moved content, compare checkpoint, measurement, flag, environment-variable,
and refusal inventories before and after. The after inventory can change only
when the migration manifest names the destination or the evidence for a stale
fact.

## Dependencies

- Issue #1463 remains open and owns the campaign.
- The committed W0 spec must be reachable before an implementation wave starts.
- The current public-doc and agent-record gates must pass without semantic
  weakening.
- The implementation needs no GPU, model artifact, oracle environment, package
  installation, or external service.
- The final merge depends on fresh review `PASS`, the operator's local gate, and
  a valid pull request body. Continuous integration completion is not a merge
  dependency for this developer-authorized documentation-only campaign.

## Gates

The implementation gate is:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-public-doc-tables.py
python3 scripts/check-supported-models.py
python3 scripts/check-env-doc.py
python3 scripts/check-surface-coverage.py
python3 scripts/check-doc-checkpoint.py --base <campaign-base> --head HEAD
git diff --check <campaign-base>..HEAD
scripts/agent-preflight.sh --staged
```

The implementer records the exact link, heading, index, and inventory commands
selected in W1. The operator reruns the same commands at the immutable final
head. No GPU, model download, or oracle run applies because this campaign does
not change runtime behavior or parity claims.

## Risks and mitigations

- **Lost evidence:** The migration manifest requires one disposition and one
  destination for each source heading.
- **Changed claims during rewriting:** Reviewers compare exact identifiers,
  values, defaults, refusals, and measurements before and after.
- **New navigation drift:** Every extracted page appears in an index, and the
  link check resolves every target.
- **Checkpoint contract violation:** The compact registry remains in
  `docs/USAGE.md` with every field required by `AGENTS.md`.
- **Concurrent public-doc edits:** The operator takes the complete target-branch
  version of keyed records and reapplies scoped edits before landing.
- **An unreadable mega-diff:** Ordered commits and review waves keep each content
  class reviewable inside one pull request.
- **Over-editing:** The human-language passes preserve facts and use neutral
  technical prose. They do not manufacture voice.
- **False current state:** `STATUS.md` statements must point to an owning matrix,
  spec, issue, or accepted evidence record.

## Git integration

The developer selected one pull request for the spec and implementation. The
branch is `row/ENG-DOCS-PUBLIC-READABILITY`. The spec commit precedes every public
documentation commit.

The operator can merge the verified documentation-only pull request to `main`
without waiting for continuous integration. This authority does not waive any
local gate, fresh review, pull request body check, attribution rule, or
non-force-push rule.

## Owed

- [#1463](https://github.com/mudler/vllm.cpp/issues/1463) owns W1 to W8 of this
  campaign. The issue closes only after the verified documentation-only pull
  request reaches `main`.

## Stop conditions

Stop and return `NEEDS_DECISION` if:

- a fact needs a new public owner outside the approved document boundary;
- a required checkpoint field cannot remain in `docs/USAGE.md`;
- a current claim conflicts with two authoritative records;
- preserving unique evidence requires a product or checker change;
- an issue requires a runtime fix rather than a documentation correction;
- a review wave proposes a new campaign scope.

Stop and return `BLOCKED` if:

- a required public-doc gate cannot pass without weakening it;
- a source section has unique evidence but no valid owning record;
- another branch changes the same keyed record and the target version cannot be
  reconciled safely;
- the final immutable head lacks a fresh reviewer `PASS`.

## Outcome

Pending implementation and final gate evidence.
