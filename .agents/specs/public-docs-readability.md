# Public documentation readability campaign

Work ID: `ENG-DOCS-PUBLIC-READABILITY`.
Roadmap owner: [`A6`, User-facing surface closure](../roadmap_v1.md).
Issue: [#1463](https://github.com/mudler/vllm.cpp/issues/1463).
Base: `c8d926ea82bd6d8f5d6312693572c84234a6a7f3`.

## Now

W1 through W8 are implemented on the integrated campaign head. The branch was
reconciled with `origin/main` at `5870cb2bf`; merge commit `df38068da` preserves
main's DFlash2, LTX-2.5, and GB10 block-wise FP8 record changes. The keyed
feature, benchmark, issue-index, and row-spec edits were inspected during the
merge.

The 275-row migration manifest accounts for every campaign-base H1, H2, and H3
heading. `docs/USAGE.md` now owns the generic CLI, server, C ABI, and C++ paths,
the checkpoint registry, and the indexes. Thirteen model recipes, seven task
guides, and four dense references have their own pages; their three directory
indexes provide one route to each extracted page. Five focused guides remain at
their canonical top-level paths because moving them would add redirects without
improving navigation.

`docs/STATUS.md` is a current-state projection. `docs/FEATURES.md`,
`docs/BUILD.md`, and `docs/ENVIRONMENT.md` retain their keyed purposes without
duplicating lifecycle history. Unique implementation, verification, and
measurement evidence remains in the manifest's named specs, completed records,
benchmark evidence, and Git history. W7 removed the temporary legacy appendix
from `docs/USAGE.md` and applied the technical-English, no-AI-slop, and
humanizer passes to the campaign-created public surface.

R1 through R5 reported `PASS` after their repair loops. Their negative checks
proved the index-link gate, seven-field checkpoint registry, manifest
destination validation, release-anchor check, and duplicate-environment-row
check detect the defects they claim to detect. The W8 stale-state assertion
failed before this update because this section still described the legacy body
and future waves. It passes after the outcome update.

Fresh full-head review R6, the operator's final local gate, pull request body
validation, push, and merge remain pending. The developer authorized the
operator to merge the verified documentation-only pull request without waiting
for continuous integration. No GPU, oracle, model download, or quiet-host
measurement applies to this documentation campaign; those external gates are
`NOT APPLICABLE`, not unrun correctness evidence.

This work ID is a branch and claim identifier, not a new engine-matrix
lifecycle row. Roadmap campaign `A6` owns the unit, and issue #1463 remains open
until the pull request reaches `main`.

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

The base has 12 top-level public Markdown files and 12,124 lines in those files.

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

## W1 migration manifest

This manifest accounts for the campaign-base H1, H2, and H3 headings. A source
line identifies headings that have the same text. `Owner` is required for
`archived` and `stale` entries. `n/a` means that the destination remains public.

| Source section | Class | Exact destination | Disposition | Evidence or owner |
|---|---|---|---|---|
| `README.md:38` ## News | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:84` ## Performance | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `README.md:110` ### vs llama.cpp, on CPU, from the same GGUF file | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `README.md:126` ### vs MLX-LM, on Apple M4, warm b=1 | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `README.md:144` ### Speculative decoding | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `README.md:160` ## Quickstart | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:183` ## Features: vLLM parity, then everything else | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `README.md:238` ## Supported models | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `README.md:303` ## Hardware | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `README.md:319` ## Build | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `README.md:339` ## Running inference (CLI) | Generic user procedure | `docs/USAGE.md` | `deduplicated` | `n/a` |
| `README.md:348` ### Multimodal INPUT and video GENERATION | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:365` ## OpenAI-compatible server | Generic user procedure | `docs/USAGE.md` | `deduplicated` | `n/a` |
| `README.md:386` ## Use it as a library (C API) | Generic user procedure | `docs/USAGE.md` | `deduplicated` | `n/a` |
| `README.md:414` ## Why vllm.cpp | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:432` ## Documentation | Navigation index | `README.md` | `kept` | `n/a` |
| `README.md:453` ## Credits, and what we borrow | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:480` ## Citation | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:496` ## Author | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:500` ## Trademarks | Public overview | `README.md` | `kept` | `n/a` |
| `README.md:515` ## License | Public overview | `README.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:1` # Benchmarks <!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished --> | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:3` ## At a glance: W5/W6 green; validated release artifacts pending | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:41` ## vLLM, online serving | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:63` ### GDN prefill kernels by GPU | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:72` ### Qwen3.6-27B by concurrency | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:99` ### Qwen3.6-27B NVFP4 `nvidia` @`0893e160` by concurrency (ModelOpt) | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:142` ### Qwen3.6-35B-A3B by concurrency | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:196` ### Qwen3.8-27B (bf16) by concurrency | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:232` ### Qwen3.8-27B quantized arms, both gates PENDING and no number quoted | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:241` ### DeepSeek-V2-Lite (MLA) | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:260` ### Laguna-S-2.1 (NVFP4) | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:291` ## Memory | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:308` ## llama.cpp, CPU | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:353` ## MLX-LM, Apple M4 | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:368` ## DwarfStar, GGUF | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:419` ## Speculative decoding | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:432` ## How we measure | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:494` ## Open gaps | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BENCHMARKS.md:561` ## Reproduce | Measurement | `docs/BENCHMARKS.md` | `kept` | `n/a` |
| `docs/BUILD.md:1` # Building vllm.cpp | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:9` ## Build out-of-source | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:22` ## CPU build (the correctness / CI reference) | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:33` ## CUDA build (NVIDIA GB10 / DGX Spark) | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:49` ### CUTLASS: the one external build dependency | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:80` ### Other CUDA families | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:102` ## Metal build (Apple Silicon) | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:117` ## Vulkan build | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:127` ## Tenstorrent build (Blackhole) | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:146` ## ROCm build (AMD GPUs) — community-verified W0, blind F6 fix | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:192` ## Nix shells | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:216` ## CMake options | Generic user procedure | `docs/BUILD.md` | `kept` | `n/a` |
| `docs/BUILD.md:243` ## Backend and hardware state | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `docs/BUILD.md:267` ## Quantization formats | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `docs/BUILD.md:277` ## Environment variables | Dense lookup material | `docs/ENVIRONMENT.md` | `deduplicated` | `n/a` |
| `docs/ENVIRONMENT.md:1` # Environment variables | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:14` ## Six of these are also config keys, and the environment wins | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:59` ## Deployment knobs | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:86` ## GGUF loading | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:102` ## MoE expert streaming (CPU) | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:133` ## Rollback and bisect switches | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:235` ## Diagnostic | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:268` ## ROCm + Gemma-4 residency (contributor #140) | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:296` ## Kernel-internal knobs (deferred) | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/ENVIRONMENT.md:308` ## Keeping this reference honest | Dense lookup material | `docs/ENVIRONMENT.md` | `kept` | `n/a` |
| `docs/FEATURES.md:1` # Features <!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished --> | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:18` ## At a glance | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:32` ## Serving and scheduling | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:48` ## KV cache and memory | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:66` ## Quantization and weight formats | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:93` ## Model coverage | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:114` ### Registered architectures | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:156` ### Standalone and non-registered lanes | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:193` ### Inventoried but blocked | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:213` ## Multimodal | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:234` ## Speculative decoding | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:248` ## Structured output and tool calling | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:262` ## Backends and hardware | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:294` ## Serving, API and operations | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:334` ## Parallelism and scale-out | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:352` ## Not supported yet | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/FEATURES.md:373` ## How to read this page | Current feature state | `docs/FEATURES.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:1` # KV offload and external KV caches | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:15` ## The flag | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:37` ## LMCache (`lm://` remote KV), the connector that works end to end | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:87` ### Identity and refusal | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:104` ## CPU + disk offload (`OffloadingConnector`), REFUSED by the engine today | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:136` ## Limitations | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/KV-OFFLOAD.md:165` ## Consuming it programmatically | Cross-model workflow | `docs/KV-OFFLOAD.md` | `kept` | `n/a` |
| `docs/RELEASES.md:1` # Binary releases | Dense lookup material | `docs/RELEASES.md` | `kept` | `n/a` |
| `docs/RELEASES.md:12` ## Primary downloads | Dense lookup material | `docs/RELEASES.md` | `kept` | `n/a` |
| `docs/RELEASES.md:35` ## Verify a download | Dense lookup material | `docs/RELEASES.md` | `kept` | `n/a` |
| `docs/RELEASES.md:52` ## Retention | Dense lookup material | `docs/RELEASES.md` | `kept` | `n/a` |
| `docs/RELEASES.md:59` ## Maintainer flow | Contributor procedure | `.agents/specs/release-binary-matrix.md` | `archived` | `.agents/specs/release-binary-matrix.md` |
| `docs/ROCM.md:1` # ROCm (AMD GPU) backend — contributor guide | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:31` ## 1. Why ROCm is the cheapest backend to add | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:59` ## 2. What a backend is, file by file | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:100` ## 3. Correctness before kernels: the reference tier | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:124` ### 3.1 The F6 fix: unified memory true by construction (approach (b)) | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:171` ## 4. Pick your first task from your hardware | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:184` ## 5. Milestones as concrete PRs | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:237` ### 5.1 Known runtime issues on the #41 boards | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:254` ### 5.2 The sequence for board owners, post-F6-fix | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:320` ## 6. What not to port | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:335` ## 7. Working with the record | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:354` ## 8. CI gates your PR will hit | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/ROCM.md:376` ## 9. Asking | Cross-model workflow | `docs/ROCM.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:1` # SGLang-compatible behaviors | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:25` ## 1. RadixAttention (automatic prefix caching) | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:73` ## 2. LPM cache-aware scheduling | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:120` ## 3. Jump-forward decoding | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:172` ## 4. Custom logits processors | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:208` ## When to enable (guidance) | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SGLANG-COMPAT.md:236` ## Default inertness | Cross-model workflow | `docs/SGLANG-COMPAT.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:1` # Speculative decoding | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:11` ## Methods | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:27` ## Which keys the JSON accepts | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:60` ## MTP | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:98` ## DFlash (block diffusion) | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:114` ## n-gram | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:128` ## DSpark (semi-autoregressive block drafting) — in progress | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:175` ### Which DSpark draft the loader will take | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:202` ## The flag | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:237` ## Measured result | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:257` ## Concurrency above 1 | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:267` ## Limitations | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/SPECULATIVE-DECODING.md:304` ## Consuming it programmatically | Cross-model workflow | `docs/SPECULATIVE-DECODING.md` | `kept` | `n/a` |
| `docs/STATUS.md:1` # vllm.cpp status | Current feature state | `docs/STATUS.md` | `kept` | `n/a` |
| `docs/STATUS.md:20` ## Parity pin | Current feature state | `docs/STATUS.md` | `kept` | `n/a` |
| `docs/STATUS.md:60` ## Capability status | Current feature state | `docs/STATUS.md` | `kept` | `n/a` |
| `docs/STATUS.md:352` ## Speculative decoding | Current feature state | `docs/FEATURES.md` for the keyed support summary; `docs/SPECULATIVE-DECODING.md` for methods, configuration, and limitations | `deduplicated` | `n/a` |
| `docs/STATUS.md:663` ## Not supported yet | Current feature state | `docs/STATUS.md` | `kept` | `n/a` |
| `docs/STATUS.md:846` ## Model family notes | Current feature state | `docs/FEATURES.md` model-coverage tables | `deduplicated` | `n/a` |
| `docs/STATUS.md:848` ### Gemma | Current feature state | `docs/FEATURES.md` registered-architecture table | `deduplicated` | `n/a` |
| `docs/STATUS.md:932` ### OLMo | Current feature state | `docs/FEATURES.md` registered-architecture table | `deduplicated` | `n/a` |
| `docs/STATUS.md:948` ### Frontier and hardware-blocked families | Current feature state | `docs/FEATURES.md` registered, standalone, and inventoried-but-blocked tables | `deduplicated` | `n/a` |
| `docs/STATUS.md:1720` ### Recent dense batch | Current family state, checkpoint gates, and implementation history | `docs/FEATURES.md` registered-architecture table for current family state; `.agents/specs/sweep-recent-dense-batch.md` for the named family, checkpoint, oracle, and gate record; `.agents/benchmark-record.md` for retained gate evidence | `deduplicated`; `archived` | `.agents/specs/sweep-recent-dense-batch.md`; `.agents/benchmark-record.md` |
| `docs/STATUS.md:1786` ## Build and test lanes | Build procedure and contributor verification detail | `docs/BUILD.md` for public build instructions; `.agents/verification.md` for the contributor gate procedure | `deduplicated`; `archived` | `.agents/verification.md` |
| `docs/STATUS.md:1879` ## Performance detail | Measurement | `docs/BENCHMARKS.md` | `deduplicated` | `n/a` |
| `docs/STATUS.md:1946` ## Backend detail | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `docs/STATUS.md:2225` ## Serving and API notes | Current feature state | `docs/FEATURES.md` | `deduplicated` | `n/a` |
| `docs/STATUS.md:2619` ## Verification and parity | Current verification procedure, checker and migration history, gate evidence, environment-document repair, and README history | `docs/STATUS.md` for the current parity pin and verification policy; `.agents/verification.md` for the active generic procedure; `scripts/check-fusion-consistency.py` and `scripts/check-runner-routing-consistency.py` for current checker behavior; `.agents/specs/arch-fusion-fold-plan-2026-07-30.md` for the Gemma fusion migration; `.agents/benchmark-record.md` for the runner/fusion gates, Gemma and Qwen gate evidence, environment-document repair, and README restructure record | `deduplicated`; `archived` | `.agents/verification.md`; `.agents/specs/arch-fusion-fold-plan-2026-07-30.md`; `.agents/benchmark-record.md` |
| `docs/STATUS.md:2681` ### Feature-gap map vs pinned vLLM 0.26 (2026-07-28) | Design or implementation history | `.agents/upstream-sync.md` | `archived` | `.agents/upstream-sync.md` |
| `docs/STATUS.md:2752` ## Upstream sync 2026-07-30 (`555967922..e04a30a77`, 198 commits) | Design or implementation history | `.agents/specs/upstream-sync-2026-07-30.md` for the ranked queue and decisions; `.agents/sync/2026-07-30-e04a30a.md` for the dated sync report; `.agents/upstream-sync.md` for the current pin and sync procedure | `archived` | `.agents/specs/upstream-sync-2026-07-30.md`; `.agents/sync/2026-07-30-e04a30a.md`; `.agents/upstream-sync.md` |
| `docs/USAGE.md:1` # Using vllm.cpp | Generic user procedure | `docs/USAGE.md` | `kept` | `n/a` |
| `docs/USAGE.md:9` ## Building | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:25` ### Host compilers | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:41` ### Setting the compiled build identity | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:57` ### One ROCm-specific behaviour | Build and backend reference; implementation history | `docs/BUILD.md` for the current HIP optimization rule; `docs/reference/server.md` for the current sampler and `max_tokens` behavior; `.agents/specs/rocm-backend-w0.md` for the hostcall investigation | `moved`; `archived` | `.agents/specs/rocm-backend-w0.md` |
| `docs/USAGE.md:72` ### ROCm op coverage is incremental (and throws are by design) | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:104` ### CUTLASS is fetched as headers only | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:118` ## Confirming which CUDA architecture a build targets | Build procedure and implementation history | `docs/BUILD.md` for the current cache and feature-table procedure; `.agents/specs/cuda-architecture-inventory.md` for the superseded cache-report investigation | `moved`; `archived` | `.agents/specs/cuda-architecture-inventory.md` |
| `docs/USAGE.md:142` ### FlashAttention-2 is used only where the build compiled it | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:168` ### A DISABLED feature removes its kernels, not the ops that do not need it | Generic user procedure | `docs/BUILD.md` | `deduplicated` | `n/a` |
| `docs/USAGE.md:193` ## Using more than one engine in a process | Dense engine-lifecycle reference and implementation history | `docs/reference/engine-lifecycle.md` for current multi-engine, multi-backend, and pool-debug behavior; `.agents/specs/pool-device-key.md` for the device-key and scratch-pool investigation | `moved`; `archived` | `.agents/specs/pool-device-key.md` |
| `docs/USAGE.md:233` ## Starting an agent-assisted contribution | Contributor procedure | `.agents/porting.md` | `archived` | `.agents/porting.md` |
| `docs/USAGE.md:261` ### `.env`: your values, and what happens when it is missing | Contributor procedure | `.agents/porting.md` | `archived` | `.agents/porting.md` |
| `docs/USAGE.md:295` ### `GPU_LOCK`: one file mutex, and only one | Contributor procedure | `.agents/porting.md` | `archived` | `.agents/porting.md` |
| `docs/USAGE.md:325` ## Running inference (CLI) | Generic user procedure, model recipe, and implementation history | `docs/USAGE.md` for the runnable CLI path and common flags; `docs/reference/model-loading.md` for shared loader rules; `docs/models/qwen3-8-27b.md` and `docs/models/qwen3-8-2-4t.md` for Qwen checkpoint-specific state; `.agents/specs/qwen38-text-only.md` and `.agents/specs/qwen35-plain-bf16-direct-load.md` for load-plan and gate history | `kept`; `moved`; `archived` | `.agents/specs/qwen38-text-only.md`; `.agents/specs/qwen35-plain-bf16-direct-load.md` |
| `docs/USAGE.md:445` ### Which HF tokenizers load | Dense lookup material | `docs/reference/model-loading.md` | `moved` | `n/a` |
| `docs/USAGE.md:474` ### Timing an encode on your own box | Dense lookup material | `docs/reference/model-loading.md` | `moved` | `n/a` |
| `docs/USAGE.md:497` ### How much memory a Vulkan load needs | Build procedure, current feature state, measurement, cross-model workflow, and design or implementation history | `docs/reference/model-loading.md` for Vulkan unified-memory sizing and `VT_VULKAN_ALLOC_STATS`; `docs/BUILD.md` for the Vulkan measurement tools and Tenstorrent build requirements; `docs/STATUS.md` for current Tenstorrent backend state; `docs/BENCHMARKS.md` for the Voxtral/Whisper FlashAttention-2 and vocoder A/B measurements; `docs/guides/vocoder-device.md` for the cross-model vocoder A/B workflow; `.agents/specs/backend-fanout-metal-vulkan-xpu.md`, `.agents/specs/tenstorrent-backend.md`, `.agents/specs/multimodal-speed.md`, and `.agents/specs/minimax-music3.md` for implementation, gate, and raw-measurement history | `moved`; `deduplicated`; `archived` | `.agents/specs/backend-fanout-metal-vulkan-xpu.md`; `.agents/specs/tenstorrent-backend.md`; `.agents/specs/multimodal-speed.md`; `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:557` ### Running the vocoder convolutions on the GPU | Cross-model workflow and design or implementation history | `docs/guides/vocoder-device.md` for device selection, defaults, supported device names, refusals, and the A/B workflow; `.agents/specs/minimax-music3.md` for byte-identity gates, rollout decisions, and implementation history | `moved`; `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:582` ### Quantized checkpoints: which weight forms load | Stale empty heading | No body to migrate: campaign-base line 582 is followed immediately by the line 583 heading, which owns the load-stat content beginning at line 585 | `stale` | Git anchor `c8d926ea82bd6d8f5d6312693572c84234a6a7f3:docs/USAGE.md:582-583` |
| `docs/USAGE.md:583` ### How long a load takes, and how to see where it goes | Dense lookup material, measurement, and implementation history | `docs/reference/model-loading.md` for `VT_LOAD_STATS` and counter semantics; `docs/BENCHMARKS.md` for accepted load ratios; `.agents/specs/load-direct-upload.md` for implementation and raw measurement evidence | `moved`; `deduplicated`; `archived` | `.agents/specs/load-direct-upload.md` |
| `docs/USAGE.md:620` ### Quantized checkpoints: which `lm_head` forms load | Model recipe and dense lookup material | `docs/models/qwen3-6.md` for the Qwen3.6 checkpoint forms and model-specific controls; `docs/reference/model-loading.md` for the shared quantized-weight loading rules | `moved` | `n/a` |
| `docs/USAGE.md:648` ### Block-wise FP8 runs on CPU, and its CUDA kernel is built but unverified | Model recipe and design or implementation history | `docs/models/qwen3-8-27b.md` for the pinned checkpoint, supported CPU arm, CUDA shape restrictions, and current refusals; `.agents/specs/model-fp8-block-linear.md` for implementation, mutation, and gate history | `moved`; `archived` | `.agents/specs/model-fp8-block-linear.md` |
| `docs/USAGE.md:771` ### A per-tensor scale has to be one F32 number | Dense lookup material | `docs/reference/model-loading.md` | `moved` | `n/a` |
| `docs/USAGE.md:796` ### One load refusal that is about this code, not your checkpoint | Model recipe and dense lookup material | `docs/models/qwen3-5.md` for the affected dense loader; `docs/reference/model-loading.md` for the shared diagnostic and reporting guidance | `moved` | `n/a` |
| `docs/USAGE.md:819` ### A refusal that names the attention backend, and what it cannot tell you | Dense lookup material | `docs/reference/model-loading.md` | `moved` | `n/a` |
| `docs/USAGE.md:883` ### Architectures that resolve but refuse to run | Dense lookup material | `docs/reference/model-loading.md` | `moved` | `n/a` |
| `docs/USAGE.md:909` ### LTX-2.5: what runs, and what it cannot do | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1102` ### Where the render spent its wall: `phase-log.json` | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1188` ### While the render runs: the `[render]` lines | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1415` ### The DFR pipeline: `--pipeline-kind dfr` | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1455` ### LTX-2.5 text-to-audio: a render with no picture | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1536` ### LTX-2.5 video guidance: `--pipeline-kind one_stage` | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:1786` ### GDN checkpoints: the `output_gate_type` key | Model recipe and dense lookup material | `docs/models/qwen3-5.md` and `docs/models/qwen3-next.md` for the affected model families; `docs/reference/model-loading.md` for the shared config-resolution, default, alias, and refusal rules | `moved` | `n/a` |
| `docs/USAGE.md:1810` ### Muse Glimmer: exactly what has been checked | Model recipe, measurement, and design or implementation history | `docs/models/muse-glimmer.md` for current supported paths, defaults, limitations, and unverified arms; `docs/BENCHMARKS.md` for the reduced-depth verification result and explicitly absent performance denominator; `.agents/specs/muse-glimmer.md` for gate, issue, and implementation history | `moved`; `deduplicated`; `archived` | `.agents/specs/muse-glimmer.md` |
| `docs/USAGE.md:1852` ## OpenAI-compatible server | Generic user procedure | `docs/USAGE.md` | `kept` | `n/a` |
| `docs/USAGE.md:1975` ### Selecting an x86 CPU ISA tier | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:1993` ### NVFP4 dense sinks | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:2032` ### The NVFP4 output head | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:2060` ### Validating a staged release archive | Release procedure and design or implementation history | `docs/RELEASES.md` for the staged-package validation command, required sidecars, and current checks; `.agents/specs/release-binary-matrix.md` for release-package gate design and implementation history | `moved`; `archived` | `.agents/specs/release-binary-matrix.md` |
| `docs/USAGE.md:2104` ## HuggingFace cache and credentials | Cross-model workflow | `docs/guides/hugging-face-access.md` | `moved` | `n/a` |
| `docs/USAGE.md:2170` ## Container images | Cross-model workflow | `docs/guides/container-images.md` | `moved` | `n/a` |
| `docs/USAGE.md:2208` ### Picking the right flags for your GPU | Cross-model workflow | `docs/guides/container-images.md` | `moved` | `n/a` |
| `docs/USAGE.md:2235` ### If the server exits at startup | Cross-model workflow | `docs/guides/container-images.md` | `moved` | `n/a` |
| `docs/USAGE.md:2243` ### Building and validating an image locally | Cross-model workflow | `docs/guides/container-images.md` | `moved` | `n/a` |
| `docs/USAGE.md:2300` ### Endpoints | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:2412` ### Speech and music generation | Generic user procedure, model recipe, measurement, and design or implementation history | `docs/USAGE.md` for the generic speech endpoint and request shape; `docs/models/minimax-music3.md` and `docs/models/indextts-2-5.md` for family-specific commands, fields, devices, limits, and refusals; `docs/BENCHMARKS.md` for accepted performance values; `.agents/specs/minimax-music3.md` for stage gates, experiments, and implementation history | `kept`; `moved`; `deduplicated`; `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:2872` ### `max_tokens`: what a non-positive value means | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:2886` ### Which token ids stop a generation | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:2908` ### Server flags | Dense lookup material | `docs/reference/server.md` | `moved` | `n/a` |
| `docs/USAGE.md:3044` ## DSpark drafts: the exact checkpoints | Model recipe | `docs/models/dspark.md` | `moved` | `n/a` |
| `docs/USAGE.md:3089` ## Muse Glimmer 30B from a GGUF k-quant | Model recipe | `docs/models/muse-glimmer.md` | `moved` | `n/a` |
| `docs/USAGE.md:3142` ## Nemotron-3.5-Lightning-30B: the exact weights, and which arms run | Model recipe | `docs/models/nemotron-3-5-lightning.md` | `moved` | `n/a` |
| `docs/USAGE.md:3159` ### The checkpoint | Model recipe | `docs/models/nemotron-3-5-lightning.md` | `moved` | `n/a` |
| `docs/USAGE.md:3186` ### The arms, and what each one costs you today | Model recipe | `docs/models/nemotron-3-5-lightning.md` | `moved` | `n/a` |
| `docs/USAGE.md:3208` ### What has NOT been measured | Model recipe | `docs/models/nemotron-3-5-lightning.md` | `moved` | `n/a` |
| `docs/USAGE.md:3216` ## MiniMax-H3: video + audio generation | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3218` ### The exact weights (so a render is reproducible) | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3245` ### The PRUNED checkpoints — more precision for the same footprint | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3285` ### The trap: this checkpoint does not serve every task | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3333` ### Writing the prompt (read this first) | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3392` ### Request fields | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3424` ### Video and audio references (`metadata`) | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3470` ### The job lifecycle | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3487` ### Video family, and family-specific load knobs | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:3528` ## Consuming it as a library (C ABI) | Generic user procedure, dense lookup material, and implementation history | `docs/USAGE.md` for one runnable C ABI example; `docs/reference/c-api.md` for the ABI version and export surface; `.agents/specs/c-api-library.md` for ABI growth and packaging history | `kept`; `moved`; `archived` | `.agents/specs/c-api-library.md` |
| `docs/USAGE.md:3603` ## Consuming it from C++ | Generic user procedure, model recipe, dense lookup material, and implementation history | `docs/USAGE.md` for one runnable C++ example; `docs/reference/c-api.md` for shared sampler and interface semantics; `docs/models/ltx-2-5.md` and `docs/models/minimax-h3.md` for video-family configuration, limits, and refusals; `.agents/specs/c-api-library.md`, `.agents/specs/ltx-2-5.md`, and `.agents/specs/lora-adapter.md` for interface, LTX, and unwired-LoRA history | `kept`; `moved`; `archived` | `.agents/specs/c-api-library.md`; `.agents/specs/ltx-2-5.md`; `.agents/specs/lora-adapter.md` |
| `docs/USAGE.md:3797` ### KV-cache events, and `kv_cache_report_mode` | Dense lookup material | `docs/reference/c-api.md` | `moved` | `n/a` |
| `docs/USAGE.md:3830` ## Multimodal input (image, video, audio to text) | Cross-model workflow | `docs/guides/multimodal-input.md` | `moved` | `n/a` |
| `docs/USAGE.md:3857` ### The second GGUF file: a `clip` multimodal projector | Cross-model workflow | `docs/guides/multimodal-input.md` | `moved` | `n/a` |
| `docs/USAGE.md:3983` ### `unsloth/Qwen3.8-27B-NVFP4` — what it is, and which arm is refused | Cross-model workflow | `docs/guides/multimodal-input.md` | `moved` | `n/a` |
| `docs/USAGE.md:4043` ### Per-prompt input limits | Cross-model workflow | `docs/guides/multimodal-input.md` | `moved` | `n/a` |
| `docs/USAGE.md:4094` ## MiniMax-H3 browser console (`vllm-video-studio`) | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:4118` ## MiniMax-H3: video + audio generation | Model recipe | `docs/models/minimax-h3.md` | `moved` | `n/a` |
| `docs/USAGE.md:4202` ## LTX-2.5: reproducing the DiT parity gate | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4274` ### The Gemma-4 text tower gate, and the interpreter it needs | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4345` ### `res2s_two_stage`: the high-quality preset, and why it is a sampler | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4392` ### Audio-to-video: rendering a clip around a soundtrack you supply | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4475` ### `ti2vid_two_stage`: the plain two-stage pipeline | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4542` ### `keyframe_interpolation`: generating the motion between pinned frames | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4618` ### Retake: regenerating a time window of an existing clip | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4663` ## LTX-2.5 quantized loaders | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4745` ### The DiT is not always quantized, and the FULL model never is | Model recipe and design or implementation history | `docs/models/ltx-2-5.md` for supported DiT formats, dtype refusals, residency, and owed real-weight run; `.agents/specs/ltx25-bf16-dit.md` for the prior refusal, upstream analysis, and gate history | `moved`; `archived` | `.agents/specs/ltx25-bf16-dit.md` |
| `docs/USAGE.md:4782` ### LTX-2.5 DiT weights: which file, and how to tell them apart | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:4841` ## Streaming routed experts from disk (capacity mode) | Cross-model workflow | `docs/guides/expert-streaming.md` | `moved` | `n/a` |
| `docs/USAGE.md:4857` ### Which device can serve it | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:4953` ### The same thing as config, and which one wins | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5122` ### `--device cuda` refuses a checkpoint it cannot hold | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5200` ## Qwen3.8-2.4T-A95B `UD-Q1_0`: 370 GiB served from a 119 GiB box | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5226` ### The exact weights | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5275` ### Build and serve | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5333` ### What the load costs | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5377` ### What decode costs, and why the ceiling is where it is | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5427` ### What this does not establish | Model recipe | `docs/models/qwen3-8-2-4t.md` | `moved` | `n/a` |
| `docs/USAGE.md:5444` ## Turning CUDA graph capture off, including the break seam | Cross-model workflow | `docs/guides/cuda-graph-control.md` | `moved` | `n/a` |
| `docs/USAGE.md:5545` ## SSE keepalives on long prefill | Cross-model workflow | `docs/guides/server-sse.md` | `moved` | `n/a` |
| `docs/USAGE.md:5590` ## Gemma4 FP8 on ROCm (RDNA4) | Model recipe | `docs/models/gemma-4.md` | `moved` | `n/a` |
| `docs/USAGE.md:5620` ## LTX-2.5 text conditioning | Model recipe | `docs/models/ltx-2-5.md` | `moved` | `n/a` |
| `docs/USAGE.md:5676` ## MiniMax-Music3: the exact weights (so a song is reproducible) | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5686` ### The arm that loads: `diffusers`, bf16 + fp32 | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5723` ### The arm that is REFUSED: the native `.pth` layout | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5739` ### The quantized arm that IS implemented: GGUF Q4_K, one component | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5754` ### The quantized arms that are REFUSED — and they are all third-party | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5782` ## MiniMax-Music3: the checkpoint loader | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5844` ### Running its gate | Contributor procedure and design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:5874` ### MiniMax-Music3: the quantized arms | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:5960` ### IndexTTS-2.5 goldens and checkpoint manifests | Model recipe and contributor procedure | `docs/models/indextts-2-5.md` for the current server recipe and limitations; `.agents/specs/indextts-2-5.md` for golden regeneration and checkpoint manifest evidence | `moved`; `archived` | `.agents/specs/indextts-2-5.md` |
| `docs/USAGE.md:6034` ## MiniMax-Music3: the autoregressive half | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:6044` ### The token gate the spec promised does not exist | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6060` ### Running the gates | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6092` ### Two things that will bite a later phase | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6119` ## MiniMax-Music3: the acoustic half | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:6137` ### There is no token gate on this half, and that is not a gap | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6145` ### Running the gates | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6185` ### Three things that will bite a later phase | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6214` ## MiniMax-Music3: the language model, and the end-to-end path | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:6221` ### The `inputs_embeds` entry the dense path did not have | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:6248` ### `num_condition_layers: 8` does not mean eight transformer layers | Model recipe | `docs/models/minimax-music3.md` | `moved` | `n/a` |
| `docs/USAGE.md:6256` ### Running the gates | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/USAGE.md:6284` ### Why no gate compares a generated song to the oracle's | Design or implementation history | `.agents/specs/minimax-music3.md` | `archived` | `.agents/specs/minimax-music3.md` |
| `docs/WEIGHT-OFFLOAD.md:1` # Weight offload | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:17` ## What works today | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:39` ## The flag | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:50` ### Fields | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:62` ### Targeting by name segment | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:79` ### The two empty-set defaults are not the same | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:86` ## What the engine refuses, and what it only warns about | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:99` ### The model has to claim support, and none does yet | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:129` ## Why the budget can overshoot by one weight | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:136` ## Limitations | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |
| `docs/WEIGHT-OFFLOAD.md:148` ## Consuming it programmatically | Cross-model workflow | `docs/WEIGHT-OFFLOAD.md` | `kept` | `n/a` |

Campaign-base heading count: `275`.

### W1 `USAGE.md` semantic audit

The second W1 repair reviewed all 14 `docs/USAGE.md` rows that the first
manifest marked `kept`. A row satisfies the `USAGE.md` contract when it teaches
a common CLI, server, C ABI, or C++ path. A short generic entry point can stay.
The same manifest row must map its other content to named destinations.

The audit retained six rows in whole or in part. They cover the introduction,
the generic CLI, server, and speech paths, and one C ABI and C++ example. It
corrected the four reviewed findings at lines 648, 4745, 5275, and 5844. It also
corrected eight other misplaced or mixed sections. These sections cover builds,
engine lifecycle, load statistics, speech, and dense CLI or library details.
Every named archive owner exists at W1. Every new public destination uses the
approved public-document architecture.

The exhaustive review checked all 120 `docs/USAGE.md` manifest rows. It found
nine incorrect destinations across model, backend, release, container, and
vocoder sections. The mixed Vulkan-load section now names a destination for
each topic and its internal evidence. Model-specific loader facts go to their
Qwen or Muse pages. Shared loader rules go to
`docs/reference/model-loading.md`. Container runtime guidance goes to the
container guide. Release-package validation goes to `docs/RELEASES.md`. No
reviewed section routes to the unrelated Nemotron, Hugging Face access, or
server-reference pages. The apparent quantized-checkpoint section at campaign-
base line 582 is an empty heading immediately followed by the line 583 heading;
the manifest removes that stale heading and assigns no nonexistent body.

The STATUS review also split three broad archival routes. The recent-dense row
now retains its current family state and names its existing sweep spec and gate
evidence. Verification keeps only the current summary and procedure in public,
while its checker, Gemma migration, gate, environment-document, and README
history route to their exact existing owners. The dated 2026-07-30 sync routes
to its named ranked-queue spec and sync report as well as the current sync
procedure.


### Duplicate sections

`docs/USAGE.md` contains two H2 sections named `MiniMax-H3: video + audio
generation`. Both move to `docs/models/minimax-h3.md`. W3 combines their user
procedures and removes repeated facts. Five MiniMax-Music3 H2 sections move to
`docs/models/minimax-music3.md`. Their gate narratives and later-phase notes
already belong to `.agents/specs/minimax-music3.md`. W3 archives those parts
and keeps the public checkpoint, command, limitation, and refusal facts.

### W1 validation commands

Run these commands from the campaign worktree. Each command exits nonzero when
its stated condition fails.

Resolve relative Markdown file links:

```sh
python3 -c 'import pathlib,re,sys,urllib.parse; fs=[pathlib.Path("README.md"),*pathlib.Path("docs").rglob("*.md")]; bad=[f"{p}: {t}" for p in fs for t in re.findall(r"(?<!!)\[[^]]*\]\(([^ )]+)",p.read_text()) if not re.match(r"(?:[a-z]+:|#|<|\{)",t) and not (p.parent/urllib.parse.unquote(t.partition("#")[0])).exists()]; print("\n".join(bad)); sys.exit(bool(bad))'
```

Require one H1 and correctly nested headings on each page under `docs/`. This
command ignores fenced code. `README.md` keeps its existing HTML title:

```sh
python3 -c 'import pathlib,re,sys; bad=[]; fs=list(pathlib.Path("docs").rglob("*.md")); exec("for p in fs:\n fence=False; levels=[]\n for s in p.read_text().splitlines():\n  if s.startswith((\"```\",\"~~~\")): fence=not fence; continue\n  m=None if fence else re.match(r\"^(#{1,6}) \",s)\n  if m: levels.append(len(m.group(1)))\n if sum(x==1 for x in levels)!=1 or any(b>a+1 for a,b in zip(levels,levels[1:])): bad.append(str(p))"); print("\n".join(bad)); sys.exit(bool(bad))'
```

Require each extracted page in exactly one index:

```sh
python3 -c 'import pathlib,re,sys,collections; roots=[pathlib.Path("docs/models"),pathlib.Path("docs/guides"),pathlib.Path("docs/reference")]; pages=[p for r in roots if r.exists() for p in r.glob("*.md")]; indexes=[pathlib.Path("docs/USAGE.md"),*[r/"README.md" for r in roots if (r/"README.md").exists()]]; links=collections.Counter((i.parent/pathlib.Path(t.partition("#")[0])).resolve() for i in indexes for t in re.findall(r"(?<!!)\[[^]]*\]\(([^ )]+\.md(?:#[^ )]+)?)\)",i.read_text())); bad=[f"{p}: {links[p.resolve()]}" for p in pages if links[p.resolve()]!=1]; print("\n".join(bad)); sys.exit(bool(bad))'
```

Require seven nonempty fields in each checkpoint registry row:

```sh
python3 -c 'import pathlib,sys; s=pathlib.Path("docs/USAGE.md").read_text(); a=s.find("<!-- checkpoint-registry:begin -->"); b=s.find("<!-- checkpoint-registry:end -->"); rows=[] if a<0 or b<a else [x for x in s[a:b].splitlines() if x.startswith("|")][2:]; bad=[x for x in rows if len(x.split("|"))-2!=7 or any(not y.strip() for y in x.split("|")[1:-1])]; print("registry missing" if a<0 or b<a else "\n".join(bad)); sys.exit(a<0 or b<a or not rows or bool(bad))'
```

Validate the destination-aware preservation record. This check validates the
manifest schema. It does not compare global literal counts. Extraction,
deduplication, archival, and deliberate rewording can correctly reduce a
literal's occurrence count. Every row must name an allowed disposition. Every
current public owner must exist. Every archived or stale fact must name
existing evidence or an issue or Git anchor. The semantic audits above record
the identifier, measurement, warning, default, and refusal reviews across
those owners.

```sh
python3 - <<'PY'
import pathlib
import re
import sys

spec = pathlib.Path(".agents/specs/public-docs-readability.md").read_text()
header = "| Source section | Class | Exact destination | Disposition | Evidence or owner |"
start = spec.find(header)
end = spec.find("\n\nCampaign-base heading count:", start)
errors = []
if start < 0 or end < 0:
    errors.append("migration manifest is missing")
    rows = []
else:
    rows = [line for line in spec[start:end].splitlines()[2:] if line.startswith("|")]

allowed = {"kept", "moved", "deduplicated", "archived", "stale"}
public_roots = ("README.md", "docs/")
internal_roots = (".agents/", "docs/bench-evidence/")
for number, row in enumerate(rows, 1):
    cells = [cell.strip() for cell in row.split("|")[1:-1]]
    if len(cells) != 5:
        errors.append(f"row {number}: expected five fields")
        continue
    source, section_class, destination, disposition, evidence = cells
    if not re.fullmatch(r"`(?:README\.md|docs/[^`]+\.md):\d+` #{1,3} .+", source):
        errors.append(f"row {number}: invalid source key")
    if not section_class:
        errors.append(f"row {number}: missing class")
    dispositions = set(re.findall(r"`([^`]+)`", disposition))
    if not dispositions or not dispositions <= allowed:
        errors.append(f"row {number}: invalid disposition {disposition}")
    destinations = re.findall(r"`((?:README\.md|docs/[^`]+\.md))`", destination)
    if dispositions & {"kept", "moved", "deduplicated"} and not destinations:
        errors.append(f"row {number}: public disposition has no public destination")
    for path in destinations:
        if not path.startswith(public_roots) or not pathlib.Path(path).is_file():
            errors.append(f"row {number}: missing public destination {path}")
    if dispositions & {"archived", "stale"}:
        owners = re.findall(r"`([^`]+)`", evidence)
        anchored = "#" in evidence or "commit" in evidence.lower() or "git" in evidence.lower()
        if evidence == "`n/a`" or not owners:
            errors.append(f"row {number}: archived or stale fact has no owner")
        for owner in owners:
            if owner.startswith(internal_roots) and not pathlib.Path(owner).is_file():
                errors.append(f"row {number}: missing evidence owner {owner}")
            elif not owner.startswith(internal_roots) and not anchored:
                errors.append(f"row {number}: unanchored evidence owner {owner}")

if len(rows) != 275:
    errors.append(f"manifest row count is {len(rows)}, expected 275")
print("\n".join(errors))
sys.exit(bool(errors))
PY
```

Require one manifest row for each campaign-base H1, H2, and H3 heading. The
command ignores fenced code and reports missing or extra source keys:

```sh
python3 -c 'import pathlib,re,subprocess,sys; spec=pathlib.Path(".agents/specs/public-docs-readability.md").read_text(); fs=["README.md",*[str(p) for p in sorted(pathlib.Path("docs").glob("*.md"))]]; expected=set(); exec("for p in fs:\n s=subprocess.run([\"git\",\"show\",f\"2dab92076:{p}\"],text=True,capture_output=True,check=True).stdout; fence=False\n for n,line in enumerate(s.splitlines(),1):\n  if line.startswith((\"```\",\"~~~\")): fence=not fence; continue\n  if not fence and re.match(r\"^#{1,3} \",line): expected.add(f\"{p}:{n}\")"); actual=set(re.findall(r"^\| `([^`]+:\d+)` #{1,3} ",spec,re.M)); bad=sorted((expected-actual)|(actual-expected)); print("\n".join(bad)); sys.exit(bool(bad))'
```

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

W1 through W8 produced the public information architecture defined by this
spec. At campaign base `c8d926ea82bd6d8f5d6312693572c84234a6a7f3`, `docs/`
contained 61 Markdown files and 25,322 lines. The integrated W8 head contains 88
Markdown files and 19,290 lines. The 27 added files are 13 model recipes plus an
index, seven task guides plus an index, and four references plus an index.

The two pages that had become chronological logs changed most:

| Surface | Base lines | W8 lines | Result |
|---|---:|---:|---|
| `docs/USAGE.md` | 6,303 | 161 | Generic workflows, indexes, and checkpoint registry |
| `docs/STATUS.md` | 2,906 | 97 | Current lifecycle and capability projection |
| `docs/BUILD.md` | 280 | 251 | Build procedures without feature or environment duplication |
| `docs/ENVIRONMENT.md` | 316 | 309 | One keyed environment-variable reference |
| `docs/FEATURES.md` | 392 | 384 | Current feature projection without claim-triage history |

Seven new task guides and four references were extracted. Five existing focused
guides remain at their top-level paths: `KV-OFFLOAD.md`, `ROCM.md`,
`SGLANG-COMPAT.md`, `SPECULATIVE-DECODING.md`, and `WEIGHT-OFFLOAD.md`. Moving
those files was rejected because their names already state one purpose and a
move would create link churn or routing pages without making the task easier to
find. A generated documentation site was also rejected because the problem was
content ownership, not rendering. Claim-changing README edits were rejected in
W6 because the README checkpoint requires an honest landing-source change; this
documentation-only campaign had none.

The structure defaults follow the public-document contract: common commands
stay in `USAGE.md`; model-specific weights, commands, limits, and refusals live
in model recipes; cross-model procedures live in guides; dense values live in
references; lifecycle and feature facts remain separate projections. The
seven-field checkpoint table remains in `USAGE.md` because `AGENTS.md` requires
that literal public surface. Lowercase kebab-case names and one index per
directory make destinations predictable without adding a publishing toolchain.

Issue outcomes at W8 are:

- #1463 owns this campaign and closes only after the pull request reaches
  `main`.
- #1275 is fixed: the feature projection states that block-wise FP8 runs on CPU
  and identifies the unproven CUDA arm.
- #342 is partially reconciled. Purpose-specific docs use the current server
  executable, ABI v23, conditional routes, and 40-entry architecture registry;
  stale README claims remain for a landing-source change.
- #704 is partially reconciled. `FEATURES.md` reports 38 parser families, while
  the README's 36-family and 40-name claims remain for the same reason.
- #1281 is not fixed. It depends on #1280 and requires three executed model
  rows. No model-run evidence exists in this documentation-only campaign, so an
  unverified quick-start page was rejected.

R1 through R5 passed after fresh review and repair waves. Negative mutations
proved a broken index link, a missing checkpoint field, a missing manifest
destination, a removed release anchor, and duplicate environment rows are
detected. The branch was first integrated with main at `5870cb2bf`, then
reconciled through `origin/main` at `ebfbcb28c`. The later integration preserves
the incoming F32 vocoder implementation and keyed records, while its new
user-facing precision and device-validation facts live on the MiniMax-Music3
model page instead of expanding `USAGE.md` back into a chronological log. The
W8 implementer documentation and
record gates pass at the outcome head; R6 and the operator's independent
exact-head rerun are intentionally not claimed here. GPU, oracle,
model-download, and quiet-host measurement gates are `NOT APPLICABLE` because
no runtime behavior, parity claim, or benchmark changes.

Remaining landing steps are fresh R6 review of the immutable full head, the
operator's exact-head gate, pull request body validation, the exact-SHA push,
and the authorized squash merge. Continuous integration completion is not a
merge dependency for this documentation-only campaign.
