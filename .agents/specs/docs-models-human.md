# DOCS-MODELS-HUMAN: make the model pages readable by a reader

Issues: [#1689](https://github.com/mudler/vllm.cpp/issues/1689) and
[#1691](https://github.com/mudler/vllm.cpp/issues/1691).

## Scope

Restructure all 13 pages under `docs/models/` so that a reader who wants to run
a model meets, in this order, what the model is, what it costs, the exact
command, and what has not been measured.

In scope:

- The 13 model pages and `docs/models/README.md`.
- `docs/guides/expert-streaming.md`, which receives the mechanism content the
  Qwen3.8 2.4T page already named it the owner of.
- The two false sentences in `docs/QUICKSTART.md` about container publication,
  and the executed row that replaces its placeholder.

Out of scope:

- Any change to a measured value, a gate, a checker, or product code.
- Rewriting existing prose only to satisfy a style rule. `AGENTS.md` binds new
  prose. Where a page was rewritten, the new text follows
  `.agents/style/prose.md`, including its prohibition on em dashes. Where a page
  was edited in place, untouched paragraphs keep their existing punctuation.
- Closing [#1281](https://github.com/mudler/vllm.cpp/issues/1281). The
  executed-row obligation and the `:latest` and `--model org/repo` forms stay
  with that row.

## The defect

Three shapes recur across the 13 pages.

1. **An identical opening line.** Twelve of the thirteen opened with `Use this
   page for <model> checkpoints, commands, supported arms, and current
   limitations`. The line repeats the title and orients nobody.
2. **Defect archaeology before the command.** `qwen3-8-2-4t.md` was 593 lines
   and spent its first 100 on issues #1123, #1124, and #1299, on which residency
   knob was resident twice, and on what a mutation proved. The serve command was
   at line 434.
3. **Spec content in a public projection.** `AGENTS.md` places rationale for a
   default and the history of a fixed defect in the row spec. `gemma-4.md`
   carried raw implementation notes with no reader-facing frame at all.

None of this is a correctness defect. Every fact on those pages was measured and
correct. The defect is order and audience.

## Design

One shape for every page:

| Position | Content |
|---|---|
| Lede | What the model is, in two or three sentences |
| Warning, when one is owed | The thing that wastes the reader's time or hardware if they learn it late |
| Quick facts | Disk, memory, device, and speed, when they are measured |
| Get the weights | Repo, revision, bytes, digests |
| Run it | The exact command, container form first where one exists |
| What it costs | Measured load, decode, and memory |
| Limits | What has not been measured, stated as such |
| Where the rest lives | Links to the guide, the reference, and the spec |

A page keeps a section it does not need. `indextts-2-5.md` already matched this
shape and is unchanged.

**Deep material is relocated, never deleted.** `AGENTS.md` forbids removing
evidence to reduce context. Two relocations happened:

- The Qwen3.8 2.4T page's mechanism content moved into
  `docs/guides/expert-streaming.md`. That guide was a 17-line stub, while the
  model page's own prose already said the guide "owns the config schema, the
  precedence rule, the statistics line, the slot count warning and what each
  device can serve". The guide is now the owner it was described as.
- `gemma-4.md`'s implementation notes stayed on the page, under a heading that
  says what they are, because the spec they belong to could not be verified to
  carry them.

## Risks and decisions

**Risk: a measured fact is lost in a rewrite.** Mitigated by a string-level
retention check over the largest rewrite. See Gates.

**Risk: relocation duplicates a fact across two documents.** `AGENTS.md` requires
each fact to live in one document. The config-key table, the precedence rule, and
the device-fit bound now live only in the guide. The model page keeps the four
values its own recipe sets and links out for the rest.

**Decision: the container form is published with its evidence boundary stated.**
The `main-cpu` image was verified to parse the `vllm_cpp` document and to serve a
model end to end, but the 370 GiB run was not repeated in a container. The page
says exactly that rather than implying the recipe was container-tested.

**Decision: the executed row in `docs/QUICKSTART.md` records the mount form.**
`--model org/repo` is blocked by #1511, so a row naming a repository identifier
would not be a run. The row names what was executed and states what it does not
establish.

## Gates

| Gate | Result |
|---|---|
| `scripts/check-quickstart-recipes.py` | `PASS` |
| `scripts/check-doc-checkpoint.py` | `PASS` |
| `scripts/check-public-doc-tables.py` | `PASS` |
| `scripts/check-env-doc.py` | `PASS` |
| `scripts/agent-preflight.sh --staged` | See Evidence |
| Fact retention on `qwen3-8-2-4t.md` | `PASS`, 57 of 57 strings |
| Container claim verified by execution | `PASS` |

## Evidence

**Fact retention.** Every checkpoint revision, sha256, byte count, tensor count,
decode figure, memory figure, refusal message, code symbol, and issue link on the
pre-change `docs/models/qwen3-8-2-4t.md` was extracted and searched for across the
post-change page and the guide that received its mechanism content. 57 strings
checked, 57 found. The one initial miss was issue #1420, whose fact had survived
while its link had not. It was restored.

**Container verification, 22 August 2026**, host `mudler-ubuntu-box`, x86_64,
Docker 29.1.2, image digest
`sha256:7f88301ea282dad778748929e7aa6869d2418c8d295eef0e7900cca8310d06e5`:

```text
$ docker run --rm ghcr.io/mudler/vllm.cpp:main-cpu --version
vllm.cpp 0.0.3 c-abi=23
```

With a mounted `Qwen/Qwen3-0.6B`, `/v1/completions` returned
`" Paris. The capital of Italy is Rome. The capital of Spain is Madrid."` for
the prompt `"The capital of France is"` at 16 tokens, `temperature` 0.

With the Qwen3.8 2.4T recipe's document and a deliberately missing model path,
the same image printed:

```text
engine: weight residency (offload_config vllm_cpp): mmap=on prefault=off expert_stream=on expert_stream_slots=4000
```

`gh api /users/mudler/packages/container/vllm.cpp` reports `visibility: public`,
and `docker manifest inspect` on `:main-cpu` reports `linux/amd64` and
`linux/arm64`.

## Stop conditions

Stop and ask if a rewrite would need a fact that is not already recorded in the
tree. This change measures nothing new except the container verification above,
and it must not become the place where an unmeasured claim enters a public page.

## Owed

- [#1281](https://github.com/mudler/vllm.cpp/issues/1281) keeps the
  executed-row table, the `:latest` tag, and the `--model org/repo` container
  line. This change filled one row and corrected one false sentence. It did not
  satisfy that row's exit condition.
- [#1511](https://github.com/mudler/vllm.cpp/issues/1511) still blocks the
  repository-identifier form on every container line.
- No GPU-lane container row has been executed on this branch.

## Now

`DONE` for the 13 model pages, `docs/models/README.md`,
`docs/guides/expert-streaming.md`, and the two corrected statements in
`docs/QUICKSTART.md`.

## Outcome

`docs/models/qwen3-8-2-4t.md` went from 593 lines to 352, and
`docs/guides/expert-streaming.md` from 17 to 300, so the mechanism content grew
a home rather than being cut. `docs/models/README.md` became an index that says
what each page answers instead of a list of 13 links.

What was rejected: moving the relocated rationale into the 13 row specs. It
would have touched a spec per page for a documentation change, and the specs
could not be verified to want it. The mechanism went to the guide the page
already named, and the rest stayed on its page under an honest heading.

Why the shape has a warning slot: the two pages that most needed restructuring,
Qwen3.8 2.4T and MiniMax-H3, both buried something that costs the reader hours.
The 2.4T page buried an 11.05 s/token decode behind a 370 GiB download, and the
H3 page buried a checkpoint and task mismatch that renders a wrong picture
instead of failing. A shape with nowhere to put that information reproduces the
defect.
