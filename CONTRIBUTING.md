# Here be dragons. Welcome!

We mean that affectionately.

vllm.cpp is a from-scratch C++20 implementation of vLLM. It aims for 1:1
behavior and feature parity, including matching tokens and meeting the
performance gates, without Python or PyTorch at inference time.

It also carries useful capabilities beyond vLLM: SGLang scheduling ideas,
llama.cpp-style deployment, and text, image, video, and audio support in one
engine. Every architecture, model family, feature, and backend must be tested
against its reference and benchmarked on the same workload.

If you want to contribute, use an agent coding tool that can read repository
instructions. Point it at this checkout and give it this first instruction:

```text
Read AGENTS.md completely and follow it before doing any work. Start by running scripts/agent-preflight.sh.
```

The agent will ask what kind of work you are doing and select the operator,
helper, or read-only path. It requests machine-specific settings only when a
gate needs them. You do not need to memorize the protocol.
[`AGENTS.md`](AGENTS.md) is the canonical index, and
[the workflow](.agents/workflow.md) is the operating manual.

## Finding work

Before choosing a task, ask the agent to:

1. Search [open issues](https://github.com/mudler/vllm.cpp/issues) and
   [pull requests](https://github.com/mudler/vllm.cpp/pulls) for the topic and a
   candidate row ID.
2. Read [`.agents/NOW.md`](.agents/NOW.md) for live claims and the current gate.
3. Run `scripts/ready-for-helper.py` to list rows that meet the helper-ready
   conditions.
4. Read the relevant [roadmap](.agents/roadmap_v1.md), its owning matrix row,
   and [coordination state](.agents/coordination.md).
5. Inspect the current implementation, tests, and recorded evidence. Confirm
   that the described gap still exists at the current branch head.
6. Claim the row only after those checks show that it remains open and unowned.

An issue, roadmap row, or helper-queue result is a lead. It is not sufficient
evidence by itself. If the work has landed, is already claimed, or no longer
matches the record, the agent reconciles that state instead of duplicating it.

## Landing a pull request

Squash-merge it: `gh pr merge --squash`. The merge method is not cosmetic,
because two gates read the commits a push actually lands on `main`, and both
fail *after* the merge, on `main`, not on your PR.

`scripts/check-role-discipline.py` inspects every commit in the pushed range and
requires each to name either its `row/<ROW-ID>` branch or its PR as `(#N)`
(`POL-PR-REQUIRED`). A squash-merge writes `(#N)` into the subject for free. A
`--merge` landing does not: the content commit keeps the subject it had on the
branch, which names neither, so the gate reports `reached main without a
reviewed row/* PR` even though a reviewed row PR is exactly where it came from.

`scripts/check-commit-trailers.py` reads the same range, so the landed message
must itself carry the `FOLLOWING_AGENTS_PROTOCOL` paragraph and the trailers
(`POL-COMMIT-TRAILERS`, `POL-AI-ATTRIBUTION`). Squash bodies are built from the
branch's commit messages, so a correctly trailered commit carries them through;
confirm the composed message in the merge dialog before confirming. A `--merge`
landing fails this too, because GitHub's generated `Merge pull request #N
from ...` message has no marker and no trailers.

Neither failure is repairable afterwards. Both gates are scoped over
`github.event.before..github.sha`, and each run's `before` is the previous run's
`sha`, so no later run re-covers a range that already went red. The remedies are
rewriting published history or an explicit waiver; getting the merge method
right is much cheaper. A red `main` from this cause does not block your next PR,
which is checked against its own base, but it does hide real regressions.

## Reference and hardware guardrails

Documentation and hardware-independent work can start without a vLLM checkout
or GPU. The protocol pins the vLLM reference for gates that need it and reads
`VLLM_SOURCE`, `VLLM_ORACLE`, gate hardware, and related machine-specific
values from the repository's untracked `.env`. The agent requests missing
settings just in time, when the applicable gate needs them.

If a required oracle or machine is unavailable, that gate stays `PENDING`.
Correctness, parity, and performance work cannot be marked complete without its
applicable pinned reference, a same-workload gate, and recorded evidence.
