# llama.cpp — the CPU and GGUF k-quant floor

Never the mirror source: llama.cpp's structure is not vLLM's, and a behavior
difference between them is settled by vLLM. What it supplies is a **floor** —
the CPU and GGUF k-quant speed and memory numbers a user can actually get today,
which is the honest denominator on every path where vLLM's own CPU support is
not the thing being compared.

The pin is **stock upstream**, tag `b10451`. That is what the sentence above
requires. A user installs a release, so the floor is a release.

**This oracle is not gateable, and the reason is recorded rather than assumed.**
Stock upstream has never been built or run on this project's hardware.
`AGENTS.md` admits `gateable = yes` only after an oracle demonstrably builds and
runs the model, so this record reads `no` and names
[#857](https://github.com/mudler/vllm.cpp/issues/857), which owes that
measurement.

**Every recorded llama.cpp number in this tree predates this pin and was taken
against something else.** From 2026-07-22 to 2026-08-16 this record pinned
`237ad9b96`, a local-only commit on branch `localai-paged` in the developer's
checkout, 65 of our own performance commits past upstream tag `b9827`, built from
a working tree carrying 27 uncommitted entries. Six of those 65 commits change
`ggml/src/ggml-cpu/`, and `570aadd7a` emits a fused Gated Delta Net op and a
discriminated SSM_CONV decode op default-on with CPU reference implementations
that stock does not have, so the CPU arm was affected and not only the CUDA arm.
The label `b9892` was derived from `git rev-list --count`, and upstream tag
`b9892` is a real, different object at `ee445f93d`.

**The affected measurements are enumerated, with a re-take verdict each, in
[`../specs/oracle-llamacpp-repin-stock.md`](../specs/oracle-llamacpp-repin-stock.md).
That spec is the one source of truth for the set, and this file deliberately does
not restate its size.** It used to say "all nine" while that spec listed twelve,
and then thirteen. A count of one file kept inside another goes stale the moment
the first file grows, which `AGENTS.md` §Records names as the coupling to avoid,
and this is the first surface a reader of the pin reaches. The set has grown
three times, each time because a sweep instrument was widened rather than because
anything new was measured, so any number you remember is a lower bound.

That spec also enumerates the distinct llama.cpp revisions those measurements ran
against, and one of them is a fork **branch** with no commit recorded anywhere in
this tree. Read the count there rather than here, for the reason above and for a
sharper one: that spec's own `## Owed` already schedules another entry, so a
number written here would be stale against a change it can already see coming.

[#1003](https://github.com/mudler/vllm.cpp/issues/1003) owes the re-take. The
superseded measurements stay where they are, with their provenance, including
[`../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md`](../specs/cpu-llamacpp-floor-remeasure-2026-07-22.md)
and the A76 dot-product, elementwise-GEMM, GDN-orientation and threadpool specs.

**Assert the tree, not only the commit.** A pin names a commit, and a commit
cannot tell you what was built. The measurements above came from a directory
somebody develops in. Before any number is recorded against this oracle, either
build from a fresh `git archive` or a fresh clone of the pinned SHA, or assert
`git status --porcelain` empty on the source tree and record that assertion
beside the number. Record the built binary's sha256 either way.

```oracle-pin
id = llama-cpp
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = CPU and GGUF k-quant speed and memory floors, quant-matched against the same weights
pin = 10bf611e533d81f739128304991c5e133c6aebd8
pin_label = b10451
pinned_on = 2026-08-16
gateable = no
evidence = #857
```
