# Oracle: `ggml-org/llama.cpp` PR #27742, the only llama.cpp that knows `qwen4exp`

A scoped, second llama.cpp record, admitted for one narrow reason. It is the only
llama.cpp that converts or loads the `qwen4exp` architecture, and the
`MODEL-MM-QWEN4-EXP` speed gate needs a llama.cpp denominator on the identical
artifact. It does not replace the [`llama-cpp`](llama-cpp.md) oracle, it does not
outrank vLLM, and it is never a mirror source.

## Why the `llama-cpp` file cannot carry this pin

`scripts/check-oracle-pins.py` admits exactly one ` ```oracle-pin ` block per
file, so one file holds one revision. That is not an accident of the checker.
The `llama-cpp` pin is deliberately **stock upstream release** `b10451`, because
the floor that oracle supplies is *"the CPU and GGUF k-quant speed and memory
numbers a user can actually get today"*. An open pull request is not something a
user can get today. Folding a PR head into that record would quietly change what
the floor means for every measurement already taken against it.

Two llama.cpp records therefore say two different true things. `llama-cpp` says
what a release does. This file says what one unmerged branch does, for one
architecture, and nothing else.

## Scope, and what this oracle may not do

Use it ONLY as a llama.cpp denominator and reference for the `qwen4exp`
architecture: the GGUF conversion in `conversion/qwen4exp.py`, the graph in
`src/models/qwen4exp.cpp`, the architecture and hyper-parameter registration, and
the CPU and GGUF k-quant speed and memory numbers those produce on a `qwen4exp`
checkpoint.

For every other path, including CPU and GGUF k-quant floors generally, the oracle
is [`llama-cpp`](llama-cpp.md) at its own stock pin. Where vLLM or vLLM-Omni
implements the behavior, that is the reference and this is not, exactly as
`AGENTS.md` section "When vLLM has no implementation" requires.

## The pin, verified rather than relayed

Measured on 27 August 2026 from refs and objects, never from a working tree. The
developer's local llama.cpp checkout is dirty and was deliberately not read. Every
command below ran against a fresh bare repository in a scratch directory whose
only remote is `https://github.com/ggml-org/llama.cpp`.

| Claim | Command | Result |
|---|---|---|
| the PR's live head | `git ls-remote origin 'refs/pull/27742/*'` | `6c5afc86ae84448ae4d744e357017e2c490ad9c3` at `refs/pull/27742/head` |
| the pinned object still exists | `git fetch --depth 1 origin 035e2273...` | rc=0, the object is servable |
| the PR is unmerged | `git merge-base --is-ancestor 035e2273... origin/master` | **rc=1** |
| the live head is unmerged too | `git merge-base --is-ancestor 6c5afc86... origin/master` | **rc=1** |
| no released llama.cpp has it | `git grep -il qwen4exp b10451` | **rc=1**, nothing tree-wide |
| that grep is not itself broken | the same grep for `qwen3vl` at `b10451` | rc=0, three or more files |
| the converter is present at the pin | `git cat-file -e 035e2273...:conversion/qwen4exp.py` | rc=0, 11037 bytes |

`origin/master` was `d7a2074112d27649303fa107eb8c94db1ee435f3` when those ran, and
`b10451` resolved to `10bf611e533d81f739128304991c5e133c6aebd8`, which is the
`llama-cpp` pin.

The `b10451` line is the reason this record exists at all. The released tag this
project pins as its llama.cpp floor cannot name the architecture, so the floor
oracle has nothing to say about this model, and a second scoped record is the only
honest way to have a llama.cpp denominator here.

## #2060 was STALE AT BIRTH, which is why a pin is a SHA

[#2060](https://github.com/mudler/vllm.cpp/issues/2060) names `035e2273` as the
head of #27742. It was not. `035e2273` had stopped being the head almost four
hours before the issue was written. Measured from the forge rather than recalled:

| Fact | Command | Result |
|---|---|---|
| when #2060 was created | `gh issue view 2060 --json createdAt` | `2026-08-27T07:53:09Z` |
| when `6c5afc86` was committed | `gh api repos/ggml-org/llama.cpp/commits/6c5afc86 --jq .commit.committer.date` | `2026-08-27T03:58:12Z` |
| when `035e2273` was committed | `gh api repos/ggml-org/llama.cpp/commits/035e2273 --jq .commit.committer.date` | `2026-08-26T15:09:53Z` |

The branch reached `6c5afc86` **3 h 55 m before #2060 existed**, and it has not
moved since: `gh api repos/ggml-org/llama.cpp/pulls/27742 --jq .head.sha` still
returns `6c5afc86...`, open and unmerged. Nothing drifted while the issue sat.
The issue was written against a SHA that was read at some earlier moment, and by
the time it was filed that SHA was thirteen commits behind.

The 12 h 48 m that separates the last two rows above is the span between the two
COMMITS. It says nothing about how long the issue waited.

A head that was already wrong when somebody wrote it down is a stronger argument
for a recorded object id than a head that moves afterwards. A moving head is at
least visible: you fetch, you see a new object, you decide what to do. A stale
transcription is invisible, and every measurement taken against the name
`#27742` inherits the error without a signal. The pin below is a 40-character
object id for that reason.

The gap between the two objects is a clean fast-forward, measured and not
assumed: `git merge-base --is-ancestor 035e2273... 6c5afc86...` returns rc=0, and
`git log 6c5afc86...035e2273...` is empty, so nothing was force-pushed away and
the pinned object is still on the branch.

The thirteen commits, oldest first:

```
cfbdc0a llama: save and restore the qwen4exp indexer KV cache
d22d2be llama: make the qwen4exp PLE n-gram history per context and serialise it
d4a943f qwen4exp: tidy comments and simplify image token read
0ac4b18 qwen4exp: support a quantized KV cache in the QSA attention path
c52ed2a llama: give qwen4exp a large-graph node budget
6a69a0c qwen4exp: drop an unused variable that breaks -Werror builds
ef9fa1b quantize: dequantize and quantize large tensors in row bands
5674c73 llama: segment the qwen4exp fused QKV for tensor split
24ea62d llama: fix the qwen4exp PLE history seq_rm(-1) iterator invalidation and the fatal-warning build
0b19188 qwen4exp: include llama-impl.h explicitly for llama_mul_mat_hadamard
213df58 convert: fix the qwen4exp lint and type-check failures
fbe1773 llama: give the qwen4exp indexer cache its own tensor names
6c5afc8 qwen4exp: double the Q split granularity for tensor parallelism
```

Those are subjects, not diffs, and one of them overstates: `24ea62d` ends "and
the fatal-warning build" while its diff carries no such hunk. What that commit
actually changes, and why the count of `-Werror` fixes below is one rather than
two, is measured in [Why the pin stays at `035e2273`](#why-the-pin-stays-at-035e2273).

Read this section before quoting a number. A measurement taken against "the PR"
with no head SHA is not reproducible, and at thirteen commits in under thirteen
hours the name `#27742` on its own identifies nothing.

## Why the pin stays at `035e2273`

The build evidence was measured at `035e2273`. That is the whole reason. The
`git archive`, the 247 translation units, the 228 `qwen4exp` strings in
`libllama.so` and the recorded `mem_size` warning all describe that object, in
[the build evidence file](../../docs/bench-evidence/oracle-llamacpp-qwen4exp-pr27742-build-20260827.md).
A pin whose evidence was measured at another revision is not a pin. It is a
number beside a paragraph that measured something else.

Say plainly what is NOT the reason, because the next reader will want to advance
this pin and needs the real cost. Advancing to `6c5afc86` breaks no anchor in
this tree. Each row below was measured on 27 August 2026:

| Checked | Result |
|---|---|
| the three `conversion/qwen.py` anchors in [`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md) (`:365`, `:387-388`, `:438`) | they resolve at stock `b10451`, which that spec states itself, so no head of #27742 can stale them |
| `conversion/qwen.py` across the two revisions | BYTE-IDENTICAL, blob `cdba8a63e9c919232e2ec80e88b01afec7967dc4` at both |
| `conversion/qwen4exp.py:19`, the class declaration that spec reads at the PR | `class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase)` at both revisions |
| `modify_tensors` carries no `hc_norm` branch | true at both: `hc_norm` does not occur anywhere in that file at either revision |
| `fbe1773` and `5674c73`, which touch tensor naming and the QKV layout | `src/` files only, and no anchor in this tree cites `src/` |

Advancing would also GAIN one `-Werror` build fix, `6a69a0c`, which removes the
exact `mem_size` warning this pin's build evidence records. `24ea62d` is not a
second one, although its subject says "and the fatal-warning build". Measured in
a fresh bare clone of `ggml-org/llama.cpp` rather than read off that subject, its
diff at this revision touches one file, `src/llama-memory-hybrid-idx.cpp`, in two
hunks: an iterator-invalidation repair in `ple_hist_rm`, and the `n_toks` bound in
`ple_hist_state_read` tightened from the literal `64` to `LLAMA_MAX_PLE_NGRAM - 1`.
The `mem_size` line was already gone by then, three commits earlier:
`git show <rev>:src/models/qwen4exp.cpp | grep -c mem_size` returns 1 at
`6a69a0c~1` and 0 at `6a69a0c`.

So the hold is a choice to keep the pin and its evidence pointing at one object.
It is not a claim that advancing is unsafe. Advance the pin in a change that
measures the build again at the new object, and this section's reason for
`035e2273` goes away with the old evidence.

## Gateability

`gateable = no`, and [#2060](https://github.com/mudler/vllm.cpp/issues/2060) owes
the measurement.

`AGENTS.md` admits `gateable = yes` only once an oracle demonstrably BUILDS and
RUNS the model. Both halves were attempted on 27 August 2026, and the honest
result is one half:

- **Builds: yes, measured.** A CPU-only build at the pinned object succeeded from
  a fresh `git archive` of `035e2273`, never from a checkout somebody develops in,
  per the tree-assertion rule in [`llama-cpp.md`](llama-cpp.md). The recipe, the
  identity chain, and the binary hash are in
  [the build evidence file](../../docs/bench-evidence/oracle-llamacpp-qwen4exp-pr27742-build-20260827.md).
- **Runs: NOT measured, for want of an artifact.** No `qwen4exp` GGUF was complete
  on this host when the attempt ran. The seven `unsloth/Qwen3.8-Flash-Next-GGUF`
  quants were mid-download. The first of them, `UD-IQ1_S`, held 3.3 GiB of
  67.56 GiB, its first shard was a 10 MiB fragment, and no `SHA256SUMS.txt` had
  appeared. Loading a truncated shard measures a truncated file, not an oracle.

So the flag says `no`, and it keeps saying `no` until somebody records a load and a
generation. A build is not a run. `llama-model.cpp` compiling proves the
architecture is declared. It says nothing about whether the graph it builds
produces coherent text on real weights. That distinction is the whole content of
the gateability rule, and this record does not blur it to look further along than
it is.

## The fidelity fact every comparison against this oracle must carry

#27742 is **mask-only**. ggml-org/llama.cpp#27739 records, with the mechanism
named, that a sparse mask over a dense cache costs what dense attention costs
under CUDA flash attention. `flash_attn_mask_to_KV_max` scans backwards and stops
at the first tile that is not all `-inf`, so a mask that keeps any late key pays
for the whole dense prefix ahead of it.

This bears on how a benchmark is REPORTED, not on whether it runs. At long context
a decode-speed win over this denominator is partly an artifact of their mask
approach and not purely our gather's merit. The short-context and
high-concurrency comparisons are unaffected. State which side ran what, exactly as
[`llama-cpp.md`](llama-cpp.md) already requires for the `blk.64` tensors that pin
silently ignores. Otherwise the ratio measures a configuration difference and reads
as a performance one.

## Pin

```oracle-pin
id = llama-cpp-qwen4exp
role = secondary
upstream = https://github.com/ggml-org/llama.cpp
scope = the qwen4exp architecture, its GGUF conversion, its graph, and the CPU and GGUF k-quant floors on a qwen4exp checkpoint, which no released llama.cpp defines
pin = 035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
pin_label = pr-27742
pinned_on = 2026-08-27
gateable = no
evidence = #2060
```
