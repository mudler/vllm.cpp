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

## The head MOVED, and that is the point

[#2060](https://github.com/mudler/vllm.cpp/issues/2060) was written against head
`035e2273`. Thirteen hours later the live head is `6c5afc86`, thirteen commits
further on. The drift is a clean fast-forward, measured and not assumed:
`git merge-base --is-ancestor 035e2273... 6c5afc86...` returns rc=0, and
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

The pin stays at `035e2273`, for one reason that is about this tree rather than
about that branch. `035e2273` is the revision
[`../specs/qwen4-exp-flash-next.md`](../specs/qwen4-exp-flash-next.md) already read
its converter anchors at, and moving the pin under those anchors without re-reading
them is how an anchor goes stale inside its own tree. Advance it deliberately, in a
change that re-reads what depends on it. Several of the commits above touch tensor
naming and the QKV layout, which is exactly the surface those anchors cite.

Read this section before quoting a number. A measurement taken against "the PR"
with no head SHA is not reproducible, and at thirteen commits a day the name
`#27742` on its own identifies nothing.

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
