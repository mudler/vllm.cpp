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

**Both halves have now been attempted on the arm the gate will use, and the
build half is measured on CUDA as well as CPU.** The 27 August record below
stands as the CPU-only measurement; the 29 August rows are the GB10 arm.

| Half | 27 Aug, CPU, x86_64 dev host | 29 Aug, CUDA, `dgx:gpu0` GB10 |
|---|---|---|
| BUILDS | yes, fresh `git archive`, 247 translation units, 228 `qwen4exp` strings in `libllama.so` | **yes**, fresh `--depth 1` fetch by SHA into an empty directory, `git rev-parse HEAD` asserted equal to the pin, `git status --porcelain` **0 bytes**, `nvcc` 13.0.88, 142 `.cu.o` and **142 `sm_121a`** cubins with 142 objects scanned, 68 `qwen4exp` strings in `libllama.so` |
| RUNS | not attempted, no artifact | **yes**: `llama-server` loaded `Qwen3.8-Flash-Next-UD-IQ1_S` and answered `/v1/completions` with 64 coherent greedy tokens |

Configure prints `Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a`, and the
cubin histogram independently agrees: every one of the 142 ELF names in
`cubin.log` reads `sm_121a`, none reads `sm_121`. A recipe that asks for `121`
therefore gets the architecture-specific `sm_121a` the GB10 wants, and the
artifact says so as well as the configure log. An earlier draft of this record
transcribed the histogram as `sm_121`, which made the strongest of the two legs
say nothing. The CUDA build recipe, the binary hashes and the identity chain are in
[the ladder-arm evidence file](../../docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md).

**The artifact that blocked the run half is no longer the blocker.** On
27 August `UD-IQ1_S` held 3.3 GiB of 67.56 GiB and its shard 1 was a 10 MiB
fragment. All seven `unsloth/Qwen3.8-Flash-Next-GGUF` quants are now staged
complete, and `UD-IQ1_S` has its first LOCAL sha256, three shards for three
against the values the spec records from the Hub. The 10,946,624-byte first
shard is the split boundary and not a fragment: every one of the seven arms has
one of exactly that size, with a different hash, and each arm's shards sum to
the published total.

**A lease cannot reach the staged quants where they are staged**, and that is
measured rather than assumed. `/workspace` inside an `rc` job is
`//192.168.68.102/Data[/rc]`, the `rc/` SUBFOLDER of the share;
`/mnt/nas_share/checkpoints` is a SIBLING of `rc/`, and inside the job
`/mnt/nas_share` does not exist and `/workspace/..` is `/`. `UD-IQ1_S` was
therefore copied to `/workspace/q4exp-bench/UD-IQ1_S/`. `orin` sees neither
path.

**`gateable = yes` as of 29 August 2026.** The run half is recorded rather than
assumed. `llama-server` at this pin loaded the staged `UD-IQ1_S` GGUF on
`dgx:gpu0` with `-ngl 99`, answered `/health` after 990 s, and returned 64
tokens to `/v1/completions` that begin " Paris." for the prompt "The capital of
France is". `/props` reports `"build_info":"b1-035e227"` and
`"model_ftype":"IQ1_S - 1.5625 bpw"`, so the server that answered is provably
the binary built at the pinned object. [#2060](https://github.com/mudler/vllm.cpp/issues/2060)
is discharged.

**Gateable is not measured.** The flag says the oracle runs. It says nothing
about any number, and this run deliberately produced none worth quoting: one
prompt, one repetition, five prompt tokens, a 4,096-token context, one slot, no
clock window and no contention control. Do not promote a timing out of that
evidence file. Every llama.cpp floor in this tree is still owed by
[#1003](https://github.com/mudler/vllm.cpp/issues/1003), and the
`MODEL-MM-QWEN4-EXP` ladder itself has not run.

**A third fidelity fact, from the run.** `/props` reports
`"modalities":{"vision":false,"video":false,"audio":false}`: this denominator
serves TEXT ONLY, and `MODEL-MM-QWEN4-EXP` is a multimodal port whose scope
includes the image and video path. An arm that runs a vision tower does strictly
more work per request. State which side ran what, or the ratio measures a
configuration difference.

**The harness that will use this oracle is
`scripts/qwen4exp-llamacpp-ladder.sh`**, with
`tests/scripts/test_qwen4exp_llamacpp_ladder.py` as its proof. It walks the
published online-serving grid (c = 1, 2, 4, 8, 16, 32 at 1,024 in / 128 out,
three repetitions) and issues every timed request through the pinned
`vllm bench serve` with the flag sequence
`online_gate.build_client_command` builds — asserted against it, warmups and
per-leg corpus partitions included — so both arms are measured by one instrument.
The build and the decode proof are `scripts/qwen4exp-llamacpp-build-cuda.sh` and
`scripts/qwen4exp-llamacpp-decode-proof.sh`, in the tree rather than beside their
own output.

**One thing this pin does NOT print, and a harness must not assume.**
`llama-server` at `035e2273` reports no KV cache size: the decode proof's server
log is the complete unfiltered output and carries no `KV self size` line, no
`llama_kv_cache:` sizing line and no allocation summary, and `/props` carries no
KV bytes. The ladder therefore refuses to run until a `KV_BYTES_PER_TOKEN`
measured on a leased load is supplied
([#2261](https://github.com/mudler/vllm.cpp/issues/2261)). A guard that defaults
that term to zero is not a guard on this oracle.

### What this section said on 27 August, and why it no longer does

Recorded as history, in the past tense, because it is superseded. On 27 August
2026 this record read `gateable = no`. Both halves had been attempted and only
one had a result: the CPU-only build at `035e2273` succeeded from a fresh
`git archive`, per the tree-assertion rule in [`llama-cpp.md`](llama-cpp.md),
with its recipe and binary hash in
[the 27 August build evidence file](../../docs/bench-evidence/oracle-llamacpp-qwen4exp-pr27742-build-20260827.md);
the RUN half had no artifact to attempt, because `UD-IQ1_S` held 3.3 GiB of
67.56 GiB with a 10 MiB first shard and loading a truncated shard measures a
truncated file rather than an oracle.

That paragraph used to stand here in the present tense — "the flag says `no`,
and it keeps saying `no` until somebody records a load and a generation" — about
forty lines below a heading that already read `gateable = yes`. Somebody has
since recorded that load and that generation, on 29 August, in the rows above.
The reasoning was right and is kept: a build is not a run, and
`llama-model.cpp` compiling proves the architecture is declared rather than that
its graph produces coherent text. Only the verdict changed.

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
gateable = yes
evidence = docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md
```
