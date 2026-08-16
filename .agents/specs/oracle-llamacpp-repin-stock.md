# ORACLE-LLAMACPP-REPIN-STOCK: the llama.cpp oracle moves from our fork to stock upstream

Issue: [#1003](https://github.com/mudler/vllm.cpp/issues/1003) (the contaminated
measurements this row enumerates and does not re-take).
Also closes the record half of [#857](https://github.com/mudler/vllm.cpp/issues/857)
(the recorded pin names an object no remote carries).
Row: `ORACLE-LLAMACPP-REPIN-STOCK`.
Related: [#433](https://github.com/mudler/vllm.cpp/issues/433) (no x86_64 arm),
[#618](https://github.com/mudler/vllm.cpp/issues/618),
[#933](https://github.com/mudler/vllm.cpp/issues/933) (the sibling
`llama-cpp-unsloth` oracle, whose record quotes this pin).

## The contradiction

`.agents/oracles/llama-cpp.md` opens by defining this comparator as "the CPU and
GGUF k-quant speed and memory numbers a user can actually get today". It then
records `pin = 237ad9b96`, which is a local-only commit on branch
`localai-paged` in the developer's checkout at `/home/mudler/_git/llama.cpp`,
carrying 65 of this project's own performance commits on top of upstream tag
`b9827`. Both statements cannot be true. A user cannot get a tree that exists on
no remote, and a floor built from 65 of our own optimizations is not a floor.

The developer has decided the repin. This row executes it and records what it
invalidates.

## Measured facts

Every command below is read-only and was run against the developer's clone. That
clone has `origin` at `https://github.com/ggml-org/llama.cpp` and `fork` at
`git@github.com:mudler/llama.cpp.git`.

| Question | Command | Result |
|---|---|---|
| Is the pin on any remote? | `git branch -r --contains 237ad9b96` | empty. Only local branch `localai-paged` contains it |
| What tag is it near? | `git describe --tags 237ad9b96` | `b9827-65-g237ad9b96` |
| Where did `b9892` come from? | `git rev-list --count 237ad9b96` | `9892`, so the label is a commit count |
| Does tag `b9892` exist? | `git rev-parse b9892^{commit}` | `ee445f93d8a0a5033a46d1960e901ef5caec9a41` |
| Is the pin that tag? | `git merge-base --is-ancestor 237ad9b96 b9892` | no, and the reverse is also no |
| What do they share? | `git merge-base 237ad9b96 b9892` | `0ed235ea2c17a19fc8238668653946721ed136fd`, which is `b9827` |
| Was the tree clean? | `git status --porcelain` | 27 entries, 25 tracked files at +2279/-762 plus 2 untracked CUDA sources |

The label is worse than imprecise. Stock reached tag `b9892` exactly 65 commits
after `b9827`, and our fork reached `237ad9b96` exactly 65 commits after the same
base, so both objects report `rev-list --count` 9892. The recorded label
therefore names a real, fetchable, different tree that sits at the same depth.
A reader who checks out `b9892` to reproduce a number gets stock upstream and has
no way to notice.

That already happened inside this repository.
`docs/bench-evidence/rpi5-a76-llamacpp-20260806.md` records that `237ad9b96`
could not be obtained and substitutes "official b9892 tag" `ee445f93d`, calling
it "the binding reconstruction". `docs/bench-evidence/cpu-x86-llamacpp-20260811.md`
records "local fork `237ad9b96`, build number 9892, the recorded pin". Two
evidence files, one label, two different trees.

### The 65 commits are ours, and six of them are on the CPU path

`git log --oneline b9827..237ad9b96` lists 65 commits, 58 files, +8557/-272.
They are native NVFP4 W4A4 FP4-MMA prefill GEMM, Marlin-style W4A16 grouped MoE
prefill GEMM, default-on full-step MoE-decode CUDA graph, paged decode-graph
reuse, fused residual-add plus RMSNorm, chunked parallel-scan GDN prefill, a TTFT
prefill-first scheduler mode, and tracing.

The scope of this oracle is CPU, so the CUDA work looks irrelevant. It is not.
Six of the 65 touch `ggml/src/ggml-cpu/`.
`git diff --numstat b9827 237ad9b96 -- ggml/src/ggml-cpu/` returns two files,
`ops.cpp` at +318/-13 and `ggml-cpu.c` at +2/-1, so the insertions total 320 and
the net is +306, of which `ops.cpp` carries +305. The changed bodies are
`ggml_compute_forward_ssm_conv_f32`, `ggml_compute_forward_gated_delta_net_*`,
`ggml_compute_forward_flash_attn_ext_f16*`, and `ggml_compute_forward_scale`.
Commit `570aadd7a` states the consequence in its own body: the fused Gated Delta
Net op and the discriminated SSM_CONV decode op are emitted "DEFAULT-ON" and are
implemented "for the CUDA-family TU ... and the CPU reference ONLY".

Stock has neither. `git grep ssm_conv_update origin/master` and
`git grep gated_delta_net_inplace origin/master` both return nothing.

The CPU floor arm built the fork with `-DGGML_CUDA=OFF` and ran
`Qwen3.5-2B-UD-Q8_K_XL.gguf`, a `qwen35` dense model whose CPU graph reaches
exactly those ops. The CPU denominator therefore ran fused kernels that stock
upstream does not have.

## Scope

**In scope.**

1. `.agents/oracles/llama-cpp.md`: the pin, the label, `gateable`, the evidence
   pointer, the two factual errors in its prose, and a new clean-tree
   requirement.
2. `docs/BENCHMARKS.md`, `docs/FEATURES.md`, `docs/STATUS.md`, `docs/BUILD.md`:
   the present-tense claims that name `237ad9b96` as the reference a reader can
   build, or that state a live verdict derived from it. The last two were found
   by the sweep, not by reading the row's own scope: `docs/STATUS.md:502` restates
   the Muse Glimmer `1.023x` unmarked and `:159` the CPU-versus-llama.cpp
   position, and `docs/BUILD.md:247` tells a reader building the project that CPU
   is "at or ahead of llama.cpp on every GGUF axis".
3. `.agents/backend-matrix.md`, `.agents/feature-matrix.md`,
   `.agents/kernel-matrix.md`: the same
   present-tense claims in the authoritative records that the public documents
   project. `BACKEND-GATE-CUDA-LLAMACPP` states the pin and links to the very
   oracle file this row repins, and `BACKEND-GATE-CPU-LLAMACPP` carries the live
   "at OR ahead of llama.cpp on every axis" verdict against the fork, which
   `.agents/quantization-matrix.md` names as "the one place this gate's live
   position lives". Leaving them unmarked makes the public projection read
   SUPERSEDED while the record it projects reads closed. Marking a denominator is
   not a lifecycle change, so this is not the state move the deferral below
   excludes.
4. `.agents/issue-index.md`: append two rows, for #857 and #1003.
5. This spec, which carries the enumeration of every contaminated measurement.

**Out of scope, deliberately.**

- **Re-taking any number.** That needs dgx.casa, a Raspberry Pi 5, and the x86
  box. This row has none of them and holds no GPU-lock authority. #1003 owns it.
- **Building or running stock `b10451`.** Same reason, and it is exactly why
  `gateable` becomes `no` rather than moving across unchanged.
- **Re-anchoring the `file:line @ 237ad9b96` source citations.** They are a real
  defect, because a reviewer cannot fetch the object a citation names, but they
  are a mechanical sweep with its own risk of silently re-pointing a line that
  moved. Listed under `## Owed`, with the counts and the commands that produce
  them in §"The owed sweep, counted reproducibly".
- **Touching the developer's llama.cpp checkouts.** `/home/mudler/_git/llama.cpp`
  and `/home/mudler/_git/llama.cpp-mtp-imatrix` are read-only to this row. No
  fetch, no commit, no build.
- **`.agents/upstream-inventory.json`.** Its `pins.llamacpp` value reads
  `237ad9b96` and looks like a second transcription of the oracle pin. It is
  not. `scripts/upstream-inventory.py:206` emits it as
  `head_of(path)[:9]` over `$LLAMACPP_SOURCE`, which defaults to
  `~/_git/llama.cpp` at `scripts/upstream-inventory.py:40`, so it is a derived
  snapshot of the developer's checkout HEAD. The two values coincide today only
  because that checkout sits on branch `localai-paged`. Editing it would assert
  something false about a checkout that has not moved.

  **No gate would catch that edit, and the truth is worse than a red gate.** An
  earlier draft of this spec claimed the edit "would turn the
  `upstream-inventory --check` gate red". It would not. `--check` compares
  exactly four things against the stored snapshot: `registry.missing_count`,
  `arch_floor.supported`, `arch_floor.supported_with_no_row`, and the two
  `devices.*_uncovered` lists (`scripts/upstream-inventory.py:282-317`). It never
  reads `pins`. Mutation-disproved rather than asserted: setting
  `pins.llamacpp` to `10bf611e5` and running
  `python3 scripts/upstream-inventory.py --check` printed
  `OK: the agent record matches the upstream inventory snapshot` and returned
  `rc=0`, with `git diff --stat` confirming the one-line edit had applied. Only
  this paragraph protects the field, so a later reader who "reconciles" the two
  values meets no resistance at all. That is exactly why the coincidence is
  recorded here.

- **Any lifecycle state change.** No matrix row moves state here, so this row
  owes no `docs/STATUS.md` edit and writes no `## Now` lifecycle transition into
  another row's spec. Whether `BACKEND-GATE-CPU-LLAMACPP` should leave its closed
  state is a judgement that belongs with the re-measurement in #1003, not with a
  record edit that has measured nothing.

## The new pin

**`b10451`, commit `10bf611e533d81f739128304991c5e133c6aebd8`, dated
2026-08-16.**

It is the newest release tag present in the developer's clone under the
remote-tracking refs of `https://github.com/ggml-org/llama.cpp`, and it is
identical to `origin/master`. `git branch -r --contains 10bf611e5` returns
`origin/master` and `origin/HEAD`, which is the reachability evidence.

**The alternatives, and why they lose.**

| Candidate | Commit | Case for | Why rejected |
|---|---|---|---|
| `b10451` | `10bf611e5` | what a user installs today, which is this oracle's stated scope | selected |
| `b9892` | `ee445f93d` | stock at the same depth as the fork, and the Pi 5 evidence already built it | perpetuates the collision this row exists to remove. Two distinct objects would both be correctly called `b9892` inside one repository history, and no reader of an older record could tell which one a number came from |
| `b9827` | `0ed235ea2` | the fork's exact merge base, so a stock-versus-fork A/B isolates our own 65 commits with zero upstream drift | that is a diagnostic, not an oracle. It answers "what did our fork add", and the oracle's question is "what can a user get" |

The "minimise drift from existing evidence" argument for `b9892` or `b9827` buys
nothing here, because no llama.cpp-side number survives the repin. The
enumeration below shows every one of them is owed a re-take regardless of which
stock revision is chosen. When there is no evidence to preserve, the only
remaining criterion is the oracle's declared purpose, and that criterion selects
the newest stable release.

`b9827` keeps a job. It is recorded in #1003 as the base for the stock-versus-fork
A/B that isolates our 65 commits from upstream drift.

**Honest limit on the choice.** `b10451` is the newest tag in the local clone's
remote-tracking refs as of 2026-08-16. This row did not fetch, because the
checkout is the developer's and read-only here, so a newer tag can exist
upstream. That does not weaken the pin. The pin is a 40-hex commit, and a commit
is reproducible whether or not a newer one exists.

**Risk this introduces, stated rather than buried.** `b10451` is 624 stock
commits past `b9827` (see §"Direction of the error" for why the earlier 122 was a
shallow-clone artefact), a window in which upstream landed its own `fused_gdn`
(`git grep fused_gdn origin/master` hits `src/llama-context.cpp`,
`src/llama-cparams.h`, `src/models/delta-net-base.cpp`, `src/models/qwen35.cpp`,
and `src/models/qwen35moe.cpp`). A stock `b10451` denominator can therefore be
faster than both the fork and `b9892` on the GDN path. Repinning does not make
the re-measurement safe, it makes it honest.

**The fork is not needed to run the model.** Stock `origin/master` carries
`src/models/qwen35.cpp` and `src/models/qwen35moe.cpp`, the `qwen35` and
`qwen35moe` arch strings at `src/llama-arch.cpp:41-42`, the eight `NEXTN` tensor
entries at `src/llama-arch.cpp:525-532`, and
`common_speculative_impl_draft_mtp` at `common/speculative.cpp:1274`. Nothing in
the model path forces the fork.

## The two factual errors in the current record

Both are corrected in this row regardless of the repin, because both would
survive a pin advance and keep misleading.

1. **`pin_label = b9892` was derived from a commit count**, not from
   `git describe`. The real nearest tag is `b9827` and the pin is 65 commits past
   it. The correction is not to write `b9827-65-g237ad9b96`, it is to pin a real
   tag whose label is the tag's own name.
2. **The justification is circular.** The file reads "The pin is a local fork,
   not upstream `master`, because the CPU floor campaign builds it with a fixed
   recipe and reads its kernels. The fork is what `237ad9b96` names." That
   argues the pin from the pin. It states no property of the fork that upstream
   lacks, and the model-path check above shows there is none.

## The clean-tree requirement

Asserting a SHA is not asserting a tree. The recorded pin was measured from a
working directory carrying 27 uncommitted entries, including
`ggml/src/ggml-cpu/ops.cpp` and `src/models/qwen35.cpp`, so the binaries that
produced the recorded numbers came from a tree in no repository at all. A pin
that names a commit cannot detect this, and no gate in this project can either,
because none of them may make a network call or reach into another checkout.

The oracle record therefore carries the obligation in prose, at the point of use.
Either build from a fresh `git archive` or a fresh clone of the pinned SHA, or
assert `git status --porcelain` empty on the source tree and record that
assertion beside the number. Record the built binary's sha256 either way.
`docs/bench-evidence/rpi5-a76-llamacpp-20260806.md` already records a
`llama-bench` sha256, which is the shape the other arms owe.

## How this set was derived

**The set below is swept, not listed.** Two successive fresh reviews each found
one favourable verdict a hand enumeration had missed, the Vulkan `MET` and the
Pi 5 RSS in the first pass and Muse Glimmer in the second. A third hand pass
would have the same failure mode, so the set is derived by a query anyone can
re-run, and the query is recorded here rather than its output alone.

**Stage 1, the candidate sites.** A line in `docs/`, `.agents/` or `benchmarks/`
is a candidate when it carries a comparison token and llama.cpp is attributed to
it, where attribution means the line names llama.cpp, or a line within three
names it, or the line is a markdown table body row whose HEADER names it. The
table-header clause is the one that matters: it is what catches
`| Prefill | 223.8 tok/s | 177.3 | 1.18x | PASS |`, whose own text never says
llama.cpp. A `git grep -C` alone cannot see that row's denominator.

```sh
git grep -n -C 3 -iE 'llama\.?cpp|llama-bench' -- docs .agents benchmarks
```

is the readable approximation and misses exactly the table-row class. The whole
sweep is below. It is short on purpose, so that it lives in this spec rather than
becoming another file the tree has to keep true:

```python
import re, subprocess
LLAMA = re.compile(r"llama[.\-_]?cpp|llama-bench", re.I)
ROW, SEP = re.compile(r"^\s*\|"), re.compile(r"^\s*\|[\s:|-]+\|\s*$")
CMP = re.compile(r"[0-9]+(?:\.[0-9]+)? *[x×]\b|[0-9]+(?:\.[0-9]+)? ?%"
                 r"|\bMET\b|\bPARITY\b|\bPASS\b|\bFAIL(?:ED|S)?\b|\btie\b|\bties\b"
                 r"|\bahead\b|\bbehind\b|\bbeats\b|\bparity\b|\bwin\b|\bwins\b"
                 r"|\bless\b|\bfaster\b|\bslower\b")
for path in subprocess.run(["git", "ls-files", "docs", ".agents", "benchmarks"],
                           capture_output=True, text=True).stdout.split():
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    header, i = {}, 0
    while i < len(lines):                 # map every table body row to its header
        if ROW.match(lines[i]) and i + 1 < len(lines) and SEP.match(lines[i + 1]):
            j = i + 2
            while j < len(lines) and ROW.match(lines[j]):
                header[j], j = i, j + 1
            i = j
        else:
            i += 1
    for n, ln in enumerate(lines):
        if not CMP.search(ln):
            continue
        near = any(LLAMA.search(lines[k])
                   for k in range(max(0, n - 3), min(len(lines), n + 4)))
        if near or (n in header and LLAMA.search(lines[header[n]])):
            print(f"{path}:{n + 1}:{ln.strip()}")
```

On this commit's tree it prints **961 candidate lines**, drawn from 93 of the
651 files it scans, and it printed 943 at `bab8e1fb3`, before this branch's own
markers were added.

**Stage 2, favourable by the page's own rule.** `docs/BENCHMARKS.md:32-34`
states the convention: throughput is ours over the reference, latency is the
reference over ours, so **1.0 or higher is a win**. Piping stage 1 through

```sh
grep -E '\b[1-9][0-9]*(\.[0-9]+)? *[x×]|\bMET\b|\bPARITY\b|\bPASS\b|\btie\b|\bahead\b|\bbeats\b|\bwin\b|% less|\bparity\b'
```

leaves **719 lines** on this commit's tree, and those are adjudicated one at a
time below. Most fall out as ours-versus-vLLM, ours-versus-ours, or superseded
ledger history.

**Stage 3, the revision behind each survivor**, read from the evidence record
that produced it rather than assumed from the pin. That is what surfaced a third
and a fourth llama.cpp revision in the tree.

The over-inclusive stage 1 is the point. A filter tuned to the verdicts already
known would reproduce the enumeration it was meant to check. It is inclusive
enough to count a few lines of its own regex, which is the correct direction for
this instrument to err in.

## Every measurement the fork pin contaminated

"Contaminated" means the llama.cpp side of the comparison came from a build of
`237ad9b96`, or from a build whose tree cannot be identified, or from a stock
revision that is neither the old pin nor the new one. Our own side of each
comparison is untouched.

| # | Evidence | Recorded llama.cpp numbers | Re-take? | Notes |
|---|---|---|---|---|
| 1 | `docs/BENCHMARKS.md` GB10 20-core CPU floor | prefill 177.3 tok/s, decode 25.4 tok/s, peak memory 2.80 GiB, giving `1.18x PASS`, `0.97x` tie, `1.01x` parity | **yes** | the highest-value row. Built fresh on dgx with `-DGGML_CUDA=OFF`, running `qwen35`, so it executed the fork's fused CPU GDN and SSM_CONV ops |
| 2 | `.agents/specs/cpu-llamacpp-floor-remeasure-2026-07-22.md` | pp512 211.16, pp128 174.63, tg128 26.13, tg32 25.80, tg32 isolated 25.16 tok/s, peak RSS 2.798 GiB, and the derived `33.5x` prefill, `11.6x` decode, `2.65x` RSS | **yes** | same recipe and host. Its ours-side table, its threadpool A/B, its op-dispatch attribution profile, and its GEMM microbenchmark are ours-only and stand |
| 3 | `.agents/specs/gguf-compute-in-quant-gemm.md` G4 and G7, mirrored in `.agents/benchmark-record.md:11046` | llama.cpp pp128 180.14 +/- 2.78, tg32 25.37 +/- 0.81 tok/s, "refreshed same session" | **yes** | a second dgx build of the same fork |
| 4 | `.agents/specs/cpu-gdn-proj-orientation-2026-07-23.md` | its llama.cpp denominator, 5 reps under one `flock` | **yes** | highest risk of the set. The row is about the GDN projection path and the fork's CPU delta is precisely in `gated_delta_net` and `ssm_conv` |
| 5 | `.agents/specs/cpu-elementwise-gemm.md` | its llama.cpp comparison legs at `237ad9b96` | **yes** for the comparison legs, **no** for the E1-E4 gate | the E1-E4 gate is bit-exactness plus our own GFLOP/s, which never touched llama.cpp |
| 6 | `.agents/specs/gguf-cpu-threadpool.md` W4 | its llama.cpp context numbers | **yes** for the context, **no** for W4 itself | W4's acceptance is a same-binary A/B of our arm at 1 versus 20 threads. It is ours-only and unaffected |
| 7 | `docs/BENCHMARKS.md` Vulkan `BENCH-VK-LLAMA`, with `.agents/environment.md:435` | decode 4.36 versus 4.35 `MET`, 7 clean legs, spread 0.69% | **yes**, and it is the most fragile verdict in this table | none of the 65 commits touch `ggml/src/ggml-vulkan/`, so the Vulkan sources match `b9827`. The build still came from the same dirty working tree, so the tree is unidentified and the number is unreproducible. The `21.5x` prefill quoted in the same cell is **not** a llama.cpp number: the source JSON says "Prefill is 21.5x its pre-campaign value on the same model", a self-comparison, so it survives the repin untouched |
| 8 | `docs/bench-evidence/cpu-x86-llamacpp-20260811.md` | peak RSS 2.8281 GiB, giving `1.0022x` open gap. Its three throughput axes are already `PENDING` | **yes** | its provenance line names "local fork `237ad9b96`, build number 9892, the recorded pin". RSS is the axis least likely to move, because the fork's deltas are compute, but the binary is still unidentified |
| 9 | `docs/bench-evidence/rpi5-a76-llamacpp-20260806.md` | prefill 27.77, decode 3.91, E2E 3.77 tok/s, peak RSS 3.747 GiB, giving `0.461x`, `0.653x`, `0.758x` | **yes**, for a different reason | **not fork-contaminated.** This file measured stock tag `b9892` at `ee445f93d` and recorded the substitution. It is a stock number against a revision that is neither the old pin nor the new one, and its record wrongly presents that tag as the project pin |
| 10 | `docs/BENCHMARKS.md:29` Muse Glimmer 30B, `.agents/specs/cpu-decode-barrier-and-attn-dispatch.md:24-32`, `docs/STATUS.md:502`, record `:19231` and `:19016` | in128 prefill 13.158, decode 5.026, in512 prefill 13.292, decode 5.091 tok/s, and the earlier 12.94 / 5.08 / 9.97 / 6.41 / 13.13 / 5.00 set with peak RSS 15.74 GiB, giving the `1.023x` prefill win, `0.194x`, `0.175x`, `0.997x` and `1.92x MORE` RSS | **yes**, for the same reason as row 9 | **not fork-contaminated.** Both runs measured stock master `704485942` (`b10362-5`, 2026-08-11), recorded in the record's own recipe block at `:18996`. It is a stock number against a third revision that is neither pin. Its `1.023x` is the fifth favourable verdict on the public page and was absent from every earlier draft of this table |
| 11 | `.agents/kernel-matrix.md:162` `KERNEL-GEMM-CPU-TILED`, record `:13766-13782` | ggml no-llamafile 212.0, 214.4, 215.4, 208.1, 209.9, 159.2 GFLOP/s against our 222.1, 220.6, 216.8, 215.4, 241.7, 141.3 on six Arm shapes, giving "at parity with ggml's stock kernel and slightly ahead on four of six shapes", plus the stock-ggml column that sizes llamafile at ~1.9x f16 and ~1.2x f32 | **yes** | built from the same fork tree with `GGML_LLAMAFILE=OFF`. None of the 65 fork commits touch `llamafile/sgemm.cpp`, so the compared kernel matches `b9827`, but the tree is the same unidentified one as row 7. This verdict is load-bearing beyond its own row: it is the evidence that the Arm 16-bit deficit is an absent capability rather than a defect in `KERNEL-GEMM-CPU-ELEM` |

Row 9 is the one to read twice. It is the only arm whose author noticed the pin
was unobtainable, and the correct handling of that discovery, a recorded explicit
substitution, still produced a number attributed to a revision the registry never
pinned. The defect is the registry, not that file.

### Four llama.cpp revisions are in play, not two

An earlier draft of this row wrote the replacement universal "all but the Pi 5
arm ran `237ad9b96`, and the Pi 5 arm ran stock `b9892`". Stage 3 of the sweep
disproves it. Read each revision from the evidence that produced it:

| Revision | What it is | Binding measurements | Where the tree records it |
|---|---|---|---|
| `237ad9b96` | our local-only fork, `b9827-65-g237ad9b96` | rows 1 to 8 and row 11 | the former `pin` in `.agents/oracles/llama-cpp.md` |
| `ee445f93d` | stock tag `b9892` | row 9, the Pi 5 arm | `docs/bench-evidence/rpi5-a76-llamacpp-20260806.md`, which records the substitution |
| `704485942` | stock master, `b10362-5`, 2026-08-11 | row 10, Muse Glimmer and #391 | `.agents/benchmark-record.md:18964,18996,19221` |
| `030ebb5` | stock tag `b10358` | none. One contended `pp32 9.79 / tg8 0.79` datapoint, recorded **NON-BINDING** with our own arm blocked in the tokenizer | `.agents/benchmark-record.md:18159` |

`704485942` appeared nowhere in `docs/`, in this spec, or in the issue index
before this pass, although it is the denominator of the only llama.cpp win on
the public page besides the GB10 prefill. That is the cost of writing a universal
instead of stating what a sweep found.

### The owed sweep, counted reproducibly

Source anchors are a separate class from measurements. The ports they justify are
not invalidated, but no citation in the set is checkable, because no reviewer can
fetch the object it names. The set includes the Vulkan port map in
`.agents/specs/vulkan-full-support.md`, the Arm quant dot anchors in
`.agents/kernel-matrix.md`, `repack.cpp:4683`, `sgemm.cpp`, and the
`ggml-quants.c` IQ2_S and MXFP4 ports. Owed, not fixed here.

**The commands matter more than the numbers, because the numbers go stale inside
this pull request.** They already did once: `a2ede63a1` and `3f63066f0` measured
109 and 67, and `24e47f265` moved them to 110 and 66 in the act of recording
them, by adding a `237ad9b96` mention to `.agents/feature-matrix.md` and
rewriting the #1003 index row so it no longer matches the `@ <sha>` form. A bare
count in a spec is a measurement of one file stored inside another, which
`AGENTS.md` §Records names as the coupling to avoid. Run these instead:

```sh
git grep -l '237ad9b96' -- .                                   # mentions
git grep -lE '@ *`?237ad9b96' -- .                             # `@ <sha>` form
for f in $(git grep -l 237ad9b96 -- .); do \
  grep -C 3 -e 237ad9b96 -- "$f" \
    | grep -qE '[A-Za-z0-9_/.+-]+\.[A-Za-z0-9_]+:[0-9]+' && echo "$f"; \
done | wc -l                                                   # path.ext:LINE within 3 lines
```

They returned 109, 67 and 52 at `a2ede63a1`, 110, 66 and 52 at `bab8e1fb3`, 111,
66 and 52 once `docs/BUILD.md` gained the fork it was measured against, and
**112**, **69** and **52** on this commit's tree. The third move was not an edit
at all. Merging `origin/main` brought in `tests/vt/iq1_golden_vectors.h` and 64
new lines of `.agents/specs/expert-streaming.md`, both citing the same
unfetchable object, and the restored `.agents/issue-index.md` carries the `@`
form too. A number in this spec goes stale when somebody else's pull request
lands. Re-measure rather than quote. The second command is the subset
that attributes a path or a source tree to the object rather than merely
discussing the SHA, and it is the size of the owed sweep: 66 files, not the
"roughly 50" an earlier draft wrote, which counted only the narrowest reading and
undercounted the work by about a third.

### Direction of the error

State it, because a reader who sees "the floor must be re-taken" will otherwise
assume the change is neutral.

Against `b9827` the fork can only be at or above stock on the paths its 65
commits target, since every one of those commits is a performance change we
wrote. The recorded llama.cpp denominators are therefore at or above a stock
`b9827` denominator. Our recorded deficits are upper bounds, and our recorded
wins are lower bounds. Rows 1 through 8 all understate our position relative to a
`b9827` stock floor.

Against `b10451` the direction is **not established**, and this row does not
claim it. 624 stock commits landed after `b9827`, upstream now carries its own
`fused_gdn`, and a re-measurement can move a ratio in either direction.

**That window is 624, not the 122 an earlier draft of this section wrote, and the
122 was an instrument defect rather than arithmetic.** It came from
`git rev-list --count b9827..b10451` in the developer's clone, and that clone is
SHALLOW: `.git/shallow` grafts at `687e77892` (2026-08-08), so
`git rev-list --count b10451` returns 122 where a full clone returns 10451, and
`git merge-base --is-ancestor b9827 b10451` answers **no** for the same reason.
Every distance across that graft is truncated and every ancestry question across
it is wrong. The real distance comes from llama.cpp's own build-number
convention, which this row already established when it showed `pin_label` was a
commit count: tag `bN` sits at depth `N`, verified in-clone for `b9827` (9827)
and `b9892` (9892), so `b9827` to `b10451` is 624 and `b9892` to `b10451` is 559.
Cross-checked against a distance the graft does NOT truncate: `704485942`
describes as `b10362-5`, predicting depth 10367 and therefore 84 commits to
`b10451`, and `git rev-list --count 704485942..10bf611e5` measures exactly 84.
The prediction and the measurement agree, which is what licenses the convention.

**Six recorded verdicts can flip against us, five of them on the public page, and
the prefill `1.18x` is not the most exposed of them.** An earlier draft of this
spec said the GB10 prefill was "the only llama.cpp comparison in the tree
recorded as a win", and #1003 was scoped by that sentence, so an agent reading it
would have re-taken prefill alone. Two later drafts enumerated by hand and each
missed one more. This list is the stage-2 output of the sweep above. Ordered by
fragility, which is the margin measured against its own noise floor and against
how much of the denominator moves, not by the size of the margin:

1. **Vulkan `BENCH-VK-LLAMA` decode, 4.36 versus 4.35 tok/s, `MET`** (row 7).
   The margin is 0.23%. Its own source,
   `benchmarks/demo/vulkan_27b_llamacpp.json`, puts the 7-leg spread at 0.69% and
   states that "that spread IS the noise floor, so this is a narrow pass, not a
   comfortable one". The verdict is already inside its own measurement
   resolution, which makes it far more exposed than an 18% prefill margin. Any
   denominator movement can flip it.
2. **Muse Glimmer 30B in128 prefill `1.023x`** (row 10), on the public page at
   `docs/BENCHMARKS.md:29` and restated at `docs/STATUS.md:502`. The margin is
   2.3%. Its own source, `.agents/benchmark-record.md:19231`, puts our arm's
   median at 13.455 with a **4.5%** leg spread over n=4 against llama.cpp's
   13.158 at 0.7%, so more than half the margin sits inside our own arm's spread
   before the denominator moves at all. By this list's stated criterion that
   ranks it second, above an 18% prefill margin and a 24.2% RSS margin, and the
   honest other half is that its denominator is already stock `704485942`, 84
   commits from `b10451`. What exposes this verdict is its noise floor, not its
   denominator. It was absent from the first two enumerations of this list.
3. **GB10 20-core peak memory `1.01x` PARITY and decode `0.97x` tie** (rows 1 and
   2). Neither is a win, and both are load-bearing for
   `BACKEND-GATE-CPU-LLAMACPP` reading closed. They are ties by declaration, so a
   denominator that moves at all in llama.cpp's favour converts them into
   recorded gaps.
4. **`KERNEL-GEMM-CPU-TILED` "at parity with ggml's stock kernel, slightly ahead
   on four of six shapes"** (row 11), in `.agents/kernel-matrix.md:162`. The one
   entry not on the public page. The bands overlap (ours 216-242, ggml 208-215
   GFLOP/s) and one of the six shapes is already behind, so the verdict is split
   4-2 before anything is re-taken. It is load-bearing for the row's attribution
   that the Arm 16-bit deficit is an absent capability rather than a defect.
5. **GB10 20-core prefill `1.18x` PASS** (row 1). A pass by 0.18. Flipping it
   needs upstream's own work in the 624-commit window to exceed our fork's on the
   CPU Gated Delta Net and SSM_CONV path outright.
6. **Pi 5 peak RSS 2.841 versus 3.747 GiB, 24.2% less, `0.758x`** (row 9). The
   least fragile of the six: its denominator was already stock `b9892`, so its
   only exposure is upstream drift over the 559 commits to `b10451` rather than
   the removal of 65 of our own.

The x86 peak RSS `1.0022x` (row 8) is recorded as an open gap against us, so it
cannot flip from favourable to unfavourable. It can only widen or close.

**One discrepancy found by the sweep is left for its owning row, not repaired
here.** Row 11's evidence at `.agents/benchmark-record.md:13771-13777` shows our
kernel ahead on **five** of six Arm shapes, not the four both the record's own
prose and `.agents/kernel-matrix.md:162` state. The narrowest of the five is
216.8 against 215.4 GFLOP/s, 0.6%, which a reader may reasonably have called a
tie. Quietly changing another row's recorded count on that guess is exactly the
silent re-derivation this row exists to stop, so it is reported rather than
edited. `KERNEL-GEMM-CPU-TILED` owns it, and the re-take under #1003 settles it
with a measurement.

## Design

Record edits only, no code. Four kinds, and the sweep decides which files are in
each kind rather than a list written from memory.

1. **`.agents/oracles/llama-cpp.md`.** Rewrite the prose to state what the
   oracle is, drop the circular fork justification, add the clean-tree
   requirement, and set `pin = 10bf611e533d81f739128304991c5e133c6aebd8`,
   `pin_label = b10451`, `pinned_on = 2026-08-16`, `gateable = no`,
   `evidence = #857`.

   `gateable = no` is the point of the row. `AGENTS.md` says an oracle is
   gateable only once it demonstrably builds and runs the model, and stock
   upstream has never been built or run on this project's hardware. Carrying
   `yes` across the repin would assert a measurement nobody took. The
   `sglang` oracle sat at a stale gateability value for two and a half weeks
   (#979), which is what that failure looks like when it is not caught.

   `evidence = #857` rather than #1003, because
   `scripts/check-oracle-pins.py` requires `gateable = no` to name the issue
   that owes the measurement, and #857's acceptance already reads "build and run
   the replacement on dgx.casa". #1003 owes the re-take of the numbers, which is
   the later obligation.

2. **`docs/FEATURES.md` and `docs/BENCHMARKS.md`.** Both carry present-tense
   claims naming `237ad9b96` as a reference a reader can build.
   `docs/FEATURES.md:14` states the reference version. `docs/BENCHMARKS.md:288`
   states the denominator "built fresh on the same host", and the `BENCH-VK-LLAMA`
   result row plus the `Reproduce` table state the Vulkan denominator. Both
   Vulkan sites must be marked, not only the recipe one: the result row is the
   one carrying the `MET` verdict. Each is corrected to name the fork it actually
   was, and to point at #1003 for the re-take. The historical numbers stay. This
   is the `bench-oracle-pin-reconcile.md` distinction applied again: a
   present-tense claim about which oracle is pinned is not narrative, and a past
   run keeps saying what it ran.

   `docs/BENCHMARKS.md` also gains the fragility ranking from §"Direction of the
   error", so a reader of the page reaches the same scope #1003 has. It goes in
   as a table rather than prose because
   `scripts/check-public-doc-tables.py` holds that page at 35 of 35 prose
   paragraphs, and a table row costs none. Each row pays for its own new marker
   inside the 220-character cell cap by shortening itself, which is what the
   checker's `MAX_ROW_CHARS` comment asks for: shorten your own row, never delete
   somebody else's. The `BENCH-VK-LLAMA` row moved its `[source]` link into the
   key cell, and the Muse Glimmer row moved its `#391` link the same way. The
   page's "Oracle pin" paragraph stops asserting a universal about which revision
   every figure ran and names what the sweep found instead, in place, because a
   new paragraph would cost somebody else's.

   `docs/STATUS.md:159` and `:502` and `docs/BUILD.md:247` get the same marker.
   They were found by the sweep rather than by this row's original scope, and
   `:502` is the same Muse Glimmer verdict the page carries, so leaving it
   unmarked would reproduce in `STATUS` exactly the projection-versus-record
   disagreement item 3 exists to prevent.

   `docs/FEATURES.md` carries the same two defects in its Vulkan paragraph, which
   reads "decode 4.36 tok/s vs llama.cpp's 4.35, parity met narrowly" with an
   unmarked denominator and quotes the `21.5x` as though it were a llama.cpp
   ratio. Both are marked there too. That edit is also what
   `scripts/check-doc-checkpoint.py` requires: editing `.agents/backend-matrix.md`
   or `.agents/feature-matrix.md` classifies the change as `feature_surface`, and
   a `feature_surface` change owes `docs/FEATURES.md`. The paragraph is trimmed
   elsewhere to stay under the 700-character cap rather than the cap being
   raised.

3. **`.agents/backend-matrix.md` and `.agents/feature-matrix.md`.** The public
   documents are projections of these records, so marking only the projection
   leaves the two disagreeing while one links to the other.
   `BACKEND-GATE-CUDA-LLAMACPP` states the pin as `237ad9b96` and links to
   `oracles/llama-cpp.md`, so it is restated as `b10451` with the fork named as
   what it was, and its source anchors are marked as superseded locations owed
   re-anchoring rather than silently re-derived.
   `BACKEND-GATE-CPU-LLAMACPP` carries the live "at OR ahead of llama.cpp on
   every axis" verdict, so the denominator behind it is marked SUPERSEDED, the
   RPi5 clause in the same cell is marked separately because it ran stock
   `b9892`, and the verdicts are ordered by how easily the re-take flips them.
   `QUANT-GGUF` in the feature matrix repeats the same "parity or better on
   every axis" claim and gets the same marker. Two more sites carry it and were
   found by the sweep: `BACKEND-CPU` in the same matrix says "the 20-core
   Arm/i8mm Qwen3.5-2B single-stream llama.cpp floor is closed", and
   `KERNEL-GEMM-CPU-TILED` in `.agents/kernel-matrix.md` says our NEON kernel is
   "at parity with ggml's stock kernel", which is row 11 of the enumeration and a
   measurement rather than a restatement. The lifecycle state of every row
   is left where it is, because that judgement belongs to the re-measurement.
   `.agents/quantization-matrix.md` needs no edit: its `QUANT-GGUF-COMPUTE` cell
   already defers to the backend matrix as the one place the live position
   lives, so marking the backend matrix is what marks it.

4. **`.agents/issue-index.md`.** Append two rows at end of file, for #857 and
   #1003, both naming row `ORACLE-LLAMACPP-REPIN-STOCK`. Append only. Neither
   raises the unowned-row count that `scripts/check-agent-record.py` ratchets,
   because both name an owning row.

**Rejected: editing the evidence files to strike their numbers.** `AGENTS.md`
§Records forbids deleting evidence to reduce context. The numbers are true
statements about runs that happened. What was wrong is the revision they were
attributed to, and that is fixed by this spec's enumeration plus #1003, not by
removing the measurement.

**Rejected: a checker that verifies the pin exists upstream.**
`scripts/check-oracle-pins.py` documents in its own module docstring why it is
network-free, and that reasoning is right. A gate that fails when GitHub is
unreachable fails on the wrong thing. This defect is caught by review and by the
clean-tree requirement at the point of use, not by a new gate.

**Rejected: repinning to `b9892` to keep the Pi 5 evidence valid.** One arm out
of nine, and it is owed a re-take anyway under a consistent pin. Buying it costs
a permanent label collision inside the repository's own history.

## Tests

No code changes, so no new unit test. The applicable gates are the record
checkers, and one of them is directly load-bearing here.

`scripts/check-oracle-pins.py` is the gate that reads this row's main edit. Its
`check_record` requires that `gateable = no` names the owing issue as `#N`, and
refuses an issue reference as evidence when `gateable = yes`. Both directions of
that rule are already covered by `tests/scripts/test_check_oracle_pins.py`, which
`scripts/agent-preflight.sh` runs. This row does not change checker semantics and
therefore owes no red-before checker test.

The falsifiability this row does owe is of its own factual claims, and every one
is a read-only command recorded in §"Measured facts", §"How this set was
derived", §"The owed sweep, counted reproducibly", and the enumeration,
reproducible by anyone with a clone of `ggml-org/llama.cpp` plus the developer's
fork.

**One instrument precondition has to be asserted before any of the llama.cpp-side
commands are trusted.** `/home/mudler/_git/llama.cpp` is a SHALLOW clone,
grafted at `687e77892`. `git rev-parse --is-shallow-repository` returns `true`
there. Every depth, distance and ancestry answer that crosses the graft is wrong
and looks exactly like a correct answer, which is how this spec came to record a
122-commit window that is really 624 and how `merge-base --is-ancestor` reports
that `b9827` is not an ancestor of `b10451`. Re-run the §"Measured facts" table
against a full clone before extending it, or derive the distance from the
build-number convention as §"Direction of the error" does and say so.

## Gates

- `scripts/agent-preflight.sh` on the committed head, with `origin/main` merged
  first so the commit-trailer block executes rather than silently skipping.
- `git status --porcelain` empty.
- `check-env-doc` and `test_check_env_doc` were **inherited red** from
  `3005447f8` (#993), which added `VT_MOE_EXPERT_STREAM`,
  `VT_MOE_EXPERT_STREAM_SLOTS`, and `VT_MOE_EXPERT_STREAM_SLOT_BYTES`
  undocumented. Filed as #1000 and #995, owned by `ENG-EXPERT-STREAM`. **Both
  are now green here.** PR #997 landed as `45b022cdc` while this branch was in
  flight, `origin/main` was merged again, and `python3 scripts/check-env-doc.py`
  passes on a pristine `origin/main` worktree. The record says so rather than
  keeping a stale waiver, because a stale exception is the same defect this row
  removes from the oracle.
- `check-agent-record` and `test_agent_record` were **inherited red from
  `origin/main` itself**, and they arrived with that same landing.
  `.agents/issue-index.md` carried two rows for #995, one appended by this
  branch's base and one by `45b022cdc`, and the checker refuses a duplicate with
  "under `merge=union` a duplicate is what two branches appending the same issue
  look like". Proved on a pristine detached worktree at `45b022cdc` with
  `git status --porcelain` empty, not attributed: `scripts/check-agent-record.py`
  returned `rc=1` there with the identical message. **Both are now green here.**
  PR #1025 (issue #1022) landed as `ff264cb82` while this branch was in flight
  and removed the malformed copy, and merging `origin/main` brings the repair in.
  The record says so rather than keeping a stale waiver.

  **The merge that clears it also reproduced it, and the automatic result was
  accepted by nothing.** `ff264cb82` deletes one #995 row while this branch's
  side leaves it untouched, which under `merge=union` reinstates it, and the
  merge reports clean while carrying the duplicate again. `.agents/issue-index.md`
  was therefore rebuilt as AGENTS.md §Records requires, by taking the complete
  target-branch version and re-applying this branch's scoped edit, so
  `git diff origin/main -- .agents/issue-index.md` shows exactly the two rows
  this branch appends and no other key changes. A keyed record that union-merges
  cleanly is not a keyed record that merged correctly.
- `test_cpu_x86_llamacpp_floor` is a **third** inherited red, and an earlier
  draft of this section named only two while runs produced three. It is
  load-dependent: at high loadavg its contended-leg case takes the
  `NO_QUIET_WINDOW` (4) exit instead of the `GIVING_UP` (2) it asserts, so the
  guarantee goes untested and the red presents as a defect in whatever diff is in
  flight. Filed as [#618](https://github.com/mudler/vllm.cpp/issues/618) and
  owned by `BACKEND-GATE-CPU-LLAMACPP`, which is the row this spec defers its
  lifecycle judgement to. Naming it here matters for that reason: the same row
  owns both the flake and the re-measurement this repin makes owed. Measured in
  both directions rather than asserted, on this branch and on a pristine
  `origin/main` worktree: it PASSED at loadavg 12.47 and FAILED at 46, 61 and
  216, with the failure text naming the load each time. It FAILED again at 21.07
  and PASSED at 14.82 and 8.36 during this pass, which is four more datapoints on
  the same threshold and no new information about the diff.
- **No benchmark gate.** Recorded `PENDING` on #1003 and on host access, not
  waived.
- **What the gate run reports, by block.** The full run on the merged head is
  `Session role` 1, `Record gates` 26, `Mutation suites` 44,
  `Committed range vs origin/main` 3, and `Commit trailers vs origin/main` 2, so
  76 results, all `ok`, and **zero SKIP**. The last block is the one this section
  cares about: it is guarded on `git merge-base --is-ancestor origin/main HEAD`
  (`scripts/agent-preflight.sh:226-234`) and it printed nothing at all before the
  merge, which is what "silently skipping" looks like. It examined every commit in
  `origin/main..HEAD`, 9 of them at this head, and the count is the thing to
  re-derive rather than trust, for the same reason the anchor counts are.

## Stop conditions

- Stop before building or running any llama.cpp revision. This row has no host
  authority and `gateable = no` is the correct record until someone does.
- Stop before editing anything inside `/home/mudler/_git/llama.cpp` or
  `/home/mudler/_git/llama.cpp-mtp-imatrix`, including a fetch.
- Stop before moving any matrix row's lifecycle state. A record edit that has
  measured nothing may not reclassify a gate.
- Stop before striking a recorded number. Mark it superseded with its reason and
  keep its provenance.
- Stop and return `NEEDS_DECISION` if evidence appears that stock `b10451`
  cannot build or run `qwen35`, because the pin choice would then rest on a false
  premise.

## Owed

- [#1003](https://github.com/mudler/vllm.cpp/issues/1003): re-take all **eleven**
  contaminated measurements against `b10451`, on the host that produced each,
  and run the `b9827` stock versus `237ad9b96` fork A/B that isolates our own 65
  commits from upstream drift. Six of the eleven carry a verdict favourable to
  us, ranked by fragility in §"Direction of the error". Re-taking the GB10
  prefill alone does not discharge this.
- [#857](https://github.com/mudler/vllm.cpp/issues/857): build and run stock
  `b10451` on dgx.casa and record the measured identity, recipe, and evidence
  that lets `gateable` become `yes`. Until then this oracle is visible debt.
- Re-anchoring the files that cite `file:line @ 237ad9b96`. The set is the
  `@ <sha>` subset counted by the second command in §"The owed sweep, counted
  reproducibly", which returns **69** on this commit's tree. Run the command
  rather than trusting the number, because it moved three times inside this pull
  request, once because another branch landed. Tracked
  under #1003, because it is the same object that cannot be fetched.
- `KERNEL-GEMM-CPU-TILED`: its recorded "ahead on four of six shapes" reads as
  five of six in its own evidence table. Reported in §"Direction of the error",
  owned by that row, and settled by the row 11 re-take under #1003.

## Now

Records only. The pin moves to stock `b10451`, `gateable` drops to `no`, and
every llama.cpp-side number in the tree is enumerated and marked owed. The
enumeration is now the output of a recorded sweep rather than a hand list, after
two successive reviews each found one favourable verdict a hand list had missed.
Eleven contaminated measurements, six favourable verdicts, four llama.cpp
revisions. No row changes lifecycle state, and no number is re-taken. The oracle
is deliberately ungateable until #857 builds and runs the new pin.
