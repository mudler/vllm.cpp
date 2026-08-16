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
2. `docs/BENCHMARKS.md`, `docs/FEATURES.md`: the present-tense claims that name
   `237ad9b96` as the reference a reader can build.
3. `.agents/backend-matrix.md`, `.agents/feature-matrix.md`: the same
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
- **Re-anchoring the `file:line @ 237ad9b96` source citations.** Measured at this
  branch's head, not estimated. `git grep -l '237ad9b96'` returns **109 files**.
  ``git grep -lE '@ *`?237ad9b96'`` returns **67**, which is the subset that
  attributes a path or a source tree to the object rather than merely discussing
  the SHA. **52** of the 109 carry a `path.ext:LINE` anchor within three lines of
  the mention. The sweep is therefore 67 files, not the "roughly 50" an earlier
  draft of this spec wrote, which counted only the narrowest reading and
  undercounted the work by about a third. They are a real defect, because a
  reviewer cannot fetch the object a citation names, but they are a mechanical
  sweep with its own risk of silently re-pointing a line that moved. Listed under
  `## Owed`.
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

**Risk this introduces, stated rather than buried.** `b10451` is 122 stock
commits past `b9827`, a window in which upstream landed its own `fused_gdn`
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

## Every measurement the fork pin contaminated

"Contaminated" means the llama.cpp side of the comparison came from a build of
`237ad9b96`, or from a build whose tree cannot be identified. Our own side of
each comparison is untouched.

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

Row 9 is the one to read twice. It is the only arm whose author noticed the pin
was unobtainable, and the correct handling of that discovery, a recorded explicit
substitution, still produced a number attributed to a revision the registry never
pinned. The defect is the registry, not that file.

### Source anchors are a separate class

**67 files** attribute a path or a source tree to `237ad9b96` in the `@ <sha>`
form, out of **109** that mention the SHA at all, and **52** carry a
`path.ext:LINE` anchor within three lines of the mention. The set includes the
Vulkan port map in `.agents/specs/vulkan-full-support.md`, the Arm quant dot
anchors in `.agents/kernel-matrix.md`, `repack.cpp:4683`, `sgemm.cpp`, and the
`ggml-quants.c` IQ2_S and MXFP4 ports. These are not measurements and the ports
they justify are not invalidated. They are still broken, because no reviewer can
fetch the object a citation names, so no citation in that set is checkable. Owed,
not fixed here.

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
claim it. 122 stock commits landed after `b9827`, upstream now carries its own
`fused_gdn`, and a re-measurement can move a ratio in either direction.

**Four recorded verdicts can flip against us, and the prefill `1.18x` is not the
most exposed of them.** An earlier draft of this spec said the GB10 prefill was
"the only llama.cpp comparison in the tree recorded as a win", and #1003 was
scoped by that sentence, so an agent reading it would have re-taken prefill
alone. The counterexamples sit on the same page. Ordered by fragility, which is
the margin measured against its own noise floor and against how much of the
denominator moves, not by the size of the margin:

1. **Vulkan `BENCH-VK-LLAMA` decode, 4.36 versus 4.35 tok/s, `MET`** (row 7).
   The margin is 0.23%. Its own source,
   `benchmarks/demo/vulkan_27b_llamacpp.json`, puts the 7-leg spread at 0.69% and
   states that "that spread IS the noise floor, so this is a narrow pass, not a
   comfortable one". The verdict is already inside its own measurement
   resolution, which makes it far more exposed than an 18% prefill margin. Any
   denominator movement can flip it.
2. **GB10 20-core peak memory `1.01x` PARITY and decode `0.97x` tie** (rows 1 and
   2). Neither is a win, and both are load-bearing for
   `BACKEND-GATE-CPU-LLAMACPP` reading closed. They are ties by declaration, so a
   denominator that moves at all in llama.cpp's favour converts them into
   recorded gaps.
3. **GB10 20-core prefill `1.18x` PASS** (row 1). A pass by 0.18. Flipping it
   needs upstream's own work in the 122-commit window to exceed our fork's on the
   CPU Gated Delta Net and SSM_CONV path outright.
4. **Pi 5 peak RSS 2.841 versus 3.747 GiB, 24.2% less, `0.758x`** (row 9). A
   second favourable comparison, and the least fragile of the four: its
   denominator was already stock `b9892`, so its only exposure is upstream drift
   between `b9892` and `b10451` rather than the removal of 65 of our own commits.

The x86 peak RSS `1.0022x` (row 8) is recorded as an open gap against us, so it
cannot flip from favourable to unfavourable. It can only widen or close.

## Design

Four record edits, no code.

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

   `docs/BENCHMARKS.md` also gains the four-row fragility ranking from
   §"Direction of the error", so a reader of the page reaches the same scope
   #1003 has. It goes in as a table rather than prose because
   `scripts/check-public-doc-tables.py` holds that page at 35 of 35 prose
   paragraphs, and a table row costs none. The `BENCH-VK-LLAMA` row pays for its
   own new marker inside the 220-character cell cap by moving its `[source]` link
   into the key cell, which is what the checker's `MAX_ROW_CHARS` comment asks
   for: shorten your own row, never delete somebody else's.

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
   every axis" claim and gets the same marker. The lifecycle state of every row
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
is a read-only command recorded in §"Measured facts" and in the enumeration,
reproducible by anyone with a clone of `ggml-org/llama.cpp` plus the developer's
fork.

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
- `check-agent-record` and `test_agent_record` are **inherited red from
  `origin/main` itself**, and they arrived with that same landing.
  `.agents/issue-index.md` now carries two rows for #995, one appended by this
  branch's base and one by `45b022cdc`, and the checker refuses a duplicate with
  "under `merge=union` a duplicate is what two branches appending the same issue
  look like". Proved on a pristine detached worktree at `45b022cdc` with
  `git status --porcelain` empty, not attributed: `scripts/check-agent-record.py`
  returns `rc=1` there with the identical message. It is not this row's, and it
  is not repaired here for two reasons. The index is append-only and a landed
  row may not be edited or deleted, and choosing which of the two #995 texts
  survives is a judgement for `ENG-EXPERT-STREAM`, which owns both.
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
  216, with the failure text naming the load each time.
- **No benchmark gate.** Recorded `PENDING` on #1003 and on host access, not
  waived.

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

- [#1003](https://github.com/mudler/vllm.cpp/issues/1003): re-take all nine
  contaminated measurements against `b10451`, on the host that produced each,
  and run the `b9827` stock versus `237ad9b96` fork A/B that isolates our own 65
  commits from upstream drift.
- [#857](https://github.com/mudler/vllm.cpp/issues/857): build and run stock
  `b10451` on dgx.casa and record the measured identity, recipe, and evidence
  that lets `gateable` become `yes`. Until then this oracle is visible debt.
- Re-anchoring the roughly 50 files that cite `file:line @ 237ad9b96`. Tracked
  under #1003, because it is the same object that cannot be fetched.

## Now

Records only. The pin moves to stock `b10451`, `gateable` drops to `no`, and
every llama.cpp-side number in the tree is enumerated and marked owed. No row
changes lifecycle state, and no number is re-taken. The oracle is deliberately
ungateable until #857 builds and runs the new pin.
