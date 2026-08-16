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
| Is the pin on any remote? | `gh api repos/ggml-org/llama.cpp/commits/237ad9b96`, and the same for `mudler/llama.cpp` | HTTP 422, `No commit found for SHA`, from both. See the control below |
| Is the pin on any remote? (SUPERSEDED instrument) | `git branch -r --contains 237ad9b96` | empty, and **this evidence is invalid**. See §"The `--contains` answer was measured with a blind instrument" |
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

### The `--contains` answer was measured with a blind instrument

This row's headline fact, "the pin is on no remote", rested on
`git branch -r --contains 237ad9b96` returning empty in the developer's clone.
That clone is shallow, grafted at `687e77892`, so a walk from any ref whose
history crosses the graft stops there. `--contains` is exactly such a walk, and a
commit below the graft reports as absent whether it is absent or not.

**Disproved with a control rather than reasoned about.** `b9827` is an upstream
release tag, so it is beyond argument an ancestor of `origin/master`. In that
clone `git branch -r --contains b9827` lists **70** remote branches and
`origin/master` is **not among them**. `--contains b9892` behaves the same way,
listing 63 and omitting `origin/master`. The instrument returns a false negative
for the one ref the question is about.

**That branch count is 70 today and read 68 when this paragraph was first
written, which changes nothing and is worth saying.** The count reads the
developer's clone, a repository outside this row's control that gains
remote-tracking refs whenever anyone fetches, so it is not reproducible from this
tree at all. **The load-bearing half is, and it reproduces exactly**: the omission
of `origin/master` from a list of branches containing an upstream release tag.
Re-run the command rather than trusting either number, and read the omission, not
the total.

The result survives, and the evidence had to be replaced. The GitHub API does not
walk local history:

```sh
gh api repos/ggml-org/llama.cpp/commits/237ad9b96   # 422, No commit found for SHA
gh api repos/mudler/llama.cpp/commits/237ad9b96     # 422, No commit found for SHA
gh api repos/ggml-org/llama.cpp/commits/10bf611e5   # 10bf611e533d81f739128304991c5e133c6aebd8
```

The third call is the positive control on the same endpoint and the same call
shape, which is what makes the two 422s mean "absent" rather than "the query was
wrong". Both the fork remote and upstream are checked, because "no remote" is a
claim about both.

**What the empty result still proves.** The graft truncates only the branches
whose walk reaches it. The 68 branches that do contain `b9827` were searched to
full depth and none of them carries `237ad9b96`, including the fork's own
`fork/localai-paged`, which is why the local branch of that name is ahead of its
remote. The finding was never wrong. Its evidence was, and a false negative that
happens to agree with the truth is still a false negative.

**The rest of the §"Measured facts" table does not rest on a graft-crossing
walk**, checked one row at a time rather than assumed. `describe`, `rev-list
--count` and `merge-base` over `237ad9b96`, `b9827` and `b9892` all run on the
fork line, which was fetched to full depth: `rev-list --count` returns **9892,
9827 and 9892** for the three, in that order, complete values, against the
truncated 122 that `b10451` returns through the graft. An earlier draft printed
the same multiset transposed, as "9827, 9892 and 9892", which reads as though the
fork tip sat at the tag's depth and loses the one coincidence this whole section
exists to explain: `237ad9b96` and `b9892` **both** count 9892, from the same
`b9827` base 65 commits back, and that collision is what made `pin_label` wrong.
`merge-base 237ad9b96 b9892` found
`0ed235ea2`, and a merge base that is found is a positive result a truncation
cannot manufacture. `git grep ... origin/master`, `git diff --numstat`, and the
`src/models/qwen35.cpp` and `src/llama-arch.cpp` model-path checks read trees,
not ancestry, and a tip tree is complete in a shallow clone. The one distance
that did cross the graft, `b9827..b10451`, was already re-derived from the
build-number convention in §"Direction of the error".

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
2b. `README.md` and `benchmarks/demo/vulkan_27b_llamacpp.json`. The front page
   was in scope from the start, because `AGENTS.md` §"Public documents" lists it
   as the surface that changes when a user-visible headline changes, and the
   three-directory sweep could not see it. Six sites, listed in the enumeration.
   The JSON is the `BENCH-VK-LLAMA` demo's own source and is what
   `scripts/check-doc-checkpoint.py` recognises as a landing source, so marking
   the denominator there is both the correct place for it and what permits the
   README edit rather than the README edit being unfunded churn.
2c. `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp` and
   `tests/vllm/test_gguf_keep_quant.cpp`. Row 12. This is not product-code work:
   it marks a llama.cpp denominator quoted in a comment that justifies a shipped
   default. No behaviour changes and no assertion moves.
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

It was the newest release tag present in the developer's clone under the
remote-tracking refs of `https://github.com/ggml-org/llama.cpp` on the day it was
selected, and it is identical to `origin/master` there.

**The reachability evidence is the API, not `--contains`.** `git branch -r
--contains 10bf611e5` does return `origin/master` and `origin/HEAD`, and a
positive `--contains` is safe under a graft in a way the empty one is not, since
a truncated walk cannot invent a path it did not find. It is still the weaker
statement, because it only says the developer's clone believes it. Both of these
resolve, from a host with no llama.cpp checkout at all:

```sh
gh api repos/ggml-org/llama.cpp/commits/10bf611e5        # 10bf611e533d81f7...
gh api repos/ggml-org/llama.cpp/git/ref/tags/b10451      # 10bf611e533d81f7...
```

The second one is what closes the collision this row exists to remove: the label
`b10451` and the commit `10bf611e533d81f739128304991c5e133c6aebd8` are bound to
each other **on the remote**, which is a check the old `pin_label = b9892` would
have failed, because that label resolves upstream to `ee445f93d` and not to the
recorded pin.

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

**Honest limit on the choice, now measured instead of hedged.** An earlier draft
wrote that this row did not fetch, so "a newer tag can exist upstream". One does.
`gh api repos/ggml-org/llama.cpp/releases/latest` returns **`b10453`**, two tags
past the pin.

The pin stays at `b10451`, and the reason is not inertia. The criterion is the
newest stable release **at selection time**, not the newest that exists at read
time. Upstream tags several times a day, so a pin re-chosen whenever somebody
looks is a moving target, which is the unpinned-oracle defect this row was opened
to remove. A pin is a 40-hex commit and is reproducible whether or not a newer
one exists. Nothing has been measured against `b10451` yet either, so advancing
it would invalidate no evidence and buy no accuracy, while costing another pass
over every site this row marks. #857 builds whatever the pin says when it runs,
and it is the row that may advance it with a measurement behind the change.

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

**The set below is swept, not listed.** Three successive fresh reviews each found
a verdict a hand enumeration had missed, the Vulkan `MET` and the Pi 5 RSS in the
first pass, Muse Glimmer in the second, and `README.md` in the third. A fourth
hand pass would have the same failure mode, so the set is derived by a query
anyone can re-run, and the query is recorded here rather than its output alone.

### The sweep's own scope was the fourth miss

The sweep was written to replace hand enumeration, and it took its **path set**
from a hand enumeration: `git ls-files docs .agents benchmarks`. `README.md` is
at the repository root and is in none of those three directories, so the front
page was outside the instrument that existed to stop exactly this. It carries the
CPU comparison table, the `1.18x llama.cpp's prefill` headline, and the Vulkan
`4.36 vs 4.35` claim, which is the most fragile verdict in the whole enumeration,
stated to every reader of the project as "matches".

`AGENTS.md` §"Public documents" lists `README.md` as a surface that changes when a
user-visible headline changes, so it was in this row's scope from the start. The
sweep could not see it, and the three-directory list looked complete because
every site anyone had thought of was inside it.

The lesson is not "add `README.md`". Adding one path is the hand enumeration
again, one level up, and it would leave `MANIFESTO.md`, `CONTRIBUTING.md`,
`website/`, and every directory this repository has not created yet in the same
hole. **A path set that is written down is a hand list no matter where it is
written.** So the sweep stops having one.

**Stage 0, the path set: every tracked file, with no path argument at all.**
`git ls-files` with no operands. There is no include list to fall behind the tree
and no exclusion list to justify, and a file added anywhere, at the root or in a
directory nobody has made yet, is in scope on the day it is committed. The
question the sweep asks is "does this repository state a llama.cpp comparison
anywhere", and that question has no principled directory boundary, because a
claim reaches a reader from a source comment as readily as from a page.

Two objections, both measured rather than argued.

*It will not scale.* It does, but **not in the two seconds an earlier draft of
this paragraph claimed**, and a wrong measurement inside a section headed "both
measured rather than argued" is worse than no number. The path set goes from 651
files to **4514**, and the candidate output goes from 961 lines to **1008**,
drawn from 117 files rather than 93. The extra 3863 files contribute 47 lines, so
scanning everything costs 47 lines of adjudication.

Timed rather than asserted, six runs of the whole-tree sweep on this host:
**40.4, 40.9 s** at loadavg 15, and **43.1, 45.5, 51.2, 54.4 s** at loadavg 62
falling to 30. The three-directory path set runs in **3.1 to 3.4 s** on the same
host. So the whole-tree sweep costs roughly **40 to 55 seconds**, is
load-sensitive, and is **more than an order of magnitude** slower than the set it
replaced, not "about the same". That is still cheap for an instrument run once
per pass, which is what the objection was actually about, and it is the honest
version of the answer: the cost is real and it is worth paying, rather than
nil.

*Binary fixtures will produce garbage.* They do not. The 456 `.npy`, 381 `.bin`
and 252 `.i32` fixtures are read with `errors="replace"` and match nothing, since
a candidate needs a comparison token **and** llama.cpp attributed within three
lines. No exclusion list is needed, so none is written, which is the point: an
exclusion list is the same hand list wearing a different hat.

**Stage 1, the candidate sites.** A line is a candidate when it carries a
comparison token and llama.cpp is attributed to it, where attribution means the
line names llama.cpp, or a line within three names it, or the line is a markdown
table body row whose HEADER names it. The table-header clause is the one that
matters: it is what catches
`| Prefill | 223.8 tok/s | 177.3 | 1.18x | PASS |`, whose own text never says
llama.cpp. A `git grep -C` alone cannot see that row's denominator.

```sh
git grep -n -C 3 -iE 'llama\.?cpp|llama-bench'
```

is the readable approximation and misses exactly the table-row class. The whole
sweep is below. It is short on purpose, so that it lives in this spec rather than
becoming another file the tree has to keep true:

```python
import re, subprocess
LLAMA = re.compile(r"llama[.\-_]?cpp|llama-bench", re.I)
ROW, SEP = re.compile(r"^\s*\|"), re.compile(r"^\s*\|[\s:|-]+\|\s*$")
CMP = re.compile(r"[0-9]+(?:\.[0-9]+)? *(?:x\b|×)|[0-9]+(?:\.[0-9]+)? ?%"
                 r"|\bMET\b|\bPARITY\b|\bPASS\b|\bFAIL(?:ED|S)?\b|\btie\b|\bties\b"
                 r"|\bahead\b|\bbehind\b|\bbeats\b|\bparity\b|\bwin\b|\bwins\b"
                 r"|\bless\b|\bfaster\b|\bslower\b|\bmatch(?:es|ed)?\b")

# SELF-TEST, and the sweep refuses to run without it. See §"A token can be
# WRONG as well as MISSING". Every shape below is one this repository actually
# writes, and the MUST_NOT strings are the recipe prose the sweep must NOT claim.
MUST = ["1.18x llama.cpp", "1.18× llama.cpp", "| 1.023× |", "2× over llama.cpp",
        "3.9×decode", "24.2 % less", "decode MET", "matches llama.cpp",
        "at parity", "ahead of pp128", "0.97x tie", "24.4 t/s slower"]
MUST_NOT = ["the llama.cpp build recipe", "unpacked at ~/lcpp-vk",
            "llama-bench was built with GGML_CUDA=OFF"]
bad = ([s for s in MUST if not CMP.search(s)]
       + [s for s in MUST_NOT if CMP.search(s)])
assert not bad, f"CMP self-test FAILED on {bad}"

for path in subprocess.run(["git", "ls-files"],           # NO path arguments
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

### The token set was the fifth hole, and it is the same defect

Widening the path set surfaced the README's CPU table and its `1.18x` headline.
It did **not** surface `README.md:310`, the hardware-table row reading
"Qwen3.6-27B decode **matches llama.cpp Vulkan** (4.36 vs 4.35)". `CMP` had no
token for the word this project uses in its own headline. The bullet at
`README.md:46` says "Vulkan **matches** llama.cpp on a 27B", and it reached the
output only because the line three below it happens to carry `0.69%`.

A declared match is a favourable verdict. `\bmatch(?:es|ed)?\b` is therefore in
`CMP` above, and in the stage-2 filter for the same reason. That is one more
instance of the rule the path set already taught: a set of accepted words is a
hand list, and it fails the same way. It is bounded here only because a verdict
has to be **written** in some vocabulary, so the honest record is that this
token set is a known-incomplete instrument whose incompleteness is now visible
in two places rather than none.

### A token can be WRONG as well as MISSING, and that is the sixth hole

The five holes above are all the same shape: something true was **absent** from
a list. This one is different and it is sharper, because the token was
**present** and did nothing.

`CMP` read `[0-9]+(?:\.[0-9]+)? *[x×]\b`. `×` is U+00D7, which is not a `\w`
character, so a `\b` immediately after it is satisfied only by a **following**
word character. The alternative therefore fired on `3.9×decode` and on nothing
else this repository writes. Measured on the numeric-ratio alternative in
isolation, so that the other twenty tokens cannot mask it:

| string | old `[x×]\b` | repaired `(?:x\b|×)` |
|---|---|---|
| `our kernel is 1.18x llama.cpp` | HIT | HIT |
| `our kernel is 1.18× llama.cpp` | **MISS** | HIT |
| `\| 1.023× \|` | **MISS** | HIT |
| `speedup 2× over llama.cpp` | **MISS** | HIT |
| `3.9×decode` | HIT | HIT |

Three of the four `×` shapes are the ones this tree uses, and the one that
worked is the one nobody writes. The repair is `(?:x\b|×)`: keep the word
boundary where it is meaningful, drop it where it cannot be satisfied.

**Why nothing caught it for five drafts.** The stage-2 filter at §"Stage 2"
writes the same idea as `*[x×]` with **no trailing `\b`**, so stage 2 was live
for `×` the whole time. The two expressions disagreed, only the narrower one
gated, and stage 2 could never surface what stage 1 refused to hand it. A
downstream filter that is correct is not evidence that the upstream one is,
and a token that is spelled in two places is a token that can be right in one.

**So the sweep now self-tests, and the assertion is the fix.** A handful of
strings that must match and must not match run before the scan, and the sweep
refuses to start when one fails. That is what distinguishes a wrong list from a
short one: reading a regex confirms a token is *there*, and only executing it
against a known-positive string confirms the token is *live*. The self-test is
armed rather than decorative, proved by running it against the defective
expression it replaces: it fails on exactly `1.18× llama.cpp`, `| 1.023× |` and
`2× over llama.cpp`, and passes on the repair.

**The lesson generalises past this row.** Every earlier hole here was found by
someone noticing a missing site. A dead branch cannot be found that way, because
the instrument reports the same clean output whether the branch is unreachable
or the tree is clean. It is the `doctest -tc` comma trap and the mutation that
never applied, in a third costume: a green result from an instrument that was
never armed. Assert the instrument fires before you believe what it did not
find.

**What it hid.** A fifth llama.cpp revision, in the row below.

### What the three sweeps measure, side by side

**All three rows are measured at `bf621287a`**, on a clean detached worktree with
`git status --porcelain` empty, and that SHA is stated rather than "this commit's
tree" for a reason given below. Each stage is counted, not quoted:

| Sweep at `bf621287a` | Files scanned | Stage 1 candidates | Files hit | Stage 2 favourable |
|---|---:|---:|---:|---:|
| `docs .agents benchmarks`, old `CMP` | 651 | 961 | 93 | 719 |
| every tracked file, old `CMP` | 4514 | **1008** | **117** | **752** |
| every tracked file, `+ match(es\|ed)` | 4514 | **1095** | **141** | **842** |

The middle row is the path widening on its own, so the two changes are
separately attributable: the path set is worth +47 candidates over +24 files, and
the token is worth a further +87 over +24 more.

**The fourth change is the `×` repair, and it is measured at `85a9a7ae7`**, the
head it was found at, on a clean tree. Two more commits had landed by then, so
these are not comparable to the rows above and are not meant to be. Only the
delta between them is, because both were taken in one pass over one tree:

| Sweep at `85a9a7ae7` | Files scanned | Stage 1 candidates | Files hit | Stage 2 favourable |
|---|---:|---:|---:|---:|
| dead `[x×]\b` | 4514 | 1118 | 141 | 852 |
| repaired `(?:x\b\|×)` | 4514 | **1225** | **144** | **958** |

The repair is worth **+107 candidates over +3 files** and +106 favourable lines,
which is a larger correction than the `match(es|ed)` token that prompted the
previous pass. Three files enter the sweep that were never in it:
`.agents/specs/deepseek-v4-flash.md`,
`.agents/specs/spec-decode-scoping-2026-07-10.md`, and
`src/vt/cuda/cuda_quant_iq_tables.cuh`.

**The SHA is load-bearing, because this table is inside its own instrument.** The
sweep scans every tracked file, and this spec is a tracked file, so writing a
count here changes the count. Measured rather than reasoned about: the same three
rows read 975, 1023 and 1117 on the worktree that carries this pass, and the last
of those moved by 3 between running the sweep and editing the stage-2 paragraph
four lines below it. The first sweep also printed 943 at `bab8e1fb3`. A number
here is a reading, taken at a named commit, of a quantity that this row's own
prose perturbs. Re-run the query. Do not quote the table.

**Stage 2, favourable by the page's own rule.** `docs/BENCHMARKS.md:32-34`
states the convention: throughput is ours over the reference, latency is the
reference over ours, so **1.0 or higher is a win**. Piping stage 1 through

```sh
grep -E '\b[1-9][0-9]*(\.[0-9]+)? *[x×]|\bMET\b|\bPARITY\b|\bPASS\b|\btie\b|\bahead\b|\bbeats\b|\bwin\b|% less|\bparity\b|\bmatch(es|ed)?\b'
```

leaves the count in the comparison table above, and those are adjudicated one at
a time below. Most fall out as ours-versus-vLLM, ours-versus-ours, or superseded
ledger history. `match(es|ed)` is here as well as in `CMP` because a declared
match is a favourable verdict: "matches llama.cpp" is how this project states a
tie on its own front page.

**Stage 3, the revision behind each survivor**, read from the evidence record
that produced it rather than assumed from the pin. That is what surfaced a third
and a fourth llama.cpp revision in the tree.

The over-inclusive stage 1 is the point. A filter tuned to the verdicts already
known would reproduce the enumeration it was meant to check. It is inclusive
enough to count a few lines of its own regex, which is the correct direction for
this instrument to err in.

### What the widened path set surfaced, adjudicated one at a time

The 47 lines the whole-repository path set adds arrive from 24 files. Every one
is adjudicated here, because a count that is not adjudicated is the hand
enumeration again with a number in front of it.

| Verdict | Sites | Disposition |
|---|---|---|
| **Contaminated, marked** | `README.md` at `:46-48` (Vulkan `4.36 vs 4.35`), `:78` (`1.18x llama.cpp's prefill`), `:113-121` (the CPU comparison table and its prose), `:155` ("matches or beats llama.cpp on GGUF"), `:308` ("At or ahead of llama.cpp on every GGUF axis"), `:310` (the Vulkan hardware row) | new SITES of enumeration rows 1, 2 and 7, on the surface with the widest audience. Marked in place, no number softened or deleted |
| **Contaminated, marked, and NEW to the enumeration** | `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:184-189` and `tests/vllm/test_gguf_keep_quant.cpp:480-481` | row 12 below. The only place the contamination reaches a shipped DEFAULT |
| **Ours-versus-ours, stands** | `include/vllm/model_executor/model_loader/gguf_keep_quant.h:165` (`1.3-3.0x on aarch64`), `tests/vt/test_vulkan_backend.cpp:896-897` (pack and rows tactic A/B), `src/vt/vulkan/shaders/vt_matmul_vec.comp:46` | caught by proximity to a `llamacpp` filename or to unrelated prose. No llama.cpp denominator |
| **A wrong figure being corrected** | `examples/bench/bench_core.h:148`, which records how a "~500x behind llama.cpp on prefill" figure came from charging decode time to prefill | a methodology warning, not a live verdict. It stands and is worth keeping |
| **Harness code, no recorded number** | `scripts/cpu-x86-llamacpp-floor.sh:295`, which computes `"ours" / "tie" / "llama.cpp"` from a ratio | it pins no revision at all, so row 8's re-take can drive it unchanged |
| **False positives** | the remaining 14 files, on `96.92 %` checkpoint fractions, `vLLM-parity goldens`, `parity/` in a test path, and `\bahead\b` inside a thread-race comment | the price of an over-inclusive stage 1, paid deliberately |

### What the `×` repair surfaced, adjudicated the same way

107 lines from 20 files. The great majority are the **same measurements already
enumerated**, written with `×` instead of `x`, which is why the repair changed
files hit by only 3: the dead branch was suppressing lines inside files the sweep
already reached through some other token. That is its own warning. A hole that
costs few new FILES can still hide a whole measurement, because the adjudicator
reads the lines, not the file list.

| Verdict | Sites | Disposition |
|---|---|---|
| **Contaminated, marked, and NEW to the enumeration** | `.agents/specs/laguna-s21-w7-speed-2026-07-31.md:15-16`, `:68`, `:100`, `:147`, `:175`, `:240`, with `.agents/benchmark-record.md:884`, `:898`, `:11753`, `:11759`, `:11761` | **row 13**, and a FIFTH llama.cpp revision. See §"Five llama.cpp revisions are in play" |
| **New SITES of enumerated rows** | the L7 restatements at `.agents/benchmark-record.md:10731`, `:10733`, `:11077`, the L5/L6 RSS chain at `:11030`, `:11116`, `:11141`, `:11147`, the row 1 prefill at `:10792`, `:10810`, the row 3 refresh at `:10906`, and the row 2, 4, 5 and 6 spec restatements in `cpu-llamacpp-floor-remeasure`, `cpu-gdn-proj-orientation:74`, `cpu-elementwise-gemm`, `gguf-cpu-threadpool` and `gguf-compute-in-quant-gemm` | already owed a re-take under their own rows. No new obligation, and no number moves |
| **A new SITE, and a deficit rather than a win** | `.agents/specs/cpu-thread-gdn-paged-2026-07-23.md:14`, `:86`, `:92`, which reads "prefill is 2.34× behind (73.97 vs 173.28 t/s)" | a site of row 5, not a thirteenth measurement: it re-uses row 5's own `pp128 173.28 ± 1.75` legs rather than taking its own. Recorded AGAINST us, so like row 8 it cannot flip from favourable to unfavourable. It can only widen or close |
| **Ours-versus-ours or ours-versus-vLLM, stands** | `.agents/specs/deepseek-v4-flash.md:443`, `deepseek-v4-last-mile.md:63`, `spec-decode-scoping-2026-07-10.md:67`, `.agents/coordination.md:573`, `.agents/benchmark-record.md:665`, `:667`, `:1176`, `:11569` | no llama.cpp denominator. Caught by proximity to the word in adjacent prose |
| **Superseded ledger and legacy history** | the 19 lines in `.agents/completed/state-events/0000-00/STATE-LEGACY-000001.md`, the 2 Kimi K3 rows in `.agents/parity-ledger.md`, and `.agents/completed/roadmap-v1-preamble-2026-07-18..2026-08-03.md` | history, which keeps saying what it said. `AGENTS.md` §Records forbids striking it |
| **False positives** | `src/vt/cuda/cuda_quant_iq_tables.cuh:19` (`~32×` warp serialization), `.agents/model-matrix.md:453`, `.agents/specs/cuda-arch-ampere-fastpath.md:287`, `expansion-map-2026-07-10.md:40-41` | the same deliberate over-inclusion. Two of the three newly-hit FILES are here |

### The sweep sees verdicts, not recipes, and that is a second surface

Stated here so that the next reader does not mistake this instrument for a
complete answer. The sweep requires a **comparison token**. A file that names
llama.cpp without stating a verdict about it produces **zero** candidates, by
design and correctly, because the sweep's question is "does this repository state
a llama.cpp comparison". Measured at `85a9a7ae7` with the repaired `CMP`: **286**
tracked files name llama.cpp, the sweep hits **144**, so **142 files name it and
are invisible to the sweep.** Under the dead `[x×]\b` the invisible set was 145.

Those 142 are build recipes, source citations, name-map references, porting
notes, and environment pages. Most owe nothing. **One of them owed something and
was found by hand.** `.agents/environment.md:435` is named in row 7's Evidence
column and read "llama.cpp at pin `237ad9b96` is unpacked at `~/lcpp-vk` with
`build-vk/bin/llama-bench` built". It carries no comparison token, so the sweep
never saw it, and it is not in the `@ <sha>` subset either, so the owed
re-anchoring sweep did not cover it. An agent taking #1003's Vulkan re-take, the
**most fragile verdict in the whole enumeration**, reads that page for the recipe
and rebuilds the superseded fork while believing it is following the pin. It is
marked by this pass.

**The general case is open, and no third stage is proposed for it.** A recipe
sweep would be a different query with a different adjudication, and inventing it
here without running it would be exactly the hand list this row keeps removing.
What this section records is the boundary: **verdicts and recipes are two
surfaces, this instrument covers one, and the pin reaches a reader through
both.** It is listed under `## Owed`.

## Every measurement the fork pin contaminated

"Contaminated" means the llama.cpp side of the comparison came from a build of
`237ad9b96`, or from a build whose tree cannot be identified, or from a stock
revision that is neither the old pin nor the new one. Our own side of each
comparison is untouched.

| # | Evidence | Recorded llama.cpp numbers | Re-take? | Notes |
|---|---|---|---|---|
| 1 | `docs/BENCHMARKS.md` GB10 20-core CPU floor, **and `README.md:113-121`, `:78`, `:155`, `:308`** | prefill 177.3 tok/s, decode 25.4 tok/s, peak memory 2.80 GiB, giving `1.18x PASS`, `0.97x` tie, `1.01x` parity | **yes** | the highest-value row. Built fresh on dgx with `-DGGML_CUDA=OFF`, running `qwen35`, so it executed the fork's fused CPU GDN and SSM_CONV ops. The README sites were found by the widened path set, not by any earlier pass, and they are the highest-traffic statement of this verdict in the repository |
| 2 | `.agents/specs/cpu-llamacpp-floor-remeasure-2026-07-22.md` | pp512 211.16, pp128 174.63, tg128 26.13, tg32 25.80, tg32 isolated 25.16 tok/s, peak RSS 2.798 GiB, and the derived `33.5x` prefill, `11.6x` decode, `2.65x` RSS | **yes** | same recipe and host. Its ours-side table, its threadpool A/B, its op-dispatch attribution profile, and its GEMM microbenchmark are ours-only and stand |
| 3 | `.agents/specs/gguf-compute-in-quant-gemm.md` G4 and G7, mirrored in `.agents/benchmark-record.md:11046` | llama.cpp pp128 180.14 +/- 2.78, tg32 25.37 +/- 0.81 tok/s, "refreshed same session" | **yes** | a second dgx build of the same fork |
| 4 | `.agents/specs/cpu-gdn-proj-orientation-2026-07-23.md` | its llama.cpp denominator, 5 reps under one `flock` | **yes** | highest risk of the set. The row is about the GDN projection path and the fork's CPU delta is precisely in `gated_delta_net` and `ssm_conv` |
| 5 | `.agents/specs/cpu-elementwise-gemm.md` | its llama.cpp comparison legs at `237ad9b96` | **yes** for the comparison legs, **no** for the E1-E4 gate | the E1-E4 gate is bit-exactness plus our own GFLOP/s, which never touched llama.cpp |
| 6 | `.agents/specs/gguf-cpu-threadpool.md` W4 | its llama.cpp context numbers | **yes** for the context, **no** for W4 itself | W4's acceptance is a same-binary A/B of our arm at 1 versus 20 threads. It is ours-only and unaffected |
| 7 | `docs/BENCHMARKS.md` Vulkan `BENCH-VK-LLAMA`, with `.agents/environment.md:435`, **`benchmarks/demo/vulkan_27b_llamacpp.json`, and `README.md:46-48` and `:310`** | decode 4.36 versus 4.35 `MET`, 7 clean legs, spread 0.69% | **yes**, and it is the most fragile verdict in this table | none of the 65 commits touch `ggml/src/ggml-vulkan/`, so the Vulkan sources match `b9827`. The build still came from the same dirty working tree, so the tree is unidentified and the number is unreproducible. The `21.5x` prefill quoted in the same cell is **not** a llama.cpp number: the source JSON says "Prefill is 21.5x its pre-campaign value on the same model", a self-comparison, so it survives the repin untouched |
| 8 | `docs/bench-evidence/cpu-x86-llamacpp-20260811.md` | peak RSS 2.8281 GiB, giving `1.0022x` open gap. Its three throughput axes are already `PENDING` | **yes** | its provenance line names "local fork `237ad9b96`, build number 9892, the recorded pin". RSS is the axis least likely to move, because the fork's deltas are compute, but the binary is still unidentified |
| 9 | `docs/bench-evidence/rpi5-a76-llamacpp-20260806.md` | prefill 27.77, decode 3.91, E2E 3.77 tok/s, peak RSS 3.747 GiB, giving `0.461x`, `0.653x`, `0.758x` | **yes**, for a different reason | **not fork-contaminated.** This file measured stock tag `b9892` at `ee445f93d` and recorded the substitution. It is a stock number against a revision that is neither the old pin nor the new one, and its record wrongly presents that tag as the project pin |
| 10 | `docs/BENCHMARKS.md:29` Muse Glimmer 30B, `.agents/specs/cpu-decode-barrier-and-attn-dispatch.md:24-32`, `docs/STATUS.md:502`, record `:19231` and `:19016` | in128 prefill 13.158, decode 5.026, in512 prefill 13.292, decode 5.091 tok/s, and the earlier 12.94 / 5.08 / 9.97 / 6.41 / 13.13 / 5.00 set with peak RSS 15.74 GiB, giving the `1.023x` prefill win, `0.194x`, `0.175x`, `0.997x` and `1.92x MORE` RSS | **yes**, for the same reason as row 9 | **not fork-contaminated.** Both runs measured stock master `704485942` (`b10362-5`, 2026-08-11), recorded in the record's own recipe block at `:18996`. It is a stock number against a third revision that is neither pin. Its `1.023x` is the fifth favourable verdict on the public page and was absent from every earlier draft of this table |
| 11 | `.agents/kernel-matrix.md:162` `KERNEL-GEMM-CPU-TILED`, record `:13766-13782` | ggml no-llamafile 212.0, 214.4, 215.4, 208.1, 209.9, 159.2 GFLOP/s against our 222.1, 220.6, 216.8, 215.4, 241.7, 141.3 on six Arm shapes, giving "at parity with ggml's stock kernel and slightly ahead on four of six shapes", plus the stock-ggml column that sizes llamafile at ~1.9x f16 and ~1.2x f32 | **yes** | built from the same fork tree with `GGML_LLAMAFILE=OFF`. None of the 65 fork commits touch `llamafile/sgemm.cpp`, so the compared kernel matches `b9827`, but the tree is the same unidentified one as row 7. This verdict is load-bearing beyond its own row: it is the evidence that the Arm 16-bit deficit is an absent capability rather than a defect in `KERNEL-GEMM-CPU-ELEM` |
| 12 | `.agents/specs/gguf-keep-quant-loader.md` L7 (`:128`, `:537`ff), restated in `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:173-228` and pinned beside a `CHECK` at `tests/vllm/test_gguf_keep_quant.cpp:478-494` | llama.cpp pp128 173.2, tg32 25.09 tok/s, peak RSS 2.798 GiB, giving "RSS gap CLOSED to `1.01x`", "prefill 204 t/s = `1.18x` AHEAD", "decode ~parity" | **yes** for the three ratios, **no** for the keep-f16 default | **the only site where a contaminated denominator reaches shipped behaviour, and the default is NOT safe from it.** `VT_GGUF_KEEP_F16` is DEFAULT ON. Its binding A/B has three axes and **two regress**: prefill about 10% worse (224 to 204 t/s) and decode about 1.4% worse, bought for 1.05 GiB of RSS. The recorded reason the prefill loss is acceptable is `:595`, "comfortably above the competitor floor", which IS the contaminated `pp128 173.2`. An earlier pass called the default safe by citing only the RSS leg. See §"The keep-f16 default rests on the contaminated floor". The product comment also quotes a `1.16x AHEAD of pp128 176.6` that no recorded run produces, see §"176.6 is not a number this tree measured". Its `173.2 / 25.09` legs are a distinct session from rows 2, 3 and 5, so this is a genuinely separate contaminated measurement rather than a restatement |

| 13 | `.agents/specs/laguna-s21-w7-speed-2026-07-31.md:15-16`, `:68`, `:100`, `:147`, `:175`, `:240`, with `.agents/benchmark-record.md:884`, `:898`, `:11753`, `:11759`, `:11761` | Laguna-S-2.1 on the identical UD-Q4_K_XL GGUF: decode **27.8 tok/s** (36.0 ms/tok, 183 GB/s = 76% of the GB10 240 GB/s peak), giving the campaign's `15x` warm and `18x` cold gap, then `18x → 4.7x` at W8 and `18x → 3.6x` at W9 | **yes** | **a FIFTH revision, and the worst-identified of the five.** The denominator is a **Poolside fork** of llama.cpp, branch `laguna` (`.agents/specs/laguna-s21-w4-2026-07-31.md:65` names `github.com/poolsideai/llama.cpp@laguna`). A branch is not a revision. `git grep -i poolside` returns no SHA for it anywhere in this tree, so by this row's own definition the tree cannot be identified and the number is unreproducible. It is not fork-`237ad9b96`-contaminated and it is not stock. No verdict here is favourable to us, so it does not join the seven, but it is the **target** the whole Laguna speed campaign is ranked against |

Row 9 is the one to read twice. It is the only arm whose author noticed the pin
was unobtainable, and the correct handling of that discovery, a recorded explicit
substitution, still produced a number attributed to a revision the registry never
pinned. The defect is the registry, not that file.

### The keep-f16 default rests on the contaminated floor, and an earlier pass said it did not

This is the correction that reaches a user's bytes, so it is stated at length.

An earlier pass of this row was asked whether the llama.cpp figure was
load-bearing for `VT_GGUF_KEEP_F16` shipping DEFAULT ON. It answered no: the
default stands on a same-binary ours-versus-ours A/B, `3.885 → 2.832 GiB` with
tokens byte-identical, which no denominator move touches. That answer is
**wrong**, and it is wrong in the most ordinary way an answer can be. It read one
row of a three-row table.

The owning A/B is `.agents/specs/gguf-keep-quant-loader.md:587-590`, mirrored at
`.agents/benchmark-record.md:10726-10728`. It has **three** axes, and **two of
them move against the default**:

| arm | peak RSS | TTFT, 3 reps | prefill | TPOT | decode |
|---|---|---|---|---|---|
| base, keep-f16 OFF | 3.885 GiB | 570/571/574 ms | 224 t/s | 40.4 ms | 24.7 t/s |
| **L7 default, keep-f16 ON** | **2.832 GiB** | 628/625/625 ms | **204 t/s** | 40.95 ms | 24.4 t/s |

Peak RSS improves by 1.05 GiB. Prefill gets about **10% worse** (571 to 625 ms
median TTFT, 224 to 204 t/s). Decode gets about **1.4% worse**. The default is a
deliberate trade, which is fine and normal. What matters is the tie-break: **why
is losing 10% of prefill acceptable?**

The spec answers that question in the llama.cpp denominator's own terms, at
`:595-597`:

> Prefill 204 t/s = 1.18x AHEAD of pp128 173.2 (prefault removed L6's 0.72x
> regression; ~9% under the keep-f16-off default's 224 t/s but **comfortably
> above the competitor floor**).

The clause "comfortably above the competitor floor" **is** the justification for
accepting the prefill loss, and that floor is `pp128 173.2`, measured against the
contaminated fork `237ad9b96`. So the contaminated denominator is not decoration
on this default. It is the reason one of its two regressions was accepted.

**The consequence, stated so the re-taking agent cannot miss it.** `b10451` is
**624 commits** past `b9827` and carries upstream's own `fused_gdn`, so the
direction is **not established** (§"Direction of the error"). If a re-taken stock
`pp128` lands **above 204 t/s**, the "comfortably above the competitor floor"
clause fails, and the default's only recorded justification for its prefill loss
is gone. The RSS leg would still stand on its own, and it may well be enough. The
point is that this would then be a **live decision** rather than a settled one,
and nothing in the tree currently tells anyone to reopen it.

**The default is NOT changed here, deliberately.** This row has measured nothing
and holds no host authority, and flipping a shipped default on an argument rather
than a measurement is the failure this row exists to remove. The decision belongs
to `QUANT-GGUF-KEEPQ-LOADER` and needs the re-take first. What changes here is
that the question is recorded as open in all four places that previously recorded
it as closed: this spec, `docs/BENCHMARKS.md`, the product comment, and #1003's
index row, which said "the default itself stands" and now says the default is
owed a decision.

**The general lesson, because it is not about keep-f16.** A default justified by
a multi-axis trade is only as sound as the tie-break on its **worst** axis.
Quoting the axis that improved and calling the acceptance ours-versus-ours is
true about the axis quoted and false about the decision. When you are asked
whether a denominator is load-bearing, find the axis that **regressed** and read
why that was accepted, because the denominator is almost always hiding there.

### 176.6 is not a number this tree measured

Row 12 carries a second defect, independent of the repin and of the default
question, and it is the same shape as the `b9892` collision this row exists to
remove: **one claim with two denominators, and the one in the product code is
unsourced.**

| Where | Claim |
|---|---|
| `src/.../gguf_keep_quant.cpp`, before this pass | "restoring prefill to **~205 t/s** = **1.16x** AHEAD of pp128 **176.6**" |
| `.agents/specs/gguf-keep-quant-loader.md:595`, the owning spec | "Prefill **204 t/s** = **1.18x** AHEAD of pp128 **173.2**" |
| `.agents/benchmark-record.md:10722`, `:10733`, the binding record | llama.cpp fresh on the same host, **pp128 173.2 ± 2.7**, and our arm **204 t/s** at **1.18x AHEAD** |

`git grep '176.6'` returns the two product-comment lines and exactly one other
site, `.agents/benchmark-record.md:20688`, which is the mean of an unrelated MoE
`bps` microbenchmark **in microseconds** and has nothing to do with llama.cpp.
`git grep '173.2'` returns the binding record, the owning spec, the L7 session,
and rows 2 and 5 of this enumeration, which is what a real denominator looks
like.

**No recorded llama.cpp run produces 176.6, and no recorded run of ours produces
205 t/s.** The pair is internally consistent, since 205/176.6 = 1.16, so it was
computed rather than mistyped, but neither operand appears in any evidence file.
The row 1 arm did record a `pp128 177.32` in a **different** session
(`.agents/benchmark-record.md:10792`), which is the nearest candidate and is
still not 176.6, and pairing it with 205 does not give 1.16 either.

**This row does not pick one.** Reconciling the pair by rewriting `176.6` to
`173.2` and `205` to `204` would silently assert that the comment always meant
the record's numbers, and that is exactly the undocumented re-derivation this row
was opened to stop. Both denominators are superseded anyway, so the difference
changes no decision today. What the comment now says is what is true: the figures
it quotes **cannot be traced to any recorded run**, the owning record says 204
against 173.2, and #1003 owes a single re-measured pair to replace both. If the
re-take finds the session that produced 176.6, it can be recorded then, with
provenance.

**Row 13 is worse than row 9, and the `×` repair is the only reason it is here.**
Row 9's author at least recorded which object they substituted. Row 13 names a
GitHub org and a branch name, which resolves to a different commit every time
anyone pushes to it, and it has driven a multi-week campaign's target since
2026-07-31. It sat behind a dead regex branch for five drafts of this spec. The
`27.8` is not a stale pin, it is a number with no pin at all.

**Row 12 was missed by the adjudication, not by the sweep, and that distinction
is the point.** `.agents/specs/gguf-keep-quant-loader.md:128` is inside the old
three-directory path set and appears in the old sweep's stage-1 output. Three
passes read past it anyway. What made it visible was the widened path set landing
its two source-code restatements in front of a reader, at which point the spec
line was easy to find. So the two failure modes are different and both are real:
the path set decides what can be seen, and the adjudication decides what is
looked at. Widening the first one repaired an instance of the second by
accident, which is not a method anyone should rely on twice.

### Five llama.cpp revisions are in play, and one of them is not a revision

This heading has now been wrong twice, in the same direction each time. The first
draft wrote the universal "all but the Pi 5 arm ran `237ad9b96`, and the Pi 5 arm
ran stock `b9892`". Stage 3 of the sweep disproved it and the heading became
"four, not two". The `×` repair disproves that one too. A section written to
replace a universal a sweep disproved was itself an undercount, which is what an
enumeration does whenever the instrument behind it is not asserted. Read each
revision from the evidence that produced it:

| Revision | What it is | Binding measurements | Where the tree records it |
|---|---|---|---|
| `237ad9b96` | our local-only fork, `b9827-65-g237ad9b96` | rows 1 to 8 and row 11 | the former `pin` in `.agents/oracles/llama-cpp.md` |
| `ee445f93d` | stock tag `b9892` | row 9, the Pi 5 arm | `docs/bench-evidence/rpi5-a76-llamacpp-20260806.md`, which records the substitution |
| `704485942` | stock master, `b10362-5`, 2026-08-11 | row 10, Muse Glimmer and #391 | `.agents/benchmark-record.md:18964,18996,19221` |
| `030ebb5` | stock tag `b10358` | none. One contended `pp32 9.79 / tg8 0.79` datapoint, recorded **NON-BINDING** with our own arm blocked in the tokenizer | `.agents/benchmark-record.md:18159` |
| **none recorded** | a **Poolside fork**, branch `laguna` at `github.com/poolsideai/llama.cpp` | **row 13**, the entire Laguna speed campaign's `27.8 tok/s` target | `.agents/specs/laguna-s21-w4-2026-07-31.md:65` and `laguna-s21-scope-2026-07-30.md:161`, both of which name the branch and neither of which names a commit |

`704485942` appeared nowhere in `docs/`, in this spec, or in the issue index
before the previous pass, although it is the denominator of the only llama.cpp
win on the public page besides the GB10 prefill. That is the cost of writing a
universal instead of stating what a sweep found.

**The fifth row is the one this section could not previously state, because it is
the only entry with an empty first column.** The other four are pinnable: three
are fetchable upstream objects and one is a local commit that at least names a
tree. A branch name is a moving reference, so row 13's denominator was never
capable of being reproduced, before or after this repin. `AGENTS.md` §"When vLLM
has no implementation" calls an unpinned upstream "a moving target, not an
oracle", and this is that, inside a campaign that has been ranking levers against
it for weeks.

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
66 and 52 once `docs/BUILD.md` gained the fork it was measured against, **112**,
**69** and **52** at `bf621287a`, and **113**, **69** and **52** once `README.md`
named the fork it had been measured against all along. That last move is the
fourth, and it is the one this pass was opened by: the front page had carried the
numbers without ever naming the object that produced them, so it was invisible to
the first count as well as to the sweep. The third move was not an edit
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

**Seven recorded verdicts can flip against us, five of them on the public page,
one of them holding up a shipped default, and the prefill `1.18x` is not the most
exposed of them.** An earlier draft of this spec said the GB10 prefill was "the
only llama.cpp comparison in the tree recorded as a win", and #1003 was scoped by
that sentence, so an agent reading it would have re-taken prefill alone. Three
later drafts enumerated by hand and each missed one more. This list is the
stage-2 output of the sweep above. Ordered by fragility, which is the margin
measured against its own noise floor and against how much of the denominator
moves, not by the size of the margin:

1. **Vulkan `BENCH-VK-LLAMA` decode, 4.36 versus 4.35 tok/s, `MET`** (row 7).
   The margin is 0.23%. Its own source,
   `benchmarks/demo/vulkan_27b_llamacpp.json`, puts the 7-leg spread at 0.69% and
   states that "that spread IS the noise floor, so this is a narrow pass, not a
   comfortable one". The verdict is already inside its own measurement
   resolution, which makes it far more exposed than an 18% prefill margin. Any
   denominator movement can flip it. It is also the claim `README.md:46` makes to
   every reader of the project, in the word "matches", which is what the widened
   sweep put in front of this list.
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
4. **keep-f16's "RSS gap CLOSED to `1.01x` llama.cpp", with "prefill `1.18x`
   AHEAD" and "decode ~parity"** (row 12). Same 2.798 GiB denominator as item 3
   and the same tie-by-declaration shape, at 2.832 against 2.798, a 1.2% deficit
   called closed. It ranks here for fragility and it is listed for a second
   reason the other six do not have: it is quoted in
   `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp` as the
   justification for `VT_GGUF_KEEP_F16` shipping DEFAULT ON, so this is the one
   verdict on the list that a user's bytes depend on rather than a page.
   **The default is not safe from it**, which corrects an earlier draft of this
   line. Its A/B trades about 10% of prefill and about 1.4% of decode for 1.05
   GiB, and the recorded reason that prefill loss is acceptable is stated in the
   contaminated floor's own terms. A re-taken stock `pp128` above 204 t/s removes
   it. See §"The keep-f16 default rests on the contaminated floor".
5. **`KERNEL-GEMM-CPU-TILED` "at parity with ggml's stock kernel, slightly ahead
   on four of six shapes"** (row 11), in `.agents/kernel-matrix.md:162`. The one
   entry not on the public page. The bands overlap (ours 216-242, ggml 208-215
   GFLOP/s) and one of the six shapes is already behind, so the verdict is split
   4-2 before anything is re-taken. It is load-bearing for the row's attribution
   that the Arm 16-bit deficit is an absent capability rather than a defect.
6. **GB10 20-core prefill `1.18x` PASS** (row 1), and the same figure on the
   front page at `README.md:78` and in its CPU table at `:113-121`. A pass by
   0.18. Flipping it needs upstream's own work in the 624-commit window to exceed
   our fork's on the CPU Gated Delta Net and SSM_CONV path outright.
7. **Pi 5 peak RSS 2.841 versus 3.747 GiB, 24.2% less, `0.758x`** (row 9). The
   least fragile of the seven: its denominator was already stock `b9892`, so its
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

**The sweep is the exception, and it carries a red-before/green-after pair like a
checker change, because it behaves like one.** The `CMP` repair is the only edit
in this row that changes what an instrument reports rather than what a document
says, so asserting it would not be enough:

1. **The self-test is armed, proved by mutation rather than by reading.** Extract
   the spec's `python` block verbatim, replace `(?:x\b|×)` with the defective
   `[x×]\b` it came from, and run it. It exits **`rc=1`** with
   `AssertionError: CMP self-test FAILED on ['1.18× llama.cpp', '| 1.023× |', '2×
   over llama.cpp']`, naming exactly the three shapes §"A token can be WRONG as
   well as MISSING" measures. The mutation was confirmed to have **applied**, by
   diffing the mutated source against the original before running it, because a
   mutation that never applied reads as a passing test.
2. **The repaired block runs green on the same extraction**, `rc=0`, with the
   assertion reached and passed rather than skipped.
3. **The block in this spec is the block that was run.** It is extracted from the
   committed markdown and executed, not retyped, so the code a reader copies is
   the code the evidence covers.

That third point is the one worth keeping. A sweep that lives in a document is
trustworthy only while the document's copy is the copy somebody ran, and this row
spent five drafts trusting a copy nobody had executed.

**One instrument precondition has to be asserted before any of the llama.cpp-side
commands are trusted.** `/home/mudler/_git/llama.cpp` is a SHALLOW clone,
grafted at `687e77892`. `git rev-parse --is-shallow-repository` returns `true`
there. Every depth, distance and ancestry answer that crosses the graft is wrong
and looks exactly like a correct answer, which is how this spec came to record a
122-commit window that is really 624, how `merge-base --is-ancestor` reports
that `b9827` is not an ancestor of `b10451`, and how the row's headline fact was
carried for four drafts on a `branch -r --contains` that cannot see
`origin/master` at all. Re-run the §"Measured facts" table against a full clone
before extending it, or derive the answer by a method the graft cannot corrupt
and say which: the build-number convention for a distance, as §"Direction of the
error" does, and the GitHub API for reachability, as §"The `--contains` answer
was measured with a blind instrument" does.

**Both replacements carry their own control, because an instrument is not
believed on its own say-so.** For the graft, `git branch -r --contains b9827`
omits `origin/master` for a commit that is beyond argument on `origin/master`,
which is a positive fact demonstrating a false negative rather than an argument
that one is possible. For the API, `gh api
repos/ggml-org/llama.cpp/commits/10bf611e5` resolves on the same endpoint and the
same call shape that returns 422 for `237ad9b96`, so the 422 means "absent" and
not "the request was malformed" or "the network is down". A negative result from
an instrument with no positive control is the shape this row has already been
burned by twice.

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

  **The review-repair pass adds a control pair rather than another datapoint.**
  It FAILED inside the full gate at loadavg 64.88 and again at 84.95, with
  `waiting for quiet: 15s busy=109%` in the failure text, which is the
  `NO_QUIET_WINDOW` path by name. It then PASSED standalone **on this branch** at
  loadavg 26.48, and PASSED on a **pristine detached `origin/main` worktree** at
  `d1b0ea3a8` with `git status --porcelain` empty at loadavg 54.45. Both arms
  therefore pass and fail on load rather than on tree. The structural half is
  stronger than either run: `git diff --stat origin/main..HEAD -- scripts/
  tests/scripts/` is **empty**, so this branch changes neither the harness nor
  the script it exercises, and it cannot be the cause. The threshold is not
  monotonic in the one-minute average, which is expected, because the harness
  gates on instantaneous contention.
- **No benchmark gate.** Recorded `PENDING` on #1003 and on host access, not
  waived.
- **What the gate run reports, by block.** On the merged head at the review-repair
  pass: `Session role` 1, `Record gates` 26, `Mutation suites` **45**,
  `Committed range vs origin/main` 3, and `Commit trailers vs origin/main` 2, so
  **77 results and zero SKIP**. Mutation suites moved 44 to 45 because
  `origin/main` landed `test_agent_preflight_skip_report` with #1030, which is
  another count in this file that a foreign merge changes.

  **The same head reports two different results, and the difference is the box,
  not the tree.** At loadavg 64.88 and again at 84.95 it returns `rc=1` with
  **76 `ok` and 1 `FAIL`**, the #618 flake. At loadavg **15.47** the identical
  commit returns `rc=0`, **`All gates green`, 77 of 77 `ok`**. That is the
  cleanest available statement of what #618 costs: a gate whose verdict depends
  on who else is using the machine reports on the machine, not on the diff.

  **The last two block headings now name the SHA they gated against**, reading
  `... vs origin/main d1b0ea3a8e64740c20ba46cd4506794113dda61b`. That is #1030's
  repair landing, and it is the direct fix for what this bullet was written
  about. It is guarded on `git merge-base --is-ancestor origin/main HEAD`
  (`scripts/agent-preflight.sh:226-234`), and this pass reproduced the original
  failure mode once more before merging: `origin/main` advanced by two commits
  mid-session, the guard went false, and the trailer block **printed nothing at
  all**. A gate that is silent is not a gate that passed. It examined every commit in
  `origin/main..HEAD`, **11** of them at `85a9a7ae7`, and the count is the thing
  to re-derive rather than trust, for the same reason the anchor counts are. It
  read 9 two commits earlier and the paragraph was not updated, which is the
  measurement-of-one-file-inside-another coupling again, at the smallest possible
  scale. Run `git rev-list --count origin/main..HEAD`.

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

- [#1003](https://github.com/mudler/vllm.cpp/issues/1003): re-take all
  **thirteen** contaminated measurements against `b10451`, on the host that
  produced each, and run the `b9827` stock versus `237ad9b96` fork A/B that
  isolates our own 65 commits from upstream drift. **Seven** of the thirteen
  carry a verdict favourable to us, ranked by fragility in §"Direction of the
  error". Re-taking the GB10 prefill alone does not discharge this. The twelfth
  arrived with the widened path set and is the one that reaches shipped
  behaviour: re-taking it also owes the comment at
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:173-228` and the one
  beside the `CHECK` at `tests/vllm/test_gguf_keep_quant.cpp:478-494`.
- **The thirteenth cannot be re-taken the same way, and it is a different
  obligation.** Row 13's denominator is a branch name, not a commit, so there is
  no object to re-measure against. Re-taking it means choosing a revision first:
  either stock `b10451` if it now carries `laguna`, or a named Poolside commit
  recorded as a sixth entry in §"Five llama.cpp revisions are in play", and only
  then running the arm. Until one is chosen, `27.8 tok/s` has no pin and the
  Laguna campaign's `15x` to `18x` target is unreproducible.
- **The keep-f16 default is owed a decision, not only a re-wording, and #1003's
  index row now says so.** See §"The keep-f16 default rests on the contaminated
  floor". Its prefill tie-break is stated in llama.cpp's own terms, so a re-taken
  stock `pp128` above 204 t/s removes the default's only recorded justification.
  The default is NOT changed here. That is `QUANT-GGUF-KEEPQ-LOADER`'s decision
  and it needs the re-take first.
- [#857](https://github.com/mudler/vllm.cpp/issues/857): build and run stock
  `b10451` on dgx.casa and record the measured identity, recipe, and evidence
  that lets `gateable` become `yes`. Until then this oracle is visible debt.
- Re-anchoring the files that cite `file:line @ 237ad9b96`. The set is the
  `@ <sha>` subset counted by the second command in §"The owed sweep, counted
  reproducibly", which returns **69** at `bf621287a` and is unmoved by this pass.
  Run the command rather than trusting the number, because it moved four times
  inside this pull request, once because another branch landed. Tracked
  under #1003, because it is the same object that cannot be fetched.
- `KERNEL-GEMM-CPU-TILED`: its recorded "ahead on four of six shapes" reads as
  five of six in its own evidence table. Reported in §"Direction of the error",
  owned by that row, and settled by the row 11 re-take under #1003.
- **The build recipes, which are a separate surface the sweep does not reach.**
  `.agents/environment.md:435` is marked by this pass, but it was found by
  reading row 7's Evidence column rather than by the sweep, and the general case
  is not closed. See §"The sweep sees verdicts, not recipes".

## Now

Records only. The pin moves to stock `b10451`, `gateable` drops to `no`, and
every llama.cpp-side number in the tree is enumerated and marked owed. The
enumeration is the output of a recorded sweep rather than a hand list, the sweep
scans **every tracked file** rather than three named directories, and it now
**self-tests before it runs**, because its `×` token was present and dead and
that is a defect no amount of re-reading the list would find. Thirteen
contaminated measurements, seven favourable verdicts, five llama.cpp revisions,
one of which is a branch name with no commit behind it. No row changes lifecycle
state, and no number is re-taken. The oracle is deliberately ungateable until
#857 builds and runs the new pin.
