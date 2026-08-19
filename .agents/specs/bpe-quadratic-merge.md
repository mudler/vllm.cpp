# SPEC-BPE-QUADRATIC-MERGE — the merge loop is quadratic, and the premise that made it look safe

**Issue:** [#1365](https://github.com/mudler/vllm.cpp/issues/1365), re-titled and
re-scoped on 2026-08-19 from "a reproducible ~4 s TTFT outlier on request 3 of
every leg" to the cause measured below.
**Kind:** one algorithm replaced inside the tokenizer, mirrored from HF
`tokenizers`. No kernel, no device code, no model numerics, no new public
surface. Every gate in this spec runs on a CPU host.
**Row:** `SPEC-BPE-QUADRATIC-MERGE` in
[`engine-matrix.md`](../engine-matrix.md), section "Loading, tokenizer, and config".
**Base:** `31f93787c`. Every local line number in this document is read there.

## Now

`origin/main` at `31f93787c` encodes every prompt through
`src/vllm/tokenizer/bpe.cpp::BpeMerge`. That function is O(n^2) in the number of
symbols of one pretoken, and the comment above it states the premise the design
rests on:

```
$ sed -n '91,93p' src/vllm/tokenizer/bpe.cpp
void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks) {
  // Repeatedly merge the lowest-ranked adjacent pair; leftmost wins ties
  // (strict < keeps the first best). O(n^2) scan; pretokens are tiny.
```

Nothing bounds a pretoken. Four of the seven pretokenizer rules return a run of
unbounded length, and on the SentencePiece family the pretoken is the entire
prompt. Measured on this tree, 64 KB of the letter `a` costs 28.1 s of one core
and 64 KB of ASCII spaces costs 48.1 s, both on a path that runs before any
length check. The row is `READY`: this spec is committed before any product
code, because the fix replaces the merge algorithm on a token-exact path and the
design has to be agreed before anyone edits the tokenizer.

## Scope

In scope:

- `src/vllm/tokenizer/bpe.cpp::BpeMerge`, replaced by a heap-driven merge that
  mirrors HF `tokenizers` `Word::merge_all`.
- Whatever `src/vllm/tokenizer/bpe.h` has to expose so that
  `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` and
  `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp`, the only two callers, reach
  the new form.
- The interning decision described under `## Design`, and the load-time refusal
  it needs.
- Red-first equivalence and cost cases in `tests/vllm/test_bpe.cpp` and one new
  CPU-tier binary described under `## Tests`.
- `docs/STATUS.md` and `docs/BENCHMARKS.md` when the row moves to `DONE`.

Out of scope, each with a stated reason:

- **A cap on pretoken length.** Argued and rejected under `## Design`.
- **The pretokenizer.** `src/vllm/tokenizer/pretokenizer.cpp` mirrors an
  upstream regex. Its unbounded runs are the regex's behaviour, not a defect,
  and shortening one changes the token identifiers we emit.
- **A word cache.** Upstream has one (`tokenizers/src/models/bpe/model.rs:475-496`)
  and it stores only sequences below `MAX_LENGTH = 256`
  (`tokenizers/src/utils/cache.rs:10`), so it does not touch the case this row
  exists for. Adding it is a separate, measurable question.
- **A request-size limit at the HTTP boundary.** Discussed under
  `## Defence in depth`, and deliberately not bundled here.
- **`ignore_merges`, the added-token split, the detokenizer.** Untouched.

## Our baseline

Every figure below was reproduced in this row, on an AMD Ryzen 9 9950X3D at
`-O2`, against a driver built from this tree's own `bpe.cpp`,
`pretokenizer.cpp` and `unicode_data.cpp`. The box carried a load average
between 130 and 190, so these are contended numbers. The **shape** is the
result. A same-binary idle re-measure belongs to the implementing row, and
`## Gates` requires it before any speed claim is accepted.

Merge table: `merges.txt` of `qwen3.8-27b-hf`, 247,586 merges, sha256
`a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d`.

### The benchmark prompts that found it

The six reconstructed `vllm bench serve` prompts of the #1365 legs, timing
`BpeSplit` only:

| prompt | bytes | pretokens | longest pretoken | current | heap prototype |
|---|---:|---:|---:|---:|---:|
| 0 | 8,798 | 698 | 88 | 3.70 ms | 3.27 ms |
| 1 | 6,629 | 344 | 119 | 2.94 ms | 2.93 ms |
| **2** | **8,034** | **1** | **8,034** | **1,356.89 ms** | **6.75 ms** |
| 3 | 6,682 | 723 | 47 | 1.65 ms | 1.93 ms |
| 4 | 6,939 | 737 | 91 | 1.69 ms | 2.01 ms |
| 5 | 5,863 | 709 | 73 | 3.72 ms | 2.59 ms |

Index 2 is the outlier request of #1365's four legs. It is one pretoken.

### The growth is quadratic

Prompt 2 truncated and tripled, one pretoken throughout:

| bytes | BPE merge |
|---:|---:|
| 2,007 | 81.57 ms |
| 4,017 | 350.61 ms |
| 6,024 | 865.44 ms |
| 8,034 | 1,933.60 ms |
| 24,102 | 14,794.38 ms |

Three times the bytes costs 7.65 times the time.

### Four rules are unbounded, not one

`src/vllm/tokenizer/pretokenizer.cpp::MatchLetterRun` (rule 2),
`src/vllm/tokenizer/pretokenizer.cpp::MatchPunctRun` (rule 4),
`src/vllm/tokenizer/pretokenizer.cpp::MatchWsNotBeforeNonSpace` (rule 6) and
`src/vllm/tokenizer/pretokenizer.cpp::MatchWs` (rule 7) each scan `while (p <
t.size())` with no cap. Only
`src/vllm/tokenizer/pretokenizer.cpp::MatchNumbers` is capped, at one codepoint
for Qwen and three for Llama-3. Each input below is 8,192 bytes returned as one
pretoken:

| input | rule | current | heap prototype |
|---|---|---:|---:|
| `a` x 8,192 | 2, letter run | 514.19 ms | 2.33 ms |
| `中` x 2,731 | 2, letter run | 438.58 ms | 0.85 ms |
| `ก` x 2,731 | 2, letter run | 505.81 ms | not run |
| `~` x 8,192 | 4, punct run | 451.81 ms | 2.20 ms |
| space x 8,192 | 7, whitespace | 813.02 ms | 3.16 ms |

The trigger is a long run of one character class, not one script. ASCII reaches
it.

### At 65,535 bytes

| input | current | heap prototype |
|---|---:|---:|
| `的` x 21,845 | 28,746.70 ms | 8.31 ms |
| `a` x 65,535 | 28,130.35 ms | not run |
| space x 65,535 | 48,098.11 ms | not run |

### The SentencePiece family pays it on ordinary English

`src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp` calls the same `BpeMerge`, and
its pretokenization is Metaspace, not the split regex. When `metaspace_split_`
is false it calls `encode_piece` on the whole input
(`src/vllm/tokenizer/tokenizer.cpp:1005-1007`). Two shipped families are in that
state: `src/vllm/tokenizer/tokenizer.cpp:566` hardcodes `false` for the Gemma
metaspace layout, and the committed Mistral golden
`tests/parity/goldens/tokenizer_mistral/tokenizer.json` declares
`"split": false` in its own `pre_tokenizer`.

Full `Tokenizer::Encode` through that committed golden, on "The quick brown fox
jumps over the lazy dog. " repeated to length. The Qwen3.6 golden is the
byte-level control on the identical text:

| prompt bytes | Mistral golden | Qwen3.6 golden |
|---:|---:|---:|
| 1,000 | 5.59 ms | 0.15 ms |
| 2,000 | 21.73 ms | 0.11 ms |
| 4,000 | 87.15 ms | 0.23 ms |
| 8,000 | 357.01 ms | 0.47 ms |
| 16,000 | 1,410.42 ms | 0.92 ms |
| 32,000 | 5,804.42 ms | not run |
| 65,536 | 24,144.72 ms | not run |

Sixteen times the bytes costs 252 times the time. That exponent is 2.0. A
Mistral or Gemma server spends 24 s of one core to tokenize a 64 KB English
document, and a user who pastes a document sends exactly that.

### Why no gate saw it

The committed 64-entry corpus at `tests/parity/goldens/tokenizer_qwen36/` has a
longest pretoken of **54 bytes**. The failing regime starts two orders of
magnitude above it. A token gate also cannot see this class of defect at all,
because the identifiers it compares are correct.

## Reachability

The encode is on the request path of a shipped server, and it runs before the
length check that a reader would expect to bound it:

```
$ sed -n '259,265p' src/vllm/v1/engine/input_processor.cpp
  std::vector<int32_t> prompt_token_ids =
      tokenizer_.EncodeWithSpecialTokens(prompt);
  ...
  ValidatePromptLen(prompt_token_ids.size());
```

`src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` runs at `:265`, five
lines after the encode at `:260`. `max_model_len` therefore bounds nothing here:
the cost is paid in full, and only then does the server decide the prompt is too
long to serve. `src/vllm/entrypoints/openai/api_server.cpp:763-764` reaches the
same encode from `/tokenize`, which needs no engine. The worker pool is sized by
`src/vllm/entrypoints/openai/api_server.cpp::HttpWorkerCount` as
`max_concurrent_streams + kControlWorkerHeadroom`, so one caller can occupy
every worker at once.

This is a remote denial of service in a shipped path. It is not a latency
curiosity, and the row is scoped that way.

## Upstream chain

vLLM does not implement a tokenizer. `vllm/tokenizers/hf.py` delegates to
HuggingFace `transformers`, which delegates to the Rust `tokenizers` crate, and
that crate holds the merge loop. So the executing chain for a vLLM encode is:

| Layer | What it decides | Anchor |
|---|---|---|
| vLLM | which tokenizer to build | `vllm/tokenizers/registry.py:176`, `vllm/tokenizers/hf.py:163` |
| `transformers` 5.14.1 | the `tokenizer.json` contract | the pin resolved inside the pinned vLLM environment, [`../oracles/transformers.md`](../oracles/transformers.md) |
| `tokenizers` 0.22.2 | **the merge algorithm** | `tokenizers/src/models/bpe/word.rs:162-250` (`Word::merge_all`) |
| `tokenizers` 0.22.2 | the merge-table representation | `tokenizers/src/models/bpe/mod.rs:9` (`type Pair = (u32, u32)`), `model.rs:19` (`MergeMap`), `model.rs:174-192` (built at load, refuses an out-of-vocabulary merge) |
| `tokenizers` 0.22.2 | the tie-break order | `tokenizers/src/models/bpe/word.rs:28-35` (`Ord for Merge`) |
| `tokenizers` 0.22.2 | the symbol list | `tokenizers/src/models/bpe/word.rs:37-53` (`Symbol`, `merge_with`) |
| `tokenizers` 0.22.2 | word construction, byte fallback, unk fusing | `tokenizers/src/models/bpe/model.rs:382-460` (`merge_word`) |
| `tokenizers` 0.22.2 | the word cache, out of scope here | `model.rs:475-496`, `tokenizers/src/utils/cache.rs:7,10` |

The crate source read for this spec is `tokenizers-0.22.2.crate`, sha256
`b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`, fetched from
`static.crates.io`. `tokenizers` is not a registered oracle in its own right; it
is the code `transformers` executes, and
[`../oracles/transformers.md`](../oracles/transformers.md) is the pin that
selects this version. The implementing row records the version it read.

## Port map

| Upstream | Ours today | After |
|---|---|---|
| `word.rs:162-250` `Word::merge_all` | `src/vllm/tokenizer/bpe.cpp::BpeMerge`, an O(n^2) rescan | the same signature, heap-driven |
| `word.rs:37-43` `Symbol {c, prev, next, len}` | `std::vector<std::string>` with `erase` per merge | a linked symbol list with `len = 0` tombstones |
| `word.rs:28-35` `Ord for Merge` | `strict <` inside the scan | an explicit comparator, with the tie pinned by a test |
| `mod.rs:9`, `model.rs:19` identifier-keyed `MergeMap` | `MergeRanks = unordered_map<string, int32_t>` in `src/vllm/tokenizer/bpe.h` | an identifier-pair-keyed table built at load |
| `src/vllm/tokenizer/bpe.cpp::MergeKey` | one `std::string` per probe | deleted; there is no key to build |
| `model.rs:180-189` `MergeTokenOutOfVocabulary` | no rule; the failure appears per request in `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` | refused at load, naming the missing token |
| `model.rs:382-460` `merge_word` | `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp`'s symbol builder | unchanged in behaviour, emitting identifiers |
| `model.rs:475-496` word cache | absent | still absent, and out of scope |

## Design

**Mirror `Word::merge_all`.** HF `tokenizers` 0.22.2, the backend the pinned
`transformers` 5.14.1 resolves, solves this at
`tokenizers/src/models/bpe/word.rs:162-250`. The crate source is
`tokenizers-0.22.2.crate`, sha256
`b238e22d44a15349529690fb07bd645cf58149a1b1e44d6cb5bd1641ff1a6223`. Its shape:

1. A doubly linked list of symbols, each carrying `c`, `prev`, `next`, `len`
   (`word.rs:37-43`). A removed symbol is tagged `len = 0` rather than erased,
   so no index shifts (`word.rs:210`) and the final `retain` drops them all at
   once (`word.rs:249`).
2. A `QuaternaryHeap` of candidate merges, seeded once from every adjacent pair
   (`word.rs:163-178`).
3. Pop the best candidate, apply it, and push only the two pairs the merge
   creates: the one with the previous symbol and the one with the next
   (`word.rs:217-244`).
4. A popped candidate is validated before use rather than removed when it goes
   stale: skip it when its left symbol was removed (`word.rs:187`), when it has
   no right neighbour (`word.rs:191`), or when the pair it names is no longer
   the pair at that position (`word.rs:198-205`).

That is O(n log n) merges of O(1) work each, against our O(n) rescan per merge.

**The order must stay identical, and upstream makes it explicit.** `Ord for
Merge` (`word.rs:28-35`) inverts both comparisons, so the max-heap yields the
lowest `rank` first and the lowest `pos` on a tie. That is the same leftmost
rule our loop gets from `strict <`. Our implementation states this in a comment
and a test pins it, because a heap silently reversing a tie is exactly the
change a token gate over ordinary text would not catch.

**Intern the symbols, which removes the string key rather than optimising it.**
Upstream's merge table is `AHashMap<(u32, u32), (u32 rank, u32 new_id)>`
(`tokenizers/src/models/bpe/mod.rs:9`, `model.rs:19`), built once at load
(`model.rs:174-192`). A symbol is a vocabulary identifier, a pair is two of
them, and `MergeKey` has no counterpart upstream. Our `MergeRanks` is
`unordered_map<string, int32_t>`, so we build one key string per probe. Mirroring
upstream deletes that whole class of work.

Interning has one precondition, and upstream states it as a refusal: a merge
whose left token, right token, or concatenation is absent from the vocabulary is
`MergeTokenOutOfVocabulary` at load (`model.rs:180-189`). Our string form has no
such rule, so a merge table can currently produce a symbol that
`src/vllm/tokenizer/tokenizer.cpp::EncodePlain` then fails on at encode time
with `symbol "..." not in vocab`. Mirroring the upstream refusal moves that
failure from per-request to load, which is where upstream puts it and where a
user can act on it. **This is a behaviour change on malformed inputs and it must
be called out in review**: a checkpoint that loads today and fails on some
prompts would be refused at load instead. That is the upstream behaviour, so it
is the behaviour we mirror, but the implementing row records which committed
goldens it verified still load.

**The two callers differ and both must be carried.**
`src/vllm/tokenizer/tokenizer.cpp::EncodePlain` starts from single mapped
codepoints, each of which is in the vocabulary for a complete byte-level
alphabet. `src/vllm/tokenizer/tokenizer.cpp::EncodePlainSp` builds symbols
first, with byte-fallback decomposition and a `kUnk` sentinel that is
deliberately not a real symbol, and only then merges. Upstream does the same
split (`model.rs:382-460` builds the `Word`; `merge_all` runs after), so the
seam survives. The sentinel needs a reserved identifier that no merge can name.

**Rejected: a cap on pretoken length.** It changes the token identifiers we
emit, so it breaks the mirror on exactly the inputs it claims to protect.
Truncation is worse, because it changes the output silently. Nobody should
reach for this as a shortcut, which is why it is written here.

**Rejected on measurement: removing the per-probe `std::string` on its own.**
This was the low-risk option and it does not work. The same loop with one
reused key buffer, measured in the same harness on the same inputs, scored
31,076.04 ms on the 65,535-byte case against the current 28,746.70 ms, and
639.39 ms against 813.02 ms on 8,192 spaces. It is within noise, and slower on
the largest case. The allocation is not the cost. The rescan is. The separable
question therefore has a measured answer of no, and the two changes are one
change: interning removes the key because the algorithm no longer names one.

## Prototype evidence

A prototype of the design, built against this tree's sources, produced symbol
output **bit-identical** to `BpeMerge` on all ten inputs it was run against: the
six benchmark prompts, 8,192 spaces, 8,192 `~`, 8,192 `a`, 2,731 `中`, and the
65,535-byte `的` case. It is a prototype, not the implementation. It validates
staleness by comparing the merged string, where the implementation must mirror
upstream's identifier comparison (`word.rs:198-205`), and it uses
`std::priority_queue` where upstream uses a 4-ary heap. Neither difference
changes the order, and both are named so the implementer does not inherit them
by accident.

## Defence in depth

A length guard at the API boundary is worth having **after** the algorithmic
fix, not instead of it, and it is not in this row's scope. Two constraints bind
it if a later row adds one:

- It must **refuse**, with an explicit error naming the limit. Truncation would
  silently change the token identifiers we return, which is the same defect as
  the rejected cap.
- It belongs where a caller can be told, at the request boundary, not inside the
  tokenizer. `src/vllm/v1/engine/input_processor.cpp::ValidatePromptLen` is the
  wrong place, because it needs the token count that the expensive step
  produces.

After the fix, 64 KB encodes in single-digit milliseconds, so the guard protects
against a future regression rather than against today's cost. File it as its own
issue when the row lands.

## Dependencies

None. The row is self-contained host code on base `31f93787c`. It needs no
oracle run, no lease, no checkpoint mount, and no other row. The two goldens its
gates read are already committed. It does not block on
[#915](https://github.com/mudler/vllm.cpp/issues/915) and #915 does not block on
it, although closing this row removes the outlier that
[`qwen38-27b-bf16-gate.md`](qwen38-27b-bf16-gate.md) records.

## Work breakdown

Non-overlapping. W1 is the whole correctness argument and can be reviewed
before any algorithm changes.

| W | Deliverable | Reviewable on its own because |
|---|---|---|
| **W1** | The equivalence corpus and harness of `## Tests to port` item 1, plus the cost and growth assertions of items 2 and 3, all against UNCHANGED code | items 1 and 6 pass, items 2 and 3 are red. That red is the defect, recorded before anything moves |
| **W2** | The identifier-keyed merge table and the load-time refusal, with `BpeMerge` still doing the O(n^2) scan over identifiers | W1 stays exactly as green and as red as it was. Only the representation moved |
| **W3** | The heap merge, the linked symbol list, the comparator, the staleness checks | W1's items 2 and 3 turn green and nothing else changes |
| **W4** | The idle-host re-measure, the `## Outcome` section, `docs/STATUS.md` and `docs/BENCHMARKS.md` | the code is frozen; this is the record |

W2 before W3 is deliberate. Interning and the heap are separable in the tree
even though they are one design, and splitting them means a reviewer never has
to hold a representation change and an algorithm change in mind at once.

## Risks

| Risk | Why it is real | Control |
|---|---|---|
| A tie is resolved differently and one token identifier changes | The heap's comparator is where leftmost-wins lives, and a reversed tie is invisible on most text | Port `Ord for Merge` (`word.rs:28-35`) with its inversion; a direct tie case in `test_bpe.cpp`, which already pins `BpeSplit("ĠĠĠ", r2) == {"ĠĠ", "Ġ"}` |
| A stale heap entry is applied | Positions and neighbours change under the heap | Mirror all three validations (`word.rs:187`, `:191`, `:198-205`); a red-first case that reaches a stale entry |
| Interning refuses a checkpoint that loads today | The load-time vocabulary rule is new for us | Verify every committed golden still loads; record which ones in `## Outcome` |
| The `kUnk` sentinel collides with a real identifier | The SentencePiece path merges a value that is not a vocabulary entry | Reserve an identifier outside the vocabulary range and assert it |
| The equivalence corpus is too small to be evidence | The existing 64-entry corpus missed this defect at 54 bytes | The corpus below is required to contain the long-pretoken regime |

## Tests to port

Red first, in this order. Each states what it fails on today and what it must
prove after.

1. **Equivalence over a corpus, token identifiers only.** A CPU-tier binary that
   encodes a corpus through the committed goldens and compares against recorded
   identifiers. The corpus is the existing 64 entries, plus the six #1365
   prompts, plus real multilingual prose in Chinese, Japanese, Thai, Lao and
   Khmer, plus the single-class runs above at a size the suite can afford. This
   test **passes on today's code** and must keep passing. It is the correctness
   gate, and its value is that it cannot go green on a faster wrong answer.
2. **A cost bound that fails today.** Encode one pretoken of a stated size and
   assert an upper bound on wall time. This is red on `31f93787c` by three
   orders of magnitude, which is the margin that makes a wall-clock assertion
   honest on a contended runner. State the bound as an absolute figure with the
   margin argued, not as a ratio against a same-run baseline.
3. **A growth-shape assertion.** Encode at n and at 2n and assert the ratio sits
   below a bound that quadratic growth cannot satisfy and linearithmic growth
   can. This is the assertion that survives a slow runner, because both halves
   move together. It is red today at a ratio near 4.
4. **The leftmost tie.** Extend the existing `test_bpe.cpp` case into one that
   distinguishes the two orders on a longer symbol list.
5. **A stale-entry case.** A merge sequence where a queued candidate is
   invalidated by an earlier merge, constructed so that applying it produces
   different identifiers.
6. **The SentencePiece arm.** The same equivalence and cost cases through
   `tests/parity/goldens/tokenizer_mistral/tokenizer.json`, whose `"split":
   false` makes the whole prompt one word. Ordinary English prose is the input,
   because that is what fails.
7. **The load-time vocabulary refusal.** A merge table naming a token absent
   from the vocabulary is refused at load, with the missing name in the message.

The reachability mutation for the fresh reviewer: delete the call to the new
merge from `src/vllm/tokenizer/tokenizer.cpp::EncodePlain` in a scratch copy and
rerun the focused gate. A gate that stays green measured a class, not the
encode path.

## Gates

**This row is unusually cheap to gate, and the plan takes that seriously.** The
tokenizer is pure host code. Nothing here needs a GPU, a lease, or a checkpoint
mount, and the two goldens the gates need are already committed:
`tests/parity/goldens/tokenizer_qwen36/tokenizer.json` (248k vocabulary) and
`tests/parity/goldens/tokenizer_mistral/tokenizer.json`.

The CPU tier can prove, with no other host:

- Token-identifier equivalence before and after, on both tokenizer families,
  over a corpus that contains the failing regime.
- The tie-break order, the stale-entry handling, and the load-time refusal.
- That the cost bound and the growth shape are red before the change and green
  after.
- Through `test_tokenizer_parity`, `test_tokenizer_parity_mistral`,
  `test_tokenizer_parity_deepseek` and `test_tokenizer_parity_gpt4o`, that all
  four already-gated families still produce their recorded identifiers.

What the CPU tier cannot prove: the end-to-end TTFT of #1365's legs on GB10.
That needs the bf16 27B server on `dgx:gpu0` under an `rc` lease, and it is the
row's closing evidence rather than its correctness gate. Run it only after the
CPU gates are green.

Required before any speed figure in this spec is accepted: an idle-host,
same-binary A/B re-measure with the load average recorded, because every figure
above was taken between load average 130 and 190.

## Owed

Nothing yet. This row files no issue it does not fix.

## Stop conditions

- Stop and return `NEEDS_DECISION` if token-identifier equivalence fails on any
  corpus entry. A single changed identifier is a correctness regression, not a
  tuning question, and no speed result excuses it.
- Stop and return `NEEDS_DECISION` if the load-time vocabulary refusal rejects
  any committed golden. That would mean the mirror costs us a checkpoint that
  works today, and the trade is a developer decision.
- Stop if the fix needs a change to `src/vllm/tokenizer/pretokenizer.cpp`. The
  pretokenizer mirrors an upstream regex and is out of scope; a design that
  needs it is the wrong design.
- Do not take a GPU or an `rc` lease for anything in `## Tests`. If a gate seems
  to need one, it is measuring the wrong thing.
