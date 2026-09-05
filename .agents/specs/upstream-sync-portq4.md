# Sync cycle `e126687a9a`, wave PORTQ-4

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, re-verified at `a700e8da6`. `scripts/check-agent-record.py` passes
on a `Row:` line whether or not the row resolves, so this note is here to stop a
reader taking it for a matrix reference. The issues this wave cites are carried
under `## Owed` below, which is the route AGENTS.md gives when no row owns the
work.
Issue: [#2680](https://github.com/mudler/vllm.cpp/issues/2680).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns
the 290-entry queue, and [`upstream-sync-portq3.md`](upstream-sync-portq3.md),
whose method this wave continues unchanged.

## Now

**Done.** 40 of 40 re-derived: **11 ALREADY_SATISFIED, 11 REAL_GAP,
18 NOT_APPLICABLE** (14 `surface-absent`, 4 `inert`). **29 of the 40 disagree with
the recorded PORT-NOW disposition.** A fresh review of #2702 corrected one label
(`[132]`, published as a false `NOT_APPLICABLE`; see the report's §6.10) and three
citations. Nine issues carry the eleven gaps
([#2683](https://github.com/mudler/vllm.cpp/issues/2683),
[#2684](https://github.com/mudler/vllm.cpp/issues/2684),
[#2685](https://github.com/mudler/vllm.cpp/issues/2685) which covers two entries,
[#2688](https://github.com/mudler/vllm.cpp/issues/2688),
[#2689](https://github.com/mudler/vllm.cpp/issues/2689),
[#2693](https://github.com/mudler/vllm.cpp/issues/2693),
[#2694](https://github.com/mudler/vllm.cpp/issues/2694),
[#2696](https://github.com/mudler/vllm.cpp/issues/2696),
[#2706](https://github.com/mudler/vllm.cpp/issues/2706), and the pre-existing
[#2650](https://github.com/mudler/vllm.cpp/issues/2650) which already owns
`608c12473f`); four deferred entries, one deferred half, and one un-owned pre-pin gap are carried
on [#2700](https://github.com/mudler/vllm.cpp/issues/2700), and a pre-pin
correctness hole found on the way is
[#2697](https://github.com/mudler/vllm.cpp/issues/2697).
Report: [`../sync/2026-09-03-portq4.md`](../sync/2026-09-03-portq4.md).

**Nothing was executed.** No build, no test run, no GPU, no lease. An
`ALREADY_SATISFIED` means the code implements the behaviour, not that a gate
observed it.

The pin did **not** advance and nothing read here is a reason to move it. The
active parity pin remains `5559679229bc961848b121ccdeaa8fa5d79bec98`.

## 1. Scope

**One question, asked forty times.** For each of PORT-NOW entries 121 to 160 of
`5559679229..e126687a9a` in upstream commit order, what is true **of this tree**?

`.agents/sync/2026-09-02-e126687.md` §4 says in as many words that **no
disposition in it is re-derived**: all 1002 are carried by SHA from
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md). A `PORT-NOW`
label is a prior wave's reading of an upstream diff. It is evidence about
upstream and says nothing about what this tree has.

Three tranches have now measured that difference and published the number:
**34 of 40, then 31 of 40, then 32 of 40 labels did not hold** against this tree,
and the record was accurate about upstream in all one hundred and twenty. This
wave asks the same question for forty more.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 and 1465 counts, and all three landed slices positionally,
   before anything is read off it.
2. One label per entry against the tree at `a700e8da6`, each carrying a verified
   `path:line` or the searches that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming a row that **exists**.

Out of scope: porting anything, advancing the pin, entries 1-120 and 161-290, and
any measured number. Nothing in this wave is executed.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l          # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                              # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | sed -n '121,160p'
```

`portnow.txt` is the 315 SHAs of `../sync/2026-09-01-cdefd9d.md` §4. The `1465`
and the `290` are the counts `../sync/2026-09-02-e126687.md` §4 publishes, so the
extraction reproduces a committed number before it is used.

**The ordering check is stronger this time, because there are now three landed
tables to reproduce.** Positions 1-40, 41-80 and 81-120 of this derivation are
compared byte-for-byte against the SHA columns of
[`../sync/2026-09-03-portq1.md`](../sync/2026-09-03-portq1.md),
[`portq2`](../sync/2026-09-03-portq2.md) and
[`portq3`](../sync/2026-09-03-portq3.md). All three must match exactly,
contiguous and disjoint, or this wave stops — a mismatch would invalidate the
landed tranches, not just this one.

### 2.2 Three labels, and a named sub-shape for the third

Unchanged from PORTQ-3 §2.2. `ALREADY_SATISFIED` cites the `file:line` or symbol
that satisfies the entry. `REAL_GAP` says what is missing, roughly how big, and
who owns it. `NOT_APPLICABLE` splits:

- **`surface-absent`** — the commit edits something this tree does not carry.
  Discarded work; there is nothing to port until the underlying feature is.
- **`inert:<gate>`** — the surface is here and the new arm is gated on something
  unreachable in every configuration this tree builds. **Deferred** work, so the
  gate is named.

**A commit's two halves can take different labels**, and PORTQ-3 §6.6 is why:
calling a whole commit `inert` when half of it has no surface at all promises a
later reader that porting the named gate makes the entry real, and for that half
it does not. Each half is labelled where they differ.

### 2.3 Upstream is read at a revision, never in a working tree

Every upstream citation is `git -C ${VLLM_SOURCE} show <rev>:<path>` or
`git show <sha>` for the diff. An anchor read in a working tree is wrong at the
pin.

### 2.4 Absence is never concluded from one grep

PORTQ-3 §2.3, carried without change, because the failure it names is the one
this method keeps paying for. Four distinct mechanisms have produced false zeros
in this pass: grepping *upstream's* identifier where this tree renames on the way
in; a malformed ERE, in which `grep -rniE "a\|b"` reads `\|` as a literal pipe;
an unquoted `--include=*.h` eaten by zsh, returning silence at rc 0; and source
text wrapping across a line break so the pattern cannot match.

So every claim of absence carries four things: two differently spelled searches
in at least two locations, searching the **concept** in this tree's vocabulary
rather than upstream's spelling; **a positive control run through the identical
probe form**, pointed at something known to exist and confirmed found; and the
**scope** of the zero, because an unstated scope makes a zero unfalsifiable.

### 2.5 Git is asked whether it already triaged the entry

`git log --oneline --all --grep '<sha>'` and a `#<PR>` search over `.agents/`,
`src/` and `include/`, both run with a positive control, before any diff is read.
A prior reading corroborates or contradicts, and either is cited.

### 2.6 The pre-pin check comes before the diff

PORTQ-3 §5 Shape D: an entry can be a hole that **predates the pin entirely**,
where upstream already carried the surface at `5559679229` and the commit only
moves it. `git show 5559679229:<path>` answers that for less than the cost of
reading the diff, and a commit-range queue structurally cannot see the shape.

### 2.7 Ordering is checked on `%ct`, never `%cd`

PORTQ-3 §2.1 retracted the opposite claim. The range contains **zero merges** and
committer timestamps are strictly monotone across all 290 entries. Rendered local
dates are timezone-local and invert without meaning. `git log -1 --format=%ct`
piped to `sort -c -n` is the whole check.

### 2.8 The reading is delegated, and the citations are re-read

Six fresh readers take five to nine entries each, clustered by surface. The
operator prints every cited `path:line` again before it lands, and re-reads every
upstream diff behind a `REAL_GAP`. PORTQ-1 corrected three citations this way and
PORTQ-3 corrected one.

## 3. Risks

- **A false zero lands as a `NOT_APPLICABLE`.** §2.4 is the whole mitigation, and
  its known hole is that a re-read cannot check a claim of absence — there is no
  citation to print again. Recording the searches is what gives a reviewer
  something to falsify.
- **A carried claim is quoted rather than checked.** PORTQ-3 §6.2 found the same
  false statement rediscovered three times because every wave wrote a new report
  and none edited the one that misled it. This wave annotates the source in place
  where it falsifies one, keeping the original text and adding a dated line.
- **An issue names a row that does not exist.** `check-agent-record.py` passes on
  a `Row:` line whether or not the row resolves, so every owner is grepped out of
  its matrix by hand and cited with its `path:line`.
- **Concurrency.** PORTQ-5 (#2679) works entries 161-200 at the same time. This
  wave writes its own report and its own spec and touches no file that wave owns.
  Any annotation to `../sync/2026-09-01-cdefd9d.md` lands at entries this tranche
  owns, hundreds of lines from that wave's hunks.

## 4. Tests and gates

This wave adds no product code and no test. Its gate is that the derivation
reproduces four committed numbers before anything is read off it:

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l    # 1465
$ comm -12 portnow.txt range.txt | wc -l                                # 290
$ diff <(portq1 table SHAs) <(sed -n '1,40p' ordered.txt)               # empty
$ diff <(portq2 table SHAs) <(sed -n '41,80p' ordered.txt)              # empty
$ diff <(portq3 table SHAs) <(sed -n '81,120p' ordered.txt)             # empty
$ for s in $(cat ordered.txt); do git log -1 --format=%ct $s; done | sort -c -n
```

Where an annotation is added to `../sync/2026-09-01-cdefd9d.md`, the SHA
extraction is re-run afterwards and must still yield **315**, proving the
annotation did not perturb the list it annotates.

`scripts/agent-preflight.sh --staged` runs before the commit. Its output is read
for `gate(s) failed` and `NOT a green` rather than its exit code, which is 0 even
when it prints failures.

## 5. Evidence

[`../sync/2026-09-03-portq4.md`](../sync/2026-09-03-portq4.md) carries the forty
labels, the searches behind every absence, the disagreement count, and what could
not be determined.

## 6. Stop conditions

- Either denominator, any of the three landed slices, or the `%ct` ordering fails
  to reproduce. That invalidates the landed tranches too, so the wave stops and
  reports rather than re-deriving around it.
- An entry cannot be settled without a build or a GPU run. It is recorded as
  undetermined with the reason; nothing in this wave is executed.

## Owed

**[#2700](https://github.com/mudler/vllm.cpp/issues/2700) carries `Row: -`**, so
AGENTS.md requires a spec to list what it owns under `## Owed`. That is this
list, and it is the obligation the `Row: -` line exists to carry —
`check-agent-record.py` passes on the line either way, so the listing is the only
thing that actually discharges it.

**The deferred entries #2700 collects** — four whole entries plus one half, each deferred until its gate
is ported, never before:

- `[126]` `3e174bb73c` — gate: `index_share_for_mtp_iteration` plus a draft head
  exposing `set_skip_topk` / `compact_topk_indices`
  (`include/vllm/model_executor/models/glm_moe_dsa.h:132`, zero consumers).
- `[132]`'s `mla_attention.py` half — gate: fp8 MLA prefill and DCP
  (`include/vt/ops.h:5217-5219`,
  `include/vllm/model_executor/layers/attention/mla_chunked_context.h:311-313`).
  Its `cache_kernels.cu` half is **not** deferred; it is
  [#2706](https://github.com/mudler/vllm.cpp/issues/2706).
- `[134]` `ce07118669` — gate: `MambaSpec::mamba_cache_mode == "align"`
  (`include/vllm/v1/kv_cache_interface.h:469`; all five production constructions
  leave it `"none"`).
- `[142]` `e3fe212eaf` — gate: a consumer for `KVCacheManager::take_new_block_ids`
  (`include/vllm/v1/core/kv_cache_manager.h:232`).
- `[147]` `f067737b2a` — gate: the `OffloadingConnector` worker half
  (`src/vllm/v1/kv_offload/kv_connector.cpp:401-412` to
  `src/vllm/entrypoints/model_loader.cpp:1176`, which throws on every device).

**Also owed by #2700, and needing a row before it can be ported:** KV block
zeroing, a pre-pin gap that queue entries `[142]`, `[155]` and PORTQ-3's `[90]`
all point at. No row owns it today.

- Porting the real gaps this wave finds, each on its own issue.
- Entries **161-290** of the queue, owed by
  [#2611](https://github.com/mudler/vllm.cpp/issues/2611); 161-200 are
  [#2679](https://github.com/mudler/vllm.cpp/issues/2679).
- The pin advance itself, owed by
  [`upstream-sync-headpin-runhalf.md`](upstream-sync-headpin-runhalf.md).
