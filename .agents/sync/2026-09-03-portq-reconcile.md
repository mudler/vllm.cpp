# The 290-entry queue against itself, and a verdict on the pin advance

Wave RECONCILE. Row `UPSTREAM-SYNC-HEADPIN`.
Issue: [#2764](https://github.com/mudler/vllm.cpp/issues/2764).
Spec: [`../specs/upstream-sync-portq-reconcile.md`](../specs/upstream-sync-portq-reconcile.md).
Tree read: `origin/main` at `f5b5b2c3c`.
Upstream read at revision, never in a working tree.

## 1. The verdict

**Three things stand between the current state and advancing the parity pin from
`5559679229` to `e126687a9a`. The PORT-NOW queue is not one of them.**

[`../oracles/vllm.md:58-70`](../oracles/vllm.md) already names four owed items.
This wave rules on which of them the rule actually requires:

| Owed item | Verdict | Why |
|---|---|---|
| 1. The 290-entry PORT-NOW queue, unworked | **Not a blocker** | "Reconcile" is not "port". §5. |
| 2. A declared token-exact gate at the target | **BLOCKER** | §2.1 |
| 3. Step-6 re-measurement of every moved denominator | **BLOCKER** | §2.2 |
| 4. A reading on `dgx:gpu0` | **BLOCKER** | §2.3 |

Nothing else blocks. In particular `qwen4_exp` not running at the candidate
([#2626](https://github.com/mudler/vllm.cpp/issues/2626)) is **not** a blocker
under the reading this wave takes, and §6 argues that and says what it costs.

### 1.1 What the advance consists of, once the three are cleared

The advance is a change of its own, and this wave does not make it. Its content
is already determined by committed records, so it is written down here rather
than re-derived later:

1. Build the target and **measure** `vllm.__version__` and the distribution
   version. Never transcribe. The runtime string is already measured once at
   `0.28.1rc1.dev132+ge126687a9`
   ([`2026-09-03-e126687-runhalf.md`](2026-09-03-e126687-runhalf.md)), and its
   `+g<sha>` segment prefixes `e126687a9a828d…`, so `assert_oracle_commit` is
   satisfiable. The distribution string is **not** measured.
2. Write the ` ```parity-pin ` block in
   [`../upstream-sync.md`](../upstream-sync.md): `vllm_commit`,
   `vllm_runtime_version`, `vllm_distribution_version`, and
   `flashinfer_version`, which moves 0.6.15.post1 → 0.6.18. The prose above the
   block carries the `transformers` floor 5.5.3 → 5.10.4
   ([`2026-09-01-cdefd9d.md:1823-1826`](2026-09-01-cdefd9d.md)).
3. Update the `oracle-pin` block in [`../oracles/vllm.md`](../oracles/vllm.md):
   `pin`, `pin_label`, `pinned_on`, `gateable`, `evidence`. Delete or fold its
   `## The candidate` section, which exists only because the pin had not moved.
4. Regenerate the golden dumps at the target and run the op, behavioural and
   model suites — `upstream-sync.md` step 6.
5. Re-measure every benchmark denominator baselined against FlashInfer
   0.6.15.post1, CUTLASS DSL 4.6.0, the `transformers` floor, and the
   `VLLM_ALLREDUCE_USE_FLASHINFER` default that `6a962071bd` vllm#52998 flips on.
6. Reconcile the recorded deviations inside this range —
   [`2026-09-02-e126687.md`](2026-09-02-e126687.md) §6 counts **three**, not
   `cdefd9d`'s four, because the narrower range excludes one.
7. Repair the two committed records the advance falsifies, which
   [`2026-09-03-portq2.md:311-330`](2026-09-03-portq2.md) §9 names: the ModelOpt
   fp8 `out_dtype` teaching at `.agents/porting.md:78`, and the
   `config/cache.py:111` anchor in six places, **one of them in product output**.
   Both are correct today and become wrong the moment the pin moves.
8. Tier the 333 INVENTORY entries and update `porting-inventory.md` §9.
9. Do **not** land [#2649](https://github.com/mudler/vllm.cpp/issues/2649)
   (`[83]`) before the pin passes `c0202c5603`, or the tree disagrees with its
   own pin — [`2026-09-03-portq3.md:218-223`](2026-09-03-portq3.md).

Item 9 is the only one that is an ordering constraint on a *port* rather than on
the advance itself, and it points the opposite way from the queue: `[83]` is a
gap **created by** the advance.

## 2. Why the three block

### 2.1 The token-exact gate

AGENTS.md §"Gates": "Correctness always comes first. Establish the declared
token-exact gate before you accept a performance result." The pin is the
reference frame every gate resolves against, so moving it without a token-exact
gate at the target leaves every correctness claim in the tree asserted against a
frame nothing has checked.

What exists is not that gate.
[`2026-09-03-e126687-runhalf.md`](2026-09-03-e126687-runhalf.md) ran
`facebook/opt-125m`, six prompts, 16 greedy tokens, and its own record refuses to
be read as a gate: `../oracles/vllm.md:63-65` — "The run above is six prompts at
16 tokens and **gates nothing**; it agreed with `tests/parity/goldens/opt_greedy`
exactly, which is informative and not a result." The agreement crosses a pin, a
device (sm_121a → sm_110) and a checkpoint materialisation, which is why the
report itself declines to bank it.

### 2.2 Step-6 re-measurement

This one is not an inference. AGENTS.md's own words are "reconcile every affected
row **and gate**", and `upstream-sync.md` step 6 says what reconciling an
affected gate means: "If benchmarks are baselined, the vLLM baseline must be
re-measured at TARGET before comparing."

Four denominators move under this candidate, and each moves gates that are
baselined against it: FlashInfer 0.6.15.post1 → 0.6.18, CUTLASS DSL 4.6.0, the
`transformers` floor 5.5.3 → 5.10.4, and `VLLM_ALLREDUCE_USE_FLASHINFER`, which
`6a962071bd` vllm#52998 flips **on** by default and which therefore moves the
vLLM production configuration used as the multi-GPU denominator. AGENTS.md
§"Gates" requires vLLM's production configuration as the denominator, so a
multi-GPU parity number carried across this advance would be quoting a
denominator that no longer exists.

**This blocker survives both readings of "reconcile every affected row",** because
it attaches to the word "gate", which the sentence names separately.

### 2.3 The `dgx:gpu0` reading

AGENTS.md: "An oracle is only gateable once it demonstrably builds and runs the
model." The candidate has demonstrated that on **one** device, and the report
says so: `runhalf.md:335-336` — "**C4. This is one device on one day.** Only
`thor:gpu0` was measured. Nothing here says the candidate builds or runs on
`dgx:gpu0`."

`dgx:gpu0` is where the binding 27B/32B grids were measured, so an oracle that
has never been built there cannot serve as their denominator. This is not a
formality: the pin's own history contains a build on `dgx:gpu0` that reported a
different version prefix from the recorded one and left #1185 open. A second
host is the axis on which that kind of surprise appears.

## 3. Method, and one defect that would have hidden a result

### 3.1 The queue, extracted by pattern

```console
$ awk '/^## 4\. PORT-NOW/{f=1;next} /^## 5\./{f=0} f' .agents/sync/2026-09-01-cdefd9d.md \
    | grep -E '^- ' | sed -E 's/^- `([0-9a-f]+)`.*/\1/'          # 315, 315 unique
$ git -C $VLLM rev-list --reverse 5559679229..e126687a9a | cut -c1-10  # 1465
$ grep -x -F -f portnow315.txt range10.txt                             # 290
```

Positive control on the same probe form: `grep -c -x -F ee11730751 range10.txt`
→ **1**; negative control `deadbeef01` → **0**. Scope: the pattern reads only
lines beginning `- ` inside `§4`, so an entry written in any other shape would be
missed; the count reproduces `cdefd9d`'s committed 315 and
[`2026-09-02-e126687.md`](2026-09-02-e126687.md)'s committed 290, which is the
check that licenses it.

The positional numbering `[1]`..`[290]` was then `diff`ed against every
`| N | \`sha\` |` row of all seven tranche reports. **The diff is empty.** That
is a positive control on the extraction and on the reports at once.

### 3.2 The parser defect, which failed toward "no relationship found"

The first extraction keyed the current file off `+++ b/<path>` alone. A commit
that **deletes** a file writes `+++ /dev/null`, so every hunk of every deleted
file was silently dropped. `[45]`→`[200]` then read **210 of 332**
where [`2026-09-03-portq7.md:394`](2026-09-03-portq7.md) §5.3 has **332 of 332**
— the exact revert reading as a 63% partial one, which is exactly the shape of a
finding a reviewer would argue about rather than reject.

The corrected parser falls back to `--- a/<path>`. It recovers 3346 removed lines
across the queue's diffs (20623 → 23969) and reproduces PORTQ-7's **332/332**,
**66/66** and **52/52** independently, from a different program. Note that
`[229]`→`[230]` and `[22]`→`[265]` read **66/66** and **52/52** under *both*
parsers: the defect is invisible on any pair where neither commit deletes a
file, so the two easiest checks a reader would run would both have passed.

**The defect is worth naming because of its polarity.** A scan that loses a class
of hunk reports fewer relationships, never more. Its failure mode is a clean
"nothing found", which is indistinguishable from a correct negative — the same
shape as a gate that stays green because it never ran.

### 3.3 Four scans, and the shared-file filter

All four are filtered by shared file, for the reason
[`portq7.md:400-408`](2026-09-03-portq7.md) §5.3 measured: two candidate pairs
scored 57% and 37.5% while sharing no file, and were a duplicated
`deepstack_num_level` idiom. The filter is part of the test.

1. **Exact inverse** (`A.plus == B.minus` and `A.minus == B.plus`).
2. **Byte-identical** (`A.plus == B.plus` and `A.minus == B.minus`).
3. **Directional** — of A's added lines, how many a later B deletes.
4. **Mirror** — of A's *deleted* lines, how many a later B *re-adds*. **New.**

Scans 3 and 4 were run twice: over the 290, and over the 1175 in-range commits
outside the queue.

**Scan 4 is the one PORTQ-7 identified as missing and did not run.**
[`portq7.md:431-433`](2026-09-03-portq7.md): "Neither shows up in §5.3, because
`[288]` and `[272]` **add a condition** rather than delete lines. A scan that only
measures deletion is blind to the commonest way upstream walks a change back."
The mirror does not catch an added condition either — only reading does, and
PORTQ-7's §5.4 did that reading — but it does catch the other blind spot of a
deletion-only scan: **A removes something and B puts it back.** That is the exact
shape of the C9 chain, and it is how `[54]`→`[115]` was found.

### 3.4 A fifth probe, which needs no pair at all

For every entry, what fraction of the lines it adds is present **verbatim in the
same file** at `e126687a9a`? This measures net effect directly, so it catches an
entry undone by two commits, or by a rewrite no single commit reverts.

**Median survival over the 272 entries with three or more added lines: 1.000.**
Six are below 0.50; three are below 0.20. Two of the six are a file relocation
(§4.3). The queue is, as a body, net-stable: the self-cancelling population is
small, and §4 enumerates it.

The probe's limit is stated rather than left implicit: verbatim survival is a
**lower bound on disturbance**. A line reformatted, renamed or moved to another
file reads as not surviving, and a semantic revert that shares no line reads as
surviving. It is a screen, not a verdict, and every low score below was read.

## 4. The net-effect ordering

### 4.1 Reproduced, not new

Every relationship PORTQ-7 §5 published reproduces here from an independent
extraction, at the same numbers:

| Relation | This wave | PORTQ-7 |
|---|---|---|
| `[45]`→`[200]` exact inverse, the **only** one in 290 | 332/332 | §5.2, §5.3 |
| `[229]`≡`[241]` byte-identical, the **only** such pair | identical | §5.1 |
| `[229]`→`[230]` | 66/66 | §5.3 |
| `[22]`→`[265]` | 52/52 | §5.3 |
| `[12]`→`[262]` | 8/8 | §5.3 |
| `[138]`→`[171]` | 53/64 | §5.3 |
| `[46]`→`[144]` | 26/64 | §5.3 |
| `[195]`→`[264]` | 8/23 | §5.3 |
| `[55]` ← `5789897aa4` ← `[162]` (C9) | 24/24 | portq2 §6.1 |
| `[201]` ← `ee11730751` | 7/147 | portq6 §5 |

The `[201]` row is worth one sentence. PORTQ-6 published 11 and corrected itself
to 7 ("This section first read … **Only 7 were**"). This wave's independent
extraction returns **7**, which corroborates the correction rather than the
original — and the first, defective parser here returned 13. Three programs, one
right answer, and the two wrong ones disagree with each other.
| `[228]`/`[236]`/`[238]` ← `9cef631f30` | 100% each | portq6 §5 |

**The brief for this wave described the queue-against-itself pass as one nobody
had done. That is not accurate, and the correction belongs here rather than in a
footnote.** PORTQ-7 §5 ran two of the four scans over all 290 and read the pairs
its scans could not see; PORTQ-6 §5 ran the out-of-queue check for its own
tranche. This wave's contribution is the two axes they did not cover — the mirror
scan and the systematic out-of-queue sweep over all 290 — plus the survival
probe.

### 4.2 New: three relationships, all on axes the prior scans could not reach

**(a) `[9]` → `[175]` → `7156c63bef`, and the last word is outside the queue.**

- `[9]` `bf2b45b5d6` vllm#42669 enables FA4 head-dim 256 on SM100 and gates it on
  a new `requires_local_attention` parameter.
- `[175]` `1f0e0bf612` vllm#52050 **deletes that parameter and disables hd256
  outright**, 20 days later. Directional score 14/65.
- `7156c63bef` vllm#52980 "[SM100] Hdim 256 optimized", 2026-08-26, **re-enables
  it** behind an entirely different gate: `uses_fa4_hd256_kernel` plus
  `_fa4_hd256_fallback_reason`, which refuses on attention sinks, logit
  softcapping, a quantized KV dtype, a KV block size not a multiple of 128,
  `mm_prefix` bidirectional attention, R-SWA, and DCP. Directional score against
  `[175]`: **10/15**.

`7156c63bef` is **in range and not in the 315**. `cdefd9d` §4 classifies it
`INVENTORY`. So the net state of FA4 hd256 at the target is set by a commit the
queue does not contain, and it is neither `[9]`'s gate nor `[175]`'s absence.

Novelty check with a positive control: `7156c63bef` and `52980` appear in **no**
tranche report; the control string `53500`, which portq6 does discuss, hits
portq6. Scope: the probe is a literal grep over the seven files for both the SHA
prefix and the PR number.

Consequence: this costs nothing today, because portq1 scopes
[#2640](https://github.com/mudler/vllm.cpp/issues/2640) to
`LookupMlaPrefillPriority` and calls both FA4 halves inert
([`portq1.md:120`](2026-09-03-portq1.md)). It costs later:
`BACKEND-CUDA-COMP-FA` must port `7156c63bef`'s gate, not `[9]`'s.

**(b) `[185]`'s FlashInfer half is deleted at `d6c2fec9fd`, and `[185]` is filed
`inert`.**

`[185]` `d1e3eee6fb` vllm#52188 adds 115 lines of FlashInfer DSpark DCP support
to `flashinfer_mla.py` and 77 to `test_mla_backends.py`. `d6c2fec9fd` vllm#53139
"[Cleanup][MLA] Remove FlashInfer DSpark DCP support", 2026-08-21, removes 121
and 62. Directional score over shared files: **163/312**, of which **105/105** is the
product file and 58/68 the test file. `d6c2fec9fd` is in range, outside the 315, and classified `IGNORE`
by `cdefd9d` ("the FlashInfer MLA backend is not ported").

This matters to the *label*, not to today's work. `inert` means, in the tranches'
own definition, that the surface exists and the new arm is gated on something
absent, so **the work becomes real the day that gate lands**. Half of `[185]`
never becomes real: the arm it adds does not exist at the target. `[185]`'s other
halves — `mla_attention.py` +29, `cp_utils.py` +21, `dflash/speculator.py` +41 —
survive and remain genuinely deferred.

PORTQ-5 already recorded a different out-of-queue relation for `[185]`
(`aeeb36b1f1` adds the refusal `[185]` deletes,
[`portq5.md:365-367`](2026-09-03-portq5.md)) and correctly told the porter to
take the post-`d1e3eee6fb` state. **That instruction is now insufficient**: the
target state is post-`d6c2fec9fd`, which is neither.

**(c) `[54]` → `[115]`, a restoration, which only the mirror scan sees.**

`[54]` `c8602c7906` vllm#50801 removes the SGL-kernel branch from
`dispatch_unquantized_gemm` and widens the dtype set. `[115]` `7ce84b99ce`
vllm#51379 "[CPU] Restore linear dispatch for small unquantized GEMMs" puts
**16 of `[54]`'s 29 deleted lines back**, seven days later, behind a new
`_CPU_SGL_GEMM_MAX_WEIGHT_BYTES` threshold and a separate AMX-FP16 ISA check that
`[54]` did not have.

It scores **1 of 46 on the directional scan** — `[115]` deletes essentially
nothing `[54]` added — and **16 of 29, 55.2%, on the mirror**. It is co-mentioned in no tranche report
(positive control: `[45]`/`[200]`'s pair hits portq5 and portq7 under the same
probe). Both entries are `NA surface-absent` here, so this is non-work today; the
finding is about the method, and about the fact that the net at the target is
`[115]`'s shape rather than the replay of either commit.

### 4.3 The false-positive class the sweep produced, and its control

`--no-renames` makes a moved file's old path read as fully deleted. Four hits are
this shape and none is a revert:

| Reads as | Actually |
|---|---|
| `[228]`, `[236]`, `[238]` ← `9cef631f30` | `run_batch.py` moved to `entrypoints/launchers/`; **both** paths exist at the target |
| `[11]` ← `fcdc7c2e9c` | spec-decode E2E tests reorganised into subdirectories |
| `[24]` ← `b8165e5e58` | exception handlers consolidated into `serve/utils/error_response.py` |

Each was checked by confirming the destination path exists at `e126687a9a`, not
by reading the subject line. PORTQ-6 §5 found the `run_batch.py` case first and
named it correctly as one path move reading as three reverts.

### 4.4 The largest ordering fact in the queue, which no tranche stated whole

**Twenty of the 290 entries edit `vllm/config/vllm.py`, and `[262]`
`4aab2b0ebe` vllm#53183 "Use MRV2 for all models by default" rewrites the
mechanism they edit.** It deletes `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` and
`default_v2_model_runner_architectures()` outright and replaces per-architecture
opt-in with `_get_v1_model_runner_unsupported_features()`.

Directional scores of earlier entries against `[262]`: `[12]` **8/8**, `[217]`
**4/12**, `[203]` **40/235**, `[138]` **10/64**. `[221]` scores only 10/762, because
`[262]` **relocates** its branch rather than deleting it: `_is_dflash2_draft` is
present in `vllm/config/vllm.py` at the target, moved into
`_get_v1_model_runner_unsupported_features`. `[12]` is the same shape carried
further — its PCP condition survives at `:2616` as
`unsupported.append("prefill context parallel")` while its own `ValueError`
string is gone, so even the entry that scores 8/8 keeps its semantics and loses
its shape. `[202]` scores 0/26 directionally; PORTQ-6 §5 establishes its net-zero
relation to `[262]` on different evidence and this wave does not restate it.

PORTQ-6 recorded `[202]`→`[262]` and `[203]`→`[262]`; PORTQ-7 recorded
`[12]`→`[262]`. **Nobody stated the cluster.** Its consequence is one rule:
anything ported from `[12]`, `[138]`, `[159]`, `[171]`, `[202]`, `[203]`, `[217]`
or `[221]` in that entry's own shape lands at a call site `[262]` deletes.

Verified absent at the target, each with a present control through the same
probe: `_validate_mrv1_piecewise_cudagraph` **absent**,
`MRV1_UNSUPPORTED_PIECEWISE_CUDAGRAPH_ARCHITECTURES` **absent**,
`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` **absent**; `ROCM_DEFAULT_MRV1_ARCHITECTURES`
**present**, `_orient_fused_weight` **present**, `is_fused_checkpoint_transposed`
**present**, `uses_fa4_hd256_kernel` **present**.

### 4.5 The three sets

**Non-work against any tree** — porting these produces a tree that disagrees with
the target:

| Entries | Reason | Source |
|---|---|---|
| `[45]` + `[200]` | exact inverse, net zero | portq5 §5.2, portq7 §5.2 |
| `[229]` ≡ `[241]` | byte-identical; one change counted twice | portq7 §5.1 |
| `[202]` (with `[262]`) | net zero | portq6 §5 |
| `[252]` (with `[272]`) on non-ROCm | net zero | portq7 §5.4 |
| `[249]`, `[275]`'s `partial_tail_offloads` half | born and died in the window | portq7 §3 |
| `[159]` | half of a net-zero pair straddling the pin | portq4 §3 |
| `[12]`, `[138]` in their own shape | `[262]`/`[171]` delete them | §4.4, portq7 §5.3 |
| `[46]` in its own shape | `[144]` deletes the heuristic | portq7 §5.3, portq4 §4 |
| `[54]`, `[175]` in their own shape | `[115]` / `7156c63bef` supersede | §4.2 (c), (a) — **new** |
| `[185]`'s `flashinfer_mla.py` half | `d6c2fec9fd` deletes it | §4.2 (b) — **new** |

Two rows in that table are this wave's reading rather than a source's, and the
difference is stated so nobody cites the wrong thing. PORTQ-7 §5.3 calls
`[46]`→`[144]` "a follow-up correction, **not** a revert", which is right about
the pair; the "non-work" claim here is narrower and comes from reading the diff:
`[144]` deletes `_orient_fused_weight`'s shape-inference body verbatim and
replaces it with an explicit `is_fused_checkpoint_transposed` flag, so `[46]`'s
heuristic is absent at the target and PORTQ-4 §4 independently calls that
heuristic "the shape-sniffing heuristic upstream deleted". `[138]` is the same
shape, one step longer: `[171]` deletes it and `[262]` then deletes `[171]`'s
replacement, verified by three absent symbols against four present controls in
§4.4.

**Fixed order** — these must be taken in the stated sequence or not at all:

| Order | Consequence of getting it wrong | Source |
|---|---|---|
| `[145]` → `[167]` → `[230]` (KV layout 4→5→6) | a half-migrated layout no upstream revision had | portq4 §6, portq6 §6.1, portq7 |
| `[230]` → `[241]` | `[230]` without `[241]` **regresses** `[229]` | portq7 §5.1 |
| `[156]` → `[129]` | `[156]` alone lands dead code | portq4 §4 |
| `[88]` → `[133]` | `[133]` alone turns over-allocation into a throw | portq4 §6 |
| `[106]` → `[107]`/`[114]` | same transition block | portq3 §7 |
| `[31]` → `[33]` | `[33]` is a hook into `[31]` | portq1 §8 |
| `[282]` → `[288]` | `[282]` alone lands a ceiling upstream rejected in 2 days | portq7 §5.4 |
| `[214]` → `[216]` | `[216]` removes 9 lines `[214]` adds, 5h later | portq6 §5 |
| pin advance → `[83]` | `[83]` before the advance makes the tree disagree with its own pin | portq3 §4 |
| `[262]` last of its cluster | anything else lands at a deleted call site | §4.4 — **new** |
| `7156c63bef`, not `[9]` or `[175]` | any other target is a gate upstream replaced | §4.2 (a) — **new** |

**Must port together** — one unit, whatever the queue positions say:

- `[152]` with `[158]`, `[216]`, `[231]`, `[233]`, `[235]` — the Dots3 NOTE
  model add (5806 added lines, 28 files) and its five in-range amendments, all
  of which edit files `[152]` creates. `[235]` alone rewrites 85 of `[152]`'s
  lines across four `dots3_note/nvidia/` files. The inverse counts are small
  because these are additions beside `[152]`, not withdrawals of it; the unit is
  the model, and the shared-file cluster is what shows it.
- `[10]` with `[278]`'s `_try_load_fp8_indexer_wk` half (portq7 §5.5).
- `[65]` with the rest of `KV-MAMBA-ALIGN` (portq2 §4).
- `[135]` with `[147]` (portq4 §6.4).
- `[290]`'s KV half with stage 6 `[230]`, on which it is literally blocked
  (portq7 §5.5).

## 5. "Reconcile every affected row" means the record, not the port

Both readings are defensible; this wave rules for the record reading, on four
grounds and with one condition.

**One. The rule chose a different verb from the one it uses elsewhere.**
AGENTS.md says "port" when it means port: "Port the upstream tests in the same
change"; `upstream-sync.md` step 5 is "**Port** the PORT-NOW queue". Step 7 is
"**Advance the pin**", and the sentence that governs it says "**reconcile**".
Reading a synonym into a document that distinguishes the two words costs the
distinction.

**Two. The port reading makes the pin unadvanceable, and the protocol says
otherwise in as many words.** `upstream-sync.md`: "A cycle that stalls mid-way
keeps the old pin and records what's left in the report ('carry-over') — the next
cycle picks it up." That describes a cycle that advances with work outstanding.
`porting-inventory.md`'s own header says the same thing from the other side:
"When the upstream sync point advances and brings new features, add them here
(with their vLLM PR references) in the right tier — the inventory tracks what
vLLM has, **even for things we haven't scheduled**." That instruction has no
meaning under a reading in which the pin cannot advance past an unported
feature. A pin that moves only when the gap set is empty is not a pin; it is a
completion certificate, and the project would have none.

**Three. AGENTS.md's own definition of a pin is a reference frame, not a claim of
completeness.** "**Parity pin (post-MVP)** — one repo-wide vLLM commit. 'We have
feature X' always means 'X as of the pin'." The gap record is
`porting-inventory.md` and the row matrices, not the pin. Advancing the pin
re-expresses 61 tracked gaps against a new frame; it does not assert they closed.

**Four. The port reading holds the advance hostage to debt the advance does not
own.** Fifty-one of the 290 entries rest **wholly** on a pre-pin hole — the
surface upstream edits already existed at `5559679229` and was never ported here
— with six more resting on one in part
([`portq7.md:321-333`](2026-09-03-portq7.md),
[`portq6.md:49-58`](2026-09-03-portq6.md), portq4 §5, portq3 §5). That 51 is a
sum of five tranche figures and **portq5's nine are never enumerated**, so read
it as the reports' own arithmetic rather than as a list this wave checked. Those gaps are
open against the **current** pin too. PORTQ-6 states the principle at
[`:61-62`](2026-09-03-portq6.md): "a gap the advance owes cannot rest wholly on a
hole the advance does not own."

**The condition, and it is met.** The record reading is only worth anything if
the records are true. Measured against the seven reports:

- **61 REAL_GAP entries, and every one carries an issue.** Zero exceptions.
  (A 62nd entry, `[152]`, carries `REAL_GAP` inside a compound label counted as
  `ALREADY_SATISFIED` in portq4 §1; its Dots3 half rides
  [#699](https://github.com/mudler/vllm.cpp/issues/699). 62 − 1 = 61 reconciles
  the two counts exactly.)
- **57 distinct issues**, because five pairs are deliberately one unit:
  `[107]`+`[114]` → #2652, `[129]`+`[156]` → #2685, `[55]`+`[162]` → #2657,
  `[133]` rides #2650, `[290]` splits across two owners.
- **Four issues name no owning row**: #2695 `[167]`, #2699 `[164]`, #2723
  `[230]`, #2735 `[278]`. In every case the report states that no row exists
  rather than inventing one, which is the disposition AGENTS.md prescribes
  ("Write a dash when no row owns it yet and a spec lists it under `## Owed`").
  #2723 is named individually under `## Owed` in
  `../specs/upstream-sync-portq6.md`. The other three are covered by the generic
  `## Owed` clause the portq5 and portq7 specs carry — "Porting the real gaps
  this wave names… each carries its own issue" — which satisfies
  `scripts/check-agent-record.py` as it is written.

So the record obligation is **met**, with one narrow tightening owed: the change
that advances the pin should name #2695, #2699 and #2735 individually under its
own `## Owed`, so that a reader who never opens a tranche spec still meets them.
That is a line of prose, not a blocker.

## 6. Gateability, twice, because the two questions are different

**The general oracle.** `e126687a9a` builds from source, ships a working
extension, and serves `facebook/opt-125m` to greedy tokens on FLASH_ATTN/FA2 in
both eager and compiled configurations, on `thor:gpu0`, 2026-09-03
([`2026-09-03-e126687-runhalf.md`](2026-09-03-e126687-runhalf.md), #2611). That
is measured, and it satisfies the sentence "demonstrably builds and runs the
model" on the permissive reading, on one device. `../oracles/vllm.md:22` still
records `gateable = yes` against `5559679229` and deliberately does **not** claim
it for the candidate. This wave does not change that record, because the change
that advances the pin is the one that owns it.

**`qwen4_exp` specifically: not gateable, and it is not a pin blocker.** The QSA
indexer's `cooperative_topk` refuses to launch —
`cooperative_topk.cu:46, cluster misconfiguration` — and the cause is honestly
unestablished: the 16-CTA hypothesis was tested and the A/B was **inconclusive
rather than negative**, because `num_rows` at that call site is a scoring-row
count and the knob did not move the variable it was chosen to move
([`runhalf.md:288-302`](2026-09-03-e126687-runhalf.md)). Thor's actual maximum
cluster size was never queried. Separately, every published safetensors arm
exceeds the largest fleet box, and upstream marks all three Qwen4Exp
architectures `is_available_online=False` at this very revision.

Why this is not a blocker: `MODEL-MM-QWEN4-EXP` cannot be *gated* at this
candidate, so the row keeps whatever state it has and #2626 stays open under it.
But that row's gate is not gateable at `5559679229` either — the architecture is
not registered there at all. **The advance strictly improves that row's position
and worsens nothing**, so refusing the advance on its account protects nothing.
Holding the pin at a revision where the model does not even exist, in order to
avoid moving to one where it exists and one kernel will not launch, is a strictly
worse position for the row that motivated the move.

The honest cost is that the advance does not deliver what it was wanted for, and
`../oracles/vllm.md:52-54` already says so: "a pin here would carry a registered,
importable, executable vLLM that still cannot serve `MODEL-MM-QWEN4-EXP`'s own
model on this fleet." That sentence should survive the advance, moved from the
candidate section into the record of what the new pin does and does not buy.

## 7. What could not be determined

- **Whether `7156c63bef`'s `INVENTORY` disposition is correct** at the tree.
  This wave establishes that it is the target's last word on FA4 hd256 and that
  no tranche saw it. It did not re-derive its disposition against `src/`, and
  `BACKEND-CUDA-COMP-FA` owes that reading.
- **`[114]` ← `ac2ae8798c`**, 5 of 12 deleted lines re-added in
  `streaming_parser_engine.py`. The file is a transition table with heavily
  repeated line shapes, so the score is as consistent with a duplicated idiom as
  with a restoration. Not resolved, and not claimed; #2652 already owns `[114]`.
- **Whether the generic `## Owed` clause satisfies AGENTS.md's intent** for
  #2695, #2699 and #2735, as opposed to satisfying its checker. §5 rules it
  sufficient and names the tightening; a reviewer may read it the other way, and
  the cost of the tightening is one line either way.
- **The honest denominator of the queue.** PORTQ-7 §10 says the figure is below
  290 and did not compute it. This wave enumerates the non-work set (§4.5) but
  did not recompute a denominator, because the tranches' labels are not uniform
  enough across compound entries to sum without re-reading all 290.
- **Anything requiring execution.** No build, no test, no lease, no device. Disk
  on the developer box ran at 98-99% throughout, and no build tree was created.

## 8. Carry-over

**The pin did not advance**, and nothing here is a reason to move it beyond
clearing the three blockers of §2. It remains
`5559679229bc961848b121ccdeaa8fa5d79bec98`.

- **R1.** §2's three blockers are unstarted. Each is a measurement on hardware.
- **R2.** The three new relationships of §4.2 are recorded here and in no matrix
  row. They are properties of the queue, and #2611 is the issue that holds it.
- **R3.** One record hygiene note for whoever writes the next sync issue: the
  PORTQ specs open with ``Row: `UPSTREAM-SYNC-HEADPIN` — inherited…``, and
  `scripts/agent-issue-index.py`'s `ROW_LINE` regex requires the line to end at
  the closing backtick. Copying the spec's form into an **issue body** makes the
  index resolve the row as a dash. This wave's own issue #2764 did exactly that
  and was corrected; the fix is verified in the regenerated snapshot.
- **R4.** The tightening of §5: name #2695, #2699 and #2735 individually under
  the advancing change's `## Owed`.
