# DSA top-k device kernel — remove the literal selection bounds

**Issue:** [#505](https://github.com/mudler/vllm.cpp/issues/505).
**Row:** `MODEL-TEXT-deepseek-v4-deepseek-v4-for-causal-lm` (`DeepseekV4ForCausalLM`, ✅).
**Claim:** `CLAIM-DSA-TOPK-BOUNDS`.
**Base:** `origin/main` @ `6db04e7bfc886c58c22a089381fbf9277f318ee2`.
**Pinned oracle:** `${VLLM_SOURCE}` @ `5559679229bc961848b121ccdeaa8fa5d79bec98` (vLLM 0.26.0.dev0).

---

## 0. Scope

`DsaTopkKernel` (`src/vt/cuda/cuda_deepseek_v4.cu:624-669` pre-fix) sized two
thread-local arrays by literal:

```cpp
bool chosen[512];    // indexed [0, n) where n is the candidate-window length
int64_t picked[64];  // written [0, topk)
```

`topk` is the caller's `index_topk` — **512 on V4-Flash, 1024 on V4-Pro** — so
`picked[64]` was 8x and 16x too small, and `chosen[512]` overflowed on any window
wider than 512. The overflow branch is the `n > topk` path. Neither bound was
asserted and neither derived from the config.

Found while assessing #504 (DeepSeek-V4-Pro), whose `index_topk` of 1024 made the
mismatch impossible to miss.

## 1. Why it was latent, stated precisely

Not a shipped defect at the time of filing. `dsa_dense = (be.gguf != nullptr)`
(`deepseek_v4.cpp:668`) forces `is_indexer` false on the real keep-quant GGUF
path, so the shipped Flash run never calls the indexer — the kernel was exercised
only at the collapsed synthetic geometry, where `topk` is small by construction.
Every pre-existing device case ran at `topk=3, nk=5`, which is why the bound was
invisible to the gate.

It mattered anyway because the real-geometry DSA sparse path is a named residual
on this row: the moment that residual lands, these literals become a silent
thread-stack overflow at the real `index_topk` rather than a loud failure.

## 2. Defect proof

The pre-fix kernel body, transcribed verbatim with its literals and driven at the
real V4-Flash width (`topk=512`, `nk=600`, so `n > topk`), under ASan:

```
==3748575==ERROR: AddressSanitizer: stack-buffer-overflow
WRITE of size 1 at 0x7a20a0e00490 thread T0
    #0 OldKernelRow old-overflow.cpp:26
```

Line 26 is the `chosen[s] = false` initialization loop. `picked[64]` then takes
512 writes in the same call. The reproduction is scratch and not committed; the
committed gate is §4.

## 3. Fix — two passes, no scratch

Replaced the mask-plus-picks approach with a threshold formulation over the same
total order the host reference sorts by (`DsaTopkSelect`: logit desc, then index
asc — a total order because candidate indices are distinct):

- **pass 1** walks the order downwards `topk` times to land on the topk-th best
  element, the selection threshold;
- **pass 2** scans the window once in ascending index order and emits every
  element outranking-or-equal to that threshold.

Exactly `topk` elements satisfy pass 2 under a total order, and they come out
already in ascending key order, so the `O(topk^2)` emit sort disappears along with
the buffers. **No per-thread scratch, no bound, no configurable limit.** Cost is
unchanged at `O(topk*n)` for pass 1 and cheaper than the revision it replaces,
which also paid the emit sort.

That comparison is to the old kernel **only**, and is not a claim of fitness for
the real geometry (corrected per the review, #552 finding 5). Pass 1 is one thread
per token row with a dependent global load per iteration, so at V4-Pro's
`index_topk=1024` the last row of a 4k prompt is on the order of 4.2M serial
loads in a single thread. Asymptotically unchanged from the pre-fix kernel, and
out of scope here — §5 names the real-geometry path as the residual that owns it.

Two defensive additions, neither load-bearing: pass 1 stops if no strictly-worse
element is found, and pass 2 carries a `w < topk` bound. **They are
belt-and-braces, not a defence against a known way to overrun the row** — the
emit count is bounded by construction. Pass 2's predicate is satisfied by exactly
`rank(th)` elements and `rank(th) <= topk`, and a NaN never satisfies it either
because `better(x, NaN)` is false, so `w` cannot exceed `topk` for any input. The
review proved this by removing the bound and by weakening it to `w <= topk`: both
left the device suite 4/4 SUCCESS and a 3M-shape fuzz clean. The original wording
here named a NaN write-past-row as the defended failure class, which was an
overclaim (#552 finding 1). The guards stay — they are free — but a comment that
overstates what it protects is worse than none, because the next reader trusts it.

## 4. Evidence

**Committed gate** — `tests/vllm/models/test_cuda_deepseek_v4.cpp`, three new
cases, all comparing device output against the independent host reference
`DsaTopkSelect` (std::stable_sort based — a genuinely separate implementation, so
the comparison is not a shared-helper tautology):

| case | shape | what it pins |
|---|---|---|
| real `index_topk` widths | `(topk,nk)` = (65,80), (512,600), (1024,1200) | just past the old `picked[64]`, then both shipped widths, each with `n > topk`; also asserts no `-1` leaks and strictly ascending emit |
| tie-heavy rows | topk=128, nk=300, quantized logits | the total order's tie-break, which distinct random logits cannot exercise |
| offset window | topk=512, nk=900, `ws=137` | the old code indexed its mask by `s - s0` and its picks by absolute `s`, so `s0` interacted with the two bounds differently |

**Local algorithm equivalence** (`algo-check`, scratch, ASan+UBSan): the new
kernel body transcribed per-row vs an independent transcription of the oracle —
**0 mismatched entries across 8 named shapes and a 4000-shape randomized sweep**
(half with coarsely quantized logits to force tie density, randomized offsets and
widths). Clean under both sanitizers. This derisked the change while the shared
GPU lock was held by other jobs; it is not a substitute for §4's device run.

**Device arm** — `test_cuda_deepseek_v4` built on `dgx.casa` (GB10, sm_121a) with
the mandatory gate flags (`-DVLLM_CPP_CUTLASS_DIR=$HOME/cutlass-4.5.0`,
`-DVLLM_CPP_TRITON=ON`), both arms from separate trees whose kernel identity is
asserted before the build so a stale tree cannot masquerade as the other arm:

- **RED** (old kernel + new tests): the device faults.

  ```
  terminate called after throwing an instance of 'std::runtime_error'
    what():  vt cuda: cudaStreamDestroy: an illegal memory access was encountered
  test_cuda_deepseek_v4.cpp:214: FATAL ERROR: test case CRASHED: SIGABRT
  [doctest] test cases:   6 |   5 passed | 1 failed | 17 skipped
  [doctest] assertions: 632 | 632 passed | 0 failed |
  [doctest] Status: FAILURE!
  ```

  Line 214 is `TEST_CASE("W7-device DSA top-k select: REAL index_topk widths
  match host BIT-EXACT (#505)")`. Script exit `134` = SIGABRT. The crash aborts
  the process, which is why only 6 cases ran and 17 were skipped.

  Note the shape of that summary: **`assertions: 632 | 632 passed | 0 failed`**
  next to `Status: FAILURE!`. A crashed case contributes no failed assertion, so
  an assertions-only reading of this log reports a clean run. The `Status` line
  and the case count are the load-bearing ones.

- **GREEN** (new kernel + new tests): full suite, nothing skipped.

  ```
  [doctest] test cases:    23 |    23 passed | 0 failed | 0 skipped
  [doctest] assertions: 83913 | 83913 passed | 0 failed |
  [doctest] Status: SUCCESS!
  ```

  `--list-test-cases` on the green binary confirms all three new cases are
  present, so the pass is not an absent-test artifact.

Both arms ran under `flock $HOME/gpu.lock` so a concurrent job could not perturb
them, and each arm asserted its own kernel identity before building — matching the
**declarations** `bool chosen[512];` / `int64_t picked[64];` rather than the
tokens, because the fixed kernel's comment cites both by name and a token grep
reports the fixed tree as the old one.

**Post-merge re-run.** `origin/main` advanced 17 commits (including the Mamba2 SSD
work) between the RED/GREEN pair and landing, so the device suite was rebuilt and
re-run from the *merged* tree rather than trusting the pre-merge green:

```
[doctest] test cases:    23 |    23 passed | 0 failed | 0 skipped
[doctest] assertions: 83913 | 83913 passed | 0 failed |
[doctest] Status: SUCCESS!
```

with the mandatory fast path hard-verified in that run's own configure log —
`CUTLASS found at /home/mudler/cutlass-4.5.0; enabling sm120a NVFP4 cutlass GEMM`
and `FlashAttention-2 prefill/decode: ENABLED for arch(es) [121a]` — and
`--list-test-cases` confirming all 3 new cases in the built binary.

Process note: the first attempt at both arms was lost to `client_loop: send
disconnect: Broken pipe` while queued on the GPU lock. The harness reported the
ssh as exit 0 while no DONE marker existed — the wrapper exited, the script never
finished. Both arms were relaunched under `setsid nohup` and gated on their marker
files rather than on the ssh status.

## 5. Upstream anchor

Unchanged by this fix; recorded because the fix must not drift from it. Upstream
selection is `ops.top_k_per_row_prefill`
(`vllm/model_executor/layers/sparse_attn_indexer.py:488-497`), and the candidate
window is built as `ks = row_start`,
`ke = row_start + (pos + 1) // COMPRESS_RATIO`
(`vllm/v1/attention/backends/mla/indexer.py:270-290`) — the full causal prefix in
**compressed**-key space, with no fixed cap. Our kernel now likewise has no cap.

Note for the real-geometry residual: our synthetic path passes `we[t] = t + 1`
over *uncompressed* keys (`deepseek_v4.cpp:806-808`), which is consistent at the
collapsed geometry but is not the upstream contract. Reconcile against that Triton
kernel, not against our host reference. That work stays out of scope here.

## 6. Stop conditions

- Do **not** reintroduce a configurable maximum `topk`. The formulation has no
  bound; adding one would re-create the class this issue closed.
- Do **not** make the kernel and the host reference share a selection helper. The
  gate's value is that two independent implementations agree; a shared helper
  would prove only self-consistency.
- Do **not** widen scope into the real-geometry DSA residual or the
  compressed-key-space candidate window (§5).

## 7. Review follow-up — [#552](https://github.com/mudler/vllm.cpp/issues/552)

A fresh reviewer (never the author) returned **PASS with 6 non-blocking
findings**, having failed to find any input where the two-pass selection diverges
from `DsaTopkSelect`: **3,000,081 fuzzed shapes across three independent
implementations** — the host reference, a transcription of the kernel body, and
the reviewer's own `O(n^2)` rank-count oracle so a shared misreading could not
pass — with **zero divergence**, output slots poisoned with `-777` so an unwritten
slot could not masquerade as legitimate `-1` padding. It reproduced the ASan
overflow and the device SIGABRT independently, and ran a 12-row device mutation
table on real sm_121a.

Landed here:

| # | finding | resolution |
|---|---|---|
| 1 | the `w < topk` comment named a NaN write-past-row as the defended failure class, which cannot occur | reworded in the kernel and in §3: belt-and-braces, bounded by construction. Guards kept. |
| 2 | no case exercised `win_start < 0` or `win_end > num_keys`; mutating either clamp away left all four cases green | new case `OUT-OF-RANGE windows are clamped like host (#552)`, four rows driving under-run, over-run, both, and an in-range control |
| 3 | `topk <= 0`: host asserts, device silently returned an empty vector | launcher now throws to mirror `deepseek_v4_dsa.cpp:76`; new case asserts **both** arms refuse |
| 4 | `DsaTopkLaunch` had no post-launch `cudaGetLastError()` | `Check(cudaGetLastError(), "dsa_topk launch")` added, matching every sibling launcher — but the finding's stated *rationale* does not survive measurement, see below |
| 5 | "strictly cheaper" could be read as real-geometry-ready | qualified in §3 and in the kernel comment: it is a comparison to the old kernel only |
| 6 | trivia: pre-fix span cited `:624-665` (actual 624-669); the comment described `n > topk` as the overflow condition | both corrected; `n > topk` only selects the branch, the overflow needs `n > 512` or `topk > 64` |

### 7.1 Finding 4's rationale was REFUTED by its own verification

The review's finding 4 held that a post-launch `cudaGetLastError()` would make the
next fault in this kernel attributable to it, instead of surfacing later as
`cudaStreamDestroy` the way #505's did. **That is wrong, and the arm built to
demonstrate it disproved it.**

Arm `prefix_with_check` — the pre-fix #505 kernel body WITH the new
`Check(cudaGetLastError(), "dsa_topk launch")` in place:

```
### prefix_with_check  exit=134
[doctest] test cases:   2 |   1 passed | 1 failed | 23 skipped
[doctest] assertions: 609 | 609 passed | 0 failed |
[doctest] Status: FAILURE!
  what():  vt cuda: cudaStreamDestroy: an illegal memory access was encountered
test_cuda_deepseek_v4.cpp:214: FATAL ERROR: test case CRASHED: SIGABRT
```

The error text is **unchanged** — still `cudaStreamDestroy`, not `dsa_topk
launch`. A stack-overflow illegal access is an *asynchronous execution* fault;
`cudaGetLastError()` immediately after a launch reports launch-*configuration*
errors (bad grid/block, shared memory over budget). Here it was not even caught by
the following `cudaStreamSynchronize`, and latched only at stream destruction.

The check is kept: it is free, it matches every sibling launcher, and it does
cover the launch-configuration class. But its comment and this spec now say what
it actually does. Writing "this makes the next fault attributable" would have
repeated finding 1's defect — an overclaiming guard comment — in the very change
that exists to correct one. The lesson is the finding's, not the reviewer's: a
plausible rationale for a cheap, obviously-correct change is still a claim, and
this one only failed because an arm was built to test it rather than to confirm it.

### 7.2 Device evidence for this change

`dgx.casa`, GB10 sm_121a, mandatory flags, fast path hard-verified in the run's own
configure log (`CUTLASS found … sm120a NVFP4`, `FlashAttention-2 … [121a]`). Each
arm is a fresh nvcc rebuild from a pristine kernel with the binary mtime verified
to advance, scoped with `-tc=` (never `-ts=`), and run under `flock $HOME/gpu.lock`.

| arm | exit | Status | reddened |
|---|---|---|---|
| baseline | 0 | `6 \| 6 passed` SUCCESS | — |
| `no_topk_guard` (drop the launcher's `topk > 0` throw) | 1 | FAILURE | the new refusal case: `CHECK_THROWS … did NOT throw at all!` |
| `no_ws_clamp` | **134** | FAILURE | the new clamp case CRASHED, SIGABRT, `illegal memory access` |
| `no_we_clamp` | 1 | FAILURE | the new clamp case, **1052** failed assertions |
| `no_launch_check` | 0 | SUCCESS | **nothing** — see §7.1; unobservable by construction |
| `prefix_with_check` | 134 | FAILURE | refutes finding 4's rationale (§7.1) |
| restored, full suite | 0 | `25 \| 25 passed \| 0 skipped`, 90062 assertions, SUCCESS | — |

Both new cases therefore have teeth, each against the mutation it was written for.
`no_launch_check` reddening nothing is the expected and honest outcome for a
diagnostic that has no observable behaviour on a passing run.

Note the false-green shape recurring twice more: `no_ws_clamp` printed
`assertions: 16865 | 16865 passed | 0 failed` and `prefix_with_check` printed
`609 | 609 passed | 0 failed`, both beside `Status: FAILURE!`.

Two infrastructure incidents, both handled rather than absorbed: round 1's clamp
mutations were **refused by their own uniqueness assertion** — the clamp pair
appears in both `DsaLogitsKernel` (:604-605) and `DsaTopkKernel` (:628-629), so a
one-line anchor was ambiguous and the assert declined to edit a kernel it was not
aiming at; round 2 re-ran with a three-line anchor unique to `DsaTopkKernel`. Then
the box **rebooted mid-arm** (`up 7 min`, the known GB10 unified-memory OOM-reboot
class), killing the run with no marker written — the remaining arms were relaunched
on the fresh box.

**Findings the review closed rather than raised.** Five mutations left the suite
green and are *semantics-preserving*, not coverage gaps — each is also undetected
by a 200k-shape fuzz against the oracle, which is how the reviewer separated the
two: removing `w < topk`, weakening it to `w <= topk`, simplifying pass 1's argmax
to `v > best_val` (equivalent, since the scan is ascending), `n <= topk` →
`n < topk` (equivalent, since at `n == topk` the full path selects all `n`), and
dropping `if (th_idx < 0) return` (reachable only via NaN). No action owed.

**The tie-heavy case is uniquely load-bearing** — confirmed, not assumed. It is
the only case in the suite that reddens a tie-break inversion or a value-only
threshold; the widths and offset cases use distinct random logits and stay green
under both mutations.

**Process deviation recorded, not repaired.** `b649a1ea2` introduced this spec in
the *same* commit as the code, where the protocol requires the spec to be
committed first. The substance was complete and the review supplies the
independent pass that #542 lacked, but the ordering was wrong and is noted here
rather than quietly dropped.

**Gate-reading trap, from the reviewer's own run.** A first scoped attempt used
`-ts='*DSA top-k*'` (suite filter) instead of `-tc=` (case filter) and printed
`test cases: 0 | 0 passed | 0 failed | 23 skipped` beside `Status: SUCCESS!` — a
live false green from a filter that matched nothing. Any scoped run of this suite
must use `-tc=` and confirm the case count is non-zero.

## Outcome

**Measured.** The two literal bounds were a real device fault, not a theoretical
one: at V4-Flash's own `index_topk` of 512 with a 600-wide window, the pre-fix
kernel takes an illegal memory access on GB10 and aborts the process. The
threshold rewrite is bit-exact against the independent host reference across both
shipped widths, tie-heavy rows and offset windows, and the full 23-case device
suite passes 83913/83913 with nothing skipped.

**What the bound cost, precisely.** `picked[64]` was 8x short for Flash and 16x
short for Pro; `chosen[512]` overflowed on any window wider than 512. Both were
invisible to every pre-existing device case because they all ran at `topk=3,
nk=5` — the gate's shape, not the model's.

**Rejected: asserting the bounds instead of removing them.** The issue itself
proposed a guard ("assert both bounds… so the kernel refuses rather than
corrupts"). A refusal would have been honest but would have left the device DSA
path unable to run the real `index_topk` at all, converting a latent overflow into
a guaranteed refusal the moment the real-geometry residual lands. The threshold
formulation needs no bound, so there is nothing left to assert. The `w < topk`
bound that remains is a NaN backstop, explicitly not a capacity limit.

**Rejected: sharing a selection helper between the kernel and the host
reference.** It would have removed the duplication and made the CPU test trivial,
but it would also have made the equivalence gate prove only self-consistency. Two
independent implementations agreeing is the whole value of this gate, so the
duplication is deliberate and recorded in §6 as a stop condition.

**Incidental finding worth keeping.** The RED log prints
`assertions: 632 | 632 passed | 0 failed` beside `Status: FAILURE!`, because a
crashed test case contributes no failed assertion. Any gate reading that reports
on assertion counts alone would have called this run clean.

**Not fixed here, deliberately.** The real-geometry DSA sparse path stays a named
residual, as does the compressed-key-space candidate window (§5) which our
synthetic path does not yet mirror. This change makes the kernel able to represent
the real widths; it does not put the real path on it.

## Now

Row unchanged at ✅. The `index_topk`-width limitation on the device DSA top-k
path is removed; the real-geometry DSA sparse path remains the named residual it
was. No lifecycle transition, so no `STATUS`/`BENCHMARKS` write is owed.
