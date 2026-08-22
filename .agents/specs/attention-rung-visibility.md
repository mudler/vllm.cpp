# Attention rung visibility: the naive kernel stops being the silent default

Issue: [#1544](https://github.com/mudler/vllm.cpp/issues/1544).
Owning row: KERNEL-ATTN-DENSE-FLASH (kernel-matrix.md), the row that already owns
`AttentionDenseFlash` and its head-dim contract. This spec is an increment on that
row, linked from its evidence cell, exactly as `fusion-consistency-audit.md` is an
increment on the fusion framework row. It is deliberately NOT added to that row's
`Spec` column, because `check-gate-commands.py` classifies a row from the FIRST
spec link in that column and the row is pinned there as `no-gates-section`;
promoting a second spec into that column would move the row into
`RUNNABLE_BASELINE` as a side effect of an unrelated change.

Note on spelling: this document writes the owning row id WITHOUT backticks for the
same reason. `check-agent-record.py::check_spec` selects a row's governing spec by
searching for the backticked token, and a second spec carrying it can change which
file is held to the structured-section contract.

## Scope

Two additive changes, both from #1544's `## Owed`. Neither changes what any
existing caller computes.

IN SCOPE:

1. A checker, `scripts/check-attention-rung-consistency.py`, that refuses a model
   translation unit which names `vt::Attention` — the naive, correctness-grade
   rung — without a recorded reason beside the call.
2. The marker comments the six deliberate call sites already deserve, in
   `whisper_audio.cpp`, `qwen3_vl_vision.cpp`, `kimi_linear_device.cpp`,
   `qwen3_5.cpp`, `nemotron_h.cpp` and `nemotron_h_device.cpp`.
3. A repair of `AttentionDenseFlash`'s advertised head-dim contract in
   `src/vt/cuda/cuda_ops.cu`, so the bound it states is the bound it can launch.
4. The shared-memory bound as a pure, unit-testable host function in
   `include/vt/ops.h`, so the arithmetic is executable on a box with no GPU.

OUT OF SCOPE, and each for a stated reason:

- `vt::Attention`'s behaviour and `OpId::kAttention`'s resolution. Both are frozen
  so text decode stays byte-identical (`src/vt/cuda/cuda_ops.cu:3120-3122`), and
  six model sites use the naive kernel as the REFERENCE arm of a numeric gate or
  an A/B knob. Auto-routing it would delete those reference arms.
- `ltx2.cpp`, `ltx2_device.cpp` and `muse_glimmer_vision.cpp`. Their `vt::Attention`
  calls are being REMOVED by other rows in flight ([#1545](https://github.com/mudler/vllm.cpp/issues/1545)
  for Muse Glimmer, the LTX-2.5 routing row for the other two). Editing the lines
  those changes replace would conflict for no gain, so the three stems are carried
  on `scripts/attention-rung-allowlist.txt` with the owning issue named.
- Opting `AttentionDenseFlash` in to the >48 KiB shared-memory cap. See
  `## Risks/decisions` D2.

## Upstream chain

vLLM never lets a model reach a kernel by omission, and it never advertises a
head-size domain it cannot serve. Both halves of this row mirror that polarity at
the pinned oracle `555967922`.

| Concern | vLLM at the pin | Ours before this row |
|---|---|---|
| Who picks the encoder attention kernel | `vllm/model_executor/models/vision.py:99` `get_vit_attn_backend(head_size, dtype)` — every vision tower ASKS a selector, which reads the shape and the override | the C++ function name the author typed; `vt::Attention` resolves straight to the naive kernel and nothing routes it up |
| What a backend's head-size domain means | `vllm/v1/attention/backend.py:155-163` — a backend DECLARES `get_supported_head_sizes()` and `supports_head_size()` is consulted BEFORE dispatch | `AttentionDenseFlash` declares `head_dim <= 256` and cannot launch above 192 (bf16) / 96 (f32) |

We cannot mirror `get_vit_attn_backend` directly: our seam has no config object and
no per-model backend enum, and the deliberate frozen-reference arms have no vLLM
counterpart. What transfers is the PROPERTY the selector gives upstream for free —
the kernel a model runs is a declared choice, not an omission. A checker is how a
C++ tree without a selector gets that property, and item 3 is simply the
`supports_head_size` half stated truthfully.

## Our baseline

Measured on this tree at `04f1cead6`.

Nine live `vt::Attention(` call sites under `src/vllm/model_executor/models/`:

| File:line | Disposition |
|---|---|
| `whisper_audio.cpp:324` | A/B rung, `VT_WHISPER_ENC_EAGER=1`, default is `AttentionDenseFlash` |
| `qwen3_vl_vision.cpp:527` | A/B rung, `VT_QWEN3VL_ATTN_EAGER=1`, default is `AttentionDenseFlash` |
| `kimi_linear_device.cpp:598` | behind `VT_KIMI_DEVICE_MLA`, default off, recorded as a measured negative |
| `qwen3_5.cpp:5279` | reference arm; non-test callers are in `tests/` |
| `nemotron_h.cpp:671` | reference arm |
| `nemotron_h_device.cpp:330` | reference arm |
| `muse_glimmer_vision.cpp:639` | THE defect, #1545, another row in flight |
| `ltx2.cpp:959`, `ltx2_device.cpp:421` | THE defect, LTX-2.5 routing row in flight |

`AttentionDenseFlash` at `src/vt/cuda/cuda_ops.cu:3336` states `d <= 256` and
requests `2 * kFlashBc * d * sizeof(Tin)` bytes of dynamic shared memory at `:3338`
with `kFlashBc = 64`. `grep -rn cudaFuncSetAttribute src/vt/cuda/` returns nothing,
so the launch is bounded by CUDA's default 48 KiB dynamic cap:

| Input dtype | Bytes per head_dim | Largest head_dim that launches | Advertised |
|---|---:|---:|---:|
| bf16 | 256 | 192 | 256 |
| f32 | 512 | 96 | 256 |

It fails LOUD — `Check(cudaGetLastError(), "attention-dense-flash launch")` at
`:3352` throws — so this is a wrong contract and a trap, never silent corruption.

## Port map

| Change | Path |
|---|---|
| Head-dim bound as pure host arithmetic | `include/vt/ops.h`, beside the `AttentionDenseFlash` declaration |
| Honest refusal at the launcher | `src/vt/cuda/cuda_ops.cu` `LaunchAttentionDenseFlash` |
| Rung-visibility checker | `scripts/check-attention-rung-consistency.py` |
| One source of truth for the tile bytes | `LaunchAttentionDenseFlash` now sizes its `shmem` request from the SAME `AttentionDenseFlashSmemBytes` the guard reads, so the two cannot disagree |
| In-flight stems, with owning issue | `scripts/attention-rung-allowlist.txt` |
| Marker comments | the six deliberate model translation units |
| Gate wiring | `scripts/agent-preflight.sh`, `.github/workflows/ci.yml` |

The checker reuses `scripts/checker_text.py::normalize_source`, so a
commented-out, `#if 0`-ed or `if (false)`-ed call is a deletion to it and never a
site, and the reported `file:line` still describes the original file. It scans
both `*.cpp` and `*.h` under the two model directories, because a call moved into
an inline function or a template in a header would otherwise leave it green, and
it keys its results on the PATH rather than the file stem, because `ltx2.cpp` and
`ltx2.h` share a stem and one would overwrite the other. The allowlist still
matches on the stem, so one entry covers a model's whole translation unit.

## Tests to port

There is nothing to port. vLLM's `supports_head_size` contract is enforced by its
selector rather than by a test that pins the arithmetic, and no upstream test
covers a shared-memory ceiling that only exists in our scalar kernel.

New, all runnable with no GPU:

| Test | Pins |
|---|---|
| `tests/scripts/test_check_attention_rung_consistency.py` | the checker's pure functions, and six mutations that must go RED |
| `tests/vt/test_ops_attention.cpp` new cases | the shared-memory arithmetic, both honest bounds, and that 256 is outside both |

One test needs a device and is declared PENDING rather than skipped quietly: a
CUDA case that calls `vt::AttentionDenseFlash` at head_dim 256 and requires the
refusal to name `AttentionDenseFast`. It emits a loud MESSAGE and returns on a box
with no CUDA backend, which is this box.

## Gates

Every leg below is CPU-only except where it says otherwise. There is no GPU lease
on this row: `dgx:gpu0` is held by the developer, and AGENTS.md forbids reaching a
fleet device outside a lease.

1. **Ran:** `python3 scripts/check-attention-rung-consistency.py` reports zero
   drift on the tree.
2. **Ran:** `python3 tests/scripts/test_check_attention_rung_consistency.py`.
3. **Ran:** `ctest -R test_ops_attention`, including the new head-dim contract
   cases.
4. **Ran:** `scripts/agent-preflight.sh --staged`.
5. **Owed:** the on-device refusal case. It needs a CUDA backend, and it is listed
   under `## Owed` with the issue that carries it.

## Dependencies

None on other rows. The three allowlisted stems depend on rows in flight only in
the sense that removing their entries is those rows' cleanup, and a stale entry is
reported and never a failure.

## Work breakdown

1. Spec, committed first.
2. The checker, its allowlist and its mutation suite.
3. Marker comments at the six deliberate sites.
4. The head-dim bound helper, the launcher refusal, and the contract tests.
5. Record edits: the owning row's evidence cell, the issue index, the claim.

## Risks/decisions

**D1 — a checker, not a selector.** #1544 owes "a selector or a checker" and both
are legitimate. A selector that auto-routes `vt::Attention` by shape was REJECTED,
and not on taste: six of the nine sites exist precisely BECAUSE they are the naive
kernel. `nemotron_h.cpp:671`, `nemotron_h_device.cpp:330` and `qwen3_5.cpp:5279`
are reference arms that tests compare against, and `whisper_audio.cpp:324` and
`qwen3_vl_vision.cpp:527` are the `*_EAGER` rungs of a same-binary A/B. Rerouting
any of them changes what the reference computes, which deletes the comparison the
gate performs; AGENTS.md names that failure directly — never make a red gate green
by widening an assertion's scope. A documented opt-in helper that picks by shape
was also rejected as insufficient on its own: it helps an author who already knows
the fast rungs exist, and #1544's defect is precisely the author who does not.

The checker inverts that. It cannot change any caller's numerics, because it runs
no code; and it fires on the one population that matters, an author naming the
naive kernel without saying why.

**D2 — narrow the bound, do not opt in to a larger cap.** `cudaFuncSetAttribute`
with `cudaFuncAttributeMaxDynamicSharedMemorySize` would make the advertised 256
true on some devices and NOT on others: head_dim 256 in f32 needs 128 KiB, above
the opt-in per-block cap of the consumer Blackwell parts this project gates on, so
the opt-in call itself can fail and the contract would still be a lie at the
widest advertised width. Narrowing is device-independent arithmetic, provable
here, and is a strict improvement for every caller: a head_dim that launches today
still launches, and one that does not now fails with a message naming the rung
that works instead of an opaque CUDA launch error from a later
`cudaGetLastError`.

That paragraph originally added "and it cannot be verified without a device, and
this row has no lease", which made the rejection read as a convenience. It is not
one, and the number is now on the record: **GB10's queried opt-in ceiling is
101,376 bytes**, measured while #1557 was reviewed. head_dim 256 in f32 wants
131,072. Opting in therefore CANNOT make the advertised 256 true for f32 on the
part this project gates on — the raise buys nothing at the width that motivated
it, and a caller at that width would have gone on falling back silently without
ever launching. Narrowing beats raising here on a measurement rather than on a
preference, which is why this row lands first and unchanged rather than
reconciling onto a cap-raising change.

Opting in remains available later as a widening for bf16 widths above 192, owned
by nobody today because no live caller needs head_dim above the honest bound
through this op.

**D3 — refuse rather than silently fall back to `AttentionDenseFast`.**
`AttentionDenseFa2KernelCuda` DOES fall through to `AttentionDenseFlash`
(`cuda_ops.cu:3395-3408`), so a silent step-down has precedent here. It was
rejected anyway. `AttentionDenseFast` re-reads K and V from global memory once per
(query, head) — that is the exact redundancy `AttentionDenseFlash` exists to
remove — so the fallback is a real, unannounced slowdown, which is #1544's disease
in miniature. This row's whole subject is that a rung change must be a declared
choice. A refusal naming `AttentionDenseFast` gives the caller the same
information and lets them make it.

**D4 — the allowlist is not a lock.** AGENTS.md forbids a record surface every
pull request must write. `scripts/attention-rung-allowlist.txt` is written only by
a change that ADDS an unmarked naive-attention site, which is the event the
checker exists to make deliberate, and it is emptied by the rows already in
flight. Two concurrent removals of different lines merge. The primary record is
still per-site and in-file: the marker comment lives beside the call it explains,
so the ordinary case touches no shared file at all.

**D5 — the marker's reason is checked for presence, not for truth.** The checker
requires a non-trivial reason string and cannot judge it. That is the same floor
`check-fusion-consistency.py` sets with its allowlist reasons. A reviewer judges
the reason; the gate only guarantees one was written.

**D6 — the detected population is one literal spelling, and that is stated.**
The scan matches `vt::Attention(` in a model `.cpp` or `.h`. Four spellings reach
the same kernel and are NOT detected — a `using vt::Attention;` plus a bare call, a
namespace alias, a `#define`, and a call through `&vt::Attention` — each verified
during review to leave the checker green with a live unmarked call. None exists in
this tree and none is how attention is called here, so this is a stated bound and
not a live hole. Widening the regex was rejected: dropping the `vt::` prefix makes
every fast rung a site, which is D1's failure mode again, and no regex reaches a
function pointer at all. Closing it needs a compiler-side population (the op
registry, or clang tooling over the real translation unit), which is a different
instrument and not this row's scope. The checker's docstring says so, so a green
reads as "no unmarked `vt::Attention(` call" and never as "no model is naive".

**D7 — the allowlist's stem set is pinned by a test, in another file.** D4 says the
CHECKER never forces a removing row to edit the allowlist, and that is still true.
`test_allowlist_holds_only_the_in_flight_stems` does force it: the expected set is
pinned, so adding or deleting a stem reds that case until the test is updated in
the same change. That is the intended shape for a parking lot — growth must be a
review decision — but it is a coupling a reader of the allowlist alone would not
see, so the allowlist header and the checker docstring both name the test.

**R1 — the launcher refusal is not executed on this box.** The pure arithmetic is
tested and mutated here, but nothing on a CPU-only box proves the launcher CALLS
it. A reviewer's reachability mutation for that leg needs a CUDA device. Stated,
not hidden; see `## Owed`.

**R2 — the 48 KiB constant.** `49152` is CUDA's default dynamic shared-memory cap
on every architecture this project supports, and the bound is INCLUSIVE. That
matters in one direction only: head_dim 192 in bf16 sits exactly at 49152 and
launches today, so an exclusive bound would refuse work that currently runs, which
would be a regression rather than a repair.

## Owed

- [#1573](https://github.com/mudler/vllm.cpp/issues/1573) — run the CUDA head-dim
  refusal case for `AttentionDenseFlash` on a leased device, and mutate the
  launcher's bound call to prove the case reaches it. PENDING a GPU lease. This
  stays owed after the merge: the CPU cases pin the arithmetic and never that the
  launcher calls it, and no lease was available for the whole branch. D2's
  101,376-byte GB10 ceiling is a queried device value and does not discharge it,
  because it bounds what an opt-in could buy rather than proving the refusal
  executes.

- [#1629](https://github.com/mudler/vllm.cpp/issues/1629) — DISCHARGED IN THIS
  ROW, and listed here because AGENTS.md wants the index row, this spec and the
  pull request body to agree, and this section is where this spec links its
  issues. Two drift locks in `tests/scripts/test_check_attention_rung_consistency.py`
  stored counts of files they do not own, so the three rows the attention-rung
  allowlist exists to unblock had no green path: `test_the_population_is_not_empty`
  asserted `>= 9` against a tree holding exactly 9 `vt::Attention(` sites, and
  `assertGreater(excused, 0)` required the shipped allowlist to stay non-empty
  forever. Both are repaired here, and neither by lowering a number, which is the
  mute-switch failure: the floor is now `>= 1`, which detects only a broken
  scanner and leaves the rename guard to
  `test_the_six_deliberate_sites_carry_a_marker`, which pins six sites by name; a
  new case proves the stem a red message names is a real source file; and the
  excused counter is pinned by two synthetic cases that build their own
  allowlisted population. Nothing is owed after the merge, so this entry is the
  link and not a debt.

- [#1631](https://github.com/mudler/vllm.cpp/issues/1631) — teach
  `scripts/check-pr-size.py` to tell a comment-only or docstring-only diff to a
  `governance_checker` from a semantic one, so a measurably false comment in a
  checker can be corrected on its own. `check-pr-size.py:170` classifies every
  `scripts/check-*.py` and `.sh` as a governance checker and `change_errors` then
  requires paired test evidence that goes red against the BASE checker, which a
  semantically identical diff cannot produce by construction. OWED after the
  merge, and it is why three comments in
  `scripts/check-attention-rung-consistency.py` ship unrepaired beside a suite
  that was repaired for #1629: `:58-61` gives the wrong reason for not widening
  the regex, `:93-96` attributes the exclusion of the fast rungs to the `\b`
  rather than to the trailing `\(`, and `:252-255` denies an equality that
  holds ON EVERY GREEN RUN, for ANY allowlist, and always did -- which is a
  larger claim than the one this entry used to make about one tree, and it is
  the one #1631 needs. The comment's stated reason is that "a marked call inside
  an allowlisted file counts in `marked`". True, and it does not separate the
  quantities: such a call is counted in `marked` AND excluded from `excused`, so
  it cancels on both sides. The only shape that separates `excused` from
  `sites - marked` is an UNMARKED call in a NON-allowlisted file -- which is
  exactly `drift_sites`, so `main` returns 1 at `:248` and the OK line never
  prints. Enumerated rather than argued: over all 64 green configurations of
  marked and unmarked calls across one allowlisted and one non-allowlisted file,
  zero break the equality, and the one configuration that breaks it is not
  green. So the printed `excused` is never anything but `sites - marked`, and
  the comment justifying a separate computation is wrong wherever a reader can
  see it. It cannot be fixed here, because
  changing what the gate accepts is what AGENTS.md `## Changing the rules or a
  checker` routes to its own row, spec and red-before evidence; attaching the
  correction to an unrelated semantic change is the alternative that section
  exists to refuse. A candidate patch is parked on the issue.

## Now

**The allowlist is empty, and the three stems it carried are discharged (#1663,
2026-08-22).** `## Scope`, the site table and `## Dependencies` above describe
`ltx2`, `ltx2_device` and `muse_glimmer_vision` as rows in flight; they have
landed. `47a918d8f` (#1579, issue #1545) routed Muse Glimmer's perception encoder
to `vt::AttentionDenseFlash`, so it names `vt::Attention` nowhere; `90e8c3c85`
(#1557, issue #1549) swapped the LTX-2.5 DiT device forward, and the two calls
that remain in `ltx2.cpp` and `ltx2_device.cpp` are the host CPU-only arm and the
`VLLM_LTX2_DIT_FLASH_ATTN=0` A/B arm, each now carrying its own marker.

D4 predicted the checker would report those entries STALE and not fail, and it
did, for the whole window from `90e8c3c85` to #1663. What D4 did not say is what
that window COSTS: a listed stem excuses its entire translation unit, so deleting
the live marker at `ltx2.cpp:959` left the checker at rc=0 (`7 carry a recorded
reason, 1 unmarked and excused`) while the stem sat there, and reds at rc=1
naming `ltx2.cpp:966` once it is gone. Both arms measured on `db648fb88` and
restored byte-for-byte against a pre-taken sha256. The deferral D4 designs for is
still right -- it keeps the removing row off this file -- but its cost is a real
blind spot in the covered files, not only untidiness, and the next row that parks
a stem should read it that way.

The change is written, CPU-gated and through one fresh scoped review, whose
findings are repaired here: the new checker registers its disabled creation-
mutation stub in `check-pr-size.py` (measured 31 of 31 cases red under the stub);
the kernel's register blocking is hoisted to `kFlashMaxPerLane` at file scope so
the `static_assert` reads the constant the kernel uses instead of the literal `8`;
the launcher's comment no longer claims the guard and the shared-memory request
come from one function; the checker states the four spellings it does not detect
and reports how many sites are unmarked and excused; and the kernel-matrix cell no
longer stores this suite's case count. The single owed leg above still needs
whoever next holds a lease.
