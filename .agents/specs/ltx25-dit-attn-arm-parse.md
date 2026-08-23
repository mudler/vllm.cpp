# LTX25-DIT-ATTN-ARM-PARSE — the DiT attention A/B knob refuses a fourth value

Issue: [#1751](https://github.com/mudler/vllm.cpp/issues/1751).
Found in flow: [#1794](https://github.com/mudler/vllm.cpp/issues/1794).
Base: `27d8bfa706ccd538adeeaf0c48c7f7ee8ba8b753` (`origin/main` at the claim).

## Scope

`VLLM_LTX2_DIT_FLASH_ATTN` selects which `vt::` attention op the LTX-2.5 DiT
self-attention calls. It is not a configuration; it is the same-binary control
that lets one build measure a three-rung ladder, and its only product is a
statement about which rung ran.

It read its own arms by two different rules
(`src/vllm/model_executor/models/ltx2_device.cpp`, the DiT self-attention
dispatch). The naive arm tested `arm[0] == '0'` — a PREFIX — so `0x`, `07` and
`0flash` each selected `vt::Attention`. The flash arm used
`std::strcmp(arm, "flash")`, so only the exact string selected
`vt::AttentionDenseFlash`. Everything matching neither fell into a bare `else`
and ran `vt::AttentionDenseFa2`, the default, with no diagnostic: a typo
(`falsh`), a case difference (`FLASH`), a trailing space, a plausible-looking
`naive` or `1`, or an empty value from an unset shell variable.

IN SCOPE: make both arms exact, refuse a fourth value by name, keep `0`, `flash`
and unset behaving exactly as they do today, and repair the two committed
harnesses that the refusal turns from wrong into refused.

OUT OF SCOPE: what the arms measure, which rung is the default, and any number.
No arm moves and no threshold moves. This row runs no GPU and takes no lease.

## Why it is a correctness problem and not a tidiness one

`flash` is the DENOMINATOR of the 2.74x recorded in
[`ltx25-dit-attn-fa2-hd128.md`](ltx25-dit-attn-fa2-hd128.md) §8. A mistyped
denominator arm did not fail. It ran the NUMERATOR's kernel a second time and
yielded ~1.00x — which is also exactly what "no speedup" looks like, so the
number could not report its own failure. That is the shape `847e22f80` already
repaired once on this knob, where the `flash` arm was protected by nothing
executable.

The recorded measurements are not exposed by this. `assert_arm_op` in
`scripts/ltx25-dit-attn-fa2-hd128-ab.sh` reads the `VT_OP_PROVIDER_STATS=1`
announcement and exits 47 on a mismatch, so every arm in that row's §8 stated its
rung. What has no such check is a person exporting the variable in a shell or a
service unit, and that is who this row is for.

## Design

The dispatch becomes an exact three-way parse:

| value | op | why |
|---|---|---|
| unset | `vt::AttentionDenseFa2` | the shipped default and the ONLY serving arm |
| `"flash"` | `vt::AttentionDenseFlash` | the #1549 default, #1551's denominator |
| `"0"` | `vt::Attention` | the naive floor, reachable from this one binary |
| anything else | `throw std::invalid_argument` | REFUSED, by name |

Three decisions in it, each argued rather than inherited.

**Unset still means FA-2.** It is the shipped default and the only serving arm
(`docs/ENVIRONMENT.md`), and no production render sets the variable. Requiring it
would refuse every render this project ships. The knob has no "off"; absence is
the serving configuration and the two named values are the measurement lanes.

**An empty value is REFUSED rather than treated as unset.** This is the one
choice here that is not forced. `export VLLM_LTX2_DIT_FLASH_ATTN=$ARM` with an
unset `ARM` produces exactly a defined-but-empty variable, and it is the case in
which an operator most believes they selected an arm. Silently defaulting there
is the whole failure this row removes.

**The refusal ECHOES the offending value.** A trailing space and a case
difference are invisible in a shell history, so a message that lists only the
accepted values leaves `flash ` and `flash` looking identical to the reader.

The read stays FRESH rather than cached, unchanged from before, so a test can
flip the arm inside one process.

## The two harnesses, and why they are in this change

Fixing the parse turns a value that ran the wrong arm into a value that refuses.
Three committed arms exported such a value, so the change is not complete
without them ([#1794](https://github.com/mudler/vllm.cpp/issues/1794)):

| harness | arm | exported | before #1551 | on `main` today |
|---|---|---|---|---|
| `ltx25-dit-attn-flash-pixel-ab.sh` | `flash` | `=1` | flash | FA-2 |
| `ltx25-dit-attn-flash-pixel-ab.sh` | `flash-ctl` | `=1` | flash | FA-2 |
| `ltx25-dit-attn-flash-ab.sh` | `flash` | unset | flash | FA-2 |

Both files were TRUE when they were written — at #1549 the knob was binary and
flash was the default — and both became false in #1551, which touched neither.
One of the three is loud (the pixel harness's `arm_report` exits 46 on
`ROUTING_BAD`, though only after the render) and one is silent: phase `[F]` of
`ltx25-dit-attn-flash-ab.sh` PRINTS the op-provider selections and asserts
nothing, so its `flash` arm renders FA-2 and publishes the ratio under the label
`flash`. `scripts/ltx25-dit-attn-fa2-hd128-ab.sh` is the contrast: written after
the rename, correct, and carrying `assert_arm_op`.

Recorded numbers are not invalidated. `ltx25-dit-attn-flash.md` §10 records
`knob=1` announcing `op=21` and `ROUTING_OK=flash`, so the binary that produced
that run is one where `=1` still selected flash. What is stale is the file on
`main` against the binary `main` builds.

## Tests

**`tests/vllm/models/test_ltx2_device.cpp`** — "an unrecognised
`VLLM_LTX2_DIT_FLASH_ATTN` value is REFUSED by name". Nine values through
`Ltx2DitForwardDevice`, the production device forward the denoise loop calls
(`src/vllm/multimodal/ltx2_video.cpp`). Three prefix values that used to select
the naive rung (`0x`, `07`, `0flash`), five that used to select the default
(`falsh`, `FLASH`, `flash `, `naive`, `1`), and a defined-but-empty one. Each
must refuse, name the variable, list the accepted arms and echo what was set.

The negative half runs beside it: `0`, `flash` and unset must NOT throw. A parse
strict enough to refuse `falsh` is strict enough to refuse `flash`, and that
would take the 47.84 s and 7.68 s rungs out of this binary — the same loss of the
A/B from the opposite direction. The routing of those two values is already gated
exactly by the case above it, which asserts per-op selection counts; this case
asserts they still run at all, so a mutation that refuses everything cannot pass.

The empty value cannot use `vllm_test::SetEnv`, which maps an empty value to a
DELETE on both platforms by documented design (`tests/support/test_env.h`). It
uses a `::setenv` guarded on `!defined(_WIN32)`, which the shim's own header
instructs, because Windows has no defined-but-empty variable to test.

**`tests/scripts/test_ltx2_dit_attn_knob_arms.py`** — the gate whose absence let
#1794 happen. It reads the accepted set out of the dispatch's own
`std::strcmp(arm, "...")` calls rather than restating it, reads each harness's
arm invocations, and asserts two halves: every value is one the dispatch parses,
and every arm selects the rung its LABEL claims. The second is the half that
matters, because `unset` is an accepted value and was still the wrong arm for a
whole row. Both extractors carry a count precondition, so a restructured file
fails rather than reporting zero arms and passing. No GPU, no lease, no
toolchain.

## Risks

- **A refusal reaching a production render.** Bounded by the unset arm: no
  shipped path sets the variable, and the negative half of the doctest case
  asserts the unset forward returns.
- **A refusal thrown from inside the denoise loop rather than at startup.** It
  fires at the first DiT forward, which is the first thing after staging, so the
  cost is the load rather than the render. Validating at forward entry instead
  would move it by microseconds and add a second read of the same variable.
- **A live consumer outside this tree** — a service unit or a shell profile
  setting `=1` — now refuses instead of rendering. That is the intended change,
  and the message names the fix.

## Gates

- `./build/tests/test_ltx2_device`
- `python3 tests/scripts/test_ltx2_dit_attn_knob_arms.py`
- `python3 tests/scripts/test_ltx25_pixel_ab_harness.py`
- `python3 scripts/check-env-doc.py`
- `scripts/agent-preflight.sh`
- `bash -n scripts/ltx25-dit-attn-flash-ab.sh`
- `bash -n scripts/ltx25-dit-attn-flash-pixel-ab.sh`

## Evidence

Recorded in the pull request body, which is the landed commit message: the red
output of both suites before their fixes, the green after, and the fresh
review's mutation results.

## Owed

Nothing at landing. The row's two issues close with it.

## Stop conditions

Stop and report rather than widening scope if a live consumer of `=1` turns up
that cannot be repaired here — a published recipe, an image, or a service unit
outside this repository. The repair is then a deprecation window with a named
warning, which is a different design and needs the developer's decision, not a
silent re-admission of the value.

Stop rather than adjusting a threshold or an arm. This row moves no number.

## Now

`ACTIVE`. Parse, refusal, both suites and the two harness repairs are written and
green locally; awaiting fresh review and the operator's gate. No merge authority
is recorded for this row.
