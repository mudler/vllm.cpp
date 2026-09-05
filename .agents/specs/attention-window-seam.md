# ENG-ATTENTION-WINDOW — one owner for the sliding-window rule

Row: `ENG-ATTENTION-WINDOW`. Issue:
[#2388](https://github.com/mudler/vllm.cpp/issues/2388).

## Scope

| | |
|---|---|
| In | Route the existing per-model sliding-window derivations through `ResolveAttentionWindow`, which already implements the rule and is reached by nothing. Reconcile what that exposes. |
| Out | Changing any model's ATTENTION MATH. The decoder result is identical and must stay identical; this row is about who owns the rule, not what it computes. |
| Out | The encoder-only symmetric case. The resolver implements it, no model in this tree asks for it, and inventing a caller would be worse than leaving it unreached. |

## What is actually true, measured rather than assumed

`ResolveAttentionWindow` (`src/vllm/model_executor/layers/attention/attention.cpp:12`)
returns, for the decoder path, exactly `{window - 1, 0}` — **byte-identical to
what all seven call sites already inline**. The refactor is therefore
behaviour-preserving on the value, and that is the premise the whole row rests
on. It is not a rewrite of attention.

It differs from the inline copies in three ways, all additive:

1. a `[1, INT32_MAX]` bound that THROWS outside the range;
2. the encoder-only symmetric `{radius, radius}` case;
3. per-layer window taking precedence over the model window, with a
   model-level disable.

**The lower bound is already enforced everywhere, by hand.** Every one of the
five model sites guards `*sliding_window > 0` before constructing the window:
`gemma2.cpp:197`, `gemma3.cpp:192`, `gemma4.cpp:335`, `olmo2.cpp:195`,
`muse_glimmer.cpp:229`. So adopting the resolver cannot start throwing where the
tree previously computed something — which is the risk that would have made this
row dangerous, and it is absent.

The UPPER bound is enforced nowhere today. No realistic window approaches
`INT32_MAX`, so adopting it is a refusal that should never fire; it is worth
having for the same reason the lower one is.

## The correction this row exists to record

An earlier reading of #2388 stated that the resolver's
`disable_model_sliding_window` parameter was "doubly unreachable" because no such
flag exists in the tree. **That is wrong.** It exists, under a different name and
twice:

- `gemma2.cpp:66` `SlidingWindowEnabled()` reading `VT_GEMMA2_SLIDING`
- `gemma3.cpp:71` `SlidingWindowEnabled()` reading `VT_GEMMA3_SLIDING`

Two identical functions, two environment variables, one behaviour. The grep that
missed them looked for `disable_sliding_window`, the upstream spelling, and a
failed grep is not proof of absence.

## What routing exposes, which is the real finding

The kill switch exists for **two of five** models. `gemma4`, `olmo2` and
`muse_glimmer` have no way to turn sliding-window attention off, so the same
debugging step that works on Gemma-3 is unavailable on Gemma-4. That asymmetry is
invisible while the rule is copied per model and becomes obvious the moment one
function owns it.

Both variables are on `scripts/env-doc-allowlist.txt` rather than in
`docs/ENVIRONMENT.md`, so neither is user-facing today.

## Work breakdown

**W1 — adopt the resolver at the five model sites.** Each site keeps its own
extra predicate (`use_rope` for muse_glimmer, the routing decision for gemma3's
per-layer pattern); those are model shape, not the window rule, and must not move
into a shared function. The value each site computes must be unchanged, and a
test asserts that per site rather than asserting the resolver in isolation.

**W2 — the two shared-path sites**
(`v1/attention/backend.cpp:323`, `mla_chunked_context.h:363`) are NOT model code
and take an already-resolved `sliding_window`. They are listed for completeness
and deliberately deferred: they sit under paths this row cannot exercise on CPU,
and moving them without a device gate would be the same unverified change this
campaign has been closing, not opening.

**W3 — the kill-switch asymmetry.** This said it was "a decision, not a
refactor ... because it changes a user-visible debugging surface", and left the
choice open between giving every model a switch and taking it from the two that
have one.

**That was wrong, and the rule it missed is the one at the top of `AGENTS.md`.**
vLLM defines this behaviour, so it is mirrored rather than decided, and asking
how a mirrored feature must behave is the thing this project does not do. At the
pin `5559679229`, `vllm/config/model.py:248`:

```python
disable_sliding_window: bool = False
"""Whether to disable sliding window. If True, we will disable the sliding
window functionality of the model, capping to sliding window size. If the
model does not support sliding window, this argument is ignored."""
```

One model-agnostic field on `ModelConfig`. Not a per-model switch, and not an
environment variable. Its docstring also answers the asymmetry directly: a model
with no window ignores the flag, so covering all five costs nothing at the sites
that have nothing to disable.

**Our two env vars are not a different spelling of it, they are a narrower
behaviour**, and that is the part worth stating precisely because it is what a
"just rename it" reading would ship:

1. `model.py:766-769` — the flag nulls `hf_text_config.sliding_window`, which
   disables the window for every layer of every model at once.
2. `model.py:2216-2233` — `_get_and_verify_max_len` additionally caps
   `max_model_len` to the sliding-window size when the flag is set.
   `VT_GEMMA2_SLIDING` and `VT_GEMMA3_SLIDING` do not do this.
3. `model.py:693-695` — the flag is set AUTOMATICALLY when a checkpoint declares
   `sliding_window == 0`, because vLLM spells "disabled" as `None` and some
   checkpoints spell it `0`. We have no analogue.

So W3 is: one `disable_sliding_window` config field carrying the upstream name
and all three semantics, consumed through `ResolveAttentionWindow`'s existing
`disable_model_sliding_window` parameter — which has never had a caller passing
anything but the default — and the two env vars deleted, with
`docs/ENVIRONMENT.md` losing them in the same change.

**The coverage gap is separate and survives the correction.** Nothing tests the
switch in either spelling; W1's mutation proved that by rewiring gemma2's site to
ignore `SlidingWindowEnabled()` and watching `test_gemma2_forward` stay green at
1003 assertions under `VT_GEMMA2_SLIDING=0`. Replacing the mechanism does not by
itself add a test, so W3's red-first test is: set the flag, assert
`ResolveAttentionWindow` returns no window for a model whose config declares one,
and mutate the flag read to confirm the test detects it.

**One part IS still a call, and it is small:** whether to keep the two env vars
as deprecated aliases for one release. Upstream has no env var to mirror, so
nothing decides it either way.

## Gates

- Per-site equality: for each of the five, the resolver's output equals the
  inline expression it replaces, over a table of windows including 1 (radius 0)
  and the largest plausible value.
- The `[1, INT32_MAX]` refusal fires, and fires by name.
- Mutation: break the resolver's `radius = window - 1` and require the per-site
  equality tests RED on a mutant that COMPILES.

## Risks

**The one that matters.** If any site's guard differs from `> 0` in a way this
spec missed, adopting the resolver changes behaviour silently, because the value
is only wrong for inputs the old guard excluded and the new one does not. The
per-site equality test is written against the ORIGINAL expression for that
reason, rather than against a shared expectation.

**Not CPU-testable end to end.** These sites feed paged attention; a value test
pins the window, not the attention. That is stated rather than implied.

## W1 outcome, including what the mutation FAILED to prove

Five model sites adopted: `gemma2`, `gemma3`, `gemma4`, `olmo2`,
`muse_glimmer`. Each keeps its own predicate — `use_rope` for muse_glimmer is a
routing decision (a NoPE layer takes no window at all), not a statement about how
wide the window is — and only the DERIVATION moved.

Green after adoption: `test_gemma2_forward` 1003, `test_gemma3_forward` 503,
`test_chunked_local_attention` 18849, `test_attention_window` 22,
`test_gemma4_honesty` 6, plus the new equality suite at 27.

**The kill switch has NO test coverage, and the mutation is how that was
found.** Rewiring gemma2's adopted site to ignore `SlidingWindowEnabled()`
compiles cleanly and `test_gemma2_forward` passes with 1003 assertions in BOTH
`VT_GEMMA2_SLIDING=0` and the default. So nothing in the suite holds that
switch, before this change or after. The routing preserves it by construction,
which is not the same as proven.

That is a pre-existing gap this row exposes rather than one it introduces, and it
sharpens W3: a debugging switch that exists for two of five models AND is
untested on both is weaker than the issue suggested.

Two mutations that did NOT compile were rejected before their numbers were
believed — `-Werror=parentheses` on `false && a || b`, and
`-Werror=unused-function` when removing the switch's only caller. A mutation
build failure reads as a passing test, and both would have.

## Now

W1 done. W2 (the two shared-path sites) and W3 (the kill-switch asymmetry, now
also a coverage gap) owed.

W3 is no longer blocked on a decision. `## Work breakdown` above records why: the
flag is `disable_sliding_window` on vLLM's `ModelConfig`, so the shape is
mirrored rather than chosen, and what was filed as a naming question turned out
to be three missing behaviours. It needs no GPU and no checkpoint.
