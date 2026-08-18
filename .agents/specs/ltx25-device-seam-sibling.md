# The diffusion lane's device seam — the sibling that never adopted it, and the gate that cannot see it

Row: `LTX25-DEVICE-SEAM-SIBLING`
Issues: [#659](https://github.com/mudler/vllm.cpp/issues/659), [#660](https://github.com/mudler/vllm.cpp/issues/660)
Campaign: [#644](https://github.com/mudler/vllm.cpp/issues/644)
Base: `11cc1d5896b480a1b652db9249319242053aca93`

Both defects were found by a **peer session's reviewer** while reviewing the
main-red repair for landing, not by this campaign. They are recorded here
because they are in this lane and this campaign owns them.

## 0. What is wrong today

`11cc1d589` routed LTX-2.5's device question through the platform seam. It fixed
the lane it was aimed at and left two things standing.

**(a) #659 — the seam was adopted, its companion guard was not.**
`src/vllm/multimodal/ltx2_video.cpp:549-562 @ 11cc1d589` — anchored on the base
SHA for the same reason §0(b)'s `minimax_h3_video.cpp:221-226 @ 11cc1d589`
citation below is, because this row rewrites that block and an unanchored number
would point into the middle of the repair the moment it lands — now asks two
questions:

```cpp
const vt::DeviceType accelerator = vllm::platforms::CurrentPlatform().device_type();
if (accelerator == vt::DeviceType::kCPU ||
    vt::TryGetBackend(accelerator) == nullptr) { Fail(...); }
```

That is "is there an accelerator, and is a backend registered for it". The
precedent it cites, `SelectQueueForModel`, asks a third question —
`src/vllm/entrypoints/model_loader.cpp:98`:

```cpp
(architecture.empty() || plat.supports_model_architecture(architecture))
```

`supports_model_architecture` exists precisely so a **partial** backend can
decline by name. Two platforms override it and both declare a short list:
`src/vllm/platforms/metal.cpp:70` and `src/vllm/platforms/tenstorrent.cpp:55`.
On such a build a `device = 1` LTX-2.5 load **was refused by name** and is now
accepted, so the failure moves from a refusal that says which piece is missing
into a kernel bind that says nothing. CUDA is unaffected, which is exactly why
this is invisible on the box that runs the gates.

The refusal that *is* there argues, correctly, that serving the CPU forward
behind an accelerator handle "would make every later timing and every 'it ran on
the GPU' claim false". A partial backend that binds and dies has the same
property one level down: it is a device claim the build cannot honour.

**(b) #660 — the gate that certified (a) is a token grep, and the sibling lane
spells its way past it.** `scripts/check-device-leakage.py:78 @ 62406c30e` —
anchored, because this row rewrites that file's head and the line is now `:224`
— is

```python
RE_KCUDA = re.compile(r"\bkCUDA\b")
```

`src/vllm/multimodal/minimax_h3_video.cpp:221-226 @ 11cc1d589` — anchored on the
base SHA, because this row is what removes it and an unanchored line number here
would point at a blank line the moment it lands — never writes that token:

```cpp
vt::DeviceType MiniMaxH3VideoDeviceType(int32_t device) {
  if (device != 0 && device != 1) { throw ...; }
  return static_cast<vt::DeviceType>(device);
}
```

It hardcodes the ABI's `0/1` into the enum by integer cast and scores **zero**
in the `kcuda` bucket. The same defect, in the sibling diffusion lane, under a
different spelling.

**The sharpest form of it:**
`tests/vllm/models/test_minimax_h3_video_fold.cpp:162 @ 62406c30e` asserts
`MiniMaxH3VideoDeviceType(1) == vt::DeviceType::kCUDA` — anchored, because this
row is what rewrites that assertion and the file now carries it at `:220`
reading `== accelerator`. The *test* spells
the token honestly and is counted; the *source* launders it and is not. The gate
therefore reads the confession and misses the act.

This is the shape the project already has a name for — an instrument that cannot
report its own blind spot returns a pass. "`kcuda` 2 → 0" is a true statement
about the token and a weaker statement about the property than it appears to be.

## 1. Scope

**In.**

1. `minimax_h3_video.cpp` resolves its device through the platform seam rather
   than by integer cast, mirroring what `ltx2_video.cpp` now does. The public
   `MiniMaxH3VideoDeviceType(int32_t)` contract in
   `include/vllm/multimodal/minimax_h3_video.h:69` is preserved — `0` is CPU,
   anything else is refused or resolved, never cast.
2. `ltx2_video.cpp` asks `supports_model_architecture` alongside the two
   questions it already asks, and refuses **by name**, naming the platform and
   the architecture, when a registered backend declines the model.
3. `check-device-leakage.py` gains a bucket that sees an integer cast to
   `vt::DeviceType`. The bucket must be **derived from the property**, not from
   the one spelling we happen to have found — see §3.

**Out.**

- Any change to what CUDA does. This row must be a no-op on a CUDA build, and
  the gate for that is the existing suite passing unchanged on this box.
- Widening or rewriting `device-leakage-baseline.json` beyond the new bucket's
  own entries. **The baseline is a ratchet**: a peer measured that
  `--write-baseline` refuses to raise it (`REFUSING to write a HIGHER baseline
  (32 → 35)`) but that **hand-editing the JSON to a higher number makes the
  checker PASS**. So the only real defence is that the diff is visible and
  reviewed. Any baseline line this row touches is called out in the PR body,
  with its reason, and the reviewer checks the file by md5 against the base for
  every key this row does not claim.
- The H3 video engine's model behaviour. This is a device-resolution change.

## 2. Upstream anchors

There is no upstream for this: vLLM has no LTX-2.5 or MiniMax-H3 video engine,
and the device seam is ours. The mirror source is therefore **our own
`SelectQueueForModel`** (`src/vllm/entrypoints/model_loader.cpp:60-105`), which
is the shape every other model path already uses, and
`include/vllm/platforms/interface.h:263`, which defines the capability question.
Mirroring an internal seam is what "route it through the shared surface" means;
a second, parallel device resolution in the diffusion lane is exactly the
hand-rolled path AGENTS.md forbids.

## 3. Design

**The capability question, at the same site as the existing two.** One added
clause, one refusal message that names the platform, the architecture and the
fact that the backend declined — not a shape error. It must be possible to read
the refusal and know that the build is partial rather than broken.

**The sibling's resolution.** `static_cast<vt::DeviceType>(device)` is replaced
by an explicit mapping: `0` → `kCPU`, non-zero → the platform's
`device_type()`, refused when that is `kCPU` or its backend is absent or it
declines the architecture. That makes the two diffusion lanes answer the device
question the same way, which is the point of the seam.

**The gate's new bucket is the hard part, and it is where this row can go
wrong.** A bucket that greps `static_cast<vt::DeviceType>` closes the one site
we found and nothing else — the same defect as the token grep, one spelling
later. The bucket is defined by the property: *an integer literal or integer
variable becoming a `vt::DeviceType` without passing through the platform
seam*. Whatever the implementation, the acceptance test is adversarial and
stated up front:

> The bucket must go RED for **at least three spellings** of the same defect
> that are not the one already in the tree — e.g. a C-style cast, a
> `vt::DeviceType(x)` functional cast, and an assignment through an
> intermediate `int`. If it only catches the one we knew about, it is a
> regression test wearing a gate's clothes, and the row says so rather than
> claiming coverage.

If the property cannot be expressed at the granularity a text checker allows,
that is a finding to record, not to paper over: the row then states the residual
blind spot in the checker's own message, because **a checker's message is the
authority on what it enforces**.

## 4. Tests

RED-first for each of the three.

1. A build with a platform that declines the architecture must refuse the LTX-2.5
   `device = 1` load **by name**. The existing test fixture pattern for a
   partial backend is `metal.cpp` / `tenstorrent.cpp`; if neither is
   constructible in the CPU test build, the test injects a stub platform rather
   than skipping — a skipped test here is the whole finding.
2. `MiniMaxH3VideoDeviceType` keeps its contract: `0` → `kCPU`, `-1` and `2`
   throw (`test_minimax_h3_video_fold.cpp:161-164 @ 62406c30e` already assert
   this and must stay green **unchanged**), and `1` resolves through the seam
   rather than by
   cast. The new assertion is that on a CPU-only build `1` is **refused**, which
   the cast could never do.
3. The checker's three-spelling adversarial test above, each spelling asserted
   RED individually, not as a batch.

**Mutations that must be run and recorded:** revert each of the three changes
independently and confirm the corresponding test goes RED; and confirm the
existing suite is byte-identical in count on CUDA-absent builds, since a changed
count is RED even when it reads green.

## 5. Risks

- **The capability guard could refuse a load that works today.** `metal.cpp` and
  `tenstorrent.cpp` are the only overriders, so the blast radius is those two
  builds — but if either currently *runs* a diffusion model despite a short
  list, this row breaks it. Check before assuming; if it does run, the finding
  is in the list, not in the guard.
- **The new bucket could red other lanes.** An integer-to-`DeviceType` cast
  elsewhere in the tree is either the same defect (fix it or record it) or a
  legitimate deserialization boundary (which needs a stated, per-entry reason,
  never a blanket directory exemption).
- **Baseline churn.** See §1 Out.

## 6. Stop conditions

- If the capability guard turns out to refuse a currently-working configuration,
  stop and return `NEEDS_DECISION` rather than either shipping the refusal or
  dropping the guard.
- If the new bucket cannot reach three independent spellings, stop and report
  the residual blind spot; do not ship a one-spelling bucket described as
  coverage.
- `dgx.casa` is not required for any of this. Nothing here is a GPU measurement.

## Findings from implementation

**The brief and this spec disagreed on one line, and this spec won.** The
dispatch brief said `test_minimax_h3_video_fold.cpp:161-164 @ 62406c30e` must
stay green *unchanged*, but §4.2 above enumerates what "the contract" means —
`0 → kCPU`, `-1` and `2` throw — and separately requires that on a CPU-only build
`1` is **refused**. Line 162 asserted `MiniMaxH3VideoDeviceType(1) == kCUDA`,
which is precisely the cast's answer and cannot survive the change. It is now
build-conditional and asserts BOTH arms: `== accelerator` where one is
registered, refused-by-name where none is. 161/163/164 are untouched. Every
number in this paragraph is `@ 62406c30e`: on the branch the four assertions sit
at `:192-194` and the two arms that replace `:162` at `:218-238`.

**`test_minimax_h3_video_fold.cpp`'s CUDA-load case registered a BACKEND and no
PLATFORM.** It could, because the cast never asked whether the build had an
accelerator. It does now, so the fixture supplies both halves. That is the
defect being visible rather than a harness concession: a build with a CUDA
backend registered and no CUDA platform is not a build that runs on CUDA.

**The architecture key is the FAMILY string** (`ltx-2.5`, `minimax-h3`), not an
HF `architectures[0]` class name. The diffusion lanes are reached through
`LoadVideoEngine`/`VideoModelParams::family` and never read an `architectures`
entry, so the family slug is the only stable identifier they have. It does mean
`supports_model_architecture`'s key space now mixes HF class names
(`OPTForCausalLM`) with family slugs; the two cannot collide, and the refusal
names the string the user actually typed. Flagged for the reviewer as the one
judgement call in the row.

**`metal.cpp` and `tenstorrent.cpp` run no diffusion model** — searched in their
own vocabulary over `src/vllm/platforms/{metal,tenstorrent}.cpp` and
`src/vt/{metal,tenstorrent}/` with `OPTForCausalLM` as a positive control in the
same command: the control hit three times, `ltx|minimax|h3|diffus|video` hit
zero. So §5's stop condition is not triggered: the guard refuses nothing that
works today.

**The `dev_cast` bucket produced one false positive on the real tree**, and it is
the interesting kind. `kv_connector.h:225`'s
`supports_worker_transfer_on(vt::DeviceType /*device*/) const` strips to
`(vt::DeviceType )` followed by `const` — textually a C-style cast. Two
discriminators fixed it (the `(` must not be glued to an identifier; the `)` must
not be followed by a declarator suffix) and M29 pins both, together with the
proof that the discriminators did not cost the real detection.

**Residual blind spots are recorded in the checker's own docstring**, per §3's
instruction: type aliases, macros and template parameters that resolve to
`DeviceType`, `bit_cast`/`memcpy`/union punning, conversions inside the unscanned
`src/vt/` leg, and the fact that nothing type-checks the operand. The bucket
flags every cast *to* `DeviceType` and relies on `DSR-ALLOW` for the legitimate
ones.

## Findings from review (PR #671, head `094ac9e4`)

**The bucket missed the purest form of the defect, and the test that ruled that
out could not see it.** `(vt::DeviceType)d` scored 1; `(vt::DeviceType)1` scored
**0**, because the C-style alternative's trailing lookahead admitted only
`[A-Za-z_(]` and `1` is a digit. Naming a device by its literal enum value is
precisely what this bucket exists to police. Worse, M29's "the discriminators did
not cost the real detection" assertion used an *identifier* operand, so the test
could not detect the gap it was written to rule out — a guard that certifies
itself, which is the disease this row exists to fix, reproduced inside the row's
own instrument. M29 now pins the literal and signed-literal operands first.

**Four more plain spellings read GREEN in scanned files** and are closed rather
than documented, each measured at zero new hits over `src/vllm` + `include/vllm`:
global-scope qualification `static_cast<::vt::DeviceType>`, the
elaborated-type-specifier `static_cast<enum vt::DeviceType>`, direct-list-init in
a *declaration* (`vt::DeviceType dt{raw}` — M23 caught only the unnamed
temporary), and pointer punning `*reinterpret_cast<vt::DeviceType*>(&raw)`.

**The pointer form is what makes three of the four cast keywords real.**
`reinterpret_cast`, `const_cast` and `dynamic_cast` to a scoped enum are
ill-formed — compile-checked, all three rejected, with `static_cast` as a live
control that compiles — so before this change those keywords could only ever
match code that does not build, creating an appearance of coverage the pattern
did not have. Matching a pointer target makes them live.

**The declaration form takes `{` and never `(`, and that costs nothing.**
`vt::DeviceType dt(raw)` is ill-formed (no implicit int→scoped-enum conversion),
so excluding it loses no real spelling, whereas admitting `(` matched every
function *definition* whose return type is `DeviceType` — measured, 3 false
positives including `MiniMaxH3VideoDeviceType` and `ResolveExplicitDeviceType`.
M34 pins that negative.

**The fold test asked two of the three questions and so red on the build class
#659 exists to serve.** `test_minimax_h3_video_fold.cpp`'s `have_accelerator`
predicate tested only `device_type() != kCPU && TryGetBackend() != nullptr`. On
Metal or Tenstorrent both are true, so the test took the `== accelerator` arm
while the source *correctly* refused, and the refusal surfaced as an uncaught
exception — a false RED, invisible on the CPU and CUDA boxes that run the gates,
which is this row's own thesis about #659 turned against its own test. The
predicate is now three-way and asserts *which* refusal, since a right refusal for
a wrong reason is a wrong diagnosis that reads as a right one.

**The decline CONSEQUENCE inverts the cited precedent, deliberately.** The PR
described the change as mirroring `model_loader.cpp:98`, and it mirrors that
site's *question* while inverting its *answer*. `:98`'s capability test lives on
the `kAuto` path, whose response to a decline is to fall through to `:104` and
**serve on CPU**; `metal.cpp:65-69` states that policy in as many words ("falls
back to the CPU reference … and runs correctly, just slowly — which is strictly
better than dying inside a kernel bind"). Both diffusion lanes instead **throw**.
That is correct, but it is the *explicit-device* path's polarity, not the
`kAuto` path's: `device = 1` is an explicit accelerator request, and
`model_loader.cpp:72-73` already says of that path "an explicit accelerator whose
queue cannot be created must FAIL the load loudly, never silently serve on CPU" —
the same argument `ltx2_video.cpp:610-613` makes for refusing rather than serving
the CPU forward behind an accelerator handle. So the lanes mirror the capability
question from one path and the failure polarity from the other, and both halves
are the seam's own.

**Residual, not this row's to fix:** `vulkan.cpp` does not override
`supports_model_architecture`, so it inherits `interface.h:263`'s default `true`
and a partial Vulkan build still binds and dies inside a kernel. Only
`metal.cpp:70` and `tenstorrent.cpp:55` narrow the claim.

## Findings from review round 2 (PR #671, head `074ef1420`)

**F5 — the row's own thesis came back for its instrument a THIRD time, and this
one was in the checker's MESSAGE.** M33 had made the pointer target real, and the
docstring then said the bucket is "anchored on the TARGET TYPE, not on the
operand and not on one cast keyword", and listed `std::bit_cast` as unreachable
"because no spelling of the target type appears at the conversion site". Nine
spellings that DO write the target type at the conversion site scored **zero**.
AGENTS.md makes a checker's own message the authority on what it enforces, so an
over-claiming message is a defect in the gate, not a wording nit — and it is #660
exactly, one sigil later.

Each spelling was **compile-verified legal** (`g++ -std=c++20 -Wall -Wextra`,
exit 0) before being called a miss, because a "miss" that does not compile is
not a miss; and each measured **0 before / 1 after**, individually:

| spelling | why it slipped |
|---|---|
| `reinterpret_cast<vt::DeviceType&>(raw)` | reference target; the standard *defines* it as M33's pointer pun, and `\*?` admits a star but not an ampersand |
| `static_cast<vt::DeviceType const>(d)` | east const |
| `static_cast<vt::DeviceType const&>(t)` | east const + reference — the form that compiles with **no** warning, so it survives a `-Werror` build where the plain east-const prvalue trips `-Wignored-qualifiers` |
| `(vt::DeviceType const)d` | east const, C-style |
| `(vt::DeviceType)*cursor` | the trailing class admitted a sign but not a dereference — **and this is the wire-decode spelling the docstring itself names as the expected `DSR-ALLOW` case**, so the one site the bucket predicted meeting was the one it could not see |
| `(vt::DeviceType)~mask`, `(vt::DeviceType)!flag` | the same hole |
| `reinterpret_cast<vt::DeviceType**>(p)` | `\*?` is one star |
| `std::bit_cast<vt::DeviceType>(raw)` | the docstring's own reason was false for it: it spells the target in full |

Three edits close them: the named-cast target takes cv-qualifiers on **either**
side of the name and a **run** of `*`/`&`; the C-style trailing class admits
`*~!`; `bit_cast` joins the cast keywords.

**What was NOT closed, and why the docstring now says so per-entry.** `&` is
deliberately absent from the C-style trailing class: `) &` and `) &&` are
ref-qualifiers on a member declarator, which is precisely the false positive the
trailing guard exists to reject, and the guard cannot tell
`void note (vt::DeviceType*) &` from a pun. So the **C-style** pointer pun
`*(vt::DeviceType*)&x` stays blind, and is named as blind with the reason that is
true *of it* — the named-cast spelling of the same pun is caught, which is what
makes it a narrow gap rather than the class. The old blind-spot list shared one
reason across three entries and that reason was false for one of them; each entry
now carries its own.

**Measured, not asserted.** Over the files in `src/vllm` + `include/vllm`, with
`\bDeviceType\b` = **162** matches as a positive control **in the same
command**: **0 new, 0 lost**. (This round said "740 files"; the scanned set is
**760** — see round 3 below, which re-derived it.) Shipped hits 1, widened hits 1 — the same
allowlisted `platform.cpp:85` registry-walk inverse. `dev_cast` stays 0, `total`
stays 32, `scripts/device-leakage-baseline.json` is untouched by this round.

**M35-M40**, one mutant per newly-closed spelling, each sub-spelling asserted
**individually** (the M29 shape), because a mutant that only exercises the case
the pattern was tuned for is a guard that certifies itself. RED-before is real
rather than narrated: with the pattern reverted in-process to its pre-repair
value, M35-M39 all go RED while M20, M33 and M40 stay GREEN — so a blanket
failure cannot pass for five findings. M40 is the negative the widening could
have cost: pointer- and reference-returning declarations, ref- and
rvalue-ref-qualified members **with a space before the paren**, volatile
members, and `std::vector<vt::DeviceType*>`.

**Anchor drift, re-derived at the merge.** `9f2b9bb9a` moved
`tenstorrent.cpp`'s `supports_model_architecture` from `:52` to `:55`; the four
citations on this branch (`ltx2_video.cpp`, `test_diffusion_device_seam.cpp`, and
§0 and the residual note above) are corrected. `metal.cpp:70`,
`interface.h:263` and `model_loader.cpp:98` were re-derived on the merged tree
and all three still hold. The `minimax_h3_video.cpp:221-226 @ 11cc1d589`
citation in §0 is anchored on the base SHA for the same reason.

**The completeness claim this paragraph originally carried was false, and round 4
withdrew it.** It said "Every `path:NN` anchor in this row's files was resolved
mechanically; the only remaining unresolvable ones are upstream Python paths and
two `examples/*/main.cpp` citations". `be9b0a6fd` repeated it as "thirty-one
citations, five did not hold", and both were counting a set that did not include
the citations this branch's own edits had just rotted. Round 4 found six more
(§"Findings from review round 4"), all of them stale *inside this pull request*,
two of them seven and fifteen lines (measured at `7502004aa`) from §0(b)'s
`minimax_h3_video.cpp:221-226 @ 11cc1d589` citation, which SHA-anchors itself
with exactly the reasoning that applies to them. An enumeration that
certifies its own completeness is the defect this row keeps finding in its own
instruments, and it found it here in the record rather than in the checker.
No paragraph in this spec now claims that every anchor was checked; §"Findings
from review round 4" states what was re-derived, how, and what the method cannot
see.

## Findings from review round 3 (PR #671, head `79ebbce42`)

**The blind-spot ENTRY bound two members under one reason, and the reason was
true of only one of them.** The note covered "a C-STYLE cast whose target is a
pointer **or reference**" with the single reason "both are `)` followed by `&`".
That is true of `*(vt::DeviceType*)&x` and **false** of `(vt::DeviceType&)raw`,
whose next character is `r`. The reviewer **constructed the feared false
positive** instead of accepting the argument, which split the entry cleanly:
widening only **inside the parens** catches the reference form *and*
`(vt::DeviceType*)vp` at **zero** new hits, while the ref-qualifier negatives
`void note (vt::DeviceType*) &;` and `… &&;` stay at zero because `&` is still
out of the **trailing** class. So the trailing-class exclusion is justified, the
C-style **pointer pun** legitimately stays blind, and the entry is now two
entries with a reason each. Round 2 closed with "the old blind-spot list shared
one reason across three entries and that reason was false for one of them; each
entry now carries its own" — and the entry it wrote to replace them did the same
thing to two. The spec paragraph was accurate about the pointer pun; the
**checker's own message**, which is the authority, said "pointer or reference".

**Eight more spellings — counted, not rounded; the table groups two of them on
one row because they share a reason — each compile-verified legal (`g++ 13.3
-std=c++20 -Wall -Wextra -fsyntax-only`, exit 0) and each measured 0 before / 1
after, individually:**

| spelling | why it slipped |
|---|---|
| `(vt::DeviceType&)raw` | the C-style target admitted no `*`/`&` run at all |
| `(vt::DeviceType*)vp` | same |
| `vt::DeviceType const d{raw}` | alternative (3), the DECLARATION form, had **no cv-group** — a **fourth** place a cv-qualifier can sit, while M36 asserted "all three places". An enumeration certifying its own completeness, which is M29's defect returning |
| `vt::DeviceType volatile d{raw}`, `struct Cfg { vt::DeviceType const kD{1}; }` | same, and the member form is the one a person actually writes |
| `std::bit_cast<vt::DeviceType, std::uint8_t>(raw)` | round 2 added `bit_cast` *because* it spells the target at the site, then terminated the target at `>` on the assumption every cast takes one template argument. `bit_cast` is a **function template with two parameters** |
| `reinterpret_cast<vt::DeviceType* const&>(p)` | the `[*&]` run stopped at the first cv-qualifier, so the reference-to-const-pointer the docstring claimed was not reachable |
| `__builtin_bit_cast(vt::DeviceType, raw)` | not a template-id at all, so (1)'s `<…>` anchor cannot reach it and (2)'s glued-identifier discriminator rejects it |

**One pre-existing FALSE positive, closed rather than widened.**
`sizeof (vt::DeviceType) + 1` already scored **1** before this round — `sizeof`
is not glued to its paren, so the identifier discriminator passes, and `+` is in
the trailing class — and admitting `*` inside the parens would have extended it
to `sizeof (vt::DeviceType*) + 1`. `sizeof` and `alignof` convert nothing, so
both are excluded outright. M45 pins it, and is RED against the shipped pattern.

**What is NOT closed, and why the checker now says so as a PROPERTY.** Four
rounds have each found a spelling the previous round's message already claimed,
and every one closed at **zero** hits — so none was a coverage/false-positive
trade; there was simply always another spelling. Widening again is not the
answer to that, and the docstring no longer implies it is. It now: (a)
enumerates the forms it **measured** itself matching, (b) gives every blind spot
a reason true of that entry alone — the C-style pointer pun `*(vt::DeviceType*)&x`
(closing it needs `&` in the trailing class, which is the ref-qualifier false
positive), a **character-literal** operand `(vt::DeviceType)'\x01'` (character
literals are blanked to whitespace by `strip_comments_and_strings` before the
pattern ever runs, so the operand is gone by match time), `memcpy`/union (no cast
**expression** to anchor on — *not* "the type is not named at the site", which is
false for the idiomatic `std::memcpy(&dt, &raw, sizeof(vt::DeviceType))`), the
unscanned `src/vt/` leg, and the absence of any type check — and (c) states the
residual as a property: **a text checker enforces a set of SPELLINGS, not "an
integer becomes a `vt::DeviceType`"**, so a green `dev_cast` means "none of the
listed spellings is present" and never "no integer becomes a DeviceType here".
The structural answer is an **AST-level check** (clang tooling, where the
destination type canonicalises and the source type is known), filed as
[#828](https://github.com/mudler/vllm.cpp/issues/828) with all four rounds'
evidence, since "we kept finding more spellings" *is* the argument for it. Not
built here.

**M41-M46 and the M36 repair**, one mutant per newly-closed spelling, each
asserted individually. **M46 is the new shape**: it pins that the **declared
blind spots are still blind**, with `(vt::DeviceType)buf[0]` as a positive
control in the same test — so if a later widening closes one, M46 goes RED and
the message must be corrected in the same change. That is the only mechanism
that keeps a checker's message the authority on what it enforces. RED-before is
real, not narrated: with `RE_DEVTYPE_CAST` reverted in-process to `79ebbce42`,
exactly **6** mutants go RED (M36, M41-M45) and M20/M33/M40/M46 stay GREEN, so a
blanket failure cannot pass for six findings; applied, 0 RED. Suite 48 → **54**
tests, the +6 fully attributed to M41-M46.

**File count re-derived, and anchored to a SHA, because it rots.** #660's
roadmap row and this spec both said "740 files". The scanned set is **760 at
`79ebbce42`** and **765 at the merge below**, because main added five source
files under the scan roots in between. Both numbers were measured with
`\bDeviceType\b` = **162** and `\bkCUDA\b` = **18** as positive controls in the
same pass, and both are unchanged by those five files, which is why every other
figure in this section holds across the merge.

That drift is the finding, not an errand. A file count of one tree stored inside
another file is a measurement that every unrelated pull request invalidates,
which is the shape AGENTS.md forbids under "never store a measurement of one
file inside another file". "740" was not wrong when it was written; it rotted.
So both records now carry the SHA the number was measured at, and a reader who
finds a mismatch knows to re-derive rather than to distrust the rest. The
durable statement is the ratio the controls give, not the absolute count.

**Re-derived again at the `4a4ab89cb` merge, and it rotted again — as predicted.**
The scanned set is **776 files**, with the controls at `\bDeviceType\b` = **163**
and `\bkCUDA\b` = **18**. `kCUDA` is unchanged across seventeen commits of main
and `DeviceType` moved by one, which is the invariant this section actually
claims; the file count moved by eleven, which is the thing it says will rot. Two
notes for the next reader, because each cost a probe here and each made a correct
figure look wrong. The controls are counted over the
**comment-and-string-stripped** text — the text the pattern is actually matched
against — and NOT over raw source, where the same two patterns read **177 / 82**.
And the enumeration must be the checker's own `rglob` over `SCAN_ROOTS`: a
`git ls-tree` walk filtered with a string prefix returns **777**, because
`include/vllm.h` starts with the characters `include/vllm` while sitting outside
the `include/vllm` root. That off-by-one is the probe's, not the record's.

## Findings from review round 4 (PR #898, head `7502004aa`)

The review returned **FAIL** on a change it found correct and a gate it found
honest. All eight findings are record, test-strength or message defects, and
every one of them is this row's own thesis pointed back at the row.

**F1 — the "refuses BY NAME" assertion could not fail.** The LTX case asserted
`msg.find(kLtx2VideoFamily)`, and `Fail()` in `ltx2_video.cpp` prefixes EVERY
message with `"ltx-2.5 video: "`, which contains the family string verbatim. So
the assertion was satisfied by boilerplate on every refusal the file can throw,
including the two the same case asserts it is NOT. Proven, not deduced:
substituting `"<redacted>"` for the family name in the DECLINES `Fail` BUILT and
left the suite GREEN. The H3 side escaped only by a spelling coincidence — its
prefix is `"minimax_h3 video: "` with an underscore against a hyphenated
`"minimax-h3"` family — which is not a property and is now not relied on. Both
lanes assert the QUOTED SLOT, `architecture '<family>'`, through one
`QuotedArchitecture()` helper that carries the reason.

**F2 — the H3 half violated `## Nothing lands dead`.** The chain existed
(`vllm_video_engine_load` → `LoadVideoEngine` → the `minimax_h3` registration →
`MiniMaxH3VideoEngine::Load` → `MiniMaxH3VideoDeviceType`), but nothing entered
through it: every H3 device assertion called the resolver directly. Replacing the
`Load`-time call with the pre-row defect `params.device == 0 ? kCPU : kCUDA`
BUILT and left `test_diffusion_device_seam` and `test_minimax_h3_video_fold`
GREEN. The fold suite's `CUDA load creates exactly one queue` case does enter
through `Load`, but its `FakeCudaPlatform` reports `kCUDA`, so the seam and the
cast return the same answer and it cannot separate them. Two H3 cases now enter
at `LoadVideoEngine` against the declining `PartialXpuPlatform`, mirroring the
two LTX cases; and the two LTX cases were moved from `Ltx2VideoEngine::Load` to
`LoadVideoEngine` as well, so both lanes are entered at the same production
point and neither skips the registry hop.

**F3 — the anchor sweep that claimed completeness was incomplete, and its own
edits are what falsified it.** Six citations were stale, all of them rotted
INSIDE this pull request. Two of them sat seven and fifteen lines (measured at
`7502004aa`) from §0(b)'s `minimax_h3_video.cpp:221-226 @ 11cc1d589` citation,
which SHA-anchors itself with exactly the reasoning that applies to them.

| citation | was | is |
|---|---|---|
| `check-device-leakage.py:78` (`RE_KCUDA`) | unanchored | `@ 62406c30e` (on the branch: `:224`) |
| `test_minimax_h3_video_fold.cpp:162` (the `kCUDA` assertion) | unanchored | `@ 62406c30e` (on the branch: `:220`, `== accelerator`) |
| `test_minimax_h3_video_fold.cpp:161-164`, twice | unanchored | `@ 62406c30e` (on the branch: the three untouched at `:192-194`, the two arms replacing `:162` at `:218-238`) |
| `ltx2_video.cpp:549-562` (the two questions) | unanchored | `@ 11cc1d589` |
| `ltx2_video.cpp:562-565` (the refusal-to-fake-it argument) | `:562-565` | `:610-613` |
| `model_loader.cpp:97` (the capability clause), twice | `:97` | `:98` |

Four more were tightened rather than repaired, because a range citation that
starts on the wrong line is the same defect one size smaller: `model_loader.cpp`'s
`SelectQueueForModel` is `:60-105` and was cited `:59-104`; its auto arm is
`:76-104` and was cited `:75-104`; the `kAuto` fall-through to CPU is `:104` and
was cited `:103`; and `ltx2_video.cpp`'s device block runs to the end of the
capability refusal, where the citation stopped four lines short and cut the
`Fail` in half. That last one is now `:609-657`, having been `:566-614` for the
length of one merge — see the next paragraph.

The completeness sentence is withdrawn, above, with the reason. What replaces it
is a statement of METHOD and of what the method cannot see: every `path:NN` on a
line this pull request ADDS was extracted mechanically from
`git diff -U0 $(git merge-base e8048ef63 HEAD)`, resolved at the final tree, and
read. It deliberately excludes citations this row did not write — the LTX lane's
upstream Python anchors and the other rows' C++ anchors in the same files —
because re-deriving those is a different row's work, and folding them in is how
the last sweep came to believe it had checked everything. A citation this row
inherits and did not touch is therefore NOT covered by this paragraph.

**It rotted again during this round, which is the paragraph's own point made
twice.** `0785cfc4d` (`LTX25-RETIRE-DEAD-ARMS`) landed on main mid-gate and
edited `ltx2_video.cpp` above this branch's device block, moving all four of its
live line anchors by 43. The re-derivation caught it because it was re-run after
the merge rather than before it, and a run whose denominator moves afterwards
proves nothing. So the merge is taken FIRST and the anchors are derived at the
tree that is pushed.

Three limits of the method, stated because the previous sweep's failure was
believing it had none. The extractor reads a `path:NN` token; it cannot tell a
LIVE citation from one being QUOTED — the `was` column of the table above writes
`:97`, `:562-565` and two unanchored fold-test numbers deliberately, as the
record of what was wrong, and a later sweep will resolve them to the wrong lines
and must not "repair" them. The count of citations is itself a measurement of
one tree, so it is not stored here: re-derive it rather than compare against a
number that rots for reasons unrelated to this row, which is the same rule the
scanned-file count above already carries.

And — the limit that let round 5 find two more — **a bare `:NN` whose path sits
in a NEIGHBOURING token is never extracted at all.** The pattern requires the
path and the number to be glued, so `` `:224` `` written one line below
`check-device-leakage.py:78`, and a bare `` `:54` `` pointing at this file
itself, are invisible to it. They are not a rare shape — on the tree this
paragraph is pushed on they are a large minority of this spec's citations, and
the exact figure is deliberately NOT stored here for the reason the limit above
already gives. Re-derive it. That class is what the sweep which wrote this
paragraph could not see, and both of round 5's anchor findings belong to it. An
extractor for them must carry the antecedent path forward and must report when
there is none, because "no antecedent" is exactly the signature of a citation
into this file rather than into the tree.

**F4 — `## Now` pointed at a closed pull request** (#671) and the pre-rebuild
branch. Corrected, with the relationship between the two stated rather than the
old number silently swapped out, because the round-1 to round-3 findings above
were made against #671's heads and a reader needs to be able to find them.

**F5 — `dev_cast` over-matches a plain copy-initialisation.**
`vt::DeviceType d{other}` — a local copy, a member default-init, or an init from
a call returning `DeviceType` — fires alternative (3) while converting nothing.
Measured: 1 hit each, against 0 for `vt::DeviceType d{}`. It is NOT narrowed,
because `vt::DeviceType d{raw}` is the real conversion M32/M36 pin and is
textually identical; narrowing to remove the false positive deletes the true
positive. The docstring gains a third section, `WHAT dev_cast OVER-MATCHES`,
stating it with its cost — `dev_cast`'s baseline is a hard 0, so the first such
line under the scan roots fails the ratchet and needs `DSR-ALLOW` — and M47 pins
it in the M46 shape, with the value-init as the negative control in the same
test. Nothing in the tree writes the form today.

**F6 — `include/vllm.h` still called the selector CUDA.** The public ABI field
`vllm_video_model_params.device`, the v14 `vllm_model_params.device` note that
cites it as precedent, `video_engine.h`'s mirror and `minimax_h3_video.h:88`
nineteen lines below the docstring this row rewrote all read `1 cuda`.
`include/vllm/config/device.h:19` repeats the same sentence and is corrected with
them; it is the fifth instance of one claim, not a fifth claim.

**F7 — #828 was cited in product code and indexed nowhere.**
`check-device-leakage.py:58,139,158` and this spec name it as what closes the
declared blind spot, and it had no row in `.agents/issue-index.md`.
`check-agent-record.py` passed only because an absent row is nothing to count,
which is the instrument fault this row keeps finding. Appended with no owning row
and listed under `## Owed` below, which is the shape the protocol defines for an
issue a row files and does not fix.

**F8** — an inverted sentence in the test header, which read as though the
refusal and the kernel death both happened.

## Findings from review round 5 (PR #898, head `bf77c944e`)

Every code, test and checker repair passed. All four findings are record
defects, and two of them are the same one: **an anchor into a file the change is
itself editing is stale by default, and this one rotted between the two halves
of a SINGLE commit.**

**FF1 — `check-device-leakage.py`'s `RE_KCUDA` was recorded at `:188` and the
recording commit is what moved it to `:224`.** `f0b465029` carries both the F3
anchor table, which writes `:188`, and the F5 `WHAT dev_cast OVER-MATCHES`
docstring section, which adds 36 lines above `RE_KCUDA`. So the F5 half
falsified the F3 half of the same commit, and no merge was involved — the usual
suspect for anchor drift, and here it was innocent. Corrected to `:224` in both
places and in the pull request body, re-derived at the tree that is pushed.

**FF4 — the `:54` self-citation pointed at a blank line, and it is the paragraph
used as the exemplar of GOOD anchoring practice.** Three places cited "the `:54`
citation that SHA-anchors itself". At `7502004aa`, `:54` was §0(b)'s
`minimax_h3_video.cpp:221-226 @ 11cc1d589` citation, and the distances "seven
and fifteen lines" resolve against it exactly. `f0b465029` then added the very
anchors F3 asked for, §0 grew, and `:54` became blank. It is now named in PROSE
rather than by line number, because a number pointing into this file is the one
citation no re-derivation of the TREE can ever check, and this row has now
rotted it twice.

**Method.** Every citation was extracted — bare continuations included, which is
the change — and every live repo-local one re-derived at the pushed tree, with
the needle for each derived from what this
spec CLAIMS the span contains rather than read back out of the cited file — a
validator that reads its expectation from its target re-derives `f(x) == f(x)`
and cannot fail. A deliberately wrong row was carried as a positive control in
the same run and reported STALE, so the harness is known to discriminate. 50
rows FRESH, control STALE. Two further observations, reported and NOT repaired
because neither is wrong: `ltx2_video.cpp:609-657` opens one line inside the
preceding comment separator rather than on the block head at `:614` (the round-4
repair was to its TAIL, which is correct at `:657`), and
`check-device-leakage.py:58,139,158` is a genuine three-way citation whose needle
cannot be unique by construction.

## Owed

- [#828](https://github.com/mudler/vllm.cpp/issues/828) — the AST-level
  `dev_cast` check. `check-device-leakage.py:58,139,158` and §"Findings from
  review round 3" both name it as what closes the declared blind spot, and
  `scripts/check-device-leakage.py`'s docstring names it as what would enforce
  the property rather than a spelling list. It is **not built here**: this row
  ships the interim regular expression, its measured spelling list, its declared
  blind spots and its declared over-match. Filed with all four rounds' evidence,
  because "we kept finding more spellings" is the argument for it. Indexed in
  [`issue-index.md`](../issue-index.md) with no owning row, which is why it is
  listed here.

## Now

`READY`, implemented and awaiting a fresh review on
`row/LTX25-DEVICE-SEAM-SIBLING-REBUILD` ([PR
#898](https://github.com/mudler/vllm.cpp/pull/898)). PR #671, named by the
earlier rounds above, is CLOSED — it carried the same work on the pre-rebuild
branch, and the round-1 to round-3 findings recorded above were made against its
heads. All three changes plus the F5 repair and the round-4 repairs are on the
rebuild branch with their RED, GREEN and mutation evidence in the PR body; next
is a fresh reviewer — not the implementer — on the immutable head, then the
operator's own gate rerun.

The row stays `READY` deliberately. A lifecycle move to `ACTIVE` owes
`docs/STATUS.md` and `docs/BENCHMARKS.md` in the same change
(scripts/check-doc-checkpoint.py), and those are projections of what the project
CLAIMS — which this row does not change until it lands. Writing them from an
unmerged PR would also put two shared files under a lock for the length of a
review. The operator moves the state, and writes those two surfaces, when it
merges.

**This row now writes no record file at all, and that is the correct outcome
rather than an omission.** It previously annotated the `#659` and `#660` rows of
`roadmap_v1.md`'s `## Open issues` table in place. `#840`
(`POLICY-ISSUE-INTAKE`, spec [`issue-intake.md`](issue-intake.md)) moved that
table out to `.agents/issue-index.md`, which is append-only and carries
`merge=union`: a row is appended and never edited, and GitHub holds the open and
closed state, so closing `#659` and `#660` costs the index no edit. The in-place
annotation this row carried is exactly the `FIXED IN FLOW` shape that spec
retired, and under a union driver it would have been duplicated rather than
merged. The two index rows are therefore left byte-for-byte as `origin/main`
holds them, `LTX25-DEVICE-SEAM-SIBLING` has no portfolio row in `roadmap_v1.md`
(checked, with `ENG-WEIGHT-OFFLOAD` at two hits as a positive control in the same
command), and the evidence that annotation carried lives here, in the surface
that owns it.
