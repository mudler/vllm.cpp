# The registry's 34 remaining downcasts were promises, not checks

Issue: [#847](https://github.com/mudler/vllm.cpp/issues/847)
Row: `FIX-REGISTRY-DOWNCAST-SWEEP`
Precedent: [#775](https://github.com/mudler/vllm.cpp/issues/775) / PR #868
(`34962d96b`), which fixed one named site and created the seam this row spends.

## 1. Scope

Replace every remaining unchecked `static_cast<XLoadedModel&>(model)` in a
registered `prepare`/`forward` entry point with the checked
`ModelAs<XLoadedModel>(model, "<Architecture>")` seam, and gate the class.

**Out of scope, deliberately:** making `forward` a virtual on `LoadedModel`
(§5.2), and adding a checker that refuses a new `static_cast<...LoadedModel&>`
(§6). Both are recorded rather than done.

## 2. What the defect is

A `static_cast` down a hierarchy is a promise the compiler is entitled to act
on. On an object whose dynamic type is not that class, every member call through
the resulting reference is undefined behaviour. UBSan's vptr check reports
`member call on address ... which does not point to an object of type 'X'`, and
`-fno-sanitize-recover=all` aborts.

Production dispatch is self-consistent — `ModelRegistry::Forward` routes through
`model.registration().factory->forward`, so the dynamic type always matches — and
that is exactly why the class is invisible. #775 surfaced only because a doctest
`StubModel` reached one of these entry points and the sanitizer lane was
watching. The other 34 were latent by the same luck.

## 3. The measured shape

`grep -rn 'static_cast<[A-Za-z0-9_:]*LoadedModel\s*&>' src/ include/` at
`98f8e046d`: **34 sites across 30 files**. Every site is inside a registered
`Prepare*`/`Forward*` whose first parameter is `LoadedModel& model`, and every
site casts that same `model`. There is no `const`-reference spelling and no
pointer spelling, so the grep is the whole class. 14 sites bind the result to a
`const auto&` and 20 to a plain `auto&`; 30 are `forward` entry points and 4 are
`prepare` (`gemma4_registry.cpp:104`, `qwen3_5_dense.cpp:109`,
`qwen3_5_moe.cpp:91`, `qwen3_vl_registry.cpp:105`).

**None of the 34 turned out to be safe.** Each one is a real downcast of a
type-erased handle, reached through a live `ModelFactory` function pointer.

One is reached conditionally and is still swept: `gemma4_registry.cpp:104` sits
behind an early return unless `VT_GEMMA4_RESIDENT_EXPERTS=1`, so it is off on the
default path. That makes it harder to reach, not safe — the cast is the defect,
not the state it casts into — and it is why the class gate uses
`qwen3_5_moe.cpp:91` for the `prepare` shape rather than this one, which would
have needed the environment variable set to run at all.

### Three corrections to #847's description of the shape

The issue's estimate was made before the sweep looked at the tree. It is wrong
in three ways that matter, because all three were reasons it gave for calling
the sweep non-mechanical.

1. **The multi-registration count is wrong.** #847 says `llama_registry.cpp`,
   `qwen3_5_dense.cpp` and `gemma4_registry.cpp` "each carry 3
   `REGISTER_VLLM_MODEL` lines" and `mistral_registry.cpp` two. Measured: each of
   those three carries **two**, and `mistral_registry.cpp` carries **one**.
2. **It missed half the affected families.** Six files register more than one
   architecture against one factory, not four: `gemma4_registry.cpp`,
   `llama_registry.cpp`, `muse_glimmer_registry.cpp`, `olmo2_registry.cpp`,
   `qwen3_5_dense.cpp`, `qwen3_5_moe.cpp`. The last three are not named in the
   issue at all.
3. **The `const` overload is not what covers the `const` sites.** Both #847 and
   the dispatch that opened this row assumed the 14 `const auto&` sites select
   `ModelAs`'s `const LoadedModel&` overload. They do not: `model` is a non-const
   `LoadedModel&` at all 34 sites, so overload resolution picks the non-const
   `ModelAs` and its `Model&` result binds to the `const auto&`. Proven by
   mutation — deleting the const overload from the header and running
   `-fsyntax-only` over all 30 swept translation units compiles all 30 (§7, M4).
   **The const overload therefore has no caller anywhere in the tree.** It is
   left in place (it is #868's, it is harmless, and deleting another row's seam
   half inside a sweep is not this row's business) and recorded under `## Owed`.

## 4. The decision the sweep had to make

`ModelAs` takes the caller's own architecture name so a refusal says *which*
entry point refused. Six entry points serve two registered architectures each,
so #847 called this a design question with three defensible answers and no
single correct string.

**It is answerable, and the answer is #847's option 1: the family primary.** The
reason it is not arbitrary is that the aliases in every one of the six cases
share **one `ModelFactory`** and therefore produce **one `LoadedModel` subclass**.
The string does not identify a registration; it identifies the *factory whose
`load_weights` produces the type this entry point opens*, and there is exactly
one of those per site. That is also literally what the message says — "was not
produced by X's own `load_weights`" — so the primary is the true answer to the
sentence, not an approximation of it.

Where a canonical name is already written down, it is taken from there rather
than chosen: `gemma4_registry.cpp`, `qwen3_5_dense.cpp` and `qwen3_5_moe.cpp`
each call `RegistrationFor("<primary>")` in their own loader, and the sweep uses
that exact string. For `llama_registry.cpp`, `olmo2_registry.cpp` and
`muse_glimmer_registry.cpp` the primary is the first `REGISTER_VLLM_MODEL` line,
which each file's own comment marks as the real model and the second as an alias
("Alias only; zero forward/loader delta", "Both arch strings map to the SAME
factory").

The refusal still reports the passed model's own registration separately, so an
alias mismatch prints both strings and neither is lost. The class gate asserts
exactly that on two alias families (§7).

Options 2 and 3 from #847 are rejected: option 2 (report only
`model.registration().architecture`) drops the entry point's identity, which is
the only thing the parameter exists to add; option 3 (an overload taking
`ModelFactory*` and reporting every architecture registered against it) needs a
reverse index from factory to registrations that the registry does not keep, to
print a list where one name is correct.

### The architecture string per site

| File | Sites | Refuses under | Note |
|---|---:|---|---|
| `commandr_registry.cpp` | 69 | `CohereForCausalLM` | file name and arch differ |
| `deepseek_v2_registry.cpp` | 87 | `DeepseekV2ForCausalLM` | |
| `deepseek_v4_registry.cpp` | 101 | `DeepseekV4ForCausalLM` | |
| `gemma2_registry.cpp` | 69 | `Gemma2ForCausalLM` | |
| `gemma3_registry.cpp` | 70 | `Gemma3ForCausalLM` | |
| `gemma4_registry.cpp` | 104, 138 | `Gemma4ForConditionalGeneration` | alias family; `RegistrationFor` anchor |
| `gemma_registry.cpp` | 68 | `GemmaForCausalLM` | |
| `glm4_moe_lite_registry.cpp` | 114 | `Glm4MoeLiteForCausalLM` | |
| `glm4_registry.cpp` | 70 | `Glm4ForCausalLM` | |
| `granite_registry.cpp` | 67 | `GraniteForCausalLM` | |
| `internlm2_registry.cpp` | 94 | `InternLM2ForCausalLM` | |
| `kimi_k3_registry.cpp` | 81 | `KimiK3ForConditionalGeneration` | |
| `kimi_linear_registry.cpp` | 90 | `KimiLinearForCausalLM` | |
| `laguna_registry.cpp` | 84 | `LagunaForCausalLM` | |
| `llama_embedding_registry.cpp` | 114 | `LlamaModel` | the arch really is `LlamaModel` |
| `llama_registry.cpp` | 103 | `LlamaForCausalLM` | alias family (`InternLM3ForCausalLM`) |
| `minicpm3_registry.cpp` | 71 | `MiniCPM3ForCausalLM` | |
| `minicpm_registry.cpp` | 67 | `MiniCPMForCausalLM` | |
| `mistral_registry.cpp` | 91 | `MistralForCausalLM` | one registration, not two |
| `muse_glimmer_registry.cpp` | 89 | `MuseGlimmerForCausalLM` | alias family (`...ForConditionalGeneration`) |
| `olmo2_registry.cpp` | 74 | `Olmo2ForCausalLM` | alias family (`Olmo3ForCausalLM`) |
| `opt_registry.cpp` | 96 | `OPTForCausalLM` | |
| `phi3_registry.cpp` | 66 | `Phi3ForCausalLM` | |
| `phi_registry.cpp` | 71 | `PhiForCausalLM` | |
| `qwen3_5_dense.cpp` | 109, 120 | `Qwen3_5ForConditionalGeneration` | alias family; `RegistrationFor` anchor |
| `qwen3_5_moe.cpp` | 91, 97 | `Qwen3_5MoeForConditionalGeneration` | alias family; `RegistrationFor` anchor |
| `qwen3_dense.cpp` | 87 | `Qwen3ForCausalLM` | |
| `qwen3_moe_registry.cpp` | 87 | `Qwen3MoeForCausalLM` | |
| `qwen3_vl_registry.cpp` | 105, 110 | `Qwen3VLForConditionalGeneration` | 105 is the inline site, §5.1 |
| `stablelm_registry.cpp` | 69 | `StableLmForCausalLM` | arch spelling is `StableLm`, type is `Stablelm` |

The sweep ran from a table keyed by file, asserting per file that the number of
`LoadedModel` casts *and* the number of casts of the expected type both equal the
planned count before writing anything, and that the plan totals 34. A plan that
matched a near-duplicate anchor, or a file whose count had drifted, refuses
rather than writing ([[assert-anchor-uniqueness-not-existence]]).

## 5. The two judgement calls inside the sweep

### 5.1 `qwen3_vl_registry.cpp:105`, the one site that is not the common shape

Every other site is `(const )?auto& x = static_cast<T&>(model);`. This one makes
the member call on the cast expression itself:

```cpp
static_cast<Qwen3VLLoadedModel&>(model).CosSinCache(queue, config);
```

It stays inline on the checked reference:

```cpp
ModelAs<Qwen3VLLoadedModel>(model, "Qwen3VLForConditionalGeneration")
    .CosSinCache(queue, config);
```

Introducing a local was considered and rejected: `ModelAs` establishes the
dynamic type before the member call either way, so a binding would change the
site's shape without changing what it does, and a sweep that quietly restyles a
site it is only meant to make safe is harder to review, not easier. The wrap is
for the 100-column limit. This is the site a uniform search-and-replace over the
other 33 would step past, so the class gate covers it explicitly.

### 5.2 No per-site comment

#868 put a six-line comment at its single site. Repeating that 34 times would be
noise, and it would put 34 copies of one explanation where the seam already
carries it once (`model_registry.h:187-206`), which is the same reason #868
authored the refusal itself out of line. The one exception is the inline site,
where the *shape* decision above is not self-evident from the diff.

## 6. The checker question #847 raised

#847 asks whether a checker should refuse a new `static_cast<...LoadedModel&>`
under `src/vllm/model_executor/models/`. **Answer: yes, and not in this change.**

Yes, because the class is exactly grep-able — one regular expression, no false
positives in the tree today, and the residue after this sweep is zero — which is
what makes a checker cheap and exact here and does not make one cheap for the
unaligned-read class in #627.

Not here, because a repository-wide refusal gate changes what every future pull
request must satisfy. AGENTS.md requires a spec, a red-before test and
green-after evidence for a change to a checker's semantics, and introducing one
is at least that. Bundling it into a 30-file mechanical sweep gives a reviewer
two unrelated things to judge at once and makes the argument for each weaker.
Recorded under `## Owed` as
[#896](https://github.com/mudler/vllm.cpp/issues/896).

## 7. Tests and evidence

`tests/vllm/models/test_registry_downcast_refusal.cpp` — one case per **shape**,
not per site, because the failure modes of a sweep are shape-shaped. Every case
enters through the production seam (`reg.factory->forward` / `->prepare`, which
is what `ModelRegistry::Forward` calls) and hands it a `ForeignLoadedModel`: a
complete, well-formed `LoadedModel` carrying the entry point's *own*
registration, so the registration is right and only the object is wrong. That is
deliberately the case a weaker check cannot catch (§8.1).

| Case | Site | Shape |
|---|---|---|
| a NON-CONST forward | `qwen3_dense.cpp:87` | plain `auto&`, single registration |
| a CONST-bound forward | `stablelm_registry.cpp:69` | `const auto&` binding |
| an ALIAS arch refuses under the FAMILY PRIMARY | `olmo2_registry.cpp:74` | alias family + `const auto&` |
| a non-const ALIAS forward names both archs | `llama_registry.cpp:103` | alias family, unrelated spellings |
| a PREPARE entry point | `qwen3_5_moe.cpp:91` + `:97` | `prepare`, plus its `forward` sibling |
| the INLINE-call prepare site | `qwen3_vl_registry.cpp:105` | §5.1 |

**RED, against the unmodified tree** at `98f8e046d`, Debug + `address,undefined`,
each case run alone because `-fno-sanitize-recover=all` aborts on the first
finding. All six reproduce #775's diagnostic verbatim at six distinct sites,
process exit 1, no doctest summary reached:

| Case | UBSan report |
|---|---|
| non-const forward | `qwen3_dense.cpp:88:50: member call on address ... not ... 'Qwen3DenseLoadedModel'` |
| const-bound forward | `stablelm_registry.cpp:70:46: ... not ... 'StablelmLoadedModel'` |
| alias, const | `olmo2_registry.cpp:75:45: ... not ... 'Olmo2LoadedModel'` |
| alias, non-const | `llama_registry.cpp:104:46: ... not ... 'LlamaLoadedModel'` |
| prepare | `qwen3_5_moe.cpp:92:51: ... not ... 'Qwen3_5MoeLoadedModel'` |
| inline prepare | `qwen3_vl_registry.cpp:105:54: ... not ... 'Qwen3VLLoadedModel'` |

Each carried `note: object is of type 'ForeignLoadedModel'` and a stack naming
the entry point. As in #775 the report lands on the member call, one line past
the cast, except at the inline site where cast and call share line 105.

**GREEN after the sweep:** 6 cases, 33 assertions, 0 failed, no sanitizer
finding.

**Mutation proof.** Each applied alone, rebuilt, run, reverted, tree verified
clean.

| # | Mutation | Result |
|---|---|---|
| M1 | the sweep itself, absent | RED — the six reports above |
| M2 | `RaiseModelTypeMismatch` message replaced by `"model type mismatch"` | RED: 6/6 cases fail, 19 of 33 assertions — the tests assert a NAMED refusal, not merely the absence of an abort |
| M3 | one **wrong architecture string**: `olmo2_registry.cpp` refuses under `"Olmo3ForCausalLM"` instead of the family primary | RED: 1 case, 1 assertion — a wrong string compiles and runs, and only this gate sees it |
| M4 | the `const ModelAs` overload deleted from the header | GREEN, all 30 swept TUs still compile under `-fsyntax-only` — which is the *finding* of §3.3, not a passing mutation |

M3 is the one that matters for this row specifically. Thirty files of mechanical
edits fail silently through a wrong architecture name: it compiles, it runs, the
refusal still fires, and only the message is wrong. Nothing but an assertion on
the message content catches it.

## 8. Rejected alternatives

### 8.1 Compare `model.registration().architecture` to the expected name

Needs no RTTI and answers the wrong question: it establishes what the
registration *claims*, not what the object *is*. The realistic defect — a caller
that resolves a registration and hands `factory->forward` a model another path
produced — carries the right architecture string on the wrong object, so this
check passes and the UB proceeds. Every case in the class gate is that shape.
Already rejected by #775; restated because a sweep is where it would be
re-proposed as the cheaper option.

### 8.2 Make `forward` a virtual on `LoadedModel`

Removes the cast class outright and is arguably the better design. It is also a
change to the `ModelFactory` seam and 30 model translation units, against a
deliberate type-erasure contract (`model_registry.h:340`, "forward remains
type-erased over LoadedModel"). Weighed here because #775 explicitly deferred it
to the sweep, and still rejected: it is a seam redesign wearing a bug fix's
clothes, and it would land in the same diff as 34 edits it makes redundant.
Recorded under `## Owed`.

### 8.3 One test per site

Thirty-four near-identical cases assert the sweep rather than the guarantee, and
they would all pass against a sweep with 34 wrong architecture strings unless
each one also hard-coded its own expected string — at which point the test is a
transcription of the diff. One case per shape, plus M3, is what actually
distinguishes a correct sweep from a plausible one.

## 9. Cost

`ModelAs` is a `dynamic_cast` and a branch. The seam is entered once per
`ModelRegistry::Forward` — one forward *step* — and never per layer, against a
step that is milliseconds of GEMMs. `prepare` is once per load.

No throughput measurement is claimed for this row and none is owed: this change
adds no work to any loop, and the honest statement is what was checked rather
than a number produced to look like evidence. What was checked is that the call
count is unchanged and per-step: the 34 sites are the *first* statement of their
entry point, each entry point is reached only through `factory->prepare` /
`factory->forward`, and `ModelRegistry::Forward` is that seam's single call site.
Nothing was moved into a loop, and no site gained a second cast.

## 10. Now

Landed as the sweep of #847. The class residue under
`src/vllm/model_executor/models/` is **zero**: `grep -rn
'static_cast<[A-Za-z0-9_:]*LoadedModel\s*&>' src/ include/` returns nothing, and
`ModelAs` has 35 call sites there — the 34 swept plus #868's NemotronH one.

## Owed

- [#896](https://github.com/mudler/vllm.cpp/issues/896) — a checker refusing a
  new `static_cast<...LoadedModel&>` under `src/vllm/model_executor/models/`
  (§6). Decided yes, scoped to its own row so the gate change is reviewable on
  its own argument.
- [#897](https://github.com/mudler/vllm.cpp/issues/897) — `ModelAs`'s
  `const LoadedModel&` overload has **no caller in the tree** (§3.3, M4). Either
  a caller appears when an entry point takes a `const LoadedModel&`, or the
  overload goes. Not decided here because it is #868's seam.
- Making `forward` virtual on `LoadedModel` and retiring the downcast class
  outright (§8.2). Weighed and rejected for this row, not for the project. No
  issue filed: it is a design option this spec records having considered, not a
  defect, and #847 already names it.
