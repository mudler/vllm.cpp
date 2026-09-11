ID: ISSUE-GH-429
Title: Ignored numeric/array bounds make the grammar a superset, not a subset -- and remove the only termination guard
Row: ENG-STRUCTURED-OUTPUT
State: OPEN
Kind: UNKNOWN
GitHub: 429
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `json_schema_to_gbnf.h` records `minimum` / `maximum` / `multipleOf` / `minItems` / `maxItems` as
> "NOT constrained (ignored where harmless)", under a heading stating that the deviations keep "the
> grammar ... a strict subset so output is always schema-valid".
>
> That framing holds for the key-order and `additionalProperties` deviations listed alongside them.
> It does not hold for the ignored bounds, and the difference is not cosmetic:
>
> * Constraining **more** tightly than the schema (fixed key order, closed objects) keeps output
>   schema-valid. Strict subset, as documented.
> * **Ignoring** a bound constrains **less** tightly. The grammar becomes a strict *superset*, so
>   the engine can emit output that violates the schema it was given. A `{"type":"number","minimum":0,
>   "maximum":1}` field can return `7.5`, and nothing in the pipeline reports it.
>
> ## The part that matters more than schema-validity: termination
>
> With the bounds dropped, these fields fall back to the primitives in `kPrimitivesBlock`:
>
> ```
> number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
> array  ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
> ```
>
> The fractional part is `[0-9]+` and the element list is `*` — both unbounded. Under constrained
> decoding an unbounded repetition next to a locally-likely character is a non-termination hazard:
> the model only has to keep preferring that character, and nothing in the grammar ever forces it to
> stop.
>
> We hit exactly this on our own schemas, on a different engine (vLLM + xgrammar), and it is not
> theoretical. A `confidence` field declared `{"type":"number","minimum":0.0,"maximum":1.0}` behaved
> correctly while the bounds were lowered. Removing them — on the theory that they were advisory —
> produced:
>
> ```
> {"label":"spam","confidence":0.99999999999999999999999999999999999999999... (800 tokens)
> finish_reason: length
> ```
>
> Generation ran to the token limit, the JSON never closed, the response was unparseable, and the
> whole budget was spent. The schema had been passing minutes earlier. Restoring `minimum`/`maximum`
> fixed it, because that engine lowers them into a bounded rule:
>
> ```
> {"type":"number"}                          ->  unbounded fractional [0-9]+
> {"type":"number","minimum":0,"maximum":1}  ->  ("0"|"1"|"0" "." [0-9]{1,6}|"1" "." [0-9]{1,6})
> ```
>
> **To be clear about what I have and have not measured:** the runaway above was observed on
> vLLM/xgrammar, not on vllm.cpp. What is established for vllm.cpp is structural — the converter does
> not read these keywords, so such a field is served by the unbounded `number` rule above, which is
> the same configuration that ran away. I have not reproduced the runaway here, and it is
> model-dependent and input-dependent in any case.
>
> The same reasoning applies to arrays: an unbounded element list has no forced stop.
>
> ## Why I am raising it rather than just sending a patch
>
> Two things are worth your call before anyone writes code:
>
> 1. **Whether "ignored where harmless" should be revised for this group.** The other documented
>    deviations really are harmless in the stated sense; these are a different class and currently sit
>    under the same sentence. Even with no code change, separating them would stop a reader
>    concluding that a bounded field is safe here.
> 2. **Where the lowering belongs.** `minimum`/`maximum` needs digit-range decomposition to express as
>    a grammar (the alternation above is one shape); `minItems`/`maxItems` is a `{m,n}` repetition and
>    is nearly free given the parser already accepts `{m,n}`. These are quite different in size and
>    might reasonably be separate rows.
>
> Happy to take either or both if you tell me which row to claim and which shape you prefer.
>
> For completeness, and because it cuts the other way: vllm.cpp's `strchar` keeps the escape branch in
> a single shared rule and has no length-bounded variant, so it is **not** exposed to the
> escape-dropping bug that the same feature area has in xgrammar (mlc-ai/xgrammar#800, patch in
> mlc-ai/xgrammar#828). The two engines currently fail in opposite directions on this keyword group.
>

## Resolution

-
