# Accept `/` in the Assisted-by model slot

Issue: ISSUE-LOCAL-01M27N44F915BK5QXR6HZ3NGGA (GitHub mirror: [#3132](https://github.com/mudler/vllm.cpp/issues/3132)).
Row: `ENG-TRAILER-SLASH-MODEL`.

`scripts/check-commit-trailers.py` rejects `/` in the MODEL slot of
`Assisted-by: AGENT:MODEL [TOOL]`. The model identifier `regolo/glm5.2` uses
`/` as the provider/model separator, the same convention HuggingFace
(`meta-llama/Llama-3`) and OpenAI (`openai/gpt-4`) use. Every commit that
cites this model mangles the `/` to `-` to pass the gate, and two commits
landed with the real slash form when CI was down (#3132).

## Scope

**In scope.** The MODEL character class of the `ASSISTED_BY` regex in
`scripts/check-commit-trailers.py`.

**Out of scope.** The AGENT character class (agent names carry no `/`). The
TOOL class (already accepts `/`). Every other assertion the checker makes. No
rule is relaxed; the grammar accepts one more character that the TOOL slot
already accepts.

## Upstream chain

None. vLLM has no counterpart to this protocol machinery.

## Design

The regex is:

```python
ASSISTED_BY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.+-]*"
    r"(?: \[[A-Za-z0-9][A-Za-z0-9_. +:/-]*\])+\Z"
)
```

The MODEL class is the second group: `[A-Za-z0-9][A-Za-z0-9_.+-]*`. Add `/`
so it reads `[A-Za-z0-9][A-Za-z0-9_./+-]*`. The `-` stays at the end of the
class (literal). `:` remains the structural AGENT:MODEL separator; `/` inside
MODEL creates no ambiguity because the split is on the first `:`.

## Risks

None material. The TOOL slot of the same regex already accepts `/`. Widening
the MODEL class to match does not weaken any other assertion.

## Tests

Red-before: a test that `AGENT:regolo/glm5.2 [maki]` is accepted fails on the
current regex (malformed Assisted-by). Green-after: the same test passes once
`/` is in the class.

## Gates

`tests/scripts/test_check_commit_trailers.py` — the attribution contract suite.

## Stop conditions

The regex accepts `regolo/glm5.2` and every existing test stays green.
