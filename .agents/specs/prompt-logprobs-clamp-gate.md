# `SAMPLE-PROMPT-LOGPROBS` — measure the `ClampPromptLogprobs` call sites

*(Live spec, 2026-08-27. Base `origin/main` `331eda888`. Pin vLLM 0.26.0.dev0
`555967922`. Issue [#1817](https://github.com/mudler/vllm.cpp/issues/1817). Row
`SAMPLE-PROMPT-LOGPROBS` (`.agents/engine-matrix.md`, `ACTIVE`). Closes the
`## Owed` entry that [prompt-logprobs.md](prompt-logprobs.md) records for
[#1817](https://github.com/mudler/vllm.cpp/issues/1817). No lifecycle state
moves here, so no row edit and no `## Now` edit is owed.)*

## Scope

`ClampPromptLogprobs` rewrites a `-inf` prompt logprob to `-9999.0` because JSON
has no spelling for infinity. The function is gated directly. Its two call sites
are reached and not measured: deleting either one leaves the focused gate green,
so a refactor can drop the call and no test says so.

**In scope:** a test fixture whose prompt logits carry a real `-inf` at a chosen
vocabulary entry, and two cases that drive the production entry points
(`OpenAIServingCompletion::create_completion` and
`OpenAIServingChat::create_chat_completion`) with `prompt_logprobs` over that
fixture. Each case asserts the served value and the serialized JSON value.

**Out of scope:** the socket tier in `test_api_server.cpp`; the sampled-logprob
floor `kLogprobFloor` (`serving_utils.cpp:37`), which is a separate clamp with
its own call sites; the OpenAI `echo` residual; any production-code change. The
clamp itself is correct, and this row adds no product code.

## Upstream anchors

- `vllm/entrypoints/generate/base/serving.py:305-317` at `555967922` —
  `clamp_prompt_logprobs`, the function this tree mirrors in
  `src/vllm/entrypoints/openai/serving_utils.cpp:290`. Upstream walks every
  position, skips a `None` position, and assigns `-9999.0` to each entry whose
  `logprob` equals `float("-inf")`. It leaves every other value untouched.
- `vllm/entrypoints/openai/completion/serving.py:520` — the call site our
  `serving_completion.cpp:355` mirrors, before any choice reads the payload.
- `vllm/entrypoints/openai/chat_completion/serving.py` — the chat call site our
  `serving_chat.cpp:1055` mirrors, on the top-level response field.
- `vllm/model_executor/layers/logits_processor.py:183` —
  `logits[..., -num_pad:] = -float("inf")`. This is where a real upstream run
  gets the value the clamp exists for: the vocabulary padding columns are masked
  to `-inf`, and a full-vocabulary `prompt_logprobs` request then carries them.
  This tree has no vocabulary padding, which is why no fixture here produced the
  value before.
- `vllm/v1/worker/gpu_model_runner.py:5688-5697` — prompt scores come from
  `compute_logits` with no sampling processor applied, so the `-inf` must come
  from the logits themselves. The comment at `:5691-5693` states this.

## Design

### Why the value could not be produced

`ComputeLogprobsKernel` (`src/vt/cpu/cpu_sample.cpp:172-185`) computes
`out[j] = row[j] - lse`. With finite float32 logits of ordinary magnitude the
result is finite, so `log_softmax` alone never yields `-inf`. The synthetic
Qwen3.5 fixture has no masked vocabulary entry either. The value therefore has to
enter as a `-inf` logit.

### How the fixture makes one

The fixture keeps the production path and changes only the weights:

1. `lm_head` is `bf16 [H, V]`, and the logit for a token `t` is
   `sum_h hidden[h] * lm_head[h][t]`. Set `lm_head[h0][t]` to bf16 `-inf` and
   every other row of that column to `0`. The column then contributes
   `-inf * hidden[h0]`, and the sign of `hidden[h0]` decides the result.
2. `hidden[h0]` must be positive at every prompt position, or the logit is
   `+inf`, the row maximum is `+inf`, and the whole position becomes `NaN`.
   The fixture pins the sign by construction rather than by luck: it writes
   `+100.0` into column `h0` of every `embed_tokens` row, so the residual stream
   carries `+100` at `h0` for every token and every layer output is of order
   `0.1`. The final RMS norm keeps the sign, and the fixture sets
   `final_norm[h0]` to `+1.0`.
3. The logit for `t` is then exactly `-inf` at every prompt position, and
   `log_softmax` returns `-inf` for it while every other entry stays finite.

A measured probe over all 32 hidden dimensions confirmed the failure mode this
construction removes: with the embedding untouched, 26 of 32 choices of `h0`
gave `NaN` at one or more of the four scored positions of a five-token prompt,
because `hidden[h0]` changed sign between positions.

The request asks for `prompt_logprobs: -1`, which widens to the whole
vocabulary, so the `-inf` entry appears at every scored position without having
to be the position's target token. The chat case uses the same value, because the
chat validator accepts `-1` (`chat_completion/protocol.py:784-790`).

### What the cases assert

- The entry for `t` equals `-9999.0` exactly at every scored position. An
  exact comparison is correct here because the clamp assigns the constant;
  `doctest::Approx` at this scale carries a tolerance of about `0.12` and would
  also accept a value the clamp never wrote. Every comparison against the
  constant in this row is exact, on the served struct and on the wire, so the
  record does not argue for exactness in one place and accept `0.12` in
  another.
- The entry for `t` is finite. This is the assertion the deleted call site
  fails, and it fails as `-inf`, not as a near miss.
- A neighbouring entry keeps its own finite value below zero, so a clamp that
  overwrote the whole position would be red as well.
- The serialized JSON carries a number, and that number is `-9999.0` exactly.
  `nlohmann::json` DUMPS a non-finite float as `null`, so an unclamped payload
  loses the value and its type over the wire. This is the JSON edge the upstream
  function exists for. The case asserts on `json::parse(json(response).dump())`
  rather than on the in-memory object, because the in-memory object still holds
  `-inf` as a number and only the serialized bytes show the loss. The wire
  comparison is exact for the same reason the struct comparison is: `dump()`
  round-trips the double without loss, so a tolerance would only widen what the
  case accepts.

## Risks

1. **The fixture drifts.** A future change to the synthetic model can change
   `hidden[h0]`. The failure direction is red, never green: a sign flip makes the
   position `NaN` and the exact comparison fails. The construction in step 2
   makes a flip need a change of more than two orders of magnitude in the
   residual stream.
2. **The `-inf` weight reaches other paths.** It reaches only the logits of one
   vocabulary entry. That entry can no longer be sampled, which the case does not
   depend on: it asserts the prompt payload, not the generated text.
3. **The case is not the socket tier.** It drives the serving handlers directly,
   which is where the call sites are. The socket tier adds HTTP framing and no
   further clamp.
4. **The other clamp stays unmeasured.** `kLogprobFloor`
   (`serving_utils.cpp:37`) floors the SAMPLED logprobs with the same constant
   and is a separate surface. This row does not claim it.

## Tests

`tests/vllm/entrypoints/openai/test_serving.cpp`, two new cases appended at the
end of the file:

- `serving_completion: a -inf prompt logprob is served as -9999.0`
- `serving_chat: a -inf prompt logprob is served as -9999.0`

The file already owns the synthetic engine harness both handlers run on, so the
cases add a weight helper and nothing else. They are appended so that a
concurrent branch editing the same file merges without a conflict.

## Gates

```sh
cmake -S . -B build -G Ninja
cmake --build build -j 16 --target test_openai_serving test_openai_api_server
./build/tests/test_openai_serving
./build/tests/test_openai_api_server
scripts/agent-preflight.sh --fail-on-skip
```

The gate is met when both binaries exit `0` with `Status: SUCCESS!`, and when
each mutation in `## Evidence` exits non-zero.

## Evidence

Each mutation deletes one production call site, rebuilds, records the compile
status, runs the focused gate, and restores the tree byte for byte against a
`sha256` taken before the edit.

| Mutation | Site | Before this row | After this row |
|---|---|---|---|
| M6 | `ClampPromptLogprobs(prompt_logprobs);` (`serving_completion.cpp:355`) | GREEN, not caught: `rc=0`, 46/46 and 634 assertions | RED: `rc=1`, 47/48, 9 failed assertions |
| M6c | `ClampPromptLogprobs(response.prompt_logprobs);` (`serving_chat.cpp:1055`) | GREEN, not caught: `rc=0`, 46/46 and 634 assertions | RED: `rc=1`, 47/48, 9 failed assertions |

Both mutations compile (`COMPILE_RC=0`), so neither red is a build failure
wearing a test failure. Each red names the value: `CHECK( -inf == -9999 )` on
the served payload, and `REQUIRE( entry.is_number() )` on the wire bytes,
because `dump()` writes the unclamped value as `null`. `test_openai_api_server`
stays 76/76 under both mutations, which is the measurement that named the
serving-tier case as the one to write. The tree was restored with
`git checkout --` and verified against a `sha256` taken before the first edit;
`sha256sum -c` reported `OK` for both files after every mutation.

The unmutated tree is green: `test_openai_serving` 48/48 with 1365 assertions
and `test_openai_api_server` 76/76 with 1007 assertions, both `rc=0`. The two
new cases carry 366 and 365 assertions on their own, so neither is an
`assertions: 0` skip wearing a pass.

## Stop conditions

- Stop and report if the mutation is already red on the base commit. The row is
  then already closed and the record is what needs the repair.
- Stop and report `NEEDS_DECISION` if the only way to produce `-inf` needs a
  production seam that exists for the test alone.
- Never widen an assertion to make a mutation red. A green-on-deletion gate is
  the defect, and a weaker assertion hides it.

## Outcome

**Measured on the base commit first.** M6 still went green at `331eda888`:
`test_openai_serving` 46/46 with 634 assertions and `test_openai_api_server`
76/76 with 1007 assertions, both exit `0`, with the deletion proven by
`git diff -U1`. The 112 commits since #1815 did not close it.

**What was rejected.** Three other routes to a `-inf`:

- A large finite negative logit. `out[j] = row[j] - lse` only overflows to
  `-inf` when the row spans about `3.4e38`, which needs the row maximum and
  minimum to be within a factor of two of the float32 limit. The window is too
  narrow to hold across positions.
- A mirrored pair of huge columns, one positive and one negative. The positive
  column makes the row maximum `+inf`, and the position becomes `NaN`.
- A production seam that hands the handler a `RequestOutput`. Both handlers hold
  concrete engine references, so this means new production surface that only a
  test would use.

**Why the embedding is pinned rather than the norm.** The sign of `hidden[h0]`
is the product of the sign of the normalized residual and the sign of the norm
weight. Only the second is a fixture constant. Pinning the residual through the
embedding makes both factors constants.
