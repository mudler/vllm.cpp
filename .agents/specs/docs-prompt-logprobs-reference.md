# Prompt log probabilities in the public documentation

## Scope

Correct the HTTP response description and simplify the usage recipe. This is a
documentation repair, with no feature lifecycle, runtime, or benchmark change.
The baseline is upstream main at `c2bac9ebf`.

| Item | Source | Documentation | Verification |
|---|---|---|---|
| Response placement | `src/vllm/entrypoints/openai/serving_completion.cpp:435`, `serving_chat.cpp:1064`, `protocol.cpp:879` | `docs/reference/server.md` | `tests/vllm/entrypoints/openai/test_api_server.cpp:4697` |
| Count and streaming validation | `src/vllm/entrypoints/openai/protocol.cpp:60` | Server reference and `docs/USAGE.md` | `tests/vllm/entrypoints/openai/test_protocol.cpp` |
| Prompt payload and echo distinction | `src/vllm/entrypoints/openai/serving_completion.cpp:456` | Server reference and usage recipe | `tests/vllm/entrypoints/openai/test_serving.cpp:500` |

## Design

The reference owns response fields, accepted counts, streaming limits, and the
unfinished echo behavior. The usage guide owns a short runnable curl recipe and
links to the reference for details. Keep the first-token null and per-token
logprob, rank, and decoded-token fields clear. Document zero separately from
positive counts and minus one. Do not imply that a request with max_tokens=1
avoids generation. Retain the recorded CUDA verification caveat without
claiming new hardware evidence.

The existing August README news already announces prompt log probabilities.
No new headline or benchmark follows from this repair. Open PRs cover GLiNER,
EXL3, Qwen3.8-Flash-Next, and multimodal documentation, so those are excluded.

## Upstream and dependencies

The implementation cites vLLM completion and chat protocol validators and
response serializers. Their port map and original evidence live in
[prompt-logprobs.md](prompt-logprobs.md), sections W2 design and W2 evidence.
This change describes checked-in code. It adds no port, test, or dependency.

## Tests and gates

Use CPU-only source review, document checks, local link and JSON validation,
and a fresh independent review. Run `scripts/agent-preflight.sh --quiet` and
report environmental skips or baseline failures separately. No GPU, model,
or oracle execution is needed because no runtime behavior changes.

## Risks and stop conditions

Do not expand into implementation changes or claim a new parity result. Stop
if a response claim cannot be grounded in the handler and serializer. Keep
existing hardware gaps explicit. New external infrastructure is out of scope.

## Owed

- ISSUE-LOCAL-01M30YN896NXJDRRQTPHAHPV8S: correct the server reference and usage recipe.
