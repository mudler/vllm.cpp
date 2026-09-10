# Align multimodal HTTP documentation

## Scope

Correct the public description of `ENG-MM-INPUT-PIPELINE` without changing its
lifecycle or implementation. The local issue is
`ISSUE-LOCAL-01M24M9SWMN0WAX7PH2ZC2YWFN`. Base: `51ded8f10`.

## Source inventory and design

| Public fact | Source or evidence | Documentation action |
|---|---|---|
| The runner passes image features to the forward | `src/vllm/v1/worker/gpu/runner.cpp:3094`; `src/vllm/entrypoints/openai/server_main.cpp:1523` | Remove the obsolete claim that all HTTP multimodal requests fail |
| Image decoding accepts square raw RGB only | `src/vllm/entrypoints/openai/server_main.cpp:1554` | Replace the JPEG example with an explicit raw-RGB example |
| Limits depend on the architecture | `src/vllm/entrypoints/openai/mm_chat_dots3note.cpp:171`; `src/vllm/entrypoints/openai/chat_mm.cpp` | Distinguish Qwen3-VL single-image input from dots3-note multiple-image and audio input; keep video refused |
| Served-forward evidence uses synthetic weights | `tests/vllm/entrypoints/openai/test_api_server_mm_forward.cpp:24`; `tests/vllm/entrypoints/openai/test_api_server_dots3_mm_forward.cpp:27` | Describe reachability tests, never real-checkpoint token parity |

Keep the README compact and add one news entry for the served multimodal path.
Put workflow details in `docs/guides/multimodal-input.md`. Align the matching
claims in `docs/FEATURES.md` and `docs/reference/server.md` if needed.
Remove duplicated execution-chain prose from the edited sections.

## Upstream and dependencies

This is a documentation correction, not a port. The existing implementation
and its upstream citations define the behavior. The production dots3-note seam
cites `common/processor.py:527-534` at `9035151d6` for its limits. The existing
multimodal-track and dots3-note specs retain the port and oracle contracts.
No new dependency or oracle revision is needed.

## Tests and gates

Run the existing CPU-only documentation checks: `check-readme-structure.py`,
`check-supported-models.py`, `check-surface-coverage.py`,
`check-quickstart-recipes.py`, and `check-agent-record.py`, all under `scripts/`.
Run their applicable existing mutation suites and `git diff --check`.
Independently review every changed behavior claim against the source inventory.
Validate new Python examples with `ast.parse` and check local Markdown links.
Run `scripts/agent-preflight.sh` and report unavailable environment dependencies
separately from documentation regressions.

No new product tests are needed because no behavior changes. The two existing
served-forward suites above define the scope of published verification; this
documentation task does not claim to rerun model inference.

## Risks and stop conditions

Do not generalize Qwen3-VL limits to dots3-note. Do not advertise PNG/JPEG,
video, multimodal GGUF inference, real-checkpoint parity, or accelerator speed.
No GPU execution, external compute, model downloads, benchmark changes, or
lifecycle transitions are in scope. Stop a claim when the source or evidence
cannot establish it.

## Work breakdown

1. Commit this inventory and the local issue before public-document edits.
2. A fresh implementer corrects the four named public documents as needed.
3. An independent reviewer checks the immutable change against source.
4. The coordinator reruns CPU documentation gates and opens a fork pull request.

## Now

Documentation correction scoped. Benchmark disposition: NOT APPLICABLE;
commands and prose change no runtime behavior or measured result.
