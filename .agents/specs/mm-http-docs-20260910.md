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

Documentation correction implemented; upstream integration remains pending.
Benchmark disposition: NOT APPLICABLE; commands and prose change no runtime
behavior or measured result. The existing feature lifecycle is unchanged.

## Verification and review correction

The five documentation checks above pass. Existing mutation suites for README
structure, surface coverage, supported models, and quickstart recipes pass
108 tests in total. Offline execution of the image example verifies the POST
endpoint, model name, MIME type, and exact RGB bytes. Empty and malformed
buffers refuse before HTTP. Local links in all four edited documents resolve.

Independent review identified a tower-loading distinction. Qwen3-VL,
MuseGlimmer, and the `clip` projector skip loading at zero modality limits
(`qwen3_vl.cpp:463`, `muse_glimmer_weights.cpp:803`, and
`src/vllm/entrypoints/model_loader.cpp:3049`). Dots3-note still loads supported
towers (`src/vllm/model_executor/models/dots3_note.cpp:782` and `:807`). The
guide, feature table, and server reference now distinguish request refusal
from tower loading. The refusal example explicitly uses default limits.

Full preflight was attempted on the local Alpine shell. It did not pass:
the C/C++ toolchain is unavailable, BusyBox `find` lacks `-printf`, and tests
that clear the environment cannot load the temporary Python libraries.
These broader environment failures do not establish a documentation regression.
No inference, GPU gate, or benchmark was run.
