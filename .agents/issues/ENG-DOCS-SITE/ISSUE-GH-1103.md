ID: ISSUE-GH-1103
Title: docs: FEATURES undercounts reasoning parsers
Row: ENG-DOCS-SITE
State: CLOSED
Kind: UNKNOWN
GitHub: 1103
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-17
Updated: 2026-08-17
Closed: 2026-08-17

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `docs/FEATURES.md` reports 10 reasoning-content parsers, but `reasoning_parser_names()` in `src/vllm/entrypoints/openai/reasoning_parsers/abstract.cpp` exposes 12 names.
>
> Update the public feature table to 12. This is documentation-only and needs no GPU.
>
> The open landing-page alignment PR #977 does not include this count.

## Resolution

GitHub records closing pull request #1104 (https://github.com/mudler/vllm.cpp/pull/1104) merged on 2026-08-17 as commit `2e025247e28e28fc8b55b088d8166a276ef9a7bd`. GitHub closed issue #1103 on 2026-08-17.
