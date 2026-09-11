ID: ISSUE-GH-976
Title: docs: align README counts with the public registry
Row: ENG-DOCS-SITE
State: OPEN
Kind: UNKNOWN
GitHub: 976
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-16
Updated: 2026-08-16
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> The README and docs/FEATURES.md drifted from current source-backed counts after recent model and ABI additions.
>
> Current drift:
>
> - README reports 37 registered architectures; `scripts/check-supported-models.py` reports 40.
> - docs/FEATURES.md reports 38 registered architectures.
> - README reports ABI v19 with 36 exports; `include/vllm.h` declares ABI v20 with 45 `VLLM_API` exports.
> - README reports 36 tool-parser families and 40 names; the current public feature table and parser registry expose 38 families and 42 names.
> - The News section does not mention the landed MiniMax-Music3 generation surface.
>
> Acceptance: update the public docs from these source anchors and pass the README, public-doc-table, and supported-model checks without changing lifecycle or benchmark claims.

## Resolution

-
