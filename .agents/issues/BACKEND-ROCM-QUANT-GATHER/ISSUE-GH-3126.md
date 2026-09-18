ID: ISSUE-GH-3126
Title: fix(BACKEND-ROCM-QUANT-GATHER): label the comparison scope from the run, not a literal
Row: BACKEND-ROCM-QUANT-GATHER
State: OPEN
Kind: UNKNOWN
GitHub: 3126
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-10
Updated: 2026-09-10
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM-QUANT-GATHER`
>
> `tools/rocm_quant_gather/compare.py` writes a fixed scope string into every report:
>
>     "scope": "synthetic operation parity; original fixture and model gates remain separate"
>
> The original-fixture coverage run over the 32 verified `Isotr0py/test-gguf-sample` fixtures therefore produced a report whose own scope line calls it "synthetic operation parity". The run itself is correct (320 outputs, 140 byte-exact, all within the upstream `atol=0.01 rtol=0.04`); only the label is wrong, and the report is cited as row evidence.
>
> Evidence: `/home/vikash/.cache/rdna3-gather-oracle-impl/original-fixture-coverage/compare/report.json` (report scope field) against its inputs (`fixture-manifest.json/manifest.json`, seal `f83b5c3e5cdb988b623c1b7e565e325a9ed9ea8085ea132bc8e08e6978bc1bbb`, exported from the 32 fixtures at revision `d82b8773934ef260d8d8a896a7c197bc69a0fac1`).
>
> Fix in flow: make the scope a command-line argument whose default preserves the previous text, and re-run the comparison so the report's label matches the run's inputs. No comparison, tolerance, or gate semantics change.

## Resolution

-
