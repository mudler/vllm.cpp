ID: ISSUE-GH-338
Title: MiniCPM/MiniCPM3 hard-code SiLU: upstream `MiniCPMMLP` (`minicpm.py:219-226`) selects `FatreluAndMul` on `hidden_act == "fatrelu"` and raises otherwise; our `parse_config` never reads `hidden_act`
Row: ROAD-V1-C1
State: UNKNOWN
Kind: bug
GitHub: 338
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:128`

### Frozen archive evidence

> | [#338](https://github.com/mudler/vllm.cpp/issues/338) | `ROAD-V1-C1` | MiniCPM/MiniCPM3 hard-code SiLU: upstream `MiniCPMMLP` (`minicpm.py:219-226`) selects `FatreluAndMul` on `hidden_act == "fatrelu"` and raises otherwise; our `parse_config` never reads `hidden_act` | bug |

## Resolution

-
