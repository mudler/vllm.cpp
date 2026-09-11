ID: ISSUE-GH-2099
Title: **`scripts/check-oracle-pins.py` never parses an `oracle-pin-lane` block, so both lane pins in `.agents/oracles/transformers.md` are unchecked prose.** Its `BLOCK` regex is `^```oracle-pin\n`, and the newline means an `oracle-pin-lane` fence never matches; a repository-wide search for `oracle-pin-lane` returns the block itself and one prose reference, so nothing reads it. MEASURED on `row/MODEL-MM-GLM53-FLASH-W0`: corrupting the `glm5_next` lane `pin`, `gateable` or `pinned_on`, and deleting the lane block outright, each leave the checker at exit 0, while the same corruption of the registry `oracle-pin` block reds it. Every rule the registry gate holds is therefore unenforced on a lane pin, and the checker's `--self-test` corpus and `tests/scripts/test_check_oracle_pins.py` name no lane case. Found by W0 (#2096) while verifying its own gate; recorded as O13 rather than repaired, because W0's scope excludes every checker and the fix is a semantic checker change that owes a spec, a red-before mutation, and a decision about which keys a lane record requires
Row: MODEL-MM-glm5-next-glm5-next-for-conditional-generation
State: UNKNOWN
Kind: gap
GitHub: 2099
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:810`

### Frozen archive evidence

> | [#2099](https://github.com/mudler/vllm.cpp/issues/2099) | `MODEL-MM-glm5-next-glm5-next-for-conditional-generation` | **`scripts/check-oracle-pins.py` never parses an `oracle-pin-lane` block, so both lane pins in `.agents/oracles/transformers.md` are unchecked prose.** Its `BLOCK` regex is `^```oracle-pin\n`, and the newline means an `oracle-pin-lane` fence never matches; a repository-wide search for `oracle-pin-lane` returns the block itself and one prose reference, so nothing reads it. MEASURED on `row/MODEL-MM-GLM53-FLASH-W0`: corrupting the `glm5_next` lane `pin`, `gateable` or `pinned_on`, and deleting the lane block outright, each leave the checker at exit 0, while the same corruption of the registry `oracle-pin` block reds it. Every rule the registry gate holds is therefore unenforced on a lane pin, and the checker's `--self-test` corpus and `tests/scripts/test_check_oracle_pins.py` name no lane case. Found by W0 (#2096) while verifying its own gate; recorded as O13 rather than repaired, because W0's scope excludes every checker and the fix is a semantic checker change that owes a spec, a red-before mutation, and a decision about which keys a lane record requires | gap |

## Resolution

-
