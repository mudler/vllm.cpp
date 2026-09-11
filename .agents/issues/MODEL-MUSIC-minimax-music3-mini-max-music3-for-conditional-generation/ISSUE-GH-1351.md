ID: ISSUE-GH-1351
Title: `scripts/check-fusion-consistency.py` Check 2 is satisfied by a **COMMENT**: it runs `_MERGED_GEMM_SEAM` over `path.read_text()` with no comment stripping, so a model TU that merely *mentions* `MlpGateUpMethod` in prose is exempt from the check forever. Measured on `minimax_music3_depth_device.cpp` ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), spec §19.7): replacing the `layers::UnquantizedMlpGateUpMethod` call with an inline hand-rolled `vt::MatmulBT` + `vt::SiluAndMul` path while leaving the two comment mentions gives `uses_merged_gemm_seam = True` and the checker returns **rc=0 green**, with the numbers bit-identical so no other gate sees it; stripping comments flips it to `False`. The only comment-aware helper in the file, `allowlisted_names`, strips `#` comments from the allowlist FILE rather than from the scanned source. NOT a defect: `MlpGateUp[A-Za-z]*Method` matching a renamed `MlpGateUpXX` is deliberate, to cover `UnquantizedMlpGateUpGeluMethod`. Fixing it is a semantic checker change — spec, red-before test, green-after evidence — and must not be done by widening the regex; stripping comments also moves the site count, so a fixture must pin both numbers
Row: MODEL-MUSIC-minimax-music3-mini-max-music3-for-conditional-generation
State: UNKNOWN
Kind: bug
GitHub: 1351
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:454`

### Frozen archive evidence

> | [#1351](https://github.com/mudler/vllm.cpp/issues/1351) | `MUSIC3-DEPTH-DEVICE` | `scripts/check-fusion-consistency.py` Check 2 is satisfied by a **COMMENT**: it runs `_MERGED_GEMM_SEAM` over `path.read_text()` with no comment stripping, so a model TU that merely *mentions* `MlpGateUpMethod` in prose is exempt from the check forever. Measured on `minimax_music3_depth_device.cpp` ([#1309](https://github.com/mudler/vllm.cpp/issues/1309), spec §19.7): replacing the `layers::UnquantizedMlpGateUpMethod` call with an inline hand-rolled `vt::MatmulBT` + `vt::SiluAndMul` path while leaving the two comment mentions gives `uses_merged_gemm_seam = True` and the checker returns **rc=0 green**, with the numbers bit-identical so no other gate sees it; stripping comments flips it to `False`. The only comment-aware helper in the file, `allowlisted_names`, strips `#` comments from the allowlist FILE rather than from the scanned source. NOT a defect: `MlpGateUp[A-Za-z]*Method` matching a renamed `MlpGateUpXX` is deliberate, to cover `UnquantizedMlpGateUpGeluMethod`. Fixing it is a semantic checker change — spec, red-before test, green-after evidence — and must not be done by widening the regex; stripping comments also moves the site count, so a fixture must pin both numbers | bug |

## Resolution

-
