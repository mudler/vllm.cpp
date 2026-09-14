ID: ISSUE-LOCAL-01M2BXFMFDNZQD41HY629KAGCC
Title: check-deepseek-v4-vision-manifests.py has no mutation suite at the path its own evidence gate names
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

scripts/check-deepseek-v4-vision-manifests.py is a new 571-line governance checker, but tests/scripts/test_check_deepseek_v4_vision_manifests.py does not exist, so the pr-size evidence gate fails the whole pull request with 'requires semantic mutation evidence'. Its 139 manifest cases live in tests/vllm/models/test_deepseek_v4_mm_loader.cpp, which that gate cannot read, and the checker is also absent at the range base with no CREATION_MUTATIONS entry, so the red-before half has nothing to mutate.

## Resolution

2026-09-12: tests/scripts/test_check_deepseek_v4_vision_manifests.py lands with 28 cases, at the exact path scripts/check-pr-size.py derives for this checker. Every case breaks one guarantee in a scratch copy of the tree and requires a named diagnostic and a counted number of disagreements: the vision count 267 -> 266 reds twice (the classification and the header cross-check), an absent fixture reds with 'missing fixture', a dropped image_end entry reds twice, one changed vision shape reds four times (shape, payload total, both digests), a widened dtype reds four times, and config.json vision_n_layers 32 -> 31 reds ten times including config_sha256, so the config and the manifests cannot drift apart. The counts were measured against the checker, not predicted. One case replaces urllib.request.urlopen with a function that fails the test and proves the default mode is offline. Measured: 'Ran 28 tests' OK at HEAD; under the DISABLED_CREATION_CHECKER stub 'Ran 28 tests' then 'FAILED (failures=19, errors=19)' with ZERO survivors. scripts/check-pr-size.py gains the CREATION_MUTATIONS entry the red-before half needs, because the checker is absent at the range base; tests/scripts/test_check_pr_size.py pins it and stays green at 61 cases. Registered in both lanes: the SUITES array of scripts/agent-preflight.sh and the DeepSeek-V4 Vision step of .github/workflows/ci.yml.
