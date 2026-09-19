ID: ISSUE-LOCAL-01M2A5T6ZPD6QXSV3Q3MSAXVJB
Title: A stray 'NN - <text>' line inside a ctest FAILED block is read as a failing test name
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

classify_ctest_cuda in tools/parity/dsv4v_w7_cuda.sh extracts the failing suites with awk '/The following tests FAILED:/{f=1; next} f && /^[[:space:]]*[0-9]+ - /{print $3}'. The f flag is never cleared, so ANY later line in the log matching '<spaces><digits> - ' is read as another failing test name. A test whose own output contains such a line -- a table, a diff, a progress counter -- would be treated as an unattributed suite failure and fail the job. Low probability on this row's suites, and it is a FALSE RED rather than a fail-open: the failure mode is refusing a run the spec attributes, never accepting one it does not. Found by fresh review 2026-09-12; filed rather than fixed to keep the repair wave scoped.

## Resolution

-
