ID: ISSUE-LOCAL-01M2A5T6N98WABAAXZH4D52CRR
Title: Three malformed bounds inputs crash dsv4v_w6_compare.py instead of reaching the ERROR exit
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

dsv4v_w6_compare.py classifies a malformed recorded profile as ERROR with exit 4, so a reader can tell 'the record is broken' from 'the data failed'. Three malformed inputs never reach that classification and die on an uncaught exception with exit 1, which reads as FAIL. (1) 'stages': null with a reason declared makes validate_profile call .get on None -> AttributeError. (2) a tag_rules entry with three elements, or a bare string instead of a pair, makes profile_for unpack it -> ValueError. (3) an empty object {} as the whole bounds file makes load_bounds succeed and judge() index a missing key -> KeyError. Each is a broken RECORD, which the file's own design says must be exit 4 ERROR, and each currently exits 1 alongside genuine bound failures. Found by fresh review 2026-09-12; filed rather than fixed because the repair wave was scoped to the fail-open and the three false-red routes.

## Resolution

-
