ID: ISSUE-GH-1629
Title: **`test_check_attention_rung_consistency.py` stored a count of the model tree, so every row on the attention-rung allowlist redded it by doing the thing the allowlist exists for.** `ShippedTreeTests::test_the_population_is_not_empty` asserted `>= 9` against a tree holding exactly 9 `vt::Attention(` sites, so a removing row had zero headroom and no green path: leaving the parked stem redded the floor (`8 not greater than or equal to 9`), and deleting it redded the floor and `test_allowlist_holds_only_the_in_flight_stems` as well -- while the allowlist header explicitly recommends the first of those two. That is the `## Records` shape AGENTS.md names, a measurement of one file stored inside another, and it blocked PR #1579 (#1545) and the LTX-2.5 routing row, which removes two of the three parked stems. FIXED HERE, and NOT by lowering the number, which is the known mute-switch: the floor became `>= 1`, because an empty population means the scanner broke and that is the only thing a raw total can honestly detect, and the guard that a rename cannot slip past stays `test_the_six_deliberate_sites_carry_a_marker`, which pins six sites BY NAME. A case that the stem in a red message names a real source file was added beside it, so a typo in the allowlist is still caught without pinning a count. A second drift lock in the same suite, `assertGreater(excused, 0)`, required the shipped allowlist to stay non-empty forever; it is replaced by two synthetic cases that build their own allowlisted population, so the excused counter is pinned without the shipped tree having to keep a stem parked. Found while landing #1578 and #1579 together -- each green in isolation, main red once both land -- and fixed in the same flow
Row: KERNEL-ATTN-DENSE-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1629
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:583`

### Frozen archive evidence

> | [#1629](https://github.com/mudler/vllm.cpp/issues/1629) | `KERNEL-ATTN-DENSE-FLASH` | **`test_check_attention_rung_consistency.py` stored a count of the model tree, so every row on the attention-rung allowlist redded it by doing the thing the allowlist exists for.** `ShippedTreeTests::test_the_population_is_not_empty` asserted `>= 9` against a tree holding exactly 9 `vt::Attention(` sites, so a removing row had zero headroom and no green path: leaving the parked stem redded the floor (`8 not greater than or equal to 9`), and deleting it redded the floor and `test_allowlist_holds_only_the_in_flight_stems` as well -- while the allowlist header explicitly recommends the first of those two. That is the `## Records` shape AGENTS.md names, a measurement of one file stored inside another, and it blocked PR #1579 (#1545) and the LTX-2.5 routing row, which removes two of the three parked stems. FIXED HERE, and NOT by lowering the number, which is the known mute-switch: the floor became `>= 1`, because an empty population means the scanner broke and that is the only thing a raw total can honestly detect, and the guard that a rename cannot slip past stays `test_the_six_deliberate_sites_carry_a_marker`, which pins six sites BY NAME. A case that the stem in a red message names a real source file was added beside it, so a typo in the allowlist is still caught without pinning a count. A second drift lock in the same suite, `assertGreater(excused, 0)`, required the shipped allowlist to stay non-empty forever; it is replaced by two synthetic cases that build their own allowlisted population, so the excused counter is pinned without the shipped tree having to keep a stem parked. Found while landing #1578 and #1579 together -- each green in isolation, main red once both land -- and fixed in the same flow | bug |

## Resolution

-
