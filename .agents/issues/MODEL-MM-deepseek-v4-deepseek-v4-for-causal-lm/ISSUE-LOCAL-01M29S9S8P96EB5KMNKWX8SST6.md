ID: ISSUE-LOCAL-01M29S9S8P96EB5KMNKWX8SST6
Title: DeepSeek-V4 vision W6: the comparator enforces no bound, and the spec's condition 3 is near-identity but reads as independent evidence
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

Two halves of one defect. (1) tools/parity/dsv4v_w6_compare.py contains no bound and emits no pass/fail verdict: its only non-zero exit is return 2 for SHAPE_MISMATCH and the terminal path returns 0 regardless of magnitude, so a future run whose image_rows.mean_rel_l2 drifts from 3.83% to 12% produces RC=0, a well-formed report, W6_PARITY_DONE and no signal at all. (2) .agents/specs/deepseek-v4-flash-vision.md sets W6 condition 3 as cells mean relative L2 <= 3.34% + 1.57% = 4.9%, where BOTH addends were measured in the same session as the 3.83% being judged. Since condition 2 independently establishes ours_f32 <-> oracle = 1.34% <= 1.57%, the triangle inequality already forces ours_bf16 <-> oracle <= 4.68% < 4.9%. Condition 3 is therefore close to an identity given the other two, not an independent test, and the spec must not let a reader quote 'the shipped bf16 path passed a 4.9% bound' as independent evidence. The conditions doing real work are 1 (sentinels exact, identity permutation) and 2 (f32 arm inside the oracle's floor). Found by fresh review 2026-09-12.

## Resolution

2026-09-12: fixed in this change, both halves. (1) tools/parity/dsv4v_w6_compare.py now reads the new committed tools/parity/dsv4v_w6_bounds.json -- the measurements with the rc job that produced each, kept OUT of the comparator so no wave can derive a bound from the run it is judging -- maps the tag to a recorded profile, prints 'VERDICT <v> tag=<t> profile=<p>' plus a BOUND line per breach, and exits 0 PASS/DIAGNOSTIC, 1 bound exceeded, 2 SHAPE_MISMATCH, 3 no rule matched. An unmatched tag is UNJUDGED and exits 3 rather than passing, which is the whole point. MEASURED on synthetic blocks: shipped-bf16 leg at 2% drift PASS exit 0; at 12% drift 'image_rows mean_rel_l2 12.0000% EXCEEDS the recorded bound 4.9000%' FAIL exit 1, which is exactly the drift this issue names; f32 leg at 2% FAIL exit 1 against the tighter 1.57% floor; floor leg at 12% DIAGNOSTIC exit 0; unknown tag UNJUDGED exit 3. (2) The spec's 'THE BOUND' section now states in full that condition 3 is close to an identity given conditions 1 and 2 -- the triangle inequality over the 3.34% self-dtype distance and condition 2's measured 1.34% already forces <= 4.68% < 4.9% -- that it must never be quoted as independent evidence, and that the conditions doing real work are 1 (sentinels exact, identity permutation) and 2 (f32 arm inside the oracle's own floor). The bound is KEPT, as the regression catcher and the recorded gate, and is now enforced by the harness instead of by a reader's arithmetic. The two downstream sites that quoted 4.9% -- the W7-CUDA verdict paragraph and the '## Now' summary -- carry the same qualification, and the bounds file repeats it beside the number.
