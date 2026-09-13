ID: ISSUE-LOCAL-01M2CKNHVKJXT29XN8WM11KF6W
Title: OWED: three latent second-reader hazards around the staged-borrow release, including the header ResidentWeight's missing expert_streamed refusal
Row: MODEL-MM-QWEN4-EXP
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

A fresh review of the staged-borrow release found three hazards that are latent today and that this row deliberately did NOT re-engineer, because each repair is a change to a shared seam with a blast radius nothing measures (see ISSUE-LOCAL-01M2CKN5516AKE7W2JVDV86Z8X). ONE. dense_attn::ResidentWeightF32 (dense_attn_block.h:389) memoizes on d_dev_f32 INDEPENDENTLY of d_dev and reads w.bytes to upcast, so a weight that ResidentWeight has staged and released and that is then upcast would read dropped pages. It is unreachable only because every ResidentWeightF32 target today is a LoadNormBf16/ExpandBf16 owned tensor whose mmap_fd is -1, which is an accident of the current loaders and not an invariant either function states. TWO. The header ResidentWeight carries no VT_CHECK(!w.expert_streamed); its translation-unit-local twin at qwen3_5.cpp:1181 does, so nothing stops a streamed tower being staged and released on the header path. This was NOT closed in the same flow on purpose: adding the refusal would make the header seam REFUSE a weight it accepts today, and GLM5-Next stages three expert banks through it at glm5_next_moe.cpp:243-245 with no test on any staging device, so a one-line VT_CHECK could remove a working path rather than close a hazard. It needs the cross-family harness first. THREE. glm5_next_moe.cpp:292 and :300 are a host-fallback arm that reads the same OwnedTensor bytes, reachable in a partial-stage window because :243-245 are three sequential, un-transactional uploads. It re-faults, so it is correct, but the cost is a full re-read of an expert bank off the file. The window is narrower than it first reads: that arm VT_CHECKs its queue is CPU (glm5_next_moe.cpp:271) while the release only fires on a non-CPU device, so reaching it needs a CPU queue paired with a separate device pointer whose banks have already staged. Recorded with that qualification rather than dropped, because the re-read is real whenever the pairing occurs.

## Resolution

-
