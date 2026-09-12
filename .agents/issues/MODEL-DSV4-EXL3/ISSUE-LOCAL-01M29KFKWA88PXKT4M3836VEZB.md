ID: ISSUE-LOCAL-01M29KFKWA88PXKT4M3836VEZB
Title: test_deepseek_v4_exl3_forward_loop_arm is registered with a bare add_test, so a whole-tree ninja build refuses and every ctest result reads Not Run
Row: MODEL-DSV4-EXL3
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: -

## Problem

tests/CMakeLists.txt:880 registers test_deepseek_v4_exl3_forward_loop_arm with a bare add_test(NAME ... COMMAND test_deepseek_v4_exl3_forward) instead of the project's vllm_cpp_add_test(...) wrapper that every sibling uses. VERIFIED HERE on a CPU configure of this tree: 'ninja -t targets all' carries a target for 26 of the 27 tests ctest registers under -R 'deepseek_v4|clip_mmproj_gguf', and test_deepseek_v4_exl3_forward_loop_arm is the one with NO target behind its ctest entry. BLAST RADIUS, as reported by the W7-CUDA fresh reviewer and not re-measured here: their whole-tree build refused over this entry, so the gate run produced ZERO test binaries, and ctest then reported 27 tests as 'Not Run'. 'Not Run' means the executable is absent, not that the test failed, so any N-of-N figure read off such a run is meaningless - which is what makes this a records hazard beyond the one entry. The intent of the entry is sound: it runs the same suite under VT_DSV4_EXL3_FUSED_MOE=0, the only way to gate the rollback arm of a flag that is read once per process, so the fix is to give it a target rather than to delete it. Found while repairing MODEL-MM-deepseek-v4 W7-CUDA review findings; not fixed there because it belongs to this row.

## Resolution

-
