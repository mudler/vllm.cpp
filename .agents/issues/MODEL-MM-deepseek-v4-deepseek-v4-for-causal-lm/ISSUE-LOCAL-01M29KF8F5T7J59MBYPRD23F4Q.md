ID: ISSUE-LOCAL-01M29KF8F5T7J59MBYPRD23F4Q
Title: 8 of 20 test_deepseek_v4_mm_reach cases fail on aarch64 because of the host-side i8mm quant repack
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

On an aarch64 i8mm host every one of the eight failures throws 'deepseek-v4 keep-quant expert/group slice requires non-repacked blocks (disable VT_CPU_QUANT_REPACK for the stacked-expert weights)' at deepseek_v4.cpp:583. The cause is PROVEN by an A/B on the SAME binary and the same box, not inferred from the message: with the repack on the suite reads '20 | 12 passed | 8 failed'; with VT_CPU_QUANT_REPACK=0 it reads '20 | 20 passed | 0 failed'. vt::cpu::QuantRepackActive() is true only on an aarch64 i8mm host, which is why these cases are green on the x86-64 devbox and red on thor. It is a HOST quant-repack defect and not a device failure. Consequence: this row's gate cannot run clean on an aarch64 host without that flag, so an aarch64 gate result is not comparable to an x86-64 one until the keep-quant expert/group slice accepts repacked blocks or refuses them earlier. Measured by W7-CUDA on thor:gpu0.

## Resolution

-
